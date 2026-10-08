"""
Main simulation runner for the Multi-Cohort SNN Sleep Disorder Simulator.

Execution flow
--------------
 1.  Fetch & process PhysioNet baseline data (MNE/fallback)
 2.  Initialise 5 cohort CohortSimulation instances (Normal, CRSWD, Psychophysiological, Paradoxical, Narcolepsy)
 3.  Step through 72h x 4320 timesteps in parallel (sequential loop)
 4.  Stream metrics to TensorBoard every step
 5.  Update live dashboard every DASHBOARD_UPDATE_INTERVAL steps
 6.  Log raster + hypnogram images every 24 simulated hours
 7.  Save summary_72h_cohorts.png multi-panel comparison figure
 8.  Print final clinical summary table + TensorBoard instructions

Usage
-----
    python main.py            # full 4320-step 72h simulation
    python main.py --dry-run  # first 20 steps only (import + shape check)

Dependencies
------------
    pip install torch scipy numpy matplotlib tensorboard mne tqdm
    pip install norse       # optional -- falls back to pure-PyTorch LIF
"""


# region Imports

from __future__ import annotations

import os
import sys
import json
import time
import argparse
import warnings
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List, Optional

matplotlib.use("Agg")  # default Agg; dashboard module switches if display available
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Ensure UTF-8 output and project root is on the path regardless of CWD
if sys.stdout is not None and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if sys.stderr is not None and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import config
from data.physionet_fetcher           import fetch_sleep_edf
from environment.circadian_env        import CircadianEnvironment
from models.process_sc                import ProcessSC
from models.flip_flop                 import FlipFlopSwitch
from models.corticothalamic_snn       import CorticothalamicSNN
from cohorts.patient_profiles         import get_cohort_params, get_all_cohorts
from evaluation.spectral_analysis     import (
    compute_band_powers, synthesize_lfp_from_spikes, get_subjective_wakefulness
)
from evaluation.metrics               import (
    compute_sleep_metrics, compute_mae, compute_rolling_mae, build_clinical_summary_table
)
from visualization.tensorboard_logger import SleepTensorBoardLogger
from visualization.live_dashboard     import LiveDashboard

try:
    from tqdm import tqdm
    _TQDM = True
except ImportError:
    _TQDM = False

# endregion Imports


# region Cohort Simulation

class CohortSimulation:
    """
    Bundles all components for one cohort into a single object.

    Each cohort owns:
     * CircadianEnvironment -- light-dark cycle with phase/sensitivity offsets
     * ProcessSC            -- homeostatic (S) + circadian (C) drives
     * FlipFlopSwitch       -- VLPO-LHA bistable sleep/wake switch
     * CorticothalamicSNN   -- thalamocortical spiking network
    """

    def __init__(self, cohort_id: int):
        self.cohort_id = cohort_id
        p = get_cohort_params(cohort_id)
        self.params = p
        self.name   = p.name

        # Sub-models
        self.env = CircadianEnvironment(
            light_on_hour     = p.light_on_hour,
            light_off_hour    = p.light_off_hour,
            phase_offset_h    = p.circadian_phase_offset_h,
            light_sensitivity = p.light_sensitivity,
        )
        self.process_sc = ProcessSC(
            tau_rise_h               = p.tau_rise_h,
            tau_decay_h              = p.tau_decay_h,
            circadian_phase_offset_h = p.circadian_phase_offset_h,
            light_sensitivity        = p.light_sensitivity,
            initial_s                = p.initial_s,
        )
        self.flip_flop = FlipFlopSwitch(
            vlpo_gain        = p.vlpo_gain,
            arousal_bias     = p.arousal_bias,
            orexin_tone      = getattr(p, "orexin_tone", 1.0),
            soremp_tendency  = getattr(p, "soremp_tendency", 0.0),
            wake_instability = getattr(p, "wake_instability", 0.0),
        )
        self.snn = CorticothalamicSNN(
            cortical_inh_scale = p.cortical_inhibition_scale,
        )

        # History buffers
        self.hypnogram:    List[int]  = []
        self.band_history: List[Dict] = []
        self.last_spikes:  Optional[np.ndarray] = None
        self.last_lfp:     Optional[np.ndarray] = None

        # Internal state
        self._is_awake = True
        print(f"  Cohort {cohort_id} [{self.name}] - SNN backend: {self.snn.backend}")

    def step(self, step_idx: int, dt_min: float = config.DT_MIN) -> dict:
        """
        Advance all sub-models by one simulation timestep.

        Parameters
        ----------
        step_idx : current simulation step (0-indexed)
        dt_min   : timestep length in minutes

        Returns
        -------
        dict of all telemetry metrics for this cohort at this step
        """
        t_min = step_idx * dt_min
        t_h   = t_min / 60.0

        # 1. Environment
        env_state = self.env.step(t_min)
        lux       = env_state["lux"]

        # 2. Process S & C
        sc_out = self.process_sc.step(self._is_awake, lux, dt_min=dt_min)

        # 3. Flip-Flop Switch
        ff_out = self.flip_flop.step(
            sc_out["sleep_propensity"],
            lux,
            dt=dt_min,
            sleep_maintenance=sc_out.get("sleep_maintenance"),
        )
        self._is_awake  = ff_out["is_awake"]
        sleep_stage     = ff_out["sleep_stage"]

        # 4. SNN (every step; using cached output if SNN skipped for speed)
        spikes, lfp_raw = self.snn(
            sleep_stage = sleep_stage,
            t_steps     = config.SNN_INTERNAL_STEPS,
        )
        self.last_spikes = spikes
        self.last_lfp    = lfp_raw

        # 5. Spectral analysis on LFP
        # Synthesize realistic LFP from spikes + stage modulation
        lfp_synth = synthesize_lfp_from_spikes(
            spikes,
            sleep_stage        = sleep_stage,
            fs                 = config.WELCH_FS,
            cortical_inh_scale = self.params.cortical_inhibition_scale,
        )
        bands = compute_band_powers(lfp_synth, fs=config.WELCH_FS)
        self.band_history.append(bands)

        # 6. Subjective wakefulness
        subj_wake = get_subjective_wakefulness(
            sleep_stage       = sleep_stage,
            cortical_beta     = bands.get("beta", 0.0),
            perception_offset = self.params.perception_offset,
        )

        # 7. Record hypnogram
        self.hypnogram.append(sleep_stage)

        return {
            # Environment
            "t_h":              t_h,
            "env_lux":          lux,
            "env_hour":         t_h % 24.0,
            # Process S & C
            "process_s":        sc_out["S"],
            "process_c":        sc_out["C"],
            "sleep_propensity": sc_out["sleep_propensity"],
            # Flip-Flop
            "sleep_stage":      sleep_stage,
            "vlpo_activity":    ff_out["vlpo"],
            "lha_activity":     ff_out["lha"],
            "is_awake":         self._is_awake,
            # Spectral
            "delta_power":      bands.get("delta", 0.0),
            "theta_power":      bands.get("theta", 0.0),
            "alpha_power":      bands.get("alpha", 0.0),
            "sigma_power":      bands.get("sigma", 0.0),
            "beta_power":       bands.get("beta",  0.0),
            # Perception
            "subj_wake":        subj_wake,
            # LFP sample for dashboard trace
            "lfp_sample":       float(np.mean(np.abs(lfp_raw[-10:]))) if len(lfp_raw) > 0 else 0.0,
        }

# endregion Cohort Simulation


# region Summary Figure

def _style_panel(ax: plt.Axes, title: str) -> None:
    ax.set_facecolor("#0d0d1a")
    ax.set_title(title, color="white", fontsize=10, pad=5, loc="left")
    ax.tick_params(colors="#777777", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2a4a")
    ax.grid(True, color="#1a1a3a", linewidth=0.4, alpha=0.5)


def save_summary_figure(
    all_cohorts:     List[CohortSimulation],
    reference_bands: Dict[str, float],
    all_metrics:     Dict[int, Dict],
    t_axis:          np.ndarray,
    all_step_data:   Dict[int, Dict[str, List]],
) -> None:
    """Build and save the 4-panel 72h summary comparison figure."""
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    fig = plt.figure(figsize=(20, 18), facecolor="#0d0d1a")
    fig.suptitle(
        "DyssomniaNN - 72-Hour Multi-Cohort SNN Sleep Disorder Simulation",
        color="white", fontsize=15, fontweight="bold", y=0.99,
    )

    gs = gridspec.GridSpec(
        4, 1, figure=fig,
        hspace=0.55,
        left=0.07, right=0.97, top=0.95, bottom=0.05,
    )

    # ---- Panel 1: Daylight + Process S & C ----
    ax1 = fig.add_subplot(gs[0])
    _style_panel(ax1, "Panel 1: Photoperiod, Process S, and Process C")

    # Daylight background
    t_h = t_axis / 60.0
    for i in range(int(config.SIM_DURATION_HOURS // 24) + 1):
        ax1.axvspan(i * 24 + 6, i * 24 + 22, color="#ffeb9c", alpha=0.07, zorder=0)

    for cid, cohort in enumerate(all_cohorts):
        color = config.COHORT_COLORS[cid]
        s_arr = np.array(all_step_data[cid]["process_s"])
        c_arr = np.array(all_step_data[cid]["process_c"])
        tt    = t_h[:len(s_arr)]
        ax1.plot(tt, s_arr, color=color, linewidth=1.2, label=f"{cohort.name} (S)")
        ax1.plot(tt, c_arr, color=color, linewidth=0.7, linestyle="--", alpha=0.6)

    ax1.set_ylabel("Process S and C Drive", color="#aaaaaa", fontsize=9)
    ax1.set_xlabel("Simulated Time (h)", color="#aaaaaa", fontsize=9)
    ax1.legend(loc="upper right", fontsize=7, facecolor="#1a1a2e",
               edgecolor="#334466", labelcolor="white", ncol=2)
    ax1.text(0.01, 0.95, "Solid = S(t)  |  Dashed = C(t)",
             transform=ax1.transAxes, color="#888888", fontsize=7, va="top")

    # ---- Panel 2: Concurrent Hypnograms ----
    ax2 = fig.add_subplot(gs[1])
    _style_panel(ax2, "Panel 2: Concurrent Hypnograms across 5 Cohorts")

    for cid, cohort in enumerate(all_cohorts):
        color = config.COHORT_COLORS[cid]
        hyp   = np.array(cohort.hypnogram, dtype=float)
        tt    = t_h[:len(hyp)]
        offset = cid * 5.5
        ax2.step(tt, hyp + offset, color=color, linewidth=0.9, where="post")
        ax2.fill_between(tt, offset, hyp + offset, step="post", alpha=0.2, color=color)
        ax2.text(-0.5, offset + 2.0, cohort.name, color=color, fontsize=7, va="center",
                 ha="right", transform=ax2.get_yaxis_transform(), clip_on=False)

    ax2.set_yticks([0, 1, 2, 3, 4])
    ax2.set_yticklabels(["W", "N1", "N2", "N3", "REM"], color="#aaaaaa", fontsize=7)
    ax2.set_xlabel("Simulated Time (h)", color="#aaaaaa", fontsize=9)

    # ---- Panel 3: Delta vs Beta power ----
    ax3 = fig.add_subplot(gs[2])
    _style_panel(ax3, "Panel 3: Spectral Power (Delta Slow-Wave vs Beta Hyperarousal)")

    for cid, cohort in enumerate(all_cohorts):
        color  = config.COHORT_COLORS[cid]
        delta  = np.array(all_step_data[cid]["delta_power"])
        beta   = np.array(all_step_data[cid]["beta_power"])
        tt     = t_h[:len(delta)]
        ax3.plot(tt, delta, color=color, linewidth=0.9,
                 label=f"{cohort.name} Δ")
        ax3.plot(tt, beta,  color=color, linewidth=0.7, linestyle=":",
                 alpha=0.7)

    ax3.set_ylabel("Relative Band Power", color="#aaaaaa", fontsize=9)
    ax3.set_xlabel("Simulated Time (h)",  color="#aaaaaa", fontsize=9)
    ax3.legend(loc="upper right", fontsize=7, facecolor="#1a1a2e",
               edgecolor="#334466", labelcolor="white", ncol=2)
    ax3.text(0.01, 0.95, "Solid = Delta  |  Dotted = Beta",
             transform=ax3.transAxes, color="#888888", fontsize=7, va="top")

    # ---- Panel 4: Clinical Summary Table ----
    ax4 = fig.add_subplot(gs[3])
    ax4.axis("off")
    ax4.set_facecolor("#0d0d1a")

    table_data = []
    col_labels = [
        "Cohort", "Sleep% (Env)", "TST(h)", "SOL(m)", "WASO(m)",
        "Cycles", "SOREMP", "Rested%", "Rest Status", "MAE PhysioNet"
    ]
    for cid, m in all_metrics.items():
        name = config.COHORT_NAMES.get(cid, f"Cohort {cid}")
        cohort = all_cohorts[cid]
        mae = compute_rolling_mae(cohort.band_history, reference_bands, window=len(cohort.band_history))
        row = [
            name,
            f"{m.get('sleep_pct_env', 0.0):.1f}%",
            f"{m.get('tst_h', 0.0):.1f}h",
            f"{m['sol_min']:.0f}",
            f"{m['waso_min']:.0f}",
            str(m['n_cycles']),
            str(m.get('n_soremps', 0)),
            f"{m.get('well_rested_index', 0.0):.1f}%",
            m.get('rest_status', 'N/A'),
            f"{mae:.4f}",
        ]
        table_data.append(row)

    tbl = ax4.table(
        cellText  = table_data,
        colLabels = col_labels,
        loc       = "center",
        cellLoc   = "center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.0, 1.8)

    # Style table cells
    for (row, col), cell in tbl.get_celld().items():
        cell.set_facecolor("#1a1a2e" if row % 2 == 0 else "#0d1020")
        cell.set_edgecolor("#334466")
        cell.set_text_props(color="white" if row > 0 else "#4FC3F7")
        if row == 0:
            cell.set_facecolor("#0d0d2e")

    # Color cohort name cells
    for row_idx in range(1, len(all_cohorts) + 1):
        cid = row_idx - 1
        cell = tbl[row_idx, 0]
        cell.set_text_props(color=config.COHORT_COLORS.get(cid, "white"))

    ax4.set_title("Panel 4 - Clinical Summary Table", color="white", fontsize=10,
                  pad=4, loc="left")

    plt.savefig(
        config.SUMMARY_FIGURE_PATH,
        dpi=150, bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    plt.close(fig)
    print(f"\n[Output] Summary figure saved -> {os.path.abspath(config.SUMMARY_FIGURE_PATH)}")

# endregion Summary Figure


# region Main

class TeeLogger:
    """Mirrors stdout/stderr to both console and a log file in real-time."""
    def __init__(self, filepath: str, *streams):
        self.file = open(filepath, "w", encoding="utf-8", buffering=1)
        self.streams = streams

    def write(self, data: str):
        for s in self.streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass
        try:
            self.file.write(data)
            self.file.flush()
        except Exception:
            pass

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass
        try:
            self.file.flush()
        except Exception:
            pass

    def close(self):
        try:
            self.file.close()
        except Exception:
            pass


def main(
    dry_run:  bool = False,
    demo_run: bool = False,
    days:     Optional[float] = None,
    gui:      bool = False,
    log_file: Optional[str] = None,
) -> None:
    """Run parallel multi-cohort simulation."""
    tee = None
    if log_file:
        log_dir = os.path.dirname(os.path.abspath(log_file))
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        tee = TeeLogger(log_file, sys.stdout)
        sys.stdout = tee

    t_start = time.time()
    print("=" * 70)
    print("  DyssomniaNN - Multi-Cohort SNN Sleep Disorder Simulator")
    print("  Parallel Cohort Simulation & Clinical Telemetry")
    print("=" * 70)

    # Step 1: PhysioNet reference data
    print("\n[1/5] Fetching PhysioNet Sleep-EDF baseline data ...")
    reference_bands = fetch_sleep_edf()
    print("      Reference bands:", {k: f"{v:.4f}" for k, v in reference_bands.items()})

    # Step 2: Initialise cohorts
    print(f"\n[2/5] Initialising {config.N_COHORTS} cohort simulation instances ...")
    cohorts: List[CohortSimulation] = []
    for cid in range(config.N_COHORTS):
        cohorts.append(CohortSimulation(cid))

    # Step 3: TensorBoard logger + live dashboard
    print("\n[3/5] Starting TensorBoard logger ...")
    os.makedirs(config.TENSORBOARD_LOG_DIR, exist_ok=True)
    tb_logger = SleepTensorBoardLogger(log_dir=config.TENSORBOARD_LOG_DIR)
    dashboard = LiveDashboard(enable_gui=(gui and not (dry_run or demo_run)))

    # Step 4: Simulation loop configuration
    if dry_run:
        n_steps = 20
        step_dt = config.DT_MIN
        sim_duration_h = 20.0 / 60.0
        mode_label = "DRY-RUN (20 steps, 20 min smoke test)"
    elif demo_run:
        d = float(days) if days is not None else 1.0
        sim_duration_h = d * 24.0
        step_dt = 10.0
        n_steps = int(sim_duration_h * 60.0 / step_dt)
        mode_label = f"DEMO-RUN ({sim_duration_h:.0f}h / {d:.0f}-day accelerated cycle, {n_steps} steps, dt=10 min)"
    else:
        d = float(days) if days is not None else float(config.SIM_DURATION_HOURS / 24.0)
        sim_duration_h = d * 24.0
        step_dt = config.DT_MIN
        n_steps = int(sim_duration_h * 60.0 / step_dt)
        mode_label = f"Full multi-day simulation ({sim_duration_h:.0f}h / {d:.0f} days, {n_steps} steps, dt=1 min)"

    print(f"\n[4/5] Running {mode_label} ({n_steps} steps, {config.N_COHORTS} cohorts) ...")
    if dry_run:
        print("      [Notice] DRY-RUN simulates only 20 min (pipeline test). Because sleep")
        print("      occurs at night, 20-min curves are baseline flat. For dynamic 24h")
        print("      sleep cycles, run: python main.py --demo-run")
    print(f"      SNN backend: {cohorts[0].snn.backend}")

    # Per-cohort step history for final figure
    all_step_data: Dict[int, Dict[str, List]] = {
        cid: {"process_s": [], "process_c": [], "delta_power": [], "beta_power": []}
        for cid in range(config.N_COHORTS)
    }

    t_axis = np.arange(n_steps, dtype=float) * step_dt

    iterator = range(n_steps)
    if _TQDM:
        iterator = tqdm(iterator, desc="Simulating", unit="step",
                        ncols=80, colour="cyan")

    for step in iterator:
        t_h = (step * step_dt) / 60.0

        # --- Step all cohorts ---
        step_results: Dict[int, dict] = {}
        for cid, cohort in enumerate(cohorts):
            metrics = cohort.step(step, dt_min=step_dt)
            step_results[cid] = metrics

            # Compute MAE
            mae = compute_mae(
                predicted_bands = {
                    b: metrics.get(f"{b}_power", 0.0) for b in config.BAND_FREQS
                },
                reference_bands = reference_bands,
            )
            metrics["mae_physionet"] = mae

            # Store history for final figure
            all_step_data[cid]["process_s"].append(metrics["process_s"])
            all_step_data[cid]["process_c"].append(metrics["process_c"])
            all_step_data[cid]["delta_power"].append(metrics["delta_power"])
            all_step_data[cid]["beta_power"].append(metrics["beta_power"])

        # --- TensorBoard: scalars ---
        # Environment (shared, logged once from cohort 0)
        tb_logger.log_environment(step, step_results[0]["env_lux"], t_h % 24.0)

        for cid, metrics in step_results.items():
            tb_logger.log_step(step, cid, {
                "process_s":     metrics["process_s"],
                "process_c":     metrics["process_c"],
                "sleep_stage":   float(metrics["sleep_stage"]),
                "subj_wake":     metrics["subj_wake"],
                "delta_power":   metrics["delta_power"],
                "theta_power":   metrics["theta_power"],
                "alpha_power":   metrics["alpha_power"],
                "sigma_power":   metrics["sigma_power"],
                "beta_power":    metrics["beta_power"],
                "mae_physionet": metrics["mae_physionet"],
                "vlpo_activity": metrics["vlpo_activity"],
                "lha_activity":  metrics["lha_activity"],
            })

        # --- TensorBoard: image logs every 24 simulated hours ---
        if step > 0 and step % config.IMAGE_LOG_INTERVAL == 0:
            spikes_c0 = cohorts[0].last_spikes
            spikes_c3 = cohorts[3].last_spikes
            if spikes_c0 is not None and spikes_c3 is not None:
                tb_logger.log_raster_image(step, spikes_c0, spikes_c3, sim_hour=t_h)
            all_hyps = [np.array(c.hypnogram) for c in cohorts]
            tb_logger.log_hypnogram_image(step, all_hyps, sim_hour=t_h)

        # --- Live dashboard ---
        for cid, metrics in step_results.items():
            dashboard.push(cid, step, metrics)
        if step % config.DASHBOARD_UPDATE_INTERVAL == 0:
            dashboard.update()

    # Step 5: Final outputs
    print("\n[5/5] Finalising outputs ...")
    tb_logger.flush()  # Flush TensorBoard

    # Compute final sleep metrics
    all_metrics: Dict[int, Dict] = {}
    for cid, cohort in enumerate(cohorts):
        hyp = np.array(cohort.hypnogram)
        s_series  = all_step_data[cid]["process_s"]
        final_s   = s_series[-1] if len(s_series) > 0 else 0.35
        s_nadir   = float(np.min(s_series)) if len(s_series) > 0 else final_s
        inh_scale = cohort.params.cortical_inhibition_scale
        phase_off = cohort.params.circadian_phase_offset_h
        arousal   = cohort.params.arousal_bias
        all_metrics[cid] = compute_sleep_metrics(
            hyp,
            dt_min=step_dt,
            final_s=final_s,
            s_nadir=s_nadir,
            cortical_inh_scale=inh_scale,
            phase_offset_h=phase_off,
            arousal_bias=arousal,
        )
        mae = compute_rolling_mae(cohort.band_history, reference_bands,
                                  window=len(cohort.band_history))
        all_metrics[cid]["mae_physionet"] = round(float(mae), 4)

    # Clinical summary table
    print("\n" + "=" * 70)
    print("  CLINICAL SLEEP SUMMARY - Multi-Cohort Simulation")
    print("=" * 70)
    summary_table = build_clinical_summary_table(all_metrics)
    print(summary_table)
    tb_logger.log_text("Clinical/Summary_Table", f"```\n{summary_table}\n```", step=n_steps)

    # MAE per cohort
    print("\n  Spectral MAE vs. PhysioNet Reference:")
    for cid, cohort in enumerate(cohorts):
        mae = all_metrics[cid]["mae_physionet"]
        print(f"    Cohort {cid} [{cohort.name}]: MAE = {mae:.5f}")

    save_summary_figure(
        all_cohorts     = cohorts,
        reference_bands = reference_bands,
        all_metrics     = all_metrics,
        t_axis          = t_axis,
        all_step_data   = all_step_data,
    )

    # Export JSON telemetry for web dashboard
    results_json_path = os.path.join(config.OUTPUT_DIR, "simulation_results.json")
    try:
        sub_step = 1 if (dry_run or demo_run) else 5
        sub_indices = list(range(0, n_steps, sub_step))
        if (n_steps - 1) not in sub_indices:
            sub_indices.append(n_steps - 1)
        export_payload = {
            "duration_hours":  sim_duration_h,
            "n_steps":         n_steps,
            "dry_run":         dry_run,
            "demo_run":        demo_run,
            "reference_bands": reference_bands,
            "clinical_metrics": {
                cid: {k: (round(v, 2) if isinstance(v, float) else v) for k, v in all_metrics[cid].items()}
                for cid in all_metrics
            },
            "cohort_names":    config.COHORT_NAMES,
            "cohort_colors":   config.COHORT_COLORS,
            "time_hours":      [round((i * step_dt) / 60.0, 2) for i in sub_indices],
            "cohort_data": {
                cid: {
                    "name":        cohorts[cid].name,
                    "process_s":   [round(all_step_data[cid]["process_s"][i], 4) for i in sub_indices],
                    "process_c":   [round(all_step_data[cid]["process_c"][i], 4) for i in sub_indices],
                    "sleep_stage": [int(cohorts[cid].hypnogram[i]) for i in sub_indices],
                    "delta_power": [round(all_step_data[cid]["delta_power"][i], 4) for i in sub_indices],
                    "beta_power":  [round(all_step_data[cid]["beta_power"][i], 4) for i in sub_indices],
                }
                for cid in range(config.N_COHORTS)
            }
        }
        with open(results_json_path, "w", encoding="utf-8") as jf:
            json.dump(export_payload, jf, indent=2)
        print(f"[Output] JSON telemetry saved -> {os.path.abspath(results_json_path)}")
    except Exception as e:
        print(f"[Warning] Failed to save JSON results: {e}")

    # Close logger
    tb_logger.close()

    # Elapsed time
    elapsed = time.time() - t_start
    print(f"\n  Total wall-clock time: {elapsed:.1f}s  ({elapsed/60:.1f} min)")

    # TensorBoard instructions
    abs_log = os.path.abspath(config.TENSORBOARD_LOG_DIR)
    print("\n" + "=" * 70)
    print("  [OK] Simulation complete!")
    print()
    print("  To launch TensorBoard, run:")
    print(f"      tensorboard --logdir=\"{abs_log}\"")
    print()
    print("  Then open:  http://localhost:6006")
    print()
    print("  TensorBoard groups to explore:")
    print(f"    * Sleep_Pressure/Process_S  - overlay all {config.N_COHORTS} cohorts")
    print("    * Circadian/Process_C       - phase differences")
    print("    * EEG_Power/Delta_Relative  - slow-wave architecture")
    print("    * EEG_Power/Beta_Relative   - paradoxical insomnia signature")
    print("    * State/Subjective_Wakefulness_Meter")
    print("    * Clinical/MAE_vs_PhysioNet")
    print("    * Raster_Plots + Hypnogram  (Images tab)")
    print("=" * 70)

    # Keep live dashboard open only if interactive GUI explicitly requested
    if gui and not (dry_run or demo_run):
        dashboard.finalize(block=True)
    else:
        plt.close("all")

    if tee is not None:
        tee.close()

# endregion Main


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DyssomniaNN - SNN Sleep Disorder Simulator")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run 20-step smoke test (20 min) for pipeline compilation check"
    )
    parser.add_argument(
        "--demo-run", action="store_true",
        help="Run accelerated 24-hour cycle (144 steps, dt=10 min, ~25s) with full day-night transitions"
    )
    parser.add_argument(
        "--days", type=float, default=None,
        help="Number of simulated days (default: 3 days in full mode, 1 day in demo mode)"
    )
    parser.add_argument(
        "--gui", action="store_true", default=False,
        help="Enable interactive GUI window"
    )
    parser.add_argument(
        "--no-gui", action="store_true", default=False,
        help="Disable interactive GUI window (headless mode is default)"
    )
    parser.add_argument(
        "--log-file", type=str, default=None,
        help="Path to real-time log file (e.g. simulation.log)"
    )
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    gui_enabled = args.gui and not args.no_gui
    main(dry_run=args.dry_run, demo_run=args.demo_run, days=args.days, gui=gui_enabled, log_file=args.log_file)
    sys.exit(0)
