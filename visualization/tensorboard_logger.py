"""
Real-time concurrent 4-cohort TensorBoard scalar and image logger.

All 4 cohorts are written with the same global_step so TensorBoard
automatically overlays them in the "Scalars" tab when you group by tag.

Usage
-----
    logger = SleepTensorBoardLogger(log_dir="./runs/sleep_simulation")
    logger.log_step(step=42, cohort_id=0, metrics={...})
    logger.log_raster_image(step=1440, spikes_c0=arr0, spikes_c3=arr3)
    logger.log_hypnogram_image(step=1440, all_hypnograms=[h0, h1, h2, h3])
    logger.close()
"""

from __future__ import annotations

import io
import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import Dict, List, Optional, Any
matplotlib.use("Agg")   # Non-interactive backend for writer thread
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False
    SummaryWriter = None


class SleepTensorBoardLogger:
    """
    Wraps PyTorch SummaryWriter to log all 4 cohorts under unified metric
    groups, enabling overlay comparison in TensorBoard.

    Parameters
    ----------
    log_dir    : TensorBoard log directory (default: config.TENSORBOARD_LOG_DIR)
    flush_secs : how often TensorBoard flushes to disk
    """

    # Scalar tag templates -- cohort name is appended for each cohort
    _SCALAR_TAGS = {
        "env_lux":       "Environment/Sunlight_Lux",
        "env_hour":      "Environment/Simulated_Hour",
        "process_s":     "Sleep_Pressure/Process_S",
        "process_c":     "Circadian/Process_C",
        "sleep_stage":   "State/Sleep_Stage",
        "subj_wake":     "State/Subjective_Wakefulness_Meter",
        "delta_power":   "EEG_Power/Delta_Relative",
        "theta_power":   "EEG_Power/Theta_Relative",
        "alpha_power":   "EEG_Power/Alpha_Relative",
        "sigma_power":   "EEG_Power/Sigma_Relative",
        "beta_power":    "EEG_Power/Beta_Relative",
        "mae_physionet": "Clinical/MAE_vs_PhysioNet",
        "vlpo_activity": "Switch/VLPO_Activity",
        "lha_activity":  "Switch/LHA_Activity",
    }

    def __init__(
        self,
        log_dir:    str = config.TENSORBOARD_LOG_DIR,
        flush_secs: int = 30,
    ):
        self.log_dir = log_dir
        self._enabled = _TB_AVAILABLE
        self._writers: Dict[Any, SummaryWriter] = {}

        if self._enabled:
            import glob
            os.makedirs(log_dir, exist_ok=True)
            for cid, name in config.COHORT_NAMES.items():
                clean_name = f"{cid}_{name.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')}"
                sub_dir = os.path.join(log_dir, clean_name)
                os.makedirs(sub_dir, exist_ok=True)
                for old_f in glob.glob(os.path.join(sub_dir, "events.out.tfevents.*")):
                    try:
                        os.remove(old_f)
                    except OSError:
                        pass
                self._writers[cid] = SummaryWriter(log_dir=sub_dir, flush_secs=flush_secs)

            env_dir = os.path.join(log_dir, "Environment")
            os.makedirs(env_dir, exist_ok=True)
            for old_f in glob.glob(os.path.join(env_dir, "events.out.tfevents.*")):
                try:
                    os.remove(old_f)
                except OSError:
                    pass
            self._writers["env"] = SummaryWriter(log_dir=env_dir, flush_secs=flush_secs)
            print(f"[TensorBoard] Multi-run logging initialized for {len(config.COHORT_NAMES)} cohorts in: {os.path.abspath(log_dir)}")
        else:
            print("[TensorBoard] WARNING: tensorboard not installed. Logging disabled.")
            print("              Install with: pip install tensorboard")

    def log_step(
        self,
        step:      int,
        cohort_id: int,
        metrics:   Dict[str, float],
    ) -> None:
        """
        Write all scalar metrics for one cohort at one simulation step.
        Logs to the cohort's dedicated sub-run under a unified base tag,
        enabling simultaneous overlay comparison of all cohorts on the same plot.

        Parameters
        ----------
        step      : global simulation step (0 -- N_STEPS)
        cohort_id : 0--4
        metrics   : dict mapping metric keys to float values
                    (keys must match _SCALAR_TAGS above)
        """
        if not self._enabled or cohort_id not in self._writers:
            return

        writer = self._writers[cohort_id]
        for key, base_tag in self._SCALAR_TAGS.items():
            if key in metrics:
                writer.add_scalar(base_tag, metrics[key], global_step=step)

    def log_environment(
        self,
        step:     int,
        lux:      float,
        sim_hour: float,
    ) -> None:
        """Log environment-level (shared) scalars into the Environment run."""
        if not self._enabled or "env" not in self._writers:
            return
        self._writers["env"].add_scalar("Environment/Sunlight_Lux",    lux,     global_step=step)
        self._writers["env"].add_scalar("Environment/Simulated_Hour",  sim_hour, global_step=step)

    def log_raster_image(
        self,
        step:      int,
        spikes_c0: np.ndarray,
        spikes_c3: np.ndarray,
        sim_hour:  float = 0.0,
    ) -> None:
        """
        Create and log a 4-panel raster comparison: Cohort 0 vs Cohort 3.

        Panels: C0 thalamic | C0 cortical | C3 thalamic | C3 cortical
        """
        if not self._enabled or "env" not in self._writers:
            return
        fig = _make_raster_figure(spikes_c0, spikes_c3, sim_hour)
        img = _fig_to_rgb(fig)
        plt.close(fig)
        self._writers["env"].add_image("Raster_Plots/Cohort_0_vs_Cohort_3", img, global_step=step)

    def log_hypnogram_image(
        self,
        step:           int,
        all_hypnograms: List[np.ndarray],
        sim_hour:       float = 0.0,
    ) -> None:
        """
        Log a 5-row hypnogram alignment figure for all cohorts.
        """
        if not self._enabled or "env" not in self._writers:
            return
        fig = _make_hypnogram_figure(all_hypnograms, sim_hour)
        img = _fig_to_rgb(fig)
        plt.close(fig)
        self._writers["env"].add_image("Hypnogram/All_Cohorts_72h", img, global_step=step)

    def log_text(self, tag: str, text: str, step: int) -> None:
        """Log a plain text entry (e.g. clinical summary table)."""
        if not self._enabled or "env" not in self._writers:
            return
        self._writers["env"].add_text(tag, text, global_step=step)

    def flush(self) -> None:
        """Force-flush pending writes to disk across all writers."""
        if self._enabled:
            for writer in self._writers.values():
                writer.flush()

    def close(self) -> None:
        """Close all SummaryWriters."""
        if self._enabled:
            for writer in self._writers.values():
                writer.close()

    @property
    def is_enabled(self) -> bool:
        return self._enabled


# region Figure builders
# (called by log methods above)

def _make_raster_figure(
    spikes_c0: np.ndarray,  # [T, N]
    spikes_c3: np.ndarray,  # [T, N]
    sim_hour:  float,
) -> plt.Figure:
    """4-panel spike raster: C0 thalamic, C0 cortical, C3 thalamic, C3 cortical."""
    n_tc = config.SNN_N_THALAMIC
    n_ce = config.SNN_N_CORTICAL_E
    n_ci = config.SNN_N_CORTICAL_I

    fig, axes = plt.subplots(2, 2, figsize=(12, 6), facecolor="#1a1a2e")
    fig.suptitle(
        f"Spike Raster — Cohort 0 (Normal) vs Cohort 3 (Paradoxical)  "
        f"|  Sim hour {sim_hour:.1f}h",
        color="white", fontsize=11, y=0.99
    )

    panels = [
        (axes[0, 0], spikes_c0[:, :n_tc],            "C0: Thalamic Relay",   "#4FC3F7"),
        (axes[0, 1], spikes_c0[:, n_tc:n_tc + n_ce], "C0: Cortical Excit.",  "#81D4FA"),
        (axes[1, 0], spikes_c3[:, :n_tc],            "C3: Thalamic Relay",   "#CE93D8"),
        (axes[1, 1], spikes_c3[:, n_tc:n_tc + n_ce], "C3: Cortical Excit.",  "#F48FB1"),
    ]

    for ax, spikes, title, color in panels:
        ax.set_facecolor("#0d0d1a")
        spike_t, spike_n = np.where(spikes > 0.5)
        ax.scatter(spike_t, spike_n, s=1.5, c=color, alpha=0.7)
        ax.set_title(title, color="white", fontsize=9)
        ax.set_xlabel("Timestep", color="#aaaaaa", fontsize=7)
        ax.set_ylabel("Neuron #",  color="#aaaaaa", fontsize=7)
        ax.tick_params(colors="#777777", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")

    plt.tight_layout()
    return fig


def _make_hypnogram_figure(
    all_hypnograms: List[np.ndarray],
    sim_hour:       float,
) -> plt.Figure:
    """4-row hypnogram alignment for all cohorts up to sim_hour."""
    n_cohorts = len(all_hypnograms)
    fig, axes = plt.subplots(n_cohorts, 1, figsize=(14, 7), facecolor="#1a1a2e",
                              sharex=True)
    fig.suptitle(
        f"Concurrent Hypnograms — 72h Multi-Cohort | Up to t={sim_hour:.1f}h",
        color="white", fontsize=11
    )

    stage_labels = {0: "Wake", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}

    for i, (ax, hyp) in enumerate(zip(axes, all_hypnograms)):
        name   = config.COHORT_NAMES.get(i, f"Cohort {i}")
        color  = config.COHORT_COLORS.get(i, "white")
        t_axis = np.arange(len(hyp)) / 60.0  # convert to hours

        ax.set_facecolor("#0d0d1a")
        ax.step(t_axis, hyp, color=color, linewidth=0.9, where="post")
        ax.fill_between(t_axis, 0, hyp, step="post", alpha=0.25, color=color)
        ax.set_yticks([0, 1, 2, 3, 4])
        ax.set_yticklabels(["Wake", "N1", "N2", "N3", "REM"],
                           color="#aaaaaa", fontsize=7)
        ax.set_ylabel(name, color=color, fontsize=8, rotation=0,
                      labelpad=75, va="center")
        ax.tick_params(colors="#777777", labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")

        # Mark light transitions
        for x in range(0, int(sim_hour) + 1, 24):
            ax.axvline(x=x, color="#555555", linewidth=0.6, linestyle=":")

    axes[-1].set_xlabel("Simulated Time (hours)", color="#aaaaaa", fontsize=9)
    plt.tight_layout()
    return fig


def _fig_to_rgb(fig: plt.Figure) -> np.ndarray:
    """Convert a matplotlib figure to a [C, H, W] RGB numpy array for TensorBoard."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=96, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    from PIL import Image
    pil_img = Image.open(buf).convert("RGB")
    arr     = np.array(pil_img)    # [H, W, 3]
    return arr.transpose(2, 0, 1)  # [3, H, W]  for TensorBoard

# endregion Figure builders


if __name__ == "__main__":
    logger = SleepTensorBoardLogger(log_dir="./runs/test_logger")
    for step in range(5):
        for cid in range(4):
            logger.log_step(step, cid, {
                "process_s":     0.3 + step * 0.02 + cid * 0.05,
                "process_c":     0.1 * step,
                "sleep_stage":   cid,
                "delta_power":   0.4,
                "beta_power":    0.08,
                "subj_wake":     80.0 - step * 5,
                "mae_physionet": 0.05,
            })
    logger.close()
    print("Logger test complete.")
