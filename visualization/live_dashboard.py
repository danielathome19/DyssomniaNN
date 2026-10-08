"""
Real-time multi-panel Matplotlib dashboard updated during simulation.

Layout (4x3 grid)
-----------------
Row 0: Process S (all 4 cohorts) | Process C  | Sleep Stage
Row 1: Delta power               | Beta power | Subjective Wakefulness
Row 2: LFP traces C0 & C3        | Hypnogram  | MAE vs PhysioNet

The dashboard updates every DASHBOARD_UPDATE_INTERVAL simulation steps
via plt.pause() in non-blocking mode.
"""

from __future__ import annotations

import os
import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import deque
from matplotlib.lines import Line2D
from typing import Dict, List, Optional
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config


class LiveDashboard:
    """
    Real-time simulation dashboard.

    Parameters
    ----------
    window_size : number of past timesteps shown in rolling plots
    enable_gui  : whether to open an interactive GUI window (default: False, headless)
    """

    _STAGE_NAMES = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}

    def __init__(self, window_size: int = 720, enable_gui: bool = False):
        self.window       = window_size
        self._step        = 0
        self._ready       = False
        self._interactive = enable_gui

        # Rolling buffers: {cohort_id → deque}
        keys = [
            "t_h", "S", "C", "stage", "delta", "beta",
            "subj_wake", "mae", "lfp",
        ]
        self._data: Dict[int, Dict[str, deque]] = {
            cid: {k: deque(maxlen=window_size) for k in keys}
            for cid in range(config.N_COHORTS)
        }
        self._full_hypnograms: Dict[int, List[int]] = {
            cid: [] for cid in range(config.N_COHORTS)
        }
        self._full_t: List[float] = []

        if self._interactive:
            try:
                import tkinter
                matplotlib.use("TkAgg")
                self._setup_figure()
            except Exception as e:
                self._interactive = False
                self._ready = False
                print(f"[Dashboard] Interactive GUI unavailable ({e}) - operating in headless mode.")

    # =========================================================================
    # Public API
    # =========================================================================
    
    def push(self, cohort_id: int, step: int, metrics: dict) -> None:
        """
        Push one timestep of metrics into the rolling buffer for a cohort.

        Parameters
        ----------
        cohort_id : 0--3
        step      : current simulation step
        metrics   : dict with keys matching buffer keys above
        """
        d = self._data[cohort_id]
        d["t_h"].append(metrics.get("t_h", step * config.DT_HOURS))
        d["S"].append(metrics.get("process_s", 0.0))
        d["C"].append(metrics.get("process_c", 0.0))
        d["stage"].append(metrics.get("sleep_stage", 0))
        d["delta"].append(metrics.get("delta_power", 0.0))
        d["beta"].append(metrics.get("beta_power", 0.0))
        d["subj_wake"].append(metrics.get("subj_wake", 0.0))
        d["mae"].append(metrics.get("mae_physionet", 0.0))
        d["lfp"].append(metrics.get("lfp_sample", 0.0))

        # Full hypnogram (for 72h panel)
        self._full_hypnograms[cohort_id].append(metrics.get("sleep_stage", 0))

        if cohort_id == 0:  # only append time once (from cohort 0)
            self._full_t.append(metrics.get("t_h", step * config.DT_HOURS))

    def update(self) -> None:
        """Redraw all dashboard panels. Call every DASHBOARD_UPDATE_INTERVAL steps."""
        if not self._interactive or not self._ready:
            return
        try:
            self._draw()
            plt.pause(0.001)
        except Exception:
            pass  # Window closed by user -- ignore

    def finalize(self, block: bool = False) -> None:
        """Keep the dashboard window open after simulation ends if block=True."""
        if self._interactive and self._ready:
            plt.ioff()
            try:
                if block:
                    plt.show()
                else:
                    plt.close(self._fig)
            except Exception:
                pass

    # =========================================================================
    # Private: Figure setup
    # =========================================================================
    
    def _setup_figure(self) -> None:
        plt.ion()
        self._fig = plt.figure(figsize=(18, 10), facecolor="#0d0d1a")
        self._fig.canvas.manager.set_window_title(
            "DyssomniaNN - Multi-Cohort SNN Sleep Simulator"
        )

        gs = gridspec.GridSpec(
            3, 3,
            figure=self._fig,
            hspace=0.45, wspace=0.35,
            left=0.06, right=0.97, top=0.93, bottom=0.07,
        )

        # --- Create axes ---
        self._ax_s     = self._fig.add_subplot(gs[0, 0])
        self._ax_c     = self._fig.add_subplot(gs[0, 1])
        self._ax_stage = self._fig.add_subplot(gs[0, 2])
        self._ax_delta = self._fig.add_subplot(gs[1, 0])
        self._ax_beta  = self._fig.add_subplot(gs[1, 1])
        self._ax_wake  = self._fig.add_subplot(gs[1, 2])
        self._ax_lfp   = self._fig.add_subplot(gs[2, 0])
        self._ax_hyp   = self._fig.add_subplot(gs[2, 1])
        self._ax_mae   = self._fig.add_subplot(gs[2, 2])

        _all_axes = [
            self._ax_s, self._ax_c, self._ax_stage,
            self._ax_delta, self._ax_beta, self._ax_wake,
            self._ax_lfp, self._ax_hyp, self._ax_mae,
        ]
        for ax in _all_axes:
            _style_ax(ax)

        # --- Title ---
        self._fig.text(
            0.5, 0.97,
            "DyssomniaNN - Real-Time 5-Cohort SNN Sleep Disorder Simulator",
            ha="center", va="top", color="white", fontsize=13,
            fontweight="bold",
            fontfamily="monospace",
        )

        # --- Legend ---
        legend_elements = [
            Line2D([0], [0], color=config.COHORT_COLORS[i], linewidth=2,
                   label=config.COHORT_NAMES[i])
            for i in range(config.N_COHORTS)
        ]
        self._fig.legend(
            handles=legend_elements, loc="upper right",
            bbox_to_anchor=(0.99, 0.96), fontsize=8,
            facecolor="#1a1a2e", edgecolor="#334466", labelcolor="white",
            ncol=2,
        )
        self._ready = True

    # =========================================================================
    # Private: Draw
    # =========================================================================
    
    def _draw(self) -> None:
        axes_map = {
            "S":         (self._ax_s,     "Process S — Homeostatic Pressure", "S(t)"),
            "C":         (self._ax_c,     "Process C — Circadian Drive",      "C(t)"),
            "delta":     (self._ax_delta, "Delta Power (0.5—4 Hz)",           "Rel. Power"),
            "beta":      (self._ax_beta,  "Beta Power (15—30 Hz)",            "Rel. Power"),
            "subj_wake": (self._ax_wake,  "Subjective Wakefulness",            "%"),
            "mae":       (self._ax_mae,   "MAE vs PhysioNet",                  "MAE"),
        }

        for key, (ax, title, ylabel) in axes_map.items():
            ax.clear()
            _style_ax(ax)
            ax.set_title(title, color="white", fontsize=8, pad=3)
            ax.set_ylabel(ylabel, color="#aaaaaa", fontsize=7)
            for cid in range(config.N_COHORTS):
                d = self._data[cid]
                if len(d["t_h"]) < 2:
                    continue
                t = np.array(d["t_h"])
                y = np.array(d[key])
                ax.plot(t, y, color=config.COHORT_COLORS[cid],
                        linewidth=0.9, alpha=0.9)
            ax.set_xlabel("Sim time (h)", color="#aaaaaa", fontsize=7)

        # --- Sleep stage ---
        self._ax_stage.clear()
        _style_ax(self._ax_stage)
        self._ax_stage.set_title("Sleep Stage", color="white", fontsize=8, pad=3)
        for cid in range(config.N_COHORTS):
            d = self._data[cid]
            if len(d["t_h"]) < 2:
                continue
            t = np.array(d["t_h"])
            y = np.array(d["stage"]) + cid * 0.1  # slight y offset per cohort
            self._ax_stage.step(t, y, color=config.COHORT_COLORS[cid],
                                linewidth=0.8, where="post", alpha=0.85)
        self._ax_stage.set_yticks([0, 1, 2, 3, 4])
        self._ax_stage.set_yticklabels(["Wake", "N1", "N2", "N3", "REM"],
                                        color="#aaaaaa", fontsize=7)
        self._ax_stage.set_xlabel("Sim time (h)", color="#aaaaaa", fontsize=7)

        # --- LFP traces (Cohort 0 & 3 only) ---
        self._ax_lfp.clear()
        _style_ax(self._ax_lfp)
        self._ax_lfp.set_title("LFP Signal (C0 Normal vs C3 Paradoxical)",
                                color="white", fontsize=8, pad=3)
        for cid in [0, 3]:
            d = self._data[cid]
            if len(d["lfp"]) < 2:
                continue
            y = np.array(d["lfp"])
            self._ax_lfp.plot(y, color=config.COHORT_COLORS[cid],
                               linewidth=0.7, alpha=0.85)
        self._ax_lfp.set_xlabel("Recent steps", color="#aaaaaa", fontsize=7)
        self._ax_lfp.set_ylabel("LFP (a.u.)",   color="#aaaaaa", fontsize=7)

        # --- Full 72h hypnogram ---
        self._ax_hyp.clear()
        _style_ax(self._ax_hyp)
        self._ax_hyp.set_title("Hypnograms -- Full 72h", color="white", fontsize=8, pad=3)
        for cid in range(config.N_COHORTS):
            hyp = self._full_hypnograms[cid]
            if len(hyp) < 2:
                continue
            t = np.array(self._full_t[:len(hyp)])
            y = np.array(hyp, dtype=float) + cid * 5.2  # stack vertically
            self._ax_hyp.step(t, y, color=config.COHORT_COLORS[cid],
                               linewidth=0.8, where="post", alpha=0.9)
        self._ax_hyp.set_xlabel("Sim time (h)", color="#aaaaaa", fontsize=7)

        self._fig.canvas.draw_idle()


def _style_ax(ax: plt.Axes) -> None:  # Helper
    ax.set_facecolor("#0d0d1a")
    ax.tick_params(colors="#777777", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2a2a4a")
    ax.grid(True, color="#1a1a3a", linewidth=0.4, alpha=0.6)


if __name__ == "__main__":
    dash = LiveDashboard()
    for step in range(200):
        t_h = step / 60.0
        for cid in range(4):
            dash.push(cid, step, {
                "t_h":           t_h,
                "process_s":     0.3 + 0.3 * np.sin(step / 50 + cid),
                "process_c":     0.2 * np.cos(step / 60),
                "sleep_stage":   int(step % 5),
                "delta_power":   0.3 + 0.1 * cid,
                "beta_power":    0.08 + 0.05 * cid,
                "subj_wake":     80 - step * 0.2,
                "mae_physionet": 0.05,
                "lfp_sample":    np.random.randn() * 0.1,
            })
        if step % 10 == 0:
            dash.update()
    dash.finalize()
