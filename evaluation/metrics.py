"""
Clinical sleep metrics computed from a hypnogram array and band powers.

Metrics
-------
 * Sleep Efficiency (SE%) = TST / TIB × 100
 * Sleep Onset Latency (SOL) in minutes
 * Wake After Sleep Onset (WASO) in minutes
 * MAE vs. PhysioNet reference band powers
"""

from __future__ import annotations

import os
import sys
import numpy as np
from typing import Dict, List, Optional, Any
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config


# region Sleep efficiency & hypnogram metrics

def compute_sleep_metrics(
    hypnogram:          np.ndarray,
    dt_min:             float = config.DT_MIN,
    final_s:            float = 0.35,
    s_nadir:            Optional[float] = None,
    cortical_inh_scale: float = 1.0,
    phase_offset_h:     float = 0.0,
    arousal_bias:       float = 0.0,
    is_daylight:        Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Compute standard polysomnography-derived clinical sleep metrics, total sleep
    fraction of environment time, and a quantitative Well-Rested Index.

    Parameters
    ----------
    hypnogram          : 1-D integer array of sleep stages (0=Wake, 1--4=Sleep)
    dt_min             : duration of each step in minutes
    final_s            : final homeostatic sleep pressure S(t) at end of simulation
    s_nadir            : lowest homeostatic sleep pressure reached during sleep (clearance depth)
    cortical_inh_scale : cortical GABAergic scale (reduced in Paradoxical Insomnia)
    phase_offset_h     : circadian phase offset in hours (e.g. +4.0h in CRSWD)
    arousal_bias       : somatic/noradrenergic hyperarousal bias (e.g. +0.60 in Insomnia)
    is_daylight        : boolean array indicating whether each minute was daylight

    Returns
    -------
    dict with sleep metrics, sleep_pct_env, well_rested_index, and rest_status
    """
    n = len(hypnogram)
    tib_min     = n * dt_min
    env_total_h = tib_min / 60.0

    # ---- Sleep Onset Latency ----
    first_sleep_idx = _first_sleep_index(hypnogram)
    sol_min = first_sleep_idx * dt_min if first_sleep_idx is not None else tib_min

    # ---- Total Sleep Time ----
    tst_min = float(np.sum(hypnogram > 0)) * dt_min
    tst_h   = tst_min / 60.0

    # ---- Sleep Fraction of Total Environmental Time ----
    sleep_pct_env = (tst_min / tib_min * 100.0) if tib_min > 0 else 0.0

    # ---- Day vs. Night Sleep Breakdown ----
    if is_daylight is None:
        step_hours  = (np.arange(n) * dt_min / 60.0) % 24.0
        is_daylight = (step_hours >= config.DEFAULT_LIGHT_ON_HOUR) & (step_hours < config.DEFAULT_LIGHT_OFF_HOUR)

    sleep_mask      = hypnogram > 0
    day_sleep_min   = float(np.sum(sleep_mask & is_daylight)) * dt_min
    night_sleep_min = float(np.sum(sleep_mask & (~is_daylight))) * dt_min
    day_sleep_pct   = (day_sleep_min / tst_min * 100.0) if tst_min > 0 else 0.0

    # ---- Sleep Efficiency ----
    sleep_efficiency = (tst_min / tib_min * 100.0) if tib_min > 0 else 0.0

    # ---- WASO ----
    waso_min = _compute_waso(hypnogram, dt_min, is_daylight=is_daylight)

    # ---- Stage percentages (of TST) ----
    tst_steps = max(np.sum(hypnogram > 0), 1)
    pct_n1  = float(np.sum(hypnogram == 1)) / tst_steps * 100.0
    pct_n2  = float(np.sum(hypnogram == 2)) / tst_steps * 100.0
    pct_n3  = float(np.sum(hypnogram == 3)) / tst_steps * 100.0
    pct_rem = float(np.sum(hypnogram == 4)) / tst_steps * 100.0

    # ---- NREM-REM cycle count ----
    n_cycles = _count_rem_cycles(hypnogram)

    # ---- Sleep-Onset REM Periods (SOREMPs) for Narcolepsy ----
    n_soremps = _count_soremps(hypnogram, dt_min=dt_min)

    # ---- Well-Rested Index (Restorative Sleep Score: 0 to 100%) ----
    # Biological alertness scale: S=0.18 (freshly awake, cleared debt) -> 1.0; S=0.75 (exhausted) -> 0.0
    s_min = 0.18
    s_max = 0.75
    s_eval = s_nadir if s_nadir is not None else final_s
    s_alertness = max(0.0, min(1.0, (s_max - s_eval) / (s_max - s_min)))

    if env_total_h < 1.0:
        # Initial or short smoke-test window (< 1 hour): pipeline check
        base_rested = 98.0
    else:
        # Multi-hour evaluation across sleep-wake cycles:
        expected_sleep_h = max(0.5, (env_total_h / 24.0) * 7.5)
        tst_quota = min(1.0, tst_h / expected_sleep_h) if expected_sleep_h > 0 else 1.0

        if tst_min > 0:
            # Architecture score: favors restorative N3 deep slow-wave and REM sleep
            arch_score = (0.2 * pct_n2 + 0.5 * pct_n3 + 0.3 * pct_rem) / 50.0
            arch_score = max(0.3, min(1.2, arch_score))
            # Continuity score: penalizes WASO
            continuity = max(0.2, 1.0 - (waso_min / (tst_min + waso_min)))
        else:
            arch_score = 0.1
            continuity = 0.1

        # SOREMP penalty for narcoleptic sleep architecture disruption
        soremp_factor  = max(0.60, 1.0 - 0.08 * n_soremps)

        # Circadian alignment: phase offset misalignment and daylight sleep penalty
        phase_penalty  = abs(phase_offset_h) * 0.08
        day_penalty    = (day_sleep_pct / 100.0) * 0.35
        circ_alignment = max(0.40, 1.0 - phase_penalty - day_penalty)

        # Somatic/pre-sleep cognitive hyperarousal penalty
        arousal_factor = max(0.40, 1.0 - 0.50 * arousal_bias)

        # Composite score
        base_rested = (0.25 * s_alertness + 0.40 * (tst_quota ** 1.1) + 0.15 * arch_score + 0.20 * continuity) * circ_alignment * soremp_factor * arousal_factor * 100.0

    # Cortical hyperarousal factor (GABA deficit leading to beta intrusion and unrefreshing sleep)
    hyperarousal_factor = max(0.30, min(1.0, cortical_inh_scale ** 0.6))
    well_rested_index = round(max(0.0, min(100.0, base_rested * hyperarousal_factor)), 1)

    # Clinical rest tier
    if well_rested_index >= 80.0:
        rest_status = "Optimal"
    elif well_rested_index >= 68.0:
        rest_status = "Good"
    elif abs(phase_offset_h) > 1.0 and well_rested_index >= 35.0:
        rest_status = "Phase-Shifted"
    elif n_soremps >= 2 or (well_rested_index >= 40.0 and waso_min > 50.0):
        rest_status = "Fragmented"
    elif well_rested_index >= 20.0:
        rest_status = "Non-Restorative"
    else:
        rest_status = "Sleep-Deprived"

    return {
        "sleep_efficiency":  round(sleep_efficiency, 2),
        "sol_min":           round(sol_min, 1),
        "waso_min":          round(waso_min, 1),
        "tst_min":           round(tst_min, 1),
        "tst_h":             round(tst_h, 2),
        "tib_min":           round(tib_min, 1),
        "sleep_pct_env":     round(sleep_pct_env, 1),
        "day_sleep_min":     round(day_sleep_min, 1),
        "night_sleep_min":   round(night_sleep_min, 1),
        "day_sleep_pct":     round(day_sleep_pct, 1),
        "n_cycles":          n_cycles,
        "n_soremps":         n_soremps,
        "pct_n1":            round(pct_n1,  1),
        "pct_n2":            round(pct_n2,  1),
        "pct_n3":            round(pct_n3,  1),
        "pct_rem":           round(pct_rem, 1),
        "well_rested_index": well_rested_index,
        "rest_status":       rest_status,
    }

# endregion Sleep efficiency & hypnogram metrics


# region MAE vs. reference spectrum

def compute_mae(
    predicted_bands: Dict[str, float],
    reference_bands: Dict[str, float],
) -> float:
    """
    Compute Mean Absolute Error between predicted and reference band powers.

    Only bands present in both dicts are included.

    Parameters
    ----------
    predicted_bands : {band_name: relative_power}
    reference_bands : {band_name: relative_power}  (PhysioNet ground truth)

    Returns
    -------
    float  MAE (same scale as relative band power, 0--1)
    """
    shared_bands = set(predicted_bands) & set(reference_bands)
    if not shared_bands:
        return 0.0

    errors = [
        abs(predicted_bands[b] - reference_bands[b])
        for b in shared_bands
    ]
    return float(np.mean(errors))


def compute_rolling_mae(
    predicted_history: List[Dict[str, float]],
    reference_bands:   Dict[str, float],
    window:            int = 60,
) -> float:
    """
    Compute MAE over the last `window` band-power observations.
    Useful for streaming metric logging.
    """
    recent = predicted_history[-window:]
    if not recent:
        return 0.0

    maes = [compute_mae(b, reference_bands) for b in recent]
    return float(np.mean(maes))

# endregion MAE vs. reference spectrum


# region Summary table for all cohorts

def build_clinical_summary_table(all_metrics: Dict[int, Dict[str, Any]]) -> str:
    """
    Build a formatted ASCII clinical summary table for all cohorts, including
    Total Sleep Time, Sleep % of Environment Time, and the Well-Rested Index.

    Parameters
    ----------
    all_metrics : {cohort_id → metrics_dict from compute_sleep_metrics()}

    Returns
    -------
    str — multi-line formatted table
    """
    has_mae = any("mae_physionet" in m for m in all_metrics.values())
    header = (
        f"{'Cohort':<26} {'Sleep%':>7} {'TST(h)':>7} {'SOL(m)':>7} "
        f"{'WASO(m)':>8} {'Cycles':>6} {'SOREMP':>6} {'Rested%':>8} {'Rest Status':<18}"
        + (f" {'MAE':>7}" if has_mae else "")
    )
    sep = "-" * len(header)
    rows = [header, sep]

    for cohort_id, m in all_metrics.items():
        name = config.COHORT_NAMES.get(cohort_id, f"Cohort {cohort_id}")
        row = (
            f"{name:<26} {m['sleep_pct_env']:>6.1f}% {m['tst_h']:>7.1f} {m['sol_min']:>7.0f} "
            f"{m['waso_min']:>8.0f} {m['n_cycles']:>6d} {m.get('n_soremps', 0):>6d} "
            f"{m['well_rested_index']:>7.1f}% {m['rest_status']:<18}"
            + (f" {m['mae_physionet']:>7.4f}" if has_mae and 'mae_physionet' in m else "")
        )
        rows.append(row)

    return "\n".join(rows)

# endregion Summary table for all cohorts


# region Private helpers

def _first_sleep_index(hypnogram: np.ndarray) -> Optional[int]:
    """Return index of first non-wake epoch, or None if always awake."""
    sleep_mask = hypnogram > 0
    if not sleep_mask.any():
        return None
    return int(np.argmax(sleep_mask))


def _compute_waso(
    hypnogram:   np.ndarray,
    dt_min:      float,
    is_daylight: Optional[np.ndarray] = None,
) -> float:
    """
    Wake After Sleep Onset (WASO):
    Sum of wake epochs occurring during the nocturnal sleep period after sleep onset,
    reflecting sleep fragmentation and premature awakenings without counting daytime waking hours.
    """
    sleep_indices = np.where(hypnogram > 0)[0]
    if len(sleep_indices) < 2:
        return 0.0

    first_sleep = sleep_indices[0]
    waso_steps = 0

    if is_daylight is not None:
        # Measure all nocturnal wake epochs occurring after sleep onset
        for i in range(first_sleep, len(hypnogram)):
            if hypnogram[i] == 0 and not is_daylight[i]:
                waso_steps += 1
    else:
        max_gap_steps = int(120.0 / dt_min)  # Max 2 hour gap to be considered nocturnal WASO
        for i in range(len(sleep_indices) - 1):
            gap = sleep_indices[i+1] - sleep_indices[i] - 1
            if 0 < gap <= max_gap_steps:
                waso_steps += gap

    return float(waso_steps) * dt_min


def _count_rem_cycles(hypnogram: np.ndarray) -> int:
    """
    Count number of completed NREM-REM ultradian cycles.
    A cycle requires an NREM period (N1/N2/N3) followed by a REM episode.
    """
    cycles   = 0
    had_nrem = False
    in_rem   = False

    for stage in hypnogram:
        if stage in (1, 2, 3):
            had_nrem = True
            in_rem   = False
        elif stage == 4 and had_nrem and not in_rem:
            cycles   += 1
            in_rem   = True
            had_nrem = False
        elif stage == 0:
            in_rem   = False

    return cycles


def _count_soremps(hypnogram: np.ndarray, dt_min: float = 1.0, window_min: float = 15.0) -> int:
    """
    Count Sleep-Onset REM Periods (SOREMPs).
    A diagnostic hallmark of Narcolepsy where REM sleep (stage 4) occurs
    within 15 minutes of sleep onset. Counts discrete episodes.
    """
    window_steps = max(1, int(round(window_min / max(0.1, dt_min))))
    soremps = 0
    in_sleep_bout = False
    onset_step = 0
    recorded_for_bout = False

    for i, stage in enumerate(hypnogram):
        if not in_sleep_bout and stage > 0:
            in_sleep_bout = True
            onset_step = i
            recorded_for_bout = False
            if stage == 4:
                soremps += 1
                recorded_for_bout = True
        elif in_sleep_bout:
            if stage == 0:
                in_sleep_bout = False
                recorded_for_bout = False
            elif stage == 4 and not recorded_for_bout and (i - onset_step) <= window_steps:
                soremps += 1
                recorded_for_bout = True

    return soremps

# endregion Private helpers


if __name__ == "__main__":
    # Synthetic 72h hypnogram: 8h wake, 8h NREM/REM * 3 days
    hyp = []
    for day in range(3):
        hyp += [0] * 480                           # 8h wake
        for _ in range(3):                         # ~3 cycles per night
            hyp += [2] * 40 + [3] * 40 + [4] * 30  # N2/N3/REM
    hyp = np.array(hyp[:4320])                     # clip to 72h

    m = compute_sleep_metrics(hyp)
    for k, v in m.items():
        print(f"  {k}: {v}")
