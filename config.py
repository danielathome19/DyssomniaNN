"""
Global constants and simulation configuration for the Multi-Cohort SNN
Sleep Disorder Simulator.

All time-domain values are in MINUTES unless explicitly labelled _H (hours).
"""

import os


# Simulation timeline
SIM_DURATION_HOURS: int   = 72           # Total simulated time
DT_MIN:             float = 1.0          # Time step in minutes
N_STEPS:            int   = int(SIM_DURATION_HOURS * 60/DT_MIN)  # 4320
DT_HOURS:           float = DT_MIN/60.0  # 1/60 h per step
SNN_INTERNAL_STEPS: int   = 50           # SNN forward-pass substeps per sim-minute
SNN_DT:             float = 1e-3         # SNN membrane time constant (seconds)

# Cohort identifiers
N_COHORTS: int = 5
COHORT_NAMES = {
    0: "Normal",
    1: "CRSWD (Delayed Phase)",
    2: "Psychophysiological Insomnia",
    3: "Paradoxical Insomnia",
    4: "Narcolepsy (Type 1)",
}

# Matplotlib colors, one per cohort
COHORT_COLORS = {
    0: "#4FC3F7",  # sky-blue
    1: "#FFB74D",  # amber
    2: "#EF5350",  # red
    3: "#AB47BC",  # purple
    4: "#00E676",  # emerald green
}

# Sleep-stage encoding
SLEEP_STAGES = {
    0: "Wake",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "REM",
}

# EEG frequency bands  (Hz)
BAND_FREQS = {
    "delta":  (0.5,  4.0),
    "theta":  (4.0,  8.0),
    "alpha":  (8.0, 13.0),
    "sigma":  (12.0, 15.0),
    "beta":   (15.0, 30.0),
}

# Welch PSD parameters (applied to SNN-generated LFP)
WELCH_FS:      float = 256.0           # Effective sampling rate of LFP (Hz)
WELCH_NPERSEG: int   = 256             # Segment length

# Circadian/photoperiod defaults (override per cohort)
DEFAULT_LIGHT_ON_HOUR:  int  = 6       # 06:00
DEFAULT_LIGHT_OFF_HOUR: int = 22       # 22:00
MAX_LUX:  float = 10_000.0
DARK_LUX: float = 0.0

# Process S & C defaults (overridden by cohort params)
DEFAULT_TAU_RISE_H:  float = 18.2      # Wake buildup time constant (hours)
DEFAULT_TAU_DECAY_H: float = 4.2       # Sleep decay time constant (hours)
CIRCADIAN_AMPLITUDE: float = 0.4       # Amplitude of C(t) sine oscillation
CIRCADIAN_PERIOD_H:  float = 24.0      # Period (hours)

# Sleep/Wake threshold logic
SLEEP_ONSET_THRESHOLD: float = 0.60    # S exceeds this → flip-flop triggered
WAKE_OFFSET_THRESHOLD: float = 0.20    # S falls below this → wake flip triggered

# SNN architecture
SNN_N_THALAMIC:   int = 20
SNN_N_CORTICAL_E: int = 20
SNN_N_CORTICAL_I: int = 10
SNN_N_TOTAL:      int = SNN_N_THALAMIC + SNN_N_CORTICAL_E + SNN_N_CORTICAL_I

# TensorBoard
TENSORBOARD_LOG_DIR:       str = "./runs/sleep_simulation"
IMAGE_LOG_INTERVAL:        int = 1440  # Every 1440 steps = 24 simulated hours
DASHBOARD_UPDATE_INTERVAL: int = 10    # Update live dashboard every N steps

# Output
OUTPUT_DIR:          str = "./outputs"
SUMMARY_FIGURE_PATH: str = os.path.join(OUTPUT_DIR, "summary_72h_cohorts.png")

# Fallback empirical PhysioNet band powers (used if MNE download fails)
# Derived from Rechtschaffen & Kales Sleep-EDF literature averages
PHYSIONET_FALLBACK_BANDS = {
    "delta": 0.42,
    "theta": 0.18,
    "alpha": 0.12,
    "sigma": 0.10,
    "beta":  0.08,
}
