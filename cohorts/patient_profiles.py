"""
Parameter configurations for the 4 experimental cohorts.

Each profile is a CohortParams dataclass. Call get_cohort_params(cohort_id)
to retrieve the configuration for a given cohort.
"""

from __future__ import annotations

import os
import sys
from typing import Dict
from dataclasses import dataclass, field
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@dataclass
class CohortParams:
    """Full parameter specification for one simulated patient cohort."""

    # --- Identity ---
    cohort_id: int
    name:      str

    # --- Process S ---
    tau_rise_h:  float                 # Wake buildup time constant (hours)
    tau_decay_h: float                 # Sleep decay time constant (hours)
    initial_s:   float                 # Initial homeostatic pressure [0, 1]

    # --- Process C/Circadian ---
    circadian_phase_offset_h: float    # Phase shift hours (+= delayed, -= advanced)
    light_sensitivity:        float    # SCN light-entrainment gain [0, 1]

    # --- Flip-Flop Switch (VLPO ↔ LHA) ---
    arousal_bias: float                # Tonic LHA hyperarousal drive [0, 1] (0 = normal)
    vlpo_gain:    float                # GABAergic VLPO->LHA inhibition scaling [0, 1]

    # --- SNN/Corticothalamic ---
    cortical_inhibition_scale: float   # Cortical I-neuron gain [0, 1]
                                       # < 1 -> paradoxical elevated Beta

    # --- Photoperiod ---
    light_on_hour:  int                # Hour lights turn on
    light_off_hour: int                # Hour lights turn off

    # --- Subjective perception override ---
    perception_offset: float = 0.0     # Added to subjective wakefulness (0 = none)
                                       # Paradoxical insomnia: +60--80 (feels awake despite sleep)

    # --- Narcolepsy/Orexinergic Neuromodulation ---
    orexin_tone:      float = 1.0      # Hypocretin/Orexin tone in LHA (1.0 = normal, 0.05 = severe loss)
    soremp_tendency:  float = 0.0      # Propensity for Sleep-Onset REM Periods (0.0 = normal, 0.85 = narcolepsy)
    wake_instability: float = 0.0      # Daytime microsleep/sleep attack probability (0.0 = stable, 0.8 = narcolepsy)


# region Cohort Definitions

_COHORT_REGISTRY: Dict[int, CohortParams] = {

    # =========================================================================
    # Cohort 0: Control/Normal
    # =========================================================================
    0: CohortParams(
        cohort_id   = 0,
        name        = "Normal",
        # Process S
        tau_rise_h  = 18.2,
        tau_decay_h = 4.2,
        initial_s   = 0.65,
        # Circadian
        circadian_phase_offset_h = 0.0,
        light_sensitivity        = 1.0,
        # Flip-Flop
        arousal_bias = 0.0,
        vlpo_gain    = 1.0,
        # SNN
        cortical_inhibition_scale = 1.0,
        # Photoperiod
        light_on_hour  = 6,
        light_off_hour = 22,
        # Perception
        perception_offset = 0.0,
        orexin_tone      = 1.0,
        soremp_tendency  = 0.0,
        wake_instability = 0.0,
    ),

    # =========================================================================
    # Cohort 1: Chronic Sleep-Wake Phase Disorder (CRSWD/Delayed Phase)
    #  * Circadian peak delayed by +4 hours relative to environmental light
    #  * SCN light-entrainment sensitivity reduced by 60%
    # =========================================================================
    1: CohortParams(
        cohort_id   = 1,
        name        = "CRSWD (Delayed Phase)",
        # Process S -- same intrinsic kinetics
        tau_rise_h  = 18.2,
        tau_decay_h = 4.2,
        initial_s   = 0.65,
        # Circadian -- CRSWD modifications
        circadian_phase_offset_h = +4.0,  # delayed by 4 hours
        light_sensitivity        =  0.4,  # 60% reduced entrainment
        # Flip-Flop -- normal switch dynamics
        arousal_bias = 0.0,
        vlpo_gain    = 1.0,
        # SNN -- normal cortical dynamics
        cortical_inhibition_scale = 1.0,
        # Photoperiod -- same light schedule, but SCN sees it as if 4h later
        light_on_hour  = 6,
        light_off_hour = 22,
        # Perception
        perception_offset = 0.0,
        orexin_tone      = 1.0,
        soremp_tendency  = 0.0,
        wake_instability = 0.0,
    ),

    # =========================================================================
    # Cohort 2: Chronic Psychophysiological Insomnia
    #  * Hyperarousal: noradrenergic/corticotropin bias to LHA wake node
    #  * Reduced VLPO GABAergic inhibition gain
    #  * Flip-flop resists switching to sleep even when S > 0.8
    # =========================================================================
    2: CohortParams(
        cohort_id   = 2,
        name        = "Psychophysiological Insomnia",
        # Process S -- same kinetics (pressure builds normally)
        tau_rise_h  = 18.2,
        tau_decay_h = 4.2,
        initial_s   = 0.65,
        # Circadian -- normal phase
        circadian_phase_offset_h = 0.0,
        light_sensitivity        = 1.0,
        # Flip-Flop -- insomnia modifications
        arousal_bias = 0.60,   # strong tonic LHA hyperarousal drive
        vlpo_gain    = 0.45,   # reduced GABAergic VLPO->LHA inhibition
        # SNN -- normal cortical dynamics
        cortical_inhibition_scale = 1.0,
        # Photoperiod
        light_on_hour  = 6,
        light_off_hour = 22,
        # Perception
        perception_offset = 15.0,  # slight overestimation of wakefulness
        orexin_tone      = 1.0,
        soremp_tendency  = 0.0,
        wake_instability = 0.0,
    ),

    # =========================================================================
    # Cohort 3: Paradoxical Insomnia (Sleep State Misperception)
    #  * Thalamic SNN executes NREM slow-waves (high Delta/Sigma)
    #  * BUT cortical inhibitory interneurons reduced -> elevated Beta/Gamma
    #  * Patient self-reports "100% awake" despite objective N2/N3 hypnogram
    # =========================================================================
    3: CohortParams(
        cohort_id   = 3,
        name        = "Paradoxical Insomnia",
        # Process S -- same kinetics
        tau_rise_h  = 18.2,
        tau_decay_h = 4.2,
        initial_s   = 0.65,
        # Circadian -- normal phase
        circadian_phase_offset_h = 0.0,
        light_sensitivity        = 1.0,
        # Flip-Flop -- normal objective switch
        arousal_bias = 0.0,
        vlpo_gain    = 1.0,
        # SNN -- paradoxical modification
        cortical_inhibition_scale = 0.25,  # drastically reduced I-neuron gain
                                           # -> cortex stays Beta-active during NREM
        # Photoperiod
        light_on_hour  = 6,
        light_off_hour = 22,
        # Perception: patient always reports being awake
        perception_offset = 75.0,
        orexin_tone       = 1.0,
        soremp_tendency   = 0.0,
        wake_instability  = 0.0,
    ),

    # =========================================================================
    # Cohort 4: Narcolepsy (Type 1/Hypocretin-Orexin Deficiency)
    #  * Autoimmune destruction (>90%) of orexin/hypocretin neurons in LHA
    #  * Severe state boundary instability: Daytime sleep attacks & microsleeps
    #  * SOREMPs: Direct intrusions of REM sleep at sleep onset
    #  * Nocturnal sleep fragmentation with frequent micro-arousals
    # =========================================================================
    4: CohortParams(
        cohort_id   = 4,
        name        = "Narcolepsy (Type 1)",
        # Process S -- normal buildup and decay kinetics
        tau_rise_h  = 18.2,
        tau_decay_h = 4.2,
        initial_s   = 0.65,
        # Circadian -- normal intrinsic pacemaker
        circadian_phase_offset_h = 0.0,
        light_sensitivity        = 0.85,
        # Flip-Flop -- depleted tonic wake stabilization
        arousal_bias = -0.30,  # diminished baseline orexinergic drive
        vlpo_gain    = 1.15,   # VLPO sleep-inducing drive easily triggers switch
        # SNN
        cortical_inhibition_scale = 1.0,
        # Photoperiod
        light_on_hour  = 6,
        light_off_hour = 22,
        # Perception
        perception_offset = 0.0,
        # Narcolepsy-specific biophysics
        orexin_tone      = 0.05,  # 95% orexin depletion
        soremp_tendency  = 0.85,  # 85% probability of Sleep-Onset REM Periods
        wake_instability = 0.75,  # daytime microsleeps/sleep attacks
    ),
}

# endregion Cohort Definitions


# region Public API

def get_cohort_params(cohort_id: int) -> CohortParams:
    """
    Return the CohortParams dataclass for the given cohort ID.

    Parameters
    ----------
    cohort_id : 0 = Normal, 1 = CRSWD, 2 = Psych. Insomnia, 3 = Paradoxical, 4 = Narcolepsy

    Raises
    ------
    KeyError if cohort_id is not in {0, 1, 2, 3, 4}
    """
    if cohort_id not in _COHORT_REGISTRY:
        raise KeyError(f"Unknown cohort_id={cohort_id}. Valid IDs: {list(_COHORT_REGISTRY)}")
    return _COHORT_REGISTRY[cohort_id]


def get_all_cohorts() -> list[CohortParams]:
    """Return all cohort params in order 0--4."""
    return [_COHORT_REGISTRY[i] for i in sorted(_COHORT_REGISTRY)]

# endregion Public API


if __name__ == "__main__":
    for c in get_all_cohorts():
        print(f"\nCohort {c.cohort_id}: {c.name}")
        print(f"  tau_rise={c.tau_rise_h}h  tau_decay={c.tau_decay_h}h")
        print(f"  phase_offset={c.circadian_phase_offset_h:+.1f}h  light_sens={c.light_sensitivity:.2f}")
        print(f"  arousal_bias={c.arousal_bias:.2f}  vlpo_gain={c.vlpo_gain:.2f}")
        print(f"  cortical_inh_scale={c.cortical_inhibition_scale:.2f}  perception_offset={c.perception_offset:.1f}")
        print(f"  orexin_tone={c.orexin_tone:.2f}  soremp={c.soremp_tendency:.2f}  instability={c.wake_instability:.2f}")

