"""
VLPO <-> LHA Mutually Inhibitory Bistable Switch.

Implements the spiking bistable flip-flop switch that governs sleep/wake
state transitions, inspired by Saper et al. (2010) and the "flip-flop"
model of sleep/wake control.

Two activity variables:
 * vlpo : VLPO (sleep-promoting, GABAergic) activity
 * lha  : LHA (wake-promoting, Orexin/MCH) activity

Each inhibits the other -- producing bistable switching dynamics.
Sleep stage is inferred from VLPO dominance level and elapsed time-in-state.

Cohort-specific modulations
-----------------------------
 * Normal          : balanced gains, threshold ~S=0.67
 * CRSWD           : identical gains, but Process S/C inputs arrive delayed
 * Psych. Insomnia : high arousal_bias on LHA; reduced VLPO→LHA inhibition gain
 * Paradoxical     : normal flip-flop (objective sleep), but cortical module
                     dissociates (handled in corticothalamic_snn.py)
"""

from __future__ import annotations

import os
import sys
import math
from typing import Optional
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config


class FlipFlopSwitch:
    """
    Bistable VLPO-LHA switch with sleep-stage state machine.

    Parameters
    ----------
    vlpo_gain    : inhibitory gain of VLPO→LHA connection [0, 1]
    arousal_bias : tonic wake drive added to LHA at every step [0, 1]
    tau_state    : time constant for activity dynamics (minutes)
    """

    # Thresholds for stage classification (VLPO activity level)
    _STAGE_THRESHOLDS = {
        "wake": (0.00, 0.25),  # VLPO < 0.25 → Wake
        "N1":   (0.25, 0.45),  # N1 transition
        "N2":   (0.45, 0.65),  # stable N2
        "N3":   (0.65, 0.85),  # deep slow-wave sleep
        "REM":  (0.85, 1.00),  # REM (high VLPO, LHA suppressed)
    }

    def __init__(
        self,
        vlpo_gain:        float = 1.0,
        arousal_bias:     float = 0.0,
        tau_state:        float = 10.0,  # minutes for smooth neurochemical transition
        orexin_tone:      float = 1.0,   # Hypocretin tone [0, 1] (0.05 in Narcolepsy)
        soremp_tendency:  float = 0.0,   # Propensity for Sleep-Onset REM Periods
        wake_instability: float = 0.0,   # Propensity for daytime sleep attacks
    ):
        self.vlpo_gain        = float(vlpo_gain)
        self.arousal_bias     = float(arousal_bias)
        self.tau_state        = float(tau_state)
        self.orexin_tone      = float(orexin_tone)
        self.soremp_tendency  = float(soremp_tendency)
        self.wake_instability = float(wake_instability)

        # Mutual inhibition strength (Saper et al. 2005/2010 bistable gain)
        self._g_inh = 1.6

        # Activity states [0, 1] (starts in waking state)
        self.vlpo: float = 0.05  # low (sleep-active GABAergic node)
        self.lha : float = 0.95  # high (wake-active monoaminergic/orexinergic node)

        # Sleep stage & cycle bookkeeping
        self._sleep_stage:               int   = 0    # 0=Wake, 1=N1, 2=N2, 3=N3, 4=REM
        self._prev_stage:                int   = 0
        self._time_in_stage:             float = 0.0  # minutes in current stage
        self._total_time_min:            float = 0.0  # total elapsed simulation minutes
        self._time_in_sleep:             float = 0.0  # continuous minutes asleep in current bout
        self._rem_cycle_timer:           float = 0.0  # minutes in current ultradian NREM-REM cycle
        self.soremp_count:               int   = 0    # Count of discrete SOREMP episodes
        self._soremp_recorded_this_bout: bool  = False

    def step(
        self,
        sleep_propensity:  float,
        lux:               float,
        dt:                Optional[float] = None,
        sleep_maintenance: Optional[float] = None,
    ) -> dict:
        """
        Advance the flip-flop switch by one simulation step using exact exponential
        integration to guarantee numerical stability regardless of timestep dt.

        Parameters
        ----------
        sleep_propensity  : S - upper_threshold (drives sleep initiation)
        lux               : environmental light level (lux)
        dt                : timestep in minutes (defaults to DT_MIN)
        sleep_maintenance : S - lower_threshold (sustains sleep until debt is cleared)
        """
        dt_val = dt if dt is not None else config.DT_MIN
        self._total_time_min += dt_val
        t_h = self._total_time_min / 60.0

        is_dark = (lux < 500.0)
        lux_norm = min(lux / config.MAX_LUX, 1.0)
        maint = sleep_maintenance if sleep_maintenance is not None else max(0.0, sleep_propensity + 0.35)

        # --- Compute input drives with hysteresis ---
        if self._sleep_stage == 0:
            # Wake state:
            # - VLPO is driven by positive sleep pressure exceeding upper threshold + darkness facilitation
            # - LHA wake center is driven by arousal bias and ambient light, suppressed by sleep pressure
            dark_facilitation = 0.40 if (is_dark and sleep_propensity >= -0.02) else 0.0
            vlpo_drive = max(0.0, sleep_propensity) * 3.5 + dark_facilitation
            wake_ambient = 0.05 if is_dark else 0.55
            lha_drive  = (self.arousal_bias + lux_norm * 0.80 + wake_ambient - max(0.0, sleep_propensity) * 2.5) * self.orexin_tone

            # Narcolepsy daytime sleep attack intrusion
            if self.wake_instability > 0.0 and not is_dark:
                # Daytime attacks cluster in afternoon hours (12:00--17:00)
                tod_h = t_h % 24.0
                if 11.5 <= tod_h <= 17.5:
                    osc = math.sin(t_h * math.pi)
                    if osc > 0.55:
                        vlpo_drive += 0.55 * self.wake_instability
                        lha_drive  *= 0.20
        else:
            # Sleep state (active hysteresis):
            # - VLPO is sustained by remaining sleep debt (S - lower_threshold) plus dark facilitation
            # - LHA remains suppressed during darkness and sleep maintenance
            vlpo_drive = max(0.0, maint) * 2.5 + (0.40 if is_dark else 0.05)
            lha_drive  = (self.arousal_bias + lux_norm * 1.0 + (0.35 if not is_dark else 0.02) - max(0.0, maint) * 1.0) * self.orexin_tone

            # Narcolepsy nocturnal sleep fragmentation (micro-arousals)
            if self.orexin_tone < 0.20:
                lha_drive += 0.18

        # --- Mutual inhibition between VLPO and LHA ---
        vlpo_inh = self.vlpo_gain * self._g_inh * self.vlpo
        lha_inh  = self._g_inh * self.lha

        target_vlpo = self._sigmoid(vlpo_drive - lha_inh)
        target_lha  = self._sigmoid(lha_drive  - vlpo_inh)

        # --- Exact Exponential Integration ---
        # Unconditionally stable for any dt_val > 0; mathematically immune to Euler explosion
        alpha = math.exp(-dt_val / self.tau_state)
        self.vlpo = float(min(max(target_vlpo + (self.vlpo - target_vlpo) * alpha, 0.0), 1.0))
        self.lha  = float(min(max(target_lha  + (self.lha  - target_lha)  * alpha, 0.0), 1.0))

        # --- Ultradian Sleep Stage Progression ---
        prev_stage = self._sleep_stage
        self._prev_stage = prev_stage
        is_asleep = (self.vlpo >= 0.45 and self.lha < 0.55)

        if is_asleep:
            if prev_stage == 0:
                # Sleep onset transition
                self._time_in_sleep = 0.0
                self._rem_cycle_timer = 0.0
                if self.soremp_tendency > 0.5:
                    # Narcolepsy Sleep-Onset REM Period (SOREMP)
                    self._sleep_stage = 4
                    if not self._soremp_recorded_this_bout:
                        self.soremp_count += 1
                        self._soremp_recorded_this_bout = True
                else:
                    self._sleep_stage = 1  # Standard N1 drowsiness
            else:
                self._time_in_sleep += dt_val
                self._rem_cycle_timer += dt_val
                cycle_pos = self._rem_cycle_timer % 90.0  # 90-minute ultradian cycle

                if self.soremp_tendency > 0.5 and self._time_in_sleep <= 25.0:
                    self._sleep_stage = 4  # SOREMP duration ~20-25 min
                elif cycle_pos < 10.0:
                    self._sleep_stage = 1  # N1 transition
                elif cycle_pos < 45.0:
                    self._sleep_stage = 2  # N2 light sleep
                elif cycle_pos < 70.0:
                    # N3 deep slow-wave sleep (favored early in night when sleep debt is high)
                    self._sleep_stage = 3 if maint > 0.15 else 2
                else:
                    self._sleep_stage = 4  # REM dreaming sleep
        else:
            self._sleep_stage = 0  # Wake
            self._time_in_sleep = 0.0
            self._soremp_recorded_this_bout = False

        if self._sleep_stage != prev_stage:
            self._time_in_stage = 0.0
        else:
            self._time_in_stage += dt_val

        return {
            "sleep_stage":     self._sleep_stage,
            "vlpo":            self.vlpo,
            "lha":             self.lha,
            "is_awake":        self._sleep_stage == 0,
            "time_in_stage_m": self._time_in_stage,
            "soremp_count":    self.soremp_count,
        }

    def reset(self) -> None:
        self.vlpo             = 0.05
        self.lha              = 0.95
        self._sleep_stage     = 0
        self._prev_stage      = 0
        self._time_in_stage   = 0.0
        self._total_time_min  = 0.0
        self._time_in_sleep   = 0.0
        self._rem_cycle_timer = 0.0
        self.soremp_count     = 0
        self._soremp_recorded_this_bout = False

    @property
    def sleep_stage(self) -> int:
        return self._sleep_stage

    @property
    def is_awake(self) -> bool:
        return self._sleep_stage == 0

    @staticmethod
    def _sigmoid(x: float, gain: float = 3.5) -> float:  # Private helper
        """Bounded sigmoid activation function."""
        clipped_x = max(-15.0, min(15.0, x))
        return 1.0 / (1.0 + math.exp(-gain * clipped_x))


if __name__ == "__main__":
    from models.process_sc import ProcessSC
    proc  = ProcessSC()
    ff    = FlipFlopSwitch(vlpo_gain=1.0, arousal_bias=0.0)
    ff_pi = FlipFlopSwitch(vlpo_gain=0.45, arousal_bias=0.6)  # Insomnia

    awake = True
    for step in range(4320):
        t_min = step * config.DT_MIN
        lux = 10000.0 if 6 <= (t_min / 60 % 24) < 22 else 0.0
        proc_out = proc.step(awake, lux)
        ff_out   = ff.step(proc_out["sleep_propensity"], lux)
        awake = ff_out["is_awake"]
        if step % 120 == 0:
            print(f"  t={t_min/60:.1f}h  stage={ff_out['sleep_stage']}  "
                  f"vlpo={ff_out['vlpo']:.3f}  lha={ff_out['lha']:.3f}  "
                  f"S={proc_out['S']:.3f}")
