"""
Borbély Two-Process Model: Process S (homeostatic sleep pressure) and
Process C (circadian drive).

Euler integration at dt = DT_HOURS per simulation step.

Equations
---------
    dS/dt = (1 - S) / τ_r   [during wake]
    dS/dt = -S / τ_d         [during sleep]
    C(t) = A * sin(2π(t - t0) / T) + light_coupling * lux_norm(t)

Sleep propensity
----------------
    propensity = S(t) - C_upper_threshold(t)

    Sleep is triggered when propensity ≥ 0 and the flip-flop switch
    crosses the threshold; wake is triggered when S falls below the
    lower circadian threshold.
"""

from __future__ import annotations

import os
import sys
import math
from typing import Optional
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config


class ProcessSC:
    """
    Continuous-time Borbély Two-Process Model with Euler integration.

    Parameters
    ----------
    tau_rise_h               : Process S buildup time constant (hours) during wake
    tau_decay_h              : Process S decay time constant (hours) during sleep
    circadian_amplitude      : amplitude A of the C(t) sine wave
    circadian_phase_offset_h : horizontal phase shift of C(t) (hours)
                               positive -> delayed (CRSWD), negative -> advanced
    light_sensitivity        : coupling coefficient of lux -> C(t) boost [0, 1]
    initial_s                : starting value of Process S (0--1)
    """

    def __init__(
        self,
        tau_rise_h:               float = config.DEFAULT_TAU_RISE_H,
        tau_decay_h:              float = config.DEFAULT_TAU_DECAY_H,
        circadian_amplitude:      float = config.CIRCADIAN_AMPLITUDE,
        circadian_phase_offset_h: float = 0.0,
        light_sensitivity:        float = 1.0,
        initial_s:                float = 0.3,
    ):
        self.tau_rise_h   = tau_rise_h
        self.tau_decay_h  = tau_decay_h
        self.amplitude    = circadian_amplitude
        self.phase_offset = circadian_phase_offset_h  # hours
        self.light_sens   = light_sensitivity
        self.period_h     = config.CIRCADIAN_PERIOD_H

        # State
        self.S: float = float(initial_s)  # homeostatic pressure [0, 1]
        self._t_h: float = 0.0            # elapsed simulation hours

    def step(self, is_awake: bool, lux: float, dt_min: Optional[float] = None) -> dict:
        """
        Advance Process S & C by one simulation timestep (dt_min or DT_HOURS).

        Parameters
        ----------
        is_awake : True if the organism is currently awake
        lux      : effective lux (already sensitivity-scaled) from environment
        dt_min   : optional timestep in minutes (defaults to DT_MIN)

        Returns
        -------
        dict with keys: S, C, upper_threshold, lower_threshold,
                        sleep_propensity, want_sleep (bool), want_wake (bool)
        """
        dt = (dt_min / 60.0) if dt_min is not None else config.DT_HOURS

        # --- Process S Euler integration ---
        if is_awake:
            dS = (1.0 - self.S) / self.tau_rise_h
        else:
            dS = -self.S / self.tau_decay_h
        self.S = float(min(max(self.S + dS * dt, 0.0), 1.0))

        # --- Process C ---
        C_raw = self._compute_c(self._t_h, lux)

        # Circadian-modulated thresholds
        upper_thresh = 0.67 + 0.25 * C_raw  # sleep onset threshold
        lower_thresh = 0.17 + 0.15 * C_raw  # wake offset threshold

        # Sleep propensity: positive when S wants to push into sleep
        sleep_propensity  = self.S - upper_thresh
        sleep_maintenance = max(0.0, self.S - lower_thresh)
        want_sleep        = sleep_propensity >= 0.0
        want_wake         = (not is_awake) and (self.S <= lower_thresh)

        # Advance internal clock
        self._t_h += dt

        return {
            "S":                 self.S,
            "C":                 C_raw,
            "upper_threshold":   upper_thresh,
            "lower_threshold":   lower_thresh,
            "sleep_propensity":  sleep_propensity,
            "sleep_maintenance": sleep_maintenance,
            "want_sleep":        want_sleep,
            "want_wake":         want_wake,
        }

    def reset(self, initial_s: float = 0.3) -> None:
        """Reset state for a new simulation run."""
        self.S    = float(initial_s)
        self._t_h = 0.0

    @property
    def current_S(self) -> float:
        return self.S

    @property
    def current_t_h(self) -> float:
        return self._t_h

    def _compute_c(self, t_h: float, lux: float) -> float:  # Private helper
        """
        C(t) = A * sin(2π(t - t0) / T) + light_coupling * lux_norm

        t0 = 10.0 hours:
          - Peaks at ~16:00 (Wake Maintenance Zone / late-afternoon alertness)
          - Nadir at ~04:00 (Core body temperature minimum / peak nocturnal sleepiness)
          - Crosses downward at ~22:00 (Lights-off sleep gate opens)
        Phase offset shifts this horizontally (e.g. +4.0h for CRSWD delayed phase).
        """
        t0 = 10.0 + self.phase_offset
        sine_arg = 2.0 * math.pi * (t_h - t0) / self.period_h
        c_sine   = self.amplitude * math.sin(sine_arg)

        # Light coupling: suppresses melatonin-like drive (raises C slightly)
        lux_norm        = min(lux / config.MAX_LUX, 1.0)
        light_component = 0.05 * self.light_sens * lux_norm

        return c_sine + light_component


if __name__ == "__main__":
    proc = ProcessSC(circadian_phase_offset_h=0.0)
    is_awake = True
    for step in range(0, 4320, 60):  # sample every hour
        if proc.S > 0.67:
            is_awake = False
        if proc.S < 0.17:
            is_awake = True
        out = proc.step(is_awake, lux=10000.0 if 6 <= (step // 60 % 24) < 22 else 0.0)
        print(f"  t={proc.current_t_h:5.1f}h  S={out['S']:.3f}  C={out['C']:.3f}  "
              f"awake={is_awake}  propensity={out['sleep_propensity']:+.3f}")
