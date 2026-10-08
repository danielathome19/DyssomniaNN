"""
24-hour day/night light exposure and zeitgeber clock.

Each cohort gets its own CircadianEnvironment instance, allowing
independent photoperiod override (e.g., CRSWD delayed phase).
"""

from __future__ import annotations

import os
import sys
import math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config


class CircadianEnvironment:
    """
    Models a 24-hour light-dark (LD) cycle and provides lux & zeitgeber
    phase angle at each simulation timestep.

    Parameters
    ----------
    light_on_hour     : hour of day when lights turn on  (default 6)
    light_off_hour    : hour of day when lights turn off (default 22)
    phase_offset_h    : constant phase offset applied to the sinusoidal
                        circadian drive (positive = delayed, negative = advanced)
    light_sensitivity : scales the effective lux reaching the SCN (0--1)
    """

    def __init__(
        self,
        light_on_hour:     float = config.DEFAULT_LIGHT_ON_HOUR,
        light_off_hour:    float = config.DEFAULT_LIGHT_OFF_HOUR,
        phase_offset_h:    float = 0.0,
        light_sensitivity: float = 1.0,
    ):
        self.light_on_hour     = light_on_hour
        self.light_off_hour    = light_off_hour
        self.phase_offset_h    = phase_offset_h
        self.light_sensitivity = float(light_sensitivity)

    def step(self, t_min: float) -> dict:
        """
        Compute environment state at simulation minute `t_min`.

        Parameters
        ----------
        t_min : elapsed simulation time in minutes (0 → N_STEPS * DT_MIN)

        Returns
        -------
        dict with keys:
            lux             : effective lux reaching SCN
            raw_lux         : raw environmental lux (ignores sensitivity)
            time_of_day_h   : hour-of-day modulo 24 (0--24)
            zeitgeber_phase : radians, 0 at midnight
            is_light        : bool
        """
        t_h           = t_min / 60.0
        time_of_day_h = t_h % 24.0

        raw_lux = self._light_level(time_of_day_h)
        eff_lux = raw_lux * self.light_sensitivity

        # Zeitgeber phase: 0 at midnight, π at noon
        phase_rad = 2.0 * math.pi * (time_of_day_h / 24.0)

        return {
            "lux":             eff_lux,
            "raw_lux":         raw_lux,
            "time_of_day_h":   time_of_day_h,
            "zeitgeber_phase": phase_rad,
            "is_light":        raw_lux > 0.0,
        }

    def is_light_at(self, t_min: float) -> bool:
        """Convenience: returns True if lights are on at minute t_min."""
        time_of_day_h = (t_min / 60.0) % 24.0
        return self._light_level(time_of_day_h) > 0.0

    def get_phase_offset(self) -> float:
        """Return the cohort-specific circadian phase offset in hours."""
        return self.phase_offset_h

    def _light_level(self, time_of_day_h: float) -> float:  # Private helper
        """Return environmental lux (pre-sensitivity) for a given hour."""
        on  = self.light_on_hour
        off = self.light_off_hour

        if on < off:
            # Normal case: lights on 06:00--22:00
            if on <= time_of_day_h < off:
                return config.MAX_LUX
        else:
            # Wrap-around case: lights on e.g. 22:00--06:00
            if time_of_day_h >= on or time_of_day_h < off:
                return config.MAX_LUX
        return config.DARK_LUX


if __name__ == "__main__":
    env_normal = CircadianEnvironment()
    env_crswd  = CircadianEnvironment(phase_offset_h=4.0, light_sensitivity=0.4)

    for h in [0, 6, 12, 18, 22, 23]:
        state = env_normal.step(h * 60)
        print(f"  Hour {h:02d}:00 | lux={state['lux']:.0f} | phase={state['zeitgeber_phase']:.2f} rad")
