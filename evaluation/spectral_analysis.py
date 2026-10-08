"""
Welch PSD-based EEG/LFP band-power analysis and subjective wakefulness
computation.

Used both for:
  1. Analysing PhysioNet reference data (physionet_fetcher.py calls this)
  2. Analysing SNN-generated LFP at each simulation timestep
"""

from __future__ import annotations

import os
import sys
import numpy as np
from typing import Dict, Tuple
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config


# region Core band-power computation

def compute_band_powers(
    lfp_signal: np.ndarray,
    fs:         float = config.WELCH_FS,
    nperseg:    int   = config.WELCH_NPERSEG,
) -> Dict[str, float]:
    """
    Compute relative band powers from a 1-D LFP/EEG signal using Welch PSD.

    Parameters
    ----------
    lfp_signal : 1-D float array (samples)
    fs         : sampling frequency (Hz)
    nperseg    : Welch window length (samples)

    Returns
    -------
    dict  {band_name -> relative_power}   values in [0, 1], sum ≤ 1
    """
    from scipy.signal import welch

    n = len(lfp_signal)
    if n < 4:
        # Signal too short --- return flat spectrum
        return {b: 1.0 / len(config.BAND_FREQS) for b in config.BAND_FREQS}

    _nperseg = min(nperseg, n)
    freqs, psd = welch(lfp_signal, fs=fs, nperseg=_nperseg)

    # Broadband normaliser (0.5 -- 30 Hz)
    total = _integrate_band(freqs, psd, (0.5, 30.0))
    if total < 1e-30:
        total = 1.0

    result: Dict[str, float] = {}
    for band_name, (f_lo, f_hi) in config.BAND_FREQS.items():
        result[band_name] = _integrate_band(freqs, psd, (f_lo, f_hi)) / total

    return result


def compute_band_powers_from_psd(
    freqs: np.ndarray,
    psd:   np.ndarray,
) -> Dict[str, float]:
    """
    Compute relative band powers directly from pre-computed PSD arrays.
    Useful when the PSD has already been computed upstream.
    """
    total = _integrate_band(freqs, psd, (0.5, 30.0))
    if total < 1e-30:
        total = 1.0
    return {
        band: _integrate_band(freqs, psd, (f_lo, f_hi)) / total
        for band, (f_lo, f_hi) in config.BAND_FREQS.items()
    }

# endregion Core band-power computation


# region Synthetic LFP generation for SNN output

def synthesize_lfp_from_spikes(
    spike_matrix:       np.ndarray,
    sleep_stage:        int,
    fs:                 float = config.WELCH_FS,
    cortical_inh_scale: float = 1.0,
    target_samples:     int = 256,
) -> np.ndarray:
    """
    Convert a binary spike matrix [T × N_neurons] into a synthetic LFP
    signal with stage-appropriate spectral characteristics.

    Strategy
    --------
    • Base LFP = low-pass filtered population firing rate interpolated to 256 samples (1 sec)
    • Stage-specific oscillatory components are added via band-limited noise
      to realistically mimic thalamocortical rhythms
    • Paradoxical insomnia: cortical_inh_scale < 1 -> amplified beta component

    Parameters
    ----------
    spike_matrix       : shape [T, N] binary spike tensor (numpy)
    sleep_stage        : 0=Wake, 1=N1, 2=N2, 3=N3, 4=REM
    fs                 : sampling frequency (Hz)
    cortical_inh_scale : scales inhibition -> controls beta amplitude
    target_samples     : number of samples to synthesize (defaults to 256 = 1s at 256 Hz)

    Returns
    -------
    lfp : 1-D float array of length target_samples
    """
    T, N = spike_matrix.shape
    pop_rate_raw = spike_matrix.mean(axis=1).astype(float) if T > 0 else np.zeros(1)

    # Interpolate population firing rate to target_samples for robust Welch spectral resolution
    x_orig = np.linspace(0, 1, max(T, 2))
    x_target = np.linspace(0, 1, target_samples)
    pop_rate = np.interp(x_target, x_orig, pop_rate_raw if T >= 2 else np.repeat(pop_rate_raw, 2))

    # Smooth with a short Gaussian kernel
    from scipy.ndimage import gaussian_filter1d
    lfp = gaussian_filter1d(pop_rate, sigma=max(1, int(fs * 0.01)))

    # Add stage-specific oscillatory components
    beta_scale = (2.0 - cortical_inh_scale)  # > 1 when inh reduced

    if sleep_stage == 0:   # Wake: broadband, dominant alpha/beta
        lfp += _band_noise(target_samples, fs,  15, 30, amp=0.20 * beta_scale)
        lfp += _band_noise(target_samples, fs,   8, 13, amp=0.15)
        lfp += _band_noise(target_samples, fs,   4,  8, amp=0.06)
        lfp += _band_noise(target_samples, fs, 0.5,  4, amp=0.03)

    elif sleep_stage == 1: # N1: alpha fading, theta emerging
        lfp += _band_noise(target_samples, fs,   4,  8, amp=0.22)
        lfp += _band_noise(target_samples, fs,   8, 13, amp=0.10)
        lfp += _band_noise(target_samples, fs,  15, 30, amp=0.05 * beta_scale)
        lfp += _band_noise(target_samples, fs, 0.5,  4, amp=0.04)

    elif sleep_stage == 2: # N2: sigma spindles (12-16 Hz) + moderate delta
        lfp += _band_noise(target_samples, fs,  12, 16, amp=0.30)  # Sleep spindles
        lfp += _band_noise(target_samples, fs, 0.5,  4, amp=0.20)  # Delta
        lfp += _band_noise(target_samples, fs,   4,  8, amp=0.10)  # Theta
        lfp += _band_noise(target_samples, fs,  15, 30, amp=0.04 * beta_scale)

    elif sleep_stage == 3: # N3: dominant slow-wave delta
        lfp += _band_noise(target_samples, fs, 0.5,  4, amp=0.55)  # High-amplitude slow delta
        lfp += _band_noise(target_samples, fs, 12,  16, amp=0.06)  # Residual spindles
        lfp += _band_noise(target_samples, fs,  4,   8, amp=0.08)  # Theta
        lfp += _band_noise(target_samples, fs, 15,  30, amp=0.04 * beta_scale)

    elif sleep_stage == 4: # REM: theta dominance + desynchronized EEG
        lfp += _band_noise(target_samples, fs,  4,  8,  amp=0.35)  # Hippocampal/cortical theta
        lfp += _band_noise(target_samples, fs,  8, 13,  amp=0.08)  # Low alpha
        lfp += _band_noise(target_samples, fs,  0.5, 4, amp=0.05)  # Low delta
        lfp += _band_noise(target_samples, fs, 15,  30, amp=0.08 * beta_scale)

    return lfp

# endregion Synthetic LFP generation for SNN output


# region Subjective wakefulness meter

def get_subjective_wakefulness(
    sleep_stage:       int,
    cortical_beta:     float,
    perception_offset: float = 0.0,
) -> float:
    """
    Compute subjective wakefulness (0--100%) from objective sleep stage.

    Normal mapping
    --------------
    Wake=100, N1=75, N2=40, N3=10, REM=25

    Paradoxical Insomnia
    --------------------
    perception_offset shifts the score upward regardless of stage,
    causing the patient to report feeling awake despite objective NREM.

    Parameters
    ----------
    sleep_stage       : objective stage from flip-flop (0--4)
    cortical_beta     : relative beta power from LFP analysis
    perception_offset : additive offset (large for paradoxical insomnia)

    Returns
    -------
    float in [0, 100]
    """
    base_map = {0: 100.0, 1: 72.0, 2: 38.0, 3: 8.0, 4: 22.0}
    base = base_map.get(sleep_stage, 50.0)

    # Beta power modulation: high cortical beta -> feels more awake
    beta_bonus = cortical_beta * 40.0

    subjective = base * 0.7 + beta_bonus * 0.3 + perception_offset
    return float(min(max(subjective, 0.0), 100.0))

# endregion Subjective wakefulness meter


# region Private helpers

def _integrate_band(freqs: np.ndarray, psd: np.ndarray, band: Tuple) -> float:
    f_lo, f_hi = band
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    if mask.sum() >= 2:
        return float(np.trapz(psd[mask], freqs[mask]))
    elif mask.sum() == 1:
        return float(psd[mask][0] * (f_hi - f_lo))
    return 0.0


def _band_noise(n_samples: int, fs: float, f_lo: float, f_hi: float, amp: float) -> np.ndarray:
    """Generate band-limited Gaussian noise via FFT zero-ing."""
    noise     = np.random.randn(n_samples)
    fft       = np.fft.rfft(noise)
    freqs     = np.fft.rfftfreq(n_samples, d=1.0 / fs)
    mask      = (freqs < f_lo) | (freqs > f_hi)
    fft[mask] = 0.0
    return np.fft.irfft(fft, n=n_samples) * amp

# endregion Private helpers


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    fake_spikes = (rng.random((256, 50)) > 0.9).astype(float)
    lfp = synthesize_lfp_from_spikes(fake_spikes, sleep_stage=3, cortical_inh_scale=0.25)
    bands = compute_band_powers(lfp, fs=256.0)
    print("Band powers (N3, paradoxical):")
    for b, v in bands.items():
        print(f"  {b:>6s}: {v:.4f}")
