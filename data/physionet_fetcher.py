"""
Automatic MNE fetcher for PhysioNet Sleep-EDF baseline data.

Downloads SC4001E0-PSG.edf (Subject 0, Recording 1) via MNE's built-in
fetcher, loads the Fpz-Cz EEG channel, and computes reference relative
band powers via Welch PSD.

Falls back to config.PHYSIONET_FALLBACK_BANDS if download or MNE fails.
"""

import os
import sys
import warnings
import numpy as np
from typing import Dict
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import config


def fetch_sleep_edf() -> Dict[str, float]:
    """
    Fetch PhysioNet Sleep-EDF Subject 0 Recording 1 and return relative
    band powers for the EEG Fpz-Cz channel.

    Returns
    -------
    dict  with keys: 'delta', 'theta', 'alpha', 'sigma', 'beta'
          values are floats in [0, 1] summing to <= 1.0
    """
    try:
        return _fetch_via_mne()
    except Exception as exc:
        warnings.warn(
            f"[PhysioNet] MNE fetch failed ({exc}). "
            "Using hardcoded fallback band powers from literature.",
            RuntimeWarning,
            stacklevel=2,
        )
        return dict(config.PHYSIONET_FALLBACK_BANDS)


# region Internal Helpers

def _fetch_via_mne() -> Dict[str, float]:
    import mne
    from mne.datasets.sleep_physionet.age import fetch_data

    print("[PhysioNet] Fetching Sleep-EDF Subject 0 Recording 1 via MNE ...")
    files = fetch_data(subjects=[0], recording=[1], verbose=False)
    psg_path = files[0][0]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = mne.io.read_raw_edf(psg_path, preload=True, verbose=False)

    # Pick the Fpz-Cz EEG channel (standard in Sleep-EDF)
    eeg_channels = [ch for ch in raw.ch_names if "EEG" in ch.upper() or "Fpz" in ch]
    if not eeg_channels:
        eeg_channels = [raw.ch_names[0]]

    raw.pick_channels([eeg_channels[0]], verbose=False)
    data = raw.get_data()[0]  # shape: (n_samples,)
    fs   = raw.info["sfreq"]  # sampling frequency

    bands = _welch_band_powers(data, fs)
    print(f"[PhysioNet] Reference band powers from {os.path.basename(psg_path)}:")
    for name, val in bands.items():
        print(f"   {name:>6s}: {val:.4f}")
    return bands


def _welch_band_powers(signal: np.ndarray, fs: float) -> Dict[str, float]:
    """Compute relative Welch PSD band powers from a 1-D EEG/LFP signal."""
    from scipy.signal import welch

    nperseg = min(int(fs * 4), len(signal))  # 4-second windows
    freqs, psd = welch(signal, fs=fs, nperseg=nperseg)

    total_power = _band_power(freqs, psd, (0.5, 30.0))
    if total_power < 1e-30:
        total_power = 1.0  # avoid division by zero on silent signals

    raw_powers: Dict[str, float] = {}
    for band_name, (f_lo, f_hi) in config.BAND_FREQS.items():
        raw_powers[band_name] = _band_power(freqs, psd, (f_lo, f_hi)) / total_power

    return raw_powers


def _band_power(freqs: np.ndarray, psd: np.ndarray, band: tuple) -> float:
    """Integrate PSD between band[0] and band[1] Hz via the trapezoidal rule."""
    f_lo, f_hi = band
    mask = (freqs >= f_lo) & (freqs <= f_hi)
    if mask.sum() < 2:
        return 0.0
    return float(np.trapz(psd[mask], freqs[mask]))

# endregion Internal Helpers


if __name__ == "__main__":
    bands = fetch_sleep_edf()
    print("\nFinal reference bands:", bands)
