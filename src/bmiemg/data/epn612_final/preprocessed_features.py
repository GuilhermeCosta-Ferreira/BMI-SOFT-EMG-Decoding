"""Validate eight-channel recordings and apply 50 Hz notch then 20 Hz high-pass."""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from bmiemg.data.epn612_final.trees_of_trees import EPN612_CHANNELS
from bmiemg.preprocessing import (
    TIME_FEATURE_FUNCTIONS,
    FREQ_FEATURE_FUNCTIONS,
    notch_filter,
)

EXPECTED_SFREQ = 200.0
FEATURE_TYPES = tuple(
    fn.__name__ for fn in [*TIME_FEATURE_FUNCTIONS, *FREQ_FEATURE_FUNCTIONS]
)


def _validate_sfreq(sfreq):
    fs = float(sfreq)
    if not np.isfinite(fs) or fs != EXPECTED_SFREQ:
        raise ValueError(f"EPN612 requires fs=200 Hz; metadata supplied {fs!r} Hz")
    return fs


def _metadata_sfreq(recording: dict) -> float:
    try:
        return _validate_sfreq(recording["generalInfo"]["samplingFrequencyInHertz"])
    except KeyError as error:
        raise ValueError(
            "Missing EPN612 generalInfo.samplingFrequencyInHertz"
        ) from error


def _stack_emg(sample: dict) -> np.ndarray:
    try:
        return np.stack([sample["emg"][ch] for ch in EPN612_CHANNELS]).astype(float)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            "Expected eight numeric, equally sized EPN612 EMG channels"
        ) from error


def preprocess_emg(signal, sfreq):
    """Filter the full recording, independently per channel, before windowing."""
    fs = _validate_sfreq(sfreq)
    signal = np.asarray(signal, dtype=float)
    if signal.ndim != 2 or signal.shape[0] != 8 or not np.isfinite(signal).all():
        raise ValueError("Expected finite EMG with shape (8, samples)")
    frame = pd.DataFrame(signal.T, columns=EPN612_CHANNELS)
    notched = notch_filter(frame, fs=fs, freq=50, q=30, single=True)
    b, a = butter(4, 20, btype="highpass", fs=fs)
    filtered = filtfilt(b, a, notched.to_numpy().T, axis=-1)
    if filtered.shape != signal.shape or not np.isfinite(filtered).all():
        raise ValueError("Invalid filtered recording")
    return filtered
