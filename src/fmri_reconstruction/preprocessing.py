from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import signal
from sklearn.preprocessing import StandardScaler


@dataclass
class StandardizationState:
    mean: np.ndarray
    scale: np.ndarray


def detrend_time(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    return signal.detrend(X, axis=0).astype(np.float32)


def bandpass_time(X: np.ndarray, tr_seconds: float, low=0.01, high=0.1, order=4) -> np.ndarray:
    X = np.asarray(X, dtype=np.float32)
    fs = 1.0 / tr_seconds
    nyq = 0.5 * fs
    if high >= nyq:
        raise ValueError(f"High cutoff {high} Hz must be below Nyquist {nyq} Hz.")
    b, a = signal.butter(order, [low / nyq, high / nyq], btype="band")
    # filtfilt uses padding; if the time-series is too short for the filter
    # padlen, skip filtering to avoid errors (caller may choose longer data).
    padlen = 3 * (max(len(a), len(b)) - 1)
    if X.shape[0] <= padlen:
        return X
    return signal.filtfilt(b, a, X, axis=0).astype(np.float32)


def fit_voxel_standardizer(X: np.ndarray) -> StandardizationState:
    X = np.asarray(X, dtype=np.float32)
    scaler = StandardScaler()
    scaler.fit(X)
    return StandardizationState(
        mean=scaler.mean_.astype(np.float32),
        scale=scaler.scale_.astype(np.float32),
    )


def apply_voxel_standardizer(X: np.ndarray, state: StandardizationState) -> np.ndarray:
    scale = np.where(state.scale == 0, 1.0, state.scale)
    return ((X - state.mean) / scale).astype(np.float32)


def preprocess_live(X: np.ndarray, tr_seconds: float = 3.0, low=0.01, high=0.1):
    X = detrend_time(X)
    X = bandpass_time(X, tr_seconds, low, high)
    state = fit_voxel_standardizer(X)
    return apply_voxel_standardizer(X, state), state


def preprocess_haxby(X: np.ndarray, tr_seconds: float = 3.0, low=0.01, high=0.1, fit_scaler: bool = True, scaler: Optional[StandardScaler] = None):
    """Preprocess Haxby-style voxel time series.

    Steps: remove rest (caller), VT ROI extraction (caller), detrend, bandpass, voxel-wise standardization.
    Returns processed array and the fitted StandardizationState (or provided one).
    """
    X = detrend_time(X)
    X = bandpass_time(X, tr_seconds, low, high)
    if fit_scaler:
        state = fit_voxel_standardizer(X)
        Xs = apply_voxel_standardizer(X, state)
        return Xs, state
    else:
        if scaler is None:
            raise ValueError("scaler must be provided when fit_scaler=False")
        state = StandardizationState(mean=scaler.mean_.astype(np.float32), scale=scaler.scale_.astype(np.float32))
        return apply_voxel_standardizer(X, state), state


def preprocess_nsd(X: np.ndarray, scaler: Optional[StandardScaler] = None, fit_scaler: bool = True):
    """Preprocess NSD MindEye HDF5 data.

    NSD preprocessing uses StandardScaler normalization on voxel features.
    ``fit_scaler=True`` fits on provided X (training data). To avoid leakage,
    callers should fit on training splits only and then pass the scaler to
    validation/test data processing.
    """
    X = np.asarray(X, dtype=np.float32)
    if fit_scaler:
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X).astype(np.float32)
        return Xs, scaler
    else:
        if scaler is None:
            raise ValueError("scaler must be provided when fit_scaler=False")
        return scaler.transform(X).astype(np.float32), scaler
