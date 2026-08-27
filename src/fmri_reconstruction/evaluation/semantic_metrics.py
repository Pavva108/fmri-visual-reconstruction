from __future__ import annotations

import numpy as np


def normalized_feature_correlation(pred, target):
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    pred = pred - pred.mean(axis=1, keepdims=True)
    target = target - target.mean(axis=1, keepdims=True)
    num = np.sum(pred * target, axis=1)
    den = np.sqrt(np.sum(pred**2, axis=1) * np.sum(target**2, axis=1)) + 1e-8
    return float(np.mean(num / den))
