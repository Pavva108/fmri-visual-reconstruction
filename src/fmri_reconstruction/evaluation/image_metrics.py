from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity
from sklearn.metrics import mean_squared_error


def pixcorr(a, b):
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def ssim(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    data_range = float(max(a.max(), b.max()) - min(a.min(), b.min()))
    if data_range <= 0:
        data_range = 1.0
    kwargs = {"data_range": data_range}
    if a.ndim == 3:
        kwargs["channel_axis"] = -1
    return float(structural_similarity(a, b, **kwargs))


def mse(a, b):
    return float(mean_squared_error(np.asarray(a).ravel(), np.asarray(b).ravel()))
