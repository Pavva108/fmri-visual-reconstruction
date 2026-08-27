from __future__ import annotations

import numpy as np


def cosine_similarity_matrix(a, b):
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
    return np.sum(a * b, axis=-1)


def mean_cosine_similarity(a, b):
    return float(np.mean(cosine_similarity_matrix(a, b)))
