from __future__ import annotations

import numpy as np


def l2_normalize(x, axis=-1, eps=1e-8):
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)


def cosine_scores(query, database):
    q = l2_normalize(query)
    db = l2_normalize(database)
    return db @ q


def top_k(query, database, k=5):
    scores = cosine_scores(query, database)
    idx = np.argsort(-scores)[:k]
    return idx, scores[idx]


def retrieve_batch(queries, database, k=5):
    return [top_k(q, database, k=k) for q in np.asarray(queries)]
