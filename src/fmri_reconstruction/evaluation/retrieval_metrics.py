from __future__ import annotations

import numpy as np


def topk_accuracy(scores, targets, k=1):
    scores = np.asarray(scores)
    targets = np.asarray(targets)
    top = np.argsort(-scores, axis=1)[:, :k]
    return float(np.mean([t in row for t, row in zip(targets, top)]))


def topk_retrieval_accuracy(query_embeddings, database_embeddings, target_indices, k=5):
    q = np.asarray(query_embeddings, dtype=np.float32)
    d = np.asarray(database_embeddings, dtype=np.float32)
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
    d = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-8)
    scores = q @ d.T
    return topk_accuracy(scores, target_indices, k=k)
