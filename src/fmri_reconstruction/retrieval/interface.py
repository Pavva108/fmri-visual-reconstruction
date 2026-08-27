from __future__ import annotations

from typing import Iterable

import numpy as np

from .database import RetrievalDatabase
from .cosine import retrieve_batch
from .split_guard import assert_no_test_overlap


def build_db(embeddings: np.ndarray, paths: Iterable[str], metadata: dict | None = None) -> RetrievalDatabase:
    db = RetrievalDatabase(embeddings=embeddings, paths=list(paths), metadata=metadata or {})
    return db.normalize()


def guarded_retrieve(queries: np.ndarray, db: RetrievalDatabase, train_paths: Iterable[str], test_paths: Iterable[str], k: int = 5):
    """Assert no overlap between train and test, then retrieve top-k for each query.

    Returns: list of (indices, scores) for each query.
    """
    assert_no_test_overlap(train_paths, test_paths)
    return retrieve_batch(queries, db.embeddings, k=k)
