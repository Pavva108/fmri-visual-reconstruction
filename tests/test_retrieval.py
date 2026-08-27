import numpy as np
import pytest

from fmri_reconstruction.retrieval.database import RetrievalDatabase
from fmri_reconstruction.retrieval.interface import build_db, guarded_retrieve
from fmri_reconstruction.retrieval.split_guard import assert_no_test_overlap


def test_build_db_and_retrieve():
    rng = np.random.RandomState(0)
    embeddings = rng.randn(10, 64).astype(np.float32)
    paths = [f"img_{i}.jpg" for i in range(10)]
    db = build_db(embeddings, paths)
    assert db.embeddings.shape == (10, 64)

    # query a random vector
    q = rng.randn(64).astype(np.float32)
    results = guarded_retrieve([q], db, train_paths=paths[:8], test_paths=paths[8:], k=3)
    assert isinstance(results, list)
    idx, scores = results[0]
    assert len(idx) == 3
    assert len(scores) == 3


def test_split_guard_raises_on_overlap():
    train = ["a.jpg", "b.jpg"]
    test = ["b.jpg", "c.jpg"]
    with pytest.raises(AssertionError):
        assert_no_test_overlap(train, test)
