import numpy as np

from fmri_reconstruction.data.sequences import make_sequences
from fmri_reconstruction.preprocessing import detrend_time
from fmri_reconstruction.retrieval.cosine import top_k


def test_sequence_shape():
    x = np.random.randn(30, 8).astype(np.float32)
    y = make_sequences(x, length=5)
    assert y.shape == (26, 5, 8)


def test_detrend_shape():
    x = np.random.randn(20, 10).astype(np.float32)
    assert detrend_time(x).shape == x.shape


def test_retrieval():
    db = np.eye(4, dtype=np.float32)
    idx, scores = top_k(db[0], db, k=2)
    assert idx[0] == 0
