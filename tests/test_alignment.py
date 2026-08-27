import numpy as np

from fmri_reconstruction.alignment import (
    SharedLatentAligner,
    NoAligner,
    RidgeAligner,
    ProcrustesAligner,
    CCAAligner,
    SRMAligner,
    Hyperalignment,
)


def synthetic_subject_data(n_subjects=3, T=60, D=50):
    rng = np.random.RandomState(0)
    base = rng.randn(T, 20)
    subjects = []
    for i in range(n_subjects):
        A = rng.randn(20, D) * 0.5
        noise = rng.randn(T, D) * 0.1
        subjects.append(base @ A + noise)
    return subjects


def test_aligners_api():
    subs = synthetic_subject_data()
    methods = [
        SharedLatentAligner(n_components=10),
        NoAligner(n_components=10),
        RidgeAligner(n_components=10),
        ProcrustesAligner(n_components=10),
        CCAAligner(n_components=10),
        SRMAligner(n_components=10, n_iter=3),
        Hyperalignment(n_components=10, n_iter=3),
    ]

    for m in methods:
        out = m.fit_transform(subs)
        assert isinstance(out, list)
        assert len(out) == len(subs)
        for o in out:
            assert o.ndim == 2
            assert o.shape[1] == 10
