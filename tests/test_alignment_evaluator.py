import numpy as np
from fmri_reconstruction.evaluation.alignment_evaluator import evaluate_alignment


def test_evaluator_shapes():
    # create three aligned subject arrays (T x D)
    T, D = 40, 10
    base = np.random.randn(T, D)
    subs = [base + np.random.randn(T, D) * 0.01 for _ in range(3)]
    res = evaluate_alignment(subs)
    assert "mean_pairwise_correlation" in res
    assert "variance_explained" in res
    assert isinstance(res["mean_pairwise_correlation"], float)
