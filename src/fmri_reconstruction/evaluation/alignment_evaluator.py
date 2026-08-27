from __future__ import annotations

from typing import Sequence, Dict, Any
import numpy as np


def mean_pairwise_correlation(aligned_subjects: Sequence[np.ndarray]) -> float:
    """Compute mean Pearson correlation across subject representations.

    aligned_subjects: List of T x D arrays (time x features) after alignment.
    Returns mean correlation across subjects vs the group-mean template.
    """
    mapped = [np.asarray(x, dtype=np.float32) for x in aligned_subjects]
    minT = min(m.shape[0] for m in mapped)
    mapped = [m[:minT] for m in mapped]
    template = np.stack(mapped, axis=0).mean(axis=0)

    corrs = []
    for m in mapped:
        # flatten along time and features
        a = m.ravel()
        b = template.ravel()
        if a.std() == 0 or b.std() == 0:
            corrs.append(0.0)
        else:
            corrs.append(float(np.corrcoef(a, b)[0, 1]))
    return float(np.mean(corrs))


def variance_explained(aligned_subjects: Sequence[np.ndarray]) -> float:
    """Compute fraction of variance explained by the group template.

    Returns average across subjects of 1 - Var(residual)/Var(original).
    """
    mapped = [np.asarray(x, dtype=np.float32) for x in aligned_subjects]
    minT = min(m.shape[0] for m in mapped)
    mapped = [m[:minT] for m in mapped]
    template = np.stack(mapped, axis=0).mean(axis=0)

    vals = []
    for m in mapped:
        orig_var = np.var(m)
        resid_var = np.var(m - template)
        if orig_var == 0:
            vals.append(0.0)
        else:
            vals.append(float(1.0 - resid_var / orig_var))
    return float(np.mean(vals))


def evaluate_alignment(aligned_subjects: Sequence[np.ndarray]) -> Dict[str, Any]:
    return {
        "mean_pairwise_correlation": mean_pairwise_correlation(aligned_subjects),
        "variance_explained": variance_explained(aligned_subjects),
    }
