"""Compare alignment methods on synthetic or real data.

This script is a lightweight comparison harness that runs multiple aligners
on supplied subject arrays and reports simple alignment metrics. It is
intended for evaluation and reproducibility; do not claim results are
paper-level reproductions without running on the original datasets.
"""
import json
import os
from pathlib import Path
import argparse
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
from fmri_reconstruction.evaluation.alignment_evaluator import evaluate_alignment


METHODS = {
    "shared_pca": SharedLatentAligner,
    "no_align": NoAligner,
    "ridge": RidgeAligner,
    "procrustes": ProcrustesAligner,
    "cca": CCAAligner,
    "srm": SRMAligner,
    "hyper": Hyperalignment,
}


def synthetic_subject_data(n_subjects=4, T=120, D=80, seed=0):
    rng = np.random.RandomState(seed)
    base = rng.randn(T, 30)
    subjects = []
    for i in range(n_subjects):
        A = rng.randn(30, D) * 0.6
        noise = rng.randn(T, D) * 0.12
        subjects.append(base @ A + noise)
    return subjects


def run_compare(subjects, methods=None, n_components=128):
    if methods is None:
        methods = list(METHODS.keys())
    results = {}
    for name in methods:
        cls = METHODS[name]
        m = cls(n_components=n_components)
        aligned = m.fit_transform(subjects)
        results[name] = evaluate_alignment(aligned)
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=str, default="alignment_comparison.json")
    args = p.parse_args()

    subjects = synthetic_subject_data()
    results = run_compare(subjects, n_components=10)
    out = Path(args.out)
    out.write_text(json.dumps(results, indent=2))
    print(f"Wrote {out}")


if __name__ == '__main__':
    main()
