from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import CCA
from scipy.linalg import orthogonal_procrustes


@dataclass
class SharedLatentAligner:
    """Practical linear baseline for subject alignment.

    This is a reusable preprocessing/alignment utility. It is not a claim that
    this PCA implementation is identical to the paper's learned alignment
    module; use the learned alignment checkpoint when reproducing the final
    reported model.
    """
    n_components: int = 128
    pca: PCA | None = None

    def fit(self, subject_arrays: Sequence[np.ndarray]):
        if not subject_arrays:
            raise ValueError("No subject arrays supplied.")
        common = min(x.shape[1] for x in subject_arrays)
        pooled = np.concatenate([x[:, :common] for x in subject_arrays], axis=0)
        self.pca = PCA(n_components=min(self.n_components, common))
        self.pca.fit(pooled)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.pca is None:
            raise RuntimeError("Call fit before transform.")
        common = min(X.shape[1], self.pca.n_features_in_)
        return self.pca.transform(X[:, :common]).astype(np.float32)

    def fit_transform(self, subject_arrays: Sequence[np.ndarray]) -> List[np.ndarray]:
        self.fit(subject_arrays)
        return [self.transform(x) for x in subject_arrays]


class NoAligner:
    """No alignment baseline. Returns per-subject PCA-reduced features

    This baseline does not attempt to align subjects to a shared space.
    To provide a uniform interface and dimensionality, a separate PCA is
    fit for each subject to reduce to ``n_components``. This is an
    IMPLEMENTATION_CHOICE and not claimed to be part of the paper.
    """
    def __init__(self, n_components: int = 128):
        self.n_components = n_components
        self.pcas: List[PCA] = []

    def fit(self, subject_arrays: Sequence[np.ndarray]):
        self.pcas = []
        for X in subject_arrays:
            p = PCA(n_components=min(self.n_components, X.shape[1]))
            p.fit(X)
            self.pcas.append(p)
        return self

    def transform(self, X: np.ndarray, subject_idx: int = 0) -> np.ndarray:
        if not self.pcas:
            raise RuntimeError("Call fit before transform.")
        p = self.pcas[subject_idx]
        return p.transform(X).astype(np.float32)

    def fit_transform(self, subject_arrays: Sequence[np.ndarray]) -> List[np.ndarray]:
        self.fit(subject_arrays)
        return [self.transform(X, i) for i, X in enumerate(subject_arrays)]


class RidgeAligner:
    """Ridge regression-based alignment to a pooled PCA target.

    For each subject this fits a Ridge regression mapping from the subject's
    voxel space to a shared PCA-derived target space. This is provided as a
    baseline; hyperparameters and exact formulation are implementation
    choices unless the paper specifies otherwise.
    """
    def __init__(self, n_components: int = 128, alpha: float = 1.0):
        self.n_components = n_components
        self.alpha = alpha
        self.target_pca: PCA | None = None
        self.models: List[Ridge] = []

    def fit(self, subject_arrays: Sequence[np.ndarray]):
        common = min(x.shape[1] for x in subject_arrays)
        pooled = np.concatenate([x[:, :common] for x in subject_arrays], axis=0)
        self.target_pca = PCA(n_components=min(self.n_components, common))
        targets = self.target_pca.fit_transform(pooled)

        # split targets back to per-subject lengths
        self.models = []
        idx = 0
        for X in subject_arrays:
            L = X.shape[0]
            Y = targets[idx: idx + L]
            idx += L
            model = Ridge(alpha=self.alpha)
            model.fit(X[:, :self.target_pca.n_features_in_], Y)
            self.models.append(model)
        return self

    def transform(self, X: np.ndarray, subject_idx: int = 0) -> np.ndarray:
        if not self.models or self.target_pca is None:
            raise RuntimeError("Call fit before transform.")
        model = self.models[subject_idx]
        return model.predict(X[:, : model.coef_.shape[1]]).astype(np.float32)

    def fit_transform(self, subject_arrays: Sequence[np.ndarray]) -> List[np.ndarray]:
        self.fit(subject_arrays)
        return [self.transform(X, i) for i, X in enumerate(subject_arrays)]


class ProcrustesAligner:
    """Orthogonal Procrustes-based alignment to pooled PCA target.

    This computes an orthogonal (rotation) matrix per subject mapping the
    subject data to a pooled PCA target using ``scipy.linalg.orthogonal_procrustes``.
    This is a baseline approximation and marked as an implementation choice.
    """
    def __init__(self, n_components: int = 128):
        self.n_components = n_components
        self.target_pca: PCA | None = None
        self.transforms: List[np.ndarray] = []

    def fit(self, subject_arrays: Sequence[np.ndarray]):
        common = min(x.shape[1] for x in subject_arrays)
        pooled = np.concatenate([x[:, :common] for x in subject_arrays], axis=0)
        self.target_pca = PCA(n_components=min(self.n_components, common))
        targets = self.target_pca.fit_transform(pooled)

        self.transforms = []
        idx = 0
        for X in subject_arrays:
            L = X.shape[0]
            Y = targets[idx: idx + L]
            idx += L
            Xp = self.target_pca.transform(X[:, : self.target_pca.n_features_in_])
            A, _ = orthogonal_procrustes(Xp, Y)
            self.transforms.append(A)
        return self

    def transform(self, X: np.ndarray, subject_idx: int = 0) -> np.ndarray:
        if not self.transforms or self.target_pca is None:
            raise RuntimeError("Call fit before transform.")
        A = self.transforms[subject_idx]
        return (X[:, : A.shape[0]] @ A).astype(np.float32)

    def fit_transform(self, subject_arrays: Sequence[np.ndarray]) -> List[np.ndarray]:
        self.fit(subject_arrays)
        return [self.transform(X, i) for i, X in enumerate(subject_arrays)]


class CCAAligner:
    """Canonical Correlation Analysis alignment baseline.

    Fits a multi-view CCA across subjects and projects to the shared CCA space.
    This is a baseline implementation using sklearn's CCA and is an
    IMPLEMENTATION_CHOICE.
    """
    def __init__(self, n_components: int = 128):
        self.n_components = n_components
        self.cca: CCA | None = None

    def fit(self, subject_arrays: Sequence[np.ndarray]):
        # concatenate time-wise across subjects as channels for CCA
        common = min(x.shape[1] for x in subject_arrays)
        views = [x[:, :common] for x in subject_arrays]
        # sklearn CCA expects two views; for >2 we concatenate pairwise
        if len(views) < 2:
            raise ValueError("CCA requires at least two subjects/views.")
        X = views[0]
        Y = np.concatenate(views[1:], axis=1)
        self.cca = CCA(n_components=min(self.n_components, X.shape[1], Y.shape[1]))
        self.cca.fit(X, Y)
        return self

    def transform(self, X: np.ndarray, subject_idx: int = 0) -> np.ndarray:
        if self.cca is None:
            raise RuntimeError("Call fit before transform.")
        # For simplicity apply the same CCA transform to any subject by
        # projecting into the learned space using the x_weights_
        return (X[:, : self.cca.x_weights_.shape[0]] @ self.cca.x_weights_).astype(np.float32)

    def fit_transform(self, subject_arrays: Sequence[np.ndarray]) -> List[np.ndarray]:
        self.fit(subject_arrays)
        return [self.transform(X, i) for i, X in enumerate(subject_arrays)]


class SRMAligner:
    """Shared Response Model (SRM) baseline (simple iterative orthogonal SRM).

    This is a lightweight SRM-like implementation that iteratively computes
    subject-wise orthogonal mappings to a shared response by alternating
    Procrustes updates. It is provided as a baseline and implementation
    choice — for production SRM use dedicated libraries such as BrainIAK.
    """
    def __init__(self, n_components: int = 128, n_iter: int = 10):
        self.n_components = n_components
        self.n_iter = n_iter
        self.shared_: np.ndarray | None = None
        self.transforms: List[np.ndarray] = []

    def fit(self, subject_arrays: Sequence[np.ndarray]):
        if not subject_arrays:
            raise ValueError("No subject arrays supplied.")
        # initialize shared response as PCA on pooled data
        common = min(x.shape[1] for x in subject_arrays)
        pooled = np.concatenate([x[:, :common] for x in subject_arrays], axis=0)
        self.target_pca = PCA(n_components=min(self.n_components, common))
        self.shared_ = self.target_pca.fit_transform(pooled)

        # initialize transforms by Procrustes from subject -> shared
        self.transforms = []
        idx = 0
        for X in subject_arrays:
            L = X.shape[0]
            Y = self.shared_[idx: idx + L]
            idx += L
            Xp = self.target_pca.transform(X[:, : self.target_pca.n_features_in_])
            A, _ = orthogonal_procrustes(Xp, Y)
            self.transforms.append(A)

        # iterate: update shared as average of transformed subjects
        for _ in range(self.n_iter):
            mapped = [self.target_pca.transform(X[:, : self.target_pca.n_features_in_]) @ self.transforms[i] for i, X in enumerate(subject_arrays)]
            # average by timepoint alignment: truncate to min length
            minT = min(m.shape[0] for m in mapped)
            stacked = np.stack([m[:minT] for m in mapped], axis=0)
            self.shared_ = stacked.mean(axis=0)

            # recompute transforms
            self.transforms = []
            for X in subject_arrays:
                Xp = self.target_pca.transform(X[:, : self.target_pca.n_features_in_])
                A, _ = orthogonal_procrustes(Xp[: self.shared_.shape[0], : self.shared_.shape[1]], self.shared_)
                self.transforms.append(A)

        return self

    def transform(self, X: np.ndarray, subject_idx: int = 0) -> np.ndarray:
        if not self.transforms:
            raise RuntimeError("Call fit before transform.")
        A = self.transforms[subject_idx]
        Xp = self.target_pca.transform(X[:, : self.target_pca.n_features_in_])
        return (Xp @ A).astype(np.float32)

    def fit_transform(self, subject_arrays: Sequence[np.ndarray]) -> List[np.ndarray]:
        self.fit(subject_arrays)
        return [self.transform(X, i) for i, X in enumerate(subject_arrays)]


class Hyperalignment:
    """Iterative hyperalignment baseline using orthogonal Procrustes.

    This implements a simple hyperalignment procedure: iteratively align each
    subject to the mean of others using orthogonal Procrustes until convergence.
    This is a baseline and an implementation choice; it is not a claim of
    equivalence to specialized hyperalignment toolkits.
    """
    def __init__(self, n_components: int = 128, n_iter: int = 10):
        self.n_components = n_components
        self.n_iter = n_iter
        self.transforms: List[np.ndarray] = []
        self.template_: np.ndarray | None = None

    def fit(self, subject_arrays: Sequence[np.ndarray]):
        common = min(x.shape[1] for x in subject_arrays)
        # project each subject to PCA first to reduce noise
        pcas = [PCA(n_components=min(self.n_components, common)).fit(X[:, :common]) for X in subject_arrays]
        projs = [p.transform(X[:, :common]) for p, X in zip(pcas, subject_arrays)]

        # initialize template as mean of projections (aligned in PCA space)
        minT = min(p.shape[0] for p in projs)
        stacked = np.stack([p[:minT] for p in projs], axis=0)
        template = stacked.mean(axis=0)

        transforms = [np.eye(template.shape[1], dtype=np.float32) for _ in projs]

        for _ in range(self.n_iter):
            for i, p in enumerate(projs):
                A, _ = orthogonal_procrustes(p[:minT], template)
                transforms[i] = A
            mapped = [projs[i][:minT] @ transforms[i] for i in range(len(projs))]
            template = np.stack(mapped, axis=0).mean(axis=0)

        self.transforms = transforms
        self.template_ = template
        return self

    def transform(self, X: np.ndarray, subject_idx: int = 0) -> np.ndarray:
        if not self.transforms:
            raise RuntimeError("Call fit before transform.")
        A = self.transforms[subject_idx]
        common = min(X.shape[1], A.shape[0])
        return (X[:, :common] @ A).astype(np.float32)

    def fit_transform(self, subject_arrays: Sequence[np.ndarray]) -> List[np.ndarray]:
        self.fit(subject_arrays)
        return [self.transform(X, i) for i, X in enumerate(subject_arrays)]
