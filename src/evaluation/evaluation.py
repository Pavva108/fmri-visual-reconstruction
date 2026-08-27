"""One public evaluation API for feature, image, alignment, and retrieval scores."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image

from fmri_reconstruction.evaluation.evaluation import (
    CosineLoss,
    CLIPDINOJointLoss,
    combined_embedding_loss,
    cosine_loss,
    cosine_similarity,
    cosine_similarity_matrix,
    compute_evaluation_metrics as _core_compute_evaluation_metrics,
    mean_cosine_similarity,
    mean_pairwise_correlation,
    mean_reciprocal_rank,
    mse,
    mse_loss,
    normalized_feature_correlation,
    pixcorr,
    psnr,
    reconstruction_loss,
    rmse,
    ssim,
    topk_accuracy,
    topk_retrieval_accuracy,
    variance_explained,
)


def evaluate_reconstruction_batch(
    original_images: Sequence[str | Path | np.ndarray],
    reconstructed_images: Sequence[str | Path | np.ndarray],
    metrics: Iterable[str] | None = None,
) -> dict[str, float]:
    """Average image metrics across a reconstruction batch.

    Images are resized to the reference dimensions before scoring, matching
    the image comparison stage in the notebooks.
    """
    metric_names = list(metrics or ("ssim", "mse", "pixcorr", "psnr"))
    values = {name: [] for name in metric_names}
    for original, reconstruction in zip(original_images, reconstructed_images):
        reference = _as_rgb_array(original)
        prediction = _as_rgb_array(reconstruction, size=(reference.shape[1], reference.shape[0]))
        scores = compute_evaluation_metrics(prediction, reference, metric_names)
        for name, value in scores.items():
            values[name].append(value)
    return {name: float(np.mean(scores)) if scores else 0.0 for name, scores in values.items()}


def compute_evaluation_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    metrics: Iterable[str] | None = None,
) -> dict[str, float]:
    """Compute any supported feature or image metric by name."""
    names = list(metrics or ("mse", "cosine_sim", "correlation"))
    feature_names = {"mse", "cosine_sim", "correlation"}
    result = _core_compute_evaluation_metrics(
        predictions, targets, [name for name in names if name in feature_names]
    )
    image_metrics = {
        "ssim": ssim,
        "pixcorr": pixcorr,
        "psnr": psnr,
        "rmse": rmse,
    }
    for name, function in image_metrics.items():
        if name in names:
            result[name] = function(predictions, targets)
    unknown = set(names) - feature_names - set(image_metrics)
    if unknown:
        raise ValueError(f"Unknown evaluation metrics: {sorted(unknown)}")
    return result


def evaluate_retrieval(
    query_embeddings: np.ndarray,
    database_embeddings: np.ndarray,
    target_indices: np.ndarray,
    ks: Iterable[int] = (1, 5, 10),
) -> dict[str, float]:
    """Return top-k retrieval accuracy and MRR for known database targets."""
    query = np.asarray(query_embeddings, dtype=np.float32)
    database = np.asarray(database_embeddings, dtype=np.float32)
    targets = np.asarray(target_indices)
    normalized_query = query / (np.linalg.norm(query, axis=1, keepdims=True) + 1e-8)
    normalized_database = database / (np.linalg.norm(database, axis=1, keepdims=True) + 1e-8)
    rankings = np.argsort(-(normalized_query @ normalized_database.T), axis=1)
    result = {f"top{k}": topk_retrieval_accuracy(query, database, targets, k) for k in ks}
    result["mrr"] = mean_reciprocal_rank(rankings.tolist(), targets.tolist())
    return result


def _as_rgb_array(value: str | Path | np.ndarray, size: tuple[int, int] | None = None) -> np.ndarray:
    if isinstance(value, (str, Path)):
        image = Image.open(value).convert("RGB")
        if size is not None:
            image = image.resize(size)
        return np.asarray(image)
    array = np.asarray(value)
    if size is not None and (array.shape[1], array.shape[0]) != size:
        image = Image.fromarray(array.astype(np.uint8)).resize(size)
        return np.asarray(image)
    return array


__all__ = [
    "cosine_similarity", "cosine_similarity_matrix", "mean_cosine_similarity",
    "normalized_feature_correlation", "pixcorr", "ssim", "mse", "rmse", "psnr",
    "topk_accuracy", "topk_retrieval_accuracy", "mean_reciprocal_rank",
    "mean_pairwise_correlation", "variance_explained", "CosineLoss",
    "CLIPDINOJointLoss", "cosine_loss", "mse_loss", "combined_embedding_loss",
    "reconstruction_loss", "compute_evaluation_metrics", "evaluate_reconstruction_batch",
    "evaluate_retrieval",
]