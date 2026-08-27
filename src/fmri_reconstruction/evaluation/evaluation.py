"""Comprehensive evaluation metrics and loss functions for fMRI visual reconstruction.

Includes feature-space metrics, image quality metrics, alignment metrics,
and various loss functions for training neural models.
"""

from __future__ import annotations

from typing import Optional, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from skimage.metrics import structural_similarity
from sklearn.metrics import mean_squared_error


# =========================================================
# FEATURE SPACE METRICS
# =========================================================

def cosine_similarity(a: np.ndarray | torch.Tensor, 
                      b: np.ndarray | torch.Tensor) -> float:
    """Compute cosine similarity between two vectors.
    
    Args:
        a: Vector or batch of vectors (D,) or (B, D)
        b: Vector or batch of vectors (D,) or (B, D)
        
    Returns:
        Cosine similarity score
    """
    if isinstance(a, torch.Tensor):
        a = a.cpu().numpy()
    if isinstance(b, torch.Tensor):
        b = b.cpu().numpy()
    
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    
    # Normalize
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
    
    # Compute cosine similarity
    if a.ndim == 1:
        return float(np.dot(a, b))
    else:
        return float(np.mean(np.sum(a * b, axis=-1)))


def cosine_similarity_matrix(a: np.ndarray, 
                              b: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity matrix.
    
    Args:
        a: Array shape (N, D)
        b: Array shape (M, D)
        
    Returns:
        Similarity matrix shape (N, M)
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    
    # Normalize
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    
    return a @ b.T


def mean_cosine_similarity(a: np.ndarray, 
                            b: np.ndarray) -> float:
    """Compute mean cosine similarity across samples.
    
    Args:
        a: Array shape (N, D)
        b: Array shape (N, D)
        
    Returns:
        Mean cosine similarity
    """
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    
    # Normalize
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    
    # Pairwise dot product
    similarities = np.sum(a * b, axis=1)
    
    return float(np.mean(similarities))


def normalized_feature_correlation(pred: np.ndarray, 
                                     target: np.ndarray) -> float:
    """Compute normalized feature correlation (Pearson).
    
    Args:
        pred: Predicted features (N, D)
        target: Target features (N, D)
        
    Returns:
        Mean normalized correlation
    """
    pred = np.asarray(pred, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    
    # Center
    pred = pred - pred.mean(axis=1, keepdims=True)
    target = target - target.mean(axis=1, keepdims=True)
    
    # Normalize
    pred_norm = np.sqrt(np.sum(pred**2, axis=1, keepdims=True)) + 1e-8
    target_norm = np.sqrt(np.sum(target**2, axis=1, keepdims=True)) + 1e-8
    
    pred = pred / pred_norm
    target = target / target_norm
    
    # Correlation
    corr = np.sum(pred * target, axis=1)
    
    return float(np.mean(corr))


# =========================================================
# IMAGE QUALITY METRICS
# =========================================================

def pixcorr(a: np.ndarray, b: np.ndarray) -> float:
    """Compute pixel-wise correlation between two images.
    
    Args:
        a: Image array
        b: Image array
        
    Returns:
        Pixel correlation coefficient
    """
    a = np.asarray(a, dtype=np.float32).ravel()
    b = np.asarray(b, dtype=np.float32).ravel()
    
    if np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    
    return float(np.corrcoef(a, b)[0, 1])


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Structural Similarity Index (SSIM).
    
    Args:
        a: Image array
        b: Image array
        
    Returns:
        SSIM score (range -1 to 1, higher is better)
    """
    a = np.asarray(a)
    b = np.asarray(b)
    
    data_range = float(max(a.max(), b.max()) - min(a.min(), b.min()))
    if data_range <= 0:
        data_range = 1.0
    
    kwargs = {"data_range": data_range}
    if a.ndim == 3:
        kwargs["channel_axis"] = -1
    
    return float(structural_similarity(a, b, **kwargs))


def mse(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Mean Squared Error.
    
    Args:
        a: Array
        b: Array
        
    Returns:
        MSE value
    """
    return float(mean_squared_error(
        np.asarray(a).ravel(),
        np.asarray(b).ravel()
    ))


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Root Mean Squared Error.
    
    Args:
        a: Array
        b: Array
        
    Returns:
        RMSE value
    """
    return float(np.sqrt(mse(a, b)))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Peak Signal-to-Noise Ratio.
    
    Args:
        a: Image array (0-255 or 0-1)
        b: Image array
        
    Returns:
        PSNR in dB
    """
    mse_val = mse(a, b)
    if mse_val == 0:
        return 100.0
    
    max_val = max(np.asarray(a).max(), np.asarray(b).max())
    if max_val <= 1:
        max_val = 1.0
    else:
        max_val = 255.0
    
    return float(20 * np.log10(max_val / np.sqrt(mse_val)))


# =========================================================
# RETRIEVAL METRICS
# =========================================================

def topk_accuracy(scores: np.ndarray, targets: np.ndarray, 
                   k: int = 1) -> float:
    """Compute top-k classification accuracy.
    
    Args:
        scores: Prediction scores (N, C)
        targets: Target indices (N,)
        k: Rank threshold
        
    Returns:
        Top-k accuracy (0-1)
    """
    scores = np.asarray(scores)
    targets = np.asarray(targets)
    
    # Get top-k predictions
    top_k_preds = np.argsort(-scores, axis=1)[:, :k]
    
    # Check if target in top-k
    matches = np.array([
        t in preds for t, preds in zip(targets, top_k_preds)
    ])
    
    return float(np.mean(matches))


def topk_retrieval_accuracy(query_embeddings: np.ndarray,
                              database_embeddings: np.ndarray,
                              target_indices: np.ndarray,
                              k: int = 5) -> float:
    """Compute top-k retrieval accuracy.
    
    Args:
        query_embeddings: Query embeddings (N, D)
        database_embeddings: Database embeddings (M, D)
        target_indices: Target indices in database (N,)
        k: Number of top results to consider
        
    Returns:
        Top-k retrieval accuracy
    """
    q = np.asarray(query_embeddings, dtype=np.float32)
    d = np.asarray(database_embeddings, dtype=np.float32)
    targets = np.asarray(target_indices)
    
    # Normalize
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)
    d = d / (np.linalg.norm(d, axis=1, keepdims=True) + 1e-8)
    
    # Compute similarities
    scores = q @ d.T
    
    return topk_accuracy(scores, targets, k=k)


def mean_reciprocal_rank(predictions: list, targets: list) -> float:
    """Compute Mean Reciprocal Rank (MRR).
    
    Args:
        predictions: List of ranked predictions
        targets: List of target elements
        
    Returns:
        MRR score (0-1)
    """
    reciprocal_ranks = []
    
    for preds, target in zip(predictions, targets):
        if target in preds:
            rank = preds.index(target) + 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)
    
    return float(np.mean(reciprocal_ranks))


# =========================================================
# ALIGNMENT METRICS
# =========================================================

def mean_pairwise_correlation(aligned_subjects: list[np.ndarray]) -> float:
    """Compute mean pairwise correlation (alignment quality).
    
    Args:
        aligned_subjects: List of aligned subject arrays (T, D)
        
    Returns:
        Mean correlation across subjects
    """
    mapped = [np.asarray(x, dtype=np.float32) for x in aligned_subjects]
    
    # Truncate to minimum length
    min_t = min(m.shape[0] for m in mapped)
    mapped = [m[:min_t] for m in mapped]
    
    # Compute template
    template = np.stack(mapped, axis=0).mean(axis=0)
    
    # Compute correlations
    corrs = []
    for m in mapped:
        a = m.ravel()
        b = template.ravel()
        
        if a.std() == 0 or b.std() == 0:
            corrs.append(0.0)
        else:
            corrs.append(float(np.corrcoef(a, b)[0, 1]))
    
    return float(np.mean(corrs))


def variance_explained(aligned_subjects: list[np.ndarray]) -> float:
    """Compute variance explained by the group template.
    
    Args:
        aligned_subjects: List of aligned subject arrays (T, D)
        
    Returns:
        Fraction of variance explained (0-1)
    """
    mapped = [np.asarray(x, dtype=np.float32) for x in aligned_subjects]
    
    # Truncate to minimum length
    min_t = min(m.shape[0] for m in mapped)
    mapped = [m[:min_t] for m in mapped]
    
    # Compute template
    template = np.stack(mapped, axis=0).mean(axis=0)
    
    # Compute variance explained
    vals = []
    for m in mapped:
        orig_var = np.var(m)
        resid_var = np.var(m - template)
        
        if orig_var == 0:
            vals.append(0.0)
        else:
            vals.append(float(1.0 - resid_var / orig_var))
    
    return float(np.mean(vals))


# =========================================================
# LOSS FUNCTIONS (PyTorch)
# =========================================================

class CosineLoss(nn.Module):
    """Cosine distance loss in feature space."""
    
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, 
                target: torch.Tensor) -> torch.Tensor:
        """Compute cosine loss.
        
        Args:
            pred: Predictions (B, D)
            target: Targets (B, D)
            
        Returns:
            Loss value
        """
        pred = F.normalize(pred, dim=-1)
        target = F.normalize(target, dim=-1)
        
        loss = 1.0 - (pred * target).sum(dim=-1)
        
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class CLIPDINOJointLoss(nn.Module):
    """Combined MSE + cosine loss for CLIP/DINO projectors.
    
    L = MSE(pred, target) + lambda_cos * (1 - cosine_similarity)
    """
    
    def __init__(self, lambda_cos: Optional[float] = None):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lambda_cos = lambda_cos
        self.cos = nn.CosineSimilarity(dim=-1, eps=1e-8)
    
    def forward(self, pred: torch.Tensor, 
                target: torch.Tensor) -> torch.Tensor:
        """Compute combined loss.
        
        Args:
            pred: Predictions (B, D)
            target: Targets (B, D)
            
        Returns:
            Loss value
        """
        mse_l = self.mse(pred, target)
        
        if self.lambda_cos is None:
            return mse_l
        
        cos_sim = self.cos(pred, target).mean()
        cos_l = 1.0 - cos_sim
        
        return mse_l + float(self.lambda_cos) * cos_l


def cosine_loss(pred: torch.Tensor, 
                 target: torch.Tensor) -> torch.Tensor:
    """Standalone cosine loss function.
    
    Args:
        pred: Predictions (B, D)
        target: Targets (B, D)
        
    Returns:
        Loss value
    """
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    return (1.0 - (pred * target).sum(dim=-1)).mean()


def mse_loss(pred: torch.Tensor, 
             target: torch.Tensor) -> torch.Tensor:
    """Mean squared error loss.
    
    Args:
        pred: Predictions
        target: Targets
        
    Returns:
        Loss value
    """
    return F.mse_loss(pred, target)


def combined_embedding_loss(
    pred_clip: torch.Tensor,
    target_clip: torch.Tensor,
    pred_dino: Optional[torch.Tensor] = None,
    target_dino: Optional[torch.Tensor] = None,
    clip_weight: float = 1.0,
    dino_weight: float = 1.0,
    loss_type: str = "cosine",
) -> torch.Tensor:
    """Combined loss for multi-modal predictions.
    
    Args:
        pred_clip: CLIP predictions (B, 512)
        target_clip: CLIP targets (B, 512)
        pred_dino: DINOv2 predictions (B, 768)
        target_dino: DINOv2 targets (B, 768)
        clip_weight: Weight for CLIP loss
        dino_weight: Weight for DINOv2 loss
        loss_type: "cosine" or "mse"
        
    Returns:
        Combined loss value
    """
    if loss_type == "cosine":
        loss_fn = cosine_loss
    else:
        loss_fn = mse_loss
    
    loss = clip_weight * loss_fn(pred_clip, target_clip)
    
    if pred_dino is not None and target_dino is not None:
        loss = loss + dino_weight * loss_fn(pred_dino, target_dino)
    
    return loss


def reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Reconstruction loss (MSE in pixel/feature space).
    
    Args:
        pred: Predicted output
        target: Target output
        reduction: "mean", "sum", or "none"
        
    Returns:
        Loss value
    """
    return F.mse_loss(pred, target, reduction=reduction)


# =========================================================
# COMBINED METRICS COMPUTATION
# =========================================================

def compute_evaluation_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    metrics: list[str] | None = None,
) -> dict[str, float]:
    """Compute multiple evaluation metrics.
    
    Args:
        predictions: Model predictions
        targets: Ground truth values
        metrics: List of metrics to compute (None = all)
        
    Returns:
        Dictionary of metric names and values
    """
    if metrics is None:
        metrics = ["mse", "cosine_sim", "correlation"]
    
    results = {}
    
    if "mse" in metrics:
        results["mse"] = mse(predictions, targets)
    
    if "cosine_sim" in metrics:
        results["cosine_sim"] = mean_cosine_similarity(predictions, targets)
    
    if "correlation" in metrics:
        results["correlation"] = normalized_feature_correlation(predictions, targets)
    
    return results


__all__ = [
    # Feature space
    "cosine_similarity",
    "cosine_similarity_matrix",
    "mean_cosine_similarity",
    "normalized_feature_correlation",
    # Image quality
    "pixcorr",
    "ssim",
    "mse",
    "rmse",
    "psnr",
    # Retrieval
    "topk_accuracy",
    "topk_retrieval_accuracy",
    "mean_reciprocal_rank",
    # Alignment
    "mean_pairwise_correlation",
    "variance_explained",
    # Loss functions
    "CosineLoss",
    "CLIPDINOJointLoss",
    "cosine_loss",
    "mse_loss",
    "combined_embedding_loss",
    "reconstruction_loss",
    "compute_evaluation_metrics",
]
