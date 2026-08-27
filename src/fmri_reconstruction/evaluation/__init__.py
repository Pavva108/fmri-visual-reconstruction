"""Evaluation metrics for fMRI visual reconstruction.

Includes feature similarity metrics, image quality metrics, alignment evaluation,
retrieval performance metrics, and comprehensive loss functions.
"""

# Legacy imports for backward compatibility
from .feature_metrics import cosine_similarity_matrix, mean_cosine_similarity
from .image_metrics import pixcorr, ssim, mse
from .semantic_metrics import normalized_feature_correlation
from .retrieval_metrics import topk_accuracy, topk_retrieval_accuracy
from .alignment_evaluator import mean_pairwise_correlation, variance_explained

# New comprehensive evaluation module
from .evaluation import (
    # Feature metrics
    cosine_similarity,
    cosine_similarity_matrix as cosine_sim_matrix,
    mean_cosine_similarity as mean_cos_sim,
    normalized_feature_correlation as norm_feature_corr,
    # Image quality
    pixcorr as pixel_correlation,
    ssim as structural_similarity,
    mse as mean_squared_error,
    rmse,
    psnr,
    # Retrieval
    topk_accuracy as top_k_accuracy,
    topk_retrieval_accuracy as top_k_retrieval_acc,
    mean_reciprocal_rank,
    # Alignment
    mean_pairwise_correlation as mean_pairwise_corr,
    variance_explained as var_explained,
    # Loss functions
    CosineLoss,
    CLIPDINOJointLoss,
    cosine_loss,
    mse_loss,
    combined_embedding_loss,
    reconstruction_loss,
    compute_evaluation_metrics,
)

__all__ = [
    # Legacy (backward compatibility)
    "cosine_similarity_matrix",
    "mean_cosine_similarity",
    "pixcorr",
    "ssim",
    "mse",
    "normalized_feature_correlation",
    "topk_accuracy",
    "topk_retrieval_accuracy",
    "mean_pairwise_correlation",
    "variance_explained",
    # Comprehensive evaluation
    "cosine_similarity",
    "cosine_sim_matrix",
    "mean_cos_sim",
    "norm_feature_corr",
    "pixel_correlation",
    "structural_similarity",
    "mean_squared_error",
    "rmse",
    "psnr",
    "top_k_accuracy",
    "top_k_retrieval_acc",
    "mean_reciprocal_rank",
    "mean_pairwise_corr",
    "var_explained",
    # Loss functions
    "CosineLoss",
    "CLIPDINOJointLoss",
    "cosine_loss",
    "mse_loss",
    "combined_embedding_loss",
    "reconstruction_loss",
    "compute_evaluation_metrics",
]
