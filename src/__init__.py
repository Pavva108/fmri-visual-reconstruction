"""fMRI visual reconstruction package.

Comprehensive toolkit for fMRI visual reconstruction using multi-modal features
(CLIP, DINOv2) with retrieval-based image generation and diffusion priors.
"""

__version__ = "0.1.0"

# Main module exports
from .fmri_reconstruction import (
    # Data
    SyntheticFMRIDataset,
    DatasetFactory,
    fetch_haxby_all,
    load_labels,
    remove_rest,
    extract_vt,
    # Evaluation
    cosine_similarity_matrix,
    mean_cosine_similarity,
    pixcorr,
    ssim,
    mse,
    topk_accuracy,
    topk_retrieval_accuracy,
    # Models
    NeuralTransformer,
    MultimodalNeuralModel,
    FusionModule,
    DiffusionPrior,
    # Retrieval
    RetrievalDatabase,
    cosine_scores,
    top_k,
)

__all__ = [
    # Version
    "__version__",
    # Data utilities
    "SyntheticFMRIDataset",
    "DatasetFactory",
    "fetch_haxby_all",
    "load_labels",
    "remove_rest",
    "extract_vt",
    # Evaluation utilities
    "cosine_similarity_matrix",
    "mean_cosine_similarity",
    "pixcorr",
    "ssim",
    "mse",
    "topk_accuracy",
    "topk_retrieval_accuracy",
    # Models
    "NeuralTransformer",
    "MultimodalNeuralModel",
    "FusionModule",
    "DiffusionPrior",
    # Retrieval
    "RetrievalDatabase",
    "cosine_scores",
    "top_k",
]
