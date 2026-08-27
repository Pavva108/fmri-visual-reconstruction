"""Reusable components for subject-agnostic fMRI visual reconstruction.

Comprehensive toolkit featuring:
- Data loading and preprocessing for Haxby, NSD, and LIVE datasets
- Multi-modal neural models (CLIP, DINOv2)
- Retrieval-based image generation with adaptive parameters
- Diffusion priors and image reconstruction
- Comprehensive evaluation metrics and loss functions
- End-to-end reconstruction pipeline
"""

__version__ = "0.1.0"

# Data utilities (including preprocessing)
from .data import (
    SyntheticFMRIDataset,
    DatasetFactory,
    fetch_haxby_all,
    load_labels,
    remove_rest,
    extract_vt,
    HAXBYY_CATEGORIES,
    resolve_data_root,
    haxby_paths,
    nsd_paths,
    live_paths,
    make_sequences,
    align_labels_to_sequences,
    # New preprocessing functions
    normalize_label,
    normalize_labels,
    load_fmri_data,
    load_stimuli_from_haxby,
    build_stimulus_paths,
    create_sequences,
    get_clip_preprocessor,
    get_dinov2_preprocessor,
    preprocess_image_batch,
    train_test_split,
    validate_data_shapes,
)

# Evaluation utilities (including comprehensive evaluation functions and loss functions)
from .evaluation import (
    cosine_similarity_matrix,
    mean_cosine_similarity,
    pixcorr,
    ssim,
    mse,
    normalized_feature_correlation,
    topk_accuracy,
    topk_retrieval_accuracy,
    mean_pairwise_correlation,
    variance_explained,
    # New comprehensive evaluation
    cosine_similarity,
    rmse,
    psnr,
    mean_reciprocal_rank,
    CosineLoss,
    CLIPDINOJointLoss as CLIPDINOJointLossEval,
    cosine_loss,
    mse_loss,
    combined_embedding_loss,
    reconstruction_loss,
    compute_evaluation_metrics,
)

# Models
from .models import (
    NeuralTransformer,
    NeuralTransformerRegressor,
    MLPProjector,
    CLIPProjector,
    DINOv2Projector,
    MultimodalNeuralModel,
    FusionModule,
    MLP,
    DiffusionPriorMLP,
    DiffusionScheduler,
    DiffusionPrior,
    DualDiffusionPrior,
    diffusion_prior_loss,
    CLIPDINOJointLoss,
    cosine_loss,
    mse_loss,
    combined_embedding_loss,
)

# Retrieval (including comprehensive retrieval system)
from .retrieval import (
    RetrievalDatabase,
    l2_normalize,
    cosine_scores,
    top_k,
    retrieve_batch,
    build_db,
    guarded_retrieve,
    assert_no_test_overlap,
    assert_subject_split,
    # New comprehensive retrieval
    HAXBY_CATEGORIES,
    SHAPE_FOCUS_CATEGORIES,
    get_category_weights,
    get_text_prompt,
    compute_edge_similarity,
    retrieve_with_structure_score,
    get_adaptive_strength,
    get_adaptive_guidance,
    get_adaptive_num_steps,
    prepare_clip_query,
    prepare_dino_query,
    full_retrieval_pipeline,
)

# Reconstruction
from .reconstruction import StableDiffusionImg2Img

# Embeddings
from .embeddings import CLIPImageEncoder, DINOv2ImageEncoder

# Pipeline (end-to-end reconstruction)
from .pipeline import (
    fMRIVisualReconstructionPipeline,
    initialize_pipeline_from_checkpoints,
    build_retrieval_database_from_haxby,
)

# Training
from .training import (
    SimpleTrainer,
    train_one_epoch,
    evaluate,
    save_checkpoint,
    load_checkpoint,
    save_model_checkpoint,
    load_model_checkpoint,
)

# Ablation
from .ablation import (
    run_model_ablation,
    AblationResult,
    run_ablation,
    standard_reconstruction_ablations,
    save_ablation_csv,
)

__all__ = [
    "__version__",
    # Data
    "SyntheticFMRIDataset",
    "DatasetFactory",
    "fetch_haxby_all",
    "load_labels",
    "remove_rest",
    "extract_vt",
    "HAXBYY_CATEGORIES",
    "resolve_data_root",
    "haxby_paths",
    "nsd_paths",
    "live_paths",
    "make_sequences",
    "align_labels_to_sequences",
    # Data preprocessing (new)
    "normalize_label",
    "normalize_labels",
    "load_fmri_data",
    "load_stimuli_from_haxby",
    "build_stimulus_paths",
    "create_sequences",
    "get_clip_preprocessor",
    "get_dinov2_preprocessor",
    "preprocess_image_batch",
    "train_test_split",
    "validate_data_shapes",
    # Evaluation
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
    # Evaluation comprehensive (new)
    "cosine_similarity",
    "rmse",
    "psnr",
    "mean_reciprocal_rank",
    "CosineLoss",
    "CLIPDINOJointLossEval",
    "cosine_loss",
    "mse_loss",
    "combined_embedding_loss",
    "reconstruction_loss",
    "compute_evaluation_metrics",
    # Models
    "NeuralTransformer",
    "NeuralTransformerRegressor",
    "MLPProjector",
    "CLIPProjector",
    "DINOv2Projector",
    "MultimodalNeuralModel",
    "FusionModule",
    "MLP",
    "DiffusionPriorMLP",
    "DiffusionScheduler",
    "DiffusionPrior",
    "DualDiffusionPrior",
    "diffusion_prior_loss",
    "CLIPDINOJointLoss",
    "cosine_loss",
    "mse_loss",
    "combined_embedding_loss",
    # Retrieval
    "RetrievalDatabase",
    "l2_normalize",
    "cosine_scores",
    "top_k",
    "retrieve_batch",
    "build_db",
    "guarded_retrieve",
    "assert_no_test_overlap",
    "assert_subject_split",
    # Retrieval comprehensive (new)
    "HAXBY_CATEGORIES",
    "SHAPE_FOCUS_CATEGORIES",
    "get_category_weights",
    "get_text_prompt",
    "compute_edge_similarity",
    "retrieve_with_structure_score",
    "get_adaptive_strength",
    "get_adaptive_guidance",
    "get_adaptive_num_steps",
    "prepare_clip_query",
    "prepare_dino_query",
    "full_retrieval_pipeline",
    # Reconstruction
    "StableDiffusionImg2Img",
    # Embeddings
    "CLIPImageEncoder",
    "DINOv2ImageEncoder",
    # Pipeline (new)
    "fMRIVisualReconstructionPipeline",
    "initialize_pipeline_from_checkpoints",
    "build_retrieval_database_from_haxby",
    # Training
    "SimpleTrainer",
    "train_one_epoch",
    "evaluate",
    "save_checkpoint",
    "load_checkpoint",
    "save_model_checkpoint",
    "load_model_checkpoint",
    # Ablation
    "run_model_ablation",
    "AblationResult",
    "run_ablation",
    "standard_reconstruction_ablations",
    "save_ablation_csv",
]
