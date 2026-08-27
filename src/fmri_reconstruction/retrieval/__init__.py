"""Retrieval utilities for fMRI visual reconstruction.

Includes embedding-based retrieval, cosine similarity search, database management,
data leakage guards, and adaptive diffusion parameter selection.
"""

# Legacy imports
from .database import RetrievalDatabase as LegacyRetrievalDatabase
from .cosine import l2_normalize, cosine_scores, top_k, retrieve_batch
from .interface import build_db, guarded_retrieve
from .split_guard import assert_no_test_overlap, assert_subject_split

# New comprehensive retrieval module
from .retrieval import (
    HAXBY_CATEGORIES,
    SHAPE_FOCUS_CATEGORIES,
    get_category_weights,
    get_text_prompt,
    RetrievalDatabase,
    compute_edge_similarity,
    retrieve_with_structure_score,
    get_adaptive_strength,
    get_adaptive_guidance,
    get_adaptive_num_steps,
    prepare_clip_query,
    prepare_dino_query,
    full_retrieval_pipeline,
)

__all__ = [
    # Legacy (backward compatibility)
    "LegacyRetrievalDatabase",
    "l2_normalize",
    "cosine_scores",
    "top_k",
    "retrieve_batch",
    "build_db",
    "guarded_retrieve",
    "assert_no_test_overlap",
    "assert_subject_split",
    # New comprehensive retrieval
    "HAXBY_CATEGORIES",
    "SHAPE_FOCUS_CATEGORIES",
    "get_category_weights",
    "get_text_prompt",
    "RetrievalDatabase",
    "compute_edge_similarity",
    "retrieve_with_structure_score",
    "get_adaptive_strength",
    "get_adaptive_guidance",
    "get_adaptive_num_steps",
    "prepare_clip_query",
    "prepare_dino_query",
    "full_retrieval_pipeline",
]
