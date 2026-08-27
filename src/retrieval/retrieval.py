"""Connect prepared fMRI input to retrieval guidance and generated output."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from fmri_reconstruction.retrieval import *
from fmri_reconstruction.retrieval import RetrievalDatabase, full_retrieval_pipeline


def retrieve_topk(
    database: RetrievalDatabase,
    category: str,
    clip_query: torch.Tensor,
    dino_query: torch.Tensor,
    k: int = 5,
) -> list[dict[str, Any]]:
    """Return ranked retrieval records with all component scores."""
    rows = database.retrieve_top_k(category, clip_query, dino_query, k=k)
    return [
        {"rank": rank, "image_path": path, "clip_score": clip, "dino_score": dino, "fusion_score": fusion}
        for rank, (path, clip, dino, fusion) in enumerate(rows, start=1)
    ]


def reconstruct_from_fmri(
    pipeline: Any,
    fmri_sequences: np.ndarray,
    labels: Sequence[str],
    image_paths: Sequence[str],
    output_dir: str | Path | None = None,
    use_structure: bool = False,
) -> dict[str, Any]:
    """Run the complete notebook flow from fMRI sequences to saved images.

    ``pipeline`` is an initialized ``fMRIVisualReconstructionPipeline``.  It
    is passed in so checkpoint and diffusion model selection stays explicit.
    """
    return pipeline.reconstruct_batch(
        np.asarray(fmri_sequences, dtype=np.float32),
        np.asarray(labels),
        list(image_paths),
        use_structure=use_structure,
        save_dir=output_dir,
    )


def retrieval_guidance(
    database: RetrievalDatabase,
    fmri_latent: torch.Tensor,
    category: str,
    clip_projector: torch.nn.Module,
    dino_projector: torch.nn.Module,
    reference_image: str | Path | None = None,
    use_structure: bool = False,
) -> dict[str, Any]:
    """Create top-1 guidance and adaptive diffusion parameters from a latent."""
    return full_retrieval_pipeline(
        database,
        fmri_latent,
        category,
        clip_projector,
        dino_projector,
        reference_image=reference_image,
        use_structure=use_structure,
    )


__all__ = [
    "RetrievalDatabase", "retrieve_topk", "retrieval_guidance", "reconstruct_from_fmri",
    "full_retrieval_pipeline", "get_category_weights", "get_text_prompt",
    "get_adaptive_strength", "get_adaptive_guidance", "get_adaptive_num_steps",
    "prepare_clip_query", "prepare_dino_query",
]