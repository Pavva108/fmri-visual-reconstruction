"""Complete fMRI visual reconstruction pipeline.

Integrates all components: data loading, model inference, retrieval,
and diffusion-based image generation into a unified workflow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .data import preprocessing, load_stimuli_from_haxby
from .models import MultimodalNeuralModel, CLIPProjector, DINOv2Projector
from .retrieval import RetrievalDatabase, full_retrieval_pipeline
from .evaluation import compute_evaluation_metrics
from .reconstruction import StableDiffusionImg2Img


class fMRIVisualReconstructionPipeline:
    """End-to-end fMRI visual reconstruction pipeline.
    
    Orchestrates data loading, neural encoding, retrieval-based guidance,
    and diffusion-based image generation.
    """
    
    def __init__(
        self,
        model: MultimodalNeuralModel,
        clip_projector: CLIPProjector,
        dino_projector: DINOv2Projector,
        retrieval_database: RetrievalDatabase,
        diffusion_pipe: StableDiffusionImg2Img,
        device: str = "cuda",
    ):
        """Initialize reconstruction pipeline.
        
        Args:
            model: Trained fMRI encoder
            clip_projector: fMRI to CLIP projector
            dino_projector: fMRI to DINOv2 projector
            retrieval_database: Built retrieval database
            diffusion_pipe: Stable Diffusion pipeline
            device: Device to use
        """
        self.model = model.to(device)
        self.clip_projector = clip_projector.to(device)
        self.dino_projector = dino_projector.to(device)
        self.database = retrieval_database
        self.diffusion_pipe = diffusion_pipe
        self.device = device
        
        # Set to eval mode
        self.model.eval()
        self.clip_projector.eval()
        self.dino_projector.eval()
    
    def encode_fmri(self, fmri_sequence: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode fMRI sequence through neural model.
        
        Args:
            fmri_sequence: fMRI sequence (sequence_length, input_dim)
            
        Returns:
            Tuple of (latent, clip_embedding, dino_embedding)
        """
        x = torch.from_numpy(fmri_sequence).float().unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            latent = self.model.encoder(x)
            
            clip_emb = self.clip_projector(latent)
            clip_emb = F.normalize(clip_emb, dim=-1)
            
            dino_emb = self.dino_projector(latent)
            dino_emb = F.normalize(dino_emb, dim=-1)
        
        return latent, clip_emb, dino_emb
    
    def retrieve_guidance(
        self,
        fmri_latent: torch.Tensor,
        category: str,
        reference_image: Optional[str | Path] = None,
        use_structure: bool = False,
    ) -> Dict[str, Any]:
        """Retrieve guidance image and adaptive parameters.
        
        Args:
            fmri_latent: fMRI latent embedding
            category: Target category
            reference_image: Optional reference for structure matching
            use_structure: Use structure-based reranking
            
        Returns:
            Dictionary with retrieval results and diffusion parameters
        """
        return full_retrieval_pipeline(
            self.database,
            fmri_latent,
            category,
            self.clip_projector,
            self.dino_projector,
            reference_image=reference_image,
            use_structure=use_structure,
            device=self.device,
        )
    
    def generate_reconstruction(
        self,
        guidance: Dict[str, Any],
        num_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: int = 42,
    ) -> Image.Image:
        """Generate image using diffusion model with guidance.
        
        Args:
            guidance: Retrieval guidance dictionary
            num_steps: Number of diffusion steps (uses adaptive if None)
            guidance_scale: Classifier-free guidance scale (uses adaptive if None)
            seed: Random seed for reproducibility
            
        Returns:
            Generated image
        """
        # Load guidance image
        init_image = Image.open(guidance["image_path"]).convert("RGB")
        
        # Use adaptive parameters if not provided
        if num_steps is None:
            num_steps = guidance["num_inference_steps"]
        if guidance_scale is None:
            guidance_scale = guidance["guidance_scale"]
        
        strength = guidance["diffusion_strength"]
        prompt = guidance["text_prompt"]
        
        # Generate with diffusion
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        output = self.diffusion_pipe.generate(
            prompt=prompt,
            image=init_image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
        )
        
        return output
    
    def reconstruct_batch(
        self,
        fmri_sequences: np.ndarray,
        labels: np.ndarray,
        image_paths: list[str],
        use_structure: bool = False,
        save_dir: Optional[str | Path] = None,
    ) -> Dict[str, Any]:
        """Reconstruct multiple fMRI samples.
        
        Args:
            fmri_sequences: fMRI sequences (N, sequence_length, input_dim)
            labels: Category labels (N,)
            image_paths: Original image paths (N,)
            use_structure: Use structure-based retrieval
            save_dir: Directory to save results
            
        Returns:
            Dictionary with results for all samples
        """
        results = {
            "reconstructions": [],
            "retrieval_scores": [],
            "metadata": [],
        }
        
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        for i, (fmri_seq, label, img_path) in enumerate(
            zip(fmri_sequences, labels, image_paths)
        ):
            print(f"Processing sample {i+1}/{len(fmri_sequences)}: {label}")
            
            # Encode fMRI
            latent, clip_emb, dino_emb = self.encode_fmri(fmri_seq)
            
            # Retrieve guidance
            guidance = self.retrieve_guidance(
                latent,
                label,
                reference_image=img_path,
                use_structure=use_structure,
            )
            
            # Generate image
            try:
                recon_image = self.generate_reconstruction(guidance)
                
                results["reconstructions"].append(recon_image)
                results["retrieval_scores"].append({
                    "category": label,
                    "clip_score": guidance["clip_score"],
                    "dino_score": guidance["dino_score"],
                    "fusion_score": guidance["fusion_score"],
                })
                results["metadata"].append({
                    "index": i,
                    "category": label,
                    "retrieval_path": guidance["image_path"],
                    "prompt": guidance["text_prompt"],
                })
                
                # Save if requested
                if save_dir:
                    recon_image.save(save_dir / f"reconstruction_{i:03d}_{label}.png")
                
            except Exception as e:
                print(f"  Error generating reconstruction: {e}")
                results["reconstructions"].append(None)
                results["retrieval_scores"].append(None)
        
        return results
    
    def evaluate_batch(
        self,
        original_images: list[str | Path],
        reconstructed_images: list[Image.Image],
        metrics: Optional[list[str]] = None,
    ) -> Dict[str, float]:
        """Evaluate reconstruction quality.
        
        Args:
            original_images: Paths to original images
            reconstructed_images: Generated reconstruction images
            metrics: List of metrics to compute
            
        Returns:
            Dictionary of metric values
        """
        if metrics is None:
            metrics = ["ssim", "mse", "cosine_sim"]
        
        all_metrics = {m: [] for m in metrics}
        
        for orig_path, recon_image in zip(original_images, reconstructed_images):
            if recon_image is None:
                continue
            
            orig_image = np.array(Image.open(orig_path).convert("RGB"))
            recon_array = np.array(recon_image.convert("RGB"))
            
            sample_metrics = compute_evaluation_metrics(
                recon_array, orig_image, metrics=metrics
            )
            
            for metric, value in sample_metrics.items():
                all_metrics[metric].append(value)
        
        # Average across samples
        return {
            m: float(np.mean(v)) if v else 0.0
            for m, v in all_metrics.items()
        }


# =========================================================
# CONVENIENT INITIALIZATION FUNCTIONS
# =========================================================

def initialize_pipeline_from_checkpoints(
    model_checkpoint: str | Path,
    clip_checkpoint: str | Path,
    dino_checkpoint: str | Path,
    database_dir: str | Path,
    diffusion_model_id: str = "runwayml/stable-diffusion-v1-5",
    device: str = "cuda",
    input_dim: int = 128,
    latent_dim: int = 256,
) -> fMRIVisualReconstructionPipeline:
    """Initialize pipeline from checkpoint files.
    
    Args:
        model_checkpoint: Path to fMRI encoder checkpoint
        clip_checkpoint: Path to CLIP projector checkpoint
        dino_checkpoint: Path to DINOv2 projector checkpoint
        database_dir: Directory containing retrieval database
        diffusion_model_id: Hugging Face model ID for Stable Diffusion
        device: Device to use
        input_dim: fMRI input dimension
        latent_dim: Latent dimension
        
    Returns:
        Initialized pipeline
    """
    # Initialize models
    model = MultimodalNeuralModel(
        input_dim=input_dim,
        latent_dim=latent_dim,
    )
    clip_projector = CLIPProjector(latent_dim, 512)
    dino_projector = DINOv2Projector(latent_dim, 768)
    
    # Load checkpoints
    model.load_state_dict(torch.load(model_checkpoint, map_location=device))
    clip_projector.load_state_dict(torch.load(clip_checkpoint, map_location=device))
    dino_projector.load_state_dict(torch.load(dino_checkpoint, map_location=device))
    
    # Load retrieval database
    database = RetrievalDatabase.load(database_dir, device=device)
    
    # Initialize diffusion pipeline
    diffusion_pipe = StableDiffusionImg2Img(
        model_id=diffusion_model_id,
        device=device,
        disable_safety=True,
    )
    
    # Create and return pipeline
    return fMRIVisualReconstructionPipeline(
        model=model,
        clip_projector=clip_projector,
        dino_projector=dino_projector,
        retrieval_database=database,
        diffusion_pipe=diffusion_pipe,
        device=device,
    )


def build_retrieval_database_from_haxby(
    categories: Optional[list[str]] = None,
    clip_encoder: Optional[Any] = None,
    dino_encoder: Optional[Any] = None,
    device: str = "cuda",
) -> RetrievalDatabase:
    """Build retrieval database from Haxby stimuli.
    
    Args:
        categories: Categories to include (None = all Haxby categories)
        clip_encoder: CLIP encoder (uses default if None)
        dino_encoder: DINOv2 encoder (uses default if None)
        device: Device to use
        
    Returns:
        Built retrieval database
    """
    from .embeddings import CLIPImageEncoder, DINOv2ImageEncoder
    
    if categories is None:
        categories = preprocessing.HAXBY_CATEGORIES
    
    if clip_encoder is None:
        clip_encoder = CLIPImageEncoder(device=device)
    if dino_encoder is None:
        dino_encoder = DINOv2ImageEncoder(device=device)
    
    # Load Haxby stimuli
    category_images = load_stimuli_from_haxby(subjects=[1])
    
    # Build database
    database = RetrievalDatabase()
    
    for category in categories:
        if category not in category_images:
            print(f"Warning: {category} not in Haxby stimuli")
            continue
        
        print(f"\nProcessing category: {category}")
        database.add_category(category)
        
        image_paths = category_images[category]
        
        # Encode images in batches
        for i in range(0, len(image_paths), 32):
            batch_paths = image_paths[i:i+32]
            
            clip_embeddings = clip_encoder.encode_paths(batch_paths)
            dino_embeddings = dino_encoder.encode_paths(batch_paths)
            
            for path, clip_emb, dino_emb in zip(
                batch_paths, clip_embeddings, dino_embeddings
            ):
                database.add_image(
                    category,
                    path,
                    torch.from_numpy(clip_emb).float(),
                    torch.from_numpy(dino_emb).float(),
                )
    
    # Finalize database
    database.finalize(device=device)
    
    return database


__all__ = [
    "fMRIVisualReconstructionPipeline",
    "initialize_pipeline_from_checkpoints",
    "build_retrieval_database_from_haxby",
]
