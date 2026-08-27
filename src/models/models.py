"""Public model registry backed by the tested ``fmri_reconstruction`` modules."""

from fmri_reconstruction.models import (
    CLIPDINOJointLoss,
    CLIPProjector,
    DINOv2Projector,
    DiffusionPrior,
    DiffusionPriorMLP,
    DiffusionScheduler,
    DualDiffusionPrior,
    FusionModule,
    MLP,
    MLPProjector,
    MultimodalNeuralModel,
    NeuralTransformer,
    NeuralTransformerRegressor,
    combined_embedding_loss,
    cosine_loss,
    diffusion_prior_loss,
    mse_loss,
)


def build_model(
    input_dim: int = 128,
    latent_dim: int = 256,
    clip_dim: int = 512,
    dino_dim: int = 768,
) -> MultimodalNeuralModel:
    """Build the shared Transformer plus CLIP/DINO prediction heads."""
    return MultimodalNeuralModel(
        input_dim=input_dim,
        latent_dim=latent_dim,
        clip_dim=clip_dim,
        dino_dim=dino_dim,
    )


__all__ = [
    "NeuralTransformer", "NeuralTransformerRegressor", "MLPProjector", "CLIPProjector",
    "DINOv2Projector", "MultimodalNeuralModel", "FusionModule", "MLP",
    "DiffusionPriorMLP", "DiffusionScheduler", "DiffusionPrior", "DualDiffusionPrior",
    "diffusion_prior_loss", "CLIPDINOJointLoss", "cosine_loss", "mse_loss",
    "combined_embedding_loss", "build_model",
]