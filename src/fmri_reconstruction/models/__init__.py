"""Neural network models for fMRI visual reconstruction.

Includes transformer encoders, multimodal models, feature projectors,
fusion modules, diffusion priors, and training losses.
"""

from .transformer import NeuralTransformer, NeuralTransformerRegressor
from .projectors import MLPProjector, CLIPProjector, DINOv2Projector
from .multimodal import MultimodalNeuralModel
from .fusion import FusionModule, MLP
from .diffusion_prior import DiffusionPriorMLP, DiffusionScheduler, DiffusionPrior, DualDiffusionPrior, diffusion_prior_loss
from .losses import CLIPDINOJointLoss, cosine_loss, mse_loss, combined_embedding_loss

__all__ = [
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
]
