"""Pre-trained image encoders for CLIP and DINOv2 embeddings.

Provides efficient batch encoding of images to feature embeddings.
"""

from .clip import CLIPImageEncoder
from .dino import DINOv2ImageEncoder

__all__ = [
    "CLIPImageEncoder",
    "DINOv2ImageEncoder",
]
