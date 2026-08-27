from __future__ import annotations

import torch
from torch import nn

from .projectors import CLIPProjector, DINOv2Projector
from .transformer import NeuralTransformer


class MultimodalNeuralModel(nn.Module):
    """Shared neural representation followed by CLIP and DINOv2 heads."""

    def __init__(
        self,
        input_dim=128,
        latent_dim=256,
        clip_dim=512,
        dino_dim=768,
        transformer_layers=4,
        attention_heads=8,
        ff_dim=512,
        dropout=0.1,
    ):
        super().__init__()
        self.encoder = NeuralTransformer(
            input_dim=input_dim,
            hidden_dim=latent_dim,
            layers=transformer_layers,
            heads=attention_heads,
            ff_dim=ff_dim,
            dropout=dropout,
        )
        self.clip_head = CLIPProjector(latent_dim, clip_dim)
        self.dino_head = DINOv2Projector(latent_dim, dino_dim)

    def forward(
        self,
        fmri,
        use_clip=True,
        use_dino=True,
        return_latent=False,
        use_retrieval=True,  # retained for ablation API compatibility
    ):
        latent = self.encoder(fmri)
        out = {"latent": latent}

        if use_clip:
            out["clip"] = self.clip_head(latent)

        if use_dino:
            out["dino"] = self.dino_head(latent)

        if not return_latent:
            out.pop("latent", None)

        return out
