from __future__ import annotations

import torch
from torch import nn


class MLPProjector(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=1024, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class CLIPProjector(MLPProjector):
    def __init__(self, in_dim=256, out_dim=512):
        super().__init__(in_dim, out_dim)


class DINOv2Projector(MLPProjector):
    def __init__(self, in_dim=256, out_dim=768):
        super().__init__(in_dim, out_dim)
