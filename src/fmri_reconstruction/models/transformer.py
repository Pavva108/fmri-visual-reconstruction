from __future__ import annotations

import torch
from torch import nn


class NeuralTransformer(nn.Module):
    def __init__(
        self,
        input_dim=128,
        hidden_dim=256,
        layers=4,
        heads=8,
        ff_dim=512,
        dropout=0.1,
    ):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        z = self.input_projection(x)
        cls = self.cls_token.expand(z.size(0), -1, -1)
        z = torch.cat([cls, z], dim=1)
        return self.norm(self.encoder(z)[:, 0])


class NeuralTransformerRegressor(nn.Module):
    def __init__(self, input_dim=128, hidden_dim=256, output_dim=512, **kwargs):
        super().__init__()
        self.encoder = NeuralTransformer(input_dim, hidden_dim, **kwargs)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.head(self.encoder(x))
