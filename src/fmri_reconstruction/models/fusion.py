from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dims: list[int], out_dim: int, dropout: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = []
        cur = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(cur, h))
            layers.append(nn.GELU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            cur = h
        layers.append(nn.Linear(cur, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FusionModule(nn.Module):
    """Configurable fusion of CLIP and DINO features.

    Modes:
      - 'concat': concatenate features then MLP -> out_dim
      - 'gated': learn per-feature gate after projecting to `out_dim`
      - 'attention': project both to `out_dim`, run self-attention over two
        tokens, then average to produce fused vector.
    """

    def __init__(
        self,
        clip_dim: int = 512,
        dino_dim: int = 768,
        out_dim: int = 1024,
        mode: str = "concat",
        hidden: int = 512,
        attn_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert mode in ("concat", "gated", "attention"), "unsupported fusion mode"
        self.mode = mode
        self.out_dim = out_dim

        if mode == "concat":
            self.fuser = MLP(clip_dim + dino_dim, [hidden], out_dim, dropout=dropout)

        elif mode == "gated":
            # Project inputs to common dim
            self.clip_proj = nn.Linear(clip_dim, out_dim)
            self.dino_proj = nn.Linear(dino_dim, out_dim)
            self.gate = nn.Sequential(nn.Linear(clip_dim + dino_dim, hidden), nn.GELU(), nn.Linear(hidden, out_dim), nn.Sigmoid())

        else:  # attention
            self.clip_proj = nn.Linear(clip_dim, out_dim)
            self.dino_proj = nn.Linear(dino_dim, out_dim)
            self.attn = nn.MultiheadAttention(embed_dim=out_dim, num_heads=attn_heads, batch_first=True)

    def forward(self, z_clip: torch.Tensor, z_dino: torch.Tensor) -> torch.Tensor:
        """Return fused representation (B, out_dim)."""
        if self.mode == "concat":
            x = torch.cat([z_clip, z_dino], dim=-1)
            return self.fuser(x)

        if self.mode == "gated":
            c = self.clip_proj(z_clip)
            d = self.dino_proj(z_dino)
            g = self.gate(torch.cat([z_clip, z_dino], dim=-1))
            return g * c + (1.0 - g) * d

        # attention mode
        c = self.clip_proj(z_clip)
        d = self.dino_proj(z_dino)
        # stack as sequence length 2
        seq = torch.stack([c, d], dim=1)  # (B, 2, out_dim)
        attn_out, _ = self.attn(seq, seq, seq)
        # pool
        return attn_out.mean(dim=1)


__all__ = ["FusionModule"]
