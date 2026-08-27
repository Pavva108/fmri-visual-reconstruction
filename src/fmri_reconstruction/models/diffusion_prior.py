from __future__ import annotations

import torch
from torch import nn


class DiffusionPriorMLP(nn.Module):
    """Feature-space denoising network used as a lightweight diffusion-prior block.

    This module is intentionally separated from the scheduler. For a faithful
    diffusion-prior reproduction, load the trained checkpoint and scheduler
    configuration used during training.
    """
    def __init__(self, dim=512, hidden=1024, time_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.net = nn.Sequential(
            nn.Linear(dim + time_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x, t):
        if t.ndim == 1:
            t = t[:, None]
        temb = self.time_mlp(t.float())
        return self.net(torch.cat([x, temb], dim=-1))


class DiffusionScheduler:
    """Simple linear beta scheduler for DDPM-style diffusion in feature space.

    Not a reproduction of any specific paper scheduler — hyperparameters are
    configurable and must be supplied via YAML when running experiments.
    """
    def __init__(self, timesteps: int = 1000, beta_start: float = 1e-6, beta_end: float = 1e-2):
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_cum = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Forward process: produce noisy x_t given x0 and noise at timesteps t."""
        # t: long tensor of shape (B,)
        a_bar = self.alpha_cum[t].to(x0.device).unsqueeze(-1)
        return torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * noise


class DiffusionPrior(nn.Module):
    """Wraps a denoising network and a scheduler for training the diffusion prior.

    The module predicts x0 from noisy x_t and timestep t. Training loss is
    computed externally to allow flexible objectives.
    """
    def __init__(self, dim: int = 512, hidden: int = 1024, time_dim: int = 128, timesteps: int = 1000):
        super().__init__()
        self.net = DiffusionPriorMLP(dim=dim, hidden=hidden, time_dim=time_dim)
        self.scheduler = DiffusionScheduler(timesteps=timesteps)

    def forward(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None) -> dict:
        """Return noisy x_t and the model's prediction of x0.

        Args:
            x0: clean features (B, dim)
            t: timesteps (B,) long
            noise: optional noise (B, dim); sampled if None
        Returns: dict with x_t, noise, pred_x0
        """
        if noise is None:
            noise = torch.randn_like(x0)
        x_t = self.scheduler.q_sample(x0, t, noise)
        pred = self.net(x_t, t.float())
        return {"x_t": x_t, "noise": noise, "pred_x0": pred}


class DualDiffusionPrior(nn.Module):
    """Container for two independent diffusion priors (CLIP and DINO).

    Each prior manages its own network and scheduler so hyperparameters may
    differ between CLIP and DINO branches.
    """
    def __init__(self, clip_dim: int = 512, dino_dim: int = 768, hidden: int = 1024, time_dim: int = 128, timesteps: int = 1000):
        super().__init__()
        self.clip_prior = DiffusionPrior(dim=clip_dim, hidden=hidden, time_dim=time_dim, timesteps=timesteps)
        self.dino_prior = DiffusionPrior(dim=dino_dim, hidden=hidden, time_dim=time_dim, timesteps=timesteps)

    def forward(self, z_clip: torch.Tensor, z_dino: torch.Tensor, t_clip: torch.Tensor, t_dino: torch.Tensor):
        out_clip = self.clip_prior(z_clip, t_clip)
        out_dino = self.dino_prior(z_dino, t_dino)
        return {"clip": out_clip, "dino": out_dino}


def diffusion_prior_loss(pred: torch.Tensor, target: torch.Tensor, model: nn.Module, lambda_cos: float | None = None, gamma: float | None = None) -> torch.Tensor:
    """Compute L = MSE + lambda * (1 - cosine) + gamma * L2(model params).

    - `pred` and `target` are tensors of same shape (B, dim)
    - `model` is used only to compute L2 regularization over parameters when gamma is set
    """
    mse = nn.functional.mse_loss(pred, target)
    total = mse
    if lambda_cos is not None:
        cos = nn.functional.cosine_similarity(pred, target, dim=-1, eps=1e-8).mean()
        total = total + float(lambda_cos) * (1.0 - cos)
    if gamma is not None and gamma != 0.0:
        sq = 0.0
        for p in model.parameters():
            sq = sq + (p.float() ** 2).sum()
        total = total + float(gamma) * sq
    return total

