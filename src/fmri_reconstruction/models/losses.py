from __future__ import annotations

import torch
from torch import nn


class CLIPDINOJointLoss(nn.Module):
    """Combined MSE + cosine loss used for CLIP/DINO projectors.

    L = MSE(pred, target) + lambda_cos * (1 - cosine_similarity(pred, target)).
    The weight `lambda_cos` is intentionally configurable and should be set
    via experiment YAML when reproducing paper experiments.
    """

    def __init__(self, lambda_cos: float | None = None):
        super().__init__()
        self.mse = nn.MSELoss()
        self.lambda_cos = lambda_cos
        self.cos = nn.CosineSimilarity(dim=-1, eps=1e-8)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mse_l = self.mse(pred, target)
        if self.lambda_cos is None:
            return mse_l
        cos_sim = self.cos(pred, target).mean()
        cos_l = 1.0 - cos_sim
        return mse_l + float(self.lambda_cos) * cos_l

    import torch.nn.functional as F


def cosine_loss(pred, target):
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    return (1.0 - (pred * target).sum(dim=-1)).mean()


def mse_loss(pred, target):
    return F.mse_loss(pred, target)


def combined_embedding_loss(pred_clip, target_clip, pred_dino=None, target_dino=None, dino_weight=1.0):
    loss = cosine_loss(pred_clip, target_clip)
    if pred_dino is not None and target_dino is not None:
        loss = loss + dino_weight * cosine_loss(pred_dino, target_dino)
    return loss
