from __future__ import annotations

import torch
from torch import nn
from torch.optim import Optimizer
from typing import Optional


class SimpleTrainer:
    """Very small training scaffold used for synthetic tests and quick experiments.

    Responsibilities:
      - run a single training step (forward, loss, backward, step)
      - support optional multiple optimizers via a simple API
    This is intentionally minimal; use as scaffolding for full training loops.
    """

    def __init__(self, model: nn.Module, optimizer: Optimizer, device: Optional[torch.device] = None):
        self.model = model
        self.optimizer = optimizer
        self.device = device or torch.device("cpu")
        self.model.to(self.device)

    def train_step(self, batch: dict, loss_fn) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        # model forward should accept batch dict keys and return dict of outputs
        outputs = self.model(**batch)
        loss = loss_fn(outputs, batch)
        loss.backward()
        self.optimizer.step()
        return float(loss.detach().cpu().item())
