from __future__ import annotations

from typing import Dict

import torch


def train_one_epoch(model, loader, optimizer, loss_fn, device="cuda") -> float:
    model.train()
    total = 0.0
    count = 0

    for batch in loader:
        x = batch["fmri"].to(device)
        target = batch["target"].to(device)

        optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = loss_fn(pred, target)
        loss.backward()
        optimizer.step()

        bs = x.shape[0]
        total += float(loss.detach()) * bs
        count += bs

    return total / max(count, 1)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device="cuda") -> float:
    model.eval()
    total = 0.0
    count = 0

    for batch in loader:
        x = batch["fmri"].to(device)
        target = batch["target"].to(device)
        pred = model(x)
        loss = loss_fn(pred, target)

        bs = x.shape[0]
        total += float(loss) * bs
        count += bs

    return total / max(count, 1)


def save_checkpoint(model, optimizer, epoch, path, extra=None):
    payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "extra": extra or {},
    }
    torch.save(payload, path)
