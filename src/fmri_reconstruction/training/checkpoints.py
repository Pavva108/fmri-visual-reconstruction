from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import torch


def save_model_checkpoint(model, path: str | Path, optimizer=None, scheduler=None, **metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "metadata": metadata,
        },
        path,
    )


def load_model_checkpoint(
    model,
    path: str | Path,
    device="cpu",
    optimizer=None,
    scheduler=None,
    strict=True,
):
    checkpoint = torch.load(path, map_location=device)

    state = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))
    model.load_state_dict(state, strict=strict)

    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return checkpoint
