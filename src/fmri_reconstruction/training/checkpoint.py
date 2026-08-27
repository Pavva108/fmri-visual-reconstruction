from __future__ import annotations

import torch
from pathlib import Path
from typing import Any


def save_checkpoint(model: torch.nn.Module, optimizer: Any, path: str | Path, extra: dict | None = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"model_state": model.state_dict()}
    if optimizer is not None:
        state["optim_state"] = optimizer.state_dict()
    if extra:
        state["extra"] = extra
    torch.save(state, str(path))


def load_checkpoint(model: torch.nn.Module, optimizer: Any, path: str | Path) -> dict:
    path = Path(path)
    ck = torch.load(str(path), map_location="cpu")
    model.load_state_dict(ck["model_state"])
    if optimizer is not None and "optim_state" in ck:
        optimizer.load_state_dict(ck["optim_state"])
    return ck.get("extra", {})
