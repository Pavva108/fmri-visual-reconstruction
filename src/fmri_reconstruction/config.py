from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class ExperimentConfig:
    dataset: str
    train_subjects: List[int]
    val_subjects: List[int]
    test_subjects: List[int]

    tr_seconds: float = 3.0
    sequence_length: int = 20
    aligned_dim: int = 128
    latent_dim: int = 256

    transformer_layers: int = 4
    attention_heads: int = 8
    ff_dim: int = 512
    dropout: float = 0.1

    clip_dim: int = 512
    dino_dim: int = 768

    retrieval_top_k: int = 5
    seed: int = 42
    device: str = "cuda"

    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(path: str | Path) -> ExperimentConfig:
    raw = load_yaml(path)
    known = {k: raw[k] for k in ExperimentConfig.__dataclass_fields__ if k in raw and k != "extra"}
    extra = {k: v for k, v in raw.items() if k not in known}
    known["extra"] = extra
    return ExperimentConfig(**known)
