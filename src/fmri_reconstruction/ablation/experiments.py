from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np


@dataclass
class AblationResult:
    name: str
    metrics: Dict[str, float]
    notes: str = ""

    def to_dict(self):
        return asdict(self)


def run_ablation(
    experiment_fn: Callable[..., Dict[str, float]],
    variants: Iterable[Dict[str, Any]],
) -> List[AblationResult]:
    """Run named ablations using the same evaluation protocol."""
    results = []
    for variant in variants:
        name = variant.pop("name")
        notes = variant.pop("notes", "")
        metrics = experiment_fn(**variant)
        results.append(AblationResult(name=name, metrics=metrics, notes=notes))
    return results


def standard_reconstruction_ablations():
    """Variants matching the project's conceptual CLIP/retrieval ablation."""
    return [
        {
            "name": "full_model",
            "use_clip": True,
            "use_dino": True,
            "use_retrieval": True,
            "notes": "CLIP + DINOv2 + retrieval.",
        },
        {
            "name": "without_clip",
            "use_clip": False,
            "use_dino": True,
            "use_retrieval": True,
            "notes": "Remove CLIP branch.",
        },
        {
            "name": "without_dino",
            "use_clip": True,
            "use_dino": False,
            "use_retrieval": True,
            "notes": "Remove DINOv2 branch.",
        },
        {
            "name": "without_retrieval",
            "use_clip": True,
            "use_dino": True,
            "use_retrieval": False,
            "notes": "Remove retrieval guidance.",
        },
        {
            "name": "clip_only",
            "use_clip": True,
            "use_dino": False,
            "use_retrieval": False,
            "notes": "CLIP-only semantic branch.",
        },
    ]


def save_ablation_csv(results: List[AblationResult], path):
    import pandas as pd

    rows = []
    for result in results:
        row = {"variant": result.name, "notes": result.notes}
        row.update(result.metrics)
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)
