from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import pandas as pd


def load_stimulus_table(path: str | Path, sheet_name=0):
    return pd.read_excel(path, sheet_name=sheet_name)


def resolve_image_paths(
    table: pd.DataFrame,
    stimulus_root: str | Path,
    path_column: str,
) -> list[Path]:
    root = Path(stimulus_root)
    if path_column not in table.columns:
        raise KeyError(f"Column {path_column!r} not found. Available: {list(table.columns)}")

    paths = []
    for value in table[path_column].astype(str):
        p = Path(value)
        paths.append(p if p.is_absolute() else root / p)
    return paths


def verify_paths(paths: Iterable[Path]) -> None:
    missing = [str(p) for p in paths if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} stimulus paths are missing. First examples: {missing[:10]}"
        )
