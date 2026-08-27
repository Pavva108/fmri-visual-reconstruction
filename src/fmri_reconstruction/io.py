from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import nibabel as nib
import numpy as np


def load_nifti(path: str | Path, dtype=np.float32):
    """Load a NIfTI image and return the image plus data array."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    img = nib.load(str(path))
    data = img.get_fdata(dtype=dtype)
    return img, data


def load_npy(path: str | Path, allow_pickle: bool = True):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=allow_pickle)


def inspect_hdf5(path: str | Path) -> List[Tuple[str, str, str]]:
    """Return (name, shape, dtype) records for HDF5 datasets."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    records = []

    def visitor(name, obj):
        if hasattr(obj, "shape"):
            records.append((name, str(obj.shape), str(obj.dtype)))

    with h5py.File(path, "r") as f:
        f.visititems(visitor)
    return records


def load_hdf5_dataset(path: str | Path, dataset_key: str) -> np.ndarray:
    path = Path(path)
    with h5py.File(path, "r") as f:
        if dataset_key not in f:
            raise KeyError(f"{dataset_key!r} not found in {path}")
        return f[dataset_key][...]


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(__import__("json").dumps(obj, indent=2, default=str), encoding="utf-8")


def save_npz(path: str | Path, **arrays) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
