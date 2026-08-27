from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def save_image_grid(images, titles=None, path=None, ncols=4, figsize=(16, 10)):
    images = list(images)
    nrows = int(np.ceil(len(images) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()

    for i, img in enumerate(images):
        if isinstance(img, (str, Path)):
            img = Image.open(img).convert("RGB")
        axes[i].imshow(img)
        axes[i].axis("off")
        if titles:
            axes[i].set_title(str(titles[i]))

    for ax in axes[len(images):]:
        ax.axis("off")

    fig.tight_layout()
    if path is not None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig


def plot_bold_slice(bold4d, time_index=0, z_index=None, path=None):
    data = np.asarray(bold4d)
    if data.ndim != 4:
        raise ValueError("Expected X×Y×Z×T BOLD data.")
    if z_index is None:
        z_index = data.shape[2] // 2

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(np.rot90(data[:, :, z_index, time_index]), cmap="gray")
    ax.set_title(f"t={time_index}, z={z_index}")
    ax.axis("off")

    if path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=200, bbox_inches="tight")
    return fig
