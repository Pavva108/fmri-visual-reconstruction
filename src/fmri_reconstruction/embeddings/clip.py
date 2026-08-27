from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch
from PIL import Image


class CLIPImageEncoder:
    def __init__(self, model_name="ViT-B-32", pretrained="openai", device=None):
        import open_clip

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model.eval().to(self.device)

    @torch.no_grad()
    def encode_paths(self, paths: Iterable[str | Path], batch_size=32) -> np.ndarray:
        paths = list(paths)
        out = []
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start:start + batch_size]
            imgs = torch.stack([
                self.preprocess(Image.open(p).convert("RGB"))
                for p in batch_paths
            ]).to(self.device)
            feat = self.model.encode_image(imgs)
            feat = feat / feat.norm(dim=-1, keepdim=True)
            out.append(feat.cpu().numpy())
        return np.concatenate(out, axis=0)
