from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image


class DINOv2ImageEncoder:
    def __init__(self, model_name="facebook/dinov2-base", device=None):
        from transformers import AutoImageProcessor, AutoModel

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).eval().to(self.device)

    @torch.no_grad()
    def encode_paths(self, paths: Iterable[str | Path], batch_size=16) -> np.ndarray:
        paths = list(paths)
        out = []
        for start in range(0, len(paths), batch_size):
            batch_paths = paths[start:start + batch_size]
            imgs = [Image.open(p).convert("RGB") for p in batch_paths]
            inputs = self.processor(images=imgs, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            feat = outputs.last_hidden_state[:, 0]
            feat = feat / feat.norm(dim=-1, keepdim=True)
            out.append(feat.cpu().numpy())
        return np.concatenate(out, axis=0)
