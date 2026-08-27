from __future__ import annotations

from pathlib import Path
import json

import numpy as np


class RetrievalDatabase:
    def __init__(self, embeddings=None, paths=None, metadata=None):
        self.embeddings = None if embeddings is None else np.asarray(embeddings, dtype=np.float32)
        self.paths = list(paths or [])
        self.metadata = metadata or {}

    def normalize(self):
        if self.embeddings is None:
            raise RuntimeError("No embeddings loaded.")
        n = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / (n + 1e-8)
        return self

    def save(self, directory: str | Path):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "embeddings.npy", self.embeddings)
        (directory / "paths.json").write_text(json.dumps(self.paths, indent=2), encoding="utf-8")
        (directory / "metadata.json").write_text(json.dumps(self.metadata, indent=2, default=str), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path):
        directory = Path(directory)
        embeddings = np.load(directory / "embeddings.npy")
        paths = json.loads((directory / "paths.json").read_text(encoding="utf-8"))
        metadata_path = directory / "metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
        return cls(embeddings, paths, metadata).normalize()
