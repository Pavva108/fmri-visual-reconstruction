#!/usr/bin/env python
"""Build a retrieval database from TRAINING stimulus images only."""

import argparse
from pathlib import Path

from fmri_reconstruction.embeddings.clip import CLIPImageEncoder
from fmri_reconstruction.retrieval.database import RetrievalDatabase


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stimulus-root", required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    root = Path(args.stimulus_root)
    paths = sorted([
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ])

    if not paths:
        raise RuntimeError("No stimulus images found.")

    encoder = CLIPImageEncoder()
    embeddings = encoder.encode_paths(paths)

    db = RetrievalDatabase(
        embeddings=embeddings,
        paths=[str(p) for p in paths],
        metadata={"split": "train_only", "encoder": "CLIP"},
    ).normalize()

    db.save(args.output)
    print(f"Saved {len(paths)} images to {args.output}")


if __name__ == "__main__":
    main()
