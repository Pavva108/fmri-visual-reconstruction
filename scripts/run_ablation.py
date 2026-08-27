#!/usr/bin/env python
"""Run a user-supplied ablation evaluator.

This script creates the standard ablation manifest. It does not invent or
report results; execute each variant with the actual trained checkpoints and
evaluation pipeline.
"""

import argparse
from pathlib import Path
import json

from fmri_reconstruction.ablation.experiments import standard_reconstruction_ablations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/results/ablation_manifest.json")
    args = parser.parse_args()

    variants = standard_reconstruction_ablations()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(variants, indent=2), encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
