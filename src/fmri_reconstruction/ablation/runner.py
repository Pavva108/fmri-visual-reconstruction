from __future__ import annotations

from typing import Dict, Optional

import torch


def run_model_ablation(
    model,
    loader,
    device="cuda",
    use_clip=True,
    use_dino=True,
    use_retrieval=True,
):
    """Evaluate a model variant when its forward() accepts these switches.

    This function intentionally does not fabricate metrics. The caller must
    provide the actual trained model, data loader and target/evaluation code.
    """
    model.eval()
    outputs = []

    with torch.no_grad():
        for batch in loader:
            x = batch["fmri"].to(device)
            out = model(
                x,
                use_clip=use_clip,
                use_dino=use_dino,
                use_retrieval=use_retrieval,
            )
            outputs.append(out)

    return outputs
