import torch
import numpy as np
import tempfile
import os

from torch import optim

from fmri_reconstruction.models.multimodal import MultimodalNeuralModel
from fmri_reconstruction.models.fusion import FusionModule
from fmri_reconstruction.models.diffusion_prior import DualDiffusionPrior, diffusion_prior_loss
from fmri_reconstruction.training.trainer import SimpleTrainer
from fmri_reconstruction.training.checkpoint import save_checkpoint, load_checkpoint


def loss_fn(outputs, batch):
    # outputs: dict with 'clip' and 'dino'
    pred_clip = outputs.get("clip")
    pred_dino = outputs.get("dino")
    target_clip = batch.get("target_clip")
    target_dino = batch.get("target_dino")
    l = torch.tensor(0.0)
    if pred_clip is not None and target_clip is not None:
        l = l + torch.nn.functional.mse_loss(pred_clip, target_clip)
    if pred_dino is not None and target_dino is not None:
        l = l + torch.nn.functional.mse_loss(pred_dino, target_dino)
    return l


def test_trainer_one_step_and_checkpoint():
    B, L, D = 2, 20, 128
    x = torch.randn(B, L, D)

    model = MultimodalNeuralModel(input_dim=128, latent_dim=256, clip_dim=512, dino_dim=768)
    optim_ = optim.Adam(model.parameters(), lr=1e-3)
    trainer = SimpleTrainer(model, optim_)

    # create synthetic targets
    with torch.no_grad():
        out = model(x)
    target_clip = out["clip"]
    target_dino = out["dino"]

    batch = {"fmri": x, "target_clip": target_clip, "target_dino": target_dino}
    loss = trainer.train_step(batch, loss_fn)
    assert isinstance(loss, float)

    # checkpoint save/load
    td = tempfile.mkdtemp()
    ck_path = os.path.join(td, "ck.pt")
    save_checkpoint(model, optim_, ck_path, extra={"step": 1})
    extra = load_checkpoint(model, optim_, ck_path)
    assert extra.get("step") == 1
