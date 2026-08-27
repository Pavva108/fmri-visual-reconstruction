import torch
import numpy as np

from fmri_reconstruction.models.transformer import NeuralTransformer
from fmri_reconstruction.models.projectors import CLIPProjector, DINOv2Projector
from fmri_reconstruction.models.losses import CLIPDINOJointLoss


def test_transformer_projectors_forward():
    # synthetic batch: batch_size x seq_len x input_dim
    B, L, D = 2, 20, 128
    x = torch.randn(B, L, D)

    # Transformer: input 128 -> hidden 256 (CLS output)
    tr = NeuralTransformer(input_dim=128, hidden_dim=256, layers=4, heads=8, ff_dim=512, dropout=0.1)
    h = tr(x)
    assert h.shape == (B, 256)

    # Projectors
    clip = CLIPProjector(in_dim=256, out_dim=512)
    dino = DINOv2Projector(in_dim=256, out_dim=768)
    z_clip = clip(h)
    z_dino = dino(h)
    assert z_clip.shape == (B, 512)
    assert z_dino.shape == (B, 768)

    # Loss (lambda_cos unspecified -> MSE only)
    target_clip = torch.randn_like(z_clip)
    loss_module = CLIPDINOJointLoss(lambda_cos=None)
    l = loss_module(z_clip, target_clip)
    assert float(l) >= 0.0

    # Loss with cosine term
    loss_module2 = CLIPDINOJointLoss(lambda_cos=0.5)
    l2 = loss_module2(z_clip, target_clip)
    assert float(l2) >= 0.0
