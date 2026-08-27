import torch

from fmri_reconstruction.models.fusion import FusionModule


def _run_mode(mode: str):
    B = 2
    clip_dim = 512
    dino_dim = 768
    out_dim = 256

    z_clip = torch.randn(B, clip_dim)
    z_dino = torch.randn(B, dino_dim)

    f = FusionModule(clip_dim=clip_dim, dino_dim=dino_dim, out_dim=out_dim, mode=mode, hidden=128, attn_heads=4)
    out = f(z_clip, z_dino)
    assert out.shape == (B, out_dim)
    # simple backward
    loss = (out ** 2).mean()
    loss.backward()


def test_fusion_concat():
    _run_mode("concat")


def test_fusion_gated():
    _run_mode("gated")


def test_fusion_attention():
    _run_mode("attention")
