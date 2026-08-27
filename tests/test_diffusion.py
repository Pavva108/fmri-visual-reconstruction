import torch
from fmri_reconstruction.models.diffusion_prior import DualDiffusionPrior, diffusion_prior_loss


def test_dual_diffusion_forward_and_loss():
    B = 2
    clip_dim = 512
    dino_dim = 768
    device = torch.device("cpu")

    z_clip = torch.randn(B, clip_dim, device=device)
    z_dino = torch.randn(B, dino_dim, device=device)

    # small timesteps for test
    t_clip = torch.randint(0, 10, (B,))
    t_dino = torch.randint(0, 10, (B,))

    model = DualDiffusionPrior(clip_dim=clip_dim, dino_dim=dino_dim, hidden=128, time_dim=16, timesteps=10)
    out = model(z_clip, z_dino, t_clip, t_dino)
    assert "clip" in out and "dino" in out
    pred_clip = out["clip"]["pred_x0"]
    pred_dino = out["dino"]["pred_x0"]
    assert pred_clip.shape == (B, clip_dim)
    assert pred_dino.shape == (B, dino_dim)

    # compute losses
    l_clip = diffusion_prior_loss(pred_clip, z_clip, model.clip_prior.net, lambda_cos=0.1, gamma=1e-4)
    l_dino = diffusion_prior_loss(pred_dino, z_dino, model.dino_prior.net, lambda_cos=0.1, gamma=1e-4)
    assert float(l_clip) >= 0.0
    assert float(l_dino) >= 0.0
