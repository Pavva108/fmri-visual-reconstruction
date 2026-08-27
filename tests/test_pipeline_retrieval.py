import torch
import numpy as np

from fmri_reconstruction.models.multimodal import MultimodalNeuralModel
from fmri_reconstruction.models.fusion import FusionModule
from fmri_reconstruction.retrieval.interface import build_db, guarded_retrieve


def test_pipeline_model_fusion_retrieval():
    # synthetic fmri -> latent -> clip/dino -> fused -> retrieval
    B, L, D = 2, 20, 128
    x = torch.randn(B, L, D)

    model = MultimodalNeuralModel(input_dim=128, latent_dim=256, clip_dim=512, dino_dim=768)
    out = model(x)
    z_clip = out["clip"]
    z_dino = out["dino"]

    fuse = FusionModule(clip_dim=512, dino_dim=768, out_dim=512, mode="concat", hidden=256)
    fused = fuse(z_clip, z_dino)
    assert fused.shape == (B, 512)

    # build a retrieval DB of random embeddings in the same dim
    rng = np.random.RandomState(0)
    db_emb = rng.randn(50, 512).astype(np.float32)
    paths = [f"img_{i}.jpg" for i in range(50)]
    db = build_db(db_emb, paths)

    # guarded retrieve (train/test split)
    results = guarded_retrieve(fused.detach().numpy(), db, train_paths=paths[:40], test_paths=paths[40:], k=5)
    assert isinstance(results, list) and len(results) == B
    idx, scores = results[0]
    assert len(idx) == 5
