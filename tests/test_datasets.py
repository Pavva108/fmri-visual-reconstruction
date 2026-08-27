import torch
from fmri_reconstruction.config import ExperimentConfig
from fmri_reconstruction.data.datasets import DatasetFactory, SyntheticFMRIDataset


def test_synthetic_dataset():
    ds = SyntheticFMRIDataset(length=10, sequence_length=20, input_dim=128, seed=0)
    assert len(ds) == 10
    sample = ds[0]
    assert "fmri" in sample and sample["fmri"].shape == (20, 128)


def test_factory_synthetic_config():
    cfg = ExperimentConfig(dataset="synthetic", train_subjects=[1], val_subjects=[], test_subjects=[], aligned_dim=128, sequence_length=20, latent_dim=256)
    ds = DatasetFactory.from_config(cfg)
    assert isinstance(ds, SyntheticFMRIDataset)
