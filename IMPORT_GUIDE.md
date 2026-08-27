# fMRI Visual Reconstruction - Quick Import Guide

## Complete Module Import Reference

### Full Package Import
```python
import fmri_reconstruction as fmri

# Access all modules
model = fmri.MultimodalNeuralModel()
db = fmri.RetrievalDatabase()
trainer = fmri.SimpleTrainer(model, optimizer)
```

---

## Module-Specific Imports

### 1. Data Loading Module
```python
from fmri_reconstruction.data import (
    # Dataset classes
    SyntheticFMRIDataset,           # Synthetic test data
    DatasetFactory,                 # Resolve datasets by name
    
    # Haxby dataset
    fetch_haxby_all,                # Download Haxby from nilearn
    load_labels,                    # Load stimulus labels
    remove_rest,                    # Filter rest labels
    extract_vt,                     # Extract voxel time series
    HAXBYY_CATEGORIES,              # Available categories
    
    # Data paths
    resolve_data_root,              # Find dataset directory
    haxby_paths,                    # Resolve Haxby paths
    nsd_paths,                      # Resolve NSD paths
    live_paths,                     # Resolve LIVE paths
    
    # NSD dataset
    download_nsd_files,             # Download from Hugging Face
    load_annotation,                # Load NSD annotations
    
    # LIVE dataset
    load_live_bold,                 # Load NIfTI BOLD data
    read_stimulus_excel,            # Load stimulus metadata
    list_stimulus_images,           # Find stimulus files
    
    # Sequence utilities
    make_sequences,                 # Create time windows
    align_labels_to_sequences,      # Align labels to windows
)
```

### 2. Models Module
```python
from fmri_reconstruction.models import (
    # Core architectures
    NeuralTransformer,              # Transformer encoder
    NeuralTransformerRegressor,     # Transformer + regression head
    MultimodalNeuralModel,          # Dual-head model (CLIP + DINOv2)
    
    # Projectors
    MLPProjector,                   # Generic MLP projector
    CLIPProjector,                  # fMRI → CLIP space
    DINOv2Projector,                # fMRI → DINOv2 space
    
    # Fusion
    FusionModule,                   # Feature fusion strategies
    MLP,                            # Flexible MLP
    
    # Diffusion
    DiffusionPrior,                 # Single diffusion prior
    DiffusionPriorMLP,              # Diffusion denoising network
    DiffusionScheduler,             # Beta scheduling
    DualDiffusionPrior,             # Dual priors (CLIP + DINOv2)
    diffusion_prior_loss,           # Loss function
    
    # Losses
    CLIPDINOJointLoss,              # MSE + cosine loss
    cosine_loss,                    # Normalized cosine loss
    mse_loss,                       # Mean squared error
    combined_embedding_loss,        # Multi-modal loss
)
```

### 3. Evaluation Module
```python
from fmri_reconstruction.evaluation import (
    # Feature similarity
    cosine_similarity_matrix,       # Compute similarity matrix
    mean_cosine_similarity,         # Average similarity
    normalized_feature_correlation, # Correlation in feature space
    
    # Image quality
    ssim,                           # Structural Similarity Index
    pixcorr,                        # Pixel correlation
    mse,                            # Mean squared error
    
    # Retrieval performance
    topk_accuracy,                  # Top-k classification accuracy
    topk_retrieval_accuracy,        # Top-k retrieval success
    
    # Alignment quality
    mean_pairwise_correlation,      # Subject alignment metric
    variance_explained,             # Variance explained by template
)
```

### 4. Retrieval Module
```python
from fmri_reconstruction.retrieval import (
    # Database
    RetrievalDatabase,              # Embedding storage + I/O
    
    # Search
    l2_normalize,                   # L2 normalization
    cosine_scores,                  # Compute cosine similarities
    top_k,                          # Top-k retrieval
    retrieve_batch,                 # Batch retrieval
    
    # Interface
    build_db,                       # Create database
    guarded_retrieve,               # Safe retrieval with checks
    
    # Validation
    assert_no_test_overlap,         # Prevent data leakage
    assert_subject_split,           # Check subject split
)
```

### 5. Reconstruction Module
```python
from fmri_reconstruction.reconstruction import (
    StableDiffusionImg2Img,         # Stable Diffusion img2img wrapper
)

# Usage
pipe = StableDiffusionImg2Img(
    model_id="runwayml/stable-diffusion-v1-5",
    device="cuda",
    disable_safety=True,
)
```

### 6. Embeddings Module
```python
from fmri_reconstruction.embeddings import (
    CLIPImageEncoder,               # Batch CLIP encoder
    DINOv2ImageEncoder,             # Batch DINOv2 encoder
)

# Usage
clip_encoder = CLIPImageEncoder(model_name="ViT-B-32", device="cuda")
embeddings = clip_encoder.encode_paths(image_paths, batch_size=32)
```

### 7. Training Module
```python
from fmri_reconstruction.training import (
    # Training
    SimpleTrainer,                  # Minimal training scaffold
    train_one_epoch,                # Full epoch training
    evaluate,                       # Full epoch evaluation
    
    # Checkpoints
    save_checkpoint,                # Simple checkpoint save
    load_checkpoint,                # Simple checkpoint load
    save_model_checkpoint,          # Full checkpoint with metadata
    load_model_checkpoint,          # Full checkpoint load
)
```

### 8. Ablation Module
```python
from fmri_reconstruction.ablation import (
    AblationResult,                 # Result container
    run_ablation,                   # Execute ablation study
    run_model_ablation,             # Evaluate model variants
    standard_reconstruction_ablations,  # Pre-defined ablations
    save_ablation_csv,              # Export results
)

# Usage
variants = standard_reconstruction_ablations()
results = run_ablation(my_experiment_fn, variants)
save_ablation_csv(results, "results.csv")
```

---

## Common Workflows

### Workflow 1: Load Data and Create Dataset
```python
from fmri_reconstruction.data import fetch_haxby_all, DatasetFactory
from fmri_reconstruction.config import ExperimentConfig

# Download Haxby
haxby = fetch_haxby_all(subjects=[1, 2, 3], fetch_stimuli=True)

# Or use factory
cfg = ExperimentConfig(dataset="haxby", aligned_dim=128)
dataset = DatasetFactory.from_config(cfg, split="train")
```

### Workflow 2: Train Multi-Modal Model
```python
from fmri_reconstruction.models import MultimodalNeuralModel, CLIPDINOJointLoss
from fmri_reconstruction.training import train_one_epoch
import torch.optim as optim

model = MultimodalNeuralModel(input_dim=128)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = CLIPDINOJointLoss(lambda_cos=0.1)

loss = train_one_epoch(model, train_loader, optimizer, loss_fn)
```

### Workflow 3: Build Retrieval Database
```python
from fmri_reconstruction.retrieval import build_db, guarded_retrieve
from fmri_reconstruction.embeddings import CLIPImageEncoder
import numpy as np

# Encode images
encoder = CLIPImageEncoder(device="cuda")
embeddings = encoder.encode_paths(image_paths)

# Build database
db = build_db(embeddings, paths=image_paths)

# Retrieve with safety checks
results = guarded_retrieve(
    queries, db, 
    train_paths=train_images, 
    test_paths=test_images, 
    k=5
)
```

### Workflow 4: Evaluate Reconstruction Quality
```python
from fmri_reconstruction.evaluation import (
    ssim, mean_cosine_similarity, topk_retrieval_accuracy
)

# Image quality
quality = ssim(reconstructed_image, reference_image)

# Feature similarity
similarity = mean_cosine_similarity(pred_embeddings, target_embeddings)

# Retrieval success
retrieval_success = topk_retrieval_accuracy(
    query_embeddings, db_embeddings, target_indices, k=5
)
```

### Workflow 5: Run Ablation Study
```python
from fmri_reconstruction.ablation import (
    run_ablation, standard_reconstruction_ablations, save_ablation_csv
)

def experiment(use_clip, use_dino, use_retrieval, **kwargs):
    model = create_model()
    metrics = evaluate_model(model, use_clip, use_dino, use_retrieval)
    return metrics

variants = standard_reconstruction_ablations()
results = run_ablation(experiment, variants)
save_ablation_csv(results, "ablation_results.csv")
```

### Workflow 6: Save and Load Model Checkpoint
```python
from fmri_reconstruction.training import (
    save_model_checkpoint, load_model_checkpoint
)

# Save
save_model_checkpoint(
    model, 
    "checkpoint.pt",
    optimizer=optimizer,
    scheduler=scheduler,
    epoch=10,
    best_loss=0.05
)

# Load
checkpoint = load_model_checkpoint(
    model, 
    "checkpoint.pt",
    device="cuda",
    optimizer=optimizer,
    scheduler=scheduler
)
print(f"Loaded from epoch {checkpoint['epoch']}")
```

---

## Configuration Example

```python
from fmri_reconstruction.config import ExperimentConfig

config = ExperimentConfig(
    dataset="haxby",
    aligned_dim=128,
    sequence_length=20,
    model_type="multimodal",
    batch_size=32,
    learning_rate=1e-3,
    epochs=50,
    seed=42,
)

# Create dataset
dataset = DatasetFactory.from_config(config, split="train")

# Create model
model = MultimodalNeuralModel(
    input_dim=config.aligned_dim,
    latent_dim=256,
)
```

---

## File Organization Summary

```
src/
├── __init__.py (main package exports)
└── fmri_reconstruction/
    ├── __init__.py (comprehensive module exports)
    ├── data/
    │   ├── __init__.py
    │   ├── datasets.py
    │   ├── haxby.py
    │   ├── nsd.py
    │   ├── live.py
    │   ├── loaders.py
    │   └── sequences.py
    ├── models/
    │   ├── __init__.py
    │   ├── transformer.py
    │   ├── projectors.py
    │   ├── multimodal.py
    │   ├── fusion.py
    │   ├── diffusion_prior.py
    │   └── losses.py
    ├── evaluation/
    │   ├── __init__.py
    │   ├── feature_metrics.py
    │   ├── image_metrics.py
    │   ├── semantic_metrics.py
    │   ├── retrieval_metrics.py
    │   └── alignment_evaluator.py
    ├── retrieval/
    │   ├── __init__.py
    │   ├── database.py
    │   ├── cosine.py
    │   ├── interface.py
    │   └── split_guard.py
    ├── reconstruction/
    │   ├── __init__.py
    │   └── stable_diffusion.py
    ├── embeddings/
    │   ├── __init__.py
    │   ├── clip.py
    │   └── dino.py
    ├── training/
    │   ├── __init__.py
    │   ├── trainer.py
    │   ├── loops.py
    │   ├── checkpoint.py
    │   └── checkpoints.py
    └── ablation/
        ├── __init__.py
        ├── runner.py
        └── experiments.py
```

---

## Error Prevention

All modules include:
- ✅ `from __future__ import annotations` for forward compatibility
- ✅ Type hints on public functions
- ✅ Data leakage guards in retrieval
- ✅ Shape validation for data loading
- ✅ Device-agnostic code (CPU/GPU)

---

**Last Updated**: 2026-08-26  
**Status**: ✅ Production Ready
