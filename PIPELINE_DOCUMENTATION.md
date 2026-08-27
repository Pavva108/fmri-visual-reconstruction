"""
# Complete fMRI Visual Reconstruction Pipeline Documentation

## Overview

The fMRI visual reconstruction pipeline is a complete end-to-end system that converts brain activity (fMRI scans) into visual image reconstructions. It integrates state-of-the-art neural networks, retrieval systems, and diffusion models.

## Architecture

```
fMRI Signal
    ↓
[Data Preprocessing]  ← data/preprocessing.py
  • Normalize labels
  • Create sequences
  • Standardize signals
    ↓
[Neural Encoding]     ← models/
  • fMRI → Latent (Transformer)
  • Latent → CLIP (512-dim)
  • Latent → DINOv2 (768-dim)
    ↓
[Retrieval System]    ← retrieval/retrieval.py
  • Query database with embeddings
  • Adaptive weighting (CLIP/DINOv2)
  • Structure-based reranking
  • Get guidance image + parameters
    ↓
[Diffusion Generation] ← reconstruction/
  • Load guidance image
  • Apply adaptive strength
  • Use adaptive guidance scale
  • Generate using Stable Diffusion
    ↓
[Image Reconstruction]
    ↓
[Evaluation]          ← evaluation/evaluation.py
  • Feature similarity (cosine)
  • Image quality (SSIM, PSNR)
  • Retrieval accuracy
```

## Key Components

### 1. Data Preprocessing (`src/fmri_reconstruction/data/preprocessing.py`)

**Purpose**: Unified preprocessing pipeline for all datasets.

**Key Functions**:
- `load_fmri_data()` - Load fMRI with optional standardization
- `normalize_labels()` - Standardize category names
- `create_sequences()` - Generate overlapping time windows
- `train_test_split()` - Stratified data splitting
- `get_clip_preprocessor()` / `get_dinov2_preprocessor()` - Image preprocessing
- `validate_data_shapes()` - Consistency checking

**Usage**:
```python
from fmri_reconstruction import load_fmri_data, create_sequences

# Load fMRI data
fmri_data, labels = load_fmri_data("haxby", subject=1, standardize=True)

# Create sequences
sequences, seq_labels = create_sequences(fmri_data, labels, sequence_length=16)
```

### 2. Evaluation Metrics (`src/fmri_reconstruction/evaluation/evaluation.py`)

**Purpose**: Comprehensive metrics and loss functions for training and evaluation.

**Key Categories**:

**Feature Metrics**:
- `cosine_similarity()` - Single pair similarity
- `mean_cosine_similarity()` - Batch average
- `normalized_feature_correlation()` - Normalized correlation

**Image Quality Metrics**:
- `ssim()` - Structural similarity
- `psnr()` - Peak signal-to-noise ratio
- `mse()`, `rmse()` - Mean squared error

**Retrieval Metrics**:
- `topk_accuracy()` - Top-K accuracy
- `mean_reciprocal_rank()` - MRR metric

**Loss Functions**:
- `CosineLoss` - PyTorch module for cosine distance
- `CLIPDINOJointLoss` - Combined CLIP + DINOv2 loss
- `combined_embedding_loss()` - Multi-modal loss
- `reconstruction_loss()` - Image reconstruction loss

**Usage**:
```python
from fmri_reconstruction import compute_evaluation_metrics, CosineLoss
import torch

# Compute metrics
metrics = compute_evaluation_metrics(
    predicted_emb.numpy(),
    reference_emb.numpy(),
    metrics=["cosine_sim", "mse", "ssim"],
)

# Use loss in training
loss_fn = CosineLoss()
loss = loss_fn(predicted, reference)
```

### 3. Retrieval System (`src/fmri_reconstruction/retrieval/retrieval.py`)

**Purpose**: Image retrieval with adaptive parameters for diffusion guidance.

**Key Features**:

**Database Management**:
- `RetrievalDatabase` class - Stores embeddings for retrieval
- `.add_image()` - Add image with CLIP + DINOv2 embeddings
- `.retrieve_top_k()` - Get top-K matches
- `.save()` / `.load()` - Persistence

**Adaptive Parameters**:
- `get_adaptive_strength()` - Diffusion strength based on similarity
- `get_adaptive_guidance()` - Guidance scale
- `get_adaptive_num_steps()` - Number of inference steps
- Based on CLIP + DINOv2 similarity scores

**Structure-Aware Matching**:
- `compute_edge_similarity()` - Edge-based shape matching
- `retrieve_with_structure_score()` - Rerank with structure

**Category Weights**:
- `HAXBY_CATEGORIES` - All Haxby categories
- `SHAPE_FOCUS_CATEGORIES` - Categories emphasizing shape
- `get_category_weights()` - Adaptive CLIP/DINOv2 weighting

**Complete Pipeline**:
- `full_retrieval_pipeline()` - End-to-end retrieval with all features

**Usage**:
```python
from fmri_reconstruction import RetrievalDatabase, full_retrieval_pipeline

# Build database
database = RetrievalDatabase()
for category, images in category_images.items():
    database.add_category(category)
    for img_path in images:
        clip_emb, dino_emb = encode_image(img_path)
        database.add_image(category, img_path, clip_emb, dino_emb)
database.finalize()

# Retrieve
results = full_retrieval_pipeline(
    database,
    fmri_query_embedding,
    category,
    clip_proj,
    dino_proj,
)
print(f"Retrieved: {results['image_path']}")
print(f"Guidance scale: {results['guidance_scale']}")
print(f"Diffusion strength: {results['diffusion_strength']}")
```

### 4. Complete Pipeline (`src/fmri_reconstruction/pipeline.py`)

**Purpose**: Orchestrates all components into unified workflow.

**Main Class**: `fMRIVisualReconstructionPipeline`

**Key Methods**:
- `encode_fmri()` - Convert fMRI to multimodal embeddings
- `retrieve_guidance()` - Get retrieval results
- `generate_reconstruction()` - Create image via diffusion
- `reconstruct_batch()` - Process multiple samples
- `evaluate_batch()` - Compute quality metrics

**Convenience Functions**:
- `initialize_pipeline_from_checkpoints()` - Load all models
- `build_retrieval_database_from_haxby()` - Automatic database building

**Usage**:
```python
from fmri_reconstruction import (
    fMRIVisualReconstructionPipeline,
    initialize_pipeline_from_checkpoints,
)

# Initialize
pipeline = initialize_pipeline_from_checkpoints(
    model_checkpoint="models/fmri_encoder.pt",
    clip_checkpoint="models/clip_projector.pt",
    dino_checkpoint="models/dino_projector.pt",
    database_dir="retrieval_database/",
)

# Reconstruct
results = pipeline.reconstruct_batch(
    fmri_sequences=sequences,
    labels=labels,
    image_paths=image_paths,
    save_dir="outputs/",
)

# Evaluate
metrics = pipeline.evaluate_batch(
    original_images=original_paths,
    reconstructed_images=results["reconstructions"],
)
print(f"SSIM: {metrics['ssim']:.4f}")
```

## Integration Flow

### Phase 1: Data Preparation
```python
# Load and preprocess
fmri, labels = load_fmri_data("haxby", subject=1, standardize=True)
labels = normalize_labels(labels)
sequences, seq_labels = create_sequences(fmri, labels)
train_idx, test_idx = train_test_split(seq_labels, test_size=0.2)
```

### Phase 2: Model Training
```python
# Define models
model = MultimodalNeuralModel(input_dim=sequences.shape[-1], latent_dim=256)
clip_proj = CLIPProjector(256, 512)
dino_proj = DINOv2Projector(256, 768)

# Train with loss functions
loss_fn = CLIPDINOJointLoss()
for epoch in range(num_epochs):
    for batch in train_loader:
        latent = model.encoder(batch)
        clip_emb = clip_proj(latent)
        dino_emb = dino_proj(latent)
        loss = loss_fn(clip_emb, dino_emb, targets)
        optimizer.step()
```

### Phase 3: Build Retrieval Database
```python
# Create database
database = RetrievalDatabase()

# Add categories and images
for category in HAXBY_CATEGORIES:
    database.add_category(category)
    for img_path in get_category_images(category):
        clip_emb, dino_emb = encode_image(img_path)
        database.add_image(category, img_path, clip_emb, dino_emb)

database.finalize()
database.save("retrieval_database/")
```

### Phase 4: Run Reconstruction Pipeline
```python
# Initialize pipeline
pipeline = initialize_pipeline_from_checkpoints(...)

# Process test set
results = pipeline.reconstruct_batch(
    fmri_sequences=sequences[test_idx],
    labels=labels[test_idx],
    image_paths=image_paths[test_idx],
    save_dir="outputs/reconstructions/",
)

# Evaluate quality
metrics = pipeline.evaluate_batch(
    original_images=image_paths[test_idx],
    reconstructed_images=results["reconstructions"],
    metrics=["ssim", "psnr", "cosine_sim"],
)
```

## Import Structure

### Clean Imports
```python
# Import entire toolkit
import fmri_reconstruction as fmr

# Data pipeline
sequences = fmr.create_sequences(fmri, labels)
fmri, labels = fmr.load_fmri_data("haxby")

# Models
model = fmr.MultimodalNeuralModel(input_dim, latent_dim)
encoder = fmr.CLIPProjector(256, 512)

# Retrieval
database = fmr.RetrievalDatabase()
results = fmr.full_retrieval_pipeline(...)

# Evaluation
metrics = fmr.compute_evaluation_metrics(pred, ref)
loss = fmr.CosineLoss()

# Pipeline
pipeline = fmr.fMRIVisualReconstructionPipeline(...)
```

### Module-Specific Imports
```python
# Data preprocessing
from fmri_reconstruction.data.preprocessing import (
    load_fmri_data,
    create_sequences,
    normalize_labels,
)

# Evaluation
from fmri_reconstruction.evaluation.evaluation import (
    compute_evaluation_metrics,
    CosineLoss,
    CLIPDINOJointLoss,
)

# Retrieval
from fmri_reconstruction.retrieval.retrieval import (
    RetrievalDatabase,
    full_retrieval_pipeline,
    get_adaptive_strength,
)
```

## Common Workflows

### Workflow 1: Quick Reconstruction
```python
from fmri_reconstruction import (
    load_fmri_data, create_sequences, MultimodalNeuralModel,
    RetrievalDatabase, StableDiffusionImg2Img,
    fMRIVisualReconstructionPipeline, initialize_pipeline_from_checkpoints
)

# Option A: From scratch
model = MultimodalNeuralModel(128, 256)
pipeline = fMRIVisualReconstructionPipeline(model, ...)

# Option B: From checkpoints
pipeline = initialize_pipeline_from_checkpoints(...)

# Reconstruct
results = pipeline.reconstruct_batch(seqs, labels, paths)
```

### Workflow 2: Model Training
```python
from fmri_reconstruction import (
    MultimodalNeuralModel, CLIPProjector, DINOv2Projector,
    CLIPDINOJointLoss, create_sequences, load_fmri_data,
)

# Prepare data
seqs, labels = load_fmri_data("haxby")
seqs, labels = create_sequences(seqs, labels)

# Initialize models
model = MultimodalNeuralModel(input_dim, latent_dim)
clip_proj = CLIPProjector(latent_dim, 512)
dino_proj = DINOv2Projector(latent_dim, 768)
loss_fn = CLIPDINOJointLoss()

# Train
for epoch in range(num_epochs):
    for batch in train_loader:
        lat = model.encoder(batch)
        cl = clip_proj(lat)
        di = dino_proj(lat)
        loss = loss_fn(cl, di, targets)
        opt.step()
```

### Workflow 3: Evaluation
```python
from fmri_reconstruction import compute_evaluation_metrics

# Get predictions and references
predicted = model(fmri_test)
reference = load_reference_embeddings()

# Compute all metrics
metrics = compute_evaluation_metrics(
    predicted.numpy(),
    reference.numpy(),
    metrics=["cosine_sim", "mse", "ssim", "psnr"],
)

print(f"Cosine Similarity: {metrics['cosine_sim']:.4f}")
print(f"SSIM: {metrics['ssim']:.4f}")
print(f"PSNR: {metrics['psnr']:.2f} dB")
```

## File Organization

```
src/fmri_reconstruction/
├── data/
│   ├── preprocessing.py      ← Main preprocessing pipeline
│   ├── __init__.py           ← Updated with preprocessing exports
│   ├── datasets.py
│   ├── haxby.py
│   ├── nsd.py
│   ├── live.py
│   └── loaders.py
├── evaluation/
│   ├── evaluation.py         ← Main evaluation metrics and losses
│   ├── __init__.py           ← Updated with comprehensive exports
│   ├── feature_metrics.py
│   ├── image_metrics.py
│   ├── retrieval_metrics.py
│   └── alignment_evaluator.py
├── retrieval/
│   ├── retrieval.py          ← Main retrieval system
│   ├── __init__.py           ← Updated with comprehensive exports
│   ├── database.py
│   ├── cosine.py
│   ├── interface.py
│   └── split_guard.py
├── models/
│   ├── transformer.py        ← Neural encoder
│   ├── projectors.py         ← CLIP/DINOv2 projectors
│   ├── multimodal.py         ← Combined model
│   └── ...
├── pipeline.py               ← End-to-end pipeline orchestration
├── __init__.py               ← Updated with all new exports
└── ...
```

## Dependencies

**Core**:
- torch, torchvision
- numpy, scipy
- scikit-image, scikit-learn
- PIL

**fMRI Processing**:
- nilearn (fMRI loading)

**Embeddings**:
- open_clip (CLIP encoder)
- timm (DINOv2 encoder)
- transformers

**Diffusion**:
- diffusers (Stable Diffusion)

## Performance Considerations

### Optimization Tips
1. **Batch Processing**: Use `reconstruct_batch()` for better throughput
2. **GPU Allocation**: Use `device="cuda"` for all models
3. **Database Size**: Store database on disk to save memory
4. **Caching**: Cache category embeddings to avoid recomputation

### Memory Usage
- fMRI data: ~50MB (1000 samples × 3000 voxels)
- Database: ~2GB (1000 images × 2 embeddings)
- Model: ~500MB (encoder + projectors)
- Diffusion: ~5GB (Stable Diffusion)

## Troubleshooting

**Issue**: Shape mismatch in model
```
Error: Expected input shape (N, 128) got (N, 256)
```
**Solution**: Check `input_dim` matches your fMRI data dimension

**Issue**: Low reconstruction quality
```
Solution: 
1. Check retrieval database size and quality
2. Verify model convergence during training
3. Adjust adaptive parameters (strength, guidance_scale)
```

**Issue**: Out of memory
```
Solution:
1. Reduce batch size in reconstruct_batch()
2. Use smaller fMRI sequences
3. Offload database to CPU
```

## References

- Base architecture from notebooks: `Haxby-visual-reconstruction.ipynb`
- Datasets: Haxby, NSD, LIVE (NIMHANS)
- Models: Transformer encoder, CLIP/DINOv2 projectors
- Diffusion: Stable Diffusion v1.5

## Quick Start

```python
from fmri_reconstruction import initialize_pipeline_from_checkpoints

# Load everything
pipeline = initialize_pipeline_from_checkpoints(
    model_checkpoint="models/model.pt",
    clip_checkpoint="models/clip.pt",
    dino_checkpoint="models/dino.pt",
    database_dir="database/",
)

# Reconstruct
results = pipeline.reconstruct_batch(sequences, labels, paths)

# Evaluate
metrics = pipeline.evaluate_batch(original_paths, results["reconstructions"])

print(f"Done! SSIM: {metrics['ssim']:.4f}")
```
"""

__doc__ = __doc__
