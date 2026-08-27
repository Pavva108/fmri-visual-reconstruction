"""
# New Modules Integration Guide

## 📁 Three New Python Modules

This project contains three comprehensive new Python modules that form the core of the fMRI visual reconstruction pipeline:

### 1. 📊 `src/fmri_reconstruction/data/preprocessing.py`
**Purpose**: Complete data preprocessing for all datasets

**What it does**:
- Loads fMRI data from Haxby, NSD, or LIVE datasets
- Normalizes category labels (e.g., "faces" → "face")
- Creates temporal sequences from continuous fMRI signals
- Preprocesses images for CLIP and DINOv2 embeddings
- Splits data for training/testing
- Validates data consistency

**Key Functions**:
```python
load_fmri_data(dataset="haxby", subject=1, standardize=True)
create_sequences(fmri_data, labels, sequence_length=16)
normalize_labels(labels)
train_test_split(labels, test_size=0.2)
```

**Integration**: Used in data loading phase of pipeline

---

### 2. 📈 `src/fmri_reconstruction/evaluation/evaluation.py`
**Purpose**: Unified evaluation metrics and loss functions

**What it does**:
- Computes feature similarity metrics (cosine similarity, correlation)
- Calculates image quality metrics (SSIM, PSNR, MSE)
- Measures retrieval performance (top-K accuracy, MRR)
- Provides PyTorch-compatible loss functions
- Supports batch evaluation

**Key Functions**:
```python
# Metrics
cosine_similarity(pred, ref)
mean_cosine_similarity(preds, refs)
ssim(img1, img2)
psnr(img1, img2)
topk_accuracy(ranks, k=3)

# Loss functions
loss = CosineLoss()
loss = CLIPDINOJointLoss()
compute_evaluation_metrics(preds, refs, metrics=["ssim", "psnr"])
```

**Integration**: Used during training (loss functions) and evaluation (metrics)

---

### 3. 🔍 `src/fmri_reconstruction/retrieval/retrieval.py`
**Purpose**: Image retrieval system with adaptive parameters

**What it does**:
- Stores image embeddings (CLIP + DINOv2) in database
- Retrieves relevant images based on fMRI embeddings
- Automatically selects diffusion parameters based on similarity
- Uses structure-aware matching for shape-focused categories
- Generates category-specific text prompts

**Key Functions**:
```python
# Database
db = RetrievalDatabase()
db.add_category("face")
db.add_image("face", path, clip_emb, dino_emb)
db.finalize()

# Retrieval
results = db.retrieve_top_k("face", query_clip, query_dino, k=3)

# Complete pipeline
guidance = full_retrieval_pipeline(
    db, fmri_latent, category,
    clip_proj, dino_proj
)
```

**Integration**: Used to guide diffusion model generation

---

## 🔄 How They Connect

```
                    Pipeline Flow

DATA LOADING (preprocessing.py)
    ↓
    • Load fMRI from dataset
    • Normalize labels
    • Create sequences
    • Split train/test
    ↓
NEURAL ENCODING (existing models)
    ↓
    • fMRI → Latent (Transformer)
    • Latent → CLIP embedding
    • Latent → DINOv2 embedding
    ↓
RETRIEVAL (retrieval.py)
    ↓
    • Query database with embeddings
    • Get top-K similar images
    • Compute adaptive parameters
    • Generate text prompts
    ↓
DIFFUSION GENERATION (existing utilities)
    ↓
    • Use retrieved image as init
    • Apply with adaptive parameters
    • Generate reconstruction
    ↓
EVALUATION (evaluation.py)
    ↓
    • Compute feature similarity
    • Measure image quality
    • Calculate retrieval accuracy
    ↓
RESULTS
```

---

## 💻 Code Example: Complete Usage

```python
from fmri_reconstruction import (
    # Data preprocessing
    load_fmri_data,
    create_sequences,
    train_test_split,
    normalize_labels,
    
    # Models
    MultimodalNeuralModel,
    CLIPProjector,
    DINOv2Projector,
    
    # Retrieval
    RetrievalDatabase,
    full_retrieval_pipeline,
    
    # Evaluation
    compute_evaluation_metrics,
    CosineLoss,
    
    # Pipeline
    fMRIVisualReconstructionPipeline,
)

# ========== PHASE 1: LOAD AND PREPROCESS DATA ==========
print("Phase 1: Loading data...")
fmri_data, labels = load_fmri_data(
    dataset="haxby",
    subject=1,
    standardize=True,
)
labels = normalize_labels(labels)
sequences, seq_labels = create_sequences(
    fmri_data,
    labels,
    sequence_length=16,
)
train_idx, test_idx = train_test_split(
    seq_labels,
    test_size=0.2,
)
print(f"✓ Loaded {len(sequences)} sequences")

# ========== PHASE 2: INITIALIZE MODELS ==========
print("Phase 2: Initializing models...")
model = MultimodalNeuralModel(
    input_dim=sequences.shape[-1],
    latent_dim=256,
)
clip_proj = CLIPProjector(256, 512)
dino_proj = DINOv2Projector(256, 768)
print("✓ Models initialized")

# ========== PHASE 3: BUILD RETRIEVAL DATABASE ==========
print("Phase 3: Building retrieval database...")
database = RetrievalDatabase()
database.add_category("face")
database.add_category("house")
# ... add images with embeddings ...
database.finalize()
print("✓ Database built")

# ========== PHASE 4: CREATE PIPELINE ==========
print("Phase 4: Creating pipeline...")
pipeline = fMRIVisualReconstructionPipeline(
    model=model,
    clip_projector=clip_proj,
    dino_projector=dino_proj,
    retrieval_database=database,
    diffusion_pipe=diffusion_pipe,
)
print("✓ Pipeline ready")

# ========== PHASE 5: RECONSTRUCT IMAGES ==========
print("Phase 5: Reconstructing images...")
results = pipeline.reconstruct_batch(
    fmri_sequences=sequences[test_idx],
    labels=labels[test_idx],
    image_paths=image_paths[test_idx],
    save_dir="outputs/",
)
print(f"✓ Generated {len(results['reconstructions'])} images")

# ========== PHASE 6: EVALUATE QUALITY ==========
print("Phase 6: Evaluating quality...")
metrics = pipeline.evaluate_batch(
    original_images=image_paths[test_idx],
    reconstructed_images=results["reconstructions"],
    metrics=["ssim", "psnr", "cosine_sim"],
)
print(f"✓ SSIM: {metrics['ssim']:.4f}")
print(f"✓ PSNR: {metrics['psnr']:.2f} dB")
print(f"✓ Cosine Similarity: {metrics['cosine_sim']:.4f}")
```

---

## 📊 Data Flow Example

```python
# Input: Raw fMRI signal
fmri = np.random.randn(1000, 3000)  # 1000 time points, 3000 voxels

# PREPROCESSING
labels = np.array(["faces"] * 100 + ["houses"] * 100)  # Raw labels
labels = normalize_labels(labels)  # → ["face", "face", ..., "house", "house"]
sequences, seq_labels = create_sequences(fmri, labels)
# → sequences: (950, 16, 3000)  [50 sequences, each with 16 timepoints]

# ENCODING
latent = model.encoder(sequences[0])  # (1, 256)
clip_emb = clip_proj(latent)          # (1, 512)
dino_emb = dino_proj(latent)          # (1, 768)

# RETRIEVAL
guidance = full_retrieval_pipeline(
    database,
    latent,
    seq_labels[0],  # "face" or "house"
    clip_proj,
    dino_proj,
)
# → guidance: {
#     "image_path": "path/to/face_123.jpg",
#     "text_prompt": "A detailed photograph of a face...",
#     "diffusion_strength": 0.7,
#     "guidance_scale": 7.5,
#     "num_inference_steps": 30,
# }

# GENERATION
reconstructed = diffusion_pipe.generate(
    prompt=guidance["text_prompt"],
    image=load_image(guidance["image_path"]),
    strength=guidance["diffusion_strength"],
    guidance_scale=guidance["guidance_scale"],
    num_steps=guidance["num_inference_steps"],
)
# → reconstructed: PIL Image (512×512)

# EVALUATION
metrics = compute_evaluation_metrics(
    np.array(reconstructed).reshape(1, -1),
    np.array(load_image(image_path)).reshape(1, -1),
    metrics=["ssim", "psnr"],
)
# → metrics: {"ssim": 0.82, "psnr": 25.3}
```

---

## 🔑 Key Design Patterns

### Pattern 1: Unified Data Loading
```python
# All datasets use same interface
fmri1, labels1 = load_fmri_data("haxby", subject=1)
fmri2, labels2 = load_fmri_data("nsd", subject=2)
fmri3, labels3 = load_fmri_data("live", subject=1)
```

### Pattern 2: Batch Evaluation
```python
# Single metric call processes all samples
metrics = compute_evaluation_metrics(
    predictions,      # (N, D)
    references,       # (N, D)
    metrics=["cosine_sim", "mse", "ssim"]
)
# → returns {"cosine_sim": 0.85, "mse": 0.12, "ssim": 0.78}
```

### Pattern 3: Adaptive Parameters
```python
# Parameters automatically chosen based on similarity
if similarity_score > 0.9:  # Very similar
    strength = 0.9          # Strong guidance
    guidance_scale = 10.0   # High guidance
    steps = 40              # Many iterations
else:  # Dissimilar
    strength = 0.5          # Weak guidance
    guidance_scale = 5.0    # Low guidance
    steps = 25              # Fewer iterations
```

### Pattern 4: Database Persistence
```python
# Build once
db = RetrievalDatabase()
# ... add images ...
db.finalize()
db.save("database/")

# Load many times
db = RetrievalDatabase.load("database/")
results = db.retrieve_top_k(...)
```

---

## 🎯 Common Tasks

### Task 1: Load and Preprocess Data
```python
from fmri_reconstruction.data.preprocessing import (
    load_fmri_data,
    normalize_labels,
    create_sequences,
)

fmri, labels = load_fmri_data("haxby", subject=1)
labels = normalize_labels(labels)
sequences, seq_labels = create_sequences(fmri, labels)
```

### Task 2: Build Retrieval Database
```python
from fmri_reconstruction.retrieval.retrieval import RetrievalDatabase

db = RetrievalDatabase()
for category, images in category_images.items():
    db.add_category(category)
    for img_path in images:
        clip_emb, dino_emb = encode_image(img_path)
        db.add_image(category, img_path, clip_emb, dino_emb)
db.finalize()
```

### Task 3: Compute Metrics
```python
from fmri_reconstruction.evaluation.evaluation import (
    compute_evaluation_metrics,
    CosineLoss,
)

# Batch metrics
metrics = compute_evaluation_metrics(pred, ref, metrics=["ssim", "psnr"])

# Loss function
loss_fn = CosineLoss()
loss = loss_fn(predicted_emb, reference_emb)
```

### Task 4: Run Complete Pipeline
```python
from fmri_reconstruction.pipeline import fMRIVisualReconstructionPipeline

pipeline = fMRIVisualReconstructionPipeline(...)
results = pipeline.reconstruct_batch(seqs, labels, paths)
metrics = pipeline.evaluate_batch(original_paths, results["reconstructions"])
```

---

## 📖 Documentation Files

1. **PIPELINE_DOCUMENTATION.md** - Technical architecture and details
2. **COMPLETE_WORKFLOW_EXAMPLES.py** - 5 working code examples
3. **validate_pipeline.py** - Test suite for all modules
4. **IMPLEMENTATION_SUMMARY.md** - Project completion summary
5. **This file** - Integration guide

---

## ✅ Quality Assurance

All three modules have been validated:
- ✅ No syntax errors
- ✅ No import errors
- ✅ Comprehensive docstrings
- ✅ Type hints throughout
- ✅ Error handling
- ✅ Input validation

---

## 🚀 Getting Started

### Quick Start (5 minutes)
```python
from fmri_reconstruction import initialize_pipeline_from_checkpoints

pipeline = initialize_pipeline_from_checkpoints(...)
results = pipeline.reconstruct_batch(sequences, labels, paths)
print(f"Generated {len(results['reconstructions'])} images!")
```

### Full Implementation (1 hour)
See `COMPLETE_WORKFLOW_EXAMPLES.py` for detailed step-by-step examples

### Deep Dive (read documentation)
See `PIPELINE_DOCUMENTATION.md` for architecture and design details

---

## 🎓 Learning Path

1. **Start here**: This file (overview and basic usage)
2. **Then read**: COMPLETE_WORKFLOW_EXAMPLES.py (working code)
3. **Then study**: PIPELINE_DOCUMENTATION.md (architecture)
4. **Then review**: Code in each module (detailed implementation)
5. **Then test**: Run validate_pipeline.py (verification)

---

## 💡 Tips & Tricks

1. **Use batch processing** - `reconstruct_batch()` is faster than loops
2. **Cache the database** - Don't rebuild, use `load()`
3. **Check similarity scores** - Use adaptive parameters
4. **Validate data** - Use `validate_data_shapes()`
5. **Save checkpoints** - Checkpoint models during training

---

## 🔗 Module Dependencies

```
preprocessing.py
  ├─ numpy, scipy
  ├─ torch
  ├─ PIL (image processing)
  ├─ torchvision (preprocessing)
  ├─ nilearn (fMRI loading)
  └─ existing data modules

evaluation.py
  ├─ numpy, torch
  ├─ scipy, scikit-image
  ├─ sklearn
  └─ existing evaluation modules

retrieval.py
  ├─ torch, numpy
  ├─ PIL (image loading)
  ├─ pathlib (file handling)
  └─ functional components

pipeline.py
  ├─ All three modules above
  ├─ Models (encoders, projectors)
  ├─ Reconstruction (diffusion)
  └─ Embeddings (CLIP, DINOv2)
```

---

**Status**: ✅ Complete and Ready to Use

These three modules form the backbone of the complete fMRI visual reconstruction pipeline.
"""

__doc__ = __doc__
