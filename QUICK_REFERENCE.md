"""
# Quick Reference Card - fMRI Reconstruction Pipeline

## 🚀 QUICK START (Copy & Paste)

### Option A: Use Pre-trained Pipeline
```python
from fmri_reconstruction import initialize_pipeline_from_checkpoints

pipeline = initialize_pipeline_from_checkpoints(
    model_checkpoint="models/fmri_encoder.pt",
    clip_checkpoint="models/clip_projector.pt",
    dino_checkpoint="models/dino_projector.pt",
    database_dir="retrieval_database/",
)

results = pipeline.reconstruct_batch(sequences, labels, image_paths)
metrics = pipeline.evaluate_batch(original_paths, results["reconstructions"])
```

### Option B: Step-by-Step
```python
from fmri_reconstruction import (
    load_fmri_data, create_sequences, MultimodalNeuralModel,
    CLIPProjector, DINOv2Projector, RetrievalDatabase,
    fMRIVisualReconstructionPipeline,
)

# Load data
fmri, labels = load_fmri_data("haxby", subject=1)
sequences, labels = create_sequences(fmri, labels)

# Create models
model = MultimodalNeuralModel(sequences.shape[-1], 256)
clip = CLIPProjector(256, 512)
dino = DINOv2Projector(256, 768)

# Build database
db = RetrievalDatabase()
# ... add images ...
db.finalize()

# Run pipeline
pipeline = fMRIVisualReconstructionPipeline(model, clip, dino, db, diffusion)
results = pipeline.reconstruct_batch(sequences, labels, paths)
```

---

## 📊 DATA OPERATIONS

### Load Data
```python
from fmri_reconstruction import load_fmri_data

fmri, labels = load_fmri_data(
    dataset="haxby",  # or "nsd", "live"
    subject=1,
    standardize=True,
    remove_rest=True,
)
```

### Preprocess Data
```python
from fmri_reconstruction import create_sequences, normalize_labels

labels = normalize_labels(labels)
sequences, seq_labels = create_sequences(
    fmri, labels,
    sequence_length=16,  # 16 timepoints
    step=1,              # 1 timepoint overlap
)
```

### Split Data
```python
from fmri_reconstruction import train_test_split

train_idx, test_idx = train_test_split(
    labels, test_size=0.2, random_state=42
)
```

---

## 🧠 MODELS

### Create Neural Encoder
```python
from fmri_reconstruction import MultimodalNeuralModel

model = MultimodalNeuralModel(
    input_dim=3000,     # fMRI dimension
    latent_dim=256,     # Latent dimension
)
```

### Create Projectors
```python
from fmri_reconstruction import CLIPProjector, DINOv2Projector

clip_proj = CLIPProjector(256, 512)      # 256 → CLIP 512
dino_proj = DINOv2Projector(256, 768)    # 256 → DINOv2 768
```

### Inference
```python
import torch

x = torch.from_numpy(sequences).float()
with torch.no_grad():
    latent = model.encoder(x)
    clip_emb = clip_proj(latent)
    dino_emb = dino_proj(latent)
```

---

## 🔍 RETRIEVAL

### Build Database
```python
from fmri_reconstruction import RetrievalDatabase

db = RetrievalDatabase()
for category in ["face", "house"]:
    db.add_category(category)
    for img_path in images[category]:
        clip_emb, dino_emb = encode_image(img_path)  # Your encoding function
        db.add_image(category, img_path, clip_emb, dino_emb)
db.finalize()
```

### Retrieve Images
```python
results = db.retrieve_top_k(
    category="face",
    query_clip=clip_emb,
    query_dino=dino_emb,
    k=3,  # Top 3
)
```

### Full Pipeline
```python
from fmri_reconstruction import full_retrieval_pipeline

guidance = full_retrieval_pipeline(
    db, fmri_latent, category,
    clip_proj, dino_proj,
    device="cuda",
)
# Returns: image_path, text_prompt, diffusion_strength, guidance_scale, etc.
```

### Persistence
```python
db.save("retrieval_database/")
db = RetrievalDatabase.load("retrieval_database/")
```

---

## 📈 EVALUATION

### Compute Metrics
```python
from fmri_reconstruction import compute_evaluation_metrics

metrics = compute_evaluation_metrics(
    predicted.numpy(),
    reference.numpy(),
    metrics=["cosine_sim", "mse", "ssim", "psnr"],
)
# Returns: {"cosine_sim": 0.85, "mse": 0.12, "ssim": 0.78, "psnr": 25.3}
```

### Loss Functions
```python
from fmri_reconstruction import CosineLoss, CLIPDINOJointLoss
import torch

# Cosine loss
loss_fn = CosineLoss()
loss = loss_fn(predicted, reference)

# Joint loss (CLIP + DINOv2)
loss_fn = CLIPDINOJointLoss()
loss = loss_fn(clip_pred, clip_ref, dino_pred, dino_ref)
```

### Feature Metrics
```python
from fmri_reconstruction import (
    cosine_similarity,
    mean_cosine_similarity,
    normalized_feature_correlation,
)

sim = cosine_similarity(pred[0], ref[0])
mean_sim = mean_cosine_similarity(preds, refs)
corr = normalized_feature_correlation(preds, refs)
```

### Image Metrics
```python
from fmri_reconstruction import ssim, psnr, mse

ssim_val = ssim(img1, img2)     # 0-1 (higher is better)
psnr_val = psnr(img1, img2)     # dB (higher is better)
mse_val = mse(img1, img2)       # (lower is better)
```

---

## 🎯 PIPELINE

### Initialize
```python
from fmri_reconstruction import fMRIVisualReconstructionPipeline

pipeline = fMRIVisualReconstructionPipeline(
    model=model,
    clip_projector=clip_proj,
    dino_projector=dino_proj,
    retrieval_database=db,
    diffusion_pipe=diffusion,
    device="cuda",
)
```

### Encode fMRI
```python
latent, clip_emb, dino_emb = pipeline.encode_fmri(fmri_sequence)
```

### Get Retrieval Guidance
```python
guidance = pipeline.retrieve_guidance(latent, category="face")
# Returns: dict with image_path, prompts, diffusion parameters
```

### Generate Reconstruction
```python
image = pipeline.generate_reconstruction(
    guidance,
    num_steps=30,          # Optional (uses adaptive if None)
    guidance_scale=7.5,    # Optional (uses adaptive if None)
    seed=42,
)
```

### Batch Processing
```python
results = pipeline.reconstruct_batch(
    fmri_sequences=sequences,
    labels=labels,
    image_paths=image_paths,
    save_dir="outputs/",
)
# Returns: {"reconstructions": [...], "retrieval_scores": [...], "metadata": [...]}
```

### Evaluate Quality
```python
metrics = pipeline.evaluate_batch(
    original_images=original_paths,
    reconstructed_images=results["reconstructions"],
    metrics=["ssim", "psnr", "cosine_sim"],
)
# Returns: {"ssim": 0.82, "psnr": 25.3, "cosine_sim": 0.85}
```

---

## 🔧 COMMON TASKS

### Task: Train Model
```python
from fmri_reconstruction import CLIPDINOJointLoss
import torch.optim as optim

loss_fn = CLIPDINOJointLoss()
optimizer = optim.Adam(list(model.parameters()) + list(clip_proj.parameters()), lr=1e-4)

for epoch in range(num_epochs):
    for batch in train_loader:
        latent = model.encoder(batch)
        clip_emb = clip_proj(latent)
        dino_emb = dino_proj(latent)
        loss = loss_fn(clip_emb, clip_target, dino_emb, dino_target)
        loss.backward()
        optimizer.step()
```

### Task: Save Checkpoints
```python
import torch

torch.save(model.state_dict(), "models/model.pt")
torch.save(clip_proj.state_dict(), "models/clip.pt")
torch.save(dino_proj.state_dict(), "models/dino.pt")
db.save("database/")
```

### Task: Load Checkpoints
```python
import torch

model.load_state_dict(torch.load("models/model.pt"))
clip_proj.load_state_dict(torch.load("models/clip.pt"))
dino_proj.load_state_dict(torch.load("models/dino.pt"))
db = RetrievalDatabase.load("database/")
```

### Task: Process New Subject
```python
fmri, labels = load_fmri_data("haxby", subject=2)
labels = normalize_labels(labels)
sequences, seq_labels = create_sequences(fmri, labels)

latent, clip_emb, dino_emb = pipeline.encode_fmri(sequences[0])
guidance = pipeline.retrieve_guidance(latent, seq_labels[0])
image = pipeline.generate_reconstruction(guidance)
```

---

## 📚 DOCUMENTATION FILES

| File | Purpose |
|------|---------|
| `PIPELINE_DOCUMENTATION.md` | Complete technical documentation |
| `COMPLETE_WORKFLOW_EXAMPLES.py` | 5 working code examples |
| `MODULES_INTEGRATION_GUIDE.md` | How modules work together |
| `IMPLEMENTATION_SUMMARY.md` | Project completion summary |
| `validate_pipeline.py` | Test suite |
| This file | Quick reference |

---

## ⚡ PERFORMANCE TIPS

### Batch Processing
```python
# Good: Process multiple samples at once
results = pipeline.reconstruct_batch(sequences, labels, paths)

# Avoid: Processing one sample at a time in a loop
for seq in sequences:
    pipeline.reconstruct_batch(seq[np.newaxis], ...)
```

### GPU Usage
```python
# Good
pipeline = fMRIVisualReconstructionPipeline(..., device="cuda")

# Avoid
pipeline = fMRIVisualReconstructionPipeline(..., device="cpu")
```

### Database Caching
```python
# Good: Load once
db = RetrievalDatabase.load("database/")
for sample in samples:
    guidance = full_retrieval_pipeline(db, ...)

# Avoid: Load every time
for sample in samples:
    db = RetrievalDatabase.load("database/")  # Slow!
    guidance = full_retrieval_pipeline(db, ...)
```

---

## 🔍 DEBUGGING

### Check Data Shape
```python
from fmri_reconstruction import validate_data_shapes

validate_data_shapes(sequences, seq_labels, image_paths)
```

### Test Imports
```python
# Quick import test
import fmri_reconstruction as fmr
print(dir(fmr))  # See all exported functions
```

### Validate Pipeline
```python
# Run validation tests
python validate_pipeline.py
```

---

## 📋 COMMON ERRORS & FIXES

| Error | Cause | Fix |
|-------|-------|-----|
| `Shape mismatch` | Input dim doesn't match | Check `input_dim` in `load_fmri_data` |
| `Out of memory` | Batch too large | Reduce batch size in `reconstruct_batch` |
| `Low quality` | Poor retrieval | Check database size and quality |
| `Import error` | Module not found | Run from project root, check sys.path |

---

## 🎓 LEARNING PATHS

### 5-Minute Quick Start
1. Copy `Option A` above
2. Have model checkpoints ready
3. Run!

### 30-Minute Overview
1. Read this file
2. Run `COMPLETE_WORKFLOW_EXAMPLES.py`
3. Understand the flow

### 2-Hour Deep Dive
1. Read `PIPELINE_DOCUMENTATION.md`
2. Study each module's code
3. Run `validate_pipeline.py`
4. Experiment with examples

### Production Setup
1. Understand all 3 modules
2. Set up data pipeline
3. Train model
4. Build retrieval database
5. Deploy pipeline

---

## 🎯 FEATURE MATRIX

| Feature | Module | Status |
|---------|--------|--------|
| Load data | preprocessing | ✅ |
| Preprocess images | preprocessing | ✅ |
| Create sequences | preprocessing | ✅ |
| Train/test split | preprocessing | ✅ |
| Neural encoder | models | ✅ |
| CLIP projector | models | ✅ |
| DINOv2 projector | models | ✅ |
| Retrieval database | retrieval | ✅ |
| Adaptive parameters | retrieval | ✅ |
| Text prompting | retrieval | ✅ |
| Feature metrics | evaluation | ✅ |
| Image metrics | evaluation | ✅ |
| Loss functions | evaluation | ✅ |
| End-to-end pipeline | pipeline | ✅ |
| Batch processing | pipeline | ✅ |
| Quality evaluation | pipeline | ✅ |

---

## 📞 SUPPORT

### For Architecture Questions
→ Read `PIPELINE_DOCUMENTATION.md`

### For Implementation Examples
→ Read `COMPLETE_WORKFLOW_EXAMPLES.py`

### For Integration Patterns
→ Read `MODULES_INTEGRATION_GUIDE.md`

### For Quick Answers
→ Read this file

### For Testing
→ Run `validate_pipeline.py`

---

## ✅ CHECKLIST

Before running pipeline:
- [ ] Models trained and checkpoints saved
- [ ] Retrieval database built and saved
- [ ] Data prepared and validated
- [ ] Imports working (run validate_pipeline.py)
- [ ] GPU available (if using CUDA)

Before going to production:
- [ ] All tests pass
- [ ] Metrics meet requirements
- [ ] Database thoroughly tested
- [ ] Performance benchmarked
- [ ] Documentation reviewed

---

**Version**: 1.0
**Status**: Ready to Use
**Last Updated**: 2024

For more details, see full documentation files.
"""

__doc__ = __doc__
