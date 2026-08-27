# Python Module Workflow

This document describes the Python implementation behind the Haxby, NSD, and LIVE visual-reconstruction notebooks. The notebooks are experiment documents; the reusable implementation lives in `src/`.

The complete flow is:

```text
dataset files
    -> dataset-specific loading and preprocessing
    -> temporal sequences and label/image alignment
    -> shared latent alignment
    -> Transformer fMRI encoder
    -> CLIP and DINOv2 projectors
    -> optional diffusion-prior refinement and feature fusion
    -> train-only image retrieval
    -> Stable Diffusion img2img reconstruction
    -> held-out evaluation and saved results
```

## 1. Installation

Run these commands from the repository root:

```powershell
python -m venv .venv
\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

The repository ignores `.venv/`; it should remain a local development environment and must not be added to source control.

The core package requires NumPy, SciPy, scikit-learn, scikit-image, pandas, Nibabel, Nilearn, HDF5 support, Pillow, PyTorch, and torchvision. CLIP, DINOv2, Diffusers, and Hugging Face utilities are needed only for the related embedding, retrieval-database, or diffusion stages. Check `requirements.txt` and `pyproject.toml` for the project dependency definitions.

## 2. Package Layout

```text
src/
├── data/                         # Short dataset-specific public adapters
│   ├── preprocessing.py          # PreparedDataset and unified dispatch
│   ├── haxby.py                  # Haxby helpers
│   ├── nsd.py                    # NSD/MindEye helpers
│   └── live.py                   # LIVE BOLD and stimulus helpers
├── evaluation/                   # Public metrics and batch evaluation
│   └── evaluation.py
├── models/                       # Public model registry and factory
│   └── models.py
├── retrieval/                    # Public top-k and pipeline helpers
│   └── retrieval.py
└── fmri_reconstruction/          # Canonical implementation package
    ├── data/                      # Loaders, sequences, and dataset classes
    ├── preprocessing.py           # Detrending, filtering, standardization
    ├── alignment.py               # Shared-latent alignment implementations
    ├── models/                    # Transformer, projectors, fusion, priors
    ├── embeddings/                # CLIP and DINOv2 image encoders
    ├── retrieval/                 # Database, cosine search, split guards
    ├── reconstruction/            # Stable Diffusion img2img wrapper
    ├── evaluation/                # Feature, image, retrieval, alignment metrics
    ├── training/                  # Training loops and checkpoints
    ├── ablation/                  # Reconstruction ablation runners
    └── pipeline.py                # End-to-end orchestration
```

`fmri_reconstruction` is the canonical implementation. The top-level packages provide shorter imports and dataset-specific entry points without duplicating the model, retrieval, or metric algorithms.

## 3. Dataset Preparation

### Haxby

Haxby data is loaded with Nilearn. The standard preparation is:

1. Load fMRI time points and labels.
2. Remove `rest` trials.
3. Extract the VT ROI when a mask is available.
4. Normalize labels such as `faces` to `face`.
5. Create overlapping temporal windows.
6. Align one label and one stimulus path to each window.

```python
import numpy as np

from data.preprocessing import preprocess_haxby

fmri = np.load("prepared/haxby_train.npy")       # (T, D)
labels = np.load("prepared/haxby_labels.npy")   # (T,)
image_paths = np.load("prepared/haxby_paths.npy", allow_pickle=True).tolist()

prepared = preprocess_haxby(
    fmri,
    labels,
    image_paths=image_paths,
    sequence_length=20,
    stride=1,
).validate()

print(prepared.fmri.shape)       # (N, 20, D)
print(prepared.labels.shape)     # (N,)
```

For Nilearn loading and VT extraction, use:

```python
from fmri_reconstruction.data.haxby import (
    extract_vt,
    fetch_haxby_all,
    load_labels,
    remove_rest,
)
```

### NSD / MindEye

The repository supports the selected NSD/MindEye HDF5 and annotation files. The NSD adapter deliberately requires the HDF5 dataset key explicitly because MindEye exports can use different internal key names.

```python
from data.nsd import load_nsd_hdf5
from data.preprocessing import preprocess_nsd

fmri = load_nsd_hdf5("Dataset/NSD/subj01_nsdgeneral.hdf5", data_key="YOUR_KEY")
prepared = preprocess_nsd(
    fmri,
    labels=None,
    image_paths=train_or_test_paths,
    standardize=True,
    sequence_length=20,
).validate()
```

Inspect an HDF5 file before selecting its key:

```python
import h5py

with h5py.File("Dataset/NSD/subj01_nsdgeneral.hdf5", "r") as handle:
    handle.visititems(lambda name, obj: print(name, type(obj).__name__))
```

Use `fmri_reconstruction.data.nsd.download_nsd_files()` when downloading selected files from the configured Hugging Face dataset is appropriate.

### LIVE

The LIVE notebook documents the raw-BOLD contract as:

```text
VT mask extraction -> linear detrending -> 0.01-0.1 Hz band-pass filter
-> voxel-wise standardization -> shared alignment
```

The supplied methodology reports a BOLD shape of `(147, 144, 36, 165)` and a TR of `3.0` seconds. The VT mask must be provided by the study; the package does not invent one.

```python
import nibabel as nib
import numpy as np

from data.preprocessing import preprocess_live

bold = nib.load("path/to/live_bold.nii.gz").get_fdata(dtype=np.float32)
vt_mask = np.load("path/to/vt_mask.npy").astype(bool)

prepared = preprocess_live(
    bold,
    vt_mask,
    tr=3.0,
    low_hz=0.01,
    high_hz=0.1,
    labels=live_labels,
    image_paths=live_image_paths,
    sequence_length=20,
).validate()
```

Additional LIVE file and validation helpers are in `fmri_reconstruction.data.live`.

## 4. Alignment and Sequence Contract

The model and pipeline expect the final aligned representation in one of these forms:

```text
(T, 128)       continuous aligned fMRI time points
(N, 20, 128)   prepared temporal sequences
```

`fmri_reconstruction.alignment` contains the available alignment implementations. The PCA/shared-latent utilities are baselines; they must not be described as a learned paper alignment unless the corresponding trained alignment model and checkpoint are used.

Sequence helpers are available from `fmri_reconstruction.data.sequences`, while the unified prepared-data object is available from `data.preprocessing.PreparedDataset`.

## 5. Models and Training

The model stages are:

```text
NeuralTransformer
    -> shared latent representation (default dimension 256)
    -> CLIPProjector (default dimension 512)
    -> DINOv2Projector (default dimension 768)
```

The combined implementation is `MultimodalNeuralModel`. The package also contains `FusionModule`, `DiffusionPrior`, `DualDiffusionPrior`, and the CLIP/DINO embedding losses.

```python
from models.models import build_model

model = build_model(
    input_dim=128,
    latent_dim=256,
    clip_dim=512,
    dino_dim=768,
)

output = model(batch_sequences, return_latent=True)
latent = output["latent"]
clip_prediction = output["clip"]
dino_prediction = output["dino"]
```

Canonical imports remain available:

```python
from fmri_reconstruction.models import (
    CLIPDINOJointLoss,
    CLIPProjector,
    DINOv2Projector,
    MultimodalNeuralModel,
    NeuralTransformer,
)
```

Training utilities and checkpoint helpers are in `fmri_reconstruction.training`. Checkpoints are experiment artifacts and are ignored by the repository's default `.gitignore` rules.

## 6. Retrieval Database and Top-k Search

Image retrieval must be built from training images only. Test images must not be inserted into the database or used to tune retrieval parameters.

```python
from fmri_reconstruction.embeddings import CLIPImageEncoder, DINOv2ImageEncoder
from fmri_reconstruction.retrieval import RetrievalDatabase

database = RetrievalDatabase()
clip_encoder = CLIPImageEncoder(device="cuda")
dino_encoder = DINOv2ImageEncoder(device="cuda")

for category, paths in train_images_by_category.items():
    clip_embeddings = clip_encoder.encode_paths(paths)
    dino_embeddings = dino_encoder.encode_paths(paths)
    for path, clip_embedding, dino_embedding in zip(
        paths, clip_embeddings, dino_embeddings
    ):
        database.add_image(category, path, clip_embedding, dino_embedding)

database.finalize(device="cuda")
database.save("outputs/retrieval_db")
```

Retrieve ranked candidates with all component scores:

```python
from retrieval.retrieval import retrieve_topk

top_matches = retrieve_topk(
    database,
    category="face",
    clip_query=clip_query,
    dino_query=dino_query,
    k=5,
)
```

The complete guidance function returns the selected image, CLIP/DINO/fusion scores, prompt, and adaptive diffusion parameters:

```python
from fmri_reconstruction.retrieval import full_retrieval_pipeline

guidance = full_retrieval_pipeline(
    database,
    fmri_latent,
    category="face",
    clip_projector=clip_projector,
    dino_projector=dino_projector,
)

print(guidance["image_path"])
print(guidance["num_inference_steps"])
```

For shape-focused categories, structure-aware reranking is available through `retrieve_with_structure_score()` and `compute_edge_similarity()`.

## 7. End-to-End Reconstruction

`fmri_reconstruction.pipeline.fMRIVisualReconstructionPipeline` coordinates encoding, retrieval, diffusion generation, batch output, and evaluation.

```python
from fmri_reconstruction.pipeline import (
    fMRIVisualReconstructionPipeline,
    initialize_pipeline_from_checkpoints,
)

pipeline = initialize_pipeline_from_checkpoints(
    model_checkpoint="checkpoints/transformer_best.pth",
    clip_checkpoint="checkpoints/clip_projector_best.pth",
    dino_checkpoint="checkpoints/dino_projector_best.pth",
    database_dir="outputs/retrieval_db",
    device="cuda",
)

results = pipeline.reconstruct_batch(
    fmri_sequences=prepared.fmri,
    labels=prepared.labels,
    image_paths=prepared.image_paths,
    save_dir="outputs/reconstructions",
)
```

The diffusion stage uses the retrieved image as the img2img initialization. Stable Diffusion weights are downloaded or loaded separately and should be selected explicitly for the experiment.

For a no-diffusion integration test, use the model, retrieval database, and evaluation APIs independently. A successful import does not constitute a valid reconstruction result.

## 8. Evaluation

The evaluation layer covers four groups of measurements:

- Image quality: SSIM, pixel correlation, MSE, RMSE, and PSNR.
- Feature quality: cosine similarity and normalized feature correlation.
- Retrieval: top-k accuracy and mean reciprocal rank.
- Alignment: mean pairwise correlation and variance explained.

Evaluate generated images as a batch:

```python
from evaluation.evaluation import evaluate_reconstruction_batch

metrics = evaluate_reconstruction_batch(
    original_images=prepared.image_paths,
    reconstructed_images=results["reconstructions"],
    metrics=["ssim", "pixcorr", "mse", "psnr"],
)
print(metrics)
```

Evaluate retrieval against known target database indices:

```python
from evaluation.evaluation import evaluate_retrieval

retrieval_metrics = evaluate_retrieval(
    query_embeddings,
    database_embeddings,
    target_indices,
    ks=(1, 5, 10),
)
```

The metric functions do not create ground truth. Scores must be computed on the held-out split using the actual reconstructed images and embeddings.

## 9. Notebook-to-Module Mapping

The three notebooks share the same decoder flow:

| Notebook stage | Python implementation |
| --- | --- |
| Haxby loading and label handling | `src/data/haxby.py`, `fmri_reconstruction.data.haxby` |
| NSD prepared arrays and HDF5 inspection | `src/data/nsd.py`, `fmri_reconstruction.data.nsd` |
| LIVE BOLD preprocessing | `src/data/live.py`, `src/data/preprocessing.py` |
| Detrending, filtering, standardization | `fmri_reconstruction/preprocessing.py` |
| Shared alignment | `fmri_reconstruction/alignment.py` |
| Transformer and projectors | `fmri_reconstruction/models/` |
| CLIP/DINO image embeddings | `fmri_reconstruction/embeddings/` |
| Diffusion priors and fusion | `fmri_reconstruction/models/` |
| Top-k retrieval and adaptive guidance | `fmri_reconstruction/retrieval/` |
| Stable Diffusion img2img | `fmri_reconstruction/reconstruction/` |
| Reconstruction orchestration | `fmri_reconstruction/pipeline.py` |
| Metrics and losses | `src/evaluation/`, `fmri_reconstruction/evaluation/` |

The notebooks intentionally require prepared aligned arrays and trained checkpoints for NSD and LIVE when those artifacts are not included in the repository. The Python modules preserve that explicit contract.

## 10. Validation and Project Commands

Compile the package:

```powershell
python -m compileall -q src
```

Run the focused test suite after installing the development dependencies:

```powershell
python -m pytest -q
```

Useful project scripts include:

```powershell
python scripts/check_environment.py
python scripts/verify_installation.py
python validate_pipeline.py
```

Use the dataset configuration files in `configs/` and the runnable scripts in `scripts/` for experiment-specific paths, subjects, checkpoints, and output locations.

## Reproducibility Rules

1. Fit normalization and alignment transforms on training data only.
2. Build retrieval databases from training images only.
3. Keep validation and test subjects/images outside training artifacts.
4. Record dataset split, sequence length, stride, model checkpoints, diffusion model ID, random seed, and output directory for each run.
5. Report metrics only from actual held-out reconstructions.
6. Do not describe the available alignment baseline as a learned alignment method without the trained alignment implementation and checkpoint.
