# Subject-Agnostic fMRI Visual Reconstruction

**Paper: ** *Subject-Agnostic fMRI Visual Reconstruction via Shared Latent Alignment and CLIP–DINOv2 Guided Diffusion*

**GitHub: ** https://github.com/Pavva108/fmri-visual-reconstruction

## Installation

1. Prepare the required fMRI datasets and obtain the permissions/terms required by each dataset. The project evaluates Haxby, the Natural Scenes Dataset (NSD), and the LIVE dataset collected at NIMHANS, Bengaluru.

2. Git clone this repository:

```
git clone https://github.com/Pavva108/fmri-visual-reconstruction.git
cd fmri-visual-reconstruction
```

3. Create and activate a Python virtual environment:

```
python -m venv .venv
```

Windows:

```
.venv\Scripts\activate
```

Linux/macOS:

```
source .venv/bin/activate
```

4. Install the required packages:

```
pip install -r requirements.txt
pip install -e .
```

5. Verify the installation:

```
python scripts/check_environment.py
python scripts/verify_installation.py
```

Large fMRI datasets, pretrained models, checkpoints, and generated reconstruction files are not intended to be committed to the Git repository. Configure their local paths through the project configuration files.

## Usage

The repository provides dataset-specific notebooks and reusable Python modules for preprocessing, alignment, neural encoding, CLIP/DINOv2 prediction, retrieval, diffusion-prior refinement, reconstruction, evaluation, and ablation experiments.

- `notebooks/Haxby-visual-reconstruction.ipynb` performs the Haxby reconstruction workflow, including preprocessing/aligned inputs, feature prediction, retrieval, Stable Diffusion reconstruction, evaluation, and ablation experiments.
- `notebooks/NSD-visual-reconstruction.ipynb` performs the NSD reconstruction workflow using prepared/preprocessed fMRI representations, subject-wise evaluation, retrieval, reconstruction, and data-efficiency experiments.
- `notebooks/LIVE-visual-reconstruction.ipynb` performs the LIVE/NIMHANS workflow using BOLD fMRI data, VT ROI preprocessing, shared alignment, feature prediction, retrieval, reconstruction, and evaluation.
- `EDA/EDA_haxby.ipynb` provides exploratory analysis and quality checks for Haxby.
- `EDA/EDA_NSD.ipynb` and `EDA/nsd_eda.ipynb` provide memory-conscious NSD inspection and EDA.
- `EDA/live_dataset_eda.ipynb` provides LIVE dataset, BOLD, stimulus, annotation, pairing, and reconstruction-readiness checks.
- `scripts/build_retrieval_db.py` prepares the image-feature retrieval database.
- `scripts/run_ablation.py` runs the available ablation workflow.
- `scripts/run_nsd.py` and `scripts/run_live.py` provide dataset-specific execution entry points.
- `scripts/inspect_nsd.py` and `scripts/inspect_live.py` provide dataset inspection utilities.

The proposed reconstruction pipeline consists of:

```
fMRI
  ↓
Dataset-specific preprocessing
  ↓
Shared latent alignment
  ↓
Transformer neural encoder
  ↓
CLIP + DINOv2 feature prediction
  ↓
Dual diffusion-prior refinement
  ↓
Multimodal feature fusion
  ↓
Top-K image retrieval
  ↓
Stable Diffusion reconstruction
  ↓
Quantitative / qualitative evaluation
```

### Datasets and subject splits

Haxby uses S1-S4 for training, S5 for validation, and S6 for testing, with 307 VT voxels and TR = 3.0 s. The visual categories are face, house, cat, shoe, chair, bottle, and scissors.

NSD uses S1-S2 for training, S5 for validation, and S7 as the held-out test subject, with TR = 1.6 s. The project supports the reported 1-hour, 7-hour, and 40-hour data-efficiency experiments.

LIVE uses S1-S4 for training, S5 for validation, and S6 for testing. The reported BOLD volume is 147 × 144 × 36 × 165 with TR = 3.0 s. LIVE preprocessing uses the VT ROI, detrending, 0.01–0.1 Hz band-pass filtering, and voxel-wise standardization.

### Model configuration

The reported neural encoder uses sequence length 20, input latent dimension 128, Transformer embedding dimension 256, 4 Transformer layers, 8 attention heads, feed-forward dimension 512, GELU activation, and dropout 0.1.

The CLIP branch predicts a 512-dimensional representation and the DINOv2 branch predicts a 768-dimensional representation. The two representations are refined and fused before retrieval-guided image reconstruction.

### Alignment

The repository contains alignment implementations/baselines for:

- No Alignment
- Ridge Regression
- Procrustes
- Shared Response Model (SRM)
- Hyperalignment
- CCA
- Shared Latent Alignment

Procrustes and CCA are additional literature baselines and should not be presented as methods evaluated in the VGST1 paper unless explicitly reproduced as additional experiments.

### Evaluation

The project evaluates reconstruction using pixel-level, feature-level, semantic, retrieval, and alignment measures.

The reported manuscript results include SSIM, PixCorr, AlexNet feature similarity, Inception similarity, CLIP similarity, EfficientNet-B0 distance, SwAV distance, and Top-K retrieval accuracy.

The reported best SSIM values for the proposed full model are 0.8184 on Haxby, 0.5333 on NSD, and 0.3917 on LIVE. Reported peak retrieval accuracies are 90.05% for Haxby, 89.56% for NSD, and 82.15% for LIVE.

The numerical values above are paper-reported reference values. New experiments must calculate their metrics from their own held-out predictions and must not hard-code manuscript results into evaluation outputs.

### Reproducibility

For held-out evaluation, preprocessing, normalization, alignment, model fitting, and retrieval database construction must be performed without using test information.

The required subject splits are:

```
Haxby: S1-S4 → S5 → S6
NSD:   S1-S2 → S5 → S7
LIVE:  S1-S4 → S5 → S6
```

The retrieval database must not contain the target test image. Training-only image databases should be used when evaluating held-out samples.

Configurations, checkpoints, preprocessing settings, subject splits, random seeds, and retrieval settings should be saved with each experiment.

## FAQ

### What is the main difference between this project and MindEye/MindEye2?

This project focuses on a subject-agnostic framework based on shared latent alignment and combines CLIP semantic features with DINOv2 structural features. The predicted representations are refined using diffusion priors, fused for retrieval, and used to guide Stable Diffusion reconstruction across Haxby, NSD, and LIVE datasets.

### What are the main stages of the proposed framework?

The framework contains dataset-specific fMRI preprocessing, shared latent alignment, Transformer-based neural encoding, CLIP and DINOv2 feature prediction, dual diffusion-prior refinement, multimodal fusion, retrieval-guided matching, and Stable Diffusion image reconstruction.

### Where are the pretrained models and checkpoints?

The repository contains the model implementations and experiment interfaces. A trained checkpoint is required for faithful inference. Do not assume that source-code availability alone provides the trained weights used for the reported manuscript results.

### Can I train on limited data?

The NSD workflow includes the reported 1-hour, 7-hour, and 40-hour data-efficiency settings while keeping validation and test subjects fixed.

### What are the expected Transformer inputs and outputs?

The intended neural-encoder input is a sequence with shape `[B, 20, 128]`, producing a 256-dimensional shared representation. The CLIP branch targets `[B, 512]` and the DINOv2 branch targets `[B, 768]`.

### How should I report reproduced results?

Clearly distinguish **PAPER REPORTED RESULT** from **LOCALLY REPRODUCED RESULT**. Recomputed results should come from the actual experiment outputs and held-out test predictions.

## Citation

If you use this repository or the associated methodology, please cite the associated VGST1 manuscript and the original datasets and methods used in your experiment.

*Subject-Agnostic fMRI Visual Reconstruction via Shared Latent Alignment and CLIP–DINOv2 Guided Diffusion*

The manuscript reports experiments on Haxby, the Natural Scenes Dataset (NSD), and LIVE data collected at NIMHANS, Bengaluru, and acknowledges support from the Vision Group on Science and Technology (VGST), Government of Karnataka, under the GRE scheme (GRD No. 1115).

Related work includes MindEye, MindEye2, latent-diffusion-based fMRI reconstruction, CLIP, DINOv2, shared-subject/cross-subject alignment, and Stable Diffusion. Please cite the original publications for any method or dataset used directly.


The authors gratefully acknowledge the financial support provided by the Vision Group on Science and Technology (VGST), Government of Karnataka, India, through the funded project under the GRE scheme (GRD No. 1115).
