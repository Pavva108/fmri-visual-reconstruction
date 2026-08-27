# Dataset Setup

This guide records the dataset-specific settings used by the repository. The
datasets themselves, pretrained models, checkpoints, and generated outputs are
not stored in Git. Obtain access to each dataset under its own terms and keep
local data outside the repository or in paths configured for your environment.

The YAML files in `configs/` are the repository's source of truth for subject
splits and model-facing dataset settings:

- `configs/haxby.yaml`
- `configs/nsd.yaml`
- `configs/live.yaml`

The complete workflows are provided in the corresponding notebooks:

- `notebooks/Haxby-visual-reconstruction.ipynb`
- `notebooks/NSD-visual-reconstruction.ipynb`
- `notebooks/LIVE-visual-reconstruction.ipynb`

## General principles

1. Obtain the required permissions and data access before running an
	experiment.
2. Do not commit raw or processed datasets, model weights, checkpoints, or
	generated reconstructions.
3. Keep preprocessing, normalization, alignment, model fitting, and retrieval
	database construction independent of the held-out test subject and test
	images.
4. Save the configuration, subject split, preprocessing settings, random seed,
	and retrieval settings with each experiment.
5. Treat results computed from a local run as locally reproduced results. Do
	not copy paper-reported values into evaluation outputs.

## Haxby

The Haxby configuration defines:

- Training subjects: S1-S4
- Validation subject: S5
- Test subject: S6
- 307 voxel features
- TR: 3.0 seconds
- Sequence length: 20
- Input dimension: 128
- Latent dimension: 256
- CLIP dimension: 512
- DINOv2 dimension: 768
- Retrieval database size: top 5

The configured Haxby retrieval categories are face, house, cat, shoe, chair,
bottle, and scissors. The reusable Haxby retrieval helper is
`build_retrieval_database_from_haxby` in `src/fmri_reconstruction/pipeline.py`.

Use `EDA/EDA_haxby.ipynb` for exploratory checks and
`notebooks/Haxby-visual-reconstruction.ipynb` for the reconstruction workflow.

## Natural Scenes Dataset (NSD)

The NSD configuration defines:

- Training subjects: S1-S2
- Validation subject: S5
- Held-out test subject: S7
- TR: 1.6 seconds
- Sequence length: 20
- Input dimension: 128
- Latent dimension: 256
- CLIP dimension: 512
- DINOv2 dimension: 768
- Retrieval database size: top 5
- Data-efficiency settings: 1, 7, and 40 hours

The configuration also records subject-specific voxel-feature counts and
111,000 reported processed samples. These values describe the repository's
configured experiment metadata; they are not a substitute for preparing the
corresponding NSD data.

Use `scripts/inspect_nsd.py` to inspect an NSD HDF5 input and
`EDA/EDA_NSD.ipynb` or `EDA/nsd_eda.ipynb` for exploratory checks. The complete
workflow is in `notebooks/NSD-visual-reconstruction.ipynb`.

## LIVE / NIMHANS

The repository refers to the LIVE data as data collected at NIMHANS,
Bengaluru. The LIVE configuration defines:

- Training subjects: S1-S4
- Validation subject: S5
- Test subject: S6
- BOLD shape: 147 x 144 x 36 x 165
- TR: 3.0 seconds
- ROI: VT
- Detrending: enabled
- Band-pass range: 0.01-0.1 Hz
- Voxel-wise standardization: enabled
- Retrieval database size: top 5

Use `scripts/inspect_live.py` to validate a BOLD input's shape and
`EDA/live_dataset_eda.ipynb` for checks covering BOLD data, stimuli,
annotations, pairing, and reconstruction readiness. The complete workflow is
in `notebooks/LIVE-visual-reconstruction.ipynb`.

## Reproducibility and attribution

For each held-out evaluation, fit data-dependent preprocessing and alignment
steps using training data only. Do not place the target test image in the
retrieval database. Keep paper-reported metrics separate from metrics
recomputed from local predictions.

The dataset names, splits, and preprocessing settings in this guide summarize
the current repository configuration. If a local experiment uses different
settings, record those settings with its outputs rather than silently changing
the repository defaults.