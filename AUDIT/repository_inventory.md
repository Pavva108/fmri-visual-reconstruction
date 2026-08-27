# Repository Inventory

This file lists every file discovered in the repository with basic metadata. Detailed per-file analysis (imports, dependencies, usage, duplication, completeness, and scientific relevance) will be filled in during Phase 1 follow-up passes.

Columns:
- path: repository-relative path
- type: file extension / type
- purpose: short inferred purpose or `See source`
- imports: `TBD` (will be populated by static inspection)
- dependencies: `TBD`
- used: `TBD`
- duplicated: `TBD`
- incomplete: `TBD`
- scientifically_relevant: `TBD`


| path | type | purpose | imports | dependencies | used | duplicated | incomplete | scientifically_relevant |
|---|---:|---|---|---|---|---|---|---|
| .gitignore | gitignore | Git ignore rules | TBD | none | yes | no | no | no |
| environment.yml | yml | Conda environment specification | TBD | see file | yes | no | no | no |
| pyproject.toml | toml | Python project metadata | TBD | see file | yes | no | no | no |
| requirements.txt | txt | pip requirements | TBD | see file | yes | no | no | no |
| LICENSE | txt | License file | - | - | yes | no | no | no |
| README_PY_MODULES.md | md | Documentation of python modules | TBD | - | yes | no | no | yes |

## Top-level scripts
| scripts/check_environment.py | py | Environment check utility | TBD | python stdlib, pip modules | yes | no | no | no |
| scripts/verify_installation.py | py | Verify installation and imports | TBD | python stdlib, project modules | yes | no | no | no |
| scripts/run_nsd.py | py | NSD experiment runner script | TBD | project modules | yes | no | maybe | yes |
| scripts/run_live.py | py | LIVE experiment runner script | TBD | project modules | yes | no | maybe | yes |
| scripts/run_ablation.py | py | Ablation experiment runner | TBD | project modules | yes | no | maybe | yes |
| scripts/inspect_nsd.py | py | NSD inspection helper | TBD | h5py/numpy | yes | no | no | yes |
| scripts/inspect_live.py | py | LIVE dataset inspection | TBD | nibabel/pandas | yes | no | no | yes |
| scripts/compare_alignment_methods.py | py | Alignment comparison on synthetic data | TBD | numpy/json | yes | no | no | yes |
| scripts/build_retrieval_db.py | py | Build retrieval DB from embeddings | TBD | numpy/json | yes | no | no | yes |

## Configs
| configs/haxby.yaml | yaml | Experiment configuration for HAXBY | TBD | used by scripts | yes | no | maybe | yes |
| configs/live.yaml | yaml | Experiment configuration for LIVE | TBD | used by scripts | yes | no | maybe | yes |
| configs/nsd.yaml | yaml | Experiment configuration for NSD | TBD | used by scripts | yes | no | maybe | yes |

## Notebooks
| notebooks/Haxby-visual-reconstruction.ipynb | ipynb | Analysis & demo notebook for Haxby | TBD | project modules, nilearn | yes | possible duplicates | maybe | yes |
| notebooks/NSD-visual-reconstruction.ipynb | ipynb | NSD demo notebook | TBD | project modules, h5py | yes | possible duplicates | maybe | yes |
| notebooks/LIVE-visual-reconstruction.ipynb | ipynb | LIVE demo notebook | TBD | project modules, nibabel | yes | possible duplicates | maybe | yes |

## Outputs and checkpoints
| outputs/README.md | md | Notes about outputs produced by experiments | TBD | - | yes | no | no | no |
| checkpoints/README_ONLY/README.md | md | Checkpoints/README placeholder | TBD | - | yes | no | yes | no |

## Dataset (Dataset/)
| Dataset/NSD/ | dir | NSD dataset artifacts (HDF5, annotations, figures) | - | large binary files: .hdf5, .npy | yes (data) | no | contains real data files | yes |
| Dataset/Live(NImhans)/ | dir | LIVE dataset artifacts (NIfTI BOLD, stimuli images, CSVs) | - | .nii.gz, .png, .csv | yes (data) | no | contains real data files | yes |
| Dataset/Haxby/ | dir | Haxby dataset README and possibly data | - | readme | yes | no | maybe | yes |

## Tests
| tests/test_shapes.py | py | Existing project test | TBD | pytest | yes | no | no | no |
| tests/test_alignment.py | py | Alignment API tests | TBD | project modules | yes | no | no | yes |
| tests/test_alignment_evaluator.py | py | Evaluator tests | TBD | project modules | yes | no | no | yes |
| tests/test_models.py | py | Transformer/projector forward tests | TBD | torch | yes | no | no | yes |
| tests/test_fusion.py | py | Fusion module tests | TBD | torch | yes | no | no | yes |
| tests/test_diffusion.py | py | Diffusion prior tests | TBD | torch | yes | no | no | yes |
| tests/test_datasets.py | py | Dataset factory tests | TBD | project modules | yes | no | no | yes |
| tests/test_retrieval.py | py | Retrieval tests | TBD | numpy | yes | no | no | yes |
| tests/test_trainer.py | py | Trainer checkpoint test | TBD | torch | yes | no | no | yes |
| tests/test_pipeline_retrieval.py | py | Integration pipeline test | TBD | numpy/torch | yes | no | no | yes |
| tests/run_*.py | py | Runnable test wrappers | TBD | project modules | yes | no | no | yes |

## Source package: src/fmri_reconstruction
| src/fmri_reconstruction/__init__.py | py | Package init | TBD | - | yes | no | no | yes |
| src/fmri_reconstruction/config.py | py | Experiment config loader | TBD | yaml | yes | no | no | yes |
| src/fmri_reconstruction/seed.py | py | Random seed utilities | TBD | numpy/random, torch | yes | no | no | yes |
| src/fmri_reconstruction/io.py | py | IO helpers | TBD | os/pathlib | yes | no | maybe | yes |
| src/fmri_reconstruction/visualization.py | py | Visualization helpers | TBD | matplotlib/numpy | yes | no | maybe | yes |
| src/fmri_reconstruction/stimulus.py | py | Stimulus handling helpers | TBD | PIL/numpy | yes | no | maybe | yes |
| src/fmri_reconstruction/metrics.py | py | Evaluation metrics (SSIM, PixCorr, etc.) | TBD | skimage/torch | yes | no | maybe | yes |

### Preprocessing & alignment
| src/fmri_reconstruction/preprocessing.py | py | Detrending, filtering, standardization | TBD | scipy/numpy | yes | no | maybe | yes |
| src/fmri_reconstruction/alignment.py | py | Alignment methods (SharedLatent + baselines) | TBD | sklearn/numpy | yes | possible duplicates avoided | maybe | yes |

### Data subpackage
| src/fmri_reconstruction/data/__init__.py | py | data package init | TBD | - | yes | no | no | yes |
| src/fmri_reconstruction/data/loaders.py | py | Resolve dataset roots and paths | TBD | os/pathlib | yes | no | no | yes |
| src/fmri_reconstruction/data/haxby.py | py | Haxby dataset helpers | TBD | nilearn/pandas | yes | no | maybe | yes |
| src/fmri_reconstruction/data/nsd.py | py | NSD download/load helpers | TBD | huggingface_hub/numpy | yes | no | maybe | yes |
| src/fmri_reconstruction/data/live.py | py | LIVE dataset helpers | TBD | nibabel/pandas | yes | no | maybe | yes |
| src/fmri_reconstruction/data/sequences.py | py | Sequence generation utilities | TBD | numpy | yes | no | no | yes |
| src/fmri_reconstruction/data/datasets.py | py | Synthetic dataset & factory | TBD | numpy/torch | yes | no | no | yes |

### Models
| src/fmri_reconstruction/models/__init__.py | py | models package init | TBD | - | yes | no | no | yes |
| src/fmri_reconstruction/models/transformer.py | py | Transformer implementation | TBD | torch | yes | no | maybe | yes |
| src/fmri_reconstruction/models/multimodal.py | py | Multimodal model wrapper | TBD | torch | yes | no | maybe | yes |
| src/fmri_reconstruction/models/projectors.py | py | CLIP/DINO projectors | TBD | torch | yes | no | maybe | yes |
| src/fmri_reconstruction/models/diffusion_prior.py | py | Diffusion prior implementations | TBD | torch | yes | no | maybe | yes |
| src/fmri_reconstruction/models/fusion.py | py | Fusion module (concat/gated/attention) | TBD | torch | yes | no | maybe | yes |
| src/fmri_reconstruction/models/losses.py | py | Loss functions (CLIP/DINO joint) | TBD | torch | yes | no | maybe | yes |

### Embeddings
| src/fmri_reconstruction/embeddings/clip.py | py | CLIP embedding helpers | TBD | transformers/clip | yes | no | maybe | yes |
| src/fmri_reconstruction/embeddings/dino.py | py | DINO embedding helpers | TBD | torch/hub | yes | no | maybe | yes |

### Retrieval
| src/fmri_reconstruction/retrieval/cosine.py | py | cosine retrieval utilities | TBD | numpy | yes | no | no | yes |
| src/fmri_reconstruction/retrieval/database.py | py | Retrieval database save/load | TBD | numpy/json | yes | no | no | yes |
| src/fmri_reconstruction/retrieval/split_guard.py | py | Split integrity checks | TBD | typing | yes | no | no | yes |
| src/fmri_reconstruction/retrieval/interface.py | py | Retrieval high-level interface | TBD | project modules | yes | no | no | yes |

### Reconstruction
| src/fmri_reconstruction/reconstruction/stable_diffusion.py | py | Stable Diffusion reconstruction interface | TBD | diffusers | yes | no | maybe | yes |

### Evaluation
| src/fmri_reconstruction/evaluation/feature_metrics.py | py | Feature-level metrics (AlexNet/CLIP) | TBD | torchvision | yes | no | maybe | yes |
| src/fmri_reconstruction/evaluation/image_metrics.py | py | Image metrics (SSIM, PixCorr) | TBD | skimage | yes | no | maybe | yes |
| src/fmri_reconstruction/evaluation/retrieval_metrics.py | py | Retrieval metrics (top-k) | TBD | numpy | yes | no | no | yes |
| src/fmri_reconstruction/evaluation/semantic_metrics.py | py | Semantic evaluation (CLIP similarities) | TBD | clip/torch | yes | no | maybe | yes |

### Ablation
| src/fmri_reconstruction/ablation/experiments.py | py | Ablation definitions | TBD | project modules | yes | no | maybe | yes |
| src/fmri_reconstruction/ablation/runner.py | py | Ablation runner script | TBD | project modules | yes | no | maybe | yes |

### Training
| src/fmri_reconstruction/training/trainer.py | py | SimpleTrainer scaffold | TBD | torch | yes | no | maybe | yes |
| src/fmri_reconstruction/training/checkpoint.py | py | checkpoint save/load helpers | TBD | torch | yes | no | no | yes |
| src/fmri_reconstruction/training/loops.py | py | training loop scaffolds (placeholder) | TBD | torch | yes | no | maybe | yes |
| src/fmri_reconstruction/training/checkpoints.py | py | training checkpoint utilities (placeholder) | TBD | torch | yes | no | maybe | yes |

## Documentation and docs
| docs/vgst1-methodology.pdf | pdf | Primary paper source (VGST1 methodology) | - | - | yes | no | no | yes |
| README.md (root) | md | repository README (may be missing/partial) | TBD | - | yes | no | maybe | yes |

----

This is an initial, file-level inventory. Next steps in the audit will:

- Extract imports from each Python file and populate the `imports` and `dependencies` columns.
- Mark whether a file is actually used by the canonical package (import graph).
- Detect duplicate implementations and placeholders ("NotImplementedError", "TODO", "FIXME").
- Perform per-file static checks against the methodology PDF and notebooks.

Generated by automated audit pass 1.

