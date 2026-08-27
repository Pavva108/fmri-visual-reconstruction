"""Complete workflow example for fMRI visual reconstruction.

This module demonstrates how to use the full pipeline from data loading
through evaluation.
"""

import numpy as np
from pathlib import Path

# Import the complete toolkit
from fmri_reconstruction import (
    # Data pipeline
    load_fmri_data,
    normalize_labels,
    create_sequences,
    train_test_split,
    get_clip_preprocessor,
    get_dinov2_preprocessor,
    preprocess_image_batch,
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
    reconstruction_loss,
    # Pipeline
    fMRIVisualReconstructionPipeline,
    initialize_pipeline_from_checkpoints,
)


def example_1_data_loading_and_preprocessing():
    """Example 1: Load and preprocess fMRI data.
    
    Demonstrates:
    - Loading fMRI data with standardization
    - Normalizing category labels
    - Creating temporal sequences
    - Training/test splitting
    """
    print("=" * 60)
    print("EXAMPLE 1: Data Loading and Preprocessing")
    print("=" * 60)
    
    # Load fMRI data for Haxby dataset
    fmri_data, labels = load_fmri_data(
        dataset="haxby",
        subject=1,
        standardize=True,
        remove_rest=True,
    )
    print(f"✓ Loaded fMRI data shape: {fmri_data.shape}")
    print(f"✓ Labels shape: {labels.shape}")
    
    # Normalize labels (e.g., 'faces' → 'face', 'houses' → 'house')
    labels = normalize_labels(labels)
    print(f"✓ Normalized labels: {np.unique(labels)}")
    
    # Create temporal sequences (sliding window)
    sequences, seq_labels = create_sequences(
        fmri_data,
        labels,
        sequence_length=16,  # Use 16 time points
        step=1,
    )
    print(f"✓ Created sequences shape: {sequences.shape}")
    
    # Split data (stratified to maintain label distribution)
    train_idx, test_idx = train_test_split(
        seq_labels,
        test_size=0.2,
        random_state=42,
    )
    print(f"✓ Train/test split: {len(train_idx)} train, {len(test_idx)} test")
    
    return sequences, seq_labels, train_idx, test_idx


def example_2_model_inference():
    """Example 2: Neural model inference on fMRI data.
    
    Demonstrates:
    - Initializing neural encoder model
    - Projecting fMRI to CLIP embedding space
    - Projecting fMRI to DINOv2 embedding space
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Model Inference")
    print("=" * 60)
    
    # Get data
    sequences, labels, train_idx, test_idx = example_1_data_loading_and_preprocessing()
    
    # Initialize model
    model = MultimodalNeuralModel(
        input_dim=sequences.shape[-1],  # fMRI dimension
        latent_dim=256,
        clip_head=True,
        dino_head=True,
    )
    
    # Initialize projectors
    clip_projector = CLIPProjector(256, 512)  # 256 → 512 (CLIP)
    dino_projector = DINOv2Projector(256, 768)  # 256 → 768 (DINOv2)
    
    # Run inference on test set
    test_sequences = sequences[test_idx]
    
    model.eval()
    import torch
    with torch.no_grad():
        # Convert to torch
        x = torch.from_numpy(test_sequences).float()
        
        # Encode fMRI
        latent = model.encoder(x)
        print(f"✓ Latent embedding shape: {latent.shape}")
        
        # Project to multimodal spaces
        clip_emb = clip_projector(latent)
        dino_emb = dino_projector(latent)
        
        print(f"✓ CLIP embedding shape: {clip_emb.shape}")
        print(f"✓ DINOv2 embedding shape: {dino_emb.shape}")
    
    return test_sequences


def example_3_retrieval_and_guidance():
    """Example 3: Image retrieval for visual guidance.
    
    Demonstrates:
    - Building retrieval database
    - Retrieving images for guidance
    - Getting adaptive diffusion parameters
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Retrieval and Guidance")
    print("=" * 60)
    
    # Create a simple retrieval database
    database = RetrievalDatabase()
    
    # Simulate adding some images for categories
    # In practice, you would load embeddings from a trained model
    database.add_category("face")
    database.add_category("house")
    
    # Add dummy images with embeddings
    import torch
    for i in range(10):
        clip_emb = torch.randn(512)
        dino_emb = torch.randn(768)
        database.add_image(
            "face",
            f"dummy_face_{i}.jpg",
            clip_emb,
            dino_emb,
        )
    
    database.finalize(device="cpu")
    print(f"✓ Database has {len(database.categories)} categories")
    print(f"✓ Database finalized")
    
    # Retrieve
    query_clip = torch.randn(512)
    query_dino = torch.randn(768)
    
    results = database.retrieve_top_k(
        "face",
        query_clip,
        query_dino,
        k=3,
    )
    print(f"✓ Retrieved {len(results)} images")
    print(f"  - Top match: {results[0]['image_path']}")


def example_4_evaluation_metrics():
    """Example 4: Computing evaluation metrics.
    
    Demonstrates:
    - Computing feature similarity metrics
    - Computing image quality metrics
    - Computing loss functions
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Evaluation Metrics")
    print("=" * 60)
    
    import torch
    
    # Simulate generated and reference embeddings
    generated_emb = torch.randn(10, 512)  # 10 samples, CLIP dimension
    reference_emb = torch.randn(10, 512)
    
    # Normalize
    generated_emb = torch.nn.functional.normalize(generated_emb, dim=-1)
    reference_emb = torch.nn.functional.normalize(reference_emb, dim=-1)
    
    # Compute metrics
    metrics = compute_evaluation_metrics(
        generated_emb.numpy(),
        reference_emb.numpy(),
        metrics=["cosine_sim", "mse"],
    )
    print(f"✓ Computed metrics: {metrics}")
    
    # Compute loss
    loss_fn = CosineLoss()
    loss = loss_fn(generated_emb, reference_emb)
    print(f"✓ Cosine loss: {loss.item():.4f}")


def example_5_complete_pipeline():
    """Example 5: Complete end-to-end pipeline.
    
    Demonstrates:
    - Initializing the full pipeline
    - Processing a batch of samples
    - Evaluating results
    """
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Complete Pipeline")
    print("=" * 60)
    
    print("""
    Full pipeline workflow:
    
    1. Data Loading & Preprocessing
       └─ Load fMRI sequences
       └─ Normalize and split
    
    2. Model Inference
       └─ fMRI → Latent encoding
       └─ Latent → CLIP & DINOv2 embeddings
    
    3. Retrieval & Guidance
       └─ Query retrieval database
       └─ Get adaptive diffusion parameters
    
    4. Diffusion-based Generation
       └─ Use retrieved image as init
       └─ Apply diffusion with adaptive parameters
    
    5. Evaluation
       └─ Compute feature similarity metrics
       └─ Compute image quality metrics
    
    Usage:
    ------
    
    # Initialize pipeline
    pipeline = initialize_pipeline_from_checkpoints(
        model_checkpoint="models/fmri_encoder.pt",
        clip_checkpoint="models/clip_projector.pt",
        dino_checkpoint="models/dino_projector.pt",
        database_dir="retrieval_database/",
    )
    
    # Reconstruct batch
    results = pipeline.reconstruct_batch(
        fmri_sequences=sequences,
        labels=labels,
        image_paths=image_paths,
    )
    
    # Evaluate
    metrics = pipeline.evaluate_batch(
        original_images=original_paths,
        reconstructed_images=results["reconstructions"],
    )
    """)


# =========================================================
# USAGE WORKFLOWS
# =========================================================

def workflow_haxby_full_reconstruction():
    """Complete workflow for Haxby dataset reconstruction.
    
    Steps:
    1. Load and preprocess data
    2. Train/load model
    3. Build retrieval database
    4. Reconstruct images
    5. Evaluate quality
    """
    print("""
    ╔═══════════════════════════════════════════════════════╗
    ║   HAXBY DATASET COMPLETE RECONSTRUCTION WORKFLOW       ║
    ╚═══════════════════════════════════════════════════════╝
    
    from pathlib import Path
    import fmri_reconstruction as fmr
    
    # 1. LOAD AND PREPROCESS DATA
    # ─────────────────────────────
    fmri_data, labels = fmr.load_fmri_data(
        dataset="haxby",
        subject=1,
        standardize=True,
    )
    labels = fmr.normalize_labels(labels)
    sequences, seq_labels = fmr.create_sequences(fmri_data, labels)
    
    # 2. TRAIN OR LOAD MODEL
    # ──────────────────────
    model = fmr.MultimodalNeuralModel(
        input_dim=sequences.shape[-1],
        latent_dim=256,
    )
    clip_proj = fmr.CLIPProjector(256, 512)
    dino_proj = fmr.DINOv2Projector(256, 768)
    
    # Load pre-trained weights
    model.load_state_dict(torch.load("checkpoints/model.pt"))
    clip_proj.load_state_dict(torch.load("checkpoints/clip.pt"))
    dino_proj.load_state_dict(torch.load("checkpoints/dino.pt"))
    
    # 3. BUILD RETRIEVAL DATABASE
    # ───────────────────────────
    database = fmr.build_retrieval_database_from_haxby(
        categories=fmr.HAXBY_CATEGORIES,
    )
    
    # 4. INITIALIZE PIPELINE
    # ──────────────────────
    pipeline = fmr.fMRIVisualReconstructionPipeline(
        model=model,
        clip_projector=clip_proj,
        dino_projector=dino_proj,
        retrieval_database=database,
        diffusion_pipe=diffusion_pipe,
    )
    
    # 5. RECONSTRUCT IMAGES
    # ────────────────────
    results = pipeline.reconstruct_batch(
        fmri_sequences=sequences[:100],
        labels=labels[:100],
        image_paths=image_paths[:100],
        save_dir="outputs/reconstructions/",
    )
    
    # 6. EVALUATE
    # ──────────
    metrics = pipeline.evaluate_batch(
        original_images=image_paths[:100],
        reconstructed_images=results["reconstructions"],
        metrics=["ssim", "psnr", "cosine_sim"],
    )
    
    print(f"SSIM: {metrics['ssim']:.4f}")
    print(f"PSNR: {metrics['psnr']:.4f}")
    """)


def workflow_quick_start():
    """Quick start: Minimal code to get running."""
    print("""
    ╔═══════════════════════════════════════════════════════╗
    ║              QUICK START GUIDE                         ║
    ╚═══════════════════════════════════════════════════════╝
    
    # 1. Basic setup
    from pathlib import Path
    import fmri_reconstruction as fmr
    
    # 2. Load data
    fmri, labels = fmr.load_fmri_data("haxby", subject=1)
    labels = fmr.normalize_labels(labels)
    
    # 3. Create sequences
    seqs, seq_labels = fmr.create_sequences(fmri, labels)
    
    # 4. Initialize model
    model = fmr.MultimodalNeuralModel(seqs.shape[-1], 256)
    
    # 5. Run inference
    import torch
    x = torch.from_numpy(seqs[:10]).float()
    with torch.no_grad():
        embeddings = model.encoder(x)
    
    print(f"Generated embeddings: {embeddings.shape}")
    """)


if __name__ == "__main__":
    # Run examples
    example_1_data_loading_and_preprocessing()
    example_2_model_inference()
    example_3_retrieval_and_guidance()
    example_4_evaluation_metrics()
    example_5_complete_pipeline()
    
    # Show complete workflows
    print("\n")
    workflow_haxby_full_reconstruction()
    print("\n")
    workflow_quick_start()
