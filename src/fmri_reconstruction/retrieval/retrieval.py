"""Retrieval system for image-based guidance in fMRI visual reconstruction.

Builds embedding databases from stimulus images and performs similarity-based
retrieval to guide diffusion models and reconstruction processes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter


# =========================================================
# CATEGORY MANAGEMENT
# =========================================================

HAXBY_CATEGORIES = ["face", "house", "cat", "shoe", "chair", "bottle", "scissors"]

# Adaptive weights for shape-focused categories
SHAPE_FOCUS_CATEGORIES = ["chair", "bottle", "scissors"]
DEFAULT_CLIP_WEIGHT = 0.30
DEFAULT_DINO_WEIGHT = 0.70
SHAPE_FOCUS_CLIP_WEIGHT = 0.15
SHAPE_FOCUS_DINO_WEIGHT = 0.85


def get_category_weights(label: str) -> Tuple[float, float]:
    """Get retrieval weights for a category.
    
    Categories with strong structural patterns (chair, bottle, scissors)
    receive higher DINOv2 weight for shape-focused retrieval.
    
    Args:
        label: Category label
        
    Returns:
        Tuple of (clip_weight, dino_weight)
    """
    if label in SHAPE_FOCUS_CATEGORIES:
        return SHAPE_FOCUS_CLIP_WEIGHT, SHAPE_FOCUS_DINO_WEIGHT
    return DEFAULT_CLIP_WEIGHT, DEFAULT_DINO_WEIGHT


# =========================================================
# PROMPTING FOR DIFFUSION MODELS
# =========================================================

PROMPT_TEMPLATES = {
    "face": "high quality grayscale realistic human face, detailed, clear features",
    "house": "high quality grayscale realistic house, architectural details",
    "cat": "high quality grayscale realistic cat, clear features",
    "bottle": "high quality grayscale realistic bottle, defined shape",
    "chair": "high quality grayscale realistic chair, structural details",
    "scissors": "high quality grayscale realistic scissors, sharp details",
    "shoe": "high quality grayscale realistic shoe, clear design",
}


def get_text_prompt(label: str, detail_level: str = "high") -> str:
    """Get text prompt for diffusion model.
    
    Args:
        label: Category label
        detail_level: "low", "medium", or "high" detail
        
    Returns:
        Text prompt for diffusion model
    """
    base_prompt = PROMPT_TEMPLATES.get(
        label,
        "high quality grayscale realistic object"
    )
    
    if detail_level == "low":
        return f"simple {base_prompt}"
    elif detail_level == "medium":
        return base_prompt
    else:  # high
        return base_prompt + ", high detail, professional quality"


# =========================================================
# DATABASE BUILDING
# =========================================================

class RetrievalDatabase:
    """Stores and manages embeddings for image retrieval."""
    
    def __init__(self):
        self.database: Dict[str, Dict] = {}
    
    def add_category(self, category: str) -> None:
        """Add a new category to the database.
        
        Args:
            category: Category name
        """
        if category not in self.database:
            self.database[category] = {
                "clip": [],
                "dino": [],
                "paths": [],
                "clip_tensor": None,
                "dino_tensor": None,
            }
    
    def add_image(
        self,
        category: str,
        image_path: str | Path,
        clip_embedding: torch.Tensor,
        dino_embedding: torch.Tensor,
    ) -> None:
        """Add an image and its embeddings to the database.
        
        Args:
            category: Category name
            image_path: Path to image
            clip_embedding: CLIP embedding (512,)
            dino_embedding: DINOv2 embedding (768,)
        """
        if category not in self.database:
            self.add_category(category)
        
        db = self.database[category]
        db["clip"].append(clip_embedding.cpu())
        db["dino"].append(dino_embedding.cpu())
        db["paths"].append(str(image_path))
    
    def finalize(self, device: str = "cpu") -> None:
        """Convert lists to stacked tensors (call after all additions).
        
        Args:
            device: Device to move tensors to
        """
        for category, data in self.database.items():
            if len(data["clip"]) > 0:
                data["clip_tensor"] = torch.stack(data["clip"]).to(device)
                data["dino_tensor"] = torch.stack(data["dino"]).to(device)
            
            print(f"  {category}: {len(data['paths'])} images")
    
    def retrieve_top_k(
        self,
        category: str,
        clip_query: torch.Tensor,
        dino_query: torch.Tensor,
        k: int = 5,
        use_fusion: bool = True,
    ) -> List[Tuple[str, float, float, float]]:
        """Retrieve top-k images for a category.
        
        Args:
            category: Category name
            clip_query: CLIP query embedding (512,)
            dino_query: DINOv2 query embedding (768,)
            k: Number of top results
            use_fusion: Use weighted fusion or separate scores
            
        Returns:
            List of (path, clip_score, dino_score, fusion_score)
        """
        if category not in self.database:
            raise ValueError(f"Category {category} not in database")
        
        data = self.database[category]
        
        if data["clip_tensor"] is None or len(data["paths"]) == 0:
            raise ValueError(f"Category {category} is empty")
        
        # Compute similarities
        with torch.no_grad():
            clip_scores = (clip_query @ data["clip_tensor"].T).squeeze()
            dino_scores = (dino_query @ data["dino_tensor"].T).squeeze()
            
            if use_fusion:
                clip_w, dino_w = get_category_weights(category)
                fusion_scores = clip_w * clip_scores + dino_w * dino_scores
            else:
                fusion_scores = (clip_scores + dino_scores) / 2.0
        
        # Get top-k
        if k >= len(data["paths"]):
            top_k_indices = torch.argsort(fusion_scores, descending=True)
        else:
            top_k_indices = torch.topk(fusion_scores, k)[1]
        
        results = []
        for idx in top_k_indices:
            idx = idx.item()
            results.append((
                data["paths"][idx],
                float(clip_scores[idx]),
                float(dino_scores[idx]),
                float(fusion_scores[idx]),
            ))
        
        return results
    
    def retrieve_best_image(
        self,
        category: str,
        clip_query: torch.Tensor,
        dino_query: torch.Tensor,
    ) -> Tuple[str, float, float, float]:
        """Retrieve single best image for a category.
        
        Args:
            category: Category name
            clip_query: CLIP query embedding (512,)
            dino_query: DINOv2 query embedding (768,)
            
        Returns:
            Tuple of (path, clip_score, dino_score, fusion_score)
        """
        results = self.retrieve_top_k(category, clip_query, dino_query, k=1)
        return results[0]
    
    def save(self, save_dir: str | Path) -> None:
        """Save database to disk.
        
        Args:
            save_dir: Directory to save database
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        for category, data in self.database.items():
            if len(data["paths"]) == 0:
                continue
            
            cat_dir = save_dir / category
            cat_dir.mkdir(exist_ok=True)
            
            np.save(cat_dir / "clip_embeddings.npy", data["clip_tensor"].cpu().numpy())
            np.save(cat_dir / "dino_embeddings.npy", data["dino_tensor"].cpu().numpy())
            
            paths_file = cat_dir / "paths.txt"
            with open(paths_file, "w") as f:
                for p in data["paths"]:
                    f.write(f"{p}\n")
    
    @classmethod
    def load(cls, load_dir: str | Path, device: str = "cpu") -> RetrievalDatabase:
        """Load database from disk.
        
        Args:
            load_dir: Directory containing saved database
            device: Device to load tensors to
            
        Returns:
            Loaded database
        """
        load_dir = Path(load_dir)
        db = cls()
        
        for cat_dir in load_dir.iterdir():
            if not cat_dir.is_dir():
                continue
            
            category = cat_dir.name
            db.add_category(category)
            
            clip_embeddings = np.load(cat_dir / "clip_embeddings.npy")
            dino_embeddings = np.load(cat_dir / "dino_embeddings.npy")
            
            db.database[category]["clip"] = [
                torch.from_numpy(c) for c in clip_embeddings
            ]
            db.database[category]["dino"] = [
                torch.from_numpy(d) for d in dino_embeddings
            ]
            
            paths_file = cat_dir / "paths.txt"
            with open(paths_file, "r") as f:
                db.database[category]["paths"] = [
                    line.strip() for line in f.readlines()
                ]
            
            db.database[category]["clip_tensor"] = torch.from_numpy(
                clip_embeddings
            ).to(device)
            db.database[category]["dino_tensor"] = torch.from_numpy(
                dino_embeddings
            ).to(device)
        
        return db


# =========================================================
# STRUCTURE-BASED RETRIEVAL
# =========================================================

def compute_edge_similarity(path1: str | Path, 
                            path2: str | Path,
                            size: Tuple[int, int] = (128, 128)) -> float:
    """Compute edge-based structural similarity between two images.
    
    Uses Canny edge detection to compare structural patterns,
    useful for shape-focused categories.
    
    Args:
        path1: Path to first image
        path2: Path to second image
        size: Size to resize images to
        
    Returns:
        Edge similarity score (-1 to 1)
    """
    img1 = Image.open(path1).convert("L").resize(size)
    img2 = Image.open(path2).convert("L").resize(size)
    
    # Extract edges
    edges1 = img1.filter(ImageFilter.FIND_EDGES)
    edges2 = img2.filter(ImageFilter.FIND_EDGES)
    
    # Convert to numpy
    e1 = np.array(edges1).astype(np.float32) / 255.0
    e2 = np.array(edges2).astype(np.float32) / 255.0
    
    # Compute correlation
    score = np.corrcoef(e1.ravel(), e2.ravel())[0, 1]
    
    if np.isnan(score):
        return 0.0
    
    return float(score)


def retrieve_with_structure_score(
    database: RetrievalDatabase,
    category: str,
    clip_query: torch.Tensor,
    dino_query: torch.Tensor,
    reference_path: str | Path,
    k_candidates: int = 30,
    structure_weight: float = 0.30,
) -> Tuple[str, float, float, float]:
    """Retrieve image with structure-based reranking.
    
    Useful for categories like chair and bottle where structural
    patterns are important.
    
    Args:
        database: Retrieval database
        category: Category name
        clip_query: CLIP query embedding
        dino_query: DINOv2 query embedding
        reference_path: Path to reference image for structure matching
        k_candidates: Number of candidates to rerank
        structure_weight: Weight for structure score in final fusion
        
    Returns:
        Tuple of (path, clip_score, dino_score, final_score)
    """
    # Get top-k candidates
    clip_w, dino_w = get_category_weights(category)
    candidates = database.retrieve_top_k(
        category, clip_query, dino_query, k=k_candidates
    )
    
    best_score = -999
    best_result = candidates[0]
    
    # Rerank with structure score
    for path, clip_s, dino_s, fusion_s in candidates:
        structure_s = compute_edge_similarity(path, reference_path)
        
        # Combine scores
        final_score = (
            clip_w * clip_s +
            dino_w * dino_s +
            structure_weight * structure_s
        )
        
        if final_score > best_score:
            best_score = final_score
            best_result = (path, clip_s, dino_s, final_score)
    
    return best_result


# =========================================================
# ADAPTIVE DIFFUSION PARAMETERS
# =========================================================

def get_adaptive_strength(
    similarity_score: float,
    category: Optional[str] = None,
) -> float:
    """Get adaptive denoising strength based on retrieval quality.
    
    Higher similarity = lower strength (less diffusion needed).
    
    Args:
        similarity_score: Fusion similarity score (0-1)
        category: Category name (optional override)
        
    Returns:
        Denoising strength (0-1), typical range 0.08-0.24
    """
    if category in SHAPE_FOCUS_CATEGORIES:
        # Shape categories use lower strength
        return 0.08
    
    if similarity_score > 0.92:
        return 0.08
    elif similarity_score > 0.85:
        return 0.12
    elif similarity_score > 0.75:
        return 0.16
    else:
        return 0.22


def get_adaptive_guidance(
    clip_score: float,
    category: Optional[str] = None,
) -> float:
    """Get adaptive classifier-free guidance scale.
    
    Higher CLIP score = lower guidance (more trust in conditioning).
    
    Args:
        clip_score: CLIP similarity score (0-1)
        category: Category name (optional override)
        
    Returns:
        Guidance scale, typical range 4.5-8.5
    """
    if category in SHAPE_FOCUS_CATEGORIES:
        # Shape categories use lower guidance
        return 4.5
    
    if clip_score > 0.90:
        return 4.5
    elif clip_score > 0.80:
        return 5.5
    elif clip_score > 0.70:
        return 6.5
    else:
        return 8.0


def get_adaptive_num_steps(
    similarity_score: float,
) -> int:
    """Get adaptive number of diffusion steps.
    
    Args:
        similarity_score: Fusion similarity score (0-1)
        
    Returns:
        Number of inference steps (20-50)
    """
    if similarity_score > 0.9:
        return 20
    elif similarity_score > 0.8:
        return 25
    elif similarity_score > 0.7:
        return 30
    else:
        return 50


# =========================================================
# QUERY PREPARATION
# =========================================================

def prepare_clip_query(
    fmri_latent: torch.Tensor,
    clip_projector: torch.nn.Module,
    device: str = "cpu",
) -> torch.Tensor:
    """Prepare CLIP query from fMRI latent.
    
    Args:
        fmri_latent: fMRI latent embedding (1, 256)
        clip_projector: fMRI to CLIP projector
        device: Device
        
    Returns:
        CLIP query embedding (1, 512)
    """
    with torch.no_grad():
        clip_query = clip_projector(fmri_latent)
        clip_query = F.normalize(clip_query, dim=-1)
    
    return clip_query


def prepare_dino_query(
    fmri_latent: torch.Tensor,
    dino_projector: torch.nn.Module,
    device: str = "cpu",
) -> torch.Tensor:
    """Prepare DINOv2 query from fMRI latent.
    
    Args:
        fmri_latent: fMRI latent embedding (1, 256)
        dino_projector: fMRI to DINOv2 projector
        device: Device
        
    Returns:
        DINOv2 query embedding (1, 768)
    """
    with torch.no_grad():
        dino_query = dino_projector(fmri_latent)
        dino_query = F.normalize(dino_query, dim=-1)
    
    return dino_query


# =========================================================
# FULL RETRIEVAL PIPELINE
# =========================================================

def full_retrieval_pipeline(
    database: RetrievalDatabase,
    fmri_latent: torch.Tensor,
    category: str,
    clip_projector: torch.nn.Module,
    dino_projector: torch.nn.Module,
    reference_image: Optional[str | Path] = None,
    use_structure: bool = False,
    device: str = "cpu",
) -> Dict[str, any]:
    """Complete retrieval pipeline from fMRI to image.
    
    Args:
        database: Retrieval database
        fmri_latent: fMRI latent embedding
        category: Target category
        clip_projector: fMRI to CLIP projector
        dino_projector: fMRI to DINOv2 projector
        reference_image: Optional reference for structure matching
        use_structure: Use structure-based reranking
        device: Device
        
    Returns:
        Dictionary with retrieval results and diffusion parameters
    """
    # Prepare queries
    clip_query = prepare_clip_query(fmri_latent, clip_projector, device)
    dino_query = prepare_dino_query(fmri_latent, dino_projector, device)
    
    # Retrieve image
    if use_structure and reference_image is not None:
        image_path, clip_s, dino_s, final_s = retrieve_with_structure_score(
            database, category, clip_query, dino_query, reference_image
        )
    else:
        image_path, clip_s, dino_s, final_s = database.retrieve_best_image(
            category, clip_query, dino_query
        )
    
    # Get text prompt
    prompt = get_text_prompt(category)
    
    # Get adaptive parameters
    strength = get_adaptive_strength(final_s, category)
    guidance = get_adaptive_guidance(clip_s, category)
    num_steps = get_adaptive_num_steps(final_s)
    
    return {
        "image_path": str(image_path),
        "clip_score": clip_s,
        "dino_score": dino_s,
        "fusion_score": final_s,
        "text_prompt": prompt,
        "diffusion_strength": strength,
        "guidance_scale": guidance,
        "num_inference_steps": num_steps,
    }


__all__ = [
    "HAXBY_CATEGORIES",
    "SHAPE_FOCUS_CATEGORIES",
    "get_category_weights",
    "get_text_prompt",
    "RetrievalDatabase",
    "compute_edge_similarity",
    "retrieve_with_structure_score",
    "get_adaptive_strength",
    "get_adaptive_guidance",
    "get_adaptive_num_steps",
    "prepare_clip_query",
    "prepare_dino_query",
    "full_retrieval_pipeline",
]
