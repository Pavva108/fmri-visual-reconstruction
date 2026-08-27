from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from PIL import Image


class StableDiffusionImg2Img:
    """Thin wrapper around Diffusers Stable Diffusion img2img.

    Features:
    - Centralized pipeline loading with device and dtype handling
    - Optionally disables the safety checker and enables attention slicing
    - Small convenience `generate` method matching notebook usage
    """

    def __init__(
        self,
        model_id: str,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None,
        disable_safety: bool = True,
        enable_xformers: bool = False,
    ):
        from diffusers import StableDiffusionImg2ImgPipeline

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if torch_dtype is None:
            torch_dtype = torch.float16 if str(self.device).startswith("cuda") else torch.float32

        self.pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
        ).to(self.device)

        if disable_safety and hasattr(self.pipe, "safety_checker"):
            try:
                self.pipe.safety_checker = None
            except Exception:
                pass

        # Enable attention slicing to reduce memory peak when available.
        try:
            self.pipe.enable_attention_slicing()
        except Exception:
            pass

        # Optional xformers-backed attention for further memory savings.
        if enable_xformers:
            try:
                # Not all Diffusers versions expose this helper; try safely.
                self.pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass

    @torch.no_grad()
    def generate(
        self,
        init_image: Image.Image,
        prompt: str,
        strength: float = 0.55,
        guidance_scale: float = 7.5,
        num_inference_steps: int = 30,
        generator_seed: Optional[int] = 42,
    ) -> Image.Image:
        """Run img2img and return a PIL Image.

        The `generator_seed` is optional; when provided a deterministic torch
        Generator is created for reproducible outputs on the selected device.
        """
        generator = None
        try:
            if generator_seed is not None:
                if str(self.device).startswith("cuda"):
                    generator = torch.Generator(device=self.device).manual_seed(generator_seed)
                else:
                    generator = torch.Generator().manual_seed(generator_seed)
        except Exception:
            generator = None

        result = self.pipe(
            prompt=prompt,
            image=init_image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )
        return result.images[0]


__all__ = ["StableDiffusionImg2Img"]
