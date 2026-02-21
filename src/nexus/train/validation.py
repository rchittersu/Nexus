"""
Validation during training: build pipeline with trained transformer, generate sample
images, and log to MLflow.

Entries format: list of {prompt, source, num_images}.
  - prompt: text prompt
  - source: image path for img2img, null for t2i
  - num_images: images to generate per entry (default 1)
"""

import logging
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from diffusers.training_utils import free_memory
from tqdm.auto import tqdm

from nexus.utils.log_utils import log_validation_images_to_mlflow

logger = logging.getLogger(__name__)


def _load_image_if_path(path: str | None) -> Image.Image | None:
    """Load image from path; return None if path is None or empty."""
    if not path:
        return None
    return Image.open(path).convert("RGB")


def run_validation(
    pipeline_cls: type,
    transformer: torch.nn.Module,
    accelerator: Any,
    step: int,
    output_dir: str | Path,
    resolution: int = 512,
    weight_dtype: torch.dtype = torch.float16,
    pretrained_path: str | None = None,
    inference_steps: int = 4,
    guidance_scale: float = 1.0,
    seed: int | None = 42,
    *,
    validation_entries: list[dict] | None = None,
) -> None:
    """
    Build a pipeline with the trained transformer, run inference, and log images to MLflow.

    validation_entries: list of {prompt, source, num_images}. source=null for t2i, path for img2img.
    num_images per entry (default 1).
    """
    normalized: list[tuple[str, Image.Image | None, int]] = []
    if validation_entries:
        for e in validation_entries:
            prompt = e.get("prompt") or e.get("text", "")
            src_path = e.get("source") or e.get("image")  # support both keys
            n = int(e.get("num_images", 1))
            normalized.append((prompt, _load_image_if_path(src_path), n))

    if not normalized:
        logger.warning("Validation skipped: no entries")
        return

    pipeline = pipeline_cls.from_pretrained(
        pretrained_path,
        transformer=transformer,
        torch_dtype=weight_dtype,
    )
    pipeline = pipeline.to(dtype=weight_dtype)
    pipeline.enable_model_cpu_offload()
    pipeline.set_progress_bar_config(disable=True)

    generator = (
        torch.Generator(device=accelerator.device).manual_seed(seed) if seed is not None else None
    )

    images = []
    for prompt, image, num_images in tqdm(normalized, desc="Validation entries", leave=False):
        for _ in range(num_images):
            with torch.autocast(device_type=accelerator.device.type, dtype=weight_dtype):
                out = pipeline(
                    prompt=prompt,
                    image=image,
                    height=resolution,
                    width=resolution,
                    generator=generator,
                    num_inference_steps=inference_steps,
                    guidance_scale=guidance_scale,
                )
            images.append(out.images[0])

    for tracker in accelerator.trackers:
        if tracker.name == "mlflow":
            log_validation_images_to_mlflow(images, step, output_dir)

    del pipeline
    free_memory()
    logger.info(f"Validation at step {step}: generated {len(images)} images")
