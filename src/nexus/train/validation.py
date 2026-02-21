"""
Validation during training: build pipeline with trained transformer, generate sample
images, and log to TensorBoard, WandB, or MLflow.
"""

import logging
from typing import Any

import numpy as np
import torch
from diffusers.training_utils import free_memory

logger = logging.getLogger(__name__)

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False


def run_validation(
    pipeline_cls: type,
    transformer: torch.nn.Module,
    validation_prompt: str,
    accelerator: Any,
    step: int,
    num_images: int = 4,
    seed: int | None = 42,
    resolution: int = 512,
    weight_dtype: torch.dtype = torch.float16,
    pretrained_path: str | None = None,
) -> None:
    """
    Build a pipeline with the trained transformer, run inference, and log images
    to the configured trackers (TensorBoard, WandB, MLflow).
    """
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
    for _ in range(num_images):
        with torch.autocast(device_type=accelerator.device.type, dtype=weight_dtype):
            out = pipeline(
                prompt=validation_prompt,
                height=resolution,
                width=resolution,
                generator=generator,
            )
        images.append(out.images[0])

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images("validation", np_images, step, dataformats="NHWC")
        if tracker.name == "wandb" and WANDB_AVAILABLE:
            tracker.log(
                {
                    "validation": [
                        wandb.Image(img, caption=f"{i}: {validation_prompt}")
                        for i, img in enumerate(images)
                    ]
                },
                step=step,
            )
        if tracker.name == "mlflow" and MLFLOW_AVAILABLE:
            for i, img in enumerate(images):
                mlflow.log_image(
                    np.asarray(img),
                    artifact_file=f"validation/step_{step}_img_{i}.png",
                    step=step,
                )

    del pipeline
    free_memory()
    logger.info(f"Validation at step {step}: generated {num_images} images")
