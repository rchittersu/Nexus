"""
Standalone validation tool: load model from config, run inference on val.json,
write to outdir/validation/, and log to MLflow run.

Usage:
    python -m nexus.tools.validate --config configs/.../t2i_distillation.yaml --val_json /path/to/val.json
    python -m nexus.tools.validate --config ... --val_json val.json --checkpoint checkpoint-500
    python -m nexus.tools.validate --config ... --val_json val.json --output_dir /path/to/out
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image
from diffusers.training_utils import free_memory
from peft import LoraConfig
from tqdm.auto import tqdm

from nexus.train.config import load_config
from nexus.utils.checkpoint_utils import resolve_checkpoint_path
from nexus.utils.log_utils import activate_mlflow_run_and_log_validation, get_output_dir

logger = logging.getLogger(__name__)

# Pipeline registry - must match main.py
_PIPELINE_COMPONENTS = {}
try:
    from diffusers import AutoencoderKLFlux2, Flux2KleinPipeline, FlowMatchEulerDiscreteScheduler

    _PIPELINE_COMPONENTS[Flux2KleinPipeline] = {
        "vae": (AutoencoderKLFlux2, "vae"),
        "scheduler": (FlowMatchEulerDiscreteScheduler, "scheduler"),
    }
except ImportError:
    pass


def _load_image_if_path(path: str | None) -> Image.Image | None:
    if not path:
        return None
    return Image.open(path).convert("RGB")


def _load_target_if_path(path: str | None) -> Image.Image | Path | None:
    """Return Image or Path for target (for logging)."""
    if not path:
        return None
    p = Path(path)
    if p.exists():
        return p
    return None


def run(
    config_path: str,
    val_json_path: str,
    output_dir: str | None = None,
    checkpoint: str | None = None,
    sync_mlflow: bool = True,
) -> None:
    """
    Load config, resolve checkpoint, run inference on val.json, save to output_dir/validation/,
    and log to MLflow run.
    """
    cfg = load_config(config_path)
    cfg._config_path = str(Path(config_path).resolve())

    # Resolve output_dir
    if output_dir:
        outdir = Path(output_dir).resolve()
    else:
        mlflow_cfg = getattr(cfg, "mlflow", None)
        if not mlflow_cfg:
            raise ValueError("Config must have mlflow section, or pass --output_dir")
        log_root = Path(getattr(cfg, "log_root", "logs")).resolve()
        outdir = get_output_dir(
            log_root,
            mlflow_cfg.experiment_name,
            mlflow_cfg.run_name,
            user=getattr(mlflow_cfg, "user", "default"),
        )
    outdir = Path(outdir)

    # Resolve checkpoint
    ckpt_path, step = resolve_checkpoint_path(str(outdir), checkpoint)
    logger.info("Using checkpoint %s (step %d)", ckpt_path, step)

    # Load val.json
    with open(val_json_path) as f:
        entries = json.load(f)
    if not entries:
        logger.warning("val.json is empty, nothing to validate")
        return

    # Parse entries: {text, source?, target?}
    normalized: list[tuple[str, Image.Image | None, Image.Image | Path | None]] = []
    for e in entries:
        text = e.get("text") or e.get("prompt") or ""
        source = _load_image_if_path(e.get("source") or e.get("image"))
        target = _load_target_if_path(e.get("target"))
        normalized.append((text, source, target))

    # Model setup (mirror main.py)
    model_cfg = cfg.model
    lora_cfg = getattr(cfg, "lora", None)
    train_mode = getattr(cfg, "train_mode", "lora")
    pipeline_cfg = getattr(cfg, "pipeline", None) or getattr(model_cfg, "pipeline", None)

    if not pipeline_cfg:
        raise ValueError("pipeline config is required")
    pretrained_path = getattr(pipeline_cfg, "pretrained_model_name_or_path", None) or getattr(
        model_cfg, "pretrained_model_name_or_path", None
    )
    if not pretrained_path:
        raise ValueError("pipeline.pretrained_model_name_or_path is required")

    pipeline_cls = pipeline_cfg._class
    components = _PIPELINE_COMPONENTS.get(pipeline_cls)
    if not components:
        raise ValueError(
            f"No pipeline component registry for {pipeline_cls}. "
            "Add vae/scheduler or extend _PIPELINE_COMPONENTS."
        )

    weight_dtype = torch.float32
    mp = getattr(cfg, "mixed_precision", None)
    if mp == "fp16":
        weight_dtype = torch.float16
    elif mp == "bf16":
        weight_dtype = torch.bfloat16

    # Load transformer
    dit_cfg = getattr(model_cfg, "dit", model_cfg.transformer)
    trans_cls = dit_cfg._class
    subfolder = dit_cfg.subfolder
    transformer = trans_cls.from_pretrained(pretrained_path, subfolder=subfolder)
    transformer.requires_grad_(False)

    if train_mode == "lora" and lora_cfg:
        target_modules = [m.strip() for m in lora_cfg.target_modules]
        lora_config = LoraConfig(
            r=lora_cfg.rank,
            lora_alpha=lora_cfg.alpha,
            lora_dropout=lora_cfg.dropout,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        transformer.add_adapter(lora_config)

    # Load checkpoint via Accelerator
    from accelerate import Accelerator

    accelerator = Accelerator()
    transformer = accelerator.prepare(transformer)
    accelerator.load_state(str(ckpt_path))
    transformer = accelerator.unwrap_model(transformer)

    # Build pipeline and run inference
    pipeline = pipeline_cls.from_pretrained(
        pretrained_path,
        transformer=transformer,
        torch_dtype=weight_dtype,
    )
    pipeline = pipeline.to(dtype=weight_dtype)
    pipeline.enable_model_cpu_offload()
    pipeline.set_progress_bar_config(disable=True)

    val_cfg = getattr(cfg, "validation", None) or SimpleNamespace()
    inference_steps = getattr(val_cfg, "inference_steps", 4)
    guidance_scale = getattr(val_cfg, "guidance_scale", 1.0)
    resolution = getattr(val_cfg, "resolution", 512)
    seed = getattr(val_cfg, "seed", 42)
    generator = (
        torch.Generator(device=accelerator.device).manual_seed(seed) if seed is not None else None
    )

    images = []
    targets = []
    for prompt, source_img, target_ref in tqdm(normalized, desc="Validation"):
        with torch.autocast(device_type=accelerator.device.type, dtype=weight_dtype):
            out = pipeline(
                prompt=prompt,
                image=source_img,
                height=resolution,
                width=resolution,
                generator=generator,
                num_inference_steps=inference_steps,
                guidance_scale=guidance_scale,
            )
        images.append(out.images[0])
        targets.append(target_ref)

    del pipeline
    free_memory()

    # Save to output_dir/validation/
    val_dir = outdir / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images):
        path = val_dir / f"validation_step{step}_img{i}.png"
        img.save(path)
    logger.info("Saved %d images to %s", len(images), val_dir)

    # Log to MLflow
    if sync_mlflow:
        activate_mlflow_run_and_log_validation(
            output_dir=outdir,
            cfg=cfg,
            images=images,
            step=step,
            targets=targets if any(t is not None for t in targets) else None,
        )


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    parser = argparse.ArgumentParser(
        description="Run validation: load model from config, infer on val.json, log to MLflow."
    )
    parser.add_argument("--config", "-c", required=True, help="Path to YAML config.")
    parser.add_argument("--val_json", "-v", required=True, help="Path to val.json (list of {text, source?, target?}).")
    parser.add_argument("--output_dir", "-o", default=None, help="Override output dir (default: from config).")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint path or dir name (e.g. checkpoint-500). Default: latest in output_dir.",
    )
    parser.add_argument(
        "--no-mlflow",
        action="store_true",
        help="Do not sync validation results to MLflow.",
    )
    args = parser.parse_args()
    run(
        config_path=args.config,
        val_json_path=args.val_json,
        output_dir=args.output_dir,
        checkpoint=args.checkpoint,
        sync_mlflow=not args.no_mlflow,
    )


if __name__ == "__main__":
    main()
