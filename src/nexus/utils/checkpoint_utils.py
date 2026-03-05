"""
Checkpoint utilities: state dict load/save, pruning.
"""

import os
import logging
import shutil
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


def load_transformer_state(path: Path) -> dict:
    """Load state dict from safetensors or .pt."""
    if path.suffix == ".safetensors":
        try:
            import safetensors.torch
            return dict(safetensors.torch.load_file(str(path)))
        except ImportError:
            return dict(torch.load(path.with_suffix(".pt"), map_location="cpu", weights_only=True))
    return dict(torch.load(path, map_location="cpu", weights_only=True))


def save_transformer_state(state: dict, path: Path) -> None:
    """Save state dict to safetensors or .pt."""
    if path.suffix == ".safetensors":
        try:
            import safetensors.torch
            safetensors.torch.save_file(state, str(path))
        except ImportError:
            torch.save(state, path.with_suffix(".pt"))
    else:
        torch.save(state, path)


def check_existing_checkpoints(cfg) -> None:
    """
    When output_dir exists: if checkpoints exist, either set resume (auto_resume)
    or raise; if no checkpoints, warn and continue.
    """
    output_dir = getattr(cfg, "output_dir", None)
    if not output_dir or not os.path.exists(output_dir):
        return
    ckpt_dirs = [d for d in os.listdir(output_dir) if d.startswith("checkpoint")]
    if ckpt_dirs:
        auto_resume = getattr(cfg, "auto_resume", False)
        if auto_resume:
            latest = sorted(ckpt_dirs, key=lambda x: int(x.split("-")[1]))[-1]
            cfg.resume_from_checkpoint = latest
        else:
            raise ValueError(
                f"Output directory {output_dir} already contains checkpoints. "
                "Use --auto_resume to resume, or --output_dir <different> for a new run."
            )
    elif os.environ.get("RANK", "0") == "0":
        logger.warning(
            "Output directory %s already exists but has no checkpoints. Starting fresh.",
            output_dir,
        )


def prune_old_checkpoints(output_dir: str, limit: int) -> None:
    """Remove oldest checkpoints if count exceeds limit."""
    dirs = sorted(
        [d for d in os.listdir(output_dir) if d.startswith("checkpoint")],
        key=lambda x: int(x.split("-")[1]),
    )
    if len(dirs) > limit:
        for d in dirs[: len(dirs) - limit]:
            shutil.rmtree(os.path.join(output_dir, d))
