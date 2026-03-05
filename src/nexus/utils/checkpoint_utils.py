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


def get_latest_checkpoint(output_dir: str) -> str | None:
    """
    Return the checkpoint dir name with the highest step (e.g. 'checkpoint-500').
    Returns None if no checkpoint-* dirs exist.
    """
    if not output_dir or not os.path.exists(output_dir):
        return None
    ckpt_dirs = [d for d in os.listdir(output_dir) if d.startswith("checkpoint-")]
    if not ckpt_dirs:
        return None
    return sorted(ckpt_dirs, key=lambda x: int(x.split("-")[1]))[-1]


def resolve_checkpoint_path(output_dir: str, checkpoint: str | None) -> tuple[Path, int]:
    """
    Resolve checkpoint to full path and step.
    - If checkpoint is None: use latest in output_dir.
    - If checkpoint is 'checkpoint-N' or relative: resolve relative to output_dir.
    - If checkpoint is absolute path: use as-is.
    Returns (full_path, step).
    """
    if checkpoint is None:
        ckpt_name = get_latest_checkpoint(output_dir)
        if not ckpt_name:
            raise ValueError(
                f"No checkpoints found in {output_dir}. Use --checkpoint to specify a path."
            )
        ckpt_path = Path(output_dir) / ckpt_name
    else:
        ckpt_path = Path(checkpoint)
        if not ckpt_path.is_absolute():
            ckpt_path = Path(output_dir) / ckpt_path
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    # Infer step from dir name (checkpoint-N)
    name = ckpt_path.name
    if name.startswith("checkpoint-"):
        step = int(name.split("-")[1])
    else:
        step = 0
    return ckpt_path.resolve(), step


def prune_old_checkpoints(output_dir: str, limit: int) -> None:
    """Remove oldest checkpoints if count exceeds limit."""
    dirs = sorted(
        [d for d in os.listdir(output_dir) if d.startswith("checkpoint")],
        key=lambda x: int(x.split("-")[1]),
    )
    if len(dirs) > limit:
        for d in dirs[: len(dirs) - limit]:
            shutil.rmtree(os.path.join(output_dir, d))
