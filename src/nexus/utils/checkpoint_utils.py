"""
Checkpoint utilities: state dict load/save, pruning, Klein save/load hooks.
"""

import os
import logging
import shutil
from pathlib import Path
from typing import Any, Callable

import torch
from diffusers.training_utils import _collate_lora_metadata, cast_training_params, _to_cpu_contiguous
from diffusers.utils import convert_unet_state_dict_to_peft
from peft import get_peft_model_state_dict, set_peft_model_state_dict

logger = logging.getLogger(__name__)
TRANSFORMER_SAFE = "transformer.safetensors"
TRANSFORMER_PT = "transformer.pt"


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


def make_dit_save_hook(
    transformer_cls: type,
    pipeline_cls: type | None,
    train_mode: str,
    accelerator: Any,
    unwrap_fn: Callable,
    is_fsdp: bool,
) -> Callable:
    """Return save hook for DiT (LoRA or full)."""

    if is_fsdp:
        raise ValueError("FSDP is not supported for save hook")

    # Adapted from Flux Klein Dreambooth Teaining Example
    # create custom saving & loading hooks so that `accelerator.save_state(...)` serializes in a nice format
    def save_model_hook(models, weights, output_dir):

        # 1) Validate and pick the transformer model
        modules_to_save: dict[str, Any] = {}
        transformer_model = None

        for model in models:
            if isinstance(unwrap_fn(model), transformer_cls):
                transformer_model = model
                modules_to_save["transformer"] = model
            else:
                raise ValueError(f"unexpected save model: {model.__class__}")

        if transformer_model is None:
            raise ValueError("No transformer model found in 'models'")

        # 2) Optionally gather FSDP state dict once
        # TODO: Handle FSDP case later
        # state_dict = accelerator.get_state_dict(model) if is_fsdp else None

        # 3) Only main process materializes the LoRA state dict
        if accelerator.is_main_process:
            peft_kwargs = {}
            # TODO: Handle FSDP case later
            # if is_fsdp:
            #     peft_kwargs["state_dict"] = state_dict

            if train_mode == "lora":
                layers_to_save = get_peft_model_state_dict(
                    unwrap_fn(transformer_model) if is_fsdp else transformer_model,
                    **peft_kwargs,
                )

                # TODO: Don't know fsdp related caveats
                # if is_fsdp:
                #     layers_to_save = _to_cpu_contiguous(layers_to_save)

                pipeline_cls.save_lora_weights(
                    output_dir,
                    transformer_lora_layers=layers_to_save,
                    **_collate_lora_metadata(modules_to_save),
                )

            else:
                transformer_to_save = unwrap_fn(transformer_model) if is_fsdp else transformer_model
                # TODO: Don't know fsdp related caveats
                transformer_to_save.save_pretrained(output_dir)

            if weights:
                weights.pop()

    return save_model_hook


def make_dit_load_hook(
    transformer_cls: type,
    pipeline_cls: type | None,
    train_mode: str,
    accelerator: Any,
    unwrap_fn: Callable,
    is_fsdp: bool,
) -> Callable:
    """Return load hook for  DiT (LoRA or full)."""

    if is_fsdp:
        raise ValueError("FSDP is not supported for load hook")

    def load_model_hook(models, input_dir):

        assert len(models) == 1, "Only one transformer model is supported"
        transformer_ = unwrap_fn(models[0])
        assert isinstance(transformer_, transformer_cls), "Transformer model is not of type transformer_cls"

        # TODO: Handle FSDP case later

        if train_mode == "lora":
            lora_state_dict = pipeline_cls.lora_state_dict(input_dir)

            transformer_state_dict = {
                f"{k.replace('transformer.', '')}": v for k, v in lora_state_dict.items() if k.startswith("transformer.")
            }

            transformer_state_dict = convert_unet_state_dict_to_peft(transformer_state_dict)
            incompatible_keys = set_peft_model_state_dict(transformer_, transformer_state_dict, adapter_name="default")
            if incompatible_keys is not None:
                # check only for unexpected keys
                unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
                if unexpected_keys:
                    logger.warning(
                        f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                        f" {unexpected_keys}. "
                    )
        else:
            input_path = Path(input_dir)
            for name in (TRANSFORMER_SAFE, TRANSFORMER_PT):
                p = input_path / name
                if p.exists():
                    transformer_.load_state_dict(load_transformer_state(p), strict=False)
                    break
            else:
                raise FileNotFoundError(
                    f"No transformer checkpoint found in {input_dir}. "
                    f"Expected {TRANSFORMER_SAFE} or {TRANSFORMER_PT}."
                )

    # We aren't using mixed precision; so we don't need to upcast trainable parameters
    return load_model_hook
