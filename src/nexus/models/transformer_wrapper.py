"""
Generic transformer wrapper: LoRA or full fine-tune for any DiT-compatible arch.

Works with Flux2, SD3, or any pipeline that supports save_lora_weights / lora_state_dict.
"""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import torch
from diffusers.training_utils import _collate_lora_metadata, cast_training_params
from diffusers.utils import convert_unet_state_dict_to_peft
from diffusers.utils.torch_utils import is_compiled_module
from peft import LoraConfig, get_peft_model_state_dict

logger = logging.getLogger(__name__)

TrainMode = Literal["lora", "full"]

TRANSFORMER_SAFE = "transformer.safetensors"
TRANSFORMER_PT = "transformer.pt"


def _unwrap(accelerator: Any, m: torch.nn.Module) -> torch.nn.Module:
    m = accelerator.unwrap_model(m)
    return m._orig_mod if is_compiled_module(m) else m


def _load_state_dict(path: Path) -> dict:
    """Load state dict from safetensors or .pt."""
    if path.suffix == ".safetensors":
        try:
            import safetensors.torch
            return dict(safetensors.torch.load_file(str(path)))
        except ImportError:
            return dict(torch.load(path, map_location="cpu", weights_only=True))
    return dict(torch.load(path, map_location="cpu", weights_only=True))


def _save_state_dict(state: dict, path: Path) -> None:
    """Save state dict to safetensors or .pt."""
    if path.suffix == ".safetensors":
        try:
            import safetensors.torch
            safetensors.torch.save_file(state, str(path))
        except ImportError:
            torch.save(state, path.with_suffix(".pt"))
    else:
        torch.save(state, path)


class TransformerWrapper:
    """
    Save/load and trainable-params helper for DiT transformers.
    LoRA or full fine-tune with unified save/load hooks for Accelerator checkpointing.
    """

    def __init__(
        self,
        transformer: torch.nn.Module,
        pretrained_path: str,
        transformer_cls: type,
        subfolder: str = "transformer",
        mode: TrainMode = "lora",
        lora_config: LoraConfig | None = None,
        pipeline_cls: type | None = None,
        component_name: str = "transformer",
    ):
        self.transformer = transformer
        self.pretrained_path = pretrained_path
        self.transformer_cls = transformer_cls
        self.subfolder = subfolder
        self.mode = mode
        self.lora_config = lora_config
        self.pipeline_cls = pipeline_cls
        self.component_name = component_name
        self._lora_kwarg = f"{component_name}_lora_layers"

        if mode == "lora" and (lora_config is None or pipeline_cls is None):
            raise ValueError("lora_config and pipeline_cls required when mode='lora'")

    def get_trainable_parameters(self) -> list:
        return [p for p in self.transformer.parameters() if p.requires_grad]

    def get_save_hook(self, accelerator: Any, is_fsdp: bool = False) -> Callable:
        def save_hook(models, weights, output_dir):
            trans_model = None
            for m in models:
                if isinstance(_unwrap(accelerator, m), self.transformer_cls):
                    trans_model = m
                    break
            if trans_model is None:
                raise ValueError(f"No transformer (type {self.transformer_cls}) in save models")

            if accelerator.is_main_process:
                unwrapped = _unwrap(accelerator, trans_model)
                if weights:
                    weights.pop()

                if self.mode == "lora":
                    lora_sd = get_peft_model_state_dict(
                        unwrapped,
                        state_dict=accelerator.get_state_dict(trans_model) if is_fsdp else None,
                    )
                    if is_fsdp:
                        from diffusers.training_utils import _to_cpu_contiguous
                        lora_sd = _to_cpu_contiguous(lora_sd)
                    self.pipeline_cls.save_lora_weights(
                        output_dir,
                        **{self._lora_kwarg: lora_sd},
                        **_collate_lora_metadata({self.component_name: trans_model}),
                    )
                else:
                    state = (
                        accelerator.get_state_dict(trans_model)
                        if is_fsdp
                        else unwrapped.state_dict()
                    )
                    _save_state_dict(state, Path(output_dir) / TRANSFORMER_SAFE)

        return save_hook

    def get_load_hook(
        self,
        accelerator: Any,
        is_fsdp: bool = False,
        mixed_precision: str | None = None,
    ) -> Callable:
        def load_hook(models, input_dir):
            input_path = Path(input_dir)
            if is_fsdp:
                trans = self.transformer_cls.from_pretrained(
                    self.pretrained_path,
                    subfolder=self.subfolder,
                )
                if self.mode == "lora":
                    trans.add_adapter(self.lora_config)
            else:
                trans = None
                while models:
                    m = models.pop()
                    if isinstance(_unwrap(accelerator, m), self.transformer_cls):
                        trans = _unwrap(accelerator, m)
                        break
                if trans is None:
                    raise ValueError("No transformer in load hook")

            if self.mode == "lora":
                from peft import set_peft_model_state_dict
                lora_sd = self.pipeline_cls.lora_state_dict(input_dir)
                prefix = f"{self.component_name}."
                trans_sd = {
                    k[len(prefix):]: v
                    for k, v in lora_sd.items()
                    if k.startswith(prefix)
                }
                trans_sd = convert_unet_state_dict_to_peft(trans_sd)
                set_peft_model_state_dict(trans, trans_sd, adapter_name="default")
            else:
                for name in (TRANSFORMER_SAFE, TRANSFORMER_PT):
                    path = input_path / name
                    if path.exists():
                        trans.load_state_dict(_load_state_dict(path), strict=False)
                        break

            if mixed_precision == "fp16":
                cast_training_params([trans])

        return load_hook

    def save_final(
        self,
        output_dir: str | Path,
        model: torch.nn.Module,
        unwrap_fn: Callable[[torch.nn.Module], torch.nn.Module],
    ) -> None:
        trans = unwrap_fn(model)
        output_dir = Path(output_dir)
        if self.mode == "lora":
            lora_sd = get_peft_model_state_dict(trans)
            self.pipeline_cls.save_lora_weights(
                str(output_dir),
                **{self._lora_kwarg: lora_sd},
                **_collate_lora_metadata({self.component_name: trans}),
            )
            logger.info("Saved LoRA weights to %s", output_dir)
        else:
            path = output_dir / TRANSFORMER_SAFE
            _save_state_dict(trans.state_dict(), path)
            logger.info("Saved full transformer to %s", path)
