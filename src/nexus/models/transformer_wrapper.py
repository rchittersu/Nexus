"""
Generic transformer wrapper: LoRA or full fine-tune for any DiT-compatible arch.

Works with Flux2, SD3, or any pipeline that supports save_lora_weights / lora_state_dict.
Component name (e.g. "transformer", "unet") is configurable.
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


class TransformerWrapper(torch.nn.Module):
    """
    Wrapper for any DiT-compatible transformer (Flux2, SD3, etc.).
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
        """
        Args:
            transformer: Loaded transformer (e.g. Flux2Transformer2DModel, SD3Transformer2DModel).
            pretrained_path: Path for resuming/loading base weights.
            transformer_cls: Class to load transformer from (for FSDP resume).
            subfolder: Model subfolder in pretrained_path.
            mode: "lora" (train LoRA only) or "full" (train all params).
            lora_config: Required when mode="lora".
            pipeline_cls: Required when mode="lora". Pipeline with save_lora_weights,
                lora_state_dict (e.g. Flux2KleinPipeline, StableDiffusion3Pipeline).
            component_name: LoRA component name: "transformer" (Flux2), "unet" (SD), etc.
                Used for save/load kwargs and state dict key filtering.
        """
        super().__init__()
        self.transformer = transformer
        self.pretrained_path = pretrained_path
        self.transformer_cls = transformer_cls
        self.subfolder = subfolder
        self.mode = mode
        self.lora_config = lora_config
        self.pipeline_cls = pipeline_cls
        self.component_name = component_name

        if mode == "lora" and (lora_config is None or pipeline_cls is None):
            raise ValueError("lora_config and pipeline_cls required when mode='lora'")

    def forward(self, *args, **kwargs):
        """Delegate to the wrapped transformer."""
        return self.transformer(*args, **kwargs)

    def get_trainable_parameters(self) -> list:
        """Return parameters that require gradients."""
        return [p for p in self.parameters() if p.requires_grad]

    def _lora_kwarg(self) -> str:
        return f"{self.component_name}_lora_layers"

    def get_save_hook(
        self,
        accelerator: Any,
        transformer_cls: type,
        is_fsdp: bool = False,
    ) -> Callable:
        """Return save hook for accelerator.register_save_state_pre_hook."""

        def _unwrap(m):
            m = accelerator.unwrap_model(m)
            return m._orig_mod if is_compiled_module(m) else m

        pipeline_cls = self.pipeline_cls
        mode = self.mode
        lora_kwarg = self._lora_kwarg()
        comp_name = self.component_name

        def save_hook(models, weights, output_dir):
            trans_model = None
            for m in models:
                if isinstance(_unwrap(m), transformer_cls):
                    trans_model = m
                    break
            if trans_model is None:
                raise ValueError(f"No transformer (type {transformer_cls}) in save models")

            if accelerator.is_main_process:
                unwrapped = _unwrap(trans_model)
                if weights:
                    weights.pop()

                if mode == "lora":
                    lora_sd = get_peft_model_state_dict(
                        unwrapped,
                        state_dict=accelerator.get_state_dict(trans_model) if is_fsdp else None,
                    )
                    if is_fsdp:
                        from diffusers.training_utils import _to_cpu_contiguous

                        lora_sd = _to_cpu_contiguous(lora_sd)
                    pipeline_cls.save_lora_weights(
                        output_dir,
                        **{lora_kwarg: lora_sd},
                        **_collate_lora_metadata({comp_name: trans_model}),
                    )
                else:
                    state_dict = (
                        accelerator.get_state_dict(trans_model)
                        if is_fsdp
                        else unwrapped.state_dict()
                    )
                    out_path = Path(output_dir) / "transformer.safetensors"
                    try:
                        import safetensors.torch

                        safetensors.torch.save_file(state_dict, str(out_path))
                    except ImportError:
                        torch.save(state_dict, Path(output_dir) / "transformer.pt")

        return save_hook

    def get_load_hook(
        self,
        accelerator: Any,
        transformer_cls: type,
        is_fsdp: bool = False,
        mixed_precision: str | None = None,
    ) -> Callable:
        """Return load hook for accelerator.register_load_state_pre_hook."""

        def _unwrap(m):
            m = accelerator.unwrap_model(m)
            return m._orig_mod if is_compiled_module(m) else m

        pretrained_path = self.pretrained_path
        subfolder = self.subfolder
        lora_config = self.lora_config
        pipeline_cls = self.pipeline_cls
        mode = self.mode
        comp_name = self.component_name
        transformer_load_cls = self.transformer_cls

        def load_hook(models, input_dir):
            trans = None
            if is_fsdp:
                trans = transformer_load_cls.from_pretrained(
                    pretrained_path,
                    subfolder=subfolder,
                )
                if mode == "lora":
                    trans.add_adapter(lora_config)
            else:
                while models:
                    m = models.pop()
                    if isinstance(_unwrap(m), transformer_cls):
                        trans = _unwrap(m)
                        break

            if trans is None:
                raise ValueError("No transformer in load hook")

            if mode == "lora":
                from peft import set_peft_model_state_dict

                lora_sd = pipeline_cls.lora_state_dict(input_dir)
                prefix = f"{comp_name}."
                trans_sd = {k[len(prefix) :]: v for k, v in lora_sd.items() if k.startswith(prefix)}
                trans_sd = convert_unet_state_dict_to_peft(trans_sd)
                set_peft_model_state_dict(trans, trans_sd, adapter_name="default")
            else:
                path = Path(input_dir) / "transformer.safetensors"
                if path.exists():
                    try:
                        import safetensors.torch

                        state = safetensors.torch.load_file(str(path))
                    except ImportError:
                        state = torch.load(str(path), map_location="cpu", weights_only=True)
                    trans.load_state_dict(state, strict=False)
                else:
                    full_path = Path(input_dir) / "transformer.pt"
                    if full_path.exists():
                        state = torch.load(full_path, map_location="cpu", weights_only=True)
                        trans.load_state_dict(state, strict=False)

            if mixed_precision == "fp16":
                cast_training_params([trans])

        return load_hook

    def save_final(
        self,
        output_dir: str | Path,
        model: torch.nn.Module,
        unwrap_fn: Callable[[torch.nn.Module], torch.nn.Module],
    ) -> None:
        """Save final weights (LoRA or full) at end of training."""
        trans = unwrap_fn(model)
        output_dir = Path(output_dir)
        comp_name = self.component_name
        lora_kwarg = self._lora_kwarg()

        if self.mode == "lora":
            lora_sd = get_peft_model_state_dict(trans)
            self.pipeline_cls.save_lora_weights(
                str(output_dir),
                **{lora_kwarg: lora_sd},
                **_collate_lora_metadata({comp_name: trans}),
            )
            logger.info(f"Saved LoRA weights to {output_dir}")
        else:
            state = trans.state_dict()
            try:
                import safetensors.torch

                safetensors.torch.save_file(state, str(output_dir / "transformer.safetensors"))
            except ImportError:
                torch.save(state, str(output_dir / "transformer.pt"))
            logger.info(f"Saved full transformer to {output_dir / 'transformer.safetensors'}")
