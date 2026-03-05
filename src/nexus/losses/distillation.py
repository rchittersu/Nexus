"""
Distillation loss: flow loss + distillation loss. Teacher created inside.
Teacher config is separate from model (student) config.
"""

import torch
from diffusers import Flux2KleinPipeline

from .context import LossContext
from .flow_matching import FlowMatchingLoss


class DistillationLoss:
    """
    Combines flow-matching loss and distillation loss. Teacher created in __init__.

    Config (via YAML):
      loss.kwargs: base, huber_delta, flow_weight, distillation_weight
      loss.teacher: pretrained_model_name_or_path, class_name, subfolder, revision, variant

    When loss.teacher is set, build_loss_fn passes teacher_cfg, accelerator, weight_dtype.
    """

    def __init__(
        self,
        base: str = "mse",
        huber_delta: float = 1.0,
        flow_weight: float = 0.5,
        distillation_weight: float = 0.5,
        teacher_cfg=None,
        accelerator=None,
        weight_dtype=None,
    ):
        self.flow_loss = FlowMatchingLoss(base=base, huber_delta=huber_delta)
        self.flow_weight = flow_weight
        self.distillation_weight = distillation_weight

        self._teacher: torch.nn.Module | None = None
        if teacher_cfg and getattr(teacher_cfg, "pretrained_model_name_or_path", None):
            if accelerator is None or weight_dtype is None:
                raise ValueError("DistillationLoss with loss.teacher requires teacher_cfg, accelerator, and weight_dtype (all passed by build_loss_fn).")
            cls = getattr(teacher_cfg, "_class", None)
            if cls is None:
                raise ValueError("loss.teacher.class_name must be resolved in config.")
            self._teacher = cls.from_pretrained(
                teacher_cfg.pretrained_model_name_or_path,
                subfolder=getattr(teacher_cfg, "subfolder", "transformer"),
                revision=getattr(teacher_cfg, "revision", None),
                variant=getattr(teacher_cfg, "variant", None),
                torch_dtype=weight_dtype,
            )
            self._teacher.requires_grad_(False)
            self._teacher.eval()
            self._teacher.to(device=accelerator.device, dtype=weight_dtype)

    def _compute_teacher_pred(self, ctx: LossContext) -> torch.Tensor | None:
        teacher = self._teacher
        if teacher is None:
            return None
        with torch.no_grad():
            out = teacher(
                hidden_states=ctx.packed_noisy,
                timestep=ctx.timesteps / 1000,
                guidance=ctx.guidance,
                encoder_hidden_states=ctx.text_embeds,
                txt_ids=ctx.text_ids,
                img_ids=ctx.model_input_ids,
                return_dict=False,
            )[0]
        out = out[:, : ctx.packed_noisy.size(1) :]
        return Flux2KleinPipeline._unpack_latents_with_ids(out, ctx.model_input_ids)

    def _compute_flow_loss(self, ctx: LossContext) -> torch.Tensor:
        target = ctx.noise - ctx.model_input
        return self.flow_loss._base_loss(
            ctx.model_output, target, ctx.weighting
        )

    def _compute_distillation_loss(
        self, pred: torch.Tensor, teacher_pred: torch.Tensor, w: torch.Tensor
    ) -> torch.Tensor:
        return self.flow_loss._base_loss(pred, teacher_pred, w)

    def __call__(self, ctx: LossContext) -> tuple[torch.Tensor, dict[str, float]]:
        flow_loss = self._compute_flow_loss(ctx)
        teacher_pred = self._compute_teacher_pred(ctx)

        if teacher_pred is not None:
            distill_loss = self._compute_distillation_loss(
                ctx.model_output, teacher_pred, ctx.weighting
            )
            total = self.flow_weight * flow_loss + self.distillation_weight * distill_loss
            logs = {
                "loss": total.detach().item(),
                "loss_flow": flow_loss.detach().item(),
                "loss_distillation": distill_loss.detach().item(),
            }
        else:
            total = flow_loss
            logs = {"loss": total.detach().item(), "loss_flow": flow_loss.detach().item()}

        return total, logs
