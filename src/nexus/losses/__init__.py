"""
Losses: FlowMatchingLoss, FlowMatchingWithPriorPreservation, DistillationLoss.

Each receives LossContext and returns (scalar, log_dict).

Configure in YAML via loss.class_name and loss.kwargs.
"""

from types import SimpleNamespace
from typing import Any, Callable

import torch

from nexus.train.config import ns_to_kwargs

from .context import LossContext
from .distillation import DistillationLoss
from .flow_matching import BASE_LOSSES, FlowMatchingLoss
from .prior_preservation import FlowMatchingWithPriorPreservation

LossCallable = Callable[[LossContext], tuple[torch.Tensor, dict]]


def build_loss_fn(
    cfg: SimpleNamespace,
    *,
    accelerator: Any = None,
    weight_dtype: torch.dtype | None = None,
) -> LossCallable:
    """Build loss from config: instantiate cfg.loss._class(**kwargs). Returns a callable (LossContext) -> (scalar, log_dict)."""
    loss_cls = getattr(cfg.loss, "_class", None)
    if loss_cls is None:
        raise ValueError("Config must define loss.class_name (e.g. nexus.losses:FlowMatchingLoss)")
    loss_kwargs = ns_to_kwargs(getattr(cfg.loss, "kwargs", None))
    teacher_cfg = getattr(cfg.loss, "teacher", None)
    if teacher_cfg is not None:
        if accelerator is None or weight_dtype is None:
            raise ValueError("loss.teacher requires accelerator and weight_dtype to be passed to build_loss_fn.")
        loss_kwargs["teacher_cfg"] = teacher_cfg
        loss_kwargs["accelerator"] = accelerator
        loss_kwargs["weight_dtype"] = weight_dtype
    if accelerator is not None:
        loss_kwargs["accelerator"] = accelerator
    if weight_dtype is not None:
        loss_kwargs["weight_dtype"] = weight_dtype
    return loss_cls(**loss_kwargs)


__all__ = [
    "LossContext",
    "FlowMatchingLoss",
    "FlowMatchingWithPriorPreservation",
    "DistillationLoss",
    "BASE_LOSSES",
    "build_loss_fn",
]
