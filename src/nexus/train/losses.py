"""
Configurable flow-matching losses: MSE, L1, Huber, etc.

All losses accept (pred, target, weighting) and return a scalar.
Weighting is applied element-wise before reduction.
"""

from abc import ABC, abstractmethod

import torch


class FlowMatchingLossBase(ABC):
    """Base class for flow-matching losses: pred vs target with optional sigma weighting."""

    @abstractmethod
    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weighting: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute loss. pred, target, weighting have same shape.
        Returns scalar (mean over batch).
        Extra kwargs (e.g. teacher_pred for distillation) are passed for extensibility.
        """
        pass


class MSELoss(FlowMatchingLossBase):
    """Weighted mean squared error. Standard flow-matching loss."""

    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weighting: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        err = (pred.float() - target.float()) ** 2
        weighted = weighting.float() * err
        # Per-sample mean over spatial dims, then batch mean
        return weighted.reshape(target.shape[0], -1).mean(1).mean()


class L1Loss(FlowMatchingLossBase):
    """Weighted L1 (MAE). More robust to outliers than MSE."""

    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weighting: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        err = (pred.float() - target.float()).abs()
        weighted = weighting.float() * err
        return weighted.reshape(target.shape[0], -1).mean(1).mean()


class HuberLoss(FlowMatchingLossBase):
    """Weighted Huber (smooth L1). Smooth near zero, linear in tails."""

    def __init__(self, delta: float = 1.0):
        self.delta = delta

    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weighting: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        err = pred.float() - target.float()
        abs_err = err.abs()
        quad = torch.where(
            abs_err <= self.delta, 0.5 * err**2, self.delta * (abs_err - 0.5 * self.delta)
        )
        weighted = weighting.float() * quad
        return weighted.reshape(target.shape[0], -1).mean(1).mean()


class LogCoshLoss(FlowMatchingLossBase):
    """Weighted log-cosh. Smooth, twice differentiable approximation to L1."""

    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weighting: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        err = pred.float() - target.float()
        logcosh = torch.log(torch.cosh(err.clamp(min=-50, max=50)))
        weighted = weighting.float() * logcosh
        return weighted.reshape(target.shape[0], -1).mean(1).mean()


class DistillationLoss(FlowMatchingLossBase):
    """
    Pure distillation loss: student predictions match teacher predictions.
    L = weighted MSE(pred, teacher_pred). Requires teacher_pred in kwargs.
    """

    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weighting: torch.Tensor,
        teacher_pred: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if teacher_pred is None:
            return pred.new_tensor(0.0)
        err = (pred.float() - teacher_pred.float()) ** 2
        weighted = weighting.float() * err
        return weighted.reshape(target.shape[0], -1).mean(1).mean()


class MetaLoss(FlowMatchingLossBase):
    """
    Combines a sequence of losses with configurable scales.
    L = sum(scale_i * loss_i(pred, target, weighting, **kwargs))
    Returns (total, breakdown_dict) for logging each loss component.
    """

    def __init__(
        self,
        losses: list[tuple["FlowMatchingLossBase", float]]
        | list[tuple["FlowMatchingLossBase", float, str]],
    ):
        """
        Args:
            losses: List of (loss_instance, scale) or (loss_instance, scale, name).
                    name is used for logging; defaults to type(loss_instance).__name__
        """
        self.losses: list[tuple[FlowMatchingLossBase, float, str]] = []
        for item in losses:
            if len(item) == 3:
                self.losses.append(item)
            else:
                loss_fn, scale = item
                self.losses.append((loss_fn, scale, type(loss_fn).__name__))

    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weighting: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        total = pred.new_tensor(0.0)
        breakdown: dict[str, float] = {}
        for loss_fn, scale, name in self.losses:
            val = loss_fn(pred=pred, target=target, weighting=weighting, **kwargs)
            scaled = scale * val
            total = total + scaled
            breakdown[name] = scaled.detach().item()
        return total, breakdown


def _loss_uses_distillation(loss_fn: FlowMatchingLossBase) -> bool:
    """True if loss requires teacher_pred (DistillationLoss or MetaLoss containing it)."""
    if type(loss_fn).__name__ == "DistillationLoss":
        return True
    if type(loss_fn).__name__ == "MetaLoss":
        return any(_loss_uses_distillation(loss_inst) for loss_inst, _, _ in loss_fn.losses)
    return False


__all__ = [
    "FlowMatchingLossBase",
    "MSELoss",
    "L1Loss",
    "HuberLoss",
    "LogCoshLoss",
    "DistillationLoss",
    "MetaLoss",
    "_loss_uses_distillation",
]
