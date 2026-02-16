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
    ) -> torch.Tensor:
        """
        Compute loss. pred, target, weighting have same shape.
        Returns scalar (mean over batch).
        """
        pass


class MSELoss(FlowMatchingLossBase):
    """Weighted mean squared error. Standard flow-matching loss."""

    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weighting: torch.Tensor,
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
    ) -> torch.Tensor:
        err = pred.float() - target.float()
        abs_err = err.abs()
        quad = torch.where(abs_err <= self.delta, 0.5 * err**2, self.delta * (abs_err - 0.5 * self.delta))
        weighted = weighting.float() * quad
        return weighted.reshape(target.shape[0], -1).mean(1).mean()


class LogCoshLoss(FlowMatchingLossBase):
    """Weighted log-cosh. Smooth, twice differentiable approximation to L1."""

    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weighting: torch.Tensor,
    ) -> torch.Tensor:
        err = pred.float() - target.float()
        logcosh = torch.log(torch.cosh(err.clamp(min=-50, max=50)))
        weighted = weighting.float() * logcosh
        return weighted.reshape(target.shape[0], -1).mean(1).mean()


__all__ = ["FlowMatchingLossBase", "MSELoss", "L1Loss", "HuberLoss", "LogCoshLoss"]
