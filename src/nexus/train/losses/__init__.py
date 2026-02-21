"""
Losses: FlowMatchingLoss, FlowMatchingWithPriorPreservation, DistillationLoss.

Each receives LossContext and returns (scalar, log_dict).

Configure in YAML via loss.class_name and loss.kwargs.
"""

from .context import LossContext
from .distillation import DistillationLoss
from .flow_matching import BASE_LOSSES, FlowMatchingLoss
from .prior_preservation import FlowMatchingWithPriorPreservation

__all__ = [
    "LossContext",
    "FlowMatchingLoss",
    "FlowMatchingWithPriorPreservation",
    "DistillationLoss",
    "BASE_LOSSES",
]
