"""
Base architecture interfaces.
Extend these in root-level implementations for specific DiT variants.
"""
from __future__ import annotations


class DiTBlockBase:
    """Placeholder base for DiT-style transformer blocks. Override in root."""

    def forward(self, x, context=None, timestep=None, **kwargs):
        raise NotImplementedError("Implement in extended module")


class EmbedderBase:
    """Base for time/condition embedders. Override in root."""

    def forward(self, t, condition=None, **kwargs):
        raise NotImplementedError("Implement in extended module")
