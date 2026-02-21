"""Tests for nexus.data.precomputed_sstk_dataset"""

import numpy as np
import pytest
import torch

from nexus.data.precomputed_sstk_dataset import collate_precomputed


class TestCollatePrecomputed:
    def test_collate_stacks_tensors(self):
        batch = [
            {
                "latents": torch.randn(16, 64, 64),
                "text_embeds": torch.randn(512, 3584),
                "text_ids": torch.zeros(512, 4, dtype=torch.int64),
                "caption": "cap1",
            },
            {
                "latents": torch.randn(16, 64, 64),
                "text_embeds": torch.randn(512, 3584),
                "text_ids": torch.zeros(512, 4, dtype=torch.int64),
                "caption": "cap2",
            },
        ]
        out = collate_precomputed(batch)
        assert out["latents"].shape == (2, 16, 64, 64)
        assert out["text_embeds"].shape == (2, 512, 3584)
        assert out["text_ids"].shape == (2, 512, 4)
        assert out["captions"] == ["cap1", "cap2"]

    def test_collate_single_sample(self):
        batch = [
            {
                "latents": torch.randn(16, 64, 64),
                "text_embeds": torch.randn(512, 3584),
                "text_ids": torch.zeros(512, 4, dtype=torch.int64),
                "caption": "",
            },
        ]
        out = collate_precomputed(batch)
        assert out["latents"].shape == (1, 16, 64, 64)
        assert len(out["captions"]) == 1
