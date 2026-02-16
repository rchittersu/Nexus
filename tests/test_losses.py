"""Tests for nexus.train.losses"""

import pytest
import torch

from nexus.train.losses import HuberLoss, L1Loss, LogCoshLoss, MSELoss


@pytest.fixture
def sample_tensors():
    """Create pred, target, weighting of shape (2, 4, 4)."""
    b, c, h, w = 2, 4, 4, 4
    pred = torch.randn(b, c, h, w)
    target = torch.randn(b, c, h, w)
    weighting = torch.ones(b, c, h, w)
    return pred, target, weighting


class TestMSELoss:
    def test_returns_scalar(self, sample_tensors):
        pred, target, weighting = sample_tensors
        loss = MSELoss()
        out = loss(pred, target, weighting)
        assert out.dim() == 0
        assert out.item() >= 0

    def test_perfect_pred_zero_loss(self, sample_tensors):
        pred, target, weighting = sample_tensors
        loss = MSELoss()
        out = loss(pred, pred, weighting)
        assert out.item() == pytest.approx(0.0)

    def test_weighting_affects_result(self, sample_tensors):
        pred, target, weighting = sample_tensors
        loss = MSELoss()
        out1 = loss(pred, target, torch.ones_like(weighting))
        out2 = loss(pred, target, 2.0 * torch.ones_like(weighting))
        assert out2.item() == pytest.approx(2.0 * out1.item())


class TestL1Loss:
    def test_returns_scalar(self, sample_tensors):
        pred, target, weighting = sample_tensors
        loss = L1Loss()
        out = loss(pred, target, weighting)
        assert out.dim() == 0
        assert out.item() >= 0

    def test_perfect_pred_zero_loss(self, sample_tensors):
        pred, target, weighting = sample_tensors
        loss = L1Loss()
        out = loss(pred, pred, weighting)
        assert out.item() == pytest.approx(0.0)


class TestHuberLoss:
    def test_returns_scalar(self, sample_tensors):
        pred, target, weighting = sample_tensors
        loss = HuberLoss(delta=1.0)
        out = loss(pred, target, weighting)
        assert out.dim() == 0
        assert out.item() >= 0

    def test_custom_delta(self, sample_tensors):
        pred, target, weighting = sample_tensors
        loss = HuberLoss(delta=2.0)
        out = loss(pred, target, weighting)
        assert out.dim() == 0


class TestLogCoshLoss:
    def test_returns_scalar(self, sample_tensors):
        pred, target, weighting = sample_tensors
        loss = LogCoshLoss()
        out = loss(pred, target, weighting)
        assert out.dim() == 0
        assert out.item() >= 0

    def test_perfect_pred_zero_loss(self, sample_tensors):
        pred, target, weighting = sample_tensors
        loss = LogCoshLoss()
        out = loss(pred, pred, weighting)
        assert out.item() == pytest.approx(0.0)


class TestLossConsistency:
    """All losses should be lower for smaller errors."""

    def test_smaller_error_lower_loss(self, sample_tensors):
        pred, target, weighting = sample_tensors
        losses = [MSELoss(), L1Loss(), HuberLoss(delta=1.0), LogCoshLoss()]
        for loss_fn in losses:
            large_err = loss_fn(pred, target + 10.0, weighting)
            small_err = loss_fn(pred, target + 0.1, weighting)
            assert small_err.item() < large_err.item()
