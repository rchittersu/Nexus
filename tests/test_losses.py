"""Tests for nexus.train.losses"""

import pytest
import torch

from nexus.train.losses import (
    DistillationLoss,
    HuberLoss,
    L1Loss,
    LogCoshLoss,
    MetaLoss,
    MSELoss,
    _loss_uses_distillation,
)


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


class TestDistillationLoss:
    def test_returns_scalar_with_teacher_pred(self, sample_tensors):
        pred, target, weighting = sample_tensors
        teacher_pred = torch.randn_like(pred)
        loss = DistillationLoss()
        out = loss(pred, target, weighting, teacher_pred=teacher_pred)
        assert out.dim() == 0
        assert out.item() >= 0

    def test_zero_loss_when_pred_matches_teacher(self, sample_tensors):
        pred, target, weighting = sample_tensors
        loss = DistillationLoss()
        out = loss(pred, pred, weighting, teacher_pred=pred)
        assert out.item() == pytest.approx(0.0)

    def test_returns_zero_when_teacher_pred_none(self, sample_tensors):
        pred, target, weighting = sample_tensors
        loss = DistillationLoss()
        out = loss(pred, target, weighting, teacher_pred=None)
        assert out.item() == pytest.approx(0.0)


class TestMetaLoss:
    def test_combines_losses_with_scales(self, sample_tensors):
        pred, target, weighting = sample_tensors
        mse = MSELoss()
        meta = MetaLoss([(mse, 0.5, "a"), (MSELoss(), 0.5, "b")])
        out_meta, breakdown = meta(pred, target, weighting)
        out_mse = mse(pred, target, weighting)
        assert out_meta.item() == pytest.approx(out_mse.item())
        assert "a" in breakdown and "b" in breakdown

    def test_scales_affect_result(self, sample_tensors):
        pred, target, weighting = sample_tensors
        meta = MetaLoss([(MSELoss(), 1.0), (MSELoss(), 2.0)])
        out, breakdown = meta(pred, target, weighting)
        single = MSELoss()(pred, target, weighting)
        assert out.item() == pytest.approx(3.0 * single.item())
        assert len(breakdown) == 2

    def test_passes_teacher_pred_to_sub_losses(self, sample_tensors):
        pred, target, weighting = sample_tensors
        teacher_pred = pred + 0.1
        meta = MetaLoss([(MSELoss(), 0.5), (DistillationLoss(), 0.5)])
        out, breakdown = meta(pred, target, weighting, teacher_pred=teacher_pred)
        assert out.dim() == 0
        assert out.item() >= 0
        assert "MSELoss" in breakdown and "DistillationLoss" in breakdown


class TestLossUsesDistillation:
    def test_distillation_loss_returns_true(self):
        assert _loss_uses_distillation(DistillationLoss()) is True

    def test_metaloss_with_distillation_returns_true(self):
        meta = MetaLoss([(MSELoss(), 1.0), (DistillationLoss(), 0.5)])
        assert _loss_uses_distillation(meta) is True

    def test_metaloss_without_distillation_returns_false(self):
        meta = MetaLoss([(MSELoss(), 1.0, "mse")])
        assert _loss_uses_distillation(meta) is False

    def test_mse_returns_false(self):
        assert _loss_uses_distillation(MSELoss()) is False


class TestLossConsistency:
    """All losses should be lower for smaller errors."""

    def test_smaller_error_lower_loss(self, sample_tensors):
        pred, target, weighting = sample_tensors
        losses = [MSELoss(), L1Loss(), HuberLoss(delta=1.0), LogCoshLoss()]
        for loss_fn in losses:
            large_err = loss_fn(pred, target + 10.0, weighting)
            small_err = loss_fn(pred, target + 0.1, weighting)
            assert small_err.item() < large_err.item()
