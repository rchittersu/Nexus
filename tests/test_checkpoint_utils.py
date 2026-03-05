"""Tests for nexus.utils.checkpoint_utils"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from nexus.utils.checkpoint_utils import (
    check_existing_checkpoints,
    get_latest_checkpoint,
    load_transformer_state,
    prune_old_checkpoints,
    resolve_checkpoint_path,
    save_transformer_state,
)


class TestTransformerStateIO:
    def test_save_and_load_pt(self, tmp_path):
        state = {"layer.weight": torch.randn(2, 3)}
        path = tmp_path / "model.pt"
        save_transformer_state(state, path)
        loaded = load_transformer_state(path)
        assert loaded.keys() == state.keys()
        assert torch.allclose(loaded["layer.weight"], state["layer.weight"])

    def test_save_and_load_safetensors(self, tmp_path):
        pytest.importorskip("safetensors")
        state = {"layer.weight": torch.randn(2, 3)}
        path = tmp_path / "model.safetensors"
        save_transformer_state(state, path)
        loaded = load_transformer_state(path)
        assert loaded.keys() == state.keys()
        assert torch.allclose(loaded["layer.weight"], state["layer.weight"])


class TestCheckExistingCheckpoints:
    def test_returns_early_when_no_output_dir(self):
        cfg = SimpleNamespace()
        check_existing_checkpoints(cfg)  # no raise

    def test_returns_early_when_output_dir_missing(self):
        cfg = SimpleNamespace(output_dir="/nonexistent/path")
        check_existing_checkpoints(cfg)  # no raise

    def test_raises_when_checkpoints_exist_without_auto_resume(self, tmp_path):
        (tmp_path / "checkpoint-100").mkdir()
        cfg = SimpleNamespace(output_dir=str(tmp_path), auto_resume=False)
        with pytest.raises(ValueError, match="already contains checkpoints"):
            check_existing_checkpoints(cfg)

    def test_sets_resume_from_checkpoint_when_auto_resume(self, tmp_path):
        for i in (100, 200):
            (tmp_path / f"checkpoint-{i}").mkdir()
        cfg = SimpleNamespace(output_dir=str(tmp_path), auto_resume=True)
        check_existing_checkpoints(cfg)
        assert cfg.resume_from_checkpoint == "checkpoint-200"


class TestPruneOldCheckpoints:
    def test_prunes_when_over_limit(self, tmp_path):
        for i in (100, 200, 300):
            (tmp_path / f"checkpoint-{i}").mkdir()
        prune_old_checkpoints(str(tmp_path), limit=2)
        remaining = sorted(d.name for d in tmp_path.iterdir() if d.is_dir())
        assert remaining == ["checkpoint-200", "checkpoint-300"]

    def test_keeps_all_under_limit(self, tmp_path):
        (tmp_path / "checkpoint-100").mkdir()
        prune_old_checkpoints(str(tmp_path), limit=3)
        assert (tmp_path / "checkpoint-100").exists()


class TestGetLatestCheckpoint:
    def test_returns_none_when_no_checkpoints(self, tmp_path):
        assert get_latest_checkpoint(str(tmp_path)) is None

    def test_returns_none_when_empty_output_dir(self, tmp_path):
        (tmp_path / "other_dir").mkdir()
        assert get_latest_checkpoint(str(tmp_path)) is None

    def test_returns_latest_by_step(self, tmp_path):
        for i in (100, 500, 200):
            (tmp_path / f"checkpoint-{i}").mkdir()
        assert get_latest_checkpoint(str(tmp_path)) == "checkpoint-500"


class TestResolveCheckpointPath:
    def test_raises_when_no_checkpoints_and_none(self, tmp_path):
        with pytest.raises(ValueError, match="No checkpoints found"):
            resolve_checkpoint_path(str(tmp_path), None)

    def test_uses_latest_when_none(self, tmp_path):
        (tmp_path / "checkpoint-100").mkdir()
        (tmp_path / "checkpoint-200").mkdir()
        path, step = resolve_checkpoint_path(str(tmp_path), None)
        assert path.name == "checkpoint-200"
        assert step == 200

    def test_resolves_relative_path(self, tmp_path):
        (tmp_path / "checkpoint-42").mkdir()
        path, step = resolve_checkpoint_path(str(tmp_path), "checkpoint-42")
        assert path == (tmp_path / "checkpoint-42").resolve()
        assert step == 42

    def test_resolves_absolute_path(self, tmp_path):
        ckpt = tmp_path / "checkpoint-99"
        ckpt.mkdir()
        path, step = resolve_checkpoint_path(str(tmp_path), str(ckpt))
        assert path == ckpt.resolve()
        assert step == 99

    def test_raises_when_checkpoint_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            resolve_checkpoint_path(str(tmp_path), "checkpoint-999")
