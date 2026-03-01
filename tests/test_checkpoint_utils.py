"""Tests for nexus.utils.checkpoint_utils"""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from nexus.utils.checkpoint_utils import (
    check_existing_checkpoints,
    load_transformer_state,
    prune_old_checkpoints,
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
