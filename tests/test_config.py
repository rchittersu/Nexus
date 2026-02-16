"""Tests for nexus.train.config"""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from nexus.train.config import load_config, ns_to_kwargs, parse_args


class TestNsToKwargs:
    """Tests for ns_to_kwargs()."""

    def test_none_returns_overrides(self):
        assert ns_to_kwargs(None, x=1) == {"x": 1}

    def test_empty_namespace_returns_overrides(self):
        ns = SimpleNamespace()
        assert ns_to_kwargs(ns, a=1) == {"a": 1}

    def test_converts_namespace_to_dict(self):
        ns = SimpleNamespace(foo=1, bar="baz")
        assert ns_to_kwargs(ns) == {"foo": 1, "bar": "baz"}

    def test_skips_private_attrs(self):
        ns = SimpleNamespace(public=1, _private=2)
        assert ns_to_kwargs(ns) == {"public": 1}

    def test_overrides_update_result(self):
        ns = SimpleNamespace(a=1, b=2)
        assert ns_to_kwargs(ns, b=99, c=3) == {"a": 1, "b": 99, "c": 3}


class TestLoadConfig:
    """Tests for load_config()."""

    def test_load_minimal_config(self, tmp_path):
        """Load config with loss class only (no heavy deps)."""
        cfg_path = tmp_path / "minimal.yaml"
        cfg_path.write_text("""
loss:
  class_name: nexus.train.losses:MSELoss
""")
        cfg = load_config(cfg_path)
        assert hasattr(cfg.loss, "_class")
        assert cfg.loss._class.__name__ == "MSELoss"

    def test_extends_merges_base(self, tmp_path):
        """Child config deep-merges over base."""
        base = tmp_path / "base.yaml"
        base.write_text("""
a: 1
b:
  x: 10
  y: 20
""")
        child = tmp_path / "child.yaml"
        child.write_text("""
extends: base.yaml
b:
  y: 99
  z: 30
c: 3
""")
        cfg = load_config(child)
        assert cfg.a == 1
        assert cfg.b.x == 10
        assert cfg.b.y == 99
        assert cfg.b.z == 30
        assert cfg.c == 3

    def test_extends_removed_from_output(self, tmp_path):
        """extends key is not in final config."""
        base = tmp_path / "base.yaml"
        base.write_text("x: 1")
        child = tmp_path / "child.yaml"
        child.write_text("extends: base.yaml\n y: 2")
        cfg = load_config(child)
        assert not hasattr(cfg, "extends")

    def test_resolve_class_invalid_format_raises(self, tmp_path):
        """Invalid class_name format raises ValueError."""
        cfg_path = tmp_path / "bad.yaml"
        cfg_path.write_text("""
loss:
  class_name: "NoColon"
""")
        with pytest.raises(ValueError, match="module:ClassName"):
            load_config(cfg_path)


class TestLoadConfigFull:
    """Tests for load_config with real klein4b config. Requires train deps."""

    @pytest.fixture
    def config_dir(self):
        return Path(__file__).resolve().parents[1] / "configs" / "klein4b"

    def test_load_run1_resolves_classes(self, config_dir):
        """run1.yaml extends base and resolves all class references."""
        if not (config_dir / "run1.yaml").exists():
            pytest.skip("configs/klein4b/run1.yaml not found")
        cfg = load_config(config_dir / "run1.yaml")
        assert hasattr(cfg.dataset, "_class")
        assert hasattr(cfg.model.transformer, "_class")
        assert hasattr(cfg.loss, "_class")
        assert hasattr(cfg.optimizer, "_class")
        assert cfg.train.max_steps == 1000


class TestParseArgs:
    """Tests for parse_args()."""

    def test_missing_config_raises(self):
        """Missing or invalid --config raises."""
        with patch("sys.argv", ["main", "--config", "/nonexistent/path.yaml"]):
            with pytest.raises((SystemExit, FileNotFoundError, OSError)):
                parse_args()

    def test_overrides_applied_when_config_exists(self):
        """CLI overrides apply when using real config. Requires train deps."""
        pytest.importorskip("diffusers")
        config_dir = Path(__file__).resolve().parents[1] / "configs" / "klein4b"
        run1 = config_dir / "run1.yaml"
        if not run1.exists():
            pytest.skip("configs/klein4b/run1.yaml not found")
        args = [
            "--config", str(run1),
            "--output_dir", "/cli/out",
            "--precomputed_data_dir", "/cli/mds",
            "--max_train_steps", "500",
        ]
        cfg = parse_args(args)
        assert cfg.output_dir == "/cli/out"
        assert cfg.dataset.kwargs.local == "/cli/mds"
        assert cfg.train.max_steps == 500
        assert hasattr(cfg, "_config_path")
        assert "run1" in cfg._config_path
