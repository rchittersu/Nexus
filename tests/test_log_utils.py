"""Tests for nexus.utils.log_utils"""
from types import SimpleNamespace

import pytest

from nexus.utils.log_utils import get_experiment_name, setup_mlflow_log_with, uses_mlflow


class TestUsesMlflow:
    def test_mlflow_string(self):
        assert uses_mlflow("mlflow") is True

    def test_mlflow_in_list(self):
        assert uses_mlflow(["tensorboard", "mlflow"]) is True

    def test_no_mlflow(self):
        assert uses_mlflow("tensorboard") is False
        assert uses_mlflow(["tensorboard"]) is False
        assert uses_mlflow(None) is False


class TestSetupMlflowLogWith:
    def test_returns_report_to_when_no_mlflow(self, tmp_path):
        result = setup_mlflow_log_with("tensorboard", tmp_path)
        assert result == "tensorboard"
        result = setup_mlflow_log_with(["tensorboard"], tmp_path)
        assert result == ["tensorboard"]

    def test_returns_tracker_when_mlflow(self, tmp_path):
        result = setup_mlflow_log_with("mlflow", tmp_path)
        from accelerate.tracking import MLflowTracker
        assert isinstance(result, MLflowTracker)
        assert (tmp_path / "mlruns").exists()

    def test_uses_mlflow_cfg(self, tmp_path):
        cfg = SimpleNamespace(experiment_name="my-exp", tracking_uri=None)
        result = setup_mlflow_log_with("mlflow", tmp_path, cfg)
        from accelerate.tracking import MLflowTracker
        assert isinstance(result, MLflowTracker)


class TestGetExperimentName:
    def test_default_when_no_mlflow(self):
        assert get_experiment_name("tensorboard", None) == "nexus-flux2"

    def test_from_mlflow_cfg(self):
        cfg = SimpleNamespace(experiment_name="my-exp")
        assert get_experiment_name("mlflow", cfg) == "my-exp"

    def test_default_when_mlflow_but_no_cfg(self):
        assert get_experiment_name("mlflow", None) == "nexus-flux2"
