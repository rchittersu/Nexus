"""Tests for nexus.utils.log_utils"""
from types import SimpleNamespace

import pytest

from nexus.utils.log_utils import _require_mlflow, get_output_dir, setup_mlflow_log_with


class TestRequireMlflow:
    def test_raises_when_report_to_not_mlflow(self):
        cfg = SimpleNamespace(report_to="tensorboard", mlflow=SimpleNamespace(experiment_name="x"))
        with pytest.raises(ValueError, match="report_to must be 'mlflow'"):
            _require_mlflow(cfg)

    def test_raises_when_no_mlflow_config(self):
        cfg = SimpleNamespace(report_to="mlflow")
        with pytest.raises(ValueError, match="mlflow config required"):
            _require_mlflow(cfg)

    def test_raises_when_no_experiment_name(self):
        cfg = SimpleNamespace(report_to="mlflow", mlflow=SimpleNamespace(run_name="x"))
        with pytest.raises(ValueError, match="experiment_name is required"):
            _require_mlflow(cfg)

    def test_raises_when_no_run_name(self):
        cfg = SimpleNamespace(report_to="mlflow", mlflow=SimpleNamespace(experiment_name="x"))
        with pytest.raises(ValueError, match="run_name is required"):
            _require_mlflow(cfg)

    def test_passes_when_configured(self):
        cfg = SimpleNamespace(report_to="mlflow", mlflow=SimpleNamespace(experiment_name="my-exp", run_name="my-run"))
        _require_mlflow(cfg)  # no raise


class TestSetupMlflowLogWith:
    def test_returns_tracker(self, tmp_path):
        cfg = SimpleNamespace(experiment_name="my-exp", run_name="my-run", tracking_uri=None)
        result = setup_mlflow_log_with(tmp_path, cfg)
        from accelerate.tracking import MLflowTracker
        assert isinstance(result, MLflowTracker)
        assert (tmp_path / "mlruns").exists()


class TestGetOutputDir:
    def test_output_dir_path(self, tmp_path):
        out = get_output_dir(tmp_path, "nexus-flux2", "flux2-dreambooth-lora")
        assert out == tmp_path / "experiments" / "nexus-flux2-flux2-dreambooth-lora"
