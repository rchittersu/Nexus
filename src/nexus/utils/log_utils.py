"""
Logging utilities: MLflow setup and tracker configuration.
"""

import os
from pathlib import Path


def uses_mlflow(report_to) -> bool:
    """Return True if MLflow is in the report_to list or string."""
    return report_to == "mlflow" or (isinstance(report_to, list) and "mlflow" in report_to)


def setup_mlflow_log_with(report_to, output_dir: str | Path, mlflow_cfg=None):
    """
    Setup MLflow tracking and return log_with for Accelerator.
    If MLflow is in report_to: creates mlruns dir, sets MLFLOW_TRACKING_URI, returns MLflowTracker.
    Otherwise returns report_to unchanged.
    """
    if not uses_mlflow(report_to):
        return report_to
    mlflow_dir = Path(output_dir).resolve() / "mlruns"
    mlflow_dir.mkdir(parents=True, exist_ok=True)
    tracking_uri = (
        getattr(mlflow_cfg, "tracking_uri", None) if mlflow_cfg else None
    ) or mlflow_dir.as_uri()
    os.environ.setdefault("MLFLOW_TRACKING_URI", tracking_uri)
    from accelerate.tracking import MLflowTracker
    experiment_name = getattr(mlflow_cfg, "experiment_name", "nexus-flux2") if mlflow_cfg else "nexus-flux2"
    mlflow_tracker = MLflowTracker(
        experiment_name=experiment_name,
        logging_dir=str(mlflow_dir),
    )
    if report_to == "mlflow":
        return mlflow_tracker
    return [t for t in report_to if t != "mlflow"] + [mlflow_tracker]


def get_experiment_name(report_to, mlflow_cfg=None) -> str:
    """Return experiment name for init_trackers."""
    if uses_mlflow(report_to) and mlflow_cfg:
        return getattr(mlflow_cfg, "experiment_name", "nexus-flux2")
    return "nexus-flux2"
