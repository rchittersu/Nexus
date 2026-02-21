"""
Logging utilities: MLflow setup and tracker configuration.
"""

import os
from pathlib import Path


def _require_mlflow(cfg) -> None:
    """Assert MLflow is configured. Raises if report_to != 'mlflow' or mlflow.experiment_name/run_name missing."""
    if cfg.report_to != "mlflow":
        raise ValueError("report_to must be 'mlflow'")
    mlflow_cfg = getattr(cfg, "mlflow", None)
    if not mlflow_cfg:
        raise ValueError("mlflow config required (mlflow: experiment_name: ..., run_name: ...)")
    if not getattr(mlflow_cfg, "experiment_name", None):
        raise ValueError("mlflow.experiment_name is required")
    if not getattr(mlflow_cfg, "run_name", None):
        raise ValueError("mlflow.run_name is required")


def setup_mlflow_log_with(log_root: str | Path, mlflow_cfg) -> "MLflowTracker":
    """
    Setup MLflow tracking. Returns MLflowTracker.
    MLflow store at project level: log_root/mlruns.
    Sets run_name from mlflow.run_name.
    """
    mlflow_dir = Path(log_root).resolve() / "mlruns"
    mlflow_dir.mkdir(parents=True, exist_ok=True)
    tracking_uri = getattr(mlflow_cfg, "tracking_uri", None) or mlflow_dir.as_uri()
    os.environ.setdefault("MLFLOW_TRACKING_URI", tracking_uri)
    from accelerate.tracking import MLflowTracker
    return MLflowTracker(
        experiment_name=mlflow_cfg.experiment_name,
        logging_dir=str(mlflow_dir),
        run_name=mlflow_cfg.run_name,
    )


def get_output_dir(log_root: str | Path, experiment_name: str, run_name: str) -> Path:
    """Output dir for a run: log_root/experiments/{experiment_name}-{run_name}."""
    return Path(log_root).resolve() / "experiments" / f"{experiment_name}-{run_name}"


def log_validation_images_to_mlflow(images: list, step: int, output_dir: str | Path) -> None:
    """
    Save validation images under output_dir/validation_images/ and log to MLflow.
    """
    import mlflow

    val_dir = Path(output_dir) / "validation_images"
    val_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images):
        path = val_dir / f"validation_step{step}_img{i}.png"
        img.save(path)
        mlflow.log_artifact(str(path), artifact_path="validation")
