"""
Logging utilities: MLflow setup and tracker configuration.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from accelerate import Accelerator


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


def setup_mlflow_log_with(
    log_root: str | Path,
    mlflow_cfg,
    mlflow_run_name: str,
    run_id: str | None = None,
) -> "MLflowTracker":
    """
    Setup MLflow tracking. Returns MLflowTracker.
    MLflow store at project level: log_root/mlruns.
    run_name is "{run_name}-{user}" for 1-to-1 mapping with output_dir.
    If run_id is given (resume), continues the existing run; otherwise creates a new run.
    """
    mlflow_dir = Path(log_root).resolve() / "mlruns"
    mlflow_dir.mkdir(parents=True, exist_ok=True)
    tracking_uri = getattr(mlflow_cfg, "tracking_uri", None) or mlflow_dir.as_uri()
    os.environ.setdefault("MLFLOW_TRACKING_URI", tracking_uri)
    from accelerate.tracking import MLflowTracker
    return MLflowTracker(
        experiment_name=mlflow_cfg.experiment_name,
        logging_dir=str(mlflow_dir),
        run_name=mlflow_run_name,
        run_id=run_id,
    )


def set_mlflow_user_tag(user: str | None) -> None:
    """Set mlflow.user tag on the current run if user is set (skip when 'default')."""
    u = (user and str(user).strip()) or ""
    if not u or u == "default":
        return
    try:
        import mlflow
        if mlflow.active_run():
            mlflow.set_tag("mlflow.user", str(user).strip())
    except Exception:
        pass


def get_output_dir(log_root: str | Path, experiment_name: str, run_name: str, user: str | None = None) -> Path:
    """Output dir for a run: log_root/experiments/{user}/{experiment_name}-{run_name}. User defaults to 'default' if null."""
    user_dir = user or "default"
    return Path(log_root).resolve() / "experiments" / user_dir / f"{experiment_name}-{run_name}"


def _find_mlflow_run_by_name(experiment_name: str, mlflow_run_name: str, tracking_uri: str | None = None) -> str | None:
    """Search for run with given name in experiment. Returns run_id if found."""
    try:
        from mlflow import MlflowClient

        client = MlflowClient(tracking_uri=tracking_uri)
        exps = client.search_experiments(filter_string=f"name = '{experiment_name}'")
        if not exps:
            return None
        exp_id = exps[0].experiment_id
        runs = client.search_runs(
            experiment_ids=[exp_id],
            filter_string=f'tags.`mlflow.runName` = "{mlflow_run_name}"',
            max_results=1,
        )
        if runs:
            return runs[0].info.run_id
    except Exception:
        pass
    return None


def _purge_mlflow_run_by_name(
    experiment_name: str, mlflow_run_name: str, tracking_uri: str | None = None
) -> None:
    """Delete existing run with given name in experiment (for overwrite when not resuming)."""
    run_id = _find_mlflow_run_by_name(experiment_name, mlflow_run_name, tracking_uri)
    if run_id:
        try:
            from mlflow import MlflowClient

            MlflowClient(tracking_uri=tracking_uri).delete_run(run_id)
        except Exception:
            pass


def setup_tracking(cfg) -> tuple[str, "MLflowTracker"]:
    """
    Validate mlflow config and setup tracking. Sets cfg.output_dir.
    MLflow run name is "{run_name}-{user}" for 1-to-1 mapping with output_dir.
    Runs checkpoint check: if resuming, finds existing run by name and continues; else creates new run.
    Returns (output_dir, log_with) for use with Accelerator.
    """
    from nexus.utils.checkpoint_utils import check_existing_checkpoints

    _require_mlflow(cfg)
    mlflow_cfg = cfg.mlflow
    log_root = Path(getattr(cfg, "log_root", "logs")).resolve()
    experiment_name = mlflow_cfg.experiment_name
    run_name = mlflow_cfg.run_name
    run_user = getattr(mlflow_cfg, "user", "default")
    output_dir = str(get_output_dir(log_root, experiment_name, run_name, user=run_user))
    cfg.output_dir = output_dir

    mlflow_run_name = f"{run_name}-{run_user}"
    check_existing_checkpoints(cfg)

    mlflow_dir = Path(log_root).resolve() / "mlruns"
    tracking_uri = getattr(mlflow_cfg, "tracking_uri", None) or mlflow_dir.as_uri()
    os.environ.setdefault("MLFLOW_TRACKING_URI", tracking_uri)

    run_id = None
    uri = tracking_uri if tracking_uri.startswith(("http", "file")) else Path(tracking_uri).resolve().as_uri()
    if getattr(cfg, "resume_from_checkpoint", None):
        run_id = _find_mlflow_run_by_name(experiment_name, mlflow_run_name, uri)
    else:
        _purge_mlflow_run_by_name(experiment_name, mlflow_run_name, uri)

    log_with = setup_mlflow_log_with(
        log_root, mlflow_cfg, mlflow_run_name=mlflow_run_name, run_id=run_id
    )
    return output_dir, log_with


def init_trackers(accelerator: "Accelerator", cfg) -> None:
    """
    Init MLflow trackers and set user tag. Call when accelerator.is_main_process.
    MLflow run name is "{run_name}-{user}" for 1-to-1 mapping with output_dir (no run_id file).
    When resuming, skip config param logging (MLflow params are immutable).
    """
    mlflow_cfg = cfg.mlflow
    resuming = bool(getattr(cfg, "resume_from_checkpoint", None))
    config_dict = {}
    if not resuming:
        for k, v in vars(cfg).items():
            if not k.startswith("_"):
                try:
                    config_dict[k] = str(v)
                except Exception:
                    config_dict[k] = repr(v)
    accelerator.init_trackers(mlflow_cfg.experiment_name, config=config_dict)
    set_mlflow_user_tag(getattr(mlflow_cfg, "user", "default"))


def log_dataset_input(
    class_name: str | None = None,
    kwargs: dict | None = None,
    name: str | None = None,
    source_path: str | Path | None = None,
    context: str = "training",
    resuming: bool = False,
) -> None:
    """
    Log dataset to MLflow run. Records class and kwargs as params. If source_path is given,
    also logs as MetaDataset for lineage. Name from config or class_name fallback.
    When resuming, skip param logging (MLflow params are immutable).
    """
    try:
        import mlflow

        dataset_name = name or (class_name.split(":")[-1] if class_name else "dataset")
        if not resuming:
            if class_name is not None:
                mlflow.log_param("dataset.class", class_name)
            if kwargs:
                mlflow.log_param("dataset.kwargs", str(kwargs))
        if source_path:
            import warnings

            from mlflow.data.dataset_source_registry import resolve_dataset_source
            from mlflow.data.meta_dataset import MetaDataset

            path = str(Path(source_path).resolve())
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                src = resolve_dataset_source(path)
            dataset = MetaDataset(source=src, name=dataset_name)
            mlflow.log_input(dataset, context=context)
    except Exception:
        pass


def log_validation_images_to_mlflow(images: list, step: int, output_dir: str | Path) -> None:
    """
    Save validation images under output_dir/validation_images/ and log to MLflow.
    Requires an active MLflow run (e.g. during training).
    """
    import mlflow

    val_dir = Path(output_dir) / "validation_images"
    val_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images):
        path = val_dir / f"validation_step{step}_img{i}.png"
        img.save(path)
        mlflow.log_artifact(str(path), artifact_path="validation")


def activate_mlflow_run_and_log_validation(
    output_dir: str | Path,
    cfg,
    images: list,
    step: int,
    targets: list | None = None,
) -> None:
    """
    Activate the MLflow run for this experiment (by name), save validation images
    to output_dir/validation/, and log them as artifacts.
    targets: optional list of (PIL.Image or path) for reference images, aligned with images.
    """
    try:
        import mlflow
        from PIL import Image
    except ImportError:
        return

    mlflow_cfg = getattr(cfg, "mlflow", None)
    if not mlflow_cfg:
        return

    experiment_name = getattr(mlflow_cfg, "experiment_name", None)
    run_name = getattr(mlflow_cfg, "run_name", None)
    if not experiment_name or not run_name:
        return

    run_user = getattr(mlflow_cfg, "user", None) or "default"
    mlflow_run_name = f"{run_name}-{run_user}"

    log_root = Path(getattr(cfg, "log_root", "logs")).resolve()
    mlflow_dir = log_root / "mlruns"
    tracking_uri = getattr(mlflow_cfg, "tracking_uri", None) or mlflow_dir.as_uri()
    uri = tracking_uri if str(tracking_uri).startswith(("http", "file")) else Path(tracking_uri).resolve().as_uri()

    run_id = _find_mlflow_run_by_name(experiment_name, mlflow_run_name, uri)
    if not run_id:
        return

    val_dir = Path(output_dir) / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)

    import os
    os.environ.setdefault("MLFLOW_TRACKING_URI", str(uri))
    with mlflow.start_run(run_id=run_id):
        for i, img in enumerate(images):
            path = val_dir / f"validation_step{step}_img{i}.png"
            img.save(path)
            mlflow.log_artifact(str(path), artifact_path="validation")
        if targets:
            for i in range(len(images)):
                if i >= len(targets) or targets[i] is None:
                    continue
                t = targets[i]
                if isinstance(t, (str, Path)):
                    src = Path(t)
                    if src.exists():
                        mlflow.log_artifact(str(src), artifact_path="validation/targets")
                elif isinstance(t, Image.Image):
                    p = val_dir / f"validation_step{step}_target{i}.png"
                    t.save(p)
                    mlflow.log_artifact(str(p), artifact_path="validation/targets")
