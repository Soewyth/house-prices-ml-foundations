from pathlib import Path

import mlflow
from sklearn.pipeline import Pipeline


def configure_mlflow(experiment_name: str = "house-prices-experiment", tracking_uri: str | None = None) -> None:
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name=experiment_name)


def start_mlflow_run(run_id: str):
    return mlflow.start_run(run_name=run_id)

def log_params(params: dict) -> None:
    mlflow.log_params(params)

def set_tags(tags: dict) -> None:
    mlflow.set_tags(tags)

def log_metrics(metrics: dict) -> None:
    mlflow.log_metrics(metrics)

def log_artifact_if_exists(file_path: str | Path, artifact_path: str | None = None) -> bool:
    file_path = Path(file_path)
    if not file_path.exists():
        print(f"[MLflow] Artifact not found, skipped: {file_path}")
        return False
    mlflow.log_artifact(str(file_path), artifact_path=artifact_path)
    return True

def log_model(model_pipeline: Pipeline, artifact_path: str  = "model") -> None:
    mlflow.sklearn.log_model(model_pipeline, artifact_path=artifact_path)