from __future__ import annotations

import json

import mlflow

from house_prices_ml_foundations.config.mlflow_config import MLFLOW_TRACKING_URI
from house_prices_ml_foundations.config.paths import (
    get_paths,
    get_project_root,
    latest_file,
)
from house_prices_ml_foundations.data.load import load_train_test
from house_prices_ml_foundations.features.build import make_features
from house_prices_ml_foundations.io.model_artifacts import save_model
from house_prices_ml_foundations.io.run_id import make_run_id
from house_prices_ml_foundations.mlops.mlflow_tracking import (
    configure_mlflow,
    log_artifact_if_exists,
    log_metrics,
    log_model,
    log_params,
    set_tags,
    start_mlflow_run,
)
from house_prices_ml_foundations.models.champion import build_champion_pipeline


def main() -> None:
    """Train Champion Model on full training data and export it as champion.joblib.

    Metrics are read from the latest holdout evaluation (script 04) to stay
    consistent with the model that is actually exported as champion.joblib."""

    configure_mlflow(
        experiment_name="house-prices-champion", tracking_uri=MLFLOW_TRACKING_URI
    )
    print(f"TRACKING URI MLFLOW : {mlflow.get_tracking_uri()}")

    run_id = make_run_id(tag="champion")
    root_dir = get_project_root()
    paths = get_paths(root_dir)
    # Paths for local files (not MLflow artifacts)
    figure_path = paths["figures"]
    report_path = paths["reports"]
    models_path = paths["models"]
    versioned_model_path = paths["models"] / f"{run_id}.joblib"
    stable_model_path = paths["models"] / "champion.joblib"

    # Load and make features on full training data
    train_df, _ = load_train_test(root_dir)
    X_train, y_train = make_features(train_df)
    pipe, champion_source = build_champion_pipeline(paths["reports"])
    pipe.fit(X_train, y_train)

    # save model locally (not as MLflow artifact) for champion.joblib export and versioning
    save_model(pipe, versioned_model_path)
    save_model(pipe, stable_model_path)

    # Holdout metrics (computed in script 04) — logged for traceability, not recomputed here
    latest_holdout_json = latest_file(report_path, "rf_final_holdout_*.json")
    with open(latest_holdout_json) as f:
        holdout_report = json.load(f)

    metrics = {
        "holdout_mae": holdout_report["holdout"]["mae"],
        "holdout_rmse": holdout_report["holdout"]["rmse"],
        "holdout_r2": holdout_report["holdout"]["r2"],
    }

    # create dict of param to log in MLflow
    params = {
        "run_id": run_id,
        "champion_source": champion_source,
        "metrics_source": "script_04_holdout",
        **{k: v for k, v in pipe.get_params().items() if k.startswith("model__")},
    }

    # === Logs MLflow ===
    with start_mlflow_run(run_id=run_id) as run:
        # tags to easily filter and find the run in MLflow UI
        set_tags({
            "run_type": "packaging",
            "model_name": "champion_rf",
            "champion_source": champion_source,
            "data": "kaggle_house_prices/train.csv",
        })
        # metadata
        log_params(params=params)
        # Metrics (from holdout evaluation — script 04)
        log_metrics(metrics=metrics)
        # Artifacts
        latest_report = latest_file(report_path, "REPORT_*.md")
        if latest_report is not None:
            log_artifact_if_exists(latest_report)

        latest_residuals_hist = latest_file(figure_path, "residuals_hist*.png")
        if latest_residuals_hist is not None:
            log_artifact_if_exists(latest_residuals_hist)

        latest_ytrue_vs_ypred = latest_file(figure_path, "ytrue_vs_ypred*.png")
        if latest_ytrue_vs_ypred is not None:
            log_artifact_if_exists(latest_ytrue_vs_ypred)

        latest_abs_error_vs_ytrue = latest_file(figure_path, "abs_error_vs_ytrue*.png")
        if latest_abs_error_vs_ytrue is not None:
            log_artifact_if_exists(latest_abs_error_vs_ytrue)

        latest_error_summary = latest_file(report_path, "error_analysis_*_summary.json")
        if latest_error_summary is not None:
            log_artifact_if_exists(latest_error_summary)

        # JSON 
        latest_holdout_json = latest_file(report_path, "rf_final_holdout_*.json")
        if latest_holdout_json is not None:
            log_artifact_if_exists(latest_holdout_json)

        latest_tuning_json = latest_file(report_path, "tuning_rf_*.json")
        if latest_tuning_json is not None:
            log_artifact_if_exists(latest_tuning_json)

        latest_comparison_json = latest_file(report_path, "report_model_comparison_*.json")
        if latest_comparison_json is not None:
            log_artifact_if_exists(latest_comparison_json)

        # Model
        log_model(pipe, artifact_path="model")

        champion_joblib = models_path / "champion.joblib"
        if champion_joblib.exists():
            log_artifact_if_exists(champion_joblib)
            print(f"[MLflow] model artifact logged: {champion_joblib.name}")
        else:
            print("[MLflow] champion.joblib not found locally, skipped.")

        print("\n=== TRAIN + LOG MLFLOW DONE ===")
        print(f"run_name      : {run_id}")
        print(f"mlflow_run_id : {run.info.run_id}")
        print(f"champion_src  : {champion_source}")

    print(" === Champion model trained and exported === ")
    print(f"Champion source: {champion_source}")
    for key, value in params.items():
        print(f" {key} : {value}")

    print(f"Versioned model saved to: {versioned_model_path}")
    print(f"Stable model saved to: {stable_model_path}")


if __name__ == "__main__":
    main()
