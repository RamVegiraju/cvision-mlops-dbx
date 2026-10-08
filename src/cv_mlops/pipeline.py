"""Orchestration-level steps shared by the job notebooks (and exercised by local tests).

Kept free of Spark so it runs anywhere: notebooks do the Spark I/O and hand pandas frames in.
"""

from __future__ import annotations

import json

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from cv_mlops import evaluation, serving

CHAMPION_ALIAS = "champion"
# Model-version tags written by evaluation and read back by selection — the registry is the
# contract between tasks, so select_best works the same whether 2 or 20 models were trained.
EVAL_TAGS = ("backbone", "test_accuracy", "test_top3_accuracy", "test_f1_macro", "latency_p95_ms", "test_digest")


def candidate_alias(backbone: str) -> str:
    return f"candidate_{backbone}"


def evaluate_model_version(
    client: MlflowClient, model_name: str, version: str, test_df: pd.DataFrame, test_digest: str
) -> dict:
    """Score ``test_df`` (columns: label, image[base64]) with a registered version and record results.

    Metrics + artifacts are logged into the version's own training run (one run = full story of
    that model), and headline numbers are mirrored to model-version tags for selection.
    """
    mv = client.get_model_version(model_name, version)
    model = mlflow.pyfunc.load_model(f"models:/{model_name}/{version}")
    backbone = mv.tags.get("backbone", "unknown")

    inp = test_df[[serving.INPUT_COLUMN]]
    preds = model.predict(inp)
    y_true, y_pred = test_df["label"].tolist(), preds["label"].tolist()
    top_hit = [t in json.loads(tk) for t, tk in zip(y_true, preds["top_k"])]

    metrics = evaluation.classification_metrics(y_true, y_pred, top_hit)
    metrics.update(evaluation.benchmark_latency(model.predict, inp.head(1)))
    metrics["test_n_images"] = len(test_df)

    with mlflow.start_run(run_id=mv.run_id):
        mlflow.log_metrics(metrics)
        mlflow.set_tag("test_digest", test_digest)
        mlflow.log_figure(
            evaluation.confusion_matrix_figure(y_true, y_pred, f"{backbone} v{version}"), "eval/confusion_matrix.png"
        )
        mlflow.log_table(evaluation.per_class_report(y_true, y_pred), "eval/per_class_metrics.json")
        wrong = test_df.assign(predicted=y_pred, confidence=preds["confidence"].values)
        wrong = wrong[wrong["label"] != wrong["predicted"]]
        mlflow.log_table(
            wrong[[c for c in ("path", "label", "predicted", "confidence") if c in wrong]].head(200),
            "eval/misclassified.json",
        )
        # Standard MLflow classifier evaluation on the static predictions (adds its own metrics/plots).
        mlflow.models.evaluate(
            data=pd.DataFrame({"target": y_true, "prediction": y_pred}),
            targets="target",
            predictions="prediction",
            model_type="classifier",
            evaluator_config={"log_model_explainability": False},
        )

    for k in ("test_accuracy", "test_top3_accuracy", "test_f1_macro", "latency_p95_ms"):
        client.set_model_version_tag(model_name, version, k, f"{metrics[k]:.6f}")
    client.set_model_version_tag(model_name, version, "test_digest", test_digest)
    return {"version": str(version), "backbone": backbone, **metrics}


def candidate_from_registry(client: MlflowClient, model_name: str, version: str) -> evaluation.Candidate:
    tags = client.get_model_version(model_name, version).tags
    missing = [t for t in ("test_accuracy", "latency_p95_ms", "test_digest") if t not in tags]
    if missing:
        raise ValueError(f"{model_name} v{version} has not been evaluated (missing tags {missing})")
    return evaluation.Candidate(
        version=str(version),
        backbone=tags.get("backbone", "unknown"),
        test_accuracy=float(tags["test_accuracy"]),
        latency_p95_ms=float(tags["latency_p95_ms"]),
        test_digest=tags["test_digest"],
    )



def promote(client: MlflowClient, model_name: str, version: str, reason: str) -> None:
    """Point @champion at the selected version — what batch scoring and serving consume."""
    client.set_registered_model_alias(model_name, CHAMPION_ALIAS, str(version))
    client.set_model_version_tag(model_name, str(version), "selection_reason", reason)
