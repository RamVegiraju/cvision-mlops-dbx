"""Local end-to-end smoke test of the exact code the job runs, on a tiny real subset (CPU, ~1 min/backbone).

Requires the Imagenette archive extracted under scratch/data (see README). Skips otherwise.
"""

import json
from pathlib import Path

import mlflow
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from mlflow.tracking import MlflowClient

from cv_mlops import config, data, evaluation, modeling, pipeline, serving

DATA_ROOT = Path(__file__).parents[1] / "scratch" / "data" / config.IMAGENETTE_DIRNAME

VERSIONS: dict[str, str] = {}

pytestmark = pytest.mark.skipif(not DATA_ROOT.exists(), reason="local Imagenette copy not present")


@pytest.fixture(scope="module")
def manifest():
    df = data.validate_images(data.build_manifest(DATA_ROOT, max_per_class=4))
    data.check_manifest(df)
    return df


@pytest.fixture(scope="module")
def tracking(tmp_path_factory):
    root = tmp_path_factory.mktemp("mlruns")
    mlflow.set_tracking_uri(f"sqlite:///{root}/mlflow.db")
    mlflow.set_registry_uri(f"sqlite:///{root}/mlflow.db")
    mlflow.set_experiment("smoke")
    yield
    mlflow.set_tracking_uri(None)


def test_manifest_shape(manifest):
    assert set(manifest["split"]) == set(config.SPLITS)
    assert manifest.groupby(["split", "label"]).size().max() == 4
    assert manifest["is_valid"].all()
    # deterministic subset → stable digest across runs
    again = data.build_manifest(DATA_ROOT, max_per_class=4)
    assert data.manifest_digest(manifest, "test") == data.manifest_digest(again, "test")


def test_check_manifest_fails_fast_on_missing_class(manifest):
    broken = manifest[~((manifest["split"] == "test") & (manifest["label"] == "church"))]
    with pytest.raises(ValueError, match="missing classes"):
        data.check_manifest(broken)


@pytest.mark.parametrize("backbone", ["mobilenet_v3_large", "efficientnet_b0"])
def test_train_log_reload_predict(manifest, tracking, backbone):
    train_df = manifest[manifest["split"] == "train"]
    val_df = manifest[manifest["split"] == "val"]
    test_df = manifest[manifest["split"] == "test"]
    model = modeling.build_model(backbone)
    counts = modeling.count_parameters(model)
    assert 0 < counts["params_trainable"] < counts["params_total"]

    train_loader = DataLoader(data.ManifestDataset(train_df, modeling.train_transform()), batch_size=16, shuffle=True)
    val_loader = DataLoader(data.ManifestDataset(val_df, modeling.eval_transform()), batch_size=16)

    with mlflow.start_run() as run:
        history = modeling.train(
            model, train_loader, val_loader, epochs=1, lr=1e-3, device=torch.device("cpu"),
            on_epoch_end=lambda r: mlflow.log_metrics({"val_accuracy": r.val_accuracy}, step=r.epoch),
        )
        info = serving.log_model(model, backbone, test_df["path"].head(2).tolist(), registered_model_name="smoke_clf")
    assert len(history) == 1 and history[0].val_accuracy >= 0
    assert info.registered_model_version is not None

    # Reload from the registry exactly as batch inference / serving will.
    loaded = mlflow.pyfunc.load_model(f"models:/smoke_clf/{info.registered_model_version}")
    inp = pd.DataFrame({serving.INPUT_COLUMN: [serving.encode_image_file(p) for p in test_df["path"]]})
    out = loaded.predict(inp)
    assert list(out.columns) == ["label", "label_idx", "confidence", "top_k"]
    assert len(out) == len(test_df)
    assert out["label"].isin(config.CLASS_NAMES).all()
    assert out["confidence"].between(0, 1).all()
    assert len(json.loads(out["top_k"].iloc[0])) == serving.TOP_K

    # Evaluation step exactly as the eval task runs it.
    client = MlflowClient()
    version = str(info.registered_model_version)
    client.set_model_version_tag("smoke_clf", version, "backbone", backbone)
    eval_df = test_df.assign(image=inp[serving.INPUT_COLUMN].values)
    result = pipeline.evaluate_model_version(client, "smoke_clf", version, eval_df, data.manifest_digest(test_df, "test"))
    assert 0 <= result["test_accuracy"] <= result["test_top3_accuracy"] <= 1
    assert result["latency_p95_ms"] > 0
    artifacts = {a.path for a in client.list_artifacts(run.info.run_id, "eval")}
    assert "eval/confusion_matrix.png" in artifacts
    assert "accuracy_score" in client.get_run(run.info.run_id).data.metrics
    VERSIONS[backbone] = version


def test_select_and_promote(tracking):
    """select_best logic against the registry populated above."""
    assert len(VERSIONS) == 2, "train tests must run first"
    client = MlflowClient()
    cands = [pipeline.candidate_from_registry(client, "smoke_clf", v) for v in VERSIONS.values()]
    winner, reason = evaluation.pick_winner(cands, accuracy_tolerance=0.01)
    pipeline.promote(client, "smoke_clf", winner.version, reason)
    champ = client.get_model_version_by_alias("smoke_clf", pipeline.CHAMPION_ALIAS)
    assert str(champ.version) == winner.version
    assert champ.tags["selection_reason"] == reason


def test_pip_requirements_pin_cpu_torch_without_cuda_tags():
    reqs = serving.pip_requirements()
    assert reqs[0].startswith("--extra-index-url")
    assert not any("+cu" in r for r in reqs)
    assert any(r.startswith("torch==") and r.endswith("+cpu") for r in reqs)
    assert all("+" not in r for r in serving.pip_requirements(cpu_torch=False))


def test_pip_requirements_without_dist_metadata(monkeypatch):
    """AI Runtime ships mlflow without dist-info; pinning must not depend on importlib.metadata."""
    from importlib.metadata import PackageNotFoundError

    def no_metadata(name):
        raise PackageNotFoundError(name)

    monkeypatch.setattr(serving, "version", no_metadata)
    reqs = serving.pip_requirements()
    assert f"mlflow=={mlflow.__version__}" in reqs
    assert any(r.startswith("pillow==") for r in reqs)
