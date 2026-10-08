# Databricks notebook source
# MAGIC %md
# MAGIC # 2 · Train (Serverless GPU · AI Runtime)
# MAGIC Transfer-learn one ImageNet backbone on the silver manifest. The same notebook runs once per
# MAGIC backbone in parallel tasks — only the `backbone` parameter differs.
# MAGIC
# MAGIC Logs params, per-epoch metrics, dataset lineage and the model (pyfunc w/ preprocessing) to MLflow,
# MAGIC registers a new version in Unity Catalog and tags it `@candidate_<backbone>`.

# COMMAND ----------

import json, os, sys

sys.path.insert(0, os.path.abspath(".."))

import mlflow
import torch
from mlflow.tracking import MlflowClient
from torch.utils.data import DataLoader

from cv_mlops import data, modeling, pipeline, serving

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
volume = dbutils.widgets.get("volume")
model_name = dbutils.widgets.get("model_name")
experiment_path = dbutils.widgets.get("experiment_path")
backbone = dbutils.widgets.get("backbone")
epochs = int(dbutils.widgets.get("epochs"))
batch_size = int(dbutils.widgets.get("batch_size"))
lr = float(dbutils.widgets.get("learning_rate"))
min_val_accuracy = float(dbutils.widgets.get("min_val_accuracy"))
job_run_id = dbutils.widgets.get("job_run_id")

silver_table = f"{catalog}.{schema}.silver_image_manifest"
volume_root = f"/Volumes/{catalog}/{schema}/{volume}"

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(experiment_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "none"
print(f"backbone={backbone} device={device} gpu={gpu_name} epochs={epochs}")

# COMMAND ----------
# MAGIC %md ## Load the silver manifest

# COMMAND ----------

silver_sdf = spark.table(silver_table).where("is_valid")
manifest = silver_sdf.toPandas()
splits = {s: manifest[manifest["split"] == s] for s in ("train", "val", "test")}
assert len(splits["train"]) and len(splits["val"]), f"empty train/val split: { {k: len(v) for k, v in splits.items()} }"

workers = min(8, os.cpu_count() or 1)
train_loader = DataLoader(
    data.ManifestDataset(splits["train"], modeling.train_transform()),
    batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, persistent_workers=True,
)
val_loader = DataLoader(
    data.ManifestDataset(splits["val"], modeling.eval_transform()),
    batch_size=batch_size, num_workers=workers, pin_memory=True,
)

# COMMAND ----------
# MAGIC %md ## Train + log + register

# COMMAND ----------

modeling.use_weight_cache(f"{volume_root}/torch_hub")  # populated by prepare_data
model = modeling.build_model(backbone)

with mlflow.start_run(run_name=f"train-{backbone}") as run:
    mlflow.set_tags({"backbone": backbone, "job_run_id": job_run_id, "stage": "train", "gpu": gpu_name})
    mlflow.log_params(
        {"backbone": backbone, "epochs": epochs, "batch_size": batch_size, "learning_rate": lr,
         "n_train": len(splits["train"]), "n_val": len(splits["val"]), **modeling.count_parameters(model)}
    )
    mlflow.log_input(mlflow.data.from_spark(silver_sdf, table_name=silver_table), context="training")

    def log_epoch(r):
        mlflow.log_metrics(
            {"train_loss": r.train_loss, "train_accuracy": r.train_accuracy,
             "val_loss": r.val_loss, "val_accuracy": r.val_accuracy, "epoch_seconds": r.seconds},
            step=r.epoch,
        )
        print(f"epoch {r.epoch}: train_acc={r.train_accuracy:.4f} val_acc={r.val_accuracy:.4f} ({r.seconds:.0f}s)")

    history = modeling.train(model, train_loader, val_loader, epochs, lr, device, on_epoch_end=log_epoch)
    final_val = history[-1].val_accuracy
    mlflow.log_metric("final_val_accuracy", final_val)

    # Fail fast: a broken run must never reach the registry.
    if final_val < min_val_accuracy:
        raise RuntimeError(f"val_accuracy {final_val:.4f} < sanity floor {min_val_accuracy}; not registering")

    info = serving.log_model(model.cpu(), backbone, splits["val"]["path"].head(2).tolist(), registered_model_name=model_name)

version = str(info.registered_model_version)
client = MlflowClient(registry_uri="databricks-uc")
client.set_model_version_tag(model_name, version, "backbone", backbone)
client.set_model_version_tag(model_name, version, "job_run_id", job_run_id)
client.set_registered_model_alias(model_name, pipeline.candidate_alias(backbone), version)
print(f"registered {model_name} v{version} as @{pipeline.candidate_alias(backbone)}")

# COMMAND ----------

dbutils.jobs.taskValues.set("model_version", version)
dbutils.jobs.taskValues.set("run_id", run.info.run_id)
dbutils.jobs.taskValues.set("val_accuracy", final_val)
dbutils.notebook.exit(json.dumps({"model_version": version, "backbone": backbone, "val_accuracy": final_val}))
