# Databricks notebook source
# MAGIC %md
# MAGIC # 3 · Evaluate (CPU)
# MAGIC Score the held-out **test** split with the *registered* model (loaded from Unity Catalog — the
# MAGIC exact artifact that would be served), on CPU, so the latency numbers reflect CPU serving.
# MAGIC
# MAGIC Logs accuracy / top-3 / F1, confusion matrix, per-class metrics, misclassified examples and
# MAGIC `mlflow.models.evaluate` output into the model's training run; mirrors headline metrics to
# MAGIC model-version tags for the selection step.

# COMMAND ----------

import json, os, sys

sys.path.insert(0, os.path.abspath(".."))

import mlflow
from mlflow.tracking import MlflowClient

from cv_mlops import pipeline, spark_io

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
model_name = dbutils.widgets.get("model_name")
experiment_path = dbutils.widgets.get("experiment_path")
version = dbutils.widgets.get("model_version")

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(experiment_path)
client = MlflowClient(registry_uri="databricks-uc")

# COMMAND ----------

test_df, test_digest = spark_io.load_test_set(spark, catalog, schema)
print(f"evaluating {model_name} v{version} on {len(test_df)} test images (digest {test_digest})")

result = pipeline.evaluate_model_version(client, model_name, version, test_df, test_digest)
print(json.dumps(result, indent=2))

# COMMAND ----------

for k in ("model_version", "test_accuracy", "latency_p95_ms"):
    dbutils.jobs.taskValues.set(k, result["version"] if k == "model_version" else result[k])
dbutils.notebook.exit(json.dumps(result))
