# Databricks notebook source
# MAGIC %md
# MAGIC # 4 · Select best model
# MAGIC Compare the two candidates evaluated in this run: highest test accuracy wins; if they are within
# MAGIC `accuracy_tolerance`, the faster model (p95 CPU latency) wins. The winner is tagged `@champion` in
# MAGIC Unity Catalog and handed to batch inference and deployment.

# COMMAND ----------

import json, os, sys

sys.path.insert(0, os.path.abspath(".."))

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient

from cv_mlops import evaluation, pipeline

model_name = dbutils.widgets.get("model_name")
experiment_path = dbutils.widgets.get("experiment_path")
candidate_versions = [v.strip() for v in dbutils.widgets.get("candidate_versions").split(",") if v.strip()]
accuracy_tolerance = float(dbutils.widgets.get("accuracy_tolerance"))
job_run_id = dbutils.widgets.get("job_run_id")

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(experiment_path)
client = MlflowClient(registry_uri="databricks-uc")

# COMMAND ----------

candidates = [pipeline.candidate_from_registry(client, model_name, v) for v in candidate_versions]
winner, reason = evaluation.pick_winner(candidates, accuracy_tolerance)
print(reason)

comparison = pd.DataFrame([{**c.__dict__, "winner": c.version == winner.version} for c in candidates])
display(comparison)

# COMMAND ----------
# MAGIC %md ## Record the decision and tag the winner `@champion`

# COMMAND ----------

with mlflow.start_run(run_name="select-best"):
    mlflow.set_tags({"stage": "selection", "job_run_id": job_run_id})
    mlflow.log_params(
        {"candidate_versions": ",".join(candidate_versions), "winner_version": winner.version,
         "winner_backbone": winner.backbone, "accuracy_tolerance": accuracy_tolerance}
    )
    mlflow.log_table(comparison, "selection/comparison.json")
    mlflow.log_text(reason + "\n", "selection/decision.txt")

pipeline.promote(client, model_name, winner.version, reason)
print(f"@champion → {model_name} v{winner.version} ({winner.backbone})")

# COMMAND ----------

dbutils.jobs.taskValues.set("winner_version", winner.version)
dbutils.jobs.taskValues.set("winner_backbone", winner.backbone)
dbutils.notebook.exit(json.dumps({"winner_version": winner.version, "backbone": winner.backbone, "reason": reason}))
