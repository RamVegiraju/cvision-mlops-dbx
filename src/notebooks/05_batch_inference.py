# Databricks notebook source
# MAGIC %md
# MAGIC # 5a · Batch (offline) inference
# MAGIC Loads the selected (`@champion`) version from Unity Catalog and scores images read from the bronze/silver
# MAGIC Delta tables → `gold_predictions` (rows replaced per model version, so re-runs are idempotent).
# MAGIC
# MAGIC Images are streamed through the model in fixed-size chunks, so memory stays flat regardless of table size.
# MAGIC (A `spark_udf` runs out of memory on serverless here: each Python worker holds a full model copy plus a batch
# MAGIC of raw images, and serverless doesn't allow tuning the Arrow batch size. For much larger tables, run the
# MAGIC same model as a `spark_udf` on a classic cluster with larger workers.)

# COMMAND ----------

import json, os, sys

sys.path.insert(0, os.path.abspath(".."))

import mlflow
import pandas as pd
from pyspark.sql import functions as F

from cv_mlops import serving, spark_io

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
model_name = dbutils.widgets.get("model_name")
version = dbutils.widgets.get("model_version")
score_split = dbutils.widgets.get("score_split")
job_run_id = dbutils.widgets.get("job_run_id")

mlflow.set_registry_uri("databricks-uc")
gold_table = spark_io.tables(catalog, schema)["gold"]
model_uri = f"models:/{model_name}/{version}"  # pinned version (== @champion), not the alias: no race with a later run
CHUNK_SIZE = 256

# COMMAND ----------

model = mlflow.pyfunc.load_model(model_uri)
images = spark_io.images_with_bytes(spark, catalog, schema, score_split)

def score(chunk: list) -> pd.DataFrame:
    batch = pd.DataFrame(chunk, columns=["path", "label", serving.INPUT_COLUMN])
    preds = model.predict(batch[[serving.INPUT_COLUMN]])
    return pd.DataFrame(
        {"path": batch["path"], "true_label": batch["label"], "predicted_label": preds["label"],
         "confidence": preds["confidence"], "top_k": preds["top_k"]}
    )

chunks, buf = [], []
for row in images.toLocalIterator(prefetchPartitions=True):  # streams partitions; never collects all bytes
    buf.append((row.path, row.label, row[serving.INPUT_COLUMN]))
    if len(buf) == CHUNK_SIZE:
        chunks.append(score(buf))
        buf = []
if buf:
    chunks.append(score(buf))
assert chunks, f"no '{score_split}' images to score"

scored = (
    spark.createDataFrame(pd.concat(chunks, ignore_index=True))
    .withColumn("model_version", F.lit(version))
    .withColumn("job_run_id", F.lit(job_run_id))
    .withColumn("scored_at", F.current_timestamp())
)

# COMMAND ----------

writer = scored.write.format("delta").mode("overwrite")
if spark.catalog.tableExists(gold_table):
    writer = writer.option("replaceWhere", f"model_version = '{version}'")
writer.saveAsTable(gold_table)
spark.sql(f"COMMENT ON TABLE {gold_table} IS 'Batch predictions per image and model version'")

summary = (
    spark.table(gold_table)
    .where(F.col("model_version") == version)
    .agg(F.count("*").alias("n"), F.avg((F.col("true_label") == F.col("predicted_label")).cast("double")).alias("accuracy"))
    .first()
)
print(f"scored {summary.n} images with v{version}; batch accuracy={summary.accuracy:.4f}")
display(spark.table(gold_table).where(F.col("model_version") == version).limit(20))

# COMMAND ----------

dbutils.notebook.exit(json.dumps({"rows": summary.n, "accuracy": summary.accuracy, "model_version": version}))
