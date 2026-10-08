# Databricks notebook source
# MAGIC %md
# MAGIC # 1 · Prepare data
# MAGIC Raw images → UC Volume → **bronze** (raw bytes, Delta) → **silver** (validated manifest with splits).
# MAGIC Also caches the pretrained ImageNet weights in the Volume so GPU training never depends on internet access.
# MAGIC
# MAGIC Fails fast on data quality problems, before any GPU time is spent.

# COMMAND ----------

import json, os, sys

sys.path.insert(0, os.path.abspath(".."))  # bundle's src/ → import cv_mlops

from pyspark.sql import functions as F

from cv_mlops import config, data, modeling

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
volume = dbutils.widgets.get("volume")
max_per_class = int(dbutils.widgets.get("max_per_class"))
backbones = [b.strip() for b in dbutils.widgets.get("backbones").split(",")]

volume_root = f"/Volumes/{catalog}/{schema}/{volume}"
bronze_table = f"{catalog}.{schema}.bronze_images"
silver_table = f"{catalog}.{schema}.silver_image_manifest"
print(f"volume={volume_root} max_per_class={max_per_class} ({'SMOKE subset' if max_per_class else 'FULL'})")

# COMMAND ----------
# MAGIC %md ## Land raw images in the Volume (idempotent)

# COMMAND ----------

dataset_dir = data.download_and_extract(volume_root)
print("dataset at", dataset_dir)

# COMMAND ----------
# MAGIC %md ## Build + validate the manifest, then run quality gates

# COMMAND ----------

manifest = data.validate_images(data.build_manifest(dataset_dir, max_per_class=max_per_class))
stats = data.check_manifest(manifest)  # raises → task fails → nothing downstream runs
stats["test_digest"] = data.manifest_digest(manifest[manifest["is_valid"]], "test")  # same set evaluate scores
print(json.dumps(stats, indent=2))

# COMMAND ----------
# MAGIC %md ## Bronze: raw image bytes as a Delta table

# COMMAND ----------

bronze = (
    spark.read.format("binaryFile")
    .option("recursiveFileLookup", "true")
    .option("pathGlobFilter", "*.JPEG")
    .load(str(dataset_dir))
    .withColumn("path", F.regexp_replace("path", "^dbfs:", ""))
    .withColumn("ingested_at", F.current_timestamp())
)
manifest_paths = spark.createDataFrame(manifest[["path"]])
(
    bronze.join(manifest_paths, "path", "inner")  # only the (possibly subset) images in scope
    .write.mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(bronze_table)
)
spark.sql(f"COMMENT ON TABLE {bronze_table} IS 'Raw Imagenette image bytes landed from the {volume} volume'")

# COMMAND ----------
# MAGIC %md ## Silver: validated manifest with deterministic train / val / test splits

# COMMAND ----------

(
    spark.createDataFrame(manifest)
    .withColumn("prepared_at", F.current_timestamp())
    .write.mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(silver_table)
)
spark.sql(f"COMMENT ON TABLE {silver_table} IS 'Validated image manifest: path, label, split, quality flags'")
display(spark.table(silver_table).groupBy("split", "label").count().orderBy("split", "label"))

# COMMAND ----------
# MAGIC %md ## Cache pretrained weights in the Volume

# COMMAND ----------

cached = modeling.cache_pretrained_weights(backbones, f"{volume_root}/torch_hub")
print("cached weights:", cached)

# COMMAND ----------

dbutils.jobs.taskValues.set("test_digest", stats["test_digest"])
dbutils.jobs.taskValues.set("n_train", stats.get("n_train", 0))
dbutils.notebook.exit(json.dumps(stats))
