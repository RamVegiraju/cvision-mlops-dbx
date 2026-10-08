"""Spark-side table access shared by the job notebooks."""

from __future__ import annotations

import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from cv_mlops import data, serving


def tables(catalog: str, schema: str) -> dict[str, str]:
    return {
        "bronze": f"{catalog}.{schema}.bronze_images",
        "silver": f"{catalog}.{schema}.silver_image_manifest",
        "gold": f"{catalog}.{schema}.gold_predictions",
    }


def images_with_bytes(spark: SparkSession, catalog: str, schema: str, split: str) -> DataFrame:
    """Valid images of one split joined with their raw bytes, as a base64 ``image`` column."""
    t = tables(catalog, schema)
    silver = spark.table(t["silver"]).where(F.col("is_valid") & (F.col("split") == split))
    bronze = spark.table(t["bronze"]).select("path", "content")
    return silver.join(bronze, "path").select(
        "path", "label", F.base64("content").alias(serving.INPUT_COLUMN)
    )


def load_test_set(spark: SparkSession, catalog: str, schema: str) -> tuple[pd.DataFrame, str]:
    """Test split as pandas (path, label, image) + its digest, in a stable order."""
    pdf = images_with_bytes(spark, catalog, schema, "test").orderBy("path").toPandas()
    return pdf, data.manifest_digest(pdf.assign(split="test"), "test")
