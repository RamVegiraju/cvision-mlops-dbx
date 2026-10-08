# Databricks notebook source
# MAGIC %md
# MAGIC # 5b · Deploy real-time endpoint (CPU)
# MAGIC Create-or-update a Model Serving endpoint to the selected (`@champion`) version — zero downtime on update (the old
# MAGIC version serves until the new one is ready). Inference tables capture every request/response to Delta.
# MAGIC Waits for **both** readiness signals, then runs a live smoke query before declaring success.

# COMMAND ----------

import json, os, sys, time

sys.path.insert(0, os.path.abspath(".."))

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound
from databricks.sdk.service.serving import (
    AiGatewayConfig,
    AiGatewayInferenceTableConfig,
    EndpointCoreConfigInput,
    EndpointTag,
    ServedEntityInput,
    ServingModelWorkloadType,
)

from cv_mlops import config, spark_io

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
model_name = dbutils.widgets.get("model_name")
version = dbutils.widgets.get("model_version")
endpoint_name = dbutils.widgets.get("endpoint_name")
workload_size = dbutils.widgets.get("workload_size")
timeout_min = int(dbutils.widgets.get("deploy_timeout_minutes"))

w = WorkspaceClient()
served = ServedEntityInput(
    entity_name=model_name,
    entity_version=version,
    workload_size=workload_size,
    workload_type=ServingModelWorkloadType.CPU,
    scale_to_zero_enabled=True,
)

# COMMAND ----------

try:
    current = w.serving_endpoints.get(endpoint_name)
except NotFound:
    current = None

if current is None:
    print(f"creating endpoint {endpoint_name} → {model_name} v{version}")
    w.serving_endpoints.create(
        name=endpoint_name,
        config=EndpointCoreConfigInput(served_entities=[served]),
        ai_gateway=AiGatewayConfig(
            inference_table_config=AiGatewayInferenceTableConfig(
                catalog_name=catalog, schema_name=schema, table_name_prefix="cv_endpoint", enabled=True
            )
        ),
        tags=[EndpointTag(key="project", value="cv-mlops-demo")],
    )
else:
    live = {(e.entity_name, e.entity_version) for e in (current.config.served_entities or [])} if current.config else set()
    if (model_name, version) in live:
        print(f"endpoint already serving v{version}; nothing to update")
    else:
        print(f"updating endpoint {endpoint_name}: {live} → v{version}")
        w.serving_endpoints.update_config(endpoint_name, served_entities=[served])

# COMMAND ----------
# MAGIC %md ## Wait for ready (`state.ready == READY` **and** `state.config_update == NOT_UPDATING`)

# COMMAND ----------

deadline = time.time() + timeout_min * 60
while True:
    st = w.serving_endpoints.get(endpoint_name).state
    ready, upd = st.ready.value if st.ready else None, st.config_update.value if st.config_update else None
    print(time.strftime("%H:%M:%S"), ready, upd)
    if upd == "UPDATE_FAILED":
        raise RuntimeError(f"endpoint update failed — check build logs for {endpoint_name}")
    if ready == "READY" and upd == "NOT_UPDATING":
        break
    if time.time() > deadline:
        raise TimeoutError(f"{endpoint_name} not ready after {timeout_min} min (ready={ready}, config_update={upd})")
    time.sleep(30)

entities = w.serving_endpoints.get(endpoint_name).config.served_entities
assert any(e.entity_version == version for e in entities), f"endpoint is not serving v{version}: {entities}"

# COMMAND ----------
# MAGIC %md ## Live smoke query

# COMMAND ----------

sample = spark_io.images_with_bytes(spark, catalog, schema, "test").limit(1).first()
resp = w.serving_endpoints.query(endpoint_name, dataframe_records=[{"image": sample.image}])
pred = resp.predictions[0]
print(f"true={sample.label} predicted={pred['label']} confidence={pred['confidence']:.3f}")
assert pred["label"] in config.CLASS_NAMES

dbutils.notebook.exit(json.dumps({"endpoint": endpoint_name, "model_version": version, "smoke_prediction": pred["label"]}))
