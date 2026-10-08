# Computer Vision MLOps on Databricks

Image classification end to end on Databricks: data prep → two models trained in parallel on serverless GPUs →
evaluation → best-model selection → batch inference and a real-time serving endpoint, in parallel. Deployed as a
[Declarative Automation Bundle](https://docs.databricks.com/aws/en/dev-tools/bundles) (formerly Databricks Asset Bundles)
and run as one [Lakeflow Job](https://docs.databricks.com/aws/en/jobs), with MLflow and Unity Catalog tracking every
model.

![Architecture](docs/architecture.svg)

- **Data:** [Imagenette](https://github.com/fastai/imagenette), 10 ImageNet classes, ~13k images
- **Models:** ImageNet-pretrained `mobilenet_v3_large` and `efficientnet_b0`, fine-tuned (`resnet50` and `vgg16` also supported)

## Quick start

### 1. Prerequisites

- A workspace with Unity Catalog, [serverless compute for jobs](https://docs.databricks.com/aws/en/jobs/compute), and
  [AI Runtime](https://docs.databricks.com/aws/en/machine-learning/ai-runtime) (serverless GPU, Public Preview) with
  the Databricks AI environment enabled
- `USE CATALOG` and `CREATE SCHEMA` on an existing catalog, plus permission to create serving endpoints
- Outbound internet from serverless to `s3.amazonaws.com`, `download.pytorch.org` and `pypi.org`
- [Databricks CLI](https://docs.databricks.com/aws/en/dev-tools/cli/install) (tested with v1.17.0)

### 2. Set up

```bash
git clone https://github.com/RamVegiraju/cvision-mlops-dbx.git && cd cvision-mlops-dbx
databricks auth login --host https://<your-workspace-url> --profile my-ws

export DATABRICKS_CONFIG_PROFILE=my-ws
export BUNDLE_VAR_catalog=<your_catalog>
```

### 3. Deploy

```bash
databricks bundle validate --strict -t dev
databricks bundle deploy -t dev
```

This creates the job `[dev] cv-mlops-training-pipeline`, plus the schema `<catalog>.cv_mlops_demo` with its
volume, registered model and MLflow experiment.

### 4. Run

```bash
# Smoke test on a small subset (~15 min)
databricks bundle run cv_training_pipeline -t dev --params max_per_class=20,epochs=1

# Full run (~30 min the first time)
databricks bundle run cv_training_pipeline -t dev
```

You can also start a run from the UI: *Jobs & Pipelines* → `[dev] cv-mlops-training-pipeline` → **Run now**.
To re-run only the failed tasks of a run: `databricks jobs repair-run <run_id> --rerun-all-failed-tasks`.

### 5. Query the endpoint

```bash
pip install databricks-sdk
python scripts/query_endpoint.py --profile my-ws --endpoint cv-image-classifier-dev path/to/image.jpg
```

Or call it over REST (the input is a base64-encoded image):

```bash
curl -X POST "https://<your-workspace-url>/serving-endpoints/cv-image-classifier-dev/invocations" \
  -H "Authorization: Bearer $DATABRICKS_TOKEN" -H "Content-Type: application/json" \
  -d "{\"dataframe_records\": [{\"image\": \"$(base64 -i image.jpg)\"}]}"   # Linux: base64 -w0
```

### 6. Clean up

```bash
databricks serving-endpoints delete cv-image-classifier-dev
databricks bundle destroy -t dev
```

### Configuration

Set with `--var name=value` on `validate` and `deploy`, or as `BUNDLE_VAR_<name>`.

| Variable | Default | Description |
|---|---|---|
| `catalog` | *(required)* | Existing catalog to create the schema in |
| `schema` | `cv_mlops_demo` | Schema for all tables, the volume and the model |
| `gpu_type` | `GPU_1xA10` | Serverless GPU for each training task |
| `backbone_a`, `backbone_b` | `mobilenet_v3_large`, `efficientnet_b0` | The two models trained in parallel |
| `epochs` | `5` | Training epochs |
| `max_per_class` | `0` (all) | Images per class and split; use a small number for quick tests |
| `accuracy_tolerance` | `0.005` | If both models are within this accuracy, the faster one wins |
| `endpoint_name` | `cv-image-classifier-dev` | Serving endpoint name |

To run the job on a schedule or when new files arrive, add a
[trigger](https://docs.databricks.com/aws/en/jobs/triggers) to `resources/cv_training_pipeline.job.yml` and redeploy.

## Pipeline

| Task | What it does | Compute |
|---|---|---|
| `prepare_data` | Lands images in a volume, builds the bronze and silver tables with train/val/test splits, and runs data-quality checks | Serverless CPU |
| `train_model_a` ∥ `train_model_b` | Fine-tunes each model, logs to MLflow, and registers a model version in Unity Catalog | Serverless GPU (one A10 each) |
| `eval_model_a` ∥ `eval_model_b` | Scores the test set: accuracy, top-3, F1, confusion matrix, CPU latency | Serverless CPU |
| `select_best` | Picks the winner (best accuracy, then speed) and sets the `@champion` alias | Serverless CPU |
| `batch_inference` ∥ `deploy_endpoint` | Writes predictions to `gold_predictions` ∥ creates or updates the serving endpoint and tests it | Serverless CPU |

## Files

| File | Purpose |
|---|---|
| `databricks.yml` | Bundle entry point: variables, `dev`/`prod` targets, what to sync. The CLI reads this file. |
| `resources/cv_training_pipeline.job.yml` | The Lakeflow Job: tasks, dependencies, GPU/CPU environments, parameters |
| `resources/cv_mlops.uc.yml` | Unity Catalog schema, volume and registered model, plus the MLflow experiment |
| `src/notebooks/01_prepare_data.py` | Task `prepare_data` |
| `src/notebooks/02_train.py` | Tasks `train_model_a` and `train_model_b` (same notebook, different `backbone`) |
| `src/notebooks/03_evaluate.py` | Tasks `eval_model_a` and `eval_model_b` |
| `src/notebooks/04_select_best.py` | Task `select_best` |
| `src/notebooks/05_batch_inference.py` | Task `batch_inference` |
| `src/notebooks/06_deploy_endpoint.py` | Task `deploy_endpoint` |
| `src/cv_mlops/config.py` | Class labels and image preprocessing constants |
| `src/cv_mlops/data.py` | Dataset download, manifest and splits, data-quality checks, PyTorch dataset |
| `src/cv_mlops/modeling.py` | Supported models, image transforms, training loop |
| `src/cv_mlops/serving.py` | MLflow model wrapper: base64 image in, label and confidence out, preprocessing included |
| `src/cv_mlops/evaluation.py` | Metrics, confusion matrix, latency benchmark, winner selection |
| `src/cv_mlops/pipeline.py` | Evaluates a registered model version, reads candidates, sets `@champion` |
| `src/cv_mlops/spark_io.py` | Table names and shared Spark reads |
| `scripts/query_endpoint.py` | Sends local images to the endpoint and prints predictions |
| `tests/` | Unit tests and a local end-to-end test (`pytest`) |
| `docs/architecture.svg` | Architecture diagram |

## Artifacts

Everything the pipeline creates in `<catalog>.cv_mlops_demo`, plus the MLflow experiment and the endpoint:

| Artifact | Type | Contents |
|---|---|---|
| `raw_images` | Volume | `imagenette2-160/` (images); `torch_hub/` (cached pretrained weights) |
| `bronze_images` | Delta table | Raw image bytes: `path`, `content`, `length`, `modificationTime`, `ingested_at` |
| `silver_image_manifest` | Delta table | `path`, `label`, `label_idx`, `split`, `width`, `height`, `is_valid` |
| `gold_predictions` | Delta table | `path`, `true_label`, `predicted_label`, `confidence`, `top_k`, `model_version`, `scored_at` |
| `image_classifier` | Registered model | One version per trained model. Aliases: `@champion` (served model), `@candidate_<backbone>`. Version tags hold the test metrics. |
| Model version files | Model artifact | `artifacts/weights.pt`, `artifacts/model_config.json`, `code/cv_mlops/`, `MLmodel`, `requirements.txt`, input examples |
| `cv_mlops_demo_dev` | MLflow experiment | Per training run: params, per-epoch metrics, GPU system metrics, dataset lineage, `eval/confusion_matrix.png`, `eval/per_class_metrics.json`, `eval/misclassified.json`. Per `select-best` run: `selection/comparison.json`, `selection/decision.txt`. |
| `cv-image-classifier-dev` | Serving endpoint | CPU, Small, scale-to-zero, serving `@champion` |
| `cv_endpoint_payload` | Delta table | Endpoint request and response log (AI Gateway inference table; rows can take up to an hour to appear) |

## Local tests

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch==2.11.0 torchvision==0.26.0 mlflow==3.13.0 \
  pillow==12.0.0 numpy==2.3.4 pandas==2.3.3 scikit-learn==1.7.2 matplotlib pytest databricks-sdk
PYTHONPATH=src .venv/bin/python -c "from cv_mlops.data import download_and_extract; download_and_extract('scratch/data')"
.venv/bin/python -m pytest -q
```
