# src/

| Folder | What it is | Used by |
|---|---|---|
| `notebooks/` | One notebook per Lakeflow Job task, numbered in pipeline order. Thin wrappers: read job parameters, do Spark I/O, pass task values to the next task. | `resources/cv_training_pipeline.job.yml` |
| `cv_mlops/` | Python library with all the logic. Notebooks import it, it's packaged into every registered model (MLflow `code_paths`), and the local tests cover it. | notebooks, the served model, `tests/` |

| Notebook | Job task(s) |
|---|---|
| `01_prepare_data.py` | `prepare_data` |
| `02_train.py` | `train_model_a`, `train_model_b` (same notebook, different `backbone`) |
| `03_evaluate.py` | `eval_model_a`, `eval_model_b` |
| `04_select_best.py` | `select_best` |
| `05_batch_inference.py` | `batch_inference` |
| `06_deploy_endpoint.py` | `deploy_endpoint` |

| Module | Responsibility |
|---|---|
| `config.py` | Class labels, image size, normalization constants |
| `data.py` | Dataset download, manifest + splits, data-quality checks, PyTorch dataset |
| `modeling.py` | Backbone registry (MobileNetV3, EfficientNet-B0, ResNet-50, VGG16), transforms, training loop |
| `serving.py` | MLflow pyfunc (base64 image → label, confidence, top-k) with preprocessing built in |
| `evaluation.py` | Metrics, confusion matrix, latency benchmark, winner selection |
| `pipeline.py` | Evaluate a registered version, read candidates from the registry, set `@champion` |
| `spark_io.py` | Table names and the Spark reads shared by notebooks |
