"""MLflow pyfunc wrapper: base64 image in -> label + confidence out.

Preprocessing lives inside the model, so the batch Spark UDF and the real-time endpoint call
exactly the same code path as offline evaluation — no training/serving skew.
"""

from __future__ import annotations

import base64
import importlib
import json
import os
import tempfile
from importlib.metadata import version

import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature

from cv_mlops import config

INPUT_COLUMN = "image"
TOP_K = 3
PREDICT_BATCH_SIZE = 32


_IMPORT_NAMES = {"pillow": "PIL"}


def _pkg(name: str) -> str:
    """Pin to the version installed in the training env, minus CUDA local tags (+cu130).

    Reads the module's ``__version__`` first: the AI Runtime image ships some packages (e.g. mlflow)
    without pip dist-info, so importlib.metadata alone raises PackageNotFoundError there.
    """
    try:
        v = importlib.import_module(_IMPORT_NAMES.get(name, name)).__version__
    except (ImportError, AttributeError):
        v = version(name)
    return f"{name}=={v.split('+')[0]}"


def pip_requirements(cpu_torch: bool = True) -> list[str]:
    """Requirements baked into the model for the serving container.

    ``cpu_torch`` pins the CPU-only torch wheels: the endpoint is CPU, and the default PyPI
    wheels drag in ~3 GB of CUDA libraries, slowing every endpoint build. Batch inference
    loads with env_manager="local" and ignores these, so this only affects serving.
    """
    reqs = [_pkg(p) for p in ("mlflow", "pillow", "numpy", "pandas")]
    torch_pkgs = [_pkg(p) for p in ("torch", "torchvision")]
    if cpu_torch:
        return ["--extra-index-url https://download.pytorch.org/whl/cpu", *[f"{p}+cpu" for p in torch_pkgs], *reqs]
    return [*torch_pkgs, *reqs]


class ImageClassifier(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import torch

        from cv_mlops.modeling import build_model, eval_transform

        with open(context.artifacts["model_config"]) as f:
            cfg = json.load(f)
        self.class_names = cfg["class_names"]
        self.model = build_model(cfg["backbone"], num_classes=len(self.class_names), pretrained=False)
        self.model.load_state_dict(torch.load(context.artifacts["weights"], map_location="cpu"))
        self.model.eval()
        self.transform = eval_transform()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:
        import torch

        from cv_mlops.data import load_rgb

        images = model_input[INPUT_COLUMN].tolist()
        probs = []
        with torch.inference_mode():
            for i in range(0, len(images), PREDICT_BATCH_SIZE):
                batch = torch.stack(
                    [self.transform(load_rgb(base64.b64decode(b))) for b in images[i : i + PREDICT_BATCH_SIZE]]
                ).to(self.device)
                probs.append(torch.softmax(self.model(batch).float(), dim=1).cpu().numpy())
        probs = np.concatenate(probs) if probs else np.empty((0, len(self.class_names)))

        top = np.argsort(-probs, axis=1)[:, :TOP_K]
        return pd.DataFrame(
            {
                "label": [self.class_names[t[0]] for t in top],
                "label_idx": top[:, 0].astype("int32"),
                "confidence": probs[np.arange(len(probs)), top[:, 0]].astype("float64"),
                "top_k": [
                    json.dumps({self.class_names[j]: round(float(p[j]), 4) for j in t}) for p, t in zip(probs, top)
                ],
            }
        )


def encode_image_file(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def log_model(torch_model, backbone: str, example_paths: list[str], registered_model_name: str | None = None):
    """Log the fine-tuned model as a self-contained pyfunc (weights + config + cv_mlops source).

    Returns the ``ModelInfo``; ``registered_model_version`` is set when ``registered_model_name`` is given.
    """
    import torch

    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    with tempfile.TemporaryDirectory() as tmp:
        weights = os.path.join(tmp, "weights.pt")
        torch.save({k: v.cpu() for k, v in torch_model.state_dict().items()}, weights)
        cfg_path = os.path.join(tmp, "model_config.json")
        with open(cfg_path, "w") as f:
            json.dump({"backbone": backbone, "class_names": config.CLASS_NAMES}, f)

        input_example = pd.DataFrame({INPUT_COLUMN: [encode_image_file(p) for p in example_paths]})
        # Signature from a real forward pass through the wrapper (also a load smoke test).
        wrapper = ImageClassifier()
        wrapper.load_context(type("Ctx", (), {"artifacts": {"weights": weights, "model_config": cfg_path}})())
        signature = infer_signature(input_example, wrapper.predict(None, input_example))

        return mlflow.pyfunc.log_model(
            name="model",
            python_model=ImageClassifier(),
            artifacts={"weights": weights, "model_config": cfg_path},
            code_paths=[pkg_dir],
            signature=signature,
            input_example=input_example.head(1),
            pip_requirements=pip_requirements(),
            registered_model_name=registered_model_name,
        )
