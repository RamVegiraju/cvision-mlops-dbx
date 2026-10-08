"""Evaluation metrics, latency benchmarking and champion/challenger selection."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from cv_mlops import config


def classification_metrics(y_true: list[str], y_pred: list[str], top_k_hit: list[bool]) -> dict:
    return {
        "test_accuracy": float(accuracy_score(y_true, y_pred)),
        "test_top3_accuracy": float(np.mean(top_k_hit)),
        "test_f1_macro": float(f1_score(y_true, y_pred, average="macro", labels=config.CLASS_NAMES, zero_division=0)),
        "test_precision_macro": float(
            precision_score(y_true, y_pred, average="macro", labels=config.CLASS_NAMES, zero_division=0)
        ),
        "test_recall_macro": float(
            recall_score(y_true, y_pred, average="macro", labels=config.CLASS_NAMES, zero_division=0)
        ),
    }


def per_class_report(y_true: list[str], y_pred: list[str]) -> pd.DataFrame:
    f1 = f1_score(y_true, y_pred, average=None, labels=config.CLASS_NAMES, zero_division=0)
    p = precision_score(y_true, y_pred, average=None, labels=config.CLASS_NAMES, zero_division=0)
    r = recall_score(y_true, y_pred, average=None, labels=config.CLASS_NAMES, zero_division=0)
    support = pd.Series(y_true).value_counts().reindex(config.CLASS_NAMES, fill_value=0).values
    return pd.DataFrame({"label": config.CLASS_NAMES, "precision": p, "recall": r, "f1": f1, "support": support})


def confusion_matrix_figure(y_true: list[str], y_pred: list[str], title: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred, labels=config.CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(config.CLASS_NAMES)), config.CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(len(config.CLASS_NAMES)), config.CLASS_NAMES)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def benchmark_latency(predict_fn, single_row_df: pd.DataFrame, n_warmup: int = 3, n_iter: int = 30) -> dict:
    """Single-image end-to-end latency (decode + preprocess + forward) — what an endpoint caller sees."""
    for _ in range(n_warmup):
        predict_fn(single_row_df)
    times = []
    for _ in range(n_iter):
        t0 = time.perf_counter()
        predict_fn(single_row_df)
        times.append((time.perf_counter() - t0) * 1000)
    return {
        "latency_p50_ms": float(np.percentile(times, 50)),
        "latency_p95_ms": float(np.percentile(times, 95)),
    }


@dataclass
class Candidate:
    version: str
    backbone: str
    test_accuracy: float
    latency_p95_ms: float
    test_digest: str


def pick_winner(candidates: list[Candidate], accuracy_tolerance: float) -> tuple[Candidate, str]:
    """Highest accuracy wins; if within ``accuracy_tolerance`` of the best, the faster model wins.

    Rationale: on CPU serving, a model that is 0.3% less accurate but 3x faster is the better product.
    """
    if not candidates:
        raise ValueError("No candidates to choose from")
    best_acc = max(c.test_accuracy for c in candidates)
    contenders = [c for c in candidates if best_acc - c.test_accuracy <= accuracy_tolerance]
    winner = min(contenders, key=lambda c: (c.latency_p95_ms, -c.test_accuracy))
    if len(contenders) > 1:
        reason = (
            f"{winner.backbone} v{winner.version} chosen: accuracy within {accuracy_tolerance:.1%} of best "
            f"({winner.test_accuracy:.4f} vs {best_acc:.4f}) and fastest p95 ({winner.latency_p95_ms:.1f} ms)"
        )
    else:
        reason = f"{winner.backbone} v{winner.version} chosen: highest test accuracy ({winner.test_accuracy:.4f})"
    return winner, reason

