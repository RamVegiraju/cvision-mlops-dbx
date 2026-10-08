"""Dataset acquisition, manifest building, validation and the torch Dataset used for training."""

from __future__ import annotations

import hashlib
import io
import os
import shutil
import tarfile
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from PIL import Image

from cv_mlops import config


def download_and_extract(dest_root: str, url: str = config.IMAGENETTE_URL) -> Path:
    """Download the Imagenette archive and extract it under ``dest_root``. Idempotent.

    Extraction happens on local disk first and is then copied to ``dest_root`` (typically a UC
    Volume): many small FUSE writes are far slower than one local extract + bulk copy.
    """
    dest = Path(dest_root) / config.IMAGENETTE_DIRNAME
    marker = dest / "_SUCCESS"
    if marker.exists():
        return dest

    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "data.tgz"
        urllib.request.urlretrieve(url, archive)
        with tarfile.open(archive) as tar:
            tar.extractall(tmp, filter="data")
        if dest.exists():  # partial copy from an earlier failed attempt
            shutil.rmtree(dest)
        _parallel_copytree(Path(tmp) / config.IMAGENETTE_DIRNAME, dest)
    marker.touch()
    return dest


def _parallel_copytree(src: Path, dest: Path, workers: int = 32) -> None:
    """copytree with threaded file copies — Volume FUSE writes are latency-bound, not bandwidth-bound."""
    files = [p for p in src.rglob("*") if p.is_file()]
    for d in {dest / f.relative_to(src).parent for f in files}:
        d.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(lambda f: shutil.copyfile(f, dest / f.relative_to(src)), files))


def _split_for(path: str, source_split: str) -> str:
    """Imagenette 'val' becomes our held-out test set; 'train' is split train/val by a stable hash."""
    if source_split == "val":
        return "test"
    bucket = int(hashlib.md5(os.path.basename(path).encode()).hexdigest(), 16) % 1000
    return "val" if bucket < config.VAL_FRACTION * 1000 else "train"


def build_manifest(dataset_root: str | Path, max_per_class: int = 0) -> pd.DataFrame:
    """One row per image: path, label, label_idx, split.

    ``max_per_class`` > 0 caps images per (split, class) — used for fast smoke runs. The cap is
    deterministic (sorted by path) so repeated smoke runs see the same subset.
    """
    root = Path(dataset_root)
    rows = []
    for source_split in ("train", "val"):
        for synset, label in config.SYNSET_TO_LABEL.items():
            class_dir = root / source_split / synset
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Missing class folder: {class_dir}")
            for f in sorted(class_dir.iterdir()):
                if f.suffix.lower() in (".jpeg", ".jpg", ".png"):
                    rows.append(
                        {
                            "path": str(f),
                            "label": label,
                            "label_idx": config.LABEL_TO_IDX[label],
                            "split": _split_for(str(f), source_split),
                        }
                    )
    df = pd.DataFrame(rows)
    if max_per_class > 0:
        df = (
            df.sort_values("path")
            .groupby(["split", "label"], group_keys=False)
            .head(max_per_class)
            .reset_index(drop=True)
        )
    return df


def _inspect_image(path: str) -> tuple[int | None, int | None, bool]:
    try:
        with Image.open(path) as im:
            im.verify()
        with Image.open(path) as im:
            w, h = im.size
        return w, h, True
    except Exception:
        return None, None, False


def validate_images(df: pd.DataFrame, workers: int = 32) -> pd.DataFrame:
    """Add ``is_valid``, ``width``, ``height`` by verifying each image (threaded; I/O bound on Volumes)."""
    with ThreadPoolExecutor(workers) as pool:
        widths, heights, valid = zip(*pool.map(_inspect_image, df["path"])) if len(df) else ((), (), ())
    return df.assign(width=list(widths), height=list(heights), is_valid=list(valid))


def check_manifest(df: pd.DataFrame, max_invalid_fraction: float = 0.01) -> dict:
    """Fail fast on data problems before any GPU time is spent. Returns summary stats."""
    problems = []
    invalid_frac = 1 - df["is_valid"].mean()
    if invalid_frac > max_invalid_fraction:
        problems.append(f"{invalid_frac:.2%} invalid images (limit {max_invalid_fraction:.0%})")
    for split in config.SPLITS:
        present = set(df.loc[(df["split"] == split) & df["is_valid"], "label"])
        missing = set(config.CLASS_NAMES) - present
        if missing:
            problems.append(f"split '{split}' missing classes: {sorted(missing)}")
    if problems:
        raise ValueError("Data quality checks failed: " + "; ".join(problems))

    counts = df[df["is_valid"]].groupby("split").size().to_dict()
    per_class = df[(df["split"] == "train") & df["is_valid"]].groupby("label").size()
    return {
        "n_images": int(len(df)),
        "n_invalid": int((~df["is_valid"]).sum()),
        **{f"n_{k}": int(v) for k, v in counts.items()},
        "train_class_imbalance_ratio": float(per_class.max() / per_class.min()),
    }


def manifest_digest(df: pd.DataFrame, split: str) -> str:
    """Stable fingerprint of a split's file list — used to tell whether two evals are comparable."""
    paths = "\n".join(sorted(os.path.basename(p) for p in df.loc[df["split"] == split, "path"]))
    return hashlib.sha256(paths.encode()).hexdigest()[:16]


def load_rgb(source) -> Image.Image:
    """Open a path or raw bytes as an RGB PIL image (Imagenette has a few grayscale/CMYK files)."""
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    with Image.open(source) as im:
        return im.convert("RGB")


try:
    import torch
    from torch.utils.data import Dataset

    class ManifestDataset(Dataset):
        """Reads images straight from a UC Volume path listed in the manifest."""

        def __init__(self, df: pd.DataFrame, transform):
            self.paths = df["path"].tolist()
            self.labels = df["label_idx"].astype(int).tolist()
            self.transform = transform

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, i):
            return self.transform(load_rgb(self.paths[i])), torch.tensor(self.labels[i])

except ImportError:  # data prep runs fine without torch installed
    pass
