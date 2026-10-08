"""Backbone registry, transforms and the transfer-learning training loop."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn
from torchvision import models, transforms

from cv_mlops import config

# Each entry: (torchvision constructor, ImageNet weights). Swapping in the customer's VGG16 is
# just backbone="vgg16" — no code change.
BACKBONES: dict[str, tuple[Callable, object]] = {
    "mobilenet_v3_large": (models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights.IMAGENET1K_V2),
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2),
    "vgg16": (models.vgg16, models.VGG16_Weights.IMAGENET1K_V1),
}


def use_weight_cache(cache_dir: str) -> None:
    """Point torch.hub at a persistent cache (a UC Volume in the job) for pretrained weights.

    prepare_data populates it once; training then never depends on internet access to
    download.pytorch.org, and every run uses byte-identical starting weights.
    """
    os.makedirs(cache_dir, exist_ok=True)
    torch.hub.set_dir(cache_dir)


def cache_pretrained_weights(backbones: list[str], cache_dir: str) -> list[str]:
    use_weight_cache(cache_dir)
    for b in backbones:
        ctor, weights = BACKBONES[b]
        ctor(weights=weights)
    return sorted(os.listdir(os.path.join(cache_dir, "checkpoints")))


def _replace_head(model: nn.Module, num_classes: int) -> nn.Module:
    """Swap the 1000-way ImageNet head for a fresh ``num_classes`` Linear layer."""
    if hasattr(model, "fc"):  # resnet
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model.fc
    last = model.classifier[-1]  # mobilenet / efficientnet / vgg
    model.classifier[-1] = nn.Linear(last.in_features, num_classes)
    return model.classifier[-1]


def build_model(backbone: str, num_classes: int = len(config.CLASS_NAMES), pretrained: bool = True) -> nn.Module:
    """Pretrained backbone with frozen features and a trainable classifier.

    ``pretrained=False`` is used when reloading a fine-tuned state_dict at inference time, so the
    serving container never needs to download ImageNet weights.
    """
    if backbone not in BACKBONES:
        raise ValueError(f"Unknown backbone '{backbone}'. Choose from {sorted(BACKBONES)}")
    ctor, weights = BACKBONES[backbone]
    model = ctor(weights=weights if pretrained else None)
    for p in model.parameters():
        p.requires_grad = False
    # Unfreeze the whole classifier block (not just the new Linear) — cheap and worth ~1-2% accuracy.
    head_block = model.fc if hasattr(model, "fc") else model.classifier
    for p in head_block.parameters():
        p.requires_grad = True
    _replace_head(model, num_classes)
    return model


def eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(config.RESIZE_SIZE),
            transforms.CenterCrop(config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
        ]
    )


def train_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(config.IMAGE_SIZE, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
        ]
    )


@dataclass
class EpochResult:
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    seconds: float


def _run_epoch(model, loader, device, criterion, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, n = 0.0, 0, 0
    use_amp = device.type == "cuda"
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
    return total_loss / max(n, 1), correct / max(n, 1)


def train(
    model: nn.Module,
    train_loader,
    val_loader,
    epochs: int,
    lr: float,
    device: torch.device,
    on_epoch_end: Callable[[EpochResult], None] | None = None,
) -> list[EpochResult]:
    """Train only the unfrozen parameters with AdamW + cosine schedule. Mixed precision on GPU."""
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    history = []
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = _run_epoch(model, train_loader, device, criterion, optimizer, scaler)
        va_loss, va_acc = _run_epoch(model, val_loader, device, criterion)
        scheduler.step()
        result = EpochResult(epoch, tr_loss, tr_acc, va_loss, va_acc, time.time() - t0)
        history.append(result)
        if on_epoch_end:
            on_epoch_end(result)
    return history


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"params_total": total, "params_trainable": trainable}
