from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn as nn
import torchmetrics


def _extract_logits(output):
    if hasattr(output, "logits"):
        return output.logits
    return output


def evaluate_classifier(
    model: nn.Module,
    loader,
    num_classes: int,
    device: str = "cpu",
) -> dict[str, Any]:
    model = model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()
    metrics = torchmetrics.MetricCollection(
        {
            "accuracy": torchmetrics.Accuracy(task="multiclass", num_classes=num_classes),
            "precision": torchmetrics.Precision(task="multiclass", num_classes=num_classes, average="macro"),
            "recall": torchmetrics.Recall(task="multiclass", num_classes=num_classes, average="macro"),
            "f1_score": torchmetrics.F1Score(task="multiclass", num_classes=num_classes, average="macro"),
        }
    ).to(device)

    total_loss = 0.0
    latencies_ms: list[float] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            start = time.perf_counter()
            outputs = _extract_logits(model(images))
            end = time.perf_counter()
            latencies_ms.append((end - start) * 1000)

            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            metrics.update(outputs, labels)

    computed = metrics.compute()
    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": float(computed["accuracy"].item()),
        "precision": float(computed["precision"].item()),
        "recall": float(computed["recall"].item()),
        "f1_score": float(computed["f1_score"].item()),
        "latency_ms": float(sum(latencies_ms) / max(1, len(latencies_ms))),
    }
