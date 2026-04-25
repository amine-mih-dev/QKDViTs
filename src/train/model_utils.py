from __future__ import annotations

import os
from typing import Any

import torch


def count_parameters(model: torch.nn.Module) -> dict[str, int]:
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    return {"total": total, "trainable": trainable}


def model_size_mb(model: torch.nn.Module) -> float:
    pid = os.getpid()
    temp_file = f"temp_model_state_{pid}.pt"
    try:
        torch.save(model.state_dict(), temp_file)
        return os.path.getsize(temp_file) / (1024 * 1024)
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


def summarize_model(model: torch.nn.Module, model_name: str = "model") -> dict[str, Any]:
    counts = count_parameters(model)
    size_mb = model_size_mb(model)
    return {
        "name": model_name,
        "total_params": counts["total"],
        "trainable_params": counts["trainable"],
        "size_mb": size_mb,
    }
