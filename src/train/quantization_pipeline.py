from __future__ import annotations

import ast
import copy
import re
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, auc, f1_score, precision_score, recall_score, roc_curve
from sklearn.preprocessing import label_binarize
from torch.utils.data import DataLoader
from torchvision import datasets

from src.core.plotting import STYLE_FIGSIZE, STYLE_LABEL_FONTSIZE, STYLE_LEGEND_FONTSIZE, STYLE_LINEWIDTH, STYLE_TICK_FONTSIZE

from .config import DEFAULT_TEACHER_BACKBONES
from .data_module import build_eval_transform
from .lightning_modules import DistillerTrainer, StandardModel
from .model_factory import build_distillable_vit, get_model_by_name, student_short_name


@dataclass
class QuantizationRunConfig:
    dataset_name: str
    run_timestamp: str
    test_dir: str
    checkpoints_dir: str = "checkpoints"
    qres_dir: str = "qres"
    aucs_dir: str = "aucs"
    visualization_dir: str = "visualizations"
    batch_size: int = 32
    num_workers: int = 4
    image_size: int = 224
    eval_device: str = "cpu"


@dataclass
class ModelRecord:
    output_model_id: str
    phase: str
    checkpoint_path: Path
    wrapper: str
    teacher_id: str | None = None
    student_id: str | None = None


def _path_priority(path: Path) -> tuple[int, float]:
    suffix_score = 3 if path.suffix.lower() == ".ckpt" else 2
    best_score = 1 if "best" in path.name.lower() else 0
    return (suffix_score + best_score, path.stat().st_mtime)


def _infer_teacher_from_name(name: str) -> str | None:
    match = re.search(r"T\d+_[A-Za-z0-9]+_GMN", name)
    if match:
        return match.group(0)
    return None


def _parse_model_record(path: Path, base_dir: Path) -> ModelRecord | None:
    try:
        parts = path.relative_to(base_dir).parts
    except ValueError:
        return None

    if len(parts) < 2:
        return None

    phase = parts[0]
    suffix = path.suffix.lower()

    if phase == "teacher":
        teacher_id = parts[1]
        return ModelRecord(
            output_model_id=teacher_id,
            phase="teacher",
            checkpoint_path=path,
            wrapper="StandardModel" if suffix == ".ckpt" else "raw",
            teacher_id=teacher_id,
        )

    if phase == "student_indep":
        student_id = parts[1]
        short_name = student_short_name(student_id)
        model_id = f"{short_name}_Independent"
        return ModelRecord(
            output_model_id=model_id,
            phase="student_indep",
            checkpoint_path=path,
            wrapper="StandardModel" if suffix == ".ckpt" else "raw",
            student_id=student_id,
        )

    if phase.startswith("distilled_"):
        student_id = phase.replace("distilled_", "", 1)
        teacher_id = None
        if len(parts) >= 2:
            teacher_id = parts[1].replace("with_T_", "")
        teacher_id = teacher_id or _infer_teacher_from_name(path.name)
        if not teacher_id:
            return None

        short_name = student_short_name(student_id)
        model_id = f"{short_name}_Distilled_from_{teacher_id}"
        return ModelRecord(
            output_model_id=model_id,
            phase="distilled",
            checkpoint_path=path,
            wrapper="DistillerTrainer" if suffix == ".ckpt" else "raw",
            teacher_id=teacher_id,
            student_id=student_id,
        )

    return None


def discover_model_records(config: QuantizationRunConfig) -> list[ModelRecord]:
    base_dir = (
        Path(config.checkpoints_dir) / config.dataset_name / config.run_timestamp
    )
    if not base_dir.exists():
        raise FileNotFoundError(f"Checkpoint run directory not found: {base_dir}")

    candidates = list(base_dir.rglob("*.ckpt")) + list(base_dir.rglob("*.pth"))
    chosen: dict[str, ModelRecord] = {}

    for candidate in candidates:
        record = _parse_model_record(candidate, base_dir)
        if record is None:
            continue

        existing = chosen.get(record.output_model_id)
        if existing is None:
            chosen[record.output_model_id] = record
            continue

        if _path_priority(record.checkpoint_path) > _path_priority(existing.checkpoint_path):
            chosen[record.output_model_id] = record

    return sorted(chosen.values(), key=lambda item: item.output_model_id)


def _build_teacher_model(teacher_id: str, num_classes: int) -> nn.Module:
    backbone = DEFAULT_TEACHER_BACKBONES.get(teacher_id)
    if not backbone:
        raise ValueError(
            f"Teacher backbone for '{teacher_id}' not found in DEFAULT_TEACHER_BACKBONES"
        )
    model, _ = get_model_by_name(
        backbone_name_str=backbone,
        num_classes=num_classes,
        timm_pretrained_flag=False,
        actor_name=f"Teacher<{teacher_id}>",
    )
    return model


def _build_student_model(student_id: str, num_classes: int) -> nn.Module:
    return build_distillable_vit(student_id, num_classes)


def _clean_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        normalized = key
        for prefix in ("module.", "model.", "student."):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :]
        cleaned[normalized] = value
    return cleaned


def load_model_from_record(record: ModelRecord, num_classes: int) -> nn.Module:
    suffix = record.checkpoint_path.suffix.lower()

    if record.phase == "teacher":
        if not record.teacher_id:
            raise ValueError(f"Missing teacher_id for {record.checkpoint_path}")
        base_model = _build_teacher_model(record.teacher_id, num_classes)
    else:
        if not record.student_id:
            raise ValueError(f"Missing student_id for {record.checkpoint_path}")
        base_model = _build_student_model(record.student_id, num_classes)

    if suffix == ".ckpt":
        if record.wrapper == "StandardModel":
            module = StandardModel.load_from_checkpoint(
                str(record.checkpoint_path),
                map_location="cpu",
                model_or_backbone_name=base_model,
            )
            return module.model

        if record.wrapper == "DistillerTrainer":
            if not record.teacher_id:
                raise ValueError(f"Missing teacher_id for distilled record {record}")
            teacher_dummy = _build_teacher_model(record.teacher_id, num_classes)
            module = DistillerTrainer.load_from_checkpoint(
                str(record.checkpoint_path),
                map_location="cpu",
                teacher_model=teacher_dummy,
                student_model=base_model,
            )
            return module.student

    state_blob = torch.load(
        str(record.checkpoint_path),
        map_location="cpu",
        weights_only=False,
    )
    if isinstance(state_blob, dict) and "state_dict" in state_blob:
        state_blob = state_blob["state_dict"]

    if not isinstance(state_blob, dict):
        raise TypeError(
            f"Unsupported state format in {record.checkpoint_path}: {type(state_blob)}"
        )

    cleaned = _clean_state_dict_keys(state_blob)
    base_model.load_state_dict(cleaned, strict=False)
    return base_model


def dynamic_quantize_model(model: nn.Module) -> nn.Module:
    model_cpu = copy.deepcopy(model).to("cpu").eval()
    return torch.ao.quantization.quantize_dynamic(
        model_cpu,
        {nn.Linear},
        dtype=torch.qint8,
    )


def _extract_logits(output):
    if hasattr(output, "logits"):
        return output.logits
    return output


def evaluate_model_with_scores(
    model: nn.Module,
    loader: DataLoader,
    num_classes: int,
    eval_device: str,
) -> dict[str, object]:
    model = model.to(eval_device)
    model.eval()
    criterion = nn.CrossEntropyLoss()

    all_labels: list[np.ndarray] = []
    all_scores: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []

    total_loss = 0.0
    timings_ms: list[float] = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(eval_device)
            labels = labels.to(eval_device)

            if str(eval_device).startswith("cuda"):
                torch.cuda.synchronize()
            start = time.perf_counter()
            outputs = _extract_logits(model(images))
            if str(eval_device).startswith("cuda"):
                torch.cuda.synchronize()
            end = time.perf_counter()
            timings_ms.append((end - start) * 1000)

            loss = criterion(outputs, labels)
            total_loss += float(loss.item()) * images.size(0)

            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(outputs, dim=1)

            all_labels.append(labels.detach().cpu().numpy())
            all_scores.append(probs.detach().cpu().numpy())
            all_predictions.append(preds.detach().cpu().numpy())

    y_true = np.concatenate(all_labels) if all_labels else np.array([])
    y_scores = np.concatenate(all_scores) if all_scores else np.empty((0, num_classes))
    y_pred = np.concatenate(all_predictions) if all_predictions else np.array([])

    if len(y_true) == 0:
        return {
            "loss": float("nan"),
            "accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1_score": float("nan"),
            "timing": float("nan"),
            "labels": y_true,
            "scores": y_scores,
        }

    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "timing": float(sum(timings_ms) / max(1, len(timings_ms))),
        "labels": y_true,
        "scores": y_scores,
    }


def _serialize_curve_map(curve_map: dict[object, np.ndarray]) -> dict[object, list[float]]:
    serialized: dict[object, list[float]] = {}
    for key, value in curve_map.items():
        serialized[key] = np.asarray(value, dtype=float).tolist()
    return serialized


def generate_and_save_auc_visualization(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    class_names: list[str],
    model_id: str,
    output_dir: Path,
) -> tuple[float, dict[object, list[float]], dict[object, list[float]]]:
    n_classes = len(class_names)
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))

    fpr: dict[object, np.ndarray] = {}
    tpr: dict[object, np.ndarray] = {}
    class_auc: dict[int, float] = {}

    for class_idx in range(n_classes):
        class_fpr, class_tpr, _ = roc_curve(
            y_true_bin[:, class_idx],
            y_scores[:, class_idx],
            drop_intermediate=False,
        )
        fpr[class_idx] = class_fpr
        tpr[class_idx] = class_tpr
        class_auc[class_idx] = float(auc(class_fpr, class_tpr))

    micro_fpr, micro_tpr, _ = roc_curve(
        y_true_bin.ravel(),
        y_scores.ravel(),
        drop_intermediate=False,
    )
    fpr["micro"] = micro_fpr
    tpr["micro"] = micro_tpr
    micro_auc = float(auc(micro_fpr, micro_tpr))

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_model_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", model_id)

    fig, ax = plt.subplots(figsize=STYLE_FIGSIZE)
    for class_idx in range(n_classes):
        ax.plot(
            fpr[class_idx],
            tpr[class_idx],
            linewidth=STYLE_LINEWIDTH,
            label=f"Class {class_names[class_idx]} (AUC={class_auc[class_idx]:.3f})",
        )

    ax.plot(
        fpr["micro"],
        tpr["micro"],
        linestyle=":",
        linewidth=STYLE_LINEWIDTH,
        color="black",
        label=f"Micro-average (AUC={micro_auc:.3f})",
    )
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Chance")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate", fontsize=STYLE_LABEL_FONTSIZE)
    ax.set_ylabel("True Positive Rate", fontsize=STYLE_LABEL_FONTSIZE)
    # ax.set_title(f"ROC Curves: {model_id}", fontsize=STYLE_LABEL_FONTSIZE)
    ax.tick_params(axis="both", which="major", labelsize=STYLE_TICK_FONTSIZE)
    ax.legend(loc="lower right", fontsize=STYLE_LEGEND_FONTSIZE)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / f"roc_auc_{safe_model_id}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return micro_auc, _serialize_curve_map(tpr), _serialize_curve_map(fpr)


def _build_test_loader(config: QuantizationRunConfig) -> tuple[DataLoader, int, list[str]]:
    dataset = datasets.ImageFolder(
        config.test_dir,
        transform=build_eval_transform(config.image_size),
    )
    use_pin_memory = torch.cuda.is_available() and str(config.eval_device).lower().startswith("cuda")
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=config.num_workers > 0,
    )
    return loader, len(dataset.classes), list(dataset.classes)


def run_quantization_sweep(config: QuantizationRunConfig) -> tuple[Path, Path, Path]:
    test_loader, num_classes, class_names = _build_test_loader(config)
    records = discover_model_records(config)
    if not records:
        raise FileNotFoundError(
            "No checkpoint records discovered for quantization sweep."
        )

    output_qres_dir = Path(config.qres_dir) / config.dataset_name / config.run_timestamp
    output_aucs_dir = Path(config.aucs_dir) / config.dataset_name / config.run_timestamp
    output_viz_dir = (
        Path(config.visualization_dir)
        / config.dataset_name
        / config.run_timestamp
        / "AUCs"
    )
    output_qres_dir.mkdir(parents=True, exist_ok=True)
    output_aucs_dir.mkdir(parents=True, exist_ok=True)
    output_viz_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, object]] = []
    auc_rows: list[dict[str, object]] = []

    for record in records:
        print(f"\n--- Quantization sweep: {record.output_model_id} ---")
        model_fp32 = load_model_from_record(record, num_classes)

        fp32_metrics = evaluate_model_with_scores(
            model_fp32,
            test_loader,
            num_classes,
            config.eval_device,
        )
        y_true = fp32_metrics.pop("labels")
        fp32_scores = fp32_metrics.pop("scores")

        fp32_micro_auc, fp32_tpr, fp32_fpr = generate_and_save_auc_visualization(
            y_true=y_true,
            y_scores=fp32_scores,
            class_names=class_names,
            model_id=f"{record.output_model_id}_FP32",
            output_dir=output_viz_dir,
        )

        model_quantized = dynamic_quantize_model(model_fp32)
        quant_metrics = evaluate_model_with_scores(
            model_quantized,
            test_loader,
            num_classes,
            "cpu",
        )
        quant_scores = quant_metrics.pop("scores")
        _ = quant_metrics.pop("labels")

        quant_micro_auc, quant_tpr, quant_fpr = generate_and_save_auc_visualization(
            y_true=y_true,
            y_scores=quant_scores,
            class_names=class_names,
            model_id=f"{record.output_model_id}_INT8",
            output_dir=output_viz_dir,
        )

        row = {"model_id": record.output_model_id}
        for key, value in fp32_metrics.items():
            row[f"fp32_{key}"] = value
        for key, value in quant_metrics.items():
            row[f"quantized_{key}"] = value
        summary_rows.append(row)

        auc_rows.append(
            {
                "model_id": record.output_model_id,
                "dataset": config.dataset_name,
                "fp32_micro_avg_auc": fp32_micro_auc,
                "quantized_micro_avg_auc": quant_micro_auc,
                "fp32_tpr": fp32_tpr,
                "fp32_fpr": fp32_fpr,
                "quantized_tpr": quant_tpr,
                "quantized_fpr": quant_fpr,
            }
        )

    qres_path = output_qres_dir / "results_quantization_comparison.csv"
    auc_csv_path = output_aucs_dir / "results_micro_average_auc.csv"
    auc_pkl_path = output_aucs_dir / "results_micro_average_auc.pkl"

    pd.DataFrame(summary_rows).to_csv(qres_path, index=False)
    auc_frame = pd.DataFrame(auc_rows)
    auc_frame.to_csv(auc_csv_path, index=False)
    auc_frame.to_pickle(auc_pkl_path)

    return qres_path, auc_csv_path, auc_pkl_path


def read_auc_table(path: str | Path) -> pd.DataFrame:
    target = Path(path)
    if target.suffix.lower() in {".pkl", ".pickle"}:
        return pd.read_pickle(target)

    frame = pd.read_csv(target)
    for column in ("fp32_tpr", "fp32_fpr", "quantized_tpr", "quantized_fpr"):
        if column in frame.columns:
            frame[column] = frame[column].map(
                lambda value: ast.literal_eval(value)
                if isinstance(value, str)
                else value
            )
    return frame
