from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import torch


@dataclass
class DataConfig:
    train_dir: str
    val_dir: str
    test_dir: str
    batch_size: int = 16
    num_workers: int = 8
    image_size: int = 224


@dataclass
class RuntimeConfig:
    accelerator: str = field(
        default_factory=lambda: "gpu" if torch.cuda.is_available() else "cpu"
    )
    devices: int = 1
    precision: str | int = field(
        default_factory=lambda: "16-mixed" if torch.cuda.is_available() else 32
    )


@dataclass
class OptimConfig:
    learning_rate: float
    epochs: int
    max_epochs_for_scheduler: int
    weight_decay: float = 0.0


@dataclass
class TeacherSpec:
    id: str
    backbone_name: str
    pretrained: bool
    training: OptimConfig
    pretrained_weights_path: str | None = None


@dataclass
class StudentSpec:
    id: str
    init_type: str
    training: OptimConfig
    backbone_name: str | None = None
    params: dict[str, object] | None = None
    pretrained: bool = True
    pretrained_weights_path: str | None = None


@dataclass
class DistillationConfig:
    learning_rate: float
    weight_decay: float
    epochs: int
    max_epochs_for_scheduler: int
    temperature: float
    alpha: float


@dataclass
class ExperimentConfig:
    dataset_name: str
    run_timestamp: str
    data: DataConfig
    runtime: RuntimeConfig
    results_log_file: str
    teachers: list[TeacherSpec]
    students: list[StudentSpec]
    distillation: DistillationConfig
    seed: int = 42
    accuracy_task: str = "multiclass"
    checkpoints_dir: str = "checkpoints"
    logs_dir: str = "logs"


DEFAULT_TEACHER_BACKBONES = {
    "T1_ResNet101_GMN": "resnet101",
    "T2_AlexNet_GMN": "alexnet",
    "T4_VGG19_GMN": "vgg19",
}


DEFAULT_VIT_ARCHITECTURES = {
    "S1_Green": {"dim": 768, "depth": 8, "heads": 4, "mlp_dim": 1024},
    "S2_Small": {"dim": 384, "depth": 8, "heads": 12, "mlp_dim": 1536},
    "S3_Base": {"dim": 768, "depth": 12, "heads": 12, "mlp_dim": 3072},
}


def ensure_run_timestamp(run_timestamp: str | None = None) -> str:
    if run_timestamp:
        return run_timestamp
    return datetime.now().strftime("%Y-%m-%d-%H-%M")


def build_results_log_file(results_dir: str, dataset_name: str, run_timestamp: str) -> str:
    filename = f"experiment_results_get_model_by_name{run_timestamp}.csv"
    return str(Path(results_dir) / dataset_name / run_timestamp / filename)


def default_teacher_specs(teacher_epochs: int = 40) -> list[TeacherSpec]:
    scheduler_epochs = max(1, teacher_epochs - 1)
    return [
        TeacherSpec(
            id="T1_ResNet101_GMN",
            backbone_name="resnet101",
            pretrained=True,
            training=OptimConfig(
                learning_rate=1e-3,
                epochs=teacher_epochs,
                max_epochs_for_scheduler=scheduler_epochs,
                weight_decay=1e-3,
            ),
        ),
        TeacherSpec(
            id="T2_AlexNet_GMN",
            backbone_name="alexnet",
            pretrained=True,
            training=OptimConfig(
                learning_rate=3e-4,
                epochs=teacher_epochs,
                max_epochs_for_scheduler=scheduler_epochs,
                weight_decay=1e-2,
            ),
        ),
        TeacherSpec(
            id="T4_VGG19_GMN",
            backbone_name="vgg19",
            pretrained=True,
            training=OptimConfig(
                learning_rate=3e-4,
                epochs=teacher_epochs,
                max_epochs_for_scheduler=scheduler_epochs,
                weight_decay=1e-2,
            ),
        ),
    ]


def default_student_specs(student_epochs: int = 100, dropout: float = 0.0) -> list[StudentSpec]:
    scheduler_epochs = max(1, int(student_epochs / 1.5))
    student_training = OptimConfig(
        learning_rate=1e-5,
        epochs=student_epochs,
        max_epochs_for_scheduler=scheduler_epochs,
        weight_decay=1e-2,
    )

    return [
        StudentSpec(
            id="S1_Green_DistillableViT",
            init_type="DistillableViT",
            training=student_training,
            params={
                "image_size": 224,
                "patch_size": 16,
                "dropout": dropout,
                **DEFAULT_VIT_ARCHITECTURES["S1_Green"],
            },
        ),
        StudentSpec(
            id="S2_Small_DistillableViT",
            init_type="DistillableViT",
            training=student_training,
            params={
                "image_size": 224,
                "patch_size": 16,
                "dropout": dropout,
                **DEFAULT_VIT_ARCHITECTURES["S2_Small"],
            },
        ),
        StudentSpec(
            id="S3_Base_DistillableViT",
            init_type="DistillableViT",
            training=student_training,
            params={
                "image_size": 224,
                "patch_size": 16,
                "dropout": dropout,
                **DEFAULT_VIT_ARCHITECTURES["S3_Base"],
            },
        ),
    ]


def default_distillation_config(student_epochs: int = 100) -> DistillationConfig:
    scheduler_epochs = max(1, int(student_epochs / 1.5))
    return DistillationConfig(
        learning_rate=1e-5,
        weight_decay=1e-2,
        epochs=student_epochs,
        max_epochs_for_scheduler=scheduler_epochs,
        temperature=1.0,
        alpha=0.4,
    )
