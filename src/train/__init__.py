from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "DEFAULT_TEACHER_BACKBONES": (".config", "DEFAULT_TEACHER_BACKBONES"),
    "DEFAULT_VIT_ARCHITECTURES": (".config", "DEFAULT_VIT_ARCHITECTURES"),
    "DataConfig": (".config", "DataConfig"),
    "DistillationConfig": (".config", "DistillationConfig"),
    "ExperimentConfig": (".config", "ExperimentConfig"),
    "OptimConfig": (".config", "OptimConfig"),
    "RuntimeConfig": (".config", "RuntimeConfig"),
    "StudentSpec": (".config", "StudentSpec"),
    "TeacherSpec": (".config", "TeacherSpec"),
    "ImageFolderDataModule": (".data_module", "ImageFolderDataModule"),
    "evaluate_classifier": (".evaluation", "evaluate_classifier"),
    "DistillerTrainer": (".lightning_modules", "DistillerTrainer"),
    "EvaluationModule": (".lightning_modules", "EvaluationModule"),
    "StandardModel": (".lightning_modules", "StandardModel"),
    "adapt_classifier_if_needed": (".model_factory", "adapt_classifier_if_needed"),
    "build_distillable_vit": (".model_factory", "build_distillable_vit"),
    "build_student_from_spec": (".model_factory", "build_student_from_spec"),
    "build_teacher_from_spec": (".model_factory", "build_teacher_from_spec"),
    "get_model_by_name": (".model_factory", "get_model_by_name"),
    "student_short_name": (".model_factory", "student_short_name"),
    "count_parameters": (".model_utils", "count_parameters"),
    "model_size_mb": (".model_utils", "model_size_mb"),
    "summarize_model": (".model_utils", "summarize_model"),
    "run_training_experiment": (".pipeline", "run_training_experiment"),
    "QuantizationRunConfig": (".quantization_pipeline", "QuantizationRunConfig"),
    "run_quantization_sweep": (".quantization_pipeline", "run_quantization_sweep"),
}

__all__ = list(_EXPORTS.keys())


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
