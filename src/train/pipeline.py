from __future__ import annotations

import csv
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import CSVLogger

from .config import ExperimentConfig, StudentSpec
from .data_module import ImageFolderDataModule
from .lightning_modules import DistillerTrainer, EvaluationModule, StandardModel
from .model_factory import build_student_from_spec


def _sanitize(value: str) -> str:
    return value.replace(":", "-").replace("/", "_")


def _normalize_test_metrics(metrics: dict[str, object]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for key, value in metrics.items():
        normalized_key = key if key.startswith("test_") else f"test_{key}"
        normalized[normalized_key] = float(value)
    return normalized


def log_results_to_csv(filename: str, data_dict: dict[str, object]) -> None:
    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()

    base_fieldnames = [
        "model_id",
        "type",
        "training_method",
        "teacher_id_for_distill",
        "test_loss",
        "test_accuracy",
        "test_precision",
        "test_recall",
        "test_f1_score",
    ]

    all_fieldnames = list(base_fieldnames)
    for key in data_dict:
        if key not in all_fieldnames:
            all_fieldnames.append(key)

    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=all_fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(data_dict)


def setup_trainer_and_callbacks(
    config: ExperimentConfig,
    phase_prefix: str,
    run_identifier: str,
    monitor_metric: str,
    monitor_mode: str,
    max_epochs: int,
) -> tuple[pl.Trainer, ModelCheckpoint, Path]:
    safe_run_identifier = _sanitize(run_identifier)
    checkpoint_dir = (
        Path(config.checkpoints_dir)
        / config.dataset_name
        / config.run_timestamp
        / phase_prefix
        / safe_run_identifier
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        monitor=monitor_metric,
        dirpath=str(checkpoint_dir),
        filename=f"best_model_{{epoch:02d}}_{{{monitor_metric}:.4f}}",
        save_top_k=1,
        mode=monitor_mode,
    )

    logger = CSVLogger(
        save_dir=str(
            Path(config.logs_dir)
            / config.dataset_name
            / config.run_timestamp
            / phase_prefix
        ),
        name=safe_run_identifier,
    )

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator=config.runtime.accelerator,
        devices=config.runtime.devices,
        precision=config.runtime.precision,
        callbacks=[checkpoint_cb, RichProgressBar(leave=False)],
        logger=logger,
    )
    return trainer, checkpoint_cb, checkpoint_dir


def evaluate_model_on_test_set(
    model_instance,
    model_id_str: str,
    num_classes: int,
    data_module: ImageFolderDataModule,
    config: ExperimentConfig,
) -> dict[str, float]:
    print(f"\n--- Evaluating: {model_id_str} on test set ---")
    data_module.setup("test")

    eval_trainer = pl.Trainer(
        accelerator=config.runtime.accelerator,
        devices=config.runtime.devices,
        precision=config.runtime.precision,
        logger=False,
        enable_checkpointing=False,
    )

    eval_module = EvaluationModule(model_instance, num_classes)
    test_results = eval_trainer.test(
        eval_module,
        datamodule=data_module,
        verbose=False,
    )
    if not test_results:
        return {}

    return _normalize_test_metrics(test_results[0])


def _load_standard_from_checkpoint(
    checkpoint_path: str,
    original_model_or_name,
) -> StandardModel:
    if isinstance(original_model_or_name, str):
        return StandardModel.load_from_checkpoint(
            checkpoint_path,
            map_location="cpu",
            model_or_backbone_name=original_model_or_name,
        )

    return StandardModel.load_from_checkpoint(
        checkpoint_path,
        map_location="cpu",
        model_or_backbone_name=original_model_or_name,
    )


def _build_standard_input(student: StudentSpec, num_classes: int):
    if student.init_type == "DistillableViT":
        return build_student_from_spec(student, num_classes)

    if student.init_type == "get_model_by_name":
        if not student.backbone_name:
            raise ValueError(f"Student {student.id} is missing backbone_name")
        return student.backbone_name

    raise ValueError(f"Unsupported student init_type: {student.init_type}")


def run_teacher_phase(
    config: ExperimentConfig,
    data_module: ImageFolderDataModule,
    num_classes: int,
) -> dict[str, torch.nn.Module]:
    print("\n" + "=" * 28 + " PHASE 1: TEACHER TRAINING " + "=" * 28)
    trained_teachers: dict[str, torch.nn.Module] = {}

    for spec in config.teachers:
        print(f"\n--- Training teacher: {spec.id} ({spec.backbone_name}) ---")
        model = StandardModel(
            model_or_backbone_name=spec.backbone_name,
            num_classes=num_classes,
            learning_rate=spec.training.learning_rate,
            weight_decay=spec.training.weight_decay,
            pretrained=spec.pretrained,
            pretrained_weights_path=spec.pretrained_weights_path,
            accuracy_task=config.accuracy_task,
            max_epochs_for_scheduler=spec.training.max_epochs_for_scheduler,
        )

        trainer, checkpoint_cb, checkpoint_dir = setup_trainer_and_callbacks(
            config=config,
            phase_prefix="teacher",
            run_identifier=spec.id,
            monitor_metric="val_loss",
            monitor_mode="min",
            max_epochs=spec.training.epochs,
        )
        trainer.fit(model, datamodule=data_module)

        trained_instance = model.model
        if checkpoint_cb.best_model_path and Path(checkpoint_cb.best_model_path).exists():
            loaded = _load_standard_from_checkpoint(
                checkpoint_cb.best_model_path,
                spec.backbone_name,
            )
            trained_instance = loaded.model
            weights_path = checkpoint_dir / f"{_sanitize(spec.id)}_best_weights.pth"
        else:
            weights_path = checkpoint_dir / f"{_sanitize(spec.id)}_last_weights.pth"

        torch.save(trained_instance.state_dict(), weights_path)
        trained_teachers[spec.id] = trained_instance

        test_metrics = evaluate_model_on_test_set(
            trained_instance,
            f"Teacher<{spec.id}>",
            num_classes,
            data_module,
            config,
        )
        row = {
            "model_id": spec.id,
            "type": "Teacher",
            "training_method": "Independent",
            "teacher_id_for_distill": "N/A",
            **test_metrics,
        }
        log_results_to_csv(config.results_log_file, row)

    return trained_teachers


def run_independent_student_phase(
    config: ExperimentConfig,
    data_module: ImageFolderDataModule,
    num_classes: int,
) -> dict[str, torch.nn.Module]:
    print("\n" + "=" * 24 + " PHASE 2: STUDENT INDEPENDENT TRAINING " + "=" * 24)
    trained_students: dict[str, torch.nn.Module] = {}

    for spec in config.students:
        print(f"\n--- Training independent student: {spec.id} ---")
        standard_input = _build_standard_input(spec, num_classes)

        model = StandardModel(
            model_or_backbone_name=standard_input,
            num_classes=num_classes,
            learning_rate=spec.training.learning_rate,
            weight_decay=spec.training.weight_decay,
            pretrained=spec.pretrained,
            pretrained_weights_path=spec.pretrained_weights_path,
            accuracy_task=config.accuracy_task,
            max_epochs_for_scheduler=spec.training.max_epochs_for_scheduler,
        )

        trainer, checkpoint_cb, checkpoint_dir = setup_trainer_and_callbacks(
            config=config,
            phase_prefix="student_indep",
            run_identifier=spec.id,
            monitor_metric="val_loss",
            monitor_mode="min",
            max_epochs=spec.training.epochs,
        )
        trainer.fit(model, datamodule=data_module)

        trained_instance = model.model
        if checkpoint_cb.best_model_path and Path(checkpoint_cb.best_model_path).exists():
            loaded = _load_standard_from_checkpoint(
                checkpoint_cb.best_model_path,
                standard_input,
            )
            trained_instance = loaded.model
            weights_path = checkpoint_dir / f"{_sanitize(spec.id)}_best_weights.pth"
        else:
            weights_path = checkpoint_dir / f"{_sanitize(spec.id)}_last_weights.pth"

        torch.save(trained_instance.state_dict(), weights_path)
        trained_students[spec.id] = trained_instance

        test_metrics = evaluate_model_on_test_set(
            trained_instance,
            f"StudentIndependent<{spec.id}>",
            num_classes,
            data_module,
            config,
        )
        row = {
            "model_id": spec.id,
            "type": "Student",
            "training_method": "Independent",
            "teacher_id_for_distill": "N/A",
            **test_metrics,
        }
        log_results_to_csv(config.results_log_file, row)

    return trained_students


def run_distillation_phase(
    config: ExperimentConfig,
    data_module: ImageFolderDataModule,
    num_classes: int,
    trained_teachers: dict[str, torch.nn.Module],
) -> None:
    print("\n" + "=" * 24 + " PHASE 3: DISTILLATION " + "=" * 24)
    if not trained_teachers:
        print("No trained teachers available. Skipping distillation phase.")
        return

    for teacher_id, teacher_model in trained_teachers.items():
        for spec in config.students:
            print(f"\n--- Distilling {spec.id} with teacher {teacher_id} ---")
            student_model = build_student_from_spec(spec, num_classes)

            distiller = DistillerTrainer(
                num_classes=num_classes,
                teacher_model=teacher_model,
                student_model=student_model,
                learning_rate=config.distillation.learning_rate,
                weight_decay=config.distillation.weight_decay,
                temperature=config.distillation.temperature,
                alpha=config.distillation.alpha,
                accuracy_task=config.accuracy_task,
                max_epochs_for_scheduler=config.distillation.max_epochs_for_scheduler,
            )

            phase_prefix = f"distilled_{spec.id}"
            run_identifier = f"with_T_{teacher_id}"
            trainer, checkpoint_cb, checkpoint_dir = setup_trainer_and_callbacks(
                config=config,
                phase_prefix=phase_prefix,
                run_identifier=run_identifier,
                monitor_metric="val_student_loss",
                monitor_mode="min",
                max_epochs=config.distillation.epochs,
            )
            trainer.fit(distiller, datamodule=data_module)

            final_student = distiller.student
            if checkpoint_cb.best_model_path and Path(checkpoint_cb.best_model_path).exists():
                dummy_student = build_student_from_spec(spec, num_classes)
                loaded = DistillerTrainer.load_from_checkpoint(
                    checkpoint_cb.best_model_path,
                    map_location="cpu",
                    teacher_model=teacher_model,
                    student_model=dummy_student,
                )
                final_student = loaded.student
                weights_path = checkpoint_dir / f"distilled_{_sanitize(spec.id)}_best_weights.pth"
            else:
                weights_path = checkpoint_dir / f"distilled_{_sanitize(spec.id)}_last_weights.pth"

            torch.save(final_student.state_dict(), weights_path)

            test_metrics = evaluate_model_on_test_set(
                final_student,
                f"Distilled<{spec.id}>_Teacher<{teacher_id}>",
                num_classes,
                data_module,
                config,
            )
            row = {
                "model_id": spec.id,
                "type": "Student",
                "training_method": "Distilled",
                "teacher_id_for_distill": teacher_id,
                **test_metrics,
            }
            log_results_to_csv(config.results_log_file, row)


def run_training_experiment(config: ExperimentConfig) -> None:
    pl.seed_everything(config.seed)

    data_module = ImageFolderDataModule(
        train_dir=config.data.train_dir,
        val_dir=config.data.val_dir,
        test_dir=config.data.test_dir,
        batch_size=config.data.batch_size,
        num_workers=config.data.num_workers,
        image_size=config.data.image_size,
    )
    data_module.setup("fit")

    if data_module.num_classes is None:
        raise ValueError("Number of classes could not be inferred from training data")

    num_classes = data_module.num_classes
    print(f"Detected {num_classes} classes: {data_module.class_names}")

    trained_teachers = run_teacher_phase(config, data_module, num_classes)
    run_independent_student_phase(config, data_module, num_classes)
    run_distillation_phase(config, data_module, num_classes, trained_teachers)
