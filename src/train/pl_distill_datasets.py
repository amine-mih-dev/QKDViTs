from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

DEFAULT_DATASET_ROOTS = {
    "banana": r"D:\malik\LI3C\Projet de recherche\PNR 2022\programmes\Programmes_Python\Quanti_Vits_Amine\ouafi\Banana Leaf Spot Diseases",
    "plantvillage": r"D:\malik\LI3C\Projet de recherche\PNR 2022\programmes\Programmes_Python\Quanti_Vits_Amine\ouafi\Tomato plantvillage and taiwan\Tomato\Plant_leaf_diseases_without _augmentation",
    "zpdd": r"D:\malik\LI3C\Projet de recherche\PNR 2022\programmes\Programmes_Python\Quanti_Vits_Amine\ouafi\programeCropped",
    "plantdoc": r"D:\malik\LI3C\Projet de recherche\PNR 2022\programmes\Programmes_Python\Quanti_Vits_Amine\ouafi\plantdoc_t_dataset",
    "ccmt": r"D:\malik\LI3C\Projet de recherche\PNR 2022\programmes\Programmes_Python\Quanti_Vits_Amine\ouafi\cmtt",
    "taiwan": r"D:\malik\LI3C\Projet de recherche\PNR 2022\programmes\Programmes_Python\Quanti_Vits_Amine\ouafi\taiwan\taiwan\data_aug_811",
}


def resolve_split_dirs(
    dataset_name: str,
    dataset_root: str | None,
    train_dir: str | None,
    val_dir: str | None,
    test_dir: str | None,
) -> tuple[str, str, str]:
    if train_dir and val_dir and test_dir:
        return train_dir, val_dir, test_dir

    root = Path(dataset_root) if dataset_root else None
    if root is None:
        default_root = DEFAULT_DATASET_ROOTS.get(dataset_name)
        if default_root:
            root = Path(default_root)

    if root is None:
        raise ValueError(
            "Dataset split paths are missing. Provide --dataset-root or explicit "
            "--train-dir/--val-dir/--test-dir."
        )

    output_style = (root / "output" / "train", root / "output" / "val", root / "output" / "test")
    if all(path.exists() for path in output_style):
        return tuple(str(path) for path in output_style)

    flat_style = (root / "train", root / "val", root / "test")
    if all(path.exists() for path in flat_style):
        return tuple(str(path) for path in flat_style)

    taiwan_style = (root / "train", root / "test", root / "val")
    if dataset_name == "taiwan" and all(path.exists() for path in taiwan_style):
        return tuple(str(path) for path in taiwan_style)

    raise FileNotFoundError(
        f"Could not resolve train/val/test directories under dataset root: {root}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run teacher/student training and distillation experiments."
    )
    parser.add_argument("--dataset-name", default="zpdd", help="Dataset key for run naming.")
    parser.add_argument("--dataset-root", default=None, help="Root folder containing train/val/test or output/train/val/test.")
    parser.add_argument("--train-dir", default=None, help="Explicit training directory.")
    parser.add_argument("--val-dir", default=None, help="Explicit validation directory.")
    parser.add_argument("--test-dir", default=None, help="Explicit test directory.")

    parser.add_argument("--timestamp", default=None, help="Run timestamp. Defaults to current time.")
    parser.add_argument("--results-dir", default="results", help="Directory for experiment CSV outputs.")
    parser.add_argument("--checkpoints-dir", default="checkpoints", help="Directory for checkpoints.")
    parser.add_argument("--logs-dir", default="logs", help="Directory for Lightning CSV logs.")

    parser.add_argument("--batch-size", type=int, default=64, help="DataLoader batch size.")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader num_workers.")
    parser.add_argument("--image-size", type=int, default=224, help="Image crop size.")

    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--accuracy-task", default="multiclass", help="Torchmetrics task type.")

    parser.add_argument("--teacher-epochs", type=int, default=40, help="Teacher phase epochs.")
    parser.add_argument("--student-epochs", type=int, default=100, help="Independent student and distillation epochs.")

    parser.add_argument(
        "--accelerator",
        default="auto",
        choices=["cpu", "gpu", "auto", None],
        help="Lightning accelerator override.",
    )
    parser.add_argument("--devices", type=int, default=1, help="Lightning devices count.")
    parser.add_argument(
        "--precision",
        default="16-mixed",
        help="Lightning precision override (e.g. 32, 16-mixed).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_dir, val_dir, test_dir = resolve_split_dirs(
        dataset_name=args.dataset_name,
        dataset_root=args.dataset_root,
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        test_dir=args.test_dir,
    )

    from src.train.config import (
        DataConfig,
        ExperimentConfig,
        RuntimeConfig,
        build_results_log_file,
        default_distillation_config,
        default_student_specs,
        default_teacher_specs,
        ensure_run_timestamp,
    )
    from src.train.pipeline import run_training_experiment

    run_timestamp = ensure_run_timestamp(args.timestamp)
    runtime = RuntimeConfig()
    if args.accelerator is not None:
        runtime.accelerator = args.accelerator
    runtime.devices = args.devices
    if args.precision is not None:
        runtime.precision = args.precision

    data = DataConfig(
        train_dir=train_dir,
        val_dir=val_dir,
        test_dir=test_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )

    teachers = default_teacher_specs(teacher_epochs=args.teacher_epochs)
    students = default_student_specs(student_epochs=args.student_epochs)
    distillation = default_distillation_config(student_epochs=args.student_epochs)

    results_log_file = build_results_log_file(
        results_dir=args.results_dir,
        dataset_name=args.dataset_name,
        run_timestamp=run_timestamp,
    )

    config = ExperimentConfig(
        dataset_name=args.dataset_name,
        run_timestamp=run_timestamp,
        data=data,
        runtime=runtime,
        results_log_file=results_log_file,
        teachers=teachers,
        students=students,
        distillation=distillation,
        seed=args.seed,
        accuracy_task=args.accuracy_task,
        checkpoints_dir=args.checkpoints_dir,
        logs_dir=args.logs_dir,
    )

    print(f"Running distillation pipeline for dataset: {config.dataset_name}")
    print(f"Run timestamp: {config.run_timestamp}")
    print(f"Results CSV: {config.results_log_file}")

    run_training_experiment(config)

    print("\nAll training phases complete.")
    print(f"Experiment results saved to: {config.results_log_file}")


if __name__ == "__main__":
    main()
