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


def resolve_test_dir(dataset_name: str, dataset_root: str | None, test_dir: str | None) -> str:
    if test_dir:
        return test_dir

    root = Path(dataset_root) if dataset_root else None
    if root is None:
        default_root = DEFAULT_DATASET_ROOTS.get(dataset_name)
        if default_root:
            root = Path(default_root)

    if root is None:
        raise ValueError(
            "Missing test dataset path. Provide --test-dir or --dataset-root."
        )

    output_style = root / "output" / "test"
    if output_style.exists():
        return str(output_style)

    if dataset_name == "taiwan":
        taiwan_eval = root / "val"
        if taiwan_eval.exists():
            return str(taiwan_eval)

    flat_style = root / "test"
    if flat_style.exists():
        return str(flat_style)

    raise FileNotFoundError(f"Could not resolve test directory under {root}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate trained checkpoints and produce quantization comparison artifacts."
    )
    parser.add_argument("--dataset-name", default="zpdd", help="Dataset key used in checkpoints and outputs.")
    parser.add_argument("--date", required=True, help="Run timestamp to load checkpoints from.")

    parser.add_argument("--dataset-root", default=None, help="Dataset root if test directory should be inferred.")
    parser.add_argument("--test-dir", default=None, help="Explicit ImageFolder test directory.")

    parser.add_argument("--checkpoints-dir", default="checkpoints", help="Root checkpoints directory.")
    parser.add_argument("--qres-dir", default="qres", help="Output root for quantization summary CSV.")
    parser.add_argument("--aucs-dir", default="aucs", help="Output root for AUC CSV/PKL files.")
    parser.add_argument(
        "--visualization-dir",
        default="visualizations",
        help="Output root for ROC/AUC plots.",
    )

    parser.add_argument("--batch-size", type=int, default=32, help="Evaluation batch size.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader num_workers.")
    parser.add_argument("--image-size", type=int, default=224, help="Evaluation crop size.")
    parser.add_argument("--eval-device", default="cpu", help="FP32 eval device (cpu or cuda).")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    test_dir = resolve_test_dir(args.dataset_name, args.dataset_root, args.test_dir)

    from src.train.quantization_pipeline import (
        QuantizationRunConfig,
        run_quantization_sweep,
    )

    config = QuantizationRunConfig(
        dataset_name=args.dataset_name,
        run_timestamp=args.date,
        test_dir=test_dir,
        checkpoints_dir=args.checkpoints_dir,
        qres_dir=args.qres_dir,
        aucs_dir=args.aucs_dir,
        visualization_dir=args.visualization_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        eval_device=args.eval_device,
    )

    print(f"Running quantization sweep for dataset: {config.dataset_name}")
    print(f"Run timestamp: {config.run_timestamp}")
    print(f"Using test directory: {config.test_dir}")

    qres_path, auc_csv_path, auc_pkl_path = run_quantization_sweep(config)

    print("\nQuantization sweep complete.")
    print(f"Saved quantization summary: {qres_path}")
    print(f"Saved AUC summary CSV: {auc_csv_path}")
    print(f"Saved AUC summary PKL: {auc_pkl_path}")


if __name__ == "__main__":
    main()
