from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sklearn.metrics import confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.core.data_loading import load_fold_csv_directory
from src.core.metrics import calculate_weighted_classification_metrics
from src.core.plotting import save_confusion_matrix


def analyze_csv_folds(input_dir: str, output_dir: str) -> dict[str, float]:
    df = load_fold_csv_directory(input_dir)
    required_columns = {"labels", "prediction"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in CSV folds: {sorted(missing)}")

    metrics = calculate_weighted_classification_metrics(df["labels"], df["prediction"])
    confusion = confusion_matrix(df["labels"], df["prediction"])

    output_path = Path(output_dir) / "fold_exp_confusion_matrix.png"
    save_confusion_matrix(confusion, output_path, title="Fold Experiment Confusion Matrix")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze CSV fold predictions.")
    parser.add_argument(
        "--input-dir",
        default="./resnet50fr/",
        help="Directory containing per-fold CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        default="report/outputs/analysis",
        help="Directory where generated figures are saved.",
    )
    args = parser.parse_args()

    metrics = analyze_csv_folds(args.input_dir, args.output_dir)
    print("Classification metrics")
    for key, value in metrics.items():
        print(f"- {key}: {value:.6f}")
    print(f"Saved confusion matrix: {Path(args.output_dir) / 'fold_exp_confusion_matrix.png'}")


if __name__ == "__main__":
    main()
