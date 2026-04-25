from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.core.data_loading import load_fold_pickle_directory
from src.core.metrics import calculate_weighted_classification_metrics, multiclass_auc_from_outputs
from src.core.plotting import save_confusion_matrix, save_multiclass_roc_curves


def analyze_pickle_folds(input_dir: str, output_dir: str) -> dict[str, float]:
    df = load_fold_pickle_directory(input_dir)
    required_columns = {"labels", "outputs"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in PKL folds: {sorted(missing)}")

    auc_data = multiclass_auc_from_outputs(df["outputs"], df["labels"])

    if "prediction" in df.columns:
        predictions = df["prediction"].to_numpy()
    else:
        predictions = np.argmax(auc_data["probabilities"], axis=1)

    metrics = calculate_weighted_classification_metrics(df["labels"], predictions)
    confusion = confusion_matrix(df["labels"], predictions)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    save_confusion_matrix(
        confusion,
        output_root / "fold_exp_pkl_confusion_matrix.png",
        title="Fold PKL Confusion Matrix",
    )
    save_multiclass_roc_curves(
        auc_data["fpr"],
        auc_data["tpr"],
        auc_data["class_auc"],
        auc_data["micro_auc"],
        output_root / "fold_exp_pkl_roc.png",
        title="Fold PKL Multiclass ROC",
    )

    return {
        "macro_auc": auc_data["macro_auc"],
        "micro_auc": auc_data["micro_auc"],
        **metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze pickled fold predictions and ROC curves.")
    parser.add_argument(
        "--input-dir",
        default="./hybridpkl05lr/",
        help="Directory containing per-fold pickle files.",
    )
    parser.add_argument(
        "--output-dir",
        default="report/outputs/analysis",
        help="Directory where generated figures are saved.",
    )
    args = parser.parse_args()

    summary = analyze_pickle_folds(args.input_dir, args.output_dir)
    print("Fold PKL summary")
    for key, value in summary.items():
        print(f"- {key}: {value:.6f}")


if __name__ == "__main__":
    main()
