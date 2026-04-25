from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
from utils.path_config import ProjectPaths

SUMMARY_COLUMNS = [
    "model_id",
    "type",
    "student_model",
    "training_method",
    "teacher_id_for_distill",
    "test_loss",
    "test_accuracy",
    "test_precision",
    "test_recall",
    "test_f1_score",
    "fp32_loss",
    "fp32_accuracy",
    "fp32_precision",
    "fp32_recall",
    "fp32_f1_score",
    "quantized_loss",
    "quantized_accuracy",
    "quantized_precision",
    "quantized_recall",
    "quantized_f1_score",
]


def generate_final_report(
    dataset_name: str,
    date_of_interest: str,
    paths: ProjectPaths,
) -> Path:
    import pandas as pd

    from src.core.plotting import (
        save_quantization_dumbbell,
        save_student_training_barplot,
        save_teacher_faceted_barplot,
    )
    from src.core.results_merge import build_run_file, merge_experiment_with_metrics

    experiment_path = build_run_file(
        paths.results_dir,
        dataset_name,
        date_of_interest,
        f"experiment_results_get_model_by_name{date_of_interest}.csv",
    )
    quant_path = build_run_file(
        paths.qres_dir,
        dataset_name,
        date_of_interest,
        "results_quantization_comparison.csv",
    )

    if not experiment_path.exists():
        raise FileNotFoundError(f"Experiment file not found: {experiment_path}")
    if not quant_path.exists():
        raise FileNotFoundError(f"Quantization file not found: {quant_path}")

    df_experiment = pd.read_csv(experiment_path)
    df_quant = pd.read_csv(quant_path)
    merged_df = merge_experiment_with_metrics(df_experiment, df_quant)

    output_dir = Path(paths.final_results_dir) / dataset_name / date_of_interest
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_columns = [column for column in SUMMARY_COLUMNS if column in merged_df.columns]
    summary_frame = merged_df[summary_columns] if summary_columns else merged_df
    summary_path = output_dir / "final_results.csv"
    summary_frame.to_csv(summary_path, index=False)

    students_df = merged_df[merged_df["type"] == "Student"].copy()

    if not students_df.empty:
        if {"student_model", "training_method", "fp32_accuracy"}.issubset(students_df.columns):
            save_student_training_barplot(
                students_df,
                value_column="fp32_accuracy",
                output_path=output_dir / "distillation_impact_student_accuracy_fp32.png",
                title="Distillation Impact on Student Accuracy (FP32)",
            )

        if {"student_model", "training_method", "quantized_accuracy"}.issubset(students_df.columns):
            save_student_training_barplot(
                students_df,
                value_column="quantized_accuracy",
                output_path=output_dir / "distillation_impact_student_accuracy_int8.png",
                title="Distillation Impact on Student Accuracy (INT8)",
            )

        distilled_df = students_df[students_df["training_method"] == "Distilled"].copy()
        if {
            "student_model",
            "teacher_id_for_distill",
            "quantized_accuracy",
        }.issubset(distilled_df.columns) and not distilled_df.empty:
            save_teacher_faceted_barplot(
                distilled_df,
                value_column="quantized_accuracy",
                output_path=output_dir / "best_teacher_by_student_int8.png",
                title="Best Teacher per Student (INT8 Accuracy)",
            )

        if {"model_id", "student_model", "teacher_id_for_distill", "fp32_f1_score", "quantized_f1_score"}.issubset(students_df.columns):
            save_quantization_dumbbell(
                students_df,
                fp32_column="fp32_f1_score",
                int8_column="quantized_f1_score",
                output_path=output_dir / "quantization_impact_f1_dumbbell.png",
                title="Quantization Impact on Student F1 Score",
            )

    return summary_path


def main() -> None:
    defaults = ProjectPaths()
    parser = argparse.ArgumentParser(description="Merge final experiment and quantization results.")
    parser.add_argument("--dataset", default="zpdd", help="Dataset name.")
    parser.add_argument("--date", default="2025-06-30-15-05", help="Run timestamp.")
    parser.add_argument("--results-dir", default=defaults.results_dir, help="Directory containing experiment results.")
    parser.add_argument("--qres-dir", default=defaults.qres_dir, help="Directory containing quantization results.")
    parser.add_argument("--final-results-dir", default=defaults.final_results_dir, help="Directory where merged outputs are saved.")
    args = parser.parse_args()

    paths = ProjectPaths(
        results_dir=args.results_dir,
        qres_dir=args.qres_dir,
        aucs_dir=defaults.aucs_dir,
        final_results_dir=args.final_results_dir,
    )

    output_file = generate_final_report(args.dataset, args.date, paths)
    print(f"Saved final merged report to: {output_file}")


if __name__ == "__main__":
    main()
