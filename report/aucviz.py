from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
from utils.path_config import ProjectPaths

DEFAULT_DATASET_DATES = {
    "plantvillage": "2025-06-12-22-27",
    "zpdd": "2025-06-30-15-05",
    "taiwan": "2025-06-23-12-54",
}


def parse_dataset_dates(value: str | None) -> dict[str, str]:
    if not value:
        return DEFAULT_DATASET_DATES.copy()

    mapping: dict[str, str] = {}
    for item in value.split(","):
        token = item.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(
                "Invalid --datasets format. Use: dataset1:date1,dataset2:date2"
            )
        dataset_name, run_date = token.split(":", 1)
        mapping[dataset_name.strip()] = run_date.strip()

    if not mapping:
        raise ValueError("No dataset/date pairs provided.")
    return mapping


def _read_table(file_path: Path) -> pd.DataFrame:
    import pandas as pd

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.suffix.lower() in {".pkl", ".pickle"}:
        return pd.read_pickle(file_path)
    return pd.read_csv(file_path)


def build_dataset_auc_frame(
    dataset_name: str,
    date_of_interest: str,
    paths: ProjectPaths,
) -> pd.DataFrame:
    import pandas as pd

    from src.core.metrics import auc_from_curve_arrays, extract_micro_curve
    from src.core.results_merge import build_run_file, merge_experiment_with_metrics

    experiment_path = build_run_file(
        paths.results_dir,
        dataset_name,
        date_of_interest,
        f"experiment_results_get_model_by_name{date_of_interest}.csv",
    )
    auc_path = build_run_file(
        paths.aucs_dir,
        dataset_name,
        date_of_interest,
        "results_micro_average_auc.pkl",
    )

    if not experiment_path.exists():
        raise FileNotFoundError(f"Experiment file not found: {experiment_path}")

    experiment_df = pd.read_csv(experiment_path)
    auc_df = _read_table(auc_path)

    merged_df = merge_experiment_with_metrics(experiment_df, auc_df)
    merged_df["dataset"] = dataset_name

    for state in ("fp32", "quantized"):
        fpr_col = f"{state}_fpr"
        tpr_col = f"{state}_tpr"
        if fpr_col not in merged_df.columns or tpr_col not in merged_df.columns:
            continue

        micro_fpr_col = f"micro_{state}_fpr"
        micro_tpr_col = f"micro_{state}_tpr"
        micro_auc_col = f"micro_{state}_auc"

        merged_df[micro_fpr_col] = merged_df[fpr_col].map(extract_micro_curve)
        merged_df[micro_tpr_col] = merged_df[tpr_col].map(extract_micro_curve)
        merged_df[micro_auc_col] = merged_df.apply(
            lambda row: auc_from_curve_arrays(row[micro_fpr_col], row[micro_tpr_col]),
            axis=1,
        )

    return merged_df


def build_dataset_final_frame(
    dataset_name: str,
    date_of_interest: str,
    paths: ProjectPaths,
) -> pd.DataFrame:
    import pandas as pd

    from src.core.results_merge import build_run_file

    final_path = build_run_file(
        paths.final_results_dir,
        dataset_name,
        date_of_interest,
        "final_results.csv",
    )
    if not final_path.exists():
        raise FileNotFoundError(
            f"Final merged report not found: {final_path}. "
            "Run report/result_final.py for this dataset/date first."
        )

    final_df = pd.read_csv(final_path)
    final_df["dataset"] = dataset_name
    return final_df


def to_long_micro_auc_frame(frame: pd.DataFrame) -> pd.DataFrame:
    import pandas as pd

    slices = []
    for state, label in (("fp32", "FP32"), ("quantized", "INT8")):
        auc_column = f"micro_{state}_auc"
        if auc_column not in frame.columns:
            continue

        columns = [
            "dataset",
            "model_id",
            "type",
            "student_model",
            "training_method",
            "teacher_id_for_distill",
            auc_column,
        ]
        available = [column for column in columns if column in frame.columns]
        part = frame[available].copy()
        part["model_state"] = label
        part = part.rename(columns={auc_column: "micro_auc"})
        slices.append(part)

    if not slices:
        return pd.DataFrame(
            columns=[
                "dataset",
                "model_id",
                "type",
                "student_model",
                "training_method",
                "teacher_id_for_distill",
                "micro_auc",
                "model_state",
            ]
        )

    return pd.concat(slices, ignore_index=True)


def generate_auc_report(
    dataset_dates: dict[str, str],
    paths: ProjectPaths,
    output_dir: str,
) -> tuple[Path, Path, Path | None, Path | None, Path | None]:
    import pandas as pd

    from src.core.plotting import (
        save_micro_auc_boxplot,
        save_student_architecture_consistency_plot,
        save_teacher_consistency_plot,
    )

    frames = [
        build_dataset_auc_frame(dataset_name, run_date, paths)
        for dataset_name, run_date in dataset_dates.items()
    ]
    merged = pd.concat(frames, ignore_index=True)
    long_auc = to_long_micro_auc_frame(merged)

    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    merged_path = target_dir / "merged_datasets_micro_auc.csv"
    long_path = target_dir / "micro_auc_long.csv"
    plot_path = target_dir / "micro_auc_boxplot.png"
    merged_final_path: Path | None = None
    teacher_consistency_path: Path | None = None
    student_consistency_path: Path | None = None

    merged.to_csv(merged_path, index=False)
    long_auc.to_csv(long_path, index=False)

    if not long_auc.empty:
        save_micro_auc_boxplot(
            long_auc,
            plot_path,
            title="Micro-Average AUC Across Datasets",
        )

    final_frames: list[pd.DataFrame] = []
    missing_final_reports: list[str] = []
    for dataset_name, run_date in dataset_dates.items():
        try:
            final_frames.append(build_dataset_final_frame(dataset_name, run_date, paths))
        except FileNotFoundError as exc:
            missing_final_reports.append(str(exc))

    if final_frames:
        merged_final = pd.concat(final_frames, ignore_index=True)
        merged_final_path = target_dir / "merged_datasets_final_results.csv"
        merged_final.to_csv(merged_final_path, index=False)

        dataset_order = list(dataset_dates.keys())

        try:
            teacher_consistency_path = target_dir / "teacher_consistency.png"
            save_teacher_consistency_plot(
                merged_final,
                output_path=teacher_consistency_path,
                dataset_order=dataset_order,
            )
        except ValueError as exc:
            teacher_consistency_path = None
            print(f"Skipped teacher consistency plot: {exc}")

        try:
            student_consistency_path = target_dir / "student_architecture_consistency.png"
            save_student_architecture_consistency_plot(
                merged_final,
                output_path=student_consistency_path,
                dataset_order=dataset_order,
            )
        except ValueError as exc:
            student_consistency_path = None
            print(f"Skipped student consistency plot: {exc}")
    elif missing_final_reports:
        print("Skipped teacher/student consistency plots because final reports are missing:")
        for message in missing_final_reports:
            print(f"- {message}")

    return (
        merged_path,
        long_path,
        merged_final_path,
        teacher_consistency_path,
        student_consistency_path,
    )


def main() -> None:
    defaults = ProjectPaths()
    parser = argparse.ArgumentParser(description="Aggregate micro-average AUC curves across datasets.")
    parser.add_argument(
        "--datasets",
        default=None,
        help="Comma-separated dataset:date pairs, example: zpdd:2025-06-30-15-05,taiwan:2025-06-23-12-54",
    )
    parser.add_argument("--results-dir", default=defaults.results_dir, help="Directory containing experiment results.")
    parser.add_argument("--aucs-dir", default=defaults.aucs_dir, help="Directory containing AUC result files.")
    parser.add_argument("--final-results-dir", default=defaults.final_results_dir, help="Directory containing final merged reports.")
    parser.add_argument("--output-dir", default=defaults.results_dir, help="Directory where merged AUC report files are saved.")
    args = parser.parse_args()

    dataset_dates = parse_dataset_dates(args.datasets)
    paths = ProjectPaths(
        results_dir=args.results_dir,
        qres_dir=defaults.qres_dir,
        aucs_dir=args.aucs_dir,
        final_results_dir=args.final_results_dir,
    )

    merged_path, long_path, merged_final_path, teacher_plot_path, student_plot_path = generate_auc_report(
        dataset_dates,
        paths,
        args.output_dir,
    )
    print(f"Saved merged AUC data to: {merged_path}")
    print(f"Saved long-format micro AUC data to: {long_path}")
    if merged_final_path is not None:
        print(f"Saved merged final-results data to: {merged_final_path}")
    if teacher_plot_path is not None:
        print(f"Saved teacher consistency plot to: {teacher_plot_path}")
    if student_plot_path is not None:
        print(f"Saved student consistency plot to: {student_plot_path}")


if __name__ == "__main__":
    main()
