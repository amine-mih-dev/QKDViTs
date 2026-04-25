from __future__ import annotations

from pathlib import Path

import pandas as pd


def build_run_file(base_dir: str, dataset_name: str, date_of_interest: str, filename: str) -> Path:
    return Path(base_dir) / dataset_name / date_of_interest / filename


def _create_join_key(row: pd.Series) -> str | None:
    if row.get("type") == "Teacher":
        return row.get("model_id")

    student_model = row.get("student_model")
    training_method = row.get("training_method")

    if training_method == "Independent":
        return f"{student_model}_Independent"
    if training_method == "Distilled":
        teacher_id = row.get("teacher_id_for_distill")
        return f"{student_model}_Distilled_from_{teacher_id}"

    return None


def _enrich_experiment_frame(experiment_df: pd.DataFrame) -> pd.DataFrame:
    df = experiment_df.copy()
    df["student_model"] = df.apply(
        lambda row: "_".join(str(row["model_id"]).split("_")[:2])
        if row.get("type") == "Student"
        else "N/A",
        axis=1,
    )
    df["join_key"] = df.apply(_create_join_key, axis=1)
    return df


def merge_experiment_with_metrics(
    experiment_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
) -> pd.DataFrame:
    if "model_id" not in metrics_df.columns:
        raise ValueError("metrics_df must include a model_id column")

    exp_df = _enrich_experiment_frame(experiment_df)
    merged = exp_df.merge(
        metrics_df,
        left_on="join_key",
        right_on="model_id",
        how="inner",
        suffixes=("_exp", "_metrics"),
    )

    if "model_id_exp" in merged.columns:
        merged = merged.rename(columns={"model_id_exp": "model_id"})
    merged = merged.drop(columns=["model_id_metrics", "join_key"], errors="ignore")

    return merged
