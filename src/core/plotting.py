from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


STYLE_FIGSIZE = (12, 7)
STYLE_DPI = 300
STYLE_BBOX = "tight"
STYLE_TITLE_FONTSIZE = 26
STYLE_LABEL_FONTSIZE = 26
STYLE_TICK_FONTSIZE = 26
STYLE_LEGEND_FONTSIZE = 16
STYLE_LINEWIDTH = 5
STYLE_MARKER_AREA = 400
STYLE_LINE_MARKER_SIZE = 20


def _style_axis(ax, title: str, xlabel: str, ylabel: str) -> None:
    # ax.set_title(title, fontsize=STYLE_TITLE_FONTSIZE)
    ax.set_xlabel(xlabel, fontsize=STYLE_LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=STYLE_LABEL_FONTSIZE)
    ax.tick_params(axis="both", which="major", labelsize=STYLE_TICK_FONTSIZE)


def _save_figure(fig, target: Path, rect: tuple[float, float, float, float] | None = None) -> None:
    if rect is not None:
        fig.tight_layout(rect=rect)
    else:
        fig.tight_layout()
    fig.savefig(target, dpi=STYLE_DPI, bbox_inches=STYLE_BBOX)


def _format_dataset_label(name: str) -> str:
    normalized = str(name).replace("_", " ").strip()
    if normalized.lower() == "zpdd":
        return "ZPDD"
    return normalized.title()


def _apply_dataset_order(frame: pd.DataFrame, dataset_order: list[str] | None) -> pd.DataFrame:
    ordered = frame.copy()
    if dataset_order:
        ordered["dataset"] = pd.Categorical(
            ordered["dataset"],
            categories=dataset_order,
            ordered=True,
        )
    return ordered


def save_confusion_matrix(
    confusion,
    output_path: str | Path,
    title: str = "Confusion Matrix",
    class_labels=None,
) -> None:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=STYLE_FIGSIZE)
    sns.heatmap(
        confusion,
        annot=True,
        fmt="d",
        cmap="Blues",
        ax=ax,
        xticklabels=class_labels,
        yticklabels=class_labels,
    )
    _style_axis(ax, title, "Predicted label", "True label")
    _save_figure(fig, target)
    plt.close(fig)


def save_multiclass_roc_curves(
    fpr_map: dict,
    tpr_map: dict,
    class_auc: dict,
    micro_auc: float,
    output_path: str | Path,
    title: str,
    class_labels: list[str] | None = None,
) -> None:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=STYLE_FIGSIZE)
    for class_idx in sorted([key for key in class_auc.keys() if isinstance(key, int)]):
        class_name = (
            class_labels[class_idx]
            if class_labels is not None and class_idx < len(class_labels)
            else str(class_idx)
        )
        ax.plot(
            fpr_map[class_idx],
            tpr_map[class_idx],
            label=f"Class {class_name} (AUC={class_auc[class_idx]:.4f})",
            linewidth=STYLE_LINEWIDTH,
        )

    if "micro" in fpr_map and "micro" in tpr_map:
        ax.plot(
            fpr_map["micro"],
            tpr_map["micro"],
            linestyle=":",
            linewidth=STYLE_LINEWIDTH,
            color="black",
            label=f"Micro-average (AUC={micro_auc:.4f})",
        )

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    _style_axis(ax, title, "False Positive Rate", "True Positive Rate")
    ax.legend(loc="lower right", fontsize=STYLE_LEGEND_FONTSIZE)
    ax.grid(True, alpha=0.3)
    _save_figure(fig, target)
    plt.close(fig)


def save_student_training_barplot(
    students_df: pd.DataFrame,
    value_column: str,
    output_path: str | Path,
    title: str,
) -> None:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=STYLE_FIGSIZE)
    sns.barplot(
        data=students_df,
        x="student_model",
        y=value_column,
        hue="training_method",
        palette="viridis",
        ax=ax,
    )
    _style_axis(ax, title, "Student model", value_column)
    for label in ax.get_xticklabels():
        label.set_rotation(12)
        label.set_ha("center")

    legend = ax.get_legend()
    if legend is not None:
        legend.set_title("Training method")
        legend.get_title().set_fontsize(STYLE_LEGEND_FONTSIZE)
        for text in legend.get_texts():
            text.set_fontsize(STYLE_LEGEND_FONTSIZE)

    _save_figure(fig, target)
    plt.close(fig)


def save_teacher_faceted_barplot(
    distilled_df: pd.DataFrame,
    value_column: str,
    output_path: str | Path,
    title: str,
) -> None:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    grid = sns.catplot(
        data=distilled_df,
        x="teacher_id_for_distill",
        y=value_column,
        col="student_model",
        kind="bar",
        palette="magma",
        height=5,
        aspect=1.0,
    )
    grid.fig.suptitle(title, y=1.04, fontsize=STYLE_TITLE_FONTSIZE)
    grid.set_axis_labels("Teacher model", value_column)

    for ax in grid.axes.flatten():
        ax.tick_params(axis="both", which="major", labelsize=STYLE_TICK_FONTSIZE - 4)
        ax.set_xlabel("Teacher model", fontsize=STYLE_LABEL_FONTSIZE - 2)
        ax.set_ylabel(value_column, fontsize=STYLE_LABEL_FONTSIZE - 2)
        for label in ax.get_xticklabels():
            label.set_rotation(35)
            label.set_ha("right")

    if grid._legend is not None:
        grid._legend.set_title(grid._legend.get_title().get_text())
        grid._legend.get_title().set_fontsize(STYLE_LEGEND_FONTSIZE)
        for text in grid._legend.get_texts():
            text.set_fontsize(STYLE_LEGEND_FONTSIZE)

    _save_figure(grid.fig, target, rect=(0, 0, 1, 0.94))
    plt.close(grid.fig)


def save_quantization_dumbbell(
    students_df: pd.DataFrame,
    fp32_column: str,
    int8_column: str,
    output_path: str | Path,
    title: str,
) -> None:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    plot_df = students_df.sort_values(
        by=["student_model", "teacher_id_for_distill"],
        na_position="first",
    ).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=STYLE_FIGSIZE)
    x_positions = list(range(len(plot_df)))

    ax.scatter(
        x_positions,
        plot_df[fp32_column],
        color="cornflowerblue",
        s=STYLE_MARKER_AREA,
        label="FP32",
        zorder=3,
    )
    ax.scatter(
        x_positions,
        plot_df[int8_column],
        color="crimson",
        s=STYLE_MARKER_AREA,
        label="INT8",
        zorder=3,
    )

    for idx, row in plot_df.iterrows():
        ax.plot(
            [idx, idx],
            [row[fp32_column], row[int8_column]],
            color="gray",
            linewidth=STYLE_LINEWIDTH,
            zorder=2,
        )

    _style_axis(ax, title, "Model configuration", "F1 score")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(plot_df["model_id"])
    for label in ax.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")

    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(fontsize=STYLE_LEGEND_FONTSIZE)
    _save_figure(fig, target)
    plt.close(fig)


def save_micro_auc_boxplot(
    long_auc_df: pd.DataFrame,
    output_path: str | Path,
    title: str,
) -> None:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=STYLE_FIGSIZE)
    sns.boxplot(
        data=long_auc_df,
        x="dataset",
        y="micro_auc",
        hue="model_state",
        palette="Set2",
        ax=ax,
    )
    _style_axis(ax, title, "Dataset", "Micro-average AUC")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    legend = ax.get_legend()
    if legend is not None:
        legend.get_title().set_fontsize(STYLE_LEGEND_FONTSIZE)
        for text in legend.get_texts():
            text.set_fontsize(STYLE_LEGEND_FONTSIZE)

    _save_figure(fig, target)
    plt.close(fig)


def save_teacher_consistency_plot(
    merged_final_df: pd.DataFrame,
    output_path: str | Path,
    title: str = "Teacher Strategy Consistency: Average Rank of Resulting Students",
    dataset_order: list[str] | None = None,
    metric_column: str = "fp32_accuracy",
) -> None:
    required_columns = {"dataset", "type", "training_method", "teacher_id_for_distill", metric_column}
    missing = required_columns - set(merged_final_df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns for teacher consistency plot: {sorted(missing)}"
        )

    working = _apply_dataset_order(merged_final_df, dataset_order)
    distilled = working[
        (working["type"] == "Student")
        & (working["training_method"] == "Distilled")
    ]
    independent = working[
        (working["type"] == "Student")
        & (working["training_method"] == "Independent")
    ]

    grouped_distilled = (
        distilled.groupby(["dataset", "teacher_id_for_distill"], observed=False)[metric_column]
        .mean()
        .reset_index()
        .rename(columns={"teacher_id_for_distill": "method"})
    )
    grouped_independent = (
        independent.groupby(["dataset"], observed=False)[metric_column]
        .mean()
        .reset_index()
    )
    grouped_independent["method"] = "Independent"

    teacher_perf_df = pd.concat([grouped_distilled, grouped_independent], ignore_index=True)
    teacher_perf_df = teacher_perf_df.dropna(subset=["dataset", metric_column])
    if teacher_perf_df.empty:
        raise ValueError("No teacher consistency data available after filtering")

    teacher_perf_df["method"] = teacher_perf_df["method"].astype(str).str.replace("_GMN", "", regex=False)
    teacher_perf_df["rank"] = teacher_perf_df.groupby("dataset", observed=False)[metric_column].rank(
        method="min",
        ascending=False,
    )

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=STYLE_FIGSIZE)
    sns.lineplot(
        data=teacher_perf_df,
        x="dataset",
        y="rank",
        hue="method",
        style="method",
        markers=True,
        markersize=STYLE_LINE_MARKER_SIZE,
        linewidth=STYLE_LINEWIDTH,
        ax=ax,
    )

    ax.invert_yaxis()
    max_rank = int(teacher_perf_df["rank"].max()) if not teacher_perf_df["rank"].isna().all() else 1
    ax.set_yticks(list(range(1, max_rank + 1)))
    _style_axis(ax, title, "Dataset", "Performance Rank (1=Best)")

    tick_positions = ax.get_xticks()
    labels = [_format_dataset_label(tick.get_text()) for tick in ax.get_xticklabels()]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(labels, ha="center")

    legend = ax.get_legend()
    if legend is not None:
        legend.set_title("Teaching Method")
        legend.get_title().set_fontsize(STYLE_LEGEND_FONTSIZE)
        for text in legend.get_texts():
            text.set_fontsize(STYLE_LEGEND_FONTSIZE)

    _save_figure(fig, target)
    plt.close(fig)


def save_student_architecture_consistency_plot(
    merged_final_df: pd.DataFrame,
    output_path: str | Path,
    title: str = "Student Architecture Consistency: Average Performance Rank",
    dataset_order: list[str] | None = None,
    metric_column: str = "fp32_accuracy",
) -> None:
    required_columns = {"dataset", "type", "student_model", metric_column}
    missing = required_columns - set(merged_final_df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns for student consistency plot: {sorted(missing)}"
        )

    working = _apply_dataset_order(merged_final_df, dataset_order)
    all_students = working[working["type"] == "Student"]
    avg_perf_by_student = (
        all_students.groupby(["dataset", "student_model"], observed=False)[metric_column]
        .mean()
        .reset_index()
    )
    avg_perf_by_student = avg_perf_by_student.dropna(subset=["dataset", "student_model", metric_column])
    if avg_perf_by_student.empty:
        raise ValueError("No student consistency data available after filtering")

    avg_perf_by_student["rank"] = avg_perf_by_student.groupby("dataset", observed=False)[metric_column].rank(
        method="min",
        ascending=False,
    )

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=STYLE_FIGSIZE)
    sns.lineplot(
        data=avg_perf_by_student,
        x="dataset",
        y="rank",
        hue="student_model",
        style="student_model",
        markers=True,
        markersize=STYLE_LINE_MARKER_SIZE,
        linewidth=STYLE_LINEWIDTH,
        ax=ax,
    )

    ax.invert_yaxis()
    max_rank = int(avg_perf_by_student["rank"].max()) if not avg_perf_by_student["rank"].isna().all() else 1
    ax.set_yticks(list(range(1, max_rank + 1)))
    _style_axis(ax, title, "Dataset", "Performance Rank (1=Best)")

    tick_positions = ax.get_xticks()
    labels = [_format_dataset_label(tick.get_text()) for tick in ax.get_xticklabels()]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(labels, ha="center")

    legend = ax.get_legend()
    if legend is not None:
        legend.set_title("Student Architecture")
        legend.get_title().set_fontsize(STYLE_LEGEND_FONTSIZE)
        for text in legend.get_texts():
            text.set_fontsize(STYLE_LEGEND_FONTSIZE)

    _save_figure(fig, target)
    plt.close(fig)
