from __future__ import annotations

import ast
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import accuracy_score, auc, f1_score, precision_score, recall_score, roc_curve
from sklearn.preprocessing import label_binarize


def calculate_weighted_classification_metrics(
    y_true: Iterable,
    y_pred: Iterable,
) -> dict[str, float]:
    y_true_np = np.asarray(list(y_true))
    y_pred_np = np.asarray(list(y_pred))
    return {
        "accuracy": float(accuracy_score(y_true_np, y_pred_np)),
        "precision": float(precision_score(y_true_np, y_pred_np, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true_np, y_pred_np, average="weighted", zero_division=0)),
        "f1_score": float(f1_score(y_true_np, y_pred_np, average="weighted", zero_division=0)),
    }


def _parse_output_row(value):
    if isinstance(value, str):
        value = ast.literal_eval(value)
    return np.asarray(value, dtype=float)


def _to_probability_matrix(outputs: Iterable) -> np.ndarray:
    if isinstance(outputs, pd.Series):
        rows = outputs.tolist()
    else:
        rows = list(outputs)

    matrix = np.vstack([_parse_output_row(row) for row in rows])
    return softmax(matrix, axis=1)


def multiclass_auc_from_outputs(outputs: Iterable, labels: Iterable) -> dict[str, object]:
    probabilities = _to_probability_matrix(outputs)
    y_true = np.asarray(list(labels))

    n_classes = probabilities.shape[1]
    y_true_bin = label_binarize(y_true, classes=np.arange(n_classes))

    class_auc: dict[int, float] = {}
    fpr: dict[object, np.ndarray] = {}
    tpr: dict[object, np.ndarray] = {}

    for class_idx in range(n_classes):
        class_fpr, class_tpr, _ = roc_curve(
            y_true_bin[:, class_idx],
            probabilities[:, class_idx],
            drop_intermediate=False,
        )
        fpr[class_idx] = class_fpr
        tpr[class_idx] = class_tpr
        class_auc[class_idx] = float(auc(class_fpr, class_tpr))

    micro_fpr, micro_tpr, _ = roc_curve(
        y_true_bin.ravel(),
        probabilities.ravel(),
        drop_intermediate=False,
    )
    fpr["micro"] = micro_fpr
    tpr["micro"] = micro_tpr

    micro_auc = float(auc(micro_fpr, micro_tpr))
    macro_auc = float(np.mean(list(class_auc.values())))

    return {
        "probabilities": probabilities,
        "class_auc": class_auc,
        "macro_auc": macro_auc,
        "micro_auc": micro_auc,
        "fpr": fpr,
        "tpr": tpr,
    }


def _to_numeric_array(value) -> np.ndarray | None:
    parsed_value = value
    if isinstance(parsed_value, str):
        try:
            parsed_value = ast.literal_eval(parsed_value)
        except (ValueError, SyntaxError):
            return None

    if isinstance(parsed_value, (list, tuple, np.ndarray, pd.Series)):
        try:
            return np.asarray(parsed_value, dtype=float)
        except (TypeError, ValueError):
            return None

    return None


def extract_micro_curve(curve_container) -> np.ndarray | None:
    parsed_container = curve_container
    if isinstance(parsed_container, str):
        try:
            parsed_container = ast.literal_eval(parsed_container)
        except (ValueError, SyntaxError):
            return None

    if not isinstance(parsed_container, dict):
        return None

    return _to_numeric_array(parsed_container.get("micro"))


def auc_from_curve_arrays(fpr_values, tpr_values) -> float:
    fpr_arr = _to_numeric_array(fpr_values)
    tpr_arr = _to_numeric_array(tpr_values)
    if fpr_arr is None or tpr_arr is None:
        return float("nan")

    if len(fpr_arr) == 0 or len(tpr_arr) == 0 or len(fpr_arr) != len(tpr_arr):
        return float("nan")

    sort_idx = np.argsort(fpr_arr)
    return float(auc(fpr_arr[sort_idx], tpr_arr[sort_idx]))
