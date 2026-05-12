import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


def to_numpy_1d(x: Any) -> np.ndarray:
    x = np.asarray(x)
    return x.reshape(-1)


def filter_valid_binary_labels(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Keep only binary labels 0/1.

    Values like -1 are treated as ignored labels.
    """
    y_true = to_numpy_1d(y_true).astype(np.int64)
    scores = to_numpy_1d(scores).astype(np.float32)

    if y_true.shape[0] != scores.shape[0]:
        raise ValueError(
            f"Length mismatch: y_true={y_true.shape[0]}, scores={scores.shape[0]}"
        )

    valid_mask = np.isin(y_true, [0, 1]) & np.isfinite(scores)
    return y_true[valid_mask], scores[valid_mask]


def average_ranks(x: np.ndarray) -> np.ndarray:
    """
    Average ranks for ties. Ranks are 1-based.
    """
    x = np.asarray(x)
    order = np.argsort(x)
    sorted_x = x[order]

    ranks = np.empty_like(sorted_x, dtype=np.float64)

    start = 0
    n = len(sorted_x)

    while start < n:
        end = start + 1
        while end < n and sorted_x[end] == sorted_x[start]:
            end += 1

        avg_rank = (start + 1 + end) / 2.0
        ranks[start:end] = avg_rank
        start = end

    output = np.empty_like(ranks)
    output[order] = ranks
    return output


def roc_auc_score_manual(y_true: np.ndarray, scores: np.ndarray) -> Optional[float]:
    """
    Manual AUROC implementation.

    Assumes higher score = more anomalous / more positive.
    """
    y_true, scores = filter_valid_binary_labels(y_true, scores)

    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))

    if n_pos == 0 or n_neg == 0:
        return None

    ranks = average_ranks(scores)
    pos_rank_sum = float(np.sum(ranks[y_true == 1]))

    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def average_precision_score_manual(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> Optional[float]:
    """
    Manual Average Precision / AUPRC implementation.

    Assumes higher score = more anomalous / more positive.
    """
    y_true, scores = filter_valid_binary_labels(y_true, scores)

    n_pos = int(np.sum(y_true == 1))
    if n_pos == 0:
        return None

    order = np.argsort(-scores)
    y_sorted = y_true[order]

    tp_cumsum = np.cumsum(y_sorted == 1)
    ranks = np.arange(1, len(y_sorted) + 1)

    precision_at_k = tp_cumsum / ranks

    ap = np.sum(precision_at_k[y_sorted == 1]) / n_pos
    return float(ap)


def confusion_from_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> Dict[str, int]:
    y_true, scores = filter_valid_binary_labels(y_true, scores)

    y_pred = (scores > threshold).astype(np.int64)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def threshold_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    y_true, scores = filter_valid_binary_labels(y_true, scores)

    if y_true.size == 0:
        return {
            "threshold": float(threshold),
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "specificity": None,
            "confusion_matrix": {
                "tp": 0,
                "fp": 0,
                "tn": 0,
                "fn": 0,
            },
        }

    cm = confusion_from_threshold(y_true, scores, threshold)

    tp = cm["tp"]
    fp = cm["fp"]
    tn = cm["tn"]
    fn = cm["fn"]

    accuracy = (tp + tn) / max(tp + fp + tn + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2.0 * precision * recall / (precision + recall)

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "confusion_matrix": cm,
    }


def best_f1_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    max_candidates: int = 512,
) -> Dict[str, Any]:
    """
    Find a threshold that maximizes F1 on the provided labels.

    This is for evaluation/reporting only. Do not use this as the main deployed
    threshold unless you have a separate validation set.
    """
    y_true, scores = filter_valid_binary_labels(y_true, scores)

    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        return {
            "threshold": None,
            "f1": None,
            "precision": None,
            "recall": None,
            "accuracy": None,
        }

    unique_scores = np.unique(scores)

    if unique_scores.size > max_candidates:
        quantiles = np.linspace(0.0, 1.0, max_candidates)
        thresholds = np.quantile(unique_scores, quantiles)
    else:
        thresholds = unique_scores

    best = None

    for threshold in thresholds:
        metrics = threshold_metrics(y_true, scores, float(threshold))

        if best is None or metrics["f1"] > best["f1"]:
            best = metrics

    return best


def fpr_at_tpr(
    y_true: np.ndarray,
    scores: np.ndarray,
    target_tpr: float = 0.95,
) -> Optional[float]:
    """
    Compute FPR at target TPR.

    Assumes higher score = more anomalous / more positive.
    """
    y_true, scores = filter_valid_binary_labels(y_true, scores)

    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))

    if n_pos == 0 or n_neg == 0:
        return None

    order = np.argsort(-scores)
    y_sorted = y_true[order]

    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)

    tpr = tp / n_pos
    fpr = fp / n_neg

    valid = tpr >= target_tpr

    if not np.any(valid):
        return None

    return float(np.min(fpr[valid]))


def precision_at_k(
    y_true: np.ndarray,
    scores: np.ndarray,
    k: int,
) -> Optional[float]:
    y_true, scores = filter_valid_binary_labels(y_true, scores)

    if y_true.size == 0:
        return None

    k = min(int(k), y_true.size)

    if k <= 0:
        return None

    top_indices = np.argsort(-scores)[:k]
    return float(np.mean(y_true[top_indices] == 1))


def evaluate_binary_scores(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: Optional[float] = None,
    threshold_quantile_if_missing: float = 0.95,
    target_tpr: float = 0.95,
) -> Dict[str, Any]:
    """
    Main binary anomaly/OOD evaluation function.

    y_true:
        0 = ID / normal
        1 = anomaly / OOD

    scores:
        higher = more anomalous
    """
    y_true, scores = filter_valid_binary_labels(y_true, scores)

    output: Dict[str, Any] = {
        "n_total": int(y_true.size),
        "n_normal": int(np.sum(y_true == 0)),
        "n_anomaly": int(np.sum(y_true == 1)),
        "score_mean": float(np.mean(scores)) if scores.size else None,
        "score_std": float(np.std(scores)) if scores.size else None,
        "score_min": float(np.min(scores)) if scores.size else None,
        "score_max": float(np.max(scores)) if scores.size else None,
        "normal_score_mean": (
            float(np.mean(scores[y_true == 0])) if np.any(y_true == 0) else None
        ),
        "anomaly_score_mean": (
            float(np.mean(scores[y_true == 1])) if np.any(y_true == 1) else None
        ),
    }

    if y_true.size == 0 or len(np.unique(y_true)) < 2:
        output.update(
            {
                "auroc": None,
                "auprc": None,
                "fpr_at_95_tpr": None,
                "threshold_metrics": None,
                "best_f1_threshold_metrics": None,
                "precision_at_10": None,
                "precision_at_50": None,
                "precision_at_100": None,
            }
        )
        return output

    output["auroc"] = roc_auc_score_manual(y_true, scores)
    output["auprc"] = average_precision_score_manual(y_true, scores)
    output["fpr_at_95_tpr"] = fpr_at_tpr(
        y_true,
        scores,
        target_tpr=target_tpr,
    )

    if threshold is None:
        threshold = float(np.quantile(scores[y_true == 0], threshold_quantile_if_missing))

    output["threshold_metrics"] = threshold_metrics(y_true, scores, threshold)
    output["best_f1_threshold_metrics"] = best_f1_threshold(y_true, scores)

    output["precision_at_10"] = precision_at_k(y_true, scores, 10)
    output["precision_at_50"] = precision_at_k(y_true, scores, 50)
    output["precision_at_100"] = precision_at_k(y_true, scores, 100)

    return output


def evaluate_by_ood_group(
    y_true: np.ndarray,
    scores: np.ndarray,
    group_labels: np.ndarray,
    group_names: Optional[Dict[int, str]] = None,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Evaluate ID vs each positive OOD group.

    Expected group_labels:
        0 = ID / normal
        1 = near-OOD
        2 = far-OOD
        other positive values = custom OOD groups
        -1 = ignored
    """
    y_true = to_numpy_1d(y_true).astype(np.int64)
    scores = to_numpy_1d(scores).astype(np.float32)
    group_labels = to_numpy_1d(group_labels).astype(np.int64)

    if not (y_true.shape[0] == scores.shape[0] == group_labels.shape[0]):
        raise ValueError(
            "Length mismatch between y_true, scores and group_labels: "
            f"{y_true.shape[0]}, {scores.shape[0]}, {group_labels.shape[0]}"
        )

    if group_names is None:
        group_names = {
            0: "id",
            1: "near_ood",
            2: "far_ood",
        }

    output: Dict[str, Any] = {}

    positive_groups = sorted(
        int(g)
        for g in np.unique(group_labels)
        if int(g) > 0
    )

    for group_id in positive_groups:
        mask = (group_labels == 0) | (group_labels == group_id)
        name = group_names.get(group_id, f"group_{group_id}")

        group_y = (group_labels[mask] == group_id).astype(np.int64)
        group_scores = scores[mask]

        output[name] = evaluate_binary_scores(
            y_true=group_y,
            scores=group_scores,
            threshold=threshold,
        )

    return output


def flatten_metrics_dict(
    metrics: Dict[str, Any],
    prefix: str = "",
) -> Dict[str, Any]:
    """
    Flatten nested metric dict into one-level key-value dictionary.
    """
    flat: Dict[str, Any] = {}

    for key, value in metrics.items():
        new_key = f"{prefix}.{key}" if prefix else key

        if isinstance(value, dict):
            flat.update(flatten_metrics_dict(value, prefix=new_key))
        else:
            flat[new_key] = value

    return flat


def make_json_serializable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): make_json_serializable(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]

    if isinstance(obj, tuple):
        return [make_json_serializable(v) for v in obj]

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, np.integer):
        return int(obj)

    if isinstance(obj, np.floating):
        value = float(obj)
        if not np.isfinite(value):
            return None
        return value

    if isinstance(obj, float):
        if not np.isfinite(obj):
            return None

    return obj


def save_metrics_json(
    metrics: Dict[str, Any],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            make_json_serializable(metrics),
            f,
            indent=2,
            ensure_ascii=False,
        )