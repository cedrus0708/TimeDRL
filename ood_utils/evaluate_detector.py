import argparse
import csv
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

try:
    from ood_utils.metrics import (
        evaluate_binary_scores,
        evaluate_by_ood_group,
        flatten_metrics_dict,
        save_metrics_json,
    )
except ImportError:
    from metrics import (
        evaluate_binary_scores,
        evaluate_by_ood_group,
        flatten_metrics_dict,
        save_metrics_json,
    )


def load_npz(path: str | Path) -> Dict[str, np.ndarray]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def get_optional_array(
    data: Dict[str, np.ndarray],
    key: Optional[str],
) -> Optional[np.ndarray]:
    if key is None:
        return None
    return data.get(key)


def auto_select_instance_score_key(scores: Dict[str, np.ndarray]) -> Optional[str]:
    candidates = [
        "sample_instance_score_max",
        "sample_instance_score_mean",
        "instance_scores",
    ]

    for key in candidates:
        if key in scores:
            return key

    return None


def auto_select_timestamp_score_key(scores: Dict[str, np.ndarray]) -> Optional[str]:
    candidates = [
        "timestamp_scores",
        "sample_timestamp_score_max",
        "sample_timestamp_score_mean",
    ]

    for key in candidates:
        if key in scores:
            return key

    return None


def derive_binary_label_from_group(group_labels: np.ndarray) -> np.ndarray:
    """
    Convert group labels to binary labels.

    Expected:
        0 = ID / normal
        1 = near-OOD
        2 = far-OOD
        -1 = ignored
    """
    group_labels = np.asarray(group_labels).reshape(-1).astype(np.int64)

    y = np.full_like(group_labels, fill_value=-1, dtype=np.int64)
    y[group_labels == 0] = 0
    y[group_labels > 0] = 1

    return y


def select_threshold_for_score_key(
    scores: Dict[str, np.ndarray],
    score_key: str,
) -> Optional[float]:
    if "instance" in score_key and "instance_threshold" in scores:
        return float(np.asarray(scores["instance_threshold"]).reshape(-1)[0])

    if "timestamp" in score_key and "timestamp_threshold" in scores:
        return float(np.asarray(scores["timestamp_threshold"]).reshape(-1)[0])

    return None


def write_flat_metrics_csv(
    summary: Dict[str, Any],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for section_name, section_metrics in summary.items():
        if not isinstance(section_metrics, dict):
            continue

        flat = flatten_metrics_dict(section_metrics)

        for metric_name, value in flat.items():
            rows.append(
                {
                    "section": section_name,
                    "metric": metric_name,
                    "value": value,
                }
            )

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["section", "metric", "value"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_top_cases_csv(
    y_true: np.ndarray,
    scores: np.ndarray,
    output_path: str | Path,
    top_k: int = 50,
    sample_index: Optional[np.ndarray] = None,
    group_labels: Optional[np.ndarray] = None,
) -> None:
    y_true = np.asarray(y_true).reshape(-1).astype(np.int64)
    scores = np.asarray(scores).reshape(-1).astype(np.float32)

    valid = np.isin(y_true, [0, 1]) & np.isfinite(scores)

    y_true = y_true[valid]
    scores = scores[valid]

    if sample_index is not None:
        sample_index = np.asarray(sample_index).reshape(-1)[valid]

    if group_labels is not None:
        group_labels = np.asarray(group_labels).reshape(-1)[valid]

    order = np.argsort(-scores)[:top_k]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "rank",
            "row_index",
            "sample_index",
            "score",
            "y_true",
            "group_label",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for rank, row_index in enumerate(order, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "row_index": int(row_index),
                    "sample_index": (
                        int(sample_index[row_index])
                        if sample_index is not None
                        else int(row_index)
                    ),
                    "score": float(scores[row_index]),
                    "y_true": int(y_true[row_index]),
                    "group_label": (
                        int(group_labels[row_index])
                        if group_labels is not None
                        else ""
                    ),
                }
            )


def evaluate_level(
    level_name: str,
    scores: Dict[str, np.ndarray],
    labels: Dict[str, np.ndarray],
    score_key: str,
    label_key: Optional[str],
    group_key: Optional[str],
    output_dir: Path,
    top_k: int,
) -> Dict[str, Any]:
    score_values = np.asarray(scores[score_key]).reshape(-1).astype(np.float32)

    if label_key is not None and label_key in labels:
        y_true = np.asarray(labels[label_key]).reshape(-1).astype(np.int64)
    elif group_key is not None and group_key in labels:
        y_true = derive_binary_label_from_group(labels[group_key])
    else:
        raise KeyError(
            f"No label was found for level '{level_name}'. "
            f"Tried label_key={label_key}, group_key={group_key}."
        )

    if y_true.shape[0] != score_values.shape[0]:
        raise ValueError(
            f"{level_name}: label/score length mismatch. "
            f"{y_true.shape[0]} labels vs {score_values.shape[0]} scores. "
            f"score_key={score_key}, label_key={label_key}"
        )

    threshold = select_threshold_for_score_key(scores, score_key)

    metrics = evaluate_binary_scores(
        y_true=y_true,
        scores=score_values,
        threshold=threshold,
    )

    if group_key is not None and group_key in labels:
        group_labels = np.asarray(labels[group_key]).reshape(-1).astype(np.int64)

        metrics["by_ood_group"] = evaluate_by_ood_group(
            y_true=y_true,
            scores=score_values,
            group_labels=group_labels,
            threshold=threshold,
        )
    else:
        group_labels = None

    sample_index = None

    sample_index = None

    if level_name == "instance":
        # If we evaluate sample-level aggregated scores, e.g.
        # sample_instance_score_max / sample_instance_score_mean,
        # then the score array is already [N_samples], so use 0..N-1.
        if score_key.startswith("sample_"):
            sample_index = np.arange(score_values.shape[0], dtype=np.int64)
        else:
            # Raw instance-level scores may be [N_samples * C],
            # so then instance_sample_index is valid.
            sample_index = labels.get("instance_sample_index")
            if sample_index is None:
                sample_index = scores.get("instance_sample_index")

    if level_name == "timestamp":
        # Raw timestamp_scores are vector-level, so timestamp_sample_index matches them.
        if score_key == "timestamp_scores":
            sample_index = labels.get("timestamp_sample_index")
            if sample_index is None:
                sample_index = scores.get("timestamp_sample_index")
        else:
            # Aggregated timestamp sample scores are already [N_samples].
            sample_index = np.arange(score_values.shape[0], dtype=np.int64)
            
    write_top_cases_csv(
        y_true=y_true,
        scores=score_values,
        output_path=output_dir / f"{level_name}_top_{top_k}_scores.csv",
        top_k=top_k,
        sample_index=sample_index,
        group_labels=group_labels,
    )

    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate anomaly/OOD detector scores against ground truth labels."
    )

    parser.add_argument(
        "--scores_path",
        type=str,
        required=True,
        help="Path to detector scores .npz.",
    )
    parser.add_argument(
        "--labels_path",
        type=str,
        required=True,
        help="Path to evaluation labels .npz.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./evaluation_reports",
        help="Output directory.",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="Output file stem. Defaults to scores file stem.",
    )

    parser.add_argument(
        "--instance_score_key",
        type=str,
        default=None,
        help="Score key for instance/sample-level evaluation.",
    )
    parser.add_argument(
        "--timestamp_score_key",
        type=str,
        default=None,
        help="Score key for timestamp/patch-level evaluation.",
    )

    parser.add_argument(
        "--sample_label_key",
        type=str,
        default="sample_label",
        help="Binary sample label key. 0=ID, 1=OOD/anomaly.",
    )
    parser.add_argument(
        "--sample_group_key",
        type=str,
        default="sample_ood_type",
        help="Sample group key. 0=ID, 1=near-OOD, 2=far-OOD.",
    )

    parser.add_argument(
        "--timestamp_label_key",
        type=str,
        default="timestamp_label",
        help="Binary timestamp/patch label key.",
    )
    parser.add_argument(
        "--timestamp_group_key",
        type=str,
        default="timestamp_ood_type",
        help="Timestamp group key.",
    )

    parser.add_argument(
        "--skip_instance",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--skip_timestamp",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    scores_path = Path(args.scores_path)
    labels_path = Path(args.labels_path)

    scores = load_npz(scores_path)
    labels = load_npz(labels_path)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_name = args.output_name
    if output_name is None:
        output_name = scores_path.stem

    summary: Dict[str, Any] = {
        "input": {
            "scores_path": str(scores_path.resolve()),
            "labels_path": str(labels_path.resolve()),
        }
    }

    if not args.skip_instance:
        instance_score_key = args.instance_score_key or auto_select_instance_score_key(scores)

        if instance_score_key is None:
            print("Skipping instance-level evaluation: no instance score key found.")
        else:
            print(f"Evaluating instance level with score key: {instance_score_key}")

            summary["instance_level"] = evaluate_level(
                level_name="instance",
                scores=scores,
                labels=labels,
                score_key=instance_score_key,
                label_key=args.sample_label_key,
                group_key=args.sample_group_key,
                output_dir=output_dir,
                top_k=args.top_k,
            )

    if not args.skip_timestamp:
        timestamp_score_key = args.timestamp_score_key or auto_select_timestamp_score_key(scores)

        if timestamp_score_key is None:
            print("Skipping timestamp-level evaluation: no timestamp score key found.")
        elif (
            args.timestamp_label_key not in labels
            and args.timestamp_group_key not in labels
        ):
            print("Skipping timestamp-level evaluation: no timestamp labels found.")
        else:
            print(f"Evaluating timestamp level with score key: {timestamp_score_key}")

            summary["timestamp_level"] = evaluate_level(
                level_name="timestamp",
                scores=scores,
                labels=labels,
                score_key=timestamp_score_key,
                label_key=args.timestamp_label_key,
                group_key=args.timestamp_group_key,
                output_dir=output_dir,
                top_k=args.top_k,
            )

    json_path = output_dir / f"{output_name}.metrics_summary.json"
    csv_path = output_dir / f"{output_name}.metrics_summary.csv"

    save_metrics_json(summary, json_path)
    write_flat_metrics_csv(summary, csv_path)

    print("\nSaved evaluation report:")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")


if __name__ == "__main__":
    main()