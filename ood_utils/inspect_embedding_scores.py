import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect and visualize embedding anomaly scores."
    )

    parser.add_argument(
        "--scores_path",
        type=str,
        required=True,
        help="Path to the .scores.npz file produced by embedding_detector.py.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./embedding_score_reports",
        help="Directory where reports and plots will be saved.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Number of top anomalous items to export.",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Optional filename prefix for generated outputs.",
    )
    parser.add_argument(
        "--hist_bins",
        type=int,
        default=80,
        help="Number of histogram bins.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for saved plots.",
    )
    parser.add_argument(
        "--max_plot_points",
        type=int,
        default=20000,
        help="Maximum number of points shown in line plots. Larger arrays are downsampled.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        default=False,
        help="Show plots interactively after saving.",
    )

    return parser.parse_args()


def load_npz(path: str | Path) -> Dict[str, np.ndarray]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Scores file not found: {path}")

    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def get_optional_array(data: Dict[str, np.ndarray], key: str) -> Optional[np.ndarray]:
    if key not in data:
        return None
    return np.asarray(data[key])


def get_optional_scalar(data: Dict[str, np.ndarray], key: str) -> Optional[float]:
    if key not in data:
        return None

    value = np.asarray(data[key])

    if value.size == 0:
        return None

    return float(value.reshape(-1)[0])


def summarize_scores(
    scores: Optional[np.ndarray],
    predictions: Optional[np.ndarray] = None,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    if scores is None or scores.size == 0:
        return {
            "available": False,
            "count": 0,
        }

    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    finite_scores = scores[np.isfinite(scores)]

    summary: Dict[str, Any] = {
        "available": True,
        "count": int(scores.size),
        "finite_count": int(finite_scores.size),
    }

    if finite_scores.size == 0:
        summary.update(
            {
                "mean": None,
                "std": None,
                "min": None,
                "q50": None,
                "q90": None,
                "q95": None,
                "q99": None,
                "max": None,
                "threshold": threshold,
            }
        )
        return summary

    summary.update(
        {
            "mean": float(np.mean(finite_scores)),
            "std": float(np.std(finite_scores)),
            "min": float(np.min(finite_scores)),
            "q50": float(np.quantile(finite_scores, 0.50)),
            "q90": float(np.quantile(finite_scores, 0.90)),
            "q95": float(np.quantile(finite_scores, 0.95)),
            "q99": float(np.quantile(finite_scores, 0.99)),
            "max": float(np.max(finite_scores)),
            "threshold": threshold,
        }
    )

    if threshold is not None:
        above = finite_scores > threshold
        summary["above_threshold_count"] = int(np.sum(above))
        summary["above_threshold_rate"] = float(np.mean(above))

    if predictions is not None and predictions.size == scores.size:
        predictions = np.asarray(predictions).reshape(-1)
        summary["prediction_anomaly_count"] = int(np.sum(predictions == 1))
        summary["prediction_anomaly_rate"] = float(np.mean(predictions == 1))

    return summary


def write_json(data: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def write_csv(rows: List[Dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write("")
        return

    fieldnames = list(rows[0].keys())

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def downsample_for_plot(
    y: np.ndarray,
    max_points: int,
) -> Tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y).reshape(-1)
    n = y.size

    if n <= max_points:
        x = np.arange(n)
        return x, y

    selected = np.linspace(0, n - 1, max_points).astype(np.int64)
    x = selected
    y_downsampled = y[selected]

    return x, y_downsampled


def plot_histogram(
    scores: Optional[np.ndarray],
    threshold: Optional[float],
    title: str,
    output_path: str | Path,
    bins: int = 80,
    dpi: int = 150,
    show: bool = False,
) -> None:
    if scores is None or scores.size == 0:
        print(f"Skipping histogram, no scores available: {title}")
        return

    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    scores = scores[np.isfinite(scores)]

    if scores.size == 0:
        print(f"Skipping histogram, no finite scores available: {title}")
        return

    plt.figure(figsize=(10, 5))
    plt.hist(scores, bins=bins)

    if threshold is not None:
        plt.axvline(threshold, linestyle="--", linewidth=2, label=f"threshold = {threshold:.6f}")
        plt.legend()

    plt.title(title)
    plt.xlabel("Anomaly score")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)

    if show:
        plt.show()

    plt.close()


def plot_score_line(
    scores: Optional[np.ndarray],
    threshold: Optional[float],
    title: str,
    output_path: str | Path,
    ylabel: str,
    max_plot_points: int = 20000,
    dpi: int = 150,
    show: bool = False,
) -> None:
    if scores is None or scores.size == 0:
        print(f"Skipping line plot, no scores available: {title}")
        return

    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    x, y = downsample_for_plot(scores, max_points=max_plot_points)

    plt.figure(figsize=(12, 5))
    plt.plot(x, y, linewidth=1)

    if threshold is not None:
        plt.axhline(threshold, linestyle="--", linewidth=2, label=f"threshold = {threshold:.6f}")
        plt.legend()

    plt.title(title)
    plt.xlabel("Index")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi)

    if show:
        plt.show()

    plt.close()


def top_indices(scores: Optional[np.ndarray], top_k: int) -> np.ndarray:
    if scores is None or scores.size == 0:
        return np.array([], dtype=np.int64)

    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    finite_mask = np.isfinite(scores)

    if not finite_mask.any():
        return np.array([], dtype=np.int64)

    finite_indices = np.where(finite_mask)[0]
    finite_scores = scores[finite_indices]

    k = min(top_k, finite_scores.size)
    local_top = np.argpartition(-finite_scores, kth=k - 1)[:k]
    local_top_sorted = local_top[np.argsort(-finite_scores[local_top])]

    return finite_indices[local_top_sorted].astype(np.int64)


def safe_array_value(
    array: Optional[np.ndarray],
    index: int,
    default: Any = "",
) -> Any:
    if array is None:
        return default

    array = np.asarray(array).reshape(-1)

    if index < 0 or index >= array.size:
        return default

    value = array[index]

    if isinstance(value, np.generic):
        return value.item()

    return value


def build_top_vector_rows(
    scores: Optional[np.ndarray],
    predictions: Optional[np.ndarray],
    sample_index: Optional[np.ndarray],
    channel_index: Optional[np.ndarray],
    patch_index: Optional[np.ndarray],
    top_k: int,
) -> List[Dict[str, Any]]:
    if scores is None:
        return []

    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    top = top_indices(scores, top_k=top_k)

    rows = []

    for rank, vector_index in enumerate(top, start=1):
        row = {
            "rank": rank,
            "vector_index": int(vector_index),
            "score": float(scores[vector_index]),
            "prediction": safe_array_value(predictions, int(vector_index)),
            "sample_index": safe_array_value(sample_index, int(vector_index)),
            "channel_index": safe_array_value(channel_index, int(vector_index)),
        }

        if patch_index is not None:
            row["patch_index"] = safe_array_value(patch_index, int(vector_index))

        rows.append(row)

    return rows


def build_top_sample_rows(
    scores: Optional[np.ndarray],
    top_k: int,
    score_name: str,
    threshold: Optional[float] = None,
) -> List[Dict[str, Any]]:
    if scores is None:
        return []

    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    top = top_indices(scores, top_k=top_k)

    rows = []

    for rank, sample_index in enumerate(top, start=1):
        score = float(scores[sample_index])

        row = {
            "rank": rank,
            "sample_index": int(sample_index),
            score_name: score,
        }

        if threshold is not None:
            row["above_vector_threshold"] = int(score > threshold)

        rows.append(row)

    return rows


def make_prefix(scores_path: Path, user_prefix: Optional[str]) -> str:
    if user_prefix is not None:
        return user_prefix

    name = scores_path.name

    if name.endswith(".npz"):
        name = name[:-4]

    return name


def main() -> None:
    args = parse_args()

    scores_path = Path(args.scores_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = make_prefix(scores_path, args.prefix)

    print(f"Loading scores: {scores_path}")
    data = load_npz(scores_path)

    print("\nAvailable keys:")
    for key in sorted(data.keys()):
        print(f"  {key}: {data[key].shape}")

    instance_scores = get_optional_array(data, "instance_scores")
    timestamp_scores = get_optional_array(data, "timestamp_scores")

    instance_predictions = get_optional_array(data, "instance_predictions")
    timestamp_predictions = get_optional_array(data, "timestamp_predictions")

    instance_threshold = get_optional_scalar(data, "instance_threshold")
    timestamp_threshold = get_optional_scalar(data, "timestamp_threshold")

    sample_instance_score_max = get_optional_array(data, "sample_instance_score_max")
    sample_instance_score_mean = get_optional_array(data, "sample_instance_score_mean")
    sample_timestamp_score_max = get_optional_array(data, "sample_timestamp_score_max")
    sample_timestamp_score_mean = get_optional_array(data, "sample_timestamp_score_mean")

    instance_sample_index = get_optional_array(data, "instance_sample_index")
    instance_channel_index = get_optional_array(data, "instance_channel_index")

    timestamp_sample_index = get_optional_array(data, "timestamp_sample_index")
    timestamp_channel_index = get_optional_array(data, "timestamp_channel_index")
    timestamp_patch_index = get_optional_array(data, "timestamp_patch_index")

    summary = {
        "scores_path": str(scores_path.resolve()),
        "available_keys": {
            key: list(value.shape)
            for key, value in data.items()
        },
        "instance_scores": summarize_scores(
            scores=instance_scores,
            predictions=instance_predictions,
            threshold=instance_threshold,
        ),
        "timestamp_scores": summarize_scores(
            scores=timestamp_scores,
            predictions=timestamp_predictions,
            threshold=timestamp_threshold,
        ),
        "sample_instance_score_max": summarize_scores(
            scores=sample_instance_score_max,
            threshold=instance_threshold,
        ),
        "sample_instance_score_mean": summarize_scores(
            scores=sample_instance_score_mean,
            threshold=instance_threshold,
        ),
        "sample_timestamp_score_max": summarize_scores(
            scores=sample_timestamp_score_max,
            threshold=timestamp_threshold,
        ),
        "sample_timestamp_score_mean": summarize_scores(
            scores=sample_timestamp_score_mean,
            threshold=timestamp_threshold,
        ),
    }

    summary_path = output_dir / f"{prefix}.score_summary.json"
    write_json(summary, summary_path)

    print(f"\nSaved summary: {summary_path}")

    # Histograms
    plot_histogram(
        scores=instance_scores,
        threshold=instance_threshold,
        title="Instance-level anomaly score distribution",
        output_path=output_dir / f"{prefix}.instance_score_hist.png",
        bins=args.hist_bins,
        dpi=args.dpi,
        show=args.show,
    )

    plot_histogram(
        scores=timestamp_scores,
        threshold=timestamp_threshold,
        title="Timestamp-level anomaly score distribution",
        output_path=output_dir / f"{prefix}.timestamp_score_hist.png",
        bins=args.hist_bins,
        dpi=args.dpi,
        show=args.show,
    )

    # Sample-level line plots
    plot_score_line(
        scores=sample_instance_score_max,
        threshold=instance_threshold,
        title="Sample index → max instance-level anomaly score",
        output_path=output_dir / f"{prefix}.sample_instance_score_max.png",
        ylabel="Max instance score",
        max_plot_points=args.max_plot_points,
        dpi=args.dpi,
        show=args.show,
    )

    plot_score_line(
        scores=sample_instance_score_mean,
        threshold=instance_threshold,
        title="Sample index → mean instance-level anomaly score",
        output_path=output_dir / f"{prefix}.sample_instance_score_mean.png",
        ylabel="Mean instance score",
        max_plot_points=args.max_plot_points,
        dpi=args.dpi,
        show=args.show,
    )

    plot_score_line(
        scores=sample_timestamp_score_max,
        threshold=timestamp_threshold,
        title="Sample index → max timestamp-level anomaly score",
        output_path=output_dir / f"{prefix}.sample_timestamp_score_max.png",
        ylabel="Max timestamp score",
        max_plot_points=args.max_plot_points,
        dpi=args.dpi,
        show=args.show,
    )

    plot_score_line(
        scores=sample_timestamp_score_mean,
        threshold=timestamp_threshold,
        title="Sample index → mean timestamp-level anomaly score",
        output_path=output_dir / f"{prefix}.sample_timestamp_score_mean.png",
        ylabel="Mean timestamp score",
        max_plot_points=args.max_plot_points,
        dpi=args.dpi,
        show=args.show,
    )

    # Top vector-level anomaly tables
    top_instance_rows = build_top_vector_rows(
        scores=instance_scores,
        predictions=instance_predictions,
        sample_index=instance_sample_index,
        channel_index=instance_channel_index,
        patch_index=None,
        top_k=args.top_k,
    )

    top_timestamp_rows = build_top_vector_rows(
        scores=timestamp_scores,
        predictions=timestamp_predictions,
        sample_index=timestamp_sample_index,
        channel_index=timestamp_channel_index,
        patch_index=timestamp_patch_index,
        top_k=args.top_k,
    )

    write_csv(
        top_instance_rows,
        output_dir / f"{prefix}.top_instance_vectors.csv",
    )

    write_csv(
        top_timestamp_rows,
        output_dir / f"{prefix}.top_timestamp_vectors.csv",
    )

    # Top sample-level anomaly tables
    write_csv(
        build_top_sample_rows(
            scores=sample_instance_score_max,
            top_k=args.top_k,
            score_name="sample_instance_score_max",
            threshold=instance_threshold,
        ),
        output_dir / f"{prefix}.top_sample_instance_score_max.csv",
    )

    write_csv(
        build_top_sample_rows(
            scores=sample_instance_score_mean,
            top_k=args.top_k,
            score_name="sample_instance_score_mean",
            threshold=instance_threshold,
        ),
        output_dir / f"{prefix}.top_sample_instance_score_mean.csv",
    )

    write_csv(
        build_top_sample_rows(
            scores=sample_timestamp_score_max,
            top_k=args.top_k,
            score_name="sample_timestamp_score_max",
            threshold=timestamp_threshold,
        ),
        output_dir / f"{prefix}.top_sample_timestamp_score_max.csv",
    )

    write_csv(
        build_top_sample_rows(
            scores=sample_timestamp_score_mean,
            top_k=args.top_k,
            score_name="sample_timestamp_score_mean",
            threshold=timestamp_threshold,
        ),
        output_dir / f"{prefix}.top_sample_timestamp_score_mean.csv",
    )

    print("\nSaved plots and top anomaly tables to:")
    print(f"  {output_dir.resolve()}")

    print("\nMain quick summary:")
    if instance_scores is not None:
        print(f"  instance scores:  count={instance_scores.size}, threshold={instance_threshold}")
    if timestamp_scores is not None:
        print(f"  timestamp scores: count={timestamp_scores.size}, threshold={timestamp_threshold}")

    if sample_instance_score_max is not None:
        top_idx = top_indices(sample_instance_score_max, top_k=1)
        if top_idx.size > 0:
            idx = int(top_idx[0])
            print(
                "  top sample by instance max: "
                f"sample={idx}, score={float(sample_instance_score_max[idx]):.6f}"
            )

    if sample_timestamp_score_max is not None:
        top_idx = top_indices(sample_timestamp_score_max, top_k=1)
        if top_idx.size > 0:
            idx = int(top_idx[0])
            print(
                "  top sample by timestamp max: "
                f"sample={idx}, score={float(sample_timestamp_score_max[idx]):.6f}"
            )


if __name__ == "__main__":
    main()