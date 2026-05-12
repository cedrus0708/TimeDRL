import argparse
import csv
from pathlib import Path
from typing import Dict, Optional

import numpy as np


def load_npz(path: str | Path) -> Dict[str, np.ndarray]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def get_score_key(data: Dict[str, np.ndarray], preferred: Optional[str], candidates):
    if preferred is not None:
        if preferred not in data:
            raise KeyError(f"Score key not found: {preferred}")
        return preferred

    for key in candidates:
        if key in data:
            return key

    return None


def write_top_scores_csv(
    scores: np.ndarray,
    output_path: str | Path,
    top_k: int = 100,
    sample_index: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
) -> None:
    scores = np.asarray(scores).reshape(-1).astype(np.float32)
    order = np.argsort(-scores)[:top_k]

    if sample_index is not None:
        sample_index = np.asarray(sample_index).reshape(-1)

    if labels is not None:
        labels = np.asarray(labels).reshape(-1)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "rank",
            "row_index",
            "sample_index",
            "score",
            "label",
        ]

        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for rank, row_idx in enumerate(order, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "row_index": int(row_idx),
                    "sample_index": (
                        int(sample_index[row_idx])
                        if sample_index is not None
                        and row_idx < sample_index.shape[0]
                        else int(row_idx)
                    ),
                    "score": float(scores[row_idx]),
                    "label": (
                        int(labels[row_idx])
                        if labels is not None and row_idx < labels.shape[0]
                        else ""
                    ),
                }
            )


def create_score_distribution_html(
    scores: Dict[str, np.ndarray],
    labels: Optional[Dict[str, np.ndarray]],
    output_path: str | Path,
    instance_score_key: Optional[str],
    timestamp_score_key: Optional[str],
) -> None:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError as exc:
        raise ImportError(
            "plotly is required for HTML visualization. Install with: pip install plotly"
        ) from exc

    subplot_titles = []
    rows = 0

    if instance_score_key is not None:
        rows += 1
        subplot_titles.append(f"Instance scores: {instance_score_key}")

    if timestamp_score_key is not None:
        rows += 1
        subplot_titles.append(f"Timestamp scores: {timestamp_score_key}")

    if rows == 0:
        raise ValueError("No score keys were provided.")

    fig = make_subplots(
        rows=rows,
        cols=1,
        subplot_titles=subplot_titles,
    )

    current_row = 1

    if instance_score_key is not None:
        instance_scores = np.asarray(scores[instance_score_key]).reshape(-1)

        fig.add_trace(
            go.Histogram(
                x=instance_scores,
                nbinsx=80,
                name="instance scores",
                opacity=0.8,
            ),
            row=current_row,
            col=1,
        )

        if "instance_threshold" in scores:
            threshold = float(np.asarray(scores["instance_threshold"]).reshape(-1)[0])
            fig.add_vline(
                x=threshold,
                line_dash="dash",
                annotation_text=f"threshold={threshold:.4f}",
                row=current_row,
                col=1,
            )

        current_row += 1

    if timestamp_score_key is not None:
        timestamp_scores = np.asarray(scores[timestamp_score_key]).reshape(-1)

        fig.add_trace(
            go.Histogram(
                x=timestamp_scores,
                nbinsx=80,
                name="timestamp scores",
                opacity=0.8,
            ),
            row=current_row,
            col=1,
        )

        if "timestamp_threshold" in scores:
            threshold = float(np.asarray(scores["timestamp_threshold"]).reshape(-1)[0])
            fig.add_vline(
                x=threshold,
                line_dash="dash",
                annotation_text=f"threshold={threshold:.4f}",
                row=current_row,
                col=1,
            )

    fig.update_layout(
        title="Anomaly score distributions",
        height=max(400, rows * 400),
        showlegend=False,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.write_html(str(output_path))


def create_sample_timestamp_heatmap_html(
    scores: Dict[str, np.ndarray],
    sample_id: int,
    output_path: str | Path,
) -> None:
    """
    Create channel x patch heatmap for one sample.

    Requires:
        timestamp_scores
        timestamp_sample_index
        timestamp_channel_index
        timestamp_patch_index
    """
    required = [
        "timestamp_scores",
        "timestamp_sample_index",
        "timestamp_channel_index",
        "timestamp_patch_index",
    ]

    for key in required:
        if key not in scores:
            raise KeyError(f"Missing key for heatmap: {key}")

    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(
            "plotly is required for HTML visualization. Install with: pip install plotly"
        ) from exc

    timestamp_scores = np.asarray(scores["timestamp_scores"]).reshape(-1)
    sample_index = np.asarray(scores["timestamp_sample_index"]).reshape(-1).astype(np.int64)
    channel_index = np.asarray(scores["timestamp_channel_index"]).reshape(-1).astype(np.int64)
    patch_index = np.asarray(scores["timestamp_patch_index"]).reshape(-1).astype(np.int64)

    mask = sample_index == int(sample_id)

    if not np.any(mask):
        raise ValueError(f"No timestamp scores found for sample_id={sample_id}")

    sample_scores = timestamp_scores[mask]
    sample_channels = channel_index[mask]
    sample_patches = patch_index[mask]

    if np.all(sample_channels == -1):
        unique_channels = np.array([0], dtype=np.int64)
        display_channels = np.zeros_like(sample_channels)
    else:
        unique_channels = np.unique(sample_channels[sample_channels >= 0])
        display_channels = sample_channels

    unique_patches = np.unique(sample_patches)

    heatmap = np.full(
        shape=(unique_channels.size, unique_patches.size),
        fill_value=np.nan,
        dtype=np.float32,
    )

    channel_to_row = {int(c): row for row, c in enumerate(unique_channels)}
    patch_to_col = {int(p): col for col, p in enumerate(unique_patches)}

    for score, channel, patch in zip(sample_scores, display_channels, sample_patches):
        channel = int(channel)
        patch = int(patch)

        if channel == -1:
            channel = 0

        row = channel_to_row[channel]
        col = patch_to_col[patch]

        if np.isnan(heatmap[row, col]):
            heatmap[row, col] = float(score)
        else:
            heatmap[row, col] = max(float(score), float(heatmap[row, col]))

    fig = go.Figure(
        data=go.Heatmap(
            z=heatmap,
            x=[f"patch_{int(p)}" for p in unique_patches],
            y=[f"channel_{int(c)}" for c in unique_channels],
            colorbar={"title": "score"},
        )
    )

    fig.update_layout(
        title=f"Timestamp anomaly heatmap for sample {sample_id}",
        xaxis_title="Patch",
        yaxis_title="Channel",
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.write_html(str(output_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create anomaly score visualization files."
    )

    parser.add_argument("--scores_path", type=str, required=True)
    parser.add_argument("--labels_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./score_visualizations")
    parser.add_argument("--output_name", type=str, default=None)

    parser.add_argument("--instance_score_key", type=str, default=None)
    parser.add_argument("--timestamp_score_key", type=str, default=None)
    parser.add_argument("--top_k", type=int, default=100)

    parser.add_argument(
        "--sample_id",
        type=int,
        default=None,
        help="Optional sample id for timestamp heatmap.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    scores_path = Path(args.scores_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_name = args.output_name or scores_path.stem

    scores = load_npz(scores_path)
    labels = load_npz(args.labels_path) if args.labels_path is not None else None

    instance_score_key = get_score_key(
        scores,
        args.instance_score_key,
        candidates=[
            "sample_instance_score_max",
            "sample_instance_score_mean",
            "instance_scores",
        ],
    )

    timestamp_score_key = get_score_key(
        scores,
        args.timestamp_score_key,
        candidates=[
            "timestamp_scores",
            "sample_timestamp_score_max",
            "sample_timestamp_score_mean",
        ],
    )

    create_score_distribution_html(
        scores=scores,
        labels=labels,
        output_path=output_dir / f"{output_name}.score_distributions.html",
        instance_score_key=instance_score_key,
        timestamp_score_key=timestamp_score_key,
    )

    if instance_score_key is not None:
        instance_labels = None
        if labels is not None and "sample_label" in labels:
            instance_labels = labels["sample_label"]

        write_top_scores_csv(
            scores=scores[instance_score_key],
            output_path=output_dir / f"{output_name}.top_instance_scores.csv",
            top_k=args.top_k,
            sample_index=None,
            labels=instance_labels,
        )

    if timestamp_score_key is not None:
        timestamp_labels = None
        if labels is not None and "timestamp_label" in labels:
            timestamp_labels = labels["timestamp_label"]

        write_top_scores_csv(
            scores=scores[timestamp_score_key],
            output_path=output_dir / f"{output_name}.top_timestamp_scores.csv",
            top_k=args.top_k,
            sample_index=scores.get("timestamp_sample_index"),
            labels=timestamp_labels,
        )

    if args.sample_id is not None:
        create_sample_timestamp_heatmap_html(
            scores=scores,
            sample_id=args.sample_id,
            output_path=output_dir / f"{output_name}.sample_{args.sample_id}_heatmap.html",
        )

    print(f"Saved visualizations to: {output_dir}")


if __name__ == "__main__":
    main()