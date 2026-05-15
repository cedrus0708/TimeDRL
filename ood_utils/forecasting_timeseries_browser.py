"""Interactive full test-set browser for TimeDRL forecasting anomaly experiments.

This creates one Plotly HTML file where the forecasting test region is shown as a
shared-x time-series dashboard:

1. original dataset values
2. injected dataset values
3. ground-truth anomaly band
4. instance/sample score band projected back to CSV rows
5. timestamp/patch score band projected back to CSV rows
6. final model detection band
7. top-K/top100 coverage band
8. metric summary cards below the plot

The important part is the projection step: TimeDRL scores are produced for
windows/patches, while the CSV is a global time series. This script maps every
window/patch score back to the global CSV row interval it covers, then aggregates
all covering scores per row, usually by max.

Typical usage:
    python ood_utils/forecasting_timeseries_browser.py \
        --original_csv_path ./dataset/forecasting/exchange_rate/exchange_rate.csv \
        --injected_csv_path ./dataset/forecasting/exchange_rate/exchange_rate_injected.csv \
        --mask_path ./ood/eval_sets/Exchange_injected_mask.npz \
        --scores_path ./ood/embedding_detectors/EXPERIMENT.scores.npz \
        --labels_path ./ood/eval_sets/Exchange_injected_test_labels.npz \
        --metrics_path ./ood/evaluation_reports/EXPERIMENT.metrics_summary.json \
        --output_dir ./ood/score_visualizations \
        --output_name EXPERIMENT \
        --seq_len 168 \
        --top_k 100
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


DATE_COLUMN_CANDIDATES = ["date", "datetime", "time", "timestamp"]


# -----------------------------------------------------------------------------
# Loading helpers
# -----------------------------------------------------------------------------


def load_npz(path: str | Path) -> Dict[str, np.ndarray]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def load_json(path: Optional[str | Path]) -> Dict[str, Any]:
    if path is None:
        return {}

    path = Path(path)
    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def arr1d(data: Dict[str, np.ndarray], key: str, dtype: Optional[Any] = None) -> Optional[np.ndarray]:
    if key not in data:
        return None
    arr = np.asarray(data[key]).reshape(-1)
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr


def scalar_from_npz(data: Dict[str, np.ndarray], key: str) -> Optional[float]:
    if key not in data:
        return None
    arr = np.asarray(data[key]).reshape(-1)
    if arr.size == 0:
        return None
    value = float(arr[0])
    if not math.isfinite(value):
        return None
    return value


def decode_str_array(value: Any) -> List[str]:
    if value is None:
        return []
    arr = np.asarray(value).reshape(-1)
    out: List[str] = []
    for item in arr:
        if isinstance(item, bytes):
            out.append(item.decode("utf-8"))
        else:
            out.append(str(item))
    return out


# -----------------------------------------------------------------------------
# Formatting helpers
# -----------------------------------------------------------------------------


def format_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    try:
        x = float(value)
    except Exception:
        return html.escape(str(value))
    if not math.isfinite(x):
        return "—"
    return f"{x:.{digits}f}".replace(".", ",")


def format_int(value: Any) -> str:
    if value is None:
        return "—"
    try:
        return f"{int(value):,}".replace(",", " ")
    except Exception:
        return html.escape(str(value))


def html_escape(value: Any) -> str:
    return html.escape("—" if value is None else str(value))


# -----------------------------------------------------------------------------
# Data selection and alignment
# -----------------------------------------------------------------------------


def infer_date_column(df: pd.DataFrame, explicit: Optional[str]) -> Optional[str]:
    if explicit:
        if explicit not in df.columns:
            raise KeyError(f"date_column not found in CSV: {explicit}")
        return explicit

    lowered = {c.lower(): c for c in df.columns}
    for candidate in DATE_COLUMN_CANDIDATES:
        if candidate in lowered:
            return lowered[candidate]
    return None


def select_value_columns(
    original_df: pd.DataFrame,
    injected_df: pd.DataFrame,
    mask_data: Dict[str, np.ndarray],
    explicit: Optional[Sequence[str]],
    max_channels: Optional[int],
) -> List[str]:
    if explicit:
        missing = [c for c in explicit if c not in original_df.columns or c not in injected_df.columns]
        if missing:
            raise KeyError(f"value_columns missing from one of the CSV files: {missing}")
        selected = list(explicit)
    elif "value_columns" in mask_data:
        selected = [c for c in decode_str_array(mask_data["value_columns"]) if c in original_df.columns and c in injected_df.columns]
    else:
        excluded = set(DATE_COLUMN_CANDIDATES)
        selected = []
        for col in original_df.columns:
            if col.lower() in excluded:
                continue
            if col not in injected_df.columns:
                continue
            if pd.api.types.is_numeric_dtype(original_df[col]) and pd.api.types.is_numeric_dtype(injected_df[col]):
                selected.append(col)

    if not selected:
        raise ValueError("No common numeric value columns found for plotting.")

    if max_channels is not None and max_channels > 0:
        selected = selected[:max_channels]

    return selected


def infer_seq_len(args: argparse.Namespace, labels: Dict[str, np.ndarray], metrics: Dict[str, Any]) -> int:
    if args.seq_len is not None:
        return int(args.seq_len)

    # Fallback: infer from overlapping consecutive window starts if possible.
    window_start = arr1d(labels, "window_start_index", np.int64)
    if window_start is not None and window_start.size >= 2:
        # This is only a fallback; the real seq_len should be passed explicitly.
        # Using 168 is common for Exchange in your current pipeline, but avoid hardcoding.
        diffs = np.diff(np.unique(window_start))
        if diffs.size > 0:
            stride_guess = int(np.median(diffs))
            if stride_guess > 1:
                return stride_guess

    raise ValueError("--seq_len is required unless it can be safely inferred.")


def infer_test_range(
    labels: Dict[str, np.ndarray],
    mask_data: Dict[str, np.ndarray],
    seq_len: int,
    test_start_index: Optional[int],
    test_end_index: Optional[int],
    csv_len: int,
) -> Tuple[int, int]:
    if test_start_index is not None and test_end_index is not None:
        start = int(test_start_index)
        end = int(test_end_index)
        return max(0, start), min(csv_len, end)

    window_start = arr1d(labels, "window_start_index", np.int64)
    if window_start is not None and window_start.size > 0:
        valid = window_start[(window_start >= 0) & (window_start < csv_len)]
        if valid.size > 0:
            start = int(valid.min()) if test_start_index is None else int(test_start_index)
            end = int(valid.max() + seq_len) if test_end_index is None else int(test_end_index)
            return max(0, start), min(csv_len, end)

    # Fallback to injection range if window mapping is missing.
    inject_start = arr1d(mask_data, "inject_start_idx", np.int64)
    inject_end = arr1d(mask_data, "inject_end_idx", np.int64)
    if inject_start is not None and inject_end is not None and inject_start.size and inject_end.size:
        start = int(inject_start[0]) if test_start_index is None else int(test_start_index)
        end = int(inject_end[0]) if test_end_index is None else int(test_end_index)
        return max(0, start), min(csv_len, end)

    if test_start_index is not None:
        return max(0, int(test_start_index)), min(csv_len, int(test_end_index or csv_len))

    raise ValueError(
        "Cannot infer test range. The labels file should contain window_start_index, "
        "or pass --test_start_index and --test_end_index."
    )


def make_x_axis(df: pd.DataFrame, date_column: Optional[str], start: int, end: int) -> np.ndarray:
    if date_column is None:
        return np.arange(start, end)
    return df[date_column].iloc[start:end].to_numpy()


# -----------------------------------------------------------------------------
# Projection from windows/patches to global CSV rows
# -----------------------------------------------------------------------------


def aggregate_intervals_to_rows(
    starts: np.ndarray,
    ends: np.ndarray,
    values: np.ndarray,
    row_start: int,
    row_end: int,
    mode: str = "max",
) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate interval-level values to one value per global CSV row.

    Returns:
        aggregated: [row_end - row_start]
        coverage_count: [row_end - row_start]
    """
    starts = np.asarray(starts).reshape(-1).astype(np.int64)
    ends = np.asarray(ends).reshape(-1).astype(np.int64)
    values = np.asarray(values).reshape(-1).astype(np.float32)

    if not (starts.shape[0] == ends.shape[0] == values.shape[0]):
        raise ValueError(
            "starts, ends and values length mismatch: "
            f"{starts.shape[0]}, {ends.shape[0]}, {values.shape[0]}"
        )

    n_rows = int(row_end - row_start)
    if n_rows <= 0:
        raise ValueError("Invalid row range.")

    if mode == "max":
        out = np.full(n_rows, np.nan, dtype=np.float32)
        count = np.zeros(n_rows, dtype=np.int64)
        for start, end, value in zip(starts, ends, values):
            if not math.isfinite(float(value)):
                continue
            local_start = max(int(start), row_start) - row_start
            local_end = min(int(end), row_end) - row_start
            if local_start >= local_end:
                continue
            current = out[local_start:local_end]
            current_nan = np.isnan(current)
            current[current_nan] = value
            current[~current_nan] = np.maximum(current[~current_nan], value)
            count[local_start:local_end] += 1
        return out, count

    if mode == "mean":
        total = np.zeros(n_rows, dtype=np.float64)
        count = np.zeros(n_rows, dtype=np.int64)
        for start, end, value in zip(starts, ends, values):
            if not math.isfinite(float(value)):
                continue
            local_start = max(int(start), row_start) - row_start
            local_end = min(int(end), row_end) - row_start
            if local_start >= local_end:
                continue
            total[local_start:local_end] += float(value)
            count[local_start:local_end] += 1
        out = np.full(n_rows, np.nan, dtype=np.float32)
        valid = count > 0
        out[valid] = (total[valid] / count[valid]).astype(np.float32)
        return out, count

    raise ValueError("mode must be one of: max, mean")


def sample_scores_to_rows(
    scores: Dict[str, np.ndarray],
    labels: Dict[str, np.ndarray],
    score_key: Optional[str],
    seq_len: int,
    row_start: int,
    row_end: int,
    aggregation: str,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if score_key is None or score_key not in scores:
        return None, None

    window_start = arr1d(labels, "window_start_index", np.int64)
    values = arr1d(scores, score_key, np.float32)
    if window_start is None or values is None:
        return None, None

    if values.shape[0] != window_start.shape[0]:
        return None, None

    starts = window_start
    ends = window_start + int(seq_len)
    return aggregate_intervals_to_rows(starts, ends, values, row_start, row_end, mode=aggregation)


def timestamp_scores_to_rows(
    scores: Dict[str, np.ndarray],
    labels: Dict[str, np.ndarray],
    score_key: Optional[str],
    row_start: int,
    row_end: int,
    aggregation: str,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if score_key is None or score_key not in scores:
        return None, None

    values = arr1d(scores, score_key, np.float32)
    starts = arr1d(labels, "timestamp_global_start_index", np.int64)
    ends = arr1d(labels, "timestamp_global_end_index", np.int64)

    if values is None or starts is None or ends is None:
        return None, None

    if not (values.shape[0] == starts.shape[0] == ends.shape[0]):
        return None, None

    valid = (starts >= 0) & (ends > starts) & np.isfinite(values)
    if not np.any(valid):
        return None, None

    return aggregate_intervals_to_rows(starts[valid], ends[valid], values[valid], row_start, row_end, mode=aggregation)


def point_mask_to_band(
    mask_data: Dict[str, np.ndarray],
    row_start: int,
    row_end: int,
) -> Optional[np.ndarray]:
    if "point_anomaly_mask" not in mask_data:
        return None
    point_mask = np.asarray(mask_data["point_anomaly_mask"]).astype(np.int64)
    if point_mask.ndim != 2:
        return None
    row_end = min(row_end, point_mask.shape[0])
    row_start = max(0, row_start)
    return point_mask[row_start:row_end, :].any(axis=1).astype(np.int64)


def point_ood_type_to_band(
    mask_data: Dict[str, np.ndarray],
    row_start: int,
    row_end: int,
) -> Optional[np.ndarray]:
    if "point_ood_type" not in mask_data:
        return None
    point_type = np.asarray(mask_data["point_ood_type"]).astype(np.int64)
    if point_type.ndim != 2:
        return None
    row_end = min(row_end, point_type.shape[0])
    row_start = max(0, row_start)
    sub = point_type[row_start:row_end, :]
    return np.max(sub, axis=1).astype(np.int64)


# -----------------------------------------------------------------------------
# Score key and threshold selection
# -----------------------------------------------------------------------------


def select_score_key(data: Dict[str, np.ndarray], preferred: Optional[str], candidates: Sequence[str]) -> Optional[str]:
    if preferred:
        if preferred not in data:
            raise KeyError(f"Score key not found in scores npz: {preferred}")
        return preferred
    for key in candidates:
        if key in data:
            return key
    return None


def threshold_for_score_key(scores: Dict[str, np.ndarray], score_key: Optional[str]) -> Optional[float]:
    if score_key is None:
        return None
    if "instance" in score_key:
        return scalar_from_npz(scores, "instance_threshold")
    if "timestamp" in score_key:
        return scalar_from_npz(scores, "timestamp_threshold")
    return None


# -----------------------------------------------------------------------------
# Top-K / top100 band
# -----------------------------------------------------------------------------


def read_top_csv_rows(path: str | Path) -> List[Dict[str, str]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"top_csv_path not found: {path}")
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def top_ranges_from_csv(
    top_csv_path: str | Path,
    top_csv_level: str,
    labels: Dict[str, np.ndarray],
    seq_len: int,
    max_rows: int,
) -> List[Tuple[int, int, int]]:
    rows = read_top_csv_rows(top_csv_path)
    rows = rows[:max_rows]

    if top_csv_level == "auto":
        lower_name = Path(top_csv_path).name.lower()
        top_csv_level = "timestamp" if "timestamp" in lower_name else "instance"

    ranges: List[Tuple[int, int, int]] = []

    if top_csv_level == "instance":
        window_start = arr1d(labels, "window_start_index", np.int64)
        if window_start is None:
            return []
        for row in rows:
            rank = int(row.get("rank", len(ranges) + 1))
            raw_sample = row.get("sample_index") or row.get("row_index")
            if raw_sample in {None, ""}:
                continue
            sample_idx = int(float(raw_sample))
            if 0 <= sample_idx < window_start.shape[0]:
                start = int(window_start[sample_idx])
                ranges.append((start, start + int(seq_len), rank))
        return ranges

    if top_csv_level == "timestamp":
        starts = arr1d(labels, "timestamp_global_start_index", np.int64)
        ends = arr1d(labels, "timestamp_global_end_index", np.int64)
        if starts is None or ends is None:
            return []
        for row in rows:
            rank = int(row.get("rank", len(ranges) + 1))
            raw_idx = row.get("row_index")
            if raw_idx in {None, ""}:
                continue
            idx = int(float(raw_idx))
            if 0 <= idx < starts.shape[0]:
                ranges.append((int(starts[idx]), int(ends[idx]), rank))
        return ranges

    raise ValueError("top_csv_level must be one of: auto, instance, timestamp")


def top_ranges_from_scores(
    scores: Dict[str, np.ndarray],
    labels: Dict[str, np.ndarray],
    score_key: Optional[str],
    seq_len: int,
    top_k: int,
) -> List[Tuple[int, int, int]]:
    if score_key is None or score_key not in scores:
        return []

    score_values = arr1d(scores, score_key, np.float32)
    if score_values is None or score_values.size == 0:
        return []

    if score_key.startswith("sample_"):
        window_start = arr1d(labels, "window_start_index", np.int64)
        if window_start is None or window_start.shape[0] != score_values.shape[0]:
            return []
        valid_score = np.where(np.isfinite(score_values), score_values, -np.inf)
        order = np.argsort(-valid_score)[:top_k]
        return [(int(window_start[i]), int(window_start[i]) + int(seq_len), rank) for rank, i in enumerate(order, start=1)]

    if score_key == "timestamp_scores":
        starts = arr1d(labels, "timestamp_global_start_index", np.int64)
        ends = arr1d(labels, "timestamp_global_end_index", np.int64)
        if starts is None or ends is None or starts.shape[0] != score_values.shape[0]:
            return []
        valid_score = np.where(np.isfinite(score_values), score_values, -np.inf)
        order = np.argsort(-valid_score)[:top_k]
        return [(int(starts[i]), int(ends[i]), rank) for rank, i in enumerate(order, start=1)]

    return []


def top_band_from_ranges(
    ranges: List[Tuple[int, int, int]],
    row_start: int,
    row_end: int,
) -> Tuple[np.ndarray, np.ndarray]:
    n = row_end - row_start
    band = np.zeros(n, dtype=np.int64)
    best_rank = np.zeros(n, dtype=np.int64)
    for start, end, rank in ranges:
        local_start = max(int(start), row_start) - row_start
        local_end = min(int(end), row_end) - row_start
        if local_start >= local_end:
            continue
        band[local_start:local_end] = 1
        current = best_rank[local_start:local_end]
        update = (current == 0) | (rank < current)
        current[update] = rank
    return band, best_rank


# -----------------------------------------------------------------------------
# Metrics HTML
# -----------------------------------------------------------------------------


def nested_get(obj: Dict[str, Any], keys: Sequence[str]) -> Any:
    cur: Any = obj
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def metric_card_html(metrics: Dict[str, Any], section: str, title: str) -> str:
    data = metrics.get(section, {})
    if not isinstance(data, dict) or not data:
        return f"<div class='card muted'><h3>{html_escape(title)}</h3><p>Nincs metrika ehhez a szinthez.</p></div>"

    metric_specs = [
        ("AUROC", ["auroc"]),
        ("AUPRC", ["auprc"]),
        ("FPR@95TPR", ["fpr_at_95_tpr"]),
        ("Accuracy", ["threshold_metrics", "accuracy"]),
        ("Precision", ["threshold_metrics", "precision"]),
        ("Recall", ["threshold_metrics", "recall"]),
        ("F1", ["threshold_metrics", "f1"]),
        ("N normal", ["n_normal"]),
        ("N anomaly", ["n_anomaly"]),
    ]

    cards = []
    for label, keys in metric_specs:
        value = nested_get(data, keys)
        value_s = format_int(value) if label.startswith("N ") else format_float(value)
        cards.append(
            "<div class='metric-box'>"
            f"<div class='metric-label'>{html_escape(label)}</div>"
            f"<div class='metric-value'>{value_s}</div>"
            "</div>"
        )

    cm = nested_get(data, ["threshold_metrics", "confusion_matrix"])
    cm_html = ""
    if isinstance(cm, dict):
        cm_html = (
            "<table class='cm-table'>"
            "<tr><th></th><th>pred normal</th><th>pred anomaly</th></tr>"
            f"<tr><th>true normal</th><td>TN={format_int(cm.get('tn'))}</td><td>FP={format_int(cm.get('fp'))}</td></tr>"
            f"<tr><th>true anomaly</th><td>FN={format_int(cm.get('fn'))}</td><td>TP={format_int(cm.get('tp'))}</td></tr>"
            "</table>"
        )

    return (
        "<div class='card'>"
        f"<h3>{html_escape(title)}</h3>"
        "<div class='metric-grid'>"
        + "".join(cards)
        + "</div>"
        + cm_html
        + "</div>"
    )


# -----------------------------------------------------------------------------
# Plot building
# -----------------------------------------------------------------------------


def add_binary_band(fig, row: int, x: np.ndarray, y: Optional[np.ndarray], name: str, hover_extra: Optional[np.ndarray] = None):
    import plotly.graph_objects as go

    if y is None:
        return
    y = np.asarray(y).reshape(-1)
    customdata = hover_extra if hover_extra is not None else y
    fig.add_trace(
        go.Bar(
            x=x,
            y=y,
            name=name,
            customdata=customdata,
            hovertemplate=f"{html.escape(name)}=%{{y}}<br>extra=%{{customdata}}<br>x=%{{x}}<extra></extra>",
        ),
        row=row,
        col=1,
    )


def add_score_trace(fig, row: int, x: np.ndarray, y: Optional[np.ndarray], name: str, threshold: Optional[float]):
    import plotly.graph_objects as go

    if y is None:
        return
    fig.add_trace(
        go.Scattergl(
            x=x,
            y=y,
            mode="lines",
            name=name,
            hovertemplate=f"{html.escape(name)}=%{{y:.6f}}<br>x=%{{x}}<extra></extra>",
        ),
        row=row,
        col=1,
    )
    if threshold is not None:
        fig.add_hline(
            y=float(threshold),
            line_dash="dash",
            annotation_text=f"threshold={threshold:.4f}",
            row=row,
            col=1,
        )


def build_figure(
    original_df: pd.DataFrame,
    injected_df: pd.DataFrame,
    value_columns: Sequence[str],
    x: np.ndarray,
    row_start: int,
    row_end: int,
    gt_band: Optional[np.ndarray],
    ood_type_band: Optional[np.ndarray],
    instance_row_score: Optional[np.ndarray],
    timestamp_row_score: Optional[np.ndarray],
    instance_threshold: Optional[float],
    timestamp_threshold: Optional[float],
    detected_band: Optional[np.ndarray],
    top_band: Optional[np.ndarray],
    top_rank_band: Optional[np.ndarray],
    title: str,
):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    rows = 7
    subplot_titles = [
        "Original dataset — test region",
        "Injected dataset — test region",
        "Ground truth anomaly band",
        "Instance/sample score projected to rows",
        "Timestamp/patch score projected to rows",
        "Model detected anomaly band",
        "Top-K / Top100 coverage band",
    ]

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.018,
        row_heights=[0.25, 0.25, 0.08, 0.12, 0.12, 0.08, 0.08],
        subplot_titles=subplot_titles,
    )

    for col in value_columns:
        fig.add_trace(
            go.Scattergl(
                x=x,
                y=original_df[col].iloc[row_start:row_end].to_numpy(),
                mode="lines",
                name=f"original:{col}",
                legendgroup=f"col:{col}",
                hovertemplate=f"original {html.escape(col)}=%{{y}}<br>x=%{{x}}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    for col in value_columns:
        fig.add_trace(
            go.Scattergl(
                x=x,
                y=injected_df[col].iloc[row_start:row_end].to_numpy(),
                mode="lines",
                name=f"injected:{col}",
                legendgroup=f"col:{col}",
                hovertemplate=f"injected {html.escape(col)}=%{{y}}<br>x=%{{x}}<extra></extra>",
            ),
            row=2,
            col=1,
        )

    add_binary_band(fig, 3, x, gt_band, "GT anomaly", hover_extra=ood_type_band)
    add_score_trace(fig, 4, x, instance_row_score, "instance_row_score", instance_threshold)
    add_score_trace(fig, 5, x, timestamp_row_score, "timestamp_row_score", timestamp_threshold)
    add_binary_band(fig, 6, x, detected_band, "model_detected")
    add_binary_band(fig, 7, x, top_band, "topK_coverage", hover_extra=top_rank_band)

    # Highlight real anomaly regions as background rectangles across all rows.
    if gt_band is not None:
        gt = np.asarray(gt_band).reshape(-1).astype(np.int64)
        if gt.size == x.shape[0]:
            starts = np.where(np.diff(np.concatenate([[0], gt, [0]])) == 1)[0]
            ends = np.where(np.diff(np.concatenate([[0], gt, [0]])) == -1)[0]
            for s, e in zip(starts, ends):
                x0 = x[s]
                x1 = x[min(e - 1, len(x) - 1)]
                fig.add_vrect(x0=x0, x1=x1, opacity=0.08, line_width=0, row="all", col=1)

    fig.update_layout(
        title=title,
        height=1180,
        hovermode="x unified",
        legend_title="Traces",
        margin={"l": 72, "r": 32, "t": 90, "b": 70},
    )
    fig.update_yaxes(title_text="original", row=1, col=1)
    fig.update_yaxes(title_text="injected", row=2, col=1)
    fig.update_yaxes(title_text="GT", row=3, col=1, range=[0, 1.15])
    fig.update_yaxes(title_text="score", row=4, col=1)
    fig.update_yaxes(title_text="score", row=5, col=1)
    fig.update_yaxes(title_text="pred", row=6, col=1, range=[0, 1.15])
    fig.update_yaxes(title_text="topK", row=7, col=1, range=[0, 1.15])
    fig.update_xaxes(title_text="CSV row / time", row=7, col=1)

    return fig


# -----------------------------------------------------------------------------
# HTML
# -----------------------------------------------------------------------------


def build_html(
    fig_html: str,
    metrics_html: str,
    info_html: str,
    title: str,
) -> str:
    css = """
    :root {
        --bg: #f8fafc;
        --card: #ffffff;
        --text: #111827;
        --muted: #6b7280;
        --border: #e5e7eb;
    }
    body {
        margin: 0;
        background: var(--bg);
        color: var(--text);
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
        background: #111827;
        color: white;
        padding: 26px 38px;
    }
    header h1 { margin: 0 0 8px 0; font-size: 26px; }
    header p { margin: 0; color: #d1d5db; }
    main { max-width: 1600px; margin: 0 auto; padding: 24px; }
    .card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 16px;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
        padding: 18px;
        margin: 16px 0;
        overflow-x: auto;
    }
    .plot-card { padding: 8px 12px 4px 12px; }
    .muted { color: var(--muted); }
    h2 { margin: 26px 0 12px 0; }
    h3 { margin: 0 0 12px 0; }
    table { width: 100%; border-collapse: collapse; font-size: 14px; }
    th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid var(--border); vertical-align: top; }
    th { width: 260px; background: #f9fafb; }
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(145px, 1fr));
        gap: 12px;
    }
    .metric-box {
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 13px;
        background: #fafafa;
    }
    .metric-label { font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: .04em; }
    .metric-value { font-size: 23px; font-weight: 700; margin-top: 3px; }
    .cm-table { margin-top: 14px; }
    """

    return (
        "<!doctype html><html lang='hu'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>{html_escape(title)}</title>"
        f"<style>{css}</style></head><body>"
        "<header>"
        f"<h1>{html_escape(title)}</h1>"
        "<p>Full forecasting test-region browser: original vs injected data, ground truth, scores, detections and top-K coverage.</p>"
        "</header><main>"
        f"<section><h2>Idősoros böngésző</h2><div class='card plot-card'>{fig_html}</div></section>"
        f"<section><h2>Beállítások és illesztés</h2>{info_html}</section>"
        f"<section><h2>Detektor metrikák</h2>{metrics_html}</section>"
        "</main></body></html>"
    )


# -----------------------------------------------------------------------------
# Main dashboard generation
# -----------------------------------------------------------------------------


def create_browser(args: argparse.Namespace) -> Path:
    try:
        import plotly.io as pio
    except ImportError as exc:
        raise ImportError("plotly is required. Install with: pip install plotly") from exc

    original_csv_path = Path(args.original_csv_path)
    injected_csv_path = Path(args.injected_csv_path)

    original_df = pd.read_csv(original_csv_path)
    injected_df = pd.read_csv(injected_csv_path)

    if len(original_df) != len(injected_df):
        raise ValueError(
            f"original and injected CSV length mismatch: {len(original_df)} != {len(injected_df)}"
        )

    scores = load_npz(args.scores_path)
    labels = load_npz(args.labels_path)
    mask_data = load_npz(args.mask_path) if args.mask_path else {}
    metrics = load_json(args.metrics_path)

    seq_len = infer_seq_len(args, labels, metrics)
    row_start, row_end = infer_test_range(
        labels=labels,
        mask_data=mask_data,
        seq_len=seq_len,
        test_start_index=args.test_start_index,
        test_end_index=args.test_end_index,
        csv_len=len(original_df),
    )

    date_column = infer_date_column(original_df, args.date_column)
    value_columns = select_value_columns(
        original_df=original_df,
        injected_df=injected_df,
        mask_data=mask_data,
        explicit=args.value_columns,
        max_channels=args.max_channels,
    )
    x = make_x_axis(original_df, date_column, row_start, row_end)

    instance_score_key = select_score_key(
        scores,
        args.instance_score_key,
        ["sample_instance_score_max", "sample_instance_score_mean", "instance_scores"],
    )
    timestamp_score_key = select_score_key(
        scores,
        args.timestamp_score_key,
        ["timestamp_scores", "sample_timestamp_score_max", "sample_timestamp_score_mean"],
    )

    # For row-level visualization, raw instance_scores cannot be projected unless they
    # are already aggregated sample scores. Prefer sample_* keys.
    if instance_score_key == "instance_scores":
        instance_row_score = None
        instance_coverage = None
    else:
        instance_row_score, instance_coverage = sample_scores_to_rows(
            scores=scores,
            labels=labels,
            score_key=instance_score_key,
            seq_len=seq_len,
            row_start=row_start,
            row_end=row_end,
            aggregation=args.row_score_aggregation,
        )

    if timestamp_score_key == "timestamp_scores":
        timestamp_row_score, timestamp_coverage = timestamp_scores_to_rows(
            scores=scores,
            labels=labels,
            score_key=timestamp_score_key,
            row_start=row_start,
            row_end=row_end,
            aggregation=args.row_score_aggregation,
        )
    else:
        timestamp_row_score, timestamp_coverage = sample_scores_to_rows(
            scores=scores,
            labels=labels,
            score_key=timestamp_score_key,
            seq_len=seq_len,
            row_start=row_start,
            row_end=row_end,
            aggregation=args.row_score_aggregation,
        )

    gt_band = point_mask_to_band(mask_data, row_start, row_end)
    ood_type_band = point_ood_type_to_band(mask_data, row_start, row_end)

    instance_threshold = threshold_for_score_key(scores, instance_score_key)
    timestamp_threshold = threshold_for_score_key(scores, timestamp_score_key)

    detected_parts = []
    if instance_row_score is not None and instance_threshold is not None:
        detected_parts.append(np.isfinite(instance_row_score) & (instance_row_score > float(instance_threshold)))
    if timestamp_row_score is not None and timestamp_threshold is not None:
        detected_parts.append(np.isfinite(timestamp_row_score) & (timestamp_row_score > float(timestamp_threshold)))

    if detected_parts:
        detected_band = np.logical_or.reduce(detected_parts).astype(np.int64)
    else:
        detected_band = None

    if args.top_csv_path:
        ranges = top_ranges_from_csv(
            top_csv_path=args.top_csv_path,
            top_csv_level=args.top_csv_level,
            labels=labels,
            seq_len=seq_len,
            max_rows=args.top_k,
        )
    else:
        top_score_key = args.top_score_key or instance_score_key or timestamp_score_key
        ranges = top_ranges_from_scores(
            scores=scores,
            labels=labels,
            score_key=top_score_key,
            seq_len=seq_len,
            top_k=args.top_k,
        )

    top_band, top_rank_band = top_band_from_ranges(ranges, row_start, row_end)

    output_name = args.output_name or Path(args.scores_path).name.replace(".scores.npz", "")
    title = args.title or f"TimeDRL forecasting test browser — {output_name}"

    fig = build_figure(
        original_df=original_df,
        injected_df=injected_df,
        value_columns=value_columns,
        x=x,
        row_start=row_start,
        row_end=row_end,
        gt_band=gt_band,
        ood_type_band=ood_type_band,
        instance_row_score=instance_row_score,
        timestamp_row_score=timestamp_row_score,
        instance_threshold=instance_threshold,
        timestamp_threshold=timestamp_threshold,
        detected_band=detected_band,
        top_band=top_band,
        top_rank_band=top_rank_band,
        title=title,
    )

    fig_html = pio.to_html(
        fig,
        include_plotlyjs="cdn",
        full_html=False,
        config={"responsive": True, "displaylogo": False, "scrollZoom": True},
    )

    metrics_html = (
        "<div class='card-grid'>"
        + metric_card_html(metrics, "instance_level", "Instance/sample-level értékelés")
        + metric_card_html(metrics, "timestamp_level", "Timestamp/patch-level értékelés")
        + "</div>"
    )

    info_rows = [
        ("original_csv_path", original_csv_path),
        ("injected_csv_path", injected_csv_path),
        ("scores_path", args.scores_path),
        ("labels_path", args.labels_path),
        ("mask_path", args.mask_path),
        ("metrics_path", args.metrics_path),
        ("test row range", f"{row_start}:{row_end}"),
        ("rows shown", row_end - row_start),
        ("date_column", date_column),
        ("value_columns", ", ".join(value_columns)),
        ("seq_len", seq_len),
        ("instance_score_key", instance_score_key),
        ("timestamp_score_key", timestamp_score_key),
        ("instance_threshold", instance_threshold),
        ("timestamp_threshold", timestamp_threshold),
        ("row_score_aggregation", args.row_score_aggregation),
        ("top source", args.top_csv_path or (args.top_score_key or instance_score_key or timestamp_score_key)),
        ("top_k", args.top_k),
    ]
    info_html = "<div class='card'><table>" + "".join(
        f"<tr><th>{html_escape(k)}</th><td>{html_escape(v)}</td></tr>" for k, v in info_rows
    ) + "</table></div>"

    html_page = build_html(
        fig_html=fig_html,
        metrics_html=metrics_html,
        info_html=info_html,
        title=title,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{output_name}.forecasting_timeseries_browser.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_page)

    return output_path


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create an interactive Plotly browser for a TimeDRL forecasting test set."
    )

    parser.add_argument("--original_csv_path", type=str, required=True)
    parser.add_argument("--injected_csv_path", type=str, required=True)
    parser.add_argument("--scores_path", type=str, required=True)
    parser.add_argument("--labels_path", type=str, required=True)
    parser.add_argument("--mask_path", type=str, default=None)
    parser.add_argument("--metrics_path", type=str, default=None)

    parser.add_argument("--output_dir", type=str, default="./ood/score_visualizations")
    parser.add_argument("--output_name", type=str, default=None)
    parser.add_argument("--title", type=str, default=None)

    parser.add_argument("--seq_len", type=int, default=None)
    parser.add_argument("--test_start_index", type=int, default=None)
    parser.add_argument("--test_end_index", type=int, default=None)

    parser.add_argument("--date_column", type=str, default=None)
    parser.add_argument("--value_columns", nargs="*", default=None)
    parser.add_argument("--max_channels", type=int, default=None)

    parser.add_argument("--instance_score_key", type=str, default=None)
    parser.add_argument("--timestamp_score_key", type=str, default=None)
    parser.add_argument("--row_score_aggregation", type=str, default="max", choices=["max", "mean"])

    parser.add_argument("--top_k", type=int, default=100)
    parser.add_argument("--top_csv_path", type=str, default=None)
    parser.add_argument("--top_csv_level", type=str, default="auto", choices=["auto", "instance", "timestamp"])
    parser.add_argument("--top_score_key", type=str, default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = create_browser(args)
    print(f"Saved forecasting time-series browser: {output_path}")


if __name__ == "__main__":
    main()
