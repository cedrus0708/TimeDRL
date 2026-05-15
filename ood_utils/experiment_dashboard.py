"""Create a single HTML dashboard for TimeDRL OOD/anomaly experiments.

This script is intentionally standalone: it reads the artifacts already produced by
embedding_bank.py, embedding_detector.py, ood_eval_sets.py and evaluate_detector.py,
then writes one experiment_dashboard.html file.

Typical usage:
    python ood_utils/experiment_dashboard.py \
        --scores_path ./ood/embedding_detectors/EXPERIMENT.scores.npz \
        --labels_path ./ood/eval_sets/EXPERIMENT_labels.npz \
        --metrics_path ./ood/evaluation_reports/EXPERIMENT.metrics_summary.json \
        --train_bank_path ./ood/embedding_banks/TRAIN_BANK.npz \
        --test_bank_path ./ood/embedding_banks/TEST_BANK.npz \
        --detector_meta_path ./ood/embedding_detectors/EXPERIMENT.detector_meta.json \
        --output_dir ./ood/score_visualizations \
        --output_name EXPERIMENT

Optional forecasting extras:
    --mask_path ./ood/eval_sets/Exchange_injected_mask.npz
    --original_csv_path ./dataset/forecasting/exchange_rate/exchange_rate.csv
    --injected_csv_path ./dataset/forecasting/exchange_rate/exchange_rate_injected.csv
    --seq_len 168
"""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


GROUP_NAMES = {
    -1: "ignored",
    0: "ID / normal",
    1: "near-OOD / anomaly",
    2: "far-OOD",
}


def load_npz(path: Optional[str | Path]) -> Dict[str, np.ndarray]:
    if path is None:
        return {}

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
        raise FileNotFoundError(f"File not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def maybe_path(path: Optional[str | Path]) -> Optional[Path]:
    if path is None:
        return None
    return Path(path)


def first_existing_path(paths: Sequence[Optional[str | Path]]) -> Optional[Path]:
    for path in paths:
        if path is None:
            continue
        p = Path(path)
        if p.exists():
            return p
    return None


def infer_sidecar_meta(npz_path: Optional[str | Path]) -> Optional[Path]:
    if npz_path is None:
        return None
    p = Path(npz_path)
    candidate = p.with_suffix(".meta.json")
    return candidate if candidate.exists() else None


def infer_detector_meta(scores_path: Optional[str | Path]) -> Optional[Path]:
    if scores_path is None:
        return None
    p = Path(scores_path)
    stem = p.name.replace(".scores.npz", "")
    candidate = p.with_name(f"{stem}.detector_meta.json")
    return candidate if candidate.exists() else None


def infer_metrics_path(scores_path: Optional[str | Path], output_dir: Optional[str | Path]) -> Optional[Path]:
    if scores_path is None or output_dir is None:
        return None
    p = Path(scores_path)
    stem = p.name.replace(".scores.npz", "")
    candidate = Path(output_dir) / f"{stem}.metrics_summary.json"
    return candidate if candidate.exists() else None


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


def safe_scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.shape == ():
            return value.item()
        if value.size == 1:
            return value.reshape(-1)[0].item()
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        x = float(value)
        return None if not math.isfinite(x) else x
    return value


def arr1d(data: Dict[str, np.ndarray], key: str, dtype: Optional[Any] = None) -> Optional[np.ndarray]:
    if key not in data:
        return None
    arr = np.asarray(data[key]).reshape(-1)
    if dtype is not None:
        arr = arr.astype(dtype)
    return arr


def auto_score_key(data: Dict[str, np.ndarray], preferred: Optional[str], candidates: Sequence[str]) -> Optional[str]:
    if preferred:
        if preferred not in data:
            raise KeyError(f"Score key not found: {preferred}")
        return preferred

    for key in candidates:
        if key in data:
            return key
    return None


def derive_binary_from_group(group: np.ndarray) -> np.ndarray:
    group = np.asarray(group).reshape(-1).astype(np.int64)
    y = np.full_like(group, fill_value=-1, dtype=np.int64)
    y[group == 0] = 0
    y[group > 0] = 1
    return y


def label_for_level(labels: Dict[str, np.ndarray], level: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    if level == "instance":
        y = arr1d(labels, "sample_label", np.int64)
        group = arr1d(labels, "sample_ood_type", np.int64)
    elif level == "timestamp":
        y = arr1d(labels, "timestamp_label", np.int64)
        group = arr1d(labels, "timestamp_ood_type", np.int64)
    else:
        raise ValueError(f"Unknown level: {level}")

    if y is None and group is not None:
        y = derive_binary_from_group(group)

    return y, group


def count_groups(group: Optional[np.ndarray]) -> Dict[str, int]:
    if group is None:
        return {}
    group = np.asarray(group).reshape(-1).astype(np.int64)
    return {GROUP_NAMES.get(int(g), f"group_{int(g)}"): int(np.sum(group == g)) for g in np.unique(group)}


def shape_or_dash(data: Dict[str, np.ndarray], key: str) -> str:
    if key not in data:
        return "—"
    return "×".join(str(dim) for dim in data[key].shape)


def read_meta(meta_path: Optional[str | Path]) -> Dict[str, Any]:
    try:
        return load_json(meta_path)
    except FileNotFoundError:
        return {}


def metadata_table_rows(obj: Dict[str, Any], keys: Sequence[str]) -> str:
    rows = []
    for key in keys:
        value = obj.get(key, None)
        if isinstance(value, (list, tuple)):
            value = ", ".join(str(v) for v in value)
        elif isinstance(value, dict):
            value = json.dumps(value, ensure_ascii=False)
        value = "—" if value is None else str(value)
        rows.append(f"<tr><th>{html.escape(key)}</th><td>{html.escape(value)}</td></tr>")
    return "\n".join(rows)


def extract_detector_config(detector_meta: Dict[str, Any], scores: Dict[str, np.ndarray]) -> Dict[str, Any]:
    instance_meta = detector_meta.get("instance_detector", {}) if detector_meta else {}
    timestamp_meta = detector_meta.get("timestamp_detector", {}) if detector_meta else {}
    base = instance_meta or timestamp_meta or {}

    out = {
        "detector_type": "kNN embedding distance",
        "k": base.get("k"),
        "metric": base.get("metric"),
        "score_mode": base.get("score_mode"),
        "normalization": base.get("normalization"),
        "threshold_quantile": base.get("threshold_quantile"),
        "instance_threshold": base.get("threshold"),
        "timestamp_threshold": timestamp_meta.get("threshold") if timestamp_meta else None,
    }

    if out["instance_threshold"] is None and "instance_threshold" in scores:
        out["instance_threshold"] = safe_scalar(scores["instance_threshold"])
    if out["timestamp_threshold"] is None and "timestamp_threshold" in scores:
        out["timestamp_threshold"] = safe_scalar(scores["timestamp_threshold"])

    return out


def extract_metric(metrics: Dict[str, Any], level: str, path: Sequence[str]) -> Any:
    obj: Any = metrics.get(level, {})
    for key in path:
        if not isinstance(obj, dict) or key not in obj:
            return None
        obj = obj[key]
    return obj


def metric_cards(metrics: Dict[str, Any], level: str, title: str) -> str:
    if level not in metrics:
        return f"<div class='card muted'><h3>{html.escape(title)}</h3><p>Nincs ehhez a szinthez metrika.</p></div>"

    items = [
        ("AUROC", extract_metric(metrics, level, ["auroc"]), 4),
        ("AUPRC", extract_metric(metrics, level, ["auprc"]), 4),
        ("F1", extract_metric(metrics, level, ["threshold_metrics", "f1"]), 4),
        ("Precision", extract_metric(metrics, level, ["threshold_metrics", "precision"]), 4),
        ("Recall", extract_metric(metrics, level, ["threshold_metrics", "recall"]), 4),
        ("Accuracy", extract_metric(metrics, level, ["threshold_metrics", "accuracy"]), 4),
    ]

    html_items = []
    for label, value, digits in items:
        html_items.append(
            "<div class='metric-card'>"
            f"<div class='metric-label'>{html.escape(label)}</div>"
            f"<div class='metric-value'>{format_float(value, digits)}</div>"
            "</div>"
        )

    n_total = extract_metric(metrics, level, ["n_total"])
    n_normal = extract_metric(metrics, level, ["n_normal"])
    n_anomaly = extract_metric(metrics, level, ["n_anomaly"])
    subtitle = f"N={format_int(n_total)} | normal={format_int(n_normal)} | anomaly/OOD={format_int(n_anomaly)}"

    return (
        "<div class='card'>"
        f"<h3>{html.escape(title)}</h3>"
        f"<p class='muted'>{subtitle}</p>"
        "<div class='metric-grid'>"
        + "".join(html_items)
        + "</div></div>"
    )


def make_confusion_figure(metrics: Dict[str, Any], level: str, title: str):
    import plotly.graph_objects as go

    cm = extract_metric(metrics, level, ["threshold_metrics", "confusion_matrix"])
    if not cm:
        return None

    z = np.asarray(
        [
            [cm.get("tn", 0), cm.get("fp", 0)],
            [cm.get("fn", 0), cm.get("tp", 0)],
        ],
        dtype=np.int64,
    )

    text = np.asarray(
        [
            [f"TN<br>{format_int(z[0, 0])}", f"FP<br>{format_int(z[0, 1])}"],
            [f"FN<br>{format_int(z[1, 0])}", f"TP<br>{format_int(z[1, 1])}"],
        ]
    )

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=["pred normal / ID", "pred anomaly / OOD"],
            y=["true normal / ID", "true anomaly / OOD"],
            text=text,
            texttemplate="%{text}",
            hovertemplate="%{y}<br>%{x}<br>count=%{z}<extra></extra>",
            colorbar={"title": "count"},
        )
    )
    fig.update_layout(title=title, height=380, margin={"l": 70, "r": 40, "t": 70, "b": 70})
    return fig


def make_score_distribution_figure(
    scores: Dict[str, np.ndarray],
    labels: Dict[str, np.ndarray],
    score_key: str,
    level: str,
    title: str,
):
    import plotly.graph_objects as go

    score_values = arr1d(scores, score_key, np.float32)
    if score_values is None:
        return None

    y, group = label_for_level(labels, level)
    if y is None or y.shape[0] != score_values.shape[0]:
        return None

    valid = np.isin(y, [0, 1]) & np.isfinite(score_values)
    if not np.any(valid):
        return None

    score_values = score_values[valid]
    y = y[valid]
    if group is not None and group.shape[0] == valid.shape[0]:
        group = group[valid]
    else:
        group = y

    fig = go.Figure()
    for g in sorted(int(v) for v in np.unique(group)):
        mask = group == g
        name = GROUP_NAMES.get(g, f"group_{g}")
        fig.add_trace(
            go.Histogram(
                x=score_values[mask],
                nbinsx=80,
                name=name,
                opacity=0.65,
                histnorm="probability density",
            )
        )

    threshold_key = "instance_threshold" if "instance" in score_key else "timestamp_threshold"
    if threshold_key in scores:
        threshold = safe_scalar(scores[threshold_key])
        if threshold is not None:
            fig.add_vline(
                x=float(threshold),
                line_dash="dash",
                annotation_text=f"threshold={format_float(threshold)}",
            )

    fig.update_layout(
        title=title,
        barmode="overlay",
        xaxis_title=score_key,
        yaxis_title="density",
        height=430,
        legend_title="label/group",
        margin={"l": 60, "r": 30, "t": 70, "b": 60},
    )
    return fig


def choose_numeric_columns(df, value_columns: Optional[Sequence[str]], max_columns: int) -> List[str]:
    import pandas as pd

    if value_columns:
        return [c for c in value_columns if c in df.columns][:max_columns]

    excluded = {"date", "datetime", "time", "timestamp"}
    cols = []
    for col in df.columns:
        if col.lower() in excluded:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols[:max_columns]


def decode_string_array(value: Any) -> List[str]:
    if value is None:
        return []
    arr = np.asarray(value).reshape(-1)
    out = []
    for item in arr:
        if isinstance(item, bytes):
            out.append(item.decode("utf-8"))
        else:
            out.append(str(item))
    return out


def make_forecasting_timeline_figure(
    csv_path: Optional[Path],
    mask_data: Dict[str, np.ndarray],
    date_column: Optional[str],
    value_columns: Optional[Sequence[str]],
    max_points: int,
    max_channels: int,
):
    if csv_path is None or not csv_path.exists() or "point_anomaly_mask" not in mask_data:
        return None

    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    df = pd.read_csv(csv_path)
    point_mask = np.asarray(mask_data["point_anomaly_mask"]).astype(np.int64)

    if point_mask.shape[0] != len(df):
        return None

    if not value_columns and "value_columns" in mask_data:
        value_columns = decode_string_array(mask_data["value_columns"])

    cols = choose_numeric_columns(df, value_columns, max_columns=max_channels)
    if not cols:
        return None

    if date_column and date_column in df.columns:
        x_full = df[date_column]
    else:
        # Try common date columns first.
        common_date = next((c for c in ["date", "datetime", "time", "timestamp"] if c in df.columns), None)
        x_full = df[common_date] if common_date else np.arange(len(df))

    step = max(1, int(math.ceil(len(df) / max_points)))
    idx = np.arange(0, len(df), step)
    x = np.asarray(x_full)[idx]
    anomaly_any = point_mask.any(axis=1).astype(np.int64)[idx]

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.72, 0.28],
        subplot_titles=["Forecasting time series", "Ground truth anomaly mask"],
    )

    for col in cols:
        fig.add_trace(
            go.Scattergl(
                x=x,
                y=df[col].to_numpy()[idx],
                mode="lines",
                name=col,
                hovertemplate=f"{html.escape(col)}=%{{y}}<br>x=%{{x}}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Bar(
            x=x,
            y=anomaly_any,
            name="any channel anomaly",
            hovertemplate="mask=%{y}<br>x=%{x}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    if "inject_start_idx" in mask_data and "inject_end_idx" in mask_data:
        start = int(safe_scalar(mask_data["inject_start_idx"]))
        end = int(safe_scalar(mask_data["inject_end_idx"]))
        if start < len(df) and end > 0:
            x0 = np.asarray(x_full)[max(0, start)]
            x1 = np.asarray(x_full)[min(len(df) - 1, end - 1)]
            fig.add_vrect(x0=x0, x1=x1, opacity=0.12, line_width=0, annotation_text="injection range")

    fig.update_layout(
        title=f"Full forecasting overview: {csv_path.name}",
        height=720,
        barmode="overlay",
        legend_title="series",
        margin={"l": 60, "r": 30, "t": 80, "b": 60},
    )
    fig.update_yaxes(title_text="value", row=1, col=1)
    fig.update_yaxes(title_text="mask", row=2, col=1)
    return fig


def make_sample_heatmap_with_overlay(
    scores: Dict[str, np.ndarray],
    labels: Dict[str, np.ndarray],
    sample_id: Optional[int],
):
    required = [
        "timestamp_scores",
        "timestamp_sample_index",
        "timestamp_channel_index",
        "timestamp_patch_index",
    ]
    if any(key not in scores for key in required):
        return None, None

    import plotly.graph_objects as go

    score_values = arr1d(scores, "timestamp_scores", np.float32)
    sample_index = arr1d(scores, "timestamp_sample_index", np.int64)
    channel_index = arr1d(scores, "timestamp_channel_index", np.int64)
    patch_index = arr1d(scores, "timestamp_patch_index", np.int64)
    timestamp_label = arr1d(labels, "timestamp_label", np.int64)
    global_start = arr1d(labels, "timestamp_global_start_index", np.int64)
    global_end = arr1d(labels, "timestamp_global_end_index", np.int64)

    if score_values is None or sample_index is None or channel_index is None or patch_index is None:
        return None, None

    if sample_id is None:
        valid = np.isfinite(score_values)
        if not np.any(valid):
            return None, None
        sample_id = int(sample_index[np.argmax(np.where(valid, score_values, -np.inf))])

    mask = sample_index == int(sample_id)
    if not np.any(mask):
        return None, None

    s_scores = score_values[mask]
    s_channels = channel_index[mask]
    s_patches = patch_index[mask]
    s_labels = timestamp_label[mask] if timestamp_label is not None and timestamp_label.shape[0] == score_values.shape[0] else None
    s_gstart = global_start[mask] if global_start is not None and global_start.shape[0] == score_values.shape[0] else None
    s_gend = global_end[mask] if global_end is not None and global_end.shape[0] == score_values.shape[0] else None

    if np.all(s_channels == -1):
        unique_channels = np.array([0], dtype=np.int64)
        display_channels = np.zeros_like(s_channels)
    else:
        unique_channels = np.unique(s_channels[s_channels >= 0])
        display_channels = s_channels

    unique_patches = np.unique(s_patches)
    heat = np.full((unique_channels.size, unique_patches.size), np.nan, dtype=np.float32)
    lab = np.zeros((unique_channels.size, unique_patches.size), dtype=np.int64)
    hover = np.full((unique_channels.size, unique_patches.size), "", dtype=object)

    channel_to_row = {int(c): i for i, c in enumerate(unique_channels)}
    patch_to_col = {int(p): i for i, p in enumerate(unique_patches)}

    for local_idx, (score, ch, patch) in enumerate(zip(s_scores, display_channels, s_patches)):
        ch = 0 if int(ch) == -1 else int(ch)
        row = channel_to_row[ch]
        col = patch_to_col[int(patch)]
        if np.isnan(heat[row, col]) or float(score) > float(heat[row, col]):
            heat[row, col] = float(score)
        if s_labels is not None:
            lab[row, col] = max(lab[row, col], int(s_labels[local_idx]))
        if s_gstart is not None and s_gend is not None:
            hover[row, col] = f"global rows: {int(s_gstart[local_idx])}–{int(s_gend[local_idx])}"

    x_labels = [f"patch_{int(p)}" for p in unique_patches]
    y_labels = [f"channel_{int(c)}" for c in unique_channels]

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=heat,
            x=x_labels,
            y=y_labels,
            customdata=hover,
            colorbar={"title": "score"},
            hovertemplate="%{y}<br>%{x}<br>score=%{z}<br>%{customdata}<extra></extra>",
        )
    )

    anomaly_rows, anomaly_cols = np.where(lab == 1)
    if anomaly_rows.size > 0:
        fig.add_trace(
            go.Scatter(
                x=[x_labels[c] for c in anomaly_cols],
                y=[y_labels[r] for r in anomaly_rows],
                mode="markers",
                name="GT anomaly patch",
                marker={
                    "symbol": "x",
                    "size": 13,
                    "line": {"width": 2},
                },
                hovertemplate="ground truth anomaly<br>%{y}<br>%{x}<extra></extra>",
            )
        )

    fig.update_layout(
        title=f"Timestamp score heatmap with ground-truth overlay — sample {sample_id}",
        xaxis_title="patch",
        yaxis_title="channel",
        height=max(420, 80 + 36 * len(y_labels)),
        margin={"l": 70, "r": 40, "t": 80, "b": 70},
    )
    return fig, sample_id


def dataframe_to_html_table(df, max_rows: int, max_cols: int) -> str:
    if df is None or df.empty:
        return "<p class='muted'>Nincs megjeleníthető CSV-részlet.</p>"
    view = df.iloc[:max_rows, :max_cols].copy()
    return view.to_html(index=False, escape=True, border=0, classes="data-table")


def top_window_snippets_html(
    csv_path: Optional[Path],
    scores: Dict[str, np.ndarray],
    labels: Dict[str, np.ndarray],
    sample_score_key: Optional[str],
    seq_len: Optional[int],
    top_windows: int,
    snippet_rows: int,
    value_columns: Optional[Sequence[str]],
) -> str:
    if csv_path is None or not csv_path.exists() or sample_score_key is None or sample_score_key not in scores:
        return "<p class='muted'>Forecasting CSV vagy sample-szintű score hiányzik, ezért a top window CSV-részletek nem készültek el.</p>"

    window_start = arr1d(labels, "window_start_index", np.int64)
    if window_start is None:
        return "<p class='muted'>A labels fájlban nincs window_start_index, ezért nem lehet globális CSV-részletet kötni a top window-khoz.</p>"

    sample_scores = arr1d(scores, sample_score_key, np.float32)
    y, group = label_for_level(labels, "instance")
    if sample_scores is None or sample_scores.shape[0] != window_start.shape[0]:
        return "<p class='muted'>A sample score hossza nem illeszkedik a window_start_index hosszához.</p>"

    import pandas as pd

    df = pd.read_csv(csv_path)
    seq_len = int(seq_len or 0)
    if seq_len <= 0:
        seq_len = int(np.median(np.diff(window_start))) if window_start.size > 1 else 1
        seq_len = max(seq_len, 1)

    selected_cols = []
    common_date = next((c for c in ["date", "datetime", "time", "timestamp"] if c in df.columns), None)
    if common_date:
        selected_cols.append(common_date)
    selected_cols.extend(choose_numeric_columns(df, value_columns, max_columns=8))
    selected_cols = list(dict.fromkeys(selected_cols))

    valid = np.isfinite(sample_scores) & (window_start >= 0)
    if not np.any(valid):
        return "<p class='muted'>Nincs véges sample score.</p>"

    order = np.argsort(-np.where(valid, sample_scores, -np.inf))[:top_windows]
    blocks = []
    for rank, sample_id in enumerate(order, start=1):
        start = int(window_start[sample_id])
        end = min(len(df), start + seq_len)
        snippet_end = min(end, start + snippet_rows)
        label = int(y[sample_id]) if y is not None and sample_id < y.shape[0] else None
        group_label = int(group[sample_id]) if group is not None and sample_id < group.shape[0] else None
        group_name = GROUP_NAMES.get(group_label, "—") if group_label is not None else "—"
        snippet = df.loc[start:snippet_end - 1, selected_cols] if selected_cols else df.iloc[start:snippet_end]
        table = dataframe_to_html_table(snippet, max_rows=snippet_rows, max_cols=12)
        more = "" if snippet_end >= end else f"<p class='muted'>Csak az első {snippet_rows} sor látszik a {seq_len} hosszú window-ból.</p>"
        blocks.append(
            "<details class='snippet-block'>"
            f"<summary>#{rank} | sample={sample_id} | rows={start}:{end} | score={format_float(sample_scores[sample_id])} | label={label if label is not None else '—'} | group={html.escape(group_name)}</summary>"
            f"{table}{more}"
            "</details>"
        )

    return "\n".join(blocks)


def figure_to_html(fig, include_plotlyjs: bool) -> str:
    if fig is None:
        return ""
    import plotly.io as pio

    return pio.to_html(
        fig,
        include_plotlyjs="cdn" if include_plotlyjs else False,
        full_html=False,
        config={"responsive": True, "displaylogo": False},
    )


def make_counts_table(title: str, counts: Dict[str, int]) -> str:
    if not counts:
        return f"<div class='card muted'><h3>{html.escape(title)}</h3><p>Nincs group label.</p></div>"
    rows = "".join(
        f"<tr><th>{html.escape(name)}</th><td>{format_int(value)}</td></tr>"
        for name, value in counts.items()
    )
    return f"<div class='card'><h3>{html.escape(title)}</h3><table>{rows}</table></div>"


def make_bank_shape_table(train_bank: Dict[str, np.ndarray], test_bank: Dict[str, np.ndarray]) -> str:
    keys = [
        "instance_embeddings",
        "timestamp_embeddings",
        "instance_sample_index",
        "timestamp_sample_index",
        "sample_labels",
        "window_start_index",
    ]
    rows = []
    for key in keys:
        rows.append(
            "<tr>"
            f"<th>{html.escape(key)}</th>"
            f"<td>{shape_or_dash(train_bank, key)}</td>"
            f"<td>{shape_or_dash(test_bank, key)}</td>"
            "</tr>"
        )
    return (
        "<div class='card'><h3>Embedding bank shape-ek</h3>"
        "<table><thead><tr><th>key</th><th>train/reference</th><th>test/query</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
    )


def make_config_panel(
    run_name: str,
    task_type: str,
    model_for: str,
    detector_cfg: Dict[str, Any],
    train_meta: Dict[str, Any],
    test_meta: Dict[str, Any],
) -> str:
    rows = [
        ("run_name", run_name),
        ("task_type", task_type),
        ("model_for", model_for),
        ("detector_type", detector_cfg.get("detector_type")),
        ("k", detector_cfg.get("k")),
        ("metric", detector_cfg.get("metric")),
        ("score_mode", detector_cfg.get("score_mode")),
        ("normalization", detector_cfg.get("normalization")),
        ("threshold_quantile", detector_cfg.get("threshold_quantile")),
        ("instance_threshold", detector_cfg.get("instance_threshold")),
        ("timestamp_threshold", detector_cfg.get("timestamp_threshold")),
        ("embedding_view", test_meta.get("embedding_view") or train_meta.get("embedding_view")),
        ("enable_channel_independence", test_meta.get("enable_channel_independence") or train_meta.get("enable_channel_independence")),
        ("seq_len", test_meta.get("seq_len") or train_meta.get("seq_len")),
        ("patch_len", test_meta.get("patch_len") or train_meta.get("patch_len")),
        ("stride", test_meta.get("stride") or train_meta.get("stride")),
    ]
    html_rows = []
    for key, value in rows:
        if isinstance(value, float):
            value_s = format_float(value)
        else:
            value_s = "—" if value is None else str(value)
        html_rows.append(f"<tr><th>{html.escape(key)}</th><td>{html.escape(value_s)}</td></tr>")

    return (
        "<div class='card'><h3>Run és detector konfiguráció</h3>"
        f"<table>{''.join(html_rows)}</table>"
        "</div>"
    )


def infer_task_type(args: argparse.Namespace, train_meta: Dict[str, Any], test_meta: Dict[str, Any], labels: Dict[str, np.ndarray]) -> str:
    if args.task_type:
        return args.task_type
    for meta in [test_meta, train_meta]:
        if meta.get("task_name"):
            return str(meta["task_name"])
    if "timestamp_global_start_index" in labels or "window_start_index" in labels:
        return "forecasting"
    return "classification"


def infer_model_for(args: argparse.Namespace, train_meta: Dict[str, Any], test_meta: Dict[str, Any]) -> str:
    if args.model_for:
        return args.model_for
    for meta in [test_meta, train_meta]:
        if meta.get("model_for"):
            return str(meta["model_for"])
        if meta.get("data_name"):
            return str(meta["data_name"])
    return "unknown"


def create_dashboard(args: argparse.Namespace) -> Path:
    try:
        import plotly  # noqa: F401
    except ImportError as exc:
        raise ImportError("plotly is required. Install with: pip install plotly") from exc

    scores_path = Path(args.scores_path)
    labels_path = Path(args.labels_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scores = load_npz(scores_path)
    labels = load_npz(labels_path)
    train_bank = load_npz(args.train_bank_path) if args.train_bank_path else {}
    test_bank = load_npz(args.test_bank_path) if args.test_bank_path else {}
    mask_data = load_npz(args.mask_path) if args.mask_path else {}

    train_meta_path = maybe_path(args.train_bank_meta_path) or infer_sidecar_meta(args.train_bank_path)
    test_meta_path = maybe_path(args.test_bank_meta_path) or infer_sidecar_meta(args.test_bank_path)
    detector_meta_path = maybe_path(args.detector_meta_path) or infer_detector_meta(scores_path)

    train_meta = read_meta(train_meta_path)
    test_meta = read_meta(test_meta_path)
    detector_meta = read_meta(detector_meta_path)

    metrics_path = maybe_path(args.metrics_path)
    if metrics_path is None and args.evaluation_reports_dir:
        metrics_path = infer_metrics_path(scores_path, args.evaluation_reports_dir)
    metrics = read_meta(metrics_path)

    config = read_meta(args.config_path) if args.config_path else {}
    run_summary = read_meta(args.run_summary_path) if args.run_summary_path else {}

    task_type = infer_task_type(args, train_meta, test_meta, labels)
    model_for = infer_model_for(args, train_meta, test_meta)
    output_name = args.output_name or scores_path.name.replace(".scores.npz", "")
    run_name = args.run_name or run_summary.get("output_name") or output_name

    detector_cfg = extract_detector_config(detector_meta, scores)

    instance_score_key = auto_score_key(
        scores,
        args.instance_score_key,
        ["sample_instance_score_max", "sample_instance_score_mean", "instance_scores"],
    )
    timestamp_score_key = auto_score_key(
        scores,
        args.timestamp_score_key,
        ["timestamp_scores", "sample_timestamp_score_max", "sample_timestamp_score_mean"],
    )

    sample_y, sample_group = label_for_level(labels, "instance")
    timestamp_y, timestamp_group = label_for_level(labels, "timestamp")

    html_parts: List[str] = []
    include_plotly = True

    def add_fig(fig) -> None:
        nonlocal include_plotly
        if fig is None:
            return
        html_parts.append("<div class='card plot-card'>" + figure_to_html(fig, include_plotly) + "</div>")
        include_plotly = False

    config_panel = make_config_panel(
        run_name=run_name,
        task_type=task_type,
        model_for=model_for,
        detector_cfg=detector_cfg,
        train_meta=train_meta,
        test_meta=test_meta,
    )
    bank_panel = make_bank_shape_table(train_bank, test_bank)
    counts_panel = (
        make_counts_table("Sample ID / near / far darabszámok", count_groups(sample_group))
        + make_counts_table("Timestamp/Patch ID / near / far darabszámok", count_groups(timestamp_group))
    )

    metric_section = (
        "<div class='two-col'>"
        + metric_cards(metrics, "instance_level", "Instance/sample-level metrikák")
        + metric_cards(metrics, "timestamp_level", "Timestamp/patch-level metrikák")
        + "</div>"
    )

    html_parts.append("<section id='overview'><h2>Áttekintés</h2>" + config_panel + bank_panel + counts_panel + metric_section + "</section>")

    html_parts.append("<section id='confusion'><h2>Confusion matrix</h2><div class='two-col'>")
    fig_cm_instance = make_confusion_figure(metrics, "instance_level", "Instance/sample-level confusion matrix")
    fig_cm_timestamp = make_confusion_figure(metrics, "timestamp_level", "Timestamp/patch-level confusion matrix")
    if fig_cm_instance is None and fig_cm_timestamp is None:
        html_parts.append("<div class='card muted'><p>Nincs threshold alapú confusion matrix a metrics summaryban.</p></div>")
    else:
        if fig_cm_instance is not None:
            html_parts.append("<div class='card plot-card'>" + figure_to_html(fig_cm_instance, include_plotly) + "</div>")
            include_plotly = False
        if fig_cm_timestamp is not None:
            html_parts.append("<div class='card plot-card'>" + figure_to_html(fig_cm_timestamp, include_plotly) + "</div>")
            include_plotly = False
    html_parts.append("</div></section>")

    html_parts.append("<section id='scores'><h2>Score distribution label szerint</h2>")
    if instance_score_key:
        add_fig(make_score_distribution_figure(scores, labels, instance_score_key, "instance", f"Instance/sample score distribution — {instance_score_key}"))
    if timestamp_score_key:
        add_fig(make_score_distribution_figure(scores, labels, timestamp_score_key, "timestamp", f"Timestamp/patch score distribution — {timestamp_score_key}"))
    html_parts.append("</section>")

    if task_type == "forecasting":
        html_parts.append("<section id='forecasting'><h2>Forecasting-specifikus nézetek</h2>")
        timeline_csv = first_existing_path([args.injected_csv_path, args.original_csv_path])
        add_fig(
            make_forecasting_timeline_figure(
                csv_path=timeline_csv,
                mask_data=mask_data,
                date_column=args.date_column,
                value_columns=args.value_columns,
                max_points=args.max_timeline_points,
                max_channels=args.max_timeline_channels,
            )
        )

        seq_len = args.seq_len or test_meta.get("seq_len") or train_meta.get("seq_len")
        html_parts.append("<div class='card'><h3>Top anomália window-k CSV-részletekkel</h3>")
        html_parts.append(
            top_window_snippets_html(
                csv_path=first_existing_path([args.original_csv_path, args.injected_csv_path]),
                scores=scores,
                labels=labels,
                sample_score_key=instance_score_key,
                seq_len=int(seq_len) if seq_len else None,
                top_windows=args.top_windows,
                snippet_rows=args.snippet_rows,
                value_columns=args.value_columns,
            )
        )
        html_parts.append("</div>")
        html_parts.append("</section>")

    html_parts.append("<section id='heatmap'><h2>Sample heatmap ground-truth overlayjel</h2>")
    heatmap_fig, chosen_sample = make_sample_heatmap_with_overlay(scores, labels, args.sample_id)
    if heatmap_fig is None:
        html_parts.append("<div class='card muted'><p>Nincs elegendő timestamp mapping/score a heatmaphez.</p></div>")
    else:
        html_parts.append(f"<p class='muted'>Megjelenített sample_id: {chosen_sample}</p>")
        add_fig(heatmap_fig)
    html_parts.append("</section>")

    html_parts.append("<section id='artifacts'><h2>Felhasznált artifactok</h2><div class='card'><table>")
    artifact_rows = [
        ("scores_path", scores_path),
        ("labels_path", labels_path),
        ("metrics_path", metrics_path),
        ("train_bank_path", args.train_bank_path),
        ("test_bank_path", args.test_bank_path),
        ("train_bank_meta_path", train_meta_path),
        ("test_bank_meta_path", test_meta_path),
        ("detector_meta_path", detector_meta_path),
        ("mask_path", args.mask_path),
        ("original_csv_path", args.original_csv_path),
        ("injected_csv_path", args.injected_csv_path),
        ("config_path", args.config_path),
        ("run_summary_path", args.run_summary_path),
    ]
    for key, value in artifact_rows:
        value_s = "—" if value is None else str(value)
        html_parts.append(f"<tr><th>{html.escape(key)}</th><td>{html.escape(value_s)}</td></tr>")
    html_parts.append("</table></div></section>")

    if config:
        html_parts.append("<section id='config-json'><h2>Config JSON</h2><div class='card'><pre>")
        html_parts.append(html.escape(json.dumps(config, indent=2, ensure_ascii=False)))
        html_parts.append("</pre></div></section>")

    css = """
    :root {
        --bg: #f7f7fb;
        --card: #ffffff;
        --text: #1f2937;
        --muted: #6b7280;
        --border: #e5e7eb;
        --accent: #334155;
    }
    body {
        margin: 0;
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        background: var(--bg);
        color: var(--text);
        line-height: 1.45;
    }
    header {
        padding: 28px 38px;
        background: #111827;
        color: white;
    }
    header h1 { margin: 0 0 8px 0; font-size: 28px; }
    header p { margin: 0; color: #d1d5db; }
    main { max-width: 1440px; margin: 0 auto; padding: 26px; }
    section { margin: 0 0 30px 0; }
    h2 { margin: 26px 0 14px 0; font-size: 22px; }
    h3 { margin: 0 0 12px 0; font-size: 17px; }
    .card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 16px;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
        padding: 18px;
        margin: 14px 0;
        overflow-x: auto;
    }
    .plot-card { padding: 8px 12px 4px 12px; }
    .two-col {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(440px, 1fr));
        gap: 16px;
        align-items: start;
    }
    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
        gap: 12px;
    }
    .metric-card {
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 14px;
        background: #fafafa;
    }
    .metric-label { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .05em; }
    .metric-value { font-size: 24px; font-weight: 700; margin-top: 4px; }
    .muted { color: var(--muted); }
    table { width: 100%; border-collapse: collapse; font-size: 14px; }
    th, td { text-align: left; border-bottom: 1px solid var(--border); padding: 8px 10px; vertical-align: top; }
    th { width: 260px; color: var(--accent); font-weight: 650; background: #fafafa; }
    .data-table th { width: auto; }
    pre {
        white-space: pre-wrap;
        word-break: break-word;
        background: #0f172a;
        color: #e5e7eb;
        padding: 18px;
        border-radius: 12px;
        overflow-x: auto;
    }
    details.snippet-block {
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 10px 12px;
        margin: 12px 0;
        background: #fcfcfd;
    }
    details.snippet-block summary {
        cursor: pointer;
        font-weight: 650;
        color: #111827;
    }
    """

    final_html = (
        "<!doctype html><html lang='hu'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>TimeDRL OOD Dashboard — {html.escape(run_name)}</title>"
        f"<style>{css}</style></head><body>"
        "<header>"
        f"<h1>TimeDRL OOD / anomaly dashboard</h1>"
        f"<p>{html.escape(model_for)} · {html.escape(task_type)} · {html.escape(run_name)}</p>"
        "</header><main>"
        + "\n".join(html_parts)
        + "</main></body></html>"
    )

    output_path = output_dir / f"{output_name}.experiment_dashboard.html"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_html)

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a single HTML dashboard for a TimeDRL OOD/anomaly experiment."
    )

    parser.add_argument("--scores_path", type=str, required=True)
    parser.add_argument("--labels_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./ood/score_visualizations")
    parser.add_argument("--output_name", type=str, default=None)

    parser.add_argument("--metrics_path", type=str, default=None)
    parser.add_argument("--evaluation_reports_dir", type=str, default=None)

    parser.add_argument("--train_bank_path", type=str, default=None)
    parser.add_argument("--test_bank_path", type=str, default=None)
    parser.add_argument("--train_bank_meta_path", type=str, default=None)
    parser.add_argument("--test_bank_meta_path", type=str, default=None)
    parser.add_argument("--detector_meta_path", type=str, default=None)

    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--run_summary_path", type=str, default=None)

    parser.add_argument("--task_type", type=str, choices=["forecasting", "classification"], default=None)
    parser.add_argument("--model_for", type=str, default=None)
    parser.add_argument("--run_name", type=str, default=None)

    parser.add_argument("--instance_score_key", type=str, default=None)
    parser.add_argument("--timestamp_score_key", type=str, default=None)

    # Forecasting-specific optional inputs.
    parser.add_argument("--mask_path", type=str, default=None)
    parser.add_argument("--original_csv_path", type=str, default=None)
    parser.add_argument("--injected_csv_path", type=str, default=None)
    parser.add_argument("--date_column", type=str, default=None)
    parser.add_argument("--value_columns", nargs="*", default=None)
    parser.add_argument("--seq_len", type=int, default=None)

    parser.add_argument("--sample_id", type=int, default=None)
    parser.add_argument("--top_windows", type=int, default=10)
    parser.add_argument("--snippet_rows", type=int, default=32)
    parser.add_argument("--max_timeline_points", type=int, default=5000)
    parser.add_argument("--max_timeline_channels", type=int, default=4)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = create_dashboard(args)
    print(f"Saved experiment dashboard: {output_path}")


if __name__ == "__main__":
    main()
