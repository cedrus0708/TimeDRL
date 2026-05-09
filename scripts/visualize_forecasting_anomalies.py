import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots


def detect_time_column(df: pd.DataFrame) -> Optional[str]:
    candidates = ["date", "datetime", "timestamp", "time"]
    lower_map = {c.lower(): c for c in df.columns}

    for cand in candidates:
        if cand in lower_map:
            return lower_map[cand]

    non_numeric_cols = [
        c for c in df.columns
        if not pd.api.types.is_numeric_dtype(df[c])
    ]
    if len(non_numeric_cols) > 0:
        return non_numeric_cols[0]

    return None


def get_numeric_columns(df: pd.DataFrame, time_col: Optional[str]) -> list[str]:
    cols = []
    for col in df.columns:
        if col == time_col:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            cols.append(col)
    return cols


def maybe_parse_time_series(x: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(x)
    except Exception:
        return x


def load_events(events_path: Path) -> pd.DataFrame:
    with open(events_path, "r", encoding="utf-8") as f:
        events = json.load(f)

    if len(events) == 0:
        return pd.DataFrame(columns=[
            "event_id", "anomaly_type", "start", "end", "channels",
            "params", "source_dataset", "source_start", "source_end"
        ])

    df = pd.DataFrame(events)

    if "channels" in df.columns:
        df["channels_str"] = df["channels"].apply(
            lambda x: ", ".join(x) if isinstance(x, list) else str(x)
        )
    else:
        df["channels_str"] = ""

    if "params" in df.columns:
        df["params_str"] = df["params"].apply(
            lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, dict) else str(x)
        )
    else:
        df["params_str"] = ""

    return df


def choose_channels(
    numeric_cols: list[str],
    user_channels: Optional[str],
    max_channels: int,
) -> list[str]:
    if user_channels is not None and user_channels.strip():
        selected = [c.strip() for c in user_channels.split(",")]
        missing = [c for c in selected if c not in numeric_cols]
        if missing:
            raise ValueError(f"Unknown channel(s): {missing}")
        return selected

    return numeric_cols[:max_channels]


def build_timeseries_figure(
    clean_df: pd.DataFrame,
    anom_df: pd.DataFrame,
    labels_timestamp_df: pd.DataFrame,
    events_df: pd.DataFrame,
    time_col: Optional[str],
    selected_channels: list[str],
) -> go.Figure:
    n_channels = len(selected_channels)

    fig = make_subplots(
        rows=n_channels + 1,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.18] * n_channels + [0.10],
        subplot_titles=selected_channels + ["Timestamp anomaly label"],
    )

    if time_col is not None:
        x = maybe_parse_time_series(anom_df[time_col])
    else:
        x = np.arange(len(anom_df))

    for i, col in enumerate(selected_channels, start=1):
        fig.add_trace(
            go.Scatter(
                x=x,
                y=clean_df[col],
                mode="lines",
                name=f"{col} - clean",
                line=dict(width=1),
                opacity=0.65,
                legendgroup=f"{col}",
                showlegend=(i == 1),
            ),
            row=i,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=x,
                y=anom_df[col],
                mode="lines",
                name=f"{col} - anomalous",
                line=dict(width=1.6),
                legendgroup=f"{col}",
                showlegend=(i == 1),
            ),
            row=i,
            col=1,
        )

    label_x = x
    label_y = labels_timestamp_df["is_anomaly"].to_numpy()

    fig.add_trace(
        go.Scatter(
            x=label_x,
            y=label_y,
            mode="lines",
            fill="tozeroy",
            name="is_anomaly",
            line=dict(width=1.2),
            showlegend=True,
        ),
        row=n_channels + 1,
        col=1,
    )

    color_map = {
        "spike": "rgba(255, 0, 0, 0.15)",
        "noise_segment": "rgba(255, 165, 0, 0.15)",
        "level_shift": "rgba(0, 128, 255, 0.15)",
        "scale_segment": "rgba(128, 0, 255, 0.15)",
        "flatline": "rgba(0, 180, 0, 0.15)",
        "dropout": "rgba(80, 80, 80, 0.18)",
        "cross_dataset_segment": "rgba(255, 0, 180, 0.18)",
    }

    for _, ev in events_df.iterrows():
        start = int(ev["start"])
        end = int(ev["end"])
        anomaly_type = ev.get("anomaly_type", "unknown")
        fillcolor = color_map.get(anomaly_type, "rgba(120, 120, 120, 0.12)")

        x0 = x.iloc[start] if hasattr(x, "iloc") else x[start]
        x1 = x.iloc[end - 1] if hasattr(x, "iloc") else x[end - 1]

        fig.add_vrect(
            x0=x0,
            x1=x1,
            fillcolor=fillcolor,
            opacity=1.0,
            line_width=0,
            layer="below",
            row="all",
            col=1,
            annotation_text=anomaly_type,
            annotation_position="top left",
        )

    fig.update_layout(
        title="Synthetic anomaly viewer: clean vs anomalous series",
        height=280 * n_channels + 240,
        hovermode="x unified",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
        margin=dict(l=60, r=30, t=80, b=50),
    )

    for r in range(1, n_channels + 1):
        fig.update_yaxes(title_text=selected_channels[r - 1], row=r, col=1)

    fig.update_yaxes(title_text="anomaly", row=n_channels + 1, col=1, range=[-0.05, 1.05])
    fig.update_xaxes(rangeslider_visible=(n_channels + 1 == n_channels + 1), row=n_channels + 1, col=1)

    return fig


def build_difference_heatmap(
    clean_df: pd.DataFrame,
    anom_df: pd.DataFrame,
    time_col: Optional[str],
    numeric_cols: list[str],
    selected_channels: list[str],
) -> go.Figure:
    if time_col is not None:
        x = maybe_parse_time_series(anom_df[time_col])
    else:
        x = np.arange(len(anom_df))

    clean = clean_df[selected_channels].to_numpy(dtype=float)
    anom = anom_df[selected_channels].to_numpy(dtype=float)

    diff = anom - clean
    std = np.nanstd(clean, axis=0)
    std = np.where(std < 1e-8, 1.0, std)

    zdiff = np.abs(diff / std.reshape(1, -1))

    fig = go.Figure(
        data=go.Heatmap(
            z=zdiff.T,
            x=x,
            y=selected_channels,
            colorbar=dict(title="|Δ| / std"),
            hovertemplate="channel=%{y}<br>x=%{x}<br>|Δ|/std=%{z:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title="Normalized anomaly intensity heatmap",
        template="plotly_white",
        height=max(350, 80 + 45 * len(selected_channels)),
        margin=dict(l=70, r=40, t=70, b=50),
    )

    return fig


def build_event_table_html(events_df: pd.DataFrame) -> str:
    if len(events_df) == 0:
        return "<p>No events found.</p>"

    cols = [
        "event_id",
        "anomaly_type",
        "start",
        "end",
        "channels_str",
        "source_dataset",
        "source_start",
        "source_end",
        "params_str",
    ]
    available_cols = [c for c in cols if c in events_df.columns]

    table_df = events_df[available_cols].copy()
    table_df = table_df.rename(columns={
        "channels_str": "channels",
        "params_str": "params",
    })

    return table_df.to_html(index=False, escape=False, classes="event-table")


def build_summary_html(
    clean_df: pd.DataFrame,
    anom_df: pd.DataFrame,
    labels_timestamp_df: pd.DataFrame,
    events_df: pd.DataFrame,
    selected_channels: list[str],
) -> str:
    total_points = len(anom_df)
    anomalous_points = int(labels_timestamp_df["is_anomaly"].sum())
    ratio = anomalous_points / total_points if total_points > 0 else 0.0

    type_counts = (
        events_df["anomaly_type"].value_counts().to_dict()
        if "anomaly_type" in events_df.columns and len(events_df) > 0
        else {}
    )

    type_counts_html = "".join(
        f"<li><b>{k}</b>: {v}</li>" for k, v in type_counts.items()
    )

    return f"""
    <div class="summary-box">
        <h2>Summary</h2>
        <ul>
            <li><b>Total timestamps:</b> {total_points}</li>
            <li><b>Anomalous timestamps:</b> {anomalous_points}</li>
            <li><b>Anomalous ratio:</b> {ratio:.4f}</li>
            <li><b>Selected channels:</b> {", ".join(selected_channels)}</li>
            <li><b>Number of events:</b> {len(events_df)}</li>
        </ul>
        <h3>Event type counts</h3>
        <ul>
            {type_counts_html if type_counts_html else "<li>No events</li>"}
        </ul>
    </div>
    """


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--clean_csv", type=str, required=True)
    parser.add_argument("--anomalous_csv", type=str, required=True)
    parser.add_argument("--labels_timestamp_csv", type=str, required=True)
    parser.add_argument("--events_json", type=str, required=True)
    parser.add_argument("--out_html", type=str, required=True)

    parser.add_argument(
        "--channels",
        type=str,
        default=None,
        help="Comma-separated channel list, e.g. 'OT,HUFL,HULL'",
    )
    parser.add_argument("--max_channels", type=int, default=6)

    args = parser.parse_args()

    clean_path = Path(args.clean_csv)
    anom_path = Path(args.anomalous_csv)
    labels_path = Path(args.labels_timestamp_csv)
    events_path = Path(args.events_json)
    out_html_path = Path(args.out_html)
    out_html_path.parent.mkdir(parents=True, exist_ok=True)

    clean_df = pd.read_csv(clean_path)
    anom_df = pd.read_csv(anom_path)
    labels_timestamp_df = pd.read_csv(labels_path)
    events_df = load_events(events_path)

    time_col = detect_time_column(anom_df)
    numeric_cols = get_numeric_columns(anom_df, time_col)

    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found in anomalous CSV.")

    selected_channels = choose_channels(
        numeric_cols=numeric_cols,
        user_channels=args.channels,
        max_channels=args.max_channels,
    )

    missing_in_clean = [c for c in selected_channels if c not in clean_df.columns]
    missing_in_anom = [c for c in selected_channels if c not in anom_df.columns]

    if missing_in_clean:
        raise ValueError(f"Selected channels missing from clean CSV: {missing_in_clean}")
    if missing_in_anom:
        raise ValueError(f"Selected channels missing from anomalous CSV: {missing_in_anom}")

    fig_ts = build_timeseries_figure(
        clean_df=clean_df,
        anom_df=anom_df,
        labels_timestamp_df=labels_timestamp_df,
        events_df=events_df,
        time_col=time_col,
        selected_channels=selected_channels,
    )

    fig_heat = build_difference_heatmap(
        clean_df=clean_df,
        anom_df=anom_df,
        time_col=time_col,
        numeric_cols=numeric_cols,
        selected_channels=selected_channels,
    )

    summary_html = build_summary_html(
        clean_df=clean_df,
        anom_df=anom_df,
        labels_timestamp_df=labels_timestamp_df,
        events_df=events_df,
        selected_channels=selected_channels,
    )

    event_table_html = build_event_table_html(events_df)

    ts_html = pio.to_html(fig_ts, include_plotlyjs="cdn", full_html=False)
    heat_html = pio.to_html(fig_heat, include_plotlyjs=False, full_html=False)

    full_html = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <title>Synthetic anomaly viewer</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 24px;
                background: #fafafa;
                color: #222;
            }}
            h1, h2, h3 {{
                margin-top: 28px;
            }}
            .summary-box {{
                background: white;
                border: 1px solid #ddd;
                padding: 16px;
                border-radius: 10px;
                margin-bottom: 24px;
            }}
            .event-table {{
                border-collapse: collapse;
                width: 100%;
                background: white;
            }}
            .event-table th, .event-table td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
                vertical-align: top;
                font-size: 13px;
            }}
            .event-table th {{
                background: #f0f0f0;
            }}
            .plot-box {{
                background: white;
                border: 1px solid #ddd;
                border-radius: 10px;
                padding: 12px;
                margin-bottom: 24px;
            }}
            .note {{
                font-size: 14px;
                color: #444;
                margin-bottom: 18px;
            }}
        </style>
    </head>
    <body>
        <h1>Synthetic anomaly viewer</h1>

        <p class="note">
            This report compares the clean and anomalous versions of the dataset,
            highlights injected anomaly regions, shows the normalized change magnitude,
            and lists all injected events.
        </p>

        {summary_html}

        <h2>1. Clean vs anomalous time series</h2>
        <div class="plot-box">
            {ts_html}
        </div>

        <h2>2. Normalized anomaly intensity heatmap</h2>
        <div class="plot-box">
            {heat_html}
        </div>

        <h2>3. Injected event list</h2>
        <div class="summary-box">
            {event_table_html}
        </div>
    </body>
    </html>
    """

    with open(out_html_path, "w", encoding="utf-8") as f:
        f.write(full_html)

    print(f"Viewer saved to: {out_html_path}")


if __name__ == "__main__":
    main()