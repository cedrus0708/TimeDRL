import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class AnomalyEvent:
    event_id: int
    anomaly_type: str
    start: int
    end: int
    channels: list[str]
    params: dict
    source_dataset: Optional[str] = None
    source_start: Optional[int] = None
    source_end: Optional[int] = None


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


def safe_std(x: np.ndarray, axis=0) -> np.ndarray:
    std = np.nanstd(x, axis=axis)
    std = np.where(std < 1e-8, 1.0, std)
    return std


def choose_channels(
    rng: np.random.Generator,
    numeric_cols: list[str],
    mode: str,
    max_channels: int,
) -> list[int]:
    c = len(numeric_cols)

    if mode == "all":
        return list(range(c))

    if mode == "single":
        return [int(rng.integers(0, c))]

    if mode == "random_subset":
        k = int(rng.integers(1, min(max_channels, c) + 1))
        return sorted(rng.choice(c, size=k, replace=False).tolist())

    raise ValueError(f"Unknown channel mode: {mode}")


def sample_non_overlapping_interval(
    rng: np.random.Generator,
    occupied: np.ndarray,
    region_start: int,
    region_end: int,
    length: int,
    padding: int = 0,
    max_tries: int = 5000,
) -> tuple[int, int]:
    if region_end - region_start < length:
        raise ValueError("Injection region is shorter than requested anomaly length.")

    for _ in range(max_tries):
        start = int(rng.integers(region_start, region_end - length + 1))
        end = start + length

        check_start = max(region_start, start - padding)
        check_end = min(region_end, end + padding)

        if not occupied[check_start:check_end].any():
            occupied[check_start:check_end] = True
            return start, end

    raise RuntimeError(
        "Could not sample a non-overlapping interval. "
        "Try fewer anomalies, shorter lengths, or smaller padding."
    )


def mark_labels(
    labels_timestamp: np.ndarray,
    labels_channel: np.ndarray,
    start: int,
    end: int,
    channel_indices: list[int],
):
    labels_timestamp[start:end] = 1
    labels_channel[start:end, channel_indices] = 1


def inject_spike(
    x: np.ndarray,
    rng: np.random.Generator,
    start: int,
    channel_indices: list[int],
    global_std: np.ndarray,
    severity: float,
):
    signs = rng.choice([-1.0, 1.0], size=len(channel_indices))
    x[start, channel_indices] += signs * severity * global_std[channel_indices]


def inject_noise_segment(
    x: np.ndarray,
    rng: np.random.Generator,
    start: int,
    end: int,
    channel_indices: list[int],
    global_std: np.ndarray,
    severity: float,
):
    length = end - start
    noise = rng.normal(
        loc=0.0,
        scale=severity * global_std[channel_indices],
        size=(length, len(channel_indices)),
    )
    x[start:end, channel_indices] += noise


def inject_level_shift(
    x: np.ndarray,
    rng: np.random.Generator,
    start: int,
    end: int,
    channel_indices: list[int],
    global_std: np.ndarray,
    severity: float,
):
    signs = rng.choice([-1.0, 1.0], size=len(channel_indices))
    shift = signs * severity * global_std[channel_indices]
    x[start:end, channel_indices] += shift


def inject_scale_segment(
    x: np.ndarray,
    rng: np.random.Generator,
    start: int,
    end: int,
    channel_indices: list[int],
    min_factor: float,
    max_factor: float,
):
    factors = rng.uniform(min_factor, max_factor, size=len(channel_indices))
    x[start:end, channel_indices] *= factors


def inject_flatline(
    x: np.ndarray,
    start: int,
    end: int,
    channel_indices: list[int],
):
    reference_idx = max(0, start - 1)
    x[start:end, channel_indices] = x[reference_idx, channel_indices]


def inject_dropout(
    x: np.ndarray,
    start: int,
    end: int,
    channel_indices: list[int],
    fill_value: float,
):
    x[start:end, channel_indices] = fill_value


def inject_cross_dataset_segment(
    x: np.ndarray,
    x_clean: np.ndarray,
    source_x: np.ndarray,
    rng: np.random.Generator,
    start: int,
    end: int,
    channel_indices: list[int],
    severity: float,
    context: int,
) -> tuple[int, int]:
    """
    Takes a segment from another dataset, e.g. Exchange, standardizes it,
    then maps it to the scale of the target Weather channels.

    This keeps the values numerically usable, but the temporal pattern comes
    from a foreign dataset, which should be a strong contextual anomaly.
    """
    length = end - start
    source_t, source_c = source_x.shape

    if source_t < length:
        raise ValueError("Source dataset is shorter than requested transplant length.")

    source_start = int(rng.integers(0, source_t - length + 1))
    source_end = source_start + length

    selected_source_channels = rng.choice(
        source_c,
        size=len(channel_indices),
        replace=source_c < len(channel_indices),
    )

    source_segment = source_x[source_start:source_end, selected_source_channels]

    source_mu = np.nanmean(source_segment, axis=0)
    source_sigma = safe_std(source_segment, axis=0)
    source_z = (source_segment - source_mu) / source_sigma

    ctx_start = max(0, start - context)
    ctx_end = min(x_clean.shape[0], end + context)

    target_context = x_clean[ctx_start:ctx_end, :][:, channel_indices]
    target_mu = np.nanmean(target_context, axis=0)
    target_sigma = safe_std(target_context, axis=0)

    transplanted = target_mu + severity * target_sigma * source_z

    x[start:end, channel_indices] = transplanted

    return source_start, source_end


def add_event(
    events: list[AnomalyEvent],
    anomaly_type: str,
    start: int,
    end: int,
    channel_indices: list[int],
    numeric_cols: list[str],
    params: dict,
    source_dataset: Optional[str] = None,
    source_start: Optional[int] = None,
    source_end: Optional[int] = None,
):
    events.append(
        AnomalyEvent(
            event_id=len(events),
            anomaly_type=anomaly_type,
            start=start,
            end=end,
            channels=[numeric_cols[i] for i in channel_indices],
            params=params,
            source_dataset=source_dataset,
            source_start=source_start,
            source_end=source_end,
        )
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--target_csv", type=str, required=True)
    parser.add_argument("--source_csv", type=str, default=None)
    parser.add_argument("--out_dir", type=str, required=True)

    parser.add_argument("--seed", type=int, default=42)

    # By default, inject only into the last 30% of the time series.
    # This is safer because the train/validation part remains clean.
    parser.add_argument("--injection_start_ratio", type=float, default=0.7)
    parser.add_argument("--injection_end_ratio", type=float, default=1.0)

    parser.add_argument("--channel_mode", type=str, default="random_subset",
                        choices=["single", "random_subset", "all"])
    parser.add_argument("--max_channels", type=int, default=3)

    parser.add_argument("--min_segment_len", type=int, default=12)
    parser.add_argument("--max_segment_len", type=int, default=96)
    parser.add_argument("--event_padding", type=int, default=12)

    parser.add_argument("--n_spikes", type=int, default=30)
    parser.add_argument("--n_noise_segments", type=int, default=10)
    parser.add_argument("--n_level_shifts", type=int, default=10)
    parser.add_argument("--n_scale_segments", type=int, default=10)
    parser.add_argument("--n_flatlines", type=int, default=5)
    parser.add_argument("--n_dropouts", type=int, default=5)
    parser.add_argument("--n_cross_dataset_segments", type=int, default=8)

    parser.add_argument("--spike_severity", type=float, default=8.0)
    parser.add_argument("--noise_severity", type=float, default=3.0)
    parser.add_argument("--level_shift_severity", type=float, default=5.0)
    parser.add_argument("--scale_min_factor", type=float, default=2.0)
    parser.add_argument("--scale_max_factor", type=float, default=4.0)
    parser.add_argument("--dropout_fill_value", type=float, default=0.0)

    # For Exchange -> Weather transplant.
    # Higher value means stronger foreign pattern.
    parser.add_argument("--cross_dataset_severity", type=float, default=4.0)
    parser.add_argument("--cross_dataset_context", type=int, default=96)

    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    target_path = Path(args.target_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_df = pd.read_csv(target_path)
    time_col = detect_time_column(target_df)
    numeric_cols = get_numeric_columns(target_df, time_col)

    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns found in target CSV.")

    x_clean = target_df[numeric_cols].to_numpy(dtype=np.float64)
    x_anom = x_clean.copy()

    t, c = x_clean.shape

    region_start = int(t * args.injection_start_ratio)
    region_end = int(t * args.injection_end_ratio)

    if not (0 <= region_start < region_end <= t):
        raise ValueError("Invalid injection region ratios.")

    global_std = safe_std(x_clean, axis=0)

    labels_timestamp = np.zeros(t, dtype=np.int64)
    labels_channel = np.zeros((t, c), dtype=np.int64)
    occupied = np.zeros(t, dtype=bool)
    events: list[AnomalyEvent] = []

    source_x = None
    source_numeric_cols = None

    if args.n_cross_dataset_segments > 0:
        if args.source_csv is None:
            raise ValueError(
                "--source_csv is required when --n_cross_dataset_segments > 0"
            )

        source_df = pd.read_csv(args.source_csv)
        source_time_col = detect_time_column(source_df)
        source_numeric_cols = get_numeric_columns(source_df, source_time_col)

        if len(source_numeric_cols) == 0:
            raise ValueError("No numeric columns found in source CSV.")

        source_x = source_df[source_numeric_cols].to_numpy(dtype=np.float64)

    def random_segment_length() -> int:
        return int(rng.integers(args.min_segment_len, args.max_segment_len + 1))

    # 1. Point spikes
    for _ in range(args.n_spikes):
        start, end = sample_non_overlapping_interval(
            rng,
            occupied,
            region_start,
            region_end,
            length=1,
            padding=args.event_padding,
        )
        channel_indices = choose_channels(
            rng, numeric_cols, args.channel_mode, args.max_channels
        )

        inject_spike(
            x_anom,
            rng,
            start,
            channel_indices,
            global_std,
            severity=args.spike_severity,
        )

        mark_labels(labels_timestamp, labels_channel, start, end, channel_indices)

        add_event(
            events,
            "spike",
            start,
            end,
            channel_indices,
            numeric_cols,
            params={"severity": args.spike_severity},
        )

    # 2. Noisy segments
    for _ in range(args.n_noise_segments):
        length = random_segment_length()
        start, end = sample_non_overlapping_interval(
            rng,
            occupied,
            region_start,
            region_end,
            length=length,
            padding=args.event_padding,
        )
        channel_indices = choose_channels(
            rng, numeric_cols, args.channel_mode, args.max_channels
        )

        inject_noise_segment(
            x_anom,
            rng,
            start,
            end,
            channel_indices,
            global_std,
            severity=args.noise_severity,
        )

        mark_labels(labels_timestamp, labels_channel, start, end, channel_indices)

        add_event(
            events,
            "noise_segment",
            start,
            end,
            channel_indices,
            numeric_cols,
            params={"severity": args.noise_severity},
        )

    # 3. Level shifts
    for _ in range(args.n_level_shifts):
        length = random_segment_length()
        start, end = sample_non_overlapping_interval(
            rng,
            occupied,
            region_start,
            region_end,
            length=length,
            padding=args.event_padding,
        )
        channel_indices = choose_channels(
            rng, numeric_cols, args.channel_mode, args.max_channels
        )

        inject_level_shift(
            x_anom,
            rng,
            start,
            end,
            channel_indices,
            global_std,
            severity=args.level_shift_severity,
        )

        mark_labels(labels_timestamp, labels_channel, start, end, channel_indices)

        add_event(
            events,
            "level_shift",
            start,
            end,
            channel_indices,
            numeric_cols,
            params={"severity": args.level_shift_severity},
        )

    # 4. Scale anomalies
    for _ in range(args.n_scale_segments):
        length = random_segment_length()
        start, end = sample_non_overlapping_interval(
            rng,
            occupied,
            region_start,
            region_end,
            length=length,
            padding=args.event_padding,
        )
        channel_indices = choose_channels(
            rng, numeric_cols, args.channel_mode, args.max_channels
        )

        inject_scale_segment(
            x_anom,
            rng,
            start,
            end,
            channel_indices,
            min_factor=args.scale_min_factor,
            max_factor=args.scale_max_factor,
        )

        mark_labels(labels_timestamp, labels_channel, start, end, channel_indices)

        add_event(
            events,
            "scale_segment",
            start,
            end,
            channel_indices,
            numeric_cols,
            params={
                "min_factor": args.scale_min_factor,
                "max_factor": args.scale_max_factor,
            },
        )

    # 5. Flatline anomalies
    for _ in range(args.n_flatlines):
        length = random_segment_length()
        start, end = sample_non_overlapping_interval(
            rng,
            occupied,
            region_start,
            region_end,
            length=length,
            padding=args.event_padding,
        )
        channel_indices = choose_channels(
            rng, numeric_cols, args.channel_mode, args.max_channels
        )

        inject_flatline(x_anom, start, end, channel_indices)

        mark_labels(labels_timestamp, labels_channel, start, end, channel_indices)

        add_event(
            events,
            "flatline",
            start,
            end,
            channel_indices,
            numeric_cols,
            params={},
        )

    # 6. Dropout anomalies
    for _ in range(args.n_dropouts):
        length = random_segment_length()
        start, end = sample_non_overlapping_interval(
            rng,
            occupied,
            region_start,
            region_end,
            length=length,
            padding=args.event_padding,
        )
        channel_indices = choose_channels(
            rng, numeric_cols, args.channel_mode, args.max_channels
        )

        inject_dropout(
            x_anom,
            start,
            end,
            channel_indices,
            fill_value=args.dropout_fill_value,
        )

        mark_labels(labels_timestamp, labels_channel, start, end, channel_indices)

        add_event(
            events,
            "dropout",
            start,
            end,
            channel_indices,
            numeric_cols,
            params={"fill_value": args.dropout_fill_value},
        )

    # 7. Cross-dataset segment transplant: e.g. Exchange -> Weather
    for _ in range(args.n_cross_dataset_segments):
        length = random_segment_length()
        start, end = sample_non_overlapping_interval(
            rng,
            occupied,
            region_start,
            region_end,
            length=length,
            padding=args.event_padding,
        )
        channel_indices = choose_channels(
            rng, numeric_cols, args.channel_mode, args.max_channels
        )

        source_start, source_end = inject_cross_dataset_segment(
            x=x_anom,
            x_clean=x_clean,
            source_x=source_x,
            rng=rng,
            start=start,
            end=end,
            channel_indices=channel_indices,
            severity=args.cross_dataset_severity,
            context=args.cross_dataset_context,
        )

        mark_labels(labels_timestamp, labels_channel, start, end, channel_indices)

        add_event(
            events,
            "cross_dataset_segment",
            start,
            end,
            channel_indices,
            numeric_cols,
            params={
                "severity": args.cross_dataset_severity,
                "context": args.cross_dataset_context,
                "source_numeric_columns": source_numeric_cols,
            },
            source_dataset=str(args.source_csv),
            source_start=source_start,
            source_end=source_end,
        )

    # Save anomalous CSV with original structure
    out_df = target_df.copy()
    out_df[numeric_cols] = x_anom

    anomalous_csv_path = out_dir / "weather_synthetic_anomalies.csv"
    out_df.to_csv(anomalous_csv_path, index=False)

    # Save timestamp labels
    labels_timestamp_df = pd.DataFrame({
        "index": np.arange(t),
        "is_anomaly": labels_timestamp,
    })

    if time_col is not None:
        labels_timestamp_df.insert(1, time_col, target_df[time_col].values)

    labels_timestamp_path = out_dir / "labels_timestamp.csv"
    labels_timestamp_df.to_csv(labels_timestamp_path, index=False)

    # Save channel labels
    labels_channel_df = pd.DataFrame(
        labels_channel,
        columns=[f"{col}_is_anomaly" for col in numeric_cols],
    )
    labels_channel_df.insert(0, "index", np.arange(t))

    if time_col is not None:
        labels_channel_df.insert(1, time_col, target_df[time_col].values)

    labels_channel_path = out_dir / "labels_channel.csv"
    labels_channel_df.to_csv(labels_channel_path, index=False)

    # Save NPY labels too
    np.save(out_dir / "labels_timestamp.npy", labels_timestamp)
    np.save(out_dir / "labels_channel.npy", labels_channel)

    # Save event metadata
    events_path = out_dir / "events.json"
    with open(events_path, "w", encoding="utf-8") as f:
        json.dump([asdict(e) for e in events], f, indent=2, ensure_ascii=False)

    # Save summary
    summary = {
        "target_csv": str(args.target_csv),
        "source_csv": str(args.source_csv),
        "output_csv": str(anomalous_csv_path),
        "time_column": time_col,
        "numeric_columns": numeric_cols,
        "n_timestamps": int(t),
        "n_channels": int(c),
        "injection_region": {
            "start_index": int(region_start),
            "end_index": int(region_end),
            "start_ratio": args.injection_start_ratio,
            "end_ratio": args.injection_end_ratio,
        },
        "n_events": len(events),
        "anomalous_timestamps": int(labels_timestamp.sum()),
        "anomalous_timestamp_ratio": float(labels_timestamp.mean()),
        "anomalous_channel_points": int(labels_channel.sum()),
    }

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Done.")
    print(f"Anomalous CSV:       {anomalous_csv_path}")
    print(f"Timestamp labels:    {labels_timestamp_path}")
    print(f"Channel labels:      {labels_channel_path}")
    print(f"Events:              {events_path}")
    print(f"Summary:             {summary_path}")
    print()
    print(f"Injected events: {len(events)}")
    print(f"Anomalous timestamp ratio: {labels_timestamp.mean():.4f}")


if __name__ == "__main__":
    main()