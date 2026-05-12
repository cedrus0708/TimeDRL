import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


ANOMALY_TYPES = [
    "spike",
    "level_shift",
    "scale",
    "noise",
    "trend",
    "flatline",
    "segment_replace",
]


def parse_columns(value_columns: Optional[List[str]], df: pd.DataFrame) -> List[str]:
    """
    Select numeric value columns for injection.

    If value_columns is None, all numeric columns except common date/time columns
    are used.
    """
    if value_columns:
        missing = [col for col in value_columns if col not in df.columns]
        if missing:
            raise KeyError(f"Missing value columns from CSV: {missing}")
        return value_columns

    excluded = {"date", "datetime", "time", "timestamp"}
    numeric_columns = []

    for col in df.columns:
        if col.lower() in excluded:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_columns.append(col)

    if not numeric_columns:
        raise ValueError("No numeric value columns found for injection.")

    return numeric_columns


def choose_segment(
    rng: np.random.Generator,
    start_idx: int,
    end_idx: int,
    min_len: int,
    max_len: int,
) -> Tuple[int, int]:
    available_len = end_idx - start_idx

    if available_len <= 0:
        raise ValueError("Invalid injection range.")

    max_len = min(max_len, available_len)
    min_len = min(min_len, max_len)

    seg_len = int(rng.integers(min_len, max_len + 1))
    start = int(rng.integers(start_idx, end_idx - seg_len + 1))
    end = start + seg_len

    return start, end


def choose_channels(
    rng: np.random.Generator,
    n_channels: int,
    channel_mode: str,
) -> np.ndarray:
    if channel_mode == "all":
        return np.arange(n_channels, dtype=np.int64)

    if channel_mode == "random_one":
        return np.array([int(rng.integers(0, n_channels))], dtype=np.int64)

    if channel_mode == "random_subset":
        n_selected = int(rng.integers(1, n_channels + 1))
        return rng.choice(n_channels, size=n_selected, replace=False).astype(np.int64)

    raise ValueError("channel_mode must be one of: all, random_one, random_subset")


def robust_local_scale(values: np.ndarray, column_indices: np.ndarray) -> np.ndarray:
    selected = values[:, column_indices]
    scale = np.std(selected, axis=0)
    scale = np.maximum(scale, 1e-6)
    return scale.astype(np.float32)


def inject_spike(
    values: np.ndarray,
    start: int,
    end: int,
    column_indices: np.ndarray,
    rng: np.random.Generator,
    magnitude: float,
) -> None:
    scale = robust_local_scale(values, column_indices)
    signs = rng.choice([-1.0, 1.0], size=len(column_indices))
    values[start:end, column_indices] += magnitude * scale * signs


def inject_level_shift(
    values: np.ndarray,
    start: int,
    end: int,
    column_indices: np.ndarray,
    rng: np.random.Generator,
    magnitude: float,
) -> None:
    scale = robust_local_scale(values, column_indices)
    signs = rng.choice([-1.0, 1.0], size=len(column_indices))
    values[start:end, column_indices] += magnitude * scale * signs


def inject_scale(
    values: np.ndarray,
    start: int,
    end: int,
    column_indices: np.ndarray,
    rng: np.random.Generator,
    magnitude: float,
) -> None:
    factor = float(rng.uniform(1.0 + magnitude / 2.0, 1.0 + magnitude))

    if rng.random() < 0.5:
        factor = 1.0 / factor

    values[start:end, column_indices] *= factor


def inject_noise(
    values: np.ndarray,
    start: int,
    end: int,
    column_indices: np.ndarray,
    rng: np.random.Generator,
    magnitude: float,
) -> None:
    scale = robust_local_scale(values, column_indices)
    noise = rng.normal(
        loc=0.0,
        scale=magnitude * scale,
        size=(end - start, len(column_indices)),
    ).astype(np.float32)

    values[start:end, column_indices] += noise


def inject_trend(
    values: np.ndarray,
    start: int,
    end: int,
    column_indices: np.ndarray,
    rng: np.random.Generator,
    magnitude: float,
) -> None:
    scale = robust_local_scale(values, column_indices)
    signs = rng.choice([-1.0, 1.0], size=len(column_indices))
    ramp = np.linspace(0.0, magnitude, end - start).reshape(-1, 1)

    values[start:end, column_indices] += ramp * scale.reshape(1, -1) * signs.reshape(1, -1)


def inject_flatline(
    values: np.ndarray,
    start: int,
    end: int,
    column_indices: np.ndarray,
) -> None:
    values[start:end, column_indices] = values[start:start + 1, column_indices]


def inject_segment_replace(
    values: np.ndarray,
    start: int,
    end: int,
    column_indices: np.ndarray,
    rng: np.random.Generator,
    source_values: Optional[np.ndarray] = None,
) -> None:
    seg_len = end - start

    if source_values is None:
        max_source_start = values.shape[0] - seg_len

        if max_source_start <= 0:
            raise ValueError("Series is too short for segment replacement.")

        source_start = int(rng.integers(0, max_source_start + 1))
        source_segment = values[source_start:source_start + seg_len, column_indices]
    else:
        if source_values.shape[0] < seg_len:
            raise ValueError("source CSV is shorter than selected segment.")

        source_start = int(rng.integers(0, source_values.shape[0] - seg_len + 1))

        if source_values.shape[1] >= values.shape[1]:
            source_segment = source_values[source_start:source_start + seg_len, column_indices]
        else:
            source_cols = rng.choice(
                source_values.shape[1],
                size=len(column_indices),
                replace=True,
            )
            source_segment = source_values[source_start:source_start + seg_len, source_cols]

    target = values[start:end, column_indices]

    target_mean = np.mean(target, axis=0, keepdims=True)
    target_std = np.maximum(np.std(target, axis=0, keepdims=True), 1e-6)

    source_mean = np.mean(source_segment, axis=0, keepdims=True)
    source_std = np.maximum(np.std(source_segment, axis=0, keepdims=True), 1e-6)

    adapted = (source_segment - source_mean) / source_std
    adapted = adapted * target_std + target_mean

    values[start:end, column_indices] = adapted


def apply_anomaly(
    values: np.ndarray,
    anomaly_type: str,
    start: int,
    end: int,
    column_indices: np.ndarray,
    rng: np.random.Generator,
    magnitude: float,
    source_values: Optional[np.ndarray] = None,
) -> None:
    if anomaly_type == "spike":
        inject_spike(values, start, end, column_indices, rng, magnitude)
    elif anomaly_type == "level_shift":
        inject_level_shift(values, start, end, column_indices, rng, magnitude)
    elif anomaly_type == "scale":
        inject_scale(values, start, end, column_indices, rng, magnitude)
    elif anomaly_type == "noise":
        inject_noise(values, start, end, column_indices, rng, magnitude)
    elif anomaly_type == "trend":
        inject_trend(values, start, end, column_indices, rng, magnitude)
    elif anomaly_type == "flatline":
        inject_flatline(values, start, end, column_indices)
    elif anomaly_type == "segment_replace":
        inject_segment_replace(
            values=values,
            start=start,
            end=end,
            column_indices=column_indices,
            rng=rng,
            source_values=source_values,
        )
    else:
        raise ValueError(f"Unknown anomaly_type: {anomaly_type}")


def inject_csv_anomalies(
    input_csv_path: str | Path,
    output_csv_path: str | Path,
    output_mask_path: str | Path,
    value_columns: Optional[List[str]] = None,
    date_column: Optional[str] = None,
    anomaly_fraction: float = 0.05,
    anomaly_types: Optional[List[str]] = None,
    min_len: int = 4,
    max_len: int = 32,
    magnitude: float = 3.0,
    channel_mode: str = "random_one",
    inject_start_ratio: float = 0.7,
    inject_end_ratio: float = 1.0,
    seed: int = 42,
    source_csv_path: Optional[str | Path] = None,
) -> Dict[str, object]:
    """
    Inject anomalies directly into an original TimeDRL forecasting CSV.

    The output CSV preserves the original columns. A separate mask .npz is saved.

    point_anomaly_mask:
        shape [T, C]
        T = full CSV row count
        C = number of selected numeric value columns
    """
    input_csv_path = Path(input_csv_path)
    output_csv_path = Path(output_csv_path)
    output_mask_path = Path(output_mask_path)

    df = pd.read_csv(input_csv_path)
    selected_columns = parse_columns(value_columns, df)

    values = df[selected_columns].to_numpy(dtype=np.float32)
    injected_values = values.copy()

    source_values = None

    if source_csv_path is not None:
        source_df = pd.read_csv(source_csv_path)
        source_columns = parse_columns(None, source_df)
        source_values = source_df[source_columns].to_numpy(dtype=np.float32)

    if anomaly_types is None:
        anomaly_types = ["spike", "level_shift", "noise", "trend", "flatline"]

    for anomaly_type in anomaly_types:
        if anomaly_type not in ANOMALY_TYPES:
            raise ValueError(f"Unknown anomaly_type: {anomaly_type}")

    rng = np.random.default_rng(seed)

    n_rows, n_channels = injected_values.shape

    inject_start_idx = int(round(n_rows * inject_start_ratio))
    inject_end_idx = int(round(n_rows * inject_end_ratio))
    inject_start_idx = max(0, min(inject_start_idx, n_rows - 1))
    inject_end_idx = max(inject_start_idx + 1, min(inject_end_idx, n_rows))

    inject_region_len = inject_end_idx - inject_start_idx
    target_total_points = int(round(inject_region_len * anomaly_fraction))

    point_anomaly_mask = np.zeros((n_rows, n_channels), dtype=np.int64)

    events = []
    covered_points = 0
    event_id = 0

    while covered_points < target_total_points:
        anomaly_type = str(rng.choice(anomaly_types))

        start, end = choose_segment(
            rng=rng,
            start_idx=inject_start_idx,
            end_idx=inject_end_idx,
            min_len=min_len,
            max_len=max_len,
        )

        column_indices = choose_channels(
            rng=rng,
            n_channels=n_channels,
            channel_mode=channel_mode,
        )

        apply_anomaly(
            values=injected_values,
            anomaly_type=anomaly_type,
            start=start,
            end=end,
            column_indices=column_indices,
            rng=rng,
            magnitude=magnitude,
            source_values=source_values,
        )

        point_anomaly_mask[start:end, column_indices] = 1

        events.append(
            {
                "event_id": event_id,
                "anomaly_type": anomaly_type,
                "start": int(start),
                "end": int(end),
                "length": int(end - start),
                "columns": [selected_columns[int(i)] for i in column_indices],
                "column_indices": [int(i) for i in column_indices],
                "ood_type": 2 if anomaly_type == "segment_replace" else 1,
            }
        )

        covered_points += int((end - start) * len(column_indices))
        event_id += 1

    df_out = df.copy()
    df_out[selected_columns] = injected_values

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    output_mask_path.parent.mkdir(parents=True, exist_ok=True)

    df_out.to_csv(output_csv_path, index=False)

    sample_ood_type_by_point = np.zeros((n_rows, n_channels), dtype=np.int64)

    for event in events:
        start = event["start"]
        end = event["end"]
        ood_type = event["ood_type"]

        for col_idx in event["column_indices"]:
            sample_ood_type_by_point[start:end, col_idx] = ood_type

    np.savez_compressed(
        output_mask_path,
        point_anomaly_mask=point_anomaly_mask.astype(np.int64),
        point_ood_type=sample_ood_type_by_point.astype(np.int64),
        value_columns=np.asarray(selected_columns),
        input_csv_path=np.asarray(str(input_csv_path)),
        output_csv_path=np.asarray(str(output_csv_path)),
        inject_start_idx=np.asarray(inject_start_idx, dtype=np.int64),
        inject_end_idx=np.asarray(inject_end_idx, dtype=np.int64),
    )

    metadata = {
        "input_csv_path": str(input_csv_path),
        "output_csv_path": str(output_csv_path),
        "output_mask_path": str(output_mask_path),
        "date_column": date_column,
        "value_columns": selected_columns,
        "anomaly_fraction": anomaly_fraction,
        "anomaly_types": anomaly_types,
        "min_len": min_len,
        "max_len": max_len,
        "magnitude": magnitude,
        "channel_mode": channel_mode,
        "inject_start_ratio": inject_start_ratio,
        "inject_end_ratio": inject_end_ratio,
        "inject_start_idx": inject_start_idx,
        "inject_end_idx": inject_end_idx,
        "seed": seed,
        "source_csv_path": str(source_csv_path) if source_csv_path is not None else None,
        "events": events,
    }

    metadata_path = output_mask_path.with_suffix(".meta.json")

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inject synthetic anomalies into original forecasting CSV files."
    )

    parser.add_argument("--input_csv_path", type=str, required=True)
    parser.add_argument("--output_csv_path", type=str, required=True)
    parser.add_argument("--output_mask_path", type=str, required=True)

    parser.add_argument(
        "--value_columns",
        nargs="*",
        default=None,
        help="Numeric columns to manipulate. If omitted, all numeric non-date columns are used.",
    )
    parser.add_argument("--date_column", type=str, default=None)

    parser.add_argument("--anomaly_fraction", type=float, default=0.05)
    parser.add_argument(
        "--anomaly_types",
        nargs="+",
        default=["spike", "level_shift", "noise", "trend", "flatline"],
        choices=ANOMALY_TYPES,
    )
    parser.add_argument("--min_len", type=int, default=4)
    parser.add_argument("--max_len", type=int, default=32)
    parser.add_argument("--magnitude", type=float, default=3.0)
    parser.add_argument(
        "--channel_mode",
        type=str,
        default="random_one",
        choices=["all", "random_one", "random_subset"],
    )

    parser.add_argument(
        "--inject_start_ratio",
        type=float,
        default=0.7,
        help="Start ratio in the full CSV where anomalies may be injected. Default: 0.7.",
    )
    parser.add_argument(
        "--inject_end_ratio",
        type=float,
        default=1.0,
        help="End ratio in the full CSV where anomalies may be injected. Default: 1.0.",
    )
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--source_csv_path",
        type=str,
        default=None,
        help="Optional source CSV for far-OOD segment_replace.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    metadata = inject_csv_anomalies(
        input_csv_path=args.input_csv_path,
        output_csv_path=args.output_csv_path,
        output_mask_path=args.output_mask_path,
        value_columns=args.value_columns,
        date_column=args.date_column,
        anomaly_fraction=args.anomaly_fraction,
        anomaly_types=args.anomaly_types,
        min_len=args.min_len,
        max_len=args.max_len,
        magnitude=args.magnitude,
        channel_mode=args.channel_mode,
        inject_start_ratio=args.inject_start_ratio,
        inject_end_ratio=args.inject_end_ratio,
        seed=args.seed,
        source_csv_path=args.source_csv_path,
    )

    print("Saved injected CSV:")
    print(f"  {metadata['output_csv_path']}")
    print("Saved mask:")
    print(f"  {metadata['output_mask_path']}")
    print("Saved metadata:")
    print(f"  {Path(metadata['output_mask_path']).with_suffix('.meta.json')}")
    print(f"Injected events: {len(metadata['events'])}")


if __name__ == "__main__":
    main()