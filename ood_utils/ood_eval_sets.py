import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def load_npz(path: str | Path) -> Dict[str, np.ndarray]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def parse_int_list(values: Optional[List[str]]) -> List[int]:
    if values is None:
        return []

    output = []
    for value in values:
        output.append(int(value))

    return output

def infer_n_samples_from_bank(bank: Dict[str, np.ndarray]) -> int:
    if "instance_sample_index" in bank:
        return int(np.max(bank["instance_sample_index"])) + 1

    if "timestamp_sample_index" in bank:
        return int(np.max(bank["timestamp_sample_index"])) + 1

    raise KeyError(
        "Cannot infer number of samples. Bank must contain "
        "'instance_sample_index' or 'timestamp_sample_index'."
    )


def get_window_start_index(
    bank: Dict[str, np.ndarray],
    n_samples: int,
    split_start_index: Optional[int] = None,
    window_start_index_key: str = "window_start_index",
) -> np.ndarray:
    """
    Get global CSV start index for each forecasting window.

    Best case:
        embedding_bank.py saved window_start_index directly.

    Fallback:
        assume local sample index maps to:
            global_start = split_start_index + local_sample_index

    This fallback is correct if the forecasting dataset creates windows with
    stride 1 over the selected split, which is the usual TimeDRL/TSLib style.
    """
    if window_start_index_key in bank:
        window_start_index = np.asarray(bank[window_start_index_key]).reshape(-1).astype(np.int64)

        if window_start_index.shape[0] != n_samples:
            raise ValueError(
                f"{window_start_index_key} length mismatch: "
                f"{window_start_index.shape[0]} != n_samples={n_samples}"
            )

        return window_start_index

    if split_start_index is None:
        raise ValueError(
            "Bank does not contain window_start_index. "
            "Pass --split_start_index, or modify embedding_bank.py to save "
            "window_start_index for forecasting banks."
        )

    return np.arange(n_samples, dtype=np.int64) + int(split_start_index)


def create_forecasting_csv_labels_from_mask_and_bank(
    mask_data: Dict[str, np.ndarray],
    bank: Dict[str, np.ndarray],
    seq_len: int,
    patch_len: int,
    stride: int,
    split_start_index: Optional[int] = None,
    window_start_index_key: str = "window_start_index",
) -> Dict[str, np.ndarray]:
    """
    Convert full-CSV point anomaly mask to sample-level and timestamp-level labels.

    mask_data["point_anomaly_mask"]:
        [T, C]

    bank timestamp mappings:
        timestamp_sample_index
        timestamp_channel_index
        timestamp_patch_index

    Output:
        sample_label: [N]
        timestamp_label: [N_timestamp_vectors]
    """
    if "point_anomaly_mask" not in mask_data:
        raise KeyError("mask_data must contain 'point_anomaly_mask'.")

    point_mask = np.asarray(mask_data["point_anomaly_mask"]).astype(np.int64)

    if point_mask.ndim != 2:
        raise ValueError(
            "CSV point_anomaly_mask must have shape [T, C]. "
            f"Got shape: {point_mask.shape}"
        )

    total_length, n_channels = point_mask.shape

    required_mapping_keys = [
        "timestamp_sample_index",
        "timestamp_channel_index",
        "timestamp_patch_index",
    ]

    for key in required_mapping_keys:
        if key not in bank:
            raise KeyError(f"Bank does not contain required key: {key}")

    n_samples = infer_n_samples_from_bank(bank)

    window_start_index = get_window_start_index(
        bank=bank,
        n_samples=n_samples,
        split_start_index=split_start_index,
        window_start_index_key=window_start_index_key,
    )

    sample_label = np.zeros(n_samples, dtype=np.int64)
    sample_ood_type = np.zeros(n_samples, dtype=np.int64)

    point_ood_type = mask_data.get("point_ood_type", None)
    if point_ood_type is not None:
        point_ood_type = np.asarray(point_ood_type).astype(np.int64)

        if point_ood_type.shape != point_mask.shape:
            raise ValueError(
                f"point_ood_type shape mismatch: {point_ood_type.shape} != {point_mask.shape}"
            )

    for sample_idx in range(n_samples):
        start = int(window_start_index[sample_idx])
        end = min(start + int(seq_len), total_length)

        if start < 0 or start >= total_length:
            sample_label[sample_idx] = -1
            sample_ood_type[sample_idx] = -1
            continue

        window_mask = point_mask[start:end, :]

        if np.any(window_mask):
            sample_label[sample_idx] = 1

            if point_ood_type is not None:
                window_types = point_ood_type[start:end, :]
                positive_types = window_types[window_types > 0]
                sample_ood_type[sample_idx] = (
                    int(np.max(positive_types)) if positive_types.size else 1
                )
            else:
                sample_ood_type[sample_idx] = 1
        else:
            sample_label[sample_idx] = 0
            sample_ood_type[sample_idx] = 0

    timestamp_sample_index = np.asarray(bank["timestamp_sample_index"]).reshape(-1).astype(np.int64)
    timestamp_channel_index = np.asarray(bank["timestamp_channel_index"]).reshape(-1).astype(np.int64)
    timestamp_patch_index = np.asarray(bank["timestamp_patch_index"]).reshape(-1).astype(np.int64)

    n_timestamp = timestamp_sample_index.shape[0]

    timestamp_label = np.zeros(n_timestamp, dtype=np.int64)
    timestamp_ood_type = np.zeros(n_timestamp, dtype=np.int64)
    timestamp_global_start_index = np.full(n_timestamp, -1, dtype=np.int64)
    timestamp_global_end_index = np.full(n_timestamp, -1, dtype=np.int64)

    for row_idx in range(n_timestamp):
        sample_idx = int(timestamp_sample_index[row_idx])
        channel_idx = int(timestamp_channel_index[row_idx])
        patch_idx = int(timestamp_patch_index[row_idx])

        if sample_idx < 0 or sample_idx >= n_samples:
            timestamp_label[row_idx] = -1
            timestamp_ood_type[row_idx] = -1
            continue

        global_start = int(window_start_index[sample_idx]) + patch_idx * int(stride)
        global_end = min(global_start + int(patch_len), total_length)

        timestamp_global_start_index[row_idx] = global_start
        timestamp_global_end_index[row_idx] = global_end

        if global_start < 0 or global_start >= total_length:
            timestamp_label[row_idx] = -1
            timestamp_ood_type[row_idx] = -1
            continue

        if channel_idx == -1:
            patch_mask = point_mask[global_start:global_end, :]
            patch_is_anomaly = bool(np.any(patch_mask))

            if point_ood_type is not None:
                patch_types = point_ood_type[global_start:global_end, :]
                positive_types = patch_types[patch_types > 0]
                patch_ood_type = int(np.max(positive_types)) if positive_types.size else 1
            else:
                patch_ood_type = 1

        else:
            if channel_idx < 0 or channel_idx >= n_channels:
                timestamp_label[row_idx] = -1
                timestamp_ood_type[row_idx] = -1
                continue

            patch_mask = point_mask[global_start:global_end, channel_idx]
            patch_is_anomaly = bool(np.any(patch_mask))

            if point_ood_type is not None:
                patch_types = point_ood_type[global_start:global_end, channel_idx]
                positive_types = patch_types[patch_types > 0]
                patch_ood_type = int(np.max(positive_types)) if positive_types.size else 1
            else:
                patch_ood_type = 1

        if patch_is_anomaly:
            timestamp_label[row_idx] = 1
            timestamp_ood_type[row_idx] = patch_ood_type
        else:
            timestamp_label[row_idx] = 0
            timestamp_ood_type[row_idx] = 0

    result = {
        "sample_label": sample_label.astype(np.int64),
        "sample_ood_type": sample_ood_type.astype(np.int64),
        "timestamp_label": timestamp_label.astype(np.int64),
        "timestamp_ood_type": timestamp_ood_type.astype(np.int64),
        "window_start_index": window_start_index.astype(np.int64),
        "timestamp_sample_index": timestamp_sample_index.astype(np.int64),
        "timestamp_channel_index": timestamp_channel_index.astype(np.int64),
        "timestamp_patch_index": timestamp_patch_index.astype(np.int64),
        "timestamp_global_start_index": timestamp_global_start_index.astype(np.int64),
        "timestamp_global_end_index": timestamp_global_end_index.astype(np.int64),
    }

    if "value_columns" in mask_data:
        result["value_columns"] = mask_data["value_columns"]

    return result

def create_classification_ood_labels_from_bank(
    bank: Dict[str, np.ndarray],
    id_classes: List[int],
    near_ood_classes: List[int],
    far_ood_classes: Optional[List[int]] = None,
) -> Dict[str, np.ndarray]:
    """
    Build sample-level OOD labels from classification sample_labels.

    sample_ood_type:
        0 = ID
        1 = near-OOD
        2 = far-OOD
        -1 = ignored / not selected

    sample_label:
        0 = ID
        1 = OOD
        -1 = ignored
    """
    if far_ood_classes is None:
        far_ood_classes = []

    if "sample_labels" not in bank:
        raise KeyError("The bank does not contain 'sample_labels'.")

    original_class_label = np.asarray(bank["sample_labels"]).reshape(-1).astype(np.int64)

    sample_ood_type = np.full_like(original_class_label, fill_value=-1, dtype=np.int64)
    sample_label = np.full_like(original_class_label, fill_value=-1, dtype=np.int64)

    id_mask = np.isin(original_class_label, id_classes)
    near_mask = np.isin(original_class_label, near_ood_classes)
    far_mask = np.isin(original_class_label, far_ood_classes)

    sample_ood_type[id_mask] = 0
    sample_label[id_mask] = 0

    sample_ood_type[near_mask] = 1
    sample_label[near_mask] = 1

    sample_ood_type[far_mask] = 2
    sample_label[far_mask] = 1

    result = {
        "sample_label": sample_label.astype(np.int64),
        "sample_ood_type": sample_ood_type.astype(np.int64),
        "original_class_label": original_class_label.astype(np.int64),
        "sample_eval_mask": (sample_label >= 0).astype(np.int64),
    }

    for key in [
        "instance_sample_index",
        "instance_channel_index",
        "timestamp_sample_index",
        "timestamp_channel_index",
        "timestamp_patch_index",
    ]:
        if key in bank:
            result[key] = bank[key]

    return result


def create_sample_labels_from_point_mask(
    point_anomaly_mask: np.ndarray,
) -> np.ndarray:
    """
    point_anomaly_mask shape:
        [N, L, C]
    """
    mask = np.asarray(point_anomaly_mask).astype(np.int64)

    if mask.ndim != 3:
        raise ValueError(
            f"Expected point_anomaly_mask with shape [N, L, C], got {mask.shape}"
        )

    return (mask.sum(axis=(1, 2)) > 0).astype(np.int64)


def create_patch_labels_from_point_mask(
    point_anomaly_mask: np.ndarray,
    timestamp_sample_index: np.ndarray,
    timestamp_channel_index: np.ndarray,
    timestamp_patch_index: np.ndarray,
    patch_len: int,
    stride: int,
) -> np.ndarray:
    """
    Convert point-level anomaly mask to timestamp/patch-level labels.

    point_anomaly_mask:
        [N, L, C]

    timestamp mapping arrays:
        timestamp_sample_index
        timestamp_channel_index
        timestamp_patch_index

    If timestamp_channel_index == -1, the patch label is computed over all channels.
    """
    mask = np.asarray(point_anomaly_mask).astype(np.int64)

    if mask.ndim != 3:
        raise ValueError(
            f"Expected point_anomaly_mask with shape [N, L, C], got {mask.shape}"
        )

    n_samples, seq_len, n_channels = mask.shape

    sample_index = np.asarray(timestamp_sample_index).reshape(-1).astype(np.int64)
    channel_index = np.asarray(timestamp_channel_index).reshape(-1).astype(np.int64)
    patch_index = np.asarray(timestamp_patch_index).reshape(-1).astype(np.int64)

    if not (
        sample_index.shape[0]
        == channel_index.shape[0]
        == patch_index.shape[0]
    ):
        raise ValueError("Timestamp mapping arrays have inconsistent lengths.")

    timestamp_label = np.zeros(sample_index.shape[0], dtype=np.int64)

    for row_idx in range(sample_index.shape[0]):
        s = int(sample_index[row_idx])
        c = int(channel_index[row_idx])
        p = int(patch_index[row_idx])

        if s < 0 or s >= n_samples:
            timestamp_label[row_idx] = -1
            continue

        start = p * stride
        end = min(start + patch_len, seq_len)

        if start >= seq_len:
            timestamp_label[row_idx] = -1
            continue

        if c == -1:
            is_anomalous = bool(mask[s, start:end, :].any())
        else:
            if c < 0 or c >= n_channels:
                timestamp_label[row_idx] = -1
                continue

            is_anomalous = bool(mask[s, start:end, c].any())

        timestamp_label[row_idx] = 1 if is_anomalous else 0

    return timestamp_label.astype(np.int64)


def create_forecasting_labels_from_injection_and_bank(
    injected_data: Dict[str, np.ndarray],
    bank: Dict[str, np.ndarray],
    patch_len: int,
    stride: int,
) -> Dict[str, np.ndarray]:
    if "point_anomaly_mask" not in injected_data:
        raise KeyError("Injected data does not contain 'point_anomaly_mask'.")

    required_mapping_keys = [
        "timestamp_sample_index",
        "timestamp_channel_index",
        "timestamp_patch_index",
    ]

    for key in required_mapping_keys:
        if key not in bank:
            raise KeyError(f"Bank does not contain required key: {key}")

    point_anomaly_mask = injected_data["point_anomaly_mask"]

    sample_label = create_sample_labels_from_point_mask(point_anomaly_mask)

    timestamp_label = create_patch_labels_from_point_mask(
        point_anomaly_mask=point_anomaly_mask,
        timestamp_sample_index=bank["timestamp_sample_index"],
        timestamp_channel_index=bank["timestamp_channel_index"],
        timestamp_patch_index=bank["timestamp_patch_index"],
        patch_len=patch_len,
        stride=stride,
    )

    sample_ood_type = injected_data.get(
        "sample_ood_type",
        sample_label.copy(),
    ).astype(np.int64)

    timestamp_ood_type = timestamp_label.copy()
    timestamp_ood_type[timestamp_label == 1] = 1

    result = {
        "sample_label": sample_label.astype(np.int64),
        "sample_ood_type": sample_ood_type.astype(np.int64),
        "timestamp_label": timestamp_label.astype(np.int64),
        "timestamp_ood_type": timestamp_ood_type.astype(np.int64),
        "point_anomaly_mask": point_anomaly_mask.astype(np.int64),
        "timestamp_sample_index": bank["timestamp_sample_index"].astype(np.int64),
        "timestamp_channel_index": bank["timestamp_channel_index"].astype(np.int64),
        "timestamp_patch_index": bank["timestamp_patch_index"].astype(np.int64),
    }

    for key in [
        "injection_start",
        "injection_end",
        "injection_channel",
        "injection_type",
    ]:
        if key in injected_data:
            result[key] = injected_data[key]

    return result


def save_npz(data: Dict[str, np.ndarray], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build OOD/anomaly evaluation label files."
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    classification = subparsers.add_parser(
        "classification-labels",
        help="Create sample-level classification OOD labels from embedding bank.",
    )
    classification.add_argument("--bank_path", type=str, required=True)
    classification.add_argument("--output_path", type=str, required=True)
    classification.add_argument("--id_classes", nargs="+", required=True)
    classification.add_argument("--near_ood_classes", nargs="+", required=True)
    classification.add_argument("--far_ood_classes", nargs="*", default=[])

    forecasting = subparsers.add_parser(
        "forecasting-labels",
        help="Create sample/timestamp labels from injected point anomaly mask.",
    )
    forecasting.add_argument("--injected_path", type=str, required=True)
    forecasting.add_argument("--bank_path", type=str, required=True)
    forecasting.add_argument("--output_path", type=str, required=True)
    forecasting.add_argument("--patch_len", type=int, required=True)
    forecasting.add_argument("--stride", type=int, required=True)

    forecasting_csv = subparsers.add_parser(
        "forecasting-csv-labels",
        help="Create sample/timestamp labels from full forecasting CSV point mask.",
    )
    forecasting_csv.add_argument("--mask_path", type=str, required=True)
    forecasting_csv.add_argument("--bank_path", type=str, required=True)
    forecasting_csv.add_argument("--output_path", type=str, required=True)

    forecasting_csv.add_argument("--seq_len", type=int, required=True)
    forecasting_csv.add_argument("--patch_len", type=int, required=True)
    forecasting_csv.add_argument("--stride", type=int, required=True)

    forecasting_csv.add_argument(
        "--split_start_index",
        type=int,
        default=None,
        help=(
            "Global CSV row index where this bank split starts. "
            "Only needed if the embedding bank does not contain window_start_index."
        ),
    )
    forecasting_csv.add_argument(
        "--window_start_index_key",
        type=str,
        default="window_start_index",
        help="Key in bank npz that stores global window start indices.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "classification-labels":
        bank = load_npz(args.bank_path)

        labels = create_classification_ood_labels_from_bank(
            bank=bank,
            id_classes=parse_int_list(args.id_classes),
            near_ood_classes=parse_int_list(args.near_ood_classes),
            far_ood_classes=parse_int_list(args.far_ood_classes),
        )

        save_npz(labels, args.output_path)

        print(f"Saved classification OOD labels: {args.output_path}")
        print(f"n samples: {labels['sample_label'].shape[0]}")
        print(f"n ID: {int(np.sum(labels['sample_label'] == 0))}")
        print(f"n OOD: {int(np.sum(labels['sample_label'] == 1))}")
        print(f"n ignored: {int(np.sum(labels['sample_label'] == -1))}")

    elif args.command == "forecasting-labels":
        injected_data = load_npz(args.injected_path)
        bank = load_npz(args.bank_path)

        labels = create_forecasting_labels_from_injection_and_bank(
            injected_data=injected_data,
            bank=bank,
            patch_len=args.patch_len,
            stride=args.stride,
        )

        save_npz(labels, args.output_path)

        print(f"Saved forecasting anomaly labels: {args.output_path}")
        print(f"n samples: {labels['sample_label'].shape[0]}")
        print(f"n anomalous samples: {int(np.sum(labels['sample_label'] == 1))}")
        print(f"n timestamp labels: {labels['timestamp_label'].shape[0]}")
        print(f"n anomalous timestamps: {int(np.sum(labels['timestamp_label'] == 1))}")

    elif args.command == "forecasting-csv-labels":
        mask_data = load_npz(args.mask_path)
        bank = load_npz(args.bank_path)

        labels = create_forecasting_csv_labels_from_mask_and_bank(
            mask_data=mask_data,
            bank=bank,
            seq_len=args.seq_len,
            patch_len=args.patch_len,
            stride=args.stride,
            split_start_index=args.split_start_index,
            window_start_index_key=args.window_start_index_key,
        )

        save_npz(labels, args.output_path)

        print(f"Saved forecasting CSV anomaly labels: {args.output_path}")
        print(f"n samples: {labels['sample_label'].shape[0]}")
        print(f"n anomalous samples: {int(np.sum(labels['sample_label'] == 1))}")
        print(f"n timestamp labels: {labels['timestamp_label'].shape[0]}")
        print(f"n anomalous timestamps: {int(np.sum(labels['timestamp_label'] == 1))}")

    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()