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

    else:
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()