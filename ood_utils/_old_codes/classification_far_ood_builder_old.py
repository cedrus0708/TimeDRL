import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from main import get_args_from_parser  # noqa: E402
from dataset_loader.dataset_loader import (  # noqa: E402
    load_classification_dataloader,
    update_args_from_dataset,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build classification eval set with true external far-OOD samples.",
        allow_abbrev=False,
    )

    parser.add_argument("--target_model_for", type=str, required=True)
    parser.add_argument("--source_data_name", type=str, required=True)

    parser.add_argument("--model_registry_path", type=str, default="./weights/args.json")
    parser.add_argument("--output_path", type=str, required=True)

    parser.add_argument("--id_classes", nargs="+", type=int, required=True)
    parser.add_argument("--near_ood_classes", nargs="+", type=int, default=[])
    parser.add_argument("--far_ood_count", type=int, default=-1)

    parser.add_argument(
        "--target_split",
        type=str,
        default="test",
        choices=["train", "valid", "test"],
    )
    parser.add_argument(
        "--source_split",
        type=str,
        default="test",
        choices=["train", "valid", "test"],
    )

    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--length_strategy",
        type=str,
        default="interpolate",
        choices=["interpolate", "crop_or_pad"],
    )
    parser.add_argument(
        "--channel_strategy",
        type=str,
        default="repeat",
        choices=["repeat", "first", "random"],
    )
    parser.add_argument(
        "--normalize_source_to_target",
        action="store_true",
        default=True,
        help="Normalize external source windows to target ID distribution.",
    )

    return parser.parse_args()


def load_model_registry(path: str | Path) -> List[Dict[str, Any]]:
    path = Path(path)

    if not path.is_absolute():
        path = REPO_ROOT / path

    with open(path, "r", encoding="utf-8") as f:
        registry = json.load(f)

    if not isinstance(registry, list):
        raise ValueError("Model registry must be a JSON list.")

    return registry


def find_model_entry(registry: List[Dict[str, Any]], model_for: str) -> Dict[str, Any]:
    matches = [
        entry
        for entry in registry
        if str(entry.get("model_for", "")).lower() == str(model_for).lower()
    ]

    if not matches:
        available = [entry.get("model_for") for entry in registry]
        raise ValueError(
            f"model_for='{model_for}' not found in registry. "
            f"Available values: {available}"
        )

    if len(matches) > 1:
        raise ValueError(f"Multiple entries found for model_for='{model_for}'.")

    return deepcopy(matches[0])


def apply_entry_to_args(args: argparse.Namespace, entry: Dict[str, Any]) -> argparse.Namespace:
    run_config = entry.get("run_config", {})
    model_config = entry.get("model_config", {})

    for config in [run_config, model_config]:
        for key, value in config.items():
            if key == "pred_len_list" and not isinstance(value, list):
                value = [int(value)]
            setattr(args, key, value)

    if hasattr(args, "pred_len_list") and args.pred_len_list:
        args.pred_len = int(args.pred_len_list[0])

    return args


def build_base_args_from_registry(
    model_for: str,
    registry_path: str | Path,
    batch_size: int,
) -> argparse.Namespace:
    original_argv = sys.argv[:]
    sys.argv = [sys.argv[0]]

    try:
        args = get_args_from_parser()
    finally:
        sys.argv = original_argv

    registry = load_model_registry(registry_path)
    entry = find_model_entry(registry, model_for)

    args = apply_entry_to_args(args, entry)
    args.root_folder = REPO_ROOT
    args.batch_size = batch_size
    args.task_name = "classification"

    args = update_args_from_dataset(args)

    return args


def build_source_args_from_target_args(
    target_args: argparse.Namespace,
    source_data_name: str,
    batch_size: int,
) -> argparse.Namespace:
    source_args = deepcopy(target_args)
    source_args.data_name = source_data_name
    source_args.task_name = "classification"
    source_args.batch_size = batch_size

    source_args = update_args_from_dataset(source_args)

    return source_args


def select_loader(loaders: Tuple[Any, Any, Any], split: str):
    train_loader, valid_loader, test_loader = loaders

    if split == "train":
        return train_loader
    if split == "valid":
        return valid_loader
    if split == "test":
        return test_loader

    raise ValueError(f"Unknown split: {split}")


def load_classification_split_arrays(
    args: argparse.Namespace,
    split: str,
    mode: str = "pretrain",
) -> Tuple[np.ndarray, np.ndarray]:
    train_loader, valid_loader, test_loader, _ = load_classification_dataloader(
        args,
        mode=mode,
    )

    loader = select_loader(
        (train_loader, valid_loader, test_loader),
        split,
    )

    dataset = loader.dataset

    x = extract_dataset_array(dataset, ["x", "data_x", "X"])
    y = extract_dataset_array(dataset, ["y", "data_y", "Y", "labels"])

    x = ensure_nlc(x)
    y = np.asarray(y).reshape(-1).astype(np.int64)

    if x.shape[0] != y.shape[0]:
        raise ValueError(f"x/y length mismatch: {x.shape[0]} != {y.shape[0]}")

    return x.astype(np.float32), y


def extract_dataset_array(dataset: Any, candidate_names: List[str]) -> np.ndarray:
    for name in candidate_names:
        if hasattr(dataset, name):
            value = getattr(dataset, name)

            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()

            return np.asarray(value)

    raise AttributeError(
        f"Could not find any of these attributes on dataset: {candidate_names}"
    )


def ensure_nlc(x: np.ndarray) -> np.ndarray:
    """
    Ensure shape [N, L, C].
    """
    x = np.asarray(x, dtype=np.float32)

    if x.ndim == 2:
        return x[:, :, None]

    if x.ndim != 3:
        raise ValueError(f"Expected x with 2 or 3 dims, got shape {x.shape}")

    return x


def resize_length_interpolate(x: np.ndarray, target_len: int) -> np.ndarray:
    n, old_len, c = x.shape

    if old_len == target_len:
        return x.astype(np.float32)

    old_grid = np.linspace(0.0, 1.0, old_len)
    new_grid = np.linspace(0.0, 1.0, target_len)

    out = np.empty((n, target_len, c), dtype=np.float32)

    for i in range(n):
        for ch in range(c):
            out[i, :, ch] = np.interp(new_grid, old_grid, x[i, :, ch])

    return out


def resize_length_crop_or_pad(x: np.ndarray, target_len: int) -> np.ndarray:
    n, old_len, c = x.shape

    if old_len == target_len:
        return x.astype(np.float32)

    if old_len > target_len:
        start = (old_len - target_len) // 2
        return x[:, start:start + target_len, :].astype(np.float32)

    out = np.zeros((n, target_len, c), dtype=np.float32)
    out[:, :old_len, :] = x
    return out


def adapt_length(
    x: np.ndarray,
    target_len: int,
    strategy: str,
) -> np.ndarray:
    if strategy == "interpolate":
        return resize_length_interpolate(x, target_len)

    if strategy == "crop_or_pad":
        return resize_length_crop_or_pad(x, target_len)

    raise ValueError(f"Unknown length strategy: {strategy}")


def adapt_channels(
    x: np.ndarray,
    target_channels: int,
    strategy: str,
    rng: np.random.Generator,
) -> np.ndarray:
    n, length, source_channels = x.shape

    if source_channels == target_channels:
        return x.astype(np.float32)

    if source_channels > target_channels:
        if strategy == "first" or strategy == "repeat":
            return x[:, :, :target_channels].astype(np.float32)

        if strategy == "random":
            selected = rng.choice(
                source_channels,
                size=target_channels,
                replace=False,
            )
            selected = np.sort(selected)
            return x[:, :, selected].astype(np.float32)

    if source_channels < target_channels:
        if strategy in {"repeat", "first", "random"}:
            repeats = int(np.ceil(target_channels / source_channels))
            tiled = np.tile(x, (1, 1, repeats))
            return tiled[:, :, :target_channels].astype(np.float32)

    raise ValueError(f"Unsupported channel adaptation: {source_channels} -> {target_channels}")


def normalize_source_to_target_distribution(
    source_x: np.ndarray,
    target_x: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Match external source windows to target distribution channel-wise.

    This avoids trivial scale differences dominating the detector.
    """
    source_mean = source_x.mean(axis=(0, 1), keepdims=True)
    source_std = source_x.std(axis=(0, 1), keepdims=True)
    source_std = np.maximum(source_std, eps)

    target_mean = target_x.mean(axis=(0, 1), keepdims=True)
    target_std = target_x.std(axis=(0, 1), keepdims=True)
    target_std = np.maximum(target_std, eps)

    return ((source_x - source_mean) / source_std * target_std + target_mean).astype(np.float32)


def sample_far_ood(
    x: np.ndarray,
    y: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    if count is None or count < 0 or count >= x.shape[0]:
        return x, y

    selected = rng.choice(x.shape[0], size=count, replace=False)
    return x[selected], y[selected]


def build_external_classification_eval_set(
    target_x: np.ndarray,
    target_y: np.ndarray,
    source_x: np.ndarray,
    source_y: np.ndarray,
    id_classes: List[int],
    near_ood_classes: List[int],
    source_data_name: str,
) -> Dict[str, np.ndarray]:
    id_mask = np.isin(target_y, id_classes)
    near_mask = np.isin(target_y, near_ood_classes)

    target_eval_mask = id_mask | near_mask

    target_x_eval = target_x[target_eval_mask]
    target_y_eval = target_y[target_eval_mask]

    target_sample_label = np.full(target_y_eval.shape[0], -1, dtype=np.int64)
    target_sample_ood_type = np.full(target_y_eval.shape[0], -1, dtype=np.int64)

    target_id_mask = np.isin(target_y_eval, id_classes)
    target_near_mask = np.isin(target_y_eval, near_ood_classes)

    target_sample_label[target_id_mask] = 0
    target_sample_ood_type[target_id_mask] = 0

    target_sample_label[target_near_mask] = 1
    target_sample_ood_type[target_near_mask] = 1

    far_sample_label = np.ones(source_x.shape[0], dtype=np.int64)
    far_sample_ood_type = np.full(source_x.shape[0], 2, dtype=np.int64)

    x = np.concatenate([target_x_eval, source_x], axis=0)

    sample_label = np.concatenate(
        [target_sample_label, far_sample_label],
        axis=0,
    )
    sample_ood_type = np.concatenate(
        [target_sample_ood_type, far_sample_ood_type],
        axis=0,
    )

    original_class_label = np.concatenate(
        [target_y_eval, source_y],
        axis=0,
    ).astype(np.int64)

    source_dataset_id = np.concatenate(
        [
            np.zeros(target_x_eval.shape[0], dtype=np.int64),
            np.ones(source_x.shape[0], dtype=np.int64),
        ],
        axis=0,
    )

    y_for_embedding_bank = sample_label.copy()

    source_dataset_name = np.array(
        ["target"] * target_x_eval.shape[0] + [source_data_name] * source_x.shape[0]
    )

    return {
        "x": x.astype(np.float32),
        "y": y_for_embedding_bank.astype(np.int64),

        "sample_label": sample_label.astype(np.int64),
        "sample_ood_type": sample_ood_type.astype(np.int64),

        "original_class_label": original_class_label.astype(np.int64),
        "source_dataset_id": source_dataset_id.astype(np.int64),
        "source_dataset_name": source_dataset_name,

        "id_classes": np.asarray(id_classes, dtype=np.int64),
        "near_ood_classes": np.asarray(near_ood_classes, dtype=np.int64),
    }


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    target_args = build_base_args_from_registry(
        model_for=args.target_model_for,
        registry_path=args.model_registry_path,
        batch_size=args.batch_size,
    )

    if target_args.task_name != "classification":
        raise ValueError("target_model_for must refer to a classification dataset.")

    source_args = build_source_args_from_target_args(
        target_args=target_args,
        source_data_name=args.source_data_name,
        batch_size=args.batch_size,
    )

    print("Loading target classification split...")
    target_x, target_y = load_classification_split_arrays(
        args=target_args,
        split=args.target_split,
    )

    print("Loading source far-OOD classification split...")
    source_x, source_y = load_classification_split_arrays(
        args=source_args,
        split=args.source_split,
    )

    target_len = target_x.shape[1]
    target_channels = target_x.shape[2]

    print(f"Target shape: {target_x.shape}")
    print(f"Source original shape: {source_x.shape}")

    source_x = adapt_length(
        source_x,
        target_len=target_len,
        strategy=args.length_strategy,
    )

    source_x = adapt_channels(
        source_x,
        target_channels=target_channels,
        strategy=args.channel_strategy,
        rng=rng,
    )

    source_x, source_y = sample_far_ood(
        source_x,
        source_y,
        count=args.far_ood_count,
        rng=rng,
    )

    id_target_x = target_x[np.isin(target_y, args.id_classes)]

    if args.normalize_source_to_target:
        source_x = normalize_source_to_target_distribution(
            source_x=source_x,
            target_x=id_target_x,
        )

    print(f"Source adapted shape: {source_x.shape}")

    result = build_external_classification_eval_set(
        target_x=target_x,
        target_y=target_y,
        source_x=source_x,
        source_y=source_y,
        id_classes=args.id_classes,
        near_ood_classes=args.near_ood_classes,
        source_data_name=args.source_data_name,
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(output_path, **result)

    print(f"\nSaved external classification OOD eval set: {output_path}")
    print(f"x shape: {result['x'].shape}")
    print(f"n ID: {int(np.sum(result['sample_label'] == 0))}")
    print(f"n near-OOD: {int(np.sum(result['sample_ood_type'] == 1))}")
    print(f"n far-OOD: {int(np.sum(result['sample_ood_type'] == 2))}")


if __name__ == "__main__":
    main()