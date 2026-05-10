import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


# Make repo root importable when running:
# python utils/embedding_bank.py ...
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from main import get_args_from_parser  # noqa: E402
from dataset_loader.dataset_loader import (  # noqa: E402
    load_classification_dataloader,
    load_forecasting_dataloader,
    update_args_from_dataset,
)
from exp.exp_classification import Exp_Classification  # noqa: E402
from exp.exp_forecasting import Exp_Forecasting  # noqa: E402

from utils.model_registry import ( # noqa: E402
    apply_model_registry_if_requested,
    get_explicit_cli_arg_names,
    print_resolved_registry_args,
)

# python ood_utils/embedding_bank.py --model_for Exchange --batch_size 8 --bank_split train --mode pretrain --embedding_view first


def parse_embedding_bank_args() -> Tuple[argparse.Namespace, List[str]]:
    """
    Parse only embedding-bank-specific arguments.

    TimeDRL/common arguments such as:
        --model_for
        --task_name
        --data_name
        --batch_size
        --use_gpu
        --use_amp

    are intentionally NOT parsed here. They are passed through to
    TimeDRL's get_args_from_parser().
    """
    parser = argparse.ArgumentParser(
        description="Build TimeDRL embedding banks for anomaly detection.",
        add_help=False,
        allow_abbrev=False,
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help=(
            "Optional explicit TimeDRL base model checkpoint path. "
            "If omitted and --model_for is given, the base model checkpoint "
            "is loaded from weights/args.json."
        ),
    )

    parser.add_argument(
        "--linear_checkpoint_path",
        type=str,
        default=None,
        help=(
            "Optional explicit linear model checkpoint path. "
            "Only needed if save_linear_outputs=True."
        ),
    )

    parser.add_argument(
        "--save_linear_outputs",
        action="store_true",
        default=False,
        help="Save logits/probabilities/predictions from the linear head.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./embedding_banks",
        help="Directory where the embedding bank files will be saved.",
    )

    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="Optional output file stem.",
    )

    parser.add_argument(
        "--bank_split",
        type=str,
        default="train",
        choices=["train", "valid", "test"],
        help="Which split should be used for the embedding bank.",
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="pretrain",
        choices=["pretrain", "linear_eval"],
        help="Which TimeDRL dataloader mode to use.",
    )

    parser.add_argument(
        "--embedding_view",
        type=str,
        default="first",
        choices=["first", "second", "mean"],
        help="Which TimeDRL view to save: first, second, or mean.",
    )

    parser.add_argument(
        "--force_no_data_aug",
        action="store_true",
        default=True,
        help="Force model.args.data_aug='none' during extraction.",
    )

    parser.add_argument(
        "--l2_normalize",
        action="store_true",
        default=False,
        help="Apply L2 normalization to saved embeddings.",
    )

    parser.add_argument(
        "--max_batches",
        type=int,
        default=None,
        help="Optional debug limit.",
    )

    parser.add_argument(
        "--allow_partial_checkpoint",
        action="store_true",
        default=False,
        help="Load checkpoint with strict=False. Avoid this unless debugging.",
    )

    bank_args, timedrl_argv = parser.parse_known_args()
    return bank_args, timedrl_argv


def build_experiment(args: argparse.Namespace):
    """
    Build the correct TimeDRL experiment wrapper.

    This gives us:
    - args.d_model
    - args.T_p
    - args.i_dim
    - model on the right device
    """
    if args.task_name == "forecasting":
        return Exp_Forecasting(args)
    if args.task_name == "classification":
        return Exp_Classification(args)
    raise NotImplementedError(f"Unknown task_name: {args.task_name}")


def load_train_valid_test_loaders(
    args: argparse.Namespace,
    mode: str,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load TimeDRL dataloaders for either forecasting or classification.
    """
    if args.task_name == "forecasting":
        train_loader, valid_loader, test_loader = load_forecasting_dataloader(
            args,
            mode=mode,
        )
        return train_loader, valid_loader, test_loader

    if args.task_name == "classification":
        train_loader, valid_loader, test_loader, _ = load_classification_dataloader(
            args,
            mode=mode,
        )
        return train_loader, valid_loader, test_loader

    raise NotImplementedError(f"Unknown task_name: {args.task_name}")


def make_ordered_loader(loader: DataLoader, batch_size: int) -> DataLoader:
    """
    Re-create a loader without shuffle.

    This is important because the reference bank should have deterministic
    sample_index mappings.
    """
    return DataLoader(
        dataset=loader.dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )


def load_model_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
    strict: bool = True,
) -> None:
    """
    Load model weights from a checkpoint.

    Supports common checkpoint formats:
    - raw state_dict
    - {"state_dict": ...}
    - {"model_state_dict": ...}
    - {"model": ...}
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
    else:
        raise TypeError(
            "Unsupported checkpoint format. Expected a state_dict-like dictionary."
        )

    # Handle checkpoints saved from DataParallel.
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        clean_key = key.replace("module.", "", 1)
        cleaned_state_dict[clean_key] = value

    load_result = model.load_state_dict(cleaned_state_dict, strict=strict)

    if not strict:
        print("Checkpoint loaded with strict=False.")
        print(f"Missing keys: {load_result.missing_keys}")
        print(f"Unexpected keys: {load_result.unexpected_keys}")


def select_embedding_view(
    t_1: torch.Tensor,
    t_2: torch.Tensor,
    i_1: torch.Tensor,
    i_2: torch.Tensor,
    embedding_view: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Select which TimeDRL view should be stored in the bank.
    """
    if embedding_view == "first":
        return t_1, i_1

    if embedding_view == "second":
        return t_2, i_2

    if embedding_view == "mean":
        return (t_1 + t_2) / 2.0, (i_1 + i_2) / 2.0

    raise ValueError(f"Unknown embedding_view: {embedding_view}")


def l2_normalize_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    L2-normalize vectors row-wise.
    """
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, eps)


def unpack_batch(
    batch: Any,
    task_name: str,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Extract batch_x and optional labels from a TimeDRL batch.

    Classification batch:
        batch_x, batch_y

    Forecasting batch:
        batch_x, batch_y, batch_x_mark, batch_y_mark
    """
    if task_name == "classification":
        batch_x, batch_y = batch
        return batch_x, batch_y

    if task_name == "forecasting":
        batch_x, _, _, _ = batch
        return batch_x, None

    raise NotImplementedError(f"Unknown task_name: {task_name}")


def flatten_embeddings_with_mapping(
    t: torch.Tensor,
    i: torch.Tensor,
    batch_x: torch.Tensor,
    sample_indices: np.ndarray,
    enable_channel_independence: bool,
) -> Dict[str, np.ndarray]:
    """
    Flatten TimeDRL embeddings into reference-bank form and create mapping arrays.

    If enable_channel_independence=True:
        i: [B*C, D_i]      -> instance_embeddings: [B*C, D_i]
        t: [B*C, T_p, D]   -> timestamp_embeddings: [B*C*T_p, D]

    If enable_channel_independence=False:
        i: [B, D_i]        -> instance_embeddings: [B, D_i]
        t: [B, T_p, D]     -> timestamp_embeddings: [B*T_p, D]

    The mapping arrays tell us which embedding belongs to which original
    sample/channel/patch.
    """
    batch_size = int(batch_x.shape[0])
    num_channels = int(batch_x.shape[-1])

    t_np = t.detach().float().cpu().numpy()
    i_np = i.detach().float().cpu().numpy()

    if enable_channel_independence:
        if i_np.shape[0] != batch_size * num_channels:
            raise ValueError(
                "Unexpected instance embedding shape for channel-independent mode. "
                f"Expected first dim B*C={batch_size * num_channels}, got {i_np.shape[0]}."
            )

        if t_np.shape[0] != batch_size * num_channels:
            raise ValueError(
                "Unexpected timestamp embedding shape for channel-independent mode. "
                f"Expected first dim B*C={batch_size * num_channels}, got {t_np.shape[0]}."
            )

        num_patches = int(t_np.shape[1])
        timestamp_dim = int(t_np.shape[2])
        instance_dim = int(i_np.shape[-1])

        instance_embeddings = i_np.reshape(batch_size * num_channels, instance_dim)
        timestamp_embeddings = t_np.reshape(
            batch_size * num_channels * num_patches,
            timestamp_dim,
        )

        instance_sample_index = np.repeat(sample_indices, num_channels)
        instance_channel_index = np.tile(np.arange(num_channels), batch_size)

        timestamp_sample_index = np.repeat(sample_indices, num_channels * num_patches)
        timestamp_channel_index = np.tile(
            np.repeat(np.arange(num_channels), num_patches),
            batch_size,
        )
        timestamp_patch_index = np.tile(
            np.arange(num_patches),
            batch_size * num_channels,
        )

    else:
        if i_np.shape[0] != batch_size:
            raise ValueError(
                "Unexpected instance embedding shape for non-channel-independent mode. "
                f"Expected first dim B={batch_size}, got {i_np.shape[0]}."
            )

        if t_np.shape[0] != batch_size:
            raise ValueError(
                "Unexpected timestamp embedding shape for non-channel-independent mode. "
                f"Expected first dim B={batch_size}, got {t_np.shape[0]}."
            )

        num_patches = int(t_np.shape[1])
        timestamp_dim = int(t_np.shape[2])
        instance_dim = int(i_np.shape[-1])

        instance_embeddings = i_np.reshape(batch_size, instance_dim)
        timestamp_embeddings = t_np.reshape(batch_size * num_patches, timestamp_dim)

        instance_sample_index = sample_indices
        instance_channel_index = np.full(batch_size, -1, dtype=np.int64)

        timestamp_sample_index = np.repeat(sample_indices, num_patches)
        timestamp_channel_index = np.full(
            batch_size * num_patches,
            -1,
            dtype=np.int64,
        )
        timestamp_patch_index = np.tile(np.arange(num_patches), batch_size)

    return {
        "instance_embeddings": instance_embeddings.astype(np.float32),
        "timestamp_embeddings": timestamp_embeddings.astype(np.float32),
        "instance_sample_index": instance_sample_index.astype(np.int64),
        "instance_channel_index": instance_channel_index.astype(np.int64),
        "timestamp_sample_index": timestamp_sample_index.astype(np.int64),
        "timestamp_channel_index": timestamp_channel_index.astype(np.int64),
        "timestamp_patch_index": timestamp_patch_index.astype(np.int64),
    }


def build_embedding_bank(
    model: torch.nn.Module,
    loader: DataLoader,
    task_name: str,
    device: torch.device,
    enable_channel_independence: bool,
    embedding_view: str = "first",
    l2_normalize: bool = False,
    max_batches: Optional[int] = None,
    use_amp: bool = False,
    linear_eval: Optional[torch.nn.Module] = None,
    save_linear_outputs: bool = False,
) -> Dict[str, np.ndarray]:
    """
    Run the trained TimeDRL model over a split and collect reference embeddings.
    """
    model.eval()

    instance_chunks: List[np.ndarray] = []
    timestamp_chunks: List[np.ndarray] = []

    instance_sample_index_chunks: List[np.ndarray] = []
    instance_channel_index_chunks: List[np.ndarray] = []

    timestamp_sample_index_chunks: List[np.ndarray] = []
    timestamp_channel_index_chunks: List[np.ndarray] = []
    timestamp_patch_index_chunks: List[np.ndarray] = []

    label_chunks: List[np.ndarray] = []

    linear_logits_chunks: List[np.ndarray] = []
    linear_probs_chunks: List[np.ndarray] = []
    linear_preds_chunks: List[np.ndarray] = []

    sample_offset = 0

    with torch.inference_mode():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Building embedding bank")):
            if max_batches is not None and batch_idx >= max_batches:
                break

            batch_x, batch_y = unpack_batch(batch, task_name)
            batch_size = int(batch_x.shape[0])

            sample_indices = np.arange(
                sample_offset,
                sample_offset + batch_size,
                dtype=np.int64,
            )
            sample_offset += batch_size

            batch_x = batch_x.float().to(device)

            with torch.cuda.amp.autocast(
                enabled=bool(use_amp and device.type == "cuda")
            ):
                (
                    t_1,
                    t_2,
                    _,
                    _,
                    i_1,
                    i_2,
                    _,
                    _,
                ) = model(batch_x)

            if save_linear_outputs:
                if linear_eval is None:
                    raise ValueError(
                        "save_linear_outputs=True, but no linear_eval model was provided."
                    )

                y_pred_1 = linear_eval(i_1)
                y_pred_2 = linear_eval(i_2)

                logits = (y_pred_1 + y_pred_2) / 2.0
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)

                linear_logits_chunks.append(logits.detach().float().cpu().numpy())
                linear_probs_chunks.append(probs.detach().float().cpu().numpy())
                linear_preds_chunks.append(preds.detach().cpu().numpy().astype(np.int64))

            t, i = select_embedding_view(
                t_1=t_1,
                t_2=t_2,
                i_1=i_1,
                i_2=i_2,
                embedding_view=embedding_view,
            )

            flattened = flatten_embeddings_with_mapping(
                t=t,
                i=i,
                batch_x=batch_x,
                sample_indices=sample_indices,
                enable_channel_independence=enable_channel_independence,
            )

            instance_chunks.append(flattened["instance_embeddings"])
            timestamp_chunks.append(flattened["timestamp_embeddings"])

            instance_sample_index_chunks.append(flattened["instance_sample_index"])
            instance_channel_index_chunks.append(flattened["instance_channel_index"])

            timestamp_sample_index_chunks.append(flattened["timestamp_sample_index"])
            timestamp_channel_index_chunks.append(flattened["timestamp_channel_index"])
            timestamp_patch_index_chunks.append(flattened["timestamp_patch_index"])

            if batch_y is not None:
                label_chunks.append(batch_y.detach().cpu().numpy().astype(np.int64))

    instance_embeddings = np.concatenate(instance_chunks, axis=0)
    timestamp_embeddings = np.concatenate(timestamp_chunks, axis=0)

    if l2_normalize:
        instance_embeddings = l2_normalize_np(instance_embeddings).astype(np.float32)
        timestamp_embeddings = l2_normalize_np(timestamp_embeddings).astype(np.float32)

    result = {
        "instance_embeddings": instance_embeddings,
        "timestamp_embeddings": timestamp_embeddings,
        "instance_sample_index": np.concatenate(instance_sample_index_chunks, axis=0),
        "instance_channel_index": np.concatenate(instance_channel_index_chunks, axis=0),
        "timestamp_sample_index": np.concatenate(timestamp_sample_index_chunks, axis=0),
        "timestamp_channel_index": np.concatenate(timestamp_channel_index_chunks, axis=0),
        "timestamp_patch_index": np.concatenate(timestamp_patch_index_chunks, axis=0),
    }

    if label_chunks:
        result["sample_labels"] = np.concatenate(label_chunks, axis=0)

    if save_linear_outputs:
        result["linear_logits"] = np.concatenate(linear_logits_chunks, axis=0)
        result["linear_probs"] = np.concatenate(linear_probs_chunks, axis=0)
        result["linear_predictions"] = np.concatenate(linear_preds_chunks, axis=0)

    return result


def save_embedding_bank(
    bank: Dict[str, np.ndarray],
    meta: Dict[str, Any],
    output_dir: Path,
    output_name: str,
) -> Tuple[Path, Path]:
    """
    Save the bank as compressed npz and metadata as json.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_path = output_dir / f"{output_name}.npz"
    json_path = output_dir / f"{output_name}.meta.json"

    np.savez_compressed(npz_path, **bank)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return npz_path, json_path


def main() -> None:
    bank_args, timedrl_argv = parse_embedding_bank_args()

    # Only pass TimeDRL/common args to TimeDRL's own parser.
    # This avoids conflicts such as --mode being interpreted as --model.
    sys.argv = [sys.argv[0]] + timedrl_argv

    explicit_arg_names = get_explicit_cli_arg_names(timedrl_argv)

    # Parse normal TimeDRL args.
    args = get_args_from_parser()

    # Apply weights/args.json config if --model_for is given.
    args = apply_model_registry_if_requested(
        args=args,
        explicit_arg_names=explicit_arg_names,
        verbose=True,
    )

    # Registry may overwrite use_gpu, so evaluate GPU availability again.
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    args.root_folder = REPO_ROOT

    # Must run AFTER registry config is applied.
    # This finalizes dataset-dependent fields like C, K, d_model, T_p, i_dim.
    args = update_args_from_dataset(args)

    if args.task_name == "forecasting":
        args.setting = f"{args.task_name}_{args.features}_{args.data_name}"
    else:
        args.setting = f"{args.task_name}_{args.data_name}"

    if getattr(args, "model_for", None) is not None:
        print_resolved_registry_args(args)

    # If checkpoint_path was not given explicitly, use the base model from args.json.
    if bank_args.checkpoint_path is None:
        registry_base_model_path = getattr(args, "registry_base_model_path", None)

        if registry_base_model_path is not None:
            bank_args.checkpoint_path = registry_base_model_path
        elif getattr(args, "base_model", None) is not None:
            bank_args.checkpoint_path = str(Path(getattr(args, "weights_dir", "./weights")) / args.base_model)

    if bank_args.checkpoint_path is None:
        raise ValueError(
            "No checkpoint path was provided. Use either --model_for or --checkpoint_path."
        )

    # Optional: auto-resolve linear checkpoint from args.json too.
    if bank_args.save_linear_outputs and bank_args.linear_checkpoint_path is None:
        registry_linear_model_path = getattr(args, "registry_linear_model_path", None)

        if registry_linear_model_path is not None:
            bank_args.linear_checkpoint_path = registry_linear_model_path
        elif getattr(args, "linear_model", None) is not None:
            bank_args.linear_checkpoint_path = str(Path(getattr(args, "weights_dir", "./weights")) / args.linear_model)

    print("\nResolved embedding-bank config:")
    print(f"  model_for:              {getattr(args, 'model_for', None)}")
    print(f"  task_name:              {args.task_name}")
    print(f"  data_name:              {args.data_name}")
    print(f"  features:               {getattr(args, 'features', None)}")
    print(f"  checkpoint_path:         {bank_args.checkpoint_path}")
    print(f"  linear_checkpoint_path:  {bank_args.linear_checkpoint_path}")
    print(f"  bank_split:              {bank_args.bank_split}")
    print(f"  mode:                    {bank_args.mode}")
    print(f"  embedding_view:          {bank_args.embedding_view}")
    print(f"  batch_size:              {args.batch_size}")

    exp = build_experiment(args)
    model = exp.model
    device = exp.device

    if bank_args.force_no_data_aug:
        model.args.data_aug = "none"

    load_model_checkpoint(
        model=model,
        checkpoint_path=Path(bank_args.checkpoint_path),
        device=device,
        strict=not bank_args.allow_partial_checkpoint,
    )

    linear_eval = None

    if bank_args.save_linear_outputs:
        if args.task_name != "classification":
            raise ValueError("save_linear_outputs is only supported for classification.")

        if bank_args.linear_checkpoint_path is None:
            raise ValueError(
                "save_linear_outputs=True, but no linear checkpoint was found. "
                "Provide --linear_checkpoint_path or add linear_model to weights/args.json."
            )

        exp._build_linear_eval()
        linear_eval = exp.linear_eval

        load_model_checkpoint(
            model=linear_eval,
            checkpoint_path=Path(bank_args.linear_checkpoint_path),
            device=device,
            strict=not bank_args.allow_partial_checkpoint,
        )

        linear_eval.eval()

    train_loader, valid_loader, test_loader = load_train_valid_test_loaders(
        args=args,
        mode=bank_args.mode,
    )

    split_to_loader = {
        "train": train_loader,
        "valid": valid_loader,
        "test": test_loader,
    }

    selected_loader = make_ordered_loader(
        split_to_loader[bank_args.bank_split],
        batch_size=args.batch_size,
    )

    bank = build_embedding_bank(
        model=model,
        loader=selected_loader,
        task_name=args.task_name,
        device=device,
        enable_channel_independence=args.enable_channel_independence,
        embedding_view=bank_args.embedding_view,
        l2_normalize=bank_args.l2_normalize,
        max_batches=bank_args.max_batches,
        use_amp=args.use_amp,
        linear_eval=linear_eval,
        save_linear_outputs=bank_args.save_linear_outputs,
    )

    output_name = bank_args.output_name
    if output_name is None:
        output_name = (
            f"{args.task_name}_{args.data_name}_{args.model}_"
            f"{bank_args.bank_split}_{bank_args.embedding_view}_embedding_bank"
        )

    meta = {
        "model_for": getattr(args, "model_for", None),
        "model_registry_path": getattr(args, "model_registry_path", None),
        "weights_dir": getattr(args, "weights_dir", None),
        "task_name": args.task_name,
        "data_name": args.data_name,
        "model": args.model,
        "setting": args.setting,
        "checkpoint_path": str(Path(bank_args.checkpoint_path).resolve()),
        "bank_split": bank_args.bank_split,
        "mode": bank_args.mode,
        "embedding_view": bank_args.embedding_view,
        "force_no_data_aug": bool(bank_args.force_no_data_aug),
        "l2_normalize": bool(bank_args.l2_normalize),
        "enable_channel_independence": bool(args.enable_channel_independence),
        "C": int(args.C),
        "seq_len": int(args.seq_len),
        "patch_len": int(args.patch_len),
        "stride": int(args.stride),
        "T_p": int(args.T_p),
        "d_model": int(args.d_model),
        "i_dim": int(args.i_dim),
        "get_i": args.get_i,
        "num_samples_seen": int(
            bank["instance_sample_index"].max() + 1
            if bank["instance_sample_index"].size > 0
            else 0
        ),
        "instance_embeddings_shape": list(bank["instance_embeddings"].shape),
        "timestamp_embeddings_shape": list(bank["timestamp_embeddings"].shape),
    }

    npz_path, json_path = save_embedding_bank(
        bank=bank,
        meta=meta,
        output_dir=Path(bank_args.output_dir),
        output_name=output_name,
    )

    print("\nDone.")
    print(f"Saved embedding bank: {npz_path}")
    print(f"Saved metadata:       {json_path}")
    print(f"Instance bank shape:  {bank['instance_embeddings'].shape}")
    print(f"Timestamp bank shape: {bank['timestamp_embeddings'].shape}")


if __name__ == "__main__":
    main()