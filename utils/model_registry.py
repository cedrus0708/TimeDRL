import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import argparse


REGISTRY_CONTROL_ARGS = {
    "model_for",
    "model_registry_path",
    "weights_dir",
    "checkpoint_role",
}


def load_model_registry(registry_path: str | Path) -> List[Dict[str, Any]]:
    registry_path = Path(registry_path)

    if not registry_path.exists():
        raise FileNotFoundError(f"Model registry not found: {registry_path}")

    with open(registry_path, "r", encoding="utf-8") as f:
        registry = json.load(f)

    if not isinstance(registry, list):
        raise ValueError("Model registry must be a JSON list.")

    return registry


def find_model_entry(
    registry: List[Dict[str, Any]],
    model_for: str,
) -> Dict[str, Any]:
    matches = [
        entry
        for entry in registry
        if str(entry.get("model_for", "")).lower() == str(model_for).lower()
    ]

    if not matches:
        available = [entry.get("model_for") for entry in registry]
        raise ValueError(
            f"model_for='{model_for}' was not found in registry. "
            f"Available values: {available}"
        )

    if len(matches) > 1:
        raise ValueError(f"Multiple entries found for model_for='{model_for}'.")

    return deepcopy(matches[0])


def normalize_config_value(key: str, value: Any) -> Any:
    """
    Normalize values coming from weights/args.json.

    Your registry currently stores pred_len_list as an int:
        "pred_len_list": 24

    TimeDRL's argparse usually expects a list:
        [24]
    """
    if key == "pred_len_list":
        if isinstance(value, list):
            return [int(v) for v in value]
        return [int(value)]

    return value


def apply_model_entry_to_args(
    args: argparse.Namespace,
    entry: Dict[str, Any],
    verbose: bool = True,
) -> argparse.Namespace:
    """
    Apply run_config and model_config from the registry to TimeDRL args.

    run_config:
        task_name, data_name, features, batch_size, use_gpu, ...

    model_config:
        base_d_model, n_layers, patch_len, stride, token_embed_type, ...
    """
    run_config = entry.get("run_config", {})
    model_config = entry.get("model_config", {})

    if not isinstance(run_config, dict):
        raise ValueError("run_config must be a dictionary.")

    if not isinstance(model_config, dict):
        raise ValueError("model_config must be a dictionary.")

    for section_name, config in [
        ("run_config", run_config),
        ("model_config", model_config),
    ]:
        for key, value in config.items():
            normalized_value = normalize_config_value(key, value)
            setattr(args, key, normalized_value)

            if verbose:
                print(f"[model registry] {section_name}.{key} = {normalized_value}")

    if hasattr(args, "pred_len_list") and args.pred_len_list:
        args.pred_len = int(args.pred_len_list[0])

        if verbose:
            print(f"[model registry] derived pred_len = {args.pred_len}")

    return args


def resolve_registry_checkpoint(
    entry: Dict[str, Any],
    checkpoint_role: str,
    weights_dir: str | Path,
) -> Path:
    checkpoints = entry.get("checkpoints", {})

    if not isinstance(checkpoints, dict):
        raise ValueError("checkpoints must be a dictionary.")

    if checkpoint_role not in checkpoints:
        raise KeyError(
            f"Checkpoint role '{checkpoint_role}' not found. "
            f"Available roles: {list(checkpoints.keys())}"
        )

    checkpoint_name = checkpoints[checkpoint_role]
    checkpoint_path = Path(checkpoint_name)

    if not checkpoint_path.is_absolute():
        checkpoint_path = Path(weights_dir) / checkpoint_path

    return checkpoint_path


def apply_registry_checkpoint_names(
    args: argparse.Namespace,
    entry: Dict[str, Any],
    weights_dir: str | Path,
    verbose: bool = True,
) -> argparse.Namespace:
    """
    Store registry checkpoint information on args.

    This does not force loading. It only makes the paths/names available
    for the existing TimeDRL logic.
    """
    checkpoints = entry.get("checkpoints", {})

    if not isinstance(checkpoints, dict):
        return args

    args.checkpoints = str(weights_dir)

    if "base_model" in checkpoints:
        args.base_model = checkpoints["base_model"]
        args.registry_base_model_path = str(Path(weights_dir) / checkpoints["base_model"])

        if verbose:
            print(f"[model registry] base_model = {args.base_model}")
            print(f"[model registry] registry_base_model_path = {args.registry_base_model_path}")

    if "linear_model" in checkpoints:
        args.linear_model = checkpoints["linear_model"]
        args.registry_linear_model_path = str(Path(weights_dir) / checkpoints["linear_model"])

        if verbose:
            print(f"[model registry] linear_model = {args.linear_model}")
            print(f"[model registry] registry_linear_model_path = {args.registry_linear_model_path}")

    return args


def get_explicit_cli_arg_names(argv: Optional[List[str]] = None) -> Set[str]:
    """
    Return explicitly provided CLI argument names.

    Example:
        ["--model_for", "Exchange", "--batch_size", "128"]

    returns:
        {"model_for", "batch_size"}
    """
    if argv is None:
        argv = sys.argv[1:]

    names = set()

    for item in argv:
        if not item.startswith("--"):
            continue

        name = item[2:].split("=", 1)[0]
        name = name.replace("-", "_")
        names.add(name)

    return names


def snapshot_cli_values(
    args: argparse.Namespace,
    explicit_arg_names: Set[str],
) -> Dict[str, Any]:
    """
    Save values that came explicitly from CLI before registry overwrites args.
    """
    snapshot = {}

    for name in explicit_arg_names:
        if hasattr(args, name):
            snapshot[name] = getattr(args, name)

    return snapshot


def reapply_explicit_cli_overrides(
    args: argparse.Namespace,
    explicit_values: Dict[str, Any],
    verbose: bool = True,
) -> argparse.Namespace:
    """
    Re-apply CLI overrides after loading registry config.

    This allows:
        python main.py --model_for Exchange --batch_size 128

    even if args.json has:
        "batch_size": 8
    """
    for key, value in explicit_values.items():
        if key in REGISTRY_CONTROL_ARGS:
            continue

        if hasattr(args, key):
            setattr(args, key, value)

            if verbose:
                print(f"[CLI override] {key} = {value}")

    return args


def apply_model_registry_if_requested(
    args: argparse.Namespace,
    explicit_arg_names: Optional[Set[str]] = None,
    verbose: bool = True,
) -> argparse.Namespace:
    """
    Main entry point for TimeDRL main.py.

    Usage:
        args = get_args_from_parser()
        args = apply_model_registry_if_requested(args)
    """
    if explicit_arg_names is None:
        explicit_arg_names = get_explicit_cli_arg_names()

    explicit_values = snapshot_cli_values(args, explicit_arg_names)

    model_for = getattr(args, "model_for", None)

    if model_for is None:
        return args

    registry_path = getattr(args, "model_registry_path", "./weights/args.json")
    weights_dir = getattr(args, "weights_dir", "./weights")

    if verbose:
        print("\n" + "=" * 80)
        print(f"Loading TimeDRL config from registry for model_for='{model_for}'")
        print(f"Registry path: {registry_path}")
        print(f"Weights dir:    {weights_dir}")
        print("=" * 80)

    registry = load_model_registry(registry_path)
    entry = find_model_entry(registry, model_for)

    args = apply_model_entry_to_args(args, entry, verbose=verbose)
    args = apply_registry_checkpoint_names(args, entry, weights_dir, verbose=verbose)

    args.model_for = model_for
    args.model_registry_path = registry_path
    args.weights_dir = weights_dir

    args = reapply_explicit_cli_overrides(
        args=args,
        explicit_values=explicit_values,
        verbose=verbose,
    )

    return args


def print_resolved_registry_args(args: argparse.Namespace) -> None:
    """
    Debug helper. Call this after update_args_from_dataset(args), because d_model
    is usually finalized there.
    """
    print("\nResolved TimeDRL args:")
    print(f"  model_for:                   {getattr(args, 'model_for', None)}")
    print(f"  task_name:                   {getattr(args, 'task_name', None)}")
    print(f"  data_name:                   {getattr(args, 'data_name', None)}")
    print(f"  features:                    {getattr(args, 'features', None)}")
    print(f"  seq_len:                     {getattr(args, 'seq_len', None)}")
    print(f"  pred_len:                    {getattr(args, 'pred_len', None)}")
    print(f"  pred_len_list:               {getattr(args, 'pred_len_list', None)}")
    print(f"  patch_len:                   {getattr(args, 'patch_len', None)}")
    print(f"  stride:                      {getattr(args, 'stride', None)}")
    print(f"  base_d_model:                {getattr(args, 'base_d_model', None)}")
    print(f"  d_model:                     {getattr(args, 'd_model', None)}")
    print(f"  n_layers:                    {getattr(args, 'n_layers', None)}")
    print(f"  n_heads:                     {getattr(args, 'n_heads', None)}")
    print(f"  token_embed_type:            {getattr(args, 'token_embed_type', None)}")
    print(f"  token_embed_kernel_size:     {getattr(args, 'token_embed_kernel_size', None)}")
    print(f"  pos_embed_type:              {getattr(args, 'pos_embed_type', None)}")
    print(f"  enable_channel_independence: {getattr(args, 'enable_channel_independence', None)}")
    print(f"  batch_size:                  {getattr(args, 'batch_size', None)}")
    print(f"  use_gpu:                     {getattr(args, 'use_gpu', None)}")
    print(f"  use_amp:                     {getattr(args, 'use_amp', None)}")
    print(f"  checkpoints:                 {getattr(args, 'checkpoints', None)}")
    print(f"  base_model:                  {getattr(args, 'base_model', None)}")
    print(f"  linear_model:                {getattr(args, 'linear_model', None)}")