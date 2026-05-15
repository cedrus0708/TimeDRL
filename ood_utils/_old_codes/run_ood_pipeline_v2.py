"""
TimeDRL OOD/anomaly detection pipeline runner v2.

Goals
-----
- One JSON config drives the whole OOD/anomaly pipeline.
- Each run is stored in a clean, timestamped run directory.
- Expensive artifacts can be reused from a content-addressed cache.
- The runner writes logs, commands, status, summary, and a global registry.

Typical usage
-------------
python ood_utils/run_ood_pipeline.py --config ./ood_configs/exchange_injected.json
python ood_utils/run_ood_pipeline.py --config ./ood_configs/har_near_ood.json --run_name debug_har
python ood_utils/run_ood_pipeline.py --config ./ood_configs/exchange_injected.json --dry_run
python ood_utils/run_ood_pipeline.py --config ./ood_configs/exchange_injected.json --no_cache

Expected repository layout
--------------------------
Run this file from the TimeDRL repository root, or keep it inside ood_utils/.
It calls the existing scripts:
- ood_utils/forecasting_csv_injection.py
- ood_utils/embedding_bank.py
- ood_utils/embedding_detector.py
- ood_utils/ood_eval_sets.py
- ood_utils/evaluate_detector.py
- ood_utils/visualize_scores.py
- ood_utils/experiment_dashboard.py              optional
- ood_utils/forecasting_timeseries_browser.py    optional, forecasting only
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]


# -----------------------------------------------------------------------------
# Basic IO helpers
# -----------------------------------------------------------------------------


def load_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON config not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON config must be an object: {path}")
    return data


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(data), f, indent=2, ensure_ascii=False)


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def resolve_repo_path(path: str | Path | None) -> Optional[Path]:
    if path is None:
        return None
    p = Path(path)
    if p.is_absolute():
        return p
    return REPO_ROOT / p


def path_str(path: str | Path) -> str:
    return str(path)


def stable_json_dumps(obj: Any) -> str:
    return json.dumps(to_jsonable(obj), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    path = Path(path)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def file_signature(path: str | Path | None, content_hash: bool = True) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    p = resolve_repo_path(path)
    assert p is not None
    if not p.exists():
        return {"path": str(p), "exists": False}

    stat = p.stat()
    sig: Dict[str, Any] = {
        "path": str(p.resolve()),
        "exists": True,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }
    if content_hash and p.is_file():
        sig["sha256"] = sha256_file(p)
    return sig


def step_cache_key(kind: str, payload: Dict[str, Any]) -> str:
    return f"{kind}_{sha256_text(stable_json_dumps(payload))[:24]}"


def copy_or_link_file(src: Path, dst: Path, mode: str = "copy") -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()

    if mode == "reference":
        # No file is created in the run directory. Caller should use src path directly.
        return

    if mode == "symlink":
        try:
            dst.symlink_to(src.resolve())
            return
        except OSError:
            # Windows often needs elevated privileges for symlinks.
            shutil.copy2(src, dst)
            return

    if mode == "hardlink":
        try:
            os.link(src, dst)
            return
        except OSError:
            shutil.copy2(src, dst)
            return

    if mode == "copy":
        shutil.copy2(src, dst)
        return

    raise ValueError("cache.materialize_mode must be one of: copy, hardlink, symlink, reference")


def copy_optional_file(src: Path, dst: Path, mode: str = "copy") -> bool:
    if not src.exists():
        return False
    copy_or_link_file(src, dst, mode=mode)
    return True


# -----------------------------------------------------------------------------
# Model registry helpers
# -----------------------------------------------------------------------------


def get_model_registry_path(config: Dict[str, Any]) -> str:
    return str(config.get("embedding_bank", {}).get("model_registry_path", "./weights/args.json"))


def load_model_registry(path: str | Path) -> List[Dict[str, Any]]:
    resolved = resolve_repo_path(path)
    assert resolved is not None
    with open(resolved, "r", encoding="utf-8") as f:
        registry = json.load(f)
    if not isinstance(registry, list):
        raise ValueError("Model registry must be a JSON list.")
    return registry


def get_model_for(config: Dict[str, Any]) -> str:
    model_for = config.get("embedding_bank", {}).get("model_for")
    if not model_for:
        raise ValueError("embedding_bank.model_for is required.")
    return str(model_for)


def find_model_entry(registry: List[Dict[str, Any]], model_for: str) -> Dict[str, Any]:
    matches = [
        entry
        for entry in registry
        if str(entry.get("model_for", "")).lower() == str(model_for).lower()
    ]
    if not matches:
        available = [entry.get("model_for") for entry in registry]
        raise ValueError(f"model_for={model_for!r} was not found. Available: {available}")
    if len(matches) > 1:
        raise ValueError(f"Multiple registry entries found for model_for={model_for!r}.")
    return matches[0]


def get_model_entry(config: Dict[str, Any]) -> Dict[str, Any]:
    registry = load_model_registry(get_model_registry_path(config))
    return find_model_entry(registry, get_model_for(config))


def get_task_type(config: Dict[str, Any]) -> str:
    entry = get_model_entry(config)
    run_config = entry.get("run_config", {})
    task_name = run_config.get("task_name")
    if task_name not in {"forecasting", "classification"}:
        raise ValueError(
            f"Invalid or missing task_name in model registry for model_for={get_model_for(config)!r}: {task_name}"
        )
    return str(task_name)


def get_registry_value(config: Dict[str, Any], key: str, default: Any = None) -> Any:
    entry = get_model_entry(config)
    for section in [entry.get("model_config", {}), entry.get("run_config", {})]:
        if key in section:
            return section[key]
    return default


# -----------------------------------------------------------------------------
# Config/name helpers
# -----------------------------------------------------------------------------


def add_optional_value_arg(cmd: List[str], name: str, value: Any) -> None:
    if value is not None:
        cmd.extend([f"--{name}", str(value)])


def add_optional_flag(cmd: List[str], name: str, enabled: bool) -> None:
    if enabled:
        cmd.append(f"--{name}")


def add_optional_list_args(cmd: List[str], name: str, values: Optional[Sequence[Any]]) -> None:
    if values:
        cmd.append(f"--{name}")
        cmd.extend(str(v) for v in values)


def detector_suffix(detector_cfg: Dict[str, Any]) -> str:
    q = str(detector_cfg.get("threshold_quantile", 0.95)).replace(".", "")
    return (
        f"knn_{detector_cfg.get('k', 5)}_"
        f"{detector_cfg.get('metric', 'euclidean')}_"
        f"{detector_cfg.get('score_mode', 'mean')}_"
        f"{detector_cfg.get('normalization', 'standardize')}_"
        f"q{q}"
    )


def class_tag(prefix: str, values: Optional[Sequence[Any]]) -> str:
    if not values:
        return f"{prefix}none"
    return prefix + "".join(str(v) for v in values)


def build_default_names(config: Dict[str, Any]) -> Dict[str, str]:
    task_type = get_task_type(config)
    model_for = get_model_for(config)
    detector_name = detector_suffix(config.get("embedding_detector", {}))

    if task_type == "forecasting":
        injection_cfg = config.get("forecasting_csv_injection", {})
        use_injection = bool(injection_cfg.get("use_injection", False))
        test_tag = "injected" if use_injection else "custom_test"
        return {
            "run_name": f"{model_for}_clean_vs_{test_tag}_{detector_name}",
            "train_bank_name": f"{model_for}_clean_train_reference_embedding_bank",
            "test_bank_name": f"{model_for}_{test_tag}_test_embedding_bank",
            "output_name": f"{model_for}_clean_vs_{test_tag}_{detector_name}",
            "labels_name": f"{model_for}_{test_tag}_test_labels.npz",
            "mask_name": f"{model_for}_{test_tag}_mask.npz",
            "injected_csv_name": f"{model_for}_{test_tag}.csv",
        }

    bank_cfg = config.get("embedding_bank", {})
    id_tag = class_tag("ID", bank_cfg.get("id_classes"))
    near_tag = class_tag("NEAR", bank_cfg.get("near_ood_classes"))
    far_classes = bank_cfg.get("far_ood_classes") or []

    if far_classes:
        far_tag = class_tag("FAR", far_classes)
        test_tag = f"{id_tag}_{near_tag}_{far_tag}"
        output_tag = f"{id_tag}_vs_{near_tag}_{far_tag}"
    else:
        test_tag = f"{id_tag}_{near_tag}"
        output_tag = f"{id_tag}_vs_{near_tag}"

    return {
        "run_name": f"{model_for}_{output_tag}_{detector_name}",
        "train_bank_name": f"{model_for}_{id_tag}_train_reference_embedding_bank",
        "test_bank_name": f"{model_for}_{test_tag}_test_embedding_bank",
        "output_name": f"{model_for}_{output_tag}_{detector_name}",
        "labels_name": f"{model_for}_{test_tag}_test_labels.npz",
    }


def merge_cli_overrides(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    config = json.loads(stable_json_dumps(config))
    config.setdefault("run", {})
    config.setdefault("cache", {})

    if args.output_root is not None:
        config["run"]["output_root"] = args.output_root
    if args.run_name is not None:
        config["run"]["run_name"] = args.run_name
    if args.no_cache:
        config["cache"]["enabled"] = False
    if args.force:
        config["cache"]["force_rebuild"] = True

    return config


# -----------------------------------------------------------------------------
# Run saver / registry
# -----------------------------------------------------------------------------


@dataclass
class StepRecord:
    name: str
    status: str
    command: Optional[str] = None
    cache_key: Optional[str] = None
    cache_hit: Optional[bool] = None
    outputs: Optional[Dict[str, str]] = None
    error: Optional[str] = None


class OODPipelineRunSaver:
    """
    Creates a run directory:

    output_root/
        _cache/
        ood_run_registry.csv
        <model_for>/
            <task_type>/
                <run_name>/
                    <timestamp>/
                        config.json
                        resolved_config.json
                        commands.txt
                        status.json
                        run_summary.json
                        logs/
                        datasets/
                        embedding_banks/
                        embedding_detectors/
                        eval_sets/
                        evaluation_reports/
                        score_visualizations/
    """

    def __init__(self, config: Dict[str, Any], config_path: Path):
        self.config = config
        self.config_path = config_path
        self.task_type = get_task_type(config)
        self.model_for = get_model_for(config)

        names = build_default_names(config)
        run_cfg = config.get("run", {})
        self.output_root = resolve_repo_path(run_cfg.get("output_root", "./ood_runs")) or (REPO_ROOT / "ood_runs")
        self.output_root.mkdir(parents=True, exist_ok=True)

        self.run_name = str(run_cfg.get("run_name") or names["run_name"])
        self.timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.experiment_name = f"{self.model_for}/{self.task_type}/{self.run_name}/{self.timestamp}"

        self.run_dir = self.output_root / self.model_for / self.task_type / self.run_name / self.timestamp
        if self.run_dir.exists():
            raise FileExistsError(f"Run directory already exists: {self.run_dir}")

        self.logs_dir = self.run_dir / "logs"
        self.datasets_dir = self.run_dir / "datasets"
        self.embedding_banks_dir = self.run_dir / "embedding_banks"
        self.embedding_detectors_dir = self.run_dir / "embedding_detectors"
        self.eval_sets_dir = self.run_dir / "eval_sets"
        self.evaluation_reports_dir = self.run_dir / "evaluation_reports"
        self.score_visualizations_dir = self.run_dir / "score_visualizations"

        for p in [
            self.run_dir,
            self.logs_dir,
            self.datasets_dir,
            self.embedding_banks_dir,
            self.embedding_detectors_dir,
            self.eval_sets_dir,
            self.evaluation_reports_dir,
            self.score_visualizations_dir,
        ]:
            p.mkdir(parents=True, exist_ok=True)

        self.registry_path = self.output_root / "ood_run_registry.csv"
        self.commands_path = self.run_dir / "commands.txt"
        self.status_path = self.run_dir / "status.json"
        self.summary_path = self.run_dir / "run_summary.json"

        shutil.copy2(config_path, self.run_dir / "config.original.json")
        save_json(config, self.run_dir / "config.json")
        save_json(config, self.run_dir / "resolved_config.json")

        self.create_registry_entry()

    def registry_fieldnames(self) -> List[str]:
        return [
            "experiment_name",
            "status",
            "task_type",
            "model_for",
            "run_name",
            "run_path",
            "config_path",
            "message",
            "results",
        ]

    def create_registry_entry(self) -> None:
        file_exists = self.registry_path.exists()
        row = {
            "experiment_name": self.experiment_name,
            "status": "running",
            "task_type": self.task_type,
            "model_for": self.model_for,
            "run_name": self.run_name,
            "run_path": str(self.run_dir),
            "config_path": str(self.config_path),
            "message": "Pipeline started.",
            "results": "",
        }
        with open(self.registry_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.registry_fieldnames())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        self.update_status("running", message="Pipeline started.")

    def update_registry(self, status: str, message: str = "", results: Optional[Dict[str, Any]] = None) -> None:
        results = results or {}
        rows: List[Dict[str, str]] = []
        found = False
        if self.registry_path.exists():
            with open(self.registry_path, "r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    if row["experiment_name"] == self.experiment_name:
                        row["status"] = status
                        row["message"] = message
                        row["results"] = stable_json_dumps(results)
                        row["run_path"] = str(self.run_dir)
                        found = True
                    rows.append(row)
        if not found:
            raise RuntimeError(f"Registry entry not found: {self.experiment_name}")
        with open(self.registry_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.registry_fieldnames())
            writer.writeheader()
            writer.writerows(rows)

    def update_status(self, status: str, message: str = "", results: Optional[Dict[str, Any]] = None) -> None:
        payload = {
            "experiment_name": self.experiment_name,
            "status": status,
            "message": message,
            "results": results or {},
            "run_dir": str(self.run_dir),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        save_json(payload, self.status_path)
        self.update_registry(status=status, message=message, results=results or {})

    def append_command(self, name: str, cmd: Sequence[str]) -> None:
        with open(self.commands_path, "a", encoding="utf-8") as f:
            f.write(f"\n# {name}\n")
            f.write(shlex.join([str(c) for c in cmd]))
            f.write("\n")


# -----------------------------------------------------------------------------
# Cache manager
# -----------------------------------------------------------------------------


class CacheManager:
    def __init__(self, config: Dict[str, Any], output_root: Path):
        cache_cfg = config.get("cache", {})
        self.enabled = bool(cache_cfg.get("enabled", True))
        self.force_rebuild = bool(cache_cfg.get("force_rebuild", False))
        self.materialize_mode = str(cache_cfg.get("materialize_mode", "copy"))
        self.hash_file_contents = bool(cache_cfg.get("hash_file_contents", True))
        self.root = resolve_repo_path(cache_cfg.get("cache_root")) or (output_root / "_cache")
        self.root.mkdir(parents=True, exist_ok=True)

    def dir(self, kind: str) -> Path:
        path = self.root / kind
        path.mkdir(parents=True, exist_ok=True)
        return path

    def manifest_path(self, kind: str, cache_key: str) -> Path:
        return self.dir(kind) / f"{cache_key}.cache.json"

    def has_valid_manifest(self, kind: str, cache_key: str, expected_outputs: Sequence[Path]) -> bool:
        if not self.enabled or self.force_rebuild:
            return False
        manifest = self.manifest_path(kind, cache_key)
        if not manifest.exists():
            return False
        for p in expected_outputs:
            if not p.exists():
                return False
        return True

    def save_manifest(self, kind: str, cache_key: str, payload: Dict[str, Any], outputs: Dict[str, Path]) -> None:
        manifest = {
            "cache_key": cache_key,
            "kind": kind,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "payload": payload,
            "outputs": {name: str(path) for name, path in outputs.items()},
        }
        save_json(manifest, self.manifest_path(kind, cache_key))


# -----------------------------------------------------------------------------
# Command execution
# -----------------------------------------------------------------------------


def run_command(saver: OODPipelineRunSaver, name: str, cmd: List[str], dry_run: bool = False) -> None:
    saver.append_command(name, cmd)
    log_path = saver.logs_dir / f"{name}.log"

    print("\n" + "=" * 88)
    print(f"[{name}]")
    print(shlex.join([str(c) for c in cmd]))
    print(f"log: {log_path}")
    print("=" * 88)

    if dry_run:
        return

    with open(log_path, "w", encoding="utf-8") as log_file:
        process = subprocess.run(
            [str(c) for c in cmd],
            cwd=REPO_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )

    if process.returncode != 0:
        raise RuntimeError(
            f"Command failed: {name}. Return code: {process.returncode}. See log: {log_path}"
        )


# -----------------------------------------------------------------------------
# Command builders
# -----------------------------------------------------------------------------


def build_train_model_command(config: Dict[str, Any]) -> Optional[List[str]]:
    if not config.get("run", {}).get("train_model", False):
        return None
    bank_cfg = config["embedding_bank"]
    cmd = [
        sys.executable,
        "main.py",
        "--model_for",
        str(bank_cfg["model_for"]),
        "--batch_size",
        str(bank_cfg.get("batch_size", 256)),
    ]
    add_optional_list_args(cmd, "id_classes", bank_cfg.get("id_classes"))
    add_optional_list_args(cmd, "near_ood_classes", bank_cfg.get("near_ood_classes"))
    add_optional_list_args(cmd, "far_ood_classes", bank_cfg.get("far_ood_classes"))
    return cmd


def build_embedding_bank_command(
    config: Dict[str, Any],
    bank_split: str,
    output_name: str,
    output_dir: Path,
    override_data_path: Optional[str | Path] = None,
    include_ood_classes: bool = False,
) -> List[str]:
    bank_cfg = config["embedding_bank"]
    task_type = get_task_type(config)

    cmd = [
        sys.executable,
        "ood_utils/embedding_bank.py",
        "--model_for",
        str(bank_cfg["model_for"]),
        "--bank_split",
        bank_split,
        "--batch_size",
        str(bank_cfg.get("batch_size", 256)),
        "--output_dir",
        path_str(output_dir),
        "--output_name",
        output_name,
    ]

    add_optional_value_arg(cmd, "model_registry_path", bank_cfg.get("model_registry_path"))
    add_optional_value_arg(cmd, "checkpoint_path", bank_cfg.get("checkpoint_path"))
    add_optional_value_arg(cmd, "linear_checkpoint_path", bank_cfg.get("linear_checkpoint_path"))
    add_optional_value_arg(cmd, "mode", bank_cfg.get("mode"))
    add_optional_value_arg(cmd, "embedding_view", bank_cfg.get("embedding_view"))
    add_optional_value_arg(cmd, "max_batches", bank_cfg.get("max_batches"))
    add_optional_flag(cmd, "l2_normalize", bool(bank_cfg.get("l2_normalize", False)))
    add_optional_flag(cmd, "allow_partial_checkpoint", bool(bank_cfg.get("allow_partial_checkpoint", False)))
    add_optional_flag(cmd, "save_linear_outputs", bool(bank_cfg.get("save_linear_outputs", False)))

    if override_data_path is not None:
        cmd.extend(["--override_data_path", str(override_data_path)])

    if task_type == "classification":
        add_optional_list_args(cmd, "id_classes", bank_cfg.get("id_classes"))
        if include_ood_classes:
            add_optional_list_args(cmd, "near_ood_classes", bank_cfg.get("near_ood_classes"))
            add_optional_list_args(cmd, "far_ood_classes", bank_cfg.get("far_ood_classes"))

    return cmd


def build_forecasting_injection_command(
    config: Dict[str, Any],
    output_csv_path: Path,
    output_mask_path: Path,
) -> List[str]:
    cfg = config.get("forecasting_csv_injection", {})
    input_csv_path = cfg.get("input_csv_path")
    if not input_csv_path:
        raise ValueError("forecasting_csv_injection.input_csv_path is required when use_injection=true.")

    cmd = [
        sys.executable,
        "ood_utils/forecasting_csv_injection.py",
        "--input_csv_path",
        str(input_csv_path),
        "--output_csv_path",
        path_str(output_csv_path),
        "--output_mask_path",
        path_str(output_mask_path),
        "--anomaly_fraction",
        str(cfg.get("anomaly_fraction", 0.05)),
        "--min_len",
        str(cfg.get("min_len", 8)),
        "--max_len",
        str(cfg.get("max_len", 32)),
        "--magnitude",
        str(cfg.get("magnitude", 3.0)),
        "--channel_mode",
        str(cfg.get("channel_mode", "random_one")),
        "--inject_start_ratio",
        str(cfg.get("inject_start_ratio", 0.7)),
        "--inject_end_ratio",
        str(cfg.get("inject_end_ratio", 1.0)),
        "--seed",
        str(cfg.get("seed", 42)),
    ]

    anomaly_types = cfg.get("anomaly_types", ["spike", "level_shift", "noise", "trend", "flatline"])
    if isinstance(anomaly_types, str):
        anomaly_types = [anomaly_types]
    cmd.append("--anomaly_types")
    cmd.extend(str(t) for t in anomaly_types)

    add_optional_list_args(cmd, "value_columns", cfg.get("value_columns"))
    add_optional_value_arg(cmd, "date_column", cfg.get("date_column"))
    add_optional_value_arg(cmd, "source_csv_path", cfg.get("source_csv_path"))
    return cmd


def build_embedding_detector_command(
    config: Dict[str, Any],
    reference_bank_path: Path,
    query_bank_path: Path,
    output_dir: Path,
    output_name: str,
) -> List[str]:
    cfg = config.get("embedding_detector", {})
    cmd = [
        sys.executable,
        "ood_utils/embedding_detector.py",
        "--reference_bank",
        path_str(reference_bank_path),
        "--query_bank",
        path_str(query_bank_path),
        "--output_dir",
        path_str(output_dir),
        "--output_name",
        output_name,
        "--k",
        str(cfg.get("k", 5)),
        "--metric",
        str(cfg.get("metric", "euclidean")),
        "--score_mode",
        str(cfg.get("score_mode", "mean")),
        "--normalization",
        str(cfg.get("normalization", "standardize")),
        "--threshold_quantile",
        str(cfg.get("threshold_quantile", 0.95)),
    ]
    add_optional_value_arg(cmd, "batch_size", cfg.get("batch_size"))
    add_optional_value_arg(cmd, "max_instance_reference_vectors", cfg.get("max_instance_reference_vectors"))
    add_optional_value_arg(cmd, "max_timestamp_reference_vectors", cfg.get("max_timestamp_reference_vectors"))
    add_optional_value_arg(cmd, "random_seed", cfg.get("random_seed"))
    return cmd


def resolve_forecasting_lengths(config: Dict[str, Any]) -> Tuple[int, int, int]:
    labels_cfg = config.get("forecasting_labels", {})
    seq_len = labels_cfg.get("seq_len", get_registry_value(config, "seq_len"))
    patch_len = labels_cfg.get("patch_len", get_registry_value(config, "patch_len"))
    stride = labels_cfg.get("stride", get_registry_value(config, "stride"))
    if seq_len is None or patch_len is None or stride is None:
        raise ValueError(
            "seq_len, patch_len and stride are required. Put them in forecasting_labels "
            "or make sure they exist in weights/args.json."
        )
    return int(seq_len), int(patch_len), int(stride)


def build_forecasting_labels_command(
    config: Dict[str, Any],
    mask_path: Path,
    test_bank_path: Path,
    labels_path: Path,
) -> List[str]:
    seq_len, patch_len, stride = resolve_forecasting_lengths(config)
    labels_cfg = config.get("forecasting_labels", {})
    cmd = [
        sys.executable,
        "ood_utils/ood_eval_sets.py",
        "forecasting-csv-labels",
        "--mask_path",
        path_str(mask_path),
        "--bank_path",
        path_str(test_bank_path),
        "--output_path",
        path_str(labels_path),
        "--seq_len",
        str(seq_len),
        "--patch_len",
        str(patch_len),
        "--stride",
        str(stride),
    ]
    add_optional_value_arg(cmd, "split_start_index", labels_cfg.get("split_start_index"))
    add_optional_value_arg(cmd, "window_start_index_key", labels_cfg.get("window_start_index_key"))
    return cmd


def build_classification_labels_command(config: Dict[str, Any], test_bank_path: Path, labels_path: Path) -> List[str]:
    bank_cfg = config["embedding_bank"]
    cmd = [
        sys.executable,
        "ood_utils/ood_eval_sets.py",
        "classification-labels",
        "--bank_path",
        path_str(test_bank_path),
        "--output_path",
        path_str(labels_path),
    ]
    add_optional_list_args(cmd, "id_classes", bank_cfg.get("id_classes"))
    add_optional_list_args(cmd, "near_ood_classes", bank_cfg.get("near_ood_classes"))
    add_optional_list_args(cmd, "far_ood_classes", bank_cfg.get("far_ood_classes"))
    return cmd


def build_evaluate_command(config: Dict[str, Any], scores_path: Path, labels_path: Path, output_dir: Path, output_name: str) -> List[str]:
    cfg = config.get("evaluate_detector", {})
    cmd = [
        sys.executable,
        "ood_utils/evaluate_detector.py",
        "--scores_path",
        path_str(scores_path),
        "--labels_path",
        path_str(labels_path),
        "--output_dir",
        path_str(output_dir),
        "--output_name",
        output_name,
        "--top_k",
        str(cfg.get("top_k", config.get("visualize_scores", {}).get("top_k", 100))),
    ]
    add_optional_value_arg(cmd, "instance_score_key", cfg.get("instance_score_key"))
    add_optional_value_arg(cmd, "timestamp_score_key", cfg.get("timestamp_score_key"))
    add_optional_value_arg(cmd, "sample_label_key", cfg.get("sample_label_key"))
    add_optional_value_arg(cmd, "sample_group_key", cfg.get("sample_group_key"))
    add_optional_value_arg(cmd, "timestamp_label_key", cfg.get("timestamp_label_key"))
    add_optional_value_arg(cmd, "timestamp_group_key", cfg.get("timestamp_group_key"))
    add_optional_flag(cmd, "skip_instance", bool(cfg.get("skip_instance", False)))
    add_optional_flag(cmd, "skip_timestamp", bool(cfg.get("skip_timestamp", False)))
    return cmd


def build_visualize_scores_command(config: Dict[str, Any], scores_path: Path, labels_path: Path, output_dir: Path, output_name: str) -> List[str]:
    cfg = config.get("visualize_scores", {})
    cmd = [
        sys.executable,
        "ood_utils/visualize_scores.py",
        "--scores_path",
        path_str(scores_path),
        "--labels_path",
        path_str(labels_path),
        "--output_dir",
        path_str(output_dir),
        "--output_name",
        output_name,
        "--top_k",
        str(cfg.get("top_k", 100)),
    ]
    add_optional_value_arg(cmd, "instance_score_key", cfg.get("instance_score_key"))
    add_optional_value_arg(cmd, "timestamp_score_key", cfg.get("timestamp_score_key"))
    add_optional_value_arg(cmd, "sample_id", cfg.get("sample_id"))
    return cmd


def build_experiment_dashboard_command(
    config: Dict[str, Any],
    task_type: str,
    model_for: str,
    scores_path: Path,
    labels_path: Path,
    metrics_path: Path,
    train_bank_path: Path,
    test_bank_path: Path,
    detector_meta_path: Path,
    output_dir: Path,
    output_name: str,
    mask_path: Optional[Path] = None,
    original_csv_path: Optional[Path] = None,
    injected_csv_path: Optional[Path] = None,
) -> List[str]:
    cfg = config.get("experiment_dashboard", {})
    cmd = [
        sys.executable,
        "ood_utils/experiment_dashboard.py",
        "--scores_path",
        path_str(scores_path),
        "--labels_path",
        path_str(labels_path),
        "--metrics_path",
        path_str(metrics_path),
        "--train_bank_path",
        path_str(train_bank_path),
        "--test_bank_path",
        path_str(test_bank_path),
        "--detector_meta_path",
        path_str(detector_meta_path),
        "--task_type",
        task_type,
        "--model_for",
        model_for,
        "--output_dir",
        path_str(output_dir),
        "--output_name",
        output_name,
    ]

    if task_type == "forecasting":
        seq_len, _, _ = resolve_forecasting_lengths(config)
        cmd.extend(["--seq_len", str(cfg.get("seq_len", seq_len))])
        if mask_path is not None:
            cmd.extend(["--mask_path", path_str(mask_path)])
        if original_csv_path is not None:
            cmd.extend(["--original_csv_path", path_str(original_csv_path)])
        if injected_csv_path is not None:
            cmd.extend(["--injected_csv_path", path_str(injected_csv_path)])

    return cmd


def build_forecasting_browser_command(
    config: Dict[str, Any],
    original_csv_path: Path,
    injected_csv_path: Path,
    mask_path: Path,
    scores_path: Path,
    labels_path: Path,
    metrics_path: Path,
    output_dir: Path,
    output_name: str,
    evaluation_reports_dir: Path,
) -> List[str]:
    cfg = config.get("forecasting_timeseries_browser", {})
    seq_len, _, _ = resolve_forecasting_lengths(config)
    top_k = int(cfg.get("top_k", config.get("visualize_scores", {}).get("top_k", 100)))
    top_level = str(cfg.get("top_csv_level", "instance"))
    top_csv_path = cfg.get("top_csv_path")
    if top_csv_path is None:
        top_csv_path = evaluation_reports_dir / f"{top_level}_top_{top_k}_scores.csv"

    cmd = [
        sys.executable,
        "ood_utils/forecasting_timeseries_browser.py",
        "--original_csv_path",
        path_str(original_csv_path),
        "--injected_csv_path",
        path_str(injected_csv_path),
        "--mask_path",
        path_str(mask_path),
        "--scores_path",
        path_str(scores_path),
        "--labels_path",
        path_str(labels_path),
        "--metrics_path",
        path_str(metrics_path),
        "--output_dir",
        path_str(output_dir),
        "--output_name",
        output_name,
        "--seq_len",
        str(cfg.get("seq_len", seq_len)),
        "--top_k",
        str(top_k),
        "--top_csv_path",
        path_str(top_csv_path),
        "--top_csv_level",
        top_level,
    ]
    return cmd


# -----------------------------------------------------------------------------
# Cache payload builders
# -----------------------------------------------------------------------------


def checkpoint_path_for_cache(config: Dict[str, Any]) -> Optional[str]:
    bank_cfg = config.get("embedding_bank", {})
    if bank_cfg.get("checkpoint_path"):
        return str(resolve_repo_path(bank_cfg.get("checkpoint_path")))
    # If checkpoint comes from model registry, include full registry content in the cache payload.
    return None


def embedding_bank_cache_payload(
    config: Dict[str, Any],
    bank_split: str,
    override_data_path: Optional[str | Path],
    include_ood_classes: bool,
    cache: CacheManager,
) -> Dict[str, Any]:
    bank_cfg = config.get("embedding_bank", {})
    payload: Dict[str, Any] = {
        "kind": "embedding_bank",
        "model_for": get_model_for(config),
        "task_type": get_task_type(config),
        "bank_split": bank_split,
        "include_ood_classes": include_ood_classes,
        "model_registry_path": str(resolve_repo_path(get_model_registry_path(config))),
        "model_registry_signature": file_signature(get_model_registry_path(config), cache.hash_file_contents),
        "checkpoint_signature": file_signature(checkpoint_path_for_cache(config), cache.hash_file_contents),
        "bank_cfg": {
            k: bank_cfg.get(k)
            for k in [
                "batch_size",
                "mode",
                "embedding_view",
                "l2_normalize",
                "max_batches",
                "save_linear_outputs",
                "linear_checkpoint_path",
                "allow_partial_checkpoint",
                "id_classes",
                "near_ood_classes" if include_ood_classes else "__skip_near__",
                "far_ood_classes" if include_ood_classes else "__skip_far__",
            ]
            if k in bank_cfg
        },
        "override_data_path": str(resolve_repo_path(override_data_path)) if override_data_path else None,
        "override_data_signature": file_signature(override_data_path, cache.hash_file_contents),
    }
    return payload


def forecasting_injection_cache_payload(config: Dict[str, Any], cache: CacheManager) -> Dict[str, Any]:
    cfg = config.get("forecasting_csv_injection", {})
    return {
        "kind": "forecasting_csv_injection",
        "input_csv_signature": file_signature(cfg.get("input_csv_path"), cache.hash_file_contents),
        "source_csv_signature": file_signature(cfg.get("source_csv_path"), cache.hash_file_contents),
        "params": {
            k: cfg.get(k)
            for k in [
                "value_columns",
                "date_column",
                "anomaly_fraction",
                "anomaly_types",
                "min_len",
                "max_len",
                "magnitude",
                "channel_mode",
                "inject_start_ratio",
                "inject_end_ratio",
                "seed",
            ]
            if k in cfg
        },
    }


def detector_cache_payload(
    config: Dict[str, Any],
    train_bank_cache_key: str,
    test_bank_cache_key: str,
) -> Dict[str, Any]:
    cfg = config.get("embedding_detector", {})
    return {
        "kind": "embedding_detector",
        "reference_bank_cache_key": train_bank_cache_key,
        "query_bank_cache_key": test_bank_cache_key,
        "detector_cfg": {
            k: cfg.get(k)
            for k in [
                "k",
                "metric",
                "score_mode",
                "normalization",
                "threshold_quantile",
                "batch_size",
                "max_instance_reference_vectors",
                "max_timestamp_reference_vectors",
                "random_seed",
            ]
            if k in cfg
        },
    }


def labels_cache_payload(
    config: Dict[str, Any],
    task_type: str,
    bank_cache_key: str,
    mask_cache_key: Optional[str] = None,
) -> Dict[str, Any]:
    if task_type == "forecasting":
        seq_len, patch_len, stride = resolve_forecasting_lengths(config)
        labels_cfg = config.get("forecasting_labels", {})
        return {
            "kind": "forecasting_labels",
            "test_bank_cache_key": bank_cache_key,
            "mask_cache_key": mask_cache_key,
            "seq_len": seq_len,
            "patch_len": patch_len,
            "stride": stride,
            "split_start_index": labels_cfg.get("split_start_index"),
            "window_start_index_key": labels_cfg.get("window_start_index_key", "window_start_index"),
        }
    bank_cfg = config.get("embedding_bank", {})
    return {
        "kind": "classification_labels",
        "test_bank_cache_key": bank_cache_key,
        "id_classes": bank_cfg.get("id_classes"),
        "near_ood_classes": bank_cfg.get("near_ood_classes"),
        "far_ood_classes": bank_cfg.get("far_ood_classes"),
    }


# -----------------------------------------------------------------------------
# Cached step helpers
# -----------------------------------------------------------------------------


def cached_embedding_bank_step(
    saver: OODPipelineRunSaver,
    cache: CacheManager,
    config: Dict[str, Any],
    step_name: str,
    bank_split: str,
    output_name: str,
    run_output_path: Path,
    override_data_path: Optional[str | Path],
    include_ood_classes: bool,
    dry_run: bool,
) -> Tuple[Path, str, StepRecord]:
    payload = embedding_bank_cache_payload(config, bank_split, override_data_path, include_ood_classes, cache)
    cache_key = step_cache_key("bank", payload)
    cache_dir = cache.dir("embedding_banks")
    cached_npz = cache_dir / f"{cache_key}.npz"
    cached_meta = cache_dir / f"{cache_key}.meta.json"

    expected = [cached_npz, cached_meta]
    if cache.has_valid_manifest("embedding_banks", cache_key, expected):
        materialized = cached_npz if cache.materialize_mode == "reference" else run_output_path
        copy_or_link_file(cached_npz, run_output_path, mode=cache.materialize_mode)
        copy_optional_file(cached_meta, run_output_path.with_suffix(".meta.json"), mode=cache.materialize_mode)
        print(f"[cache hit] {step_name}: {cache_key}")
        return materialized, cache_key, StepRecord(
            name=step_name,
            status="cache_hit",
            cache_key=cache_key,
            cache_hit=True,
            outputs={"bank": str(materialized)},
        )

    cmd = build_embedding_bank_command(
        config=config,
        bank_split=bank_split,
        output_name=output_name,
        output_dir=run_output_path.parent,
        override_data_path=override_data_path,
        include_ood_classes=include_ood_classes,
    )
    if not dry_run:
        run_command(saver, step_name, cmd, dry_run=False)
        if not run_output_path.exists():
            raise FileNotFoundError(f"Embedding bank was not created: {run_output_path}")
        if cache.enabled:
            shutil.copy2(run_output_path, cached_npz)
            meta_path = run_output_path.with_suffix(".meta.json")
            if meta_path.exists():
                shutil.copy2(meta_path, cached_meta)
            cache.save_manifest(
                "embedding_banks",
                cache_key,
                payload,
                {"npz": cached_npz, "meta_json": cached_meta},
            )
    else:
        run_command(saver, step_name, cmd, dry_run=True)

    return run_output_path, cache_key, StepRecord(
        name=step_name,
        status="skipped_dry_run" if dry_run else "finished",
        command=shlex.join(cmd),
        cache_key=cache_key,
        cache_hit=False,
        outputs={"bank": str(run_output_path)},
    )


def cached_forecasting_injection_step(
    saver: OODPipelineRunSaver,
    cache: CacheManager,
    config: Dict[str, Any],
    output_csv_path: Path,
    output_mask_path: Path,
    dry_run: bool,
) -> Tuple[Path, Path, str, StepRecord]:
    payload = forecasting_injection_cache_payload(config, cache)
    cache_key = step_cache_key("inject", payload)
    cache_dir = cache.dir("forecasting_injections")
    cached_csv = cache_dir / f"{cache_key}.csv"
    cached_mask = cache_dir / f"{cache_key}.mask.npz"
    cached_meta = cache_dir / f"{cache_key}.mask.meta.json"

    expected = [cached_csv, cached_mask]
    if cache.has_valid_manifest("forecasting_injections", cache_key, expected):
        csv_path = cached_csv if cache.materialize_mode == "reference" else output_csv_path
        mask_path = cached_mask if cache.materialize_mode == "reference" else output_mask_path
        copy_or_link_file(cached_csv, output_csv_path, mode=cache.materialize_mode)
        copy_or_link_file(cached_mask, output_mask_path, mode=cache.materialize_mode)
        copy_optional_file(cached_meta, output_mask_path.with_suffix(".meta.json"), mode=cache.materialize_mode)
        print(f"[cache hit] 02_forecasting_csv_injection: {cache_key}")
        return csv_path, mask_path, cache_key, StepRecord(
            name="02_forecasting_csv_injection",
            status="cache_hit",
            cache_key=cache_key,
            cache_hit=True,
            outputs={"injected_csv": str(csv_path), "mask": str(mask_path)},
        )

    cmd = build_forecasting_injection_command(config, output_csv_path, output_mask_path)
    if not dry_run:
        run_command(saver, "02_forecasting_csv_injection", cmd, dry_run=False)
        if not output_csv_path.exists() or not output_mask_path.exists():
            raise FileNotFoundError("Forecasting injection did not create the expected CSV/mask outputs.")
        if cache.enabled:
            shutil.copy2(output_csv_path, cached_csv)
            shutil.copy2(output_mask_path, cached_mask)
            meta_path = output_mask_path.with_suffix(".meta.json")
            if meta_path.exists():
                shutil.copy2(meta_path, cached_meta)
            cache.save_manifest(
                "forecasting_injections",
                cache_key,
                payload,
                {"csv": cached_csv, "mask": cached_mask, "meta_json": cached_meta},
            )
    else:
        run_command(saver, "02_forecasting_csv_injection", cmd, dry_run=True)

    return output_csv_path, output_mask_path, cache_key, StepRecord(
        name="02_forecasting_csv_injection",
        status="skipped_dry_run" if dry_run else "finished",
        command=shlex.join(cmd),
        cache_key=cache_key,
        cache_hit=False,
        outputs={"injected_csv": str(output_csv_path), "mask": str(output_mask_path)},
    )


def cached_detector_step(
    saver: OODPipelineRunSaver,
    cache: CacheManager,
    config: Dict[str, Any],
    train_bank_path: Path,
    test_bank_path: Path,
    train_bank_cache_key: str,
    test_bank_cache_key: str,
    output_name: str,
    dry_run: bool,
) -> Tuple[Path, Path, str, StepRecord]:
    payload = detector_cache_payload(config, train_bank_cache_key, test_bank_cache_key)
    cache_key = step_cache_key("detector", payload)
    cache_dir = cache.dir("embedding_detectors")

    cached_scores = cache_dir / f"{cache_key}.scores.npz"
    cached_instance = cache_dir / f"{cache_key}.instance_detector.npz"
    cached_timestamp = cache_dir / f"{cache_key}.timestamp_detector.npz"
    cached_meta = cache_dir / f"{cache_key}.detector_meta.json"

    run_scores = saver.embedding_detectors_dir / f"{output_name}.scores.npz"
    run_instance = saver.embedding_detectors_dir / f"{output_name}.instance_detector.npz"
    run_timestamp = saver.embedding_detectors_dir / f"{output_name}.timestamp_detector.npz"
    run_meta = saver.embedding_detectors_dir / f"{output_name}.detector_meta.json"

    expected = [cached_scores, cached_instance, cached_timestamp, cached_meta]
    if cache.has_valid_manifest("embedding_detectors", cache_key, expected):
        scores_path = cached_scores if cache.materialize_mode == "reference" else run_scores
        meta_path = cached_meta if cache.materialize_mode == "reference" else run_meta
        copy_or_link_file(cached_scores, run_scores, mode=cache.materialize_mode)
        copy_or_link_file(cached_instance, run_instance, mode=cache.materialize_mode)
        copy_or_link_file(cached_timestamp, run_timestamp, mode=cache.materialize_mode)
        copy_or_link_file(cached_meta, run_meta, mode=cache.materialize_mode)
        print(f"[cache hit] 04_embedding_detector: {cache_key}")
        return scores_path, meta_path, cache_key, StepRecord(
            name="04_embedding_detector",
            status="cache_hit",
            cache_key=cache_key,
            cache_hit=True,
            outputs={"scores": str(scores_path), "detector_meta": str(meta_path)},
        )

    cmd = build_embedding_detector_command(
        config=config,
        reference_bank_path=train_bank_path,
        query_bank_path=test_bank_path,
        output_dir=saver.embedding_detectors_dir,
        output_name=output_name,
    )
    if not dry_run:
        run_command(saver, "04_embedding_detector", cmd, dry_run=False)
        for p in [run_scores, run_instance, run_timestamp, run_meta]:
            if not p.exists():
                raise FileNotFoundError(f"Detector output was not created: {p}")
        if cache.enabled:
            shutil.copy2(run_scores, cached_scores)
            shutil.copy2(run_instance, cached_instance)
            shutil.copy2(run_timestamp, cached_timestamp)
            shutil.copy2(run_meta, cached_meta)
            cache.save_manifest(
                "embedding_detectors",
                cache_key,
                payload,
                {
                    "scores": cached_scores,
                    "instance_detector": cached_instance,
                    "timestamp_detector": cached_timestamp,
                    "meta_json": cached_meta,
                },
            )
    else:
        run_command(saver, "04_embedding_detector", cmd, dry_run=True)

    return run_scores, run_meta, cache_key, StepRecord(
        name="04_embedding_detector",
        status="skipped_dry_run" if dry_run else "finished",
        command=shlex.join(cmd),
        cache_key=cache_key,
        cache_hit=False,
        outputs={"scores": str(run_scores), "detector_meta": str(run_meta)},
    )


def cached_labels_step(
    saver: OODPipelineRunSaver,
    cache: CacheManager,
    config: Dict[str, Any],
    task_type: str,
    test_bank_path: Path,
    labels_path: Path,
    test_bank_cache_key: str,
    mask_path: Optional[Path],
    mask_cache_key: Optional[str],
    dry_run: bool,
) -> Tuple[Path, str, StepRecord]:
    payload = labels_cache_payload(config, task_type, test_bank_cache_key, mask_cache_key)
    cache_key = step_cache_key("labels", payload)
    cache_dir = cache.dir("eval_sets")
    cached_labels = cache_dir / f"{cache_key}.npz"

    if cache.has_valid_manifest("eval_sets", cache_key, [cached_labels]):
        out_path = cached_labels if cache.materialize_mode == "reference" else labels_path
        copy_or_link_file(cached_labels, labels_path, mode=cache.materialize_mode)
        print(f"[cache hit] 05_build_eval_labels: {cache_key}")
        return out_path, cache_key, StepRecord(
            name="05_build_eval_labels",
            status="cache_hit",
            cache_key=cache_key,
            cache_hit=True,
            outputs={"labels": str(out_path)},
        )

    if task_type == "forecasting":
        if mask_path is None:
            raise ValueError("mask_path is required for forecasting labels.")
        cmd = build_forecasting_labels_command(config, mask_path, test_bank_path, labels_path)
    else:
        cmd = build_classification_labels_command(config, test_bank_path, labels_path)

    if not dry_run:
        run_command(saver, "05_build_eval_labels", cmd, dry_run=False)
        if not labels_path.exists():
            raise FileNotFoundError(f"Labels file was not created: {labels_path}")
        if cache.enabled:
            shutil.copy2(labels_path, cached_labels)
            cache.save_manifest("eval_sets", cache_key, payload, {"labels": cached_labels})
    else:
        run_command(saver, "05_build_eval_labels", cmd, dry_run=True)

    return labels_path, cache_key, StepRecord(
        name="05_build_eval_labels",
        status="skipped_dry_run" if dry_run else "finished",
        command=shlex.join(cmd),
        cache_key=cache_key,
        cache_hit=False,
        outputs={"labels": str(labels_path)},
    )


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------


def run_uncached_step(
    saver: OODPipelineRunSaver,
    name: str,
    cmd: List[str],
    dry_run: bool,
) -> StepRecord:
    run_command(saver, name, cmd, dry_run=dry_run)
    return StepRecord(
        name=name,
        status="skipped_dry_run" if dry_run else "finished",
        command=shlex.join(cmd),
        cache_hit=False,
    )


def read_metrics_summary(metrics_path: Path) -> Dict[str, Any]:
    if not metrics_path.exists():
        return {}
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_summary(saver: OODPipelineRunSaver, summary: Dict[str, Any]) -> None:
    save_json(summary, saver.summary_path)


def run_pipeline(config_path: Path, args: argparse.Namespace) -> None:
    raw_config = load_json(config_path)
    config = merge_cli_overrides(raw_config, args)

    names = build_default_names(config)
    saver = OODPipelineRunSaver(config=config, config_path=config_path)
    cache = CacheManager(config=config, output_root=saver.output_root)

    task_type = saver.task_type
    model_for = saver.model_for

    train_bank_name = names["train_bank_name"]
    test_bank_name = names["test_bank_name"]
    output_name = names["output_name"]
    labels_name = names["labels_name"]

    run_train_bank_path = saver.embedding_banks_dir / f"{train_bank_name}.npz"
    run_test_bank_path = saver.embedding_banks_dir / f"{test_bank_name}.npz"
    labels_path = saver.eval_sets_dir / labels_name
    metrics_path = saver.evaluation_reports_dir / f"{output_name}.metrics_summary.json"

    summary: Dict[str, Any] = {
        "experiment_name": saver.experiment_name,
        "run_dir": str(saver.run_dir),
        "task_type": task_type,
        "model_for": model_for,
        "run_name": saver.run_name,
        "output_name": output_name,
        "cache": {
            "enabled": cache.enabled,
            "force_rebuild": cache.force_rebuild,
            "cache_root": str(cache.root),
            "materialize_mode": cache.materialize_mode,
            "hash_file_contents": cache.hash_file_contents,
        },
        "paths": {
            "datasets": str(saver.datasets_dir),
            "embedding_banks": str(saver.embedding_banks_dir),
            "embedding_detectors": str(saver.embedding_detectors_dir),
            "eval_sets": str(saver.eval_sets_dir),
            "evaluation_reports": str(saver.evaluation_reports_dir),
            "score_visualizations": str(saver.score_visualizations_dir),
            "logs": str(saver.logs_dir),
            "commands": str(saver.commands_path),
        },
        "steps": [],
    }

    def add_step(record: StepRecord) -> None:
        summary["steps"].append(to_jsonable(record.__dict__))
        write_summary(saver, summary)

    try:
        # 00 optional model training
        train_model_cmd = build_train_model_command(config)
        if train_model_cmd is not None:
            add_step(run_uncached_step(saver, "00_train_model", train_model_cmd, args.dry_run))

        # 01 train/reference bank, cached aggressively
        train_bank_path, train_bank_cache_key, record = cached_embedding_bank_step(
            saver=saver,
            cache=cache,
            config=config,
            step_name="01_train_reference_bank",
            bank_split="train",
            output_name=train_bank_name,
            run_output_path=run_train_bank_path,
            override_data_path=None,
            include_ood_classes=False,
            dry_run=args.dry_run,
        )
        add_step(record)

        injected_csv_path: Optional[Path] = None
        original_csv_path: Optional[Path] = None
        mask_path: Optional[Path] = None
        mask_cache_key: Optional[str] = None
        test_override_data_path: Optional[str | Path] = None

        # 02/03 task-specific test data and test bank
        if task_type == "forecasting":
            injection_cfg = config.get("forecasting_csv_injection", {})
            use_injection = bool(injection_cfg.get("use_injection", False))
            original_csv_path = resolve_repo_path(injection_cfg.get("input_csv_path"))

            if use_injection:
                run_injected_csv = saver.datasets_dir / names["injected_csv_name"]
                run_mask = saver.eval_sets_dir / names["mask_name"]
                injected_csv_path, mask_path, mask_cache_key, record = cached_forecasting_injection_step(
                    saver=saver,
                    cache=cache,
                    config=config,
                    output_csv_path=run_injected_csv,
                    output_mask_path=run_mask,
                    dry_run=args.dry_run,
                )
                add_step(record)
                test_override_data_path = injected_csv_path
            else:
                override_path = config.get("embedding_bank", {}).get("override_data_path")
                mask_raw = injection_cfg.get("output_mask_path") or config.get("forecasting_labels", {}).get("mask_path")
                if override_path is None:
                    raise ValueError(
                        "For forecasting without injection, set embedding_bank.override_data_path."
                    )
                if mask_raw is None:
                    raise ValueError(
                        "For forecasting without injection, set forecasting_csv_injection.output_mask_path "
                        "or forecasting_labels.mask_path."
                    )
                test_override_data_path = resolve_repo_path(override_path)
                injected_csv_path = resolve_repo_path(override_path)
                mask_path = resolve_repo_path(mask_raw)
                mask_cache_key = sha256_text(stable_json_dumps(file_signature(mask_path, cache.hash_file_contents)))[:24]

            test_bank_path, test_bank_cache_key, record = cached_embedding_bank_step(
                saver=saver,
                cache=cache,
                config=config,
                step_name="03_test_embedding_bank",
                bank_split="test",
                output_name=test_bank_name,
                run_output_path=run_test_bank_path,
                override_data_path=test_override_data_path,
                include_ood_classes=False,
                dry_run=args.dry_run,
            )
            add_step(record)

        elif task_type == "classification":
            test_bank_path, test_bank_cache_key, record = cached_embedding_bank_step(
                saver=saver,
                cache=cache,
                config=config,
                step_name="03_test_embedding_bank",
                bank_split="test",
                output_name=test_bank_name,
                run_output_path=run_test_bank_path,
                override_data_path=None,
                include_ood_classes=True,
                dry_run=args.dry_run,
            )
            add_step(record)
        else:
            raise ValueError(f"Unknown task_type: {task_type}")

        # 04 detector, cached on train/test bank cache keys + detector params
        scores_path, detector_meta_path, detector_cache_key, record = cached_detector_step(
            saver=saver,
            cache=cache,
            config=config,
            train_bank_path=train_bank_path,
            test_bank_path=test_bank_path,
            train_bank_cache_key=train_bank_cache_key,
            test_bank_cache_key=test_bank_cache_key,
            output_name=output_name,
            dry_run=args.dry_run,
        )
        add_step(record)

        # 05 labels, cached
        labels_path, labels_cache_key, record = cached_labels_step(
            saver=saver,
            cache=cache,
            config=config,
            task_type=task_type,
            test_bank_path=test_bank_path,
            labels_path=labels_path,
            test_bank_cache_key=test_bank_cache_key,
            mask_path=mask_path,
            mask_cache_key=mask_cache_key,
            dry_run=args.dry_run,
        )
        add_step(record)

        # 06 evaluate: cheap, always run to generate current-run reports
        eval_cmd = build_evaluate_command(config, scores_path, labels_path, saver.evaluation_reports_dir, output_name)
        add_step(run_uncached_step(saver, "06_evaluate_detector", eval_cmd, args.dry_run))

        # 07 old visualizer: optional, default true
        vis_cfg = config.get("visualize_scores", {})
        if bool(vis_cfg.get("enabled", True)):
            vis_cmd = build_visualize_scores_command(config, scores_path, labels_path, saver.score_visualizations_dir, output_name)
            add_step(run_uncached_step(saver, "07_visualize_scores", vis_cmd, args.dry_run))

        # 08 new experiment dashboard: optional, default true if script exists
        dash_cfg = config.get("experiment_dashboard", {})
        dashboard_script_exists = (REPO_ROOT / "ood_utils" / "experiment_dashboard.py").exists()
        if bool(dash_cfg.get("enabled", dashboard_script_exists)):
            dashboard_cmd = build_experiment_dashboard_command(
                config=config,
                task_type=task_type,
                model_for=model_for,
                scores_path=scores_path,
                labels_path=labels_path,
                metrics_path=metrics_path,
                train_bank_path=train_bank_path,
                test_bank_path=test_bank_path,
                detector_meta_path=detector_meta_path,
                output_dir=saver.score_visualizations_dir,
                output_name=output_name,
                mask_path=mask_path,
                original_csv_path=original_csv_path,
                injected_csv_path=injected_csv_path,
            )
            add_step(run_uncached_step(saver, "08_experiment_dashboard", dashboard_cmd, args.dry_run))

        # 09 forecasting full browser: optional, default true if forecasting and script exists
        browser_cfg = config.get("forecasting_timeseries_browser", {})
        browser_script_exists = (REPO_ROOT / "ood_utils" / "forecasting_timeseries_browser.py").exists()
        if task_type == "forecasting" and bool(browser_cfg.get("enabled", browser_script_exists)):
            if original_csv_path is None or injected_csv_path is None or mask_path is None:
                raise ValueError("Forecasting browser needs original_csv_path, injected_csv_path and mask_path.")
            browser_cmd = build_forecasting_browser_command(
                config=config,
                original_csv_path=original_csv_path,
                injected_csv_path=injected_csv_path,
                mask_path=mask_path,
                scores_path=scores_path,
                labels_path=labels_path,
                metrics_path=metrics_path,
                output_dir=saver.score_visualizations_dir,
                output_name=output_name,
                evaluation_reports_dir=saver.evaluation_reports_dir,
            )
            add_step(run_uncached_step(saver, "09_forecasting_timeseries_browser", browser_cmd, args.dry_run))

        if not args.dry_run:
            summary["metrics_summary"] = read_metrics_summary(metrics_path)

        summary["final_outputs"] = {
            "train_bank": str(train_bank_path),
            "test_bank": str(test_bank_path),
            "scores": str(scores_path),
            "labels": str(labels_path),
            "metrics": str(metrics_path),
            "detector_meta": str(detector_meta_path),
            "score_visualizations": str(saver.score_visualizations_dir),
        }
        if mask_path is not None:
            summary["final_outputs"]["mask"] = str(mask_path)
        if injected_csv_path is not None:
            summary["final_outputs"]["injected_csv"] = str(injected_csv_path)

        write_summary(saver, summary)
        saver.update_status(
            "dry_run" if args.dry_run else "finished",
            message="Dry run finished." if args.dry_run else "Pipeline finished.",
            results=summary,
        )

        print("\nPipeline finished.")
        print(f"Run directory: {saver.run_dir}")
        print(f"Commands:      {saver.commands_path}")
        print(f"Summary:       {saver.summary_path}")
        print(f"Registry:      {saver.registry_path}")

    except Exception as exc:
        error_text = traceback.format_exc()
        error_path = saver.run_dir / "error.txt"
        with open(error_path, "w", encoding="utf-8") as f:
            f.write(error_text)

        summary["error"] = str(exc)
        summary["error_path"] = str(error_path)
        write_summary(saver, summary)
        saver.update_status("failed", message=str(exc), results=summary)

        print("\nPipeline failed.")
        print(f"Run directory: {saver.run_dir}")
        print(f"Error log:     {error_path}")
        raise


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full TimeDRL OOD/anomaly detection pipeline from a JSON config."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to pipeline JSON config.")
    parser.add_argument("--run_name", type=str, default=None, help="Override run.run_name from config.")
    parser.add_argument("--output_root", type=str, default=None, help="Override run.output_root from config.")
    parser.add_argument("--dry_run", action="store_true", default=False, help="Print/write commands but do not execute them.")
    parser.add_argument("--no_cache", action="store_true", default=False, help="Disable all cache reuse for this run.")
    parser.add_argument("--force", action="store_true", default=False, help="Force rebuild cached artifacts and overwrite cache entries.")
    parser.add_argument(
        "--continue_on_error",
        action="store_true",
        default=False,
        help="Deprecated placeholder. The v2 runner stops on the first failed step.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(config_path=Path(args.config), args=args)


if __name__ == "__main__":
    main()
