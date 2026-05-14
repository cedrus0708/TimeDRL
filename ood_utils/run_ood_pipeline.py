import argparse
import csv
import json
import shlex
import shutil
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_repo_path(path: str | Path) -> Path:
    path = Path(path)

    if path.is_absolute():
        return path

    return REPO_ROOT / path


def load_model_registry(path: str | Path) -> List[Dict[str, Any]]:
    path = resolve_repo_path(path)

    with open(path, "r", encoding="utf-8") as f:
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

    return matches[0]


def get_model_registry_path(config: Dict[str, Any]) -> str:
    return config.get("embedding_bank", {}).get(
        "model_registry_path",
        "./weights/args.json",
    )


def get_model_for(config: Dict[str, Any]) -> str:
    model_for = config.get("embedding_bank", {}).get("model_for")

    if not model_for:
        raise ValueError("embedding_bank.model_for is required.")

    return str(model_for)


def get_model_entry(config: Dict[str, Any]) -> Dict[str, Any]:
    registry_path = get_model_registry_path(config)
    registry = load_model_registry(registry_path)
    return find_model_entry(registry, get_model_for(config))


def get_task_type(config: Dict[str, Any]) -> str:
    entry = get_model_entry(config)
    run_config = entry.get("run_config", {})

    task_name = run_config.get("task_name")

    if task_name not in {"forecasting", "classification"}:
        raise ValueError(
            f"Invalid or missing task_name in model registry for "
            f"model_for='{get_model_for(config)}': {task_name}"
        )

    return str(task_name)


def get_registry_value(
    config: Dict[str, Any],
    key: str,
    default: Any = None,
) -> Any:
    entry = get_model_entry(config)

    model_config = entry.get("model_config", {})
    run_config = entry.get("run_config", {})

    if key in model_config:
        return model_config[key]

    if key in run_config:
        return run_config[key]

    return default


def path_str(path: str | Path) -> str:
    return str(path)


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}

    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]

    if isinstance(obj, tuple):
        return [to_jsonable(v) for v in obj]

    return obj


def add_optional_value_arg(
    cmd: List[str],
    name: str,
    value: Any,
) -> None:
    if value is not None:
        cmd.extend([f"--{name}", str(value)])


def add_optional_list_args(
    cmd: List[str],
    name: str,
    values: Optional[List[Any]],
) -> None:
    if values:
        cmd.append(f"--{name}")
        cmd.extend(str(v) for v in values)


def detector_suffix(detector_cfg: Dict[str, Any]) -> str:
    threshold = str(detector_cfg.get("threshold_quantile", 0.95)).replace(".", "")

    return (
        f"knn_{detector_cfg.get('k', 5)}_"
        f"{detector_cfg.get('metric', 'euclidean')}_"
        f"{detector_cfg.get('score_mode', 'mean')}_"
        f"{detector_cfg.get('normalization', 'standardize')}_"
        f"q{threshold}"
    )


def class_tag(prefix: str, values: Optional[List[Any]]) -> str:
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

    if task_type == "classification":
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

    raise ValueError(f"Unknown task_type: {task_type}")


class OODPipelineRunSaver:
    """
    Run manager for the full OOD/anomaly detection pipeline.

    Creates:
        run_dir/
            config.json
            resolved_config.json
            commands.txt
            status.json
            run_summary.json
            logs/
            embedding_banks/
            embedding_detectors/
            eval_sets/
            evaluation_reports/
            score_visualizations/
            datasets/
    """

    def __init__(self, config: Dict[str, Any], config_path: Path):
        self.config = config
        self.config_path = config_path

        self.task_type = get_task_type(config)
        self.model_for = get_model_for(config)

        default_names = build_default_names(config)

        run_cfg = config.get("run", {})
        output_root = Path(run_cfg.get("output_root", "./ood_runs"))
        output_root.mkdir(parents=True, exist_ok=True)

        self.run_name = run_cfg.get("run_name") or default_names["run_name"]
        self.timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.experiment_name = f"{self.model_for}/{self.run_name}/{self.timestamp}"

        self.run_dir = output_root / self.model_for / self.run_name / self.timestamp

        if self.run_dir.exists():
            raise FileExistsError(f"Run directory already exists: {self.run_dir}")

        self.logs_dir = self.run_dir / "logs"
        self.embedding_banks_dir = self.run_dir / "embedding_banks"
        self.embedding_detectors_dir = self.run_dir / "embedding_detectors"
        self.eval_sets_dir = self.run_dir / "eval_sets"
        self.evaluation_reports_dir = self.run_dir / "evaluation_reports"
        self.score_visualizations_dir = self.run_dir / "score_visualizations"
        self.datasets_dir = self.run_dir / "datasets"

        for path in [
            self.run_dir,
            self.logs_dir,
            self.embedding_banks_dir,
            self.embedding_detectors_dir,
            self.eval_sets_dir,
            self.evaluation_reports_dir,
            self.score_visualizations_dir,
            self.datasets_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)

        self.registry_path = output_root / "ood_run_registry.csv"

        shutil.copy2(config_path, self.run_dir / "config.json")
        self.save_json(config, self.run_dir / "resolved_config.json")

        self.commands_path = self.run_dir / "commands.txt"
        self.status_path = self.run_dir / "status.json"
        self.summary_path = self.run_dir / "run_summary.json"

        self.create_registry_entry()

    def save_json(self, data: Dict[str, Any], path: Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(to_jsonable(data), f, indent=2, ensure_ascii=False)

    def append_command(self, name: str, cmd: List[str]) -> None:
        with open(self.commands_path, "a", encoding="utf-8") as f:
            f.write(f"\n# {name}\n")
            f.write(shlex.join(cmd))
            f.write("\n")

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
            "message": "",
            "results": "",
        }

        with open(self.registry_path, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.registry_fieldnames())
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)

        self.update_status("running", message="Pipeline started.")

    def update_registry(
        self,
        status: str,
        message: str = "",
        results: Optional[Dict[str, Any]] = None,
    ) -> None:
        if results is None:
            results = {}

        rows = []
        found = False

        with open(self.registry_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)

            for row in reader:
                if row["experiment_name"] == self.experiment_name:
                    row["status"] = status
                    row["message"] = message
                    row["results"] = json.dumps(to_jsonable(results), ensure_ascii=False)
                    row["run_path"] = str(self.run_dir)
                    found = True

                rows.append(row)

        if not found:
            raise RuntimeError(f"Registry entry not found: {self.experiment_name}")

        with open(self.registry_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.registry_fieldnames())
            writer.writeheader()
            writer.writerows(rows)

    def update_status(
        self,
        status: str,
        message: str = "",
        results: Optional[Dict[str, Any]] = None,
    ) -> None:
        payload = {
            "experiment_name": self.experiment_name,
            "status": status,
            "message": message,
            "results": results or {},
            "run_dir": str(self.run_dir),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }

        self.save_json(payload, self.status_path)
        self.update_registry(status=status, message=message, results=results or {})


def run_command(
    saver: OODPipelineRunSaver,
    name: str,
    cmd: List[str],
    dry_run: bool = False,
) -> None:
    saver.append_command(name, cmd)

    log_path = saver.logs_dir / f"{name}.log"

    print("\n" + "=" * 80)
    print(f"[{name}]")
    print(shlex.join(cmd))
    print(f"log: {log_path}")
    print("=" * 80)

    if dry_run:
        return

    with open(log_path, "w", encoding="utf-8") as log_file:
        process = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )

    if process.returncode != 0:
        raise RuntimeError(
            f"Command failed: {name}. "
            f"Return code: {process.returncode}. "
            f"See log: {log_path}"
        )


def build_train_model_command(config: Dict[str, Any]) -> Optional[List[str]]:
    run_cfg = config.get("run", {})

    if not run_cfg.get("train_model", False):
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

    return cmd


def build_embedding_bank_command(
    config: Dict[str, Any],
    bank_split: str,
    output_name: str,
    output_dir: Path,
    override_data_path: Optional[str] = None,
    include_near_ood: bool = False,
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

    if override_data_path is not None:
        cmd.extend(["--override_data_path", str(override_data_path)])

    if task_type == "classification":
        add_optional_list_args(cmd, "id_classes", bank_cfg.get("id_classes"))

        if include_near_ood:
            add_optional_list_args(cmd, "near_ood_classes", bank_cfg.get("near_ood_classes"))
            add_optional_list_args(cmd, "far_ood_classes", bank_cfg.get("far_ood_classes"))

    return cmd


def build_forecasting_injection_command(
    config: Dict[str, Any],
    saver: OODPipelineRunSaver,
    names: Dict[str, str],
) -> Tuple[List[str], Path, Path]:
    injection_cfg = config.get("forecasting_csv_injection", {})

    input_csv_path = injection_cfg.get("input_csv_path")

    if not input_csv_path:
        raise ValueError(
            "forecasting_csv_injection.input_csv_path is required when use_injection=true."
        )

    output_csv_path = injection_cfg.get("output_csv_path")
    if output_csv_path is None:
        output_csv_path = saver.datasets_dir / names["injected_csv_name"]
    else:
        output_csv_path = Path(output_csv_path)

    output_mask_path = injection_cfg.get("output_mask_path")
    if output_mask_path is None:
        output_mask_path = saver.eval_sets_dir / names["mask_name"]
    else:
        output_mask_path = Path(output_mask_path)

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
        str(injection_cfg.get("anomaly_fraction", 0.05)),
        "--min_len",
        str(injection_cfg.get("min_len", 8)),
        "--max_len",
        str(injection_cfg.get("max_len", 32)),
        "--magnitude",
        str(injection_cfg.get("magnitude", 3.0)),
        "--channel_mode",
        str(injection_cfg.get("channel_mode", "random_one")),
        "--inject_start_ratio",
        str(injection_cfg.get("inject_start_ratio", 0.7)),
        "--inject_end_ratio",
        str(injection_cfg.get("inject_end_ratio", 1.0)),
        "--seed",
        str(injection_cfg.get("seed", 42)),
    ]

    anomaly_types = injection_cfg.get(
        "anomaly_types",
        ["spike", "level_shift", "noise", "trend", "flatline"],
    )

    if isinstance(anomaly_types, str):
        anomaly_types = [anomaly_types]

    cmd.append("--anomaly_types")
    cmd.extend(str(t) for t in anomaly_types)

    add_optional_value_arg(cmd, "source_csv_path", injection_cfg.get("source_csv_path"))

    return cmd, output_csv_path, output_mask_path


def build_embedding_detector_command(
    config: Dict[str, Any],
    saver: OODPipelineRunSaver,
    reference_bank_path: Path,
    query_bank_path: Path,
    output_name: str,
) -> List[str]:
    detector_cfg = config.get("embedding_detector", {})

    return [
        sys.executable,
        "ood_utils/embedding_detector.py",
        "--reference_bank",
        path_str(reference_bank_path),
        "--query_bank",
        path_str(query_bank_path),
        "--output_dir",
        path_str(saver.embedding_detectors_dir),
        "--output_name",
        output_name,
        "--k",
        str(detector_cfg.get("k", 5)),
        "--metric",
        str(detector_cfg.get("metric", "euclidean")),
        "--score_mode",
        str(detector_cfg.get("score_mode", "mean")),
        "--normalization",
        str(detector_cfg.get("normalization", "standardize")),
        "--threshold_quantile",
        str(detector_cfg.get("threshold_quantile", 0.95)),
    ]


def build_classification_labels_command(
    config: Dict[str, Any],
    test_bank_path: Path,
    labels_path: Path,
) -> List[str]:
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


def build_forecasting_labels_command(
    config: Dict[str, Any],
    mask_path: Path,
    test_bank_path: Path,
    labels_path: Path,
) -> List[str]:
    labels_cfg = config.get("forecasting_labels", {})

    seq_len = labels_cfg.get("seq_len")
    patch_len = labels_cfg.get("patch_len")
    stride = labels_cfg.get("stride")

    if seq_len is None:
        seq_len = get_registry_value(config, "seq_len")

    if patch_len is None:
        patch_len = get_registry_value(config, "patch_len")

    if stride is None:
        stride = get_registry_value(config, "stride")

    if seq_len is None or patch_len is None or stride is None:
        raise ValueError(
            "seq_len, patch_len and stride are required. "
            "They should be available from weights/args.json, "
            "or explicitly set in forecasting_labels."
        )

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

    return cmd


def build_evaluate_command(
    saver: OODPipelineRunSaver,
    scores_path: Path,
    labels_path: Path,
    output_name: str,
) -> List[str]:
    return [
        sys.executable,
        "ood_utils/evaluate_detector.py",
        "--scores_path",
        path_str(scores_path),
        "--labels_path",
        path_str(labels_path),
        "--output_dir",
        path_str(saver.evaluation_reports_dir),
        "--output_name",
        output_name,
    ]


def build_visualize_command(
    config: Dict[str, Any],
    saver: OODPipelineRunSaver,
    scores_path: Path,
    labels_path: Path,
    output_name: str,
) -> List[str]:
    viz_cfg = config.get("visualize_scores", {})

    cmd = [
        sys.executable,
        "ood_utils/visualize_scores.py",
        "--scores_path",
        path_str(scores_path),
        "--labels_path",
        path_str(labels_path),
        "--output_dir",
        path_str(saver.score_visualizations_dir),
        "--output_name",
        output_name,
        "--top_k",
        str(viz_cfg.get("top_k", 100)),
    ]

    sample_id = viz_cfg.get("sample_id")
    if sample_id is not None:
        cmd.extend(["--sample_id", str(sample_id)])

    return cmd


def read_metrics_summary(
    saver: OODPipelineRunSaver,
    output_name: str,
) -> Dict[str, Any]:
    metrics_path = saver.evaluation_reports_dir / f"{output_name}.metrics_summary.json"

    if not metrics_path.exists():
        return {}

    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_pipeline(
    config_path: Path,
    dry_run: bool = False,
    continue_on_error: bool = False,
) -> None:
    config = load_json(config_path)

    names = build_default_names(config)
    saver = OODPipelineRunSaver(config=config, config_path=config_path)

    task_type = get_task_type(config)
    model_for = get_model_for(config)

    train_bank_name = names["train_bank_name"]
    test_bank_name = names["test_bank_name"]
    output_name = names["output_name"]
    labels_name = names["labels_name"]

    train_bank_path = saver.embedding_banks_dir / f"{train_bank_name}.npz"
    test_bank_path = saver.embedding_banks_dir / f"{test_bank_name}.npz"
    scores_path = saver.embedding_detectors_dir / f"{output_name}.scores.npz"
    labels_path = saver.eval_sets_dir / labels_name

    steps: List[Tuple[str, List[str]]] = []

    train_model_cmd = build_train_model_command(config)
    if train_model_cmd is not None:
        steps.append(("00_train_model", train_model_cmd))

    steps.append(
        (
            "01_train_reference_bank",
            build_embedding_bank_command(
                config=config,
                bank_split="train",
                output_name=train_bank_name,
                output_dir=saver.embedding_banks_dir,
                include_near_ood=False,
            ),
        )
    )

    if task_type == "forecasting":
        injection_cfg = config.get("forecasting_csv_injection", {})
        use_injection = bool(injection_cfg.get("use_injection", False))

        if use_injection:
            injection_cmd, injected_csv_path, mask_path = build_forecasting_injection_command(
                config=config,
                saver=saver,
                names=names,
            )
            steps.append(("02_forecasting_csv_injection", injection_cmd))
            test_override_data_path = str(injected_csv_path)
        else:
            test_override_data_path = config.get("embedding_bank", {}).get("override_data_path")
            mask_path_raw = injection_cfg.get("output_mask_path")

            if test_override_data_path is None:
                raise ValueError(
                    "For forecasting without injection, set "
                    "embedding_bank.override_data_path to the evaluation CSV path."
                )

            if mask_path_raw is None:
                raise ValueError(
                    "For forecasting without injection, set "
                    "forecasting_csv_injection.output_mask_path to an existing mask file."
                )

            mask_path = Path(mask_path_raw)

        steps.append(
            (
                "03_test_embedding_bank",
                build_embedding_bank_command(
                    config=config,
                    bank_split="test",
                    output_name=test_bank_name,
                    output_dir=saver.embedding_banks_dir,
                    override_data_path=test_override_data_path,
                ),
            )
        )

        steps.append(
            (
                "04_embedding_detector",
                build_embedding_detector_command(
                    config=config,
                    saver=saver,
                    reference_bank_path=train_bank_path,
                    query_bank_path=test_bank_path,
                    output_name=output_name,
                ),
            )
        )

        steps.append(
            (
                "05_build_eval_labels",
                build_forecasting_labels_command(
                    config=config,
                    mask_path=Path(mask_path),
                    test_bank_path=test_bank_path,
                    labels_path=labels_path,
                ),
            )
        )

    elif task_type == "classification":
        steps.append(
            (
                "03_test_embedding_bank",
                build_embedding_bank_command(
                    config=config,
                    bank_split="test",
                    output_name=test_bank_name,
                    output_dir=saver.embedding_banks_dir,
                    include_near_ood=True,
                ),
            )
        )

        steps.append(
            (
                "04_embedding_detector",
                build_embedding_detector_command(
                    config=config,
                    saver=saver,
                    reference_bank_path=train_bank_path,
                    query_bank_path=test_bank_path,
                    output_name=output_name,
                ),
            )
        )

        steps.append(
            (
                "05_build_eval_labels",
                build_classification_labels_command(
                    config=config,
                    test_bank_path=test_bank_path,
                    labels_path=labels_path,
                ),
            )
        )

    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    steps.append(
        (
            "06_evaluate_detector",
            build_evaluate_command(
                saver=saver,
                scores_path=scores_path,
                labels_path=labels_path,
                output_name=output_name,
            ),
        )
    )

    steps.append(
        (
            "07_visualize_scores",
            build_visualize_command(
                config=config,
                saver=saver,
                scores_path=scores_path,
                labels_path=labels_path,
                output_name=output_name,
            ),
        )
    )

    summary: Dict[str, Any] = {
        "run_dir": str(saver.run_dir),
        "task_type": task_type,
        "model_for": model_for,
        "output_name": output_name,
        "paths": {
            "train_bank": str(train_bank_path),
            "test_bank": str(test_bank_path),
            "scores": str(scores_path),
            "labels": str(labels_path),
            "evaluation_reports": str(saver.evaluation_reports_dir),
            "score_visualizations": str(saver.score_visualizations_dir),
            "logs": str(saver.logs_dir),
        },
        "steps": [],
    }

    try:
        for name, cmd in steps:
            step_record: Dict[str, Any] = {
                "name": name,
                "command": shlex.join(cmd),
                "status": "pending",
            }

            try:
                run_command(saver, name, cmd, dry_run=dry_run)
                step_record["status"] = "skipped_dry_run" if dry_run else "finished"

            except Exception as exc:
                step_record["status"] = "failed"
                step_record["error"] = str(exc)
                summary["steps"].append(step_record)
                saver.save_json(summary, saver.summary_path)
                saver.update_status("failed", message=str(exc), results=summary)

                if continue_on_error:
                    print(f"Step failed but continue_on_error=True: {name}")
                    continue

                raise

            summary["steps"].append(step_record)
            saver.save_json(summary, saver.summary_path)

        if not dry_run:
            summary["metrics_summary"] = read_metrics_summary(saver, output_name)

        saver.save_json(summary, saver.summary_path)
        saver.update_status(
            "dry_run" if dry_run else "finished",
            message="Dry run finished." if dry_run else "Pipeline finished.",
            results=summary,
        )

        print("\nPipeline finished.")
        print(f"Run directory: {saver.run_dir}")
        print(f"Commands: {saver.commands_path}")
        print(f"Summary: {saver.summary_path}")

    except Exception as exc:
        error_text = traceback.format_exc()
        error_path = saver.run_dir / "error.txt"

        with open(error_path, "w", encoding="utf-8") as f:
            f.write(error_text)

        saver.update_status("failed", message=str(exc), results=summary)

        print("\nPipeline failed.")
        print(f"Run directory: {saver.run_dir}")
        print(f"Error log: {error_path}")

        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the full TimeDRL OOD/anomaly detection pipeline from a JSON config."
    )

    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dry_run", action="store_true", default=False)
    parser.add_argument("--continue_on_error", action="store_true", default=False)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    run_pipeline(
        config_path=Path(args.config),
        dry_run=args.dry_run,
        continue_on_error=args.continue_on_error,
    )


if __name__ == "__main__":
    main()