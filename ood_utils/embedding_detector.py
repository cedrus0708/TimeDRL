import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from tqdm import tqdm


class KNNEmbeddingDetector:
    """
    kNN distance based embedding anomaly detector.

    The detector stores a reference embedding bank built from normal / ID train data.
    A new embedding is anomalous if it is far from its k nearest reference embeddings.

    Typical score:
        score(x) = mean distance to k nearest train embeddings

    Supported:
        - euclidean distance
        - squared_euclidean distance
        - cosine distance
        - optional standardization / L2 normalization
        - quantile based threshold calibration
    """

    def __init__(
        self,
        k: int = 5,
        metric: str = "euclidean",
        score_mode: str = "mean",
        normalization: str = "standardize",
        threshold_quantile: float = 0.95,
        batch_size: int = 1024,
        eps: float = 1e-12,
    ):
        if k < 1:
            raise ValueError("k must be >= 1.")

        if metric not in {"euclidean", "squared_euclidean", "cosine"}:
            raise ValueError(
                "metric must be one of: 'euclidean', 'squared_euclidean', 'cosine'."
            )

        if score_mode not in {"mean", "median", "kth"}:
            raise ValueError("score_mode must be one of: 'mean', 'median', 'kth'.")

        if normalization not in {"none", "standardize", "l2", "standardize_l2"}:
            raise ValueError(
                "normalization must be one of: "
                "'none', 'standardize', 'l2', 'standardize_l2'."
            )

        if not 0.0 < threshold_quantile < 1.0:
            raise ValueError("threshold_quantile must be between 0 and 1.")

        self.k = int(k)
        self.metric = metric
        self.score_mode = score_mode
        self.normalization = normalization
        self.threshold_quantile = float(threshold_quantile)
        self.batch_size = int(batch_size)
        self.eps = float(eps)

        self.reference_embeddings_: Optional[np.ndarray] = None
        self.reference_for_distance_: Optional[np.ndarray] = None

        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

        self.threshold_: Optional[float] = None
        self.calibration_scores_: Optional[np.ndarray] = None

    @staticmethod
    def _ensure_2d(x: np.ndarray) -> np.ndarray:
        """
        Convert embeddings to [N, D].

        Accepts:
            [D]
            [N, D]
            [B, T, D]
            [B, C, T, D]
        """
        x = np.asarray(x, dtype=np.float32)

        if x.ndim == 1:
            return x.reshape(1, -1)

        if x.ndim == 2:
            return x

        if x.ndim > 2:
            return x.reshape(-1, x.shape[-1])

        raise ValueError(f"Unsupported embedding shape: {x.shape}")

    def _fit_normalizer(self, reference_embeddings: np.ndarray) -> None:
        if self.normalization in {"standardize", "standardize_l2"}:
            self.mean_ = reference_embeddings.mean(axis=0, keepdims=True)
            self.scale_ = reference_embeddings.std(axis=0, keepdims=True)
            self.scale_ = np.maximum(self.scale_, self.eps)
        else:
            self.mean_ = None
            self.scale_ = None

    def _apply_normalizer(self, embeddings: np.ndarray) -> np.ndarray:
        x = self._ensure_2d(embeddings).astype(np.float32)

        if not np.isfinite(x).all():
            raise ValueError("Embeddings contain NaN or Inf values.")

        if self.normalization in {"standardize", "standardize_l2"}:
            if self.mean_ is None or self.scale_ is None:
                raise RuntimeError("Detector normalizer is not fitted.")
            x = (x - self.mean_) / self.scale_

        if self.normalization in {"l2", "standardize_l2"}:
            x = self._l2_normalize(x)

        return x.astype(np.float32)

    def _metric_ready(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Prepare embeddings for the selected distance metric.

        Cosine distance needs L2-normalized vectors.
        """
        if self.metric == "cosine":
            return self._l2_normalize(embeddings)
        return embeddings.astype(np.float32)

    def _l2_normalize(self, x: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        return x / np.maximum(norm, self.eps)

    def fit(
        self,
        reference_embeddings: np.ndarray,
        calibration_embeddings: Optional[np.ndarray] = None,
    ) -> "KNNEmbeddingDetector":
        """
        Fit detector on normal / ID reference embeddings.

        If calibration_embeddings is None, threshold is calibrated on the reference
        embeddings using leave-one-out scoring. This means that each reference point
        is scored against the reference bank while excluding itself.

        If calibration_embeddings is given, threshold is calibrated on those embeddings
        without leave-one-out exclusion.
        """
        reference_embeddings = self._ensure_2d(reference_embeddings)

        if reference_embeddings.shape[0] < 2:
            raise ValueError("At least 2 reference embeddings are needed.")

        self._fit_normalizer(reference_embeddings)

        normalized_reference = self._apply_normalizer(reference_embeddings)

        self.reference_embeddings_ = normalized_reference
        self.reference_for_distance_ = self._metric_ready(normalized_reference)

        if calibration_embeddings is None:
            if self.reference_embeddings_.shape[0] <= self.k:
                raise ValueError(
                    "Reference bank is too small for leave-one-out calibration. "
                    f"Need more than k={self.k} reference embeddings."
                )

            calibration_scores = self._score_normalized(
                query_for_distance=self.reference_for_distance_,
                exclude_self=True,
                return_neighbors=False,
            )[0]
        else:
            calibration_scores = self.score(
                calibration_embeddings,
                return_neighbors=False,
            )

        self.calibration_scores_ = calibration_scores.astype(np.float32)
        self.threshold_ = float(
            np.quantile(self.calibration_scores_, self.threshold_quantile)
        )

        return self

    def score(
        self,
        query_embeddings: np.ndarray,
        return_neighbors: bool = False,
    ):
        """
        Score new embeddings.

        Returns:
            scores if return_neighbors=False

            or

            scores, neighbor_indices, neighbor_distances if return_neighbors=True
        """
        self._check_is_fitted()

        query_embeddings = self._apply_normalizer(query_embeddings)
        query_for_distance = self._metric_ready(query_embeddings)

        scores, neighbor_indices, neighbor_distances = self._score_normalized(
            query_for_distance=query_for_distance,
            exclude_self=False,
            return_neighbors=return_neighbors,
        )

        if return_neighbors:
            return scores, neighbor_indices, neighbor_distances

        return scores

    def _score_normalized(
        self,
        query_for_distance: np.ndarray,
        exclude_self: bool,
        return_neighbors: bool,
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        self._check_is_fitted()

        reference = self.reference_for_distance_
        assert reference is not None

        n_query = query_for_distance.shape[0]
        n_reference = reference.shape[0]

        max_available_neighbors = n_reference - 1 if exclude_self else n_reference
        if self.k > max_available_neighbors:
            raise ValueError(
                f"k={self.k} is too large for n_reference={n_reference} "
                f"with exclude_self={exclude_self}."
            )

        all_scores = []
        all_neighbor_indices = []
        all_neighbor_distances = []

        for start in tqdm(
            range(0, n_query, self.batch_size),
            desc="Scoring embeddings",
            leave=False,
        ):
            end = min(start + self.batch_size, n_query)
            query_chunk = query_for_distance[start:end]

            distances = self._pairwise_distances(query_chunk, reference)

            if exclude_self:
                row_indices = np.arange(end - start)
                col_indices = np.arange(start, end)

                valid_mask = col_indices < n_reference
                distances[row_indices[valid_mask], col_indices[valid_mask]] = np.inf

            neighbor_indices, neighbor_distances = self._topk_smallest(distances)

            scores = self._aggregate_neighbor_distances(neighbor_distances)

            all_scores.append(scores.astype(np.float32))

            if return_neighbors:
                all_neighbor_indices.append(neighbor_indices.astype(np.int64))
                all_neighbor_distances.append(neighbor_distances.astype(np.float32))

        scores = np.concatenate(all_scores, axis=0)

        if return_neighbors:
            return (
                scores,
                np.concatenate(all_neighbor_indices, axis=0),
                np.concatenate(all_neighbor_distances, axis=0),
            )

        return scores, None, None

    def _pairwise_distances(
        self,
        query: np.ndarray,
        reference: np.ndarray,
    ) -> np.ndarray:
        if self.metric in {"euclidean", "squared_euclidean"}:
            query_norm = np.sum(query * query, axis=1, keepdims=True)
            reference_norm = np.sum(reference * reference, axis=1, keepdims=True).T

            dist_sq = query_norm + reference_norm - 2.0 * np.matmul(query, reference.T)
            dist_sq = np.maximum(dist_sq, 0.0)

            if self.metric == "squared_euclidean":
                return dist_sq.astype(np.float32)

            return np.sqrt(dist_sq).astype(np.float32)

        if self.metric == "cosine":
            similarity = np.matmul(query, reference.T)
            distance = 1.0 - similarity
            return np.clip(distance, 0.0, 2.0).astype(np.float32)

        raise ValueError(f"Unknown metric: {self.metric}")

    def _topk_smallest(self, distances: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        kth_index = self.k - 1

        unsorted_indices = np.argpartition(distances, kth=kth_index, axis=1)[
            :, : self.k
        ]
        unsorted_distances = np.take_along_axis(
            distances,
            unsorted_indices,
            axis=1,
        )

        order = np.argsort(unsorted_distances, axis=1)

        neighbor_indices = np.take_along_axis(unsorted_indices, order, axis=1)
        neighbor_distances = np.take_along_axis(unsorted_distances, order, axis=1)

        return neighbor_indices, neighbor_distances

    def _aggregate_neighbor_distances(self, neighbor_distances: np.ndarray) -> np.ndarray:
        if self.score_mode == "mean":
            return neighbor_distances.mean(axis=1)

        if self.score_mode == "median":
            return np.median(neighbor_distances, axis=1)

        if self.score_mode == "kth":
            return neighbor_distances[:, -1]

        raise ValueError(f"Unknown score_mode: {self.score_mode}")

    def predict_from_scores(
        self,
        scores: np.ndarray,
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        """
        Convert anomaly scores to binary anomaly predictions.

        Returns:
            0 = normal / ID
            1 = anomaly / OOD
        """
        if threshold is None:
            if self.threshold_ is None:
                raise RuntimeError("No threshold is available. Fit the detector first.")
            threshold = self.threshold_

        scores = np.asarray(scores, dtype=np.float32)
        return (scores > threshold).astype(np.int64)

    def predict(
        self,
        query_embeddings: np.ndarray,
        threshold: Optional[float] = None,
    ) -> np.ndarray:
        scores = self.score(query_embeddings)
        return self.predict_from_scores(scores, threshold=threshold)

    def _check_is_fitted(self) -> None:
        if self.reference_embeddings_ is None or self.reference_for_distance_ is None:
            raise RuntimeError("Detector is not fitted yet.")

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "k": self.k,
            "metric": self.metric,
            "score_mode": self.score_mode,
            "normalization": self.normalization,
            "threshold_quantile": self.threshold_quantile,
            "batch_size": self.batch_size,
            "eps": self.eps,
            "threshold": self.threshold_,
            "reference_shape": (
                None
                if self.reference_embeddings_ is None
                else list(self.reference_embeddings_.shape)
            ),
            "calibration_scores_shape": (
                None
                if self.calibration_scores_ is None
                else list(self.calibration_scores_.shape)
            ),
        }

    def save(self, path: str | Path) -> None:
        """
        Save detector to .npz.

        The reference embeddings are saved too, because kNN needs the reference bank
        at inference time.
        """
        self._check_is_fitted()

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            path,
            reference_embeddings=self.reference_embeddings_,
            mean_=np.array([] if self.mean_ is None else self.mean_, dtype=np.float32),
            scale_=np.array([] if self.scale_ is None else self.scale_, dtype=np.float32),
            threshold_=np.array(
                [np.nan if self.threshold_ is None else self.threshold_],
                dtype=np.float32,
            ),
            calibration_scores_=np.array(
                []
                if self.calibration_scores_ is None
                else self.calibration_scores_,
                dtype=np.float32,
            ),
            metadata_json=json.dumps(self.to_metadata(), ensure_ascii=False),
        )

    @classmethod
    def load(cls, path: str | Path) -> "KNNEmbeddingDetector":
        path = Path(path)

        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata_json"].item()))

            detector = cls(
                k=int(metadata["k"]),
                metric=metadata["metric"],
                score_mode=metadata["score_mode"],
                normalization=metadata["normalization"],
                threshold_quantile=float(metadata["threshold_quantile"]),
                batch_size=int(metadata["batch_size"]),
                eps=float(metadata["eps"]),
            )

            detector.reference_embeddings_ = data["reference_embeddings"].astype(
                np.float32
            )
            detector.reference_for_distance_ = detector._metric_ready(
                detector.reference_embeddings_
            )

            mean = data["mean_"]
            scale = data["scale_"]

            detector.mean_ = None if mean.size == 0 else mean.astype(np.float32)
            detector.scale_ = None if scale.size == 0 else scale.astype(np.float32)

            threshold = float(data["threshold_"][0])
            detector.threshold_ = None if np.isnan(threshold) else threshold

            calibration_scores = data["calibration_scores_"]
            detector.calibration_scores_ = (
                None
                if calibration_scores.size == 0
                else calibration_scores.astype(np.float32)
            )

        return detector


def load_embedding_bank(path: str | Path) -> Dict[str, np.ndarray]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Embedding bank not found: {path}")

    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}

def subsample_embeddings(
    embeddings: np.ndarray,
    max_vectors: int | None,
    random_seed: int = 2023,
) -> np.ndarray:
    """
    Randomly subsample embedding vectors for faster kNN fitting.

    This is especially useful for timestamp embeddings, where the full bank
    can be extremely large.
    """
    embeddings = np.asarray(embeddings)

    if max_vectors is None:
        return embeddings

    if max_vectors <= 0:
        raise ValueError("max_vectors must be positive or None.")

    n = embeddings.shape[0]

    if n <= max_vectors:
        return embeddings

    rng = np.random.default_rng(random_seed)
    selected = rng.choice(n, size=max_vectors, replace=False)

    return embeddings[selected]

def fit_detectors_from_bank(
    bank: Dict[str, np.ndarray],
    k: int = 5,
    metric: str = "euclidean",
    score_mode: str = "mean",
    normalization: str = "standardize",
    threshold_quantile: float = 0.95,
    batch_size: int = 1024,
    max_instance_reference_vectors: int | None = None,
    max_timestamp_reference_vectors: int | None = 50000,
    random_seed: int = 2023,
) -> Dict[str, KNNEmbeddingDetector]:
    """
    Fit one detector for instance embeddings and one detector for timestamp embeddings.
    """
    required_keys = ["instance_embeddings", "timestamp_embeddings"]
    for key in required_keys:
        if key not in bank:
            raise KeyError(f"Missing key from embedding bank: {key}")

    instance_reference = subsample_embeddings(
        embeddings=bank["instance_embeddings"],
        max_vectors=max_instance_reference_vectors,
        random_seed=random_seed,
    )

    timestamp_reference = subsample_embeddings(
        embeddings=bank["timestamp_embeddings"],
        max_vectors=max_timestamp_reference_vectors,
        random_seed=random_seed,
    )

    print("\nReference sizes used for kNN:")
    print(f"  instance reference:  {instance_reference.shape}")
    print(f"  timestamp reference: {timestamp_reference.shape}")

    instance_detector = KNNEmbeddingDetector(
        k=k,
        metric=metric,
        score_mode=score_mode,
        normalization=normalization,
        threshold_quantile=threshold_quantile,
        batch_size=batch_size,
    ).fit(instance_reference)

    timestamp_detector = KNNEmbeddingDetector(
        k=k,
        metric=metric,
        score_mode=score_mode,
        normalization=normalization,
        threshold_quantile=threshold_quantile,
        batch_size=batch_size,
    ).fit(timestamp_reference)

    return {
        "instance": instance_detector,
        "timestamp": timestamp_detector,
    }


def score_bank(
    detectors: Dict[str, KNNEmbeddingDetector],
    query_bank: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    Score an embedding bank with already fitted instance/timestamp detectors.
    """
    instance_scores = detectors["instance"].score(query_bank["instance_embeddings"])
    timestamp_scores = detectors["timestamp"].score(query_bank["timestamp_embeddings"])

    instance_predictions = detectors["instance"].predict_from_scores(instance_scores)
    timestamp_predictions = detectors["timestamp"].predict_from_scores(timestamp_scores)

    result = {
        "instance_scores": instance_scores.astype(np.float32),
        "timestamp_scores": timestamp_scores.astype(np.float32),
        "instance_predictions": instance_predictions.astype(np.int64),
        "timestamp_predictions": timestamp_predictions.astype(np.int64),
        "instance_threshold": np.array(
            [detectors["instance"].threshold_],
            dtype=np.float32,
        ),
        "timestamp_threshold": np.array(
            [detectors["timestamp"].threshold_],
            dtype=np.float32,
        ),
    }

    mapping_keys = [
        "instance_sample_index",
        "instance_channel_index",
        "timestamp_sample_index",
        "timestamp_channel_index",
        "timestamp_patch_index",
        "sample_labels",
    ]

    for key in mapping_keys:
        if key in query_bank:
            result[key] = query_bank[key]

    if "instance_sample_index" in query_bank:
        result["sample_instance_score_max"] = aggregate_scores_by_index(
            scores=instance_scores,
            index=query_bank["instance_sample_index"],
            mode="max",
        )
        result["sample_instance_score_mean"] = aggregate_scores_by_index(
            scores=instance_scores,
            index=query_bank["instance_sample_index"],
            mode="mean",
        )

    if "timestamp_sample_index" in query_bank:
        result["sample_timestamp_score_max"] = aggregate_scores_by_index(
            scores=timestamp_scores,
            index=query_bank["timestamp_sample_index"],
            mode="max",
        )
        result["sample_timestamp_score_mean"] = aggregate_scores_by_index(
            scores=timestamp_scores,
            index=query_bank["timestamp_sample_index"],
            mode="mean",
        )

    return result


def aggregate_scores_by_index(
    scores: np.ndarray,
    index: np.ndarray,
    mode: str = "max",
) -> np.ndarray:
    """
    Aggregate flattened vector scores back to sample-level scores.

    Useful when:
        - enable_channel_independence=True
        - timestamp embeddings are flattened as sample x channel x patch

    Supported modes:
        - max
        - mean
        - median
    """
    scores = np.asarray(scores, dtype=np.float32)
    index = np.asarray(index, dtype=np.int64)

    if scores.shape[0] != index.shape[0]:
        raise ValueError(
            f"scores and index length mismatch: {scores.shape[0]} != {index.shape[0]}"
        )

    if index.size == 0:
        return np.array([], dtype=np.float32)

    n_groups = int(index.max()) + 1

    if mode == "max":
        output = np.full(n_groups, -np.inf, dtype=np.float32)
        np.maximum.at(output, index, scores)
        return output

    if mode == "mean":
        output = np.zeros(n_groups, dtype=np.float32)
        counts = np.zeros(n_groups, dtype=np.float32)

        np.add.at(output, index, scores)
        np.add.at(counts, index, 1.0)

        return output / np.maximum(counts, 1.0)

    if mode == "median":
        output = np.zeros(n_groups, dtype=np.float32)
        for group_id in range(n_groups):
            group_scores = scores[index == group_id]
            if group_scores.size > 0:
                output[group_id] = np.median(group_scores)
        return output

    raise ValueError("mode must be one of: 'max', 'mean', 'median'.")


def save_score_result(
    result: Dict[str, np.ndarray],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **result)


def save_detector_metadata(
    detectors: Dict[str, KNNEmbeddingDetector],
    output_path: str | Path,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "instance_detector": detectors["instance"].to_metadata(),
        "timestamp_detector": detectors["timestamp"].to_metadata(),
    }

    if extra_meta is not None:
        metadata["extra"] = extra_meta

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit and apply kNN embedding anomaly detectors."
    )

    parser.add_argument(
        "--reference_bank",
        type=str,
        required=True,
        help="Path to the train/reference embedding bank .npz file.",
    )
    parser.add_argument(
        "--query_bank",
        type=str,
        default=None,
        help=(
            "Optional path to another embedding bank .npz file to score. "
            "If omitted, only detectors are fitted and saved."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./ood/embedding_detectors",
        help="Output directory.",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="Optional output file stem.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of nearest neighbors.",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="euclidean",
        choices=["euclidean", "squared_euclidean", "cosine"],
        help="Distance metric.",
    )
    parser.add_argument(
        "--score_mode",
        type=str,
        default="mean",
        choices=["mean", "median", "kth"],
        help="How to aggregate k nearest neighbor distances into one anomaly score.",
    )
    parser.add_argument(
        "--normalization",
        type=str,
        default="standardize",
        choices=["none", "standardize", "l2", "standardize_l2"],
        help="Embedding normalization before kNN distance calculation.",
    )
    parser.add_argument(
        "--threshold_quantile",
        type=float,
        default=0.95,
        help="Quantile used to calibrate anomaly threshold from reference scores.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Scoring batch size. Lower this if memory usage is too high.",
    )
    parser.add_argument(
        "--max_instance_reference_vectors",
        type=int,
        default=None,
        help="Optional cap for instance reference vectors used by kNN.",
    )

    parser.add_argument(
        "--max_timestamp_reference_vectors",
        type=int,
        default=50000,
        help=(
            "Optional cap for timestamp reference vectors used by kNN. "
            "Default: 50000, because full timestamp banks can be very large."
        ),
    )

    parser.add_argument(
        "--random_seed",
        type=int,
        default=2023,
        help="Random seed for reference subsampling.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    reference_bank_path = Path(args.reference_bank)
    output_dir = Path(args.output_dir)

    output_name = args.output_name
    if output_name is None:
        output_name = reference_bank_path.stem.replace(".npz", "")

    print(f"Loading reference bank: {reference_bank_path}")
    reference_bank = load_embedding_bank(reference_bank_path)

    print("Fitting instance and timestamp kNN detectors...")
    detectors = fit_detectors_from_bank(
        bank=reference_bank,
        k=args.k,
        metric=args.metric,
        score_mode=args.score_mode,
        normalization=args.normalization,
        threshold_quantile=args.threshold_quantile,
        batch_size=args.batch_size,
        max_instance_reference_vectors=args.max_instance_reference_vectors,
        max_timestamp_reference_vectors=args.max_timestamp_reference_vectors,
        random_seed=args.random_seed
    )

    instance_detector_path = output_dir / f"{output_name}.instance_detector.npz"
    timestamp_detector_path = output_dir / f"{output_name}.timestamp_detector.npz"
    metadata_path = output_dir / f"{output_name}.detector_meta.json"

    detectors["instance"].save(instance_detector_path)
    detectors["timestamp"].save(timestamp_detector_path)

    save_detector_metadata(
        detectors=detectors,
        output_path=metadata_path,
        extra_meta={
            "reference_bank": str(reference_bank_path.resolve()),
        },
    )

    print("\nDetectors saved:")
    print(f"  instance detector:  {instance_detector_path}")
    print(f"  timestamp detector: {timestamp_detector_path}")
    print(f"  metadata:           {metadata_path}")

    print("\nThresholds:")
    print(f"  instance threshold:  {detectors['instance'].threshold_:.6f}")
    print(f"  timestamp threshold: {detectors['timestamp'].threshold_:.6f}")

    if args.query_bank is not None:
        query_bank_path = Path(args.query_bank)

        print(f"\nLoading query bank: {query_bank_path}")
        query_bank = load_embedding_bank(query_bank_path)

        print("Scoring query bank...")
        result = score_bank(
            detectors=detectors,
            query_bank=query_bank,
        )

        score_output_path = output_dir / f"{output_name}.scores.npz"
        save_score_result(result, score_output_path)

        print(f"\nScores saved: {score_output_path}")

        instance_rate = result["instance_predictions"].mean()
        timestamp_rate = result["timestamp_predictions"].mean()

        print("\nAnomaly rates:")
        print(f"  instance anomaly rate:  {instance_rate:.4f}")
        print(f"  timestamp anomaly rate: {timestamp_rate:.4f}")


if __name__ == "__main__":
    main()