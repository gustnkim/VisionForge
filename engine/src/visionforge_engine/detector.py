from __future__ import annotations

import json
import math
import os
import random
import time
from collections.abc import Iterable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageOps

from .models import (
    BoundingBox,
    Detection,
    InferenceResult,
    TrainingMetrics,
    TrainingResult,
    WarningItem,
    resolved_path,
)
from .quality import sha256_file

ENGINE_NAME = "visionforge-linear-feature-detector-v2"
FEATURE_VERSION = 2
FEATURE_IMAGE_SIZE = 32
HISTOGRAM_BINS = 8
ORIENTATION_BINS = 9


def _project_root(dataset_path: Path) -> Path:
    for parent in dataset_path.parents:
        if (parent / "project.json").exists():
            return parent
    raise ValueError("dataset manifest is not inside a VisionForge project")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected: {path}")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    temporary.replace(path)


def _feature_vector(image: Image.Image, feature_version: int = FEATURE_VERSION) -> np.ndarray:
    resized = image.convert("RGB").resize(
        (FEATURE_IMAGE_SIZE, FEATURE_IMAGE_SIZE),
        Image.Resampling.BILINEAR,
    )
    pixels = np.asarray(resized, dtype=np.float32) / 255.0
    features: list[np.ndarray] = []

    for channel in range(3):
        histogram, _ = np.histogram(
            pixels[:, :, channel],
            bins=HISTOGRAM_BINS,
            range=(0.0, 1.0),
        )
        features.append(histogram.astype(np.float32) / pixels[:, :, channel].size)

    gray = pixels.mean(axis=2)
    gradient_x = np.zeros_like(gray)
    gradient_y = np.zeros_like(gray)
    gradient_x[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gradient_y[1:-1, :] = gray[2:, :] - gray[:-2, :]
    magnitude = np.hypot(gradient_x, gradient_y)
    orientation = (np.arctan2(gradient_y, gradient_x) + math.pi) % math.pi
    orientation_histogram, _ = np.histogram(
        orientation,
        bins=ORIENTATION_BINS,
        range=(0.0, math.pi),
        weights=magnitude,
    )
    orientation_histogram = orientation_histogram.astype(np.float32)
    orientation_histogram /= max(float(orientation_histogram.sum()), 1e-6)
    features.append(orientation_histogram)

    channel_stats = np.concatenate((pixels.mean(axis=(0, 1)), pixels.std(axis=(0, 1))))
    features.append(channel_stats.astype(np.float32))
    spatial = []
    for row in range(2):
        for column in range(2):
            block = pixels[
                row * FEATURE_IMAGE_SIZE // 2 : (row + 1) * FEATURE_IMAGE_SIZE // 2,
                column * FEATURE_IMAGE_SIZE // 2 : (column + 1) * FEATURE_IMAGE_SIZE // 2,
            ]
            spatial.extend(block.mean(axis=(0, 1)).tolist())
    features.append(np.asarray(spatial, dtype=np.float32))
    features.append(
        np.asarray(
            [
                float(magnitude.mean()),
                float(magnitude.std()),
                float((magnitude > 0.12).mean()),
            ],
            dtype=np.float32,
        )
    )
    if feature_version >= 2:
        normalized_gray = (gray - float(gray.mean())) / max(float(gray.std()), 1e-4)
        pooled_gray = normalized_gray.reshape(8, 4, 8, 4).mean(axis=(1, 3))
        features.append(pooled_gray.astype(np.float32).ravel())

        local_orientations = []
        for row in range(4):
            for column in range(4):
                row_start = row * FEATURE_IMAGE_SIZE // 4
                row_end = (row + 1) * FEATURE_IMAGE_SIZE // 4
                column_start = column * FEATURE_IMAGE_SIZE // 4
                column_end = (column + 1) * FEATURE_IMAGE_SIZE // 4
                cell_orientation = orientation[row_start:row_end, column_start:column_end]
                cell_magnitude = magnitude[row_start:row_end, column_start:column_end]
                histogram, _ = np.histogram(
                    cell_orientation,
                    bins=ORIENTATION_BINS,
                    range=(0.0, math.pi),
                    weights=cell_magnitude,
                )
                histogram = histogram.astype(np.float32)
                histogram /= max(float(np.linalg.norm(histogram)), 1e-6)
                local_orientations.append(histogram)
        features.append(np.concatenate(local_orientations))
    return np.concatenate(features)


def _bounded_box(annotation: dict[str, Any], image: Image.Image) -> tuple[int, int, int, int]:
    left = max(0, int(annotation["xMin"]))
    top = max(0, int(annotation["yMin"]))
    right = min(image.width, int(annotation["xMax"]))
    bottom = min(image.height, int(annotation["yMax"]))
    if right <= left or bottom <= top:
        raise ValueError("annotation has an empty bounding box")
    return left, top, right, bottom


def _positive_variants(patch: Image.Image) -> Iterable[Image.Image]:
    yield patch
    yield ImageOps.mirror(patch)
    yield ImageEnhance.Brightness(patch).enhance(0.84)
    yield ImageEnhance.Brightness(patch).enhance(1.16)


def _negative_windows(
    image: Image.Image,
    randomizer: random.Random,
    aspect_ratios: list[float],
    count: int = 10,
) -> Iterable[Image.Image]:
    for _ in range(count):
        relative_height = randomizer.uniform(0.16, 0.62)
        height = max(12, min(image.height, round(image.height * relative_height)))
        aspect = randomizer.choice(aspect_ratios) if aspect_ratios else 1.0
        width = max(12, min(image.width, round(height * aspect)))
        left = randomizer.randint(0, max(0, image.width - width))
        top = randomizer.randint(0, max(0, image.height - height))
        yield image.crop((left, top, left + width, top + height))


def _corner_negatives(
    image: Image.Image,
    positive_box: tuple[int, int, int, int],
) -> Iterable[Image.Image]:
    left, top, right, bottom = positive_box
    candidates = [
        (0, 0, left, image.height),
        (right, 0, image.width, image.height),
        (left, 0, right, top),
        (left, bottom, right, image.height),
    ]
    for box in candidates:
        if box[2] - box[0] >= 12 and box[3] - box[1] >= 12:
            yield image.crop(box)


def _score(feature: np.ndarray, model: dict[str, Any]) -> float:
    mean = np.asarray(model["featureMean"], dtype=np.float32)
    scale = np.asarray(model["featureScale"], dtype=np.float32)
    weights = np.asarray(model["weights"], dtype=np.float32)
    normalized = (feature - mean) / scale
    return float(normalized @ weights)


def _confidence(score: float, model: dict[str, Any]) -> float:
    threshold = float(model["scoreThreshold"])
    scale = max(float(model["scoreScale"]), 1e-3)
    exponent = max(-30.0, min(30.0, -(score - threshold) / scale))
    return 1.0 / (1.0 + math.exp(exponent))


def _iou(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    intersection_width = max(0, min(left[2], right[2]) - max(left[0], right[0]))
    intersection_height = max(0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = intersection_width * intersection_height
    if intersection == 0:
        return 0.0
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    return intersection / max(left_area + right_area - intersection, 1)


def _candidate_windows(
    image: Image.Image,
    model: dict[str, Any],
) -> Iterable[tuple[int, int, int, int]]:
    ratios = [float(value) for value in model.get("aspectRatios", [1.0])]
    heights = [float(value) for value in model.get("relativeHeights", [0.25, 0.4])]
    scales = {0.12, 0.18, 0.26, 0.36, 0.5, 0.68}
    for height in heights:
        scales.update((height * 0.75, height, height * 1.25))

    for relative_height in sorted(value for value in scales if 0.08 <= value <= 0.82):
        window_height = max(12, min(image.height, round(image.height * relative_height)))
        for aspect in ratios:
            window_width = max(12, min(image.width, round(window_height * aspect)))
            stride = max(6, min(window_width, window_height) // 4)
            max_left = max(0, image.width - window_width)
            max_top = max(0, image.height - window_height)
            left_positions = list(range(0, max_left + 1, stride))
            top_positions = list(range(0, max_top + 1, stride))
            if not left_positions or left_positions[-1] != max_left:
                left_positions.append(max_left)
            if not top_positions or top_positions[-1] != max_top:
                top_positions.append(max_top)
            for top in top_positions:
                for left in left_positions:
                    yield left, top, left + window_width, top + window_height


def _detect_with_max_confidence(
    image: Image.Image,
    model: dict[str, Any],
    threshold: float,
) -> tuple[list[Detection], float]:
    candidates: list[tuple[float, float, tuple[int, int, int, int]]] = []
    max_confidence = 0.0
    feature_version = int(model.get("featureVersion", 1))
    for box in _candidate_windows(image, model):
        patch = image.crop(box)
        score = _score(_feature_vector(patch, feature_version), model)
        confidence = _confidence(score, model)
        max_confidence = max(max_confidence, confidence)
        if confidence >= threshold:
            candidates.append((confidence, score, box))

    candidates.sort(key=lambda value: (value[0], value[1]), reverse=True)
    selected: list[tuple[float, tuple[int, int, int, int]]] = []
    for confidence, _, box in candidates:
        if all(_iou(box, kept_box) < 0.32 for _, kept_box in selected):
            selected.append((confidence, box))
        if len(selected) >= int(model.get("maxDetections", 20)):
            break

    detections = [
        Detection(
            class_id=str(model["classId"]),
            class_name=str(model["className"]),
            confidence=round(confidence, 6),
            bounding_box=BoundingBox(*box),
        )
        for confidence, box in selected
    ]
    return detections, max_confidence


def _detect(image: Image.Image, model: dict[str, Any], threshold: float) -> list[Detection]:
    detections, _ = _detect_with_max_confidence(image, model, threshold)
    return detections


def _evaluate(
    dataset: dict[str, Any],
    project_root: Path,
    model: dict[str, Any],
) -> TrainingMetrics:
    validation = [item for item in dataset["items"] if item["split"] == "validation"]
    evaluation_split = "validation"
    if not validation:
        validation = [item for item in dataset["items"] if item["split"] == "train"]
        evaluation_split = "train_fallback"

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    positive_images = 0
    negative_images = 0
    matched_ious: list[float] = []
    for item in validation:
        with Image.open(project_root / item["path"]) as source:
            image = source.convert("RGB")
        detections = _detect(image, model, threshold=0.55)
        expected = [
            (
                int(annotation["xMin"]),
                int(annotation["yMin"]),
                int(annotation["xMax"]),
                int(annotation["yMax"]),
            )
            for annotation in item["annotations"]
        ]
        if expected:
            positive_images += 1
            best_iou = max(
                (
                    _iou(
                        (
                            detection.bounding_box.x_min,
                            detection.bounding_box.y_min,
                            detection.bounding_box.x_max,
                            detection.bounding_box.y_max,
                        ),
                        expected_box,
                    )
                    for detection in detections
                    for expected_box in expected
                ),
                default=0.0,
            )
            if best_iou >= 0.5:
                true_positives += 1
                matched_ious.append(best_iou)
                false_positives += max(0, len(detections) - 1)
            else:
                false_negatives += 1
                false_positives += len(detections)
        else:
            negative_images += 1
            false_positives += len(detections)

    precision = true_positives / max(true_positives + false_positives, 1)
    recall = true_positives / max(true_positives + false_negatives, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return TrainingMetrics(
        evaluation_split=evaluation_split,
        positive_images=positive_images,
        negative_images=negative_images,
        true_positives=true_positives,
        false_positives=false_positives,
        false_negatives=false_negatives,
        precision=round(precision, 6),
        recall=round(recall, 6),
        f1=round(f1, 6),
        mean_iou=round(float(np.mean(matched_ious)) if matched_ious else 0.0, 6),
    )


def train_linear_detector(
    dataset_manifest_path: str | Path,
    output_directory: str | Path,
    seed: int,
) -> TrainingResult:
    dataset_path = resolved_path(dataset_manifest_path)
    output_directory = resolved_path(output_directory)
    model_path = output_directory / "model.json"
    metrics_path = output_directory / "metrics.json"
    warnings: list[WarningItem] = []

    try:
        dataset = _load_json(dataset_path)
        project_root = _project_root(dataset_path)
        randomizer = random.Random(seed)
        positive_features: list[np.ndarray] = []
        negative_features: list[np.ndarray] = []
        aspect_ratios: list[float] = []
        relative_heights: list[float] = []

        train_items = [item for item in dataset["items"] if item["split"] == "train"]
        for item in train_items:
            with Image.open(project_root / item["path"]) as source:
                image = source.convert("RGB")
            annotations = item.get("annotations", [])
            if annotations:
                for annotation in annotations:
                    box = _bounded_box(annotation, image)
                    patch = image.crop(box)
                    aspect_ratios.append(patch.width / max(patch.height, 1))
                    relative_heights.append(patch.height / max(image.height, 1))
                    positive_features.extend(
                        _feature_vector(variant) for variant in _positive_variants(patch)
                    )
                    negative_features.extend(
                        _feature_vector(negative) for negative in _corner_negatives(image, box)
                    )

        normalized_ratios = [min(3.0, max(0.33, value)) for value in aspect_ratios]
        for item in train_items:
            if item.get("annotations"):
                continue
            with Image.open(project_root / item["path"]) as source:
                image = source.convert("RGB")
            negative_features.extend(
                _feature_vector(window)
                for window in _negative_windows(image, randomizer, normalized_ratios)
            )

        if not positive_features:
            raise ValueError("training split has no positive bounding boxes")
        if not negative_features:
            raise ValueError("training split has no usable negative regions")

        positive = np.stack(positive_features)
        negative = np.stack(negative_features)
        combined = np.concatenate((positive, negative), axis=0)
        feature_mean = combined.mean(axis=0)
        feature_scale = combined.std(axis=0)
        feature_scale[feature_scale < 1e-4] = 1.0
        positive_normalized = (positive - feature_mean) / feature_scale
        negative_normalized = (negative - feature_mean) / feature_scale
        weights = positive_normalized.mean(axis=0) - negative_normalized.mean(axis=0)
        weight_norm = float(np.linalg.norm(weights))
        if weight_norm < 1e-6:
            raise ValueError("positive and negative features cannot be separated")
        weights /= weight_norm
        positive_scores = positive_normalized @ weights
        negative_scores = negative_normalized @ weights
        positive_floor = float(np.percentile(positive_scores, 20))
        negative_ceiling = float(np.percentile(negative_scores, 90))
        score_threshold = (positive_floor + negative_ceiling) / 2.0
        score_scale = max(
            float(np.std(np.concatenate((positive_scores, negative_scores)))) * 0.35,
            0.08,
        )

        ratio_quantiles = np.quantile(normalized_ratios, [0.2, 0.5, 0.8]).tolist()
        height_quantiles = np.quantile(relative_heights, [0.2, 0.5, 0.8]).tolist()
        model_id = str(uuid4())
        model = {
            "schemaVersion": 1,
            "modelId": model_id,
            "engine": ENGINE_NAME,
            "experimentalBaseline": True,
            "createdAt": datetime.now(UTC).isoformat(),
            "datasetId": dataset["id"],
            "datasetVersion": dataset["version"],
            "datasetChecksumSha256": sha256_file(dataset_path),
            "taskSpecId": dataset.get("taskSpecId"),
            "taskSpecRevision": dataset.get("taskSpecRevision"),
            "classId": dataset["classId"],
            "className": dataset["className"],
            "featureImageSize": FEATURE_IMAGE_SIZE,
            "featureVersion": FEATURE_VERSION,
            "featureMean": feature_mean.round(8).tolist(),
            "featureScale": feature_scale.round(8).tolist(),
            "weights": weights.round(8).tolist(),
            "scoreThreshold": round(score_threshold, 8),
            "scoreScale": round(score_scale, 8),
            "aspectRatios": [round(float(value), 6) for value in ratio_quantiles],
            "relativeHeights": [round(float(value), 6) for value in height_quantiles],
            "defaultConfidenceThreshold": 0.55,
            "nmsIouThreshold": 0.32,
            "maxDetections": 20,
            "training": {
                "seed": seed,
                "positiveFeatureCount": len(positive_features),
                "negativeFeatureCount": len(negative_features),
                "augmentationStorage": "ephemeral_features_only",
            },
        }
        _atomic_json(model_path, model)
        metrics = _evaluate(dataset, project_root, model)
        _atomic_json(metrics_path, asdict(metrics))

        if metrics.evaluation_split == "train_fallback":
            warnings.append(
                WarningItem(
                    code="no_validation_split",
                    message="독립 검증 세트가 없어 학습 데이터로만 기본 동작을 확인했습니다.",
                )
            )
        warnings.append(
            WarningItem(
                code="experimental_baseline",
                message=(
                    "현재 모델은 경량 CPU 기준선입니다. "
                    "실제 촬영 평가 후 ONNX 탐지 모델과 비교해야 합니다."
                ),
            )
        )
        return TrainingResult(
            status="succeeded",
            model_path=str(model_path),
            metrics_path=str(metrics_path),
            model_id=model_id,
            checksum_sha256=sha256_file(model_path),
            engine_name=ENGINE_NAME,
            deployment_status="experimental",
            metrics=metrics,
            warnings=warnings,
        )
    except (FileNotFoundError, KeyError, OSError, TypeError, ValueError) as error:
        return TrainingResult(
            status="failed",
            model_path=str(model_path),
            metrics_path=str(metrics_path),
            engine_name=ENGINE_NAME,
            deployment_status="experimental",
            error_code="training_failed",
            error_message=str(error),
        )


def infer_linear_batch(
    model_path: str | Path,
    input_paths: Iterable[str | Path],
    output_directory: str | Path,
    confidence_threshold: float | None = None,
) -> list[InferenceResult]:
    model_path = resolved_path(model_path)
    output_directory = resolved_path(output_directory)
    model = _load_json(model_path)
    threshold = (
        float(confidence_threshold)
        if confidence_threshold is not None
        else float(model.get("defaultConfidenceThreshold", 0.55))
    )
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("confidence threshold must be between zero and one")

    results: list[InferenceResult] = []
    for index, value in enumerate(input_paths):
        input_path = resolved_path(value)
        output_path = output_directory / f"result-{index + 1:06d}.png"
        started = time.perf_counter()
        try:
            with Image.open(input_path) as source:
                image = source.convert("RGB")
            detections, max_confidence = _detect_with_max_confidence(image, model, threshold)
            preview = image.copy()
            draw = ImageDraw.Draw(preview)
            for detection in detections:
                box = detection.bounding_box
                draw.rectangle(
                    (box.x_min, box.y_min, box.x_max, box.y_max),
                    outline=(237, 91, 42),
                    width=max(2, min(image.size) // 240),
                )
                draw.text(
                    (box.x_min + 3, max(0, box.y_min - 13)),
                    f"{detection.class_name} {detection.confidence:.2f}",
                    fill=(237, 91, 42),
                )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = output_path.with_suffix(".png.tmp")
            preview.save(temporary, format="PNG", optimize=True)
            temporary.replace(output_path)
            results.append(
                InferenceResult(
                    status="succeeded",
                    input_path=str(input_path),
                    output_path=str(output_path),
                    detections=detections,
                    max_confidence=round(max_confidence, 6),
                    checksum_sha256=sha256_file(output_path),
                    width=image.width,
                    height=image.height,
                    elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
                )
            )
        except (FileNotFoundError, OSError, TypeError, ValueError) as error:
            results.append(
                InferenceResult(
                    status="failed",
                    input_path=str(input_path),
                    output_path=str(output_path),
                    elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
                    error_code="inference_failed",
                    error_message=str(error),
                )
            )
    return results


def train_detector(
    dataset_manifest_path: str | Path,
    output_directory: str | Path,
    seed: int,
    backend: str = "auto",
    training_policy: dict[str, Any] | None = None,
) -> TrainingResult:
    selected = os.environ.get("VISIONFORGE_MODEL_BACKEND", backend).strip().lower()
    if selected in {"linear", "baseline", ENGINE_NAME.lower()}:
        return train_linear_detector(dataset_manifest_path, output_directory, seed)
    if selected not in {
        "auto",
        "torch",
        "torchvision",
        "torchvision_fasterrcnn_mobilenet_v3",
        "visionforge-torchvision-fasterrcnn-mobilenet-v3-v1",
    }:
        return TrainingResult(
            status="failed",
            model_path=str(resolved_path(output_directory) / "model.json"),
            metrics_path=str(resolved_path(output_directory) / "metrics.json"),
            error_code="unsupported_backend",
            error_message=f"unsupported model backend: {backend}",
        )
    try:
        from .torch_detector import TorchTrainingPolicy, train_torch_detector

        policy = TorchTrainingPolicy(**(training_policy or {}))
        return train_torch_detector(dataset_manifest_path, output_directory, seed, policy)
    except ImportError as error:
        return TrainingResult(
            status="failed",
            model_path=str(resolved_path(output_directory) / "model.json"),
            metrics_path=str(resolved_path(output_directory) / "metrics.json"),
            error_code="backend_unavailable",
            error_message=f"PyTorch backend is unavailable: {error}",
        )


def infer_batch(
    model_path: str | Path,
    input_paths: Iterable[str | Path],
    output_directory: str | Path,
    confidence_threshold: float | None = None,
) -> list[InferenceResult]:
    resolved_model = resolved_path(model_path)
    model = _load_json(resolved_model)
    engine = str(model.get("engine", ""))
    if engine == ENGINE_NAME or engine == "visionforge-linear-feature-detector-v1":
        return infer_linear_batch(
            resolved_model,
            input_paths,
            output_directory,
            confidence_threshold,
        )
    if engine == "visionforge-torchvision-fasterrcnn-mobilenet-v3-v1":
        from .torch_detector import infer_torch_batch

        return infer_torch_batch(
            resolved_model,
            input_paths,
            output_directory,
            confidence_threshold,
        )
    raise ValueError(f"unsupported model engine: {engine}")
