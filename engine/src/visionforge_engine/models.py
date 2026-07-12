from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

ItemStatus = Literal["succeeded", "failed"]


@dataclass(slots=True)
class WarningItem:
    code: str
    message: str
    value: float | int | str | None = None


@dataclass(slots=True)
class ImageInspection:
    path: str
    status: ItemStatus
    checksum_sha256: str | None = None
    perceptual_hash: str | None = None
    image_format: str | None = None
    width: int | None = None
    height: int | None = None
    file_size: int | None = None
    brightness_mean: float | None = None
    contrast_stddev: float | None = None
    blur_score: float | None = None
    has_alpha: bool | None = None
    warnings: list[WarningItem] = field(default_factory=list)
    error_code: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class BoundingBox:
    x_min: int
    y_min: int
    x_max: int
    y_max: int

    @property
    def width(self) -> int:
        return self.x_max - self.x_min

    @property
    def height(self) -> int:
        return self.y_max - self.y_min


@dataclass(slots=True)
class GenerationResult:
    status: ItemStatus
    output_path: str
    source_target: str
    source_background: str
    seed: int
    bounding_box: BoundingBox | None = None
    checksum_sha256: str | None = None
    width: int | None = None
    height: int | None = None
    recipe: dict[str, Any] = field(default_factory=dict)
    warnings: list[WarningItem] = field(default_factory=list)
    error_code: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TrainingMetrics:
    evaluation_split: str
    positive_images: int
    negative_images: int
    true_positives: int
    false_positives: int
    false_negatives: int
    precision: float
    recall: float
    f1: float
    mean_iou: float


@dataclass(slots=True)
class TrainingResult:
    status: ItemStatus
    model_path: str
    metrics_path: str
    model_id: str | None = None
    checksum_sha256: str | None = None
    engine_name: str | None = None
    deployment_status: str | None = None
    metrics: TrainingMetrics | None = None
    warnings: list[WarningItem] = field(default_factory=list)
    error_code: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Detection:
    class_id: str
    class_name: str
    confidence: float
    bounding_box: BoundingBox


@dataclass(slots=True)
class InferenceResult:
    status: ItemStatus
    input_path: str
    output_path: str
    detections: list[Detection] = field(default_factory=list)
    max_confidence: float | None = None
    checksum_sha256: str | None = None
    width: int | None = None
    height: int | None = None
    elapsed_ms: float | None = None
    error_code: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ModelPackageResult:
    status: ItemStatus
    package_path: str
    package_id: str | None = None
    package_checksum_sha256: str | None = None
    class_id: str | None = None
    class_name: str | None = None
    engine_name: str | None = None
    deployment_status: str | None = None
    model_path: str | None = None
    metrics_path: str | None = None
    task_spec_path: str | None = None
    manifest: dict[str, Any] = field(default_factory=dict)
    warnings: list[WarningItem] = field(default_factory=list)
    error_code: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolved_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()
