from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from collections.abc import Iterable
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import numpy as np
from PIL import Image, ImageDraw, ImageOps

from .hardware import HardwareProfile, select_torch_device
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

ENGINE_NAME = "visionforge-torchvision-fasterrcnn-mobilenet-v3-v1"
ARCHITECTURE = "fasterrcnn_mobilenet_v3_large_fpn"
ARTIFACT_NAME = "model.pt"
CHECKPOINT_NAME = "checkpoint.pt"
BEST_STATE_NAME = "best-state.pt"
PRETRAINED_WEIGHT_NAME = "fasterrcnn_mobilenet_v3_large_fpn-fb6a3cc7.pth"
PRETRAINED_WEIGHT_SHA256 = "fb6a3cc702b1df54c18a44b26708cd083614211062d0c36d2ca7bf9270df3533"


@dataclass(slots=True)
class TorchTrainingPolicy:
    epochs: int = 20
    batch_size: int | None = None
    gradient_accumulation: int | None = None
    learning_rate: float = 0.0002
    weight_decay: float = 0.0001
    min_size: int = 640
    max_size: int = 960
    trainable_backbone_layers: int = 2
    early_stopping_patience: int = 5
    checkpoint_minutes: int = 30
    score_threshold: float = 0.5
    target_precision: float = 0.98
    pretrained: bool = True
    device: str = "auto"

    def validate(self) -> None:
        if not 1 <= self.epochs <= 500:
            raise ValueError("epochs must be between 1 and 500")
        if self.batch_size is not None and not 1 <= self.batch_size <= 64:
            raise ValueError("batch size must be between 1 and 64")
        if self.gradient_accumulation is not None and not 1 <= self.gradient_accumulation <= 256:
            raise ValueError("gradient accumulation must be between 1 and 256")
        if not 1e-7 <= self.learning_rate <= 1.0:
            raise ValueError("learning rate is outside the supported range")
        if not 320 <= self.min_size <= self.max_size <= 2048:
            raise ValueError("training image size is outside the supported range")
        if not 0 <= self.trainable_backbone_layers <= 6:
            raise ValueError("trainable backbone layers must be between 0 and 6")
        if not 0.0 <= self.score_threshold <= 1.0:
            raise ValueError("score threshold must be between zero and one")
        if not 0.0 <= self.target_precision <= 1.0:
            raise ValueError("target precision must be between zero and one")


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object expected: {path}")
    return value


def _project_root(dataset_path: Path) -> Path:
    for parent in dataset_path.parents:
        if (parent / "project.json").exists():
            return parent
    raise ValueError("dataset manifest is not inside a VisionForge project")


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    temporary.replace(path)


def _atomic_torch_save(value: Any, path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _write_progress(path: Path, phase: str, **values: Any) -> None:
    _atomic_json(
        path,
        {
            "schemaVersion": 1,
            "phase": phase,
            "updatedAt": datetime.now(UTC).isoformat(),
            **values,
        },
    )


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device_defaults(device: str, profile: HardwareProfile) -> tuple[int, int, int]:
    if device == "mps" or profile.profile == "APPLE_M1_16_BASELINE":
        return 1, 8, 0
    if device == "cuda":
        memory = profile.accelerator_memory_bytes or 0
        if memory >= 24 * 1024**3:
            return 4, 2, 0
        return 2, 4, 0
    return 1, 8, 0


class _DetectionDataset:
    def __init__(self, items: list[dict[str, Any]], project_root: Path) -> None:
        self.items = items
        self.project_root = project_root

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[Any, dict[str, Any]]:
        import torch
        from torchvision.transforms.v2.functional import pil_to_tensor, to_dtype

        item = self.items[index]
        with Image.open(self.project_root / item["path"]) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
        tensor = to_dtype(pil_to_tensor(image), torch.float32, scale=True)
        boxes: list[list[float]] = []
        for annotation in item.get("annotations", []):
            left = max(0.0, min(float(image.width), float(annotation["xMin"])))
            top = max(0.0, min(float(image.height), float(annotation["yMin"])))
            right = max(0.0, min(float(image.width), float(annotation["xMax"])))
            bottom = max(0.0, min(float(image.height), float(annotation["yMax"])))
            if right > left and bottom > top:
                boxes.append([left, top, right, bottom])
        box_tensor = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels = torch.ones((len(boxes),), dtype=torch.int64)
        area = (
            (box_tensor[:, 2] - box_tensor[:, 0]) * (box_tensor[:, 3] - box_tensor[:, 1])
            if boxes
            else torch.zeros((0,), dtype=torch.float32)
        )
        target = {
            "boxes": box_tensor,
            "labels": labels,
            "image_id": torch.tensor(index, dtype=torch.int64),
            "area": area,
            "iscrowd": torch.zeros((len(boxes),), dtype=torch.int64),
        }
        return tensor, target


def _collate(batch: list[tuple[Any, dict[str, Any]]]) -> tuple[list[Any], list[dict[str, Any]]]:
    images, targets = zip(*batch, strict=True)
    return list(images), list(targets)


def _pretrained_weight_path() -> Path | None:
    candidates = []
    configured = os.environ.get("VISIONFORGE_PRETRAINED_WEIGHT")
    if configured:
        candidates.append(Path(configured).expanduser())
    bundle_root = getattr(sys, "_MEIPASS", None)
    if bundle_root:
        candidates.append(Path(bundle_root) / "resources/weights" / PRETRAINED_WEIGHT_NAME)
    candidates.append(
        Path(__file__).resolve().parents[2] / "resources/weights" / PRETRAINED_WEIGHT_NAME
    )
    for candidate in candidates:
        if candidate.is_file():
            if sha256_file(candidate) != PRETRAINED_WEIGHT_SHA256:
                raise ValueError("bundled pretrained model checksum mismatch")
            return candidate
    return None


def _configure_trainable_backbone(model: Any, layers: int) -> None:
    body_children = list(model.backbone.body.children())
    for parameter in model.backbone.body.parameters():
        parameter.requires_grad_(False)
    if layers > 0 and body_children:
        trainable_children = max(1, math.ceil(len(body_children) * min(layers, 6) / 6))
        for child in body_children[-trainable_children:]:
            for parameter in child.parameters():
                parameter.requires_grad_(True)
    for parameter in model.backbone.fpn.parameters():
        parameter.requires_grad_(True)


def _build_model(policy: TorchTrainingPolicy, pretrained: bool) -> Any:
    import torch
    from torchvision.models.detection import (
        FasterRCNN_MobileNet_V3_Large_FPN_Weights,
        fasterrcnn_mobilenet_v3_large_fpn,
    )
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    bundled_weight = _pretrained_weight_path() if pretrained else None
    allow_download = os.environ.get("VISIONFORGE_ALLOW_WEIGHT_DOWNLOAD") == "1"
    if pretrained and bundled_weight is None and not allow_download:
        raise FileNotFoundError(
            "bundled pretrained model is missing; offline training cannot start"
        )
    weights = (
        FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT
        if pretrained and bundled_weight is None and allow_download
        else None
    )
    model = fasterrcnn_mobilenet_v3_large_fpn(
        weights=weights,
        weights_backbone=None,
        trainable_backbone_layers=None,
        min_size=policy.min_size,
        max_size=policy.max_size,
        box_score_thresh=0.0,
        box_detections_per_img=100,
    )
    if bundled_weight is not None:
        state = torch.load(bundled_weight, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    previous_predictor = model.roi_heads.box_predictor
    in_features = previous_predictor.cls_score.in_features
    predictor = FastRCNNPredictor(in_features, 2)
    if pretrained:
        with torch.no_grad():
            predictor.cls_score.weight[0].copy_(previous_predictor.cls_score.weight[0])
            predictor.cls_score.bias[0].copy_(previous_predictor.cls_score.bias[0])
            predictor.cls_score.weight[1].copy_(
                previous_predictor.cls_score.weight[1:].mean(dim=0)
            )
            predictor.cls_score.bias[1].copy_(
                previous_predictor.cls_score.bias[1:].mean(dim=0)
            )
            previous_box_weights = previous_predictor.bbox_pred.weight.reshape(
                -1, 4, in_features
            )
            previous_box_bias = previous_predictor.bbox_pred.bias.reshape(-1, 4)
            predictor.bbox_pred.weight[:4].copy_(previous_box_weights[0])
            predictor.bbox_pred.bias[:4].copy_(previous_box_bias[0])
            predictor.bbox_pred.weight[4:].copy_(previous_box_weights[1:].mean(dim=0))
            predictor.bbox_pred.bias[4:].copy_(previous_box_bias[1:].mean(dim=0))
    model.roi_heads.box_predictor = predictor
    _configure_trainable_backbone(model, policy.trainable_backbone_layers)
    return model


def _to_device_target(target: dict[str, Any], device: Any) -> dict[str, Any]:
    return {key: value.to(device) for key, value in target.items()}


def _box_iou(left: Iterable[float], right: Iterable[float]) -> float:
    left = list(left)
    right = list(right)
    intersection_width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    intersection_height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = intersection_width * intersection_height
    if intersection <= 0:
        return 0.0
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return intersection / max(left_area + right_area - intersection, 1e-9)


def _metrics_for_threshold(
    predictions: list[dict[str, Any]],
    targets: list[dict[str, Any]],
    threshold: float,
    evaluation_split: str,
) -> TrainingMetrics:
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    positive_images = 0
    negative_images = 0
    matched_ious: list[float] = []
    for prediction, target in zip(predictions, targets, strict=True):
        expected = target["boxes"].tolist()
        if expected:
            positive_images += 1
        else:
            negative_images += 1
        detections = [
            (score, box)
            for score, box in zip(
                prediction["scores"].tolist(),
                prediction["boxes"].tolist(),
                strict=True,
            )
            if score >= threshold
        ]
        detections.sort(key=lambda value: value[0], reverse=True)
        unmatched = set(range(len(expected)))
        for _, box in detections:
            best_index = None
            best_iou = 0.0
            for expected_index in unmatched:
                overlap = _box_iou(box, expected[expected_index])
                if overlap > best_iou:
                    best_index = expected_index
                    best_iou = overlap
            if best_index is not None and best_iou >= 0.5:
                true_positives += 1
                unmatched.remove(best_index)
                matched_ious.append(best_iou)
            else:
                false_positives += 1
        false_negatives += len(unmatched)
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


def _evaluate_model(
    model: Any,
    dataset: _DetectionDataset,
    device: Any,
    evaluation_split: str,
    policy: TorchTrainingPolicy,
) -> tuple[TrainingMetrics, float]:
    import torch

    predictions: list[dict[str, Any]] = []
    targets: list[dict[str, Any]] = []
    model.eval()
    with torch.inference_mode():
        for index in range(len(dataset)):
            image, target = dataset[index]
            output = model([image.to(device)])[0]
            predictions.append(
                {
                    "boxes": output["boxes"].detach().cpu(),
                    "scores": output["scores"].detach().cpu(),
                }
            )
            targets.append({"boxes": target["boxes"].cpu()})

    score_values = [
        float(score)
        for prediction in predictions
        for score in prediction["scores"].tolist()
        if math.isfinite(float(score))
    ]
    candidates = {round(value, 3) for value in np.linspace(0.25, 0.9, 14).tolist()}
    if score_values:
        candidates.update(
            round(float(value), 3)
            for value in np.quantile(score_values, [0.25, 0.5, 0.75, 0.9]).tolist()
        )
    measured = [
        (
            threshold,
            _metrics_for_threshold(predictions, targets, threshold, evaluation_split),
        )
        for threshold in sorted(value for value in candidates if 0.05 <= value <= 0.95)
    ]
    precision_candidates = [
        value
        for value in measured
        if value[1].precision >= policy.target_precision and value[1].true_positives > 0
    ]
    if precision_candidates:
        selected = max(
            precision_candidates,
            key=lambda value: (value[1].recall, value[1].f1, value[0]),
        )
    else:
        selected = max(
            measured,
            key=lambda value: (value[1].f1, value[1].precision, value[1].recall, value[0]),
        )
    return selected[1], round(float(selected[0]), 3)


def _checkpoint_payload(
    model: Any,
    optimizer: Any,
    scheduler: Any,
    epoch: int,
    next_batch: int,
    best_f1: float,
    best_threshold: float,
    dataset_checksum: str,
    policy: TorchTrainingPolicy,
) -> dict[str, Any]:
    return {
        "formatVersion": 1,
        "engine": ENGINE_NAME,
        "datasetChecksumSha256": dataset_checksum,
        "policy": asdict(policy),
        "epoch": epoch,
        "nextBatch": next_batch,
        "bestF1": best_f1,
        "bestThreshold": best_threshold,
        "modelState": model.state_dict(),
        "optimizerState": optimizer.state_dict(),
        "schedulerState": scheduler.state_dict(),
    }


def _move_optimizer_state(optimizer: Any, device: Any) -> None:
    import torch

    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def train_torch_detector(
    dataset_manifest_path: str | Path,
    output_directory: str | Path,
    seed: int,
    policy: TorchTrainingPolicy | None = None,
) -> TrainingResult:
    import torch
    from torch.utils.data import DataLoader

    policy = policy or TorchTrainingPolicy()
    policy.validate()
    dataset_path = resolved_path(dataset_manifest_path)
    output_directory = resolved_path(output_directory)
    model_path = output_directory / "model.json"
    artifact_path = output_directory / ARTIFACT_NAME
    metrics_path = output_directory / "metrics.json"
    checkpoint_path = output_directory / CHECKPOINT_NAME
    best_state_path = output_directory / BEST_STATE_NAME
    progress_path = output_directory / "progress.json"
    warnings: list[WarningItem] = []

    try:
        _write_progress(progress_path, "preparing")
        dataset = _load_json(dataset_path)
        project_root = _project_root(dataset_path)
        dataset_checksum = sha256_file(dataset_path)
        train_items = [item for item in dataset["items"] if item["split"] == "train"]
        validation_items = [
            item for item in dataset["items"] if item["split"] == "validation"
        ]
        evaluation_split = "validation"
        if not validation_items:
            validation_items = train_items
            evaluation_split = "train_fallback"
            warnings.append(
                WarningItem(
                    code="no_validation_split",
                    message="독립 검증 세트가 없어 학습 데이터로만 기본 동작을 확인했습니다.",
                )
            )
        if not train_items:
            raise ValueError("training split is empty")
        if not any(item.get("annotations") for item in train_items):
            raise ValueError("training split has no positive bounding boxes")
        if not any(not item.get("annotations") for item in train_items):
            warnings.append(
                WarningItem(
                    code="no_explicit_negative_images",
                    message="명시적인 부정 학습 이미지가 없어 오탐 억제 성능이 제한될 수 있습니다.",
                )
            )

        _seed_everything(seed)
        device_name, hardware = select_torch_device(policy.device)
        device = torch.device(device_name)
        default_batch, default_accumulation, worker_count = _device_defaults(
            device_name, hardware
        )
        batch_size = policy.batch_size or default_batch
        accumulation = policy.gradient_accumulation or default_accumulation
        train_dataset = _DetectionDataset(train_items, project_root)
        validation_dataset = _DetectionDataset(validation_items, project_root)
        model = _build_model(policy, pretrained=policy.pretrained).to(device)
        _write_progress(
            progress_path,
            "model_ready",
            device=device_name,
            hardwareProfile=hardware.profile,
            epochs=policy.epochs,
            batchSize=batch_size,
            gradientAccumulation=accumulation,
        )
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(
            parameters,
            lr=policy.learning_rate,
            weight_decay=policy.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, policy.epochs)
        )
        scaler = torch.amp.GradScaler("cuda", enabled=device_name == "cuda")
        use_amp = device_name == "cuda"
        start_epoch = 0
        start_batch = 0
        best_f1 = -1.0
        best_threshold = policy.score_threshold
        resumed = False

        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            if (
                checkpoint.get("engine") == ENGINE_NAME
                and checkpoint.get("datasetChecksumSha256") == dataset_checksum
                and checkpoint.get("policy") == asdict(policy)
            ):
                model.load_state_dict(checkpoint["modelState"])
                optimizer.load_state_dict(checkpoint["optimizerState"])
                scheduler.load_state_dict(checkpoint["schedulerState"])
                _move_optimizer_state(optimizer, device)
                start_epoch = int(checkpoint.get("epoch", 0))
                start_batch = int(checkpoint.get("nextBatch", 0))
                best_f1 = float(checkpoint.get("bestF1", -1.0))
                best_threshold = float(
                    checkpoint.get("bestThreshold", policy.score_threshold)
                )
                resumed = True
                warnings.append(
                    WarningItem(
                        code="training_resumed",
                        message="마지막 정상 체크포인트부터 학습을 재개했습니다.",
                    )
                )
            else:
                checkpoint_path.unlink(missing_ok=True)
                best_state_path.unlink(missing_ok=True)
                warnings.append(
                    WarningItem(
                        code="incompatible_checkpoint_discarded",
                        message=(
                            "데이터셋 또는 학습 정책이 달라 기존 체크포인트를 "
                            "사용하지 않았습니다."
                        ),
                    )
                )
        elif best_state_path.exists():
            best_state_path.unlink()

        best_metrics: TrainingMetrics | None = None
        epochs_without_improvement = 0
        last_checkpoint = time.monotonic()
        last_progress_update = 0.0
        for epoch in range(start_epoch, policy.epochs):
            generator = torch.Generator().manual_seed(seed + epoch)
            loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                generator=generator,
                num_workers=worker_count,
                collate_fn=_collate,
                pin_memory=device_name == "cuda",
            )
            model.train()
            optimizer.zero_grad(set_to_none=True)
            total_batches = len(loader)
            for batch_index, (images, targets) in enumerate(loader):
                if epoch == start_epoch and batch_index < start_batch:
                    continue
                images = [image.to(device) for image in images]
                targets = [_to_device_target(target, device) for target in targets]
                context = (
                    torch.autocast(device_type="cuda", dtype=torch.float16)
                    if use_amp
                    else nullcontext()
                )
                with context:
                    losses = model(images, targets)
                    loss = sum(losses.values()) / accumulation
                if not torch.isfinite(loss):
                    raise ValueError("training loss became non-finite")
                scaler.scale(loss).backward()
                should_step = (batch_index + 1) % accumulation == 0 or (
                    batch_index + 1 == total_batches
                )
                if should_step:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(parameters, max_norm=5.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                now = time.monotonic()
                if (
                    batch_index == 0
                    or batch_index + 1 == total_batches
                    or now - last_progress_update >= 5
                ):
                    _write_progress(
                        progress_path,
                        "training",
                        epoch=epoch + 1,
                        totalEpochs=policy.epochs,
                        batch=batch_index + 1,
                        totalBatches=total_batches,
                        loss=round(float(loss.detach().cpu()) * accumulation, 6),
                    )
                    last_progress_update = now

                checkpoint_due = (
                    time.monotonic() - last_checkpoint >= policy.checkpoint_minutes * 60
                )
                if checkpoint_due and should_step:
                    _atomic_torch_save(
                        _checkpoint_payload(
                            model,
                            optimizer,
                            scheduler,
                            epoch,
                            batch_index + 1,
                            best_f1,
                            best_threshold,
                            dataset_checksum,
                            policy,
                        ),
                        checkpoint_path,
                    )
                    last_checkpoint = time.monotonic()

            scheduler.step()
            _write_progress(
                progress_path,
                "evaluating",
                epoch=epoch + 1,
                totalEpochs=policy.epochs,
            )
            metrics, threshold = _evaluate_model(
                model, validation_dataset, device, evaluation_split, policy
            )
            if metrics.f1 > best_f1 + 1e-6:
                best_f1 = metrics.f1
                best_threshold = threshold
                best_metrics = metrics
                epochs_without_improvement = 0
                _atomic_torch_save(
                    {
                        "formatVersion": 1,
                        "engine": ENGINE_NAME,
                        "modelState": model.state_dict(),
                    },
                    best_state_path,
                )
            else:
                epochs_without_improvement += 1

            _atomic_torch_save(
                _checkpoint_payload(
                    model,
                    optimizer,
                    scheduler,
                    epoch + 1,
                    0,
                    best_f1,
                    best_threshold,
                    dataset_checksum,
                    policy,
                ),
                checkpoint_path,
            )
            _write_progress(
                progress_path,
                "checkpointed",
                epoch=epoch + 1,
                totalEpochs=policy.epochs,
                bestF1=best_f1,
                bestThreshold=best_threshold,
            )
            start_batch = 0
            if epochs_without_improvement >= policy.early_stopping_patience:
                warnings.append(
                    WarningItem(
                        code="early_stopping",
                        message=(
                            "검증 점수가 개선되지 않아 가장 좋은 체크포인트로 "
                            "학습을 종료했습니다."
                        ),
                        value=epoch + 1,
                    )
                )
                break

        if best_state_path.exists():
            best_state = torch.load(best_state_path, map_location="cpu", weights_only=True)
            model.load_state_dict(best_state["modelState"])
        if best_metrics is None:
            best_metrics, best_threshold = _evaluate_model(
                model, validation_dataset, device, evaluation_split, policy
            )

        model = model.to("cpu")
        _write_progress(progress_path, "finalizing", bestF1=best_f1)
        artifact = {
            "formatVersion": 1,
            "engine": ENGINE_NAME,
            "architecture": ARCHITECTURE,
            "numClasses": 2,
            "stateDict": model.state_dict(),
        }
        _atomic_torch_save(artifact, artifact_path)
        model_id = str(uuid4())
        quality_status = "experimental" if evaluation_split == "train_fallback" else "candidate"
        model_metadata = {
            "schemaVersion": 1,
            "modelId": model_id,
            "engine": ENGINE_NAME,
            "format": "pytorch-state-dict-v1",
            "architecture": ARCHITECTURE,
            "artifactPath": ARTIFACT_NAME,
            "artifactSha256": sha256_file(artifact_path),
            "experimentalBaseline": False,
            "deploymentStatus": quality_status,
            "createdAt": datetime.now(UTC).isoformat(),
            "datasetId": dataset["id"],
            "datasetVersion": dataset["version"],
            "datasetChecksumSha256": dataset_checksum,
            "taskSpecId": dataset.get("taskSpecId"),
            "taskSpecRevision": dataset.get("taskSpecRevision"),
            "classId": dataset["classId"],
            "className": dataset["className"],
            "defaultConfidenceThreshold": best_threshold,
            "nmsIouThreshold": 0.45,
            "maxDetections": 50,
            "preprocessing": {
                "colorMode": "RGB",
                "exifOrientation": "apply",
                "minSize": policy.min_size,
                "maxSize": policy.max_size,
            },
            "tiling": {
                "enabled": True,
                "triggerSize": 1600,
                "tileSize": 1280,
                "overlap": 0.2,
                "maxTiles": 64,
            },
            "training": {
                "seed": seed,
                "policy": asdict(policy),
                "batchSize": batch_size,
                "gradientAccumulation": accumulation,
                "executionProvider": device_name,
                "hardwareProfile": hardware.to_dict(),
                "resumed": resumed,
                "augmentationStorage": "ephemeral_tensors_only",
            },
            "qualityGate": {
                "status": quality_status,
                "reason": "fixed_real_photo_evaluation_required",
                "targetPrecision": policy.target_precision,
                "evaluationSplit": evaluation_split,
            },
        }
        _atomic_json(model_path, model_metadata)
        _atomic_json(metrics_path, asdict(best_metrics))
        checkpoint_path.unlink(missing_ok=True)
        best_state_path.unlink(missing_ok=True)
        warnings.append(
            WarningItem(
                code="real_photo_evaluation_required",
                message="실제 촬영 고정 평가 세트의 품질 게이트를 통과하기 전에는 후보 모델입니다.",
            )
        )
        warnings.append(
            WarningItem(
                code="training_hardware",
                message=f"{hardware.profile} 프로필의 {device_name} 실행기로 학습했습니다.",
                value=hardware.accelerator_name,
            )
        )
        _write_progress(
            progress_path,
            "completed",
            modelId=model_id,
            bestF1=best_metrics.f1,
            confidenceThreshold=best_threshold,
        )
        return TrainingResult(
            status="succeeded",
            model_path=str(model_path),
            metrics_path=str(metrics_path),
            model_id=model_id,
            checksum_sha256=sha256_file(model_path),
            engine_name=ENGINE_NAME,
            deployment_status=quality_status,
            metrics=best_metrics,
            warnings=warnings,
        )
    except (
        FileNotFoundError,
        ImportError,
        KeyError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as error:
        if "out of memory" in str(error).lower():
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            except (ImportError, RuntimeError):
                pass
        _write_progress(progress_path, "failed", errorMessage=str(error))
        return TrainingResult(
            status="failed",
            model_path=str(model_path),
            metrics_path=str(metrics_path),
            engine_name=ENGINE_NAME,
            deployment_status="failed",
            error_code="torch_training_failed",
            error_message=str(error),
        )


def _load_trained_model(model_path: Path, device: Any) -> tuple[dict[str, Any], Any]:
    import torch

    metadata = _load_json(model_path)
    if metadata.get("engine") != ENGINE_NAME:
        raise ValueError("model is not a supported TorchVision detector")
    artifact_name = str(metadata.get("artifactPath", ""))
    if Path(artifact_name).name != artifact_name:
        raise ValueError("model artifact path is invalid")
    artifact_path = model_path.parent / artifact_name
    if sha256_file(artifact_path) != metadata.get("artifactSha256"):
        raise ValueError("model artifact checksum mismatch")
    preprocessing = metadata.get("preprocessing", {})
    policy = TorchTrainingPolicy(
        pretrained=False,
        min_size=int(preprocessing.get("minSize", 640)),
        max_size=int(preprocessing.get("maxSize", 960)),
    )
    model = _build_model(policy, pretrained=False)
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
    if artifact.get("engine") != ENGINE_NAME:
        raise ValueError("model artifact engine mismatch")
    model.load_state_dict(artifact["stateDict"])
    model.to(device).eval()
    return metadata, model


def _tile_regions(image: Image.Image, metadata: dict[str, Any]) -> list[tuple[int, int, int, int]]:
    tiling = metadata.get("tiling", {})
    regions = [(0, 0, image.width, image.height)]
    if not tiling.get("enabled", True):
        return regions
    trigger = int(tiling.get("triggerSize", 1600))
    if max(image.size) <= trigger:
        return regions
    tile_size = max(512, int(tiling.get("tileSize", 1280)))
    overlap = min(0.5, max(0.0, float(tiling.get("overlap", 0.2))))
    stride = max(1, round(tile_size * (1.0 - overlap)))
    max_tiles = max(1, int(tiling.get("maxTiles", 64)))
    lefts = list(range(0, max(1, image.width - tile_size + 1), stride))
    tops = list(range(0, max(1, image.height - tile_size + 1), stride))
    final_left = max(0, image.width - tile_size)
    final_top = max(0, image.height - tile_size)
    if not lefts or lefts[-1] != final_left:
        lefts.append(final_left)
    if not tops or tops[-1] != final_top:
        tops.append(final_top)
    for top in tops:
        for left in lefts:
            region = (
                left,
                top,
                min(image.width, left + tile_size),
                min(image.height, top + tile_size),
            )
            if region != regions[0]:
                regions.append(region)
            if len(regions) >= max_tiles + 1:
                return regions
    return regions


def _nms(
    candidates: list[tuple[float, tuple[int, int, int, int]]],
    threshold: float,
    maximum: int,
) -> list[tuple[float, tuple[int, int, int, int]]]:
    candidates.sort(key=lambda value: (value[0], value[1]), reverse=True)
    selected: list[tuple[float, tuple[int, int, int, int]]] = []
    for confidence, box in candidates:
        if all(_box_iou(box, kept_box) < threshold for _, kept_box in selected):
            selected.append((confidence, box))
        if len(selected) >= maximum:
            break
    return selected


def _image_detections(
    image: Image.Image,
    metadata: dict[str, Any],
    model: Any,
    device: Any,
    threshold: float,
) -> tuple[list[Detection], float]:
    import torch
    from torchvision.transforms.v2.functional import pil_to_tensor, to_dtype

    candidates: list[tuple[float, tuple[int, int, int, int]]] = []
    max_confidence = 0.0
    with torch.inference_mode():
        for left, top, right, bottom in _tile_regions(image, metadata):
            crop = image.crop((left, top, right, bottom))
            tensor = to_dtype(pil_to_tensor(crop), torch.float32, scale=True).to(device)
            output = model([tensor])[0]
            boxes = output["boxes"].detach().cpu().tolist()
            scores = output["scores"].detach().cpu().tolist()
            for score, box in zip(scores, boxes, strict=True):
                score = float(score)
                max_confidence = max(max_confidence, score)
                if score < threshold:
                    continue
                mapped = (
                    max(0, min(image.width, round(float(box[0])) + left)),
                    max(0, min(image.height, round(float(box[1])) + top)),
                    max(0, min(image.width, round(float(box[2])) + left)),
                    max(0, min(image.height, round(float(box[3])) + top)),
                )
                if mapped[2] > mapped[0] and mapped[3] > mapped[1]:
                    candidates.append((score, mapped))
    selected = _nms(
        candidates,
        float(metadata.get("nmsIouThreshold", 0.45)),
        int(metadata.get("maxDetections", 50)),
    )
    detections = [
        Detection(
            class_id=str(metadata["classId"]),
            class_name=str(metadata["className"]),
            confidence=round(score, 6),
            bounding_box=BoundingBox(*box),
        )
        for score, box in selected
    ]
    return detections, max_confidence


def infer_torch_batch(
    model_path: str | Path,
    input_paths: Iterable[str | Path],
    output_directory: str | Path,
    confidence_threshold: float | None = None,
) -> list[InferenceResult]:
    import torch

    model_path = resolved_path(model_path)
    output_directory = resolved_path(output_directory)
    device_name, _ = select_torch_device()
    device = torch.device(device_name)
    metadata, model = _load_trained_model(model_path, device)
    threshold = (
        float(confidence_threshold)
        if confidence_threshold is not None
        else float(metadata.get("defaultConfidenceThreshold", 0.5))
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
                image = ImageOps.exif_transpose(source).convert("RGB")
            detections, max_confidence = _image_detections(
                image, metadata, model, device, threshold
            )
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
        except (FileNotFoundError, OSError, RuntimeError, TypeError, ValueError) as error:
            results.append(
                InferenceResult(
                    status="failed",
                    input_path=str(input_path),
                    output_path=str(output_path),
                    elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
                    error_code="torch_inference_failed",
                    error_message=str(error),
                )
            )
    return results
