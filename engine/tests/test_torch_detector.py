from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from visionforge_engine.detector import infer_batch, train_detector
from visionforge_engine.quality import sha256_file

RUN_TORCH_INTEGRATION = os.environ.get("VISIONFORGE_RUN_TORCH_INTEGRATION") == "1"
USE_PRETRAINED = os.environ.get("VISIONFORGE_TEST_PRETRAINED") == "1"


def _write_image(path: Path, box: tuple[int, int, int, int] | None) -> None:
    image = Image.new("RGB", (96, 96), (205, 214, 197))
    if box is not None:
        ImageDraw.Draw(image).rounded_rectangle(box, radius=4, fill=(220, 64, 35))
    image.save(path)


@pytest.mark.skipif(not RUN_TORCH_INTEGRATION, reason="explicit Torch integration test")
def test_torch_backend_trains_packages_weights_and_runs_inference(tmp_path: Path) -> None:
    project = tmp_path / "project"
    assets = project / "assets"
    dataset_directory = project / "datasets" / "v0001"
    dataset_directory.mkdir(parents=True)
    assets.mkdir(parents=True)
    (project / "project.json").write_text("{}", encoding="utf-8")

    fixtures = [
        ("positive-train-1.png", (18, 20, 58, 62), "train"),
        ("positive-train-2.png", (32, 26, 72, 68), "train"),
        ("negative-train.png", None, "train"),
        ("positive-validation.png", (26, 24, 66, 66), "validation"),
        ("negative-validation.png", None, "validation"),
    ]
    items = []
    for index, (name, box, split) in enumerate(fixtures):
        path = assets / name
        _write_image(path, box)
        annotations = []
        if box is not None:
            annotations.append(
                {
                    "classId": "class-1",
                    "xMin": box[0],
                    "yMin": box[1],
                    "xMax": box[2],
                    "yMax": box[3],
                    "source": "test",
                    "userModified": False,
                }
            )
        items.append(
            {
                "assetId": f"asset-{index}",
                "role": "generated_positive" if box else "background",
                "path": f"assets/{name}",
                "checksumSha256": sha256_file(path),
                "width": 96,
                "height": 96,
                "split": split,
                "groupKey": f"group-{index}",
                "annotations": annotations,
            }
        )

    dataset_path = dataset_directory / "dataset.json"
    dataset_path.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "id": "torch-dataset-1",
                "version": 1,
                "classId": "class-1",
                "className": "빨간 부품",
                "items": items,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    training = train_detector(
        dataset_path,
        project / "models" / "torch-model",
        2026,
        backend="torch",
        training_policy={
            "epochs": 1,
            "batch_size": 1,
            "gradient_accumulation": 1,
            "min_size": 320,
            "max_size": 320,
            "trainable_backbone_layers": 0,
            "early_stopping_patience": 1,
            "pretrained": USE_PRETRAINED,
        },
    )

    assert training.status == "succeeded", training.error_message
    model_path = Path(training.model_path)
    metadata = json.loads(model_path.read_text(encoding="utf-8"))
    assert metadata["engine"] == "visionforge-torchvision-fasterrcnn-mobilenet-v3-v1"
    assert (model_path.parent / metadata["artifactPath"]).exists()

    results = infer_batch(
        model_path,
        [assets / "positive-validation.png"],
        project / "assets" / "torch-results",
        confidence_threshold=0.0,
    )
    assert results[0].status == "succeeded", results[0].error_message
    assert Path(results[0].output_path).exists()
