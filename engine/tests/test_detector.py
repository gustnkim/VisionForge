from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw

from visionforge_engine.detector import infer_batch, train_detector
from visionforge_engine.quality import sha256_file


def _positive(path: Path, box: tuple[int, int, int, int]) -> None:
    image = Image.new("RGB", (96, 96), (210, 218, 201))
    ImageDraw.Draw(image).rounded_rectangle(box, radius=4, fill=(220, 64, 35))
    image.save(path)


def _negative(path: Path, color: tuple[int, int, int]) -> None:
    Image.new("RGB", (96, 96), color).save(path)


def test_training_and_batch_inference_are_local_and_item_isolated(tmp_path: Path) -> None:
    project = tmp_path / "project"
    assets = project / "assets"
    dataset_directory = project / "datasets" / "v0001"
    dataset_directory.mkdir(parents=True)
    (project / "project.json").write_text("{}", encoding="utf-8")

    items = []
    for index, box in enumerate(((20, 22, 54, 58), (36, 18, 70, 54), (24, 38, 58, 74))):
        path = assets / f"positive-{index}.png"
        path.parent.mkdir(parents=True, exist_ok=True)
        _positive(path, box)
        items.append(
            {
                "assetId": f"positive-{index}",
                "role": "generated_positive",
                "path": f"assets/{path.name}",
                "checksumSha256": sha256_file(path),
                "width": 96,
                "height": 96,
                "split": "train" if index < 2 else "validation",
                "groupKey": f"group-{index}",
                "annotations": [
                    {
                        "classId": "class-1",
                        "xMin": box[0],
                        "yMin": box[1],
                        "xMax": box[2],
                        "yMax": box[3],
                        "source": "test",
                        "userModified": False,
                    }
                ],
            }
        )
    for index, color in enumerate(((205, 214, 197), (181, 194, 176))):
        path = assets / f"negative-{index}.png"
        _negative(path, color)
        items.append(
            {
                "assetId": f"negative-{index}",
                "role": "background",
                "path": f"assets/{path.name}",
                "checksumSha256": sha256_file(path),
                "width": 96,
                "height": 96,
                "split": "train" if index == 0 else "validation",
                "groupKey": f"negative-{index}",
                "annotations": [],
            }
        )

    dataset_path = dataset_directory / "dataset.json"
    dataset_path.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "id": "dataset-1",
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
        project / "models" / "model-1",
        2026,
        backend="linear",
    )
    assert training.status == "succeeded", training.error_message
    assert Path(training.model_path).exists()
    assert training.metrics is not None

    inference_image = assets / "inference.png"
    _positive(inference_image, (30, 28, 66, 66))
    broken_image = assets / "broken.png"
    broken_image.write_text("not an image", encoding="utf-8")
    results = infer_batch(
        training.model_path,
        [inference_image, broken_image],
        project / "assets" / "results",
        confidence_threshold=0.45,
    )

    assert results[0].status == "succeeded"
    assert results[0].detections
    assert results[0].max_confidence is not None
    assert 0.0 <= results[0].max_confidence <= 1.0
    assert Path(results[0].output_path).exists()
    assert results[1].status == "failed"
