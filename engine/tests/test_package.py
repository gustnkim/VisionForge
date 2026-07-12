from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

from visionforge_engine.package import export_model_package, import_model_package


def _model_files(directory: Path) -> tuple[Path, Path]:
    model = directory / "model.json"
    metrics = directory / "metrics.json"
    model.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "modelId": "model-1",
                "engine": "visionforge-linear-feature-detector-v1",
                "experimentalBaseline": True,
                "classId": "class-1",
                "className": "부품",
                "featureImageSize": 32,
                "defaultConfidenceThreshold": 0.55,
                "nmsIouThreshold": 0.32,
                "maxDetections": 20,
                "datasetId": "dataset-1",
                "datasetVersion": 1,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    metrics.write_text(json.dumps({"precision": 0.8, "recall": 0.7}), encoding="utf-8")
    return model, metrics


def test_model_package_round_trip_validates_and_extracts(tmp_path: Path) -> None:
    model, metrics = _model_files(tmp_path)
    package = tmp_path / "part.vfmodel"
    task_spec = {
        "schemaVersion": 1,
        "id": "task-1",
        "revision": 3,
        "taskType": "object_presence",
        "classId": "class-1",
        "className": "부품",
        "outputPolicy": {
            "presentFolder": "포함",
            "absentFolder": "미포함",
        },
    }
    exported = export_model_package(model, metrics, package, task_spec=task_spec)
    assert exported.status == "succeeded", exported.error_message

    imported = import_model_package(package, tmp_path / "imported")
    assert imported.status == "succeeded", imported.error_message
    assert imported.package_id == exported.package_id
    assert Path(imported.model_path or "").exists()
    assert Path(imported.task_spec_path or "").exists()


def test_model_package_rejects_path_traversal(tmp_path: Path) -> None:
    package = tmp_path / "unsafe.vfmodel"
    with zipfile.ZipFile(package, "w") as archive:
        archive.writestr("../escape.txt", "no")

    imported = import_model_package(package, tmp_path / "extract")
    assert imported.status == "failed"
    assert not (tmp_path / "escape.txt").exists()


def test_torch_model_package_includes_and_validates_weight_artifact(tmp_path: Path) -> None:
    artifact = tmp_path / "model.pt"
    artifact.write_bytes(b"safe-test-weight-artifact")
    model = tmp_path / "model.json"
    metrics = tmp_path / "metrics.json"
    model.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "modelId": "torch-model-1",
                "engine": "visionforge-torchvision-fasterrcnn-mobilenet-v3-v1",
                "format": "pytorch-state-dict-v1",
                "artifactPath": artifact.name,
                "artifactSha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                "experimentalBaseline": False,
                "deploymentStatus": "candidate",
                "classId": "class-1",
                "className": "부품",
                "defaultConfidenceThreshold": 0.72,
                "nmsIouThreshold": 0.45,
                "maxDetections": 50,
                "datasetId": "dataset-1",
                "datasetVersion": 1,
                "preprocessing": {
                    "colorMode": "RGB",
                    "minSize": 640,
                    "maxSize": 960,
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    metrics.write_text(json.dumps({"precision": 0.9, "recall": 0.8}), encoding="utf-8")
    package = tmp_path / "torch-model.vfmodel"

    exported = export_model_package(model, metrics, package)
    assert exported.status == "succeeded", exported.error_message
    with zipfile.ZipFile(package) as archive:
        assert archive.read("model/model.pt") == artifact.read_bytes()
        compatibility = json.loads(archive.read("compatibility.json"))
        assert compatibility["executionProviders"] == ["mps", "cuda", "cpu"]
        assert "licenses/pytorch-LICENSE.txt" in archive.namelist()
        assert "licenses/torchvision-LICENSE.txt" in archive.namelist()
        assert "licenses/PRETRAINED_MODEL_NOTICE.md" in archive.namelist()

    imported = import_model_package(package, tmp_path / "torch-imported")
    assert imported.status == "succeeded", imported.error_message
    assert (tmp_path / "torch-imported/model/model.pt").read_bytes() == artifact.read_bytes()
