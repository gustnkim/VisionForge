from __future__ import annotations

import hashlib
import json
import stat
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

from .models import ModelPackageResult, WarningItem, resolved_path
from .quality import sha256_file

PACKAGE_MAGIC = "VISIONFORGE_MODEL"
PACKAGE_SCHEMA_VERSION = 1
MAX_ENTRIES = 64
MAX_UNCOMPRESSED_BYTES = 1024 * 1024 * 1024
MAX_COMPRESSION_RATIO = 100.0
REQUIRED_FILES = {
    "manifest.json",
    "model/model.json",
    "labels.json",
    "pipeline/preprocessing.json",
    "pipeline/postprocessing.json",
    "provenance/metrics.json",
    "compatibility.json",
    "integrity/checksums.json",
}


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _resource_file(relative_path: str) -> bytes:
    bundle_root = getattr(sys, "_MEIPASS", None)
    roots = []
    if bundle_root:
        roots.append(Path(bundle_root) / "resources")
    roots.append(Path(__file__).resolve().parents[2] / "resources")
    for root in roots:
        candidate = root.joinpath(*PurePosixPath(relative_path).parts)
        if candidate.is_file():
            return candidate.read_bytes()
    raise FileNotFoundError(f"required bundled resource is missing: {relative_path}")


def _safe_member(name: str) -> bool:
    path = PurePosixPath(name)
    return bool(name) and not path.is_absolute() and ".." not in path.parts and "\\" not in name


def _write_member(archive: zipfile.ZipFile, name: str, content: bytes) -> None:
    information = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
    information.compress_type = zipfile.ZIP_DEFLATED
    information.external_attr = 0o100644 << 16
    archive.writestr(information, content)


def _manifest(
    model: dict[str, Any],
    package_id: str,
    app_version: str,
    task_spec: dict[str, Any] | None,
) -> dict[str, Any]:
    model_format = model.get("format")
    if not model_format:
        model_format = (
            "visionforge-linear-json-v2"
            if int(model.get("featureVersion", 1)) >= 2
            else "visionforge-linear-json-v1"
        )
    model_manifest = {
        "id": model["modelId"],
        "format": model_format,
        "path": "model/model.json",
        "engine": model["engine"],
        "experimentalBaseline": bool(model.get("experimentalBaseline", False)),
        "deploymentStatus": model.get("deploymentStatus", "experimental"),
    }
    artifact_name = model.get("artifactPath")
    if artifact_name:
        model_manifest["artifact"] = f"model/{artifact_name}"
    manifest = {
        "magic": PACKAGE_MAGIC,
        "schemaVersion": PACKAGE_SCHEMA_VERSION,
        "packageId": package_id,
        "createdAt": datetime.now(UTC).isoformat(),
        "createdBy": {"application": "VisionForge", "version": app_version},
        "model": model_manifest,
        "class": {"id": model["classId"], "name": model["className"], "index": 0},
        "pipeline": {
            "preprocessing": "pipeline/preprocessing.json",
            "postprocessing": "pipeline/postprocessing.json",
        },
        "provenance": {
            "datasetId": model.get("datasetId"),
            "datasetVersion": model.get("datasetVersion"),
            "taskSpecId": model.get("taskSpecId"),
            "taskSpecRevision": model.get("taskSpecRevision"),
            "metrics": "provenance/metrics.json",
        },
        "compatibility": "compatibility.json",
        "integrity": {"algorithm": "SHA-256", "checksums": "integrity/checksums.json"},
    }
    if task_spec is not None:
        manifest["taskSpec"] = {
            "path": "task/task-spec.json",
            "schemaVersion": task_spec.get("schemaVersion"),
            "taskType": task_spec.get("taskType"),
            "revision": task_spec.get("revision"),
        }
    return manifest


def export_model_package(
    model_path: str | Path,
    metrics_path: str | Path,
    package_path: str | Path,
    app_version: str = "0.1.0",
    task_spec: dict[str, Any] | None = None,
) -> ModelPackageResult:
    model_path = resolved_path(model_path)
    metrics_path = resolved_path(metrics_path)
    package_path = resolved_path(package_path)
    warnings: list[WarningItem] = []
    try:
        if package_path.suffix.lower() != ".vfmodel":
            raise ValueError("model package extension must be .vfmodel")
        with model_path.open(encoding="utf-8") as stream:
            model = json.load(stream)
        with metrics_path.open(encoding="utf-8") as stream:
            metrics = json.load(stream)
        if model.get("schemaVersion") != 1 or not model.get("modelId"):
            raise ValueError("unsupported VisionForge model schema")
        if task_spec is not None:
            if task_spec.get("schemaVersion") != 1:
                raise ValueError("unsupported task spec schema")
            if task_spec.get("taskType") != "object_presence":
                raise ValueError("unsupported task spec type")
            if task_spec.get("className") != model.get("className"):
                raise ValueError("task spec class does not match the model")

        package_id = str(uuid4())
        preprocessing = model.get("preprocessing") or {
            "colorMode": "RGB",
            "exifOrientation": "apply",
            "featureImageSize": model["featureImageSize"],
            "featureVersion": int(model.get("featureVersion", 1)),
            "resizeInterpolation": "bilinear",
        }
        execution_providers = (
            ["mps", "cuda", "cpu"]
            if str(model.get("engine", "")).startswith("visionforge-torchvision-")
            else ["cpu"]
        )
        files: dict[str, bytes] = {
            "manifest.json": _json_bytes(_manifest(model, package_id, app_version, task_spec)),
            "model/model.json": model_path.read_bytes(),
            "labels.json": _json_bytes(
                {"classes": [{"id": model["classId"], "name": model["className"], "index": 0}]}
            ),
            "pipeline/preprocessing.json": _json_bytes(preprocessing),
            "pipeline/postprocessing.json": _json_bytes(
                {
                    "confidenceThreshold": model["defaultConfidenceThreshold"],
                    "nmsIouThreshold": model["nmsIouThreshold"],
                    "maxDetections": model["maxDetections"],
                    "sort": ["confidence_desc", "box_coordinates_asc"],
                }
            ),
            "provenance/metrics.json": _json_bytes(metrics),
            "compatibility.json": _json_bytes(
                {
                    "operatingSystems": ["windows", "macos"],
                    "architectures": ["x86_64", "aarch64"],
                    "executionProviders": execution_providers,
                    "minimumVisionForgeVersion": "0.1.0",
                }
            ),
            "licenses/pytorch-LICENSE.txt": _resource_file("licenses/pytorch-LICENSE.txt"),
            "licenses/pytorch-NOTICE.txt": _resource_file("licenses/pytorch-NOTICE.txt"),
            "licenses/torchvision-LICENSE.txt": _resource_file(
                "licenses/torchvision-LICENSE.txt"
            ),
            "licenses/PRETRAINED_MODEL_NOTICE.md": _resource_file(
                "PRETRAINED_MODEL_NOTICE.md"
            ),
        }
        artifact_name = model.get("artifactPath")
        if artifact_name:
            artifact_name = str(artifact_name)
            if Path(artifact_name).name != artifact_name:
                raise ValueError("model artifact path must be a file name")
            artifact_path = model_path.parent / artifact_name
            expected_checksum = model.get("artifactSha256")
            if expected_checksum and sha256_file(artifact_path) != expected_checksum:
                raise ValueError("model artifact checksum does not match model metadata")
            files[f"model/{artifact_name}"] = artifact_path.read_bytes()
        if task_spec is not None:
            files["task/task-spec.json"] = _json_bytes(task_spec)
        checksums = {name: _sha256_bytes(content) for name, content in sorted(files.items())}
        files["integrity/checksums.json"] = _json_bytes(
            {"algorithm": "SHA-256", "files": checksums}
        )

        package_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = package_path.with_suffix(".vfmodel.tmp")
        with zipfile.ZipFile(temporary, "w", allowZip64=False) as archive:
            for name, content in sorted(files.items()):
                _write_member(archive, name, content)
        temporary.replace(package_path)
        validated = import_model_package(package_path, None)
        if validated.status != "succeeded":
            raise ValueError(validated.error_message or "package self-validation failed")
        if model.get("experimentalBaseline"):
            warnings.append(
                WarningItem(
                    code="experimental_baseline",
                    message="이 패키지는 실제 촬영 평가가 필요한 실험 기준선 모델입니다.",
                )
            )
        return ModelPackageResult(
            status="succeeded",
            package_path=str(package_path),
            package_id=package_id,
            package_checksum_sha256=sha256_file(package_path),
            class_id=str(model["classId"]),
            class_name=str(model["className"]),
            engine_name=str(model["engine"]),
            deployment_status=str(model.get("deploymentStatus", "experimental")),
            model_path=str(model_path),
            metrics_path=str(metrics_path),
            task_spec_path=None,
            manifest=validated.manifest,
            warnings=warnings,
        )
    except (
        FileNotFoundError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
        zipfile.BadZipFile,
    ) as error:
        return ModelPackageResult(
            status="failed",
            package_path=str(package_path),
            error_code="package_export_failed",
            error_message=str(error),
        )


def _validated_archive(
    package_path: Path,
) -> tuple[zipfile.ZipFile, dict[str, bytes], dict[str, Any]]:
    if package_path.suffix.lower() != ".vfmodel":
        raise ValueError("model package extension must be .vfmodel")
    archive = zipfile.ZipFile(package_path, "r")
    information = archive.infolist()
    if len(information) > MAX_ENTRIES:
        archive.close()
        raise ValueError("model package contains too many entries")

    total_uncompressed = 0
    names: set[str] = set()
    for item in information:
        if not _safe_member(item.filename) or item.is_dir():
            archive.close()
            raise ValueError(f"unsafe model package entry: {item.filename}")
        mode = item.external_attr >> 16
        if mode and stat.S_ISLNK(mode):
            archive.close()
            raise ValueError(f"symbolic links are not allowed: {item.filename}")
        if item.filename in names:
            archive.close()
            raise ValueError(f"duplicate model package entry: {item.filename}")
        names.add(item.filename)
        total_uncompressed += item.file_size
        ratio = item.file_size / max(item.compress_size, 1)
        if ratio > MAX_COMPRESSION_RATIO:
            archive.close()
            raise ValueError(f"suspicious compression ratio: {item.filename}")
    if total_uncompressed > MAX_UNCOMPRESSED_BYTES:
        archive.close()
        raise ValueError("model package is too large after extraction")
    missing = REQUIRED_FILES - names
    if missing:
        archive.close()
        raise ValueError(f"model package is missing files: {', '.join(sorted(missing))}")

    files = {item.filename: archive.read(item) for item in information}
    manifest = json.loads(files["manifest.json"])
    if manifest.get("magic") != PACKAGE_MAGIC:
        archive.close()
        raise ValueError("model package magic header is invalid")
    if manifest.get("schemaVersion") != PACKAGE_SCHEMA_VERSION:
        archive.close()
        raise ValueError("unsupported model package schema")
    integrity = json.loads(files["integrity/checksums.json"])
    if integrity.get("algorithm") != "SHA-256":
        archive.close()
        raise ValueError("unsupported package checksum algorithm")
    expected = integrity.get("files", {})
    for name, checksum in expected.items():
        if name not in files or _sha256_bytes(files[name]) != checksum:
            archive.close()
            raise ValueError(f"model package checksum mismatch: {name}")
    if set(expected) != names - {"integrity/checksums.json"}:
        archive.close()
        raise ValueError("model package checksum inventory is incomplete")
    return archive, files, manifest


def import_model_package(
    package_path: str | Path,
    extract_directory: str | Path | None,
) -> ModelPackageResult:
    package_path = resolved_path(package_path)
    archive: zipfile.ZipFile | None = None
    try:
        archive, files, manifest = _validated_archive(package_path)
        model = json.loads(files["model/model.json"])
        package_model = manifest["model"]
        package_class = manifest["class"]
        if package_model.get("engine") != model.get("engine"):
            raise ValueError("package engine metadata does not match the model")
        if package_class.get("id") != model.get("classId"):
            raise ValueError("package class metadata does not match the model")
        if model.get("engine") not in {
            "visionforge-linear-feature-detector-v1",
            "visionforge-linear-feature-detector-v2",
            "visionforge-torchvision-fasterrcnn-mobilenet-v3-v1",
        }:
            raise ValueError("this VisionForge build does not support the packaged model engine")
        artifact_name = model.get("artifactPath")
        if artifact_name:
            artifact_name = str(artifact_name)
            if Path(artifact_name).name != artifact_name:
                raise ValueError("packaged model artifact path is invalid")
            package_artifact = f"model/{artifact_name}"
            if package_model.get("artifact") != package_artifact or package_artifact not in files:
                raise ValueError("packaged model artifact is missing")
            expected_checksum = model.get("artifactSha256")
            if expected_checksum and _sha256_bytes(files[package_artifact]) != expected_checksum:
                raise ValueError("packaged model artifact checksum mismatch")

        model_output = None
        metrics_output = None
        task_spec_output = None
        task_spec_metadata = manifest.get("taskSpec")
        if task_spec_metadata is not None:
            task_spec_name = str(task_spec_metadata.get("path", ""))
            if task_spec_name != "task/task-spec.json" or task_spec_name not in files:
                raise ValueError("model package task spec path is invalid")
            task_spec = json.loads(files[task_spec_name])
            if task_spec.get("schemaVersion") != 1:
                raise ValueError("unsupported packaged task spec schema")
            if task_spec.get("className") != package_class.get("name"):
                raise ValueError("packaged task spec class does not match the model")
        if extract_directory is not None:
            destination = resolved_path(extract_directory)
            destination.mkdir(parents=True, exist_ok=True)
            for name, content in files.items():
                output = destination.joinpath(*PurePosixPath(name).parts)
                output.parent.mkdir(parents=True, exist_ok=True)
                temporary = output.with_suffix(f"{output.suffix}.tmp")
                temporary.write_bytes(content)
                temporary.replace(output)
            model_output = destination / "model/model.json"
            metrics_output = destination / "provenance/metrics.json"
            if task_spec_metadata is not None:
                task_spec_output = destination / "task/task-spec.json"

        return ModelPackageResult(
            status="succeeded",
            package_path=str(package_path),
            package_id=str(manifest["packageId"]),
            package_checksum_sha256=sha256_file(package_path),
            class_id=str(package_class["id"]),
            class_name=str(package_class["name"]),
            engine_name=str(package_model["engine"]),
            deployment_status=str(package_model.get("deploymentStatus", "experimental")),
            model_path=str(model_output) if model_output else None,
            metrics_path=str(metrics_output) if metrics_output else None,
            task_spec_path=str(task_spec_output) if task_spec_output else None,
            manifest=manifest,
        )
    except (
        FileNotFoundError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
        zipfile.BadZipFile,
    ) as error:
        return ModelPackageResult(
            status="failed",
            package_path=str(package_path),
            error_code="package_import_failed",
            error_message=str(error),
        )
    finally:
        if archive is not None:
            archive.close()
