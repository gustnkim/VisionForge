from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .compositor import GenerationPolicy, generate_batch
from .detector import infer_batch, train_detector
from .hardware import probe_hardware
from .package import export_model_package, import_model_package
from .quality import QualityPolicy, inspect_images


def _configure_utf8_stdio() -> None:
    for stream in (sys.stdin, sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")


def _load_payload(path: str) -> dict[str, Any]:
    if path == "-":
        return json.load(sys.stdin)
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def _write_result(value: Any) -> None:
    json.dump(value, sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")


def _inspect(payload: dict[str, Any]) -> list[dict[str, Any]]:
    policy = QualityPolicy(**payload.get("policy", {}))
    return [item.to_dict() for item in inspect_images(payload["paths"], policy)]


def _generate(payload: dict[str, Any]) -> list[dict[str, Any]]:
    policy = GenerationPolicy(**payload.get("policy", {}))
    results = generate_batch(
        target_paths=payload["target_paths"],
        background_paths=payload["background_paths"],
        output_directory=payload["output_directory"],
        count=int(payload["count"]),
        seed=int(payload["seed"]),
        policy=policy,
    )
    return [item.to_dict() for item in results]


def _train(payload: dict[str, Any]) -> dict[str, Any]:
    result = train_detector(
        dataset_manifest_path=payload["dataset_manifest_path"],
        output_directory=payload["output_directory"],
        seed=int(payload["seed"]),
        backend=str(payload.get("backend", "auto")),
        training_policy=payload.get("training_policy"),
    )
    return result.to_dict()


def _infer(payload: dict[str, Any]) -> list[dict[str, Any]]:
    results = infer_batch(
        model_path=payload["model_path"],
        input_paths=payload["input_paths"],
        output_directory=payload["output_directory"],
        confidence_threshold=payload.get("confidence_threshold"),
    )
    return [item.to_dict() for item in results]


def _export_model(payload: dict[str, Any]) -> dict[str, Any]:
    return export_model_package(
        model_path=payload["model_path"],
        metrics_path=payload["metrics_path"],
        package_path=payload["package_path"],
        app_version=str(payload.get("app_version", "0.1.0")),
        task_spec=payload.get("task_spec"),
    ).to_dict()


def _import_model(payload: dict[str, Any]) -> dict[str, Any]:
    return import_model_package(
        package_path=payload["package_path"],
        extract_directory=payload.get("extract_directory"),
    ).to_dict()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="visionforge-engine")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in (
        "system-profile",
        "inspect",
        "generate",
        "train",
        "infer",
        "export-model",
        "import-model",
    ):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--input", required=True, help="JSON request path or '-' for stdin")
    return parser


def main() -> None:
    _configure_utf8_stdio()
    args = build_parser().parse_args()
    payload = _load_payload(args.input)
    try:
        if args.command == "system-profile":
            _write_result(
                {
                    "status": "succeeded",
                    "result": probe_hardware(payload.get("path")).to_dict(),
                }
            )
        elif args.command == "inspect":
            _write_result({"status": "succeeded", "items": _inspect(payload)})
        elif args.command == "generate":
            _write_result({"status": "succeeded", "items": _generate(payload)})
        elif args.command == "train":
            result = _train(payload)
            _write_result({"status": result["status"], "result": result})
        elif args.command == "infer":
            _write_result({"status": "succeeded", "items": _infer(payload)})
        elif args.command == "export-model":
            result = _export_model(payload)
            _write_result({"status": result["status"], "result": result})
        else:
            result = _import_model(payload)
            _write_result({"status": result["status"], "result": result})
    except (KeyError, TypeError, ValueError) as error:
        _write_result(
            {
                "status": "failed",
                "error_code": "invalid_request",
                "error_message": str(error),
            }
        )
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
