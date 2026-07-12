from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .models import ImageInspection, WarningItem, resolved_path


@dataclass(frozen=True, slots=True)
class QualityPolicy:
    min_edge: int = 256
    max_pixels: int = 40_000_000
    low_brightness: float = 28.0
    high_brightness: float = 232.0
    low_contrast: float = 18.0
    blur_warning_score: float = 45.0


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def difference_hash(image: Image.Image) -> str:
    sample = ImageOps.grayscale(image).resize((9, 8), Image.Resampling.LANCZOS)
    pixels = np.asarray(sample, dtype=np.int16)
    bits = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in bits.ravel():
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def hamming_distance(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def laplacian_variance(image: Image.Image) -> float:
    gray = np.asarray(ImageOps.grayscale(image), dtype=np.float32)
    if min(gray.shape) < 3:
        return 0.0
    padded = np.pad(gray, 1, mode="edge")
    laplacian = (
        -4.0 * padded[1:-1, 1:-1]
        + padded[:-2, 1:-1]
        + padded[2:, 1:-1]
        + padded[1:-1, :-2]
        + padded[1:-1, 2:]
    )
    return float(np.var(laplacian))


def inspect_image(
    path_value: str | Path,
    policy: QualityPolicy | None = None,
) -> ImageInspection:
    policy = policy or QualityPolicy()
    path = resolved_path(path_value)

    try:
        file_size = path.stat().st_size
        checksum = sha256_file(path)
        with Image.open(path) as source:
            source.verify()
        with Image.open(path) as decoded:
            image_format = decoded.format
            image = ImageOps.exif_transpose(decoded).copy()

        width, height = image.size
        rgb = image.convert("RGB")
        gray = np.asarray(ImageOps.grayscale(rgb), dtype=np.float32)
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))
        blur_score = laplacian_variance(rgb)
        warnings: list[WarningItem] = []

        if min(width, height) < policy.min_edge:
            warnings.append(
                WarningItem(
                    code="small_image",
                    message="이미지의 짧은 변이 권장 크기보다 작습니다.",
                    value=min(width, height),
                )
            )
        if width * height > policy.max_pixels:
            warnings.append(
                WarningItem(
                    code="large_image",
                    message="이미지 픽셀 수가 권장 한도를 초과합니다.",
                    value=width * height,
                )
            )
        if brightness < policy.low_brightness:
            warnings.append(
                WarningItem("underexposed", "이미지가 매우 어둡습니다.", round(brightness, 2))
            )
        elif brightness > policy.high_brightness:
            warnings.append(
                WarningItem("overexposed", "이미지가 매우 밝습니다.", round(brightness, 2))
            )
        if contrast < policy.low_contrast:
            warnings.append(
                WarningItem("low_contrast", "이미지 대비가 낮습니다.", round(contrast, 2))
            )
        if blur_score < policy.blur_warning_score:
            warnings.append(
                WarningItem("possible_blur", "초점이 흐릴 가능성이 있습니다.", round(blur_score, 2))
            )

        return ImageInspection(
            path=str(path),
            status="succeeded",
            checksum_sha256=checksum,
            perceptual_hash=difference_hash(rgb),
            image_format=image_format,
            width=width,
            height=height,
            file_size=file_size,
            brightness_mean=round(brightness, 4),
            contrast_stddev=round(contrast, 4),
            blur_score=round(blur_score, 4),
            has_alpha="A" in image.getbands(),
            warnings=warnings,
        )
    except FileNotFoundError:
        return ImageInspection(
            path=str(path),
            status="failed",
            error_code="file_not_found",
            error_message="이미지 파일을 찾을 수 없습니다.",
        )
    except (UnidentifiedImageError, OSError, ValueError) as error:
        return ImageInspection(
            path=str(path),
            status="failed",
            error_code="decode_failed",
            error_message=str(error),
        )


def inspect_images(
    paths: Iterable[str | Path],
    policy: QualityPolicy | None = None,
) -> list[ImageInspection]:
    return [inspect_image(path, policy) for path in paths]
