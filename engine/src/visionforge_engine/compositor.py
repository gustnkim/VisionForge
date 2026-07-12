from __future__ import annotations

import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from .models import BoundingBox, GenerationResult, WarningItem, resolved_path
from .quality import sha256_file


@dataclass(frozen=True, slots=True)
class GenerationSpec:
    target_path: str | Path
    background_path: str | Path
    output_path: str | Path
    seed: int
    scale: float = 1.0
    rotation_degrees: float = 0.0
    position_x: float = 0.5
    position_y: float = 0.5
    brightness: float = 1.0
    contrast: float = 1.0
    blur_radius: float = 0.0
    noise_stddev: float = 0.0
    occlusion_ratio: float = 0.0
    shadow_opacity: int = 70
    shadow_offset_x: int = 4
    shadow_offset_y: int = 6


@dataclass(frozen=True, slots=True)
class GenerationPolicy:
    scale_min: float = 0.45
    scale_max: float = 1.25
    rotation_min: float = -25.0
    rotation_max: float = 25.0
    brightness_min: float = 0.78
    brightness_max: float = 1.22
    contrast_min: float = 0.9
    contrast_max: float = 1.1
    blur_radius_max: float = 0.7
    noise_stddev_max: float = 0.0
    occlusion_max: float = 0.0


def _prepare_foreground(
    foreground: Image.Image,
    warnings: list[WarningItem],
) -> Image.Image:
    alpha = foreground.getchannel("A")
    if alpha.getextrema() != (255, 255):
        return foreground

    pixels = np.asarray(foreground.convert("RGB"), dtype=np.float32)
    border = np.concatenate(
        (pixels[0], pixels[-1], pixels[:, 0], pixels[:, -1]),
        axis=0,
    )
    background_color = np.median(border, axis=0)
    border_distance = np.linalg.norm(border - background_color, axis=1)
    distance = np.linalg.norm(pixels - background_color, axis=2)
    background_limit = max(12.0, float(np.percentile(border_distance, 95)) * 1.5)
    foreground_limit = background_limit + max(26.0, background_limit * 1.4)
    mask = np.clip(
        (distance - background_limit) / max(foreground_limit - background_limit, 1.0),
        0.0,
        1.0,
    )
    alpha_mask = Image.fromarray(np.rint(mask * 255).astype(np.uint8))
    alpha_mask = alpha_mask.filter(ImageFilter.MedianFilter(3)).filter(
        ImageFilter.GaussianBlur(0.8)
    )
    visible_ratio = float((np.asarray(alpha_mask) > 48).mean())
    if 0.015 <= visible_ratio <= 0.88:
        prepared = foreground.copy()
        prepared.putalpha(alpha_mask)
        warnings.append(
            WarningItem(
                code="automatic_background_mask",
                message=(
                    "가장자리 색을 기준으로 임시 전경 마스크를 만들었습니다. "
                    "합성 결과를 검토해 주세요."
                ),
                value=round(visible_ratio, 4),
            )
        )
        return prepared

    warnings.append(
        WarningItem(
            code="opaque_foreground",
            message="안전하게 분리할 배경을 찾지 못해 전체 사각형을 전경으로 사용했습니다.",
            value=round(visible_ratio, 4),
        )
    )
    return foreground


def _save_image(image: Image.Image, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        image.convert("RGB").save(output_path, quality=92, optimize=True)
    else:
        image.save(output_path, format="PNG", optimize=True)


def _fit_foreground(foreground: Image.Image, background: Image.Image, scale: float) -> Image.Image:
    if scale <= 0:
        raise ValueError("scale must be greater than zero")
    max_base = min(background.width * 0.55, background.height * 0.55)
    longest_edge = max(foreground.size)
    normalized_scale = min(1.0, max_base / longest_edge) if longest_edge else 1.0
    final_scale = normalized_scale * scale
    width = max(1, round(foreground.width * final_scale))
    height = max(1, round(foreground.height * final_scale))
    return foreground.resize((width, height), Image.Resampling.LANCZOS)


def _placement(
    background: Image.Image,
    foreground: Image.Image,
    x: float,
    y: float,
) -> tuple[int, int]:
    x = min(1.0, max(0.0, x))
    y = min(1.0, max(0.0, y))
    available_x = max(0, background.width - foreground.width)
    available_y = max(0, background.height - foreground.height)
    return round(available_x * x), round(available_y * y)


def _apply_occlusion(
    canvas: Image.Image,
    background: Image.Image,
    visible_mask: Image.Image,
    mask_box: tuple[int, int, int, int],
    ratio: float,
    seed: int,
) -> tuple[int, int, int, int] | None:
    ratio = min(0.8, max(0.0, ratio))
    if ratio <= 0.0:
        return None

    randomizer = random.Random(seed ^ 0x4F43_434C)
    left, top, right, bottom = mask_box
    width = right - left
    height = bottom - top
    if width < 4 or height < 4:
        return None

    if randomizer.random() < 0.5:
        occlusion_width = max(1, round(width * ratio))
        start = randomizer.randint(left, max(left, right - occlusion_width))
        rectangle = (start, top, min(right, start + occlusion_width), bottom)
    else:
        occlusion_height = max(1, round(height * ratio))
        start = randomizer.randint(top, max(top, bottom - occlusion_height))
        rectangle = (left, start, right, min(bottom, start + occlusion_height))

    patch = background.crop(rectangle)
    canvas.paste(patch, rectangle[:2])
    ImageDraw.Draw(visible_mask).rectangle(rectangle, fill=0)
    return rectangle


def _apply_noise(image: Image.Image, stddev: float, seed: int) -> Image.Image:
    stddev = min(0.25, max(0.0, stddev))
    if stddev <= 0.0:
        return image
    rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    rgb = rgba[:, :, :3].astype(np.float32)
    noise = np.random.default_rng(seed).normal(0.0, stddev * 255.0, rgb.shape)
    noisy = np.clip(rgb + noise, 0.0, 255.0).astype(np.uint8)
    result = np.empty_like(rgba)
    result[:, :, :3] = noisy
    result[:, :, 3] = rgba[:, :, 3]
    return Image.fromarray(result, mode="RGBA")


def render_composite(spec: GenerationSpec) -> GenerationResult:
    target_path = resolved_path(spec.target_path)
    background_path = resolved_path(spec.background_path)
    output_path = resolved_path(spec.output_path)
    warnings: list[WarningItem] = []

    try:
        with Image.open(background_path) as source_background:
            background = source_background.convert("RGBA")
        with Image.open(target_path) as source_target:
            foreground = source_target.convert("RGBA")

        foreground = _prepare_foreground(foreground, warnings)

        foreground = _fit_foreground(foreground, background, spec.scale)
        if spec.brightness != 1.0:
            rgb = ImageEnhance.Brightness(foreground.convert("RGB")).enhance(spec.brightness)
            rgb.putalpha(foreground.getchannel("A"))
            foreground = rgb
        if spec.contrast != 1.0:
            rgb = ImageEnhance.Contrast(foreground.convert("RGB")).enhance(spec.contrast)
            rgb.putalpha(foreground.getchannel("A"))
            foreground = rgb
        if spec.blur_radius > 0:
            foreground = foreground.filter(ImageFilter.GaussianBlur(spec.blur_radius))
        if spec.rotation_degrees:
            foreground = foreground.rotate(
                spec.rotation_degrees,
                resample=Image.Resampling.BICUBIC,
                expand=True,
            )

        x, y = _placement(background, foreground, spec.position_x, spec.position_y)
        visible_mask = Image.new("L", background.size, 0)
        visible_mask.paste(foreground.getchannel("A"), (x, y), foreground.getchannel("A"))
        initial_mask_box = visible_mask.getbbox()
        if initial_mask_box is None:
            raise ValueError("foreground is fully transparent after transformation")

        canvas = background.copy()
        if spec.shadow_opacity > 0:
            shadow = Image.new("RGBA", background.size, (0, 0, 0, 0))
            shadow_alpha = foreground.getchannel("A").point(
                lambda value: round(value * min(255, spec.shadow_opacity) / 255)
            )
            shadow_layer = Image.new("RGBA", foreground.size, (0, 0, 0, 0))
            shadow_layer.putalpha(shadow_alpha.filter(ImageFilter.GaussianBlur(3.0)))
            shadow.alpha_composite(
                shadow_layer,
                (x + spec.shadow_offset_x, y + spec.shadow_offset_y),
            )
            canvas = Image.alpha_composite(canvas, shadow)
        canvas.alpha_composite(foreground, (x, y))
        occlusion_box = _apply_occlusion(
            canvas,
            background,
            visible_mask,
            initial_mask_box,
            spec.occlusion_ratio,
            spec.seed,
        )
        mask_box = visible_mask.getbbox()
        if mask_box is None:
            raise ValueError("foreground is fully occluded after transformation")
        canvas = _apply_noise(canvas, spec.noise_stddev, spec.seed ^ 0x4E4F_4953)
        _save_image(canvas, output_path)

        bounding_box = BoundingBox(*mask_box)
        return GenerationResult(
            status="succeeded",
            output_path=str(output_path),
            source_target=str(target_path),
            source_background=str(background_path),
            seed=spec.seed,
            bounding_box=bounding_box,
            checksum_sha256=sha256_file(output_path),
            width=canvas.width,
            height=canvas.height,
            warnings=warnings,
            recipe={
                "scale": spec.scale,
                "rotation_degrees": spec.rotation_degrees,
                "position_x": spec.position_x,
                "position_y": spec.position_y,
                "brightness": spec.brightness,
                "contrast": spec.contrast,
                "blur_radius": spec.blur_radius,
                "noise_stddev": spec.noise_stddev,
                "occlusion_ratio": spec.occlusion_ratio,
                "occlusion_box": occlusion_box,
                "shadow_opacity": spec.shadow_opacity,
            },
        )
    except (FileNotFoundError, OSError, ValueError) as error:
        return GenerationResult(
            status="failed",
            output_path=str(output_path),
            source_target=str(target_path),
            source_background=str(background_path),
            seed=spec.seed,
            error_code="generation_failed",
            error_message=str(error),
        )


def generate_batch(
    target_paths: Iterable[str | Path],
    background_paths: Iterable[str | Path],
    output_directory: str | Path,
    count: int,
    seed: int,
    policy: GenerationPolicy | None = None,
) -> list[GenerationResult]:
    targets = tuple(target_paths)
    backgrounds = tuple(background_paths)
    if not targets:
        raise ValueError("at least one target image is required")
    if not backgrounds:
        raise ValueError("at least one background image is required")
    if count < 1:
        raise ValueError("count must be at least one")

    policy = policy or GenerationPolicy()
    if not 0 < policy.scale_min <= policy.scale_max:
        raise ValueError("scale range is invalid")
    if policy.rotation_min > policy.rotation_max:
        raise ValueError("rotation range is invalid")
    if not 0 < policy.brightness_min <= policy.brightness_max:
        raise ValueError("brightness range is invalid")
    if not 0 < policy.contrast_min <= policy.contrast_max:
        raise ValueError("contrast range is invalid")
    if not 0 <= policy.blur_radius_max <= 20:
        raise ValueError("blur radius range is invalid")
    if not 0 <= policy.noise_stddev_max <= 0.25:
        raise ValueError("noise range is invalid")
    if not 0 <= policy.occlusion_max <= 0.8:
        raise ValueError("occlusion range is invalid")
    output_directory = resolved_path(output_directory)
    randomizer = random.Random(seed)
    results: list[GenerationResult] = []

    # Each image is fully written before the next item starts, keeping memory bounded.
    for index in range(count):
        item_seed = randomizer.getrandbits(63)
        spec = GenerationSpec(
            target_path=randomizer.choice(targets),
            background_path=randomizer.choice(backgrounds),
            output_path=output_directory / f"generated-{index + 1:06d}.png",
            seed=item_seed,
            scale=randomizer.uniform(policy.scale_min, policy.scale_max),
            rotation_degrees=randomizer.uniform(policy.rotation_min, policy.rotation_max),
            position_x=randomizer.random(),
            position_y=randomizer.random(),
            brightness=randomizer.uniform(policy.brightness_min, policy.brightness_max),
            contrast=randomizer.uniform(policy.contrast_min, policy.contrast_max),
            blur_radius=randomizer.uniform(0.0, policy.blur_radius_max),
            noise_stddev=randomizer.uniform(0.0, policy.noise_stddev_max),
            occlusion_ratio=randomizer.uniform(0.0, policy.occlusion_max),
        )
        results.append(render_composite(spec))
    return results
