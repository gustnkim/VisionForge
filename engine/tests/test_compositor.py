import hashlib
from pathlib import Path

from PIL import Image, ImageDraw

from visionforge_engine.compositor import (
    GenerationPolicy,
    GenerationSpec,
    generate_batch,
    render_composite,
)


def _fixtures(tmp_path: Path) -> tuple[Path, Path]:
    target_path = tmp_path / "target.png"
    background_path = tmp_path / "background.png"

    target = Image.new("RGBA", (32, 24), (0, 0, 0, 0))
    ImageDraw.Draw(target).rectangle((4, 3, 27, 20), fill=(220, 70, 30, 255))
    target.save(target_path)
    Image.new("RGB", (160, 120), (228, 222, 205)).save(background_path)
    return target_path, background_path


def test_render_composite_uses_visible_alpha_for_box(tmp_path: Path) -> None:
    target_path, background_path = _fixtures(tmp_path)
    output_path = tmp_path / "result.png"

    result = render_composite(
        GenerationSpec(
            target_path=target_path,
            background_path=background_path,
            output_path=output_path,
            seed=7,
            scale=1.0,
            rotation_degrees=0,
            position_x=0,
            position_y=0,
            shadow_opacity=0,
        )
    )

    assert result.status == "succeeded"
    assert result.bounding_box is not None
    assert result.bounding_box.x_min == 4
    assert result.bounding_box.y_min == 3
    assert result.bounding_box.x_max == 28
    assert result.bounding_box.y_max == 21
    assert result.width == 160
    assert result.height == 120
    assert result.checksum_sha256 is not None
    assert output_path.exists()


def test_batch_generation_is_deterministic(tmp_path: Path) -> None:
    target_path, background_path = _fixtures(tmp_path)
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    policy = GenerationPolicy(blur_radius_max=0)

    first = generate_batch([target_path], [background_path], first_dir, 3, 2026, policy)
    second = generate_batch([target_path], [background_path], second_dir, 3, 2026, policy)

    assert [item.recipe for item in first] == [item.recipe for item in second]
    assert [item.bounding_box for item in first] == [item.bounding_box for item in second]
    for index in range(3):
        first_hash = hashlib.sha256((first_dir / f"generated-{index + 1:06d}.png").read_bytes())
        second_hash = hashlib.sha256((second_dir / f"generated-{index + 1:06d}.png").read_bytes())
        assert first_hash.digest() == second_hash.digest()


def test_opaque_target_gets_a_conservative_color_background_mask(tmp_path: Path) -> None:
    target_path = tmp_path / "opaque-target.png"
    background_path = tmp_path / "scene.png"
    output_path = tmp_path / "masked-result.png"
    target = Image.new("RGB", (64, 64), (245, 244, 238))
    ImageDraw.Draw(target).rectangle((20, 18, 43, 45), fill=(214, 58, 34))
    target.save(target_path)
    Image.new("RGB", (160, 120), (190, 202, 180)).save(background_path)

    result = render_composite(
        GenerationSpec(
            target_path=target_path,
            background_path=background_path,
            output_path=output_path,
            seed=8,
            scale=1,
            rotation_degrees=0,
            position_x=0.5,
            position_y=0.5,
            shadow_opacity=0,
        )
    )

    assert result.status == "succeeded"
    assert result.bounding_box is not None
    assert result.bounding_box.width < 40
    assert result.bounding_box.height < 45
    assert any(warning.code == "automatic_background_mask" for warning in result.warnings)


def test_scenario_effects_are_recorded_and_keep_a_visible_box(tmp_path: Path) -> None:
    target_path, background_path = _fixtures(tmp_path)
    output_path = tmp_path / "scenario-result.png"

    result = render_composite(
        GenerationSpec(
            target_path=target_path,
            background_path=background_path,
            output_path=output_path,
            seed=99,
            scale=0.8,
            contrast=0.78,
            noise_stddev=0.03,
            occlusion_ratio=0.3,
            shadow_opacity=0,
        )
    )

    assert result.status == "succeeded"
    assert result.bounding_box is not None
    assert result.recipe["contrast"] == 0.78
    assert result.recipe["noise_stddev"] == 0.03
    assert result.recipe["occlusion_ratio"] == 0.3
    assert result.recipe["occlusion_box"] is not None
