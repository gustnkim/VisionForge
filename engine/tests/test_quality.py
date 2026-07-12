from pathlib import Path

from PIL import Image

from visionforge_engine.quality import hamming_distance, inspect_image


def _save_test_image(path: Path, color: tuple[int, int, int]) -> None:
    image = Image.new("RGB", (320, 240), color)
    for x in range(0, image.width, 16):
        image.paste((255 - color[0], color[1], color[2]), (x, 0, x + 8, image.height))
    image.save(path)


def test_inspection_returns_reproducible_hashes(tmp_path: Path) -> None:
    first_path = tmp_path / "first.png"
    second_path = tmp_path / "second.png"
    _save_test_image(first_path, (40, 100, 180))
    second_path.write_bytes(first_path.read_bytes())

    first = inspect_image(first_path)
    second = inspect_image(second_path)

    assert first.status == "succeeded"
    assert first.width == 320
    assert first.height == 240
    assert first.checksum_sha256 == second.checksum_sha256
    assert first.perceptual_hash == second.perceptual_hash
    assert hamming_distance(first.perceptual_hash or "0", second.perceptual_hash or "0") == 0


def test_corrupt_image_fails_without_raising(tmp_path: Path) -> None:
    corrupt = tmp_path / "broken.png"
    corrupt.write_bytes(b"not-an-image")

    result = inspect_image(corrupt)

    assert result.status == "failed"
    assert result.error_code == "decode_failed"

