"""VisionForge offline image engine."""

from .compositor import GenerationPolicy, GenerationSpec, generate_batch, render_composite
from .quality import QualityPolicy, hamming_distance, inspect_image, inspect_images

__all__ = [
    "GenerationPolicy",
    "GenerationSpec",
    "QualityPolicy",
    "generate_batch",
    "hamming_distance",
    "inspect_image",
    "inspect_images",
    "render_composite",
]

__version__ = "0.1.0"

