"""Evidence-removal image transforms."""

from __future__ import annotations

from typing import Iterable

from PIL import Image, ImageFilter


Box = tuple[int, int, int, int]


def mask_regions(
    image: Image.Image,
    boxes: Iterable[Box],
    fill_color: tuple[int, int, int] = (160, 160, 160),
) -> Image.Image:
    """Fill evidence boxes with a solid color."""
    out = image.copy()
    for box in boxes:
        x1, y1, x2, y2 = box
        patch = Image.new("RGB", (max(1, x2 - x1), max(1, y2 - y1)), fill_color)
        out.paste(patch, (x1, y1))
    return out


def blur_regions(image: Image.Image, boxes: Iterable[Box], radius: float = 8.0) -> Image.Image:
    """Apply local Gaussian blur over evidence boxes."""
    out = image.copy()
    for box in boxes:
        crop = out.crop(box)
        out.paste(crop.filter(ImageFilter.GaussianBlur(radius=radius)), box)
    return out
