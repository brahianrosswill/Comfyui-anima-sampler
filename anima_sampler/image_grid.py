"""Image grid helpers used by comparison and experiment nodes."""

from __future__ import annotations

import math
from typing import Any, Iterable


def build_labeled_comparison_grid(
    images: Iterable[Any],
    labels: list[str],
    *,
    columns: int,
    label_height: int,
    gap: int,
):
    """Return a ComfyUI IMAGE tensor containing labeled comparison tiles."""

    image_list = list(images)
    if not image_list:
        raise ValueError("images must contain at least one image tensor")
    if len(image_list) != len(labels):
        raise ValueError("images and labels must have the same length")
    if columns < 1:
        raise ValueError("columns must be at least 1")

    torch = _import_torch()
    pil_images = [_tensor_to_pil(image) for image in image_list]
    tile_w = max(image.width for image in pil_images)
    tile_h = max(image.height for image in pil_images)
    columns = min(columns, len(pil_images))
    rows = math.ceil(len(pil_images) / columns)

    labeled_tiles = [
        _make_labeled_tile(image, label, tile_w=tile_w, tile_h=tile_h, label_height=label_height)
        for image, label in zip(pil_images, labels)
    ]

    grid_w = columns * tile_w + max(0, columns - 1) * gap
    grid_h = rows * (tile_h + label_height) + max(0, rows - 1) * gap

    from PIL import Image

    grid = Image.new("RGB", (grid_w, grid_h), (24, 24, 24))
    for index, tile in enumerate(labeled_tiles):
        row = index // columns
        col = index % columns
        x = col * (tile_w + gap)
        y = row * (tile_h + label_height + gap)
        grid.paste(tile, (x, y))

    tensor = _pil_to_tensor(grid, torch)
    return tensor


def _make_labeled_tile(image, label: str, *, tile_w: int, tile_h: int, label_height: int):
    from PIL import Image, ImageDraw, ImageFont

    tile = Image.new("RGB", (tile_w, tile_h + label_height), (18, 18, 18))
    x = (tile_w - image.width) // 2
    y = (tile_h - image.height) // 2
    tile.paste(image, (x, y))

    draw = ImageDraw.Draw(tile)
    draw.rectangle((0, tile_h, tile_w, tile_h + label_height), fill=(28, 28, 28))
    font = ImageFont.load_default()
    lines = _wrap_label(label, max_chars=max(20, tile_w // 8))
    text_y = tile_h + 8
    for line in lines:
        if text_y > tile_h + label_height - 12:
            break
        draw.text((10, text_y), line, fill=(238, 238, 238), font=font)
        text_y += 14
    return tile


def _wrap_label(text: str, *, max_chars: int) -> list[str]:
    lines: list[str] = []
    for raw_line in text.splitlines():
        current = ""
        for word in raw_line.split():
            if not current:
                current = word
            elif len(current) + 1 + len(word) <= max_chars:
                current += " " + word
            else:
                lines.append(current)
                current = word
        if current:
            lines.append(current)
    return lines


def _tensor_to_pil(tensor):
    import numpy as np
    from PIL import Image

    if len(tensor.shape) == 4:
        tensor = tensor[0]
    array = tensor.detach().cpu().float().clamp(0, 1).numpy()
    array = (array * 255.0).round().astype(np.uint8)
    return Image.fromarray(array)


def _pil_to_tensor(image, torch):
    import numpy as np

    array = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def _import_torch():
    import torch

    return torch
