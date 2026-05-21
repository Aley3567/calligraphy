from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp", ".tif", ".tiff"}


def iter_image_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def stable_id(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def infer_char_from_path(path: Path) -> str:
    stem = path.stem.strip()
    if len(stem) == 1:
        return stem
    for part in reversed(path.parts):
        part = part.strip()
        if len(part) == 1:
            return part
    return stem[:1] if stem else ""


def open_grayscale(path: Path) -> Image.Image:
    img = Image.open(path)
    if getattr(img, "is_animated", False):
        img.seek(0)
    return img.convert("L")


def normalize_ink_image(img: Image.Image, image_size: int, resize_mode: str = "thumbnail", padding: int = 16) -> Image.Image:
    img = ImageOps.exif_transpose(img).convert("L")
    arr = np.asarray(img).astype(np.float32)

    # Keep black ink on white background. If the image is inverted, fix it.
    border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    if border.mean() < arr.mean():
        arr = 255.0 - arr

    mask = arr < 245
    if mask.any():
        ys, xs = np.where(mask)
        x0, x1 = xs.min(), xs.max() + 1
        y0, y1 = ys.min(), ys.max() + 1
        arr = arr[y0:y1, x0:x1]

    max_side = image_size - padding
    if max_side <= 0:
        raise ValueError(f"padding must be smaller than image_size, got image_size={image_size}, padding={padding}")

    cropped = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if resize_mode == "thumbnail":
        cropped.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    elif resize_mode == "fit":
        scale = min(max_side / float(cropped.width), max_side / float(cropped.height))
        new_size = (max(1, int(round(cropped.width * scale))), max(1, int(round(cropped.height * scale))))
        cropped = cropped.resize(new_size, Image.Resampling.LANCZOS)
    else:
        raise ValueError(f"unsupported resize_mode: {resize_mode}")

    canvas = Image.new("L", (image_size, image_size), 255)
    left = (image_size - cropped.width) // 2
    top = (image_size - cropped.height) // 2
    canvas.paste(cropped, (left, top))
    return canvas


def render_content_char(char: str, font_path: Path, image_size: int, font_scale: float = 0.78) -> Image.Image:
    font_size = max(1, int(image_size * font_scale))
    font = ImageFont.truetype(str(font_path), font_size)
    canvas = Image.new("L", (image_size, image_size), 255)
    draw = ImageDraw.Draw(canvas)
    bbox = draw.textbbox((0, 0), char, font=font)
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    x = (image_size - width) // 2 - bbox[0]
    y = (image_size - height) // 2 - bbox[1]
    draw.text((x, y), char, font=font, fill=0)
    return canvas


def save_pair(content: Image.Image, target: Image.Image, out_dir: Path, sample_id: str) -> None:
    (out_dir / "content").mkdir(parents=True, exist_ok=True)
    (out_dir / "target").mkdir(parents=True, exist_ok=True)
    content.save(out_dir / "content" / f"{sample_id}.png")
    target.save(out_dir / "target" / f"{sample_id}.png")


def make_grid(images: list[Image.Image], cols: int, pad: int = 8, bg: int = 255) -> Image.Image:
    if not images:
        return Image.new("L", (1, 1), bg)
    widths = [img.width for img in images]
    heights = [img.height for img in images]
    cell_w = max(widths)
    cell_h = max(heights)
    rows = (len(images) + cols - 1) // cols
    grid = Image.new("L", (cols * cell_w + (cols + 1) * pad, rows * cell_h + (rows + 1) * pad), bg)
    for idx, img in enumerate(images):
        row, col = divmod(idx, cols)
        x = pad + col * (cell_w + pad) + (cell_w - img.width) // 2
        y = pad + row * (cell_h + pad) + (cell_h - img.height) // 2
        grid.paste(img, (x, y))
    return grid


def find_default_font() -> Optional[Path]:
    candidates = [
        Path("/System/Library/Fonts/Supplemental/Songti.ttc"),
        Path("/System/Library/Fonts/Supplemental/STSong.ttf"),
        Path("/System/Library/Fonts/Supplemental/Kaiti.ttc"),
        Path("/System/Library/Fonts/PingFang.ttc"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return None
