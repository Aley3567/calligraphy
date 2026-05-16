from __future__ import annotations

import argparse
import csv
from collections import deque
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from common_images import make_grid, render_content_char
from train_unet_baseline import UNetGenerator


def image_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("L")).astype(np.float32) / 255.0
    arr = arr * 2.0 - 1.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    arr = tensor.detach().cpu().squeeze(0).numpy()
    arr = ((arr + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def component_count(mask: np.ndarray) -> int:
    seen = np.zeros(mask.shape, dtype=bool)
    count = 0
    height, width = mask.shape
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or seen[y, x]:
                continue
            count += 1
            q: deque[tuple[int, int]] = deque([(y, x)])
            seen[y, x] = True
            while q:
                cy, cx = q.popleft()
                for ny in range(max(0, cy - 1), min(height, cy + 2)):
                    for nx in range(max(0, cx - 1), min(width, cx + 2)):
                        if mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            q.append((ny, nx))
    return count


def metrics(img: Image.Image) -> dict[str, float | int]:
    arr = np.asarray(img.convert("L")).astype(np.float32)
    mask = arr < 220
    ink_ratio = float(mask.mean())
    if mask.any():
        ys, xs = np.where(mask)
        bbox_w = int(xs.max() - xs.min() + 1)
        bbox_h = int(ys.max() - ys.min() + 1)
    else:
        bbox_w = 0
        bbox_h = 0
    return {
        "ink_ratio": ink_ratio,
        "bbox_w": bbox_w,
        "bbox_h": bbox_h,
        "component_count": component_count(mask),
        "mean_gray": float(arr.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fixed eval board and simple structural metrics.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--content-font", required=True, type=Path)
    parser.add_argument("--text-file", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--image-size", default=128, type=int)
    args = parser.parse_args()

    chars = "".join(line.strip() for line in args.text_file.read_text(encoding="utf-8").splitlines())
    chars = "".join(ch for ch in chars if not ch.isspace())
    out_dir = args.out_dir.expanduser().resolve()
    (out_dir / "generated").mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint.expanduser().resolve(), map_location=device)
    state = ckpt.get("model") or ckpt.get("generator")
    if state is None:
        raise KeyError("checkpoint must contain 'model' or 'generator'")
    model = UNetGenerator().to(device)
    model.load_state_dict(state)
    model.eval()

    images: list[Image.Image] = []
    rows: list[dict] = []
    with torch.no_grad():
        for idx, char in enumerate(chars):
            content = render_content_char(char, args.content_font.expanduser().resolve(), args.image_size)
            pred = tensor_to_image(model(image_to_tensor(content).to(device))[0])
            pred.save(out_dir / "generated" / f"{idx:03d}_{char}.png")
            images.extend([content, pred])
            row = {"char": char, **metrics(pred)}
            rows.append(row)

    make_grid(images, cols=2).save(out_dir / "eval_board.png")
    with (out_dir / "quality_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["char", "ink_ratio", "bbox_w", "bbox_h", "component_count", "mean_gray"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()

