from __future__ import annotations

import argparse
import csv
import json
from collections import deque
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from common_images import make_grid, render_content_char
from structure_maps import (
    binary_ink_mask,
    distance_to_mask,
    hole_map,
    mask_iou,
    skeleton_recall_precision,
    stats_to_dict,
    summarize_binary,
    zhang_suen_skeleton,
)
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


def read_chars(text_file: Path) -> list[tuple[str, str]]:
    chars = "".join(line.strip() for line in text_file.read_text(encoding="utf-8").splitlines())
    return [("fixed", ch) for ch in chars if not ch.isspace()]


def read_groups(groups_file: Path) -> list[tuple[str, str]]:
    data = json.loads(groups_file.read_text(encoding="utf-8"))
    grouped: list[tuple[str, str]] = []
    for group, chars in data.items():
        if isinstance(chars, str):
            iterable = [ch for ch in chars if not ch.isspace()]
        else:
            iterable = chars
        for char in iterable:
            grouped.append((group, str(char)))
    return grouped


def normalized_dt(mask: np.ndarray) -> np.ndarray:
    dist = distance_to_mask(mask)
    max_value = float(dist.max())
    if max_value > 0:
        dist = dist / max_value
    return dist


def structural_metrics(content: Image.Image, pred: Image.Image, threshold: int, tolerance: float) -> dict[str, float | int]:
    content_mask = binary_ink_mask(content, threshold=threshold)
    pred_mask = binary_ink_mask(pred, threshold=threshold)
    content_skel = zhang_suen_skeleton(content_mask)
    content_holes = hole_map(content_mask)
    pred_holes = hole_map(pred_mask)
    recall, precision = skeleton_recall_precision(content_skel, pred_mask, tolerance=tolerance)
    content_stats = summarize_binary(content_mask)
    pred_stats = summarize_binary(pred_mask)
    dt_mae = float(np.abs(normalized_dt(content_mask) - normalized_dt(pred_mask)).mean())
    hole_recall = 1.0
    if content_holes.any():
        hole_recall = float((pred_holes[content_holes]).mean())
    return {
        **stats_to_dict("content", content_stats),
        **stats_to_dict("pred", pred_stats),
        "content_pred_mask_iou": mask_iou(content_mask, pred_mask),
        "content_skeleton_recall": recall,
        "content_skeleton_precision": precision,
        "content_hole_pixel_recall": hole_recall,
        "distance_transform_mae": dt_mae,
        "ink_ratio_delta": float(pred_mask.mean() - content_mask.mean()),
        "hole_count_delta": int(pred_stats.hole_count - content_stats.hole_count),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fixed eval board and simple structural metrics.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--content-font", required=True, type=Path)
    parser.add_argument("--text-file", type=Path)
    parser.add_argument("--groups-file", type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--image-size", default=128, type=int)
    parser.add_argument("--threshold", default=220, type=int)
    parser.add_argument("--skeleton-tolerance", default=2.0, type=float)
    args = parser.parse_args()

    if not args.text_file and not args.groups_file:
        raise ValueError("provide --text-file or --groups-file")
    grouped_chars = read_groups(args.groups_file) if args.groups_file else read_chars(args.text_file)
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
        for idx, (group, char) in enumerate(grouped_chars):
            content = render_content_char(char, args.content_font.expanduser().resolve(), args.image_size)
            pred = tensor_to_image(model(image_to_tensor(content).to(device))[0])
            pred.save(out_dir / "generated" / f"{idx:03d}_{group}_{char}.png")
            images.extend([content, pred])
            row = {
                "group": group,
                "char": char,
                **metrics(pred),
                **structural_metrics(content, pred, args.threshold, args.skeleton_tolerance),
            }
            rows.append(row)

    make_grid(images, cols=2).save(out_dir / "eval_board.png")
    with (out_dir / "quality_metrics.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    summary: dict[str, dict[str, float]] = {}
    for group in sorted({row["group"] for row in rows}):
        group_rows = [row for row in rows if row["group"] == group]
        summary[group] = {
            key: float(np.mean([float(row[key]) for row in group_rows]))
            for key in [
                "pred_ink_ratio",
                "pred_local_density_max",
                "content_skeleton_recall",
                "content_skeleton_precision",
                "content_hole_pixel_recall",
                "distance_transform_mae",
            ]
            if key in group_rows[0]
        }
    with (out_dir / "quality_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
