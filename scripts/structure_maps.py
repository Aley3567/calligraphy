from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

try:
    from scipy import ndimage
except Exception:  # noqa: BLE001
    ndimage = None


@dataclass(frozen=True)
class BinaryStats:
    ink_ratio: float
    bbox_x0: int
    bbox_y0: int
    bbox_x1: int
    bbox_y1: int
    bbox_w: int
    bbox_h: int
    bbox_cx: float
    bbox_cy: float
    component_count: int
    hole_count: int
    hole_ratio: float
    skeleton_ratio: float
    edge_ratio: float
    local_density_mean: float
    local_density_max: float


def load_gray(path: Path) -> Image.Image:
    return Image.open(path).convert("L")


def image_to_gray_array(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L"), dtype=np.uint8)


def binary_ink_mask(img: Image.Image, threshold: int = 220) -> np.ndarray:
    return image_to_gray_array(img) < threshold


def bool_to_image(mask: np.ndarray) -> Image.Image:
    return Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L")


def float_to_image(arr: np.ndarray) -> Image.Image:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.size == 0:
        return Image.new("L", (1, 1), 0)
    max_value = float(arr.max())
    if max_value > 0:
        arr = arr / max_value
    return Image.fromarray(np.clip(arr * 255.0, 0, 255).astype(np.uint8), mode="L")


def save_map(mask_or_arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if mask_or_arr.dtype == bool:
        bool_to_image(mask_or_arr).save(path)
    else:
        float_to_image(mask_or_arr).save(path)


def bbox_from_mask(mask: np.ndarray) -> tuple[int, int, int, int]:
    if not mask.any():
        return 0, 0, 0, 0
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def erode(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    out = np.ones(mask.shape, dtype=bool)
    for dy in range(3):
        for dx in range(3):
            out &= padded[dy : dy + mask.shape[0], dx : dx + mask.shape[1]]
    return out


def edge_map(mask: np.ndarray) -> np.ndarray:
    return mask & ~erode(mask)


def connected_component_count(mask: np.ndarray, min_area: int = 1) -> int:
    seen = np.zeros(mask.shape, dtype=bool)
    count = 0
    height, width = mask.shape
    for y in range(height):
        for x in range(width):
            if not mask[y, x] or seen[y, x]:
                continue
            area = 0
            q: deque[tuple[int, int]] = deque([(y, x)])
            seen[y, x] = True
            while q:
                cy, cx = q.popleft()
                area += 1
                for ny in range(max(0, cy - 1), min(height, cy + 2)):
                    for nx in range(max(0, cx - 1), min(width, cx + 2)):
                        if mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            q.append((ny, nx))
            if area >= min_area:
                count += 1
    return count


def hole_map(mask: np.ndarray, min_area: int = 4) -> np.ndarray:
    background = ~mask
    seen = np.zeros(mask.shape, dtype=bool)
    holes = np.zeros(mask.shape, dtype=bool)
    height, width = mask.shape
    for y in range(height):
        for x in range(width):
            if not background[y, x] or seen[y, x]:
                continue
            pixels: list[tuple[int, int]] = []
            touches_border = False
            q: deque[tuple[int, int]] = deque([(y, x)])
            seen[y, x] = True
            while q:
                cy, cx = q.popleft()
                pixels.append((cy, cx))
                if cy in {0, height - 1} or cx in {0, width - 1}:
                    touches_border = True
                for ny in range(max(0, cy - 1), min(height, cy + 2)):
                    for nx in range(max(0, cx - 1), min(width, cx + 2)):
                        if background[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            q.append((ny, nx))
            if not touches_border and len(pixels) >= min_area:
                for py, px in pixels:
                    holes[py, px] = True
    return holes


def zhang_suen_skeleton(mask: np.ndarray, max_iter: int = 100) -> np.ndarray:
    skel = mask.astype(bool).copy()
    if min(skel.shape) < 3:
        return skel

    def shifted(arr: np.ndarray) -> tuple[np.ndarray, ...]:
        padded = np.pad(arr, 1, mode="constant", constant_values=False)
        h, w = arr.shape
        p2 = padded[0:h, 1 : w + 1]
        p3 = padded[0:h, 2 : w + 2]
        p4 = padded[1 : h + 1, 2 : w + 2]
        p5 = padded[2 : h + 2, 2 : w + 2]
        p6 = padded[2 : h + 2, 1 : w + 1]
        p7 = padded[2 : h + 2, 0:w]
        p8 = padded[1 : h + 1, 0:w]
        p9 = padded[0:h, 0:w]
        return p2, p3, p4, p5, p6, p7, p8, p9

    for _ in range(max_iter):
        changed = False
        for step in (0, 1):
            p2, p3, p4, p5, p6, p7, p8, p9 = shifted(skel)
            n_sum = p2.astype(np.uint8) + p3 + p4 + p5 + p6 + p7 + p8 + p9
            transitions = (
                (~p2 & p3).astype(np.uint8)
                + (~p3 & p4)
                + (~p4 & p5)
                + (~p5 & p6)
                + (~p6 & p7)
                + (~p7 & p8)
                + (~p8 & p9)
                + (~p9 & p2)
            )
            common = skel & (n_sum >= 2) & (n_sum <= 6) & (transitions == 1)
            if step == 0:
                remove = common & ~(p2 & p4 & p6) & ~(p4 & p6 & p8)
            else:
                remove = common & ~(p2 & p4 & p8) & ~(p2 & p6 & p8)
            if remove.any():
                changed = True
                skel[remove] = False
        if not changed:
            break
    return skel


def distance_to_mask(mask: np.ndarray) -> np.ndarray:
    if ndimage is not None:
        return ndimage.distance_transform_edt(~mask).astype(np.float32)

    height, width = mask.shape
    inf = height + width + 1
    dist = np.where(mask, 0.0, float(inf)).astype(np.float32)
    root2 = float(np.sqrt(2.0))
    for y in range(height):
        for x in range(width):
            best = dist[y, x]
            if y > 0:
                best = min(best, dist[y - 1, x] + 1.0)
            if x > 0:
                best = min(best, dist[y, x - 1] + 1.0)
            if y > 0 and x > 0:
                best = min(best, dist[y - 1, x - 1] + root2)
            if y > 0 and x + 1 < width:
                best = min(best, dist[y - 1, x + 1] + root2)
            dist[y, x] = best
    for y in range(height - 1, -1, -1):
        for x in range(width - 1, -1, -1):
            best = dist[y, x]
            if y + 1 < height:
                best = min(best, dist[y + 1, x] + 1.0)
            if x + 1 < width:
                best = min(best, dist[y, x + 1] + 1.0)
            if y + 1 < height and x + 1 < width:
                best = min(best, dist[y + 1, x + 1] + root2)
            if y + 1 < height and x > 0:
                best = min(best, dist[y + 1, x - 1] + root2)
            dist[y, x] = best
    return dist


def foreground_distance(mask: np.ndarray) -> np.ndarray:
    if ndimage is not None:
        return ndimage.distance_transform_edt(mask).astype(np.float32)
    return np.where(mask, distance_to_mask(~mask), 0.0)


def local_density(mask: np.ndarray, window: int = 16) -> np.ndarray:
    if ndimage is not None:
        return ndimage.uniform_filter(mask.astype(np.float32), size=window, mode="constant")
    arr = mask.astype(np.float32)
    kernel = Image.fromarray((arr * 255).astype(np.uint8), mode="L")
    # BoxBlur radius is a fast local mean approximation available in Pillow.
    blurred = kernel.filter(ImageFilter.BoxBlur(max(1, window // 2)))
    return np.asarray(blurred, dtype=np.float32) / 255.0


def skeleton_recall_precision(reference_skel: np.ndarray, candidate_mask: np.ndarray, tolerance: float = 2.0) -> tuple[float, float]:
    if not reference_skel.any():
        return 1.0, 1.0 if not candidate_mask.any() else 0.0
    if not candidate_mask.any():
        return 0.0, 0.0
    ref_to_candidate = distance_to_mask(candidate_mask)
    candidate_skel = zhang_suen_skeleton(candidate_mask)
    cand_to_ref = distance_to_mask(reference_skel)
    recall = float((ref_to_candidate[reference_skel] <= tolerance).mean())
    precision = float((cand_to_ref[candidate_skel] <= tolerance).mean()) if candidate_skel.any() else 0.0
    return recall, precision


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    union = a | b
    if not union.any():
        return 1.0
    return float((a & b).sum() / union.sum())


def summarize_binary(mask: np.ndarray) -> BinaryStats:
    x0, y0, x1, y1 = bbox_from_mask(mask)
    bbox_w = x1 - x0
    bbox_h = y1 - y0
    holes = hole_map(mask)
    skel = zhang_suen_skeleton(mask)
    edges = edge_map(mask)
    density = local_density(mask)
    return BinaryStats(
        ink_ratio=float(mask.mean()),
        bbox_x0=x0,
        bbox_y0=y0,
        bbox_x1=x1,
        bbox_y1=y1,
        bbox_w=bbox_w,
        bbox_h=bbox_h,
        bbox_cx=float((x0 + x1) / 2.0) if bbox_w else 0.0,
        bbox_cy=float((y0 + y1) / 2.0) if bbox_h else 0.0,
        component_count=connected_component_count(mask, min_area=4),
        hole_count=connected_component_count(holes, min_area=4),
        hole_ratio=float(holes.mean()),
        skeleton_ratio=float(skel.mean()),
        edge_ratio=float(edges.mean()),
        local_density_mean=float(density.mean()),
        local_density_max=float(density.max()) if density.size else 0.0,
    )


def stats_to_dict(prefix: str, stats: BinaryStats) -> dict[str, float | int]:
    return {f"{prefix}_{key}": value for key, value in stats.__dict__.items()}
