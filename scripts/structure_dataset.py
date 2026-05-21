from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def _load_luma(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def load_ink(path: Path) -> torch.Tensor:
    return torch.from_numpy(1.0 - _load_luma(path)).float()


def load_map(path: Path) -> torch.Tensor:
    return torch.from_numpy(_load_luma(path)).float()


def to_float(row: dict[str, str], key: str, default: float = 0.0) -> float:
    value = row.get(key, "")
    return float(value) if value not in {"", None} else default


class StructureGlyphDataset(Dataset):
    def __init__(self, data_dir: Path, manifest_path: Path, max_items: int = 0):
        self.data_dir = data_dir.expanduser().resolve()
        self.manifest_path = manifest_path.expanduser().resolve()
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"manifest not found: {self.manifest_path}")
        with self.manifest_path.open("r", encoding="utf-8", newline="") as f:
            self.rows = list(csv.DictReader(f))
        if max_items > 0:
            self.rows = self.rows[:max_items]
        if not self.rows:
            raise ValueError(f"empty manifest: {self.manifest_path}")
        self._validate_manifest_stats()

    def __len__(self) -> int:
        return len(self.rows)

    def _path(self, row: dict[str, str], key: str) -> Path:
        value = row.get(key)
        if not value:
            raise KeyError(f"missing manifest column: {key}")
        return self.data_dir / value

    def _validate_manifest_stats(self) -> None:
        ratio_tolerance = 1e-6
        bbox_tolerance = 0.5
        for index, row in enumerate(self.rows):
            for prefix in ("content", "target"):
                mask_path = self._path(row, f"{prefix}_mask_path")
                if not mask_path.exists():
                    raise FileNotFoundError(f"{prefix} mask not found for manifest row {index}: {mask_path}")
                mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8) > 127
                ink_key = f"{prefix}_ink_ratio"
                if ink_key in row and row[ink_key] not in {"", None}:
                    actual = float(mask.mean())
                    expected = to_float(row, ink_key)
                    if abs(actual - expected) > ratio_tolerance:
                        raise ValueError(
                            f"manifest/data_dir mismatch at row {index} sample={row.get('sample_id', '')} "
                            f"{ink_key}: manifest={expected:.9f} actual={actual:.9f} "
                            f"data_dir={self.data_dir} manifest={self.manifest_path}"
                        )

                bbox_keys = [f"{prefix}_bbox_x0", f"{prefix}_bbox_y0", f"{prefix}_bbox_x1", f"{prefix}_bbox_y1"]
                if all(key in row and row[key] not in {"", None} for key in bbox_keys):
                    if mask.any():
                        ys, xs = np.where(mask)
                        actual_bbox = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
                    else:
                        actual_bbox = [0.0, 0.0, 0.0, 0.0]
                    expected_bbox = [to_float(row, key) for key in bbox_keys]
                    if any(abs(a - e) > bbox_tolerance for a, e in zip(actual_bbox, expected_bbox)):
                        raise ValueError(
                            f"manifest/data_dir mismatch at row {index} sample={row.get('sample_id', '')} "
                            f"{prefix}_bbox: manifest={expected_bbox} actual={actual_bbox} "
                            f"data_dir={self.data_dir} manifest={self.manifest_path}"
                        )

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        content_gray = load_ink(self._path(row, "content_path"))
        content_mask = load_map(self._path(row, "content_mask_path"))
        content_skeleton = load_map(self._path(row, "content_skeleton_path"))
        content_distance = load_map(self._path(row, "content_distance_path"))
        content_hole = load_map(self._path(row, "content_hole_path"))

        target_gray = load_ink(self._path(row, "target_path"))
        target_mask = load_map(self._path(row, "target_mask_path"))
        target_skeleton = load_map(self._path(row, "target_skeleton_path"))
        target_distance = load_map(self._path(row, "target_distance_path"))
        target_edge = load_map(self._path(row, "target_edge_path"))
        target_hole = load_map(self._path(row, "target_hole_path"))

        height, width = target_gray.shape
        denom_x = max(float(width - 1), 1.0)
        denom_y = max(float(height - 1), 1.0)
        bbox = torch.tensor(
            [
                to_float(row, "target_bbox_x0") / denom_x,
                to_float(row, "target_bbox_y0") / denom_y,
                max(to_float(row, "target_bbox_x1") - 1.0, 0.0) / denom_x,
                max(to_float(row, "target_bbox_y1") - 1.0, 0.0) / denom_y,
            ],
            dtype=torch.float32,
        ).clamp(0.0, 1.0)

        return {
            "input": torch.stack([content_gray, content_mask, content_skeleton, content_distance, content_hole], dim=0),
            "content_gray": content_gray.unsqueeze(0),
            "content_mask": content_mask.unsqueeze(0),
            "content_skeleton": content_skeleton.unsqueeze(0),
            "content_distance": content_distance.unsqueeze(0),
            "content_hole": content_hole.unsqueeze(0),
            "target_gray": target_gray.unsqueeze(0),
            "target_mask": target_mask.unsqueeze(0),
            "target_skeleton": target_skeleton.unsqueeze(0),
            "target_distance": target_distance.unsqueeze(0),
            "target_edge": target_edge.unsqueeze(0),
            "target_hole": target_hole.unsqueeze(0),
            "target_bbox": bbox,
            "sample_id": row.get("sample_id", ""),
            "char": row.get("char", ""),
            "filter_flags": row.get("filter_flags", ""),
        }


def make_structure_datasets(
    data_dir: Path,
    split_dir: Path,
    max_train_items: int = 0,
    max_val_items: int = 0,
) -> tuple[StructureGlyphDataset, StructureGlyphDataset]:
    split_dir = split_dir.expanduser().resolve()
    train = StructureGlyphDataset(data_dir, split_dir / "train_manifest.csv", max_train_items)
    val = StructureGlyphDataset(data_dir, split_dir / "val_manifest.csv", max_val_items)
    return train, val
