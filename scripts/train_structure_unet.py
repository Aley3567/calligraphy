from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from common_images import make_grid
from structure_dataset import StructureGlyphDataset, make_structure_datasets
from structure_losses import LossConfig, branch1_losses, tensor_dict_to_float
from structure_unet import (
    StructureInkUNetBranch1,
    StructureInkUNetBranch1LowpassInput,
    StructureInkUNetBranch1NoHighresSkip,
    StructureInkUNetBranch1WeakHighresSkip,
    StructureMultiHeadUNet,
)


TENSOR_KEYS = {
    "input",
    "content_gray",
    "content_mask",
    "content_skeleton",
    "content_distance",
    "content_hole",
    "target_gray",
    "target_mask",
    "target_skeleton",
    "target_edge",
    "target_hole",
    "target_bbox",
}
LOSS_KEYS = [
    "total",
    "visual_proxy",
    "gray",
    "mask",
    "edge",
    "edge_dice",
    "final_dice",
    "final_precision",
    "final_recall",
    "final_area",
    "outside_target",
    "skeleton_cover",
    "density",
    "hole",
    "target_hole",
    "bbox",
    "bbox_center",
    "bbox_scale",
    "final_bbox",
    "final_bbox_center",
    "final_bbox_scale",
    "ink_target",
    "skeleton_target",
    "edge_mass",
    "anti_content_dice_margin",
]
RAW_LOSS_KEYS = [
    "gray_raw",
    "mask_raw",
    "edge_raw",
    "edge_dice_raw",
    "final_dice_raw",
    "final_precision_raw",
    "final_recall_raw",
    "final_area_raw",
    "outside_target_raw",
    "density_raw",
    "hole_raw",
    "target_hole_raw",
    "bbox_raw",
    "bbox_center_raw",
    "bbox_scale_raw",
    "final_bbox_raw",
    "final_bbox_center_raw",
    "final_bbox_scale_raw",
    "edge_mass_raw",
    "skeleton_cover_raw",
    "gray_full_raw",
    "fg_gray_l1_raw",
    "final_mask_l1_raw",
    "ink_fg_mean_raw",
    "ink_fg_std_raw",
    "target_fg_mean_raw",
    "target_fg_std_raw",
    "ink_ratio_absdiff_raw",
    "ink_target_raw",
    "skeleton_target_raw",
    "anti_content_dice_margin_raw",
    "pred_target_dice_raw",
    "pred_content_dice_raw",
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


def read_manifest_rows(split_dir: Path) -> tuple[list[str], list[dict[str, str]]]:
    rows: list[dict[str, str]] = []
    fieldnames: list[str] | None = None
    for name in ["train_manifest.csv", "val_manifest.csv"]:
        path = split_dir / name
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if fieldnames is None:
                fieldnames = list(reader.fieldnames or [])
            rows.extend(reader)
    return fieldnames or [], rows


def flatten_fixed_chars(spec: dict[str, Any]) -> list[tuple[str, str]]:
    groups = spec.get("groups", {})
    if isinstance(groups, dict):
        return [(str(group), str(char)) for group, chars in groups.items() for char in chars]
    return [("fixed", str(char)) for char in spec.get("chars", [])]


def build_fixed_manifest(data_dir: Path, split_dir: Path, out_dir: Path, fixed_sample_json: Path) -> Path:
    spec = load_json(fixed_sample_json)
    fieldnames, rows = read_manifest_rows(split_dir)
    by_char: dict[str, dict[str, str]] = {}
    for row in rows:
        char = row.get("char", "")
        if char and char not in by_char:
            by_char[char] = row

    replacements = {str(k): str(v) for k, v in spec.get("replacements", {}).items()}
    selected: list[dict[str, str]] = []
    records: list[dict[str, str]] = []
    used_sample_ids: set[str] = set()
    requested_items = flatten_fixed_chars(spec)
    for group, requested in requested_items:
        selected_char = requested if requested in by_char else replacements.get(requested, requested)
        status = "requested" if selected_char == requested else "replacement"
        if selected_char not in by_char:
            records.append({"group": group, "requested": requested, "selected": "", "sample_id": "", "status": "missing"})
            continue
        row = by_char[selected_char]
        sample_id = row.get("sample_id", "")
        if sample_id in used_sample_ids:
            records.append({"group": group, "requested": requested, "selected": selected_char, "sample_id": sample_id, "status": "duplicate_skipped"})
            continue
        selected.append(row)
        used_sample_ids.add(sample_id)
        records.append({"group": group, "requested": requested, "selected": selected_char, "sample_id": sample_id, "status": status})

    fixed_dir = out_dir / "fixed_gateA"
    fixed_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = fixed_dir / "gateA_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)
    (fixed_dir / "fixed_sample_selection.json").write_text(
        json.dumps(
            {
                "fixed_sample_json": str(fixed_sample_json.expanduser().resolve()),
                "data_dir": str(data_dir),
                "split_dir": str(split_dir),
                "requested_count": len(requested_items),
                "selected_count": len(selected),
                "records": records,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if not selected:
        raise ValueError(f"no fixed samples selected from {fixed_sample_json}")
    return manifest_path


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if key in TENSOR_KEYS and torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=device.type == "cuda")
        else:
            moved[key] = value
    return moved


def tensor_to_ink_image(tensor: torch.Tensor) -> Image.Image:
    arr = tensor.detach().cpu().squeeze().float().numpy()
    arr = (255.0 - arr.clip(0.0, 1.0) * 255.0).astype(np.uint8)
    return Image.fromarray(arr)


def route_outputs(outputs: dict[str, torch.Tensor], config: dict[str, Any]) -> dict[str, torch.Tensor]:
    routed = dict(outputs)
    if bool(config.get("final_from_mask", False)):
        routed["final_ink"] = routed["mask"]
        return routed
    if not bool(config.get("detach_mask_for_final", False)):
        return outputs
    routed["final_ink"] = routed["mask"].detach() * routed["ink"]
    return routed


def build_model_input(batch: dict[str, Any], config: dict[str, Any]) -> torch.Tensor:
    tensors = [batch["input"]]
    for key in [str(value) for value in config.get("extra_input_keys", [])]:
        if key not in batch:
            raise KeyError(f"extra input key not found in batch: {key}")
        value = batch[key]
        if not torch.is_tensor(value):
            raise TypeError(f"extra input key must be a tensor: {key}")
        if value.ndim == 3:
            value = value.unsqueeze(1)
        tensors.append(value)
    return torch.cat(tensors, dim=1)


def create_model(config: dict[str, Any]) -> nn.Module:
    model_name = str(config.get("model_name", "StructureInkUNetBranch1"))
    kwargs = {
        "input_channels": int(config.get("input_channels", 5)),
        "base": int(config.get("base_channels", 64)),
    }
    if model_name == "StructureInkUNetBranch1":
        return StructureInkUNetBranch1(**kwargs)
    if model_name == "StructureInkUNetBranch1LowpassInput":
        return StructureInkUNetBranch1LowpassInput(**kwargs)
    if model_name == "StructureInkUNetBranch1WeakHighresSkip":
        return StructureInkUNetBranch1WeakHighresSkip(**kwargs)
    if model_name == "StructureInkUNetBranch1NoHighresSkip":
        return StructureInkUNetBranch1NoHighresSkip(**kwargs)
    if model_name == "StructureMultiHeadUNet":
        return StructureMultiHeadUNet(**kwargs)
    raise ValueError(f"unsupported model_name: {model_name}")


def set_trainable_scope(model: nn.Module, config: dict[str, Any]) -> list[torch.nn.Parameter]:
    scope = str(config.get("trainable_scope", "all"))
    for param in model.parameters():
        param.requires_grad = True
    if scope == "all":
        return [param for param in model.parameters() if param.requires_grad]
    if scope == "ink_head_only":
        if not hasattr(model, "ink_head"):
            raise ValueError("trainable_scope=ink_head_only requires model.ink_head")
        for param in model.parameters():
            param.requires_grad = False
        for param in model.ink_head.parameters():
            param.requires_grad = True
        return [param for param in model.ink_head.parameters() if param.requires_grad]
    raise ValueError(f"unsupported trainable_scope: {scope}")


def set_training_mode(model: nn.Module, config: dict[str, Any]) -> None:
    if str(config.get("trainable_scope", "all")) == "ink_head_only":
        model.eval()
        model.ink_head.train()
    else:
        model.train()


def load_init_checkpoint(model: nn.Module, checkpoint: Path, device: torch.device) -> dict[str, Any]:
    ckpt = torch.load(checkpoint.expanduser().resolve(), map_location=device)
    if "model" not in ckpt:
        raise KeyError(f"checkpoint must contain model state: {checkpoint}")
    model.load_state_dict(ckpt["model"])
    return {"path": str(checkpoint.expanduser().resolve()), "epoch": int(ckpt.get("epoch", 0)), "model_name": str(ckpt.get("model_name", ""))}


def preview_images_for_sample(
    columns: list[str],
    batch: dict[str, Any],
    outputs: dict[str, torch.Tensor],
    index: int,
) -> list[Image.Image]:
    mapping = {
        "content": batch["input"][index, 0],
        "content_gray": batch["content_gray"][index],
        "pred_mask": outputs["mask"][index],
        "mask": outputs["mask"][index],
        "pred_ink": outputs["ink"][index],
        "ink": outputs["ink"][index],
        "pred_edge": outputs.get("edge_pred", outputs["ink"])[index],
        "edge_pred": outputs.get("edge_pred", outputs["ink"])[index],
        "pred_skeleton": outputs.get("skeleton_pred", outputs["mask"])[index],
        "skeleton_pred": outputs.get("skeleton_pred", outputs["mask"])[index],
        "final_ink": outputs["final_ink"][index],
        "final": outputs["final_ink"][index],
        "target": batch["target_gray"][index],
        "target_gray": batch["target_gray"][index],
        "target_mask": batch["target_mask"][index],
        "target_edge": batch["target_edge"][index],
        "target_skeleton": batch["target_skeleton"][index],
    }
    return [tensor_to_ink_image(mapping[column]) for column in columns]


def save_preview(
    model: nn.Module,
    loader: DataLoader,
    out_path: Path,
    device: torch.device,
    max_items: int,
    config: dict[str, Any],
) -> None:
    model.eval()
    images: list[Image.Image] = []
    records: list[dict[str, str]] = []
    columns = [str(v) for v in config.get("preview_columns", ["content", "pred_mask", "final_ink", "target"])]
    with torch.no_grad():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            model_x = build_model_input(batch, config)
            outputs = route_outputs(model(model_x), config)
            take = min(model_x.size(0), max_items - len(images) // len(columns))
            for i in range(take):
                images.extend(preview_images_for_sample(columns, batch, outputs, i))
                records.append(
                    {
                        "sample_id": str(batch.get("sample_id", [""])[i]),
                        "char": str(batch.get("char", [""])[i]),
                        "filter_flags": str(batch.get("filter_flags", [""])[i]),
                    }
                )
            if len(images) >= max_items * len(columns):
                break
    if images:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        make_grid(images, cols=len(columns)).save(out_path)
        out_path.with_suffix(".json").write_text(
            json.dumps({"columns": columns, "records": records}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    set_training_mode(model, config)


def mean_dict(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def finite_report(losses: dict[str, float]) -> dict[str, bool]:
    return {key: bool(np.isfinite(value)) for key, value in losses.items()}


def bbox_iou_from_rows(row: dict[str, str]) -> float:
    cx0 = int(float(row["content_bbox_x0"]))
    cy0 = int(float(row["content_bbox_y0"]))
    cx1 = int(float(row["content_bbox_x1"]))
    cy1 = int(float(row["content_bbox_y1"]))
    tx0 = int(float(row["target_bbox_x0"]))
    ty0 = int(float(row["target_bbox_y0"]))
    tx1 = int(float(row["target_bbox_x1"]))
    ty1 = int(float(row["target_bbox_y1"]))
    cw = max(0, cx1 - cx0)
    ch = max(0, cy1 - cy0)
    tw = max(0, tx1 - tx0)
    th = max(0, ty1 - ty0)
    if cw == 0 or ch == 0 or tw == 0 or th == 0:
        return 0.0
    ix0 = max(cx0, tx0)
    iy0 = max(cy0, ty0)
    ix1 = min(cx1, tx1)
    iy1 = min(cy1, ty1)
    inter = max(0, ix1 - ix0) * max(0, iy1 - iy0)
    return float(inter) / max(float(cw * ch + tw * th - inter), 1.0)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def validate_data_contract(data_dir: Path, split_dir: Path, out_dir: Path, config: dict[str, Any]) -> None:
    contract = config.get("data_contract", {})
    if not isinstance(contract, dict) or not contract:
        return

    train_rows = read_csv_rows(split_dir / "train_manifest.csv")
    val_rows = read_csv_rows(split_dir / "val_manifest.csv")
    rows = train_rows + val_rows
    sample_ids = [row.get("sample_id", "") for row in rows]
    duplicate_count = len(sample_ids) - len(set(sample_ids))
    train_chars = {row.get("char", "") for row in train_rows if row.get("char", "")}
    val_chars = {row.get("char", "") for row in val_rows if row.get("char", "")}
    train_val_char_overlap = sorted(train_chars & val_chars)
    empty_content_count = sum(1 for row in rows if "empty_content" in str(row.get("filter_flags", "")).split("|"))
    scales = sorted({str(row.get("content_font_scale", "")) for row in rows})
    ink_ratios = [float(row["content_ink_ratio"]) / max(float(row["target_ink_ratio"]), 1e-8) for row in rows]
    bbox_ious = [bbox_iou_from_rows(row) for row in rows]
    ink_ratio_gt_1_5 = sum(1 for value in ink_ratios if value > 1.5)
    bbox_iou_lt_0_8 = sum(1 for value in bbox_ious if value < 0.8)
    both_bad = sum(1 for ink_ratio, bbox_iou in zip(ink_ratios, bbox_ious) if ink_ratio > 1.5 and bbox_iou < 0.8)

    report = {
        "data_dir": str(data_dir),
        "split_dir": str(split_dir),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "duplicate_sample_ids": duplicate_count,
        "train_val_char_overlap_count": len(train_val_char_overlap),
        "train_val_char_overlap": train_val_char_overlap[:50],
        "empty_content_count": empty_content_count,
        "content_font_scale_values": scales,
        "ink_ratio_median": float(np.median(ink_ratios)) if ink_ratios else 0.0,
        "ink_ratio_p90": float(np.quantile(ink_ratios, 0.90)) if ink_ratios else 0.0,
        "bbox_iou_median": float(np.median(bbox_ious)) if bbox_ious else 0.0,
        "bbox_iou_p10": float(np.quantile(bbox_ious, 0.10)) if bbox_ious else 0.0,
        "ink_ratio_gt_1_5": ink_ratio_gt_1_5,
        "bbox_iou_lt_0_8": bbox_iou_lt_0_8,
        "ink_ratio_gt_1_5_and_bbox_iou_lt_0_8": both_bad,
        "contract": contract,
        "pass": True,
        "failures": [],
    }

    failures: list[str] = []
    if bool(contract.get("forbid_antialias", True)) and ("antialias" in str(data_dir) or "antialias" in str(split_dir)):
        failures.append("antialias path is forbidden")
    expected_train = int(contract.get("expected_train_rows", 0))
    if expected_train and len(train_rows) != expected_train:
        failures.append(f"train row count mismatch: expected {expected_train}, got {len(train_rows)}")
    expected_val = int(contract.get("expected_val_rows", 0))
    if expected_val and len(val_rows) != expected_val:
        failures.append(f"val row count mismatch: expected {expected_val}, got {len(val_rows)}")
    expected_scale = contract.get("expected_content_font_scale")
    if expected_scale is not None and scales != [str(expected_scale)]:
        failures.append(f"content_font_scale mismatch: expected only {expected_scale}, got {scales}")
    if duplicate_count > int(contract.get("max_duplicate_sample_ids", 0)):
        failures.append(f"duplicate sample ids: {duplicate_count}")
    max_char_overlap = int(contract.get("max_train_val_char_overlap", 10**9))
    if len(train_val_char_overlap) > max_char_overlap:
        failures.append(
            f"train/val char overlap: {len(train_val_char_overlap)} "
            f"(max {max_char_overlap}) examples={train_val_char_overlap[:10]}"
        )
    if empty_content_count > int(contract.get("max_empty_content", 0)):
        failures.append(f"empty content rows: {empty_content_count}")
    if ink_ratio_gt_1_5 > int(contract.get("max_ink_ratio_gt_1_5", 10**9)):
        failures.append(f"ink_ratio>1.5 rows too high: {ink_ratio_gt_1_5}")
    if bbox_iou_lt_0_8 > int(contract.get("max_bbox_iou_lt_0_8", 10**9)):
        failures.append(f"bbox_iou<0.8 rows too high: {bbox_iou_lt_0_8}")
    if both_bad > int(contract.get("max_ink_ratio_gt_1_5_and_bbox_iou_lt_0_8", 10**9)):
        failures.append(f"combined ink/bbox bad rows too high: {both_bad}")

    report["failures"] = failures
    report["pass"] = not failures
    (out_dir / "data_contract_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if failures:
        raise ValueError("data contract failed: " + "; ".join(failures))


def run_eval(
    model: nn.Module,
    loader: DataLoader,
    loss_config: LossConfig,
    device: torch.device,
    config: dict[str, Any],
) -> tuple[dict[str, float], list[str]]:
    model.eval()
    rows: list[dict[str, float]] = []
    warnings: list[str] = []
    with torch.no_grad():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            model_x = build_model_input(batch, config)
            outputs = route_outputs(model(model_x), config)
            _total, losses, batch_warnings = branch1_losses(outputs, batch, loss_config)
            rows.append(tensor_dict_to_float(losses))
            warnings.extend(batch_warnings)
    set_training_mode(model, config)
    return mean_dict(rows), sorted(set(warnings))


def write_calibration_report(
    path: Path,
    args: argparse.Namespace,
    config: dict[str, Any],
    train_set: StructureGlyphDataset,
    val_set: StructureGlyphDataset,
    batch: dict[str, Any],
    losses: dict[str, float],
    warnings: list[str],
    device: torch.device,
) -> None:
    report = {
        "mode": "calibrate",
        "model_name": str(config.get("model_name", "StructureInkUNetBranch1")),
        "device": str(device),
        "data_dir": str(args.data_dir.expanduser().resolve()),
        "split_dir": str(args.split_dir.expanduser().resolve()),
        "fixed_sample_json": str(args.fixed_sample_json.expanduser().resolve()) if args.fixed_sample_json else "",
        "train_items": len(train_set),
        "val_items": len(val_set),
        "batch_shapes": {
            "input": list(batch["input"].shape),
            "model_input": list(build_model_input(batch, config).shape),
            "content_hole": list(batch["content_hole"].shape),
            "target_gray": list(batch["target_gray"].shape),
            "target_mask": list(batch["target_mask"].shape),
            "target_edge": list(batch["target_edge"].shape),
            "target_hole": list(batch["target_hole"].shape),
            "target_bbox": list(batch["target_bbox"].shape),
        },
        "losses": losses,
        "finite": finite_report(losses),
        "warnings": sorted(set(warnings)),
        "config": config,
    }
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: dict[str, Any],
    train_losses: list[dict[str, float]],
    val_losses: list[dict[str, float]],
    best: dict[str, float] | None = None,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "config": config,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best": best or {},
            "model_name": str(config.get("model_name", "StructureInkUNetBranch1")),
        },
        path,
    )


def make_log_writer(path: Path) -> tuple[Any, csv.DictWriter]:
    fieldnames = ["mode", "epoch", "split", "device"] + LOSS_KEYS + RAW_LOSS_KEYS + ["warnings"]
    f = path.open("w", encoding="utf-8", newline="")
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    return f, writer


def log_row(
    writer: csv.DictWriter,
    mode: str,
    epoch: int,
    split: str,
    device: torch.device,
    losses: dict[str, float],
    warnings: list[str],
) -> None:
    row: dict[str, Any] = {"mode": mode, "epoch": epoch, "split": split, "device": str(device), "warnings": ";".join(sorted(set(warnings)))}
    for key in LOSS_KEYS + RAW_LOSS_KEYS:
        row[key] = losses.get(key, "")
    writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or calibrate the Branch 1 structure-aware U-Net.")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--split-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--batch-size", default=4, type=int)
    parser.add_argument("--num-workers", default=0, type=int)
    parser.add_argument("--max-train-items", default=0, type=int)
    parser.add_argument("--max-val-items", default=0, type=int)
    parser.add_argument("--fixed-sample-json", default=None, type=Path)
    parser.add_argument("--calibrate-only", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--lr", default=0.0, type=float)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--init-checkpoint", default="", type=str)
    args = parser.parse_args()

    config = load_json(args.config)
    seed = args.seed or int(config.get("seed", 42))
    set_seed(seed)

    out_dir = args.out_dir.expanduser().resolve()
    checkpoint_dir = out_dir / "checkpoints"
    preview_dir = out_dir / "previews"
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device(args.device)
    data_dir = args.data_dir.expanduser().resolve()
    split_dir = args.split_dir.expanduser().resolve()
    validate_data_contract(data_dir, split_dir, out_dir, config)
    if args.fixed_sample_json:
        fixed_manifest = build_fixed_manifest(data_dir, split_dir, out_dir, args.fixed_sample_json.expanduser().resolve())
        train_set = StructureGlyphDataset(data_dir, fixed_manifest, args.max_train_items)
        val_set = StructureGlyphDataset(data_dir, fixed_manifest, args.max_val_items)
    elif config.get("fixed_sample_json"):
        fixed_manifest = build_fixed_manifest(data_dir, split_dir, out_dir, Path(str(config["fixed_sample_json"])).expanduser().resolve())
        train_set = StructureGlyphDataset(data_dir, fixed_manifest, args.max_train_items)
        val_set = StructureGlyphDataset(data_dir, fixed_manifest, args.max_val_items)
    else:
        train_set, val_set = make_structure_datasets(
            data_dir,
            split_dir,
            max_train_items=args.max_train_items,
            max_val_items=args.max_val_items,
        )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=not args.calibrate_only,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = create_model(config).to(device)
    if args.init_checkpoint:
        init_meta = load_init_checkpoint(model, Path(args.init_checkpoint), device)
        config["init_checkpoint"] = init_meta
    trainable_params = set_trainable_scope(model, config)
    if not trainable_params:
        raise RuntimeError("no trainable parameters selected")
    lr = args.lr or float(config.get("learning_rate", 2e-4))
    betas = tuple(float(v) for v in config.get("betas", [0.5, 0.999]))
    optimizer = torch.optim.Adam(trainable_params, lr=lr, betas=betas)
    loss_config = LossConfig(
        weights={str(k): float(v) for k, v in config.get("loss_weights", {}).items()},
        density_window=int(config.get("density_window", 16)),
        bbox_eps=float(config.get("bbox_eps", 1e-6)),
        mask_blur_kernel=int(config.get("mask_blur_kernel", 1)),
        edge_source=str(config.get("edge_source", "mask")),
        ink_target_key=str(config.get("ink_target_key", "")),
        skeleton_target_key=str(config.get("skeleton_target_key", "")),
        gray_loss=str(config.get("gray_loss", "l1")),
        foreground_gray_weight=float(config.get("foreground_gray_weight", 0.3)),
        anti_content_margin=float(config.get("anti_content_margin", 0.0)),
        visual_proxy_weights={str(k): float(v) for k, v in config.get("visual_proxy_weights", {}).items()},
    )

    log_f, log_writer = make_log_writer(out_dir / "train_log.csv")
    train_losses: list[dict[str, float]] = []
    val_losses: list[dict[str, float]] = []
    best = {
        "val_total": float("inf"),
        "visual_proxy": float("inf"),
        "appearance_proxy": float("inf"),
        "val_total_epoch": 0,
        "visual_proxy_epoch": 0,
        "appearance_proxy_epoch": 0,
    }
    checkpoint_epochs = {int(v) for v in config.get("checkpoint_epochs", [])}
    checkpoint_every = int(config.get("checkpoint_every", 0))
    try:
        model.eval()
        with torch.no_grad():
            raw_batch = next(iter(train_loader))
            calibration_batch = move_batch(raw_batch, device)
            calibration_x = build_model_input(calibration_batch, config)
            calibration_outputs = route_outputs(model(calibration_x), config)
            _calibration_total, calibration_losses_tensor, calibration_warnings = branch1_losses(
                calibration_outputs,
                calibration_batch,
                loss_config,
            )
            calibration_losses = tensor_dict_to_float(calibration_losses_tensor)
        set_training_mode(model, config)
        write_calibration_report(
            out_dir / "loss_scale_report.json",
            args,
            config,
            train_set,
            val_set,
            calibration_batch,
            calibration_losses,
            calibration_warnings,
            device,
        )

        if args.calibrate_only:
            save_checkpoint(checkpoint_dir / "last.pt", model, optimizer, 0, config, train_losses, val_losses, best)
            save_preview(model, val_loader, preview_dir / "calibrate_batch.png", device, int(config.get("preview_items", 6)), config)
            log_row(log_writer, "calibrate", 0, "train", device, calibration_losses, calibration_warnings)
            print(
                json.dumps(
                    {"mode": "calibrate", "out_dir": str(out_dir), "losses": calibration_losses, "warnings": calibration_warnings},
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return

        if bool(config.get("save_initial_checkpoint", False)):
            save_checkpoint(checkpoint_dir / "epoch_000_before_probe.pt", model, optimizer, 0, config, train_losses, val_losses, best)
            save_preview(model, val_loader, preview_dir / "epoch_000_before_probe.png", device, int(config.get("preview_items", 6)), config)

        for epoch in range(1, args.epochs + 1):
            set_training_mode(model, config)
            epoch_rows: list[dict[str, float]] = []
            epoch_warnings: list[str] = []
            for raw_batch in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"):
                batch = move_batch(raw_batch, device)
                model_x = build_model_input(batch, config)
                outputs = route_outputs(model(model_x), config)
                total, losses_tensor, warnings = branch1_losses(outputs, batch, loss_config)
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                optimizer.step()
                epoch_rows.append(tensor_dict_to_float(losses_tensor))
                epoch_warnings.extend(warnings)

            train_mean = mean_dict(epoch_rows)
            val_mean, val_warnings = run_eval(model, val_loader, loss_config, device, config)
            train_losses.append(train_mean)
            val_losses.append(val_mean)
            log_row(log_writer, "train", epoch, "train", device, train_mean, epoch_warnings)
            log_row(log_writer, "train", epoch, "val", device, val_mean, val_warnings)
            log_f.flush()

            val_total = float(val_mean.get("total", float("inf")))
            visual_proxy = float(val_mean.get("visual_proxy", float("inf")))
            save_best_val_total = val_total < best["val_total"]
            save_best_visual_proxy = visual_proxy < best["visual_proxy"]
            save_best_appearance = bool(config.get("save_appearance_checkpoint", False)) and visual_proxy < best["appearance_proxy"]
            if save_best_val_total:
                best["val_total"] = val_total
                best["val_total_epoch"] = epoch
            if save_best_visual_proxy:
                best["visual_proxy"] = visual_proxy
                best["visual_proxy_epoch"] = epoch
            if save_best_appearance:
                best["appearance_proxy"] = visual_proxy
                best["appearance_proxy_epoch"] = epoch

            if save_best_val_total:
                save_checkpoint(checkpoint_dir / "best_val_total.pt", model, optimizer, epoch, config, train_losses, val_losses, best)
            if save_best_visual_proxy:
                save_checkpoint(checkpoint_dir / "best_visual_proxy.pt", model, optimizer, epoch, config, train_losses, val_losses, best)
            if save_best_appearance:
                save_checkpoint(checkpoint_dir / "best_appearance_proxy.pt", model, optimizer, epoch, config, train_losses, val_losses, best)

            save_checkpoint(checkpoint_dir / "last.pt", model, optimizer, epoch, config, train_losses, val_losses, best)
            if epoch in checkpoint_epochs or (checkpoint_every > 0 and epoch % checkpoint_every == 0):
                save_checkpoint(checkpoint_dir / f"epoch_{epoch:03d}.pt", model, optimizer, epoch, config, train_losses, val_losses, best)
            save_preview(model, val_loader, preview_dir / f"epoch_{epoch:03d}.png", device, int(config.get("preview_items", 6)), config)
            print(
                f"epoch={epoch} train_total={train_mean.get('total', float('nan')):.6f} "
                f"val_total={val_mean.get('total', float('nan')):.6f} "
                f"val_proxy={val_mean.get('visual_proxy', float('nan')):.6f} "
                f"val_fg_gray={val_mean.get('fg_gray_l1_raw', float('nan')):.6f} "
                f"val_ink_std={val_mean.get('ink_fg_std_raw', float('nan')):.6f} "
                f"val_final_mask_l1={val_mean.get('final_mask_l1_raw', float('nan')):.6f} "
                f"best_val_epoch={int(best['val_total_epoch'])} best_proxy_epoch={int(best['visual_proxy_epoch'])}"
            )
    finally:
        log_f.close()


if __name__ == "__main__":
    main()
