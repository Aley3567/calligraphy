from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from tqdm import tqdm

from common_images import make_grid
from structure_dataset import StructureGlyphDataset, make_structure_datasets
from structure_losses import LossConfig, branch1_losses, tensor_dict_to_float
from structure_unet import StructureInkUNetBranch1


TENSOR_KEYS = {"input", "target_gray", "target_mask", "target_edge", "target_hole", "target_bbox"}
LOSS_KEYS = ["total", "gray", "mask", "edge", "density", "hole", "bbox"]
RAW_LOSS_KEYS = [f"{key}_raw" for key in LOSS_KEYS if key != "total"]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))


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


def save_preview(
    model: StructureInkUNetBranch1,
    loader: DataLoader,
    out_path: Path,
    device: torch.device,
    max_items: int,
) -> None:
    model.eval()
    images: list[Image.Image] = []
    with torch.no_grad():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            outputs = model(batch["input"])
            take = min(batch["input"].size(0), max_items - len(images) // 4)
            for i in range(take):
                images.extend(
                    [
                        tensor_to_ink_image(batch["input"][i, 0]),
                        tensor_to_ink_image(outputs["mask"][i]),
                        tensor_to_ink_image(outputs["final_ink"][i]),
                        tensor_to_ink_image(batch["target_gray"][i]),
                    ]
                )
            if len(images) >= max_items * 4:
                break
    if images:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        make_grid(images, cols=4).save(out_path)
    model.train()


def mean_dict(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def finite_report(losses: dict[str, float]) -> dict[str, bool]:
    return {key: bool(np.isfinite(value)) for key, value in losses.items()}


def run_eval(
    model: StructureInkUNetBranch1,
    loader: DataLoader,
    loss_config: LossConfig,
    device: torch.device,
) -> tuple[dict[str, float], list[str]]:
    model.eval()
    rows: list[dict[str, float]] = []
    warnings: list[str] = []
    with torch.no_grad():
        for raw_batch in loader:
            batch = move_batch(raw_batch, device)
            outputs = model(batch["input"])
            _total, losses, batch_warnings = branch1_losses(outputs, batch, loss_config)
            rows.append(tensor_dict_to_float(losses))
            warnings.extend(batch_warnings)
    model.train()
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
        "model_name": "StructureInkUNetBranch1",
        "device": str(device),
        "data_dir": str(args.data_dir.expanduser().resolve()),
        "split_dir": str(args.split_dir.expanduser().resolve()),
        "train_items": len(train_set),
        "val_items": len(val_set),
        "batch_shapes": {
            "input": list(batch["input"].shape),
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
    model: StructureInkUNetBranch1,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: dict[str, Any],
    train_losses: list[dict[str, float]],
    val_losses: list[dict[str, float]],
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "config": config,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "model_name": "StructureInkUNetBranch1",
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
    parser.add_argument("--calibrate-only", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--lr", default=0.0, type=float)
    parser.add_argument("--seed", default=0, type=int)
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
    train_set, val_set = make_structure_datasets(
        args.data_dir.expanduser().resolve(),
        args.split_dir.expanduser().resolve(),
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

    model = StructureInkUNetBranch1(
        input_channels=int(config.get("input_channels", 5)),
        base=int(config.get("base_channels", 64)),
    ).to(device)
    lr = args.lr or float(config.get("learning_rate", 2e-4))
    betas = tuple(float(v) for v in config.get("betas", [0.5, 0.999]))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)
    loss_config = LossConfig(
        weights={str(k): float(v) for k, v in config.get("loss_weights", {}).items()},
        density_window=int(config.get("density_window", 16)),
        bbox_eps=float(config.get("bbox_eps", 1e-6)),
    )

    log_f, log_writer = make_log_writer(out_dir / "train_log.csv")
    train_losses: list[dict[str, float]] = []
    val_losses: list[dict[str, float]] = []
    try:
        model.eval()
        with torch.no_grad():
            raw_batch = next(iter(train_loader))
            calibration_batch = move_batch(raw_batch, device)
            calibration_outputs = model(calibration_batch["input"])
            _calibration_total, calibration_losses_tensor, calibration_warnings = branch1_losses(
                calibration_outputs,
                calibration_batch,
                loss_config,
            )
            calibration_losses = tensor_dict_to_float(calibration_losses_tensor)
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
            save_checkpoint(checkpoint_dir / "last.pt", model, optimizer, 0, config, train_losses, val_losses)
            save_preview(model, val_loader, preview_dir / "calibrate_batch.png", device, int(config.get("preview_items", 6)))
            log_row(log_writer, "calibrate", 0, "train", device, calibration_losses, calibration_warnings)
            print(
                json.dumps(
                    {"mode": "calibrate", "out_dir": str(out_dir), "losses": calibration_losses, "warnings": calibration_warnings},
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return

        for epoch in range(1, args.epochs + 1):
            model.train()
            epoch_rows: list[dict[str, float]] = []
            epoch_warnings: list[str] = []
            for raw_batch in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"):
                batch = move_batch(raw_batch, device)
                outputs = model(batch["input"])
                total, losses_tensor, warnings = branch1_losses(outputs, batch, loss_config)
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                optimizer.step()
                epoch_rows.append(tensor_dict_to_float(losses_tensor))
                epoch_warnings.extend(warnings)

            train_mean = mean_dict(epoch_rows)
            val_mean, val_warnings = run_eval(model, val_loader, loss_config, device)
            train_losses.append(train_mean)
            val_losses.append(val_mean)
            log_row(log_writer, "train", epoch, "train", device, train_mean, epoch_warnings)
            log_row(log_writer, "train", epoch, "val", device, val_mean, val_warnings)
            log_f.flush()

            save_checkpoint(checkpoint_dir / "last.pt", model, optimizer, epoch, config, train_losses, val_losses)
            save_preview(model, val_loader, preview_dir / f"epoch_{epoch:03d}.png", device, int(config.get("preview_items", 6)))
            print(
                f"epoch={epoch} train_total={train_mean.get('total', float('nan')):.6f} "
                f"val_total={val_mean.get('total', float('nan')):.6f}"
            )
    finally:
        log_f.close()


if __name__ == "__main__":
    main()
