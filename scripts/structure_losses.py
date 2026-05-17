from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LossConfig:
    weights: dict[str, float]
    density_window: int = 16
    bbox_eps: float = 1e-6


def soft_edge(mask: torch.Tensor) -> torch.Tensor:
    dilated = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
    eroded = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
    return (dilated - eroded).clamp(0.0, 1.0)


def local_density(img: torch.Tensor, window: int) -> torch.Tensor:
    return F.avg_pool2d(img, kernel_size=window, stride=1, padding=window // 2)


def soft_bbox(mask: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    batch, _channels, height, width = mask.shape
    device = mask.device
    dtype = mask.dtype
    x = torch.linspace(0.0, 1.0, width, device=device, dtype=dtype).view(1, 1, 1, width)
    y = torch.linspace(0.0, 1.0, height, device=device, dtype=dtype).view(1, 1, height, 1)
    mass = mask.sum(dim=(2, 3), keepdim=True)
    stable = (mass.view(batch) > eps).float()

    x_mean = (mask * x).sum(dim=(2, 3), keepdim=True) / mass.clamp_min(eps)
    y_mean = (mask * y).sum(dim=(2, 3), keepdim=True) / mass.clamp_min(eps)
    x_std = torch.sqrt(((mask * (x - x_mean).pow(2)).sum(dim=(2, 3), keepdim=True) / mass.clamp_min(eps)).clamp_min(eps))
    y_std = torch.sqrt(((mask * (y - y_mean).pow(2)).sum(dim=(2, 3), keepdim=True) / mass.clamp_min(eps)).clamp_min(eps))

    x0 = (x_mean - 2.0 * x_std).view(batch).clamp(0.0, 1.0)
    y0 = (y_mean - 2.0 * y_std).view(batch).clamp(0.0, 1.0)
    x1 = (x_mean + 2.0 * x_std).view(batch).clamp(0.0, 1.0)
    y1 = (y_mean + 2.0 * y_std).view(batch).clamp(0.0, 1.0)
    return torch.stack([x0, y0, x1, y1], dim=1), stable


def branch1_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: LossConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], list[str]]:
    pred_mask = outputs["mask"]
    pred_final = outputs["final_ink"]
    target_gray = batch["target_gray"]
    target_mask = batch["target_mask"]
    target_edge = batch["target_edge"]
    target_hole = batch["target_hole"]
    target_bbox = batch["target_bbox"]

    raw = {
        "gray": F.l1_loss(pred_final, target_gray),
        "mask": F.binary_cross_entropy(pred_mask.clamp(1e-6, 1.0 - 1e-6), target_mask),
        "edge": F.l1_loss(soft_edge(pred_mask), target_edge),
        "density": F.l1_loss(local_density(pred_final, config.density_window), local_density(target_gray, config.density_window)),
        "hole": (pred_mask * target_hole).mean(),
    }

    pred_bbox, stable = soft_bbox(pred_mask, config.bbox_eps)
    if stable.any():
        raw["bbox"] = F.l1_loss(pred_bbox[stable.bool()], target_bbox[stable.bool()])
    else:
        raw["bbox"] = pred_mask.new_tensor(0.0)

    warnings: list[str] = []
    if not bool(stable.all().item()):
        warnings.append("bbox_unstable_for_empty_or_low_mass_prediction")

    total = pred_mask.new_tensor(0.0)
    weighted: dict[str, torch.Tensor] = {}
    for key, value in raw.items():
        weight = float(config.weights.get(key, 0.0))
        weighted[f"{key}_raw"] = value.detach()
        weighted[key] = value * weight
        total = total + weighted[key]
    weighted["total"] = total.detach()
    return total, weighted, warnings


def tensor_dict_to_float(losses: dict[str, torch.Tensor]) -> dict[str, float]:
    return {key: float(value.detach().cpu().item()) for key, value in losses.items()}
