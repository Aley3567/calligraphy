from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LossConfig:
    weights: dict[str, float]
    density_window: int = 16
    bbox_eps: float = 1e-6
    mask_blur_kernel: int = 3
    edge_source: str = "mask"
    ink_target_key: str = ""
    skeleton_target_key: str = ""
    gray_loss: str = "l1"
    foreground_gray_weight: float = 0.3
    anti_content_margin: float = 0.0
    visual_proxy_weights: dict[str, float] | None = None


def soft_edge(mask: torch.Tensor) -> torch.Tensor:
    dilated = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
    eroded = -F.max_pool2d(-mask, kernel_size=3, stride=1, padding=1)
    return (dilated - eroded).clamp(0.0, 1.0)


def local_density(img: torch.Tensor, window: int) -> torch.Tensor:
    return F.avg_pool2d(img, kernel_size=window, stride=1, padding=window // 2)


def soft_dilate(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    kernel = radius * 2 + 1
    return F.max_pool2d(mask, kernel_size=kernel, stride=1, padding=radius).clamp(0.0, 1.0)


def soft_target(mask: torch.Tensor, kernel: int) -> torch.Tensor:
    if kernel <= 1:
        return mask
    if kernel % 2 == 0:
        raise ValueError(f"mask_blur_kernel must be odd, got {kernel}")
    return F.avg_pool2d(mask, kernel_size=kernel, stride=1, padding=kernel // 2).clamp(0.0, 1.0)


def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred_flat = pred.flatten(start_dim=1)
    target_flat = target.flatten(start_dim=1)
    inter = (pred_flat * target_flat).sum(dim=1)
    denom = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    return (1.0 - (2.0 * inter + eps) / (denom + eps)).mean()


def dice_score_per_sample(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred_flat = pred.flatten(start_dim=1)
    target_flat = target.flatten(start_dim=1)
    inter = (pred_flat * target_flat).sum(dim=1)
    denom = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    return (2.0 * inter + eps) / (denom + eps)


def foreground_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    area = mask.sum(dim=(2, 3)).clamp_min(eps)
    return ((pred - target).abs() * mask).sum(dim=(2, 3)).div(area).mean()


def foreground_mean_std(values: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    area = mask.sum(dim=(2, 3), keepdim=True).clamp_min(eps)
    mean = (values * mask).sum(dim=(2, 3), keepdim=True) / area
    var = (((values - mean) * mask).pow(2)).sum(dim=(2, 3), keepdim=True) / area
    return mean.view(values.size(0)).mean(), torch.sqrt(var.clamp_min(0.0)).view(values.size(0)).mean()


def foreground_recall_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    missed = ((1.0 - pred).clamp(0.0, 1.0) * target).sum(dim=(1, 2, 3))
    target_area = target.sum(dim=(1, 2, 3)).clamp_min(eps)
    return (missed / target_area).mean()


def foreground_precision_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    false_positive = (pred * (1.0 - target).clamp(0.0, 1.0)).sum(dim=(1, 2, 3))
    pred_area = pred.sum(dim=(1, 2, 3)).clamp_min(eps)
    return (false_positive / pred_area).mean()


def area_ratio_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred_area = pred.sum(dim=(1, 2, 3))
    target_area = target.sum(dim=(1, 2, 3)).clamp_min(eps)
    return (pred_area / target_area - 1.0).abs().mean()


def outside_target_loss(pred: torch.Tensor, target: torch.Tensor, radius: int = 2, eps: float = 1e-6) -> torch.Tensor:
    allowed = soft_dilate(target, radius)
    outside = (1.0 - allowed).clamp(0.0, 1.0)
    pred_area = pred.sum(dim=(1, 2, 3)).clamp_min(eps)
    return (pred * outside).sum(dim=(1, 2, 3)).div(pred_area).mean()


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


def bbox_center_scale_loss(pred_bbox: torch.Tensor, target_bbox: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pred_center = torch.stack([(pred_bbox[:, 0] + pred_bbox[:, 2]) * 0.5, (pred_bbox[:, 1] + pred_bbox[:, 3]) * 0.5], dim=1)
    target_center = torch.stack([(target_bbox[:, 0] + target_bbox[:, 2]) * 0.5, (target_bbox[:, 1] + target_bbox[:, 3]) * 0.5], dim=1)
    pred_size = torch.stack([(pred_bbox[:, 2] - pred_bbox[:, 0]).clamp_min(0.0), (pred_bbox[:, 3] - pred_bbox[:, 1]).clamp_min(0.0)], dim=1)
    target_size = torch.stack([(target_bbox[:, 2] - target_bbox[:, 0]).clamp_min(0.0), (target_bbox[:, 3] - target_bbox[:, 1]).clamp_min(0.0)], dim=1)
    return F.l1_loss(pred_center, target_center), F.l1_loss(pred_size, target_size)


def branch1_losses(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: LossConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], list[str]]:
    pred_mask = outputs["mask"]
    pred_ink = outputs["ink"]
    pred_final = outputs["final_ink"]
    target_gray = batch["target_gray"]
    target_mask = batch["target_mask"]
    target_skeleton = batch.get("target_skeleton", target_mask)
    target_edge = batch["target_edge"]
    target_bbox = batch["target_bbox"]
    content_mask = batch.get("content_mask")
    if content_mask is None:
        content_mask = batch["input"][:, 1:2]
    content_hole = batch.get("content_hole")
    if content_hole is None:
        content_hole = batch["input"][:, 4:5]
    target_hole = batch.get("target_hole")

    target_mask_soft = soft_target(target_mask, config.mask_blur_kernel)
    mask_loss = 0.5 * dice_loss(pred_mask, target_mask) + 0.5 * F.l1_loss(pred_mask, target_mask_soft)
    edge_input = pred_final if config.edge_source == "final_ink" else pred_mask
    hole_area = content_hole.sum(dim=(2, 3)).clamp_min(1.0)
    hole_loss = ((pred_mask * content_hole).sum(dim=(2, 3)) / hole_area).mean()
    if target_hole is not None:
        target_hole_area = target_hole.sum(dim=(2, 3)).clamp_min(1.0)
        target_hole_loss = ((pred_final * target_hole).sum(dim=(2, 3)) / target_hole_area).mean()
    else:
        target_hole_loss = pred_mask.new_tensor(0.0)
    edge_map = soft_edge(edge_input)
    gray_full = F.l1_loss(pred_final, target_gray)
    fg_gray = foreground_l1(pred_final, target_gray, target_mask_soft)
    if config.gray_loss == "foreground_balanced":
        fg_weight = float(config.foreground_gray_weight)
        gray_loss = (1.0 - fg_weight) * gray_full + fg_weight * fg_gray
    elif config.gray_loss == "l1":
        gray_loss = gray_full
    else:
        raise ValueError(f"unsupported gray_loss: {config.gray_loss}")

    pred_fg_mask = (pred_mask.detach() > 0.5).float()
    target_fg_mask = (target_mask.detach() > 0.5).float()
    ink_fg_mean, ink_fg_std = foreground_mean_std(pred_ink, pred_fg_mask)
    target_fg_mean, target_fg_std = foreground_mean_std(target_gray, target_fg_mask)
    pred_target_dice = dice_score_per_sample(pred_final, target_mask)
    pred_content_dice = dice_score_per_sample(pred_final, content_mask)
    anti_content_margin_loss = F.relu(pred_content_dice - pred_target_dice + config.anti_content_margin).mean()

    raw = {
        "gray": gray_loss,
        "mask": mask_loss,
        "edge": F.l1_loss(edge_map, target_edge),
        "edge_dice": dice_loss(edge_map, target_edge),
        "final_dice": dice_loss(pred_final, target_mask),
        "final_precision": foreground_precision_loss(pred_final, target_mask),
        "final_recall": foreground_recall_loss(pred_final, target_mask),
        "final_area": area_ratio_loss(pred_final, target_mask),
        "outside_target": outside_target_loss(pred_final, target_mask, radius=2),
        "skeleton_cover": (((1.0 - pred_mask) * target_skeleton).sum(dim=(2, 3)) / target_skeleton.sum(dim=(2, 3)).clamp_min(1.0)).mean(),
        "density": F.l1_loss(local_density(pred_final, config.density_window), local_density(target_gray, config.density_window)),
        "hole": hole_loss,
        "target_hole": target_hole_loss,
        "gray_full": gray_full,
        "fg_gray_l1": fg_gray,
        "final_mask_l1": F.l1_loss(pred_final, pred_mask.detach()),
        "ink_fg_mean": ink_fg_mean,
        "ink_fg_std": ink_fg_std,
        "target_fg_mean": target_fg_mean,
        "target_fg_std": target_fg_std,
        "ink_ratio_absdiff": (pred_final.mean(dim=(1, 2, 3)) - target_gray.mean(dim=(1, 2, 3))).abs().mean(),
        "anti_content_dice_margin": anti_content_margin_loss,
        "pred_target_dice": pred_target_dice.mean(),
        "pred_content_dice": pred_content_dice.mean(),
    }
    if config.ink_target_key:
        if config.ink_target_key not in batch:
            raise KeyError(f"ink target key not found in batch: {config.ink_target_key}")
        ink_target = batch[config.ink_target_key]
        raw["ink_target"] = 0.5 * dice_loss(pred_ink, ink_target) + 0.5 * F.l1_loss(pred_ink, ink_target)
    if config.skeleton_target_key:
        if "skeleton_pred" not in outputs:
            raise KeyError("skeleton_target_key requires model output: skeleton_pred")
        if config.skeleton_target_key not in batch:
            raise KeyError(f"skeleton target key not found in batch: {config.skeleton_target_key}")
        skeleton_target = batch[config.skeleton_target_key]
        raw["skeleton_target"] = 0.5 * dice_loss(outputs["skeleton_pred"], skeleton_target) + 0.5 * F.l1_loss(outputs["skeleton_pred"], skeleton_target)
    raw["edge_mass"] = (edge_map.mean(dim=(1, 2, 3)) - target_edge.mean(dim=(1, 2, 3))).abs().mean()

    pred_bbox, stable = soft_bbox(pred_mask, config.bbox_eps)
    if stable.any():
        raw["bbox"] = F.l1_loss(pred_bbox[stable.bool()], target_bbox[stable.bool()])
        center, scale = bbox_center_scale_loss(pred_bbox[stable.bool()], target_bbox[stable.bool()])
        raw["bbox_center"] = center
        raw["bbox_scale"] = scale
    else:
        raw["bbox"] = pred_mask.new_tensor(0.0)
        raw["bbox_center"] = pred_mask.new_tensor(0.0)
        raw["bbox_scale"] = pred_mask.new_tensor(0.0)

    final_bbox, final_stable = soft_bbox(pred_final, config.bbox_eps)
    if final_stable.any():
        raw["final_bbox"] = F.l1_loss(final_bbox[final_stable.bool()], target_bbox[final_stable.bool()])
        final_center, final_scale = bbox_center_scale_loss(final_bbox[final_stable.bool()], target_bbox[final_stable.bool()])
        raw["final_bbox_center"] = final_center
        raw["final_bbox_scale"] = final_scale
    else:
        raw["final_bbox"] = pred_mask.new_tensor(0.0)
        raw["final_bbox_center"] = pred_mask.new_tensor(0.0)
        raw["final_bbox_scale"] = pred_mask.new_tensor(0.0)

    warnings: list[str] = []
    if not bool(stable.all().item()):
        warnings.append("bbox_unstable_for_empty_or_low_mass_prediction")
    if not bool(final_stable.all().item()):
        warnings.append("final_bbox_unstable_for_empty_or_low_mass_prediction")
    if float(target_fg_std.detach().cpu().item()) < 1e-3:
        warnings.append("target_foreground_gray_signal_low")

    total = pred_mask.new_tensor(0.0)
    weighted: dict[str, torch.Tensor] = {}
    proxy_weights = config.visual_proxy_weights or {"gray": 1.0, "mask": 0.5, "hole": 0.5, "edge_mass": 0.1}
    visual_proxy = pred_mask.new_tensor(0.0)
    for key, value in raw.items():
        weight = float(config.weights.get(key, 0.0))
        weighted[f"{key}_raw"] = value.detach()
        weighted[key] = value * weight
        if key in {
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
            "edge_mass",
            "bbox",
            "bbox_center",
            "bbox_scale",
            "final_bbox",
            "final_bbox_center",
            "final_bbox_scale",
            "ink_target",
            "skeleton_target",
            "anti_content_dice_margin",
        }:
            total = total + weighted[key]
        visual_proxy = visual_proxy + value.detach() * float(proxy_weights.get(key, 0.0))
    weighted["total"] = total.detach()
    weighted["visual_proxy"] = visual_proxy.detach()
    return total, weighted, warnings


def tensor_dict_to_float(losses: dict[str, torch.Tensor]) -> dict[str, float]:
    return {key: float(value.detach().cpu().item()) for key, value in losses.items()}
