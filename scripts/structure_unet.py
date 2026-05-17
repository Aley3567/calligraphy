from __future__ import annotations

import torch
from torch import nn


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, norm: bool = True):
        super().__init__()
        layers: list[nn.Module] = [nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=not norm)]
        if norm:
            layers.append(nn.BatchNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: bool = False):
        super().__init__()
        layers: list[nn.Module] = [
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return torch.cat([x, skip], dim=1)


class StructureInkUNetBranch1(nn.Module):
    def __init__(self, input_channels: int = 5, base: int = 64):
        super().__init__()
        self.d1 = Down(input_channels, base, norm=False)
        self.d2 = Down(base, base * 2)
        self.d3 = Down(base * 2, base * 4)
        self.d4 = Down(base * 4, base * 8)
        self.d5 = Down(base * 8, base * 8)
        self.bottleneck = Down(base * 8, base * 8)

        self.u1 = Up(base * 8, base * 8, dropout=True)
        self.u2 = Up(base * 16, base * 8, dropout=True)
        self.u3 = Up(base * 16, base * 4)
        self.u4 = Up(base * 8, base * 2)
        self.u5 = Up(base * 4, base)
        self.final_features = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(base * 2, base, 3, 1, 1),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
        )
        self.mask_head = nn.Sequential(nn.Conv2d(base, 1, 3, 1, 1), nn.Sigmoid())
        self.ink_head = nn.Sequential(nn.Conv2d(base, 1, 3, 1, 1), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)
        b = self.bottleneck(d5)
        u1 = self.u1(b, d5)
        u2 = self.u2(u1, d4)
        u3 = self.u3(u2, d3)
        u4 = self.u4(u3, d2)
        u5 = self.u5(u4, d1)
        features = self.final_features(u5)
        pred_mask = self.mask_head(features)
        pred_ink = self.ink_head(features)
        return {
            "mask": pred_mask,
            "ink": pred_ink,
            "final_ink": pred_mask * pred_ink,
        }
