from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from train_unet_baseline import PairDataset, UNetGenerator, save_preview


class PatchDiscriminator(nn.Module):
    def __init__(self, base: int = 64):
        super().__init__()

        def block(in_ch: int, out_ch: int, stride: int, norm: bool = True) -> list[nn.Module]:
            layers: list[nn.Module] = [nn.Conv2d(in_ch, out_ch, 4, stride, 1, bias=not norm)]
            if norm:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.net = nn.Sequential(
            *block(2, base, 2, norm=False),
            *block(base, base * 2, 2),
            *block(base * 2, base * 4, 2),
            *block(base * 4, base * 8, 1),
            nn.Conv2d(base * 8, 1, 4, 1, 1),
        )

    def forward(self, content: torch.Tensor, target_or_fake: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([content, target_or_fake], dim=1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Pix2Pix baseline for calligraphy generation.")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--lr", default=2e-4, type=float)
    parser.add_argument("--lambda-l1", default=100.0, type=float)
    parser.add_argument("--image-size", default=128, type=int)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out_dir = args.out_dir.expanduser().resolve()
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    dataset = PairDataset(args.data_dir.expanduser().resolve())
    val_size = max(1, int(len(dataset) * 0.1))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = UNetGenerator().to(device)
    discriminator = PatchDiscriminator().to(device)
    opt_g = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    l1 = nn.L1Loss()

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
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

    best_val = float("inf")
    log_path = out_dir / "train_log.csv"
    with log_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "g_loss", "d_loss", "val_l1", "device"])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            generator.train()
            discriminator.train()
            g_losses: list[float] = []
            d_losses: list[float] = []

            for content, target in tqdm(train_loader, desc=f"pix2pix epoch {epoch}/{args.epochs}"):
                content = content.to(device)
                target = target.to(device)

                with torch.no_grad():
                    fake_detached = generator(content)

                real_logits = discriminator(content, target)
                fake_logits = discriminator(content, fake_detached)
                d_loss = 0.5 * (
                    bce(real_logits, torch.ones_like(real_logits))
                    + bce(fake_logits, torch.zeros_like(fake_logits))
                )
                opt_d.zero_grad(set_to_none=True)
                d_loss.backward()
                opt_d.step()

                fake = generator(content)
                fake_logits_for_g = discriminator(content, fake)
                gan_loss = bce(fake_logits_for_g, torch.ones_like(fake_logits_for_g))
                recon_loss = l1(fake, target)
                g_loss = gan_loss + args.lambda_l1 * recon_loss
                opt_g.zero_grad(set_to_none=True)
                g_loss.backward()
                opt_g.step()

                g_losses.append(float(g_loss.item()))
                d_losses.append(float(d_loss.item()))

            generator.eval()
            val_losses: list[float] = []
            with torch.no_grad():
                for content, target in val_loader:
                    content = content.to(device)
                    target = target.to(device)
                    pred = generator(content)
                    val_losses.append(float(l1(pred, target).item()))

            val_l1 = float(np.mean(val_losses))
            row = {
                "epoch": epoch,
                "g_loss": float(np.mean(g_losses)),
                "d_loss": float(np.mean(d_losses)),
                "val_l1": val_l1,
                "device": str(device),
            }
            writer.writerow(row)
            f.flush()

            ckpt = {
                "generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                "epoch": epoch,
                "val_l1": val_l1,
                "image_size": args.image_size,
                "model_name": "Pix2Pix_UNet_PatchGAN",
            }
            torch.save(ckpt, out_dir / "checkpoints" / "last.pt")
            if val_l1 < best_val:
                best_val = val_l1
                torch.save(ckpt, out_dir / "checkpoints" / "best.pt")

            save_preview(generator, val_loader, out_dir / "previews" / f"epoch_{epoch:03d}.png", device)
            print(f"epoch={epoch} g={row['g_loss']:.4f} d={row['d_loss']:.4f} val_l1={val_l1:.4f} best={best_val:.4f}")


if __name__ == "__main__":
    main()

