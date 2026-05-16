from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

from common_images import make_grid


class PairDataset(Dataset):
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        manifest = data_dir / "manifest.csv"
        if not manifest.exists():
            raise FileNotFoundError(f"manifest not found: {manifest}")
        with manifest.open("r", encoding="utf-8", newline="") as f:
            self.rows = list(csv.DictReader(f))
        if not self.rows:
            raise ValueError(f"empty manifest: {manifest}")

    def __len__(self) -> int:
        return len(self.rows)

    def _load(self, rel_path: str) -> torch.Tensor:
        img = Image.open(self.data_dir / rel_path).convert("L")
        arr = np.asarray(img).astype(np.float32) / 255.0
        arr = arr * 2.0 - 1.0
        return torch.from_numpy(arr).unsqueeze(0)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[idx]
        return self._load(row["content_path"]), self._load(row["target_path"])


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


class UNetGenerator(nn.Module):
    def __init__(self, base: int = 64):
        super().__init__()
        self.d1 = Down(1, base, norm=False)
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
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(base * 2, 1, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        return self.final(u5)


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    arr = tensor.detach().cpu().squeeze(0).numpy()
    arr = ((arr + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def save_preview(model: nn.Module, loader: DataLoader, out_path: Path, device: torch.device, max_items: int = 8) -> None:
    model.eval()
    images: list[Image.Image] = []
    with torch.no_grad():
        for content, target in loader:
            content = content.to(device)
            pred = model(content)
            for i in range(min(content.size(0), max_items - len(images) // 3)):
                images.extend([tensor_to_image(content[i]), tensor_to_image(pred[i]), tensor_to_image(target[i])])
            if len(images) >= max_items * 3:
                break
    out_path.parent.mkdir(parents=True, exist_ok=True)
    make_grid(images, cols=3).save(out_path)
    model.train()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train U-Net baseline for single-writer calligraphy generation.")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--lr", default=2e-4, type=float)
    parser.add_argument("--lambda-l1", default=100.0, type=float)
    parser.add_argument("--image-size", default=128, type=int)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    dataset = PairDataset(args.data_dir.expanduser().resolve())
    val_size = max(1, int(len(dataset) * 0.1))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetGenerator().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
    l1 = nn.L1Loss()

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=device.type == "cuda")

    best_val = float("inf")
    log_path = out_dir / "train_log.csv"
    with log_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_l1", "val_l1", "device"])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            model.train()
            train_losses: list[float] = []
            for content, target in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"):
                content = content.to(device)
                target = target.to(device)
                pred = model(content)
                loss = l1(pred, target) * args.lambda_l1
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                train_losses.append(float(loss.item() / args.lambda_l1))

            model.eval()
            val_losses: list[float] = []
            with torch.no_grad():
                for content, target in val_loader:
                    content = content.to(device)
                    target = target.to(device)
                    pred = model(content)
                    val_losses.append(float(l1(pred, target).item()))

            train_l1 = float(np.mean(train_losses))
            val_l1 = float(np.mean(val_losses))
            writer.writerow({"epoch": epoch, "train_l1": train_l1, "val_l1": val_l1, "device": str(device)})
            f.flush()

            ckpt = {
                "model": model.state_dict(),
                "epoch": epoch,
                "train_l1": train_l1,
                "val_l1": val_l1,
                "image_size": args.image_size,
                "model_name": "UNetGenerator",
            }
            torch.save(ckpt, out_dir / "checkpoints" / "last.pt")
            if val_l1 < best_val:
                best_val = val_l1
                torch.save(ckpt, out_dir / "checkpoints" / "best.pt")

            save_preview(model, val_loader, out_dir / "previews" / f"epoch_{epoch:03d}.png", device)
            print(f"epoch={epoch} train_l1={train_l1:.4f} val_l1={val_l1:.4f} best={best_val:.4f}")


if __name__ == "__main__":
    main()
