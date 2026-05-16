from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from common_images import make_grid, render_content_char
from train_unet_baseline import UNetGenerator


def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    arr = tensor.detach().cpu().squeeze(0).numpy()
    arr = ((arr + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def image_to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("L")).astype(np.float32) / 255.0
    arr = arr * 2.0 - 1.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fixed character evaluation board.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--content-font", required=True, type=Path)
    parser.add_argument("--text", default="一中国永道德山水风月")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--image-size", default=128, type=int)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetGenerator().to(device)
    ckpt = torch.load(args.checkpoint.expanduser().resolve(), map_location=device)
    state = ckpt.get("model") or ckpt.get("generator")
    if state is None:
        raise KeyError("checkpoint must contain 'model' or 'generator'")
    model.load_state_dict(state)
    model.eval()

    images: list[Image.Image] = []
    with torch.no_grad():
        for char in args.text:
            if char.isspace():
                continue
            content = render_content_char(char, args.content_font.expanduser().resolve(), args.image_size)
            pred = model(image_to_tensor(content).to(device))[0]
            images.extend([content, tensor_to_image(pred)])

    args.out.expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
    make_grid(images, cols=2).save(args.out.expanduser().resolve())
    print(f"saved eval board: {args.out}")


if __name__ == "__main__":
    main()
