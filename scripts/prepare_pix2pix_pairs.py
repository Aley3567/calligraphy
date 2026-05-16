from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from tqdm import tqdm

from common_images import (
    infer_char_from_path,
    iter_image_files,
    normalize_ink_image,
    open_grayscale,
    render_content_char,
    save_pair,
    stable_id,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare single-writer paired content/target data.")
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--writer-name", required=True)
    parser.add_argument("--content-font", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--image-size", default=128, type=int)
    parser.add_argument("--max-items", default=0, type=int, help="0 means no limit")
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()

    raw_root = args.raw_root.expanduser().resolve()
    writer_dir = raw_root / args.writer_name
    font_path = args.content_font.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not writer_dir.exists():
        raise FileNotFoundError(f"writer folder not found: {writer_dir}")
    if not font_path.exists():
        raise FileNotFoundError(f"content font not found: {font_path}")

    files = list(iter_image_files(writer_dir))
    random.Random(args.seed).shuffle(files)
    if args.max_items > 0:
        files = files[: args.max_items]

    rows: list[dict] = []
    skipped: list[dict] = []
    for path in tqdm(files, desc="prepare pairs"):
        char = infer_char_from_path(path)
        if not char or len(char) != 1:
            skipped.append({"path": str(path), "reason": f"cannot infer single char: {char!r}"})
            continue

        sample_id = stable_id(f"{args.writer_name}/{path.relative_to(writer_dir)}")
        try:
            target = normalize_ink_image(open_grayscale(path), args.image_size)
            content = render_content_char(char, font_path, args.image_size)
            save_pair(content, target, out_dir, sample_id)
        except Exception as exc:  # noqa: BLE001
            skipped.append({"path": str(path), "reason": str(exc)})
            continue

        rows.append(
            {
                "sample_id": sample_id,
                "writer": args.writer_name,
                "char": char,
                "source_path": str(path),
                "content_path": f"content/{sample_id}.png",
                "target_path": f"target/{sample_id}.png",
            }
        )

    with (out_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["sample_id", "writer", "char", "source_path", "content_path", "target_path"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with (out_dir / "prepare_summary.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "writer": args.writer_name,
                "raw_root": str(raw_root),
                "writer_dir": str(writer_dir),
                "content_font": str(font_path),
                "image_size": args.image_size,
                "pair_count": len(rows),
                "skipped_count": len(skipped),
                "skipped_examples": skipped[:100],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"prepared pairs: {len(rows)}")
    print(f"skipped: {len(skipped)}")
    print(f"out: {out_dir}")


if __name__ == "__main__":
    main()

