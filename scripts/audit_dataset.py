from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image

from common_images import IMAGE_EXTS, infer_char_from_path, iter_image_files


def verify_image(path: Path) -> tuple[bool, str]:
    try:
        with Image.open(path) as img:
            img.verify()
        return True, ""
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit writer-folder calligraphy dataset.")
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()

    raw_root = args.raw_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if not raw_root.exists():
        raise FileNotFoundError(f"raw root not found: {raw_root}")

    writer_stats: dict[str, dict] = {}
    bad_images: list[dict] = []
    all_files = list(iter_image_files(raw_root))

    by_writer: dict[str, list[Path]] = defaultdict(list)
    for path in all_files:
        rel = path.relative_to(raw_root)
        writer = rel.parts[0] if len(rel.parts) > 1 else "ROOT"
        by_writer[writer].append(path)

    for writer, files in sorted(by_writer.items()):
        char_counter: Counter[str] = Counter()
        ext_counter: Counter[str] = Counter()
        ok_count = 0
        for path in files:
            ext_counter[path.suffix.lower()] += 1
            char = infer_char_from_path(path)
            if char:
                char_counter[char] += 1
            ok, err = verify_image(path)
            if ok:
                ok_count += 1
            else:
                bad_images.append({"writer": writer, "path": str(path), "error": err})

        writer_stats[writer] = {
            "writer": writer,
            "image_count": len(files),
            "ok_image_count": ok_count,
            "bad_image_count": len(files) - ok_count,
            "unique_char_count": len(char_counter),
            "duplicate_char_count": sum(count - 1 for count in char_counter.values() if count > 1),
            "extensions": dict(ext_counter),
            "top_chars": char_counter.most_common(30),
        }

    summary = {
        "raw_root": str(raw_root),
        "total_image_files": len(all_files),
        "supported_extensions": sorted(IMAGE_EXTS),
        "writer_count": len(writer_stats),
        "recommended_writers": sorted(
            writer_stats.values(),
            key=lambda row: (row["unique_char_count"], row["ok_image_count"]),
            reverse=True,
        )[:20],
        "writers": writer_stats,
        "bad_images": bad_images[:1000],
    }

    with (out_dir / "audit_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with (out_dir / "writer_summary.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["writer", "image_count", "ok_image_count", "bad_image_count", "unique_char_count", "duplicate_char_count"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(writer_stats.values(), key=lambda x: x["unique_char_count"], reverse=True):
            writer.writerow({key: row[key] for key in fieldnames})

    with (out_dir / "bad_images.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = ["writer", "path", "error"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(bad_images)

    print(f"audit complete: {out_dir}")
    print(f"writers: {len(writer_stats)}")
    print(f"images: {len(all_files)}")
    if summary["recommended_writers"]:
        top = summary["recommended_writers"][0]
        print(f"top writer candidate: {top['writer']} unique_chars={top['unique_char_count']} images={top['ok_image_count']}")


if __name__ == "__main__":
    main()

