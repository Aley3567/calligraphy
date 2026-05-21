from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path


def split_flags(value: str) -> list[str]:
    return [flag for flag in value.split("|") if flag]


def write_manifest(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def flag_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for row in rows:
        counter.update(split_flags(row.get("filter_flags", "")))
    return dict(sorted(counter.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare deterministic train/val manifests for structure-aware training.")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--train-ratio", default=0.9, type=float)
    parser.add_argument("--train-count", default=0, type=int)
    parser.add_argument("--val-count", default=0, type=int)
    args = parser.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    manifest_path = data_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    if args.train_count <= 0 and not 0.0 < args.train_ratio < 1.0:
        raise ValueError("--train-ratio must be between 0 and 1")

    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not rows:
        raise ValueError(f"empty manifest: {manifest_path}")

    rejected: list[dict[str, str]] = []
    usable: list[dict[str, str]] = []
    for row in rows:
        flags = set(split_flags(row.get("filter_flags", "")))
        if "empty_content" in flags:
            rejected.append(row)
        else:
            usable.append(row)

    rng = random.Random(args.seed)
    rng.shuffle(usable)
    if args.train_count > 0 or args.val_count > 0:
        if args.train_count <= 0 or args.val_count <= 0:
            raise ValueError("--train-count and --val-count must be used together")
        requested = args.train_count + args.val_count
        if requested > len(usable):
            raise ValueError(f"requested {requested} rows but only {len(usable)} usable rows are available")
        train_count = args.train_count
        val_count = args.val_count
        train_rows = usable[:train_count]
        val_rows = usable[train_count : train_count + val_count]
    else:
        train_count = int(len(usable) * args.train_ratio)
        train_rows = usable[:train_count]
        val_rows = usable[train_count:]

    write_manifest(out_dir / "train_manifest.csv", train_rows, fieldnames)
    write_manifest(out_dir / "val_manifest.csv", val_rows, fieldnames)
    write_manifest(out_dir / "rejected_empty_content_manifest.csv", rejected, fieldnames)

    summary = {
        "data_dir": str(data_dir),
        "out_dir": str(out_dir),
        "source_manifest": str(manifest_path),
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "train_count_requested": args.train_count,
        "val_count_requested": args.val_count,
        "total_rows": len(rows),
        "usable_rows": len(usable),
        "rejected_empty_content_rows": len(rejected),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "all_flag_counts": flag_counts(rows),
        "usable_flag_counts": flag_counts(usable),
        "train_flag_counts": flag_counts(train_rows),
        "val_flag_counts": flag_counts(val_rows),
        "rejected_flag_counts": flag_counts(rejected),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "split_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
