from __future__ import annotations

import argparse
import csv
import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tqdm import tqdm

from structure_maps import (
    binary_ink_mask,
    edge_map,
    foreground_distance,
    hole_map,
    load_gray,
    save_map,
    stats_to_dict,
    summarize_binary,
    zhang_suen_skeleton,
)


MAP_SPECS = [
    ("content_mask", "content_mask_path"),
    ("content_skeleton", "content_skeleton_path"),
    ("content_distance", "content_distance_path"),
    ("content_hole", "content_hole_path"),
    ("target_mask", "target_mask_path"),
    ("target_skeleton", "target_skeleton_path"),
    ("target_distance", "target_distance_path"),
    ("target_edge", "target_edge_path"),
    ("target_hole", "target_hole_path"),
]


def rel_map_path(map_name: str, sample_id: str) -> str:
    return f"{map_name}/{sample_id}.png"


def copy_pair(data_dir: Path, out_dir: Path, row: dict[str, str]) -> None:
    for key in ("content_path", "target_path"):
        src = data_dir / row[key]
        dst = out_dir / row[key]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def build_maps(img_path: Path, out_dir: Path, sample_id: str, prefix: str, threshold: int) -> dict[str, str]:
    img = load_gray(img_path)
    mask = binary_ink_mask(img, threshold=threshold)
    skeleton = zhang_suen_skeleton(mask)
    distance = foreground_distance(mask)
    holes = hole_map(mask)

    paths: dict[str, str] = {}
    for name, arr in [
        (f"{prefix}_mask", mask),
        (f"{prefix}_skeleton", skeleton),
        (f"{prefix}_distance", distance),
        (f"{prefix}_hole", holes),
    ]:
        rel_path = rel_map_path(name, sample_id)
        save_map(arr, out_dir / rel_path)
        paths[f"{name}_path"] = rel_path

    if prefix == "target":
        rel_path = rel_map_path("target_edge", sample_id)
        save_map(edge_map(mask), out_dir / rel_path)
        paths["target_edge_path"] = rel_path

    return paths


def filter_flags(row: dict[str, str | float | int]) -> list[str]:
    flags: list[str] = []
    target_ink = float(row["target_ink_ratio"])
    target_bbox_w = int(row["target_bbox_w"])
    target_bbox_h = int(row["target_bbox_h"])
    target_holes = int(row["target_hole_count"])
    content_holes = int(row["content_hole_count"])
    content_bbox_w = int(row["content_bbox_w"])
    content_bbox_h = int(row["content_bbox_h"])
    if content_bbox_w == 0 or content_bbox_h == 0:
        flags.append("empty_content")
    if target_ink < 0.015:
        flags.append("too_light")
    if target_ink > 0.45:
        flags.append("too_dark")
    if target_bbox_w == 0 or target_bbox_h == 0:
        flags.append("empty_target")
    if target_bbox_w > 250 or target_bbox_h > 250:
        flags.append("extreme_bbox")
    if content_holes > 0 and target_holes < content_holes:
        flags.append("possible_lost_hole")
    if float(row["target_local_density_max"]) > 0.98 and target_ink > 0.30:
        flags.append("local_density_too_high")
    return flags


def process_one_task(
    task: tuple[dict[str, str], str, str, int, bool],
) -> tuple[dict[str, str | float | int] | None, dict[str, str | float | int | list[str]] | None, dict[str, str] | None]:
    row, data_dir_str, out_dir_str, threshold, no_copy_pairs = task
    data_dir = Path(data_dir_str)
    out_dir = Path(out_dir_str)
    sample_id = row["sample_id"]
    try:
        if not no_copy_pairs:
            copy_pair(data_dir, out_dir, row)

        content_path = data_dir / row["content_path"]
        target_path = data_dir / row["target_path"]
        content_mask = binary_ink_mask(load_gray(content_path), threshold=threshold)
        target_mask = binary_ink_mask(load_gray(target_path), threshold=threshold)
        content_stats = summarize_binary(content_mask)
        target_stats = summarize_binary(target_mask)

        map_paths = {
            **build_maps(content_path, out_dir, sample_id, "content", threshold),
            **build_maps(target_path, out_dir, sample_id, "target", threshold),
        }
        enriched: dict[str, str | float | int] = {
            **row,
            **map_paths,
            **stats_to_dict("content", content_stats),
            **stats_to_dict("target", target_stats),
        }
        flags = filter_flags(enriched)
        enriched["filter_flags"] = "|".join(flags)
        return enriched, {**enriched, "filter_flags_list": flags}, None
    except Exception as exc:  # noqa: BLE001
        return None, None, {"sample_id": sample_id, "path": row.get("source_path", ""), "error": str(exc)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build structure maps and metadata for paired calligraphy data.")
    parser.add_argument("--data-dir", required=True, type=Path, help="Existing paired dataset with manifest.csv.")
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--threshold", default=220, type=int)
    parser.add_argument("--max-items", default=0, type=int, help="0 means no limit.")
    parser.add_argument("--no-copy-pairs", action="store_true", help="Only write maps and manifest references.")
    parser.add_argument("--workers", default=1, type=int, help="Parallel worker processes for structure extraction.")
    args = parser.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    manifest_path = data_dir / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    out_dir.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if args.max_items > 0:
        rows = rows[: args.max_items]

    output_rows: list[dict[str, str | float | int]] = []
    metadata_rows: list[dict[str, str | float | int | list[str]]] = []
    skipped: list[dict[str, str]] = []

    tasks = [(row, str(data_dir), str(out_dir), args.threshold, args.no_copy_pairs) for row in rows]
    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            iterator = executor.map(process_one_task, tasks, chunksize=8)
            results = tqdm(iterator, total=len(tasks), desc="build structure maps")
            for enriched, metadata, skip in results:
                if enriched is not None and metadata is not None:
                    output_rows.append(enriched)
                    metadata_rows.append(metadata)
                if skip is not None:
                    skipped.append(skip)
    else:
        for task in tqdm(tasks, desc="build structure maps"):
            enriched, metadata, skip = process_one_task(task)
            if enriched is not None and metadata is not None:
                output_rows.append(enriched)
                metadata_rows.append(metadata)
            if skip is not None:
                skipped.append(skip)

    fieldnames: list[str] = []
    for row in output_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with (out_dir / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    with (out_dir / "metadata.jsonl").open("w", encoding="utf-8") as f:
        for row in metadata_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "input_data_dir": str(data_dir),
        "out_dir": str(out_dir),
        "threshold": args.threshold,
        "workers": args.workers,
        "processed_count": len(output_rows),
        "skipped_count": len(skipped),
        "flag_counts": {},
        "skipped_examples": skipped[:100],
        "map_specs": MAP_SPECS,
    }
    flag_counts: dict[str, int] = {}
    for row in output_rows:
        flags = str(row.get("filter_flags", ""))
        for flag in [item for item in flags.split("|") if item]:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
    summary["flag_counts"] = flag_counts
    with (out_dir / "structure_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"processed: {len(output_rows)}")
    print(f"skipped: {len(skipped)}")
    print(f"out: {out_dir}")


if __name__ == "__main__":
    main()
