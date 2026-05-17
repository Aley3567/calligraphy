from __future__ import annotations

import json
import os
import subprocess
import sys
import tarfile
import time
import zipfile
from pathlib import Path

import modal


APP_NAME = "calligraphy-unet-training"
VOLUME_NAME = "calligraphy-training"
VOL = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
VOL_ROOT = Path("/vol")
LOCAL_ZIP = Path("data/raw/chinese-calligraphy-dataset-with-calligrapher-221030.zip")
LOCAL_PREPARED_256 = Path("data/processed/zhaomengfu_full_256")
LOCAL_PREPARED_256_TAR = Path("data/processed/zhaomengfu_full_256.tar.gz")
REMOTE_ZIP = VOL_ROOT / "data/raw/chinese-calligraphy-dataset-with-calligrapher-221030.zip"
REMOTE_ZIP_IN_VOLUME = Path("/data/raw/chinese-calligraphy-dataset-with-calligrapher-221030.zip")
REMOTE_PREPARED_256 = VOL_ROOT / "data/processed/zhaomengfu_full_256"
REMOTE_PREPARED_256_IN_VOLUME = Path("/data/processed/zhaomengfu_full_256")
REMOTE_PREPARED_256_TAR = VOL_ROOT / "data/processed/zhaomengfu_full_256.tar.gz"
REMOTE_PREPARED_256_TAR_IN_VOLUME = Path("/data/processed/zhaomengfu_full_256.tar.gz")
REMOTE_RAW_ROOT = VOL_ROOT / "data/raw/extracted_calligrapher_dataset/chinese-calligraphy-dataset-with-calligrapher"

image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime")
    .apt_install("fonts-arphic-uming", "fontconfig")
    .pip_install("Pillow>=9.5", "numpy>=1.23", "tqdm>=4.65")
    .add_local_dir(Path(__file__).resolve().parents[1] / "scripts", remote_path="/root/calligraphy/scripts", copy=True)
    .add_local_dir(Path(__file__).resolve().parents[1] / "configs", remote_path="/root/calligraphy/configs", copy=True)
)

app = modal.App(APP_NAME)


def run(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def ensure_dataset_extracted() -> None:
    if REMOTE_RAW_ROOT.exists():
        print(f"dataset already extracted: {REMOTE_RAW_ROOT}", flush=True)
        return
    if not REMOTE_ZIP.exists():
        raise FileNotFoundError(f"missing remote zip: {REMOTE_ZIP}")
    extract_root = VOL_ROOT / "data/raw/extracted_calligrapher_dataset"
    extract_root.mkdir(parents=True, exist_ok=True)
    print(f"extracting {REMOTE_ZIP} -> {extract_root}", flush=True)
    with zipfile.ZipFile(REMOTE_ZIP) as zf:
        zf.extractall(extract_root)
    if not REMOTE_RAW_ROOT.exists():
        raise FileNotFoundError(f"expected extracted root not found: {REMOTE_RAW_ROOT}")


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/vol": VOL},
    timeout=60 * 60 * 8,
    cpu=8,
    memory=32768,
)
def train_unet_remote(
    writer: str = "楷-赵孟俯三门记",
    image_size: int = 256,
    epochs: int = 100,
    batch_size: int = 16,
    max_items: int = 0,
) -> dict:
    repo = Path("/root/calligraphy")
    font = Path("/usr/share/fonts/truetype/arphic/uming.ttc")
    if not font.exists():
        raise FileNotFoundError(f"font not found: {font}")

    ensure_dataset_extracted()

    run_name = f"modal_unet_{image_size}_{writer.replace('/', '_')}_{int(time.time())}"
    processed_dir = VOL_ROOT / "data/processed" / f"{run_name}_pairs"
    out_dir = VOL_ROOT / "outputs" / run_name
    processed_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    run(
        [
            sys.executable,
            "scripts/prepare_glyph_pairs.py",
            "--raw-root",
            str(REMOTE_RAW_ROOT),
            "--writer-name",
            writer,
            "--content-font",
            str(font),
            "--out-dir",
            str(processed_dir),
            "--image-size",
            str(image_size),
            "--max-items",
            str(max_items),
        ],
        cwd=repo,
    )

    run(
        [
            sys.executable,
            "scripts/train_unet_baseline.py",
            "--data-dir",
            str(processed_dir),
            "--out-dir",
            str(out_dir),
            "--epochs",
            str(epochs),
            "--batch-size",
            str(batch_size),
            "--image-size",
            str(image_size),
            "--num-workers",
            "4",
        ],
        cwd=repo,
    )

    run(
        [
            sys.executable,
            "scripts/evaluate_quality.py",
            "--checkpoint",
            str(out_dir / "checkpoints/best.pt"),
            "--content-font",
            str(font),
            "--text-file",
            "configs/eval_chars_stage1.txt",
            "--out-dir",
            str(out_dir / "fixed_eval_best"),
            "--image-size",
            str(image_size),
        ],
        cwd=repo,
    )

    result = {
        "run_name": run_name,
        "writer": writer,
        "image_size": image_size,
        "epochs": epochs,
        "batch_size": batch_size,
        "processed_dir": str(processed_dir),
        "out_dir": str(out_dir),
        "best_checkpoint": str(out_dir / "checkpoints/best.pt"),
        "eval_board": str(out_dir / "fixed_eval_best/eval_board.png"),
        "quality_metrics": str(out_dir / "fixed_eval_best/quality_metrics.csv"),
    }
    (out_dir / "modal_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    VOL.commit()
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/vol": VOL},
    timeout=60 * 60 * 8,
    cpu=8,
    memory=32768,
)
def train_prepared_unet_remote(
    image_size: int = 256,
    epochs: int = 100,
    batch_size: int = 16,
) -> dict:
    repo = Path("/root/calligraphy")
    font = Path("/usr/share/fonts/truetype/arphic/uming.ttc")
    if not REMOTE_PREPARED_256.exists():
        if not REMOTE_PREPARED_256_TAR.exists():
            raise FileNotFoundError(f"prepared dataset tar not found: {REMOTE_PREPARED_256_TAR}")
        print(f"extracting prepared dataset: {REMOTE_PREPARED_256_TAR}", flush=True)
        with tarfile.open(REMOTE_PREPARED_256_TAR, "r:gz") as tf:
            tf.extractall(VOL_ROOT / "data/processed")
        if not REMOTE_PREPARED_256.exists():
            raise FileNotFoundError(f"prepared dataset not found after extract: {REMOTE_PREPARED_256}")

    run_name = f"modal_prepared_unet_{image_size}_{int(time.time())}"
    out_dir = VOL_ROOT / "outputs" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    run(
        [
            sys.executable,
            "scripts/train_unet_baseline.py",
            "--data-dir",
            str(REMOTE_PREPARED_256),
            "--out-dir",
            str(out_dir),
            "--epochs",
            str(epochs),
            "--batch-size",
            str(batch_size),
            "--image-size",
            str(image_size),
            "--num-workers",
            "4",
        ],
        cwd=repo,
    )

    run(
        [
            sys.executable,
            "scripts/evaluate_quality.py",
            "--checkpoint",
            str(out_dir / "checkpoints/best.pt"),
            "--content-font",
            str(font),
            "--text-file",
            "configs/eval_chars_stage1.txt",
            "--out-dir",
            str(out_dir / "fixed_eval_best"),
            "--image-size",
            str(image_size),
        ],
        cwd=repo,
    )

    result = {
        "run_name": run_name,
        "image_size": image_size,
        "epochs": epochs,
        "batch_size": batch_size,
        "data_dir": str(REMOTE_PREPARED_256),
        "out_dir": str(out_dir),
        "best_checkpoint": str(out_dir / "checkpoints/best.pt"),
        "eval_board": str(out_dir / "fixed_eval_best/eval_board.png"),
        "quality_metrics": str(out_dir / "fixed_eval_best/quality_metrics.csv"),
    }
    (out_dir / "modal_result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    VOL.commit()
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


@app.local_entrypoint()
def main(
    writer: str = "楷-赵孟俯三门记",
    image_size: int = 256,
    epochs: int = 100,
    batch_size: int = 16,
    max_items: int = 0,
    detach: bool = True,
    prepared: bool = True,
):
    if prepared:
        if not LOCAL_PREPARED_256_TAR.exists():
            raise FileNotFoundError(f"local prepared dataset tar not found: {LOCAL_PREPARED_256_TAR.resolve()}")
        print(f"uploading prepared dataset tar to Modal Volume {VOLUME_NAME}", flush=True)
        with VOL.batch_upload(force=True) as batch:
            batch.put_file(str(LOCAL_PREPARED_256_TAR), str(REMOTE_PREPARED_256_TAR_IN_VOLUME))
        if detach:
            call = train_prepared_unet_remote.spawn(image_size, epochs, batch_size)
            print(f"spawned_call_id={call.object_id}", flush=True)
            print(f"volume={VOLUME_NAME}", flush=True)
        else:
            result = train_prepared_unet_remote.remote(image_size, epochs, batch_size)
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return

    if not LOCAL_ZIP.exists():
        raise FileNotFoundError(f"local dataset zip not found: {LOCAL_ZIP.resolve()}")

    print(f"uploading dataset zip to Modal Volume {VOLUME_NAME} if needed", flush=True)
    with VOL.batch_upload(force=True) as batch:
        batch.put_file(str(LOCAL_ZIP), str(REMOTE_ZIP_IN_VOLUME))

    if detach:
        call = train_unet_remote.spawn(writer, image_size, epochs, batch_size, max_items)
        print(f"spawned_call_id={call.object_id}", flush=True)
        print(f"volume={VOLUME_NAME}", flush=True)
        print("monitor with: python3 -m modal app list", flush=True)
    else:
        result = train_unet_remote.remote(writer, image_size, epochs, batch_size, max_items)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
