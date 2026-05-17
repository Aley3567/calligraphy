from __future__ import annotations

import json
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import modal


APP_NAME = "calligraphy-structure-unet-training-prep"
VOLUME_NAME = "calligraphy-training"
VOL = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
VOL_ROOT = Path("/vol")

DEFAULT_REMOTE_STRUCTURE_DIR = Path("/data/processed/modal_structure_256_1778997487")
REMOTE_RESULT_NAME = "modal_structure_train_prep_result.json"
REMOTE_TRAIN_RESULT_NAME = "modal_structure_train_result.json"

image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime")
    .pip_install("Pillow>=9.5", "numpy>=1.23", "tqdm>=4.65", "scipy>=1.10", "torch>=2.0")
    .add_local_dir(Path(__file__).resolve().parents[1] / "scripts", remote_path="/root/calligraphy/scripts", copy=True)
    .add_local_dir(Path(__file__).resolve().parents[1] / "configs", remote_path="/root/calligraphy/configs", copy=True)
)

app = modal.App(APP_NAME)


def volume_path(path: str | Path) -> Path:
    path = Path(path)
    return VOL_ROOT / str(path).lstrip("/")


def run(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def safe_extract_tar(tar_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tf:
        def is_safe(member: tarfile.TarInfo) -> bool:
            target = (out_dir / member.name).resolve()
            return str(target).startswith(str(out_dir.resolve()))

        members = tf.getmembers()
        if not all(is_safe(member) for member in members):
            raise ValueError(f"unsafe tar member in {tar_path}")
        top_level = sorted({Path(member.name).parts[0] for member in members if Path(member.name).parts})
        tf.extractall(out_dir, members=members)
    if len(top_level) == 1 and (out_dir / top_level[0] / "manifest.csv").exists():
        return out_dir / top_level[0]
    if (out_dir / "manifest.csv").exists():
        return out_dir
    raise FileNotFoundError(f"manifest.csv not found after extracting {tar_path} into {out_dir}")


@app.function(
    image=image,
    volumes={"/vol": VOL},
    timeout=60 * 60 * 4,
    cpu=8,
    memory=32768,
)
def run_structure_train_prep(
    mode: str,
    remote_structure_dir: str,
    workers: int = 8,
    epochs: int = 1,
    batch_size: int = 4,
    local_tar_in_volume: str = "",
) -> dict:
    if mode not in {"calibrate", "smoke"}:
        raise ValueError(f"remote execution only supports calibrate/smoke, got {mode}")

    repo = Path("/root/calligraphy")
    data_dir = volume_path(remote_structure_dir)
    if local_tar_in_volume:
        tar_path = volume_path(local_tar_in_volume)
        if not tar_path.exists():
            raise FileNotFoundError(f"uploaded structure tar not found: {tar_path}")
        extract_root = VOL_ROOT / "data/processed" / f"uploaded_structure_{int(time.time())}"
        data_dir = safe_extract_tar(tar_path, extract_root)

    if not (data_dir / "manifest.csv").exists():
        raise FileNotFoundError(f"structure manifest not found: {data_dir / 'manifest.csv'}")

    run_name = f"modal_structure_unet_branch1_{mode}_{int(time.time())}"
    split_dir = VOL_ROOT / "data/processed" / f"{run_name}_splits"
    out_dir = VOL_ROOT / "outputs" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    run(
        [
            sys.executable,
            "scripts/prepare_structure_splits.py",
            "--data-dir",
            str(data_dir),
            "--out-dir",
            str(split_dir),
            "--seed",
            "42",
        ],
        cwd=repo,
    )

    train_cmd = [
        sys.executable,
        "scripts/train_structure_unet.py",
        "--data-dir",
        str(data_dir),
        "--split-dir",
        str(split_dir),
        "--out-dir",
        str(out_dir),
        "--config",
        "configs/structure_unet_branch1.json",
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(workers),
    ]
    if mode == "calibrate":
        train_cmd.extend(["--max-train-items", "8", "--max-val-items", "4", "--calibrate-only"])
    else:
        train_cmd.extend(["--epochs", str(epochs), "--max-train-items", "32", "--max-val-items", "16"])
    run(train_cmd, cwd=repo)

    result = {
        "mode": mode,
        "run_name": run_name,
        "volume": VOLUME_NAME,
        "data_dir": str(data_dir),
        "split_dir": str(split_dir),
        "out_dir": str(out_dir),
        "workers": workers,
        "batch_size": batch_size,
        "epochs": epochs if mode == "smoke" else 0,
        "loss_scale_report": str(out_dir / "loss_scale_report.json"),
        "train_log": str(out_dir / "train_log.csv"),
        "last_checkpoint": str(out_dir / "checkpoints/last.pt"),
        "previews": str(out_dir / "previews"),
    }
    result_path = out_dir / REMOTE_RESULT_NAME
    result["result_path"] = str(result_path)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    VOL.commit()
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/vol": VOL},
    timeout=60 * 60 * 12,
    cpu=8,
    memory=32768,
)
def run_structure_train_full(
    run_name: str,
    remote_structure_dir: str,
    workers: int = 8,
    epochs: int = 100,
    batch_size: int = 16,
    local_tar_in_volume: str = "",
) -> dict:
    repo = Path("/root/calligraphy")
    data_dir = volume_path(remote_structure_dir)
    if local_tar_in_volume:
        tar_path = volume_path(local_tar_in_volume)
        if not tar_path.exists():
            raise FileNotFoundError(f"uploaded structure tar not found: {tar_path}")
        extract_root = VOL_ROOT / "data/processed" / f"uploaded_structure_{int(time.time())}"
        data_dir = safe_extract_tar(tar_path, extract_root)

    manifest = data_dir / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"structure manifest not found: {manifest}")
    print(f"structure manifest found: {manifest}", flush=True)

    split_dir = VOL_ROOT / "data/processed" / f"{run_name}_splits"
    out_dir = VOL_ROOT / "outputs" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    run(
        [
            sys.executable,
            "scripts/prepare_structure_splits.py",
            "--data-dir",
            str(data_dir),
            "--out-dir",
            str(split_dir),
            "--seed",
            "42",
        ],
        cwd=repo,
    )

    run(
        [
            sys.executable,
            "scripts/train_structure_unet.py",
            "--data-dir",
            str(data_dir),
            "--split-dir",
            str(split_dir),
            "--out-dir",
            str(out_dir),
            "--config",
            "configs/structure_unet_branch1.json",
            "--epochs",
            str(epochs),
            "--batch-size",
            str(batch_size),
            "--num-workers",
            str(workers),
        ],
        cwd=repo,
    )

    result = {
        "mode": "train",
        "run_name": run_name,
        "volume": VOLUME_NAME,
        "data_dir": str(data_dir),
        "split_dir": str(split_dir),
        "out_dir": str(out_dir),
        "workers": workers,
        "batch_size": batch_size,
        "epochs": epochs,
        "loss_scale_report": str(out_dir / "loss_scale_report.json"),
        "train_log": str(out_dir / "train_log.csv"),
        "last_checkpoint": str(out_dir / "checkpoints/last.pt"),
        "previews": str(out_dir / "previews"),
    }
    result_path = out_dir / REMOTE_TRAIN_RESULT_NAME
    result["result_path"] = str(result_path)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    VOL.commit()
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def prepared_result(remote_structure_dir: str, workers: int, epochs: int, batch_size: int) -> dict:
    command = [
        "python3",
        "-m",
        "modal",
        "run",
        "scripts/modal_train_structure_unet.py",
        "--mode",
        "train",
        "--remote-structure-dir",
        remote_structure_dir,
        "--workers",
        str(workers),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
    ]
    return {
        "mode": "prepared",
        "spawned": False,
        "reason": "prepared mode is intentionally command/entrypoint preparation only in this stage",
        "volume": VOLUME_NAME,
        "remote_structure_dir": str(volume_path(remote_structure_dir)),
        "full_training_command": command,
        "expected_outputs": {
            "split_dir": "/vol/data/processed/modal_structure_unet_branch1_train_<timestamp>_splits",
            "out_dir": "/vol/outputs/modal_structure_unet_branch1_train_<timestamp>",
            "checkpoint": "checkpoints/last.pt",
            "train_log": "train_log.csv",
            "loss_scale_report": "loss_scale_report.json",
            "previews": "previews/",
            "result": REMOTE_TRAIN_RESULT_NAME,
        },
    }


@app.local_entrypoint()
def main(
    mode: str = "calibrate",
    workers: int = 8,
    detach: bool = True,
    remote_structure_dir: str = str(DEFAULT_REMOTE_STRUCTURE_DIR),
    local_structure_tar: str = "",
    epochs: int = 1,
    batch_size: int = 4,
) -> None:
    if mode not in {"calibrate", "smoke", "prepared", "train"}:
        raise ValueError(f"mode must be calibrate, smoke, prepared, or train; got {mode}")

    local_tar_in_volume = ""
    if local_structure_tar:
        local_tar = Path(local_structure_tar).expanduser().resolve()
        if not local_tar.exists():
            raise FileNotFoundError(f"local structure tar not found: {local_tar}")
        local_tar_in_volume = f"/data/processed/{local_tar.name}"
        print(f"uploading structure tar to Modal Volume {VOLUME_NAME}: {local_tar}", flush=True)
        with VOL.batch_upload(force=True) as batch:
            batch.put_file(str(local_tar), local_tar_in_volume)

    if mode == "prepared":
        result = prepared_result(remote_structure_dir, workers, epochs, batch_size)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return

    if mode == "train":
        run_name = f"modal_structure_unet_branch1_train_{int(time.time())}"
        out_dir = VOL_ROOT / "outputs" / run_name
        result_path = out_dir / REMOTE_TRAIN_RESULT_NAME
        planned = {
            "mode": mode,
            "run_name": run_name,
            "volume": VOLUME_NAME,
            "remote_structure_dir": str(volume_path(remote_structure_dir)),
            "split_dir": str(VOL_ROOT / "data/processed" / f"{run_name}_splits"),
            "out_dir": str(out_dir),
            "train_log": str(out_dir / "train_log.csv"),
            "last_checkpoint": str(out_dir / "checkpoints/last.pt"),
            "result_path": str(result_path),
            "epochs": epochs,
            "batch_size": batch_size,
            "workers": workers,
        }
        if detach:
            call = run_structure_train_full.spawn(run_name, remote_structure_dir, workers, epochs, batch_size, local_tar_in_volume)
            planned["spawned_call_id"] = call.object_id
            print(json.dumps(planned, ensure_ascii=False, indent=2), flush=True)
        else:
            result = run_structure_train_full.remote(run_name, remote_structure_dir, workers, epochs, batch_size, local_tar_in_volume)
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return

    if detach:
        call = run_structure_train_prep.spawn(mode, remote_structure_dir, workers, epochs, batch_size, local_tar_in_volume)
        print(f"spawned_call_id={call.object_id}", flush=True)
        print(f"volume={VOLUME_NAME}", flush=True)
        print(f"mode={mode}", flush=True)
    else:
        result = run_structure_train_prep.remote(mode, remote_structure_dir, workers, epochs, batch_size, local_tar_in_volume)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
