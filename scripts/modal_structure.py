from __future__ import annotations

import json
import subprocess
import sys
import tarfile
import time
from pathlib import Path

import modal


APP_NAME = "calligraphy-structure-preprocessing"
VOLUME_NAME = "calligraphy-training"
VOL = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
VOL_ROOT = Path("/vol")

LOCAL_PREPARED_256_TAR = Path("data/processed/zhaomengfu_full_256.tar.gz")
LOCAL_BASELINE_BEST = Path("artifacts/baseline_unet_l1_zhaomengfu_256_100ep/checkpoints/best.pt")

REMOTE_PREPARED_256 = VOL_ROOT / "data/processed/zhaomengfu_full_256"
REMOTE_PREPARED_256_TAR = VOL_ROOT / "data/processed/zhaomengfu_full_256.tar.gz"
REMOTE_PREPARED_256_TAR_IN_VOLUME = Path("/data/processed/zhaomengfu_full_256.tar.gz")
REMOTE_BASELINE_BEST = VOL_ROOT / "artifacts/baseline_unet_l1_zhaomengfu_256_100ep/checkpoints/best.pt"
REMOTE_BASELINE_BEST_IN_VOLUME = Path("/artifacts/baseline_unet_l1_zhaomengfu_256_100ep/checkpoints/best.pt")

image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime")
    .apt_install("fonts-arphic-uming", "fontconfig")
    .pip_install("Pillow>=9.5", "numpy>=1.23", "tqdm>=4.65", "scipy>=1.10", "torch>=2.0")
    .add_local_dir(Path(__file__).resolve().parents[1] / "scripts", remote_path="/root/calligraphy/scripts", copy=True)
    .add_local_dir(Path(__file__).resolve().parents[1] / "configs", remote_path="/root/calligraphy/configs", copy=True)
)

app = modal.App(APP_NAME)


def run(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def ensure_prepared_dataset() -> None:
    if REMOTE_PREPARED_256.exists():
        print(f"prepared dataset already extracted: {REMOTE_PREPARED_256}", flush=True)
        return
    if not REMOTE_PREPARED_256_TAR.exists():
        raise FileNotFoundError(f"prepared dataset tar not found: {REMOTE_PREPARED_256_TAR}")
    print(f"extracting prepared dataset: {REMOTE_PREPARED_256_TAR}", flush=True)
    (VOL_ROOT / "data/processed").mkdir(parents=True, exist_ok=True)
    with tarfile.open(REMOTE_PREPARED_256_TAR, "r:gz") as tf:
        tf.extractall(VOL_ROOT / "data/processed")
    if not REMOTE_PREPARED_256.exists():
        raise FileNotFoundError(f"prepared dataset not found after extract: {REMOTE_PREPARED_256}")


@app.function(
    image=image,
    volumes={"/vol": VOL},
    timeout=60 * 60 * 4,
    cpu=8,
    memory=32768,
)
def build_structure_remote(
    threshold: int = 220,
    workers: int = 8,
    run_eval: bool = True,
) -> dict:
    repo = Path("/root/calligraphy")
    font = Path("/usr/share/fonts/truetype/arphic/uming.ttc")
    ensure_prepared_dataset()

    run_name = f"modal_structure_256_{int(time.time())}"
    structure_dir = VOL_ROOT / "data/processed" / run_name
    eval_dir = VOL_ROOT / "outputs" / f"{run_name}_baseline_eval"
    structure_dir.mkdir(parents=True, exist_ok=True)

    run(
        [
            sys.executable,
            "scripts/build_structure_dataset.py",
            "--data-dir",
            str(REMOTE_PREPARED_256),
            "--out-dir",
            str(structure_dir),
            "--threshold",
            str(threshold),
            "--workers",
            str(workers),
        ],
        cwd=repo,
    )

    result = {
        "run_name": run_name,
        "threshold": threshold,
        "workers": workers,
        "structure_dir": str(structure_dir),
        "structure_manifest": str(structure_dir / "manifest.csv"),
        "structure_summary": str(structure_dir / "structure_summary.json"),
    }

    if run_eval:
        if not REMOTE_BASELINE_BEST.exists():
            raise FileNotFoundError(f"baseline checkpoint not found: {REMOTE_BASELINE_BEST}")
        run(
            [
                sys.executable,
                "scripts/evaluate_quality.py",
                "--checkpoint",
                str(REMOTE_BASELINE_BEST),
                "--content-font",
                str(font),
                "--groups-file",
                "configs/eval_groups_stage1.json",
                "--out-dir",
                str(eval_dir),
                "--image-size",
                "256",
            ],
            cwd=repo,
        )
        result.update(
            {
                "eval_dir": str(eval_dir),
                "eval_board": str(eval_dir / "eval_board.png"),
                "quality_metrics": str(eval_dir / "quality_metrics.csv"),
                "quality_summary": str(eval_dir / "quality_summary.json"),
            }
        )

    result_path = VOL_ROOT / "outputs" / run_name / "modal_structure_result.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    result["result_path"] = str(result_path)
    VOL.commit()
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


@app.local_entrypoint()
def main(
    threshold: int = 220,
    workers: int = 8,
    detach: bool = True,
    run_eval: bool = True,
) -> None:
    if not LOCAL_PREPARED_256_TAR.exists():
        raise FileNotFoundError(f"local prepared dataset tar not found: {LOCAL_PREPARED_256_TAR.resolve()}")
    print(f"uploading prepared dataset tar to Modal Volume {VOLUME_NAME}", flush=True)
    with VOL.batch_upload(force=True) as batch:
        batch.put_file(str(LOCAL_PREPARED_256_TAR), str(REMOTE_PREPARED_256_TAR_IN_VOLUME))

    if run_eval:
        if not LOCAL_BASELINE_BEST.exists():
            raise FileNotFoundError(f"local baseline checkpoint not found: {LOCAL_BASELINE_BEST.resolve()}")
        print(f"uploading baseline checkpoint to Modal Volume {VOLUME_NAME}", flush=True)
        with VOL.batch_upload(force=True) as batch:
            batch.put_file(str(LOCAL_BASELINE_BEST), str(REMOTE_BASELINE_BEST_IN_VOLUME))

    if detach:
        call = build_structure_remote.spawn(threshold, workers, run_eval)
        print(f"spawned_call_id={call.object_id}", flush=True)
        print(f"volume={VOLUME_NAME}", flush=True)
        print("task=structure_preprocessing_and_optional_eval", flush=True)
    else:
        result = build_structure_remote.remote(threshold, workers, run_eval)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
