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
DEFAULT_CONFIG = "configs/branch1_clean_512_64_scale052_structure_gate.json"
DEFAULT_V2_BEST_VISUAL_PROXY = Path("/outputs/modal_structure_unet_branch1_v2_train_1779015934/checkpoints/best_visual_proxy.pt")
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


def remote_volume_path(path: str | Path) -> Path:
    path = Path(path)
    if str(path).startswith(str(VOL_ROOT)):
        return path
    return volume_path(path)


def run(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def candidate_checkpoints(checkpoint_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    candidates.extend(sorted(checkpoint_dir.glob("epoch_*.pt")))
    for name in ["last.pt", "best_val_total.pt", "best_visual_proxy.pt"]:
        path = checkpoint_dir / name
        if path.exists():
            candidates.append(path)

    seen: set[Path] = set()
    unique: list[Path] = []
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


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
    manifests = sorted(out_dir.rglob("manifest.csv"))
    if len(manifests) == 1:
        return manifests[0].parent
    raise FileNotFoundError(f"manifest.csv not found after extracting {tar_path} into {out_dir}")


def safe_extract_split_tar(tar_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tf:
        def is_safe(member: tarfile.TarInfo) -> bool:
            target = (out_dir / member.name).resolve()
            return str(target).startswith(str(out_dir.resolve()))

        members = tf.getmembers()
        if not all(is_safe(member) for member in members):
            raise ValueError(f"unsafe tar member in {tar_path}")
        tf.extractall(out_dir, members=members)

    candidates = [out_dir] + [path.parent for path in out_dir.rglob("train_manifest.csv")]
    for candidate in candidates:
        if (candidate / "train_manifest.csv").exists() and (candidate / "val_manifest.csv").exists():
            return candidate
    raise FileNotFoundError(f"train_manifest.csv/val_manifest.csv not found after extracting {tar_path} into {out_dir}")


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
    config_path: str = DEFAULT_CONFIG,
    workers: int = 8,
    epochs: int = 1,
    batch_size: int = 4,
    local_tar_in_volume: str = "",
    init_checkpoint: str = "",
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

    config_label = Path(config_path).stem.replace("structure_unet_", "")
    run_name = f"modal_structure_unet_{config_label}_{mode}_{int(time.time())}"
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
        config_path,
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(workers),
    ]
    if init_checkpoint:
        ckpt_path = remote_volume_path(init_checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"init checkpoint not found: {ckpt_path}")
        train_cmd.extend(["--init-checkpoint", str(ckpt_path)])
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
        "config": config_path,
        "init_checkpoint": str(remote_volume_path(init_checkpoint)) if init_checkpoint else "",
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
    config_path: str = DEFAULT_CONFIG,
    workers: int = 8,
    epochs: int = 100,
    batch_size: int = 16,
    local_tar_in_volume: str = "",
    local_split_tar_in_volume: str = "",
    init_checkpoint: str = "",
    result_mode: str = "train",
    max_train_items: int = 0,
    max_val_items: int = 0,
    eval_after: bool = True,
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

    if local_split_tar_in_volume:
        split_tar_path = volume_path(local_split_tar_in_volume)
        if not split_tar_path.exists():
            raise FileNotFoundError(f"uploaded split tar not found: {split_tar_path}")
        split_dir = safe_extract_split_tar(split_tar_path, split_dir)
    else:
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
        config_path,
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--num-workers",
        str(workers),
    ]
    if init_checkpoint:
        ckpt_path = remote_volume_path(init_checkpoint)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"init checkpoint not found: {ckpt_path}")
        train_cmd.extend(["--init-checkpoint", str(ckpt_path)])
    if max_train_items > 0:
        train_cmd.extend(["--max-train-items", str(max_train_items)])
    if max_val_items > 0:
        train_cmd.extend(["--max-val-items", str(max_val_items)])
    run(train_cmd, cwd=repo)

    eval_results: dict[str, str] = {}
    shortcut_results: dict[str, str] = {}
    if eval_after:
        for ckpt in candidate_checkpoints(out_dir / "checkpoints"):
            checkpoint_name = ckpt.name
            for output_key in ["final_ink", "mask"]:
                eval_out = out_dir / "shape_audit" / f"{checkpoint_name.replace('.pt', '')}_{output_key}"
                eval_cmd = [
                    sys.executable,
                    "scripts/eval_alpha_shape_metrics.py",
                    "--data-dir",
                    str(data_dir),
                    "--split-dir",
                    str(split_dir),
                    "--model-type",
                    "structure_unet_branch1",
                    "--ckpt",
                    str(ckpt),
                    "--out-dir",
                    str(eval_out),
                    "--output-key",
                    output_key,
                    "--batch-size",
                    str(batch_size),
                    "--device",
                    "cuda",
                    "--preview-items",
                    "36",
                    "--skip-per-sample",
                ]
                if max_train_items > 0:
                    eval_cmd.extend(["--max-train-items", str(max_train_items)])
                if max_val_items > 0:
                    eval_cmd.extend(["--max-val-items", str(max_val_items)])
                run(eval_cmd, cwd=repo)
                eval_results[f"{checkpoint_name}:{output_key}"] = str(eval_out / "result.json")
                if output_key == "final_ink":
                    shortcut_out = out_dir / "shortcut_audit" / checkpoint_name.replace(".pt", "")
                    shortcut_cmd = [
                        sys.executable,
                        "scripts/diagnose_content_copy_shortcut.py",
                        "--data-dir",
                        str(data_dir),
                        "--split-dir",
                        str(split_dir),
                        "--ckpt",
                        str(ckpt),
                        "--out-dir",
                        str(shortcut_out),
                        "--output-key",
                        output_key,
                        "--split",
                        "val",
                        "--batch-size",
                        str(batch_size),
                        "--device",
                        "cuda",
                    ]
                    run(shortcut_cmd, cwd=repo)
                    shortcut_results[f"{checkpoint_name}:{output_key}"] = str(shortcut_out / "result.json")

    result = {
        "mode": result_mode,
        "run_name": run_name,
        "volume": VOLUME_NAME,
        "data_dir": str(data_dir),
        "split_dir": str(split_dir),
        "out_dir": str(out_dir),
        "workers": workers,
        "batch_size": batch_size,
        "config": config_path,
        "init_checkpoint": str(remote_volume_path(init_checkpoint)) if init_checkpoint else "",
        "epochs": epochs,
        "max_train_items": max_train_items,
        "max_val_items": max_val_items,
        "loss_scale_report": str(out_dir / "loss_scale_report.json"),
        "train_log": str(out_dir / "train_log.csv"),
        "last_checkpoint": str(out_dir / "checkpoints/last.pt"),
        "previews": str(out_dir / "previews"),
        "shape_audit_results": eval_results,
        "shortcut_audit_results": shortcut_results,
    }
    result_path = out_dir / REMOTE_TRAIN_RESULT_NAME
    result["result_path"] = str(result_path)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    VOL.commit()
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


@app.function(
    image=image,
    gpu="A10G",
    volumes={"/vol": VOL},
    timeout=60 * 60 * 4,
    cpu=8,
    memory=32768,
)
def run_structure_eval_existing(
    run_name: str,
    remote_structure_dir: str,
    split_dir: str,
    batch_size: int = 16,
    max_train_items: int = 0,
    max_val_items: int = 0,
) -> dict:
    repo = Path("/root/calligraphy")
    data_dir = volume_path(remote_structure_dir)
    resolved_split_dir = remote_volume_path(split_dir)
    out_dir = VOL_ROOT / "outputs" / run_name
    if not (data_dir / "manifest.csv").exists():
        raise FileNotFoundError(f"structure manifest not found: {data_dir / 'manifest.csv'}")
    if not (resolved_split_dir / "train_manifest.csv").exists() or not (resolved_split_dir / "val_manifest.csv").exists():
        raise FileNotFoundError(f"train/val manifests not found in split dir: {resolved_split_dir}")
    if not (out_dir / "checkpoints").exists():
        raise FileNotFoundError(f"checkpoint directory not found: {out_dir / 'checkpoints'}")

    eval_results: dict[str, str] = {}
    shortcut_results: dict[str, str] = {}
    for ckpt in candidate_checkpoints(out_dir / "checkpoints"):
        checkpoint_name = ckpt.name
        for output_key in ["final_ink", "mask"]:
            eval_out = out_dir / "shape_audit" / f"{checkpoint_name.replace('.pt', '')}_{output_key}"
            eval_cmd = [
                sys.executable,
                "scripts/eval_alpha_shape_metrics.py",
                "--data-dir",
                str(data_dir),
                "--split-dir",
                str(resolved_split_dir),
                "--model-type",
                "structure_unet_branch1",
                "--ckpt",
                str(ckpt),
                "--out-dir",
                str(eval_out),
                "--output-key",
                output_key,
                "--batch-size",
                str(batch_size),
                "--device",
                "cuda",
                "--preview-items",
                "36",
                "--skip-per-sample",
            ]
            if max_train_items > 0:
                eval_cmd.extend(["--max-train-items", str(max_train_items)])
            if max_val_items > 0:
                eval_cmd.extend(["--max-val-items", str(max_val_items)])
            run(eval_cmd, cwd=repo)
            eval_results[f"{checkpoint_name}:{output_key}"] = str(eval_out / "result.json")
            if output_key == "final_ink":
                shortcut_out = out_dir / "shortcut_audit" / checkpoint_name.replace(".pt", "")
                shortcut_cmd = [
                    sys.executable,
                    "scripts/diagnose_content_copy_shortcut.py",
                    "--data-dir",
                    str(data_dir),
                    "--split-dir",
                    str(resolved_split_dir),
                    "--ckpt",
                    str(ckpt),
                    "--out-dir",
                    str(shortcut_out),
                    "--output-key",
                    output_key,
                    "--split",
                    "val",
                    "--batch-size",
                    str(batch_size),
                    "--device",
                    "cuda",
                ]
                run(shortcut_cmd, cwd=repo)
                shortcut_results[f"{checkpoint_name}:{output_key}"] = str(shortcut_out / "result.json")

    result = {
        "mode": "eval_existing",
        "run_name": run_name,
        "volume": VOLUME_NAME,
        "data_dir": str(data_dir),
        "split_dir": str(resolved_split_dir),
        "out_dir": str(out_dir),
        "batch_size": batch_size,
        "max_train_items": max_train_items,
        "max_val_items": max_val_items,
        "shape_audit_results": eval_results,
        "shortcut_audit_results": shortcut_results,
    }
    result_path = out_dir / "modal_structure_eval_existing_result.json"
    result["result_path"] = str(result_path)
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    VOL.commit()
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    return result


def prepared_result(
    remote_structure_dir: str,
    config_path: str,
    workers: int,
    epochs: int,
    batch_size: int,
    init_checkpoint: str,
    mode: str,
    max_train_items: int = 0,
    max_val_items: int = 0,
) -> dict:
    command = [
        "python3",
        "-m",
        "modal",
        "run",
        "--detach",
        "scripts/modal_train_structure_unet.py",
        "--mode",
        mode,
        "--remote-structure-dir",
        remote_structure_dir,
        "--config",
        config_path,
        "--workers",
        str(workers),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(batch_size),
        "--detach",
    ]
    if init_checkpoint:
        command.extend(["--init-checkpoint", init_checkpoint])
    if max_train_items > 0:
        command.extend(["--max-train-items", str(max_train_items)])
    if max_val_items > 0:
        command.extend(["--max-val-items", str(max_val_items)])
    return {
        "mode": "prepared",
        "spawned": False,
        "reason": "prepared mode is intentionally command/entrypoint preparation only in this stage",
        "volume": VOLUME_NAME,
        "remote_structure_dir": str(volume_path(remote_structure_dir)),
        "config": config_path,
        "init_checkpoint": str(remote_volume_path(init_checkpoint)) if init_checkpoint else "",
        "max_train_items": max_train_items,
        "max_val_items": max_val_items,
        "full_training_command": command,
        "expected_outputs": {
            "split_dir": "/vol/data/processed/modal_structure_unet_branch1_train_<timestamp>_splits",
            "out_dir": "/vol/outputs/modal_structure_unet_branch1_train_<timestamp>",
            "checkpoint": "checkpoints/epoch_*.pt plus last.pt",
            "train_log": "train_log.csv",
            "loss_scale_report": "loss_scale_report.json",
            "previews": "previews/",
            "result": REMOTE_TRAIN_RESULT_NAME,
        },
    }


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {"missing": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def _round_metrics(values: dict, keys: list[str]) -> dict:
    out: dict[str, float | str | bool | int] = {}
    for key in keys:
        value = values.get(key)
        if isinstance(value, (int, float)):
            out[key] = round(float(value), 4)
        else:
            out[key] = value
    return out


@app.function(
    image=image,
    volumes={"/vol": VOL},
    timeout=60 * 10,
    cpu=2,
    memory=4096,
)
def summarize_structure_run_remote(run_name: str) -> dict:
    out_dir = VOL_ROOT / "outputs" / run_name
    result = _load_json(out_dir / REMOTE_TRAIN_RESULT_NAME)
    data_contract = _load_json(out_dir / "data_contract_report.json")
    selectors = ["epoch_003", "last", "best_val_total", "best_visual_proxy"]
    selector_summaries: dict[str, dict] = {}
    for selector in selectors:
        shape = _load_json(out_dir / "shape_audit" / f"{selector}_final_ink" / "result.json")
        shortcut = _load_json(out_dir / "shortcut_audit" / selector / "result.json")
        if "missing" in shape or "missing" in shortcut:
            selector_summaries[selector] = {"shape": shape, "shortcut": shortcut}
            continue
        val = shape.get("summary", {}).get("val", {})
        gate = shape.get("diagnostic_gate", {})
        shortcut_val = shortcut.get("splits", {}).get("val", {})
        selector_summaries[selector] = {
            "shape": _round_metrics(
                val,
                [
                    "alpha_binary_dice",
                    "bbox_iou",
                    "boundary_f1_tol2",
                    "outside_pred_fraction",
                    "pred_area_ratio",
                    "hole_preserve",
                    "target_hole_preserve",
                    "shape_gain",
                    "target_precision",
                    "target_recall",
                ],
            ),
            "metric_gate_pass": bool(gate.get("overall_pass", False)),
            "shortcut": _round_metrics(
                shortcut_val,
                [
                    "pred_content_dice_mean",
                    "pred_target_dice_mean",
                    "pred_content_bbox_iou_mean",
                    "pred_target_bbox_iou_mean",
                    "pred_content_boundary_f1_tol2_mean",
                    "pred_target_boundary_f1_tol2_mean",
                    "content_like_majority_count",
                    "content_like_majority_fraction",
                    "target_like_majority_count",
                    "target_like_majority_fraction",
                ],
            ),
            "shortcut_gate": shortcut_val.get("shortcut_gate", {}),
        }

    primary: dict = {}
    for selector in ["last", "epoch_003", "best_val_total", "best_visual_proxy"]:
        candidate = selector_summaries.get(selector, {})
        if isinstance(candidate, dict) and "shortcut_gate" in candidate:
            primary = candidate
            break
    shortcut_gate = primary.get("shortcut_gate", {}) if isinstance(primary, dict) else {}
    metric_gate_pass = bool(primary.get("metric_gate_pass", False)) if isinstance(primary, dict) else False
    decision = "diagnose"
    reason = "missing primary selector summary"
    if shortcut_gate.get("hard_veto") or shortcut_gate.get("can_promote_or_continue") is False:
        decision = "reject"
        reason = "shortcut gate hard veto or cannot promote/continue"
    elif not metric_gate_pass:
        decision = "diagnose"
        reason = "shortcut gate did not hard veto, but metric gate failed"
    else:
        decision = "continue"
        reason = "metric gate and shortcut gate allow continued investigation; visual/generalization review still required"

    summary = {
        "mode": "remote_summary",
        "run_name": run_name,
        "out_dir": str(out_dir),
        "result_exists": "missing" not in result,
        "data_contract_pass": bool(data_contract.get("pass", False)),
        "decision": decision,
        "reason": reason,
        "selectors": selector_summaries,
        "result_path": str(out_dir / REMOTE_TRAIN_RESULT_NAME),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


@app.local_entrypoint()
def main(
    mode: str = "calibrate",
    workers: int = 8,
    detach: bool = True,
    remote_structure_dir: str = str(DEFAULT_REMOTE_STRUCTURE_DIR),
    config: str = DEFAULT_CONFIG,
    local_structure_tar: str = "",
    local_split_tar: str = "",
    epochs: int = 1,
    batch_size: int = 4,
    init_checkpoint: str = "",
    max_train_items: int = 0,
    max_val_items: int = 0,
    eval_run_name: str = "",
    eval_split_dir: str = "",
) -> None:
    if mode not in {"calibrate", "smoke", "prepared", "train", "probe", "eval_existing", "summarize"}:
        raise ValueError(f"mode must be calibrate, smoke, prepared, train, probe, eval_existing, or summarize; got {mode}")

    local_tar_in_volume = ""
    local_split_tar_in_volume = ""
    if local_structure_tar:
        local_tar = Path(local_structure_tar).expanduser().resolve()
        if not local_tar.exists():
            raise FileNotFoundError(f"local structure tar not found: {local_tar}")
        local_tar_in_volume = f"/data/processed/{local_tar.name}"
        print(f"uploading structure tar to Modal Volume {VOLUME_NAME}: {local_tar}", flush=True)
        with VOL.batch_upload(force=True) as batch:
            batch.put_file(str(local_tar), local_tar_in_volume)
    if local_split_tar:
        split_tar = Path(local_split_tar).expanduser().resolve()
        if not split_tar.exists():
            raise FileNotFoundError(f"local split tar not found: {split_tar}")
        local_split_tar_in_volume = f"/data/processed/{split_tar.name}"
        print(f"uploading split tar to Modal Volume {VOLUME_NAME}: {split_tar}", flush=True)
        with VOL.batch_upload(force=True) as batch:
            batch.put_file(str(split_tar), local_split_tar_in_volume)

    if mode == "prepared":
        result = prepared_result(remote_structure_dir, config, workers, epochs, batch_size, init_checkpoint, "train", max_train_items, max_val_items)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return

    if mode == "summarize":
        if not eval_run_name:
            raise ValueError("summarize requires --eval-run-name")
        result = summarize_structure_run_remote.remote(eval_run_name)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return

    if mode == "eval_existing":
        if not eval_run_name or not eval_split_dir:
            raise ValueError("eval_existing requires --eval-run-name and --eval-split-dir")
        if detach:
            call = run_structure_eval_existing.spawn(
                eval_run_name,
                remote_structure_dir,
                eval_split_dir,
                batch_size,
                max_train_items,
                max_val_items,
            )
            print(
                json.dumps(
                    {
                        "mode": mode,
                        "run_name": eval_run_name,
                        "volume": VOLUME_NAME,
                        "remote_structure_dir": str(volume_path(remote_structure_dir)),
                        "split_dir": str(remote_volume_path(eval_split_dir)),
                        "spawned_call_id": call.object_id,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                flush=True,
            )
        else:
            result = run_structure_eval_existing.remote(
                eval_run_name,
                remote_structure_dir,
                eval_split_dir,
                batch_size,
                max_train_items,
                max_val_items,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return

    if mode in {"train", "probe"}:
        if mode == "probe" and not init_checkpoint:
            init_checkpoint = str(DEFAULT_V2_BEST_VISUAL_PROXY)
        config_label = Path(config).stem.replace("structure_unet_", "")
        run_name = f"modal_structure_unet_{config_label}_{mode}_{int(time.time())}"
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
            "config": config,
            "init_checkpoint": str(remote_volume_path(init_checkpoint)) if init_checkpoint else "",
            "local_split_tar_in_volume": local_split_tar_in_volume,
            "max_train_items": max_train_items,
            "max_val_items": max_val_items,
        }
        if detach:
            planned["detach_note"] = "Durable background training requires Modal CLI --detach before the script path."
            call = run_structure_train_full.spawn(
                run_name,
                remote_structure_dir,
                config,
                workers,
                epochs,
                batch_size,
                local_tar_in_volume,
                local_split_tar_in_volume,
                init_checkpoint,
                mode,
                max_train_items,
                max_val_items,
                True,
            )
            planned["spawned_call_id"] = call.object_id
            print(json.dumps(planned, ensure_ascii=False, indent=2), flush=True)
        else:
            print(json.dumps(planned, ensure_ascii=False, indent=2), flush=True)
            result = run_structure_train_full.remote(
                run_name,
                remote_structure_dir,
                config,
                workers,
                epochs,
                batch_size,
                local_tar_in_volume,
                local_split_tar_in_volume,
                init_checkpoint,
                mode,
                max_train_items,
                max_val_items,
                True,
            )
            print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
        return

    if detach:
        call = run_structure_train_prep.spawn(mode, remote_structure_dir, config, workers, epochs, batch_size, local_tar_in_volume, init_checkpoint)
        print(f"spawned_call_id={call.object_id}", flush=True)
        print(f"volume={VOLUME_NAME}", flush=True)
        print(f"mode={mode}", flush=True)
    else:
        result = run_structure_train_prep.remote(mode, remote_structure_dir, config, workers, epochs, batch_size, local_tar_in_volume, init_checkpoint)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
