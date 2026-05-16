from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run(cmd: list[str], timeout: int = 60) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def append_event(path: Path, event: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


def latest_output_dir(volume: str) -> str | None:
    proc = run(["python3", "-m", "modal", "volume", "ls", volume, "/outputs"], timeout=60)
    if proc.returncode != 0:
        return None
    names = [line.strip() for line in proc.stdout.splitlines() if line.strip().startswith("outputs/")]
    if not names:
        return None
    return sorted(names)[-1]


def download_train_log(volume: str, remote_dir: str, local_dir: Path) -> Path | None:
    target = local_dir / "train_log.csv"
    proc = run(
        ["python3", "-m", "modal", "volume", "get", volume, f"/{remote_dir}/train_log.csv", str(target), "--force"],
        timeout=120,
    )
    if proc.returncode != 0 or not target.exists():
        return None
    return target


def parse_last_epoch(log_path: Path) -> dict:
    if log_path.stat().st_size == 0:
        return {}
    with log_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    row = rows[-1]
    parsed = dict(row)
    for key in ["epoch", "train_l1", "val_l1"]:
        if key in parsed:
            try:
                parsed[key] = int(parsed[key]) if key == "epoch" else float(parsed[key])
            except ValueError:
                pass
    return parsed


def app_status(app_id: str) -> str:
    proc = run(["python3", "-m", "modal", "app", "list"], timeout=60)
    if proc.returncode != 0:
        return "unknown"
    for line in proc.stdout.splitlines():
        if app_id in line:
            return line.strip()
    return "not_listed"


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch Modal training by polling app status and Volume artifacts.")
    parser.add_argument("--app-id", required=True)
    parser.add_argument("--volume", default="calligraphy-training")
    parser.add_argument("--interval", default=300, type=int)
    parser.add_argument("--max-epochs", default=100, type=int)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--remote-dir", default="")
    args = parser.parse_args()

    out_dir = args.out_dir.expanduser().resolve()
    state_file = out_dir / "monitor-state.json"
    events_file = out_dir / "events.jsonl"
    result_file = out_dir / "monitor-result.json"
    handoff_file = out_dir / "monitor-handoff.md"
    local_artifacts = out_dir / "artifacts"
    local_artifacts.mkdir(parents=True, exist_ok=True)

    last_epoch = None
    while True:
        event = {"ts": now(), "app_id": args.app_id, "status": app_status(args.app_id)}
        remote_dir = args.remote_dir or latest_output_dir(args.volume)
        event["remote_dir"] = remote_dir
        if remote_dir:
            log_path = download_train_log(args.volume, remote_dir, local_artifacts)
            if log_path:
                epoch_info = parse_last_epoch(log_path)
                event["latest_epoch"] = epoch_info
                current_epoch = epoch_info.get("epoch")
                if current_epoch != last_epoch or current_epoch is None:
                    last_epoch = current_epoch
                    append_event(events_file, event)

                state = {
                    "updated_at": now(),
                    "app_id": args.app_id,
                    "volume": args.volume,
                    "remote_dir": remote_dir,
                    "latest_epoch": epoch_info,
                    "state_file": str(state_file),
                    "events_file": str(events_file),
                    "result_file": str(result_file),
                    "handoff_file": str(handoff_file),
                }
                write_json(state_file, state)

                if current_epoch and int(current_epoch) >= args.max_epochs:
                    write_json(result_file, {"status": "complete", **state})
                    handoff_file.write_text(
                        f"# Modal Training Monitor\n\nstatus: complete\napp_id: {args.app_id}\nremote_dir: {remote_dir}\nlatest_epoch: {epoch_info}\n",
                        encoding="utf-8",
                    )
                    break
            else:
                append_event(events_file, {**event, "note": "train_log_not_available"})
                write_json(state_file, {**event, "updated_at": now()})
        else:
            append_event(events_file, {**event, "note": "remote_dir_not_available"})
            write_json(state_file, {**event, "updated_at": now()})

        if "stopped" in event["status"].lower() and last_epoch != args.max_epochs:
            write_json(result_file, {"status": "stopped_before_complete", "last_event": event})
            handoff_file.write_text(
                f"# Modal Training Monitor\n\nstatus: stopped_before_complete\napp_id: {args.app_id}\nlast_event: {event}\n",
                encoding="utf-8",
            )
            break

        time.sleep(args.interval)


if __name__ == "__main__":
    main()
