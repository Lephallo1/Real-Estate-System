"""Helpers for launching long-running admin actions from Flask routes."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .admin_actions import ADMIN_ACTIONS, AdminActionSpec


@dataclass(slots=True)
class ActionJobState:
    action_key: str
    label: str
    description: str
    status: str
    message: str
    pid: int | None
    started_at: str | None
    finished_at: str | None
    updated_at: str | None
    stdout: str
    stderr: str
    can_run_here: bool
    availability_message: str
    running: bool
    status_tone: str


def get_action_spec(action_key: str) -> AdminActionSpec:
    return ADMIN_ACTIONS[action_key]


def action_choices() -> list[dict[str, str]]:
    return [{"key": spec.key, "label": spec.label} for spec in ADMIN_ACTIONS.values()]


def is_railway_runtime(environ: dict[str, str] | None = None) -> bool:
    env = environ or os.environ
    return bool(
        env.get("RAILWAY_ENVIRONMENT")
        or env.get("RAILWAY_PROJECT_ID")
        or env.get("RAILWAY_SERVICE_ID")
    )


def action_availability(action_key: str, *, environ: dict[str, str] | None = None) -> tuple[bool, str]:
    spec = get_action_spec(action_key)
    if spec.local_only and is_railway_runtime(environ=environ):
        return (
            False,
            "Full training is disabled on Railway. Run this module locally from the terminal or the training notebook.",
        )
    return True, ""


def admin_job_dir(base_dir: str | Path, action_key: str) -> Path:
    return Path(base_dir) / "generated" / "artifacts" / "admin_jobs" / action_key


def status_path(base_dir: str | Path, action_key: str) -> Path:
    return admin_job_dir(base_dir, action_key) / "status.json"


def stdout_path(base_dir: str | Path, action_key: str) -> Path:
    return admin_job_dir(base_dir, action_key) / "stdout.log"


def stderr_path(base_dir: str | Path, action_key: str) -> Path:
    return admin_job_dir(base_dir, action_key) / "stderr.log"


def ensure_job_files(base_dir: str | Path, action_key: str) -> None:
    job_dir = admin_job_dir(base_dir, action_key)
    job_dir.mkdir(parents=True, exist_ok=True)
    for path in (stdout_path(base_dir, action_key), stderr_path(base_dir, action_key)):
        if not path.exists():
            path.write_text("", encoding="utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_job_status(base_dir: str | Path, action_key: str, payload: dict[str, object]) -> None:
    ensure_job_files(base_dir, action_key)
    merged = {
        **read_job_status(base_dir, action_key),
        "action_key": action_key,
        "updated_at": _utc_now(),
        **payload,
    }
    status_path(base_dir, action_key).write_text(
        json.dumps(merged, indent=2),
        encoding="utf-8",
    )


def read_job_status(base_dir: str | Path, action_key: str) -> dict[str, object]:
    path = status_path(base_dir, action_key)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _tail_text(path: Path, *, limit_chars: int = 4000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    if len(text) <= limit_chars:
        return text
    return text[-limit_chars:]


def read_action_job(base_dir: str | Path, action_key: str) -> ActionJobState:
    spec = get_action_spec(action_key)
    can_run_here, availability_message = action_availability(action_key)
    payload = read_job_status(base_dir, action_key)
    status = str(payload.get("status", "idle") or "idle")
    status_tone = {
        "idle": "neutral",
        "running": "blue",
        "succeeded": "green",
        "failed": "danger",
        "blocked": "orange",
    }.get(status, "neutral")
    message = str(payload.get("message", "") or "")
    if status == "idle" and not message:
        message = "No job has been launched for this module yet."
    return ActionJobState(
        action_key=action_key,
        label=spec.label,
        description=spec.description,
        status=status,
        message=message,
        pid=int(payload["pid"]) if payload.get("pid") else None,
        started_at=str(payload.get("started_at")) if payload.get("started_at") else None,
        finished_at=str(payload.get("finished_at")) if payload.get("finished_at") else None,
        updated_at=str(payload.get("updated_at")) if payload.get("updated_at") else None,
        stdout=_tail_text(stdout_path(base_dir, action_key)),
        stderr=_tail_text(stderr_path(base_dir, action_key)),
        can_run_here=can_run_here,
        availability_message=availability_message,
        running=status == "running",
        status_tone=status_tone,
    )


def start_action_job(base_dir: str | Path, action_key: str) -> ActionJobState:
    """Launch one admin action in the background and return the latest state."""

    root = Path(base_dir)
    ensure_job_files(root, action_key)
    current = read_action_job(root, action_key)
    if current.running:
        return current

    can_run_here, availability_message = action_availability(action_key)
    if not can_run_here:
        write_job_status(
            root,
            action_key,
            {
                "status": "blocked",
                "message": availability_message,
                "finished_at": _utc_now(),
            },
        )
        return read_action_job(root, action_key)

    stdout_path(root, action_key).write_text("", encoding="utf-8")
    stderr_path(root, action_key).write_text("", encoding="utf-8")

    command = [
        sys.executable,
        str(root / "scripts" / "run_admin_action_job.py"),
        "--base-dir",
        str(root),
        "--action-key",
        action_key,
    ]
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    process = subprocess.Popen(
        command,
        cwd=root,
        text=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=creationflags,
        start_new_session=True,
    )
    write_job_status(
        root,
        action_key,
        {
            "status": "running",
            "message": f"{get_action_spec(action_key).label} started in the background.",
            "pid": process.pid,
            "started_at": _utc_now(),
        },
    )
    return read_action_job(root, action_key)
