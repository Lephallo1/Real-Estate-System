from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from _bootstrap import ensure_project_root

ensure_project_root()

from lesotho_property_ai.web.admin_actions import ADMIN_ACTIONS
from lesotho_property_ai.web.task_actions import (
    ensure_job_files,
    stderr_path,
    stdout_path,
    write_job_status,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one admin dashboard action in the background.")
    parser.add_argument("--base-dir", default=".")
    parser.add_argument("--action-key", required=True, choices=sorted(ADMIN_ACTIONS))
    return parser


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> int:
    args = build_parser().parse_args()
    root = Path(args.base_dir).resolve()
    spec = ADMIN_ACTIONS[args.action_key]
    ensure_job_files(root, spec.key)

    stdout_log = stdout_path(root, spec.key)
    stderr_log = stderr_path(root, spec.key)
    write_job_status(
        root,
        spec.key,
        {
            "status": "running",
            "message": f"{spec.label} is running.",
        },
    )

    with stdout_log.open("a", encoding="utf-8") as stdout_handle, stderr_log.open("a", encoding="utf-8") as stderr_handle:
        stdout_handle.write(f"== {spec.label} ==\n")
        stdout_handle.flush()

        for script_name, script_args in spec.commands:
            command = [sys.executable, str(root / "scripts" / script_name), *script_args]
            stdout_handle.write(f"$ {' '.join(command)}\n")
            stdout_handle.flush()

            completed = subprocess.run(
                command,
                cwd=root,
                capture_output=True,
                text=True,
            )
            if completed.stdout:
                stdout_handle.write(completed.stdout)
                if not completed.stdout.endswith("\n"):
                    stdout_handle.write("\n")
            if completed.stderr:
                stderr_handle.write(completed.stderr)
                if not completed.stderr.endswith("\n"):
                    stderr_handle.write("\n")
            stdout_handle.flush()
            stderr_handle.flush()

            if completed.returncode != 0:
                write_job_status(
                    root,
                    spec.key,
                    {
                        "status": "failed",
                        "message": f"{spec.label} failed while running {script_name}.",
                        "finished_at": _utc_now(),
                    },
                )
                return completed.returncode

    write_job_status(
        root,
        spec.key,
        {
            "status": "succeeded",
            "message": f"{spec.label} completed successfully.",
            "finished_at": _utc_now(),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
