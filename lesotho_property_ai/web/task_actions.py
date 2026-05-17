"""Helpers for launching long-running admin actions from Flask routes."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class ActionResult:
    success: bool
    command: list[str]
    stdout: str
    stderr: str
    returncode: int


def run_script(base_dir: str | Path, script_name: str, *args: str) -> ActionResult:
    """Run one project helper script and capture its output for the dashboard."""

    root = Path(base_dir)
    command = [sys.executable, str(root / "scripts" / script_name), *args]
    completed = subprocess.run(
        command,
        cwd=root,
        capture_output=True,
        text=True,
    )
    return ActionResult(
        success=completed.returncode == 0,
        command=command,
        stdout=completed.stdout,
        stderr=completed.stderr,
        returncode=completed.returncode,
    )
