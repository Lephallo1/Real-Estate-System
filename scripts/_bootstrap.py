"""Helpers for running project scripts from the dedicated `scripts/` folder."""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_project_root() -> Path:
    """Add the project root to `sys.path` so sibling package imports keep working."""

    project_root = Path(__file__).resolve().parents[1]
    project_root_text = str(project_root)
    if project_root_text not in sys.path:
        sys.path.insert(0, project_root_text)
    return project_root
