"""Accumulates PhaseResult entries into report.json, written atomically.

Report.write_atomic is called after every phase (per ARCHITECTURE.md) so a
crash mid-pipeline always leaves a valid, parseable partial report on disk.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from mesh_pipeline.context import PhaseResult


class Report:
    """In-memory accumulator for a single job's report.json."""

    def __init__(self, input_path: str, config_path: str) -> None:
        self.input = {"path": input_path}
        self.config_path = config_path
        self.started_at = _now_iso()
        self.finished_at: str | None = None
        self.status = "running"
        self.phases: list[dict[str, Any]] = []

    def add_phase(self, result: PhaseResult, duration_s: float) -> None:
        entry = asdict(result)
        entry["duration_s"] = duration_s
        self.phases.append(entry)

    def finalize(self, status: str) -> None:
        self.status = status
        self.finished_at = _now_iso()

    def to_dict(self) -> dict[str, Any]:
        return {
            "input": self.input,
            "config_path": self.config_path,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "status": self.status,
            "phases": self.phases,
        }

    def write_atomic(self, path: Path) -> None:
        """Write report.json via temp file + os.replace (crash-safe)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(self.to_dict(), indent=2, default=str)
        fd, tmp_path = tempfile.mkstemp(
            dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "w") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
