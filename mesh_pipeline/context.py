"""Pipeline execution context and per-phase result type.

Holds the job directory layout and the object "name registry" that phases use
instead of caching bpy.types.Object references (stale references are the #1
headless-Blender footgun — see docs/blender-body-mesh-cleanup.md Phase 0).

This module is written so that PhaseResult (and PipelineContext construction)
can be imported and unit-tested without bpy installed; bpy is only imported
lazily inside methods that actually touch the Blender scene.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PhaseResult:
    """Outcome of a single phase run, as recorded into report.json."""

    phase: str  # e.g. "p2_weld_split"
    status: str  # "ok" | "needs_review" | "failed"
    metrics: dict = field(default_factory=dict)  # JSON-serializable before/after numbers
    failures: list[str] = field(default_factory=list)  # assertion failure messages
    notes: list[str] = field(default_factory=list)  # informational messages


class PipelineContext:
    """Per-job state shared across phases.

    job_dir must already contain input/ output/ snapshots/ qa_renders/
    (the CLI creates these before running phases).
    """

    def __init__(
        self,
        job_dir: Path,
        names: dict[str, str] | None = None,
        backup_names: dict[str, str] | None = None,
    ) -> None:
        self.job_dir = Path(job_dir)
        self.names: dict[str, str] = dict(names) if names else {}
        self.backup_names: dict[str, str] = dict(backup_names) if backup_names else {}

    def obj(self, role: str):
        """Re-fetch bpy.data.objects[self.names[role]] every call.

        Never cache the returned Object across operator calls -- call this
        again after any bpy.ops call that might invalidate references.
        """
        import bpy

        if role not in self.names:
            raise KeyError(
                f"PipelineContext.obj: no object registered for role {role!r}. "
                f"Known roles: {sorted(self.names)}"
            )
        name = self.names[role]
        try:
            return bpy.data.objects[name]
        except KeyError as exc:
            raise KeyError(
                f"PipelineContext.obj: role {role!r} maps to object name {name!r}, "
                f"but no such object exists in bpy.data.objects "
                f"(known objects: {sorted(o.name for o in bpy.data.objects)})"
            ) from exc

    def save_snapshot(self, tag: str) -> Path:
        """Save a copy of the current .blend to snapshots/<tag>.blend.

        Uses copy=True so the in-memory session (current filepath, undo stack)
        is left untouched -- this is a side-save, not a "save as". The name
        registry is saved alongside as <tag>.names.json so a resumed run
        (--start-phase) can restore ctx.names / ctx.backup_names.
        """
        import bpy

        path = self.job_dir / "snapshots" / f"{tag}.blend"
        path.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(path), copy=True)
        registry = {"names": self.names, "backup_names": self.backup_names}
        (path.with_suffix(".names.json")).write_text(
            json.dumps(registry, indent=2, sort_keys=True)
        )
        return path

    def load_registry(self, tag: str) -> bool:
        """Restore names/backup_names from snapshots/<tag>.names.json.

        Returns True if the registry file existed and was loaded.
        """
        path = self.job_dir / "snapshots" / f"{tag}.names.json"
        if not path.exists():
            return False
        registry = json.loads(path.read_text())
        self.names = dict(registry.get("names", {}))
        self.backup_names = dict(registry.get("backup_names", {}))
        return True

    def ensure_object_mode(self) -> None:
        """Defensively force every object into OBJECT mode.

        Headless jobs shouldn't ever be in Edit Mode between phases, but a
        prior phase leaving an object in Edit Mode would break primitive_*_add
        (injects geometry into the edited mesh) and bm.to_mesh() calls, so we
        re-assert this at phase boundaries rather than trusting it.
        """
        import bpy

        for o in bpy.data.objects:
            if o.mode != "OBJECT":
                bpy.context.view_layer.objects.active = o
                bpy.ops.object.mode_set(mode="OBJECT")
