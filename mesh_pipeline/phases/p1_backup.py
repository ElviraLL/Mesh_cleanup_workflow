"""P1 -- Backup.

Duplicate every mesh object into a hidden "_backup_pre_cleanup" collection
before any destructive edit (docs/blender-uv-atlas-rebake.md Phase 1). These
backups keep the original materials + UVs and are later the *bake source* for
p8 (selected-to-active rebake from old geometry onto the cleaned mesh) -- do
not rename materials or touch UVs on them here.
"""

from __future__ import annotations

import bpy

from mesh_pipeline.context import PhaseResult, PipelineContext

PHASE_NAME = "p1_backup"
DESTRUCTIVE = False

BACKUP_COLLECTION_NAME = "_backup_pre_cleanup"
BACKUP_SUFFIX = "_backup"


def run(ctx: PipelineContext, cfg: dict) -> PhaseResult:
    bpy.context.view_layer.update()

    mesh_objects = [
        o
        for o in bpy.data.objects
        if o.type == "MESH" and not o.name.endswith(BACKUP_SUFFIX)
    ]
    if not mesh_objects:
        return PhaseResult(
            phase=PHASE_NAME,
            status="failed",
            failures=["no mesh objects found to back up"],
        )

    col = bpy.data.collections.get(BACKUP_COLLECTION_NAME)
    if col is None:
        col = bpy.data.collections.new(BACKUP_COLLECTION_NAME)
    if col.name not in {c.name for c in bpy.context.scene.collection.children}:
        bpy.context.scene.collection.children.link(col)

    role_by_object_name = {name: role for role, name in ctx.names.items()}

    backed_up: list[str] = []
    for obj in mesh_objects:
        dup = obj.copy()
        dup.data = obj.data.copy()
        dup.name = f"{obj.name}{BACKUP_SUFFIX}"
        col.objects.link(dup)
        dup.hide_set(True)
        dup.hide_render = True

        role = role_by_object_name.get(obj.name, obj.name)
        ctx.backup_names[role] = dup.name
        backed_up.append(obj.name)

    col.hide_viewport = True
    col.hide_render = True

    notes = [
        f"backed up {len(backed_up)} mesh object(s) into '{col.name}' "
        f"(hidden viewport+render)",
        f"backup_names registry: {ctx.backup_names}",
    ]

    metrics = {
        "backed_up": backed_up,
        "backup_collection": col.name,
    }

    return PhaseResult(phase=PHASE_NAME, status="ok", metrics=metrics, notes=notes)
