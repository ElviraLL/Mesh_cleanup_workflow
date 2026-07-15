"""P0 -- Diagnose.

Collects before-touching-anything scene stats (see docs/blender-body-mesh-
cleanup.md Phase 1 and docs/blender-uv-atlas-rebake.md Phase 0) and normalizes
the up-axis per cfg["input"]["up_axis"] (ARCHITECTURE.md Phase 0 pitfalls: do
this before any measurement that depends on world-space bbox orientation).

Never destructive to topology -- only ever rotates whole-object transforms
(matrix_world), never touches vertex/face data.
"""

from __future__ import annotations

import math

import bpy
import bmesh
from mathutils import Matrix, Vector

from mesh_pipeline import geom
from mesh_pipeline.context import PhaseResult, PipelineContext

PHASE_NAME = "p0_diagnose"
DESTRUCTIVE = False

_TOP_N_COMPONENTS = 30

_AXIS_VECTORS: dict[str, Vector] = {
    "x": Vector((1.0, 0.0, 0.0)),
    "-x": Vector((-1.0, 0.0, 0.0)),
    "y": Vector((0.0, 1.0, 0.0)),
    "-y": Vector((0.0, -1.0, 0.0)),
    "z": Vector((0.0, 0.0, 1.0)),
    "-z": Vector((0.0, 0.0, -1.0)),
}


def run(ctx: PipelineContext, cfg: dict) -> PhaseResult:
    mesh_objects = [o for o in bpy.data.objects if o.type == "MESH"]
    if not mesh_objects:
        return PhaseResult(
            phase=PHASE_NAME,
            status="failed",
            failures=["no mesh objects found in scene"],
        )

    notes = _normalize_up_axis(mesh_objects, cfg["input"]["up_axis"])
    bpy.context.view_layer.update()

    total_verts = 0
    total_faces = 0
    total_components = 0
    total_boundary = 0
    total_nonmanifold = 0
    materials: list[str] = []
    modifiers_info: list[dict] = []
    uv_layers_by_object: dict[str, list[str]] = {}
    uv_islands_values: list[int | None] = []
    all_component_entries: list[dict] = []

    for obj in mesh_objects:
        me = obj.data
        bm = bmesh.new()
        bm.from_mesh(me)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()

        v = len(bm.verts)
        f = len(bm.faces)
        boundary = sum(1 for e in bm.edges if e.is_boundary)
        nonmanifold = sum(1 for e in bm.edges if not e.is_manifold)

        details, n_comp = _component_details(bm, obj, _TOP_N_COMPONENTS)
        for d in details:
            d["object"] = obj.name
        all_component_entries.extend(details)

        bm.free()

        total_verts += v
        total_faces += f
        total_boundary += boundary
        total_nonmanifold += nonmanifold
        total_components += n_comp

        for slot in obj.material_slots:
            if slot.material is not None and slot.material.name not in materials:
                materials.append(slot.material.name)

        modifiers_info.append(
            {"object": obj.name, "modifiers": [[m.name, m.type] for m in obj.modifiers]}
        )
        uv_layers_by_object[obj.name] = [uv.name for uv in me.uv_layers]
        uv_islands_values.append(geom.uv_island_count(me))

    all_component_entries.sort(key=lambda d: d["vert_count"], reverse=True)
    top_components = all_component_entries[:_TOP_N_COMPONENTS]

    if all(v is None for v in uv_islands_values):
        uv_islands: int | None = None
    else:
        uv_islands = sum(v for v in uv_islands_values if v is not None)

    metrics = {
        "verts": total_verts,
        "faces": total_faces,
        "components": total_components,
        "boundary_edges": total_boundary,
        "nonmanifold_edges": total_nonmanifold,
        "uv_islands": uv_islands,
        "materials": materials,
        # Extra (non-contract) detail kept for debugging / later phases.
        "components_top30": top_components,
        "modifiers": modifiers_info,
        "uv_layers": uv_layers_by_object,
    }

    notes.append(
        f"scanned {len(mesh_objects)} mesh object(s): verts={total_verts} "
        f"faces={total_faces} components={total_components} "
        f"boundary_edges={total_boundary} nonmanifold_edges={total_nonmanifold} "
        f"uv_islands={uv_islands} materials={materials}"
    )

    return PhaseResult(phase=PHASE_NAME, status="ok", metrics=metrics, notes=notes)


def _component_details(bm, obj, top_n: int) -> tuple[list[dict], int]:
    """Per-connected-component vert count + world-space bbox, largest top_n only."""
    mat = obj.matrix_world
    comps = geom.bm_connected_components(bm)
    comps_sorted = sorted(comps, key=len, reverse=True)
    bm.verts.ensure_lookup_table()

    details: list[dict] = []
    for comp in comps_sorted[:top_n]:
        coords = [mat @ bm.verts[i].co for i in comp]
        xs = [c.x for c in coords]
        ys = [c.y for c in coords]
        zs = [c.z for c in coords]
        details.append(
            {
                "vert_count": len(comp),
                "bbox_center": [
                    (min(xs) + max(xs)) / 2.0,
                    (min(ys) + max(ys)) / 2.0,
                    (min(zs) + max(zs)) / 2.0,
                ],
                "bbox_dims": [
                    max(xs) - min(xs),
                    max(ys) - min(ys),
                    max(zs) - min(zs),
                ],
            }
        )
    return details, len(comps)


def _normalize_up_axis(mesh_objects: list, up_axis_cfg: str) -> list[str]:
    """Detect/normalize up-axis; rotates obj.matrix_world only (never mesh data).

    "auto": measure the combined world bbox; if the longest axis is Y and Z's
    extent is small relative to Y (i.e. this looks like a Y-up mesh lying on
    its "back" in Blender's Z-up space), rotate Y -> Z.
    "y" / "z": honored explicitly regardless of bbox shape.
    Other explicit axis labels (x/-x/-y/-z) are also honored, for completeness.
    """
    bpy.context.view_layer.update()
    notes: list[str] = []

    mins = [math.inf] * 3
    maxs = [-math.inf] * 3
    for o in mesh_objects:
        mat = o.matrix_world
        for c in o.bound_box:
            wc = mat @ Vector(c)
            for k in range(3):
                mins[k] = min(mins[k], wc[k])
                maxs[k] = max(maxs[k], wc[k])
    dims = [maxs[k] - mins[k] for k in range(3)]

    label: str | None
    if up_axis_cfg == "auto":
        longest_axis = max(range(3), key=lambda k: dims[k])
        if longest_axis == 1 and dims[2] < 0.5 * dims[1]:
            label = "y"
            notes.append(
                f"input.up_axis=auto: detected Y-up "
                f"(dims x={dims[0]:.4f} y={dims[1]:.4f} z={dims[2]:.4f}); rotating Y->Z"
            )
        else:
            notes.append(
                f"input.up_axis=auto: dims x={dims[0]:.4f} y={dims[1]:.4f} "
                f"z={dims[2]:.4f}; already Z-up, no rotation applied"
            )
            return notes
    elif up_axis_cfg == "z":
        notes.append("input.up_axis=z: honored explicitly, no rotation applied")
        return notes
    else:
        label = up_axis_cfg
        notes.append(f"input.up_axis={label}: honored explicitly, rotating {label}->Z")

    up_vec = _AXIS_VECTORS[label]
    rot_mat: Matrix = up_vec.rotation_difference(Vector((0.0, 0.0, 1.0))).to_matrix().to_4x4()

    for o in mesh_objects:
        o.matrix_world = rot_mat @ o.matrix_world
    bpy.context.view_layer.update()
    return notes
