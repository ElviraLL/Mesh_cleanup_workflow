"""P7 -- UV atlas.

Re-unwrap fragmented UVs and pack everything into one shared 0-1 atlas space,
per docs/blender-uv-atlas-rebake.md Phase 4 and
docs/uv-atlas-references/snippets.md sections 9-10 ("6-axis segmentation +
region absorption + seams" / "seam-based unwrap + pack").

Runs entirely on the *cleaned* objects (never touches `_backup_pre_cleanup` --
those keep their original UVs for p8's bake source). Steps, mapped onto the
task spec:

1. Count UV islands on the body before touching anything
   (`geom.uv_island_count`, reused as-is -- this is the "union-find over
   faces welded in UV space" counter; there is no built-in property for it).
2. Decide which body faces need a fresh unwrap. Per material slot: if
   `uv.keep_clean_islands` is off, or the mesh has no UV layer, or the slot's
   own island count is already close to its own face count (confetti /
   per-triangle unwrap), those faces are "dirty" and get re-unwrapped. This
   generalizes the doc's body-level heuristic ("uv_islands ~= face count or
   no UV layer -> re-unwrap all body faces") to per-material granularity so a
   body with one clean material and one confetti material only touches the
   confetti one.
3. Dominant-normal-axis segmentation (snippet 9) restricted to the dirty
   faces: label each dirty face by argmax |normal . axis| over the 6 cardinal
   directions, then call the *shared* `geom.absorb_small_regions` pure helper
   (built here from a bmesh-derived adjacency list, per ARCHITECTURE.md's
   "reuse uv_island_count and absorb_small_regions" instruction) to merge
   speckle regions smaller than `uv.min_region` into their most-common
   neighboring region.
4. Clear all seams; mark a seam on every edge whose two faces end up with a
   different final label (a "clean" face effectively has label=None, so the
   dirty/clean boundary always gets a seam too -- this keeps the fresh
   unwrap from bleeding into material slots that were left alone).
5. Edit mode on the body alone: select the dirty faces via bmesh (never
   view-based selection) and run `bpy.ops.uv.unwrap(method='ANGLE_BASED')`.
6. Multi-object edit mode across every deliverable mesh object (body +
   whichever of eye_l/eye_r/teeth*/tongue exist in ctx.names -- eyes/teeth
   are separate objects, parented not joined, so this is the only way to
   pack their islands into the *same* shared atlas space) with all faces
   selected, then `bpy.ops.uv.pack_islands`. All deliverable objects' UV
   layers are renamed to "UVMap" first since multi-object UV ops require
   matching layer names.
7. Recount islands after, and compute `island_overlap_count` via pairwise
   UV-space bounding-box overlap across every island of every deliverable
   object (documented approximation -- true polygon overlap would need
   rasterization; bbox overlap is the accepted v1 test per the task spec).
"""

from __future__ import annotations

from collections import Counter

import bmesh
import bpy
from mathutils import Vector

from mesh_pipeline import geom
from mesh_pipeline.context import PhaseResult, PipelineContext

PHASE_NAME = "p7_uv_atlas"
DESTRUCTIVE = True

UV_LAYER_NAME = "UVMap"

# Role keys that count as "deliverable" mesh objects sharing the atlas space.
# Matches export.names in config/schema.py, plus "teeth" -- p2_weld_split's
# classifier currently only produces a single "teeth" role (p5/p6, which
# would split it into teeth_u/teeth_l/tongue, are not implemented yet).
_DELIVERABLE_ROLE_KEYS = [
    "body",
    "eye_l",
    "eye_r",
    "teeth_u",
    "teeth_l",
    "teeth",
    "tongue",
]

# A material-slot's own faces are considered "dirty" (confetti / no clean
# unwrap) once its island count reaches this fraction of its own face count.
_CONFETTI_RATIO = 0.9

_AXES: list[tuple[float, float, float]] = [
    (1.0, 0.0, 0.0),
    (-1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, -1.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.0, 0.0, -1.0),
]


def run(ctx: PipelineContext, cfg: dict) -> PhaseResult:
    ctx.ensure_object_mode()
    bpy.context.view_layer.update()

    uv_cfg = cfg["uv"]
    min_region = int(uv_cfg["min_region"])
    keep_clean = bool(uv_cfg["keep_clean_islands"])

    notes: list[str] = []

    body = ctx.obj("body")
    body_name = body.name

    # -- (1) islands before, on the body only --------------------------------
    uv_islands_before = geom.uv_island_count(body.data)
    notes.append(f"uv_islands_before (body, {body_name!r}) = {uv_islands_before}")

    # -- (2)+(3)+(4): decide dirty faces, segment, absorb, mark seams --------
    dirty_indices, region_count = _resegment_body(body, min_region, keep_clean, notes)

    # -- (5) unwrap the dirty faces on the body alone ------------------------
    if dirty_indices:
        _unwrap_dirty_faces(body_name, dirty_indices)
        notes.append(f"unwrapped {len(dirty_indices)} dirty face(s) into {region_count} region(s)")
    else:
        notes.append("no dirty faces; skipped uv.unwrap (all material slots already clean)")

    # -- (6) shared pack across every deliverable object ---------------------
    deliverable_names = _deliverable_object_names(ctx)
    _ensure_uv_layer_named(deliverable_names, UV_LAYER_NAME)
    _pack_islands_multi_object(deliverable_names, body_name)
    notes.append(
        f"packed islands for deliverable objects: {sorted(deliverable_names)} "
        f"(active={body_name!r})"
    )

    # -- (7) islands after + overlap check -----------------------------------
    body_after = ctx.obj("body")
    uv_islands_after = geom.uv_island_count(body_after.data)

    all_bboxes: list[tuple[float, float, float, float]] = []
    per_object_islands: dict[str, int] = {}
    for name in deliverable_names:
        obj = bpy.data.objects[name]
        bboxes = _uv_island_bboxes(obj.data)
        per_object_islands[name] = len(bboxes)
        all_bboxes.extend(bboxes)

    island_overlap_count = _count_bbox_overlaps(all_bboxes)

    notes.append(f"uv_islands_after (body, {body_name!r}) = {uv_islands_after}")
    notes.append(f"per-object island counts (post-pack): {per_object_islands}")
    notes.append(
        f"island_overlap_count = {island_overlap_count} "
        "(pairwise UV bbox overlap across all deliverable objects' islands -- "
        "an approximation, not exact polygon overlap; see module docstring)"
    )

    metrics = {
        "uv_islands_before": uv_islands_before,
        "uv_islands_after": uv_islands_after,
        "island_overlap_count": island_overlap_count,
    }

    return PhaseResult(phase=PHASE_NAME, status="ok", metrics=metrics, notes=notes)


# ---------------------------------------------------------------------------
# (2)+(3)+(4): dirty-face decision, segmentation, absorption, seams
# ---------------------------------------------------------------------------


def _resegment_body(
    body, min_region: int, keep_clean: bool, notes: list[str]
) -> tuple[list[int], int]:
    """Mutates body.data: clears seams, marks new ones.

    Returns (dirty_face_indices, region_count). No verts/faces are added or
    removed in this pass (only the seam flag changes), so the returned face
    indices stay valid against body.data.polygons after this call --
    _unwrap_dirty_faces re-derives its own bmesh from edit-mesh and indexes
    into it with these same integers.
    """
    me = body.data
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.faces.ensure_lookup_table()
    bm.edges.ensure_lookup_table()

    uvl = bm.loops.layers.uv.active
    dirty_indices = _dirty_face_indices(bm, uvl, keep_clean, notes)

    region_count = 0
    if dirty_indices:
        local_of = {fi: i for i, fi in enumerate(dirty_indices)}
        labels = [_dominant_axis_label(bm.faces[fi].normal) for fi in dirty_indices]
        adjacency: list[list[int]] = [[] for _ in dirty_indices]
        for i, fi in enumerate(dirty_indices):
            for e in bm.faces[fi].edges:
                for nf in e.link_faces:
                    if nf.index != fi and nf.index in local_of:
                        adjacency[i].append(local_of[nf.index])
        labels = geom.absorb_small_regions(labels, adjacency, min_region)
        region_count = len(set(labels))
        final_label = {fi: labels[i] for i, fi in enumerate(dirty_indices)}
    else:
        final_label = {}

    for e in bm.edges:
        e.seam = False
    for e in bm.edges:
        if len(e.link_faces) != 2:
            continue
        f1, f2 = e.link_faces
        lab1 = final_label.get(f1.index)
        lab2 = final_label.get(f2.index)
        if lab1 is None and lab2 is None:
            continue  # neither face touched -- leave whatever unwrap it already has alone
        if lab1 != lab2:
            e.seam = True

    bm.to_mesh(me)
    me.update()
    bm.free()
    return dirty_indices, region_count


def _dirty_face_indices(bm, uvl, keep_clean: bool, notes: list[str]) -> list[int]:
    """Faces that need a fresh unwrap, decided per material slot.

    If `keep_clean` is off, or there is no UV layer at all, every face is
    dirty. Otherwise each material slot's faces are tested independently:
    the slot is "confetti" (dirty) once its own island count reaches
    `_CONFETTI_RATIO` of its own face count.
    """
    n = len(bm.faces)
    if not keep_clean or uvl is None:
        notes.append(
            "dirty-face decision: keep_clean_islands off or no UV layer -> "
            "re-unwrapping all body faces"
        )
        return list(range(n))

    by_mat: dict[int, list[int]] = {}
    for f in bm.faces:
        by_mat.setdefault(f.material_index, []).append(f.index)

    dirty: list[int] = []
    for mat_idx, face_idxs in sorted(by_mat.items()):
        islands = _island_count_subset(bm, uvl, face_idxs)
        is_dirty = islands >= _CONFETTI_RATIO * len(face_idxs)
        notes.append(
            f"dirty-face decision: material_index={mat_idx} faces={len(face_idxs)} "
            f"islands={islands} -> {'DIRTY (re-unwrap)' if is_dirty else 'clean (kept)'}"
        )
        if is_dirty:
            dirty.extend(face_idxs)
    return dirty


def _island_count_subset(bm, uvl, face_idxs: list[int]) -> int:
    """Union-find island count restricted to a subset of faces (by index)."""
    face_set = set(face_idxs)
    local_of = {fi: i for i, fi in enumerate(face_idxs)}
    ds = geom.DisjointSet(len(face_idxs))
    edge_map: dict[int, list] = {}
    for fi in face_idxs:
        f = bm.faces[fi]
        for l in f.loops:
            edge_map.setdefault(l.edge.index, []).append((fi, l))
    for lst in edge_map.values():
        if len(lst) == 2:
            (f1, l1), (f2, l2) = lst
            if f1 not in face_set or f2 not in face_set:
                continue
            a1 = l1[uvl].uv
            b1 = l1.link_loop_next[uvl].uv
            a2 = l2.link_loop_next[uvl].uv
            b2 = l2[uvl].uv
            if (a1 - a2).length < 1e-6 and (b1 - b2).length < 1e-6:
                ds.union(local_of[f1], local_of[f2])
    return len(ds.groups())


def _dominant_axis_label(normal) -> int:
    return max(range(6), key=lambda i: normal.x * _AXES[i][0] + normal.y * _AXES[i][1] + normal.z * _AXES[i][2])


# ---------------------------------------------------------------------------
# (5): edit-mode unwrap of the dirty faces
# ---------------------------------------------------------------------------


def _unwrap_dirty_faces(body_name: str, dirty_indices: list[int]) -> None:
    """Edit-mode select the dirty faces (by index, via bmesh -- never via
    view3d/click selection) and run the seam-based angle-based unwrap.
    """
    dirty_set = set(dirty_indices)
    body = bpy.data.objects[body_name]
    me = body.data

    bpy.ops.object.select_all(action="DESELECT")
    bpy.context.view_layer.objects.active = body
    body.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bm = bmesh.from_edit_mesh(me)
    bm.faces.ensure_lookup_table()
    for f in bm.faces:
        f.select = f.index in dirty_set
    bmesh.update_edit_mesh(me)

    bpy.ops.uv.unwrap(method="ANGLE_BASED", margin=0.003, correct_aspect=True)
    bpy.ops.object.mode_set(mode="OBJECT")


# ---------------------------------------------------------------------------
# (6): shared multi-object pack
# ---------------------------------------------------------------------------


def _deliverable_object_names(ctx: PipelineContext) -> list[str]:
    """Every ctx.names role that represents a deliverable mesh object.

    Body is always included when registered; eyes/teeth/tongue are added
    only if that role exists yet (p5/p6 are feature-flagged and may not have
    run, or may not exist at all in the current pipeline build).
    """
    names: list[str] = []
    seen: set[str] = set()
    for role in _DELIVERABLE_ROLE_KEYS:
        name = ctx.names.get(role)
        if name and name in bpy.data.objects and name not in seen:
            obj = bpy.data.objects[name]
            if obj.type == "MESH":
                names.append(name)
                seen.add(name)
    return names


def _ensure_uv_layer_named(object_names: list[str], layer_name: str) -> None:
    for name in object_names:
        me = bpy.data.objects[name].data
        if len(me.uv_layers) == 0:
            me.uv_layers.new(name=layer_name)
        else:
            active = me.uv_layers.active or me.uv_layers[0]
            if active.name != layer_name:
                active.name = layer_name
            me.uv_layers.active = active


def _pack_islands_multi_object(object_names: list[str], active_name: str) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    for name in object_names:
        bpy.data.objects[name].select_set(True)
    bpy.context.view_layer.objects.active = bpy.data.objects[active_name]
    bpy.ops.object.mode_set(mode="EDIT")

    for name in object_names:
        obj = bpy.data.objects[name]
        bm = bmesh.from_edit_mesh(obj.data)
        uvl = bm.loops.layers.uv.active
        for f in bm.faces:
            f.select = True
        if uvl is not None:
            for f in bm.faces:
                for l in f.loops:
                    l[uvl].select = True
                    if hasattr(l[uvl], "select_edge"):
                        l[uvl].select_edge = True
        bmesh.update_edit_mesh(obj.data)

    try:
        bpy.ops.uv.pack_islands(rotate=True, margin=0.004)
    except TypeError:
        bpy.ops.uv.pack_islands(margin=0.004)  # older API without `rotate`

    bpy.ops.object.mode_set(mode="OBJECT")


# ---------------------------------------------------------------------------
# (7): post-pack island bboxes + overlap count
# ---------------------------------------------------------------------------


def _uv_island_bboxes(me) -> list[tuple[float, float, float, float]]:
    bm = bmesh.new()
    bm.from_mesh(me)
    try:
        uvl = bm.loops.layers.uv.active
        if uvl is None:
            return []
        bm.faces.ensure_lookup_table()
        ds = geom.DisjointSet(len(bm.faces))
        edge_map: dict[int, list] = {}
        for f in bm.faces:
            for l in f.loops:
                edge_map.setdefault(l.edge.index, []).append((f.index, l))
        for lst in edge_map.values():
            if len(lst) == 2:
                (f1, l1), (f2, l2) = lst
                a1 = l1[uvl].uv
                b1 = l1.link_loop_next[uvl].uv
                a2 = l2.link_loop_next[uvl].uv
                b2 = l2[uvl].uv
                if (a1 - a2).length < 1e-6 and (b1 - b2).length < 1e-6:
                    ds.union(f1, f2)

        groups = ds.groups()
        bboxes: list[tuple[float, float, float, float]] = []
        for members in groups.values():
            xs: list[float] = []
            ys: list[float] = []
            for fi in members:
                for l in bm.faces[fi].loops:
                    uv = l[uvl].uv
                    xs.append(uv.x)
                    ys.append(uv.y)
            if xs:
                bboxes.append((min(xs), min(ys), max(xs), max(ys)))
        return bboxes
    finally:
        bm.free()


def _bbox_overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1


def _count_bbox_overlaps(bboxes: list[tuple[float, float, float, float]]) -> int:
    n = len(bboxes)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if _bbox_overlap(bboxes[i], bboxes[j]):
                count += 1
    return count
