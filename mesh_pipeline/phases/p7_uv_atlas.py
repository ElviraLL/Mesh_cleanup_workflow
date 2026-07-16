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
6. `bpy.ops.uv.pack_islands` on the BODY ONLY. The deliverable hierarchy is
   parented, NOT joined (PLAN.md non-goal), and each non-body deliverable
   keeps its own material/texture -- e.g. p6's procedural iris writes its own
   equirect UVs. Packing across objects would move every island (the docs'
   own warning) and silently break those textures while buying nothing, since
   p8 bakes the atlas for the body alone. The body's UV layer is normalized
   to "UVMap" first.
7. Recount islands after, and compute `island_overlap_count` via a two-stage
   test across the body's islands: pairwise UV-space bounding-box overlap as
   a cheap candidate filter, then an exact point-in-triangle confirmation
   (fan-triangulated island geometry, up to 40 sample points from the
   smaller candidate island tested against the larger's triangles) so
   organic/concave islands that merely share a bounding box -- but not
   actual UV-space area -- are no longer counted as false positives. Also
   records `faces_total` (body face count at phase end) and
   `faces_reunwrapped` (how many dirty faces were actually re-unwrapped in
   step 5) so assertions can tell whether p7 owns the resulting island count
   or merely preserved the input's pre-existing layout.
"""

from __future__ import annotations

import bmesh
import bpy

from mesh_pipeline import geom
from mesh_pipeline.context import PhaseResult, PipelineContext

PHASE_NAME = "p7_uv_atlas"
DESTRUCTIVE = True

UV_LAYER_NAME = "UVMap"

# Material names p5/p6 stamp onto boolean-created cavity faces. Those faces
# inherit no real UVs from the cutters, so they must always be re-unwrapped
# (see _dirty_face_indices). Imported from the owning phase modules so a
# rename there cannot silently drift.
from mesh_pipeline.phases.p5_mouth import _MOUTH_INTERIOR_MAT
from mesh_pipeline.phases.p6_eyes import _SOCKET_INTERIOR_MAT

_ALWAYS_DIRTY_MATERIALS = {_MOUTH_INTERIOR_MAT, _SOCKET_INTERIOR_MAT}


# A material-slot's own faces are considered "dirty" (confetti / no clean
# unwrap) once its island count reaches this fraction of its own face count.
_CONFETTI_RATIO = 0.9

# UV islands smaller than this many faces inside an otherwise-clean slot are
# re-unwrapped anyway (broken sliver remnants; pack_islands stacks them
# inside larger islands -- see _dirty_face_indices).
_SLIVER_ISLAND_FACES = 4

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

    # -- (6) pack the BODY's islands only -------------------------------------
    # Design decision (see PLAN.md non-goals): the deliverable hierarchy is
    # parented, NOT joined, and each non-body deliverable keeps its own
    # material/texture (e.g. p6's procedural iris writes its own equirect
    # UVs). Packing across objects would move every island (docs: "packing
    # moves/scales *every* island") and silently break those textures, while
    # buying nothing -- only the body is baked into the atlas in p8. So the
    # shared 0-1 space is scoped to the body object alone.
    deliverable_names = [body_name]
    _ensure_uv_layer_named(deliverable_names, UV_LAYER_NAME)
    _pack_islands_multi_object(deliverable_names, body_name)
    notes.append(
        f"packed islands for body only (active={body_name!r}); non-body "
        "deliverables keep their own UVs/textures (parented-not-joined design)"
    )

    # -- (7) islands after + overlap check -----------------------------------
    body_after = ctx.obj("body")
    uv_islands_after = geom.uv_island_count(body_after.data)
    faces_total = len(body_after.data.polygons)
    faces_reunwrapped = len(dirty_indices)

    all_islands: list[dict] = []
    per_object_islands: dict[str, int] = {}
    for name in deliverable_names:
        obj = bpy.data.objects[name]
        islands = _uv_island_data(obj.data)
        per_object_islands[name] = len(islands)
        all_islands.extend(islands)

    island_overlap_count, candidate_pair_count = _count_island_overlaps(all_islands)

    notes.append(f"uv_islands_after (body, {body_name!r}) = {uv_islands_after}")
    notes.append(
        f"faces_total (body, {body_name!r}) = {faces_total}; "
        f"faces_reunwrapped = {faces_reunwrapped}"
    )
    notes.append(f"per-object island counts (post-pack): {per_object_islands}")
    notes.append(
        f"island_overlap_count = {island_overlap_count} "
        f"({candidate_pair_count} bbox-candidate pair(s) from stage 1, confirmed "
        "by exact point-in-triangle test in stage 2 -- see module docstring)"
    )

    metrics = {
        "uv_islands_before": uv_islands_before,
        "uv_islands_after": uv_islands_after,
        "island_overlap_count": island_overlap_count,
        "faces_total": faces_total,
        "faces_reunwrapped": faces_reunwrapped,
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
    mat_names = [
        (slot.material.name if slot.material else "") for slot in body.material_slots
    ]
    dirty_indices = _dirty_face_indices(bm, uvl, keep_clean, mat_names, notes)

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


def _dirty_face_indices(
    bm, uvl, keep_clean: bool, mat_names: list[str], notes: list[str]
) -> list[int]:
    """Faces that need a fresh unwrap, decided per material slot.

    If `keep_clean` is off, or there is no UV layer at all, every face is
    dirty. Otherwise each material slot's faces are tested independently:
    the slot is "confetti" (dirty) once its own island count reaches
    `_CONFETTI_RATIO` of its own face count.

    Boolean-created interior slots (mouth cavity, eye sockets) are ALWAYS
    dirty regardless of their island count: the p5/p6 cutters carry no UV
    layer, so the faces they stamp onto the body inherit degenerate UVs
    stacked near the origin -- which look like "one clean island" to the
    confetti test but overlap everything after packing (found by the e2e
    full run: island_overlap_count=7, all from cavity slots).
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
        mat_name = mat_names[mat_idx] if 0 <= mat_idx < len(mat_names) else ""
        if mat_name in _ALWAYS_DIRTY_MATERIALS:
            notes.append(
                f"dirty-face decision: material_index={mat_idx} ({mat_name!r}) "
                f"faces={len(face_idxs)} -> DIRTY (boolean-created interior, "
                "always re-unwrapped)"
            )
            dirty.extend(face_idxs)
            continue
        groups = _island_groups_subset(bm, uvl, face_idxs)
        islands = len(groups)
        is_dirty = islands >= _CONFETTI_RATIO * len(face_idxs)
        notes.append(
            f"dirty-face decision: material_index={mat_idx} faces={len(face_idxs)} "
            f"islands={islands} -> {'DIRTY (re-unwrap)' if is_dirty else 'clean (kept)'}"
        )
        if is_dirty:
            dirty.extend(face_idxs)
            continue
        # Within a kept-clean slot, sliver islands (< _SLIVER_ISLAND_FACES
        # faces) are re-unwrapped anyway: they're broken remnants of the soup
        # layout, not preserved artistry. pack_islands cannot place them
        # meaningfully and stacks them inside larger islands (avatar_003
        # calibration: 5 genuine post-pack overlaps, all 1-2-face slivers).
        sliver_faces = [fi for g in groups if len(g) < _SLIVER_ISLAND_FACES for fi in g]
        if sliver_faces:
            notes.append(
                f"dirty-face decision: material_index={mat_idx}: "
                f"{len(sliver_faces)} face(s) in sliver islands "
                f"(< {_SLIVER_ISLAND_FACES} faces) -> DIRTY despite clean slot"
            )
            dirty.extend(sliver_faces)
    return dirty


def _island_groups_subset(bm, uvl, face_idxs: list[int]) -> list[list[int]]:
    """Union-find UV islands restricted to a subset of faces (by index).

    Returns the islands as lists of face indices (len() of the result is the
    island count previously returned by _island_count_subset).
    """
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
    return [[face_idxs[i] for i in members] for members in ds.groups().values()]


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
# (6): pack (body only -- see run() step 6 for the design rationale)
# ---------------------------------------------------------------------------



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
        for f in bm.faces:
            f.select = True
        # Note: BMLoopUV in this Blender version exposes only `.uv`/`.pin_uv`
        # (no per-loop UV `.select`) -- mesh-domain face selection alone is
        # sufficient for uv.pack_islands to consider these faces (verified
        # empirically: it packs correctly across multiple edit-mode objects
        # from face.select alone).
        bmesh.update_edit_mesh(obj.data)

    try:
        bpy.ops.uv.pack_islands(rotate=True, margin=0.004)
    except TypeError:
        bpy.ops.uv.pack_islands(margin=0.004)  # older API without `rotate`

    bpy.ops.object.mode_set(mode="OBJECT")


# ---------------------------------------------------------------------------
# (7): post-pack island geometry + two-stage overlap test
# ---------------------------------------------------------------------------
#
# Stage 1 (cheap filter): pairwise UV-space bounding-box overlap, exactly as
# before -- this alone produced false positives on organic/concave islands
# that share a bbox without sharing any actual UV-space area (real-avatar
# finding: 758 bbox "overlaps" that were mostly non-overlapping concave
# shapes packed edge-to-edge).
#
# Stage 2 (exact confirmation): for every stage-1 candidate pair, sample up
# to 40 points from the smaller island (its faces' raw loop UV coordinates
# plus each face's UV centroid) and test each point against the larger
# island's fan-triangulated faces with a strict barycentric point-in-triangle
# test (epsilon 1e-7, so touching-but-not-overlapping edges don't count).
# Triangle lists and sample points are precomputed once per island (not per
# pair) so the O(candidates x samples x triangles) stage-2 cost stays cheap
# even at ~800 candidate pairs.


_TRI_EPS = 1e-7
_TRI_AREA_EPS = 1e-12
_MAX_SAMPLE_POINTS = 40


def _face_uv_triangles(
    loop_uvs: list[tuple[float, float]],
) -> list[tuple[tuple[float, float], tuple[float, float], tuple[float, float]]]:
    """Fan-triangulate a face's loop UVs from loop 0 (works for tri/quad/ngon).

    Degenerate (near-zero-area) triangles are dropped so they can never
    register a spurious "inside" hit.
    """
    triangles = []
    if len(loop_uvs) < 3:
        return triangles
    p0 = loop_uvs[0]
    for k in range(1, len(loop_uvs) - 1):
        tri = (p0, loop_uvs[k], loop_uvs[k + 1])
        (ax, ay), (bx, by), (cx, cy) = tri
        area2 = abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay))
        if area2 > _TRI_AREA_EPS:
            triangles.append(tri)
    return triangles


def _point_in_triangle_strict(
    p: tuple[float, float],
    tri: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
) -> bool:
    """Strict (open) barycentric point-in-triangle test, epsilon 1e-7."""
    (ax, ay), (bx, by), (cx, cy) = tri
    px, py = p
    denom = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
    if abs(denom) < _TRI_EPS:
        return False  # degenerate triangle -- never a hit
    a = ((by - cy) * (px - cx) + (cx - bx) * (py - cy)) / denom
    b = ((cy - ay) * (px - cx) + (ax - cx) * (py - cy)) / denom
    c = 1.0 - a - b
    return a > _TRI_EPS and b > _TRI_EPS and c > _TRI_EPS


def _uv_island_data(me) -> list[dict]:
    """Per-island UV data used by the two-stage overlap test: bbox, a
    precomputed fan-triangulation of every member face, and up to
    `_MAX_SAMPLE_POINTS` sample points (raw loop UVs + face centroids,
    evenly strided down if there are more than the cap).
    """
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
        islands: list[dict] = []
        for members in groups.values():
            xs: list[float] = []
            ys: list[float] = []
            triangles: list[
                tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
            ] = []
            points: list[tuple[float, float]] = []
            for fi in members:
                loop_uvs = [(l[uvl].uv.x, l[uvl].uv.y) for l in bm.faces[fi].loops]
                if not loop_uvs:
                    continue
                for x, y in loop_uvs:
                    xs.append(x)
                    ys.append(y)
                    points.append((x, y))
                points.append(
                    (
                        sum(x for x, _ in loop_uvs) / len(loop_uvs),
                        sum(y for _, y in loop_uvs) / len(loop_uvs),
                    )
                )
                triangles.extend(_face_uv_triangles(loop_uvs))
            if not xs:
                continue
            if len(points) > _MAX_SAMPLE_POINTS:
                stride = len(points) / float(_MAX_SAMPLE_POINTS)
                points = [points[int(i * stride)] for i in range(_MAX_SAMPLE_POINTS)]
            islands.append(
                {
                    "bbox": (min(xs), min(ys), max(xs), max(ys)),
                    "triangles": triangles,
                    "points": points,
                }
            )
        return islands
    finally:
        bm.free()


def _bbox_overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return ax0 < bx1 and bx0 < ax1 and ay0 < by1 and by0 < ay1


def _islands_overlap_exact(island_a: dict, island_b: dict) -> bool:
    """Point-in-triangle confirmation for one stage-1 bbox-candidate pair.

    Samples the smaller island's precomputed points against the larger
    island's precomputed triangles (both directions would be redundant --
    if any point of either island lies inside the other, the pair overlaps,
    so sampling the smaller side is sufficient and cheaper).
    """
    if len(island_a["points"]) <= len(island_b["points"]):
        small, large = island_a, island_b
    else:
        small, large = island_b, island_a
    if not small["points"] or not large["triangles"]:
        return False
    for p in small["points"]:
        for tri in large["triangles"]:
            if _point_in_triangle_strict(p, tri):
                return True
    return False


def _count_island_overlaps(islands: list[dict]) -> tuple[int, int]:
    """Return (confirmed_overlap_count, stage1_candidate_pair_count)."""
    n = len(islands)
    candidates = 0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            if not _bbox_overlap(islands[i]["bbox"], islands[j]["bbox"]):
                continue
            candidates += 1
            if _islands_overlap_exact(islands[i], islands[j]):
                count += 1
    return count, candidates
