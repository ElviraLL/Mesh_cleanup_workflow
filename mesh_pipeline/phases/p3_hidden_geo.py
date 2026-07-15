"""P3 -- hidden geometry removal.

Two passes, per docs/blender-body-mesh-cleanup.md Phase 3 ("the core trick"):

  A. Whole-part visibility. Build ONE combined BVH from every mesh object's
     evaluated world-space triangles (excluding the hidden `_backup_pre_cleanup`
     collection), tagging which object owns each triangle. For every loose
     part other than the body, sample up to ~200 verts and cast rays from far
     outside along `hidden_geometry.part_rays` fibonacci-sphere directions; a
     part is visible only if the first BVH hit along some (vertex, direction)
     sightline belongs to that part. Parts that never come up first-hit are
     interior junk and get deleted -- EXCEPT eye_l/eye_r, which sit behind
     cornea/lid surfaces and always test as internal but must survive.

  B. Fused inner layer on the body. Per-face pass: for each body face, offset
     the face center by `normal * 1e-5` and try `hidden_geometry.face_rays`
     fibonacci directions (own outward normal first, for early exit) against
     a combined BVH rebuilt from the surviving objects; a face is visible if
     any ray escapes (no hit) before `far`. Progress is stored in a face int
     attribute "vis_flag" (bm.faces.layers.int -> mesh attribute on
     `bm.to_mesh()`), computed in chunks to keep memory sane on big meshes.
     The visible set is dilated by `hidden_geometry.dilate_rings` adjacency
     rings BEFORE deletion (protects concave detail like nostrils/ears), then
     interior faces are deleted, followed by wire edges/loose verts, then
     orphan islands smaller than 40 verts.

DESTRUCTIVE: the runner snapshots snapshots/pre_p3_hidden_geo.blend before
this module runs.
"""

from __future__ import annotations

import math

import bmesh
import bpy
from mathutils import Quaternion, Vector
from mathutils.bvhtree import BVHTree

from mesh_pipeline import geom
from mesh_pipeline.context import PhaseResult, PipelineContext

PHASE_NAME = "p3_hidden_geo"
DESTRUCTIVE = True

_MAX_PART_SAMPLE_VERTS = 200
_FACE_CHUNK = 20_000
_MIN_ISLAND_VERTS = 40
_BACKUP_COLLECTION = "_backup_pre_cleanup"
_NEVER_DELETE_ROLES = ("eye_l", "eye_r")

# geom.fibonacci_sphere's first direction is always exactly (0, 1, 0) (and in
# general its points sit on tidy lat/long-ish rings). Humanoid meshes are
# almost always bilaterally symmetric about x=0, so an axis-aligned ray
# through an x~0 vertex can travel exactly along a mesh seam/meridian plane
# -- BVHTree.ray_cast is numerically unstable on such exactly-grazing rays
# and can miss the true nearest hit (observed directly: a ray correctly
# hitting an outer shell when tested alone would instead report a farther
# object's surface as "closer" once other geometry shared the same BVH).
# Rotating the whole sampling direction set by a small fixed angle around a
# generic (non-axis-aligned) skew axis breaks that exact alignment for
# every symmetric mesh without materially changing what the discrete
# direction set samples.
_JITTER_AXIS = Vector((0.4172, 0.5911, 0.6883)).normalized()
_JITTER_ANGLE = math.radians(1.0)
_JITTER_QUAT = Quaternion(_JITTER_AXIS, _JITTER_ANGLE)


def _deskewed(directions: list[Vector]) -> list[Vector]:
    return [_JITTER_QUAT @ d for d in directions]


def run(ctx: PipelineContext, cfg: dict) -> PhaseResult:
    ctx.ensure_object_mode()
    bpy.context.view_layer.update()

    notes: list[str] = []
    failures: list[str] = []

    if "body" not in ctx.names:
        raise KeyError("p3_hidden_geo: ctx.names has no 'body' role registered")
    body_obj = ctx.obj("body")

    backup_names = _backup_object_names()

    hg_cfg = cfg["hidden_geometry"]
    part_rays = int(hg_cfg["part_rays"])
    face_rays = int(hg_cfg["face_rays"])
    dilate_rings = int(hg_cfg["dilate_rings"])

    # ---- Pass A: whole-part visibility ------------------------------------
    mesh_objs = [
        o for o in bpy.data.objects if o.type == "MESH" and o.name not in backup_names
    ]
    depsgraph = bpy.context.evaluated_depsgraph_get()
    tri_sources = [(o.name, _world_triangles(o, depsgraph)) for o in mesh_objs]
    bvh_a, tri_owner_a = _build_bvh(tri_sources)
    far_a = _far_distance(tri_sources)

    # Scene-wide face totals, captured before any deletion. deleted_face_ratio
    # (assertion band 0.30-0.80, docs: "expect 50-70% of an AI mesh to be
    # hidden junk") is a holistic measure spanning BOTH passes -- whole junk
    # parts deleted in A plus the fused-inner-layer faces deleted in B --
    # not just the body's own per-face pass, otherwise a mesh whose junk is
    # entirely separate loose parts (no fused inner shell) would score 0.
    total_faces_before = sum(len(o.data.polygons) for o in mesh_objs)

    keep_names = {ctx.names[role] for role in _NEVER_DELETE_ROLES if role in ctx.names}
    role_by_name: dict[str, str] = {}
    for role, name in ctx.names.items():
        role_by_name.setdefault(name, role)

    directions_a = _deskewed([Vector(d) for d in geom.fibonacci_sphere(part_rays)])

    to_delete_names: list[str] = []
    if bvh_a is not None:
        for o in mesh_objs:
            if o.name == body_obj.name:
                continue
            if o.name in keep_names:
                notes.append(f"part {o.name!r} kept (eye role, never deleted)")
                continue
            visible = _part_is_visible(o, bvh_a, tri_owner_a, directions_a, far_a)
            if not visible:
                to_delete_names.append(o.name)

    parts_deleted: list[str] = []
    parts_faces_deleted = 0
    for name in to_delete_names:
        parts_deleted.append(role_by_name.get(name, name))
        obj = bpy.data.objects.get(name)
        if obj is not None:
            parts_faces_deleted += len(obj.data.polygons)
            bpy.data.objects.remove(obj, do_unlink=True)

    # Deleted objects must not linger in the name registry for later phases.
    deleted_set = set(to_delete_names)
    for role, name in list(ctx.names.items()):
        if name in deleted_set:
            del ctx.names[role]

    bpy.context.view_layer.update()

    # ---- Pass B: per-face visibility on the body ---------------------------
    body_obj = ctx.obj("body")
    me = body_obj.data
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.faces.ensure_lookup_table()
    bm.verts.ensure_lookup_table()

    body_faces_before = len(bm.faces)

    surviving_objs = [
        o for o in bpy.data.objects if o.type == "MESH" and o.name not in backup_names
    ]
    depsgraph = bpy.context.evaluated_depsgraph_get()
    tri_sources_b = []
    for o in surviving_objs:
        if o.name == body_obj.name:
            continue
        tri_sources_b.append((o.name, _world_triangles(o, depsgraph)))

    # The body's own current (not-yet-written-back) geometry, straight from bm
    # -- this must be part of the occlusion BVH too (self-occlusion in
    # concave regions), and using bm avoids relying on a stale evaluated mesh.
    mw = body_obj.matrix_world.copy()
    body_tris = [tuple(mw @ l.vert.co for l in tri) for tri in bm.calc_loop_triangles()]
    tri_sources_b.append((body_obj.name, body_tris))

    bvh_b, _ = _build_bvh(tri_sources_b)
    far_b = _far_distance(tri_sources_b)
    directions_b = _deskewed([Vector(d) for d in geom.fibonacci_sphere(face_rays)])

    vis_layer = bm.faces.layers.int.new("vis_flag")
    all_faces = bm.faces[:]
    mw3 = mw.to_3x3()
    if bvh_b is not None:
        for start in range(0, len(all_faces), _FACE_CHUNK):
            chunk = all_faces[start : start + _FACE_CHUNK]
            for f in chunk:
                center_w = mw @ f.calc_center_median()
                normal_w = mw3 @ f.normal
                if normal_w.length > 1e-12:
                    normal_w.normalize()
                else:
                    normal_w = Vector((0.0, 0.0, 1.0))
                origin = center_w + normal_w * 1e-5
                escapes = _face_escapes(origin, normal_w, directions_b, bvh_b, far_b)
                f[vis_layer] = 1 if escapes else 0
    else:
        for f in all_faces:
            f[vis_layer] = 1  # nothing to occlude against; keep everything

    visible_idx = {f.index for f in bm.faces if f[vis_layer] == 1}
    visible_idx = _dilate(bm, visible_idx, dilate_rings)

    delete_faces = [f for f in bm.faces if f.index not in visible_idx]
    if delete_faces:
        bmesh.ops.delete(bm, geom=delete_faces, context="FACES_ONLY")

    _delete_wire_and_loose(bm)

    components = geom.bm_connected_components(bm)
    bm.verts.ensure_lookup_table()
    small_verts = []
    for comp in components:
        if len(comp) < _MIN_ISLAND_VERTS:
            small_verts.extend(bm.verts[i] for i in comp)
    if small_verts:
        bmesh.ops.delete(bm, geom=small_verts, context="VERTS")

    body_faces_after = len(bm.faces)
    bm.to_mesh(me)
    me.update()
    bm.free()

    body_faces_deleted = body_faces_before - body_faces_after
    faces_deleted = parts_faces_deleted + body_faces_deleted
    deleted_face_ratio = (faces_deleted / total_faces_before) if total_faces_before else 0.0

    notes.append(
        f"pass A: deleted {len(parts_deleted)} whole part(s), {parts_faces_deleted} faces; "
        f"pass B (body): {body_faces_before} -> {body_faces_after} faces "
        f"({body_faces_deleted} deleted)"
    )

    metrics = {
        "parts_deleted": parts_deleted,
        "faces_before": total_faces_before,
        "faces_deleted": faces_deleted,
        "deleted_face_ratio": deleted_face_ratio,
    }

    return PhaseResult(
        phase=PHASE_NAME, status="ok", metrics=metrics, failures=failures, notes=notes
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _backup_object_names() -> set[str]:
    coll = bpy.data.collections.get(_BACKUP_COLLECTION)
    if coll is None:
        return set()
    return {o.name for o in coll.objects}


def _world_triangles(obj, depsgraph) -> list[tuple]:
    """Evaluated world-space triangles (vertex-position 3-tuples) for obj."""
    ev = obj.evaluated_get(depsgraph)
    me = ev.to_mesh()
    if me is None:
        return []
    try:
        me.calc_loop_triangles()
        mw = obj.matrix_world
        verts = me.vertices
        tris = []
        for lt in me.loop_triangles:
            a, b, c = lt.vertices
            tris.append((mw @ verts[a].co, mw @ verts[b].co, mw @ verts[c].co))
        return tris
    finally:
        ev.to_mesh_clear()


def _build_bvh(tri_sources: list[tuple[str, list[tuple]]]):
    """Combined BVH over all triangle sources, plus a parallel owner list.

    Returns (bvh_or_None, owners) where owners[i] is the object name that
    triangle i (as given to BVHTree.FromPolygons) belongs to -- this is the
    "tagging which object each triangle belongs to" the docs call for,
    implemented as a flat lookup rather than sorted index ranges (equivalent
    O(1) owner lookup, simpler to build incrementally).
    """
    verts: list = []
    polys: list[tuple[int, int, int]] = []
    owners: list[str] = []
    for name, tris in tri_sources:
        for tri in tris:
            base = len(verts)
            verts.append(tri[0])
            verts.append(tri[1])
            verts.append(tri[2])
            polys.append((base, base + 1, base + 2))
            owners.append(name)
    if not polys:
        return None, owners
    bvh = BVHTree.FromPolygons(verts, polys, all_triangles=True, epsilon=1e-6)
    return bvh, owners


def _far_distance(tri_sources: list[tuple[str, list[tuple]]]) -> float:
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for _, tris in tri_sources:
        for tri in tris:
            for v in tri:
                xs.append(v.x)
                ys.append(v.y)
                zs.append(v.z)
    if not xs:
        return 1000.0
    dx, dy, dz = max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)
    diag = math.sqrt(dx * dx + dy * dy + dz * dz)
    return max(diag * 3.0, 1.0)


def _part_is_visible(obj, bvh, tri_owner, directions, far: float) -> bool:
    verts_world = [obj.matrix_world @ v.co for v in obj.data.vertices]
    if not verts_world:
        return False
    n = len(verts_world)
    if n > _MAX_PART_SAMPLE_VERTS:
        stride = max(1, n // _MAX_PART_SAMPLE_VERTS)
        sample = verts_world[::stride][:_MAX_PART_SAMPLE_VERTS]
    else:
        sample = verts_world
    for v in sample:
        for d in directions:
            origin = v + d * far
            ray_dir = v - origin
            if ray_dir.length < 1e-9:
                continue
            ray_dir.normalize()
            hit = bvh.ray_cast(origin, ray_dir, far * 1.5)
            if hit[0] is None:
                continue
            idx = hit[2]
            if tri_owner[idx] == obj.name:
                return True
    return False


def _face_escapes(origin, normal, directions, bvh, far: float) -> bool:
    """True if some ray from origin (own normal tried first) never hits bvh."""
    hit = bvh.ray_cast(origin, normal, far)
    if hit[0] is None:
        return True
    for d in directions:
        if d.dot(normal) <= 0.05:
            continue  # skip rays pointing back into the surface
        hit = bvh.ray_cast(origin, d, far)
        if hit[0] is None:
            return True
    return False


def _dilate(bm, visible_idx: set[int], rings: int) -> set[int]:
    bm.faces.ensure_lookup_table()
    visible = set(visible_idx)
    frontier = set(visible)
    for _ in range(max(0, rings)):
        nxt: set[int] = set()
        for fi in frontier:
            f = bm.faces[fi]
            for e in f.edges:
                for lf in e.link_faces:
                    if lf.index not in visible:
                        nxt.add(lf.index)
        if not nxt:
            break
        visible |= nxt
        frontier = nxt
    return visible


def _delete_wire_and_loose(bm) -> None:
    bm.edges.ensure_lookup_table()
    wire = [e for e in bm.edges if len(e.link_faces) == 0]
    if wire:
        bmesh.ops.delete(bm, geom=wire, context="EDGES")
    bm.verts.ensure_lookup_table()
    loose = [v for v in bm.verts if len(v.link_edges) == 0]
    if loose:
        bmesh.ops.delete(bm, geom=loose, context="VERTS")
