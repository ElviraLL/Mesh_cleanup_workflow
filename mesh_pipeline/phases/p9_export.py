"""P9 -- Naming, parenting, penetration testing, GLB export.

docs/blender-body-mesh-cleanup.md Phase 7 (penetration testing) + Phase 8
(wrap-up naming), PLAN.md export config (`hierarchy: parent_under_body`).

Steps:
  1. Rename deliverable objects per cfg["export"]["names"] (only roles present
     in ctx.names) and update ctx.names to the applied (possibly
     collision-suffixed) names.
  2. Parent every non-body deliverable under the body, keeping its current
     world transform (`matrix_parent_inverse = body.matrix_world.inverted()`).
  3. Penetration test: BVHTree.overlap() gives CANDIDATE triangle pairs only
     (docs: "BVH overlap alone is NOT sufficient" -- it flags bbox-node
     near-misses at millimeter clearances). Each candidate is confirmed with
     an exact triangle-triangle intersection test before being counted, and
     candidates whose intersection point falls inside an "opening window"
     (eye sockets, mouth) are excluded -- internal overlap behind lids/teeth
     is normal and desired there; only surface poke-through OUTSIDE openings
     is a defect.
  4. Export GLB with only the deliverables selected (backups from p1 live in
     the hidden "_backup_pre_cleanup" collection and are never selected, so
     `use_selection=True` excludes them without needing to touch collection
     exclusion state).
"""

from __future__ import annotations

import json
import math

import bmesh
import bpy
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.geometry import intersect_ray_tri

from mesh_pipeline.context import PhaseResult, PipelineContext

PHASE_NAME = "p9_export"
DESTRUCTIVE = False

# Margin multiplier applied around an eyeball's own world-space footprint to
# build its x/z "opening window" box -- generous on purpose (false negatives
# on the exclusion, i.e. wrongly counting a normal behind-the-lid overlap as a
# penetration, are the failure mode we're guarding against here; a slightly
# oversized window can only hide a *real* penetration if it pokes through
# right next to the socket, which the lid geometry itself makes unlikely).
_EYE_WINDOW_MARGIN = 2.0

# Fallback mouth window (used only when p5_mouth's opening_z_range isn't
# available in report.json, e.g. mouth phase not yet run / disabled): assumes
# a Z-up humanoid standing with feet at bbox z-min and head at bbox z-max, and
# brackets the lower portion of the head where a mouth typically sits.
_MOUTH_Z_FRACTION = (0.78, 0.90)
_MOUTH_X_HALF_WIDTH_FRACTION = 0.20  # central 40% of body width


def run(ctx: PipelineContext, cfg: dict) -> PhaseResult:
    ctx.ensure_object_mode()
    export_cfg = cfg["export"]

    rename_notes = _rename_deliverables(ctx, export_cfg["names"])
    parent_notes = _parent_under_body(ctx, export_cfg)
    penetration_count, candidate_count, windows, pen_notes = _count_penetrations(ctx, cfg)
    export_path, object_names = _export_glb(ctx, export_cfg)

    metrics = {
        "penetration_count": penetration_count,
        "export_path": str(export_path),
        "object_names": object_names,
        # extra, informative (not part of the required contract):
        "penetration_candidates": candidate_count,
        "opening_windows": [
            {"xmin": w[0], "xmax": w[1], "zmin": w[2], "zmax": w[3], "label": w[4]}
            for w in windows
        ],
    }
    notes = [*rename_notes, *parent_notes, *pen_notes]

    return PhaseResult(phase=PHASE_NAME, status="ok", metrics=metrics, notes=notes)


# ---------------------------------------------------------------------------
# (1) rename deliverables
# ---------------------------------------------------------------------------


def _rename_deliverables(ctx: PipelineContext, names_cfg: dict) -> list[str]:
    notes = []
    for role, new_name in names_cfg.items():
        if role not in ctx.names:
            continue
        obj = ctx.obj(role)
        old_name = obj.name
        obj.name = new_name
        ctx.names[role] = obj.name  # Blender may suffix on collision; record the real name
        if obj.name != old_name:
            notes.append(f"renamed {role}: '{old_name}' -> '{obj.name}'")
    return notes


# ---------------------------------------------------------------------------
# (2) parent non-body deliverables under body, keeping world transform
# ---------------------------------------------------------------------------


def _parent_under_body(ctx: PipelineContext, export_cfg: dict) -> list[str]:
    if export_cfg["hierarchy"] != "parent_under_body":
        return [f"hierarchy={export_cfg['hierarchy']!r} not recognized; skipping parenting"]
    if "body" not in ctx.names:
        return ["parent_under_body: no 'body' role registered; skipping parenting"]

    notes = []
    for role in export_cfg["names"]:
        if role == "body" or role not in ctx.names:
            continue
        bpy.context.view_layer.update()
        body = ctx.obj("body")
        child = ctx.obj(role)
        child.parent = body
        child.matrix_parent_inverse = body.matrix_world.inverted()
        notes.append(f"parented '{child.name}' under '{body.name}' (keep_transform)")
    bpy.context.view_layer.update()
    return notes


# ---------------------------------------------------------------------------
# (3) penetration testing: BVH candidates -> exact tri-tri confirmation
# ---------------------------------------------------------------------------


def _build_world_bvh(obj):
    """Triangulated, world-space BVH + parallel list of triangle vertex coords."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    me = eval_obj.to_mesh()
    try:
        bm = bmesh.new()
        bm.from_mesh(me)
        bmesh.ops.triangulate(bm, faces=bm.faces)
        bm.verts.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        mat = obj.matrix_world.copy()
        world_verts = [mat @ v.co for v in bm.verts]
        tris_vert_idx = [[v.index for v in f.verts] for f in bm.faces]
        tri_coords = [tuple(world_verts[i] for i in tri) for tri in tris_vert_idx]
        bvh = BVHTree.FromPolygons(world_verts, tris_vert_idx)
        bm.free()
        return bvh, tri_coords
    finally:
        eval_obj.to_mesh_clear()


def _edge_hits_tri(p0: Vector, p1: Vector, tri) -> bool:
    """Segment p0->p1 vs triangle `tri`, bounded to the segment length.

    `intersect_ray_tri` only reports forward (t >= 0) hits along `ray` from
    `orig`, so bounding `dist <= length` turns it into an exact bounded
    segment-vs-triangle test (docs Phase 7: "each edge ... via
    intersect_ray_tri ... segment-length bounded").
    """
    d = p1 - p0
    length = d.length
    if length < 1e-9:
        return False
    hit = intersect_ray_tri(tri[0], tri[1], tri[2], d / length, p0, True)
    if hit is None:
        return False
    return (hit - p0).length <= length + 1e-7


def _tri_tri_intersect(tri_a, tri_b) -> bool:
    """Exact triangle-triangle intersection.

    "Both directions" per docs Phase 7: test every edge of A against triangle
    B, AND every edge of B against triangle A -- a single direction misses
    configurations where B pokes through A along an edge of A itself (e.g. a
    corner poke) rather than piercing straight through a face.
    """
    a1, a2, a3 = tri_a
    b1, b2, b3 = tri_b
    for p0, p1 in ((a1, a2), (a2, a3), (a3, a1)):
        if _edge_hits_tri(p0, p1, tri_b):
            return True
    for p0, p1 in ((b1, b2), (b2, b3), (b3, b1)):
        if _edge_hits_tri(p0, p1, tri_a):
            return True
    return False


def _tri_centroid(tri) -> Vector:
    return (tri[0] + tri[1] + tri[2]) / 3.0


def _world_bbox(obj):
    bpy.context.view_layer.update()
    mat = obj.matrix_world
    corners = [mat @ Vector(c) for c in obj.bound_box]
    xs = [c.x for c in corners]
    ys = [c.y for c in corners]
    zs = [c.z for c in corners]
    return (min(xs), max(xs)), (min(ys), max(ys)), (min(zs), max(zs))


def _read_p5_opening_z_range(ctx: PipelineContext) -> tuple[float, float] | None:
    """Best-effort read of p5_mouth's opening_z_range from report.json.

    p9 only receives `ctx`/`cfg`, not earlier phases' metrics, so this reaches
    into report.json (written after every phase, per ARCHITECTURE.md) to find
    it -- if the report is missing, unreadable, or p5 was skipped/hasn't run,
    the caller falls back to the documented z-band heuristic.
    """
    report_path = ctx.job_dir / "report.json"
    if not report_path.exists():
        return None
    try:
        data = json.loads(report_path.read_text())
    except Exception:
        return None
    for phase in data.get("phases", []):
        if phase.get("phase") != "p5_mouth":
            continue
        metrics = phase.get("metrics") or {}
        if metrics.get("skipped"):
            return None
        rng = metrics.get("opening_z_range")
        if isinstance(rng, (list, tuple)) and len(rng) == 2:
            try:
                return float(rng[0]), float(rng[1])
            except (TypeError, ValueError):
                return None
    return None


def _opening_windows(ctx: PipelineContext, cfg: dict) -> list[tuple]:
    """x/z boxes (xmin, xmax, zmin, zmax, label) where overlap is expected.

    Eye windows are built directly from each present eyeball object's world
    bbox (centered on the eyeball, generously margined). The mouth window
    prefers p5_mouth's measured opening_z_range (see
    `_read_p5_opening_z_range`); lacking that, it falls back to a documented
    Z-up-humanoid heuristic z-band, x-bracketed to the central 40% of the
    body's width.
    """
    windows: list[tuple] = []

    for role in ("eye_l", "eye_r"):
        if role not in ctx.names:
            continue
        obj = ctx.obj(role)
        (xmin, xmax), (_ymin, _ymax), (zmin, zmax) = _world_bbox(obj)
        cx, cz = (xmin + xmax) / 2.0, (zmin + zmax) / 2.0
        rx = max((xmax - xmin) / 2.0, 1e-6) * _EYE_WINDOW_MARGIN
        rz = max((zmax - zmin) / 2.0, 1e-6) * _EYE_WINDOW_MARGIN
        windows.append((cx - rx, cx + rx, cz - rz, cz + rz, f"eye_opening:{role}"))

    if cfg.get("mouth", {}).get("enabled", True) and "body" in ctx.names:
        body = ctx.obj("body")
        (xmin, xmax), (_ymin, _ymax), (zmin, zmax) = _world_bbox(body)
        width = xmax - xmin
        height = zmax - zmin
        cx = (xmin + xmax) / 2.0

        z_range = _read_p5_opening_z_range(ctx)
        if z_range is None:
            z_range = (
                zmin + _MOUTH_Z_FRACTION[0] * height,
                zmin + _MOUTH_Z_FRACTION[1] * height,
            )
        z_lo, z_hi = sorted(z_range)
        x_half = _MOUTH_X_HALF_WIDTH_FRACTION * width
        windows.append((cx - x_half, cx + x_half, z_lo, z_hi, "mouth_opening"))

    return windows


def _in_window(loc: Vector, windows: list[tuple]) -> bool:
    for xmin, xmax, zmin, zmax, _label in windows:
        if xmin <= loc.x <= xmax and zmin <= loc.z <= zmax:
            return True
    return False


def _count_penetrations(ctx: PipelineContext, cfg: dict):
    notes: list[str] = []
    if "body" not in ctx.names:
        notes.append("penetration test: no 'body' role registered; penetration_count=0")
        return 0, 0, [], notes

    bpy.context.view_layer.update()
    body_bvh, body_tris = _build_world_bvh(ctx.obj("body"))
    windows = _opening_windows(ctx, cfg)

    export_names = cfg["export"]["names"]
    total = 0
    candidates = 0
    for role in export_names:
        if role == "body" or role not in ctx.names:
            continue
        other_obj = ctx.obj(role)
        other_bvh, other_tris = _build_world_bvh(other_obj)
        pairs = body_bvh.overlap(other_bvh)
        candidates += len(pairs)
        role_count = 0
        for i_body, i_other in pairs:
            tri_a = body_tris[i_body]
            tri_b = other_tris[i_other]
            if not _tri_tri_intersect(tri_a, tri_b):
                continue  # BVH near-miss, not a real intersection
            loc = _tri_centroid(tri_b)
            if _in_window(loc, windows):
                continue  # normal internal overlap behind an opening (lid/lip)
            role_count += 1
        if role_count:
            notes.append(f"penetration: {role_count} confirmed poke-through tri(s) for '{role}'")
        total += role_count

    notes.append(
        f"penetration test: {candidates} BVH candidate pair(s), "
        f"{total} confirmed penetration(s) outside {len(windows)} opening window(s)"
    )
    return total, candidates, windows, notes


# ---------------------------------------------------------------------------
# (4) export GLB (deliverables only, backups excluded via selection)
# ---------------------------------------------------------------------------


def _export_glb(ctx: PipelineContext, export_cfg: dict):
    if "body" not in ctx.names:
        raise RuntimeError("p9_export: no 'body' role registered; nothing to export")

    bpy.context.view_layer.update()
    for o in bpy.data.objects:
        o.select_set(False)

    object_names: list[str] = []
    for role in export_cfg["names"]:
        if role not in ctx.names:
            continue
        obj = ctx.obj(role)
        obj.select_set(True)
        object_names.append(obj.name)

    bpy.context.view_layer.objects.active = ctx.obj("body")

    out_dir = ctx.job_dir / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cleaned.glb"

    bpy.ops.export_scene.gltf(
        filepath=str(out_path),
        export_format="GLB",
        use_selection=True,
    )

    return out_path, object_names
