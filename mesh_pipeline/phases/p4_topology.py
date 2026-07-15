"""P4 -- topology cleanup: flaps, fins, holes, despike.

Order per docs/blender-uv-atlas-rebake.md Phase 2 ("topology cleanup order
matters") and docs/blender-body-mesh-cleanup.md Phase 4:

  1. gentle remove_doubles(2e-5) + dissolve_degenerate.
  2. delete wire edges (no linked faces) and loose verts.
  3. iteratively delete flap faces (faces with >=2 edges shared by 3+ faces)
     until converged -- deleting flaps exposes new ones.
  4. iteratively delete fin faces (>=1 over-shared edge AND >=2 free edges).
  5. hole fill: boundary-loop histogram (reusing geom.DisjointSet, the same
     algorithm as geom.boundary_loop_histogram); only loops with
     <= hole_fill.max_loop_edges edges are filled, honoring
     hole_fill.protected_zones (never auto-fill a loop near an eyeball --
     "prevents capping eye sockets"; large loops are intentional openings and
     are left alone regardless of zone config).
  6. triangulate fill n-gons, recalc_face_normals.
  7. targeted despike (interior edges with dihedral > 120 deg, light
     Laplacian lerp 0.3 on just those verts, 1-2 passes) ONLY if the spike
     count is > 0.1% of verts -- "never smooth the whole mesh".

Also computes boundary_by_zband: remaining boundary edges classified by
world-space Z into legs (bottom 40% of body bbox) / body (40-75%) /
head_hair (top 25%) -- "0 in the body/legs, the rest in hair/cloth is the
realistic success state" (docs). Getting to zero everywhere is not the goal.

DESTRUCTIVE: the runner snapshots snapshots/pre_p4_topology.blend before this
module runs.
"""

from __future__ import annotations

import math

import bmesh
import bpy
from mathutils import Vector

from mesh_pipeline import geom
from mesh_pipeline.context import PhaseResult, PipelineContext

PHASE_NAME = "p4_topology"
DESTRUCTIVE = True

_WELD_DIST = 2e-5
_MAX_FLAP_ITERS = 30
_MAX_FIN_ITERS = 30
_DESPIKE_DIHEDRAL_DEG = 120.0
_DESPIKE_LERP = 0.3
_DESPIKE_PASSES = 2
_DESPIKE_TRIGGER_RATIO = 0.001  # 0.1% of verts
_LEG_BAND_FRAC = 0.40
_BODY_BAND_FRAC = 0.75


def run(ctx: PipelineContext, cfg: dict) -> PhaseResult:
    ctx.ensure_object_mode()
    bpy.context.view_layer.update()

    if "body" not in ctx.names:
        raise KeyError("p4_topology: ctx.names has no 'body' role registered")
    body_obj = ctx.obj("body")
    me = body_obj.data

    bm = bmesh.new()
    bm.from_mesh(me)
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()

    notes: list[str] = []

    boundary_edges_before = sum(1 for e in bm.edges if e.is_boundary)

    # -- 1. gentle weld + dissolve_degenerate --------------------------------
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=_WELD_DIST)
    bmesh.ops.dissolve_degenerate(bm, edges=bm.edges, dist=_WELD_DIST)

    # -- 2. delete wire edges + loose verts -----------------------------------
    _delete_wire_and_loose(bm)

    # -- 3. iteratively delete flap faces --------------------------------------
    flaps_deleted = _iteratively_delete(bm, _flap_faces, _MAX_FLAP_ITERS, notes, "flap")
    _delete_wire_and_loose(bm)

    # -- 4. iteratively delete fin faces ----------------------------------------
    fins_deleted = _iteratively_delete(bm, _fin_faces, _MAX_FIN_ITERS, notes, "fin")
    _delete_wire_and_loose(bm)

    # -- 5. hole fill (boundary-loop histogram, protected zones) ----------------
    hole_cfg = cfg["hole_fill"]
    max_loop_edges = int(hole_cfg["max_loop_edges"])
    zones = hole_cfg.get("protected_zones", [])

    histogram = geom.boundary_loop_histogram(bm)
    notes.append(f"boundary-loop histogram before fill: {histogram}")

    eye_zones = _eyeball_protect_zones(ctx, zones)
    if any(z.get("type") == "lip_region" for z in zones):
        notes.append(
            "lip_region protected zone: mouth location is not yet identifiable "
            "before p5_mouth runs; zone not enforced this phase (fallback per docs)"
        )

    loops = _boundary_loops(bm)
    filled_faces: list = []
    holes_filled = 0
    skipped_protected = 0
    for loop_edges in loops:
        size = len(loop_edges)
        if size == 0 or size > max_loop_edges:
            continue
        median = _loop_median_world(loop_edges, body_obj.matrix_world)
        if _in_protected_zone(median, eye_zones):
            skipped_protected += 1
            continue
        result = bmesh.ops.holes_fill(bm, edges=loop_edges, sides=0)
        new_faces = result.get("faces", [])
        filled_faces.extend(new_faces)
        if new_faces:
            holes_filled += 1

    if skipped_protected:
        notes.append(f"hole fill: skipped {skipped_protected} loop(s) inside protected zones")

    # -- 6. triangulate fill n-gons, recalc_face_normals -------------------------
    ngon_fills = [f for f in filled_faces if f.is_valid and len(f.verts) > 3]
    if ngon_fills:
        bmesh.ops.triangulate(bm, faces=ngon_fills, quad_method="BEAUTY", ngon_method="BEAUTY")
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])

    # -- 7. targeted despike -------------------------------------------------------
    spike_verts = _spike_verts(bm, _DESPIKE_DIHEDRAL_DEG)
    total_verts = max(len(bm.verts), 1)
    spike_ratio = len(spike_verts) / total_verts
    if spike_ratio > _DESPIKE_TRIGGER_RATIO:
        _despike(bm, spike_verts, _DESPIKE_LERP, _DESPIKE_PASSES)
        notes.append(
            f"despike: {len(spike_verts)} verts ({spike_ratio:.4%}) on >120deg interior "
            f"edges, smoothed over {_DESPIKE_PASSES} pass(es)"
        )
    else:
        notes.append(
            f"despike: {len(spike_verts)} verts ({spike_ratio:.4%}) below the 0.1% trigger; "
            "skipped (never smooth the whole mesh for a handful of spikes)"
        )

    # -- boundary_by_zband (measured on the final topology) -----------------------
    mw = body_obj.matrix_world.copy()
    zs_all = [(mw @ v.co).z for v in bm.verts]
    zmin = min(zs_all) if zs_all else 0.0
    zmax = max(zs_all) if zs_all else 0.0
    boundary_by_zband = _boundary_by_zband(bm, mw, zmin, zmax)
    boundary_edges_after = sum(boundary_by_zband.values())

    bm.to_mesh(me)
    me.update()
    bm.free()

    metrics = {
        "boundary_edges_before": boundary_edges_before,
        "boundary_edges_after": boundary_edges_after,
        "boundary_by_zband": boundary_by_zband,
        "holes_filled": holes_filled,
        "flaps_deleted": flaps_deleted,
        "fins_deleted": fins_deleted,
    }

    notes.append(
        f"boundary_edges {boundary_edges_before} -> {boundary_edges_after}; "
        f"by zband: {boundary_by_zband}; flaps_deleted={flaps_deleted} "
        f"fins_deleted={fins_deleted} holes_filled={holes_filled}"
    )

    return PhaseResult(phase=PHASE_NAME, status="ok", metrics=metrics, notes=notes)


# ---------------------------------------------------------------------------
# steps 2-4: wire/loose cleanup, flap/fin deletion
# ---------------------------------------------------------------------------


def _delete_wire_and_loose(bm) -> None:
    bm.edges.ensure_lookup_table()
    wire = [e for e in bm.edges if len(e.link_faces) == 0]
    if wire:
        bmesh.ops.delete(bm, geom=wire, context="EDGES")
    bm.verts.ensure_lookup_table()
    loose = [v for v in bm.verts if len(v.link_edges) == 0]
    if loose:
        bmesh.ops.delete(bm, geom=loose, context="VERTS")


def _flap_faces(bm) -> list:
    """Faces with >=2 edges shared by 3+ faces (docs: never trim globally,
    only this specific over-shared-edge signature -- 2+ boundary edges alone
    would wrongly eat hair cards)."""
    out = []
    for f in bm.faces:
        over_shared = sum(1 for e in f.edges if len(e.link_faces) >= 3)
        if over_shared >= 2:
            out.append(f)
    return out


def _fin_faces(bm) -> list:
    """Faces with >=1 over-shared edge AND >=2 free (boundary/wire) edges."""
    out = []
    for f in bm.faces:
        over_shared = 0
        free = 0
        for e in f.edges:
            n = len(e.link_faces)
            if n >= 3:
                over_shared += 1
            if n <= 1:
                free += 1
        if over_shared >= 1 and free >= 2:
            out.append(f)
    return out


def _iteratively_delete(bm, finder, max_iters: int, notes: list[str], label: str) -> int:
    total = 0
    for i in range(max_iters):
        bm.faces.ensure_lookup_table()
        faces = finder(bm)
        if not faces:
            break
        bmesh.ops.delete(bm, geom=faces, context="FACES_ONLY")
        total += len(faces)
    else:
        notes.append(f"{label} deletion hit the {max_iters}-iteration guard; may not have converged")
    return total


# ---------------------------------------------------------------------------
# step 5: hole fill
# ---------------------------------------------------------------------------


def _boundary_loops(bm) -> list[list]:
    """Union-find over boundary edges -> list of loops (each a list of BMEdge).

    Same algorithm as geom.boundary_loop_histogram (reuses geom.DisjointSet
    directly) but returns the edge membership per loop instead of just a size
    histogram, since hole-filling needs the actual edges plus a per-loop
    median point for the protected-zone check.
    """
    bm.edges.ensure_lookup_table()
    boundary = [e for e in bm.edges if e.is_boundary]
    index_of = {e.index: i for i, e in enumerate(boundary)}
    ds = geom.DisjointSet(len(boundary))
    for e in boundary:
        i = index_of[e.index]
        for v in e.verts:
            for e2 in v.link_edges:
                if e2 is not e and e2.is_boundary and e2.index in index_of:
                    ds.union(i, index_of[e2.index])
    groups = ds.groups()
    return [[boundary[i] for i in members] for members in groups.values()]


def _loop_median_world(loop_edges: list, matrix_world) -> Vector:
    verts = set()
    for e in loop_edges:
        verts.add(e.verts[0])
        verts.add(e.verts[1])
    if not verts:
        return matrix_world.translation.copy()
    total = Vector((0.0, 0.0, 0.0))
    for v in verts:
        total += matrix_world @ v.co
    return total / len(verts)


def _eyeball_protect_zones(ctx: PipelineContext, zones: list[dict]) -> list[tuple[Vector, float]]:
    """[(center_world, radius_world)] for each configured eye_regions zone,
    one entry per eyeball role that is still present in ctx.names (p3 may
    have deleted a "wrongly-called-junk" eye but never eye_l/eye_r by
    contract, so both are normally present).
    """
    radius_factor = 1.5
    enabled = False
    for z in zones:
        if z.get("type") == "eye_regions":
            enabled = True
            radius_factor = float(z.get("radius_factor", 1.5))
            break
    if not enabled:
        return []

    out: list[tuple[Vector, float]] = []
    for role in ("eye_l", "eye_r"):
        if role not in ctx.names:
            continue
        obj = bpy.data.objects.get(ctx.names[role])
        if obj is None or not obj.data.vertices:
            continue
        bpy.context.view_layer.update()
        mw = obj.matrix_world
        verts_w = [mw @ v.co for v in obj.data.vertices]
        center = sum(verts_w, Vector((0.0, 0.0, 0.0))) / len(verts_w)
        radius = max((v - center).length for v in verts_w)
        out.append((center, radius * radius_factor))
    return out


def _in_protected_zone(point: Vector, zones: list[tuple[Vector, float]]) -> bool:
    for center, radius in zones:
        if (point - center).length <= radius:
            return True
    return False


# ---------------------------------------------------------------------------
# step 7: despike
# ---------------------------------------------------------------------------


def _spike_verts(bm, dihedral_deg: float) -> set:
    threshold = math.radians(dihedral_deg)
    spikes = set()
    bm.edges.ensure_lookup_table()
    for e in bm.edges:
        if len(e.link_faces) != 2:
            continue
        angle = e.calc_face_angle(None)
        if angle is not None and angle > threshold:
            spikes.add(e.verts[0])
            spikes.add(e.verts[1])
    return spikes


def _despike(bm, spike_verts: set, lerp: float, passes: int) -> None:
    for _ in range(passes):
        new_positions = {}
        for v in spike_verts:
            if not v.is_valid:
                continue
            neighbors = [e.other_vert(v) for e in v.link_edges]
            if not neighbors:
                continue
            avg = Vector((0.0, 0.0, 0.0))
            for n in neighbors:
                avg += n.co
            avg /= len(neighbors)
            new_positions[v] = v.co.lerp(avg, lerp)
        for v, co in new_positions.items():
            v.co = co


# ---------------------------------------------------------------------------
# boundary_by_zband
# ---------------------------------------------------------------------------


def _zband(z: float, zmin: float, zmax: float) -> str:
    height = zmax - zmin
    if height <= 1e-12:
        return "body"
    frac = (z - zmin) / height
    if frac <= _LEG_BAND_FRAC:
        return "legs"
    if frac <= _BODY_BAND_FRAC:
        return "body"
    return "head_hair"


def _boundary_by_zband(bm, matrix_world, zmin: float, zmax: float) -> dict[str, int]:
    counts = {"legs": 0, "body": 0, "head_hair": 0}
    bm.edges.ensure_lookup_table()
    for e in bm.edges:
        if not e.is_boundary:
            continue
        mid_z = ((matrix_world @ e.verts[0].co).z + (matrix_world @ e.verts[1].co).z) / 2.0
        counts[_zband(mid_z, zmin, zmax)] += 1
    return counts
