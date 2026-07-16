"""P2 -- Weld and split.

docs/blender-body-mesh-cleanup.md Phase 2 + docs/blender-uv-atlas-rebake.md
Phase 2 "weld-distance sweep":

1. Sweep weld distance on THROWAWAY bmesh copies (never touch the real mesh
   during the sweep) across cfg["weld"]["sweep"], log-spaced probes; pick the
   last distance before the merge rate exceeds cfg["weld"]["auto_pick"]["max_merge_rate"].
2. Apply that distance for real via bmesh.ops.remove_doubles + dissolve_degenerate.
3. bpy.ops.mesh.separate(type='LOOSE') (the one bpy.ops exception -- there is
   no bmesh equivalent) to split the welded mesh into its connected components.
4. Classify the resulting objects by size + bbox heuristics (largest = body;
   symmetric small pair near top-front = eye_l/eye_r; wide flat stack near
   the mouth = teeth; anything else = part_N) and register them in ctx.names.
"""

from __future__ import annotations

import bpy
import bmesh
from mathutils import Vector

from mesh_pipeline import geom
from mesh_pipeline.context import PhaseResult, PipelineContext

PHASE_NAME = "p2_weld_split"
DESTRUCTIVE = True

_SWEEP_PROBES = 6
# An eye candidate must be considerably smaller than the body to avoid
# mistaking two similarly-sized torso/limb fragments for an eye pair.
_EYE_MAX_SIZE_RATIO_TO_BODY = 0.5
_EYE_SIZE_PAIR_TOLERANCE = 0.5  # min(a,b)/max(a,b) must be >= this to call a/b a "pair"
_EYE_MIRROR_TOLERANCE = 0.35  # |x_a + x_b| < tolerance * max(|x_a|, |x_b|)
_TEETH_WIDTH_TO_HEIGHT_RATIO = 1.5


def run(ctx: PipelineContext, cfg: dict) -> PhaseResult:
    ctx.ensure_object_mode()
    weld_cfg = cfg["weld"]
    sweep_lo, sweep_hi = weld_cfg["sweep"][0], weld_cfg["sweep"][-1]
    max_merge_rate = weld_cfg["auto_pick"]["max_merge_rate"]

    body_name = _resolve_body_name(ctx)
    body_obj = bpy.data.objects[body_name]
    me = body_obj.data

    # -- (a) weld-distance sweep on throwaway bmesh copies -----------------
    sweep_results = _weld_sweep(me, sweep_lo, sweep_hi, _SWEEP_PROBES)
    chosen_distance = _pick_weld_distance(sweep_results, max_merge_rate)

    # components_before: re-measure fresh, never trust a cached p0 number.
    bm_before = bmesh.new()
    bm_before.from_mesh(me)
    verts_before = len(bm_before.verts)
    components_before = len(geom.bm_connected_components(bm_before))
    bm_before.free()

    # -- (b) apply the chosen weld distance for real ------------------------
    verts_merged = _apply_weld(me, chosen_distance)

    # -- (c) separate LOOSE ---------------------------------------------------
    part_names = _separate_loose(body_name)

    # components_after: number of resulting loose parts (== weld post-count).
    components_after = len(part_names)

    # -- (d) classify parts ---------------------------------------------------
    roles, classification_notes = _classify_parts(part_names)
    ctx.names.update(roles)

    metrics = {
        "components_before": components_before,
        "components_after": components_after,
        "weld_distance": chosen_distance,
        "verts_merged": verts_merged,
        "parts": roles,
    }

    notes = [
        f"weld sweep: {sweep_results}",
        f"chosen weld_distance={chosen_distance} "
        f"(verts_before={verts_before}, verts_merged={verts_merged})",
        f"components_before={components_before} -> components_after={components_after}",
        *classification_notes,
    ]

    return PhaseResult(phase=PHASE_NAME, status="ok", metrics=metrics, notes=notes)


# ---------------------------------------------------------------------------
# (a) weld-distance sweep
# ---------------------------------------------------------------------------


def _resolve_body_name(ctx: PipelineContext) -> str:
    if "body" in ctx.names and ctx.names["body"] in bpy.data.objects:
        return ctx.names["body"]

    mesh_objs = [
        o for o in bpy.data.objects if o.type == "MESH" and not o.name.endswith("_backup")
    ]
    if not mesh_objs:
        raise RuntimeError("p2_weld_split: no (non-backup) mesh objects in scene")
    if len(mesh_objs) == 1:
        name = mesh_objs[0].name
    else:
        name = max(mesh_objs, key=lambda o: len(o.data.vertices)).name
    ctx.names["body"] = name
    return name


def _log_spaced(lo: float, hi: float, n: int) -> list[float]:
    if n <= 1 or lo <= 0 or hi <= lo:
        return [lo]
    ratio = hi / lo
    return [lo * (ratio ** (i / (n - 1))) for i in range(n)]


def _weld_sweep(me, lo: float, hi: float, n_probes: int) -> list[dict]:
    """Probe remove_doubles on throwaway bmesh copies; never touches `me`."""
    distances = _log_spaced(lo, hi, n_probes)
    results: list[dict] = []
    for dist in distances:
        bm = bmesh.new()
        bm.from_mesh(me)
        before = len(bm.verts)
        bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=dist)
        after = len(bm.verts)
        merged = before - after
        rate = (merged / before) if before else 0.0
        components = len(geom.bm_connected_components(bm))
        bm.free()
        results.append(
            {
                "dist": dist,
                "merged": merged,
                "rate": rate,
                "components": components,
            }
        )
    return results


def _pick_weld_distance(sweep_results: list[dict], max_merge_rate: float) -> float:
    """First (smallest) threshold before the merge rate exceeds max_merge_rate.

    Scans ascending distances, keeps advancing the chosen distance while its
    merge rate stays <= max_merge_rate, and stops at the first one that
    exceeds it. Falls back to the smallest probed distance if even that one
    already exceeds the budget.
    """
    if not sweep_results:
        raise ValueError("weld sweep produced no results")
    chosen = sweep_results[0]["dist"]
    for r in sweep_results:
        if r["rate"] <= max_merge_rate:
            chosen = r["dist"]
        else:
            break
    return chosen


# ---------------------------------------------------------------------------
# (b) apply weld for real
# ---------------------------------------------------------------------------


def _apply_weld(me, distance: float) -> int:
    bm = bmesh.new()
    bm.from_mesh(me)
    before = len(bm.verts)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=distance)
    bmesh.ops.dissolve_degenerate(bm, edges=bm.edges, dist=max(distance * 0.5, 1e-6))
    after = len(bm.verts)
    bm.to_mesh(me)
    me.update()
    bm.free()
    return before - after


# ---------------------------------------------------------------------------
# (c) separate LOOSE
# ---------------------------------------------------------------------------


def _separate_loose(body_name: str) -> list[str]:
    for o in bpy.data.objects:
        o.select_set(False)
    body_obj = bpy.data.objects[body_name]
    body_obj.select_set(True)
    bpy.context.view_layer.objects.active = body_obj

    before_names = {o.name for o in bpy.data.objects}
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.separate(type="LOOSE")
    bpy.ops.object.mode_set(mode="OBJECT")
    after_names = {o.name for o in bpy.data.objects}

    new_names = sorted(after_names - before_names)
    # The original object survives (holding one component); re-fetch by name,
    # never trust a cached reference across the operator calls above.
    return [body_name, *new_names]


# ---------------------------------------------------------------------------
# (d) classify parts
# ---------------------------------------------------------------------------


def _world_bbox(obj) -> dict:
    bpy.context.view_layer.update()
    mat = obj.matrix_world
    corners = [mat @ Vector(c) for c in obj.bound_box]
    xs = [c.x for c in corners]
    ys = [c.y for c in corners]
    zs = [c.z for c in corners]
    return {
        "center": (
            (min(xs) + max(xs)) / 2.0,
            (min(ys) + max(ys)) / 2.0,
            (min(zs) + max(zs)) / 2.0,
        ),
        "dims": (max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)),
    }


def _part_info(name: str) -> dict:
    obj = bpy.data.objects[name]
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    verts = len(bm.verts)
    faces = len(bm.faces)
    bm.free()
    return {
        "name": name,
        "verts": verts,
        "faces": faces,
        "bbox": _world_bbox(obj),
    }


def _classify_parts(part_names: list[str]) -> tuple[dict[str, str], list[str]]:
    parts_info = [_part_info(n) for n in part_names]
    parts_info.sort(key=lambda p: p["faces"], reverse=True)

    body_info = parts_info[0]
    rest = [p for p in parts_info[1:]]
    used: set[str] = set()
    notes: list[str] = [
        f"classify: body='{body_info['name']}' "
        f"(verts={body_info['verts']}, faces={body_info['faces']}, "
        f"bbox_center={body_info['bbox']['center']}, bbox_dims={body_info['bbox']['dims']})"
    ]

    roles: dict[str, str] = {"body": body_info["name"]}
    body_center = body_info["bbox"]["center"]
    body_verts = max(body_info["verts"], 1)

    # -- head z-band: eye candidates must sit on the HEAD, not merely above
    # body-bbox-center. A body-center-only rule misclassifies anything above
    # the torso midline as "eye-height" -- on avatar_003 this caught a
    # mirrored necklace bead pair at zfrac 0.55 as eye_l/eye_r. Compute the
    # band from the body's own world verts (subsampled) via the shared
    # geom.head_z_band heuristic (see its docstring for the avatar_003
    # calibration numbers); fall back to the old body-center rule, with an
    # honest note, when no sane band can be found.
    body_obj_for_band = bpy.data.objects[body_info["name"]]
    xz_points = geom.body_xz_points(body_obj_for_band)
    head_band = geom.head_z_band(xz_points)
    if head_band is not None:
        band_lo, band_hi = head_band
        notes.append(
            f"classify: head z-band (geom.head_z_band) = [{band_lo:.4f},{band_hi:.4f}]; "
            "eye-pair candidates must have both centers inside this band"
        )
    else:
        band_lo = band_hi = None
        notes.append(
            "classify: geom.head_z_band returned None (degenerate/no band found); "
            "falling back to body-bbox-center-only rule for eye-pair upper_ok check"
        )

    # -- eye pair: symmetric small pair near top-front of the body bbox -----
    eye_pair: tuple[dict, dict] | None = None
    for i in range(len(rest)):
        a = rest[i]
        if a["name"] in used:
            continue
        for j in range(i + 1, len(rest)):
            b = rest[j]
            if b["name"] in used:
                continue
            if a["verts"] > _EYE_MAX_SIZE_RATIO_TO_BODY * body_verts:
                continue
            size_ratio = min(a["verts"], b["verts"]) / max(a["verts"], b["verts"], 1)
            if size_ratio < _EYE_SIZE_PAIR_TOLERANCE:
                continue
            ax, _ay, az = a["bbox"]["center"]
            bx, _by, bz = b["bbox"]["center"]
            mirror_ok = abs(ax + bx) < _EYE_MIRROR_TOLERANCE * max(abs(ax), abs(bx), 1e-6)
            if band_lo is not None:
                upper_ok = band_lo <= az <= band_hi and band_lo <= bz <= band_hi
            else:
                upper_ok = az > body_center[2] and bz > body_center[2]
            if mirror_ok and upper_ok:
                eye_pair = (a, b)
                break
        if eye_pair:
            break

    if eye_pair:
        a, b = eye_pair
        if a["bbox"]["center"][0] >= b["bbox"]["center"][0]:
            eye_l, eye_r = a, b
        else:
            eye_l, eye_r = b, a
        roles["eye_l"] = eye_l["name"]
        roles["eye_r"] = eye_r["name"]
        used.update([a["name"], b["name"]])
        notes.append(
            f"classify: eye pair found -> eye_l='{eye_l['name']}' eye_r='{eye_r['name']}'"
        )
    else:
        band_desc = (
            f"head z-band [{band_lo:.4f},{band_hi:.4f}]" if band_lo is not None
            else "body-bbox-center fallback rule"
        )
        notes.append(
            "classify: no symmetric small pair found inside the "
            f"{band_desc} (no eyes classified)"
        )

    # -- teeth: wide, flat, low-in-bbox stack --------------------------------
    for p in rest:
        if p["name"] in used:
            continue
        dx, _dy, dz = p["bbox"]["dims"]
        cz = p["bbox"]["center"][2]
        wide_flat = dx > _TEETH_WIDTH_TO_HEIGHT_RATIO * max(dz, 1e-6)
        lower_ok = cz < body_center[2]
        if wide_flat and lower_ok:
            roles["teeth"] = p["name"]
            used.add(p["name"])
            notes.append(f"classify: teeth -> '{p['name']}'")
            break

    # -- everything else -> part_N -------------------------------------------
    idx = 0
    for p in rest:
        if p["name"] in used:
            continue
        roles[f"part_{idx}"] = p["name"]
        notes.append(
            f"classify: unclassified -> part_{idx}='{p['name']}' "
            f"(verts={p['verts']}, faces={p['faces']})"
        )
        idx += 1

    return roles, notes
