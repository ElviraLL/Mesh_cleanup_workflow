"""Phase 5 -- closed-lips mouth bag.

One boolean does both jobs: a DIFFERENCE with a "closed-lips bag" cutter
pushed through the lip line splits the lips AND creates the cavity walls
(cutter faces inherit the cutter's material, so we assign a dark
"Mouth_Interior" material to the cutter before the boolean). The cutter is a
scaled uvsphere whose FRONT half (toward the lip surface) is tapered, via a
per-vertex z-ramp, down to a thin horizontal slit (cfg mouth.lip_gap_mm) --
this is what makes the carved lips read as CLOSED (touching) at the skin
surface while still being topologically split into independent upper/lower
lip rims a rig can pull apart later. The BACK half of the cutter (inside the
head) keeps its full interior "bag" height (cfg mouth.bag_height), which is
where teeth + tongue actually live.

Two placement modes:

  - "carved_closed" (no usable teeth reference on the input mesh): teeth are
    built as two half-torus arches (bmesh.ops has no create_torus in this bpy
    build) and a tongue, placed relative to the *measured* bag interior (not
    the thin slit, and not guessed).
  - "reference_kept" (the input mesh already has a real teeth part -- common
    on Trellis 2 / AI avatar output, see p2_weld_split's 'teeth' role): the
    existing teeth object is kept as-is and used as the placement reference
    -- the bag cutter is sized/positioned to fully enclose it (+20% margin)
    instead of building duplicate arches. Only a tongue is still built (there
    is no tongue classifier).

See docs/blender-body-mesh-cleanup.md Phase 5 and PLAN.md `mouth:` config.
Uses the bmesh/data API exclusively for primitive creation (never
bpy.ops.mesh.primitive_*_add) per the ARCHITECTURE.md/docs pitfall list, and
never caches a bpy.types.Object reference across an operator call that can
invalidate it (modifier_apply) -- every use re-fetches via ctx.obj("body").

Unit-convention note: cfg mm-based values (teeth_recess_mm, lip_gap_mm) are
converted to Blender units assuming 1 Blender unit == 1 meter (the common
glTF/Blender default for a human-scale character). This is a documented
assumption, not a measured fact -- it is recorded in PhaseResult.notes on
every run so it is visible in report.json.
"""

from __future__ import annotations

import math

from mesh_pipeline import geom
from mesh_pipeline.context import PhaseResult

PHASE_NAME = "p5_mouth"
DESTRUCTIVE = True

_MM = 0.001  # 1 Blender unit == 1 meter (documented assumption, see module docstring)
_MOUTH_INTERIOR_MAT = "Mouth_Interior"
_BAG_DEPTH_MARGIN_MM = 2.0  # bag_z_range only counts Mouth_Interior faces this far behind the lip surface
_REFERENCE_MARGIN = 1.2  # +20% margin when sizing the bag cutter around an existing teeth object


def _empty_metrics() -> dict:
    return {
        "skipped": None,
        "mode": None,
        "teeth_recess_mm": None,
        "opening_z_range": None,
        "bag_z_range": None,
        "teeth_z_range": None,
        "lip_gap_mm_measured": None,
        "teeth_reference_kept": None,
        "lips_topologically_split": None,
    }


def run(ctx, cfg: dict) -> PhaseResult:
    mouth_cfg = cfg["mouth"]
    if not mouth_cfg["enabled"]:
        metrics = _empty_metrics()
        metrics["skipped"] = True
        return PhaseResult(
            phase=PHASE_NAME, status="ok", metrics=metrics, notes=["skipped: disabled"]
        )

    import bpy

    ctx.ensure_object_mode()

    notes = [
        "unit assumption: 1 Blender unit == 1 meter, used to convert "
        "mouth.teeth_recess_mm/lip_gap_mm into scene units"
    ]

    if "body" not in ctx.names:
        metrics = _empty_metrics()
        metrics["skipped"] = False
        return PhaseResult(
            phase=PHASE_NAME,
            status="needs_review",
            metrics=metrics,
            notes=notes,
            failures=["p5_mouth: no 'body' role registered in ctx.names; cannot locate mesh to carve"],
        )

    body = ctx.obj("body")
    bpy.context.view_layer.update()
    bbox = _world_bbox(body)
    body_height = bbox["max"].z - bbox["min"].z
    if body_height <= 0:
        metrics = _empty_metrics()
        metrics["skipped"] = False
        return PhaseResult(
            phase=PHASE_NAME,
            status="needs_review",
            metrics=metrics,
            notes=notes,
            failures=[f"p5_mouth: body world bbox has non-positive height ({body_height})"],
        )

    lip = geom.find_lip_line(body, bbox, notes)
    if lip is None:
        metrics = _empty_metrics()
        metrics["skipped"] = False
        return PhaseResult(
            phase=PHASE_NAME,
            status="needs_review",
            metrics=metrics,
            notes=notes,
            failures=[
                "p5_mouth: could not detect a lip line (no boundary/sharp-crease edge "
                "cluster found forming a wide-x/thin-z band in the candidate front "
                "mouth z-region); mouth carving skipped -- needs manual lip marking or "
                "a config override"
            ],
        )
    fissure_z, mouth_x, front_sign, lip_surface_y = lip

    # --- 0. reference-teeth detection: is there already a real teeth part? ---
    mode = "carved_closed"
    teeth_ref_bbox = None
    if "teeth" in ctx.names:
        teeth_obj = ctx.obj("teeth")
        t_bbox = _world_bbox(teeth_obj)
        t_cx = (t_bbox["min"].x + t_bbox["max"].x) / 2.0
        t_cz = (t_bbox["min"].z + t_bbox["max"].z) / 2.0
        if (
            abs(t_cz - fissure_z) <= 0.06 * body_height
            and abs(t_cx - mouth_x) <= 0.06 * body_height
        ):
            mode = "reference_kept"
            teeth_ref_bbox = t_bbox
            notes.append(
                f"reference teeth found: '{teeth_obj.name}' bbox_center=(x={t_cx:.4f}, "
                f"z={t_cz:.4f}) is within the mouth region -> mode=reference_kept "
                "(existing teeth kept, no duplicate arches built)"
            )
        else:
            notes.append(
                f"'teeth' role present ('{teeth_obj.name}') but its bbox center "
                f"(x={t_cx:.4f}, z={t_cz:.4f}) is outside the mouth region "
                f"(mouth_x={mouth_x:.4f}, fissure_z={fissure_z:.4f}, tolerance "
                f"0.06*body_height={0.06 * body_height:.4f}); treating it as a "
                "misclassified part -- ignored for reference-mode purposes"
            )

    # --- 1. cutter: closed-lips bag, dark Mouth_Interior material assigned FIRST ---
    size_override = None
    if mode == "reference_kept":
        size_override = _reference_bag_size(
            mouth_cfg, body_height, mouth_x, fissure_z, front_sign, lip_surface_y, teeth_ref_bbox
        )
        notes.append(
            f"reference bag sizing: rx={size_override['rx']:.4f} "
            f"ry={size_override['ry']:.4f} rz={size_override['rz']:.4f} "
            f"(encloses teeth bbox + {int((_REFERENCE_MARGIN - 1) * 100)}% margin)"
        )

    cutter_name = _make_mouth_cutter(
        mouth_cfg, body_height, mouth_x, fissure_z, front_sign, lip_surface_y,
        size_override=size_override, notes=notes,
    )

    # --- 2. boolean DIFFERENCE (also splits the lips + tints cavity walls) ---
    body = ctx.obj("body")
    _boolean_difference(body, cutter_name)
    body = ctx.obj("body")
    bpy.context.view_layer.update()

    mat_idx = _material_index(body.data, _MOUTH_INTERIOR_MAT)
    if mat_idx is None:
        metrics = _empty_metrics()
        metrics["skipped"] = False
        notes.append(
            "boolean cut applied but the Mouth_Interior material slot is not present "
            "on the body afterward; skipping teeth/tongue placement"
        )
        return PhaseResult(
            phase=PHASE_NAME,
            status="needs_review",
            metrics=metrics,
            notes=notes,
            failures=["p5_mouth: Mouth_Interior material missing after the boolean cut"],
        )

    # Safety-net squeeze: on smooth synthetic geometry the tapered cutter
    # alone produces a clean, uniformly thin slit (verified in tests/
    # make_fixture.py's e2e fixture). Real AI-generated meshes are far less
    # regular near the mouth -- verified directly on avatar_003: even with
    # the body pre-subdivided to sub-mm resolution in the mouth region, the
    # boolean still produced SEVERAL disconnected rim loops (not one
    # continuous slit) spanning tens of mm in Z, because the cutter's
    # razor-thin front cap does not reliably poke through an irregular real
    # surface everywhere across the mouth width. Rather than chase a
    # perfectly clean single-pass cut against arbitrary input meshes, pull
    # every rim vertex's Z back to within target_half_gap of fissure_z after
    # the fact -- this decouples "the cutter reliably carves an opening"
    # (its job) from "the opening reads as closed" (guaranteed here,
    # regardless of how irregular the raw cut came out).
    lip_gap_units = mouth_cfg.get("lip_gap_mm", 0.4) * _MM
    target_half_gap = lip_gap_units / 2.0
    squeezed = _squeeze_slit_to_target(body, mat_idx, fissure_z, target_half_gap)
    if squeezed:
        notes.append(
            f"slit squeeze: pulled {squeezed} rim vertex/vertices back to within "
            f"{target_half_gap / _MM:.3f}mm of fissure_z (safety net for irregular "
            "real-mesh cuts, see note above)"
        )
        body = ctx.obj("body")
        bpy.context.view_layer.update()

    opening = _measure_mouth_opening(body, mat_idx)
    if opening is None:
        metrics = _empty_metrics()
        metrics["skipped"] = False
        notes.append(
            "boolean cut applied but no Mouth_Interior-bordered rim edges were found "
            "afterward (cutter may not have intersected the skin surface); skipping "
            "teeth/tongue placement"
        )
        return PhaseResult(
            phase=PHASE_NAME,
            status="needs_review",
            metrics=metrics,
            notes=notes,
            failures=["p5_mouth: could not measure the opening rim after the boolean cut"],
        )

    z_lo, z_hi = opening["z_range"]
    x_lo, x_hi = opening["x_range"]
    opening_height = max(z_hi - z_lo, 1e-6)
    opening_width = max(x_hi - x_lo, 1e-6)
    rim_front_y = max(opening["y_values"]) if front_sign > 0 else min(opening["y_values"])
    lip_gap_mm_measured = (z_hi - z_lo) / _MM
    notes.append(
        f"opening (slit) rim measured: x=[{x_lo:.4f},{x_hi:.4f}] z=[{z_lo:.4f},{z_hi:.4f}] "
        f"rim_front_y={rim_front_y:.4f} lip_gap_mm_measured={lip_gap_mm_measured:.4f}"
    )

    bag_range = _measure_bag_extent(body, mat_idx, front_sign, lip_surface_y)
    if bag_range is not None:
        notes.append(f"bag interior z-range measured: [{bag_range[0]:.4f},{bag_range[1]:.4f}]")
    else:
        notes.append(
            "bag interior z-range could not be measured (no Mouth_Interior faces "
            f"found deeper than {_BAG_DEPTH_MARGIN_MM}mm behind the lip surface); "
            "falling back to the slit range for teeth/tongue centering"
        )

    split_ok = _lips_topologically_split(body, mat_idx, fissure_z)
    notes.append(f"lips_topologically_split={split_ok}")

    # --- 3. teeth + tongue ---
    if mode == "reference_kept":
        actual_recess_mm = None
        t_bbox = teeth_ref_bbox
        teeth_z_range = [t_bbox["min"].z, t_bbox["max"].z]
        _build_tongue_reference(ctx, mouth_x, front_sign, t_bbox)
        notes.append(
            f"reference mode: kept existing teeth object, teeth_z_range={teeth_z_range}"
        )
    else:
        recess_mm_mid = sum(mouth_cfg["teeth_recess_mm"]) / 2.0
        recess_units = recess_mm_mid * _MM
        target_front_y = rim_front_y - front_sign * recess_units

        if bag_range is not None:
            z_center = (bag_range[0] + bag_range[1]) / 2.0
            bag_height_extent = max(bag_range[1] - bag_range[0], 1e-6)
        else:
            z_center = (z_lo + z_hi) / 2.0
            bag_height_extent = opening_height

        major_radius = 0.5 * opening_width * 0.8
        minor_radius = min(major_radius * 0.35, bag_height_extent * 0.4)
        minor_radius = max(minor_radius, 1e-4)
        z_upper = z_center + 0.15 * bag_height_extent
        z_lower = z_center - 0.15 * bag_height_extent

        teeth_mat = _get_or_create_material("Teeth", (0.85, 0.83, 0.78, 1.0))
        upper = _build_arch_object(
            "Teeth_Upper", major_radius, minor_radius, front_sign, squash_x=0.78
        )
        lower = _build_arch_object(
            "Teeth_Lower", major_radius, minor_radius, front_sign, squash_x=0.78
        )
        for arch_obj in (upper, lower):
            arch_obj.data.materials.append(teeth_mat)

        upper_recess = _place_arch(
            upper, mouth_x, z_upper, target_front_y, rim_front_y, front_sign, major_radius, minor_radius
        )
        lower_recess = _place_arch(
            lower, mouth_x, z_lower, target_front_y, rim_front_y, front_sign, major_radius, minor_radius
        )

        ctx.names["teeth_u"] = upper.name
        ctx.names["teeth_l"] = lower.name

        tongue_mat = _get_or_create_material("Tongue", (0.55, 0.14, 0.18, 1.0))
        tongue_y = target_front_y - front_sign * (major_radius * 0.5)
        tongue_z = z_center - 0.3 * bag_height_extent
        tongue = _build_tongue_object(mouth_x, tongue_y, tongue_z, major_radius, minor_radius)
        tongue.data.materials.append(tongue_mat)
        ctx.names["tongue"] = tongue.name

        bpy.context.view_layer.update()
        teeth_z_vals = []
        for o in (upper, lower):
            wb = _world_bbox(o)
            teeth_z_vals.extend([wb["min"].z, wb["max"].z])
        teeth_z_range = [min(teeth_z_vals), max(teeth_z_vals)]

        actual_recess_mm = ((upper_recess + lower_recess) / 2.0) / _MM
        notes.append(
            f"teeth placed: major_radius={major_radius:.4f} minor_radius={minor_radius:.4f} "
            f"upper_recess_mm={upper_recess / _MM:.3f} lower_recess_mm={lower_recess / _MM:.3f} "
            f"z_center={z_center:.4f} (bag-interior centered)"
        )

    bag_check_range = bag_range if bag_range is not None else [z_lo, z_hi]
    if not (teeth_z_range[0] <= bag_check_range[1] and teeth_z_range[1] >= bag_check_range[0]):
        notes.append(
            "WARNING: teeth_z_range does not overlap bag_z_range -- assertion "
            "table (P5) is expected to flag this"
        )

    metrics = {
        "skipped": False,
        "mode": mode,
        "teeth_recess_mm": actual_recess_mm,
        "opening_z_range": [z_lo, z_hi],
        "bag_z_range": bag_range,
        "teeth_z_range": teeth_z_range,
        "lip_gap_mm_measured": lip_gap_mm_measured,
        "teeth_reference_kept": mode == "reference_kept",
        "lips_topologically_split": split_ok,
    }
    return PhaseResult(phase=PHASE_NAME, status="ok", metrics=metrics, notes=notes)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _world_bbox(obj) -> dict:
    import bpy
    from mathutils import Vector

    bpy.context.view_layer.update()
    mat = obj.matrix_world
    corners = [mat @ Vector(c) for c in obj.bound_box]
    xs = [c.x for c in corners]
    ys = [c.y for c in corners]
    zs = [c.z for c in corners]
    return {"min": Vector((min(xs), min(ys), min(zs))), "max": Vector((max(xs), max(ys), max(zs)))}


def _get_or_create_material(name: str, rgba: tuple):
    import bpy

    mat = bpy.data.materials.get(name)
    if mat is not None:
        return mat
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = rgba
    mat.diffuse_color = rgba
    return mat


def _boolean_difference(target_obj, cutter_obj_name: str) -> None:
    import bpy

    cutter = bpy.data.objects[cutter_obj_name]
    mod = target_obj.modifiers.new("p5_bool_diff", "BOOLEAN")
    mod.object = cutter
    mod.operation = "DIFFERENCE"
    mod.solver = "EXACT"
    # TRANSFER (not the INDEX default) is required so cutter faces actually
    # carry the cutter's material into a new slot on the target -- this is
    # how the Mouth_Interior material reaches the cavity walls; INDEX would
    # silently keep every new face on the target's existing material index.
    mod.material_mode = "TRANSFER"
    bpy.context.view_layer.objects.active = target_obj
    bpy.ops.object.modifier_apply(modifier=mod.name)
    cutter_mesh = cutter.data
    bpy.data.objects.remove(cutter, do_unlink=True)
    if cutter_mesh.users == 0:
        bpy.data.meshes.remove(cutter_mesh)


def _material_index(me, name: str):
    for i, m in enumerate(me.materials):
        if m is not None and m.name == name:
            return i
    return None


def _reference_bag_size(
    mouth_cfg: dict, body_height: float, mouth_x: float, fissure_z: float,
    front_sign: int, lip_surface_y: float, teeth_bbox: dict,
) -> dict:
    """Half-extents (rx, ry, rz) for a bag cutter centered at (mouth_x,
    <computed>, fissure_z) that fully encloses `teeth_bbox` with a
    _REFERENCE_MARGIN (20%) margin on every axis.

    x/z are straightforward (the cutter's x/z location is fixed at
    mouth_x/fissure_z, independent of rx/rz, so "reach" is just the distance
    from that fixed center to the margined bbox edge). y is solved in closed
    form because _make_mouth_cutter's center_y placement formula itself
    depends on ry (poke_margin = 0.15*ry): the cutter's front-most point is
    always lip_surface_y + front_sign*0.15*ry (poke past the lip surface by
    15% of ry, wherever ry ends up), and its back-most reach is
    lip_surface_y - front_sign*1.85*ry -- solving "back-most reach covers the
    margined teeth back edge" for ry gives the 1.85 factor below.
    """
    default_rx = mouth_cfg["cutter_radii"][0] * body_height
    default_ry = mouth_cfg["cutter_radii"][1] * body_height
    default_rz = mouth_cfg.get("bag_height", mouth_cfg["cutter_radii"][2]) * body_height

    txmin, txmax = teeth_bbox["min"].x, teeth_bbox["max"].x
    tymin, tymax = teeth_bbox["min"].y, teeth_bbox["max"].y
    tzmin, tzmax = teeth_bbox["min"].z, teeth_bbox["max"].z
    t_cx = (txmin + txmax) / 2.0
    t_cz = (tzmin + tzmax) / 2.0

    half_x_margined = 0.5 * _REFERENCE_MARGIN * max(txmax - txmin, 1e-6)
    half_z_margined = 0.5 * _REFERENCE_MARGIN * max(tzmax - tzmin, 1e-6)
    rx = max(default_rx, abs(t_cx - mouth_x) + half_x_margined)
    rz = max(default_rz, abs(t_cz - fissure_z) + half_z_margined)

    tooth_depth_margined = 0.5 * _REFERENCE_MARGIN * max(tymax - tymin, 1e-6)
    back_target = (tymin - tooth_depth_margined) if front_sign > 0 else (tymax + tooth_depth_margined)
    depth_needed = front_sign * (lip_surface_y - back_target)  # inward distance from lip surface
    ry = default_ry
    if depth_needed > 0:
        ry = max(default_ry, depth_needed / 1.85)

    return {"rx": rx, "ry": ry, "rz": rz}


def _make_mouth_cutter(
    mouth_cfg: dict, body_height: float, mouth_x: float, fissure_z: float,
    front_sign: int, lip_surface_y: float, size_override: dict | None = None,
    notes: list | None = None,
) -> str:
    """Build the closed-lips bag cutter.

    Local axes: x=width, y=depth (front = toward the lips, along front_sign),
    z=height. Starts as an ellipsoid (rx, ry, rz), then a per-vertex z-ramp
    on the FRONT half (front_sign * y_local > 0) tapers z down to a thin
    horizontal slit of total height cfg mouth.lip_gap_mm exactly at the
    local-Y depth (y0) where the cutter actually crosses the lip surface
    (see the closed-form solve below), via a smoothstep of y_local/y0 -- the
    BACK half (inside the head) is left at full rz height, which is the
    "bag" that holds teeth/tongue.
    """
    import bpy
    import bmesh
    from mathutils import Matrix

    if size_override is not None:
        rx, ry, rz = size_override["rx"], size_override["ry"], size_override["rz"]
    else:
        rx = mouth_cfg["cutter_radii"][0] * body_height
        ry = mouth_cfg["cutter_radii"][1] * body_height
        rz = mouth_cfg.get("bag_height", mouth_cfg["cutter_radii"][2]) * body_height

    poke_margin = 0.15 * ry
    y0 = ry - poke_margin  # local-Y depth (magnitude) where the cutter surface
    # crosses the world plane Y=lip_surface_y -- since the cutter is only
    # translated (never rotated/sheared), EVERY point of its surface with
    # world Y==lip_surface_y has this exact same local Y, regardless of X or
    # Z (see _make_mouth_cutter's derivation notes below); this is where the
    # boolean actually carves the skin, not the sphere's Y-pole at y=ry.
    center_y = lip_surface_y - front_sign * y0

    lip_gap_units = mouth_cfg.get("lip_gap_mm", 0.4) * _MM
    slit_half = lip_gap_units / 2.0

    # The untapered ellipsoid's own half-height at local depth y0, x=0 is
    # rz*sqrt(1-(y0/ry)^2) (the ellipsoid equation) -- NOT rz, because y0 is
    # already partway toward the pole. A taper that reaches its nominal
    # "full slit_half/rz" factor only in the limit y->ry (the actual pole)
    # therefore stays well short of that factor AT y0, where the cut really
    # happens (verified empirically: with the taper's endpoint at y=ry, the
    # ellipsoid's own narrowing plus an under-shot taper factor left
    # lip_gap_mm_measured ~1.6-1.8mm regardless of mesh resolution -- a
    # geometry bug, not a discretization artifact). Solving directly for the
    # taper factor f0 that makes the cutter's half-height AT y0 equal
    # slit_half exactly: rz*f0*sqrt(1-(y0/ry)^2) == slit_half.
    sqrt_term = math.sqrt(max(0.0, 1.0 - (y0 / ry) ** 2)) if ry > 1e-9 else 0.0
    degenerate = rz <= 1e-9 or sqrt_term <= 1e-6
    slit_factor = 1.0
    if not degenerate:
        slit_factor = slit_half / (rz * sqrt_term)
        if slit_factor >= 1.0:
            degenerate = True  # target slit isn't thinner than the natural shape at y0
    if degenerate and notes is not None:
        notes.append(
            f"mouth cutter: lip_gap_mm target ({lip_gap_units:.5f}) is not thinner than "
            f"the bag's natural cross-section at the poke depth (rz={rz:.5f}); "
            "skipping the slit taper (degenerate config), cutter stays a plain bag"
        )
    slit_factor = max(min(slit_factor, 1.0), 1e-6)

    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=24, v_segments=20, radius=1.0)
    # bmesh.ops.create_uvsphere puts its poles on Z, so its "rings" (latitude
    # bands) are constant-Z, NOT constant-Y -- within any one ring, Y varies
    # freely from -sin(phi) to +sin(phi) as the ring wraps around in X/Y.
    # Tapering per-vertex on raw Y (as a first attempt did) therefore only
    # thins the handful of vertices nearest the Y-pole; vertices on the SAME
    # ring but nearer the ring's X-extremes (the mouth CORNERS once scaled)
    # keep their full pre-taper Z and the "slit" ends up tall at the corners
    # regardless of resolution (verified empirically: lip_gap_mm_measured
    # stayed ~1.8mm at both 32x24 and 160x120 sphere segments -- a geometric
    # effect, not a discretization artifact). Rotating -90 degrees about X
    # first ((x,y,z) -> (x,z,-y), a proper rotation, det=+1) moves the poles
    # onto Y, so rings become constant-Y depth bands and the per-vertex taper
    # below -- which depends only on Y -- now scales EVERY vertex in a ring
    # (i.e. the full X/Z circle at that depth) by the same factor, giving a
    # uniformly thin slit across the whole mouth width instead of just at
    # its center.
    for v in bm.verts:
        vx, vy, vz = v.co.x, v.co.y, v.co.z
        v.co.x = vx
        v.co.y = vz
        v.co.z = -vy
    bmesh.ops.scale(bm, vec=(rx, ry, rz), space=Matrix.Identity(4), verts=bm.verts)

    if not degenerate:
        for v in bm.verts:
            y_local = v.co.y
            if front_sign * y_local > 0:
                # Normalized by y0 (the poke-depth ring), not ry (the pole):
                # t reaches 1.0 (full taper, f == slit_factor) exactly at the
                # ring where the boolean cut happens, and stays clamped at
                # slit_factor for the remaining sliver out to the pole (y0 <
                # ry by poke_margin) -- that sliver just pokes past the skin
                # and is never visible, so flattening it out is harmless.
                t = min(abs(y_local) / y0, 1.0) if y0 > 1e-9 else 1.0
                s = t * t * (3.0 - 2.0 * t)  # smoothstep(0,1,t)
                f = 1.0 + (slit_factor - 1.0) * s
                v.co.z *= f

    me = bpy.data.meshes.new("MouthCutter_mesh")
    bm.to_mesh(me)
    bm.free()

    mat = _get_or_create_material(_MOUTH_INTERIOR_MAT, (0.02, 0.005, 0.006, 1.0))
    me.materials.append(mat)
    for p in me.polygons:
        p.material_index = 0

    obj = bpy.data.objects.new("MouthCutter", me)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = (mouth_x, center_y, fissure_z)
    bpy.context.view_layer.update()
    return obj.name


def _squeeze_slit_to_target(body_obj, mat_idx: int, fissure_z: float, target_half_gap: float) -> int:
    """Pull every Mouth_Interior-rim vertex's world Z back to within
    `target_half_gap` of `fissure_z`, clamping (not rescaling) so vertices
    already inside the target stay put -- only outliers move. Returns the
    number of vertices moved. See the safety-net comment at the call site
    for why this exists (irregular real meshes can produce a much wider raw
    cut than the cutter's own thin design intends).
    """
    import bmesh
    from mathutils import Vector

    me = body_obj.data
    bm = bmesh.new()
    try:
        bm.from_mesh(me)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        world = body_obj.matrix_world
        world_inv = world.inverted()

        rim_vert_idx: set[int] = set()
        for e in bm.edges:
            faces = e.link_faces
            if len(faces) != 2:
                continue
            mis = {f.material_index for f in faces}
            if mat_idx in mis and len(mis) == 2:
                rim_vert_idx.add(e.verts[0].index)
                rim_vert_idx.add(e.verts[1].index)

        moved = 0
        for vi in rim_vert_idx:
            v = bm.verts[vi]
            w = world @ v.co
            offset = w.z - fissure_z
            if abs(offset) > target_half_gap:
                clamped_offset = target_half_gap if offset > 0 else -target_half_gap
                w2 = Vector((w.x, w.y, fissure_z + clamped_offset))
                v.co = world_inv @ w2
                moved += 1

        if moved:
            bm.to_mesh(me)
            me.update()
        return moved
    finally:
        bm.free()


def _measure_mouth_opening(body_obj, mat_idx: int):
    import bmesh

    bm = bmesh.new()
    try:
        bm.from_mesh(body_obj.data)
        bm.faces.ensure_lookup_table()
        world = body_obj.matrix_world
        rim_pts = []
        for e in bm.edges:
            faces = e.link_faces
            if len(faces) != 2:
                continue
            mis = {f.material_index for f in faces}
            if mat_idx in mis and len(mis) == 2:
                for v in e.verts:
                    rim_pts.append(world @ v.co)
        if not rim_pts:
            return None
        xs = [p.x for p in rim_pts]
        ys = [p.y for p in rim_pts]
        zs = [p.z for p in rim_pts]
        return {
            "x_range": (min(xs), max(xs)),
            "z_range": (min(zs), max(zs)),
            "y_values": ys,
        }
    finally:
        bm.free()


def _measure_bag_extent(body_obj, mat_idx: int, front_sign: int, lip_surface_y: float):
    """z-extent of Mouth_Interior faces deeper than _BAG_DEPTH_MARGIN_MM
    behind the lip surface -- the interior cavity ("bag"), as opposed to
    `_measure_mouth_opening`'s thin slit rim at the outer skin surface.
    Returns None if no such faces are found.
    """
    import bmesh

    depth_min = _BAG_DEPTH_MARGIN_MM * _MM
    bm = bmesh.new()
    try:
        bm.from_mesh(body_obj.data)
        bm.faces.ensure_lookup_table()
        world = body_obj.matrix_world
        zs = []
        for f in bm.faces:
            if f.material_index != mat_idx:
                continue
            center_w = world @ f.calc_center_median()
            depth = front_sign * (lip_surface_y - center_w.y)
            if depth >= depth_min:
                for v in f.verts:
                    zs.append((world @ v.co).z)
        if not zs:
            return None
        return [min(zs), max(zs)]
    finally:
        bm.free()


def _lips_topologically_split(body_obj, mat_idx: int, fissure_z: float) -> bool:
    """True iff the upper-lip slit rim and lower-lip slit rim share no
    vertices -- i.e. the boolean cut actually topologically separated the
    upper and lower lips (required for a rig to open the jaw later) rather
    than leaving them fused at some point along the slit.

    A single thin closed-lips slit is one continuous rim loop (a Jordan
    curve) around a small opening -- by construction it MUST cross the
    fissure_z plane at exactly two points (the mouth corners), and any
    vertex classification keyed on EDGE midpoint (or on non-strict "z >=
    fissure_z") inevitably assigns those two corner vertices to both
    buckets, since one of their two rim edges heads into the upper arc and
    the other into the lower arc -- that is expected, healthy topology (real
    lip corners are a single point too), not a fused cut. So classification
    here only counts a vertex toward `upper`/`lower` via a "pure" rim edge
    (BOTH endpoints strictly on the same side of fissure_z); an edge that
    straddles fissure_z (a mouth-corner transition edge) contributes to
    neither bucket. A genuinely fused/botched cut -- where the taper failed
    and a wide band of material still connects across the slit -- produces
    actual PURE-edge vertices shared between both sides (not just the two
    corner transition points), which this still catches.
    """
    import bmesh

    bm = bmesh.new()
    try:
        bm.from_mesh(body_obj.data)
        bm.edges.ensure_lookup_table()
        world = body_obj.matrix_world
        upper_verts: set[int] = set()
        lower_verts: set[int] = set()
        for e in bm.edges:
            faces = e.link_faces
            if len(faces) != 2:
                continue
            mis = {f.material_index for f in faces}
            if mat_idx not in mis or len(mis) != 2:
                continue
            v0, v1 = e.verts
            z0 = (world @ v0.co).z
            z1 = (world @ v1.co).z
            if z0 > fissure_z and z1 > fissure_z:
                upper_verts.add(v0.index)
                upper_verts.add(v1.index)
            elif z0 < fissure_z and z1 < fissure_z:
                lower_verts.add(v0.index)
                lower_verts.add(v1.index)
            # else: straddles fissure_z (a mouth-corner transition edge) --
            # contributes to neither bucket, see docstring.
        if not upper_verts and not lower_verts:
            return False
        return upper_verts.isdisjoint(lower_verts)
    finally:
        bm.free()


def _build_arch_object(name, major_radius, minor_radius, front_sign, squash_x, major_segments=20, minor_segments=8):
    """Manually-built half-torus "dental arch" tube (bmesh.ops has no create_torus).

    Swept 180 degrees centered on the front axis (equivalent to "build a
    torus, delete the back half"), open ends capped with holes_fill. Front
    axis is Y (scaled by front_sign); the sweep radius is in the XY plane,
    tube cross-section spans the outward-radial and +Z directions (standard
    torus parametrization), so the result naturally squashes correctly on X.
    """
    import bpy
    import bmesh
    from mathutils import Vector, Matrix

    bm = bmesh.new()
    theta_max = math.pi / 2.0
    front_axis = Vector((0.0, float(front_sign), 0.0))
    side_axis = Vector((1.0, 0.0, 0.0))
    up_axis = Vector((0.0, 0.0, 1.0))

    grid = []
    for i in range(major_segments + 1):
        theta = -theta_max + (2 * theta_max) * i / major_segments
        radial = front_axis * math.cos(theta) + side_axis * math.sin(theta)
        ring_center = radial * major_radius
        row = []
        for j in range(minor_segments):
            phi = 2 * math.pi * j / minor_segments
            offset = radial * (minor_radius * math.cos(phi)) + up_axis * (minor_radius * math.sin(phi))
            row.append(bm.verts.new(ring_center + offset))
        grid.append(row)
    bm.verts.ensure_lookup_table()

    for i in range(major_segments):
        for j in range(minor_segments):
            a = grid[i][j]
            b = grid[i][(j + 1) % minor_segments]
            c = grid[i + 1][(j + 1) % minor_segments]
            d = grid[i + 1][j]
            bm.faces.new((a, b, c, d))
    bm.faces.ensure_lookup_table()
    bm.edges.ensure_lookup_table()

    boundary_edges = [e for e in bm.edges if e.is_boundary]
    if boundary_edges:
        bmesh.ops.holes_fill(bm, edges=boundary_edges, sides=0)

    bmesh.ops.scale(bm, vec=(squash_x, 1.0, 1.0), space=Matrix.Identity(4), verts=bm.verts)
    bm.normal_update()

    me = bpy.data.meshes.new(f"{name}_mesh")
    bm.to_mesh(me)
    bm.free()
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _place_arch(obj, mouth_x, z_target, target_front_y, rim_front_y, front_sign, major_radius, minor_radius) -> float:
    """Position an arch so its front-most point sits at target_front_y.

    The arch mesh is built with its front-most local point at
    front_sign * (major_radius + minor_radius) along Y (squash only scales
    X, so this analytic value is exact); solving for object.location.y is
    then a closed-form shift. Returns the actual achieved recess distance
    (signed distance behind rim_front_y along the front axis) in scene
    units, re-measured from the placed object for honesty rather than
    trusting the analytic target.
    """
    import bpy
    from mathutils import Vector

    local_front_y = front_sign * (major_radius + minor_radius)
    obj.location = Vector((mouth_x, target_front_y - local_front_y, z_target))
    bpy.context.view_layer.update()

    # re-measure actual front-most vertex world Y for an honest recess figure
    mat = obj.matrix_world
    front_ys = [(mat @ Vector(v.co)).y for v in obj.data.vertices]
    actual_front_y = max(front_ys) if front_sign > 0 else min(front_ys)
    recess = front_sign * (rim_front_y - actual_front_y)
    return recess


def _build_tongue_object(x, y, z, major_radius, minor_radius):
    import bpy
    import bmesh
    from mathutils import Matrix

    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=16, v_segments=10, radius=1.0)
    sx = major_radius * 0.6
    sy = major_radius * 0.9
    sz = max(minor_radius * 0.6, 1e-4)
    bmesh.ops.scale(bm, vec=(sx, sy, sz), space=Matrix.Identity(4), verts=bm.verts)
    me = bpy.data.meshes.new("Tongue_mesh")
    bm.to_mesh(me)
    bm.free()
    obj = bpy.data.objects.new("Tongue", me)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = (x, y, z)
    return obj


def _build_tongue_reference(ctx, mouth_x: float, front_sign: int, teeth_bbox: dict) -> None:
    """Build a tongue positioned inside the bag, below/behind the reference
    teeth object's bbox center -- there is no tongue classifier, so a tongue
    is always built even in reference_kept mode.
    """
    txmin, txmax = teeth_bbox["min"].x, teeth_bbox["max"].x
    tymin, tymax = teeth_bbox["min"].y, teeth_bbox["max"].y
    tzmin, tzmax = teeth_bbox["min"].z, teeth_bbox["max"].z

    teeth_width = max(txmax - txmin, 1e-6)
    teeth_depth = max(tymax - tymin, 1e-6)
    teeth_height = max(tzmax - tzmin, 1e-6)

    major_radius = 0.5 * teeth_width * 0.9
    minor_radius = max(0.4 * teeth_height, 1e-4)

    teeth_back_y = tymin if front_sign > 0 else tymax  # more-inward extent
    tongue_y = teeth_back_y - front_sign * 0.3 * teeth_depth  # further behind
    tongue_z = tzmin - 0.3 * teeth_height  # below the teeth

    tongue_mat = _get_or_create_material("Tongue", (0.55, 0.14, 0.18, 1.0))
    tongue = _build_tongue_object(mouth_x, tongue_y, tongue_z, major_radius, minor_radius)
    tongue.data.materials.append(tongue_mat)
    ctx.names["tongue"] = tongue.name
