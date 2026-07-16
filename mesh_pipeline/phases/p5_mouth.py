"""Phase 5 -- mouth bag.

One boolean does both jobs: a DIFFERENCE with an ellipsoid pushed through the
lip line splits the lips AND creates the cavity walls (cutter faces inherit
the cutter's material, so we assign a dark "Mouth_Interior" material to the
cutter before the boolean). Teeth (two half-torus arches, built manually
since bmesh.ops has no create_torus in this bpy build) and a tongue are then
placed relative to the *measured* opening rim, not guessed.

See docs/blender-body-mesh-cleanup.md Phase 5 and PLAN.md `mouth:` config.
Uses the bmesh/data API exclusively for primitive creation (never
bpy.ops.mesh.primitive_*_add) per the ARCHITECTURE.md/docs pitfall list, and
never caches a bpy.types.Object reference across an operator call that can
invalidate it (modifier_apply) -- every use re-fetches via ctx.obj("body").

Unit-convention note: cfg mm-based values (teeth_recess_mm) are converted to
Blender units assuming 1 Blender unit == 1 meter (the common glTF/Blender
default for a human-scale character). This is a documented assumption, not a
measured fact -- it is recorded in PhaseResult.notes on every run so it is
visible in report.json.
"""

from __future__ import annotations

import math

from mesh_pipeline import geom
from mesh_pipeline.context import PhaseResult
from mesh_pipeline.geom import DisjointSet

PHASE_NAME = "p5_mouth"
DESTRUCTIVE = True

_MM = 0.001  # 1 Blender unit == 1 meter (documented assumption, see module docstring)
_MOUTH_INTERIOR_MAT = "Mouth_Interior"


def _empty_metrics() -> dict:
    return {
        "skipped": None,
        "teeth_recess_mm": None,
        "opening_z_range": None,
        "teeth_z_range": None,
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
        "mouth.teeth_recess_mm into scene units"
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

    lip = _find_lip_line(body, bbox, notes)
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

    # --- 1. cutter: ellipsoid, dark Mouth_Interior material assigned FIRST ---
    cutter_name = _make_mouth_cutter(
        mouth_cfg, body_height, mouth_x, fissure_z, front_sign, lip_surface_y
    )

    # --- 2. boolean DIFFERENCE (also splits the lips + tints cavity walls) ---
    body = ctx.obj("body")
    _boolean_difference(body, cutter_name)
    body = ctx.obj("body")
    bpy.context.view_layer.update()

    opening = _measure_mouth_opening(body)
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
    notes.append(
        f"opening rim measured: x=[{x_lo:.4f},{x_hi:.4f}] z=[{z_lo:.4f},{z_hi:.4f}] "
        f"rim_front_y={rim_front_y:.4f}"
    )

    # --- 3. teeth + tongue ---
    recess_mm_mid = sum(mouth_cfg["teeth_recess_mm"]) / 2.0
    recess_units = recess_mm_mid * _MM
    target_front_y = rim_front_y - front_sign * recess_units

    major_radius = 0.5 * opening_width * 0.8
    minor_radius = min(major_radius * 0.35, opening_height * 0.4)
    minor_radius = max(minor_radius, 1e-4)
    z_center = (z_lo + z_hi) / 2.0
    z_upper = z_center + 0.15 * opening_height
    z_lower = z_center - 0.15 * opening_height

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
    tongue_z = z_center - 0.3 * opening_height
    tongue = _build_tongue_object(
        mouth_x, tongue_y, tongue_z, major_radius, minor_radius
    )
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
        f"upper_recess_mm={upper_recess / _MM:.3f} lower_recess_mm={lower_recess / _MM:.3f}"
    )
    if opening_height > 0 and not (
        teeth_z_range[0] <= z_hi and teeth_z_range[1] >= z_lo
    ):
        notes.append(
            "WARNING: teeth_z_range does not overlap opening_z_range -- assertion "
            "table (P5) is expected to flag this"
        )

    metrics = {
        "skipped": False,
        "teeth_recess_mm": actual_recess_mm,
        "opening_z_range": [z_lo, z_hi],
        "teeth_z_range": teeth_z_range,
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


def _find_lip_line(body_obj, bbox: dict, notes: list):
    """Detect the lip line: boundary/sharp-crease edges in the front mouth region.

    Heuristic (documented per task spec):
    1. "Head z-band" = geom.head_z_band(body world verts), which finds the
       contiguous top z-slices before the body's x-width profile "explodes"
       into shoulders/arms (see its docstring for the avatar_003 calibration
       numbers). This replaces a fixed "top 30% of bbox height" rule, which
       was too permissive: on avatar_003 that rule's z-band reached down
       into the neck/upper chest and let a necklace pendant's crease (zfrac
       0.82) get misdetected as the lip line. The top 2% of the resulting
       band is still excluded (scalp/hair). Falls back to the fixed top-30%
       rule, with a note, if geom.head_z_band returns None (degenerate
       geometry / no band found).
    2. Front axis: within that z-band, compute the median |y| ("head radius")
       and compare how far the extreme +Y vertex and extreme -Y vertex
       protrude past it. AI-generated heads typically model a distinct nose
       bump on the front but keep the back of the skull close to the smooth
       median radius, so the side with the larger protrusion is called front.
       This is a heuristic and will misfire on faces with no modeled nose
       bump (e.g. a bare sphere) -- see the caller's needs_review fallback.
    3. Candidate lip edges = boundary edges OR marked-sharp edges (BMEdge.smooth
       == False) OR high dihedral-angle edges (>35 deg), restricted to the
       front half of the head z-band.
    4. Cluster candidates by shared-vertex adjacency (union-find); the winning
       cluster is the widest-in-x/thinnest-in-z one (a lip line is a roughly
       horizontal band across the mouth, not a vertical crease).

    Returns (fissure_z, mouth_x, front_sign, lip_surface_y) or None.
    """
    import bmesh

    bm = bmesh.new()
    try:
        bm.from_mesh(body_obj.data)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        mat = body_obj.matrix_world

        zmin, zmax = bbox["min"].z, bbox["max"].z
        height = zmax - zmin
        head_band = geom.head_z_band(geom.body_xz_points(body_obj))
        if head_band is not None:
            band_lo, band_hi = head_band
            head_lo = band_lo
            head_hi = min(band_hi, zmax - 0.02 * height)  # still exclude scalp/hair tip
            notes.append(
                f"head z-band (geom.head_z_band): [{head_lo:.4f},{head_hi:.4f}]"
            )
        else:
            head_lo = zmax - 0.30 * height
            head_hi = zmax - 0.02 * height
            notes.append(
                "geom.head_z_band returned None (degenerate/no band found); "
                "falling back to fixed top-30% head z-band rule "
                f"[{head_lo:.4f},{head_hi:.4f}]"
            )

        # x-centrality constraint: the mouth sits on the sagittal plane. In a
        # T-pose the wrists/hands are at the SAME height as the chin, and a
        # glove seam there is exactly the kind of sharp-edge cluster the
        # detector otherwise latches onto (avatar_003: lip line "found" at
        # x=0.46 on the wrist -> mouth carved into the arm, caught by p9's
        # penetration test). Restrict everything to the central band of the
        # x extent.
        x_center = (bbox["min"].x + bbox["max"].x) / 2.0
        x_half_limit = 0.10 * max(bbox["max"].x - bbox["min"].x, 1e-9)

        def _central(wco) -> bool:
            return abs(wco.x - x_center) <= x_half_limit

        # front-axis heuristic
        band_ys = []
        for v in bm.verts:
            wco = mat @ v.co
            if head_lo <= wco.z <= head_hi and _central(wco):
                band_ys.append(wco.y)
        if not band_ys:
            notes.append("lip detection: no vertices found in the candidate head z-band")
            return None
        band_ys_sorted = sorted(abs(y) for y in band_ys)
        median_y_abs = band_ys_sorted[len(band_ys_sorted) // 2]
        y_max = max(band_ys)
        y_min = min(band_ys)
        protrusion_pos = y_max - median_y_abs
        protrusion_neg = (-y_min) - median_y_abs
        front_sign = 1 if protrusion_pos >= protrusion_neg else -1
        notes.append(
            "front-axis heuristic (mouth): within head z-band "
            f"[{head_lo:.4f},{head_hi:.4f}], compared how far the extreme +Y "
            f"({protrusion_pos:.4f} past median|y|={median_y_abs:.4f}) and -Y "
            f"({protrusion_neg:.4f} past median) vertices protrude (nose-bump "
            f"asymmetry) -> front_sign={front_sign}"
        )

        candidates = []
        for e in bm.edges:
            v0, v1 = e.verts
            w0 = mat @ v0.co
            w1 = mat @ v1.co
            mid_z = (w0.z + w1.z) / 2.0
            mid_y = (w0.y + w1.y) / 2.0
            if not (head_lo <= mid_z <= head_hi):
                continue
            if front_sign * mid_y <= 0:
                continue
            mid = (w0 + w1) / 2.0
            if not _central(mid):
                continue
            is_boundary = e.is_boundary
            is_sharp = not e.smooth
            is_crease = False
            if len(e.link_faces) == 2:
                is_crease = e.calc_face_angle() > math.radians(35)
            if is_boundary or is_sharp or is_crease:
                candidates.append(e)

        if not candidates:
            notes.append(
                "lip detection: no boundary/sharp/crease edges found on the front "
                "side of the head z-band"
            )
            return None

        vert_to_edges: dict = {}
        for i, e in enumerate(candidates):
            for v in e.verts:
                vert_to_edges.setdefault(v.index, []).append(i)
        ds = DisjointSet(len(candidates))
        for idxs in vert_to_edges.values():
            for a, b in zip(idxs, idxs[1:]):
                ds.union(a, b)
        groups = ds.groups()

        best_members = None
        best_verts_world = None
        best_score = -1.0
        for members in groups.values():
            if len(members) < 3:
                continue
            verts_world = []
            for i in members:
                e = candidates[i]
                for v in e.verts:
                    verts_world.append(mat @ v.co)
            xs = [p.x for p in verts_world]
            zs = [p.z for p in verts_world]
            x_extent = max(xs) - min(xs)
            z_extent = (max(zs) - min(zs)) + 1e-6
            aspect = x_extent / z_extent
            score = aspect * len(members)
            if aspect > 1.0 and score > best_score:
                best_score = score
                best_members = members
                best_verts_world = verts_world

        if best_members is None:
            notes.append(
                "lip detection: front-side candidate edges did not form a "
                "wide-x/thin-z (lip-like) cluster"
            )
            return None

        xs = [p.x for p in best_verts_world]
        ys = [p.y for p in best_verts_world]
        zs = [p.z for p in best_verts_world]
        fissure_z = sum(zs) / len(zs)
        mouth_x = sum(xs) / len(xs)
        lip_surface_y = max(ys) if front_sign > 0 else min(ys)
        notes.append(
            f"lip line detected: {len(best_members)} candidate edges, "
            f"fissure_z={fissure_z:.4f}, mouth_x={mouth_x:.4f}, "
            f"lip_surface_y={lip_surface_y:.4f}"
        )
        return fissure_z, mouth_x, front_sign, lip_surface_y
    finally:
        bm.free()


def _make_mouth_cutter(
    mouth_cfg: dict, body_height: float, mouth_x: float, fissure_z: float,
    front_sign: int, lip_surface_y: float,
) -> str:
    import bpy
    import bmesh
    from mathutils import Matrix

    rx, ry, rz = (r * body_height for r in mouth_cfg["cutter_radii"])
    poke_margin = 0.15 * ry
    center_y = lip_surface_y - front_sign * (ry - poke_margin)

    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=24, v_segments=16, radius=1.0)
    bmesh.ops.scale(bm, vec=(rx, ry, rz), space=Matrix.Identity(4), verts=bm.verts)
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


def _measure_mouth_opening(body_obj):
    import bmesh

    me = body_obj.data
    mat_idx = None
    for i, m in enumerate(me.materials):
        if m is not None and m.name == _MOUTH_INTERIOR_MAT:
            mat_idx = i
            break
    if mat_idx is None:
        return None

    bm = bmesh.new()
    try:
        bm.from_mesh(me)
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
