"""Phase 6 -- eye openings and eyeball fit/replace.

Closed-lid sculpts cover the eyeballs entirely; carve almond openings with
boolean ellipsoid DIFFERENCE at the fissure line (~2mm below the eyeball
equator, per cfg eyes.fissure_offset_mm). If cfg eyes.replace_dirty_eyeballs,
replace each eyeball with a clean UV sphere (data API) sized by a
max-inscribed-sphere fit against the (post-carve) skin, with a procedural
iris texture. Socket interior geometry is intentionally NOT built by default
(cfg eyes.build_socket == False) -- see docs/blender-body-mesh-cleanup.md
Phase 6 ("it is often the right call to skip the socket entirely").

Uses the bmesh/data API exclusively for primitive creation (never
bpy.ops.mesh.primitive_*_add), re-fetches ctx.obj("body") after every
operator call that could invalidate a cached reference, and measures eyeball
centers fresh (not from any earlier phase) at carve time per the docs
pitfall notes.

Unit-convention note: cfg mm-based values (fissure_offset_mm, the 0.5mm
inscribed-sphere margin) are converted to Blender units assuming 1 Blender
unit == 1 meter. Documented assumption, recorded in PhaseResult.notes.
"""

from __future__ import annotations

import math

from mesh_pipeline.context import PhaseResult
from mesh_pipeline.geom import fibonacci_sphere
from mesh_pipeline.phases.p5_mouth import _get_or_create_material

_SOCKET_INTERIOR_MAT = "Eye_Socket_Interior"

PHASE_NAME = "p6_eyes"
DESTRUCTIVE = True

_MM = 0.001  # 1 Blender unit == 1 meter (documented assumption, see module docstring)
_MARGIN_MM = 0.5
_FIT_OFFSETS_MM = (-2.0, -1.0, 0.0, 1.0, 2.0)
_FIT_RAY_COUNT = 24
_ASYMMETRY_THRESHOLD = 0.30


def _empty_metrics() -> dict:
    return {
        "skipped": None,
        "clearance_l": None,
        "clearance_r": None,
        "asymmetry_ratio": None,
        "local_visibility_retried": None,
    }


def run(ctx, cfg: dict) -> PhaseResult:
    eyes_cfg = cfg["eyes"]
    if not eyes_cfg["enabled"]:
        metrics = _empty_metrics()
        metrics["skipped"] = True
        return PhaseResult(
            phase=PHASE_NAME, status="ok", metrics=metrics, notes=["skipped: disabled"]
        )

    import bpy
    from mathutils import Vector

    ctx.ensure_object_mode()

    notes = [
        "unit assumption: 1 Blender unit == 1 meter, used to convert "
        "eyes.fissure_offset_mm and the inscribed-sphere fit margin into scene units"
    ]

    missing_roles = [r for r in ("eye_l", "eye_r") if r not in ctx.names]
    failures = []
    if "body" not in ctx.names:
        failures.append("p6_eyes: no 'body' role registered in ctx.names")
    if missing_roles:
        failures.append(
            f"p6_eyes: missing eye role(s) {missing_roles} in ctx.names -- expected "
            "from p2 classification; cannot carve/measure eyes"
        )
    if failures:
        metrics = _empty_metrics()
        metrics["skipped"] = False
        metrics["local_visibility_retried"] = False
        return PhaseResult(phase=PHASE_NAME, status="needs_review", metrics=metrics, notes=notes, failures=failures)

    body = ctx.obj("body")
    bpy.context.view_layer.update()

    eye_l_info = _measure_eyeball(ctx.obj("eye_l"))
    eye_r_info = _measure_eyeball(ctx.obj("eye_r"))
    notes.append(
        f"measured eye_l center={tuple(round(c, 4) for c in eye_l_info['center'])} "
        f"radius={eye_l_info['radius']:.4f}"
    )
    notes.append(
        f"measured eye_r center={tuple(round(c, 4) for c in eye_r_info['center'])} "
        f"radius={eye_r_info['radius']:.4f}"
    )

    bbox = _world_bbox(body)
    body_center_y = (bbox["min"].y + bbox["max"].y) / 2.0
    avg_eye_y = (eye_l_info["center"].y + eye_r_info["center"].y) / 2.0
    front_sign = 1 if avg_eye_y >= body_center_y else -1
    notes.append(
        "front-axis heuristic (eyes): eyes sit on the front of the face by "
        "definition, so front_sign is read directly from eye Y position relative "
        f"to body bbox center (avg_eye_y={avg_eye_y:.4f}, "
        f"body_center_y={body_center_y:.4f}) -> front_sign={front_sign}"
    )

    # --- carve almond openings (fresh BVH rebuilt after each cut) ---
    bvh = _world_bvh(body)
    for role, info in (("eye_l", eye_l_info), ("eye_r", eye_r_info)):
        _carve_eye_opening(body, bvh, eyes_cfg, info, front_sign)
        body = ctx.obj("body")
        bvh = _world_bvh(body)

    if eyes_cfg["build_socket"]:
        notes.append("socket building not implemented in v1 (high risk, see docs)")

    replace = eyes_cfg["replace_dirty_eyeballs"]
    if replace:
        for role, info in (("eye_l", eye_l_info), ("eye_r", eye_r_info)):
            new_name = _replace_eyeball(ctx, role, info, front_sign, bvh)
            notes.append(f"replaced {role} -> {new_name} radius={info['fit_radius']:.4f}")

    # --- final clearance measurement ---
    body = ctx.obj("body")
    bvh_final = _world_bvh(body)
    directions = [Vector(d) for d in fibonacci_sphere(_FIT_RAY_COUNT)]

    def _current(info):
        if replace:
            return info["fit_center"], info["fit_radius"]
        return info["center"], info["radius"]

    cl_center, cl_radius = _current(eye_l_info)
    cr_center, cr_radius = _current(eye_r_info)
    max_dist_l = cl_radius * 20 + 0.05
    max_dist_r = cr_radius * 20 + 0.05

    clearance_l = _measure_clearance(cl_center, bvh_final, directions, max_dist_l)
    clearance_r = _measure_clearance(cr_center, bvh_final, directions, max_dist_r)
    asymmetry = _asymmetry(clearance_l, clearance_r)

    local_retry = False
    if asymmetry is not None and asymmetry > _ASYMMETRY_THRESHOLD:
        _local_visibility_retry(
            body, [(cl_center, cl_radius), (cr_center, cr_radius)],
            cfg["hidden_geometry"]["face_rays"], notes,
        )
        local_retry = True
        body = ctx.obj("body")
        bvh_final = _world_bvh(body)
        clearance_l = _measure_clearance(cl_center, bvh_final, directions, max_dist_l)
        clearance_r = _measure_clearance(cr_center, bvh_final, directions, max_dist_r)
        asymmetry = _asymmetry(clearance_l, clearance_r)

    notes.append(
        f"final clearance: l={clearance_l} r={clearance_r} asymmetry_ratio={asymmetry} "
        f"local_visibility_retried={local_retry}"
    )

    metrics = {
        "skipped": False,
        "clearance_l": clearance_l,
        "clearance_r": clearance_r,
        "asymmetry_ratio": asymmetry,
        "local_visibility_retried": local_retry,
    }
    status = "ok"
    if clearance_l is None or clearance_r is None:
        status = "needs_review"
        failures.append(
            "p6_eyes: could not measure clearance for one or both eyes (no ray hit "
            "the skin from the eyeball center in any sampled direction)"
        )
    return PhaseResult(phase=PHASE_NAME, status=status, metrics=metrics, notes=notes, failures=failures)


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


def _world_bvh(obj):
    import bpy
    from mathutils.bvhtree import BVHTree

    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = obj.evaluated_get(depsgraph)
    me = eval_obj.to_mesh()
    mat = obj.matrix_world.copy()
    verts = [mat @ v.co for v in me.vertices]
    polys = [list(p.vertices) for p in me.polygons]
    bvh = BVHTree.FromPolygons(verts, polys, epsilon=1e-7)
    eval_obj.to_mesh_clear()
    return bvh


def _measure_eyeball(obj) -> dict:
    """Fresh world-space center/radius via median distance-from-centroid.

    Median (not mean/bbox) is used so a few dented/prosthetic-junk vertices
    don't skew the radius estimate -- robust against exactly the "dirty
    eyeball" case this phase exists to fix.
    """
    import bpy
    import bmesh

    bpy.context.view_layer.update()
    mat = obj.matrix_world
    bm = bmesh.new()
    try:
        bm.from_mesh(obj.data)
        bm.verts.ensure_lookup_table()
        pts = [mat @ v.co for v in bm.verts]
    finally:
        bm.free()
    if not pts:
        raise ValueError(f"eyeball object {obj.name!r} has no vertices")
    n = len(pts)
    cx = sum(p.x for p in pts) / n
    cy = sum(p.y for p in pts) / n
    cz = sum(p.z for p in pts) / n
    from mathutils import Vector

    center = Vector((cx, cy, cz))
    dists = sorted((p - center).length for p in pts)
    radius = dists[len(dists) // 2]
    return {"center": center, "radius": radius}


def _carve_eye_opening(body, bvh, eyes_cfg: dict, info: dict, front_sign: int) -> None:
    import bpy
    import bmesh
    from mathutils import Vector, Matrix

    center = info["center"]
    radius = info["radius"]
    fissure_z = center.z + eyes_cfg["fissure_offset_mm"] * _MM

    opening_height = eyes_cfg["opening_ratio"] * (2.0 * radius)
    half_h = opening_height / 2.0
    half_w = radius * 1.05

    far = center.y + front_sign * (radius * 20.0 + 1.0)
    origin = Vector((center.x, far, fissure_z))
    direction = Vector((0.0, -front_sign, 0.0))
    hit = bvh.ray_cast(origin, direction, radius * 40.0 + 2.0)
    lid_y = hit[0].y if hit[0] is not None else center.y + front_sign * radius

    poke = 0.15 * half_w
    front_tip_y = lid_y + front_sign * poke
    back_y = center.y
    center_y = (front_tip_y + back_y) / 2.0
    half_depth = abs(front_tip_y - back_y) / 2.0 + 1e-5

    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=20, v_segments=14, radius=1.0)
    bmesh.ops.scale(bm, vec=(half_w, half_depth, half_h), space=Matrix.Identity(4), verts=bm.verts)
    me = bpy.data.meshes.new("EyeCutter_mesh")
    bm.to_mesh(me)
    bm.free()

    # Assign a dark socket-interior material to the cutter BEFORE the boolean:
    # material_mode='TRANSFER' stamps it onto the carved cavity walls, exactly
    # like p5's Mouth_Interior. Without this the cavity faces inherit the
    # body's own material slot 0 (skin) with non-confetti UVs, which dilutes
    # p7's per-material confetti ratio and can silently disable the body
    # re-unwrap (found by the e2e run: ratio 0.826 < 0.9 threshold).
    mat = _get_or_create_material(_SOCKET_INTERIOR_MAT, (0.02, 0.01, 0.01, 1.0))
    me.materials.append(mat)
    for p in me.polygons:
        p.material_index = 0

    obj = bpy.data.objects.new("EyeCutter", me)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = (center.x, center_y, fissure_z)
    bpy.context.view_layer.update()

    _boolean_difference(body, obj.name)


def _boolean_difference(target_obj, cutter_obj_name: str) -> None:
    import bpy

    cutter = bpy.data.objects[cutter_obj_name]
    mod = target_obj.modifiers.new("p6_bool_diff", "BOOLEAN")
    mod.object = cutter
    mod.operation = "DIFFERENCE"
    mod.solver = "EXACT"
    mod.material_mode = "TRANSFER"  # see p5_mouth._boolean_difference for why
    bpy.context.view_layer.objects.active = target_obj
    bpy.ops.object.modifier_apply(modifier=mod.name)
    cutter_mesh = cutter.data
    bpy.data.objects.remove(cutter, do_unlink=True)
    if cutter_mesh.users == 0:
        bpy.data.meshes.remove(cutter_mesh)


def _measure_clearance(center, bvh, directions, max_dist):
    """Min ray-cast distance from center to the skin; rays that miss (pass
    through the opening) are excluded, not treated as zero clearance."""
    clearances = []
    for d in directions:
        hit = bvh.ray_cast(center, d, max_dist)
        if hit[3] is not None:
            clearances.append(hit[3])
    if not clearances:
        return None
    return min(clearances)


def _asymmetry(clearance_l, clearance_r):
    if clearance_l is None or clearance_r is None:
        return None
    denom = max(clearance_l, clearance_r)
    if denom <= 0:
        return None
    return abs(clearance_l - clearance_r) / denom


def _fit_inscribed_sphere(center0, bvh, directions, margin, max_dist):
    """Grid-search the eyeball center (+-2mm per axis) maximizing the min
    ray-cast clearance to the skin; radius = best min clearance - margin.
    Rays through the opening (no hit) impose no constraint."""
    from mathutils import Vector

    best_center = center0
    best_min_clear = _measure_clearance(center0, bvh, directions, max_dist)
    if best_min_clear is None:
        best_min_clear = margin * 2.0  # degenerate: nothing hit; fall back to a small sphere

    for dx in _FIT_OFFSETS_MM:
        for dy in _FIT_OFFSETS_MM:
            for dz in _FIT_OFFSETS_MM:
                if dx == 0.0 and dy == 0.0 and dz == 0.0:
                    continue
                cand = center0 + Vector((dx, dy, dz)) * _MM
                mc = _measure_clearance(cand, bvh, directions, max_dist)
                if mc is not None and mc > best_min_clear:
                    best_min_clear = mc
                    best_center = cand

    radius = max(best_min_clear - margin, margin)
    return best_center, radius


def _replace_eyeball(ctx, role: str, info: dict, front_sign: int, bvh) -> str:
    import bpy
    import bmesh
    from mathutils import Vector, Matrix

    directions = [Vector(d) for d in fibonacci_sphere(_FIT_RAY_COUNT)]
    max_dist = info["radius"] * 20.0 + 0.05
    fit_center, fit_radius = _fit_inscribed_sphere(
        info["center"], bvh, directions, _MARGIN_MM * _MM, max_dist
    )
    info["fit_center"] = fit_center
    info["fit_radius"] = fit_radius

    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=32, v_segments=20, radius=1.0)
    bm.faces.ensure_lookup_table()
    uv_layer = bm.loops.layers.uv.new("UVMap")
    for f in bm.faces:
        loop_uvs = []
        for loop in f.loops:
            co = loop.vert.co  # canonical (pre-rotation, pre-translation) radius-1 coords
            z = max(-1.0, min(1.0, co.z))
            theta = math.acos(z)
            v = 1.0 - theta / math.pi  # v=1 at the +Z pole == pupil
            u = math.atan2(co.y, co.x) / (2 * math.pi) + 0.5
            loop_uvs.append([u, v])
        us = [uv[0] for uv in loop_uvs]
        if us and (max(us) - min(us)) > 0.5:
            for uv in loop_uvs:
                if uv[0] < 0.5:
                    uv[0] += 1.0
        for loop, uv in zip(f.loops, loop_uvs):
            loop[uv_layer].uv = uv

    # pole rotation: canonical pole is +Z; rotate so it faces the detected front axis
    target = Vector((0.0, float(front_sign), 0.0))
    rot = Vector((0.0, 0.0, 1.0)).rotation_difference(target).to_matrix().to_4x4()
    bmesh.ops.rotate(bm, cent=(0.0, 0.0, 0.0), matrix=rot, verts=bm.verts)
    bmesh.ops.scale(bm, vec=(fit_radius, fit_radius, fit_radius), space=Matrix.Identity(4), verts=bm.verts)

    me = bpy.data.meshes.new(f"{role}_clean_mesh")
    bm.to_mesh(me)
    bm.free()

    mat = _make_iris_material(role)
    me.materials.append(mat)

    new_obj = bpy.data.objects.new(f"{role}_clean", me)
    bpy.context.scene.collection.objects.link(new_obj)
    new_obj.location = fit_center
    bpy.context.view_layer.update()

    old_obj = ctx.obj(role)
    old_mesh = old_obj.data
    bpy.data.objects.remove(old_obj, do_unlink=True)
    if old_mesh.users == 0:
        bpy.data.meshes.remove(old_mesh)

    ctx.names[role] = new_obj.name
    return new_obj.name


def _make_iris_material(role: str):
    import bpy

    name = f"Iris_{role}"
    mat = bpy.data.materials.get(name)
    if mat is not None:
        return mat
    img = _make_iris_image(f"IrisTex_{role}", 128, 128)
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = img
    if bsdf is not None:
        nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


def _make_iris_image(name: str, w: int, h: int):
    """Equirect iris texture: v=1 pole = pupil, zoned pupil/iris/sclera by
    polar angle, radial fibers = multi-frequency sin over azimuth with
    integer frequencies so the u=0/u=1 seam is invisible."""
    import bpy

    img = bpy.data.images.new(name, width=w, height=h)
    pupil_v = 0.94
    iris_v = 0.55
    freq_a, freq_b = 23, 37

    px = [0.0] * (w * h * 4)
    for j in range(h):
        v = j / (h - 1)
        for i in range(w):
            u = i / (w - 1)
            idx = (j * w + i) * 4
            if v >= pupil_v:
                r, g, b = 0.02, 0.02, 0.02
            elif v >= iris_v:
                phi = u * 2 * math.pi
                fiber = 0.5 + 0.5 * (0.6 * math.sin(freq_a * phi) + 0.4 * math.sin(freq_b * phi))
                t = (v - iris_v) / (pupil_v - iris_v)
                shade = 0.25 + 0.35 * fiber
                r, g, b = shade * 0.45, shade * 0.55, shade * 0.65
                edge = min(t, 1.0 - t)
                if edge < 0.06:
                    darken = edge / 0.06
                    r *= darken
                    g *= darken
                    b *= darken
            else:
                r, g, b = 0.92, 0.90, 0.88
            px[idx] = r
            px[idx + 1] = g
            px[idx + 2] = b
            px[idx + 3] = 1.0
    img.pixels = px
    img.pack()
    return img


def _local_visibility_retry(body, centers_radii, face_rays: int, notes: list) -> int:
    """One local per-face visibility pass in the eye region: delete faces
    within 1.5x eyeball radius of either eye center that are fully hidden
    (no fibonacci-sphere ray from center+normal*eps escapes to the skin
    exterior). Mirrors the Phase 3 hidden-geometry method, scoped locally."""
    import bmesh
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree

    me = body.data
    mat = body.matrix_world
    bm = bmesh.new()
    bm.from_mesh(me)
    bm.faces.ensure_lookup_table()
    bm.verts.ensure_lookup_table()

    verts_world = [mat @ v.co for v in bm.verts]
    polys = [[v.index for v in f.verts] for f in bm.faces]
    bvh = BVHTree.FromPolygons(verts_world, polys, epsilon=1e-7)
    directions = [Vector(d) for d in fibonacci_sphere(face_rays)]

    to_delete = []
    for f in bm.faces:
        center_world = mat @ f.calc_center_median()
        in_region = any((center_world - c).length <= 1.5 * r for c, r in centers_radii)
        if not in_region:
            continue
        normal_world = (mat.to_3x3() @ f.normal).normalized()
        origin = center_world + normal_world * 1e-4
        visible = False
        for d in directions:
            if d.dot(normal_world) <= 0.0:
                continue
            hit = bvh.ray_cast(origin, d, 10.0)
            if hit[0] is None:
                visible = True
                break
        if not visible:
            to_delete.append(f)

    deleted = len(to_delete)
    if to_delete:
        bmesh.ops.delete(bm, geom=to_delete, context="FACES")
        bm.to_mesh(me)
        me.update()
    bm.free()
    notes.append(
        f"local visibility retry: scanned faces within 1.5x eyeball radius of "
        f"each eye center, deleted {deleted} hidden faces"
    )
    return deleted
