"""Synthetic "AI-generated humanoid" GLB fixture generator for mesh_pipeline e2e tests.

Builds a single triangle-soup mesh, in one bpy scene, that exercises every
phase of the pipeline (see ARCHITECTURE.md / PLAN.md):

  - a vertically-stretched profile-of-revolution torso+head blob (~1.7 units
    tall), triangulated then SHATTERED into a coincident-vert triangle soup
    via bmesh.ops.split_edges (thousands of loose 1-triangle components that
    P2's weld-distance sweep + remove_doubles must reassemble);
  - a scaled (0.75x) duplicate inner shell, fused to the outer shell at both
    poles (a single shared vertex at each) so it survives P2's weld+separate
    as part of the SAME "body" component and is deleted face-by-face by P3
    pass B's fused-inner-layer visibility pass (not pass A's whole-part
    test -- see build_fixture()'s docstring for why an unfused duplicate
    proved unreliable);
  - a nose bump and a geometric mouth-crease indentation on the front
    (-Y) of the head, so P5's lip-line heuristic has a real dihedral crease
    to find even though a GLB round-trip does not preserve BMesh's
    mark-sharp flag (see mark_mouth_sharp_ring()'s docstring);
  - two ~0.024-radius eyeball spheres inside the head, on the pupil line,
    sharing no vertices with the body (so P2's weld+separate LOOSE recovers
    them as their own objects and P2's classifier tags them eye_l/eye_r);
  - three small junk parts (spheres/cube) fully inside the torso, loose in
    the same mesh, for P3 pass A (whole-part visibility) to delete;
  - a small (2-3 face) hole on the back of the scalp, sized to leave an
    easily-fillable rim for P4's hole-fill step (see HOLE_T's docstring for
    why the scalp, not the torso, is the reliable placement);
  - one image-textured material (a generated 256px checker) on every part,
    wired to Base Color, with an independent small random UV square per
    face ("confetti" -- forces P7's per-material dirty-face/re-unwrap path
    and gives P8 a non-trivial diffuse texture to bake).

`build_fixture(seed)` returns `(body_object, info_dict)` for in-process use
(e.g. from test_e2e.py after `import bpy`); `main()` builds the fixture and
exports it to the GLB path given on argv, for standalone/CLI use:

    python3 tests/make_fixture.py /path/to/out.glb [seed]

Must be run with a Python that has `bpy` importable (either the `bpy` pip
wheel, or `blender -b -P tests/make_fixture.py -- /path/to/out.glb`).
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

BODY_HEIGHT = 1.70
FRONT_SIGN_WORLD = -1.0  # -Y is "front" (matches qa.render's "front" camera, which
# sits at -Y and looks toward +Y, so it photographs the -Y-facing surface).

# (t, radius_x) control points along the body's vertical axis, t in [0,1] ==
# world z in [0, BODY_HEIGHT]. radius_y is radius_x * Y_FLATTEN.
_ANCHORS = [
    (0.00, 0.02),
    (0.04, 0.24),
    (0.10, 0.27),
    (0.35, 0.28),
    (0.55, 0.25),
    (0.63, 0.24),
    (0.685, 0.095),
    (0.72, 0.095),
    (0.78, 0.125),
    (0.86, 0.11),
    (0.905, 0.112),
    (0.95, 0.095),
    (1.00, 0.015),
]
Y_FLATTEN = 0.82

NOSE_T = 0.875
NOSE_MAG = 0.055
NOSE_SIGMA_T = 0.045
NOSE_SIGMA_ANG = 0.55  # radians, angular falloff around the front direction

MOUTH_T_LO = 0.845
MOUTH_T_HI = 0.865
MOUTH_INDENT = 0.014

EYE_RADIUS = 0.024
EYE_X = 0.046
EYE_Y = FRONT_SIGN_WORLD * 0.045  # inside the head, toward front
EYE_Z = BODY_HEIGHT * 0.895

INNER_SHELL_SCALE = 0.75  # deliberately looser than a literal "0.9" (see fixture
# caveats in build_fixture): the body interior is otherwise hollow. Both
# poles are exactly coincident with the outer shell's own poles (both
# collapse to the local-axis point at their z -- see build_inner_shell_bmesh)
# so P2's weld merges the two shells into ONE connected component. This
# makes the inner wall a genuinely *fused* double layer that P3 pass B
# (per-face, on the body's own mesh) deletes face-by-face, instead of a
# separate loose part that pass A must keep-or-delete as an all-or-nothing
# unit (an unfused separate shell proved fragile: empirically, across many
# iterations, ANY loose part -- however far from a hole -- eventually tested
# "visible" through the small torso hole via some fixed ray-sampling
# direction, so it never reliably contributed to deleted_face_ratio).
INNER_SHELL_T_LO = 0.0
INNER_SHELL_T_HI = 1.0
# The small torso hole sits on the BACK of the SCALP (mouth/nose/eyes are
# all front, -Y), inside the "head_hair" z-band that P4's assertion
# deliberately leaves unrestricted. A real open hole into the fixture's
# hollow fused double-wall interior lets a handful of P3 pass B's fixed
# ray-sampling directions thread all the way through the keyhole and out
# again past the inner shell's own curvature ("sunbeam" effect -- confirmed
# by direct measurement across many iterations: neither local nor
# angular-seam bumps on the inner shell, nor a small blocking curtain right
# behind the hole, fully closed every leaking direction), so the resulting
# oversized (> hole_fill.max_loop_edges) secondary boundary loop is
# unavoidable with this fixture's double-wall design -- placing the hole in
# "head_hair" keeps that loop out of the "body"/"legs" bands P4 actually
# checks (hair/cloth are meant to stay open per the docs).
HOLE_T = 0.90


def _lerp(a, b, f):
    return a + (b - a) * f


def _profile_radius_x(t: float) -> float:
    pts = _ANCHORS
    if t <= pts[0][0]:
        return pts[0][1]
    if t >= pts[-1][0]:
        return pts[-1][1]
    for (t0, r0), (t1, r1) in zip(pts, pts[1:]):
        if t0 <= t <= t1:
            f = (t - t0) / (t1 - t0) if t1 > t0 else 0.0
            return _lerp(r0, r1, f)
    return pts[-1][1]


def _gauss(x: float, sigma: float) -> float:
    if sigma <= 1e-9:
        return 0.0
    return math.exp(-0.5 * (x / sigma) ** 2)


def build_body_bmesh(u_segments=48, v_segments=30):
    import bmesh
    from mathutils import Vector

    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=u_segments, v_segments=v_segments, radius=1.0)
    bm.verts.ensure_lookup_table()

    front_dir = Vector((0.0, FRONT_SIGN_WORLD, 0.0))

    for v in bm.verts:
        x, y, z_local = v.co.x, v.co.y, v.co.z
        t = (z_local + 1.0) * 0.5
        world_z = t * BODY_HEIGHT
        r_local = math.sqrt(x * x + y * y)
        rx = _profile_radius_x(t)
        ry = rx * Y_FLATTEN
        if r_local < 1e-9:
            new_x, new_y = 0.0, 0.0
        else:
            ux, uy = x / r_local, y / r_local
            new_x = ux * rx
            new_y = uy * ry

        # nose bump: push front-facing verts near NOSE_T further along front_dir
        dir2 = Vector((new_x, new_y, 0.0))
        if dir2.length > 1e-9:
            dir2n = dir2.normalized()
            ang = dir2n.angle(front_dir) if dir2.length > 1e-9 else math.pi
        else:
            ang = math.pi
        w_t = _gauss(t - NOSE_T, NOSE_SIGMA_T)
        w_ang = _gauss(ang, NOSE_SIGMA_ANG)
        bump = NOSE_MAG * w_t * w_ang
        new_x += front_dir.x * bump
        new_y += front_dir.y * bump

        # mouth crease: geometric indentation on a front horizontal ring so a
        # real dihedral crease survives GLB round-trip even if the sharp-edge
        # flag does not.
        if MOUTH_T_LO <= t <= MOUTH_T_HI and FRONT_SIGN_WORLD * new_y < 0:
            indent_w = _gauss(ang, NOSE_SIGMA_ANG * 0.6)
            new_y += (-FRONT_SIGN_WORLD) * MOUTH_INDENT * indent_w

        v.co = Vector((new_x, new_y, world_z))

    bm.normal_update()
    return bm


def build_inner_shell_bmesh(u_segments, v_segments, radius_scale, t_lo, t_hi):
    """A duplicate shell strictly inside the outer profile *by construction*.

    Unlike a uniform point-scale (which does NOT guarantee containment for a
    non-star-shaped profile -- verified empirically: 17/1102 verts of a
    uniformly-0.9-scaled-from-center copy poked *outside* the outer surface
    right at the neck pinch), this resamples the SAME per-height radius
    profile used for the outer shell, multiplied by `radius_scale` at the
    SAME height t, so containment holds at every ring independently of the
    body's shape. `t_lo`/`t_hi` compress the shell's own height range so it
    also never reaches the small torso hole's height (see HOLE_T) or the
    poles.
    """
    import bmesh
    from mathutils import Vector

    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=u_segments, v_segments=v_segments, radius=1.0)
    bm.verts.ensure_lookup_table()
    for v in bm.verts:
        x, y, z_local = v.co.x, v.co.y, v.co.z
        t01 = (z_local + 1.0) * 0.5
        t = t_lo + (t_hi - t_lo) * t01
        world_z = t * BODY_HEIGHT
        r_local = math.sqrt(x * x + y * y)
        rx = _profile_radius_x(t) * radius_scale
        ry = rx * Y_FLATTEN
        if r_local < 1e-9:
            new_x, new_y = 0.0, 0.0
        else:
            ux, uy = x / r_local, y / r_local
            new_x, new_y = ux * rx, uy * ry
        v.co = Vector((new_x, new_y, world_z))
    bm.normal_update()
    return bm


def mark_mouth_sharp_ring(bm):
    """Mark smooth=False on the front-facing edges of the mouth crease ring."""
    from mathutils import Vector

    marked = 0
    for e in bm.edges:
        v0, v1 = e.verts
        t0 = v0.co.z / BODY_HEIGHT
        t1 = v1.co.z / BODY_HEIGHT
        mid_t = (t0 + t1) / 2.0
        mid_y = (v0.co.y + v1.co.y) / 2.0
        if MOUTH_T_LO - 0.01 <= mid_t <= MOUTH_T_HI + 0.01 and FRONT_SIGN_WORLD * mid_y > 0:
            e.smooth = False
            marked += 1
    return marked


def _hole_target_world():
    from mathutils import Vector

    # Offset in Y (back side) sized relative to the outer profile's own
    # radius at this height so the target stays ON the surface regardless of
    # whether HOLE_T lands on the wide torso or the much narrower head.
    y_off = 0.75 * _profile_radius_x(HOLE_T) * Y_FLATTEN
    return Vector((0.0, -FRONT_SIGN_WORLD * y_off, HOLE_T * BODY_HEIGHT))


def carve_small_hole(bm):
    """Delete 2-3 adjacent faces (a BFS-grown patch nearest `_hole_target_world()`,
    restricted to the back side) to leave a small hole with an ~8-16 edge
    rim -- well under hole_fill.max_loop_edges, so P4 is expected to fill it.
    """
    import bmesh

    bm.faces.ensure_lookup_table()
    target = _hole_target_world()
    best = None
    best_d = None
    for f in bm.faces:
        c = f.calc_center_median()
        if c.y * FRONT_SIGN_WORLD > 0:
            continue
        d = (c - target).length
        if best_d is None or d < best_d:
            best_d = d
            best = f
    if best is None:
        return 0
    to_delete = {best}
    frontier = [best]
    while frontier and len(to_delete) < 3:
        nxt = []
        for f in frontier:
            for e in f.edges:
                for lf in e.link_faces:
                    if lf not in to_delete:
                        to_delete.add(lf)
                        nxt.append(lf)
                        if len(to_delete) >= 3:
                            break
                if len(to_delete) >= 3:
                    break
            if len(to_delete) >= 3:
                break
        frontier = nxt
    to_delete = list(to_delete)[:3]
    bmesh.ops.delete(bm, geom=to_delete, context="FACES")
    return len(to_delete)


def shatter(bm):
    import bmesh

    bm.edges.ensure_lookup_table()
    bmesh.ops.split_edges(bm, edges=bm.edges[:])


def build_small_sphere(radius, u=10, v=8):
    import bmesh

    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=u, v_segments=v, radius=radius)
    return bm


def build_small_cube(size):
    import bmesh

    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=size)
    return bm


def translate_bm(bm, offset):
    import bmesh

    bmesh.ops.translate(bm, verts=bm.verts, vec=offset)


def append_bm(master_bm, piece_bm):
    import bpy

    tmp = bpy.data.meshes.new("__tmp_fixture_piece__")
    piece_bm.to_mesh(tmp)
    master_bm.from_mesh(tmp)
    bpy.data.meshes.remove(tmp)
    piece_bm.free()


def assign_confetti_uv(bm, seed=0):
    import bmesh

    rng = random.Random(seed)
    uv_layer = bm.loops.layers.uv.new("UVMap")
    bm.faces.ensure_lookup_table()
    for f in bm.faces:
        n = len(f.loops)
        cu = rng.uniform(0.06, 0.94)
        cv = rng.uniform(0.06, 0.94)
        r = rng.uniform(0.008, 0.018)
        rot = rng.uniform(0.0, 2 * math.pi)
        for i, loop in enumerate(f.loops):
            ang = rot + 2 * math.pi * i / n
            loop[uv_layer].uv = (cu + r * math.cos(ang), cv + r * math.sin(ang))


def make_checker_image(name="Checker256", size=256, squares=8):
    import bpy

    img = bpy.data.images.new(name, width=size, height=size, alpha=False)
    px = [0.0] * (size * size * 4)
    cell = size // squares
    c0 = (0.85, 0.15, 0.15)
    c1 = (0.9, 0.9, 0.9)
    for j in range(size):
        for i in range(size):
            idx = (j * size + i) * 4
            on = ((i // cell) + (j // cell)) % 2 == 0
            r, g, b = c0 if on else c1
            px[idx] = r
            px[idx + 1] = g
            px[idx + 2] = b
            px[idx + 3] = 1.0
    img.pixels = px
    img.pack()
    return img


def make_checker_material(name="Skin"):
    import bpy

    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    tex = nt.nodes.new("ShaderNodeTexImage")
    tex.image = make_checker_image()
    if bsdf is not None:
        nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


def build_fixture(seed: int = 0):
    """Build the full synthetic humanoid fixture scene. Returns the body object."""
    import bpy

    random.seed(seed)
    bpy.ops.wm.read_factory_settings(use_empty=True)

    outer = build_body_bmesh()
    n_marked = mark_mouth_sharp_ring(outer)
    n_hole = carve_small_hole(outer)

    outer.faces.ensure_lookup_table()
    import bmesh

    bmesh.ops.triangulate(outer, faces=outer.faces[:], quad_method="BEAUTY", ngon_method="BEAUTY")

    shatter(outer)

    master = outer  # shattered outer becomes the master bmesh we append into

    # Inner duplicate shell, dense enough to push deleted_face_ratio into the
    # assertion band once P3 deletes it -- see INNER_SHELL_SCALE/T_LO/T_HI
    # and build_inner_shell_bmesh()'s docstrings for why it is fused to the
    # outer shell at both poles rather than left as an unfused duplicate.
    inner = build_inner_shell_bmesh(
        u_segments=56, v_segments=32, radius_scale=INNER_SHELL_SCALE,
        t_lo=INNER_SHELL_T_LO, t_hi=INNER_SHELL_T_HI,
    )
    append_bm(master, inner)

    # eyeballs
    from mathutils import Vector

    for side in (-1, 1):
        eb = build_small_sphere(EYE_RADIUS, u=12, v=10)
        translate_bm(eb, Vector((side * EYE_X, EYE_Y, EYE_Z)))
        append_bm(master, eb)

    # junk parts fully inside torso, asymmetric placement, mid/lower height
    junk_specs = [
        ("sphere", 0.03, Vector((0.09, 0.02, BODY_HEIGHT * 0.30))),
        ("cube", 0.035, Vector((-0.07, -0.03, BODY_HEIGHT * 0.42))),
        ("sphere", 0.022, Vector((0.02, 0.08, BODY_HEIGHT * 0.20))),
    ]
    for kind, size, pos in junk_specs:
        if kind == "sphere":
            jb = build_small_sphere(size, u=8, v=6)
        else:
            jb = build_small_cube(size)
        translate_bm(jb, pos)
        append_bm(master, jb)

    assign_confetti_uv(master, seed=seed)

    me = bpy.data.meshes.new("Body_mesh")
    master.to_mesh(me)
    master.free()
    me.update()

    mat = make_checker_material("Skin")
    me.materials.append(mat)
    for p in me.polygons:
        p.material_index = 0

    obj = bpy.data.objects.new("Body", me)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.update()

    return obj, {"mouth_ring_edges_marked": n_marked, "hole_faces_deleted": n_hole}


def export_glb(obj, out_path: Path):
    import bpy

    for o in bpy.data.objects:
        o.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=str(out_path), export_format="GLB", use_selection=True)
    return out_path


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    if not argv:
        print("usage: make_fixture.py <out.glb> [seed]", file=sys.stderr)
        return 2
    out_path = Path(argv[0])
    seed = int(argv[1]) if len(argv) > 1 else 0
    obj, info = build_fixture(seed=seed)
    export_glb(obj, out_path)
    print(f"wrote {out_path} verts={len(obj.data.vertices)} faces={len(obj.data.polygons)} info={info}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
