"""Shared geometry helpers used by multiple phases.

Kept bpy-optional wherever possible so the pure-python pieces (fibonacci_sphere,
DisjointSet, absorb_small_regions) can be unit-tested without Blender. Anything
that needs bmesh/bpy imports those modules lazily inside the function body.
"""

from __future__ import annotations

import math
from collections import Counter


def fibonacci_sphere(n: int) -> list[tuple[float, float, float]]:
    """Return n unit direction vectors, evenly distributed over the sphere.

    Pure python (no bpy/mathutils dependency) so it can be reused for both
    part-level and per-face visibility ray casting (hidden_geometry.part_rays
    / face_rays in the config).
    """
    if n <= 0:
        return []
    points: list[tuple[float, float, float]] = []
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(n):
        y = 1.0 - (i / float(n - 1)) * 2.0 if n > 1 else 0.0
        radius_at_y = math.sqrt(max(0.0, 1.0 - y * y))
        theta = golden_angle * i
        x = math.cos(theta) * radius_at_y
        z = math.sin(theta) * radius_at_y
        points.append((x, y, z))
    return points


_HEAD_BAND_N_SLICES = 60
_HEAD_BAND_BASELINE_SLICES = 6  # first N non-empty top slices used as the "how wide is the head" reference
_HEAD_BAND_RATIO = 2.5          # explosion trigger: slice width > RATIO * baseline
_HEAD_BAND_GLOBAL_FRAC = 0.15   # ...AND slice width > this fraction of the whole-body x-width
_HEAD_BAND_MIN_POINTS = 20


def head_z_band(xz_points: list[tuple[float, float]]) -> tuple[float, float] | None:
    """Find the head's [z_lo, z_hi] band from a scatter of world-space (x, z) points.

    Pure python (no bpy/mathutils import) so it is unit-testable without
    Blender -- see tests/test_geom.py. Callers (p2/p5/p6) collect the (x, z)
    pairs from a body mesh's world-space vertices (subsampled for speed,
    see `body_xz_points`).

    Method
    ------
    1. Slice z into `_HEAD_BAND_N_SLICES` (=60) buckets from zmax down to
       zmin; per-slice x-width = max(x) - min(x) of points falling in it.
    2. Baseline = median width of the first `_HEAD_BAND_BASELINE_SLICES`
       (=6) *non-empty* slices from the top. This is deliberately a
       fixed-count window taken from the scalp/hair region, not a fixed
       fraction of height, so it stays meaningful across differently
       proportioned bodies.
    3. Scan slices top-down; the head band ends at the first slice whose
       width "explodes" past the baseline: width > max(RATIO * baseline,
       GLOBAL_FRAC * global_width). Both conditions matter: the ratio test
       catches a narrow head suddenly widening into shoulders; the
       global-width floor stops a tiny numeric baseline (e.g. a near-zero
       hair-tip width) from making the ratio test fire on ordinary noise.
       The band is everything above that slice (top of bbox down to the
       exploding slice's upper edge).
    4. Returns None if there are too few points, the bbox is degenerate
       (zero height/width), fewer than 3 non-empty baseline slices exist,
       or the width profile never explodes (e.g. an isolated bust/head mesh
       with no shoulders to detect against) -- callers must fall back to a
       documented default rule in that case, never silently proceed.

    Calibration (avatar_003_body.glb, 17413 verts, a ~1-unit-tall T-pose
    body with long hair -- see $SCRATCH/profile_avatar003.py output used to
    tune this):

        zmax=0.49883 zmin=-0.49944 height=0.99827 global_width=0.99303

        slice  zfrac_from_top  width    (zfrac_from_bottom)
          0        0.000       0.0687     1.000   <- hair tip
          1        0.017       0.0854     0.983
          2        0.033       0.0880     0.967
          3        0.050       0.0912     0.950
          4        0.067       0.0931     0.933
          5        0.083       0.1059     0.917
          6        0.100       0.1206     0.900
          7        0.117       0.1371     0.883
          8        0.133       0.2377     0.867   <- explosion slice
          9        0.150       0.2935     0.850
         10        0.167       0.5632     0.833
         ...
         14        0.233       0.9930     0.767   <- T-pose arm peak

    baseline = median(widths[0:6]) = median(0.0687,0.0854,0.0880,0.0912,
    0.0931,0.1059) = 0.0896. threshold = max(2.5*0.0896, 0.15*0.99303) =
    max(0.2240, 0.1490) = 0.2240. Slice 7 (0.1371) stays under threshold
    (moderate hair widening tolerated); slice 8 (0.2377) exceeds it ->
    explosion_idx=8 -> z_lo = zmax - (8/60)*height = 0.3657, i.e. the band
    is z in [0.3657, 0.49883], zfrac_from_bottom in [0.8667, 1.0]. This
    comfortably contains the real eye/mouth region (zfrac ~0.85-0.95) while
    excluding a chest pendant crease at zfrac 0.82 and mirrored necklace
    beads at zfrac 0.55 (the bug this function exists to fix -- see
    p2_weld_split._classify_parts and p5_mouth._find_lip_line).
    """
    if not xz_points or len(xz_points) < _HEAD_BAND_MIN_POINTS:
        return None

    xs = [p[0] for p in xz_points]
    zs = [p[1] for p in xz_points]
    zmax, zmin = max(zs), min(zs)
    height = zmax - zmin
    global_width = max(xs) - min(xs)
    if height <= 1e-9 or global_width <= 1e-9:
        return None

    n = _HEAD_BAND_N_SLICES
    slice_xs: list[list[float]] = [[] for _ in range(n)]
    for x, z in xz_points:
        frac_from_top = (zmax - z) / height
        idx = min(n - 1, max(0, int(frac_from_top * n)))
        slice_xs[idx].append(x)

    widths: list[float | None] = [
        (max(b) - min(b)) if b else None for b in slice_xs
    ]

    non_empty_top = [w for w in widths if w is not None][:_HEAD_BAND_BASELINE_SLICES]
    if len(non_empty_top) < 3:
        return None
    sorted_top = sorted(non_empty_top)
    baseline = sorted_top[len(sorted_top) // 2]
    baseline = max(baseline, 1e-9)  # a zero baseline must not veto the ratio test

    threshold = max(_HEAD_BAND_RATIO * baseline, _HEAD_BAND_GLOBAL_FRAC * global_width)

    explosion_idx = None
    for i, w in enumerate(widths):
        if w is not None and w > threshold:
            explosion_idx = i
            break

    if explosion_idx is None or explosion_idx == 0:
        # Never exploded (no shoulders found in the profile -- e.g. an
        # isolated head/bust mesh) or exploded on the very first slice
        # (degenerate) -- no sane band to report.
        return None

    z_lo = zmax - (explosion_idx / n) * height
    return (z_lo, zmax)


class DisjointSet:
    """Pure-python union-find over integer indices [0, n)."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

    def groups(self) -> dict[int, list[int]]:
        """Return {root: [members]} -- one entry per connected component."""
        out: dict[int, list[int]] = {}
        for i in range(len(self.parent)):
            out.setdefault(self.find(i), []).append(i)
        return out


# ---------------------------------------------------------------------------
# bmesh-dependent helpers (bpy/bmesh imported lazily)
# ---------------------------------------------------------------------------


def body_xz_points(obj, max_samples: int = 4000) -> list[tuple[float, float]]:
    """World-space (x, z) samples of `obj`'s vertices, subsampled for speed.

    Shared by p2 (eye-pair head-z-band gating), p5 (lip-line head z-band),
    and p6 (post-carve eyeball sanity check) so all three phases compute
    `head_z_band` from the SAME sampling of the same body mesh. Uses
    `obj.data.vertices` directly (not bmesh) since only positions are
    needed; calls `view_layer.update()` first per the matrix_world staleness
    pitfall (ARCHITECTURE.md / CLAUDE.md rule 4).
    """
    import bpy  # noqa: F401 (ensures Blender context is available before use)

    bpy.context.view_layer.update()
    mat = obj.matrix_world
    verts = obj.data.vertices
    n = len(verts)
    if n == 0:
        return []
    stride = max(1, n // max_samples)
    pts = []
    for i in range(0, n, stride):
        co = mat @ verts[i].co
        pts.append((co.x, co.z))
    return pts


def eyes_world_centroid(ctx):
    """World-space centroid of the classified eyeballs, or None.

    Shared anatomical anchor: front_sign_from_eyes uses its Y, and the
    lip-line search uses its Z (the mouth sits a predictable fraction of the
    eye-to-chin distance below the eyes).
    """
    import bpy
    from mathutils import Vector

    names = getattr(ctx, "names", {})
    eye_names = [names.get("eye_l"), names.get("eye_r")]
    if not all(n and n in bpy.data.objects for n in eye_names):
        return None
    bpy.context.view_layer.update()
    total = Vector((0.0, 0.0, 0.0))
    count = 0
    for n in eye_names:
        obj = bpy.data.objects[n]
        mw = obj.matrix_world
        for v in obj.data.vertices:
            total += mw @ v.co
            count += 1
    if count == 0:
        return None
    return total / count


def front_sign_from_eyes(ctx, body_bbox: dict) -> int | None:
    """Face side (+1/-1 along Y) from the classified eyeballs, or None.

    Eyes sit on the front of the face by definition, so when p2 classified an
    eye pair this beats any protrusion heuristic (long hair out-protrudes the
    nose on real characters). Uses world-space vertex centroids of both eye
    objects vs the body bbox center.
    """
    centroid = eyes_world_centroid(ctx)
    if centroid is None:
        return None
    center_y = (body_bbox["min"].y + body_bbox["max"].y) / 2.0
    return 1 if centroid.y >= center_y else -1


def find_lip_line(
    body_obj,
    bbox: dict,
    notes: list,
    known_front_sign: int | None = None,
    eye_z: float | None = None,
):
    """Detect the lip line: boundary/sharp-crease edges in the front mouth region.

    Shared by p5_mouth (lip-line search for the closed-lips mouth-bag cutter)
    and p3_hidden_geo (pre-cleanup mouth-region face protection, so real
    mouth-bag/teeth interior geometry on the input mesh survives P3's hidden-
    geometry deletion pass) -- moved here from p5_mouth.py so both phases can
    import one implementation instead of duplicating the heuristic.

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
        head_band = head_z_band(body_xz_points(body_obj))
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

        # Eye-anchored mouth window: anthropometrically the mouth fissure sits
        # roughly 45-85% of the eye-to-chin distance below the eyes. Without
        # this, the nose-base crease wins over the true lip crease (avatar_003:
        # detected z=0.414 = nose base, real mouth ~0.388; eyes z=0.431,
        # chin=band_lo=0.366 -> window [0.376, 0.402]).
        if eye_z is not None and eye_z > head_lo:
            eye_to_chin = eye_z - head_lo
            window_hi = eye_z - 0.45 * eye_to_chin
            window_lo = eye_z - 0.85 * eye_to_chin
            new_lo = max(head_lo, window_lo)
            new_hi = min(head_hi, window_hi)
            if new_hi > new_lo:
                head_lo, head_hi = new_lo, new_hi
                notes.append(
                    f"eye-anchored mouth z-window applied: [{head_lo:.4f},{head_hi:.4f}] "
                    f"(eye_z={eye_z:.4f}, 45-85% of eye-to-chin below the eyes)"
                )
            else:
                notes.append(
                    "eye-anchored mouth z-window degenerate; keeping head band as-is"
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

        # front-axis: prefer the caller-provided sign (derived from classified
        # eyeballs, which sit on the face side by definition -- see
        # front_sign_from_eyes). The nose-protrusion fallback below FAILS on
        # long-haired characters: avatar_003's back-of-head hair bulge
        # (0.0527) out-protrudes the nose (0.0408), which sent the mouth
        # carve into the back of the head. Only trust protrusion when there
        # is no eye-derived sign.
        if known_front_sign in (1, -1):
            front_sign = known_front_sign
            notes.append(
                f"front-axis (mouth): using caller-provided front_sign={front_sign} "
                "(derived from classified eyeballs; protrusion heuristic skipped)"
            )
        else:
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
                "asymmetry) -> front_sign="
                f"{front_sign} (CAUTION: hair bulges can beat the nose; "
                "eye-derived sign is preferred when available)"
            )

        def _collect(crease_deg: float) -> list:
            found = []
            threshold = math.radians(crease_deg)
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
                    is_crease = e.calc_face_angle() > threshold
                if is_boundary or is_sharp or is_crease:
                    found.append(e)
            return found

        # Detection ladder: crisp creases first, then soft ones. Real sculpts
        # often model closed lips as a shallow (<35 deg) crease.
        candidates = _collect(35.0)
        if not candidates:
            candidates = _collect(22.0)
            if candidates:
                notes.append(
                    "lip detection: no >35deg creases; found candidates at the "
                    "relaxed 22deg threshold"
                )

        def _synthesize():
            """Anthropometric fallback: eye-anchored window midpoint + surface probe.

            Only available when eye_z was provided (the window is then narrow
            and reliable). Honest degradation for smooth-lipped sculpts with
            no detectable crease cluster.
            """
            if eye_z is None:
                return None
            fissure_z = (head_lo + head_hi) / 2.0
            central_front_ys = [
                (mat @ v.co).y
                for v in bm.verts
                if abs((mat @ v.co).z - fissure_z) <= 0.02 * height
                and _central(mat @ v.co)
                and front_sign * (mat @ v.co).y > 0
            ]
            if not central_front_ys:
                return None
            lip_surface_y = (
                max(central_front_ys) if front_sign > 0 else min(central_front_ys)
            )
            notes.append(
                "lip detection: no usable crease cluster; SYNTHESIZED lip line "
                f"from eye anatomy: fissure_z={fissure_z:.4f} (window midpoint), "
                f"mouth_x={x_center:.4f}, lip_surface_y={lip_surface_y:.4f} "
                "(front-surface probe)"
            )
            return (fissure_z, x_center, front_sign, lip_surface_y)

        if not candidates:
            synthesized = _synthesize()
            if synthesized is not None:
                return synthesized
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
            synthesized = _synthesize()
            if synthesized is not None:
                return synthesized
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


def bm_connected_components(bm) -> list[set[int]]:
    """Vertex-index connected components of a bmesh, via edge adjacency."""
    bm.verts.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    ds = DisjointSet(len(bm.verts))
    for e in bm.edges:
        ds.union(e.verts[0].index, e.verts[1].index)
    return [set(members) for members in ds.groups().values()]


def uv_island_count(me) -> int | None:
    """Union-find over faces whose shared edges are welded in UV space.

    Ported from docs/uv-atlas-references/snippets.md section 2. Returns None
    if the mesh has no active UV layer.
    """
    import bmesh

    bm = bmesh.new()
    bm.from_mesh(me)
    try:
        uvl = bm.loops.layers.uv.active
        if not uvl:
            return None
        bm.faces.ensure_lookup_table()
        ds = DisjointSet(len(bm.faces))
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
        return len(ds.groups())
    finally:
        bm.free()


def boundary_loop_histogram(bm) -> dict[int, int]:
    """Union-find over boundary edges -> {loop_size: count}.

    Boundary edges sharing a vertex are unioned into the same loop; the
    returned histogram maps loop edge-count to how many loops have that size.
    Small sizes (1-15ish) are slits worth filling; a few large ones are
    intentional openings (mouth, eye sockets) that must not be auto-filled.
    """
    bm.edges.ensure_lookup_table()
    boundary = [e for e in bm.edges if e.is_boundary]
    index_of = {e.index: i for i, e in enumerate(boundary)}
    ds = DisjointSet(len(boundary))
    for e in boundary:
        i = index_of[e.index]
        for v in e.verts:
            for e2 in v.link_edges:
                if e2 is not e and e2.is_boundary and e2.index in index_of:
                    ds.union(i, index_of[e2.index])
    sizes = Counter(len(members) for members in ds.groups().values())
    return dict(sorted(sizes.items()))


# ---------------------------------------------------------------------------
# Pure-data segmentation helpers (no bpy/bmesh dependency at all)
# ---------------------------------------------------------------------------


def _label_regions(labels: list[int], adjacency: list[list[int]]) -> list[list[int]]:
    """Connected components of same-label faces, per the adjacency graph."""
    n = len(labels)
    seen = [False] * n
    regions: list[list[int]] = []
    for start in range(n):
        if seen[start]:
            continue
        lab = labels[start]
        seen[start] = True
        stack = [start]
        comp = [start]
        while stack:
            cur = stack.pop()
            for nb in adjacency[cur]:
                if not seen[nb] and labels[nb] == lab:
                    seen[nb] = True
                    comp.append(nb)
                    stack.append(nb)
        regions.append(comp)
    return regions


def absorb_small_regions(
    labels: list[int], adjacency: list[list[int]], min_region: int
) -> list[int]:
    """Relabel small connected regions into their most-common bordering region.

    Pure python, taking plain data (face_labels + adjacency lists) so it's
    unit-testable without bpy. This is connected-component absorption, NOT a
    per-face majority vote: a "region" is a whole connected group of
    same-label faces, and the entire region is relabeled at once to whichever
    label borders it most (by shared-adjacency-edge count), iterating until
    no region smaller than min_region remains (or nothing changes).

    Ported from docs/uv-atlas-references/snippets.md section 9's regions()
    absorption loop, generalized from bmesh faces to plain face indices.
    """
    labels = list(labels)
    max_iterations = len(labels) + 1
    for _ in range(max_iterations):
        regions = _label_regions(labels, adjacency)
        small = [r for r in regions if len(r) < min_region]
        if not small:
            break
        changed = False
        for comp in small:
            cset = set(comp)
            border: Counter = Counter()
            for f in comp:
                for nb in adjacency[f]:
                    if nb not in cset:
                        border[labels[nb]] += 1
            if not border:
                continue
            new_label = border.most_common(1)[0][0]
            if new_label != labels[comp[0]]:
                changed = True
            for f in comp:
                labels[f] = new_label
        if not changed:
            break
    return labels
