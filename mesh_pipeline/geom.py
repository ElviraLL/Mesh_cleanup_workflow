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
