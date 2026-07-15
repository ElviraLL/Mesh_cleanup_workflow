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
