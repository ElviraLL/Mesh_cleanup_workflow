---
name: blender-mesh-cleanup-uv-atlas
description: Clean up messy mesh topology, consolidate multi-part models into one object, re-unwrap UVs into a shared map with few large islands, and rebake textures onto the new layout — all through Blender MCP. Use this skill whenever the user asks (via Blender MCP) to clean up a mesh, fix topology, merge/join mesh parts, reduce UV seams or islands, rebake or re-unwrap UVs, create a texture atlas, fix non-manifold geometry, or prepare a scanned/photogrammetry/AI-generated character or prop for export. Trigger even if they only mention one part (e.g. "too many seams" or "join everything into one mesh") — the phases below are designed to be used independently or together.
---

# Blender Mesh Cleanup + Shared UV Atlas (via Blender MCP)

A battle-tested pipeline for taking a messy multi-object model (typically a scan,
photogrammetry, or AI-generated mesh) and producing: one consolidated mesh, clean
topology, a single shared UV map with few chunky islands, and textures rebaked onto
the new layout.

All geometry work runs as Python inside Blender via `blender:execute_blender_code`.
Work in small chunks — one logical operation per call — so failures are isolated and
progress is visible. Full copy-paste code for every phase lives in
`references/snippets.md`; read it when you reach each phase rather than improvising,
since several snippets encode non-obvious fixes.

## Core principles

1. **Diagnose before touching anything.** Most "UV problems" are actually topology
   problems. Measure first; the numbers dictate the plan.
2. **Never destructive.** Duplicate every mesh into a hidden `_backup_pre_cleanup`
   collection before the first edit. The backups also serve later as the *bake
   source* with the original UVs — this is essential, not just insurance.
3. **Verify after every phase** with counts (non-manifold edges, boundary loops, UV
   islands), not vibes. Re-run the same diagnostics so improvement is quantified.
4. **A fresh unwrap breaks existing texture mapping.** If any material has image
   textures, plan the atlas rebake (Phase 5) before you re-unwrap, and tell the user.

## Phase 0 — Diagnose

Run `blender:get_scene_info`, then for every mesh collect: vert/face counts, parent,
modifiers, shape keys, vertex groups, UV layer names, materials, object scale.
These determine join safety (see Phase 3 hazards). Then run the topology report per
mesh: non-manifold edges, boundary edges, loose verts/edges, zero-area faces,
duplicate verts (probe with `remove_doubles` on a throwaway bmesh), and UV island
count (union-find over faces welded in UV space — snippet provided; you cannot get
this number from any built-in property).

Red flags and what they mean:
- **UV islands ≈ face count** → per-triangle/lightmap-style unwrap; needs full re-unwrap.
- **Many boundary edges but one connected shell** → slits and holes, not separate parts.
- **Edges with 3+ linked faces** → interior flaps/fins; these break unwrappers.
- **verts:faces ≈ 1:2 (all triangles)** → scan-like mesh; expect noisy normals and
  plan for the segmentation unwrap in Phase 4, not plain Smart UV Project.

## Phase 1 — Backup

Duplicate object + mesh data for every part into `_backup_pre_cleanup`, hide the
collection from viewport and render. Do this before *any* edit.

## Phase 2 — Topology cleanup (order matters)

Run a **weld-distance sweep** first (try 1e-4 … 5e-3 on throwaway bmesh copies,
report verts merged at each) and pick a threshold that stitches gaps without eating
detail — roughly 0.1% of the object's bounding size. If a candidate distance merges
more than a few percent of all verts, it is too aggressive.

Then, in this order (each step exposes work for the next):
1. `remove_doubles` at the chosen distance; `dissolve_degenerate`.
2. Delete wire edges (no linked faces) and loose verts.
3. **Iteratively** delete flap faces (≥2 edges shared by 3+ faces) until converged —
   deleting flaps exposes new ones, so loop with a max-iteration guard.
4. **Iteratively** delete fin faces (≥1 over-shared edge AND ≥2 free edges).
5. Fill holes — but first build a **boundary-loop size histogram** (union-find over
   boundary edges). Small loops (≲ 15–20 edges) are damage: fill them with
   `holes_fill(sides=N)`. Large loops are usually intentional openings (eye sockets,
   mouths, necks) — leave them open. When unsure, show the histogram to the user.
6. Triangulate any fill n-gons, `recalc_face_normals`.
7. If the mesh is spiky (scan noise), do a **targeted despike**: collect verts on
   interior edges with dihedral > 120°, lightly Laplacian-smooth only those
   (lerp factor ~0.3, 1–2 passes). Never smooth the whole mesh.

Some non-manifold edges are structural (e.g. where a mouth-interior shell attaches).
Getting to zero is not the goal; removing junk that fragments the unwrap is.

## Phase 3 — Join into one object

Hazard checklist before `object.join()` — inspect, warn, and resolve first:
- **Shape keys** on any part → join destroys them on non-active objects; stop and ask.
- **Un-applied modifiers** → decide with the user: apply or carry over.
- **Non-uniform / mismatched scales** → apply transforms first or UVs and welds skew.
- **UV layer names must match** across all parts (rename to a common name, e.g.
  `UVMap`) or join produces multiple UV layers.

Select all parts, make the target (e.g. `Body`) active, join. Materials become slots
automatically. Verify after: material slot list, single UV layer, parent preserved.

## Phase 4 — UV re-unwrap (the key insight)

**Do not reach for Smart UV Project on noisy triangulated meshes.** It splits charts
by pairwise face-normal angle; on scan meshes with noisy normals it produces
confetti (1000+ islands) at *any* angle limit. Diagnose first: histogram the
dihedral angles of interior edges. If hundreds of edges exceed ~89°, angle-based
projection will fail.

Use **dominant-normal-axis segmentation** instead:
1. Label each target face by which of the 6 axis directions (±X ±Y ±Z) its normal
   most agrees with. This is robust to noise — a normal must swing >45° to change label.
2. Clean speckles by **connected-component absorption**: find same-label connected
   regions; any region smaller than `MIN_REGION` faces (≈40 is a good start) gets
   relabeled to its most-common bordering region. Iterate until stable. Do NOT use
   per-face majority-vote smoothing — it oscillates and never converges.
3. Mark seams on edges between different final regions (clear all seams first).
4. Select only the faces being re-unwrapped, `uv.unwrap(method='ANGLE_BASED')` —
   the seam-based unwrap, not smart_project.
5. Select everything and `uv.pack_islands` to place all islands (including parts
   that kept their original unwrap) into shared 0–1 space.

If some parts already have clean unwraps (eyes, teeth, props), **keep them**: only
re-unwrap the fragmented faces (select by material index), then pack all together.
Remember packing moves/scales *every* island — so even untouched parts need the
Phase 5 rebake if they have textures.

Tune island count with `MIN_REGION` (bigger → fewer, larger islands). Report final
island count with the same union-find used in Phase 0 so improvement is measurable.

## Phase 5 — Texture atlas rebake

Only needed if materials carry image textures (check every material's node tree for
`TEX_IMAGE` nodes and what they feed — Base Color? Roughness via Separate Color?
Normal?). Bake one atlas per *channel in use*, not just base color.

Selected-to-active bake from the backups (old geometry + old UVs) onto the new mesh:
1. Give backup objects **material copies** first (`mat.copy()`, suffix `_orig`) so
   rewiring the live materials later doesn't corrupt the backups.
2. Create atlas images (user-chosen resolution; sRGB for color, Non-Color for
   roughness/normal data).
3. Add a `BakeTarget` image node to **every** material on the new mesh, set it as
   the **active node** in each node tree — Cycles writes the bake there.
4. Cycles, low samples (4–8 is plenty for color passes), `use_selected_to_active`,
   `max_ray_distance` slightly larger than how far cleanup moved the surface
   (~0.02–0.03 on a 1-unit mesh), margin ≥16 px, and for DIFFUSE disable
   direct/indirect so only color bakes.
5. Select backups (unhide them), active = new mesh, `object.bake(type='DIFFUSE')`;
   swap the target image and repeat with `type='ROUGHNESS'` (or NORMAL) as needed.
6. `image.pack()` each atlas into the .blend, rewire each material (atlas → Base
   Color; atlas → Roughness with Non-Color), delete the now-unused old texture
   nodes from the live materials only, re-hide backups.

Ask the user for resolution and scope (full atlas vs. one material) before baking.

## Phase 6 — Verify and report

- Recount UV islands and non-manifold edges; report before → after numbers.
- `blender:get_viewport_screenshot` for a visual check that textures still map.
- Remind the user: backups live in `_backup_pre_cleanup` (delete when confident),
  render engine may have been switched to Cycles for baking, and atlases are packed
  in the .blend (save the file, or save images externally).

## Failure modes seen in the wild

- MCP call times out → Blender not running / addon not connected; walk the user
  through the BlenderMCP sidebar panel and retry, don't hammer the tool.
- `holes_fill` leaves boundaries behind after flap deletion → run another fill pass
  after each deletion round; boundary count can *rise* mid-cleanup and that's fine.
- Unwrap warning "failed to solve N islands" → usually one degenerate region;
  packing still works. Note it to the user; only chase it if visible stretching.
- Majority-vote label smoothing oscillates forever → use region absorption instead.
