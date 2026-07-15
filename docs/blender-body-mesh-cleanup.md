---
name: body-mesh-cleanup
description: Clean up AI-generated or messy character/body meshes in Blender via the Blender MCP tools (execute_blender_code, get_scene_info, get_viewport_screenshot). Use this skill whenever the user wants to repair, clean, or restructure a humanoid/character mesh — including welding fragmented "triangle soup" geometry, removing hidden inner layers or duplicate shells, separating eyeballs into their own objects, deleting or rebuilding mouth bags (teeth/tongue), carving eye or mouth openings, fixing mesh/eyeball penetration, or making a mesh watertight. Also consult it for ANY multi-step Blender-MCP editing session, because it documents critical MCP session pitfalls (edit-mode failures, user undo, stale references) that apply beyond mesh cleanup.
---

# Body Mesh Cleanup (Blender MCP)

Battle-tested workflow for cleaning AI-generated character meshes (Hunyuan3D, Rodin,
photogrammetry, game rips). These meshes typically arrive as: tens of thousands of
disconnected fragments, a hidden inner layer duplicating most of the surface, eyeballs
and mouth-bag geometry fused into one object, and non-manifold soup.

**Golden rules: diagnose numerically before editing, snapshot before every destructive
phase, verify visually (screenshot) after every phase, and never trust cached state
across tool calls.**

## Phase 0 — MCP session hygiene (read this first, it will save you)

These failures all actually happened. Every `execute_blender_code` script should be
written defensively:

1. **Force Object Mode at the top of EVERY script.** The user inspects the mesh in Edit
   Mode between your calls. In Edit Mode: `bm.to_mesh()` raises, `transform_apply` fails
   poll, and worst of all `primitive_*_add` **injects the primitive into the edited
   mesh** as loose geometry (junk you'll have to hunt down later).
   ```python
   for o in bpy.data.objects:
       if o.mode != 'OBJECT':
           bpy.context.view_layer.objects.active = o
           bpy.ops.object.mode_set(mode='OBJECT')
   ```
2. **Never trust `bpy.context.active_object` after an operator.** Detect new objects by
   name-set difference:
   ```python
   before = set(o.name for o in bpy.data.objects)
   bpy.ops.mesh.primitive_uv_sphere_add(...)
   new_obj = bpy.data.objects[(set(o.name for o in bpy.data.objects) - before).pop()]
   ```
   If the set difference is empty, STOP and inspect — do not continue the script.
3. **Prefer the data API over operators.** `bmesh.ops.create_uvsphere` +
   `bpy.data.objects.new` + `collection.objects.link` cannot fail on context. Note:
   `create_uvsphere(calc_uvs=True)` UVs may not survive `to_mesh` — write UVs manually.
4. **`matrix_world` is stale within a call** after you change `location`/`scale`. Call
   `bpy.context.view_layer.update()` before reading world positions.
5. **The user can press Ctrl+Z between your calls.** An undo can silently roll back
   several of your phases (object positions, even resurrect deleted objects). At the
   start of each phase, re-verify state (object list, vert counts, key positions) — do
   not use numbers cached from earlier messages when precision matters. Re-measure.
6. **Never swap mesh datablocks to "restore" an object** (`body.data = snap.data.copy()`
   then removing the old mesh can invalidate/destroy the object). Instead rebuild:
   `new = snap.copy(); new.data = snap.data.copy(); link; rename`.
7. **Deleted objects orphan their meshes** (`users=0`) — recoverable until file save.
   Check `bpy.data.meshes` before declaring data lost.
8. **Snapshot before every boolean / large deletion:**
   ```python
   snap = obj.copy(); snap.data = obj.data.copy()
   snap.name = "Body_preX_SNAP"; bpy.context.collection.objects.link(snap)
   snap.hide_set(True); snap.hide_render = True
   ```
9. **Chunk heavy loops** (~100k faces per call for per-face ray casting) and persist
   progress in a face int attribute (`me.attributes.new("vis_flag", 'INT', 'FACE')`) so
   state survives across calls.
10. **Screenshots:** request `max_size=640` (full size can exceed the 1 MB tool-result
    limit). Set the view via `space.region_3d.view_location/.view_distance` +
    `bpy.ops.view3d.view_axis` inside a `temp_override`. Use
    `space.shading.type='MATERIAL'` to check textures, `show_xray=True` to check for
    inner layers. Screenshot-verify after every phase; numbers lie less than eyes, but
    eyes catch what you didn't think to measure.

## Phase 1 — Diagnose

Collect before touching anything: vert/face counts, materials, UV layers, modifiers,
connected components (sizes + bbox centers/dims), boundary-edge and non-manifold-edge
counts. Component bbox centers identify parts anatomically (symmetric pair near face
top = eyeballs; wide flat stack near mouth = teeth; component spanning the whole body =
outer skin; large component hidden under clothing = inner layer).

Signature of AI-generated soup: tens of thousands of components where 80%+ have ≤10
verts, and ~half of all edges are boundary edges → fragments share coincident verts.

## Phase 2 — Weld and split

1. `bmesh.ops.remove_doubles(bm, verts, dist=1e-6)` — expect thousands of fragments to
   collapse into a handful of real parts. Dry-run on a bmesh copy first and recount
   components before writing back.
2. `bpy.ops.mesh.separate(type='LOOSE')` → classify each object by size + center.
3. Rename/keep the eyeballs (material + UVs transfer automatically with separation).

## Phase 3 — Remove hidden geometry (the core trick)

Two levels, both needed:

**A. Whole-part visibility (loose objects).** Simple "ray from surface escapes to sky"
tests are confounded by layered hair cards. Decisive method: build ONE combined BVH,
then per part sample verts and cast from far outside along ~26 directions; the part is
visible only if the first hit belongs to it. **Then confirm visually**: hide the main
body, screenshot; show only the main body, screenshot — if the model looks complete
with only the main part, everything else is interior. Delete all internal parts
(keep eyeballs even though they test as internal — they sit behind cornea/lid surfaces).

**B. Fused inner layer (same component as the skin).** Welding fuses inner shells to
the outer skin — X-ray view shows double surfaces. Per-face pass:
- 48 Fibonacci-sphere directions; a face is visible if any ray from
  `center + dir*1e-5` escapes (try `+normal` first for early exit).
- Chunk ~100k faces/call, persist in a face int attribute.
- **Dilate the visible set by 2 adjacency rings before deleting** — protects concave
  areas (nostrils, ears, armpits) from getting holes punched in them.
- Delete interior faces, loose edges/verts, then sub-40-vert orphan islands.

Expect 50–70% of an AI mesh to be hidden junk.

**Hidden junk causes phantom obstacles later.** If clearance ray-casts (eye fitting,
socket building) report an impossible obstacle, run a LOCAL visibility pass in that
region before believing it — leftover interior flaps (kept by the dilation collar) sit
millimeters behind lids and choke every measurement.

## Phase 4 — Holes and watertightness

- `bpy.ops.mesh.fill_holes(sides=0)` in Edit Mode, plus `bmesh.ops.holes_fill`;
  `dissolve_degenerate` + gentle `remove_doubles(2e-5)` between passes helps.
- **Do not chase zero boundary edges.** Hair cards and clothing borders are open by
  design. Report boundary edges by region (z-bands) — "0 in the body/legs, the rest in
  hair/cloth" is the realistic success state.
- Never trim "flap" faces globally (faces with 2+ boundary edges) — that eats hair cards.

## Phase 5 — Mouth bag

One boolean does both jobs: **DIFFERENCE with an ellipsoid pushed through the lip line
splits the lips AND creates the cavity walls**, and cutter faces inherit the cutter's
material — assign a dark `Mouth_Interior` material to the cutter first.

- Find the lip line first: boundary edges or crease near the mouth region give the z of
  the fissure. Cutter ≈ scaled UV sphere, e.g. radii (0.020, 0.042, 0.0085) on a
  1-unit-tall body, centered so the front tip pokes just past the lips.
- Teeth: torus, delete back half in bmesh, cap with `holes_fill`, squash x ≈ 0.78.
  Tongue: squashed sphere.
- **Verify placement numerically**: measure the opening rim (edges where a cavity-material
  face meets a skin face), then check teeth front-y sits 1–2 mm behind the rim front, and
  arches overlap the opening z-range. AI eyeball/teeth prosthetics are routinely ~1 cm
  too deep — floating teeth read as "no lips".

## Phase 6 — Eyes and eye openings

- Closed-lid sculpts cover the eyeballs entirely; carve almond openings with boolean
  ellipsoids. **Carve at the fissure line ≈ 2 mm BELOW the eyeball equator** (closed
  lids meet below center), x centered on the eyeballs — measure eyeball centers fresh
  at carve time.
- If eyeballs are dented/dirty, replace them with clean UV spheres (data API!, pole
  rotated to face −Y) + a procedural iris texture (equirect image: v=1 pole = pupil;
  zones pupil/iris/limbal by polar angle; radial fibers = multi-frequency sin over
  azimuth with integer frequencies so the seam is invisible). Write equirect UVs
  per-loop with a per-face seam fix (if a face's u-range spans >0.5, add 1 to the small
  u's).
- **Placement = max-inscribed-sphere fit**: grid-search the center (±few mm) maximizing
  the minimum ray-cast clearance to the skin over sphere-vert directions; radius =
  min-clearance − margin. Rays through the opening return no hit = no constraint
  (eye may bulge into the opening — that's the visible part). If left/right fit wildly
  asymmetric → suspect hidden junk (see Phase 3).
- **Socket interior geometry (cup wrapping the eyeball) is genuinely hard.** Pitfalls
  hit in practice: cups placed by rays from a center get squeezed inside the eye by
  grazing rim hits; chords between coarse rings slice the cornea; dented eyeballs break
  radial assumptions. If attempted: use analytic spheres for the eyes, build the shell
  at sphere_radius + 0.6 mm (pure math, normals aligned with the eye), and make eye
  clearance a hard constraint that wins over skin clearance. **It is often the right
  call to skip the socket entirely**: opening + eyeball behind it + dark corners reads
  fine. Offer that trade-off early.

## Phase 7 — Penetration testing (do not trust overlap alone)

- `BVHTree.overlap()` pair counts include near-misses at millimeter clearances —
  **confirm with exact triangle–triangle tests** (each edge of tri A vs tri B via
  `intersect_ray_tri`, both directions, segment-length bounded) before declaring
  penetration.
- Vert-level check: `find_nearest`, outside iff `(p - loc).dot(normal) > 0`; exclude an
  x/z window around intentional openings (visible cornea through the hole is a false
  positive).
- Internal overlap behind lids is normal and desired; only surface poke-through beyond
  openings is a defect. Fix order: re-center on the opening → per-side back-off →
  shrink-to-fit → last-resort surgical vert pulls (hidden quadrants only; note dents
  break later radial math — prefer resizing).

## Phase 8 — Wrap up

- Report per-task status, keep hidden snapshots until the user saves a new .blend,
  remove unused material slots, and name deliverable objects (`Body`, `Eye_L/R`,
  `Teeth_Upper/Lower`, `Tongue`).
- Retopo: automated (Quadriflow/voxel remesh) destroys UVs → needs unwrap + texture
  bake, and never produces artist-grade face loops. State this trade-off and let the
  user choose; a cleaned triangulated mesh is a legitimate end state.
