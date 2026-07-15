# Headless Blender Mesh-Cleanup Pipeline — Implementation Plan

## Context (read first)

This project converts a battle-tested **interactive** Blender MCP workflow into an
**automated headless backend service**. The workflow cleans up AI-generated humanoid
character meshes (Hunyuan3D / Rodin style): welds triangle soup, removes hidden interior
geometry, carves mouth/eye openings, places eyes/teeth/tongue, re-unwraps UVs into a
shared atlas, rebakes textures, and exports GLB.

**All domain knowledge lives in `docs/`. Read these three files before writing any code:**

1. `docs/blender-body-mesh-cleanup.md` — the mesh-cleanup phases (weld, hidden-geometry
   removal, mouth bag, eyes, penetration testing) plus hard-won bpy/bmesh pitfalls.
2. `docs/blender-uv-atlas-rebake.md` — topology cleanup order, join hazards,
   dominant-normal-axis UV segmentation, and the atlas rebake procedure.
3. `docs/uv-atlas-references/snippets.md` — complete copy-paste bpy code for the UV/bake
   phases. Prefer adapting these snippets over rewriting from scratch; they encode
   non-obvious fixes.

The original workflow relied on a human/agent looking at viewport screenshots between
steps. This pipeline replaces that with: (a) numeric assertions after every phase,
(b) headless QA renders, (c) a `needs_review` job state for anything assertions can't
decide. **Do not silently proceed past a failed assertion.**

### MCP → headless translation notes

The docs were written for an interactive MCP session. In headless mode
(`blender --background`):

- Pitfalls that DISAPPEAR: user switching to Edit Mode, user Ctrl+Z between calls, the
  ~4-minute tool timeout (bake samples can be raised), screenshot size limits.
- Pitfalls that REMAIN: stale object references after operators (always re-fetch by
  name), `matrix_world` staleness (call `view_layer.update()`), operators that fail
  context poll in background mode → **prefer the bmesh/data API over `bpy.ops` wherever
  the docs show both**. `pack_islands` and `object.bake` work fine in background on
  Blender 4.x.
- Viewport screenshots → replace with `bpy.ops.render.render` from placed cameras
  (see QA renders below).

## Target repo structure

```
mesh_pipeline/
  cli.py                 # entry: blender -b -P cli.py -- --input x.glb --config config/humanoid.yaml --job-dir out/job1
  config/
    humanoid.yaml        # all tunables with defaults listed below
    schema.py            # pydantic-style validation of config
  phases/
    p0_diagnose.py       # scene stats, component analysis, UV island count
    p1_backup.py         # snapshot originals into hidden collection (also the bake source later)
    p2_weld_split.py     # weld-distance sweep, remove_doubles, separate LOOSE, classify parts
    p3_hidden_geo.py     # part-level + per-face visibility ray-casting, delete interior
    p4_topology.py       # flaps/fins/holes/despike (order from docs)
    p5_mouth.py          # mouth-bag boolean, teeth, tongue        [feature-flagged]
    p6_eyes.py           # eye opening carve, eyeball fit/replace  [feature-flagged]
    p7_uv_atlas.py       # dominant-axis segmentation unwrap + pack
    p8_bake.py           # selected-to-active atlas rebake from backups
    p9_export.py         # naming, parenting, GLB export
  qa/
    assertions.py        # per-phase numeric assertions (below)
    render.py            # 6-view + x-ray QA renders
    vlm.py               # M3: Claude API visual QA (stub until M3)
  report.py              # accumulates report.json across phases
  runner/                # M1: batch regression runner
  service/               # M2: FastAPI + RQ worker
```

### Core execution contract

- Every phase implements `run(ctx: PipelineContext, cfg: Config) -> PhaseResult`.
- `ctx` holds a **name registry** for objects (e.g. `ctx.names["body"] = "Body"`).
  Never cache `bpy.types.Object` references across operator calls — re-fetch
  `bpy.data.objects[name]` every time.
- Before every destructive phase: `bpy.ops.wm.save_as_mainfile` a snapshot
  `snapshots/pre_pN.blend` into the job dir (replaces the in-scene `_SNAP` objects from
  the docs; enables resume-from-phase during review).
- On assertion failure: write the failure into `report.json`, save
  `snapshots/failed_pN.blend`, render QA views, set job status `needs_review`, stop.
- `report.json` accumulates per-phase before/after metrics so improvement is quantified.

## Config defaults (validated on a real project — keep these)

```yaml
input:
  up_axis: auto          # detect + normalize
weld:
  sweep: [1e-4, 5e-3]    # probe on throwaway bmesh copies
  auto_pick: first threshold before merge-rate exceeds 3% of verts
hidden_geometry:
  part_rays: 26          # part-level visibility directions
  face_rays: 48          # fibonacci-sphere per-face pass
  dilate_rings: 2        # protect concave areas (nostrils/ears)
hole_fill:
  max_loop_edges: 20
  protected_zones:       # NEVER auto-fill loops inside these (prevents capping eye sockets)
    - eye_regions        # sphere around each eyeball center, r = 1.5x eyeball radius
    - lip_region         # box around detected lip line
mouth:
  enabled: true          # feature flag — pipeline must run fine with this off
  cutter_radii: [0.020, 0.042, 0.0085]   # per 1-unit body height, scale accordingly
  teeth_recess_mm: [1, 2]                # assertion range behind lip rim
eyes:
  enabled: true          # feature flag
  opening_ratio: 0.72    # of original eye height, centered on pupil line
  fissure_offset_mm: -2  # carve 2mm BELOW eyeball equator
  replace_dirty_eyeballs: true           # clean UV spheres + procedural iris (see docs)
  build_socket: false    # socket interior is high-risk; default OFF (docs explain)
uv:
  min_region: 40         # dominant-axis segmentation absorption threshold
  keep_clean_islands: true               # don't re-unwrap eyes/teeth if already clean
bake:
  resolution: 2048
  channels: [diffuse]    # detect from material node trees; add roughness/normal if used
  samples: 32            # headless has no timeout; can afford more than MCP's 4-8
  max_ray_distance: 0.025
  margin_px: 16
export:
  format: glb
  names: {body: Body, eye_l: Eye_L, eye_r: Eye_R, teeth_u: Teeth_Upper, teeth_l: Teeth_Lower, tongue: Tongue}
  hierarchy: parent_under_body           # parented, NOT joined
qa:
  render_views: [front, back, left, right, top, three_quarter]
  xray_view: true
  resolution: 640
```

## Per-phase assertions (qa/assertions.py)

Failing any → `needs_review`, not hard failure:

| After | Assertion |
|---|---|
| P2 | connected components collapsed from thousands to < 20 |
| P3 | deleted-face ratio in 30–80% band (below = missed junk, above = over-deletion) |
| P4 | boundary edges near zero in body/leg z-bands (hair/cloth bands may stay open) |
| P5 | teeth front-y is 1–2 mm behind lip rim; tooth arches overlap opening z-range |
| P6 | left/right eye clearance asymmetry < 30% — if exceeded, auto-run ONE local per-face visibility pass in the eye region and re-measure (hidden junk is the usual cause), then re-assert |
| P7 | UV island count ≤ config target; zero island overlap after pack |
| P8 | bake target images non-empty; no all-black atlas |
| P9 | exact triangle–triangle intersection tests = zero penetration outside opening windows (BVH overlap alone is NOT sufficient — see docs) |

## Milestones

### M0 — single-machine CLI (bulk of the work)

Build everything above except `runner/`, `service/`, `qa/vlm.py`.
Suggested implementation order: scaffolding (cli/ctx/report/config) → P0 → P1 → P2 →
P3 → P4 → P7 → P8 → P9 → then P5/P6 last (highest-risk heuristics; the feature flags
mean the pipeline is already useful without them).

**Acceptance:** one command on a raw AI-generated humanoid GLB produces: cleaned GLB +
report.json + QA renders, with results matching the manual workflow documented in docs/.

Testing without real meshes: generate synthetic fixtures in Blender (a sphere shattered
into coincident-vert fragments + a shrunken inner copy) to unit-test P2/P3 logic.

### M1 — batch regression

`runner/` batch-runs a directory of test meshes, aggregates per-assertion pass rates
into a CSV. Used to calibrate thresholds before trusting automation.

### M2 — service

FastAPI: `POST /jobs` (mesh upload + config overrides), `GET /jobs/{id}`,
`GET /jobs/{id}/artifacts`. Redis + RQ; each worker job runs
`subprocess.run(["blender","-b","-P","cli.py","--", ...])` with a 30-min wall-clock
kill. Job dir: `input/ output/ snapshots/ qa_renders/ report.json`.
States: `queued → running → done | needs_review | failed`.

### M3 — VLM QA + review loop

`qa/vlm.py` sends the QA renders to the Claude API, prompt demands JSON-only:
`{pass: bool, issues: [{view, problem, severity}]}` — checking bake smears, opening
shape, lip readability, poke-through. VLM fail → `needs_review` with issues in the
report. Plus a minimal review page listing `needs_review` jobs with renders +
assertion details and approve / retune-and-rerun actions.

## Non-goals

- No retopology (Quadriflow/remesh destroys UVs and never gives artist-grade loops —
  cleaned triangulated mesh is the accepted end state).
- No socket interior geometry by default (`build_socket: false`).
- No generality beyond humanoid AI-generated meshes in v1; part classification
  heuristics (symmetric pair near face top = eyes, etc.) assume this input class.
