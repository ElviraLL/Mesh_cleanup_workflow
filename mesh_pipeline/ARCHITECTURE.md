# Pipeline architecture contract

This file is the binding interface spec for all `mesh_pipeline` modules. Phase
implementations must conform to it exactly. Read `PLAN.md` and `docs/` for domain
rationale; this file defines the code contract.

## Runtime target

- Blender 4.x headless: `blender -b -P mesh_pipeline/cli.py -- <args>`, or the `bpy`
  pip wheel (`python -m mesh_pipeline.cli <args>`). Code must work in both.
- Dependencies: Python stdlib + `bpy`. YAML config requires PyYAML in Blender's
  Python (`.json` config accepted as fallback, same structure).
- **Never** use operators requiring a view/window context (`bpy.ops.view3d.*`,
  screenshots). QA visuals come from `bpy.ops.render.render(write_still=True)` with
  explicitly created cameras.

## Core types (`mesh_pipeline/context.py`)

```python
@dataclass
class PhaseResult:
    phase: str                    # e.g. "p2_weld_split"
    status: str                   # "ok" | "needs_review" | "failed"
    metrics: dict                 # JSON-serializable before/after numbers
    failures: list[str]           # assertion failure messages (empty when ok)
    notes: list[str]              # informational messages

class PipelineContext:
    job_dir: Path                 # contains input/ output/ snapshots/ qa_renders/
    names: dict[str, str]         # role -> object name: "body", "eye_l", "eye_r",
                                  # "teeth_u", "teeth_l", "tongue", plus dynamic roles
    backup_names: dict[str, str]  # role -> backup object name (bake sources)

    def obj(self, role: str) -> bpy.types.Object   # ALWAYS re-fetch bpy.data.objects[name]
    def save_snapshot(self, tag: str) -> Path      # snapshots/<tag>.blend via save_as_mainfile
    def ensure_object_mode(self) -> None
```

Rules encoded here (from docs, still apply headless):
- Never cache `bpy.types.Object` across operator calls — store names, re-fetch.
- After moving/scaling objects call `bpy.context.view_layer.update()` before reading
  `matrix_world`.
- Detect operator-created objects by name-set difference, never `context.active_object`.

## Phase module contract (`mesh_pipeline/phases/pN_*.py`)

Each phase module exposes exactly:

```python
PHASE_NAME = "pN_shortname"
DESTRUCTIVE = True | False        # True -> runner snapshots pre_pN.blend first

def run(ctx: PipelineContext, cfg: dict) -> PhaseResult: ...
```

- `cfg` is the full validated config dict; phases read their own section.
- Phases NEVER call `sys.exit`, never swallow exceptions silently. An unexpected
  exception propagates; the runner converts it to status `failed`.
- Phases record before/after metrics in `PhaseResult.metrics` (e.g.
  `{"components_before": 41230, "components_after": 7}`).
- Phases set `needs_review` themselves only for domain conditions the assertion table
  covers; the runner also applies `qa/assertions.py` checks after each phase.
- Feature-flagged phases (p5, p6) return `status="ok"`, `notes=["skipped: disabled"]`
  immediately when their flag is off.

## Runner behavior (`cli.py`)

Order: p0, p1, p2, p3, p4, p5, p6, p7, p8, p9.

Per phase: (1) if DESTRUCTIVE, `ctx.save_snapshot(f"pre_{PHASE_NAME}")`; (2) run;
(3) run assertion checks; (4) append result to `report.json`. On `needs_review` or
`failed`: save `snapshots/failed_<PHASE_NAME>.blend`, render QA views, write report,
**stop** (exit code 2 for needs_review, 1 for failed, 0 for success).

CLI args (after `--`): `--input <mesh.glb>` (also .fbx/.obj), `--config <yaml|json>`,
`--job-dir <dir>`, `--start-phase <name>` (resume: opens `snapshots/pre_<name>.blend`),
`--only-phase <name>` (debug).

## Assertions (`qa/assertions.py`)

```python
def check(phase_name: str, ctx, cfg, metrics: dict) -> list[str]  # [] = pass
```

Implements the PLAN.md table (P2 component count, P3 deletion band, P4 boundary
z-bands, P5 teeth recess, P6 eye asymmetry with one auto local-visibility retry,
P7 island count/overlap, P8 non-empty bake, P9 exact tri-tri penetration = 0).

P7's `uv.max_islands` cap applies only when p7's re-unwrap owns the resulting
layout (`faces_reunwrapped > 0.5 * faces_total`); when p7 kept the input's
pre-existing unwrap and only touched a minority of faces, the island count
reflects the input, not p7's work, so the cap is skipped. `island_overlap_count
== 0` is always required, computed via the two-stage bbox-candidate +
point-in-triangle test described in `p7_uv_atlas.py`.

## Report (`report.py`)

`report.json` schema:
```json
{
  "input": {...}, "config_path": "...", "started_at": "...", "finished_at": "...",
  "status": "done|needs_review|failed",
  "phases": [ {PhaseResult fields + "duration_s"} ]
}
```
Written atomically (temp file + rename) after every phase so a crash leaves a valid
partial report.

## Metrics key contract (phases -> assertions)

`qa/assertions.py` reads these exact keys from `PhaseResult.metrics`. Phases MUST
emit them (assertions treat a missing key as a failure, not a pass):

| Phase | Required metrics keys |
|---|---|
| p0_diagnose | `verts`, `faces`, `components`, `boundary_edges`, `nonmanifold_edges`, `uv_islands` (int or null), `materials` (list) |
| p1_backup | `backed_up` (list of object names), `backup_collection` |
| p2_weld_split | `components_before`, `components_after`, `weld_distance`, `verts_merged`, `parts` (dict role->name for classified parts) |
| p3_hidden_geo | `parts_deleted` (list), `faces_before`, `faces_deleted`, `deleted_face_ratio` (0..1), `faces_protected_mouth` (int, 0 when mouth disabled or no lip line found) |
| p4_topology | `boundary_edges_before`, `boundary_edges_after`, `boundary_by_zband` (dict zband->count; zbands: `"body"`, `"legs"`, `"head_hair"`), `holes_filled`, `flaps_deleted`, `fins_deleted` |
| p5_mouth | `skipped` (bool), `mode` (`"carved_closed"` \| `"reference_kept"` \| `null`), `teeth_recess_mm` (float, null in `reference_kept` mode), `opening_z_range` ([lo,hi], thin closed-lips slit at the outer skin surface), `bag_z_range` ([lo,hi], interior cavity where teeth/tongue live, null if unmeasurable), `teeth_z_range` ([lo,hi]), `lip_gap_mm_measured` (float, mm), `teeth_reference_kept` (bool), `lips_topologically_split` (bool) |
| p6_eyes | `skipped` (bool), `clearance_l`, `clearance_r`, `asymmetry_ratio` (abs(l-r)/max(l,r)), `local_visibility_retried` (bool) |
| p7_uv_atlas | `uv_islands_before`, `uv_islands_after`, `island_overlap_count`, `faces_total`, `faces_reunwrapped` |
| p8_bake | `channels` (list), `atlas_stats` (dict channel -> {mean, nonzero_ratio}) |
| p9_export | `penetration_count`, `export_path`, `object_names` (list) |

When a feature-flagged phase is skipped it sets `skipped: true` and assertions
pass it unconditionally.

## Shared geometry helpers (`mesh_pipeline/geom.py`)

Pure-Python/bmesh utilities used by multiple phases — keep bpy-optional where
possible so unit tests can run without Blender:
- `fibonacci_sphere(n)` -> list of unit direction tuples (pure python)
- `connected_components(bm)` -> list of vert-index sets
- `uv_island_count(me)` union-find (from docs snippets)
- `boundary_loop_histogram(bm)` union-find over boundary edges
- `head_z_band(xz_points)` -> `(z_lo, z_hi) | None`, pure python: finds the head's
  z-range from a body's world (x, z) vertex scatter by slicing z into ~60 buckets
  and locating where the x-width profile "explodes" into shoulders/arms; used by
  p2 (eye-pair gating), p5 (lip-line search band), p6 (post-carve sanity check)
- `body_xz_points(obj, max_samples=4000)` -> subsampled world (x, z) vertex list,
  the shared input to `head_z_band` for all three callers above
- axis-label segmentation + region absorption helpers (pure logic on adjacency
  graphs, taking plain data structures, so they are testable without bpy)
