# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project overview

This repo builds a **headless (non-MCP) Blender pipeline** that automatically cleans up
AI-generated character meshes — primarily output from **Trellis 2** generation. Typical
input defects: dual/duplicated surface layers, fragmented triangle soup, extra hidden
mesh pieces, bad geometry and topology, holes, and non-manifold edges. Optional
feature-flagged phases add **eyeball placement/replacement** and a **mouth bag**
(cavity + teeth + tongue).

The workflow originated as an interactive Blender MCP session driven by a human/agent
looking at screenshots. This project's goal is to **remove the MCP dependency
entirely**: everything runs via `blender --background` with Python (`bpy`/`bmesh`),
numeric assertions replace eyeballing, and headless QA renders replace viewport
screenshots.

## Source of truth

- `PLAN.md` — the implementation plan: repo layout, phase list (P0–P9), config
  defaults, per-phase assertions, milestones (M0 CLI → M1 batch runner → M2 service →
  M3 VLM QA). Follow it unless the user redirects.
- `docs/blender-body-mesh-cleanup.md` — mesh-cleanup domain knowledge: weld/split,
  hidden-geometry removal via ray-casting, mouth bag, eye carving/fitting,
  penetration testing, and hard-won bpy/bmesh pitfalls.
- `docs/blender-uv-atlas-rebake.md` — topology cleanup order, join hazards,
  dominant-normal-axis UV segmentation, atlas rebake procedure.
- `docs/uv-atlas-references/snippets.md` — complete, tested bpy code for the UV/bake
  phases. **Adapt these snippets instead of rewriting from scratch** — they encode
  non-obvious fixes.

Read the relevant doc(s) before writing code for any phase. The docs were written for
an interactive MCP session; translate per the "MCP → headless" notes in `PLAN.md`
(e.g. no user undo/edit-mode hazards, but stale references and `bpy.ops` context-poll
failures remain — prefer the `bmesh`/data API).

## Model orchestration policy

This project deliberately splits work between model tiers. The main session runs on
**Fable 5**; mechanical work is delegated to **Sonnet 5** subagents via the Agent tool
(`model: "sonnet"`).

**Fable 5 (main session) — decision making and result checking:**
- Architectural and algorithmic decisions (thresholds, phase ordering, heuristics).
- Interpreting assertion failures, QA renders, and `report.json` metrics; deciding
  whether a result passes or needs rework.
- Reviewing/verifying code written by subagents before commit.
- Anything ambiguous enough to need the docs' domain judgment.

**Sonnet 5 (subagents, `model: "sonnet"`) — reading, execution, code writing:**
- Bulk file reading and codebase exploration.
- Writing code for a well-specified task (give the subagent the exact spec, relevant
  doc excerpts, and acceptance criteria).
- Running Blender/CLI commands, tests, and batch jobs, then reporting output.

Pattern: Fable 5 decides *what* and *why*, writes the spec, and checks the result;
Sonnet 5 subagents do the *how*. Don't delegate judgment calls to subagents, and don't
burn the main context on mechanical file reading or long command output.

## Non-negotiable engineering rules

1. **Never silently proceed past a failed assertion.** Write the failure to
   `report.json`, save `snapshots/failed_pN.blend`, render QA views, set job status
   `needs_review`, and stop.
2. **Never cache `bpy.types.Object` references across operator calls.** Keep a name
   registry in the pipeline context and re-fetch `bpy.data.objects[name]` every time.
3. **Prefer `bmesh`/data API over `bpy.ops`** — many operators fail context poll in
   background mode.
4. **`matrix_world` is stale** after changing location/scale — call
   `bpy.context.view_layer.update()` before reading world positions.
5. **Snapshot before every destructive phase** (`snapshots/pre_pN.blend` in the job
   dir) so runs can resume from any phase.
6. **Diagnose numerically before editing, re-measure after.** Before/after metrics go
   into `report.json` for every phase.
7. Feature flags (`mouth.enabled`, `eyes.enabled`) must actually work — the pipeline
   must run cleanly with them off.
8. Don't chase zero boundary edges globally (hair cards/cloth are open by design), and
   don't trust `BVHTree.overlap()` alone for penetration — confirm with exact
   triangle–triangle tests.

## Runtime quirks (hard-won — do not rediscover)

- **`import bpy` must come before `import bmesh`** with the pip `bpy` wheel;
  the reverse order raises ModuleNotFoundError.
- **Pin `PYTHONHASHSEED=0`** for reproducible runs — randomized string hashing
  was observed to change UV packing results between otherwise-identical runs.
  The CLI warns when it isn't pinned.
- **Headless Workbench/EEVEE rendering needs GL**; without libEGL the render
  SIGABRTs the whole process (uncatchable). On bare containers:
  `apt install libegl1 libegl-mesa0 libgl1-mesa-dri` (Mesa software GL).
  `qa/render.py` preflights this and degrades to a report note.
- **Boolean modifiers need `material_mode='TRANSFER'`** (Blender 5.x defaults
  to INDEX) or cutter materials never reach the carved cavity faces — the
  mouth-bag/eye-socket material trick silently fails without it.
- Boolean-created cavity faces inherit **degenerate UVs** from UV-less
  cutters; p7 always re-unwraps the Mouth_Interior / Eye_Socket_Interior
  material slots for this reason.

## Running the pipeline

```bash
blender -b -P mesh_pipeline/cli.py -- \
  --input path/to/mesh.glb \
  --config mesh_pipeline/config/humanoid.yaml \
  --job-dir out/job1
```

Job dir output: cleaned GLB, `report.json`, `qa_renders/`, `snapshots/`.

Testing without real meshes: build synthetic fixtures in Blender (shattered sphere with
coincident verts + shrunken inner shell copy) to exercise the weld/split and
hidden-geometry phases.

## Git conventions

- Development happens on feature branches (current: 
  `claude/mesh-cleanup-trellis-workflow-t1wbx7`); `main` holds reviewed work.
- Commit per phase/logical unit with descriptive messages; never mix unrelated phases
  in one commit.
