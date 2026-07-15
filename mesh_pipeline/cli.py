"""Headless entry point for the mesh-cleanup pipeline.

Usage:
    blender -b -P mesh_pipeline/cli.py -- --input mesh.glb \\
        --config mesh_pipeline/config/humanoid.yaml --job-dir out/job1
    # or, with the bpy pip wheel:
    python -m mesh_pipeline.cli -- --input mesh.glb \\
        --config mesh_pipeline/config/humanoid.yaml --job-dir out/job1

Both invocations put pipeline args after a literal "--"; find_cli_argv() locates
it regardless of how many blender/python args precede it.

Runner behavior (see ARCHITECTURE.md "Runner behavior"):
  - Builds job_dir/{input,output,snapshots,qa_renders}.
  - Factory-resets the scene and imports the input mesh (fresh start), or
    reopens a snapshot .blend (resume via --start-phase / --only-phase).
  - Runs phases p0..p9 in order. Missing phase modules are tolerated (recorded
    as a stub "not implemented yet" ok result) so the runner works before all
    phases exist.
  - Snapshots snapshots/pre_<phase>.blend before DESTRUCTIVE phases.
  - Runs qa.assertions.check() after each phase if that module exists.
  - On needs_review/failed: snapshots snapshots/failed_<phase>.blend, attempts
    QA renders (qa.render, tolerated if missing), writes report.json, stops.
  - Exit codes: 0 done, 1 failed, 2 needs_review.
"""

from __future__ import annotations

import argparse
import importlib
import shutil
import sys
import time
import traceback
from pathlib import Path

from mesh_pipeline.config.schema import load_config
from mesh_pipeline.context import PhaseResult, PipelineContext
from mesh_pipeline.report import Report

PHASE_ORDER: list[str] = [
    "p0_diagnose",
    "p1_backup",
    "p2_weld_split",
    "p3_hidden_geo",
    "p4_topology",
    "p5_mouth",
    "p6_eyes",
    "p7_uv_atlas",
    "p8_bake",
    "p9_export",
]

_GLTF_EXTS = {".glb", ".gltf"}


def find_cli_argv(argv: list[str]) -> list[str]:
    """Return the args after a literal '--', working for both
    `blender -b -P cli.py -- <args>` and `python -m mesh_pipeline.cli -- <args>`.
    """
    if "--" in argv:
        return argv[argv.index("--") + 1 :]
    return argv[1:]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mesh_pipeline.cli", description="Headless mesh-cleanup pipeline runner"
    )
    parser.add_argument("--input", required=True, help="input mesh: .glb/.gltf, .obj, or .fbx")
    parser.add_argument("--config", required=True, help="pipeline config: .yaml/.yml or .json")
    parser.add_argument("--job-dir", required=True, help="job output directory")
    parser.add_argument(
        "--start-phase",
        default=None,
        help="resume: open snapshots/pre_<name>.blend and continue from this phase",
    )
    parser.add_argument(
        "--only-phase", default=None, help="debug: run exactly one phase, then stop"
    )
    return parser


def ensure_job_dirs(job_dir: Path) -> None:
    for sub in ("input", "output", "snapshots", "qa_renders"):
        (job_dir / sub).mkdir(parents=True, exist_ok=True)


def import_input_mesh(input_path: Path) -> list[str]:
    """Import the mesh into the (already factory-reset) scene.

    Returns the names of newly-created objects, detected by name-set
    difference (never trust context.active_object after an operator).
    """
    import bpy

    ext = input_path.suffix.lower()
    before = {o.name for o in bpy.data.objects}
    if ext in _GLTF_EXTS:
        bpy.ops.import_scene.gltf(filepath=str(input_path))
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=str(input_path))
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(input_path))
    else:
        raise ValueError(f"Unsupported input mesh extension: {ext!r} ({input_path})")
    after = {o.name for o in bpy.data.objects}
    new_names = sorted(after - before)
    if not new_names:
        raise RuntimeError(
            f"Import of {input_path} did not create any new objects (empty file?)"
        )
    return new_names


def initial_role_names(new_object_names: list[str]) -> dict[str, str]:
    """Best-effort initial name registry from a fresh import.

    Real part classification (body/eye_l/eye_r/teeth_*/tongue) is a phase
    concern (p0/p2); here we only register what we can know for certain: the
    raw imported objects, under dynamic "import_N" roles, plus a "body" alias
    when there is exactly one mesh object (the common case for a single
    triangle-soup humanoid mesh before P2 separates it).
    """
    import bpy

    names: dict[str, str] = {}
    mesh_names = [n for n in new_object_names if bpy.data.objects[n].type == "MESH"]
    for i, name in enumerate(mesh_names):
        names[f"import_{i}"] = name
    if len(mesh_names) == 1:
        names["body"] = mesh_names[0]
    return names


def _module_is_missing(exc: ModuleNotFoundError, module_name: str) -> bool:
    missing = exc.name or ""
    return module_name == missing or module_name.startswith(missing + ".")


def load_phase_module(phase_name: str):
    """Return the phase module, or None if it doesn't exist yet (tolerated)."""
    module_name = f"mesh_pipeline.phases.{phase_name}"
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if _module_is_missing(exc, module_name):
            return None
        raise


def load_assertions_module():
    try:
        return importlib.import_module("mesh_pipeline.qa.assertions")
    except ModuleNotFoundError as exc:
        if _module_is_missing(exc, "mesh_pipeline.qa.assertions"):
            return None
        raise


def load_render_module():
    try:
        return importlib.import_module("mesh_pipeline.qa.render")
    except ModuleNotFoundError as exc:
        if _module_is_missing(exc, "mesh_pipeline.qa.render"):
            return None
        raise


def run_phase_assertions(phase_name: str, ctx: PipelineContext, cfg: dict, result: PhaseResult) -> None:
    """Run qa.assertions.check() if present; escalate ok -> needs_review on failure."""
    assertions_mod = load_assertions_module()
    if assertions_mod is None or not hasattr(assertions_mod, "check"):
        return
    extra_failures = assertions_mod.check(phase_name, ctx, cfg, result.metrics) or []
    if extra_failures:
        result.failures = [*result.failures, *extra_failures]
        if result.status == "ok":
            result.status = "needs_review"


def attempt_qa_renders(ctx: PipelineContext, cfg: dict, tag: str) -> list[str]:
    """Attempt QA renders via qa.render; tolerate a missing module or failure."""
    render_mod = load_render_module()
    if render_mod is None:
        return ["qa.render module not present; QA renders skipped"]
    if not hasattr(render_mod, "render"):
        return ["qa.render module present but has no render(); QA renders skipped"]
    try:
        render_mod.render(ctx, cfg, tag=tag)
        return []
    except Exception as exc:  # QA rendering must never mask the underlying failure
        return [f"QA render failed: {type(exc).__name__}: {exc}"]


def run_one_phase(phase_name: str, ctx: PipelineContext, cfg: dict) -> PhaseResult:
    module = load_phase_module(phase_name)
    if module is None:
        return PhaseResult(phase=phase_name, status="ok", notes=["not implemented yet"])

    destructive = bool(getattr(module, "DESTRUCTIVE", False))
    if destructive:
        ctx.ensure_object_mode()
        ctx.save_snapshot(f"pre_{phase_name}")

    try:
        result = module.run(ctx, cfg)
    except Exception as exc:  # phases never call sys.exit; unexpected errors -> failed
        result = PhaseResult(
            phase=phase_name,
            status="failed",
            failures=[f"{type(exc).__name__}: {exc}"],
            notes=[traceback.format_exc()],
        )

    if result.status != "failed":
        run_phase_assertions(phase_name, ctx, cfg, result)
    return result


def resolve_scene_setup(
    job_dir: Path, input_path: Path, phase_list: list[str]
) -> None:
    """Fresh factory-reset + import, or reopen a resume snapshot."""
    import bpy

    resume_target = phase_list[0]
    is_resume = resume_target != PHASE_ORDER[0]
    if is_resume:
        snapshot_path = job_dir / "snapshots" / f"pre_{resume_target}.blend"
        if not snapshot_path.exists():
            raise FileNotFoundError(
                f"Cannot resume at phase {resume_target!r}: {snapshot_path} does not "
                "exist (it is only written before a DESTRUCTIVE phase runs)."
            )
        bpy.ops.wm.open_mainfile(filepath=str(snapshot_path))
    else:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        dest = job_dir / "input" / input_path.name
        shutil.copy2(input_path, dest)
        import_input_mesh(dest)


def hash_seed_note() -> str | None:
    """Warn when PYTHONHASHSEED is not pinned.

    Empirically (see tests/test_e2e.py): with a randomized hash seed,
    repeated runs on identical input produced differing p7 island packing
    (island_overlap_count varied run-to-run) while all other metrics stayed
    bit-identical. Until the exact order-dependence is isolated, pin
    PYTHONHASHSEED=0 for reproducible pipelines (batch regression runs in
    particular should always pin it).
    """
    import os

    seed = os.environ.get("PYTHONHASHSEED")
    if seed is None or seed == "random":
        return (
            "PYTHONHASHSEED is not pinned; UV packing results may vary "
            "between otherwise-identical runs. Set PYTHONHASHSEED=0 for "
            "reproducible output."
        )
    return None


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(find_cli_argv(sys.argv if argv is None else argv))

    job_dir = Path(args.job_dir).resolve()
    input_path = Path(args.input).resolve()
    config_path = Path(args.config).resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input mesh not found: {input_path}")

    ensure_job_dirs(job_dir)
    cfg = load_config(config_path)

    if args.only_phase:
        if args.only_phase not in PHASE_ORDER:
            raise ValueError(f"--only-phase {args.only_phase!r} not in {PHASE_ORDER}")
        phase_list = [args.only_phase]
    elif args.start_phase:
        if args.start_phase not in PHASE_ORDER:
            raise ValueError(f"--start-phase {args.start_phase!r} not in {PHASE_ORDER}")
        start_idx = PHASE_ORDER.index(args.start_phase)
        phase_list = PHASE_ORDER[start_idx:]
    else:
        phase_list = list(PHASE_ORDER)

    resolve_scene_setup(job_dir, input_path, phase_list)

    import bpy

    ctx = PipelineContext(job_dir=job_dir)
    if phase_list[0] == PHASE_ORDER[0]:
        new_names = [o.name for o in bpy.data.objects]
        ctx.names = initial_role_names(new_names)
    else:
        # Resume: restore the name registry saved alongside the snapshot.
        if not ctx.load_registry(f"pre_{phase_list[0]}"):
            raise FileNotFoundError(
                f"Cannot resume at phase {phase_list[0]!r}: registry file "
                f"snapshots/pre_{phase_list[0]}.names.json not found."
            )

    report = Report(input_path=str(input_path), config_path=str(config_path))
    note = hash_seed_note()
    if note:
        print(f"[mesh_pipeline] warning: {note}", file=sys.stderr)
        report.input["warning"] = note
    report_path = job_dir / "report.json"
    report.write_atomic(report_path)

    exit_code = 0
    final_status = "done"

    for phase_name in phase_list:
        t0 = time.monotonic()
        result = run_one_phase(phase_name, ctx, cfg)
        duration_s = time.monotonic() - t0

        report.add_phase(result, duration_s)
        report.write_atomic(report_path)

        if result.status in ("needs_review", "failed"):
            try:
                ctx.ensure_object_mode()
                ctx.save_snapshot(f"failed_{phase_name}")
            except Exception as exc:
                result.notes.append(f"failed to save failure snapshot: {exc}")
            result.notes.extend(attempt_qa_renders(ctx, cfg, tag=f"failed_{phase_name}"))
            report.phases[-1] = {**report.phases[-1], "notes": result.notes}
            report.write_atomic(report_path)

            final_status = result.status
            exit_code = 2 if result.status == "needs_review" else 1
            break

    report.finalize(final_status)
    report.write_atomic(report_path)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
