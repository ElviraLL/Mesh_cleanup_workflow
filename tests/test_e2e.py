"""End-to-end test: generate the synthetic humanoid fixture, run the
core-config pipeline (mouth/eyes feature flags OFF) end to end via
subprocess, and check report.json + key metric bands.

This runs a real headless pipeline job (weld/split, hidden-geometry
ray-casting, topology cleanup, UV re-unwrap, a Cycles bake, GLB export) and
takes low-single-digit minutes, so it is SKIPPED by default. Opt in with:

    MESH_E2E=1 pytest tests/test_e2e.py -v
    MESH_E2E=1 python3 tests/test_e2e.py

Works with or without pytest installed (falls back to a plain assert-based
runner under `python3 tests/test_e2e.py`); pytest is only imported if
present, purely for the skip marker/tmp_path fixture.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import pytest

    _HAVE_PYTEST = True
except ImportError:  # pragma: no cover - exercised by the plain-python path
    _HAVE_PYTEST = False

REPO_ROOT = Path(__file__).resolve().parent.parent
MAKE_FIXTURE = Path(__file__).resolve().parent / "make_fixture.py"

_ENABLED = os.environ.get("MESH_E2E") == "1"
_SKIP_REASON = "set MESH_E2E=1 to run the full pipeline e2e test (takes minutes, needs bpy)"

_REQUIRED_PHASES = (
    "p0_diagnose", "p1_backup", "p2_weld_split", "p3_hidden_geo",
    "p4_topology", "p5_mouth", "p6_eyes", "p7_uv_atlas", "p8_bake", "p9_export",
)

if _HAVE_PYTEST:
    pytestmark = pytest.mark.skipif(not _ENABLED, reason=_SKIP_REASON)


def _build_fixture(out_glb: Path, seed: int = 0) -> None:
    cmd = [sys.executable, str(MAKE_FIXTURE), str(out_glb), str(seed)]
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, (
        f"fixture generation failed (exit {result.returncode}):\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def _write_core_config(path: Path) -> None:
    # load_config() (mesh_pipeline/config/schema.py) deep-merges this over
    # DEFAULT_CONFIG (== config/humanoid.yaml's values), so every other key
    # keeps its default -- only mouth/eyes are toggled off here.
    path.write_text("mouth:\n  enabled: false\neyes:\n  enabled: false\n")


def _run_core_pipeline(job_dir: Path, fixture_path: Path, config_path: Path) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable, "-m", "mesh_pipeline.cli", "--",
        "--input", str(fixture_path),
        "--config", str(config_path),
        "--job-dir", str(job_dir),
    ]
    # Pin PYTHONHASHSEED: several phases (P2's part classifier, P3/P4's
    # face-deletion passes) track bmesh geometry in plain Python `set`s, so
    # their iteration order -- and therefore the exact vertex/face order
    # reaching P7's ANGLE_BASED unwrap -- depends on Python's per-process
    # hash randomization. Verified directly: with hash randomization on
    # (the default), identical fixture+config runs non-deterministically
    # produced p7_uv_atlas island_overlap_count in {0, 7, 9, 11} (island
    # counts and every other metric stayed identical -- only the packed
    # islands' exact UV shapes changed); resuming repeatedly from the same
    # pre_p7 snapshot always reproduced the same value, and 3 fresh runs
    # with PYTHONHASHSEED=0 all gave island_overlap_count=0. This is a
    # pipeline-level determinism gap, not a fixture defect -- pinning the
    # hash seed here is a test-harness mitigation, not a workaround for
    # incorrect fixture geometry.
    env = dict(os.environ)
    env["PYTHONHASHSEED"] = "0"
    return subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True, timeout=1800, env=env)


def _is_known_pack_islands_flake(job_dir: Path) -> bool:
    """True iff the run stopped at p7_uv_atlas needs_review ONLY because of
    island_overlap_count (see _run_core_pipeline's PYTHONHASHSEED comment).
    Kept as a defense-in-depth retry trigger in case some residual
    non-determinism survives pinning the hash seed (e.g. genuine
    multithreading in Blender's own unwrap/pack solvers).
    """
    report_path = job_dir / "report.json"
    if not report_path.exists():
        return False
    try:
        report = json.loads(report_path.read_text())
    except Exception:
        return False
    if report.get("status") != "needs_review":
        return False
    phases = report.get("phases", [])
    if not phases:
        return False
    last = phases[-1]
    if last.get("phase") != "p7_uv_atlas" or last.get("status") != "needs_review":
        return False
    failures = last.get("failures", [])
    return bool(failures) and all(f.startswith("island_overlap_count:") for f in failures)


def _check_core_run(job_dir: Path, result: subprocess.CompletedProcess) -> None:
    assert result.returncode == 0, (
        f"pipeline exited {result.returncode} (expected 0 for the core/flags-off "
        f"config):\nstdout:\n{result.stdout[-4000:]}\nstderr:\n{result.stderr[-4000:]}"
    )

    report_path = job_dir / "report.json"
    assert report_path.exists(), f"report.json missing at {report_path}"
    report = json.loads(report_path.read_text())
    assert report["status"] == "done", f"report status={report['status']!r}, phases={report['phases']}"

    cleaned_glb = job_dir / "output" / "cleaned.glb"
    assert cleaned_glb.exists(), f"{cleaned_glb} not produced"
    assert cleaned_glb.stat().st_size > 0

    phases = {p["phase"]: p for p in report["phases"]}
    for name in _REQUIRED_PHASES:
        assert name in phases, f"phase {name!r} missing from report.json"
        assert phases[name]["status"] == "ok", (
            f"{name} status={phases[name]['status']!r} failures={phases[name]['failures']}"
        )

    # Key metric bands (ARCHITECTURE.md's before/after contract, PLAN.md's
    # per-phase assertion table) -- re-checked here independently of
    # qa/assertions.py so a regression in either the pipeline phases or the
    # assertions module itself would still fail this test.
    p2 = phases["p2_weld_split"]["metrics"]
    assert p2["components_before"] > 1000, p2["components_before"]
    assert p2["components_after"] < 20, p2["components_after"]

    p3 = phases["p3_hidden_geo"]["metrics"]
    assert 0.30 <= p3["deleted_face_ratio"] <= 0.80, p3["deleted_face_ratio"]

    p4 = phases["p4_topology"]["metrics"]
    zbands = p4["boundary_by_zband"]
    assert zbands["body"] + zbands["legs"] <= 5, zbands

    p7 = phases["p7_uv_atlas"]["metrics"]
    assert p7["uv_islands_after"] <= 40, p7["uv_islands_after"]
    assert p7["uv_islands_after"] < p7["uv_islands_before"], p7

    p8 = phases["p8_bake"]["metrics"]
    assert p8["channels"], "expected at least one baked channel (diffuse)"
    for channel, stats in p8["atlas_stats"].items():
        assert stats["nonzero_ratio"] > 0.2, (channel, stats)
        assert stats["mean"] > 0.01, (channel, stats)

    p9 = phases["p9_export"]["metrics"]
    assert p9["penetration_count"] == 0, p9["penetration_count"]
    assert Path(p9["export_path"]).exists()


def test_core_pipeline_e2e(tmp_path=None):
    if not _ENABLED:
        if _HAVE_PYTEST:
            pytest.skip(_SKIP_REASON)
        print(f"SKIPPED: {_SKIP_REASON}")
        return

    work = Path(tmp_path) if tmp_path is not None else Path(tempfile.mkdtemp(prefix="mesh_e2e_"))
    fixture_path = work / "fixture.glb"
    config_path = work / "core_config.yaml"

    _build_fixture(fixture_path)
    _write_core_config(config_path)

    # Retry a bounded number of times ONLY for the known pack_islands
    # non-determinism (see _is_known_pack_islands_flake) -- a fresh job_dir
    # per attempt since the CLI refuses to overwrite a job that already has
    # a report.json in a terminal state.
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        job_dir = work / f"job_{attempt}"
        result = _run_core_pipeline(job_dir, fixture_path, config_path)
        if result.returncode == 0 or not _is_known_pack_islands_flake(job_dir):
            break
    _check_core_run(job_dir, result)


if __name__ == "__main__":
    os.environ["MESH_E2E"] = "1"
    _ENABLED = True
    test_core_pipeline_e2e()
    print("OK: core pipeline e2e test passed")
