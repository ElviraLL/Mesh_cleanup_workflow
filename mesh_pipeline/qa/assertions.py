"""Per-phase numeric assertions (PLAN.md "Per-phase assertions" table).

`check(phase_name, ctx, cfg, metrics) -> list[str]` is called by
`cli.run_phase_assertions` after every phase; a non-empty return escalates an
"ok" PhaseResult to "needs_review" (cli.py handles the escalation, this module
only judges). Reads the exact metrics keys documented in ARCHITECTURE.md's
"Metrics key contract" table -- a *missing* required key is a failure (not a
free pass), so a phase that forgets to emit a contract key gets caught here
instead of silently sailing through.

Kept bpy-free (pure dict/number logic) so it can be unit-tested without
Blender -- `ctx` is accepted for interface symmetry with cli.py's call site
and for phases that might need scene lookups in the future, but no assertion
below actually touches bpy.
"""

from __future__ import annotations

from typing import Any


def check(phase_name: str, ctx, cfg: dict, metrics: dict) -> list[str]:
    handler = _HANDLERS.get(phase_name)
    if handler is None:
        return []
    return handler(ctx, cfg, metrics or {})


def _require(metrics: dict, keys: list[str], failures: list[str]) -> bool:
    """Append a failure for each missing key; return True iff all present."""
    ok = True
    for key in keys:
        if key not in metrics:
            failures.append(f"{key}: missing required metrics key")
            ok = False
    return ok


def _skip_if_flagged(metrics: dict) -> bool:
    return bool(metrics.get("skipped"))


# ---------------------------------------------------------------------------
# p0 / p1 -- no assertions
# ---------------------------------------------------------------------------


def _p0(ctx, cfg, metrics) -> list[str]:
    return []


def _p1(ctx, cfg, metrics) -> list[str]:
    return []


# ---------------------------------------------------------------------------
# p2 -- connected components collapsed to < 20
# ---------------------------------------------------------------------------


def _p2(ctx, cfg, metrics) -> list[str]:
    failures: list[str] = []
    if not _require(metrics, ["components_after"], failures):
        return failures
    n = metrics["components_after"]
    if not (isinstance(n, (int, float)) and n < 20):
        failures.append(f"components_after: expected < 20, got {n!r}")
    return failures


# ---------------------------------------------------------------------------
# p3 -- deleted-face ratio in [0.30, 0.80]
# ---------------------------------------------------------------------------


def _p3(ctx, cfg, metrics) -> list[str]:
    failures: list[str] = []
    if not _require(metrics, ["deleted_face_ratio"], failures):
        return failures
    r = metrics["deleted_face_ratio"]
    if not (isinstance(r, (int, float)) and 0.30 <= r <= 0.80):
        failures.append(
            f"deleted_face_ratio: expected in [0.30, 0.80], got {r!r}"
        )
    return failures


# ---------------------------------------------------------------------------
# p4 -- boundary edges near zero in body/legs z-bands; head_hair unrestricted
# ---------------------------------------------------------------------------


def _p4(ctx, cfg, metrics) -> list[str]:
    failures: list[str] = []
    if not _require(metrics, ["boundary_by_zband"], failures):
        return failures
    zbands = metrics["boundary_by_zband"]
    if not isinstance(zbands, dict):
        failures.append(f"boundary_by_zband: expected a dict, got {type(zbands).__name__}")
        return failures
    missing = [z for z in ("body", "legs") if z not in zbands]
    if missing:
        for z in missing:
            failures.append(f"boundary_by_zband[{z!r}]: missing required zband key")
        return failures
    body_legs = zbands["body"] + zbands["legs"]
    if body_legs > 5:
        failures.append(
            "boundary_by_zband: body+legs boundary edges expected <= 5 combined, "
            f"got {body_legs} (body={zbands['body']!r}, legs={zbands['legs']!r})"
        )
    return failures


# ---------------------------------------------------------------------------
# p5 -- teeth recess 1-2mm behind lip rim; arches overlap opening z-range
# ---------------------------------------------------------------------------


def _p5(ctx, cfg, metrics) -> list[str]:
    if _skip_if_flagged(metrics):
        return []
    failures: list[str] = []
    if not _require(metrics, ["teeth_recess_mm", "opening_z_range", "teeth_z_range"], failures):
        return failures
    recess = metrics["teeth_recess_mm"]
    if not (isinstance(recess, (int, float)) and 1.0 <= recess <= 2.0):
        failures.append(f"teeth_recess_mm: expected in [1.0, 2.0], got {recess!r}")

    opening = metrics["opening_z_range"]
    teeth = metrics["teeth_z_range"]
    if not _valid_range(opening):
        failures.append(f"opening_z_range: expected a [lo, hi] pair, got {opening!r}")
    elif not _valid_range(teeth):
        failures.append(f"teeth_z_range: expected a [lo, hi] pair, got {teeth!r}")
    elif not _ranges_overlap(opening, teeth):
        failures.append(
            f"teeth_z_range {teeth!r} does not overlap opening_z_range {opening!r}"
        )
    return failures


def _valid_range(r: Any) -> bool:
    return (
        isinstance(r, (list, tuple))
        and len(r) == 2
        and all(isinstance(v, (int, float)) for v in r)
    )


def _ranges_overlap(a, b) -> bool:
    a_lo, a_hi = sorted(a)
    b_lo, b_hi = sorted(b)
    return a_lo <= b_hi and b_lo <= a_hi


# ---------------------------------------------------------------------------
# p6 -- left/right eye clearance asymmetry < 30%
# ---------------------------------------------------------------------------


def _p6(ctx, cfg, metrics) -> list[str]:
    if _skip_if_flagged(metrics):
        return []
    failures: list[str] = []
    if not _require(metrics, ["asymmetry_ratio"], failures):
        return failures
    ratio = metrics["asymmetry_ratio"]
    if not (isinstance(ratio, (int, float)) and ratio < 0.30):
        retried = metrics.get("local_visibility_retried")
        context = (
            " (already retried with a local visibility pass, per docs Phase 3/6 -- "
            "this is the final value)"
            if retried
            else " (no local-visibility retry recorded)"
        )
        failures.append(
            f"asymmetry_ratio: expected < 0.30, got {ratio!r}{context}"
        )
    return failures


# ---------------------------------------------------------------------------
# p7 -- UV island count <= target; zero island overlap after pack
# ---------------------------------------------------------------------------


def _p7(ctx, cfg, metrics) -> list[str]:
    failures: list[str] = []
    if not _require(metrics, ["uv_islands_after", "island_overlap_count"], failures):
        return failures
    max_islands = cfg.get("uv", {}).get("max_islands", 40)
    islands = metrics["uv_islands_after"]
    if not (isinstance(islands, (int, float)) and islands <= max_islands):
        failures.append(
            f"uv_islands_after: expected <= {max_islands} (uv.max_islands), got {islands!r}"
        )
    overlap = metrics["island_overlap_count"]
    if not (isinstance(overlap, (int, float)) and overlap == 0):
        failures.append(f"island_overlap_count: expected 0, got {overlap!r}")
    return failures


# ---------------------------------------------------------------------------
# p8 -- every baked channel non-empty (nonzero_ratio > 0.2, mean > 0.01)
# ---------------------------------------------------------------------------


def _p8(ctx, cfg, metrics) -> list[str]:
    failures: list[str] = []
    if not _require(metrics, ["channels", "atlas_stats"], failures):
        return failures
    channels = metrics["channels"]
    if channels == []:
        return []  # no textures baked -- assertion passes unconditionally
    atlas_stats = metrics["atlas_stats"]
    if not isinstance(atlas_stats, dict):
        failures.append(f"atlas_stats: expected a dict, got {type(atlas_stats).__name__}")
        return failures
    for ch in channels:
        if ch not in atlas_stats:
            failures.append(f"atlas_stats[{ch!r}]: missing required channel key")
            continue
        stats = atlas_stats[ch]
        if not isinstance(stats, dict):
            failures.append(f"atlas_stats[{ch!r}]: expected a dict, got {type(stats).__name__}")
            continue
        nonzero_ratio = stats.get("nonzero_ratio")
        mean = stats.get("mean")
        if not (isinstance(nonzero_ratio, (int, float)) and nonzero_ratio > 0.2):
            failures.append(
                f"atlas_stats[{ch!r}].nonzero_ratio: expected > 0.2, got {nonzero_ratio!r}"
            )
        if not (isinstance(mean, (int, float)) and mean > 0.01):
            failures.append(f"atlas_stats[{ch!r}].mean: expected > 0.01, got {mean!r}")
    return failures


# ---------------------------------------------------------------------------
# p9 -- zero exact-tri-tri penetrations outside opening windows
# ---------------------------------------------------------------------------


def _p9(ctx, cfg, metrics) -> list[str]:
    failures: list[str] = []
    if not _require(metrics, ["penetration_count"], failures):
        return failures
    n = metrics["penetration_count"]
    if not (isinstance(n, (int, float)) and n == 0):
        failures.append(f"penetration_count: expected 0, got {n!r}")
    return failures


_HANDLERS = {
    "p0_diagnose": _p0,
    "p1_backup": _p1,
    "p2_weld_split": _p2,
    "p3_hidden_geo": _p3,
    "p4_topology": _p4,
    "p5_mouth": _p5,
    "p6_eyes": _p6,
    "p7_uv_atlas": _p7,
    "p8_bake": _p8,
    "p9_export": _p9,
}
