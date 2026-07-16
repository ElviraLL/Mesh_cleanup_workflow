"""Pure-python unit tests for mesh_pipeline.geom.head_z_band.

No bpy dependency -- head_z_band takes plain (x, z) tuples. Run with:

    python3 tests/test_geom.py
    pytest tests/test_geom.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from mesh_pipeline.geom import head_z_band  # noqa: E402


def _build_tpose_profile(head_width: float, neck_width: float, shoulder_width: float,
                          n_per_region: int = 40) -> list[tuple[float, float]]:
    """Synthetic (x, z) scatter over z in [0, 1]:

    - top 15% (z in [0.85, 1.0]): head, width = head_width
    - next 5% (z in [0.80, 0.85]): neck, width = neck_width (narrower than head)
    - below z=0.80: shoulders/arms/torso, width = shoulder_width (T-pose span)

    Points are laid out on 60 evenly spaced z rows (matching head_z_band's
    internal bucket count) so every region contributes several non-empty
    slices, with 5 x-samples spanning [-width/2, width/2] per row.
    """
    pts: list[tuple[float, float]] = []
    n_rows = 60
    for row in range(n_rows):
        z = 1.0 - row / (n_rows - 1)
        if z >= 0.85:
            w = head_width
        elif z >= 0.80:
            w = neck_width
        else:
            w = shoulder_width
        for k in range(5):
            x = -w / 2.0 + w * k / 4.0
            pts.append((x, z))
    return pts


def test_tpose_like_profile_band_is_top_15_to_20_pct():
    pts = _build_tpose_profile(head_width=0.12, neck_width=0.07, shoulder_width=0.9)
    band = head_z_band(pts)
    assert band is not None, "expected a band, got None"
    z_lo, z_hi = band
    assert z_hi == 1.0
    # z_lo should land in the neck/shoulder transition -> zfrac_from_bottom (z_lo)
    # should be roughly 0.80-0.85 (top 15-20% of height).
    assert 0.78 <= z_lo <= 0.87, f"expected z_lo in [0.78, 0.87], got {z_lo}"


def test_hair_widened_head_still_bands_top_15_to_20_pct():
    # Hair widens the head from 0.12 to 0.18 -- band boundary should barely move.
    pts = _build_tpose_profile(head_width=0.18, neck_width=0.07, shoulder_width=0.9)
    band = head_z_band(pts)
    assert band is not None, "expected a band, got None"
    z_lo, z_hi = band
    assert z_hi == 1.0
    assert 0.78 <= z_lo <= 0.87, f"expected z_lo in [0.78, 0.87], got {z_lo}"


def test_no_shoulders_returns_none():
    # A uniformly narrow profile (e.g. an isolated head/bust mesh) never
    # "explodes" -- head_z_band must honestly report it cannot find a band,
    # not guess.
    pts = [(x, z) for z in [i / 59.0 for i in range(60)] for x in (-0.1, -0.05, 0.0, 0.05, 0.1)]
    band = head_z_band(pts)
    assert band is None


def test_degenerate_input_returns_none():
    assert head_z_band([]) is None
    assert head_z_band([(0.0, 0.0)] * 30) is None  # zero height/width
    assert head_z_band([(0.0, 0.0), (0.01, 0.01)]) is None  # too few points


def test_avatar003_like_gradual_widening_excludes_torso_bead_height():
    """Regression shape for the real bug: a gradually-widening head/hair
    silhouette (like avatar_003's) that only "explodes" partway down,
    while a mirrored bead pair sits much lower (zfrac 0.55) and must fall
    outside the returned band.
    """
    n_rows = 60
    pts: list[tuple[float, float]] = []
    # Roughly mirrors the avatar_003 calibration table in geom.py's docstring.
    widths_top_to_bottom = [
        0.0687, 0.0854, 0.0880, 0.0912, 0.0931, 0.1059, 0.1206, 0.1371,
        0.2377, 0.2935, 0.5632, 0.8343, 0.9574, 0.9841, 0.9930, 0.9648,
    ]
    for row in range(n_rows):
        z = 1.0 - row / (n_rows - 1)
        if row < len(widths_top_to_bottom):
            w = widths_top_to_bottom[row]
        else:
            w = 0.22  # torso width below the arm peak
        for k in range(5):
            x = -w / 2.0 + w * k / 4.0
            pts.append((x, z))
    band = head_z_band(pts)
    assert band is not None
    z_lo, _z_hi = band
    bead_z = 0.55  # zfrac_from_bottom, matches the ground-truth bug report
    assert bead_z < z_lo, f"expected bead z={bead_z} outside band (z_lo={z_lo})"


def _run_all():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
