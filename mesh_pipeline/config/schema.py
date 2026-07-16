"""Pure-stdlib config schema, defaults, and validation (no pydantic).

DEFAULT_CONFIG mirrors PLAN.md's YAML block exactly (with its two pseudo-YAML
bits -- weld.auto_pick and hole_fill.protected_zones -- turned into proper
structured values; see mesh_pipeline/config/humanoid.yaml for the on-disk
version with comments).

load_config(path) reads a .yaml/.yml or .json file, deep-merges it over
DEFAULT_CONFIG, validates types/ranges, and returns the merged dict. Unknown
keys anywhere in the (non-open) config tree raise a ValueError listing every
offending dotted path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Dotted paths whose *values* are open dicts (arbitrary string keys), so the
# unknown-key check does not descend into them.
_OPEN_DICT_PATHS = {"export.names"}

DEFAULT_CONFIG: dict[str, Any] = {
    "input": {
        "up_axis": "auto",  # detect + normalize
    },
    "weld": {
        "sweep": [1e-4, 5e-3],  # probe on throwaway bmesh copies
        "auto_pick": {
            # first threshold before merge-rate exceeds this fraction of verts
            "strategy": "max_merge_rate",
            "max_merge_rate": 0.03,
        },
    },
    "hidden_geometry": {
        "part_rays": 26,  # part-level visibility directions
        "face_rays": 48,  # fibonacci-sphere per-face pass
        "dilate_rings": 2,  # protect concave areas (nostrils/ears)
    },
    "hole_fill": {
        "max_loop_edges": 20,
        # NEVER auto-fill loops inside these zones (prevents capping eye sockets)
        "protected_zones": [
            {
                "type": "eye_regions",
                # sphere around each eyeball center, r = radius_factor * eyeball radius
                "radius_factor": 1.5,
            },
            {
                "type": "lip_region",  # box around detected lip line
            },
        ],
    },
    "mouth": {
        "enabled": True,  # feature flag -- pipeline must run fine with this off
        "cutter_radii": [0.020, 0.042, 0.0085],  # per 1-unit body height, scale accordingly (rx, ry fallback-rz)
        "bag_height": 0.012,  # per 1-unit body height -- interior cavity half-height (rz) for the closed-lips bag
        "lip_gap_mm": 0.4,  # target closed-lip slit height at the outer skin surface
        "teeth_recess_mm": [1, 2],  # assertion range behind lip rim (carved_closed mode only)
    },
    "eyes": {
        "enabled": True,  # feature flag
        "opening_ratio": 0.72,  # of original eye height, centered on pupil line
        "fissure_offset_mm": -2,  # carve 2mm BELOW eyeball equator
        "replace_dirty_eyeballs": True,  # clean UV spheres + procedural iris
        "build_socket": False,  # socket interior is high-risk; default OFF
    },
    "uv": {
        "min_region": 40,  # dominant-axis segmentation absorption threshold
        "keep_clean_islands": True,  # don't re-unwrap eyes/teeth if already clean
        "max_islands": 40,  # P7 assertion target: uv_islands_after must be <= this
    },
    "bake": {
        "resolution": 2048,
        "channels": ["diffuse"],  # detect from material node trees; add roughness/normal if used
        "samples": 32,  # headless has no timeout; can afford more than MCP's 4-8
        "max_ray_distance": 0.025,
        "margin_px": 16,
    },
    "export": {
        "format": "glb",
        "names": {
            "body": "Body",
            "eye_l": "Eye_L",
            "eye_r": "Eye_R",
            "teeth_u": "Teeth_Upper",
            "teeth_l": "Teeth_Lower",
            "teeth": "Teeth",  # reference_kept mode: existing input teeth part, kept as-is
            "tongue": "Tongue",
        },
        "hierarchy": "parent_under_body",  # parented, NOT joined
    },
    "qa": {
        "render_views": ["front", "back", "left", "right", "top", "three_quarter"],
        "xray_view": True,
        "resolution": 640,
        # p4 assertion: max boundary edges tolerated in body+legs z-bands
        # (tiny unfillable slit fragments are normal on real AI meshes)
        "max_body_boundary_edges": 20,
    },
}

_VALID_UP_AXES = {"auto", "x", "y", "z", "-x", "-y", "-z"}
_VALID_BAKE_CHANNELS = {"diffuse", "roughness", "normal", "metallic", "emission"}
_VALID_EXPORT_FORMATS = {"glb", "gltf"}
_VALID_HIERARCHY = {"parent_under_body"}
_VALID_QA_VIEWS = {"front", "back", "left", "right", "top", "bottom", "three_quarter"}
_VALID_ZONE_TYPES = {"eye_regions", "lip_region"}


class ConfigError(ValueError):
    """Raised for unknown keys or failed validation in a loaded config."""


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML or JSON config, merge over defaults, validate, return it."""
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            raise ConfigError(
                "PyYAML is required to load .yaml config files but is not installed "
                "in this Python environment. Install it into Blender's bundled "
                "Python, e.g.:\n"
                "  <blender_dir>/4.x/python/bin/python3.11 -m pip install pyyaml\n"
                "or pass a .json config instead (same structure)."
            ) from exc
        with path.open("r") as f:
            user_cfg = yaml.safe_load(f) or {}
    elif suffix == ".json":
        with path.open("r") as f:
            user_cfg = json.load(f)
    else:
        raise ConfigError(f"Unsupported config extension {suffix!r} (use .yaml/.yml/.json)")

    if not isinstance(user_cfg, dict):
        raise ConfigError(f"Config root must be a mapping, got {type(user_cfg).__name__}")

    unknown: list[str] = []
    merged = _deep_merge(DEFAULT_CONFIG, user_cfg, "", unknown)
    if unknown:
        raise ConfigError(
            "Unknown config key(s): " + ", ".join(sorted(unknown))
        )

    errors = _validate(merged)
    if errors:
        raise ConfigError("Invalid config:\n  " + "\n  ".join(errors))

    return merged


def _deep_merge(default: dict, user: dict, path: str, unknown: list[str]) -> dict:
    merged = dict(default)
    for key, value in user.items():
        full_path = f"{path}.{key}" if path else key
        if key not in default:
            unknown.append(full_path)
            continue
        default_value = default[key]
        if (
            isinstance(default_value, dict)
            and isinstance(value, dict)
            and full_path not in _OPEN_DICT_PATHS
        ):
            merged[key] = _deep_merge(default_value, value, full_path, unknown)
        else:
            merged[key] = value
    return merged


def _is_power_of_two(n: Any) -> bool:
    return isinstance(n, int) and not isinstance(n, bool) and n > 0 and (n & (n - 1)) == 0


def _is_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _validate(cfg: dict[str, Any]) -> list[str]:
    errors: list[str] = []

    def check(cond: bool, msg: str) -> None:
        if not cond:
            errors.append(msg)

    inp = cfg["input"]
    check(inp["up_axis"] in _VALID_UP_AXES, f"input.up_axis must be one of {_VALID_UP_AXES}")

    weld = cfg["weld"]
    sweep = weld["sweep"]
    check(
        isinstance(sweep, list) and len(sweep) >= 2 and all(_is_number(v) and v > 0 for v in sweep),
        "weld.sweep must be a list of 2+ positive numbers",
    )
    if isinstance(sweep, list) and all(_is_number(v) for v in sweep):
        check(sweep == sorted(sweep), "weld.sweep must be ascending")
    auto_pick = weld["auto_pick"]
    check(isinstance(auto_pick, dict) and "strategy" in auto_pick, "weld.auto_pick.strategy is required")
    mmr = auto_pick.get("max_merge_rate")
    check(_is_number(mmr) and 0 < mmr <= 1, "weld.auto_pick.max_merge_rate must be in (0, 1]")

    hg = cfg["hidden_geometry"]
    check(isinstance(hg["part_rays"], int) and hg["part_rays"] > 0, "hidden_geometry.part_rays must be a positive int")
    check(isinstance(hg["face_rays"], int) and hg["face_rays"] > 0, "hidden_geometry.face_rays must be a positive int")
    check(isinstance(hg["dilate_rings"], int) and hg["dilate_rings"] >= 0, "hidden_geometry.dilate_rings must be a non-negative int")

    hf = cfg["hole_fill"]
    check(isinstance(hf["max_loop_edges"], int) and hf["max_loop_edges"] > 0, "hole_fill.max_loop_edges must be a positive int")
    zones = hf["protected_zones"]
    check(isinstance(zones, list), "hole_fill.protected_zones must be a list")
    if isinstance(zones, list):
        for i, zone in enumerate(zones):
            check(isinstance(zone, dict) and zone.get("type") in _VALID_ZONE_TYPES, f"hole_fill.protected_zones[{i}].type must be one of {_VALID_ZONE_TYPES}")

    mouth = cfg["mouth"]
    check(isinstance(mouth["enabled"], bool), "mouth.enabled must be a bool")
    cr = mouth["cutter_radii"]
    check(isinstance(cr, list) and len(cr) == 3 and all(_is_number(v) and v > 0 for v in cr), "mouth.cutter_radii must be a list of 3 positive numbers")
    check(_is_number(mouth["bag_height"]) and mouth["bag_height"] > 0, "mouth.bag_height must be a positive number")
    check(_is_number(mouth["lip_gap_mm"]) and mouth["lip_gap_mm"] > 0, "mouth.lip_gap_mm must be a positive number")
    trm = mouth["teeth_recess_mm"]
    check(isinstance(trm, list) and len(trm) == 2 and all(_is_number(v) for v in trm) and trm[0] <= trm[1], "mouth.teeth_recess_mm must be an ascending [min, max] pair")

    eyes = cfg["eyes"]
    check(isinstance(eyes["enabled"], bool), "eyes.enabled must be a bool")
    check(_is_number(eyes["opening_ratio"]) and 0 < eyes["opening_ratio"] <= 1, "eyes.opening_ratio must be in (0, 1]")
    check(_is_number(eyes["fissure_offset_mm"]), "eyes.fissure_offset_mm must be a number")
    check(isinstance(eyes["replace_dirty_eyeballs"], bool), "eyes.replace_dirty_eyeballs must be a bool")
    check(isinstance(eyes["build_socket"], bool), "eyes.build_socket must be a bool")

    uv = cfg["uv"]
    check(isinstance(uv["min_region"], int) and uv["min_region"] > 0, "uv.min_region must be a positive int")
    check(isinstance(uv["keep_clean_islands"], bool), "uv.keep_clean_islands must be a bool")
    check(isinstance(uv["max_islands"], int) and uv["max_islands"] > 0, "uv.max_islands must be a positive int")

    bake = cfg["bake"]
    check(_is_power_of_two(bake["resolution"]), "bake.resolution must be a positive power of two")
    channels = bake["channels"]
    check(
        isinstance(channels, list) and len(channels) > 0 and all(c in _VALID_BAKE_CHANNELS for c in channels),
        f"bake.channels must be a non-empty list from {_VALID_BAKE_CHANNELS}",
    )
    check(isinstance(bake["samples"], int) and bake["samples"] > 0, "bake.samples must be a positive int")
    check(_is_number(bake["max_ray_distance"]) and bake["max_ray_distance"] > 0, "bake.max_ray_distance must be a positive number")
    check(isinstance(bake["margin_px"], int) and bake["margin_px"] >= 0, "bake.margin_px must be a non-negative int")

    export = cfg["export"]
    check(export["format"] in _VALID_EXPORT_FORMATS, f"export.format must be one of {_VALID_EXPORT_FORMATS}")
    names = export["names"]
    check(
        isinstance(names, dict) and all(isinstance(k, str) and isinstance(v, str) for k, v in names.items()),
        "export.names must be a dict of str -> str",
    )
    check(export["hierarchy"] in _VALID_HIERARCHY, f"export.hierarchy must be one of {_VALID_HIERARCHY}")

    qa = cfg["qa"]
    views = qa["render_views"]
    check(
        isinstance(views, list) and len(views) > 0 and all(v in _VALID_QA_VIEWS for v in views),
        f"qa.render_views must be a non-empty list from {_VALID_QA_VIEWS}",
    )
    check(isinstance(qa["xray_view"], bool), "qa.xray_view must be a bool")
    check(isinstance(qa["resolution"], int) and qa["resolution"] > 0, "qa.resolution must be a positive int")
    check(
        isinstance(qa["max_body_boundary_edges"], int) and qa["max_body_boundary_edges"] >= 0,
        "qa.max_body_boundary_edges must be a non-negative int",
    )

    return errors
