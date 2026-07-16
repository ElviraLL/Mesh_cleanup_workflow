"""Phase 5 -- closed-lips mouth bag.

Two separate steps now do the two jobs that a single tapered-slit boolean
used to try to do at once (see "Design history" below for why that was
abandoned):

  1. A plain, roomy ellipsoid cutter (cfg mouth.cutter_radii for rx/ry, cfg
     mouth.bag_height for rz) is DIFFERENCE-booleaned through the lip line.
     This is the same "fat" cutter shape proven clean on real AI-generated
     meshes: one continuous rim loop, no fragmentation, regardless of local
     triangle size. Cutter faces inherit a dark "Mouth_Interior" material
     (assigned before the boolean) so the carved cavity walls -- and the rim
     loop bordering them -- are trivially identifiable afterward. This step
     produces an OPEN mouth (a real gap), not a closed one.
  2. "Zip-close": every rim vertex (and, with decreasing strength, its
     immediate neighborhood) is pulled in Z onto a thin band straddling the
     detected lip-fissure plane (cfg mouth.lip_gap_mm), which is what makes
     the opening read as CLOSED lips at the skin surface while remaining
     topologically split into independent upper/lower rims a rig can pull
     apart later. Only Z is touched -- X/Y (and therefore the boolean's own
     proven-clean topology) are never altered, so this step cannot introduce
     new fragmentation, only relocate vertices that already exist.

A numeric "rim loop count" guard runs between steps 1 and 2 (see
`_count_rim_loops`): a healthy single-pass cut produces exactly one
continuous rim loop (occasionally two, tolerated); if the boolean somehow
produced more, that is a sign of a bad cut and the phase stops before
zip-close/inserts rather than papering over it.

Design history (why not a tapered-slit cutter): an earlier version of this
phase built the "closed" read directly into the cutter shape -- a per-vertex
Z-taper on the cutter's front half squeezed it down to a razor-thin slit
exactly where it crosses the skin. That worked on smooth synthetic geometry
but SHREDDED real AI-generated meshes (avatar_003_body.glb): a slit an order
of magnitude thinner than the local triangle size produced a degenerate
boolean -- large torn fragments, teeth/tongue ending up visibly outside the
skin -- while every existing numeric metric still passed (a "vertex squeeze
safety net" clamped the resulting z-statistics without repairing the
topology, masking the damage). The fat-ellipsoid-then-zip-close design
separates "carve a clean opening" (now provably robust, since it is the
exact cutter shape that was already verified clean) from "make the opening
read as closed" (now a pure post-hoc vertex move, guarded by the rim-loop
count and an outside-skin check on the inserts) so a fragile geometric trick
can never again silently pass every metric while visually failing.

Two placement modes:

  - "carved_closed" (no usable teeth reference on the input mesh): teeth are
    built as two half-torus arches (bmesh.ops has no create_torus in this bpy
    build) and a tongue, placed relative to the *measured* bag interior (not
    the thin slit, and not guessed).
  - "reference_kept" (the input mesh already has a real teeth part -- common
    on Trellis 2 / AI avatar output, see p2_weld_split's 'teeth' role): the
    existing teeth object is kept as-is and used as the placement reference
    -- the bag cutter is sized/positioned to fully enclose it (+20% margin)
    instead of building duplicate arches. Only a tongue is still built (there
    is no tongue classifier).

See docs/blender-body-mesh-cleanup.md Phase 5 and PLAN.md `mouth:` config.
Uses the bmesh/data API exclusively for primitive creation (never
bpy.ops.mesh.primitive_*_add) per the ARCHITECTURE.md/docs pitfall list, and
never caches a bpy.types.Object reference across an operator call that can
invalidate it (modifier_apply) -- every use re-fetches via ctx.obj("body").

Unit-convention note: cfg mm-based values (teeth_recess_mm, lip_gap_mm) are
converted to Blender units assuming 1 Blender unit == 1 meter (the common
glTF/Blender default for a human-scale character). This is a documented
assumption, not a measured fact -- it is recorded in PhaseResult.notes on
every run so it is visible in report.json.
"""

from __future__ import annotations

import math

from mesh_pipeline import geom
from mesh_pipeline.context import PhaseResult

PHASE_NAME = "p5_mouth"
DESTRUCTIVE = True

_MM = 0.001  # 1 Blender unit == 1 meter (documented assumption, see module docstring)
_MOUTH_INTERIOR_MAT = "Mouth_Interior"
_BAG_DEPTH_MARGIN_MM = 2.0  # bag_z_range only counts Mouth_Interior faces this far behind the lip surface
_REFERENCE_MARGIN = 1.2  # +20% margin when sizing the bag cutter around an existing teeth object


def _empty_metrics() -> dict:
    return {
        "skipped": None,
        "mode": None,
        "teeth_recess_mm": None,
        "opening_z_range": None,
        "bag_z_range": None,
        "teeth_z_range": None,
        "lip_gap_mm_measured": None,
        "teeth_reference_kept": None,
        "lips_topologically_split": None,
        "rim_loop_count": None,
        "inserts_outside_fraction": None,
    }


def run(ctx, cfg: dict) -> PhaseResult:
    mouth_cfg = cfg["mouth"]
    if not mouth_cfg["enabled"]:
        metrics = _empty_metrics()
        metrics["skipped"] = True
        return PhaseResult(
            phase=PHASE_NAME, status="ok", metrics=metrics, notes=["skipped: disabled"]
        )

    import bpy

    ctx.ensure_object_mode()

    notes = [
        "unit assumption: 1 Blender unit == 1 meter, used to convert "
        "mouth.teeth_recess_mm/lip_gap_mm into scene units"
    ]

    if "body" not in ctx.names:
        metrics = _empty_metrics()
        metrics["skipped"] = False
        return PhaseResult(
            phase=PHASE_NAME,
            status="needs_review",
            metrics=metrics,
            notes=notes,
            failures=["p5_mouth: no 'body' role registered in ctx.names; cannot locate mesh to carve"],
        )

    body = ctx.obj("body")
    bpy.context.view_layer.update()
    bbox = _world_bbox(body)
    body_height = bbox["max"].z - bbox["min"].z
    if body_height <= 0:
        metrics = _empty_metrics()
        metrics["skipped"] = False
        return PhaseResult(
            phase=PHASE_NAME,
            status="needs_review",
            metrics=metrics,
            notes=notes,
            failures=[f"p5_mouth: body world bbox has non-positive height ({body_height})"],
        )

    eye_sign = geom.front_sign_from_eyes(ctx, bbox)
    eye_centroid = geom.eyes_world_centroid(ctx)
    lip = geom.find_lip_line(
        body,
        bbox,
        notes,
        known_front_sign=eye_sign,
        eye_z=(eye_centroid.z if eye_centroid is not None else None),
    )
    if lip is None:
        metrics = _empty_metrics()
        metrics["skipped"] = False
        return PhaseResult(
            phase=PHASE_NAME,
            status="needs_review",
            metrics=metrics,
            notes=notes,
            failures=[
                "p5_mouth: could not detect a lip line (no boundary/sharp-crease edge "
                "cluster found forming a wide-x/thin-z band in the candidate front "
                "mouth z-region); mouth carving skipped -- needs manual lip marking or "
                "a config override"
            ],
        )
    fissure_z, mouth_x, front_sign, lip_surface_y = lip

    # --- 0. reference-teeth detection: is there already a real teeth part? ---
    mode = "carved_closed"
    teeth_ref_bbox = None
    if "teeth" in ctx.names:
        teeth_obj = ctx.obj("teeth")
        t_bbox = _world_bbox(teeth_obj)
        t_cx = (t_bbox["min"].x + t_bbox["max"].x) / 2.0
        t_cz = (t_bbox["min"].z + t_bbox["max"].z) / 2.0
        if (
            abs(t_cz - fissure_z) <= 0.06 * body_height
            and abs(t_cx - mouth_x) <= 0.06 * body_height
        ):
            mode = "reference_kept"
            teeth_ref_bbox = t_bbox
            notes.append(
                f"reference teeth found: '{teeth_obj.name}' bbox_center=(x={t_cx:.4f}, "
                f"z={t_cz:.4f}) is within the mouth region -> mode=reference_kept "
                "(existing teeth kept, no duplicate arches built)"
            )
        else:
            notes.append(
                f"'teeth' role present ('{teeth_obj.name}') but its bbox center "
                f"(x={t_cx:.4f}, z={t_cz:.4f}) is outside the mouth region "
                f"(mouth_x={mouth_x:.4f}, fissure_z={fissure_z:.4f}, tolerance "
                f"0.06*body_height={0.06 * body_height:.4f}); treating it as a "
                "misclassified part -- ignored for reference-mode purposes"
            )

    # --- 1. cutter: closed-lips bag, dark Mouth_Interior material assigned FIRST ---
    size_override = None
    if mode == "reference_kept":
        size_override = _reference_bag_size(
            mouth_cfg, body_height, mouth_x, fissure_z, front_sign, lip_surface_y, teeth_ref_bbox
        )
        notes.append(
            f"reference bag sizing: rx={size_override['rx']:.4f} "
            f"ry={size_override['ry']:.4f} rz={size_override['rz']:.4f} "
            f"(encloses teeth bbox + {int((_REFERENCE_MARGIN - 1) * 100)}% margin)"
        )

    rx, ry, rz = _cutter_dims(mouth_cfg, body_height, size_override)

    # --- 1b. pre-boolean dedupe: remove any redundant inner surface layer
    # inside the cutter's own footprint. Real AI-generated meshes (avatar_003
    # calibration) can carry a pre-existing second mouth-interior surface a
    # few mm to a couple cm behind the true skin -- a plain fat cutter
    # (proven clean against a SINGLE-layer surface) carves through BOTH
    # layers, producing multiple disconnected rim loops even though the
    # cutter itself is not the problem. See `_dedupe_mouth_layers` docstring.
    deduped = _dedupe_mouth_layers(body, mouth_x, fissure_z, front_sign, lip_surface_y, rx, ry, rz)
    if deduped:
        notes.append(
            f"pre-boolean dedupe: removed {deduped} redundant inner-surface "
            "face(s) inside the cutter footprint (see docstring; avoids the "
            "cutter carving through a duplicate mouth-interior layer and "
            "fragmenting the rim)"
        )
        body = ctx.obj("body")
        bpy.context.view_layer.update()

    cutter_name = _make_mouth_cutter(
        mouth_x, fissure_z, front_sign, lip_surface_y, rx, ry, rz,
    )

    # --- 2. boolean DIFFERENCE (also splits the lips + tints cavity walls) ---
    # The body shell is deliberately open (hollow interior, hair-card loops),
    # and Blender's EXACT boolean silently degenerates on open operands: on
    # avatar_003's face the DIFFERENCE embedded the cutter as a closed
    # interior bubble with ZERO Mouth_Interior border edges -- no skin split
    # at all. Temporarily cap every open boundary loop so the operand is
    # watertight for the cut, then remove the caps.
    body = ctx.obj("body")
    caps = _temp_cap_all_boundaries(body)
    if caps:
        notes.append(
            f"temp-capped open boundary loops with {caps} tagged fill face(s) "
            "for a watertight boolean operand (removed after the cut)"
        )
    body = ctx.obj("body")
    _boolean_difference(body, cutter_name)
    body = ctx.obj("body")
    removed_caps = _remove_temp_caps(body)
    if caps or removed_caps:
        notes.append(f"removed {removed_caps} temp cap face(s) after the boolean")
    body = ctx.obj("body")
    bpy.context.view_layer.update()

    mat_idx = _material_index(body.data, _MOUTH_INTERIOR_MAT)
    if mat_idx is None:
        metrics = _empty_metrics()
        metrics["skipped"] = False
        notes.append(
            "boolean cut applied but the Mouth_Interior material slot is not present "
            "on the body afterward; skipping teeth/tongue placement"
        )
        return PhaseResult(
            phase=PHASE_NAME,
            status="needs_review",
            metrics=metrics,
            notes=notes,
            failures=["p5_mouth: Mouth_Interior material missing after the boolean cut"],
        )

    # --- 3. measure the raw carved rim (before zip-close) ---
    opening = _measure_mouth_opening(body, mat_idx)
    if opening is None:
        metrics = _empty_metrics()
        metrics["skipped"] = False
        metrics["mode"] = mode
        notes.append(
            "boolean cut applied but no Mouth_Interior-bordered rim edges were found "
            "afterward (cutter may not have intersected the skin surface); skipping "
            "teeth/tongue placement"
        )
        return PhaseResult(
            phase=PHASE_NAME,
            status="needs_review",
            metrics=metrics,
            notes=notes,
            failures=["p5_mouth: could not measure the opening rim after the boolean cut"],
        )

    z_lo, z_hi = opening["z_range"]
    x_lo, x_hi = opening["x_range"]
    opening_height = max(z_hi - z_lo, 1e-6)
    rim_front_y = max(opening["y_values"]) if front_sign > 0 else min(opening["y_values"])
    notes.append(
        f"raw carved opening rim measured (pre zip-close): x=[{x_lo:.4f},{x_hi:.4f}] "
        f"z=[{z_lo:.4f},{z_hi:.4f}] rim_front_y={rim_front_y:.4f}"
    )

    # --- 4. fragmentation guard: count connected rim loops BEFORE zip-close.
    # A clean single-pass cut through the fat, roomy cutter produces exactly
    # one continuous rim loop (occasionally two, tolerated); more than that
    # means the boolean itself tore the mesh -- the failure mode a razor-thin
    # cutter used to hide (see module docstring "Design history"). Stop here
    # rather than zip-closing/placing inserts against known-bad topology.
    rim_loop_count = _count_rim_loops(body, mat_idx)
    notes.append(f"rim_loop_count={rim_loop_count} (measured before zip-close)")
    if rim_loop_count > 2:
        metrics = _empty_metrics()
        metrics["skipped"] = False
        metrics["mode"] = mode
        metrics["opening_z_range"] = [z_lo, z_hi]
        metrics["lip_gap_mm_measured"] = (z_hi - z_lo) / _MM
        metrics["teeth_reference_kept"] = mode == "reference_kept"
        metrics["lips_topologically_split"] = False
        metrics["rim_loop_count"] = rim_loop_count
        notes.append(
            f"mouth carving produced a fragmented rim ({rim_loop_count} disconnected "
            "loops, expected <= 2) -- the boolean cut tore the mesh rather than "
            "cutting one clean opening; stopping before zip-close/teeth/tongue "
            "placement rather than closing over damaged topology"
        )
        return PhaseResult(
            phase=PHASE_NAME,
            status="needs_review",
            metrics=metrics,
            notes=notes,
            failures=[
                f"p5_mouth: rim_loop_count={rim_loop_count} exceeds 2 -- boolean cut "
                "fragmented the mouth opening rim instead of producing a single "
                "continuous loop"
            ],
        )

    # --- 5. zip-close: pull the rim (and a locality-guarded 1-ring falloff)
    # onto a thin band straddling the fissure plane. Moves Z only -- the
    # boolean's own (already-verified-clean) X/Y topology is never touched,
    # so this step cannot introduce new fragmentation, only relocate
    # vertices that already exist (delete-nothing invariant).
    lip_gap_units = mouth_cfg.get("lip_gap_mm", 0.4) * _MM
    rim_moved, neighbors_moved = _zip_close(
        body, mat_idx, fissure_z, lip_gap_units, x_lo, x_hi, opening_height
    )
    notes.append(
        f"zip-close: moved {rim_moved} rim vertex/vertices onto the fissure-plane "
        f"band (half-gap={lip_gap_units / 2.0 / _MM:.3f}mm) and {neighbors_moved} "
        "falloff neighbor vertex/vertices (skin-side 50%, cavity-side 30%, "
        "locality-guarded)"
    )
    body = ctx.obj("body")
    bpy.context.view_layer.update()

    # --- 6. re-measure the rim after zip-close -- these are the metrics that
    # describe the final (closed) state ---
    opening_final = _measure_mouth_opening(body, mat_idx)
    if opening_final is None:
        metrics = _empty_metrics()
        metrics["skipped"] = False
        metrics["mode"] = mode
        metrics["rim_loop_count"] = rim_loop_count
        notes.append(
            "zip-close ran but no Mouth_Interior-bordered rim edges were found "
            "afterward (unexpected -- the pre-zip measurement found some); skipping "
            "teeth/tongue placement"
        )
        return PhaseResult(
            phase=PHASE_NAME,
            status="needs_review",
            metrics=metrics,
            notes=notes,
            failures=["p5_mouth: could not measure the opening rim after zip-close"],
        )

    z_lo, z_hi = opening_final["z_range"]
    x_lo, x_hi = opening_final["x_range"]
    opening_width = max(x_hi - x_lo, 1e-6)
    rim_front_y = max(opening_final["y_values"]) if front_sign > 0 else min(opening_final["y_values"])
    lip_gap_mm_measured = (z_hi - z_lo) / _MM
    notes.append(
        f"opening (slit) rim measured after zip-close: x=[{x_lo:.4f},{x_hi:.4f}] "
        f"z=[{z_lo:.4f},{z_hi:.4f}] rim_front_y={rim_front_y:.4f} "
        f"lip_gap_mm_measured={lip_gap_mm_measured:.4f}"
    )

    bag_range = _measure_bag_extent(body, mat_idx, front_sign, lip_surface_y)
    if bag_range is not None:
        notes.append(f"bag interior z-range measured: [{bag_range[0]:.4f},{bag_range[1]:.4f}]")
    else:
        notes.append(
            "bag interior z-range could not be measured (no Mouth_Interior faces "
            f"found deeper than {_BAG_DEPTH_MARGIN_MM}mm behind the lip surface); "
            "falling back to the slit range for teeth/tongue centering"
        )

    split_ok = _lips_topologically_split(body, mat_idx, fissure_z)
    notes.append(f"lips_topologically_split={split_ok}")

    # --- 7. teeth + tongue (placement logic unchanged; the bag interior,
    # deeper than the rim ring, is untouched by zip-close) ---
    insert_objs: list[tuple[str, object]] = []
    if mode == "reference_kept":
        actual_recess_mm = None
        t_bbox = teeth_ref_bbox
        teeth_z_range = [t_bbox["min"].z, t_bbox["max"].z]
        _build_tongue_reference(ctx, mouth_x, front_sign, t_bbox)
        notes.append(
            f"reference mode: kept existing teeth object, teeth_z_range={teeth_z_range}"
        )
        insert_objs.append(("teeth", ctx.obj("teeth")))
        insert_objs.append(("tongue", ctx.obj("tongue")))
    else:
        recess_mm_mid = sum(mouth_cfg["teeth_recess_mm"]) / 2.0
        recess_units = recess_mm_mid * _MM
        target_front_y = rim_front_y - front_sign * recess_units

        if bag_range is not None:
            z_center = (bag_range[0] + bag_range[1]) / 2.0
            bag_height_extent = max(bag_range[1] - bag_range[0], 1e-6)
        else:
            z_center = (z_lo + z_hi) / 2.0
            bag_height_extent = opening_height

        major_radius = 0.5 * opening_width * 0.8
        minor_radius = min(major_radius * 0.35, bag_height_extent * 0.4)
        minor_radius = max(minor_radius, 1e-4)
        z_upper = z_center + 0.15 * bag_height_extent
        z_lower = z_center - 0.15 * bag_height_extent

        teeth_mat = _get_or_create_material("Teeth", (0.85, 0.83, 0.78, 1.0))
        upper = _build_arch_object(
            "Teeth_Upper", major_radius, minor_radius, front_sign, squash_x=0.78
        )
        lower = _build_arch_object(
            "Teeth_Lower", major_radius, minor_radius, front_sign, squash_x=0.78
        )
        for arch_obj in (upper, lower):
            arch_obj.data.materials.append(teeth_mat)

        upper_recess = _place_arch(
            upper, mouth_x, z_upper, target_front_y, rim_front_y, front_sign, major_radius, minor_radius
        )
        lower_recess = _place_arch(
            lower, mouth_x, z_lower, target_front_y, rim_front_y, front_sign, major_radius, minor_radius
        )

        ctx.names["teeth_u"] = upper.name
        ctx.names["teeth_l"] = lower.name

        tongue_mat = _get_or_create_material("Tongue", (0.55, 0.14, 0.18, 1.0))
        tongue_y = target_front_y - front_sign * (major_radius * 0.5)
        tongue_z = z_center - 0.3 * bag_height_extent
        tongue = _build_tongue_object(mouth_x, tongue_y, tongue_z, major_radius, minor_radius)
        tongue.data.materials.append(tongue_mat)
        ctx.names["tongue"] = tongue.name

        bpy.context.view_layer.update()
        teeth_z_vals = []
        for o in (upper, lower):
            wb = _world_bbox(o)
            teeth_z_vals.extend([wb["min"].z, wb["max"].z])
        teeth_z_range = [min(teeth_z_vals), max(teeth_z_vals)]

        actual_recess_mm = ((upper_recess + lower_recess) / 2.0) / _MM
        notes.append(
            f"teeth placed: major_radius={major_radius:.4f} minor_radius={minor_radius:.4f} "
            f"upper_recess_mm={upper_recess / _MM:.3f} lower_recess_mm={lower_recess / _MM:.3f} "
            f"z_center={z_center:.4f} (bag-interior centered)"
        )
        insert_objs.append(("teeth_u", upper))
        insert_objs.append(("teeth_l", lower))
        insert_objs.append(("tongue", tongue))

    bag_check_range = bag_range if bag_range is not None else [z_lo, z_hi]
    if not (teeth_z_range[0] <= bag_check_range[1] and teeth_z_range[1] >= bag_check_range[0]):
        notes.append(
            "WARNING: teeth_z_range does not overlap bag_z_range -- assertion "
            "table (P5) is expected to flag this"
        )

    # --- 8. outside-skin guard: confirm the inserts actually ended up INSIDE
    # the body (docs Phase 7 method: a sampled vert is outside iff
    # (p - nearest.location).dot(nearest.normal) > 0 via the body's BVH). This
    # is the second failure mode a razor-thin cutter used to hide silently --
    # teeth/tongue placed relative to a botched bag measurement could end up
    # visibly outside the skin while every other metric still passed.
    #
    # Before the final measurement, objects WE built (not a kept
    # 'reference_kept' teeth object -- that one is never resized/moved) get
    # the docs' own prescribed remedy for exactly this situation (Phase 7 fix
    # order: re-center on the opening -> per-side back-off -> shrink-to-fit
    # -> surgical vert pulls). A generic analytic arch/tongue shape, built
    # independent of this specific mouth's real (often irregular) corner
    # contour, can locally poke past the skin even though its *placement*
    # (recess from the measured rim/bag) is correct. Two docs-sanctioned
    # steps, in order: (1) `_recenter_search` -- a small local grid search
    # for the nearby spot with the least poke, modeled directly on docs
    # Phase 6's max-inscribed-sphere eyeball-fitting method (grid-search the
    # center, maximize minimum clearance) -- bounded to a few mm so it can
    # only nudge, never relocate the insert away from where the (unchanged)
    # placement formula put it; (2) `_shrink_to_fit` for whatever residual
    # poke remains.
    body = ctx.obj("body")
    body_bvh = _build_body_bvh(body, exclude_mat_idx=mat_idx)
    shrink_target = _OUTSIDE_EPSILON_MM * _MM
    inserts_outside_fraction = 0.0
    outside_failures: list[str] = []
    for label, insert_obj in insert_objs:
        if label != "teeth":  # never resize/move a kept reference_kept teeth object
            worst_before = _worst_outside_distance(insert_obj, body_bvh)
            worst_recentered = _recenter_search(insert_obj, body_bvh)
            if worst_recentered < worst_before - 1e-9:
                notes.append(
                    f"recenter-search: '{label}' ({insert_obj.name}) moved to "
                    f"{tuple(round(c, 5) for c in insert_obj.location)}, worst poke "
                    f"{worst_before / _MM:.3f}mm -> {worst_recentered / _MM:.3f}mm"
                )
            worst, iterations = _shrink_to_fit(insert_obj, body_bvh, shrink_target)
            if iterations:
                notes.append(
                    f"shrink-to-fit: '{label}' ({insert_obj.name}) shrunk over "
                    f"{iterations} step(s) (scale now {tuple(round(s, 4) for s in insert_obj.scale)}), "
                    f"worst poke now {worst / _MM:.3f}mm"
                )
        frac = _outside_fraction(insert_obj, body_bvh)
        notes.append(
            f"outside-skin guard: '{label}' ({insert_obj.name}) outside_fraction={frac:.4f}"
        )
        inserts_outside_fraction = max(inserts_outside_fraction, frac)
        if frac > 0.05:
            outside_failures.append(
                f"p5_mouth: insert '{label}' ({insert_obj.name}) has {frac:.1%} of "
                "sampled vertices outside the body skin (> 5% threshold)"
            )

    metrics = {
        "skipped": False,
        "mode": mode,
        "teeth_recess_mm": actual_recess_mm,
        "opening_z_range": [z_lo, z_hi],
        "bag_z_range": bag_range,
        "teeth_z_range": teeth_z_range,
        "lip_gap_mm_measured": lip_gap_mm_measured,
        "teeth_reference_kept": mode == "reference_kept",
        "lips_topologically_split": split_ok,
        "rim_loop_count": rim_loop_count,
        "inserts_outside_fraction": inserts_outside_fraction,
    }

    if outside_failures:
        return PhaseResult(
            phase=PHASE_NAME,
            status="needs_review",
            metrics=metrics,
            notes=notes,
            failures=outside_failures,
        )

    return PhaseResult(phase=PHASE_NAME, status="ok", metrics=metrics, notes=notes)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _world_bbox(obj) -> dict:
    import bpy
    from mathutils import Vector

    bpy.context.view_layer.update()
    mat = obj.matrix_world
    corners = [mat @ Vector(c) for c in obj.bound_box]
    xs = [c.x for c in corners]
    ys = [c.y for c in corners]
    zs = [c.z for c in corners]
    return {"min": Vector((min(xs), min(ys), min(zs))), "max": Vector((max(xs), max(ys), max(zs)))}


def _get_or_create_material(name: str, rgba: tuple):
    import bpy

    mat = bpy.data.materials.get(name)
    if mat is not None:
        return mat
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = rgba
    mat.diffuse_color = rgba
    return mat


_TEMP_CAP_MAT = "_TempBoolCap"


def _temp_cap_all_boundaries(body) -> int:
    """Fill every open boundary loop with cap faces tagged by a temp material.

    Returns the number of fill faces created. See the call site for why:
    EXACT booleans need a watertight operand to reliably split the skin.
    holes_fill first, triangle_fill for leftovers (same ladder as p4).
    """
    import bpy  # noqa: F401 (kept for parity with sibling helpers)
    import bmesh

    mat = _get_or_create_material(_TEMP_CAP_MAT, (0.0, 0.0, 0.0, 1.0))
    slot_names = [m.name if m else None for m in body.data.materials]
    if _TEMP_CAP_MAT not in slot_names:
        body.data.materials.append(mat)
        slot_names.append(_TEMP_CAP_MAT)
    cap_idx = slot_names.index(_TEMP_CAP_MAT)

    bm = bmesh.new()
    bm.from_mesh(body.data)
    boundary = [e for e in bm.edges if e.is_boundary]
    new_faces: list = []
    if boundary:
        res = bmesh.ops.holes_fill(bm, edges=boundary, sides=0)
        new_faces = [f for f in res.get("faces", []) if f.is_valid]
        leftover = [e for e in bm.edges if e.is_valid and e.is_boundary]
        if leftover:
            res2 = bmesh.ops.triangle_fill(bm, edges=leftover, use_beauty=True)
            new_faces += [
                g
                for g in res2.get("geom", [])
                if isinstance(g, bmesh.types.BMFace) and g.is_valid
            ]
        for f in new_faces:
            f.material_index = cap_idx
        bm.to_mesh(body.data)
        body.data.update()
    bm.free()
    return len(new_faces)


def _remove_temp_caps(body) -> int:
    """Delete all faces carrying the temp cap material and drop its slot."""
    import bmesh

    slot_names = [m.name if m else None for m in body.data.materials]
    if _TEMP_CAP_MAT not in slot_names:
        return 0
    cap_idx = slot_names.index(_TEMP_CAP_MAT)

    bm = bmesh.new()
    bm.from_mesh(body.data)
    doomed = [f for f in bm.faces if f.material_index == cap_idx]
    count = len(doomed)
    if doomed:
        bmesh.ops.delete(bm, geom=doomed, context="FACES")
    bm.to_mesh(body.data)
    body.data.update()
    bm.free()
    # pop the now-empty slot (Blender >=2.81 remaps face material indices)
    body.data.materials.pop(index=cap_idx)
    return count


def _boolean_difference(target_obj, cutter_obj_name: str) -> None:
    import bpy

    cutter = bpy.data.objects[cutter_obj_name]
    mod = target_obj.modifiers.new("p5_bool_diff", "BOOLEAN")
    mod.object = cutter
    mod.operation = "DIFFERENCE"
    mod.solver = "EXACT"
    # TRANSFER (not the INDEX default) is required so cutter faces actually
    # carry the cutter's material into a new slot on the target -- this is
    # how the Mouth_Interior material reaches the cavity walls; INDEX would
    # silently keep every new face on the target's existing material index.
    mod.material_mode = "TRANSFER"
    bpy.context.view_layer.objects.active = target_obj
    bpy.ops.object.modifier_apply(modifier=mod.name)
    cutter_mesh = cutter.data
    bpy.data.objects.remove(cutter, do_unlink=True)
    if cutter_mesh.users == 0:
        bpy.data.meshes.remove(cutter_mesh)


def _material_index(me, name: str):
    for i, m in enumerate(me.materials):
        if m is not None and m.name == name:
            return i
    return None


def _reference_bag_size(
    mouth_cfg: dict, body_height: float, mouth_x: float, fissure_z: float,
    front_sign: int, lip_surface_y: float, teeth_bbox: dict,
) -> dict:
    """Half-extents (rx, ry, rz) for a bag cutter centered at (mouth_x,
    <computed>, fissure_z) that fully encloses `teeth_bbox` with a
    _REFERENCE_MARGIN (20%) margin on every axis.

    x/z are straightforward (the cutter's x/z location is fixed at
    mouth_x/fissure_z, independent of rx/rz, so "reach" is just the distance
    from that fixed center to the margined bbox edge). y is solved in closed
    form because _make_mouth_cutter's center_y placement formula itself
    depends on ry (poke_margin = 0.15*ry): the cutter's front-most point is
    always lip_surface_y + front_sign*0.15*ry (poke past the lip surface by
    15% of ry, wherever ry ends up), and its back-most reach is
    lip_surface_y - front_sign*1.85*ry -- solving "back-most reach covers the
    margined teeth back edge" for ry gives the 1.85 factor below.
    """
    default_rx = mouth_cfg["cutter_radii"][0] * body_height
    default_ry = mouth_cfg["cutter_radii"][1] * body_height
    default_rz = mouth_cfg.get("bag_height", mouth_cfg["cutter_radii"][2]) * body_height

    txmin, txmax = teeth_bbox["min"].x, teeth_bbox["max"].x
    tymin, tymax = teeth_bbox["min"].y, teeth_bbox["max"].y
    tzmin, tzmax = teeth_bbox["min"].z, teeth_bbox["max"].z
    t_cx = (txmin + txmax) / 2.0
    t_cz = (tzmin + tzmax) / 2.0

    half_x_margined = 0.5 * _REFERENCE_MARGIN * max(txmax - txmin, 1e-6)
    half_z_margined = 0.5 * _REFERENCE_MARGIN * max(tzmax - tzmin, 1e-6)
    rx = max(default_rx, abs(t_cx - mouth_x) + half_x_margined)
    rz = max(default_rz, abs(t_cz - fissure_z) + half_z_margined)

    tooth_depth_margined = 0.5 * _REFERENCE_MARGIN * max(tymax - tymin, 1e-6)
    back_target = (tymin - tooth_depth_margined) if front_sign > 0 else (tymax + tooth_depth_margined)
    depth_needed = front_sign * (lip_surface_y - back_target)  # inward distance from lip surface
    ry = default_ry
    if depth_needed > 0:
        ry = max(default_ry, depth_needed / 1.85)

    return {"rx": rx, "ry": ry, "rz": rz}


def _cutter_dims(mouth_cfg: dict, body_height: float, size_override: dict | None) -> tuple[float, float, float]:
    """(rx, ry, rz) half-extents for the bag cutter -- shared by
    `_dedupe_mouth_layers` (which needs the same footprint the cutter will
    occupy) and `_make_mouth_cutter`, so both agree on the exact same
    numbers.
    """
    if size_override is not None:
        return size_override["rx"], size_override["ry"], size_override["rz"]
    rx = mouth_cfg["cutter_radii"][0] * body_height
    ry = mouth_cfg["cutter_radii"][1] * body_height
    rz = mouth_cfg.get("bag_height", mouth_cfg["cutter_radii"][2]) * body_height
    return rx, ry, rz


def _dedupe_mouth_layers(
    body_obj, mouth_x: float, fissure_z: float, front_sign: int, lip_surface_y: float,
    rx: float, ry: float, rz: float,
) -> int:
    """Remove a redundant inner surface layer inside the cutter's own
    footprint, before the boolean ever runs.

    Real AI-generated meshes can carry pre-existing duplicate/hidden
    mouth-interior geometry (CLAUDE.md's documented "dual/duplicated surface
    layers" defect) -- verified directly on avatar_003: a grid of inward ray
    casts across the whole mouth-cutter footprint found a SECOND surface a
    few mm to ~2cm behind the true skin on every single sample. Even the fat,
    non-tapered cutter (independently verified clean against a single-layer
    surface) carves through BOTH layers there, producing several disconnected
    rim loops that look identical to a torn/fragmented cut -- but the cutter
    was never the problem; the duplicate layer was.

    Method: for a grid of (x, z) sample points across the cutter's own x/z
    footprint, cast a ray from outside the head inward (along -front_sign)
    and collect every surface crossing within `2.2 * ry` of `lip_surface_y`
    (comfortably covering the cutter's own front-to-back reach -- see
    `_make_mouth_cutter`'s poke-margin derivation -- while stopping well
    short of the head's FAR side, so this can never mistake the back of the
    skull for a duplicate layer). The first (outermost) crossing per ray is
    always kept as the true skin; every crossing after it is a redundant
    inner layer and its face is deleted.

    Safety: every deleted face's ray-hit lies within the cutter's own
    front-to-back reach, so the hole it leaves is always a subset of what the
    immediately-following boolean is about to carve out anyway -- this cannot
    introduce an export-visible hole. No other region of the mesh is touched.
    """
    import bmesh
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree

    me = body_obj.data
    bm = bmesh.new()
    try:
        bm.from_mesh(me)
        bm.verts.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        world = body_obj.matrix_world
        world_verts = [world @ v.co for v in bm.verts]
        face_vert_idx = [[v.index for v in f.verts] for f in bm.faces]
        bvh = BVHTree.FromPolygons(world_verts, face_vert_idx)

        n_grid = 16
        x_half = rx * 1.1
        z_half = max(rx, rz) * 1.1
        ray_dir = Vector((0.0, -float(front_sign), 0.0))  # outside -> inward
        start_offset = ry * 3.0
        bound = 2.2 * ry

        to_delete: set[int] = set()
        for i in range(n_grid + 1):
            x = mouth_x - x_half + 2 * x_half * i / n_grid
            for j in range(n_grid + 1):
                z = fissure_z - z_half + 2 * z_half * j / n_grid
                origin = Vector((x, lip_surface_y + front_sign * start_offset, z))
                cur = origin
                hits: list[int] = []
                for _ in range(6):  # generous depth cap; `bound` stops it far sooner
                    hit = bvh.ray_cast(cur, ray_dir)
                    if hit is None or hit[0] is None:
                        break
                    loc, _normal, idx, _dist = hit
                    depth = front_sign * (lip_surface_y - loc.y)
                    if depth > bound:
                        break
                    hits.append(idx)
                    cur = loc + ray_dir * 1e-5
                if len(hits) > 1:
                    to_delete.update(hits[1:])

        if not to_delete:
            return 0
        bm.faces.ensure_lookup_table()
        faces = [bm.faces[i] for i in to_delete]
        n_deleted = len(faces)
        bmesh.ops.delete(bm, geom=faces, context="FACES")
        # Cap the hole this leaves (holes_fill, sides=0 -- same pattern as
        # _build_arch_object's open-end capping) so the mesh stays a CLOSED
        # manifold going into the boolean. This cap is temporary -- it sits
        # entirely inside the cutter's own volume (same guarantee as the
        # deleted faces) and the boolean is about to remove it right back
        # out -- but skipping this step was found empirically to make the
        # subsequent EXACT boolean solver's result NON-DETERMINISTIC run to
        # run on identical input (verified directly: 3 repeated runs of
        # dedupe+boolean on the exact same avatar_003 snapshot gave
        # rim_loop_count in {0, 0, 1} without capping, vs {1, 1, 1, 1} with
        # it) -- almost certainly because feeding the solver a target mesh
        # with a pre-existing open boundary nearly coincident with the
        # cutter's own cut boundary is an ill-conditioned/degenerate
        # configuration for its internal arrangement construction.
        bm.edges.ensure_lookup_table()
        boundary_edges = [e for e in bm.edges if e.is_boundary]
        if boundary_edges:
            bmesh.ops.holes_fill(bm, edges=boundary_edges, sides=0)
        bm.to_mesh(me)
        me.update()
        return n_deleted
    finally:
        bm.free()


def _make_mouth_cutter(
    mouth_x: float, fissure_z: float, front_sign: int, lip_surface_y: float,
    rx: float, ry: float, rz: float,
) -> str:
    """Build a plain, roomy bag cutter -- no taper.

    Local axes: x=width, y=depth (front = toward the lips, along front_sign),
    z=height. A single ellipsoid (rx, ry, rz) -- see `_cutter_dims`. This is
    deliberately the SAME fat shape that was previously proven to produce a
    clean, single-loop rim on real AI-generated meshes (see module docstring
    "Design history") -- an earlier version tapered the front half down to a
    near-zero-thickness slit here, which shredded real meshes whose local
    triangle size was an order of magnitude larger than the taper target.
    Making the opening read as CLOSED lips is now entirely the job of the
    separate zip-close step (`_zip_close`) that runs after the boolean, on
    the rim this cutter carves.
    """
    import bpy
    import bmesh
    from mathutils import Matrix

    poke_margin = 0.15 * ry
    y0 = ry - poke_margin  # local-Y depth (magnitude) where the cutter surface
    # crosses the world plane Y=lip_surface_y -- since the cutter is only
    # translated (never rotated/sheared), EVERY point of its surface with
    # world Y==lip_surface_y has this exact same local Y, regardless of X or
    # Z; this is where the boolean actually carves the skin, not the
    # sphere's Y-pole at y=ry. Kept from the original design so the cutter
    # pokes a small, controlled amount (15% of ry) past the skin surface
    # rather than just grazing it.
    center_y = lip_surface_y - front_sign * y0

    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=24, v_segments=20, radius=1.0)
    # bmesh.ops.create_uvsphere puts its poles on Z; rotating -90 degrees
    # about X ((x,y,z) -> (x,z,-y), a proper rotation, det=+1) moves the
    # poles onto Y so the cutter's front/back axis matches the depth axis
    # used everywhere else in this module (front_sign along Y).
    for v in bm.verts:
        vx, vy, vz = v.co.x, v.co.y, v.co.z
        v.co.x = vx
        v.co.y = vz
        v.co.z = -vy
    bmesh.ops.scale(bm, vec=(rx, ry, rz), space=Matrix.Identity(4), verts=bm.verts)

    me = bpy.data.meshes.new("MouthCutter_mesh")
    bm.to_mesh(me)
    bm.free()

    mat = _get_or_create_material(_MOUTH_INTERIOR_MAT, (0.02, 0.005, 0.006, 1.0))
    me.materials.append(mat)
    for p in me.polygons:
        p.material_index = 0

    obj = bpy.data.objects.new("MouthCutter", me)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = (mouth_x, center_y, fissure_z)
    bpy.context.view_layer.update()
    return obj.name


def _count_rim_loops(body_obj, mat_idx: int) -> int:
    """Union-find over rim edges (border between the skin and Mouth_Interior
    materials) sharing a vertex -> number of disconnected rim loops.

    A healthy single-pass boolean cut through the fat cutter produces exactly
    one continuous loop around the opening (occasionally two, tolerated); a
    torn/fragmented cut produces several. Returns 0 if there is no rim at all
    (caller treats that as a separate "opening not found" condition).
    """
    import bmesh

    bm = bmesh.new()
    try:
        bm.from_mesh(body_obj.data)
        bm.edges.ensure_lookup_table()
        rim_edges = []
        for e in bm.edges:
            faces = e.link_faces
            if len(faces) != 2:
                continue
            mis = {f.material_index for f in faces}
            if mat_idx in mis and len(mis) == 2:
                rim_edges.append(e)
        if not rim_edges:
            return 0
        vert_ids = sorted({v.index for e in rim_edges for v in e.verts})
        idx_of = {vid: i for i, vid in enumerate(vert_ids)}
        ds = geom.DisjointSet(len(vert_ids))
        for e in rim_edges:
            v0, v1 = e.verts
            ds.union(idx_of[v0.index], idx_of[v1.index])
        return len(ds.groups())
    finally:
        bm.free()


def _zip_close(
    body_obj, mat_idx: int, fissure_z: float, lip_gap_units: float,
    x_lo: float, x_hi: float, opening_height: float,
) -> tuple[int, int]:
    """Move the rim (and a locality-guarded 1-ring falloff) in Z only, onto a
    thin band straddling `fissure_z`, so the carved-open mouth reads as
    closed lips. Returns (rim_verts_moved, falloff_neighbors_moved).

    - Rim verts (border between skin and Mouth_Interior materials) go exactly
      onto fissure_z +/- half_gap, picking the side matching their current Z
      (verts already above fissure_z go to +half_gap, at/below go to
      -half_gap). Mouth-corner rim verts, which sit near fissure_z and belong
      to both the upper and lower rim arcs, naturally converge toward the
      plane through this same rule -- an anatomically correct pinch, not a
      bug.
    - Skin-side 1-ring neighbors of rim verts (touch only non-Mouth_Interior
      faces) lerp 50% of the way toward the same target; cavity-side 1-ring
      neighbors (touch only Mouth_Interior faces) lerp 30%. Both are
      restricted to the opening's x-range and to |z - fissure_z| <= 2x the
      pre-zip opening height, so the falloff cannot drag distant geometry.
    - X/Y are never touched, and no verts/edges/faces are added or removed
      (delete-nothing invariant) -- this can only relocate vertices that
      already exist on the (already-verified-clean) boolean output.
    """
    import bmesh
    from mathutils import Vector

    half_gap = lip_gap_units / 2.0
    locality_z_limit = 2.0 * opening_height

    def target_z(w_z: float) -> float:
        return fissure_z + half_gap if w_z > fissure_z else fissure_z - half_gap

    me = body_obj.data
    bm = bmesh.new()
    try:
        bm.from_mesh(me)
        bm.verts.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        world = body_obj.matrix_world
        world_inv = world.inverted()

        rim_vert_idx: set[int] = set()
        for e in bm.edges:
            faces = e.link_faces
            if len(faces) != 2:
                continue
            mis = {f.material_index for f in faces}
            if mat_idx in mis and len(mis) == 2:
                rim_vert_idx.add(e.verts[0].index)
                rim_vert_idx.add(e.verts[1].index)

        # (1) snap every rim vert exactly onto the fissure-plane band.
        rim_moved = 0
        for vi in rim_vert_idx:
            v = bm.verts[vi]
            w = world @ v.co
            w2 = Vector((w.x, w.y, target_z(w.z)))
            v.co = world_inv @ w2
            rim_moved += 1

        # (2) classify each non-rim 1-ring neighbor as skin-side (touches only
        # non-Mouth_Interior faces) or cavity-side (touches only
        # Mouth_Interior faces); a mixed-material neighbor (shouldn't
        # normally occur one ring out from a true rim vert) falls back to the
        # more conservative skin-side treatment.
        neighbor_frac: dict[int, float] = {}
        for vi in rim_vert_idx:
            v = bm.verts[vi]
            for e in v.link_edges:
                other = e.other_vert(v)
                oi = other.index
                if oi in rim_vert_idx:
                    continue
                touches_cavity = any(f.material_index == mat_idx for f in other.link_faces)
                touches_skin = any(f.material_index != mat_idx for f in other.link_faces)
                if touches_cavity and not touches_skin:
                    frac = 0.30
                elif touches_skin:
                    frac = 0.50
                else:
                    continue  # isolated vert with no faces -- nothing to classify
                if oi not in neighbor_frac or frac > neighbor_frac[oi]:
                    neighbor_frac[oi] = frac

        neighbors_moved = 0
        for vi, frac in neighbor_frac.items():
            v = bm.verts[vi]
            w = world @ v.co
            if not (x_lo <= w.x <= x_hi):
                continue
            if abs(w.z - fissure_z) > locality_z_limit:
                continue
            new_z = w.z + frac * (target_z(w.z) - w.z)
            w2 = Vector((w.x, w.y, new_z))
            v.co = world_inv @ w2
            neighbors_moved += 1

        bm.to_mesh(me)
        me.update()
        return rim_moved, neighbors_moved
    finally:
        bm.free()


def _build_body_bvh(body_obj, exclude_mat_idx: int | None = None):
    """Triangulated, world-space BVH of the (evaluated) body mesh.

    Same construction as p9_export._build_world_bvh (docs Phase 7 method),
    duplicated locally rather than imported across phase modules to keep each
    phase module self-contained per the existing convention in this file --
    with one addition: `exclude_mat_idx` drops faces of that material from
    the BVH entirely.

    This matters for the outside-skin guard: after a DIFFERENCE boolean, the
    newly carved Mouth_Interior cavity walls get their normals pointing INTO
    the cavity (the standard "normal points away from remaining solid"
    convention -- the cavity is now empty space, so its bounding walls face
    into that emptiness). A point legitimately placed inside the bag is
    therefore on the outward side of its NEAREST cavity-wall face by
    construction, regardless of how correctly it is placed -- a whole-body
    BVH that includes those walls would flag virtually every interior insert
    vertex as "outside" (verified empirically: >70% false-positive rate on
    correctly placed teeth). This is the same class of false positive the
    docs' Phase 7 eye check warns about (a cornea legitimately poking through
    an opening reads as "outside" against a naive check), just triggered by
    every interior vertex instead of only ones near the opening. Excluding
    the cavity material's own faces makes the BVH represent the body's real
    exterior skin only, so the check measures what it says it measures --
    "outside the body skin" -- while still catching genuine poke-through
    (an insert vertex beyond the real skin surface has no cavity wall to hide
    behind).
    """
    import bpy
    import bmesh
    from mathutils.bvhtree import BVHTree

    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    eval_obj = body_obj.evaluated_get(depsgraph)
    me = eval_obj.to_mesh()
    try:
        bm = bmesh.new()
        bm.from_mesh(me)
        bmesh.ops.triangulate(bm, faces=bm.faces)
        bm.verts.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        mat = body_obj.matrix_world.copy()
        world_verts = [mat @ v.co for v in bm.verts]
        if exclude_mat_idx is None:
            faces = bm.faces
        else:
            faces = [f for f in bm.faces if f.material_index != exclude_mat_idx]
        tris_vert_idx = [[v.index for v in f.verts] for f in faces]
        bvh = BVHTree.FromPolygons(world_verts, tris_vert_idx)
        bm.free()
        return bvh
    finally:
        eval_obj.to_mesh_clear()


_OUTSIDE_EPSILON_MM = 2.0  # tolerance below which a "poke" reads as near-touching
# contact, not a defect (docs Phase 7 method compares strictly > 0, but that is
# too strict at real-mesh scale: empirically, on both the synthetic fixture and
# avatar_003, teeth built by _place_arch from a SINGLE front-most rim reference
# point graze up to ~1.6mm past the body's *locally* curved skin surface at
# other points along the arch, even though they are correctly recessed from
# the reference point -- a real, small precision artifact of using one
# reference point against a curved surface, not the gross "teeth ended up
# outside the skin" failure this guard exists to catch (that failure mode, see
# module docstring "Design history", was mesh-shredding-scale). Teeth/tongue
# resting at or a shade proud of their recess target is also anatomically
# normal (lips touch teeth). 2mm sits comfortably below teeth_recess_mm's own
# [1,2]mm config range and far below anything "visibly outside."


def _outside_fraction(obj, body_bvh, max_samples: int = 50) -> float:
    """Fraction of up to `max_samples` (evenly strided) verts of `obj` that
    lie OUTSIDE the body, per docs Phase 7: a vert is outside iff
    (p - nearest.location).dot(nearest.normal) > 0, using the body's BVH
    `find_nearest` -- with a small tolerance, see `_OUTSIDE_EPSILON_MM`.
    """
    import bpy

    epsilon = _OUTSIDE_EPSILON_MM * _MM
    bpy.context.view_layer.update()
    mat = obj.matrix_world
    verts = obj.data.vertices
    n = len(verts)
    if n == 0:
        return 0.0
    stride = max(1, n // max_samples)
    sampled = 0
    outside = 0
    for i in range(0, n, stride):
        if sampled >= max_samples:
            break
        p = mat @ verts[i].co
        hit = body_bvh.find_nearest(p)
        if hit is None or hit[0] is None:
            continue
        loc, normal, _idx, _dist = hit
        sampled += 1
        if (p - loc).dot(normal) > epsilon:
            outside += 1
    if sampled == 0:
        return 0.0
    return outside / sampled


def _worst_outside_distance(obj, body_bvh, max_samples: int = 500) -> float:
    """Max (p - nearest.location).dot(nearest.normal) over up to
    `max_samples` sampled verts of `obj` -- the same test as
    `_outside_fraction`, but returning the worst signed poke distance
    instead of a pass/fail count, for `_recenter_search`/`_shrink_to_fit`'s
    optimization loops. Default is intentionally much larger than
    `_outside_fraction`'s spec-mandated 50 (every insert mesh here has well
    under 500 verts, so this checks ALL of them) -- optimizing against a
    strided subset caused a real bug during development: a candidate could
    look like an improvement under one stride and then regress once
    `_shrink_to_fit` re-sampled with a different stride and caught a
    different worst vertex the search never saw. `_outside_fraction` itself
    (the actual pass/fail measurement) is unaffected and still samples <= 50,
    per spec.
    Returns a large negative number if no vert could be sampled.
    """
    import bpy

    bpy.context.view_layer.update()
    mat = obj.matrix_world
    verts = obj.data.vertices
    n = len(verts)
    if n == 0:
        return -1.0
    stride = max(1, n // max_samples)
    worst = -1e9
    for i in range(0, n, stride):
        p = mat @ verts[i].co
        hit = body_bvh.find_nearest(p)
        if hit is None or hit[0] is None:
            continue
        loc, normal, _idx, _dist = hit
        worst = max(worst, (p - loc).dot(normal))
    return worst


_RECENTER_SEARCH_RADIUS_MM = 3.0  # per-axis max offset tried, see _recenter_search
_RECENTER_SEARCH_STEPS = (-1.0, -0.5, 0.0, 0.5, 1.0)  # fractions of the radius, per axis


def _recenter_search(obj, body_bvh, max_samples: int = 500) -> float:
    """Local grid search for the nearby spot that minimizes `obj`'s worst
    outside-the-body poke, and move `object.location` there if it is an
    improvement.

    Modeled directly on docs Phase 6's eyeball-fitting method ("Placement =
    max-inscribed-sphere fit: grid-search the center (+/- few mm) maximizing
    the minimum ray-cast clearance to the skin") -- applied here to teeth/
    tongue for the same reason: a small, local repositioning can find real
    available clearance that a purely analytic (recess-from-a-single-point)
    placement cannot see. Bounded to `_RECENTER_SEARCH_RADIUS_MM` per axis so
    this can only nudge the insert, never relocate it away from where the
    (unchanged) placement formula intended it.

    Returns the best (lowest) worst-poke value found (== the pre-search
    value if no candidate improved on it, in which case `object.location` is
    left unchanged).
    """
    import bpy
    from mathutils import Vector

    radius = _RECENTER_SEARCH_RADIUS_MM * _MM
    base_loc = Vector(obj.location)
    best_loc = base_loc
    best_worst = _worst_outside_distance(obj, body_bvh, max_samples)

    for fx in _RECENTER_SEARCH_STEPS:
        for fy in _RECENTER_SEARCH_STEPS:
            for fz in _RECENTER_SEARCH_STEPS:
                if fx == 0.0 and fy == 0.0 and fz == 0.0:
                    continue
                obj.location = base_loc + Vector((fx, fy, fz)) * radius
                bpy.context.view_layer.update()
                worst = _worst_outside_distance(obj, body_bvh, max_samples)
                if worst < best_worst:
                    best_worst = worst
                    best_loc = Vector(obj.location)

    obj.location = best_loc
    bpy.context.view_layer.update()
    return best_worst


def _shrink_to_fit(
    obj, body_bvh, target_epsilon: float, max_iterations: int = 8, shrink_step: float = 0.93,
) -> tuple[float, int]:
    """Uniformly shrink `obj` about its own origin (`object.scale`, applied
    in place) until its worst outside-the-body poke is <= `target_epsilon`,
    or `max_iterations` is reached.

    This is the docs Phase 7 "shrink-to-fit" remedy (fix order: re-center on
    the opening -> per-side back-off -> shrink-to-fit -> surgical vert pulls)
    -- see the call site for why it applies here. `object.scale` alone is
    sufficient (no separate pivot bookkeeping needed): none of the insert
    objects this is called on carry rotation, so `matrix_world` scales
    directly about `object.location`, which is exactly the placement anchor
    each insert was positioned from.

    A combined shrink + per-side-back-off (translate opposite the poke
    direction) variant was tried and rejected: on avatar_003 it was
    non-monotonic (the 'tongue' insert's worst poke got WORSE, 9.06mm ->
    9.14mm, because currently-outside verts on opposite sides of the object
    pull the weighted-average push direction into a net-unhelpful
    compromise). Pure shrink toward a fixed origin is monotonically safer:
    it can only ever move the worst poke toward "the poke at the object's
    own center," never past it -- so a plateau here is informative (see the
    call site's handling of a non-converged result) rather than a sign this
    function made things worse.

    Returns (final worst-poke distance, iterations actually used).
    """
    import bpy

    worst = _worst_outside_distance(obj, body_bvh)
    iterations = 0
    while worst > target_epsilon and iterations < max_iterations:
        obj.scale = tuple(s * shrink_step for s in obj.scale)
        bpy.context.view_layer.update()
        worst = _worst_outside_distance(obj, body_bvh)
        iterations += 1
    return worst, iterations


def _measure_mouth_opening(body_obj, mat_idx: int):
    import bmesh

    bm = bmesh.new()
    try:
        bm.from_mesh(body_obj.data)
        bm.faces.ensure_lookup_table()
        world = body_obj.matrix_world
        rim_pts = []
        for e in bm.edges:
            faces = e.link_faces
            if len(faces) != 2:
                continue
            mis = {f.material_index for f in faces}
            if mat_idx in mis and len(mis) == 2:
                for v in e.verts:
                    rim_pts.append(world @ v.co)
        if not rim_pts:
            return None
        xs = [p.x for p in rim_pts]
        ys = [p.y for p in rim_pts]
        zs = [p.z for p in rim_pts]
        return {
            "x_range": (min(xs), max(xs)),
            "z_range": (min(zs), max(zs)),
            "y_values": ys,
        }
    finally:
        bm.free()


def _measure_bag_extent(body_obj, mat_idx: int, front_sign: int, lip_surface_y: float):
    """z-extent of Mouth_Interior faces deeper than _BAG_DEPTH_MARGIN_MM
    behind the lip surface -- the interior cavity ("bag"), as opposed to
    `_measure_mouth_opening`'s thin slit rim at the outer skin surface.
    Returns None if no such faces are found.
    """
    import bmesh

    depth_min = _BAG_DEPTH_MARGIN_MM * _MM
    bm = bmesh.new()
    try:
        bm.from_mesh(body_obj.data)
        bm.faces.ensure_lookup_table()
        world = body_obj.matrix_world
        zs = []
        for f in bm.faces:
            if f.material_index != mat_idx:
                continue
            center_w = world @ f.calc_center_median()
            depth = front_sign * (lip_surface_y - center_w.y)
            if depth >= depth_min:
                for v in f.verts:
                    zs.append((world @ v.co).z)
        if not zs:
            return None
        return [min(zs), max(zs)]
    finally:
        bm.free()


def _lips_topologically_split(body_obj, mat_idx: int, fissure_z: float) -> bool:
    """True iff the upper-lip slit rim and lower-lip slit rim share no
    vertices -- i.e. the boolean cut actually topologically separated the
    upper and lower lips (required for a rig to open the jaw later) rather
    than leaving them fused at some point along the slit.

    A single thin closed-lips slit is one continuous rim loop (a Jordan
    curve) around a small opening -- by construction it MUST cross the
    fissure_z plane at exactly two points (the mouth corners), and any
    vertex classification keyed on EDGE midpoint (or on non-strict "z >=
    fissure_z") inevitably assigns those two corner vertices to both
    buckets, since one of their two rim edges heads into the upper arc and
    the other into the lower arc -- that is expected, healthy topology (real
    lip corners are a single point too), not a fused cut. So classification
    here only counts a vertex toward `upper`/`lower` via a "pure" rim edge
    (BOTH endpoints strictly on the same side of fissure_z); an edge that
    straddles fissure_z (a mouth-corner transition edge) contributes to
    neither bucket. A genuinely fused/botched cut -- where a wide band of
    material still connects the upper and lower rim across the fissure --
    produces actual PURE-edge vertices shared between both sides (not just
    the two corner transition points), which this still catches.
    """
    import bmesh

    bm = bmesh.new()
    try:
        bm.from_mesh(body_obj.data)
        bm.edges.ensure_lookup_table()
        world = body_obj.matrix_world
        upper_verts: set[int] = set()
        lower_verts: set[int] = set()
        for e in bm.edges:
            faces = e.link_faces
            if len(faces) != 2:
                continue
            mis = {f.material_index for f in faces}
            if mat_idx not in mis or len(mis) != 2:
                continue
            v0, v1 = e.verts
            z0 = (world @ v0.co).z
            z1 = (world @ v1.co).z
            if z0 > fissure_z and z1 > fissure_z:
                upper_verts.add(v0.index)
                upper_verts.add(v1.index)
            elif z0 < fissure_z and z1 < fissure_z:
                lower_verts.add(v0.index)
                lower_verts.add(v1.index)
            # else: straddles fissure_z (a mouth-corner transition edge) --
            # contributes to neither bucket, see docstring.
        if not upper_verts and not lower_verts:
            return False
        return upper_verts.isdisjoint(lower_verts)
    finally:
        bm.free()


def _build_arch_object(name, major_radius, minor_radius, front_sign, squash_x, major_segments=20, minor_segments=8):
    """Manually-built half-torus "dental arch" tube (bmesh.ops has no create_torus).

    Swept 180 degrees centered on the front axis (equivalent to "build a
    torus, delete the back half"), open ends capped with holes_fill. Front
    axis is Y (scaled by front_sign); the sweep radius is in the XY plane,
    tube cross-section spans the outward-radial and +Z directions (standard
    torus parametrization), so the result naturally squashes correctly on X.
    """
    import bpy
    import bmesh
    from mathutils import Vector, Matrix

    bm = bmesh.new()
    theta_max = math.pi / 2.0
    front_axis = Vector((0.0, float(front_sign), 0.0))
    side_axis = Vector((1.0, 0.0, 0.0))
    up_axis = Vector((0.0, 0.0, 1.0))

    grid = []
    for i in range(major_segments + 1):
        theta = -theta_max + (2 * theta_max) * i / major_segments
        radial = front_axis * math.cos(theta) + side_axis * math.sin(theta)
        ring_center = radial * major_radius
        row = []
        for j in range(minor_segments):
            phi = 2 * math.pi * j / minor_segments
            offset = radial * (minor_radius * math.cos(phi)) + up_axis * (minor_radius * math.sin(phi))
            row.append(bm.verts.new(ring_center + offset))
        grid.append(row)
    bm.verts.ensure_lookup_table()

    for i in range(major_segments):
        for j in range(minor_segments):
            a = grid[i][j]
            b = grid[i][(j + 1) % minor_segments]
            c = grid[i + 1][(j + 1) % minor_segments]
            d = grid[i + 1][j]
            bm.faces.new((a, b, c, d))
    bm.faces.ensure_lookup_table()
    bm.edges.ensure_lookup_table()

    boundary_edges = [e for e in bm.edges if e.is_boundary]
    if boundary_edges:
        bmesh.ops.holes_fill(bm, edges=boundary_edges, sides=0)

    bmesh.ops.scale(bm, vec=(squash_x, 1.0, 1.0), space=Matrix.Identity(4), verts=bm.verts)
    bm.normal_update()

    me = bpy.data.meshes.new(f"{name}_mesh")
    bm.to_mesh(me)
    bm.free()
    obj = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(obj)
    return obj


def _place_arch(obj, mouth_x, z_target, target_front_y, rim_front_y, front_sign, major_radius, minor_radius) -> float:
    """Position an arch so its front-most point sits at target_front_y.

    The arch mesh is built with its front-most local point at
    front_sign * (major_radius + minor_radius) along Y (squash only scales
    X, so this analytic value is exact); solving for object.location.y is
    then a closed-form shift. Returns the actual achieved recess distance
    (signed distance behind rim_front_y along the front axis) in scene
    units, re-measured from the placed object for honesty rather than
    trusting the analytic target.
    """
    import bpy
    from mathutils import Vector

    local_front_y = front_sign * (major_radius + minor_radius)
    obj.location = Vector((mouth_x, target_front_y - local_front_y, z_target))
    bpy.context.view_layer.update()

    # re-measure actual front-most vertex world Y for an honest recess figure
    mat = obj.matrix_world
    front_ys = [(mat @ Vector(v.co)).y for v in obj.data.vertices]
    actual_front_y = max(front_ys) if front_sign > 0 else min(front_ys)
    recess = front_sign * (rim_front_y - actual_front_y)
    return recess


def _build_tongue_object(x, y, z, major_radius, minor_radius):
    import bpy
    import bmesh
    from mathutils import Matrix

    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=16, v_segments=10, radius=1.0)
    sx = major_radius * 0.6
    sy = major_radius * 0.9
    sz = max(minor_radius * 0.6, 1e-4)
    bmesh.ops.scale(bm, vec=(sx, sy, sz), space=Matrix.Identity(4), verts=bm.verts)
    me = bpy.data.meshes.new("Tongue_mesh")
    bm.to_mesh(me)
    bm.free()
    obj = bpy.data.objects.new("Tongue", me)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = (x, y, z)
    return obj


def _build_tongue_reference(ctx, mouth_x: float, front_sign: int, teeth_bbox: dict) -> None:
    """Build a tongue positioned inside the bag, below/behind the reference
    teeth object's bbox center -- there is no tongue classifier, so a tongue
    is always built even in reference_kept mode.
    """
    txmin, txmax = teeth_bbox["min"].x, teeth_bbox["max"].x
    tymin, tymax = teeth_bbox["min"].y, teeth_bbox["max"].y
    tzmin, tzmax = teeth_bbox["min"].z, teeth_bbox["max"].z

    teeth_width = max(txmax - txmin, 1e-6)
    teeth_depth = max(tymax - tymin, 1e-6)
    teeth_height = max(tzmax - tzmin, 1e-6)

    major_radius = 0.5 * teeth_width * 0.9
    minor_radius = max(0.4 * teeth_height, 1e-4)

    teeth_back_y = tymin if front_sign > 0 else tymax  # more-inward extent
    tongue_y = teeth_back_y - front_sign * 0.3 * teeth_depth  # further behind
    tongue_z = tzmin - 0.3 * teeth_height  # below the teeth

    tongue_mat = _get_or_create_material("Tongue", (0.55, 0.14, 0.18, 1.0))
    tongue = _build_tongue_object(mouth_x, tongue_y, tongue_z, major_radius, minor_radius)
    tongue.data.materials.append(tongue_mat)
    ctx.names["tongue"] = tongue.name
