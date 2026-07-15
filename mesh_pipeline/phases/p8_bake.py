"""P8 -- texture atlas rebake.

Selected-to-active Cycles bake from the `_backup_pre_cleanup` objects (old
geometry, old UVs, original materials) onto the cleaned body's new shared
atlas UVs, per docs/blender-uv-atlas-rebake.md Phase 5 and
docs/uv-atlas-references/snippets.md section 11 ("Atlas bake setup, bake,
rewire").

Not DESTRUCTIVE (geometry is untouched -- only images/materials change) but
still returns metrics per the phase contract.

Steps, mapped onto the task spec:

1. Scan every backup object's *original* materials for TEX_IMAGE nodes and
   trace forward through the node tree to see which Principled BSDF input
   they ultimately feed (Base Color -> diffuse, Roughness -> roughness,
   Normal -> normal, Metallic -> metallic, Emission Color -> emission).
   Intersect with cfg["bake"]["channels"]. No image textures anywhere ->
   skip cleanly.
2. Copy the backups' materials (suffix "_orig") before rewiring anything, so
   the live cleaned-mesh materials can be freely modified afterward without
   corrupting the bake source.
3. Create one atlas image per channel (sRGB for diffuse/emission, Non-Color
   for roughness/normal/metallic) and add a uniquely-named bake-target
   TEX_IMAGE node to every material slot on the body, set active.
4. Cycles, use_selected_to_active, samples/max_ray_distance/margin from cfg;
   DIFFUSE bakes with direct/indirect passes disabled (color-only). Backups
   are unhidden + selected as bake source, body is the sole active target
   (matches the task spec's "active = cleaned body", not a per-object loop
   -- eyes/teeth are not separately baked in this phase).
5. Pack each atlas into the .blend and also save it as a PNG into
   job_dir/output/.
6. Rewire the body's live materials to point at the new atlases and delete
   the now-unused old TEX_IMAGE nodes (backups' "_orig" copies are left
   untouched).

Metrics: channels (list), atlas_stats (dict channel -> {mean, nonzero_ratio}).
"""

from __future__ import annotations

import bpy

from mesh_pipeline.context import PhaseResult, PipelineContext

PHASE_NAME = "p8_bake"
DESTRUCTIVE = False

_BACKUP_SUFFIX = "_orig"
_BAKE_NODE_PREFIX = "BakeTarget_"

# Principled BSDF input socket name(s) each channel maps to (first match
# wins; Blender 4.x renamed "Emission" -> "Emission Color").
_CHANNEL_BSDF_INPUTS: dict[str, tuple[str, ...]] = {
    "diffuse": ("Base Color",),
    "roughness": ("Roughness",),
    "normal": ("Normal",),
    "metallic": ("Metallic",),
    "emission": ("Emission Color", "Emission"),
}

_CHANNEL_COLORSPACE: dict[str, str] = {
    "diffuse": "sRGB",
    "emission": "sRGB",
    "roughness": "Non-Color",
    "normal": "Non-Color",
    "metallic": "Non-Color",
}

# Cycles bake pass type per channel. "normal" and the diffuse-family passes
# are native Cycles bake types; "metallic" has no native pass, so it is
# baked via an EMIT passthrough (the source material's Metallic input is
# temporarily rewired into an Emission shader -- EMIT just captures whatever
# is plugged into Emission, ignoring lighting, which is the standard trick
# for baking an arbitrary scalar/texture input that has no dedicated pass).
_CHANNEL_BAKE_TYPE: dict[str, str] = {
    "diffuse": "DIFFUSE",
    "roughness": "ROUGHNESS",
    "normal": "NORMAL",
    "emission": "EMIT",
    "metallic": "EMIT",
}

_TRACE_MAX_DEPTH = 6


def run(ctx: PipelineContext, cfg: dict) -> PhaseResult:
    ctx.ensure_object_mode()
    bpy.context.view_layer.update()

    notes: list[str] = []
    bake_cfg = cfg["bake"]

    backup_names = sorted(set(ctx.backup_names.values()) & {o.name for o in bpy.data.objects})
    backup_objs = [bpy.data.objects[n] for n in backup_names if bpy.data.objects[n].type == "MESH"]
    if not backup_objs:
        raise RuntimeError(
            "p8_bake: no backup mesh objects found via ctx.backup_names -- p1_backup must "
            "run before p8"
        )

    detected = _detect_channels(backup_objs)
    channels = [c for c in bake_cfg["channels"] if c in detected]

    if not detected:
        return PhaseResult(
            phase=PHASE_NAME,
            status="ok",
            metrics={"channels": [], "atlas_stats": {}},
            notes=["no image textures; bake skipped"],
        )
    if not channels:
        return PhaseResult(
            phase=PHASE_NAME,
            status="ok",
            metrics={"channels": [], "atlas_stats": {}},
            notes=[
                f"image textures found (detected channels={sorted(detected)}) but none "
                f"intersect cfg.bake.channels={bake_cfg['channels']}; bake skipped"
            ],
        )
    notes.append(f"detected channels in backup materials: {sorted(detected)}; baking: {channels}")

    body = ctx.obj("body")
    body_name = body.name

    _copy_backup_materials(backup_objs, notes)

    resolution = int(bake_cfg["resolution"])
    atlas_images = {ch: _make_atlas_image(f"Atlas_{ch}", resolution, _CHANNEL_COLORSPACE[ch]) for ch in channels}

    bake_nodes = _add_bake_target_nodes(body, channels, atlas_images)

    _configure_cycles(bake_cfg)

    was_hidden = _unhide_backups(backup_names)
    try:
        for channel in channels:
            _set_active_bake_nodes(body, channel, bake_nodes)
            restore_fns = []
            if channel == "metallic":
                restore_fns = _rig_emit_passthrough(backup_objs, "metallic")
            try:
                _select_for_bake(backup_names, body_name)
                bake_type = _CHANNEL_BAKE_TYPE[channel]
                use_pass_direct = channel != "diffuse"
                use_pass_indirect = channel != "diffuse"
                scene = bpy.context.scene
                scene.render.bake.use_pass_direct = use_pass_direct
                scene.render.bake.use_pass_indirect = use_pass_indirect
                scene.render.bake.use_pass_color = True
                bpy.ops.object.bake(
                    type=bake_type,
                    use_selected_to_active=True,
                    max_ray_distance=float(bake_cfg["max_ray_distance"]),
                    margin=int(bake_cfg["margin_px"]),
                )
                notes.append(f"baked channel={channel!r} type={bake_type!r} -> Atlas_{channel}")
            finally:
                for restore in restore_fns:
                    restore()
    finally:
        _rehide_backups(backup_names, was_hidden)

    output_dir = ctx.job_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    atlas_stats: dict[str, dict] = {}
    for channel, img in atlas_images.items():
        png_path = output_dir / f"atlas_{channel}.png"
        _save_atlas(img, png_path)
        atlas_stats[channel] = _image_stats(img)
        notes.append(
            f"atlas {channel!r}: saved {png_path} mean={atlas_stats[channel]['mean']:.4f} "
            f"nonzero_ratio={atlas_stats[channel]['nonzero_ratio']:.4f}"
        )

    _rewire_body_materials(body, channels, atlas_images, bake_nodes, notes)

    metrics = {"channels": channels, "atlas_stats": atlas_stats}
    return PhaseResult(phase=PHASE_NAME, status="ok", metrics=metrics, notes=notes)


# ---------------------------------------------------------------------------
# (1) channel detection
# ---------------------------------------------------------------------------


def _detect_channels(backup_objs: list) -> set[str]:
    found: set[str] = set()
    for ob in backup_objs:
        for slot in ob.material_slots:
            mat = slot.material
            if mat is None or mat.node_tree is None:
                continue
            nt = mat.node_tree
            for node in nt.nodes:
                if node.type != "TEX_IMAGE" or node.image is None:
                    continue
                for out in node.outputs:
                    for link in out.links:
                        found |= _trace_channel(link, 0)
    return found


def _trace_channel(link, depth: int) -> set[str]:
    if depth > _TRACE_MAX_DEPTH:
        return set()
    target = link.to_node
    socket_name = link.to_socket.name
    if target.type == "BSDF_PRINCIPLED":
        for channel, socket_names in _CHANNEL_BSDF_INPUTS.items():
            if socket_name in socket_names:
                return {channel}
        return set()
    # Any intermediate node (Normal Map, Separate Color, Mix, Gamma, ...):
    # keep tracing forward through all of its outputs.
    result: set[str] = set()
    for out in target.outputs:
        for nxt in out.links:
            result |= _trace_channel(nxt, depth + 1)
    return result


# ---------------------------------------------------------------------------
# (2) backup material copies
# ---------------------------------------------------------------------------


def _copy_backup_materials(backup_objs: list, notes: list[str]) -> None:
    copied: dict[str, object] = {}
    n = 0
    for ob in backup_objs:
        for slot in ob.material_slots:
            mat = slot.material
            if mat is None:
                continue
            if mat.name.endswith(_BACKUP_SUFFIX):
                continue  # already a copy from a prior p8 run
            if mat.name not in copied:
                c = mat.copy()
                c.name = mat.name + _BACKUP_SUFFIX
                copied[mat.name] = c
                n += 1
            slot.material = copied[mat.name]
    notes.append(f"copied {n} backup material(s) with suffix {_BACKUP_SUFFIX!r}")


# ---------------------------------------------------------------------------
# (3) atlas images + bake-target nodes
# ---------------------------------------------------------------------------


def _make_atlas_image(name: str, size: int, colorspace: str):
    existing = bpy.data.images.get(name)
    if existing is not None:
        bpy.data.images.remove(existing)
    img = bpy.data.images.new(name, size, size, alpha=False)
    img.colorspace_settings.name = colorspace
    return img


def _add_bake_target_nodes(body, channels: list[str], atlas_images: dict) -> dict[str, dict]:
    """Returns {channel: {material_name: node}} -- one persistent node per
    channel per material slot (reused across the bake loop, never renamed
    mid-loop, so step (6)'s rewire can find them again by name).
    """
    bake_nodes: dict[str, dict] = {ch: {} for ch in channels}
    for slot in body.material_slots:
        mat = slot.material
        if mat is None:
            continue
        nt = mat.node_tree
        for channel in channels:
            node_name = f"{_BAKE_NODE_PREFIX}{channel}"
            node = nt.nodes.get(node_name)
            if node is None:
                node = nt.nodes.new("ShaderNodeTexImage")
            node.name = node.label = node_name
            node.image = atlas_images[channel]
            bake_nodes[channel][mat.name] = node
    return bake_nodes


def _set_active_bake_nodes(body, channel: str, bake_nodes: dict) -> None:
    for slot in body.material_slots:
        mat = slot.material
        if mat is None:
            continue
        node = bake_nodes[channel].get(mat.name)
        if node is not None:
            mat.node_tree.nodes.active = node


# ---------------------------------------------------------------------------
# (4) Cycles configuration + bake
# ---------------------------------------------------------------------------


def _configure_cycles(bake_cfg: dict) -> None:
    try:
        bpy.ops.preferences.addon_enable(module="cycles")
    except Exception:
        pass  # already enabled, or bundled by default in this build
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = int(bake_cfg["samples"])
    scene.cycles.use_denoising = False
    try:
        scene.cycles.device = "CPU"
    except Exception:
        pass
    b = scene.render.bake
    b.use_selected_to_active = True
    b.use_cage = False
    b.cage_extrusion = 0.01
    b.max_ray_distance = float(bake_cfg["max_ray_distance"])
    b.margin = int(bake_cfg["margin_px"])


def _unhide_backups(backup_names: list[str]) -> dict[str, tuple[bool, bool]]:
    """Unhide backup objects (+ their collection) for the bake; returns prior state to restore."""
    prior: dict[str, tuple[bool, bool]] = {}
    col = bpy.data.collections.get("_backup_pre_cleanup")
    if col is not None:
        prior["__collection__"] = (col.hide_viewport, col.hide_render)  # type: ignore[assignment]
        col.hide_viewport = False
        col.hide_render = False
    for name in backup_names:
        obj = bpy.data.objects[name]
        prior[name] = (obj.hide_get(), obj.hide_render)
        obj.hide_set(False)
        obj.hide_render = False
    return prior


def _rehide_backups(backup_names: list[str], prior: dict[str, tuple[bool, bool]]) -> None:
    for name in backup_names:
        obj = bpy.data.objects.get(name)
        if obj is None:
            continue
        hv, hr = prior.get(name, (True, True))
        obj.hide_set(hv)
        obj.hide_render = hr
    col = bpy.data.collections.get("_backup_pre_cleanup")
    if col is not None and "__collection__" in prior:
        hv, hr = prior["__collection__"]
        col.hide_viewport = hv
        col.hide_render = hr


def _select_for_bake(backup_names: list[str], body_name: str) -> None:
    bpy.ops.object.select_all(action="DESELECT")
    for name in backup_names:
        bpy.data.objects[name].select_set(True)
    body = bpy.data.objects[body_name]
    body.select_set(True)
    bpy.context.view_layer.objects.active = body


def _find_bsdf(nt):
    return next((n for n in nt.nodes if n.type == "BSDF_PRINCIPLED"), None)


def _find_output_node(nt):
    for n in nt.nodes:
        if n.type == "OUTPUT_MATERIAL" and n.is_active_output:
            return n
    return next((n for n in nt.nodes if n.type == "OUTPUT_MATERIAL"), None)


def _rig_emit_passthrough(backup_objs: list, channel: str):
    """Temporarily reroute each backup material's Metallic input into an
    Emission shader plugged straight into the output's Surface socket, so an
    EMIT bake captures that scalar/texture with no lighting influence.
    Returns a list of zero-arg restore callables.
    """
    restores = []
    socket_names = _CHANNEL_BSDF_INPUTS[channel]
    seen_materials: set[str] = set()
    for ob in backup_objs:
        for slot in ob.material_slots:
            mat = slot.material
            if mat is None or mat.name in seen_materials or mat.node_tree is None:
                continue
            seen_materials.add(mat.name)
            nt = mat.node_tree
            bsdf = _find_bsdf(nt)
            out = _find_output_node(nt)
            if bsdf is None or out is None:
                continue
            socket = next((bsdf.inputs[n] for n in socket_names if n in bsdf.inputs), None)
            if socket is None:
                continue

            src_output = None
            for link in nt.links:
                if link.to_socket == socket:
                    src_output = link.from_socket
                    break

            emit = nt.nodes.new("ShaderNodeEmission")
            if src_output is not None:
                nt.links.new(src_output, emit.inputs["Color"])
            else:
                v = socket.default_value
                if isinstance(v, (int, float)):
                    emit.inputs["Color"].default_value = (v, v, v, 1.0)
                else:
                    emit.inputs["Color"].default_value = (v[0], v[1], v[2], 1.0)

            prior_link = None
            for link in list(nt.links):
                if link.to_node == out and link.to_socket.name == "Surface":
                    prior_link = (link.from_socket, link.to_socket)
                    nt.links.remove(link)
                    break
            nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])

            def _restore(nt=nt, emit=emit, prior_link=prior_link, out=out):
                nt.nodes.remove(emit)
                if prior_link is not None:
                    nt.links.new(*prior_link)

            restores.append(_restore)
    return restores


# ---------------------------------------------------------------------------
# (5) pack + save PNG
# ---------------------------------------------------------------------------


def _save_atlas(img, path) -> None:
    img.filepath_raw = str(path)
    img.file_format = "PNG"
    img.save()
    img.pack()


def _image_stats(img, stride: int = 7) -> dict:
    width, height = img.size
    n_px = width * height
    if n_px == 0:
        return {"mean": 0.0, "nonzero_ratio": 0.0}
    channels = img.channels or 4
    try:
        import numpy as np

        arr = np.empty(n_px * channels, dtype=np.float32)
        img.pixels.foreach_get(arr)
        arr = arr.reshape(-1, channels)[::stride, :3]
        lum = arr.mean(axis=1)
        mean = float(lum.mean()) if lum.size else 0.0
        nonzero_ratio = float((lum > 1e-4).mean()) if lum.size else 0.0
    except ImportError:
        pixels = img.pixels[:]
        total = 0.0
        nonzero = 0
        count = 0
        for i in range(0, n_px, stride):
            base = i * channels
            r, g, b = pixels[base], pixels[base + 1], pixels[base + 2]
            lum = (r + g + b) / 3.0
            total += lum
            if lum > 1e-4:
                nonzero += 1
            count += 1
        mean = total / count if count else 0.0
        nonzero_ratio = nonzero / count if count else 0.0
    return {"mean": mean, "nonzero_ratio": nonzero_ratio}


# ---------------------------------------------------------------------------
# (6) rewire live materials
# ---------------------------------------------------------------------------


def _rewire_body_materials(body, channels: list[str], atlas_images: dict, bake_nodes: dict, notes: list[str]) -> None:
    for slot in body.material_slots:
        mat = slot.material
        if mat is None:
            continue
        nt = mat.node_tree
        bsdf = _find_bsdf(nt)

        keep_names = {f"{_BAKE_NODE_PREFIX}{ch}" for ch in channels}
        for node in list(nt.nodes):
            if node.type == "TEX_IMAGE" and node.name not in keep_names:
                nt.nodes.remove(node)

        if bsdf is None:
            continue
        for channel in channels:
            node = bake_nodes[channel].get(mat.name)
            if node is None:
                continue
            if channel == "normal":
                nmap = nt.nodes.new("ShaderNodeNormalMap")
                nt.links.new(node.outputs["Color"], nmap.inputs["Color"])
                socket_names = _CHANNEL_BSDF_INPUTS["normal"]
                socket = next((bsdf.inputs[n] for n in socket_names if n in bsdf.inputs), None)
                if socket is not None:
                    nt.links.new(nmap.outputs["Normal"], socket)
            else:
                socket_names = _CHANNEL_BSDF_INPUTS[channel]
                socket = next((bsdf.inputs[n] for n in socket_names if n in bsdf.inputs), None)
                if socket is not None:
                    nt.links.new(node.outputs["Color"], socket)
        notes.append(f"rewired material {mat.name!r} to atlases for channels {channels}")
