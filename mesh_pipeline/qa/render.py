"""Headless QA renders: 6 orthographic views + one x-ray-ish view.

No screenshots, no `bpy.ops.view3d.*` (ARCHITECTURE.md forbids view/window-context
operators in headless mode). Cameras are created via the data API and pointed with
`to_track_quat`, then `bpy.ops.render.render(write_still=True)` writes PNGs.

Called as `render.render(ctx, cfg, tag=...)` after every phase (see cli.py
`attempt_qa_renders`), so it must work at ANY point in the pipeline, not just after
p9 -- "deliverable geometry" here means every currently-visible mesh object (backup
copies are excluded because p1_backup hides them: `hide_set(True)` +
`hide_render = True`, per docs/blender-body-mesh-cleanup.md Phase 0 rule 8).
"""

from __future__ import annotations

import math
from pathlib import Path

_VIEW_OFFSETS_XY_Z_UP = {
    # Unit vector from the bbox center to the camera, i.e. "camera is on this
    # side of the subject". Blender convention: front view looks along +Y
    # (camera sits at -Y), matching Numpad-1 Front view.
    "front": (0.0, -1.0, 0.0),
    "back": (0.0, 1.0, 0.0),
    # "left"/"right" label the SIDE OF THE SCENE the camera stands on (not the
    # character's anatomical left/right, which would be mirrored) -- per spec:
    # left = -X, right = +X.
    "left": (-1.0, 0.0, 0.0),
    "right": (1.0, 0.0, 0.0),
    "top": (0.0, 0.0, 1.0),
    "bottom": (0.0, 0.0, -1.0),
}


def _three_quarter_offset():
    from mathutils import Vector

    # "front-left elevated 30 deg": azimuth halfway between front(-Y) and
    # left(-X), then tilted 30 degrees above the horizon.
    horiz = Vector((-1.0, -1.0, 0.0)).normalized()
    elev = math.radians(30.0)
    return Vector((horiz.x * math.cos(elev), horiz.y * math.cos(elev), math.sin(elev)))


def _deliverable_mesh_objects():
    import bpy

    return [
        o
        for o in bpy.data.objects
        if o.type == "MESH" and not o.hide_render and o.visible_get()
    ]


def _world_bbox(objs):
    from mathutils import Vector

    import bpy

    bpy.context.view_layer.update()
    xs, ys, zs = [], [], []
    for o in objs:
        mat = o.matrix_world
        for corner in o.bound_box:
            p = mat @ Vector(corner)
            xs.append(p.x)
            ys.append(p.y)
            zs.append(p.z)
    if not xs:
        return Vector((0.0, 0.0, 0.0)), 1.0
    center = Vector(((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, (min(zs) + max(zs)) / 2))
    radius = max(
        (max(xs) - min(xs)) / 2,
        (max(ys) - min(ys)) / 2,
        (max(zs) - min(zs)) / 2,
        1e-4,
    ) * math.sqrt(3)  # half-diagonal upper bound: safe for any camera axis
    return center, radius


def _add_camera(name: str, location, target, ortho_scale: float, clip_end: float):
    import bpy
    from mathutils import Vector

    cam_data = bpy.data.cameras.new(name)
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = ortho_scale
    cam_data.clip_start = 0.01
    cam_data.clip_end = clip_end
    cam_obj = bpy.data.objects.new(name, cam_data)
    bpy.context.scene.collection.objects.link(cam_obj)
    cam_obj.location = location
    direction = Vector(target) - Vector(location)
    if direction.length < 1e-9:
        direction = Vector((0.0, -1.0, 0.0))
    direction.normalize()
    cam_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    return cam_obj


def _save_scene_settings(scene):
    shading = scene.display.shading
    return {
        "engine": scene.render.engine,
        "res_x": scene.render.resolution_x,
        "res_y": scene.render.resolution_y,
        "filepath": scene.render.filepath,
        "file_format": scene.render.image_settings.file_format,
        "camera": scene.camera,
        "shading_light": shading.light,
        "shading_color_type": shading.color_type,
        "shading_show_xray": shading.show_xray,
        "shading_xray_alpha": shading.xray_alpha,
    }


def _restore_scene_settings(scene, saved: dict) -> None:
    shading = scene.display.shading
    scene.render.engine = saved["engine"]
    scene.render.resolution_x = saved["res_x"]
    scene.render.resolution_y = saved["res_y"]
    scene.render.filepath = saved["filepath"]
    scene.render.image_settings.file_format = saved["file_format"]
    scene.camera = saved["camera"]
    shading.light = saved["shading_light"]
    shading.color_type = saved["shading_color_type"]
    shading.show_xray = saved["shading_show_xray"]
    shading.xray_alpha = saved["shading_xray_alpha"]


def _require_gl() -> None:
    """Fail with a catchable error when no EGL/GL library is present.

    Workbench (and Eevee) rendering without GL does not raise -- it SIGABRTs
    the whole Blender process, which the pipeline runner cannot catch. A
    missing-library preflight turns that into an ordinary exception that
    cli.attempt_qa_renders records as a note instead of killing the job.
    (On headless boxes: `apt install libegl1 libegl-mesa0 libgl1-mesa-dri`
    provides software GL via Mesa.)
    """
    import ctypes.util

    if ctypes.util.find_library("EGL") is None and ctypes.util.find_library("GL") is None:
        raise RuntimeError(
            "No EGL/GL library found; skipping QA renders (Workbench would "
            "abort the process). Install Mesa: libegl1 libegl-mesa0 libgl1-mesa-dri."
        )


def render(ctx, cfg, tag: str) -> list[Path]:
    """Render QA views of every currently-visible deliverable mesh.

    Views come from cfg["qa"]["render_views"] (front/back/left/right/top/
    three_quarter), at cfg["qa"]["resolution"] square, using BLENDER_WORKBENCH
    (deterministic, CPU-headless-safe, no lamp setup needed -- "STUDIO"
    lighting is a fixed matcap-like environment). If cfg["qa"]["xray_view"] is
    set, one additional front-facing render is written with
    `scene.display.shading.show_xray = True` (this IS honored by the
    Workbench render engine, unlike Eevee/Cycles which ignore
    `scene.display.shading` -- verified empirically; no clay/emission-material
    fallback needed).

    Returns the list of written PNG paths. Restores prior render/shading
    settings before returning (this function is called after every phase, not
    just at the end of the pipeline).
    """
    import bpy

    _require_gl()

    scene = bpy.context.scene
    saved = _save_scene_settings(scene)

    out_dir = ctx.job_dir / "qa_renders"
    out_dir.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    temp_objs = []
    try:
        objs = _deliverable_mesh_objects()
        center, radius = _world_bbox(objs)
        dist = radius * 4.0 + 1.0
        ortho_scale = radius * 2.0 * 1.10  # 10% margin around the bounding sphere

        resolution = int(cfg["qa"]["resolution"])
        scene.render.engine = "BLENDER_WORKBENCH"
        scene.render.resolution_x = resolution
        scene.render.resolution_y = resolution
        scene.render.image_settings.file_format = "PNG"
        shading = scene.display.shading
        shading.light = "STUDIO"
        shading.color_type = "MATERIAL"
        shading.show_xray = False

        views = list(cfg["qa"]["render_views"])

        def _offset(view_name: str):
            if view_name == "three_quarter":
                return _three_quarter_offset()
            from mathutils import Vector

            return Vector(_VIEW_OFFSETS_XY_Z_UP[view_name])

        for view in views:
            offset = _offset(view)
            location = center + offset * dist
            cam = _add_camera(f"_QA_CAM_{tag}_{view}", location, center, ortho_scale, dist * 4.0)
            temp_objs.append(cam)
            scene.camera = cam
            out_path = out_dir / f"{tag}_{view}.png"
            scene.render.filepath = str(out_path)
            bpy.ops.render.render(write_still=True)
            written.append(out_path)

        if cfg["qa"].get("xray_view"):
            xray_view = "front" if "front" in views else views[0]
            offset = _offset(xray_view)
            location = center + offset * dist
            cam = _add_camera(f"_QA_CAM_{tag}_xray", location, center, ortho_scale, dist * 4.0)
            temp_objs.append(cam)
            scene.camera = cam
            shading.show_xray = True
            shading.xray_alpha = 0.3
            out_path = out_dir / f"{tag}_xray.png"
            scene.render.filepath = str(out_path)
            bpy.ops.render.render(write_still=True)
            written.append(out_path)
            shading.show_xray = False

        return written
    finally:
        for cam in temp_objs:
            cam_data = cam.data
            bpy.data.objects.remove(cam, do_unlink=True)
            if cam_data is not None and cam_data.users == 0:
                bpy.data.cameras.remove(cam_data)
        _restore_scene_settings(scene, saved)
