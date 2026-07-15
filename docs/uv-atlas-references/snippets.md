# Code snippets (run via blender:execute_blender_code)

Each block is self-contained. Adapt object/material names. Run one phase per call.

## Contents
1. Per-object inspection report
2. Topology diagnostics + UV island counter (union-find)
3. Weld-distance sweep
4. Backup collection
5. Cleanup pass: weld / degenerate / wire / flaps / fins / holes
6. Boundary-loop histogram
7. Targeted despike
8. Join
9. 6-axis segmentation + region absorption + seam marking
10. Seam-based unwrap + pack
11. Atlas bake setup, bake, rewire

---

## 1. Per-object inspection report

```python
import bpy, json
report = {}
for obj in bpy.data.objects:
    if obj.type != 'MESH': continue
    me = obj.data
    report[obj.name] = {
        "verts": len(me.vertices), "faces": len(me.polygons),
        "parent": obj.parent.name if obj.parent else None,
        "modifiers": [(m.name, m.type) for m in obj.modifiers],
        "shape_keys": len(me.shape_keys.key_blocks) if me.shape_keys else 0,
        "uv_layers": [uv.name for uv in me.uv_layers],
        "materials": [ms.material.name if ms.material else None for ms in obj.material_slots],
        "vertex_groups": len(obj.vertex_groups),
        "scale": tuple(round(s,4) for s in obj.scale),
    }
print(json.dumps(report, indent=1))
```

## 2. Topology diagnostics + UV island counter

The UV island counter is a union-find over faces whose shared edges are welded in
UV space. There is no built-in property for this; reuse it whenever you need an
island count (before/after comparisons).

```python
import bpy, bmesh, json

def uv_island_count(me):
    bm = bmesh.new(); bm.from_mesh(me)
    uvl = bm.loops.layers.uv.active
    if not uvl: bm.free(); return None
    parent = list(range(len(bm.faces)))
    def find(x):
        while parent[x]!=x: parent[x]=parent[parent[x]]; x=parent[x]
        return x
    def union(a,b):
        ra,rb=find(a),find(b)
        if ra!=rb: parent[ra]=rb
    bm.faces.ensure_lookup_table()
    edge_map = {}
    for f in bm.faces:
        for l in f.loops:
            edge_map.setdefault(l.edge.index, []).append((f.index, l))
    for lst in edge_map.values():
        if len(lst)==2:
            (f1,l1),(f2,l2)=lst
            a1=l1[uvl].uv; b1=l1.link_loop_next[uvl].uv
            a2=l2.link_loop_next[uvl].uv; b2=l2[uvl].uv
            if (a1-a2).length<1e-6 and (b1-b2).length<1e-6: union(f1,f2)
    n = len({find(i) for i in range(len(bm.faces))})
    bm.free(); return n

def topo_report(obj):
    bm = bmesh.new(); bm.from_mesh(obj.data)
    r = {
      "non_manifold_edges": sum(1 for e in bm.edges if not e.is_manifold),
      "boundary_edges": sum(1 for e in bm.edges if e.is_boundary),
      "multi_face_edges": sum(1 for e in bm.edges if len(e.link_faces) > 2),
      "wire_edges": sum(1 for e in bm.edges if not e.link_faces),
      "loose_verts": sum(1 for v in bm.verts if not v.link_edges),
      "zero_area_faces": sum(1 for f in bm.faces if f.calc_area() < 1e-9),
    }
    bm.free()
    r["uv_islands"] = uv_island_count(obj.data)
    return r

for obj in bpy.data.objects:
    if obj.type == 'MESH':
        print(obj.name, json.dumps(topo_report(obj)))
```

## 3. Weld-distance sweep (non-destructive)

```python
import bpy, bmesh
obj = bpy.data.objects["Body"]
print("dimensions:", tuple(round(d,4) for d in obj.dimensions))
for dist in [1e-4, 5e-4, 1e-3, 2e-3, 5e-3]:
    bm = bmesh.new(); bm.from_mesh(obj.data)
    before = len(bm.verts)
    bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=dist)
    merged = before - len(bm.verts)
    nm = sum(1 for e in bm.edges if not e.is_manifold)
    bm.free()
    print(f"dist={dist}: merged {merged} ({100*merged/before:.1f}%), non-manifold after={nm}")
```
Pick ~0.1% of bounding size; reject any distance merging more than a few % of verts.

## 4. Backup collection

```python
import bpy
names = [o.name for o in bpy.data.objects if o.type == 'MESH']
col = bpy.data.collections.get("_backup_pre_cleanup") or bpy.data.collections.new("_backup_pre_cleanup")
if col.name not in [c.name for c in bpy.context.scene.collection.children]:
    bpy.context.scene.collection.children.link(col)
for n in names:
    src = bpy.data.objects[n]
    dup = src.copy(); dup.data = src.data.copy(); dup.name = n + "_backup"
    col.objects.link(dup)
col.hide_viewport = True; col.hide_render = True
```

## 5. Cleanup pass

Weld + degenerate + wire, then loop flaps, then loop fins, then fill:

```python
import bpy, bmesh
obj = bpy.data.objects["Body"]
bm = bmesh.new(); bm.from_mesh(obj.data)
bmesh.ops.remove_doubles(bm, verts=bm.verts, dist=0.001)      # from sweep
bmesh.ops.dissolve_degenerate(bm, edges=bm.edges, dist=0.0005)
bm.edges.ensure_lookup_table()
wire = [e for e in bm.edges if not e.link_faces]
bmesh.ops.delete(bm, geom=wire, context='EDGES')
bm.to_mesh(obj.data); bm.free(); obj.data.update()
```

Flap loop (faces with ≥2 over-shared edges), then fin loop (≥1 over-shared AND
≥2 free edges) — same skeleton, swap the predicate:

```python
import bpy, bmesh
obj = bpy.data.objects["Body"]
for it in range(6):
    bm = bmesh.new(); bm.from_mesh(obj.data)
    bm.edges.ensure_lookup_table(); bm.faces.ensure_lookup_table()
    bad = []
    for f in bm.faces:
        over = sum(1 for e in f.edges if len(e.link_faces) > 2)
        free = sum(1 for e in f.edges if len(e.link_faces) == 1)
        if over >= 2:            # flap predicate; fins: over >= 1 and free >= 2
            bad.append(f)
    if not bad: bm.free(); break
    bmesh.ops.delete(bm, geom=bad, context='FACES')
    bm.verts.ensure_lookup_table()
    loose = [v for v in bm.verts if not v.link_edges]
    bmesh.ops.delete(bm, geom=loose, context='VERTS')
    bmesh.ops.holes_fill(bm, edges=bm.edges, sides=12)
    ngons = [f for f in bm.faces if len(f.verts) > 3]
    if ngons: bmesh.ops.triangulate(bm, faces=ngons)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
    bm.to_mesh(obj.data); bm.free(); obj.data.update()
```

Final fill: choose `sides` from the boundary-loop histogram (below) so intentional
openings stay open.

## 6. Boundary-loop histogram

```python
import bpy, bmesh
from collections import Counter
obj = bpy.data.objects["Body"]
bm = bmesh.new(); bm.from_mesh(obj.data)
bound = [e for e in bm.edges if e.is_boundary]
bset = set(bound); visited = set(); loops = []
for e in bound:
    if e in visited: continue
    loop = {e}; visited.add(e); stack=[e]
    while stack:
        cur = stack.pop()
        for v in cur.verts:
            for e2 in v.link_edges:
                if e2 in bset and e2 not in visited:
                    visited.add(e2); loop.add(e2); stack.append(e2)
    loops.append(len(loop))
print("loops:", len(loops), dict(sorted(Counter(loops).items())))
bm.free()
```
Small sizes (1–15ish) = slits to fill. Few large ones = intentional openings.

## 7. Targeted despike

```python
import bpy, bmesh, math
from mathutils import Vector
obj = bpy.data.objects["Body"]
bm = bmesh.new(); bm.from_mesh(obj.data)
spike = set()
for e in bm.edges:
    if len(e.link_faces) == 2:
        if e.link_faces[0].normal.dot(e.link_faces[1].normal) < math.cos(math.radians(120)):
            spike.update(e.verts)
for _ in range(2):
    new_pos = {}
    for v in spike:
        if not v.link_edges: continue
        avg = Vector((0,0,0))
        for e in v.link_edges: avg += e.other_vert(v).co
        new_pos[v] = v.co.lerp(avg / len(v.link_edges), 0.3)
    for v, p in new_pos.items(): v.co = p
bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
bm.to_mesh(obj.data); bm.free(); obj.data.update()
```

## 8. Join

```python
import bpy
target = bpy.data.objects["Body"]
parts = ["Eye_L","Eye_R","Teeth_Upper","Teeth_Lower","Tongue"]
bpy.ops.object.mode_set(mode='OBJECT')
bpy.ops.object.select_all(action='DESELECT')
for n in parts: bpy.data.objects[n].select_set(True)
target.select_set(True)
bpy.context.view_layer.objects.active = target
bpy.ops.object.join()
target.data.name = target.name
```
Verify UV layer names matched beforehand (`me.uv_layers[i].name = "UVMap"`).

## 9. 6-axis segmentation + region absorption + seams

```python
import bpy, bmesh
from mathutils import Vector
from collections import Counter

body = bpy.data.objects["Body"]
bm = bmesh.new(); bm.from_mesh(body.data)
bm.faces.ensure_lookup_table(); bm.edges.ensure_lookup_table()

mats = [ms.material.name for ms in body.material_slots]
target_idx = {mats.index("Material_0")}   # material slots to re-unwrap
axes = [Vector((1,0,0)),Vector((-1,0,0)),Vector((0,1,0)),
        Vector((0,-1,0)),Vector((0,0,1)),Vector((0,0,-1))]

label = {f: max(range(6), key=lambda i: f.normal.dot(axes[i]))
         for f in bm.faces if f.material_index in target_idx}

def regions(label):
    seen, regs = set(), []
    for f in label:
        if f in seen: continue
        stack, comp, lab = [f], [f], label[f]; seen.add(f)
        while stack:
            cur = stack.pop()
            for e in cur.edges:
                for nf in e.link_faces:
                    if nf in label and nf not in seen and label[nf] == lab:
                        seen.add(nf); stack.append(nf); comp.append(nf)
        regs.append(comp)
    return regs

MIN_REGION = 40    # bigger => fewer, larger islands
for it in range(20):
    small = [r for r in regions(label) if len(r) < MIN_REGION]
    if not small: break
    for comp in small:
        cset = set(comp); border = Counter()
        for f in comp:
            for e in f.edges:
                for nf in e.link_faces:
                    if nf in label and nf not in cset:
                        border[label[nf]] += 1
        if border:
            newlab = border.most_common(1)[0][0]
            for f in comp: label[f] = newlab

regs = regions(label)
rid = {f: i for i, comp in enumerate(regs) for f in comp}
for e in bm.edges: e.seam = False
for e in bm.edges:
    lf = [f for f in e.link_faces if f in rid]
    if len(lf) == 2 and rid[lf[0]] != rid[lf[1]]: e.seam = True
print("regions:", len(regs))
bm.to_mesh(body.data); bm.free(); body.data.update()
```

## 10. Seam-based unwrap + pack

```python
import bpy, bmesh
body = bpy.data.objects["Body"]
bpy.context.view_layer.objects.active = body
bpy.ops.object.select_all(action='DESELECT'); body.select_set(True)
bpy.ops.object.mode_set(mode='EDIT')
bpy.ops.mesh.select_mode(type='FACE')
bpy.ops.mesh.select_all(action='DESELECT')
bm = bmesh.from_edit_mesh(body.data)
mats = [ms.material.name for ms in body.material_slots]
target_idx = {mats.index("Material_0")}
for f in bm.faces: f.select = f.material_index in target_idx
bmesh.update_edit_mesh(body.data)
bpy.ops.uv.unwrap(method='ANGLE_BASED', margin=0.003, correct_aspect=True)
bpy.ops.mesh.select_all(action='SELECT')
bpy.ops.uv.select_all(action='SELECT')
try: bpy.ops.uv.pack_islands(rotate=True, margin=0.004)
except TypeError: bpy.ops.uv.pack_islands(margin=0.004)   # older API
bpy.ops.object.mode_set(mode='OBJECT')
```

## 11. Atlas bake

Setup (backup material copies, atlas images, active bake-target nodes, settings):

```python
import bpy
scene = bpy.context.scene
body = bpy.data.objects["Body"]
backup_col = bpy.data.collections["_backup_pre_cleanup"]
backup_col.hide_viewport = False; backup_col.hide_render = False

copied = {}
for ob in backup_col.objects:
    for slot in ob.material_slots:
        if slot.material:
            if slot.material.name not in copied:
                c = slot.material.copy(); c.name = slot.material.name + "_orig"
                copied[slot.material.name] = c
            slot.material = copied[slot.material.name]

def make_img(name, size, cs):
    if name in bpy.data.images: bpy.data.images.remove(bpy.data.images[name])
    img = bpy.data.images.new(name, size, size, alpha=False)
    img.colorspace_settings.name = cs; return img
atlas_col = make_img("Atlas_BaseColor", 2048, "sRGB")
atlas_rgh = make_img("Atlas_Roughness", 2048, "Non-Color")

for slot in body.material_slots:
    mat = slot.material; mat.use_nodes = True
    nt = mat.node_tree
    node = nt.nodes.get("BakeTarget") or nt.nodes.new('ShaderNodeTexImage')
    node.name = node.label = "BakeTarget"; node.image = atlas_col
    nt.nodes.active = node

scene.render.engine = 'CYCLES'
scene.cycles.samples = 4; scene.cycles.use_denoising = False
b = scene.render.bake
b.use_selected_to_active = True; b.use_cage = False
b.cage_extrusion = 0.01; b.max_ray_distance = 0.025; b.margin = 16
b.use_pass_direct = False; b.use_pass_indirect = False; b.use_pass_color = True
```

Bake (repeat with the roughness image + `type='ROUGHNESS'`):

```python
import bpy
body = bpy.data.objects["Body"]
backup_col = bpy.data.collections["_backup_pre_cleanup"]
bpy.ops.object.mode_set(mode='OBJECT')
bpy.ops.object.select_all(action='DESELECT')
for ob in backup_col.objects:
    ob.hide_set(False); ob.select_set(True)
body.select_set(True)
bpy.context.view_layer.objects.active = body
bpy.ops.object.bake(type='DIFFUSE')
bpy.data.images["Atlas_BaseColor"].pack()
```

Rewire and hide backups:

```python
import bpy
body = bpy.data.objects["Body"]
atlas_col = bpy.data.images["Atlas_BaseColor"]
atlas_rgh = bpy.data.images["Atlas_Roughness"]
for slot in body.material_slots:
    nt = slot.material.node_tree
    bsdf = next((n for n in nt.nodes if n.type == 'BSDF_PRINCIPLED'), None)
    if not bsdf: continue
    for n in list(nt.nodes):
        if n.name != "BakeTarget" and n.type in {'TEX_IMAGE','SEPARATE_COLOR'}:
            nt.nodes.remove(n)
    ncol = nt.nodes["BakeTarget"]
    ncol.name = ncol.label = "Atlas_BaseColor"; ncol.image = atlas_col
    nt.links.new(ncol.outputs['Color'], bsdf.inputs['Base Color'])
    nrgh = nt.nodes.new('ShaderNodeTexImage')
    nrgh.name = nrgh.label = "Atlas_Roughness"; nrgh.image = atlas_rgh
    nt.links.new(nrgh.outputs['Color'], bsdf.inputs['Roughness'])
col = bpy.data.collections["_backup_pre_cleanup"]
for ob in col.objects: ob.select_set(False)
col.hide_viewport = True; col.hide_render = True
```
