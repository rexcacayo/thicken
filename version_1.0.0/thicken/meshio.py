"""Leer y escribir mallas de Blender con numpy (coordenadas de mundo, materiales por cara)."""
import bpy
import numpy as np


def read_arrays(context, obj, mm=1.0):
    """(V en mundo × mm, F triángulos, material por triángulo) de la malla evaluada."""
    graph = context.evaluated_depsgraph_get()
    ev = obj.evaluated_get(graph)
    me = ev.to_mesh()
    try:
        me.calc_loop_triangles()
        nt = len(me.loop_triangles)
        tri = np.empty(nt * 3, np.int64); me.loop_triangles.foreach_get('vertices', tri)
        pidx = np.empty(nt, np.int64); me.loop_triangles.foreach_get('polygon_index', pidx)
        pmat = np.empty(len(me.polygons), np.int64); me.polygons.foreach_get('material_index', pmat)
        co = np.empty(len(me.vertices) * 3); me.vertices.foreach_get('co', co)
    finally:
        ev.to_mesh_clear()
    M = np.array(obj.matrix_world)
    V = (co.reshape(-1, 3) @ M[:3, :3].T + M[:3, 3]) * mm
    F = tri.reshape(-1, 3)
    if np.linalg.det(M[:3, :3]) < 0:
        F = F[:, ::-1]
    mat = pmat[pidx] if len(pmat) else np.zeros(len(F), np.int64)
    return V, F, mat


def new_mesh(name, V, F, materials=None, mat_index=None):
    me = bpy.data.meshes.new(name)
    me.vertices.add(len(V))
    me.vertices.foreach_set('co', np.asarray(V, np.float32).ravel())
    me.loops.add(len(F) * 3)
    me.loops.foreach_set('vertex_index', np.asarray(F, np.int32).ravel())
    me.polygons.add(len(F))
    me.polygons.foreach_set('loop_start', np.arange(0, len(F) * 3, 3, dtype=np.int32))
    if materials:
        for m in materials:
            me.materials.append(m)
        if mat_index is not None and len(mat_index):
            me.polygons.foreach_set('material_index', np.asarray(mat_index, np.int32))
    me.update(calc_edges=True)
    return me


def new_object_like(context, src, name, V_bu, F, mat):
    """Objeto nuevo junto al original, con sus materiales, en coordenadas de mundo."""
    col = src.users_collection[0] if src.users_collection else context.scene.collection
    me = new_mesh(name, V_bu, F, [s.material for s in src.material_slots], mat)
    obj = bpy.data.objects.new(name, me)
    col.objects.link(obj)
    return obj


def replace_mesh(obj, V_bu, F, mat):
    me = new_mesh(obj.data.name, V_bu, F, [s.material for s in obj.material_slots], mat)
    old = obj.data
    obj.data = me
    obj.matrix_world.identity()
    if old.users == 0:
        bpy.data.meshes.remove(old)


def select_only(context, obj):
    for o in context.selected_objects:
        o.select_set(False)
    obj.select_set(True)
    context.view_layer.objects.active = obj
