"""Núcleo común 1.4.0 (copia en thicken) de los complementos del taller (Blender 4.2 a 5.x).

Regla de la casa: TODO se ejecuta en modo Objeto.
- Nunca se entra ni se sale de modo Edición por código.
- Las piezas se eligen con objetos ayudantes que se mueven en modo Objeto:
  un «plano de corte» para separar y un «cortador» (cilindro, caja, esfera o
  cualquier malla cerrada) para sacar insertos.
- Si el usuario ya tenía caras marcadas de una sesión de Edición, también se
  pueden usar: se leen de la malla sin cambiar de modo.
- El original nunca se modifica: se oculta y se trabaja en copias.
- Booleanas sin bpy.ops (modificador evaluado → malla nueva): no dependen del
  contexto ni del objeto activo.
Todas las distancias públicas van en milímetros.
"""
import math
import re
import struct
from pathlib import Path

import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

ROLE = 'taller_role'
ROLE_PLANE = 'plano_corte'
ROLE_CUTTER = 'cortador'
TARGET = 'taller_modelo'
SHAPE = 'taller_forma'


class AddonError(ValueError):
    pass


# --------------------------------------------------------------------------- modo Objeto
def in_object_mode(context):
    return context.mode == 'OBJECT'


def poll_object_mode(cls, context):
    if context.mode != 'OBJECT':
        cls.poll_message_set('Pasa a modo Objeto (Tab) para usar este botón.')
        return False
    return True


def draw_mode_warning(layout, context):
    """Devuelve True si hay que avisar (y dibuja el aviso con botón para volver)."""
    if context.mode == 'OBJECT':
        return False
    box = layout.box()
    box.alert = True
    box.label(text='Estos botones trabajan en modo Objeto', icon='ERROR')
    box.operator('object.mode_set', text='Volver a modo Objeto', icon='OBJECT_DATAMODE').mode = 'OBJECT'
    return True


def is_helper(obj):
    return obj is not None and obj.get(ROLE) in (ROLE_PLANE, ROLE_CUTTER)


# --------------------------------------------------------------------------- unidades y nombres
UNIT_PROP = 'taller_unidad'          # unidad resuelta, se hereda en las piezas
_UNIT_MM = {'MM': 1.0, 'CM': 10.0, 'M': 1000.0}


def guess_unit(max_dimension):
    """Misma heurística que BigPrint: se asume una pieza imprimible (10 a 1000 mm)."""
    d = abs(float(max_dimension))
    if d <= 0.0 or d >= 20.0:
        return 'MM'          # 20..2000 unidades: casi siempre un STL en mm
    if d >= 2.0:
        return 'CM'
    return 'M'               # < 2 unidades: modelado en metros (defecto de Blender)


def resolve_unit(scene, unit='AUTO', obj=None):
    """Devuelve (código, mm por unidad). AUTO usa la unidad heredada o la adivina por tamaño."""
    if unit == 'SCENE':
        return 'SCENE', scene.unit_settings.scale_length * 1000.0
    if unit in _UNIT_MM:
        return unit, _UNIT_MM[unit]
    code = obj.get(UNIT_PROP) if obj is not None else None
    if code not in _UNIT_MM:
        code = guess_unit(max(obj.dimensions) if obj is not None else 0.0)
    return code, _UNIT_MM[code]


def factor(scene, unit='AUTO', obj=None):
    """Milímetros por unidad de Blender."""
    return resolve_unit(scene, unit, obj)[1]


UNIT_NAMES = {'MM': 'mm', 'CM': 'cm', 'M': 'm', 'SCENE': 'escena'}


def safe_name(name):
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '_', str(name)).strip(' .')[:100] or 'Pieza'
    if re.match(r'^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)', name, re.I):
        name = '_' + name
    return name


def unique_name(name, used):
    base = safe_name(name)
    candidate = base
    i = 2
    while candidate.casefold() in used:
        candidate = f'{base}_{i:03d}'
        i += 1
    used.add(candidate.casefold())
    return candidate


# --------------------------------------------------------------------------- exportación STL
def _world_triangles(obj, graph, scale):
    """Triángulos de la malla evaluada en mm, vectorizado. Devuelve (vértices n×3×3, normales, material)."""
    evaluated = obj.evaluated_get(graph)
    me = evaluated.to_mesh()
    try:
        me.calc_loop_triangles()
        n = len(me.loop_triangles)
        if not n:
            return np.zeros((0, 3, 3)), np.zeros((0, 3)), np.zeros(0, int)
        tri = np.empty(n * 3, np.int32)
        me.loop_triangles.foreach_get('vertices', tri)
        mats = np.empty(n, np.int32)
        me.loop_triangles.foreach_get('material_index', mats)
        co = np.empty(len(me.vertices) * 3, np.float64)
        me.vertices.foreach_get('co', co)
    finally:
        evaluated.to_mesh_clear()
    m = np.array(obj.matrix_world, dtype=np.float64)
    co = (co.reshape(-1, 3) @ m[:3, :3].T + m[:3, 3]) * scale
    tri = tri.reshape(-1, 3)
    if np.linalg.det(m[:3, :3]) < 0:
        tri = tri[:, [0, 2, 1]]
    v = co[tri]
    normal = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    length = np.linalg.norm(normal, axis=1)
    ok = length > 1e-12
    return v[ok], normal[ok] / length[ok, None], mats[ok]


def _write_stl(target, v, normal):
    record = np.zeros(len(v), dtype=[('n', '<f4', (3,)), ('v', '<f4', (3, 3)), ('a', '<u2')])
    record['n'] = normal
    record['v'] = v
    with target.open('xb') as handle:  # nunca sobrescribe
        handle.write(b'Blender - coordenadas en milimetros'.ljust(80, b'\0'))
        handle.write(struct.pack('<I', len(v)))
        handle.write(record.tobytes())


def _material_name(obj, index):
    slots = obj.material_slots
    mat = slots[index].material if 0 <= index < len(slots) else None
    return mat.name if mat else 'Sin_Color'


def export_folder(path):
    """Carpeta real de salida. Con el .blend sin guardar, `//` no apunta a ningún sitio:
    se usa Documentos/<nombre> en vez de fallar."""
    if path.startswith('//') and not bpy.data.filepath:
        docs = Path.home() / 'Documents'
        return str((docs if docs.is_dir() else Path.home()) / (path[2:].strip('/\\') or 'STLs'))
    return path


def export_meshes(context, path, scope='SELECTED', unit='AUTO', by_color=False):
    """Exporta un STL binario en mm por objeto. Con by_color, una carpeta por color y,
    si un objeto tiene varios materiales, cada material sale en su carpeta."""
    if not path.strip():
        raise AddonError('Selecciona una carpeta de salida.')
    path = export_folder(path)
    if context.mode != 'OBJECT':
        raise AddonError('Exporta desde el modo Objeto.')
    candidates = context.selected_objects if scope == 'SELECTED' else context.visible_objects
    objects = sorted((o for o in candidates if o.type == 'MESH' and not is_helper(o)), key=lambda o: o.name)
    if not objects:
        raise AddonError('No hay mallas seleccionadas/visibles para exportar.')
    dest = Path(bpy.path.abspath(path)).resolve()
    if dest.exists() and not dest.is_dir():
        raise AddonError('La ruta de salida no es una carpeta.')
    graph = context.evaluated_depsgraph_get()
    color_dirs, used_dirs, planned = {}, set(), []
    for obj in objects:  # primero se calcula todo; si algo falla no queda nada a medias
        v, normal, mats = _world_triangles(obj, graph, factor(context.scene, unit, obj))
        if not len(v):
            raise AddonError(f'{obj.name}: la malla no tiene triángulos válidos.')
        if not by_color:
            planned.append((dest, obj.name, v, normal))
            continue
        groups = {}
        for index in np.unique(mats):
            groups.setdefault(_material_name(obj, int(index)), []).append(mats == index)
        for color, masks in groups.items():
            keep = np.logical_or.reduce(masks)
            if color not in color_dirs:
                color_dirs[color] = unique_name(color, used_dirs)
            planned.append((dest / color_dirs[color], obj.name, v[keep], normal[keep]))
    created = []
    try:
        for folder, name, v, normal in planned:
            folder.mkdir(parents=True, exist_ok=True)
            used = {p.stem.casefold() for p in folder.iterdir()}
            target = folder / (unique_name(name, used) + '.stl')
            _write_stl(target, v, normal)
            created.append(target)
    except Exception:
        for file in created:
            if file.is_file():
                file.unlink()
        raise
    return created


# --------------------------------------------------------------------------- mallas auxiliares
def new_object(bm, name, collection, source=None):
    data = bpy.data.meshes.new(name)
    bm.to_mesh(data)
    data.update()
    obj = bpy.data.objects.new(name, data)
    collection.objects.link(obj)
    if source is not None:
        for slot in source.material_slots:
            data.materials.append(slot.material)
    return obj


def remove_objects(objects):
    for obj in reversed(list(objects)):
        if obj is not None and obj.name in bpy.data.objects:
            data = obj.data
            bpy.data.objects.remove(obj, do_unlink=True)
            if data is not None and data.users == 0:
                bpy.data.meshes.remove(data)


def world_bmesh(context, obj):
    """bmesh de la malla evaluada (con modificadores) en coordenadas de mundo."""
    graph = context.evaluated_depsgraph_get()
    evaluated = obj.evaluated_get(graph)
    me = evaluated.to_mesh()
    try:
        bm = bmesh.new()
        bm.from_mesh(me)
    finally:
        evaluated.to_mesh_clear()
    bm.transform(obj.matrix_world)
    if obj.matrix_world.determinant() < 0:
        bmesh.ops.reverse_faces(bm, faces=bm.faces[:])
    return bm


def solid_info(obj):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    try:
        closed = bool(bm.faces) and all(e.is_manifold for e in bm.edges)
        volume = abs(bm.calc_volume(signed=True)) if closed else 0.0
        return closed, volume
    finally:
        bm.free()


def require_solid(context, obj):
    bm = world_bmesh(context, obj)
    try:
        if not bm.faces or any(not e.is_manifold for e in bm.edges):
            raise AddonError(f'{obj.name}: la malla debe estar cerrada (manifold). Repárala antes, por ejemplo en 3D LAB.')
    finally:
        bm.free()


def _solvers():
    return set(bpy.types.BooleanModifier.bl_rna.properties['solver'].enum_items.keys())


def boolean(context, obj, cutter, operation='DIFFERENCE'):
    """Aplica una booleana sin bpy.ops: se evalúa el modificador y se copia el resultado.
    Usa el solver «Manifold» (rápido) si esta versión de Blender lo tiene; si falla, «Exacto»."""
    before = solid_info(obj)[1]
    solvers = [s for s in ('MANIFOLD', 'EXACT') if s in _solvers()]
    last = None
    for solver in solvers:
        mod = obj.modifiers.new('Taller_booleana', 'BOOLEAN')
        mod.operation = operation
        mod.solver = solver
        mod.object = cutter
        try:
            context.view_layer.update()
            graph = context.evaluated_depsgraph_get()
            new = bpy.data.meshes.new_from_object(obj.evaluated_get(graph), preserve_all_data_layers=True, depsgraph=graph)
        finally:
            obj.modifiers.remove(mod)
        old = obj.data
        obj.data = new
        closed, after = solid_info(obj)
        valid = closed and after > 0 and (after < before - max(before * 1e-9, 1e-15) if operation == 'DIFFERENCE' else True)
        if valid:
            new.name = old.name
            if old.users == 0:
                bpy.data.meshes.remove(old)
            return
        obj.data = old
        bpy.data.meshes.remove(new)
        last = solver
    raise AddonError(f'La booleana no dio un sólido cerrado (solver {last}). El original se conserva.')


# --------------------------------------------------------------------------- primitivas
def cylinder_bmesh(center, axis, radius, height, segments=64):
    bm = bmesh.new()
    q = Vector(axis).normalized().to_track_quat('Z', 'Y')
    m = Matrix.Translation(Vector(center)) @ q.to_matrix().to_4x4()
    bmesh.ops.create_cone(bm, cap_ends=True, cap_tris=False, segments=segments,
                          radius1=radius, radius2=radius, depth=height, matrix=m)
    return bm


def cylinder(name, center, axis, radius, height, collection):
    bm = cylinder_bmesh(center, axis, radius, height)
    try:
        return new_object(bm, name, collection)
    finally:
        bm.free()


def primitive_bmesh(shape, size):
    """Primitiva centrada en el origen; size = (x, y, z) en unidades de Blender."""
    bm = bmesh.new()
    sx, sy, sz = size
    if shape == 'BOX':
        bmesh.ops.create_cube(bm, size=1.0)
        bmesh.ops.scale(bm, vec=(sx, sy, sz), verts=bm.verts[:])
    elif shape == 'SPHERE':
        bmesh.ops.create_uvsphere(bm, u_segments=48, v_segments=24, radius=.5)
        bmesh.ops.scale(bm, vec=(sx, sy, sz), verts=bm.verts[:])
    else:
        bmesh.ops.create_cone(bm, cap_ends=True, segments=64, radius1=.5, radius2=.5, depth=1.0)
        bmesh.ops.scale(bm, vec=(sx, sy, sz), verts=bm.verts[:])
    return bm


# --------------------------------------------------------------------------- ayudantes en escena
def _collection_of(obj, context):
    return obj.users_collection[0] if obj.users_collection else context.scene.collection


def pick_model(context):
    """El modelo con el que trabajar: la malla activa o seleccionada que no sea un ayudante;
    si sólo hay un ayudante seleccionado, el modelo que tiene apuntado."""
    active = context.active_object
    candidates = [active] + list(context.selected_objects)
    for obj in candidates:
        if obj is not None and obj.type == 'MESH' and not is_helper(obj):
            return obj
    for obj in candidates:
        if is_helper(obj):
            target = bpy.data.objects.get(obj.get(TARGET, ''))
            if target is not None and target.type == 'MESH':
                return target
    raise AddonError('Selecciona el modelo (una malla) en modo Objeto.')


def find_helper(context, role, model):
    """Ayudante a usar: el seleccionado; si no, el único de la escena para ese modelo (o el único)."""
    selected = [o for o in context.selected_objects if o.get(ROLE) == role]
    if len(selected) == 1:
        return selected[0]
    if len(selected) > 1:
        raise AddonError('Hay varios ayudantes seleccionados: deja sólo uno.')
    everyone = [o for o in context.scene.objects if o.get(ROLE) == role]
    mine = [o for o in everyone if o.get(TARGET) == model.name]
    for group in (mine, everyone):
        if len(group) == 1:
            return group[0]
        if len(group) > 1:
            raise AddonError('Hay varios ayudantes en la escena: selecciona el que quieres usar.')
    return None


def find_plane_quiet(context, model):
    """Plano de corte del modelo, o None (sin errores: para el panel)."""
    try:
        return find_helper(context, ROLE_PLANE, model)
    except AddonError:
        return True   # hay varios: que el botón quede activo y el operador explique


def _model_box(model):
    corners = [model.matrix_world @ Vector(c) for c in model.bound_box]
    lo = Vector([min(c[i] for c in corners) for i in range(3)])
    hi = Vector([max(c[i] for c in corners) for i in range(3)])
    return lo, hi


def add_cut_plane(context, model, axis='Z'):
    """Plano guía (malla de 4 vértices, alambre) en el centro del modelo. Se mueve y gira libremente."""
    lo, hi = _model_box(model)
    center = (lo + hi) / 2
    size = max((hi - lo).length * .6, 1e-4)
    bm = bmesh.new()
    try:
        for x, y in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            bm.verts.new((x * .5, y * .5, 0))
        bm.faces.new(bm.verts[:])
        obj = new_object(bm, 'Plano_corte', _collection_of(model, context))
    finally:
        bm.free()
    obj.scale = (size, size, size)
    obj.location = center
    obj.rotation_euler = {'X': (0, math.pi / 2, 0), 'Y': (math.pi / 2, 0, 0)}.get(axis, (0, 0, 0))
    obj.display_type = 'WIRE'
    obj.show_in_front = True
    obj.hide_render = True
    obj[ROLE] = ROLE_PLANE
    obj[TARGET] = model.name
    _select_only(context, obj)
    return obj


def plane_frame(plane):
    m = plane.matrix_world
    return m.translation.copy(), (m.to_3x3() @ Vector((0, 0, 1))).normalized()


def add_cutter(context, model, shape='CYLINDER', size_mm=(10, 10, 6), unit='AUTO'):
    """Cortador para insertos: se coloca sobre la zona (ojo, logo...) hundido lo que se quiera de profundidad."""
    scale = factor(context.scene, unit, model)
    size = tuple(s / scale for s in size_mm)
    lo, hi = _model_box(model)
    bm = primitive_bmesh(shape, size)
    try:
        obj = new_object(bm, 'Cortador', _collection_of(model, context))
    finally:
        bm.free()
    obj.location = ((lo.x + hi.x) / 2, lo.y, (lo.z + hi.z) / 2)   # delante del modelo, a la altura media
    obj.display_type = 'WIRE'
    obj.show_in_front = True
    obj.hide_render = True
    obj[ROLE] = ROLE_CUTTER
    obj[TARGET] = model.name
    obj[SHAPE] = shape
    obj['taller_tam'] = list(size)
    _select_only(context, obj)
    return obj


def _select_only(context, obj):
    for o in context.selected_objects:
        o.select_set(False)
    obj.select_set(True)
    context.view_layer.objects.active = obj


# --------------------------------------------------------------------------- geometría 2D del corte
def _plane_axes(normal):
    u = normal.orthogonal().normalized()
    return u, normal.cross(u).normalized()


def _segments(loops_2d):
    a = np.concatenate([lp for lp in loops_2d])
    b = np.concatenate([np.roll(lp, -1, axis=0) for lp in loops_2d])
    return a, b


def _inside(points, a, b):
    """Par-impar respecto a todos los contornos (admite islas y huecos)."""
    x, y = points[:, 0:1], points[:, 1:2]
    ax, ay, bx, by = a[:, 0], a[:, 1], b[:, 0], b[:, 1]
    crosses = (ay > y) != (by > y)
    with np.errstate(divide='ignore', invalid='ignore'):
        xint = (bx - ax) * (y - ay) / (by - ay) + ax
    return (np.count_nonzero(crosses & (x < xint), axis=1) % 2) == 1


def _clearance(points, a, b):
    """Distancia de cada punto al contorno más cercano (negativa si está fuera)."""
    out = np.empty(len(points))
    d = b - a
    dd = np.maximum((d ** 2).sum(1), 1e-30)
    for start in range(0, len(points), 256):
        p = points[start:start + 256, None, :]
        t = np.clip(((p - a) * d).sum(2) / dd, 0, 1)
        dist = np.sqrt((((a + t[..., None] * d) - p) ** 2).sum(2)).min(1)
        out[start:start + 256] = dist
    inside = _inside(points, a, b)
    return np.where(inside, out, -out)


def _islands(loops_2d):
    """Agrupa los contornos de la sección en zonas: cada contorno exterior con sus huecos.
    El corte del Lincoln por el cuello, por ejemplo, da dos zonas: cuello y brazo."""
    def contains(outer, point):
        a, b = _segments([outer])
        return bool(_inside(np.array([point]), a, b)[0])

    depth = []
    for i, lp in enumerate(loops_2d):
        depth.append(sum(contains(other, lp[0]) for j, other in enumerate(loops_2d) if j != i))
    outers = [i for i, d in enumerate(depth) if d % 2 == 0]
    groups = {i: [loops_2d[i]] for i in outers}
    for i, d in enumerate(depth):
        if d % 2 == 1:   # hueco: va con el exterior más profundo que lo contiene
            owners = [o for o in outers if contains(loops_2d[o], loops_2d[i][0])]
            if owners:
                groups[max(owners, key=lambda o: depth[o])].append(loops_2d[i])
    return list(groups.values())


def _area(loop):
    x, y = loop[:, 0], loop[:, 1]
    return abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))) / 2


def _candidates(island, radius, wall, deep_enough, grid=41, limit=400):
    """Puntos de la zona donde cabe el alojamiento: lejos del borde (pared) y con material
    suficiente a los dos lados del corte. Devuelve (puntos, holguras, mejor holgura vista)."""
    a, b = _segments(island)
    lo, hi = a.min(0), a.max(0)
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], grid)[1:-1], np.linspace(lo[1], hi[1], grid)[1:-1])
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    clear = _clearance(pts, a, b)
    best_seen = float(clear.max()) if len(clear) else 0.0
    ok = clear >= radius + wall
    pts, clear = pts[ok], clear[ok]
    if not len(pts):
        return pts, clear, best_seen
    order = np.argsort(-clear)                       # primero los más interiores
    step = max(1, len(order) // limit)
    order = order[::step]
    keep = [i for i in order if deep_enough(pts[i])]
    return pts[keep], clear[keep], best_seen


def connector_points(loops_2d, radius, wall, count, deep_enough=lambda p: True):
    """Posiciones de los alojamientos, por zona de la sección.

    - En cada zona se buscan puntos con pared mínima alrededor y material a ambos lados.
    - count=2: los dos puntos válidos más separados (frenan el giro); si no caben, uno.
    - count=1: el punto más interior.
    Devuelve (puntos, informe) donde informe = {'zonas', 'con_iman', 'sin_sitio', 'max_holgura'}."""
    islands = sorted(_islands(loops_2d), key=lambda isl: -_area(isl[0]))
    points, used, skipped, best = [], 0, 0, 0.0
    for island in islands:
        pts, clear, seen = _candidates(island, radius, wall, deep_enough)
        best = max(best, seen)
        if not len(pts):
            skipped += 1
            continue
        chosen = [tuple(pts[int(np.argmax(clear))])]
        if count >= 2 and len(pts) > 1:
            dist = np.linalg.norm(pts[:, None] - pts[None], axis=2)
            i, j = np.unravel_index(int(np.argmax(dist)), dist.shape)
            if dist[i, j] >= 2 * radius + wall:
                chosen = [tuple(pts[i]), tuple(pts[j])]
        points += chosen
        used += 1
    return points, {'zonas': len(islands), 'con_iman': used, 'sin_sitio': skipped, 'max_holgura': best}


# --------------------------------------------------------------------------- separar por plano
def _cap(bm, normal):
    """Tapa las aberturas del corte. triangle_fill admite secciones con huecos (piezas vaciadas)."""
    edges = [e for e in bm.edges if e.is_boundary]
    if not edges:
        return []
    faces = [g for g in bmesh.ops.triangle_fill(bm, use_beauty=True, use_dissolve=False, edges=edges, normal=normal)['geom']
             if isinstance(g, bmesh.types.BMFace)]
    rest = [e for e in bm.edges if e.is_boundary]
    if rest:
        raise AddonError('No se pudo cerrar la cara del corte (contorno abierto o cruzado).')
    return faces


def _half(context, source, co, normal, keep_positive, name, collection):
    bm = world_bmesh(context, source)
    try:
        bmesh.ops.bisect_plane(bm, geom=bm.verts[:] + bm.edges[:] + bm.faces[:], dist=1e-7,
                               plane_co=co, plane_no=normal, clear_outer=keep_positive is False, clear_inner=keep_positive)
        if not bm.faces:
            raise AddonError('El plano no corta el modelo: muévelo para que lo atraviese.')
        cap_normal = -normal if keep_positive else normal     # la tapa mira hacia fuera de cada mitad
        cap = _cap(bm, cap_normal)
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
        if not cap:
            raise AddonError('El plano no corta el modelo: muévelo para que lo atraviese.')
        return new_object(bm, name, collection, source)
    finally:
        bm.free()


def _section_loops(context, source, co, normal):
    """Contornos de la sección del modelo por el plano, en 2D (u, v)."""
    bm = world_bmesh(context, source)
    try:
        cut = bmesh.ops.bisect_plane(bm, geom=bm.verts[:] + bm.edges[:] + bm.faces[:], dist=1e-7,
                                     plane_co=co, plane_no=normal)['geom_cut']
        edges = [e for e in cut if isinstance(e, bmesh.types.BMEdge)]
        u, v = _plane_axes(normal)
        adjacent = {}
        for e in edges:
            for vert in e.verts:
                adjacent.setdefault(vert, []).append(e)
        seen, loops = set(), []
        for e in edges:
            if e in seen:
                continue
            start = e.verts[0]
            current, last, ring = start, None, []
            while True:
                ring.append(current.co.copy())
                nxt = next((x for x in adjacent.get(current, []) if x is not last and x not in seen), None)
                if nxt is None:
                    break
                seen.add(nxt)
                current, last = nxt.other_vert(current), nxt
                if current is start:
                    break
            if len(ring) >= 3:
                loops.append(np.array([((p - co).dot(u), (p - co).dot(v)) for p in ring]))
        if not loops:
            raise AddonError('El plano no corta el modelo: muévelo para que lo atraviese.')
        return loops, u, v
    finally:
        bm.free()


def depth_tester(context, model, co, normal, u, v, radius, depth, wall, mm_factor):
    """Función que dice si en un punto (u, v) del corte hay material para el alojamiento a los
    dos lados: centro y 8 puntos del borde, rayos hacia arriba y hacia abajo del plano."""
    bm = world_bmesh(context, model)
    try:
        tree = BVHTree.FromBMesh(bm)
    finally:
        bm.free()
    need = depth + wall
    # Paso inicial relativo al tamaño: con coordenadas de cientos de unidades, 0,001 está por
    # debajo de la precisión de float32 y el rayo chocaba con la propia tapa del corte.
    eps = max(.01 / mm_factor, 1e-6 * max(1.0, co.length))
    q = normal.to_track_quat('Z', 'Y')
    rim = [Vector()] + [q @ Vector((radius * math.cos(i * math.pi / 4), radius * math.sin(i * math.pi / 4), 0)) for i in range(8)]

    def deep_enough(p2):
        point = co + u * float(p2[0]) + v * float(p2[1])
        for direction in (normal, -normal):
            for off in rim:
                loc, _n, _i, dist = tree.ray_cast(point + off + direction * eps, direction)
                if loc is None or dist + eps < need:
                    return False
        return True
    return deep_enough


KIND_NAMES = {'MAGNET': 'imán', 'DOWEL': 'espiga'}


def _finish(context, source, outputs, helpers=()):
    source.hide_set(True)
    source.hide_render = True
    source['taller_original_conservado'] = True
    # El plano ya ha servido: se borra para que no se reutilice por error con otra pieza.
    remove_objects([h for h in helpers if h is not None and h.name in bpy.data.objects])
    for o in context.selected_objects:
        o.select_set(False)
    for o in outputs:
        o['taller_origen'] = source.name
        o.select_set(True)
    context.view_layer.objects.active = outputs[0]


def split_with_plane(context, kind, radius_mm, length_mm, tolerance_mm, wall_mm, mode='AUTO', unit='AUTO'):
    """Separa el modelo por el plano de corte ayudante (se conserva para scripts antiguos)."""
    model = pick_model(context)
    plane = find_helper(context, ROLE_PLANE, model)
    if plane is None:
        raise AddonError('Añade primero un plano de corte y colócalo donde quieras separar.')
    co, normal = plane_frame(plane)
    return split_at(context, model, co, normal, kind, radius_mm, length_mm, tolerance_mm, wall_mm, mode, unit, helpers=[plane])


# --------------------------------------------------------------------------- eje de giro
PIVOT_CLEARANCE = 0.2      # holgura radial mínima para que gire (mm)


def pivot_dims(radius_mm, length_mm, tolerance_mm):
    """Medidas del eje de giro a presión, en mm.

    El eje sale de la pieza de abajo (_01) y entra en la de arriba (_02). Cerca de la punta
    tiene un reborde que pasa a presión por la garganta del agujero y luego queda suelto en
    una cámara más ancha: así gira libre pero no se sale. Una ranura en la punta le deja
    cerrarse al entrar. En ejes pequeños (Ø < 4 mm) no hay reborde ni ranura: solo
    giro a fricción."""
    r = float(radius_mm)
    L = float(length_mm)
    c = max(float(tolerance_mm), PIVOT_CLEARANCE)
    snap = r >= 2.0 and L >= 4.0
    lip = min(.4, .08 * 2 * r) if snap else 0.0
    lip_h = min(1.2, L * .25) if snap else 0.0
    z_lip = L - 1.6 * lip_h if snap else L          # donde empieza el reborde
    return {'r': r, 'L': L, 'c': c, 'snap': snap, 'lip': lip, 'lip_h': lip_h, 'z_lip': z_lip,
            'embed': max(.6, .15 * L), 'chamfer': min(.5, .2 * r),
            'slot_w': max(.8, .35 * r), 'slot_z': L * .35,
            'hole_r': r + lip + c, 'hole_depth': L + c}


def _revolve(profile, segments=48):
    """bmesh cerrado por revolución de un perfil (radio, z) alrededor del eje Z."""
    bm = bmesh.new()
    rings = []
    for rad, z in profile:
        if rad <= 1e-9:
            rings.append([bm.verts.new((0.0, 0.0, z))])
        else:
            rings.append([bm.verts.new((rad * math.cos(2 * math.pi * i / segments),
                                        rad * math.sin(2 * math.pi * i / segments), z)) for i in range(segments)])
    for lo, hi in zip(rings, rings[1:]):
        if len(lo) == 1 and len(hi) == 1:
            continue
        for i in range(segments):
            j = (i + 1) % segments
            if len(lo) == 1:
                bm.faces.new((lo[0], hi[j], hi[i]))
            elif len(hi) == 1:
                bm.faces.new((lo[i], lo[j], hi[0]))
            else:
                bm.faces.new((lo[i], lo[j], hi[j], hi[i]))
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    return bm


def _frame_object(bm, name, point, normal, scale, collection):
    """Coloca un bmesh hecho en mm (eje Z = normal del corte, z=0 en el corte) en el mundo."""
    q = Vector(normal).normalized().to_track_quat('Z', 'Y')
    m = Matrix.Translation(Vector(point)) @ q.to_matrix().to_4x4() @ Matrix.Scale(1.0 / scale, 4)
    bm.transform(m)
    return new_object(bm, name, collection)


def pivot_objects(point, normal, d, scale, collection):
    """(eje, cortador del agujero, cortador de la ranura o None) como objetos temporales."""
    r, L, ch = d['r'], d['L'], d['chamfer']
    if d['snap']:
        z0, lip, lh = d['z_lip'], d['lip'], d['lip_h']
        shaft = [(0, -d['embed']), (r, -d['embed']), (r, z0), (r + lip, z0 + .3 * lh),
                 (r - ch, L), (0, L)]
    else:
        shaft = [(0, -d['embed']), (r, -d['embed']), (r, L - ch), (r - ch, L), (0, L)]
    hr, c = r + d['c'], d['c']
    if d['snap']:
        throat = d['z_lip'] - c
        hole = [(0, -.5), (hr, -.5), (hr, throat), (d['hole_r'], throat), (d['hole_r'], d['hole_depth']), (0, d['hole_depth'])]
    else:
        hole = [(0, -.5), (hr, -.5), (hr, d['hole_depth']), (0, d['hole_depth'])]
    peg = _frame_object(_revolve(shaft), 'Eje_temporal', point, normal, scale, collection)
    cut = _frame_object(_revolve(hole), 'Cortador_temporal', point, normal, scale, collection)
    slot = None
    if d['snap']:
        bm = bmesh.new()
        bmesh.ops.create_cube(bm, size=1.0)
        h = L - d['slot_z'] + 1.0
        bmesh.ops.scale(bm, vec=(d['slot_w'], 2 * (r + d['lip']) + 2, h), verts=bm.verts[:])
        bmesh.ops.translate(bm, vec=(0, 0, d['slot_z'] + h / 2), verts=bm.verts[:])
        slot = _frame_object(bm, 'Ranura_temporal', point, normal, scale, collection)
    return peg, cut, slot


KIND_NAMES['PIVOT'] = 'eje de giro'


def split_at(context, model, co, normal, kind, radius_mm, length_mm, tolerance_mm, wall_mm,
             mode='AUTO', unit='AUTO', helpers=()):
    """Separa `model` por el plano (co, normal) y pone los conectores en las dos caras.
    kind: 'MAGNET' (alojamiento de imán en cada cara), 'DOWEL' (agujeros + espiga suelta),
    'PIVOT' (eje de giro a presión: la pieza de arriba gira sobre la de abajo) o 'NONE'.
    Devuelve un texto con el resultado para mostrarlo al usuario."""
    require_solid(context, model)
    unit_code, scale = resolve_unit(context.scene, unit, model)
    co, normal = Vector(co), Vector(normal).normalized()
    col = _collection_of(model, context)
    points, report, pivot = [], None, None
    if kind == 'PIVOT':
        pivot = pivot_dims(radius_mm, length_mm, tolerance_mm)
        radius = pivot['hole_r'] / scale
        depth = pivot['hole_depth'] / scale
        wall = wall_mm / scale
        loops, u, v = _section_loops(context, model, co, normal)
        deep = depth_tester(context, model, co, normal, u, v, radius, depth, wall, scale)
        points, report = connector_points(loops, radius, wall, 1, deep)
        points = points[:1]         # un solo eje: la pieza gira alrededor de él
        if not points:
            fits = max(0.0, (report['max_holgura'] - wall) * scale - pivot['lip'] - pivot['c']) * 2
            raise AddonError(f'No cabe un eje de giro de Ø{radius_mm * 2:g} mm con {wall_mm:g} mm de pared en este corte '
                             f'(lo más grueso admite unos Ø{fits:.1f} mm). Usa uno más fino o mueve el corte.')
    elif kind != 'NONE':
        radius = (radius_mm + tolerance_mm) / scale
        depth = ((length_mm / 2 + tolerance_mm) if kind == 'DOWEL' else (length_mm + tolerance_mm)) / scale
        wall = wall_mm / scale
        loops, u, v = _section_loops(context, model, co, normal)
        deep = depth_tester(context, model, co, normal, u, v, radius, depth, wall, scale)
        want = 1 if mode == 'ONE' else 2
        points, report = connector_points(loops, radius, wall, want, deep)
        if not points:
            fits = max(0.0, (report['max_holgura'] - wall) * scale - tolerance_mm) * 2
            raise AddonError(
                f'No cabe ninguna {KIND_NAMES[kind]} de Ø{radius_mm * 2:g} mm con {wall_mm:g} mm de pared en este corte '
                f'(lo más grueso admite unos Ø{fits:.1f} mm). Usa uno más pequeño o mueve el corte.'
                if kind == 'DOWEL' else
                f'No cabe ningún imán de Ø{radius_mm * 2:g} × {length_mm:g} mm con {wall_mm:g} mm de pared en este corte '
                f'(lo más grueso admite unos Ø{fits:.1f} mm). Usa uno más pequeño o mueve el corte.')
        if mode == 'TWO' and len(points) < 2:
            raise AddonError('No caben dos alojamientos en este corte. Elige «Automático» para poner uno.')
    created = []
    try:
        base = _half(context, model, co, normal, False, model.name + '_01', col); created.append(base)
        piece = _half(context, model, co, normal, True, model.name + '_02', col); created.append(piece)
        for x, y in points:
            point = co + u * x + v * y
            if kind == 'PIVOT':
                peg, cut, slot = pivot_objects(point, normal, pivot, scale, col)
                created += [o for o in (peg, cut, slot) if o is not None]
                if slot is not None:
                    boolean(context, peg, slot)
                boolean(context, piece, cut)
                boolean(context, base, peg, 'UNION')
                for o in (peg, cut, slot):
                    if o is not None:
                        created.remove(o)
                remove_objects([o for o in (peg, cut, slot) if o is not None])
                continue
            cutter = cylinder('Cortador_temporal', point, normal, radius, depth * 2, col); created.append(cutter)
            boolean(context, base, cutter)
            boolean(context, piece, cutter)
            created.remove(cutter); remove_objects([cutter])
        outputs = [base, piece]
        if kind == 'DOWEL':
            lo, hi = _model_box(model)
            r = radius_mm / scale
            for i in range(len(points)):
                dowel = cylinder(model.name + '_Espiga', (hi.x + r * 4, lo.y + i * r * 4, r), (0, 1, 0), r, length_mm / scale, col)
                created.append(dowel); outputs.append(dowel)
        if unit_code != 'SCENE':
            for o in outputs:
                o[UNIT_PROP] = unit_code
        _finish(context, model, outputs, list(helpers))
    except Exception:
        remove_objects(created)
        raise
    if kind == 'NONE':
        return 'Separado en 2 piezas sin conectores. Original conservado y oculto.'
    if kind == 'PIVOT':
        extra = ' a presión' if pivot['snap'] else ' (a fricción: eje pequeño)'
        text = f'Separado con eje de giro Ø{radius_mm * 2:g} mm{extra}: la pieza de arriba gira'
        if report['zonas'] > 1:
            text += f' (el eje va en la zona más gruesa; las otras {report["zonas"] - 1} se pegan)'
        return text + '. Original conservado y oculto.'
    name = 'imán(es)' if kind == 'MAGNET' else 'espiga(s)'
    text = f'Separado: {len(points)} {name} por cara'
    if report['zonas'] > 1:
        text += f' en {report["con_iman"]} de {report["zonas"]} zonas del corte'
        if report['sin_sitio']:
            text += f' ({report["sin_sitio"]} demasiado fina{"s" if report["sin_sitio"] > 1 else ""})'
    return text + '. Original conservado y oculto.'


# --------------------------------------------------------------------------- línea de corte
def fit_plane(points):
    """Plano que mejor pasa por los puntos de la línea (PCA). Devuelve (origen, normal, desviación).
    Si sale casi recto (< 3° de un eje) se endereza: los cortes rectos quedan más limpios."""
    pts = np.array([tuple(p) for p in points], dtype=float)
    if len(pts) < 3:
        raise AddonError('Hacen falta al menos tres puntos: rodea la pieza con la línea.')
    c = pts.mean(0)
    _u, sv, vt = np.linalg.svd(pts - c)
    if sv[1] < 1e-9 * max(sv[0], 1e-12):
        raise AddonError('Los puntos están en línea recta: rodea la pieza con la línea.')
    n = vt[2]
    dom = int(np.argmax(np.abs(n)))
    if n[dom] < 0:
        n = -n
    if abs(n[dom]) >= math.cos(math.radians(3.0)):
        n = np.zeros(3); n[dom] = 1.0
    dev = float(np.abs((pts - c) @ n).max())
    return Vector(c), Vector(n), dev


def tilt_degrees(normal):
    return math.degrees(math.acos(min(1.0, max(abs(normal.x), abs(normal.y), abs(normal.z)))))


# --------------------------------------------------------------------------- caras marcadas (sin modo Edición)
def marked_mask(obj):
    """Caras marcadas en la última sesión de Edición, leídas de la malla en modo Objeto."""
    if obj.type != 'MESH':
        raise AddonError('El objeto activo no es una malla.')
    if any(m.show_viewport for m in obj.modifiers):
        raise AddonError('La malla tiene modificadores activos: aplícalos en una copia antes de usar caras marcadas.')
    mask = np.zeros(len(obj.data.polygons), bool)
    obj.data.polygons.foreach_get('select', mask)
    if not mask.any() or mask.all():
        raise AddonError('No hay caras marcadas (o están todas). Usa mejor el plano de corte o el cortador.')
    return mask


def _mesh_bmesh(obj):
    bm = bmesh.new()
    bm.from_mesh(obj.data)
    bm.transform(obj.matrix_world)
    bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
    bm.faces.ensure_lookup_table()
    return bm


def split_marked(context, kind, radius_mm, length_mm, tolerance_mm, wall_mm, mode='AUTO', unit='AUTO'):
    """Como 1.1 pero en modo Objeto: separa las caras marcadas por su borde (que debe ser plano)."""
    model = pick_model(context)
    require_solid(context, model)
    mask = marked_mask(model)
    bm = _mesh_bmesh(model)
    try:
        selected = [f for f in bm.faces if mask[f.index]]
        chosen = set(selected)
        edges = [e for e in bm.edges if sum(f in chosen for f in e.link_faces) == 1]
        pts = [v.co.copy() for e in edges for v in e.verts]
    finally:
        bm.free()
    if not pts:
        raise AddonError('No se ha encontrado el borde de las caras marcadas.')
    center = sum(pts, Vector()) / len(pts)
    arr = np.array([tuple(p - center) for p in pts])
    normal = Vector(np.linalg.svd(arr, full_matrices=False)[2][-1])
    scale = factor(context.scene, unit, model)
    if np.abs(arr @ np.array(normal)).max() * scale > .02:
        raise AddonError('El borde de las caras marcadas no es plano (> 0,02 mm). Usa el plano de corte.')
    # Orientar la normal hacia las caras marcadas y reutilizar el corte por plano.
    marked_center = sum((model.matrix_world @ model.data.polygons[i].center for i in np.nonzero(mask)[0][:500]), Vector()) / min(500, int(mask.sum()))
    if (marked_center - center).dot(normal) < 0:
        normal = -normal
    col = _collection_of(model, context)
    temp = add_cut_plane(context, model)
    temp.matrix_world = Matrix.Translation(center) @ normal.to_track_quat('Z', 'Y').to_matrix().to_4x4()
    try:
        _select_only(context, temp)
        model.select_set(True)
        return split_with_plane(context, kind, radius_mm, length_mm, tolerance_mm, wall_mm, mode, unit)
    except AddonError as exc:
        if 'no corta' in str(exc):
            raise AddonError('Las caras marcadas no encierran volumen: marca toda la parte que quieres separar, '
                             'no sólo una cara plana (para eso usa un inserto).') from None
        raise
    finally:
        remove_objects([temp])


# --------------------------------------------------------------------------- insertos
def _grown_cutter(context, cutter, tolerance):
    """Copia del cortador agrandada `tolerance` (unidades de Blender) hacia fuera."""
    col = _collection_of(cutter, context)
    shape = cutter.get(SHAPE)
    if shape and 'taller_tam' in cutter:
        m = cutter.matrix_world
        loc, rot, sca = m.decompose()
        size = [abs(s) * k + 2 * tolerance for s, k in zip(cutter['taller_tam'], sca)]
        bm = primitive_bmesh(shape, size)
        bm.transform(Matrix.Translation(loc) @ rot.to_matrix().to_4x4())
    else:
        bm = world_bmesh(context, cutter)
        bm.normal_update()
        for v in bm.verts:
            v.co += v.normal * tolerance
    try:
        return new_object(bm, 'Cortador_holgura', col)
    finally:
        bm.free()


def _world_copy(context, obj, name, collection):
    bm = world_bmesh(context, obj)
    try:
        return new_object(bm, name, collection, obj)
    finally:
        bm.free()


def inlay_from_cutter(context, tolerance_mm, unit='AUTO'):
    """Inserto = modelo ∩ cortador. Base = modelo − cortador agrandado por la holgura."""
    model = pick_model(context)
    cutter = find_helper(context, ROLE_CUTTER, model)
    if cutter is None:
        raise AddonError('Añade primero un cortador y húndelo en la zona que quieres sacar.')
    require_solid(context, model)
    require_solid(context, cutter)
    scale = factor(context.scene, unit, model)
    col = _collection_of(model, context)
    created = []
    try:
        cutter_world = _world_copy(context, cutter, 'Cortador_mundo', col); created.append(cutter_world)
        insert = _world_copy(context, model, model.name + '_Inserto', col); created.append(insert)
        boolean(context, insert, cutter_world, 'INTERSECT')
        if solid_info(insert)[1] <= 0:
            raise AddonError('El cortador no toca el modelo: colócalo sobre la zona.')
        base = _world_copy(context, model, model.name + '_Base', col); created.append(base)
        grown = _grown_cutter(context, cutter, tolerance_mm / scale); created.append(grown)
        boolean(context, base, grown)
        for tmp in (cutter_world, grown):
            created.remove(tmp)
        remove_objects([cutter_world, grown])
        _finish(context, model, [base, insert], [cutter])
        return base, insert
    except Exception:
        remove_objects(created)
        raise


def inlay_marked(context, depth_mm, tolerance_mm, wall_mm, unit='AUTO'):
    """Inserto a partir de caras marcadas (modo Objeto): el parche se engrosa hacia dentro."""
    model = pick_model(context)
    require_solid(context, model)
    mask = marked_mask(model)
    scale = factor(context.scene, unit, model)
    col = _collection_of(model, context)
    created = []
    try:
        whole = _mesh_bmesh(model)
        try:
            tree = BVHTree.FromBMesh(whole)
            eps = .001 / scale
            for f in whole.faces:
                if mask[f.index]:
                    loc, _n, _i, distance = tree.ray_cast(f.calc_center_median() - f.normal * eps, -f.normal)
                    if loc is None or distance < (depth_mm + tolerance_mm + wall_mm) / scale:
                        raise AddonError('El inserto atravesaría la pieza. Reduce la profundidad.')
            base = new_object(whole, model.name + '_Base', col, model); created.append(base)
        finally:
            whole.free()
        patch = _mesh_bmesh(model)
        try:
            bmesh.ops.delete(patch, geom=[f for f in patch.faces if not mask[f.index]], context='FACES')
            insert = new_object(patch, model.name + '_Inserto', col, model); created.append(insert)
        finally:
            patch.free()
        # Engrosar hacia dentro con bmesh (sin modificadores ni bpy.ops).
        bm = bmesh.new()
        bm.from_mesh(insert.data)
        try:
            bmesh.ops.solidify(bm, geom=bm.faces[:], thickness=-depth_mm / scale)
            bmesh.ops.recalc_face_normals(bm, faces=bm.faces[:])
            bm.to_mesh(insert.data)
        finally:
            bm.free()
        if not solid_info(insert)[0]:
            raise AddonError('El parche no admite un grosor cerrado válido.')
        grown = insert.copy()
        grown.data = insert.data.copy()
        col.objects.link(grown)
        created.append(grown)
        bm = bmesh.new()
        bm.from_mesh(grown.data)
        try:
            bm.normal_update()
            for v in bm.verts:
                v.co += v.normal * (tolerance_mm / scale)
            bm.to_mesh(grown.data)
        finally:
            bm.free()
        boolean(context, base, grown)
        created.remove(grown)
        remove_objects([grown])
        _finish(context, model, [base, insert])
        return base, insert
    except Exception:
        remove_objects(created)
        raise


# --------------------------------------------------------------------------- registro
def register_classes(classes, prop, settings):
    for cls in classes:
        bpy.utils.register_class(cls)
    setattr(bpy.types.Scene, prop, bpy.props.PointerProperty(type=settings))


def unregister_classes(classes, prop):
    if hasattr(bpy.types.Scene, prop):
        delattr(bpy.types.Scene, prop)
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)


UNIT_ITEMS = [('AUTO', 'Automático', 'Detecta mm, cm o m por el tamaño del modelo (como BigPrint)'),
              ('SCENE', 'Escala de la escena', 'Respeta los metros por unidad de la escena'),
              ('MM', '1 unidad = 1 mm', 'Para STL importados con coordenadas numéricas en milímetros'),
              ('M', '1 unidad = 1 m', 'Coordenadas expresadas en metros')]
SCOPE_ITEMS = [('SELECTED', 'Seleccionadas', 'Solo las mallas seleccionadas'),
               ('VISIBLE', 'Visibles', 'Todas las mallas visibles de la vista actual')]
SHAPE_ITEMS = [('CYLINDER', 'Cilindro', 'Ojos, botones, remaches'),
               ('BOX', 'Caja', 'Placas, logos rectangulares'),
               ('SPHERE', 'Esfera', 'Zonas redondeadas')]
REGION_ITEMS = [('PLANE', 'Plano de corte', 'Un plano que colocas en modo Objeto'),
                ('MARKED', 'Caras marcadas', 'Caras que dejaste seleccionadas en una sesión de Edición')]


# --------------------------------------------------------------------------- «clic y listo»
class ClickPlacer:
    """Mezcla para operadores modales en modo Objeto: un ayudante sigue al ratón sobre el
    modelo y con un clic se ejecuta la acción. Rueda = tamaño/giro, X/Y/Z = orientación,
    Esc o botón derecho = cancelar. El botón central sigue sirviendo para orbitar."""
    hint = ''

    # a implementar: create_helper(context, model), place(context, helper, loc, normal), commit(context, model, helper)
    def wheel(self, context, helper, up):
        pass

    def key(self, context, helper, key):
        return False

    def invoke(self, context, event):
        if context.area is None or context.area.type != 'VIEW_3D':
            self.report({'ERROR'}, 'Usa este botón desde la Vista 3D.')
            return {'CANCELLED'}
        try:
            self.model = pick_model(context)
            require_solid(context, self.model)
            bm = world_bmesh(context, self.model)
            try:
                self.tree = BVHTree.FromBMesh(bm)
            finally:
                bm.free()
            self.helper = self.create_helper(context, self.model)
        except AddonError as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        self.hit = None
        self.helper.hide_set(True)
        context.window_manager.modal_handler_add(self)
        context.area.header_text_set(self.hint)
        return {'RUNNING_MODAL'}

    def _ray(self, context, event):
        from bpy_extras import view3d_utils
        region = next((r for r in context.area.regions if r.type == 'WINDOW'), None)
        rv3d = context.area.spaces.active.region_3d
        if region is None or rv3d is None:
            return None
        x, y = event.mouse_x - region.x, event.mouse_y - region.y
        if not (0 <= x < region.width and 0 <= y < region.height):
            return None
        origin = view3d_utils.region_2d_to_origin_3d(region, rv3d, (x, y))
        direction = view3d_utils.region_2d_to_vector_3d(region, rv3d, (x, y))
        loc, normal, _i, _d = self.tree.ray_cast(origin, direction)
        if loc is None:
            return None
        if normal.dot(direction) > 0:
            normal = -normal
        return loc, normal

    def _end(self, context):
        context.area.header_text_set(None)

    def modal(self, context, event):
        # La vista se maneja como siempre: botón central gira, rueda hace zoom.
        if event.type == 'MIDDLEMOUSE':
            self._orbit = event.value == 'PRESS'
            return {'PASS_THROUGH'}
        if getattr(self, '_orbit', False) and event.type in {'MOUSEMOVE', 'INBETWEEN_MOUSEMOVE'}:
            return {'PASS_THROUGH'}
        if event.type in {'TRACKPADPAN', 'TRACKPADZOOM', 'NDOF_MOTION'} or (event.type.startswith('NUMPAD') and event.value == 'PRESS'):
            return {'PASS_THROUGH'}
        if event.type in {'ESC', 'RIGHTMOUSE'} and event.value == 'PRESS':
            remove_objects([self.helper])
            _select_only(context, self.model)
            self._end(context)
            return {'CANCELLED'}
        if event.type in {'WHEELUPMOUSE', 'WHEELDOWNMOUSE'}:
            if not event.shift:
                return {'PASS_THROUGH'}          # rueda: zoom normal · Shift + rueda: girar el ayudante
            self.wheel(context, self.helper, event.type == 'WHEELUPMOUSE')
            if self.hit:
                self.place(context, self.helper, *self.hit)
            return {'RUNNING_MODAL'}
        if event.value == 'PRESS' and self.key(context, self.helper, event.type):
            if self.hit:
                self.place(context, self.helper, *self.hit)
            return {'RUNNING_MODAL'}
        if event.type == 'MOUSEMOVE':
            self.hit = self._ray(context, event)
            self.helper.hide_set(self.hit is None)
            if self.hit:
                self.place(context, self.helper, *self.hit)
            return {'RUNNING_MODAL'}
        if event.type == 'LEFTMOUSE' and event.value == 'PRESS':
            if not self.hit:
                return {'RUNNING_MODAL'}
            self.helper.hide_set(False)
            self._end(context)
            try:
                message = self.commit(context, self.model, self.helper)
            except AddonError as exc:
                remove_objects([self.helper])
                _select_only(context, self.model)
                self.report({'ERROR'}, str(exc))
                return {'CANCELLED'}
            self.report({'INFO'}, message)
            return {'FINISHED'}
        return {'RUNNING_MODAL'}


AXIS_NAMES = {'Z': 'horizontal', 'X': 'vertical (X)', 'Y': 'vertical (Y)'}


def place_plane(plane, loc, axis):
    plane.location = loc
    plane.rotation_euler = {'X': (0, math.pi / 2, 0), 'Y': (math.pi / 2, 0, 0)}.get(axis, (0, 0, 0))


def place_cutter(cutter, loc, normal):
    """El cortador se centra en el punto y se alinea con la superficie: la mitad queda dentro."""
    cutter.location = loc
    cutter.rotation_euler = normal.to_track_quat('Z', 'Y').to_euler()


def commit_split(context, model, plane, *args):
    _select_only(context, plane)
    model.select_set(True)
    return split_with_plane(context, *args)


def commit_inlay(context, model, cutter, tolerance_mm, unit):
    _select_only(context, cutter)
    model.select_set(True)
    return inlay_from_cutter(context, tolerance_mm, unit)
