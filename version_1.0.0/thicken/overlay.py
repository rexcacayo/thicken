"""Puntos de colores sobre el modelo (zonas finas) sin tocar la malla.

Se dibujan con un manejador de la vista 3D; quitar el aviso es borrar los puntos."""
import bpy
import gpu
from gpu_extras.batch import batch_for_shader

_DATA = {'red': [], 'amber': [], 'handle': None}
RED = (0.92, 0.27, 0.23, 1.0)
AMBER = (1.0, 0.71, 0.28, 1.0)


def _draw():
    if not _DATA['red'] and not _DATA['amber']:
        return
    shader = gpu.shader.from_builtin('UNIFORM_COLOR')
    gpu.state.blend_set('ALPHA')
    gpu.state.depth_test_set('LESS_EQUAL')
    gpu.state.point_size_set(5.0)
    for pts, color in ((_DATA['amber'], AMBER), (_DATA['red'], RED)):
        if pts:
            batch = batch_for_shader(shader, 'POINTS', {'pos': pts})
            shader.uniform_float('color', color)
            batch.draw(shader)
    gpu.state.point_size_set(1.0)
    gpu.state.depth_test_set('NONE')
    gpu.state.blend_set('NONE')


def register():
    if _DATA['handle'] is None:
        _DATA['handle'] = bpy.types.SpaceView3D.draw_handler_add(_draw, (), 'WINDOW', 'POST_VIEW')


def unregister():
    if _DATA['handle'] is not None:
        bpy.types.SpaceView3D.draw_handler_remove(_DATA['handle'], 'WINDOW')
        _DATA['handle'] = None
    clear()


def show(red, amber):
    """`red` y `amber`: listas de puntos (x, y, z) en coordenadas de mundo (unidades de Blender)."""
    _DATA['red'] = [tuple(map(float, p)) for p in red]
    _DATA['amber'] = [tuple(map(float, p)) for p in amber]
    _redraw()


def clear():
    _DATA['red'], _DATA['amber'] = [], []
    _redraw()


def active():
    return bool(_DATA['red'] or _DATA['amber'])


def _redraw():
    wm = getattr(bpy.context, 'window_manager', None)
    for win in (wm.windows if wm else ()):
        for area in win.screen.areas:
            if area.type == 'VIEW_3D':
                area.tag_redraw()
