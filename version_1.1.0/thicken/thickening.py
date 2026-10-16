# SPDX-FileCopyrightText: 2026 Ricardo Lugaresi
# SPDX-License-Identifier: GPL-3.0-or-later

"""Thicken: engorda solo las zonas más finas que el mínimo imprimible.

Cada vértice de una zona fina se empuja hacia fuera la mitad de lo que le falta, así la
pared crece por las dos caras y queda centrada. El empuje se suaviza alrededor para que no
queden escalones entre la zona engordada y el resto, que no se toca.
"""
import numpy as np

from . import common, meshio, meshops as mo


def _thickness_all(V, F, max_dist, limit=400_000):
    """Grosor en todos los vértices (o en una muestra, propagada al vecino más cercano)."""
    used = np.unique(F)
    if len(used) <= limit:
        idx, t = mo.sample_thickness(V, F, max_dist, samples=len(used) + 1)
        full = np.full(len(V), float(max_dist)); full[idx] = t
        return full
    from mathutils import Vector
    from mathutils.kdtree import KDTree
    idx, t = mo.sample_thickness(V, F, max_dist, samples=limit)
    kd = KDTree(len(idx))
    for k, i in enumerate(idx):
        kd.insert(Vector(V[i]), k)
    kd.balance()
    full = np.full(len(V), float(max_dist))
    for i in used:
        full[i] = t[kd.find(Vector(V[i]))[1]]
    return full


def _thickness_at(V, F, idx, max_dist):
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    nrm = mo.vertex_normals_full(V, F)
    tree = BVHTree.FromPolygons(V.tolist(), F.tolist(), all_triangles=True)
    eps = 1e-4 * max(1.0, float(np.abs(V).max()))
    out = np.full(len(idx), float(max_dist))
    for k, i in enumerate(idx):
        hit = tree.ray_cast(Vector(V[i] - nrm[i] * eps), Vector(-nrm[i]), max_dist)
        if hit[0] is not None:
            out[k] = hit[3] + eps
    return out


def analyze(context, obj, min_wall, nozzle, unit='AUTO'):
    code, mm = common.resolve_unit(context.scene, unit, obj)
    V, F, _m = meshio.read_arrays(context, obj, mm)
    idx, t = mo.sample_thickness(V, F, max_dist=min_wall * 1.05)
    red = t < nozzle
    amber = (t >= nozzle) & (t < min_wall)
    pts = V[idx] / mm
    return {'red_share': float(red.mean()), 'amber_share': float(amber.mean()),
            'red_pts': pts[red], 'amber_pts': pts[amber]}


def _edges(F):
    return np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])


def thicken(context, obj, min_wall, unit='AUTO', blend_rounds=6, smooth_rounds=3):
    """Crea `<modelo>_grueso` con las zonas finas engordadas hasta `min_wall` mm."""
    code, mm = common.resolve_unit(context.scene, unit, obj)
    V, F, mat = meshio.read_arrays(context, obj, mm)
    V, F, keep = mo.weld(V, F)
    mat = mat[keep]
    t = _thickness_all(V, F, max_dist=min_wall * 1.05)
    # (dividir las zonas finas de mallas de caras grandes queda para una versión futura)
    deficit = np.clip(min_wall - t, 0.0, min_wall) * 0.5        # lo que crece cada cara
    nthin = int((deficit > 1e-6).sum())
    if not nthin:
        raise common.AddonError(f'No hay zonas más finas que {min_wall:g} mm: no hace falta engordar nada.')
    E = _edges(F)
    d0 = deficit.copy()
    d = deficit.copy()
    for _ in range(blend_rounds):          # rampa suave alrededor de la zona fina
        acc = np.zeros_like(d); cnt = np.zeros_like(d)
        np.add.at(acc, E[:, 0], d[E[:, 1]]); np.add.at(cnt, E[:, 0], 1)
        np.add.at(acc, E[:, 1], d[E[:, 0]]); np.add.at(cnt, E[:, 1], 1)
        avg = acc / np.maximum(cnt, 1)
        d = np.maximum(d0, np.maximum(d, avg * 0.85))
    for _ in range(smooth_rounds):         # sin escalones, pero sin bajar del mínimo
        acc = np.zeros_like(d); cnt = np.zeros_like(d)
        np.add.at(acc, E[:, 0], d[E[:, 1]]); np.add.at(cnt, E[:, 0], 1)
        np.add.at(acc, E[:, 1], d[E[:, 0]]); np.add.at(cnt, E[:, 1], 1)
        d = np.maximum(d0, (d + acc) / (1 + cnt))
    nrm = mo.vertex_normals_full(V, F)
    nrm = mo.smooth_vectors(nrm, E, rounds=2)
    V2 = V + nrm * d[:, None]
    # repasos: se vuelve a medir solo donde se engordó y se completa lo que falte
    zone = np.nonzero(d > 1e-6)[0]
    for _ in range(0):                    # (repasos desactivados: en mallas sucias empeoran)
        if not len(zone):
            break
        t2 = _thickness_at(V2, F, zone, min_wall * 1.05)
        extra = np.clip(min_wall - t2, 0.0, min_wall) * 0.5
        if not (extra > 0.02).any():
            break
        add = np.zeros(len(V)); add[zone] = extra
        for _r in range(2):
            acc = np.zeros_like(add); cnt = np.zeros_like(add)
            np.add.at(acc, E[:, 0], add[E[:, 1]]); np.add.at(cnt, E[:, 0], 1)
            np.add.at(acc, E[:, 1], add[E[:, 0]]); np.add.at(cnt, E[:, 1], 1)
            add = np.maximum(add, acc / np.maximum(cnt, 1) * 0.7)
        n2 = mo.smooth_vectors(mo.vertex_normals_full(V2, F), E, rounds=2)
        V2 = V2 + n2 * add[:, None]
        d = d + add
    name = f'{obj.name}_grueso'
    out = meshio.new_object_like(context, obj, name, V2 / mm, F, mat)
    out[common.UNIT_PROP] = code
    out['taller_origen'] = obj.name
    obj.hide_set(True); obj.hide_render = True
    meshio.select_only(context, out)
    grown = float(d.max() * 2)
    return out, nthin / max(1, len(np.unique(F))), grown


# --------------------------------------------------------------------------- solo grosor
AXES = {'X': 0, 'Y': 1, 'Z': 2}


def plate_axis(V, axis='AUTO'):
    """Eje del grosor: el que se pide o la medida más pequeña de la pieza."""
    if axis in AXES:
        return AXES[axis]
    return int(np.argmin(V.max(0) - V.min(0)))


def thicken_plate(context, obj, target, axis='AUTO', side='BOTH', unit='AUTO', smooth_rounds=4):
    """Engorda solo a lo largo de un eje: la silueta vista desde ese eje no cambia ni una
    décima (sirve para incrustaciones, logos, aletas, placas). Cada columna de la pieza que
    mida menos de `target` mm de grosor crece hasta `target`; las paredes laterales se estiran
    en proporción, sin moverse de su sitio en planta."""
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    code, mm = common.resolve_unit(context.scene, unit, obj)
    V, F, mat = meshio.read_arrays(context, obj, mm)
    V, F, keep = mo.weld(V, F)
    mat = mat[keep]
    k = plate_axis(V, axis)
    a = np.zeros(3); a[k] = 1.0
    nrm = mo.vertex_normals_full(V, F)
    tree = BVHTree.FromPolygons(V.tolist(), F.tolist(), all_triangles=True)
    span = float((V.max(0) - V.min(0)).max()) * 2 + 1.0
    eps = 1e-3 * max(1.0, float(np.abs(V).max()) * 1e-2)
    used = np.unique(F)
    up = np.full(len(V), np.nan); down = np.full(len(V), np.nan)
    va, vb = Vector(a), Vector(-a)
    for i in used:
        o = Vector(V[i] - nrm[i] * eps)          # un pelín dentro de la pieza
        h1 = tree.ray_cast(o, va, span)
        h2 = tree.ray_cast(o, vb, span)
        if h1[0] is not None and h2[0] is not None:
            na = float(nrm[i] @ a)                  # se corrige el pelín de entrada
            up[i], down[i] = max(h1[3] - eps * na, 0.0), max(h2[3] + eps * na, 0.0)
    ok = ~np.isnan(up)
    T = np.where(ok, up + down, np.inf)                   # grosor de la columna
    extra = np.where(ok, np.clip(target - T, 0.0, None), 0.0)
    nthin = int((extra > 1e-4).sum())
    if not nthin:
        raise common.AddonError(f'La pieza ya mide {target:g} mm o más de grosor en el eje {"XYZ"[k]}: no hace falta engordar.')
    E = _edges(F)
    need = extra.copy()
    for _ in range(smooth_rounds):                        # sin escalones entre columnas vecinas
        acc = np.zeros_like(extra); cnt = np.zeros_like(extra)
        np.add.at(acc, E[:, 0], extra[E[:, 1]]); np.add.at(cnt, E[:, 0], 1)
        np.add.at(acc, E[:, 1], extra[E[:, 0]]); np.add.at(cnt, E[:, 1], 1)
        extra = np.maximum(need, (extra + acc) / (1 + cnt))
    s = np.where(ok, down / np.where(ok, T, 1.0), 0.5)    # 0 = cara de abajo, 1 = cara de arriba
    s = np.clip(s, 0.0, 1.0)
    shift = extra * (s if side == 'UP' else (s - 0.5))
    V2 = V + a[None, :] * shift[:, None]
    out = meshio.new_object_like(context, obj, f'{obj.name}_grueso', V2 / mm, F, mat)
    out[common.UNIT_PROP] = code
    out['taller_origen'] = obj.name
    obj.hide_set(True); obj.hide_render = True
    meshio.select_only(context, out)
    return out, nthin / max(1, len(used)), float(extra.max()), 'XYZ'[k]
