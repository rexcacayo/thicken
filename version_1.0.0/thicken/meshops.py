"""Separar por colores: una pieza por color, con el hueco hecho en la pieza base.

Funciona con cualquier malla pintada por caras (materiales): 3MF pintados, OBJ con
materiales, modelos de IA con colores asignados. Todo en milímetros y en numpy; lo único
de Blender que se usa aquí es el BVHTree (rayos para medir el grosor).

Pasos:
 1. Soldar: los 3MF pintados traen las fronteras de color sin unir. Se unen los vértices
    que coinciden y la malla queda cerrada.
 2. Zonas: grupos de caras conectadas del mismo color.
 3. Limpiar motas: zonas diminutas (triangulitos sueltos) pasan al color de alrededor.
 4. Clasificar: cada zona tiene un ancho típico (2·área/contorno). Si es más estrecha que
    el mínimo imprimible, se marca «pintar» y se queda en la base.
 5. Piezas: cada zona de color se convierte en un cascarón macizo de grosor `depth`
    (menos donde el modelo es más fino). Su cara de fuera es la del modelo.
 6. Hueco: se resta a la base el mismo cascarón con holgura.
"""
import math

import numpy as np

# --------------------------------------------------------------------------- malla
def weld(V, F, tol=1e-4):
    """Une vértices a menos de `tol` mm y quita triángulos degenerados o repetidos."""
    keys = np.round(V / tol).astype(np.int64)
    _u, idx, inv = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    V2 = V[idx]
    F2 = inv[F]
    ok = (F2[:, 0] != F2[:, 1]) & (F2[:, 1] != F2[:, 2]) & (F2[:, 2] != F2[:, 0])
    keep = np.nonzero(ok)[0]
    F2 = F2[keep]
    # triángulos duplicados (mismos tres vértices): se queda uno
    s = np.sort(F2, axis=1)
    _u2, first = np.unique(s, axis=0, return_index=True)
    first = np.sort(first)
    return V2, F2[first], keep[first]


def edge_table(F):
    """Aristas sin orientación, cuántas caras tocan cada una y las dos caras de cada arista."""
    m = len(F)
    he = np.stack([F, np.roll(F, -1, axis=1)], axis=2).reshape(-1, 2)     # medias aristas a→b
    face_of = np.repeat(np.arange(m), 3)
    lo, hi = np.minimum(he[:, 0], he[:, 1]), np.maximum(he[:, 0], he[:, 1])
    key = lo.astype(np.int64) * (int(F.max()) + 1) + hi
    uk, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    order = np.argsort(inv, kind='stable')
    start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    two = cnt == 2
    fa = face_of[order[start[two]]]
    fb = face_of[order[start[two] + 1]]
    ends = np.stack([lo[order[start]], hi[order[start]]], axis=1)
    return {'cnt': cnt, 'fa': fa, 'fb': fb, 'ends2': ends[two], 'inv': inv, 'he': he, 'face_of': face_of}


def is_closed(table):
    return bool(np.all(table['cnt'] == 2))


def face_normals_areas(V, F):
    c = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    a = np.linalg.norm(c, axis=1)
    n = c / np.maximum(a, 1e-30)[:, None]
    return n, a * 0.5


# --------------------------------------------------------------------------- zonas
def components(n, a, b):
    """Componentes conexas (union-find vectorizado) de un grafo con aristas (a, b)."""
    parent = np.arange(n)
    if len(a) == 0:
        return parent
    for _ in range(200):
        pa, pb = parent[a], parent[b]
        diff = pa != pb
        if not diff.any():
            break
        lo, hi = np.minimum(pa[diff], pb[diff]), np.maximum(pa[diff], pb[diff])
        np.minimum.at(parent, hi, lo)
        while True:
            p2 = parent[parent]
            if np.array_equal(p2, parent):
                break
            parent = p2
    _u, lab = np.unique(parent, return_inverse=True)
    return lab.reshape(-1)


def regions(F, mat, table):
    same = mat[table['fa']] == mat[table['fb']]
    return components(len(F), table['fa'][same], table['fb'][same])


def clean_specks(V, F, mat, table, min_area=1.0, passes=3):
    """Las zonas de menos de `min_area` mm² toman el color que más las rodea."""
    mat = mat.copy()
    _n, area = face_normals_areas(V, F)
    e = table['ends2']
    elen = np.linalg.norm(V[e[:, 0]] - V[e[:, 1]], axis=1)
    nmat = int(mat.max()) + 1
    changed_total = 0
    for _ in range(passes):
        lab = regions(F, mat, table)
        nreg = int(lab.max()) + 1
        rarea = np.bincount(lab, weights=area, minlength=nreg)
        small = rarea < min_area
        fa, fb = table['fa'], table['fb']
        cross = lab[fa] != lab[fb]
        ra = np.concatenate([lab[fa][cross], lab[fb][cross]])
        mb = np.concatenate([mat[fb][cross], mat[fa][cross]])
        w = np.concatenate([elen[cross], elen[cross]])
        sel = small[ra]
        if not sel.any():
            break
        acc = np.bincount(ra[sel] * nmat + mb[sel], weights=w[sel], minlength=nreg * nmat).reshape(nreg, nmat)
        best = acc.argmax(1)
        has = acc.max(1) > 0
        regmat = np.zeros(nreg, dtype=mat.dtype)
        regmat[lab] = mat
        target = np.where(small & has, best, regmat)
        newmat = target[lab]
        n_changed = int((newmat != mat).sum())
        mat = newmat
        changed_total += n_changed
        if n_changed == 0:
            break
    return mat, changed_total


def region_stats(V, F, mat, table, lab):
    """Por zona: color, área (mm²), contorno (mm), ancho típico (mm), nº de caras."""
    _n, area = face_normals_areas(V, F)
    nreg = int(lab.max()) + 1
    e = table['ends2']
    elen = np.linalg.norm(V[e[:, 0]] - V[e[:, 1]], axis=1)
    fa, fb = table['fa'], table['fb']
    cross = lab[fa] != lab[fb]
    per = np.bincount(lab[fa][cross], weights=elen[cross], minlength=nreg) + \
        np.bincount(lab[fb][cross], weights=elen[cross], minlength=nreg)
    rarea = np.bincount(lab, weights=area, minlength=nreg)
    rmat = np.zeros(nreg, dtype=np.int64)
    rmat[lab] = mat
    width = np.where(per > 0, 2 * rarea / np.maximum(per, 1e-12), np.inf)
    nfaces = np.bincount(lab, minlength=nreg)
    return {'mat': rmat, 'area': rarea, 'perimeter': per, 'width': width, 'faces': nfaces}


def color_summary(stats, base, min_width):
    """Resumen por color para el panel."""
    out = {}
    for m in np.unique(stats['mat']):
        sel = stats['mat'] == m
        w = stats['width'][sel]
        a = stats['area'][sel]
        fine = w < min_width
        out[int(m)] = {
            'zones': int(sel.sum()), 'area': float(a.sum()),
            'fine_zones': int(fine.sum()), 'fine_area': float(a[fine].sum()),
            'min_width': float(w[np.isfinite(w)].min()) if np.isfinite(w).any() else float('inf'),
            'base': int(m) == int(base),
        }
    return out


# --------------------------------------------------------------------------- cascarones
def _region_geometry(V, F, faces):
    """Vértices locales, caras locales y aristas de borde orientadas de una zona."""
    Fr = F[faces]
    used, local = np.unique(Fr, return_inverse=True)
    Fl = local.reshape(-1, 3)
    he = np.stack([Fl, np.roll(Fl, -1, axis=1)], axis=2).reshape(-1, 2)
    lo, hi = np.minimum(he[:, 0], he[:, 1]), np.maximum(he[:, 0], he[:, 1])
    key = lo.astype(np.int64) * (len(used) + 1) + hi
    _uk, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    border = he[cnt[inv.reshape(-1)] == 1]
    return used, Fl, border


def _vertex_normals(Vr, Fl):
    n, a = face_normals_areas(Vr, Fl)
    vn = np.zeros_like(Vr)
    for k in range(3):
        np.add.at(vn, Fl[:, k], n * a[:, None])
    return vn / np.maximum(np.linalg.norm(vn, axis=1), 1e-30)[:, None]


def smooth_vectors(vecs, edges, rounds=8, fixed=None):
    """Promedia un campo de direcciones con sus vecinos y lo renormaliza: los bordes de
    pintura en escalera dan normales locas; suavizadas, el cascarón sale limpio."""
    v = vecs.copy()
    for _ in range(rounds):
        acc = v.copy()
        np.add.at(acc, edges[:, 0], v[edges[:, 1]])
        np.add.at(acc, edges[:, 1], v[edges[:, 0]])
        if fixed is not None:
            acc[fixed] = v[fixed]
        v = acc / np.maximum(np.linalg.norm(acc, axis=1), 1e-30)[:, None]
    return v


def vertex_normals_full(V, F):
    n, a = face_normals_areas(V, F)
    vn = np.zeros_like(V)
    for k in range(3):
        np.add.at(vn, F[:, k], n * a[:, None])
    return vn / np.maximum(np.linalg.norm(vn, axis=1), 1e-30)[:, None]


def region_normals(vn_full, used, Fl, rounds=8):
    edges = np.concatenate([Fl[:, [0, 1]], Fl[:, [1, 2]], Fl[:, [2, 0]]])
    return smooth_vectors(vn_full[used], edges, rounds)


def _border_inward(Vr, Fl, border, vn):
    """Dirección, dentro de la superficie, hacia el interior de la zona en cada vértice de borde."""
    d = np.zeros_like(Vr)
    a, b = Vr[border[:, 0]], Vr[border[:, 1]]
    # la cara está a la izquierda de a→b vista desde fuera: n × (b - a)
    left = np.cross(vn[border[:, 0]] + vn[border[:, 1]], b - a)
    for col in (0, 1):
        np.add.at(d, border[:, col], left)
    d -= vn * (d * vn).sum(1)[:, None]
    ln = np.linalg.norm(d, axis=1)
    d = d / np.maximum(ln, 1e-30)[:, None]
    # suavizar a lo largo del borde: la escalera de la pintura no debe dar pinchos
    if len(border):
        d = smooth_vectors(d, border, rounds=6)
        d -= vn * (d * vn).sum(1)[:, None]
        d = d / np.maximum(np.linalg.norm(d, axis=1), 1e-30)[:, None]
    return d, ln > 0


def _smooth_scalar(values, Fl, rounds=4):
    v = values.copy()
    edges = np.concatenate([Fl[:, [0, 1]], Fl[:, [1, 2]], Fl[:, [2, 0]]])
    for _ in range(rounds):
        acc = np.zeros_like(v); cnt = np.zeros_like(v)
        np.add.at(acc, edges[:, 0], v[edges[:, 1]]); np.add.at(cnt, edges[:, 0], 1)
        np.add.at(acc, edges[:, 1], v[edges[:, 0]]); np.add.at(cnt, edges[:, 1], 1)
        v = np.minimum(v, (v + acc) / (1 + cnt))     # solo se adelgaza, nunca engorda
    return v


def shell(Vr, Fl, border, vn, depth_v, outer_offset=0.0, inset=0.0):
    """Cascarón cerrado: cara de fuera = la zona (desplazada `outer_offset` hacia fuera),
    cara de dentro = la zona hundida `depth_v` y paredes por el borde. `inset` encoge la
    zona por el borde (holgura lateral de la pieza)."""
    outer = Vr.copy()
    if inset != 0 and len(border):
        dirs, ok = _border_inward(Vr, Fl, border, vn)
        bv = np.unique(border)
        bv = bv[ok[bv]]
        outer[bv] += dirs[bv] * inset
    inner = outer - vn * depth_v[:, None]
    outer = outer + vn * outer_offset
    nv = len(Vr)
    verts = np.vstack([outer, inner])
    a, b = border[:, 0], border[:, 1]
    walls = np.concatenate([np.stack([b, a, a + nv], 1), np.stack([b, a + nv, b + nv], 1)])
    faces = np.vstack([Fl, Fl[:, ::-1] + nv, walls])
    return verts, faces


def signed_volume(Vs, Fs):
    a, b, c = Vs[Fs[:, 0]], Vs[Fs[:, 1]], Vs[Fs[:, 2]]
    return float(np.einsum('ij,ij->i', a, np.cross(b, c)).sum() / 6.0)


def thickness(V, vn, tree, idx, max_dist):
    """Grosor del modelo bajo cada vértice (rayo hacia dentro), recortado a `max_dist`."""
    from mathutils import Vector
    out = np.full(len(idx), max_dist)
    eps = 1e-3
    for k, i in enumerate(idx):
        o = V[i] - vn[k] * eps
        hit = tree.ray_cast(Vector(o), Vector(-vn[k]), max_dist)
        if hit[0] is not None:
            out[k] = hit[3] + eps
    return out


# --------------------------------------------------------------------------- uniones en T
def fix_t_junctions(V, F, mat, max_steps=64, rel_tol=1e-3):
    """Cose las uniones en T de los 3MF pintados.

    Al pintar, el programa subdivide los triángulos de la frontera de color, pero el vecino
    no: queda una arista larga en un lado y varias cortas en el otro, con los mismos
    extremos. Se busca, para cada arista abierta a→b, un camino de aristas abiertas casi
    alineadas que vaya de b a a; si existe, sus vértices se insertan en la arista larga y
    ese triángulo se rehace en abanico desde su centro. Devuelve (V, F, mat, cosidas)."""
    table = edge_table(F)
    cnt, inv, he, face_of = table['cnt'], table['inv'], table['he'], table['face_of']
    open_idx = np.nonzero(cnt[inv] == 1)[0]
    if not len(open_idx):
        return V, F, mat, 0
    # vecinos por aristas abiertas, sin mirar la orientación (algunos 3MF la traen mezclada)
    starts = {}
    for h in open_idx:
        a0, b0 = int(he[h, 0]), int(he[h, 1])
        starts.setdefault(a0, []).append((int(h), b0))
        starts.setdefault(b0, []).append((int(h), a0))
    inserts = {}            # media arista larga -> lista de vértices intermedios (de a hacia b)
    for h in open_idx:
        a, b = int(he[h, 0]), int(he[h, 1])
        pa, pb = V[a], V[b]
        seg = pa - pb                     # recorremos de b hacia a
        L = float(np.linalg.norm(seg))
        if L <= 0:
            continue
        u = seg / L
        tol = max(rel_tol * L, 1e-6)
        cur, path, along = b, [], 0.0
        ok = False
        for _ in range(max_steps):
            nxt = None
            for h2, c in starts.get(cur, ()):
                if h2 == h:
                    continue
                d = V[c] - pb
                t = float(d @ u)
                if t <= along + 1e-12 or t > L + tol:
                    continue
                if np.linalg.norm(d - u * t) > tol:
                    continue
                nxt = (c, t)
                break
            if nxt is None:
                break
            cur, along = nxt
            if cur == a:
                ok = True
                break
            path.append(cur)
        if ok and path:
            inserts[int(h)] = path[::-1]          # de a hacia b
    if not inserts:
        return V, F, mat, 0
    # rehacer los triángulos afectados
    by_face = {}
    for h, pts in inserts.items():
        by_face.setdefault(int(face_of[h]), {})[h % 3] = pts
    newV = [V]
    newF, newM = [], []
    nv = len(V)
    drop = np.zeros(len(F), bool)
    for f, edges in by_face.items():
        tri = F[f]
        poly = []
        for k in range(3):
            poly.append(int(tri[k]))
            poly.extend(edges.get(k, []))
        centre = V[poly].mean(0)
        newV.append(centre[None, :])
        cidx = nv; nv += 1
        for i in range(len(poly)):
            newF.append((cidx, poly[i], poly[(i + 1) % len(poly)]))
            newM.append(mat[f])
        drop[f] = True
    V2 = np.vstack(newV)
    keep = ~drop
    F2 = np.vstack([F[keep], np.array(newF, dtype=F.dtype)])
    M2 = np.concatenate([mat[keep], np.array(newM, dtype=mat.dtype)])
    return V2, F2, M2, len(inserts)


def border_loops(border):
    """Ordena las aristas de borde (a→b) en lazos cerrados de índices."""
    nxt = {}
    bad = set()
    for a, b in border:
        a, b = int(a), int(b)
        if a in nxt:
            bad.add(a)
        nxt[a] = b
    loops, seen = [], set()
    for start in nxt:
        if start in seen or start in bad:
            continue
        loop, cur = [], start
        while cur not in seen and cur in nxt and cur not in bad:
            seen.add(cur); loop.append(cur); cur = nxt[cur]
        if cur == start and len(loop) >= 3:
            loops.append(np.array(loop))
    return loops


def _smooth_loop(P, window):
    """Media móvil por longitud de arco (lazo cerrado)."""
    seg = np.linalg.norm(np.roll(P, -1, axis=0) - P, axis=1)
    L = seg.sum()
    if L <= 0 or window <= 0:
        return P
    s = np.concatenate([[0.0], np.cumsum(seg)[:-1]])
    h = min(window / 8.0, L / 64.0)
    n = max(16, int(L / h))
    u = np.linspace(0, L, n, endpoint=False)
    Pc = np.vstack([P, P[:1]]); sc = np.concatenate([s, [L]])
    R = np.stack([np.interp(u, sc, Pc[:, k]) for k in range(3)], 1)
    # en lazos más cortos que la ventana, la ventana se limita a medio lazo
    k = max(1, min(int(round(window / (L / n))), (n - 1) // 2))
    kern = np.ones(2 * k + 1) / (2 * k + 1)
    Rs = np.stack([np.convolve(np.concatenate([R[-k:, j], R[:, j], R[:k, j]]), kern, 'valid') for j in range(3)], 1)
    uc = np.concatenate([u, [L]]); Rc = np.vstack([Rs, Rs[:1]])
    return np.stack([np.interp(s, uc, Rc[:, j]) for j in range(3)], 1)


def relax_patch(Vr, Fl, border, tree, window=1.2, band=2.0, rounds=60):
    """Alisa la frontera de color (la escalera de la pintura) sin cambiar la forma.

    1. Cada lazo de borde se suaviza con una media móvil de `window` mm de arco.
    2. Los vértices interiores a menos de `band` mm del borde se relajan para seguirlo
       (sin pliegues).
    3. Todo se devuelve a la superficie original del modelo."""
    from mathutils import Vector
    from mathutils.kdtree import KDTree
    P = Vr.copy()
    if not len(border):
        return P
    for loop in border_loops(border):
        P[loop] = _smooth_loop(P[loop], window)
    bverts = np.unique(border)
    kd = KDTree(len(bverts))
    for i, v in enumerate(bverts):
        kd.insert(Vector(Vr[v]), i)
    kd.balance()
    dist = np.array([kd.find(Vector(p))[2] for p in Vr])
    is_b = np.zeros(len(P), bool); is_b[bverts] = True
    move = (dist < band) & ~is_b
    if move.any():
        edges = np.concatenate([Fl[:, [0, 1]], Fl[:, [1, 2]], Fl[:, [2, 0]]])
        edges = edges[move[edges[:, 0]] | move[edges[:, 1]]]
        for _ in range(rounds):
            acc = np.zeros_like(P); cnt = np.zeros(len(P))
            np.add.at(acc, edges[:, 0], P[edges[:, 1]]); np.add.at(cnt, edges[:, 0], 1)
            np.add.at(acc, edges[:, 1], P[edges[:, 0]]); np.add.at(cnt, edges[:, 1], 1)
            sel = move & (cnt > 0)
            P[sel] = P[sel] + (acc[sel] / cnt[sel][:, None] - P[sel]) * 0.6
    for i in np.nonzero(move | is_b)[0]:
        hit = tree.find_nearest(Vector(P[i]))
        if hit[0] is not None:
            P[i] = hit[0]
    return P


def smooth_offsets(vn, d, Fl, border, window=1.5, rounds=40):
    """Suaviza normales y grosor a lo largo del borde (media por arco) y en una franja
    interior, para que la cara de dentro y las paredes salgan lisas."""
    vn = vn.copy(); d = d.copy()
    loops = border_loops(border) if len(border) else []
    for loop in loops:
        vn[loop] = _smooth_loop(vn[loop], window)
        d3 = np.stack([d[loop], np.zeros(len(loop)), np.zeros(len(loop))], 1)
        d[loop] = _smooth_loop(d3, window)[:, 0]
    vn /= np.maximum(np.linalg.norm(vn, axis=1), 1e-30)[:, None]
    if loops:
        edges = np.concatenate([Fl[:, [0, 1]], Fl[:, [1, 2]], Fl[:, [2, 0]]])
        fixed = np.zeros(len(vn), bool); fixed[np.unique(border)] = True
        for _ in range(rounds):
            acc = vn.copy(); cnt = np.ones(len(vn))
            np.add.at(acc, edges[:, 0], vn[edges[:, 1]]); np.add.at(cnt, edges[:, 0], 1)
            np.add.at(acc, edges[:, 1], vn[edges[:, 0]]); np.add.at(cnt, edges[:, 1], 1)
            new = acc / np.maximum(np.linalg.norm(acc, axis=1), 1e-30)[:, None]
            vn[~fixed] = new[~fixed]
    return vn, d


def subdivide_borders(V, F, mat, target=0.3, rounds=4, ring=1):
    """Refina los triángulos junto a las fronteras de color hasta que sus aristas midan
    menos de `target` mm. En modelos con pocas caras la frontera va por triángulos grandes
    y el corte sale a dientes; con triángulos finos, el alisado de la frontera puede
    dibujar una línea limpia. Las uniones en T que deja se cosen después."""
    total = 0
    for _ in range(rounds):
        table = edge_table(F)
        fa, fb = table['fa'], table['fb']
        diff = mat[fa] != mat[fb]
        bverts = np.zeros(len(V), bool)
        bverts[table['ends2'][diff].ravel()] = True
        sel = bverts[F].any(1)
        for _r in range(ring):                 # un anillo más alrededor
            near = np.zeros(len(V), bool); near[F[sel].ravel()] = True
            sel = near[F].any(1)
        e = np.stack([np.linalg.norm(V[F[:, k]] - V[F[:, (k + 1) % 3]], axis=1) for k in range(3)], 1)
        sel &= e.max(1) > target
        if not sel.any():
            break
        idx = np.nonzero(sel)[0]
        Fs = F[idx]
        a = np.concatenate([Fs[:, 0], Fs[:, 1], Fs[:, 2]])
        b = np.concatenate([Fs[:, 1], Fs[:, 2], Fs[:, 0]])
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        key = lo.astype(np.int64) * (len(V) + 1) + hi
        uk, inv = np.unique(key, return_inverse=True)
        inv = inv.reshape(-1)
        mids = (V[uk // (len(V) + 1)] + V[uk % (len(V) + 1)]) / 2
        m = len(idx)
        mid_id = len(V) + inv
        m01, m12, m20 = mid_id[:m], mid_id[m:2 * m], mid_id[2 * m:]
        v0, v1, v2 = Fs[:, 0], Fs[:, 1], Fs[:, 2]
        newF = np.concatenate([np.stack([v0, m01, m20], 1), np.stack([m01, v1, m12], 1),
                               np.stack([m20, m12, v2], 1), np.stack([m01, m12, m20], 1)])
        newM = np.tile(mat[idx], 4)
        keep = np.ones(len(F), bool); keep[idx] = False
        V = np.vstack([V, mids])
        F = np.vstack([F[keep], newF])
        mat = np.concatenate([mat[keep], newM])
        total += m
        # coser las uniones en T con los vecinos sin dividir
        for _k in range(3):
            V, F, mat, n = fix_t_junctions(V, F, mat)
            if not n:
                break
    return V, F, mat, total


# =========================================================================== Mesh Doctor
def orientation_errors(table):
    """Aristas cuyas dos caras la recorren en el mismo sentido (normales incoherentes)."""
    cnt, inv, he = table['cnt'], table['inv'], table['he']
    order = np.argsort(inv, kind='stable')
    start = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    two = np.nonzero(cnt == 2)[0]
    h1 = order[start[two]]; h2 = order[start[two] + 1]
    same = he[h1, 0] == he[h2, 0]
    return int(same.sum())


def vertex_components(nv, F):
    a = np.concatenate([F[:, 0], F[:, 1]]); b = np.concatenate([F[:, 1], F[:, 2]])
    vlab = components(nv, a, b)
    return vlab[F[:, 0]]


def component_stats(V, F, flab, table):
    """Por pieza: caras, área, volumen con signo, cerrada, caja."""
    n = int(flab.max()) + 1 if len(flab) else 0
    _nrm, area = face_normals_areas(V, F)
    vol6 = np.einsum('ij,ij->i', V[F[:, 0]], np.cross(V[F[:, 1]], V[F[:, 2]]))
    open_face = np.zeros(len(F), bool)
    bad = table['cnt'][table['inv']] != 2
    open_face[table['face_of'][bad]] = True
    out = {
        'faces': np.bincount(flab, minlength=n),
        'area': np.bincount(flab, weights=area, minlength=n),
        'volume': np.bincount(flab, weights=vol6, minlength=n) / 6.0,
        'open': np.bincount(flab, weights=open_face.astype(float), minlength=n) > 0,
    }
    lo = np.full((n, 3), np.inf); hi = np.full((n, 3), -np.inf)
    P = V[F[:, 0]]
    for k in range(3):
        np.minimum.at(lo[:, k], flab, P[:, k]); np.maximum.at(hi[:, k], flab, P[:, k])
    out['lo'], out['hi'] = lo, hi
    return out


def boundary_loops_all(table):
    """Lazos de borde (agujeros) a partir de las medias aristas abiertas."""
    he, cnt, inv = table['he'], table['cnt'], table['inv']
    open_he = he[cnt[inv] == 1]
    return border_loops(open_he)
    open_he = he[cnt[inv] == 1]
    return border_loops(open_he)


# =========================================================================== grosor
def sample_thickness(V, F, max_dist, samples=30000, seed=1):
    """Grosor bajo vértices de muestra: rayo desde cada vértice hacia dentro (contra su
    normal). Devuelve (índices de vértice, grosor en las unidades de V; max_dist si no choca)."""
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree
    nrm = vertex_normals_full(V, F)
    tree = BVHTree.FromPolygons(V.tolist(), F.tolist(), all_triangles=True)
    idx = np.unique(F)
    if len(idx) > samples:
        idx = np.sort(np.random.default_rng(seed).choice(idx, samples, replace=False))
    out = np.full(len(idx), float(max_dist))
    eps = 1e-4 * max(1.0, float(np.abs(V).max()))
    for k, i in enumerate(idx):
        hit = tree.ray_cast(Vector(V[i] - nrm[i] * eps), Vector(-nrm[i]), max_dist)
        if hit[0] is not None:
            out[k] = hit[3] + eps
    return idx, out


def subdivide_faces(V, F, mat, sel, target, rounds=3):
    """Divide en 4 los triángulos marcados cuyas aristas pasan de `target` (y cose las
    uniones en T que deja con sus vecinos). `sel`: máscara booleana por cara."""
    vmask = np.zeros(len(V), bool)
    vmask[F[sel].ravel()] = True
    for _ in range(rounds):
        sel = vmask[F].all(1)
        e = np.stack([np.linalg.norm(V[F[:, k]] - V[F[:, (k + 1) % 3]], axis=1) for k in range(3)], 1)
        s = sel & (e.max(1) > target)
        if not s.any():
            break
        idx = np.nonzero(s)[0]
        Fs = F[idx]
        a = np.concatenate([Fs[:, 0], Fs[:, 1], Fs[:, 2]])
        b = np.concatenate([Fs[:, 1], Fs[:, 2], Fs[:, 0]])
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        key = lo.astype(np.int64) * (len(V) + 1) + hi
        uk, inv = np.unique(key, return_inverse=True)
        inv = inv.reshape(-1)
        mids = (V[uk // (len(V) + 1)] + V[uk % (len(V) + 1)]) / 2
        m = len(idx)
        mid = len(V) + inv
        m01, m12, m20 = mid[:m], mid[m:2 * m], mid[2 * m:]
        v0, v1, v2 = Fs[:, 0], Fs[:, 1], Fs[:, 2]
        newF = np.concatenate([np.stack([v0, m01, m20], 1), np.stack([m01, v1, m12], 1),
                               np.stack([m20, m12, v2], 1), np.stack([m01, m12, m20], 1)])
        keep = np.ones(len(F), bool); keep[idx] = False
        V = np.vstack([V, mids])
        F = np.vstack([F[keep], newF])
        mat = np.concatenate([mat[keep], np.tile(mat[idx], 4)])
        for _k in range(3):
            V, F, mat, n = fix_t_junctions(V, F, mat)
            if not n:
                break
        vmask = np.concatenate([vmask, np.ones(len(V) - len(vmask), bool)])
    return V, F, mat
