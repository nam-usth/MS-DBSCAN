"""Coarse-to-fine (pyramid) segmentation of full-resolution meshes.

Idea: cluster a strongly simplified copy
of the mesh first, then carry the labels to finer copies and recompute only
the vertices near the boundaries of the coarse clusters.

Levels
------
Every level is a simplification of the ORIGINAL mesh with fast_simplification,
and keeps the exact map  original vertex -> level vertex  that the edge
collapses define (`replay_simplification`). Labels therefore move between
levels through the original vertices, without any nearest-point search.

The coarsest level is the published preprocessing: 90 % reduction followed by
100 Laplacian iterations. On 3D Puzzles it is identical, triangle for triangle
and coordinate for coordinate, to the meshes of every earlier experiment
(checked), so any segmentation chosen there can be lifted unchanged.
`coarse_cap` limits the coarse level to that many vertices for very large
meshes (a 280 000-vertex scan would otherwise give a 28 000-vertex coarse
level, where clusters chain across rounded edges). With the default 6000 it
changes none of the 36 evaluation fragments (largest coarse level: 5591).

Refinement on one level
-----------------------
1. lift: each level vertex takes the majority label of the original vertices
   collapsed into it (ties: smallest label);
2. band: vertices within `band` coarse edge lengths (converted to hops on this
   level) of a label boundary are cleared; everything else keeps its label;
   a segment that would lose all its vertices is not cleared;
3. local proxies: for every label s and every band vertex within reach, the
   proxy normal is the mean normal of the `k_proxy` nearest kept vertices of s
   (within `reach` = band + 1 coarse edge lengths);
4. each band vertex takes the label whose proxy normal is closest to its own
   normal (planar facets: the boundary lands on the bisector of the crease);
5. connectivity: a piece of a label that touches no kept vertex of that label
   is cleared again and the cleared vertices are filled breadth-first from
   adjacent labelled vertices with the closest normal (as v2's noise
   assignment), so no new disconnected segment is created.
"""
from __future__ import annotations

from dataclasses import dataclass

import fast_simplification as fs
import numpy as np
import pyvista as pv
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .meshio import TriMesh

NONE = -1


@dataclass
class Level:
    mesh: TriMesh
    to_level: np.ndarray  # original vertex -> vertex of this level
    edge_length: float  # mean edge length before any smoothing


def make_level(V: np.ndarray, F: np.ndarray, keep: float | None = None, target_vertices: int | None = None,
               smooth_iter: int = 0) -> Level:
    """Simplify the original (V, F) to `keep` of its faces (or to about `target_vertices`)."""
    n_faces = len(F)
    if target_vertices is not None:
        keep = min(1.0, 2.0 * target_vertices / n_faces)
    if keep is None or keep >= 1.0:
        m = TriMesh(V, F)
        return Level(m, np.arange(len(V)), m.mean_edge_length)
    pts, tri, coll = fs.simplify(V, F, target_reduction=1.0 - keep, return_collapses=True)
    _, tri2, mapping = fs.replay_simplification(V.astype(np.float32), F.astype(np.int64), coll)
    if not np.array_equal(tri, tri2):
        raise RuntimeError("replayed simplification differs from the simplification")
    edge = TriMesh(pts, tri).mean_edge_length
    if smooth_iter:
        poly = pv.PolyData(pts, np.hstack([np.full((len(tri), 1), 3), tri]).ravel()).smooth(n_iter=smooth_iter)
        pts = np.asarray(poly.points)
    return Level(TriMesh(pts, tri), np.asarray(mapping, dtype=np.int64), edge)


def coarse_level(V: np.ndarray, F: np.ndarray, coarse_cap: int = 6000, coarse_floor: int = 100) -> Level:
    """The published preprocessing (90 % reduction, 100 Laplacian iterations), capped in size.

    Meshes whose coarse level would have fewer than `coarse_floor` vertices are used as
    they are, neither simplified nor smoothed (66 of the 129 BreakingBad fragments
    have fewer than 1 000 vertices, down to 4). The rule changes no 3D Puzzles fragment
    (smallest coarse level there: 119 vertices)."""
    if 0.1 * len(V) < coarse_floor:
        return make_level(V, F, keep=1.0)
    if 0.1 * len(V) > coarse_cap:
        return make_level(V, F, target_vertices=coarse_cap, smooth_iter=100)
    return make_level(V, F, keep=0.1, smooth_iter=100)


def face_to_vertex_labels(mesh: TriMesh, flabels: np.ndarray, dense_limit: int = 512) -> np.ndarray:
    """Each vertex takes the label with the largest incident face area (ties: smallest label).

    Up to `dense_limit` labels the vote is a dense argmax; above it,
    the same sums are read row by row from the sparse matrix (k-means split into
    connected pieces can give more than 512 labels). Both give the same labels."""
    fl = np.asarray(flabels)
    uniq, inv = np.unique(fl, return_inverse=True)
    k = len(uniq)
    rows = mesh.F.ravel()
    cols = np.repeat(inv, 3)
    w = np.repeat(mesh.face_area, 3)
    M = sp.csr_matrix((w, (rows, cols)), shape=(mesh.n_vertices, k))
    if k <= dense_limit:
        return uniq[M.toarray().argmax(axis=1)]
    M.sum_duplicates()
    M.sort_indices()
    return uniq[np.asarray(M.argmax(axis=1)).ravel()]


def lift(labels: np.ndarray, src: Level, dst: Level) -> np.ndarray:
    """Labels of `src` vertices -> labels of `dst` vertices, through the original vertices."""
    orig = np.asarray(labels)[src.to_level]
    n_dst = dst.mesh.n_vertices
    uniq, inv = np.unique(orig, return_inverse=True)
    key = dst.to_level * len(uniq) + inv
    k, cnt = np.unique(key, return_counts=True)
    vert, lab = k // len(uniq), k % len(uniq)
    order = np.lexsort((lab, -cnt, vert))  # per vertex: highest count, then smallest label
    vert, lab = vert[order], lab[order]
    first = np.r_[True, vert[1:] != vert[:-1]]
    out = np.zeros(n_dst, dtype=uniq.dtype)
    hit = np.zeros(n_dst, dtype=bool)  # a mask, not a sentinel: -1 can be a real label (unassigned noise)
    out[vert[first]] = uniq[lab[first]]
    hit[vert[first]] = True
    if not hit.all():
        raise RuntimeError(f"{int((~hit).sum())} level vertices received no original vertex")
    return out


def _hops_from(adj: sp.csr_matrix, start: np.ndarray, hops: int) -> np.ndarray:
    reached = start.copy()
    front = start.copy()
    for _ in range(hops):
        front = (adj @ front.astype(np.int8)) > 0
        front &= ~reached
        if not front.any():
            break
        reached |= front
    return reached


def _fill(adj: sp.csr_matrix, normals: np.ndarray, labels: np.ndarray, rounds: np.ndarray | None = None) -> np.ndarray:
    coo = adj.tocoo()
    a, c = coo.row, coo.col
    d = np.linalg.norm(normals[a] - normals[c], axis=1)
    t = 0
    while True:
        e = (labels[a] == NONE) & (labels[c] != NONE)
        if not e.any():
            break
        t += 1
        ea, ec, ed = a[e], c[e], d[e]
        order = np.lexsort((ec, ed, ea))
        ea, ec = ea[order], ec[order]
        first = np.r_[True, ea[1:] != ea[:-1]]
        labels[ea[first]] = labels[ec[first]]
        if rounds is not None:
            rounds[ea[first]] = t - 1
    return labels


def refine(level: Level, labels: np.ndarray, coarse_edge: float, band: float = 2.0, k_proxy: int = 16,
           return_info: bool = False, trace=None):
    mesh = level.mesh
    adj = mesh.vertex_adjacency
    normals = mesh.vertex_normal
    names, labels = np.unique(np.asarray(labels), return_inverse=True)  # any label values (v1 uses -10 and 0)

    def show(x):  # dense index -> original label; UNSEEN / NONE stay as they are
        return np.where(x >= 0, names[np.maximum(x, 0)], x)

    out, info = _refine_dense(mesh, adj, normals, labels, level.edge_length, coarse_edge, band, k_proxy,
                              trace, show)
    return (names[out], info) if return_info else names[out]


def _refine_dense(mesh, adj, normals, labels, edge_length, coarse_edge, band, k_proxy, trace=None, show=None):
    from .trace import UNSEEN

    hops = max(1, int(np.ceil(band * coarse_edge / edge_length)))
    coo = adj.tocoo()
    boundary = np.zeros(mesh.n_vertices, dtype=bool)
    diff = labels[coo.row] != labels[coo.col]
    boundary[coo.row[diff]] = True
    in_band = _hops_from(adj, boundary, hops - 1) if boundary.any() else boundary
    kept_per_label = np.bincount(labels[~in_band], minlength=labels.max() + 1)
    in_band &= kept_per_label[labels] > 0  # never erase a whole segment
    kept = ~in_band
    band_idx = np.flatnonzero(in_band)
    info = {"hops": hops, "band_fraction": float(in_band.mean()), "orphans_refilled": 0}
    cleared = np.where(in_band, UNSEEN, labels)
    if trace is not None:
        trace.add("Clear the boundary band", show(labels), show(cleared),
                  note=f"{100 * in_band.mean():.1f} % of the vertices lie within {hops} hops of a boundary and are recomputed")
    if len(band_idx) == 0:
        return labels, info

    reach = (band + 1.0) * coarse_edge
    best_d = np.full(len(band_idx), np.inf)
    best_l = np.full(len(band_idx), NONE)
    P = mesh.V
    for s in np.unique(labels[kept]):
        ks = np.flatnonzero(kept & (labels == s))
        tree = cKDTree(P[ks])
        k = min(k_proxy, len(ks))
        dist, nn = tree.query(P[band_idx], k=k, distance_upper_bound=reach)
        if k == 1:
            dist, nn = dist[:, None], nn[:, None]
        valid = np.isfinite(dist)
        has = valid.any(axis=1)
        if not has.any():
            continue
        nn = np.where(valid, nn, 0)
        proxy = (normals[ks][nn] * valid[..., None]).sum(axis=1)
        proxy /= np.maximum(np.linalg.norm(proxy, axis=1, keepdims=True), 1e-12)
        d = np.linalg.norm(normals[band_idx] - proxy, axis=1)
        better = has & (d < best_d)
        best_d[better], best_l[better] = d[better], s
    new = labels.copy()
    new[band_idx] = best_l  # NONE where no label was within reach
    if trace is not None:
        t = np.zeros(len(labels), dtype=np.int64)
        rank = np.argsort(np.argsort(best_d, kind="stable"), kind="stable")
        t[band_idx] = (24 * rank) // max(len(band_idx), 1)  # most confident vertices first, 24 frames
        trace.add("Refine the band (closest local proxy normal)", show(cleared), show(np.where(new == NONE, UNSEEN, new)),
                  time=t, note=f"proxy = mean normal of the {k_proxy} nearest kept vertices of each label")
        refined = new.copy()

    # connectivity: pieces that touch no kept vertex of their label are cleared and refilled
    _, comp = connected_components(
        sp.csr_matrix((np.ones(int((new[coo.row] == new[coo.col]).sum()), dtype=bool),
                       (coo.row[new[coo.row] == new[coo.col]], coo.col[new[coo.row] == new[coo.col]])),
                      shape=adj.shape), directed=False)
    anchored = np.zeros(comp.max() + 1, dtype=bool)
    anchored[comp[kept]] = True
    new[~anchored[comp]] = NONE
    info["orphans_refilled"] = int((new == NONE).sum())
    rounds = np.zeros(len(new), dtype=np.int64)
    new = _fill(adj, normals, new, rounds)
    if trace is not None and info["orphans_refilled"]:
        trace.add("Reconnect pieces cut off from their segment", show(np.where(refined == NONE, UNSEEN, refined)), show(new),
                  time=rounds, note=f"{info['orphans_refilled']} vertices refilled breadth-first from adjacent segments")
    return new, info


def pyramid(V: np.ndarray, F: np.ndarray, coarse_labels: np.ndarray, coarse: Level,
            steps: tuple = (1.0,), band: float = 2.0, k_proxy: int = 16, trace=None):
    """Carry vertex labels of the coarse level to the original mesh through `steps`
    (fractions of the original faces kept, increasing, ending with 1.0).
    Returns (vertex labels on the original mesh, per-step info)."""
    cur_level, cur = coarse, np.asarray(coarse_labels)
    infos = []
    for keep in steps:
        lvl = make_level(V, F, keep=keep)
        lab = lift(cur, cur_level, lvl)
        if trace is not None:
            trace.set_level(f"level {lvl.mesh.n_vertices} vertices", lvl.mesh.V, lvl.mesh.F)
            trace.add(f"Lift labels to {lvl.mesh.n_vertices} vertices", None, lab,
                      note="each vertex takes the label of the coarse vertex it was collapsed into")
        lab, info = refine(lvl, lab, cur_level.edge_length, band=band, k_proxy=k_proxy, return_info=True, trace=trace)
        info["keep"] = keep
        info["n_vertices"] = lvl.mesh.n_vertices
        infos.append(info)
        cur_level, cur = lvl, lab
    return cur, infos
