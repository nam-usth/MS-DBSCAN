"""MS-DBSCAN exactly as published (MTAP 2025) - a behaviour-preserving port.

Source of truth: legacy/original_app/utils/multi_spaces_dbscan.py,
function multi_spaces_dbscan_cluster. Neighbour sets are computed with sparse
matrices instead of Python lists, but every decision of the original loop is
kept, including the ones that differ from the paper's Algorithm 1:

* a vertex is "core" when it has MORE THAN 4 neighbours (`> minPts`), and in
  modes 2-3 the vertex itself is not in its own neighbourhood;
* a cluster expands through every reached vertex, core or not;
* vertices are visited in the pop order of a Python set. A non-core vertex
  popped before any cluster reaches it keeps label 0 forever and blocks
  expansion through it. Label 0 is therefore a real output label: an
  order-dependent set of vertices that need not be connected;
* vertices with an empty neighbourhood get label -10 (outliers).

Modes (the GUI's "feature type"):
    1  normal space only (no n-ring, no curvature)
    2  normals rescaled by tanh(curvature), intersected with the n-ring; eps_curv unused
    3  normal space, n-ring and curvature space intersected
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

from .meshio import TriMesh

OUTLIER = -10
MIN_PTS = 4


def neighbour_graph(mesh: TriMesh, mode: int, eps_normal: float, n_ring: int, eps_curv: float) -> sp.csr_matrix:
    """Boolean CSR matrix whose row i is the published GetNeighbors(v_i, E)."""
    n = mesh.n_vertices
    normals = mesh.vertex_normal
    curv = mesh.curvature_max
    if mode == 1:
        tree = cKDTree(normals)
        lists = tree.query_ball_point(normals, r=eps_normal)
        counts = np.fromiter((len(x) for x in lists), dtype=np.int64, count=n)
        rows = np.repeat(np.arange(n), counts)
        cols = np.concatenate([np.asarray(x, dtype=np.int64) for x in lists])
        return sp.csr_matrix((np.ones(len(rows), dtype=bool), (rows, cols)), shape=(n, n))
    if mode == 2:
        modifier = np.tanh(curv * 0.05) - 0.5
        normals = normals * (1.0 + 0.1 * modifier)[:, None]
    ring = mesh.ring_pairs(n_ring).tocoo()
    i, j = ring.row, ring.col
    keep = np.linalg.norm(normals[i] - normals[j], axis=1) <= eps_normal
    if mode == 3:
        keep &= np.abs(curv[i] - curv[j]) <= eps_curv
    return sp.csr_matrix((np.ones(int(keep.sum()), dtype=bool), (i[keep], j[keep])), shape=(n, n))


def cluster(mesh: TriMesh, mode: int = 3, eps_normal: float = 0.16, n_ring: int = 3, eps_curv: float = 2.0):
    """Per-vertex labels: >= 1 cluster id, 0 = never reached, -10 = outlier."""
    return cluster_graph(neighbour_graph(mesh, mode, eps_normal, n_ring, eps_curv))


def neighbour_graph_normal_curv(mesh: TriMesh, eps_normal: float, eps_curv: float) -> sp.csr_matrix:
    """NOT in the released tool: normal space and curvature space without the
    n-ring (the vertex itself included, as in mode 1). Exists only so that the
    space ablation has all four combinations."""
    G = neighbour_graph(mesh, 1, eps_normal, 0, 0.0).tocoo()
    curv = mesh.curvature_max
    keep = np.abs(curv[G.row] - curv[G.col]) <= eps_curv
    n = mesh.n_vertices
    return sp.csr_matrix((np.ones(int(keep.sum()), dtype=bool), (G.row[keep], G.col[keep])), shape=(n, n))


def cluster_graph(G: sp.csr_matrix):
    """The released clustering loop on a given neighbour graph."""
    n = G.shape[0]
    indptr, indices = G.indptr, G.indices
    degree = np.diff(indptr)
    core = degree > MIN_PTS
    labels = np.zeros(n, dtype=int)
    labels[degree == 0] = OUTLIER

    unvisited = set(range(n))  # same construction as the original, hence same pop order
    for v in np.flatnonzero(degree == 0):
        unvisited.remove(int(v))
    in_unvisited = np.ones(n, dtype=bool)
    in_unvisited[degree == 0] = False

    cid = 1
    while unvisited:
        p = unvisited.pop()
        in_unvisited[p] = False
        if not core[p]:
            continue
        labels[p] = cid
        frontier = np.array([p])
        while len(frontier):
            nb = np.unique(np.concatenate([indices[indptr[v]:indptr[v + 1]] for v in frontier]))
            nb = nb[in_unvisited[nb]]
            if not len(nb):
                break
            labels[nb] = cid
            in_unvisited[nb] = False
            for v in nb.tolist():
                unvisited.remove(v)  # not difference_update: that may resize the table and change pop order
            frontier = nb
        cid += 1
    return labels
