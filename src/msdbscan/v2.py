"""MS-DBSCAN v2 - the multi-space neighbourhood of the paper with the defects
of the released code removed. Each change is a separate switch so that its
effect can be measured on its own.

What stays (the paper's idea, Algorithm 2):
    N(i) = ring_n(i)  ∩  {j : ||n_i - n_j|| <= eps_normal}  ∩  {j : |c_i - c_j| <= eps_curv}

What changes:
 1. DBSCAN semantics as in Algorithm 1 of the paper, which the released code
    does not implement: an element is core when |N(i)| + 1 >= min_pts; clusters
    are the connected components of the core-core graph; a non-core element
    with a core neighbour joins that neighbour's cluster. The result does not
    depend on element numbering (v1's does, and leaves an order-dependent
    "label 0").
 2. `element="face"`: cluster faces on the dual graph instead of vertices, so
    a sharp edge is a boundary between elements rather than a row of elements
    that belong to neither side, and no vertex->face vote is needed.
 3. `curv_scale="edge"`: curvature multiplied by the mean edge length, which
    makes eps_curv dimensionless (v1's is in 1/length, so it changes with the
    object's size). `eps_curv=inf` switches the curvature space off.
 4. `assign_noise`: elements left unassigned take, in breadth-first rounds over
    the mesh adjacency, the label of the adjacent assigned element with the
    closest normal. Without it they stay as one noise label, as in v1.

min_pts is 5 everywhere: the released code's `len(N) > 4`.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from .meshio import TriMesh
from .postprocess import face_graph

NOISE = -1
MIN_PTS = 5


def ring_pairs(adj: sp.csr_matrix, n: int) -> sp.csr_matrix:
    """R[i,j] = True iff 1 <= hop distance(i,j) <= n on the given adjacency."""
    A = adj.astype(np.int8)
    reach, frontier = A.copy(), A.copy()
    for _ in range(n - 1):
        frontier = frontier @ A
        frontier.data[:] = 1
        reach = reach + frontier
        reach.data[:] = 1
    reach = reach.tolil()
    reach.setdiag(0)
    reach = reach.tocsr()
    reach.eliminate_zeros()
    return reach.astype(bool)


def _elements(mesh: TriMesh, element: str, curv_scale: str):
    curv_v = mesh.curvature_max * (mesh.mean_edge_length if curv_scale == "edge" else 1.0)
    if element == "vertex":
        return mesh.vertex_adjacency, mesh.vertex_normal, curv_v
    if element == "face":
        return face_graph(mesh), mesh.face_normal, curv_v[mesh.F].mean(axis=1)
    raise ValueError(element)


def cluster(mesh: TriMesh, eps_normal: float, n_ring: int = 3, eps_curv: float = np.inf, element: str = "face",
            curv_scale: str = "edge", assign_noise: bool = True, min_pts: int = MIN_PTS, radius: float | None = None,
            fuse_lambda: float | None = None, ring: sp.csr_matrix | None = None, trace=None):
    """Returns (labels per element, element kind). Labels >= 0, NOISE = -1.

    `fuse_lambda` (optional) replaces the separate curvature test ("curvature
    separate": normal ball AND curvature ball) by one ball in a fused space
    ("curvature fuse", the 2023 idea): each element is mapped to the point
    r * n on a sphere of radius r = 1 + fuse_lambda * |curvature| (curvature
    scaled by `curv_scale`), and eps_normal is applied to the distance between
    these points:
        |p_i - p_j|^2 = lam^2 (|c_i| - |c_j|)^2 + 2 r_i r_j (1 - cos theta_ij).
    The radius starts at 1 rather than at |c| because with r = |c| every flat
    element sits at the centre and all planar facets would fall into one
    cluster. fuse_lambda = 0 is exactly the normal-only neighbourhood. The same
    fused distance orders border and noise assignment. Requires eps_curv = inf.

    `radius` (optional) replaces the n-ring by a ball in space: two elements are
    candidates when their positions are within radius * sqrt(total surface area).
    Unlike a hop count this does not change with mesh resolution or object size
    (after the pipeline's decimation the fragments range from 120 to 5591
    vertices, so 3 hops is 3 % of one object and a third of another). A ball can
    join elements that are not connected on the surface, so in this mode the
    core components are split into connected pieces before borders are assigned.
    """
    adj, normals, curv = _elements(mesh, element, curv_scale)
    if fuse_lambda is not None:
        if np.isfinite(eps_curv):
            raise ValueError("fuse_lambda replaces the curvature test; use eps_curv=inf")
        normals = normals * (1.0 + fuse_lambda * np.abs(curv))[:, None]
    n = adj.shape[0]
    if radius is None:
        # `ring` (optional) = ring_pairs(adj, n_ring) computed once by the caller and reused across a grid
        ring = (ring if ring is not None else ring_pairs(adj, n_ring)).tocoo()
        if ring.shape != (n, n):
            raise ValueError("ring does not match the mesh")
        i, j = ring.row, ring.col
    else:
        from scipy.spatial import cKDTree

        pos = mesh.face_centroid if element == "face" else mesh.V
        pairs = cKDTree(pos).query_pairs(radius * np.sqrt(mesh.face_area.sum()), output_type="ndarray")
        i = np.concatenate([pairs[:, 0], pairs[:, 1]])
        j = np.concatenate([pairs[:, 1], pairs[:, 0]])
    dn = np.linalg.norm(normals[i] - normals[j], axis=1)
    keep = dn <= eps_normal
    if np.isfinite(eps_curv):
        keep &= np.abs(curv[i] - curv[j]) <= eps_curv
    i, j, dn = i[keep], j[keep], dn[keep]

    degree = np.bincount(i, minlength=n) + 1  # the element itself counts, as in the paper
    core = degree >= min_pts
    cc = core[i] & core[j]
    g = sp.csr_matrix((np.ones(int(cc.sum()), dtype=bool), (i[cc], j[cc])), shape=(n, n))
    _, comp = connected_components(g, directed=False)
    if radius is not None:
        # keep only mesh-adjacent links inside each component, then recompute components
        coo0 = adj.tocoo()
        same = core[coo0.row] & core[coo0.col] & (comp[coo0.row] == comp[coo0.col])
        g2 = sp.csr_matrix((np.ones(int(same.sum()), dtype=bool), (coo0.row[same], coo0.col[same])), shape=(n, n))
        _, comp = connected_components(g2, directed=False)
    labels = np.full(n, NOISE)
    _, labels[core] = np.unique(comp[core], return_inverse=True)
    if trace is not None:  # replay: core elements, then clusters growing from their densest core element
        from .trace import CORE, UNSEEN, bfs_depth

        core_view = np.where(core, CORE, UNSEEN)
        trace.add("Core elements", None, core_view,
                  note=f"{int(core.sum())} of {n} elements have >= {min_pts - 1} neighbours within eps")
        order = np.lexsort((np.arange(n), -degree))
        order = order[core[order]]
        _, first_of_cluster = np.unique(labels[order], return_index=True)
        depth = bfs_depth(g if radius is None else g2, order[first_of_cluster], core)
        trace.add("Cluster expansion (density-reachable)", core_view, np.where(core, labels, UNSEEN),
                  time=np.where(core, depth, 0), note=f"{int(labels.max()) + 1} clusters grow from their densest core element")
        before_border = np.where(core, labels, UNSEEN)

    # border elements: the core neighbour with the closest normal decides
    b = ~core[i] & core[j]
    if b.any():
        bi, bj, bd = i[b], j[b], dn[b]
        order = np.lexsort((bj, bd, bi))  # by element, then distance, then index (deterministic)
        bi, bj = bi[order], bj[order]
        first = np.r_[True, bi[1:] != bi[:-1]]
        labels[bi[first]] = labels[bj[first]]
    if trace is not None:
        trace.add("Border elements", before_border, labels,
                  note=f"{int(((labels >= 0) & ~core).sum())} non-core elements join the core neighbour with the closest normal; "
                       f"{int((labels == NOISE).sum())} stay unassigned")

    if assign_noise:
        coo = adj.tocoo()
        a, c = coo.row, coo.col
        da = np.linalg.norm(normals[a] - normals[c], axis=1)
        before_noise, rounds, t = labels.copy(), np.zeros(n, dtype=np.int64), 0
        while True:
            e = (labels[a] == NOISE) & (labels[c] != NOISE)
            if not e.any():
                if trace is not None and t:
                    trace.add("Unassigned elements join a neighbour", before_noise, labels, time=rounds,
                              note=f"{int((before_noise == NOISE).sum())} elements, breadth-first, closest normal first")
                break
            t += 1
            ea, ec, ed = a[e], c[e], da[e]
            order = np.lexsort((ec, ed, ea))
            ea, ec = ea[order], ec[order]
            first = np.r_[True, ea[1:] != ea[:-1]]
            labels[ea[first]] = labels[ec[first]]
            rounds[ea[first]] = t - 1
    return labels, element


def stats(mesh: TriMesh, **kw) -> dict:
    """Witness quantities for sweeps: how many elements are core / border / noise."""
    kw2 = dict(kw, assign_noise=False)
    labels, element = cluster(mesh, **kw2)
    return {"n_elements": len(labels), "noise_fraction": float((labels == NOISE).mean()),
            "n_clusters": int(labels.max() + 1)}
