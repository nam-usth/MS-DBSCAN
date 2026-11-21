"""Competing surface-segmentation methods behind one interface.

Every entry of METHODS maps a name to
    fn(mesh, **params) -> (labels, element)      element in {"vertex", "face"}
    grid: list of parameter dicts                (the tuning budget, stated in every table)
    oracle: what ground-truth information the method is given, if any

All are our own implementations except AMD-DBSCAN's parameter search (Wang et
al., DSAA 2022). That module is not distributed here, so running the AMD-DBSCAN
entry raises ImportError. MS-DBSCAN Studio does not use it.
"""
from __future__ import annotations

import itertools

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from . import legacy_v1
from .meshio import TriMesh


def _grid(**axes):
    keys = list(axes)
    return [dict(zip(keys, v)) for v in itertools.product(*axes.values())]


# ------------------------------------------------------------ published method
def ms_dbscan_v1_3space(mesh, eps_normal, n_ring, eps_curv):
    return legacy_v1.cluster(mesh, 3, eps_normal, n_ring, eps_curv), "vertex_legacy"


def ms_dbscan_v1_normal(mesh, eps_normal):
    return legacy_v1.cluster(mesh, 1, eps_normal, 3, 2.0), "vertex_legacy"


# ------------------------------------------------- density clustering, normals
def dbscan_normal(mesh, eps, min_samples):
    """Textbook DBSCAN (scikit-learn) on unit vertex normals: one feature space, no connectivity."""
    from sklearn.cluster import DBSCAN

    return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(mesh.vertex_normal), "vertex"


def hdbscan_normal(mesh, min_cluster_size, min_samples):
    from sklearn.cluster import HDBSCAN

    return HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples).fit_predict(mesh.vertex_normal), "vertex"


def kmeans_normal(mesh, n_clusters):
    """k-means on face normals. n_clusters is the ground-truth facet count (oracle)."""
    from sklearn.cluster import KMeans

    return KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit_predict(mesh.face_normal), "face"


# ------------------------------------------------------------ mesh-graph methods
def _dihedral_deg(mesh: TriMesh):
    p = mesh.face_adjacency
    c = np.clip((mesh.face_normal[p[:, 0]] * mesh.face_normal[p[:, 1]]).sum(axis=1), -1.0, 1.0)
    return p, np.degrees(np.arccos(c))


def dihedral_cut(mesh, angle_deg):
    """Cut every edge whose two faces differ by more than angle_deg; segments are
    the connected components of what is left (CAD 'feature angle' segmentation)."""
    p, ang = _dihedral_deg(mesh)
    keep = ang <= angle_deg
    m = mesh.n_faces
    g = sp.coo_matrix((np.ones(int(keep.sum()), dtype=bool), (p[keep, 0], p[keep, 1])), shape=(m, m))
    return connected_components(g, directed=False)[1], "face"


def region_growing(mesh, angle_deg, curv_quantile):
    """Smoothness-constrained region growing (Rabbani et al. 2006, as in PCL) on the
    vertex graph: seeds in increasing |curvature|; a neighbour joins when its normal
    is within angle_deg of the current seed's; it becomes a seed itself when its
    |curvature| is below the curv_quantile quantile of the mesh."""
    n = mesh.n_vertices
    A = mesh.vertex_adjacency
    indptr, indices = A.indptr, A.indices
    normals = mesh.vertex_normal
    curv = np.abs(mesh.curvature_mean)
    c_th = np.quantile(curv, curv_quantile)
    cos_th = np.cos(np.radians(angle_deg))
    labels = np.full(n, -1)
    cid = 0
    for start in np.argsort(curv, kind="stable"):
        if labels[start] != -1:
            continue
        labels[start] = cid
        seeds = [start]
        while seeds:
            s = seeds.pop()
            nb = indices[indptr[s]:indptr[s + 1]]
            nb = nb[labels[nb] == -1]
            if not len(nb):
                continue
            ok = nb[normals[nb] @ normals[s] >= cos_th]  # mesh normals are oriented: no abs()
            labels[ok] = cid
            seeds.extend(ok[curv[ok] <= c_th].tolist())
        cid += 1
    return labels, "vertex"


def felzenszwalb(mesh, k):
    """Graph-based segmentation (Felzenszwalb & Huttenlocher 2004) on the face dual
    graph, edge weight = dihedral angle in degrees, internal-difference threshold k/|C|."""
    p, w = _dihedral_deg(mesh)
    order = np.argsort(w, kind="stable")
    m = mesh.n_faces
    parent = np.arange(m)
    size = np.ones(m, dtype=np.int64)
    internal = np.zeros(m)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for e in order:
        a, b = find(p[e, 0]), find(p[e, 1])
        if a == b:
            continue
        if w[e] <= min(internal[a] + k / size[a], internal[b] + k / size[b]):
            if size[a] < size[b]:
                a, b = b, a
            parent[b] = a
            size[a] += size[b]
            internal[a] = w[e]
    return np.array([find(i) for i in range(m)]), "face"


def spectral_dual(mesh, n_clusters, sigma_deg):
    """Spectral clustering of the face dual graph (in the spirit of Liu & Zhang 2004),
    affinity exp(-dihedral/sigma). n_clusters is the ground-truth facet count (oracle)."""
    from sklearn.cluster import KMeans
    from scipy.sparse.linalg import eigsh

    p, ang = _dihedral_deg(mesh)
    m = mesh.n_faces
    w = np.exp(-ang / sigma_deg)
    W = sp.coo_matrix((w, (p[:, 0], p[:, 1])), shape=(m, m))
    W = (W + W.T).tocsr()
    d = np.asarray(W.sum(axis=1)).ravel()
    d[d == 0] = 1.0
    Dm = sp.diags(1.0 / np.sqrt(d))
    S = Dm @ W @ Dm
    k = min(n_clusters, m - 2)
    rng = np.random.default_rng(0)
    vals, vecs = eigsh(S, k=k, which="LA", v0=rng.standard_normal(m), tol=1e-4, maxiter=5000)
    vecs = vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)
    return KMeans(n_clusters=n_clusters, n_init=10, random_state=0).fit_predict(vecs), "face"


# -------------------------------------------------------------------- registry
METHODS = {
    "MS-DBSCAN v1 (3 spaces)": dict(
        fn=ms_dbscan_v1_3space, oracle="",
        grid=_grid(eps_normal=[0.08, 0.12, 0.16, 0.24, 0.32], n_ring=[3, 5, 8], eps_curv=[2.0, 5.0, 10.0, 20.0])),
    "MS-DBSCAN v1 (normal only)": dict(
        fn=ms_dbscan_v1_normal, oracle="",
        grid=_grid(eps_normal=[0.02, 0.04, 0.08, 0.12, 0.16, 0.24, 0.32, 0.45])),
    "DBSCAN (normals)": dict(
        fn=dbscan_normal, oracle="",
        grid=_grid(eps=[0.02, 0.04, 0.08, 0.12, 0.16, 0.24, 0.32, 0.45], min_samples=[4, 10])),
    "HDBSCAN (normals)": dict(
        fn=hdbscan_normal, oracle="",
        grid=_grid(min_cluster_size=[5, 10, 20, 50, 100], min_samples=[1, 5, 10])),
    "Region growing": dict(
        fn=region_growing, oracle="",
        grid=_grid(angle_deg=[3, 5, 8, 12, 20, 30], curv_quantile=[0.5, 0.75, 0.9, 1.0])),
    "Dihedral-angle cut": dict(
        fn=dihedral_cut, oracle="",
        grid=_grid(angle_deg=[2, 5, 8, 12, 16, 20, 25, 30, 40, 50, 60])),
    "Felzenszwalb (dual graph)": dict(
        fn=felzenszwalb, oracle="",
        grid=_grid(k=[5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000])),
    "k-means (face normals)": dict(
        fn=kmeans_normal, oracle="number of facets", grid=[{}]),
    "Spectral (dual graph)": dict(
        fn=spectral_dual, oracle="number of facets", grid=_grid(sigma_deg=[2.0, 5.0, 10.0, 20.0])),
}


# ------------------------------------------------------------------ MS-DBSCAN v2
def _v2(element, curv_scale, assign_noise):
    def run(mesh, eps_normal, n_ring, eps_curv):
        from . import v2

        return v2.cluster(mesh, eps_normal, n_ring, eps_curv, element=element,
                          curv_scale=curv_scale, assign_noise=assign_noise)
    return run


_V1_GRID = dict(eps_normal=[0.08, 0.12, 0.16, 0.24, 0.32], n_ring=[3, 5, 8])
# Ablation ladder, one change per step from the published method. Same 60-configuration budget each.
V2_METHODS = {
    # step 1: only the DBSCAN semantics change (vertices, raw curvature, noise kept as one label)
    "MS-DBSCAN v2-a (vertex, canonical)": dict(
        fn=_v2("vertex", "raw", False), oracle="", grid=_grid(**_V1_GRID, eps_curv=[2.0, 5.0, 10.0, 20.0])),
    # step 2: + unassigned elements take the label of the nearest assigned neighbour
    "MS-DBSCAN v2-b (vertex, +noise assign)": dict(
        fn=_v2("vertex", "raw", True), oracle="", grid=_grid(**_V1_GRID, eps_curv=[2.0, 5.0, 10.0, 20.0])),
    # step 3: + dimensionless curvature (eps_curv in units of 1/mean edge length; inf = space off)
    "MS-DBSCAN v2-c (vertex, +scaled curv)": dict(
        fn=_v2("vertex", "edge", True), oracle="", grid=_grid(**_V1_GRID, eps_curv=[0.1, 0.3, 1.0, float("inf")])),
    # step 4: + faces as elements
    "MS-DBSCAN v2 (face)": dict(
        fn=_v2("face", "edge", True), oracle="", grid=_grid(**_V1_GRID, eps_curv=[0.1, 0.3, 1.0, float("inf")])),
}


def _v2_radius(mesh, eps_normal, radius, eps_curv):
    from . import v2

    return v2.cluster(mesh, eps_normal, eps_curv=eps_curv, element="face", radius=radius)


# step 5: + neighbourhood size in units of the object, not in mesh hops
V2_METHODS["MS-DBSCAN v2-s (face, scale-free ring)"] = dict(
    fn=_v2_radius, oracle="",
    grid=_grid(eps_normal=[0.08, 0.12, 0.16, 0.24, 0.32], radius=[0.03, 0.06, 0.12], eps_curv=[0.1, 0.3, 1.0, float("inf")]))


def _v2_fuse(mesh, eps_normal, n_ring, fuse_lambda):
    from . import v2

    return v2.cluster(mesh, eps_normal, n_ring, element="vertex", curv_scale="edge", assign_noise=True,
                      fuse_lambda=fuse_lambda)


# step 3': v2-c with "curvature separate" replaced by "curvature fuse" (radius 1 + lam*|c|).
# Same 60-configuration budget; lam = 0 is the same segmentation as v2-c with eps_curv = inf.
# lam values: |c| x mean edge has median 0.34, 90th percentile 1.7 over the 45 objects, so
# lam = 0.1 moves only creases by about one eps step, lam = 1 moves facet interiors too.
V2_METHODS["MS-DBSCAN v2-fuse (vertex, curvature in radius)"] = dict(
    fn=_v2_fuse, oracle="", grid=_grid(**_V1_GRID, fuse_lambda=[0.0, 0.1, 0.3, 1.0]))
METHODS.update(V2_METHODS)


# ------------------------------------------------------------------ AMD-DBSCAN
def _amd(mesh, eps_scale, minpts_scale):
    from .amd import amd_dbscan

    return amd_dbscan(mesh, eps_scale, minpts_scale)


AMD_METHODS = {
    "AMD-DBSCAN (normals)": dict(
        fn=_amd, oracle="",
        grid=[dict(eps_scale=1.0, minpts_scale=1.0), dict(eps_scale=0.5, minpts_scale=0.4),
              dict(eps_scale=0.15, minpts_scale=0.01)]),
}
METHODS.update(AMD_METHODS)


# ------------------------------------------------- Variational Shape Approximation
def vsa(mesh, n_proxies, n_iter=10, n_clusters=None):
    """Variational Shape Approximation (Cohen-Steiner, Alliez, Desbrun 2004), L2,1 metric:
    planar proxies, Lloyd iterations, each partition grown from one seed face by a
    priority queue on area * ||n_face - n_proxy||^2. Our implementation; seeds are drawn
    with a fixed random state. n_clusters (ground-truth facet count) overrides n_proxies
    in the oracle variant."""
    import heapq

    from .postprocess import face_graph

    k = int(n_clusters if n_clusters is not None else n_proxies)
    m = mesh.n_faces
    k = min(k, m)
    A = face_graph(mesh)
    indptr, indices = A.indptr, A.indices
    fn, area = mesh.face_normal, mesh.face_area
    seeds = np.random.default_rng(0).choice(m, size=k, replace=False)
    proxy = fn[seeds].copy()
    labels = np.full(m, -1)
    for _ in range(n_iter):
        labels[:] = -1
        heap = [(0.0, int(f), c) for c, f in enumerate(seeds)]
        heapq.heapify(heap)
        while heap:
            _, f, c = heapq.heappop(heap)
            if labels[f] != -1:
                continue
            labels[f] = c
            for nb in indices[indptr[f]:indptr[f + 1]]:
                if labels[nb] == -1:
                    d = fn[nb] - proxy[c]
                    heapq.heappush(heap, (float(area[nb] * (d @ d)), int(nb), c))
        new_seeds = seeds.copy()
        for c in range(k):
            sel = np.flatnonzero(labels == c)
            if not len(sel):
                continue
            nrm = (fn[sel] * area[sel, None]).sum(axis=0)
            ln = np.linalg.norm(nrm)
            if ln > 0:
                proxy[c] = nrm / ln
            new_seeds[c] = sel[np.argmin(((fn[sel] - proxy[c]) ** 2).sum(axis=1))]
        if (new_seeds == seeds).all():
            break
        seeds = new_seeds
    # faces in mesh components that hold no seed stay unassigned: one extra label, reported as is
    return labels, "face"


VSA_METHODS = {
    "VSA": dict(fn=vsa, oracle="", grid=_grid(n_proxies=[2, 3, 4, 5, 6, 8, 10, 14, 20, 32])),
    "VSA (oracle K)": dict(fn=vsa, oracle="number of facets", grid=[dict(n_proxies=0)]),
}
METHODS.update(VSA_METHODS)
