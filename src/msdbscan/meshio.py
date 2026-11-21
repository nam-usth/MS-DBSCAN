"""Triangle-mesh container and the geometric quantities every method reads.

Normals and curvature are taken from PyVista/VTK on purpose: the published
MS-DBSCAN (MTAP 2025) reads `point_normals` and `curvature('maximum')`, so
using the same source keeps the re-implementation comparable with the paper.
"""
from __future__ import annotations

from functools import cached_property

import numpy as np
import pyvista as pv
import scipy.sparse as sp
import vtk

vtk.vtkObject.GlobalWarningDisplayOff()  # the OBJ reader warns on every 'vn' line


def to_polydata(V: np.ndarray, F: np.ndarray) -> pv.PolyData:
    faces = np.hstack([np.full((len(F), 1), 3, dtype=np.int64), F.astype(np.int64)])
    return pv.PolyData(np.asarray(V, dtype=float), faces.ravel())


class TriMesh:
    """Vertices V (n,3), faces F (m,3) and lazily computed derived quantities."""

    def __init__(self, V, F, name: str = ""):
        self.V = np.asarray(V, dtype=float)
        self.F = np.asarray(F, dtype=np.int64)
        self.name = name

    # -- sizes ---------------------------------------------------------------
    @property
    def n_vertices(self) -> int:
        return len(self.V)

    @property
    def n_faces(self) -> int:
        return len(self.F)

    @cached_property
    def pv(self) -> pv.PolyData:
        return to_polydata(self.V, self.F)

    # -- per-face ------------------------------------------------------------
    @cached_property
    def face_area(self) -> np.ndarray:
        a, b, c = (self.V[self.F[:, i]] for i in range(3))
        return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)

    @cached_property
    def face_normal(self) -> np.ndarray:
        a, b, c = (self.V[self.F[:, i]] for i in range(3))
        n = np.cross(b - a, c - a)
        ln = np.linalg.norm(n, axis=1, keepdims=True)
        return n / np.where(ln > 0, ln, 1.0)

    @cached_property
    def face_centroid(self) -> np.ndarray:
        return self.V[self.F].mean(axis=1)

    # -- per-vertex ----------------------------------------------------------
    @cached_property
    def vertex_normal(self) -> np.ndarray:
        """VTK point normals, as read by the published implementation."""
        return np.asarray(self.pv.point_normals, dtype=float)

    @cached_property
    def curvature_max(self) -> np.ndarray:
        """VTK maximum principal curvature, as read by the published implementation."""
        return np.asarray(self.pv.curvature(curv_type="maximum"), dtype=float)

    @cached_property
    def curvature_mean(self) -> np.ndarray:
        return np.asarray(self.pv.curvature(curv_type="mean"), dtype=float)

    # -- connectivity --------------------------------------------------------
    @cached_property
    def edges(self) -> np.ndarray:
        """Unique undirected edges (k,2), i < j."""
        e = np.vstack([self.F[:, [0, 1]], self.F[:, [1, 2]], self.F[:, [2, 0]]])
        e.sort(axis=1)
        return np.unique(e, axis=0)

    @cached_property
    def mean_edge_length(self) -> float:
        e = self.edges
        return float(np.linalg.norm(self.V[e[:, 0]] - self.V[e[:, 1]], axis=1).mean())

    @cached_property
    def vertex_adjacency(self) -> sp.csr_matrix:
        e = self.edges
        n = self.n_vertices
        data = np.ones(len(e), dtype=bool)
        A = sp.coo_matrix((data, (e[:, 0], e[:, 1])), shape=(n, n))
        return (A + A.T).tocsr()

    @cached_property
    def face_adjacency(self) -> np.ndarray:
        """Pairs of faces sharing an edge (k,2). Non-manifold edges give all pairs."""
        m = self.n_faces
        e = np.vstack([self.F[:, [0, 1]], self.F[:, [1, 2]], self.F[:, [2, 0]]])
        e.sort(axis=1)
        fid = np.tile(np.arange(m), 3)
        key = e[:, 0] * self.n_vertices + e[:, 1]
        order = np.argsort(key, kind="stable")
        key, fid = key[order], fid[order]
        same = key[1:] == key[:-1]
        pairs = np.stack([fid[:-1][same], fid[1:][same]], axis=1)
        pairs.sort(axis=1)
        return np.unique(pairs, axis=0)

    def ring_pairs(self, n: int) -> sp.csr_matrix:
        """Boolean matrix R with R[i,j] = True iff 1 <= hop distance(i,j) <= n."""
        A = self.vertex_adjacency.astype(np.int8)
        reach = A.copy()
        frontier = A.copy()
        for _ in range(n - 1):
            frontier = (frontier @ A)
            frontier.data[:] = 1
            reach = reach + frontier
            reach.data[:] = 1
        reach = reach.tolil()
        reach.setdiag(0)
        reach = reach.tocsr()
        reach.eliminate_zeros()
        return reach.astype(bool)


def load_obj(path: str) -> TriMesh:
    """Read a triangle OBJ without merging or reordering anything."""
    import trimesh

    m = trimesh.load(path, process=False, force="mesh")
    return TriMesh(np.asarray(m.vertices), np.asarray(m.faces), name=str(path))
