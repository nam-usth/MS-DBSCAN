"""The spaces that MS-DBSCAN v2-fuse combines, vertex by vertex (Detail tab of MS-DBSCAN Studio).

v2.cluster(mesh, eps, n_ring, element="vertex", curv_scale="edge", fuse_lambda=lam) decides whether
vertex j is a neighbour of vertex i from three quantities:
  * normal space: the unit vertex normal n_i;
  * curvature:    c_i = maximum principal curvature x mean edge length (dimensionless);
  * n-ring:       the vertices 1 .. n_ring edges away on the mesh;
and combines them as  N(i) = n-ring(i) ∩ {j : |p_i - p_j| <= eps}  with  p = (1 + lam |c|) n  (the
fused space). A vertex is core when |N(i)| + 1 >= MIN_PTS.

This module computes these quantities with v2's own helpers so that they can be shown one by one; it
does not cluster. The app's --selftest checks that its core vertices are the ones v2.cluster records.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from . import v2


@dataclass
class Spaces:
    eps_normal: float
    n_ring: int
    fuse_lambda: float
    normal: np.ndarray        # (n, 3) unit vertex normals
    curv: np.ndarray          # (n,) signed curvature x mean edge length
    fused: np.ndarray         # (n, 3) (1 + lam |curv|) * normal
    ring: sp.csr_matrix       # (n, n) bool, 1 <= hops <= n_ring
    n_neighbours: np.ndarray  # |N(i)|, the vertex itself not counted
    core: np.ndarray          # |N(i)| + 1 >= v2.MIN_PTS

    def neighbourhood(self, i: int) -> dict:
        """Index arrays for vertex i (i itself excluded): its n-ring, the balls of radius eps in the normal
        space and in the fused space (over all vertices), and N(i) = n-ring ∩ fused ball."""
        ring = np.sort(self.ring.indices[self.ring.indptr[i]:self.ring.indptr[i + 1]])
        dn = np.linalg.norm(self.normal - self.normal[i], axis=1)
        df = np.linalg.norm(self.fused - self.fused[i], axis=1)
        other = np.arange(len(dn)) != i
        return dict(ring=ring,
                    normal_ball=np.flatnonzero((dn <= self.eps_normal) & other),
                    fused_ball=np.flatnonzero((df <= self.eps_normal) & other),
                    N=ring[df[ring] <= self.eps_normal],
                    core=bool(self.core[i]))


def spaces(mesh, eps_normal: float, n_ring: int, fuse_lambda: float = 0.0) -> Spaces:
    adj, normal, curv = v2._elements(mesh, "vertex", "edge")
    fused = normal * (1.0 + fuse_lambda * np.abs(curv))[:, None]  # the line of v2.cluster
    ring = v2.ring_pairs(adj, n_ring).tocsr()
    coo = ring.tocoo()
    keep = np.linalg.norm(fused[coo.row] - fused[coo.col], axis=1) <= eps_normal
    n_nb = np.bincount(coo.row[keep], minlength=adj.shape[0])
    return Spaces(float(eps_normal), int(n_ring), float(fuse_lambda), normal, curv, fused, ring, n_nb,
                  n_nb + 1 >= v2.MIN_PTS)
