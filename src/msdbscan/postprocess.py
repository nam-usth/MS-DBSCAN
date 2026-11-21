"""Steps shared by every method: vertex->face labels and small-region merging."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from .meshio import TriMesh


def vertex_to_face_labels(mesh: TriMesh, vlabels: np.ndarray) -> np.ndarray:
    """Published rule (`counter_sort`): the label held by two or three corners;
    if all three differ, the smallest label."""
    l = np.sort(np.asarray(vlabels)[mesh.F], axis=1)
    return np.where(l[:, 1] == l[:, 2], l[:, 2], l[:, 0])


def vertex_to_face_labels_legacy(mesh: TriMesh, vlabels: np.ndarray) -> np.ndarray:
    """The released tool's `counter_sort`, bug included. It compares the most
    common label with ITS OWN COUNT instead of comparing the two highest counts:
      - two corners share label L, third has M:  M if L == 2 else L
      - three different labels (a, b, c):        min(a, b, c) if a == 1 else a
    Used only to score the published method as published."""
    l = np.asarray(vlabels)[mesh.F]
    a, b, c = l[:, 0], l[:, 1], l[:, 2]
    s = np.sort(l, axis=1)
    all_same = s[:, 0] == s[:, 2]
    two_same = ~all_same & ((s[:, 0] == s[:, 1]) | (s[:, 1] == s[:, 2]))
    major = s[:, 1]
    minor = np.where(s[:, 0] == s[:, 1], s[:, 2], s[:, 0])
    out = np.where(a == 1, s[:, 0], a)                      # three different labels
    out = np.where(two_same, np.where(major == 2, minor, major), out)
    return np.where(all_same, a, out)


def hole_threshold_gui(n_vertices: int) -> int:
    """What the released tool applies: max(min(20, floor(0.025 V)), 4)."""
    return max(min(20, int(0.025 * n_vertices)), 4)


def hole_threshold_paper(n_vertices: int) -> int:
    """What Section 3.4 of the paper states: max(floor(0.025 V), 4)."""
    return max(int(0.025 * n_vertices), 4)


def _components(adj: sp.csr_matrix, labels: np.ndarray):
    """Connected components of the graph restricted to same-label edges."""
    coo = adj.tocoo()
    same = labels[coo.row] == labels[coo.col]
    g = sp.csr_matrix((np.ones(int(same.sum()), dtype=bool), (coo.row[same], coo.col[same])), shape=adj.shape)
    return connected_components(g, directed=False)


def fill_holes(adj: sp.csr_matrix, labels: np.ndarray, threshold: int) -> np.ndarray:
    """Published hole filling (Algorithm 3) on any element graph (vertices or faces).

    For each label in increasing order (the order is fixed once, as in the
    original), every connected piece of that label with fewer than `threshold`
    elements takes the label most frequent among the elements just outside its
    boundary. Ties go to the smallest label (the original leaves them to
    hash order). Pieces relabelled early change what later labels see, as in
    the original.
    """
    # Implementation: the pieces are tracked with a union-find over the connected
    # components of the input labelling instead of recomputing the components of the
    # whole graph for every label. Relabelling a piece can only merge it with adjacent
    # pieces of its new label, never split anything, so the union-find reproduces the
    # components the original would recompute. Identical output to the per-label
    # recomputation on every case tested; on a
    # 112 000-face graph with 790 pieces it takes well under a second instead of 24 s.
    labels = np.asarray(labels).copy()
    n_comp, comp = _components(adj, labels)
    coo = adj.tocoo()
    r, c = coo.row, coo.col
    cross = comp[r] != comp[c]
    key, w = np.unique(comp[r[cross]].astype(np.int64) * n_comp + comp[c[cross]], return_counts=True)
    ka, kb = key // n_comp, key % n_comp  # directed component edges with element-edge counts, sorted by ka
    starts = np.searchsorted(ka, np.arange(n_comp + 1))
    comp_label = np.empty(n_comp, dtype=labels.dtype)
    comp_label[comp] = labels
    parent = np.arange(n_comp)
    size = np.bincount(comp, minlength=n_comp).astype(np.int64)
    members = [[i] for i in range(n_comp)]
    by_label: dict = {}
    for i, lab in enumerate(comp_label.tolist()):
        by_label.setdefault(lab, set()).add(i)

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def neighbours(root):
        for m in members[root]:
            for t in range(starts[m], starts[m + 1]):
                yield kb[t], w[t]

    for target in np.unique(labels).tolist():
        small = [ro for ro in sorted(by_label.get(target, ())) if size[ro] < threshold]
        decisions = []
        for ro in small:
            counts: dict = {}
            for q, wq in neighbours(ro):
                rq = find(q)
                if rq != ro:
                    lab = comp_label[rq]
                    if lab != target:
                        counts[lab] = counts.get(lab, 0) + int(wq)
            if counts:  # empty: a whole connected component of the mesh, nothing to merge into
                best = max(counts.values())
                decisions.append((ro, min(k for k, v in counts.items() if v == best)))  # ties: smallest label
        for ro, lab in decisions:  # pieces of one label are never adjacent, so the order does not matter
            by_label[target].discard(ro)
            comp_label[ro] = lab
            by_label.setdefault(lab, set()).add(ro)
            root = ro
            for q, _ in list(neighbours(ro)):
                rq, rr = find(q), find(root)
                if rq != rr and comp_label[rq] == lab:
                    big, sm = (rq, rr) if len(members[rq]) >= len(members[rr]) else (rr, rq)
                    parent[sm] = big
                    size[big] += size[sm]
                    members[big].extend(members[sm])
                    members[sm] = []
                    by_label[lab].discard(sm)
                    root = big
    roots = np.array([find(i) for i in range(n_comp)])
    return comp_label[roots][comp]


def face_graph(mesh: TriMesh) -> sp.csr_matrix:
    p = mesh.face_adjacency
    m = mesh.n_faces
    A = sp.coo_matrix((np.ones(len(p), dtype=bool), (p[:, 0], p[:, 1])), shape=(m, m))
    return (A + A.T).tocsr()


def split_disconnected(adj: sp.csr_matrix, labels: np.ndarray) -> np.ndarray:
    """Give every connected piece its own label (labels become 0..k-1)."""
    _, comp = _components(adj, labels)
    return comp
