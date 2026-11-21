"""Criteria that rank segmentations of one mesh WITHOUT ground truth.

Both read only the face normals and face areas of the mesh and the candidate
labels, so they can be applied to the output of any method.

dispersion   D = sum_f a_f ||n_f - m_seg(f)||^2 / sum_f a_f, with m_s the unit
             area-weighted mean normal of segment s. 0 when every segment is flat.
             D always falls as segments are split, so it is used only inside
             the description-length rule  J = D + lam * K  (K = number of segments).
silhouette   area-weighted mean over faces of (b - a) / max(a, b), a = distance of
             the face normal to its own segment mean, b = to the nearest other
             segment mean. Needs no parameter. Undefined for a single segment: -1.
"""
from __future__ import annotations

import numpy as np


def criteria(face_normal: np.ndarray, face_area: np.ndarray, labels: np.ndarray) -> dict:
    _, inv = np.unique(labels, return_inverse=True)
    k = int(inv.max()) + 1
    w = face_area / face_area.sum()
    mean = np.zeros((k, 3))
    np.add.at(mean, inv, face_normal * w[:, None])
    ln = np.linalg.norm(mean, axis=1, keepdims=True)
    mean = mean / np.where(ln > 0, ln, 1.0)
    d_own = np.linalg.norm(face_normal - mean[inv], axis=1)
    out = {"k": k, "dispersion": float((w * d_own ** 2).sum())}
    if k < 2:
        out["silhouette"] = -1.0
        return out
    # distance of every face to every segment mean, in blocks to bound memory
    b = np.full(len(labels), np.inf)
    for s0 in range(0, k, 64):
        d = np.linalg.norm(face_normal[:, None, :] - mean[None, s0:s0 + 64, :], axis=2)
        own = (inv >= s0) & (inv < s0 + 64)
        d[np.flatnonzero(own), inv[own] - s0] = np.inf
        b = np.minimum(b, d.min(axis=1))
    den = np.maximum(d_own, b)
    s = np.where(den > 0, (b - d_own) / np.where(den > 0, den, 1.0), 0.0)
    out["silhouette"] = float((w * s).sum())
    return out
