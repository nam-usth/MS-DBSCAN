"""One call from a mesh of any size to per-face segment labels (used by the Qt app).

    coarse level (published preprocessing, capped)  ->  MS-DBSCAN v2-fuse on it
    (parameters given, or chosen per mesh by the normal-space silhouette over the
    60-configuration grid x merge off/on, no ground truth)  ->  shared merge  ->
    pyramid to the original mesh.

Every step can be recorded in a `Trace` for the replay.
"""
from __future__ import annotations

import time

import numpy as np

from . import pyramid as pyr
from . import v2
from .baselines import METHODS
from .postprocess import fill_holes, hole_threshold_gui, vertex_to_face_labels
from .selfselect import criteria
from .trace import Trace

FUSE = "MS-DBSCAN v2-fuse (vertex, curvature in radius)"
GRID = sorted(METHODS[FUSE]["grid"], key=lambda p: (p["n_ring"], p["eps_normal"], p["fuse_lambda"]))
REFINE = {"coarse only": None, "lift": "lift", "band 1": dict(steps=(1.0,), band=1.0),
          "band 2": dict(steps=(1.0,), band=2.0), "two steps": dict(steps=(0.3, 1.0), band=1.0)}


def _cluster(mesh, p, ring=None, trace=None):
    labels, _ = v2.cluster(mesh, p["eps_normal"], p["n_ring"], element="vertex", curv_scale="edge", assign_noise=True,
                           fuse_lambda=p["fuse_lambda"], ring=ring, trace=trace)
    return labels


def auto_select(mesh, progress=None, trace: Trace | None = None):
    """Best (silhouette) of the 60 configurations x merge off/on on `mesh`. Returns (params, merge, table)."""
    thr = hole_threshold_gui(mesh.n_vertices)
    table, frames, captions, ring_n, ring = [], [], [], None, None
    best = None
    for idx, p in enumerate(GRID):
        if p["n_ring"] != ring_n:
            ring_n, ring = p["n_ring"], v2.ring_pairs(mesh.vertex_adjacency, p["n_ring"])
        lab = _cluster(mesh, p, ring)
        for merge in (False, True):
            v = fill_holes(mesh.vertex_adjacency, lab, thr) if merge else lab
            sil = criteria(mesh.face_normal, mesh.face_area, vertex_to_face_labels(mesh, v))["silhouette"]
            table.append(dict(p, merge=merge, silhouette=sil, segments=int(len(np.unique(v)))))
            if best is None or sil > best[0]:
                best = (sil, p, merge)
            if trace is not None:
                frames.append(v)
                captions.append(f"eps {p['eps_normal']}  ring {p['n_ring']}  lambda {p['fuse_lambda']}  merge {'on' if merge else 'off'}"
                                f"  ->  {len(np.unique(v))} segments, silhouette {sil:+.3f}"
                                f"{'   (best so far)' if best[1] is p and best[2] == merge else ''}")
        if progress:
            progress(int(60 * (idx + 1) / len(GRID)), f"parameter search {idx + 1}/{len(GRID)}")
    if trace is not None:
        trace.add_frames("Parameter search (silhouette, no ground truth)", frames, captions,
                         note=f"{len(frames)} candidates; the highest silhouette wins: {best[0]:+.3f}")
    return best[1], best[2], table


def segment(V: np.ndarray, F: np.ndarray, params: dict | None = None, merge: bool | None = None,
            refine: str = "band 1", coarse_cap: int = 6000, progress=None, trace: Trace | None = None) -> dict:
    t0 = time.time()
    timings = {}
    say = progress or (lambda *_: None)
    say(0, "building the coarse level")
    coarse = pyr.coarse_level(np.asarray(V, dtype=float), np.asarray(F, dtype=np.int64), coarse_cap=coarse_cap)
    timings["coarse level"] = time.time() - t0
    if trace is not None:
        trace.set_level("coarse", coarse.mesh.V, coarse.mesh.F)
    table = None
    t = time.time()
    if params is None:
        params, merge, table = auto_select(coarse.mesh, progress, trace)
    timings["parameter search"] = time.time() - t
    say(62, "clustering the coarse level")
    t = time.time()
    lab = _cluster(coarse.mesh, params, trace=trace)
    v = lab
    if merge:
        v = fill_holes(coarse.mesh.vertex_adjacency, lab, hole_threshold_gui(coarse.mesh.n_vertices))
        if trace is not None:
            trace.add("Merge small regions", lab, v, note="pieces below the released hole-filling threshold join a neighbour")
    timings["coarse clustering"] = time.time() - t
    t = time.time()
    mode = REFINE[refine]
    say(75, "carrying the labels to the full mesh")
    if mode is None:
        out_mesh, vertex_labels = coarse.mesh, v
    elif mode == "lift":
        full = pyr.make_level(V, F, keep=1.0)
        vertex_labels = pyr.lift(v, coarse, full)
        out_mesh = full.mesh
        if trace is not None:
            trace.set_level("full", out_mesh.V, out_mesh.F)
            trace.add(f"Lift labels to {out_mesh.n_vertices} vertices", None, vertex_labels)
    else:
        vertex_labels, _ = pyr.pyramid(V, F, v, coarse, trace=trace, **mode)
        out_mesh = pyr.make_level(V, F, keep=1.0).mesh
    timings["pyramid"] = time.time() - t
    face_labels = vertex_to_face_labels(out_mesh, vertex_labels)
    say(100, "done")
    return dict(vertex_labels=vertex_labels, face_labels=face_labels, mesh=out_mesh, coarse_vertices=coarse.mesh.n_vertices,
                params=params, merge=bool(merge), table=table, timings=timings, seconds=time.time() - t0,
                n_segments=int(len(np.unique(face_labels))))
