"""Recording of the clustering process, for the replay in the Qt app.

A trace is a list of stages on named levels (meshes). A stage holds the labels
of every element at its start and at its end, and for every element the frame
at which it switches from the first to the second. Frame f of a stage shows

    labels = where(time <= f, end, start)

so a stage of n frames costs three arrays, whatever n is. A stage can instead
hold a list of whole label arrays (`frames`), used for the parameter search.

Special label values, drawn in fixed colours by the app:
    UNSEEN = -2   not reached yet
    NOISE  = -1   unassigned
    CORE   = -3   core element whose cluster is not drawn yet
Recording never changes a result: every hook only copies arrays.
"""
from __future__ import annotations

import numpy as np

UNSEEN, NOISE, CORE = -2, -1, -3


class Trace:
    def __init__(self):
        self.levels: dict[str, tuple[np.ndarray, np.ndarray, str]] = {}  # key -> (V, F, element kind)
        self.stages: list[dict] = []
        self.level = None

    def set_level(self, key: str, V: np.ndarray, F: np.ndarray, element: str = "vertex"):
        self.levels[key] = (np.asarray(V), np.asarray(F), element)
        self.level = key

    def add(self, name: str, start, end, time=None, note: str = ""):
        end = np.asarray(end).copy()
        start = np.asarray(start).copy() if start is not None else np.full(len(end), UNSEEN)
        time = np.zeros(len(end), dtype=np.int32) if time is None else np.asarray(time, dtype=np.int32).copy()
        time[start == end] = 0
        self.stages.append(dict(name=name, level=self.level, start=start, end=end, time=time,
                                n_frames=int(max(time.max(), 0)) + 1, note=note))

    def add_frames(self, name: str, frames: list, captions: list, note: str = ""):
        self.stages.append(dict(name=name, level=self.level, frames=[np.asarray(f).copy() for f in frames],
                                captions=list(captions), n_frames=len(frames), note=note))

    def last_labels(self):
        """Labels at the end of the last stage on the current level (UNSEEN if none)."""
        for s in reversed(self.stages):
            if s["level"] == self.level:
                return s["end"] if "end" in s else s["frames"][-1]
        return None

    def frame(self, stage: dict, f: int) -> np.ndarray:
        if "frames" in stage:
            return stage["frames"][min(f, len(stage["frames"]) - 1)]
        return np.where(stage["time"] <= f, stage["end"], stage["start"])

    @property
    def n_frames(self) -> int:
        return sum(s["n_frames"] for s in self.stages)


def bfs_depth(adj, sources: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    """Hop distance from the nearest source over `adj` restricted to `allowed` elements (-1 = not reached)."""
    n = adj.shape[0]
    depth = np.full(n, -1, dtype=np.int64)
    front = np.zeros(n, dtype=bool)
    front[sources] = True
    depth[front] = 0
    d = 0
    while front.any():
        d += 1
        nxt = (adj @ front.astype(np.int8)) > 0
        nxt &= allowed & (depth < 0)
        depth[nxt] = d
        front = nxt
    return depth
