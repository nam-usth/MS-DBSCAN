"""MS-DBSCAN Studio - segment a mesh of any size and replay how the clusters form.

    python app/msdbscan_studio.py [mesh.obj|.ply]
    python app/msdbscan_studio.py --selftest mesh.obj out.png   (headless check, saves a screenshot)

Pipeline (src/msdbscan/pipeline.py): coarse level -> MS-DBSCAN v2-fuse (parameters
chosen per mesh by the normal-space silhouette, or given) -> pyramid to the full mesh.
The computation runs in a worker thread; the replay only reads the recorded trace.
The layout follows the original MS-DBSCAN application (legacy/original_app/app/
GUI_remastered_Qt.py): a fixed-width control panel whose top-level regions have a red
border and whose parameter groups have a blue border, and Basic / Help tabs.

Release builds: app/build_release.py (PyInstaller, Windows and Linux).
"""
from __future__ import annotations

import colorsys
import json
import multiprocessing
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

__version__ = "1.0.0"
FROZEN = getattr(sys, "frozen", False)
SELFTEST = len(sys.argv) >= 4 and sys.argv[1] == "--selftest"


def selftest_failed(text: str):
    """End a self-test with the traceback in <out>_selftest.txt and exit code 1. A windowed build would
    otherwise stop at a modal 'Unhandled exception' dialog and the release script would wait for it."""
    msg = f"SELFTEST FAILED ({__version__}, {'frozen' if FROZEN else 'source'})\n{text}"
    out = Path(sys.argv[3])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_name(out.stem + "_selftest.txt").write_text(msg + "\n", encoding="utf-8")
    if sys.stdout is not None:
        print(msg, flush=True)
    sys.exit(1)


if FROZEN:  # PyInstaller bundle: data files sit next to the bundled modules
    RES = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    ROOT = Path.cwd()
else:
    ROOT = Path(__file__).resolve().parents[1]
    RES = Path(__file__).resolve().parent
    sys.path.insert(0, str(ROOT / "src"))

try:  # a release missing a library fails here (e.g. a DLL of VTK)
    import pyvista as pv  # noqa: E402
    from matplotlib import colormaps  # noqa: E402
    from PyQt5 import QtCore, QtGui, QtWidgets  # noqa: E402
    from pyvistaqt import QtInteractor  # noqa: E402

    from vtkmodules.vtkRenderingCore import vtkCellPicker, vtkMapper, vtkPointPicker  # noqa: E402

    from msdbscan import pipeline  # noqa: E402
    from msdbscan.meshio import TriMesh, load_obj  # noqa: E402
    from msdbscan.spaces import spaces as vertex_spaces  # noqa: E402
    from msdbscan.trace import CORE, NOISE, UNSEEN, Trace  # noqa: E402
    from msdbscan.v2 import MIN_PTS  # noqa: E402
except Exception:
    if SELFTEST:
        selftest_failed(traceback.format_exc())
    raise

# Segment boundaries lie on the faces they separate; without a depth offset the faces hide about half of each
# 5-pixel line. -200 depth units showed the whole width on a test sphere without showing the back side.
vtkMapper.SetResolveCoincidentTopologyToPolygonOffset()
BOUNDARY_OFFSET = -200
SPECIAL = {UNSEEN: (0.86, 0.86, 0.86), NOISE: (0.30, 0.30, 0.30), CORE: (1.00, 0.78, 0.10)}
# Colour settings. Segments: "Distinct hues" gives each label a fixed hue, far from the hues of the
# labels next to it; "Jet" spreads the labels of a stage from blue to red, as the original application
# did. Maps (Detail tab): colour map of the curvature panel, then of the n-ring and N(i) count panels.
SEGMENT_SCHEMES = ("Distinct hues", "Jet (original app)")
MAP_SCHEMES = {"Coolwarm, viridis": ("coolwarm", "viridis"), "Turbo": ("turbo", "turbo"), "Jet (original app)": ("jet", "jet")}
CORE_ON_JET = (1.00, 0.00, 1.00)  # jet has a yellow of its own, so core elements turn magenta
MESH_EXT = (".obj", ".ply", ".stl", ".vtk")
TITLE = "3D Mesh Segmentation with Multi-space DBSCAN"  # the original application's title
DEFAULTS = dict(eps_normal=0.16, n_ring=3, fuse_lambda=0.3)
# Detail tab, after a vertex is picked: the same colour means the same set in every panel
PICK, IN_N, RING_ONLY, BALL_ONLY, REST = (220, 30, 30), (40, 170, 60), (245, 150, 30), (100, 160, 235), (205, 205, 205)
DETAIL_LEGEND = ("<span style='color:#dc1e1e'>■</span> vertex &nbsp; "
                 "<span style='color:#28aa3c'>■</span> N(i) &nbsp; "
                 "<span style='color:#f5961e'>■</span> n-ring only &nbsp; "
                 "<span style='color:#64a0eb'>■</span> ball only")
# Style of the original application: the left panel cascades its title placement to every
# group; top-level regions get a red border, parameter groups a blue one.
PANEL_STYLE = ("QGroupBox { border: none; padding-top: 16px; margin-top: 11px; } "
               "QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top center; padding: 0 3px; }")
RED = "QGroupBox { border: 1px solid red; }"
BLUE = "QGroupBox { border: 1px solid blue; }"


class Region(QtWidgets.QGroupBox):
    """A bordered region of the control panel that can be folded to its title.

    The arrow button sits in the gap of the top border, before the title (the leading spaces of
    the title make room for it); the arrow or a click on the title row folds or unfolds the
    content. Widgets go into `content`, a layout of the requested type."""

    def __init__(self, title: str, style: str, layout=QtWidgets.QVBoxLayout, expanded: bool = True):
        super().__init__(f"      {title} ")
        self.setStyleSheet(style)
        outer = QtWidgets.QVBoxLayout(self)
        self.margins = outer.contentsMargins()
        self.body = QtWidgets.QWidget()
        self.content = layout(self.body)
        self.content.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.body)
        self.arrow = QtWidgets.QToolButton(self)
        self.arrow.setAutoRaise(True)
        self.arrow.setFixedSize(16, 16)
        self.arrow.setCursor(QtCore.Qt.PointingHandCursor)
        self.arrow.clicked.connect(lambda: self.set_expanded(not self.expanded))
        self.set_expanded(expanded)

    def set_expanded(self, on: bool):
        self.expanded = bool(on)
        self.body.setVisible(self.expanded)
        self.arrow.setArrowType(QtCore.Qt.DownArrow if self.expanded else QtCore.Qt.RightArrow)
        self.arrow.setToolTip("Collapse" if self.expanded else "Expand")
        m = self.margins
        self.layout().setContentsMargins(m.left(), m.top() if self.expanded else 0, m.right(),
                                         m.bottom() if self.expanded else 0)
        # folded: no taller than the title, even where the panel gives the region a stretch factor
        self.setSizePolicy(QtWidgets.QSizePolicy.Preferred,
                           QtWidgets.QSizePolicy.Preferred if self.expanded else QtWidgets.QSizePolicy.Maximum)
        self.updateGeometry()

    def _folded(self, size: QtCore.QSize) -> QtCore.QSize:
        # QGroupBox keeps one text line of room for content even when it has none; folded, only the title row
        if not self.expanded:
            size.setHeight(self._title_rect().bottom() + 9)
        return size

    def sizeHint(self):
        return self._folded(super().sizeHint())

    def minimumSizeHint(self):
        return self._folded(super().minimumSizeHint())

    def _title_rect(self) -> QtCore.QRect:
        opt = QtWidgets.QStyleOptionGroupBox()
        self.initStyleOption(opt)
        return self.style().subControlRect(QtWidgets.QStyle.CC_GroupBox, opt, QtWidgets.QStyle.SC_GroupBoxLabel, self)

    def _place_arrow(self):
        r = self._title_rect()
        self.arrow.move(r.left() + 5, r.center().y() - self.arrow.height() // 2 + 1)

    def resizeEvent(self, e):
        super().resizeEvent(e)
        self._place_arrow()

    def showEvent(self, e):
        super().showEvent(e)
        self._place_arrow()

    def mousePressEvent(self, e):
        if e.button() == QtCore.Qt.LeftButton and e.pos().y() <= self._title_rect().bottom():
            self.set_expanded(not self.expanded)
            return
        super().mousePressEvent(e)


def palette(labels: np.ndarray, scheme: str = SEGMENT_SCHEMES[0]) -> np.ndarray:
    """Colour per label value, fixed colours for the special values.

    Distinct hues: golden-ratio hues, the same for a label value everywhere. Jet: label / largest label
    shown, on the jet map, as in the original application (pyvista scales a colour map to the data
    shown); a segment can therefore change colour when the largest label disappears, e.g. in a merge."""
    out = np.empty((len(labels), 3))
    uniq, inv = np.unique(labels, return_inverse=True)
    jet = scheme == SEGMENT_SCHEMES[1]
    top = max(int(uniq.max()), 1)
    cols = []
    for u in uniq:
        if jet and u == CORE:
            cols.append(CORE_ON_JET)
        elif u in SPECIAL:
            cols.append(SPECIAL[u])
        elif u < 0:  # e.g. the published method's outlier label -10
            cols.append(SPECIAL[NOISE])
        elif jet:
            cols.append(colormaps["jet"](min(int(u) / top, 1.0))[:3])
        else:
            h = (0.11 + 0.618034 * int(u)) % 1.0
            cols.append(colorsys.hsv_to_rgb(h, 0.55 + 0.35 * ((int(u) * 7) % 3) / 2, 0.92))
    out[:] = np.asarray(cols)[inv]
    return (out * 255).astype(np.uint8)


def segment_count(vlabels: np.ndarray) -> tuple[int, int]:
    """(segments, unassigned vertices) of a result. The pipeline's n_segments counts the distinct face labels,
    the unassigned label (NOISE, -1) included, so it is not shown."""
    v = np.asarray(vlabels)
    return int(len(np.unique(v[v >= 0]))), int((v < 0).sum())


def face_labels(F: np.ndarray, vlabels: np.ndarray) -> np.ndarray:
    """Label held by two or three corners, else the smallest (the vote used everywhere in the project)."""
    l = np.sort(vlabels[F], axis=1)
    return np.where(l[:, 1] == l[:, 2], l[:, 2], l[:, 0])


def edge_face_pairs(F: np.ndarray):
    """Faces that share a mesh edge: (edge vertex ids (m, 2), face a (m,), face b (m,)). An edge with k
    faces gives k - 1 pairs, each against its first face; an edge of the mesh border gives none."""
    e = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    f = np.tile(np.arange(len(F)), 3)
    order = np.lexsort((e[:, 1], e[:, 0]))
    e, f = e[order], f[order]
    new = np.r_[True, (e[1:] != e[:-1]).any(axis=1)]
    first = np.maximum.accumulate(np.where(new, np.arange(len(e)), 0))
    return e[~new], f[first[~new]], f[~new]


def boundary_edges(pairs, flab: np.ndarray) -> np.ndarray:
    """Segment boundaries, the original application's "Show edge" (highlight_edge): every mesh edge whose
    faces do not all carry the same label. Vertex ids (m, 2), one row per edge."""
    ev, fa, fb = pairs
    e = ev[flab[fa] != flab[fb]]
    return np.unique(e, axis=0) if len(e) else e


def load_mesh(path: str):
    if path.lower().endswith(".obj"):
        m = load_obj(path)
        return m.V, m.F
    poly = pv.read(path).triangulate()
    return np.asarray(poly.points, dtype=float), poly.faces.reshape(-1, 4)[:, 1:].astype(np.int64)


class ListBox(QtWidgets.QListWidget):
    """A list whose preferred height is its minimum height (Qt's default is 192 px), so the panel fits
    a short screen; spare height still goes to the lists through the panel's stretch factors."""

    def __init__(self, min_height: int):
        super().__init__()
        self.setMinimumHeight(min_height)

    def sizeHint(self):
        return QtCore.QSize(super().sizeHint().width(), self.minimumHeight())


class Worker(QtCore.QThread):
    progress = QtCore.pyqtSignal(int, str)
    done = QtCore.pyqtSignal(object, object, object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, V, F, kwargs):
        super().__init__()
        self.V, self.F, self.kwargs = V, F, kwargs

    def run(self):
        try:
            tr = Trace()
            res = pipeline.segment(self.V, self.F, progress=lambda p, s: self.progress.emit(p, s), trace=tr, **self.kwargs)
            # faces of every mesh edge, per level, for Show edge: built here so that the window does not pause
            # when the replay first reaches a level (0.4 s on a 277 638-vertex scan)
            self.progress.emit(100, "preparing the segment boundaries")
            t0 = time.time()
            pairs = {k: edge_face_pairs(np.asarray(F)) for k, (_, F, _) in tr.levels.items()}
            res["timings"]["boundary edges"] = time.time() - t0
            self.done.emit(res, tr, pairs)
        except Exception:  # shown to the user, never swallowed
            self.failed.emit(traceback.format_exc())


class Studio(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{TITLE} — MS-DBSCAN Studio {__version__}")
        icon = RES / "icon.png"
        if icon.exists():
            self.setWindowIcon(QtGui.QIcon(str(icon)))
        scr = QtWidgets.QApplication.primaryScreen().availableGeometry()
        self.resize(min(1500, int(0.9 * scr.width())), min(1050, int(0.9 * scr.height())))
        self.folder = str(ROOT / "data" / "raw") if (ROOT / "data" / "raw").is_dir() else str(Path.cwd())
        self.V = self.F = None
        self.path = None
        self.trace: Trace | None = None
        self.result = None
        self.frame_index = []  # global frame -> (stage, local frame)
        self.level_shown = None
        self.poly = {}
        self.edge_pairs = {}  # level -> faces sharing each mesh edge (segment boundaries)
        self.boundary_shown = None
        self.worker = None
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        self._build()

    # ------------------------------------------------------------------ layout
    def _build(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        self.tabs = QtWidgets.QTabWidget()
        QtWidgets.QGridLayout(central).addWidget(self.tabs, 0, 0)
        basic = QtWidgets.QWidget()
        self.tabs.addTab(basic, "Basic")
        grid = QtWidgets.QHBoxLayout(basic)

        # ---- control panel: fixed width, regions with red borders, parameter groups with blue ones
        panel = QtWidgets.QGroupBox()
        panel.setStyleSheet(PANEL_STYLE)
        panel.setFixedWidth(450)
        side = QtWidgets.QVBoxLayout(panel)
        scroll = QtWidgets.QScrollArea()  # on a short screen the panel scrolls instead of squashing its rows
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        scroll.setStyleSheet("QScrollArea, QScrollArea > QWidget > QWidget { background: transparent; }")
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setFixedWidth(450 + scroll.verticalScrollBar().sizeHint().width() + 4)
        grid.addWidget(scroll, 0)

        files = Region("3D files list", RED)
        fl = files.content
        self.file_list = ListBox(130)
        self.file_list.itemDoubleClicked.connect(lambda _: self.preview())
        fl.addWidget(self.file_list)
        row = QtWidgets.QHBoxLayout()
        b = QtWidgets.QPushButton("Choose a Folder")
        b.clicked.connect(self.select_folder)
        row.addWidget(b)
        b = QtWidgets.QPushButton("Preview")
        b.setToolTip("Load the selected mesh (or double-click it)")
        b.clicked.connect(self.preview)
        row.addWidget(b)
        fl.addLayout(row)
        self.info = QtWidgets.QLabel("No mesh loaded")
        self.info.setWordWrap(True)
        fl.addWidget(self.info)
        side.addWidget(files, 2)
        self.update_listbox()

        seg = Region("Surface segmentation", RED)
        sl = seg.content
        settings = Region("Settings", BLUE, QtWidgets.QGridLayout)
        sg = settings.content
        self.auto = QtWidgets.QComboBox()
        self.auto.addItems(["Auto (silhouette)", "Manual (thresholds)"])
        self.auto.setToolTip("Per mesh: the configuration with the highest normal-space silhouette among "
                             "60 configurations x merge off/on, chosen without ground truth")
        self.auto.currentIndexChanged.connect(lambda i: self._toggle_manual(i == 0))
        self.merge = QtWidgets.QComboBox()
        self.merge.addItems(["Disable", "Enable"])
        self.merge.setCurrentIndex(1)
        self.refine = QtWidgets.QComboBox()
        self.refine.addItems(list(pipeline.REFINE))
        self.refine.setCurrentText("band 1")
        self.cap = QtWidgets.QSpinBox(minimum=500, maximum=50000, singleStep=500, value=6000)
        self.edges = QtWidgets.QComboBox()
        self.edges.addItems(["Disable", "Enable"])
        self.edges.setToolTip("Draw the boundaries between segments as pink lines (as the original application)")
        self.edges.currentIndexChanged.connect(self._recolour)
        self.seg_colours = QtWidgets.QComboBox()
        self.seg_colours.addItems(SEGMENT_SCHEMES)
        self.seg_colours.setToolTip("Distinct hues: segments with neighbouring ids get far-apart hues.\n"
                                    "Jet: the original application's colours, segment ids from blue to red "
                                    "(core elements in magenta during the replay).")
        self.map_colours = QtWidgets.QComboBox()
        self.map_colours.addItems(list(MAP_SCHEMES))
        self.map_colours.setToolTip("Colour map of the Detail tab: curvature panel, then the n-ring and N(i) panels")
        for c in (self.seg_colours, self.map_colours):
            c.currentIndexChanged.connect(self._recolour)
        for r, (text, w) in enumerate((("Parameters", self.auto), ("Hole filling", self.merge),
                                       ("To full resolution", self.refine), ("Coarse level cap", self.cap),
                                       ("Show edge", self.edges), ("Segment colours", self.seg_colours),
                                       ("Map colours", self.map_colours))):
            sg.addWidget(QtWidgets.QLabel(text), r, 0)
            sg.addWidget(w, r, 1)
        sl.addWidget(settings)

        self.thresholds = Region("Thresholds", BLUE, QtWidgets.QGridLayout)  # folded while the parameters are automatic
        tg = self.thresholds.content
        self.eps = QtWidgets.QDoubleSpinBox(decimals=3, minimum=0.01, maximum=1.5, singleStep=0.02, value=DEFAULTS["eps_normal"])
        self.ring = QtWidgets.QSpinBox(minimum=1, maximum=20, value=DEFAULTS["n_ring"])
        self.lam = QtWidgets.QDoubleSpinBox(decimals=2, minimum=0.0, maximum=8.0, singleStep=0.1, value=DEFAULTS["fuse_lambda"])
        for r, (text, w) in enumerate((("Eps normal", self.eps), ("Eps n-ring", self.ring), ("Curvature fuse λ", self.lam))):
            tg.addWidget(QtWidgets.QLabel(text), r, 0)
            tg.addWidget(w, r, 1)
        self.default_btn = QtWidgets.QPushButton("Load default")
        self.default_btn.clicked.connect(self.load_default)
        tg.addWidget(self.default_btn, 3, 0, 1, 2)
        sl.addWidget(self.thresholds)
        self._toggle_manual(True)

        self.run_btn = QtWidgets.QPushButton("Compute")
        self.run_btn.setToolTip("Run segmentation")
        self.run_btn.clicked.connect(self.run)
        self.run_btn.setEnabled(False)
        sl.addWidget(self.run_btn)
        self.bar = QtWidgets.QProgressBar()
        sl.addWidget(self.bar)
        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)
        sl.addWidget(self.status)
        side.addWidget(seg)

        self.stages_region = Region("Replay stages", RED, expanded=False)  # unfolds when a result arrives
        stl = self.stages_region.content
        self.stage_list = ListBox(150)
        self.stage_list.setWordWrap(True)
        self.stage_list.setToolTip("Click a stage to jump to it")
        self.stage_list.itemClicked.connect(self._jump_to_stage)
        stl.addWidget(self.stage_list)
        side.addWidget(self.stages_region, 3)

        self.export_region = Region("Export", RED, QtWidgets.QGridLayout, expanded=False)
        eg = self.export_region.content
        self.save_btn = QtWidgets.QPushButton("Save labels…")
        self.save_btn.clicked.connect(self.save_labels)
        self.shot_btn = QtWidgets.QPushButton("Screenshot…")
        self.shot_btn.clicked.connect(self.screenshot)
        self.gif_btn = QtWidgets.QPushButton("Replay to GIF…")
        self.gif_btn.clicked.connect(self.export_gif)
        for w in (self.save_btn, self.shot_btn, self.gif_btn):
            w.setEnabled(False)
        eg.addWidget(self.save_btn, 0, 0)
        eg.addWidget(self.shot_btn, 0, 1)
        eg.addWidget(self.gif_btn, 1, 0, 1, 2)
        side.addWidget(self.export_region)
        side.addStretch(0)  # takes the height that folded regions give up

        # ---- display
        display = QtWidgets.QFrame()
        right = QtWidgets.QVBoxLayout(display)
        grid.addWidget(display, 1)
        self.view = QtInteractor(display)
        self.view.set_background("white")
        right.addWidget(self.view.interactor, 1)
        self.caption = QtWidgets.QLabel("")
        self.caption.setWordWrap(True)
        right.addWidget(self.caption)
        ctl = QtWidgets.QHBoxLayout()
        self.play_btn = QtWidgets.QPushButton("Play")
        self.play_btn.clicked.connect(self.toggle_play)
        prev_b = QtWidgets.QPushButton("◀")
        prev_b.clicked.connect(lambda: self.slider.setValue(self.slider.value() - 1))
        next_b = QtWidgets.QPushButton("▶")
        next_b.clicked.connect(lambda: self.slider.setValue(self.slider.value() + 1))
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.valueChanged.connect(self.show_frame)
        self.speed = QtWidgets.QComboBox()
        self.speed.addItems(["5 fps", "10 fps", "20 fps", "40 fps"])
        self.speed.setCurrentIndex(1)
        self.speed.currentIndexChanged.connect(self._set_speed)
        for w in (self.play_btn, prev_b, next_b):
            ctl.addWidget(w)
        ctl.addWidget(self.slider, 1)
        ctl.addWidget(self.speed)
        right.addLayout(ctl)
        for w in (self.play_btn, prev_b, next_b, self.slider):
            w.setEnabled(False)
        self.replay_widgets = (self.play_btn, prev_b, next_b, self.slider)

        # ---- Detail tab: the spaces MS-DBSCAN combines, one panel each, on the coarse level of the last result
        self.detail_tab = QtWidgets.QWidget()
        self.tabs.addTab(self.detail_tab, "Detail")
        dl = QtWidgets.QVBoxLayout(self.detail_tab)
        top = QtWidgets.QHBoxLayout()
        self.detail_info = QtWidgets.QLabel()
        self.detail_info.setWordWrap(True)
        self.detail_info.setTextFormat(QtCore.Qt.RichText)
        top.addWidget(self.detail_info, 1)
        self.clear_pick_btn = QtWidgets.QPushButton("Clear")
        self.clear_pick_btn.setToolTip("Back to the per-vertex maps")
        self.clear_pick_btn.clicked.connect(lambda: self.detail_pick(None))
        top.addWidget(self.clear_pick_btn, 0, QtCore.Qt.AlignTop)
        dl.addLayout(top)
        self.detail_view = QtInteractor(self.detail_tab, shape=(2, 3), border=True, border_color="#c8c8c8", border_width=1.0)
        self.detail_view.set_background("white")
        dl.addWidget(self.detail_view.interactor, 1)
        self.detail_view.iren.add_observer("LeftButtonPressEvent", self._detail_press)
        self.detail_view.iren.add_observer("LeftButtonReleaseEvent", self._detail_release)
        self._press = None
        self.detail = None  # filled from the result when the tab is first shown
        self.detail_linked = False
        self.tabs.currentChanged.connect(self._tab_changed)
        self._detail_reset()

        # ---- Help tab (README shipped with the app)
        help_view = QtWidgets.QTextBrowser()
        help_view.setOpenExternalLinks(True)
        readme = RES / "README.md"
        help_view.setMarkdown(readme.read_text(encoding="utf-8") if readme.exists() else "README.md not found.")
        self.tabs.addTab(help_view, "Help")

    def _toggle_manual(self, auto):
        for w in (self.eps, self.ring, self.lam, self.merge, self.default_btn):
            w.setEnabled(not auto)
        self.thresholds.set_expanded(not auto)

    def _recolour(self, *_):
        """Redraw the replay frame and the Detail tab with the chosen colours and boundaries (no recomputation)."""
        if self.trace is not None and self.frame_index:
            self.show_frame(self.slider.value())
        if self.detail is not None:
            self._draw_detail(self.detail_pick_index)

    def load_default(self):
        self.eps.setValue(DEFAULTS["eps_normal"])
        self.ring.setValue(DEFAULTS["n_ring"])
        self.lam.setValue(DEFAULTS["fuse_lambda"])

    # ------------------------------------------------------------------ files
    def select_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select Folder", self.folder)
        if folder:
            self.folder = folder
            self.update_listbox()

    def update_listbox(self, select=None):
        try:
            names = sorted(f for f in os.listdir(self.folder) if f.lower().endswith(MESH_EXT))
        except OSError as e:
            names = []
            self.info.setText(f"Cannot list {self.folder}: {e}")
        self.file_list.clear()
        self.file_list.addItems(names)
        self.file_list.setToolTip(self.folder)
        if select in names:
            self.file_list.setCurrentRow(names.index(select))

    def preview(self):
        item = self.file_list.currentItem()
        if item is None:
            self.info.setText("Select a mesh in the list first (or choose a folder)")
            return
        self.open_mesh(os.path.join(self.folder, item.text()))

    # ------------------------------------------------------------------ actions
    def open_mesh(self, path=None):
        if not path:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open mesh", self.folder,
                                                            "Meshes (*.obj *.ply *.stl *.vtk)")
        if not path:
            return
        folder, name = os.path.split(os.path.abspath(path))
        if os.path.normcase(folder) != os.path.normcase(os.path.abspath(self.folder)):
            self.folder = folder
            self.update_listbox(select=name)
        try:
            self.V, self.F = load_mesh(path)
        except Exception:
            QtWidgets.QMessageBox.critical(self, "Cannot read mesh", traceback.format_exc())
            return
        self.path = path
        self.trace = self.result = None
        self._detail_reset()
        self.stage_list.clear()
        for w in self.replay_widgets + (self.save_btn, self.shot_btn, self.gif_btn):
            w.setEnabled(False)
        self.info.setText(f"{os.path.basename(path)}\n{len(self.V)} vertices, {len(self.F)} faces")
        self.run_btn.setEnabled(True)
        self.poly = {"input": self._polydata(self.V, self.F)}
        self.poly["input"].cell_data["rgb"] = np.full((len(self.F), 3), 210, dtype=np.uint8)
        self.edge_pairs = {}
        self.view.remove_actor("boundary", reset_camera=False, render=False)
        self.boundary_shown = None
        self.level_shown = None
        self._show_level("input", force=True)
        self.view.reset_camera()

    def run(self):
        kwargs = dict(refine=self.refine.currentText(), coarse_cap=self.cap.value())
        if self.auto.currentIndex() == 1:  # manual
            kwargs.update(params=dict(eps_normal=self.eps.value(), n_ring=self.ring.value(), fuse_lambda=self.lam.value()),
                          merge=self.merge.currentIndex() == 1)
        self.run_btn.setEnabled(False)
        self.status.setText("running…")
        self.t_start = time.time()
        self.worker = Worker(self.V, self.F, kwargs)
        self.worker.progress.connect(lambda p, s: (self.bar.setValue(p), self.status.setText(s)))
        self.worker.done.connect(self._finished)
        self.worker.failed.connect(self._failed)
        self.worker.start()

    def _failed(self, text):
        self.run_btn.setEnabled(True)
        self.status.setText("failed")
        self.last_error = text
        if not getattr(self, "headless", False):
            QtWidgets.QMessageBox.critical(self, "Segmentation failed", text)

    def _finished(self, res, tr, pairs):
        self.result, self.trace = res, tr
        self.run_btn.setEnabled(True)
        t = res["timings"]
        p = res["params"]
        n_seg, n_un = segment_count(res["vertex_labels"])
        self.status.setText(
            f"{n_seg} segments" + (f", {n_un} vertices unassigned" if n_un else "")
            + f" in {res['seconds']:.1f} s (coarse level {res['coarse_vertices']} vertices)\n"
            f"eps {p['eps_normal']}, ring {p['n_ring']}, lambda {p['fuse_lambda']}, merge {'on' if res['merge'] else 'off'}\n"
            + ", ".join(f"{k} {v:.1f} s" for k, v in t.items()))
        self.poly = {k: self._polydata(V, F) for k, (V, F, _) in tr.levels.items()}
        self.edge_pairs = dict(pairs)  # built by the worker for every level
        self.frame_index = [(si, f) for si, s in enumerate(tr.stages) for f in range(s["n_frames"])]
        self.stage_list.clear()
        start = 0
        for s in tr.stages:
            it = QtWidgets.QListWidgetItem(f"{s['name']}  [{s['level']}, {s['n_frames']} frames]")
            it.setData(QtCore.Qt.UserRole, start)
            it.setToolTip(s["note"])
            self.stage_list.addItem(it)
            start += s["n_frames"]
        for w in self.replay_widgets + (self.save_btn, self.shot_btn, self.gif_btn):
            w.setEnabled(True)
        self.stages_region.set_expanded(True)
        self.export_region.set_expanded(True)
        self.slider.blockSignals(True)
        self.slider.setRange(0, len(self.frame_index) - 1)
        self.slider.blockSignals(False)
        self.level_shown = None
        self.slider.setValue(len(self.frame_index) - 1)  # show the result first
        self.show_frame(self.slider.value())
        self._detail_reset()
        if self.tabs.currentWidget() is self.detail_tab:
            self.populate_detail()

    # ------------------------------------------------------------------ Detail tab
    def _detail_reset(self):
        """Forget the spaces of the previous result; they are computed again when the tab is shown."""
        self.detail = None
        self.detail_pick_index = None
        self.clear_pick_btn.setEnabled(False)
        for name in ("data", "ref", "ball", "marker", "title", "boundary"):
            self.detail_view.remove_actor(name, reset_camera=False, render=False)
        for title in list(self.detail_view.scalar_bars.keys()):
            self.detail_view.remove_scalar_bar(title, render=False)
        self.detail_info.setText("Press <b>Compute</b> first.")
        self.detail_view.render()

    def _tab_changed(self, _index):
        if self.tabs.currentWidget() is self.detail_tab and self.detail is None and self.trace is not None:
            self.populate_detail()

    def populate_detail(self):
        """Spaces of the coarse level of the last result, with its thresholds (msdbscan.spaces, v2's own code)."""
        V, F, _ = self.trace.levels["coarse"]
        mesh = TriMesh(V, F)
        p = self.result["params"]
        S = vertex_spaces(mesh, p["eps_normal"], p["n_ring"], p["fuse_lambda"])
        # coarse labels at the end of the clustering (after hole filling, before the pyramid)
        labels = next(s["end"] if "end" in s else s["frames"][-1] for s in reversed(self.trace.stages) if s["level"] == "coarse")
        faces = np.hstack([np.full((len(F), 1), 3), F]).ravel()
        self.detail = dict(mesh=mesh, S=S, labels=np.asarray(labels), faces=faces,
                           how="auto" if self.result["table"] is not None else "manual")
        self._draw_detail(None)
        if not self.detail_linked:  # the four mesh panels turn together, and so do the two spheres
            self.detail_view.link_views([1, 3, 4, 5])
            self.detail_view.link_views([0, 2])
            self.detail_linked = True
        # the spheres share one camera that fits 99 % of the fused points (a few spikes of curvature would
        # otherwise shrink the unit sphere to a dot); the mesh panels share another
        R = max(1.05, float(np.percentile(np.linalg.norm(S.fused, axis=1), 99)))
        self.detail_view.subplot(0, 2)
        self.detail_view.reset_camera(bounds=(-R, R, -R, R, -R, R))
        self.detail_view.subplot(0, 1)
        self.detail_view.reset_camera()
        self.detail_view.render()

    def detail_pick(self, i):
        if self.detail is None:
            return
        self.detail_pick_index = i
        self.clear_pick_btn.setEnabled(i is not None)
        self._draw_detail(i)

    def _draw_detail(self, pick):
        d, dv = self.detail, self.detail_view
        S, mesh, lab = d["S"], d["mesh"], d["labels"]
        n = mesh.n_vertices
        nb = S.neighbourhood(pick) if pick is not None else None
        seg_rgb = palette(lab, self.seg_colours.currentText())
        curv_map, count_map = MAP_SCHEMES[self.map_colours.currentText()]
        for name in ("data", "ref", "ball", "marker"):
            dv.remove_actor(name, reset_camera=False, render=False)
        for title in list(dv.scalar_bars.keys()):
            dv.remove_scalar_bar(title, render=False)

        def sets(ball):  # per-vertex colours after a pick; ball = the eps ball of the panel's space
            c = np.tile(np.array(REST, dtype=np.uint8), (n, 1))
            c[ball] = BALL_ONLY
            c[nb["ring"]] = RING_ONLY
            c[np.intersect1d(nb["ring"], ball)] = IN_N
            c[pick] = PICK
            return c

        def surface(**kw):
            return pv.PolyData(mesh.V, d["faces"]), dict(name="data", reset_camera=False, smooth_shading=False,
                                                         ambient=0.25, diffuse=0.85, **kw)

        def bar(title):  # vertical, at the right edge: the title sits above the labels
            return dict(title=title, n_labels=3, fmt="%.1f", title_font_size=12, label_font_size=11, vertical=True,
                        position_x=0.86, position_y=0.12, width=0.07, height=0.55, color="black")

        def mark():
            if pick is not None:
                dv.add_mesh(pv.Sphere(radius=0.9 * mesh.mean_edge_length, center=mesh.V[pick]), color=np.array(PICK) / 255,
                            name="marker", pickable=False, reset_camera=False)

        def sphere_panel(points, ball_idx, title):
            dv.add_mesh(pv.Sphere(radius=1.0, theta_resolution=60, phi_resolution=30), color="lightgrey", opacity=0.12,
                        name="ref", pickable=False, reset_camera=False)
            cloud = pv.PolyData(np.asarray(points, dtype=float))
            cloud.point_data["rgb"] = seg_rgb if pick is None else sets(ball_idx)
            # flat square points: shaded balls are hard to read at this size
            dv.add_mesh(cloud, scalars="rgb", rgb=True, point_size=5, render_points_as_spheres=False, name="data",
                        reset_camera=False)
            if pick is not None:
                dv.add_mesh(pv.Sphere(radius=S.eps_normal, center=points[pick]), color=np.array(PICK) / 255, opacity=0.2,
                            name="ball", pickable=False, reset_camera=False)
            dv.add_text(title, position="upper_edge", font_size=10, color="black", name="title")

        dv.subplot(0, 0)
        sphere_panel(S.normal, nb["normal_ball"] if nb else None, "Normal space")
        dv.subplot(0, 1)
        q = float(np.percentile(np.abs(S.curv), 98)) or 1.0
        m, kw = surface()
        m.point_data["c"] = S.curv
        dv.add_mesh(m, scalars="c", cmap=curv_map, clim=[-q, q], scalar_bar_args=bar("c"), **kw)
        mark()
        dv.add_text("Curvature", position="upper_edge", font_size=10, color="black", name="title")
        dv.subplot(0, 2)
        sphere_panel(S.fused, nb["fused_ball"] if nb else None, f"Fused space (lambda {S.fuse_lambda:g})")
        dv.subplot(1, 0)
        m, kw = surface()
        if pick is None:
            m.point_data["n-ring size"] = np.diff(S.ring.indptr)
            dv.add_mesh(m, scalars="n-ring size", cmap=count_map, scalar_bar_args=bar("n-ring"), **kw)
        else:
            c = np.tile(np.array(REST, dtype=np.uint8), (n, 1))
            c[nb["ring"]] = RING_ONLY
            c[pick] = PICK
            m.point_data["rgb"] = c
            dv.add_mesh(m, scalars="rgb", rgb=True, **kw)
        mark()
        dv.add_text(f"n-ring (n = {S.n_ring})", position="upper_edge", font_size=10, color="black", name="title")
        dv.subplot(1, 1)
        m, kw = surface()
        if pick is None:
            m.point_data["N(i) size (grey: not core)"] = S.n_neighbours
            dv.add_mesh(m, scalars="N(i) size (grey: not core)", cmap=count_map, below_color="dimgray",
                        clim=[MIN_PTS - 1.5, max(float(S.n_neighbours.max()), MIN_PTS)],
                        scalar_bar_args=bar("N(i)"), **kw)
        else:
            m.point_data["rgb"] = sets(nb["fused_ball"])
            dv.add_mesh(m, scalars="rgb", rgb=True, **kw)
        mark()
        dv.add_text("N(i) = n-ring AND ball", position="upper_edge", font_size=10, color="black", name="title")
        dv.subplot(1, 2)
        m, kw = surface()
        m.point_data["rgb"] = seg_rgb
        dv.add_mesh(m, scalars="rgb", rgb=True, **kw)
        d["boundary"] = self._draw_boundary(dv, "coarse", mesh.V, mesh.F, face_labels(mesh.F, lab))
        mark()
        dv.add_text("Result", position="upper_edge", font_size=10, color="black", name="title")
        dv.render()

        head = f"eps {S.eps_normal:g} · n-ring {S.n_ring} · λ {S.fuse_lambda:g} ({d['how']}) · "
        if pick is None:
            self.detail_info.setText(head + f"core {int(S.core.sum())}/{n} · <b>click a vertex</b>")
        else:
            self.detail_info.setText(
                head + f"<b>vertex {pick}</b>: n-ring {len(nb['ring'])} · ball {len(nb['fused_ball'])} · "
                f"<b>N(i) {len(nb['N'])}</b> → {'core' if nb['core'] else 'not core'} &nbsp;&nbsp; " + DETAIL_LEGEND)

    # pyvista attaches mouse observers to the interactor style, so the position is read from the interactor
    def _detail_press(self, _obj, _event):
        self._press = self.detail_view.iren.interactor.GetEventPosition()

    def _detail_release(self, _obj, _event):
        press, self._press = self._press, None
        if press is None or self.detail is None:
            return
        iren = self.detail_view.iren.interactor
        x, y = iren.GetEventPosition()
        if abs(x - press[0]) + abs(y - press[1]) > 3:  # a drag turns the camera; only a click picks
            return
        ren = iren.FindPokedRenderer(x, y)
        cloud = any(tuple(ren.GetViewport()) == tuple(self.detail_view.renderers[k].GetViewport()) for k in (0, 2))
        # Sphere panels: the point nearest the click ray, within 0.5 % of the window diagonal. Mesh panels:
        # the face the ray hits first, with no tolerance; with one, a part of the mesh nearer the camera a few
        # pixels away from the click wins (on a 277k-vertex statue a click picked a vertex 9 edges away).
        picker = vtkPointPicker() if cloud else vtkCellPicker()
        picker.SetTolerance(0.005 if cloud else 0.0)
        # only the data layers are pickable; every panel's points are the coarse vertices in the same order
        if picker.Pick(x, y, 0, ren) and picker.GetActor() is not None:
            pid = picker.GetPointId()
            if 0 <= pid < self.detail["mesh"].n_vertices:
                self.detail_pick(int(pid))

    # ------------------------------------------------------------------ replay
    def _polydata(self, V, F):
        return pv.PolyData(np.asarray(V, dtype=float), np.hstack([np.full((len(F), 1), 3), F]).ravel())

    def _show_level(self, key, force=False):
        if key is None or (key == self.level_shown and not force):
            return
        cam = self.view.camera_position if self.level_shown is not None else None
        self.view.remove_actor("mesh", reset_camera=False)  # keep lights and camera (clear() drops the lights)
        self.view.add_mesh(self.poly[key], scalars="rgb", rgb=True, name="mesh", smooth_shading=False,
                           ambient=0.25, diffuse=0.85, specular=0.15, specular_power=20)
        if cam is not None:
            self.view.camera_position = cam
        self.level_shown = key

    def show_frame(self, g):
        if not self.trace or not self.frame_index:
            return
        g = int(np.clip(g, 0, len(self.frame_index) - 1))
        si, f = self.frame_index[g]
        stage = self.trace.stages[si]
        V, F, _ = self.trace.levels[stage["level"]]
        lab = self.trace.frame(stage, f)
        poly = self.poly[stage["level"]]
        flab = face_labels(F, lab)
        poly.cell_data["rgb"] = palette(flab, self.seg_colours.currentText())
        if stage["level"] != self.level_shown:
            self._show_level(stage["level"])
        self.boundary_shown = self._draw_boundary(self.view, stage["level"], V, F, flab)
        self.view.render()
        cap = stage["captions"][f] if "captions" in stage else stage["note"]
        self.caption.setText(f"<b>{si + 1}/{len(self.trace.stages)} {stage['name']}</b> — frame {f + 1}/{stage['n_frames']} "
                             f"on the {stage['level']} ({len(V)} vertices)<br>{cap}")
        self.stage_list.blockSignals(True)
        self.stage_list.setCurrentRow(si)
        self.stage_list.blockSignals(False)

    def _draw_boundary(self, plotter, key, V, F, flab):
        """Pink lines on the segment boundaries of the shown labels when Show edge is on (the original
        application drew them with line width 5); the active subplot of `plotter` gets them."""
        plotter.remove_actor("boundary", reset_camera=False, render=False)
        if self.edges.currentIndex() != 1:
            return None
        if key not in self.edge_pairs:  # per level, kept until the next mesh or result
            self.edge_pairs[key] = edge_face_pairs(np.asarray(F))
        e = boundary_edges(self.edge_pairs[key], flab)
        if len(e):
            used, inv = np.unique(e, return_inverse=True)
            lines = np.hstack([np.full((len(e), 1), 2), inv.reshape(-1, 2)]).ravel()
            actor = plotter.add_mesh(pv.PolyData(np.asarray(V, dtype=float)[used], lines=lines), color="pink",
                                     line_width=5, name="boundary", pickable=False, reset_camera=False)
            actor.mapper.SetRelativeCoincidentTopologyLineOffsetParameters(0, BOUNDARY_OFFSET)
        return e

    def _jump_to_stage(self, item):
        self.slider.setValue(item.data(QtCore.Qt.UserRole))

    def toggle_play(self):
        if self.timer.isActive():
            self.timer.stop()
            self.play_btn.setText("Play")
            return
        if self.slider.value() >= self.slider.maximum():
            self.slider.setValue(0)
        self._set_speed()
        self.timer.start()
        self.play_btn.setText("Pause")

    def _set_speed(self, *_):
        fps = [5, 10, 20, 40][self.speed.currentIndex()]
        self.timer.setInterval(int(1000 / fps))

    def _tick(self):
        if self.slider.value() >= self.slider.maximum():
            self.toggle_play()
            return
        self.slider.setValue(self.slider.value() + 1)

    # ------------------------------------------------------------------ export
    def save_labels(self):
        stem = os.path.splitext(self.path)[0] + "_msdbscan"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save labels", stem + ".npz", "NumPy archive (*.npz)")
        if not path:
            return
        r = self.result
        np.savez_compressed(path, V=r["mesh"].V, F=r["mesh"].F, face_labels=r["face_labels"], vertex_labels=r["vertex_labels"],
                            params=json.dumps(dict(r["params"], merge=r["merge"], refine=self.refine.currentText(),
                                                   coarse_cap=self.cap.value(), source=self.path)))
        self.status.setText(f"saved {path}")

    def screenshot(self, path=None):
        if not path:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Screenshot", os.path.splitext(self.path)[0] + ".png", "PNG (*.png)")
        if path:
            self.view.screenshot(path)
            self.status.setText(f"saved {path}")


    def export_gif(self, path=None, max_frames=150, fps=12):
        """Write the replay as an animated GIF (at most `max_frames` frames, every stage keeps its last frame)."""
        if not path:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Replay to GIF", os.path.splitext(self.path)[0] + "_replay.gif",
                                                            "GIF (*.gif)")
        if not path:
            return
        # every stage gets its first and last frame; the parameter search at most 16 frames, other stages at most 30
        keep, start = set(), 0
        for s in self.trace.stages:
            cap = 16 if "frames" in s else 30
            keep |= {start + int(f) for f in np.linspace(0, s["n_frames"] - 1, min(s["n_frames"], cap)).round()}
            start += s["n_frames"]
        if len(keep) > max_frames:
            keep = set(sorted(keep)[:: int(np.ceil(len(keep) / max_frames))]) | {start - 1}
        self.view.open_gif(path, fps=fps)
        for g in sorted(keep):
            self.slider.setValue(g)
            self.show_frame(g)
            self.view.write_frame()
        self.view.mwriter.close()
        self.status.setText(f"saved {path} ({len(keep)} frames)")


def window_capture(w, png):
    """The whole window as a PNG. grab() cannot read the OpenGL views, so the visible one is pasted in."""
    pm = w.grab()
    painter = QtGui.QPainter(pm)
    for view in (w.view, w.detail_view):
        if not view.interactor.isVisible():
            continue
        img = np.ascontiguousarray(view.screenshot(return_img=True)[:, :, :3])
        qimg = QtGui.QImage(img.data, img.shape[1], img.shape[0], 3 * img.shape[1], QtGui.QImage.Format_RGB888)
        painter.drawImage(QtCore.QRect(view.interactor.mapTo(w, QtCore.QPoint(0, 0)), view.interactor.size()), qimg)
    painter.end()
    pm.save(png)


def detail_check(w) -> str:
    """Detail tab of the self-test: the core vertices of msdbscan.spaces on the coarse mesh rebuilt from the
    trace must be the ones the clustering recorded; then one vertex is picked (the core vertex with the
    most n-ring vertices in other segments among those facing the camera)."""
    w.tabs.setCurrentWidget(w.detail_tab)
    QtWidgets.QApplication.processEvents()
    if w.detail is None:
        raise RuntimeError("the Detail tab was not filled")
    S, lab = w.detail["S"], w.detail["labels"]
    recorded = next(s for s in reversed(w.trace.stages) if s["name"] == "Core elements")["end"] == CORE
    if not np.array_equal(S.core, recorded):
        raise RuntimeError(f"Detail tab: {int((S.core != recorded).sum())} vertices differ in core status from the clustering")
    R = S.ring.tocoo()
    cross = np.bincount(R.row[lab[R.row] != lab[R.col]], minlength=len(lab))
    V = w.detail["mesh"].V
    cam = np.asarray(w.detail_view.renderers[4].camera.position)
    to_cam = cam - V
    facing = (S.normal * to_cam).sum(axis=1) / np.linalg.norm(to_cam, axis=1) > 0.5
    # facing the camera is not enough on a statue: another part can hide the vertex, and the click then
    # rightly picks that part. Visible = the first hit of the ray from the camera is at the vertex
    # (pyvista ray_trace, vtkOBBTree; independent of the app's click handling).
    surf = pv.PolyData(V, w.detail["faces"])
    hidden = 0
    for i in np.argsort(-np.where(S.core & facing, cross, -1))[:300]:
        pts, _ = surf.ray_trace(cam, V[i], first_point=True)
        if len(pts) and np.linalg.norm(pts - V[i]) < w.detail["mesh"].mean_edge_length:
            break
        hidden += 1
    else:
        raise RuntimeError("Detail tab: no visible core vertex among the 300 best candidates")
    i = int(i)
    # a real click through VTK on that vertex in the N(i) panel must pick it or a vertex of a face around it
    from vtkmodules.vtkRenderingCore import vtkCoordinate
    coord = vtkCoordinate()
    coord.SetCoordinateSystemToWorld()
    coord.SetValue(*V[i])
    x, y = coord.GetComputedDisplayValue(w.detail_view.renderers[4])
    iren = w.detail_view.iren.interactor
    iren.SetEventInformation(int(x), int(y))
    iren.InvokeEvent("LeftButtonPressEvent")
    iren.InvokeEvent("LeftButtonReleaseEvent")
    QtWidgets.QApplication.processEvents()
    clicked = w.detail_pick_index
    F = w.detail["mesh"].F
    around = set(F[(F == i).any(axis=1)].ravel().tolist())
    if clicked not in around:
        raise RuntimeError(f"Detail tab: a click on vertex {i} at pixel ({x}, {y}) picked {clicked}")
    # a click on the vertex's point in the fused-space panel must pick a point drawn under the cursor
    # (normals of neighbouring vertices often fall on the same pixel, so any of them is a right answer)
    def pixel(p, r):
        coord.SetValue(*p)
        return np.array(coord.GetComputedDoubleDisplayValue(w.detail_view.renderers[r]))
    x2, y2 = pixel(S.fused[i], 2)
    iren.SetEventInformation(int(x2), int(y2))
    iren.InvokeEvent("LeftButtonPressEvent")
    iren.InvokeEvent("LeftButtonReleaseEvent")
    QtWidgets.QApplication.processEvents()
    clicked2 = w.detail_pick_index
    off = float(np.linalg.norm(pixel(S.fused[clicked2], 2) - (int(x2), int(y2)))) if clicked2 is not None else np.inf
    if off > 3:
        raise RuntimeError(f"Detail tab: a click on the fused point of vertex {i} picked {clicked2}, drawn {off:.1f} px away")
    w.detail_pick(i)
    QtWidgets.QApplication.processEvents()
    nb = S.neighbourhood(i)
    return (f"DETAIL OK core vertices {int(S.core.sum())}/{len(lab)} identical to the clustering; "
            f"click on vertex {i} picked {clicked} ({hidden} better candidates hidden from the camera); "
            f"click on its fused point picked {clicked2} ({off:.1f} px from the click); vertex {i}: "
            f"n-ring {len(nb['ring'])}, normal ball {len(nb['normal_ball'])}, fused ball {len(nb['fused_ball'])}, "
            f"N(i) {len(nb['N'])}, {'core' if nb['core'] else 'not core'}")


def colour_check(w, png_jet) -> str:
    """Colour settings of the self-test: switching to jet must redraw the replay frame and the Detail tab
    in jet (captured to png_jet), and switching back must give the first colours again."""
    app = QtWidgets.QApplication.instance()
    w.detail_pick(None)
    g = w.slider.value()
    si, f = w.frame_index[g]
    stage = w.trace.stages[si]
    F = w.trace.levels[stage["level"]][1]
    face_lab = face_labels(F, w.trace.frame(stage, f))
    shown = lambda: np.array(w.poly[stage["level"]].cell_data["rgb"])  # noqa: E731
    result_rgb = lambda: np.array(w.detail_view.renderers[5].actors["data"].mapper.dataset.point_data["rgb"])  # noqa: E731

    def curv_mid():  # colour of c = 0 in the curvature panel (middle of its symmetric range)
        lut = w.detail_view.renderers[1].actors["data"].mapper.lookup_table
        return np.array(lut.GetTableValue(lut.GetNumberOfTableValues() // 2)[:3])

    before, before_detail = shown(), result_rgb()
    w.seg_colours.setCurrentText(SEGMENT_SCHEMES[1])
    w.map_colours.setCurrentText("Jet (original app)")
    app.processEvents()
    jet_rgb = palette(face_lab, SEGMENT_SCHEMES[1])
    if not np.array_equal(shown(), jet_rgb) or np.array_equal(jet_rgb, before):
        raise RuntimeError("colour setting: the replay frame was not redrawn in jet")
    if not np.array_equal(result_rgb(), palette(w.detail["labels"], SEGMENT_SCHEMES[1])):
        raise RuntimeError("colour setting: the Detail result panel was not redrawn in jet")
    # the last frame and the Detail result show the same segments, so they must get the same jet colours
    lab_d = w.detail["labels"]
    same_ids = np.unique(lab_d[lab_d >= 0]).tolist() == np.unique(face_lab[face_lab >= 0]).tolist()
    if same_ids:
        pairs_b = {tuple(c) for c in np.unique(np.c_[face_lab, jet_rgb], axis=0).tolist()}
        pairs_d = {tuple(c) for c in np.unique(np.c_[lab_d, result_rgb()], axis=0).tolist()}
        if pairs_b != pairs_d:
            raise RuntimeError("colour setting: a segment has different jet colours in the Basic and Detail tabs")
    if np.abs(curv_mid() - np.array(colormaps["jet"](0.5)[:3])).max() > 0.03:
        raise RuntimeError(f"colour setting: the curvature panel is not in jet (middle colour {curv_mid()})")
    window_capture(w, png_jet)
    n_seg = len(np.unique(face_lab[face_lab >= 0]))
    w.seg_colours.setCurrentIndex(0)
    w.map_colours.setCurrentIndex(0)
    app.processEvents()
    if not (np.array_equal(shown(), before) and np.array_equal(result_rgb(), before_detail)):
        raise RuntimeError("colour setting: switching back did not give the first colours")
    if np.abs(curv_mid() - np.array(colormaps["coolwarm"](0.5)[:3])).max() > 0.03:
        raise RuntimeError("colour setting: the curvature panel did not go back to coolwarm")
    return (f"COLOURS OK replay frame ({n_seg} segments) and Detail tab redrawn in jet and back; same jet colour per "
            f"segment in both tabs: {'yes' if same_ids else 'not compared (the tabs show different segment ids)'}; "
            f"jet capture {png_jet}")


def boundary_check(w, png_edges) -> str:
    """Show edge of the self-test: with it on, the lines drawn on the last replay frame and on the Detail
    result must be the mesh edges whose faces carry more than one label, counted here by brute force as
    the original highlight_edge defines them; with it off, no line may remain."""
    from collections import defaultdict
    app = QtWidgets.QApplication.instance()

    def brute(F, flab):
        faces_of = defaultdict(list)
        for fi, (a, b, c) in enumerate(np.asarray(F).tolist()):
            for u, v in ((a, b), (b, c), (c, a)):
                faces_of[(min(u, v), max(u, v))].append(fi)
        return {e for e, fs in faces_of.items() if len({int(flab[f]) for f in fs}) > 1}

    def drawn(plotter, renderer):
        actor = plotter.renderers[renderer].actors.get("boundary")
        return 0 if actor is None else actor.mapper.dataset.n_cells

    si, f = w.frame_index[w.slider.value()]
    stage = w.trace.stages[si]
    _, F, _ = w.trace.levels[stage["level"]]
    flab = face_labels(F, w.trace.frame(stage, f))
    mesh_d = w.detail["mesh"]
    flab_d = face_labels(mesh_d.F, w.detail["labels"])
    # the worker must have built the edge table of every level, so switching Show edge on builds nothing
    missing = set(w.trace.levels) - set(w.edge_pairs)
    if missing:
        raise RuntimeError(f"Show edge: no edge table from the worker for {sorted(missing)}")
    tables = {k: id(v) for k, v in w.edge_pairs.items()}
    w.edges.setCurrentIndex(1)
    app.processEvents()
    for where, got, ref, n in (("replay frame", w.boundary_shown, brute(F, flab), drawn(w.view, 0)),
                               ("Detail result", w.detail["boundary"], brute(mesh_d.F, flab_d), drawn(w.detail_view, 5))):
        if got is None or {tuple(e) for e in got.tolist()} != ref or n != len(ref) or not ref:
            raise RuntimeError(f"Show edge, {where}: {n} lines drawn, {0 if got is None else len(got)} computed, "
                               f"{len(ref)} by brute force")
    if {k: id(v) for k, v in w.edge_pairs.items()} != tables:
        raise RuntimeError("Show edge: an edge table was built again in the window thread")
    w.tabs.setCurrentIndex(0)
    app.processEvents()
    window_capture(w, png_edges)
    n_frame, n_detail = len(w.boundary_shown), len(w.detail["boundary"])
    w.edges.setCurrentIndex(0)
    app.processEvents()
    if drawn(w.view, 0) or drawn(w.detail_view, 5):
        raise RuntimeError("Show edge: lines remain after switching it off")
    return (f"EDGES OK segment boundaries = brute force: {n_frame} edges on the {stage['level']}, "
            f"{n_detail} on the Detail result; none after switching off; edge tables of {len(tables)} levels built "
            f"by the worker in {w.result['timings']['boundary edges']:.2f} s, none in the window thread; capture {png_edges}")


def selftest(mesh_path, png):
    """Headless: load, segment through the worker, step through the replay, save a screenshot."""
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    w = Studio()
    w.headless = True  # errors go to the log, no dialog
    w.show()
    w.open_mesh(mesh_path)
    w.run()
    w.worker.wait()
    app.processEvents()
    if w.trace is None:
        w.close()
        selftest_failed(str(getattr(w, "last_error", w.status.text())))
    # the count shown must leave out the unassigned label (checked before the GIF export rewrites the status line)
    n_seg, n_un = segment_count(w.result["vertex_labels"])
    if not w.status.text().startswith(f"{n_seg} segments" + (f", {n_un} vertices unassigned" if n_un else "") + " in "):
        raise RuntimeError(f"segment count: the status line does not show {n_seg} segments and {n_un} unassigned vertices")
    n = len(w.frame_index)
    for g in list(range(0, n, max(1, n // 40))) + [n - 1]:
        w.slider.setValue(g)
        app.processEvents()
    w.screenshot(png)
    window_capture(w, png.replace(".png", "_window.png"))
    w.export_gif(png.replace(".png", "_replay.gif"), max_frames=90)
    detail = detail_check(w)
    window_capture(w, png.replace(".png", "_detail.png"))
    colours = colour_check(w, png.replace(".png", "_jet.png"))
    edges = boundary_check(w, png.replace(".png", "_edges.png"))
    lines = [f"SELFTEST OK stages {len(w.trace.stages)} frames {n} segments {n_seg} unassigned vertices {n_un} "
             f"seconds {w.result['seconds']:.1f} -> {png}", detail, colours, edges]
    lines += [f"  {s['name']:55s} {s['level']:28s} frames {s['n_frames']}" for s in w.trace.stages]
    print("\n".join(lines), flush=True)
    # a windowed (no console) build has no stdout, so the result is also written to a file
    Path(png).with_name(Path(png).stem + "_selftest.txt").write_text(
        f"MS-DBSCAN Studio {__version__} ({'frozen' if FROZEN else 'source'}), mesh {mesh_path}\n" + "\n".join(lines) + "\n",
        encoding="utf-8")
    w.close()


if __name__ == "__main__":
    multiprocessing.freeze_support()  # needed in a PyInstaller bundle if any library starts worker processes
    if len(sys.argv) == 2 and sys.argv[1] == "--version":
        print(f"MS-DBSCAN Studio {__version__}")
        sys.exit(0)
    if SELFTEST:
        try:
            selftest(sys.argv[2], sys.argv[3])
        except Exception:
            selftest_failed(traceback.format_exc())
        sys.exit(0)
    app = QtWidgets.QApplication(sys.argv)
    win = Studio()
    win.show()
    if len(sys.argv) > 1:
        win.open_mesh(sys.argv[1])
    sys.exit(app.exec_())
