# PyInstaller spec for MS-DBSCAN Studio, Windows and Linux (one-folder build).
# Used by app/build_release.py; PyInstaller does not cross-compile, so run it on each OS.
import importlib.util
import os
import sys
from pathlib import Path

APP = Path(SPECPATH)
ROOT = APP.parent
NAME = "MSDBSCAN-Studio"
WINDOWS = sys.platform == "win32"

if WINDOWS:
    # The VTK wheel keeps its DLLs in site-packages/vtk.libs (delvewheel); vtkmodules/__init__.py adds that
    # folder at import time, but PyInstaller does not search it. Without this the bundle has no
    # vtkCommonCore-9.4.2.dll and the frozen app stops at "import pyvista" (DLL load failed). PyInstaller
    # resolves DLL dependencies through PATH, and keeps the folder name, so vtk.libs/ lands in the bundle
    # where the delvewheel code looks for it.
    VTK_LIBS = Path(importlib.util.find_spec("vtkmodules").submodule_search_locations[0]).parent / "vtk.libs"
    if not VTK_LIBS.is_dir():
        raise SystemExit(f"VTK DLL folder not found: {VTK_LIBS}")
    os.environ["PATH"] = str(VTK_LIBS) + os.pathsep + os.environ.get("PATH", "")
    # On a conda CPython older than 3.10 (this environment, and the frozen app, which carries conda's
    # python39.dll) the delvewheel code does not call os.add_dll_directory: it preloads the DLLs listed in
    # vtk.libs/.load-order-vtk-<version>. Without that file in the bundle nothing is preloaded and
    # vtkCommonCore fails to load although all its DLLs are there.
    VTK_LOAD_ORDER = sorted(VTK_LIBS.glob(".load-order-*"))
    if not VTK_LOAD_ORDER:
        raise SystemExit(f"no .load-order-* file in {VTK_LIBS}")
else:
    VTK_LOAD_ORDER = []

a = Analysis(
    [str(APP / "msdbscan_studio.py")],
    pathex=[str(ROOT / "src")],  # the clustering code is the package src/msdbscan
    datas=[(str(APP / "icon.png"), "."), (str(APP / "README.md"), ".")] + [(str(p), "vtk.libs") for p in VTK_LOAD_ORDER],
    hiddenimports=[],
    # Never imported by the app (checked by listing sys.modules after a source-mode self-test: PyQt5 is used
    # only through QtCore, QtGui and QtWidgets). PyInstaller follows optional imports of the development
    # environment's packages, which pulled in all of these; QtWebEngine stopped the build (missing locales).
    excludes=["tensorflow", "torch", "open3d", "keras", "IPython", "jupyter_client", "notebook", "tkinter",
              "PySide2", "PySide6", "PyQt6", "pytest", "sphinx",
              "sympy", "paramiko", "cryptography", "pandas", "numba", "llvmlite", "dask", "bokeh", "plotly",
              "tornado", "zmq", "pyarrow", "h5py", "tables", "netCDF4", "sqlite3", "trame", "trame_client",
              "trame_server", "trame_vtk", "trame_vuetify",
              "embreex"]  # trimesh's optional ray tracer; its embree4.dll cannot be resolved and nothing here casts rays
             + ["PyQt5." + m for m in ("QtWebEngine", "QtWebEngineCore", "QtWebEngineWidgets", "QtWebChannel",
                                       "QtWebSockets", "QtQml", "QtQuick", "QtQuick3D", "QtQuickWidgets",
                                       "QtBluetooth", "QtLocation", "QtPositioning", "QtMultimedia",
                                       "QtMultimediaWidgets", "QtNfc", "QtSensors", "QtSerialPort", "QtSql",
                                       "QtTextToSpeech", "QtRemoteObjects", "QtDesigner", "QtHelp",
                                       "QtXmlPatterns", "QtTest")],
    hooksconfig={"matplotlib": {"backends": "Agg"}},  # matplotlib is used by pyvista for colour maps only
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=NAME,
    console=not WINDOWS,  # Windows: no console window; Linux: keep stdout for the terminal
    icon=str(APP / "icon.png") if WINDOWS else None,  # converted to .ico by PyInstaller (needs Pillow)
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, name=NAME, upx=False)
