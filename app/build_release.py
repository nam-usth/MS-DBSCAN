"""Build a release of MS-DBSCAN Studio with PyInstaller, then test the frozen app and archive it.

PyInstaller does not cross-compile: run this on Windows for the Windows release and on Linux
(for example WSL Ubuntu, see app/build_linux.sh) for the Linux release.

    python app/build_release.py              # build, self-test the frozen app, archive
    python app/build_release.py --no-test    # build and archive without the self-test

Output (repository root):
    release/MSDBSCAN-Studio-<version>-<os>-x64.zip | .tar.gz   the one-folder build
    release/MSDBSCAN-Studio-<version>-<os>-x64.sha256
    release/build-info-<os>.json     Python, PyInstaller and library versions, self-test result
    release/selftest-<os>/           the frozen app's self-test screenshots and log
Work files: build/pyinstaller/<os>/ or --workdir <dir>/<os>/ (rebuilt every time; PyInstaller log in it).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import time
from importlib import metadata
from pathlib import Path

APP = Path(__file__).resolve().parent
ROOT = APP.parent
NAME = "MSDBSCAN-Studio"
OS = {"win32": "windows", "linux": "linux"}.get(sys.platform)
TEST_MESH = ROOT / "evaluation_dataset" / "SIGGRAPH_2006" / "brick_part06.obj"
PACKAGES = ["pyinstaller", "pyinstaller-hooks-contrib", "numpy", "scipy", "scikit-learn", "vtk", "pyvista",
            "pyvistaqt", "PyQt5", "PyQt5-Qt5", "QtPy", "trimesh", "fast-simplification", "imageio", "Pillow"]
RUNTIME_NOTE = {
    "windows": "Windows 10 or 11, 64-bit. Unzip and run MSDBSCAN-Studio.exe.",
    "linux": ("64-bit Linux with glibc 2.35 or newer (built on Ubuntu 22.04), an X11 or Wayland session and OpenGL.\n"
              "Extract and run ./MSDBSCAN-Studio. If Qt reports a missing 'xcb' platform plugin, install the X11\n"
              "client libraries, on Debian/Ubuntu:\n"
              "  sudo apt install libgl1 libegl1 libxkbcommon-x11-0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \\\n"
              "                   libxcb-render-util0 libxcb-xinerama0 libxcb-randr0 libxcb-shape0 libxcb-xfixes0 libdbus-1-3"),
}


def version() -> str:
    m = re.search(r'^__version__ = "([^"]+)"', (APP / "msdbscan_studio.py").read_text(encoding="utf-8"), re.M)
    if not m:
        sys.exit("cannot read __version__ from app/msdbscan_studio.py")
    return m.group(1)


def versions() -> dict:
    out = {"python": platform.python_version(), "platform": platform.platform()}
    for p in PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def wait_for_display(seconds: int = 60):
    """Linux: the self-test opens a window. WSLg creates its X socket some seconds after WSL starts, and a
    self-test started before that aborts inside Qt ("could not connect to display")."""
    disp = os.environ.get("DISPLAY", "")
    if not disp.startswith(":"):
        if not os.environ.get("WAYLAND_DISPLAY"):
            sys.exit("no display (DISPLAY and WAYLAND_DISPLAY unset): the self-test needs one; use --no-test to skip it")
        return
    sock = Path("/tmp/.X11-unix") / ("X" + disp[1:].split(".")[0])
    for _ in range(seconds):
        if sock.exists():
            return
        time.sleep(1)
    sys.exit(f"no X display: {sock} missing after {seconds} s (DISPLAY={disp}); use --no-test to skip the self-test")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-test", action="store_true", help="skip the self-test of the frozen app")
    ap.add_argument("--workdir", help="where PyInstaller works (default build/pyinstaller in the repository); "
                                      "on WSL use a Linux path, not the mounted Windows drive")
    args = ap.parse_args()
    if OS is None:
        sys.exit(f"unsupported platform {sys.platform}")
    ver = version()
    work = (Path(args.workdir).expanduser().resolve() if args.workdir else ROOT / "build" / "pyinstaller") / OS
    dist = work / "dist"
    release = ROOT / "release"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    release.mkdir(exist_ok=True)

    # ---- PyInstaller
    t0 = time.time()
    log = work / "pyinstaller.log"
    with open(log, "w", encoding="utf-8") as f:
        rc = subprocess.call([sys.executable, "-m", "PyInstaller", str(APP / "msdbscan_studio.spec"), "--noconfirm",
                              "--clean", "--distpath", str(dist), "--workpath", str(work / "work")],
                             stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
    if rc != 0:
        sys.exit(f"PyInstaller failed (exit {rc}); see {log}")
    folder = dist / NAME
    exe = folder / (NAME + (".exe" if OS == "windows" else ""))
    if not exe.exists():
        sys.exit(f"PyInstaller finished but {exe} is missing; see {log}")
    size = sum(p.stat().st_size for p in folder.rglob("*") if p.is_file() and not p.is_symlink())  # Linux: shared-library links
    print(f"built {folder} ({size / 2**20:.0f} MB) in {time.time() - t0:.0f} s", flush=True)
    (folder / "README.txt").write_text(
        f"MS-DBSCAN Studio {ver} ({OS}, x64)\n\n{RUNTIME_NOTE[OS]}\n\n"
        "Self-test without opening a session: MSDBSCAN-Studio --selftest <mesh.obj> <out.png>\n"
        "(writes out.png, out_window.png, out_detail.png, out_jet.png, out_edges.png, out_replay.gif and out_selftest.txt).\n"
        "Usage and the meaning of each stage: README.md in this folder, also in the Help tab.\n", encoding="utf-8")

    # ---- self-test of the frozen app (a real segmentation of a 55 896-vertex scan)
    test = {"run": False}
    if not args.no_test:
        if not TEST_MESH.exists():
            sys.exit(f"test mesh missing: {TEST_MESH} (use --no-test to skip)")
        out = release / f"selftest-{OS}"
        if out.exists():
            shutil.rmtree(out)
        out.mkdir()
        png = out / "selftest_brick_part06.png"
        # Run as on a machine without this Python: no conda or venv folders on PATH (Windows looks for DLLs
        # there) and no LD_LIBRARY_PATH, so a library the bundle lacks makes the test fail.
        env = {k: v for k, v in os.environ.items() if k.upper() not in ("PATH", "LD_LIBRARY_PATH", "PYTHONPATH", "PYTHONHOME")}
        if OS == "windows":
            sysroot = os.environ.get("SystemRoot", r"C:\Windows")
            env["PATH"] = os.pathsep.join([os.path.join(sysroot, "System32"), sysroot])
        else:
            env["PATH"] = "/usr/bin:/bin"
            wait_for_display()
        t0 = time.time()
        rc = subprocess.call([str(exe), "--selftest", str(TEST_MESH), str(png)], cwd=out, timeout=900, env=env)
        txt = png.with_name(png.stem + "_selftest.txt")
        result = txt.read_text(encoding="utf-8") if txt.exists() else "(no self-test file written)"
        test = {"run": True, "exit_code": rc, "seconds": round(time.time() - t0, 1), "result": result.splitlines()}
        print(result, flush=True)
        if rc != 0 or "SELFTEST OK" not in result:
            sys.exit(f"frozen self-test FAILED (exit {rc}); output in {out}")

    # ---- archive
    stem = f"{NAME}-{ver}-{OS}-x64"
    if OS == "windows":
        archive = Path(shutil.make_archive(str(release / stem), "zip", root_dir=dist, base_dir=NAME))
    else:
        archive = release / f"{stem}.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:  # keeps the executable bit
            tar.add(folder, arcname=NAME)
    digest = sha256(archive)
    (release / f"{stem}.sha256").write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    info = {"version": ver, "os": OS, "archive": archive.name, "sha256": digest, "archive_mb": round(archive.stat().st_size / 2**20, 1),
            "folder_mb": round(size / 2**20, 1), "built": time.strftime("%Y-%m-%d %H:%M:%S"), "versions": versions(), "selftest": test}
    (release / f"build-info-{OS}.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(f"RELEASE OK {archive} ({info['archive_mb']} MB) sha256 {digest}", flush=True)


if __name__ == "__main__":
    main()
