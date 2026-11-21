#!/usr/bin/env bash
# Build the Linux release of MS-DBSCAN Studio. Run it inside Linux (here: WSL Ubuntu 22.04):
#
#   bash app/build_linux.sh            # build, self-test the frozen app, archive
#   bash app/build_linux.sh --no-test
#
# Creates a virtual environment outside the repository (BUILD_VENV, default
# ~/.venvs/msdbscan-studio-build), installs app/requirements-build.txt into it, and runs
# app/build_release.py with its work folder on the Linux file system (BUILD_WORK, default
# ~/msdbscan-studio-build), not on a mounted Windows drive. The archive, its checksum, the
# build info and the self-test output go to release/ in the repository.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${BUILD_VENV:-$HOME/.venvs/msdbscan-studio-build}"
WORK="${BUILD_WORK:-$HOME/msdbscan-studio-build}"
if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV"
fi
"$VENV/bin/python" -m pip install --upgrade pip
"$VENV/bin/python" -m pip install -r "$REPO/app/requirements-build.txt"
"$VENV/bin/python" -u "$REPO/app/build_release.py" --workdir "$WORK" "$@"
