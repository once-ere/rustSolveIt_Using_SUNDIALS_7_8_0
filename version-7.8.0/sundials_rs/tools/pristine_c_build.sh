#!/usr/bin/env bash
# pristine_c_build.sh — build the UNMODIFIED upstream SUNDIALS 7.8.0 C
# library and its serial examples with the host toolchain, out of source.
#
# Purpose: root-cause the variants that diverge from the shipped reference
# .out files. A divergence is a *port* defect only if the Rust output
# differs from what the pristine C produces on this same host; if the C
# and the Rust agree and both differ from the shipped .out, the reference
# is stale and the port is correct. That comparison is the only honest way
# to classify a divergence, and it must be made natively — the macOS
# sibling made it against Apple-clang binaries, which says nothing here.
#
#   tools/pristine_c_build.sh [build-dir]
#
# Nothing is written into the upstream tree; it stays read-only.
set -u
export PATH="$HOME/.cargo/bin:$PATH"
cd "$(dirname "$0")/.."
WS_ROOT="$PWD"
# Normally the upstream C tree is this workspace's parent. In the WSL
# sandbox the parent only carries an `examples` symlink, so fall back to
# $SUNDIALS_C_SRC and then to the checkout on the Windows volume.
SRC="$(cd .. && pwd)"
if [ ! -f "$SRC/CMakeLists.txt" ]; then
  SRC="${SUNDIALS_C_SRC:-/mnt/c/Users/nsh/Developer/sundials-7.8.0}"
fi
[ -f "$SRC/CMakeLists.txt" ] || { echo "no upstream C tree at $SRC"; exit 1; }
BUILD="${1:-$HOME/sdl/cbuild}"
LOG="$WS_ROOT/logs/pristine-c-build.log"
mkdir -p "$WS_ROOT/logs"

echo "source:  $SRC"
echo "build:   $BUILD"
echo "log:     $LOG"

cmake -S "$SRC" -B "$BUILD" \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SHARED_LIBS=OFF \
      -DEXAMPLES_ENABLE_C=ON \
      -DEXAMPLES_INSTALL=OFF \
      -DBUILD_ARKODE=ON -DBUILD_CVODE=ON -DBUILD_CVODES=ON \
      -DBUILD_IDA=ON -DBUILD_IDAS=ON -DBUILD_KINSOL=ON \
      -DENABLE_MPI=OFF -DENABLE_OPENMP=OFF -DENABLE_PTHREAD=OFF \
      -DENABLE_KLU=OFF -DENABLE_LAPACK=OFF -DENABLE_SUPERLUMT=OFF \
      -DSUNDIALS_ENABLE_MONITORING=ON \
      > "$LOG" 2>&1 || { echo "CONFIGURE FAILED — see $LOG"; tail -20 "$LOG"; exit 1; }

cmake --build "$BUILD" -j "$(nproc)" >> "$LOG" 2>&1 \
  || { echo "BUILD FAILED — see $LOG"; tail -30 "$LOG"; exit 1; }

echo "built. example binaries:"
find "$BUILD/examples" -maxdepth 3 -type f -perm -u+x ! -name '*.*' | wc -l
