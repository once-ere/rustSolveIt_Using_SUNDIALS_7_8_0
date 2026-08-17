#!/usr/bin/env bash
# c_build.sh — build the unmodified upstream SUNDIALS 7.8.0 C library and
# every example this machine has the dependencies for.
#
#   tools/c_build.sh [build-dir]        # default: build/c
#
# The upstream C source tree is located, in order, from:
#   1. $SUNDIALS_C_SRC
#   2. ./upstream-c            (a symlink you create; see requirements.md)
#   3. ..                      (the layout the sibling repositories use)
#
# Nothing is written into that tree; the build is entirely out of source.
#
# Which backends get switched on is decided by probing the machine, and the
# decision is printed and logged so the example results can be read against
# it. Anything that could not be enabled is reported by
# tools/c_requirements.sh and listed in requirements.md.
#
# Optional third-party backends are tried as a ladder: the full set first,
# then with the riskiest entries dropped one at a time, until CMake
# configures. That way one uncooperative package costs only itself instead
# of collapsing the whole build to serial-only. The level that succeeded is
# logged.
set -u

cd "$(dirname "$0")/.."
WS=$PWD
BUILD=${1:-$WS/build/c}
LOG=$WS/logs/c_build.log
mkdir -p "$WS/logs" "$(dirname "$BUILD")"

SRC=${SUNDIALS_C_SRC:-}
if [ -z "$SRC" ] || [ ! -f "$SRC/CMakeLists.txt" ]; then
  if [ -f "$WS/upstream-c/CMakeLists.txt" ]; then
    SRC=$WS/upstream-c
  elif [ -f "$WS/../CMakeLists.txt" ]; then
    SRC=$(cd .. && pwd)
  fi
fi
[ -n "$SRC" ] && [ -f "$SRC/CMakeLists.txt" ] || {
  echo "ERROR: no upstream SUNDIALS C tree found."
  echo "       set SUNDIALS_C_SRC=/path/to/sundials-7.8.0, or"
  echo "       ln -s /path/to/sundials-7.8.0 upstream-c"
  exit 1
}
SRC=$(cd "$SRC" && pwd)
LIBDIR=/usr/lib/x86_64-linux-gnu

have()      { command -v "$1" >/dev/null 2>&1; }
have_file() { [ -e "$1" ]; }
have_glob() { ls $1 >/dev/null 2>&1; }

: >"$LOG"
say() { printf '%s\n' "$*" | tee -a "$LOG"; }
note() { printf '  %-16s %s\n' "$1" "$2" | tee -a "$LOG"; }

say "== probing optional backends =="

# ---- always-on / compiler-provided --------------------------------------
BASE=(-DENABLE_OPENMP=ON -DENABLE_PTHREAD=ON)
note OpenMP on
note pthread on
if have gfortran; then
  BASE+=(-DBUILD_FORTRAN_MODULE_INTERFACE=ON); note Fortran on
else
  BASE+=(-DBUILD_FORTRAN_MODULE_INTERFACE=OFF); note Fortran "off (no gfortran)"
fi

# ---- optional third-party layers, in descending priority ----------------
# Each entry is a name plus the cmake arguments that enable it. The ladder
# drops from the end, so the most valuable ones survive longest.
TPL_NAME=()
TPL_ARGS=()
add_tpl() { TPL_NAME+=("$1"); shift; TPL_ARGS+=("$*"); }

if have mpicc && mpicc -show >/dev/null 2>&1; then
  add_tpl MPI "-DENABLE_MPI=ON -DMPI_C_COMPILER=$(command -v mpicc) -DMPI_CXX_COMPILER=$(command -v mpicxx)"
  note MPI on
else
  note MPI "off (mpicc not usable)"
fi

if have_glob "$LIBDIR/liblapack.so*" && have_glob "$LIBDIR/libblas.so*"; then
  add_tpl LAPACK "-DENABLE_LAPACK=ON"; note LAPACK on
else note LAPACK "off (liblapack-dev)"; fi

if have_file /usr/include/suitesparse/klu.h && have_glob "$LIBDIR/libklu.*"; then
  add_tpl KLU "-DENABLE_KLU=ON -DKLU_INCLUDE_DIR=/usr/include/suitesparse -DKLU_LIBRARY_DIR=$LIBDIR"
  note KLU on
else note KLU "off (libsuitesparse-dev)"; fi

if have_file /usr/include/hypre/HYPRE.h && have_glob "$LIBDIR/libHYPRE.*"; then
  add_tpl HYPRE "-DENABLE_HYPRE=ON -DHYPRE_INCLUDE_DIR=/usr/include/hypre -DHYPRE_LIBRARY_DIR=$LIBDIR"
  note hypre on
else note hypre "off (libhypre-dev)"; fi

if have_file /usr/include/superlu-dist/superlu_ddefs.h && have_glob "$LIBDIR/libsuperlu_dist.*"; then
  add_tpl SUPERLUDIST "-DENABLE_SUPERLUDIST=ON -DSUPERLUDIST_DIR=/usr -DSUPERLUDIST_INCLUDE_DIRS=/usr/include/superlu-dist -DSUPERLUDIST_LIBRARIES=$LIBDIR/libsuperlu_dist.a -DSUPERLUDIST_INDEX_SIZE=32"
  note SuperLU_DIST on
else note SuperLU_DIST "off (libsuperlu-dist-dev)"; fi

if have_file "$LIBDIR/cmake/Kokkos/KokkosConfig.cmake"; then
  add_tpl KOKKOS "-DENABLE_KOKKOS=ON -DKokkos_DIR=$LIBDIR/cmake/Kokkos"
  note Kokkos on
else note Kokkos "off (libkokkos-dev)"; fi

if have_file "$LIBDIR/cmake/Trilinos/TrilinosConfig.cmake"; then
  add_tpl TRILINOS "-DSUNDIALS_ENABLE_TRILINOS=ON -DTrilinos_DIR=$LIBDIR/cmake/Trilinos"
  note Trilinos on
else note Trilinos "off (libtrilinos-tpetra-dev)"; fi

PETSC_ROOT=""
for d in /usr/lib/petscdir/petsc-real /usr/lib/petsc /usr/lib/petscdir/3.24; do
  [ -f "$d/include/petsc.h" ] && { PETSC_ROOT=$d; break; }
done
if [ -n "$PETSC_ROOT" ]; then
  add_tpl PETSC "-DENABLE_PETSC=ON -DPETSC_DIR=$PETSC_ROOT"
  note PETSc "on ($PETSC_ROOT)"
else note PETSc "off (petsc-dev)"; fi

NVCC=$(command -v nvcc || echo /usr/local/cuda/bin/nvcc)
if [ -x "$NVCC" ]; then
  CC_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
  [ -z "$CC_CAP" ] && CC_CAP=70
  add_tpl CUDA "-DENABLE_CUDA=ON -DCMAKE_CUDA_COMPILER=$NVCC -DCMAKE_CUDA_ARCHITECTURES=$CC_CAP"
  note CUDA "on (nvcc $NVCC, sm_$CC_CAP)"
else note CUDA "off (nvcc not found)"; fi

if have_file /usr/include/magma_v2.h && { have_glob "$LIBDIR/libmagma.*" || have_glob '/usr/lib/libmagma.*'; }; then
  add_tpl MAGMA "-DENABLE_MAGMA=ON"; note MAGMA on
else note MAGMA "off (header without library, or not installed)"; fi

say ""
say "source: $SRC"
say "build:  $BUILD"
say "cc:     $(cc --version | head -1)"
say "cmake:  $(cmake --version | head -1)"
say ""

configure_with() { # <n tpls to keep>
  local keep=$1 args=() i
  for ((i = 0; i < keep; i++)); do
    # shellcheck disable=SC2206  # each entry is a pre-split argument list
    args+=(${TPL_ARGS[$i]})
  done
  cmake -S "$SRC" -B "$BUILD" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF \
    -DEXAMPLES_ENABLE_C=ON \
    -DEXAMPLES_ENABLE_CXX=ON \
    -DEXAMPLES_INSTALL=OFF \
    -DBUILD_ARKODE=ON -DBUILD_CVODE=ON -DBUILD_CVODES=ON \
    -DBUILD_IDA=ON -DBUILD_IDAS=ON -DBUILD_KINSOL=ON \
    -DSUNDIALS_ENABLE_MONITORING=ON \
    "${BASE[@]}" "${args[@]}" >>"$LOG" 2>&1
}

n=${#TPL_NAME[@]}
level=-1
FAILED_TPL=()
FAILED_WHY=()
while [ "$n" -ge 0 ]; do
  rm -rf "$BUILD" 2>/dev/null
  kept="${TPL_NAME[*]:0:$n}"
  say "configuring with: ${kept:-<no optional backends>}"
  if ! configure_with "$n"; then
    why="configure"
  elif ! cmake --build "$BUILD" -j "$(nproc)" >>"$LOG" 2>&1; then
    why="build"
  else
    level=$n; break
  fi
  if [ "$n" -eq 0 ]; then break; fi
  dropped=${TPL_NAME[$((n - 1))]}
  say "  -> $why failed; dropping $dropped and retrying"
  FAILED_TPL+=("$dropped"); FAILED_WHY+=("$why")
  n=$((n - 1))
done

if [ "$level" -lt 0 ]; then
  say "BUILD FAILED at every level — see $LOG"
  tail -30 "$LOG"
  exit 1
fi
say ""
say "configured and built with: ${TPL_NAME[*]:0:$level}"
if [ ${#FAILED_TPL[@]} -gt 0 ]; then
  say "dropped (present on this machine but unusable with this toolchain):"
  for ((i = 0; i < ${#FAILED_TPL[@]}; i++)); do
    say "  ${FAILED_TPL[$i]} — failed at ${FAILED_WHY[$i]}"
  done
fi
say ""

cnt=$(find "$BUILD/examples" -type f -perm -u+x ! -name '*.*' 2>/dev/null | wc -l)
say "built $cnt example binaries under $BUILD/examples"
