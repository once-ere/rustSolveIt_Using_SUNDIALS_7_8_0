#!/usr/bin/env bash
# c_requirements.sh — probe this machine for everything the SUNDIALS 7.8.0
# example suite can use, and print a markdown table of what is present and
# what is missing (with the apt package that would supply it).
#
#   tools/c_requirements.sh            # human-readable table
#   tools/c_requirements.sh --md       # same, markdown only (for requirements.md)
#
# Nothing is installed and nothing needs root: this only looks.
set -u

MD=0
[ "${1:-}" = "--md" ] && MD=1

rows=()
have_cmd() { command -v "$1" >/dev/null 2>&1; }
have_hdr() { for d in /usr/include /usr/local/include /usr/include/x86_64-linux-gnu; do
               [ -e "$d/$1" ] && return 0; done; return 1; }
have_lib() { ls /usr/lib/x86_64-linux-gnu/$1 >/dev/null 2>&1 ||
             ls /usr/lib/gcc/x86_64-linux-gnu/*/$1 >/dev/null 2>&1; }

# name | what it unlocks | probe result | apt package ("" = not packaged)
check() { # <label> <status> <examples-unlocked> <apt>
  rows+=("$1|$2|$3|$4")
}

probe() { # <label> <examples> <apt> <test...>
  local label=$1 ex=$2 apt=$3; shift 3
  if "$@"; then check "$label" "present" "$ex" "$apt"
  else check "$label" "MISSING" "$ex" "$apt"; fi
}

have_cmake_pkg() { [ -f "/usr/lib/x86_64-linux-gnu/cmake/$1/$1Config.cmake" ]; }
have_petsc() { for d in /usr/lib/petscdir/petsc-real /usr/lib/petsc /usr/lib/petscdir/3.24; do
                 [ -f "$d/include/petsc.h" ] && return 0; done; return 1; }
have_mpi()   { command -v mpicc >/dev/null 2>&1 && mpicc -show >/dev/null 2>&1; }
have_nvcc()  { command -v nvcc >/dev/null 2>&1 || [ -x /usr/local/cuda/bin/nvcc ]; }
have_magma() { [ -f /usr/include/magma_v2.h ] && have_lib 'libmagma.*'; }

probe "C compiler (cc)"    "everything"                       "gcc"                  have_cmd cc
probe "C++ compiler"       "examples/*/CXX_*"                 "g++"                  have_cmd c++
probe "Fortran compiler"   "examples/*/F2003_*"               "gfortran"             have_cmd gfortran
probe "CMake"              "the whole C build"                "cmake"                have_cmd cmake
probe "OpenMP runtime"     "examples/*/C_openmp"              "libgomp1 (with gcc)"  have_lib 'libgomp.so*'
probe "libquadmath"        "tools/libm_oracle.c ulp figures"  "gcc"                  have_lib 'libquadmath.so*'
probe "BLAS"               "the *_dnsL / *_bndL LAPACK examples" "libblas-dev"       have_lib 'libblas.so*'
probe "LAPACK"             "the *_dnsL / *_bndL LAPACK examples" "liblapack-dev"     have_lib 'liblapack.so*'
probe "MPI (mpicc works)"  "examples/*/{parallel,C_parallel,CXX_parallel}" "libopenmpi-dev" have_mpi
probe "KLU (SuiteSparse)"  "the 11 *_klu examples"            "libsuitesparse-dev"   have_hdr suitesparse/klu.h
probe "SuperLU_MT"         "the 9 *_sps / *_slu examples"     ""                     have_hdr superlu_mt/slu_mt_ddefs.h
probe "SuperLU_DIST"       "examples/*/superludist"           "libsuperlu-dist-dev"  have_hdr superlu-dist/superlu_ddefs.h
probe "hypre"              "examples/*/{parhyp,C_parhyp,CXX_parhyp}" "libhypre-dev"  have_hdr hypre/HYPRE.h
probe "PETSc"              "examples/*/{petsc,C_petsc}"       "petsc-dev"            have_petsc
probe "Trilinos (Tpetra)"  "examples/ida/trilinos"            "libtrilinos-tpetra-dev" have_cmake_pkg Trilinos
probe "CUDA (nvcc)"        "examples/*/{cuda,mpicuda}"        "nvidia-cuda-toolkit"  have_nvcc
probe "Kokkos"             "examples/cvode/kokkos"            "libkokkos-dev"        have_cmake_pkg Kokkos
probe "MAGMA"              "examples/cvode/magma"             "libmagma-dev"         have_magma
probe "Ginkgo"             "examples/cvode/ginkgo"            ""                     have_hdr ginkgo/ginkgo.hpp
probe "RAJA"               "examples/*/raja"                  ""                     have_hdr RAJA/RAJA.hpp
probe "oneMKL (SYCL)"      "examples/cvode/{CXX_onemkl,CXX_sycl}" ""                 have_cmd icpx
probe "XBraid"             "examples/arkode/CXX_xbraid"       ""                     have_hdr braid.h

if [ $MD -eq 0 ]; then
  printf '%-22s %-8s %s\n' "COMPONENT" "STATUS" "APT PACKAGE"
  printf '%-22s %-8s %s\n' "----------------------" "--------" "-----------"
fi
printf '| component | status | what it unlocks | apt package |\n'
printf '|---|---|---|---|\n'
for r in "${rows[@]}"; do
  IFS='|' read -r label status ex apt <<<"$r"
  if [ -z "$apt" ]; then apt="_not packaged for Ubuntu_"; else apt="\`$apt\`"; fi
  if [ "$status" = "present" ]; then status="present"; else status="**MISSING**"; fi
  printf '| %s | %s | %s | %s |\n' "$label" "$status" "$ex" "$apt"
done
