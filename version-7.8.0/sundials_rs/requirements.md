# requirements.md — what this machine has, and what it is missing

**Machine of record.** Ubuntu 26.04 LTS ("Resolute Raccoon"), x86-64,
glibc 2.43, gcc 15.2.0, cmake 4.2.3, rustc/cargo 1.96.1, 24 cores,
NVIDIA GeForce RTX 5090 Laptop GPU (driver 595.84, CUDA 13.1).
Everything in `c-results/`, `rust-results/` and `differences/` was produced
on this machine.

Probe this machine yourself — it needs no root and installs nothing:

```bash
tools/c_requirements.sh
```

It does **not** regenerate §1: it prints four columns rather than five, has
no pthreads row, and reports MAGMA as MISSING because its `have_lib` helper
searches only `/usr/lib/x86_64-linux-gnu` while Ubuntu's MAGMA lives in
`/usr/lib`. §1 below is maintained by hand against the build log, and the
two disagree on that row.

---

## 1. Status table (probed, not assumed)

"Present" means the headers and libraries are installed. Whether SUNDIALS
can actually *use* them is a separate question, answered in §3.

| component | installed | usable by the build | what it unlocks | apt package |
|---|---|---|---|---|
| C compiler (cc) | yes | yes | everything | `gcc` |
| C++ compiler | yes | yes | `examples/*/CXX_*` | `g++` |
| Fortran compiler | yes | yes | `examples/*/F2003_*` | `gfortran` |
| CMake | yes | yes | the whole C build | `cmake` |
| OpenMP runtime | yes | yes | `examples/*/C_openmp` | ships with `gcc` |
| pthreads | yes | yes | the pthreads NVector | ships with glibc |
| libquadmath | yes | yes | the ulp figures in `LIBM.md` | ships with `gcc` |
| BLAS / LAPACK | yes | yes | the `*_dnsL` / `*_bndL` examples | `libblas-dev` `liblapack-dev` |
| MPI (OpenMPI) | yes | yes | `parallel`, `C_parallel`, `CXX_parallel`, `F2003_parallel` | `libopenmpi-dev` |
| KLU (SuiteSparse) | yes | yes | 12 `*_klu` example programs, 15 argv variants | `libsuitesparse-dev` |
| hypre | yes | yes | `parhyp`, `C_parhyp`, `CXX_parhyp` | `libhypre-dev` |
| **PETSc** | yes | **no** — §3.1 | `petsc`, `C_petsc` | `petsc-dev` |
| **SuperLU_DIST** | yes | **no** — §3.2 | `superludist`, `CXX_superludist` | `libsuperlu-dist-dev` |
| **Kokkos** | yes | **no** — §3.3 | `cvode/kokkos` | `libkokkos-dev` |
| **Trilinos (Tpetra)** | yes | **no** — §3.4 | `ida/trilinos` | `libtrilinos-tpetra-dev` + `libtrilinos-teuchos-dev` |
| **CUDA 13.1 + RTX 5090** | yes | **no** — §3.5 | `cuda`, `mpicuda` | already installed |
| **MAGMA** | yes | **no** — §3.6 | `cvode/magma` | `libmagma-dev` |
| **SuperLU_MT** | **no** | no | 10 `*_sps` / `*_slu` example programs | _not packaged for Ubuntu_ |
| **Ginkgo** | **no** | no | `cvode/ginkgo` | _not packaged for Ubuntu_ |
| **RAJA** | **no** | no | `*/raja`, `mpiraja` | _not packaged for Ubuntu_ |
| **oneMKL / SYCL** | **no** | no | `CXX_onemkl`, `CXX_sycl` | _not packaged for Ubuntu_ |
| **XBraid** | **no** | no | `arkode/CXX_xbraid` | _not packaged for Ubuntu_ |
| **HIP / ROCm** | **no** | no | `cvode/hip` | _not installed_ |
| **OpenMP device offload** | n/a | no | `*/C_openmpdev` | needs an offload-capable toolchain |

23 of the upstream tree's 68 example directories produced no rows at all,
holding 30 example source files between them. Every one is accounted for by
a row above; `tools/cross_gate.py` is not involved here, but the list is
reproducible:

```bash
comm -23 <(ls -d upstream-c/examples/*/*/ | sed 's|.*examples/||;s|/$||' | sort) \
         <(awk -F'\t' 'NR>1{print $1}' c-results/index.tsv | sort -u)
```

## 2. What was installed for this work

```bash
sudo apt install libopenmpi-dev libsuitesparse-dev libhypre-dev petsc-dev libsuperlu-dist-dev libtrilinos-tpetra-dev libkokkos-dev libmagma-dev
```

Effect, measured: the C example build went from **164 binaries to 233**, and
the executed variant set from 259 to the number in
[`c-results/README.md`](c-results/README.md). MPI, KLU and hypre all became
usable; the other five — PETSc, SuperLU_DIST, Trilinos, Kokkos and MAGMA —
did not, for the reasons in §3. (Eight packages were installed, mapping to
eight components; three worked.)

`libtrilinos-teuchos-dev` was installed afterwards, on the evidence in
§3.4. It removed the error it was diagnosed for, but did not make Trilinos
usable: the failure simply moved one layer down, into the same broken
Kokkos package described in §3.3. Both are now blocked by that one Ubuntu
bug.

## 3. Installed but not usable, with the exact reason

`tools/c_build.sh` enables optional backends as a **ladder**: it tries the
full set, then drops entries one at a time until CMake both configures and
builds. The level that succeeded is printed and logged. The set that works
on this machine is **MPI + LAPACK + KLU + hypre** (`logs/c_build.log:7728`).

**How far each diagnosis below is evidence.** A ladder attributes a failure
to whatever it dropped next, which is not the same as knowing what failed.
Three of the six diagnoses quote text that is in the committed log, and
three do not:

| § | backend | quoted text in `logs/c_build.log`? |
|---|---|---|
| 3.1 | PETSc | **no** — inferred; that level failed on the Kokkos error (log:1040) |
| 3.2 | SuperLU_DIST | yes, log:1689 |
| 3.3 | Kokkos | yes, log:1483 and 5 more |
| 3.4 | Trilinos | yes (Teuchos, then Kokkos again) |
| 3.5 | CUDA | yes, log:290 |
| 3.6 | MAGMA | **no** — inferred; that level failed on the CUDA error (log:213) |

No isolation run is committed for any of them, so 3.1 and 3.6 are
reasoned diagnoses rather than captured transcripts, and are marked as such
below. Anyone wanting them upgraded to evidence should configure those two
backends alone and commit the logs.

### 3.1 PETSc — index-width mismatch with SUNDIALS

> **Diagnosis, not a transcript.** The recorded PETSc level failed on the
> Kokkos error of §3.3 before any PETSc source was compiled, so the block
> below is what this mismatch produces, read off
> `src/sunnonlinsol/petscsnes/sunnonlinsol_petscsnes.c:352` and Ubuntu's
> `petscconf.h` — it is not in `logs/c_build.log`.

```
src/sunnonlinsol/petscsnes/sunnonlinsol_petscsnes.c:352:54: error:
  passing argument 2 of 'SNESGetIterationNumber' from incompatible pointer type
  expected 'PetscInt *' {aka 'int *'} but argument is of type 'sunindextype *' {aka 'long int *'}
```

SUNDIALS defaults to a 64-bit `sunindextype`; Ubuntu's PETSc 3.24 is built
with a 32-bit `PetscInt`. gcc 15 treats the mismatch as an error rather
than a warning. Not a packaging fault on either side — the two were
configured with different index widths.

*Workaround, not applied here:* rebuild with `-DSUNDIALS_INDEX_SIZE=32`.
That changes the index type of the whole library, so its example outputs
would no longer be the same configuration the Rust port is compared
against. It is left off deliberately.

### 3.2 SuperLU_DIST — upstream find-module cannot read Ubuntu's config header

```
Could NOT find SUPERLUDIST (missing: SUPERLUDIST_INDEX_SIZE)
  (found suitable version "9.2.1", minimum required is "7.0.0")
  cmake/tpl/FindSUPERLUDIST.cmake:157
```

The library is found and its version accepted, but
`FindSUPERLUDIST.cmake` derives `SUPERLUDIST_INDEX_SIZE` by grepping
`superlu_dist_config.h` for `#define XSDK_INDEX_SIZE 64`. Ubuntu's header
instead contains `/* #undef XSDK_INDEX_SIZE */` followed by an `#if
defined(...)` guard, and the module leaves the variable unset. Passing
`-DSUPERLUDIST_INDEX_SIZE=32` on the command line does not help, because
the module re-`set(... CACHE ... FORCE)`s it. This is a SUNDIALS 7.8.0
find-module limitation.

### 3.3 Kokkos — the Ubuntu package's CMake config is broken

```
The imported target "Kokkos::kokkosalgorithms" references the file
   "/usr/lib/x86_64-linux-gnu/libkokkosalgorithms.a"
but this file does not exist.
```

`libkokkos-dev` 5.0.2-2 installs `libkokkoscore.so` and
`libkokkoscontainers.so` and nothing else, yet its CMake config declares
five imported targets, and **two of them are `STATIC IMPORTED`** —
`kokkosalgorithms` (`KokkosTargets.cmake:89`) and `kokkossimd` (`:97`) —
whose `.a` files exist nowhere on the system. The other three are
`kokkoscore` and `kokkoscontainers` (`SHARED`, and their `.so` files do
exist) and the `kokkos` umbrella (`INTERFACE`). In Kokkos 5.x
those components are header-only (`/usr/include/kokkos/Kokkos_StdAlgorithms.hpp`,
`/usr/include/kokkos/std_algorithms/`), so the archives are not merely
missing — they should not exist at all, and the packaged config is simply
wrong. Nothing on the SUNDIALS side can work around it, and it blocks
Trilinos too (§3.4).

### 3.4 Trilinos — a missing dependency, and then the Kokkos bug

Trilinos failed twice, for two different reasons. The first was a genuine
missing dependency:

```
CMake Error at .../cmake/TpetraCore/TpetraCoreConfig.cmake:202 (include):
  include could not find requested file:
    /usr/lib/x86_64-linux-gnu/cmake/TpetraCore/../Teuchos/TeuchosConfig.cmake
```

`libtrilinos-tpetra-dev` does not depend on the Teuchos development
package, but its CMake config includes it. Installing
`libtrilinos-teuchos-dev` fixed that error exactly as predicted — and
exposed the next one:

```
CMake Error at .../cmake/Kokkos/KokkosTargets.cmake:130 (message):
  The imported target "Kokkos::kokkosalgorithms" references the file
     "/usr/lib/x86_64-linux-gnu/libkokkosalgorithms.a"
  but this file does not exist.
```

That is §3.3 again. `TrilinosConfig.cmake` lists `Kokkos` among its
required components, and there is exactly one Kokkos CMake package on the
system — the broken one. `libtrilinos-kokkos-dev` is **not installed here and was not tried**, so
this is an expectation rather than a result: Trilinos 16.1 resolves the
component through `/usr/lib/x86_64-linux-gnu/cmake/Kokkos`, which is the
broken config, and the runtime package that owns that directory
(`libtrilinos-kokkos-16.1`) is already present.

**So Trilinos is not fixable by installing packages on this machine.** It
becomes usable only when the `libkokkos-dev` CMake config stops declaring
static targets for components Kokkos 5.x ships as header-only.

### 3.5 CUDA — nvcc 13.1 headers clash with glibc 2.43

```
/usr/include/x86_64-linux-gnu/bits/mathcalls.h(206): error: exception
  specification is incompatible with that of previous function "rsqrt"
  (declared at line 629 of .../crt/math_functions.h)
```

The GPU and driver are fine — this fails during CMake's compiler
identification, before any SUNDIALS code is reached. glibc **2.42** added
`rsqrt` to `<math.h>` (its NEWS lists it under "Power and absolute-value
functions: compoundn, pown, powr, rootn, rsqrt"; 2.43 only added AArch64
vector variants); CUDA 13.1's `crt/math_functions.h` declares it with
an incompatible exception specification. It needs a CUDA release that
knows about the newer glibc; no build flag avoids it.

### 3.6 MAGMA — blocked by 3.5

> **Diagnosis, not a transcript.** MAGMA was the first entry the ladder
> dropped, and its level failed on the CUDA compiler-identification error of
> §3.5 (`logs/c_build.log:213`) — SUNDIALS' own MAGMA logic was never
> reached. The message below is what SUNDIALS emits once configuration gets
> that far; it appears in no log in this repository.

```
SUNDIALS_MAGMA_BACKENDS includes CUDA but CUDA is not enabled.
```

MAGMA is installed (in `/usr/lib`, not the multiarch directory) and would
be usable as soon as CUDA is.

## 4. Cannot be fixed with apt at all

| component | why | consequence |
|---|---|---|
| **SuperLU_MT** | not in the Ubuntu archive at any version; upstream ships source only | 10 `*_sps` / `*_slu` example programs cannot be built — the 9 in the serial directories plus `arkode/C_superlu-mt/ark_brusselator1D_FEM_slu`. The 9 serial ones are exactly the 9 variants the Rust port does not translate, so no comparison is lost. |
| **Ginkgo, RAJA, XBraid, oneMKL** | not in the Ubuntu archive | 11 GPU / parallel-framework example programs cannot be built, across `cvode/ginkgo`, `cvode/raja`, `ida/raja`, `ida/mpiraja`, `arkode/CXX_xbraid`, `cvode/CXX_onemkl` and `cvode/CXX_sycl`. None has a serial Rust counterpart. |

## 4a. Cross-architecture verification — done, under emulation

Every measurement in this repository is **x86-64**. The reference gate has
been run on six hosts and two libcs (`evidence/purerust-libm-gate/` upstream),
but all six are x86-64 containers on an x86-64 machine. **arm64 is the one
platform claim still resting on argument rather than measurement**, and the
argument is decent — the crate tree is `std`-only with no `cfg(target_arch)`,
and the pure-Rust libm is built on `sqrt`, `mul_add` and integer arithmetic,
all exactly specified by IEEE-754 — but an argument is not a gate run.

Running it here needs user-mode emulation, which is not installed:

| component | installed | candidate | what it unlocks |
|---|---|---|---|
| `qemu-user-binfmt` | **now yes** | 1:10.2.1+ds-1ubuntu3.2 | `podman run --platform linux/arm64`, hence `tools/gate_in_container.sh` on aarch64 |
| `qemu-user` | **no** | 1:10.2.1+ds-1ubuntu3.2 | pulled in as a dependency |
| `binfmt-support` | **no** | 2.2.2-8 | registers the aarch64 handler |
| `qemu-user-static` | **no** | _(none on Ubuntu 26.04)_ | the usual package elsewhere; use `qemu-user-binfmt` here |

```bash
sudo apt install qemu-user-binfmt
# then, upstream:
tools/gate_in_container.sh --platform linux/arm64 debian:13
```

**Resolved on 2026-08-14.** `qemu-user-binfmt 1:10.2.1+ds-1ubuntu3.2` is
installed, `/proc/sys/fs/binfmt_misc/qemu-aarch64` is registered, and the gate
has been run: **Debian 13 on aarch64 gives 145 / 34 / 20 with a DIFF list
byte-identical to the same image on x86-64.** The log is
`evidence/purerust-libm-gate/gate-debian-13-arm64.txt` upstream, headed
`aarch64 [EMULATED]`. It is QEMU user-mode on an x86-64 host, not arm64
silicon — see the caveat at the end of this section, which still applies.
Separately, `cargo check --target aarch64-unknown-linux-{gnu,musl} --workspace
--all-targets` is clean: 0 errors, 0 warnings, 7 crates, 119 example targets.

The table below is kept as the record of what was missing.
`gate_in_container.sh` takes `--platform` (or `$GATE_PLATFORM`), checks
`/proc/sys/fs/binfmt_misc/qemu-<arch>` before pulling anything and prints the
apt line if it is absent, tags the log with the architecture so an emulated
run cannot overwrite a native one, and records `uname -m` plus an
`[EMULATED]` marker in each log header. Two install attempts on 2026-08-12
left the package still absent (`dpkg-query` reports not-installed, no
`/usr/bin/qemu-aarch64`, no handler registered), so the run was deferred
rather than faked.

Current state, for anyone checking: `podman pull --platform linux/arm64` works
and the image is on disk, but running it fails with `exec container process
/bin/uname: Exec format error`, and `/proc/sys/fs/binfmt_misc/` registers only
`llvm-21-runtime` and `python3.14` — no `qemu-aarch64`.

**A caveat worth reading before treating an emulated run as settled.** QEMU
user-mode is not an arm64 CPU. For the operations this port depends on it
should be bit-equivalent — IEEE-754 pins `sqrt` and fused multiply-add
exactly, and QEMU implements them with softfloat to that specification — so a
green emulated gate would be real evidence. It would still be weaker than a
run on hardware, and should be labelled as emulated wherever it is reported.
The failure mode it cannot rule out is a genuine aarch64 code-generation
difference that QEMU happens to reproduce faithfully in the same wrong way,
which is unlikely but not impossible.

## 5. Rust-side requirements

The Rust workspace deliberately has **no dependencies at all** — no
external crates, no build script, no system library beyond what `std`
itself links:

```bash
cargo build --workspace        # nothing is downloaded
cargo test  --workspace --lib
```

`cargo` was therefore never used to install a package, and no package name
needed to be added to this file on the Rust side. That is a design
constraint of the port (`CLAUDE.md` hard rule 2), not an accident.

Optional, and only for regenerating documentation rather than for building:

| tool | used by | needed? |
|---|---|---|
| `python3` (stdlib only) | `tools/gen_libm_constants.py`, `tools/compare_results.py`, `tools/make_reports.py` | for the libm tables and the report generation |
| `python3` + `mpmath` | independent cross-check of the libm tables | optional; `pip install mpmath` |
| `libquadmath` (ships with gcc) | `tools/libm_oracle.c` | without it the differential still runs, but reports agreement only, not ulp accuracy |

## 6. The KLU gap, and what closed it

This section used to record a coverage gap: the `*_klu` examples built on
the C side and had no pure-Rust counterpart, because KLU is a third-party
sparse-direct C library and this port forbids FFI. That gap is closed.

`crates/sundials_core/src/sundials_sparse_lu.rs` implements a left-looking
sparse LU (Gilbert & Peierls) with KLU's default threshold partial pivoting,
under a faithful translation of SUNDIALS' own BSD-3 `sunlinsol_klu.c`.
**All 11 `*_klu` examples in the six serial directories are ported and ran
(11/11 exit 0); 4 are byte-identical to the C.**

What replaced the gap is not nothing, and it is worth stating plainly: a
**second substitution of third-party numerics**, alongside the libm. The
libm substitution has a control build (`--features host-libm`) that
attributes its divergences; this one cannot have, because there is no KLU to
switch back to. The 7 `*_klu` variants that differ numerically are therefore
attributed by direct verification of the replacement solver rather than by
A/B — see [`differences/ATTRIBUTION.md`](differences/ATTRIBUTION.md).

The 15 `*_klu` argv variants outside those six directories (`arkode/C_klu`
and the three `*_klu_f2003`) build and run on the C side and are **not**
ported: the port covers only the C serial directories.
