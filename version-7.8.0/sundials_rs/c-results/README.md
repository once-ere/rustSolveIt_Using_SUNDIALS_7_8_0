# c-results — every upstream C example this toolchain could build

This directory records what the **unmodified upstream SUNDIALS 7.8.0
C examples** actually printed on this machine. It is raw evidence:
the `.stdout` files are the bytes the processes wrote, with nothing
filtered, rounded or edited.

"Every" is scoped, and the scope is large: 337 variants came out of 45 of the upstream tree's 68 example directories. The other 23 produced nothing, because a backend they need is missing or unusable here; every one is accounted for in
[`../requirements.md`](../requirements.md).

## Provenance

| item | value |
|---|---|
| generated | 2026-08-14 14:07:36 UTC |
| operating system | Ubuntu 26.04 LTS |
| kernel / platform | Linux-7.0.0-29-generic-x86_64-with-glibc2.43 |
| architecture | x86_64 |
| C library | ldd (Ubuntu GLIBC 2.43-2ubuntu2.3) 2.43 |
| C compiler | cc (Ubuntu 15.2.0-16ubuntu1) 15.2.0 |
| C++ compiler | c++ (Ubuntu 15.2.0-16ubuntu1) 15.2.0 |
| Fortran compiler | GNU Fortran (Ubuntu 15.2.0-16ubuntu1) 15.2.0 |
| CMake | cmake version 4.2.3 |
| rustc | rustc 1.96.1 (31fca3adb 2026-06-26) |
| cargo | cargo 1.96.1 (356927216 2026-06-26) |
| CPU cores | 24 |

The C sources are an unpacked SUNDIALS 7.8.0 tree, used read-only. On
the machine above it was `/home/youruser/Developer/sundials-7.8.0`, reached
through the `upstream-c` symlink; that path is recorded in every
`.meta` file and is provenance, not a dependency — point the symlink
at your own copy and the pipeline reproduces. The vendored `examples/`
tree is the same sources, and supplies the CMake tuples that decide
which command-line variants each example is run with.

## How to reproduce all of it

```bash
tools/c_build.sh          # configure + build, out of source, into build/c
tools/c_examples_run.sh   # run every binary, once per declared argv variant
python3 tools/make_reports.py
```

`tools/c_build.sh` prints which optional backends it was able to switch
on; anything it could not is listed in [`../requirements.md`](../requirements.md).

## Headline result

**337 (example, argv) variants were executed. 337 exited 0, and 336 also report a completed solve.**

| status | variants |
|---|---|
| OK | 337 |

## Layout of this directory

| path | contents |
|---|---|
| `index.tsv` | one row per variant: directory, example, argv, exit status, wall time, stdout size and SHA-256 |
| `raw/<dir>/<variant>.stdout` | exactly what the process printed to stdout |
| `raw/<dir>/<variant>.stderr` | exactly what it printed to stderr |
| `raw/<dir>/<variant>.meta` | the binary, the argv, the working directory, the exit code, the timing and the full SHA-256 |
| `by-solver/*.md` | the per-solver tables below |

A `<variant>` is the example name, plus `__` and the argv with spaces
turned into underscores when the example is declared with arguments.

## Checking any single row yourself

```bash
cat c-results/raw/cvode/serial/cvRoberts_dns.meta      # what was run
cat c-results/raw/cvode/serial/cvRoberts_dns.stdout    # what it printed
sha256sum c-results/raw/cvode/serial/cvRoberts_dns.stdout
```

The `.meta` file carries the full digest; `index.tsv` carries only its
**first 16 hex characters**, so compare against the `.meta` line or
against `sha256sum ... | cut -c1-16`.

## Run-to-run reproducibility

The whole pipeline has been executed four times on this machine, and
the captured `.stdout` files compared between runs with git — a byte
comparison, not a tolerance. The strongest single statement, and the
one anyone can re-check, is about the most recent re-run: it rebuilt
the C library and all 233 example binaries from source, re-ran every
variant on both sides, and **every capture in the compared set came
back byte-identical to the committed one** — 190 C, 199 Rust, 0 diffs.

```bash
tools/c_build.sh && tools/c_examples_run.sh && tools/rust_examples_run.sh
git status --porcelain c-results rust-results   # only .meta timings should move
```

The earlier runs are weaker evidence for part of the set: the four runs
were not runs of the same build, because KLU only became usable partway
through, so the eleven `*_klu` serial variants have fewer repetitions
behind them than the other 179.

| set | variants | reproduced byte for byte |
|---|---:|---|
| the six *serial* directories (the compared set) | 190 | **all of them** |
| every Rust example (`rust-results/`) | 199 | **all of them** |
| `*/C_openmp` and `*/F2003_openmp` | 11 | up to 6 differ between runs |
| `*/*parallel` (MPI) | 52 | 1 reorders between runs |

The 6 that move are OpenMP examples run with a thread count as argv: `ark_heat1D_omp 4`, `idaFoodWeb_kry_omp 4`, `idasFoodWeb_kry_omp 4`, `kinFoodWeb_kry_omp 4`, `idaHeat2D_kry_omp_f2003 4`, `idaHeat2D_kry_omp_f2003 8`. This is expected and is not a defect in anything: an OpenMP
reduction sums partial results in whatever order the threads finish, so
a dot product or a norm differs in its last bits from run to run, and
inside an iterative solver that changes the iteration counts. Compare
`kinFoodWeb_kry_omp 4`, which reported `nni = 7, nli = 229` on one run
and `nni = 10, nli = 378` on the next.

The MPI case is a different animal and worth separating, because it
looks alarming and is not: `kin_diagon_kry_f2003` runs under `mpirun
-np 4`, and between runs its 47 lines come out in a **different order**
with every number identical -- four ranks writing to one stream, not a
different answer. `sort`ing both captures makes them equal. The OpenMP
movers are the real nondeterminism: there the numbers themselves change.

None of these is in the compared set, so `differences/` is unaffected.
It is recorded here because a reader is entitled to know which numbers
in this directory are stable and which are not.

### `.stderr` moves too, and not because of the port

63 of the 337 runs currently carry an **hwloc** topology
warning on stderr — all of them MPI examples, inheriting a complaint from
OpenMPI about how it reads this machine's CPU layout. It appeared between
two otherwise identical pipeline runs, so a `git diff` of the captures
shows dozens of moved `.stderr` files and no moved `.stdout`.

Harmless, and checkably so: none is in the compared set,
[`../tools/compare_results.py`](../tools/compare_results.py) opens only
`.stdout`, and the runs still exit 0.

```bash
grep -rl hwloc c-results/raw --include='*.stderr' | wc -l
```

## Per-solver tables (serial examples — these are the ones with a Rust counterpart)

* [ARKODE — `arkode/C_serial`](by-solver/arkode_C_serial.md) — 78 variants
* [CVODE — `cvode/serial`](by-solver/cvode_serial.md) — 23 variants
* [CVODES — `cvodes/serial`](by-solver/cvodes_serial.md) — 36 variants
* [IDA — `ida/serial`](by-solver/ida_serial.md) — 13 variants
* [IDAS — `idas/serial`](by-solver/idas_serial.md) — 19 variants
* [KINSOL — `kinsol/serial`](by-solver/kinsol_serial.md) — 21 variants

## Runs that exited 0 but did not succeed

Exit status is not the whole story: 1 of the 337 runs
returned 0 while their own output reports a failed solve. None is in the
compared set, so `differences/` is unaffected, but a table of exit codes
alone would read as though everything worked.

| directory | variant | what it reports |
|---|---|---|
| `arkode/CXX_lapack` | `ark_heat2D_lsrk_domeigest` | ARKodeEvolve returned with flag = -99 — `lsrkStep_ComputeNewDomEig`: SUNDomEigEstimator_Estimate failed (arkode_lsrkstep.c:2898) |

## Other example families that were also built and run

These have no pure-Rust counterpart because the port translates only
the six **C** serial directories -- 63 of the 147 rows below are themselves serial, in C++ or Fortran, so parallelism is not
the reason. They do not appear in `differences/`, and are recorded
because the instruction was to build and execute *all* examples.

| directory | variants | all exited 0 |
|---|---|---|
| `arkode/CXX_lapack` | 1 | yes |
| `arkode/CXX_manyvector` | 1 | yes |
| `arkode/CXX_parallel` | 6 | yes |
| `arkode/CXX_parhyp` | 4 | yes |
| `arkode/CXX_serial` | 18 | yes |
| `arkode/C_klu` | 1 | yes |
| `arkode/C_manyvector` | 1 | yes |
| `arkode/C_openmp` | 2 | yes |
| `arkode/C_parallel` | 5 | yes |
| `arkode/C_parhyp` | 1 | yes |
| `arkode/F2003_custom` | 6 | yes |
| `arkode/F2003_parallel` | 6 | yes |
| `arkode/F2003_serial` | 19 | yes |
| `cvode/CXX_parallel` | 1 | yes |
| `cvode/CXX_parhyp` | 2 | yes |
| `cvode/CXX_serial` | 3 | yes |
| `cvode/C_mpimanyvector` | 1 | yes |
| `cvode/C_openmp` | 1 | yes |
| `cvode/F2003_parallel` | 3 | yes |
| `cvode/F2003_serial` | 12 | yes |
| `cvode/parallel` | 4 | yes |
| `cvode/parhyp` | 1 | yes |
| `cvodes/C_openmp` | 1 | yes |
| `cvodes/F2003_serial` | 3 | yes |
| `cvodes/parallel` | 9 | yes |
| `ida/C_openmp` | 2 | yes |
| `ida/F2003_openmp` | 2 | yes |
| `ida/F2003_parallel` | 1 | yes |
| `ida/F2003_serial` | 2 | yes |
| `ida/parallel` | 4 | yes |
| `idas/C_openmp` | 2 | yes |
| `idas/F2003_serial` | 2 | yes |
| `idas/parallel` | 8 | yes |
| `kinsol/CXX_parallel` | 2 | yes |
| `kinsol/CXX_parhyp` | 2 | yes |
| `kinsol/C_openmp` | 1 | yes |
| `kinsol/F2003_parallel` | 1 | yes |
| `kinsol/F2003_serial` | 4 | yes |
| `kinsol/parallel` | 2 | yes |

## Which optional backends were reachable

Read off the run itself: a backend counts as present here when the
examples that need it produced rows in `index.tsv`. See
[`../requirements.md`](../requirements.md) for the probe results and the
exact `apt` command.

| backend | example variants that ran | on this machine |
|---|---:|---|
| KLU (SuiteSparse) | 15 | **present** |
| SuperLU_MT | 0 | absent |
| MPI | 63 | **present** |
| hypre | 10 | **present** |
| PETSc | 0 | absent |
| LAPACK | 5 | **present** |
| CUDA / RAJA / Kokkos / MAGMA / Ginkgo / SYCL / XBraid | 0 | absent |

The absent ones remove their example families from this run entirely --
there is no output on either side, so nothing is being hidden by their
absence.

