# rust-results — every ported example, built and executed here

This directory records what the **pure-Rust translations** of the
upstream serial examples printed on this machine. The rows of the
provenance table that matter here are the OS, the architecture and
`rustc`/`cargo`: these binaries link no C toolchain and call the host
libm for nothing. The C compiler rows are carried for comparison with
`c-results/`.

Same rules as `c-results/`: the `.stdout` files are raw process
output -- for the
190 variants that ran. The 9 `NOT_PORTED` ones have empty placeholder
files, because no binary exists to run.

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

## How to reproduce all of it

```bash
cargo build --release --workspace --examples
tools/rust_examples_run.sh
python3 tools/make_reports.py
```

No network access and no package installation is involved: the
workspace has **zero external crates**, so `cargo build` compiles only
the seven crates in `crates/`. Nothing was added to
[`../requirements.md`](../requirements.md) on the Rust side because
nothing needed to be.

## Headline result

**199 (example, argv) variants, 190 exited 0, 9 have no Rust counterpart.**

| status | variants |
|---|---|
| OK | 190 |
| NOT_PORTED | 9 |

`NOT_PORTED` marks the 9 `*_sps` / `*_slu` examples, and only those.
They need SuperLU_MT, a third-party sparse-direct **C** library that a
port forbidding `unsafe`, FFI and external crates cannot call --  and
that is not in the Ubuntu archive at any version, so the C side cannot
build them either. **No comparison** is lost by their absence -- there
is no output on either side to compare. Whether the SuperLU_MT code
path itself would have exposed anything is not measured here and is
not claimed.

The 11 `*_klu` examples in these six serial directories *are* ported.
(15 `*_klu` variants exist across the whole C build; the 4 outside
these directories are out of the port's scope.) KLU itself is fully
available on this machine and the C side uses it -- it is unreachable
*from Rust*, which forbids FFI, not unreachable like SuperLU_MT. So
they run on the independent
pure-Rust sparse LU in `crates/sundials_core/src/sundials_sparse_lu.rs`
instead. Four of them still match the C byte for byte.

See [`../requirements.md`](../requirements.md) §1 and §4 for SuperLU_MT,
§6 for the KLU substitution.

## What makes these runs reproducible

Unlike the C binaries, these do not call the host C library for any
elementary function. `exp`, `log`, `pow`, `expm1`, `log1p`, `sin`,
`cos`, `atan`, `asin`, `acos`, `sinh`, `cosh` and `acosh` are all
implemented in `crates/sundials_core/src/sundials_libm.rs`, so the
numbers below do not move when the host glibc moves. See
[`../LIBM.md`](../LIBM.md).

## Layout of this directory

| path | contents |
|---|---|
| `index.tsv` | one row per variant |
| `raw/<dir>/<variant>.stdout` | exactly what the process printed |
| `raw/<dir>/<variant>.stderr` | stderr |
| `raw/<dir>/<variant>.meta` | binary, argv, cwd, exit code, timing, SHA-256 |
| `by-solver/*.md` | the per-solver tables below |

## Per-solver tables

* [ARKODE — `arkode_rs`](by-solver/arkode_C_serial.md) — 78 variants
* [CVODE — `cvode_rs`](by-solver/cvode_serial.md) — 24 variants
* [CVODES — `cvodes_rs`](by-solver/cvodes_serial.md) — 39 variants
* [IDA — `ida_rs`](by-solver/ida_serial.md) — 14 variants
* [IDAS — `idas_rs`](by-solver/idas_serial.md) — 22 variants
* [KINSOL — `kinsol_rs`](by-solver/kinsol_serial.md) — 22 variants

