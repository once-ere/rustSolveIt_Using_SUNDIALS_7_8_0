# STATUS — inherited macOS/arm64 status record

Project: **SUNDIALS_7_8_Rust_port_for_Linux** — but this file is the sibling
macOS port's status record, kept because it documents how the translation
was finished phase by phase.

> **For this repository's status read [`current_status.md`](current_status.md).**
> Linux / x86-64 / glibc results: `cargo build --workspace` 0 warnings,
> `cargo test --workspace --lib` 25 passed, deterministic `pow` 0 mismatches
> over 25.9M inputs against the native glibc `pow`, and the example gate at
> **153 IDENTICAL / 26 reference-side / 20 excluded**, 0 port defects.

The port is finished. All eight phases are done, every crate is verified
against the upstream reference outputs, and the cumulative gate passes.
For the public guide read `sundials.md`; for per-variant evidence
`VERIFICATION.md` (Part A is Linux, Part B the inherited macOS record); for
per-file status `PROGRESS.md`.

> ## ⚠ Platform scope of the sections below
>
> **"Verified" in the rest of this document means verified on macOS running
> on Apple Silicon (arm64), against Apple's libm, with the pristine upstream
> C comparison binaries built by Apple clang.** The table below, the
> cold-tree bullets and the 52 documented divergences are results on that
> platform. On Linux/glibc/x86-64 the corresponding tally is 153 / 26 / 20 —
> 26 of those 52 are byte-identical here.
>
> The sources are portable — `std` only, no `unsafe`, no FFI, no
> `cfg(target_os)`/`cfg(target_arch)` — and build warning-free and pass all 25
> unit tests elsewhere. Only the **output** claims are platform-bound,
> because `sin`, `cos`, `exp`, `ln` and the inverse/hyperbolic functions come
> from the host libm; `pow` alone was made host-independent. `README.md`
> § "Platform scope" and `sundials.md`
> §9 give the full statement and what a port to another platform would have
> to redo.

## Final state

Counts in the last three columns are **`(example, argv)` variants**, not
example programs — one program can contribute several variants (`kinsol_rs`
has 9 programs but 22 variants). "byte-identical", "documented" and "excluded"
are macOS/arm64 results.

| crate | modules | example programs | variants: byte-identical | documented | excluded |
|---|---:|---:|---:|---:|---:|
| `sundials_core` | 51 | — | — | — | — |
| `cvode_rs` | 12 | 18 | 12 | 9 | 3 |
| `cvodes_rs` | 16 | 26 | 23 | 10 | 6 |
| `kinsol_rs` | 8 | 9 | 19 | 1 | 2 |
| `ida_rs` | 8 | 8 | 10 | 1 | 3 |
| `idas_rs` | 12 | 13 | 12 | 4 | 6 |
| `arkode_rs` | 34 | 34 | 51 | 27 | 0 |
| **total** | **141** | **108** | **127** | **52** | **20** |

Verified first-hand on 2026-08-09 on **macOS / Apple Silicon (arm64)** from a
**cold tree** (`cargo clean` first, so nothing below is a cached result):

* `cargo build --workspace` → all seven crates recompiled, **zero warnings**.
* `cargo test --workspace --lib` → **25 passed, 0 failed**.
* `tools/verify_examples.sh all` → **127 IDENTICAL / 52 documented /
  20 excluded** over all **199** variants. Zero FAIL, zero NO-REF, zero
  NO-BINARY, and byte-for-byte identical to the previous gate run — no
  variant changed status.
* Invariants re-checked by grep over `crates/*/src` and `crates/*/examples`:
  **0** `unsafe`, **0** `f64::powf`/`powi` (every power routes through the
  deterministic `SUNRpowerR`), **0** `todo!`/`unimplemented!`, **0** uses of
  Rust `{:e}`, and `Cargo.lock` holding exactly the 7 workspace packages —
  no external crates. `PROGRESS.md` has **0** remaining `todo` entries.

Cross-platform sanity check, same date, **Windows 11 x86-64**:
`cargo build --workspace` warning-free and `cargo test --workspace --lib`
25 passed / 0 failed. `tools/verify_examples.sh` was **not** run there — it
needs POSIX `bash` and the upstream C tree as its parent, and its verdicts
would in any case be Microsoft-CRT verdicts, not the ones recorded here.

## What the 52 documented divergences are

None is a port defect. Each is a case where the shipped `.out` cannot be
reproduced by its own C source on this machine — macOS/arm64, Apple libm;
the classification is a diagnosis about *this* host's libm and does not
transfer — established by building the
pristine upstream C locally (CMake Release, clang, `-O3 -DNDEBUG
-ffp-contract=off`, logging level 2, error checks off, profiling off,
monitoring on, serial) and comparing against that. Three classes:

* **`ref-libm`** — the reference embeds the generating host's glibc
  `sin`/`exp`/`pow` rounding inside the integration feedback loop. A one-ulp
  difference forks the step-size trajectory. Several `.out` files in one
  family require *mutually incompatible* libm versions, so no single build
  can match them all.
* **stale upstream reference** — the `.out` predates its own source: the
  `SUN_TABLE_WIDTH` 28→29 change (whitespace-only diffs across a statistics
  block), a changed format string, or regeneration on a new CI host.
* **LAPACK→native** — the `*L` examples run the native dense/band solvers, so
  factorisation arithmetic differs in the last digit.

## Notable defects the byte-identity gate caught

The gate paid for itself; these were all found by output comparison, not by
review:

1. **Deterministic `pow`, unfused FMA** (`sundials_math.rs`) — the final
   `scale + scale * tmp` must be a single fused multiply-add, as in the glibc
   build the references came from. Unfused it rounds the wrong way near a
   midpoint, forking `ark_robertson` into a 228-line diff. Fusing it turned
   eight ARKODE variants identical.
2. **Platform `pow` at all** — the original `f64::powf` differs from glibc by
   one ulp on rare inputs inside the step-size heuristics; replacing it with
   the ported ARM/musl algorithm fixed three examples at once.
3. **Newton solver retry loop** — an initial-residual failure fell into the
   jbad-retry block instead of breaking out, spinning forever on a recoverable
   RHS flag (`cvRoberts_dns_negsol` hung).
4. **IDAS sensitivity `user_data`** — the "`None` means pass the integrator's
   `user_data`" fallback was never implemented at six call sites, so
   user-supplied sensitivity residuals got `None` and panicked.
5. **CVODES `cv_p`** — the sensitivity parameter array was an owned copy, so
   internal difference-quotient perturbations never reached the user's RHS.
   Now shared as `Rc<RefCell<Vec<sunrealtype>>>`, with a regression test whose
   negative control reproduces the original defect.

## Known limitations

1. **Unexercised code.** Every `*_bbdpre` module is compile-only — BBD
   preconditioning is MPI-only upstream, so no serial reference example can
   regression-test it. Same for the excluded KLU/SuperLU paths.
2. **Adjoint steppers are compiler-checked only.** `ERKStepCreateAdjointStepper`
   and its cluster are translated and build clean, but no serial reference
   example exercises them, so line-by-line reading is the only check they have
   had. Their `user_data` does not alias the forward memory (deviation class 6)
   — a caller porting from C must call `SUNAdjointStepper_SetUserData` itself.
3. **Accepted deviations.** Thirteen numbered classes in `ARCHITECTURE.md`,
   each verified unobservable on any path a valid serial example takes.

4. **One platform.** Everything above is macOS/Apple-Silicon evidence. The
   port has never been run against the reference outputs on Linux, on Windows
   or on x86-64, and it is not expected to be byte-identical there: the host
   libm supplies `sin`, `cos`, `asin`, `acos`, `atan`, `sinh`, `cosh`,
   `acosh`, `exp` and `ln`, and one ulp is enough to fork a step-size
   trajectory. (`sqrt`, `mul_add`, `ceil`, `round`, `abs` and `copysign` are
   IEEE-754 specified and portable; `pow` was deliberately taken off the host
   libm.) `sundials.md` §9 lists what a port to another platform has to redo.

## Rules that still bind future work

* Byte-identical stdout against the upstream `.out`, noise-filtered
  symmetrically, is the only pass condition (`tools/verify_examples.sh`), and
  it is a **macOS/arm64** pass condition — never generalise a verdict to
  another platform, and never add conditional compilation to make one tree
  cover two.
* When a shipped `.out` cannot be reproduced, the fallback bar is byte-identity
  against a locally built pristine upstream C binary — never tune an example to
  match a reference.
* Zero `unsafe`, zero FFI, zero external crates, zero build warnings.
* Once a crate's examples verify green they stay green; the cumulative gate is
  `tools/verify_examples.sh all`.

## Documents

| file | what it holds |
|---|---|
| `sundials.md` | public guide: crate map, worked example, API conventions |
| `VERIFICATION.md` | per-variant results and the evidence for every exception |
| `PROGRESS.md` | per-file port status |
| `ARCHITECTURE.md` | cross-module contracts and the 13 accepted deviation classes |
| `POW_FMA_EXACTNESS.md` | whether the deterministic `pow` is bit-exact with the reference libm — the FMA-contraction investigation, the 20M-input differential measurements, and the bounded limits of that claim |
