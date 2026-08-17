# SUNDIALS_7_8_Rust_port_for_Linux — workspace rules

Pure-Rust port of SUNDIALS 7.8.0. The upstream C tree is the parent
directory (`../src/`, `../include/`, `../examples/`) and is **read-only**.
This workspace is its own git repo; git is the undo mechanism.

Read `current_status.md` first — it is the resume anchor.

## Target platform — binding on every rule below

**This port is scoped to Linux on Intel/AMD x86-64 with glibc**, measured on
Ubuntu 24.04 / glibc 2.39 / gcc 13.3.0 / rustc 1.93.1. The Rust sources are
portable (`std` only, no `cfg(target_os)`/`cfg(target_arch)`) and build and
unit-test anywhere, but every *numerical* claim — byte-identical output, the
199-variant gate, each per-variant classification in `VERIFICATION.md`, and
the `pow` differential — is a glibc-on-x86-64 result.

**Verified coverage is glibc 2.36 through 2.41** (Debian 12, Ubuntu 24.04,
Debian 13, Fedora 41): the full gate was re-run natively in those
containers and gives the identical 153 / 26 / 20 variant set. On **Arch
(glibc 2.44) three more variants diverge** — `ark_analytic_lsrk_domeigest`
(x2) and `ark_analytic_lsrk_varjac` — because 2.44 changed `sinh`, `cosh`
and `acosh`, which the library calls from exactly one place,
`arkode_lsrkstep.rs:87`. Do **not** widen the claim to "any glibc": that
was asserted once, and `tools/glibc_sweep.sh` disproved it. It does not
carry to musl, to arm64, or to Windows.

Why this platform is the favourable one: the upstream reference `.out` files
were generated on a glibc host, and `sin`, `cos`, `asin`, `acos`, `atan`,
`sinh`, `cosh`, `acosh`, `exp` and `ln` resolve to the host libm through
`f64`'s unspecified-precision methods — so here they land on the very
implementation the references came from. `pow` is additionally made
host-independent (the ported ARM optimized-routines/musl algorithm in
`sundials_math.rs`, which *is* glibc >= 2.28's `e_pow.c`) and is measured
bit-exact against the native glibc `pow` by `tools/pow_differential.sh`.
`sqrt`, `mul_add`, `ceil`, `round`, `abs` and `copysign` are IEEE-754
specified and portable — do not list them as host-dependent.

Any statement added to any document in this repo that asserts a verification
result must carry that scope explicitly. Ports for other platforms are
separate repositories (see the sibling
`SUNDIALS_7_8_Rust_port_for_AppleSilicon_macos`, from which this workspace's
crate tree is inherited unchanged), never conditional compilation inside
this one.

## Hard rules

1. **Fidelity first.** Line-by-line faithful translation: control flow,
   constants, tolerances, heuristics, error/return codes, and argument
   lists (names, order, meaning) match the parent C function exactly.
   Preserve arithmetic order — acceptance is byte-identical printed output.
2. Zero `unsafe`, zero FFI, zero external crates (std only), zero warnings
   from `cargo build --workspace`.
3. Never stub a missing symbol — its definition is under `../src/` or
   `../include/`; port it into `sundials_core`.
4. Public API keeps exact C names and return-flag conventions
   (`CV_SUCCESS = 0`; negative = fatal, positive = recoverable). Crate
   roots carry `#![allow(non_snake_case, non_camel_case_types,
   non_upper_case_globals)]`.
5. All float output goes through
   `sundials_core::sundials_utils::{fmt_e, fmt_f, fmt_g}` — never `{:e}`.
6. C buffer aliasing (e.g. CVODE `cv_y` / user `yout`): copy back at
   **every** return path, including early-error and rootfinding exits.
   All of CVODE(S), IDA(S), ARKODE do this.
7. Once a crate's examples verify green they stay green — the cumulative
   regression gate runs `tools/verify_examples.sh` for all crates ported
   so far at every phase gate.

## Module layout

- Module = C file base name + `.rs` (`cvodes_nls_stg1.c` →
  `crates/cvodes_rs/src/cvodes_nls_stg1.rs`; `arkode_impl.h` →
  `arkode_impl.rs`). Public `include/` headers fold into the matching
  module.
- Solver crates re-export every shared `sundials_core` module at root and
  provide a flat prelude so examples can `use cvode_rs::*;`.
- One `[[example]]` entry per translated example; example name = C base
  name.
- `user_data` is `Option<Box<dyn Any>>`; callbacks are plain `fn`
  pointers. Aliasing vector ops get in-place methods; free functions
  (`N_VLinearSum`) serve distinct operands.

## Workflow

- Commit after every ported file (or small coherent group); tag phase
  gates (`phase2-cvode-green`, …).
- After EVERY build/test/run: `… 2>&1 | tee <log>` then **Read the log**
  before the next edit. Never re-run a command that produced no output.
- Max two attempts per failing command, then switch strategy.
- Read each in-scope C file exactly once, at translation time, completely.
  Never read excluded paths (GPU/MPI/KLU/LAPACK/Fortran/xbraid trees).
- Update `PROGRESS.md` (per-file status: todo | ported | building |
  committed) and `VERIFICATION.md` (per-variant status) as units land.
- Resume after context loss from this file + `PROGRESS.md` + `git log` —
  do not re-explore the tree.

## Verification

`tools/verify_examples.sh [crate|all|list]` parses the upstream
CMakeLists tuples (199 variants), builds release examples, runs each
variant with exact argv, diffs against `../examples/...` references
(noise-filtered symmetrically), and writes `logs/summary.txt`. Read only
the summary; open individual diffs only for non-IDENTICAL lines.
CLI-option variants use bare `<solverid>.<key>` tokens (no leading
dashes); the parser prefix-matches literally.

Current Linux/x86-64 gate: **153 IDENTICAL / 26 reference-side / 20
excluded**, 0 port defects — where "0 port defects" is a measurement, not a
judgement call: `tools/pristine_c_build.sh` builds the upstream C library
and examples with cmake/gcc out of source, and `tools/compare_pristine_c.sh`
(plus `tools/compare_lapack_substituted.sh` for the two `*L` examples) runs
every divergent variant as Rust, as pristine C, and against the shipped
reference. **A divergence is a port defect only when Rust != pristine C on
the same host.** All 26 currently come out `same`. Re-run these three after
any change that could move numeric output; never reclassify a variant from
the reference alone.

`tools/classify_diffs.sh` is the second pass —
it re-diffs the non-IDENTICAL variants under `tr -s ' '` and `diff -w` so a
whitespace-only divergence (stale `SUN_TABLE_WIDTH` 28 -> 29 references) can
be told from a content one without opening 26 diffs. Never widen
`noise_filter()` to swallow last-ulp drift, and never tune an example to
match a reference.

Cross-distribution tooling, for any change that could move numeric output
or widen a platform claim: `tools/glibc_sweep.sh` fingerprints each
distribution's libm function by function (FNV-1a over 1M inputs, via
`tools/libm_probe.c`) — cheap, needs only a C compiler per container, and
it *predicts* which variants are at risk. `tools/gate_in_container.sh
<image>...` then runs the full gate natively inside those distributions to
confirm whether the difference is output-observable. Never state a
distribution claim these two have not been run for.

`tools/pow_differential.sh [domain|random|all]` builds `tools/pow_oracle.c`
with the host compiler and runs the two `pow_glibc_vs_native_oracle_*` tests
against it. Re-run it after **any** change to `pow_glibc` — the example gate
is blind to that class of defect (POW_FMA_EXACTNESS.md §6). Keep
`pow_corpus` in `sundials_math.rs` byte-for-byte in step with the corpus
generator in `pow_oracle.c`; if they drift, the differential silently
compares different inputs.

## Working from Windows

`tools/wsl_sync_build.sh {build|test|rel|gate|pow|sync}` rsyncs this working
copy into a WSL Ubuntu sandbox (`~/sdl/port`, with `~/sdl/examples`
symlinked at the upstream C tree so `../examples/...` resolves) and runs the
step there. It strips CRLF from `tools/*.sh` first; `.gitattributes` pins
those files to LF so a Windows checkout cannot break the shebang lines.
Invoke it as `wsl.exe -d Ubuntu-24.04 -- bash tools/wsl_sync_build.sh <step>`
— do **not** pass `$PATH` inside a `bash -c` string, the interop layer
pre-expands it and the Windows paths containing `(x86)` break bash parsing.
