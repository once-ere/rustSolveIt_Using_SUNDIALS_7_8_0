# Port ALL of SUNDIALS 7.8.0 to pure Rust — `sundials_7_8_0__rs`

Produce a **100% faithful port of this SUNDIALS 7.8.0 C library to a
pure-Rust Cargo workspace at `./sundials_7_8_0__rs/`** (created inside this
directory, `/Users/youruser/Developer/sundials-7.8.0/`). Translate 100%
faithfully with line-by-line fidelity; every ported function's argument
list MUST have 100% alignment (names, order, meaning) with its parent C
function's arguments. Verify against the upstream serial example reference
outputs shipped in `./examples/`. Plan your solution after understanding
this entire prompt, then work until completion. Prepare docs.

---

## 0. Environment facts (do not assume otherwise)

- The current directory **is** the upstream SUNDIALS 7.8.0 C source tree.
  Treat everything in it as **read-only reference material**, except the new
  `./sundials_7_8_0__rs/` workspace directory, which you create and own.
- **There is no donor port.** No `cvode_rs`, `sundials-7.7.0`, or
  `cvode-7.7.0` tree exists in this environment. If global configuration or
  memory refers to such reference paths, they are stale — ignore them.
  Every line of Rust in this project is ported fresh from `./src/`,
  `./include/`, and `./examples/`.
- This directory is not a git repository. Run `git init` inside
  `./sundials_7_8_0__rs/` — the workspace is its own repo.

## 1. Scope — what "everything, in pure Rust" means

Pure Rust: **zero `unsafe`, zero FFI, zero external crate dependencies,
zero warnings** — `std` only. Consequently the port covers every code path
that is meaningful without external C/Fortran/GPU/MPI libraries, and
explicitly excludes the rest. Every exclusion below is a *hard* exclusion —
do not attempt it, do not add FFI to "complete" it, do not spend tokens
reading it.

**IN SCOPE** (translate 100% faithfully; each `*_impl.h` and public
`include/` header ports together with its `.c` module):

| Unit | Source | C lines (top-level `.c`) |
|---|---|---|
| sundials core | `src/sundials/*.c` except the two excluded below, plus `src/sundials/sundatanode/sundatanode_inmem.c` and the header-only `src/sundials/stl/sunstl_vector.h` | ~10,000 |
| serial vectors/matrices/solvers | `src/nvector/serial`; `src/sunmatrix/{band,dense,sparse}`; `src/sunlinsol/{band,dense,pcg,spbcgs,spfgmr,spgmr,sptfqmr}`; `src/sunnonlinsol/{newton,fixedpoint,auto}` (`auto` is new in 7.8.0 — do not skip it) | ~11,700 |
| controllers/estimators | `src/sunadaptcontroller/{soderlind,imexgus,mrihtol}`, `src/sundomeigest/{power,arnoldi}`, `src/sunadjointcheckpointscheme/fixed`, `src/sunmemory/system` | ~3,400 |
| cvode | `src/cvode/*.c` incl. `cvode_resize.c`, `cvode_cli.c`, `cvode_fused_stubs.c` (serial stubs; the `.cpp` GPU twin is excluded) | ~12,500 |
| cvodes | `src/cvodes/*.c` (incl. `cvodea.c`, `cvodea_io.c`, `cvodes_resize.c`, `cvodes_cli.c`) | ~25,800 |
| kinsol | `src/kinsol/*.c` | ~6,700 |
| ida | `src/ida/*.c` | ~9,500 |
| idas | `src/idas/*.c` (incl. `idaa.c`, `idaa_io.c`) | ~22,900 |
| arkode | `src/arkode/*.c` + the four `.def` butcher/coupling tables (`xbraid/` subdir excluded) | ~50,000 |
| serial examples | `examples/cvode/serial`, `examples/cvodes/serial`, `examples/kinsol/serial`, `examples/ida/serial`, `examples/idas/serial`, `examples/arkode/C_serial` — 128 C programs total (21+32+11+11+19+34), of which 20 require KLU/SuperLU and are excluded (list below) → **108 ported programs**, verified against ~199 reference `.out` variants | — |

**EXCLUDED** (never read, never port):

- Every `fmod_int32/`, `fmod_int64/` directory (Fortran bindings), anywhere.
- `src/nvector/{cuda,hip,sycl,raja,openmp,openmpdev,pthreads,parallel,parhyp,petsc,trilinos,mpiplusx,manyvector}`.
- `src/sunlinsol/{klu,superlumt,superludist,lapackband,lapackdense,magmadense,onemkldense,cusolversp}`.
- `src/sunmatrix/{cusparse,magmadense,onemkldense,slunrloc}`.
- `src/sunmemory/{cuda,hip,sycl}`; `src/sunnonlinsol/petscsnes`.
- `src/sundials/sundials_mpi_errors.c`, `src/sundials/sundials_xbraid.c`,
  and the GPU/vendor headers there (`sundials_cuda*.h`, `sundials_hip*.h`,
  `sundials_sycl.h`, `sundials_cusolver.h`, `sundials_cusparse.h`,
  `sundials_reductions.hpp`, `sundials_adiak_metadata.h`,
  `sundials_lapack_defs.h.in`).
- `src/cvode/cvode_fused_gpu.cpp`; `src/arkode/xbraid/`.
- MPI-only `#ifdef` branches inside otherwise-serial files (port the serial
  branch).
- The `doc/`, `benchmarks/`, `cmake/`, `external/`, `test/`, `suntools/`
  trees (note: the tools tree here is named `suntools`, not `tools`).
- All non-serial example dirs (`parallel`, `C_openmp*`, `petsc`, `cuda`,
  `hip`, `raja`, `kokkos`, `magma`, `ginkgo`, `trilinos`, `superludist`,
  `F2003_*`, `CXX_*`, `C_klu`, `C_superlu-mt`, `C_manyvector`,
  `C_mpimanyvector`, `C_parallel`, `C_parhyp`, `C_petsc`, `C_openmpdev`,
  `CUDA_mpi`, `mpicuda`, `mpiraja`, …).
- Serial examples that require an excluded KLU/SuperLU backend — excluded
  and recorded as `excluded(klu)` / `excluded(superlu)` in
  `VERIFICATION.md`. Exactly these 20:
  - cvode: `cvRoberts_klu.c`, `cvRoberts_block_klu.c`, `cvRoberts_sps.c`
  - cvodes: `cvsRoberts_klu.c`, `cvsRoberts_sps.c`, `cvsRoberts_FSA_klu.c`,
    `cvsRoberts_FSA_sps.c`, `cvsRoberts_ASAi_klu.c`, `cvsRoberts_ASAi_sps.c`
  - kinsol: `kinFerTron_klu.c`, `kinRoboKin_slu.c`
  - ida: `idaHeat2D_klu.c`, `idaRoberts_klu.c`, `idaRoberts_sps.c`
  - idas: `idasRoberts_klu.c`, `idasRoberts_sps.c`, `idasRoberts_FSA_klu.c`,
    `idasRoberts_FSA_sps.c`, `idasRoberts_ASAi_klu.c`,
    `idasRoberts_ASAi_sps.c`
- Examples using LAPACK linear solvers (`*L` names — exactly four:
  `cvAdvDiff_bndL.c`, `cvRoberts_dnsL.c` in `examples/cvode/serial`;
  `cvsAdvDiff_bndL.c`, `cvsRoberts_dnsL.c` in `examples/cvodes/serial`)
  are **ported** using the native dense/band solvers instead — expect and
  document last-digit output differences in `VERIFICATION.md`.
- Python/Matlab plotting scripts and `*_stats.csv` files in example dirs
  are not verification targets; ignore them.

## 2. Target architecture — Cargo workspace

```
sundials_7_8_0__rs/           # git repo root — `git init` here
  Cargo.toml                  # [workspace] members = crates/*
  CLAUDE.md                   # workspace-wide rules (authored in Phase 0)
  ARCHITECTURE.md             # cross-module contracts (authored in Phase 0, extended per solver)
  PROGRESS.md                 # durable per-file port checklist (see §4)
  VERIFICATION.md             # durable per-example verification matrix (see §6)
  tools/verify_examples.sh    # batch example-verification harness (see §6)
  logs/                       # harness + build logs (gitignored)
  crates/
    sundials_core/            # everything shared: sundials_*, nvector_serial,
                              # sunmatrix_*, sunlinsol_*, sunnonlinsol_*,
                              # sunadaptcontroller_*, sundomeigest_*,
                              # sunadjointcheckpointscheme_*, sunmemory_*,
                              # plus sundials_utils (fmt_e/fmt_f/fmt_g)
    cvode_rs/                 # solver crates: <c file>.rs modules + examples/
    cvodes_rs/
    kinsol_rs/
    ida_rs/
    idas_rs/
    arkode_rs/
```

- Module = original C file base name + `.rs` (`cvodes_nls_stg1.c` →
  `crates/cvodes_rs/src/cvodes_nls_stg1.rs`; impl headers like
  `arkode_impl.h` → `arkode_impl.rs`). Public `include/` headers hold
  constants, return flags, and inline helpers — fold their content into the
  corresponding module (check `include/<pkg>/` for every module you port).
- Public API keeps exact C names (`CVode`, `CVodeB`, `IDASolve`, `KINSol`,
  `ARKStepCreate`, `SUNAdaptController_Soderlind`, …) and exact C
  return-flag conventions (`CV_SUCCESS = 0`, `IDA_SUCCESS = 0`; negative =
  fatal, positive = recoverable). Crate roots carry
  `#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]`.
- Each solver crate re-exports every shared module from `sundials_core` at
  its root (e.g. in `crates/cvode_rs/src/lib.rs`:
  `pub use sundials_core::nvector_serial;` — one line per shared module)
  and provides a flat prelude (`pub use crate::*;` pattern) so examples can
  say `use cvode_rs::*;`.
- Each solver crate's `Cargo.toml` gets an explicit `[[example]]` entry per
  translated example; example names = C file base names.
- The four `.def` files (`arkode_butcher_erk.def`, `arkode_butcher_dirk.def`,
  `arkode_mri_tables.def`, `arkode_splittingstep_coefficients.def`) are C
  X-macro tables: translate to Rust `const` table data inside the module of
  the `.c` file that includes them. 7.8.0 added new tables (SSP ERK/DIRK,
  ASCHER, IMEX-MRI-GARK ARK2 couplings) — port the complete current tables,
  not a remembered older set.
- `user_data` is `Option<Box<dyn Any>>`; user callbacks stay plain `fn`
  pointers. Vector ops that may alias operands get in-place methods; free
  functions (e.g. `N_VLinearSum`) serve the distinct-operand case.
- OS/toolchain-specific C (timers in `sundials_profiler.c`, file utils in
  `sundials_futils.c`, env access in `sundials_logger.c`/`sundials_cli.c`)
  maps to `std` equivalents (`std::time::Instant`, `std::fs`,
  `std::env`); keep the public API and observable behavior identical.
- `sundials_core::sundials_utils` must provide `fmt_e`, `fmt_f`, `fmt_g`
  with exact C `printf` `%e/%f/%g` semantics (including `%g` trailing-zero
  stripping and exponent form). There is no donor to copy — author these
  first, with unit tests against known C outputs, because every example's
  byte-identical output depends on them. Never use Rust's `{:e}`.

## 3. Non-negotiable engineering rules

1. **Fidelity first.** Control flow, constants, tolerances, heuristics, and
   error/return codes must match the C source in behavior. Preserve
   arithmetic order — floating point is not associative, and the acceptance
   test is byte-identical printed output.
2. **DO NOT BREAK ANYTHING.** Once a crate's examples verify green, they
   must remain green after every later phase (cumulative regression gate,
   §6).
3. Zero `unsafe`, zero external dependencies, zero warnings from
   `cargo build --workspace`.
4. Watch C pointer aliasing when translating APIs: where C aliases buffers
   (e.g. CVODE aliases its internal `cv_y` with the user's `yout`), the
   Rust port must copy back at **every** return path, including
   early-error and rootfinding-return exits. Check every solver for the
   same pattern (CVODE(S), IDA(S), and ARKODE all do this).
5. All floating-point output formatting goes through
   `sundials_core::sundials_utils::{fmt_e, fmt_f, fmt_g}` — never `{:e}`.
6. When a symbol is missing, its definition is under `./src/` or
   `./include/` — port it into `sundials_core`; never invent a stub or
   placeholder.
7. Read 100% of every in-scope C file you port — a detailed, line-by-line,
   low-level scan. Long files (`cvodes.c` is ~10,100 lines, `idas.c`
   ~8,900, `arkode.c` ~4,100) are read in sequential chunks, translating as
   you go. Never skim-and-guess.

## 4. Workflow discipline (MANDATORY)

1. **Undo mechanism = git**, not file copies: `git init` in
   `sundials_7_8_0__rs/`, commit after every ported file (or small coherent
   group) and tag every phase gate (`phase2-cvode-green`, …). Before any
   wholesale/destructive operation, run `git status` and look at what you
   would overwrite; if the tree contains work you don't recognize, stop and
   branch instead of overwriting. Never modify files outside the workspace;
   the upstream C tree is read-only.
2. **After EVERY build or test command, capture output to a file and Read
   it back before deciding the next edit — never re-run a command that
   returned no visible output.** Concretely:
   `cargo build --workspace 2>&1 | tee /tmp/build.log`, then Read
   `/tmp/build.log`. This applies to `cargo build`, `cargo test`,
   `cargo run --example …`, diffs, and the harness — no exceptions.
3. **No more than two attempts at any single failing command before
   switching strategy** (different diagnostic, smaller reproduction,
   re-read the C source, or consult `ARCHITECTURE.md`).
4. **Token economy.** Read each in-scope C file exactly once, at
   translation time. Never read excluded paths (§1). Record design
   decisions in `ARCHITECTURE.md` the first time you make them and reuse
   them instead of re-deriving. Verify examples in batch via the harness
   (§6), not one manual run at a time. For `cvodes` and `idas`, exploit
   that they are supersets of `cvode`/`ida`: port against your
   already-ported sibling modules, using the C sources to identify exactly
   what was added (sensitivity, adjoint, quadrature paths) rather than
   re-deriving the shared majority.
5. **Durable state.** Maintain `PROGRESS.md` (one checklist line per
   in-scope C file: `todo | ported | building | committed`) and
   `VERIFICATION.md` (§6), updating them as each unit lands. If your
   context is compacted, resume from `CLAUDE.md` + `PROGRESS.md` +
   `git log` — do not re-explore the tree.

## 5. Execution phases (work in this exact order; status update after each)

**Phase 0 — Workspace bootstrap (no solver translation).**
Create `sundials_7_8_0__rs/` laid out as §2 (`cargo new --lib` per crate,
then edit). Author `CLAUDE.md` (distill §§1–6 of this prompt into workspace
rules), `ARCHITECTURE.md` (record the §2 contracts: module naming,
re-export shim, `user_data` model, callback signatures, aliasing/copy-back
rule), `PROGRESS.md` pre-populated with every in-scope C file from §1, and
`VERIFICATION.md` pre-populated with every in-scope example variant (§6).
Write `tools/verify_examples.sh`. Implement and unit-test
`sundials_utils::{fmt_e, fmt_f, fmt_g}`. Gate: workspace builds clean,
`cargo test` green. Commit, tag `phase0-skeleton`.

**Phase 1 — sundials_core.**
Port the shared foundation, roughly in dependency order:
`sundials_math.c`, `sundials_errors.c`, `sundials_context.c`,
`sundials_nvector.c` + `nvector/serial/nvector_serial.c`,
`sundials_matrix.c` + `sunmatrix/{band,dense,sparse}`,
`sundials_direct.c`, `sundials_band.c`, `sundials_dense.c`,
`sundials_iterative.c`, `sundials_linearsolver.c` +
`sunlinsol/{band,dense,spgmr,spfgmr,spbcgs,sptfqmr,pcg}`,
`sundials_nonlinearsolver.c` + `sunnonlinsol/{newton,fixedpoint,auto}`,
`sundials_nvector_senswrapper.c` (required by cvodes/idas sensitivities),
`sundials_memory.c` + `sunmemory/system`, `sundials_logger.c`,
`sundials_profiler.c`, `sundials_futils.c`, `sundials_hashmap.c`,
`sundials_version.c`, `sundials_cli.c` (7.8.0 command-line option loading —
several reference outputs exercise it), `sundials_adaptcontroller.c` +
`sunadaptcontroller/{soderlind,imexgus,mrihtol}`.
You may defer the adjoint/stepper/datanode subset —
`sundials_stepper.c`, `sundials_adjointstepper.c`,
`sundials_adjointcheckpointscheme.c` + `sunadjointcheckpointscheme/fixed`,
`sundials_datanode.c` + `sundatanode_inmem.c` + `stl/sunstl_vector.h`,
`sundials_domeigestimator.c` + `sundomeigest/{power,arnoldi}` — to the
start of Phase 7 (arkode needs them: LSRK needs sundomeigest;
`ark_lotka_volterra_ASA` needs the adjoint stack). Note any deferral in
`PROGRESS.md`. Gate: build clean, `cargo test` green. Commit, tag.

**Phase 2 — cvode (~12.5k lines; establishes every solver pattern).**
Modules: `cvode.c`, `cvode_io.c`, `cvode_ls.c`, `cvode_nls.c`,
`cvode_diag.c`, `cvode_proj.c`, `cvode_bandpre.c`, `cvode_bbdpre.c`
(serial branch), `cvode_resize.c`, `cvode_cli.c`, `cvode_fused_stubs.c` +
impl headers. Then all 18 in-scope `examples/cvode/serial` examples
(including the two `*L` ones on native solvers). This phase fixes the
project-wide patterns for error handling, `yout` copy-back, ls/nls
interfaces, rootfinding, and CLI options — get it fully green before
moving on. Gate (§6). Tag `phase2-cvode-green`.

**Phase 3 — cvodes (~25.8k; maximum reuse from Phase 2).** Modules:
`cvodes.c`, `cvodea.c`, `cvodea_io.c`, `cvodes_io/ls/nls{,_sim,_stg,_stg1}/
proj/diag/bandpre/bbdpre/resize/cli` + impl headers. Then all 26 in-scope
`examples/cvodes/serial` examples (including its two `*L` ones —
`cvsAdvDiff_bndL`, `cvsRoberts_dnsL` — on native solvers). Gate. Tag.

**Phase 4 — kinsol (~6.7k; standalone, quick win).** `kinsol.c`,
`kinsol_aa.c`, `kinsol_orth.c`, `kinsol_ls/io/bbdpre/cli` + impl headers;
9 in-scope serial examples (many `kinAnalytic_fp` argument variants — see
§6). Gate. Tag.

**Phase 5 — ida (~9.5k).** `ida.c`, `ida_ic.c`, `ida_ls/nls/io/bbdpre/cli`
+ impl headers; 8 in-scope serial examples. Gate. Tag.

**Phase 6 — idas (~22.9k; diff-driven from ida + Phase-3 sensitivity
patterns).** All `idas_*.c` + `idaa.c`, `idaa_io.c`; 13 in-scope serial
examples. Gate. Tag.

**Phase 7 — arkode (~50k; largest — everything else must already be
green).** First port any core modules deferred in Phase 1. Suggested
internal order: butcher tables (`arkode_butcher*.c` + `.def`) →
`arkode_interp.c` → core `arkode.c`/`arkode_io.c`/`arkode_adapt.c` →
`arkode_root.c` → `arkode_erkstep*` → `arkode_ls.c` + `arkode_arkstep*` →
`arkode_sprk.c`/`arkode_sprkstep*` → `arkode_lsrkstep*` (needs
sundomeigest) → `arkode_mri_tables.c` + `arkode_mristep*` →
`arkode_splittingstep*` (+ `_coefficients`) / `arkode_forcingstep.c` →
`arkode_bandpre.c`/`arkode_bbdpre.c`/`arkode_relaxation.c`/
`arkode_user_controller.c`/`arkode_sunstepper.c`/`arkode_cli.c`. The
`xbraid/` subdir is excluded. Then all 34 `examples/arkode/C_serial`
examples (several have companion `.h` files — `ark_kepler.h`,
`ark_harmonic_symplectic.h`, `ark_damped_harmonic_symplectic.h` — port
them alongside). Gate. Tag.

**Phase 8 — Full sweep + documentation.**
Run the complete harness across all crates; every `PROGRESS.md` line must
be `committed`, every `VERIFICATION.md` line `identical`,
`last-digit(reason)`, or `excluded(reason)` — no blanks, no silent skips.
Fix any warning. Then write `sundials_7_8_0__rs/sundials.md`: complete,
correct, precise documentation aimed at an ignorant U.S. high-school
student — covering what ODE/DAE/nonlinear solving is, all six solvers'
logic, all core data structures, the full user-facing API (every public
function, its arguments, return flags), and worked examples — written so
the student can teach it to classmates and the teacher. Finalize
`CLAUDE.md` and `ARCHITECTURE.md`. Final commit, tag `v1-complete`.

## 6. Verification protocol

Many upstream examples ship **multiple reference outputs, one per
command-line argument set**. The mapping lives in each example dir's
`CMakeLists.txt` as `"<name>\;<args>\;<label>"` tuples; the reference
file is `<name>.out` when `<args>` is empty, else
`<name>_<args with spaces replaced by underscores>.out` (e.g.
`kinAnalytic_fp\;--m_aa 2\;` → `kinAnalytic_fp_--m_aa_2.out`;
`ark_kepler` with `--stepper ERK --step-mode adapt` →
`ark_kepler_--stepper_ERK_--step-mode_adapt.out`). Some variants exercise
the 7.8.0 CLI option system, whose tokens are **bare
`<solverid>.<key>` pairs with NO leading dashes** (e.g.
`idaAnalytic_mels_ida.scalar_tolerances_1e-3_1e-8.out` comes from argv
`ida.scalar_tolerances 1e-3 1e-8`; the parser in `ida_cli.c` /
`sundials_cli.c` prefix-matches `ida.` literally and would silently skip
`--ida.*` — port that exact behavior).

Known parsing quirks of the upstream `CMakeLists.txt` files (with these
handled, all 199 `.out` files across the six dirs are claimed — verified;
there are no orphans):

- In `examples/arkode/C_serial` the first tuple field is the **source file
  name** (`ark_kepler.c\;--stepper ERK --step-mode adapt\;develop`) —
  strip the `.c` extension before applying the naming rule. The other
  five serial dirs use the bare example name.
- `examples/kinsol/serial`'s KLU/SuperLU_MT lists use **2-field**
  `"<name>\;<label>"` tuples (no args slot) — treat missing args as
  empty so `kinFerTron_klu.out`/`kinRoboKin_slu.out` are claimed (then
  marked excluded per §1).
- Tuples inside `if(SUNDIALS_ENABLE_MONITORING)` blocks
  (`cvKrylovDemo_ls\;0 1`, `cvsKrylovDemo_ls\;0 1`,
  `ark_brusselator_fp.c\;1`) are still real variants with shipped `.out`
  files — a plain text-level tuple scan picks them up; do not evaluate
  CMake conditionals.
- Skip commented-out lines (e.g. `examples/ida/serial` has
  `# "idaHeat2D_sps\;develop"` with no corresponding `.out`).

`tools/verify_examples.sh [crate|all]` must:

1. Parse the upstream `CMakeLists.txt` of each in-scope example dir to
   enumerate `(example, args)` pairs and their expected `.out` files;
   cross-check that every `.out` file in the dir is claimed by some pair
   (investigate any orphan — never silently skip a reference file).
   `VERIFICATION.md` carries **one line per (example, args) variant**.
2. Build once per crate in release
   (`cargo build --release --examples -p <crate> 2>&1 | tee logs/build-<crate>.log`).
3. Run each variant with exactly those argv, `tee` stdout to
   `logs/<expected-out-name>`, and diff against the upstream reference at
   `../examples/<solver>/<serial-dir>/<name>[_<args>].out` (path relative
   to the workspace root, which sits inside the C tree).
4. Append one line per variant to `logs/summary.txt`:
   `<example> [<args>]  IDENTICAL | DIFF(n lines) | FAIL(exit code)`.
   You then Read **only** `logs/summary.txt`, and open individual diffs
   only for non-IDENTICAL lines.

- Target: byte-identical output. Acceptable documented exceptions:
  `*L` (LAPACK→native) examples and iterative-solver last-digit drift —
  each such variant gets a one-line justification in `VERIFICATION.md`.
- Examples that print timings or machine-dependent noise: the harness
  filters those lines symmetrically from both sides before diffing
  (document the filter in the script).
- `*_stats.csv` reference files are not verification targets.
- **Phase gate** = workspace builds with zero warnings + `cargo test`
  green + harness summary clean for **all crates ported so far** (the
  cumulative regression gate), each output tee'd to a log and Read back.

## 7. Definition of done

- [ ] `cargo build --workspace 2>&1 | tee …` → zero errors, zero warnings.
- [ ] `cargo test --workspace` green.
- [ ] Every in-scope C file from §1 has a same-named `.rs` module, ported
      100% faithfully; `PROGRESS.md` shows all `committed`.
- [ ] All 108 in-scope example programs build; every reference variant in
      `VERIFICATION.md` is `identical`, `last-digit(reason)`, or
      `excluded(reason)` — no blanks.
- [ ] Earlier-phase crates still verify green (nothing broken).
- [ ] `sundials.md`, `CLAUDE.md`, `ARCHITECTURE.md`, `PROGRESS.md`,
      `VERIFICATION.md` written and current.
- [ ] Git history: per-file commits, per-phase tags, `v1-complete`.
