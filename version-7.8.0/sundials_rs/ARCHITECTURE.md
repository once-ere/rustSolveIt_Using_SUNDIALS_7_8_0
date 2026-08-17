# ARCHITECTURE — cross-module contracts

Pinned decisions. Read before modifying shared types; extend (append) when
a new contract is fixed, never silently change an existing one.

## Platform contract

`SUNDIALS_7_8_Rust_port_for_Linux` targets **Linux on Intel/AMD x86-64 with
glibc**. (The contracts below were fixed in the sibling
`SUNDIALS_7_8_Rust_port_for_AppleSilicon_macos`, whose crate tree this
workspace inherits unchanged; they describe the translation and are
platform-neutral.) Two consequences bind everything below:

* **No conditional compilation, ever.** There is no `cfg(target_os)` or
  `cfg(target_arch)` in the tree and none may be added. A port for another
  platform is a separate repository, so that "what was verified, and where"
  is a property of the repository rather than of a build flag.
* **Host libm is an input, not an implementation detail.** `sunrealtype`
  arithmetic is IEEE-754 and reproducible everywhere, but `SUNRsqrt`,
  `SUNRexp`, `SUNRceil`, `SUNRround` and every `sin`/`cos`/`exp`/`log` in the
  examples go through `f64`'s methods, which Rust `std` forwards to the host
  libm. Only `SUNRpowerR` was taken off the host libm (`pow_glibc` in
  `sundials_math.rs`). Every deviation class below was argued unobservable
  *on this platform*; the argument does not automatically survive a change of
  libm.

## Crate graph

`sundials_core` ← {`cvode_rs`, `cvodes_rs`, `kinsol_rs`, `ida_rs`,
`idas_rs`, `arkode_rs`}. Solver crates never depend on each other.

## Module naming

One Rust module per upstream C file, named after its base name. Impl
headers (`*_impl.h`) and public `include/<pkg>/*.h` content fold into the
matching module. `.def` X-macro tables become `const` table data in the
including module.

## Core type model (fixed in Phase 1)

- `sunrealtype` = `f64`, `sunindextype` = `i64`, `suncountertype` = `i64`,
  `sunbooleantype` = `bool`, `SUNErrCode` = `i32`, `SUNComm` = `i32`.
- **Handle model.** Every C heap object reached through a pointer
  (`N_Vector`, `SUNMatrix`, `SUNLinearSolver`, `SUNNonlinearSolver`,
  `SUNAdaptController`, `SUNContext`, solver mems, …) becomes
  `pub type X = Rc<X_>` where `X_` holds `content: RefCell<Box<dyn Any>>`
  (C `void* content`) plus an ops struct of plain `fn` pointers (where the
  C API has one) and the `sunctx` handle. Cloning the `Rc` is the C
  pointer copy; `Rc::ptr_eq` is C pointer equality.
- Ops are plain `fn` pointers taking `&Handle` arguments — identical call
  shape to C — and mutate through the `RefCell`. User-supplied override
  implementations keep working exactly as in C.
- `SUNContext` owns the error handler stack, logger, and profiler exactly
  as in C.
- `user_data`: `Option<Box<dyn Any>>`, passed to callbacks as
  `Option<&mut dyn Any>` (C: `void*`). Solver internals `Option::take`
  the box out of the mem record around each callback invocation, so the
  callback gets exclusive access without re-borrowing the mem.
- Solver mems: the public handle is `Rc<RefCell<CVodeMemRec>>`-style;
  public API functions borrow once at entry and pass `&mut CVodeMemRec`
  internally — matching C's `cv_mem->` style with zero borrow churn.
- User callbacks are plain `fn` pointer types matching the C signature
  argument-for-argument (same names in the same order in Rust signatures).
  Do not change a callback signature without updating every example.

## Aliasing / copy-back rule

Where C aliases internal state with user buffers (CVODE `cv_y`/`yout`,
IDA `ida_yy`/`yret` etc., ARKODE `ycur`/`yout`), the Rust port copies the
internal buffer to the user buffer at every return path — success, early
error, and root-return alike.

Vector ops: free functions (`N_VLinearSum(a, &x, b, &y, &z)`) mirror the
C call shape for **all** call sites and are alias-safe by construction —
implementations detect operand aliasing (`Rc::ptr_eq`) and take a single
mutable borrow for the aliased case. This satisfies the in-place-method
contract trivially: the free function *is* safe under aliasing, so C call
sites translate 1:1 whether or not they alias. In-place convenience
methods may exist additionally but are never required for safety.

## Formatting

`sundials_core::sundials_utils::{fmt_e, fmt_f, fmt_g}` implement C
`printf` `%e/%f/%g` exactly (default precision 6; `%g` strips trailing
zeros; exponent `e±dd`, at least two digits; `inf`/`nan` lowercase).
Width variants `fmt_ew/fmt_fw/fmt_gw(x, width, prec)` right-justify with
spaces (C `%W.Pe`). Never use Rust `{:e}`.

## OS mapping

`sundials_profiler.c` timers → `std::time::Instant`;
`sundials_futils.c` → `std::fs`; env access in logger/CLI → `std::env`.
Public API and observable behavior identical to C.

## Error/return conventions

Exact C flag names and values (`CV_SUCCESS`, `CV_MEM_NULL`, `IDA_SUCCESS`,
`KIN_SUCCESS`, `ARK_SUCCESS`, `SUN_SUCCESS`, …). Negative = fatal,
positive = recoverable. Functions that return flags in C return the same
integer type in Rust; output parameters in C (`T *out`) become `&mut T`
in the same argument position with the same name.

## Established porting patterns (locked during Phase 1)

- **Content downcast**: every implementation module defines
  `fn content_mut(X) -> RefMut<'_, ContentStruct>` via
  `RefMut::map(x.content.borrow_mut(), downcast_mut)`. Public accessor
  macros (`NV_DATA_S`, `SM_DATA_D`, …) are functions returning `RefMut`
  guards; drop the guard before any other op on the same object.
- **Granular borrow rule**: never hold a RefCell borrow (of a mem, a
  solver content, or vector data) across a call that can re-enter it —
  callbacks (RHS, ATimes, Psolve, Jacobian) reach integrator state
  through their own handle. Iterative-solver `solve` ops move ALL content
  state into locals at entry (`Option::take` for `Box<dyn Any>` callback
  data, `mem::take` for arrays, `Rc` clones for vectors), run the C
  algorithm inside a closure returning the flag, and restore + write back
  (numiters/resnorm/zeroguess/last_flag) at one exit point. Final flag
  values are identical to C's multi-return-path writes because
  logging-level-2 builds have no observable effects in between.
- **`SUNCheck*`/`SUNAssert`**: release no-ops; call sites evaluate the
  call and continue (`let _ = f(...)` where C had `(void)`).
  `SUNLogInfo`/`SUNLogDebug`/`SUNLogExtraDebug*` compile away entirely at
  logging level 2 and are omitted at translation time.
- **CLI parsing**: `argv: &[String]` with `argv[0]` = program name; C
  `atoi` maps to `s.trim().parse().unwrap_or(0)`; prefix matching is
  literal (`<id>.` with no leading dashes).
- **C output params**: `T *out` → `&mut T` same position/name;
  functions returning object pointers return `Option<Handle>`
  (NULL = `None`). Constructors that C would fail with NULL return
  `None`.
- **fmt helpers**: `fmt_e/f/g(x, prec)`, width variants
  `fmt_ew/fmt_fw/fmt_gw(x, width, prec)`, and `sun_format_e/g/sg(x)` for
  the `SUN_FORMAT_E/G/SG` macros ("% .15e" / "%.15g" / "%+.15g").
- **Vector arrays**: C `N_Vector*` → `&[N_Vector]` (handles are Rc
  clones); `N_Vector**` → `&[Vec<N_Vector>]`. Row-wise Hessenberg
  `sunrealtype**` → `&mut [Vec<f64>]`; column-pointer arrays
  (`SUNDlsMat cols`) → `dls_cols()` chunks_mut views.

## Accepted deviation classes (adversarially verified, Phase 1)

These divergences from the release-C reference build are deliberate,
verified unobservable on any path a valid serial example takes, and must
be applied CONSISTENTLY in later phases:

1. **Kept failure-path checks.** Where release C compiles out
   `SUNAssert`/`SUNCheckCall` (silently proceeding on misuse), ported
   modules may keep the check as a plain `if` returning the error code
   (fn-pointer-presence checks, `file_name` guards, propagated sub-call
   errors in Set*/Initialize forwarding). Only observable on invalid
   usage or malformed CLI input.
2. **Ownership snapshots.** `SUNMemoryHelper_Wrap`/`_Alias` take/clone
   owned `Vec<u8>` buffers instead of aliasing raw pointers; no
   write-through. Any future consumer ported from C that mutates a
   wrapped buffer afterward must mutate through the SUNMemory handle.
3. **C-locale ASCII whitespace.** `atoi`/`atol`/`SUNStrToReal` skip only
   ASCII whitespace (matching C-locale `isspace`/`strtod`), implemented
   via `trim_start_matches` — never Unicode `trim`.
4. **Unsigned wrap.** C `size_t` counter arithmetic that can underflow
   maps to `wrapping_sub` (never a panicking `-=`).
5. **C UB → deterministic panic.** NULL deref / out-of-bounds /
   double-free in C map to Rust panics at the same site.
   **Named exception (Phase 7, ARKODE MRISR embedding stage).**
   `MRIStepCoupling_Alloc` gives every `G` matrix `stages+1` ROWS but only
   `stages` COLUMNS (`src/arkode/arkode_mri_tables.c:181-203`), so on the
   embedding iteration of `mriStep_TakeStepMRISR` (`stage == stages`,
   taken whenever `!fixedstep || AccumErrorType != ARK_ACCUMERROR_NONE`)
   upstream's `SUNRabs(step_mem->MRIC->G[0][stage][stage])`
   (`src/arkode/arkode_mristep.c:2592`) reads one element past the end of a
   `calloc`'d row. Unlike the rest of class 5 this IS on a valid path — the
   default adaptive ImEx MRI tables (`ARKODE_IMEX_MRI_SR21/32/43`) select
   it — so a panic there would abort runs the C reference completes. The
   embedding row has no diagonal entry by construction (every in-bounds use
   of that row is `G[0][stage][j]` for `j < stage`), and zeroed `calloc`
   storage is what upstream relies on, so `arkode_mristep.rs` reads ZERO
   for that one out-of-range column: `impl_corr` is false on the embedding
   stage and the dependent `gamma` update is then unreachable. Do NOT
   generalize this to other out-of-bounds sites.
6. **`user_data` pointer-snapshot sites (Phase 2).** C code that
   snapshots the raw `user_data` pointer and reuses it later
   (CVODE `cv_e_data = cv_user_data` in `cvInitialSetup`; CVLS
   `P_data = cv_user_data` at `CVodeSetLinearSolver`) cannot alias a
   `Box`: the port passes the CURRENT `cv_user_data` box at call time
   instead. Divergent only when `CVodeSetUserData` is called
   mid-integration after the snapshot point — no reference example
   does this, and the Rust behavior matches the documented SUNDIALS
   semantics. Same class: `void*`-returning getters
   (`CVodeGetUserData`, `CVodeGetNonlinearSystemData`) SWAP the box
   with the caller's out-param; the caller must hand it back (via
   `CVodeSetUserData` or a second swap) before the integrator next
   invokes a user callback.
7. **Hoisted callback fn-pointers within one evaluation.** DQ loops
   (`cvLsDQJtimes` retries, dense/band DQJac column loops) copy the
   RHS/jt fn pointer to a local before the loop where C re-reads the
   field each iteration; a callback re-entrantly swapping the fn
   mid-evaluation would take effect one call later than in C. This is
   the locked move-state-into-locals pattern; observable by no valid
   example.
8. **Shared sensitivity parameter array (`SensParams`, Phase 3).** Not a
   divergence but the pattern that AVOIDS one: `CVodeSetSensParams` stores
   the caller's raw pointer in C (`cv_mem->cv_p = p;`) and the internal
   difference-quotient routines (`cvSensRhs1InternalDQ`,
   `cvQuadSensRhs1InternalDQ`) perturb `cv_p[which]` IN PLACE around each
   call to the user's `f`/`fQ` — the callback, reading the same memory
   through `user_data`, sees the perturbed parameter, and that aliasing IS
   the DQ mechanism. The port reproduces it with the handle model:
   `pub type SensParams = Rc<RefCell<Vec<sunrealtype>>>` (`cvodes_impl`),
   `cv_p: Option<SensParams>` (`None` = C `NULL`), and
   `CVodeSetSensParams(cvode_mem, p: Option<SensParams>, pbar, plist)`.
   **Example authors: keep the parameter array in your user data as a
   `SensParams` and hand `CVodeSetSensParams` a CLONE of that same handle**
   — an owned `Vec` copy silently yields zero sensitivities. Callbacks read
   it as `data.p.borrow()[i]` and must never hold that borrow across a
   solver call. `pbar`/`plist` stay borrowed slices: C copies them
   element-wise into `cv_mem`'s own arrays and never writes back. Solver
   internals write the perturbation through `cv_p_set`/`cv_p_get`
   (`cvodes.rs`), which borrow the cell for a single statement, never
   across the callback. Faithful to C, the DQ routines restore `psave`
   only on the success path: a nonzero return from `f`/`fQ` leaves the
   perturbed value in the shared array exactly as the C code does.
   **IDAS (Phase 6) applies the identical contract**: `SensParams` in
   `idas_impl`, `ida_p: Option<SensParams>`,
   `IDASetSensParams(ida_mem, p: Option<SensParams>, pbar, plist)`, and
   `ida_p_set`/`ida_p_get` in `idas.rs` for `IDASensRes1DQ` and
   `IDAQuadSensRhs1InternalDQ`. The `ida_p == NULL` tests in `IDASolve`
   and `IDAInitialSetup` are `ida_p.is_none()`.
9. **Rust source coordinates in logger WARNING lines (Phase 4).** Every
   `*ProcessError(mem, code, __LINE__, __func__, __FILE__, …)` call site
   maps to `line!() as i32` / `file!()`, so the `file:line` field that
   `sunCombineFileAndLine` embeds carries the Rust path, e.g.
   `[WARNING][rank 0][crates/kinsol_rs/src/kinsol_cli.rs:362][kinSetFromCommandLine]`
   where C prints `[…/src/kinsol/kinsol_cli.c:175][kinSetFromCommandLine]`.
   `[ERROR]` lines go to `stderr` (not captured by the reference `.out`
   files), but at the reference logging level 2 the `*_WARNING` branch of
   `*ProcessError` queues through `SUNLogger` to **stdout**, so this field
   IS output-observable. No reference example variant reaches a warning
   path today (verified for kinsol: the only upstream CLI variant is
   `kinRoberts_fp kinsol.m_aa 1`, which is handled and byte-identical).
   Before a variant that trips one is accepted, `tools/verify_examples.sh`
   must strip the `[<file>:<line>]` field symmetrically from both sides in
   `noise_filter()`; the func name, level, rank and message text all match
   C character-for-character and stay diffed.
10. **Missing-vararg substitution (Phase 6).** A few upstream
   `*ProcessError` call sites pass a `MSG_*` format string containing
   `MSG_TIME` (`%g`) with NO vararg for it, so release C prints an
   indeterminate value: IDAS `IDAQuadSetup` uses `MSG_QRHSFUNC_FAILED`
   (`src/idas/idas.c:5181`) and `MSG_QSRHSFUNC_FAILED` (`:5205`) this way.
   The port supplies `ida_tn` — the value every sibling call site passes to
   the same message. This is NOT class 5 (C UB -> deterministic *panic*);
   it is C UB -> deterministic *substituted value*, chosen because the
   alternative (printing garbage) is not expressible in safe Rust. Only
   reachable when the quadrature RHS / quadrature-sensitivity RHS fails
   unrecoverably on the very first step; no reference example does.
   **Phase 7 adds one ARKODE site**: `arkStopTests` uses
   `MSG_ARK_RHSFUNC_FAILED` (`src/arkode/arkode.c:2380`) with no vararg for
   the embedded `MSG_TIME`; the port supplies `ark_mem->tcur`, the value
   every sibling call site passes to the same message (e.g.
   `arkHandleFailure`, `src/arkode/arkode.c:2872`). Reachable only when
   `step_fullrhs` fails after roots were found in the previous step.
11. **Owning callback tokens (Phase 5/6).** C stores a NON-owning
   `CVodeMem`/`IDAMem` pointer as the DQ-Jacobian, Jacobian-times and
   `SUNLinSolSetATimes`/`SUNLinSolSetPreconditioner` token
   (`idas_ls.c:236/:253/:364`, same in `cvode_ls.c`/`ida_ls.c`). Under the
   handle model an `Rc` clone IS the C pointer copy, so those tokens own a
   reference. `*LsFree` clears `*_lmem`, breaking the mem<->lmem cycle, but
   the SUNLinearSolver's `A_data`/`P_data` keeps the solver mem alive: the
   mem and its N_Vectors are reclaimed at `SUNLinSolFree` rather than at
   `*Free`. Resource-lifetime only — no arithmetic or output effect, and
   every reference example frees the linear solver after the integrator.
   Do NOT "fix" it by detaching the token inside `*LsFree`: that would add
   a call C never makes.
12. **ERKStep discrete-adjoint cluster — CLOSED (Phase 7).** Originally
   recorded as a deferred public-API gap: `erkStep_TakeStep_Adjoint`,
   `erkStep_fe_Adj`, `erkStep_SUNStepperReInit` and the public
   `ERKStepCreateAdjointStepper` (`src/arkode/arkode_erkstep.c:1043-1943`)
   were not translated, so `erkStep_Init` installed `erkStep_TakeStep`
   unconditionally. **All four are now ported** into
   `crates/arkode_rs/src/arkode_erkstep.rs`, and the `do_adjoint` branch of
   `erkStep_Init` is restored verbatim —
   `if m.do_adjoint { m.step = Some(erkStep_TakeStep_Adjoint) } else { … }`,
   upstream `:518`'s ternary. The gap is closed; what remains are the two
   binding notes below, both instances of already-numbered classes.
   * **Deviation class 6 (`user_data` cannot alias).** C
     `ERKStepCreateAdjointStepper` ends with
     `SUNAdjointStepper_SetUserData(*adj_stepper_ptr, ark_mem->user_data)`,
     aliasing the FORWARD integrator's `user_data` into the adjoint stepper
     — the forward RHS still dereferences it during
     `SUNAdjointStepper_RecomputeFwd`. A `Box` cannot alias, so the port
     passes `None` and leaves the token with the forward memory, exactly as
     `ARKStepCreateAdjointStepper` does; a caller that needs `adj_f` to see
     user data must hand the adjoint stepper its own copy with
     `SUNAdjointStepper_SetUserData`. `erkStep_fe_Adj` takes
     `adj_stepper.user_data` for the duration of the `adj_f` call and
     restores it on every path.
   * **Deviation class 5 (C UB → deterministic panic), one narrow spot.**
     `erkStep_SUNStepperReInit` forwards `step_mem->f` to `ERKStepReInit`,
     whose Rust signature takes a non-nullable `ARKRhsFn` (the C NULL check
     is handled by the type system), so a missing `f` panics there instead
     of returning `ARK_ILL_INPUT` from the callee. Unreachable: the field is
     set by `ERKStepCreate`/`ERKStepReInit` and never cleared.
   The `nvec` operand lists use `step_mem.cvals`/`step_mem.Xvecs` through the
   module's `erkStep_LinearCombination` helper (matching ERKStep's own C,
   which does the same), not the local operand lists ARKStep's port uses.
   Still true, and the reason this stays a tracked item: **no serial
   reference example exercises the ERKStep adjoint path**, so the compiler
   and a line-by-line reading of the C are the only checks it has had.
13. **Rootfinding state moved into locals for the Illinois search
   (Phase 7).** `arkRootfind` (`crates/arkode_rs/src/arkode_root.rs`)
   `mem::take`s all six rootfinding arrays (`glo`, `ghi`, `grout`,
   `iroots`, `rootdir`, `gactive`) out of `ark_mem.root_mem` for the
   duration of the search and writes them back at the single exit — the
   locked move-state-into-locals pattern (see "Established porting
   patterns"), applied here because the user `g` callback and
   `ARKodeGetDky` run inside that window. In C the arrays stay live in
   `rootmem`, so a `g` that re-entrantly calls `ARKodeGetRootInfo`,
   `ARKodeSetRootDirection` or `arkPrintRootMem` reads defined-but-stale
   data; in Rust those APIs see empty `Vec`s and index-panic or print
   nothing (`nrtfn` is not taken and stays nonzero). No upstream ARKODE
   example queries root state from inside `g`, and all arithmetic
   (alpha doubling/halving, the secant `tmid`, the fracint/fracsub inward
   adjustment, the imax/maxfrac selection) plus the fields written on
   every C return path are identical.
