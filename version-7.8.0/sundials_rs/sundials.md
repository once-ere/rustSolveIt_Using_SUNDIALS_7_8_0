# SUNDIALS_7_8_Rust_port_for_Linux

*SUNDIALS 7.8.0 — pure-Rust port, verified on Linux / Intel-AMD x86-64 / glibc.*

A line-by-line translation of [SUNDIALS](https://github.com/LLNL/sundials)
7.8.0 into safe Rust. The C control flow, constants, tolerances, heuristics,
return codes and argument lists are preserved function by function; the
acceptance criterion is **byte-identical printed output** against the C
reference examples, not "close enough".

> ## ⚠ Platform scope: Linux on x86-64 with glibc — and a note on this file
>
> **Every numerical claim for *this* repository was established on Ubuntu
> 24.04 x86-64, glibc 2.39, gcc 13.3.0, rustc 1.93.1.** The current gate is
> **153 IDENTICAL / 26 reference-side divergences / 20 excluded**, 0 port
> defects, and the deterministic `pow` is bit-exact against the native glibc
> `pow` over 25.9 million measured inputs. The authoritative statements are
> in [`README.md`](README.md), [`current_status.md`](current_status.md) and
> **Part A** of [`VERIFICATION.md`](VERIFICATION.md).
>
> **This guide is inherited, essentially unchanged, from the sibling
> macOS/Apple-Silicon port**, together with the whole crate tree it
> describes. Everything it says about the *translation* — the crate map, the
> API conventions, the handle model, the worked examples, §§1–4 and §§6–8 —
> is platform-independent and accurate here. Where it discusses **Apple
> libm, arm64, the 127 / 52 / 20 tally, or "porting to another platform"
> (§5, §9)**, read it as the inherited discussion of the macOS port, not as a
> statement about this one; §9's checklist is in fact the recipe that *was*
> followed to produce this repository, and `current_status.md` records the
> outcome.
>
> The Rust sources are portable: `std` only, no `unsafe`, no FFI, and no
> `cfg(target_os)` or `cfg(target_arch)` anywhere. What is platform-bound is
> byte-identical **output**: the library and the examples reach `sin`, `cos`,
> `asin`, `acos`, `atan`, `sinh`, `cosh`, `acosh`, `exp` and `ln` through
> `f64` methods that Rust `std` documents as having *unspecified precision*
> and forwards to the **host** libm. On glibc that host libm is the one the
> upstream references were generated with — which is exactly why the gate
> improves from 127 to 153 IDENTICAL here. `pow` is additionally taken off
> the host (§5, §8). `sqrt`, `mul_add`, `ceil`, `round`, `abs` and `copysign`
> are IEEE-754 specified and identical on every target — they are not part of
> the problem, and must not be described as if they were.
>
> Ports scoped to other platforms belong in separate repositories. This one
> contains no conditional compilation and will not grow any — see §9.

## 1. What this is — and is not

| | |
|---|---|
| `unsafe` blocks | 0 (`grep -c unsafe` over `crates/*/src` and `crates/*/examples` = 0) |
| FFI / `libc` / raw pointers | none |
| External crates | none — `Cargo.lock` contains exactly the 7 workspace packages |
| Build warnings | 0 from `cargo build --workspace` after a full recompile |
| Rust edition | 2021, `std` only |
| Version | tracks upstream 7.8.0 (`SUNDIALS_VERSION` = `"7.8.0"`) |
| Conditional compilation | none — 0 `cfg(target_os)`, 0 `cfg(target_arch)` in the tree |
| Verified platform | **Linux on Intel/AMD x86-64**, glibc 2.39, gcc 13.3.0 (Ubuntu 24.04) |
| Distribution coverage | measured: glibc 2.36–2.41 (Debian 12, Ubuntu 24.04, Debian 13, Fedora 41) give the identical gate result; Arch/glibc 2.44 moves 3 LSRK variants; musl out of scope — see `README.md` § "Distribution coverage" |
| Other platforms | builds and unit-tests anywhere Rust targets; byte-identical output **not** claimed off glibc/x86-64 (the macOS/arm64 sibling repo claims it there) |

Roughly 192k lines of Rust across 141 modules, plus 108 translated example
programs covering every serial upstream example for CVODE(S), KINSOL, IDA(S)
and ARKODE.

**Not included.** The port is **serial only**. There are no MPI, GPU
(CUDA/HIP/SYCL/RAJA/Kokkos), KLU, SuperLU_MT/DIST, LAPACK, Fortran-interface
or XBraid backends, and no stub wrappers pretending to be them. Where an
upstream example depends on one of those, it is either excluded from the
verification gate (KLU/SuperLU — 20 variants) or run against the native
equivalent (the `*L` LAPACK examples use the native dense/band solvers).

Vector/matrix/linear-solver support is the serial set: `nvector_serial` and
`nvector_manyvector`; dense, band and sparse `SUNMatrix`; dense, band, PCG,
SPGMR, SPFGMR, SPBCGS and SPTFQMR `SUNLinearSolver`; Newton, fixed-point and
auto `SUNNonlinearSolver`.

## 2. Crate map

`sundials_core` is the shared library; the six solver crates depend on it and
**never on each other**.

```
                       sundials_core (51 modules)
   ┌──────────┬──────────┬────┴─────┬──────────┬──────────┐
 cvode_rs  cvodes_rs  kinsol_rs   ida_rs   idas_rs   arkode_rs
   (12)      (16)        (8)        (8)      (12)      (34)
```

| crate | upstream | provides |
|---|---|---|
| `sundials_core` | `src/sundials`, `src/nvector`, `src/sunmatrix`, `src/sunlinsol`, `src/sunnonlinsol`, `src/sunadaptcontroller`, `src/sundomeigest`, `src/sunmemory`, `src/sunadjointcheckpointscheme` | `SUNContext`, `N_Vector`, `SUNMatrix`, `SUNLinearSolver`, `SUNNonlinearSolver`, `SUNAdaptController`, `SUNDomEigEstimator`, `SUNMemoryHelper`, logger, profiler, error handling, `SUNStepper`, `SUNAdjointStepper`, CLI parsing, math and `printf`-exact formatting |
| `cvode_rs` | `src/cvode` | ODE IVP, Adams/BDF; `_ls`, `_diag`, `_bandpre`, `_bbdpre`, `_proj`, `_resize` |
| `cvodes_rs` | `src/cvodes` | CVODE + forward (FSA) and adjoint (ASA) sensitivity analysis |
| `kinsol_rs` | `src/kinsol` | nonlinear algebraic systems; Newton, Picard, fixed-point with Anderson acceleration |
| `ida_rs` | `src/ida` | DAE IVP (index-1), incl. `IDACalcIC` |
| `idas_rs` | `src/idas` | IDA + FSA and ASA |
| `arkode_rs` | `src/arkode` | ARKStep, ERKStep, SPRKStep, MRIStep, LSRKStep, SplittingStep, ForcingStep, Butcher/MRI/SPRK tables, relaxation, rootfinding |

Every solver crate re-exports the shared `sundials_core` modules at its own
root and offers a flat `prelude`, so an example needs one `use` line.

## 3. Getting started

The crates are not on crates.io. Point Cargo at the workspace:

```toml
[dependencies]
cvode_rs = { path = "…/SUNDIALS_7_8_Rust_port_for_AppleSilicon_macos/crates/cvode_rs" }
```

```rust
use cvode_rs::prelude::*;   // brings in CVode*, N_V*, SUNMat*, SUNLinSol*, fmt_*
```

### A complete program

The Robertson kinetics problem, dense Newton, scalar `rtol` + vector `atol`.
This compiles and runs against the crates as written (it is a trimmed
`crates/cvode_rs/examples/cvRoberts_dns.rs` — that file adds rootfinding, a
stats CSV and a solution check).

```rust
#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

use std::any::Any;
use cvode_rs::prelude::*;

struct UserData { p: [sunrealtype; 3] }

const NEQ: sunindextype = 3;

fn main() {
    /* 1. Context. `&mut Option<SUNContext>` is C's `SUNContext*` out-param. */
    let mut sunctx: Option<SUNContext> = None;
    if SUNContext_Create(SUN_COMM_NULL, &mut sunctx) != 0 { std::process::exit(1); }
    let ctx = sunctx.clone().unwrap();

    /* 2. State vector. Constructors return Option<Handle>; None == C NULL. */
    let y = N_VNew_Serial(NEQ, &ctx).expect("N_VNew_Serial");
    {
        let mut yd = N_VGetArrayPointer(&y).expect("N_VGetArrayPointer");
        yd[0] = 1.0; yd[1] = 0.0; yd[2] = 0.0;
    }

    /* 3. Integrator memory. `cv` is an Rc clone of the same object —
          cloning a handle IS the C pointer copy. */
    let mut cvode_mem = CVodeCreate(CV_BDF, &ctx);
    let cv = cvode_mem.clone().expect("CVodeCreate");

    /* 4. RHS + initial condition; scalar rtol, vector atol. */
    let mut retval = CVodeInit(&cv, f, 0.0, &y);
    let abstol = N_VNew_Serial(NEQ, &ctx).expect("N_VNew_Serial");
    {
        let mut ad = N_VGetArrayPointer(&abstol).expect("N_VGetArrayPointer");
        ad[0] = 1.0e-8; ad[1] = 1.0e-14; ad[2] = 1.0e-6;
    }
    retval |= CVodeSVtolerances(&cv, 1.0e-4, &abstol);

    /* 5. Dense matrix + dense linear solver. `Some(&A)` is C's non-NULL A. */
    let A  = SUNDenseMatrix(NEQ, NEQ, &ctx).expect("SUNDenseMatrix");
    let LS = SUNLinSol_Dense(&y, &A, &ctx).expect("SUNLinSol_Dense");
    retval |= CVodeSetLinearSolver(&cv, &LS, Some(&A));

    /* 6. user_data is Option<Box<dyn Any>>; the callback downcasts it. */
    retval |= CVodeSetUserData(&cv, Some(Box::new(UserData { p: [0.04, 1.0e4, 3.0e7] })));
    if retval != CV_SUCCESS { std::process::exit(1); }

    /* 7. Integrate. `&mut t` is C's `sunrealtype *tret`. */
    let mut t: sunrealtype = 0.0;
    let mut tout: sunrealtype = 0.4;
    for _ in 0..12 {
        let flag = CVode(&cv, tout, &y, &mut t, CV_NORMAL);
        if flag < 0 { eprint!("CVode failed: {}\n", flag); break; }
        let yd = N_VGetArrayPointer(&y).expect("N_VGetArrayPointer");
        /* C: printf("At t = %0.4e      y =%14.6e  %14.6e  %14.6e\n", …) */
        print!("At t = {}      y ={}  {}  {}\n",
               fmt_e(t, 4), fmt_ew(yd[0], 14, 6), fmt_ew(yd[1], 14, 6), fmt_ew(yd[2], 14, 6));
        tout *= 10.0;
    }

    print!("\nFinal Statistics:\n");
    let _ = CVodePrintAllStats(&cv, &SUNFile::Stdout, SUNOutputFormat::SUN_OUTPUTFORMAT_TABLE);

    /* 8. Teardown, in C's order. */
    N_VDestroy(y);
    N_VDestroy(abstol);
    CVodeFree(&mut cvode_mem);
    let _ = SUNLinSolFree(Some(LS));
    SUNMatDestroy(A);
    let _ = SUNContext_Free(&mut sunctx);
}

/* CVRhsFn — same argument names, order and meaning as the C typedef. */
fn f(_t: sunrealtype, y: &N_Vector, ydot: &N_Vector,
     user_data: &mut Option<Box<dyn Any>>) -> i32 {
    let data = user_data.as_mut()
        .and_then(|b| b.downcast_mut::<UserData>())
        .expect("user_data is UserData");
    let (p1, p2, p3) = (data.p[0], data.p[1], data.p[2]);

    let yd = N_VGetArrayPointer(y).expect("N_VGetArrayPointer");
    let (y1, y2, y3) = (yd[0], yd[1], yd[2]);
    drop(yd);                      // release the borrow before touching ydot

    let mut dd = N_VGetArrayPointer(ydot).expect("N_VGetArrayPointer");
    dd[0] = -p1 * y1 + p2 * y2 * y3;
    dd[2] = p3 * y2 * y2;
    dd[1] = -dd[0] - dd[2];
    0
}
```

For ARKODE the shape is the same with `ARKStepCreate(fe, fi, T0, &y, &ctx)` /
`ARKodeSStolerances` / `ARKodeSetLinearSolver` / `ARKodeEvolve(&mem, tout, &y,
&mut t, ARK_NORMAL)` — see `crates/arkode_rs/examples/ark_analytic.rs`.

Running the shipped examples:

```sh
cargo run -p cvode_rs --example cvRoberts_dns
cargo build --workspace                 # library build, warning-free
cargo test  --workspace --lib           # unit tests
tools/verify_examples.sh cvode_rs       # or `all`, or `list`
```

The first three work on any platform Rust targets. The fourth is a POSIX
`bash` script that additionally needs the read-only upstream SUNDIALS 7.8.0 C
tree as its **parent directory**, and its verdicts are only meaningful on
macOS/arm64 — see the platform-scope callout at the top and §9.

## 4. Translating C knowledge

| C | Rust | note |
|---|---|---|
| `CVodeInit`, `CV_SUCCESS`, `IDA_MEM_NULL`, … | same identifiers | names, values and sign conventions preserved (`0` success, negative fatal, positive recoverable) |
| `N_Vector`, `SUNMatrix`, `CVodeMem`, … | `Rc<…>` handles | **cloning a handle is the C pointer copy**; `Rc::ptr_eq` is C pointer equality |
| `T *out` output parameter | `&mut T` in the same position, same name | e.g. `CVode(&cv, tout, &y, &mut t, CV_NORMAL)` |
| function returning an object pointer | `Option<Handle>` | `None` is C `NULL`; every constructor that C can fail with NULL returns `Option` |
| nullable C function pointer argument | `Option<FnType>` | `CVodeRootInit(&cv, 2, Some(g))`, `ARKStepCreate(None, Some(f), …)` |
| `void *user_data` | `Option<Box<dyn Any>>` | set with `CVodeSetUserData(&cv, Some(Box::new(data)))`; callbacks receive `&mut Option<Box<dyn Any>>` and do `.as_mut().and_then(\|b\| b.downcast_mut::<UserData>())` |
| `void *` out-param getters | **swap** getters | `CVodeGetUserData(&cv, &mut out)` moves the box out; hand it back (via `CVodeSetUserData` or a second swap) before the integrator calls a user callback again |
| `printf("%e"/"%f"/"%g")` | `fmt_e/fmt_f/fmt_g(x, prec)`, width forms `fmt_ew/fmt_fw/fmt_gw(x, width, prec)` | Rust's `{:e}` does **not** match C `printf`; the fmt helpers do (default precision 6, `e±dd` with ≥2 exponent digits, `%g` trailing-zero stripping, lowercase `inf`/`nan`) |
| `SUN_FORMAT_E/G/SG` macros | `sun_format_e/g/sg(x)` | `"% .15e"` / `"%.15g"` / `"%+.15g"` |
| `N_Vector *` array | `&[N_Vector]` | elements are `Rc` clones; `N_Vector **` → `&[Vec<N_Vector>]` |
| `NV_DATA_S(v)`, `SM_ELEMENT_D(A,i,j)` | accessor functions returning `RefMut` guards | drop the guard before another op touches the same object |
| `argv` | `&[String]` with `argv[0]` = program name | `atoi` semantics preserved (unparsable → 0); CLI keys are bare `<solverid>.<key>` tokens, no leading dashes |

Two shared-array conventions have no direct C analogue and are worth reading
before writing sensitivity code or reaching for a getter:

**`SensParams` (sensitivity parameters).** In C, `CVodeSetSensParams` stores
*the caller's pointer*, and the internal difference-quotient routines perturb
`p[which]` in place around each user RHS call — the callback reading the same
memory through `user_data` sees the perturbed value, and that aliasing *is* the
DQ mechanism. The port reproduces it with
`pub type SensParams = Rc<RefCell<Vec<sunrealtype>>>`
(`cvodes_impl`, `idas_impl`) and
`CVodeSetSensParams(mem, p: Option<SensParams>, pbar, plist)` /
`IDASetSensParams(…)`. **Keep the parameter array in your user data as a
`SensParams` and pass a `clone()` of that same handle** — handing over an owned
`Vec` copy silently produces zero sensitivities. Callbacks read it as
`data.p.borrow()[i]` and must never hold that borrow across a solver call.
`pbar`/`plist` stay plain slices, because C copies them element-wise.
Worked usage: `crates/cvodes_rs/examples/cvsAdvDiff_FSA_non.rs`.

**`user_data` swap-getters.** `CVodeGetUserData` and friends return a `void*`
in C without transferring ownership. A `Box` cannot alias, so the Rust version
`std::mem::swap`s the box into your out-parameter. The integrator then holds
`None` until you give it back. Same rule for
`CVodeGetNonlinearSystemData`.

## 5. Fidelity, and how it is verified

**The gate is byte-identity.** `tools/verify_examples.sh` parses the upstream
`CMakeLists.txt` files for every `(example, argv)` tuple — **199 variants** —
builds the Rust examples in release, runs each with the exact argv, and diffs
stdout against `../examples/<solver>/<serial dir>/<name>[_<args>].out`. Only
genuinely machine-dependent lines (`Total run time`, `CPU time`, wall clock)
are filtered, and symmetrically from both sides. A variant passes only at zero
diff lines. `tools/verify_examples.sh list` regenerates the tuple set and
cross-checks that no shipped `.out` file is unclaimed.

**Results, from `tools/verify_examples.sh all` run for this document on
macOS/arm64 (Apple clang, Apple libm). Every number in the table below is a
result on that platform and is not claimed on any other:**

| crate | variants | byte-identical | documented exception | excluded (KLU/SuperLU) | swept? |
|---|---:|---:|---:|---:|---|
| `cvode_rs`  |  24 | 12 |  9 | 3 | yes |
| `cvodes_rs` |  39 | 23 | 10 | 6 | yes |
| `kinsol_rs` |  22 | 19 |  1 | 2 | yes |
| `ida_rs`    |  14 | 10 |  1 | 3 | yes |
| `idas_rs`   |  22 | 12 |  4 | 6 | yes |
| `arkode_rs` |  78 | 51 | 27 | 0 | yes |
| **total**   | **199** | **127** | **52** | **20** | |

All six crates are swept. Of the 179 variants actually run (199 minus the 20
KLU/SuperLU exclusions), **127 are byte-identical to the shipped reference
output and the remaining 52 are the documented reference-side exceptions of §6
— zero are unexplained, and zero are port defects.** Every one of the 52 was
root-caused against a locally built pristine upstream C binary; see
`VERIFICATION.md` for the per-variant evidence.

Read those 52 honestly: they are not "close enough" passes. Each is a case
where the shipped `.out` cannot be reproduced by its own C source on this
machine — most often because the file was generated on a glibc host whose
`sin`/`exp`/`pow` differ from Apple libm by one ulp somewhere inside the
integration feedback loop, or because the reference predates a formatting
change in SUNDIALS itself. The bar for accepting one is byte-identity against
a pristine local upstream C build, i.e. the same standard a C user on this
machine would be held to.

The ARKODE sweep turned up the last real port defect, and it was in this
port's own deterministic `pow`: `pow_exp_inline` closed with an unfused
`scale + scale * tmp`, where the glibc build the references came from
contracts that into a single FMA. On near-midpoint results the unfused form
rounds the other way, which forked `ark_robertson` into a 228-line diff.
Fusing it (`scale.mul_add(tmp, scale)`) turned eight ARKODE variants
byte-identical, with zero regressions across all 199.

**Build and test state, verified first-hand for this document** (macOS arm64,
Apple clang toolchain):

* `cargo clean -p` for all seven crates, then `cargo build --workspace` →
  all seven compile, **zero warnings**.
* `cargo test --workspace --lib` → **25 passed, 0 failed** (23 in
  `sundials_core` — `pow` bit-exactness, `printf` formatting vectors, CLI
  `atoi` semantics, nvector ops; 1 each in `cvodes_rs` and `idas_rs` covering
  the DQ-sensitivity parameter aliasing). The other four crates carry no lib
  tests — the example gate is their regression suite.
* Release example builds for all six solver crates: zero warning lines.

Those two commands were re-run on **Windows 11 x86-64** on 2026-08-09 as a
portability check: `cargo build --workspace` is warning-free there too and
`cargo test --workspace --lib` gives the same 25 passed / 0 failed. That is
the whole of what is known about non-Apple platforms. The example gate was
**not** run there and its result is not predicted; see §9.

Two engineering decisions do most of the work behind byte-identity:

1. **Deterministic `pow`.** `SUNRpowerR` does not call platform libm. It runs a
   port of the ARM optimized-routines `pow` — the same algorithm glibc ≥ 2.28
   uses, i.e. the libm that generated the reference outputs. Apple's `pow` is
   1 ulp off on rare arguments inside the step-size heuristics, which was
   enough to fork three CVODE examples before the port; all three are
   byte-identical with it.
2. **Exact `printf`.** All float output goes through `fmt_e/fmt_f/fmt_g`,
   which reimplement C conversion specifiers rather than approximating them.

**Read decision 1 precisely, because it is the source of the most common
misreading of this project.** The ported routine is ordinary host-independent
Rust; the "ARM optimized-routines" provenance is an accident of where that
algorithm was written, not a statement about the CPU it runs on. It removes
the host libm from the `pow` path **and only from the `pow` path**. Every
other host-dependent function the library and the examples evaluate — `sin`,
`cos`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `acosh`, `exp`, `ln` — is
still `f64`'s method, which resolves to the host libm. Nothing analogous was
done for them, and that is exactly why the byte-identity results above are
macOS/Apple-Silicon results rather than portable ones. (`sqrt`, `mul_add`,
`ceil`, `round`, `abs` and `copysign` need no such treatment: IEEE-754
specifies them exactly and Rust guarantees the specified result on every
target.)

One non-libm substitution belongs in the same paragraph: three ARKODE LSRK
dominant-eigenvalue examples reproduce the BSD/glibc `rand()` TYPE_3
generator in Rust, because their reference outputs depend on that exact
sequence. See `NOTICE`.

## 6. Known deviations

### 6a. Port-side accepted deviations

`ARCHITECTURE.md` fixes 13 numbered classes, each adversarially reviewed and
argued unobservable on any path a valid serial example takes. In plain terms:

1. **Kept failure-path checks** — where release C compiles out `SUNAssert`
   and proceeds on misuse, the port may keep the check and return the error
   code. Observable only on invalid usage.
2. **Ownership snapshots** — `SUNMemoryHelper_Wrap`/`_Alias` own their buffer
   instead of aliasing a raw pointer; no write-through to the original.
3. **C-locale whitespace** — `atoi`/`atol`/`SUNStrToReal` skip ASCII
   whitespace only, never Unicode.
4. **Unsigned wrap** — C `size_t` counter arithmetic that can underflow maps
   to `wrapping_sub`, not a panicking `-=`.
5. **C undefined behaviour → deterministic panic** — NULL deref, out-of-bounds
   and double-free panic at the same site instead of being undefined. One
   *named exception*: `MRIStepCoupling_Alloc` allocates `G` rows one column
   short, so upstream reads one element past a `calloc`'d row on the MRISR
   embedding stage — a *reachable* path with the default adaptive ImEx MRI
   tables. The port returns zero there (what zeroed `calloc` storage gives C)
   rather than panicking. Not generalized anywhere else.
6. **`user_data` pointer snapshots** — C sites that cache the raw `user_data`
   pointer and reuse it later pass the *current* box instead. Divergent only
   if you call `Set*UserData` mid-integration after the snapshot point. Same
   class: the swap-getters described in §4.
7. **Hoisted callback function pointers** — DQ loops copy the RHS/Jtimes fn
   pointer to a local before the loop where C re-reads the field each
   iteration. A callback that swaps the fn re-entrantly mid-evaluation would
   take effect one call later.
8. **Shared sensitivity parameters (`SensParams`)** — not a divergence but the
   pattern that avoids one; see §4.
9. **Rust source coordinates in logger WARNING lines** — the `[file:line]`
   field of a `[WARNING]` line carries the Rust path, not the C path. This
   *is* output-observable at the reference logging level; no reference variant
   reaches a warning path today, and the harness must strip the field
   symmetrically before any variant that does is accepted.
10. **Missing-vararg substitution** — three upstream `*ProcessError` sites pass
    a format string containing `%g` with no vararg, so release C prints an
    indeterminate value. The port supplies the value every sibling call site
    passes (`ida_tn`, `ark_mem->tcur`). Reachable only on unrecoverable
    first-step RHS failures.
11. **Owning callback tokens** — C stores a non-owning solver-mem pointer as
    the DQ-Jacobian / ATimes / preconditioner token; under the handle model an
    `Rc` clone owns a reference, so the solver mem is reclaimed at
    `SUNLinSolFree` rather than at `*Free`. Lifetime only, no arithmetic
    effect; free the linear solver after the integrator, as every reference
    example does.
12. **ERKStep discrete adjoint not translated** — `ERKStepCreateAdjointStepper`
    and its `TakeStep_Adjoint` cluster are a tracked public-API gap. The
    ARKStep adjoint cluster *is* ported. No reference example exercises the
    ERKStep one.
13. **Rootfinding state moved into locals** — `arkRootfind` moves its six
    arrays out of the mem for the duration of the Illinois search. A `g`
    callback that re-entrantly queries root state would see empty vectors
    where C sees stale-but-defined data. No upstream example does this; all
    arithmetic and all fields written on C's return paths are identical.

### 6b. Reference-output exception classes

52 variants do not match their shipped `.out` byte-for-byte. In every case the
port was shown equal to a **locally built pristine upstream C binary**
(CMake Release, Apple clang, `-O3 -DNDEBUG -ffp-contract=off`, logging level 2,
error checks off, profiling off, serial — the upstream Release defaults), and
that is the bar actually used here.

**These classifications are macOS/Apple-Silicon diagnoses.** "The reference
does not match" is a statement about *this* host's libm versus the generating
host's; the counts, the class assignments and the per-variant evidence would
all have to be re-derived on any other OS or architecture, against a pristine
C build made there. Do not carry this table to another platform.

| class | count | what it is |
|---|---:|---|
| `ref-libm` | 32 | The shipped `.out` embeds the generating host's glibc `sin`/`cos`/`exp`/`log`/`atan` rounding *inside the integration feedback loop*, so a 1-ulp libm difference flips a step-size or order decision downstream. |
| stale upstream reference | 18 | The shipped `.out` cannot be produced by the C source it ships with, on any platform. Counted here are the 14 rows tagged `stale-ref` in `VERIFICATION.md` plus the four written out longhand: `cvPendulum_dns` (`.out` anomaly), `cvRoberts_dns_negsol` (stale line 20), `kinRoboKin_dns` (`SUN_TABLE_WIDTH` 28), `idasAkzoNob_ASAi_dns` (trailing whitespace stripped). |
| LAPACK → native | 2 | `cvRoberts_dnsL` / `cvsRoberts_dnsL` use the native dense solver instead of LAPACK; different factorization arithmetic gives last-digit drift. Proven by rebuilding the C example with `SUNLinSol_Dense` substituted — output then matches the port byte-for-byte. |
| **total** | **52** | matches the 52 of §5; the per-variant rows are in `VERIFICATION.md` |

The `ref-libm` set is mutually inconsistent in the upstream tree: the three
diurnal-family references demand *three different* `sin` implementations
(glibc ≥ 2.28 for one, correctly-rounded pre-2.27 glibc for the others), so no
single libm — and therefore no faithful port on any one machine — can match all
of them at once. For `idaFoodWeb_bnd` the chain is closed completely:
substituting one correctly-rounded `sin` value at the single mesh argument
`FOURPI*15/19` into the pristine C build reproduces the shipped `.out`
byte-for-byte.

**The blunt version:** several `.out` files shipped with SUNDIALS 7.8.0 cannot
be reproduced by their own C source on this machine, and some cannot be
reproduced on any machine — `cvPendulum_dns.out` and `cvsPendulum_dns.out`
print two different exponent widths from one `%8.2e` conversion;
`idasAkzoNob_ASAi_dns.out` has been trailing-whitespace-stripped while its
sibling has not; `kinRoboKin_dns.out` uses a `SUN_TABLE_WIDTH` the shipped
header contradicts. Where the shipped reference and the shipped source
disagree, this port follows the source, and records the evidence rather than
tuning the example to match. Every case is written up per-variant in
`VERIFICATION.md` with the measurements behind it.

## 7. Repository layout

```
SUNDIALS_7_8_Rust_port_for_AppleSilicon_macos/
├── Cargo.toml              workspace: 7 members
├── crates/
│   ├── sundials_core/src/  51 modules — one per upstream C file
│   │   └── sundials_utils_cvec.txt   1500+ C-printf vectors (macOS/arm64-generated, committed)
│   ├── cvode_rs/           src/ (12 modules) + examples/ (18)
│   ├── cvodes_rs/          src/ (16) + examples/ (26)
│   ├── kinsol_rs/          src/  (8) + examples/  (9)
│   ├── ida_rs/             src/  (8) + examples/  (8)
│   ├── idas_rs/            src/ (12) + examples/ (13)
│   └── arkode_rs/          src/ (34) + examples/ (34)
├── tools/verify_examples.sh   the acceptance harness (bash; macOS/arm64-scoped)
├── logs/                      harness output (git-ignored)
├── README.md                  front page + the platform-scope statement
├── ARCHITECTURE.md            handle model, locked patterns, deviation classes
├── VERIFICATION.md            per-variant matrix + evidence for every exception
├── PROGRESS.md                per-file port status
├── STATUS.md                  resume snapshot
├── POW_FMA_EXACTNESS.md       how exact the deterministic pow is, and its limits
├── LICENSE                    upstream SUNDIALS BSD-3-Clause, verbatim
├── NOTICE                     third-party notices (SUNDIALS, ARM/musl pow)
└── CLAUDE.md                  the rules the port was built under
```

The directory name carries the platform because the *evidence* does: a tree
of the same sources verified against a different libm is a different project,
with a different `VERIFICATION.md`.

Module naming is mechanical: one Rust module per upstream C file, named after
its base name (`src/cvodes/cvodes_nls_stg1.c` →
`crates/cvodes_rs/src/cvodes_nls_stg1.rs`). `*_impl.h` and the public
`include/<pkg>/*.h` content folds into the matching module; `.def` X-macro
tables become `const` table data inside the module that includes them.

**Where to look for what**

| question | file |
|---|---|
| Can I trust variant *X*? | `VERIFICATION.md` — per-variant row plus evidence |
| Why does the API look like this? | `ARCHITECTURE.md` — handle model, locked patterns |
| Is file *Y* ported? | `PROGRESS.md` |
| How is "verified" defined? | `tools/verify_examples.sh` |
| What is a real call sequence? | `crates/*/examples/` |

## 8. Licensing and provenance

**This is a derivative work of SUNDIALS**, distributed by Lawrence Livermore
National Laboratory under the **BSD-3-Clause** licence
(Copyright © 2002–2026 Lawrence Livermore National Security, Southern
Methodist University, University of Maryland Baltimore County, and the SUNDIALS
contributors). The algorithms, structure, identifiers, comments and reference
outputs come from the upstream 7.8.0 tree; the upstream `LICENSE` and `NOTICE`
files sit at the root of that tree, and this workspace lives inside it, at
`sundials-7.8.0/SUNDIALS_7_8_Rust_port_for_AppleSilicon_macos`.
BSD-3-Clause obligations — retaining the copyright notice and disclaimer, and
not using LLNL's or the contributors' names to endorse this port — apply to
any redistribution. This workspace carries its own verbatim copy of the
upstream `LICENSE` at its root, alongside a `NOTICE` covering both upstream
components.

**One third-party component.** The deterministic `pow` in
`crates/sundials_core/src/sundials_math.rs` (the block starting at the
"Deterministic double-precision pow" banner, ~line 184, and `pow_glibc` at
line 611) is a port of the **ARM optimized-routines `pow` via musl
`src/math/pow.c`**, Copyright © 2018 Arm Limited,
`SPDX-License-Identifier: MIT`. The attribution is carried in the source
comment at the head of that block. It is used instead of `f64::powf` for the
byte-identity reason given in §5.

Despite the name, that component is **not** the Apple-Silicon-specific part of
this project — it is portable Rust with no architecture-conditional code, and
the project is scoped to macOS/arm64 for the unrelated reason set out in the
callout at the top and in §9. Its FMA-contraction map was nonetheless
*measured* on arm64; `POW_FMA_EXACTNESS.md` states what that does and does not
establish elsewhere.

This port is **not** an LLNL product and is not endorsed by the SUNDIALS
project.

## 9. Porting this work to another platform

Nothing in the Rust source has to change in order to *compile* on another OS
or architecture: there is no `unsafe`, no FFI, no external crate and no
conditional compilation, and the workspace builds warning-free and passes its
25 unit tests on Windows 11 x86-64 as well as on macOS/arm64.

What does not transfer is the evidence. To make the byte-identity claim hold
elsewhere, three things must be redone on the target platform:

1. **Re-establish the libm baseline.** The host-dependent call set is exactly
   `sin`, `cos`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `acosh`, `exp` and
   `ln` — the `f64` methods Rust `std` documents as having *unspecified
   precision*. There are **5** such call sites in library code (`SUNRexp` in
   `sundials_math.rs`; `SUNRlog`, `SUNRsinh`, `SUNRcosh`, `SUNRacosh` in
   `arkode_lsrkstep.rs`) and **209** in the examples, concentrated in
   `cv/cvsPendulum_dns`, `ida/idas/idasSlCrank*`, `cv/cvsDiurnal_kry` and
   `ark_kepler`. `sqrt`, `mul_add`, `ceil`, `round`, `abs` and `copysign` are
   IEEE-754 specified and need nothing done to them; `powf`/`powi` are called
   nowhere. Decide which implementation the port should reproduce on the
   target and, where the host's differs, port those functions the way `pow`
   was ported here (`sundials_math.rs`, `POW_FMA_EXACTNESS.md`) — including
   re-measuring the FMA-contraction map against an oracle built natively on
   *that* target rather than trusting the arm64 measurements.
2. **Re-run and re-classify the gate.** The 52 documented exceptions of §6b
   are Apple-libm diagnoses. On another host a different set of variants
   diverges; each one must be root-caused against a pristine upstream C binary
   built on that host, with the classification (`ref-libm`, `stale-ref`,
   LAPACK→native) argued from measurement. Never tune an example to match a
   reference.
3. **Re-host the harness.** `tools/verify_examples.sh` needs POSIX `bash`,
   `cargo`, and the read-only upstream SUNDIALS 7.8.0 C tree as its parent
   directory. On Windows it needs a POSIX environment (Git Bash, MSYS2, WSL)
   or a rewrite; it will not run under `cmd.exe` or PowerShell.
4. **Re-generate the `printf` corpus, or decide not to.**
   `crates/sundials_core/src/sundials_utils_cvec.txt` holds 1500+ conversion
   vectors produced by one host's C `printf`, committed and pulled in with
   `include_str!`. Because it is committed data rather than a live call, the
   `matches_c_printf_vectors` test is portable evidence *generated* on
   macOS/arm64 — it passes unchanged on Windows x86-64. Re-generate it only if
   the target's `printf` is believed to differ; `fmt_e/fmt_f/fmt_g` themselves
   contain no host calls.

Those ports belong in their own repositories. This one deliberately contains
no platform conditionals, so that "what was verified, and where" stays a
property of the repository rather than of a build flag.
