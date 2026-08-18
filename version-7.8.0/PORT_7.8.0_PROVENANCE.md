# Porting rustSolveIt from SUNDIALS 7.7.0 to SUNDIALS 7.8.0

This repository is `once-ere/rustSolveIt` with its numerical engine
replaced. Everything else — the physics, the command language, the
notebooks, the quantum and special-function libraries — is unchanged,
and this file is the evidence that it is unchanged.

*Date of the port: 17 August 2026.*

---

## 1. What actually moved

| | before (`once-ere/rustSolveIt`) | after (this repository) |
|---|---|---|
| engine | `once-ere/sundials_rs@faabb7f` | `once-ere/SUNDIALS_7_8_Rust_port_for_Linux@780b916` |
| SUNDIALS version translated | 7.7.0 | **7.8.0** |
| vendored files | 394 | **2,929** |
| crates | `sundials_core`, `cvode_rs`, `arkode_rs` (+3 unused) | `sundials_core`, `cvode_rs`, `cvodes_rs`, `ida_rs`, `idas_rs`, `kinsol_rs`, `arkode_rs` |
| first-party files touched | — | `physical_object/src/integrate.rs` and four examples |

The engine lives at `sundials_rs/` exactly as before, still excluded
from the outer workspace (`Cargo.toml` `exclude`), still read-only.

**Byte-identity of the vendored copy.** Reproduce it:

```bash
git clone https://github.com/once-ere/SUNDIALS_7_8_Rust_port_for_Linux.git
diff -rq --exclude=.git --exclude=target \
     SUNDIALS_7_8_Rust_port_for_Linux sundials_rs        # silent
```

Content hash of the vendored tree (every file, sorted, SHA-256 of the
SHA-256 list):

```
e03471994fef7fa5c784ef4f82a54f9c24b5f56b4d46304976a5be17a0e1594a
```

---

## 2. The API delta, call by call

The 7.8.0 translation is *more* faithful to the C API than its
predecessor was. Where 7.7.0 exposed a convenient Rust shape, 7.8.0
models C's opaque pointer, its `NULL` return and its
`free(p); p = NULL;` teardown. That is the whole reason first-party
code had to change: not one constant, tolerance, or heuristic moved.

| concern | 7.7.0 | 7.8.0 | why it changed |
|---|---|---|---|
| context | `SUNContext_Create() -> SUNContext` | `SUNContext_Create(SUN_COMM_NULL, &mut Option<SUNContext>) -> SUNErrCode` | C's signature is `(SUNComm, SUNContext*)` returning an error code |
| vectors | `N_VNew_Serial(n, &ctx) -> NVector` | `-> Option<N_Vector>` | C returns `NULL` on failure |
| vector data | `y.data` — a public `Vec<f64>` | `N_VGetArrayPointer(&y) -> Option<RefMut<Vec<f64>>>` | C's `sunrealtype*`; the guard is the borrow |
| vector type | `struct NVector { data }` | `N_Vector = Rc<_generic_N_Vector>` with an ops table | C's `N_Vector` is a handle with a vtable |
| matrix / LS | `SUNDenseMatrix(..) -> SUNMatrix` | `-> Option<SUNMatrix>` (same for `SUNLinSol_Dense`) | `NULL` return |
| solver memory | `CVodeCreate(..) -> CVodeMem` | `-> Option<CVodeMem>`, `CVodeMem = Rc<RefCell<CVodeMemRec>>` | `NULL` return; shared handle |
| every entry point | `&mut CVodeMem` | `&CVodeMem` | C passes the pointer; mutation is interior |
| attach LS | `CVodeSetLinearSolver(&mut m, ls, Some(a))` | `CVodeSetLinearSolver(&m, &ls, Some(&a))` | borrows, not moves |
| stepping | `CVode(&mut m, tout, &mut y, &mut t, task)` | `CVode(&m, tout, &y, &mut t, task)` | `yout` is written through the handle |
| RHS callback | `fn(f64, &NVector, &mut NVector, &mut UserData)` | `fn(sunrealtype, &N_Vector, &N_Vector, &mut Option<Box<dyn Any>>)` | matches `CVRhsFn` |
| root callback | `fn(f64, &NVector, &mut [f64], &mut UserData)` | `fn(sunrealtype, &N_Vector, &mut [sunrealtype], &mut Option<Box<dyn Any>>)` | matches `CVRootFn` |
| SPRK creation | `SPRKStepCreate(Some(f1), Some(f2), ..)` | `SPRKStepCreate(f1, f2, ..)` | C requires both; the type system now says so |
| no-warn root | `CVodeSetNoInactiveRootWarn(&mut m)` (void) | returns `i32` | C returns a flag; it is now checked |
| teardown | `CVodeFree(mem)` | `CVodeFree(&mut Some(mem))`, plus `SUNLinSolFree`, `SUNMatDestroy`, `N_VDestroy`, `SUNContext_Free` | C's blank-the-pointer idiom |
| printf helpers | `fmt_e(x, width, prec)` | `fmt_e(x, prec)` / `fmt_ew(x, width, prec)` | the padded and unpadded conversions are now separate functions, as in C's `%.*e` vs `%*.*e` |

ARKODE moved in lockstep: `ARKodeMem = Rc<RefCell<…>>`, every
`ARKodeSet*`/`ARKodeGet*`/`ARKodeEvolve`/`ARKodeReset` takes
`&ARKodeMem`, and `ARKodeEvolve` takes `yout: &N_Vector`.

### 2.1 The one rule the new handle model imposes

`N_VGetArrayPointer` hands back a `RefMut` — a **live borrow** of the
vector's payload. Holding it across a solver call that touches the same
vector would panic at runtime. So every access in `integrate.rs` is
scoped, through two helpers that take the guard, do the work, and drop
it:

```rust
fn with_data<R>(v: &N_Vector, f: impl FnOnce(&[f64]) -> R) -> Option<R>;
fn with_data_mut<R>(v: &N_Vector, f: impl FnOnce(&mut [f64]) -> R) -> Option<R>;
```

`None` means "not a serial vector" and is turned into a named `Err`,
never an `unwrap` — hard rule 5 (missing symbols are reported, not
invented) applies to missing *data pointers* too.

### 2.2 What did **not** change

Nothing in the algorithm. The Zeno burst counter, the anti-tunneling
`hmax`, the quaternion renormalization threshold `QUAT_RENORM_TOL`, the
event loop's `CV_ROOT_RETURN` handling, the end-of-interval penetration
sweep, the separability gate on SPRK, every tolerance and every
arithmetic order are line-for-line what they were. The diff is 235
added and 86 removed lines in a 1,163-line file, and every one of them
is a signature, a guard scope, or a comment.

The four examples changed only where they printed:
`fmt_e(x, 0, 6)` became `fmt_e(x, 6)` and `fmt_f(x, 14, 8)` became
`fmt_fw(x, 14, 8)` — 21 call sites, chosen so the emitted bytes are
identical (7.7.0's `pad` and 7.8.0's `fmt_*w` are the same
right-justification).

---

## 3. Evidence that the physics did not move

Every artifact below is in [`evidence/port-7.8.0/`](evidence/port-7.8.0).
Each was produced by running the *same input* against both trees and
diffing the output.

### 3.1 The six self-checking physics examples — byte-identical

```bash
for ex in kepler_orbit outer_solar_system tumbling_body \
          charged_in_b_field newtons_cradle bouncing_ball_restitution; do
  cargo run -q -p physical_object --release --example $ex
done
```

`diff examples-7.8.0.log examples-7.7.0-baseline.log` is **empty**.
That includes the anchors the project pins by name:

| anchor | value under 7.7.0 | value under 7.8.0 |
|---|---|---|
| Pluto at t = 500,000 d | `x = 31.78592516  y = 38.63618957  z = 3.19279415` | identical |
| outer-solar-system energy drift | `7.835809e-07` | identical |
| outer-solar-system internal steps | `12581` | identical |
| Kepler e = 0.6, `\|dA\|/\|A\|` (Runge–Lenz) | `1.131858e-07` | identical |
| tumbling body `\|dL\|/\|L\|` | `0.000000e+00` | identical |
| gyroradius, measured | `1.000000` | identical |
| cradle final velocities | `[0, 0, 0, 0, 1]` | identical |
| bounce TOI | `0.9486832980505138` | identical |

### 3.2 The twelve collision scripts — byte-identical

```bash
for f in scripts/collisions/*.posim; do posim --script "$f"; done
```

`diff collision-scripts-7.8.0.log collision-scripts-7.7.0-baseline.log`
is **empty** (367 lines of output, 12 scripts).

### 3.3 All 59 dynamic notebooks — byte-identical

Every notebook in `dynamic_notebooks/`, including the 34 Routh problems,
run in batch mode. After normalising the two things that are genuinely
not deterministic — the OS-assigned scene port and the working
directory — `diff dynamic-notebooks-7.8.0.log
dynamic-notebooks-7.7.0-baseline.log` is **empty**. Every closed-form
check each notebook makes against the integrator still lands on the same
digits.

### 3.4 The test suite — 568 passed, 0 failed

**At the moment of the port**, unchanged in count and in composition:
40 `physical_object` lib + 19 collision + 9 conservation + 109 posim +
92 quantum + 233 special_functions + 11 vendored identities +
55 doctests. Log: `cargo-test-workspace.log`.

That equality of counts is the point of this section: the engine swap
added no tests and removed none. The suite has since grown to **592**
with the work that wired CVODES, IDA, IDAS and KINSOL into the
simulator (§5) — a later change, on top of a port that moved nothing.

### 3.5 Build cleanliness

`cargo build --workspace --all-targets` produces **no warnings**.
Every crate root still carries `#![forbid(unsafe_code)]` and
`#![deny(warnings)]`. `Cargo.lock` still lists only local crates:

```
arkode_rs 7.8.0   cvode_rs 7.8.0   sundials_core 7.8.0
physical_object 0.1.0   posim 0.1.0   quantum 0.1.0
spec_math 0.1.6   special_functions 0.1.0
```

Zero entries from crates.io — the dependency rule holds.

---

## 4. A finding, not a regression

While recording the browser videos (§ `videos/`) one scenario drifted
3.2 % in energy through 25 collisions, where the corresponding Rust
regression test asserts `|dE|/E < 1e-6`. The cause is **not** the
solver: posim's default `system.g_constant` is 1, so three bodies
rattling in a small box also attract each other, and the softened
pairwise force at `softening = 1e-6` is nearly singular when their
surfaces touch. The test's helper `free_system` sets `G = 0` on
purpose, and its comment says why: *"pure free flight + collisions, so
every outcome is exact."*

Setting `set system.g_constant = 0` reproduces the test exactly —
`|dE/E| = 5.118e-16` through 36 collision events. Both trees agree to
the last bit on both variants (`7.671422387443062` with gravity,
either version).

This is worth writing down because it is the sort of thing that looks
like a solver regression and is not: **a conservation claim carries its
system settings with it.**

---

## 5. What is new in this repository

| path | what |
|---|---|
| `sundials_rs/` | the 7.8.0 engine, with its own `VERIFICATION.md`, `differences/`, `evidence/`, `c-results/` and `rust-results/` — every one of its examples diffed against the upstream C reference |
| `recorder/` | records a posim run into a self-contained browser player |
| `videos/*.html` | three recorded videos, openable offline, no CDN |
| `videos/scenes/*.posim` | the scripts they were recorded from |
| `evidence/port-7.8.0/` | the logs quoted above |
| `SolveIt.md` / `.tex` / `.pdf` | the full solution guide with 16 additional worked examples |
| `grammar.md` / `.tex` / `.pdf` | the language spec, updated for 7.8.0, with the video section |
| `ARCHITECTURE.md`, `CLAUDE.md` | rewritten for this repository |

The 7.8.0 engine also brings four solver families the 7.7.0 vendoring
did not include — **CVODES**, **IDA**, **IDAS** and **KINSOL**.
`physical_object` does not use them yet; they are present, they build,
and they carry their own verified examples.

---

## 6. Reproducing the whole thing from scratch

```bash
git clone https://github.com/once-ere/rustSolveIt_Using_SUNDIALS_7_8_0.git
cd rustSolveIt_Using_SUNDIALS_7_8_0/version-7.8.0
cargo build --workspace --all-targets      # must be warning-free
cargo test  --workspace                    # 592 passed
cargo run -p physical_object --release --example outer_solar_system
python3 recorder/src/record_video.py videos/scenes/kepler_ellipse.posim \
        -o /tmp/kepler.html --frames 360 --dt 0.02
```

No network access is needed after the clone, and nothing comes from
crates.io.
