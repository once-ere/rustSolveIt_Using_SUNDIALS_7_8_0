# Clean-room reimplementation record — Stage 1

**Date:** 2026-07-26
**Scope:** replacing the licence-encumbered numerical routines found in
the SolveIt C++ sources with independently written Rust, so that
`rustSimulate` can carry the physics without carrying the licences.

This document exists so the claim "clean room" can be *audited* rather
than merely asserted.

---

## 1. What was encumbered, and why it matters

The review of the SolveIt 2002 C/C++/Fortran sources and the 2026 C++23
upgrade found four licensing problems, all on live quantum-mechanics
code paths:

| routine | where | origin | licence position |
|---|---|---|---|
| `bessj0`, `bessj1`, `bessj` | `QMEvolve.h`, `DataQM_Scatt1D.h` | *Numerical Recipes* | **Not redistributable.** The NR licence permits use in your own programs but forbids redistribution of the source. |
| `zbrent` | `TrajectoryRecord.h` | *Numerical Recipes* | Same. |
| tridiagonal solver, cyclic tridiagonal solver | QM propagator | line-for-line GSL | **GPL-3.0.** Copying it into this repo would place the whole work under GPL-3.0. |
| GIAC (expression evaluation) | user-typed potentials | GIAC | **GPL-2.0.** Same infection problem. |

The important distinction: **the algorithms are not encumbered — the
expressions of them are.** Miller's downward recurrence, the Thomas
algorithm, Sherman–Morrison and Brent's method are all standard
textbook mathematics, published decades before any of these
implementations and freely usable. What may not ship is *that source
code*. So the fix is to write the mathematics again, not to work around
the licence.

**No encumbered source ships in this repository** — but see §7, because
that was briefly untrue and the correction is part of the record.

---

## 2. Method

For each replacement I worked from the *mathematical statement* of the
algorithm — the recurrence relation, the elimination scheme, the
normalisation identity — citing DLMF equation numbers and Abramowitz &
Stegun (1964, a US Government work in the public domain, and therefore
safe to quote formulas from).

Concretely, and this is the part that makes the claim checkable:

> **I did not open `QMEvolve.h`, `DataQM_Scatt1D.h`, or
> `TrajectoryRecord.h` while writing any of these modules.** The only
> information carried across was the *list of routine names and what
> each is for*, taken from the licensing review — which is a functional
> specification, not an expression.

Two design choices were made specifically to keep the boundary clean:

1. **No rational-approximation coefficient tables.** The NR/Cephes-style
   `bessj0` works from tabulated minimax polynomial coefficients. Those
   coefficient tables are exactly the kind of thing that would look
   copied if it were copied. The implementation here uses the
   normalisation-sum approach instead, which requires **no magic
   constants at all** — every number in the file is either a loop bound
   or a scaling threshold with a stated reason.
2. **Different interface shape.** `bessel_j_array` returns the whole
   table in one pass because that is what the physics wants; it is not
   a signature-for-signature port of `bessj(n, x)`.

---

## 3. The replacements

### 3.1 `special_functions::bessel` — replaces `bessj0`, `bessj1`, `bessj`

| | |
|---|---|
| **file** | `special_functions/src/bessel.rs` |
| **public API** | `bessel_j_array(n_max, x) -> Result<Vec<f64>, String>`, `bessel_j(n, x) -> Result<f64, String>` |
| **algorithm** | Miller's downward recurrence |
| **citations** | recurrence `J_{n-1} + J_{n+1} = (2n/x) J_n` — DLMF 10.6.1 (<https://dlmf.nist.gov/10.6.E1>), A&S 9.1.27; normalisation `J_0 + 2(J_2 + J_4 + …) = 1` — DLMF 10.12.4 (<https://dlmf.nist.gov/10.12.E4>), A&S 9.1.46; parity `J_n(-x) = (-1)^n J_n(x)` — A&S 9.1.35 |

Why downward: `J_n(x)` decays super-exponentially once `n > x`, so the
upward recurrence amplifies round-off into the (growing) `Y_n` solution
and destroys the answer. Recurring downward from an artificial seed
does the opposite — the contamination decays — and the arbitrary seed
is removed at the end by imposing the normalisation identity.

**A defect found and fixed during testing.** The first version chose the
seed order as `n_max + 20 + (2√x + x/2)`. At `x = 45` that is only 70,
barely above `x` itself, and `J_n(45)` has not yet begun to decay at
`n = 70` — so the recurrence had not converged and `J_0(45)` was correct
to only ~9 digits (ours `1.15818670747e-1` vs Cephes `1.15818670673e-1`).
The cross-check against the vendored Cephes caught it. The seed is now
`n_max + 30 + (1.5x + 12√x)`, and the measurement that motivated the
change is recorded in a comment in the source so nobody tightens it
back.

**Verification** (6 tests):
- cross-validation against the independently written vendored Cephes
  `jv` at `x ∈ {0.1, 0.7, 1, 2.5, 5, 9, 20, 45}` for `n = 0..15`, to
  1e-10 relative — *two entirely different algorithms agreeing is the
  strongest evidence available*;
- the normalisation identity holds on the **output** to 1e-12 (not a
  tautology: it is imposed on the unnormalised values, so its survival
  confirms the scaling step);
- the three-term recurrence is satisfied to 1e-12 for `n = 1..24`;
- `J_0(0) = 1`, `J_n(0) = 0`, parity, and the first zero of `J_0` at
  2.404825557695773;
- `J_30(1) ≈ 1e-49` — positive and correctly tiny, the exact regime
  where upward recurrence fails;
- invalid input (negative order, NaN, ∞) returns `Err`.

### 3.2 `special_functions::tridiag` — replaces both GSL-derived solvers

| | |
|---|---|
| **file** | `special_functions/src/tridiag.rs` |
| **public API** | `solve_tridiag` (real), `solve_tridiag_c` (complex), `solve_cyclic_tridiag_c` (periodic, complex) |
| **algorithm** | Thomas algorithm (Gaussian elimination specialised to a tridiagonal band: one forward sweep, one back substitution); Sherman–Morrison rank-one correction for the cyclic case |

The cyclic case writes the periodic matrix as `A' + u vᵀ` with `A'`
tridiagonal, solves twice against `A'`, and combines as
`x = y − z (v·y)/(1 + v·z)`.

Both solvers are complex-valued because the Crank–Nicolson operator
`1 + iH dt/2` is complex; the real entry point converts and delegates
rather than duplicating the sweep.

**Stability, stated honestly.** The Thomas algorithm does no pivoting,
so it is not unconditionally stable. It *is* stable for diagonally
dominant systems, which the Crank–Nicolson operator is by construction
(the leading `1` keeps the diagonal away from zero). A pivot that
collapses is returned as an `Err` naming the row, never divided
through silently.

**A defect found during testing — in the test, not the code.** The
hand-written expected solution for a 3×3 system was wrong; the solver
was right. The corrected value is derived in a comment at the assertion
(`x = [0.5, 0, 1.5]`, verified by substitution). The residual test
`‖Ax − b‖ → 0` had been passing throughout and is the check that
actually verifies the routine — fixed expected-value constants are the
weaker instrument, and this is the fourth time in this project that a
hand-computed constant, not the numerics, was the error.

**Verification** (6 tests + 2 doctests):
- residual `‖Ax − b‖ < 1e-11` on a 60×60 complex system with varying
  bands;
- cyclic residual `< 1e-10` on a 40×40 periodic system;
- the cyclic solver reproduces the plain one when the corners vanish;
- **Crank–Nicolson unitarity**: a free-particle packet propagated 25
  steps with periodic boundaries conserves its norm to `< 1e-10`;
- invalid input — empty, length mismatch, zero pivot, NaN, and `n < 3`
  for the cyclic form — all return `Err`.

### 3.3 `special_functions::quadrature::brent_root` — replaces `zbrent`

Already in the tree from the earlier milestone. Brent's method
(bisection + secant + inverse quadratic interpolation with the standard
safeguards), written from the method description. 24 unit tests plus
doctests; mutation-tested to confirm the suite is non-vacuous.

### 3.4 GIAC — no replacement needed

GIAC was doing symbolic evaluation of user-typed potentials. `posim`
already has its own lexer → parser → stack-machine expression
evaluator, written for this project from the start, which covers that
use. Nothing to port.

### 3.5 `special_functions::complex` — supporting type

`Complex64` (`special_functions/src/complex.rs`), written from the
definitions. Needed because the propagator is complex and the project
takes no external dependencies. Deliberately minimal: arithmetic,
conjugate, modulus, argument, `exp`, `from_polar`, `inv`. Verified
against arithmetic identities, `z·z̄ = |z|²`, `i² = −1`, Euler's
identity, `|e^{it}| = 1`, and `e^{a+b} = e^a e^b`.

---

## 4. End-to-end evidence

Unit tests prove the pieces; this proves they do the job they were
written for. `special_functions/examples/scatter_1d.rs` reproduces the
`DataQM_Scatt1D` scenario — a Gaussian packet scattering off a
rectangular barrier — using only clean-room code.

Setup: ħ = m = 1, 3999 grid points on [−100, 100] at dx = 0.05, dt =
0.005 for 6000 steps (to t = 30), barrier V₀ = 2.5 of width 1, packet
k₀ = 2 (E₀ = 2.0), σ = 2, launched at x₀ = −25.

| quantity | measured |
|---|---|
| reflected `R` | 0.670188 |
| transmitted `T` | 0.329812 |
| still inside the barrier | 5.6e-7 |
| `R + T + inside` | 1.00000000000147 |
| analytic `T` at the central energy E₀ | 0.316660 |
| analytic `T` averaged over &#124;φ(k)&#124;² | 0.330650 |
| **relative difference** | **0.254 %** |
| **worst norm drift, 6000 steps** | **1.47e-12** |

Two remarks on reading this table.

**The norm drift is the sharp test.** The Cayley operator is unitary
for *any* time step — that is exact mathematics, not an asymptotic
statement — so drift cannot be blamed on dt. 1.47e-12 accumulated over
6000 solves is the tridiagonal routine behaving correctly; a wrong
solver shows up here immediately and unmistakably.

**The momentum averaging is not cosmetic.** The packet's energy spread
straddles the barrier top (E₀ = 2.0 against V₀ = 2.5, with σ = 2 giving
Δk = 0.25), so the transmission varies strongly across the packet.
Averaging the analytic coefficient over |φ(k)|² moves the prediction
from 0.3167 to 0.3307 and the agreement from 4.1 % to 0.25 % — the
simulation is right and the naive single-energy comparison would have
been the thing that was wrong.

---

## 5. State of the tree

| | |
|---|---|
| workspace tests | **568 passed workspace-wide**, 0 failed (212 before Stage 1) |
| build warnings | **0** |
| `unsafe` in `special_functions` | none — `#![forbid(unsafe_code)]` at the crate root |
| external dependencies | none |
| encumbered source in the repo | **none** |

New in Stage 1: 3 modules (`bessel`, `tridiag`, `complex`), 1 example
(`scatter_1d`), 18 tests.

---

## 6. What is not claimed

- These are replacements for the *specific* routines the review flagged,
  not a general-purpose linear algebra or Bessel library.
- `bessel_j_array` covers integer order and real argument. `Y_n`, `I_n`,
  `K_n` and complex arguments come from the vendored Cephes or remain
  future work; complex-argument Bessel is still the deferred
  milestone-2 item.
- The Thomas algorithm's lack of pivoting is a real limitation, stated
  in the module documentation rather than hidden. Systems that are not
  diagonally dominant should not use it.
- This is Stage 1 of the SolveIt port. It removes the licensing
  blockers; it does not by itself constitute the ported simulator.

---

## 7. Incident: the reference trees were published, and the remediation

Recorded in full because a clean-room claim is only worth what its
audit trail is worth, and because the failure mode is one that will
recur in any repository that keeps encumbered reference material on
disk.

### What happened

Across the three commits of 2026-07-26 in the *previous* repository,
**749 files
of the SolveIt 2002 C/C++ sources — 56 MB — were tracked and pushed to
the then-public `once-ere/rustSimulate`.** Among them were the three
files named in the licensing review:

- `obsolete_or_historic/SolveIt/QM/QMEvolve.h`
- `obsolete_or_historic/SolveIt/QM/DataQM_Scatt1D.h`
- `obsolete_or_historic/SolveIt/RigidBody/TrajectoryRecord.h`

Eight tracked files matched Numerical Recipes or GSL signatures. For the
duration, the public repository redistributed NR code (whose licence
forbids exactly that) and GPL-3.0 GSL-derived code inside an otherwise
permissively licensed work.

### Root cause

`.gitignore` contained:

```
./obsolete_or_historic
```

A gitignore pattern containing a slash is **anchored to the directory
holding the `.gitignore`**, and a leading `./` therefore asks git to
match a directory literally named `.`. The pattern matched nothing. It
looked correct, it sat under a descriptive comment, and it never
ignored a single file.

The trap was then sprung by a `git add -A` that assumed the ignore rule
was doing its job.

This is the **second** gitignore defect in this project, and the same
species as the first: the earlier export dropped 77 reference files
because `*.out` was unanchored. Both were silent — nothing errors when
a pattern matches the wrong set.

### Why it was not caught sooner

Three checks all missed it, and each miss is instructive:

1. The pre-commit scan looked for NR/GSL strings **in the new source
   files being added**, not in what the repository already tracked. It
   was scoped to the wrong set.
2. The fresh-clone certification ran
   `ls -d obsolete_or_historic SolveIt || echo absent`. `ls` succeeded
   on the first path, so the `||` branch never fired — and the success
   output scrolled past under a heading that said the opposite. The
   check was structurally incapable of failing loudly.
3. No check ever asked git the direct question: *does this ignore rule
   match this file?*

### Remediation

The first round was containment, inside the original repository:

1. Visibility set to **private** immediately, to stop ongoing exposure
   before anything else was attempted.
2. Full backup taken first — a verified `git bundle` of all refs plus a
   local `backup/pre-rewrite` branch, never pushed.
3. The three affected commits rewritten with
   `git filter-branch --index-filter` to strip the tree from history, so
   the objects became unreachable rather than merely absent from the tip.
4. `.gitignore` corrected to `/obsolete_or_historic/` — leading slash to
   anchor, trailing slash to restrict to directories — with the sibling
   trees added and a comment recording the `./` trap.
5. Force-pushed, then re-certified from a fresh plain clone.

That was **not sufficient**, and it is worth being precise about why.
Force-pushing makes objects unreachable; it does not delete them.
GitHub does not garbage-collect on demand, and a fetch of the old
commit by direct SHA still succeeded afterwards, still yielding all 749
files. The SHA was not a secret — it had been in the public commit list
for the whole exposure window, and GitHub's events feed is archived by
third parties. The only remaining lever inside that repository would
have been a GitHub Support request to purge unreachable objects.

So the repository was **deleted outright** and this one created fresh
and private. Deleting destroys the objects with it: no Support ticket,
no waiting, no residual copies. The fork count had been 0 throughout, so
no fork network held an independent copy. **This repository has never
contained the reference trees in any commit**, which the certification
script verifies against history and not merely against the tip.

Two structural changes came out of it, and they matter more than the
cleanup did:

- **The reference trees no longer live inside the working tree at all.**
  Previously `.gitignore` was the single thing standing between
  `git add -A` and a licensing incident — and a gitignore rule is a
  thing you can get silently wrong, as this project did twice. A file
  outside the repository cannot be added by any command regardless of
  how the ignore rules are written. The `.gitignore` entries remain as a
  second layer, but nothing depends on them any more.
- **The repository stays private until the port is complete** and
  `scripts/certify_clean.sh` passes against a fresh plain clone. The
  asymmetry is the whole argument: private costs nothing during
  development, while public makes the next mistake permanent.

### The standing rule this produces

**Never trust a gitignore pattern that has not been interrogated.**
### The gate for stale documentation (added Stage 2E)

Three stages running found prose that was accurate when written and
false by the time it was read — including this file, which claimed 230
workspace tests against 556, and `EXPORT_PROVENANCE.md`, the document a
reader uses to *audit* the release, still saying a repaired defect
"remains unfixed" and telling them to expect 104 tests.

The lesson was written down as an action three times. Stage 2E
mechanised it, as two gates in `scripts/certify_clean.sh`:

* **documented test counts match the tree.** Only the two canonical
  live phrasings are checked — `expect N passed` and
  `N passed workspace-wide`. A dated historical record ("104 passed at
  export time") is history and must not be rewritten; the check that
  cannot tell a record from a claim is the check that cries wolf.
* **no retired claim has reappeared.** A list of real sentences that
  shipped and became false, with the stage that retired each.
  **Quoting one is not asserting it** — both `PROJECT_STATUS.md` and
  `airy_uniform.rs` quote their retired sentence while explaining that
  it was retired, which is exactly the wanted behaviour, so quoted
  spans are stripped before matching.

Both carry self-tests in the existing style, and both were verified to
FAIL on deliberately reintroduced staleness before being trusted.

Adding an ignore rule for anything sensitive is not complete until

```
git check-ignore -v <a real path under it>
```

prints the matching rule. A pattern that matches nothing is
indistinguishable from a correct one until the moment it costs you.

And the corollary for verification scripts: a check whose failure path
is an `||` after a command that can partially succeed is not a check.
Assert on the count, and make the failing case print loudly.

---

## 8. The converse case: a faithful port of the author's *own* work

Everything above is about code that could **not** be copied. The rule
has a mirror image that matters just as much, because applying the
clean-room procedure where it does not belong destroys value rather than
protecting it.

`CQMEvolve1D::EVOLVE_NASH` (`QM/QMEvolve1D.cpp`, ~20 lines) is the one
piece of original numerical work in the SolveIt C++. It is not derived
from Numerical Recipes, GSL, or GIAC; it is the author's own scheme.
There is therefore **no reason to reimplement it from a description** —
the right thing is a faithful port, checked against the original
algorithm rather than merely against the physics.

That is `quantum::nash`, and the check is
`the_port_reproduces_the_original_algorithm_to_rounding`: a
statement-for-statement transliteration of the C++ loop lives in the
test module and the port is required to match it, at the original's own
constants (`Lambda = 0.92`, `NumOrder = 16`), to **1e-12 after a
thousand steps**. Measured, the drift is 3.5e-16 after one step and
1.9e-13 after a thousand — floating-point summation order, nothing else.

One dependency of that kernel *was* encumbered and is not transliterated:
`bessj`, the Numerical Recipes Bessel routine that supplied `J_M(Lambda)`.
The clean-room `special_functions::bessel::bessel_j_array` (§3.1)
supplies the same values. So the port is faithful to the original
*scheme* while none of the encumbered *code* comes across — which is
exactly the line this document exists to draw.

### What the port makes explicit

The C++ ran on two baked-in dimensionless constants. The port takes
physical quantities and derives them, which turns a convention into a
testable identity:

```text
   lambda = hbar dt / (m h^2)          v_j = V_j m h^2 / hbar^2
```

`lambda_follows_its_definition_in_every_unit` pins the first over three
unit systems; the faithfulness test drives the port *through* that
mapping, so the mapping is exercised rather than asserted.

### Two things the port fixes, and one it deliberately does not

* **The index wrap.** `mu()` added or subtracted `N` exactly once, so a
  stencil wider than the grid read outside the array — latent in
  SolveIt, where `NumOrder` was 16 and `NDATA` was in the hundreds. The
  port uses a real modulo, and
  `a_stencil_wider_than_the_grid_still_wraps_correctly` pins the case.
* **The truncation order.** It was the constant 16. `order_for` now
  derives it from `lambda` and a tolerance, by summing the Bessel tail
  rather than estimating it. At `lambda = 0.92` the needed order is 14,
  so the shipped 16 carried a small margin.
* **The splitting is offered both ways, and the default is the
  original's.** Lie–Trotter is first order in `dt`; `Splitting::Strang`
  puts half a potential phase on each side of the stencil and is second
  order. Measured against the same diagonalised reference, the error
  quarters rather than halves per step refinement, and at a fixed step
  it lands over 100x closer.

  It is close to free: consecutive Strang steps fuse their adjacent
  half-phases into full ones, so a run pays one extra half-phase in
  total rather than one per step, and `the_fused_run_matches_repeated_steps`
  proves that optimisation changes nothing.

  `NashPropagator::new` still defaults to `Splitting::Lie`, because the
  default behaviour of a port must be what the original does. Strang is
  reached through `with_splitting`, so nothing about the faithfulness
  claim above is weakened by its existence.
