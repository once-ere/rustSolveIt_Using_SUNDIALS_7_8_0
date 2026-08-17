# Special Functions — Reference and Examples

*The special-function layer of rustSimulate, cited to the NIST DLMF.
Every function below has at least one **intermediate** example (direct
evaluation you can check by hand) and one **expert** example (the
function doing real work inside a computation).*

This document grows one module at a time, alongside the code. Sections
marked **planned** are not implemented yet and say so rather than
pretending otherwise.

**Citation policy.** Functions cite DLMF equations by number and
permalink. The DLMF is copyright NIST and is *cited, never reproduced*;
formulas displayed here are from Abramowitz & Stegun (1964), a
public-domain US Government work, or written in our own notation. See
[THIRD_PARTY.md](THIRD_PARTY.md).

---

## Status at a glance

| module | source | state |
|---|---|---|
| `sph_bessel` — spherical Bessel jₙ, yₙ, derivatives | **native** | ✅ implemented, 7 tests + cross-validation |
| gamma, erf, Ei/Si/Ci, incomplete γ/B, Airy, cylindrical Bessel, elliptic, Jacobi, ζ | vendored Cephes | ✅ available, 11 identity tests |
| `legendre` — Pₙ, Pₗᵐ, normalised P̄ₗᵐ, Yₗᵐ | **native** | ✅ implemented, tests + overflow policy |
| `orthopoly` — Hermite, Laguerre, Chebyshev, Gegenbauer, Jacobi | **native** | ✅ implemented, 24 tests, mutation-tested |
| `eigen` — cyclic Jacobi symmetric eigensolver | **native** | ✅ implemented |
| `quadrature` — Gauss–Legendre, adaptive Simpson, Brent roots | **native** | ✅ implemented, 24 tests + 5 doctests |
| `complex` — `Complex64` | **native** | ✅ implemented (Stage 1) |
| `tridiag` — Thomas + Sherman–Morrison, real and complex | **native** | ✅ implemented (Stage 1), clean-room |
| `bessel` — integer-order Jₙ table | **native** | ✅ implemented (Stage 1), clean-room |
| `gamma_complex`, `bessel_complex`, `bessel_cnu`, `bessel_cnu_large`, `bessel_scaled`, `hankel`, `airy_complex`, `airy_uniform`, `debye`, `lanczos` | **native** | ✅ implemented — see §11, and `grammar.md` §4.1 for the accuracy laws |
| `wigner` — 3j, 6j, **9j**, Clebsch–Gordan | **native** | ✅ implemented, reachable from the notebook |

Modules marked *clean-room* replace licence-encumbered routines from the
SolveIt C++ sources; see [CLEANROOM_PROVENANCE.md](CLEANROOM_PROVENANCE.md)
for the audit record.

---

## 1. `sph_bessel` — spherical Bessel functions

Solutions of the radial Helmholtz equation in spherical coordinates.
They appear in every partial-wave expansion, in the free-particle
solution of the Schrödinger equation, and in Mie scattering.

Relation to the cylindrical functions — `DLMF 10.47.3`
(<https://dlmf.nist.gov/10.47.E3>):

```
    j_n(x) = sqrt(pi/2x) · J_{n+1/2}(x)
    y_n(x) = sqrt(pi/2x) · Y_{n+1/2}(x)
```

### API

| function | meaning | domain |
|---|---|---|
| `sph_j(n, x)` | jₙ(x), first kind | n ≥ 0, x finite (any sign) |
| `sph_y(n, x)` | yₙ(x), second kind | n ≥ 0, **x > 0** (singular at 0) |
| `sph_j_prime(n, x)` | jₙ′(x) | as `sph_j` |
| `sph_y_prime(n, x)` | yₙ′(x) | as `sph_y` |

All return `Result<f64, String>`; a domain violation is an `Err` with a
message naming the argument, never a silent `NaN`.

### Intermediate example — check against the closed forms

The two lowest orders have elementary closed forms (A&S 10.1.11):
j₀(x) = sin x / x and j₁(x) = sin x / x² − cos x / x.

```rust
use special_functions::sph_bessel::{sph_j, sph_y};

let x = 1.3_f64;
assert!((sph_j(0, x)? - x.sin() / x).abs() < 1e-15);
assert!((sph_j(1, x)? - (x.sin()/(x*x) - x.cos()/x)).abs() < 1e-14);

// y_0(x) = -cos x / x, and it diverges at the origin:
assert!((sph_y(0, x)? + x.cos() / x).abs() < 1e-15);
assert!(sph_y(0, 0.0).is_err());          // reported, not NaN
# Ok::<(), String>(())
```

Note the decay once the order exceeds the argument — this is the
behaviour that makes naive evaluation fail:

```rust
# use special_functions::sph_bessel::sph_j;
assert!(sph_j(20, 1.0)?.abs() < 1e-25);   // j_20(1) ~ 7.6e-26
# Ok::<(), String>(())
```

### Expert example — hard-sphere scattering phase shifts

For a hard sphere of radius *a*, the ℓ-th partial-wave phase shift of a
wave with momentum *k* is

```
    tan(delta_l) = j_l(ka) / y_l(ka)
```

and the total elastic cross-section is
σ = (4π/k²) Σ_ℓ (2ℓ+1) sin²(δ_ℓ). In the long-wavelength limit ka → 0
only the ℓ = 0 term survives and σ → 4πa², four times the geometric
cross-section — the classic result this example reproduces.

```rust
use special_functions::sph_bessel::{sph_j, sph_y};

fn hard_sphere_cross_section(k: f64, a: f64, l_max: i32) -> Result<f64, String> {
    let ka = k * a;
    let mut sigma = 0.0;
    for l in 0..=l_max {
        let delta = (sph_j(l, ka)? / sph_y(l, ka)?).atan();
        sigma += (2 * l + 1) as f64 * delta.sin().powi(2);
    }
    Ok(4.0 * std::f64::consts::PI / (k * k) * sigma)
}

// Low energy: the cross-section approaches 4 pi a^2.
let a = 1.0;
let sigma = hard_sphere_cross_section(1e-3, a, 6)?;
let geometric = std::f64::consts::PI * a * a;
assert!((sigma / geometric - 4.0).abs() < 1e-4);
# Ok::<(), String>(())
```

Two things worth noticing. The sum truncates safely at small `l_max`
because jₗ(ka) vanishes like (ka)^ℓ, so high partial waves contribute
nothing at low energy — physically, a slow particle cannot resolve the
sphere. And `sph_y` is the function that *diverges* at small argument,
which is exactly why the ratio jₗ/yₗ → 0 and the phase shifts vanish.

### Why the implementation looks the way it does

Both functions satisfy the same recurrence (`DLMF 10.51.1`,
<https://dlmf.nist.gov/10.51.E1>):

```
    f_{n+1}(x) = ((2n+1)/x) · f_n(x) - f_{n-1}(x)
```

but its numerical stability differs completely:

* **yₙ grows** with n, so recurring *upward* is stable — that is what
  `sph_y` does, seeded by the closed forms for y₀ and y₁.
* **jₙ decays** once n > x. Recurring upward there amplifies the seed's
  rounding error until the answer is noise. `sph_j` therefore uses
  **Miller's algorithm**: seed an artificial value far above the wanted
  order, recur *downward* (the direction in which the wanted solution
  dominates), rescaling to stay in range, then fix the overall scale
  against the exactly known j₀ = sin x / x.

A regression test, `sph_j_upward_recurrence_is_unstable_for_n_gt_x`,
runs the naive upward version alongside the real one and asserts the
naive result is wrong by more than six orders of magnitude at
n = 20, x = 1. It exists so nobody "simplifies" the implementation.

Small arguments (x < 10⁻⁴) use the leading series jₙ ≈ xⁿ/(2n+1)!!
(A&S 10.1.2) with a second-order correction, because the closed forms
suffer catastrophic cancellation there.

### How it is verified

| check | what it pins |
|---|---|
| closed forms, n ≤ 2 | absolute correctness at low order |
| **Wronskian** jₙyₙ′ − jₙ′yₙ = 1/x² (A&S 10.1.31) | 12 orders × 5 arguments — the strongest single identity |
| three-term recurrence | consistency across n for both functions |
| naive-upward blow-up | the stability property itself |
| small-x series limit | the x → 0 branch |
| parity jₙ(−x) = (−1)ⁿjₙ(x), origin values | sign and special-value handling |
| domain errors | `Err`, never silent `NaN` |
| **cross-validation** vs vendored J₍ₙ₊½₎ | two independent implementations agree |

That last one is the most valuable: our Miller-recurrence code and
Cephes's cylindrical Bessel of half-integer order were written
independently, and `DLMF 10.47.3` says they must agree. They do, to
1e-9 relative across n ∈ [0,10) and x ∈ [0.25, 40].

---

## 2. Vendored classical chapters (Cephes translation)

Available today through `special_functions::cephes`, covering DLMF
chapters 5–10, 19, 22 and 25 on **real arguments**. Provenance and the
licence caveat are in [THIRD_PARTY.md](THIRD_PARTY.md).

These are not our implementations, so we verify them rather than trust
them — `tests/vendored_identities.rs` checks:

| family | identity checked |
|---|---|
| Γ | Γ(½)=√π; Γ(n+1)=n! to n=15; reflection Γ(z)Γ(1−z)=π/sin πz; duplication |
| ψ | ψ(1)=−γ; ψ(z+1)=ψ(z)+1/z |
| erf | erf+erfc=1; odd symmetry; erf(6)→1 |
| Ai, Bi | Wronskian AiBi′−Ai′Bi = 1/π across x ∈ [−6,6]; exact values at 0 |
| Jᵥ, Yᵥ | Wronskian = 2/(πx); J₍±½₎ closed forms; recurrence |
| K, E | Legendre relation E(m)K(1−m)+E(1−m)K(m)−K(m)K(1−m) = π/2 |
| sn, cn, dn | sn²+cn²=1; m·sn²+dn²=1 |
| ζ | ζ(2)=π²/6, ζ(4)=π⁴/90, ζ(6)=π⁶/945 |

**Argument conventions**, pinned empirically before the tests were
written (getting these wrong yields tests that pass for the wrong
reason):

* `ellpe(m)` takes the parameter **m** directly — E(0)=π/2, E(1)=1.
* `ellpk(m₁)` takes the **complement** m₁ = 1 − m — so K(m) is
  `ellpk(1.0 - m)`.

---

---

## 3. `legendre` — Legendre, associated Legendre, spherical harmonics

The angular part of every central-potential problem. `DLMF 14`.

| function | returns |
|---|---|
| `legendre_p(n, x)` | Pₙ(x), Bonnet recurrence |
| `legendre_p_prime(n, x)` | Pₙ′(x), endpoints in closed form |
| `assoc_legendre_p(l, m, x)` | Pₗᵐ(x), **unnormalised**, Condon–Shortley phase |
| `norm_assoc_legendre_p(l, m, x)` | P̄ₗᵐ(x), **fully normalised** |
| `sph_harm(l, m, θ, φ)` | Yₗᵐ as `(re, im)` |
| `sph_harm_real(l, m, θ, φ)` | real spherical harmonic (orbitals) |

### Intermediate — values you can check by hand

```rust
use special_functions::legendre::{legendre_p, assoc_legendre_p, sph_harm};
use std::f64::consts::PI;

assert!((legendre_p(7, 1.0)? - 1.0).abs() < 1e-14);        // P_n(1) = 1
let x = 0.5_f64;
assert!((assoc_legendre_p(2, 2, x)? - 3.0*(1.0-x*x)).abs() < 1e-14);  // P_2^2 = 3(1-x^2)
let (re, im) = sph_harm(0, 0, 1.0, 2.0)?;                  // Y_0^0 = 1/sqrt(4 pi)
assert!((re - 1.0/(4.0*PI).sqrt()).abs() < 1e-15 && im.abs() < 1e-18);
# Ok::<(), String>(())
```

### Expert — why two families exist

Overflow in `Pₗᵐ` is governed by the **order m**, not the degree ℓ —
the seed carries `(2m−1)!!`. Measured on this implementation at x = 0.3:

| (ℓ, m) | `assoc_legendre_p` | `norm_assoc_legendre_p` |
|---|---|---|
| (100, 50) | −3.4×10⁹⁷ | −9.8×10⁻² |
| (200, 100) | −8.4×10²²⁷ | −2.6×10⁻¹ |
| (170, 170) | **overflows** | 3.6×10⁻⁴ |
| (300, 150) | **NaN** | 3.3×10⁻¹ |
| (200, 0) | −9.8×10⁻³ | −5.5×10⁻² |

The normalised form is computed *directly in the normalised basis* —
seed written as a product of factors below one, ascent with normalised
coefficients — so nothing ever leaves O(1). Scaling the raw value by
`N(l,m)` afterwards would not work: the constant underflows exactly
where `Pₗᵐ` overflows. The raw form returns `Err` naming the remedy
rather than handing back `inf`/`NaN`.

*Verified by:* Bonnet recurrence; P(±1) special values; orthogonality
∫PₘPₙ = 2δ/(2n+1) by quadrature; the m=0 reduction; the negative-order
relation `DLMF 14.9.3`; agreement between the two families wherever both
are representable; spherical-harmonic orthonormality 2π∫|P̄|² = 1; and
known Y values.

---

## 4. `orthopoly` — the classical orthogonal polynomials

Hermite (Hₙ and Heₙ), Laguerre (Lₙ and generalised Lₙ^α), Chebyshev
(Tₙ, Uₙ), Gegenbauer, Jacobi — all by stable three-term recurrences,
`DLMF 18.9`.

### Intermediate

```rust
use special_functions::orthopoly::{hermite_h, chebyshev_t};
let x = 0.37_f64;
assert!((hermite_h(2, x)? - (4.0*x*x - 2.0)).abs() < 1e-14);   // H_2 = 4x^2-2
// T_n(cos t) = cos(n t) — the sharpest Chebyshev check there is
let t = 0.9_f64;
assert!((chebyshev_t(5, t.cos())? - (5.0*t).cos()).abs() < 1e-13);
# Ok::<(), String>(())
```

### Expert — the quantum harmonic oscillator, two independent ways

`examples/harmonic_oscillator.rs` (runnable) solves ĤΨ = EΨ for
H = −½∂ₓ² + ½x² by two routes that **share no code**, then compares:

* *Analytic* — ψₙ(x) = (2ⁿn!√π)^(−½) Hₙ(x)e^(−x²/2), with orthonormality
  and energies obtained by Gauss–Legendre quadrature.
* *Numerical* — discretise H on a grid and diagonalise with `eigen`.

Measured:

```
worst |<n|n> - 1| = 7.77e-16      worst |<m|n>| = 6.95e-16
virial energies E_n = 2<V>:  worst error 3.55e-15  (n = 0..5)

finite-difference error growth        grid refinement
  n=0  observed  1.0x  predicted  1.0x     N=100  err 7.855e-4
  n=1            5.0x            5.0x      N=200  err 1.981e-4  (fell 3.97x)
  n=2           13.0x           13.0x      N=400  err 4.976e-5  (fell 3.98x)
  n=3           25.0x           25.0x
  n=4           41.0x           41.0x    expected drop: 4x (2nd order)
  n=5           61.0x           61.0x
```

Two things worth extracting. The analytic route is at machine
precision — that is `orthopoly` and `quadrature` agreeing to 16 digits.
The finite-difference error is **not noise**: it tracks `2n²+2n+1`
exactly, because higher states oscillate faster and a 3-point stencil
resolves them less well; and halving h drops the error 4×, the defining
property of a second-order stencil. Asserting those *laws* is a much
stronger test than any fixed tolerance.

*Verified by:* low-order closed forms; endpoint and origin values;
parity; recurrences across degree; T(cos t) = cos(nt); reductions
(Gegenbauer→Chebyshev U and Legendre, Jacobi→Legendre and Chebyshev T);
orthogonality by quadrature for four families; and the He/H convention
relation. The suite was additionally **mutation-tested** — seven
deliberate bugs injected, each caught by 3–6 independent tests — so it
is demonstrably not vacuous.

---

## 5. `eigen` — dense real-symmetric eigenproblems

`jacobi_eigen(a)` → `(eigenvalues ascending, eigenvectors)`;
`eigenvalues(a)` when the vectors are not needed.

Jacobi rather than QR, deliberately: unconditionally convergent,
eigenvectors orthogonal by construction, more accurate for the *small*
eigenvalues a ground state lives among, and short enough to be obviously
correct. For n of order 10²–10³ the constant factor is irrelevant.

### Intermediate
```rust
use special_functions::eigen::jacobi_eigen;
let a = vec![vec![2.0, 1.0], vec![1.0, 2.0]];
let (vals, _) = jacobi_eigen(&a).unwrap();
assert!((vals[0] - 1.0).abs() < 1e-12 && (vals[1] - 3.0).abs() < 1e-12);
```

### Expert — a spectrum with a known closed form
The tridiagonal Laplacian has exact eigenvalues 4sin²(kπ/2(n+1)); the
test checks all 40 of them to 1e-10. Scaled to a box of length L it
becomes the particle in a box, where the discrete levels approach
n²π²/L² from **below** with relative deficit (kπh)²/12 — asserted as a
law, within 10%.

*Verified by:* diagonal input; a 2×2 closed form; A**v** = λ**v** for
every pair; eigenvector orthonormality; trace and determinant
invariants; the analytic tridiagonal spectrum; and rejection of empty,
ragged, asymmetric and non-finite input.

---

## 6. `quadrature` — integration and root-finding

`gauss_legendre(n)`, `integrate(f, a, b, n)`,
`integrate_adaptive(f, a, b, tol)`, `brent_root(f, a, b, tol)`,
`find_roots(f, a, b, steps, tol)`.

### Intermediate
```rust
use special_functions::quadrature::{integrate, brent_root};
// exact for polynomials up to degree 2n-1
let v = integrate(|x| x*x*x, 0.0, 1.0, 3).unwrap();
assert!((v - 0.25).abs() < 1e-14);
// cos x = x
let r = brent_root(|x| x.cos() - x, 0.0, 1.0, 1e-12).unwrap();
assert!((r - 0.739085133215).abs() < 1e-9);
```

### Expert — the honest failure modes
Two behaviours are pinned by tests rather than hidden:

1. **`find_roots` cannot distinguish a pole from a root.** A pole also
   flips the sign of f, and Brent will converge on it. Check `f` at each
   returned value if your function can blow up — likely for eigenvalue
   mismatch functions. Even-multiplicity roots are invisible for the
   same structural reason.
2. **`integrate_adaptive` errors rather than guessing.** `1/√x` on
   [0,1] returns `Err`: the integral is finite but Simpson cannot
   resolve the endpoint, so the honest report is failure, not a
   plausible wrong number. Split the interval or remove the singularity.

*Verified by:* degree-exactness to 2n−1 **and** the converse (not exact
at 2n, so the bound is sharp); nodes as Legendre roots; weights summing
to the interval width; known integrals; superlinear Brent convergence
counted in function evaluations (bisection could not achieve it); and
reachable non-convergence branches.

---

## 7. `complex` — `Complex64`

Quantum wavefunctions are complex, and the project takes no external
dependencies, so the type is written here from the definitions:
arithmetic, `conj`, `abs`, `norm_sqr`, `arg`, `exp`, `from_polar`,
`inv`. Deliberately minimal — it exists to serve the propagator, not to
be a numeric tower.

### Intermediate
```rust
use special_functions::complex::Complex64 as C;
let a = C::new(3.0, -4.0);
assert_eq!(a.abs(), 5.0);                   // 3-4-5 triangle
assert_eq!(C::I * C::I, C::real(-1.0));     // i^2 = -1
// Euler: e^{i*pi} = -1
let z = (C::I * std::f64::consts::PI).exp();
assert!((z.re + 1.0).abs() < 1e-15 && z.im.abs() < 1e-15);
```

### Expert — why `norm_sqr` is separate from `abs`
`abs` costs a `hypot`; a probability density `|ψ|²` does not need one,
and a propagation loop evaluates it at every grid point at every step.
Keeping the squared modulus as its own operation is not micro-tuning —
it is also *more accurate*, since `hypot` then squaring round-trips
through a square root for no reason.

*Verified by:* arithmetic identities, `z·z̄ = |z|²`, `z/z = 1`, `i² = −1`,
Euler's identity, `|e^{it}| = 1` across several `t`, and `e^{a+b} = eᵃeᵇ`.

---

## 8. `tridiag` — tridiagonal and cyclic tridiagonal solvers

**Clean-room replacement for the GSL-derived (GPL-3.0) solvers in the
SolveIt QM propagator.**

Discretising `i ∂ψ/∂t = Hψ` in the Cayley form
`(1 + iH dt/2) ψⁿ⁺¹ = (1 − iH dt/2) ψⁿ` leaves a tridiagonal system to
solve every step; periodic boundaries make it cyclic.

### API
```rust
solve_tridiag(sub, diag, sup, rhs)                        -> Result<Vec<f64>, String>
solve_tridiag_c(sub, diag, sup, rhs)                      -> Result<Vec<Complex64>, String>
solve_cyclic_tridiag_c(sub, diag, sup, bl, tr, rhs)       -> Result<Vec<Complex64>, String>
```
`sub[i]` multiplies `x[i-1]`, `diag[i]` multiplies `x[i]`, `sup[i]`
multiplies `x[i+1]`; all three have length `n`, with `sub[0]` and
`sup[n-1]` unused. For the cyclic form `bl` is the entry at row `n−1`,
column `0` and `tr` the one at row `0`, column `n−1`.

### Intermediate
```rust
use special_functions::tridiag::solve_tridiag;
// [[2,1,0],[1,2,1],[0,1,2]] x = [1,2,3]
let x = solve_tridiag(&[0.0,1.0,1.0], &[2.0,2.0,2.0], &[1.0,1.0,0.0], &[1.0,2.0,3.0])?;
// x1 = 1-2x0 and x2 = 1+x0 reduce row 2 to -2x0+3 = 2, so x = [0.5, 0, 1.5]
assert!((x[0]-0.5).abs() < 1e-14);
assert!((x[2]-1.5).abs() < 1e-14);
# Ok::<(), String>(())
```

### Expert — one Crank–Nicolson step, and why unitarity is the test
```rust
use special_functions::complex::Complex64 as C;
use special_functions::tridiag::solve_tridiag_c;
let (n, dx, dt) = (400usize, 0.05_f64, 0.01_f64);
let k = 0.5 / (dx*dx);                 // H = -1/2 d^2/dx^2, free particle
let half = C::I * (dt/2.0);
let sub  = vec![half * C::real(-k); n];
let sup  = vec![half * C::real(-k); n];
let diag = vec![C::ONE + half * C::real(2.0*k); n];
// build rhs = (1 - iH dt/2) psi, then:
// let psi_next = solve_tridiag_c(&sub, &diag, &sup, &rhs)?;
```
The Cayley operator is **unitary for any `dt`** — that is exact
mathematics, not an asymptotic statement. So if the norm of `ψ` drifts,
the time step is not to blame: the *solver* is wrong. That makes norm
conservation the sharpest available test of this module, far sharper
than any residual tolerance, and it is what the test suite and the
`scatter_1d` example both assert.

Run the full scattering demonstration:
```
cargo run -p special_functions --release --example scatter_1d
```
It propagates 6000 steps and conserves the norm to **1.5e-12**, with the
transmitted fraction matching the momentum-averaged analytic barrier
coefficient to **0.25 %**.

### The limitation, stated plainly
The Thomas algorithm performs **no pivoting**, so it is not
unconditionally stable. It is stable for diagonally dominant systems —
which the Crank–Nicolson operator is by construction, since the leading
`1` holds the diagonal away from zero. For anything else, do not use it.
A pivot that collapses returns `Err` naming the row rather than dividing
through and producing plausible garbage.

*Verified by:* residual `‖Ax−b‖` on 60×60 complex and 40×40 cyclic
systems; the cyclic solver reducing to the plain one when the corners
vanish; end-to-end Crank–Nicolson norm conservation; and `Err` on empty
input, length mismatch, zero pivot, non-finite entries, and `n < 3` for
the cyclic form.

---

## 9. `bessel` — integer-order Jₙ, whole table in one pass

**Clean-room replacement for the *Numerical Recipes* `bessj0`, `bessj1`
and `bessj`** (not redistributable).

A Bessel-expanded propagator needs the entire set `J₀(λ) … J_N(λ)` at a
single argument. Computing them one at a time throws away the structure;
the downward recurrence produces all of them in one sweep.

### API
```rust
bessel_j_array(n_max: usize, x: f64) -> Result<Vec<f64>, String>   // J_0 .. J_{n_max}
bessel_j(n: i32, x: f64)            -> Result<f64, String>
```

### Intermediate
```rust
use special_functions::bessel::{bessel_j, bessel_j_array};
let z = bessel_j_array(4, 0.0)?;
assert_eq!(z[0], 1.0);                        // J_0(0) = 1
assert!(z[1..].iter().all(|&v| v == 0.0));    // J_n(0) = 0
// parity: J_1 odd, J_0 even
assert!((bessel_j(1, -1.4)? + bessel_j(1, 1.4)?).abs() < 1e-14);
assert!((bessel_j(0, -1.4)? - bessel_j(0, 1.4)?).abs() < 1e-14);
// first zero of J_0
assert!(bessel_j(0, 2.404_825_557_695_773)?.abs() < 1e-12);
# Ok::<(), String>(())
```

### Expert — why downward, and why no coefficient tables
`Jₙ(x)` decays super-exponentially once `n > x`; `Yₙ(x)` grows. The
upward recurrence therefore amplifies round-off into the growing
solution and destroys the answer — `J₃₀(1) ≈ 1e-49` is unreachable that
way. **Miller's algorithm** recurs *downward* from an artificial seed far
above the wanted order, so the contamination decays instead, and removes
the arbitrary seed at the end by imposing

    J₀(x) + 2[J₂(x) + J₄(x) + …] = 1     (DLMF 10.12.4, A&S 9.1.46)

This choice is also what makes the module defensible as clean-room work:
the recurrence needs **no tabulated minimax coefficients at all**. Every
constant in the file is a loop bound or a documented scaling threshold.

**The seed order matters, and getting it wrong is quiet.** The first
version started at `n_max + 20 + (2√x + x/2)`. At `x = 45` that is 70 —
barely above `x`, where `Jₙ(45)` has not begun to decay — so the
recurrence had not converged and `J₀(45)` was right to only ~9 digits.
Nothing crashed; the answer was simply slightly wrong. The cross-check
against the vendored Cephes caught it, and the seed is now
`n_max + 30 + (1.5x + 12√x)`.

*Verified by:* cross-validation against the independently written
vendored Cephes `jv` for `n = 0..15` at eight arguments up to `x = 45`
(two unrelated algorithms agreeing); the normalisation identity holding
on the **output**, which is not a tautology since it is imposed on the
unnormalised values; the three-term recurrence to 1e-12; exact values,
parity, and the first zero of `J₀`; `J₃₀(1)` correctly ≈ 1e-49; and
`Err` on negative order, NaN and ∞.

---

## 10. `wigner` — 3j, 6j, 9j and Clebsch–Gordan

Recoupling coefficients for angular momentum. `wigner_3j`,
`wigner_6j`, `wigner_9j` and `clebsch_gordan` are all implemented and are
registered notebook builtins.

`wigner_9j` takes its nine arguments **row by row**: it recouples four
angular momenta and is the overlap between coupling (1,2) and (3,4) first
versus (1,3) and (2,4) first. It is evaluated as a single sum over 6-j
symbols and vanishes when any of the six triads fails to close.

**Angular momenta may be half-integers**, so these four take plain numbers
rather than demanding whole ones — `clebsch_gordan(0.5, 0.5, 0.5, -0.5, 1, 0)`
is exactly what you write for two spin-½ particles. A coupling that violates
a selection rule returns **0**, which is the mathematically correct answer and
not an error; a value that is not an angular momentum at all (`j = 0.3`, or a
negative `j`) *is* an error.

```rust
use special_functions::wigner::{wigner_3j, clebsch_gordan};
// the triangle rule: |j1-j2| <= j3 <= j1+j2, or the symbol vanishes
assert_eq!(wigner_3j(1.0, 1.0, 5.0, 0.0, 0.0, 0.0)?, 0.0);
// two spin-1/2 particles coupling to the triplet m = 0 state: 1/sqrt(2)
let cg = clebsch_gordan(0.5, 0.5, 0.5, -0.5, 1.0, 0.0)?;
assert!((cg - (0.5_f64).sqrt()).abs() < 1e-14);
# Ok::<(), String>(())
```

---

## 11. The complex-argument and large-argument chapters

Ten further modules carry the material `grammar.md` §4.1 documents in
depth — the accuracy laws, the route selection and the measured error
surfaces live there, because that is the document a notebook user reads.
This table is the map from those entry points back to the source:

| module | what it holds |
|---|---|
| `complex` | `Complex64` (§7 above) |
| `gamma_complex` | `gamma_z`, `ln_gamma_z`, `rgamma_z` — Stirling with argument shifting, deliberately **not** Lanczos, whose coefficient tables are most often reproduced from *Numerical Recipes* |
| `bessel_complex` | `bessel_j_z`, `bessel_i_z`, `bessel_y_z`, `bessel_k_z` — whole order, complex argument |
| `bessel_cnu` | `bessel_*_nu` — any real **or complex** order, by ascending series and reflection |
| `bessel_cnu_large` | the uniform expansions for large order |
| `bessel_scaled` | `bessel_*_scaled`, `hankel_*_scaled` — the exponential factored out, so `K_0(2000)` exists at all |
| `hankel` | `hankel_h1_*`, `hankel_h2_*`, `sph_hankel_*` — the travelling-wave pair |
| `airy_complex` | `airy_z` → `[Ai, Ai', Bi, Bi']`, verified by the elementary Wronskian `Ai Bi' − Ai' Bi = 1/π` |
| `airy_uniform` | Olver's uniform Airy-type expansion (DLMF 10.20) at the turning point |
| `debye` | the Debye polynomials and the uniform expansions either side of it |
| `lanczos` | the Lanczos eigensolver used by the 2-D and 3-D bound-state solvers |

Each is reachable from the notebook under the name `grammar.md` §4.1 gives
it; none of them adds a grammar production, because registering a builtin
is not a grammar change.
