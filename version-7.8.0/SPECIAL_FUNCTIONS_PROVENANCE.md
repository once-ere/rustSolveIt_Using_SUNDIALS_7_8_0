# Special functions — provenance, coverage, and test results

**Date:** 2026-07-26
**Scope:** the `special_functions` crate and its exposure through the
posim language.

This is the report the DLMF task asked for. It leads with the coverage
number because that is the fact most likely to be overstated, and
everything else here is only worth as much as that number is honest.

---

## 1. Coverage, stated first

> **13 of the DLMF's 33 function chapters have some coverage. None of
> the 33 is complete.**

That is roughly 40 % of chapters *touched* and considerably less than
40 % of DLMF content implemented. Twenty chapters have nothing at all.

The original task was "port the complete NIST DLMF". That is not what
this is, and the reason is structural rather than an excuse:

- **No pure-Rust DLMF port exists.** `DLMF language:Rust` on GitHub
  returns zero repositories; a crates.io full-text search for "dlmf"
  returns two unrelated crates. The alternative branch of the task —
  import rather than port — has nothing to import.
- **No complete DLMF implementation exists in any language.**
  Mathematica reaches roughly 90 % of the function chapters (Heun only
  since 2020), Maple ~85 %, mpmath ~70 %, Arb/FLINT ~65 %, SciPy ~55 %,
  GSL ~48 %, Boost.Math ~42 %.
- **The DLMF is a reference handbook of properties, not an algorithm
  specification.** Most chapters state identities and asymptotics with
  no numerically stable evaluation recipe. "Porting" a chapter is
  numerical-analysis research, not transcription. Chapters 29 (Lamé),
  32 (Painlevé), 35 (matrix argument) and 36 (coalescing saddles) have
  no solid general implementation in *any* system.

So this milestone implements what a physics simulator actually consumes,
and the matrix below states plainly what is absent.

### The 33-chapter matrix

| ch. | title | status | what exists here |
|---|---|---|---|
| 4 | Elementary Functions | partial | `sqrt abs sin cos exp log` in the language; Rust `std` underneath |
| 5 | Gamma Function | partial | **native** Γ, lnΓ and 1/Γ at COMPLEX argument (Stirling with shifting); **vendored** Cephes: gamma, lgamma, rgamma, digamma, poch |
| 6 | Exponential, Log, Sine and Cosine Integrals | partial | **vendored**: Ei, Eₙ, Si, Ci, Shi, Chi |
| 7 | Error Functions, Dawson, Fresnel | partial | **vendored**: erf, erfc + inverses, Dawson, Fresnel |
| 8 | Incomplete Gamma and Related | partial | **vendored**: incomplete gamma/beta + inverses |
| 9 | Airy and Related | partial | **native** Ai, Ai', Bi, Bi' at COMPLEX argument; **vendored**: the same, real argument |
| 10 | Bessel Functions | partial | **vendored** cylindrical Jᵥ Yᵥ Iᵥ Kᵥ (real arg); **native** spherical jₙ yₙ and derivatives; **native** integer-order Jₙ whole-table; **native Jₙ, Yₙ, Iₙ, Kₙ for COMPLEX argument**; **native J_ν, Y_ν, I_ν, K_ν for real NON-INTEGER order at complex argument**; **native H^(1), H^(2) and their derivatives, cylindrical (any real order, complex argument) and spherical (real argument)**; **native SCALED forms of all six by the asymptotic expansions of DLMF 10.17/10.40 plus order recurrence**; **native COMPLEX ORDER for J, Y, I, K** |
| 11 | Struve and Related | **none** | — |
| 12 | Parabolic Cylinder | **none** | — |
| 13 | Confluent Hypergeometric | **none** | — |
| 14 | Legendre and Related | partial | **native**: Pₙ, P′ₙ, Pₗᵐ, normalised P̄ₗᵐ, Yₗᵐ complex and real |
| 15 | Hypergeometric Function | **none** | — |
| 16 | Generalized Hypergeometric, Meijer G | **none** | — |
| 17 | q-Hypergeometric | **none** | — |
| 18 | Orthogonal Polynomials | partial | **native**: Hermite Hₙ/Heₙ, Laguerre Lₙ/Lₙ^α, Chebyshev Tₙ/Uₙ, Gegenbauer Cₙ^α, Jacobi Pₙ^{α,β}. The discrete Askey-scheme families are absent |
| 19 | Elliptic Integrals | partial | **vendored**: Legendre and Carlson forms |
| 20 | Theta Functions | **none** | — |
| 21 | Multidimensional Theta | **none** | — |
| 22 | Jacobian Elliptic Functions | partial | **vendored**: sn, cn, dn |
| 23 | Weierstrass Elliptic and Modular | **none** | — |
| 24 | Bernoulli and Euler Polynomials | **none** | — |
| 25 | Zeta and Related | partial | **vendored**: ζ, Hurwitz ζ, dilogarithm |
| 26 | Combinatorial Analysis | **none** | — |
| 27 | Functions of Number Theory | **none** | — |
| 28 | Mathieu Functions and Hill's Equation | **none** | — |
| 29 | Lamé Functions | **none** | no general implementation exists anywhere |
| 30 | Spheroidal Wave Functions | **none** | — |
| 31 | Heun Functions | **none** | — |
| 32 | Painlevé Transcendents | **none** | no general implementation exists anywhere |
| 33 | Coulomb Functions | **none** | — |
| 34 | 3j, 6j, 9j Symbols | partial | **native**: 3-j, 6-j, **9-j**, Clebsch–Gordan. The chapter's asymptotics and generating functions are absent |
| 35 | Functions of Matrix Argument | **none** | no general implementation exists anywhere |
| 36 | Integrals with Coalescing Saddles | **none** | no general implementation exists anywhere |

Beyond the DLMF chapters, the crate also carries the numerical
infrastructure a simulator needs and the DLMF does not catalogue:
`eigen` (cyclic Jacobi, dense), `lanczos` (matrix-free symmetric
eigensolver for problems too large to form), `quadrature`
(Gauss–Legendre, adaptive Simpson, Brent roots), `tridiag` (Thomas,
Sherman–Morrison) and `complex`.

---

## 2. Licensing and provenance

### DLMF itself

The DLMF is **not** public domain — its authors assigned copyright to
NIST, which permits limited copying for research and teaching and
forbids commercial reproduction. NIST publishes no reference
implementation.

The policy followed here: **cite DLMF equation numbers and implement the
mathematics independently.** No DLMF prose is reproduced and no equation
displays are bulk-transcribed. Where a formula needed to be *visible* in
documentation it is taken from Abramowitz & Stegun (1964), a genuine US
Government work in the public domain.

### Vendored code

`vendor/spec_math/` — from <https://github.com/matthew-romanowicz/spec_math>,
a Rust translation of Cephes (Stephen L. Moshier). 102 files, **101
byte-identical to upstream**; only `src/lib.rs` differs, carrying a
provenance header plus `#![forbid(unsafe_code)]` and
`#![allow(dead_code)]`. Zero runtime dependencies, no `unsafe`.

**Licensing caveat, recorded rather than papered over:** the upstream
repository ships *no LICENSE file*. `MIT OR Apache-2.0` is declared only
in its `Cargo.toml`. That declaration is reproduced in `THIRD_PARTY.md`
along with the Cephes lineage, and it is flagged for resolution upstream
if certainty is wanted. See [THIRD_PARTY.md](THIRD_PARTY.md).

**Rejected, with reasons:** `puruspe` — algorithms and identifiers derive
from *Numerical Recipes*, whose licence forbids redistribution.
`scilib` — GPL-3.0, would infect the repository, and dead since 2023.
`libm` — 21 `unsafe` sites.

### Clean-room replacements

`bessel`, `tridiag` and `complex` replace licence-encumbered routines
from the SolveIt C++ sources (Numerical Recipes and GSL). Their audit
record is separate and detailed:
[CLEANROOM_PROVENANCE.md](CLEANROOM_PROVENANCE.md).

---

## 3. Per-module provenance

| module | origin | algorithm | citation |
|---|---|---|---|
| `sph_bessel` | native | Miller downward recurrence below `n > x`, upward above | DLMF 10.51, A&S 10.1 |
| `legendre` | native | ascending recurrence in ℓ seeded by Pₘᵐ; normalised form computed *directly* in the normalised basis so it never overflows | DLMF 14.10, A&S 8.5 |
| `orthopoly` | native | three-term recurrences; Clenshaw for series | DLMF 18.9, A&S 22.7 |
| `wigner` | native | Racah single-sum for 3-j and 6-j, all factorials in logarithms; 9-j as a single sum over 6-j | DLMF 34.2.4, 34.4.1, 34.6.1; Edmonds 1957 §3.6 |
| `bessel` | native, **clean-room** | Miller downward recurrence, scale fixed by `J₀+2(J₂+J₄+…)=1` | DLMF 10.6.1, 10.12.4; A&S 9.1.27, 9.1.46 |
| `airy_complex` | native | Ascending series (DLMF 9.4.1–9.4.4) to `\|z\| ≈ 6`; asymptotic expansions (9.7.5, 9.7.6) in `ζ = ⅔z^{3/2}` for `\|arg z\| ≤ 2π/3`, with `u_k = (6k−5)(6k−3)(6k−1)/(216k(2k−1))u_{k−1}` and `v_k = −(6k+1)/(6k−1)u_k`; and nearer the cut the connection `Ai(z) + ωAi(ωz) + ω²Ai(ω²z) = 0` (9.2.12), which rotates into the good sector. `Bi` from `Ai` at the two rotated points (9.2.10), never needing its own expansion — its asymptotic is valid only for `\|arg z\| < π/3` | DLMF 9.2.3–9.2.7, 9.2.10, 9.2.12, 9.4.1–9.4.4, 9.7.5, 9.7.6 |
| `gamma_complex` | native | `ln Γ` by the Stirling asymptotic series with **argument shifting** — `Γ(z+1) = zΓ(z)` repeatedly until `Re z ≥ 14`, then `(z−½)ln z − z + ½ln2π + Σ B₂ₙ/(2n(2n−1)z^{2n−1})`; reflection `Γ(z)Γ(1−z) = π/sin πz` for the left half plane. **Lanczos was rejected on licensing grounds**: its coefficients are a table, and the circulating tables are most often reproduced from *Numerical Recipes*. Stirling needs only the Bernoulli numbers, which a test re-derives from `Σ C(m+1,j)B_j = 0` | DLMF 5.5.1, 5.5.3, 5.11.1, 5.11.3 |
| `bessel_cnu` | native | The same ascending series as `bessel_complex`, with `1/Γ(ν+k+1)` advanced by its own recurrence so the complex gamma is evaluated once; `Y` and `K` by the reflections, which are **better** conditioned off the real axis since `\|sin νπ\|` grows like `e^{π\|Im ν\|}`. A real order (within 1e-13) is handed to the real-order routines, which handle negative whole orders the recurrence cannot restart from | DLMF 10.2.2, 10.2.3, 10.25.2, 10.27.4 |
| `bessel_cnu_large` | native | The `1/z` asymptotics of DLMF 10.17.5/6 and 10.40.1/2 at **complex order** — the order enters only as `μ = 4ν²`, polynomially, so they extend unchanged — plus the DLMF 10.41 uniform expansions in the sector `\|arg ν\| < π/2` (10.41.5). Two validity facts the truncation estimate cannot see are checked separately: `\|4ν²\| ≤ 2\|z\|`, and the `exp(π\|Im ν\| − 2Re z)` size of the term 10.40.1 drops | DLMF 10.17.5, 10.17.6, 10.40.1, 10.40.2, 10.41.3–10.41.5 |
| `airy_uniform` (complex order) | native | DLMF 10.20 at complex order, restricted to `\|1 − z/ν\| ≤ 0.25`. Needed no new mathematics: near the turning point `ζ`, the prefactor and `A_k`, `B_k` all come from the generated `w = 1 − x` Taylor series, which are complex-safe by construction. The closed forms outside that neighbourhood do **not** continue naively — `ζ`'s two branch formulas meet at `x = 1` and the principal ⅔ power does not carry across — but 10.20 is the wrong tool there anyway | DLMF 10.20.4, 10.20.5 |
| `airy_uniform` | native | Olver's uniform Airy-type expansion, DLMF 10.20.4/10.20.5, valid **through** the turning point `z = ν`. `ζ(x)` from 10.20.2/3; `A_k`, `B_k` from 10.20.10/11 built on the same Debye polynomials, with `λ_j`, `μ_j` from 10.20.12/13. Their closed forms cancel catastrophically as `ζ → 0` (each term is `O(w^{−3(2k+1)/2})` where the sum is `O(1)`), so near the turning point they come from Taylor series in `w = 1−x` **generated at 70 decimal digits** and verified in-crate against the closed forms. `A₁(0) = −1/225` and `ζ'(1) = −2^{1/3}` are known independently and both come out right | DLMF 10.20.2–10.20.5, 10.20.10–10.20.13 |
| `debye` (oscillatory) | native | The same `F±` formulas continued past `x = 1`, where they read as the Hankel functions instead of `J` and `Y`: `H1 = 2F₊`, `H2 = 2iF₋`, so `J = F₊ + iF₋` and `Y = −iF₊ − F₋`. **Those constants were identified by experiment**, by continuing `F₊` and dividing by each of `J`, `Y`, `H1`, `H2` in turn, then checked against `bessel_j_c`/`bessel_y_c` wherever those are independently sound. Works at complex order; guarded to `\|ν\| ≥ 8`, since a `1/ν` series' terms looking small is not the same as `ν` being large | DLMF 10.19.3, 10.19.4, 10.19.6 |
| `debye` | native | The Debye polynomials `U_k(p)` by the exact recurrence `U_{k+1} = ½p²(1−p²)U_k' + ⅛∫₀^p(1−5t²)U_k` (DLMF 10.41.9), carried out on coefficient vectors rather than transcribed; then the uniform large-order expansions `I_ν`, `K_ν` (10.41.3/4) and the Debye `J_ν`, `Y_ν` for `z < ν` (10.19.3/4). Optimal truncation supplies the error estimate, so these compete with the `1/z` routes on measured terms | DLMF 10.19.3, 10.19.4, 10.41.3, 10.41.4, 10.41.9, 10.41.10 |
| `bessel_scaled` | native | The asymptotic expansions `e^{-iz}H1 ~ sqrt(2/πz)e^{-i(νπ/2+π/4)}S(i)`, `e^zK ~ sqrt(π/2z)S(1)`, `e^{-z}I ~ S(-1)/sqrt(2πz)` with `a_k = a_{k-1}(4ν²-(2k-1)²)/8k`; `J` and `Y` from the Hankel pair without forming the envelope; optimal truncation supplying its own error estimate, which then **selects between the asymptotic and the ascending series by comparing estimates**; upward order recurrence for `K`, `Y`, `H1`, `H2`; the I–K Wronskian with a continued-fraction ratio anchoring `I` at large order | DLMF 10.6.1, 10.17.5, 10.17.6, 10.28.2, 10.29.1, 10.34.1, 10.27.6, 10.40.1, 10.40.2 |
| `hankel` | native | `H1 = J + iY`, `H2 = J - iY` (DLMF 10.4.3) over both the integer- and non-integer-order routines; derivatives from `C'_ν = C_{ν-1} - (ν/z)C_ν` (DLMF 10.6.2), which holds for every cylinder function, with `C'_0 = -C_1` for the one order that would need `C_{-1}`; spherical `h1 = j_n + i y_n` (DLMF 10.47.5) on the real line | DLMF 10.2.5, 10.4.3, 10.5.4, 10.6.2, 10.27.8, 10.47.5, 10.49.6, 10.50.1 |
| `bessel_complex` (non-integer order) | native | One ascending series for J and one for I, evaluated at ±ν; Y and K then follow from the reflection formulas, which are singular at whole ν and so are handed to the integer routines within 1e-9 of one. `1/Γ` is taken from the vendored reciprocal gamma, which is **zero** at the poles — that is what makes `J_{−n} = (−1)ⁿJₙ` fall out, and it keeps ν ≳ 171 in range where Γ itself would overflow | DLMF 10.2.2, 10.2.3, 10.25.2, 10.27.4 |
| `bessel_complex` | native | Jₙ/Iₙ by the same Miller recurrence (both it and the normalisation are identities in `z`); Yₙ by the ascending series with the log and digamma terms, then **upward** recurrence — the stable direction for Y and the opposite of J's; Kₙ by identity from Jₙ + iYₙ | DLMF 10.6.1, 10.8.1, 10.27.6, 10.27.8, 10.35.1 |
| `tridiag` | native, **clean-room** | Thomas algorithm; Sherman–Morrison for the cyclic case | textbook |
| `eigen` | native | cyclic Jacobi, ≤100 sweeps | textbook |
| `lanczos` | native | Lanczos with full reorthogonalisation + deflation, matrix-free | textbook |
| `quadrature` | native | Gauss–Legendre via Newton on Legendre roots; adaptive Simpson; Brent | textbook |
| `complex` | native | from the definitions | — |
| chapters 5–9, 19, 22, 25 | **vendored** Cephes | Moshier's rational approximations and continued fractions | upstream |

---

## 4. Test results, and what they actually establish

**568 passed workspace-wide; zero failures; zero build warnings;
`cargo clippy --workspace --all-targets` reports zero errors and zero
warnings.**

| suite | count |
|---|---|
| `special_functions` unit tests | 98 |
| `special_functions` vendored-identity tests | 11 |
| doctests | 31 |
| `posim` (incl. 12 special-function bridge tests) | 51 |
| `physical_object` unit | 40 |
| collision integration | 16 |
| conservation integration | 9 |
| **total** | **256** |

Per module: `sph_bessel` 7, `legendre` 10, `orthopoly` 24, `eigen` 8,
`quadrature` 24, `tridiag` 6, `bessel` 6, `wigner` 11, `complex` 2.

### What the tests are designed to catch

The suite deliberately avoids leaning on fixed tolerances against
remembered constants, because **that is exactly what failed repeatedly
during development**. Four times a hand-supplied "expected" value was
wrong while the code was right:

1. a tridiagonal 3×3 solution (`[0.25, 0.5, 1.25]` asserted; the true
   answer is `[0.5, 0, 1.5]`);
2. `legendre_p(2, 5.0)` asserted to be an error, when Pₙ is a polynomial
   defined on all of ℝ and P₂(5) = 37;
3. the 6-j value `{1 1 2; 1 1 2}` asserted as −1/30; it is **+1/30**;
4. a 6-j orthogonality test whose `p` range ignored the (a,d,p) and
   (c,b,p) triangle conditions, reporting a failure at a legitimate zero.

In every case a *structural* check — a residual, an identity, an
orthogonality sum — was already passing and was correct. The lesson is
recorded here because it shaped the suite:

- **Residuals over expected values.** `‖Ax − b‖ → 0` cannot be
  misremembered.
- **Orthogonality and normalisation sums.** For `wigner`, the
  Clebsch–Gordan rows summing to 1 and the 6-j relation
  `Σₓ(2x+1){a b x; c d p}{a b x; c d q} = δₚq/(2p+1)` fix sign *and*
  normalisation absolutely, with no table lookup. They are also the
  tests most sensitive to cancellation in the alternating Racah sum,
  which is the real accuracy risk in that module.
- **Closed forms swept over a range**, not spot-checked. `(j j 0; m −m 0)
  = (−1)^{j−m}/√(2j+1)` is verified across integer and half-integer j
  and every m; `{a b c; 0 c b}` likewise.
- **Convergence laws instead of tolerances.** The particle-in-a-box
  error is asserted to equal `(kπh)²/12`; the oscillator's
  finite-difference error to track `2n²+2n+1`; halving `h` to drop the
  error 4×. These fail if the *physics* is wrong, and pass at whatever
  absolute magnitude the grid dictates.
- **Cross-validation against independent machinery.** The native
  integer-order `bessel_j_array` is checked against the vendored Cephes
  `jv` — two entirely unrelated algorithms — and against the native
  spherical Bessel via the half-integer identity.
- **Mutation testing.** `scripts/mutation_probe.sh` breaks the program
  deliberately, in the places where this project's real defects have
  historically lived — branch choices, guard thresholds, safety factors,
  sampling offsets — and requires the suite to catch each one. A passing
  suite is evidence only if a broken program would fail it.

  Progression: **10/14** on the first run, **12/14** after Stage 2G,
  **18/19** after Stage 2H widened the table (complex gamma's Stirling
  shift, complex Airy's series/asymptotic switch and its `Ai(0)`
  constant, the restitution threshold, and the Zeno guard's *effect* as
  distinct from its threshold). The survivors, worked through rather
  than filed:

  | mutation | verdict |
  |---|---|
  | `zeta-anchor` (1.5 → 1.0 in the Stage 2D branch anchor) | **equivalent mutant.** Inside the guarded sector `\|arg(z/ν)\| ≤ 0.8` both coefficients unwrap to the same branch, so no input distinguishes them. It only bites where the route already refuses. |
  | `asym-floor` (5e-14 → 1e-300) | **closed in 2G.** The floor binds where optimal truncation reports *zero*, which happens exactly at `ν = 1/2` where the `1/z` series terminates — Stage 15's defect. Pinned against the closed form `J_{1/2}(z) = √(2/πz) sin z`. |
  | `cap-cells` (200 → 8) | **closed in 2G.** The convergence test rolled its own geometry, so the shipped constant was unasserted; `leak` is now compared against a 16× finer computation. |
  | `zeno-count` (64 → 1) | **resolved in 2H, by asking a better question.** The threshold's exact value is unconstrained because `resolve_impulses` batches every flagged pair into ONE event, so bursts above 1 arise only in true Zeno — where any value in [1, 64] gives the same intended plastic behaviour. What matters is the guard's *behaviour*, and two replacement mutations pin it: forcing it always on (`= 0`) and never on (`= usize::MAX`) are **both caught**. The number 64 is a tuning choice inside a range where nothing observable changes, which is a different thing from an untested guard. |

  Investigating `zeta-anchor` produced the more useful finding: the
  Stage 2D verification compared the closed form with the series **in
  their overlap**, which is where the answer was already known.
  `the_branch_anchor_is_constrained_away_from_the_turning_point` now
  exercises the extended route far from the turning point against the
  `1/z` Hankel pair — the test that stage should have had, whether or
  not it moves this particular mutant.

### When the instrument is the thing that is wrong

Stage 24 is recorded separately because what it found was not a wrong
value but a **wrong measurement**, and the wrong measurement had been
certifying four other stages.

The J–Y Wronskian `J_{ν+1}Y_ν − J_νY_{ν+1} = 2/(πz)` had been the
crate's sharpest check: it holds at every order, its right-hand side is
elementary, and no consistently-wrong pair can satisfy it. From Stage 13
onward the residual was divided by the largest product formed, "so the
metric's own cancellation is divided out".

That scaling is valid on the real axis and **vacuous off it**. `J` and
`Y` are both `(H1 ± H2)/2`, so wherever one Hankel function dominates —
any complex order off the real axis, or any order with appreciable
`|Im z|` — they are *the same function* to within `|H1/H2|`. The
Wronskian's true size is `|H1 H2|` while each product forming it is
`|H2|²`, so the scaled residual is `|H1/H2|` and nothing else. Measured
at `ν = 5 + 2i, z = 200 + 80i`: `|H1| = 2.2e-35`, `|H2| = 1.3e32`, and
the scaled residual came out **8.2e-24** — reported as a triumph, and
in fact just the ratio. Removing the scaling does not help; it would
then demand an accuracy of `|H1/H2|` relative, which no correct
implementation can deliver. **The J–Y Wronskian cannot resolve below
`|H1/H2|`, under any scaling.** The four sweeps that used it are now
floored at that resolution, and `hankel_ratio` exists to compute it.

Three consequences followed immediately, all in `debye::jy_debye_c`:

1. **`t = (1 − x²)^{1/2}` on the principal branch**, whose cut is
   `x²` real ≥ 1 — *precisely the oscillatory region the expansion
   exists to cover*. Crossing it negates `t`, which exchanges the two
   solutions, so `H1` was returned as `H2`. At `ν = 5 + 2i, z = 60 + 30i`
   the answer was wrong by `|H2/H1| = 2e23`.
2. **The same defect at real order.** `x = z/ν` is complex as soon as
   `z` is, so `ν = 20, z = 300 + 40i` fired it too. This was shipping.
3. **The prefactor `(2πνt)^{-1/2}`, a separate branch**, crossed once
   `arg z > π/2`, negating *both* members. Every bilinear check —
   the Wronskian included — is blind to a shared sign, so this one
   survived by construction.

Moving the cut is not sufficient (`i(x²−1)^{1/2}` merely relocates it to
the imaginary-`x` ray, and measurement showed `arg(z/ν) = 1.2` still
swapping). The branch is now **chosen against the answer's own leading
exponent**, `ν(t − α) → i(z − νπ/2)` from DLMF 10.17.5, and the
prefactor's argument is **unwrapped** rather than taken principal.
Reverting either fix fails
`debye::tests::the_branch_choices_are_right_off_the_real_axis`.

The general lesson, and the reason this section exists: *an identity
that is bilinear in the quantities under test cannot see a shared sign,
and an identity scaled by its own largest term cannot see below the
ratio of the terms.* Both are cheap to check and neither had been.

### Observed accuracy, and where the worst error lives

| module | observed | note |
|---|---|---|
| `sph_bessel` | ≤1e-14 against closed forms | Miller recurrence; the upward direction is *proved unstable* by a test rather than merely asserted |
| `legendre` | ≤1e-13 | `assoc_legendre_p` **overflows f64** and returns `Err`; the driver is the ORDER m, via the (2m−1)!! seed — not ℓ, as an earlier draft of the docs wrongly claimed. `norm_assoc_legendre_p` stays O(1) and is the fix |
| `orthopoly` | ≤1e-13 | worst at high degree with large argument, as the recurrences predict |
| `bessel_complex` (non-integer order) | ~1e-16·e^L, where L = \|z\|−\|Im z\| for J and Y and L = \|z\|+Re z for I and K | **weakest regime:** J and Y on the real axis, I and K on the *positive* real axis — the two families fail in opposite directions and the integer-order advice does not transfer. Machine precision up the imaginary axis to \|z\|=70. Large **order** is free (1e-13 at ν=150). The bound 1e-14·e^L is pinned by `documented_accuracy_bounds_hold`; the surface is printed by `examples/bessel_nu_accuracy.rs`, measured against the half-integer closed forms |
| `bessel_complex` (10.20 claim) | corrected in Stage 16 | The previous stage argued DLMF 10.20 was unnecessary, from a measurement across `z/ν` of 0.95 to 1.1 that showed 1e-14. **The sampling was too coarse and the conclusion was wrong**: at 0.85 and 0.90 the error reached 1.4e-9. The `O(ν⁻²)` figure quoted against it was also wrong — that is the expansion truncated at `A₀, B₀`, and three terms are kept, giving `O(ν⁻⁶)`. Adding it improved the band by three to four orders |
| `bessel_scaled` (selection) | fixed in Stage 16 | The order recurrence's error estimate was floored at one `eps` regardless of how many steps it took, so at `ν = 100.5, x = 85.4` it claimed 1.4e-11 while delivering 1.4e-9 — and on that claim it beat the Airy-type expansion, which was giving 3.7e-15. Floored at `steps × eps`. **Adding a better method is worth nothing if the comparison that selects it is dishonest** |
| `bessel_complex` (branch) | fixed in Stage 14 | `bessel_k_c` was on the **wrong sheet** for `arg z > π/2`: the identity it uses rotates the argument by `i`, which pushed `arg(iz)` past `π` and wrapped `ln` to the far side of `Y`'s cut. The I–K Wronskian residual was 3e8 at `arg z = 2.5` where the accuracy law predicts 5e-8. Fixed by conjugation (`K_ν(z̄) = conj K_ν(z)`), pinned by `k_stays_on_the_right_sheet_past_the_imaginary_axis`. Found by the Stage-14 cross-check of the scaled routines against the series |
| `bessel_complex` (delivered) | J–Y Wronskian ≤1e-10, mostly ≤1e-15, over `\|z\|` 5–40 across the upper half plane | This is what the FUNCTIONS give since Stage 19. The four route laws below are what each individual METHOD costs, and are now upper bounds used by the selector rather than descriptions of the result |
| `bessel_complex` | **four laws, not one** — ~1e-16·e^L with L = \|Im z\| for J, \|Re z\| for I, \|z\|−\|Im z\| for Y, and max(2\|Re z\|,\|z\|)+Re z for K | **This row was wrong until Stage 13.** It quoted J's law for the whole module. J and I are Miller recurrence and hold up superbly (1e-15 at x=35 on their good axis); Y is an ascending series and K is built from J and Y at imaginary argument, so on the **real** axis — where J is at its best — Y is wrong in the first digit by x=40 and K is worthless past x≈12. The old measurement used only the generating-function identity, which involves no Y at all; a Hankel asymptotic test at x=40 exposed it. Now measured against six independent Cephes cross-checks and pinned by `integer_order_accuracy_laws_hold` |
| `bessel_cnu_large` | worst chosen-value J–Y Wronskian residual **1.5e-7** over a 15905-point sweep in order, \|z\| and arg z; typically 1e-14 or better | **weakest regime:** `\|z\|` comparable to `\|ν\|` — the turning point, which needs DLMF 10.20 at complex order and therefore **complex Airy**, not implemented. Three estimate defects found by the sweep and fixed: the `\|4ν²\|` validity condition (actual error 2163× the estimate without it), the `exp(π\|Im ν\|)` factor on 10.40.1's dropped term (678×), and a safety factor of 150 rather than 10 on optimal truncation at moderate `\|z\|` (110×) |
| `bessel_complex` (the cut) | fixed in Stage 20 | The two `1/z` Hankel expansions have sectors ending at `arg z = π`, leaving a wedge either side of the cut where only the ascending series applied — J–Y Wronskian residual 1.0 at `\|z\| = 60`. Closed by the continuation `w = −z` (DLMF 10.11.3/4 with `m = ±1`), which is exact algebra and is tested as such where both routes overlap. `K` needed only its sector widened, DLMF 10.40.2 being valid to `\|arg z\| < 3π/2`. Verified by the exact identities `J_n(xe^{iπ}) = (−1)ⁿJ_n(x)` and `Y_n(xe^{iπ}) = (−1)ⁿ[Y_n(x) + 2iJ_n(x)]` — the Wronskian is useless on the cut, being dominated there by the recessive Hankel member. `J` exact, `Y` 1e-14 to `\|z\| = 300`, wedge 1e-12; branch jump still exactly `4i(−1)ⁿJ_n` |
| `bessel_complex` (routes) | fixed in Stage 19 | `J`, `Y` and `K` each gained a second route, chosen against the first by estimate: `J` and `Y` near the imaginary axis from `I`/`K` on the rotated argument `w = −iz` (DLMF 10.27.6 and the connection formula `Y_n(z) = i^{n+1}I_n(w) − (2/π)i^{−n}K_n(w)`), and `Y` on the real axis and `K` everywhere from their own `1/z` expansions. All are built on `bessel_cnu_large`, whose expansions are self-contained series — **anything reaching back through the scaled or non-integer routines would recurse forever**, since `bessel_k_c` is built on `bessel_y_c`. Measured after: J–Y Wronskian residual ≤1e-10 and mostly ≤1e-15 over `\|z\|` 5–40 and the upper half plane, against 1e-1 before |
| `bessel_complex` (integer Y) | **found in Stage 18, FIXED in Stage 19** | `bessel_y_c` builds `Y_n` by upward recurrence in `n`, which is stable for real argument but not near the **imaginary** axis, where `Y_n` is a combination whose recessive part the recurrence amplifies. At `n = 2, z = 29.4e^{1.6i}` the `1/z` expansion closes the J–Y Wronskian to 7e-26 and the integer series to 4.5e-6. The Stage-13 accuracy law does not describe this. `bessel_cnu` carries an `exp(\|Im z\|)` term so the selector stops trusting that route, and Stage 19 replaced the route itself |
| `airy_complex` | Wronskian residual **≤1e-11** over `\|z\|` 0.1–200 and the full range of `arg z`, except **1e-9** in the annulus `3 ≤ \|z\| ≤ 10`; typically 1e-15 | **weakest regime:** the positive real axis near `\|z\| = 6`, the crossover where the series has spent its digits on cancellation and the expansion is not yet converged — `Ai` is exponentially recessive there. Verified by `Ai Bi' − Ai' Bi = 1/π` (exact and elementary), the connection formula away from where it is used, conjugation symmetry, `Ai'' = zAi` by finite difference, and Cephes on the real axis. Two estimate terms were needed that the truncation alone misses: the series' own multi-term cancellation (×100, measured), and `exp(ζ)`'s rounding `3\|ζ\|ε`, without which `Bi(80)` claimed forty orders more accuracy than it had |
| `gamma_complex` | ~1e-14 relative; the limit is the fourteen shifts needed to reach the Stirling regime, each of which rounds | **weakest regime:** none in particular — verified against the closed form `\|Γ(1+iy)\|² = πy/sinh πy`, the recurrence, reflection and duplication formulas at complex argument, and Cephes on the real axis. A transcription bug in the Stirling loop (`position()` matched −1/30 twice, since B₄ = B₈) cost 3e-11 until the exact-value test caught it |
| `bessel_cnu` | inherits `bessel_complex`'s law with one term added: ~1e-16·e^L, L = \|z\|−\|Im z\|+Im ν·arg z. Complex order is **free on the positive real axis** | **weakest regime:** large \|Im ν\| well off the real axis. Verified by both Wronskians (whose right-hand sides do not involve the order at all), the order recurrence, conjugation symmetry — which a Wronskian cannot catch, since a consistently conjugated pair still satisfies it — and the classical fact that `K_{iy}(x)` is real |
| `airy_uniform` | 1e-14 across `z/ν` from 0.85 to 1.5 at every order tried to 1000; its own truncation estimate is `O(ν⁻⁶)` (three terms kept) — 7e-12 at ν=100, 4e-13 at ν=200 | **weakest regime:** small `ν`, where a `1/ν²` expansion has little to work with. Verified against Cephes and by the J–Y Wronskian. It improved the band `0.85 ≤ z/ν ≤ 0.98` by three to four orders — see the correction note below |
| `debye` | 1e-14 or better for `z/ν ≲ 0.85`; the truncation estimate grows to 1e-3 by `z/ν = 0.99` and says so | **weakest regime:** the turning point `z ≈ ν`, where `U_k(1/√(1−x²))` diverges. Verified by the **J–Y Wronskian**, elementary on the right, because at these orders Cephes is the less accurate party — at `ν = 400.5, x = 160` our Wronskian residual is 1.2e-13 and truncation estimate 4e-32 while Cephes `jv` disagrees by 1.4e-9. That comparison is an assertion, not a remark |
| `bessel_scaled` | machine precision essentially everywhere: worst measured disagreement with the ascending series, over 864 grid points where the series' own law says it is sound, is 4e-8 for J and 9e-7 for Y at extreme order; typically 1e-15 | **weakest regime:** `\|z\|` and `ν` large and comparable, which needs the uniform Airy-type expansions of DLMF 10.20 and **returns an error** rather than a number. Reaches values no unscaled routine can represent — `e^xK_0(2000)` where `K_0` is ~1e-870, `e^{-x}I_0(1000)` where `I_0` is ~e^1000, `H1` 700 nepers above the real axis — and orders where the vendored Cephes `kn` overflows |
| `hankel` | cylindrical: ~1e-16·e^(3\|Im z\|) on the bad side of each; spherical: as accurate as `sph_bessel`, i.e. everywhere | **weakest regime:** `H1` above the real axis and `H2` below it. `H1` decays like e^(−Im z) while J and Y each grow like e^(\|Im z\|), so the sum cancels — good to Im z≈8, three digits at 10, gone by 12. Intrinsic to the combination: switching between the `_z` and `_nu` routes changes nothing (measured, factor 1.5), since at whole order `_nu` delegates Y to `_z`. Same law as K on the real axis, because `K_ν(y) = (π/2)i^(ν+1)H1_ν(iy)` makes them the same computation |
| `bessel` | ≤1e-10 vs Cephes | **weakest regime:** the seed order must sit well above x. At `x = 45` an under-sized seed gave only ~9 correct digits — caught by the cross-check, fixed, and the measurement recorded in the source |
| `wigner` | ≤1e-12 on orthogonality sums | **weakest regime:** large j, where the alternating Racah sum cancels catastrophically. This is a property of the formula, not the implementation, and it is stated in the module docs |
| `tridiag` | residual ≤1e-11; CN norm drift **1.47e-12** over 6000 steps | no pivoting — stable for diagonally dominant systems only, documented rather than hidden |
| `eigen` | ≤1e-12 | dense, `O(n^3)` — practical to a few hundred rows |
| `lanczos` | residuals ≤1e-8 on 4900-dim problems | matrix-free; **weakest regime:** clustered-but-not-equal eigenvalues, where deflation needs more passes to separate them. Cross-checked against `eigen` on problems small enough for both |
| `quadrature` | degree-exactness to 2n−1, **and the converse** (not exact at 2n, so the bound is sharp) | |

### End-to-end evidence

Unit tests prove the pieces; three examples prove they do the job:

- `scatter_1d` — Gaussian packet off a rectangular barrier, 6000
  Crank–Nicolson steps. Norm drift **1.47e-12** (the Cayley operator is
  unitary for *any* dt, so this measures the solver, not the timestep);
  transmitted fraction within **0.25 %** of the analytic coefficient
  averaged over the packet's own momentum distribution.
- `harmonic_oscillator` — solved two independent ways (analytic Hermite
  eigenfunctions vs finite-difference diagonalisation) that share no
  code.
- `hard_sphere` — the documented expert example, σ → 4πa².

### What is NOT covered

- **Complex order** for the Bessel family. Order is real throughout;
  `bessel_j_nu(1.3, z)` is supported, `bessel_j_nu(1 + 2i, z)` is not.
- **Uniform asymptotics for large |z| at non-integer order.** The
  ascending series is the only method here, so the ranges in §5 are
  hard limits rather than performance advice. Where the order happens
  to be whole, the integer-order Miller routines reach much further
  along the real axis.
- **The band `1 < |z/ν| < 8` at large `|arg(z/ν)|`.** Stage 2D closed
  the same band inside `|arg(z/ν)| ≤ 0.8`, taking it from 41 % to
  **97.5 %** served, by extending DLMF 10.20 past the turning point:
  the obstacle was `ζ`'s branch under the principal `2/3` power, not
  the expansion's validity. Outside that sector the branch anchor
  (`arg F → 1.5 arg w`) is not measured, and the route refuses.
  *Formerly stated as:* the band `1 < |z/ν| < 8` away from the real
  axis, and `1 < |z/ν| < 2` at order past about 8. This replaces the earlier
  entry "a sliver at `4 ≲ |ν| ≲ 8`", which Stage 24 closed — that
  sliver was an artefact of guarding the Debye route by *order* when
  its accuracy is governed by `|z|/|ν|` and `arg(z/ν)`. Restating the
  guard in the measured variables took the sliver from 94% to 97%
  served. The band named here is what the same measurement exposed as
  genuinely uncovered: before Stage 24 those points were accepted with
  error estimates up to **1e14 times too small**.
- **`bessel_complex::bessel_y_nu` is wrong, not merely inaccurate, for
  non-integer `ν` past `|z| ≈ 30`.** Stage 2I measured what was
  recorded here as "a ridge near `z/ν ≈ 1.3` where `Y` reaches about
  1e-7 at moderate order". That entry understated it by four decades:
  the error is **3.09e4** relative at `ν = 36.8, z/ν = 1.48` and
  **2.18** at `z/ν = 1.30`. The 1e-7 figure was simply the value at
  `ν = 20.5`, where the sweep that produced it stopped.

  The cause is in the ingredient, not the combination:
  `Y_ν = [J_ν cos νπ − J_{−ν}]/sin νπ`, and `J_{−36.8}(54.5)` comes
  from an ascending series whose terms reach `e^54 ≈ 3e23` to produce a
  result of order 0.1.

  Adjudicated by the J–Y Wronskian against Cephes — our residual
  3.7e-3, Cephes 2.2e-23. **Cephes is right here**, the reverse of
  Stages 15 and 19 where it was the looser party; which one is correct
  has to be measured each time rather than assumed from precedent.

  **`bessel_cnu::bessel_y_cnu` is unaffected** and accurate to 4.8e-13
  at exactly those points, because it compares error estimates across
  routes. `the_selector_is_accurate_where_the_raw_reflection_is_not`
  pins both halves — the selector's accuracy and the raw route's
  failure — so neither can change silently.

  **Stage 2J added the guard**, and it is measured rather than modelled.
  The ascending series now reports its own cancellation —
  `max|term| / |sum|`, the largest quantity formed over the answer
  produced — and the reflection multiplies that by its own. The route
  refuses when `loss × eps` exceeds **1e-3**:

  | ν, z | actual error | `loss × eps` | |
  |---|---|---|---|
  | 36.8, 54.46 | 3.09e4 | 1.02 | refuse |
  | 36.8, 47.84 | 2.18 | 9.1e-3 | refuse |
  | 7.15, 61.8 | (near a zero of `Y`) | 8.7e-2 | refuse |
  | 20.5, 30.34 | 7.8e-8 | 1.9e-6 | allow |
  | 12.3, 16.60 | 7.5e-12 | 1.2e-11 | allow |

  The threshold separates every measured case with the closest allowed
  one 500× inside it. The indicator is **not** a proven bound — at
  `ν = 36.8, z = 47.84` the actual error is 240× larger than
  `loss × eps` — and the margin exists to carry exactly that, stated
  rather than implied.

  One existing test moved as a result: a Hankel asymptotic check used
  `ν = 1/2, z = 30`, which the guard now refuses. The module had
  already documented `Y` as "unusable past 30", so the guard agreeing
  with the documentation is the point — **the test moved in rather than
  the threshold moving out.**
- **A wider number type.** For `z` well below `ν`, `J` is below the
  smallest double and `Y` above the largest. The expansions determine
  those values; `f64` cannot carry them, and the routines say so and
  quote the logarithm rather than returning 0 or ∞.
- Accuracy at very large j in `wigner`, as above.
- The twenty chapters marked **none** in §1.

---

## 5. Language integration

All 30 registered functions are callable from posim; see
[grammar.md](grammar.md) §4.1 and §4.2 and Example 15.

The genuine parse-time work was **argument-domain checking**, not
syntax: the call production already admitted any builtin. An integer
order must be a whole number — `hermite_h(2.5, 1)` is refused rather
than truncated to `hermite_h(2, 1)`, which would return a confident
wrong answer. Angular momenta are the deliberate exception: they may be
half-integral, so the wigner entry points take plain numbers and
validate in the library.

`Value::Complex` and the imaginary literal `3i` were added to reach the
complex Crank–Nicolson solvers — a real lexer, parser and VM change
rather than a registration.

**The lockstep rule is now mechanical.** A test asserts that every
registered name appears in `HELP_TEXT`, the `parser.rs` EBNF comment,
`grammar.md` and `grammar.tex`; the build fails otherwise. It was
verified to actually fail by registering an undocumented name. This
replaced enforcement-by-discipline, which had already let the entire
integration be skipped once.

---

## 6. Honest summary

**Pros.** Everything present is tested by structural properties rather
than remembered constants, and cross-validated where two independent
routes exist. Zero `unsafe`, zero external dependencies, zero warnings,
zero clippy findings. The clean-room replacements remove real licensing
blockers. The mathematical infrastructure for 1-D quantum mechanics —
complex arithmetic, a unitary propagator, a dense eigensolver,
quadrature, the orthogonal polynomial families — is present and
demonstrated end to end.

**Cons.** Coverage is ~40 % of chapters and less than that of content.
Every one of the twenty untouched chapters is absent, as are complex
*order* and large-|z| asymptotics within chapter 10. `wigner` degrades
at large j and
`assoc_legendre_p` overflows at large order — both documented, neither
fixed. The vendored dependency has an unresolved LICENSE-file question
upstream. Two-dimensional quantum problems will need a different
linear-algebra path, since a full 2-D Crank–Nicolson operator is not
tridiagonal.

**The claim being made is narrow and, I believe, defensible:** this is a
tested, documented, licence-clean foundation sufficient for the 1-D
quantum mechanics the SolveIt port needs. It is not the DLMF.
