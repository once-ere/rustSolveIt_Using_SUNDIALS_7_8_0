# Project Status — rustSimulate

*An honest assessment: what exists, what is verified, what it is good
and bad at, and whether the mathematical infrastructure is ready to
carry quantum-mechanics problems.*

Date: 2026-07-26 · Repository: <https://github.com/once-ere/rustSimulate>

---

## 1. What exists today

| component | state | evidence |
|---|---|---|
| `physical_object` — the union struct, get/set API, observables | complete | 40 library tests |
| collision detection & impulse response | complete | 19 collision tests, analytic TOI checks |
| `integrate` — CVODE Adams/BDF + ARKODE SPRK, event rootfinding | complete | 9 conservation tests; solar-system energy drift < 1e-6 over 1369 years |
| `posim` — lexer → parser → stack machine → notebook/machine/scene | complete | 109 tests |
| `quantum` — nash / qm1d / qm2d / qm3d / transfer / absorber / isosurface | complete | 92 tests + 5 doctests |
| scene window (std-only HTTP + WebSocket + canvas) | complete | live browser verification |
| JupyterLab kernel | complete | protocol test + 7-cell ZMQ test |
| `sundials_rs` — pure-Rust SUNDIALS 7.8.0 | vendored, byte-identical | 2,929 files, tree hash verified |
| `special_functions` — spherical Bessel (native) | new | 7 unit + 3 doctests |
| `special_functions` — Cephes classical chapters | vendored | 11 identity tests |

**568 passed workspace-wide, zero warnings, zero `unsafe`, zero
crates.io dependencies.**

---

- **`quantum::nash`** — the Bessel-expanded split-operator propagator
  ported faithfully from the original C++ `EVOLVE_NASH`, on a periodic
  grid. Explicit and matrix-free where `qm1d`'s Crank–Nicolson solves a
  tridiagonal system; unitary at any step size; first order in `dt`.
  Verified against a transliteration of the original, against a closed
  form on plane waves, and against diagonalisation. See
  `CLEANROOM_PROVENANCE.md` §8.

- **`quantum::transfer`** — fixed-energy scattering by transfer matrix:
  `T(E)`, `R(E)`, flux balance, and a measured conditioning number. It
  is *exact* when the potential is piecewise constant on the cells and
  second order otherwise. This is what resolves resonances narrower than
  a wavepacket's momentum spread; see `TUNNELING_RESULTS.md` §5.
- **`quantum::absorber`** — the absorbing boundary, designed rather than
  tuned. The CAP's reflection and leakage are computed exactly from the
  same transfer matrix, so `choose_strength` optimises over a *band* of
  wavenumbers instead of one packet's nominal `k`. Reachable from the
  language as `QM TRANSMISSION` and `QM SCAN`.

## 2. Pros

**The dependency posture is genuinely unusual and valuable.** Nothing
comes from crates.io. `Cargo.lock` lists seven local crates and nothing
else, which is machine-checkable. A clone builds offline, forever, with
no supply-chain surface and no version drift. Very few numerical
projects can say that.

**The integration layer is real, not a toy.** Every trajectory comes
from SUNDIALS with proper error control or symplectic structure. The
outer-solar-system example reproduces the reference implementation's
Pluto position to eight decimals across 500 000 days.

**Verification is by independent means, not self-comparison.** Energy,
momentum and angular-momentum conservation; analytic times-of-impact;
Wronskians; closed forms; recurrence relations; and — the strongest
pattern — cross-validation between two independently written
implementations (our Miller-recurrence jₙ against Cephes's half-integer
Jᵥ).

**The failure modes are documented rather than hidden.** The
output-granularity defect and the parallel disk–disk rootfinding
limitation are written down with measurements, not buried.

**Provenance is auditable.** The export carries a SHA-256 manifest;
vendored code records its upstream commit and every intentional
modification.

**The language layer makes experiments cheap.** A physics question
becomes a few lines in a notebook, with a live 3-D window and a
JupyterLab kernel, instead of a recompile.

## 3. Cons

**Coverage of the DLMF is small and will stay small.** Roughly 11 of 33
function chapters, partially — call it ~15–20 % of content. That is not
a defect of effort; no complete implementation exists in any language.
But nobody should mistake this for a general special-function library.

**The vendored `spec_math` has a licence weakness.** No `LICENSE` file
upstream; `MIT OR Apache-2.0` is declared only in `Cargo.toml`. Plus a
Cephes-derivation question. Recorded in `THIRD_PARTY.md`, unresolved.

**The output-granularity defect is repaired.** It used to be listed
here as unfixed: the trajectory depended on how often output was
requested, `|dE/E|` of 6.9e-8 at interval 0.001 against 2.3e-1 at 0.125.
The cause was the Zeno guard counting events per *output interval*, so
ordinary elastic collisions were forced plastic when the caller asked
for fewer snapshots. The guard now counts a time-local **burst**, and
the coarse run conserves energy to 1.0e-7 while resolving 9160
collisions instead of 898. See `box_of_shapes_m32.md` §5.

**Everything is `f64`.** No arbitrary precision, no `f32`, no SIMD, and
values genuinely outside `f64` range are refused rather than returned.
*(This entry used to add "and real — no complex arithmetic anywhere".
That has not been true for many stages: the VM has a `Complex` value
type, and `special_functions` carries complex argument **and** complex
order throughout Bessel, Airy and gamma.)*

**Linear algebra stops at 3×3** in `physical_object`'s `linalg`.
*(Partly retired: `special_functions` now carries a dense Jacobi
eigensolver (`eigen.rs`), Lanczos for large symmetric operators
(`lanczos.rs`), and real/complex tridiagonal solvers (`tridiag.rs`).
There is still no general dense matrix type or decomposition beyond
those.)*

**No optimisation** as a reusable component. *(The quadrature and
root-finding halves of this entry are retired: `quadrature.rs` provides
Gauss–Legendre and adaptive quadrature, Brent root-finding and
`find_roots` scanning.)*

**Single-threaded and unprofiled**, apart from the scene playback
thread. Performance has never been measured against another simulator.

**The bus factor is one, and the test suite is the only specification.**

---

## 4. Is the mathematical infrastructure ready for quantum mechanics?

**Partly — and the honest answer is no, not for the general case.**
Here is the audit rather than a verdict.

### Verified and usable today

| capability | status | QM relevance |
|---|---|---|
| Spherical Bessel jₙ, yₙ, derivatives | ✅ native, Wronskian + cross-validated | free particle in spherical coordinates; partial-wave expansions; infinite spherical well |
| Airy Ai, Bi, derivatives | ✅ vendored, Wronskian-verified | WKB connection formulas; linear potentials; turning points |
| Γ, ψ, incomplete Γ/B | ✅ verified | normalisation constants throughout |
| erf, Fresnel, Ei/Si/Ci | ✅ verified | Gaussian wave packets |
| Elliptic K/E, Jacobi sn/cn/dn | ✅ verified | pendulum, anharmonic and periodic potentials |
| Adaptive/stiff ODE integration with event rootfinding | ✅ 9 conservation tests | time evolution; shooting methods for bound states |

### Missing, in order of how badly it blocks

*(This list is preserved as the 2026-07-26 audit; every item on it has
since been built. The italic notes record where.)*

1. **Complex arithmetic — the single biggest gap.** Wavefunctions are
   complex by construction. *(Retired: the VM has `Value::Complex`, and
   `special_functions` carries complex argument and complex order
   throughout `complex.rs`, `bessel_complex.rs`, `airy_complex.rs`,
   `gamma_complex.rs`.)*
2. **Associated Legendre Pₗᵐ and spherical harmonics Yₗᵐ.** Without
   them there is no angular part of *any* central-potential problem.
   *(Retired: built in `legendre.rs`, with normalised forms and real
   and complex `Yₗᵐ`.)*
3. **Generalised Laguerre Lₙ^α.** The hydrogen radial functions.
   *(Retired: built in `orthopoly.rs`.)*
4. **Hermite Hₙ.** The harmonic oscillator. *(Retired: built in
   `orthopoly.rs`, both conventions.)*
5. **Wigner 3j/6j and Clebsch–Gordan.** Angular-momentum coupling and
   selection rules. *(Retired: built in `wigner.rs`, including 9j.)*
6. **A general eigensolver.** Matrix mechanics *is* diagonalisation.
   *(Retired: `eigen.rs` (dense Jacobi) and `lanczos.rs` (large
   symmetric operators) power `QM STATES` in one, two and three
   dimensions.)*
7. **Quadrature.** Matrix elements ⟨ψ|V|ψ⟩ are integrals. *(Retired:
   `quadrature.rs` — Gauss–Legendre and adaptive.)*
8. **Root-finding as a component.** *(Retired: `brent_root` and
   `find_roots` in `quadrature.rs`, decoupled from CVODE's
   collision-event rootfinder.)*

### What could be built *today*, honestly

With only what is already verified: **partial-wave scattering** from a
hard sphere or square well (jₙ, yₙ are done and cross-validated), **WKB
tunnelling** through barriers using Airy connection formulas, and
**1-D time evolution** — because the time-dependent Schrödinger equation
can be integrated as *coupled real ODEs* by splitting ψ into real and
imaginary parts, which the existing CVODE layer already handles. That
last route sidesteps the complex-arithmetic gap entirely for dynamics,
though not for analysis.

What cannot be done today: hydrogen, the harmonic oscillator in its
usual basis, anything requiring diagonalisation, and anything requiring
genuine complex algebra at the language level.

---

## 5. Plan to make quantum mechanics tractable

Four phases, ordered so each ends with something demonstrable rather
than a half-finished layer.

### Phase A — the analytic eigenfunctions *(largest payoff per unit work)*

Complete the three planned native modules:

* `legendre` — Pₙ, Pₙ′, associated Pₗᵐ, normalised P̄ₗᵐ, and real/complex
  Yₗᵐ. Ascending recurrence in ℓ seeded by Pₘᵐ; compute the normalised
  form directly so large ℓ does not overflow.
* `orthopoly` — Hermite Hₙ and Heₙ, generalised Laguerre Lₙ^α,
  Chebyshev, Gegenbauer, Jacobi, via three-term recurrences.
* `wigner` — 3j, 6j, Clebsch–Gordan with log-gamma factorials and exact
  triangle/selection rules checked first.

*Verification:* orthonormality integrals (∫PₘPₙ = 2δₘₙ/(2n+1),
∮YₗᵐYₗ′ᵐ′* = δ), the spherical-harmonic addition theorem, Wigner
orthogonality and symmetry relations, and analytic energies.

*Demonstrable end state:* hydrogen orbitals and harmonic-oscillator
eigenstates evaluated and plotted in the scene window, with
normalisation and orthogonality verified numerically.

### Phase B — complex arithmetic end to end

A `Complex64` in `special_functions`, a `Value::Complex` in the VM with
literal syntax, and complex arguments for the functions that need them.
**This is the phase that requires genuine lexer + parser work**, unlike
adding real-valued builtins, which the existing `FUNC(args)` production
already accommodates. Vendoring `complex-bessel` (AMOS/TOMS 644) becomes
worthwhile here.

*Demonstrable end state:* wavefunctions manipulated directly in the
notebook — probability densities, expectation values, phases.

### Phase C — the numerical toolbox

A dense real-symmetric/Hermitian eigensolver (Jacobi rotations first:
~200 lines, unconditionally stable, easy to verify against known
spectra), Gauss–Legendre and Gauss–Hermite quadrature, and Brent
root-finding — all pure Rust, no dependencies.

*Verification:* eigenvalues of matrices with known spectra; quadrature
exact for polynomials up to the rule's degree; orthogonality of computed
eigenvectors.

*Demonstrable end state:* the variational method and matrix mechanics —
put any 1-D potential in a basis, diagonalise, get the spectrum.
Particle in a box, finite well, anharmonic oscillator, double well.

### Phase D — time-dependent dynamics

TDSE via real/imaginary splitting on the existing CVODE layer (can begin
during Phase A — it needs nothing new), then upgraded to genuine complex
integration after Phase B. Wave-packet scattering rendered live in the
scene window.

*Verification:* norm conservation, energy conservation for stationary
states, analytic Gaussian wave-packet spreading, and revival times.

### Sequencing note

Phase A is the highest value and lowest risk, and it is already planned
work. **Phase C is the one currently missing from every plan and is what
truly unlocks "quantum mechanics problems" in general** — without an
eigensolver you are restricted to the handful of analytically soluble
systems. If the goal is a QM teaching or research tool rather than a
demonstration of a few closed forms, Phase C should be promoted to run
alongside Phase A.
