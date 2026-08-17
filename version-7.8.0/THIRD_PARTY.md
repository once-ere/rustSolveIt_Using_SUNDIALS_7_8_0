# Third-Party Code

Everything rustSimulate builds against lives in this repository — there
are no crates.io dependencies. This file records where the vendored code
came from and under what terms.

---

## `sundials_rs/` — the numerical integration engine

| | |
|---|---|
| Upstream | `once-ere/SUNDIALS_7_8_Rust_port_for_Linux` @ `780b916` (2026-08-14) |
| What it is | A pure-Rust, zero-`unsafe`, dependency-free translation of SUNDIALS 7.8.0 (LLNL) |
| Licence | BSD-3-Clause (`sundials_rs/LICENSE`, `sundials_rs/NOTICE`) |
| Modified here? | **No.** 2,929 files, byte-identical; content hash `e0347199…` |

The predecessor of this repository, `once-ere/rustSolveIt`, vendored
`once-ere/sundials_rs@faabb7f` — the same translation at **7.7.0**, in
394 files. The upgrade to 7.8.0 and the evidence that it changed no
physics are recorded in
[`PORT_7.8.0_PROVENANCE.md`](PORT_7.8.0_PROVENANCE.md); the original
export's provenance is in
[`EXPORT_PROVENANCE.md`](EXPORT_PROVENANCE.md).

Reproduce the byte-identity claim:

```bash
git clone https://github.com/once-ere/SUNDIALS_7_8_Rust_port_for_Linux.git
diff -rq --exclude=.git --exclude=target \
     SUNDIALS_7_8_Rust_port_for_Linux sundials_rs
```

---

## `vendor/spec_math/` — the classical special-function chapters

| | |
|---|---|
| Upstream | <https://github.com/matthew-romanowicz/spec_math> @ `faa5c938d3714f890ffd8bffb0a43dc86b3d9ea9` |
| Author | Matthew Romanowicz |
| What it is | A Rust re-implementation of the **Cephes** mathematical library (Stephen L. Moshier) |
| Licence | `MIT OR Apache-2.0` — see the caveat below |
| Dependencies | **none** (verified: the `[dependencies]` table is empty) |
| `unsafe` | **none** (verified: zero occurrences in `src/`) |
| Size | 102 files, 19,787 lines |
| Modified here? | **One file.** 101 of 102 files are byte-identical to upstream; only `src/lib.rs` differs |

### The single modification

`src/lib.rs` gains a provenance header and exactly two attributes.
Nothing else in the crate is touched:

```rust
#![forbid(unsafe_code)]   // required by this project (CLAUDE.md rule 2)
#![allow(dead_code)]      // upstream leaves the Lanczos helpers unreferenced
```

The `allow(dead_code)` is not cosmetic: without it the crate emits seven
`never used` warnings for the Lanczos approximation constants and
helpers, which would break this project's zero-warning guarantee. The
upstream code is otherwise unaltered — no algorithm, constant or
signature was changed.

Resulting `src/lib.rs` SHA-256: `62f95e2dc532e65ec34197d5489c923d024a9bce45682dfba722132891ae475e`

### ⚠️ Licence caveat — read this

The upstream repository **contains no `LICENSE` file**. The
`MIT OR Apache-2.0` grant appears **only** in its `Cargo.toml`
(`license = "MIT OR Apache-2.0"`), which is also what was published to
crates.io. That is a public declaration of intent and is the basis on
which this code is vendored, but it is weaker than a licence text in the
repository.

Two things a maintainer of this project should know:

1. **Ask upstream to add `LICENSE-MIT` and `LICENSE-APACHE` files.**
   Until then, the declaration above is the whole of the grant.
2. **Cephes provenance.** The code is a translation of Stephen L.
   Moshier's Cephes library. SciPy vendors Cephes under a specific grant
   of permission from Moshier. Whether a Rust re-implementation may be
   relicensed as MIT/Apache-2.0 is a question worth a lawyer's glance
   before this repository is used commercially.

This is recorded rather than papered over so the risk is visible.

### What it provides

Mapped to DLMF chapters: **5** (gamma, log-gamma, reciprocal gamma,
digamma, Pochhammer), **6** (Ei, Eₙ, Si, Ci, Shi, Chi), **7** (erf,
erfc and inverses, Dawson, Fresnel), **8** (incomplete gamma and beta
with inverses), **9** (Airy, real argument), **10** (J₀ J₁ Jᵥ, Y₀ Y₁ Yₙ
Yᵥ, I₀ I₁, K₀ K₁ Kₙ — real argument), **19** (Legendre and Carlson
elliptic integrals), **22** (Jacobi sn, cn, dn), **25** (ζ, Hurwitz ζ,
dilogarithm), plus statistical distributions outside the DLMF's scope.

---

## Not vendored, and why

| candidate | reason rejected |
|---|---|
| `puruspe` | Algorithms and identifiers derive from *Numerical Recipes*, whose licence forbids redistribution — a real risk for vendored source in a public repository. |
| `scilib` | GPL-3.0. Vendoring would impose GPL-3.0 on this repository. Unmaintained since 2023. |
| `libm` | Contains `unsafe` (architecture intrinsics), which this project forbids. |
| `ellip` | BSD-3-Clause and excellent (published error tables, JOSS paper), but pulls four dependencies including a proc-macro. `spec_math` already covers the Legendre and Carlson forms. Reconsider if Bulirsch's `cel`/`el1`/`el2`/`el3` are ever needed. |
| `complex-bessel` | Wanted — a faithful AMOS/TOMS 644 port for the complex plane. Deferred: it needs `num-complex` and `num-traits` vendored too, and complex arguments require a new value type in the posim VM. Staged for a later milestone rather than half-done. |
| `rgsl`, `special-fun`, `arb-sys`, `flint-sys` | FFI wrappers around C libraries — not pure Rust; several are GPL/LGPL. |

---

## The NIST DLMF itself

The DLMF is **cited, never reproduced**. It is copyright NIST — its
authors assigned their rights to the agency, so it is not public domain
despite being a government product. NIST permits limited copying for
research and teaching and prohibits reproduction for commercial
purposes.

This project therefore:

* cites equations by number and permalink, e.g. `DLMF 10.47.3`
  <https://dlmf.nist.gov/10.47.E3>;
* implements the mathematics independently;
* reproduces **no** DLMF prose or equation displays;
* takes any formula shown in documentation from **Abramowitz & Stegun
  (1964)**, a US Government work in the public domain, or writes it in
  its own notation.

Reference work: *NIST Digital Library of Mathematical Functions*,
<https://dlmf.nist.gov/>, Release 1.2.7 of 2026-06-15, F. W. J. Olver,
A. B. Olde Daalhuis, D. W. Lozier, B. I. Schneider, R. F. Boisvert,
C. W. Clark, B. R. Miller, B. V. Saunders, H. S. Cohl and M. A. McClain,
eds. NIST publishes **no** reference implementation; its "Software" page
is a bibliography of third-party packages.
