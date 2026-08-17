//! Special functions for rustSimulate, cited to the NIST DLMF.
//!
//! # What this crate is
//!
//! Two layers behind one uniform API:
//!
//! 1. **Vendored** — the classical chapters come from a vendored copy of
//!    `spec_math`, a Rust translation of Cephes. See `THIRD_PARTY.md` at
//!    the repository root for provenance and licence.
//! 2. **Native** — the families `spec_math` lacks are implemented here:
//!    spherical Bessel functions, Legendre and associated Legendre
//!    functions (hence spherical harmonics), the classical orthogonal
//!    polynomials, and Wigner 3j/6j symbols.
//!
//! # Citations
//!
//! Every function names the DLMF equation it implements, in both the
//! human form and the permalink form, e.g. `DLMF 10.47.3`
//! <https://dlmf.nist.gov/10.47.E3>. The DLMF is copyright NIST; this
//! crate **cites** it and implements the mathematics independently — no
//! DLMF text is reproduced. Formulas shown in documentation are taken
//! from Abramowitz & Stegun (1964), a public-domain US Government work,
//! or written in our own notation.
//!
//! Cite the reference work as: *NIST Digital Library of Mathematical
//! Functions*, <https://dlmf.nist.gov/>, Release 1.2.7 of 2026-06-15.
//!
//! # Error policy
//!
//! Public functions return `Result<f64, String>` with an actionable
//! message naming the offending argument. Nothing panics, and no
//! function returns a silent `NaN` where the input was invalid — a
//! domain error is reported as an error.
//!
//! # Coverage
//!
//! This crate covers roughly 11–13 of the DLMF's 33 function chapters,
//! partially. It is **not** a complete DLMF implementation — no such
//! thing exists in any language. `SPECIAL_FUNCTIONS_PROVENANCE.md`
//! carries the honest per-chapter coverage matrix.
#![forbid(unsafe_code)]
#![deny(warnings)]
#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]

pub mod bessel;
pub mod bessel_complex;
pub mod hankel;
pub mod debye;
pub mod gamma_complex;
pub mod bessel_cnu;
pub mod bessel_cnu_large;
pub mod airy_complex;
pub mod airy_uniform;
pub mod bessel_scaled;
pub mod wigner;
pub mod complex;
pub mod tridiag;
pub mod eigen;
pub mod lanczos;
pub mod legendre;
pub mod orthopoly;
pub mod quadrature;
pub mod sph_bessel;

/// Re-exports of the vendored Cephes translation, so callers have one
/// import path. These are *not* our implementations — see
/// `THIRD_PARTY.md`.
pub mod cephes {
    pub use spec_math::cephes64;
}

/// Relative-difference helper used throughout the test suites.
///
/// Returns `|a - b| / max(|b|, tiny)`, so it degrades gracefully to an
/// absolute comparison when the reference value is itself near zero.
pub fn rel_err(a: f64, b: f64) -> f64 {
    let d = (a - b).abs();
    let s = b.abs();
    if s > 1.0e-300 {
        d / s
    } else {
        d
    }
}
