//! Bessel functions of **complex order**.
//!
//! Everything before this stage took a real order. That was recorded as
//! the last remaining gap in chapter 10's coverage from Stage 12 onward,
//! and it is the one entry in the list that no expansion could close,
//! because the obstacle was not an algorithm — it was that
//! `1/Gamma(nu + k + 1)` had no meaning here for complex `nu`. With
//! [`crate::gamma_complex`] in place the obstacle is gone and the
//! ascending series works unchanged:
//!
//! ```text
//!   J_nu(z) = (z/2)^nu sum_k (-1)^k (z^2/4)^k / (k! Gamma(nu+k+1))
//!   I_nu(z) = (z/2)^nu sum_k        (z^2/4)^k / (k! Gamma(nu+k+1))
//!   Y_nu(z) = [J_nu(z) cos(nu pi) - J_{-nu}(z)] / sin(nu pi)
//!   K_nu(z) = (pi/2) [I_{-nu}(z) - I_nu(z)] / sin(nu pi)
//! ```
//!
//! Two things change with a complex order, and both are consequences of
//! the same fact — that `nu` now appears in an exponent:
//!
//! * `(z/2)^nu` is `exp(nu ln(z/2))`, and with `nu` complex its
//!   **modulus** depends on `arg z`, not just on `|z|`. A branch choice
//!   is no longer a phase convention. The cut is `ln`'s, along the
//!   negative real axis, and it is inherited by every function here.
//! * `sin(nu pi)` in the reflections grows like `exp(pi |Im nu|)`, so
//!   the reflections get **better** conditioned as `nu` leaves the real
//!   axis, not worse. The 0/0 that forces a near-integer handover for
//!   real order only threatens when `Im nu` is also small, and the
//!   handover tests exactly that.
//!
//! # Accuracy
//!
//! The series is the same one [`crate::bessel_complex`] uses, so it
//! carries the same law — the working precision is spent on the ratio
//! between the largest term, `exp(|z|)`, and the answer. What is new is
//! that the answer's size now depends on `Im nu` too: `|J_nu(z)|` picks
//! up a factor `exp(-Im nu * arg z)`, so the loss is
//!
//! ```text
//!   relative error ~ 1e-16 exp(|z| - |Im z| + Im nu * arg z)
//! ```
//!
//! measured and pinned by `the_complex_order_loss_law_holds`. On the
//! positive real axis `arg z = 0` and complex order costs nothing at
//! all; well off it, a large `|Im nu|` is expensive in one direction and
//! free in the other.
//!
//! # Beyond the series
//!
//! [`crate::bessel_cnu_large`] carries the `1/z` asymptotics and the
//! DLMF 10.41 uniform expansions to complex order, and they are offered
//! here as further candidates, chosen by comparing error estimates as
//! everywhere else in this crate. Between them the reach extends from
//! `|z| ~ 25` to `|z| ~ 600` and beyond, and to orders the series
//! cannot represent.
//!
//! [`crate::airy_uniform::jy_airy_c`] adds the **Airy-type** expansion of
//! DLMF 10.20 at complex order, which covers the turning point
//! `|z| ~ |nu|` — the region neither of the others reaches. It needed
//! complex Airy, which is why it arrived two stages after the rest.
//!
//! [`crate::debye::jy_debye_c`] covers the **Debye region** on both
//! sides of the turning point — `|z|` a few times `|nu|`, and `|z|`
//! below it — at complex order, so the band that had no method at all
//! now has one.
//!
//! That sliver — `4 <~ |nu| <~ 8` with `|z|` a few times larger — was
//! **closed in Stage 24, but not by adding a method**. Measurement
//! found that the Debye route was refusing it for the wrong reason: it
//! carried an *order* guard, `|nu| >= 8`, while the thing that actually
//! governs its accuracy is `|z|/|nu|` together with `arg(z/nu)`. With
//! the guard restated in those variables, small orders are admitted
//! wherever the argument is large enough, and the sliver goes from
//! 94% served to **97%**.
//!
//! The same measurement found three **branch defects** in that route
//! which had been returning confident, plausible, wrong values — see
//! [`crate::debye::jy_debye_c`]. They fired only where `z/nu` is off
//! the real axis, including at *real* order with complex argument, so
//! every check in the crate had missed them.
//!
//! Stage 24 left a band uncovered — `1 < |z/nu| < 8` off the real axis,
//! and `1 < |z/nu| < 2` past order 8 — having established that the
//! points there had previously been *accepted* with estimates up to
//! 1e14 too small.
//!
//! **Stage 2D closed most of it.** The Airy-type expansion of DLMF
//! 10.20 is uniformly valid across exactly that band; what restricted
//! it to a neighbourhood of the turning point was a branch, not a
//! limit. With [`crate::airy_uniform::zeta_c`] supplying it, the band
//! goes from 41 % to **97.5 %** served inside `|arg(z/nu)| <= 0.8`.
//! Beyond that sector the branch anchor is not measured and the route
//! refuses, so what remains open is the same band at large argument.

use crate::bessel_complex::{bessel_i_nu, bessel_j_nu, bessel_k_nu, bessel_y_nu};
use crate::complex::Complex64 as C;
use crate::gamma_complex::rgamma_c;

/// How close to real `nu` must be before the real-order routines take
/// over. They are better there: they use `rgamma` from the vendored
/// Cephes rather than a shifted Stirling, and they handle negative
/// whole orders, where the series' recurrence for `1/Gamma` cannot
/// restart from a zero.
const REAL_TOL: f64 = 1e-13;

fn is_real(nu: C) -> bool {
    nu.im.abs() <= REAL_TOL * nu.re.abs().max(1.0)
}

/// How wrong an answer may be estimated to be before the routine
/// refuses instead. Six digits, matching [`crate::bessel_scaled`].
const TOL: f64 = 1e-6;

/// The ascending series' estimated relative error.
///
/// The law from the module note, one term per function. The extra
/// `Im nu * arg z` is what complex order adds, and it is why complex
/// order is free on the positive real axis.
fn series_error(loss: f64) -> f64 {
    if loss > 700.0 {
        f64::INFINITY
    } else {
        // 1e-14, not 1e-16. The law is a MODEL of the cancellation, and
        // measured against the J-Y Wronskian over a 16798-point sweep it
        // runs about two decades optimistic in the corner where
        // |Im nu| is large — at nu = 0.5 + 5i, |z| = 30 it claimed
        // 6.6e-7 and the Wronskian said 4.7e-5. Two decades of slack
        // makes the gate at TOL mean what it says.
        (1e-14 * loss.exp()).max(1e-16)
    }
}

/// The series route as a candidate: the value, and what it is worth.
fn series_candidate(v: Result<C, String>, loss: f64) -> Cand {
    match v {
        Ok(x) if x.is_finite() => Some((x, series_error(loss))),
        _ => None,
    }
}

type Cand = Option<(C, f64)>;

fn better(a: Cand, b: Cand) -> Cand {
    match (a, b) {
        (Some(x), Some(y)) => Some(if x.1 <= y.1 { x } else { y }),
        (x, None) => x,
        (None, y) => y,
    }
}

/// Turn the best candidate into a result, or say why there is none.
fn accept(c: Cand, what: &str, nu: C, z: C) -> Result<C, String> {
    match c {
        Some((v, e)) if e <= TOL => Ok(v),
        Some((_, e)) => Err(format!(
            "{what}: no method is accurate at nu = {nu:?}, z = {z:?} — the best \
             available estimates {e:.1e}. Four methods were offered and none \
             could speak for this point: the ascending series has cancelled away \
             its digits at this |z|; the 1/z expansions need |4 nu^2| small \
             compared with |z|; the Airy-type expansion of DLMF 10.20 reaches \
             only |1 - z/nu| <= 0.25 around the turning point; and the Debye \
             expansion is trusted only for |z| >= 2|nu| near the real axis, or \
             |z| >= 8|nu| within |arg(z/nu)| <= 1.2. What is left uncovered is \
             mainly the band 1 < |z/nu| < 8 off the real axis, and 1 < |z/nu| < 2 \
             at large order — measured gaps, not oversights."
        )),
        None => Err(format!(
            "{what}: no method produced a finite value at nu = {nu:?}, z = {z:?}. \
             The value may simply be outside f64 range."
        )),
    }
}

/// `J` and `Y` from the Debye expansions at complex order — the band
/// `|z|` a few times `|nu|`, on either side of the turning point.
fn debye_candidates(nu: C, z: C) -> (Cand, Cand) {
    let (j, y) = crate::debye::jy_debye_c(nu, z);
    let f = |u: crate::debye::Uniform| u.value.is_finite().then_some((u.value, u.err.max(1e-16)));
    (j.and_then(f), y.and_then(f))
}

/// `J` and `Y` from the uniform Airy-type expansion of DLMF 10.20 at
/// complex order, near the turning point.
///
/// This is the region neither the ascending series nor the `1/z`
/// expansions reach — `|z|` comparable to `|nu|`, both complex — and
/// which Stage 18 recorded as needing complex Airy. It has that now.
fn airy_candidates(nu: C, z: C) -> (Cand, Cand) {
    match crate::airy_uniform::jy_airy_c(nu, z) {
        Some((j, y)) => (
            j.value.is_finite().then_some((j.value, j.err.max(1e-16))),
            y.value.is_finite().then_some((y.value, y.err.max(1e-16))),
        ),
        None => (None, None),
    }
}

/// `|Im nu * arg z|`, the term complex order adds to every loss law.
fn order_term(nu: C, z: C) -> f64 {
    (nu.im * z.arg()).abs()
}

/// Extra loss carried by the **integer-order** `Y` route near the
/// imaginary axis.
///
/// `bessel_y_c` builds `Y_n` by upward recurrence in `n` from `Y_0` and
/// `Y_1`. That direction is stable for real argument, where `Y` is the
/// dominant solution in order — but at nearly-imaginary argument `Y_n`
/// is a combination whose recessive part the recurrence amplifies, and
/// the accuracy law recorded in Stage 13 does not describe it.
///
/// This stage found it by accident and the Wronskian adjudicated: at
/// `nu = 2, z = 29.4 e^{1.6i}` the `1/z` expansion closes the J-Y
/// Wronskian to **7e-26** while the integer series closes it to
/// **4.5e-6**. Without this term the selector believed the series'
/// claim of 1e-16 and returned the worse number.
///
/// The bound is `exp(|Im z|)` relative — empirical, and deliberately
/// generous, because it is guarding a defect in another module rather
/// than modelling one here. Fixing `bessel_y_c` itself is a separate
/// job; this makes the selector stop trusting it.
fn integer_y_recurrence_loss(nu: C, z: C) -> f64 {
    let near_whole = nu.im == 0.0 && (nu.re - nu.re.round()).abs() < 1e-9;
    if near_whole {
        z.im.abs()
    } else {
        0.0
    }
}

fn check(nu: C, z: C, what: &str) -> Result<(), String> {
    if !nu.is_finite() {
        return Err(format!("{what}: the order must be finite, got {nu:?}"));
    }
    if !z.is_finite() {
        return Err(format!("{what}: z must be finite, got {z:?}"));
    }
    Ok(())
}

/// The shared ascending series at complex order. `alternating` selects
/// `J` (true) or `I`.
///
/// `1/Gamma(nu+k+1)` is advanced by its own recurrence,
/// `1/Gamma(w+1) = (1/Gamma(w))/w`, so the complex gamma is evaluated
/// **once** rather than per term. For genuinely complex `nu` no
/// `nu+k+1` is ever a non-positive integer, so the recurrence never has
/// to restart from a zero — which is exactly why a real order is handed
/// off instead of being folded in here.
fn cnu_series(nu: C, z: C, alternating: bool) -> Result<C, String> {
    if z.abs() == 0.0 {
        return if nu.im == 0.0 && nu.re == 0.0 {
            Ok(C::ONE)
        } else if nu.im == 0.0 && nu.re > 0.0 {
            Ok(C::ZERO)
        } else {
            Err("bessel_cnu: z = 0 is a branch point unless nu is a non-negative real".to_string())
        };
    }
    let half = z * 0.5;
    let q = half * half;
    let step = if alternating { q * -1.0 } else { q };

    let mut rg = rgamma_c(nu + C::ONE)?; // 1/Gamma(nu+1)
    let mut sum = rg;
    let mut term_pow = C::ONE;
    let mut fact_k = 1.0_f64;
    for k in 0..400 {
        let w = nu + C::real(k as f64 + 1.0); // 1/Gamma(nu+k+2) = rg/w
        if w.abs() == 0.0 {
            return Err(format!("bessel_cnu: 1/Gamma recurrence hits a pole at nu = {nu:?}"));
        }
        rg = rg * w.inv();
        term_pow = term_pow * step;
        fact_k *= (k + 1) as f64;
        let add = term_pow * rg * (1.0 / fact_k);
        sum = sum + add;
        if k > 6 && add.abs() <= 1e-18 * sum.abs().max(1e-300) {
            break;
        }
    }
    let out = half.powc(nu) * sum;
    if !out.is_finite() {
        return Err(format!("bessel_cnu: the series overflowed at nu = {nu:?}, z = {z:?}"));
    }
    Ok(out)
}

/// `J_nu(z)` for **complex order** and complex argument.
///
/// A real order (within `1e-13`) is handed to
/// [`crate::bessel_complex::bessel_j_nu`], which is better there.
///
/// # Errors
/// A non-finite order or argument, `z = 0` at an order that is not a
/// non-negative real, or overflow.
///
/// # Examples
/// ```
/// use special_functions::bessel_cnu::bessel_j_cnu;
/// use special_functions::complex::Complex64 as C;
/// // For real nu and real z the answer is real, and complex order
/// // reproduces it.
/// let v = bessel_j_cnu(C::real(0.5), C::real(2.0)).unwrap();
/// let want = (2.0 / (std::f64::consts::PI * 2.0)).sqrt() * 2.0_f64.sin();
/// assert!((v.re - want).abs() < 1e-13);
/// ```
pub fn bessel_j_cnu(nu: C, z: C) -> Result<C, String> {
    check(nu, z, "bessel_j_cnu")?;
    if z.abs() == 0.0 {
        return if is_real(nu) { bessel_j_nu(nu.re, z) } else { cnu_series(nu, z, true) };
    }
    let ser = if is_real(nu) { bessel_j_nu(nu.re, z) } else { cnu_series(nu, z, true) };
    let s = series_candidate(ser, z.abs() - z.im.abs() + order_term(nu, z));
    let a = crate::bessel_cnu_large::j_asym(nu, z);
    let u = airy_candidates(nu, z).0;
    let d = debye_candidates(nu, z).0;
    accept(better(better(better(s, a), u), d), "bessel_j_cnu", nu, z)
}

/// `I_nu(z)` for complex order.
///
/// # Errors
/// As [`bessel_j_cnu`].
pub fn bessel_i_cnu(nu: C, z: C) -> Result<C, String> {
    check(nu, z, "bessel_i_cnu")?;
    if z.abs() == 0.0 {
        return if is_real(nu) { bessel_i_nu(nu.re, z) } else { cnu_series(nu, z, false) };
    }
    let ser = if is_real(nu) { bessel_i_nu(nu.re, z) } else { cnu_series(nu, z, false) };
    let s = series_candidate(ser, z.abs() - z.re + order_term(nu, z));
    let a = crate::bessel_cnu_large::i_asym(nu, z);
    let u = crate::bessel_cnu_large::ik_uniform_unscaled(nu, z).0;
    accept(better(better(s, a), u), "bessel_i_cnu", nu, z)
}

/// `Y_nu(z)` for complex order, by the reflection
/// `Y_nu = [J_nu cos(nu pi) - J_{-nu}] / sin(nu pi)` (DLMF 10.2.3).
///
/// Unlike the real-order case this needs no special handling near whole
/// numbers unless `Im nu` is *also* small: `|sin(nu pi)|` grows like
/// `exp(pi |Im nu|)`, so the division is well conditioned as soon as the
/// order leaves the real axis.
///
/// # Errors
/// As [`bessel_j_cnu`]; also `z = 0`, where `Y` is singular.
pub fn bessel_y_cnu(nu: C, z: C) -> Result<C, String> {
    check(nu, z, "bessel_y_cnu")?;
    if is_real(nu) && z.abs() != 0.0 {
        let s = series_candidate(
            bessel_y_nu(nu.re, z),
            (z.abs() - z.im.abs() + order_term(nu, z))
                .max(integer_y_recurrence_loss(nu, z)),
        );
        let a = crate::bessel_cnu_large::y_asym(nu, z);
        let u = airy_candidates(nu, z).1;
        let d = debye_candidates(nu, z).1;
        return accept(better(better(better(s, a), u), d), "bessel_y_cnu", nu, z);
    }
    if is_real(nu) {
        return bessel_y_nu(nu.re, z);
    }
    if z.abs() == 0.0 {
        return Err("bessel_y_cnu: Y is singular at z = 0".to_string());
    }
    let sn = (nu * std::f64::consts::PI).sin();
    let cs = (nu * std::f64::consts::PI).cos();
    let ser = cnu_series(nu, z, true)
        .and_then(|jp| cnu_series(nu * -1.0, z, true).map(|jm| (jp * cs - jm) * sn.inv()));
    let s = series_candidate(ser, z.abs() - z.im.abs() + order_term(nu, z));
    let a = crate::bessel_cnu_large::y_asym(nu, z);
    let u = airy_candidates(nu, z).1;
    let d = debye_candidates(nu, z).1;
    accept(better(better(better(s, a), u), d), "bessel_y_cnu", nu, z)
}

/// `K_nu(z)` for complex order, by
/// `K_nu = (pi/2)[I_{-nu} - I_nu]/sin(nu pi)` (DLMF 10.27.4).
///
/// `K_{-nu} = K_nu` holds for complex order too, and the test suite
/// checks it rather than exploiting it.
///
/// # Errors
/// As [`bessel_y_cnu`].
pub fn bessel_k_cnu(nu: C, z: C) -> Result<C, String> {
    check(nu, z, "bessel_k_cnu")?;
    if is_real(nu) && z.abs() != 0.0 {
        let s = series_candidate(bessel_k_nu(nu.re, z), z.abs() + z.re + order_term(nu, z));
        let a = crate::bessel_cnu_large::k_asym(nu, z);
        let u = crate::bessel_cnu_large::ik_uniform_unscaled(nu, z).1;
        return accept(better(better(s, a), u), "bessel_k_cnu", nu, z);
    }
    if is_real(nu) {
        return bessel_k_nu(nu.re, z);
    }
    if z.abs() == 0.0 {
        return Err("bessel_k_cnu: K is singular at z = 0".to_string());
    }
    let sn = (nu * std::f64::consts::PI).sin();
    let ser = cnu_series(nu, z, false).and_then(|ip| {
        cnu_series(nu * -1.0, z, false)
            .map(|im| (im - ip) * sn.inv() * (std::f64::consts::PI / 2.0))
    });
    let s = series_candidate(ser, z.abs() + z.re + order_term(nu, z));
    let a = crate::bessel_cnu_large::k_asym(nu, z);
    let u = crate::bessel_cnu_large::ik_uniform_unscaled(nu, z).1;
    accept(better(better(s, a), u), "bessel_k_cnu", nu, z)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: C, b: C, tol: f64) -> bool {
        (a - b).abs() <= tol * b.abs().max(1e-300)
    }

    /// A complex order whose imaginary part is tiny but not zero must
    /// reproduce the real-order routines. This is the join between the
    /// new code and everything before it, and it exercises the series
    /// itself — an order of `1e-11 i` is far outside the handover
    /// tolerance, so the complex path really runs.
    #[test]
    fn a_nearly_real_order_reproduces_the_real_order_routines() {
        for &nu in &[0.3_f64, 1.7, 2.5, 5.2, -0.8] {
            for &(re, im) in &[(1.0, 0.0), (4.0, 0.0), (2.0, 1.5), (0.7, -2.0), (-3.0, 1.0)] {
                let z = C::new(re, im);
                let c = C::new(nu, 1e-11);
                assert!(
                    close(bessel_j_cnu(c, z).unwrap(), bessel_j_nu(nu, z).unwrap(), 1e-9),
                    "J at nu={nu}, z={z:?}"
                );
                assert!(
                    close(bessel_i_cnu(c, z).unwrap(), bessel_i_nu(nu, z).unwrap(), 1e-9),
                    "I at nu={nu}, z={z:?}"
                );
                assert!(
                    close(bessel_y_cnu(c, z).unwrap(), bessel_y_nu(nu, z).unwrap(), 1e-8),
                    "Y at nu={nu}, z={z:?}"
                );
                assert!(
                    close(bessel_k_cnu(c, z).unwrap(), bessel_k_nu(nu, z).unwrap(), 1e-8),
                    "K at nu={nu}, z={z:?}"
                );
            }
        }
    }

    /// **The ridge, measured.** The selector is accurate exactly where
    /// the raw reflection route is not.
    ///
    /// `SPECIAL_FUNCTIONS_PROVENANCE.md` carried this as "a ridge near
    /// `z/nu ~ 1.3` where `Y` reaches about 1e-7 at moderate order".
    /// Stage 2I measured it and the entry understated it badly: on
    /// `bessel_complex::bessel_y_nu` the error is **3.09e4** relative at
    /// `nu = 36.8, z/nu = 1.48` and **2.18** at `z/nu = 1.30`. The 1e-7
    /// figure was the value at `nu = 20.5`, which is where the sweep
    /// that recorded it happened to stop.
    ///
    /// The cause is inside the ingredient, not the combination:
    /// `Y_nu = [J_nu cos(nu pi) - J_{-nu}]/sin(nu pi)`, and `J_{-36.8}`
    /// at `z = 54` comes from an ascending series whose terms reach
    /// `exp(54) ~ 3e23` to produce a result of order 0.1.
    ///
    /// The J-Y Wronskian adjudicated it against Cephes: our residual
    /// 3.7e-3, Cephes 2.2e-23. Cephes is right and the raw route is
    /// wrong — the reverse of Stages 15 and 19, where Cephes was the
    /// looser party. Which one is right has to be measured each time.
    ///
    /// This pins the part that is verified: the selector compares
    /// estimates across routes and lands on one that works.
    #[test]
    fn the_selector_is_accurate_where_the_raw_reflection_is_not() {
        for &(nu, frac) in &[(36.8_f64, 1.48_f64), (36.8, 1.30), (20.5, 1.48), (12.3, 1.35)] {
            let z = nu * frac;
            let got = bessel_y_cnu(C::real(nu), C::real(z))
                .unwrap_or_else(|e| panic!("nu = {nu}, z/nu = {frac}: {e}"));
            let want = spec_math::cephes64::yv(nu, z);
            let rel = (got.re - want).abs() / want.abs();
            assert!(
                rel < 1e-9,
                "nu = {nu}, z/nu = {frac}: selector gave {} against Cephes {want} ({rel:.2e})",
                got.re
            );
            // ...and the raw route must not quietly return a wrong
            // number. Stage 2J gave it a measured guard, so at the two
            // points 2I recorded it now REFUSES; the test tracks that
            // rather than the old behaviour.
            let raw = crate::bessel_complex::bessel_y_nu(nu, C::real(z));
            if nu > 30.0 {
                let e = raw.expect_err(
                    "the raw reflection route must refuse where it cannot deliver",
                );
                assert!(e.contains("precision"), "and say why: {e}");
                assert!(e.contains("bessel_y_cnu"), "and where to go instead: {e}");
            } else {
                let raw_rel = (raw.unwrap().re - want).abs() / want.abs();
                assert!(raw_rel < 1e-6, "below the guard it should still be usable");
            }
        }
    }

    /// The J-Y Wronskian `J_{nu+1} Y_nu - J_nu Y_{nu+1} = 2/(pi z)`
    /// (DLMF 10.5.2) holds for **every** order, complex included, and
    /// its right-hand side does not involve the order at all. That makes
    /// it the sharpest available check on a genuinely complex order:
    /// nothing about it can be satisfied by a consistently wrong pair.
    #[test]
    fn the_j_y_wronskian_holds_at_complex_order() {
        for &(a, b) in &[(0.3, 0.5), (1.2, 1.0), (2.0, -0.7), (0.0, 2.0), (-1.5, 1.3)] {
            for &(re, im) in &[(1.0, 0.0), (3.0, 0.0), (2.0, 1.0), (1.5, -1.5)] {
                let (nu, z) = (C::new(a, b), C::new(re, im));
                let w = bessel_j_cnu(nu + C::ONE, z).unwrap() * bessel_y_cnu(nu, z).unwrap()
                    - bessel_j_cnu(nu, z).unwrap() * bessel_y_cnu(nu + C::ONE, z).unwrap();
                let want = z.inv() * (2.0 / std::f64::consts::PI);
                assert!(close(w, want, 1e-10), "nu={nu:?} z={z:?}: {w:?} vs {want:?}");
            }
        }
    }

    /// `I_nu K_{nu+1} + I_{nu+1} K_nu = 1/z` (DLMF 10.28.2), likewise
    /// order-independent on the right.
    #[test]
    fn the_i_k_wronskian_holds_at_complex_order() {
        for &(a, b) in &[(0.3, 0.5), (1.2, 1.0), (2.0, -0.7), (-0.4, 1.6)] {
            for &(re, im) in &[(1.0, 0.0), (3.0, 0.0), (2.0, 1.0), (1.5, -1.5)] {
                let (nu, z) = (C::new(a, b), C::new(re, im));
                let w = bessel_i_cnu(nu, z).unwrap() * bessel_k_cnu(nu + C::ONE, z).unwrap()
                    + bessel_i_cnu(nu + C::ONE, z).unwrap() * bessel_k_cnu(nu, z).unwrap();
                assert!(close(w, z.inv(), 1e-10), "nu={nu:?} z={z:?}");
            }
        }
    }

    /// The three-term recurrence in order, `C_{nu-1} + C_{nu+1} =
    /// (2 nu/z) C_nu` (DLMF 10.6.1), with every value computed by its
    /// own independent series evaluation.
    #[test]
    fn the_order_recurrence_holds_at_complex_order() {
        for &(a, b) in &[(0.4, 0.8), (1.6, -1.2), (3.0, 2.0)] {
            for &(re, im) in &[(1.0, 0.0), (4.0, 0.0), (2.0, 1.5)] {
                let (nu, z) = (C::new(a, b), C::new(re, im));
                let f = z.inv() * nu * 2.0;
                let lhs = bessel_j_cnu(nu - C::ONE, z).unwrap()
                    + bessel_j_cnu(nu + C::ONE, z).unwrap();
                assert!(close(lhs, bessel_j_cnu(nu, z).unwrap() * f, 1e-11), "J rec");
                let lhs = bessel_y_cnu(nu - C::ONE, z).unwrap()
                    + bessel_y_cnu(nu + C::ONE, z).unwrap();
                assert!(close(lhs, bessel_y_cnu(nu, z).unwrap() * f, 1e-9), "Y rec");
                // I's recurrence carries a minus sign (DLMF 10.29.1).
                let lhs = bessel_i_cnu(nu - C::ONE, z).unwrap()
                    - bessel_i_cnu(nu + C::ONE, z).unwrap();
                assert!(close(lhs, bessel_i_cnu(nu, z).unwrap() * f, 1e-11), "I rec");
            }
        }
    }

    /// Conjugation: the series has real coefficients in the sense that
    /// `J_{conj nu}(conj z) = conj(J_nu(z))`. This catches a wrong
    /// branch in `powc` or in the complex gamma, which a Wronskian
    /// cannot — a consistently conjugated pair still satisfies it.
    #[test]
    fn conjugating_both_arguments_conjugates_the_result() {
        for &(a, b) in &[(0.7, 1.1), (2.3, -0.9), (-1.2, 2.5)] {
            for &(re, im) in &[(2.0, 1.0), (1.0, -2.0), (3.5, 0.5)] {
                let (nu, z) = (C::new(a, b), C::new(re, im));
                for (name, f) in [
                    ("J", bessel_j_cnu as fn(C, C) -> Result<C, String>),
                    ("I", bessel_i_cnu),
                    ("Y", bessel_y_cnu),
                    ("K", bessel_k_cnu),
                ] {
                    let a = f(nu.conj(), z.conj()).unwrap();
                    let b = f(nu, z).unwrap().conj();
                    assert!(close(a, b, 1e-11), "{name} conjugation at nu={nu:?} z={z:?}");
                }
            }
        }
    }

    /// `K_{-nu} = K_nu` for complex order too. The two sides run
    /// different arithmetic — the reflection swaps which `I` is which —
    /// so agreement is a real check.
    #[test]
    fn k_is_even_in_the_order() {
        for &(a, b) in &[(0.6, 0.9), (1.8, -1.4), (0.0, 2.2)] {
            for &(re, im) in &[(1.5, 0.0), (2.0, 1.0)] {
                let (nu, z) = (C::new(a, b), C::new(re, im));
                assert!(
                    close(bessel_k_cnu(nu * -1.0, z).unwrap(), bessel_k_cnu(nu, z).unwrap(), 1e-11),
                    "K_-nu != K_nu at nu={nu:?}"
                );
            }
        }
    }

    /// The reflections are BETTER conditioned off the real axis, not
    /// worse: `|sin(nu pi)|` grows like `exp(pi |Im nu|)`. So an order
    /// sitting exactly on a whole number needs no special handling at
    /// all provided its imaginary part is not also small — the case
    /// that would be 0/0 for a real order.
    #[test]
    fn a_whole_real_part_is_not_a_special_case_off_the_real_axis() {
        for &n in &[0.0_f64, 1.0, 3.0] {
            for &b in &[0.5_f64, 1.0, 3.0] {
                let (nu, z) = (C::new(n, b), C::new(2.0, 0.5));
                let y = bessel_y_cnu(nu, z).unwrap();
                assert!(y.is_finite(), "Y at nu={nu:?} should be finite");
                // Checked by the Wronskian, which needs no reference.
                let w = bessel_j_cnu(nu + C::ONE, z).unwrap() * y
                    - bessel_j_cnu(nu, z).unwrap()
                        * bessel_y_cnu(nu + C::ONE, z).unwrap();
                let want = z.inv() * (2.0 / std::f64::consts::PI);
                assert!(close(w, want, 1e-10), "Wronskian at nu={nu:?}");
            }
        }
        // sin(nu pi) really does grow: this is the reason, stated as a
        // measurement rather than a remark.
        let s1 = (C::new(1.0, 0.5) * std::f64::consts::PI).sin().abs();
        let s2 = (C::new(1.0, 3.0) * std::f64::consts::PI).sin().abs();
        assert!(s1 > 2.0 && s2 > 6000.0, "|sin(nu pi)| = {s1}, {s2}");
    }

    /// The loss law the module documents:
    /// `1e-16 exp(|z| - |Im z| + Im nu * arg z)`. Complex order is free
    /// on the positive real axis and costs `Im nu * arg z` elsewhere.
    /// **On the instrument.** This used to divide the J-Y Wronskian
    /// residual by its own largest term, "so the metric's cancellation
    /// is divided out". Stage 24 measured what that actually leaves: at
    /// `nu = 5 + 2i, z = 200 + 80i` the two Hankel functions differ in
    /// size by 1e67, `J` and `Y` are then the same function to within
    /// 4e-24, and the scaled residual came out **8.2e-24** — not
    /// accuracy, just `|H1/H2|`. The unscaled form is no better: it
    /// would demand an accuracy of `|H1/H2|` relative, which no correct
    /// implementation can deliver. The J-Y Wronskian simply **cannot
    /// resolve below `|H1/H2|`**, so the tolerance is floored there.
    #[test]
    fn the_complex_order_loss_law_holds() {
        for &b in &[0.5_f64, 2.0, 5.0] {
            for &arg in &[0.0_f64, 0.8, -0.8, 2.0] {
                for &r in &[1.0_f64, 4.0, 10.0] {
                    let (nu, z) = (C::new(1.3, b), C::from_polar(r, arg));
                    let l = z.abs() - z.im.abs() + (b * arg).abs();
                    let bound = (1e-14 * l.exp()).max(1e-13);
                    if bound > 1e-3 {
                        continue;
                    }
                    let j0 = bessel_j_cnu(nu, z).unwrap();
                    let j1 = bessel_j_cnu(nu + C::ONE, z).unwrap();
                    let y0 = bessel_y_cnu(nu, z).unwrap();
                    let y1 = bessel_y_cnu(nu + C::ONE, z).unwrap();
                    let w = j1 * y0 - j0 * y1;
                    let want = z.inv() * (2.0 / std::f64::consts::PI);
                    let scale = (j1 * y0).abs() + (j0 * y1).abs();
                    // The floor the instrument cannot see past.
                    let floor = crate::bessel_cnu_large::hankel_ratio(nu, z)
                        .map_or(0.0, |r| 1.0 / r);
                    assert!(
                        (w - want).abs() / scale <= bound.max(10.0 * floor),
                        "nu={nu:?} z={z:?}: residual {:.2e} exceeds {bound:.1e}",
                        (w - want).abs() / scale
                    );
                }
            }
        }
    }

    /// `K_{i y}(x)` is **real** for real `y` and real positive `x` —
    /// the Macdonald function of imaginary order, which is why it turns
    /// up as an eigenfunction in problems on the half line. Nothing in
    /// the implementation arranges this: `K` is built from two `I`s at
    /// `+nu` and `-nu` divided by `sin(nu pi)`, all three of them
    /// thoroughly complex. It comes out real because the mathematics
    /// says so, which makes it a strong end-to-end check.
    #[test]
    fn k_of_imaginary_order_is_real() {
        for &y in &[0.3_f64, 1.0, 2.5, 6.0] {
            for &x in &[0.5_f64, 1.0, 3.0, 8.0] {
                let v = bessel_k_cnu(C::new(0.0, y), C::real(x)).unwrap();
                assert!(
                    v.im.abs() <= 1e-12 * v.re.abs().max(1e-8),
                    "K_({y}i)({x}) should be real, got {v:?}"
                );
                // ... and equal to K at the conjugate order, which is
                // the other half of the same statement.
                let c = bessel_k_cnu(C::new(0.0, -y), C::real(x)).unwrap();
                assert!(close(c, v, 1e-12), "K_(-{y}i) != K_({y}i)");
            }
        }
    }

    /// The whole thing, judged by the J-Y Wronskian across a grid of
    /// order and argument that no method here covers alone.
    ///
    /// This is the test that matters, because for complex order there is
    /// no reference implementation to compare against — and it judges
    /// the value the routine actually **chose**, not the one a
    /// particular route would have produced. Both of the estimate
    /// defects this stage found showed up here first: the selector
    /// preferring a series whose claim was two decades optimistic, and
    /// preferring an integer-order `Y` whose upward recurrence is
    /// unstable near the imaginary axis.
    #[test]
    fn the_chosen_values_satisfy_the_wronskian_across_the_plane() {
        let mut judged = 0;
        let mut refused = 0;
        let mut blind = 0;
        let mut worst = 0.0_f64;
        for &(a, b) in &[
            (1.3_f64, 0.0_f64),
            (2.0, 0.0),
            (0.5, 0.0),
            (1.3, 2.0),
            (0.5, 5.0),
            (-0.7, 1.0),
            (3.0, -2.0),
            (0.0, 3.0),
        ] {
            for i in 0..14 {
                let r = 6.0 + i as f64 * 4.0;
                for k in 0..21 {
                    let arg = -3.0 + k as f64 * 0.3;
                    let (nu, z) = (C::new(a, b), C::from_polar(r, arg));
                    let (Ok(j0), Ok(j1), Ok(y0), Ok(y1)) = (
                        bessel_j_cnu(nu, z),
                        bessel_j_cnu(nu + C::ONE, z),
                        bessel_y_cnu(nu, z),
                        bessel_y_cnu(nu + C::ONE, z),
                    ) else {
                        refused += 1;
                        continue;
                    };
                    let w = j1 * y0 - j0 * y1;
                    let want = z.inv() * (2.0 / std::f64::consts::PI);
                    let scale = (j1 * y0).abs() + (j0 * y1).abs();
                    // The values can be finite while their products are
                    // not; that is the metric's limit, not the routine's.
                    if !scale.is_finite() || scale == 0.0 {
                        continue;
                    }
                    judged += 1;
                    let e = (w - want).abs() / scale;
                    // The J-Y Wronskian cannot resolve below |H1/H2|;
                    // see `the_complex_order_loss_law_holds`. Where the
                    // two Hankel functions are far apart this metric
                    // reports their ratio, not an error.
                    let floor = crate::bessel_cnu_large::hankel_ratio(nu, z)
                        .map_or(0.0, |r| 1.0 / r);
                    if floor > TOL {
                        blind += 1;
                        continue;
                    }
                    worst = worst.max(e);
                    assert!(
                        e <= TOL,
                        "nu={nu:?}, |z|={r}, arg={arg:.2}: residual {e:.2e} exceeds the \
                         {TOL:.0e} these routines promise"
                    );
                }
            }
        }
        assert!(
            judged - blind > 500,
            "only {} points were judged ({blind} blind to |H1/H2|)",
            judged - blind
        );
        assert!(refused > 0, "the refusal path should be exercised too");
        assert!(worst > 1e-12, "worst was {worst:.1e} — is the grid reaching anything hard?");
    }

    /// The large-|z| routes must be reached and used: `|z| = 150` is far
    /// past where the ascending series survives at any order, and Stage
    /// 17 refused it outright.
    #[test]
    fn the_large_argument_routes_extend_the_reach() {
        for &(a, b) in &[(1.3_f64, 0.0_f64), (1.3, 2.0), (0.0, 3.0)] {
            for &r in &[60.0_f64, 150.0, 600.0] {
                let (nu, z) = (C::new(a, b), C::from_polar(r, 0.4));
                let j = bessel_j_cnu(nu, z);
                assert!(j.is_ok(), "J at nu={nu:?}, |z|={r} should now work: {j:?}");
                assert!(bessel_k_cnu(nu, z).is_ok(), "K at nu={nu:?}, |z|={r}");
            }
        }
    }

    #[test]
    fn complex_order_edge_cases() {
        let nu = C::new(1.0, 1.0);
        assert!(bessel_y_cnu(nu, C::ZERO).is_err(), "Y at z = 0");
        assert!(bessel_k_cnu(nu, C::ZERO).is_err(), "K at z = 0");
        assert!(bessel_j_cnu(nu, C::ZERO).is_err(), "z = 0 at complex order");
        // ... but a non-negative real order at z = 0 is fine.
        assert_eq!(bessel_j_cnu(C::real(1.5), C::ZERO).unwrap(), C::ZERO);
        assert_eq!(bessel_j_cnu(C::ZERO, C::ZERO).unwrap(), C::ONE);
        assert!(bessel_j_cnu(C::new(f64::NAN, 0.0), C::ONE).is_err(), "NaN order");
        assert!(
            bessel_j_cnu(nu, C::new(f64::INFINITY, 0.0)).is_err(),
            "infinite argument"
        );
    }
}
