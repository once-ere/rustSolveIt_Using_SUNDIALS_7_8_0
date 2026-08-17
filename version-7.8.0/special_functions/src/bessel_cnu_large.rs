//! The large-argument and large-order expansions, at **complex order**.
//!
//! Stage 17 left complex order reaching only as far as the ascending
//! series does, and recorded that as the remaining limit: the Debye,
//! Airy-type and `1/z` machinery are expansions *in* the order, and
//! their uniformity is stated for real order. This module works out
//! which of those extend, and extends them.
//!
//! # What extends, and why
//!
//! * **The `1/z` asymptotics** (DLMF 10.17.5, 10.17.6, 10.40.1,
//!   10.40.2) are expansions at *fixed* order, and the order enters
//!   only through `mu = 4 nu^2` — polynomially. Nothing in them assumes
//!   `nu` is real. This is the extension that matters, because the gap
//!   it fills is `|z|` large at **any** order, which the measurement
//!   showed failing from `|z| ~ 30` upward regardless of `nu`.
//! * **The uniform expansions of DLMF 10.41** for `I` and `K` extend to
//!   complex order in the sector `|arg nu| < pi/2` (DLMF 10.41.5), with
//!   the same Debye polynomials. That fills the large-`|nu|` rows.
//! * **The Airy-type expansion of DLMF 10.20 is not implemented in
//!   this module**: it needs `Ai(nu^(2/3) zeta)`, and with `nu` complex
//!   that argument is complex — at the time this module was written the
//!   crate had only the real-argument Airy from the vendored Cephes.
//!   That obstacle was later removed: [`crate::airy_complex`] supplies
//!   complex-argument Airy and [`crate::airy_uniform::jy_airy_c`]
//!   carries DLMF 10.20 to complex order, so the turning point
//!   `z ~ nu` is covered there rather than here.
//!
//! # Estimates, and the one that has to be measured
//!
//! Every route reports an optimal-truncation estimate, as everywhere
//! else in this crate, so the caller chooses by comparison rather than
//! by a validity rule. The exception is `J` and `Y` built from the
//! Hankel pair: there the loss is a **cancellation**, which depends on
//! the values and not on the expansion, so it is measured from them.

use crate::complex::Complex64 as C;

/// `sum_k c^k a_k(nu) / z^k` at optimal truncation, for complex order.
///
/// The same series as [`crate::bessel_scaled`]; kept here rather than
/// shared because that module's copy is wrapped in real-order plumbing
/// all the way down. The recurrence is
/// `a_k = a_{k-1} (4 nu^2 - (2k-1)^2)/(8k)`.
fn asym_sum_c(nu: C, z: C, c: C) -> (C, f64) {
    let mu = nu * nu * 4.0;
    let step = c * z.inv();
    let mut term = C::ONE;
    let mut sum = C::ONE;
    let mut smallest = 1.0_f64;
    for k in 1..=100 {
        let f = (mu - C::real(((2 * k - 1) as f64).powi(2))) * (1.0 / (8.0 * k as f64));
        let next = term * step * f;
        let m = next.abs();
        if !m.is_finite() || m >= smallest {
            return (sum, m.min(smallest * 4.0));
        }
        sum = sum + next;
        term = next;
        smallest = m;
        if m == 0.0 {
            return (sum, 0.0);
        }
    }
    (sum, smallest)
}

/// A value with the estimate of its relative error, or nothing.
pub type Cand = Option<(C, f64)>;

/// Is the order small enough for an expansion **in `1/z` at fixed
/// order** to mean anything here?
///
/// The order enters as `mu = 4 nu^2`, and the terms
/// `a_k = a_{k-1}(mu - (2k-1)^2)/(8k z)` only start shrinking once
/// `2k-1` passes `|2 nu|`. If `|mu|` is comparable to `|z|` the series
/// crawls before it falls, and optimal truncation badly understates the
/// error: measured, at `nu = 0.5 + 5i` and `|z| = 22` the actual error
/// was **2163 times** the estimate. Requiring `|mu| <= 2|z|` keeps the
/// terms falling from the start, and with it the worst ratio over the
/// same grid drops into single figures.
///
/// This is a *validity* condition, not a tolerance. It is the thing a
/// truncation estimate structurally cannot tell you, which is why it is
/// checked separately rather than folded into a safety factor.
fn order_is_small_enough(nu: C, z: C) -> bool {
    let mu = (nu * nu * 4.0).abs();
    // `8|z|` is the condition that the FIRST ratio `|mu - 1|/(8|z|)` is
    // below 1, i.e. the terms start shrinking immediately. That is
    // enough for a real order, where the coefficients are real and the
    // series behaves. It is not enough once the order is complex: at
    // `nu = 0.5 + 5i` and `|z| = 22` the terms shrink but slowly, and
    // the actual error ran 2163 times the truncation estimate. So a
    // complex order is held to the stricter `2|z|`.
    let limit = if nu.im == 0.0 { 8.0 } else { 2.0 };
    mu <= limit * z.abs()
}

/// Floor on any estimate.
///
/// Not one `eps` but fifty. An expansion that terminates exactly is
/// still only *evaluated* to `f64` precision, and the evaluation here
/// is a `powf` prefactor, an `exp`, and a twenty-term complex sum.
/// Measured: at `nu = 1.3, |z| = 18, arg z = 0.7` the truncation
/// estimate bottoms out at 2e-15 while the actual error is 3.2e-14, and
/// the difference is exactly that accumulated rounding. A floor of one
/// `eps` would have the routine claiming an accuracy its arithmetic
/// cannot deliver — the same class of dishonest estimate that Stage 16
/// found was preventing a better method from being chosen.
const FLOOR: f64 = 5e-14;

/// Optimal truncation gives the size of the first omitted term, which
/// is the classical *estimate* and not a bound.
///
/// Measured over a 12089-point sweep in `|z|`, `arg z` and complex
/// order, the worst ratio of actual error to smallest term is about
/// **110**, at moderate `|z|` where the series only just converges —
/// at `nu = 1.3, |z| = 8` the smallest term is 3.1e-8 and the error
/// 6.5e-7. 150 makes the reported figure a bound over that sweep.
///
/// [`crate::bessel_scaled`] settled on 10 for the same quantity, and
/// the difference is not a disagreement: that module offers its routes
/// only where `|z|` is already large, while these are offered from
/// `|z| ~ 6` upward and have to be honest there too.
const SAFETY: f64 = 150.0;

fn ok(v: C, e: f64) -> Cand {
    (v.is_finite() && e.is_finite()).then_some((v, (e * SAFETY).max(FLOOR)))
}

/// `H1_nu(z)` and `H2_nu(z)`, unscaled, from DLMF 10.17.5 and 10.17.6.
///
/// Returns both and the worse truncation estimate. The sector guards
/// are the ones Stage 14 measured the need for: 10.17.5 holds for
/// `-pi < arg z < 2pi` and 10.17.6 for `-2pi < arg z < pi`, so the
/// negative real axis is interior for one and on the boundary for the
/// other, and a margin is kept from each.
fn hankel_pair(nu: C, z: C) -> Option<(C, C, f64)> {
    if !order_is_small_enough(nu, z) {
        return None;
    }
    // pi/3, not the pi/4 the real-order version keeps. Measured: at
    // `arg z = -2.2` — inside a pi/4 margin — the actual error ran 222
    // times the truncation estimate, because near the sector boundary
    // the recessive member's expansion is subject to the Stokes
    // phenomenon and optimal truncation does not see it.
    let m = std::f64::consts::FRAC_PI_3;
    if z.arg() <= -std::f64::consts::PI + m || z.arg() >= std::f64::consts::PI - m {
        return None;
    }
    let (s1, e1) = asym_sum_c(nu, z, C::I);
    let (s2, e2) = asym_sum_c(nu, z, C::I * -1.0);
    let pref = (C::real(2.0 / std::f64::consts::PI) * z.inv()).powf(0.5);
    let ph = nu * std::f64::consts::FRAC_PI_2 + C::real(std::f64::consts::FRAC_PI_4);
    let h1 = pref * (C::I * ph * -1.0).exp() * s1 * (C::I * z).exp();
    let h2 = pref * (C::I * ph).exp() * s2 * (C::I * z * -1.0).exp();
    (h1.is_finite() && h2.is_finite()).then_some((h1, h2, e1.max(e2)))
}

/// The Hankel pair near `arg z = ±pi`, by continuing from the positive
/// real axis (DLMF 10.11.3, 10.11.4 with `m = ±1`).
///
/// The `1/z` expansions of DLMF 10.17.5 and 10.17.6 hold on
/// `-pi < arg z < 2pi` and `-2pi < arg z < pi`, so the negative real
/// axis is interior to one and on the boundary of the other, and
/// [`hankel_pair`] keeps a `pi/3` margin from it. Everything in that
/// wedge then fell back to the ascending series and its `exp(|z|)`
/// loss — measured, a J-Y Wronskian residual of 1.0 at `|z| = 60`.
///
/// The remedy is not a new expansion but a change of variable. With
/// `w = -z`, so that `arg w` is near zero where both expansions are at
/// their best:
///
/// ```text
///   arg z ~ +pi:  H1(z) = -e^(-i nu pi) H2(w)
///                 H2(z) =  e^(i nu pi) H1(w) + 2 cos(nu pi) H2(w)
///   arg z ~ -pi:  H1(z) =  2 cos(nu pi) H1(w) + e^(-i nu pi) H2(w)
///                 H2(z) = -e^(i nu pi) H1(w)
/// ```
///
/// The two cases are different because `z` is on **different sides of
/// the cut**, and that is the point rather than an inconvenience: `Y`
/// and the Hankel functions really are discontinuous there, and the
/// side is chosen by the sign of `Im z` — including its signed zero,
/// which is the convention the rest of this crate already follows.
///
/// On the negative real axis `|H1(w)| = |H2(w)|`, so neither
/// combination cancels; the estimate carries through unchanged.
fn hankel_pair_continued(nu: C, z: C) -> Option<(C, C, f64)> {
    let a = z.arg();
    // Only inside the wedge `hankel_pair` refuses.
    if a.abs() <= std::f64::consts::PI - std::f64::consts::FRAC_PI_3 {
        return None;
    }
    let w = z * -1.0;
    let (h1w, h2w, e) = hankel_pair(nu, w)?;
    let epi = (C::I * nu * std::f64::consts::PI).exp();
    let two_cos = (nu * std::f64::consts::PI).cos() * 2.0;
    let (h1, h2) = if a >= 0.0 {
        (h2w * epi.inv() * -1.0, h1w * epi + h2w * two_cos)
    } else {
        (h1w * two_cos + h2w * epi.inv(), h1w * epi * -1.0)
    };
    (h1.is_finite() && h2.is_finite()).then_some((h1, h2, e))
}

/// The Hankel pair by whichever route reaches this `z`.
pub(crate) fn hankel_pair_any(nu: C, z: C) -> Option<(C, C, f64)> {
    hankel_pair(nu, z).or_else(|| hankel_pair_continued(nu, z))
}

/// `|H1_nu(z) / H2_nu(z)|`, the factor by which the J-Y Wronskian
/// degenerates as a measuring instrument.
///
/// `J` and `Y` are both `(H1 +- H2)/2`, so when one Hankel function
/// dominates the other they are **the same function to within
/// `|H1/H2|`**. Their Wronskian is of size `|H1 H2|` while each product
/// forming it is of size `|H2|^2`, so a residual scaled by the largest
/// term measures `|H1/H2|` and nothing about the values. Tests use this
/// to know when that instrument has nothing left to say.
#[cfg(test)]
pub(crate) fn hankel_ratio(nu: C, z: C) -> Option<f64> {
    let (h1, h2, _) = hankel_pair_any(nu, z)?;
    let (a, b) = (h1.abs(), h2.abs());
    let r = (a / b).max(b / a);
    r.is_finite().then_some(r)
}

/// How much forming `J` or `Y` from the Hankel pair costs, measured
/// from the values rather than modelled — the sum of the ingredient
/// magnitudes over the magnitude of the result.
fn cancellation(a: C, b: C, result: C) -> f64 {
    let bottom = result.abs();
    if bottom == 0.0 || !bottom.is_finite() {
        return f64::INFINITY;
    }
    ((a.abs() + b.abs()) / bottom).max(1.0)
}

/// `J_nu(z)` by the large-argument route.
pub fn j_asym(nu: C, z: C) -> Cand {
    let (h1, h2, e) = hankel_pair_any(nu, z)?;
    let v = (h1 + h2) * 0.5;
    ok(v, e.max(FLOOR) * cancellation(h1, h2, v))
}

/// `Y_nu(z)` by the large-argument route.
pub fn y_asym(nu: C, z: C) -> Cand {
    let (h1, h2, e) = hankel_pair_any(nu, z)?;
    let v = (h1 - h2) / (C::I * 2.0);
    ok(v, e.max(FLOOR) * cancellation(h1, h2, v))
}

/// `K_nu(z)` from DLMF 10.40.2 — one term, no cancellation.
pub fn k_asym(nu: C, z: C) -> Cand {
    // DLMF 10.40.2 holds for `|arg z| < 3 pi/2`, so the whole principal
    // sheet is interior to it — unlike the Hankel expansions, whose
    // sectors end at `pi`. The pi/4 margin kept here originally was
    // copied from those and was simply too strict: it cut `K` off from
    // the negative real axis for no reason the mathematics gives.
    if !order_is_small_enough(nu, z) {
        return None;
    }
    let (s, e) = asym_sum_c(nu, z, C::ONE);
    let pref = (C::real(std::f64::consts::PI * 0.5) * z.inv()).powf(0.5);
    ok(pref * (z * -1.0).exp() * s, e)
}

/// `I_nu(z)` from DLMF 10.40.1, first term only.
///
/// The dropped second term is `exp(+-(nu + 1/2) pi i) exp(-z)/sqrt(2 pi z)`
/// times its own series, and the truncation estimate cannot see it —
/// the trap Stage 14 fell into at `nu = 1/2`, where the series
/// terminates exactly and the estimate was zero while the dropped term
/// was the whole error.
///
/// **At complex order that term is bigger than it looks.** Its size
/// relative to the one kept is not `exp(-2 Re z)` but
/// `exp(pi |Im nu| - 2 Re z)`, because `|exp(i(nu+1/2)pi)|` is
/// `exp(-pi Im nu)` and the sign is the caller's to lose. Measured: at
/// `nu = 3i` and `z = 14.9 - 10.2i` the real-order form of this
/// estimate understated the error by a factor of 678, and the
/// difference is `exp(3 pi)` to within the noise.
pub fn i_asym(nu: C, z: C) -> Cand {
    if !order_is_small_enough(nu, z) || z.re <= 0.0 || z.re < z.im.abs() {
        return None;
    }
    let (s, e) = asym_sum_c(nu, z, C::ONE * -1.0);
    let pref = (C::real(1.0 / (2.0 * std::f64::consts::PI)) * z.inv()).powf(0.5);
    let dropped = (std::f64::consts::PI * nu.im.abs() - 2.0 * z.re).exp();
    ok(pref * z.exp() * s, e.max(dropped))
}

/// `I_nu(z)` and `K_nu(z)` from the uniform expansions of DLMF 10.41 at
/// complex order, valid for `|arg nu| < pi/2` (DLMF 10.41.5).
///
/// [`crate::debye::ik_uniform`] returns them scaled; this undoes the
/// scaling, which is where they leave `f64` range if they are going to.
pub fn ik_uniform_unscaled(nu: C, z: C) -> (Cand, Cand) {
    if nu.abs() == 0.0 || nu.re <= 0.0 {
        return (None, None);
    }
    let (i, k) = crate::debye::ik_uniform(nu, z);
    (
        i.and_then(|u| ok(u.value * z.re.abs().exp(), u.err)),
        k.and_then(|u| ok(u.value * (z * -1.0).exp(), u.err)),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bessel_cnu::{bessel_i_cnu, bessel_j_cnu, bessel_k_cnu, bessel_y_cnu};

    fn close(a: C, b: C, tol: f64) -> bool {
        (a - b).abs() <= tol * b.abs().max(1e-300)
    }

    /// Where the ascending series is still sound, the expansions must
    /// agree with it. This is the join between the two, and it is the
    /// only place a reference exists at all for complex order.
    #[test]
    fn the_expansions_agree_with_the_series_where_it_is_sound() {
        let mut checked = 0;
        for &(a, b) in &[(1.3_f64, 0.0_f64), (1.3, 2.0), (0.5, 5.0), (-0.7, 1.0), (3.0, -2.0)] {
            for &r in &[8.0_f64, 12.0, 18.0, 25.0] {
                for &arg in &[0.0_f64, 0.7, -0.7, 1.5, 2.0] {
                    let (nu, z) = (C::new(a, b), C::from_polar(r, arg));
                    // The series' law, from Stage 17.
                    let loss = r - z.im.abs() + (b * arg).abs();
                    if 1e-16 * loss.exp() > 1e-13 {
                        continue;
                    }
                    // Asserted against each route's OWN estimate. A
                    // fixed tolerance would be testing the wrong thing:
                    // `I` at Re z = 6.1 disagrees by 5e-6, and its
                    // estimate says 4.8e-6, because that is the size of
                    // the exponentially small term DLMF 10.40.1 drops.
                    // The estimate is right; a flat 1e-6 was not.
                    let bound = |e: f64| e.max(1e-15);
                    if let Some((v, e)) = j_asym(nu, z) {
                        checked += 1;
                        let w = bessel_j_cnu(nu, z).unwrap();
                        assert!(close(v, w, bound(e)), "J at nu={nu:?} z={z:?}: {v:?} vs {w:?}");
                    }
                    if let Some((v, e)) = y_asym(nu, z) {
                        let w = bessel_y_cnu(nu, z).unwrap();
                        assert!(close(v, w, bound(e)), "Y at nu={nu:?} z={z:?}: {v:?} vs {w:?}");
                    }
                    if let Some((v, e)) = k_asym(nu, z) {
                        let w = bessel_k_cnu(nu, z).unwrap();
                        assert!(close(v, w, bound(e)), "K at nu={nu:?} z={z:?}: {v:?} vs {w:?}");
                    }
                    if let Some((v, e)) = i_asym(nu, z) {
                        let w = bessel_i_cnu(nu, z).unwrap();
                        assert!(close(v, w, bound(e)), "I at nu={nu:?} z={z:?}: {v:?} vs {w:?}");
                    }
                }
            }
        }
        assert!(checked > 20, "only {checked} comparisons were in range");
    }

    /// The half-integer closed forms hold for the expansions too, and at
    /// a `|z|` no series here can reach. `H1_{1/2}(z) = -i sqrt(2/(pi z))
    /// exp(iz)` — a complex order is not involved, but it pins the
    /// prefactor and phase that a complex order then rides on.
    #[test]
    fn the_half_integer_closed_form_pins_the_prefactor() {
        for &(re, im) in &[(200.0, 0.0), (500.0, -3.0), (60.0, -40.0)] {
            let z = C::new(re, im);
            let (h1, _, e) = hankel_pair(C::real(0.5), z).unwrap();
            let want = C::I * -1.0
                * (C::real(2.0 / std::f64::consts::PI) * z.inv()).powf(0.5)
                * (C::I * z).exp();
            assert!(close(h1, want, 1e-13), "H1_1/2 at {z:?}: estimate {e:.1e}");
        }
    }

    /// The J-Y Wronskian, computed entirely from the expansions, at
    /// complex order and at a magnitude the series cannot reach. Its
    /// right-hand side involves neither the order nor any Bessel
    /// function, so it judges them without a reference.
    /// `FLOOR` is what stops a route claiming an accuracy its own
    /// arithmetic cannot deliver.
    ///
    /// Stage 2F's mutation probe dropped it from 5e-14 to 1e-300 and the
    /// whole suite still passed — nothing asserted the constant's value.
    /// That is the exact shape of the Stage 16 defect it exists to
    /// prevent: an estimate that bottoms out too low wins every
    /// comparison in the selector, and a worse route is chosen.
    ///
    /// It binds where optimal truncation reports **zero**, and there is
    /// a place that happens exactly: at `nu = 1/2` the `1/z` Hankel
    /// series *terminates*, so the first omitted term is identically 0
    /// and the estimate would be 0 too. Stage 15 found that
    /// `0 x 1e14 = 0` let this route win every comparison it entered.
    ///
    /// The value is still not exact — a `powf` prefactor, an `exp` and a
    /// complex sum are evaluated in `f64` — so the honest estimate is the
    /// floor, and the closed form `J_{1/2}(z) = sqrt(2/(pi z)) sin z`
    /// says what the error really is.
    #[test]
    fn the_estimate_floor_binds_where_truncation_reports_zero() {
        let nu = C::real(0.5);
        for &(r, th) in &[(18.0_f64, 0.7_f64), (40.0, -0.4), (120.0, 0.0)] {
            let z = C::from_polar(r, th);
            let (j, e) = j_asym(nu, z).expect("the 1/z route reaches nu = 1/2");

            // The series terminates here, so the reported estimate is
            // the floor and nothing else.
            assert!(
                e >= FLOOR,
                "at nu = 1/2 truncation reports 0, so the estimate must be the floor, got {e:.2e}"
            );

            // And the floor must actually cover the error made.
            let sin = ((C::I * z).exp() - (C::I * z * -1.0).exp()) * (C::I * 2.0).inv();
            let want = (C::real(2.0 / std::f64::consts::PI) * z.inv()).powf(0.5) * sin;
            let actual = (j - want).abs() / want.abs();
            assert!(
                actual <= e,
                "|z| = {r}: the routine claims {e:.2e} and is wrong by {actual:.2e}"
            );
        }
    }

    #[test]
    fn the_wronskian_holds_far_beyond_the_series() {
        for &(a, b) in &[(1.3_f64, 0.0_f64), (2.0, 3.0), (0.0, 6.0), (-1.5, 2.5)] {
            for &r in &[60.0_f64, 150.0, 600.0] {
                for &arg in &[0.0_f64, 0.6, -1.2] {
                    let (nu, z) = (C::new(a, b), C::from_polar(r, arg));
                    let (Some((j0, _)), Some((j1, _))) =
                        (j_asym(nu, z), j_asym(nu + C::ONE, z))
                    else {
                        continue;
                    };
                    let (Some((y0, _)), Some((y1, _))) =
                        (y_asym(nu, z), y_asym(nu + C::ONE, z))
                    else {
                        continue;
                    };
                    let w = j1 * y0 - j0 * y1;
                    let want = z.inv() * (2.0 / std::f64::consts::PI);
                    let scale = (j1 * y0).abs() + (j0 * y1).abs();
                    // The VALUES can be finite while their PRODUCTS are
                    // not: at Im z = -559 both J and Y are about 1e243,
                    // so `J Y` overflows even though neither factor
                    // does. That is a limit of the metric, not of the
                    // expansion, so those points are skipped rather than
                    // reported as failures.
                    if !scale.is_finite() || scale == 0.0 {
                        continue;
                    }
                    // Floored at |H1/H2|: past that the J-Y Wronskian
                    // is measuring the Hankel ratio, not these values.
                    let floor = hankel_ratio(nu, z).map_or(0.0, |r| 1.0 / r);
                    assert!(
                        (w - want).abs() / scale < (1e-11f64).max(10.0 * floor),
                        "nu={nu:?} z={z:?}: residual {:.2e}",
                        (w - want).abs() / scale
                    );
                }
            }
        }
    }

    /// The I-K Wronskian likewise, and it reaches the uniform route as
    /// well as the `1/z` one.
    #[test]
    fn the_i_k_wronskian_holds_at_complex_order() {
        for &(a, b) in &[(2.0_f64, 1.0_f64), (5.0, 4.0), (20.0, 10.0)] {
            for &r in &[30.0_f64, 90.0, 300.0] {
                let (nu, z) = (C::new(a, b), C::real(r));
                let (i0, k0) = (i_asym(nu, z), k_asym(nu, z));
                let (i1, k1) = (i_asym(nu + C::ONE, z), k_asym(nu + C::ONE, z));
                let (Some((i0, _)), Some((i1, _)), Some((k0, _)), Some((k1, _))) =
                    (i0, i1, k0, k1)
                else {
                    continue;
                };
                let w = i0 * k1 + i1 * k0;
                let scale = (i0 * k1).abs() + (i1 * k0).abs();
                assert!(
                    (w - z.inv()).abs() / scale < 1e-11,
                    "nu={nu:?} r={r}: residual {:.2e}",
                    (w - z.inv()).abs() / scale
                );
            }
        }
    }

    /// The uniform route at large complex order, where nothing else
    /// reaches, judged by the same Wronskian.
    #[test]
    fn the_uniform_route_holds_at_large_complex_order() {
        for &(a, b) in &[(60.0_f64, 10.0_f64), (150.0, 20.0), (400.0, 50.0)] {
            for &r in &[20.0_f64, 60.0, 200.0] {
                let (nu, z) = (C::new(a, b), C::real(r));
                let (i0, k0) = ik_uniform_unscaled(nu, z);
                let (i1, k1) = ik_uniform_unscaled(nu + C::ONE, z);
                let (Some((i0, _)), Some((i1, _)), Some((k0, _)), Some((k1, _))) =
                    (i0, i1, k0, k1)
                else {
                    continue;
                };
                let w = i0 * k1 + i1 * k0;
                let scale = (i0 * k1).abs() + (i1 * k0).abs();
                assert!(
                    (w - z.inv()).abs() / scale < 1e-10,
                    "nu={nu:?} r={r}: residual {:.2e}",
                    (w - z.inv()).abs() / scale
                );
            }
        }
    }

    /// The continuation is exact algebra, so where **both** routes
    /// apply they must agree to the last bit. That is the sharpest test
    /// of DLMF 10.11.3/4 as transcribed here: a sign error in either
    /// formula shows up immediately, and no reference is involved.
    #[test]
    fn the_continuation_agrees_with_the_direct_route_where_both_apply() {
        for &(a, b) in &[(1.3_f64, 0.0_f64), (2.0, 0.0), (0.5, 1.0), (1.7, -2.0)] {
            for &r in &[15.0_f64, 40.0, 120.0] {
                for &arg in &[1.6_f64, 1.9, 2.05, -1.6, -1.9, -2.05] {
                    let (nu, z) = (C::new(a, b), C::from_polar(r, arg));
                    let (Some((d1, d2, _)), Some((c1, c2, _))) =
                        (hankel_pair(nu, z), hankel_pair_continued(nu, z))
                    else {
                        continue;
                    };
                    // The recessive member of the pair is only
                    // determined to `eps` times the dominant one — the
                    // Stokes phenomenon — so agreement is asserted
                    // relative to the LARGER of the two.
                    let scale = d1.abs().max(d2.abs());
                    assert!(
                        (c1 - d1).abs() <= 1e-12 * scale && (c2 - d2).abs() <= 1e-12 * scale,
                        "nu={nu:?} z={z:?}: direct ({d1:?}, {d2:?}) vs continued ({c1:?}, {c2:?})"
                    );
                }
            }
        }
    }

    #[test]
    fn large_order_edge_cases() {
        let nu = C::new(1.0, 1.0);
        // The negative real axis is outside both Hankel sectors — and
        // since Stage 20 it is reached anyway, by continuing from the
        // positive one. This assertion used to say `is_none()`.
        assert!(j_asym(nu, C::real(-50.0)).is_some(), "arg z = pi is now covered");
        // I's expansion is only used near the real axis.
        assert!(i_asym(nu, C::new(1.0, 50.0)).is_none(), "near the imaginary axis");
        assert!(i_asym(nu, C::real(-10.0)).is_none(), "Re z < 0");
        // The uniform route needs Re nu > 0.
        assert!(ik_uniform_unscaled(C::new(-1.0, 1.0), C::ONE).0.is_none());
        assert!(ik_uniform_unscaled(C::ZERO, C::ONE).0.is_none());
    }
}
