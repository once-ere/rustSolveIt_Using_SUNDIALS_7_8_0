//! Airy functions at **complex argument**.
//!
//! The crate has had real-argument Airy from the vendored Cephes since
//! the first milestone, and that was enough until Stage 18, which
//! recorded the obstacle exactly: the uniform Airy-type expansion of
//! DLMF 10.20 at complex order needs `Ai(nu^(2/3) zeta)` with a complex
//! argument, and there was nothing to call. This is that.
//!
//! # Three regimes, one connection formula
//!
//! * **Small `|z|`** — the ascending series (DLMF 9.4.1 to 9.4.4).
//!   `Ai = c1 f - c2 g` and `Bi = sqrt(3)(c1 f + c2 g)`, with `f` and
//!   `g` entire and their term ratios `z^3/((3k+2)(3k+3))` and
//!   `z^3/((3k+3)(3k+4))`. Entire, so convergent everywhere — but `Ai`
//!   is exponentially small on the positive real axis while `f` and `g`
//!   are exponentially large, so past `|z| ~ 6` the cancellation costs
//!   more than the series is worth.
//!
//! * **Large `|z|`, `|arg z| <= 2pi/3`** — the asymptotic expansions
//!   (DLMF 9.7.5, 9.7.6) in `zeta = (2/3) z^(3/2)`, with
//!   `u_k = (6k-5)(6k-3)(6k-1)/(216 k (2k-1)) u_{k-1}` and
//!   `v_k = -(6k+1)/(6k-1) u_k`.
//!
//! * **Large `|z|`, nearer the negative real axis** — the connection
//!   formula `Ai(z) + w Ai(wz) + w^2 Ai(w^2 z) = 0` with
//!   `w = exp(2 pi i/3)` (DLMF 9.2.12). Rotating by `w` and `w^2` takes
//!   `arg z ~ pi` to `arg ~ -pi/3` and `+pi/3`, where the asymptotic is
//!   at its best. On the negative real axis both rotated points give
//!   `|exp(-zeta)| = 1`, so the combination does not cancel — it is the
//!   same trick Stage 20 used for the Hankel functions, and for the
//!   same reason.
//!
//! # The crossover band
//!
//! Between `|z| = 3` and `10` the series has spent most of its digits
//! on cancellation and the asymptotic is not yet converged. Over a
//! 48000-point sweep the worst Wronskian residual inside that annulus
//! is **3.8e-10**, against **3.7e-12** everywhere outside it. The
//! plane-wide test states the annulus as a separate bound rather than
//! loosening the whole thing to accommodate it.
//!
//! `Bi` never needs an expansion of its own past the series. Its own
//! asymptotic is only valid for `|arg z| < pi/3` anyway, and
//! `Bi(z) = e^(i pi/6) Ai(z e^(2 pi i/3)) + e^(-i pi/6) Ai(z e^(-2 pi i/3))`
//! (DLMF 9.2.10) gives it everywhere from an `Ai` that already works.
//!
//! # Verification
//!
//! The Wronskian `Ai(z) Bi'(z) - Ai'(z) Bi(z) = 1/pi` (DLMF 9.2.7) is
//! **exact, elementary, and involves neither the argument nor any
//! transcendental function on the right**. For a function with no
//! reference implementation at complex argument that is as good a test
//! as exists, and it is what the plane-wide test uses.

use crate::complex::Complex64 as C;

/// `Ai(0)` and `-Ai'(0)`: `3^(-2/3)/Gamma(2/3)` and
/// `3^(-1/3)/Gamma(1/3)` (DLMF 9.2.3, 9.2.4).
const C1: f64 = 0.355_028_053_887_817_2;
const C2: f64 = 0.258_819_403_792_806_8;

/// Beyond this the ascending series has cancelled away more than it is
/// worth. `Ai(x) ~ exp(-2x^(3/2)/3)` while the series terms are
/// `exp(+2x^(3/2)/3)`, so the loss is `exp(4|z|^(3/2)/3)` — about
/// `1e-9` of relative error by `|z| = 6`.
const SERIES_LIMIT: f64 = 6.0;

/// How many `u_k`, `v_k` to build. Optimal truncation is far shorter.
const N_UV: usize = 20;

fn uv() -> &'static (Vec<f64>, Vec<f64>) {
    use std::sync::OnceLock;
    static T: OnceLock<(Vec<f64>, Vec<f64>)> = OnceLock::new();
    T.get_or_init(|| {
        let mut u = vec![1.0_f64];
        for k in 1..N_UV {
            let kf = k as f64;
            let num = (6.0 * kf - 5.0) * (6.0 * kf - 3.0) * (6.0 * kf - 1.0);
            u.push(u[k - 1] * num / (216.0 * kf * (2.0 * kf - 1.0)));
        }
        let v: Vec<f64> = u
            .iter()
            .enumerate()
            .map(|(k, &x)| {
                if k == 0 {
                    1.0
                } else {
                    -(6.0 * k as f64 + 1.0) / (6.0 * k as f64 - 1.0) * x
                }
            })
            .collect();
        (u, v)
    })
}

/// The four values, with an estimate of their shared relative error.
pub struct Airy {
    pub ai: C,
    pub aip: C,
    pub bi: C,
    pub bip: C,
    /// Estimated relative error, from optimal truncation or from the
    /// series' cancellation, whichever route was taken.
    pub err: f64,
}

/// `sum_k s^k c_k / zeta^k` at optimal truncation, over `u` or `v`.
fn asym_sum(coeffs: &[f64], zeta: C, alternating: bool) -> (C, f64) {
    let zi = zeta.inv();
    let mut term = C::ONE;
    let mut sum = C::ONE;
    let mut smallest = 1.0_f64;
    for (k, &c) in coeffs.iter().enumerate().skip(1) {
        let prev = coeffs[k - 1];
        if prev == 0.0 {
            break;
        }
        let mut r = c / prev;
        if alternating {
            r = -r;
        }
        term = term * zi * r;
        let m = term.abs();
        if !m.is_finite() || m >= smallest {
            return (sum, m.min(smallest * 4.0));
        }
        sum = sum + term;
        smallest = m;
    }
    (sum, smallest)
}

/// `Ai` and `Ai'` by the asymptotic expansion, for `|arg z| <= 2pi/3`.
fn ai_asym(z: C) -> Option<(C, C, f64)> {
    let (u, v) = uv();
    let zeta = z.powf(1.5) * (2.0 / 3.0);
    if !zeta.is_finite() || zeta.abs() == 0.0 {
        return None;
    }
    let (su, eu) = asym_sum(u, zeta, true);
    let (sv, ev) = asym_sum(v, zeta, true);
    let root = z.powf(0.25);
    let e = (zeta * -1.0).exp();
    let pre = 1.0 / (2.0 * std::f64::consts::PI.sqrt());
    let ai = e * root.inv() * su * pre;
    let aip = e * root * sv * (-pre);
    // The estimate must carry the EXPONENTIAL's own rounding, not just
    // the series'. `exp(zeta)` is only known to `|zeta| * eps` relative,
    // because `zeta` itself is: at `|z| = 80` that is `477 * 2.2e-16`,
    // or 1e-13, while the truncation says 2.5e-42. Measured against
    // Cephes, 1e-13 is exactly what `Bi(80)` delivers — so without this
    // term the routine would claim forty orders more than it has.
    // Three times, not once: `Bi` is assembled from two `Ai` values
    // (DLMF 9.2.10) and each carries this, and `Bi(80)` measures 3.4e-13
    // against a single-term model of 1.1e-13.
    let rounding = 3.0 * zeta.abs() * f64::EPSILON;
    (ai.is_finite() && aip.is_finite()).then_some((ai, aip, eu.max(ev).max(rounding)))
}

/// `Ai` and `Ai'` anywhere large, rotating into the good sector when
/// the argument is near the negative real axis.
fn ai_large(z: C) -> Option<(C, C, f64)> {
    if z.arg().abs() <= 2.0 * std::f64::consts::PI / 3.0 {
        return ai_asym(z);
    }
    // Ai(z) = -w Ai(wz) - w^2 Ai(w^2 z),  w = exp(2 pi i/3);
    // differentiating, Ai'(z) = -w^2 Ai'(wz) - w Ai'(w^2 z).
    let w = C::from_polar(1.0, 2.0 * std::f64::consts::PI / 3.0);
    let w2 = w * w;
    let (a1, p1, e1) = ai_asym(z * w)?;
    let (a2, p2, e2) = ai_asym(z * w2)?;
    let ai = a1 * w * -1.0 - a2 * w2;
    let aip = p1 * w2 * -1.0 - p2 * w;
    // The two rotated values can cancel — measured, not modelled, the
    // same guard the Hankel pair carries. Without it the estimate at
    // `z = -4 + 5i` claimed 1e-10 and delivered 1.4e-7, because the
    // combination there is a difference of two larger numbers.
    let cancel = |x: C, y: C, r: C| {
        let bottom = r.abs();
        if bottom == 0.0 || !bottom.is_finite() {
            f64::INFINITY
        } else {
            ((x.abs() + y.abs()) / bottom).max(1.0)
        }
    };
    let e = e1.max(e2).max(1e-16) * cancel(a1, a2, ai).max(cancel(p1, p2, aip));
    (ai.is_finite() && aip.is_finite()).then_some((ai, aip, e))
}

/// All four by the ascending series (DLMF 9.4.1 to 9.4.4).
fn series(z: C) -> Option<Airy> {
    let z3 = z * z * z;
    // f and g, and their derivatives, each by their own term ratio.
    let (mut f, mut g) = (C::ONE, z);
    let (mut fp, mut gp) = (z * z * 0.5, C::ONE);
    let (mut tf, mut tg) = (C::ONE, z);
    let (mut tfp, mut tgp) = (z * z * 0.5, C::ONE);
    let mut largest = 1.0_f64;
    for k in 0..60 {
        let kf = k as f64;
        tf = tf * z3 * (1.0 / ((3.0 * kf + 2.0) * (3.0 * kf + 3.0)));
        tg = tg * z3 * (1.0 / ((3.0 * kf + 3.0) * (3.0 * kf + 4.0)));
        // f' starts at k = 1, so its ratio is indexed one behind.
        if k >= 1 {
            tfp = tfp * z3 * ((kf + 1.0) / (kf * (3.0 * kf + 2.0) * (3.0 * kf + 3.0)));
        }
        tgp = tgp * z3 * (1.0 / ((3.0 * kf + 1.0) * (3.0 * kf + 3.0)));
        f = f + tf;
        g = g + tg;
        if k >= 1 {
            fp = fp + tfp;
        }
        gp = gp + tgp;
        largest = largest.max(tf.abs()).max(tg.abs());
        if tf.abs() + tg.abs() <= 1e-18 * (f.abs() + g.abs()).max(1e-300) && k > 3 {
            break;
        }
    }
    let ai = f * C1 - g * C2;
    let aip = fp * C1 - gp * C2;
    let root3 = 3.0_f64.sqrt();
    let bi = (f * C1 + g * C2) * root3;
    let bip = (fp * C1 + gp * C2) * root3;
    // The cancellation, measured from the values rather than modelled:
    // the largest term formed over the smallest answer produced.
    let smallest = ai.abs().min(bi.abs()).max(1e-300);
    // The factor of 100 is measured, not decorative. The bare ratio
    // counts the loss from the single largest term against the smallest
    // answer, and misses two further contributions of the same kind:
    // every one of the ~15 terms carries its own rounding, and
    // `c1 f - c2 g` cancels again on top of that. At `z = 5` the bare
    // ratio says 2e-10 and the delivered error is 8.6e-9.
    let err = (1e-14 * largest / smallest).max(1e-16);
    (ai.is_finite() && bi.is_finite()).then_some(Airy { ai, aip, bi, bip, err })
}

/// `Ai(z)`, `Ai'(z)`, `Bi(z)`, `Bi'(z)`.
///
/// `Bi` comes from `Ai` at the two rotated points (DLMF 9.2.10) rather
/// than from an expansion of its own, since its own is valid only for
/// `|arg z| < pi/3`.
///
/// # Errors
/// A non-finite argument, or a point no route reaches accurately.
///
/// # Examples
/// ```
/// use special_functions::airy_complex::airy_c;
/// use special_functions::complex::Complex64 as C;
/// // The Wronskian Ai Bi' - Ai' Bi = 1/pi, exactly, everywhere.
/// let a = airy_c(C::new(2.0, -3.0)).unwrap();
/// let w = a.ai * a.bip - a.aip * a.bi;
/// assert!((w - C::real(1.0 / std::f64::consts::PI)).abs() < 1e-14);
/// ```
pub fn airy_c(z: C) -> Result<Airy, String> {
    if !z.is_finite() {
        return Err(format!("airy_c: z must be finite, got {z:?}"));
    }
    let ser = if z.abs() <= SERIES_LIMIT { series(z) } else { None };
    // Past the series, Ai from its own routes and Bi from the
    // connection formula, which needs Ai at both rotated points.
    let big = (|| {
        let (ai, aip, e0) = ai_large(z)?;
        let r = C::from_polar(1.0, 2.0 * std::f64::consts::PI / 3.0);
        let (a1, p1, e1) = ai_large(z * r)?;
        let (a2, p2, e2) = ai_large(z * r.conj())?;
        let s = C::from_polar(1.0, std::f64::consts::PI / 6.0);
        let t = C::from_polar(1.0, 5.0 * std::f64::consts::PI / 6.0);
        let bi = a1 * s + a2 * s.conj();
        let bip = p1 * t + p2 * t.conj();
        (bi.is_finite() && bip.is_finite()).then_some(Airy {
            ai,
            aip,
            bi,
            bip,
            err: e0.max(e1).max(e2),
        })
    })();
    let best = match (ser, big) {
        (Some(a), Some(b)) => {
            if a.err <= b.err {
                a
            } else {
                b
            }
        }
        (Some(a), None) => a,
        (None, Some(b)) => b,
        (None, None) => {
            return Err(format!("airy_c: no route reaches z = {z:?}"));
        }
    };
    // All four, not just the one a route happened to check. In the
    // sector where `Ai` is dominant it reaches `exp(492)` by `|z| = 100`
    // — the value is real and simply outside f64, and saying so beats
    // handing back a NaN that only shows up when someone forms a
    // product with it.
    if !(best.ai.is_finite() && best.aip.is_finite() && best.bi.is_finite() && best.bip.is_finite())
    {
        return Err(format!(
            "airy_c: at z = {z:?} at least one of Ai, Ai', Bi, Bi' is outside f64 range \
             — in the sector where a solution is dominant it grows like \
             exp(2|z|^(3/2)/3), which passes the top of the type near |z| = 90"
        ));
    }
    Ok(best)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The Wronskian `Ai Bi' - Ai' Bi = 1/pi` (DLMF 9.2.7), across the
    /// plane. Exact, elementary, and free of both the argument and any
    /// transcendental function on the right — for a function with no
    /// complex reference implementation available, this is the test.
    #[test]
    fn the_wronskian_holds_across_the_plane() {
        let want = C::real(1.0 / std::f64::consts::PI);
        let mut worst = 0.0_f64;
        let mut judged = 0;
        let mut refused = 0;
        for &r in &[0.1_f64, 0.5, 1.0, 3.0, 5.0, 6.0, 6.4, 7.0, 8.0, 10.0, 20.0, 50.0, 200.0] {
            for k in 0..40 {
                let a = -std::f64::consts::PI + k as f64 * std::f64::consts::PI / 20.0;
                let z = C::from_polar(r, a);
                // Past |z| ~ 90 the dominant solution leaves f64 off the
                // real axis — Ai grows like exp(2|z|^(3/2)/3) in its own
                // sector — so those points are refused, correctly, and
                // there is nothing to judge.
                let Ok(v) = airy_c(z) else {
                    refused += 1;
                    continue;
                };
                let w = v.ai * v.bip - v.aip * v.bi;
                // Scaled by the size of the products, so the metric's
                // own cancellation is divided out rather than measured.
                let scale = (v.ai * v.bip).abs() + (v.aip * v.bi).abs();
                // The VALUES can be finite while their PRODUCTS are not:
                // at |z| = 90, arg z = 1.6 both Ai and Bi are about
                // 1e182, so `Ai Bi'` overflows though neither factor
                // does. That is the metric's limit, not the routine's,
                // and skipping those points is honest where pretending
                // the routine failed would not be.
                if !scale.is_finite() || scale == 0.0 {
                    continue;
                }
                let e = (w - want).abs() / scale.max(want.abs());
                worst = worst.max(e);
                judged += 1;
                // The bound names the weak band rather than hiding it.
                // Between |z| = 3 and 10 the series has spent its
                // digits on cancellation and the expansion is not yet
                // converged. Over a 48000-point sweep the worst inside
                // that annulus is 3.8e-10 and the worst outside it
                // 3.7e-12.
                let crossover = (3.0..=10.0).contains(&r);
                let bound = if crossover { 1e-9 } else { 1e-11 };
                assert!(
                    e < bound,
                    "|z|={r}, arg={a:.2}: residual {e:.2e} exceeds {bound:.0e}"
                );
            }
        }
        assert!(judged > 150, "only {judged} points were judged");
        assert!(refused > 0, "the refusal path should be exercised too");
        assert!(worst > 1e-17, "worst was {worst:.1e} — is this reaching anything?");
    }

    /// Against the vendored Cephes on the real axis, both directions.
    /// Cephes is a wholly separate implementation and covers the one
    /// line where a reference exists at all.
    #[test]
    fn the_real_axis_matches_cephes() {
        for &x in &[-30.0_f64, -12.0, -5.0, -1.0, 0.0, 1.0, 5.0, 12.0, 30.0, 80.0] {
            let (ai, aip, bi, bip) = spec_math::cephes64::airy(x);
            let v = airy_c(C::real(x)).unwrap();
            // Bounded by the routine's OWN estimate, floored at 1e-13.
            // On the positive real axis at |z| ~ 5 the series has spent
            // most of its digits on cancellation and the expansion is
            // not yet converged — Ai is exponentially recessive there,
            // and 1e-8 is what the crossover honestly costs. A flat
            // tolerance would either hide that or fail on it.
            let bound = (3.0 * v.err).max(1e-13);
            let cmp = |got: C, want: f64, name: &str| {
                assert!(
                    got.im.abs() <= 1e-12 * got.re.abs().max(1e-12),
                    "{name}({x}) should be real, got {got:?}"
                );
                assert!(
                    (got.re - want).abs() <= bound * want.abs().max(1e-11),
                    "{name}({x}): {} vs {want}, estimate {:.1e}",
                    got.re,
                    v.err
                );
            };
            cmp(v.ai, ai, "Ai");
            cmp(v.aip, aip, "Ai'");
            // Bi overflows in Cephes well before we do.
            if bi.is_finite() && bi.abs() < 1e290 {
                cmp(v.bi, bi, "Bi");
                cmp(v.bip, bip, "Bi'");
            }
        }
    }

    /// The values at the origin are known in closed form
    /// (DLMF 9.2.3 to 9.2.6), and they pin the two constants everything
    /// else is built on.
    #[test]
    fn the_origin_values_are_exact() {
        let v = airy_c(C::ZERO).unwrap();
        assert!((v.ai.re - C1).abs() < 1e-16, "Ai(0)");
        assert!((v.aip.re + C2).abs() < 1e-16, "Ai'(0)");
        assert!((v.bi.re - C1 * 3.0_f64.sqrt()).abs() < 1e-15, "Bi(0)");
        assert!((v.bip.re - C2 * 3.0_f64.sqrt()).abs() < 1e-15, "Bi'(0)");
        // ... and those constants are 3^(-2/3)/Gamma(2/3) and
        // 3^(-1/3)/Gamma(1/3), checked against the complex gamma.
        let g23 = crate::gamma_complex::gamma_c(C::real(2.0 / 3.0)).unwrap().re;
        let g13 = crate::gamma_complex::gamma_c(C::real(1.0 / 3.0)).unwrap().re;
        // 1e-13: the complex gamma's own accuracy is 1e-14, so a
        // tighter bound here would be testing it rather than these.
        assert!((C1 - 3.0_f64.powf(-2.0 / 3.0) / g23).abs() < 1e-13);
        assert!((C2 - 3.0_f64.powf(-1.0 / 3.0) / g13).abs() < 1e-13);
    }

    /// The connection formula `Ai(z) + w Ai(wz) + w^2 Ai(w^2 z) = 0`
    /// (DLMF 9.2.12). It is used internally near the negative real axis,
    /// so testing it *everywhere else* is a genuine check rather than a
    /// tautology.
    #[test]
    fn the_connection_formula_holds() {
        let w = C::from_polar(1.0, 2.0 * std::f64::consts::PI / 3.0);
        for &r in &[0.5_f64, 2.0, 5.0, 15.0, 40.0] {
            for k in 0..8 {
                let a = -0.9 + k as f64 * 0.25;
                let z = C::from_polar(r, a);
                let (v0, v1, v2) = (
                    airy_c(z).unwrap(),
                    airy_c(z * w).unwrap(),
                    airy_c(z * w * w).unwrap(),
                );
                let s = v0.ai + v1.ai * w + v2.ai * w * w;
                let scale = v0.ai.abs() + v1.ai.abs() + v2.ai.abs();
                assert!(
                    s.abs() <= 1e-11 * scale,
                    "|z|={r}, arg={a:.2}: {:.2e}",
                    s.abs() / scale
                );
            }
        }
    }

    /// `Ai` and `Bi` are real on the real axis and have real Taylor
    /// coefficients, so `f(conj z) = conj f(z)`. A Wronskian cannot
    /// catch a wrongly conjugated pair; this can.
    #[test]
    fn conjugation_symmetry_holds() {
        for &(re, im) in &[(2.0, 3.0), (-4.0, 1.5), (0.5, -0.5), (-20.0, 8.0), (30.0, -10.0)] {
            let z = C::new(re, im);
            let a = airy_c(z).unwrap();
            let b = airy_c(z.conj()).unwrap();
            for (x, y, name) in [
                (a.ai, b.ai, "Ai"),
                (a.aip, b.aip, "Ai'"),
                (a.bi, b.bi, "Bi"),
                (a.bip, b.bip, "Bi'"),
            ] {
                assert!(
                    (x.conj() - y).abs() <= 1e-12 * y.abs().max(1e-12),
                    "{name} conjugation at {z:?}"
                );
            }
        }
    }

    /// `Ai'' = z Ai`, the defining equation, by a central difference.
    /// Nothing in the implementation enforces it — the series and the
    /// asymptotic are both written from their own formulas — so it is
    /// an independent check that they solve the right problem.
    #[test]
    fn the_differential_equation_is_satisfied() {
        for &(re, im) in &[(1.0, 0.0), (2.0, 1.0), (-3.0, 0.5), (-1.0, -2.0), (4.0, -1.0)] {
            let z = C::new(re, im);
            // 1e-3, not 1e-4: a second difference amplifies rounding by
            // 1/h^2, so a smaller step makes this worse, not better.
            let h = 1e-3;
            let (a, b, c) = (
                airy_c(z - C::real(h)).unwrap().ai,
                airy_c(z).unwrap().ai,
                airy_c(z + C::real(h)).unwrap().ai,
            );
            let second = (a + c - b * 2.0) * (1.0 / (h * h));
            let want = z * b;
            assert!(
                // 1e-5 is the central difference's own truncation:
                // h^2 times a fourth derivative that grows like z^2 —
                // not the Airy routine's accuracy, which is far better.
                (second - want).abs() <= 1e-5 * want.abs().max(1e-6),
                "Ai'' = z Ai at {z:?}: {second:?} vs {want:?}"
            );
        }
    }

    #[test]
    fn airy_edge_cases() {
        assert!(airy_c(C::new(f64::NAN, 0.0)).is_err(), "NaN");
        assert!(airy_c(C::new(f64::INFINITY, 0.0)).is_err(), "infinite");
        // The origin and its neighbourhood are the series' business and
        // must not fall through to an expansion in 1/zeta.
        assert!(airy_c(C::ZERO).is_ok());
        assert!(airy_c(C::new(1e-8, -1e-8)).is_ok());
    }
}
