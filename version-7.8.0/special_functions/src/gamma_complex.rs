//! `Gamma` at **complex argument**, by Stirling with argument shifting.
//!
//! The crate has had a real `gamma` from the vendored Cephes since the
//! first milestone. Complex order in the Bessel functions needs
//! `1/Gamma(nu + k + 1)` with `nu` complex, and no amount of care with
//! the real one supplies that.
//!
//! # Why Stirling and not Lanczos
//!
//! Lanczos is the usual choice and is a little faster. It is rejected
//! here for the reason that has shaped this whole crate: **its
//! coefficients are a table**, and the tables in circulation come from
//! sources whose licensing this project will not inherit — the widely
//! copied `g = 7, n = 9` set is most often reproduced from *Numerical
//! Recipes*. Stirling needs no table. It needs the Bernoulli numbers,
//! which are defined by a recurrence this file can state and a test can
//! check:
//!
//! ```text
//!   ln Gamma(z) = (z - 1/2) ln z - z + (1/2) ln(2 pi)
//!                 + sum_{n>=1} B_{2n} / (2n (2n-1) z^(2n-1))
//! ```
//!
//! That asymptotic series is only good for large `|z|`, so a small
//! argument is **shifted** first, using `Gamma(z+1) = z Gamma(z)`
//! repeatedly until `Re z` is comfortably past 12, and the logs of the
//! shift factors are subtracted at the end.
//!
//! # Branches
//!
//! `ln_gamma_c` returns the value the algorithm computes, which is
//! continuous where the shifting keeps it so but is **not** the
//! principal branch everywhere — `ln Gamma` has branch points at every
//! pole, and no single-valued choice is natural. Take
//! [`gamma_c`] when the value rather than a logarithm is wanted; it is
//! single-valued and the exponential of any branch gives the same
//! answer.
//!
//! [`rgamma_c`] is the reciprocal, and it is the one the Bessel series
//! wants: it is **zero** at the poles rather than infinite, so a series
//! term that ought to vanish does.

use crate::complex::Complex64 as C;

/// `B_2, B_4, ..., B_20` as exact fractions.
///
/// Written as ratios rather than decimals because each is exact in
/// `f64` that way and inexact otherwise — `B_12 = -691/2730` has no
/// finite decimal form. `bernoulli_numbers_satisfy_their_recurrence`
/// re-derives them from the definition, so this is a transcription that
/// the crate checks rather than trusts.
const B: [(f64, f64); 10] = [
    (1.0, 6.0),
    (-1.0, 30.0),
    (1.0, 42.0),
    (-1.0, 30.0),
    (5.0, 66.0),
    (-691.0, 2730.0),
    (7.0, 6.0),
    (-3617.0, 510.0),
    (43867.0, 798.0),
    (-174611.0, 330.0),
];

/// `Re z` past which the Stirling series is used directly. Below it the
/// argument is shifted up. At 14 the last term kept, `B_20/(20*19*z^19)`,
/// is about 1e-22 relative, so the series is not the limiting error.
const SHIFT_TO: f64 = 14.0;

/// `ln Gamma(z)` by the Stirling series alone — caller ensures `Re z`
/// is large enough.
fn stirling(z: C) -> C {
    let zi = z.inv();
    let z2i = zi * zi;
    let mut w = zi;
    let mut sum = C::ZERO;
    // The n-th kept term is B_2n / (2n (2n-1) z^(2n-1)), so `n` is the
    // one-based position in the table. The first draft found it with
    // `position(|x| x == (num, den))`, which returns the FIRST match —
    // and B_4 and B_8 are both -1/30, so B_8's term was weighted as if
    // it were B_4's. The result was 3e-11 instead of machine precision,
    // which is exactly the sort of error that looks like "asymptotic
    // series, what did you expect" and is not.
    for (i, &(num, den)) in B.iter().enumerate() {
        let n = (i + 1) as f64;
        sum = sum + w * (num / (den * 2.0 * n * (2.0 * n - 1.0)));
        w = w * z2i;
    }
    (z - C::real(0.5)) * z.ln() - z + C::real(0.5 * (2.0 * std::f64::consts::PI).ln()) + sum
}

/// `ln Gamma(z)`.
///
/// # Errors
/// A non-finite argument, or a pole (`z` a non-positive integer).
///
/// # Examples
/// ```
/// use special_functions::gamma_complex::ln_gamma_c;
/// use special_functions::complex::Complex64 as C;
/// // ln Gamma(1/2) = ln sqrt(pi)
/// let v = ln_gamma_c(C::real(0.5)).unwrap();
/// assert!((v.re - std::f64::consts::PI.sqrt().ln()).abs() < 1e-14);
/// assert!(v.im.abs() < 1e-15);
/// ```
pub fn ln_gamma_c(z: C) -> Result<C, String> {
    if !z.is_finite() {
        return Err(format!("ln_gamma_c: z must be finite, got {z:?}"));
    }
    if z.im == 0.0 && z.re <= 0.0 && z.re == z.re.round() {
        return Err(format!("ln_gamma_c: pole at z = {}", z.re));
    }
    // Reflection for the left half plane: Gamma(z)Gamma(1-z) = pi/sin(pi z).
    if z.re < 0.5 {
        let one_minus = C::ONE - z;
        let s = (z * std::f64::consts::PI).sin();
        if s.abs() == 0.0 {
            return Err(format!("ln_gamma_c: pole at z = {z:?}"));
        }
        return Ok(C::real(std::f64::consts::PI.ln()) - s.ln() - ln_gamma_c(one_minus)?);
    }
    // Shift right until Stirling applies, accumulating the logs of the
    // factors that Gamma(z+1) = z Gamma(z) introduces.
    let mut w = z;
    let mut acc = C::ZERO;
    while w.re < SHIFT_TO {
        acc = acc + w.ln();
        w = w + C::ONE;
    }
    Ok(stirling(w) - acc)
}

/// `Gamma(z)`.
///
/// # Errors
/// As [`ln_gamma_c`], plus overflow.
///
/// # Examples
/// ```
/// use special_functions::gamma_complex::gamma_c;
/// use special_functions::complex::Complex64 as C;
/// // Gamma(n+1) = n!
/// let v = gamma_c(C::real(6.0)).unwrap();
/// assert!((v.re - 120.0).abs() < 1e-11);
/// ```
pub fn gamma_c(z: C) -> Result<C, String> {
    let out = ln_gamma_c(z)?.exp();
    if !out.is_finite() {
        return Err(format!("gamma_c: overflow at z = {z:?}"));
    }
    Ok(out)
}

/// `1/Gamma(z)`, **entire**: exactly zero at every pole of `Gamma`.
///
/// This is the form the Bessel series needs. Dividing by `Gamma` instead
/// would give `1/inf` at the poles if it were lucky and `1/NaN` if it
/// were not, and would overflow long before that — `Gamma` leaves `f64`
/// range past about 171 on the real axis while its reciprocal simply
/// becomes small.
///
/// # Errors
/// A non-finite argument.
pub fn rgamma_c(z: C) -> Result<C, String> {
    if !z.is_finite() {
        return Err(format!("rgamma_c: z must be finite, got {z:?}"));
    }
    if z.im == 0.0 && z.re <= 0.0 && z.re == z.re.round() {
        return Ok(C::ZERO);
    }
    let l = ln_gamma_c(z)?;
    Ok((l * -1.0).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The hardcoded Bernoulli numbers, re-derived from the definition
    /// `sum_{j=0}^{m} C(m+1, j) B_j = 0`. Ten fractions transcribed by
    /// hand is ten chances to be wrong; this is the check.
    #[test]
    fn bernoulli_numbers_satisfy_their_recurrence() {
        let n = 21;
        let mut b = vec![0.0_f64; n + 1];
        b[0] = 1.0;
        // Binomials as exact f64 (they are integers well inside range).
        let mut c = vec![vec![0.0_f64; n + 2]; n + 2];
        for i in 0..n + 2 {
            c[i][0] = 1.0;
            for j in 1..=i {
                c[i][j] = c[i - 1][j - 1] + c[i - 1][j];
            }
        }
        for m in 1..=n {
            let mut s = 0.0;
            for j in 0..m {
                s += c[m + 1][j] * b[j];
            }
            b[m] = -s / c[m + 1][m];
        }
        for (i, &(num, den)) in B.iter().enumerate() {
            let want = num / den;
            let got = b[2 * (i + 1)];
            assert!(
                (got - want).abs() <= 1e-12 * want.abs(),
                "B_{} is {got}, table says {want}",
                2 * (i + 1)
            );
        }
        assert!((b[1] + 0.5).abs() < 1e-15, "B_1 should be -1/2");
        // Odd Bernoulli numbers past the first vanish. The recurrence
        // accumulates rounding as it climbs, so this is checked where
        // that is still small.
        for k in (3..=15).step_by(2) {
            assert!(b[k].abs() < 1e-12, "B_{k} should vanish, got {}", b[k]);
        }
    }

    /// Exact values, on the real axis, where the answer is a fact and
    /// not a comparison.
    #[test]
    fn the_exact_values_are_exact() {
        // Gamma(1/2) = sqrt(pi)
        let v = gamma_c(C::real(0.5)).unwrap();
        // 1e-13 relative, not machine epsilon: reaching Re z = 14 from
        // 0.5 takes fourteen shifts, and each contributes a rounding.
        // That is the method's real accuracy and is worth stating.
        let want = std::f64::consts::PI.sqrt();
        assert!((v.re - want).abs() < 1e-13 * want, "Gamma(1/2) = {v:?}");
        assert!(v.im.abs() < 1e-15);
        // Gamma(n+1) = n!
        let mut f = 1.0_f64;
        for n in 1..=20 {
            f *= n as f64;
            let v = gamma_c(C::real(n as f64 + 1.0)).unwrap();
            assert!((v.re - f).abs() <= 1e-12 * f, "Gamma({}) = {} vs {f}", n + 1, v.re);
        }
        // Gamma(1) = Gamma(2) = 1
        assert!((gamma_c(C::ONE).unwrap().re - 1.0).abs() < 1e-13);
        assert!((gamma_c(C::real(2.0)).unwrap().re - 1.0).abs() < 1e-13);
    }

    /// `|Gamma(1+iy)|^2 = pi y / sinh(pi y)` — a closed form on the
    /// imaginary axis, elementary on the right, and the sharpest test
    /// available for genuinely complex argument.
    #[test]
    fn the_imaginary_axis_closed_form_holds() {
        for &y in &[0.1_f64, 0.5, 1.0, 2.0, 5.0, 12.0, 30.0] {
            let v = gamma_c(C::new(1.0, y)).unwrap();
            let want = std::f64::consts::PI * y / (std::f64::consts::PI * y).sinh();
            assert!(
                (v.norm_sqr() - want).abs() <= 1e-12 * want,
                "|Gamma(1+{y}i)|^2 = {} vs {want}",
                v.norm_sqr()
            );
        }
    }

    /// The recurrence, the reflection and the duplication formula, all
    /// at complex argument. None of them can be satisfied by a
    /// consistently wrong function.
    #[test]
    fn the_functional_equations_hold() {
        for &(re, im) in &[
            (0.7, 0.3),
            (2.5, -1.5),
            (-3.4, 0.8),
            (8.0, 6.0),
            (0.2, -0.1),
            (-0.5, 2.0),
        ] {
            let z = C::new(re, im);
            // Gamma(z+1) = z Gamma(z)
            let a = gamma_c(z + C::ONE).unwrap();
            let b = gamma_c(z).unwrap() * z;
            assert!((a - b).abs() <= 1e-12 * b.abs(), "recurrence at {z:?}");
            // Gamma(z) Gamma(1-z) = pi / sin(pi z)
            let a = gamma_c(z).unwrap() * gamma_c(C::ONE - z).unwrap();
            let b = (z * std::f64::consts::PI).sin().inv() * std::f64::consts::PI;
            assert!((a - b).abs() <= 1e-11 * b.abs(), "reflection at {z:?}");
            // Gamma(z) Gamma(z+1/2) = 2^(1-2z) sqrt(pi) Gamma(2z)
            let a = gamma_c(z).unwrap() * gamma_c(z + C::real(0.5)).unwrap();
            let b = C::real(2.0).powc(C::ONE - z * 2.0)
                * std::f64::consts::PI.sqrt()
                * gamma_c(z * 2.0).unwrap();
            assert!((a - b).abs() <= 1e-10 * b.abs(), "duplication at {z:?}");
        }
    }

    /// Against the vendored Cephes on the real axis, which is an
    /// entirely separate implementation.
    #[test]
    fn the_real_axis_matches_cephes() {
        for &x in &[0.1_f64, 0.5, 1.3, 4.7, 20.0, 60.0, 150.0, -2.5, -7.3] {
            let want = spec_math::cephes64::gamma(x);
            if !want.is_finite() {
                continue;
            }
            let got = gamma_c(C::real(x)).unwrap();
            assert!(
                (got.re - want).abs() <= 1e-12 * want.abs(),
                "Gamma({x}): {} vs {want}",
                got.re
            );
            assert!(got.im.abs() <= 1e-12 * want.abs(), "should be real at x = {x}");
        }
        // ln Gamma too, where Gamma itself has left f64 range.
        for &x in &[200.0_f64, 1000.0, 1e5] {
            let want = spec_math::cephes64::lgam(x);
            let got = ln_gamma_c(C::real(x)).unwrap();
            assert!(
                (got.re - want).abs() <= 1e-13 * want.abs(),
                "lnGamma({x}): {} vs {want}",
                got.re
            );
        }
    }

    /// The reciprocal is entire: zero at the poles, not infinity, and
    /// that is what makes it usable inside a series.
    #[test]
    fn the_reciprocal_vanishes_at_the_poles() {
        for n in 0..8 {
            let v = rgamma_c(C::real(-(n as f64))).unwrap();
            assert_eq!(v, C::ZERO, "1/Gamma(-{n}) should be exactly 0");
        }
        // ... and agrees with 1/Gamma away from them.
        for &(re, im) in &[(1.5, 0.0), (0.3, 0.7), (-2.4, 1.1)] {
            let z = C::new(re, im);
            let a = rgamma_c(z).unwrap();
            let b = gamma_c(z).unwrap().inv();
            assert!((a - b).abs() <= 1e-11 * b.abs(), "1/Gamma at {z:?}");
        }
    }

    #[test]
    fn gamma_edge_cases() {
        assert!(ln_gamma_c(C::ZERO).is_err(), "pole at 0");
        assert!(ln_gamma_c(C::real(-3.0)).is_err(), "pole at -3");
        assert!(gamma_c(C::new(f64::INFINITY, 0.0)).is_err(), "infinite z");
        assert!(rgamma_c(C::new(f64::NAN, 0.0)).is_err(), "NaN z");
        assert!(gamma_c(C::real(200.0)).is_err(), "Gamma(200) overflows f64");
    }
}
