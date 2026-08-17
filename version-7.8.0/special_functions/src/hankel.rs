//! Hankel functions — the travelling-wave pair.
//!
//! `J` and `Y` are the standing-wave basis of the Bessel equation.
//! `H^(1)` and `H^(2)` are the *travelling*-wave basis of the same
//! equation, and for wave problems they are the ones you actually want:
//!
//! ```text
//!     H1_nu(z) = J_nu(z) + i Y_nu(z)        (DLMF 10.4.3)
//!     H2_nu(z) = J_nu(z) - i Y_nu(z)
//! ```
//!
//! With a time convention `exp(-i omega t)`, `H1` is the **outgoing**
//! wave and `H2` the incoming one: at large real `x`,
//! `H1_nu(x) ~ sqrt(2/(pi x)) exp(i(x - nu pi/2 - pi/4))`, a pure
//! `exp(+ikr)/sqrt(r)` cylindrical wave. That is why a scattering
//! boundary condition is stated in terms of `H1` and not `J` and `Y`.
//!
//! # Why these are entry points rather than a note in the manual
//!
//! Every function here is two calls to routines this crate already
//! exports, and the module documentation used to say exactly that —
//! "constructible, so not registered". That was the wrong call, for a
//! reason the accuracy discussion below makes concrete: **the naive
//! construction silently loses most of its digits in half the plane**,
//! and a user assembling `J + iY` by hand has no way to know. A named
//! entry point is where that knowledge can live.
//!
//! # Spherical Hankel
//!
//! `h1_n(x) = j_n(x) + i y_n(x)` and `h2_n = j_n - i y_n` are the
//! three-dimensional counterparts, and they carry the same meaning: the
//! radial part of an outgoing/incoming spherical wave. These take a
//! **real** argument, matching [`crate::sph_bessel`], and return a
//! complex value. Their closed forms are elementary — `h1_0(x)` is
//! exactly `-i exp(ix)/x` — which makes them easy to check absolutely.
//!
//! # Accuracy: the cancellation is intrinsic, not a defect
//!
//! In the upper half plane `H1` **decays** like `exp(-Im z)` while `J`
//! and `Y` each **grow** like `exp(|Im z|)`. Their sum therefore cancels
//! by a factor `exp(2 Im z)`, and no rearrangement of `J + iY` avoids
//! it. `H2` is the mirror image, bad for `Im z < 0`. So:
//!
//! * `H1` is accurate for `Im z <= 0` and degrades above the real axis,
//! * `H2` is accurate for `Im z >= 0` and degrades below it,
//! * on the real axis both are fine, and `H2 = conj(H1)` exactly.
//!
//! This is the same shape of problem as `K` versus `I`, and it has the
//! same resolution in the literature: a genuinely different method
//! (uniform asymptotics, or the scaled Hankel routines of AMOS, which
//! return `exp(-iz) H1` and let the caller supply the exponential),
//! which is **not** implemented here.
//!
//! Switching routes does not help. `examples/hankel_accuracy.rs`
//! measures both and they agree to within a factor of 1.5 across the
//! plane, because at whole order the non-integer routine delegates `Y`
//! to the integer one and so shares the dominant ingredient. Above the
//! real axis the loss follows `1e-16 exp(3 Im z)` — the same law `K`
//! obeys on the real axis, and for the same reason: `H1_nu(iy)` and
//! `K_nu(y)` are the same computation up to a constant (DLMF 10.27.8).
//! `H1` is good to `|Im z| ~ 8` and gone by `|Im z| ~ 12.`

use crate::bessel_complex::{
    bessel_j_c, bessel_j_nu, bessel_y_c, bessel_y_nu,
};
use crate::complex::Complex64 as C;
use crate::sph_bessel::{sph_j, sph_j_prime, sph_y, sph_y_prime};

/// Assemble `J + s i Y` where `s` is `+1` for `H1` and `-1` for `H2`.
///
/// Refuses a non-finite result. Far above the real axis `J` and `Y` are
/// each about `exp(|Im z|)` while `H1` is about `exp(-Im z)`, so at
/// `Im z = 700` the ingredients are at the top of `f64` and their sum
/// can come back `inf - inf`. Returning `NaN` from that is worse than
/// returning nothing: use [`crate::bessel_scaled::hankel_h1_scaled_nu`],
/// which never forms the large factor.
fn combine(j: C, y: C, s: f64, what: &str, z: C) -> Result<C, String> {
    let v = j + C::I * y * s;
    if v.is_finite() {
        Ok(v)
    } else {
        Err(format!(
            "{what}: J and Y are about exp(|Im z|) at z = {z:?} and their combination \
             left f64 range. The scaled form carries this without forming the \
             exponential."
        ))
    }
}

/// `H1_n(z) = J_n(z) + i Y_n(z)` for **whole** order `n >= 0`
/// (DLMF 10.4.3, <https://dlmf.nist.gov/10.4.E3>).
///
/// Uses the integer-order routines, so `J` comes from Miller recurrence
/// and `Y` from the logarithmic series. **Accurate for `Im z <= 0`**;
/// above the real axis see the module note on cancellation.
///
/// # Errors
/// A negative order, `z = 0` (where `Y` is singular), or a non-finite
/// argument.
///
/// # Examples
/// ```
/// use special_functions::hankel::{hankel_h1_c, hankel_h2_c};
/// use special_functions::complex::Complex64 as C;
/// // On the real axis H1 and H2 are complex conjugates, so |H1|^2 is
/// // J^2 + Y^2 — the combination that appears in scattering cross
/// // sections, and the reason H1 rather than J or Y is the natural
/// // object there.
/// let h1 = hankel_h1_c(0, C::real(3.0)).unwrap();
/// let h2 = hankel_h2_c(0, C::real(3.0)).unwrap();
/// assert!((h2 - h1.conj()).abs() < 1e-15);
/// assert!(h1.im > 0.0, "Im H1_0(3) is Y_0(3), which is positive");
/// ```
pub fn hankel_h1_c(n: i32, z: C) -> Result<C, String> {
    combine(bessel_j_c(n, z)?, bessel_y_c(n, z)?, 1.0, "hankel_h1_c", z)
}

/// `H2_n(z) = J_n(z) - i Y_n(z)` for whole order `n >= 0`.
///
/// **Accurate for `Im z >= 0`** — the mirror image of [`hankel_h1_c`].
///
/// # Errors
/// As [`hankel_h1_c`].
pub fn hankel_h2_c(n: i32, z: C) -> Result<C, String> {
    combine(bessel_j_c(n, z)?, bessel_y_c(n, z)?, -1.0, "hankel_h2_c", z)
}

/// `H1_nu(z)` for any **real** order, integer or not.
///
/// Built from the ascending-series routines, so it inherits their range
/// (`|z|` up to about 15 on the real axis) as well as the `H1`
/// cancellation above the real axis. At a **whole** order it is not
/// meaningfully different from [`hankel_h1_c`] — measured, the two agree
/// to within a factor of 1.5 everywhere — because `bessel_y_nu` hands
/// whole orders to `bessel_y_c`, so both routes share the ingredient
/// that dominates the error. Use this one when the order is not whole;
/// there is nothing to gain by preferring it when it is.
///
/// # Errors
/// `z = 0`, or a non-finite order or argument.
///
/// # Examples
/// ```
/// use special_functions::hankel::hankel_h1_nu;
/// use special_functions::complex::Complex64 as C;
/// // H1_{1/2}(z) = -i sqrt(2/(pi z)) exp(i z), exactly, for complex z.
/// let z = C::new(2.0, -0.5);
/// let got = hankel_h1_nu(0.5, z).unwrap();
/// let want = C::I * -1.0
///     * (C::real(2.0 / std::f64::consts::PI) * z.inv()).powf(0.5)
///     * (C::I * z).exp();
/// assert!((got - want).abs() < 1e-12);
/// ```
pub fn hankel_h1_nu(nu: f64, z: C) -> Result<C, String> {
    combine(bessel_j_nu(nu, z)?, bessel_y_nu(nu, z)?, 1.0, "hankel_h1_nu", z)
}

/// `H2_nu(z)` for any real order.
///
/// # Errors
/// As [`hankel_h1_nu`].
pub fn hankel_h2_nu(nu: f64, z: C) -> Result<C, String> {
    combine(bessel_j_nu(nu, z)?, bessel_y_nu(nu, z)?, -1.0, "hankel_h2_nu", z)
}

/// The derivative of any cylinder function,
/// `C'_nu(z) = C_{nu-1}(z) - (nu/z) C_nu(z)`
/// (DLMF 10.6.2, <https://dlmf.nist.gov/10.6.E2>).
///
/// `at` supplies the function at a given order. Order 0 is special: it
/// would need `C_{-1}`, and the identity `C'_0 = -C_1` covers it
/// (the symmetric form `(C_{-1} - C_1)/2` gives the same thing, since
/// `C_{-1} = -C_1` for whole orders).
fn cylinder_prime(
    nu: f64,
    z: C,
    at: impl Fn(f64, C) -> Result<C, String>,
) -> Result<C, String> {
    if z.abs() == 0.0 {
        return Err("hankel: the derivative is singular at z = 0".to_string());
    }
    if nu == 0.0 {
        return Ok(at(1.0, z)? * -1.0);
    }
    Ok(at(nu - 1.0, z)? - at(nu, z)? * (z.inv() * nu))
}

/// `H1_n'(z)` for whole order `n >= 0`.
///
/// # Errors
/// As [`hankel_h1_c`].
pub fn hankel_h1_prime_c(n: i32, z: C) -> Result<C, String> {
    if n < 0 {
        return Err(format!("hankel_h1_prime_c: order must be >= 0, got {n}"));
    }
    cylinder_prime(n as f64, z, |v, w| hankel_h1_c(v as i32, w))
}

/// `H2_n'(z)` for whole order `n >= 0`.
///
/// # Errors
/// As [`hankel_h1_c`].
pub fn hankel_h2_prime_c(n: i32, z: C) -> Result<C, String> {
    if n < 0 {
        return Err(format!("hankel_h2_prime_c: order must be >= 0, got {n}"));
    }
    cylinder_prime(n as f64, z, |v, w| hankel_h2_c(v as i32, w))
}

/// `H1_nu'(z)` for any real order.
///
/// # Errors
/// As [`hankel_h1_nu`].
pub fn hankel_h1_prime_nu(nu: f64, z: C) -> Result<C, String> {
    cylinder_prime(nu, z, hankel_h1_nu)
}

/// `H2_nu'(z)` for any real order.
///
/// # Errors
/// As [`hankel_h1_nu`].
pub fn hankel_h2_prime_nu(nu: f64, z: C) -> Result<C, String> {
    cylinder_prime(nu, z, hankel_h2_nu)
}

// ---------------------------------------------------------------------
// Spherical
// ---------------------------------------------------------------------

/// `h1_n(x) = j_n(x) + i y_n(x)`, the outgoing spherical wave
/// (DLMF 10.47.5, <https://dlmf.nist.gov/10.47.E5>).
///
/// Real argument, complex result. `x h1_n(x) -> exp(i(x - (n+1)pi/2))`
/// as `x` grows, so `|x h1_n(x)| -> 1`: the amplitude falls exactly as
/// `1/r`, which is what "outgoing spherical wave" means.
///
/// # Errors
/// A negative order, or `x <= 0` where `y_n` is singular.
///
/// # Examples
/// ```
/// use special_functions::hankel::sph_hankel_h1;
/// use special_functions::complex::Complex64 as C;
/// // h1_0(x) = -i exp(ix)/x, exactly.
/// let x = 2.3_f64;
/// let got = sph_hankel_h1(0, x).unwrap();
/// let want = C::I * -1.0 * C::from_polar(1.0 / x, x);
/// assert!((got - want).abs() < 1e-14);
/// ```
pub fn sph_hankel_h1(n: i32, x: f64) -> Result<C, String> {
    Ok(C::new(sph_j(n, x)?, sph_y(n, x)?))
}

/// `h2_n(x) = j_n(x) - i y_n(x)`, the incoming spherical wave.
///
/// On the real axis this is exactly `conj(h1_n(x))`, and the test suite
/// checks that rather than assuming it.
///
/// # Errors
/// As [`sph_hankel_h1`].
pub fn sph_hankel_h2(n: i32, x: f64) -> Result<C, String> {
    Ok(C::new(sph_j(n, x)?, -sph_y(n, x)?))
}

/// `h1_n'(x)`.
///
/// # Errors
/// As [`sph_hankel_h1`].
pub fn sph_hankel_h1_prime(n: i32, x: f64) -> Result<C, String> {
    Ok(C::new(sph_j_prime(n, x)?, sph_y_prime(n, x)?))
}

/// `h2_n'(x)`.
///
/// # Errors
/// As [`sph_hankel_h1`].
pub fn sph_hankel_h2_prime(n: i32, x: f64) -> Result<C, String> {
    Ok(C::new(sph_j_prime(n, x)?, -sph_y_prime(n, x)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: C, b: C, tol: f64) -> bool {
        (a - b).abs() <= tol * (1.0_f64).max(a.abs().max(b.abs()))
    }

    /// `H1_{1/2}(z) = -i sqrt(2/(pi z)) exp(iz)` and
    /// `H2_{1/2}(z) = i sqrt(2/(pi z)) exp(-iz)`, exactly, for complex
    /// `z`. These follow from `Y_{1/2} = -J_{-1/2}` and the half-integer
    /// closed forms, and they involve no Bessel code on the right, so
    /// they are absolute checks.
    ///
    /// Each is tested only on the half plane where it is well
    /// conditioned — that restriction is the module's central claim, and
    /// `accuracy_is_a_half_plane_for_each` below is what tests the
    /// claim itself.
    #[test]
    fn half_integer_closed_forms() {
        let pref = |z: C| (C::real(2.0 / std::f64::consts::PI) * z.inv()).powf(0.5);
        for &(re, im) in &[(0.5, 0.0), (2.0, 0.0), (7.0, 0.0), (3.0, -1.5), (1.2, -3.0)] {
            let z = C::new(re, im);
            let got = hankel_h1_nu(0.5, z).unwrap();
            let want = C::I * -1.0 * pref(z) * (C::I * z).exp();
            assert!(close(got, want, 1e-11), "H1_1/2({z:?}): {got:?} vs {want:?}");
        }
        for &(re, im) in &[(0.5, 0.0), (2.0, 0.0), (7.0, 0.0), (3.0, 1.5), (1.2, 3.0)] {
            let z = C::new(re, im);
            let got = hankel_h2_nu(0.5, z).unwrap();
            let want = C::I * pref(z) * (C::I * z * -1.0).exp();
            assert!(close(got, want, 1e-11), "H2_1/2({z:?}): {got:?} vs {want:?}");
        }
    }

    /// The Hankel Wronskian
    /// `H1_nu H2_nu' - H1_nu' H2_nu = -4i/(pi z)` (DLMF 10.5.4).
    ///
    /// The right-hand side is elementary, so a consistently wrong pair
    /// cannot satisfy it. Kept on the real axis, where both members of
    /// the pair are well conditioned at once.
    #[test]
    fn the_hankel_wronskian_holds() {
        for &nu in &[0.0, 0.5, 1.0, 1.3, 2.0, 3.7] {
            for &x in &[0.4, 1.0, 3.0, 6.0, 11.0] {
                let z = C::real(x);
                let w = hankel_h1_nu(nu, z).unwrap() * hankel_h2_prime_nu(nu, z).unwrap()
                    - hankel_h1_prime_nu(nu, z).unwrap() * hankel_h2_nu(nu, z).unwrap();
                let want = C::I * -4.0 * z.inv() * (1.0 / std::f64::consts::PI);
                assert!(close(w, want, 1e-10), "Wronskian nu={nu} x={x}: {w:?} vs {want:?}");
            }
        }
    }

    /// For real order and real positive argument the two Hankel
    /// functions are complex conjugates, because `J` and `Y` are both
    /// real there. That also makes `|H1|^2 = J^2 + Y^2`, the combination
    /// that appears in scattering amplitudes.
    #[test]
    fn conjugate_on_the_real_axis() {
        for &nu in &[0.0, 0.5, 1.0, 2.4] {
            for &x in &[0.3, 1.0, 4.0, 9.0] {
                let z = C::real(x);
                let h1 = hankel_h1_nu(nu, z).unwrap();
                let h2 = hankel_h2_nu(nu, z).unwrap();
                assert!(close(h2, h1.conj(), 1e-13), "H2 != conj(H1) at nu={nu} x={x}");
                let j = bessel_j_nu(nu, z).unwrap().re;
                let y = bessel_y_nu(nu, z).unwrap().re;
                assert!(
                    (h1.norm_sqr() - (j * j + y * y)).abs() <= 1e-12 * (j * j + y * y),
                    "|H1|^2 != J^2 + Y^2 at nu={nu} x={x}"
                );
            }
        }
    }

    /// The travelling-wave asymptotic
    /// `H1_nu(x) ~ sqrt(2/(pi x)) exp(i(x - nu pi/2 - pi/4))`
    /// (DLMF 10.2.5). This is what makes `H1` "outgoing", so it is the
    /// property most worth testing, and the leading correction is
    /// `O(1/x)` — the assertion tightens with `x` rather than being a
    /// fixed loose tolerance.
    #[test]
    fn the_outgoing_asymptotic_is_approached() {
        for &nu in &[0.0, 0.5, 1.0, 2.0] {
            // Stops at 30 on purpose: past that the ANSWER stops being
            // trustworthy, because Y_n on the real axis comes from an
            // ascending series and loses |z|/ln10 digits (at x = 40 it
            // is wrong in the first digit). That limit was mis-stated
            // until this test found it — see `bessel_complex`.
            // 30 was here until Stage 2J. `bessel_y_nu`'s reflection
            // now refuses it — at nu = 1/2, z = 30 the ascending series
            // has spent 1.1e-3 of its precision, and the module already
            // documented Y as "unusable past 30". The guard agreeing
            // with the documentation is the point; the test moves in
            // rather than the threshold moving out.
            for &x in &[10.0_f64, 16.0, 22.0] {
                // `hankel_h1_nu`, not `hankel_h1_c(nu as i32, ..)` —
                // the first draft wrote the latter and silently tested
                // H1_0 against the nu = 1/2 asymptotic, which is the
                // very truncation this project refuses to do elsewhere.
                let h = hankel_h1_nu(nu, C::real(x)).unwrap();
                let want = C::from_polar(
                    (2.0 / (std::f64::consts::PI * x)).sqrt(),
                    x - nu * std::f64::consts::FRAC_PI_2 - std::f64::consts::FRAC_PI_4,
                );
                // The first correction term is (4 nu^2 - 1)/(8x), so the
                // bound must scale as 1/x or it tests nothing at x = 80.
                let tol = 2.0 * (4.0 * nu * nu + 1.0) / (8.0 * x);
                let e = (h - want).abs() / want.abs();
                assert!(e <= tol, "H1_{nu}({x}): relative gap {e:.2e} exceeds {tol:.2e}");
            }
        }
    }

    /// Whole order through the general routine must agree with the
    /// integer routine, which is a different algorithm underneath
    /// (Miller recurrence and the logarithmic series, versus one
    /// ascending series evaluated at plus and minus nu).
    #[test]
    fn the_two_routes_agree_at_whole_order() {
        for n in 0..5 {
            for &(re, im) in &[(0.9, 0.0), (3.0, 0.0), (6.0, 0.0), (2.0, -1.0)] {
                let z = C::new(re, im);
                assert!(
                    close(hankel_h1_nu(n as f64, z).unwrap(), hankel_h1_c(n, z).unwrap(), 1e-10),
                    "H1_{n}({z:?})"
                );
                assert!(
                    close(hankel_h2_nu(n as f64, z).unwrap(), hankel_h2_c(n, z).unwrap(), 1e-10),
                    "H2_{n}({z:?})"
                );
                assert!(
                    close(
                        hankel_h1_prime_nu(n as f64, z).unwrap(),
                        hankel_h1_prime_c(n, z).unwrap(),
                        1e-10
                    ),
                    "H1'_{n}({z:?})"
                );
            }
        }
    }

    /// Any cylinder function satisfies `C_{nu-1} + C_{nu+1} =
    /// (2 nu/z) C_nu`. `H1` and `H2` are cylinder functions, so this
    /// must hold for them — a check that the linear combination was
    /// formed correctly rather than, say, with the sign flipped.
    #[test]
    fn hankel_obeys_the_cylinder_recurrence() {
        for &nu in &[0.4, 1.0, 2.6, 4.0] {
            for &(re, im) in &[(1.0, 0.0), (5.0, 0.0), (2.0, -1.2)] {
                let z = C::new(re, im);
                let f = z.inv() * (2.0 * nu);
                for (name, h) in [
                    ("H1", hankel_h1_nu as fn(f64, C) -> Result<C, String>),
                    ("H2", hankel_h2_nu as fn(f64, C) -> Result<C, String>),
                ] {
                    let lhs = h(nu - 1.0, z).unwrap() + h(nu + 1.0, z).unwrap();
                    assert!(
                        close(lhs, h(nu, z).unwrap() * f, 1e-9),
                        "{name} recurrence at nu={nu} z={z:?}"
                    );
                }
            }
        }
    }

    /// The module's central accuracy claim, stated as a test rather than
    /// as prose: `H1` survives in the LOWER half plane and `H2` in the
    /// upper, and the failure is not symmetric between them by accident
    /// — it is the same `exp(2|Im z|)` cancellation seen from two sides.
    ///
    /// Both are measured against the exact half-integer closed form.
    #[test]
    fn accuracy_is_a_half_plane_for_each() {
        let pref = |z: C| (C::real(2.0 / std::f64::consts::PI) * z.inv()).powf(0.5);
        let h1_err = |z: C| {
            let want = C::I * -1.0 * pref(z) * (C::I * z).exp();
            (hankel_h1_nu(0.5, z).unwrap() - want).abs() / want.abs()
        };
        let h2_err = |z: C| {
            let want = C::I * pref(z) * (C::I * z * -1.0).exp();
            (hankel_h2_nu(0.5, z).unwrap() - want).abs() / want.abs()
        };

        // The good half planes, out to where the underlying series
        // itself gives out.
        for &y in &[0.0, 1.0, 3.0, 6.0] {
            let z = C::new(4.0, -y);
            assert!(h1_err(z) < 1e-12, "H1 should be sound below the axis at {z:?}");
            let z = C::new(4.0, y);
            assert!(h2_err(z) < 1e-12, "H2 should be sound above the axis at {z:?}");
        }

        // ... and the bad ones. This direction is the point: if a future
        // change made the naive construction accurate everywhere, that
        // would be a real improvement and this assertion should fail
        // loudly rather than pass silently.
        let bad = C::from_polar(16.0, 1.0);
        assert!(
            h1_err(bad) > 1e-7,
            "H1 is expected to have lost most of its digits at {bad:?}, but did not"
        );
        assert!(
            h2_err(bad.conj()) > 1e-7,
            "H2 is expected to fail below the axis, mirroring H1"
        );
    }

    // ---- spherical ----------------------------------------------------

    /// `h1_0(x) = -i exp(ix)/x` and `h1_1(x) = -(1 + i/x) exp(ix)/x`,
    /// exactly (DLMF 10.49.6-7). Elementary right-hand sides again.
    #[test]
    fn spherical_closed_forms() {
        for &x in &[0.4_f64, 1.0, 3.0, 7.5, 20.0] {
            let e = C::from_polar(1.0, x);
            let got = sph_hankel_h1(0, x).unwrap();
            let want = C::I * -1.0 * e * (1.0 / x);
            assert!(close(got, want, 1e-13), "h1_0({x}): {got:?} vs {want:?}");

            let got = sph_hankel_h1(1, x).unwrap();
            let want = (C::ONE + C::I * (1.0 / x)) * e * (-1.0 / x);
            assert!(close(got, want, 1e-13), "h1_1({x}): {got:?} vs {want:?}");

            // h2 is the conjugate for real x, so both are covered at once.
            assert!(
                close(sph_hankel_h2(1, x).unwrap(), sph_hankel_h1(1, x).unwrap().conj(), 1e-14),
                "h2_1 != conj(h1_1) at {x}"
            );
        }
    }

    /// The spherical Hankel Wronskian
    /// `h1_n h2_n' - h1_n' h2_n = -2i/x^2` (from `j_n y_n' - j_n' y_n =
    /// 1/x^2`, DLMF 10.50.1).
    #[test]
    fn the_spherical_wronskian_holds() {
        for n in 0..6 {
            for &x in &[0.5, 1.0, 4.0, 12.0] {
                let w = sph_hankel_h1(n, x).unwrap() * sph_hankel_h2_prime(n, x).unwrap()
                    - sph_hankel_h1_prime(n, x).unwrap() * sph_hankel_h2(n, x).unwrap();
                let want = C::I * (-2.0 / (x * x));
                assert!(close(w, want, 1e-11), "spherical Wronskian n={n} x={x}");
            }
        }
    }

    /// "Outgoing spherical wave" is a statement about amplitude:
    /// `|x h1_n(x)| -> 1`. Nothing about the implementation forces this,
    /// so it is a real end-to-end check of both `j_n` and `y_n` at once.
    #[test]
    fn the_spherical_amplitude_falls_exactly_as_one_over_r() {
        // n = 0 is not asymptotic at all: h1_0 = -i exp(ix)/x, so
        // |x h1_0(x)| is 1 to the last bit at every x. Asserting a
        // decreasing gap there would be asserting on rounding noise,
        // which is what the first version of this test did.
        for &x in &[50.0_f64, 200.0, 800.0] {
            let a = (sph_hankel_h1(0, x).unwrap() * x).abs();
            assert!((a - 1.0).abs() < 1e-15, "|x h1_0({x})| = {a}, should be exactly 1");
        }
        // For n >= 1 the gap is n(n+1)/(4x^2) — measured, and asserted
        // within a factor of two so it tests the LAW and not one point.
        for n in 1..5 {
            for &x in &[50.0_f64, 200.0, 800.0] {
                let gap = (sph_hankel_h1(n, x).unwrap() * x).abs() - 1.0;
                let want = (n * (n + 1)) as f64 / (4.0 * x * x);
                assert!(
                    gap > 0.5 * want && gap < 2.0 * want,
                    "|x h1_{n}({x})| - 1 = {gap:.3e}, expected about {want:.3e}"
                );
            }
        }
    }

    #[test]
    fn hankel_edge_cases() {
        assert!(hankel_h1_c(-1, C::ONE).is_err(), "negative whole order");
        assert!(hankel_h1_c(0, C::ZERO).is_err(), "z = 0");
        assert!(hankel_h2_c(0, C::ZERO).is_err(), "z = 0");
        assert!(hankel_h1_nu(0.5, C::ZERO).is_err(), "z = 0");
        assert!(hankel_h1_prime_nu(0.5, C::ZERO).is_err(), "derivative at z = 0");
        assert!(hankel_h1_prime_c(-1, C::ONE).is_err(), "negative order");
        assert!(hankel_h2_prime_c(-1, C::ONE).is_err(), "negative order");
        assert!(sph_hankel_h1(-1, 1.0).is_err(), "negative order");
        assert!(sph_hankel_h1(0, 0.0).is_err(), "y_n is singular at x = 0");
        assert!(sph_hankel_h2(0, -1.0).is_err(), "negative argument");
        assert!(sph_hankel_h1_prime(0, 0.0).is_err(), "singular");
        assert!(sph_hankel_h2_prime(0, 0.0).is_err(), "singular");
    }
}
