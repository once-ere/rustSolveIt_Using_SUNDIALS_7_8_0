//! Spherical Bessel functions of the first and second kind.
//!
//! These solve the radial part of the Helmholtz equation in spherical
//! coordinates, so they appear everywhere in scattering, multipole
//! expansions and the quantum free particle. The vendored Cephes
//! translation has no spherical variants, so they are implemented here.
//!
//! Relation to the cylindrical functions (`DLMF 10.47.3`,
//! <https://dlmf.nist.gov/10.47.E3>):
//! `j_n(x) = sqrt(pi/2x) * J_{n+1/2}(x)` and
//! `y_n(x) = sqrt(pi/2x) * Y_{n+1/2}(x)`.
//!
//! # Numerical strategy — and why it matters
//!
//! Both functions satisfy the same three-term recurrence
//! (`DLMF 10.51.1`, <https://dlmf.nist.gov/10.51.E1>):
//!
//! ```text
//!     f_{n+1}(x) = ((2n+1)/x) * f_n(x) - f_{n-1}(x)
//! ```
//!
//! but its stability differs completely between them:
//!
//! * `y_n` **grows** with `n`, so upward recurrence is stable and is
//!   what we use.
//! * `j_n` **decays** rapidly once `n > x`. Recurring upward there
//!   amplifies the rounding error in the seed until the result is pure
//!   noise — for `n = 20, x = 1` the naive upward answer is wrong by
//!   many orders of magnitude. We therefore use **Miller's algorithm**:
//!   start from an artificial seed far above the wanted order, recur
//!   *downward* (the direction in which the wanted solution dominates),
//!   then fix the scale using the exactly known `j_0(x) = sin(x)/x`.
//!
//! The `sph_j_upward_recurrence_is_unstable_for_n_gt_x` test pins this
//! behaviour so nobody "simplifies" the implementation later.

/// Order above which the small-argument series is preferred over the
/// closed forms; chosen so `x^n / (2n+1)!!` cannot overflow the ratio.
const SMALL_X: f64 = 1.0e-4;

/// Spherical Bessel function of the first kind, `j_n(x)`.
///
/// `DLMF 10.47.3` <https://dlmf.nist.gov/10.47.E3>; A&S 10.1.1.
///
/// Special values: `j_0(0) = 1`, and `j_n(0) = 0` for `n >= 1`.
/// For integer order the parity relation `j_n(-x) = (-1)^n j_n(x)`
/// (A&S 10.1.7) extends the function to negative arguments.
///
/// # Errors
/// Returns `Err` for negative order or non-finite argument.
///
/// # Examples
/// ```
/// use special_functions::sph_bessel::sph_j;
/// // j_0(x) = sin(x)/x
/// let x = 1.3_f64;
/// assert!((sph_j(0, x).unwrap() - x.sin() / x).abs() < 1e-15);
/// // and j_n decays hard once n exceeds x
/// assert!(sph_j(20, 1.0).unwrap().abs() < 1e-25);
/// ```
pub fn sph_j(n: i32, x: f64) -> Result<f64, String> {
    if n < 0 {
        return Err(format!("sph_j: order n must be >= 0, got {n}"));
    }
    if !x.is_finite() {
        return Err(format!("sph_j: argument x must be finite, got {x}"));
    }
    // Parity: j_n(-x) = (-1)^n j_n(x)   (A&S 10.1.7)
    if x < 0.0 {
        let v = sph_j(n, -x)?;
        return Ok(if n % 2 == 0 { v } else { -v });
    }
    if x == 0.0 {
        return Ok(if n == 0 { 1.0 } else { 0.0 });
    }

    // Small argument: the leading series term, j_n(x) ~ x^n/(2n+1)!!
    // (A&S 10.1.2). Using the closed forms here would cancel badly.
    if x < SMALL_X {
        let mut t = 1.0_f64;
        for k in 1..=n {
            t *= x / (2 * k + 1) as f64;
        }
        // second-order correction keeps this accurate to ~x^2
        return Ok(t * (1.0 - x * x / (2.0 * (2 * n + 3) as f64)));
    }

    if n == 0 {
        return Ok(x.sin() / x);
    }
    if n == 1 {
        return Ok(x.sin() / (x * x) - x.cos() / x);
    }

    // Upward recurrence is stable only while the solution is growing,
    // i.e. roughly x > n. Otherwise fall through to Miller.
    if x > n as f64 {
        let mut fm1 = x.sin() / x; // j_0
        let mut f = x.sin() / (x * x) - x.cos() / x; // j_1
        for k in 1..n {
            let fp1 = (2 * k + 1) as f64 / x * f - fm1;
            fm1 = f;
            f = fp1;
        }
        return Ok(f);
    }

    // ---- Miller's algorithm: downward recurrence + normalisation ----
    // Start well above the wanted order so the unwanted (growing)
    // solution has died away by the time we reach n.
    let start = n + 20 + (10.0 * (n as f64).sqrt()) as i32;
    let mut fp1 = 0.0_f64; // f_{k+1}
    let mut f = 1.0e-280_f64; // f_k, arbitrary seed
    let mut wanted = 0.0_f64;
    for k in (0..=start).rev() {
        let fm1 = (2 * k + 3) as f64 / x * f - fp1; // gives f_k from f_{k+1}, f_{k+2}
        fp1 = f;
        f = fm1;
        if k == n {
            wanted = f;
        }
        // Rescale to keep the recurrence inside f64 range.
        if f.abs() > 1.0e250 {
            f *= 1.0e-250;
            fp1 *= 1.0e-250;
            wanted *= 1.0e-250;
        }
    }
    // `f` now holds the unnormalised j_0; the true value fixes the scale.
    let true_j0 = x.sin() / x;
    if f == 0.0 || !f.is_finite() {
        return Err(format!(
            "sph_j: downward recurrence failed to normalise for n={n}, x={x}"
        ));
    }
    Ok(wanted * (true_j0 / f))
}

/// Spherical Bessel function of the second kind, `y_n(x)`
/// (the spherical Neumann function).
///
/// `DLMF 10.47.4` <https://dlmf.nist.gov/10.47.E4>; A&S 10.1.1.
///
/// Singular at the origin: `y_n(x) -> -inf` as `x -> 0+`.
///
/// # Errors
/// Returns `Err` for negative order, non-finite argument, or `x <= 0`
/// (the function is real-valued only for positive argument).
///
/// # Examples
/// ```
/// use special_functions::sph_bessel::sph_y;
/// // y_0(x) = -cos(x)/x
/// let x = 2.1_f64;
/// assert!((sph_y(0, x).unwrap() + x.cos() / x).abs() < 1e-15);
/// ```
pub fn sph_y(n: i32, x: f64) -> Result<f64, String> {
    if n < 0 {
        return Err(format!("sph_y: order n must be >= 0, got {n}"));
    }
    if !x.is_finite() {
        return Err(format!("sph_y: argument x must be finite, got {x}"));
    }
    if x <= 0.0 {
        return Err(format!(
            "sph_y: argument x must be > 0 (y_n is singular at 0), got {x}"
        ));
    }
    let y0 = -x.cos() / x;
    if n == 0 {
        return Ok(y0);
    }
    let y1 = -x.cos() / (x * x) - x.sin() / x;
    if n == 1 {
        return Ok(y1);
    }
    // y_n grows with n: upward recurrence is the stable direction.
    let mut fm1 = y0;
    let mut f = y1;
    for k in 1..n {
        let fp1 = (2 * k + 1) as f64 / x * f - fm1;
        fm1 = f;
        f = fp1;
    }
    Ok(f)
}

/// Derivative of `j_n`, from `DLMF 10.51.2`
/// <https://dlmf.nist.gov/10.51.E2>: `j_n'(x) = j_{n-1}(x) - ((n+1)/x) j_n(x)`.
///
/// # Examples
/// ```
/// use special_functions::sph_bessel::sph_j_prime;
/// // j_0'(x) = (x cos x - sin x)/x^2
/// let x = 1.7_f64;
/// let expect = (x * x.cos() - x.sin()) / (x * x);
/// assert!((sph_j_prime(0, x).unwrap() - expect).abs() < 1e-14);
/// ```
pub fn sph_j_prime(n: i32, x: f64) -> Result<f64, String> {
    if n < 0 {
        return Err(format!("sph_j_prime: order n must be >= 0, got {n}"));
    }
    if x == 0.0 {
        // j_0'(0) = 0, j_1'(0) = 1/3, j_n'(0) = 0 for n >= 2
        return Ok(match n {
            1 => 1.0 / 3.0,
            _ => 0.0,
        });
    }
    if n == 0 {
        // j_0' = -j_1
        return Ok(-sph_j(1, x)?);
    }
    Ok(sph_j(n - 1, x)? - (n + 1) as f64 / x * sph_j(n, x)?)
}

/// Derivative of `y_n`, same recurrence as `j_n'`
/// (`DLMF 10.51.2` <https://dlmf.nist.gov/10.51.E2>).
pub fn sph_y_prime(n: i32, x: f64) -> Result<f64, String> {
    if n < 0 {
        return Err(format!("sph_y_prime: order n must be >= 0, got {n}"));
    }
    if n == 0 {
        return Ok(-sph_y(1, x)?);
    }
    Ok(sph_y(n - 1, x)? - (n + 1) as f64 / x * sph_y(n, x)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rel_err;

    // ---- 1. closed forms (A&S 10.1.11-10.1.12) -------------------------
    #[test]
    fn closed_forms_for_low_orders() {
        for &x in &[0.3, 1.0, 2.5, 7.0, 30.0] {
            assert!(rel_err(sph_j(0, x).unwrap(), x.sin() / x) < 1e-14);
            assert!(rel_err(sph_j(1, x).unwrap(), x.sin() / (x * x) - x.cos() / x) < 1e-13);
            let j2 = (3.0 / (x * x * x) - 1.0 / x) * x.sin() - 3.0 / (x * x) * x.cos();
            assert!(rel_err(sph_j(2, x).unwrap(), j2) < 1e-12, "j2 at x={x}");
            assert!(rel_err(sph_y(0, x).unwrap(), -x.cos() / x) < 1e-14);
            assert!(rel_err(sph_y(1, x).unwrap(), -x.cos() / (x * x) - x.sin() / x) < 1e-13);
        }
    }

    // ---- 2. the Wronskian, the strongest single check ------------------
    // A&S 10.1.31:  j_n(x) y_n'(x) - j_n'(x) y_n(x) = 1/x^2
    #[test]
    fn wronskian_holds_across_orders_and_arguments() {
        for n in 0..12 {
            for &x in &[0.5, 1.0, 3.0, 8.0, 25.0] {
                let w = sph_j(n, x).unwrap() * sph_y_prime(n, x).unwrap()
                    - sph_j_prime(n, x).unwrap() * sph_y(n, x).unwrap();
                assert!(
                    rel_err(w, 1.0 / (x * x)) < 1e-9,
                    "Wronskian failed n={n} x={x}: got {w}, want {}",
                    1.0 / (x * x)
                );
            }
        }
    }

    // ---- 3. the recurrence both functions must satisfy -----------------
    #[test]
    fn three_term_recurrence_is_satisfied() {
        for &x in &[0.7, 2.0, 6.0, 15.0] {
            for n in 1..15 {
                let lhs = sph_j(n + 1, x).unwrap();
                let rhs = (2 * n + 1) as f64 / x * sph_j(n, x).unwrap() - sph_j(n - 1, x).unwrap();
                assert!(rel_err(lhs, rhs) < 1e-8, "j recurrence n={n} x={x}");
                let lhs = sph_y(n + 1, x).unwrap();
                let rhs = (2 * n + 1) as f64 / x * sph_y(n, x).unwrap() - sph_y(n - 1, x).unwrap();
                assert!(rel_err(lhs, rhs) < 1e-8, "y recurrence n={n} x={x}");
            }
        }
    }

    // ---- 4. the stability property Miller's algorithm exists to fix ----
    #[test]
    fn sph_j_upward_recurrence_is_unstable_for_n_gt_x() {
        // Naive upward recurrence, the implementation we must NOT use.
        fn naive_upward(n: i32, x: f64) -> f64 {
            let mut fm1 = x.sin() / x;
            let mut f = x.sin() / (x * x) - x.cos() / x;
            for k in 1..n {
                let fp1 = (2 * k + 1) as f64 / x * f - fm1;
                fm1 = f;
                f = fp1;
            }
            f
        }
        let (n, x) = (20, 1.0);
        let good = sph_j(n, x).unwrap();
        let bad = naive_upward(n, x);
        // True j_20(1) ~ 7.6e-26: it must be tiny and positive.
        assert!(good > 0.0 && good < 1e-24, "j_20(1) = {good}");
        // The naive result is garbage by many orders of magnitude —
        // this is exactly why the implementation uses Miller downward.
        assert!(
            (bad / good).abs() > 1e6,
            "upward recurrence was expected to blow up; got {bad} vs {good}"
        );
    }

    // ---- 5. small-argument limit, A&S 10.1.2 --------------------------
    #[test]
    fn small_argument_series_matches_leading_term() {
        let x: f64 = 1e-6;
        // j_n(x) -> x^n / (2n+1)!!
        let mut dfact = 1.0_f64; // (2n+1)!!
        for n in 0..6 {
            dfact *= (2 * n + 1) as f64;
            let expect = x.powi(n) / dfact;
            assert!(
                rel_err(sph_j(n, x).unwrap(), expect) < 1e-10,
                "small-x j_{n}"
            );
        }
    }

    // ---- 6. parity and special values ---------------------------------
    #[test]
    fn parity_and_origin_values() {
        assert_eq!(sph_j(0, 0.0).unwrap(), 1.0);
        for n in 1..6 {
            assert_eq!(sph_j(n, 0.0).unwrap(), 0.0);
        }
        for n in 0..6 {
            let a = sph_j(n, -2.3).unwrap();
            let b = sph_j(n, 2.3).unwrap();
            let expect = if n % 2 == 0 { b } else { -b };
            assert!(rel_err(a, expect) < 1e-14, "parity n={n}");
        }
    }

    // ---- 7. domain errors are reported, never silently NaN ------------
    #[test]
    fn invalid_arguments_are_errors() {
        assert!(sph_j(-1, 1.0).is_err());
        assert!(sph_j(0, f64::NAN).is_err());
        assert!(sph_j(0, f64::INFINITY).is_err());
        assert!(sph_y(0, 0.0).is_err(), "y_n is singular at the origin");
        assert!(sph_y(0, -1.0).is_err());
        assert!(sph_y(-2, 1.0).is_err());
    }
}
