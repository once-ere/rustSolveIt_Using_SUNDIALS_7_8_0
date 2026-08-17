//! Integer-order Bessel functions of the first kind, `J_n(x)`, with an
//! efficient whole-table routine.
//!
//! A Bessel-expanded split-operator propagator needs the *whole* set
//! `J_0(lambda) … J_N(lambda)` at a single argument, once per run.
//! Computing them one at a time wastes the structure; the downward
//! recurrence produces all of them in one pass.
//!
//! # Provenance
//!
//! **Clean-room.** Implemented from the mathematics:
//!
//! * the three-term recurrence `J_{n-1}(x) + J_{n+1}(x) = (2n/x) J_n(x)`
//!   (`DLMF 10.6.1` <https://dlmf.nist.gov/10.6.E1>; A&S 9.1.27);
//! * **Miller's algorithm** — recur *downward* from an artificial seed
//!   above the wanted order, because `J_n` decays once `n > x` and the
//!   downward direction is the stable one;
//! * fixing the overall scale with the normalisation identity
//!   `J_0(x) + 2 [J_2(x) + J_4(x) + …] = 1`
//!   (`DLMF 10.12.4` <https://dlmf.nist.gov/10.12.E4>; A&S 9.1.46).
//!
//! No rational-approximation coefficient tables are used, and no
//! third-party implementation was consulted. The normalisation-sum
//! approach needs no magic constants at all — which is precisely why it
//! is the right choice for a clean-room replacement.
//!
//! Cross-checked in the test suite against the independently written
//! vendored Cephes `jv`, which uses entirely different machinery.

/// `J_0(x) … J_{n_max}(x)` in one pass.
///
/// Returns a vector of length `n_max + 1`.
///
/// # Errors
/// Non-finite `x`.
///
/// # Examples
/// ```
/// use special_functions::bessel::bessel_j_array;
/// let j = bessel_j_array(6, 2.5).unwrap();
/// assert_eq!(j.len(), 7);
/// // J_0(0) = 1 and every higher order vanishes at the origin
/// let z = bessel_j_array(4, 0.0).unwrap();
/// assert_eq!(z[0], 1.0);
/// assert!(z[1..].iter().all(|&v| v == 0.0));
/// ```
pub fn bessel_j_array(n_max: usize, x: f64) -> Result<Vec<f64>, String> {
    if !x.is_finite() {
        return Err(format!("bessel_j_array: x must be finite, got {x}"));
    }
    // J_n(-x) = (-1)^n J_n(x)  (A&S 9.1.35)
    if x < 0.0 {
        let mut v = bessel_j_array(n_max, -x)?;
        for (n, e) in v.iter_mut().enumerate() {
            if n % 2 == 1 {
                *e = -*e;
            }
        }
        return Ok(v);
    }
    if x == 0.0 {
        let mut v = vec![0.0; n_max + 1];
        v[0] = 1.0;
        return Ok(v);
    }

    // Start the downward recurrence comfortably above both the wanted
    // order and the argument, so the unwanted (growing) solution has
    // decayed away by the time we reach n_max.
    // The seed order must sit well ABOVE both n_max and x: J_n(x) only
    // begins its rapid decay once n exceeds x, so a start merely a
    // little above x leaves the recurrence under-converged. (Measured:
    // start = n_max + 20 + 0.5x gave only ~9 correct digits at x = 45.)
    let start = n_max + 30 + (1.5 * x + 12.0 * x.sqrt()) as usize;

    let mut jp1 = 0.0_f64; // J_{k+1}
    let mut j = 1.0e-290_f64; // J_k, arbitrary seed
    let mut out = vec![0.0_f64; n_max + 1];
    // Accumulates J_0 + 2(J_2 + J_4 + ...) in the *unnormalised* scale.
    let mut sum = 0.0_f64;

    for k in (0..=start).rev() {
        // J_{k-1} from J_k and J_{k+1}; here `j` holds J_{k+1} after the
        // shift, so compute the value at index k.
        let jm1 = (2 * (k + 1)) as f64 / x * j - jp1;
        jp1 = j;
        j = jm1;

        // Rescale if the recurrence is about to overflow.
        if j.abs() > 1.0e250 {
            let s = 1.0e-250;
            j *= s;
            jp1 *= s;
            sum *= s;
            for e in out.iter_mut() {
                *e *= s;
            }
        }
        if k <= n_max {
            out[k] = j;
        }
        // Even orders (excluding 0) enter the identity with weight 2.
        if k % 2 == 0 {
            sum += if k == 0 { j } else { 2.0 * j };
        }
    }

    if sum == 0.0 || !sum.is_finite() {
        return Err(format!(
            "bessel_j_array: normalisation failed for n_max={n_max}, x={x}"
        ));
    }
    let scale = 1.0 / sum;
    for e in out.iter_mut() {
        *e *= scale;
    }
    Ok(out)
}

/// A single `J_n(x)` for integer `n >= 0`.
///
/// Prefer [`bessel_j_array`] when a whole table is needed — this
/// computes one via the same pass and discards the rest.
///
/// # Errors
/// Negative order, or non-finite `x`.
///
/// # Examples
/// ```
/// use special_functions::bessel::bessel_j;
/// // J_1 is odd, J_0 is even
/// assert!((bessel_j(1, -1.4).unwrap() + bessel_j(1, 1.4).unwrap()).abs() < 1e-14);
/// assert!((bessel_j(0, -1.4).unwrap() - bessel_j(0, 1.4).unwrap()).abs() < 1e-14);
/// ```
pub fn bessel_j(n: i32, x: f64) -> Result<f64, String> {
    if n < 0 {
        return Err(format!("bessel_j: order n must be >= 0, got {n}"));
    }
    Ok(bessel_j_array(n as usize, x)?[n as usize])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rel_err;
    use spec_math::cephes64::jv;

    /// Cross-validation against the vendored Cephes implementation,
    /// which is written on completely different principles. Two
    /// independent routes agreeing is the strongest evidence available.
    #[test]
    fn agrees_with_the_vendored_cephes_implementation() {
        for &x in &[0.1, 0.7, 1.0, 2.5, 5.0, 9.0, 20.0, 45.0] {
            let ours = bessel_j_array(15, x).unwrap();
            for (n, &v) in ours.iter().enumerate() {
                let theirs = jv(n as f64, x);
                // Loosen only where both are denormal-small (n >> x).
                let tol = if theirs.abs() < 1e-14 { 1e-6 } else { 1e-10 };
                assert!(
                    rel_err(v, theirs) < tol || (v - theirs).abs() < 1e-16,
                    "J_{n}({x}): ours={v:e} cephes={theirs:e}"
                );
            }
        }
    }

    /// The identity used to fix the scale must hold for the *output*,
    /// which is a genuine check rather than a tautology: it is imposed
    /// on the unnormalised values, so its survival after scaling
    /// confirms the scaling itself.
    #[test]
    fn normalisation_identity_holds() {
        for &x in &[0.3, 1.5, 4.0, 12.0] {
            let j = bessel_j_array(60, x).unwrap();
            let mut s = j[0];
            let mut n = 2;
            while n < j.len() {
                s += 2.0 * j[n];
                n += 2;
            }
            assert!((s - 1.0).abs() < 1e-12, "sum at x={x} was {s}");
        }
    }

    #[test]
    fn three_term_recurrence_is_satisfied() {
        for &x in &[0.6, 2.0, 7.0, 18.0] {
            let j = bessel_j_array(25, x).unwrap();
            for n in 1..24 {
                let lhs = j[n - 1] + j[n + 1];
                let rhs = 2.0 * n as f64 / x * j[n];
                assert!((lhs - rhs).abs() < 1e-12, "recurrence n={n} x={x}");
            }
        }
    }

    #[test]
    fn known_values_and_parity() {
        // J_0(0) = 1, J_n(0) = 0
        let z = bessel_j_array(5, 0.0).unwrap();
        assert_eq!(z[0], 1.0);
        assert!(z[1..].iter().all(|&v| v == 0.0));
        // parity J_n(-x) = (-1)^n J_n(x)
        let p = bessel_j_array(6, 3.3).unwrap();
        let m = bessel_j_array(6, -3.3).unwrap();
        for n in 0..=6 {
            let want = if n % 2 == 0 { p[n] } else { -p[n] };
            assert!(rel_err(m[n], want) < 1e-13, "parity n={n}");
        }
        // J_0 has its first zero near 2.404825557695773
        assert!(bessel_j(0, 2.404_825_557_695_773).unwrap().abs() < 1e-12);
    }

    #[test]
    fn decays_hard_once_order_exceeds_argument() {
        // J_30(1) ~ 1e-49 — the regime where upward recurrence fails
        let v = bessel_j(30, 1.0).unwrap();
        assert!(v > 0.0 && v < 1e-40, "J_30(1) = {v:e}");
    }

    #[test]
    fn invalid_input_is_reported() {
        assert!(bessel_j(-1, 1.0).is_err());
        assert!(bessel_j(0, f64::NAN).is_err());
        assert!(bessel_j_array(3, f64::INFINITY).is_err());
    }
}
