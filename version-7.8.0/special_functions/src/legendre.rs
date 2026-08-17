//! Legendre polynomials, associated Legendre functions and spherical
//! harmonics.
//!
//! These are the angular part of every central-potential problem: the
//! rigid rotor, the hydrogen atom, multipole expansions, and any
//! partial-wave analysis. `DLMF 14` <https://dlmf.nist.gov/14>.
//!
//! # Two families, on purpose
//!
//! * [`assoc_legendre_p`] returns the **unnormalised** `P_l^m(x)` with
//!   the Condon–Shortley phase, the convention of Abramowitz & Stegun.
//!   Convenient and directly comparable to textbook tables, but it
//!   overflows `f64` around `l ~ 150` because of the `(l+m)!` growth.
//! * [`norm_assoc_legendre_p`] returns the **fully normalised**
//!   `Pbar_l^m(x)` used by spherical harmonics, computed *directly* in
//!   the normalised basis. Every intermediate stays O(1), so it is
//!   stable to very high degree — which is the only way to evaluate
//!   `Y_l^m` for large `l` at all.
//!
//! Computing `Pbar` as `N(l,m) * P_l^m` would defeat the purpose: the
//! normalisation constant underflows exactly where `P_l^m` overflows.

use std::f64::consts::PI;

/// Legendre polynomial `P_n(x)` by Bonnet's recurrence,
/// `DLMF 14.10.3` <https://dlmf.nist.gov/14.10.E3>:
/// `(n+1) P_{n+1} = (2n+1) x P_n - n P_{n-1}`.
///
/// The recurrence is stable upward for `|x| <= 1`, which is the only
/// range where these are the physically relevant solutions.
///
/// # Errors
/// `n < 0`, or non-finite `x`.
///
/// # Examples
/// ```
/// use special_functions::legendre::legendre_p;
/// // P_n(1) = 1 for every n; P_n(-1) = (-1)^n
/// assert!((legendre_p(7, 1.0).unwrap() - 1.0).abs() < 1e-14);
/// assert!((legendre_p(7, -1.0).unwrap() + 1.0).abs() < 1e-14);
/// // P_2(x) = (3x^2 - 1)/2
/// let x = 0.37_f64;
/// assert!((legendre_p(2, x).unwrap() - (3.0*x*x - 1.0)/2.0).abs() < 1e-15);
/// ```
pub fn legendre_p(n: i32, x: f64) -> Result<f64, String> {
    if n < 0 {
        return Err(format!("legendre_p: degree n must be >= 0, got {n}"));
    }
    if !x.is_finite() {
        return Err(format!("legendre_p: x must be finite, got {x}"));
    }
    if n == 0 {
        return Ok(1.0);
    }
    let mut pm1 = 1.0_f64; // P_0
    let mut p = x; // P_1
    for k in 1..n {
        let pp1 = ((2 * k + 1) as f64 * x * p - k as f64 * pm1) / (k + 1) as f64;
        pm1 = p;
        p = pp1;
    }
    Ok(p)
}

/// Derivative `P_n'(x)`.
///
/// Uses `(x^2 - 1) P_n'(x) = n [x P_n(x) - P_{n-1}(x)]`
/// (`DLMF 14.10.5` <https://dlmf.nist.gov/14.10.E5>), with the
/// endpoints handled in closed form because that relation is
/// indeterminate there: `P_n'(1) = n(n+1)/2` and
/// `P_n'(-1) = (-1)^{n+1} n(n+1)/2`.
///
/// # Examples
/// ```
/// use special_functions::legendre::legendre_p_prime;
/// // P_2'(x) = 3x
/// assert!((legendre_p_prime(2, 0.4).unwrap() - 1.2).abs() < 1e-14);
/// // endpoint closed form, where the general relation is 0/0
/// assert!((legendre_p_prime(3, 1.0).unwrap() - 6.0).abs() < 1e-12);
/// ```
pub fn legendre_p_prime(n: i32, x: f64) -> Result<f64, String> {
    if n < 0 {
        return Err(format!("legendre_p_prime: degree n must be >= 0, got {n}"));
    }
    if !x.is_finite() {
        return Err(format!("legendre_p_prime: x must be finite, got {x}"));
    }
    if n == 0 {
        return Ok(0.0);
    }
    let nn = n as f64;
    if (x.abs() - 1.0).abs() < 1.0e-14 {
        let v = nn * (nn + 1.0) / 2.0;
        return Ok(if x > 0.0 {
            v
        } else if n % 2 == 0 {
            -v
        } else {
            v
        });
    }
    Ok(nn * (x * legendre_p(n, x)? - legendre_p(n - 1, x)?) / (x * x - 1.0))
}

/// Associated Legendre function `P_l^m(x)`, **unnormalised**, including
/// the Condon–Shortley phase `(-1)^m` (A&S 8.6.6 convention).
///
/// `DLMF 14.7.10` <https://dlmf.nist.gov/14.7.E10> for the recurrence.
///
/// Built from the closed-form seed
/// `P_m^m = (-1)^m (2m-1)!! (1-x^2)^{m/2}`, then
/// `P_{m+1}^m = x (2m+1) P_m^m`, then ascending in `l`:
/// `(l-m) P_l^m = x (2l-1) P_{l-1}^m - (l+m-1) P_{l-2}^m`.
///
/// # Overflow — driven by `m`, not by `l`
/// The seed alone carries `(2m-1)!!`, so large **order** is what breaks
/// `f64`, and the degree matters much less. Measured on this
/// implementation at `x = 0.3`: `P_100^50 ~ -3.4e97` and
/// `P_200^100 ~ -8.4e227` are still finite, but `P_170^170` already
/// overflows to infinity and `P_300^150` degenerates to `NaN`. Rather
/// than hand back such a value, this function returns `Err` and points
/// you at [`norm_assoc_legendre_p`], which stays O(1) for every one of
/// those cases.
///
/// # Errors
/// `l < 0`, `m` outside `[-l, l]`, `|x| > 1`, or a result too large to
/// represent (see above).
///
/// # Examples
/// ```
/// use special_functions::legendre::assoc_legendre_p;
/// let x = 0.5_f64;
/// let s = (1.0 - x*x).sqrt();
/// // P_1^1 = -(1-x^2)^{1/2}   (Condon-Shortley phase)
/// assert!((assoc_legendre_p(1, 1, x).unwrap() + s).abs() < 1e-14);
/// // P_2^2 = 3(1-x^2)
/// assert!((assoc_legendre_p(2, 2, x).unwrap() - 3.0*(1.0-x*x)).abs() < 1e-14);
/// // m = 0 reduces to the ordinary Legendre polynomial
/// assert!((assoc_legendre_p(3, 0, x).unwrap()
///          - special_functions::legendre::legendre_p(3, x).unwrap()).abs() < 1e-14);
/// ```
pub fn assoc_legendre_p(l: i32, m: i32, x: f64) -> Result<f64, String> {
    if l < 0 {
        return Err(format!("assoc_legendre_p: degree l must be >= 0, got {l}"));
    }
    if m.abs() > l {
        return Err(format!(
            "assoc_legendre_p: order m must satisfy |m| <= l, got l={l}, m={m}"
        ));
    }
    if !x.is_finite() || x.abs() > 1.0 {
        return Err(format!(
            "assoc_legendre_p: x must lie in [-1, 1], got {x}"
        ));
    }
    // Negative order via DLMF 14.9.3:
    //   P_l^{-m} = (-1)^m (l-m)!/(l+m)! P_l^m
    if m < 0 {
        let mm = -m;
        let mut ratio = 1.0_f64; // (l-mm)!/(l+mm)!
        for k in (l - mm + 1)..=(l + mm) {
            ratio /= k as f64;
        }
        let v = assoc_legendre_p(l, mm, x)?;
        return Ok(if mm % 2 == 0 { ratio * v } else { -ratio * v });
    }

    // Seed P_m^m = (-1)^m (2m-1)!! (1-x^2)^{m/2}
    let mut pmm = 1.0_f64;
    if m > 0 {
        let somx2 = (1.0 - x * x).max(0.0).sqrt();
        let mut fact = 1.0_f64;
        for _ in 0..m {
            pmm *= -fact * somx2;
            fact += 2.0;
        }
    }
    if l == m {
        return finite_or_overflow(pmm, l, m);
    }
    // P_{m+1}^m
    let mut pmmp1 = x * (2 * m + 1) as f64 * pmm;
    if l == m + 1 {
        return finite_or_overflow(pmmp1, l, m);
    }
    // Ascend in l
    let mut pll = 0.0_f64;
    for ll in (m + 2)..=l {
        pll = (x * (2 * ll - 1) as f64 * pmmp1 - (ll + m - 1) as f64 * pmm) / (ll - m) as f64;
        pmm = pmmp1;
        pmmp1 = pll;
    }
    finite_or_overflow(pll, l, m)
}

/// The unnormalised family can legitimately exceed `f64` (see the note
/// on [`assoc_legendre_p`]). Returning `inf`/`NaN` would violate this
/// project's "no silent NaN" rule, so overflow is reported as an error
/// that names the remedy.
fn finite_or_overflow(v: f64, l: i32, m: i32) -> Result<f64, String> {
    if v.is_finite() {
        Ok(v)
    } else {
        Err(format!(
            "assoc_legendre_p: P_{l}^{m} overflows f64 (the (2m-1)!! seed grows with the ORDER m); \
             use norm_assoc_legendre_p(l, m, x), which stays O(1) for these arguments"
        ))
    }
}

/// Fully **normalised** associated Legendre function
///
/// ```text
///     Pbar_l^m(x) = sqrt[ (2l+1)/(4 pi) * (l-m)!/(l+m)! ] * P_l^m(x)
/// ```
///
/// computed directly in the normalised basis so nothing overflows.
/// This is the factor spherical harmonics carry:
/// `Y_l^m(theta, phi) = Pbar_l^m(cos theta) e^{i m phi}`.
/// `DLMF 14.30.1` <https://dlmf.nist.gov/14.30.E1>.
///
/// The seed is written as a product of factors below one,
/// `Pbar_m^m = (-1)^m sqrt[(2m+1)/(4 pi) * prod_{k=1}^{m} (2k-1)/(2k)] (1-x^2)^{m/2}`,
/// and the ascent uses normalised coefficients, so every intermediate
/// value stays O(1).
///
/// # Errors
/// `l < 0`, `|m| > l`, or `|x| > 1`.
///
/// # Examples
/// ```
/// use special_functions::legendre::norm_assoc_legendre_p;
/// use std::f64::consts::PI;
/// // Pbar_0^0 = 1/sqrt(4 pi)
/// let v = norm_assoc_legendre_p(0, 0, 0.3).unwrap();
/// assert!((v - 1.0/(4.0*PI).sqrt()).abs() < 1e-15);
/// // stable where the unnormalised form would overflow
/// assert!(norm_assoc_legendre_p(400, 200, 0.5).unwrap().is_finite());
/// ```
pub fn norm_assoc_legendre_p(l: i32, m: i32, x: f64) -> Result<f64, String> {
    if l < 0 {
        return Err(format!(
            "norm_assoc_legendre_p: degree l must be >= 0, got {l}"
        ));
    }
    if m.abs() > l {
        return Err(format!(
            "norm_assoc_legendre_p: order m must satisfy |m| <= l, got l={l}, m={m}"
        ));
    }
    if !x.is_finite() || x.abs() > 1.0 {
        return Err(format!(
            "norm_assoc_legendre_p: x must lie in [-1, 1], got {x}"
        ));
    }
    // Pbar_l^{-m} = (-1)^m Pbar_l^m : the factorial ratio is already
    // inside the normalisation, so only the phase survives.
    if m < 0 {
        let v = norm_assoc_legendre_p(l, -m, x)?;
        return Ok(if (-m) % 2 == 0 { v } else { -v });
    }

    let somx2 = (1.0 - x * x).max(0.0).sqrt();
    // Pbar_m^m, built from factors that never leave O(1).
    let mut prod = 1.0_f64; // prod (2k-1)/(2k)
    for k in 1..=m {
        prod *= (2 * k - 1) as f64 / (2 * k) as f64;
    }
    let mut pmm = ((2 * m + 1) as f64 / (4.0 * PI) * prod).sqrt() * somx2.powi(m);
    if m % 2 == 1 {
        pmm = -pmm;
    }
    if l == m {
        return Ok(pmm);
    }
    // Pbar_{m+1}^m = sqrt(2m+3) x Pbar_m^m
    let mut pmmp1 = ((2 * m + 3) as f64).sqrt() * x * pmm;
    if l == m + 1 {
        return Ok(pmmp1);
    }
    let mut pll = 0.0_f64;
    for ll in (m + 2)..=l {
        let l2 = ll as f64;
        let m2 = m as f64;
        let a = ((4.0 * l2 * l2 - 1.0) / (l2 * l2 - m2 * m2)).sqrt();
        let b = (((l2 - 1.0) * (l2 - 1.0) - m2 * m2) / (4.0 * (l2 - 1.0) * (l2 - 1.0) - 1.0)).sqrt();
        pll = a * (x * pmmp1 - b * pmm);
        pmm = pmmp1;
        pmmp1 = pll;
    }
    Ok(pll)
}

/// Complex spherical harmonic `Y_l^m(theta, phi)`, returned as the pair
/// `(re, im)` because the crate has no complex type yet.
///
/// `Y_l^m = Pbar_l^m(cos theta) * (cos(m phi), sin(m phi))`,
/// `DLMF 14.30.1` <https://dlmf.nist.gov/14.30.E1>. Orthonormal on the
/// sphere: the integral of `|Y_l^m|^2` over all solid angle is 1.
///
/// # Examples
/// ```
/// use special_functions::legendre::sph_harm;
/// use std::f64::consts::PI;
/// // Y_0^0 = 1/sqrt(4 pi), independent of angle
/// let (re, im) = sph_harm(0, 0, 1.0, 2.0).unwrap();
/// assert!((re - 1.0/(4.0*PI).sqrt()).abs() < 1e-15 && im.abs() < 1e-18);
/// ```
pub fn sph_harm(l: i32, m: i32, theta: f64, phi: f64) -> Result<(f64, f64), String> {
    if !theta.is_finite() || !phi.is_finite() {
        return Err(format!(
            "sph_harm: theta and phi must be finite, got theta={theta}, phi={phi}"
        ));
    }
    let p = norm_assoc_legendre_p(l, m, theta.cos())?;
    let a = m as f64 * phi;
    Ok((p * a.cos(), p * a.sin()))
}

/// Real spherical harmonic, the basis used for orbitals and for
/// real-valued multipole expansions.
///
/// ```text
///   m > 0 :  sqrt(2) * Pbar_l^m(cos theta) * cos(m phi)
///   m = 0 :  Pbar_l^0(cos theta)
///   m < 0 :  sqrt(2) * Pbar_l^|m|(cos theta) * sin(|m| phi)
/// ```
///
/// Orthonormal over the sphere, like the complex family.
///
/// # Examples
/// ```
/// use special_functions::legendre::sph_harm_real;
/// use std::f64::consts::PI;
/// // the s orbital is constant
/// assert!((sph_harm_real(0, 0, 0.7, 1.9).unwrap() - 1.0/(4.0*PI).sqrt()).abs() < 1e-15);
/// // p_z ~ cos(theta) vanishes in the equatorial plane
/// assert!(sph_harm_real(1, 0, PI/2.0, 0.0).unwrap().abs() < 1e-16);
/// ```
pub fn sph_harm_real(l: i32, m: i32, theta: f64, phi: f64) -> Result<f64, String> {
    if !theta.is_finite() || !phi.is_finite() {
        return Err(format!(
            "sph_harm_real: theta and phi must be finite, got theta={theta}, phi={phi}"
        ));
    }
    let am = m.abs();
    let p = norm_assoc_legendre_p(l, am, theta.cos())?;
    if m == 0 {
        return Ok(p);
    }
    let s = std::f64::consts::SQRT_2;
    let a = am as f64 * phi;
    Ok(if m > 0 { s * p * a.cos() } else { s * p * a.sin() })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rel_err;

    /// Composite Simpson on a fine grid — good enough to confirm
    /// orthogonality integrals to ~1e-10 for smooth integrands.
    fn simpson<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, n: usize) -> f64 {
        let n = if n.is_multiple_of(2) { n } else { n + 1 };
        let h = (b - a) / n as f64;
        let mut s = f(a) + f(b);
        for i in 1..n {
            let x = a + i as f64 * h;
            s += if i % 2 == 1 { 4.0 * f(x) } else { 2.0 * f(x) };
        }
        s * h / 3.0
    }

    #[test]
    fn legendre_special_values_and_closed_forms() {
        for n in 0..12 {
            assert!(rel_err(legendre_p(n, 1.0).unwrap(), 1.0) < 1e-13, "P_{n}(1)");
            let want = if n % 2 == 0 { 1.0 } else { -1.0 };
            assert!(rel_err(legendre_p(n, -1.0).unwrap(), want) < 1e-13, "P_{n}(-1)");
        }
        for &x in &[-0.9, -0.3, 0.0, 0.42, 0.87] {
            assert!(rel_err(legendre_p(1, x).unwrap(), x) < 1e-14);
            assert!(rel_err(legendre_p(2, x).unwrap(), (3.0 * x * x - 1.0) / 2.0) < 1e-13);
            let p3 = (5.0 * x * x * x - 3.0 * x) / 2.0;
            assert!(rel_err(legendre_p(3, x).unwrap(), p3) < 1e-12, "P_3 at {x}");
        }
    }

    #[test]
    fn bonnet_recurrence_is_satisfied() {
        for &x in &[-0.77, -0.1, 0.33, 0.95] {
            for n in 1..20 {
                let lhs = (n + 1) as f64 * legendre_p(n + 1, x).unwrap();
                let rhs = (2 * n + 1) as f64 * x * legendre_p(n, x).unwrap()
                    - n as f64 * legendre_p(n - 1, x).unwrap();
                assert!(rel_err(lhs, rhs) < 1e-11, "Bonnet n={n} x={x}");
            }
        }
    }

    // Orthogonality: integral of P_m P_n over [-1,1] = 2 delta_{mn}/(2n+1)
    #[test]
    fn legendre_orthogonality_by_quadrature() {
        for m in 0..6 {
            for n in 0..6 {
                let v = simpson(
                    |x| legendre_p(m, x).unwrap() * legendre_p(n, x).unwrap(),
                    -1.0,
                    1.0,
                    4000,
                );
                let want = if m == n { 2.0 / (2 * n + 1) as f64 } else { 0.0 };
                assert!((v - want).abs() < 1e-9, "orthogonality m={m} n={n}: {v}");
            }
        }
    }

    #[test]
    fn associated_legendre_closed_forms_and_m0_reduction() {
        for &x in &[-0.8_f64, -0.2, 0.15, 0.6] {
            let s = (1.0 - x * x).sqrt();
            assert!(rel_err(assoc_legendre_p(1, 1, x).unwrap(), -s) < 1e-13);
            assert!(rel_err(assoc_legendre_p(2, 1, x).unwrap(), -3.0 * x * s) < 1e-13);
            assert!(rel_err(assoc_legendre_p(2, 2, x).unwrap(), 3.0 * (1.0 - x * x)) < 1e-13);
            for l in 0..7 {
                assert!(
                    rel_err(
                        assoc_legendre_p(l, 0, x).unwrap(),
                        legendre_p(l, x).unwrap()
                    ) < 1e-13,
                    "m=0 reduction l={l}"
                );
            }
        }
    }

    // P_l^{-m} = (-1)^m (l-m)!/(l+m)! P_l^m   (DLMF 14.9.3)
    #[test]
    fn associated_legendre_negative_order_relation() {
        for l in 0..6 {
            for m in 1..=l {
                for &x in &[-0.5, 0.25, 0.8] {
                    let neg = assoc_legendre_p(l, -m, x).unwrap();
                    let mut ratio = 1.0_f64;
                    for k in (l - m + 1)..=(l + m) {
                        ratio /= k as f64;
                    }
                    let want = if m % 2 == 0 { 1.0 } else { -1.0 }
                        * ratio
                        * assoc_legendre_p(l, m, x).unwrap();
                    assert!(rel_err(neg, want) < 1e-12, "l={l} m={m}");
                }
            }
        }
    }

    // The normalisation must reproduce N(l,m) * P_l^m where that is
    // still representable — this ties the two families together.
    #[test]
    fn normalised_matches_the_scaled_unnormalised_form() {
        for l in 0..9 {
            for m in 0..=l {
                for &x in &[-0.7, 0.0, 0.41, 0.9] {
                    let mut ratio = 1.0_f64; // (l-m)!/(l+m)!
                    for k in (l - m + 1)..=(l + m) {
                        ratio /= k as f64;
                    }
                    let n = ((2 * l + 1) as f64 / (4.0 * PI) * ratio).sqrt();
                    let want = n * assoc_legendre_p(l, m, x).unwrap();
                    let got = norm_assoc_legendre_p(l, m, x).unwrap();
                    assert!(rel_err(got, want) < 1e-11, "l={l} m={m} x={x}");
                }
            }
        }
    }

    // The whole point of the normalised form: it survives arguments
    // where the raw one leaves f64 entirely. The magnitudes quoted here
    // were MEASURED from this implementation, not assumed.
    #[test]
    fn normalised_form_is_stable_where_the_raw_form_dies() {
        for &(l, m) in &[(200, 0), (200, 100), (400, 200), (700, 3), (300, 300)] {
            let v = norm_assoc_legendre_p(l, m, 0.3).unwrap();
            assert!(v.is_finite(), "Pbar_{l}^{m} not finite: {v}");
            assert!(v.abs() < 10.0, "Pbar should stay O(1), got {v}");
        }
        // Overflow is governed by the ORDER m, not the degree l: these
        // are still representable...
        assert!(assoc_legendre_p(100, 50, 0.3).unwrap().abs() > 1e90);
        assert!(assoc_legendre_p(200, 100, 0.3).unwrap().abs() > 1e200);
        // ...while raising m past ~170 leaves f64, and is reported as an
        // error rather than handed back as inf/NaN.
        assert!(assoc_legendre_p(170, 170, 0.3).is_err());
        assert!(assoc_legendre_p(300, 150, 0.3).is_err());
        let msg = assoc_legendre_p(300, 300, 0.3).unwrap_err();
        assert!(msg.contains("norm_assoc_legendre_p"), "error should name the remedy: {msg}");
        // ...but m = 0 never overflows, however large the degree.
        assert!(assoc_legendre_p(200, 0, 0.3).unwrap().is_finite());
    }

    // Orthonormality on the sphere: 2 pi * integral over x of Pbar^2 = 1
    #[test]
    fn spherical_harmonics_are_orthonormal() {
        for l in 0..5 {
            for m in 0..=l {
                let v = simpson(
                    |x| {
                        let p = norm_assoc_legendre_p(l, m, x).unwrap();
                        p * p
                    },
                    -1.0,
                    1.0,
                    6000,
                );
                assert!(
                    (2.0 * PI * v - 1.0).abs() < 1e-8,
                    "norm l={l} m={m}: {}",
                    2.0 * PI * v
                );
            }
        }
        // different l, same m, must be orthogonal
        for (l1, l2, m) in [(1, 3, 0), (2, 4, 1), (2, 5, 2)] {
            let v = simpson(
                |x| {
                    norm_assoc_legendre_p(l1, m, x).unwrap()
                        * norm_assoc_legendre_p(l2, m, x).unwrap()
                },
                -1.0,
                1.0,
                6000,
            );
            assert!((v).abs() < 1e-9, "orthogonality l={l1},{l2} m={m}: {v}");
        }
    }

    #[test]
    fn spherical_harmonic_known_values() {
        // Y_0^0 = 1/sqrt(4 pi)
        let (re, im) = sph_harm(0, 0, 0.9, 1.4).unwrap();
        assert!(rel_err(re, 1.0 / (4.0 * PI).sqrt()) < 1e-14 && im.abs() < 1e-18);
        // Y_1^0 = sqrt(3/(4 pi)) cos(theta)
        for &t in &[0.0, 0.6, 1.57, 2.5] {
            let (re, _) = sph_harm(1, 0, t, 0.0).unwrap();
            assert!(rel_err(re, (3.0 / (4.0 * PI)).sqrt() * t.cos()) < 1e-12);
        }
        // |Y_1^1| = sqrt(3/(8 pi)) sin(theta)
        for &t in &[0.3, 1.0, 2.2] {
            let (re, im) = sph_harm(1, 1, t, 0.77).unwrap();
            let mag = (re * re + im * im).sqrt();
            assert!(rel_err(mag, (3.0 / (8.0 * PI)).sqrt() * t.sin()) < 1e-12);
        }
        // real p_z vanishes on the equator
        assert!(sph_harm_real(1, 0, PI / 2.0, 0.0).unwrap().abs() < 1e-16);
    }

    #[test]
    fn invalid_arguments_are_errors() {
        assert!(legendre_p(-1, 0.5).is_err());
        assert!(legendre_p(2, f64::NAN).is_err());
        assert!(assoc_legendre_p(2, 3, 0.5).is_err(), "|m| > l");
        assert!(assoc_legendre_p(2, 1, 1.5).is_err(), "|x| > 1");
        assert!(norm_assoc_legendre_p(-1, 0, 0.5).is_err());
        assert!(sph_harm(1, 0, f64::INFINITY, 0.0).is_err());
    }
}
