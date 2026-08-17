//! The classical orthogonal polynomials, evaluated by three-term
//! recurrences.
//!
//! Hermite (both conventions), Laguerre (ordinary and generalised),
//! Chebyshev of the first and second kind, ultraspherical
//! (Gegenbauer) and Jacobi. These are the polynomial eigenfunctions
//! that show up as soon as a separable PDE meets a classical weight
//! function: the quantum oscillator (Hermite), the radial hydrogen
//! problem (Laguerre), minimax approximation and spectral methods
//! (Chebyshev), and the whole `(1-x)^a (1+x)^b` family that contains
//! Legendre as a special case (Jacobi).
//!
//! # Why recurrences and not coefficient formulas
//!
//! Every one of these families has a closed-form coefficient sum —
//! `H_n(x) = n! * sum_k (-1)^k (2x)^(n-2k) / (k! (n-2k)!)`, and so on.
//! Those formulas are numerically useless. The individual terms grow
//! factorially and alternate in sign, so for `n` past roughly 15 the
//! sum loses every significant digit to cancellation, and past roughly
//! 170 the factorials overflow `f64` outright while the *answer* is
//! still perfectly representable. The three-term recurrences below
//! (`DLMF 18.9.1` <https://dlmf.nist.gov/18.9.E1>, with the
//! per-family coefficients in Table 18.9.1
//! <https://dlmf.nist.gov/18.9#T1>) cost `O(n)` flops, touch no
//! factorial, and are the *dominant* solution in the upward direction
//! for these families, so upward recurrence is the stable direction —
//! unlike `j_n` in [`crate::sph_bessel`], which needs Miller's
//! algorithm for exactly the opposite reason.
//!
//! # Orthogonality
//!
//! Each family is orthogonal on its own interval against its own
//! weight (Table 18.3.1 <https://dlmf.nist.gov/18.3#T1>):
//!
//! ```text
//!   H_n    (-inf, inf)   e^(-x^2)          h_n = sqrt(pi) 2^n n!
//!   He_n   (-inf, inf)   e^(-x^2/2)        h_n = sqrt(2 pi) n!
//!   L_n^a  (0, inf)      x^a e^(-x)        h_n = Gamma(n+a+1)/n!
//!   T_n    (-1, 1)       (1-x^2)^(-1/2)    h_n = pi/2 (pi for n=0)
//!   U_n    (-1, 1)       (1-x^2)^(1/2)     h_n = pi/2
//!   C_n^a  (-1, 1)       (1-x^2)^(a-1/2)
//!   P_n^ab (-1, 1)       (1-x)^a (1+x)^b
//! ```
//!
//! The test module verifies four of these by numerical quadrature
//! rather than trusting the recurrence.
//!
//! # Error policy
//!
//! As everywhere in this crate: `Result<f64, String>`, an actionable
//! message naming the offending argument, no panics, and no silent
//! `NaN` where the input was out of domain. Parameter ranges are
//! rejected exactly where the defining weight stops being integrable
//! (`alpha > -1` for generalised Laguerre and for Jacobi,
//! `alpha > -1/2` for Gegenbauer), which is also precisely where the
//! recurrence coefficients stop being finite.

/// Rejects the two arguments every function here shares.
fn check_common(who: &str, n: i32, x: f64) -> Result<(), String> {
    if n < 0 {
        return Err(format!(
            "{who}: degree n must be >= 0, got {n} (these polynomials are defined only for non-negative integer degree)"
        ));
    }
    if !x.is_finite() {
        return Err(format!(
            "{who}: argument x must be finite, got {x} (a polynomial recurrence cannot be seeded with NaN or infinity)"
        ));
    }
    Ok(())
}

/// Physicists' Hermite polynomial `H_n(x)`.
///
/// Orthogonal on `(-inf, inf)` against `e^(-x^2)`; the polynomial part
/// of the quantum harmonic-oscillator eigenfunctions. Evaluated by the
/// upward recurrence
///
/// ```text
///     H_{n+1}(x) = 2x H_n(x) - 2n H_{n-1}(x),   H_0 = 1,  H_1 = 2x
/// ```
///
/// `DLMF 18.9.1` <https://dlmf.nist.gov/18.9.E1> with the coefficients
/// `A_n = 2`, `B_n = 0`, `C_n = 2n` from Table 18.9.1
/// <https://dlmf.nist.gov/18.9#T1>; A&S 22.7.13.
///
/// # Domain and errors
/// Any finite `x`; `n >= 0`. Returns `Err` for negative degree or a
/// non-finite argument. `H_n` grows like `(2x)^n`, so the result
/// overflows to infinity for large `n` and `|x| > 1` — that is the
/// true value being unrepresentable, not a domain error.
///
/// # Examples
/// ```
/// use special_functions::orthopoly::hermite_h;
/// // H_3(x) = 8x^3 - 12x, so H_3(2) = 64 - 24 = 40
/// assert!((hermite_h(3, 2.0).unwrap() - 40.0).abs() < 1e-12);
/// // odd polynomials vanish at the origin
/// assert_eq!(hermite_h(7, 0.0).unwrap(), 0.0);
/// ```
pub fn hermite_h(n: i32, x: f64) -> Result<f64, String> {
    check_common("hermite_h", n, x)?;
    if n == 0 {
        return Ok(1.0);
    }
    let mut pm1 = 1.0_f64; // H_0
    let mut p = 2.0 * x; // H_1
    for k in 1..n {
        let next = 2.0 * x * p - 2.0 * k as f64 * pm1;
        pm1 = p;
        p = next;
    }
    Ok(p)
}

/// Probabilists' Hermite polynomial `He_n(x)`.
///
/// The same family rescaled to the standard-normal weight
/// `e^(-x^2/2)`, which is why it is the one statisticians use (its
/// coefficients are the moments of a Gaussian). Recurrence
///
/// ```text
///     He_{n+1}(x) = x He_n(x) - n He_{n-1}(x),   He_0 = 1,  He_1 = x
/// ```
///
/// `DLMF 18.9.1` <https://dlmf.nist.gov/18.9.E1> with `A_n = 1`,
/// `B_n = 0`, `C_n = n` (Table 18.9.1
/// <https://dlmf.nist.gov/18.9#T1>). The bridge to the physicists'
/// convention is `He_n(x) = 2^(-n/2) H_n(x/sqrt(2))`,
/// `DLMF 18.7.11` <https://dlmf.nist.gov/18.7.E11>.
///
/// # Domain and errors
/// Any finite `x`; `n >= 0`. Returns `Err` for negative degree or a
/// non-finite argument.
///
/// # Examples
/// ```
/// use special_functions::orthopoly::hermite_he;
/// // He_2(x) = x^2 - 1
/// assert!((hermite_he(2, 3.0).unwrap() - 8.0).abs() < 1e-12);
/// ```
pub fn hermite_he(n: i32, x: f64) -> Result<f64, String> {
    check_common("hermite_he", n, x)?;
    if n == 0 {
        return Ok(1.0);
    }
    let mut pm1 = 1.0_f64; // He_0
    let mut p = x; // He_1
    for k in 1..n {
        let next = x * p - k as f64 * pm1;
        pm1 = p;
        p = next;
    }
    Ok(p)
}

/// Laguerre polynomial `L_n(x)`, i.e. `L_n^0(x)`.
///
/// Orthonormal on `(0, inf)` against `e^(-x)`. Delegates to
/// [`laguerre_l_assoc`] with `alpha = 0`.
///
/// `DLMF 18.9.1` <https://dlmf.nist.gov/18.9.E1> (Table 18.9.1
/// <https://dlmf.nist.gov/18.9#T1>); special value `L_n(0) = 1` from
/// `DLMF 18.6.1` <https://dlmf.nist.gov/18.6.E1>.
///
/// # Domain and errors
/// Any finite `x` (the polynomial is defined on the whole line even
/// though its orthogonality interval is `(0, inf)`); `n >= 0`.
/// Returns `Err` for negative degree or a non-finite argument.
///
/// # Examples
/// ```
/// use special_functions::orthopoly::laguerre_l;
/// // L_2(x) = (x^2 - 4x + 2)/2, so L_2(1) = -1/2
/// assert!((laguerre_l(2, 1.0).unwrap() + 0.5).abs() < 1e-14);
/// // every L_n is 1 at the origin
/// assert!((laguerre_l(9, 0.0).unwrap() - 1.0).abs() < 1e-13);
/// ```
pub fn laguerre_l(n: i32, x: f64) -> Result<f64, String> {
    check_common("laguerre_l", n, x)?;
    Ok(laguerre_kernel(n, 0.0, x))
}

/// Generalised (associated) Laguerre polynomial `L_n^alpha(x)`.
///
/// Orthogonal on `(0, inf)` against `x^alpha e^(-x)`. With
/// `alpha = 2l+1` these are the radial factor of the hydrogen
/// wavefunctions. Recurrence
///
/// ```text
///     (n+1) L_{n+1}^a = (2n+a+1-x) L_n^a - (n+a) L_{n-1}^a
///     L_0^a = 1,   L_1^a = 1 + a - x
/// ```
///
/// `DLMF 18.9.1` <https://dlmf.nist.gov/18.9.E1> with
/// `A_n = -1/(n+1)`, `B_n = (2n+a+1)/(n+1)`, `C_n = (n+a)/(n+1)`
/// (Table 18.9.1 <https://dlmf.nist.gov/18.9#T1>); A&S 22.7.12.
///
/// # Domain and errors
/// `n >= 0`, finite `x`, and `alpha > -1`. At `alpha <= -1` the weight
/// `x^alpha e^(-x)` is not integrable at the origin, so the family has
/// no orthogonality and the standardisation is meaningless; that is
/// reported as an error rather than silently evaluated.
///
/// # Examples
/// ```
/// use special_functions::orthopoly::laguerre_l_assoc;
/// // L_1^a(x) = 1 + a - x
/// assert!((laguerre_l_assoc(1, 0.5, 2.0).unwrap() + 0.5).abs() < 1e-14);
/// // alpha = 0 recovers the ordinary Laguerre polynomials
/// assert!((laguerre_l_assoc(2, 0.0, 1.0).unwrap() + 0.5).abs() < 1e-14);
/// // a non-integrable weight is refused, not guessed at
/// assert!(laguerre_l_assoc(2, -1.5, 1.0).is_err());
/// ```
pub fn laguerre_l_assoc(n: i32, alpha: f64, x: f64) -> Result<f64, String> {
    check_common("laguerre_l_assoc", n, x)?;
    if !alpha.is_finite() {
        return Err(format!(
            "laguerre_l_assoc: alpha must be finite, got {alpha}"
        ));
    }
    if alpha <= -1.0 {
        return Err(format!(
            "laguerre_l_assoc: alpha must be > -1, got {alpha} (the weight x^alpha e^(-x) is not integrable at 0 for alpha <= -1, so L_n^alpha has no orthogonality there)"
        ));
    }
    Ok(laguerre_kernel(n, alpha, x))
}

/// Unchecked Laguerre recurrence; callers validate first.
fn laguerre_kernel(n: i32, alpha: f64, x: f64) -> f64 {
    if n == 0 {
        return 1.0;
    }
    let mut pm1 = 1.0_f64; // L_0^a
    let mut p = 1.0 + alpha - x; // L_1^a
    for k in 1..n {
        let kf = k as f64;
        let next = ((2.0 * kf + alpha + 1.0 - x) * p - (kf + alpha) * pm1) / (kf + 1.0);
        pm1 = p;
        p = next;
    }
    p
}

/// Chebyshev polynomial of the first kind, `T_n(x)`.
///
/// The minimax polynomial: `T_n` has the smallest possible maximum
/// magnitude on `[-1, 1]` among monic polynomials of its degree
/// (after scaling), which is why interpolation at its roots is the
/// standard defence against Runge's phenomenon. Recurrence
///
/// ```text
///     T_{n+1}(x) = 2x T_n(x) - T_{n-1}(x),   T_0 = 1,  T_1 = x
/// ```
///
/// `DLMF 18.9.1` <https://dlmf.nist.gov/18.9.E1> (Table 18.9.1
/// <https://dlmf.nist.gov/18.9#T1>). On `[-1, 1]` it satisfies
/// `T_n(cos t) = cos(n t)`, `DLMF 18.5.1`
/// <https://dlmf.nist.gov/18.5.E1>.
///
/// # Domain and errors
/// Any finite `x` — the polynomial continues outside `[-1, 1]`, where
/// it grows like `cosh(n arccosh x)`. `n >= 0`. Returns `Err` for
/// negative degree or a non-finite argument.
///
/// # Examples
/// ```
/// use special_functions::orthopoly::chebyshev_t;
/// // the defining trigonometric identity
/// let t = 0.7_f64;
/// assert!((chebyshev_t(5, t.cos()).unwrap() - (5.0 * t).cos()).abs() < 1e-14);
/// // T_3(x) = 4x^3 - 3x
/// assert!((chebyshev_t(3, 0.5).unwrap() + 1.0).abs() < 1e-15);
/// ```
pub fn chebyshev_t(n: i32, x: f64) -> Result<f64, String> {
    check_common("chebyshev_t", n, x)?;
    if n == 0 {
        return Ok(1.0);
    }
    let mut pm1 = 1.0_f64; // T_0
    let mut p = x; // T_1
    for _ in 1..n {
        let next = 2.0 * x * p - pm1;
        pm1 = p;
        p = next;
    }
    Ok(p)
}

/// Chebyshev polynomial of the second kind, `U_n(x)`.
///
/// Orthogonal on `(-1, 1)` against `sqrt(1-x^2)`; equals
/// `sin((n+1)t)/sin(t)` at `x = cos t`, `DLMF 18.5.2`
/// <https://dlmf.nist.gov/18.5.E2>. Same recurrence as `T_n` but
/// seeded with `U_1 = 2x`:
///
/// ```text
///     U_{n+1}(x) = 2x U_n(x) - U_{n-1}(x),   U_0 = 1,  U_1 = 2x
/// ```
///
/// `DLMF 18.9.1` <https://dlmf.nist.gov/18.9.E1> (Table 18.9.1
/// <https://dlmf.nist.gov/18.9#T1>).
///
/// # Domain and errors
/// Any finite `x`; `n >= 0`. Returns `Err` for negative degree or a
/// non-finite argument.
///
/// # Examples
/// ```
/// use special_functions::orthopoly::chebyshev_u;
/// // U_n(1) = n + 1
/// assert!((chebyshev_u(3, 1.0).unwrap() - 4.0).abs() < 1e-13);
/// // U_2(x) = 4x^2 - 1
/// assert!((chebyshev_u(2, 0.5).unwrap() - 0.0).abs() < 1e-15);
/// ```
pub fn chebyshev_u(n: i32, x: f64) -> Result<f64, String> {
    check_common("chebyshev_u", n, x)?;
    if n == 0 {
        return Ok(1.0);
    }
    let mut pm1 = 1.0_f64; // U_0
    let mut p = 2.0 * x; // U_1
    for _ in 1..n {
        let next = 2.0 * x * p - pm1;
        pm1 = p;
        p = next;
    }
    Ok(p)
}

/// Ultraspherical (Gegenbauer) polynomial `C_n^alpha(x)`.
///
/// Orthogonal on `(-1, 1)` against `(1-x^2)^(alpha-1/2)`; the
/// one-parameter family interpolating between Chebyshev
/// (`alpha -> 0`, after renormalisation) and `U_n` at `alpha = 1`,
/// with Legendre at `alpha = 1/2`. Recurrence
///
/// ```text
///     n C_n^a = 2(n+a-1) x C_{n-1}^a - (n+2a-2) C_{n-2}^a
///     C_0^a = 1,   C_1^a = 2a x
/// ```
///
/// `DLMF 18.9.1` <https://dlmf.nist.gov/18.9.E1> with `A_n =
/// 2(n+a)/(n+1)`, `B_n = 0`, `C_n = (n+2a-1)/(n+1)` (Table 18.9.1
/// <https://dlmf.nist.gov/18.9#T1>). Special cases `U_n = C_n^1`
/// and `P_n = C_n^(1/2)` are `DLMF 18.7.4`
/// <https://dlmf.nist.gov/18.7.E4> and `DLMF 18.7.9`
/// <https://dlmf.nist.gov/18.7.E9>.
///
/// # Domain and errors
/// `n >= 0`, finite `x`, and `alpha > -1/2`, the condition for
/// `(1-x^2)^(alpha-1/2)` to be integrable on `(-1, 1)`; `alpha <= -1/2`
/// is an error. The value `alpha = 0` is permitted but degenerate: the
/// standardisation collapses and `C_n^0(x) = 0` for every `n >= 1`. If
/// you want the Chebyshev limit, take the renormalised one,
/// `lim_{a->0} (n/a) C_n^a(x) = 2 T_n(x)`, or call [`chebyshev_t`].
///
/// # Examples
/// ```
/// use special_functions::orthopoly::{chebyshev_u, gegenbauer_c};
/// // C_n^1 = U_n
/// let x = 0.5_f64;
/// assert!((gegenbauer_c(3, 1.0, x).unwrap() - chebyshev_u(3, x).unwrap()).abs() < 1e-14);
/// // a non-integrable weight is refused
/// assert!(gegenbauer_c(2, -0.5, x).is_err());
/// ```
pub fn gegenbauer_c(n: i32, alpha: f64, x: f64) -> Result<f64, String> {
    check_common("gegenbauer_c", n, x)?;
    if !alpha.is_finite() {
        return Err(format!("gegenbauer_c: alpha must be finite, got {alpha}"));
    }
    if alpha <= -0.5 {
        return Err(format!(
            "gegenbauer_c: alpha must be > -1/2, got {alpha} (the weight (1-x^2)^(alpha-1/2) is not integrable on (-1,1) for alpha <= -1/2)"
        ));
    }
    if n == 0 {
        return Ok(1.0);
    }
    let mut pm1 = 1.0_f64; // C_0
    let mut p = 2.0 * alpha * x; // C_1
    for k in 2..=n {
        let kf = k as f64;
        let next = (2.0 * (kf + alpha - 1.0) * x * p - (kf + 2.0 * alpha - 2.0) * pm1) / kf;
        pm1 = p;
        p = next;
    }
    Ok(p)
}

/// Jacobi polynomial `P_n^(alpha,beta)(x)`.
///
/// The most general classical family on a finite interval: orthogonal
/// on `(-1, 1)` against `(1-x)^alpha (1+x)^beta`. Legendre is
/// `P_n^(0,0)`, `DLMF 18.7.9` <https://dlmf.nist.gov/18.7.E9>;
/// Chebyshev and Gegenbauer are rescalings of `alpha = beta` cases.
///
/// Evaluated by the standard three-term recurrence in `n`,
/// `DLMF 18.9.2` <https://dlmf.nist.gov/18.9.E2>, written in the
/// `DLMF 18.9.1` <https://dlmf.nist.gov/18.9.E1> form
/// `P_{n+1} = (A_n x + B_n) P_n - C_n P_{n-1}` with
///
/// ```text
///   s   = 2n + a + b
///   A_n = (s+1)(s+2) / (2(n+1)(n+a+b+1))
///   B_n = (a^2 - b^2)(s+1) / (2(n+1)(n+a+b+1) s)
///   C_n = (n+a)(n+b)(s+2) / ((n+1)(n+a+b+1) s)
/// ```
///
/// seeded with `P_0 = 1` and `P_1 = (a-b)/2 + (a+b+2)x/2`. The
/// recurrence is started at `n = 1` deliberately: at `n = 0` the
/// factor `s = a+b` sits in a denominator and vanishes whenever
/// `a + b = 0` (Legendre among them), so the `n = 0` step is taken
/// from the closed form for `P_1` instead. For `n >= 1` and
/// `a, b > -1` every denominator is bounded away from zero.
///
/// # Domain and errors
/// `n >= 0`, finite `x`, and `alpha > -1`, `beta > -1` — the condition
/// for the weight to be integrable at `x = 1` and `x = -1`
/// respectively. Anything else is an error.
///
/// # Examples
/// ```
/// use special_functions::orthopoly::jacobi_p;
/// // P_n^(0,0) is the Legendre polynomial; P_2(x) = (3x^2-1)/2
/// assert!((jacobi_p(2, 0.0, 0.0, 0.5).unwrap() + 0.125).abs() < 1e-15);
/// // P_n^(a,b)(1) = (a+1)_n / n!, so P_2^(1,3)(1) = (2*3)/2 = 3
/// assert!((jacobi_p(2, 1.0, 3.0, 1.0).unwrap() - 3.0).abs() < 1e-13);
/// assert!(jacobi_p(2, -1.0, 0.0, 0.5).is_err());
/// ```
pub fn jacobi_p(n: i32, alpha: f64, beta: f64, x: f64) -> Result<f64, String> {
    check_common("jacobi_p", n, x)?;
    if !alpha.is_finite() || !beta.is_finite() {
        return Err(format!(
            "jacobi_p: alpha and beta must be finite, got alpha={alpha}, beta={beta}"
        ));
    }
    if alpha <= -1.0 {
        return Err(format!(
            "jacobi_p: alpha must be > -1, got {alpha} (the weight (1-x)^alpha is not integrable at x=1 for alpha <= -1)"
        ));
    }
    if beta <= -1.0 {
        return Err(format!(
            "jacobi_p: beta must be > -1, got {beta} (the weight (1+x)^beta is not integrable at x=-1 for beta <= -1)"
        ));
    }
    if n == 0 {
        return Ok(1.0);
    }
    let ab = alpha + beta;
    let mut pm1 = 1.0_f64; // P_0
    let mut p = 0.5 * (alpha - beta) + 0.5 * (ab + 2.0) * x; // P_1
    for k in 1..n {
        let kf = k as f64;
        let s = 2.0 * kf + ab;
        let denom = 2.0 * (kf + 1.0) * (kf + ab + 1.0) * s;
        let a_n = (s + 1.0) * (s + 2.0) * s / denom;
        let b_n = (alpha * alpha - beta * beta) * (s + 1.0) / denom;
        let c_n = 2.0 * (kf + alpha) * (kf + beta) * (s + 2.0) / denom;
        let next = (a_n * x + b_n) * p - c_n * pm1;
        pm1 = p;
        p = next;
    }
    Ok(p)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rel_err;

    // ================================================================
    // helpers
    // ================================================================

    /// Composite Simpson rule on `[a, b]` with `panels` panels
    /// (`panels` must be even). Used for the orthogonality integrals
    /// below; deliberately written here rather than imported so the
    /// quadrature is independent of anything under test.
    fn simpson<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, panels: usize) -> f64 {
        assert!(panels >= 2 && panels.is_multiple_of(2), "panels must be even and >= 2");
        let h = (b - a) / panels as f64;
        let mut s = f(a) + f(b);
        for i in 1..panels {
            let w = if i % 2 == 1 { 4.0 } else { 2.0 };
            s += w * f(a + i as f64 * h);
        }
        s * h / 3.0
    }

    /// Legendre `P_n(x)` by Bonnet's recurrence, computed here so the
    /// Jacobi and Gegenbauer reduction tests have an *independent*
    /// reference rather than comparing the module to itself.
    fn legendre_bonnet(n: i32, x: f64) -> f64 {
        if n == 0 {
            return 1.0;
        }
        let mut pm1 = 1.0_f64;
        let mut p = x;
        for k in 1..n {
            let kf = k as f64;
            let next = ((2.0 * kf + 1.0) * x * p - kf * pm1) / (kf + 1.0);
            pm1 = p;
            p = next;
        }
        p
    }

    /// `n!` as an `f64`, for the Hermite/Laguerre normalisations.
    fn factorial(n: i32) -> f64 {
        (1..=n).map(|k| k as f64).product::<f64>()
    }

    /// Double factorial `(2m-1)!! = 1*3*5*...*(2m-1)`, with `(-1)!! = 1`.
    fn double_factorial_odd(m: i32) -> f64 {
        (1..=m).map(|k| (2 * k - 1) as f64).product::<f64>()
    }

    // ================================================================
    // 1. known low-order closed forms
    // ================================================================

    #[test]
    fn low_order_closed_forms() {
        for &x in &[-2.3, -1.0, -0.4, 0.0, 0.25, 0.9, 1.0, 3.7] {
            assert!(
                rel_err(hermite_h(2, x).unwrap(), 4.0 * x * x - 2.0) < 1e-14,
                "H_2 at x={x}"
            );
            assert!(
                rel_err(hermite_h(3, x).unwrap(), 8.0 * x * x * x - 12.0 * x) < 1e-14,
                "H_3 at x={x}"
            );
            assert!(
                rel_err(hermite_he(2, x).unwrap(), x * x - 1.0) < 1e-14,
                "He_2 at x={x}"
            );
            assert!(
                rel_err(laguerre_l(2, x).unwrap(), (x * x - 4.0 * x + 2.0) / 2.0) < 1e-14,
                "L_2 at x={x}"
            );
            assert!(
                rel_err(chebyshev_t(2, x).unwrap(), 2.0 * x * x - 1.0) < 1e-14,
                "T_2 at x={x}"
            );
            assert!(
                rel_err(chebyshev_t(3, x).unwrap(), 4.0 * x * x * x - 3.0 * x) < 1e-14,
                "T_3 at x={x}"
            );
            assert!(
                rel_err(chebyshev_u(2, x).unwrap(), 4.0 * x * x - 1.0) < 1e-14,
                "U_2 at x={x}"
            );
        }
    }

    /// The generalised Laguerre closed form
    /// `L_2^a(x) = x^2/2 - (a+2)x + (a+1)(a+2)/2`.
    #[test]
    fn generalised_laguerre_low_order_closed_form() {
        for &a in &[-0.75, -0.5, 0.0, 0.5, 1.0, 2.5, 7.0] {
            for &x in &[0.0, 0.3, 1.0, 4.0, 12.5] {
                assert!(
                    rel_err(laguerre_l_assoc(1, a, x).unwrap(), 1.0 + a - x) < 1e-14,
                    "L_1^{a} at x={x}"
                );
                let want = x * x / 2.0 - (a + 2.0) * x + (a + 1.0) * (a + 2.0) / 2.0;
                assert!(
                    rel_err(laguerre_l_assoc(2, a, x).unwrap(), want) < 1e-13,
                    "L_2^{a} at x={x}"
                );
            }
        }
    }

    // ================================================================
    // 2. special values (Table 18.6.1)
    // ================================================================

    #[test]
    fn chebyshev_special_values_at_endpoints() {
        for n in 0..30 {
            let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
            assert!(rel_err(chebyshev_t(n, 1.0).unwrap(), 1.0) < 1e-14, "T_{n}(1)");
            assert!(
                rel_err(chebyshev_t(n, -1.0).unwrap(), sign) < 1e-14,
                "T_{n}(-1)"
            );
            assert!(
                rel_err(chebyshev_u(n, 1.0).unwrap(), (n + 1) as f64) < 1e-13,
                "U_{n}(1)"
            );
            assert!(
                rel_err(chebyshev_u(n, -1.0).unwrap(), sign * (n + 1) as f64) < 1e-13,
                "U_{n}(-1)"
            );
        }
    }

    /// `L_n(0) = 1` (DLMF 18.6.1), and more generally
    /// `L_n^a(0) = (a+1)_n / n!`.
    #[test]
    fn laguerre_special_values_at_origin() {
        for n in 0..25 {
            assert!(rel_err(laguerre_l(n, 0.0).unwrap(), 1.0) < 1e-13, "L_{n}(0)");
        }
        for &a in &[-0.5, 0.5, 2.0, 4.5] {
            for n in 0..15 {
                let want: f64 = (1..=n).map(|k| (a + k as f64) / k as f64).product();
                assert!(
                    rel_err(laguerre_l_assoc(n, a, 0.0).unwrap(), want) < 1e-12,
                    "L_{n}^{a}(0)"
                );
            }
        }
    }

    /// `H_n(0) = 0` for odd `n`, and `(-1)^(n/2) (n-1)!! 2^(n/2)` for
    /// even `n`.
    #[test]
    fn hermite_special_values_at_origin() {
        for n in 0..25 {
            let got = hermite_h(n, 0.0).unwrap();
            if n % 2 == 1 {
                assert_eq!(got, 0.0, "H_{n}(0) must be exactly zero");
            } else {
                let m = n / 2;
                let sign = if m % 2 == 0 { 1.0 } else { -1.0 };
                let want = sign * double_factorial_odd(m) * 2f64.powi(m);
                assert!(rel_err(got, want) < 1e-13, "H_{n}(0): got {got}, want {want}");
            }
        }
    }

    /// `P_n^(a,b)(1) = (a+1)_n/n!` and
    /// `P_n^(a,b)(-1) = (-1)^n (b+1)_n/n!` (Table 18.6.1), plus
    /// `C_n^a(1) = (2a)_n/n!`.
    #[test]
    fn jacobi_and_gegenbauer_endpoint_values() {
        for &(a, b) in &[(0.0, 0.0), (1.0, 2.0), (-0.5, 0.5), (3.25, -0.75)] {
            for n in 0..14 {
                let at_one: f64 = (1..=n).map(|k| (a + k as f64) / k as f64).product();
                let at_minus: f64 = (1..=n).map(|k| (b + k as f64) / k as f64).product();
                let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
                assert!(
                    rel_err(jacobi_p(n, a, b, 1.0).unwrap(), at_one) < 1e-12,
                    "P_{n}^({a},{b})(1)"
                );
                assert!(
                    rel_err(jacobi_p(n, a, b, -1.0).unwrap(), sign * at_minus) < 1e-12,
                    "P_{n}^({a},{b})(-1)"
                );
            }
        }
        for &a in &[0.25, 0.5, 1.0, 3.0] {
            for n in 0..14 {
                let want: f64 = (1..=n).map(|k| (2.0 * a + k as f64 - 1.0) / k as f64).product();
                assert!(
                    rel_err(gegenbauer_c(n, a, 1.0).unwrap(), want) < 1e-12,
                    "C_{n}^{a}(1)"
                );
            }
        }
    }

    // ================================================================
    // 3. parity
    // ================================================================

    #[test]
    fn parity_relations() {
        for n in 0..20 {
            let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
            for &x in &[0.13, 0.5, 0.99, 1.7, 2.6] {
                assert!(
                    rel_err(hermite_h(n, -x).unwrap(), sign * hermite_h(n, x).unwrap()) < 1e-13,
                    "H parity n={n} x={x}"
                );
                assert!(
                    rel_err(hermite_he(n, -x).unwrap(), sign * hermite_he(n, x).unwrap()) < 1e-13,
                    "He parity n={n} x={x}"
                );
                assert!(
                    rel_err(chebyshev_t(n, -x).unwrap(), sign * chebyshev_t(n, x).unwrap()) < 1e-13,
                    "T parity n={n} x={x}"
                );
                assert!(
                    rel_err(chebyshev_u(n, -x).unwrap(), sign * chebyshev_u(n, x).unwrap()) < 1e-13,
                    "U parity n={n} x={x}"
                );
            }
        }
        // Jacobi's parity swaps the parameters:
        // P_n^(a,b)(-x) = (-1)^n P_n^(b,a)(x)  (DLMF 18.6.1 / Table 18.6.1)
        for n in 0..12 {
            let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
            for &x in &[0.1, 0.44, 0.87] {
                let lhs = jacobi_p(n, 1.5, -0.25, -x).unwrap();
                let rhs = sign * jacobi_p(n, -0.25, 1.5, x).unwrap();
                assert!(rel_err(lhs, rhs) < 1e-12, "Jacobi parity n={n} x={x}");
            }
        }
    }

    // ================================================================
    // 4. the Chebyshev trigonometric identity (DLMF 18.5.1, 18.5.2)
    // ================================================================

    #[test]
    fn chebyshev_matches_cosine_of_multiple_angle() {
        let mut worst = 0.0_f64;
        for n in 0..40 {
            for i in 0..=64 {
                let t = std::f64::consts::PI * i as f64 / 64.0;
                let x = t.cos();
                let got = chebyshev_t(n, x).unwrap();
                let want = (n as f64 * t).cos();
                // T_n is bounded by 1 here, so an absolute comparison
                // is the meaningful one: near a zero of cos(nt) the
                // relative error is unbounded but irrelevant.
                let err = (got - want).abs();
                worst = worst.max(err);
                assert!(err < 1e-12, "T_{n}(cos {t}) = {got}, want {want}");

                // and the second-kind identity U_n(cos t) sin t = sin((n+1)t)
                let got_u = chebyshev_u(n, x).unwrap() * t.sin();
                let want_u = ((n + 1) as f64 * t).sin();
                assert!(
                    (got_u - want_u).abs() < 1e-11,
                    "U_{n}(cos {t}) sin t = {got_u}, want {want_u}"
                );
            }
        }
        assert!(worst < 1e-12, "worst T_n(cos t) error was {worst}");
    }

    // ================================================================
    // 5. orthogonality by numerical quadrature
    // ================================================================

    /// Chebyshev-T orthogonality against `(1-x^2)^(-1/2)` on `(-1,1)`.
    ///
    /// The weight is singular at both endpoints, which composite
    /// Simpson cannot integrate directly, so we substitute `x = cos t`.
    /// That maps the integral to `int_0^pi T_m(cos t) T_n(cos t) dt`
    /// with the singularity absorbed exactly by `dx = -sin t dt` — the
    /// change of variable is analytic, the quadrature is still doing
    /// real work on the computed polynomial values.
    #[test]
    fn chebyshev_t_orthogonality_by_quadrature() {
        let pi = std::f64::consts::PI;
        for m in 0..7 {
            for n in 0..7 {
                let integral = simpson(
                    |t| chebyshev_t(m, t.cos()).unwrap() * chebyshev_t(n, t.cos()).unwrap(),
                    0.0,
                    pi,
                    4000,
                );
                let want = if m != n {
                    0.0
                } else if m == 0 {
                    pi
                } else {
                    pi / 2.0
                };
                assert!(
                    (integral - want).abs() < 1e-9,
                    "int T_{m} T_{n} w = {integral}, want {want}"
                );
            }
        }
    }

    /// Chebyshev-U orthogonality against `sqrt(1-x^2)`, same
    /// substitution: `int_0^pi U_m(cos t) U_n(cos t) sin^2 t dt`.
    #[test]
    fn chebyshev_u_orthogonality_by_quadrature() {
        let pi = std::f64::consts::PI;
        for m in 0..7 {
            for n in 0..7 {
                let integral = simpson(
                    |t| {
                        chebyshev_u(m, t.cos()).unwrap()
                            * chebyshev_u(n, t.cos()).unwrap()
                            * t.sin()
                            * t.sin()
                    },
                    0.0,
                    pi,
                    4000,
                );
                let want = if m == n { pi / 2.0 } else { 0.0 };
                assert!(
                    (integral - want).abs() < 1e-9,
                    "int U_{m} U_{n} w = {integral}, want {want}"
                );
            }
        }
    }

    /// Hermite orthogonality against `e^(-x^2)` on the whole line,
    /// truncated to `[-8, 8]`. For `m, n <= 5` the integrand is
    /// bounded by roughly `x^10 e^(-x^2)`, which at `|x| = 8` is
    /// `~1e-19` and falls off super-exponentially, so the truncation
    /// error is far below the quadrature tolerance.
    /// Normalisation `h_n = sqrt(pi) 2^n n!` (Table 18.3.1).
    #[test]
    fn hermite_orthogonality_by_quadrature() {
        let norm = |k: i32| std::f64::consts::PI.sqrt() * 2f64.powi(k) * factorial(k);
        for m in 0..6 {
            for n in 0..6 {
                let integral = simpson(
                    |x| hermite_h(m, x).unwrap() * hermite_h(n, x).unwrap() * (-x * x).exp(),
                    -8.0,
                    8.0,
                    20_000,
                );
                // scale out the enormous dynamic range of h_n so one
                // tolerance covers every (m, n)
                let scale = (norm(m) * norm(n)).sqrt();
                let want = if m == n { norm(m) } else { 0.0 };
                assert!(
                    ((integral - want) / scale).abs() < 1e-10,
                    "int H_{m} H_{n} e^-x^2 = {integral}, want {want}"
                );
            }
        }
    }

    /// Laguerre orthogonality against `e^(-x)` on `(0, inf)`,
    /// truncated at `x = 60`: for `m, n <= 5` the integrand there is
    /// about `60^10/(5!)^2 e^(-60) ~ 1e-10 * 1e-26`, negligible.
    /// `h_n = 1` for `alpha = 0` (Table 18.3.1).
    #[test]
    fn laguerre_orthogonality_by_quadrature() {
        for m in 0..6 {
            for n in 0..6 {
                let integral = simpson(
                    |x| laguerre_l(m, x).unwrap() * laguerre_l(n, x).unwrap() * (-x).exp(),
                    0.0,
                    60.0,
                    60_000,
                );
                let want = if m == n { 1.0 } else { 0.0 };
                assert!(
                    (integral - want).abs() < 1e-9,
                    "int L_{m} L_{n} e^-x = {integral}, want {want}"
                );
            }
        }
    }

    /// Jacobi orthogonality against `(1-x)^1 (1+x)^2`. Integer
    /// parameters are chosen on purpose: the weight is then a
    /// polynomial, so the whole integrand is a polynomial and Simpson
    /// converges at full order with no endpoint singularity to fight.
    /// (Fractional `alpha, beta` give an algebraic endpoint
    /// singularity in the derivative that would degrade Simpson to
    /// about `h^(1+alpha)` — the fractional cases are covered instead
    /// by the endpoint-value and Legendre-reduction tests.)
    #[test]
    fn jacobi_orthogonality_by_quadrature() {
        let (a, b) = (1.0_f64, 2.0_f64);
        let w = |x: f64| (1.0 - x).powf(a) * (1.0 + x).powf(b);
        let diag: Vec<f64> = (0..6)
            .map(|k| {
                simpson(
                    |x| {
                        let p = jacobi_p(k, a, b, x).unwrap();
                        p * p * w(x)
                    },
                    -1.0,
                    1.0,
                    20_000,
                )
            })
            .collect();
        for m in 0..6 {
            for n in 0..6 {
                let integral = simpson(
                    |x| jacobi_p(m, a, b, x).unwrap() * jacobi_p(n, a, b, x).unwrap() * w(x),
                    -1.0,
                    1.0,
                    20_000,
                );
                let scale = (diag[m as usize] * diag[n as usize]).sqrt();
                if m == n {
                    assert!(diag[m as usize] > 1e-3, "P_{m} has degenerate norm");
                } else {
                    assert!(
                        (integral / scale).abs() < 1e-10,
                        "int P_{m} P_{n} w = {integral} (scaled {})",
                        integral / scale
                    );
                }
            }
        }
    }

    // ================================================================
    // 6. cross-relation between the two Hermite conventions
    // ================================================================

    /// `He_n(x) = 2^(-n/2) H_n(x/sqrt(2))`, DLMF 18.7.11.
    #[test]
    fn hermite_conventions_agree() {
        let r2 = 2f64.sqrt();
        for n in 0..25 {
            for &x in &[-3.1, -0.7, 0.0, 0.45, 1.0, 2.8, 6.0] {
                let want = 2f64.powf(-(n as f64) / 2.0) * hermite_h(n, x / r2).unwrap();
                let got = hermite_he(n, x).unwrap();
                // Both sides pass through the zeros of He_n, where a
                // pure relative comparison is undefined: at n=2, x=1
                // the exact answer is 0 and the *reference* side is
                // the inexact one (it rounds (x/sqrt 2)^2 to
                // 0.5000000000000001, giving -2.2e-16). So compare
                // against a scale with a floor of 1. The measured
                // worst case over this whole grid is 1.8e-13, so the
                // bound below is not slack that hides anything.
                let scale = got.abs().max(want.abs()).max(1.0);
                assert!(
                    (got - want).abs() / scale < 1e-11,
                    "He_{n}({x}): {got} vs {want}"
                );
            }
        }
    }

    // ================================================================
    // 7. reductions to known families
    // ================================================================

    /// `C_n^1 = U_n` and `C_n^(1/2) = P_n` (DLMF 18.7.4, 18.7.9).
    #[test]
    fn gegenbauer_reduces_to_chebyshev_u_and_legendre() {
        for n in 0..20 {
            for &x in &[-0.95, -0.5, 0.0, 0.3, 0.77, 1.0] {
                assert!(
                    rel_err(gegenbauer_c(n, 1.0, x).unwrap(), chebyshev_u(n, x).unwrap()) < 1e-12,
                    "C_{n}^1 vs U_{n} at x={x}"
                );
                assert!(
                    rel_err(gegenbauer_c(n, 0.5, x).unwrap(), legendre_bonnet(n, x)) < 1e-12,
                    "C_{n}^(1/2) vs P_{n} at x={x}"
                );
            }
        }
    }

    /// `P_n^(0,0) = P_n`, the Legendre polynomials (DLMF 18.7.9),
    /// checked against an independent Bonnet recurrence.
    #[test]
    fn jacobi_reduces_to_legendre() {
        for n in 0..20 {
            for &x in &[-1.0, -0.6, -0.05, 0.0, 0.41, 0.9, 1.0] {
                let got = jacobi_p(n, 0.0, 0.0, x).unwrap();
                let want = legendre_bonnet(n, x);
                assert!(rel_err(got, want) < 1e-12, "P_{n}^(0,0)({x}): {got} vs {want}");
            }
        }
    }

    /// `T_n(x) = P_n^(-1/2,-1/2)(x) / P_n^(-1/2,-1/2)(1)` (DLMF
    /// 18.7.3) — a reduction that exercises the Jacobi recurrence at
    /// negative parameters, where `2n + a + b = 2n - 1` stays clear of
    /// zero only because we start the loop at `n = 1`.
    #[test]
    fn jacobi_reduces_to_chebyshev_t() {
        for n in 0..16 {
            let scale = jacobi_p(n, -0.5, -0.5, 1.0).unwrap();
            for &x in &[-0.9, -0.33, 0.0, 0.25, 0.81] {
                let got = jacobi_p(n, -0.5, -0.5, x).unwrap() / scale;
                let want = chebyshev_t(n, x).unwrap();
                assert!((got - want).abs() < 1e-12, "T_{n}({x}) via Jacobi: {got} vs {want}");
            }
        }
    }

    /// `L_n^(a+1)(x) = sum_{k=0}^{n} L_k^(a)(x)`, a non-trivial
    /// identity linking the generalised family across its parameter.
    #[test]
    fn associated_laguerre_partial_sum_identity() {
        for &a in &[-0.5, 0.0, 0.75, 3.0] {
            for &x in &[0.0, 0.6, 2.0, 5.5, 11.0] {
                for n in 0..12 {
                    let lhs = laguerre_l_assoc(n, a + 1.0, x).unwrap();
                    let rhs: f64 = (0..=n).map(|k| laguerre_l_assoc(k, a, x).unwrap()).sum();
                    let scale = (0..=n)
                        .map(|k| laguerre_l_assoc(k, a, x).unwrap().abs())
                        .fold(1.0_f64, f64::max);
                    assert!(
                        (lhs - rhs).abs() / scale < 1e-11,
                        "L_{n}^({a}+1)({x}) = {lhs}, sum = {rhs}"
                    );
                }
            }
        }
    }

    // ================================================================
    // 8. error cases
    // ================================================================

    #[test]
    fn negative_degree_is_an_error() {
        assert!(hermite_h(-1, 0.5).is_err());
        assert!(hermite_he(-3, 0.5).is_err());
        assert!(laguerre_l(-1, 0.5).is_err());
        assert!(laguerre_l_assoc(-1, 0.5, 0.5).is_err());
        assert!(chebyshev_t(-1, 0.5).is_err());
        assert!(chebyshev_u(-1, 0.5).is_err());
        assert!(gegenbauer_c(-1, 1.0, 0.5).is_err());
        assert!(jacobi_p(-1, 0.0, 0.0, 0.5).is_err());
    }

    #[test]
    fn non_finite_argument_is_an_error() {
        for &bad in &[f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(hermite_h(2, bad).is_err());
            assert!(hermite_he(2, bad).is_err());
            assert!(laguerre_l(2, bad).is_err());
            assert!(laguerre_l_assoc(2, 0.5, bad).is_err());
            assert!(chebyshev_t(2, bad).is_err());
            assert!(chebyshev_u(2, bad).is_err());
            assert!(gegenbauer_c(2, 1.0, bad).is_err());
            assert!(jacobi_p(2, 0.0, 0.0, bad).is_err());
        }
    }

    #[test]
    fn out_of_range_parameters_are_errors() {
        // generalised Laguerre needs alpha > -1
        assert!(laguerre_l_assoc(3, -1.0, 1.0).is_err());
        assert!(laguerre_l_assoc(3, -2.5, 1.0).is_err());
        assert!(laguerre_l_assoc(3, f64::NAN, 1.0).is_err());
        assert!(laguerre_l_assoc(3, -0.999, 1.0).is_ok());

        // Gegenbauer needs alpha > -1/2
        assert!(gegenbauer_c(3, -0.5, 0.5).is_err());
        assert!(gegenbauer_c(3, -1.0, 0.5).is_err());
        assert!(gegenbauer_c(3, f64::INFINITY, 0.5).is_err());
        assert!(gegenbauer_c(3, -0.499, 0.5).is_ok());

        // Jacobi needs alpha > -1 and beta > -1
        assert!(jacobi_p(3, -1.0, 0.0, 0.5).is_err());
        assert!(jacobi_p(3, 0.0, -1.0, 0.5).is_err());
        assert!(jacobi_p(3, -4.0, -4.0, 0.5).is_err());
        assert!(jacobi_p(3, f64::NAN, 0.0, 0.5).is_err());
        assert!(jacobi_p(3, -0.999, -0.999, 0.5).is_ok());
    }

    /// The error messages have to name the argument and say what is
    /// wrong — an `Err("")` would satisfy the tests above.
    #[test]
    fn error_messages_are_actionable() {
        let e = hermite_h(-2, 1.0).unwrap_err();
        assert!(e.contains("hermite_h") && e.contains("-2"), "{e}");
        let e = gegenbauer_c(2, -0.75, 0.5).unwrap_err();
        assert!(e.contains("alpha") && e.contains("-1/2"), "{e}");
        let e = laguerre_l_assoc(2, -3.0, 0.5).unwrap_err();
        assert!(e.contains("alpha") && e.contains("> -1"), "{e}");
        let e = jacobi_p(2, 0.0, -2.0, 0.5).unwrap_err();
        assert!(e.contains("beta"), "{e}");
    }

    /// `alpha = 0` is the documented degenerate Gegenbauer case: the
    /// standardisation collapses to zero for every positive degree.
    /// Pinned so nobody "fixes" it into a silent Chebyshev.
    #[test]
    fn gegenbauer_alpha_zero_is_degenerate_not_chebyshev() {
        for n in 1..8 {
            assert_eq!(gegenbauer_c(n, 0.0, 0.6).unwrap(), 0.0, "C_{n}^0");
        }
        assert_eq!(gegenbauer_c(0, 0.0, 0.6).unwrap(), 1.0);
        // The renormalised limit is the Chebyshev one:
        //     lim_{a->0} (n/a) C_n^a(x) = 2 T_n(x).
        // A single small `a` with a single tolerance would be a weak
        // check, because the limit error is only O(a) and any tolerance
        // loose enough to pass would also pass a badly wrong constant.
        // Instead assert the *order* of convergence: the error must
        // both be bounded by ~4a and fall by a factor of ten for every
        // factor of ten in `a`. That pins the limit and its rate.
        let worst_err = |a: f64| -> f64 {
            (1..8)
                .map(|n| {
                    let scaled = n as f64 / a * gegenbauer_c(n, a, 0.6).unwrap();
                    (scaled - 2.0 * chebyshev_t(n, 0.6).unwrap()).abs()
                })
                .fold(0.0_f64, f64::max)
        };
        let errs: Vec<f64> = [1e-4, 1e-5, 1e-6].iter().map(|&a| worst_err(a)).collect();
        for (i, &a) in [1e-4, 1e-5, 1e-6].iter().enumerate() {
            assert!(errs[i] < 5.0 * a, "limit error {} not O(a) at a={a}", errs[i]);
        }
        for w in errs.windows(2) {
            let ratio = w[0] / w[1];
            assert!(
                (8.0..12.0).contains(&ratio),
                "convergence is not first order: ratio {ratio}"
            );
        }
    }

    // ================================================================
    // 9. the recurrences themselves, as stated in the docs
    // ================================================================

    #[test]
    fn recurrences_are_satisfied_across_degree() {
        for &x in &[-1.4, -0.3, 0.0, 0.55, 2.2] {
            for n in 1..18 {
                let nf = n as f64;
                let scale = |a: f64, b: f64| a.abs().max(b.abs()).max(1.0);

                let (hm, h, hp) = (
                    hermite_h(n - 1, x).unwrap(),
                    hermite_h(n, x).unwrap(),
                    hermite_h(n + 1, x).unwrap(),
                );
                assert!(
                    (hp - (2.0 * x * h - 2.0 * nf * hm)).abs() / scale(hp, h) < 1e-12,
                    "H recurrence n={n}"
                );

                let (em, e, ep) = (
                    hermite_he(n - 1, x).unwrap(),
                    hermite_he(n, x).unwrap(),
                    hermite_he(n + 1, x).unwrap(),
                );
                assert!(
                    (ep - (x * e - nf * em)).abs() / scale(ep, e) < 1e-12,
                    "He recurrence n={n}"
                );

                let (lm, l, lp) = (
                    laguerre_l(n - 1, x).unwrap(),
                    laguerre_l(n, x).unwrap(),
                    laguerre_l(n + 1, x).unwrap(),
                );
                assert!(
                    ((nf + 1.0) * lp - ((2.0 * nf + 1.0 - x) * l - nf * lm)).abs() / scale(lp, l)
                        < 1e-11,
                    "L recurrence n={n}"
                );

                let (tm, t, tp) = (
                    chebyshev_t(n - 1, x).unwrap(),
                    chebyshev_t(n, x).unwrap(),
                    chebyshev_t(n + 1, x).unwrap(),
                );
                assert!(
                    (tp - (2.0 * x * t - tm)).abs() / scale(tp, t) < 1e-12,
                    "T recurrence n={n}"
                );
            }
        }
    }
}

