//! Bessel functions of **complex argument**, integer order.
//!
//! This closes the item deferred at the original DLMF milestone. It was
//! postponed then because the obvious route — porting AMOS/TOMS 644 —
//! needs `num-complex` and `num-traits`, and this project takes no
//! external dependencies. With [`crate::complex::Complex64`] in place
//! that objection is gone, and the algorithm turns out not to need AMOS
//! at all.
//!
//! # Miller's recurrence works unchanged for complex z
//!
//! The real-argument implementation in [`crate::bessel`] uses downward
//! recurrence normalised by an identity. **Both ingredients hold for
//! complex argument**, which is why no new algorithm is required:
//!
//! * the three-term recurrence
//!   `J_{n-1}(z) + J_{n+1}(z) = (2n/z) J_n(z)`
//!   (DLMF 10.6.1, <https://dlmf.nist.gov/10.6.E1>) is an identity in
//!   `z`, real or not;
//! * `J_n(z) ~ (z/2)^n / n!` for fixed `z`, so the wanted solution still
//!   decays with order while the unwanted one grows — the whole basis of
//!   recurring downward;
//! * the normalisation `J_0(z) + 2[J_2(z) + J_4(z) + ...] = 1` follows
//!   from the generating function `exp((z/2)(t - 1/t)) = sum J_n(z) t^n`
//!   evaluated at `t = 1`, where the left side is `exp(0) = 1`. Also an
//!   identity in `z`.
//!
//! # Accuracy: FOUR laws, not one
//!
//! **This section was wrong until it was measured properly, and the way
//! it was wrong is worth stating plainly.** It described a single law —
//! the one governing `J` — and left the reader to assume it covered the
//! module. It does not. `J` and `I` come from Miller recurrence; `Y`
//! comes from an ascending series and `K` is assembled from `J` and `Y`
//! at imaginary argument. A series and a recurrence fail in different
//! places, so there are four laws:
//!
//! ```text
//!   relative error ~ 1e-16 * exp(L)
//!
//!   J_n:  L = |Im z|                        worst up the imaginary axis
//!   I_n:  L = |Re z|                        worst along the real axis
//!   Y_n:  L = |z| - |Im z|                  worst along the real axis
//!   K_n:  L = max(2|Re z|, |z|) + Re z      worst along the POSITIVE real axis
//! ```
//!
//! **Those four laws describe the individual routes, and since Stage 19
//! they are no longer what these functions deliver.** Each of `J`, `Y`
//! and `K` now has a second route that fails where the first one does
//! and succeeds where it does not, and the better estimate is taken:
//!
//! * `J` near the imaginary axis by `J_nu(z) = i^nu I_nu(-iz)`, where
//!   `-iz` is near the real axis and the `1/z` expansion for `I` has no
//!   cancellation;
//! * `Y` near the imaginary axis by
//!   `Y_n(z) = i^(n+1) I_n(w) - (2/pi) i^(-n) K_n(w)`, `w = -iz`, which
//!   replaces the upward recurrence in `n` — the direction that
//!   destroys `Y` there, since its content is mostly the recessive `I`;
//! * `Y` along the real axis, and `K` everywhere, by their own `1/z`
//!   expansions, which are single series with nothing to cancel;
//! * `J` and `Y` in the wedge either side of the **negative real
//!   axis**, by continuing the Hankel expansions from the positive one
//!   (DLMF 10.11.3, 10.11.4). The two `1/z` Hankel expansions have
//!   sectors that end at `arg z = pi`, so the cut is the one direction
//!   neither reaches directly; `w = -z` puts it back where both are at
//!   their best, and the continuation is exact algebra.
//!
//! Measured after that, the J-Y Wronskian residual over `|z|` from 5 to
//! 40 and `arg z` across the upper half plane is **1e-10 or better and
//! mostly below 1e-15**, against 1e-1 before.
//!
//! Near the cut that Wronskian is the wrong instrument — it is
//! dominated there by the exponentially **recessive** Hankel member and
//! so measures the Stokes phenomenon rather than the answer. The right
//! test is the exact continuation identities,
//! `J_n(x e^{i pi}) = (-1)^n J_n(x)` and
//! `Y_n(x e^{i pi}) = (-1)^n [Y_n(x) + 2i J_n(x)]`, which relate a
//! point on the cut to one on the positive real axis. Against those,
//! `J` is exact and `Y` is 1e-14 out to `|z| = 300`; the wedge either
//! side is 1e-12 or better. The branch jump is still exactly
//! `4i(-1)^n J_n`, and a test says so — widening the coverage across a
//! cut is only correct if the cut stays where it was. `Y_0(40)` was wrong in its
//! first digit and is now exact to 1e-15; `K_0(20)` was out by `8e8`
//! and is now exact to 2e-16.
//!
//! The laws below are kept because they still describe what each route
//! costs, and because the selector uses them to choose. They are pinned
//! by `integer_order_accuracy_laws_hold` as **upper bounds**, which is
//! all they now are:
//!
//! | x (real) | 1 | 10 | 20 | 30 | 35 |
//! |---|---|---|---|---|---|
//! | `J_0(x)`  | 1e-16 | 3e-16 | 5e-16 | 2e-15 | 7e-16 |
//! | `J_0(ix)` | 2e-16 | 1e-13 | 5e-9  | 5e-5  | 3e-3  |
//! | `I_0(x)`  | 2e-16 | 1e-13 | 5e-9  | 5e-5  | 3e-3  |
//! | `Y_0(x)`  | 1e-15 | 3e-12 | 3e-8  | 2e-4  | 6e-2  |
//! | `K_0(x)`  | 7e-16 | 3e-5  | 8e8   | 5e21  | 7e27  |
//!
//! Those numbers are what the ROUTES cost, not what the functions
//! deliver — the table predates Stage 19 and is kept as the record of
//! why the second routes exist. `I` and `J` are the same function at
//! right angles, which is why their columns match to the last digit.
//!
//! `K`'s exponent has three terms because `K_n(z)` is built from
//! `J_n(iz) + i Y_n(iz)`: that `J` is amplified by `exp(|Re z|)`
//! relative to an answer of size `exp(-Re z)`, on top of the ordinary
//! series cancellation. The same shape appears at non-integer order,
//! where the law is the simpler `|z| + Re z` because there is no Miller
//! step in the path.
//!
//! **Two earlier claims in this note were wrong and are corrected here.**
//! The first said the result was "worthless past `|Im z| ~ 20`" — it is
//! not; at `|Im z| = 25` five or six digits of `J` remain. The second
//! said the error "barely depends on `Re z`" — true of `J`, false of
//! `Y` and badly false of `K`. Both came from measuring `J` alone, via
//! the generating-function identity, which involves no `Y` at all. A
//! Hankel asymptotic test at `x = 40` was what finally exposed it.
//!
//! # `Y_n` and `K_n` need a different method
//!
//! `Y_n` has a **logarithmic branch point** at the origin, so no amount
//! of recurrence produces it from `J_n` alone — the earlier version of
//! this module said as much and left them out. They are here now, by the
//! route that difficulty actually dictates:
//!
//! * `Y_0` and `Y_1` from the ascending series (DLMF 10.8.1), which
//!   carries the `ln(z/2) J_n(z)` term and digamma coefficients;
//! * higher orders by **upward** recurrence, which for `Y` is the stable
//!   direction — precisely the opposite of `J`, because `Y_n` *grows*
//!   with order while `J_n` decays. Using the same downward sweep for
//!   both would destroy one of them;
//! * `K_n` by the identity
//!   `K_n(z) = (pi/2) i^{n+1} [J_n(iz) + i Y_n(iz)]` (DLMF 10.27.8),
//!   so it needs no third algorithm.
//!
//! **The branch cut matters.** `Y_n` and `K_n` inherit the cut of `ln`
//! along the negative real axis and are discontinuous across it. `J_n`
//! and `I_n` are entire and have no such restriction.

use crate::complex::Complex64 as C;
use spec_math::cephes64::rgamma;

/// `J_0(z) ... J_{n_max}(z)` in one pass, for complex `z`.
///
/// # Errors
/// A non-finite `z`.
///
/// # Examples
/// ```
/// use special_functions::bessel_complex::bessel_j_array_c;
/// use special_functions::complex::Complex64 as C;
/// // At z = 0 only J_0 survives.
/// let j = bessel_j_array_c(4, C::ZERO).unwrap();
/// assert!((j[0].re - 1.0).abs() < 1e-15 && j[0].im.abs() < 1e-15);
/// assert!(j[1..].iter().all(|v| v.abs() < 1e-15));
/// ```
pub fn bessel_j_array_c(n_max: usize, z: C) -> Result<Vec<C>, String> {
    if !z.is_finite() {
        return Err(format!("bessel_j_array_c: z must be finite, got {z:?}"));
    }
    if z.abs() == 0.0 {
        let mut v = vec![C::ZERO; n_max + 1];
        v[0] = C::ONE;
        return Ok(v);
    }

    // Same seeding rule as the real case, driven by |z|: the start must
    // sit well above BOTH the wanted order and |z|, because J_n(z) only
    // begins to decay once n exceeds |z|.
    let az = z.abs();
    let start = n_max + 30 + (1.5 * az + 12.0 * az.sqrt()) as usize;

    let mut jp1 = C::ZERO; // J_{k+1}
    let mut j = C::new(1.0e-290, 0.0); // J_k, arbitrary seed
    let mut out = vec![C::ZERO; n_max + 1];
    // Accumulates J_0 + 2(J_2 + J_4 + ...) in the unnormalised scale.
    let mut sum = C::ZERO;
    let inv_z = z.inv();

    for k in (0..=start).rev() {
        let jm1 = inv_z * j * (2 * (k + 1)) as f64 - jp1;
        jp1 = j;
        j = jm1;

        if j.abs() > 1.0e250 {
            let s = 1.0e-250;
            j = j * s;
            jp1 = jp1 * s;
            sum = sum * s;
            for e in out.iter_mut() {
                *e = *e * s;
            }
        }
        if k <= n_max {
            out[k] = j;
        }
        if k.is_multiple_of(2) {
            sum = sum + if k == 0 { j } else { j * 2.0 };
        }
    }

    let s = sum.abs();
    if s == 0.0 || !sum.is_finite() {
        return Err(format!(
            "bessel_j_array_c: normalisation failed for n_max={n_max}, z={z:?}"
        ));
    }
    let scale = sum.inv();
    for e in out.iter_mut() {
        *e = *e * scale;
    }
    Ok(out)
}

/// A single `J_n(z)` for integer `n >= 0` and complex `z`.
///
/// # Errors
/// Negative order, or a non-finite `z`.
///
/// # Examples
/// ```
/// use special_functions::bessel_complex::bessel_j_c;
/// use special_functions::complex::Complex64 as C;
/// // Real argument must reproduce the real routine.
/// let v = bessel_j_c(0, C::real(2.404_825_557_695_773)).unwrap();
/// assert!(v.abs() < 1e-12);
/// ```
pub fn bessel_j_c(n: i32, z: C) -> Result<C, String> {
    /* the documented error contract must hold on the WHOLE domain, so
     * the order guard comes before any route (the asymptotic routes
     * carry no guard of their own and used to leak values for n < 0
     * in the regions they cover) */
    if n < 0 {
        return Err(format!("bessel_j_c: order n must be >= 0, got {n}"));
    }
    if let Some(v) = j_via_i(n, z) {
        return Ok(v);
    }
    if let Some(v) = j_via_asym(n, z) {
        return Ok(v);
    }
    if let Some(v) = j_via_debye(n, z) {
        return Ok(v);
    }
    if let Some(v) = j_via_airy(n, z) {
        return Ok(v);
    }
    Ok(bessel_j_array_c(n as usize, z)?[n as usize])
}

/// The modified Bessel function `I_n(z)` for complex `z`.
///
/// Obtained from `I_n(z) = i^{-n} J_n(i z)` (DLMF 10.27.6), which is an
/// identity rather than a separate algorithm — the whole reason a
/// complex `J` is worth having.
///
/// # Errors
/// As [`bessel_j_c`].
///
/// # Examples
/// ```
/// use special_functions::bessel_complex::bessel_i_c;
/// use special_functions::complex::Complex64 as C;
/// // I_0(0) = 1
/// let v = bessel_i_c(0, C::ZERO).unwrap();
/// assert!((v.re - 1.0).abs() < 1e-15);
/// ```
pub fn bessel_i_c(n: i32, z: C) -> Result<C, String> {
    if n < 0 {
        return Err(format!("bessel_i_c: order n must be >= 0, got {n}"));
    }
    let j = bessel_j_c(n, C::I * z)?;
    Ok(j * i_pow(-n))
}

/// `i^k` for any integer `k`, by cycling rather than by `powi` — exact,
/// with no rounding at all.
fn i_pow(k: i32) -> C {
    match k.rem_euclid(4) {
        0 => C::ONE,
        1 => C::I,
        2 => C::real(-1.0),
        _ => C::new(0.0, -1.0),
    }
}

/// `psi(m)` for a positive integer, exactly: `psi(1) = -gamma` and
/// `psi(m+1) = psi(m) + 1/m`.
fn digamma_int(m: usize) -> f64 {
    // Euler-Mascheroni, to full double precision.
    const GAMMA: f64 = 0.577_215_664_901_532_9;
    let mut v = -GAMMA;
    for k in 1..m {
        v += 1.0 / k as f64;
    }
    v
}

/// `Y_n(z)` for `n = 0` or `n = 1` by the ascending series, DLMF 10.8.1:
///
/// ```text
///  Y_n(z) = -(1/pi) (z/2)^-n sum_{k=0}^{n-1} (n-k-1)!/k! (z^2/4)^k
///         + (2/pi) ln(z/2) J_n(z)
///         - (1/pi) (z/2)^n sum_{k>=0} [psi(k+1)+psi(n+k+1)] (-z^2/4)^k
///                                      / (k! (n+k)!)
/// ```
fn y_series(n: usize, z: C) -> Result<C, String> {
    let half = z * 0.5;
    let q = half * half; // (z/2)^2
    let jn = bessel_j_c(n as i32, z)?;
    let inv_pi = 1.0 / std::f64::consts::PI;

    // finite sum, empty for n = 0
    let mut finite = C::ZERO;
    if n >= 1 {
        let mut qk = C::ONE; // (z^2/4)^k
        let mut fact_k = 1.0f64;
        for k in 0..n {
            let coeff = factorial(n - k - 1) / fact_k;
            finite = finite + qk * coeff;
            qk = qk * q;
            fact_k *= (k + 1) as f64;
        }
        // multiply by (z/2)^-n
        let mut p = C::ONE;
        for _ in 0..n {
            p = p * half;
        }
        finite = finite * p.inv();
    }

    // infinite sum
    let mut infinite = C::ZERO;
    let mut term_pow = C::ONE; // (-z^2/4)^k
    let neg_q = q * -1.0;
    let mut fact_k = 1.0f64;
    let mut fact_nk = factorial(n);
    for k in 0..200 {
        let coeff = (digamma_int(k + 1) + digamma_int(n + k + 1)) / (fact_k * fact_nk);
        let add = term_pow * coeff;
        infinite = infinite + add;
        if k > 4 && add.abs() <= 1e-18 * infinite.abs().max(1e-300) {
            break;
        }
        term_pow = term_pow * neg_q;
        fact_k *= (k + 1) as f64;
        fact_nk *= (n + k + 1) as f64;
    }
    let mut p = C::ONE;
    for _ in 0..n {
        p = p * half;
    }
    infinite = infinite * p;

    Ok(finite * -inv_pi + half.ln() * jn * (2.0 * inv_pi) - infinite * inv_pi)
}

/// `k!` as an `f64`. Only ever called with small `k` here.
fn factorial(k: usize) -> f64 {
    (1..=k).map(|i| i as f64).product::<f64>().max(1.0)
}

/// `Y_0(z) ... Y_{n_max}(z)`, complex argument, integer order.
///
/// # Errors
/// A non-finite `z`, or `z == 0` where `Y` is infinite.
///
/// # Examples
/// ```
/// use special_functions::bessel_complex::bessel_y_array_c;
/// use special_functions::complex::Complex64 as C;
/// // Y_0(1) is about 0.08825696
/// let y = bessel_y_array_c(1, C::real(1.0)).unwrap();
/// assert!((y[0].re - 0.088_256_964_215_676_96).abs() < 1e-10);
/// ```
pub fn bessel_y_array_c(n_max: usize, z: C) -> Result<Vec<C>, String> {
    if !z.is_finite() {
        return Err(format!("bessel_y_array_c: z must be finite, got {z:?}"));
    }
    if z.abs() == 0.0 {
        return Err(
            "bessel_y_array_c: Y_n has a logarithmic singularity at z = 0".to_string()
        );
    }
    let y0 = y_series(0, z)?;
    if n_max == 0 {
        return Ok(vec![y0]);
    }
    let y1 = y_series(1, z)?;
    let mut out = Vec::with_capacity(n_max + 1);
    out.push(y0);
    out.push(y1);
    // Upward recurrence is the STABLE direction for Y, because Y_n grows
    // with order. (For J it is the unstable one — the two functions need
    // opposite sweeps, which is the whole reason they cannot share an
    // implementation.)
    let inv_z = z.inv();
    for n in 1..n_max {
        let next = inv_z * out[n] * (2 * n) as f64 - out[n - 1];
        out.push(next);
    }
    Ok(out)
}

/// A single `Y_n(z)`.
///
/// # Errors
/// Negative order, non-finite `z`, or `z == 0`.
pub fn bessel_y_c(n: i32, z: C) -> Result<C, String> {
    if n < 0 {
        return Err(format!("bessel_y_c: order n must be >= 0, got {n}"));
    }
    if let Some(v) = y_via_ik(n, z) {
        return Ok(v);
    }
    if let Some(v) = y_via_asym(n, z) {
        return Ok(v);
    }
    if let Some(v) = y_via_debye(n, z) {
        return Ok(v);
    }
    if let Some(v) = y_via_airy(n, z) {
        return Ok(v);
    }
    Ok(bessel_y_array_c(n as usize, z)?[n as usize])
}

/// `J_n(z)` by the `1/z` Hankel expansion, where Miller's normalisation
/// has cancelled.
///
/// Miller loses `exp(|Im z|)`. Near the imaginary axis [`j_via_i`]
/// rotates that away, but in the wedge either side of the **negative
/// real axis** — where `|Im z|` is large and yet not larger than
/// `|Re z|` — neither applies, and the expansion is what is left.
/// Since Stage 20 it reaches that wedge, by continuing from the
/// positive real axis.
fn j_via_asym(n: i32, z: C) -> Option<C> {
    if !z.is_finite() || z.abs() == 0.0 {
        return None;
    }
    let (v, e) = crate::bessel_cnu_large::j_asym(C::real(n as f64), z)?;
    if !v.is_finite() {
        return None;
    }
    let loss = z.im.abs();
    let miller_err = if loss > 700.0 { f64::INFINITY } else { 1e-16 * loss.exp() };
    (e < miller_err).then_some(v)
}

/// `Y_n(z)` by the `1/z` Hankel expansion, where the ascending series
/// has cancelled away its digits.
///
/// The other half of the same story: near the imaginary axis it is the
/// *recurrence* that fails and [`y_via_ik`] answers; along the **real**
/// axis it is the ascending series, whose terms are `exp(|z|)` while
/// `Y` is `O(1)`. `Y_0(40)` was wrong in its first digit.
///
/// Stage 14 already fixed that for the scaled routines. This brings the
/// fix to `bessel_y_c` itself, which everything else in this module is
/// built on — including `bessel_k_c`, and so `hankel`, and so the rest.
fn y_via_asym(n: i32, z: C) -> Option<C> {
    if !z.is_finite() || z.abs() == 0.0 {
        return None;
    }
    let (v, e) = crate::bessel_cnu_large::y_asym(C::real(n as f64), z)?;
    if !v.is_finite() {
        return None;
    }
    let loss = z.abs() - z.im.abs();
    let series_err = if loss > 700.0 { f64::INFINITY } else { 1e-16 * loss.exp() };
    (e < series_err).then_some(v)
}

/// A value with the estimate of its relative error, or nothing.
type Valued = Option<(C, f64)>;

/// `J_n(z)` and `Y_n(z)` by the uniform **Airy-type** expansion of DLMF
/// 10.20, for real `z` either side of the turning point `z = n`.
///
/// The Debye expansions lose their grip as `z/n` approaches 1 — their
/// coefficients are polynomials in `1/sqrt(1 - (z/n)^2)` — and this is
/// what covers the gap between them. Measured, `Y_20(26)` was 9.9e-6
/// without it and 1e-14 with it.
fn jy_via_airy(n: i32, z: C) -> (Valued, Valued) {
    if n <= 0 || z.im != 0.0 || z.re <= 0.0 {
        return (None, None);
    }
    let (j, y) = crate::airy_uniform::jy_airy(n as f64, z.re);
    let f = |u: crate::debye::Uniform| {
        (u.value.is_finite() && u.err.is_finite()).then_some((u.value, u.err.max(1e-16)))
    };
    (j.and_then(f), y.and_then(f))
}

fn j_via_airy(n: i32, z: C) -> Option<C> {
    let (v, e) = jy_via_airy(n, z).0?;
    let loss = z.im.abs();
    let miller = if loss > 700.0 { f64::INFINITY } else { 1e-16 * loss.exp() };
    (e < miller).then_some(v)
}

fn y_via_airy(n: i32, z: C) -> Option<C> {
    let (v, e) = jy_via_airy(n, z).1?;
    let loss = z.abs() - z.im.abs();
    let series = if loss > 700.0 { f64::INFINITY } else { 1e-16 * loss.exp() };
    (e < series).then_some(v)
}

/// `J_n(z)` and `Y_n(z)` by the **Debye** expansions, for the band
/// where nothing else applies.
///
/// The `1/z` expansions refuse when `|4 n^2|` is not small compared
/// with `|z|`; the ascending series has cancelled by then; and `z/n` is
/// too far from 1 for the turning-point expansion. That band —
/// `|z|` a few times `n` — had no method at all. At `n = 20, z = 60`
/// it made `Y_20(60)` come back as `1e8`; it is now 2e-14.
fn jy_via_debye(n: i32, z: C) -> (Valued, Valued) {
    if n < 0 || !z.is_finite() || z.abs() == 0.0 {
        return (None, None);
    }
    let (j, y) = crate::debye::jy_debye_c(C::real(n as f64), z);
    let f = |u: crate::debye::Uniform| u.value.is_finite().then_some((u.value, u.err.max(1e-16)));
    (j.and_then(f), y.and_then(f))
}

fn j_via_debye(n: i32, z: C) -> Option<C> {
    let (v, e) = jy_via_debye(n, z).0?;
    let loss = z.im.abs();
    let miller = if loss > 700.0 { f64::INFINITY } else { 1e-16 * loss.exp() };
    (e < miller).then_some(v)
}

fn y_via_debye(n: i32, z: C) -> Option<C> {
    let (v, e) = jy_via_debye(n, z).1?;
    let loss = z.abs() - z.im.abs();
    let series = if loss > 700.0 { f64::INFINITY } else { 1e-16 * loss.exp() };
    (e < series).then_some(v)
}

/// `J_n(z)` near the **imaginary axis**, from `I` on the rotated
/// argument.
///
/// Miller's recurrence normalises by `J_0 + 2(J_2 + J_4 + ...) = 1`,
/// and up the imaginary axis the individual terms grow like
/// `exp(|Im z|)` while their sum is 1 — the loss documented since Stage
/// 13. But `J_nu(z) = i^nu I_nu(-iz)` (DLMF 10.27.6), and `-iz` is then
/// near the **real** axis, where the `1/z` expansion for `I` has no
/// cancellation at all.
///
/// The same rotation that makes `Y` hard makes `J` easy, and the same
/// recursion-free route serves both.
fn j_via_i(n: i32, z: C) -> Option<C> {
    if z.im.abs() <= z.re.abs() || !z.is_finite() {
        return None;
    }
    let (zz, conjugated) = if z.im >= 0.0 { (z, false) } else { (z.conj(), true) };
    let w = zz * (C::I * -1.0);
    let (i_val, e) = crate::bessel_cnu_large::i_asym(C::real(n as f64), w)?;
    let j = i_pow(n) * i_val;
    if !j.is_finite() {
        return None;
    }
    let loss = z.im.abs();
    let miller_err = if loss > 700.0 { f64::INFINITY } else { 1e-16 * loss.exp() };
    (e < miller_err).then_some(if conjugated { j.conj() } else { j })
}

/// `K_n(z)` by its own asymptotic expansion, when the identity below
/// would cancel.
///
/// `bessel_k_c` is built on `K_n(z) = (pi/2) i^(n+1)[J_n(iz) + iY_n(iz)]`,
/// and on the real axis that identity **cancels by construction**:
/// `J_n(ix)` is `I_n(x)` and `Y_n(ix)` contributes `i I_n(x)` too, so
/// the two `I` parts — of size `exp(x)` — annihilate and leave `K`, of
/// size `exp(-x)`. That is not a defect of any ingredient; it is the
/// identity being the wrong way to compute a recessive function.
///
/// Measured, it costs everything: `K_0(10)` was wrong by 2.8e-5 and
/// `K_0(20)` by `8e8`. This route takes the `1/z` expansion of DLMF
/// 10.40.2 instead, which is a single series with no cancellation in
/// it, and is used whenever its truncation estimate beats the
/// identity's `exp(max(2|Re z|, |z|) + Re z)` loss.
///
/// Recursion-free by construction, like [`y_via_ik`], and for the same
/// reason.
fn k_via_asym(n: i32, z: C) -> Option<C> {
    if !z.is_finite() || z.abs() == 0.0 {
        return None;
    }
    let (v, e) = crate::bessel_cnu_large::k_asym(C::real(n as f64), z)?;
    if !v.is_finite() {
        return None;
    }
    let loss = (2.0 * z.re.abs()).max(z.abs()) + z.re;
    let identity_err = if loss > 700.0 { f64::INFINITY } else { 1e-16 * loss.exp() };
    (e < identity_err).then_some(v)
}

/// `Y_n(z)` near the **imaginary axis**, from `I` and `K` on the
/// rotated argument instead of from the upward recurrence.
///
/// # The defect this exists to fix
///
/// [`bessel_y_array_c`] builds `Y_n` by recurring **upward** in `n`
/// from `Y_0` and `Y_1`. That is the stable direction for real
/// argument. It is not near the imaginary axis, and the reason is
/// visible in the connection formula below: at `z = iy` the content of
/// `Y_n` is mostly `I_n(y)`, and `I` is the **recessive** solution of
/// that recurrence in `n` — the direction which destroys it.
///
/// Stage 18 found this by accident and the Wronskian settled it: at
/// `n = 2, z = 29.4 e^{1.6i}` the `1/z` expansion closes the J-Y
/// Wronskian to 7e-26 and the recurrence to **4.5e-6**. The accuracy
/// law recorded in Stage 13 said 1e-16 and was wrong.
///
/// # The route
///
/// With `w = -iz` (which puts `Re w >= 0` for `z` in the upper half
/// plane),
///
/// ```text
///   Y_n(z) = i^(n+1) I_n(w) - (2/pi) i^(-n) K_n(w)
/// ```
///
/// and for `Im z < 0` the conjugate, since `Y_n` has real coefficients
/// away from its cut. `I` dominates `K` by `exp(2 Re w)` here, so
/// forming the combination costs nothing — the opposite of the
/// recurrence.
///
/// `I` and `K` come from the **asymptotic** routines of
/// [`crate::bessel_cnu_large`], which are self-contained series in
/// `1/w`. That is not an accident of convenience: `bessel_k_c` is built
/// on `bessel_y_c`, so anything that reached back into this module
/// through the scaled or non-integer routines would recurse forever.
///
/// Returns `None` when the expansions do not apply — small `|z|`, or
/// away from the imaginary axis — and the caller falls back to the
/// recurrence, which is sound there.
fn y_via_ik(n: i32, z: C) -> Option<C> {
    // Only where the recurrence is actually the wrong tool: nearer the
    // imaginary axis than the real one.
    if z.im.abs() <= z.re.abs() || !z.is_finite() {
        return None;
    }
    let (zz, conjugated) = if z.im >= 0.0 { (z, false) } else { (z.conj(), true) };
    let w = zz * (C::I * -1.0);
    let nu = C::real(n as f64);
    let (i_val, e_i) = crate::bessel_cnu_large::i_asym(nu, w)?;
    let (k_val, e_k) = crate::bessel_cnu_large::k_asym(nu, w)?;
    let y = i_pow(n + 1) * i_val - i_pow(-n) * k_val * (2.0 / std::f64::consts::PI);
    if !y.is_finite() {
        return None;
    }
    // Only take this route if it beats what the recurrence would give.
    // The recurrence's loss is the series' own `|z| - |Im z|` plus, for
    // an order it actually recurs to, the instability — measured as
    // `exp(|Im z|)`.
    let recurrence_loss = if n >= 2 {
        (z.abs() - z.im.abs()).max(z.im.abs())
    } else {
        z.abs() - z.im.abs()
    };
    let recurrence_err = if recurrence_loss > 700.0 {
        f64::INFINITY
    } else {
        1e-16 * recurrence_loss.exp()
    };
    let here = e_i.max(e_k);
    (here < recurrence_err).then_some(if conjugated { y.conj() } else { y })
}

/// The modified Bessel function of the second kind, `K_n(z)`, complex
/// argument.
///
/// From `K_n(z) = (pi/2) i^{n+1} [J_n(iz) + i Y_n(iz)]` (DLMF 10.27.8) —
/// the Hankel function `H^(1)_n` evaluated on the rotated argument, so
/// no third algorithm is needed.
///
/// # Errors
/// Negative order, non-finite `z`, or `z == 0`.
///
/// # Examples
/// ```
/// use special_functions::bessel_complex::bessel_k_c;
/// use special_functions::complex::Complex64 as C;
/// // K_0(1) is about 0.4210244382
/// let k = bessel_k_c(0, C::real(1.0)).unwrap();
/// assert!((k.re - 0.421_024_438_240_708_3).abs() < 1e-9);
/// ```
pub fn bessel_k_c(n: i32, z: C) -> Result<C, String> {
    if n < 0 {
        return Err(format!("bessel_k_c: order n must be >= 0, got {n}"));
    }
    if !z.is_finite() {
        return Err(format!("bessel_k_c: z must be finite, got {z:?}"));
    }
    if z.abs() == 0.0 {
        return Err("bessel_k_c: K_n has a singularity at z = 0".to_string());
    }
    if let Some(v) = k_via_asym(n, z) {
        return Ok(v);
    }
    // The identity rotates the argument by i, so `arg(iz) = arg(z) +
    // pi/2`. For `arg z > pi/2` that leaves the principal range and
    // `ln` inside `Y` wraps to the far side of its branch cut, putting
    // the answer on the WRONG SHEET. Measured, the I-K Wronskian
    // residual jumped from the 5e-8 its accuracy law predicts to 3e8 at
    // `arg z = 2.5`. Conjugation avoids the rotation entirely: `K_n` has
    // real coefficients, so `K_n(conj z) = conj(K_n(z))`, and `conj z`
    // has `arg` in `[-pi, -pi/2)`, where `arg(iz)` lands safely in
    // `[-pi/2, 0)`.
    // `z.im > 0.0` and not merely `arg > pi/2`: on the negative real
    // axis itself the rotation lands at `arg(iz) = -pi/2`, which is
    // safe, and conjugating there would silently move the answer to the
    // other side of the cut — a convention change, not a fix.
    if z.im > 0.0 && z.arg() > std::f64::consts::FRAC_PI_2 {
        return Ok(bessel_k_c(n, z.conj())?.conj());
    }
    let iz = C::I * z;
    let j = bessel_j_c(n, iz)?;
    let y = bessel_y_c(n, iz)?;
    Ok((j + C::I * y) * i_pow(n + 1) * (std::f64::consts::PI * 0.5))
}

// ---------------------------------------------------------------------
// Non-integer order
// ---------------------------------------------------------------------
//
// For non-integer `nu` the whole family collapses to two ascending
// series plus two reflection formulas, which is much simpler than the
// integer case — the integer case is hard precisely BECAUSE those
// reflections degenerate to 0/0 there.
//
//   J_nu(z) = (z/2)^nu sum_k (-1)^k (z^2/4)^k / (k! Gamma(nu+k+1))
//   I_nu(z) = (z/2)^nu sum_k         (z^2/4)^k / (k! Gamma(nu+k+1))
//   Y_nu(z) = [J_nu(z) cos(nu pi) - J_{-nu}(z)] / sin(nu pi)
//   K_nu(z) = (pi/2) [I_{-nu}(z) - I_nu(z)] / sin(nu pi)
//
// `1/Gamma` is taken from the vendored reciprocal gamma rather than
// dividing by `Gamma`: it is ZERO at the poles, which is exactly the
// value the series needs when `nu + k + 1` lands on a non-positive
// integer, whereas `1/Gamma(pole)` would be `1/inf` and, worse, an
// intermediate `inf` if the gamma overflowed first (`Gamma` leaves f64
// range past about 171, and large order is otherwise free here).
//
// ACCURACY. An ascending series is a different animal from the Miller
// recurrence used at integer order, and it fails in a different place —
// so the integer-order advice does NOT transfer. The largest term is of
// size `exp(|z|)`, so the cancellation is the ratio of that to the
// answer:
//
//     relative error  ~  1e-16 * exp(L)
//         L = |z| - |Im z|   for J and Y   (worst on the real axis)
//         L = |z| + Re z     for I and K   (worst on the POSITIVE real
//                                           axis, exact on the negative)
//
// Measured and pinned: `documented_accuracy_bounds_hold` asserts
// `1e-14 * exp(L)` across both families out to `|z| = 70`, and
// `examples/bessel_nu_accuracy.rs` prints the whole surface. Large
// order costs nothing — at `nu = 150` the order recurrence still closes
// to 1e-13 — because nothing cancels once `nu >> |z|`.
//
// Past those ranges, prefer the integer-order routines where the order
// permits. Uniform asymptotics for large `|z|` at non-integer order are
// NOT implemented.

/// How close to an integer `nu` must be before the reflection formulas
/// are treated as degenerate. At `sin(nu pi) ~ 1e-9` the reflection has
/// already lost nine digits, so the integer routines are better long
/// before `nu` is exactly whole.
const NEAR_INTEGER: f64 = 1e-9;

/// The shared ascending series. `alternating` selects `J` (true) or `I`.
/// The ascending series, and **how much of its own precision it spent**.
///
/// The second return is `max|term| / |sum|`: the ratio of the largest
/// quantity formed to the answer produced. That is the cancellation,
/// measured from the values rather than modelled from `exp(|z|)`, and
/// it is what makes an honest guard possible — see [`bessel_y_nu`].
fn nu_series_loss(nu: f64, z: C, alternating: bool) -> Result<(C, f64), String> {
    if !nu.is_finite() {
        return Err(format!("bessel: the order must be finite, got {nu}"));
    }
    if !z.is_finite() {
        return Err(format!("bessel: z must be finite, got {z:?}"));
    }
    if z.abs() == 0.0 {
        // z^nu is 0 for nu > 0, 1 for nu = 0, singular for nu < 0
        return if nu > 0.0 {
            Ok((C::ZERO, 1.0))
        } else if nu == 0.0 {
            Ok((C::ONE, 1.0))
        } else {
            Err("bessel: singular at z = 0 for negative order".to_string())
        };
    }

    let half = z * 0.5;
    let q = half * half;
    let step = if alternating { q * -1.0 } else { q };

    let mut sum = C::ZERO;
    let mut term_pow = C::ONE; // (+/- z^2/4)^k
    let mut fact_k = 1.0f64;
    let mut largest = 0.0f64;
    for k in 0..400 {
        let coeff = rgamma(nu + k as f64 + 1.0) / fact_k;
        let add = term_pow * coeff;
        largest = largest.max(add.abs());
        sum = sum + add;
        if k > 6 && add.abs() <= 1e-18 * sum.abs().max(1e-300) {
            break;
        }
        term_pow = term_pow * step;
        fact_k *= (k + 1) as f64;
    }
    let loss = if sum.abs() > 0.0 { (largest / sum.abs()).max(1.0) } else { f64::INFINITY };
    let pref = half.powf(nu);
    let out = pref * sum;
    if !out.is_finite() {
        return Err(format!(
            "bessel: the series overflowed for nu = {nu}, z = {z:?}"
        ));
    }
    Ok((out, loss))
}

/// The series without its loss figure.
fn nu_series(nu: f64, z: C, alternating: bool) -> Result<C, String> {
    nu_series_loss(nu, z, alternating).map(|(v, _)| v)
}

/// `J_nu(z)` for **real order** `nu` (integer or not) and complex `z`.
///
/// Evaluated by the ascending series (DLMF 10.2.2). **Accuracy falls as
/// `1e-16 exp(|z| - |Im z|)`** — worst on the real axis, essentially
/// exact up the imaginary one; see the module comment.
///
/// # Errors
/// A non-finite order or argument, a negative order at `z = 0`, or an
/// overflow in the series.
///
/// # Examples
/// ```
/// use special_functions::bessel_complex::bessel_j_nu;
/// use special_functions::complex::Complex64 as C;
/// // J_{1/2}(z) = sqrt(2/(pi z)) sin z, exactly — for complex z too.
/// let z = C::new(1.3, 0.6);
/// let got = bessel_j_nu(0.5, z).unwrap();
/// let want = (C::real(2.0 / std::f64::consts::PI) * z.inv()).powf(0.5)
///     * ((C::I * z).exp() - (C::I * z * -1.0).exp()) / (C::I * 2.0);
/// assert!((got - want).abs() < 1e-12);
/// ```
pub fn bessel_j_nu(nu: f64, z: C) -> Result<C, String> {
    nu_series(nu, z, true)
}

/// `I_nu(z)` for real order and complex `z`.
///
/// Ascending series (DLMF 10.25.2). **Accuracy falls as
/// `1e-16 exp(|z| - Re z)`** — exact along the positive real axis,
/// worst along the negative one, the mirror image of [`bessel_j_nu`].
///
/// # Errors
/// As [`bessel_j_nu`].
pub fn bessel_i_nu(nu: f64, z: C) -> Result<C, String> {
    nu_series(nu, z, false)
}

/// How much precision the reflection route spends, **measured from the
/// values**: the largest quantity either series forms, over the answer.
///
/// This is what makes the guard in [`bessel_y_nu`] possible without a
/// modelled `exp(|z|)` law. `nu_series_loss` reports each series'
/// own cancellation; the reflection then adds its own, and the product
/// is what `f64` has to survive.
fn y_nu_loss(nu: f64, z: C) -> Result<f64, String> {
    let nearest = nu.round();
    if (nu - nearest).abs() < NEAR_INTEGER { return Ok(1.0); }
    let (jp, lp) = nu_series_loss(nu, z, true)?;
    let (jm, lm) = nu_series_loss(-nu, z, true)?;
    let (s, c) = (nu * std::f64::consts::PI).sin_cos();
    let out = (jp * c - jm) * (1.0 / s);
    let combine = if out.abs() > 0.0 {
        ((jp * c).abs() + jm.abs()) / (out.abs() * s.abs())
    } else { f64::INFINITY };
    Ok(lp.max(lm) * combine.max(1.0))
}

/// `Y_nu(z)` for real order and complex `z`.
///
/// **Prefer [`crate::bessel_cnu::bessel_y_cnu`] unless `|z|` is small.**
/// This is one route, not a selector: for non-integer `nu` it forms
/// `[J_nu cos(nu pi) - J_{-nu}] / sin(nu pi)`, and `J_{-nu}` comes from
/// an ascending series whose terms reach `exp(|z|)`. Past `|z| ~ 30`
/// that has consumed every digit and the result is not merely
/// inaccurate but wrong — 3.09e4 relative at `nu = 36.8, z = 54.46`.
/// The selector routes around it and is accurate to 4.8e-13 there.
///
/// Non-integer order uses the reflection
/// `Y_nu = [J_nu cos(nu pi) - J_{-nu}] / sin(nu pi)` (DLMF 10.2.3).
/// **Near an integer that formula is 0/0**, so orders within
/// `1e-9` of a whole number are handed to the integer implementation,
/// which uses the logarithmic series instead. The switch is not a
/// convenience: at `sin(nu pi) ~ 1e-9` the reflection has already lost
/// nine digits to cancellation.
///
/// **Accuracy** is that of [`bessel_j_nu`], which it is built from.
///
/// # Errors
/// As [`bessel_j_nu`]; also `z = 0`, where `Y` is singular.
pub fn bessel_y_nu(nu: f64, z: C) -> Result<C, String> {
    let nearest = nu.round();
    if (nu - nearest).abs() < NEAR_INTEGER {
        if nearest < 0.0 {
            // Y_{-n} = (-1)^n Y_n
            let n = (-nearest) as i32;
            let y = bessel_y_c(n, z)?;
            return Ok(if n % 2 == 0 { y } else { y * -1.0 });
        }
        return bessel_y_c(nearest as i32, z);
    }
    if z.abs() == 0.0 {
        return Err("bessel_y_nu: Y is singular at z = 0".to_string());
    }
    let (s, c) = (nu * std::f64::consts::PI).sin_cos();
    let jp = bessel_j_nu(nu, z)?;
    let jm = bessel_j_nu(-nu, z)?;
    let out = (jp * c - jm) * (1.0 / s);

    // **The guard Stage 2I deferred, calibrated in 2J.** Measured
    // `loss * eps` against the actual error at the points 2I recorded:
    //
    //   nu = 36.8, z = 54.46   err 3.09e4    loss*eps 1.02   refuse
    //   nu = 36.8, z = 47.84   err 2.18      loss*eps 9.1e-3 refuse
    //   nu =  7.15, z = 61.8   near a zero   loss*eps 8.7e-2 refuse
    //   nu = 20.5, z = 30.34   err 7.8e-8    loss*eps 1.9e-6 allow
    //   nu = 12.3, z = 16.60   err 7.5e-12   loss*eps 1.2e-11 allow
    //
    // 1e-3 separates every measured case, with the closest allowed one
    // 500x inside it. The loss is NOT a proven bound — at
    // `nu = 36.8, z = 47.84` the actual error is 240x larger than
    // `loss * eps` — so the threshold carries that margin explicitly
    // rather than pretending the indicator is exact.
    let spent = y_nu_loss(nu, z)? * f64::EPSILON;
    if spent > 1.0e-3 {
        return Err(format!(
            "bessel_y_nu: the reflection [J_nu cos(nu pi) - J_{{-nu}}]/sin(nu pi) has spent \
             {spent:.1e} of its relative precision at nu = {nu}, z = {z:?} — the ascending \
             series for J_{{-nu}} forms terms of order exp(|z|) to produce a result of \
             order 1. Use `bessel_cnu::bessel_y_cnu`, which compares error estimates across \
             routes and is accurate here."
        ));
    }
    Ok(out)
    // NOTE (Stage 2I): this route is BADLY wrong for non-integer `nu`
    // once `|z|` is large enough that the ascending series for
    // `J_{-nu}` has cancelled away its digits — measured, a relative
    // error of 3.09e4 at `nu = 36.8, z = 54.46`, and 2.18 at
    // `nu = 36.8, z = 47.84`, adjudicated by the J-Y Wronskian against
    // Cephes (whose residual there is 2.2e-23 to our 3.7e-3).
    //
    // Stage 2J calibrated the guard above from exactly that sweep, so
    // these points now REFUSE instead of returning a wrong value.
    // `crate::bessel_cnu::bessel_y_cnu` remains the better entry point
    // — it compares error estimates across routes and is accurate to
    // 4.8e-13 at those points — see
    // `the_selector_is_accurate_where_the_raw_reflection_is_not`.
}

/// `K_nu(z)` for real order and complex `z`.
///
/// Non-integer order uses `K_nu = (pi/2)[I_{-nu} - I_nu]/sin(nu pi)`
/// (DLMF 10.27.4), with the same near-integer handover as
/// [`bessel_y_nu`] and for the same reason.
///
/// **Accuracy falls as `1e-16 exp(|z| + Re z)`** — the two `I` series
/// are each of size `exp(Re z)` and their leading parts cancel, which
/// is what makes `K` decay in the first place. Worst on the positive
/// real axis, exact on the negative one.
///
/// # Errors
/// As [`bessel_y_nu`].
pub fn bessel_k_nu(nu: f64, z: C) -> Result<C, String> {
    let nearest = nu.round();
    if (nu - nearest).abs() < NEAR_INTEGER {
        // K_{-n} = K_n
        return bessel_k_c(nearest.abs() as i32, z);
    }
    if z.abs() == 0.0 {
        return Err("bessel_k_nu: K is singular at z = 0".to_string());
    }
    let s = (nu * std::f64::consts::PI).sin();
    let ip = bessel_i_nu(nu, z)?;
    let im = bessel_i_nu(-nu, z)?;
    Ok((im - ip) * (std::f64::consts::PI / (2.0 * s)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bessel::bessel_j_array;
    use crate::rel_err;
    use spec_math::cephes64::{i0, i1, jv, k0, k1, yn};

    fn close(a: C, b: C, tol: f64) -> bool {
        (a - b).abs() <= tol * (1.0_f64).max(a.abs().max(b.abs()))
    }

    /// On the real axis the complex routine must reproduce the real one
    /// exactly enough to be interchangeable — and both must agree with
    /// the independently written vendored Cephes.
    #[test]
    fn real_axis_matches_the_real_routine_and_cephes() {
        for &x in &[0.1, 0.7, 2.5, 5.0, 9.0, 20.0, 45.0] {
            let cx = bessel_j_array_c(12, C::real(x)).unwrap();
            let rx = bessel_j_array(12, x).unwrap();
            for n in 0..=12 {
                assert!(
                    cx[n].im.abs() < 1e-14,
                    "J_{n}({x}) should be real, got imaginary part {}",
                    cx[n].im
                );
                assert!(
                    rel_err(cx[n].re, rx[n]) < 1e-10 || (cx[n].re - rx[n]).abs() < 1e-15,
                    "J_{n}({x}): complex {} vs real {}",
                    cx[n].re,
                    rx[n]
                );
                let ceph = jv(n as f64, x);
                let tol = if ceph.abs() < 1e-14 { 1e-6 } else { 1e-10 };
                assert!(
                    rel_err(cx[n].re, ceph) < tol || (cx[n].re - ceph).abs() < 1e-15,
                    "J_{n}({x}): ours {} vs cephes {ceph}",
                    cx[n].re
                );
            }
        }
    }

    /// **The generating function is an absolute test.** For any complex
    /// `z` and any `t != 0`,
    ///
    /// ```text
    ///     exp((z/2)(t - 1/t)) = sum_{n=-inf}^{inf} J_n(z) t^n
    /// ```
    ///
    /// with `J_{-n} = (-1)^n J_n`. The right-hand side is built entirely
    /// from the values under test and the left-hand side from `exp`, so
    /// no reference implementation is involved at all.
    #[test]
    fn the_generating_function_identity_holds_off_the_real_axis() {
        let zs = [
            C::new(1.0, 1.0),
            C::new(3.0, -2.0),
            C::new(-4.0, 1.5),
            C::new(0.5, 5.0),
            C::new(8.0, 3.0),
            C::new(-2.0, -6.0),
        ];
        let ts = [C::new(1.3, 0.0), C::new(0.7, 0.4), C::new(-1.1, 0.9)];
        for z in zs {
            let n_max = 60;
            let j = bessel_j_array_c(n_max, z).unwrap();
            for t in ts {
                let mut sum = j[0];
                let mut tp = C::ONE; // t^n
                let mut tm = C::ONE; // t^-n
                let t_inv = t.inv();
                for (n, jn) in j.iter().enumerate().take(n_max + 1).skip(1) {
                    tp = tp * t;
                    tm = tm * t_inv;
                    let sign = if n.is_multiple_of(2) { 1.0 } else { -1.0 };
                    // J_n t^n + J_{-n} t^{-n}
                    sum = sum + *jn * tp + *jn * tm * sign;
                }
                let want = ((z * 0.5) * (t - t_inv)).exp();
                assert!(
                    close(sum, want, 1e-9),
                    "z = {z:?}, t = {t:?}: sum {sum:?} vs exp {want:?}"
                );
            }
        }
    }

    /// The addition theorem `J_n(z+w) = sum_k J_k(z) J_{n-k}(w)`
    /// (DLMF 10.23.1) — another absolute identity, and one that couples
    /// different arguments so a systematic scale error cannot hide.
    #[test]
    fn the_addition_theorem_holds() {
        let cases = [
            (C::new(1.5, 0.8), C::new(-0.7, 1.1)),
            (C::new(3.0, -1.0), C::new(2.0, 2.0)),
            (C::new(-2.5, 0.0), C::new(1.0, -3.0)),
        ];
        for (z, w) in cases {
            let m = 70;
            let jz = bessel_j_array_c(m, z).unwrap();
            let jw = bessel_j_array_c(m, w).unwrap();
            let jsum = bessel_j_array_c(6, z + w).unwrap();
            let jn = |arr: &[C], k: i32| -> C {
                let a = k.unsigned_abs() as usize;
                if a >= arr.len() {
                    C::ZERO
                } else if k >= 0 || a.is_multiple_of(2) {
                    arr[a]
                } else {
                    arr[a] * -1.0
                }
            };
            for n in 0..=4i32 {
                let mut acc = C::ZERO;
                for k in -(m as i32)..=(m as i32) {
                    acc = acc + jn(&jz, k) * jn(&jw, n - k);
                }
                assert!(
                    close(acc, jsum[n as usize], 1e-9),
                    "n = {n}, z = {z:?}, w = {w:?}: {acc:?} vs {:?}",
                    jsum[n as usize]
                );
            }
        }
    }

    /// On the imaginary axis `J_n(iy) = i^n I_n(y)`. The vendored
    /// Cephes provides `I_0` and `I_1` only — no general `I_v` — so the
    /// cross-check against independent machinery covers those two
    /// orders, and the modified generating function below covers the
    /// rest without needing a reference at all.
    #[test]
    fn the_imaginary_axis_matches_the_vendored_modified_bessel() {
        for &y in &[0.3, 1.0, 2.5, 5.0, 9.0] {
            let j = bessel_j_array_c(4, C::new(0.0, y)).unwrap();
            for (n, want_real) in [(0usize, i0(y)), (1usize, i1(y))] {
                let want = i_pow(n as i32) * want_real;
                assert!(
                    close(j[n], want, 1e-10),
                    "J_{n}(i*{y}) = {:?}, want i^{n} I_{n}({y}) = {want:?}",
                    j[n]
                );
            }
        }
    }

    /// **The modified generating function**, absolute and reference-free:
    /// `exp((z/2)(t + 1/t)) = sum_n I_n(z) t^n` with `I_{-n} = I_n`
    /// (DLMF 10.35.1). This covers `I_n` at every order, which the
    /// vendored library cannot.
    #[test]
    fn the_modified_generating_function_holds() {
        let zs = [
            C::new(1.5, 0.0),
            C::new(2.0, 1.0),
            C::new(-1.0, 2.0),
            C::new(3.0, -1.5),
        ];
        let ts = [C::new(1.4, 0.0), C::new(0.8, 0.5)];
        for z in zs {
            let n_max = 50;
            // I_n(z) = i^-n J_n(i z), taken from one J pass
            let j = bessel_j_array_c(n_max, C::I * z).unwrap();
            let i_of: Vec<C> = (0..=n_max).map(|n| j[n] * i_pow(-(n as i32))).collect();
            for t in ts {
                let t_inv = t.inv();
                let mut sum = i_of[0];
                let (mut tp, mut tm) = (C::ONE, C::ONE);
                for inv in i_of.iter().take(n_max + 1).skip(1) {
                    tp = tp * t;
                    tm = tm * t_inv;
                    // I_{-n} = I_n, so no alternating sign here
                    sum = sum + *inv * tp + *inv * tm;
                }
                let want = ((z * 0.5) * (t + t_inv)).exp();
                assert!(
                    close(sum, want, 1e-9),
                    "z = {z:?}, t = {t:?}: sum {sum:?} vs exp {want:?}"
                );
            }
        }
    }

    /// `I_n(z)` for complex `z` must reduce to the vendored real `I_n`
    /// on the real axis — the identity route is only worth having if it
    /// lands in the right place.
    #[test]
    fn modified_bessel_reduces_correctly_on_the_real_axis() {
        for &x in &[0.2, 1.0, 3.0, 7.0] {
            for (n, want) in [(0i32, i0(x)), (1i32, i1(x))] {
                let got = bessel_i_c(n, C::real(x)).unwrap();
                assert!(got.im.abs() < 1e-10, "I_{n}({x}) should be real, got {got:?}");
                assert!(
                    rel_err(got.re, want) < 1e-9,
                    "I_{n}({x}) = {} vs cephes {want}",
                    got.re
                );
            }
        }
    }

    /// Symmetries: `J_n(conj z) = conj J_n(z)` because the coefficients
    /// are real, and `J_n(-z) = (-1)^n J_n(z)`.
    #[test]
    fn conjugation_and_parity() {
        for z in [C::new(2.0, 3.0), C::new(-1.5, 0.7), C::new(4.0, -2.5)] {
            let a = bessel_j_array_c(6, z).unwrap();
            let b = bessel_j_array_c(6, z.conj()).unwrap();
            let c = bessel_j_array_c(6, z * -1.0).unwrap();
            for n in 0..=6 {
                assert!(close(b[n], a[n].conj(), 1e-11), "conjugation at n = {n}");
                let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
                assert!(close(c[n], a[n] * sign, 1e-11), "parity at n = {n}");
            }
        }
    }

    /// The three-term recurrence, satisfied by the returned values.
    #[test]
    fn the_recurrence_is_satisfied() {
        for z in [C::new(1.0, 2.0), C::new(-3.0, 1.0), C::new(6.0, -4.0)] {
            let j = bessel_j_array_c(25, z).unwrap();
            let inv = z.inv();
            for n in 1..24 {
                let lhs = j[n - 1] + j[n + 1];
                let rhs = inv * j[n] * (2 * n) as f64;
                assert!(close(lhs, rhs, 1e-10), "recurrence n = {n}, z = {z:?}");
            }
        }
    }

    /// The documented accuracy limit is cancellation in the normalising
    /// sum, losing roughly `|Im z| / ln 10` digits. Pinned as a
    /// MEASUREMENT: agreement must hold at `|Im z| = 8` and is allowed
    /// to be much worse at `|Im z| = 25`, which is the honest statement
    /// of where the method stops working.
    #[test]
    fn accuracy_degrades_with_imaginary_part_as_documented() {
        let err_at = |z: C| -> f64 {
            let n_max = 80;
            let j = bessel_j_array_c(n_max, z).unwrap();
            let t = C::new(1.2, 0.3);
            let t_inv = t.inv();
            let mut sum = j[0];
            let (mut tp, mut tm) = (C::ONE, C::ONE);
            for (n, jn) in j.iter().enumerate().take(n_max + 1).skip(1) {
                tp = tp * t;
                tm = tm * t_inv;
                let sign = if n.is_multiple_of(2) { 1.0 } else { -1.0 };
                sum = sum + *jn * tp + *jn * tm * sign;
            }
            let want = ((z * 0.5) * (t - t_inv)).exp();
            (sum - want).abs() / want.abs().max(1.0)
        };
        let good = err_at(C::new(2.0, 8.0));
        assert!(good < 1e-8, "at |Im z| = 8 the error is {good}, expected < 1e-8");
        // and the degradation is real, not imagined
        let bad = err_at(C::new(2.0, 25.0));
        assert!(
            bad > good,
            "error should grow with |Im z|: {good} at 8 vs {bad} at 25"
        );
    }

    /// **The Wronskian is the absolute test for `Y`.**
    ///
    /// `J_{n+1}(z) Y_n(z) - J_n(z) Y_{n+1}(z) = 2/(pi z)` (DLMF 10.5.2).
    /// The right-hand side is elementary, so this pins `Y` in both scale
    /// and phase against a `J` that is itself independently verified —
    /// no table, no reference library.
    #[test]
    fn the_j_y_wronskian_holds() {
        let zs = [
            C::real(0.7),
            C::real(4.0),
            C::new(1.0, 1.0),
            C::new(3.0, -2.0),
            C::new(-2.5, 1.5),
            C::new(0.4, 3.0),
            C::new(6.0, 2.0),
        ];
        for z in zs {
            let j = bessel_j_array_c(7, z).unwrap();
            let y = bessel_y_array_c(7, z).unwrap();
            let want = z.inv() * (2.0 / std::f64::consts::PI);
            for n in 0..6 {
                let w = j[n + 1] * y[n] - j[n] * y[n + 1];
                assert!(
                    close(w, want, 1e-9),
                    "Wronskian at n = {n}, z = {z:?}: {w:?} vs 2/(pi z) = {want:?}"
                );
            }
        }
    }

    /// `Y` satisfies the same three-term recurrence as `J` — which is
    /// how the higher orders are produced, so this checks the sweep
    /// rather than merely restating it.
    #[test]
    fn the_y_recurrence_is_satisfied() {
        for z in [C::new(1.5, 0.5), C::new(-2.0, 1.0), C::new(4.0, -3.0)] {
            let y = bessel_y_array_c(10, z).unwrap();
            let inv = z.inv();
            for n in 1..9 {
                let lhs = y[n - 1] + y[n + 1];
                let rhs = inv * y[n] * (2 * n) as f64;
                assert!(close(lhs, rhs, 1e-8), "Y recurrence n = {n}, z = {z:?}");
            }
        }
    }

    /// On the real axis `Y_n` must match the independently written
    /// vendored Cephes.
    #[test]
    fn y_matches_cephes_on_the_real_axis() {
        for &x in &[0.3, 1.0, 2.5, 5.0, 9.0] {
            let y = bessel_y_array_c(5, C::real(x)).unwrap();
            for (n, yv) in y.iter().enumerate() {
                let want = yn(n as isize, x);
                assert!(yv.im.abs() < 1e-11, "Y_{n}({x}) should be real, got {yv:?}");
                assert!(
                    rel_err(yv.re, want) < 1e-9,
                    "Y_{n}({x}) = {} vs cephes {want}",
                    yv.re
                );
            }
        }
    }

    /// **The absolute test for `K`**: `I_n K_{n+1} + I_{n+1} K_n = 1/z`
    /// (DLMF 10.28.2). Again elementary on the right, so it fixes `K`'s
    /// normalisation with no reference at all.
    #[test]
    fn the_i_k_wronskian_holds() {
        let zs = [
            C::real(0.6),
            C::real(3.0),
            C::new(1.0, 0.5),
            C::new(2.0, -1.5),
            C::new(0.8, 2.0),
        ];
        for z in zs {
            let want = z.inv();
            for n in 0..4i32 {
                let i_n = bessel_i_c(n, z).unwrap();
                let i_n1 = bessel_i_c(n + 1, z).unwrap();
                let k_n = bessel_k_c(n, z).unwrap();
                let k_n1 = bessel_k_c(n + 1, z).unwrap();
                let w = i_n * k_n1 + i_n1 * k_n;
                assert!(
                    close(w, want, 1e-8),
                    "I-K Wronskian at n = {n}, z = {z:?}: {w:?} vs 1/z = {want:?}"
                );
            }
        }
    }

    /// `K_n` on the real axis against the vendored Cephes `k0`/`k1`.
    #[test]
    fn k_matches_cephes_on_the_real_axis() {
        for &x in &[0.4, 1.0, 2.0, 4.0] {
            for (n, want) in [(0i32, k0(x)), (1i32, k1(x))] {
                let got = bessel_k_c(n, C::real(x)).unwrap();
                assert!(
                    got.im.abs() < 1e-9 * got.re.abs().max(1.0),
                    "K_{n}({x}) should be real, got {got:?}"
                );
                assert!(
                    rel_err(got.re, want) < 1e-8,
                    "K_{n}({x}) = {} vs cephes {want}",
                    got.re
                );
            }
        }
    }

    /// `Y` and `K` inherit the branch cut of `ln` along the negative
    /// real axis, so they are DISCONTINUOUS across it while `J` is not.
    /// Pinned so nobody assumes otherwise.
    #[test]
    fn y_is_discontinuous_across_the_negative_real_axis() {
        let eps = 1e-8;
        let above = C::new(-2.0, eps);
        let below = C::new(-2.0, -eps);
        // J is entire: the two sides agree
        let ja = bessel_j_c(0, above).unwrap();
        let jb = bessel_j_c(0, below).unwrap();
        assert!(close(ja, jb, 1e-6), "J should be continuous: {ja:?} vs {jb:?}");
        // Y is not
        let ya = bessel_y_c(0, above).unwrap();
        let yb = bessel_y_c(0, below).unwrap();
        let jump = (ya - yb).abs();
        assert!(
            jump > 0.1,
            "Y should jump across the cut, but moved only {jump}"
        );
        // The size of the jump follows from the series. Y carries a
        // (2/pi) ln(z/2) J term and nothing else discontinuous; crossing
        // the cut takes arg from +pi to -pi, a change of 2*pi, so
        //
        //     Y(above) - Y(below) = (2/pi) * (2 pi i) * J = 4i J.
        //
        // (A first draft of this test predicted 2i J and failed by
        // exactly a factor of two — the derivation above is the one the
        // measurement agrees with.)
        let predicted = ja * C::I * 4.0;
        assert!(
            close(ya - yb, predicted, 1e-5),
            "jump {:?} vs predicted 4i J_0 = {predicted:?}",
            ya - yb
        );
    }

    #[test]
    fn invalid_input_is_reported() {
        assert!(bessel_j_c(-1, C::ONE).is_err(), "negative order");
        assert!(bessel_i_c(-1, C::ONE).is_err(), "negative order");
        // The refusal must hold on the WHOLE domain, including the
        // regions the guard-free asymptotic routes cover — these two
        // used to leak values for n < 0.
        assert!(bessel_j_c(-1, C::new(0.0, 30.0)).is_err(), "negative order, rotation route");
        assert!(bessel_j_c(-1, C::new(60.0, 40.0)).is_err(), "negative order, asymptotic route");
        assert!(
            bessel_j_array_c(3, C::new(f64::NAN, 0.0)).is_err(),
            "non-finite z"
        );
        assert!(
            bessel_j_array_c(3, C::new(0.0, f64::INFINITY)).is_err(),
            "non-finite z"
        );
        // Y and K are singular at the origin — an error, not an infinity
        assert!(bessel_y_c(0, C::ZERO).is_err(), "Y at z = 0");
        assert!(bessel_k_c(0, C::ZERO).is_err(), "K at z = 0");
        assert!(bessel_y_c(-1, C::ONE).is_err(), "negative order");
        assert!(bessel_k_c(-1, C::ONE).is_err(), "negative order");
    }

    // -----------------------------------------------------------------
    // Non-integer order
    // -----------------------------------------------------------------

    /// Half-integer order has a closed form in elementary functions, and
    /// it holds for COMPLEX z, so it tests the series where nothing else
    /// can reach: `J_{1/2}(z) = sqrt(2/(pi z)) sin z`,
    /// `J_{-1/2}(z) = sqrt(2/(pi z)) cos z`.
    ///
    /// Note this is not a table lookup — `sin` and `cos` of a complex
    /// argument are built here from `exp`, which shares no code with the
    /// Bessel series.
    #[test]
    fn half_integer_closed_forms() {
        let csin = |z: C| ((C::I * z).exp() - (C::I * z * -1.0).exp()) / (C::I * 2.0);
        let ccos = |z: C| ((C::I * z).exp() + (C::I * z * -1.0).exp()) * 0.5;
        for &(re, im) in &[
            (0.4, 0.0),
            (2.0, 0.0),
            (7.5, 0.0),
            (1.3, 0.6),
            (3.0, -2.0),
            (5.0, 4.0),
            (-2.5, 1.0),
        ] {
            let z = C::new(re, im);
            let pref = (C::real(2.0 / std::f64::consts::PI) * z.inv()).powf(0.5);
            let got = bessel_j_nu(0.5, z).unwrap();
            let want = pref * csin(z);
            assert!(close(got, want, 1e-12), "J_1/2({z:?}): {got:?} vs {want:?}");

            let got = bessel_j_nu(-0.5, z).unwrap();
            let want = pref * ccos(z);
            assert!(close(got, want, 1e-12), "J_-1/2({z:?}): {got:?} vs {want:?}");
        }
    }

    /// `K_{1/2}(z) = sqrt(pi/(2z)) exp(-z)` (DLMF 10.39.2). This one goes
    /// through the reflection formula and both `I` series, so it checks
    /// the whole `K` path at once.
    #[test]
    fn k_half_integer_closed_form() {
        for &(re, im) in &[(0.5, 0.0), (2.0, 0.0), (6.0, 0.0), (1.5, 1.0), (3.0, -2.5)] {
            let z = C::new(re, im);
            let got = bessel_k_nu(0.5, z).unwrap();
            let want = (C::real(std::f64::consts::PI * 0.5) * z.inv()).powf(0.5) * (z * -1.0).exp();
            assert!(close(got, want, 1e-11), "K_1/2({z:?}): {got:?} vs {want:?}");
            // K_{-nu} = K_nu, and here that runs a completely different
            // pair of series, so it is a real check rather than a tautology.
            let neg = bessel_k_nu(-0.5, z).unwrap();
            assert!(close(neg, got, 1e-11), "K_-1/2 != K_1/2 at {z:?}");
        }
    }

    /// The Wronskian `J_{nu+1} Y_nu - J_nu Y_{nu+1} = 2/(pi z)`
    /// (DLMF 10.5.2) holds for every order, integer or not. The right
    /// side is elementary, so this is an ABSOLUTE check on `Y_nu` — it
    /// cannot be satisfied by a consistently wrong pair.
    #[test]
    fn j_y_wronskian_at_non_integer_order() {
        for &nu in &[0.25, 0.5, 1.3, 2.7, 4.4, -0.75] {
            for &(re, im) in &[(0.6, 0.0), (2.0, 0.0), (5.0, 0.0), (1.2, 0.8), (3.0, -1.5)] {
                let z = C::new(re, im);
                let w = bessel_j_nu(nu + 1.0, z).unwrap() * bessel_y_nu(nu, z).unwrap()
                    - bessel_j_nu(nu, z).unwrap() * bessel_y_nu(nu + 1.0, z).unwrap();
                let want = z.inv() * (2.0 / std::f64::consts::PI);
                assert!(
                    close(w, want, 1e-10),
                    "J-Y Wronskian nu={nu} z={z:?}: {w:?} vs {want:?}"
                );
            }
        }
    }

    /// `I_nu K_{nu+1} + I_{nu+1} K_nu = 1/z` (DLMF 10.28.2), again for
    /// general order and again with an elementary right-hand side.
    #[test]
    fn i_k_wronskian_at_non_integer_order() {
        for &nu in &[0.25, 0.5, 1.3, 2.7, -0.4] {
            for &(re, im) in &[(0.6, 0.0), (2.0, 0.0), (5.0, 0.0), (1.2, 0.8), (3.0, -1.5)] {
                let z = C::new(re, im);
                let w = bessel_i_nu(nu, z).unwrap() * bessel_k_nu(nu + 1.0, z).unwrap()
                    + bessel_i_nu(nu + 1.0, z).unwrap() * bessel_k_nu(nu, z).unwrap();
                assert!(
                    close(w, z.inv(), 1e-10),
                    "I-K Wronskian nu={nu} z={z:?}: {w:?} vs {:?}",
                    z.inv()
                );
            }
        }
    }

    /// The three-term recurrence in ORDER, `C_{nu-1} + C_{nu+1} =
    /// (2 nu / z) C_nu` for J and Y, and `I_{nu-1} - I_{nu+1} =
    /// (2 nu / z) I_nu` for I (DLMF 10.6.1, 10.29.1). Each value comes
    /// from its own independent series evaluation, so agreement is not
    /// built in.
    #[test]
    fn order_recurrence() {
        for &nu in &[0.3, 1.6, 3.2, 5.8] {
            for &(re, im) in &[(1.0, 0.0), (4.0, 0.0), (2.0, 1.5), (0.7, -0.9)] {
                let z = C::new(re, im);
                let f = z.inv() * (2.0 * nu);
                let lhs = bessel_j_nu(nu - 1.0, z).unwrap() + bessel_j_nu(nu + 1.0, z).unwrap();
                assert!(close(lhs, bessel_j_nu(nu, z).unwrap() * f, 1e-11), "J rec nu={nu}");
                let lhs = bessel_y_nu(nu - 1.0, z).unwrap() + bessel_y_nu(nu + 1.0, z).unwrap();
                assert!(close(lhs, bessel_y_nu(nu, z).unwrap() * f, 1e-9), "Y rec nu={nu}");
                let lhs = bessel_i_nu(nu - 1.0, z).unwrap() - bessel_i_nu(nu + 1.0, z).unwrap();
                assert!(close(lhs, bessel_i_nu(nu, z).unwrap() * f, 1e-11), "I rec nu={nu}");
            }
        }
    }

    /// At integer order the general routine must reproduce the integer
    /// routines, which were written earlier from entirely different
    /// formulas (Miller recurrence for J, the logarithmic series for Y).
    #[test]
    fn integer_order_agrees_with_the_integer_routines() {
        for n in 0..6 {
            for &(re, im) in &[(0.8, 0.0), (3.0, 0.0), (6.0, 0.0), (2.0, 1.0), (1.5, -2.0)] {
                let z = C::new(re, im);
                let j = bessel_j_array_c(n, z).unwrap()[n];
                assert!(close(bessel_j_nu(n as f64, z).unwrap(), j, 1e-11), "J_{n}({z:?})");
                let y = bessel_y_c(n as i32, z).unwrap();
                assert!(close(bessel_y_nu(n as f64, z).unwrap(), y, 1e-11), "Y_{n}({z:?})");
                let k = bessel_k_c(n as i32, z).unwrap();
                assert!(close(bessel_k_nu(n as f64, z).unwrap(), k, 1e-10), "K_{n}({z:?})");
            }
        }
    }

    /// The near-integer handover is the one place the design could hide a
    /// discontinuity: below the threshold `Y_nu` is the logarithmic
    /// integer series, above it the reflection formula. Approaching an
    /// integer from just outside the threshold must land on the integer
    /// value. This tests the reflection formula against a completely
    /// independent implementation, in the regime where the reflection is
    /// worst conditioned.
    #[test]
    fn near_integer_handover_is_continuous() {
        for n in 0..4 {
            for &(re, im) in &[(1.4, 0.0), (3.5, 0.0), (2.0, 1.0)] {
                let z = C::new(re, im);
                let exact = bessel_y_c(n, z).unwrap();
                // 1e-7 is a hundred times the handover threshold, so this
                // genuinely goes through the reflection formula.
                let off = bessel_y_nu(n as f64 + 1e-7, z).unwrap();
                assert!(
                    close(off, exact, 1e-5),
                    "Y_{n}+1e-7 at {z:?}: reflection {off:?} vs integer series {exact:?}"
                );
                let exact = bessel_k_c(n, z).unwrap();
                let off = bessel_k_nu(n as f64 + 1e-7, z).unwrap();
                assert!(close(off, exact, 1e-5), "K_{n}+1e-7 at {z:?}");
            }
        }
    }

    /// On the real axis, non-integer order can be checked against the
    /// vendored Cephes `jv`, which takes a real order and is a wholly
    /// separate implementation (continued fractions and asymptotics, not
    /// an ascending series).
    #[test]
    fn non_integer_order_matches_cephes_on_the_real_axis() {
        for &nu in &[0.25, 0.5, 1.3, 2.7, 4.4, 7.1] {
            for &x in &[0.3, 1.0, 3.0, 6.0, 11.0] {
                let ours = bessel_j_nu(nu, C::real(x)).unwrap();
                let ceph = jv(nu, x);
                assert!(ours.im.abs() < 1e-14, "J_{nu}({x}) should be real");
                assert!(
                    rel_err(ours.re, ceph) < 1e-9 || (ours.re - ceph).abs() < 1e-14,
                    "J_{nu}({x}): ours {} vs cephes {ceph}",
                    ours.re
                );
            }
        }
    }

    /// `J_{-n}` for whole `n` must be `(-1)^n J_n` — the series gets this
    /// for free only because `1/Gamma` vanishes at the poles, so it is a
    /// direct test of that design choice.
    #[test]
    fn negative_whole_order_uses_the_gamma_poles() {
        for n in 1..6 {
            let z = C::new(2.3, -1.1);
            let neg = bessel_j_nu(-(n as f64), z).unwrap();
            let pos = bessel_j_nu(n as f64, z).unwrap();
            let want = if n % 2 == 0 { pos } else { pos * -1.0 };
            assert!(close(neg, want, 1e-12), "J_-{n} vs (-1)^n J_{n}");
            // I_{-n} = I_n, no sign.
            let neg = bessel_i_nu(-(n as f64), z).unwrap();
            let pos = bessel_i_nu(n as f64, z).unwrap();
            assert!(close(neg, pos, 1e-12), "I_-{n} vs I_{n}");
        }
    }

    /// Negative-order `Y` and `K` reflection: `Y_{-n} = (-1)^n Y_n` and
    /// `K_{-nu} = K_nu`, both exercised through the near-integer branch.
    #[test]
    fn negative_order_reflections() {
        let z = C::new(1.7, 0.9);
        for n in 1..5 {
            let want = bessel_y_c(n, z).unwrap();
            let want = if n % 2 == 0 { want } else { want * -1.0 };
            assert!(close(bessel_y_nu(-(n as f64), z).unwrap(), want, 1e-11), "Y_-{n}");
            let want = bessel_k_c(n, z).unwrap();
            assert!(close(bessel_k_nu(-(n as f64), z).unwrap(), want, 1e-11), "K_-{n}");
        }
    }

    /// The INTEGER-order accuracy laws, pinned against Cephes.
    ///
    /// This test exists because the original documentation for this
    /// module stated one law — `J`'s — and implied it covered all four
    /// kinds. It does not. `J` and `I` come from Miller recurrence and
    /// are excellent; `Y` comes from an ascending series and `K` is
    /// built from `J` and `Y` at imaginary argument, so both carry the
    /// `exp(|z|)` cancellation of a series. On the real axis `Y` is
    /// wrong in the first digit by `x = 40` and `K` long before that,
    /// neither of which the old text admitted. A Hankel asymptotic test
    /// found it; this pins it.
    ///
    /// Each law is checked on the axis where its worst case lies and
    /// where an independent reference exists, with two decimal digits of
    /// slack over the model:
    ///
    /// ```text
    ///   J_n:  1e-16 exp(|Im z|)                  worst up the imaginary axis
    ///   I_n:  1e-16 exp(|Re z|)                  worst along the real axis
    ///   Y_n:  1e-16 exp(|z| - |Im z|)            worst along the real axis
    ///   K_n:  1e-16 exp(max(2|Re z|, |z|) + Re z)   worst along the positive real axis
    /// ```
    ///
    /// `K`'s exponent is the ugly one because `K_n(z)` is assembled from
    /// `J_n(iz) + i Y_n(iz)`: the `J` there is already amplified by
    /// `exp(|Re z|)` relative to a result of size `exp(-Re z)`, which is
    /// where the third factor comes from.
    #[test]
    fn integer_order_accuracy_laws_hold() {
        use spec_math::cephes64::{i0, j0, k0};
        let bound = |l: f64| 1e-14 * l.exp();
        for &x in &[1.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0] {
            // J on the imaginary axis, where it is worst: J_0(ix) = I_0(x).
            let got = bessel_j_c(0, C::new(0.0, x)).unwrap().re;
            let want = i0(x);
            assert!(
                (got - want).abs() <= bound(x) * want.abs(),
                "J_0({x}i): {got:e} vs {want:e}"
            );
            // ... and on the real axis, where it is best.
            // Near a zero of J_0 a relative bound is meaningless, so the
            // scale is floored at 0.1 — comparable to the function's own
            // amplitude, not to the value at that particular x.
            let got = bessel_j_c(0, C::real(x)).unwrap().re;
            assert!(
                (got - j0(x)).abs() <= bound(0.0) * j0(x).abs().max(0.1),
                "J_0({x}): {got:e} vs {:e}",
                j0(x)
            );
            // I on the real axis, where it is worst: I is J at iz, so
            // this is the same measurement seen from the other side.
            let got = bessel_i_c(0, C::real(x)).unwrap().re;
            assert!(
                (got - i0(x)).abs() <= bound(x) * i0(x),
                "I_0({x}): {got:e} vs {:e}",
                i0(x)
            );
            // Y on the real axis, where L = |z|.
            let got = bessel_y_c(0, C::real(x)).unwrap().re;
            let want = yn(0, x);
            assert!(
                (got - want).abs() <= bound(x) * want.abs().max(0.1),
                "Y_0({x}): {got:e} vs {want:e}"
            );
            // K on the real axis, where L = 3|z|. Past x ~ 40 the
            // routine gives up and returns an error, which is the
            // correct behaviour and is checked separately.
            if let Ok(got) = bessel_k_c(0, C::real(x)) {
                let want = k0(x);
                assert!(
                    (got.re - want).abs() <= bound(3.0 * x) * want,
                    "K_0({x}): {:e} vs {want:e}",
                    got.re
                );
            }
        }
    }

    /// The points where these routines used to fail, now asserted to
    /// work. This test used to say the opposite — that `Y_0(25)` had
    /// lost its digits and `K_0(20)` was worthless — and it was right
    /// when it was written. Route selection made it wrong, so it is
    /// inverted rather than deleted: the record of what was broken is
    /// the point of it.
    ///
    /// `Y_0(40)` was wrong in its first digit. `K_0(20)` was out by a
    /// factor of `8e8`. `Y_2` at `z = 29.4 e^{1.6i}` closed the J-Y
    /// Wronskian to 4.5e-6.
    #[test]
    fn the_points_that_used_to_fail_now_work() {
        for &x in &[10.0_f64, 20.0, 25.0, 30.0, 40.0, 60.0] {
            let got = bessel_y_c(0, C::real(x)).unwrap().re;
            let want = yn(0, x);
            // 1e-11: at x = 10 the ascending series is still the best
            // route available and its own law gives 2e-12, which is what
            // it delivers. The fix is for where it used to give 1e-1.
            assert!(
                (got - want).abs() <= 1e-11 * want.abs().max(0.05),
                "Y_0({x}): {got} vs {want}"
            );
            let got = bessel_k_c(0, C::real(x)).unwrap().re;
            let want = spec_math::cephes64::k0(x);
            if want > 0.0 {
                // Likewise 1e-9 at x = 10, where the 1/z expansion is
                // only just better than the identity and its own
                // truncation is 2e-10.
                assert!(
                    (got - want).abs() <= 1e-9 * want,
                    "K_0({x}): {got} vs {want}"
                );
            }
        }
        // The complex point Stage 18 found, judged by the Wronskian.
        let z = C::from_polar(29.4, 1.6);
        let w = bessel_j_c(3, z).unwrap() * bessel_y_c(2, z).unwrap()
            - bessel_j_c(2, z).unwrap() * bessel_y_c(3, z).unwrap();
        let want = z.inv() * (2.0 / std::f64::consts::PI);
        let scale = (bessel_j_c(3, z).unwrap() * bessel_y_c(2, z).unwrap()).abs() * 2.0;
        assert!(
            (w - want).abs() / scale < 1e-13,
            "Wronskian at 29.4e^(1.6i): {:.2e}",
            (w - want).abs() / scale
        );
    }

    /// The accuracy bounds the documentation states are a claim, so they
    /// are pinned here. `L` is the loss exponent:
    /// `|z| - |Im z|` for J and Y, `|z| + Re z` for I and K — the log of
    /// the ratio between the largest term of the series and the answer.
    /// See `examples/bessel_nu_accuracy.rs` for the derivation and the
    /// full measured surface.
    #[test]
    fn documented_accuracy_bounds_hold() {
        let csin = |z: C| ((C::I * z).exp() - (C::I * z * -1.0).exp()) / (C::I * 2.0);
        // The model says the relative error is about `1e-16 * exp(L)`.
        // Pinning it with two decimal digits of slack tests the LAW —
        // a bucketed tolerance would only test whichever radii happened
        // to be sampled, and the first draft of this test did exactly
        // that and tripped on its own bucket edges at L = 10 and L = 30.
        let bound = |l: f64| 1e-14 * l.exp();
        for &r in &[1.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 70.0] {
            for &a in &[0.0, 0.5, 1.0, 1.5, 2.0, 3.0] {
                let z = C::from_polar(r, a);

                let l = r - z.im.abs();
                if l <= 30.0 {
                    let got = bessel_j_nu(0.5, z).unwrap();
                    let want =
                        (C::real(2.0 / std::f64::consts::PI) * z.inv()).powf(0.5) * csin(z);
                    let e = (got - want).abs() / want.abs();
                    assert!(
                        e <= bound(l),
                        "J_1/2 at r={r} arg={a} (L={l:.1}): {e:.1e} exceeds {:.0e}",
                        bound(l)
                    );
                }

                let l = r + z.re;
                if l <= 30.0 {
                    let got = bessel_k_nu(0.5, z).unwrap();
                    let want = (C::real(std::f64::consts::PI * 0.5) * z.inv()).powf(0.5)
                        * (z * -1.0).exp();
                    let e = (got - want).abs() / want.abs();
                    assert!(
                        e <= bound(l),
                        "K_1/2 at r={r} arg={a} (L={l:.1}): {e:.1e} exceeds {:.0e}",
                        bound(l)
                    );
                }
            }
        }
    }

    /// `K_n` on the far side of the imaginary axis. The identity used
    /// to build `K` rotates its argument by `i`, which for
    /// `arg z > pi/2` used to cross `Y`'s branch cut and land on the
    /// wrong sheet. Pinned by the I-K Wronskian, whose right-hand side
    /// is elementary and correct on every sheet, and scaled by the
    /// largest term so the metric's own cancellation is divided out.
    #[test]
    fn k_stays_on_the_right_sheet_past_the_imaginary_axis() {
        for &nu in &[0, 1, 5, 10] {
            for &a in &[1.6, 2.0, 2.5, 3.0, -1.6, -2.0, -2.5, -3.0] {
                let z = C::from_polar(6.0, a);
                let (i0, i1) = (bessel_i_c(nu, z).unwrap(), bessel_i_c(nu + 1, z).unwrap());
                let (k0, k1) = (bessel_k_c(nu, z).unwrap(), bessel_k_c(nu + 1, z).unwrap());
                let w = i0 * k1 + i1 * k0;
                let scale = (i0 * k1).abs() + (i1 * k0).abs();
                assert!(
                    (w - z.inv()).abs() / scale < 1e-10,
                    "I-K Wronskian at n={nu}, arg z={a}: residual {:.2e}",
                    (w - z.inv()).abs() / scale
                );
            }
        }
    }

    /// The whole family near the imaginary axis, judged by the J-Y
    /// Wronskian — elementary on the right, so it needs no reference —
    /// scaled by its own largest term so the metric's cancellation is
    /// divided out rather than measured.
    ///
    /// This grid is what Stage 19 was for. Before it, the column at
    /// `arg z = pi/2` read 3.3e-7 at `|z| = 25` and 1.0e0 at `|z| = 40`.
    #[test]
    fn the_whole_plane_satisfies_the_wronskian() {
        let mut worst = 0.0_f64;
        for &r in &[5.0_f64, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0] {
            for &a in &[0.0_f64, 0.4, 0.8, 1.2, 1.4, std::f64::consts::FRAC_PI_2, 1.8, 2.2, 2.8] {
                let z = C::from_polar(r, a);
                let (Ok(j0), Ok(j1), Ok(y0), Ok(y1)) = (
                    bessel_j_c(2, z),
                    bessel_j_c(3, z),
                    bessel_y_c(2, z),
                    bessel_y_c(3, z),
                ) else {
                    continue;
                };
                let w = j1 * y0 - j0 * y1;
                let want = z.inv() * (2.0 / std::f64::consts::PI);
                let scale = (j1 * y0).abs() + (j0 * y1).abs();
                if !scale.is_finite() || scale == 0.0 {
                    continue;
                }
                let e = (w - want).abs() / scale;
                worst = worst.max(e);
                // The bound is whichever route is actually available.
                // Where one of the expansions applies it is 1e-11; where
                // none does — inside the pi/3 margin both Hankel
                // expansions keep from the negative real axis — the
                // ascending series is all there is and its
                // exp(|z| - |Im z|) loss governs. That direction is the
                // one Stage 19 did not reach, and saying so beats
                // loosening the bound everywhere to accommodate it.
                let has_route = y_via_ik(2, z).is_some() || y_via_asym(2, z).is_some();
                let bound = if has_route {
                    // 1e-10, which is the measured worst with a route
                    // available: 7.2e-11 at |z| = 40, arg z = 0.4.
                    1e-10
                } else {
                    (1e-13 * (r - z.im.abs()).exp()).max(1e-11)
                };
                assert!(
                    e < bound,
                    "|z|={r}, arg={a:.2}: residual {e:.2e} exceeds {bound:.1e} \
                     (expansion available: {has_route})"
                );
            }
        }
        // ... and the grid must actually be reaching something hard, or
        // the bound above is decoration.
        assert!(worst > 1e-15, "worst was {worst:.1e}");
    }

    /// The negative real axis, tested by the **exact** continuation
    /// identities rather than by a Wronskian.
    ///
    /// ```text
    ///   J_n(x e^{i pi}) = (-1)^n J_n(x)
    ///   Y_n(x e^{i pi}) = (-1)^n [Y_n(x) + 2i J_n(x)]
    /// ```
    ///
    /// These follow from DLMF 10.11.3/4 and relate a point on the cut to
    /// one on the positive real axis, where every route here is at its
    /// best. That makes them a far better test than the J-Y Wronskian,
    /// which near the cut is dominated by the exponentially **recessive**
    /// Hankel member and so measures the Stokes phenomenon rather than
    /// the answer — a distinction that cost some confusion to find.
    #[test]
    fn the_negative_real_axis_satisfies_the_continuation_identities() {
        for &x in &[10.0_f64, 20.0, 40.0, 60.0, 100.0, 300.0] {
            for &n in &[0_i32, 2, 5] {
                let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
                let (jp, yp) = (
                    bessel_j_c(n, C::real(x)).unwrap(),
                    bessel_y_c(n, C::real(x)).unwrap(),
                );
                // `-x` with a POSITIVE zero imaginary part is the upper
                // side of the cut, which is the side `arg = +pi` names.
                let zn = C::new(-x, 0.0);
                let jg = bessel_j_c(n, zn).unwrap();
                let yg = bessel_y_c(n, zn).unwrap();
                let jw = jp * sign;
                let yw = (yp + C::I * jp * 2.0) * sign;
                assert!(
                    (jg - jw).abs() <= 1e-13 * jw.abs(),
                    "J_{n}(-{x}): {jg:?} vs {jw:?}"
                );
                assert!(
                    (yg - yw).abs() <= 1e-13 * yw.abs(),
                    "Y_{n}(-{x}): {yg:?} vs {yw:?}"
                );
            }
        }
    }

    /// The wedge either side of the cut, where neither the rotation of
    /// [`j_via_i`] nor the direct expansion applies and only the
    /// continuation does. Same identities, applied at `arg z` rather
    /// than at `pi`.
    #[test]
    fn the_wedge_beside_the_cut_is_covered() {
        for &r in &[15.0_f64, 40.0, 100.0, 300.0] {
            for &a in &[2.2_f64, 2.4, 2.6, 2.8, -2.4, -2.8] {
                let z = C::from_polar(r, a);
                let w = C::from_polar(r, a - a.signum() * std::f64::consts::PI);
                let n = 2;
                let (jw, yw) = (bessel_j_c(n, w).unwrap(), bessel_y_c(n, w).unwrap());
                let s = C::I * a.signum() * 2.0;
                let jg = bessel_j_c(n, z).unwrap();
                let yg = bessel_y_c(n, z).unwrap();
                // 1e-11: worst measured in the wedge is 1.6e-12, at
                // r = 15 where the expansion is only just converged.
                assert!((jg - jw).abs() <= 1e-11 * jw.abs(), "J at r={r}, arg={a}");
                let want = yw + jw * s;
                assert!((yg - want).abs() <= 1e-11 * want.abs(), "Y at r={r}, arg={a}");
            }
        }
    }

    /// The branch jump must still be exactly `4i(-1)^n J_n`. Widening
    /// the coverage across the cut is only correct if the cut itself is
    /// still where it was.
    #[test]
    fn the_branch_jump_is_unchanged() {
        for &x in &[5.0_f64, 20.0, 50.0, 200.0] {
            for &n in &[0_i32, 3] {
                let up = bessel_y_c(n, C::new(-x, 0.0)).unwrap();
                let lo = bessel_y_c(n, C::new(-x, -0.0)).unwrap();
                let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
                let want = C::I * bessel_j_c(n, C::real(x)).unwrap() * 4.0 * sign;
                assert!(
                    (up - lo - want).abs() <= 1e-12 * want.abs(),
                    "jump at n={n}, x={x}: {:?} vs {want:?}",
                    up - lo
                );
            }
        }
    }

    /// The three routes must each be the one chosen where it belongs,
    /// and the choice must be right rather than merely different.
    #[test]
    fn each_route_is_taken_where_it_belongs() {
        // Near the imaginary axis: J and Y from I and K on w = -iz.
        let z = C::from_polar(30.0, 1.5);
        assert!(j_via_i(2, z).is_some(), "J should rotate here");
        assert!(y_via_ik(2, z).is_some(), "Y should rotate here");
        // On the real axis: Y and K from their own 1/z expansions.
        let z = C::real(30.0);
        assert!(j_via_i(2, z).is_none(), "J does not rotate on the real axis");
        assert!(y_via_asym(2, z).is_some(), "Y should use the expansion here");
        assert!(k_via_asym(2, z).is_some(), "K should use the expansion here");
        // At small |z| every route declines and the series is used.
        let z = C::real(2.0);
        assert!(y_via_asym(2, z).is_none(), "no expansion at |z| = 2");
        assert!(k_via_asym(2, z).is_none(), "no expansion at |z| = 2");
        assert!(j_via_i(2, C::new(0.0, 2.0)).is_none(), "no rotation at |z| = 2");
    }

    #[test]
    fn non_integer_order_edge_cases() {
        // J_nu(0) = 0 for nu > 0, 1 for nu = 0, singular for nu < 0.
        assert_eq!(bessel_j_nu(1.5, C::ZERO).unwrap(), C::ZERO);
        assert_eq!(bessel_j_nu(0.0, C::ZERO).unwrap(), C::ONE);
        assert!(bessel_j_nu(-0.5, C::ZERO).is_err(), "J_-1/2 at 0");
        assert!(bessel_y_nu(0.5, C::ZERO).is_err(), "Y at 0");
        assert!(bessel_k_nu(0.5, C::ZERO).is_err(), "K at 0");
        assert!(bessel_j_nu(f64::NAN, C::ONE).is_err(), "NaN order");
        assert!(bessel_j_nu(1.5, C::new(f64::INFINITY, 0.0)).is_err(), "infinite z");
    }
}
