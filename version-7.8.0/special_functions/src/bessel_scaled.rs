//! Scaled Bessel and Hankel functions — the fix, not the disclaimer.
//!
//! The previous two stages documented a limit and stopped there: `H1`
//! above the real axis, `K` along it, and `Y` past `|z| ~ 30` all lose
//! their digits, because each is a small quantity computed from large
//! ones. The documentation called for "a scaled formulation (the AMOS
//! approach), which is not implemented here". This is that formulation.
//!
//! # Scaling alone is not the fix
//!
//! Worth being blunt about, because the name invites the wrong idea:
//! multiplying an already-cancelled `H1` by `exp(-iz)` recovers nothing.
//! The digits are gone before the multiplication. Scaling is only useful
//! when the **algorithm** never forms the large intermediate, and that
//! requires a different expansion:
//!
//! ```text
//!   exp(-iz) H1_nu(z) ~ sqrt(2/(pi z)) exp(-i(nu pi/2 + pi/4)) S(i)
//!   exp(+iz) H2_nu(z) ~ sqrt(2/(pi z)) exp(+i(nu pi/2 + pi/4)) S(-i)
//!   exp(z)   K_nu(z)  ~ sqrt(pi/(2z))                          S(1)
//!   exp(-z)  I_nu(z)  ~ 1/sqrt(2 pi z)                         S(-1)
//!
//!   where  S(c) = sum_k c^k a_k(nu) / z^k
//!          a_0 = 1,  a_k = a_{k-1} (4nu^2 - (2k-1)^2) / (8k)
//! ```
//!
//! (DLMF 10.17.5, 10.17.6, 10.40.2, 10.40.1.) Each right-hand side is a
//! plain series in `1/z` whose leading term is 1. **There is no
//! cancellation anywhere in it**, so the scaled quantity comes out to
//! full precision no matter how extreme the exponential it stands in
//! for. `J` and `Y` then follow from `H1` and `H2` without ever forming
//! the growing exponential:
//!
//! ```text
//!   exp(-|Im z|) J_nu(z) = [e^(iz - |Im z|) H1s + e^(-iz - |Im z|) H2s] / 2
//! ```
//!
//! and both those exponentials have modulus at most 1 by construction.
//!
//! # Asymptotic series need a stopping rule, and it doubles as a test
//!
//! `S(c)` diverges. Its terms fall while `2k - 1 < 2|nu|` is false and
//! `|z|` is large, then turn and grow, and the best a truncated
//! asymptotic series can do is about the size of its smallest term.
//! So the sum stops when a term stops shrinking, and **reports that
//! term's size as its own error estimate**.
//!
//! That estimate is what selects the method. If it is small enough, the
//! asymptotic result is used; otherwise the ascending-series routines
//! are, since the two are accurate in complementary places (`|z|` small
//! versus `|z|` large). If neither can deliver, the routine **returns an
//! error** rather than a plausible wrong number — which is a real change
//! from the unscaled routines, where `bessel_y_c(0, 40)` used to return
//! a confident first digit that was wrong.
//!
//! # Where the asymptotic does not reach, and what covers it
//!
//! The expansion is in `1/z` at fixed order, so it needs `|z|` large
//! compared with `nu^2`, not merely large. At `nu = 10` it wants
//! `|z| >~ 100`. Large order is therefore a different problem, and it
//! is answered by a different expansion — in `1/nu` — which lives in
//! [`crate::debye`] and is offered here as a further candidate. The two
//! together cover the real axis at every order tried up to 1000.
//!
//! What is left is not a method gap but a **representation** one: for
//! `z` well below `nu`, `J` is smaller than the smallest `f64` and `Y`
//! larger than the largest. Those points return an error saying so, and
//! quoting the logarithm, which is a different statement from "nothing
//! was accurate enough" and the more useful one.
//!
//! The uniform **Airy-type** expansion of DLMF 10.20 covers the turning
//! point itself and is in [`crate::airy_uniform`]; it is offered here as
//! a further candidate and wins in the band around `z ~ 0.85 nu` to
//! `0.98 nu`, where it is three to four orders better than anything
//! else available.

use crate::bessel_complex::{
    bessel_i_c, bessel_i_nu, bessel_j_c, bessel_j_nu, bessel_k_nu, bessel_y_nu,
};
use crate::complex::Complex64 as C;

/// The largest relative error an asymptotic evaluation may claim before
/// the ascending series is preferred instead. One digit short of `f64`
/// resolution: past this the series is usually the better of the two,
/// and where it is not, both are reported as failing.
const ASYM_TOL: f64 = 1e-15;

/// Terms are never taken past this; a legitimate optimal truncation is
/// far shorter, so reaching the cap means the series was not converging
/// in the first place.
const MAX_TERMS: usize = 100;

/// How wrong the ascending-series fallback may be estimated to be
/// before the routine refuses instead. Six digits: below that the answer
/// is not worth returning, and returning it anyway is exactly the
/// behaviour these routines exist to replace.
const SERIES_TOL: f64 = 1e-6;

/// Estimated relative error of the ascending-series route, from the
/// laws measured in `bessel_complex`: the working precision is spent on
/// the ratio between the largest quantity the series forms, `exp(|z|)`,
/// and the answer.
///
/// `loss` is the exponent for the unscaled function; the scaled form
/// divides by an exponential of modulus `exp(scale)`, which multiplies
/// the relative error by nothing but changes what "the answer" is, so
/// the caller passes the combined exponent directly.
fn series_error(loss: f64) -> f64 {
    if loss > 700.0 {
        return f64::INFINITY;
    }
    1e-16 * loss.exp()
}

/// `sum_k c^k a_k(nu) / z^k`, truncated where the terms stop shrinking.
///
/// Returns the sum and an estimate of its relative error — the size of
/// the first omitted term, which is the classical bound for an optimally
/// truncated asymptotic series. The sum's leading term is 1, so an
/// absolute term size is already a relative error.
fn asym_sum(nu: C, z: C, c: C) -> (C, f64) {
    // `mu = 4 nu^2` is the only place the order enters, and it enters
    // polynomially — so this works for a COMPLEX order with no change
    // beyond the type. DLMF 10.17.5/10.17.6 are stated for fixed `nu`,
    // which may be complex.
    let mu = nu * nu * 4.0;
    let step = c * z.inv();
    let mut term = C::ONE;
    let mut sum = C::ONE;
    let mut smallest = 1.0_f64;

    for k in 1..=MAX_TERMS {
        let f = (mu - C::real(((2 * k - 1) as f64).powi(2))) * (1.0 / (8.0 * k as f64));
        let next = term * step * f;
        let m = next.abs();
        // Optimal truncation: stop BEFORE the first term that does not
        // shrink. Adding it would make the answer worse, not better.
        if !m.is_finite() || m >= smallest {
            return (sum, m.min(smallest * 4.0));
        }
        sum = sum + next;
        term = next;
        smallest = m;
        // An exactly-terminating case: nu = half an odd integer makes
        // some a_k vanish and every later one with it. Nothing is left
        // to estimate, so the result is exact.
        if m == 0.0 {
            return (sum, 0.0);
        }
    }
    (sum, smallest)
}

/// The truncation estimate alone, for measurement and diagnostics.
#[doc(hidden)]
pub fn asym_error_estimate(nu: f64, z: C) -> f64 {
    asym_sum(C::real(nu), z, C::ONE).1
}

/// `sqrt(2/(pi z))`, the common Hankel prefactor.
fn hankel_prefactor(z: C) -> C {
    (C::real(2.0 / std::f64::consts::PI) * z.inv()).powf(0.5)
}

fn check(nu: f64, z: C, what: &str) -> Result<(), String> {
    if !nu.is_finite() {
        return Err(format!("{what}: the order must be finite, got {nu}"));
    }
    if !z.is_finite() {
        return Err(format!("{what}: z must be finite, got {z:?}"));
    }
    if z.abs() == 0.0 {
        return Err(format!("{what}: the scaled forms are undefined at z = 0"));
    }
    Ok(())
}

/// The message produced when neither method reaches the requested point.
///
/// It names both estimates, because "I cannot do this" is only useful
/// if it says how badly and which way out is missing.
fn no_method(what: &str, nu: f64, z: C, asym_err: f64, series_err: f64) -> String {
    // Two failures that look the same from outside are worth telling
    // apart: no method reached the point, versus the value is perfectly
    // well determined and simply outside f64. At nu = 400.5, x = 40 the
    // true J is about e^-1013 and Y about e^+1010.
    if z.im == 0.0 && z.re > 0.0 {
        if let Some((lj, ly)) = crate::debye::jy_log_magnitude(nu.abs(), z.re) {
            let l = if what.contains("_j_") { lj } else { ly };
            if !(-745.0..=709.0).contains(&l) && (what.contains("_j_") || what.contains("_y_")) {
                return format!(
                    "{what}: the value at nu = {nu}, z = {:?} is outside f64 range — \
                     its natural logarithm is about {l:.0}, against a representable \
                     range of about -745 to 709. This is not a failure of method: \
                     the large-order expansion determines it, but no scaling this \
                     crate offers can carry it as an f64.",
                    z.re
                );
            }
        }
    }
    format!(
        "{what}: neither method is accurate at nu = {nu}, z = {z:?}. \
         The ascending series would have a relative error of about \
         {series_err:.1e} here, and the asymptotic expansion — which needs \
         |z| large compared with nu^2, not merely large — truncates at \
         about {asym_err:.1e}. Both are out of range here. The general \
         remedy is the uniform Airy-type expansions of DLMF 10.20, which \
         cover the region where |z| and nu are comparable; they are in \
         `crate::airy_uniform` and are offered as candidates where they \
         apply, so this point is beyond them too."
    )
}

// ---------------------------------------------------------------------
// Choosing a method
// ---------------------------------------------------------------------
//
// Two candidates exist at every point: the asymptotic expansion above,
// and the ascending-series routines of `bessel_complex`. Each carries an
// error estimate — the asymptotic's from its own optimal truncation, the
// series' from the laws measured in `bessel_complex` — so the choice is
// made by COMPARING THEM, not by a hard radius.
//
// That distinction is not cosmetic. A hard `|z| >= 18` switch left a
// hole in `K` between x = 8 and x = 17 where the asymptotic had been
// declared not-yet-good and the series already-not-good, even though the
// asymptotic was delivering ten correct digits throughout. Comparing the
// estimates closes it, and the crossover lands where it should.

/// `nu` as a whole number, if it is one (within the tolerance the
/// `bessel_complex` routines themselves use).
fn whole(nu: f64) -> Option<i32> {
    let n = nu.round();
    if (nu - n).abs() < 1e-9 && n.abs() < i32::MAX as f64 {
        Some(n as i32)
    } else {
        None
    }
}

/// A candidate answer together with its estimated relative error.
///
/// `None` means the route could not produce anything usable here — it
/// errored, or overflowed to a non-finite value. **A failed fallback
/// must not fail the call**: the first version of this module wrote
/// `series_route(nu, z)?` and so propagated the series' overflow at
/// `x = 50` even though the asymptotic had the answer to 45 digits.
/// Every route is now optional, and only the comparison decides.
type Candidate = Option<(C, f64)>;

/// Wrap a routine's result and its loss exponent into a [`Candidate`],
/// discarding anything that errored or came back non-finite.
fn candidate(v: Result<C, String>, loss: f64) -> Candidate {
    match v {
        Ok(x) if x.is_finite() => Some((x, series_error(loss))),
        _ => None,
    }
}

/// Reject an exact zero. `I` and `K` have no zeros, so a returned zero
/// is underflow wearing an answer's clothes — and it arrives with a
/// small claimed error, which is worse than arriving with none.
fn nonzero(c: Candidate) -> Candidate {
    c.filter(|(v, _)| v.abs() > 0.0)
}

/// The better of two candidates, or whichever one exists.
fn better(a: Candidate, b: Candidate) -> Candidate {
    match (a, b) {
        (Some(x), Some(y)) => Some(if x.1 <= y.1 { x } else { y }),
        (x, None) => x,
        (None, y) => y,
    }
}

/// Turn a candidate into a result, refusing anything too inaccurate to
/// be worth returning.
fn accept(c: Candidate, what: &str, nu: f64, z: C, asym_err: f64) -> Result<C, String> {
    match c {
        Some((v, e)) if e <= SERIES_TOL => Ok(v),
        Some((_, e)) => Err(no_method(what, nu, z, asym_err, e)),
        None => Err(no_method(what, nu, z, asym_err, f64::INFINITY)),
    }
}

/// `J_nu(z)` by whichever ascending route is better, with its loss.
///
/// A whole order can use Miller recurrence, whose loss is `|Im z|` — far
/// better on the real axis than the series' `|z| - |Im z|`. Non-integer
/// orders have only the series.
fn j_route(nu: f64, z: C) -> Candidate {
    match whole(nu) {
        Some(n) if n >= 0 => candidate(bessel_j_c(n, z), z.im.abs()),
        _ => candidate(bessel_j_nu(nu, z), z.abs() - z.im.abs()),
    }
}

fn y_route(nu: f64, z: C) -> Candidate {
    // Y is an ascending series either way, so the law is the same.
    candidate(bessel_y_nu(nu, z), z.abs() - z.im.abs())
}

fn i_route(nu: f64, z: C) -> Candidate {
    match whole(nu) {
        Some(n) if n >= 0 => candidate(bessel_i_c(n, z), z.re.abs()),
        _ => candidate(bessel_i_nu(nu, z), z.abs() - z.re),
    }
}

fn k_route(nu: f64, z: C) -> Candidate {
    // A whole order goes through J and Y at imaginary argument, which
    // adds a Miller factor on top of the series cancellation.
    let loss = match whole(nu) {
        Some(_) => (2.0 * z.re.abs()).max(z.abs()) + z.re,
        None => z.abs() + z.re,
    };
    candidate(bessel_k_nu(nu, z), loss)
}

/// The scaled Hankel pair from the ascending routes, as candidates.
///
/// The scaling exponential is applied to the assembled value, so the
/// loss exponent is [`hankel_loss`] and not the ingredients' own.
fn hankel_series_candidates(nu: f64, z: C) -> (Candidate, Candidate) {
    let (Some((j, _)), Some((y, _))) = (j_route(nu, z), y_route(nu, z)) else {
        return (None, None);
    };
    // Recover the raw loss exponents; `candidate` has already turned
    // them into errors, so recompute them here rather than thread them.
    let l_j = match whole(nu) {
        Some(n) if n >= 0 => z.im.abs(),
        _ => z.abs() - z.im.abs(),
    };
    let l_y = z.abs() - z.im.abs();
    let h1 = (j + C::I * y) * (C::I * z * -1.0).exp();
    let h2 = (j - C::I * y) * (C::I * z).exp();
    (
        (h1.is_finite()).then(|| (h1, series_error(hankel_loss(l_j, l_y, z, 1.0)))),
        (h2.is_finite()).then(|| (h2, series_error(hankel_loss(l_j, l_y, z, -1.0)))),
    )
}

/// Loss exponent for a scaled Hankel value assembled from `J` and `Y`.
///
/// `H1s = e^(-iz)(J + iY)` has modulus about `1/sqrt(|z|)` while `J` and
/// `Y` are about `e^(|Im z|)/sqrt(|z|)`, so an ingredient's relative
/// error `e^L` becomes `e^(L + |Im z| + s*Im z)` in the result, with
/// `s = +1` for `H1` and `-1` for `H2`.
fn hankel_loss(l_j: f64, l_y: f64, z: C, s: f64) -> f64 {
    l_j.max(l_y) + z.im.abs() + s * z.im
}

/// Upward recurrence in ORDER, which is stable for every function in
/// this module that needs it: `K`, `Y`, `H1` and `H2` all GROW with
/// order, so the wanted solution is the dominant one.
///
/// `sign` is `-1` for the cylinder recurrence
/// `C_{nu+1} = (2nu/z) C_nu - C_{nu-1}` and `+1` for the modified one
/// `K_{nu+1} = (2nu/z) K_nu + K_{nu-1}` (DLMF 10.6.1, 10.29.1). The
/// scaled forms obey the same relations, since the exponential factor
/// does not depend on the order.
fn recur_up(nu: f64, z: C, mut a: C, mut b: C, base: f64, steps: u32, sign: f64) -> C {
    let zi = z.inv();
    for k in 0..steps {
        let order = base + 1.0 + k as f64;
        let c = b * (zi * (2.0 * order)) + a * sign;
        a = b;
        b = c;
    }
    let _ = nu;
    b
}

/// Split `nu >= 2` into a base order in `[0, 1)` plus a whole number of
/// upward steps. The asymptotic expansion is at its best at low order,
/// so this is how a large order is reached at moderate `|z|`.
fn recurrence_plan(nu: f64) -> Option<(f64, u32)> {
    if nu < 2.0 {
        return None;
    }
    let steps = nu.floor();
    Some((nu - steps, steps as u32))
}

// ---------------------------------------------------------------------
// Hankel
// ---------------------------------------------------------------------

/// The sectors in which the Hankel asymptotic expansions are valid.
///
/// DLMF 10.17.5 holds for `-pi < arg z < 2pi` and 10.17.6 for
/// `-2pi < arg z < pi`, so the negative real axis is **interior** for
/// one and **on the boundary** for the other. Using the pair there gave
/// a J-Y Wronskian residual of 7.3 at `z = -15` — the expansion was not
/// wrong by a little, it was outside its sector. A margin of `pi/4` is
/// kept from each boundary.
const SECTOR_MARGIN: f64 = std::f64::consts::FRAC_PI_4;

fn h1_sector_ok(z: C) -> bool {
    z.arg() > -std::f64::consts::PI + SECTOR_MARGIN
}

fn h2_sector_ok(z: C) -> bool {
    z.arg() < std::f64::consts::PI - SECTOR_MARGIN
}

/// Both scaled Hankel values by the asymptotic expansion alone, with the
/// worse of the two truncation estimates.
fn hankel_pair_asym(nu: f64, z: C) -> (C, C, f64) {
    let (s1, e1) = asym_sum(C::real(nu), z, C::I);
    let (s2, e2) = asym_sum(C::real(nu), z, C::I * -1.0);
    let pref = hankel_prefactor(z);
    let ph = nu * std::f64::consts::FRAC_PI_2 + std::f64::consts::FRAC_PI_4;
    (
        pref * (C::I * -1.0 * ph).exp() * s1,
        pref * (C::I * ph).exp() * s2,
        e1.max(e2),
    )
}

/// Both scaled Hankel values, by the asymptotic expansion directly or by
/// recurring up in order from where the expansion is good.
fn hankel_pair_best(nu: f64, z: C) -> (C, C, f64) {
    let direct = hankel_pair_asym(nu, z);
    if direct.2 <= ASYM_TOL {
        return direct;
    }
    let Some((base, steps)) = recurrence_plan(nu) else {
        return direct;
    };
    let (a1, a2, e0) = hankel_pair_asym(base, z);
    let (b1, b2, e1) = hankel_pair_asym(base + 1.0, z);
    let err = e0.max(e1);
    if err >= direct.2 {
        return direct;
    }
    // Off the real axis one member of the pair is exponentially
    // RECESSIVE — `H1 ~ e^(iz)` and `H2 ~ e^(-iz)`, so one is
    // `e^(2|Im z|)` smaller than the other. Recurring in order mixes
    // them, and the recessive one's contribution is only determined to
    // that same factor. Measured: at nu = 30, z = 25e^(-2i), the
    // recurrence returned an `H1` too small by 2e5 while claiming
    // 1e-23, and the giveaway was that `Y` came back exactly equal to
    // `-i J`, which only happens when one member has been lost. The
    // estimate must carry the factor; on the real axis it is 1 and
    // nothing changes.
    let recessive = (2.0 * z.im.abs()).min(700.0).exp();
    // A recurrence of `steps` stages accumulates rounding, so its floor
    // is `steps * eps` and not `eps`. Measured: at nu = 100.5, x = 85.4
    // the route claimed 1.4e-11 and delivered 1.4e-9 — exactly the
    // hundred steps it had taken — and on that claim it was beating the
    // Airy-type expansion, which was giving 3.7e-15 there.
    let floor = steps as f64 * 1e-16;
    (
        recur_up(nu, z, a1, b1, base, steps - 1, -1.0),
        recur_up(nu, z, a2, b2, base, steps - 1, -1.0),
        (err * recessive).max(floor),
    )
}

/// `exp(-iz) H1_nu(z)` — the outgoing Hankel function with its
/// travelling-wave factor removed (DLMF 10.17.5,
/// <https://dlmf.nist.gov/10.17.E5>).
///
/// This is the function to use above the real axis, where `H1` itself is
/// a small difference of large numbers. Multiply by `exp(iz)` when you
/// want the unscaled value — but only if that product is representable
/// and you can afford the digits, which is the whole reason this form
/// exists.
///
/// # Errors
/// `z = 0`, a non-finite argument, or a point where neither the
/// asymptotic expansion nor the ascending series is accurate (see the
/// module note on the transition region).
///
/// # Examples
/// ```
/// use special_functions::bessel_scaled::hankel_h1_scaled_nu;
/// use special_functions::complex::Complex64 as C;
/// // At nu = 1/2 the answer is elementary and EXACT:
/// // H1_{1/2}(z) = -i sqrt(2/(pi z)) exp(iz), so the scaled form is
/// // just the prefactor, with no exponential left in it at all.
/// let z = C::new(3.0, 25.0);          // far above the real axis
/// let got = hankel_h1_scaled_nu(0.5, z).unwrap();
/// let want = C::I * -1.0
///     * (C::real(2.0 / std::f64::consts::PI) * z.inv()).powf(0.5);
/// assert!((got - want).abs() / want.abs() < 1e-14);
/// ```
pub fn hankel_h1_scaled_nu(nu: f64, z: C) -> Result<C, String> {
    check(nu, z, "hankel_h1_scaled")?;
    let (h1, _, asym_err) = hankel_pair_best(nu, z);
    let a = (h1.is_finite() && h1_sector_ok(z)).then_some((h1, asym_err));
    let (s, _) = hankel_series_candidates(nu, z);
    accept(better(a, s), "hankel_h1_scaled", nu, z, asym_err)
}

/// `exp(iz) H2_nu(z)` — the incoming Hankel function, scaled
/// (DLMF 10.17.6). The mirror of [`hankel_h1_scaled_nu`]: this is the
/// form to use *below* the real axis.
///
/// # Errors
/// As [`hankel_h1_scaled_nu`].
pub fn hankel_h2_scaled_nu(nu: f64, z: C) -> Result<C, String> {
    check(nu, z, "hankel_h2_scaled")?;
    let (_, h2, asym_err) = hankel_pair_best(nu, z);
    let a = (h2.is_finite() && h2_sector_ok(z)).then_some((h2, asym_err));
    let (_, s) = hankel_series_candidates(nu, z);
    accept(better(a, s), "hankel_h2_scaled", nu, z, asym_err)
}

// ---------------------------------------------------------------------
// Modified
// ---------------------------------------------------------------------

fn k_asym(nu: f64, z: C) -> (C, f64) {
    let (s, e) = asym_sum(C::real(nu), z, C::ONE);
    let pref = (C::real(std::f64::consts::PI * 0.5) * z.inv()).powf(0.5);
    (pref * s, e)
}

fn k_best(nu: f64, z: C) -> (C, f64) {
    let direct = k_asym(nu, z);
    if direct.1 <= ASYM_TOL {
        return direct;
    }
    let Some((base, steps)) = recurrence_plan(nu) else {
        return direct;
    };
    let (a, e0) = k_asym(base, z);
    let (b, e1) = k_asym(base + 1.0, z);
    let err = e0.max(e1);
    if err >= direct.1 {
        return direct;
    }
    (recur_up(nu, z, a, b, base, steps - 1, 1.0), err)
}

/// The body of [`bessel_k_scaled_nu`], exposing the candidate so `I`'s
/// Wronskian anchor can inherit an honest error estimate.
fn k_scaled_candidate(nu: f64, z: C) -> (Candidate, f64) {
    let (k_a, asym_err) = k_best(nu.abs(), z);
    let a = k_a.is_finite().then_some((k_a, asym_err));
    let s = k_route(nu.abs(), z)
        .and_then(|(v, e)| (v * z.exp()).is_finite().then(|| (v * z.exp(), e)));
    (better(a, s), asym_err)
}

/// `I_{nu+1}(z) / I_nu(z)` by its continued fraction, from the
/// recurrence `I_{nu-1} - I_{nu+1} = (2 nu/z) I_nu` (DLMF 10.29.1)
/// rearranged into `r_nu = 1/(2 nu/z + r_{nu+1})`.
///
/// Evaluated by modified Lentz, which self-terminates: no guess about
/// how many terms are needed, and the convergence test is the answer's
/// own. The ratio is well conditioned everywhere — it is the ratio of
/// two members of the *recessive* solution, which is what makes it
/// usable where `I` itself cannot be computed directly.
///
/// Returns `None` if it fails to converge.
fn i_ratio_cf(nu: f64, z: C) -> Option<C> {
    let tiny = 1e-300;
    let zi = z.inv();
    let mut f = C::real(tiny);
    let mut c = f;
    let mut d = C::ZERO;
    for j in 1..=10_000 {
        let b = zi * (2.0 * (nu + j as f64));
        d = b + d;
        if d.abs() == 0.0 {
            d = C::real(tiny);
        }
        c = b + c.inv();
        if c.abs() == 0.0 {
            c = C::real(tiny);
        }
        d = d.inv();
        let delta = c * d;
        f = f * delta;
        if (delta - C::ONE).abs() < 1e-16 {
            return f.is_finite().then_some(f);
        }
    }
    None
}

/// `exp(-|Re z|) I_nu(z)` from the Wronskian
/// `I_nu K_{nu+1} + I_{nu+1} K_nu = 1/z` (DLMF 10.28.2), using the
/// continued-fraction ratio to eliminate `I_{nu+1}`.
///
/// This is the classical anchor, and here it is what makes `I` reach as
/// far as `K` does: at `nu = 30, z = 200` the asymptotic expansion wants
/// `|z| >> nu^2 = 900` and the ascending series has cancelled away 200
/// nepers, but `K` is exact and the ratio is well conditioned, so `I`
/// comes out of the two of them.
fn i_from_wronskian(nu: f64, z: C) -> Candidate {
    let r = i_ratio_cf(nu, z)?;
    let (k0, _) = k_scaled_candidate(nu, z);
    let (k1, _) = k_scaled_candidate(nu + 1.0, z);
    let ((k0, e0), (k1, e1)) = (k0?, k1?);
    // Scaled: Is_nu Ks_{nu+1} + Is_{nu+1} Ks_nu = e^(z - |Re z|)/z,
    // and Is_{nu+1} = r Is_nu.
    let denom = z * (k1 + k0 * r);
    if denom.abs() == 0.0 {
        return None;
    }
    let v = (z - C::real(z.re.abs())).exp() * denom.inv();
    // The anchor divides by a sum of two positive-real-part quantities,
    // so it neither amplifies nor cancels; the error is the K estimates'.
    v.is_finite().then_some((v, e0.max(e1)))
}

/// `exp(z) K_nu(z)` (DLMF 10.40.2, <https://dlmf.nist.gov/10.40.E2>).
///
/// `K` decays like `exp(-z)`, so this is the form that stays O(1) and,
/// more to the point, the form that can be computed without subtracting
/// two `I`s. Unscaled `K` on the real axis is worthless past about
/// `x = 12` and has underflowed to zero entirely by `x = 745`; this is
/// accurate everywhere the expansion reaches, which on the real axis is
/// everywhere at all.
///
/// # Errors
/// As [`hankel_h1_scaled_nu`].
///
/// # Examples
/// ```
/// use special_functions::bessel_scaled::bessel_k_scaled_nu;
/// use special_functions::complex::Complex64 as C;
/// // K_{1/2}(z) = sqrt(pi/(2z)) exp(-z) exactly, so exp(z) K_{1/2}(z)
/// // is the prefactor alone — at any z, however large.
/// let z = C::real(200.0);
/// let got = bessel_k_scaled_nu(0.5, z).unwrap();
/// let want = (C::real(std::f64::consts::PI * 0.5) * z.inv()).powf(0.5);
/// assert!((got - want).abs() / want.abs() < 1e-15);
/// ```
pub fn bessel_k_scaled_nu(nu: f64, z: C) -> Result<C, String> {
    check(nu, z, "bessel_k_scaled")?;
    // K_{-nu} = K_nu, so only the magnitude of the order matters.
    let (c, asym_err) = k_scaled_candidate(nu.abs(), z);
    let u = ik_uniform_candidates(nu.abs(), z).1;
    accept(nonzero(better(c, u)), "bessel_k_scaled", nu, z, asym_err)
}

/// `exp(-|Re z|) I_nu(z)`.
///
/// `I` grows like `exp(Re z)`, so unscaled it overflows `f64` at around
/// `Re z = 710` while the scaled form stays O(1) forever.
///
/// The asymptotic expansion (DLMF 10.40.1) is used only within 45
/// degrees of the real axis. Its second, exponentially small term is
/// dropped there, which is legitimate because it is smaller than the
/// truncation error of the first; nearer the imaginary axis the two
/// become comparable and dropping one would be wrong — and that is
/// precisely where the ascending series is at its best, since `I`'s
/// cancellation grows with `|z| - Re z`.
///
/// # Errors
/// As [`hankel_h1_scaled_nu`].
pub fn bessel_i_scaled_nu(nu: f64, z: C) -> Result<C, String> {
    check(nu, z, "bessel_i_scaled")?;
    // Re z < 0 is reflected away first. `I_nu(z e^(m pi i)) =
    // e^(m nu pi i) I_nu(z)` (DLMF 10.34.1), and the scaling `exp(-|Re z|)`
    // is unchanged by the reflection, so the scaled values differ by the
    // same phase.
    if z.re < 0.0 {
        let m = if z.im >= 0.0 { 1.0 } else { -1.0 };
        let v = bessel_i_scaled_nu(nu, z * -1.0)?;
        return Ok(v * (C::I * (m * std::f64::consts::PI * nu)).exp());
    }
    // Near the imaginary axis `I` is `J` in disguise —
    // `I_nu(z) = e^(-i nu pi/2) J_nu(iz)` (DLMF 10.27.6) — and `iz` is
    // then near the REAL axis, which is exactly where `J` is at its
    // best. The scalings line up: `|Im(iz)| = |Re z|`.
    let via_j = if z.re < z.im.abs() {
        let (c, _) = j_scaled_candidate(nu, C::I * z);
        nonzero(c.and_then(|(v, e)| {
            let ph = (C::I * (-std::f64::consts::FRAC_PI_2 * nu)).exp();
            let w = v * ph;
            w.is_finite().then_some((w, e))
        }))
    } else {
        None
    };
    let mut asym_err = f64::INFINITY;
    let mut a: Candidate = None;
    if z.re > 0.0 && z.re >= z.im.abs() {
        let (sum, e) = asym_sum(C::real(nu), z, C::ONE * -1.0);
        let pref = (C::real(1.0 / (2.0 * std::f64::consts::PI)) * z.inv()).powf(0.5);
        // exp(z - |Re z|) has modulus 1 here, so nothing large is formed.
        let v = pref * (z - C::real(z.re.abs())).exp() * sum;
        // DLMF 10.40.1 has a SECOND term, of relative size exp(-2 Re z),
        // which is dropped here. The truncation estimate cannot see it,
        // and at nu = 1/2 that is fatal: every a_k past the first
        // vanishes, so the truncation estimate is exactly 0 and the
        // dropped term is the entire error. Measured, this showed up as
        // a 0.69 residual in the I-K Wronskian at |z| = 0.5. The
        // estimate must therefore carry both.
        asym_err = e.max((-2.0 * z.re).exp());
        a = v.is_finite().then_some((v, asym_err));
    }
    let s = i_route(nu, z).and_then(|(v, e)| {
        let w = v * (-z.re.abs()).exp();
        w.is_finite().then_some((w, e))
    });
    let w = i_from_wronskian(nu.abs(), z);
    let u = ik_uniform_candidates(nu.abs(), z).0;
    accept(
        nonzero(better(better(better(better(a, s), w), u), via_j)),
        "bessel_i_scaled",
        nu,
        z,
        asym_err,
    )
}

// ---------------------------------------------------------------------
// Ordinary, via the Hankel pair
// ---------------------------------------------------------------------

/// Both scaled Hankel values with their scaling exponentials folded in,
/// or `None` if the asymptotic route is not available here.
///
/// The exponentials `exp(±iz - |Im z|)` are formed as single `exp` calls
/// with non-positive real part, so neither can overflow and the large
/// factor is never built. That is the entire trick.
fn scaled_pair(nu: f64, z: C) -> Option<(C, C, f64)> {
    // J and Y need BOTH members, so only the intersection of the two
    // sectors will do: |arg z| <= 3pi/4.
    if !h1_sector_ok(z) || !h2_sector_ok(z) {
        return None;
    }
    let (h1, h2, err) = hankel_pair_best(nu, z);
    if !h1.is_finite() || !h2.is_finite() {
        return None;
    }
    let a = C::real(z.im.abs());
    Some((
        h1 * (C::I * z - a).exp(),
        h2 * (C::I * z * -1.0 - a).exp(),
        err,
    ))
}

/// How much accuracy forming `J` or `Y` from the Hankel pair costs.
///
/// Measured from the values themselves rather than modelled: the sum of
/// the ingredient magnitudes over the magnitude of the result. It is 1
/// when nothing cancels and enormous when everything does — for example
/// at `nu = 40, x = 5`, where `H1` is about `1e32` and `J` about
/// `1e-32`, so building `J` from the pair would be worthless. Without
/// this factor the routine would return that value with a confident
/// `1e-5` error estimate attached.
/// The measured cancellation, floored at machine epsilon.
///
/// The floor is not cosmetic. At `nu = 1/2` the asymptotic terminates
/// exactly and the truncation estimate is 0, so multiplying it by any
/// cancellation factor left 0 — and a `J` built from two Hankel values
/// that cancel by `1e14` was returned with a claimed error of zero. At
/// `nu = 400.5, z = 240` that value was wrong by a factor of `5e89`.
/// An exact expansion is still only evaluated to `f64` precision.
const EVAL_FLOOR: f64 = 1e-16;

fn cancellation(h1: C, h2: C, result: C) -> f64 {
    let top = h1.abs() + h2.abs();
    let bottom = result.abs();
    if bottom == 0.0 || !bottom.is_finite() {
        return f64::INFINITY;
    }
    (top / bottom).max(1.0)
}

/// `exp(-|Im z|) J_nu(z)`.
///
/// `J` is the one member of the family that Miller's recurrence already
/// handles well, so this exists for uniformity and for very large `|z|`,
/// where the recurrence's seed order — and so its cost — grows with
/// `|z|` while the asymptotic series only gets shorter.
///
/// # Errors
/// As [`hankel_h1_scaled_nu`].
pub fn bessel_j_scaled_nu(nu: f64, z: C) -> Result<C, String> {
    check(nu, z, "bessel_j_scaled")?;
    let (c, asym_err) = j_scaled_candidate(nu, z);
    accept(c, "bessel_j_scaled", nu, z, asym_err)
}

/// The large-order routes from [`crate::debye`], as candidates.
///
/// These are expansions in `1/nu` rather than `1/z`, so they cover
/// exactly what the rest of this module cannot: `z` below `nu`, where
/// `J` is exponentially small and every method here built it as the
/// difference of two exponentially large numbers.
///
/// On the real axis the `exp(-|Im z|)` scaling is 1, so the Debye values
/// need no adjustment; off it they do not apply and return `None`.
/// Safety factor on the large-order truncation estimates.
///
/// Optimal truncation gives the size of the first omitted term, which
/// is an estimate and not a bound — and measured, it runs optimistic.
/// At `nu = 10, x = 5` the DLMF 10.41 expansion for `K` claimed better
/// than the 1/z route and delivered 3.6e-10 against that route's
/// 1.4e-16, so it was winning comparisons it should have lost. Ten is
/// enough to order them correctly without discarding the regions where
/// these are the only methods there are.
const LARGE_ORDER_SAFETY: f64 = 10.0;

fn jy_debye_candidates(nu: f64, z: C) -> (Candidate, Candidate) {
    if z.im != 0.0 || z.re <= 0.0 || nu <= 0.0 {
        return (None, None);
    }
    let (j, y) = crate::debye::jy_debye(nu, z.re);
    let (aj, ay) = crate::airy_uniform::jy_airy(nu, z.re);
    let e = |u: crate::debye::Uniform| {
        (u.value, (u.err * LARGE_ORDER_SAFETY).max(EVAL_FLOOR))
    };
    // Olver's Airy-type expansion (DLMF 10.20) is offered alongside the
    // Debye one. They are uniformly valid in complementary places — the
    // Debye expansions away from the turning point, the Airy-type one
    // through it — and both report optimal-truncation estimates, so the
    // selector picks between them on measured terms.
    (better(j.map(e), aj.map(e)), better(y.map(e), ay.map(e)))
}

/// The DLMF 10.41 uniform expansions for `I` and `K`, as candidates in
/// this module's scalings.
fn ik_uniform_candidates(nu: f64, z: C) -> (Candidate, Candidate) {
    if nu <= 0.0 || z.re <= 0.0 || z.arg().abs() >= std::f64::consts::FRAC_PI_2 {
        return (None, None);
    }
    let (i, k) = crate::debye::ik_uniform(C::real(nu), z);
    // An exact zero here is underflow, not an answer: I and K have no
    // zeros. Returning it with a small claimed error is the same lie
    // these routines exist to stop telling.
    let e = |u: crate::debye::Uniform| {
        (u.value.abs() > 0.0)
            .then(|| (u.value, (u.err * LARGE_ORDER_SAFETY).max(EVAL_FLOOR)))
    };
    (i.and_then(e), k.and_then(e))
}

/// The body of [`bessel_j_scaled_nu`], returning the candidate and the
/// asymptotic estimate so that `I` can build on it and inherit an
/// honest error rather than a nominal one.
fn j_scaled_candidate(nu: f64, z: C) -> (Candidate, f64) {
    let (a, asym_err) = match scaled_pair(nu, z) {
        Some((h1, h2, e)) => {
            let v = (h1 + h2) * 0.5;
            let e = e.max(EVAL_FLOOR) * cancellation(h1, h2, v);
            (v.is_finite().then_some((v, e)), e)
        }
        None => (None, f64::INFINITY),
    };
    let s = j_route(nu, z).and_then(|(v, e)| {
        let w = v * (-z.im.abs()).exp();
        w.is_finite().then_some((w, e))
    });
    let d = jy_debye_candidates(nu, z).0;
    (better(better(a, s), d), asym_err)
}

/// `exp(-|Im z|) Y_nu(z)`.
///
/// This closes the defect found in Stage 13: unscaled `Y_0` on the real
/// axis was wrong in the first digit by `x = 40`, because it came from
/// an ascending series whose terms are `exp(|z|)`. Here it comes from
/// the Hankel pair instead, and holds to `x = 5000` and beyond.
///
/// # Errors
/// As [`hankel_h1_scaled_nu`].
///
/// # Examples
/// ```
/// use special_functions::bessel_scaled::bessel_y_scaled_nu;
/// use special_functions::complex::Complex64 as C;
/// // On the real axis the scaling is 1, so this IS Y_0(40) — the value
/// // the unscaled routine gets wrong in its first digit.
/// let got = bessel_y_scaled_nu(0.0, C::real(40.0)).unwrap();
/// let want = spec_math::cephes64::yn(0, 40.0);
/// assert!((got.re - want).abs() < 1e-14);
/// assert!(got.im.abs() < 1e-15);
/// ```
pub fn bessel_y_scaled_nu(nu: f64, z: C) -> Result<C, String> {
    check(nu, z, "bessel_y_scaled")?;
    let (a, asym_err) = match scaled_pair(nu, z) {
        Some((h1, h2, e)) => {
            let v = (h1 - h2) / (C::I * 2.0);
            let e = e.max(EVAL_FLOOR) * cancellation(h1, h2, v);
            (v.is_finite().then_some((v, e)), e)
        }
        None => (None, f64::INFINITY),
    };
    let s = y_route(nu, z).and_then(|(v, e)| {
        let w = v * (-z.im.abs()).exp();
        w.is_finite().then_some((w, e))
    });
    let d = jy_debye_candidates(nu, z).1;
    accept(better(better(a, s), d), "bessel_y_scaled", nu, z, asym_err)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bessel_complex::{bessel_i_nu, bessel_j_nu, bessel_k_nu, bessel_y_nu};
    use spec_math::cephes64::{i0, i1, j0, kn, yv};

    fn close(a: C, b: C, tol: f64) -> bool {
        (a - b).abs() <= tol * b.abs().max(1e-300)
    }

    /// The half-integer closed forms are exact and, once scaled, contain
    /// no exponential at all — so they hold at `|z| = 1000` just as well
    /// as at `|z| = 1`. That is the sharpest statement of what scaling
    /// buys, and no unscaled routine here can even be asked it: at
    /// `Im z = 700` the ingredients `J` and `Y` are about `e^700`, past
    /// the top of `f64`.
    #[test]
    fn half_integer_closed_forms_hold_at_any_magnitude() {
        for &(re, im) in &[
            (1.0, 0.0),
            (30.0, 0.0),
            (200.0, 0.0),
            (1000.0, 0.0),
            (3.0, 25.0),
            (5.0, 300.0),
            (100.0, -700.0),
            (-40.0, 60.0),
        ] {
            let z = C::new(re, im);
            let want = C::I * -1.0 * hankel_prefactor(z);
            let got = hankel_h1_scaled_nu(0.5, z).unwrap();
            assert!(close(got, want, 1e-13), "H1s_1/2({z:?}): {got:?} vs {want:?}");
            let want = C::I * hankel_prefactor(z);
            let got = hankel_h2_scaled_nu(0.5, z).unwrap();
            assert!(close(got, want, 1e-13), "H2s_1/2({z:?}): {got:?} vs {want:?}");
        }
        // exp(z) K_{1/2}(z) = sqrt(pi/(2z)), likewise.
        for &(re, im) in &[(1.0, 0.0), (50.0, 0.0), (500.0, 0.0), (20.0, 40.0), (-3.0, 90.0)] {
            let z = C::new(re, im);
            let want = (C::real(std::f64::consts::PI * 0.5) * z.inv()).powf(0.5);
            let got = bessel_k_scaled_nu(0.5, z).unwrap();
            assert!(close(got, want, 1e-13), "Ks_1/2({z:?}): {got:?} vs {want:?}");
        }
    }

    /// The defect this module was written to close. `Y_0` on the real
    /// axis against Cephes, at and past the points where the unscaled
    /// routine was documented to fail.
    #[test]
    fn scaled_y_is_correct_where_the_unscaled_one_is_not() {
        use crate::bessel_complex::bessel_y_c;
        for &x in &[20.0, 25.0, 30.0, 40.0, 60.0, 120.0, 500.0, 5000.0] {
            let want = yv(0.0, x);
            let got = bessel_y_scaled_nu(0.0, C::real(x)).unwrap();
            assert!(got.im.abs() < 1e-14, "Y_0({x}) should be real, got {got:?}");
            assert!(
                (got.re - want).abs() <= 1e-12 * want.abs().max(0.05),
                "scaled Y_0({x}): {} vs {want}",
                got.re
            );
        }
        // This used to assert the CONTRAST — that the unscaled route was
        // wrong in its first digit at x = 40. Stage 19 fixed
        // `bessel_y_c` itself, so the contrast is gone and the two now
        // agree. Asserting the agreement is what keeps the record: if
        // `bessel_y_c` ever regresses, this fails here as well as in its
        // own module.
        let unscaled = bessel_y_c(0, C::real(40.0)).unwrap().re;
        let want = yv(0.0, 40.0);
        assert!(
            (unscaled - want).abs() <= 1e-12 * want.abs(),
            "bessel_y_c(0, 40) should now be right too: {unscaled} vs {want}"
        );
    }

    /// `Y_nu` across order as well, against Cephes `yv`, which takes a
    /// real order and is an entirely separate implementation. This is
    /// what exercises the upward recurrence in order.
    #[test]
    fn scaled_y_holds_across_order() {
        for &nu in &[0.0, 0.5, 1.3, 5.0, 10.0, 20.0, 40.0] {
            for &x in &[5.0, 15.0, 25.0, 40.0, 100.0, 1000.0] {
                let want = yv(nu, x);
                let got = bessel_y_scaled_nu(nu, C::real(x)).unwrap().re;
                assert!(
                    (got - want).abs() <= 1e-11 * want.abs().max(1e-3),
                    "Y_{nu}({x}): {got} vs {want}"
                );
            }
        }
    }

    /// `K` on the real axis, the other documented casualty. Unscaled it
    /// was worthless past `x = 12`; scaled it is exact far past the
    /// point where the unscaled value has underflowed to zero and where
    /// even the vendored Cephes `kn` overflows.
    #[test]
    fn scaled_k_is_correct_where_the_unscaled_one_is_not() {
        for &n in &[0_i32, 1, 2, 5, 10, 20] {
            for &x in &[1.0, 5.0, 15.0, 25.0, 50.0, 200.0] {
                let want = kn(n as isize, x) * x.exp();
                let got = bessel_k_scaled_nu(n as f64, C::real(x)).unwrap();
                assert!(
                    (got.re - want).abs() <= 1e-10 * want,
                    "exp(x) K_{n}({x}): {} vs {want}",
                    got.re
                );
            }
        }
        // Past x ~ 745 the unscaled K_0 is below the smallest f64, so no
        // unscaled routine can represent it at all. The scaled one is
        // still O(1/sqrt(x)), and matches its own leading asymptotic
        // plus the first correction, -1/(8x).
        assert_eq!(spec_math::cephes64::k0(2000.0), 0.0, "unscaled K_0(2000) must underflow");
        let got = bessel_k_scaled_nu(0.0, C::real(2000.0)).unwrap().re;
        let lead = (std::f64::consts::PI / 4000.0).sqrt();
        assert!(
            (got / lead - 1.0 + 1.0 / 16_000.0).abs() < 1e-6,
            "exp(x)K_0(2000)/lead - 1 = {:.3e}, expected -1/(8x) = {:.3e}",
            got / lead - 1.0,
            -1.0 / 16_000.0
        );
        // Cephes kn gives up entirely at order 40; ours does not, and
        // the order recurrence closes on itself exactly.
        assert!(kn(40, 25.0).is_infinite(), "cephes kn(40, .) is expected to overflow");
        let z = C::real(25.0);
        let (a, b, c) = (
            bessel_k_scaled_nu(39.0, z).unwrap(),
            bessel_k_scaled_nu(40.0, z).unwrap(),
            bessel_k_scaled_nu(41.0, z).unwrap(),
        );
        assert!(b.re.is_finite() && b.re > 0.0, "K_40(25) scaled = {b:?}");
        assert!(
            close(c, a + b * (z.inv() * 80.0), 1e-13),
            "K_{{nu+1}} = K_{{nu-1}} + (2nu/z)K_nu should close exactly"
        );
    }

    /// `I` scaled: correct on the real axis, and defined far past the
    /// point where the unscaled value overflows.
    #[test]
    fn scaled_i_is_correct_and_survives_overflow() {
        for &x in &[1.0, 5.0, 15.0, 40.0, 100.0, 400.0, 700.0] {
            for (nu, want) in [(0.0, i0(x) * (-x).exp()), (1.0, i1(x) * (-x).exp())] {
                let got = bessel_i_scaled_nu(nu, C::real(x)).unwrap();
                assert!(
                    (got.re - want).abs() <= 1e-12 * want,
                    "exp(-x) I_{nu}({x}): {} vs {want}",
                    got.re
                );
            }
        }
        // I_0(1000) is about e^1000, far past f64 range.
        assert!(i0(1000.0).is_infinite(), "unscaled I_0(1000) is expected to overflow");
        for &x in &[1000.0_f64, 5000.0] {
            let got = bessel_i_scaled_nu(0.0, C::real(x)).unwrap().re;
            let lead = 1.0 / (2.0 * std::f64::consts::PI * x).sqrt();
            // Leading term plus the first correction +1/(8x).
            assert!(
                (got / lead - 1.0 - 1.0 / (8.0 * x)).abs() < 1e-6,
                "exp(-x)I_0({x})/lead - 1 = {:.3e}, expected 1/(8x) = {:.3e}",
                got / lead - 1.0,
                1.0 / (8.0 * x)
            );
        }
    }

    /// `J` scaled, against Cephes on the real axis where the scaling is
    /// the identity.
    #[test]
    fn scaled_j_matches_the_established_routines() {
        for &x in &[5.0, 20.0, 50.0, 200.0, 1000.0] {
            let got = bessel_j_scaled_nu(0.0, C::real(x)).unwrap();
            assert!(
                (got.re - j0(x)).abs() <= 1e-12 * j0(x).abs().max(0.05),
                "J_0({x}): {} vs {}",
                got.re,
                j0(x)
            );
        }
    }

    /// The systematic cross-check: everywhere the ascending series' own
    /// law says it is good to 1e-13, the scaled routines must agree with
    /// it. This is what caught the two design errors recorded in the
    /// source — the `I` expansion's dropped exponentially small term,
    /// and the order recurrence mixing a recessive Hankel member off the
    /// real axis. Both showed up here as disagreements of 1e0 and worse.
    #[test]
    fn the_scaled_routines_agree_with_the_series_wherever_the_series_is_sound() {
        let mut checked = 0;
        for &nu in &[0.0_f64, 0.5, 2.7, 10.0, 30.0] {
            for &r in &[0.3_f64, 1.0, 3.0, 8.0, 15.0, 25.0] {
                for k in 0..24 {
                    let a = -std::f64::consts::PI + k as f64 * std::f64::consts::PI / 12.0;
                    let z = C::from_polar(r, a);
                    // The negative real axis is the branch cut of Y and
                    // K. The value there is a convention, not a fact,
                    // and the two routes approach it from opposite
                    // sides, so comparing them measures the convention.
                    if a.abs() > std::f64::consts::PI - 1e-9 {
                        continue;
                    }
                    let sound = |loss: f64| series_error(loss) <= 1e-13;
                    let mut cmp = |name: &str, got: Result<C, String>, want: Result<C, String>, sc: C| {
                        if let (Ok(g), Ok(w)) = (got, want) {
                            let w = w * sc;
                            if w.is_finite() && w.abs() > 1e-290 {
                                checked += 1;
                                assert!(
                                    close(g, w, 1e-6),
                                    "{name} at nu={nu} z={z:?}: scaled {g:?} vs series {w:?}"
                                );
                            }
                        }
                    };
                    let e_im = C::real((-z.im.abs()).exp());
                    if sound(z.abs() - z.im.abs()) {
                        cmp("J", bessel_j_scaled_nu(nu, z), bessel_j_nu(nu, z), e_im);
                        cmp("Y", bessel_y_scaled_nu(nu, z), bessel_y_nu(nu, z), e_im);
                    }
                    // K's loss law differs between the whole-order and
                    // fractional-order routes; using the fractional one
                    // for nu = 10 declared the series sound where its
                    // own law predicted 3e-6, and the disagreement was
                    // exactly that size.
                    let k_loss = match whole(nu) {
                        Some(_) => (2.0 * z.re.abs()).max(z.abs()) + z.re,
                        None => z.abs() + z.re,
                    };
                    if sound(k_loss) {
                        cmp("K", bessel_k_scaled_nu(nu, z), bessel_k_nu(nu, z), z.exp());
                    }
                    if sound(z.abs() - z.re) {
                        let e_re = C::real((-z.re.abs()).exp());
                        cmp("I", bessel_i_scaled_nu(nu, z), bessel_i_nu(nu, z), e_re);
                    }
                }
            }
        }
        assert!(checked > 500, "only {checked} comparisons were in range");
    }

    /// The two expansions must agree with each other through
    /// `K_nu(z) = (pi/2) i^(nu+1) H1_nu(iz)` (DLMF 10.27.8). They are
    /// written independently — different `c` in the sum, different
    /// prefactor — so this is a real cross-check and not an identity of
    /// the code. Note `exp(-i(iz)) = exp(z)`, so the scalings line up.
    #[test]
    fn the_k_and_h1_expansions_agree() {
        for &nu in &[0.0, 0.5, 1.0, 2.3] {
            for &x in &[20.0, 50.0, 200.0] {
                let z = C::real(x);
                let k = bessel_k_scaled_nu(nu, z).unwrap();
                let h = hankel_h1_scaled_nu(nu, C::I * z).unwrap();
                let phase = (C::I * (std::f64::consts::FRAC_PI_2 * (nu + 1.0))).exp();
                let want = h * phase * (std::f64::consts::PI * 0.5);
                assert!(close(k, want, 1e-12), "nu={nu} x={x}: {k:?} vs {want:?}");
            }
        }
    }

    /// The I-K Wronskian `I_nu K_{nu+1} + I_{nu+1} K_nu = 1/z`
    /// (DLMF 10.28.2) in scaled variables, where it reads
    /// `Is Ks' + Is' Ks = exp(z - |Re z|)/z`. Elementary on the right,
    /// and it reaches orders and magnitudes at which neither unscaled
    /// factor exists as an `f64`.
    #[test]
    fn the_scaled_i_k_wronskian_holds() {
        for &nu in &[0.0, 0.5, 2.7, 10.0, 30.0] {
            for &r in &[0.5, 3.0, 15.0, 60.0, 300.0, 2000.0] {
                for &a in &[0.0, 0.4, 0.8, 1.2, -0.4, -1.2] {
                    let z = C::from_polar(r, a);
                    let (i0v, i1v) = (
                        bessel_i_scaled_nu(nu, z).unwrap(),
                        bessel_i_scaled_nu(nu + 1.0, z).unwrap(),
                    );
                    let (k0v, k1v) = (
                        bessel_k_scaled_nu(nu, z).unwrap(),
                        bessel_k_scaled_nu(nu + 1.0, z).unwrap(),
                    );
                    let w = i0v * k1v + i1v * k0v;
                    let want = (z - C::real(z.re.abs())).exp() * z.inv();
                    // Scaled by the largest term, so the metric's own
                    // cancellation is divided out rather than measured.
                    let scale = (i0v * k1v).abs() + (i1v * k0v).abs();
                    assert!(
                        (w - want).abs() / scale < 1e-6,
                        "Wronskian nu={nu} z={z:?}: residual {:.2e}",
                        (w - want).abs() / scale
                    );
                }
            }
        }
    }

    /// Scaled times the exponential must reproduce the unscaled value
    /// wherever the unscaled value is itself sound. This is the
    /// compatibility statement a caller needs before switching.
    #[test]
    fn unscaling_reproduces_the_unscaled_routines_where_those_are_sound() {
        use crate::hankel::hankel_h1_nu;
        for &(re, im) in &[(6.0, 0.0), (9.0, -2.0), (7.0, -4.0), (10.0, 1.0)] {
            let z = C::new(re, im);
            let s = hankel_h1_scaled_nu(1.0, z).unwrap();
            let want = hankel_h1_nu(1.0, z).unwrap();
            assert!(close(s * (C::I * z).exp(), want, 1e-9), "H1 at {z:?}");
        }
    }

    /// The stopping rule must actually stop, and its error estimate must
    /// be honest: it has to improve as `|z|` grows, or it is not
    /// measuring anything.
    #[test]
    fn the_truncation_estimate_tracks_reality() {
        let mut prev = 0.0;
        for &x in &[8.0, 12.0, 20.0, 40.0, 100.0] {
            let (_, err) = asym_sum(C::real(0.0), C::real(x), C::ONE);
            assert!(err.is_finite(), "estimate must be a number at x={x}");
            if prev > 0.0 {
                assert!(err < prev, "estimate should improve with |z|: {err:e} at x={x}");
            }
            prev = err;
        }
        // At small |z| the series diverges immediately and the estimate
        // must say so rather than quietly returning the first term.
        let (_, err) = asym_sum(C::real(0.0), C::real(1.0), C::ONE);
        assert!(err > 1e-3, "the estimate at x = 1 should be large, got {err:e}");
        // A half-odd-integer order terminates exactly: nu = 1/2 makes
        // a_1 = (1 - 1)/8 = 0 and every later coefficient with it. This
        // is also why the `I` expansion needs its own dropped-term
        // estimate — here the truncation estimate is 0 and yet the
        // omitted second term of DLMF 10.40.1 is not.
        let (s, err) = asym_sum(C::real(0.5), C::real(3.0), C::ONE);
        assert_eq!(err, 0.0, "nu = 1/2 should terminate exactly");
        assert_eq!(s, C::ONE, "and its sum is 1");
    }

    /// The Airy-type expansion must actually be selected in the band it
    /// was added for. Adding a method the selector never picks is worth
    /// nothing, and that is exactly what happened until the order
    /// recurrence's error estimate was floored by its step count.
    #[test]
    fn the_airy_type_route_is_chosen_across_the_turning_point() {
        for &(nu, frac) in &[
            (40.5_f64, 0.70_f64),
            (100.5, 0.85),
            (200.5, 0.95),
            (400.5, 0.95),
            (1000.5, 0.95),
        ] {
            let x = nu * frac;
            let chosen = bessel_j_scaled_nu(nu, C::real(x)).unwrap().re;
            let airy = crate::airy_uniform::jy_airy(nu, x).0.unwrap().value.re;
            assert!(
                (chosen - airy).abs() <= 1e-14 * airy.abs(),
                "at nu={nu}, x/nu={frac} the selector took {chosen} \
                 rather than the Airy-type value {airy}"
            );
            // ... and it is the right answer, not merely the chosen one.
            let want = spec_math::cephes64::jv(nu, x);
            assert!(
                (chosen - want).abs() <= 1e-11 * want.abs(),
                "at nu={nu}, x/nu={frac}: {chosen} vs cephes {want}"
            );
        }
    }

    /// The refusal must be reachable, and it must distinguish the two
    /// reasons. Since the large-order expansions were added, most points
    /// that used to fail now succeed, and the ones that remain fail
    /// because the ANSWER is outside f64 — a different statement, and
    /// the more useful one.
    #[test]
    fn refusals_say_which_kind_of_failure_it_is() {
        // J_400.5(40) is about e^-805: determined, but not representable.
        let e = bessel_j_scaled_nu(400.5, C::real(40.0)).unwrap_err();
        assert!(e.contains("outside f64 range"), "wrong diagnosis: {e}");
        assert!(e.contains("-805"), "should quote the logarithm: {e}");
        // Y_400.5(40) is about e^+798, the mirror.
        let e = bessel_y_scaled_nu(400.5, C::real(40.0)).unwrap_err();
        assert!(e.contains("outside f64 range"), "wrong diagnosis: {e}");
        // And the genuine no-method message still exists, for I near the
        // imaginary axis at an order no expansion here reaches.
        let e = bessel_i_scaled_nu(4000.0, C::new(1e-6, 300.0)).unwrap_err();
        assert!(e.contains("neither method"), "unhelpful message: {e}");
        assert!(e.contains("10.20"), "should name the missing method: {e}");
    }

    /// The large-order routes must actually be reached and used. These
    /// are the exact points at which the previous stage returned wrong
    /// numbers: `J` below `nu`, built as the difference of two much
    /// larger Hankel values.
    #[test]
    fn the_large_order_route_fixes_the_recessive_j_region() {
        for &(nu, x, want_rel) in &[
            (100.5_f64, 60.3_f64, 1e-12_f64),
            (200.5, 120.3, 1e-12),
            (400.5, 240.3, 1e-8),
            (1000.5, 600.3, 1e-8),
        ] {
            let got = bessel_j_scaled_nu(nu, C::real(x)).unwrap().re;
            let want = spec_math::cephes64::jv(nu, x);
            assert!(
                (got - want).abs() <= want_rel * want.abs(),
                "J_{nu}({x}): {got} vs {want}"
            );
        }
        // The floor on the truncation estimate is what makes this work:
        // at nu = 1/2 the 1/z expansion terminates exactly, so its
        // estimate was 0, and multiplying 0 by a cancellation factor of
        // 1e14 still gave 0. The Hankel route then won every comparison
        // it entered. Removing the floor must break this test, so it is
        // exercised directly rather than asserted about.
        // `scaled_pair`, not `hankel_pair_best`: the latter returns the
        // pair before the exponentials are folded in, and on the real
        // axis those carry the phases that make the sum cancel. Using
        // the wrong one here showed no cancellation at all.
        let (h1, h2, e) = scaled_pair(400.5, C::real(240.3)).expect("pair should exist");
        let v = (h1 + h2) * 0.5;
        assert!(
            e.max(EVAL_FLOOR) * cancellation(h1, h2, v) > SERIES_TOL,
            "the Hankel route at nu = 400.5, z = 240.3 must be rejected"
        );
    }

    #[test]
    fn scaled_edge_cases() {
        for f in [
            hankel_h1_scaled_nu,
            hankel_h2_scaled_nu,
            bessel_k_scaled_nu,
            bessel_i_scaled_nu,
            bessel_j_scaled_nu,
            bessel_y_scaled_nu,
        ] {
            assert!(f(0.0, C::ZERO).is_err(), "z = 0 must be refused");
            assert!(f(f64::NAN, C::ONE).is_err(), "NaN order must be refused");
            assert!(f(0.0, C::new(f64::INFINITY, 0.0)).is_err(), "infinite z");
        }
    }
}
