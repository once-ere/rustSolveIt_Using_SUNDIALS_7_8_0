//! Numerical quadrature and root finding — the toolbox the rest of the
//! crate leans on.
//!
//! Special functions are defined as often by an integral or by a
//! transcendental equation as by a series, so a crate of them needs a
//! reliable integrator and a reliable root finder. This module supplies
//! both, with no external dependencies:
//!
//! * [`gauss_legendre`] — nodes and weights of the `n`-point
//!   Gauss–Legendre rule on `[-1, 1]`, found by Newton iteration on the
//!   Legendre polynomial.
//! * [`integrate`] — that rule affinely mapped onto `[a, b]`. Optimal
//!   for smooth integrands: `n` points integrate every polynomial of
//!   degree `<= 2n - 1` exactly.
//! * [`integrate_adaptive`] — adaptive Simpson with Richardson error
//!   control, for integrands with local structure that a fixed rule
//!   would step straight over.
//! * [`brent_root`] — Brent's method: inverse quadratic interpolation
//!   with a guaranteed bisection fallback, so it is both superlinear
//!   and unconditionally convergent on a bracketing interval.
//! * [`find_roots`] — bracket by scanning, then refine each sign change
//!   with Brent. This is how you hunt eigenvalues of a shooting problem
//!   or the zeros of a Bessel function.
//!
//! # Why Gauss–Legendre, and what "exact" means
//!
//! The `n` nodes are the roots of the Legendre polynomial `P_n`, and
//! choosing them (rather than fixing them in advance, as Newton–Cotes
//! does) buys `n` extra degrees of freedom. The resulting rule is exact
//! for polynomials up to degree `2n - 1` — twice the degree a fixed
//! `n`-point rule reaches. The `gauss_legendre_is_exact_to_degree_2n_1`
//! test checks that defining property directly rather than merely
//! checking a few integrals, because it is the property that fails
//! first if the nodes or weights drift.
//!
//! # Error policy
//!
//! Every public function returns `Result<_, String>` with a message
//! naming the offending argument. Nothing panics, and nothing returns a
//! silent `NaN`: a non-finite bound, an empty rule, a degenerate
//! interval, an interval that does not bracket a root, or an iteration
//! that fails to converge is reported as an error.
//!
//! # References
//!
//! * Abramowitz & Stegun (1964), §25.4 (Gauss quadrature) and §22
//!   (Legendre polynomials) — a public-domain US Government work.
//! * `DLMF 3.5(v)` <https://dlmf.nist.gov/3.5.v> for Gauss quadrature;
//!   `DLMF 18.3` <https://dlmf.nist.gov/18.3> for the Legendre roots.
//! * R. P. Brent, *Algorithms for Minimization without Derivatives*
//!   (Prentice-Hall, 1973), chapter 4.

use std::f64::consts::PI;

/// Newton iterations allowed per Legendre root before giving up.
/// Newton converges quadratically from the standard asymptotic guess,
/// so five or six is typical; a hundred means something is wrong.
const NEWTON_MAX_ITER: usize = 100;

/// Newton is considered converged once the step falls below this.
/// Nodes live in `[-1, 1]`, so an absolute test is the right one.
const NEWTON_TOL: f64 = 1.0e-15;

/// Brent iterations allowed before giving up. Brent is guaranteed to
/// converge in at most `O(log2((b-a)/tol)^2)` steps, so for any sane
/// tolerance this cap is never reached.
const BRENT_MAX_ITER: usize = 200;

/// Bisection levels allowed in [`integrate_adaptive`]. Each level halves
/// the interval, so 50 levels resolve a feature `2^-50` of the way
/// across it; anything finer is below `f64` resolution of the abscissa.
const ADAPTIVE_MAX_DEPTH: usize = 50;

// ---------------------------------------------------------------------
// Legendre polynomial and its derivative
// ---------------------------------------------------------------------

/// Evaluate `P_n(x)` and `P_n'(x)` together for `n >= 1`.
///
/// `P_n` comes from the three-term recurrence (A&S 22.7.10)
///
/// ```text
///     j P_j(x) = (2j - 1) x P_{j-1}(x) - (j - 1) P_{j-2}(x)
/// ```
///
/// and the derivative from the companion identity (A&S 22.8.5)
///
/// ```text
///     (x^2 - 1) P_n'(x) = n [ x P_n(x) - P_{n-1}(x) ]
/// ```
///
/// which is why the recurrence is run keeping `P_{n-1}` alongside
/// `P_n`. At `x = ±1` that identity is 0/0, so the closed forms
/// `P_n'(1) = n(n+1)/2` and `P_n'(-1) = (-1)^(n+1) n(n+1)/2` are used
/// instead — the Gauss nodes are strictly interior, but the guard keeps
/// the helper from ever manufacturing a `NaN`.
fn legendre_p_dp(n: usize, x: f64) -> (f64, f64) {
    debug_assert!(n >= 1);
    let mut p_prev = 1.0_f64; // P_0
    let mut p = x; // P_1
    for j in 2..=n {
        let jf = j as f64;
        let p_next = ((2.0 * jf - 1.0) * x * p - (jf - 1.0) * p_prev) / jf;
        p_prev = p;
        p = p_next;
    }
    let denom = x * x - 1.0;
    let nf = n as f64;
    let dp = if denom != 0.0 {
        nf * (x * p - p_prev) / denom
    } else {
        let h = 0.5 * nf * (nf + 1.0);
        if x > 0.0 || n % 2 == 1 {
            h
        } else {
            -h
        }
    };
    (p, dp)
}

/// Newton-polish one root of `P_n` starting from `guess`.
///
/// Returns the root together with `P_n'` evaluated there, since the
/// Gauss weight needs exactly that derivative and recomputing it at the
/// converged point is both cheaper and more accurate than reusing the
/// value from the last-but-one iterate.
///
/// `max_iter` is a parameter rather than a constant so the
/// non-convergence path is reachable from the tests.
fn legendre_root_newton(n: usize, guess: f64, max_iter: usize) -> Result<(f64, f64), String> {
    let mut z = guess;
    for _ in 0..max_iter {
        let (p, dp) = legendre_p_dp(n, z);
        if dp == 0.0 || !dp.is_finite() {
            return Err(format!(
                "gauss_legendre: Newton iteration for a root of P_{n} hit a zero or non-finite \
                 derivative at x = {z}; the rule cannot be built"
            ));
        }
        let dz = p / dp;
        z -= dz;
        if !z.is_finite() {
            return Err(format!(
                "gauss_legendre: Newton iteration for a root of P_{n} left the real line \
                 (iterate {z}); the rule cannot be built"
            ));
        }
        if dz.abs() <= NEWTON_TOL {
            let (_, dp) = legendre_p_dp(n, z);
            return Ok((z, dp));
        }
    }
    Err(format!(
        "gauss_legendre: Newton iteration for a root of P_{n} failed to converge within \
         {max_iter} iterations (last iterate {z}); this should not happen for any n, so please \
         report it as a bug"
    ))
}

// ---------------------------------------------------------------------
// Gauss-Legendre nodes and weights
// ---------------------------------------------------------------------

/// Nodes and weights of the `n`-point Gauss–Legendre rule on `[-1, 1]`.
///
/// Returns `(nodes, weights)`, both of length `n`, with the nodes in
/// **ascending** order.
///
/// # Algorithm
///
/// The nodes are the roots of the Legendre polynomial `P_n`
/// (`DLMF 3.5.19` <https://dlmf.nist.gov/3.5.E19>). The `i`-th root is
/// approximated by the classical asymptotic estimate
///
/// ```text
///     x_i ~ cos( pi (i - 1/4) / (n + 1/2) )
/// ```
///
/// and then polished by Newton's method on `P_n`, using the derivative
/// from the recurrence identity — quadratic convergence, so five or six
/// iterations suffice. The weight follows in closed form
/// (A&S 25.4.29):
///
/// ```text
///     w_i = 2 / [ (1 - x_i^2) P_n'(x_i)^2 ]
/// ```
///
/// Only the `ceil(n/2)` non-negative roots are computed; the rest come
/// from the symmetry `P_n(-x) = (-1)^n P_n(x)`, which makes the node set
/// antisymmetric and the weight set symmetric — so the symmetry is
/// exact by construction, not merely to rounding.
///
/// # Errors
///
/// * `n == 0` — a zero-point rule integrates nothing.
/// * Newton fails to converge, or meets a zero derivative. Neither
///   should happen for any `n`; both are reported rather than papered
///   over.
///
/// # Examples
///
/// ```
/// use special_functions::quadrature::gauss_legendre;
///
/// let (x, w) = gauss_legendre(5).unwrap();
/// // The weights of any Gauss-Legendre rule on [-1, 1] sum to 2.
/// let sum: f64 = w.iter().sum();
/// assert!((sum - 2.0).abs() < 1e-14);
/// // Nodes are antisymmetric about the origin and strictly interior.
/// assert!((x[0] + x[4]).abs() < 1e-15);
/// assert!(x.iter().all(|&xi| xi.abs() < 1.0));
/// // n = 5 integrates x^8 exactly: 2/9.
/// let q: f64 = x.iter().zip(&w).map(|(&xi, &wi)| wi * xi.powi(8)).sum();
/// assert!((q - 2.0 / 9.0).abs() < 1e-14);
///
/// assert!(gauss_legendre(0).is_err());
/// ```
pub fn gauss_legendre(n: usize) -> Result<(Vec<f64>, Vec<f64>), String> {
    if n == 0 {
        return Err(
            "gauss_legendre: n must be >= 1, got 0 (a zero-point rule integrates nothing)"
                .to_string(),
        );
    }

    let mut nodes = vec![0.0_f64; n];
    let mut weights = vec![0.0_f64; n];
    let half = n.div_ceil(2);

    for i in 0..half {
        // Asymptotic guess for the i-th root counted from x = +1
        // (A&S 25.4.30 region); i is 0-based here, so i + 1 - 1/4.
        let guess = (PI * (i as f64 + 0.75) / (n as f64 + 0.5)).cos();
        let (z, dp) = legendre_root_newton(n, guess, NEWTON_MAX_ITER)?;

        let denom = (1.0 - z * z) * dp * dp;
        if !denom.is_finite() || denom <= 0.0 {
            return Err(format!(
                "gauss_legendre: degenerate weight for root {i} of P_{n} at x = {z} \
                 (denominator {denom}); the rule cannot be built"
            ));
        }
        let wi = 2.0 / denom;

        // Assign the negative image first: for odd n the middle index is
        // its own mirror, and the second write leaves a clean +0.0.
        nodes[i] = -z;
        weights[i] = wi;
        nodes[n - 1 - i] = z;
        weights[n - 1 - i] = wi;
    }

    Ok((nodes, weights))
}

// ---------------------------------------------------------------------
// Fixed-order Gauss-Legendre integration
// ---------------------------------------------------------------------

/// Integrate `f` over `[a, b]` with the `n`-point Gauss–Legendre rule.
///
/// # Algorithm
///
/// The rule from [`gauss_legendre`] lives on `[-1, 1]`, so it is carried
/// onto `[a, b]` by the affine map
///
/// ```text
///     x(t) = (b - a) t / 2 + (a + b) / 2,   dx = (b - a) dt / 2
/// ```
///
/// giving `integral ~ ((b-a)/2) * sum_i w_i f(x(t_i))`. An affine map
/// preserves polynomial degree, so the degree-`2n-1` exactness carries
/// over unchanged: on `[a, b]` the rule is still exact for every
/// polynomial of degree `<= 2n - 1`.
///
/// This is the right tool for a smooth integrand. For one with a spike,
/// a kink or an endpoint singularity, reach for [`integrate_adaptive`]
/// — a fixed rule can step straight over a narrow feature and return a
/// confident wrong answer.
///
/// # Errors
///
/// * `n == 0`.
/// * `a` or `b` not finite.
/// * `b <= a` (a degenerate or reversed interval).
/// * `f` returns a non-finite value at any node — reported with the
///   offending abscissa rather than propagated as a silent `NaN`.
/// * Any error from [`gauss_legendre`].
///
/// # Examples
///
/// ```
/// use special_functions::quadrature::integrate;
///
/// // Integral of sin x over [0, pi] is exactly 2.
/// let q = integrate(f64::sin, 0.0, std::f64::consts::PI, 20).unwrap();
/// assert!((q - 2.0).abs() < 1e-14);
///
/// // Integral of 1/(1 + x^2) over [0, 1] is pi/4.
/// let q = integrate(|x| 1.0 / (1.0 + x * x), 0.0, 1.0, 12).unwrap();
/// assert!((q - std::f64::consts::FRAC_PI_4).abs() < 1e-14);
///
/// assert!(integrate(f64::sin, 1.0, 0.0, 8).is_err()); // b <= a
/// assert!(integrate(f64::sin, 0.0, 1.0, 0).is_err()); // empty rule
/// ```
pub fn integrate<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, n: usize) -> Result<f64, String> {
    check_interval("integrate", a, b)?;
    if n == 0 {
        return Err(
            "integrate: n must be >= 1, got 0 (a zero-point rule integrates nothing)".to_string(),
        );
    }

    let (nodes, weights) = gauss_legendre(n)?;
    let half_width = 0.5 * (b - a);
    let midpoint = 0.5 * (a + b);

    let mut acc = 0.0_f64;
    for (&t, &w) in nodes.iter().zip(weights.iter()) {
        let x = half_width * t + midpoint;
        let fx = f(x);
        if !fx.is_finite() {
            return Err(format!(
                "integrate: the integrand is not finite at the quadrature node x = {x} \
                 (value {fx}); integrate over a subinterval that avoids the singularity, or \
                 remove it analytically"
            ));
        }
        acc += w * fx;
    }

    let result = half_width * acc;
    if !result.is_finite() {
        return Err(format!(
            "integrate: the quadrature sum overflowed to {result} on [{a}, {b}] with n = {n}"
        ));
    }
    Ok(result)
}

// ---------------------------------------------------------------------
// Adaptive Simpson
// ---------------------------------------------------------------------

/// Simpson's rule on `[a, b]` from the three ordinates already in hand.
#[inline]
fn simpson(a: f64, b: f64, fa: f64, fm: f64, fb: f64) -> f64 {
    (b - a) / 6.0 * (fa + 4.0 * fm + fb)
}

/// One level of adaptive bisection.
///
/// `whole` is Simpson's estimate over `[a, b]` from the caller; this
/// recomputes it as the sum of the two half-interval estimates and uses
/// the difference as the Richardson error estimate. Simpson's rule has
/// a fourth-order error term, so halving the interval divides the error
/// by 16 and the difference between the two estimates is 15 times the
/// error of the finer one — hence the `15 * tol` acceptance test and the
/// `delta / 15` correction that upgrades the answer to Boole-like
/// accuracy at no extra function evaluations.
///
/// The long argument list is deliberate: `fa`, `fm`, `fb` and `whole`
/// are handed down so each recursion level reuses the ordinates its
/// parent already paid for. Bundling them into a struct would hide the
/// one property that matters here — every level costs exactly two new
/// evaluations of `f`.
#[allow(clippy::too_many_arguments)]
fn adaptive_step<F: Fn(f64) -> f64>(
    f: &F,
    a: f64,
    b: f64,
    fa: f64,
    fm: f64,
    fb: f64,
    whole: f64,
    tol: f64,
    depth: usize,
) -> Result<f64, String> {
    let m = 0.5 * (a + b);
    let lm = 0.5 * (a + m);
    let rm = 0.5 * (m + b);
    let flm = f(lm);
    let frm = f(rm);
    for (x, v) in [(lm, flm), (rm, frm)] {
        if !v.is_finite() {
            return Err(format!(
                "integrate_adaptive: the integrand is not finite at x = {x} (value {v}); \
                 integrate over a subinterval that avoids the singularity, or remove it \
                 analytically"
            ));
        }
    }

    let left = simpson(a, m, fa, flm, fm);
    let right = simpson(m, b, fm, frm, fb);
    let delta = left + right - whole;

    if delta.abs() <= 15.0 * tol {
        return Ok(left + right + delta / 15.0);
    }
    if depth == 0 {
        return Err(format!(
            "integrate_adaptive: could not reach the requested tolerance on [{a}, {b}] within \
             {ADAPTIVE_MAX_DEPTH} bisections (local error estimate {}); the integrand is \
             probably singular or discontinuous there — split the interval at the feature, or \
             loosen tol",
            (delta / 15.0).abs()
        ));
    }

    let l = adaptive_step(f, a, m, fa, flm, fm, left, 0.5 * tol, depth - 1)?;
    let r = adaptive_step(f, m, b, fm, frm, fb, right, 0.5 * tol, depth - 1)?;
    Ok(l + r)
}

/// Integrate `f` over `[a, b]` by adaptive Simpson to an absolute
/// tolerance.
///
/// # Algorithm
///
/// Simpson's rule is applied to the whole interval and to its two
/// halves. The difference between the two estimates is a Richardson
/// estimate of the error (Simpson's error is `O(h^4)`, so bisection
/// shrinks it by 16 and the discrepancy is 15 times the finer error).
/// If that is within tolerance the halves are accepted, with the
/// `delta / 15` extrapolation added; otherwise each half is recursed on
/// with half the tolerance, so the accepted local errors sum to no more
/// than `tol`. Recursion is capped at 50 levels.
///
/// Effort goes where the integrand needs it: a spike gets bisected a
/// dozen times while the flat tails are accepted immediately. That is
/// what makes this the right tool where [`integrate`] is not — but for
/// a genuinely smooth integrand Gauss–Legendre wins on both accuracy
/// and function evaluations.
///
/// # Errors
///
/// * `a` or `b` not finite, or `b <= a`.
/// * `tol` not finite, or `tol <= 0`.
/// * `f` returns a non-finite value anywhere it is sampled.
/// * The tolerance is not reached within 50 bisections — the honest
///   report for a singular or discontinuous integrand, rather than a
///   plausible-looking wrong number.
///
/// # Examples
///
/// ```
/// use special_functions::quadrature::integrate_adaptive;
///
/// // Integral of exp(x) over [0, 1] is e - 1.
/// let q = integrate_adaptive(f64::exp, 0.0, 1.0, 1e-12).unwrap();
/// assert!((q - (std::f64::consts::E - 1.0)).abs() < 1e-10);
///
/// // A peak a fixed rule could miss: 1/(1 + 100 x^2) over [-1, 1],
/// // whose exact value is atan(10)/5.
/// let q = integrate_adaptive(|x| 1.0 / (1.0 + 100.0 * x * x), -1.0, 1.0, 1e-12).unwrap();
/// assert!((q - 10.0_f64.atan() / 5.0).abs() < 1e-10);
///
/// assert!(integrate_adaptive(f64::exp, 0.0, 1.0, 0.0).is_err()); // tol <= 0
/// ```
pub fn integrate_adaptive<F: Fn(f64) -> f64>(
    f: F,
    a: f64,
    b: f64,
    tol: f64,
) -> Result<f64, String> {
    check_interval("integrate_adaptive", a, b)?;
    check_tol("integrate_adaptive", tol)?;

    let m = 0.5 * (a + b);
    let fa = f(a);
    let fm = f(m);
    let fb = f(b);
    for (x, v) in [(a, fa), (m, fm), (b, fb)] {
        if !v.is_finite() {
            return Err(format!(
                "integrate_adaptive: the integrand is not finite at x = {x} (value {v}); \
                 integrate over a subinterval that avoids the singularity, or remove it \
                 analytically"
            ));
        }
    }

    let whole = simpson(a, b, fa, fm, fb);
    let result = adaptive_step(&f, a, b, fa, fm, fb, whole, tol, ADAPTIVE_MAX_DEPTH)?;
    if !result.is_finite() {
        return Err(format!(
            "integrate_adaptive: the accumulated quadrature overflowed to {result} on [{a}, {b}]"
        ));
    }
    Ok(result)
}

// ---------------------------------------------------------------------
// Brent's method
// ---------------------------------------------------------------------

/// Brent's method with an explicit iteration cap.
///
/// Split out from [`brent_root`] so that the non-convergence branch is
/// reachable from the tests; the public entry point supplies
/// [`BRENT_MAX_ITER`].
fn brent_root_capped<F: Fn(f64) -> f64>(
    f: F,
    a: f64,
    b: f64,
    tol: f64,
    max_iter: usize,
) -> Result<f64, String> {
    check_interval("brent_root", a, b)?;
    check_tol("brent_root", tol)?;

    let (mut a, mut b) = (a, b);
    let (mut fa, mut fb) = (f(a), f(b));
    for (x, v) in [(a, fa), (b, fb)] {
        if !v.is_finite() {
            return Err(format!(
                "brent_root: f is not finite at the bracket endpoint x = {x} (value {v})"
            ));
        }
    }

    if fa == 0.0 {
        return Ok(a);
    }
    if fb == 0.0 {
        return Ok(b);
    }
    if (fa > 0.0) == (fb > 0.0) {
        return Err(format!(
            "brent_root: [{a}, {b}] does not bracket a root — f({a}) = {fa} and f({b}) = {fb} \
             have the same sign. Brent's method needs a sign change; widen the interval or scan \
             it with find_roots first"
        ));
    }

    // Keep b as the better of the two estimates.
    if fa.abs() < fb.abs() {
        std::mem::swap(&mut a, &mut b);
        std::mem::swap(&mut fa, &mut fb);
    }

    let mut c = a; // previous iterate
    let mut fc = fa;
    let mut d = a; // iterate before that; only read once mflag is false
    let mut mflag = true;

    for _ in 0..max_iter {
        if fb == 0.0 || (b - a).abs() < tol {
            return Ok(b);
        }

        // Inverse quadratic interpolation when the three ordinates are
        // distinct, secant otherwise.
        let s = if fa != fc && fb != fc {
            a * fb * fc / ((fa - fb) * (fa - fc))
                + b * fa * fc / ((fb - fa) * (fb - fc))
                + c * fa * fb / ((fc - fa) * (fc - fb))
        } else {
            b - fb * (b - a) / (fb - fa)
        };

        // Brent's acceptance conditions: reject the interpolant unless
        // it lands in the middle three quarters of [b, (3a+b)/4] and is
        // shrinking the step fast enough. Otherwise bisect, which caps
        // the worst case at bisection's guaranteed convergence.
        let quarter = (3.0 * a + b) / 4.0;
        let (lo, hi) = if quarter < b {
            (quarter, b)
        } else {
            (b, quarter)
        };
        let reject = !(s.is_finite() && s > lo && s < hi)
            || (mflag && (s - b).abs() >= 0.5 * (b - c).abs())
            || (!mflag && (s - b).abs() >= 0.5 * (c - d).abs())
            || (mflag && (b - c).abs() < tol)
            || (!mflag && (c - d).abs() < tol);

        let s = if reject {
            mflag = true;
            0.5 * (a + b)
        } else {
            mflag = false;
            s
        };

        let fs = f(s);
        if !fs.is_finite() {
            return Err(format!(
                "brent_root: f is not finite at the trial point x = {s} (value {fs}); the \
                 bracket probably straddles a pole rather than a root"
            ));
        }
        if fs == 0.0 {
            return Ok(s);
        }

        d = c;
        c = b;
        fc = fb;

        // Keep the sign change: replace whichever endpoint shares s's sign.
        if (fa > 0.0) != (fs > 0.0) {
            b = s;
            fb = fs;
        } else {
            a = s;
            fa = fs;
        }
        if fa.abs() < fb.abs() {
            std::mem::swap(&mut a, &mut b);
            std::mem::swap(&mut fa, &mut fb);
        }
    }

    Err(format!(
        "brent_root: failed to converge to tol = {tol} within {max_iter} iterations; the \
         bracket is down to [{a}, {b}] with f = [{fa}, {fb}] — loosen tol, or check that the \
         sign change is a root and not a pole"
    ))
}

/// Find a root of `f` in the bracketing interval `[a, b]` by Brent's
/// method.
///
/// # Algorithm
///
/// Brent (1973) combines three ideas. Inverse quadratic interpolation
/// through the last three iterates converges superlinearly near a
/// simple root; the secant step covers the case where two ordinates
/// coincide; and a set of acceptance tests rejects any interpolated
/// step that lands outside the bracket or fails to shrink the step by
/// at least half, falling back to bisection. The bracket `[a, b]` is
/// maintained with `f(a)` and `f(b)` of opposite sign throughout, so
/// convergence is guaranteed at no worse than bisection's rate while
/// being superlinear in practice — typically a handful of iterations to
/// machine precision.
///
/// The returned value is the endpoint of a bracket narrower than `tol`,
/// so `tol` is an absolute bound on the abscissa error.
///
/// # Errors
///
/// * `a` or `b` not finite, or `b <= a`.
/// * `tol` not finite, or `tol <= 0`.
/// * **`f(a)` and `f(b)` do not straddle zero** — the interval does not
///   bracket a root and Brent's guarantee does not apply. Use
///   [`find_roots`] to locate a bracket first.
/// * `f` returns a non-finite value at any point sampled.
/// * No convergence within 200 iterations.
///
/// # Examples
///
/// ```
/// use special_functions::quadrature::brent_root;
///
/// // The Dottie number: the unique fixed point of cosine.
/// let r = brent_root(|x| x.cos() - x, 0.0, 2.0, 1e-14).unwrap();
/// assert!((r - 0.739085133215160_f64).abs() < 1e-12);
///
/// // Wallis's cubic, x^3 - 2x - 5 = 0.
/// let r = brent_root(|x| x * x * x - 2.0 * x - 5.0, 2.0, 3.0, 1e-14).unwrap();
/// assert!((r - 2.094551481542326_f64).abs() < 1e-12);
///
/// // Same sign at both ends: no bracket, so this is an error.
/// assert!(brent_root(|x| x.cos() - x, 1.0, 2.0, 1e-12).is_err());
/// ```
pub fn brent_root<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, tol: f64) -> Result<f64, String> {
    brent_root_capped(f, a, b, tol, BRENT_MAX_ITER)
}

// ---------------------------------------------------------------------
// Bracket and refine
// ---------------------------------------------------------------------

/// Find every root of `f` in `[a, b]` that a scan of `steps`
/// subintervals can see, each refined by Brent's method.
///
/// # Algorithm
///
/// Brent needs a bracket, and finding brackets is the part no root
/// finder can do for you. This walks `a = x_0 < x_1 < ... < x_steps = b`
/// evaluating `f` once per node, records any node where `f` is exactly
/// zero, and hands every sign change `f(x_k) f(x_{k+1}) < 0` to
/// [`brent_root`]. Roots are returned in ascending order.
///
/// This is the standard way to hunt the eigenvalues of a shooting
/// problem (where the roots are the zeros of a mismatch function) and
/// to locate the zeros of a special function without an asymptotic
/// formula for them.
///
/// # The resolution caveat
///
/// A scan sees only what it samples. Two roots inside one subinterval
/// produce no sign change and are both missed, as is any root of even
/// multiplicity (where `f` touches zero without crossing). Choose
/// `steps` finer than the closest spacing you expect. Conversely a
/// *pole* also flips the sign of `f`, and will be reported as a root;
/// if `f` can blow up in `[a, b]`, check `f` at each returned value.
///
/// # Errors
///
/// * `a` or `b` not finite, or `b <= a`.
/// * `steps == 0`.
/// * `tol` not finite, or `tol <= 0`.
/// * `f` returns a non-finite value at any scan node.
/// * Any error from [`brent_root`] on a refined bracket.
///
/// # Examples
///
/// ```
/// use special_functions::quadrature::find_roots;
/// use std::f64::consts::PI;
///
/// // The zeros of sin x in [0, 10] are 0, pi, 2pi and 3pi.
/// let roots = find_roots(f64::sin, 0.0, 10.0, 200, 1e-12).unwrap();
/// assert_eq!(roots.len(), 4);
/// for (k, &r) in roots.iter().enumerate() {
///     assert!((r - k as f64 * PI).abs() < 1e-10);
/// }
///
/// assert!(find_roots(f64::sin, 0.0, 10.0, 0, 1e-12).is_err());
/// ```
pub fn find_roots<F: Fn(f64) -> f64 + Copy>(
    f: F,
    a: f64,
    b: f64,
    steps: usize,
    tol: f64,
) -> Result<Vec<f64>, String> {
    check_interval("find_roots", a, b)?;
    check_tol("find_roots", tol)?;
    if steps == 0 {
        return Err(
            "find_roots: steps must be >= 1, got 0 (a scan with no subintervals sees nothing)"
                .to_string(),
        );
    }

    let h = (b - a) / steps as f64;
    let mut roots: Vec<f64> = Vec::new();

    let node = |k: usize| -> f64 {
        if k == steps {
            b // exact right endpoint, never a + steps*h rounded
        } else {
            a + k as f64 * h
        }
    };
    let sample = |x: f64| -> Result<f64, String> {
        let v = f(x);
        if v.is_finite() {
            Ok(v)
        } else {
            Err(format!(
                "find_roots: f is not finite at the scan node x = {x} (value {v}); restrict the \
                 scan to an interval where f is finite"
            ))
        }
    };

    let push = |r: f64, roots: &mut Vec<f64>| {
        // Guard against a root being reported twice from two adjacent
        // brackets that share an endpoint.
        let separation = tol.max(f64::EPSILON * (1.0 + r.abs()));
        if roots
            .last()
            .is_none_or(|&last| (r - last).abs() > separation)
        {
            roots.push(r);
        }
    };

    let mut x0 = node(0);
    let mut f0 = sample(x0)?;
    if f0 == 0.0 {
        push(x0, &mut roots);
    }

    for k in 1..=steps {
        let x1 = node(k);
        let f1 = sample(x1)?;

        if f1 == 0.0 {
            push(x1, &mut roots);
        } else if (f0 < 0.0) != (f1 < 0.0) && f0 != 0.0 {
            let r = brent_root(f, x0, x1, tol)?;
            push(r, &mut roots);
        }

        x0 = x1;
        f0 = f1;
    }

    Ok(roots)
}

// ---------------------------------------------------------------------
// Shared argument checks
// ---------------------------------------------------------------------

/// Validate an integration or bracketing interval.
fn check_interval(who: &str, a: f64, b: f64) -> Result<(), String> {
    if !a.is_finite() {
        return Err(format!("{who}: lower bound a must be finite, got {a}"));
    }
    if !b.is_finite() {
        return Err(format!("{who}: upper bound b must be finite, got {b}"));
    }
    if b <= a {
        return Err(format!(
            "{who}: need b > a, got a = {a} and b = {b} (an empty or reversed interval)"
        ));
    }
    Ok(())
}

/// Validate a tolerance.
fn check_tol(who: &str, tol: f64) -> Result<(), String> {
    if !tol.is_finite() || tol <= 0.0 {
        return Err(format!("{who}: tol must be finite and > 0, got {tol}"));
    }
    Ok(())
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Exact value of the integral of x^k over [-1, 1].
    fn monomial_exact(k: u32) -> f64 {
        if k % 2 == 1 {
            0.0
        } else {
            2.0 / (k as f64 + 1.0)
        }
    }

    // ---- 1. the defining property --------------------------------

    /// An n-point Gauss-Legendre rule integrates every polynomial of
    /// degree <= 2n-1 exactly. This is *the* property; everything else
    /// about the rule follows from it.
    #[test]
    fn gauss_legendre_is_exact_to_degree_2n_minus_1() {
        for n in 1..=24_usize {
            let (x, w) = gauss_legendre(n).unwrap();
            for k in 0..=(2 * n as u32 - 1) {
                let q: f64 = x
                    .iter()
                    .zip(w.iter())
                    .map(|(&xi, &wi)| wi * xi.powi(k as i32))
                    .sum();
                let exact = monomial_exact(k);
                assert!(
                    (q - exact).abs() < 1e-14,
                    "n = {n}, degree {k}: got {q}, want {exact} (err {:e})",
                    (q - exact).abs()
                );
            }
        }
    }

    /// And it is *not* exact one degree further — degree 2n is where the
    /// rule genuinely runs out, which is what makes the bound sharp
    /// rather than merely sufficient.
    #[test]
    fn gauss_legendre_is_not_exact_at_degree_2n() {
        for n in 1..=8_usize {
            let (x, w) = gauss_legendre(n).unwrap();
            let k = 2 * n as u32;
            let q: f64 = x
                .iter()
                .zip(w.iter())
                .map(|(&xi, &wi)| wi * xi.powi(k as i32))
                .sum();
            let exact = monomial_exact(k);
            assert!(
                (q - exact).abs() > 1e-6,
                "n = {n}: rule was unexpectedly exact at degree {k}"
            );
        }
    }

    /// The same exactness after the affine map onto a general interval.
    #[test]
    fn integrate_is_exact_for_polynomials_on_a_general_interval() {
        let (a, b) = (-0.7_f64, 2.3_f64);
        for n in 1..=12_usize {
            for k in 0..=(2 * n as u32 - 1) {
                let q = integrate(|x: f64| x.powi(k as i32), a, b, n).unwrap();
                let e = k + 1;
                let exact = (b.powi(e as i32) - a.powi(e as i32)) / e as f64;
                assert!(
                    (q - exact).abs() < 1e-13 * (1.0 + exact.abs()),
                    "n = {n}, degree {k}: got {q}, want {exact}"
                );
            }
        }
    }

    // ---- 2. known integrals --------------------------------------

    #[test]
    fn known_integrals_gauss_legendre() {
        let q = integrate(f64::sin, 0.0, PI, 20).unwrap();
        assert!((q - 2.0).abs() < 1e-14, "sin over [0,pi]: {q}");

        let q = integrate(f64::exp, 0.0, 1.0, 20).unwrap();
        let exact = std::f64::consts::E - 1.0;
        assert!((q - exact).abs() < 1e-15, "exp over [0,1]: {q}");

        let q = integrate(|x: f64| 1.0 / (1.0 + x * x), 0.0, 1.0, 20).unwrap();
        assert!(
            (q - std::f64::consts::FRAC_PI_4).abs() < 1e-15,
            "1/(1+x^2) over [0,1]: {q}"
        );

        // ln 2 = integral of 1/x over [1, 2].
        let q = integrate(|x: f64| 1.0 / x, 1.0, 2.0, 30).unwrap();
        assert!(
            (q - std::f64::consts::LN_2).abs() < 1e-15,
            "1/x over [1,2]: {q}"
        );
    }

    /// Convergence is spectral for a smooth integrand: doubling the
    /// order should cross machine precision, not creep toward it.
    #[test]
    fn gauss_legendre_converges_spectrally() {
        let err = |n: usize| (integrate(f64::sin, 0.0, PI, n).unwrap() - 2.0).abs();
        assert!(err(4) < 1e-4, "n=4 err {}", err(4));
        assert!(err(8) < 1e-11, "n=8 err {}", err(8));
        assert!(err(16) < 1e-15, "n=16 err {}", err(16));
    }

    // ---- 3. node and weight sanity -------------------------------

    #[test]
    fn nodes_and_weights_are_sane() {
        for n in 1..=32_usize {
            let (x, w) = gauss_legendre(n).unwrap();
            assert_eq!(x.len(), n);
            assert_eq!(w.len(), n);

            let sum: f64 = w.iter().sum();
            assert!((sum - 2.0).abs() < 1e-14, "n = {n}: weights sum to {sum}");

            for (i, &wi) in w.iter().enumerate() {
                assert!(wi > 0.0, "n = {n}: weight {i} is {wi}, not positive");
            }
            for (i, &xi) in x.iter().enumerate() {
                assert!(
                    xi > -1.0 && xi < 1.0,
                    "n = {n}: node {i} is {xi}, not strictly inside (-1, 1)"
                );
            }
            for i in 0..n - 1 {
                assert!(
                    x[i] < x[i + 1],
                    "n = {n}: nodes are not strictly ascending at {i}"
                );
            }
            for i in 0..n {
                assert!(
                    (x[i] + x[n - 1 - i]).abs() < 1e-15,
                    "n = {n}: nodes {i} and {} are not symmetric: {} vs {}",
                    n - 1 - i,
                    x[i],
                    x[n - 1 - i]
                );
                assert!(
                    (w[i] - w[n - 1 - i]).abs() < 1e-15,
                    "n = {n}: weights {i} and {} differ",
                    n - 1 - i
                );
            }
            // Odd rules have a node exactly at the origin.
            if n % 2 == 1 {
                assert!(
                    x[n / 2].abs() < 1e-16,
                    "n = {n}: centre node is {}",
                    x[n / 2]
                );
            }
        }
    }

    /// The mapped weights sum to the width of the interval, which is
    /// just the degree-0 exactness stated in the units a caller cares
    /// about.
    #[test]
    fn mapped_weights_sum_to_the_interval_width() {
        for (a, b) in [(0.0, 1.0), (-3.0, 4.5), (2.0, 2.0 + PI)] {
            for n in [1_usize, 2, 5, 17] {
                let q = integrate(|_| 1.0, a, b, n).unwrap();
                assert!(
                    (q - (b - a)).abs() < 1e-14 * (1.0 + (b - a)),
                    "[{a},{b}] n={n}: got {q}, want {}",
                    b - a
                );
            }
        }
    }

    /// Small rules against their textbook closed forms.
    #[test]
    fn small_rules_match_closed_forms() {
        let (x, w) = gauss_legendre(1).unwrap();
        assert!(x[0].abs() < 1e-16);
        assert!((w[0] - 2.0).abs() < 1e-15);

        let (x, w) = gauss_legendre(2).unwrap();
        let r = 1.0 / 3.0_f64.sqrt();
        assert!((x[0] + r).abs() < 1e-15 && (x[1] - r).abs() < 1e-15);
        assert!((w[0] - 1.0).abs() < 1e-15 && (w[1] - 1.0).abs() < 1e-15);

        let (x, w) = gauss_legendre(3).unwrap();
        let r = (3.0_f64 / 5.0).sqrt();
        assert!((x[0] + r).abs() < 1e-15 && x[1].abs() < 1e-16 && (x[2] - r).abs() < 1e-15);
        assert!((w[0] - 5.0 / 9.0).abs() < 1e-15);
        assert!((w[1] - 8.0 / 9.0).abs() < 1e-15);
        assert!((w[2] - 5.0 / 9.0).abs() < 1e-15);
    }

    // ---- 4. adaptive Simpson --------------------------------------

    #[test]
    fn adaptive_matches_gauss_legendre_on_smooth_integrands() {
        // name, integrand, lower bound, upper bound
        type Case = (&'static str, fn(f64) -> f64, f64, f64);
        let cases: [Case; 3] = [
            ("sin", f64::sin, 0.0, PI),
            ("exp", f64::exp, 0.0, 1.0),
            ("cos", f64::cos, -1.0, 2.0),
        ];
        for (name, f, a, b) in cases {
            let gl = integrate(f, a, b, 40).unwrap();
            let ad = integrate_adaptive(f, a, b, 1e-12).unwrap();
            assert!(
                (gl - ad).abs() < 1e-10 * (1.0 + gl.abs()),
                "{name}: gauss {gl} vs adaptive {ad}"
            );
        }
    }

    #[test]
    fn adaptive_known_integrals() {
        let q = integrate_adaptive(f64::sin, 0.0, PI, 1e-13).unwrap();
        assert!((q - 2.0).abs() < 1e-11, "sin over [0,pi]: {q}");

        let q = integrate_adaptive(f64::exp, 0.0, 1.0, 1e-13).unwrap();
        assert!((q - (std::f64::consts::E - 1.0)).abs() < 1e-11, "exp: {q}");

        let q = integrate_adaptive(|x: f64| 1.0 / (1.0 + x * x), 0.0, 1.0, 1e-13).unwrap();
        assert!(
            (q - std::f64::consts::FRAC_PI_4).abs() < 1e-11,
            "1/(1+x^2): {q}"
        );
    }

    /// The peaked case, which is the reason the adaptive routine exists.
    /// The Lorentzian 1/(1 + 100 x^2) has a spike of half-width 0.1 in
    /// the middle of [-1, 1]; its exact integral is atan(10)/5.
    #[test]
    fn adaptive_handles_a_peaked_integrand() {
        let f = |x: f64| 1.0 / (1.0 + 100.0 * x * x);
        let exact = 10.0_f64.atan() / 5.0;

        for tol in [1e-6, 1e-9, 1e-12] {
            let q = integrate_adaptive(f, -1.0, 1.0, tol).unwrap();
            assert!(
                (q - exact).abs() < 10.0 * tol,
                "tol {tol}: got {q}, want {exact}, err {:e}",
                (q - exact).abs()
            );
        }

        // A much sharper peak, 1/(1 + 10^6 x^2) over [-1, 1]: half-width
        // 1e-3, exact value atan(1000)/500.
        let g = |x: f64| 1.0 / (1.0 + 1.0e6 * x * x);
        let exact_g = 1000.0_f64.atan() / 500.0;
        let q = integrate_adaptive(g, -1.0, 1.0, 1e-10).unwrap();
        assert!(
            (q - exact_g).abs() < 1e-8,
            "sharp peak: got {q}, want {exact_g}"
        );
    }

    /// Tightening the tolerance must actually tighten the answer.
    #[test]
    fn adaptive_tolerance_is_meaningful() {
        let f = |x: f64| (-x * x).exp();
        // integral of exp(-x^2) over [0, 2] = sqrt(pi)/2 * erf(2)
        let reference = integrate(f, 0.0, 2.0, 60).unwrap();
        let coarse = (integrate_adaptive(f, 0.0, 2.0, 1e-4).unwrap() - reference).abs();
        let fine = (integrate_adaptive(f, 0.0, 2.0, 1e-13).unwrap() - reference).abs();
        assert!(coarse < 1e-3, "coarse err {coarse}");
        assert!(fine < 1e-12, "fine err {fine}");
        assert!(
            fine <= coarse,
            "tightening tol did not help: {fine} vs {coarse}"
        );
    }

    // ---- 5. Brent on known roots ---------------------------------

    #[test]
    fn brent_finds_known_roots() {
        // The Dottie number, cos x = x.
        let r = brent_root(|x: f64| x.cos() - x, 0.0, 2.0, 1e-15).unwrap();
        assert!(
            (r - 0.7390851332151607_f64).abs() < 1e-12,
            "cos x - x: {r:.16}"
        );

        // Wallis's cubic, x^3 - 2x - 5 = 0 (Newton's own example).
        let r = brent_root(|x: f64| x * x * x - 2.0 * x - 5.0, 2.0, 3.0, 1e-15).unwrap();
        assert!(
            (r - 2.0945514815423265_f64).abs() < 1e-12,
            "x^3-2x-5: {r:.16}"
        );

        // sqrt(2) to 1e-12.
        let r = brent_root(|x: f64| x * x - 2.0, 0.0, 2.0, 1e-15).unwrap();
        assert!(
            (r - std::f64::consts::SQRT_2).abs() < 1e-12,
            "sqrt(2): {r:.16}"
        );

        // A few more, over wide and lopsided brackets.
        let r = brent_root(|x: f64| x.exp() - 2.0, -1.0, 5.0, 1e-15).unwrap();
        assert!((r - std::f64::consts::LN_2).abs() < 1e-12, "ln 2: {r:.16}");

        let r = brent_root(f64::sin, 3.0, 4.0, 1e-15).unwrap();
        assert!((r - PI).abs() < 1e-12, "pi: {r:.16}");

        // Endpoint that is already a root is returned as-is.
        let r = brent_root(f64::sin, 0.0, 1.0, 1e-12).unwrap();
        assert_eq!(r, 0.0);
    }

    /// Brent must beat bisection: 1e-15 on a smooth root in far fewer
    /// evaluations than the ~50 bisection would need.
    #[test]
    fn brent_converges_superlinearly() {
        use std::cell::Cell;
        let count = Cell::new(0_usize);
        let f = |x: f64| {
            count.set(count.get() + 1);
            x.cos() - x
        };
        let r = brent_root(f, 0.0, 2.0, 1e-15).unwrap();
        assert!((r - 0.7390851332151607_f64).abs() < 1e-12);
        assert!(
            count.get() < 25,
            "took {} evaluations — that is bisection, not Brent",
            count.get()
        );
    }

    // ---- 6. non-bracketing intervals ------------------------------

    #[test]
    fn brent_rejects_a_non_bracketing_interval() {
        // cos x - x is negative at both ends of [1, 2].
        let e = brent_root(|x: f64| x.cos() - x, 1.0, 2.0, 1e-12).unwrap_err();
        assert!(e.contains("bracket"), "unhelpful message: {e}");

        // x^2 + 1 never vanishes.
        assert!(brent_root(|x: f64| x * x + 1.0, -5.0, 5.0, 1e-12).is_err());

        // A double root: f touches zero without crossing, so there is no
        // sign change and Brent's guarantee genuinely does not apply.
        assert!(brent_root(|x: f64| x * x, -1.0, 1.0, 1e-12).is_err());

        // Positive at both ends of a wide interval.
        assert!(brent_root(f64::cos, -1.0, 1.0, 1e-12).is_err());
    }

    // ---- 7. find_roots --------------------------------------------

    #[test]
    fn find_roots_recovers_the_zeros_of_sine() {
        let roots = find_roots(f64::sin, 0.0, 10.0, 200, 1e-13).unwrap();
        assert_eq!(roots.len(), 4, "got {roots:?}");
        for (k, &r) in roots.iter().enumerate() {
            let want = k as f64 * PI;
            assert!(
                (r - want).abs() < 1e-11,
                "root {k}: got {r:.15}, want {want:.15}"
            );
        }
        // Ascending, as documented.
        for i in 0..roots.len() - 1 {
            assert!(roots[i] < roots[i + 1]);
        }
    }

    #[test]
    fn find_roots_on_a_polynomial_and_a_shifted_scan() {
        // (x-1)(x-2)(x-3)
        let p = |x: f64| (x - 1.0) * (x - 2.0) * (x - 3.0);
        let roots = find_roots(p, 0.0, 4.0, 100, 1e-14).unwrap();
        assert_eq!(roots.len(), 3, "got {roots:?}");
        for (i, &want) in [1.0, 2.0, 3.0].iter().enumerate() {
            assert!((roots[i] - want).abs() < 1e-12, "got {roots:?}");
        }

        // An interval whose endpoints are not roots, to exercise the
        // pure sign-change path with no exact-zero node.
        let roots = find_roots(f64::sin, 0.5, 10.0, 300, 1e-13).unwrap();
        assert_eq!(roots.len(), 3, "got {roots:?}");
        for (k, &r) in roots.iter().enumerate() {
            assert!((r - (k + 1) as f64 * PI).abs() < 1e-11);
        }

        // A scan sees only what it samples: the documented resolution
        // caveat, pinned. sin(50x) has 16 zeros in [0, 1] (at k pi/50
        // for k = 0..15), but a five-step scan can only report the
        // handful of sign changes it happens to straddle.
        let wiggly = |x: f64| (50.0 * x).sin();
        let fine = find_roots(wiggly, 0.0, 1.0, 500, 1e-13).unwrap();
        assert_eq!(fine.len(), 16, "got {fine:?}");
        for (k, &r) in fine.iter().enumerate() {
            assert!(
                (r - k as f64 * PI / 50.0).abs() < 1e-11,
                "zero {k}: got {r:.15}"
            );
        }
        let coarse = find_roots(wiggly, 0.0, 1.0, 5, 1e-13).unwrap();
        assert!(
            coarse.len() < fine.len(),
            "a five-step scan missed nothing, which cannot be right: {coarse:?}"
        );

        // Even multiplicity is invisible to a sign-change scan too:
        // (x-2)^2 touches zero without crossing.
        let touch = |x: f64| (x - 2.0) * (x - 2.0);
        assert!(find_roots(touch, 0.0, 5.0, 97, 1e-12).unwrap().is_empty());
    }

    #[test]
    fn find_roots_returns_an_empty_vec_when_there_are_none() {
        let roots = find_roots(|x: f64| x * x + 1.0, -5.0, 5.0, 100, 1e-12).unwrap();
        assert!(roots.is_empty(), "got {roots:?}");
    }

    // ---- 8. error paths -------------------------------------------

    #[test]
    fn bad_inputs_are_errors() {
        // gauss_legendre
        assert!(gauss_legendre(0).is_err());

        // integrate
        assert!(integrate(f64::sin, 0.0, 1.0, 0).is_err());
        assert!(integrate(f64::sin, f64::NAN, 1.0, 8).is_err());
        assert!(integrate(f64::sin, 0.0, f64::INFINITY, 8).is_err());
        assert!(integrate(f64::sin, 1.0, 1.0, 8).is_err());
        assert!(integrate(f64::sin, 2.0, 1.0, 8).is_err());
        assert!(integrate(|x: f64| 1.0 / (x - 0.5), 0.0, 1.0, 8).is_ok()); // no node at 0.5
        assert!(integrate(|_| f64::NAN, 0.0, 1.0, 8).is_err());
        assert!(integrate(|_| f64::INFINITY, 0.0, 1.0, 8).is_err());

        // integrate_adaptive
        assert!(integrate_adaptive(f64::sin, 1.0, 0.0, 1e-9).is_err());
        assert!(integrate_adaptive(f64::sin, 0.0, f64::NAN, 1e-9).is_err());
        assert!(integrate_adaptive(f64::sin, 0.0, 1.0, 0.0).is_err());
        assert!(integrate_adaptive(f64::sin, 0.0, 1.0, -1e-9).is_err());
        assert!(integrate_adaptive(f64::sin, 0.0, 1.0, f64::NAN).is_err());
        assert!(integrate_adaptive(|_| f64::NAN, 0.0, 1.0, 1e-9).is_err());

        // brent_root
        assert!(brent_root(f64::sin, 3.0, 3.0, 1e-12).is_err());
        assert!(brent_root(f64::sin, 4.0, 3.0, 1e-12).is_err());
        assert!(brent_root(f64::sin, f64::NAN, 4.0, 1e-12).is_err());
        assert!(brent_root(f64::sin, 3.0, 4.0, 0.0).is_err());
        assert!(brent_root(f64::sin, 3.0, 4.0, -1.0).is_err());
        assert!(brent_root(|_| f64::NAN, 3.0, 4.0, 1e-12).is_err());

        // find_roots
        assert!(find_roots(f64::sin, 0.0, 10.0, 0, 1e-12).is_err());
        assert!(find_roots(f64::sin, 10.0, 0.0, 10, 1e-12).is_err());
        assert!(find_roots(f64::sin, 0.0, f64::INFINITY, 10, 1e-12).is_err());
        assert!(find_roots(f64::sin, 0.0, 10.0, 10, 0.0).is_err());
        assert!(find_roots(|_| f64::NAN, 0.0, 10.0, 10, 1e-12).is_err());
    }

    /// Error messages name the routine and say what to do about it —
    /// the project's error policy, checked rather than assumed.
    #[test]
    fn error_messages_are_actionable() {
        let e = gauss_legendre(0).unwrap_err();
        assert!(e.starts_with("gauss_legendre:"), "{e}");
        let e = integrate(f64::sin, 2.0, 1.0, 4).unwrap_err();
        assert!(e.starts_with("integrate:") && e.contains("b > a"), "{e}");
        let e = integrate_adaptive(f64::sin, 0.0, 1.0, -1.0).unwrap_err();
        assert!(
            e.starts_with("integrate_adaptive:") && e.contains("tol"),
            "{e}"
        );
        let e = find_roots(f64::sin, 0.0, 1.0, 0, 1e-9).unwrap_err();
        assert!(e.starts_with("find_roots:") && e.contains("steps"), "{e}");
    }

    /// The non-convergence branches, reached through the capped
    /// internals. Both are unreachable from the public API for any sane
    /// input, which is exactly why they need testing here.
    #[test]
    fn iteration_caps_report_non_convergence() {
        // Newton with no iterations at all cannot converge.
        let e = legendre_root_newton(8, 0.9, 0).unwrap_err();
        assert!(
            e.contains("failed to converge") && e.contains("gauss_legendre:"),
            "{e}"
        );
        // Nor with too few for the asymptotic guess to be polished.
        assert!(legendre_root_newton(64, 0.999, 1).is_err());
        // But it does converge with the real cap.
        let (z, dp) = legendre_root_newton(8, 0.9, NEWTON_MAX_ITER).unwrap();
        assert!(z.abs() < 1.0 && dp.is_finite());
        let (p, _) = legendre_p_dp(8, z);
        assert!(p.abs() < 1e-15, "P_8({z}) = {p}");

        // Brent cut off after a single iteration cannot reach 1e-15.
        let e =
            brent_root_capped(|x: f64| x * x * x - 2.0 * x - 5.0, 2.0, 3.0, 1e-15, 1).unwrap_err();
        assert!(
            e.contains("failed to converge") && e.starts_with("brent_root:"),
            "{e}"
        );
        assert!(brent_root_capped(|x: f64| x.cos() - x, 0.0, 2.0, 1e-15, 2).is_err());
        // With the real cap it converges.
        assert!(brent_root_capped(|x: f64| x.cos() - x, 0.0, 2.0, 1e-15, BRENT_MAX_ITER).is_ok());
    }

    /// Adaptive Simpson refuses to pretend it resolved a singularity.
    #[test]
    fn adaptive_reports_an_unreachable_tolerance() {
        // 1/sqrt(x) is integrable on (0, 1] but Simpson cannot resolve
        // the endpoint; f(0) is infinite, which is caught immediately.
        let e = integrate_adaptive(|x: f64| 1.0 / x.sqrt(), 0.0, 1.0, 1e-12).unwrap_err();
        assert!(e.starts_with("integrate_adaptive:"), "{e}");

        // Shifting off the singularity keeps f finite everywhere, so
        // this one has to exhaust the recursion instead.
        let e = integrate_adaptive(|x: f64| 1.0 / x.sqrt(), 1e-300, 1.0, 1e-14).unwrap_err();
        assert!(e.contains("bisections") || e.contains("not finite"), "{e}");
    }

    // ---- the Legendre helper itself -------------------------------

    #[test]
    fn legendre_helper_matches_closed_forms() {
        for &x in &[-1.0_f64, -0.7, -0.3, 0.0, 0.25, 0.6, 1.0] {
            let (p1, d1) = legendre_p_dp(1, x);
            assert!(
                (p1 - x).abs() < 1e-15 && (d1 - 1.0).abs() < 1e-13,
                "n=1 x={x}"
            );

            let (p2, d2) = legendre_p_dp(2, x);
            assert!((p2 - 0.5 * (3.0 * x * x - 1.0)).abs() < 1e-15, "n=2 x={x}");
            assert!((d2 - 3.0 * x).abs() < 1e-13, "n=2' x={x}: {d2}");

            let (p3, d3) = legendre_p_dp(3, x);
            assert!(
                (p3 - 0.5 * (5.0 * x * x * x - 3.0 * x)).abs() < 1e-15,
                "n=3 x={x}"
            );
            assert!(
                (d3 - 0.5 * (15.0 * x * x - 3.0)).abs() < 1e-13,
                "n=3' x={x}"
            );
        }
        // The x = +-1 guard: no NaN, and the right closed form.
        assert!((legendre_p_dp(4, 1.0).1 - 10.0).abs() < 1e-13);
        assert!((legendre_p_dp(4, -1.0).1 + 10.0).abs() < 1e-13);
        assert!((legendre_p_dp(5, -1.0).1 - 15.0).abs() < 1e-13);
    }

    /// Every returned node really is a root of P_n, checked against the
    /// polynomial rather than against the weights that were derived
    /// from it.
    #[test]
    fn nodes_are_roots_of_the_legendre_polynomial() {
        for n in 1..=20_usize {
            let (x, _) = gauss_legendre(n).unwrap();
            for &xi in &x {
                let (p, dp) = legendre_p_dp(n, xi);
                // Scale by the derivative: |P_n / P_n'| is the distance
                // to the true root, which is the meaningful quantity.
                assert!(
                    (p / dp).abs() < 1e-15,
                    "n = {n}: P_n({xi}) = {p} is not a root (offset {})",
                    p / dp
                );
            }
        }
    }
}
