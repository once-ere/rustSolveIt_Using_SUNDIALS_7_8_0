//! Tridiagonal and cyclic-tridiagonal linear solvers, real and complex.
//!
//! These are what a Crank–Nicolson propagator needs: discretising
//! `i dpsi/dt = H psi` in the Cayley form
//! `(1 + iH dt/2) psi^{n+1} = (1 - iH dt/2) psi^n`
//! leaves a tridiagonal system to solve at every step, and periodic
//! boundary conditions make it cyclic.
//!
//! # Provenance
//!
//! **Clean-room.** Both algorithms are textbook and are implemented
//! here from their mathematical statements:
//!
//! * the tridiagonal solve is the **Thomas algorithm** — Gaussian
//!   elimination specialised to a tridiagonal matrix: one forward sweep
//!   eliminating the sub-diagonal, one back substitution;
//! * the cyclic case is the **Sherman–Morrison** rank-one correction,
//!   writing the cyclic matrix as `A' + u v^T` with `A'` tridiagonal,
//!   solving twice against `A'` and combining.
//!
//! No third-party implementation was consulted or transcribed.
//!
//! # Stability note
//!
//! The Thomas algorithm performs no pivoting, so it is not
//! unconditionally stable. It *is* stable for diagonally dominant or
//! symmetric positive-definite systems, which covers the Crank–Nicolson
//! operator: `1 + iH dt/2` has a strictly non-zero diagonal because of
//! the leading 1. A pivot that collapses to (near) zero is reported as
//! an error rather than producing silent garbage.

use crate::complex::Complex64;

/// Solve a real tridiagonal system.
///
/// `sub[i]` multiplies `x[i-1]` in row `i` (so `sub[0]` is unused),
/// `diag[i]` multiplies `x[i]`, and `sup[i]` multiplies `x[i+1]` (so
/// `sup[n-1]` is unused). All three slices have length `n`.
///
/// # Errors
/// Mismatched lengths, empty input, non-finite entries, or a pivot too
/// small to divide by (the matrix is singular or needs pivoting).
///
/// # Examples
/// ```
/// use special_functions::tridiag::solve_tridiag;
/// // [[2,1,0],[1,2,1],[0,1,2]] x = [1,2,3]  ->  x = [0.5, 0, 1.5]
/// let x = solve_tridiag(&[0.0,1.0,1.0], &[2.0,2.0,2.0], &[1.0,1.0,0.0], &[1.0,2.0,3.0]).unwrap();
/// assert!((x[0]-0.5).abs() < 1e-14);
/// assert!((x[1]-0.0).abs() < 1e-14);
/// assert!((x[2]-1.5).abs() < 1e-14);
/// ```
pub fn solve_tridiag(
    sub: &[f64],
    diag: &[f64],
    sup: &[f64],
    rhs: &[f64],
) -> Result<Vec<f64>, String> {
    let c: Vec<Complex64> = sub.iter().map(|&v| Complex64::real(v)).collect();
    let d: Vec<Complex64> = diag.iter().map(|&v| Complex64::real(v)).collect();
    let e: Vec<Complex64> = sup.iter().map(|&v| Complex64::real(v)).collect();
    let b: Vec<Complex64> = rhs.iter().map(|&v| Complex64::real(v)).collect();
    Ok(solve_tridiag_c(&c, &d, &e, &b)?
        .into_iter()
        .map(|z| z.re)
        .collect())
}

/// Solve a complex tridiagonal system by the Thomas algorithm.
///
/// Layout matches [`solve_tridiag`]. This is the workhorse behind a
/// Crank–Nicolson step.
///
/// # Errors
/// As [`solve_tridiag`].
///
/// # Examples
/// ```
/// use special_functions::complex::Complex64 as C;
/// use special_functions::tridiag::solve_tridiag_c;
/// // A purely imaginary diagonal: i*x = 1  =>  x = -i
/// let x = solve_tridiag_c(&[C::ZERO], &[C::I], &[C::ZERO], &[C::ONE]).unwrap();
/// assert!((x[0].re).abs() < 1e-15 && (x[0].im + 1.0).abs() < 1e-15);
/// ```
pub fn solve_tridiag_c(
    sub: &[Complex64],
    diag: &[Complex64],
    sup: &[Complex64],
    rhs: &[Complex64],
) -> Result<Vec<Complex64>, String> {
    let n = diag.len();
    if n == 0 {
        return Err("solve_tridiag: system is empty".to_string());
    }
    if sub.len() != n || sup.len() != n || rhs.len() != n {
        return Err(format!(
            "solve_tridiag: all slices must have length {n}; got sub={}, diag={n}, sup={}, rhs={}",
            sub.len(),
            sup.len(),
            rhs.len()
        ));
    }
    for (name, s) in [("sub", sub), ("diag", diag), ("sup", sup), ("rhs", rhs)] {
        if let Some(i) = s.iter().position(|z| !z.is_finite()) {
            return Err(format!("solve_tridiag: {name}[{i}] is not finite"));
        }
    }

    // Scale below which a pivot is treated as a failure rather than
    // divided by; relative to the largest entry so it is scale-free.
    let scale = diag
        .iter()
        .chain(sub)
        .chain(sup)
        .fold(0.0_f64, |m, z| m.max(z.abs()))
        .max(1.0);
    let tiny = 1.0e-13 * scale;

    // Forward sweep: eliminate the sub-diagonal.
    let mut cp = vec![Complex64::ZERO; n]; // modified super-diagonal
    let mut dp = vec![Complex64::ZERO; n]; // modified right-hand side
    let mut pivot = diag[0];
    if pivot.abs() <= tiny {
        return Err("solve_tridiag: zero pivot at row 0 (matrix is singular or needs pivoting)"
            .to_string());
    }
    cp[0] = sup[0] / pivot;
    dp[0] = rhs[0] / pivot;
    for i in 1..n {
        pivot = diag[i] - sub[i] * cp[i - 1];
        if pivot.abs() <= tiny {
            return Err(format!(
                "solve_tridiag: zero pivot at row {i} (matrix is singular or needs pivoting)"
            ));
        }
        cp[i] = sup[i] / pivot;
        dp[i] = (rhs[i] - sub[i] * dp[i - 1]) / pivot;
    }

    // Back substitution.
    let mut x = vec![Complex64::ZERO; n];
    x[n - 1] = dp[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = dp[i] - cp[i] * x[i + 1];
    }
    Ok(x)
}

/// Solve a **cyclic** (periodic) complex tridiagonal system, i.e. a
/// tridiagonal matrix with the two corner entries filled in:
/// `corner_bl` sits at row `n-1`, column `0`, and `corner_tr` at row
/// `0`, column `n-1`.
///
/// This is what periodic boundary conditions produce.
///
/// Method: Sherman–Morrison. Write `A = A' + u v^T` where `A'` is
/// tridiagonal with two diagonal entries perturbed, solve `A' y = rhs`
/// and `A' z = u`, then
/// `x = y - z (v·y) / (1 + v·z)`.
///
/// # Errors
/// As [`solve_tridiag_c`], plus `n < 3` (the corners are meaningless
/// for a smaller system — use the plain solver) and a vanishing
/// Sherman–Morrison denominator.
pub fn solve_cyclic_tridiag_c(
    sub: &[Complex64],
    diag: &[Complex64],
    sup: &[Complex64],
    corner_bl: Complex64,
    corner_tr: Complex64,
    rhs: &[Complex64],
) -> Result<Vec<Complex64>, String> {
    let n = diag.len();
    if n < 3 {
        return Err(format!(
            "solve_cyclic_tridiag: needs n >= 3 (got {n}); for smaller systems the corners \
             overlap the band — use solve_tridiag_c"
        ));
    }
    if sub.len() != n || sup.len() != n || rhs.len() != n {
        return Err("solve_cyclic_tridiag: all slices must have the same length".to_string());
    }

    // gamma is a free parameter; -diag[0] keeps the perturbed diagonal
    // well away from zero.
    let gamma = if diag[0].abs() > 0.0 {
        -diag[0]
    } else {
        Complex64::real(-1.0)
    };

    let mut d2 = diag.to_vec();
    d2[0] = diag[0] - gamma;
    d2[n - 1] = diag[n - 1] - corner_tr * corner_bl / gamma;

    // u = (gamma, 0, ..., 0, corner_bl);  v = (1, 0, ..., 0, corner_tr/gamma)
    let mut u = vec![Complex64::ZERO; n];
    u[0] = gamma;
    u[n - 1] = corner_bl;

    let y = solve_tridiag_c(sub, &d2, sup, rhs)?;
    let z = solve_tridiag_c(sub, &d2, sup, &u)?;

    let v_dot = |w: &[Complex64]| w[0] + w[n - 1] * (corner_tr / gamma);
    let denom = Complex64::ONE + v_dot(&z);
    if denom.abs() <= 1.0e-14 {
        return Err(
            "solve_cyclic_tridiag: Sherman-Morrison denominator vanished (matrix is singular)"
                .to_string(),
        );
    }
    let factor = v_dot(&y) / denom;
    Ok(y.iter().zip(&z).map(|(&yi, &zi)| yi - zi * factor).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::complex::Complex64 as C;

    /// Multiply a tridiagonal matrix by a vector, for residual checks.
    fn tri_mul(sub: &[C], diag: &[C], sup: &[C], x: &[C]) -> Vec<C> {
        let n = x.len();
        (0..n)
            .map(|i| {
                let mut s = diag[i] * x[i];
                if i > 0 {
                    s = s + sub[i] * x[i - 1];
                }
                if i + 1 < n {
                    s = s + sup[i] * x[i + 1];
                }
                s
            })
            .collect()
    }

    fn cyc_mul(
        sub: &[C],
        diag: &[C],
        sup: &[C],
        bl: C,
        tr: C,
        x: &[C],
    ) -> Vec<C> {
        let n = x.len();
        let mut y = tri_mul(sub, diag, sup, x);
        y[0] = y[0] + tr * x[n - 1];
        y[n - 1] = y[n - 1] + bl * x[0];
        y
    }

    fn max_resid(a: &[C], b: &[C]) -> f64 {
        a.iter()
            .zip(b)
            .map(|(p, q)| (*p - *q).abs())
            .fold(0.0_f64, f64::max)
    }

    #[test]
    fn real_system_matches_hand_solution() {
        // [[2,1,0],[1,2,1],[0,1,2]] x = [1,2,3]
        let x = solve_tridiag(
            &[0.0, 1.0, 1.0],
            &[2.0, 2.0, 2.0],
            &[1.0, 1.0, 0.0],
            &[1.0, 2.0, 3.0],
        )
        .unwrap();
        // Verified by hand: x1 = 1-2x0 and x2 = 1+x0 reduce row 2 to
        // -2x0 + 3 = 2, so x = [0.5, 0, 1.5]. (An earlier version of
        // this test carried a WRONG hand-solution while the solver was
        // correct — the residual check below is the one that matters.)
        assert!((x[0] - 0.5).abs() < 1e-14, "x0 = {}", x[0]);
        assert!((x[1] - 0.0).abs() < 1e-14, "x1 = {}", x[1]);
        assert!((x[2] - 1.5).abs() < 1e-14, "x2 = {}", x[2]);
    }

    /// The defining property: the residual `A x - b` must vanish.
    #[test]
    fn residual_vanishes_for_a_complex_system() {
        let n = 60;
        let sub: Vec<C> = (0..n).map(|i| C::new(-1.0, 0.05 * i as f64)).collect();
        let diag: Vec<C> = (0..n).map(|i| C::new(4.0 + 0.1 * i as f64, 1.0)).collect();
        let sup: Vec<C> = (0..n).map(|i| C::new(-1.0, -0.03 * i as f64)).collect();
        let rhs: Vec<C> = (0..n)
            .map(|i| C::new((i as f64).sin(), (0.5 * i as f64).cos()))
            .collect();
        let x = solve_tridiag_c(&sub, &diag, &sup, &rhs).unwrap();
        let r = tri_mul(&sub, &diag, &sup, &x);
        assert!(max_resid(&r, &rhs) < 1e-11, "residual {}", max_resid(&r, &rhs));
    }

    #[test]
    fn cyclic_residual_vanishes() {
        let n = 40;
        let sub = vec![C::new(-1.0, 0.0); n];
        let diag = vec![C::new(3.0, 0.7); n];
        let sup = vec![C::new(-1.0, 0.0); n];
        let bl = C::new(-1.0, 0.0);
        let tr = C::new(-1.0, 0.0);
        let rhs: Vec<C> = (0..n).map(|i| C::new(1.0, (i as f64) * 0.1)).collect();
        let x = solve_cyclic_tridiag_c(&sub, &diag, &sup, bl, tr, &rhs).unwrap();
        let r = cyc_mul(&sub, &diag, &sup, bl, tr, &x);
        assert!(max_resid(&r, &rhs) < 1e-10, "residual {}", max_resid(&r, &rhs));
    }

    /// With zero corners the cyclic solver must reproduce the plain one.
    #[test]
    fn cyclic_reduces_to_plain_when_corners_vanish() {
        let n = 20;
        let sub = vec![C::new(-1.0, 0.1); n];
        let diag = vec![C::new(5.0, 0.0); n];
        let sup = vec![C::new(-1.0, -0.2); n];
        let rhs: Vec<C> = (0..n).map(|i| C::real(i as f64)).collect();
        let a = solve_tridiag_c(&sub, &diag, &sup, &rhs).unwrap();
        let b = solve_cyclic_tridiag_c(&sub, &diag, &sup, C::ZERO, C::ZERO, &rhs).unwrap();
        assert!(max_resid(&a, &b) < 1e-12);
    }

    /// A Crank–Nicolson step must preserve the norm of the state: the
    /// Cayley operator is unitary. This is the physics the solver has to
    /// get right, and it is checked here end to end on a free particle.
    #[test]
    fn crank_nicolson_step_is_unitary_on_a_free_particle() {
        let n = 200usize;
        let dx = 0.05_f64;
        let dt = 0.01_f64;
        // psi_0: a Gaussian packet with momentum
        let mut psi: Vec<C> = (0..n)
            .map(|i| {
                let x = (i as f64 - n as f64 / 2.0) * dx;
                let g = (-x * x / 2.0).exp();
                C::from_polar(g, 3.0 * x)
            })
            .collect();
        let norm0: f64 = psi.iter().map(|z| z.norm_sqr()).sum::<f64>() * dx;

        // H = -1/2 d^2/dx^2 (free), tridiagonal with periodic wrap.
        let k = 0.5 / (dx * dx);
        let off = C::real(-k);
        let dia = C::real(2.0 * k);
        let half = C::I * (dt / 2.0);

        for _ in 0..25 {
            // rhs = (1 - i H dt/2) psi   (periodic)
            let rhs: Vec<C> = (0..n)
                .map(|i| {
                    let l = psi[(i + n - 1) % n];
                    let r = psi[(i + 1) % n];
                    let hp = dia * psi[i] + off * l + off * r;
                    psi[i] - half * hp
                })
                .collect();
            // (1 + i H dt/2) psi_new = rhs
            let sub = vec![half * off; n];
            let sup = vec![half * off; n];
            let diag = vec![C::ONE + half * dia; n];
            psi = solve_cyclic_tridiag_c(&sub, &diag, &sup, half * off, half * off, &rhs).unwrap();
        }
        let norm1: f64 = psi.iter().map(|z| z.norm_sqr()).sum::<f64>() * dx;
        assert!(
            (norm1 / norm0 - 1.0).abs() < 1e-10,
            "Crank-Nicolson must conserve the norm: {norm0} -> {norm1}"
        );
    }

    #[test]
    fn invalid_input_is_reported() {
        assert!(solve_tridiag(&[], &[], &[], &[]).is_err());
        assert!(solve_tridiag(&[0.0], &[1.0], &[0.0], &[1.0, 2.0]).is_err(), "length mismatch");
        assert!(
            solve_tridiag(&[0.0], &[0.0], &[0.0], &[1.0]).is_err(),
            "zero pivot must be reported"
        );
        assert!(
            solve_tridiag(&[0.0], &[f64::NAN], &[0.0], &[1.0]).is_err(),
            "non-finite entry"
        );
        let z = C::ZERO;
        assert!(
            solve_cyclic_tridiag_c(&[z; 2], &[z; 2], &[z; 2], z, z, &[z; 2]).is_err(),
            "n < 3 must be refused"
        );
    }
}
