//! The **Nash propagator** — a Bessel-expanded split-operator scheme.
//!
//! This is a faithful port of `CQMEvolve1D::EVOLVE_NASH` from the
//! original SolveIt C++ (`QM/QMEvolve1D.cpp`). It is the one piece of
//! genuinely original numerical work in that codebase — everything else
//! in the QM tree is either standard Crank–Nicolson or a
//! licence-encumbered library routine — and it is about twenty lines
//! long.
//!
//! # What the scheme is
//!
//! Discretise `H = -hbar^2/2m d^2/dx^2 + V(x)` on a **periodic** grid of
//! spacing `h` with the usual three-point stencil. Writing `S` for the
//! one-site shift `(S psi)_j = psi_{j+1}` and `kappa = hbar^2/(2 m h^2)`,
//!
//! ```text
//!   H = kappa (2 - S - S^-1) + V
//! ```
//!
//! Split the exponential into its diagonal and off-diagonal parts:
//!
//! ```text
//!   exp(-i H dt/hbar) ~ exp(-i lambda (1 + v))  *  exp(i (lambda/2)(S + S^-1))
//!
//!   lambda = hbar dt / (m h^2)          v_j = V_j m h^2 / hbar^2
//! ```
//!
//! The first factor is diagonal, so it is a pointwise phase. The second
//! is where the idea is. `S` has Fourier symbol `e^{i theta}`, so
//! `(S + S^-1)/2` has symbol `cos theta`, and the **Jacobi–Anger**
//! expansion
//!
//! ```text
//!   exp(i lambda cos theta) = sum_{M=-inf}^{inf} i^M J_M(lambda) e^{i M theta}
//! ```
//!
//! (DLMF 10.12.3) turns the kinetic exponential into a *finite-difference
//! stencil* whose coefficients are Bessel functions:
//!
//! ```text
//!   exp(i (lambda/2)(S + S^-1)) = sum_M i^M J_M(lambda) S^M
//! ```
//!
//! Because `J_{-M} = (-1)^M J_M` and `i^{-M} = (-1)^M i^M`, the term at
//! `-M` carries the *same* coefficient as the one at `+M`, so the two
//! sides fold together and one pass over the grid does the work:
//!
//! ```text
//!   psi_j <- exp(-i lambda (1 + v_j)) [ J_0(lambda) psi_j
//!            + sum_{M=1}^{K} i^M J_M(lambda) (psi_{j-M} + psi_{j+M}) ]
//! ```
//!
//! # Why it is worth having
//!
//! * **Explicit and matrix-free.** Crank–Nicolson ([`crate::qm1d::Propagator`])
//!   solves a tridiagonal system every step. This solves nothing.
//! * **Unitary by construction**, not by cancellation. Each factor is a
//!   unitary operator, so the norm is conserved to rounding for *any*
//!   step size — it is not a stability condition that can be violated.
//! * **The kinetic factor is essentially exact.** Truncating
//!   Jacobi–Anger at `K` leaves `2 sum_{M>K} |J_M(lambda)|`, and
//!   `|J_M(lambda)| <= (lambda/2)^M / M!`, so the error falls
//!   *superexponentially* once `K` passes `lambda`. At the original's
//!   settings — `lambda = 0.92`, `K = 16` — it is below `1e-16` and the
//!   kinetic half costs nothing in accuracy at all.
//!
//! # What limits it, and the one-line fix
//!
//! The accuracy is set by the **splitting**, not by the Bessel series.
//! `1 + v` and `S + S^-1` do not commute unless `V` is constant, so the
//! original's Lie–Trotter product is first order: the global error is
//! `O(dt)`, measured in `the_splitting_error_is_first_order_in_dt`.
//!
//! That is a property of the *ordering*, not of the Bessel idea, and
//! reordering fixes it. [`Splitting::Strang`] puts half a potential
//! phase on each side of the stencil and is **second order** —
//! `strang_is_second_order_in_dt` measures the error quartering as `dt`
//! halves, and `strang_beats_lie_at_the_same_step` measures it landing
//! more than 100x closer at the same step size.
//!
//! It is very nearly free. Consecutive Strang steps put a trailing
//! half-phase against a leading one, and those fuse into a single full
//! phase, so [`NashPropagator::run`] pays one extra half-phase over the
//! whole run rather than one per step.
//!
//! The default is still [`Splitting::Lie`], because the default has to
//! be what the original does — this is a port. Strang is opt-in through
//! [`NashPropagator::with_splitting`].
//!
//! The stencil is dense over `2K+1` points, so a step costs `O(n K)`
//! against Crank–Nicolson's `O(n)`. At `K = 16` that is real work, and
//! it buys an explicit scheme with no solver in it.
//!
//! # Boundary conditions
//!
//! **Periodic**, which is what the original's index wrap does and is a
//! genuine difference from [`crate::qm1d::Grid`], whose walls are
//! Dirichlet. A packet leaving the right edge re-enters at the left.

use special_functions::bessel::bessel_j_array;
use special_functions::complex::Complex64 as C;

/// A uniform **periodic** grid: `n` points, with `x_min` and `x_max`
/// identified.
///
/// Unlike [`crate::qm1d::Grid`] every point is a degree of freedom —
/// there are no walls to pin — so the spacing is `(x_max - x_min) / n`
/// and the first point sits exactly on `x_min`.
#[derive(Clone, Debug, PartialEq)]
pub struct PeriodicGrid {
    pub x_min: f64,
    pub x_max: f64,
    pub n: usize,
}

impl PeriodicGrid {
    /// # Errors
    /// `n == 0`, a non-finite bound, or `x_max <= x_min`.
    pub fn new(x_min: f64, x_max: f64, n: usize) -> Result<Self, String> {
        if n == 0 {
            return Err("PeriodicGrid: needs at least one point (n = 0)".to_string());
        }
        if !x_min.is_finite() || !x_max.is_finite() {
            return Err(format!("PeriodicGrid: bounds must be finite, got [{x_min}, {x_max}]"));
        }
        if x_max <= x_min {
            return Err(format!("PeriodicGrid: x_max ({x_max}) must exceed x_min ({x_min})"));
        }
        Ok(Self { x_min, x_max, n })
    }

    /// Spacing `h = (x_max - x_min) / n`.
    pub fn h(&self) -> f64 {
        (self.x_max - self.x_min) / self.n as f64
    }

    /// Position of point `i`. `x(0)` is exactly `x_min`.
    pub fn x(&self, i: usize) -> f64 {
        self.x_min + i as f64 * self.h()
    }

    /// All positions.
    pub fn points(&self) -> Vec<f64> {
        (0..self.n).map(|i| self.x(i)).collect()
    }
}

/// The smallest truncation order whose Bessel tail is below `tol`.
///
/// The tail is `2 sum_{M>K} |J_M(lambda)|`, and it is *not* estimated —
/// it is summed, out to where the terms leave `f64` range.
///
/// # Errors
/// A non-finite or negative `lambda`, a non-positive `tol`, or a
/// `lambda` so large that no order below the cap reaches `tol`.
///
/// # Examples
/// ```
/// use quantum::nash::order_for;
/// // The original SolveIt settings: lambda = 0.92 needs far fewer than
/// // the 16 terms it used.
/// let k = order_for(0.92, f64::EPSILON).unwrap();
/// assert!(k <= 16, "k = {k}");
/// ```
pub fn order_for(lambda: f64, tol: f64) -> Result<usize, String> {
    if !lambda.is_finite() || lambda < 0.0 {
        return Err(format!("order_for: lambda must be finite and non-negative, got {lambda}"));
    }
    if tol.is_nan() || tol <= 0.0 {
        return Err(format!("order_for: tol must be positive, got {tol}"));
    }
    // `truncation_bound` is floored at one eps because the stencil is
    // evaluated in f64, so a tolerance below that can never be met and
    // asking for it is a mistake worth naming rather than a search
    // worth running.
    if tol < f64::EPSILON {
        return Err(format!(
            "order_for: tol = {tol:.1e} is below the f64 evaluation floor {:.1e}; \
             no truncation order can deliver it",
            f64::EPSILON
        ));
    }
    const CAP: usize = 4096;
    for k in 1..=CAP {
        if truncation_bound(lambda, k)? <= tol {
            return Ok(k);
        }
    }
    Err(format!(
        "order_for: lambda = {lambda} needs more than {CAP} terms to reach {tol:.1e}; \
         reduce dt or coarsen the grid, since lambda = hbar dt / (m h^2)"
    ))
}

/// `2 sum_{M > k} |J_M(lambda)|` — the sup-norm error the truncated
/// Jacobi–Anger stencil makes on the kinetic symbol.
///
/// This is a **bound on the whole operator**, uniform in wavenumber:
/// the truncated stencil has symbol `P_k(theta)` and the exact one is
/// `exp(i lambda cos theta)`, and their difference is the tail of an
/// absolutely convergent series, so `|P_k - exact| <= 2 sum_{M>k} |J_M|`
/// for every `theta`.
///
/// Floored at one `eps`: the stencil is *evaluated* in `f64`, so
/// claiming better than rounding would be dishonest whatever the tail
/// does.
///
/// # Errors
/// A non-finite `lambda`.
pub fn truncation_bound(lambda: f64, k: usize) -> Result<f64, String> {
    // 60 orders past k is far beyond where (lambda/2)^M / M! has left
    // f64 range for any lambda this scheme is usable at; the terms are
    // exactly zero long before the end of the sum.
    let top = k + 60;
    let j = bessel_j_array(top, lambda)?;
    let tail: f64 = j[(k + 1).min(j.len())..].iter().map(|v| v.abs()).sum();
    Ok((2.0 * tail).max(f64::EPSILON))
}

/// Which way the exponential is split.
///
/// The two share every ingredient — the same Bessel stencil, the same
/// `lambda`, the same diagonal phase. They differ only in how the
/// factors are ordered, and that ordering is worth a whole order of
/// accuracy.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Splitting {
    /// `exp(-i L (1+v)) exp(i (L/2)(S + S^-1))` — what the original C++
    /// does, and the default here. First order in `dt`.
    Lie,
    /// `exp(-i L (1+v)/2) exp(i (L/2)(S + S^-1)) exp(-i L (1+v)/2)` —
    /// **second** order in `dt`, for one extra pointwise multiply.
    ///
    /// The gain is free in a way that is easy to miss: consecutive steps
    /// have a trailing half-phase against a leading half-phase, and
    /// those fuse into a single full phase. So `run` costs one extra
    /// half-phase in total, not one per step, and a run of `N` Strang
    /// steps costs essentially what `N` Lie steps cost while converging
    /// an order faster. [`NashPropagator::run`] does that fusion and
    /// `the_fused_run_matches_repeated_steps` proves it changes nothing.
    Strang,
}

/// A Bessel-expanded split-operator propagator on a periodic grid.
///
/// Built once for a fixed potential and time step; every step then costs
/// `O(n K)` multiply-adds and no allocation beyond one scratch buffer.
pub struct NashPropagator {
    grid: PeriodicGrid,
    hbar: f64,
    mass: f64,
    dt: f64,
    lambda: f64,
    /// `i^M J_M(lambda)` for `M = 0 ..= order`.
    coeff: Vec<C>,
    /// `exp(-i lambda (1 + v_j))`, the diagonal factor.
    phase: Vec<C>,
    /// The same factor at half a step, for [`Splitting::Strang`].
    half_phase: Vec<C>,
    /// The physical potential, kept so observables can be formed with
    /// the **periodic** Hamiltonian rather than a Dirichlet one.
    v: Vec<f64>,
    splitting: Splitting,
    truncation: f64,
}

impl NashPropagator {
    /// Build the propagator for potential `v` sampled at every grid
    /// point.
    ///
    /// `order` is the Jacobi–Anger truncation `K`. Pass `None` to have
    /// it chosen so the kinetic stencil is exact to rounding, which is
    /// almost always what you want — the cost is linear in `K` and the
    /// accuracy is superexponential, so there is no reason to skimp.
    ///
    /// # Errors
    /// A grid/potential length mismatch, a non-finite or non-positive
    /// `hbar`, `mass` or `dt`, a non-finite potential sample, or a
    /// `lambda` too large for the truncation to converge.
    pub fn new(
        grid: PeriodicGrid,
        v: &[f64],
        hbar: f64,
        mass: f64,
        dt: f64,
        order: Option<usize>,
    ) -> Result<Self, String> {
        if v.len() != grid.n {
            return Err(format!(
                "NashPropagator: the potential has {} samples but the grid has {} points",
                v.len(),
                grid.n
            ));
        }
        for (i, &vi) in v.iter().enumerate() {
            if !vi.is_finite() {
                return Err(format!("NashPropagator: V[{i}] is not finite ({vi})"));
            }
        }
        for (name, value) in [("hbar", hbar), ("mass", mass), ("dt", dt)] {
            if !value.is_finite() || value <= 0.0 {
                return Err(format!(
                    "NashPropagator: {name} must be finite and positive, got {value}"
                ));
            }
        }
        let h = grid.h();
        // lambda = hbar dt / (m h^2), v_j = V_j m h^2 / hbar^2. Together
        // these reproduce `-i H dt / hbar` exactly:
        //     -i lambda (1 + v_j) + i (lambda/2)(S + S^-1)
        let lambda = hbar * dt / (mass * h * h);
        if !lambda.is_finite() {
            return Err(format!(
                "NashPropagator: lambda = hbar dt / (m h^2) overflowed (h = {h:.3e})"
            ));
        }
        let order = match order {
            Some(k) => k,
            None => order_for(lambda, f64::EPSILON)?,
        };
        let truncation = truncation_bound(lambda, order)?;

        let j = bessel_j_array(order, lambda)?;
        // i^M, built by repeated multiplication rather than by a match
        // on M mod 4, so the sequence is right by construction.
        let mut coeff = Vec::with_capacity(order + 1);
        let mut power = C::ONE;
        for &jm in &j {
            coeff.push(power * jm);
            power = power * C::I;
        }

        let scale = mass * h * h / (hbar * hbar);
        let arg: Vec<f64> = v.iter().map(|&vi| -lambda * (1.0 + vi * scale)).collect();
        let phase = arg.iter().map(|&a| C::from_polar(1.0, a)).collect();
        let half_phase = arg.iter().map(|&a| C::from_polar(1.0, 0.5 * a)).collect();

        Ok(Self {
            grid,
            hbar,
            mass,
            dt,
            lambda,
            coeff,
            phase,
            half_phase,
            v: v.to_vec(),
            splitting: Splitting::Lie,
            truncation,
        })
    }

    /// Choose the splitting. The default is [`Splitting::Lie`], which
    /// is what the original C++ does.
    ///
    /// # Examples
    /// ```
    /// use quantum::nash::{NashPropagator, PeriodicGrid, Splitting};
    /// let grid = PeriodicGrid::new(0.0, 1.0, 16).unwrap();
    /// let p = NashPropagator::new(grid, &[0.0; 16], 1.0, 1.0, 1e-3, None)
    ///     .unwrap()
    ///     .with_splitting(Splitting::Strang);
    /// assert_eq!(p.splitting(), Splitting::Strang);
    /// ```
    #[must_use]
    pub fn with_splitting(mut self, splitting: Splitting) -> Self {
        self.splitting = splitting;
        self
    }

    pub fn splitting(&self) -> Splitting {
        self.splitting
    }

    pub fn grid(&self) -> &PeriodicGrid {
        &self.grid
    }

    pub fn dt(&self) -> f64 {
        self.dt
    }

    pub fn hbar(&self) -> f64 {
        self.hbar
    }

    pub fn mass(&self) -> f64 {
        self.mass
    }

    /// `lambda = hbar dt / (m h^2)` — the single dimensionless number
    /// the scheme runs on.
    pub fn lambda(&self) -> f64 {
        self.lambda
    }

    /// The Jacobi–Anger truncation order `K`.
    pub fn order(&self) -> usize {
        self.coeff.len() - 1
    }

    /// Bound on the error the truncation makes in the kinetic factor,
    /// uniform over wavenumber. See [`truncation_bound`].
    pub fn truncation_error(&self) -> f64 {
        self.truncation
    }

    /// The Bessel stencil alone — `exp(i (lambda/2)(S + S^-1))`, with
    /// no diagonal factor. Both splittings are built from this.
    fn kinetic(&self, psi: &mut [C], scratch: &mut Vec<C>) {
        scratch.clear();
        scratch.extend_from_slice(psi);
        let n = psi.len();
        let y = &scratch[..];
        for (i, out) in psi.iter_mut().enumerate() {
            let mut acc = y[i] * self.coeff[0];
            for (m, &c) in self.coeff.iter().enumerate().skip(1) {
                // Proper modular indexing. The original subtracted or
                // added N exactly once, which reads outside the array
                // whenever the stencil is wider than the grid — latent
                // in the C++ because NumOrder was 16 and NDATA was in
                // the hundreds.
                // `a_stencil_wider_than_the_grid_still_wraps_correctly`
                // pins the case that would have caught it.
                let left = y[(i + n - m % n) % n];
                let right = y[(i + m) % n];
                acc = acc + (left + right) * c;
            }
            *out = acc;
        }
    }

    fn apply(psi: &mut [C], phase: &[C]) {
        for (z, p) in psi.iter_mut().zip(phase) {
            *z = *p * *z;
        }
    }

    /// One step, in place. `scratch` is resized to hold a copy of `psi`.
    fn step_with(&self, psi: &mut [C], scratch: &mut Vec<C>) {
        match self.splitting {
            Splitting::Lie => {
                self.kinetic(psi, scratch);
                Self::apply(psi, &self.phase);
            }
            Splitting::Strang => {
                Self::apply(psi, &self.half_phase);
                self.kinetic(psi, scratch);
                Self::apply(psi, &self.half_phase);
            }
        }
    }

    /// Advance `psi` by one time step, in place.
    ///
    /// # Errors
    /// A length mismatch with the grid.
    pub fn step(&self, psi: &mut [C]) -> Result<(), String> {
        self.check(psi)?;
        let mut scratch = Vec::with_capacity(psi.len());
        self.step_with(psi, &mut scratch);
        Ok(())
    }

    /// Advance `psi` by `steps` steps, in place, allocating once.
    ///
    /// For [`Splitting::Strang`] the half-phases between consecutive
    /// steps are **fused** into full phases, so `N` steps cost `N`
    /// stencils, `N - 1` full phases and two half phases rather than
    /// `2N` half phases. The result is identical — that is what
    /// `the_fused_run_matches_repeated_steps` checks — and the cost
    /// comes out level with Lie for an order more accuracy.
    ///
    /// # Errors
    /// A length mismatch with the grid.
    pub fn run(&self, psi: &mut [C], steps: usize) -> Result<(), String> {
        self.check(psi)?;
        if steps == 0 {
            return Ok(());
        }
        let mut scratch = Vec::with_capacity(psi.len());
        match self.splitting {
            Splitting::Lie => {
                for _ in 0..steps {
                    self.step_with(psi, &mut scratch);
                }
            }
            Splitting::Strang => {
                Self::apply(psi, &self.half_phase);
                for k in 0..steps {
                    self.kinetic(psi, &mut scratch);
                    if k + 1 == steps {
                        Self::apply(psi, &self.half_phase);
                    } else {
                        Self::apply(psi, &self.phase);
                    }
                }
            }
        }
        Ok(())
    }

    /// `<psi|H|psi> / <psi|psi>` with the **periodic** Hamiltonian.
    ///
    /// This exists because the obvious alternative is wrong: forming the
    /// energy with a Dirichlet Hamiltonian on the same samples drops the
    /// two wrap terms, and those are exactly the terms that matter when
    /// a packet is near the seam — which is the only situation in which
    /// a periodic run differs from a Dirichlet one at all.
    ///
    /// # Errors
    /// A length mismatch with the grid, or an identically zero `psi`.
    pub fn energy(&self, psi: &[C]) -> Result<f64, String> {
        self.check(psi)?;
        let n = psi.len();
        let h = self.grid.h();
        let kappa = self.hbar * self.hbar / (2.0 * self.mass * h * h);
        let mut num = 0.0;
        let mut den = 0.0;
        for j in 0..n {
            let lap = psi[j] * 2.0 - psi[(j + n - 1) % n] - psi[(j + 1) % n];
            let hpsi = lap * kappa + psi[j] * self.v[j];
            // <psi|H|psi> is real because H is Hermitian; take the real
            // part rather than asserting it.
            num += (psi[j].conj() * hpsi).re;
            den += psi[j].norm_sqr();
        }
        if den == 0.0 {
            return Err("NashPropagator::energy: psi is identically zero".to_string());
        }
        Ok(num / den)
    }

    fn check(&self, psi: &[C]) -> Result<(), String> {
        if psi.len() != self.grid.n {
            return Err(format!(
                "NashPropagator: psi has {} points but the grid has {}",
                psi.len(),
                self.grid.n
            ));
        }
        Ok(())
    }
}

/// `sum_j |psi_j|^2 h` — the discrete norm on a periodic grid.
pub fn norm(psi: &[C], h: f64) -> f64 {
    psi.iter().map(|z| z.norm_sqr()).sum::<f64>() * h
}

#[cfg(test)]
mod tests {
    use super::*;
    use special_functions::eigen::jacobi_eigen;

    /// A normalised Gaussian wave packet, periodised by construction
    /// only in the sense that it has decayed to nothing at the edges.
    fn packet(grid: &PeriodicGrid, x0: f64, k0: f64, width: f64) -> Vec<C> {
        let mut psi: Vec<C> = (0..grid.n)
            .map(|i| {
                let x = grid.x(i);
                let g = (-((x - x0) / width).powi(2) / 2.0).exp();
                C::from_polar(g, k0 * x)
            })
            .collect();
        let s = norm(&psi, grid.h()).sqrt();
        for z in &mut psi {
            *z = *z * (1.0 / s);
        }
        psi
    }

    /// The exact propagator, by diagonalising the periodic Hamiltonian.
    ///
    /// This shares **no code** with the scheme under test: it builds the
    /// matrix, hands it to the Jacobi eigensolver, and sums
    /// `exp(-i E_m t / hbar) <phi_m|psi> phi_m`. It is the reference the
    /// splitting error is measured against.
    fn exact(grid: &PeriodicGrid, v: &[f64], hbar: f64, mass: f64, t: f64, psi0: &[C]) -> Vec<C> {
        let n = grid.n;
        let h = grid.h();
        let kappa = hbar * hbar / (2.0 * mass * h * h);
        let mut a = vec![vec![0.0; n]; n];
        for i in 0..n {
            a[i][i] = 2.0 * kappa + v[i];
            a[i][(i + 1) % n] += -kappa;
            a[i][(i + n - 1) % n] += -kappa;
        }
        let (e, vecs) = jacobi_eigen(&a).unwrap();
        // `vecs[m]` IS the eigenvector for `e[m]` — a row, not a column.
        // Reading it the other way silently produces a reference that
        // is wrong by O(1) and, being dt-independent, looks like a
        // scheme that has stopped converging.
        let mut out = vec![C::ZERO; n];
        for (m, phi) in vecs.iter().enumerate() {
            let mut c = C::ZERO;
            for (i, &p) in phi.iter().enumerate() {
                c = c + psi0[i] * p;
            }
            let ph = C::from_polar(1.0, -e[m] * t / hbar);
            for (i, &p) in phi.iter().enumerate() {
                out[i] = out[i] + c * ph * p;
            }
        }
        out
    }

    fn max_diff(a: &[C], b: &[C]) -> f64 {
        a.iter().zip(b).map(|(p, q)| (*p - *q).abs()).fold(0.0, f64::max)
    }

    /// **The sharpest check available, and it is exact.**
    ///
    /// With `V = 0` the two factors commute — `1 + v` is a constant —
    /// so there is no splitting error left, and the scheme reproduces
    /// `exp(-i H dt/hbar)` to the Bessel truncation alone. On a plane
    /// wave `psi_j = exp(i k x_j)` that propagator is a single known
    /// phase, `exp(-i lambda (1 - cos(k h)))`, so the test compares a
    /// number against a closed form with nothing fitted.
    ///
    /// It exercises lambda, every Bessel coefficient, every power of
    /// `i`, and the periodic wrap simultaneously: get any one of them
    /// wrong and this fails.
    #[test]
    fn a_plane_wave_gets_exactly_its_analytic_phase() {
        let grid = PeriodicGrid::new(0.0, 1.0, 64).unwrap();
        let v = vec![0.0; grid.n];
        let dt = 2.0e-4;
        let p = NashPropagator::new(grid.clone(), &v, 1.0, 1.0, dt, None).unwrap();
        let h = grid.h();
        for m in [0_i32, 1, 3, 7, 16, 31] {
            let k = 2.0 * std::f64::consts::PI * f64::from(m);
            let mut psi: Vec<C> =
                (0..grid.n).map(|i| C::from_polar(1.0, k * grid.x(i))).collect();
            let before = psi.clone();
            p.step(&mut psi).unwrap();
            let want = C::from_polar(1.0, -p.lambda() * (1.0 - (k * h).cos()));
            let got: Vec<C> = before.iter().map(|z| *z * want).collect();
            let e = max_diff(&psi, &got);
            assert!(e < 1e-13, "m = {m}: {e:.2e}");
        }
    }

    /// Unitary because each factor is, not because the errors cancelled.
    ///
    /// This holds for **any** step size — it is not a stability
    /// condition — so the test deliberately includes a `dt` far past
    /// where the scheme is accurate. Losing accuracy and losing norm are
    /// different failures and this separates them.
    #[test]
    fn the_norm_is_conserved_at_every_step_size() {
        let grid = PeriodicGrid::new(-8.0, 8.0, 128).unwrap();
        let v: Vec<f64> = grid.points().iter().map(|x| 0.5 * x * x).collect();
        for &dt in &[1e-4, 1e-2, 0.1, 1.0] {
            let p = NashPropagator::new(grid.clone(), &v, 1.0, 1.0, dt, None).unwrap();
            let mut psi = packet(&grid, -2.0, 3.0, 0.7);
            let n0 = norm(&psi, grid.h());
            p.run(&mut psi, 40).unwrap();
            let n1 = norm(&psi, grid.h());
            assert!(
                (n1 - n0).abs() < 1e-12,
                "dt = {dt}: norm {n0:.16} -> {n1:.16}, lambda = {:.2}",
                p.lambda()
            );
        }
    }

    /// The splitting is Lie–Trotter, so the global error is `O(dt)`.
    ///
    /// Measured against the diagonalised propagator, which shares no
    /// code with the scheme. Halving `dt` must halve the error; the
    /// assertion is on the *ratio*, so it cannot be satisfied by a
    /// scheme that is merely small-and-wrong.
    #[test]
    fn the_splitting_error_is_first_order_in_dt() {
        let grid = PeriodicGrid::new(-6.0, 6.0, 48).unwrap();
        let v: Vec<f64> = grid.points().iter().map(|x| 0.5 * x * x).collect();
        let psi0 = packet(&grid, -1.0, 2.0, 0.8);
        let t = 0.05;
        let want = exact(&grid, &v, 1.0, 1.0, t, &psi0);

        let mut errs = vec![];
        for steps in [25_usize, 50, 100, 200] {
            let dt = t / steps as f64;
            let p = NashPropagator::new(grid.clone(), &v, 1.0, 1.0, dt, None).unwrap();
            let mut psi = psi0.clone();
            p.run(&mut psi, steps).unwrap();
            errs.push(max_diff(&psi, &want));
        }
        for w in errs.windows(2) {
            let ratio = w[0] / w[1];
            assert!(
                (1.7..2.4).contains(&ratio),
                "halving dt should halve the error; got ratio {ratio:.3} from {errs:?}"
            );
        }
        assert!(errs[3] < 1e-3, "and the finest step should be accurate: {:.2e}", errs[3]);
    }

    /// The periodic energy, against a closed form.
    ///
    /// A plane wave is an exact eigenvector of the periodic Hamiltonian
    /// at constant potential, with eigenvalue `2 kappa (1 - cos k h) + V`
    /// — the lattice dispersion again. Nothing is fitted.
    #[test]
    fn the_energy_is_the_lattice_dispersion_on_a_plane_wave() {
        let grid = PeriodicGrid::new(0.0, 1.0, 64).unwrap();
        for &v0 in &[0.0_f64, 2.5, -1.25] {
            let v = vec![v0; grid.n];
            let p = NashPropagator::new(grid.clone(), &v, 1.0, 1.0, 1e-3, None).unwrap();
            let h = grid.h();
            let kappa = 1.0 / (2.0 * h * h);
            for m in [0_i32, 1, 5, 17, 32] {
                let k = 2.0 * std::f64::consts::PI * f64::from(m);
                let psi: Vec<C> =
                    (0..grid.n).map(|i| C::from_polar(1.0, k * grid.x(i))).collect();
                let want = 2.0 * kappa * (1.0 - (k * h).cos()) + v0;
                let got = p.energy(&psi).unwrap();
                assert!(
                    (got - want).abs() <= 1e-9 * want.abs().max(1.0),
                    "m = {m}, V = {v0}: {got} vs {want}"
                );
            }
        }
        let grid = PeriodicGrid::new(0.0, 1.0, 8).unwrap();
        let p = NashPropagator::new(grid, &[0.0; 8], 1.0, 1.0, 1e-3, None).unwrap();
        assert!(p.energy(&[C::ZERO; 8]).is_err());
        assert!(p.energy(&[C::ONE; 3]).is_err());
    }

    /// Strang is **second** order: halving `dt` quarters the error.
    ///
    /// Same reference as the Lie test — diagonalising `H` — so the two
    /// convergence rates are measured by the same instrument and the
    /// comparison between them means something.
    #[test]
    fn strang_is_second_order_in_dt() {
        let grid = PeriodicGrid::new(-6.0, 6.0, 48).unwrap();
        let v: Vec<f64> = grid.points().iter().map(|x| 0.5 * x * x).collect();
        let psi0 = packet(&grid, -1.0, 2.0, 0.8);
        let t = 0.05;
        let want = exact(&grid, &v, 1.0, 1.0, t, &psi0);

        let mut errs = vec![];
        for steps in [25_usize, 50, 100, 200] {
            let dt = t / steps as f64;
            let p = NashPropagator::new(grid.clone(), &v, 1.0, 1.0, dt, None)
                .unwrap()
                .with_splitting(Splitting::Strang);
            let mut psi = psi0.clone();
            p.run(&mut psi, steps).unwrap();
            errs.push(max_diff(&psi, &want));
        }
        for w in errs.windows(2) {
            let ratio = w[0] / w[1];
            assert!(
                (3.4..4.6).contains(&ratio),
                "halving dt should quarter the error; got ratio {ratio:.3} from {errs:?}"
            );
        }
    }

    /// The whole point, stated as a comparison rather than a claim:
    /// at the *same* step size Strang is dramatically closer.
    #[test]
    fn strang_beats_lie_at_the_same_step() {
        let grid = PeriodicGrid::new(-6.0, 6.0, 48).unwrap();
        let v: Vec<f64> = grid.points().iter().map(|x| 0.5 * x * x).collect();
        let psi0 = packet(&grid, -1.0, 2.0, 0.8);
        let t = 0.05;
        let steps = 100_usize;
        let want = exact(&grid, &v, 1.0, 1.0, t, &psi0);
        let base = NashPropagator::new(grid.clone(), &v, 1.0, 1.0, t / steps as f64, None).unwrap();

        let mut a = psi0.clone();
        base.run(&mut a, steps).unwrap();
        let lie = max_diff(&a, &want);

        let mut b = psi0.clone();
        base.with_splitting(Splitting::Strang).run(&mut b, steps).unwrap();
        let strang = max_diff(&b, &want);

        assert!(strang * 100.0 < lie, "Strang {strang:.2e} vs Lie {lie:.2e}");
    }

    /// `run` fuses the half-phases between consecutive Strang steps
    /// into full ones. That is an optimisation, so it has to be proved
    /// to change nothing.
    #[test]
    fn the_fused_run_matches_repeated_steps() {
        let grid = PeriodicGrid::new(-4.0, 4.0, 64).unwrap();
        let v: Vec<f64> = grid.points().iter().map(|x| 0.3 * x * x - 0.2 * x).collect();
        let p = NashPropagator::new(grid.clone(), &v, 1.0, 1.0, 3e-3, None)
            .unwrap()
            .with_splitting(Splitting::Strang);
        let psi0 = packet(&grid, -1.0, 3.0, 0.7);

        for steps in [0_usize, 1, 2, 5, 37] {
            let mut fused = psi0.clone();
            p.run(&mut fused, steps).unwrap();
            let mut plain = psi0.clone();
            for _ in 0..steps {
                p.step(&mut plain).unwrap();
            }
            let e = max_diff(&fused, &plain);
            assert!(e < 1e-14, "{steps} steps: fused and plain differ by {e:.2e}");
        }
    }

    /// Strang is still a product of unitaries, so it still conserves the
    /// norm at any step size — the second order buys accuracy, not
    /// stability, and those remain independent.
    #[test]
    fn strang_is_unitary_too() {
        let grid = PeriodicGrid::new(-8.0, 8.0, 128).unwrap();
        let v: Vec<f64> = grid.points().iter().map(|x| 0.5 * x * x).collect();
        for &dt in &[1e-4, 1e-2, 0.1, 1.0] {
            let p = NashPropagator::new(grid.clone(), &v, 1.0, 1.0, dt, None)
                .unwrap()
                .with_splitting(Splitting::Strang);
            let mut psi = packet(&grid, -2.0, 3.0, 0.7);
            let n0 = norm(&psi, grid.h());
            p.run(&mut psi, 40).unwrap();
            assert!((norm(&psi, grid.h()) - n0).abs() < 1e-12, "dt = {dt}");
        }
    }

    /// With `V = 0` the diagonal factor is constant, so it commutes with
    /// the stencil and the two splittings are the *same operator*.
    /// Strang must therefore still be exact on a plane wave, and must
    /// agree with Lie to rounding.
    #[test]
    fn with_no_potential_the_two_splittings_coincide() {
        let grid = PeriodicGrid::new(0.0, 1.0, 64).unwrap();
        let v = vec![0.0; grid.n];
        let lie = NashPropagator::new(grid.clone(), &v, 1.0, 1.0, 2.0e-4, None).unwrap();
        let strang = NashPropagator::new(grid.clone(), &v, 1.0, 1.0, 2.0e-4, None)
            .unwrap()
            .with_splitting(Splitting::Strang);
        for m in [1_i32, 7, 31] {
            let k = 2.0 * std::f64::consts::PI * f64::from(m);
            let psi0: Vec<C> =
                (0..grid.n).map(|i| C::from_polar(1.0, k * grid.x(i))).collect();
            let want = C::from_polar(1.0, -lie.lambda() * (1.0 - (k * grid.h()).cos()));
            let target: Vec<C> = psi0.iter().map(|z| *z * want).collect();

            let mut a = psi0.clone();
            strang.step(&mut a).unwrap();
            assert!(max_diff(&a, &target) < 1e-13, "Strang is not exact at m = {m}");

            let mut b = psi0.clone();
            lie.step(&mut b).unwrap();
            assert!(max_diff(&a, &b) < 1e-14, "the splittings must coincide at V = 0");
        }
    }

    /// The Bessel tail is the *only* thing the kinetic factor gets
    /// wrong, and it falls superexponentially.
    ///
    /// Compares the truncated stencil's symbol with `exp(i lambda cos
    /// theta)` directly, over a sweep of `theta`, and checks the
    /// measured sup-norm error against the reported bound.
    #[test]
    fn the_truncation_bound_actually_bounds_the_symbol_error() {
        for &lambda in &[0.25_f64, 0.92, 3.0, 8.0] {
            for k in 1..=20 {
                let bound = truncation_bound(lambda, k).unwrap();
                let j = bessel_j_array(k, lambda).unwrap();
                let mut worst: f64 = 0.0;
                for s in 0..97 {
                    let theta = std::f64::consts::PI * f64::from(s) / 96.0;
                    let mut sym = C::real(j[0]);
                    let mut power = C::I;
                    for (m, &jm) in j.iter().enumerate().skip(1) {
                        sym = sym + power * jm * (2.0 * (f64::from(m as u32) * theta).cos());
                        power = power * C::I;
                    }
                    let exact = C::from_polar(1.0, lambda * theta.cos());
                    worst = worst.max((sym - exact).abs());
                }
                assert!(
                    worst <= bound * 1.5 + 1e-15,
                    "lambda = {lambda}, k = {k}: measured {worst:.2e} exceeds bound {bound:.2e}"
                );
            }
        }
    }

    /// `order_for` is honest in both directions: the order it returns
    /// meets the tolerance, and the one below it does not.
    #[test]
    fn order_for_returns_the_smallest_order_that_works() {
        let tol = f64::EPSILON;
        for &lambda in &[0.5_f64, 0.92, 4.0, 12.0] {
            let k = order_for(lambda, tol).unwrap();
            assert!(truncation_bound(lambda, k).unwrap() <= tol, "lambda = {lambda}");
            if k > 1 {
                assert!(
                    truncation_bound(lambda, k - 1).unwrap() > tol,
                    "lambda = {lambda}: k = {k} is not minimal"
                );
            }
        }
        // And a tolerance the arithmetic cannot honour is refused, not
        // searched for.
        assert!(order_for(0.92, 1e-20).is_err());
        // The original's 16 terms at lambda = 0.92 were generous, which
        // is worth knowing rather than guessing at.
        assert!(order_for(0.92, f64::EPSILON).unwrap() < 16);
    }

    /// The wrap is a real modulo, not the original's single add.
    ///
    /// `EVOLVE_NASH` computed `mu(i-M, N)` by adding `N` **once**, so a
    /// stencil wider than the grid indexed outside the array. It never
    /// fired in SolveIt because `NumOrder` was 16 and `NDATA` was in the
    /// hundreds. Here it is simply correct, and this is the case that
    /// says so.
    #[test]
    fn a_stencil_wider_than_the_grid_still_wraps_correctly() {
        let grid = PeriodicGrid::new(0.0, 1.0, 5).unwrap();
        let v = vec![0.0; grid.n];
        let p = NashPropagator::new(grid.clone(), &v, 1.0, 1.0, 1e-3, Some(23)).unwrap();
        assert!(p.order() > grid.n, "the stencil must be wider than the grid to test this");
        let mut psi = vec![C::ZERO; grid.n];
        psi[0] = C::ONE;
        p.step(&mut psi).unwrap();
        assert!(psi.iter().all(|z| z.is_finite()));
        // A constant state is an eigenvector of any periodic stencil, so
        // it may only pick up a phase — a wrap that lands on the wrong
        // site would break that.
        let mut flat = vec![C::ONE; grid.n];
        p.step(&mut flat).unwrap();
        let first = flat[0];
        assert!(flat.iter().all(|z| (*z - first).abs() < 1e-14), "constant state was distorted");
    }

    /// Periodicity, stated as the identity it actually is.
    ///
    /// A translation-invariant propagator must **commute with the
    /// lattice shift**, and on a periodic grid that shift wraps. So
    /// evolving a cyclically shifted state must give the shifted
    /// evolved state — exactly, not approximately, and for shifts that
    /// cross the seam.
    ///
    /// This is far sharper than watching a packet travel and checking
    /// where it lands. That version was tried first and was wrong in an
    /// instructive way: on a lattice a packet moves at the *group*
    /// velocity `(hbar/m h) sin(k h)`, not `hbar k / m`, and at
    /// `k h = 1.56` those differ by 40%. Positions are a bad instrument
    /// here; an exact commutation is a good one.
    #[test]
    fn the_propagator_commutes_with_the_wrapping_shift() {
        let grid = PeriodicGrid::new(0.0, 10.0, 64).unwrap();
        // Constant potential, so the operator really is translation
        // invariant; a varying V would break the symmetry being tested.
        let v = vec![0.7; grid.n];
        let p = NashPropagator::new(grid.clone(), &v, 1.0, 1.0, 5e-3, None).unwrap();
        let psi0 = packet(&grid, 3.0, 6.0, 0.6);

        for shift in [1_usize, 7, 61, 63] {
            let shifted: Vec<C> =
                (0..grid.n).map(|i| psi0[(i + grid.n - shift) % grid.n]).collect();

            let mut a = shifted.clone();
            p.run(&mut a, 30).unwrap();

            let mut b = psi0.clone();
            p.run(&mut b, 30).unwrap();
            let b: Vec<C> = (0..grid.n).map(|i| b[(i + grid.n - shift) % grid.n]).collect();

            let e = max_diff(&a, &b);
            assert!(e < 1e-14, "shift {shift} crosses the seam and must be exact: {e:.2e}");
        }
    }

    /// A packet sitting **on** the seam is not a special case.
    ///
    /// Half its amplitude is at the top of the array and half at the
    /// bottom. Nothing may be lost there, and the density must stay
    /// smooth across the join.
    #[test]
    fn a_packet_straddling_the_seam_loses_nothing() {
        let grid = PeriodicGrid::new(0.0, 10.0, 256).unwrap();
        let v = vec![0.0; grid.n];
        let p = NashPropagator::new(grid.clone(), &v, 1.0, 1.0, 2e-3, None).unwrap();
        // Centred on x_min, so it wraps round to the far end.
        let mut psi: Vec<C> = (0..grid.n)
            .map(|i| {
                let mut d = grid.x(i) - grid.x_min;
                if d > 5.0 {
                    d -= 10.0;
                }
                C::from_polar((-(d / 0.6_f64).powi(2) / 2.0).exp(), 4.0 * d)
            })
            .collect();
        let s = norm(&psi, grid.h()).sqrt();
        for z in &mut psi {
            *z = *z * (1.0 / s);
        }
        let both_ends = psi[0].norm_sqr().min(psi[grid.n - 1].norm_sqr());
        assert!(both_ends > 1e-3, "the packet must actually straddle the seam");

        p.run(&mut psi, 300).unwrap();
        assert!((norm(&psi, grid.h()) - 1.0).abs() < 1e-12);
        // No kink at the join: the jump across the seam is no larger
        // than the largest jump anywhere else on the grid.
        let seam = (psi[0] - psi[grid.n - 1]).abs();
        let worst = (1..grid.n).map(|i| (psi[i] - psi[i - 1]).abs()).fold(0.0, f64::max);
        assert!(seam <= worst * 1.5, "kink at the seam: {seam:.2e} vs worst {worst:.2e}");
    }

    /// A statement-for-statement transliteration of the original
    /// `CQMEvolve1D::EVOLVE_NASH`, kept as the reference this port is
    /// judged against.
    ///
    /// It reproduces the C++ exactly, *including* `mu()`'s single-add
    /// wrap and the order in which the accumulation happens, and it is
    /// deliberately naive: coefficients recomputed as written, no
    /// folding, no precomputation. `bessj` is Numerical Recipes and is
    /// **not** transliterated — the clean-room `bessel_j_array` supplies
    /// the same `J_M(lambda)`.
    fn literal_evolve_nash(
        psi: &mut [C],
        jn: &[f64],
        im: &[C],
        u: &[C],
        num_order: usize,
        steps: usize,
    ) {
        let ndata = psi.len() as i64;
        let mu = |i: i64, n: i64| -> usize {
            let mut m = i;
            if i < 0 {
                m += n;
            }
            if i >= n {
                m -= n;
            }
            m as usize
        };
        for _ in 0..steps {
            let y = psi.to_vec();
            for i in 0..ndata {
                let mut acc = y[i as usize] * jn[0];
                for m in 1..=num_order as i64 {
                    acc = acc
                        + im[m as usize]
                            * jn[m as usize]
                            * (y[mu(i - m, ndata)] + y[mu(i + m, ndata)]);
                }
                psi[i as usize] = u[i as usize] * acc;
            }
        }
    }

    /// **The faithfulness test.** The port must agree with the original
    /// algorithm, at the original's own constants, to rounding.
    ///
    /// `lambda = 0.92` and `NumOrder = 16` are what SolveIt shipped.
    /// The port takes physical `dt`, `h`, `hbar`, `m` and `V` instead of
    /// the baked-in dimensionless pair, so the test drives it through
    /// that mapping — `dt = lambda h^2`, `V_j = v_j / h^2` at
    /// `hbar = m = 1` — which is also what makes the mapping itself
    /// testable rather than asserted.
    ///
    /// Any drift here is summation order and nothing else: the port
    /// folds the `+-M` pair and precomputes `i^M J_M`, the reference
    /// does neither.
    #[test]
    fn the_port_reproduces_the_original_algorithm_to_rounding() {
        let lambda = 0.92_f64;
        let num_order = 16_usize;
        let n = 400_usize;

        let jn = bessel_j_array(num_order, lambda).unwrap();
        let mut im = Vec::with_capacity(num_order + 1);
        let mut power = C::ONE;
        for _ in 0..=num_order {
            im.push(power);
            power = power * C::I;
        }

        let vdim: Vec<f64> = (0..n)
            .map(|j| {
                let t = j as f64 / n as f64;
                0.35 * (std::f64::consts::TAU * t).sin()
            })
            .collect();
        // u[x] = cos(L(1+v)) - i sin(L(1+v)), written as the original
        // wrote it rather than as exp(-i L (1+v)).
        let u: Vec<C> = vdim
            .iter()
            .map(|&vi| C::new((lambda * (1.0 + vi)).cos(), -(lambda * (1.0 + vi)).sin()))
            .collect();

        let grid = PeriodicGrid::new(-4.0, 4.0, n).unwrap();
        let h = grid.h();
        let psi0: Vec<C> = (0..n)
            .map(|j| {
                let x = grid.x(j);
                let g = (-((x + 2.0) / 0.5_f64).powi(2)).exp();
                C::from_polar(g, 64.25 * x)
            })
            .collect();

        let dt = lambda * h * h;
        let vphys: Vec<f64> = vdim.iter().map(|&v| v / (h * h)).collect();
        let prop =
            NashPropagator::new(grid, &vphys, 1.0, 1.0, dt, Some(num_order)).unwrap();
        assert!((prop.lambda() - lambda).abs() < 1e-14, "the mapping must reproduce lambda");

        for steps in [1_usize, 10, 100, 1000] {
            let mut a = psi0.clone();
            let mut b = psi0.clone();
            literal_evolve_nash(&mut a, &jn, &im, &u, num_order, steps);
            prop.run(&mut b, steps).unwrap();
            let d = max_diff(&a, &b);
            assert!(d < 1e-12, "after {steps} steps the port drifted {d:.2e} from the original");
        }
    }

    #[test]
    fn it_refuses_bad_input_rather_than_guessing() {
        let grid = PeriodicGrid::new(0.0, 1.0, 8).unwrap();
        assert!(PeriodicGrid::new(0.0, 1.0, 0).is_err());
        assert!(PeriodicGrid::new(1.0, 0.0, 8).is_err());
        assert!(NashPropagator::new(grid.clone(), &[0.0; 7], 1.0, 1.0, 0.1, None).is_err());
        assert!(NashPropagator::new(grid.clone(), &[0.0; 8], 0.0, 1.0, 0.1, None).is_err());
        assert!(NashPropagator::new(grid.clone(), &[0.0; 8], 1.0, -1.0, 0.1, None).is_err());
        assert!(NashPropagator::new(grid.clone(), &[f64::NAN; 8], 1.0, 1.0, 0.1, None).is_err());
        assert!(order_for(1.0, 0.0).is_err());
        assert!(order_for(f64::NAN, 1e-12).is_err());
        let p = NashPropagator::new(grid, &[0.0; 8], 1.0, 1.0, 0.1, None).unwrap();
        assert!(p.step(&mut [C::ONE; 3]).is_err());
    }

    /// The units mapping, checked against the definition rather than
    /// against itself: `lambda = hbar dt / (m h^2)`.
    #[test]
    fn lambda_follows_its_definition_in_every_unit() {
        for &(hbar, mass, dt, n) in &[(1.0, 1.0, 0.01, 32), (2.0, 3.0, 0.05, 64), (0.5, 0.25, 0.2, 16)]
        {
            let grid = PeriodicGrid::new(-1.0, 3.0, n).unwrap();
            let p =
                NashPropagator::new(grid.clone(), &vec![0.0; n], hbar, mass, dt, None).unwrap();
            let want = hbar * dt / (mass * grid.h() * grid.h());
            assert!((p.lambda() - want).abs() <= 1e-15 * want.abs());
        }
    }
}
