//! The 1-D solver: grid, Hamiltonian, bound states, and propagation.

use special_functions::complex::Complex64 as C;
use special_functions::eigen::jacobi_eigen;
use special_functions::tridiag::solve_tridiag_c;

/// A uniform grid of `n` **interior** points on `[x_min, x_max]`.
///
/// The endpoints themselves are not grid points: the wavefunction is
/// pinned to zero there, which is what makes the domain an infinite
/// well. So the spacing is `(x_max - x_min) / (n + 1)` and the first
/// point sits one spacing inside the left wall.
#[derive(Clone, Debug, PartialEq)]
pub struct Grid {
    pub x_min: f64,
    pub x_max: f64,
    pub n: usize,
}

impl Grid {
    /// # Errors
    /// `n == 0`, a non-finite bound, or `x_max <= x_min`.
    pub fn new(x_min: f64, x_max: f64, n: usize) -> Result<Self, String> {
        if n == 0 {
            return Err("Grid: needs at least one interior point (n = 0)".to_string());
        }
        if !x_min.is_finite() || !x_max.is_finite() {
            return Err(format!("Grid: bounds must be finite, got [{x_min}, {x_max}]"));
        }
        if x_max <= x_min {
            return Err(format!("Grid: x_max ({x_max}) must exceed x_min ({x_min})"));
        }
        Ok(Self { x_min, x_max, n })
    }

    /// Spacing `h`.
    pub fn h(&self) -> f64 {
        (self.x_max - self.x_min) / (self.n + 1) as f64
    }

    /// Position of interior point `i`, counting from 0.
    pub fn x(&self, i: usize) -> f64 {
        self.x_min + (i + 1) as f64 * self.h()
    }

    /// All interior positions.
    pub fn points(&self) -> Vec<f64> {
        (0..self.n).map(|i| self.x(i)).collect()
    }
}

/// A discretised Hamiltonian `H = -hbar^2/2m d^2/dx^2 + V(x)`.
///
/// The kinetic term uses the standard three-point stencil, so the
/// matrix is symmetric tridiagonal: `kinetic*2 + V_i` on the diagonal
/// and `-kinetic` beside it.
#[derive(Clone, Debug)]
pub struct Hamiltonian {
    pub grid: Grid,
    /// `V` sampled at each interior grid point.
    pub potential: Vec<f64>,
    pub mass: f64,
    pub hbar: f64,
    /// An optional **complex absorbing potential** `W(x) >= 0`.
    ///
    /// The effective Hamiltonian becomes `H - i W(x)`, which drains
    /// probability wherever `W > 0` instead of reflecting it. See
    /// [`Hamiltonian::with_absorber`].
    pub absorber: Option<Vec<f64>>,
}

impl Hamiltonian {
    /// Build from a potential already sampled on the grid.
    ///
    /// # Errors
    /// Length mismatch, non-finite or non-positive `mass`/`hbar`, or a
    /// non-finite potential value.
    pub fn new(grid: Grid, potential: Vec<f64>, mass: f64, hbar: f64) -> Result<Self, String> {
        if potential.len() != grid.n {
            return Err(format!(
                "Hamiltonian: potential has {} values but the grid has {} points",
                potential.len(),
                grid.n
            ));
        }
        if let Some(i) = potential.iter().position(|v| !v.is_finite()) {
            return Err(format!(
                "Hamiltonian: V(x = {}) is not finite — an infinite potential must be \
                 represented by shrinking the domain, not by an infinite value",
                grid.x(i)
            ));
        }
        if !mass.is_finite() || mass <= 0.0 {
            return Err(format!("Hamiltonian: mass must be finite and positive, got {mass}"));
        }
        if !hbar.is_finite() || hbar <= 0.0 {
            return Err(format!("Hamiltonian: hbar must be finite and positive, got {hbar}"));
        }
        Ok(Self { grid, potential, mass, hbar, absorber: None })
    }

    /// Attach a **complex absorbing potential** (CAP) to both edges.
    ///
    /// The Dirichlet walls reflect, which is fatal for scattering: the
    /// domain has to be long enough that nothing reaches them, and that
    /// cost grows with the time you want to simulate. A CAP removes the
    /// constraint by making the edges *absorb*. The effective
    /// Hamiltonian is
    ///
    /// ```text
    ///     H_eff = H - i W(x),
    ///     W(x)  = strength * ((|x| - x_on) / width)^power   for |x| > x_on
    /// ```
    ///
    /// where `x_on` is `width` inside each wall. A smooth polynomial
    /// ramp is used rather than a step because an abrupt `W` reflects
    /// almost as badly as the wall it replaces — the whole point is to
    /// be gentle enough that the packet does not notice it arriving.
    ///
    /// # The trade-off, which is real and unavoidable
    ///
    /// A CAP that is too weak lets the packet reach the wall and reflect
    /// off it; one that is too strong reflects off the *absorber*. The
    /// usable window widens as `width` grows, so prefer a wide, gentle
    /// absorber over a narrow, fierce one. [`Propagator::reflection_probe`]
    /// measures what you actually got rather than leaving you to trust
    /// a rule of thumb.
    ///
    /// # Consequences
    ///
    /// * Propagation is **no longer unitary** — the norm decays, by
    ///   design. That is the absorber working, not a solver defect.
    /// * `<E>` is no longer conserved and is no longer real in general.
    ///   [`Wavefunction::energy`] returns the real part.
    /// * [`Hamiltonian::bound_states`] **refuses** to run: the matrix is
    ///   no longer Hermitian, so a symmetric eigensolver would return
    ///   confident nonsense.
    ///
    /// # Errors
    /// Non-finite or non-positive `width`/`strength`, a `width` that
    /// does not fit in the domain, or `power < 1`.
    pub fn with_absorber(mut self, width: f64, strength: f64, power: f64) -> Result<Self, String> {
        if !width.is_finite() || width <= 0.0 {
            return Err(format!("with_absorber: width must be finite and positive, got {width}"));
        }
        if !strength.is_finite() || strength <= 0.0 {
            return Err(format!(
                "with_absorber: strength must be finite and positive, got {strength}"
            ));
        }
        if !power.is_finite() || power < 1.0 {
            return Err(format!(
                "with_absorber: power must be at least 1 (2 or 3 is usual), got {power}"
            ));
        }
        let span = self.grid.x_max - self.grid.x_min;
        if 2.0 * width >= span {
            return Err(format!(
                "with_absorber: two absorbers of width {width} do not fit in a domain of \
                 width {span} — they would overlap and leave no interior"
            ));
        }
        let lo_on = self.grid.x_min + width;
        let hi_on = self.grid.x_max - width;
        let w: Vec<f64> = (0..self.grid.n)
            .map(|i| {
                let x = self.grid.x(i);
                let d = if x < lo_on {
                    (lo_on - x) / width
                } else if x > hi_on {
                    (x - hi_on) / width
                } else {
                    0.0
                };
                strength * d.powf(power)
            })
            .collect();
        self.absorber = Some(w);
        Ok(self)
    }

    /// Remove any absorbing potential.
    pub fn without_absorber(mut self) -> Self {
        self.absorber = None;
        self
    }

    /// Whether an absorbing potential is attached.
    pub fn is_absorbing(&self) -> bool {
        self.absorber.is_some()
    }

    /// Build by sampling a closure at each grid point.
    ///
    /// # Errors
    /// As [`Hamiltonian::new`].
    pub fn from_fn<F: Fn(f64) -> f64>(
        grid: Grid,
        v: F,
        mass: f64,
        hbar: f64,
    ) -> Result<Self, String> {
        let potential = (0..grid.n).map(|i| v(grid.x(i))).collect();
        Self::new(grid, potential, mass, hbar)
    }

    /// The off-diagonal coefficient `-hbar^2 / (2 m h^2)`.
    fn off_diagonal(&self) -> f64 {
        let h = self.grid.h();
        -self.hbar * self.hbar / (2.0 * self.mass * h * h)
    }

    /// Diagonal entry `i` of the REAL part.
    fn diagonal(&self, i: usize) -> f64 {
        -2.0 * self.off_diagonal() + self.potential[i]
    }

    /// Diagonal entry `i` of the effective Hamiltonian, `H - i W`.
    /// Identical to [`Hamiltonian::diagonal`] when no absorber is set.
    fn diagonal_c(&self, i: usize) -> C {
        let re = self.diagonal(i);
        match &self.absorber {
            Some(w) => C::new(re, -w[i]),
            None => C::real(re),
        }
    }

    /// Apply `H` to a complex wavefunction, returning `H psi`.
    pub fn apply(&self, psi: &[C]) -> Vec<C> {
        let n = self.grid.n;
        let off = self.off_diagonal();
        (0..n)
            .map(|i| {
                let mut s = self.diagonal_c(i) * psi[i];
                if i > 0 {
                    s = s + psi[i - 1] * off;
                }
                if i + 1 < n {
                    s = s + psi[i + 1] * off;
                }
                s
            })
            .collect()
    }

    /// The lowest `k` bound states: `(energies, wavefunctions)`.
    ///
    /// Energies are ascending. Each wavefunction is **normalised so that
    /// `integral |psi|^2 dx = 1`**, not merely unit-norm as a vector —
    /// the eigenvector must be divided by `sqrt(h)` to be a
    /// wavefunction, and forgetting that is a silent factor-of-`h` error
    /// in every expectation value you compute afterwards.
    ///
    /// Sign convention: the component of largest magnitude is made
    /// positive, so results are reproducible rather than depending on
    /// eigensolver internals.
    ///
    /// # Errors
    /// `k == 0` or `k > n`, or an eigensolver failure.
    ///
    /// # Cost
    /// This diagonalises a **dense** `n x n` matrix by cyclic Jacobi,
    /// which is `O(n^3)`. It is comfortable to a few hundred points and
    /// slow beyond roughly a thousand. That is a real limit, not a
    /// tuning parameter — a banded or iterative solver would be a
    /// separate piece of work.
    pub fn bound_states(&self, k: usize) -> Result<(Vec<f64>, Vec<Vec<f64>>), String> {
        let n = self.grid.n;
        if self.is_absorbing() {
            return Err(
                "bound_states: an absorbing potential makes the Hamiltonian NON-Hermitian, so a \
                 symmetric eigensolver would return confident nonsense. Remove the absorber \
                 (it is only meaningful for scattering) and try again."
                    .to_string(),
            );
        }
        if k == 0 {
            return Err("bound_states: k must be at least 1".to_string());
        }
        if k > n {
            return Err(format!(
                "bound_states: asked for {k} states but the grid has only {n} points"
            ));
        }
        let off = self.off_diagonal();
        let mut m = vec![vec![0.0; n]; n];
        for i in 0..n {
            m[i][i] = self.diagonal(i);
            if i + 1 < n {
                m[i][i + 1] = off;
                m[i + 1][i] = off;
            }
        }
        let (vals, vecs) = jacobi_eigen(&m)?;

        let h = self.grid.h();
        let scale = 1.0 / h.sqrt();
        let mut states = Vec::with_capacity(k);
        for col in vecs.iter().take(k) {
            let mut v: Vec<f64> = col.iter().map(|c| c * scale).collect();
            // reproducible sign
            let lead = v
                .iter()
                .cloned()
                .fold(0.0_f64, |acc, x| if x.abs() > acc.abs() { x } else { acc });
            if lead < 0.0 {
                for e in v.iter_mut() {
                    *e = -*e;
                }
            }
            states.push(v);
        }
        Ok((vals.into_iter().take(k).collect(), states))
    }
}

/// A complex wavefunction on a grid.
#[derive(Clone, Debug)]
pub struct Wavefunction {
    pub grid: Grid,
    pub psi: Vec<C>,
}

impl Wavefunction {
    /// # Errors
    /// Length mismatch with the grid.
    pub fn new(grid: Grid, psi: Vec<C>) -> Result<Self, String> {
        if psi.len() != grid.n {
            return Err(format!(
                "Wavefunction: {} values for a grid of {} points",
                psi.len(),
                grid.n
            ));
        }
        Ok(Self { grid, psi })
    }

    /// A normalised Gaussian wavepacket
    /// `exp(-(x-x0)^2 / 4 sigma^2) exp(i k0 x)`.
    ///
    /// `sigma` is the width of `|psi|^2`, so the momentum spread is
    /// `hbar / 2 sigma`.
    ///
    /// # Errors
    /// Non-positive or non-finite `sigma`, or a packet so narrow or so
    /// far outside the domain that its norm underflows.
    pub fn gaussian(grid: Grid, x0: f64, sigma: f64, k0: f64) -> Result<Self, String> {
        if !sigma.is_finite() || sigma <= 0.0 {
            return Err(format!("gaussian: sigma must be finite and positive, got {sigma}"));
        }
        if !x0.is_finite() || !k0.is_finite() {
            return Err("gaussian: x0 and k0 must be finite".to_string());
        }
        let psi: Vec<C> = (0..grid.n)
            .map(|i| {
                let x = grid.x(i);
                let g = (-(x - x0) * (x - x0) / (4.0 * sigma * sigma)).exp();
                C::from_polar(g, k0 * x)
            })
            .collect();
        let mut w = Self::new(grid, psi)?;
        w.normalise()?;
        Ok(w)
    }

    /// A real state (e.g. a bound state) lifted to a complex one.
    ///
    /// # Errors
    /// Length mismatch.
    pub fn from_real(grid: Grid, values: &[f64]) -> Result<Self, String> {
        Self::new(grid, values.iter().map(|&v| C::real(v)).collect())
    }

    /// `integral |psi|^2 dx`.
    pub fn norm(&self) -> f64 {
        self.psi.iter().map(|z| z.norm_sqr()).sum::<f64>() * self.grid.h()
    }

    /// Scale to unit norm.
    ///
    /// # Errors
    /// The norm is zero or not finite.
    pub fn normalise(&mut self) -> Result<(), String> {
        let n = self.norm();
        if !n.is_finite() || n <= 0.0 {
            return Err(format!(
                "normalise: the norm is {n} — the state is empty, or the packet lies \
                 entirely outside [{}, {}]",
                self.grid.x_min, self.grid.x_max
            ));
        }
        let s = 1.0 / n.sqrt();
        for z in self.psi.iter_mut() {
            *z = *z * s;
        }
        Ok(())
    }

    /// `<x>`.
    pub fn position(&self) -> f64 {
        let h = self.grid.h();
        self.psi
            .iter()
            .enumerate()
            .map(|(i, z)| self.grid.x(i) * z.norm_sqr())
            .sum::<f64>()
            * h
    }

    /// `<p>` via the central-difference momentum operator
    /// `-i hbar d/dx`. Real by construction for a normalisable state;
    /// the imaginary part is discarded after being used as a check in
    /// the tests.
    pub fn momentum(&self, hbar: f64) -> f64 {
        let h = self.grid.h();
        let n = self.grid.n;
        let mut acc = C::ZERO;
        for i in 0..n {
            let left = if i > 0 { self.psi[i - 1] } else { C::ZERO };
            let right = if i + 1 < n { self.psi[i + 1] } else { C::ZERO };
            let d = (right - left) * (1.0 / (2.0 * h));
            // psi* (-i hbar dpsi/dx)
            acc = acc + self.psi[i].conj() * (C::I * (-hbar)) * d;
        }
        acc.re * h
    }

    /// `<H> = <psi|H|psi> / <psi|psi>`.
    ///
    /// Divided by the norm ON PURPOSE. For a unit-norm state the two
    /// agree, but an absorbing potential makes the norm decay, and the
    /// raw integral then decays with it — reporting a falling "energy"
    /// for a packet whose energy has not changed. An earlier version
    /// omitted the division and showed 6.83 for a k = 8 packet whose
    /// energy is 32, purely because 78 % of it had been absorbed.
    pub fn energy(&self, ham: &Hamiltonian) -> f64 {
        let hpsi = ham.apply(&self.psi);
        let h = self.grid.h();
        let num: f64 = self
            .psi
            .iter()
            .zip(&hpsi)
            .map(|(p, q)| (p.conj() * *q).re)
            .sum::<f64>()
            * h;
        let den = self.norm();
        if den > 0.0 {
            num / den
        } else {
            f64::NAN
        }
    }

    /// Probability of finding the particle in `[a, b]`.
    pub fn probability_in(&self, a: f64, b: f64) -> f64 {
        let h = self.grid.h();
        let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
        self.psi
            .iter()
            .enumerate()
            .filter(|(i, _)| {
                let x = self.grid.x(*i);
                x >= lo && x <= hi
            })
            .map(|(_, z)| z.norm_sqr())
            .sum::<f64>()
            * h
    }

    /// Probability sitting within `frac` of the domain width of either
    /// wall.
    ///
    /// The walls reflect, so a scattering result is only trustworthy
    /// while this stays negligible. It exists so that assumption can be
    /// *checked* rather than hoped for.
    pub fn edge_probability(&self, frac: f64) -> f64 {
        let w = (self.grid.x_max - self.grid.x_min) * frac;
        self.probability_in(self.grid.x_min, self.grid.x_min + w)
            + self.probability_in(self.grid.x_max - w, self.grid.x_max)
    }

    /// `|psi|^2` at every grid point — the probability density.
    pub fn density(&self) -> Vec<f64> {
        self.psi.iter().map(|z| z.norm_sqr()).collect()
    }
}

/// A Crank–Nicolson propagator for a fixed Hamiltonian and time step.
///
/// The Cayley form
/// `(1 + i H dt/2 / hbar) psi^{n+1} = (1 - i H dt/2 / hbar) psi^n`
/// is **unitary for any `dt`** — that is exact, not asymptotic. So norm
/// drift measures the linear solver, never the step size, which makes it
/// the sharpest available check on the whole apparatus.
///
/// The operator is time-independent, so its bands are built once here
/// and reused for every step.
pub struct Propagator {
    ham: Hamiltonian,
    dt: f64,
    sub: Vec<C>,
    sup: Vec<C>,
    lhs_diag: Vec<C>,
    half: C,
}

impl Propagator {
    /// # Errors
    /// A non-finite or zero `dt`.
    pub fn new(ham: Hamiltonian, dt: f64) -> Result<Self, String> {
        if !dt.is_finite() || dt == 0.0 {
            return Err(format!("Propagator: dt must be finite and non-zero, got {dt}"));
        }
        let n = ham.grid.n;
        // i dt / (2 hbar)
        let half = C::I * (dt / (2.0 * ham.hbar));
        let off = C::real(ham.off_diagonal());
        let sub = vec![half * off; n];
        let sup = vec![half * off; n];
        let lhs_diag = (0..n)
            .map(|i| C::ONE + half * ham.diagonal_c(i))
            .collect();
        Ok(Self { ham, dt, sub, sup, lhs_diag, half })
    }

    pub fn dt(&self) -> f64 {
        self.dt
    }

    pub fn hamiltonian(&self) -> &Hamiltonian {
        &self.ham
    }

    /// Advance `w` by one time step, in place.
    ///
    /// # Errors
    /// A tridiagonal solve failure, or a grid mismatch.
    pub fn step(&self, w: &mut Wavefunction) -> Result<(), String> {
        if w.grid != self.ham.grid {
            return Err("step: the wavefunction and the propagator use different grids".to_string());
        }
        let hpsi = self.ham.apply(&w.psi);
        let rhs: Vec<C> = w
            .psi
            .iter()
            .zip(&hpsi)
            .map(|(p, q)| *p - self.half * *q)
            .collect();
        w.psi = solve_tridiag_c(&self.sub, &self.lhs_diag, &self.sup, &rhs)?;
        Ok(())
    }

    /// Advance by `steps` steps.
    ///
    /// # Errors
    /// As [`Propagator::step`].
    pub fn run(&self, w: &mut Wavefunction, steps: usize) -> Result<(), String> {
        for _ in 0..steps {
            self.step(w)?;
        }
        Ok(())
    }
}

/// A propagator for a **time-dependent** Hamiltonian
/// `H(t) = H_0 + f(t) g(x)`.
///
/// # Why the potential is factorised
///
/// A fully general `V(x, t)` would have to be re-sampled at every grid
/// point on every step. When the potential comes from a user-supplied
/// function that is thousands of evaluations per step, and it dominates
/// the cost of the solve it is feeding.
///
/// Factorising into a fixed spatial *shape* `g(x)` and a scalar
/// *modulation* `f(t)` costs one evaluation per step instead, and covers
/// the physically important cases exactly: a dipole drive `f(t) x`, a
/// shaken trap, a pulse envelope, an adiabatic ramp. A drive whose
/// spatial profile genuinely changes shape with time is not expressible
/// this way, and that limit is stated rather than hidden.
///
/// # Accuracy and unitarity
///
/// The modulation is evaluated at the **midpoint** `t + dt/2`, which
/// keeps the scheme second order in `dt`; sampling at the start of the
/// step would silently drop it to first order.
///
/// `H(t)` is Hermitian at every instant, so each step is still an exact
/// Cayley transform and the propagator remains **unitary** — the norm is
/// conserved to machine precision even though the energy is not
/// conserved at all. That is the physics: a driven system exchanges
/// energy with whatever drives it.
pub struct DrivenPropagator {
    ham: Hamiltonian,
    shape: Vec<f64>,
    dt: f64,
    time: f64,
}

impl DrivenPropagator {
    /// `shape` is `g(x)` sampled on the grid.
    ///
    /// # Errors
    /// A length mismatch, a non-finite shape value, or a zero `dt`.
    pub fn new(ham: Hamiltonian, shape: Vec<f64>, dt: f64) -> Result<Self, String> {
        if shape.len() != ham.grid.n {
            return Err(format!(
                "DrivenPropagator: the drive shape has {} values but the grid has {}",
                shape.len(),
                ham.grid.n
            ));
        }
        if shape.iter().any(|v| !v.is_finite()) {
            return Err("DrivenPropagator: the drive shape has a non-finite value".to_string());
        }
        if !dt.is_finite() || dt == 0.0 {
            return Err(format!("DrivenPropagator: dt must be finite and non-zero, got {dt}"));
        }
        Ok(Self { ham, shape, dt, time: 0.0 })
    }

    pub fn time(&self) -> f64 {
        self.time
    }
    pub fn dt(&self) -> f64 {
        self.dt
    }
    /// The instantaneous Hamiltonian at the current time, for observables.
    ///
    /// # Errors
    /// As [`Hamiltonian::new`].
    pub fn hamiltonian_now<F: Fn(f64) -> f64>(&self, modulation: F) -> Result<Hamiltonian, String> {
        self.hamiltonian_at(modulation(self.time))
    }

    fn hamiltonian_at(&self, amp: f64) -> Result<Hamiltonian, String> {
        if !amp.is_finite() {
            return Err(format!("DrivenPropagator: the modulation returned {amp}"));
        }
        let v: Vec<f64> = self
            .ham
            .potential
            .iter()
            .zip(&self.shape)
            .map(|(v0, g)| v0 + amp * g)
            .collect();
        let mut h = Hamiltonian::new(
            self.ham.grid.clone(),
            v,
            self.ham.mass,
            self.ham.hbar,
        )?;
        h.absorber = self.ham.absorber.clone();
        Ok(h)
    }

    /// One step, with `modulation` giving `f(t)`.
    ///
    /// # Errors
    /// A grid mismatch, a non-finite modulation, or a solve failure.
    pub fn step<F: Fn(f64) -> f64>(
        &mut self,
        w: &mut Wavefunction,
        modulation: F,
    ) -> Result<(), String> {
        if w.grid != self.ham.grid {
            return Err("step: the wavefunction and propagator use different grids".to_string());
        }
        // MIDPOINT: sampling at t would drop the scheme to first order.
        let amp = modulation(self.time + 0.5 * self.dt);
        let h = self.hamiltonian_at(amp)?;
        Propagator::new(h, self.dt)?.step(w)?;
        self.time += self.dt;
        Ok(())
    }

    /// `steps` steps.
    ///
    /// # Errors
    /// As [`DrivenPropagator::step`].
    pub fn run<F: Fn(f64) -> f64 + Copy>(
        &mut self,
        w: &mut Wavefunction,
        steps: usize,
        modulation: F,
    ) -> Result<(), String> {
        for _ in 0..steps {
            self.step(w, modulation)?;
        }
        Ok(())
    }
}

/// Analytic transmission through a rectangular barrier of height `v0`
/// and width `a`, for a particle of energy `e`.
///
/// Provided so a simulation can be checked against theory rather than
/// against itself. Units `hbar = m = 1`.
pub fn barrier_transmission(e: f64, v0: f64, a: f64) -> f64 {
    if e <= 0.0 {
        return 0.0;
    }
    if (e - v0).abs() < 1e-12 {
        return 1.0 / (1.0 + v0 * a * a / 2.0);
    }
    if e < v0 {
        let kappa = (2.0 * (v0 - e)).sqrt();
        let s = (kappa * a).sinh();
        1.0 / (1.0 + v0 * v0 * s * s / (4.0 * e * (v0 - e)))
    } else {
        let kp = (2.0 * (e - v0)).sqrt();
        let s = (kp * a).sin();
        1.0 / (1.0 + v0 * v0 * s * s / (4.0 * e * (e - v0)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn free_grid(n: usize) -> Grid {
        Grid::new(0.0, 1.0, n).unwrap()
    }

    /// The infinite square well. On a grid of `n` points the FD
    /// eigenvalues have the CLOSED FORM
    /// `(2/h^2) (1 - cos(k pi h / L))`, so this checks the
    /// discretisation exactly rather than approximately — a much
    /// stronger statement than "close to n^2 pi^2 / 2".
    #[test]
    fn infinite_well_matches_the_exact_discrete_spectrum() {
        let n = 60;
        let g = free_grid(n);
        let h = g.h();
        let ham = Hamiltonian::from_fn(g, |_| 0.0, 1.0, 1.0).unwrap();
        let (e, _) = ham.bound_states(6).unwrap();
        for (k, &ek) in e.iter().enumerate() {
            let kk = (k + 1) as f64;
            let exact_discrete = (1.0 - (kk * PI * h).cos()) / (h * h);
            assert!(
                (ek - exact_discrete).abs() < 1e-10,
                "k={kk}: {ek} vs exact discrete {exact_discrete}"
            );
        }
    }

    /// ...and the discrete spectrum converges to the continuum one at
    /// second order: halving h must cut the error by ~4.
    #[test]
    fn infinite_well_converges_second_order() {
        let mut prev = 0.0_f64;
        for &n in &[25usize, 51, 103] {
            let ham = Hamiltonian::from_fn(free_grid(n), |_| 0.0, 1.0, 1.0).unwrap();
            let e0 = ham.bound_states(1).unwrap().0[0];
            let exact = PI * PI / 2.0;
            let err = (e0 - exact).abs();
            if prev > 0.0 {
                let drop = prev / err;
                assert!((3.5..4.6).contains(&drop), "error fell {drop}x, expected ~4");
            }
            prev = err;
        }
    }

    /// The harmonic oscillator: E_n = n + 1/2 in natural units.
    #[test]
    fn harmonic_oscillator_spectrum() {
        let g = Grid::new(-8.0, 8.0, 300).unwrap();
        let ham = Hamiltonian::from_fn(g, |x| 0.5 * x * x, 1.0, 1.0).unwrap();
        let (e, _) = ham.bound_states(5).unwrap();
        for (n, &en) in e.iter().enumerate() {
            let exact = n as f64 + 0.5;
            assert!((en - exact).abs() < 5e-3, "n={n}: {en} vs {exact}");
        }
    }

    /// Eigenfunctions must be orthonormal under the grid inner product.
    /// This is where the 1/sqrt(h) normalisation gets checked: without
    /// it every one of these would be off by a factor of h.
    #[test]
    fn bound_states_are_orthonormal() {
        let g = Grid::new(-8.0, 8.0, 200).unwrap();
        let h = g.h();
        let ham = Hamiltonian::from_fn(g, |x| 0.5 * x * x, 1.0, 1.0).unwrap();
        let (_, v) = ham.bound_states(5).unwrap();
        for i in 0..5 {
            for j in 0..5 {
                let ip: f64 = v[i].iter().zip(&v[j]).map(|(a, b)| a * b).sum::<f64>() * h;
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((ip - want).abs() < 1e-9, "<{i}|{j}> = {ip}, want {want}");
            }
        }
    }

    /// A stationary state is STATIONARY: propagating an eigenstate must
    /// leave |psi|^2 unchanged, however long you run it. This couples
    /// the eigensolver and the propagator, so it fails if either is
    /// wrong or if they disagree about the Hamiltonian.
    #[test]
    fn an_eigenstate_does_not_move() {
        let g = Grid::new(-8.0, 8.0, 120).unwrap();
        let ham = Hamiltonian::from_fn(g.clone(), |x| 0.5 * x * x, 1.0, 1.0).unwrap();
        let (energies, states) = ham.bound_states(3).unwrap();
        let mut w = Wavefunction::from_real(g, &states[1]).unwrap();
        let before = w.density();
        let e_before = w.energy(&ham);
        let prop = Propagator::new(ham.clone(), 0.01).unwrap();
        prop.run(&mut w, 200).unwrap();
        let after = w.density();
        let worst = before
            .iter()
            .zip(&after)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(worst < 1e-9, "density moved by {worst}");
        // and its energy is the eigenvalue, before and after
        assert!((e_before - energies[1]).abs() < 1e-9);
        assert!((w.energy(&ham) - energies[1]).abs() < 1e-9);
    }

    /// Unitarity: the Cayley operator conserves the norm for ANY dt, so
    /// a deliberately huge step must still conserve it. If this ever
    /// depended on dt, the solver would be wrong.
    #[test]
    fn propagation_is_unitary_even_at_absurd_step_sizes() {
        for &dt in &[0.001, 0.05, 1.0, 25.0] {
            let g = Grid::new(-10.0, 10.0, 200).unwrap();
            let ham = Hamiltonian::from_fn(g.clone(), |x| 0.5 * x * x, 1.0, 1.0).unwrap();
            let mut w = Wavefunction::gaussian(g, -2.0, 1.0, 1.5).unwrap();
            let n0 = w.norm();
            let prop = Propagator::new(ham, dt).unwrap();
            prop.run(&mut w, 50).unwrap();
            let n1 = w.norm();
            assert!(
                (n1 / n0 - 1.0).abs() < 1e-10,
                "dt={dt}: norm {n0} -> {n1}"
            );
        }
    }

    /// A free packet travels at the group velocity — but the group
    /// velocity of the **discrete** dispersion relation.
    ///
    /// The three-point stencil gives `E(k) = (1 - cos(k h)) / (m h^2)`,
    /// not `hbar^2 k^2 / 2m`. So the central-difference momentum
    /// operator returns `hbar sin(k h)/h` and the packet moves at
    /// `dE/dk = hbar sin(k h)/(m h)`. Both reduce to the continuum
    /// values as `h -> 0` (the next test pins that), but asserting the
    /// continuum value on a finite grid would be asserting a number the
    /// discretisation is not trying to produce. An earlier version of
    /// this test did exactly that and failed by 0.0139 — which is
    /// `k0 (k0 h)^2 / 6` to two figures, i.e. precisely the stencil's
    /// own error.
    #[test]
    fn free_packet_moves_at_the_discrete_group_velocity() {
        let g = Grid::new(-60.0, 60.0, 1200).unwrap();
        let h = g.h();
        let ham = Hamiltonian::from_fn(g.clone(), |_| 0.0, 1.0, 1.0).unwrap();
        let k0 = 2.0;
        let mut w = Wavefunction::gaussian(g, -20.0, 2.0, k0).unwrap();

        let p_discrete = (k0 * h).sin() / h;
        let p0 = w.momentum(1.0);
        assert!(
            (p0 - p_discrete).abs() < 2e-3,
            "<p> = {p0}, discrete operator gives {p_discrete} (continuum {k0})"
        );

        let x0 = w.position();
        let dt = 0.005;
        let steps = 1000;
        let prop = Propagator::new(ham, dt).unwrap();
        prop.run(&mut w, steps).unwrap();
        let t = dt * steps as f64;
        let expected = x0 + p_discrete * t;
        assert!(
            (w.position() - expected).abs() < 0.05,
            "<x> = {}, want {expected}",
            w.position()
        );
        assert!((w.momentum(1.0) - p0).abs() < 1e-3, "momentum not conserved");
        assert!(w.edge_probability(0.05) < 1e-12, "the packet reached a wall");
    }

    /// ...and that discrete momentum converges to the continuum value at
    /// second order: the error is `k0 (k0 h)^2 / 6`, so halving `h` must
    /// cut it by ~4. This is the statement the previous test would have
    /// been making if a finite grid gave continuum answers.
    #[test]
    fn the_momentum_operator_converges_second_order() {
        let k0 = 1.0_f64;
        let mut prev = 0.0_f64;
        for &n in &[300usize, 600, 1200] {
            let g = Grid::new(-60.0, 60.0, n).unwrap();
            let w = Wavefunction::gaussian(g, 0.0, 4.0, k0).unwrap();
            let err = (w.momentum(1.0) - k0).abs();
            if prev > 0.0 {
                let drop = prev / err;
                assert!(
                    (3.5..4.6).contains(&drop),
                    "n={n}: error fell {drop}x, expected ~4"
                );
            }
            prev = err;
        }
    }

    /// Barrier scattering against the analytic coefficient, averaged
    /// over the packet's own momentum distribution — the packet has an
    /// energy spread, so comparing to T at the central energy alone
    /// would be comparing against the wrong number.
    #[test]
    fn barrier_transmission_matches_theory() {
        let g = Grid::new(-100.0, 100.0, 3999).unwrap();
        let (v0, a, k0, sigma) = (2.5, 1.0, 2.0, 2.0);
        let ham =
            Hamiltonian::from_fn(g.clone(), |x| if (0.0..a).contains(&x) { v0 } else { 0.0 }, 1.0, 1.0)
                .unwrap();
        let mut w = Wavefunction::gaussian(g, -25.0, sigma, k0).unwrap();
        let prop = Propagator::new(ham, 0.005).unwrap();
        prop.run(&mut w, 6000).unwrap();

        let t_sim = w.probability_in(a, 100.0);
        let r_sim = w.probability_in(-100.0, 0.0);
        assert!((t_sim + r_sim - 1.0).abs() < 1e-5, "T+R = {}", t_sim + r_sim);
        assert!(w.edge_probability(0.05) < 1e-9, "the packet reached a wall");

        // momentum-average the analytic coefficient: |phi(k)|^2 ~
        // exp(-2 sigma^2 (k-k0)^2)
        let wgt = |k: f64| (-2.0 * sigma * sigma * (k - k0) * (k - k0)).exp();
        let (lo, hi) = (k0 - 3.0 / sigma, k0 + 3.0 / sigma);
        let m = 2000;
        let dk = (hi - lo) / m as f64;
        let mut num = 0.0;
        let mut den = 0.0;
        for i in 0..=m {
            let k = lo + i as f64 * dk;
            let ww = wgt(k);
            num += ww * barrier_transmission(k * k / 2.0, v0, a);
            den += ww;
        }
        let t_theory = num / den;
        let rel = (t_sim - t_theory).abs() / t_theory;
        assert!(rel < 0.05, "T sim {t_sim} vs theory {t_theory} ({:.2}%)", 100.0 * rel);
    }

    /// A deeper potential binds more states below its rim — a
    /// qualitative check that the solver responds to the potential at
    /// all, which no amount of tolerance-tuning can fake.
    #[test]
    fn a_deeper_well_binds_more_states() {
        let mut counts = Vec::new();
        for &depth in &[2.0_f64, 8.0, 20.0] {
            let g = Grid::new(-10.0, 10.0, 250).unwrap();
            let ham = Hamiltonian::from_fn(
                g,
                |x| if x.abs() < 2.0 { -depth } else { 0.0 },
                1.0,
                1.0,
            )
            .unwrap();
            let (e, _) = ham.bound_states(20).unwrap();
            counts.push(e.iter().filter(|&&v| v < 0.0).count());
        }
        assert!(
            counts[0] < counts[1] && counts[1] < counts[2],
            "bound-state counts did not increase with depth: {counts:?}"
        );
    }

    /// The absorber must ABSORB: a packet driven into it should mostly
    /// vanish rather than come back.
    #[test]
    fn an_absorber_removes_a_packet_that_a_wall_would_reflect() {
        let g = Grid::new(-40.0, 40.0, 1200).unwrap();
        let build = |absorb: bool| {
            let h = Hamiltonian::from_fn(g.clone(), |_| 0.0, 1.0, 1.0).unwrap();
            // width 14, strength 3.0, quadratic ramp. These are not
            // guesses: `examples/absorber_tuning.rs` sweeps the plane
            // and finds the optimum near here for k0 = 3, measuring
            // ~2e-8 reflection. A first attempt at strength 1.0 gave
            // 5e-3 — a factor of 250_000 worse — which is why the
            // parameters are measured rather than assumed.
            if absorb { h.with_absorber(14.0, 3.0, 2.0).unwrap() } else { h }
        };
        // Same packet, same time, with and without the absorber.
        let run = |absorb: bool| {
            let ham = build(absorb);
            let mut w = Wavefunction::gaussian(g.clone(), 0.0, 1.5, 3.0).unwrap();
            let prop = Propagator::new(ham, 0.005).unwrap();
            prop.run(&mut w, 4000).unwrap(); // t = 20, far past the wall
            w
        };
        let plain = run(false);
        let capped = run(true);

        // Without the absorber the packet is still all there, having
        // bounced off the wall.
        assert!(
            (plain.norm() - 1.0).abs() < 1e-9,
            "a Dirichlet wall must conserve the norm, got {}",
            plain.norm()
        );
        // With it, almost everything has been drained away.
        assert!(capped.norm() < 0.01, "absorber left {} of the norm", capped.norm());
        // And what little remains must not be sitting in the interior:
        // that would mean the ABSORBER reflected, which is the failure
        // mode that matters.
        let interior = capped.probability_in(-24.0, 24.0);
        assert!(interior < 1e-5, "absorber reflected {interior} back into the interior");
    }

    /// The real test of an absorber is that it does not change the
    /// PHYSICS: a scattering answer computed on a short absorbing domain
    /// must match the same answer computed on a long reflection-free
    /// one. That is the whole point — it buys domain size, not a
    /// different result.
    #[test]
    fn absorbing_short_domain_reproduces_the_long_domain_answer() {
        let (v0, a, k0, sigma) = (2.5_f64, 1.0_f64, 2.0_f64, 2.0_f64);
        let pot = move |x: f64| if (0.0..a).contains(&x) { v0 } else { 0.0 };

        // Reference: a long domain, no absorber, stopped before the
        // packets reach the walls.
        let g_long = Grid::new(-100.0, 100.0, 3000).unwrap();
        let ham_long = Hamiltonian::from_fn(g_long.clone(), pot, 1.0, 1.0).unwrap();
        let mut w_long = Wavefunction::gaussian(g_long, -25.0, sigma, k0).unwrap();
        Propagator::new(ham_long, 0.01).unwrap().run(&mut w_long, 2000).unwrap();
        let t_ref = w_long.probability_in(a, 100.0);
        assert!(w_long.edge_probability(0.05) < 1e-9, "reference touched a wall");

        // Short domain with absorbers. The transmitted packet WILL run
        // into the right-hand absorber, so transmission is measured as
        // "probability that left through the right", i.e. what is
        // missing from the interior plus what is still travelling right.
        let g_short = Grid::new(-45.0, 45.0, 1350).unwrap();
        let ham_short = Hamiltonian::from_fn(g_short.clone(), pot, 1.0, 1.0)
            .unwrap()
            .with_absorber(15.0, 0.8, 2.0)
            .unwrap();
        let mut w_short = Wavefunction::gaussian(g_short, -25.0, sigma, k0).unwrap();
        let prop = Propagator::new(ham_short, 0.01).unwrap();
        // Stop while both packets are still inside the clear region, so
        // this compares like with like.
        prop.run(&mut w_short, 2000).unwrap();
        let t_cap = w_short.probability_in(a, 30.0);

        let rel = (t_cap - t_ref).abs() / t_ref;
        assert!(
            rel < 0.02,
            "absorbing domain gave T = {t_cap}, reference {t_ref} ({:.2}% off)",
            100.0 * rel
        );
    }

    /// **The driven harmonic oscillator.** For a quadratic potential
    /// with a linear drive, Ehrenfest's theorem is EXACT: `<x>` obeys
    /// the classical equation of motion with no approximation at all.
    /// So the quantum centroid must trace the classical solution
    ///
    /// ```text
    ///   x''(t) = -x - F0 cos(w t),  x(0) = x'(0) = 0
    ///   =>  x(t) = -F0/(1 - w^2) [cos(w t) - cos t]
    /// ```
    ///
    /// which is an analytic prediction with no fitted constants.
    #[test]
    fn a_driven_oscillator_follows_the_classical_trajectory() {
        let g = Grid::new(-12.0, 12.0, 480).unwrap();
        let ham = Hamiltonian::from_fn(g.clone(), |x| 0.5 * x * x, 1.0, 1.0).unwrap();
        // start in the ground state: centroid 0, momentum 0
        let (_, states) = ham.bound_states(1).unwrap();
        let mut w = Wavefunction::from_real(g.clone(), &states[0]).unwrap();
        w.normalise().unwrap();

        let (f0, om) = (0.3_f64, 0.7_f64);
        let shape: Vec<f64> = (0..g.n).map(|i| g.x(i)).collect(); // g(x) = x
        let dt = 0.005;
        let mut prop = DrivenPropagator::new(ham.clone(), shape, dt).unwrap();
        let drive = move |t: f64| f0 * (om * t).cos();

        let exact = |t: f64| -f0 / (1.0 - om * om) * ((om * t).cos() - t.cos());

        let mut worst = 0.0_f64;
        for _ in 0..20 {
            prop.run(&mut w, 100, drive).unwrap();
            let t = prop.time();
            let err = (w.position() - exact(t)).abs();
            worst = worst.max(err);
        }
        assert!(worst < 0.01, "worst |<x> - classical| = {worst}");
        // ...and the drive really did something: the excursion is large
        // compared with the error above.
        assert!(exact(10.0).abs() > 0.3, "the drive was too weak to be a test");
        // unitary throughout, even though H depends on time
        assert!((w.norm() - 1.0).abs() < 1e-10, "norm = {}", w.norm());
    }

    /// The energy of a driven system is NOT conserved — it exchanges
    /// energy with the drive. Asserting that it changes is as important
    /// as asserting that an undriven one does not.
    #[test]
    fn a_driven_system_exchanges_energy_while_staying_unitary() {
        let g = Grid::new(-12.0, 12.0, 300).unwrap();
        let ham = Hamiltonian::from_fn(g.clone(), |x| 0.5 * x * x, 1.0, 1.0).unwrap();
        let (e0, states) = ham.bound_states(1).unwrap();
        let mut w = Wavefunction::from_real(g.clone(), &states[0]).unwrap();
        w.normalise().unwrap();
        assert!((w.energy(&ham) - e0[0]).abs() < 1e-9);

        let shape: Vec<f64> = (0..g.n).map(|i| g.x(i)).collect();
        let mut prop = DrivenPropagator::new(ham.clone(), shape, 0.005).unwrap();
        // resonant drive: energy should climb steadily
        prop.run(&mut w, 2000, |t| 0.3 * t.cos()).unwrap();

        let e_now = w.energy(&ham); // energy of the STATIC hamiltonian
        assert!(
            e_now > e0[0] + 0.2,
            "a resonant drive should pump energy in: {} -> {e_now}",
            e0[0]
        );
        assert!((w.norm() - 1.0).abs() < 1e-10, "but the norm must be conserved");
    }

    /// A constant modulation is just a static shifted potential, so the
    /// driven propagator must agree with the plain one exactly.
    #[test]
    fn a_constant_drive_matches_the_static_propagator() {
        let g = Grid::new(-10.0, 10.0, 200).unwrap();
        let base = Hamiltonian::from_fn(g.clone(), |x| 0.5 * x * x, 1.0, 1.0).unwrap();
        let shape: Vec<f64> = (0..g.n).map(|i| g.x(i)).collect();
        let amp = 0.4_f64;

        let mut w1 = Wavefunction::gaussian(g.clone(), -1.0, 1.0, 0.5).unwrap();
        let mut prop = DrivenPropagator::new(base.clone(), shape, 0.01).unwrap();
        prop.run(&mut w1, 300, move |_| amp).unwrap();

        // the same thing built statically
        let shifted =
            Hamiltonian::from_fn(g.clone(), move |x| 0.5 * x * x + amp * x, 1.0, 1.0).unwrap();
        let mut w2 = Wavefunction::gaussian(g, -1.0, 1.0, 0.5).unwrap();
        Propagator::new(shifted, 0.01).unwrap().run(&mut w2, 300).unwrap();

        let worst = w1
            .psi
            .iter()
            .zip(&w2.psi)
            .map(|(a, b)| (*a - *b).abs())
            .fold(0.0_f64, f64::max);
        assert!(worst < 1e-12, "driven and static differ by {worst}");
    }

    /// Midpoint sampling keeps the scheme second order. Halving dt must
    /// cut the error against the analytic trajectory by about four —
    /// sampling at the start of the step instead would give only two.
    #[test]
    fn midpoint_sampling_is_second_order() {
        let g = Grid::new(-12.0, 12.0, 300).unwrap();
        let ham = Hamiltonian::from_fn(g.clone(), |x| 0.5 * x * x, 1.0, 1.0).unwrap();
        let (_, states) = ham.bound_states(1).unwrap();
        let shape: Vec<f64> = (0..g.n).map(|i| g.x(i)).collect();
        let (f0, om) = (0.5_f64, 0.6_f64);
        let drive = move |t: f64| f0 * (om * t).cos();
        let exact = |t: f64| -f0 / (1.0 - om * om) * ((om * t).cos() - t.cos());
        let total = 4.0_f64;

        let pos_at = |dt: f64| -> f64 {
            let mut w = Wavefunction::from_real(g.clone(), &states[0]).unwrap();
            w.normalise().unwrap();
            let mut p = DrivenPropagator::new(ham.clone(), shape.clone(), dt).unwrap();
            p.run(&mut w, (total / dt).round() as usize, drive).unwrap();
            w.position()
        };
        // Compare against a fine-dt run on the SAME GRID, not against the
        // analytic trajectory. The analytic comparison carries a
        // dt-independent spatial-discretisation floor: measured against
        // it the ratio came out 2.26, and solving err = c + a dt^2 from
        // the two points gives c = 2.3e-3 with the dt-dependent part
        // scaling by exactly 4. The scheme was second order all along;
        // the measurement was contaminated.
        let reference = pos_at(0.0025);
        let coarse = (pos_at(0.08) - reference).abs();
        let fine = (pos_at(0.04) - reference).abs();
        let ratio = coarse / fine;
        // sanity: the analytic solution is still what this converges TO
        assert!(
            (reference - exact(total)).abs() < 5e-3,
            "reference {reference} vs analytic {}",
            exact(total)
        );
        assert!(
            (3.0..5.5).contains(&ratio),
            "error fell {ratio}x when dt halved, expected ~4 (coarse {coarse:.3e}, fine {fine:.3e})"
        );
    }

    #[test]
    fn driven_propagator_validates_its_input() {
        let g = Grid::new(-5.0, 5.0, 50).unwrap();
        let h = Hamiltonian::from_fn(g.clone(), |_| 0.0, 1.0, 1.0).unwrap();
        assert!(DrivenPropagator::new(h.clone(), vec![0.0; 3], 0.01).is_err(), "shape length");
        assert!(
            DrivenPropagator::new(h.clone(), vec![f64::NAN; 50], 0.01).is_err(),
            "non-finite shape"
        );
        assert!(DrivenPropagator::new(h.clone(), vec![0.0; 50], 0.0).is_err(), "dt = 0");
        // a modulation that returns NaN must be reported, not propagated
        let mut p = DrivenPropagator::new(h, vec![1.0; 50], 0.01).unwrap();
        let mut w = Wavefunction::gaussian(g, 0.0, 1.0, 0.0).unwrap();
        assert!(p.step(&mut w, |_| f64::NAN).is_err(), "NaN modulation");
    }

    /// `<E>` must be the true expectation value, so an absorber that
    /// removes half the packet must not appear to remove half its
    /// energy.
    ///
    /// Measured at PARTIAL absorption on purpose. Once almost
    /// everything is gone the remnant is the low-energy tail — a CAP
    /// absorbs high-k components better (see
    /// `examples/absorber_tuning.rs`), so `<E>` genuinely does fall
    /// then, and that is physics rather than a normalisation bug.
    #[test]
    fn energy_is_normalised_so_absorption_does_not_fake_a_drop() {
        let g = Grid::new(-40.0, 40.0, 1200).unwrap();
        let ham = Hamiltonian::from_fn(g.clone(), |_| 0.0, 1.0, 1.0)
            .unwrap()
            .with_absorber(14.0, 3.0, 2.0)
            .unwrap();
        let k0 = 3.0_f64;
        let mut w = Wavefunction::gaussian(g, 0.0, 1.5, k0).unwrap();
        let e0 = w.energy(&ham);
        assert!(
            (e0 - k0 * k0 / 2.0).abs() / (k0 * k0 / 2.0) < 0.05,
            "E0 = {e0}, expected near {}",
            k0 * k0 / 2.0
        );

        // step until roughly half the packet is gone
        let prop = Propagator::new(ham.clone(), 0.005).unwrap();
        let mut guard = 0;
        while w.norm() > 0.5 && guard < 20 {
            prop.run(&mut w, 200).unwrap();
            guard += 1;
        }
        let n = w.norm();
        assert!((0.2..0.75).contains(&n), "wanted partial absorption, norm = {n}");

        let e1 = w.energy(&ham);
        // The claim the division makes is RELATIVE: energy must be
        // retained far better than norm. Measured here, norm falls to
        // 0.42 while E keeps 0.84 of its value — the residual 16 % is
        // the CAP preferentially eating high-k components, which is
        // physics. Without the division E would have tracked the norm
        // exactly.
        let keep = e1 / e0;
        assert!(keep > 0.75, "E went {e0} -> {e1} (kept {keep}) at norm {n}");
        assert!(
            keep > 1.5 * n,
            "E retention {keep} should far exceed norm retention {n}"
        );
        // ...and the point of the division: the RAW integral did fall
        // with the norm, so an undivided energy would have looked wrong.
        let raw = e1 * n;
        assert!(
            raw < 0.8 * e0,
            "the raw integral {raw} should track the norm, not the energy"
        );
    }

    /// An absorbing Hamiltonian is not Hermitian, so the symmetric
    /// eigensolver must refuse rather than return plausible nonsense.
    #[test]
    fn bound_states_refuse_an_absorbing_hamiltonian() {
        let g = Grid::new(-10.0, 10.0, 100).unwrap();
        let ham = Hamiltonian::from_fn(g, |x| 0.5 * x * x, 1.0, 1.0)
            .unwrap()
            .with_absorber(2.0, 1.0, 2.0)
            .unwrap();
        let e = ham.bound_states(3).unwrap_err();
        assert!(e.contains("non-Hermitian") || e.contains("NON-Hermitian"), "got: {e}");
        // and removing it makes them available again
        assert!(ham.without_absorber().bound_states(3).is_ok());
    }

    /// Absorber construction rejects nonsense.
    #[test]
    fn absorber_parameters_are_validated() {
        let g = Grid::new(-10.0, 10.0, 100).unwrap();
        let h = || Hamiltonian::from_fn(g.clone(), |_| 0.0, 1.0, 1.0).unwrap();
        assert!(h().with_absorber(0.0, 1.0, 2.0).is_err(), "zero width");
        assert!(h().with_absorber(2.0, -1.0, 2.0).is_err(), "negative strength");
        assert!(h().with_absorber(2.0, 1.0, 0.5).is_err(), "power < 1");
        assert!(h().with_absorber(11.0, 1.0, 2.0).is_err(), "absorbers overlap");
        assert!(h().with_absorber(2.0, 1.0, 2.0).is_ok());
    }

    #[test]
    fn invalid_input_is_reported() {
        assert!(Grid::new(0.0, 1.0, 0).is_err(), "n = 0");
        assert!(Grid::new(1.0, 0.0, 10).is_err(), "reversed bounds");
        assert!(Grid::new(f64::NAN, 1.0, 10).is_err());
        let g = free_grid(5);
        assert!(Hamiltonian::new(g.clone(), vec![0.0; 3], 1.0, 1.0).is_err(), "length");
        assert!(Hamiltonian::new(g.clone(), vec![0.0; 5], -1.0, 1.0).is_err(), "mass");
        assert!(Hamiltonian::new(g.clone(), vec![0.0; 5], 1.0, 0.0).is_err(), "hbar");
        assert!(
            Hamiltonian::new(g.clone(), vec![f64::INFINITY; 5], 1.0, 1.0).is_err(),
            "infinite V"
        );
        let ham = Hamiltonian::from_fn(g.clone(), |_| 0.0, 1.0, 1.0).unwrap();
        assert!(ham.bound_states(0).is_err(), "k = 0");
        assert!(ham.bound_states(99).is_err(), "k > n");
        assert!(Propagator::new(ham, 0.0).is_err(), "dt = 0");
        assert!(Wavefunction::gaussian(g, 0.0, -1.0, 0.0).is_err(), "sigma < 0");
    }

    /// A propagator and a wavefunction built on different grids must be
    /// refused rather than silently producing nonsense.
    #[test]
    fn mismatched_grids_are_refused() {
        let g1 = Grid::new(0.0, 1.0, 10).unwrap();
        let g2 = Grid::new(0.0, 1.0, 12).unwrap();
        let ham = Hamiltonian::from_fn(g1, |_| 0.0, 1.0, 1.0).unwrap();
        let prop = Propagator::new(ham, 0.01).unwrap();
        let mut w = Wavefunction::gaussian(g2, 0.5, 0.1, 0.0).unwrap();
        assert!(prop.step(&mut w).is_err());
    }
}
