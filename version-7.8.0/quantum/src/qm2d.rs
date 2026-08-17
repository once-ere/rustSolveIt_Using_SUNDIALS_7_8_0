//! Two-dimensional quantum mechanics by ADI (alternating direction
//! implicit) propagation.
//!
//! # Why 2-D needs a different method
//!
//! In 1-D the Crank–Nicolson operator is tridiagonal, so one solve per
//! step gives an exactly unitary propagator. In 2-D it is **not**: the
//! five-point Laplacian couples each point to its neighbours in both
//! directions, and the matrix is block-tridiagonal with bandwidth `nx`.
//! Solving that directly costs `O(nx^3 ny)` per step, which is hopeless.
//!
//! ADI splits the step by direction. Each half-step is implicit in one
//! direction and explicit in the other, so it reduces to a *set of
//! independent tridiagonal solves* — `ny` of them along x, then `nx`
//! along y — and the cost falls to `O(nx ny)` per step, the same order
//! as 1-D.
//!
//! # Why Strang splitting rather than Peaceman–Rachford
//!
//! The textbook ADI scheme is Peaceman–Rachford, which is implicit in x
//! against an explicit y and then swaps. It is second-order accurate and
//! unconditionally stable — but it is **not unitary**, because the two
//! half-steps use *different* operators, so the norm drifts at
//! `O(dt^2)` per step.
//!
//! This module instead applies a **Cayley transform in each direction
//! separately**, composed by Strang splitting:
//!
//! ```text
//!     psi^{n+1} = U_x(dt/2) U_y(dt) U_x(dt/2) psi^n
//!     U_d(tau)  = (1 + i tau A_d / 2 hbar)^{-1} (1 - i tau A_d / 2 hbar)
//! ```
//!
//! Each `A_d` is Hermitian, so each `U_d` is **exactly unitary**, and a
//! product of unitaries is unitary. The norm is therefore conserved to
//! machine precision for *any* time step — exactly as in 1-D — while the
//! splitting error remains `O(dt^2)` in the *dynamics*.
//!
//! That distinction matters for testing. Norm conservation stays a sharp
//! check on the linear algebra, entirely separate from the accuracy
//! question, instead of the two being tangled together in one drifting
//! number.
//!
//! The potential is split evenly, `A_x = T_x + V/2` and
//! `A_y = T_y + V/2`, so each direction carries half of it.

use special_functions::complex::Complex64 as C;
use special_functions::lanczos::{lanczos_lowest, Stop};
use special_functions::tridiag::solve_tridiag_c;

/// A uniform 2-D grid of interior points, with Dirichlet walls all
/// round. Points are stored row-major: index `iy * nx + ix`.
#[derive(Clone, Debug, PartialEq)]
pub struct Grid2 {
    pub x_min: f64,
    pub x_max: f64,
    pub nx: usize,
    pub y_min: f64,
    pub y_max: f64,
    pub ny: usize,
}

impl Grid2 {
    /// # Errors
    /// A zero count, a non-finite bound, or a reversed range.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        x_min: f64,
        x_max: f64,
        nx: usize,
        y_min: f64,
        y_max: f64,
        ny: usize,
    ) -> Result<Self, String> {
        if nx == 0 || ny == 0 {
            return Err(format!("Grid2: needs at least one point per axis, got {nx} x {ny}"));
        }
        for (lo, hi, ax) in [(x_min, x_max, "x"), (y_min, y_max, "y")] {
            if !lo.is_finite() || !hi.is_finite() {
                return Err(format!("Grid2: {ax} bounds must be finite, got [{lo}, {hi}]"));
            }
            if hi <= lo {
                return Err(format!("Grid2: {ax}_max ({hi}) must exceed {ax}_min ({lo})"));
            }
        }
        Ok(Self { x_min, x_max, nx, y_min, y_max, ny })
    }

    pub fn hx(&self) -> f64 {
        (self.x_max - self.x_min) / (self.nx + 1) as f64
    }
    pub fn hy(&self) -> f64 {
        (self.y_max - self.y_min) / (self.ny + 1) as f64
    }
    pub fn x(&self, ix: usize) -> f64 {
        self.x_min + (ix + 1) as f64 * self.hx()
    }
    pub fn y(&self, iy: usize) -> f64 {
        self.y_min + (iy + 1) as f64 * self.hy()
    }
    /// Row-major flat index.
    pub fn idx(&self, ix: usize, iy: usize) -> usize {
        iy * self.nx + ix
    }
    /// Total interior points.
    pub fn len(&self) -> usize {
        self.nx * self.ny
    }
    /// Always false for a grid built via `new` (every axis has n >= 1).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Cell area, for turning a sum into an integral.
    pub fn cell(&self) -> f64 {
        self.hx() * self.hy()
    }
}

/// A discretised 2-D Hamiltonian.
#[derive(Clone, Debug)]
pub struct Hamiltonian2 {
    pub grid: Grid2,
    /// `V` at every grid point, row-major.
    pub potential: Vec<f64>,
    pub mass: f64,
    pub hbar: f64,
    /// Optional complex absorbing potential `W >= 0`, as in 1-D.
    pub absorber: Option<Vec<f64>>,
}

impl Hamiltonian2 {
    /// # Errors
    /// Length mismatch, non-finite potential, non-positive mass/hbar.
    pub fn new(
        grid: Grid2,
        potential: Vec<f64>,
        mass: f64,
        hbar: f64,
    ) -> Result<Self, String> {
        if potential.len() != grid.len() {
            return Err(format!(
                "Hamiltonian2: potential has {} values but the grid has {}",
                potential.len(),
                grid.len()
            ));
        }
        if potential.iter().any(|v| !v.is_finite()) {
            return Err("Hamiltonian2: the potential has a non-finite value".to_string());
        }
        if !mass.is_finite() || mass <= 0.0 {
            return Err(format!("Hamiltonian2: mass must be finite and positive, got {mass}"));
        }
        if !hbar.is_finite() || hbar <= 0.0 {
            return Err(format!("Hamiltonian2: hbar must be finite and positive, got {hbar}"));
        }
        Ok(Self { grid, potential, mass, hbar, absorber: None })
    }

    /// Sample a closure of `(x, y)`.
    ///
    /// # Errors
    /// As [`Hamiltonian2::new`].
    pub fn from_fn<F: Fn(f64, f64) -> f64>(
        grid: Grid2,
        v: F,
        mass: f64,
        hbar: f64,
    ) -> Result<Self, String> {
        let mut p = Vec::with_capacity(grid.len());
        for iy in 0..grid.ny {
            for ix in 0..grid.nx {
                p.push(v(grid.x(ix), grid.y(iy)));
            }
        }
        Self::new(grid, p, mass, hbar)
    }

    /// Attach absorbing edges on all four sides — the 2-D counterpart of
    /// [`crate::qm1d::Hamiltonian::with_absorber`]. `W` is the larger of
    /// the two per-axis ramps, so a corner absorbs as strongly as an
    /// edge rather than twice as strongly.
    ///
    /// # Errors
    /// Non-positive `width`/`strength`, `power < 1`, or a width that
    /// does not fit.
    pub fn with_absorber(mut self, width: f64, strength: f64, power: f64) -> Result<Self, String> {
        if !width.is_finite() || width <= 0.0 || !strength.is_finite() || strength <= 0.0 {
            return Err("with_absorber: width and strength must be finite and positive".to_string());
        }
        if !power.is_finite() || power < 1.0 {
            return Err(format!("with_absorber: power must be at least 1, got {power}"));
        }
        let g = &self.grid;
        if 2.0 * width >= g.x_max - g.x_min || 2.0 * width >= g.y_max - g.y_min {
            return Err(format!(
                "with_absorber: width {width} does not fit in the domain on both axes"
            ));
        }
        let ramp = |c: f64, lo: f64, hi: f64| -> f64 {
            let d = if c < lo + width {
                (lo + width - c) / width
            } else if c > hi - width {
                (c - (hi - width)) / width
            } else {
                0.0
            };
            d.powf(power)
        };
        let mut w = Vec::with_capacity(g.len());
        for iy in 0..g.ny {
            for ix in 0..g.nx {
                let a = ramp(g.x(ix), g.x_min, g.x_max);
                let b = ramp(g.y(iy), g.y_min, g.y_max);
                w.push(strength * a.max(b));
            }
        }
        self.absorber = Some(w);
        Ok(self)
    }

    pub fn is_absorbing(&self) -> bool {
        self.absorber.is_some()
    }

    fn off_x(&self) -> f64 {
        let h = self.grid.hx();
        -self.hbar * self.hbar / (2.0 * self.mass * h * h)
    }
    fn off_y(&self) -> f64 {
        let h = self.grid.hy();
        -self.hbar * self.hbar / (2.0 * self.mass * h * h)
    }

    /// The full `H psi`, used for `<E>`. Five-point stencil.
    pub fn apply(&self, psi: &[C]) -> Vec<C> {
        let g = &self.grid;
        let (ox, oy) = (self.off_x(), self.off_y());
        let mut out = vec![C::ZERO; g.len()];
        for iy in 0..g.ny {
            for ix in 0..g.nx {
                let k = g.idx(ix, iy);
                let diag_re = -2.0 * ox - 2.0 * oy + self.potential[k];
                let d = match &self.absorber {
                    Some(w) => C::new(diag_re, -w[k]),
                    None => C::real(diag_re),
                };
                let mut s = d * psi[k];
                if ix > 0 {
                    s = s + psi[g.idx(ix - 1, iy)] * ox;
                }
                if ix + 1 < g.nx {
                    s = s + psi[g.idx(ix + 1, iy)] * ox;
                }
                if iy > 0 {
                    s = s + psi[g.idx(ix, iy - 1)] * oy;
                }
                if iy + 1 < g.ny {
                    s = s + psi[g.idx(ix, iy + 1)] * oy;
                }
                out[k] = s;
            }
        }
        out
    }

    /// `H psi` for a REAL vector — what the eigensolver needs.
    ///
    /// Bound states of a real Hamiltonian can be chosen real, so the
    /// eigenproblem is real symmetric and there is no reason to carry
    /// complex arithmetic through it.
    pub fn apply_real(&self, psi: &[f64]) -> Vec<f64> {
        let g = &self.grid;
        let (ox, oy) = (self.off_x(), self.off_y());
        let mut out = vec![0.0; g.len()];
        for iy in 0..g.ny {
            for ix in 0..g.nx {
                let k = g.idx(ix, iy);
                let mut s = (-2.0 * ox - 2.0 * oy + self.potential[k]) * psi[k];
                if ix > 0 {
                    s += ox * psi[g.idx(ix - 1, iy)];
                }
                if ix + 1 < g.nx {
                    s += ox * psi[g.idx(ix + 1, iy)];
                }
                if iy > 0 {
                    s += oy * psi[g.idx(ix, iy - 1)];
                }
                if iy + 1 < g.ny {
                    s += oy * psi[g.idx(ix, iy + 1)];
                }
                out[k] = s;
            }
        }
        out
    }

    /// The lowest `k` bound states, by matrix-free Lanczos.
    ///
    /// The 2-D Hamiltonian is `(nx*ny)^2`, so a dense eigensolver is out
    /// of the question — a 200 x 200 grid would need a 40 000 x 40 000
    /// matrix, about 12 GB. Lanczos needs only the five-point stencil,
    /// at `O(nx*ny)` per application.
    ///
    /// Wavefunctions are normalised so that `integral |psi|^2 dx dy = 1`,
    /// which means dividing the unit-length eigenvector by
    /// `sqrt(hx*hy)`. Skipping that is a silent factor-of-cell-area
    /// error in every expectation value afterwards.
    ///
    /// Degeneracies are resolved: see the note in
    /// `special_functions::lanczos`. The 2-D isotropic oscillator's
    /// `E = 2, 2` and `E = 3, 3, 3` come out with the right
    /// multiplicities and orthogonal partners.
    ///
    /// Pass `max_iters = 0` to let the solver choose its own Krylov
    /// budget, scaling with `k` and capped at the grid size; any
    /// positive value overrides it.
    ///
    /// # Errors
    /// `k == 0` or `k > nx*ny`, an absorbing Hamiltonian (not
    /// Hermitian), or a Lanczos failure. A run that hits `max_iters`
    /// without converging is NOT an error — the residuals are returned
    /// so the caller can judge — but `converged` says so.
    pub fn bound_states(
        &self,
        k: usize,
        max_iters: usize,
    ) -> Result<BoundStates2, String> {
        if self.is_absorbing() {
            return Err(
                "bound_states: an absorbing potential makes the Hamiltonian NON-Hermitian, so a \
                 symmetric eigensolver would return confident nonsense. Remove the absorber."
                    .to_string(),
            );
        }
        let n = self.grid.len();
        // A Krylov space of 4k+40 is far too small for a 2-D grid: the
        // first attempt used it and every state came back unconverged.
        // Scale with the problem, and let the caller override.
        let budget = if max_iters > 0 {
            max_iters
        } else {
            (30 * k + 200).min(n)
        };
        let r = lanczos_lowest(n, k, |v| self.apply_real(v), 1e-7, budget)?;
        let s = 1.0 / self.grid.cell().sqrt();
        let states = r
            .vectors
            .into_iter()
            .map(|mut v| {
                for x in v.iter_mut() {
                    *x *= s;
                }
                // reproducible sign: largest component positive
                let lead = v
                    .iter()
                    .cloned()
                    .fold(0.0_f64, |acc, x| if x.abs() > acc.abs() { x } else { acc });
                if lead < 0.0 {
                    for x in v.iter_mut() {
                        *x = -*x;
                    }
                }
                v
            })
            .collect();
        Ok(BoundStates2 {
            energies: r.values,
            states,
            residuals: r.residuals,
            iterations: r.iterations,
            converged: r.stop != Stop::MaxIters,
        })
    }

    /// Diagonal of the directional operator `A_d = T_d + V/2` (plus the
    /// absorber's half share) at a point.
    fn diag_dir(&self, k: usize, off: f64) -> C {
        let re = -2.0 * off + 0.5 * self.potential[k];
        match &self.absorber {
            Some(w) => C::new(re, -0.5 * w[k]),
            None => C::real(re),
        }
    }
}

/// What [`Hamiltonian2::bound_states`] returns.
///
/// The residuals are part of the answer, not diagnostics: an iterative
/// eigensolver has no exact stopping point, and a caller who cannot see
/// how well a state converged cannot know whether to trust it.
#[derive(Clone, Debug)]
pub struct BoundStates2 {
    /// Ascending.
    pub energies: Vec<f64>,
    /// Normalised so `integral |psi|^2 dx dy = 1`, row-major.
    pub states: Vec<Vec<f64>>,
    /// `‖H psi - E psi‖` per state.
    pub residuals: Vec<f64>,
    /// Total Lanczos iterations across all deflation passes.
    pub iterations: usize,
    /// False if the iteration limit was hit before the tolerance.
    pub converged: bool,
}

/// Lift a real amplitude to a complex one — a named function so callers
/// outside this crate need not depend on the complex type's spelling.
pub fn real_to_complex(v: &f64) -> C {
    C::real(*v)
}

/// A 2-D wavefunction.
#[derive(Clone, Debug)]
pub struct Wavefunction2 {
    pub grid: Grid2,
    pub psi: Vec<C>,
}

impl Wavefunction2 {
    /// # Errors
    /// Length mismatch.
    pub fn new(grid: Grid2, psi: Vec<C>) -> Result<Self, String> {
        if psi.len() != grid.len() {
            return Err(format!(
                "Wavefunction2: {} values for a grid of {}",
                psi.len(),
                grid.len()
            ));
        }
        Ok(Self { grid, psi })
    }

    /// A normalised 2-D Gaussian packet with mean momentum
    /// `hbar (kx, ky)`.
    ///
    /// # Errors
    /// Non-positive widths, non-finite parameters, or a packet whose
    /// norm underflows because it lies outside the domain.
    #[allow(clippy::too_many_arguments)]
    pub fn gaussian(
        grid: Grid2,
        x0: f64,
        y0: f64,
        sigma_x: f64,
        sigma_y: f64,
        kx: f64,
        ky: f64,
    ) -> Result<Self, String> {
        if !(sigma_x.is_finite() && sigma_x > 0.0 && sigma_y.is_finite() && sigma_y > 0.0) {
            return Err("gaussian: sigma_x and sigma_y must be finite and positive".to_string());
        }
        if ![x0, y0, kx, ky].iter().all(|v| v.is_finite()) {
            return Err("gaussian: x0, y0, kx, ky must be finite".to_string());
        }
        let mut psi = Vec::with_capacity(grid.len());
        for iy in 0..grid.ny {
            for ix in 0..grid.nx {
                let (x, y) = (grid.x(ix), grid.y(iy));
                let ax = (x - x0) * (x - x0) / (4.0 * sigma_x * sigma_x);
                let ay = (y - y0) * (y - y0) / (4.0 * sigma_y * sigma_y);
                psi.push(C::from_polar((-(ax + ay)).exp(), kx * x + ky * y));
            }
        }
        let mut w = Self::new(grid, psi)?;
        w.normalise()?;
        Ok(w)
    }

    /// `integral |psi|^2 dx dy`.
    pub fn norm(&self) -> f64 {
        self.psi.iter().map(|z| z.norm_sqr()).sum::<f64>() * self.grid.cell()
    }

    /// # Errors
    /// The norm is zero or not finite.
    pub fn normalise(&mut self) -> Result<(), String> {
        let n = self.norm();
        if !n.is_finite() || n <= 0.0 {
            return Err(format!("normalise: the norm is {n} — the state is empty"));
        }
        let s = 1.0 / n.sqrt();
        for z in self.psi.iter_mut() {
            *z = *z * s;
        }
        Ok(())
    }

    /// `(<x>, <y>)`.
    pub fn centroid(&self) -> (f64, f64) {
        let g = &self.grid;
        let (mut sx, mut sy) = (0.0, 0.0);
        for iy in 0..g.ny {
            for ix in 0..g.nx {
                let p = self.psi[g.idx(ix, iy)].norm_sqr();
                sx += g.x(ix) * p;
                sy += g.y(iy) * p;
            }
        }
        (sx * g.cell(), sy * g.cell())
    }

    /// `<H> = <psi|H|psi> / <psi|psi>`.
    ///
    /// Divided by the norm, for the same reason as in 1-D: an absorbing
    /// potential makes the norm decay, and an undivided integral would
    /// report a falling energy for a packet whose energy is unchanged.
    pub fn energy(&self, ham: &Hamiltonian2) -> f64 {
        let hp = ham.apply(&self.psi);
        let num: f64 = self
            .psi
            .iter()
            .zip(&hp)
            .map(|(a, b)| (a.conj() * *b).re)
            .sum::<f64>()
            * self.grid.cell();
        let den = self.norm();
        if den > 0.0 {
            num / den
        } else {
            f64::NAN
        }
    }

    /// Probability inside the rectangle `[xa, xb] x [ya, yb]`.
    pub fn probability_in(&self, xa: f64, xb: f64, ya: f64, yb: f64) -> f64 {
        let g = &self.grid;
        let (x0, x1) = if xa <= xb { (xa, xb) } else { (xb, xa) };
        let (y0, y1) = if ya <= yb { (ya, yb) } else { (yb, ya) };
        let mut s = 0.0;
        for iy in 0..g.ny {
            let y = g.y(iy);
            if y < y0 || y > y1 {
                continue;
            }
            for ix in 0..g.nx {
                let x = g.x(ix);
                if x >= x0 && x <= x1 {
                    s += self.psi[g.idx(ix, iy)].norm_sqr();
                }
            }
        }
        s * g.cell()
    }

    /// `|psi|^2` at every point, row-major.
    pub fn density(&self) -> Vec<f64> {
        self.psi.iter().map(|z| z.norm_sqr()).collect()
    }

    /// Probability within `frac` of the domain size of any wall.
    pub fn edge_probability(&self, frac: f64) -> f64 {
        let g = &self.grid;
        let wx = (g.x_max - g.x_min) * frac;
        let wy = (g.y_max - g.y_min) * frac;
        let mut s = 0.0;
        for iy in 0..g.ny {
            let y = g.y(iy);
            for ix in 0..g.nx {
                let x = g.x(ix);
                if x < g.x_min + wx
                    || x > g.x_max - wx
                    || y < g.y_min + wy
                    || y > g.y_max - wy
                {
                    s += self.psi[g.idx(ix, iy)].norm_sqr();
                }
            }
        }
        s * g.cell()
    }
}

/// One-directional Cayley step, precomputed.
struct DirStep {
    /// Bands of `1 + i tau A / 2hbar`, one line at a time (flattened).
    sub: Vec<C>,
    diag: Vec<C>,
    sup: Vec<C>,
    /// `i tau / (2 hbar)`.
    half: C,
    off: f64,
}

/// An ADI propagator: Strang-composed Cayley steps,
/// `U_x(dt/2) U_y(dt) U_x(dt/2)`.
///
/// Each factor is exactly unitary, so the norm is conserved to machine
/// precision for any `dt`; the splitting error is `O(dt^2)` in the
/// dynamics only.
pub struct Propagator2 {
    ham: Hamiltonian2,
    dt: f64,
    /// x-direction at dt/2, y-direction at dt.
    xs: DirStep,
    ys: DirStep,
}

impl Propagator2 {
    /// # Errors
    /// A non-finite or zero `dt`.
    pub fn new(ham: Hamiltonian2, dt: f64) -> Result<Self, String> {
        if !dt.is_finite() || dt == 0.0 {
            return Err(format!("Propagator2: dt must be finite and non-zero, got {dt}"));
        }
        let g = ham.grid.clone();
        let build = |tau: f64, along_x: bool| -> DirStep {
            let half = C::I * (tau / (2.0 * ham.hbar));
            let off = if along_x { ham.off_x() } else { ham.off_y() };
            let n = g.len();
            let mut sub = vec![C::ZERO; n];
            let mut diag = vec![C::ZERO; n];
            let mut sup = vec![C::ZERO; n];
            let o = half * C::real(off);
            for iy in 0..g.ny {
                for ix in 0..g.nx {
                    let k = g.idx(ix, iy);
                    sub[k] = o;
                    sup[k] = o;
                    diag[k] = C::ONE + half * ham.diag_dir(k, off);
                }
            }
            DirStep { sub, diag, sup, half, off }
        };
        let xs = build(dt / 2.0, true);
        let ys = build(dt, false);
        Ok(Self { ham, dt, xs, ys })
    }

    pub fn dt(&self) -> f64 {
        self.dt
    }
    pub fn hamiltonian(&self) -> &Hamiltonian2 {
        &self.ham
    }

    /// Apply one directional Cayley factor in place.
    ///
    /// `along_x` selects rows (contiguous) or columns (strided). Each
    /// line is an independent tridiagonal system — that independence is
    /// the whole reason ADI is affordable.
    fn apply_dir(&self, w: &mut Wavefunction2, s: &DirStep, along_x: bool) -> Result<(), String> {
        let g = self.ham.grid.clone();
        let (lines, len) = if along_x { (g.ny, g.nx) } else { (g.nx, g.ny) };
        let mut sub = vec![C::ZERO; len];
        let mut diag = vec![C::ZERO; len];
        let mut sup = vec![C::ZERO; len];
        let mut rhs = vec![C::ZERO; len];
        for l in 0..lines {
            // gather the line and form rhs = (1 - i tau A / 2hbar) psi
            for j in 0..len {
                let k = if along_x { g.idx(j, l) } else { g.idx(l, j) };
                sub[j] = s.sub[k];
                diag[j] = s.diag[k];
                sup[j] = s.sup[k];
                let a_diag = self.ham.diag_dir(k, s.off);
                let mut ap = a_diag * w.psi[k];
                if j > 0 {
                    let km = if along_x { g.idx(j - 1, l) } else { g.idx(l, j - 1) };
                    ap = ap + w.psi[km] * s.off;
                }
                if j + 1 < len {
                    let kp = if along_x { g.idx(j + 1, l) } else { g.idx(l, j + 1) };
                    ap = ap + w.psi[kp] * s.off;
                }
                rhs[j] = w.psi[k] - s.half * ap;
            }
            let sol = solve_tridiag_c(&sub, &diag, &sup, &rhs)?;
            for (j, v) in sol.into_iter().enumerate() {
                let k = if along_x { g.idx(j, l) } else { g.idx(l, j) };
                w.psi[k] = v;
            }
        }
        Ok(())
    }

    /// Advance by one step.
    ///
    /// # Errors
    /// A grid mismatch or a tridiagonal solve failure.
    pub fn step(&self, w: &mut Wavefunction2) -> Result<(), String> {
        if w.grid != self.ham.grid {
            return Err("step: the wavefunction and propagator use different grids".to_string());
        }
        self.apply_dir(w, &self.xs, true)?;
        self.apply_dir(w, &self.ys, false)?;
        self.apply_dir(w, &self.xs, true)?;
        Ok(())
    }

    /// Advance by `steps` steps.
    ///
    /// # Errors
    /// As [`Propagator2::step`].
    pub fn run(&self, w: &mut Wavefunction2, steps: usize) -> Result<(), String> {
        for _ in 0..steps {
            self.step(w)?;
        }
        Ok(())
    }
}

/// A 2-D propagator for `H(t) = H_0 + f(t) g(x, y)` — the ADI
/// counterpart of [`crate::qm1d::DrivenPropagator`], with the same
/// factorisation and the same midpoint sampling.
pub struct DrivenPropagator2 {
    ham: Hamiltonian2,
    shape: Vec<f64>,
    dt: f64,
    time: f64,
}

impl DrivenPropagator2 {
    /// # Errors
    /// Shape length mismatch, non-finite shape, or a zero `dt`.
    pub fn new(ham: Hamiltonian2, shape: Vec<f64>, dt: f64) -> Result<Self, String> {
        if shape.len() != ham.grid.len() {
            return Err(format!(
                "DrivenPropagator2: the drive shape has {} values but the grid has {}",
                shape.len(),
                ham.grid.len()
            ));
        }
        if shape.iter().any(|v| !v.is_finite()) {
            return Err("DrivenPropagator2: the drive shape has a non-finite value".to_string());
        }
        if !dt.is_finite() || dt == 0.0 {
            return Err(format!("DrivenPropagator2: dt must be finite and non-zero, got {dt}"));
        }
        Ok(Self { ham, shape, dt, time: 0.0 })
    }

    pub fn time(&self) -> f64 {
        self.time
    }
    pub fn dt(&self) -> f64 {
        self.dt
    }

    fn hamiltonian_at(&self, amp: f64) -> Result<Hamiltonian2, String> {
        if !amp.is_finite() {
            return Err(format!("DrivenPropagator2: the modulation returned {amp}"));
        }
        let v: Vec<f64> = self
            .ham
            .potential
            .iter()
            .zip(&self.shape)
            .map(|(v0, g)| v0 + amp * g)
            .collect();
        let mut h = Hamiltonian2::new(
            self.ham.grid.clone(),
            v,
            self.ham.mass,
            self.ham.hbar,
        )?;
        h.absorber = self.ham.absorber.clone();
        Ok(h)
    }

    /// One ADI step with the modulation taken at the midpoint.
    ///
    /// # Errors
    /// Grid mismatch, non-finite modulation, or a solve failure.
    pub fn step<F: Fn(f64) -> f64>(
        &mut self,
        w: &mut Wavefunction2,
        modulation: F,
    ) -> Result<(), String> {
        if w.grid != self.ham.grid {
            return Err("step: the wavefunction and propagator use different grids".to_string());
        }
        let amp = modulation(self.time + 0.5 * self.dt);
        Propagator2::new(self.hamiltonian_at(amp)?, self.dt)?.step(w)?;
        self.time += self.dt;
        Ok(())
    }

    /// `steps` steps.
    ///
    /// # Errors
    /// As [`DrivenPropagator2::step`].
    pub fn run<F: Fn(f64) -> f64 + Copy>(
        &mut self,
        w: &mut Wavefunction2,
        steps: usize,
        modulation: F,
    ) -> Result<(), String> {
        for _ in 0..steps {
            self.step(w, modulation)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qm1d;
    use std::f64::consts::PI;

    /// Unitarity is the sharp test, and Strang-composed Cayley factors
    /// are unitary for ANY dt — so a deliberately huge step must still
    /// conserve the norm. Peaceman–Rachford would drift here; that is
    /// exactly why this scheme was chosen.
    #[test]
    fn adi_is_unitary_at_any_step_size() {
        for &dt in &[0.001, 0.05, 1.0, 20.0] {
            let g = Grid2::new(-8.0, 8.0, 40, -8.0, 8.0, 40).unwrap();
            let ham =
                Hamiltonian2::from_fn(g.clone(), |x, y| 0.5 * (x * x + y * y), 1.0, 1.0).unwrap();
            let mut w =
                Wavefunction2::gaussian(g, -1.0, 0.5, 1.0, 1.0, 1.0, -0.5).unwrap();
            let n0 = w.norm();
            Propagator2::new(ham, dt).unwrap().run(&mut w, 40).unwrap();
            let n1 = w.norm();
            assert!(
                (n1 / n0 - 1.0).abs() < 1e-10,
                "dt = {dt}: norm {n0} -> {n1}"
            );
        }
    }

    /// A potential independent of y, with a y-independent initial state,
    /// is really a 1-D problem. The 2-D ADI solver must reproduce the
    /// 1-D Crank–Nicolson solver — two different algorithms, one answer.
    /// This is the strongest correctness check available, because
    /// nothing about the codes is shared beyond the tridiagonal solver.
    #[test]
    fn a_separable_problem_reproduces_the_1d_solver() {
        let (nx, xlo, xhi) = (200usize, -30.0_f64, 30.0_f64);
        let v = |x: f64| if (0.0..1.0).contains(&x) { 2.0 } else { 0.0 };

        // 1-D reference
        let g1 = qm1d::Grid::new(xlo, xhi, nx).unwrap();
        let h1 = qm1d::Hamiltonian::from_fn(g1.clone(), v, 1.0, 1.0).unwrap();
        let mut w1 = qm1d::Wavefunction::gaussian(g1, -10.0, 2.0, 2.0).unwrap();
        qm1d::Propagator::new(h1, 0.005).unwrap().run(&mut w1, 1200).unwrap();
        let t1 = w1.probability_in(1.0, 30.0);

        // 2-D, uniform in y, with periodic-free (Dirichlet) y walls far
        // enough away that the y profile stays flat over the run.
        let g2 = Grid2::new(xlo, xhi, nx, -30.0, 30.0, 24).unwrap();
        let h2 = Hamiltonian2::from_fn(g2.clone(), |x, _| v(x), 1.0, 1.0).unwrap();
        // A very wide y Gaussian approximates "uniform in y".
        let mut w2 =
            Wavefunction2::gaussian(g2.clone(), -10.0, 0.0, 2.0, 200.0, 2.0, 0.0).unwrap();
        Propagator2::new(h2, 0.005).unwrap().run(&mut w2, 1200).unwrap();
        let t2 = w2.probability_in(1.0, 30.0, -30.0, 30.0);

        let rel = (t2 - t1).abs() / t1;
        assert!(rel < 0.02, "2-D gave T = {t2}, 1-D reference {t1} ({:.2}% apart)", 100.0 * rel);
    }

    /// A 2-D free packet's centroid moves at the group velocity in both
    /// directions independently.
    #[test]
    fn a_free_packet_drifts_in_both_directions() {
        let g = Grid2::new(-40.0, 40.0, 160, -40.0, 40.0, 160).unwrap();
        let ham = Hamiltonian2::from_fn(g.clone(), |_, _| 0.0, 1.0, 1.0).unwrap();
        let (kx, ky) = (1.5_f64, -0.8_f64);
        let mut w = Wavefunction2::gaussian(g.clone(), -10.0, 6.0, 3.0, 3.0, kx, ky).unwrap();
        let (x0, y0) = w.centroid();
        let (dt, steps) = (0.01_f64, 600usize);
        Propagator2::new(ham, dt).unwrap().run(&mut w, steps).unwrap();
        let t = dt * steps as f64;
        let (x1, y1) = w.centroid();
        // discrete group velocity, as in 1-D: hbar sin(k h)/(m h)
        let vx = (kx * g.hx()).sin() / g.hx();
        let vy = (ky * g.hy()).sin() / g.hy();
        assert!((x1 - (x0 + vx * t)).abs() < 0.2, "<x> = {x1}, want {}", x0 + vx * t);
        assert!((y1 - (y0 + vy * t)).abs() < 0.2, "<y> = {y1}, want {}", y0 + vy * t);
        assert!(w.edge_probability(0.05) < 1e-9, "the packet reached a wall");
    }

    /// The 2-D isotropic oscillator has E = nx + ny + 1. A product of
    /// two 1-D ground states is the 2-D ground state, so propagating it
    /// must hold both the density and the energy still.
    #[test]
    fn the_2d_oscillator_ground_state_is_stationary() {
        let g1 = qm1d::Grid::new(-7.0, 7.0, 60).unwrap();
        let h1 = qm1d::Hamiltonian::from_fn(g1.clone(), |x| 0.5 * x * x, 1.0, 1.0).unwrap();
        let (e1, s1) = h1.bound_states(1).unwrap();

        let g2 = Grid2::new(-7.0, 7.0, 60, -7.0, 7.0, 60).unwrap();
        let ham = Hamiltonian2::from_fn(g2.clone(), |x, y| 0.5 * (x * x + y * y), 1.0, 1.0)
            .unwrap();
        // psi(x,y) = phi0(x) phi0(y)
        let mut psi = Vec::with_capacity(g2.len());
        for iy in 0..g2.ny {
            for ix in 0..g2.nx {
                psi.push(C::real(s1[0][ix] * s1[0][iy]));
            }
        }
        let mut w = Wavefunction2::new(g2, psi).unwrap();
        w.normalise().unwrap();

        // E = 2 * E_1d(0), i.e. 1 in natural units
        let e_before = w.energy(&ham);
        assert!(
            (e_before - 2.0 * e1[0]).abs() < 1e-6,
            "E = {e_before}, want {}",
            2.0 * e1[0]
        );
        assert!((e_before - 1.0).abs() < 5e-3, "E = {e_before}, want ~1");

        // The product state IS an exact eigenvector of the 2-D discrete
        // Hamiltonian, so it would be perfectly stationary under the
        // unsplit propagator. It is NOT perfectly stationary here, and
        // the reason is worth stating: with A_x = T_x + V/2 and
        // A_y = T_y + V/2, the commutator
        //     [A_x, A_y] = [T_x, V_x/2] + [V_y/2, T_y]
        // is non-zero even for a SEPARABLE potential, so Strang
        // splitting has a genuine O(dt^2) error. (Splitting V by axis
        // instead would cancel it, but a general V does not decompose
        // that way, so the even split is the honest default.)
        //
        // So the assertion is on the SCALING, not a magnitude: halving
        // dt must cut the drift by about four.
        let drift = |dt: f64, steps: usize| -> f64 {
            let mut wl = w.clone();
            let d0 = wl.density();
            Propagator2::new(ham.clone(), dt).unwrap().run(&mut wl, steps).unwrap();
            wl.density()
                .iter()
                .zip(&d0)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max)
        };
        let coarse = drift(0.02, 100);
        let fine = drift(0.01, 200);
        assert!(coarse < 1e-4, "drift {coarse} is too large to be splitting error alone");
        let ratio = coarse / fine;
        assert!(
            (3.0..5.2).contains(&ratio),
            "drift fell {ratio}x when dt halved, expected ~4 (coarse {coarse:.3e}, fine {fine:.3e})"
        );

        // The ENERGY, by contrast, is conserved far better: each Cayley
        // factor is unitary and the splitting error largely cancels over
        // the symmetric Strang sequence.
        let mut wl = w.clone();
        Propagator2::new(ham.clone(), 0.01).unwrap().run(&mut wl, 200).unwrap();
        assert!((wl.energy(&ham) - e_before).abs() < 1e-8, "energy drifted");
        assert!((wl.norm() - 1.0).abs() < 1e-12, "norm drifted");
    }

    /// The 2-D infinite well: on a square of side L the exact continuum
    /// spectrum is `(nx^2+ny^2) pi^2 / 2 L^2`. Checked through the
    /// energy of a product sine state, which is what the discretisation
    /// should reproduce to second order.
    #[test]
    fn infinite_square_well_energy() {
        let l = 1.0_f64;
        let n = 40usize;
        let g = Grid2::new(0.0, l, n, 0.0, l, n).unwrap();
        let ham = Hamiltonian2::from_fn(g.clone(), |_, _| 0.0, 1.0, 1.0).unwrap();
        // psi = 2/L sin(pi x/L) sin(2 pi y/L): the (1,2) state
        let mut psi = Vec::with_capacity(g.len());
        for iy in 0..g.ny {
            for ix in 0..g.nx {
                psi.push(C::real(
                    (PI * g.x(ix) / l).sin() * (2.0 * PI * g.y(iy) / l).sin(),
                ));
            }
        }
        let mut w = Wavefunction2::new(g, psi).unwrap();
        w.normalise().unwrap();
        let exact = (1.0 + 4.0) * PI * PI / (2.0 * l * l);
        let got = w.energy(&ham);
        assert!((got - exact).abs() / exact < 0.01, "E = {got}, exact {exact}");
    }

    /// The splitting error is second order: halving dt must cut the
    /// deviation from a reference solution by about four. Norm
    /// conservation is exact regardless, so this measures the DYNAMICS
    /// alone — which is the point of choosing a unitary splitting.
    #[test]
    fn splitting_error_is_second_order() {
        let g = Grid2::new(-8.0, 8.0, 48, -8.0, 8.0, 48).unwrap();
        let make = || Hamiltonian2::from_fn(g.clone(), |x, y| 0.5 * (x * x + y * y), 1.0, 1.0).unwrap();
        let start = || Wavefunction2::gaussian(g.clone(), -1.5, 1.0, 1.0, 1.0, 0.8, -0.4).unwrap();
        let total = 1.0_f64;
        let run = |dt: f64| {
            let mut w = start();
            let steps = (total / dt).round() as usize;
            Propagator2::new(make(), dt).unwrap().run(&mut w, steps).unwrap();
            w.density()
        };
        let reference = run(0.0005);
        let err = |d: Vec<f64>| {
            d.iter()
                .zip(&reference)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max)
        };
        let e_coarse = err(run(0.04));
        let e_fine = err(run(0.02));
        let ratio = e_coarse / e_fine;
        assert!(
            (3.0..5.2).contains(&ratio),
            "error fell {ratio}x when dt halved, expected ~4 (coarse {e_coarse:.3e}, fine {e_fine:.3e})"
        );
    }

    /// Absorbing edges work in 2-D too: a packet driven into a corner is
    /// swallowed rather than bounced.
    #[test]
    fn absorbing_edges_swallow_a_packet() {
        let g = Grid2::new(-20.0, 20.0, 100, -20.0, 20.0, 100).unwrap();
        let ham = Hamiltonian2::from_fn(g.clone(), |_, _| 0.0, 1.0, 1.0)
            .unwrap()
            .with_absorber(7.0, 3.0, 2.0)
            .unwrap();
        let mut w = Wavefunction2::gaussian(g, 0.0, 0.0, 1.5, 1.5, 3.0, 3.0).unwrap();
        Propagator2::new(ham, 0.005).unwrap().run(&mut w, 2000).unwrap();
        assert!(w.norm() < 0.02, "absorber left {} of the norm", w.norm());
        assert!(
            w.probability_in(-11.0, 11.0, -11.0, 11.0) < 1e-4,
            "the absorber reflected back into the interior"
        );
    }

    /// The 2-D isotropic oscillator: `E = nx + ny + 1`, so the spectrum
    /// is 1, 2, 2, 3, 3, 3 — **degenerate**, which is exactly what a
    /// single-vector Krylov method cannot see and deflation exists to
    /// fix. This is the test the whole Lanczos design is for.
    #[test]
    fn the_2d_oscillator_spectrum_with_its_degeneracies() {
        let g = Grid2::new(-7.0, 7.0, 70, -7.0, 7.0, 70).unwrap();
        let ham =
            Hamiltonian2::from_fn(g.clone(), |x, y| 0.5 * (x * x + y * y), 1.0, 1.0).unwrap();
        let b = ham.bound_states(6, 400).unwrap();
        let want = [1.0, 2.0, 2.0, 3.0, 3.0, 3.0];
        for (j, w) in want.iter().enumerate() {
            assert!(
                (b.energies[j] - w).abs() < 0.02,
                "E[{j}] = {}, want {w} (all: {:?})",
                b.energies[j],
                b.energies
            );
        }
        assert!(b.converged, "did not converge in the iteration budget");
        assert!(
            b.residuals.iter().all(|r| *r < 1e-6),
            "residuals too large: {:?}",
            b.residuals
        );
    }

    /// Degenerate partners must be genuinely different states, not two
    /// copies of one. Checked by orthogonality under the grid inner
    /// product — the giveaway for a ghost.
    #[test]
    fn degenerate_states_are_orthonormal() {
        let g = Grid2::new(-7.0, 7.0, 60, -7.0, 7.0, 60).unwrap();
        let ham =
            Hamiltonian2::from_fn(g.clone(), |x, y| 0.5 * (x * x + y * y), 1.0, 1.0).unwrap();
        let b = ham.bound_states(6, 400).unwrap();
        let cell = g.cell();
        for i in 0..6 {
            for j in 0..6 {
                let ip: f64 =
                    b.states[i].iter().zip(&b.states[j]).map(|(a, c)| a * c).sum::<f64>() * cell;
                let want = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (ip - want).abs() < 1e-6,
                    "<{i}|{j}> = {ip}, want {want}"
                );
            }
        }
    }

    /// A separable potential must give the 1-D spectrum summed: the 2-D
    /// energies are `E_i + E_j` from the 1-D solver, which is a
    /// cross-check against an entirely different eigensolver (dense
    /// Jacobi in 1-D versus matrix-free Lanczos here).
    #[test]
    fn separable_2d_spectrum_is_the_sum_of_1d_spectra() {
        // an ASYMMETRIC well, so the levels are non-degenerate and the
        // pairing is unambiguous
        let vx = |x: f64| 0.5 * x * x;
        let vy = |y: f64| 2.0 * y * y;
        let g1x = qm1d::Grid::new(-7.0, 7.0, 60).unwrap();
        let g1y = qm1d::Grid::new(-7.0, 7.0, 60).unwrap();
        let ex = qm1d::Hamiltonian::from_fn(g1x, vx, 1.0, 1.0)
            .unwrap()
            .bound_states(3)
            .unwrap()
            .0;
        let ey = qm1d::Hamiltonian::from_fn(g1y, vy, 1.0, 1.0)
            .unwrap()
            .bound_states(3)
            .unwrap()
            .0;
        let mut sums: Vec<f64> = Vec::new();
        for a in &ex {
            for b in &ey {
                sums.push(a + b);
            }
        }
        sums.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let g = Grid2::new(-7.0, 7.0, 60, -7.0, 7.0, 60).unwrap();
        let ham = Hamiltonian2::from_fn(g, move |x, y| vx(x) + vy(y), 1.0, 1.0).unwrap();
        let b = ham.bound_states(4, 400).unwrap();
        for (j, (got, want)) in b.energies.iter().zip(&sums).take(4).enumerate() {
            assert!((got - want).abs() < 1e-6, "E[{j}] = {got} vs 1-D sum {want}");
        }
    }

    /// A bound state must be stationary under ADI propagation — the
    /// eigensolver and the propagator must agree about the Hamiltonian.
    #[test]
    fn a_2d_bound_state_is_stationary() {
        let g = Grid2::new(-6.0, 6.0, 48, -6.0, 6.0, 48).unwrap();
        let ham =
            Hamiltonian2::from_fn(g.clone(), |x, y| 0.5 * (x * x + y * y), 1.0, 1.0).unwrap();
        let b = ham.bound_states(1, 400).unwrap();
        let mut w =
            Wavefunction2::new(g, b.states[0].iter().map(|&v| C::real(v)).collect()).unwrap();
        w.normalise().unwrap();
        let e0 = w.energy(&ham);
        assert!((e0 - b.energies[0]).abs() < 1e-6, "E {e0} vs eigenvalue {}", b.energies[0]);
        let d0 = w.density();
        Propagator2::new(ham.clone(), 0.01).unwrap().run(&mut w, 100).unwrap();
        let worst = d0
            .iter()
            .zip(&w.density())
            .map(|(a, c)| (a - c).abs())
            .fold(0.0_f64, f64::max);
        assert!(worst < 1e-5, "the ground-state density moved by {worst}");
        assert!((w.energy(&ham) - e0).abs() < 1e-8, "energy drifted");
    }

    /// The eigensolver must refuse an absorbing Hamiltonian.
    #[test]
    fn bound_states_refuse_an_absorbing_hamiltonian() {
        let g = Grid2::new(-6.0, 6.0, 30, -6.0, 6.0, 30).unwrap();
        let ham = Hamiltonian2::from_fn(g, |x, y| 0.5 * (x * x + y * y), 1.0, 1.0)
            .unwrap()
            .with_absorber(2.0, 1.0, 2.0)
            .unwrap();
        assert!(ham.bound_states(2, 200).unwrap_err().contains("Hermitian"));
    }

    /// The 2-D driven oscillator: Ehrenfest is exact for a quadratic
    /// potential with a linear drive in EACH direction independently,
    /// so driving along x alone must move `<x>` on the classical
    /// trajectory and leave `<y>` at zero. The second half is the real
    /// content — it would fail if the ADI directions were crossed.
    #[test]
    fn a_2d_drive_moves_only_the_driven_axis() {
        let g = Grid2::new(-9.0, 9.0, 72, -9.0, 9.0, 72).unwrap();
        let ham =
            Hamiltonian2::from_fn(g.clone(), |x, y| 0.5 * (x * x + y * y), 1.0, 1.0).unwrap();
        let b = ham.bound_states(1, 300).unwrap();
        let mut w =
            Wavefunction2::new(g.clone(), b.states[0].iter().map(|&v| C::real(v)).collect())
                .unwrap();
        w.normalise().unwrap();

        // g(x, y) = x: a dipole drive along x only
        let mut shape = Vec::with_capacity(g.len());
        for iy in 0..g.ny {
            for ix in 0..g.nx {
                let _ = iy;
                shape.push(g.x(ix));
            }
        }
        let (f0, om) = (0.3_f64, 0.7_f64);
        let dt = 0.005;
        let mut prop = DrivenPropagator2::new(ham, shape, dt).unwrap();
        prop.run(&mut w, 1000, move |t| f0 * (om * t).cos()).unwrap();

        let t = dt * 1000.0;
        let exact = -f0 / (1.0 - om * om) * ((om * t).cos() - t.cos());
        let (cx, cy) = w.centroid();
        assert!((cx - exact).abs() < 0.02, "<x> = {cx}, classical {exact}");
        assert!(cy.abs() < 1e-9, "<y> = {cy}, must not move — the drive is along x only");
        assert!((w.norm() - 1.0).abs() < 1e-10, "norm = {}", w.norm());
    }

    #[test]
    fn invalid_input_is_reported() {
        assert!(Grid2::new(0.0, 1.0, 0, 0.0, 1.0, 5).is_err(), "nx = 0");
        assert!(Grid2::new(1.0, 0.0, 5, 0.0, 1.0, 5).is_err(), "reversed x");
        assert!(Grid2::new(0.0, 1.0, 5, 1.0, 0.0, 5).is_err(), "reversed y");
        let g = Grid2::new(0.0, 1.0, 4, 0.0, 1.0, 3).unwrap();
        assert_eq!(g.len(), 12);
        assert!(Hamiltonian2::new(g.clone(), vec![0.0; 5], 1.0, 1.0).is_err(), "length");
        assert!(Hamiltonian2::new(g.clone(), vec![0.0; 12], 0.0, 1.0).is_err(), "mass");
        let h = Hamiltonian2::from_fn(g.clone(), |_, _| 0.0, 1.0, 1.0).unwrap();
        assert!(Propagator2::new(h.clone(), 0.0).is_err(), "dt = 0");
        assert!(h.with_absorber(1.0, 1.0, 2.0).is_err(), "absorber too wide for the domain");
        assert!(
            Wavefunction2::gaussian(g, 0.0, 0.0, -1.0, 1.0, 0.0, 0.0).is_err(),
            "sigma < 0"
        );
    }

    /// Row-major indexing must be consistent between the grid, the
    /// Hamiltonian and the propagator — an axis swap here would be
    /// invisible on a square grid, so the test uses a RECTANGULAR one.
    #[test]
    fn indexing_is_consistent_on_a_non_square_grid() {
        let g = Grid2::new(-4.0, 4.0, 30, -2.0, 2.0, 14).unwrap();
        assert_eq!(g.len(), 30 * 14);
        assert_eq!(g.idx(29, 13), 30 * 14 - 1);
        // a potential that depends only on x must be constant down each
        // column and vary along each row
        let ham = Hamiltonian2::from_fn(g.clone(), |x, _| x, 1.0, 1.0).unwrap();
        for iy in 0..g.ny {
            for ix in 0..g.nx {
                let v = ham.potential[g.idx(ix, iy)];
                assert!((v - g.x(ix)).abs() < 1e-12, "V at ({ix},{iy}) = {v}, want {}", g.x(ix));
            }
        }
        // and propagation must run without a shape error
        let mut w = Wavefunction2::gaussian(g, 0.0, 0.0, 1.0, 0.5, 0.5, 0.0).unwrap();
        let p = Propagator2::new(ham, 0.01).unwrap();
        p.run(&mut w, 10).unwrap();
        assert!((w.norm() - 1.0).abs() < 1e-10);
    }
}
