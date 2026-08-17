//! Three-dimensional quantum mechanics by ADI.
//!
//! The structure is the 2-D scheme with one more direction: a Cayley
//! transform per axis, Strang-composed so the whole step stays exactly
//! unitary,
//!
//! ```text
//!   psi(t+dt) = U_x(dt/2) U_y(dt/2) U_z(dt) U_y(dt/2) U_x(dt/2) psi(t)
//! ```
//!
//! with `A_d = T_d + V/3`, each `A_d` Hermitian and each `U_d` therefore
//! unitary. Every factor reduces to independent tridiagonal solves along
//! lines of the grid, so a step costs `O(nx*ny*nz)`.
//!
//! # Memory is the binding constraint, not arithmetic
//!
//! Three dimensions change the economics. A 100³ grid is a million
//! points: one complex wavefunction is 16 MB, which is fine — but the
//! 2-D propagator precomputes full-size band arrays, three per
//! direction, and doing that here would cost about 144 MB before any
//! work began.
//!
//! So this module builds bands **per line, on the fly**. The off-diagonal
//! is constant along a direction and the diagonal is a cheap function of
//! the potential, so the only extra storage is `O(max(nx, ny, nz))`. The
//! cost is recomputing a few numbers per point per step, which is far
//! less than the memory traffic it avoids.
//!
//! # What does not scale
//!
//! Propagation is comfortable at 100³ and beyond. **Bound states are
//! not.** [`Hamiltonian3::bound_states`] uses the same Lanczos solver as
//! 2-D, whose full reorthogonalisation costs `O(m^2 n)` — at a million
//! points and a few hundred Krylov vectors that is `10^{11}` operations
//! and gigabytes of Krylov basis. It is practical to roughly 40³ and is
//! documented as such rather than left to be discovered.

use special_functions::complex::Complex64 as C;
use special_functions::lanczos::{lanczos_lowest, Stop};
use special_functions::tridiag::solve_tridiag_c;

/// A uniform 3-D grid of interior points, Dirichlet on all six faces.
/// Index order is `((iz * ny) + iy) * nx + ix` — x fastest.
#[derive(Clone, Debug, PartialEq)]
pub struct Grid3 {
    pub x_min: f64,
    pub x_max: f64,
    pub nx: usize,
    pub y_min: f64,
    pub y_max: f64,
    pub ny: usize,
    pub z_min: f64,
    pub z_max: f64,
    pub nz: usize,
}

/// Which axis a directional sweep runs along.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Axis {
    X,
    Y,
    Z,
}

impl Grid3 {
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
        z_min: f64,
        z_max: f64,
        nz: usize,
    ) -> Result<Self, String> {
        if nx == 0 || ny == 0 || nz == 0 {
            return Err(format!(
                "Grid3: needs at least one point per axis, got {nx} x {ny} x {nz}"
            ));
        }
        for (lo, hi, ax) in [
            (x_min, x_max, "x"),
            (y_min, y_max, "y"),
            (z_min, z_max, "z"),
        ] {
            if !lo.is_finite() || !hi.is_finite() {
                return Err(format!("Grid3: {ax} bounds must be finite, got [{lo}, {hi}]"));
            }
            if hi <= lo {
                return Err(format!("Grid3: {ax}_max ({hi}) must exceed {ax}_min ({lo})"));
            }
        }
        Ok(Self { x_min, x_max, nx, y_min, y_max, ny, z_min, z_max, nz })
    }

    pub fn hx(&self) -> f64 {
        (self.x_max - self.x_min) / (self.nx + 1) as f64
    }
    pub fn hy(&self) -> f64 {
        (self.y_max - self.y_min) / (self.ny + 1) as f64
    }
    pub fn hz(&self) -> f64 {
        (self.z_max - self.z_min) / (self.nz + 1) as f64
    }
    pub fn x(&self, i: usize) -> f64 {
        self.x_min + (i + 1) as f64 * self.hx()
    }
    pub fn y(&self, i: usize) -> f64 {
        self.y_min + (i + 1) as f64 * self.hy()
    }
    pub fn z(&self, i: usize) -> f64 {
        self.z_min + (i + 1) as f64 * self.hz()
    }
    pub fn idx(&self, ix: usize, iy: usize, iz: usize) -> usize {
        (iz * self.ny + iy) * self.nx + ix
    }
    pub fn len(&self) -> usize {
        self.nx * self.ny * self.nz
    }
    /// Always false for a grid built via `new` (every axis has n >= 1).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Cell volume, turning a sum into an integral.
    pub fn cell(&self) -> f64 {
        self.hx() * self.hy() * self.hz()
    }

    /// Points along `axis`, and the flat-index stride between them.
    fn line(&self, axis: Axis) -> (usize, usize) {
        match axis {
            Axis::X => (self.nx, 1),
            Axis::Y => (self.ny, self.nx),
            Axis::Z => (self.nz, self.nx * self.ny),
        }
    }

    /// The flat index of the start of line number `l` along `axis`.
    ///
    /// Lines are enumerated over the *other* two axes; getting this
    /// wrong is invisible on a cubic grid, which is why the tests use a
    /// grid with three different extents.
    fn line_base(&self, axis: Axis, l: usize) -> usize {
        match axis {
            // l enumerates (iy, iz)
            Axis::X => {
                let iy = l % self.ny;
                let iz = l / self.ny;
                self.idx(0, iy, iz)
            }
            // l enumerates (ix, iz)
            Axis::Y => {
                let ix = l % self.nx;
                let iz = l / self.nx;
                self.idx(ix, 0, iz)
            }
            // l enumerates (ix, iy)
            Axis::Z => {
                let ix = l % self.nx;
                let iy = l / self.nx;
                self.idx(ix, iy, 0)
            }
        }
    }

    /// How many lines run along `axis`.
    fn line_count(&self, axis: Axis) -> usize {
        match axis {
            Axis::X => self.ny * self.nz,
            Axis::Y => self.nx * self.nz,
            Axis::Z => self.nx * self.ny,
        }
    }
}

/// A discretised 3-D Hamiltonian, seven-point stencil.
#[derive(Clone, Debug)]
pub struct Hamiltonian3 {
    pub grid: Grid3,
    pub potential: Vec<f64>,
    pub mass: f64,
    pub hbar: f64,
    pub absorber: Option<Vec<f64>>,
}

impl Hamiltonian3 {
    /// # Errors
    /// Length mismatch, non-finite potential, non-positive mass/hbar.
    pub fn new(grid: Grid3, potential: Vec<f64>, mass: f64, hbar: f64) -> Result<Self, String> {
        if potential.len() != grid.len() {
            return Err(format!(
                "Hamiltonian3: potential has {} values but the grid has {}",
                potential.len(),
                grid.len()
            ));
        }
        if potential.iter().any(|v| !v.is_finite()) {
            return Err("Hamiltonian3: the potential has a non-finite value".to_string());
        }
        if !mass.is_finite() || mass <= 0.0 {
            return Err(format!("Hamiltonian3: mass must be finite and positive, got {mass}"));
        }
        if !hbar.is_finite() || hbar <= 0.0 {
            return Err(format!("Hamiltonian3: hbar must be finite and positive, got {hbar}"));
        }
        Ok(Self { grid, potential, mass, hbar, absorber: None })
    }

    /// Sample a closure of `(x, y, z)`.
    ///
    /// # Errors
    /// As [`Hamiltonian3::new`].
    pub fn from_fn<F: Fn(f64, f64, f64) -> f64>(
        grid: Grid3,
        v: F,
        mass: f64,
        hbar: f64,
    ) -> Result<Self, String> {
        let mut p = Vec::with_capacity(grid.len());
        for iz in 0..grid.nz {
            for iy in 0..grid.ny {
                for ix in 0..grid.nx {
                    p.push(v(grid.x(ix), grid.y(iy), grid.z(iz)));
                }
            }
        }
        Self::new(grid, p, mass, hbar)
    }

    /// Absorbing faces on all six sides. `W` is the largest of the three
    /// per-axis ramps, so an edge or corner absorbs as strongly as a
    /// face rather than two or three times as strongly.
    ///
    /// # Errors
    /// Non-positive `width`/`strength`, `power < 1`, or a width that
    /// does not fit on every axis.
    pub fn with_absorber(mut self, width: f64, strength: f64, power: f64) -> Result<Self, String> {
        if !width.is_finite() || width <= 0.0 || !strength.is_finite() || strength <= 0.0 {
            return Err("with_absorber: width and strength must be finite and positive".to_string());
        }
        if !power.is_finite() || power < 1.0 {
            return Err(format!("with_absorber: power must be at least 1, got {power}"));
        }
        let g = &self.grid;
        for (lo, hi, ax) in [
            (g.x_min, g.x_max, "x"),
            (g.y_min, g.y_max, "y"),
            (g.z_min, g.z_max, "z"),
        ] {
            if 2.0 * width >= hi - lo {
                return Err(format!(
                    "with_absorber: width {width} does not fit on the {ax} axis"
                ));
            }
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
        for iz in 0..g.nz {
            for iy in 0..g.ny {
                for ix in 0..g.nx {
                    let a = ramp(g.x(ix), g.x_min, g.x_max);
                    let b = ramp(g.y(iy), g.y_min, g.y_max);
                    let c = ramp(g.z(iz), g.z_min, g.z_max);
                    w.push(strength * a.max(b).max(c));
                }
            }
        }
        self.absorber = Some(w);
        Ok(self)
    }

    pub fn is_absorbing(&self) -> bool {
        self.absorber.is_some()
    }

    fn off(&self, axis: Axis) -> f64 {
        let h = match axis {
            Axis::X => self.grid.hx(),
            Axis::Y => self.grid.hy(),
            Axis::Z => self.grid.hz(),
        };
        -self.hbar * self.hbar / (2.0 * self.mass * h * h)
    }

    /// `H psi` for a real vector — the seven-point stencil, for the
    /// eigensolver.
    pub fn apply_real(&self, psi: &[f64]) -> Vec<f64> {
        let g = &self.grid;
        let (ox, oy, oz) = (self.off(Axis::X), self.off(Axis::Y), self.off(Axis::Z));
        let diag0 = -2.0 * (ox + oy + oz);
        let mut out = vec![0.0; g.len()];
        for iz in 0..g.nz {
            for iy in 0..g.ny {
                for ix in 0..g.nx {
                    let k = g.idx(ix, iy, iz);
                    let mut s = (diag0 + self.potential[k]) * psi[k];
                    if ix > 0 {
                        s += ox * psi[k - 1];
                    }
                    if ix + 1 < g.nx {
                        s += ox * psi[k + 1];
                    }
                    if iy > 0 {
                        s += oy * psi[k - g.nx];
                    }
                    if iy + 1 < g.ny {
                        s += oy * psi[k + g.nx];
                    }
                    if iz > 0 {
                        s += oz * psi[k - g.nx * g.ny];
                    }
                    if iz + 1 < g.nz {
                        s += oz * psi[k + g.nx * g.ny];
                    }
                    out[k] = s;
                }
            }
        }
        out
    }

    /// `H psi` for a complex vector, including any absorber.
    pub fn apply(&self, psi: &[C]) -> Vec<C> {
        let g = &self.grid;
        let (ox, oy, oz) = (self.off(Axis::X), self.off(Axis::Y), self.off(Axis::Z));
        let diag0 = -2.0 * (ox + oy + oz);
        let mut out = vec![C::ZERO; g.len()];
        for iz in 0..g.nz {
            for iy in 0..g.ny {
                for ix in 0..g.nx {
                    let k = g.idx(ix, iy, iz);
                    let d = match &self.absorber {
                        Some(w) => C::new(diag0 + self.potential[k], -w[k]),
                        None => C::real(diag0 + self.potential[k]),
                    };
                    let mut s = d * psi[k];
                    if ix > 0 {
                        s = s + psi[k - 1] * ox;
                    }
                    if ix + 1 < g.nx {
                        s = s + psi[k + 1] * ox;
                    }
                    if iy > 0 {
                        s = s + psi[k - g.nx] * oy;
                    }
                    if iy + 1 < g.ny {
                        s = s + psi[k + g.nx] * oy;
                    }
                    if iz > 0 {
                        s = s + psi[k - g.nx * g.ny] * oz;
                    }
                    if iz + 1 < g.nz {
                        s = s + psi[k + g.nx * g.ny] * oz;
                    }
                    out[k] = s;
                }
            }
        }
        out
    }

    /// Diagonal of `A_d = T_d + V/3` (plus a third of the absorber).
    fn diag_dir(&self, k: usize, off: f64) -> C {
        let re = -2.0 * off + self.potential[k] / 3.0;
        match &self.absorber {
            Some(w) => C::new(re, -w[k] / 3.0),
            None => C::real(re),
        }
    }

    /// The lowest `k` bound states, by matrix-free Lanczos.
    ///
    /// # Cost
    /// The eigensolver's full reorthogonalisation is `O(m^2 n)` and it
    /// stores the whole Krylov basis, so this is practical to roughly
    /// **40³** and not beyond. Propagation has no such limit.
    ///
    /// Pass `max_iters = 0` to let the solver choose its own Krylov
    /// budget, scaling with `k` and capped at the grid size; any
    /// positive value overrides it.
    ///
    /// # Errors
    /// `k == 0` or `k > n`, an absorbing Hamiltonian, or a Lanczos
    /// failure.
    pub fn bound_states(&self, k: usize, max_iters: usize) -> Result<BoundStates3, String> {
        if self.is_absorbing() {
            return Err(
                "bound_states: an absorbing potential makes the Hamiltonian NON-Hermitian, so a \
                 symmetric eigensolver would return confident nonsense. Remove the absorber."
                    .to_string(),
            );
        }
        let n = self.grid.len();
        let budget = if max_iters > 0 { max_iters } else { (30 * k + 200).min(n) };
        let r = lanczos_lowest(n, k, |v| self.apply_real(v), 1e-7, budget)?;
        let s = 1.0 / self.grid.cell().sqrt();
        let states = r
            .vectors
            .into_iter()
            .map(|mut v| {
                for x in v.iter_mut() {
                    *x *= s;
                }
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
        Ok(BoundStates3 {
            energies: r.values,
            states,
            residuals: r.residuals,
            iterations: r.iterations,
            converged: r.stop != Stop::MaxIters,
        })
    }
}

/// What [`Hamiltonian3::bound_states`] returns.
#[derive(Clone, Debug)]
pub struct BoundStates3 {
    pub energies: Vec<f64>,
    pub states: Vec<Vec<f64>>,
    pub residuals: Vec<f64>,
    pub iterations: usize,
    pub converged: bool,
}

/// Lift a real amplitude to a complex one, so callers need not depend
/// on the complex type's spelling.
pub fn real_to_complex(v: &f64) -> C {
    C::real(*v)
}

/// A 3-D wavefunction.
#[derive(Clone, Debug)]
pub struct Wavefunction3 {
    pub grid: Grid3,
    pub psi: Vec<C>,
}

impl Wavefunction3 {
    /// # Errors
    /// Length mismatch.
    pub fn new(grid: Grid3, psi: Vec<C>) -> Result<Self, String> {
        if psi.len() != grid.len() {
            return Err(format!(
                "Wavefunction3: {} values for a grid of {}",
                psi.len(),
                grid.len()
            ));
        }
        Ok(Self { grid, psi })
    }

    /// A normalised 3-D Gaussian packet.
    ///
    /// # Errors
    /// Non-positive widths, non-finite parameters, or a vanishing norm.
    #[allow(clippy::too_many_arguments)]
    pub fn gaussian(
        grid: Grid3,
        centre: (f64, f64, f64),
        sigma: (f64, f64, f64),
        k: (f64, f64, f64),
    ) -> Result<Self, String> {
        let (x0, y0, z0) = centre;
        let (sx, sy, sz) = sigma;
        let (kx, ky, kz) = k;
        if ![sx, sy, sz].iter().all(|s| s.is_finite() && *s > 0.0) {
            return Err("gaussian: every sigma must be finite and positive".to_string());
        }
        if ![x0, y0, z0, kx, ky, kz].iter().all(|v| v.is_finite()) {
            return Err("gaussian: centre and momentum must be finite".to_string());
        }
        let mut psi = Vec::with_capacity(grid.len());
        for iz in 0..grid.nz {
            let z = grid.z(iz);
            for iy in 0..grid.ny {
                let y = grid.y(iy);
                for ix in 0..grid.nx {
                    let x = grid.x(ix);
                    let a = (x - x0) * (x - x0) / (4.0 * sx * sx)
                        + (y - y0) * (y - y0) / (4.0 * sy * sy)
                        + (z - z0) * (z - z0) / (4.0 * sz * sz);
                    psi.push(C::from_polar((-a).exp(), kx * x + ky * y + kz * z));
                }
            }
        }
        let mut w = Self::new(grid, psi)?;
        w.normalise()?;
        Ok(w)
    }

    /// `integral |psi|^2 dV`.
    pub fn norm(&self) -> f64 {
        self.psi.iter().map(|z| z.norm_sqr()).sum::<f64>() * self.grid.cell()
    }

    /// # Errors
    /// Zero or non-finite norm.
    pub fn normalise(&mut self) -> Result<(), String> {
        let n = self.norm();
        if !n.is_finite() || n <= 0.0 {
            return Err(format!("normalise: the norm is {n}"));
        }
        let s = 1.0 / n.sqrt();
        for z in self.psi.iter_mut() {
            *z = *z * s;
        }
        Ok(())
    }

    /// `(<x>, <y>, <z>)`.
    pub fn centroid(&self) -> (f64, f64, f64) {
        let g = &self.grid;
        let (mut sx, mut sy, mut sz) = (0.0, 0.0, 0.0);
        for iz in 0..g.nz {
            let z = g.z(iz);
            for iy in 0..g.ny {
                let y = g.y(iy);
                for ix in 0..g.nx {
                    let p = self.psi[g.idx(ix, iy, iz)].norm_sqr();
                    sx += g.x(ix) * p;
                    sy += y * p;
                    sz += z * p;
                }
            }
        }
        let c = g.cell();
        (sx * c, sy * c, sz * c)
    }

    /// `<H> = <psi|H|psi> / <psi|psi>`, divided by the norm for the same
    /// reason as in lower dimensions.
    pub fn energy(&self, ham: &Hamiltonian3) -> f64 {
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

    /// Probability inside a box.
    pub fn probability_in(
        &self,
        xr: (f64, f64),
        yr: (f64, f64),
        zr: (f64, f64),
    ) -> f64 {
        let g = &self.grid;
        let ord = |(a, b): (f64, f64)| if a <= b { (a, b) } else { (b, a) };
        let (x0, x1) = ord(xr);
        let (y0, y1) = ord(yr);
        let (z0, z1) = ord(zr);
        let mut s = 0.0;
        for iz in 0..g.nz {
            let z = g.z(iz);
            if z < z0 || z > z1 {
                continue;
            }
            for iy in 0..g.ny {
                let y = g.y(iy);
                if y < y0 || y > y1 {
                    continue;
                }
                for ix in 0..g.nx {
                    let x = g.x(ix);
                    if x >= x0 && x <= x1 {
                        s += self.psi[g.idx(ix, iy, iz)].norm_sqr();
                    }
                }
            }
        }
        s * g.cell()
    }

    /// `|psi|^2` everywhere.
    pub fn density(&self) -> Vec<f64> {
        self.psi.iter().map(|z| z.norm_sqr()).collect()
    }

    /// A **marginal probability density**: the density integrated over
    /// the axis named by `over`.
    ///
    /// These are the honest way to look at a volume. `P(x, y) =
    /// integral |psi|^2 dz` is the probability of finding the particle
    /// at `(x, y)` whatever its `z` — a real observable, not a rendering
    /// convention — and each marginal integrates to the full norm,
    /// which the tests check.
    ///
    /// Returned row-major over the two remaining axes, in their natural
    /// order: `Axis::Z` gives `P(x, y)` indexed `iy * nx + ix`,
    /// `Axis::Y` gives `P(x, z)` indexed `iz * nx + ix`, and `Axis::X`
    /// gives `P(y, z)` indexed `iz * ny + iy`.
    pub fn marginal(&self, over: Axis) -> Vec<f64> {
        let g = &self.grid;
        let (n1, n2) = match over {
            Axis::Z => (g.nx, g.ny),
            Axis::Y => (g.nx, g.nz),
            Axis::X => (g.ny, g.nz),
        };
        // the element of the integrated axis
        let dh = match over {
            Axis::X => g.hx(),
            Axis::Y => g.hy(),
            Axis::Z => g.hz(),
        };
        let mut out = vec![0.0; n1 * n2];
        for iz in 0..g.nz {
            for iy in 0..g.ny {
                for ix in 0..g.nx {
                    let p = self.psi[g.idx(ix, iy, iz)].norm_sqr();
                    let k = match over {
                        Axis::Z => iy * n1 + ix,
                        Axis::Y => iz * n1 + ix,
                        Axis::X => iz * n1 + iy,
                    };
                    out[k] += p;
                }
            }
        }
        for v in out.iter_mut() {
            *v *= dh;
        }
        out
    }

    /// Probability within `frac` of the domain size of any face.
    pub fn edge_probability(&self, frac: f64) -> f64 {
        let g = &self.grid;
        let (wx, wy, wz) = (
            (g.x_max - g.x_min) * frac,
            (g.y_max - g.y_min) * frac,
            (g.z_max - g.z_min) * frac,
        );
        let mut s = 0.0;
        for iz in 0..g.nz {
            let z = g.z(iz);
            for iy in 0..g.ny {
                let y = g.y(iy);
                for ix in 0..g.nx {
                    let x = g.x(ix);
                    if x < g.x_min + wx
                        || x > g.x_max - wx
                        || y < g.y_min + wy
                        || y > g.y_max - wy
                        || z < g.z_min + wz
                        || z > g.z_max - wz
                    {
                        s += self.psi[g.idx(ix, iy, iz)].norm_sqr();
                    }
                }
            }
        }
        s * g.cell()
    }
}

/// A 3-D ADI propagator: `U_x(dt/2) U_y(dt/2) U_z(dt) U_y(dt/2) U_x(dt/2)`.
///
/// Every factor is exactly unitary, so the norm is conserved to machine
/// precision for any `dt`; the splitting error is `O(dt^2)` in the
/// dynamics.
pub struct Propagator3 {
    ham: Hamiltonian3,
    dt: f64,
}

impl Propagator3 {
    /// # Errors
    /// A non-finite or zero `dt`.
    pub fn new(ham: Hamiltonian3, dt: f64) -> Result<Self, String> {
        if !dt.is_finite() || dt == 0.0 {
            return Err(format!("Propagator3: dt must be finite and non-zero, got {dt}"));
        }
        Ok(Self { ham, dt })
    }

    pub fn dt(&self) -> f64 {
        self.dt
    }
    pub fn hamiltonian(&self) -> &Hamiltonian3 {
        &self.ham
    }

    /// One directional Cayley factor, over every line along `axis`.
    ///
    /// Bands are built per line rather than stored for the whole grid —
    /// see the module note on memory.
    fn apply_dir(&self, w: &mut Wavefunction3, axis: Axis, tau: f64) -> Result<(), String> {
        let g = self.ham.grid.clone();
        let (len, stride) = g.line(axis);
        let off = self.ham.off(axis);
        let half = C::I * (tau / (2.0 * self.ham.hbar));
        let o = half * C::real(off);

        let sub = vec![o; len];
        let sup = vec![o; len];
        let mut diag = vec![C::ZERO; len];
        let mut rhs = vec![C::ZERO; len];

        for l in 0..g.line_count(axis) {
            let base = g.line_base(axis, l);
            for j in 0..len {
                let k = base + j * stride;
                let a_diag = self.ham.diag_dir(k, off);
                diag[j] = C::ONE + half * a_diag;
                let mut ap = a_diag * w.psi[k];
                if j > 0 {
                    ap = ap + w.psi[k - stride] * off;
                }
                if j + 1 < len {
                    ap = ap + w.psi[k + stride] * off;
                }
                rhs[j] = w.psi[k] - half * ap;
            }
            let sol = solve_tridiag_c(&sub, &diag, &sup, &rhs)?;
            for (j, v) in sol.into_iter().enumerate() {
                w.psi[base + j * stride] = v;
            }
        }
        Ok(())
    }

    /// Advance by one step.
    ///
    /// # Errors
    /// Grid mismatch or a solve failure.
    pub fn step(&self, w: &mut Wavefunction3) -> Result<(), String> {
        if w.grid != self.ham.grid {
            return Err("step: the wavefunction and propagator use different grids".to_string());
        }
        let h = self.dt / 2.0;
        self.apply_dir(w, Axis::X, h)?;
        self.apply_dir(w, Axis::Y, h)?;
        self.apply_dir(w, Axis::Z, self.dt)?;
        self.apply_dir(w, Axis::Y, h)?;
        self.apply_dir(w, Axis::X, h)?;
        Ok(())
    }

    /// `steps` steps.
    ///
    /// # Errors
    /// As [`Propagator3::step`].
    pub fn run(&self, w: &mut Wavefunction3, steps: usize) -> Result<(), String> {
        for _ in 0..steps {
            self.step(w)?;
        }
        Ok(())
    }
}

/// A 3-D propagator for `H(t) = H_0 + f(t) g(x, y, z)`, matching the
/// 1-D and 2-D drives: a fixed spatial shape, a scalar modulation
/// sampled at the step midpoint.
pub struct DrivenPropagator3 {
    ham: Hamiltonian3,
    shape: Vec<f64>,
    dt: f64,
    time: f64,
}

impl DrivenPropagator3 {
    /// # Errors
    /// Shape length mismatch, non-finite shape, or a zero `dt`.
    pub fn new(ham: Hamiltonian3, shape: Vec<f64>, dt: f64) -> Result<Self, String> {
        if shape.len() != ham.grid.len() {
            return Err(format!(
                "DrivenPropagator3: the drive shape has {} values but the grid has {}",
                shape.len(),
                ham.grid.len()
            ));
        }
        if shape.iter().any(|v| !v.is_finite()) {
            return Err("DrivenPropagator3: the drive shape has a non-finite value".to_string());
        }
        if !dt.is_finite() || dt == 0.0 {
            return Err(format!("DrivenPropagator3: dt must be finite and non-zero, got {dt}"));
        }
        Ok(Self { ham, shape, dt, time: 0.0 })
    }

    pub fn time(&self) -> f64 {
        self.time
    }
    pub fn dt(&self) -> f64 {
        self.dt
    }

    fn hamiltonian_at(&self, amp: f64) -> Result<Hamiltonian3, String> {
        if !amp.is_finite() {
            return Err(format!("DrivenPropagator3: the modulation returned {amp}"));
        }
        let v: Vec<f64> = self
            .ham
            .potential
            .iter()
            .zip(&self.shape)
            .map(|(v0, g)| v0 + amp * g)
            .collect();
        let mut h = Hamiltonian3::new(
            self.ham.grid.clone(),
            v,
            self.ham.mass,
            self.ham.hbar,
        )?;
        h.absorber = self.ham.absorber.clone();
        Ok(h)
    }

    /// One step, modulation taken at the midpoint.
    ///
    /// # Errors
    /// Grid mismatch, non-finite modulation, or a solve failure.
    pub fn step<F: Fn(f64) -> f64>(
        &mut self,
        w: &mut Wavefunction3,
        modulation: F,
    ) -> Result<(), String> {
        if w.grid != self.ham.grid {
            return Err("step: the wavefunction and propagator use different grids".to_string());
        }
        let amp = modulation(self.time + 0.5 * self.dt);
        Propagator3::new(self.hamiltonian_at(amp)?, self.dt)?.step(w)?;
        self.time += self.dt;
        Ok(())
    }

    /// `steps` steps.
    ///
    /// # Errors
    /// As [`DrivenPropagator3::step`].
    pub fn run<F: Fn(f64) -> f64 + Copy>(
        &mut self,
        w: &mut Wavefunction3,
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

    fn cube(n: usize, a: f64) -> Grid3 {
        Grid3::new(-a, a, n, -a, a, n, -a, a, n).unwrap()
    }

    /// Unitarity for ANY step size — the sharp check, as in 1-D and 2-D.
    #[test]
    fn adi_3d_is_unitary_at_any_step_size() {
        for &dt in &[0.002, 0.05, 1.0, 15.0] {
            let g = cube(20, 6.0);
            let ham = Hamiltonian3::from_fn(
                g.clone(),
                |x, y, z| 0.5 * (x * x + y * y + z * z),
                1.0,
                1.0,
            )
            .unwrap();
            let mut w = Wavefunction3::gaussian(
                g,
                (-1.0, 0.5, 0.3),
                (1.0, 1.0, 1.0),
                (0.7, -0.4, 0.2),
            )
            .unwrap();
            let n0 = w.norm();
            Propagator3::new(ham, dt).unwrap().run(&mut w, 20).unwrap();
            assert!(
                (w.norm() / n0 - 1.0).abs() < 1e-10,
                "dt = {dt}: norm {n0} -> {}",
                w.norm()
            );
        }
    }

    /// A free packet drifts along all three axes at the discrete group
    /// velocity, independently.
    #[test]
    fn a_free_packet_drifts_in_three_directions() {
        let g = Grid3::new(-18.0, 18.0, 60, -18.0, 18.0, 60, -18.0, 18.0, 60).unwrap();
        let ham = Hamiltonian3::from_fn(g.clone(), |_, _, _| 0.0, 1.0, 1.0).unwrap();
        let k = (1.2_f64, -0.8_f64, 0.5_f64);
        let mut w = Wavefunction3::gaussian(g.clone(), (-4.0, 2.0, -1.0), (2.0, 2.0, 2.0), k)
            .unwrap();
        let (x0, y0, z0) = w.centroid();
        let (dt, steps) = (0.01_f64, 300usize);
        Propagator3::new(ham, dt).unwrap().run(&mut w, steps).unwrap();
        let t = dt * steps as f64;
        let (x1, y1, z1) = w.centroid();
        let v = |kk: f64, h: f64| (kk * h).sin() / h;
        assert!((x1 - (x0 + v(k.0, g.hx()) * t)).abs() < 0.1, "<x> = {x1}");
        assert!((y1 - (y0 + v(k.1, g.hy()) * t)).abs() < 0.1, "<y> = {y1}");
        assert!((z1 - (z0 + v(k.2, g.hz()) * t)).abs() < 0.1, "<z> = {z1}");
        assert!(w.edge_probability(0.05) < 1e-9, "the packet reached a face");
    }

    /// The 3-D isotropic oscillator ground state is the product of three
    /// 1-D ground states, with `E = 3/2`. It must be stationary, which
    /// couples the eigensolver, the stencil and the propagator.
    #[test]
    fn the_3d_oscillator_ground_state_is_stationary() {
        let n = 24;
        let a = 6.0;
        let g1 = qm1d::Grid::new(-a, a, n).unwrap();
        let h1 = qm1d::Hamiltonian::from_fn(g1, |x| 0.5 * x * x, 1.0, 1.0).unwrap();
        let (e1, s1) = h1.bound_states(1).unwrap();

        let g = cube(n, a);
        let ham =
            Hamiltonian3::from_fn(g.clone(), |x, y, z| 0.5 * (x * x + y * y + z * z), 1.0, 1.0)
                .unwrap();
        let mut psi = Vec::with_capacity(g.len());
        for iz in 0..g.nz {
            for iy in 0..g.ny {
                for ix in 0..g.nx {
                    psi.push(C::real(s1[0][ix] * s1[0][iy] * s1[0][iz]));
                }
            }
        }
        let mut w = Wavefunction3::new(g, psi).unwrap();
        w.normalise().unwrap();

        let e = w.energy(&ham);
        // The EXACT statement: on this grid the 3-D energy is three
        // times the 1-D one, and that holds to rounding.
        assert!((e - 3.0 * e1[0]).abs() < 1e-6, "E = {e}, want {}", 3.0 * e1[0]);
        // The continuum value 3/2 is only approached as h -> 0. Here
        // h = 0.48, which is coarse on purpose to keep the test quick,
        // so the offset is ~0.022 of second-order discretisation error.
        // Asserting 1.5 tightly would be asserting a number the
        // discretisation is not trying to produce.
        assert!((e - 1.5).abs() < 0.04, "E = {e}, want ~1.5 within grid error");

        let d0 = w.density();
        Propagator3::new(ham.clone(), 0.01).unwrap().run(&mut w, 60).unwrap();
        let worst = d0
            .iter()
            .zip(&w.density())
            .map(|(p, q)| (p - q).abs())
            .fold(0.0_f64, f64::max);
        assert!(worst < 1e-6, "the ground-state density moved by {worst}");
        assert!((w.energy(&ham) - e).abs() < 1e-8, "energy drifted");
    }

    /// Bound states in 3-D: the isotropic oscillator has
    /// `E = nx+ny+nz+3/2`, so the first excited level is **three-fold
    /// degenerate**. That degeneracy is the point — it is what deflation
    /// in the Lanczos solver exists to resolve.
    ///
    /// Checked against the EXACT discrete spectrum, obtained by summing
    /// 1-D eigenvalues from the (unrelated) dense solver on the same
    /// grid. Comparing against the continuum 1.5 and 2.5 would be
    /// comparing against numbers this grid is not trying to produce.
    #[test]
    fn the_3d_oscillator_spectrum_with_its_degeneracy() {
        let (a, n) = (5.5_f64, 22usize);
        let g1 = qm1d::Grid::new(-a, a, n).unwrap();
        let e1 = qm1d::Hamiltonian::from_fn(g1, |x| 0.5 * x * x, 1.0, 1.0)
            .unwrap()
            .bound_states(2)
            .unwrap()
            .0;
        let want = [
            3.0 * e1[0],                 // (0,0,0)
            2.0 * e1[0] + e1[1],         // (1,0,0) and its two partners
            2.0 * e1[0] + e1[1],
            2.0 * e1[0] + e1[1],
        ];

        let g = cube(n, a);
        let ham =
            Hamiltonian3::from_fn(g.clone(), |x, y, z| 0.5 * (x * x + y * y + z * z), 1.0, 1.0)
                .unwrap();
        let b = ham.bound_states(4, 260).unwrap();
        for (j, wj) in want.iter().enumerate() {
            assert!(
                (b.energies[j] - wj).abs() < 1e-5,
                "E[{j}] = {} vs exact discrete {wj} (all: {:?})",
                b.energies[j],
                b.energies
            );
        }
        // the three degenerate partners must agree with EACH OTHER far
        // more closely than with anything else
        let spread = b.energies[3] - b.energies[1];
        assert!(spread.abs() < 1e-9, "the degenerate triplet split by {spread}");
        // and they must be genuinely distinct states
        let cell = g.cell();
        for i in 1..4 {
            for j in 1..4 {
                let ip: f64 =
                    b.states[i].iter().zip(&b.states[j]).map(|(p, q)| p * q).sum::<f64>() * cell;
                let target = if i == j { 1.0 } else { 0.0 };
                assert!((ip - target).abs() < 1e-5, "<{i}|{j}> = {ip}, want {target}");
            }
        }
    }

    /// A separable 3-D problem must give the sum of three 1-D spectra —
    /// a cross-check against an entirely different solver (dense Jacobi
    /// in 1-D versus matrix-free Lanczos here).
    #[test]
    fn a_separable_3d_ground_state_matches_the_1d_sum() {
        let (a, n) = (6.0_f64, 22usize);
        // three DIFFERENT frequencies, so nothing is degenerate and the
        // pairing is unambiguous
        let (wx, wy, wz) = (1.0_f64, 2.0_f64, 3.0_f64);
        let e1 = |om: f64| {
            let g1 = qm1d::Grid::new(-a, a, n).unwrap();
            qm1d::Hamiltonian::from_fn(g1, move |x| 0.5 * om * om * x * x, 1.0, 1.0)
                .unwrap()
                .bound_states(1)
                .unwrap()
                .0[0]
        };
        let want = e1(wx) + e1(wy) + e1(wz);

        let g = cube(n, a);
        let ham = Hamiltonian3::from_fn(
            g,
            move |x, y, z| 0.5 * (wx * wx * x * x + wy * wy * y * y + wz * wz * z * z),
            1.0,
            1.0,
        )
        .unwrap();
        let b = ham.bound_states(1, 260).unwrap();
        assert!(
            (b.energies[0] - want).abs() < 1e-5,
            "E0 = {} vs 1-D sum {want}",
            b.energies[0]
        );
    }

    /// A z-independent problem is really 2-D. Cross-checking the 3-D
    /// propagator against the 2-D one is the strongest correctness test
    /// available: the two share only the tridiagonal solver.
    #[test]
    fn a_z_independent_problem_matches_the_2d_solver() {
        use crate::qm2d;
        let v = |x: f64| if (0.0..1.0).contains(&x) { 2.0 } else { 0.0 };
        let (nx, a) = (80usize, 20.0_f64);

        let g2 = qm2d::Grid2::new(-a, a, nx, -a, a, 20).unwrap();
        let h2 = qm2d::Hamiltonian2::from_fn(g2.clone(), |x, _| v(x), 1.0, 1.0).unwrap();
        let mut w2 =
            qm2d::Wavefunction2::gaussian(g2, -8.0, 0.0, 1.5, 100.0, 2.0, 0.0).unwrap();
        qm2d::Propagator2::new(h2, 0.01).unwrap().run(&mut w2, 400).unwrap();
        let t2 = w2.probability_in(1.0, a, -a, a);

        let g3 = Grid3::new(-a, a, nx, -a, a, 20, -a, a, 20).unwrap();
        let h3 = Hamiltonian3::from_fn(g3.clone(), |x, _, _| v(x), 1.0, 1.0).unwrap();
        let mut w3 = Wavefunction3::gaussian(
            g3,
            (-8.0, 0.0, 0.0),
            (1.5, 100.0, 100.0),
            (2.0, 0.0, 0.0),
        )
        .unwrap();
        Propagator3::new(h3, 0.01).unwrap().run(&mut w3, 400).unwrap();
        let t3 = w3.probability_in((1.0, a), (-a, a), (-a, a));

        let rel = (t3 - t2).abs() / t2;
        assert!(rel < 0.02, "3-D gave T = {t3}, 2-D reference {t2} ({:.2}% apart)", 100.0 * rel);
    }

    /// Absorbing faces work in 3-D too.
    ///
    /// The run length is sized from the DISCRETE group velocity
    /// `sin(k h)/h`, not from `k`. At the resolutions a 3-D test can
    /// afford, those differ a lot: here `k = 2` with `h = 0.39` travels
    /// at 1.80, not 2.00, and an earlier version of this test sized the
    /// run from `k` and stopped before the packet had finished arriving.
    #[test]
    fn absorbing_faces_swallow_a_packet() {
        let (a, n) = (8.0_f64, 40usize);
        let g = cube(n, a);
        let h = g.hx();
        let ham = Hamiltonian3::from_fn(g.clone(), |_, _, _| 0.0, 1.0, 1.0)
            .unwrap()
            .with_absorber(3.0, 3.0, 2.0)
            .unwrap();
        assert!(ham.is_absorbing());
        let k = 2.0_f64;
        let mut w =
            Wavefunction3::gaussian(g, (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (k, k, k)).unwrap();
        // travel to the absorber (|x| > a - 3) at the discrete speed,
        // then well past it
        let v = (k * h).sin() / h;
        let travel = (a - 3.0) / v;
        let dt = 0.005;
        let steps = ((3.0 * travel) / dt).round() as usize;
        Propagator3::new(ham, dt).unwrap().run(&mut w, steps).unwrap();
        assert!(w.norm() < 0.05, "absorber left {} of the norm", w.norm());
        assert!(
            w.probability_in((-4.0, 4.0), (-4.0, 4.0), (-4.0, 4.0)) < 1e-3,
            "the absorber reflected back into the interior"
        );
    }

    /// **A non-cubic grid**, because an axis or stride mix-up is
    /// completely invisible on a cube. Three different extents and three
    /// different point counts.
    #[test]
    fn indexing_is_consistent_on_a_non_cubic_grid() {
        let g = Grid3::new(-4.0, 4.0, 13, -2.0, 2.0, 7, -6.0, 6.0, 5).unwrap();
        assert_eq!(g.len(), 13 * 7 * 5);
        assert_eq!(g.idx(12, 6, 4), 13 * 7 * 5 - 1);
        // a potential depending on ONE coordinate must vary only along it
        for (name, f) in [
            ("x", Box::new(|x: f64, _: f64, _: f64| x) as Box<dyn Fn(f64, f64, f64) -> f64>),
            ("y", Box::new(|_: f64, y: f64, _: f64| y)),
            ("z", Box::new(|_: f64, _: f64, z: f64| z)),
        ] {
            let ham = Hamiltonian3::from_fn(g.clone(), &f, 1.0, 1.0).unwrap();
            for iz in 0..g.nz {
                for iy in 0..g.ny {
                    for ix in 0..g.nx {
                        let want = f(g.x(ix), g.y(iy), g.z(iz));
                        let got = ham.potential[g.idx(ix, iy, iz)];
                        assert!(
                            (got - want).abs() < 1e-12,
                            "{name}: V({ix},{iy},{iz}) = {got}, want {want}"
                        );
                    }
                }
            }
        }
        // every line along every axis must have the right length and
        // stay inside the grid
        for axis in [Axis::X, Axis::Y, Axis::Z] {
            let (len, stride) = g.line(axis);
            for l in 0..g.line_count(axis) {
                let base = g.line_base(axis, l);
                let last = base + (len - 1) * stride;
                assert!(last < g.len(), "{axis:?} line {l} runs off the end");
            }
            // the lines must together cover every point exactly once
            let mut seen = vec![0u8; g.len()];
            for l in 0..g.line_count(axis) {
                let base = g.line_base(axis, l);
                for j in 0..len {
                    seen[base + j * stride] += 1;
                }
            }
            assert!(
                seen.iter().all(|&c| c == 1),
                "{axis:?} lines do not tile the grid exactly once"
            );
        }
        // and propagation runs
        let ham = Hamiltonian3::from_fn(g.clone(), |_, _, _| 0.0, 1.0, 1.0).unwrap();
        let mut w =
            Wavefunction3::gaussian(g, (0.0, 0.0, 0.0), (1.0, 0.5, 1.5), (0.3, 0.0, -0.2))
                .unwrap();
        Propagator3::new(ham, 0.01).unwrap().run(&mut w, 10).unwrap();
        assert!((w.norm() - 1.0).abs() < 1e-10);
    }

    /// Every marginal must integrate to the full norm — that is what
    /// makes it a probability density rather than a picture. Checked on
    /// a NON-CUBIC grid, where a wrong axis or a wrong spacing factor
    /// would show up.
    #[test]
    fn marginals_are_probability_densities() {
        let g = Grid3::new(-5.0, 5.0, 17, -3.0, 3.0, 11, -4.0, 4.0, 13).unwrap();
        let w = Wavefunction3::gaussian(
            g.clone(),
            (0.5, -0.3, 0.2),
            (1.0, 0.8, 1.2),
            (0.4, -0.2, 0.3),
        )
        .unwrap();
        let total = w.norm();
        assert!((total - 1.0).abs() < 1e-12);

        for (over, (n1, n2), (d1, d2)) in [
            (Axis::Z, (g.nx, g.ny), (g.hx(), g.hy())),
            (Axis::Y, (g.nx, g.nz), (g.hx(), g.hz())),
            (Axis::X, (g.ny, g.nz), (g.hy(), g.hz())),
        ] {
            let m = w.marginal(over);
            assert_eq!(m.len(), n1 * n2, "{over:?} marginal has the wrong shape");
            let sum: f64 = m.iter().sum::<f64>() * d1 * d2;
            assert!(
                (sum - total).abs() < 1e-12,
                "{over:?} marginal integrates to {sum}, want {total}"
            );
            assert!(m.iter().all(|v| *v >= 0.0), "a density went negative");
        }
    }

    /// The marginal must peak where the packet is, on the right axes.
    /// An axis swap would put the peak in the wrong place, and on a
    /// non-cubic grid it would also change the shape.
    #[test]
    fn marginals_peak_at_the_packet_centre() {
        let g = Grid3::new(-6.0, 6.0, 25, -6.0, 6.0, 19, -6.0, 6.0, 15).unwrap();
        let (x0, y0, z0) = (2.0_f64, -1.5_f64, 3.0_f64);
        let w = Wavefunction3::gaussian(
            g.clone(),
            (x0, y0, z0),
            (0.8, 0.8, 0.8),
            (0.0, 0.0, 0.0),
        )
        .unwrap();

        // P(x, y): peak at (x0, y0)
        let m = w.marginal(Axis::Z);
        let (mut best, mut at) = (f64::NEG_INFINITY, (0usize, 0usize));
        for iy in 0..g.ny {
            for ix in 0..g.nx {
                let v = m[iy * g.nx + ix];
                if v > best {
                    best = v;
                    at = (ix, iy);
                }
            }
        }
        assert!((g.x(at.0) - x0).abs() < g.hx(), "P(x,y) peaks at x = {}", g.x(at.0));
        assert!((g.y(at.1) - y0).abs() < g.hy(), "P(x,y) peaks at y = {}", g.y(at.1));

        // P(y, z): peak at (y0, z0)
        let m = w.marginal(Axis::X);
        let (mut best, mut at) = (f64::NEG_INFINITY, (0usize, 0usize));
        for iz in 0..g.nz {
            for iy in 0..g.ny {
                let v = m[iz * g.ny + iy];
                if v > best {
                    best = v;
                    at = (iy, iz);
                }
            }
        }
        assert!((g.y(at.0) - y0).abs() < g.hy(), "P(y,z) peaks at y = {}", g.y(at.0));
        assert!((g.z(at.1) - z0).abs() < g.hz(), "P(y,z) peaks at z = {}", g.z(at.1));
    }

    /// The 3-D driven oscillator: Ehrenfest is exact per axis, so a
    /// drive along x must move `<x>` on the classical trajectory and
    /// leave `<y>` and `<z>` at zero. The last part is what would break
    /// if the five ADI sweeps were composed wrongly.
    #[test]
    fn a_3d_drive_moves_only_the_driven_axis() {
        // h = 0.303. The tolerance below is set by this: Ehrenfest is
        // exact in the CONTINUUM, but the discrete Laplacian shifts the
        // oscillator's effective frequency, and the resonance
        // denominator (1 - w^2) is sensitive to that. At h = 0.519 the
        // deviation was 0.064; at h = 0.303 it is about a third of that,
        // consistent with second order. The 1-D test pins the same drive
        // code quantitatively at h = 0.05, where agreement is 0.01.
        let (a, n) = (5.0_f64, 32usize);
        let g = cube(n, a);
        let ham =
            Hamiltonian3::from_fn(g.clone(), |x, y, z| 0.5 * (x * x + y * y + z * z), 1.0, 1.0)
                .unwrap();
        // ground state as a product of 1-D ground states
        let g1 = qm1d::Grid::new(-a, a, n).unwrap();
        let s1 = qm1d::Hamiltonian::from_fn(g1, |x| 0.5 * x * x, 1.0, 1.0)
            .unwrap()
            .bound_states(1)
            .unwrap()
            .1;
        let mut psi = Vec::with_capacity(g.len());
        for iz in 0..g.nz {
            for iy in 0..g.ny {
                for ix in 0..g.nx {
                    psi.push(C::real(s1[0][ix] * s1[0][iy] * s1[0][iz]));
                }
            }
        }
        let mut w = Wavefunction3::new(g.clone(), psi).unwrap();
        w.normalise().unwrap();

        // g(x, y, z) = x
        let mut shape = Vec::with_capacity(g.len());
        for _iz in 0..g.nz {
            for _iy in 0..g.ny {
                for ix in 0..g.nx {
                    shape.push(g.x(ix));
                }
            }
        }
        let (f0, om, dt, steps) = (0.3_f64, 0.7_f64, 0.01_f64, 500usize);
        let mut prop = DrivenPropagator3::new(ham, shape, dt).unwrap();
        prop.run(&mut w, steps, move |t| f0 * (om * t).cos()).unwrap();

        let t = dt * steps as f64;
        let exact = -f0 / (1.0 - om * om) * ((om * t).cos() - t.cos());
        let (cx, cy, cz) = w.centroid();
        assert!(
            (cx - exact).abs() < 0.04,
            "<x> = {cx}, classical {exact} (grid h = {})",
            g.hx()
        );
        assert!(cy.abs() < 1e-9, "<y> = {cy}, must not move");
        assert!(cz.abs() < 1e-9, "<z> = {cz}, must not move");
        assert!((w.norm() - 1.0).abs() < 1e-10, "norm = {}", w.norm());
    }

    #[test]
    fn invalid_input_is_reported() {
        assert!(Grid3::new(0.0, 1.0, 0, 0.0, 1.0, 2, 0.0, 1.0, 2).is_err(), "nx = 0");
        assert!(Grid3::new(1.0, 0.0, 2, 0.0, 1.0, 2, 0.0, 1.0, 2).is_err(), "reversed x");
        assert!(Grid3::new(0.0, 1.0, 2, 0.0, 1.0, 2, 1.0, 0.0, 2).is_err(), "reversed z");
        let g = Grid3::new(0.0, 1.0, 3, 0.0, 1.0, 2, 0.0, 1.0, 2).unwrap();
        assert!(Hamiltonian3::new(g.clone(), vec![0.0; 5], 1.0, 1.0).is_err(), "length");
        assert!(Hamiltonian3::new(g.clone(), vec![0.0; 12], -1.0, 1.0).is_err(), "mass");
        let h = Hamiltonian3::from_fn(g.clone(), |_, _, _| 0.0, 1.0, 1.0).unwrap();
        assert!(Propagator3::new(h.clone(), 0.0).is_err(), "dt = 0");
        assert!(h.clone().with_absorber(1.0, 1.0, 2.0).is_err(), "absorber too wide");
        assert!(h.bound_states(0, 10).is_err(), "k = 0");
        assert!(
            Wavefunction3::gaussian(g, (0.0, 0.0, 0.0), (-1.0, 1.0, 1.0), (0.0, 0.0, 0.0))
                .is_err(),
            "sigma < 0"
        );
    }

    /// The eigensolver must refuse an absorbing Hamiltonian.
    #[test]
    fn bound_states_refuse_an_absorbing_hamiltonian() {
        let g = cube(10, 5.0);
        let ham = Hamiltonian3::from_fn(g, |x, y, z| 0.5 * (x * x + y * y + z * z), 1.0, 1.0)
            .unwrap()
            .with_absorber(1.5, 1.0, 2.0)
            .unwrap();
        assert!(ham.bound_states(2, 100).unwrap_err().contains("Hermitian"));
    }
}
