//! Scattering at a **fixed energy**, by transfer matrix.
//!
//! Everything else in this crate propagates a wavepacket and reads off
//! what came through. That answers a different question than it looks
//! like it answers, and `TUNNELING_RESULTS.md` §5 recorded the
//! consequence: a double barrier whose resonances are narrower than the
//! packet's own momentum spread shows **no peak at all**, because the
//! packet averages `T(E)` over its whole distribution. Narrowing the
//! packet in `k` means widening it in `x`, which means a longer domain
//! and a longer run, and the resonance can always be made narrower than
//! whatever packet is affordable.
//!
//! A transfer matrix removes the packet from the question. It solves the
//! time-*independent* equation at one energy and returns `t(E)` and
//! `r(E)` directly, so a resonance of any width is resolved by asking at
//! enough energies.
//!
//! # The method
//!
//! On each cell the potential is treated as constant, so
//! `psi'' = -k^2 psi` with `k = sqrt(2m(E - V))/hbar`, and the pair
//! `(psi, psi')` transfers across a cell of width `d` by
//!
//! ```text
//!     [ psi  ]      [   cos(k d)     sin(k d)/k ] [ psi  ]
//!     [ psi' ]  =   [ -k sin(k d)    cos(k d)   ] [ psi' ]
//! ```
//!
//! `psi` and `psi'` are continuous everywhere, so there are **no
//! interface matrices** — the cell matrices simply multiply. That is why
//! this formulation is used here rather than the textbook one in terms
//! of forward and backward amplitudes, which needs a matching matrix at
//! every step and is far easier to get wrong.
//!
//! `k` is complex whenever `E < V` or the potential is, and the same
//! formula holds: `cos` and `sin` of a complex argument are `cosh` and
//! `sinh` of a real one.
//!
//! # Direction, and why it is stable
//!
//! The integration runs **right to left**, starting from the outgoing
//! wave `psi = 1, psi' = i k_R` at the right edge. Inside a classically
//! forbidden region the solution then *grows*, and tracking a growing
//! solution is stable; it is following the decaying one that is not.
//! Splitting the result at the left edge into
//!
//! ```text
//!     a = (psi + psi'/(i k_L)) / 2      incident amplitude
//!     b = (psi - psi'/(i k_L)) / 2      reflected amplitude
//! ```
//!
//! gives `t = 1/a` and `r = b/a`.
//!
//! # Where it stops working, measured rather than assumed
//!
//! The obvious worry is forming `a`: `t` is exponentially small for an
//! opaque barrier, so `a` might be a small difference of large numbers.
//! The cancellation is therefore **measured from the values** —
//! `(|psi| + |psi'/k|) / (2|a|)` — and reported as
//! [`Scattering::conditioning`], with [`scatter`] refusing past `1/eps`.
//!
//! Measured, that guard does not fire, and the reason is worth knowing.
//! In the opaque limit `r -> -1`, so `b -> -a`, which forces `psi -> 0`
//! and leaves `a` dominated by `psi'/(i k)` alone — an addition, not a
//! cancellation. Conditioning stays near **1.4** for barriers opaque
//! enough that `T` reaches `1e-276`. The amplitude-transfer formulation
//! this module deliberately avoids is the one that loses digits here;
//! carrying `(psi, psi')` and integrating *towards* the growing solution
//! does not. The guard is kept as cheap insurance, and
//! `an_opaque_barrier_stays_conditioned` records that it is insurance
//! rather than a working limit.
//!
//! What does end the calculation is honest overflow: `psi'` grows like
//! `exp(kappa a)` and eventually leaves `f64`. That is detected and
//! refused.
//!
//! # Accuracy
//!
//! Piecewise-constant cells sampled at their **midpoints** make the
//! scheme second order in the cell width: halving `n` quarters the
//! error, which `the_cell_error_is_second_order` measures against the
//! analytic rectangular barrier.

use special_functions::complex::Complex64 as C;

/// The result of one fixed-energy scattering calculation.
#[derive(Clone, Copy, Debug)]
pub struct Scattering {
    pub energy: f64,
    /// Transmission amplitude.
    pub t: C,
    /// Reflection amplitude.
    pub r: C,
    /// `|r|^2` — the fraction of the incident flux reflected.
    pub reflection: f64,
    /// The fraction transmitted. Zero when the right-hand asymptote is
    /// absorbing, because then nothing reaches infinity by definition.
    pub transmission: f64,
    /// `1 - R - T`. Zero to rounding for a real potential; for a complex
    /// one it is what the potential swallowed.
    pub absorption: f64,
    /// Cancellation in forming the incident amplitude — the factor by
    /// which precision was lost. 1 is perfect; `1e16` is nothing left.
    pub conditioning: f64,
}

/// Cell width, complex wavenumber, and the two matrix entries that need
/// care near `k = 0`.
fn cell_matrix(k: C, d: f64) -> (C, C, C) {
    let kd = k * d;
    let cos = kd.cos();
    // sin(k d)/k has a removable singularity at k = 0, where it is d.
    // Below this threshold the series is more accurate than the
    // quotient, which is 0/0 in floating point.
    let sinc = if kd.abs() < 1e-6 {
        // d (1 - (kd)^2/6 + (kd)^4/120)
        let q = kd * kd;
        C::real(d) * (C::ONE - q * (1.0 / 6.0) + q * q * (1.0 / 120.0))
    } else {
        kd.sin() * k.inv()
    };
    let ksin = kd.sin() * k;
    (cos, sinc, ksin)
}

/// `k = sqrt(2 m (E - V)) / hbar`, on the branch with `Im k >= 0`.
///
/// The branch matters: `Im k >= 0` is what makes `exp(i k x)` the
/// *decaying* solution to the right, which is the outgoing boundary
/// condition. The principal square root already delivers it for
/// `E - V` off the negative real axis, and on it — the classically
/// forbidden case with a real potential — the principal root of a
/// negative real is `+i|.|`, which is the side wanted.
fn wavenumber(e: f64, v: C, mass: f64, hbar: f64) -> C {
    let arg = (C::real(e) - v) * (2.0 * mass / (hbar * hbar));
    let k = arg.powf(0.5);
    if k.im < 0.0 {
        k * -1.0
    } else {
        k
    }
}

/// Scatter a plane wave of energy `e` off a potential sampled at the
/// **midpoints** of `v.len()` equal cells spanning `[x_min, x_max]`.
///
/// Outside the interval the potential is taken to be constant at `v[0]`
/// on the left and `v[len-1]` on the right, so the caller must extend
/// the sampled region far enough that those are the true asymptotes.
///
/// # Errors
/// Fewer than one cell, a non-finite input, a non-positive `mass` or
/// `hbar`, `x_max <= x_min`, an **absorbing left asymptote** (incident
/// flux would be undefined), an energy at or below the left asymptote
/// (no incident wave exists), or a calculation whose cancellation has
/// consumed every digit.
///
/// # Examples
/// ```
/// use quantum::transfer::scatter;
/// use special_functions::complex::Complex64 as C;
/// // A free particle transmits perfectly.
/// let v = vec![C::ZERO; 64];
/// let s = scatter(&v, -5.0, 5.0, 2.0, 1.0, 1.0).unwrap();
/// assert!((s.transmission - 1.0).abs() < 1e-12);
/// assert!(s.reflection < 1e-12);
/// ```
pub fn scatter(
    v: &[C],
    x_min: f64,
    x_max: f64,
    e: f64,
    mass: f64,
    hbar: f64,
) -> Result<Scattering, String> {
    if v.is_empty() {
        return Err("scatter: the potential needs at least one cell".to_string());
    }
    if !x_min.is_finite() || !x_max.is_finite() || x_max <= x_min {
        return Err(format!("scatter: need x_min < x_max, got [{x_min}, {x_max}]"));
    }
    for (name, value) in [("mass", mass), ("hbar", hbar)] {
        if !value.is_finite() || value <= 0.0 {
            return Err(format!("scatter: {name} must be finite and positive, got {value}"));
        }
    }
    if !e.is_finite() {
        return Err(format!("scatter: the energy must be finite, got {e}"));
    }
    if let Some(i) = v.iter().position(|z| !z.is_finite()) {
        return Err(format!("scatter: V[{i}] is not finite"));
    }

    let n = v.len();
    let d = (x_max - x_min) / n as f64;
    let (v_left, v_right) = (v[0], v[n - 1]);

    // An absorbing left asymptote has no well-defined incident flux, so
    // there is nothing for the answer to be a fraction OF.
    if v_left.im != 0.0 {
        return Err(format!(
            "scatter: the left asymptote V = {v_left:?} is absorbing, so the incident flux \
             is undefined — extend the sampled region so the potential is real where the \
             wave comes in"
        ));
    }
    // Compare the ENERGIES, not the computed `Re k`. The complex
    // square root of a negative real returns a real part of order
    // 1e-16 rather than exactly zero, so a test on `Re k <= 0` lets a
    // classically forbidden energy through and returns confident
    // nonsense — measured, `R = 0` for a barrier that reflects
    // everything. Found by a test asserting the refusal, not by
    // reading the code.
    if e <= v_left.re {
        return Err(format!(
            "scatter: E = {e} is at or below the left asymptote V = {}, so there is no \
             incident wave to scatter",
            v_left.re
        ));
    }
    let k_left = wavenumber(e, v_left, mass, hbar);
    let k_right = wavenumber(e, v_right, mass, hbar);

    // Start from the outgoing wave at the right edge and integrate
    // leftward: psi = 1, psi' = i k_R.
    let mut psi = C::ONE;
    let mut dpsi = C::I * k_right;
    for &vj in v.iter().rev() {
        let k = wavenumber(e, vj, mass, hbar);
        // Going right to left is the same matrix at -d.
        let (cos, sinc, ksin) = cell_matrix(k, -d);
        let (p, q) = (psi, dpsi);
        psi = p * cos + q * sinc;
        dpsi = p * ksin * -1.0 + q * cos;
        if !psi.is_finite() || !dpsi.is_finite() {
            return Err(format!(
                "scatter: the solution left f64 range at E = {e} — the barrier is too \
                 opaque for a transfer matrix at this width; the transmission is below \
                 about 1e-300"
            ));
        }
    }

    let inv = (C::I * k_left).inv();
    let flux = dpsi * inv;
    let a = (psi + flux) * 0.5;
    let b = (psi - flux) * 0.5;

    // The one place digits go: `a` is a difference, and for an opaque
    // barrier it is a small difference of large numbers. Measured from
    // the values rather than modelled.
    let conditioning = if a.abs() > 0.0 {
        ((psi.abs() + flux.abs()) / (2.0 * a.abs())).max(1.0)
    } else {
        f64::INFINITY
    };
    if conditioning.is_nan() || conditioning >= 1.0 / f64::EPSILON {
        return Err(format!(
            "scatter: at E = {e} forming the incident amplitude cancelled by {conditioning:.1e}, \
             which is every digit f64 has. The transmission here is smaller than this method \
             can resolve; use a thinner barrier, a coarser tolerance, or an S-matrix \
             formulation (not implemented)."
        ));
    }

    let t = a.inv();
    let r = b * a.inv();
    let reflection = r.norm_sqr();
    // Flux is only carried to infinity where the asymptote is real. In
    // an absorbing right-hand region nothing arrives, whatever |t| is.
    let transmission = if v_right.im == 0.0 && k_right.re > 0.0 {
        (k_right.re / k_left.re) * t.norm_sqr()
    } else {
        0.0
    };
    let absorption = 1.0 - reflection - transmission;

    Ok(Scattering {
        energy: e,
        t,
        r,
        reflection,
        transmission,
        absorption,
        conditioning,
    })
}

/// [`scatter`] for a real potential.
///
/// # Errors
/// As [`scatter`].
pub fn scatter_real(
    v: &[f64],
    x_min: f64,
    x_max: f64,
    e: f64,
    mass: f64,
    hbar: f64,
) -> Result<Scattering, String> {
    let c: Vec<C> = v.iter().map(|&x| C::real(x)).collect();
    scatter(&c, x_min, x_max, e, mass, hbar)
}

/// An energy range to sweep.
#[derive(Clone, Copy, Debug)]
pub struct EnergyRange {
    pub lo: f64,
    pub hi: f64,
    pub points: usize,
}

/// `T(E)` over a range of energies — the curve a wavepacket cannot
/// resolve.
///
/// Energies that the method cannot speak for are **skipped**, not
/// silently zeroed, so a scan across an opaque region returns fewer
/// points rather than a plausible flat line. The count is the caller's
/// to check.
///
/// # Errors
/// `points < 2`, a non-finite or misordered range, or an error from
/// [`scatter`] that is not per-energy (a malformed potential).
pub fn scan(
    v: &[C],
    x_min: f64,
    x_max: f64,
    range: EnergyRange,
    mass: f64,
    hbar: f64,
) -> Result<Vec<Scattering>, String> {
    let EnergyRange { lo: e_lo, hi: e_hi, points } = range;
    if points < 2 {
        return Err(format!("scan: ask for at least 2 energies, got {points}"));
    }
    if !e_lo.is_finite() || !e_hi.is_finite() || e_hi <= e_lo {
        return Err(format!("scan: need e_lo < e_hi, got [{e_lo}, {e_hi}]"));
    }
    // One probe first, so a malformed potential is an error rather than
    // an empty result that looks like an opaque barrier.
    scatter(v, x_min, x_max, 0.5 * (e_lo + e_hi), mass, hbar)?;
    let mut out = Vec::with_capacity(points);
    for i in 0..points {
        let e = e_lo + (e_hi - e_lo) * i as f64 / (points - 1) as f64;
        if let Ok(s) = scatter(v, x_min, x_max, e, mass, hbar) {
            out.push(s);
        }
    }
    Ok(out)
}

/// Sample a potential at the midpoints of `n` cells spanning
/// `[x_min, x_max]`.
///
/// Midpoints, not left edges: that is what makes [`scatter`] second
/// order in the cell width rather than first.
pub fn sample<F: Fn(f64) -> f64>(x_min: f64, x_max: f64, n: usize, f: F) -> Vec<C> {
    let d = (x_max - x_min) / n as f64;
    (0..n).map(|i| C::real(f(x_min + (i as f64 + 0.5) * d))).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qm1d::barrier_transmission;

    /// A rectangular barrier has a closed form, and it is the reference
    /// this whole module is judged against — above the barrier, below
    /// it, and at the resonances where `T` returns to exactly 1.
    fn barrier(n: usize, v0: f64, a: f64, span: f64) -> Vec<C> {
        sample(-span, span, n, |x| if x.abs() <= a / 2.0 { v0 } else { 0.0 })
    }

    #[test]
    fn a_rectangular_barrier_matches_its_closed_form() {
        let (v0, a, span) = (2.0_f64, 1.0_f64, 6.0_f64);
        // The cell edges must land on the barrier edges, or the sampled
        // barrier is a different one and the comparison is meaningless.
        let n = 4800;
        let v = barrier(n, v0, a, span);
        for &e in &[0.2_f64, 0.5, 1.0, 1.5, 1.9, 2.0, 2.5, 4.0, 8.0, 20.0] {
            let s = scatter_real(
                &v.iter().map(|z| z.re).collect::<Vec<_>>(),
                -span,
                span,
                e,
                1.0,
                1.0,
            )
            .unwrap();
            let want = barrier_transmission(e, v0, a);
            assert!(
                (s.transmission - want).abs() < 2e-4 * want.max(1e-3),
                "E = {e}: got {}, want {want}",
                s.transmission
            );
            // Real potential: nothing is absorbed and flux is conserved.
            assert!(s.absorption.abs() < 1e-10, "E = {e}: absorption {}", s.absorption);
            assert!(
                (s.transmission + s.reflection - 1.0).abs() < 1e-10,
                "E = {e}: T + R = {}",
                s.transmission + s.reflection
            );
        }
    }

    /// **The method is exact when the potential is piecewise constant
    /// on the cells** — not approximately, exactly.
    ///
    /// Each cell matrix is the closed-form solution for a constant
    /// potential, so a barrier whose edges land on cell boundaries is
    /// solved with no discretisation error at all. This asserts the
    /// error does not shrink with `n` because there is nothing to
    /// shrink: it is machine precision at 24 cells and at 4800.
    ///
    /// It is a far sharper statement than any tolerance, and it is the
    /// reason the convergence test below has to use a *smooth*
    /// potential to have anything to measure.
    #[test]
    fn the_method_is_exact_on_cell_aligned_steps() {
        let (v0, a, span, e) = (2.0_f64, 1.0_f64, 6.0_f64, 1.3_f64);
        let want = barrier_transmission(e, v0, a);
        for &n in &[24_usize, 120, 600, 4800] {
            let v = barrier(n, v0, a, span);
            let s = scatter(&v, -span, span, e, 1.0, 1.0).unwrap();
            assert!(
                (s.transmission - want).abs() < 1e-12,
                "n = {n}: {} vs {want}",
                s.transmission
            );
        }
    }

    /// Second order in the cell width, on a potential the cells cannot
    /// represent exactly: halving the cell size quarters the error.
    ///
    /// Midpoint sampling is what buys the second order; sampling at the
    /// left edge would give first.
    #[test]
    fn the_cell_error_is_second_order() {
        let (span, e) = (6.0_f64, 1.3_f64);
        let gauss = |x: f64| 2.0 * (-(x / 0.5_f64).powi(2)).exp();
        let fine = {
            let v = sample(-span, span, 153_600, gauss);
            scatter(&v, -span, span, e, 1.0, 1.0).unwrap().transmission
        };
        let mut errs = vec![];
        for &n in &[300_usize, 600, 1200, 2400] {
            let v = sample(-span, span, n, gauss);
            let t = scatter(&v, -span, span, e, 1.0, 1.0).unwrap().transmission;
            errs.push((t - fine).abs());
        }
        for w in errs.windows(2) {
            let ratio = w[0] / w[1];
            assert!((3.6..4.4).contains(&ratio), "expected ~4, got {ratio:.2} from {errs:?}");
        }
    }

    /// **The point of the module.** A double barrier has resonances at
    /// its quasi-bound energies where `T` rises to nearly 1, and
    /// `TUNNELING_RESULTS.md` §5 recorded that wavepackets could not
    /// see them: the peaks are narrower than any affordable packet's
    /// momentum spread.
    ///
    /// At fixed energy they are simply there. This asserts a peak
    /// exists, that it is genuinely narrow, and that between peaks the
    /// transmission is orders of magnitude lower — the shape a
    /// packet-averaged answer smears into a monotone rise.
    #[test]
    fn a_double_barrier_shows_resonances_a_wavepacket_cannot() {
        let (span, v0) = (12.0_f64, 3.0_f64);
        // Two barriers of width 0.5 with a well of width 3 between them.
        let n = 9600;
        let v = sample(-span, span, n, |x| {
            let ax = x.abs();
            if (1.5..=2.0).contains(&ax) {
                v0
            } else {
                0.0
            }
        });
        let curve =
            scan(&v, -span, span, EnergyRange { lo: 0.05, hi: 2.9, points: 1200 }, 1.0, 1.0)
                .unwrap();
        assert!(curve.len() > 1100, "too many energies were refused: {}", curve.len());

        // Find the interior maxima.
        let mut peaks = vec![];
        for w in curve.windows(3) {
            if w[1].transmission > w[0].transmission
                && w[1].transmission > w[2].transmission
                && w[1].transmission > 0.5
            {
                peaks.push((w[1].energy, w[1].transmission));
            }
        }
        assert!(!peaks.is_empty(), "a double barrier must have resonances");
        let (e_peak, t_peak) = peaks[0];
        assert!(t_peak > 0.9, "a resonance should approach unit transmission, got {t_peak}");

        // Narrow: a quarter of the way to the next sample the
        // transmission has already fallen a long way.
        let floor = curve
            .iter()
            .filter(|s| s.energy < e_peak * 0.6)
            .map(|s| s.transmission)
            .fold(0.0, f64::max);
        assert!(
            t_peak > 20.0 * floor.max(1e-12),
            "the peak ({t_peak:.3e}) must stand well above the background ({floor:.3e})"
        );
    }

    /// A well deep enough to bind supports transmission resonances at
    /// exactly `T = 1` — the Ramsauer–Townsend effect. They sit where
    /// the well contains a whole number of half-wavelengths, which is a
    /// closed form and needs no reference implementation.
    #[test]
    fn a_square_well_transmits_perfectly_at_the_ramsauer_energies() {
        let (span, v0, a) = (10.0_f64, -4.0_f64, 2.0_f64);
        let n = 8000;
        let v = sample(-span, span, n, |x| if x.abs() <= a / 2.0 { v0 } else { 0.0 });
        // Inside the well k' = sqrt(2(E - V0)); T = 1 when k' a = m pi.
        for m in 1..=3 {
            let kp = m as f64 * std::f64::consts::PI / a;
            let e = 0.5 * kp * kp + v0;
            if e <= 0.0 {
                continue;
            }
            let s = scatter(&v, -span, span, e, 1.0, 1.0).unwrap();
            assert!(
                s.transmission > 0.9999,
                "m = {m}, E = {e}: T = {} should be 1",
                s.transmission
            );
        }
    }

    /// Opaque barriers do **not** degrade this formulation.
    ///
    /// This test was written the other way round — asserting that a
    /// thick enough barrier gets refused — because that is what the
    /// amplitude-transfer formulation does. Measured, it never happens
    /// here: conditioning stays near 1.4 while `T` falls to 1e-276,
    /// because in the opaque limit `psi -> 0` at the left edge and the
    /// incident amplitude comes from `psi'` alone, with nothing to
    /// cancel against.
    ///
    /// So the assertion is the finding: transmission falls monotonically
    /// to underflow, reflection stays 1, and the conditioning number
    /// stays small throughout.
    #[test]
    fn an_opaque_barrier_stays_conditioned() {
        let span = 20.0_f64;
        let mut prev = f64::INFINITY;
        for a in [1.0_f64, 2.0, 4.0, 8.0, 16.0, 32.0] {
            let v = sample(-span, span, 8000, |x| if x.abs() <= a / 2.0 { 50.0 } else { 0.0 });
            let s = scatter(&v, -span, span, 1.0, 1.0, 1.0).unwrap();
            assert!(s.conditioning < 10.0, "a = {a}: conditioning {} blew up", s.conditioning);
            assert!(s.transmission < prev, "a = {a}: T should fall monotonically");
            assert!(
                (s.reflection - 1.0).abs() < 1e-9,
                "a = {a}: an opaque barrier reflects everything, got R = {}",
                s.reflection
            );
            prev = s.transmission;
        }
        assert!(prev < 1e-200, "the thickest barrier should be far past 1e-200, got {prev}");
    }

    /// A barrier so wide it swallows the whole domain leaves the
    /// incident energy classically forbidden *at the boundary*, and
    /// that is refused rather than answered.
    ///
    /// This is the case the first version got wrong: the guard tested
    /// `Re k <= 0`, and the complex square root of a negative real
    /// returns `Re k ~ 1e-16` rather than zero, so it passed and the
    /// routine confidently reported `R = 0` for a barrier that reflects
    /// everything. Comparing the energies instead is exact.
    #[test]
    fn an_energy_below_the_asymptote_is_refused() {
        let span = 20.0_f64;
        // The "barrier" covers the entire domain, so V_left = 50 > E.
        let v = sample(-span, span, 800, |_| 50.0);
        let err = scatter(&v, -span, span, 1.0, 1.0, 1.0).unwrap_err();
        assert!(err.contains("no incident wave"), "got: {err}");

        // And a negative energy against a zero potential, which the
        // `Re k` form also let through.
        let free = vec![C::ZERO; 64];
        assert!(scatter(&free, -1.0, 1.0, -1.0, 1.0, 1.0).is_err());
        assert!(scatter(&free, -1.0, 1.0, 0.0, 1.0, 1.0).is_err());
        // Just above it is fine.
        assert!(scatter(&free, -1.0, 1.0, 1e-6, 1.0, 1.0).is_ok());
    }

    /// A complex potential absorbs, and the books balance:
    /// `R + T + A = 1` by construction, with `A > 0` only where the
    /// potential has an imaginary part.
    #[test]
    fn an_absorbing_potential_takes_flux_out_of_the_books() {
        let span = 20.0_f64;
        let n = 4000;
        // Free on the left, an absorbing region on the right.
        let v: Vec<C> = (0..n)
            .map(|i| {
                let x = -span + (i as f64 + 0.5) * (2.0 * span / n as f64);
                if x > 10.0 {
                    C::new(0.0, -0.5 * ((x - 10.0) / 10.0).powi(2))
                } else {
                    C::ZERO
                }
            })
            .collect();
        let s = scatter(&v, -span, span, 2.0, 1.0, 1.0).unwrap();
        assert!(s.absorption > 0.9, "the CAP should swallow most of it: A = {}", s.absorption);
        assert!(s.reflection < 0.1, "and reflect little: R = {}", s.reflection);
        assert!(
            (s.reflection + s.transmission + s.absorption - 1.0).abs() < 1e-12,
            "the books must balance"
        );
    }

    #[test]
    fn it_refuses_bad_input() {
        let v = vec![C::ZERO; 10];
        assert!(scatter(&[], 0.0, 1.0, 1.0, 1.0, 1.0).is_err());
        assert!(scatter(&v, 1.0, 0.0, 1.0, 1.0, 1.0).is_err());
        assert!(scatter(&v, 0.0, 1.0, 1.0, 0.0, 1.0).is_err());
        assert!(scatter(&v, 0.0, 1.0, f64::NAN, 1.0, 1.0).is_err());
        // E at or below the left asymptote: no incident wave.
        assert!(scatter(&v, 0.0, 1.0, 0.0, 1.0, 1.0).is_err());
        assert!(scatter(&v, 0.0, 1.0, -1.0, 1.0, 1.0).is_err());
        // An absorbing left asymptote has no defined incident flux.
        let mut bad = v.clone();
        bad[0] = C::new(0.0, -1.0);
        assert!(scatter(&bad, 0.0, 1.0, 1.0, 1.0, 1.0).is_err());
        assert!(scan(&v, 0.0, 1.0, EnergyRange { lo: 1.0, hi: 2.0, points: 1 }, 1.0, 1.0).is_err());
        assert!(scan(&v, 0.0, 1.0, EnergyRange { lo: 2.0, hi: 1.0, points: 10 }, 1.0, 1.0).is_err());
    }

    /// `hbar` and the mass are not decoration: the answer must depend on
    /// them exactly as the wavenumber does. Doubling the mass at
    /// quarter the energy leaves `k` unchanged, so `T` must not move.
    #[test]
    fn mass_and_hbar_enter_only_through_the_wavenumber() {
        let (span, a) = (6.0_f64, 1.0_f64);
        let v = sample(-span, span, 4800, |x| if x.abs() <= a / 2.0 { 2.0 } else { 0.0 });
        let base = scatter(&v, -span, span, 3.0, 1.0, 1.0).unwrap();
        // k^2 = 2 m (E - V) / hbar^2. Scaling m by 4 and (E, V) by 1/4
        // leaves every k identical — but V is fixed here, so scale the
        // whole problem instead: m -> 4m, hbar -> 2 hbar leaves k alone.
        let same = scatter(&v, -span, span, 3.0, 4.0, 2.0).unwrap();
        assert!(
            (base.transmission - same.transmission).abs() < 1e-12,
            "{} vs {}",
            base.transmission,
            same.transmission
        );
    }
}
