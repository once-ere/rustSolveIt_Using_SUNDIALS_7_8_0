//! Designing the absorbing boundary, instead of tuning it.
//!
//! A complex absorbing potential `V -> V - i W(x)` lets a wavepacket
//! leave the domain without bouncing off the Dirichlet wall. It has two
//! failure modes that pull against each other: too weak and the packet
//! sails through to the wall and reflects off *that*; too strong and it
//! reflects off the absorber's own leading edge, because a sharp change
//! in the potential is a mirror whether the potential is real or
//! imaginary.
//!
//! Until now the compromise was found by **propagating packets and
//! looking** — `quantum/examples/absorber_tuning.rs` fires a Gaussian at
//! the edge for 5000 steps and reports what is left. That works, and it
//! is slow, noisy, and answers only for the one packet you fired.
//!
//! It is also unnecessary. The absorber is a potential, so what it does
//! to a wave of wavenumber `k` is a **scattering problem at fixed
//! energy**, and [`crate::transfer`] solves those exactly. This module
//! is that observation:
//!
//! ```text
//!     free  |  W(x) ramp of `width`  |  free
//!             R <-                     -> T
//! ```
//!
//! * `R` is what the absorber reflects — the failure mode of an
//!   absorber that is too strong.
//! * `T` is what **leaks through** and would hit the wall behind it —
//!   the failure mode of one that is too weak.
//! * `R + T` is therefore the figure of merit, and the packet sweep
//!   could not separate the two terms at all.
//!
//! No propagation, no packet, no time step: the answer is a function of
//! `k` alone, and a band of `k` costs one evaluation per sample.
//!
//! # What it says that a packet sweep could not
//!
//! Measured over `k` in `[1, 4]` with a quadratic ramp, optimising the
//! strength for the band gives `eta = 4.4` and a worst-case escape of
//! `1.4e-2`, against `5.1e-2` for the strength previously in use — a
//! real but modest gain. Tripling the *width* to 18 takes the same
//! quantity to `3.9e-6`.
//!
//! That is the useful conclusion, and it is not one a single-packet
//! sweep can reach: over a band, the worst case is set by the **longest
//! wavelength**, and the cure for a long wavelength is a wider ramp, not
//! a stronger one. Strength is what you tune; width is what you buy.
//!
//! # Agreement with the method it replaces
//!
//! `matches_the_propagated_measurement` checks this against the packet
//! sweep it supersedes. They agree to within the packet's own
//! interpretation — which is the point of keeping both.

use crate::transfer::{scatter, Scattering};
use special_functions::complex::Complex64 as C;

/// The shape of the absorbing ramp: `W(x) = strength (x/width)^power`.
#[derive(Clone, Copy, Debug)]
pub struct Ramp {
    pub width: f64,
    pub power: f64,
}

/// A band of wavenumbers to design against, and how finely to sample it.
#[derive(Clone, Copy, Debug)]
pub struct Band {
    pub k_lo: f64,
    pub k_hi: f64,
    pub samples: usize,
}

/// What an absorber does to a wave of one wavenumber.
#[derive(Clone, Copy, Debug)]
pub struct Leak {
    pub k: f64,
    /// Reflected by the absorber's own leading edge.
    pub reflection: f64,
    /// Passed straight through, to hit whatever is behind it.
    pub leakage: f64,
    /// Swallowed, which is the point.
    pub absorbed: f64,
}

impl Leak {
    /// `R + T` — everything the absorber failed to swallow, and so
    /// everything that ends up back in the interior sooner or later.
    pub fn escaped(&self) -> f64 {
        self.reflection + self.leakage
    }
}

/// Cells per unit length.
///
/// Measured, with the ramp edges aligned to cell boundaries (see
/// [`geometry`]): the escape fraction converges at exactly the second
/// order [`crate::transfer`] promises — ratios of 4.00 across five
/// halvings — and at 200 cells per unit length the relative error is
/// **3.2e-6**. That is three orders of magnitude finer than the
/// differences between candidate absorbers, and costs 0.2 ms a call.
///
/// The first version of this constant carried the claim "converged to
/// 1e-9", which was written without measuring and was wrong by three
/// decades. `the_result_is_converged_in_the_cell_count` now measures it.
const CELLS_PER_LENGTH: f64 = 200.0;

fn geometry(ramp: Ramp, strength: f64, k: f64) -> Result<(Vec<C>, f64, f64), String> {
    let Ramp { width, power } = ramp;
    if !width.is_finite() || width <= 0.0 {
        return Err(format!("absorber: width must be finite and positive, got {width}"));
    }
    if !strength.is_finite() || strength < 0.0 {
        return Err(format!("absorber: strength must be finite and non-negative, got {strength}"));
    }
    if !power.is_finite() || power < 1.0 {
        return Err(format!("absorber: power must be at least 1, got {power}"));
    }
    if !k.is_finite() || k <= 0.0 {
        return Err(format!("absorber: k must be finite and positive, got {k}"));
    }
    // **The ramp edges must land on cell boundaries.** Sampling a fixed
    // interval with a varying cell count makes the first and last
    // partially covered cells flip in and out, so the *shape* wobbles
    // as the resolution changes and the convergence is destroyed —
    // measured, ratios of 1.62, 4.68, 0.89, 3.68 instead of a clean 4.
    // Choosing the cell size from the ramp width and then padding by a
    // whole number of cells fixes the geometry at every resolution, and
    // the second order reappears exactly.
    let n_ramp = ((width * CELLS_PER_LENGTH).ceil() as usize).max(16);
    let d = width / n_ramp as f64;
    // A pad of free space on each side so both asymptotes are real: the
    // left one so the incident flux is defined, the right one so the
    // leakage is a flux rather than an amplitude. One wavelength is
    // ample — free space only adds phase.
    let pad_target = (2.0 * std::f64::consts::PI / k).max(width * 0.25);
    let n_pad = ((pad_target / d).ceil() as usize).max(4);
    let pad = n_pad as f64 * d;
    let n = n_ramp + 2 * n_pad;
    if n > 4_000_000 {
        return Err(format!(
            "absorber: width {width} at k = {k} would need {n} cells; widen the ramp or narrow the band"
        ));
    }
    let v = (0..n)
        .map(|i| {
            if i >= n_pad && i < n_pad + n_ramp {
                let x = ((i - n_pad) as f64 + 0.5) * d;
                C::new(0.0, -strength * (x / width).powf(power))
            } else {
                C::ZERO
            }
        })
        .collect();
    Ok((v, -pad, width + pad))
}

/// What the absorber does to a wave of wavenumber `k`.
///
/// # Errors
/// A non-positive or non-finite parameter, or a failure inside
/// [`scatter`].
///
/// # Examples
/// ```
/// use quantum::absorber::{leak, Ramp};
/// // Width 6, strength 3, quadratic ramp, k = 2. Almost all of the
/// // escape is LEAKAGE through a slightly-too-weak absorber, not
/// // reflection off it — R = 5.6e-6 against T = 4.2e-3. Separating
/// // those two is the whole reason for computing rather than firing a
/// // packet, which only ever sees the sum.
/// let l = leak(Ramp { width: 6.0, power: 2.0 }, 3.0, 2.0, 1.0, 1.0).unwrap();
/// assert!(l.escaped() < 5e-3);
/// assert!(l.leakage > 100.0 * l.reflection);
/// ```
pub fn leak(ramp: Ramp, strength: f64, k: f64, mass: f64, hbar: f64) -> Result<Leak, String> {
    let (v, x_min, x_max) = geometry(ramp, strength, k)?;
    let e = hbar * hbar * k * k / (2.0 * mass);
    let Scattering { reflection, transmission, absorption, .. } =
        scatter(&v, x_min, x_max, e, mass, hbar)?;
    Ok(Leak { k, reflection, leakage: transmission, absorbed: absorption })
}

/// The worst `R + T` over a band of wavenumbers.
///
/// A simulation is only as good as its worst-behaved component, so the
/// band maximum is the number to design against — not the value at some
/// nominal `k0`, which is what firing one packet reports.
///
/// # Errors
/// A misordered or non-positive band, fewer than 2 samples, or an error
/// from [`leak`].
pub fn worst_escape(
    ramp: Ramp,
    strength: f64,
    band: Band,
    mass: f64,
    hbar: f64,
) -> Result<f64, String> {
    let Band { k_lo, k_hi, samples } = band;
    if k_lo.is_nan() || k_lo <= 0.0 || k_hi < k_lo || !k_hi.is_finite() {
        return Err(format!("absorber: need 0 < k_lo <= k_hi, got [{k_lo}, {k_hi}]"));
    }
    if samples < 2 {
        return Err(format!("absorber: use at least 2 samples, got {samples}"));
    }
    let mut worst: f64 = 0.0;
    for i in 0..samples {
        let k = k_lo + (k_hi - k_lo) * i as f64 / (samples - 1) as f64;
        worst = worst.max(leak(ramp, strength, k, mass, hbar)?.escaped());
    }
    Ok(worst)
}

/// The strength that minimises the worst escape over a band, and that
/// worst value.
///
/// The trade-off is unimodal in `log(strength)` — too weak leaks, too
/// strong reflects — so a ternary search on the logarithm finds the
/// minimum without a sweep. `absorber_tuning.rs` measured that shape by
/// propagation; this exploits it.
///
/// # Errors
/// As [`worst_escape`], or a search that never found a finite value.
///
/// # Examples
/// ```
/// use quantum::absorber::{choose_strength, worst_escape, Band, Ramp};
/// // Design for k in [1, 4] with a ramp 6 wide.
/// let (eta, worst) = choose_strength(Ramp { width: 6.0, power: 2.0 }, Band { k_lo: 1.0, k_hi: 4.0, samples: 9 }, 1.0, 1.0).unwrap();
/// // It beats the strength that was in use before this existed.
/// let old = worst_escape(Ramp { width: 6.0, power: 2.0 }, 3.0, Band { k_lo: 1.0, k_hi: 4.0, samples: 9 }, 1.0, 1.0).unwrap();
/// assert!(worst < old, "{worst:.2e} should beat {old:.2e}");
///
/// // But strength can only do so much: over a WIDE band the worst case
/// // is set by the longest wavelength, and the cure for that is width.
/// // Same band, three times the ramp:
/// let (_, wide) = choose_strength(Ramp { width: 18.0, power: 2.0 }, Band { k_lo: 1.0, k_hi: 4.0, samples: 9 }, 1.0, 1.0).unwrap();
/// assert!(wide < worst / 1000.0, "{wide:.2e} vs {worst:.2e}");
/// assert!(eta > 0.0);
/// ```
pub fn choose_strength(ramp: Ramp, band: Band, mass: f64, hbar: f64) -> Result<(f64, f64), String> {
    let f = |log_eta: f64| -> f64 {
        worst_escape(ramp, log_eta.exp(), band, mass, hbar).unwrap_or(f64::INFINITY)
    };
    // 1e-4 to 1e4 in strength covers every regime the tuning example
    // found, with the optimum near 3 for the widths in use.
    let (mut lo, mut hi) = (-9.0_f64, 9.0_f64);
    for _ in 0..80 {
        let a = lo + (hi - lo) / 3.0;
        let b = hi - (hi - lo) / 3.0;
        if f(a) < f(b) {
            hi = b;
        } else {
            lo = a;
        }
    }
    let eta = (0.5 * (lo + hi)).exp();
    let worst = worst_escape(ramp, eta, band, mass, hbar)?;
    if !worst.is_finite() {
        return Err("absorber: the search found no usable strength".to_string());
    }
    Ok((eta, worst))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qm1d::{Grid, Hamiltonian, Propagator, Wavefunction};

    /// **The cross-check that matters**: the exact calculation against
    /// the propagated one it replaces.
    ///
    /// `absorber_tuning.rs` fires a Gaussian at the edge and reports the
    /// fraction still in the interior after a long run. That number is
    /// an average over the packet's own spread in `k`, so it cannot be
    /// expected to match a single-`k` answer exactly — but it must
    /// agree in magnitude, and it must move the same way when the
    /// absorber is made worse.
    ///
    /// Two methods sharing no code, agreeing on a quantity that spans
    /// several orders of magnitude, is the strongest evidence available
    /// here that either is right.
    fn propagated_escape(width: f64, strength: f64, power: f64, k0: f64) -> f64 {
        let g = Grid::new(-40.0, 40.0, 1600).unwrap();
        let ham = Hamiltonian::from_fn(g.clone(), |_| 0.0, 1.0, 1.0)
            .unwrap()
            .with_absorber(width, strength, power)
            .unwrap();
        let mut w = Wavefunction::gaussian(g, 0.0, 1.5, k0).unwrap();
        let prop = Propagator::new(ham, 0.005).unwrap();
        prop.run(&mut w, 5000).unwrap();
        w.norm()
    }

    #[test]
    fn matches_the_propagated_measurement() {
        let (power, k0) = (2.0_f64, 2.0_f64);
        // A deliberately poor absorber and a good one, so the
        // comparison spans orders of magnitude rather than testing one
        // point.
        for &(width, strength) in &[(6.0_f64, 0.05_f64), (6.0, 0.3), (6.0, 3.0)] {
            let exact = leak(Ramp { width, power }, strength, k0, 1.0, 1.0).unwrap().escaped();
            let packet = propagated_escape(width, strength, power, k0);
            // Same order of magnitude. The packet carries a spread of
            // k around k0 and the absorber is k-dependent, so a tight
            // tolerance here would be measuring the packet, not the
            // absorber.
            let ratio = (exact.max(1e-12) / packet.max(1e-12)).log10().abs();
            assert!(
                ratio < 1.3,
                "width {width}, eta {strength}: exact {exact:.3e} vs propagated \
                 {packet:.3e} — more than a decade apart"
            );
        }
    }

    /// Both failure modes exist, and they are on opposite sides of the
    /// optimum. This is the shape the whole module depends on.
    #[test]
    fn too_weak_leaks_and_too_strong_reflects() {
        let (width, power, k) = (6.0_f64, 2.0_f64, 2.0_f64);
        let weak = leak(Ramp { width, power }, 0.02, k, 1.0, 1.0).unwrap();
        // The optimum is found, not guessed: an earlier version of this
        // test hard-coded eta = 3 as "good" and it was not good — at
        // width 6 and k = 2 it still leaks 4e-3, which is worse than
        // the real optimum by two decades.
        let (eta, _) = choose_strength(Ramp { width, power }, Band { k_lo: k, k_hi: k, samples: 2 }, 1.0, 1.0).unwrap();
        let good = leak(Ramp { width, power }, eta, k, 1.0, 1.0).unwrap();
        let strong = leak(Ramp { width, power }, 500.0, k, 1.0, 1.0).unwrap();

        // Too weak: it leaks through, and that dominates.
        assert!(weak.leakage > 100.0 * weak.reflection, "weak: {weak:?}");
        // Too strong: it reflects off its own edge, and that dominates.
        assert!(strong.reflection > 100.0 * strong.leakage, "strong: {strong:?}");
        // In between beats both, by a lot.
        assert!(good.escaped() < weak.escaped() / 100.0, "good {good:?} vs weak {weak:?}");
        assert!(good.escaped() < strong.escaped() / 100.0, "good {good:?} vs strong {strong:?}");
    }

    /// The chosen strength must actually be the best available — not
    /// merely good. Checked against a brute-force sweep, which is what
    /// the search exists to avoid paying for.
    #[test]
    fn the_chosen_strength_beats_a_brute_force_sweep() {
        let (width, power) = (6.0_f64, 2.0_f64);
        let (k_lo, k_hi) = (1.0_f64, 4.0_f64);
        let (eta, worst) = choose_strength(Ramp { width, power }, Band { k_lo, k_hi, samples: 9 }, 1.0, 1.0).unwrap();

        let mut best = f64::INFINITY;
        for i in 0..=60 {
            let e = (-4.0 + 8.0 * i as f64 / 60.0_f64).exp2();
            best = best.min(worst_escape(Ramp { width, power }, e, Band { k_lo, k_hi, samples: 9 }, 1.0, 1.0).unwrap());
        }
        assert!(
            worst <= best * 1.05,
            "the search found {worst:.3e} at eta = {eta:.3}, a sweep found {best:.3e}"
        );
    }

    /// Designing for a band is not the same as designing for its
    /// midpoint, which is what firing one packet does.
    ///
    /// The band optimum must beat the midpoint-tuned absorber **at the
    /// band edges**, which is exactly where a single-packet tuning is
    /// blind.
    #[test]
    fn designing_for_a_band_beats_tuning_at_one_energy() {
        let (width, power) = (6.0_f64, 2.0_f64);
        let (k_lo, k_hi) = (0.7_f64, 6.0_f64);
        let (band_eta, band_worst) = choose_strength(Ramp { width, power }, Band { k_lo, k_hi, samples: 13 }, 1.0, 1.0).unwrap();
        let mid = 0.5 * (k_lo + k_hi);
        let (mid_eta, _) = choose_strength(Ramp { width, power }, Band { k_lo: mid, k_hi: mid, samples: 2 }, 1.0, 1.0).unwrap();
        let mid_worst = worst_escape(Ramp { width, power }, mid_eta, Band { k_lo, k_hi, samples: 13 }, 1.0, 1.0).unwrap();
        assert!(
            band_worst < mid_worst,
            "band design (eta {band_eta:.3}) gave {band_worst:.3e}; tuning at k = {mid} \
             (eta {mid_eta:.3}) gave {mid_worst:.3e} across the same band"
        );
    }

    /// A gentler ramp reflects less, which is the reason `power` is a
    /// parameter at all.
    #[test]
    fn a_higher_ramp_exponent_reflects_less_at_fixed_strength() {
        let (width, k, eta) = (6.0_f64, 2.0_f64, 30.0_f64);
        let mut prev = f64::INFINITY;
        for p in [1.0_f64, 2.0, 3.0] {
            let r = leak(Ramp { width, power: p }, eta, k, 1.0, 1.0).unwrap().reflection;
            assert!(r < prev, "power {p}: reflection {r:.3e} should beat {prev:.3e}");
            prev = r;
        }
    }

    /// The cell count is a choice, so it has to be justified by
    /// measurement rather than by looking reasonable.
    ///
    /// With the ramp edges pinned to cell boundaries the escape
    /// fraction converges at **exactly** second order, so this asserts
    /// the 1/n^2 law itself — a far sharper statement than any single
    /// tolerance — and then that the shipped resolution sits where the
    /// error is below 1e-5 relative.
    #[test]
    fn the_result_is_converged_in_the_cell_count() {
        use crate::transfer::scatter;
        let (width, eta, power, k) = (6.0_f64, 3.0_f64, 2.0_f64, 2.0_f64);

        let escape_at = |cells_per_length: f64| -> f64 {
            let n_ramp = ((width * cells_per_length).ceil() as usize).max(16);
            let d = width / n_ramp as f64;
            let n_pad = (((2.0 * std::f64::consts::PI / k) / d).ceil() as usize).max(4);
            let pad = n_pad as f64 * d;
            let n = n_ramp + 2 * n_pad;
            let v: Vec<C> = (0..n)
                .map(|i| {
                    if i >= n_pad && i < n_pad + n_ramp {
                        let x = ((i - n_pad) as f64 + 0.5) * d;
                        C::new(0.0, -eta * (x / width).powf(power))
                    } else {
                        C::ZERO
                    }
                })
                .collect();
            let s = scatter(&v, -pad, width + pad, 0.5 * k * k, 1.0, 1.0).unwrap();
            s.reflection + s.transmission
        };

        let reference = escape_at(12_800.0);
        let mut errs = vec![];
        for &cpl in &[100.0_f64, 200.0, 400.0, 800.0] {
            errs.push((escape_at(cpl) - reference).abs() / reference);
        }
        for w in errs.windows(2) {
            let ratio = w[0] / w[1];
            assert!((3.7..4.3).contains(&ratio), "expected ~4, got {ratio:.2} from {errs:?}");
        }
        // And the shipped resolution is comfortably inside 1e-5.
        assert!(errs[1] < 1e-5, "at {CELLS_PER_LENGTH} cells/length the error is {}", errs[1]);
        // The library agrees with the hand-rolled geometry above.
        let lib = leak(Ramp { width, power }, eta, k, 1.0, 1.0).unwrap().escaped();
        assert!((lib - escape_at(CELLS_PER_LENGTH)).abs() < 1e-12 * lib);
    }

    /// The SHIPPED resolution, not a locally rolled one.
    ///
    /// `the_result_is_converged_in_the_cell_count` builds its own
    /// geometry to measure the 1/n^2 law, so Stage 2F's probe could drop
    /// `CELLS_PER_LENGTH` from 200 to 8 and the suite still passed — the
    /// convergence was pinned but the constant in use was not. This
    /// compares what `leak` actually returns against a computation 16
    /// times finer.
    #[test]
    fn the_shipped_resolution_is_accurate_enough() {
        use crate::transfer::scatter;
        let ramp = Ramp { width: 6.0, power: 2.0 };
        for &(eta, k) in &[(3.0_f64, 2.0_f64), (10.0, 1.0), (1.0, 4.0)] {
            let shipped = leak(ramp, eta, k, 1.0, 1.0).unwrap().escaped();

            let fine = 16.0 * CELLS_PER_LENGTH;
            let n_ramp = ((ramp.width * fine).ceil() as usize).max(16);
            let d = ramp.width / n_ramp as f64;
            let n_pad = ((((2.0 * std::f64::consts::PI / k).max(ramp.width * 0.25)) / d).ceil()
                as usize)
                .max(4);
            let pad = n_pad as f64 * d;
            let v: Vec<C> = (0..n_ramp + 2 * n_pad)
                .map(|i| {
                    if i >= n_pad && i < n_pad + n_ramp {
                        let x = ((i - n_pad) as f64 + 0.5) * d;
                        C::new(0.0, -eta * (x / ramp.width).powf(ramp.power))
                    } else {
                        C::ZERO
                    }
                })
                .collect();
            let s = scatter(&v, -pad, ramp.width + pad, 0.5 * k * k, 1.0, 1.0).unwrap();
            let reference = s.reflection + s.transmission;

            let rel = (shipped - reference).abs() / reference.max(1e-300);
            assert!(
                rel < 1e-4,
                "eta = {eta}, k = {k}: the shipped resolution gives {shipped:.6e} against \
                 {reference:.6e} at 16x — {rel:.1e} relative"
            );
        }
    }

    #[test]
    fn it_refuses_bad_input() {
        assert!(leak(Ramp { width: 0.0, power: 2.0 }, 1.0, 1.0, 1.0, 1.0).is_err());
        assert!(leak(Ramp { width: 1.0, power: 2.0 }, -1.0, 1.0, 1.0, 1.0).is_err());
        assert!(leak(Ramp { width: 1.0, power: 0.5 }, 1.0, 1.0, 1.0, 1.0).is_err());
        assert!(leak(Ramp { width: 1.0, power: 2.0 }, 1.0, 0.0, 1.0, 1.0).is_err());
        assert!(worst_escape(Ramp { width: 1.0, power: 2.0 }, 1.0, Band { k_lo: 2.0, k_hi: 1.0, samples: 5 }, 1.0, 1.0).is_err());
        assert!(worst_escape(Ramp { width: 1.0, power: 2.0 }, 1.0, Band { k_lo: 1.0, k_hi: 2.0, samples: 1 }, 1.0, 1.0).is_err());
    }
}
