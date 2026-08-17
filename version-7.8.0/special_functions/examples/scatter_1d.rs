//! 1-D quantum scattering off a rectangular barrier, by Crank–Nicolson.
//!
//! This is the end-to-end exercise of the **clean-room** replacements
//! written for Stage 1: a Gaussian wave packet is launched at a barrier,
//! propagated with the Cayley form
//!
//!     (1 + i H dt/2) psi^{n+1} = (1 - i H dt/2) psi^n
//!
//! which is a complex tridiagonal solve per step, and the transmitted
//! and reflected probabilities are compared against the closed-form
//! plane-wave transmission coefficient averaged over the packet's own
//! momentum distribution.
//!
//! Units: hbar = m = 1, so E = k^2/2.
//!
//! Two things are checked, and they are checks of very different kinds:
//!
//! * **Unitarity** — the total norm must be conserved to near machine
//!   precision. This is exact, not asymptotic: the Cayley operator is
//!   unitary for any dt, so any drift is the *solver* being wrong, not
//!   discretisation error. It is the sharpest possible test of the
//!   tridiagonal routine.
//! * **Transmission** — |T| from the simulation against the analytic
//!   T(E) weighted by |phi(k)|^2. This is only second-order accurate in
//!   dx and dt, so it is checked to a few percent.
//!
//! Run: cargo run -p special_functions --release --example scatter_1d

use special_functions::complex::Complex64 as C;
use special_functions::quadrature::integrate;
use special_functions::tridiag::solve_tridiag_c;

/// Analytic transmission through a rectangular barrier of height `v0`
/// and width `a`, for a plane wave of energy `e`. Both branches below
/// and above the barrier top (A&S-standard textbook result).
fn transmission(e: f64, v0: f64, a: f64) -> f64 {
    if e <= 0.0 {
        return 0.0;
    }
    if (e - v0).abs() < 1e-12 {
        // The E = V0 limit, where both branches degenerate.
        let t = 1.0 + v0 * a * a / 2.0;
        return 1.0 / t;
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

fn main() {
    // ---- problem setup ------------------------------------------------
    let x_min = -100.0_f64;
    let x_max = 100.0_f64;
    let dx = 0.05_f64;
    let n = ((x_max - x_min) / dx) as usize - 1; // interior points only
    let dt = 0.005_f64;
    let n_steps = 6000usize; // to t = 30

    let v0 = 2.5_f64; // barrier height
    let bw = 1.0_f64; // barrier width, sitting on [0, bw]
    let k0 = 2.0_f64; // central wavenumber  ->  E0 = 2.0
    let sigma = 2.0_f64; // packet width
    let x0 = -25.0_f64; // launch point

    let x = |i: usize| x_min + (i + 1) as f64 * dx;
    let pot = |xx: f64| if (0.0..bw).contains(&xx) { v0 } else { 0.0 };

    println!("1-D scattering off a rectangular barrier (hbar = m = 1)\n");
    println!("  grid    : {n} points, x in [{x_min}, {x_max}], dx = {dx}");
    println!("  stepping: dt = {dt}, {n_steps} steps  ->  t = {}", dt * n_steps as f64);
    println!("  barrier : V0 = {v0}, width {bw} on [0, {bw}]");
    println!("  packet  : k0 = {k0} (E0 = {:.3}), sigma = {sigma}, x0 = {x0}\n", k0 * k0 / 2.0);

    // ---- initial packet ------------------------------------------------
    let mut psi: Vec<C> = (0..n)
        .map(|i| {
            let xx = x(i);
            let g = (-(xx - x0) * (xx - x0) / (4.0 * sigma * sigma)).exp();
            C::from_polar(g, k0 * xx)
        })
        .collect();
    let norm0: f64 = psi.iter().map(|z| z.norm_sqr()).sum::<f64>() * dx;
    for p in psi.iter_mut() {
        *p = *p * (1.0 / norm0.sqrt());
    }
    let norm0: f64 = psi.iter().map(|z| z.norm_sqr()).sum::<f64>() * dx;

    // ---- the Crank-Nicolson operator (time independent, built once) ----
    // H = -1/2 d^2/dx^2 + V, three-point stencil, Dirichlet walls.
    let kin = 0.5 / (dx * dx);
    let half = C::I * (dt / 2.0);
    let off_h = C::real(-kin);
    let diag_h: Vec<C> = (0..n).map(|i| C::real(2.0 * kin + pot(x(i)))).collect();

    let sub: Vec<C> = vec![half * off_h; n];
    let sup: Vec<C> = vec![half * off_h; n];
    let lhs_diag: Vec<C> = diag_h.iter().map(|&d| C::ONE + half * d).collect();

    // ---- propagate ------------------------------------------------------
    let mut worst_norm_drift = 0.0_f64;
    for step in 1..=n_steps {
        // rhs = (1 - i H dt/2) psi
        let rhs: Vec<C> = (0..n)
            .map(|i| {
                let l = if i > 0 { psi[i - 1] } else { C::ZERO };
                let r = if i + 1 < n { psi[i + 1] } else { C::ZERO };
                let hp = diag_h[i] * psi[i] + off_h * l + off_h * r;
                psi[i] - half * hp
            })
            .collect();
        psi = solve_tridiag_c(&sub, &lhs_diag, &sup, &rhs).expect("tridiagonal solve failed");

        if step % 500 == 0 {
            let nrm: f64 = psi.iter().map(|z| z.norm_sqr()).sum::<f64>() * dx;
            let drift = (nrm / norm0 - 1.0).abs();
            worst_norm_drift = worst_norm_drift.max(drift);
            println!("    t = {:6.2}   norm = {nrm:.14}   drift = {drift:.2e}", step as f64 * dt);
        }
    }

    // ---- measure transmission and reflection ---------------------------
    let mut refl = 0.0;
    let mut trans = 0.0;
    let mut inside = 0.0;
    for (i, p) in psi.iter().enumerate() {
        let xx = x(i);
        let d = p.norm_sqr() * dx;
        if xx < 0.0 {
            refl += d;
        } else if xx > bw {
            trans += d;
        } else {
            inside += d;
        }
    }

    // ---- analytic prediction, averaged over the packet's momenta -------
    // |phi(k)|^2 ~ exp(-2 sigma^2 (k - k0)^2)
    let w = |k: f64| (-2.0 * sigma * sigma * (k - k0) * (k - k0)).exp();
    let k_lo = k0 - 6.0 / (2.0 * sigma);
    let k_hi = k0 + 6.0 / (2.0 * sigma);
    let num = integrate(|k| w(k) * transmission(k * k / 2.0, v0, bw), k_lo, k_hi, 400).unwrap();
    let den = integrate(w, k_lo, k_hi, 400).unwrap();
    let t_avg = num / den;
    let t_plane = transmission(k0 * k0 / 2.0, v0, bw);

    println!("\n  reflected   R = {refl:.6}");
    println!("  transmitted T = {trans:.6}");
    println!("  still in barrier = {inside:.2e}");
    println!("  R + T + inside  = {:.14}", refl + trans + inside);
    println!("\n  analytic T at the central energy E0 = {:.3}:  {t_plane:.6}", k0 * k0 / 2.0);
    println!("  analytic T averaged over |phi(k)|^2       :  {t_avg:.6}");
    let rel = (trans - t_avg).abs() / t_avg;
    println!("  relative difference                       :  {:.3} %", 100.0 * rel);

    println!("\n  worst norm drift over the whole run: {worst_norm_drift:.2e}");
    println!("  (the Cayley operator is unitary for ANY dt, so this");
    println!("   measures the tridiagonal solver, not the time step)");

    // Unitarity is exact mathematics — hold it to near machine precision.
    // Transmission is second-order accurate in dx and dt — a few percent.
    let ok = worst_norm_drift < 1e-9 && rel < 0.05 && (refl + trans + inside - 1.0).abs() < 1e-9;
    if ok {
        println!("\nSUCCESS: the propagator is unitary to {worst_norm_drift:.1e} and the");
        println!("transmitted fraction matches the momentum-averaged analytic");
        println!("result to {:.2} %.", 100.0 * rel);
    } else {
        println!("\nFAILURE: a check exceeded its tolerance.");
        std::process::exit(1);
    }
}
