//! The quantum harmonic oscillator, solved TWO independent ways, as an
//! end-to-end check that the mathematical infrastructure is sound.
//!
//! Units: hbar = m = omega = 1, so H = -1/2 d^2/dx^2 + 1/2 x^2 and the
//! exact spectrum is E_n = n + 1/2.
//!
//! Route 1 — ANALYTIC, using the Hermite polynomials:
//!     psi_n(x) = (2^n n! sqrt(pi))^(-1/2) H_n(x) exp(-x^2/2)
//! We verify these are orthonormal by numerical quadrature, which
//! exercises `orthopoly` and `quadrature` together.
//!
//! Route 2 — NUMERICAL, with no special functions at all: discretise
//! the Hamiltonian on a finite-difference grid and diagonalise it with
//! the Jacobi eigensolver. This is the matrix-mechanics path, and it is
//! what makes *arbitrary* potentials tractable.
//!
//! The two routes share no code. Agreement between them is real
//! evidence rather than a self-consistency check.
//!
//! Run: cargo run -p special_functions --release --example harmonic_oscillator

use special_functions::eigen::eigenvalues;
use special_functions::orthopoly::hermite_h;
use special_functions::quadrature::integrate;

/// Analytic eigenfunction psi_n(x) of the oscillator.
fn psi(n: i32, x: f64) -> Result<f64, String> {
    // normalisation (2^n n! sqrt(pi))^{-1/2}, built multiplicatively so
    // n! never forms explicitly
    let mut norm = 1.0_f64 / std::f64::consts::PI.sqrt().sqrt();
    for k in 1..=n {
        norm /= (2.0 * k as f64).sqrt();
    }
    Ok(norm * hermite_h(n, x)? * (-0.5 * x * x).exp())
}

fn main() {
    println!("Quantum harmonic oscillator   (hbar = m = omega = 1)\n");

    // ---- Route 1: analytic eigenfunctions ---------------------------
    // Orthonormality: <m|n> = delta_mn, by Gauss-Legendre on [-10,10]
    // (the Gaussian is < 1e-21 at the ends, so truncation is harmless).
    println!("Route 1 — analytic psi_n from Hermite polynomials");
    println!("  overlap matrix <m|n>, should be the identity:");
    let mut worst_off = 0.0_f64;
    let mut worst_diag = 0.0_f64;
    for m in 0..6 {
        print!("   ");
        for n in 0..6 {
            let v = integrate(|x| psi(m, x).unwrap() * psi(n, x).unwrap(), -10.0, 10.0, 400)
                .unwrap();
            print!("{v:8.5}");
            if m == n {
                worst_diag = worst_diag.max((v - 1.0).abs());
            } else {
                worst_off = worst_off.max(v.abs());
            }
        }
        println!();
    }
    println!("  worst |<n|n> - 1| = {worst_diag:.2e}");
    println!("  worst |<m|n>|     = {worst_off:.2e}\n");

    // Energy by the expectation value <n|H|n>, computed from the
    // analytic wavefunction with H acting via the known identity
    //   -1/2 psi'' + 1/2 x^2 psi = E psi.
    // We use the virial route instead: <T> = <V> = E/2, so
    //   E_n = 2 <V> = 2 * integral( 1/2 x^2 |psi|^2 ).
    println!("  energies from the virial theorem, E_n = 2<V>:");
    let mut worst_e = 0.0_f64;
    for n in 0..6 {
        let v = integrate(
            |x| 0.5 * x * x * psi(n, x).unwrap().powi(2),
            -12.0,
            12.0,
            500,
        )
        .unwrap();
        let e = 2.0 * v;
        let exact = n as f64 + 0.5;
        worst_e = worst_e.max((e - exact).abs());
        println!("    n={n}:  E = {e:.10}   exact = {exact:.1}");
    }
    println!("  worst error = {worst_e:.2e}\n");

    // ---- Route 2: finite differences + diagonalisation ---------------
    // No special functions anywhere in this route.
    println!("Route 2 — finite-difference Hamiltonian + Jacobi eigensolver");
    let l = 8.0_f64; // box half-width
    let n_grid = 400usize;
    let h = 2.0 * l / (n_grid + 1) as f64;
    let mut ham = vec![vec![0.0; n_grid]; n_grid];
    for i in 0..n_grid {
        let x = -l + (i + 1) as f64 * h;
        ham[i][i] = 1.0 / (h * h) + 0.5 * x * x; // -1/2 * (-2/h^2) + V
        if i + 1 < n_grid {
            ham[i][i + 1] = -0.5 / (h * h);
            ham[i + 1][i] = -0.5 / (h * h);
        }
    }
    let vals = eigenvalues(&ham).unwrap();
    println!("  grid: {n_grid} points on [{:.0}, {:.0}], h = {h:.5}", -l, l);
    let mut worst_fd = 0.0_f64;
    for (n, &e) in vals.iter().enumerate().take(6) {
        let exact = n as f64 + 0.5;
        let err = (e - exact).abs();
        worst_fd = worst_fd.max(err);
        println!("    n={n}:  E = {e:.10}   exact = {exact:.1}   err = {err:.2e}");
    }
    println!("  worst error = {worst_fd:.2e}\n");

    // The finite-difference error above is not noise: it grows like
    // (2n^2 + 2n + 1), because higher states oscillate faster and a
    // 3-point stencil resolves them less well. Check the SCALING, which
    // is a far stronger statement than any single tolerance.
    println!("  error growth with n (should track 2n^2+2n+1):");
    let e0 = (vals[0] - 0.5).abs();
    for (n, &e) in vals.iter().enumerate().take(6) {
        let ratio = (e - (n as f64 + 0.5)).abs() / e0;
        let predicted = (2 * n * n + 2 * n + 1) as f64;
        println!("    n={n}:  observed {ratio:6.1}x   predicted {predicted:6.1}x");
    }

    // ---- Second-order convergence: halve h, error should fall 4x -----
    println!("\nGrid refinement (the defining property of the stencil):");
    let mut prev_err = 0.0_f64;
    let mut order_ok = true;
    for &ng in &[100usize, 200, 400] {
        let hh = 2.0 * l / (ng + 1) as f64;
        let mut hm = vec![vec![0.0; ng]; ng];
        for i in 0..ng {
            let x = -l + (i + 1) as f64 * hh;
            hm[i][i] = 1.0 / (hh * hh) + 0.5 * x * x;
            if i + 1 < ng {
                hm[i][i + 1] = -0.5 / (hh * hh);
                hm[i + 1][i] = -0.5 / (hh * hh);
            }
        }
        let e = eigenvalues(&hm).unwrap()[0];
        let err = (e - 0.5).abs();
        if prev_err > 0.0 {
            let drop = prev_err / err;
            println!("    N={ng:4}  h={hh:.5}  err={err:.3e}  (fell {drop:.2}x, expect ~4)");
            if !(3.5..4.5).contains(&drop) {
                order_ok = false;
            }
        } else {
            println!("    N={ng:4}  h={hh:.5}  err={err:.3e}");
        }
        prev_err = err;
    }

    // ---- Agreement between two independent routes -------------------
    println!("\nCross-check: the two routes share no code.");
    println!("  worst |E_numerical - E_exact| over n=0..5 = {worst_fd:.2e}");

    // The analytic route must be near machine precision; the
    // finite-difference route is only asked to be second-order accurate,
    // which is all a 3-point stencil promises.
    let ok = worst_diag < 1e-9 && worst_off < 1e-9 && worst_e < 1e-8 && worst_fd < 5e-3 && order_ok;
    if ok {
        println!("\nSUCCESS: analytic eigenfunctions are orthonormal, the virial");
        println!("theorem reproduces E_n = n + 1/2, and an independent");
        println!("finite-difference diagonalisation agrees.");
    } else {
        println!("\nFAILURE: one of the checks exceeded its tolerance.");
        std::process::exit(1);
    }
}
