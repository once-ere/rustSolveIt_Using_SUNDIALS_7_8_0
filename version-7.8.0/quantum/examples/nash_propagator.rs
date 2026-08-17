//! The Nash propagator, measured against things that can contradict it.
//!
//! `EVOLVE_NASH` is the one piece of original numerical work in the
//! SolveIt C++: a split-operator scheme whose kinetic half is applied
//! by a **Bessel stencil** rather than by an FFT or a linear solve. This
//! example does not describe it — it measures it, three ways:
//!
//! 1. against a **closed form**, on a plane wave, where the scheme is
//!    exact and the answer is a single known phase;
//! 2. against **diagonalisation**, with a potential switched on, where
//!    the splitting is the only error left — for both the original's
//!    Lie ordering and the second-order Strang one;
//! 3. against **its own error bound**, for the Bessel truncation.
//!
//! Run: cargo run -p quantum --release --example nash_propagator

use quantum::nash::{norm, order_for, truncation_bound, NashPropagator, PeriodicGrid, Splitting};
use special_functions::complex::Complex64 as C;
use special_functions::eigen::jacobi_eigen;

fn packet(grid: &PeriodicGrid, x0: f64, k0: f64, width: f64) -> Vec<C> {
    let mut psi: Vec<C> = (0..grid.n)
        .map(|i| {
            let x = grid.x(i);
            C::from_polar((-((x - x0) / width).powi(2) / 2.0).exp(), k0 * x)
        })
        .collect();
    let s = norm(&psi, grid.h()).sqrt();
    for z in &mut psi {
        *z = *z * (1.0 / s);
    }
    psi
}

/// `exp(-i H t / hbar)` by diagonalising the periodic Hamiltonian.
/// Shares no code with the propagator.
fn exact(grid: &PeriodicGrid, v: &[f64], t: f64, psi0: &[C]) -> Vec<C> {
    let n = grid.n;
    let h = grid.h();
    let kappa = 1.0 / (2.0 * h * h);
    let mut a = vec![vec![0.0; n]; n];
    for i in 0..n {
        a[i][i] = 2.0 * kappa + v[i];
        a[i][(i + 1) % n] += -kappa;
        a[i][(i + n - 1) % n] += -kappa;
    }
    let (e, vecs) = jacobi_eigen(&a).unwrap();
    let mut out = vec![C::ZERO; n];
    for (m, phi) in vecs.iter().enumerate() {
        let mut c = C::ZERO;
        for (i, &p) in phi.iter().enumerate() {
            c = c + psi0[i] * p;
        }
        let ph = C::from_polar(1.0, -e[m] * t);
        for (i, &p) in phi.iter().enumerate() {
            out[i] = out[i] + c * ph * p;
        }
    }
    out
}

fn max_diff(a: &[C], b: &[C]) -> f64 {
    a.iter().zip(b).map(|(p, q)| (*p - *q).abs()).fold(0.0, f64::max)
}

fn main() {
    println!("The Nash propagator: exp(-i H dt) as a Bessel stencil\n");
    println!("    psi_j <- e^(-i L (1+v_j)) [ J_0(L) psi_j");
    println!("                + sum_M i^M J_M(L) (psi_(j-M) + psi_(j+M)) ]\n");
    println!("with L = hbar dt / (m h^2). One pass over the grid, no solver.\n");

    // -----------------------------------------------------------------
    println!("1. Free particle: the scheme is EXACT, so this is a closed form.\n");
    println!("With V = 0 the two factors commute and no splitting error exists.");
    println!("A plane wave exp(i k x) must be multiplied by exactly");
    println!("exp(-i L (1 - cos k h)) — the lattice dispersion, not the continuum one.\n");
    let grid = PeriodicGrid::new(0.0, 1.0, 64).unwrap();
    let free = vec![0.0; grid.n];
    let p = NashPropagator::new(grid.clone(), &free, 1.0, 1.0, 2.0e-4, None).unwrap();
    println!("   lambda = {:.4},  order = {},  truncation <= {:.1e}\n",
             p.lambda(), p.order(), p.truncation_error());
    println!("   {:>6} {:>14} {:>14}", "mode", "|error|", "phase applied");
    for m in [1_i32, 3, 7, 16, 31] {
        let k = 2.0 * std::f64::consts::PI * f64::from(m);
        let mut psi: Vec<C> = (0..grid.n).map(|i| C::from_polar(1.0, k * grid.x(i))).collect();
        let before = psi.clone();
        p.step(&mut psi).unwrap();
        let want = C::from_polar(1.0, -p.lambda() * (1.0 - (k * grid.h()).cos()));
        let got: Vec<C> = before.iter().map(|z| *z * want).collect();
        println!("   {m:>6} {:>14.2e} {:>14.6}", max_diff(&psi, &got), want.arg());
    }

    // -----------------------------------------------------------------
    println!("\n\n2. With a potential, the SPLITTING is what is left — and how it is");
    println!("   ordered decides whether that is O(dt) or O(dt^2).\n");
    println!("Harmonic well on [-6, 6], measured against diagonalising H.\n");
    let grid = PeriodicGrid::new(-6.0, 6.0, 48).unwrap();
    let v: Vec<f64> = grid.points().iter().map(|x| 0.5 * x * x).collect();
    let psi0 = packet(&grid, -1.0, 2.0, 0.8);
    let t = 0.05;
    let want = exact(&grid, &v, t, &psi0);
    println!("   {:>8} {:>10} {:>12} {:>7} {:>12} {:>7} {:>10}",
             "steps", "dt", "Lie", "ratio", "Strang", "ratio", "gain");
    let (mut plie, mut pstrang) = (f64::NAN, f64::NAN);
    for steps in [25_usize, 50, 100, 200, 400] {
        let dt = t / steps as f64;
        let base = NashPropagator::new(grid.clone(), &v, 1.0, 1.0, dt, None).unwrap();

        let mut a = psi0.clone();
        base.run(&mut a, steps).unwrap();
        let lie = max_diff(&a, &want);

        let strang_prop = NashPropagator::new(grid.clone(), &v, 1.0, 1.0, dt, None)
            .unwrap()
            .with_splitting(Splitting::Strang);
        let mut b = psi0.clone();
        strang_prop.run(&mut b, steps).unwrap();
        let strang = max_diff(&b, &want);

        let rl = if plie.is_finite() { format!("{:.2}", plie / lie) } else { "-".into() };
        let rs =
            if pstrang.is_finite() { format!("{:.2}", pstrang / strang) } else { "-".into() };
        println!("   {steps:>8} {dt:>10.2e} {lie:>12.3e} {rl:>7} {strang:>12.3e} {rs:>7} {:>10.0}x",
                 lie / strang);
        plie = lie;
        pstrang = strang;
    }
    println!("\n   Lie halves, Strang quarters — first order against second. The gain");
    println!("   column is what that is worth at a fixed step, and it grows as the");
    println!("   step shrinks, because the two are converging at different rates.");
    println!("\n   Strang costs one extra pointwise multiply per step in principle and");
    println!("   almost nothing in practice: consecutive steps put a trailing half");
    println!("   phase against a leading one and `run` fuses them, so a whole run");
    println!("   pays one extra half phase rather than one per step.");
    println!("\n   Lie remains the default. This is a port, so the default has to be");
    println!("   what the original does; Strang is reached by with_splitting.");

    println!("\n   Norm drift, which the splitting does NOT affect — both are products");
    println!("   of unitaries, so this is flat in dt for either:\n");
    println!("   {:>10} {:>16} {:>16}", "dt", "Lie", "Strang");
    for &dt in &[1e-4_f64, 1e-2, 0.1, 1.0] {
        let base = NashPropagator::new(grid.clone(), &v, 1.0, 1.0, dt, None).unwrap();
        let mut a = psi0.clone();
        base.run(&mut a, 40).unwrap();
        let mut b = psi0.clone();
        NashPropagator::new(grid.clone(), &v, 1.0, 1.0, dt, None)
            .unwrap()
            .with_splitting(Splitting::Strang)
            .run(&mut b, 40)
            .unwrap();
        println!("   {dt:>10.0e} {:>16.2e} {:>16.2e}",
                 (norm(&a, grid.h()) - 1.0).abs(),
                 (norm(&b, grid.h()) - 1.0).abs());
    }
    println!("\n   Accuracy and stability fail independently here: at dt = 1 both are");
    println!("   normalised to 1e-15 and neither is remotely accurate.");

    // -----------------------------------------------------------------
    println!("\n\n3. The Bessel truncation, which is NOT what limits the scheme.\n");
    println!("Truncating Jacobi-Anger at K leaves 2 sum_(M>K) |J_M(L)|, and");
    println!("|J_M(L)| <= (L/2)^M / M!, so it falls superexponentially past K ~ L.\n");
    println!("   {:>8} {:>12} {:>12} {:>12} {:>12}", "K", "L=0.92", "L=3", "L=8", "L=20");
    for k in [2_usize, 4, 6, 8, 12, 16, 24, 32] {
        print!("   {k:>8}");
        for l in [0.92_f64, 3.0, 8.0, 20.0] {
            print!(" {:>12.1e}", truncation_bound(l, k).unwrap());
        }
        println!();
    }
    println!("\n   SolveIt shipped L = 0.92 with K = 16. The order actually needed");
    println!("   for the stencil to be exact to rounding is {}, so the shipped value",
             order_for(0.92, f64::EPSILON).unwrap());
    println!("   carried a small margin over it — harmless, and worth knowing");
    println!("   rather than guessing at.");
    println!("\n   Read the table the other way and it is a step-size limit: L grows");
    println!("   like dt/h^2, so refining the grid at fixed dt costs stencil width.");

    // -----------------------------------------------------------------
    println!("\n\nPractical guidance:");
    println!("  boundaries    periodic — a packet leaving one edge re-enters at the");
    println!("                other. qm1d's Grid is Dirichlet; they are not the same");
    println!("                domain and results are not interchangeable");
    println!("  accuracy      Lie (the default, and the original) is first order in");
    println!("                dt; Strang is second. Prefer Strang unless you are");
    println!("                reproducing SolveIt output — it costs almost nothing");
    println!("  splitting     with_splitting(Splitting::Strang). At V = 0 the two");
    println!("                coincide exactly, so it only matters with a potential");
    println!("  stability     unconditional: norm is conserved to rounding at any dt,");
    println!("                so a wrong answer here stays a normalised wrong answer");
    println!("  cost          O(n K) per step and no solver, against Crank-Nicolson's");
    println!("                O(n) with a tridiagonal solve");
    println!("  choosing K    leave it to order_for; it is linear in cost and");
    println!("                superexponential in accuracy, so there is nothing to save");
}
