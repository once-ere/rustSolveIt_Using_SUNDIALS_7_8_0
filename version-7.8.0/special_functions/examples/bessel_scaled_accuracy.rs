//! What scaling buys: the same functions, computed where they could not
//! previously be computed at all.
//!
//! Stages 11 to 13 measured three failures and documented them:
//! `Y` on the real axis wrong in its first digit by `x = 40`, `K`
//! worthless past `x = 12`, `H1` gone by `Im z = 12`. Each was the same
//! disease — a small quantity assembled from large ones. This example
//! measures the cure.
//!
//! Every reference is independent of the code under test: Cephes on the
//! real axis, and the half-integer closed forms (which are exact, and
//! once scaled contain no exponential at all) everywhere else.
//!
//! Run: cargo run -p special_functions --release --example bessel_scaled_accuracy

use special_functions::bessel_complex::{bessel_k_c, bessel_y_c};
use special_functions::bessel_scaled::{
    bessel_i_scaled_nu, bessel_k_scaled_nu, bessel_y_scaled_nu, hankel_h1_scaled_nu,
};
use special_functions::complex::Complex64 as C;
use special_functions::hankel::hankel_h1_nu;
use spec_math::cephes64::{i0, k0, kn, yv};

fn rel(got: f64, want: f64) -> f64 {
    (got - want).abs() / want.abs().max(1e-300)
}

fn show(v: Result<C, String>, want: f64) -> String {
    match v {
        Ok(g) if g.re.is_finite() => format!("{:>10.1e}", rel(g.re, want)),
        _ => format!("{:>10}", "refused"),
    }
}

fn main() {
    println!("Scaled Bessel and Hankel: the three documented failures, cured\n");

    // -----------------------------------------------------------------
    println!("1. Y_0 on the real axis, against Cephes yv.\n");
    println!("{:>7} {:>12} {:>12}", "x", "unscaled", "scaled");
    println!("  -----+{}", "-".repeat(26));
    for x in [10.0f64, 20.0, 30.0, 40.0, 60.0, 120.0, 500.0, 5000.0] {
        let want = yv(0.0, x);
        let old = bessel_y_c(0, C::real(x)).map(|v| C::real(v.re));
        // On the real axis the scaling exp(-|Im z|) is 1, so the scaled
        // value IS Y_0(x) — no unscaling needed to compare.
        let new = bessel_y_scaled_nu(0.0, C::real(x));
        println!("{x:>7.0} {} {}", show(old, want), show(new, want));
    }
    println!("\nThe unscaled column is the ascending series losing |z| nepers.");
    println!("The scaled column is the asymptotic expansion, which has no");
    println!("cancellation in it at all — its terms are a series in 1/z whose");
    println!("leading term is 1.");

    // -----------------------------------------------------------------
    println!("\n\n2. exp(x) K_0(x), against Cephes k0.\n");
    println!("{:>7} {:>12} {:>12}", "x", "unscaled", "scaled");
    println!("  -----+{}", "-".repeat(26));
    for x in [5.0f64, 10.0, 15.0, 20.0, 30.0, 100.0, 400.0] {
        let want = k0(x) * x.exp();
        let old = bessel_k_c(0, C::real(x)).map(|v| C::real(v.re * x.exp()));
        let new = bessel_k_scaled_nu(0.0, C::real(x));
        println!("{x:>7.0} {} {}", show(old, want), show(new, want));
    }
    println!("\nPast x = 745 the unscaled K_0 is smaller than the smallest f64, so");
    println!("no unscaled routine can represent it whatever its accuracy:");
    let got = bessel_k_scaled_nu(0.0, C::real(2000.0)).unwrap().re;
    let lead = (std::f64::consts::PI / 4000.0).sqrt();
    println!("  k0(2000) = {} (underflowed)", k0(2000.0));
    println!("  exp(x)K_0(2000) = {got:.12e}");
    println!("  leading term sqrt(pi/2x) = {lead:.12e}");
    println!(
        "  ratio - 1 = {:.4e}, and the first correction -1/(8x) is {:.4e}",
        got / lead - 1.0,
        -1.0 / 16_000.0
    );

    // -----------------------------------------------------------------
    println!("\n\n3. exp(-iz) H1_0(z) up the imaginary axis, against");
    println!("   H1_0(iy) = -(2i/pi) K_0(y) (DLMF 10.27.8, Cephes k0).\n");
    println!("{:>7} {:>12} {:>12} {:>14}", "Im z", "unscaled", "scaled", "|H1_0(iy)|");
    println!("  -----+{}", "-".repeat(40));
    for y in [5.0f64, 8.0, 10.0, 12.0, 15.0, 20.0, 40.0] {
        let z = C::new(0.0, y);
        let want = C::I * (-2.0 / std::f64::consts::PI) * k0(y);
        let wants = want * (C::I * z * -1.0).exp();
        let old = hankel_h1_nu(0.0, z).map(|v| v * (C::I * z * -1.0).exp());
        let new = hankel_h1_scaled_nu(0.0, z);
        let f = |v: Result<C, String>| match v {
            Ok(g) => format!("{:>12.1e}", (g - wants).abs() / wants.abs()),
            Err(_) => format!("{:>12}", "refused"),
        };
        println!("{y:>7.0} {} {} {:>14.2e}", f(old), f(new), want.abs());
    }
    println!("\nAnd far past where the ingredients themselves exist as f64 — at");
    println!("Im z = 700, J and Y are about e^700, above the top of the type:");
    for y in [100.0f64, 300.0, 700.0] {
        let z = C::new(3.0, y);
        let got = hankel_h1_scaled_nu(0.5, z).unwrap();
        let want = C::I * -1.0 * (C::real(2.0 / std::f64::consts::PI) * z.inv()).powf(0.5);
        println!(
            "  Im z = {y:>4.0}: exp(-iz)H1_{{1/2}} matches its closed form to {:.1e}",
            (got - want).abs() / want.abs()
        );
    }

    // -----------------------------------------------------------------
    println!("\n\n4. exp(-x) I_0(x), past the overflow of the unscaled value.\n");
    for x in [100.0f64, 700.0, 1000.0, 5000.0] {
        let got = bessel_i_scaled_nu(0.0, C::real(x)).unwrap().re;
        let unscaled = i0(x);
        let note = if unscaled.is_finite() {
            format!("{:>10.1e}", rel(got, unscaled * (-x).exp()))
        } else {
            format!("{:>10}", "overflow")
        };
        println!("  x = {x:>6.0}   scaled = {got:.12e}   unscaled i0(x) {note}");
    }

    // -----------------------------------------------------------------
    println!("\n\n5. Order. Cephes kn gives up at order 40; the recurrence does not.\n");
    println!("{:>5} {:>16} {:>16}", "n", "cephes kn(n,25)", "ours exp(x)K_n(25)");
    for n in [10i32, 20, 30, 40, 80] {
        let c = kn(n as isize, 25.0);
        let ours = bessel_k_scaled_nu(n as f64, C::real(25.0)).unwrap().re;
        let cs = if c.is_finite() {
            format!("{:>16.6e}", c * 25.0f64.exp())
        } else {
            format!("{:>16}", "overflow")
        };
        println!("{n:>5} {cs} {ours:>16.6e}");
    }
    println!("\n(the upward recurrence in order closes on itself exactly, which is");
    println!(" what the test suite pins where no reference exists)");

    // -----------------------------------------------------------------
    println!("\n\nWhat is still missing:");
    println!("  * The uniform Airy-type expansions of DLMF 10.20, for |z| and nu");
    println!("    large and comparable. Everything here is either an expansion in");
    println!("    1/z at fixed order or a recurrence in order from one.");
    println!("  * Complex ORDER.");
    println!("  When neither route reaches a point, the routine RETURNS AN ERROR");
    println!("  naming both estimates. That is the substantive difference from the");
    println!("  unscaled routines, which returned a confident wrong first digit.");
}
