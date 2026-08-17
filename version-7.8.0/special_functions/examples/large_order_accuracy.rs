//! Large order: where a 1/z expansion stops and a 1/nu expansion starts.
//!
//! Stage 14 left one hole, and this example is what located it: `J_nu(z)`
//! for `z` below `nu`. There `J` is exponentially small while the Hankel
//! functions it was being built from are exponentially large, so the
//! subtraction destroyed everything. At `nu = 400.5, z = 240` the answer
//! came back wrong by a factor of `5e89` — with a claimed error of zero,
//! because at `nu = 1/2` the 1/z expansion terminates exactly and its
//! truncation estimate was literally 0.
//!
//! The remedy is an expansion in `1/nu` instead: the Debye polynomials
//! and DLMF 10.19 / 10.41, which produce the small number directly.
//!
//! The turning point `z ~ nu` has its own expansion — Olver's uniform
//! Airy-type one, DLMF 10.20. An earlier version of this example argued
//! it was unnecessary because `z/nu` from 0.95 to 1.1 already measured
//! 1e-14. **That sampling was too coarse.** At 0.85 and 0.90 the error
//! reached 1.4e-9, and the third table below is now what it fixed.
//!
//! Run: cargo run -p special_functions --release --example large_order_accuracy

use special_functions::bessel_scaled::{
    bessel_j_scaled_nu, bessel_k_scaled_nu, bessel_y_scaled_nu,
};
use special_functions::complex::Complex64 as C;
use special_functions::debye::jy_debye;
use spec_math::cephes64::{jv, yv};

const FRACS: [f64; 9] = [0.1, 0.3, 0.5, 0.7, 0.85, 0.95, 1.0, 1.2, 2.0];

fn cell(v: Result<C, String>, want: f64) -> String {
    match v {
        Ok(g) if want != 0.0 && want.is_finite() => {
            format!("{:>10.1e}", (g.re - want).abs() / want.abs())
        }
        Ok(_) => format!("{:>10}", "no ref"),
        Err(e) if e.contains("outside f64") => format!("{:>10}", "no f64"),
        Err(_) => format!("{:>10}", "REFUSED"),
    }
}

fn main() {
    println!("Large order, real axis. Relative error against Cephes jv/yv.\n");
    println!("  'no f64'  = the value is determined but outside f64 range");
    println!("  'no ref'  = Cephes cannot represent it");
    println!("  'REFUSED' = no method here\n");

    for (name, f) in [
        ("J", bessel_j_scaled_nu as fn(f64, C) -> Result<C, String>),
        ("Y", bessel_y_scaled_nu),
    ] {
        println!("{name}_nu(x), columns are x/nu:\n");
        print!("{:>8} |", "nu");
        for fr in FRACS {
            print!(" {fr:>9.2}");
        }
        println!();
        println!("  -------+{}", "-".repeat(10 * FRACS.len()));
        for nu in [10.5f64, 40.5, 100.5, 200.5, 400.5, 1000.5] {
            print!("{nu:>8.1} |");
            for fr in FRACS {
                let x = nu * fr;
                let want = if name == "J" { jv(nu, x) } else { yv(nu, x) };
                print!(" {}", cell(f(nu, C::real(x)), want));
            }
            println!();
        }
        println!();
    }

    println!("Every catastrophic entry is gone. Before this stage the J row at");
    println!("nu = 400.5 read 5.4e89 at x/nu = 0.6 and the nu = 1000.5 row read");
    println!("2.4e246 — numbers returned with confident small error estimates.\n");

    // -----------------------------------------------------------------
    println!("\nThe turning-point band, and what DLMF 10.20 changed there.\n");
    println!("The numbers on the right are what the other routes gave before");
    println!("Olver's expansion was added; on the left is what they give now.\n");
    println!(
        "{:>8} {:>10} {:>12} | {:>10} {:>12}",
        "nu", "x/nu=0.85", "was", "x/nu=0.95", "was"
    );
    let was: [(f64, f64, f64); 4] = [
        (100.5, 1.4e-9, 2.4e-14),
        (200.5, 3.7e-11, 1.0e-12),
        (400.5, 4.4e-15, 1.3e-10),
        (1000.5, 7.3e-15, 1.3e-10),
    ];
    // Two things landed together and both were needed. Adding the
    // expansion was not enough on its own: the Hankel route was winning
    // the selection with an estimate that ignored the rounding a
    // hundred-step order recurrence accumulates, claiming 1.4e-11 while
    // delivering 1.4e-9. Flooring that estimate at `steps * eps` is what
    // let the better method be chosen.
    for (nu, w85, w95) in was {
        let c85 = cell(bessel_j_scaled_nu(nu, C::real(nu * 0.85)), jv(nu, nu * 0.85));
        let c95 = cell(bessel_j_scaled_nu(nu, C::real(nu * 0.95)), jv(nu, nu * 0.95));
        println!("{nu:>8.1} {c85} {w85:>12.1e} | {c95} {w95:>12.1e}");
    }
    println!("\nThree to four orders, in exactly the band a turning-point");
    println!("expansion is for. The earlier claim that 10.20 had nothing to fix");
    println!("came from sampling 0.95 to 1.1 and not 0.85 to 0.95 — the argument");
    println!("was sound, the measurement behind it was not.\n");
    println!("Two changes were needed, not one. The expansion alone did not help");
    println!("at nu = 100.5: the Hankel route kept winning the selection with an");
    println!("estimate that ignored the rounding a hundred-step order recurrence");
    println!("accumulates — it claimed 1.4e-11 and delivered 1.4e-9. A better");
    println!("method is only used if the comparison that picks it is honest.\n");
    println!("Its own accuracy is O(nu^-6) with three terms kept, and the hard");
    println!("part was not the expansion but its coefficients: A_k and B_k are");
    println!("sums of terms each singular at the turning point, cancelling exactly.");
    println!("Near zeta = 0 they come from Taylor series generated at 70 digits;");
    println!("A_1(0) = -1/225 and zeta'(1) = -2^(1/3) are known independently and");
    println!("both come out right.\n");
    println!("What no expansion can fix is the 'no f64' cells: there the value is");
    println!("determined and the number type is the limit.");

    // -----------------------------------------------------------------
    println!("\n\nThe Debye truncation estimate, which is what chooses the method.\n");
    println!("It has to grow towards the turning point, or the caller cannot know");
    println!("to stop trusting it. nu = 400:\n");
    println!("{:>10} {:>14}", "x/nu", "estimate");
    for fr in [0.1f64, 0.3, 0.5, 0.7, 0.85, 0.95, 0.99] {
        let e = jy_debye(400.0, 400.0 * fr)
            .0
            .map(|u| u.err)
            .unwrap_or(f64::INFINITY);
        println!("{fr:>10.2} {e:>14.1e}");
    }

    // -----------------------------------------------------------------
    println!("\n\nAnd the modified pair, from the same polynomials (DLMF 10.41).");
    println!("exp(x) K_nu(x) at orders where the vendored Cephes kn overflows:\n");
    println!("{:>8} {:>16} {:>16}", "nu", "x = nu", "x = 10 nu");
    for nu in [50.0f64, 200.0, 800.0, 3000.0] {
        let a = bessel_k_scaled_nu(nu, C::real(nu)).map(|v| v.re);
        let b = bessel_k_scaled_nu(nu, C::real(10.0 * nu)).map(|v| v.re);
        let f = |v: Result<f64, String>| match v {
            Ok(x) => format!("{x:>16.6e}"),
            Err(_) => format!("{:>16}", "out of range"),
        };
        println!("{nu:>8.0} {} {}", f(a), f(b));
    }
    println!("\n(pinned by the I-K Wronskian, whose right-hand side is elementary —");
    println!(" there is no reference implementation to compare against up here)");
}
