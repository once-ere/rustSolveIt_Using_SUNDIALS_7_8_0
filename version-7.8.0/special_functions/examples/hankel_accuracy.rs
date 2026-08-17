//! Where the Hankel functions give out.
//!
//! `H1 = J + iY` looks like a formality. It is not: in the upper half
//! plane `H1` **decays** like `exp(-Im z)` while `J` and `Y` each
//! **grow** like `exp(|Im z|)`, so forming the sum throws away
//! `2 Im z` nepers. That is intrinsic to the combination and no
//! rearrangement avoids it. `H2` is the mirror image.
//!
//! A natural question is whether one of the two routes — integer-order
//! (Miller for `J`, ascending series for `Y`) or non-integer-order (one
//! ascending series at `+nu` and `-nu`) — survives longer. **Measured,
//! the answer is that it makes essentially no difference**, and the
//! reason is worth knowing: at whole order the non-integer routine hands
//! `Y` straight to the integer one, so the two share the dominant
//! ingredient. Choosing a route does not buy anything here. Only a
//! genuinely different formulation does.
//!
//! Every reference here is independent of the code under test:
//!
//! ```text
//!   real axis:       H1_0(x)  = j0(x) + i yn(0,x)        (Cephes)
//!   imaginary axis:  H1_0(iy) = -(2i/pi) K_0(y)          (Cephes k0,
//!                                                         via DLMF 10.27.8)
//!   any z, nu=1/2:   H1_{1/2}(z) = -i sqrt(2/(pi z)) exp(iz)
//! ```
//!
//! Run: cargo run -p special_functions --release --example hankel_accuracy

use special_functions::complex::Complex64 as C;
use special_functions::hankel::{hankel_h1_c, hankel_h1_nu, sph_hankel_h1};
use spec_math::cephes64::{j0, k0, yn};

fn rel(got: C, want: C) -> f64 {
    (got - want).abs() / want.abs().max(1e-300)
}

fn show(v: Result<C, String>, want: C) -> String {
    match v {
        Ok(g) => format!("{:>11.1e}", rel(g, want)),
        Err(_) => format!("{:>11}", "refused"),
    }
}

fn main() {
    println!("Hankel accuracy: two routes, measured against independent references\n");

    // -----------------------------------------------------------------
    println!("1. Real axis. H1_0(x) = J_0(x) + i Y_0(x), both from Cephes.\n");
    println!("{:>6} {:>11} {:>11}", "x", "H1_0 _c", "H1_0 _nu");
    println!("  ----+{}", "-".repeat(24));
    for x in [1.0f64, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0] {
        let want = C::new(j0(x), yn(0, x));
        println!(
            "{x:>6.0} {} {}",
            show(hankel_h1_c(0, C::real(x)), want),
            show(hankel_h1_nu(0.0, C::real(x)), want)
        );
    }
    println!("\nOn the real axis there is no Hankel cancellation at all — |H1| and");
    println!("|J| are the same size — so this table is just Y's accuracy showing");
    println!("through, and the two routes are the same ascending series for Y.");

    // -----------------------------------------------------------------
    println!("\n\n2. Positive imaginary axis, where H1 is at its worst.\n");
    println!("K_nu(z) = (pi/2) i^(nu+1) H1_nu(iz) (DLMF 10.27.8), so at nu = 0");
    println!("H1_0(iy) = -(2i/pi) K_0(y) — and Cephes k0 gives the right side to");
    println!("machine precision. H1_0(iy) is of size exp(-y) while its ingredients");
    println!("are of size exp(+y), which is the whole problem.\n");
    println!("{:>6} {:>11} {:>11} {:>13}", "y", "H1_0 _c", "H1_0 _nu", "|H1_0(iy)|");
    println!("  ----+{}", "-".repeat(37));
    for y in [1.0f64, 3.0, 5.0, 8.0, 10.0, 12.0, 15.0, 20.0] {
        let want = C::I * (-2.0 / std::f64::consts::PI) * k0(y);
        println!(
            "{y:>6.0} {} {} {:>13.2e}",
            show(hankel_h1_c(0, C::new(0.0, y)), want),
            show(hankel_h1_nu(0.0, C::new(0.0, y)), want),
            want.abs()
        );
    }

    println!("\nThe two columns are within a factor of 1.5 of each other everywhere.");
    println!("That is the useful result, and it is not what one would guess from the");
    println!("fact that they use different algorithms for J: at WHOLE order the _nu");
    println!("routine delegates Y to the integer one (the reflection formula is 0/0");
    println!("there), so both routes share the ingredient that dominates. Picking a");
    println!("route buys nothing. Only a different formulation would.\n");
    println!("The loss follows 1e-16 exp(3 Im z) — the same law K obeys on the real");
    println!("axis, which is no coincidence: H1_0(iy) IS K_0(y) up to a constant, so");
    println!("these are literally the same computation seen twice. J at imaginary");
    println!("argument carries a relative error of exp(|Im z|) on a value of size");
    println!("exp(|Im z|), and the answer is of size exp(-Im z): three factors.\n");
    println!("{:>6} {:>13} {:>13}", "y", "measured", "1e-16 e^(3y)");
    for y in [5.0f64, 10.0, 15.0, 20.0] {
        let want = C::I * (-2.0 / std::f64::consts::PI) * k0(y);
        println!(
            "{y:>6.0} {:>13.1e} {:>13.1e}",
            rel(hankel_h1_c(0, C::new(0.0, y)).unwrap(), want),
            1e-16 * (3.0 * y).exp()
        );
    }
    println!("\nPractically: H1 above the real axis is good to |Im z| ~ 8 and gone");
    println!("by |Im z| ~ 12. Below the axis it is fine. H2 is the mirror.");

    // -----------------------------------------------------------------
    println!("\n\n3. The half plane, at nu = 1/2 where the closed form is exact.\n");
    println!("H1_{{1/2}}(z) = -i sqrt(2/(pi z)) exp(iz), for complex z.\n");
    let args: [f64; 9] = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5];
    print!("   |z| |");
    for a in args {
        print!(" {a:>8.1}");
    }
    println!("   <- arg z");
    println!("  -----+{}", "-".repeat(9 * args.len()));
    for r in [1.0f64, 2.0, 4.0, 8.0, 12.0, 16.0, 20.0] {
        print!("  {r:>4.0} |");
        for a in args {
            let z = C::from_polar(r, a);
            let want = C::I * -1.0
                * (C::real(2.0 / std::f64::consts::PI) * z.inv()).powf(0.5)
                * (C::I * z).exp();
            print!(" {:>8.1e}", rel(hankel_h1_nu(0.5, z).unwrap(), want));
        }
        println!();
    }
    println!("\nNegative arg (below the real axis) is exact to |z| = 20; positive arg");
    println!("degrades exactly as the cancellation argument says. H2 is this table");
    println!("reflected — it is sound above the axis and fails below it.");

    // -----------------------------------------------------------------
    println!("\n\n4. Spherical Hankel: the outgoing-wave property.\n");
    println!("|x h1_n(x)| -> 1 is what makes h1 an outgoing spherical wave. For");
    println!("n = 0 it is EXACT at every x, since h1_0 = -i exp(ix)/x. For n >= 1");
    println!("the gap is n(n+1)/(4 x^2):\n");
    println!("{:>4} {:>12} {:>12} {:>12}", "n", "x=50", "x=200", "x=800");
    for n in 0..5 {
        print!("{n:>4}");
        for x in [50.0f64, 200.0, 800.0] {
            print!(" {:>12.2e}", (sph_hankel_h1(n, x).unwrap() * x).abs() - 1.0);
        }
        println!("   (law {:>8.2e} at x=50)", (n * (n + 1)) as f64 / (4.0 * 2500.0));
    }

    // -----------------------------------------------------------------
    println!("\n\nPractical guidance:");
    println!("  real axis        both routes fine to |z| ~ 15; past 30 it is Y that");
    println!("                   fails, not the Hankel combination");
    println!("  Im z < 0         use H1 — sound to |z| ~ 20, either route");
    println!("  Im z > 0         use H2 there instead. If you truly need H1 above");
    println!("                   the axis, expect 1e-16 exp(3 Im z): fine to 8,");
    println!("                   gone by 12, and switching routes will not help");
    println!("  spherical        real argument only, and accurate throughout — j_n");
    println!("                   and y_n are recurrences on a real line with no");
    println!("                   cancellation of this kind");
    println!("\n  A scaled Hankel (the AMOS approach: return exp(-iz) H1 and let the");
    println!("  caller supply the exponential) would remove the problem entirely and");
    println!("  is NOT implemented here.");
}
