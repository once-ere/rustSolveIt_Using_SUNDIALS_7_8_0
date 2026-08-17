//! Where complex-argument Bessel stops being trustworthy — all four kinds.
//!
//! **This example used to measure `J` only, and that was the defect.**
//! It checked the generating-function identity
//! `exp((z/2)(t - 1/t)) = sum_n J_n(z) t^n`, which involves no `Y` at
//! all, then let the module documentation state the resulting law as if
//! it governed the module. It does not. `J` and `I` come from Miller
//! recurrence; `Y` comes from an ascending series and `K` is assembled
//! from `J` and `Y` at imaginary argument. Those fail in different
//! places, and on the real axis — where `J` is at its *best* — `Y` is
//! wrong in the first digit by `x = 40` and `K` is worthless by `x = 15`.
//!
//! Nothing in the code changed. What changed is that the claim is now
//! measured for every kind rather than one, on the axis where each is at
//! its worst, against a reference that shares no code with it.
//!
//! Run: cargo run -p special_functions --release --example bessel_complex_accuracy

use special_functions::bessel_complex::{
    bessel_i_c, bessel_j_array_c, bessel_j_c, bessel_k_c, bessel_y_c,
};
use special_functions::complex::Complex64 as C;
use spec_math::cephes64::{i0, j0, k0, yn};

/// Relative error in the generating-function identity at `z`. This
/// measures `J` and nothing else — kept because it is the one check that
/// needs no reference implementation at all, but no longer mistaken for
/// a statement about the module.
fn gen_err(z: C) -> f64 {
    let n_max = 90;
    let j = match bessel_j_array_c(n_max, z) {
        Ok(v) => v,
        Err(_) => return f64::NAN,
    };
    let t = C::new(1.2, 0.3);
    let t_inv = t.inv();
    let mut sum = j[0];
    let (mut tp, mut tm) = (C::ONE, C::ONE);
    for (n, jn) in j.iter().enumerate().take(n_max + 1).skip(1) {
        tp = tp * t;
        tm = tm * t_inv;
        let sign = if n % 2 == 0 { 1.0 } else { -1.0 };
        sum = sum + *jn * tp + *jn * tm * sign;
    }
    let want = ((z * 0.5) * (t - t_inv)).exp();
    (sum - want).abs() / want.abs().max(1.0)
}

fn rel(got: f64, want: f64) -> f64 {
    (got - want).abs() / want.abs().max(1e-300)
}

fn main() {
    println!("Complex-argument Bessel, integer order: where each kind gives out\n");

    // -----------------------------------------------------------------
    // 1. The six independent cross-checks.
    // -----------------------------------------------------------------
    println!("Six exact cross-checks against the vendored Cephes, each on the axis");
    println!("where that kind is at its WORST. Cephes is real-argument only, which");
    println!("is why the imaginary-axis entries go through the identities");
    println!("J_0(ix) = I_0(x) and I_0(ix) = J_0(x).\n");

    println!(
        "{:>5} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "x", "J_0(x)", "J_0(ix)", "I_0(x)", "I_0(ix)", "Y_0(x)", "K_0(x)"
    );
    println!(
        "{:>5} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "", "vs j0", "vs i0", "vs i0", "vs j0", "vs yn", "vs k0"
    );
    println!("  ----+{}", "-".repeat(66));
    for x in [1.0f64, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0] {
        let k = match bessel_k_c(0, C::real(x)) {
            Ok(v) => format!("{:>10.1e}", rel(v.re, k0(x))),
            Err(_) => format!("{:>10}", "refused"),
        };
        println!(
            "{x:>5.0} {:>10.1e} {:>10.1e} {:>10.1e} {:>10.1e} {:>10.1e} {k}",
            rel(bessel_j_c(0, C::real(x)).unwrap().re, j0(x)),
            rel(bessel_j_c(0, C::new(0.0, x)).unwrap().re, i0(x)),
            rel(bessel_i_c(0, C::real(x)).unwrap().re, i0(x)),
            rel(bessel_i_c(0, C::new(0.0, x)).unwrap().re, j0(x)),
            rel(bessel_y_c(0, C::real(x)).unwrap().re, yn(0, x)),
        );
    }

    println!("\nRead the columns, not the headline:");
    println!("  * J on the real axis is machine precision to x = 35.");
    println!("  * J up the imaginary axis, and I along the real axis, are the SAME");
    println!("    measurement — I_n(z) is J_n(iz) — which is why those columns match");
    println!("    digit for digit. Both lose |Re z| (resp. |Im z|) nepers.");
    println!("  * Y on the real axis is NOT like J. It comes from an ascending");
    println!("    series, so it loses |z| nepers: wrong in the first digit by x=40.");
    println!("  * K is the worst by far, and unusable past about x = 12.");

    // -----------------------------------------------------------------
    // 2. The laws.
    // -----------------------------------------------------------------
    println!("\n\nThe four laws, and how the measurement tracks them.\n");
    println!("Every one is the same statement: the working precision is spent on");
    println!("the ratio between the largest quantity the algorithm forms and the");
    println!("answer it has to produce.\n");
    println!("    relative error ~ 1e-16 * exp(L)\n");
    println!("  J_n   L = |Im z|                       (Miller: terms are ~e^|Im z|,");
    println!("                                          the normalising sum is 1)");
    println!("  I_n   L = |Re z|                       (the same, at right angles)");
    println!("  Y_n   L = |z| - |Im z|                 (series terms ~e^|z|,");
    println!("                                          |Y| ~ e^|Im z|)");
    println!("  K_n   L = max(2|Re z|, |z|) + Re z     (built from J and Y at iz,");
    println!("                                          answer of size e^-Re z)\n");

    println!("{:>5} {:>12} {:>12} {:>12} {:>12}", "x", "Y meas", "Y law", "K meas", "K law");
    for x in [1.0f64, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0] {
        let km = match bessel_k_c(0, C::real(x)) {
            Ok(v) => format!("{:>12.1e}", rel(v.re, k0(x))),
            Err(_) => format!("{:>12}", "refused"),
        };
        println!(
            "{x:>5.0} {:>12.1e} {:>12.1e} {km} {:>12.1e}",
            rel(bessel_y_c(0, C::real(x)).unwrap().re, yn(0, x)),
            1e-16 * x.exp(),
            1e-16 * (3.0 * x).exp(),
        );
    }
    println!("\nMeasured runs one to two decades BELOW the law throughout, which is");
    println!("the usual sqrt(2 pi |z|) that the neper counting drops. The law is the");
    println!("model; the table is the measurement; `integer_order_accuracy_laws_hold`");
    println!("pins the model with two digits of slack.");

    // -----------------------------------------------------------------
    // 3. The J-only measurement, correctly labelled.
    // -----------------------------------------------------------------
    println!("\n\nThe generating-function identity — J ONLY.\n");
    println!("This needs no reference implementation: both sides are computed here,");
    println!("the left from the values under test and the right from exp. It is a");
    println!("good check and a narrow one, and treating it as a statement about the");
    println!("module is exactly the mistake this example used to make.\n");

    let res: [f64; 6] = [0.0, 2.0, 5.0, 10.0, 15.0, 25.0];
    let ims: [f64; 7] = [0.0, 2.0, 5.0, 8.0, 12.0, 18.0, 25.0];

    print!("      Re z |");
    for re in res {
        print!(" {re:>9.0}");
    }
    println!();
    println!("  ---------+{}", "-".repeat(10 * res.len()));
    for im in ims {
        print!("  Im z {im:>4.0} |");
        for re in res {
            print!(" {:>9.1e}", gen_err(C::new(re, im)));
        }
        println!();
    }

    println!("\nFor J the error is governed by |Im z| almost independently of Re z,");
    println!("because the loss is cancellation in the normalising sum rather than");
    println!("anything about the recurrence.\n");

    println!("Digits of J retained, against the predicted 16 - |Im z|/ln(10):");
    println!("    {:>6} {:>12} {:>12}", "Im z", "measured", "predicted");
    for im in [0.0, 2.0, 5.0, 8.0, 12.0, 18.0, 25.0] {
        let e = gen_err(C::new(3.0, im));
        let measured = if e > 0.0 { -e.log10() } else { 16.0 };
        let predicted = 16.0 - im / std::f64::consts::LN_10;
        println!("    {im:>6.0} {measured:>12.1} {predicted:>12.1}");
    }

    println!("\nThat law is GENTLER than a first guess suggests. This note originally");
    println!("said J was \"worthless past |Im z| ~ 20\"; it is not — at |Im z| = 25");
    println!("five or six good digits remain.\n");

    // -----------------------------------------------------------------
    // 4. Guidance.
    // -----------------------------------------------------------------
    println!("Practical guidance, justified by the tables above:");
    println!("  |z| <= 10        every kind is good to 1e-12 or better, anywhere");
    println!("  J_n, I_n         good to |Im z| ~ 25 (resp. |Re z| ~ 25); 1e-13 at 8,");
    println!("                   1e-8 at 18, 1e-6 at 25");
    println!("  Y_n              |z| <= 15 for 1e-11; 1e-8 at 20; unusable past 30");
    println!("  K_n              Re z <= 8 for 1e-11; unusable past Re z ~ 12, and");
    println!("                   it REFUSES rather than lying past about 35");
    println!("  beyond that      a scaled or asymptotic method is needed, and none");
    println!("                   is implemented here");
    println!("\nFor non-integer order the laws are different again — simpler, since");
    println!("no Miller step is involved. See example `bessel_nu_accuracy`.");
}
