//! Where the non-integer-order Bessel functions stop being trustworthy.
//!
//! The integer-order complex routines use Miller recurrence and lose
//! accuracy with `|Im z|`. The non-integer-order routines are a
//! completely different algorithm — an ascending power series — so they
//! have a completely different failure surface, and the point of this
//! example is that the two are *not* interchangeable advice.
//!
//! The measurement uses closed forms rather than a reference library.
//! At half-integer order the answers are elementary and exact for
//! **complex** `z` as well as real:
//!
//! ```text
//!     J_{1/2}(z) = sqrt(2/(pi z)) sin z         (DLMF 10.16.1)
//!     K_{1/2}(z) = sqrt(pi/(2z)) exp(-z)        (DLMF 10.39.2)
//! ```
//!
//! Neither right-hand side shares any code with the series, so what is
//! printed below is a genuine relative error, not a self-consistency
//! check.
//!
//! Run: cargo run -p special_functions --release --example bessel_nu_accuracy

use special_functions::bessel_complex::{bessel_j_nu, bessel_k_nu};
use special_functions::complex::Complex64 as C;

/// `sin z` for complex `z`, built from `exp` — no Bessel code involved.
fn csin(z: C) -> C {
    ((C::I * z).exp() - (C::I * z * -1.0).exp()) / (C::I * 2.0)
}

fn j_half_err(z: C) -> f64 {
    let got = bessel_j_nu(0.5, z).unwrap();
    let want = (C::real(2.0 / std::f64::consts::PI) * z.inv()).powf(0.5) * csin(z);
    (got - want).abs() / want.abs()
}

fn k_half_err(z: C) -> f64 {
    let got = bessel_k_nu(0.5, z).unwrap();
    let want = (C::real(std::f64::consts::PI * 0.5) * z.inv()).powf(0.5) * (z * -1.0).exp();
    (got - want).abs() / want.abs()
}

const ARGS: [f64; 6] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0];

fn table(name: &str, radii: &[f64], f: impl Fn(C) -> f64) {
    println!("{name}");
    print!("     |z| |");
    for a in ARGS {
        print!(" {a:>9.2}");
    }
    println!("   <- arg z (radians)");
    println!("  ------+{}", "-".repeat(10 * ARGS.len()));
    for &r in radii {
        print!("  {r:>6.0} |");
        for a in ARGS {
            print!(" {:>9.1e}", f(C::from_polar(r, a)));
        }
        println!();
    }
    println!();
}

fn main() {
    println!("Non-integer order: measured relative error\n");

    table(
        "J_{1/2}(z) against sqrt(2/(pi z)) sin z",
        &[1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0, 70.0],
        j_half_err,
    );

    table(
        "K_{1/2}(z) against sqrt(pi/(2z)) exp(-z)",
        &[1.0, 2.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0],
        k_half_err,
    );

    println!("Read the two tables together — they fail in OPPOSITE directions.\n");

    println!("Both series have terms whose largest is of size ~exp(|z|), because");
    println!("the term ratio depends on |z| alone. What differs is the size of the");
    println!("answer they must add up to, and the ratio of the two IS the");
    println!("cancellation — so one law covers both:\n");
    println!("      digits lost  =  [ |z| - ln|answer| ] / ln 10\n");
    println!("  J_nu(z) ~ exp(|Im z|)   ->  lost = (|z| - |Im z|) / ln 10");
    println!("  K_nu(z) ~ exp(-Re z)    ->  lost = (|z| + Re z)  / ln 10\n");
    println!("K is small precisely because it is a DIFFERENCE of two I's, each of");
    println!("size exp(Re z), whose leading parts cancel; that cancellation is the");
    println!("definition of K, not a defect of the method.\n");
    println!("So J is worst on the real axis and exact up the imaginary axis;");
    println!("K is worst on the positive real axis and exact along the negative");
    println!("one. The arg = 1.5 column of the J table (nearly imaginary z) is at");
    println!("machine precision out to |z| = 70; the arg = 3.0 column of the K");
    println!("table (nearly negative-real z) likewise.\n");

    println!("Digits kept, measured against the predicted laws:");
    println!(
        "  {:>6} {:>10} {:>10} {:>4} {:>10} {:>10}",
        "|z|", "J real", "predicted", "", "K real", "predicted"
    );
    for r in [1.0f64, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0] {
        let jd = -j_half_err(C::real(r)).log10();
        let kd = -k_half_err(C::real(r)).log10();
        // On the positive real axis |Im z| = 0 and Re z = |z|, so the two
        // laws specialise to 16 - |z|/ln10 and 16 - 2|z|/ln10. This is the
        // worst case for both functions at once.
        let jp = 16.0 - r / std::f64::consts::LN_10;
        let kp = 16.0 - 2.0 * r / std::f64::consts::LN_10;
        println!("  {r:>6.0} {jd:>10.1} {jp:>10.1} {:>4} {kd:>10.1} {kp:>10.1}", "");
    }

    println!("\nMeasured tracks predicted to about a digit, and is consistently a");
    println!("little BETTER — the largest term is exp(|z|)/sqrt(2 pi |z|), not");
    println!("exp(|z|), and the missing sqrt is worth roughly that much.\n");

    // Large ORDER is a separate axis, and the closed form does not reach
    // it, so this uses the order recurrence instead — three independent
    // series evaluations that must satisfy an exact identity.
    println!("Large order, at z = 3 + i: residual in");
    println!("J_(nu-1) + J_(nu+1) = (2 nu / z) J_nu\n");
    println!("  {:>8} {:>12} {:>12}", "nu", "residual", "|J_nu(z)|");
    let z = C::new(3.0, 1.0);
    for nu in [0.3f64, 5.5, 10.5, 20.5, 40.5, 80.5, 150.5] {
        let lhs = bessel_j_nu(nu - 1.0, z).unwrap() + bessel_j_nu(nu + 1.0, z).unwrap();
        let rhs = bessel_j_nu(nu, z).unwrap() * (z.inv() * (2.0 * nu));
        let res = (lhs - rhs).abs() / rhs.abs();
        println!("  {nu:>8.1} {res:>12.1e} {:>12.1e}", bessel_j_nu(nu, z).unwrap().abs());
    }
    println!("\nNo degradation with order, and the values stay meaningful down to");
    println!("1e-234. That is why the series uses the RECIPROCAL gamma: Gamma(nu+k+1)");
    println!("overflows f64 past about 171, so dividing by it would return inf and");
    println!("then 0 for every term, whereas 1/Gamma simply becomes very small.\n");

    println!("Practical guidance. Write L for the loss exponent of the law above");
    println!("— L = |z| - |Im z| for J and Y, L = |z| + Re z for I and K:\n");
    println!("      relative error  ~  1e-16 * exp(L)\n");
    println!("    L <= 10   typically 1e-12, never worse than 2e-10");
    println!("    L <= 20   typically 1e-8,  never worse than 5e-6");
    println!("    L <= 30   typically 1e-4,  never worse than 1e-1");
    println!("    L >  35   no correct digits at all\n");
    println!("The \"never worse\" column is the model with two decimal digits of");
    println!("slack, and that is exactly what the test suite pins — a bound of");
    println!("1e-14 * exp(L) at every point of the two tables above. Pinning the");
    println!("law rather than a bucketed tolerance matters: L is only the leading");
    println!("term, and the number of terms summed and the sqrt(2 pi |z|) in the");
    println!("prefactor both grow slowly with |z| on top of it.\n");
    println!("On the real axis, where each is at its worst, that is |z| <= 10, 20,");
    println!("30 for J and Y, and |z| <= 5, 10, 15 for I and K.\n");
    println!("  Large ORDER is free: at nu = 150 and |z| = 3 the order recurrence");
    println!("  still closes to 1e-13, because nothing cancels when nu >> |z|.");
    println!("\n  Beyond those ranges use the INTEGER-order routines where the");
    println!("  order allows it — they are Miller recurrence and hold up much");
    println!("  further along the real axis. Uniform asymptotics for large |z|");
    println!("  at non-integer order are NOT implemented here.");
}
