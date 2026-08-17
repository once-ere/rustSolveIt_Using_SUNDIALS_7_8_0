//! Independent verification of the **vendored** Cephes translation.
//!
//! Vendoring `spec_math` inherits Cephes's long track record, but a track
//! record is not evidence about *this* copy on *this* toolchain. These
//! tests check the vendored functions against mathematical identities we
//! evaluate ourselves — exact special values, Wronskians, reflection and
//! recurrence relations — so the claim "the classical chapters are
//! verified" rests on measurement rather than reputation.
//!
//! Argument conventions were pinned empirically before writing these
//! (a wrong convention yields tests that pass for the wrong reason):
//!   * `ellpe(m)`  takes the parameter m directly  — E(0) = pi/2, E(1) = 1
//!   * `ellpk(m1)` takes the COMPLEMENT m1 = 1 - m — K(m=0) = ellpk(1)
//!
//! Formula sources are Abramowitz & Stegun (1964), public domain. DLMF
//! equation numbers are cited for reference only.

use spec_math::cephes64::{airy, ellpe, ellpj, ellpk, erf, erfc, gamma, jv, psi, riemann_zeta, yv};
use special_functions::rel_err;
use special_functions::sph_bessel::sph_j;

const PI: f64 = std::f64::consts::PI;
/// Euler-Mascheroni constant (A&S 6.1.3).
const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;

// ---------------------------------------------------------------- gamma
// DLMF 5.4.6 / A&S 6.1.8 (reflection), A&S 6.1.6 (half-integer)

#[test]
fn gamma_exact_values_and_reflection() {
    // Gamma(1/2) = sqrt(pi)
    assert!(rel_err(gamma(0.5), PI.sqrt()) < 1e-15);
    // Gamma(n+1) = n!
    let mut fact = 1.0_f64;
    for n in 1..=15 {
        fact *= n as f64;
        assert!(rel_err(gamma((n + 1) as f64), fact) < 1e-13, "Gamma({}+1)", n);
    }
    // Reflection: Gamma(z)Gamma(1-z) = pi / sin(pi z)
    for &z in &[0.1, 0.25, 0.37, 0.5, 0.75, 0.9] {
        let lhs = gamma(z) * gamma(1.0 - z);
        assert!(rel_err(lhs, PI / (PI * z).sin()) < 1e-13, "reflection z={z}");
    }
    // Duplication: Gamma(z)Gamma(z+1/2) = 2^(1-2z) sqrt(pi) Gamma(2z)
    for &z in &[0.3, 0.8, 1.5, 2.7] {
        let lhs = gamma(z) * gamma(z + 0.5);
        let rhs = (2.0_f64).powf(1.0 - 2.0 * z) * PI.sqrt() * gamma(2.0 * z);
        assert!(rel_err(lhs, rhs) < 1e-12, "duplication z={z}");
    }
}

#[test]
fn digamma_at_one_is_negative_euler_gamma() {
    // psi(1) = -gamma  (A&S 6.3.2)
    assert!(rel_err(psi(1.0), -EULER_GAMMA) < 1e-13);
    // psi(z+1) = psi(z) + 1/z  (A&S 6.3.5)
    for &z in &[0.4, 1.3, 2.9, 7.5] {
        assert!(rel_err(psi(z + 1.0), psi(z) + 1.0 / z) < 1e-12, "psi rec z={z}");
    }
}

// ------------------------------------------------------- error function
// DLMF 7.2 / A&S 7.1

#[test]
fn erf_identities() {
    assert_eq!(erf(0.0), 0.0);
    for &x in &[0.1, 0.7, 1.0, 2.5, 4.0] {
        // erf + erfc = 1
        assert!(rel_err(erf(x) + erfc(x), 1.0) < 1e-14, "erf+erfc at {x}");
        // odd symmetry
        assert!(rel_err(erf(-x), -erf(x)) < 1e-14, "parity at {x}");
    }
    // erf(x) -> 1 as x grows
    assert!((erf(6.0) - 1.0).abs() < 1e-15);
}

// -------------------------------------------------------------- Airy
// DLMF 9.2 / A&S 10.4. Wronskian A&S 10.4.10.

#[test]
fn airy_wronskian_and_origin_values() {
    // Ai Bi' - Ai' Bi = 1/pi, everywhere
    for &x in &[-6.0, -2.0, -0.5, 0.0, 0.5, 2.0, 6.0] {
        let (ai, aip, bi, bip) = airy(x);
        assert!(
            rel_err(ai * bip - aip * bi, 1.0 / PI) < 1e-11,
            "Airy Wronskian at x={x}"
        );
    }
    // Ai(0) = 3^(-2/3)/Gamma(2/3), Bi(0) = 3^(-1/6)/Gamma(2/3)  (A&S 10.4.4-5)
    let (ai0, _, bi0, _) = airy(0.0);
    assert!(rel_err(ai0, (3.0_f64).powf(-2.0 / 3.0) / gamma(2.0 / 3.0)) < 1e-12);
    assert!(rel_err(bi0, (3.0_f64).powf(-1.0 / 6.0) / gamma(2.0 / 3.0)) < 1e-12);
}

// ------------------------------------------------------------- Bessel
// DLMF 10.5.2 (Wronskian) / A&S 9.1.16; A&S 9.1.27 (recurrence)

fn jv_prime(v: f64, x: f64) -> f64 {
    0.5 * (jv(v - 1.0, x) - jv(v + 1.0, x))
}
fn yv_prime(v: f64, x: f64) -> f64 {
    0.5 * (yv(v - 1.0, x) - yv(v + 1.0, x))
}

#[test]
fn bessel_wronskian() {
    // J_v(x) Y_v'(x) - J_v'(x) Y_v(x) = 2/(pi x)
    for &v in &[0.0, 0.5, 1.0, 2.0, 3.7] {
        for &x in &[0.5, 1.0, 3.0, 10.0, 30.0] {
            let w = jv(v, x) * yv_prime(v, x) - jv_prime(v, x) * yv(v, x);
            assert!(
                rel_err(w, 2.0 / (PI * x)) < 1e-9,
                "Bessel Wronskian v={v} x={x}: {w}"
            );
        }
    }
}

#[test]
fn bessel_half_integer_closed_form() {
    // J_{1/2}(x) = sqrt(2/(pi x)) sin x   (A&S 10.1.11)
    for &x in &[0.3, 1.0, 4.0, 12.0] {
        let expect = (2.0 / (PI * x)).sqrt() * x.sin();
        assert!(rel_err(jv(0.5, x), expect) < 1e-12, "J_1/2 at {x}");
        // J_{-1/2}(x) = sqrt(2/(pi x)) cos x
        let expect = (2.0 / (PI * x)).sqrt() * x.cos();
        assert!(rel_err(jv(-0.5, x), expect) < 1e-12, "J_-1/2 at {x}");
    }
}

#[test]
fn bessel_recurrence() {
    // J_{v-1}(x) + J_{v+1}(x) = (2v/x) J_v(x)
    for &v in &[1.0, 2.5, 5.0] {
        for &x in &[1.0, 4.0, 15.0] {
            let lhs = jv(v - 1.0, x) + jv(v + 1.0, x);
            assert!(rel_err(lhs, 2.0 * v / x * jv(v, x)) < 1e-11, "rec v={v} x={x}");
        }
    }
}

// -------- cross-validation: OUR native module against the VENDORED one --
// This is the strongest check in the file: two independently written
// implementations, one ours (Miller downward recurrence) and one Cephes's
// (cylindrical J of half-integer order), must agree via DLMF 10.47.3:
//     j_n(x) = sqrt(pi/2x) J_{n+1/2}(x)

#[test]
fn native_spherical_bessel_agrees_with_vendored_cylindrical() {
    for n in 0..10 {
        for &x in &[0.25, 0.9, 2.0, 5.5, 13.0, 40.0] {
            let ours = sph_j(n, x).unwrap();
            let theirs = (PI / (2.0 * x)).sqrt() * jv(n as f64 + 0.5, x);
            // Loosened only where both are denormal-small (n >> x).
            let tol = if ours.abs() < 1e-15 { 1e-7 } else { 1e-9 };
            assert!(
                rel_err(ours, theirs) < tol,
                "j_{n}({x}): ours={ours:e} vendored={theirs:e}"
            );
        }
    }
}

// ---------------------------------------------------------- elliptic
// DLMF 19.7.1 (Legendre relation) / A&S 17.3.13

#[test]
fn elliptic_special_values_and_legendre_relation() {
    // K(m=0) = E(m=0) = pi/2 ; E(m=1) = 1
    assert!(rel_err(ellpk(1.0), PI / 2.0) < 1e-14); // ellpk takes m1 = 1-m
    assert!(rel_err(ellpe(0.0), PI / 2.0) < 1e-14);
    assert!(rel_err(ellpe(1.0), 1.0) < 1e-14);

    // Legendre: E(m)K(1-m) + E(1-m)K(m) - K(m)K(1-m) = pi/2
    for &m in &[0.1, 0.3, 0.5, 0.7, 0.9] {
        let m1 = 1.0 - m;
        let (k_m, k_m1) = (ellpk(m1), ellpk(m));
        let (e_m, e_m1) = (ellpe(m), ellpe(m1));
        let lhs = e_m * k_m1 + e_m1 * k_m - k_m * k_m1;
        assert!(rel_err(lhs, PI / 2.0) < 1e-12, "Legendre relation m={m}");
    }
}

#[test]
fn jacobi_elliptic_pythagorean_identities() {
    // sn^2 + cn^2 = 1 and m*sn^2 + dn^2 = 1   (A&S 16.1.1-16.1.2)
    for &m in &[0.0, 0.2, 0.5, 0.8, 0.99] {
        for &u in &[0.0, 0.4, 1.1, 2.7, 5.0] {
            let (sn, cn, dn, _) = ellpj(u, m);
            assert!(rel_err(sn * sn + cn * cn, 1.0) < 1e-13, "sn2+cn2 m={m} u={u}");
            assert!(
                rel_err(m * sn * sn + dn * dn, 1.0) < 1e-13,
                "m sn2+dn2 m={m} u={u}"
            );
        }
    }
}

// -------------------------------------------------------------- zeta
// DLMF 25.6.1 / A&S 23.2

#[test]
fn zeta_at_even_integers() {
    assert!(rel_err(riemann_zeta(2.0), PI * PI / 6.0) < 1e-13);
    assert!(rel_err(riemann_zeta(4.0), PI.powi(4) / 90.0) < 1e-13);
    assert!(rel_err(riemann_zeta(6.0), PI.powi(6) / 945.0) < 1e-13);
}
