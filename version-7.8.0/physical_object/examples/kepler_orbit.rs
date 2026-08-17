/* Kepler two-body problem through the physical_object API.
 *
 * A test particle orbits a heavy central mass with eccentricity
 * e = 0.6 and G*M = 1 — the same orbit as the SUNDIALS `ark_kepler`
 * example (reduced form). Checked invariants:
 *   - total energy,
 *   - orbital angular momentum L = r x p,
 *   - the Laplace-Runge-Lenz vector A (points along the major axis;
 *     its conservation is the hallmark of the exact 1/r potential).
 *
 * Runs both solver paths: CVODE Adams (adaptive) and the symplectic
 * ARKODE SPRKStep McLachlan 4-4 (fixed step). */
#![forbid(unsafe_code)]
#![deny(warnings)]
#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]

use ::physical_object::linalg::Vec3;
use ::physical_object::physical_object::physical_object;
use ::physical_object::integrate::run;
use ::physical_object::{Method, PhysicalObjectSystem};
use sundials_core::sundials_utils::fmt_e;

const ECC: f64 = 0.6;
/// A unit-mass orbiter around a heavy central body with G scaled so
/// that G * M_total = 1: this reproduces ark_kepler's central-force
/// form, and with m = 1 the legacy Laplace-vector recipe
/// `A = p x L - m (G M_total) r_hat` is the true conserved LRL vector.
const M_CENTRAL: f64 = 1.0e9;
const TF: f64 = 100.0; /* ~15.9 orbits (period 2*pi) */

fn build_system(method: Method) -> PhysicalObjectSystem {
    let central = physical_object::new_point(0, M_CENTRAL, Vec3::zeros(), Vec3::zeros());
    let orbiter = physical_object::new_point(
        1,
        1.0,
        Vec3::new(1.0 - ECC, 0.0, 0.0),
        Vec3::new(0.0, ((1.0 + ECC) / (1.0 - ECC)).sqrt(), 0.0),
    );
    let mut system = PhysicalObjectSystem::new(vec![central, orbiter], 1.0 / (M_CENTRAL + 1.0));
    system.softening = 0.0;
    system.method = method;
    system
}

fn report_drifts(label: &str, system: &mut PhysicalObjectSystem) -> (f64, f64, f64) {
    let com = system.center_of_mass();
    let k = system.g_constant * system.total_mass();
    let e0 = system.total_energy();
    let l0 = system.objects[1].orbital_angular_momentum(com);
    let a0 = system.objects[1].laplace_vector(com, k);

    let report = run(system, TF, 10).expect(label);

    let com = system.center_of_mass();
    let e1 = system.total_energy();
    let l1 = system.objects[1].orbital_angular_momentum(com);
    let a1 = system.objects[1].laplace_vector(com, k);

    let de = ((e1 - e0) / e0).abs();
    let dl = (l1 - l0).norm() / l0.norm();
    let da = (a1 - a0).norm() / a0.norm().max(f64::MIN_POSITIVE);

    println!("\n{label}");
    println!("  steps taken            = {}", report.nst);
    println!("  |dE/E|                 = {}", fmt_e(de, 6));
    println!("  |dL|/|L|               = {}", fmt_e(dl, 6));
    println!("  |dA|/|A| (Runge-Lenz)  = {}", fmt_e(da, 6));
    (de, dl, da)
}

fn main() {
    println!("Kepler orbit, e = {ECC}, t in [0, {TF}] (~15.9 orbits)");

    let mut adams = build_system(Method::Adams);
    let (de, dl, da) = report_drifts("CVODE Adams (rtol 1e-10, atol 1e-12)", &mut adams);
    assert!(de < 1.0e-6, "Adams energy drift too large: {de}");
    assert!(dl < 1.0e-6, "Adams angular momentum drift too large: {dl}");
    assert!(da < 1.0e-4, "Adams Laplace vector drift too large: {da}");

    let mut sprk = build_system(Method::Sprk {
        table: "ARKODE_SPRK_MCLACHLAN_4_4".to_string(),
        dt: 0.01,
    });
    let (de, dl, _da) = report_drifts("ARKODE SPRKStep MCLACHLAN_4_4 (fixed dt = 0.01)", &mut sprk);
    assert!(de < 1.0e-6, "SPRK energy drift too large: {de}");
    assert!(dl < 1.0e-8, "SPRK angular momentum drift too large: {dl}");

    println!("\nSUCCESS: all conservation checks passed");
}
