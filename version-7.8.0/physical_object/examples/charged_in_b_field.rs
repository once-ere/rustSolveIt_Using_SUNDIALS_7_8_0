/* Charged sphere gyrating in a uniform magnetic field.
 *
 * F = q v x B with v ⟂ B produces circular motion with
 *   gyroradius  r = m v / (|q| B)
 *   period      T = 2 pi m / (|q| B).
 * The Lorentz force is velocity-dependent (non-separable), so this
 * exercises the CVODE BDF path. Checks: speed conservation and the
 * analytic gyroradius extracted from the trajectory. */
#![forbid(unsafe_code)]
#![deny(warnings)]
#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]

use ::physical_object::boundary::Boundary;
use ::physical_object::linalg::Vec3;
use ::physical_object::physical_object::physical_object;
use ::physical_object::integrate::run;
use ::physical_object::{Method, PhysicalObjectSystem};
use sundials_core::sundials_utils::{fmt_e, fmt_f};

fn main() {
    let mass = 2.0;
    let charge = -1.5;
    let speed = 3.0;
    let b = 4.0;

    let ball = physical_object::new_from_shape(
        0,
        mass,
        charge,
        Vec3::zeros(),
        Vec3::new(speed, 0.0, 0.0),
        Vec3::zeros(),
        Boundary::Sphere { radius: 0.1 },
    );

    let mut system = PhysicalObjectSystem::new(vec![ball], 0.0);
    system.b_field = Vec3::new(0.0, 0.0, b);
    system.method = Method::Bdf;
    system.rtol = 1.0e-10;
    system.atol = 1.0e-12;

    let r_expect = mass * speed / (charge.abs() * b);
    let period = 2.0 * std::f64::consts::PI * mass / (charge.abs() * b);

    /* one full gyration, sampled densely to measure the radius */
    let report = run(&mut system, period, 200).expect("run");

    /* the orbit circles about a guiding center at distance r from the
     * start; radius = half the max displacement across the circle */
    let mut max_disp: f64 = 0.0;
    for s in &report.snapshots {
        max_disp = max_disp.max(s.center_of_mass.norm());
    }
    let r_measured = max_disp / 2.0;

    let v_final = system.objects[0].linear_velocity().norm();
    let closure = system.objects[0].get_position().norm();

    println!("Charged sphere in uniform B (q = {charge}, B = {b}, |v| = {speed})");
    println!("  expected gyroradius   = {}", fmt_f(r_expect, 6));
    println!("  measured gyroradius   = {}", fmt_f(r_measured, 6));
    println!("  |v| drift             = {}", fmt_e((v_final - speed).abs() / speed, 6));
    println!("  orbit closure |x(T)|  = {}", fmt_e(closure, 6));

    assert!((r_measured - r_expect).abs() / r_expect < 1.0e-4, "gyroradius mismatch");
    assert!((v_final - speed).abs() / speed < 1.0e-7, "speed not conserved");
    assert!(closure < 1.0e-5, "orbit did not close after one period");
    println!("SUCCESS: gyration matches analytic solution");
}
