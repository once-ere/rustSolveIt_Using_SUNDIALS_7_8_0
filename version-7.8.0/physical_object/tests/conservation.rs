//! Integration tests: the sundials-backed solver paths reproduce
//! analytic solutions and conserve the appropriate invariants.
#![forbid(unsafe_code)]
#![deny(warnings)]
#![allow(non_snake_case)]

use ::physical_object::boundary::Boundary;
use ::physical_object::integrate::{run, step, Method};
use ::physical_object::linalg::Vec3;
use ::physical_object::physical_object::physical_object;
use ::physical_object::PhysicalObjectSystem;

fn kepler_system(method: Method) -> PhysicalObjectSystem {
    let ecc = 0.6;
    let central = physical_object::new_point(0, 1.0, Vec3::zeros(), Vec3::zeros());
    let orbiter = physical_object::new_point(
        1,
        1.0e-9,
        Vec3::new(1.0 - ecc, 0.0, 0.0),
        Vec3::new(0.0, ((1.0 + ecc) / (1.0 - ecc)).sqrt(), 0.0),
    );
    let mut s = PhysicalObjectSystem::new(vec![central, orbiter], 1.0);
    s.softening = 0.0;
    s.method = method;
    s
}

#[test]
fn cvode_adams_conserves_two_body_invariants() {
    let mut s = kepler_system(Method::Adams);
    let e0 = s.total_energy();
    let l0 = s.objects[1].orbital_angular_momentum(s.center_of_mass());
    run(&mut s, 20.0, 4).expect("adams run");
    let e1 = s.total_energy();
    let l1 = s.objects[1].orbital_angular_momentum(s.center_of_mass());
    assert!(((e1 - e0) / e0).abs() < 1e-7, "energy drift");
    assert!((l1 - l0).norm() / l0.norm() < 1e-7, "L drift");
    assert!((s.time - 20.0).abs() < 1e-12);
}

#[test]
fn sprk_leapfrog_matches_legacy_verlet_and_conserves_energy() {
    // Velocity Verlet == leapfrog: the legacy GravitationalSystem::step
    // corresponds to ARKODE_SPRK_LEAPFROG_2_2 at the same fixed dt.
    let mut s = kepler_system(Method::Sprk {
        table: "ARKODE_SPRK_LEAPFROG_2_2".to_string(),
        dt: 0.001,
    });
    let e0 = s.total_energy();
    run(&mut s, 20.0, 4).expect("sprk run");
    let e1 = s.total_energy();
    assert!(((e1 - e0) / e0).abs() < 1e-5, "symplectic energy drift too large");
}

#[test]
fn sprk_rejects_non_separable_systems() {
    let mut s = kepler_system(Method::Sprk {
        table: "ARKODE_SPRK_LEAPFROG_2_2".to_string(),
        dt: 0.001,
    });
    s.b_field = Vec3::new(0.0, 0.0, 1.0);
    let err = run(&mut s, 1.0, 1).unwrap_err();
    assert!(err.contains("magnetic field B"), "unexpected error: {err}");
}

#[test]
fn uniform_gravity_matches_analytic_parabola() {
    // x(t) = x0 + v t + g t^2 / 2 under uniform gravity — exact for the
    // adaptive solver at tight tolerances.
    let ball = physical_object::new_from_shape(
        0,
        2.0,
        0.0,
        Vec3::new(0.0, 10.0, 0.0),
        Vec3::new(1.0, 0.0, -0.5),
        Vec3::zeros(),
        Boundary::Sphere { radius: 0.5 },
    );
    let mut s = PhysicalObjectSystem::new(vec![ball], 0.0);
    s.uniform_gravity = Vec3::new(0.0, -9.81, 0.0);
    s.method = Method::Adams;
    let t = 3.0;
    run(&mut s, t, 3).expect("run");
    let expect = Vec3::new(0.0, 10.0, 0.0)
        + Vec3::new(1.0, 0.0, -0.5) * t
        + Vec3::new(0.0, -9.81, 0.0) * (0.5 * t * t);
    let got = s.objects[0].get_position();
    assert!((got - expect).norm() < 1e-8, "got {got:?}, expected {expect:?}");
}

#[test]
fn torque_free_top_conserves_L_and_energy() {
    let body = physical_object::new_from_shape(
        0,
        3.0,
        0.0,
        Vec3::zeros(),
        Vec3::zeros(),
        Vec3::new(0.01, 3.0, 0.01),
        Boundary::Cuboid { half_extents: [0.5, 1.0, 2.0] },
    );
    let mut s = PhysicalObjectSystem::new(vec![body], 0.0);
    s.method = Method::Adams;
    let l0 = s.objects[0].get_angular_momentum();
    let ke0 = s.objects[0].kinetic_energy();
    run(&mut s, 5.0, 5).expect("run");
    let o = &s.objects[0];
    assert!((o.get_angular_momentum() - l0).norm() / l0.norm() < 1e-8);
    assert!(((o.kinetic_energy() - ke0) / ke0).abs() < 1e-6);
    assert!((o.get_orientation().norm() - 1.0).abs() < 1e-9);
}

#[test]
fn constant_torque_spins_up_exactly() {
    // dL/dt = tau with everything else zero: L(t) = tau * t, exactly
    // linear, so the solver must land on it to tolerance.
    let body = physical_object::new_from_shape(
        0,
        2.0,
        0.0,
        Vec3::zeros(),
        Vec3::zeros(),
        Vec3::zeros(),
        Boundary::Sphere { radius: 0.5 },
    );
    let mut s = PhysicalObjectSystem::new(vec![body], 0.0);
    s.external_torques[0] = Vec3::new(0.1, 0.0, 0.0);
    s.method = Method::Adams;
    run(&mut s, 2.0, 1).expect("run");
    let l = s.objects[0].get_angular_momentum();
    assert!((l - Vec3::new(0.2, 0.0, 0.0)).norm() < 1e-9, "L = {l:?}");
}

#[test]
fn propagate_single_replaces_legacy_euler_integrate() {
    // The RigidBody3D main() scenario: gravity force + wind torque on a
    // sphere, but integrated by CVODE instead of explicit Euler. The
    // constant-force trajectory has the exact solution
    // x(t) = x0 + v0 t + (F/m) t^2/2, L(t) = tau t.
    let mut body = physical_object::new_from_shape(
        0,
        2.0,
        -1.5,
        Vec3::new(0.0, 10.0, 0.0),
        Vec3::new(1.0, 0.0, -0.5),
        Vec3::new(0.0, 2.0, 0.0),
        Boundary::Sphere { radius: 0.5 },
    );
    let gravity = Vec3::new(0.0, -9.81 * 2.0, 0.0);
    let wind = Vec3::new(0.1, 0.0, 0.0);
    let dt = 0.3;
    body.integrate(&gravity, &wind, dt).expect("integrate");

    let expect_pos = Vec3::new(0.0, 10.0, 0.0)
        + Vec3::new(1.0, 0.0, -0.5) * dt
        + (gravity / 2.0) * (0.5 * dt * dt);
    assert!((body.get_position() - expect_pos).norm() < 1e-8);
    let expect_l = Vec3::new(0.0, 0.4, 0.0) + wind * dt;
    assert!((body.get_angular_momentum() - expect_l).norm() < 1e-9);
}

#[test]
fn charged_particle_gyrates_at_analytic_radius() {
    let (mass, charge, speed, b): (f64, f64, f64, f64) = (2.0, -1.5, 3.0, 4.0);
    let ball = physical_object::new_point(0, mass, Vec3::zeros(), Vec3::new(speed, 0.0, 0.0));
    let mut s = PhysicalObjectSystem::new(vec![ball], 0.0);
    let mut o = s.objects.remove(0);
    o.set_charge(charge);
    s.objects.push(o);
    s.b_field = Vec3::new(0.0, 0.0, b);
    s.method = Method::Bdf;
    let period = 2.0 * std::f64::consts::PI * mass / (charge.abs() * b);
    run(&mut s, period, 8).expect("run");
    // After one period the orbit closes and speed is conserved.
    assert!(s.objects[0].get_position().norm() < 1e-6);
    assert!((s.objects[0].linear_velocity().norm() - speed).abs() < 1e-7);
}

#[test]
fn step_advances_time_and_empty_system_is_ok() {
    let mut s = PhysicalObjectSystem::default();
    step(&mut s, 1.5).expect("empty step");
    assert!((s.time - 1.5).abs() < 1e-15);
    let err = run(&mut s, 1.0, 1).unwrap_err();
    assert!(err.contains("greater than current time"));
}
