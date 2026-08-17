//! Bouncing ball, self-checking: a ball dropped on a static slab under
//! uniform gravity with restitution e = 0.8 must rebound to e² of the
//! drop height, and the impact time must match √(2h/g). The impact is
//! detected by SUNDIALS event rootfinding at the exact touch.
#![forbid(unsafe_code)]
#![deny(warnings)]
#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]

use ::physical_object::boundary::Boundary;
use ::physical_object::integrate;
use ::physical_object::linalg::{Mat3, Vec3};
use ::physical_object::physical_object::physical_object;
use ::physical_object::PhysicalObjectSystem;

fn main() {
    let mut floor = physical_object::new_from_shape(
        0,
        1.0,
        0.0,
        Vec3::new(0.0, -0.5, 0.0),
        Vec3::zeros(),
        Vec3::zeros(),
        Boundary::Cuboid { half_extents: [5.0, 0.5, 5.0] },
    );
    floor.set_inverse_mass(0.0);
    floor.set_inverse_inertia_tensor(Mat3::zeros());

    let mut ball = physical_object::new_from_shape(
        1,
        1.0,
        0.0,
        Vec3::new(0.0, 5.0, 0.0),
        Vec3::zeros(),
        Vec3::zeros(),
        Boundary::Sphere { radius: 0.5 },
    );
    ball.set_restitution(0.8);

    let mut sys = PhysicalObjectSystem::new(vec![floor, ball], 0.0);
    sys.uniform_gravity = Vec3::new(0.0, -10.0, 0.0);

    // Drop height of the center above rest (0.5): h = 4.5.
    let report = integrate::run(&mut sys, 1.8, 360).expect("bounce run");
    let c = sys.contacts.first().expect("an impact must be recorded");
    let t_analytic = (2.0f64 * 4.5 / 10.0).sqrt();
    let apex = report
        .snapshots
        .iter()
        .filter(|s| s.t > c.t)
        .map(|s| s.center_of_mass.y)
        .fold(f64::NEG_INFINITY, f64::max);
    let apex_analytic = 0.5 + 0.8 * 0.8 * 4.5; // rest height + e² h

    println!("Bouncing ball: h = 4.5, g = 10, e = 0.8");
    println!("  impact time  : {} (analytic {t_analytic})", c.t);
    println!("  impact normal: [{}, {}, {}] (floor → ball)", c.normal.x, c.normal.y, c.normal.z);
    println!("  rebound apex : {apex} (analytic {apex_analytic})");

    assert!((c.t - t_analytic).abs() < 1e-7, "FAILURE: impact time off: {}", c.t);
    assert!((c.normal - Vec3::new(0.0, 1.0, 0.0)).norm() < 1e-9, "FAILURE: normal must be +y");
    assert!((apex - apex_analytic).abs() < 2e-3, "FAILURE: apex {apex} vs {apex_analytic}");
    println!("SUCCESS: rebound apex = e^2 x drop height, impact at the exact TOI");
}
