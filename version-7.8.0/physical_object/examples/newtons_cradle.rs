//! Newton's cradle, self-checking: an incomer strikes a chain of four
//! touching spheres; the impulse must propagate through the chain so
//! that ONLY the far sphere exits, at the incomer's speed, with total
//! momentum conserved. All integration is sundials CVODE with event
//! rootfinding; the multi-contact propagation is the collision
//! solver's Gauss–Seidel pass.
#![forbid(unsafe_code)]
#![deny(warnings)]
#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]

use ::physical_object::boundary::Boundary;
use ::physical_object::integrate;
use ::physical_object::linalg::Vec3;
use ::physical_object::physical_object::physical_object;
use ::physical_object::PhysicalObjectSystem;

fn sphere(id: usize, x: f64, vx: f64) -> physical_object {
    physical_object::new_from_shape(
        id,
        1.0,
        0.0,
        Vec3::new(x, 0.0, 0.0),
        Vec3::new(vx, 0.0, 0.0),
        Vec3::zeros(),
        Boundary::Sphere { radius: 0.5 },
    )
}

fn main() {
    let mut sys = PhysicalObjectSystem::new(
        vec![
            sphere(0, -3.0, 1.0),
            sphere(1, 0.0, 0.0),
            sphere(2, 1.0, 0.0),
            sphere(3, 2.0, 0.0),
            sphere(4, 3.0, 0.0),
        ],
        0.0,
    );

    let report = integrate::run(&mut sys, 4.0, 1).expect("cradle run");
    let v: Vec<f64> = sys.objects.iter().map(|o| o.get_velocity().x).collect();
    let p = sys.total_momentum();

    println!("Newton's cradle: 1 incomer at v = 1 into 4 touching spheres");
    println!("  collisions resolved : {}", report.ncollisions);
    println!("  final velocities    : {v:?}");
    println!("  total momentum      : [{}, {}, {}]", p.x, p.y, p.z);

    let chain_still = v[0].abs() < 1e-6 && v[1].abs() < 1e-6 && v[2].abs() < 1e-6 && v[3].abs() < 1e-6;
    let far_exits = (v[4] - 1.0).abs() < 1e-6;
    let momentum_ok = (p.x - 1.0).abs() < 1e-9 && p.y.abs() < 1e-12 && p.z.abs() < 1e-12;

    assert!(chain_still, "FAILURE: the chain must stay at rest, got {v:?}");
    assert!(far_exits, "FAILURE: the far sphere must exit at v = 1, got {}", v[4]);
    assert!(momentum_ok, "FAILURE: momentum not conserved: {p:?}");
    println!("SUCCESS: impulse propagated through the chain, momentum conserved");
}
