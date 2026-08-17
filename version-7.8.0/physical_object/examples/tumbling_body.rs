/* Torque-free tumbling rigid body (Dzhanibekov / tennis-racket setup).
 *
 * A cuboid with three distinct principal moments spins almost exactly
 * about its unstable intermediate axis; the CVODE path integrates the
 * full quaternion + angular-momentum state. Torque-free invariants:
 *   - the world-frame angular momentum vector L (all 3 components),
 *   - the rotational kinetic energy 1/2 w . L,
 *   - the unit norm of the orientation quaternion. */
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
use sundials_core::sundials_utils::fmt_e;

fn main() {
    /* half-extents (0.5, 1.0, 2.0) -> distinct principal moments */
    let body = physical_object::new_from_shape(
        0,
        3.0,
        0.0,
        Vec3::zeros(),
        Vec3::zeros(),
        /* spin about the intermediate axis + tiny perturbation */
        Vec3::new(1.0e-4, 3.0, 1.0e-4),
        Boundary::Cuboid { half_extents: [0.5, 1.0, 2.0] },
    );

    let mut system = PhysicalObjectSystem::new(vec![body], 0.0);
    system.method = Method::Adams;

    let l0 = system.objects[0].get_angular_momentum();
    let ke0 = system.objects[0].kinetic_energy();

    let report = run(&mut system, 25.0, 25).expect("run");

    let o = &system.objects[0];
    let l1 = o.get_angular_momentum();
    let ke1 = o.kinetic_energy();
    let qerr = (o.get_orientation().norm() - 1.0).abs();
    let dl = (l1 - l0).norm() / l0.norm();
    let dke = ((ke1 - ke0) / ke0).abs();

    println!("Torque-free tumbling cuboid, t in [0, 25], {} snapshots", report.snapshots.len());
    println!("  internal steps    = {}", report.nst);
    println!("  |dL|/|L|          = {}", fmt_e(dl, 6));
    println!("  |dKE|/KE          = {}", fmt_e(dke, 6));
    println!("  | |q| - 1 |       = {}", fmt_e(qerr, 6));
    println!("  final w           = {:?}", o.angular_velocity());

    assert!(dl < 1.0e-7, "angular momentum drift too large: {dl}");
    assert!(dke < 1.0e-6, "rotational energy drift too large: {dke}");
    assert!(qerr < 1.0e-9, "quaternion norm drift too large: {qerr}");
    println!("SUCCESS: rigid-body invariants conserved");
}
