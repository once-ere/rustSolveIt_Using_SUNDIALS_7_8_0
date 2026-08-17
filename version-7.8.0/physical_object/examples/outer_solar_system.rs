/* Outer solar system N-body problem through the physical_object API.
 *
 * Same problem, constants and success criterion as the donor example
 * `sundials_rs/crates/cvode_rs/examples/solar_system.rs` (Sun + inner
 * planets lumped, Jupiter, Saturn, Uranus, Neptune, Pluto; AU / day /
 * solar-mass units), integrated here by the physical_object CVODE
 * driver instead of a hand-written one. Success: relative energy drift
 * < 1e-6 over 500 000 days. */
#![forbid(unsafe_code)]
#![deny(warnings)]
#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(non_upper_case_globals)]

use ::physical_object::linalg::Vec3;
use ::physical_object::physical_object::physical_object;
use ::physical_object::integrate::run;
use ::physical_object::{Method, PhysicalObjectSystem};
use sundials_core::sundials_utils::{fmt_e, fmt_ew, fmt_f, fmt_fw};

const NBODY: usize = 6;
const G: f64 = 2.95912208286e-4;

const MASS: [f64; NBODY] = [
    1.00000597682,    /* Sun + inner planets */
    9.54786104043e-4, /* Jupiter */
    2.85583733151e-4, /* Saturn  */
    4.37273164546e-5, /* Uranus  */
    5.17759138449e-5, /* Neptune */
    1.0 / 1.3e8,      /* Pluto   */
];

const Q0: [[f64; 3]; NBODY] = [
    [0.0, 0.0, 0.0],
    [-3.5023653, -3.8169847, -1.5507963],
    [9.0755314, -3.0458353, -1.6483708],
    [8.3101420, -16.2901086, -7.2521278],
    [11.4707666, -25.7294829, -10.8169456],
    [-15.5387357, -25.2225594, -3.1902382],
];
const V0: [[f64; 3]; NBODY] = [
    [0.0, 0.0, 0.0],
    [0.00565429, -0.00412490, -0.00190589],
    [0.00168318, 0.00483525, 0.00192462],
    [0.00354178, 0.00137102, 0.00055029],
    [0.00288930, 0.00114527, 0.00039677],
    [0.00276725, -0.00170702, -0.00136504],
];

fn main() {
    let objects: Vec<physical_object> = (0..NBODY)
        .map(|i| {
            physical_object::new_point(
                i,
                MASS[i],
                Vec3::from_array(Q0[i]),
                Vec3::from_array(V0[i]),
            )
        })
        .collect();

    let mut system = PhysicalObjectSystem::new(objects, G);
    system.softening = 0.0; /* the donor problem is unsoftened */
    system.method = Method::Adams;
    system.rtol = 1.0e-10;
    system.atol = 1.0e-12;

    let e0 = system.total_energy();

    println!("\nOuter solar system N-body problem (6 bodies, 78 equations incl. rigid DOF)");
    println!("physical_object -> sundials_rs CVODE (Adams + Newton + dense DQ Jacobian)\n");
    println!(
        "{:>10}  {:>14} {:>14} {:>14}  {:>16}",
        "t (days)", "com x (AU)", "com y (AU)", "com z (AU)", "energy drift"
    );

    let report = run(&mut system, 500_000.0, 10)
        .expect("integration failed");

    for snap in &report.snapshots {
        let drift = ((snap.energy - e0) / e0).abs();
        println!(
            "{:>10}  {} {} {}  {:>16}",
            fmt_fw(snap.t, 10, 0),
            fmt_fw(snap.center_of_mass.x, 14, 8),
            fmt_fw(snap.center_of_mass.y, 14, 8),
            fmt_fw(snap.center_of_mass.z, 14, 8),
            fmt_ew(drift, 16, 6)
        );
    }

    let pluto = system.objects[5].get_position();
    println!("\nPluto at t = {} days:", fmt_f(system.time, 0));
    println!(
        "  x = {}  y = {}  z = {}",
        fmt_fw(pluto.x, 14, 8),
        fmt_fw(pluto.y, 14, 8),
        fmt_fw(pluto.z, 14, 8)
    );

    println!("\nFinal Statistics:");
    println!("  internal steps     = {}", report.nst);
    println!("  rhs evaluations    = {}", report.nfe);
    println!("  nonlinear iters    = {}", report.nni);
    println!("  error test fails   = {}", report.netf);

    let e = system.total_energy();
    let drift = ((e - e0) / e0).abs();
    println!("  relative energy drift over 500000 days = {}", fmt_e(drift, 6));
    if drift < 1.0e-6 {
        println!("  SUCCESS: energy drift within 1e-6");
    } else {
        println!("  FAILURE: energy drift exceeds 1e-6");
        std::process::exit(1);
    }
}
