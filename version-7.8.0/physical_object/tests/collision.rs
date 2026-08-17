//! Collision-detection integration tests: every expectation is a
//! closed-form result of elementary mechanics, and every trajectory is
//! integrated by the real sundials CVODE path with event rootfinding —
//! no mocked physics anywhere.
#![forbid(unsafe_code)]
#![deny(warnings)]
#![allow(non_snake_case)]

use ::physical_object::boundary::Boundary;
use ::physical_object::integrate;
use ::physical_object::linalg::{Mat3, Vec3};
use ::physical_object::physical_object::physical_object;
use ::physical_object::PhysicalObjectSystem;

fn sphere(id: usize, mass: f64, r: f64, pos: Vec3, vel: Vec3) -> physical_object {
    physical_object::new_from_shape(
        id,
        mass,
        0.0,
        pos,
        vel,
        Vec3::zeros(),
        Boundary::Sphere { radius: r },
    )
}

fn cuboid(id: usize, mass: f64, h: [f64; 3], pos: Vec3, vel: Vec3) -> physical_object {
    physical_object::new_from_shape(
        id,
        mass,
        0.0,
        pos,
        vel,
        Vec3::zeros(),
        Boundary::Cuboid { half_extents: h },
    )
}

fn free_system(objects: Vec<physical_object>) -> PhysicalObjectSystem {
    // G = 0: pure free flight + collisions, so every outcome is exact.
    PhysicalObjectSystem::new(objects, 0.0)
}

/// T8 — equal masses, head on, e = 1: velocities exchange exactly; the
/// event time equals the analytic time of impact; momentum and energy
/// are conserved through the impulse.
#[test]
fn equal_mass_head_on_exchange_at_analytic_toi() {
    let mut sys = free_system(vec![
        sphere(0, 1.0, 0.5, Vec3::new(-2.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)),
        sphere(1, 1.0, 0.5, Vec3::new(2.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0)),
    ]);
    let e0 = sys.total_energy();
    let report = integrate::run(&mut sys, 3.0, 1).expect("run");

    // Gap 4 − 2r = 3, approach speed 2 → TOI = 1.5.
    assert_eq!(report.ncollisions, 1, "exactly one impulse");
    let c = &sys.contacts[0];
    assert!((c.t - 1.5).abs() < 1e-9, "TOI = {} (analytic 1.5)", c.t);
    assert_eq!((c.i, c.j), (0, 1));
    assert!((c.normal - Vec3::new(1.0, 0.0, 0.0)).norm() < 1e-9, "normal i→j = +x");
    assert!((c.rel_vel_n + 2.0).abs() < 1e-9, "approach speed 2");
    assert!((c.normal.norm() - 1.0).abs() < 1e-12, "unit normal");

    // Velocities exchanged.
    let v0 = sys.objects[0].get_velocity();
    let v1 = sys.objects[1].get_velocity();
    assert!((v0 - Vec3::new(-1.0, 0.0, 0.0)).norm() < 1e-9, "v0 = {v0:?}");
    assert!((v1 - Vec3::new(1.0, 0.0, 0.0)).norm() < 1e-9, "v1 = {v1:?}");
    // Conservation through the impulse.
    assert!(sys.total_momentum().norm() < 1e-12);
    assert!((sys.total_energy() - e0).abs() < 1e-9);
    assert!(report.nge > 0, "root function was evaluated");
}

/// T9 — unequal masses (1 vs 3), target at rest, e = 1: the 1-D elastic
/// formulas give v1' = −1/2, v2' = +1/2.
#[test]
fn unequal_mass_1d_elastic_formulas() {
    let mut sys = free_system(vec![
        sphere(0, 1.0, 0.5, Vec3::new(-3.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)),
        sphere(1, 3.0, 0.5, Vec3::zeros(), Vec3::zeros()),
    ]);
    integrate::run(&mut sys, 4.0, 1).expect("run");
    let v0 = sys.objects[0].get_velocity().x;
    let v1 = sys.objects[1].get_velocity().x;
    assert!((v0 + 0.5).abs() < 1e-9, "v0' = (m1−m2)/(m1+m2) = −0.5, got {v0}");
    assert!((v1 - 0.5).abs() < 1e-9, "v1' = 2m1/(m1+m2) = +0.5, got {v1}");
}

/// T10 — restitution: separation speed = e × approach speed (e is the
/// pair min); e = 0 gives a common velocity with the exact plastic
/// kinetic-energy loss ½ μ v_rel².
#[test]
fn restitution_ratio_and_plastic_loss() {
    // e = min(0.5, 1.0) = 0.5.
    let mut sys = free_system(vec![
        sphere(0, 1.0, 0.5, Vec3::new(-2.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)),
        sphere(1, 1.0, 0.5, Vec3::new(2.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0)),
    ]);
    sys.objects[0].set_restitution(0.5);
    integrate::run(&mut sys, 3.0, 1).expect("run");
    let sep_speed = sys.objects[1].get_velocity().x - sys.objects[0].get_velocity().x;
    assert!((sep_speed - 1.0).abs() < 1e-9, "e·(approach 2) = 1, got {sep_speed}");

    // Perfectly plastic pair: common velocity, ΔKE = ½ μ v_rel².
    let mut sys2 = free_system(vec![
        sphere(0, 2.0, 0.5, Vec3::new(-2.0, 0.0, 0.0), Vec3::new(3.0, 0.0, 0.0)),
        sphere(1, 1.0, 0.5, Vec3::new(2.0, 0.0, 0.0), Vec3::zeros()),
    ]);
    sys2.objects[0].set_restitution(0.0);
    let ke_before = sys2.total_energy();
    integrate::run(&mut sys2, 3.0, 1).expect("run");
    let v0 = sys2.objects[0].get_velocity().x;
    let v1 = sys2.objects[1].get_velocity().x;
    let v_common = 2.0 * 3.0 / 3.0; // total p / total m = 2
    assert!((v0 - v_common).abs() < 1e-9 && (v1 - v_common).abs() < 1e-9);
    let mu = 2.0 * 1.0 / 3.0;
    let expected_loss = 0.5 * mu * 9.0;
    assert!(((ke_before - sys2.total_energy()) - expected_loss).abs() < 1e-9);
}

/// T11 — oblique impact on a static sphere: the recorded normal is the
/// line of centers at the interpolated TOI, and the reflection follows
/// v' = v − 2(v·n)n exactly.
#[test]
fn oblique_impact_normal_is_line_of_centers() {
    let mut target = sphere(1, 1.0, 0.5, Vec3::zeros(), Vec3::zeros());
    target.set_inverse_mass(0.0);
    target.set_inverse_inertia_tensor(Mat3::zeros());
    let mut sys = free_system(vec![
        sphere(0, 1.0, 0.5, Vec3::new(-3.0, 0.6, 0.0), Vec3::new(1.0, 0.0, 0.0)),
        target,
    ]);
    integrate::run(&mut sys, 5.0, 1).expect("run");

    // Contact when |Δ| = 1 with Δy = 0.6 → moving center at x* = −0.8;
    // normal (mover → target) = (0.8, −0.6, 0).
    let c = &sys.contacts[0];
    let expect_n = Vec3::new(0.8, -0.6, 0.0);
    assert!((c.normal - expect_n).norm() < 1e-6, "normal = {:?}", c.normal);
    // e = 1 static reflection: v' = v − 2(v·n)n = (−0.28, 0.96, 0).
    let v = sys.objects[0].get_velocity();
    assert!((v - Vec3::new(-0.28, 0.96, 0.0)).norm() < 1e-6, "v' = {v:?}");
    // Speed preserved (elastic off a static wall).
    assert!((v.norm() - 1.0).abs() < 1e-9);
}

/// T12 — off-center hit on a free cuboid: the angular impulse is
/// exactly r × (j_n n), and both total momentum and total angular
/// momentum about the origin are conserved by the action–reaction pair.
#[test]
fn off_center_cuboid_hit_spin_up_conserves_totals() {
    let ball = sphere(0, 1.0, 0.25, Vec3::new(-2.0, 0.5, 0.0), Vec3::new(4.0, 0.0, 0.0));
    let boxy = cuboid(1, 2.0, [0.5, 1.0, 1.0], Vec3::zeros(), Vec3::zeros());
    let mut sys = free_system(vec![ball, boxy]);
    let p_before = sys.total_momentum();
    let l_before = sys.total_angular_momentum(Vec3::zeros());
    integrate::run(&mut sys, 1.0, 1).expect("run");

    assert!(!sys.contacts.is_empty(), "the ball must strike the box");
    let c = &sys.contacts[0];
    // Face contact on the −x face of the box, at the ball's height.
    assert!((c.normal - Vec3::new(1.0, 0.0, 0.0)).norm() < 1e-6, "n = {:?}", c.normal);
    assert!((c.point.y - 0.5).abs() < 1e-6, "hit at y = 0.5");

    // ΔL of the box = r × (j_n n) with r from its center to the point.
    let r = c.point - Vec3::zeros();
    let expected_dl = r.cross(c.normal * c.impulse_n);
    let dl = sys.objects[1].get_angular_momentum();
    assert!((dl - expected_dl).norm() < 1e-9, "ΔL = {dl:?} vs {expected_dl:?}");
    assert!(dl.z < 0.0, "hit above center with +x impulse → clockwise about z");

    // Totals conserved by the action–reaction pair.
    assert!((sys.total_momentum() - p_before).norm() < 1e-9);
    assert!((sys.total_angular_momentum(Vec3::zeros()) - l_before).norm() < 1e-9);
}

/// T13 — Newton's cradle: three touching spheres, one incomer; the
/// impulse propagates through the chain (Gauss–Seidel passes at the
/// single root event) and only the far sphere exits.
#[test]
fn newtons_cradle_three_spheres() {
    let mut sys = free_system(vec![
        sphere(0, 1.0, 0.5, Vec3::new(-3.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)),
        sphere(1, 1.0, 0.5, Vec3::new(0.0, 0.0, 0.0), Vec3::zeros()),
        sphere(2, 1.0, 0.5, Vec3::new(1.0, 0.0, 0.0), Vec3::zeros()),
    ]);
    integrate::run(&mut sys, 4.0, 1).expect("run");
    let v: Vec<f64> = sys.objects.iter().map(|o| o.get_velocity().x).collect();
    assert!(v[0].abs() < 1e-6, "incomer stops, v0 = {}", v[0]);
    assert!(v[1].abs() < 1e-6, "middle stays, v1 = {}", v[1]);
    assert!((v[2] - 1.0).abs() < 1e-6, "far sphere exits at v, v2 = {}", v[2]);
    assert!(sys.total_momentum().norm() > 0.0); // sanity: momentum moved through
}

/// T14 — bouncing ball on a static slab under uniform gravity with
/// e = 0.8: the rebound apex is e² of the drop height (energy argument),
/// read from the center-of-mass track (the static slab has zero mass).
#[test]
fn bouncing_ball_apex_ratio_is_e_squared() {
    let mut floor = cuboid(0, 1.0, [5.0, 0.5, 5.0], Vec3::new(0.0, -0.5, 0.0), Vec3::zeros());
    floor.set_inverse_mass(0.0);
    floor.set_inverse_inertia_tensor(Mat3::zeros());
    let mut ball = sphere(1, 1.0, 0.5, Vec3::new(0.0, 5.0, 0.0), Vec3::zeros());
    ball.set_restitution(0.8);
    let mut sys = free_system(vec![floor, ball]);
    sys.uniform_gravity = Vec3::new(0.0, -10.0, 0.0);

    // Drop height of the center above its resting height (0.5): 4.5.
    // Impact at t = √(2·4.5/10) ≈ 0.9487; rebound apex 0.64·4.5 = 2.88
    // above rest → apex center height 3.38, reached ≈ 0.759 s later.
    let report = integrate::run(&mut sys, 1.8, 360).expect("run");
    assert!(report.ncollisions >= 1);
    let c = &sys.contacts[0];
    let t_imp = (2.0f64 * 4.5 / 10.0).sqrt();
    assert!((c.t - t_imp).abs() < 1e-7, "impact t = {} (analytic {t_imp})", c.t);
    assert!((c.normal - Vec3::new(0.0, 1.0, 0.0)).norm() < 1e-9, "floor → ball = +y");

    // Apex AFTER the bounce (the pre-impact snapshots include the
    // higher initial drop position).
    let apex = report
        .snapshots
        .iter()
        .filter(|s| s.t > c.t)
        .map(|s| s.center_of_mass.y)
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        (apex - 3.38).abs() < 2e-3,
        "apex {apex} vs 0.5 + e²·4.5 = 3.38 (sampled at 5 ms)"
    );
}

/// T15 — the rootfinding differentiator: a fast, small sphere meets a
/// thin static plate inside one large output step and must NOT tunnel;
/// the event lands on the analytic TOI to nanosecond-scale precision.
#[test]
fn thin_wall_no_tunneling_at_large_output_step() {
    let mut wall = cuboid(0, 1.0, [0.005, 5.0, 5.0], Vec3::zeros(), Vec3::zeros());
    wall.set_inverse_mass(0.0);
    wall.set_inverse_inertia_tensor(Mat3::zeros());
    let bullet = sphere(1, 1.0, 0.01, Vec3::new(-2.0, 0.0, 0.0), Vec3::new(100.0, 0.0, 0.0));
    let mut sys = free_system(vec![wall, bullet]);

    let report = integrate::run(&mut sys, 0.05, 1).expect("run");
    assert_eq!(report.ncollisions, 1);
    let c = &sys.contacts[0];
    let t_analytic = (2.0 - 0.005 - 0.01) / 100.0;
    assert!((c.t - t_analytic).abs() < 1e-9, "TOI = {} (analytic {t_analytic})", c.t);
    let v = sys.objects[1].get_velocity();
    assert!((v - Vec3::new(-100.0, 0.0, 0.0)).norm() < 1e-9, "reflected exactly, v = {v:?}");
    assert!(
        sys.objects[1].get_position().x < -0.015,
        "bullet is back on its own side (no tunneling)"
    );
}

/// T19 — the symplectic SPRK path detects the same event: two
/// non-spinning spheres exchange velocities under LEAPFROG_2_2, and
/// the spinning-body separability gate still rejects with its
/// established message.
#[test]
fn sprk_path_detects_collisions_and_keeps_its_gate() {
    let mut sys = free_system(vec![
        sphere(0, 1.0, 0.5, Vec3::new(-2.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0)),
        sphere(1, 1.0, 0.5, Vec3::new(2.0, 0.0, 0.0), Vec3::new(-1.0, 0.0, 0.0)),
    ]);
    sys.method = ::physical_object::Method::Sprk {
        table: "ARKODE_SPRK_LEAPFROG_2_2".to_string(),
        dt: 1e-3,
    };
    let report = integrate::run(&mut sys, 3.0, 1).expect("sprk run");
    assert!(report.ncollisions >= 1, "SPRK path resolved the impact");
    let v0 = sys.objects[0].get_velocity().x;
    let v1 = sys.objects[1].get_velocity().x;
    assert!((v0 + 1.0).abs() < 1e-6, "exchange under SPRK, v0 = {v0}");
    assert!((v1 - 1.0).abs() < 1e-6, "exchange under SPRK, v1 = {v1}");

    // The separability gate is untouched: a spinning rigid body is
    // still rejected with the established message.
    let mut spinner = free_system(vec![
        physical_object::new_from_shape(
            0,
            1.0,
            0.0,
            Vec3::zeros(),
            Vec3::zeros(),
            Vec3::new(0.0, 0.0, 2.0),
            Boundary::Sphere { radius: 0.5 },
        ),
        sphere(1, 1.0, 0.5, Vec3::new(3.0, 0.0, 0.0), Vec3::zeros()),
    ]);
    spinner.method = ::physical_object::Method::Sprk {
        table: "ARKODE_SPRK_LEAPFROG_2_2".to_string(),
        dt: 1e-3,
    };
    let err = integrate::run(&mut spinner, 1.0, 1).expect_err("gate must reject");
    assert!(err.contains("translational dynamics only"), "gate message: {err}");
}

/// T17 — structural invariance: a system with zero collidable pairs
/// (two point masses in mutual orbit) takes the identical CVODE code
/// path whether collisions are enabled or not — bit-identical states,
/// identical step counts, zero root evaluations.
#[test]
fn zero_pair_systems_are_bit_identical_with_collide_on() {
    let build = || {
        let mut s = PhysicalObjectSystem::new(
            vec![
                physical_object::new_point(0, 3.0, Vec3::new(-1.0, 0.0, 0.0), Vec3::new(0.0, -0.5, 0.0)),
                physical_object::new_point(1, 3.0, Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 0.5, 0.0)),
            ],
            1.0,
        );
        s.softening = 1e-6;
        s
    };
    let mut on = build();
    on.collide_enabled = true;
    let rep_on = integrate::run(&mut on, 2.0, 4).expect("run on");

    let mut off = build();
    off.collide_enabled = false;
    let rep_off = integrate::run(&mut off, 2.0, 4).expect("run off");

    assert_eq!(on.pack_state(), off.pack_state(), "bit-identical trajectories");
    assert_eq!(rep_on.nst, rep_off.nst, "identical internal step counts");
    assert_eq!(rep_on.nge, 0, "rootfinding never armed without collidable pairs");
    assert_eq!(rep_on.ncollisions, 0);
}

/// T18 — armed but quiet: two spheres that never meet during the run.
/// Rootfinding is armed (nge > 0) yet no event fires; the trajectory
/// agrees with the unarmed run to solver tolerance (the anti-tunneling
/// step cap may change the internal step sequence, not the physics).
#[test]
fn armed_but_quiet_matches_unarmed_within_tolerance() {
    let build = || {
        free_system(vec![
            sphere(0, 1.0, 0.1, Vec3::new(-5.0, 0.0, 0.0), Vec3::new(0.3, 0.0, 0.0)),
            sphere(1, 1.0, 0.1, Vec3::new(5.0, 0.0, 0.0), Vec3::new(0.3, 0.0, 0.0)),
        ])
    };
    let mut on = build();
    on.collide_enabled = true;
    let rep_on = integrate::run(&mut on, 3.0, 1).expect("run on");
    let mut off = build();
    off.collide_enabled = false;
    integrate::run(&mut off, 3.0, 1).expect("run off");

    assert!(rep_on.nge > 0, "roots were armed and evaluated");
    assert_eq!(rep_on.ncollisions, 0, "no event fired");
    let a = on.pack_state();
    let b = off.pack_state();
    for (x, y) in a.iter().zip(b.iter()) {
        assert!((x - y).abs() < 1e-12, "trajectories agree to tolerance: {x} vs {y}");
    }
}

// --------------------------------------------------------------------
// The rigid, "infinitely massive" bounding box (inverse mass = 0) and
// the extended shape set (torus / disk / cylinder).
// --------------------------------------------------------------------

/// One static wall slab of a bounding box: infinitely massive in the
/// only representation the equations of motion ever use — inverse mass
/// (and inverse inertia) identically zero.
fn static_slab(id: usize, h: [f64; 3], pos: Vec3) -> physical_object {
    let mut w = cuboid(id, 1.0, h, pos, Vec3::zeros());
    w.set_inverse_mass(0.0);
    w.set_inverse_inertia_tensor(Mat3::zeros());
    w
}

/// The six slabs of an axis-aligned cubic box with inner half-extent
/// `half` centered at the origin (slab thickness `half/2`, slabs
/// oversized so their inner faces tile the whole box surface).
fn bounding_box_slabs(first_id: usize, half: f64) -> Vec<physical_object> {
    let t = 0.5 * half; // slab half-thickness
    let big = half + 2.0 * t; // cross-section half-extent, covers corners
    let c = half + t; // slab center distance from the origin
    vec![
        static_slab(first_id, [t, big, big], Vec3::new(c, 0.0, 0.0)),
        static_slab(first_id + 1, [t, big, big], Vec3::new(-c, 0.0, 0.0)),
        static_slab(first_id + 2, [big, t, big], Vec3::new(0.0, c, 0.0)),
        static_slab(first_id + 3, [big, t, big], Vec3::new(0.0, -c, 0.0)),
        static_slab(first_id + 4, [big, big, t], Vec3::new(0.0, 0.0, c)),
        static_slab(first_id + 5, [big, big, t], Vec3::new(0.0, 0.0, -c)),
    ]
}

/// T19 — a ball rattling inside a rigid 4×4×4 box of six inverse-mass-0
/// slabs: every wall bounce is a pure elastic reflection, so speed and
/// kinetic energy are conserved exactly, the ball never escapes, and
/// the walls remain bit-identically at rest (no impulse is ever written
/// to a static side).
#[test]
fn ball_in_a_rigid_box_conserves_energy_and_walls_never_move() {
    let mut objects = bounding_box_slabs(0, 2.0);
    objects.push(sphere(6, 1.0, 0.5, Vec3::zeros(), Vec3::new(3.0, 1.7, 0.9)));
    let mut sys = free_system(objects);
    let e0 = sys.total_energy();
    let speed0 = sys.objects[6].get_velocity().norm();

    let report = integrate::run(&mut sys, 6.0, 60).expect("run");

    assert!(report.ncollisions >= 5, "several wall bounces, got {}", report.ncollisions);
    assert!((sys.total_energy() - e0).abs() < 1e-9 * e0.abs().max(1.0), "elastic box conserves E");
    assert!((sys.objects[6].get_velocity().norm() - speed0).abs() < 1e-9, "|v| preserved");
    let p = sys.objects[6].get_position();
    assert!(p.x.abs() <= 1.5 + 1e-6 && p.y.abs() <= 1.5 + 1e-6 && p.z.abs() <= 1.5 + 1e-6,
        "ball stays inside the box: {p:?}");
    for k in 0..6 {
        assert_eq!(sys.objects[k].get_momentum(), Vec3::zeros(), "wall {k} momentum");
        assert_eq!(sys.objects[k].get_angular_momentum(), Vec3::zeros(), "wall {k} spin");
        assert_eq!(sys.objects[k].get_inverse_mass(), 0.0, "wall {k} stays static");
    }
    // The walls' positions are bit-identical to construction.
    let fresh = bounding_box_slabs(0, 2.0);
    for (k, w) in fresh.iter().enumerate().take(6) {
        assert_eq!(sys.objects[k].get_position(), w.get_position(), "wall {k} position");
    }
}

/// T20 — the torus hole is real: a point particle aimed through the
/// center of a free torus (ring 1.5, tube 0.5) passes through without
/// any event (its separation function never reaches zero), while a
/// fat ball on a nearby line hits the tube at the analytic time of
/// impact and the free torus recoils, conserving momentum and energy.
#[test]
fn point_threads_the_torus_hole_but_a_fat_ball_bounces() {
    let torus = || {
        physical_object::new_from_shape(
            0,
            1.0,
            0.0,
            Vec3::zeros(),
            Vec3::zeros(),
            Vec3::zeros(),
            Boundary::Torus { ring_radius: 1.5, tube_radius: 0.5 },
        )
    };

    // A point particle through the hole: no contact, torus unmoved.
    let mut sys = free_system(vec![
        torus(),
        physical_object::new_point(1, 1.0, Vec3::new(0.0, 0.0, -3.0), Vec3::new(0.0, 0.0, 4.0)),
    ]);
    let report = integrate::run(&mut sys, 1.5, 15).expect("run");
    assert_eq!(report.ncollisions, 0, "the hole is passable");
    assert!(sys.objects[1].get_position().z > 2.9, "point came out the other side");
    assert_eq!(sys.objects[0].get_momentum(), Vec3::zeros(), "torus untouched");

    // A fat ball offset from the axis: sep(z) = √(1.44 + z²) − 1.7 = 0
    // at z = −√1.45 (approaching), so TOI = (3 − √1.45)/4.
    let mut sys2 = free_system(vec![
        torus(),
        sphere(1, 1.0, 1.2, Vec3::new(0.3, 0.0, -3.0), Vec3::new(0.0, 0.0, 4.0)),
    ]);
    let e0 = sys2.total_energy();
    let p0 = sys2.total_momentum();
    let report2 = integrate::run(&mut sys2, 1.5, 15).expect("run");
    assert!(report2.ncollisions >= 1, "the tube is not passable for a fat ball");
    let toi = (3.0 - 1.45f64.sqrt()) / 4.0;
    let c = &sys2.contacts[0];
    assert!((c.t - toi).abs() < 1e-6, "TOI {} vs analytic {toi}", c.t);
    assert!((sys2.total_momentum() - p0).norm() < 1e-9, "momentum conserved (torus recoils)");
    assert!((sys2.total_energy() - e0).abs() < 1e-9 * e0.max(1.0), "elastic: energy conserved");
    assert!(sys2.objects[0].get_momentum().norm() > 1e-6, "free torus actually recoiled");
}

/// T21 — a miniature of the manager's demo: cylinder + disk + cube
/// flying inside the rigid box, all restitution 1. Every impulse is
/// elastic (contact-point normal velocity reversal conserves kinetic
/// energy exactly, whatever the contact normal), so total energy is
/// conserved through many mixed shape-shape and shape-wall events and
/// every body stays inside the box.
#[test]
fn mixed_shapes_rattle_in_the_box_conserving_energy() {
    let mut objects = bounding_box_slabs(0, 2.0);
    objects.push(physical_object::new_from_shape(
        6,
        2.0,
        0.0,
        Vec3::new(-1.0, 0.3, 0.0),
        Vec3::new(2.0, 0.5, -1.0),
        Vec3::zeros(),
        Boundary::Cylinder { radius: 0.25, half_height: 0.75 },
    ));
    objects.push(physical_object::new_from_shape(
        7,
        2.0 / 3.0,
        0.0,
        Vec3::new(1.0, -0.8, 0.5),
        Vec3::new(-1.0, 1.0, 0.6),
        Vec3::zeros(),
        Boundary::Disk { radius: 1.0 },
    ));
    objects.push(cuboid(
        8,
        5.0 / 3.0,
        [0.5, 0.5, 0.5],
        Vec3::new(0.2, 1.1, -0.9),
        Vec3::new(0.5, -2.0, 1.0),
    ));
    let mut sys = free_system(objects);
    let e0 = sys.total_energy();

    let report = integrate::run(&mut sys, 3.0, 300).expect("run");

    assert!(report.ncollisions >= 3, "events happened: {}", report.ncollisions);
    let drift = (sys.total_energy() - e0).abs() / e0;
    assert!(drift < 1e-6, "energy drift {drift} through {} events", report.ncollisions);
    for k in 6..9 {
        let p = sys.objects[k].get_position();
        assert!(
            p.x.abs() < 2.0 && p.y.abs() < 2.0 && p.z.abs() < 2.0,
            "obj{k} inside the box: {p:?}"
        );
    }
    for k in 0..6 {
        assert_eq!(sys.objects[k].get_momentum(), Vec3::zeros(), "wall {k} at rest");
    }
}

/// T22 — the anti-tunneling cap is acceleration-aware: a small ball
/// RELEASED FROM REST above a thin static plate under uniform gravity
/// must still be caught (the instantaneous relative speed is zero at
/// arm time; only the reachable speed v + a·Δt gives a finite cap).
#[test]
fn ball_released_from_rest_does_not_tunnel_the_thin_plate() {
    let mut plate = cuboid(0, 1.0, [4.0, 0.005, 4.0], Vec3::zeros(), Vec3::zeros());
    plate.set_inverse_mass(0.0);
    plate.set_inverse_inertia_tensor(Mat3::zeros());
    let mut sys = free_system(vec![
        plate,
        sphere(1, 1.0, 0.01, Vec3::new(0.0, 5.0, 0.0), Vec3::zeros()),
    ]);
    sys.uniform_gravity = Vec3::new(0.0, -10.0, 0.0);

    let report = integrate::run(&mut sys, 2.0, 1).expect("run");
    assert!(report.ncollisions >= 1, "the plate was hit, not tunneled");
    let y = sys.objects[1].get_position().y;
    assert!(y > 0.0, "elastic bounce keeps the ball above the plate: y = {y}");
    let c = &sys.contacts[0];
    // Analytic TOI: falls 5 − 0.005 − 0.01 = 4.985 → t = √(2·4.985/10).
    let toi = (2.0f64 * 4.985 / 10.0).sqrt();
    assert!((c.t - toi).abs() < 1e-6, "TOI {} vs analytic {toi}", c.t);
}

/// T23 — two rigid dumbbells collide elastically in free space: total
/// energy, total linear momentum AND total angular momentum (about the
/// origin) are conserved. L is conserved exactly because the impulse
/// pair acts at one shared contact point (action–reaction along the
/// same line ⇒ zero net torque about any point).
#[test]
fn colliding_dumbbells_conserve_energy_momentum_and_angular_momentum() {
    let (mass, db) = ::physical_object::boundary::dumbbell(1.0, 2.0, 0.5, 0.25, 0.25, 0.1, 1.0)
        .expect("valid dumbbell");
    let mk = |id: usize, pos: Vec3, vel: Vec3, w: Vec3| {
        ::physical_object::physical_object::physical_object::new_from_shape(
            id, mass, 0.0, pos, vel, w, db,
        )
    };
    // Slightly offset approach so the impact is off-center: the hit
    // exchanges linear AND angular momentum between the bodies.
    let mut sys = free_system(vec![
        mk(0, Vec3::new(-2.0, 0.15, 0.0), Vec3::new(1.5, 0.0, 0.0), Vec3::new(0.0, 0.0, 0.6)),
        mk(1, Vec3::new(2.0, -0.15, 0.0), Vec3::new(-1.5, 0.0, 0.0), Vec3::new(0.4, 0.0, 0.0)),
    ]);
    let e0 = sys.total_energy();
    let p0 = sys.total_momentum();
    let l0 = sys.total_angular_momentum(Vec3::zeros());

    let report = integrate::run(&mut sys, 3.0, 60).expect("run");

    assert!(report.ncollisions >= 1, "the dumbbells actually met: {}", report.ncollisions);
    let e1 = sys.total_energy();
    let p1 = sys.total_momentum();
    let l1 = sys.total_angular_momentum(Vec3::zeros());
    assert!(((e1 - e0) / e0).abs() < 1e-8, "energy: {e0} -> {e1}");
    assert!((p1 - p0).norm() < 1e-9, "linear momentum: {p0:?} -> {p1:?}");
    assert!((l1 - l0).norm() < 1e-8, "angular momentum: {l0:?} -> {l1:?}");
    // The collision genuinely redistributed spin.
    assert!(sys.objects[0].get_angular_momentum().norm() > 1e-6);
}

/// **The output-granularity defect**, reproduced as a test.
///
/// `PROJECT_STATUS.md` recorded it as a real numerical defect, left
/// unrepaired: the trajectory depends on how often output is
/// requested. `box_of_shapes_m32.md` §5 measured `|dE/E|` of 6.9e-8 at
/// output interval 0.001 against 2.3e-1 at 0.125 — 23 % of the energy
/// gone, with no warning and a smooth-looking animation.
///
/// The cause is the Zeno guard counting events **per output interval**:
/// past 64 it forces restitution to zero, so ordinary elastic
/// collisions become plastic purely because the caller asked for fewer
/// snapshots. Requesting output is supposed to be an observation, and
/// an observation must not change the physics.
///
/// A fast light sphere between two static walls collides about 100
/// times per unit time here, so one coarse interval crosses the
/// threshold and a fine one does not.
#[test]
fn energy_does_not_depend_on_how_often_output_is_requested() {
    let build = || {
        let mut left = cuboid(1, 1.0, [0.1, 2.0, 2.0], Vec3::new(-1.0, 0.0, 0.0), Vec3::zeros());
        left.set_inverse_mass(0.0);
        left.set_inverse_inertia_tensor(Mat3::zeros());
        let mut right = cuboid(2, 1.0, [0.1, 2.0, 2.0], Vec3::new(1.0, 0.0, 0.0), Vec3::zeros());
        right.set_inverse_mass(0.0);
        right.set_inverse_inertia_tensor(Mat3::zeros());
        free_system(vec![
            sphere(0, 1.0, 0.05, Vec3::zeros(), Vec3::new(400.0, 0.0, 0.0)),
            left,
            right,
        ])
    };
    let energy = |s: &PhysicalObjectSystem| -> f64 {
        let v = s.objects[0].get_velocity().norm();
        0.5 * v * v
    };

    let mut fine = build();
    let e0 = energy(&fine);
    let fine_report = integrate::run(&mut fine, 1.0, 400).expect("fine run");
    let e_fine = energy(&fine);

    let mut coarse = build();
    let coarse_report = integrate::run(&mut coarse, 1.0, 2).expect("coarse run");
    let e_coarse = energy(&coarse);

    // Both must conserve energy: these are elastic collisions off
    // static walls, so the speed is an exact invariant.
    assert!(
        (e_fine - e0).abs() < 1e-6 * e0,
        "fine output: energy {e0} -> {e_fine} over {} collisions",
        fine_report.ncollisions
    );
    assert!(
        (e_coarse - e0).abs() < 1e-6 * e0,
        "COARSE output: energy {e0} -> {e_coarse} over {} collisions — asking for fewer \
         snapshots changed the physics",
        coarse_report.ncollisions
    );
    // And they must agree with each other.
    assert!(
        (e_fine - e_coarse).abs() < 1e-6 * e0,
        "fine {e_fine} vs coarse {e_coarse}: the trajectory depends on the output interval"
    );
}

/// The Zeno guard must still fire — a burst-based rule that never
/// escalates would be a worse defect than the interval-based one it
/// replaced, because a settling ball would never terminate.
///
/// A ball bouncing with `e = 0.5` under gravity has bounce intervals
/// falling geometrically, so infinitely many impacts occur before the
/// settling time. This runs well past that point: it must **finish**,
/// come to rest on the floor, and not sink through it.
#[test]
fn a_settling_ball_is_caught_by_the_zeno_guard_and_terminates() {
    let mut floor = cuboid(0, 1.0, [5.0, 0.5, 5.0], Vec3::new(0.0, -0.5, 0.0), Vec3::zeros());
    floor.set_inverse_mass(0.0);
    floor.set_inverse_inertia_tensor(Mat3::zeros());
    let mut ball = sphere(1, 1.0, 0.5, Vec3::new(0.0, 2.0, 0.0), Vec3::zeros());
    ball.set_restitution(0.5);
    let mut sys = free_system(vec![floor, ball]);
    sys.uniform_gravity = Vec3::new(0.0, -10.0, 0.0);

    // Total flight time for e = 0.5 from h = 1.5 is finite:
    // t = sqrt(2h/g) (1 + 2e/(1-e)) = 0.548 * 3 = 1.64. Run to 6.
    let report = integrate::run(&mut sys, 6.0, 60).expect("the run must terminate");
    assert!(report.ncollisions > 5, "it should bounce several times first");

    let y = sys.objects[1].get_position().y;
    let v = sys.objects[1].get_velocity().norm();
    // At rest ON the floor: centre at the sphere radius above y = 0.
    assert!((y - 0.5).abs() < 5e-2, "the ball should settle at y = 0.5, got {y}");
    assert!(v < 1.0, "and it should be nearly at rest, got |v| = {v}");
}

/// `MAX_EVENTS_IN_BURST` must be large enough to let genuinely
/// simultaneous contacts resolve elastically.
///
/// Stage 2F's mutation probe dropped it from 64 to 1 and the suite still
/// passed: `a_settling_ball_is_caught_by_the_zeno_guard_and_terminates`
/// checks that the guard *fires*, and nothing checked that it does not
/// fire too early. Escalating after a single event turns the second
/// contact of any simultaneous pair plastic.
///
/// A ball driven into the corner where two static slabs meet produces
/// two contacts at the same instant — one burst, two events — and an
/// elastic corner reflection must reverse both velocity components and
/// preserve the speed exactly.
#[test]
fn simultaneous_contacts_stay_elastic() {
    let mut wall_x = cuboid(1, 1.0, [0.5, 4.0, 4.0], Vec3::new(-1.5, 0.0, 0.0), Vec3::zeros());
    wall_x.set_inverse_mass(0.0);
    wall_x.set_inverse_inertia_tensor(Mat3::zeros());
    let mut wall_y = cuboid(2, 1.0, [4.0, 0.5, 4.0], Vec3::new(0.0, -1.5, 0.0), Vec3::zeros());
    wall_y.set_inverse_mass(0.0);
    wall_y.set_inverse_inertia_tensor(Mat3::zeros());
    // Symmetric approach: the ball reaches both faces at the same time.
    let ball = sphere(0, 1.0, 0.25, Vec3::new(2.0, 2.0, 0.0), Vec3::new(-3.0, -3.0, 0.0));
    let mut sys = free_system(vec![ball, wall_x, wall_y]);

    let speed0 = sys.objects[0].get_velocity().norm();
    let report = integrate::run(&mut sys, 2.0, 20).expect("run");
    assert!(report.ncollisions >= 2, "the corner should give two contacts");

    let v = sys.objects[0].get_velocity();
    assert!(
        (v.norm() - speed0).abs() < 1e-9 * speed0,
        "an elastic corner reflection must preserve speed: {} -> {}",
        speed0,
        v.norm()
    );
    // Both components reversed: the ball leaves the way it came.
    assert!(v.x > 0.0 && v.y > 0.0, "it should be heading back out, got {v:?}");
}
