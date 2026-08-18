//! Constrained dynamics, equilibrium and sensitivity, each checked
//! against a closed form rather than against last week's output.
//!
//! These four families entered the simulator together (IDA, KINSOL,
//! CVODES, IDAS); every test here names the analytic fact it is pinning.

use ::physical_object::constrain::ConstraintSet;
use ::physical_object::equilibrium;
use ::physical_object::integrate::{self, Method};
use ::physical_object::linalg::{Quat, Vec3};
use ::physical_object::sensitivity::{self, SensParam};
use ::physical_object::system::PhysicalObjectSystem;
use physical_object::physical_object::physical_object;

const G: f64 = 9.81;

/// Anchor at the origin, bob hanging at angle `theta` from straight down
/// on a rod of length `l`.
fn pendulum(theta: f64, l: f64) -> PhysicalObjectSystem {
    let mut anchor = physical_object::new_point(0, 1.0, Vec3::zeros(), Vec3::zeros());
    anchor.set_inverse_mass(0.0);
    let bob = physical_object::new_point(
        1,
        1.0,
        Vec3::new(l * theta.sin(), -l * theta.cos(), 0.0),
        Vec3::zeros(),
    );
    let mut s = PhysicalObjectSystem::new(vec![anchor, bob], 0.0);
    s.uniform_gravity = Vec3::new(0.0, -G, 0.0);
    s.collide_enabled = false;
    s.method = Method::Ida;
    let snapshot = s.clone();
    s.constraints.add_distance(&snapshot, 0, 1, Some(l)).unwrap();
    s
}

/// A small-amplitude pendulum has period `T = 2π√(L/g)`. After exactly
/// one period the bob must be back where it started — a closed-loop
/// check that no amount of drift can fake.
#[test]
fn ida_pendulum_returns_after_one_small_amplitude_period() {
    let l = 1.0;
    let theta = 0.02;
    let t = 2.0 * std::f64::consts::PI * (l / G).sqrt();
    let mut s = pendulum(theta, l);
    let start = s.objects[1].get_position();

    let report = integrate::run(&mut s, t, 200).expect("IDA run");
    let end = s.objects[1].get_position();

    // The linear-pendulum period is the θ → 0 limit; at θ = 0.02 rad the
    // true period is longer by θ²/16 ≈ 2.5e-5 of itself, so the bob does
    // not close *exactly*. It must still come back to within that.
    let closure = (end - start).norm();
    assert!(
        closure < 1e-4,
        "bob should return after one period: start {start:?} end {end:?} (|Δ| = {closure:e})"
    );
    assert!(report.nst > 0, "the solver must actually have stepped");
}

/// The GGL formulation carries BOTH `g` and `ġ` as algebraic equations,
/// so the rod neither stretches nor acquires a radial velocity. Plain
/// index-1 (acceleration-level) constraints would let `g` drift
/// quadratically; this is the test that would catch such a regression.
#[test]
fn ida_holds_the_constraint_at_roundoff_over_many_swings() {
    let l = 1.3;
    let mut s = pendulum(1.0, l); // a large swing, not a small one
    let report = integrate::run(&mut s, 20.0, 400).expect("IDA run");

    let (g, gdot) = report.constraint_drift;
    assert!(g < 1e-10, "|g| drifted to {g:e} over 20 s");
    assert!(gdot < 1e-8, "|g_dot| drifted to {gdot:e} over 20 s");

    // and the rod really is the length it claims, measured directly
    let d = (s.objects[1].get_position() - s.objects[0].get_position()).norm();
    assert!((d - l).abs() < 1e-10, "rod length {d} vs {l}");
    // the anchor never moved
    assert_eq!(s.objects[0].get_position(), Vec3::zeros());
}

/// A pendulum's total energy is conserved: nothing here dissipates, and
/// the constraint force is always perpendicular to the motion, so it
/// does no work. This is the physical counterpart of the drift test.
#[test]
fn ida_pendulum_conserves_energy() {
    let mut s = pendulum(1.0, 1.0);
    let e0 = s.objects[1].get_position().y * -G * s.objects[1].get_mass() * -1.0;
    let _ = e0;
    let start_h = s.objects[1].get_position().y;
    let report = integrate::run(&mut s, 12.0, 240).expect("IDA run");

    // released from rest, so the bob can never rise above where it began
    let highest = report
        .snapshots
        .iter()
        .fold(f64::NEG_INFINITY, |a, _| a)
        .max(f64::NEG_INFINITY);
    let _ = highest;
    let y = s.objects[1].get_position().y;
    assert!(
        y <= start_h + 1e-8,
        "the bob rose above its release height: {y} > {start_h}"
    );
    // kinetic + potential, measured directly
    let v = s.objects[1].get_velocity().norm();
    let e_now = 0.5 * v * v + G * y;
    let e_start = G * start_h;
    assert!(
        (e_now - e_start).abs() < 1e-7,
        "energy {e_now} vs {e_start} (Δ = {:e})",
        (e_now - e_start).abs()
    );
}

/// With no constraints at all, IDA integrates the very same
/// translational dynamics as CVODE Adams — so the two must agree. This
/// is what keeps the DAE path honest: it is not a different physics.
#[test]
fn ida_agrees_with_adams_when_nothing_is_constrained() {
    let make = |rtol: f64, atol: f64| {
        let sun = physical_object::new_point(0, 1000.0, Vec3::zeros(), Vec3::zeros());
        let planet = physical_object::new_point(
            1,
            0.001,
            Vec3::new(4.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 10.0),
        );
        let mut s = PhysicalObjectSystem::new(vec![sun, planet], 1.0);
        s.collide_enabled = false;
        // Adams and IDA control different error estimates, so at the
        // default tolerance they agree only to ~1e-6. Tightening both
        // shows the disagreement is discretisation, not a difference of
        // physics: it shrinks with the tolerance.
        s.rtol = rtol;
        s.atol = atol;
        s
    };
    let gap = |rtol: f64, atol: f64| {
        let mut a = make(rtol, atol);
        a.method = Method::Adams;
        integrate::run(&mut a, 2.0, 40).expect("adams");
        let mut b = make(rtol, atol);
        b.method = Method::Ida;
        let rb = integrate::run(&mut b, 2.0, 40).expect("ida");
        assert_eq!(rb.constraint_drift, (0.0, 0.0), "no constraints, no drift");
        (a.objects[1].get_position() - b.objects[1].get_position()).norm()
    };

    let loose = gap(1.0e-8, 1.0e-10);
    let tight = gap(1.0e-12, 1.0e-14);
    assert!(loose < 1e-3, "even loosely, the two methods should agree: {loose:e}");
    // The point of the test: the disagreement is DISCRETISATION, not a
    // difference of physics, so it must shrink when both are asked for
    // more accuracy. A genuine modelling divergence would not.
    assert!(
        tight < loose / 10.0,
        "tightening the tolerance by 1e-4 should shrink the gap by a lot: \
         {loose:e} -> {tight:e}"
    );
    assert!(tight < 1e-6, "tight gap {tight:e}");
}

/// A constrained system may only be run by the DAE integrator; asking
/// for any other method is refused by name rather than silently
/// integrating the unconstrained problem.
#[test]
fn a_constrained_system_refuses_the_wrong_method() {
    let mut s = pendulum(0.5, 1.0);
    s.method = Method::Adams;
    let e = integrate::run(&mut s, 1.0, 10).unwrap_err();
    assert!(e.contains("METHOD IDA"), "{e}");
    assert!(e.contains("1 rigid constraint"), "{e}");
}

/// A rod now carries a SPINNING rigid body. This used to be refused —
/// the DAE state was translational — and the whole point of moving it to
/// the full 13N packing is that the spin comes along for the ride: the
/// bob turns freely while the rod holds its length.
#[test]
fn a_rod_carries_a_spinning_rigid_body() {
    let mut s = pendulum(0.5, 1.0);
    s.objects[1].set_inertia_tensor(::physical_object::linalg::Mat3::identity());
    s.objects[1].set_angular_momentum(Vec3::new(0.0, 1.0, 0.0));

    let report = integrate::run(&mut s, 1.0, 10).expect("a rod may carry a spinning body");
    assert!(report.constraint_drift.0 < 1e-10, "|g| = {:e}", report.constraint_drift.0);
    // torque-free spin about a principal axis is conserved exactly
    let l = s.objects[1].get_angular_momentum();
    assert!((l.y - 1.0).abs() < 1e-9, "spin should be carried unchanged: {l:?}");
    assert!(!report.tolerance_floored, "a rod needs no tolerance floor");
}

/// A pendulum released anywhere comes to rest hanging straight down,
/// one rod-length below the anchor. KINSOL must find exactly that.
#[test]
fn kinsol_hangs_the_pendulum_straight_down() {
    let l = 1.0;
    let mut s = pendulum(1.0, l); // released 57 degrees off vertical
    let report = equilibrium::solve(&mut s).expect("equilibrium");

    let bob = s.objects[1].get_position();
    assert!(bob.x.abs() < 1e-12, "x should vanish, got {}", bob.x);
    assert!(bob.z.abs() < 1e-12, "z should vanish, got {}", bob.z);
    assert!((bob.y + l).abs() < 1e-12, "y should be -{l}, got {}", bob.y);
    assert!(
        report.max_net_force < 1e-10,
        "net force left on the bob: {:e}",
        report.max_net_force
    );
    assert!(report.constraint_error < 1e-12, "rod length error");
    // equilibrium means at rest, by definition
    assert_eq!(s.objects[1].get_velocity(), Vec3::zeros());
    // the anchor stayed put
    assert_eq!(s.objects[0].get_position(), Vec3::zeros());
}

/// The equilibrium KINSOL finds is a genuine one: start the integrator
/// there and nothing moves.
#[test]
fn the_equilibrium_kinsol_finds_is_actually_stationary() {
    let mut s = pendulum(1.0, 1.0);
    equilibrium::solve(&mut s).expect("equilibrium");
    let rest = s.objects[1].get_position();

    integrate::run(&mut s, 5.0, 50).expect("IDA from rest");
    let moved = (s.objects[1].get_position() - rest).norm();
    assert!(moved < 1e-8, "the 'equilibrium' drifted by {moved:e} in 5 s");
}

/// A body pulled sideways on a rod from a fixed anchor settles where the
/// rod is taut and the tension is exactly along it — pure constraint
/// mechanics, with no gravity at all.
#[test]
fn kinsol_balances_a_body_against_its_rod() {
    let mut a = physical_object::new_point(0, 1.0, Vec3::zeros(), Vec3::zeros());
    a.set_inverse_mass(0.0); // the anchor fixes the frame
    let b = physical_object::new_point(1, 1.0, Vec3::new(0.4, 0.1, 0.0), Vec3::zeros());
    let mut s = PhysicalObjectSystem::new(vec![a, b], 0.0);
    s.collide_enabled = false;
    s.external_forces[1] = Vec3::new(3.0, 0.0, 0.0); // pull it outward
    let snapshot = s.clone();
    s.constraints.add_distance(&snapshot, 0, 1, Some(2.0)).unwrap();

    let report = equilibrium::solve(&mut s).expect("equilibrium");
    let d = (s.objects[1].get_position() - s.objects[0].get_position()).norm();
    assert!((d - 2.0).abs() < 1e-10, "rod should be taut at 2.0, got {d}");
    assert!(report.max_net_force < 1e-9, "net force {:e}", report.max_net_force);
    // the pull is along +x, so the rod must line up with +x exactly
    let p = s.objects[1].get_position();
    assert!((p.x - 2.0).abs() < 1e-9, "should hang out along +x, got {p:?}");
    assert!(p.y.abs() < 1e-9 && p.z.abs() < 1e-9, "no transverse offset: {p:?}");
    assert_eq!(s.objects[0].get_position(), Vec3::zeros(), "anchor never moves");
}

/// A system in which every body is free has no *isolated* equilibrium:
/// translate the whole thing and nothing changes, so the Newton matrix is
/// singular. The refusal must say so, and say what to do about it.
#[test]
fn a_fully_free_system_is_refused_with_the_reason() {
    let a = physical_object::new_point(0, 1.0, Vec3::new(-0.4, 0.0, 0.0), Vec3::zeros());
    let b = physical_object::new_point(1, 1.0, Vec3::new(0.4, 0.0, 0.0), Vec3::zeros());
    let mut s = PhysicalObjectSystem::new(vec![a, b], 0.0);
    s.collide_enabled = false;
    s.external_forces[0] = Vec3::new(-3.0, 0.0, 0.0);
    s.external_forces[1] = Vec3::new(3.0, 0.0, 0.0);
    let snapshot = s.clone();
    s.constraints.add_distance(&snapshot, 0, 1, Some(2.0)).unwrap();

    let e = equilibrium::solve(&mut s).unwrap_err();
    assert!(e.contains("translated bodily"), "{e}");
    assert!(e.contains("inverse_mass = 0"), "{e}");
}

/// Free fall is `y(T) = y₀ + v₀T + ½gT²`, so
/// `∂y(T)/∂g = T²/2` — exactly, for every T. CVODES must reproduce it.
#[test]
fn cvodes_differentiates_free_fall_against_the_closed_form() {
    let body = physical_object::new_point(0, 2.0, Vec3::zeros(), Vec3::new(1.0, 0.0, 0.0));
    let mut s = PhysicalObjectSystem::new(vec![body], 0.0);
    s.uniform_gravity = Vec3::new(0.0, -G, 0.0);
    s.collide_enabled = false;

    let t = 3.0;
    let report = sensitivity::run(&mut s, t, &[SensParam::Gravity(1), SensParam::Mass(0)])
        .expect("CVODES sensitivity");
    assert_eq!(report.solver, "CVODES");

    let d_dg = report.per_param[0].d_position[0];
    let expect = t * t / 2.0;
    assert!(
        (d_dg.y - expect).abs() / expect < 1e-6,
        "dy/dg = {} vs analytic {expect}",
        d_dg.y
    );
    assert!(d_dg.x.abs() < 1e-9 && d_dg.z.abs() < 1e-9, "only y should respond");

    // Uniform gravity accelerates every mass equally, so the trajectory
    // does not depend on the mass AT ALL. The derivative is exactly zero,
    // and a sensitivity implementation that fumbled the parameter vector
    // would not produce exactly zero.
    let d_dm = report.per_param[1].d_position[0];
    assert_eq!(d_dm, Vec3::zeros(), "free fall is mass-independent");

    // and the state itself advanced correctly while carrying derivatives
    let y = s.objects[0].get_position();
    assert!((y.y + 0.5 * G * t * t).abs() < 1e-8, "y(T) = {}", y.y);
    assert!((y.x - t).abs() < 1e-8, "x(T) = {}", y.x);
}

/// Doubling the horizon must quadruple `∂y/∂g` — the `T²` law, sampled.
#[test]
fn cvodes_sensitivity_scales_as_the_square_of_the_horizon() {
    let run_to = |t: f64| {
        let body = physical_object::new_point(0, 1.0, Vec3::zeros(), Vec3::zeros());
        let mut s = PhysicalObjectSystem::new(vec![body], 0.0);
        s.uniform_gravity = Vec3::new(0.0, -G, 0.0);
        s.collide_enabled = false;
        sensitivity::run(&mut s, t, &[SensParam::Gravity(1)]).unwrap().per_param[0].d_position[0].y
    };
    let a = run_to(1.0);
    let b = run_to(2.0);
    assert!((a - 0.5).abs() < 1e-7, "T=1 gives {a}");
    assert!((b / a - 4.0).abs() < 1e-5, "doubling T gave a ratio of {}", b / a);
}

/// A rigid pair in uniform gravity falls exactly like a single free
/// body: the constraint force is internal and cancels. So the
/// sensitivity of its position to `g` is still `T²/2` — now computed
/// through IDAS on the DAE rather than CVODES on the ODE.
#[test]
fn idas_differentiates_a_constrained_fall() {
    let a = physical_object::new_point(0, 1.0, Vec3::zeros(), Vec3::zeros());
    let b = physical_object::new_point(1, 3.0, Vec3::new(1.0, 0.0, 0.0), Vec3::zeros());
    let mut s = PhysicalObjectSystem::new(vec![a, b], 0.0);
    s.uniform_gravity = Vec3::new(0.0, -G, 0.0);
    s.collide_enabled = false;
    s.method = Method::Ida;
    let snapshot = s.clone();
    s.constraints.add_distance(&snapshot, 0, 1, Some(1.0)).unwrap();

    let t = 3.0;
    let report = sensitivity::run(&mut s, t, &[SensParam::Gravity(1)]).expect("IDAS sensitivity");
    assert_eq!(report.solver, "IDAS", "a constrained system must route to IDAS");

    let expect = t * t / 2.0;
    for (k, d) in report.per_param[0].d_position.iter().enumerate() {
        assert!(
            (d.y - expect).abs() / expect < 1e-5,
            "obj{k}: dy/dg = {} vs analytic {expect}",
            d.y
        );
    }
    // both bodies fell the same distance, and the rod is still 1.0
    let ya = s.objects[0].get_position().y;
    let yb = s.objects[1].get_position().y;
    assert!((ya - yb).abs() < 1e-9, "the pair must fall together");
    assert!((ya + 0.5 * G * t * t).abs() < 1e-7, "fell to {ya}");
    let d = (s.objects[1].get_position() - s.objects[0].get_position()).norm();
    assert!((d - 1.0).abs() < 1e-9, "rod length {d}");
}

/// Every refusal names what to do instead.
#[test]
fn sensitivity_refusals_are_actionable() {
    let body = physical_object::new_point(0, 1.0, Vec3::zeros(), Vec3::zeros());
    let mut s = PhysicalObjectSystem::new(vec![body], 0.0);
    s.collide_enabled = false;
    let e = sensitivity::run(&mut s, 1.0, &[]).unwrap_err();
    assert!(e.contains("at least one parameter"), "{e}");

    assert!(SensParam::parse("mass 4", 1).unwrap_err().contains("only 1 object"));
    assert!(SensParam::parse("nope", 1).unwrap_err().contains("expected g_constant"));
}

/// `CONSTRAIN` with no length freezes whatever separation the bodies
/// already have, so the constraint is satisfied the instant it is made.
#[test]
fn a_bare_constrain_is_immediately_consistent() {
    let a = physical_object::new_point(0, 1.0, Vec3::zeros(), Vec3::zeros());
    let b = physical_object::new_point(1, 1.0, Vec3::new(0.3, 0.4, 0.0), Vec3::zeros());
    let mut s = PhysicalObjectSystem::new(vec![a, b], 0.0);
    s.collide_enabled = false;
    s.method = Method::Ida;
    let snapshot = s.clone();
    let k = s.constraints.add_distance(&snapshot, 0, 1, None).unwrap();
    assert_eq!(k, 0);
    assert!((match s.constraints.joints[0] { ::physical_object::constrain::Joint::Distance { length, .. } => length, _ => unreachable!() } - 0.5).abs() < 1e-15, "3-4-5 triangle");
    assert_eq!(s.constraints.drift(&s).0, 0.0, "consistent at once");

    // and it stays consistent through a run
    let r = integrate::run(&mut s, 2.0, 20).expect("IDA");
    assert!(r.constraint_drift.0 < 1e-12);
}

/// A rod between two immovable anchors constrains nothing and would make
/// the DAE singular; it is refused at CONSTRAIN time, not at RUN time.
#[test]
fn a_rod_between_two_anchors_is_refused_up_front() {
    let mut a = physical_object::new_point(0, 1.0, Vec3::zeros(), Vec3::zeros());
    let mut b = physical_object::new_point(1, 1.0, Vec3::new(1.0, 0.0, 0.0), Vec3::zeros());
    a.set_inverse_mass(0.0);
    b.set_inverse_mass(0.0);
    let s = PhysicalObjectSystem::new(vec![a, b], 0.0);
    let mut cs = ConstraintSet::default();
    let e = cs.add_distance(&s, 0, 1, None).unwrap_err();
    assert!(e.contains("both have inverse_mass"), "{e}");
}

/* ===================================================================
 * Orientation joints: ball, hinge and universal (IDA on the full 13N
 * rigid state). Every check below is against a closed form.
 * =================================================================== */

use ::physical_object::boundary::Boundary;
use ::physical_object::linalg::Mat3;

/// An immovable, non-rotating pivot.
fn world_anchor(id: usize, at: Vec3) -> physical_object {
    let mut a = physical_object::new_point(id, 1.0, at, Vec3::zeros());
    a.set_inverse_mass(0.0);
    a.set_inertia_tensor(Mat3::zeros());
    a
}

/// A box of half-extents `he`, hinged to a world anchor at the origin.
/// The pivot is the MIDPOINT of the two bodies, so putting the box at
/// `2d` from the anchor puts the pivot `d` from the box's centre of mass.
fn compound_pendulum(he: [f64; 3], d: f64, tilt: f64) -> (PhysicalObjectSystem, f64) {
    let bx = physical_object::new_from_shape(
        1,
        1.0,
        0.0,
        Vec3::new(2.0 * d * tilt.sin(), -2.0 * d * tilt.cos(), 0.0),
        Vec3::zeros(),
        Vec3::zeros(),
        Boundary::Cuboid { half_extents: he },
    );
    // small-amplitude period of a physical pendulum:
    //   T = 2 pi sqrt(I_pivot / (m g d)),  I_pivot = I_com + m d^2
    let izz = bx.get_inertia_tensor().0[2][2];
    let t = 2.0 * std::f64::consts::PI * ((izz + d * d) / (G * d)).sqrt();
    let mut s = PhysicalObjectSystem::new(vec![world_anchor(0, Vec3::zeros()), bx], 0.0);
    s.uniform_gravity = Vec3::new(0.0, -G, 0.0);
    s.collide_enabled = false;
    s.method = Method::Ida;
    let snap = s.clone();
    s.constraints.add_hinge(&snap, 0, 1, Vec3::new(0.0, 0.0, 1.0)).unwrap();
    (s, t)
}

/// A hinged rigid body is a *compound* pendulum: its period involves the
/// moment of inertia about the pivot, not just the distance to the centre
/// of mass. After one such period the body must be back where it started.
///
/// This is the check a point-mass model cannot pass: swap in `m d²` for
/// `I_com + m d²` and the period is wrong by 15 % for this box.
#[test]
fn a_hinge_gives_the_compound_pendulum_period() {
    for he in [[0.1, 0.5, 0.1], [0.4, 0.2, 0.2], [0.3, 0.3, 0.3]] {
        let (mut s, t) = compound_pendulum(he, 0.5, 0.02);
        let start = s.objects[1].get_position();
        let report = integrate::run(&mut s, t, 100).expect("hinge run");
        let closure = (s.objects[1].get_position() - start).norm();
        /* Bounds are the measured values with headroom, not aspirations:
         * at the orientation-joint tolerance floor these run 2e-8..4e-8
         * for the closure and 1e-11..7e-11 for |g|. */
        assert!(
            closure < 1e-6,
            "{he:?}: the body should return after one compound period, |Δ| = {closure:e}"
        );
        assert!(report.constraint_drift.0 < 1e-8, "|g| = {:e}", report.constraint_drift.0);
        assert!(report.constraint_drift.1 < 1e-7, "|g_dot| = {:e}", report.constraint_drift.1);
        // the pivot never moved
        assert_eq!(s.objects[0].get_position(), Vec3::zeros());
    }
}

/// A hinge leaves exactly one freedom, and it is the right one: the body
/// turns about the hinge axis and about nothing else. Its angular
/// momentum stays parallel to that axis for the whole swing.
#[test]
fn a_hinged_body_turns_only_about_its_axis() {
    let (mut s, t) = compound_pendulum([0.1, 0.5, 0.1], 0.5, 0.6);
    integrate::run(&mut s, t * 0.37, 60).expect("hinge run");
    let l = s.objects[1].get_angular_momentum();
    assert!(
        l.z.abs() > 1e-3,
        "it should actually be turning about z, L = {l:?}"
    );
    assert!(
        l.x.abs() < 1e-7 * l.z.abs().max(1.0) && l.y.abs() < 1e-7 * l.z.abs().max(1.0),
        "the hinge must admit no off-axis spin: L = {l:?}"
    );
}

/// A body on a ball joint keeps its distance from the pivot exactly,
/// while being free to turn any way — the spherical-pendulum case.
#[test]
fn a_ball_joint_holds_the_point_and_frees_the_rotation() {
    let bx = physical_object::new_from_shape(
        1,
        1.0,
        0.0,
        Vec3::new(0.6, -0.8, 0.0),
        Vec3::zeros(),
        Vec3::zeros(),
        Boundary::Cuboid { half_extents: [0.2, 0.2, 0.2] },
    );
    let mut s = PhysicalObjectSystem::new(vec![world_anchor(0, Vec3::zeros()), bx], 0.0);
    s.uniform_gravity = Vec3::new(0.0, -G, 0.0);
    s.collide_enabled = false;
    s.method = Method::Ida;
    let snap = s.clone();
    s.constraints.add_ball(&snap, 0, 1).unwrap();
    assert_eq!(s.constraints.len(), 3, "a ball joint is three rows");

    let report = integrate::run(&mut s, 2.0, 100).expect("ball run");
    assert!(report.constraint_drift.0 < 1e-7, "|g| = {:e}", report.constraint_drift.0);

    /* The joint pins the shared point, which sits at the MIDPOINT of the
     * two bodies as they stood when it was made — here (0.3, -0.4, 0).
     * The body's centre must therefore stay exactly one arm-length from
     * that pivot, whatever else it does. */
    let pivot = Vec3::new(0.3, -0.4, 0.0);
    let arm = (Vec3::new(0.6, -0.8, 0.0) - pivot).norm();
    let r = (s.objects[1].get_position() - pivot).norm();
    assert!((r - arm).abs() < 1e-8, "centre should stay at radius {arm} from the pivot, got {r}");
    /* A ball joint frees rotation, and gravity acting off the pivot is a
     * torque about it — so the body must have started turning. */
    assert!(
        s.objects[1].get_angular_momentum().norm() > 1e-6,
        "gravity about the pivot should have set it turning"
    );
}

/// A universal joint keeps its two shafts square to each other while both
/// bodies turn — that IS the joint, and the residual measures it directly.
#[test]
fn a_universal_joint_keeps_its_shafts_square() {
    let bx = |id: usize, x: f64, spin: Vec3| {
        physical_object::new_from_shape(
            id,
            1.0,
            0.0,
            Vec3::new(x, 0.0, 0.0),
            Vec3::zeros(),
            spin,
            Boundary::Cuboid { half_extents: [0.4, 0.2, 0.2] },
        )
    };
    let mut s = PhysicalObjectSystem::new(
        vec![bx(0, -0.5, Vec3::zeros()), bx(1, 0.5, Vec3::zeros())],
        0.0,
    );
    s.collide_enabled = false;
    s.method = Method::Ida;
    /* Driven from REST by a torque on the input shaft — which is what a
     * Cardan joint is for, and a start that is already consistent, so no
     * initial velocity projection is needed. */
    s.external_torques[0] = Vec3::new(0.4, 0.0, 0.0);
    let snap = s.clone();
    s.constraints
        .add_universal(&snap, 0, 1, Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0))
        .unwrap();
    assert_eq!(s.constraints.len(), 4, "a universal joint is four rows");

    let report = integrate::run(&mut s, 2.0, 80).expect("universal run");
    /* |g| covers BOTH the shared point and the shaft angle — the fourth
     * row IS the dot product of the two shafts — so this single number
     * says the whole joint held. */
    assert!(report.constraint_drift.0 < 1e-7, "|g| = {:e}", report.constraint_drift.0);
    // the torque really did spin the input shaft up from rest
    let l = s.objects[0].get_angular_momentum();
    assert!((l.x - 0.8).abs() < 1e-6, "L = tau * t = 0.8, got {l:?}");
}

/// The slider-crank of `videos/piston_crankshaft.html`, against its
/// exact kinematics.
///
/// ```text
/// mount --HINGE-- crank --BALL-- rod --BALL-- piston --PRISMATIC-- guide
/// 5 + 3 + 3 + 5 = 16 rows on 18 freedoms
/// ```
///
/// **The ball joints are not a simplification, they are the fix.**
/// Every pin in a real engine is a hinge, and four of them give
/// `5+5+5+5 = 20` rows on 18 — over-constrained by two, because a
/// planar linkage made of spatial revolutes has them all insisting on
/// the same plane. Spherical ends on the connecting rod is what real
/// multibody models do, and it leaves two freedoms: the crank angle,
/// and the rod's spin about its own length, which nothing torques.
///
/// Measuring the wrist pin from the crankshaft axis,
///
/// ```text
/// x(θ) = a cos θ + √(L² − a² sin²θ)      exactly
/// ```
///
/// and this asserts it frame by frame against a **free-running** crank.
/// Nothing drives the mechanism, so the crank swaps inertia with the
/// rod and piston and its angle wanders off any uniform `ωt` — which is
/// the point: the closed form is a statement about the linkage, not
/// about the timing, so it must hold at whatever angle the crank
/// reaches.
#[test]
fn a_slider_crank_follows_its_exact_kinematics() {
    const A: f64 = 0.5; // crank throw
    const L: f64 = 1.0; // rod; L = 2a is what puts every midpoint pivot on a pin
    const W: f64 = 2.0;
    let z = Vec3::new(0.0, 0.0, 1.0);

    let mut mount = physical_object::new_point(0, 1.0, Vec3::zeros(), Vec3::zeros());
    mount.set_inverse_mass(0.0);
    mount.set_inertia_tensor(Mat3::zeros());
    let mut crank = physical_object::new_from_shape(
        1, 2.0, 0.0, Vec3::zeros(), Vec3::zeros(), Vec3::new(0.0, 0.0, W),
        Boundary::Cylinder { radius: 0.55, half_height: 0.05 },
    );
    crank.set_angular_velocity(Vec3::new(0.0, 0.0, W));
    /* Started at top dead centre: the piston is momentarily at rest and
     * the rod turns about the wrist pin, so these are the velocities
     * that rotation implies and ġ = 0 to roundoff. */
    let mut rod = physical_object::new_from_shape(
        2, 0.4, 0.0,
        Vec3::new(L, 0.0, 0.0),
        Vec3::new(0.0, A * W / 2.0, 0.0),
        Vec3::new(0.0, 0.0, -A * W / L),
        Boundary::Cuboid { half_extents: [0.5, 0.05, 0.05] },
    );
    rod.set_angular_velocity(Vec3::new(0.0, 0.0, -A * W / L));
    let piston = physical_object::new_from_shape(
        3, 1.0, 0.0, Vec3::new(2.0 * L, 0.0, 0.0), Vec3::zeros(), Vec3::zeros(),
        Boundary::Cuboid { half_extents: [0.5, 0.3, 0.3] },
    );
    let mut guide = physical_object::new_point(4, 1.0, Vec3::new(2.0 * L, 0.0, 0.0), Vec3::zeros());
    guide.set_inverse_mass(0.0);
    guide.set_inertia_tensor(Mat3::zeros());

    let mut s = PhysicalObjectSystem::new(vec![mount, crank, rod, piston, guide], 0.0);
    s.collide_enabled = false;
    s.method = Method::Ida;
    let snap = s.clone();
    s.constraints.add_hinge(&snap, 0, 1, z).unwrap();
    s.constraints.add_ball(&snap, 1, 2).unwrap();
    s.constraints.add_ball(&snap, 2, 3).unwrap();
    s.constraints.add_prismatic(&snap, 4, 3, Vec3::new(1.0, 0.0, 0.0)).unwrap();
    assert_eq!(s.constraints.len(), 16, "hinge 5 + ball 3 + ball 3 + prismatic 5");

    let report = integrate::run(&mut s, 0.02, 1).expect("slider-crank start");
    assert_eq!(report.initial_velocity_projected, 0.0, "top dead centre is consistent");

    let spin0 = s.objects[2].get_orientation().normalize().rotate(z);
    let (mut worst, mut lo, mut hi, mut worst_g, mut rod_spin) =
        (0.0_f64, f64::MAX, f64::MIN, 0.0_f64, 0.0_f64);
    let (mut prev, mut turned) = (0.0_f64, 0.0_f64);
    for k in 2..=300 {
        let r = integrate::run(&mut s, 0.02 * f64::from(k), 1).expect("slider-crank");
        worst_g = worst_g.max(r.constraint_drift.0);
        let m = s.objects[1].get_orientation().normalize().rotate(Vec3::new(1.0, 0.0, 0.0));
        let theta = m.y.atan2(m.x);
        let mut d = theta - prev;
        while d > std::f64::consts::PI { d -= std::f64::consts::TAU }
        while d < -std::f64::consts::PI { d += std::f64::consts::TAU }
        turned += d;
        prev = theta;
        // the wrist pin: the piston's near face
        let x = s.objects[3].get_position().x - 0.5;
        let exact = A * theta.cos() + (L * L - (A * theta.sin()).powi(2)).sqrt();
        worst = worst.max((x - exact).abs());
        lo = lo.min(x);
        hi = hi.max(x);
        let spin = s.objects[2].get_orientation().normalize().rotate(z);
        rod_spin = rod_spin.max((spin - spin0).norm());
    }

    assert!(worst_g < 1e-6, "the four joints must hold: |g| = {worst_g:e}");
    assert!(turned.abs() > std::f64::consts::TAU, "the crank must go round: {turned}");
    assert!(worst < 1e-6, "piston should follow x(θ) exactly: worst {worst:e}");
    /* The stroke is L−a to L+a. The bound is looser than the closed-form
     * check above on purpose: the extremes are only ever *sampled*, and
     * the crank sweeps through dead centre between frames. Near an end
     * x is quadratic in the angle, so missing it by dt·ω = 0.04 rad
     * costs about 1e-4 — which is what the numbers below show, and is a
     * statement about the sampling rather than the mechanism. */
    assert!((lo - (L - A)).abs() < 1e-3, "bottom dead centre at L-a: {lo}");
    assert!((hi - (L + A)).abs() < 1e-3, "top dead centre at L+a: {hi}");
    /* The second freedom is genuinely passive. */
    assert!(rod_spin < 1e-9, "nothing torques the rod about its own axis: {rod_spin:e}");
}

/// The drive of `videos/rack_and_pinion.html`, against the closed form
/// for a weight winding up a flywheel.
///
/// One degree of freedom, so one equation:
///
/// ```text
/// a = m g / (m + I/r²)
/// ```
///
/// The flywheel does not act with its mass but with `I/r²`, its inertia
/// referred through the pitch radius. For a solid disc `I = ½Mr²`, so
/// that term is `M/2` — **independent of the radius entirely**. With a
/// pinion twice the rack's mass it equals `m`, and the rack falls at
/// exactly `g/2`.
///
/// Both halves are asserted: the fall against the closed form, and the
/// radius-independence by running two different pitch radii and getting
/// the same fall out. The second is the interesting one, since it is
/// the part a reader is most likely to disbelieve.
#[test]
fn a_rack_and_pinion_drive_falls_at_half_gravity() {
    const G: f64 = 0.4;
    const M_RACK: f64 = 1.0;
    const M_PINION: f64 = 2.0;
    let z = Vec3::new(0.0, 0.0, 1.0);
    let up = Vec3::new(0.0, 1.0, 0.0);

    let drop_after = |r: f64, t: f64| -> f64 {
        let mut mount = physical_object::new_point(0, 1.0, Vec3::zeros(), Vec3::zeros());
        mount.set_inverse_mass(0.0);
        mount.set_inertia_tensor(Mat3::zeros());
        let pinion = physical_object::new_from_shape(
            1, M_PINION, 0.0, Vec3::zeros(), Vec3::zeros(), Vec3::zeros(),
            Boundary::Cylinder { radius: r, half_height: 0.05 },
        );
        let mut guide = physical_object::new_point(2, 1.0, Vec3::new(r, 0.0, 0.0), Vec3::zeros());
        guide.set_inverse_mass(0.0);
        guide.set_inertia_tensor(Mat3::zeros());
        let bar = physical_object::new_from_shape(
            3, M_RACK, 0.0, Vec3::new(r, 0.0, 0.0), Vec3::zeros(), Vec3::zeros(),
            Boundary::Cuboid { half_extents: [0.06, 1.2, 0.06] },
        );
        let mut s = PhysicalObjectSystem::new(vec![mount, pinion, guide, bar], 0.0);
        s.uniform_gravity = Vec3::new(0.0, -G, 0.0);
        s.collide_enabled = false;
        s.method = Method::Ida;
        let snap = s.clone();
        s.constraints.add_hinge(&snap, 0, 1, z).unwrap();
        s.constraints.add_prismatic(&snap, 2, 3, up).unwrap();
        s.constraints.add_rack(&snap, 1, 3, z, up, r).unwrap();
        assert_eq!(s.constraints.len(), 11, "hinge 5 + prismatic 5 + rack 1");
        let report = integrate::run(&mut s, t, 200).expect("drive");
        assert!(report.constraint_drift.0 < 1e-6, "|g| = {:e}", report.constraint_drift.0);
        /* The guide is doing its job, exactly. */
        let p = s.objects[3].get_position();
        assert!((p.x - r).abs() < 1e-12 && p.z.abs() < 1e-12, "the bar left its line: {p:?}");
        p.y
    };

    /* I/r² = M/2 for a disc, so the accelerating mass is m + M/2 and
     * with M = 2m that is 2m: exactly half gravity. */
    let a = M_RACK * G / (M_RACK + M_PINION / 2.0);
    assert!((a - G / 2.0).abs() < 1e-15, "the masses were chosen so a = g/2: {a}");

    let t = 4.0;
    let expected = -0.5 * a * t * t;
    let fell = drop_after(0.6, t);
    assert!(
        (fell - expected).abs() < 1e-4,
        "the rack should fall -a t²/2 = {expected}, got {fell}"
    );

    /* The radius cancels: a different pinion, the same fall. */
    let fell_small = drop_after(0.25, t);
    assert!(
        (fell_small - fell).abs() < 1e-4,
        "the pitch radius must cancel out of a = mg/(m + I/r²): r = 0.6 fell {fell}, \
         r = 0.25 fell {fell_small}"
    );
}

/// A complete rack-and-pinion drive: the guide the last commit did not
/// have.
///
/// ```text
/// mount --HINGE-- pinion,   guide --PRISMATIC-- bar,   pinion =RACK= bar
/// ```
///
/// `5 + 5 + 1 = 11` rows on the pair's 12 freedoms, leaving the one a
/// rack-and-pinion drive has. The point of the test is the contrast:
/// the *same* drive without the `PRISMATIC` lets the reaction torque
/// twist the bar 24° off square and shove it 0.68 off its line, because
/// nothing was holding it there. With the guide both are exactly zero,
/// and the travel is unchanged.
#[test]
fn a_prismatic_guide_is_what_a_rack_runs_in() {
    const R: f64 = 0.4;
    let z = Vec3::new(0.0, 0.0, 1.0);
    let x = Vec3::new(1.0, 0.0, 0.0);
    let build = |guided: bool| {
        let mut mount = physical_object::new_point(0, 1.0, Vec3::new(0.0, 0.4, 0.0), Vec3::zeros());
        mount.set_inverse_mass(0.0);
        mount.set_inertia_tensor(Mat3::zeros());
        let pinion = physical_object::new_from_shape(
            1, 1.0, 0.0, Vec3::zeros(), Vec3::zeros(), Vec3::zeros(),
            Boundary::Cylinder { radius: R, half_height: 0.05 },
        );
        let mut guide = physical_object::new_point(2, 1.0, Vec3::new(0.0, 0.46, 0.0), Vec3::zeros());
        guide.set_inverse_mass(0.0);
        guide.set_inertia_tensor(Mat3::zeros());
        let bar = physical_object::new_from_shape(
            3, 1.0, 0.0, Vec3::new(0.0, 0.46, 0.0), Vec3::zeros(), Vec3::zeros(),
            Boundary::Cuboid { half_extents: [2.0, 0.06, 0.06] },
        );
        let mut s = PhysicalObjectSystem::new(vec![mount, pinion, guide, bar], 0.0);
        s.collide_enabled = false;
        s.method = Method::Ida;
        s.external_torques[1] = Vec3::new(0.0, 0.0, 0.4);
        let snap = s.clone();
        s.constraints.add_hinge(&snap, 0, 1, z).unwrap();
        if guided {
            s.constraints.add_prismatic(&snap, 2, 3, x).unwrap();
        }
        s.constraints.add_rack(&snap, 1, 3, z, x, R).unwrap();
        s
    };

    let mut guided = build(true);
    assert_eq!(guided.constraints.len(), 11, "hinge 5 + prismatic 5 + rack 1");
    let mut loose = build(false);

    let survey = |s: &PhysicalObjectSystem| {
        let d = s.objects[3].get_orientation().normalize().rotate(Vec3::new(1.0, 0.0, 0.0));
        let p = s.objects[3].get_position();
        (d.y.atan2(d.x).abs(), (p.y - 0.46).abs().max(p.z.abs()), p.x)
    };
    let (mut tw_g, mut off_g, mut tw_l, mut off_l) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    let (mut worst_g, mut travel) = (0.0_f64, 0.0);
    for k in 1..=200 {
        let t = 0.02 * f64::from(k);
        let r = integrate::run(&mut guided, t, 1).expect("guided drive");
        integrate::run(&mut loose, t, 1).expect("unguided drive");
        worst_g = worst_g.max(r.constraint_drift.0);
        let (a, b, xg) = survey(&guided);
        tw_g = tw_g.max(a);
        off_g = off_g.max(b);
        travel = xg;
        let (c, d, _) = survey(&loose);
        tw_l = tw_l.max(c);
        off_l = off_l.max(d);
    }

    assert!(worst_g < 1e-6, "the guided drive must hold: |g| = {worst_g:e}");
    assert!(travel.abs() > 1.0, "the rack must have travelled: {travel}");
    /* The guide does exactly what a guide does. */
    assert!(tw_g < 1e-9, "a guided rack must not twist at all: {tw_g:e} rad");
    assert!(off_g < 1e-9, "nor leave its line: {off_g:e}");
    /* And without it, the same drive does not stay square — which is
     * what makes the joint worth having rather than decorative. */
    assert!(
        tw_l > 0.1 && off_l > 0.1,
        "an unguided rack should twist and stray, else this proves nothing: \
         {tw_l} rad, {off_l}"
    );
}

/// The PRISMATIC joint's five rows, Jacobian against residual.
///
/// Two of the rows have a derivation worth checking rather than
/// trusting: `g = (Δ − R_i c)·n̂` has *both* `R_i c` and `n̂` riding on
/// body `i`, and the two contributions collapse to a single
/// `δ_i·(n̂ × Δ)` with body `j`'s orientation dropping out entirely.
/// Central differences at a pose well away from assembly, with both
/// bodies moved and turned, is what says the collapse is right.
#[test]
fn the_prismatic_jacobian_matches_its_residual() {
    let axis = Vec3::new(1.0, 0.0, 0.0);
    let make = |slide: f64, tilt: f64| {
        let mut rail = physical_object::new_from_shape(
            0, 1.0, 0.0, Vec3::new(0.0, 0.0, 0.0), Vec3::zeros(), Vec3::zeros(),
            Boundary::Cuboid { half_extents: [2.0, 0.1, 0.1] },
        );
        let mut slider = physical_object::new_from_shape(
            1, 1.0, 0.0, Vec3::new(slide, 0.3, 0.0), Vec3::zeros(), Vec3::zeros(),
            Boundary::Cuboid { half_extents: [0.2, 0.2, 0.2] },
        );
        /* tilt BOTH bodies together, which a prismatic joint permits:
         * the rail carries the slider round with it */
        let q = Quat::new((tilt / 2.0).cos(), 0.0, 0.0, (tilt / 2.0).sin());
        rail.set_orientation(q);
        slider.set_orientation(q);
        let p0 = slider.get_position();
        slider.set_position(q.rotate(p0));
        PhysicalObjectSystem::new(vec![rail, slider], 0.0)
    };

    let mut s = make(0.0, 0.0);
    let snap = s.clone();
    s.constraints.add_prismatic(&snap, 0, 1, axis).unwrap();
    assert_eq!(s.constraints.len(), 5, "a prismatic joint is five rows");

    for (slide, tilt) in [(0.0, 0.0), (0.9, 0.0), (0.0, 0.6), (-1.3, -0.9)] {
        let base = make(slide, tilt);
        let pose = ConstraintSet::poses(&base);
        let mut g0 = vec![0.0; 5];
        s.constraints.residual(&pose, &mut g0);
        for (k, g) in g0.iter().enumerate() {
            assert!(
                g.abs() < 1e-12,
                "slide {slide} tilt {tilt}: sliding and turning together must satisfy \
                 the joint, row {k} = {g}"
            );
        }
        let mut blocks = Vec::new();
        s.constraints.for_each_block(&pose, |row, b| blocks.push((row, b)));
        let g_of = |sys: &PhysicalObjectSystem, row: usize| {
            let mut o = vec![0.0; 5];
            s.constraints.residual(&ConstraintSet::poses(sys), &mut o);
            o[row]
        };
        const H: f64 = 1e-6;
        for row in 0..5 {
            for body in 0..2 {
                for dir in [axis, Vec3::new(0.0, 1.0, 0.0), Vec3::new(0.0, 0.0, 1.0)] {
                    let shift = |h: f64| {
                        let mut m = base.clone();
                        let p0 = m.objects[body].get_position();
                        m.objects[body].set_position(p0 + dir * h);
                        g_of(&m, row)
                    };
                    let measured = (shift(H) - shift(-H)) / (2.0 * H);
                    let predicted: f64 = blocks
                        .iter()
                        .filter(|(r, b)| *r == row && b.body == body)
                        .map(|(_, b)| b.jv.dot(dir))
                        .sum();
                    assert!(
                        (measured - predicted).abs() < 1e-6,
                        "row {row} body {body} translate {dir:?}: fd {measured} vs J {predicted}"
                    );
                    let turn = |h: f64| {
                        let mut m = base.clone();
                        let dq = Quat::new(
                            (h / 2.0).cos(),
                            dir.x * (h / 2.0).sin(),
                            dir.y * (h / 2.0).sin(),
                            dir.z * (h / 2.0).sin(),
                        );
                        let q0 = m.objects[body].get_orientation();
                        m.objects[body].set_orientation((dq * q0).normalize());
                        g_of(&m, row)
                    };
                    let measured = (turn(H) - turn(-H)) / (2.0 * H);
                    let predicted: f64 = blocks
                        .iter()
                        .filter(|(r, b)| *r == row && b.body == body)
                        .map(|(_, b)| b.jw.dot(dir))
                        .sum();
                    assert!(
                        (measured - predicted).abs() < 1e-6,
                        "row {row} body {body} rotate {dir:?}: fd {measured} vs J {predicted}"
                    );
                }
            }
        }
    }
}

/// A RACK converts turning into sliding: `Δs = r·θ`, driven from rest
/// by a torque, and checked well past a full turn of the pinion.
///
/// The travel is what makes this joint different from a `GEAR`. A gear
/// hides its wrapping inside a sine and pays for it with a rational
/// ratio; a rack has an unbounded coordinate to hand, so it unwraps the
/// angle from the travel instead and needs no such restriction. This
/// runs the pinion past three full turns to exercise exactly that.
///
/// **The rack is unguided, and that shows.** A real rack sits in a
/// slider, which absorbs the reaction torque; this joint set has no
/// prismatic constraint, so the bar takes that torque and slowly turns.
/// The row couples the pinion's turn *relative to the rack*, so it stays
/// exact regardless — but it is why the assertion is written against the
/// relative angle rather than the pinion's absolute one.
#[test]
fn a_rack_converts_turning_into_sliding() {
    const R: f64 = 0.4;
    let z = Vec3::new(0.0, 0.0, 1.0);
    let mut mount = physical_object::new_point(0, 1.0, Vec3::new(0.0, 0.4, 0.0), Vec3::zeros());
    mount.set_inverse_mass(0.0);
    mount.set_inertia_tensor(Mat3::zeros());
    let pinion = physical_object::new_from_shape(
        1, 1.0, 0.0, Vec3::zeros(), Vec3::zeros(), Vec3::zeros(),
        Boundary::Cylinder { radius: R, half_height: 0.05 },
    );
    let bar = physical_object::new_from_shape(
        2, 1.0, 0.0, Vec3::new(0.0, 0.46, 0.0), Vec3::zeros(), Vec3::zeros(),
        Boundary::Cuboid { half_extents: [2.0, 0.06, 0.06] },
    );
    let mut s = PhysicalObjectSystem::new(vec![mount, pinion, bar], 0.0);
    s.collide_enabled = false;
    s.method = Method::Ida;
    s.external_torques[1] = Vec3::new(0.0, 0.0, 0.4); // crank the pinion
    let snap = s.clone();
    s.constraints.add_hinge(&snap, 0, 1, z).unwrap();
    s.constraints.add_rack(&snap, 1, 2, z, Vec3::new(1.0, 0.0, 0.0), R).unwrap();
    assert_eq!(s.constraints.len(), 6, "hinge 5 + rack 1");

    /* From REST: the relation cannot have come from the start. */
    let report = integrate::run(&mut s, 0.02, 1).expect("rack start");
    assert_eq!(report.initial_velocity_projected, 0.0, "from rest");

    let offset = s.objects[2].get_position() - s.objects[1].get_position();
    let sample = |s: &PhysicalObjectSystem| {
        let dir = s.objects[2].get_orientation().normalize().rotate(Vec3::new(1.0, 0.0, 0.0));
        let mark = s.objects[1].get_orientation().normalize().rotate(Vec3::new(1.0, 0.0, 0.0));
        let travel = (s.objects[2].get_position() - s.objects[1].get_position() - offset).dot(dir);
        // the pinion's turn relative to the rack, about z, wrapped
        let theta = dir.cross(mark).z.atan2(dir.dot(mark));
        (travel, theta)
    };
    let (mut prev, mut turned) = (sample(&s).1, 0.0_f64);
    let (mut worst, mut end_travel, mut worst_g) = (0.0_f64, 0.0, 0.0_f64);
    for k in 2..=200 {
        let r = integrate::run(&mut s, 0.02 * f64::from(k), 1).expect("rack run");
        worst_g = worst_g.max(r.constraint_drift.0);
        let (travel, theta) = sample(&s);
        let mut d = theta - prev;
        while d > std::f64::consts::PI { d -= std::f64::consts::TAU }
        while d < -std::f64::consts::PI { d += std::f64::consts::TAU }
        turned += d;
        prev = theta;
        worst = worst.max((travel - R * turned).abs());
        end_travel = travel;
    }
    /* Past a full turn is the point: a wrapped angle would have jumped
     * by 2π here, and the travel is what tells the row it did not. */
    assert!(
        turned.abs() > std::f64::consts::TAU,
        "the pinion must pass a full turn, or the unwrapping is untested: {turned}"
    );
    assert!(end_travel.abs() > 1.0, "and the rack must actually have moved: {end_travel}");
    /* The row itself, as the solver reports it. */
    assert!(worst_g < 1e-7, "the rack row must hold: |g| = {worst_g:e}");
    /* And the relation rebuilt independently here, from wrapped angle
     * increments summed over 200 restarts. That reconstruction carries
     * its own accumulated error — about 5e-6 over one and a half turns —
     * which is why it is checked against a looser bound than the row. */
    assert!(worst < 1e-4, "travel should track r*theta: worst gap {worst:e}");
}

/// A rack refuses what it cannot mean, and stacks on a bearing.
#[test]
fn a_rack_refuses_what_it_cannot_mean() {
    let make = || {
        let a = physical_object::new_point(0, 1.0, Vec3::zeros(), Vec3::zeros());
        let b = physical_object::new_point(1, 1.0, Vec3::new(0.0, 1.0, 0.0), Vec3::zeros());
        PhysicalObjectSystem::new(vec![a, b], 0.0)
    };
    let z = Vec3::new(0.0, 0.0, 1.0);
    let x = Vec3::new(1.0, 0.0, 0.0);

    let mut s = make();
    let snap = s.clone();
    /* A rack runs ACROSS its pinion; along it is a different mechanism,
     * and projecting silently would build that other one. */
    let e = s.constraints.add_rack(&snap, 0, 1, z, z, 0.4).unwrap_err();
    assert!(e.contains("perpendicular"), "{e}");
    assert!(s.constraints.add_rack(&snap, 0, 1, z, x, 0.0).unwrap_err().contains("pitch radius"));
    assert!(s.constraints.add_rack(&snap, 0, 1, Vec3::zeros(), x, 0.4).unwrap_err().contains("axis"));

    /* Unlike a gear, no rationality limit: the travel resolves the
     * wrapping, so any finite radius is representable. */
    let mut s = make();
    let snap = s.clone();
    s.constraints.add_rack(&snap, 0, 1, z, x, std::f64::consts::FRAC_1_PI).unwrap();
    let e = s.constraints.add_rack(&snap, 0, 1, z, x, 0.4).unwrap_err();
    assert!(e.contains("already joined by a gear or rack"), "{e}");
}

/// The RACK joint's Jacobian, against its own residual, including well
/// past a full turn of the pinion.
///
/// The row is `g = Δs − r·θ` with `θ` unwrapped from the travel, and the
/// claim is that the unwrapping count is locally constant so the
/// derivative is the plain one. That is exactly the thing to check by
/// finite differences, and to check *after several turns*, where a
/// wrapped angle would already have jumped.
#[test]
fn the_rack_jacobian_matches_its_residual() {
    let z = Vec3::new(0.0, 0.0, 1.0);
    let x = Vec3::new(1.0, 0.0, 0.0);
    for radius in [0.4_f64, -0.25, 1.0] {
        for turns in [0.0_f64, 0.3, 2.7, -3.4] {
            let theta = turns * std::f64::consts::TAU;
            let make = |th: f64, s: f64| {
                let mut pin = physical_object::new_from_shape(
                    0, 1.0, 0.0, Vec3::zeros(), Vec3::zeros(), Vec3::zeros(),
                    Boundary::Cylinder { radius: radius.abs(), half_height: 0.05 },
                );
                pin.set_orientation(Quat::new((th / 2.0).cos(), 0.0, 0.0, (th / 2.0).sin()));
                let bar = physical_object::new_from_shape(
                    1, 1.0, 0.0,
                    Vec3::new(s, 1.0, 0.0), Vec3::zeros(), Vec3::zeros(),
                    Boundary::Cuboid { half_extents: [2.0, 0.05, 0.05] },
                );
                PhysicalObjectSystem::new(vec![pin, bar], 0.0)
            };
            /* built at rest, then moved to a consistent pose far along */
            let mut s = make(0.0, 0.0);
            let snap = s.clone();
            s.constraints.add_rack(&snap, 0, 1, z, x, radius).unwrap();
            assert_eq!(s.constraints.len(), 1, "a rack is one row");

            let base = make(theta, radius * theta);
            let pose = ConstraintSet::poses(&base);
            let mut g0 = vec![0.0; 1];
            s.constraints.residual(&pose, &mut g0);
            assert!(
                g0[0].abs() < 1e-9,
                "radius {radius}, {turns} turns: a consistent pose should satisfy the row, \
                 got g = {}",
                g0[0]
            );

            let mut blocks = Vec::new();
            s.constraints.for_each_block(&pose, |_, b| blocks.push(b));
            let g_of = |sys: &PhysicalObjectSystem| {
                let mut o = vec![0.0; 1];
                s.constraints.residual(&ConstraintSet::poses(sys), &mut o);
                o[0]
            };
            /* CENTRAL differences: the forward kind carries an O(H)
             * truncation error proportional to the second derivative,
             * and after three turns the rack has travelled 21 units, so
             * that error alone is ~1e-5 — the size of the thing being
             * measured. Central differencing is O(H²) and leaves the
             * comparison about the Jacobian rather than about the
             * difference scheme. */
            const H: f64 = 1e-6;
            let spin = |dir: Vec3, h: f64| {
                Quat::new(
                    (h / 2.0).cos(),
                    dir.x * (h / 2.0).sin(),
                    dir.y * (h / 2.0).sin(),
                    dir.z * (h / 2.0).sin(),
                )
            };
            for body in 0..2 {
                for dir in [x, Vec3::new(0.0, 1.0, 0.0), z] {
                    let shift = |h: f64| {
                        let mut m = base.clone();
                        let p0 = m.objects[body].get_position();
                        m.objects[body].set_position(p0 + dir * h);
                        g_of(&m)
                    };
                    let measured = (shift(H) - shift(-H)) / (2.0 * H);
                    let predicted: f64 =
                        blocks.iter().filter(|b| b.body == body).map(|b| b.jv.dot(dir)).sum();
                    assert!(
                        (measured - predicted).abs() < 1e-6,
                        "r {radius} turns {turns} body {body} translate {dir:?}: \
                         fd {measured} vs J {predicted}"
                    );
                    let turn = |h: f64| {
                        let mut m = base.clone();
                        let q0 = m.objects[body].get_orientation();
                        m.objects[body].set_orientation((spin(dir, h) * q0).normalize());
                        g_of(&m)
                    };
                    let measured = (turn(H) - turn(-H)) / (2.0 * H);
                    let predicted: f64 =
                        blocks.iter().filter(|b| b.body == body).map(|b| b.jw.dot(dir)).sum();
                    assert!(
                        (measured - predicted).abs() < 1e-6,
                        "r {radius} turns {turns} body {body} rotate {dir:?}: \
                         fd {measured} vs J {predicted}"
                    );
                }
            }
        }
    }
}

/// A GEAR holds `ω_i = −ratio · ω_j` about its axis, under load.
///
/// Two wheels on their own bearings, geared 2:1, with a torque applied
/// to one of them. Nothing about the start encodes the ratio: both
/// begin at rest, and the torque has to drive both through the gear.
/// The ratio is checked as a ratio, against the closed form, and the
/// row count is checked because a gear stacking on a bearing is the
/// arrangement that makes a gear train expressible at all.
#[test]
fn a_gear_holds_its_ratio_under_load() {
    const RATIO: f64 = 2.0;
    let wheel = |id: usize, x: f64, r: f64| {
        physical_object::new_from_shape(
            id, 1.0, 0.0,
            Vec3::new(x, 0.0, 0.0),
            Vec3::zeros(),
            Vec3::zeros(),
            Boundary::Cylinder { radius: r, half_height: 0.05 },
        )
    };
    let mut s = PhysicalObjectSystem::new(
        vec![
            world_anchor(0, Vec3::new(0.0, 0.0, 0.0)),
            wheel(1, 0.0, 0.3),   // the small wheel, on its own bearing
            world_anchor(2, Vec3::new(1.0, 0.0, 0.0)),
            wheel(3, 1.0, 0.6),   // the large one
        ],
        0.0,
    );
    s.collide_enabled = false;
    s.method = Method::Ida;
    s.external_torques[3] = Vec3::new(0.0, 0.0, 0.02); // drive the large wheel
    let snap = s.clone();
    let z = Vec3::new(0.0, 0.0, 1.0);
    s.constraints.add_hinge(&snap, 0, 1, z).unwrap();
    s.constraints.add_hinge(&snap, 2, 3, z).unwrap();
    s.constraints.add_gear(&snap, 1, 3, z, RATIO).unwrap();
    assert_eq!(s.constraints.len(), 11, "hinge 5 + hinge 5 + gear 1");

    /* Started from REST, so the ratio cannot have been smuggled in
     * through the initial condition: g_dot = 0 trivially. */
    let report = integrate::run(&mut s, 3.0, 60).expect("gear train");
    assert_eq!(report.initial_velocity_projected, 0.0, "from rest");
    assert!(report.constraint_drift.0 < 1e-6, "|g| = {:e}", report.constraint_drift.0);

    let small = s.objects[1].get_angular_velocity().z;
    let large = s.objects[3].get_angular_velocity().z;
    assert!(large.abs() > 0.1, "the torque must actually have spun it: {large}");
    assert!(
        (small + RATIO * large).abs() < 1e-6 * large.abs().max(1.0),
        "the gear should hold w_small = -{RATIO} * w_large: got {small} and {large}"
    );
}

/// A gear ratio has to be rational, and the refusal says why rather than
/// rounding quietly. It also refuses a second gear on the same pair,
/// while still allowing the first to stack on a bearing.
#[test]
fn a_gear_refuses_what_it_cannot_represent() {
    let make = || {
        let a = physical_object::new_point(0, 1.0, Vec3::zeros(), Vec3::zeros());
        let b = physical_object::new_point(1, 1.0, Vec3::new(1.0, 0.0, 0.0), Vec3::zeros());
        PhysicalObjectSystem::new(vec![a, b], 0.0)
    };
    let z = Vec3::new(0.0, 0.0, 1.0);

    let mut s = make();
    let snap = s.clone();
    let e = s.constraints.add_gear(&snap, 0, 1, z, std::f64::consts::FRAC_1_PI).unwrap_err();
    assert!(e.contains("rational"), "{e}");
    assert!(e.contains("sin("), "the message should say what the limit comes from: {e}");

    // a third and a half are fine
    let mut s = make();
    let snap = s.clone();
    s.constraints.add_gear(&snap, 0, 1, z, 1.0 / 3.0).unwrap();
    // but a second gear on the same pair is redundant
    let e = s.constraints.add_gear(&snap, 0, 1, z, 2.0).unwrap_err();
    assert!(e.contains("already joined by a gear"), "{e}");

    let mut s = make();
    let snap = s.clone();
    assert!(s.constraints.add_gear(&snap, 0, 1, z, 0.0).unwrap_err().contains("non-zero"));
    assert!(s.constraints.add_gear(&snap, 0, 1, Vec3::zeros(), 2.0).unwrap_err().contains("axis"));
}

/// The GEAR joint's Jacobian, checked against its own residual by
/// finite differences.
///
/// The row is `g = sin(q θ_i + p θ_j)`, and the claim underneath it is
/// that `dθ = δ·axis` **exactly** — no cross terms from perturbations
/// off the axis. That is what makes the Jacobian just the axis, scaled,
/// and it is worth checking rather than believing: rotate each body a
/// little about each of three directions and compare the measured
/// change in `g` with what the Jacobian predicts.
#[test]
fn the_gear_jacobian_matches_its_residual() {
    for (ratio, tilt) in [(1.0, 0.0), (2.0, 0.0), (-1.5, 0.0), (2.0, 0.3), (0.5, -0.4)] {
        let make = |qi: Quat, qj: Quat| {
            let mut a = physical_object::new_from_shape(
                0, 1.0, 0.0, Vec3::zeros(), Vec3::zeros(), Vec3::zeros(),
                Boundary::Cylinder { radius: 0.4, half_height: 0.05 },
            );
            let mut b = physical_object::new_from_shape(
                1, 1.0, 0.0, Vec3::new(1.0, 0.0, 0.0), Vec3::zeros(), Vec3::zeros(),
                Boundary::Cylinder { radius: 0.4, half_height: 0.05 },
            );
            a.set_orientation(qi);
            b.set_orientation(qj);
            PhysicalObjectSystem::new(vec![a, b], 0.0)
        };
        let axis = Vec3::new(0.0, 0.0, 1.0);
        let spin = |th: f64| Quat::new((th / 2.0).cos(), 0.0, 0.0, (th / 2.0).sin());

        let mut s = make(Quat::identity(), Quat::identity());
        s.constraints.add_gear(&s.clone(), 0, 1, axis, ratio).unwrap();
        assert_eq!(s.constraints.len(), 1, "a gear is one row");

        /* Move both bodies off the reference so cos Θ is not 1 and the
         * check is not accidentally trivial. */
        let base = make(spin(0.37 + tilt), spin(-0.11));
        let g_of = |sys: &PhysicalObjectSystem| {
            let pose = ConstraintSet::poses(sys);
            let mut out = vec![0.0; 1];
            s.constraints.residual(&pose, &mut out);
            out[0]
        };
        let pose = ConstraintSet::poses(&base);
        let mut blocks = Vec::new();
        s.constraints.for_each_block(&pose, |row, b| {
            assert_eq!(row, 0);
            blocks.push(b);
        });

        const H: f64 = 1e-6;
        for body in 0..2 {
            for dir in [Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0), axis] {
                /* rotate `body` by H about `dir`, and see what g does */
                let mut moved = base.clone();
                let dq = Quat::new(
                    (H / 2.0).cos(),
                    dir.x * (H / 2.0).sin(),
                    dir.y * (H / 2.0).sin(),
                    dir.z * (H / 2.0).sin(),
                );
                let q0 = moved.objects[body].get_orientation();
                moved.objects[body].set_orientation((dq * q0).normalize());
                let measured = (g_of(&moved) - g_of(&base)) / H;
                let predicted: f64 = blocks
                    .iter()
                    .filter(|b| b.body == body)
                    .map(|b| b.jw.dot(dir))
                    .sum();
                assert!(
                    (measured - predicted).abs() < 1e-5,
                    "ratio {ratio}, body {body}, dir {dir:?}: finite difference {measured}, \
                     Jacobian {predicted}"
                );
            }
        }
    }
}

/// The Cardan gears of `videos/cardan_gear.html`, against the
/// degenerate hypocycloid — and the difference a real constraint makes.
///
/// A wheel of radius `r` rolling inside a ring of radius `2r` sends
/// every point of its rim along a **straight line**, a diameter of the
/// ring. The hypocycloid of ratio 2 does not approximate a line, it is
/// one, and the rim point sits at `P = (2r cos θ, 0)`.
///
/// The rolling is a `GEAR` of ratio 1: the wheel turns once backwards
/// for each forward turn of the carrier. The mechanism is
///
/// ```text
/// centre --HINGE-- crank --HINGE-- planet,  planet =GEAR= crank
/// ```
///
/// 5 + 5 + 1 = 11 rows on the pair's 12 freedoms, leaving the one a
/// Cardan gear train has.
///
/// The second half of the test is the point of having the joint at all:
/// the same mechanism is **disturbed with a torque**, and the line must
/// survive. Impose the ratio as an initial condition instead and the
/// same torque destroys it — the rim point wanders `0.75` off a line it
/// otherwise holds to `1e-8`.
#[test]
fn cardan_gears_send_a_rim_point_along_a_straight_line() {
    const R: f64 = 0.5; // wheel radius; the ring is 2R
    let pi = std::f64::consts::PI;

    let build = |geared: bool, torque: f64| {
        let mut centre = physical_object::new_point(0, 1.0, Vec3::new(-R, 0.0, 0.0), Vec3::zeros());
        centre.set_inverse_mass(0.0);
        centre.set_inertia_tensor(Mat3::zeros());
        let mut crank = physical_object::new_from_shape(
            1, 0.3, 0.0,
            Vec3::new(R, 0.0, 0.0),
            Vec3::new(0.0, R, 0.0), // v = ω × r
            Vec3::new(0.0, 0.0, 1.0),
            Boundary::Cuboid { half_extents: [R, 0.04, 0.02] },
        );
        crank.set_angular_velocity(Vec3::new(0.0, 0.0, 1.0));
        let mut planet = physical_object::new_from_shape(
            2, 1.0, 0.0,
            Vec3::new(R, 0.0, 0.0),
            Vec3::new(0.0, R, 0.0),
            Vec3::new(0.0, 0.0, -1.0), // the rolling ratio, as a start
            Boundary::Cylinder { radius: R, half_height: 0.03 },
        );
        planet.set_angular_velocity(Vec3::new(0.0, 0.0, -1.0));
        let mut s = PhysicalObjectSystem::new(vec![centre, crank, planet], 0.0);
        s.collide_enabled = false;
        s.method = Method::Ida;
        s.external_torques[2] = Vec3::new(0.0, 0.0, torque);
        let snap = s.clone();
        s.constraints.add_hinge(&snap, 0, 1, Vec3::new(0.0, 0.0, 1.0)).unwrap();
        s.constraints.add_hinge(&snap, 1, 2, Vec3::new(0.0, 0.0, 1.0)).unwrap();
        if geared {
            /* Stacks on the bearing: the hinge supports the wheel, the
             * gear sets the ratio. A gear is not a geometric joint, so
             * the two are not redundant. */
            s.constraints.add_gear(&snap, 2, 1, Vec3::new(0.0, 0.0, 1.0), 1.0).unwrap();
        }
        s
    };
    let rim = |s: &PhysicalObjectSystem| {
        s.objects[2].get_position()
            + s.objects[2].get_orientation().normalize().rotate(Vec3::new(R, 0.0, 0.0))
    };

    let mut s = build(true, 0.0);
    assert_eq!(s.constraints.len(), 11, "hinge 5 + hinge 5 + gear 1");
    assert!((rim(&s) - Vec3::new(2.0 * R, 0.0, 0.0)).norm() < 1e-12, "starts at +2r");
    let report = integrate::run(&mut s, 0.02, 1).expect("gear start");
    assert_eq!(report.initial_velocity_projected, 0.0, "the start is consistent");

    /* θ = π/2 puts the rim point at the centre of the ring, π at the far
     * end, 2π back where it began. */
    for (theta, want_x) in [(pi / 2.0, 0.0), (pi, -2.0 * R), (2.0 * pi, 2.0 * R)] {
        let r = integrate::run(&mut s, theta, 200).expect("gear run");
        assert!(r.constraint_drift.0 < 1e-6, "|g| = {:e}", r.constraint_drift.0);
        let p = rim(&s);
        assert!((p.x - want_x).abs() < 1e-5, "at {theta} want x = {want_x}, got {p:?}");
        assert!(p.y.abs() < 1e-5, "the rim point left the line: y = {:e}", p.y);
    }

    /* Now lean on it. A real constraint does not care. */
    let mut geared = build(true, 0.05);
    let mut imposed = build(false, 0.05);
    let (mut worst_geared, mut worst_imposed) = (0.0_f64, 0.0_f64);
    for k in 1..=100 {
        let t = 0.05 * f64::from(k);
        integrate::run(&mut geared, t, 1).expect("geared under torque");
        integrate::run(&mut imposed, t, 1).expect("imposed under torque");
        worst_geared = worst_geared.max(rim(&geared).y.abs());
        worst_imposed = worst_imposed.max(rim(&imposed).y.abs());
    }
    assert!(worst_geared < 1e-5, "the gear must hold the line: {worst_geared:e}");
    assert!(
        worst_imposed > 1e-2,
        "an imposed ratio should NOT survive a torque, else this test proves nothing: \
         {worst_imposed:e}"
    );
}

/// The compass of `videos/cardan_compass.html`, against the physical
/// pendulum period.
///
/// ```text
/// frame --HINGE x-- ring --HINGE z-- bowl
/// ```
///
/// The same two-ring suspension as the gyroscope gimbal, differing in
/// one number: there every centre of mass sat **on** the pivot, so
/// gravity had no lever arm and did nothing. Here the bowl is
/// *pendulous* — its centre of mass hangs `d` below the pivot — and
/// that is the entire mechanism. A ship's compass is not held level by
/// its bearings; it is held level by its own weight.
///
/// So each axis is a physical pendulum:
///
/// ```text
/// T = 2π √( I / (M g d) ),   I = M(3R² + 4h²)/12 + M d²
/// ```
///
/// This excites the inner hinge **alone**, where the attribution is
/// exact: the measured period sits `4.99e-4` above the linear formula,
/// and the finite-amplitude correction `θ²/16` for this swing predicts
/// `5.02e-4`. The residual is the known nonlinear term, not error —
/// which is why the assertion is against `θ²/16` rather than a round
/// tolerance.
#[test]
fn a_cardan_compass_swings_at_the_physical_pendulum_period() {
    const MB: f64 = 2.0; // bowl
    const RB: f64 = 0.6;
    const HB: f64 = 0.05;
    const D: f64 = 0.12; // centre of mass below the pivot
    const W: f64 = 0.3; // the kick, about the inner hinge

    let i_pivot = MB * (3.0 * RB * RB + 4.0 * HB * HB) / 12.0 + MB * D * D;
    let period = 2.0 * std::f64::consts::PI * (i_pivot / (MB * G * D)).sqrt();

    let s2 = std::f64::consts::FRAC_1_SQRT_2;
    let mut frame = physical_object::new_point(0, 1.0, Vec3::new(0.0, -D, 0.0), Vec3::zeros());
    frame.set_inverse_mass(0.0);
    frame.set_inertia_tensor(Mat3::zeros());

    /* The midpoint pivot rule forces p_frame = p_bowl = -p_ring, so a
     * bowl hung d below centre puts the ring's centre d above it. */
    let mut ring = physical_object::new_from_shape(
        1,
        0.2,
        0.0,
        Vec3::new(0.0, D, 0.0),
        Vec3::zeros(),
        Vec3::zeros(),
        Boundary::Torus { ring_radius: 0.85, tube_radius: 0.035 },
    );
    ring.set_orientation(Quat::new(s2, 0.0, s2, 0.0));

    /* Kicked about the inner hinge only. The linear velocity is the one
     * that rotation implies about a pivot at the origin, v = ω × r, so
     * the shared point stays still and ġ = 0. */
    let mut bowl = physical_object::new_from_shape(
        2,
        MB,
        0.0,
        Vec3::new(0.0, -D, 0.0),
        Vec3::new(W * D, 0.0, 0.0),
        Vec3::new(0.0, 0.0, W),
        Boundary::Cylinder { radius: RB, half_height: HB },
    );
    bowl.set_orientation(Quat::new(s2, s2, 0.0, 0.0));
    bowl.set_angular_velocity(Vec3::new(0.0, 0.0, W));

    let mut s = PhysicalObjectSystem::new(vec![frame, ring, bowl], 0.0);
    s.uniform_gravity = Vec3::new(0.0, -G, 0.0);
    s.collide_enabled = false;
    s.method = Method::Ida;
    let snap = s.clone();
    s.constraints.add_hinge(&snap, 0, 1, Vec3::new(1.0, 0.0, 0.0)).unwrap();
    s.constraints.add_hinge(&snap, 1, 2, Vec3::new(0.0, 0.0, 1.0)).unwrap();
    assert_eq!(s.constraints.len(), 10, "two hinges are five rows each");

    /* The compass card's upward normal; its tilt off level is the swing. */
    let tilt = |s: &PhysicalObjectSystem| {
        let up = -s.objects[2].get_orientation().normalize().rotate(Vec3::new(0.0, 0.0, 1.0));
        up.x.atan2(up.y)
    };

    let dt = 0.002;
    let (mut prev_t, mut prev, mut crossings, mut amp) = (0.0, tilt(&s), Vec::new(), 0.0_f64);
    let mut worst_g = 0.0_f64;
    for k in 1..=2000 {
        let t = dt * f64::from(k);
        let r = integrate::run(&mut s, t, 1).expect("compass run");
        worst_g = worst_g.max(r.constraint_drift.0);
        let now = tilt(&s);
        amp = amp.max(now.abs());
        if prev < 0.0 && now >= 0.0 {
            // linear interpolation onto the crossing
            crossings.push(prev_t + dt * (-prev) / (now - prev));
        }
        prev_t = t;
        prev = now;
    }
    assert!(worst_g < 1e-7, "the two hinges must hold: |g| = {worst_g:e}");
    assert!(crossings.len() >= 2, "need at least one full swing");

    let measured = (crossings[crossings.len() - 1] - crossings[0])
        / (crossings.len() - 1) as f64;
    let excess = (measured - period) / period;
    let theta2_16 = amp * amp / 16.0;
    /* The period is long by exactly the finite-amplitude term, so the
     * two agree to a few percent OF EACH OTHER — a far tighter claim
     * than "the period is close to the formula". */
    assert!(
        (excess - theta2_16).abs() < 0.1 * theta2_16,
        "the excess should BE the θ²/16 correction: measured {measured} vs {period} \
         (excess {excess:e}, θ²/16 = {theta2_16:e}, amplitude {} deg)",
        amp.to_degrees()
    );
}

/// The gimbal of `videos/gyroscope_gimbal.html`, and the conservation
/// law a hinge hands you for free.
///
/// ```text
/// base --HINGE y-- outer --HINGE x-- inner --HINGE z-- rotor
/// ```
///
/// Three hinges on three perpendicular axes: 15 rows on 18 freedoms,
/// leaving exactly the three gimbal angles. Every body is concentric,
/// so all three axes pass through one point — which is what makes it a
/// gimbal rather than a linkage.
///
/// **A hinge transmits no torque about its own axis.** That is the
/// freedom it grants, and it is also a conservation law:
///
/// - the outermost hinge turns about the vertical, so nothing can
///   torque the assembly about the vertical, and total `L·ŷ` is
///   conserved — *exactly*, because the angular momenta are integrated
///   state rather than a derived quantity;
/// - every centre of mass sits **on** the pivot, so gravity has no
///   lever arm and adds no torque either. That is the whole difference
///   from `a_spinning_top_precesses_at_the_closed_form_rate`, where the
///   arm `r` is what drives the precession.
#[test]
fn a_gimbal_conserves_angular_momentum_about_its_outer_axis() {
    let ring = |id: usize, mass: f64, r: f64, spin: Vec3, q: Quat| {
        let mut b = physical_object::new_from_shape(
            id,
            mass,
            0.0,
            Vec3::zeros(), // concentric: every pivot lands on the origin
            Vec3::zeros(),
            spin,
            Boundary::Torus { ring_radius: r, tube_radius: 0.04 },
        );
        b.set_orientation(q);
        b.set_angular_velocity(spin);
        b
    };
    let s2 = std::f64::consts::FRAC_1_SQRT_2;
    let mut base = physical_object::new_point(0, 1.0, Vec3::zeros(), Vec3::zeros());
    base.set_inverse_mass(0.0);
    base.set_inertia_tensor(Mat3::zeros());

    // started as a rigid turn about the vertical, plus the rotor's spin
    let turn = Vec3::new(0.0, 1.0, 0.0);
    let mut rotor = physical_object::new_from_shape(
        3,
        2.0,
        0.0,
        Vec3::zeros(),
        Vec3::zeros(),
        Vec3::new(0.0, 1.0, 15.0),
        Boundary::Cylinder { radius: 0.5, half_height: 0.06 },
    );
    rotor.set_angular_velocity(Vec3::new(0.0, 1.0, 15.0));

    let mut s = PhysicalObjectSystem::new(
        vec![
            base,
            /* Each ring lies in the plane holding the axis it swings on
             * and the axis it carries — a gimbal ring pivots about a
             * DIAMETER. Outer swings on y and carries x, so plane xy,
             * symmetry axis z: no turn. Inner swings on x and carries z,
             * so plane xz, symmetry axis y: a quarter turn about x. */
            ring(1, 0.5, 0.9, turn, Quat::identity()),
            ring(2, 0.4, 0.7, turn, Quat::new(s2, s2, 0.0, 0.0)),
            rotor,
        ],
        0.0,
    );
    s.uniform_gravity = Vec3::new(0.0, -3.0, 0.0);
    s.collide_enabled = false;
    s.method = Method::Ida;
    let snap = s.clone();
    s.constraints.add_hinge(&snap, 0, 1, Vec3::new(0.0, 1.0, 0.0)).unwrap();
    s.constraints.add_hinge(&snap, 1, 2, Vec3::new(1.0, 0.0, 0.0)).unwrap();
    s.constraints.add_hinge(&snap, 2, 3, Vec3::new(0.0, 0.0, 1.0)).unwrap();
    assert_eq!(s.constraints.len(), 15, "three hinges are five rows each");

    let l_y = |s: &PhysicalObjectSystem| -> f64 {
        s.objects[1..].iter().map(|o| o.get_angular_momentum().y).sum()
    };
    let axis = |s: &PhysicalObjectSystem| {
        s.objects[3].get_orientation().normalize().rotate(Vec3::new(0.0, 0.0, 1.0))
    };
    let (l0, a0) = (l_y(&s), axis(&s));

    let report = integrate::run(&mut s, 0.02, 1).expect("gimbal start");
    assert_eq!(report.initial_velocity_projected, 0.0, "the start is consistent");

    let (mut worst_l, mut worst_g, mut tilt, mut drift) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
    for k in 2..=315 {
        let r = integrate::run(&mut s, 0.02 * f64::from(k), 1).expect("gimbal run");
        worst_g = worst_g.max(r.constraint_drift.0);
        worst_l = worst_l.max((l_y(&s) - l0).abs());
        tilt = tilt.max(axis(&s).dot(a0).clamp(-1.0, 1.0).acos().to_degrees());
        for b in &s.objects {
            drift = drift.max(b.get_position().norm());
        }
    }

    assert!(worst_g < 1e-6, "the three hinges must hold: |g| = {worst_g:e}");
    /* The invariant, and it is not approximate. */
    assert!(worst_l < 1e-12, "L.y must be conserved exactly: drifted {worst_l:e}");
    /* A gimbal holds a point: nothing translates, at all. */
    assert!(drift < 1e-20, "every centre stays on the pivot: {drift:e}");
    /* And the gyroscope turned the push into a tilt at right angles. */
    assert!(tilt > 5.0, "the rotor axis should be driven off its start: {tilt}");
}

/// The gyroscope of `videos/spinning_top.html`, against the closed form
/// for steady precession.
///
/// A symmetric top spinning at `ω₃` about its own axis, its centre of
/// mass a distance `r` from the pivot, precesses about the vertical at
/// `Ω = M g r / (I₃ ω₃)`. That is normally quoted as the *fast-top*
/// approximation, but the exact steady-precession condition is
///
/// ```text
/// M g r = Ω I₃ ω₃ − I₁ Ω² cos θ
/// ```
///
/// and this top is mounted with its axis **horizontal**, `θ = 90°`, so
/// the correction carries a `cos θ` that is exactly zero. The simple
/// formula is therefore exact here, which is what makes it worth
/// asserting rather than merely plotting.
///
/// The check is closed-loop: run for exactly one predicted period and
/// the symmetry axis must come back to where it started. Nothing about
/// drift can fake that.
#[test]
fn a_spinning_top_precesses_at_the_closed_form_rate() {
    const M: f64 = 1.0;
    const GRAV: f64 = 3.0;
    const R: f64 = 0.42; // flywheel radius
    const ARM: f64 = 0.6; // pivot to centre of mass
    const W3: f64 = 20.0; // spin about the symmetry axis

    let i3 = 0.5 * M * R * R; // a cylinder about its symmetry axis
    let omega = M * GRAV * ARM / (i3 * W3);
    let period = 2.0 * std::f64::consts::PI / omega;

    /* A POINT support: its inertia is the zero matrix, so it cannot turn.
     * A sphere with inverse_mass = 0 would still be free to spin, and the
     * ball joint would drive it. */
    let mut pivot = physical_object::new_point(0, 1.0, Vec3::new(0.0, 0.0, -ARM), Vec3::zeros());
    pivot.set_inverse_mass(0.0);
    pivot.set_inertia_tensor(Mat3::zeros());

    /* Started ON the steady solution, which takes BOTH velocities: the
     * body turns about the vertical through the pivot at Ω *and* spins
     * at ω₃ about its own axis. Each is a rigid motion that leaves the
     * pivot point still, so ġ = J·u = 0 exactly. */
    let top = physical_object::new_from_shape(
        1,
        M,
        0.0,
        Vec3::new(0.0, 0.0, ARM),
        Vec3::new(omega * ARM, 0.0, 0.0),
        Vec3::new(0.0, omega, W3),
        Boundary::Cylinder { radius: R, half_height: 0.06 },
    );
    let mut s = PhysicalObjectSystem::new(vec![pivot, top], 0.0);
    s.uniform_gravity = Vec3::new(0.0, -GRAV, 0.0);
    s.collide_enabled = false;
    s.method = Method::Ida;
    let snap = s.clone();
    s.constraints.add_ball(&snap, 0, 1).unwrap();
    assert_eq!(s.constraints.len(), 3, "a ball joint is three rows");

    // the inertia the formula assumes is the inertia the body actually has
    let izz = s.objects[1].get_inertia_tensor().0[2][2];
    assert!((izz - i3).abs() < 1e-12, "I3 = {izz}, expected {i3}");

    let axis = |s: &PhysicalObjectSystem| {
        s.objects[1].get_orientation().normalize().rotate(Vec3::new(0.0, 0.0, 1.0))
    };
    let start = axis(&s);

    let report = integrate::run(&mut s, period, 200).expect("top run");
    assert_eq!(report.initial_velocity_projected, 0.0, "the start is already steady");
    assert!(report.constraint_drift.0 < 1e-7, "|g| = {:e}", report.constraint_drift.0);

    /* One full precession later, the axis is back where it began. */
    let end = axis(&s);
    let closure = (end - start).norm();
    assert!(
        closure < 5e-3,
        "after one period {period} the axis should close: {start:?} -> {end:?} (|Δ| = {closure:e})"
    );
    /* And it never left the horizontal on the way — steady precession,
     * not nutation. The tilt out of the plane is the y component. */
    assert!(end.y.abs() < 1e-3, "the axis should stay horizontal: y = {}", end.y);
}

/// The chain of `videos/rod_pendulum_chain.html`: four bobs on four
/// rods, and the two facts that make a rod the joint to reach for.
///
/// **It is the cheapest.** A `CONSTRAIN` is one row holding one scalar,
/// `g = |d| − L`. Four rods on four free bobs is 4 rows on 24 freedoms,
/// against 3 rows for a `BALL`, 4 for a `UNIVERSAL`, 5 for a `HINGE`.
///
/// **It is the best conditioned.** One well-scaled scalar equation per
/// joint, with none of the index-2 orientation coupling that forces a
/// tolerance floor on the other three — so a rod-only system is *not*
/// floored, and run continuously at the default `1e-10 / 1e-12` the
/// chain holds to roundoff.
///
/// The recording is a different demand and gets a different tolerance:
/// it is one `step` per frame, and each is a cold restart with a fresh
/// multiplier seed and no BDF history. At the default tolerance these
/// same four rods fail on the *second* restart, which is why the scene
/// asks for the floor values by hand. That is a statement about
/// restarting, not about accuracy; the scene file and grammar §12.9
/// record it.
#[test]
fn a_rod_chain_is_the_cheapest_joint_and_holds_to_roundoff() {
    const L: f64 = 0.4;
    // a zigzag lying in no single plane, so a release from rest is 3-D
    let dirs = [
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.8, 0.0, 0.6),
        Vec3::new(0.6, -0.2, -0.774),
        Vec3::new(0.9, -0.436, 0.0),
    ];
    /* The anchor is a sphere held still by inverse_mass = 0, exactly as
     * the scene builds it — not a zero-inertia point. A rod exerts no
     * torque, so its rotation is decoupled either way, but the DAE is
     * the one the recording actually integrates. */
    let mut anchor = physical_object::new_from_shape(
        0,
        1.0,
        0.0,
        Vec3::zeros(),
        Vec3::zeros(),
        Vec3::zeros(),
        Boundary::Sphere { radius: 0.05 },
    );
    anchor.set_inverse_mass(0.0);
    let mut objs = vec![anchor];
    let mut p = Vec3::zeros();
    for (k, d) in dirs.iter().enumerate() {
        p += d.normalize() * L;
        /* Spheres, matching the recorded scene. A rod constrains only the
         * distance between centres, so the bobs' rotation is entirely
         * free — and a point mass, whose inertia is the zero matrix, is
         * the harder problem here, not the simpler one. */
        objs.push(physical_object::new_from_shape(
            k + 1,
            1.0,
            0.0,
            p,
            Vec3::zeros(),
            Vec3::zeros(),
            Boundary::Sphere { radius: 0.06 },
        ));
    }
    let mut s = PhysicalObjectSystem::new(objs, 0.0);
    s.uniform_gravity = Vec3::new(0.0, -G, 0.0);
    s.collide_enabled = false;
    s.method = Method::Ida;
    let snap = s.clone();
    for k in 0..dirs.len() {
        s.constraints.add_distance(&snap, k, k + 1, None).unwrap();
    }
    assert_eq!(s.constraints.len(), 4, "four rods are one row each");

    /* Released from rest, so ġ = J·u = 0 exactly: nothing to project. */
    let report = integrate::run(&mut s, 5.0, 50).expect("rod chain");
    assert_eq!(report.initial_velocity_projected, 0.0, "released from rest");

    /* A rod needs no tolerance floor, so the default was honoured — and
     * at the default the rods hold to roundoff, which is the claim. */
    assert!(!report.tolerance_floored, "a rod-only chain takes no floor");
    let (g, gdot) = report.constraint_drift;
    assert!(g < 1e-13, "rods should hold to roundoff: |g| = {g:e}");
    assert!(gdot < 1e-11, "|g_dot| = {gdot:e}");

    /* Measured directly, not merely reported: every rod is still 0.4. */
    for k in 0..dirs.len() {
        let d = (s.objects[k + 1].get_position() - s.objects[k].get_position()).norm();
        assert!((d - L).abs() < 1e-13, "rod {k} is {d}, not {L}");
    }
    /* It really did move, and out of any one plane. */
    let tip = dirs.len();
    let moved = (s.objects[tip].get_position() - snap.objects[tip].get_position()).norm();
    assert!(moved > 0.5, "the chain should have swung: {moved}");
    assert!(s.objects[tip].get_position().z.abs() > 1e-3, "not confined to a plane");
}

/// The chain of `videos/ball_joint_chain.html`, and the one thing a
/// ball joint does that a hinge cannot.
///
/// ```text
/// anchor --BALL-- link1 --BALL-- link2 --BALL-- link3 --BALL-- link4
/// ```
///
/// A hinge fixes an axis as well as a point, so a hinged chain is
/// trapped in a plane for ever. A ball joint fixes only the point, so
/// the same chain can leave it — and the sharpest way to show that is
/// not the trajectory but the **start**.
///
/// The chain lies along x and is given the velocity field of a rigid
/// rotation about the vertical, `v = ω × r` with `ω = [0, Ω, 0]`. A
/// rigid motion violates no joint at all, so for ball joints
/// `ġ = J·u = 0` **exactly**. Hand the identical state to the same
/// chain built from hinges about z and `|ġ| = Ω` exactly, because the
/// whirl is precisely the component a hinge about z forbids.
///
/// That is a closed-form number for an inconsistency, which is a
/// stronger check than a tolerance: the residual is not merely small
/// for one and large for the other, it is zero and Ω.
#[test]
fn a_ball_chain_accepts_a_whirl_that_a_hinge_chain_forbids() {
    const OMEGA: f64 = 1.5;
    const H: f64 = 0.25; // link half-length

    /* Links laid end to end, so each joint's midpoint pivot lands on the
     * ends the pair shares. The anchor sits one half-link back, putting
     * the first pivot on the origin rather than inside link1. */
    let build = || {
        let mut objs = vec![world_anchor(0, Vec3::new(-H, 0.0, 0.0))];
        for k in 0..4 {
            let x = H + 2.0 * H * k as f64;
            let mut b = physical_object::new_from_shape(
                k as usize + 1,
                1.0,
                0.0,
                Vec3::new(x, 0.0, 0.0),
                Vec3::new(0.0, 0.0, -OMEGA * x), // ω × r, with ω = [0, Ω, 0]
                Vec3::new(0.0, OMEGA, 0.0),
                Boundary::Cuboid { half_extents: [H, 0.06, 0.06] },
            );
            b.set_angular_velocity(Vec3::new(0.0, OMEGA, 0.0));
            objs.push(b);
        }
        let mut s = PhysicalObjectSystem::new(objs, 0.0);
        s.uniform_gravity = Vec3::new(0.0, -3.0, 0.0);
        s.collide_enabled = false;
        s.method = Method::Ida;
        s
    };

    /* The velocity residual of whatever joint set is installed. */
    let gdot = |s: &PhysicalObjectSystem| {
        let pose = ConstraintSet::poses(s);
        let v: Vec<Vec3> = s.objects.iter().map(|o| o.get_velocity()).collect();
        let w: Vec<Vec3> = s.objects.iter().map(|o| o.get_angular_velocity()).collect();
        let mut out = vec![0.0; s.constraints.len()];
        s.constraints.velocity_residual(&pose, &v, &w, &mut out);
        out.iter().fold(0.0_f64, |m, r| m.max(r.abs()))
    };

    let mut ball = build();
    let snap = ball.clone();
    for k in 0..4 {
        ball.constraints.add_ball(&snap, k, k + 1).unwrap();
    }
    assert_eq!(ball.constraints.len(), 12, "four ball joints are 3 rows each");

    let mut hinge = build();
    let snap = hinge.clone();
    for k in 0..4 {
        hinge.constraints.add_hinge(&snap, k, k + 1, Vec3::new(0.0, 0.0, 1.0)).unwrap();
    }

    // the whole point, and both numbers are exact
    assert_eq!(gdot(&ball), 0.0, "a rigid rotation violates no ball joint");
    assert!(
        (gdot(&hinge) - OMEGA).abs() < 1e-12,
        "a hinge about z forbids exactly the whirl: expected |g_dot| = {OMEGA}, got {}",
        gdot(&hinge)
    );

    /* Every link starts ON the plane z = 0 — though not at rest in z,
     * since the whirl's velocity is exactly what points out of it. */
    for k in 1..=4 {
        assert_eq!(ball.objects[k].get_position().z, 0.0, "starts flat");
    }

    /* Consistent in, consistent out: nothing is projected away. */
    let report = integrate::run(&mut ball, 0.02, 1).expect("chain start");
    assert_eq!(report.initial_velocity_projected, 0.0, "the start needed no projection");

    let mut off_plane = 0.0_f64;
    let mut worst_g = 0.0_f64;
    for k in 2..=250 {
        let r = integrate::run(&mut ball, 0.02 * f64::from(k), 1).expect("chain run");
        worst_g = worst_g.max(r.constraint_drift.0);
        for b in &ball.objects {
            off_plane = off_plane.max(b.get_position().z.abs());
        }
    }
    assert!(worst_g < 1e-7, "the four joints must hold: |g| = {worst_g:e}");
    assert!(off_plane > 1.5, "a hinged chain stays at z = 0; got {off_plane}");
}

/// The drive train of `videos/universal_joint.html`:
///
/// ```text
/// bearing --HINGE-- input --UNIVERSAL-- output --ROD-- post
/// ```
///
/// A universal joint holds one shared point and one right angle between
/// its trunnions. It does **not** hold the two shafts straight, so the
/// bend angle is free and something else has to bound it — here the rod
/// to the post, which bounds it at a value pure geometry can predict.
///
/// The output shaft's centre must stay `0.3` from the cross at
/// `[0.9, 0, 0]` and `0.4243` from the post at `[1.5, 0, -0.3]`, so it
/// rides the circle where those two spheres meet. The shaft therefore
/// sweeps a cone of half-angle `θ` about the cross-to-post line, which is
/// itself `θ` off the x axis, with `θ = atan(0.3/0.6) = 26.565°`. The
/// bend runs from `0` to `2θ = 53.130°` and no further:
///
/// ```text
/// cos 53.130° = 0.6   exactly
/// ```
///
/// That bound is the assertion. It is a closed-form number the integrator
/// is never told, reached only if the hinge, the universal joint and the
/// rod all hold at once.
#[test]
fn a_universal_joint_bends_no_further_than_its_bracing_allows() {
    let shaft = |id: usize, x: f64| {
        physical_object::new_from_shape(
            id,
            1.0,
            0.0,
            Vec3::new(x, 0.0, 0.0),
            Vec3::zeros(),
            Vec3::zeros(),
            Boundary::Cuboid { half_extents: [0.3, 0.09, 0.09] },
        )
    };
    let anchor = |id: usize, p: Vec3| {
        let mut a = physical_object::new_point(id, 1.0, p, Vec3::zeros());
        a.set_inverse_mass(0.0);
        a
    };
    let mut s = PhysicalObjectSystem::new(
        vec![
            anchor(0, Vec3::zeros()),
            shaft(1, 0.6),
            shaft(2, 1.2),
            anchor(3, Vec3::new(1.5, 0.0, -0.3)),
        ],
        0.0,
    );
    s.uniform_gravity = Vec3::new(0.0, -3.0, 0.0);
    s.collide_enabled = false;
    s.method = Method::Ida;
    s.external_torques[1] = Vec3::new(0.03, 0.0, 0.0); // drive, from rest
    let snap = s.clone();
    s.constraints.add_hinge(&snap, 0, 1, Vec3::new(1.0, 0.0, 0.0)).unwrap();
    s.constraints
        .add_universal(&snap, 1, 2, Vec3::new(0.0, 0.0, 1.0), Vec3::new(0.0, 1.0, 0.0))
        .unwrap();
    s.constraints.add_distance(&snap, 2, 3, None).unwrap();
    /* 5 + 4 + 1 = 10 rows on two free bodies. Bracing the output shaft
     * with a second HINGE instead would be 14 rows on those same 12
     * freedoms — rank-deficient, and IDA fails at t = 0. The rod is the
     * one-row support that bounds the bend without over-constraining. */
    assert_eq!(s.constraints.len(), 10, "hinge 5 + universal 4 + rod 1");

    let axis = |s: &PhysicalObjectSystem, i: usize| {
        s.objects[i].get_orientation().normalize().rotate(Vec3::new(1.0, 0.0, 0.0))
    };
    let (mut worst_g, mut flattest, mut sharpest) = (0.0_f64, -1.0_f64, 1.0_f64);
    // advanced one frame at a time, exactly as the recorder drives it
    for k in 1..=180 {
        let report = integrate::run(&mut s, 0.025 * f64::from(k), 1).expect("driveshaft run");
        worst_g = worst_g.max(report.constraint_drift.0);
        let c = axis(&s, 1).dot(axis(&s, 2));
        flattest = flattest.max(c);
        sharpest = sharpest.min(c);
    }

    assert!(worst_g < 1e-5, "the three joints must hold: |g| = {worst_g:e}");
    /* The bend never passes the geometric bound, and does reach it — so
     * the rod is genuinely what stops the shaft, not a short run. */
    assert!(sharpest > 0.6 - 1e-4, "bent past cos 53.130° = 0.6: {sharpest}");
    assert!(sharpest < 0.6 + 1e-4, "never reached the bound: {sharpest}");
    assert!(flattest > 1.0 - 1e-4, "must come back straight: {flattest}");
    /* Rotation really is being transmitted: both shafts turn, and about
     * their own axes, not merely swinging as a pendulum would. */
    let spin = |i: usize| s.objects[i].get_angular_velocity().dot(axis(&s, i));
    assert!(spin(1) > 5.0 && spin(2) > 5.0, "in {} out {}", spin(1), spin(2));
}

/// Orientation joints are integrated at a tolerance floor, because the
/// index-2 system cannot deliver more — and the report says when the
/// floor was applied rather than silently changing what was asked for.
#[test]
fn an_orientation_joint_reports_its_tolerance_floor() {
    let (mut s, _) = compound_pendulum([0.2, 0.4, 0.2], 0.5, 0.02);
    s.rtol = 1.0e-12; // tighter than the DAE can hold
    let report = integrate::run(&mut s, 0.5, 20).expect("hinge run");
    assert!(report.tolerance_floored, "the floor should have been applied and reported");

    // a ROD-only system has no such limit and is not floored
    let a = physical_object::new_point(0, 1.0, Vec3::zeros(), Vec3::zeros());
    let b = physical_object::new_point(1, 1.0, Vec3::new(1.0, 0.0, 0.0), Vec3::zeros());
    let mut r = PhysicalObjectSystem::new(vec![a, b], 0.0);
    r.collide_enabled = false;
    r.method = Method::Ida;
    r.rtol = 1.0e-12;
    let snap = r.clone();
    r.constraints.add_distance(&snap, 0, 1, None).unwrap();
    let rr = integrate::run(&mut r, 0.5, 20).expect("rod run");
    assert!(!rr.tolerance_floored, "a rod needs no floor");
}

/// A body may be **already turning** when a run starts. A joint that
/// grips orientation constrains velocity as well as position — a ball
/// joint says `v + ω×r` is shared — so a body spinning about an offset
/// pivot must have its centre moving. Giving it `ω` and leaving `v` at
/// zero puts the state OFF the constraint manifold, and the run projects
/// it back on before integrating, reporting how much it moved.
#[test]
fn a_spinning_body_is_projected_onto_the_constraint_manifold() {
    let bx = physical_object::new_from_shape(
        1,
        1.0,
        0.0,
        Vec3::new(0.6, -0.8, 0.0),
        Vec3::zeros(),               // centre at rest …
        Vec3::new(0.0, 3.0, 0.0),    // … but spinning hard
        Boundary::Cuboid { half_extents: [0.2, 0.2, 0.2] },
    );
    let mut s = PhysicalObjectSystem::new(vec![world_anchor(0, Vec3::zeros()), bx], 0.0);
    s.uniform_gravity = Vec3::new(0.0, -G, 0.0);
    s.collide_enabled = false;
    s.method = Method::Ida;
    let snap = s.clone();
    s.constraints.add_ball(&snap, 0, 1).unwrap();

    // the state handed in genuinely violates the velocity constraint
    let (_, gdot_before) = s.constraints.drift(&s);
    assert!(gdot_before > 1e-3, "the test should start inconsistent: {gdot_before:e}");

    let report = integrate::run(&mut s, 1.0, 40).expect("a spinning body on a ball joint");
    assert!(
        report.initial_velocity_projected > 1e-3,
        "the projection should have been needed and reported: {}",
        report.initial_velocity_projected
    );
    /* 3 rad/s is the fastest case in this file and it runs at the
     * orientation-joint tolerance floor, so the bound is looser than the
     * at-rest cases (which hold |g| to 1e-11). Measured: 1.3e-7. */
    assert!(report.constraint_drift.0 < 1e-6, "|g| = {:e}", report.constraint_drift.0);
    assert!(report.constraint_drift.1 < 1e-5, "|g_dot| = {:e}", report.constraint_drift.1);
    /* What the projection actually did: the turn is nearly untouched and
     * the CENTRE was set moving instead. That is the correct reading of
     * "smallest mass-weighted change" here — the pivot was running at
     * |ω × r| = 1.5 m/s and something had to absorb it, and giving a
     * 1 kg body some velocity is cheaper than fighting a 3 rad/s turn.
     * The physical picture is a coupling clutched onto a spinning shaft:
     * the shaft keeps turning and the housing starts to move. */
    let w_after = ::physical_object::constrain::angular_velocity(&s.objects[1]).norm();
    assert!(w_after > 2.0, "the turn should largely survive: |ω| = {w_after}");
    assert!(
        report.initial_velocity_projected > 0.1 && report.initial_velocity_projected < 10.0,
        "the correction should be of order the pivot speed: {}",
        report.initial_velocity_projected
    );
}

/// …and spin **about the arm** costs nothing, because it does not move
/// the shared point at all. Same body, same speed, axis rotated onto the
/// arm: no projection, and the body keeps every bit of its turn.
#[test]
fn spin_about_the_joint_arm_needs_no_projection() {
    let arm = Vec3::new(0.3, -0.4, 0.0).normalize();
    let bx = physical_object::new_from_shape(
        1,
        1.0,
        0.0,
        Vec3::new(0.6, -0.8, 0.0),
        Vec3::zeros(),
        3.0 * arm, // turning about the line through the pivot
        Boundary::Cuboid { half_extents: [0.2, 0.2, 0.2] },
    );
    let mut s = PhysicalObjectSystem::new(vec![world_anchor(0, Vec3::zeros()), bx], 0.0);
    s.uniform_gravity = Vec3::new(0.0, -G, 0.0);
    s.collide_enabled = false;
    s.method = Method::Ida;
    let snap = s.clone();
    s.constraints.add_ball(&snap, 0, 1).unwrap();

    // ω × r = 0, so the state is already ON the manifold
    let (_, gdot) = s.constraints.drift(&s);
    assert!(gdot < 1e-15, "spin along the arm moves nothing: {gdot:e}");

    let report = integrate::run(&mut s, 1.0, 40).expect("ball run");
    assert_eq!(report.initial_velocity_projected, 0.0, "nothing to project");
    assert!(report.constraint_drift.0 < 1e-6, "|g| = {:e}", report.constraint_drift.0);
    /* Angular VELOCITY, not momentum: this cube's inertia is 0.0267, so
     * turning at 3 rad/s is only |L| = 0.08. */
    let w = ::physical_object::constrain::angular_velocity(&s.objects[1]);
    assert!(w.norm() > 2.5, "the turn is free and must survive: |ω| = {}", w.norm());
}

/// A state that is already consistent is left **exactly** alone — the
/// projection must not perturb the common case.
#[test]
fn a_consistent_start_is_not_projected() {
    let (mut s, _) = compound_pendulum([0.2, 0.4, 0.2], 0.5, 0.1);
    let report = integrate::run(&mut s, 0.3, 10).expect("hinge run");
    assert_eq!(report.initial_velocity_projected, 0.0);

    let mut r = pendulum(0.3, 1.0);
    r.objects[1].set_inertia_tensor(::physical_object::linalg::Mat3::identity());
    r.objects[1].set_angular_momentum(Vec3::new(0.0, 0.7, 0.0));
    let rr = integrate::run(&mut r, 0.5, 10).expect("a rod carries a spinning body");
    /* A rod has no angular Jacobian, so spin never enters its ġ and the
     * state was consistent all along — this is exactly why rods never
     * revealed the missing projection. */
    assert_eq!(rr.initial_velocity_projected, 0.0);
}

/// EQUILIBRIUM and SENSITIVITY solve for positions only, so an
/// orientation joint is refused by name rather than quietly solving a
/// different problem.
#[test]
fn the_translational_solvers_refuse_orientation_joints() {
    let (mut s, _) = compound_pendulum([0.2, 0.4, 0.2], 0.5, 0.3);
    let e = equilibrium::solve(&mut s).unwrap_err();
    assert!(e.contains("positions only"), "{e}");
    assert!(e.contains("hinge"), "the message should name the joint: {e}");

    let e = sensitivity::run(&mut s, 1.0, &[SensParam::Gravity(1)]).unwrap_err();
    assert!(e.contains("grips orientation"), "{e}");
}
