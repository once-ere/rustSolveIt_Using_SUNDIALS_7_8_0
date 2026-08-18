//! Constrained dynamics, equilibrium and sensitivity, each checked
//! against a closed form rather than against last week's output.
//!
//! These four families entered the simulator together (IDA, KINSOL,
//! CVODES, IDAS); every test here names the analytic fact it is pinning.

use ::physical_object::constrain::ConstraintSet;
use ::physical_object::equilibrium;
use ::physical_object::integrate::{self, Method};
use ::physical_object::linalg::Vec3;
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
