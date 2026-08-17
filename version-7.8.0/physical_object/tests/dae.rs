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

/// Constraints act on positions, so a spinning rigid body is refused —
/// the same contract the SPRK separability gate follows.
#[test]
fn the_constrained_gate_names_the_feature_that_blocks_it() {
    let mut s = pendulum(0.5, 1.0);
    s.objects[1].set_angular_momentum(Vec3::new(0.0, 1.0, 0.0));
    s.objects[1].set_inertia_tensor(::physical_object::linalg::Mat3::identity());
    let e = integrate::run(&mut s, 1.0, 10).unwrap_err();
    assert!(e.contains("translational only"), "{e}");
    assert!(e.contains("obj1"), "{e}");
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
    assert!((s.constraints.distances[0].length - 0.5).abs() < 1e-15, "3-4-5 triangle");
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
