//! All numerical time integration for the simulator.
//!
//! **Every** integration in this crate goes through the pure-Rust
//! `sundials_rs` solvers — no hand-rolled Euler/Verlet steppers survive
//! from the legacy code:
//!
//! - [`Method::Adams`] / [`Method::Bdf`] — CVODE (Adams-Moulton or BDF)
//!   with Newton iteration and the dense linear solver
//!   (difference-quotient Jacobian), following the driving pattern of
//!   `cvode_rs/examples/solar_system.rs`. Integrates the full 13N state
//!   `[pos, momentum, quaternion, angular momentum]` per object.
//! - [`Method::Sprk`] — ARKODE SPRKStep symplectic partitioned
//!   Runge-Kutta with a fixed step, following
//!   `arkode_rs/examples/ark_kepler.rs`. Only valid for separable
//!   systems (point masses, no magnetic coupling, no torques); the
//!   legacy `GravitationalSystem::step` velocity-Verlet corresponds to
//!   `ARKODE_SPRK_LEAPFROG_2_2`.
//!
//! CVODE/ARKODE callbacks are plain `fn` pointers, so the right-hand
//! sides cannot borrow the system: all parameters are cloned into the
//! solver's `user_data` (`Option<Box<dyn Any>>`) and downcast inside
//! the callback (a failed downcast returns the unrecoverable flag `-1`).
//!
//! # SUNDIALS 7.8.0 handle model
//!
//! The vendored engine is the pure-Rust translation of **SUNDIALS
//! 7.8.0**, which models the C API's opaque pointers faithfully:
//!
//! - `N_Vector` is `Rc<_generic_N_Vector>` with `RefCell` content, not a
//!   struct with a public `data: Vec<f64>` field. Element access goes
//!   through [`N_VGetArrayPointer`], which hands back a `RefMut` guard
//!   over the payload — the Rust stand-in for C's `sunrealtype*`. A
//!   guard is a live borrow: it **must** be dropped before any solver
//!   entry point touches the same vector, which is why every access
//!   below sits in its own block (see [`with_data`] / [`with_data_mut`]).
//! - `CVodeMem` and `ARKodeMem` are `Rc<RefCell<…>>` handles, so every
//!   entry point takes `&CVodeMem` / `&ARKodeMem` (shared, C-style
//!   "pass the pointer") rather than `&mut`.
//! - Constructors return `Option<…>` where C returns a possibly-`NULL`
//!   pointer. Each `None` is reported here as a named `Err`, never
//!   unwrapped — a missing symbol or a failed allocation must say which
//!   call produced it (CLAUDE.md hard rule 5).
//! - Destructors take ownership (`N_VDestroy(v)`, `SUNMatDestroy(A)`) or
//!   an `&mut Option<…>` they blank (`CVodeFree`, `ARKodeFree`,
//!   `SUNContext_Free`), mirroring C's `free(p); p = NULL;`.
//!
//! Callback signatures follow the 7.8.0 types verbatim:
//! `CVRhsFn = fn(sunrealtype, &N_Vector, &N_Vector, &mut Option<Box<dyn
//! Any>>) -> i32` — note `ydot` is a **shared** reference whose content
//! is reached through its own `RefMut` guard.

use crate::boundary::{self, Boundary};
use crate::collide;
use crate::constrain::{Anchors, ConstraintSet, Pose};
use crate::linalg::{Mat3, Quat, Vec3};
use crate::physical_object::physical_object;
use crate::system::{PhysicalObjectSystem, VARS_PER_OBJECT};

use std::any::Any;

use sundials_core::nvector_serial::N_VNew_Serial;
use sundials_core::sundials_context::{SUNContext, SUNContext_Create, SUNContext_Free};
use sundials_core::sundials_linearsolver::SUNLinSolFree;
use sundials_core::sundials_matrix::SUNMatDestroy;
use sundials_core::sundials_nvector::{N_VDestroy, N_VGetArrayPointer, N_Vector};
use sundials_core::sundials_types::{sunrealtype, SUN_COMM_NULL};
use sundials_core::sunlinsol_dense::SUNLinSol_Dense;
use sundials_core::sunmatrix_dense::SUNDenseMatrix;

use cvode_rs::cvode::{
    CVode, CVodeCreate, CVodeFree, CVodeInit, CVodeReInit, CVodeRootInit, CVodeSStolerances,
};
use cvode_rs::cvode_impl::{
    CVodeMem, CV_ADAMS, CV_BDF, CV_NORMAL, CV_ROOT_RETURN, CV_SUCCESS,
};
use cvode_rs::cvode_io::{
    CVodeGetNumErrTestFails, CVodeGetNumGEvals, CVodeGetNumNonlinSolvIters, CVodeGetNumRhsEvals,
    CVodeGetNumSteps, CVodeGetRootInfo, CVodeSetMaxNumSteps, CVodeSetMaxStep,
    CVodeSetNoInactiveRootWarn, CVodeSetRootDirection, CVodeSetUserData,
};
use cvode_rs::cvode_ls::CVodeSetLinearSolver;

use arkode_rs::arkode::{ARKodeEvolve, ARKodeFree, ARKodeReset};
use arkode_rs::arkode_impl::{ARK_NORMAL, ARK_ROOT_RETURN};
use arkode_rs::arkode_io::{
    ARKodeGetRootInfo, ARKodeSetFixedStep, ARKodeSetMaxNumSteps, ARKodeSetRootDirection,
    ARKodeSetUserData,
};
use arkode_rs::arkode_root::ARKodeRootInit;
use arkode_rs::arkode_sprkstep::SPRKStepCreate;
use arkode_rs::arkode_sprkstep_io::SPRKStepSetMethodName;

use ida_rs::ida::{IDACreate, IDAFree, IDAInit, IDASStolerances, IDASolve};
use ida_rs::ida_impl::{IDA_NORMAL, IDA_SUCCESS};
use ida_rs::ida_io::{
    IDAGetNumErrTestFails, IDAGetNumNonlinSolvIters, IDAGetNumResEvals, IDAGetNumSteps,
    IDASetId, IDASetMaxNumSteps, IDASetSuppressAlg, IDASetUserData,
};
use ida_rs::ida_ls::IDASetLinearSolver;

/// The 7.8.0 user-data slot: exactly the callback parameter type of
/// `CVRhsFn`/`CVRootFn`/`ARKRhsFn`/`ARKRootFn`.
type UserData = Option<Box<dyn Any>>;

/// Reads an `N_Vector`'s payload through a `RefMut` guard and drops the
/// guard before returning — the only safe way to touch vector data
/// between solver calls.
///
/// Returns `None` when the vector is not a serial vector with an array
/// pointer (7.8.0 `N_VGetArrayPointer` returns `NULL` for those).
fn with_data<R>(v: &N_Vector, f: impl FnOnce(&[f64]) -> R) -> Option<R> {
    let d = N_VGetArrayPointer(v)?;
    Some(f(&d))
}

/// Mutable counterpart of [`with_data`].
fn with_data_mut<R>(v: &N_Vector, f: impl FnOnce(&mut [f64]) -> R) -> Option<R> {
    let mut d = N_VGetArrayPointer(v)?;
    Some(f(&mut d))
}

/// The `N_VGetArrayPointer` failure message used on every access path.
fn no_array(what: &str) -> String {
    format!("N_VGetArrayPointer returned NULL for {what} (not a serial N_Vector)")
}

/// Tolerance floor for a DAE whose joints grip **orientation**.
///
/// The GGL system is index 2: the multiplier `μ` enters the *kinematic*
/// equations, so the local error of `q̇` and `Q̇` carries an O(h) term
/// that `IDASetSuppressAlg` cannot remove — it suppresses the
/// multipliers themselves, not their trace in the differential
/// variables. With a rod that trace is small; with a hinge, `μ` has five
/// components acting on both `v` and `ω`, and `ω` drives the quaternion.
///
/// Measured across twelve compound pendulums — four inertias × three
/// release angles — the boundary is sharp and uniform: `1e-6` converges
/// in every one, holding `|g|` to between `1e-11` and `1e-9`, and `1e-8`
/// converges in none. Tightening past the floor buys no accuracy and
/// costs the run.
///
/// (Before the initial velocities were projected onto the constraint
/// manifold this boundary was *erratic* in the release angle, which is
/// what first suggested a conditioning problem. It was not: it was an
/// inconsistent starting state. Projecting made the boundary uniform —
/// but did not remove it, because the index-2 accuracy ceiling is real.)
///
/// So the floor is applied, and [`RunReport::tolerance_floored`] says it
/// was, rather than silently changing what the caller asked for.
const ROT_JOINT_RTOL_FLOOR: f64 = 1.0e-6;
const ROT_JOINT_ATOL_FLOOR: f64 = 1.0e-8;

/// Quaternion-norm drift beyond which the packed state is renormalized
/// and CVODE re-initialized (re-init discards the multistep history, so
/// it is only done when actually needed).
const QUAT_RENORM_TOL: f64 = 1.0e-10;

/// Integration method — every variant is a sundials_rs solver.
#[derive(Clone, Debug, PartialEq)]
pub enum Method {
    /// CVODE Adams-Moulton (non-stiff default).
    Adams,
    /// CVODE BDF (stiff problems, e.g. fast magnetic gyration).
    Bdf,
    /// ARKODE SPRKStep symplectic method with fixed step `dt`;
    /// `table` is an ARKODE SPRK table name such as
    /// `"ARKODE_SPRK_MCLACHLAN_4_4"` or `"ARKODE_SPRK_LEAPFROG_2_2"`.
    Sprk { table: String, dt: f64 },
    /// IDA on the GGL-stabilized index-2 DAE — the only method that can
    /// honour a [`crate::constrain::ConstraintSet`]. Translational only
    /// (`ConstraintSet::gate`). With no constraints it is an ordinary
    /// BDF integration of the same translational dynamics, which is
    /// exactly how the unconstrained cross-check in the tests works.
    Ida,
}

/// Conserved-quantity snapshot recorded at each output time.
#[derive(Clone, Debug)]
pub struct Snapshot {
    pub t: f64,
    pub energy: f64,
    pub total_momentum: Vec3,
    pub total_angular_momentum: Vec3,
    pub center_of_mass: Vec3,
}

/// Result of a solver run: per-output snapshots plus solver statistics.
#[derive(Clone, Debug, Default)]
pub struct RunReport {
    pub snapshots: Vec<Snapshot>,
    /// Internal solver steps (CVODE reports its adaptive count; the
    /// SPRK path computes this from the fixed step count).
    pub nst: i64,
    /// Right-hand-side evaluations (CVODE paths only — the SPRK path
    /// leaves this 0 even though it evaluates forces every stage).
    pub nfe: i64,
    /// Nonlinear (Newton) iterations (CVODE paths only).
    pub nni: i64,
    /// Local error test failures (CVODE paths only).
    pub netf: i64,
    /// Root-function (pairwise-separation) evaluations — nonzero only
    /// when collision rootfinding was armed.
    pub nge: i64,
    /// Collision impulses resolved during this run.
    pub ncollisions: u64,
    /// Worst `(|g|, |ġ|)` over the constraint set at the final state —
    /// how far the rods actually stretched. Zero-length when the system
    /// is unconstrained. The GGL formulation drives BOTH to roundoff;
    /// a growing first component means the constraint is drifting and
    /// the answer is quietly wrong (see `constrain.rs`).
    pub constraint_drift: (f64, f64),
    /// The run asked for a tighter tolerance than an orientation-gripping
    /// DAE can deliver, and it was raised to the floor
    /// (`ROT_JOINT_RTOL_FLOOR`). Only ever true on the IDA path with a
    /// BALL/HINGE/UNIVERSAL joint.
    pub tolerance_floored: bool,
    /// Largest velocity change the run had to make to put the starting
    /// state ON the constraint manifold — see
    /// [`project_initial_velocities`]. Zero when the caller's velocities
    /// were already consistent, which is the usual case.
    pub initial_velocity_projected: f64,
}

/// Parameters snapshot handed to the CVODE right-hand side.
#[derive(Clone, Debug)]
struct RhsParams {
    n: usize,
    g: f64,
    softening: f64,
    uniform_gravity: Vec3,
    e_field: Vec3,
    b_field: Vec3,
    masses: Vec<f64>,
    inverse_masses: Vec<f64>,
    charges: Vec<f64>,
    inverse_inertia: Vec<Mat3>,
    magnetic: Vec<Mat3>,
    ext_force: Vec<Vec3>,
    ext_torque: Vec<Vec3>,
    /// Collidable pairs, in root-function component order (empty when
    /// collision rootfinding is not armed).
    pairs: Vec<(usize, usize)>,
    /// Boundary of every object (the root function needs geometry;
    /// poses come from the packed state `y`).
    boundaries: Vec<Boundary>,
}

impl RhsParams {
    fn from_system(s: &PhysicalObjectSystem) -> Self {
        Self {
            pairs: Vec::new(),
            boundaries: s.objects.iter().map(|o| o.get_boundary()).collect(),
            n: s.objects.len(),
            g: s.g_constant,
            softening: s.softening,
            uniform_gravity: s.uniform_gravity,
            e_field: s.e_field,
            b_field: s.b_field,
            masses: s.objects.iter().map(|o| o.get_mass()).collect(),
            inverse_masses: s.objects.iter().map(|o| o.get_inverse_mass()).collect(),
            charges: s.objects.iter().map(|o| o.get_charge()).collect(),
            inverse_inertia: s.objects.iter().map(|o| o.get_inverse_inertia_tensor()).collect(),
            magnetic: s.objects.iter().map(|o| o.get_magnetic_moment_tensor()).collect(),
            ext_force: s.external_forces.clone(),
            ext_torque: s.external_torques.clone(),
        }
    }
}

fn read_vec3(d: &[f64], at: usize) -> Vec3 {
    Vec3::new(d[at], d[at + 1], d[at + 2])
}

fn write_vec3(d: &mut [f64], at: usize, v: Vec3) {
    d[at] = v.x;
    d[at + 1] = v.y;
    d[at + 2] = v.z;
}

/// Full 13N right-hand side:
/// `dq/dt = p m⁻¹`;
/// `dp/dt = Σ G m_i m_j Δ/(|Δ|²+ε²)^{3/2} + m g + qE + q v×B + F_ext`;
/// `dq̂/dt = ½ (0, w) ⊗ q̂` with `w = (R I⁻¹ Rᵀ) L`;
/// `dL/dt = τ_ext + (R M Rᵀ) B`.
fn rhs_full(_t: sunrealtype, y: &N_Vector, ydot: &N_Vector, user_data: &mut UserData) -> i32 {
    let params = match user_data.as_mut().and_then(|b| b.downcast_mut::<RhsParams>()) {
        Some(p) => p,
        None => return -1,
    };
    /* Two live `RefMut` guards: `y` and `ydot` are distinct handles on
     * every CVODE call path, so the borrows never collide. */
    let d = match N_VGetArrayPointer(y) {
        Some(d) => d,
        None => return -1,
    };
    let mut out_guard = match N_VGetArrayPointer(ydot) {
        Some(d) => d,
        None => return -1,
    };
    let d = &d[..];
    let out = &mut out_guard[..];
    let n = params.n;

    for i in 0..n {
        let b = VARS_PER_OBJECT * i;
        let pos_i = read_vec3(d, b);
        let mom_i = read_vec3(d, b + 3);
        let quat_i = Quat::new(d[b + 6], d[b + 7], d[b + 8], d[b + 9]);
        let ang_i = read_vec3(d, b + 10);

        let v_i = mom_i * params.inverse_masses[i];

        /* dq/dt = v */
        write_vec3(out, b, v_i);

        /* dp/dt: softened pairwise gravity (donor arithmetic order) ... */
        let mut force = Vec3::zeros();
        for j in 0..n {
            if i == j {
                continue;
            }
            let bj = VARS_PER_OBJECT * j;
            let r_vec = read_vec3(d, bj) - pos_i;
            let dist_sq = r_vec.norm_squared() + params.softening * params.softening;
            let dist = dist_sq.sqrt();
            force += (params.g * params.masses[i] * params.masses[j] / (dist_sq * dist)) * r_vec;
        }
        /* ... + uniform gravity + Lorentz + external */
        force += params.masses[i] * params.uniform_gravity;
        force += params.charges[i] * (params.e_field + v_i.cross(params.b_field));
        force += params.ext_force[i];
        write_vec3(out, b + 3, force);

        /* dq̂/dt = ½ (0, w) ⊗ q̂ ; R from the normalized copy */
        let r = quat_i.normalize().to_rotation_matrix();
        let omega = r * params.inverse_inertia[i] * r.transpose() * ang_i;
        let qdot = (Quat::pure(omega) * quat_i) * 0.5;
        out[b + 6] = qdot.w;
        out[b + 7] = qdot.x;
        out[b + 8] = qdot.y;
        out[b + 9] = qdot.z;

        /* dL/dt = τ_ext + (R M Rᵀ) B */
        let torque = params.ext_torque[i] + r * params.magnetic[i] * r.transpose() * params.b_field;
        write_vec3(out, b + 10, torque);
    }
    0
}

/// CVODE root function for collision detection: `gout[k]` is the signed
/// separation of collidable pair `k` (positive = separated), computed
/// from the packed state `y` with exactly the same geometry as
/// [`collide::pair_separation`]. A downward zero crossing is a contact
/// event; CVODE interpolates the state onto the root — the precise
/// time of impact.
fn g_contacts(
    _t: sunrealtype,
    y: &N_Vector,
    gout: &mut [sunrealtype],
    user_data: &mut UserData,
) -> i32 {
    let params = match user_data.as_mut().and_then(|b| b.downcast_mut::<RhsParams>()) {
        Some(p) => p,
        None => return -1,
    };
    let d = match N_VGetArrayPointer(y) {
        Some(d) => d,
        None => return -1,
    };
    let d = &d[..];
    for (k, &(i, j)) in params.pairs.iter().enumerate() {
        let bi = VARS_PER_OBJECT * i;
        let bj = VARS_PER_OBJECT * j;
        let pos_i = read_vec3(d, bi);
        let quat_i = Quat::new(d[bi + 6], d[bi + 7], d[bi + 8], d[bi + 9]).normalize();
        let pos_j = read_vec3(d, bj);
        let quat_j = Quat::new(d[bj + 6], d[bj + 7], d[bj + 8], d[bj + 9]).normalize();
        gout[k] = collide::separation_at(
            &params.boundaries[i],
            pos_i,
            quat_i,
            &params.boundaries[j],
            pos_j,
            quat_j,
        );
    }
    0
}

/// Max absolute row sum (the ∞-norm) of a 3×3 matrix — a cheap upper
/// bound on how much the matrix can stretch a vector.
fn mat_inf_norm(m: &Mat3) -> f64 {
    let a = m.0;
    let row = |r: usize| a[r][0].abs() + a[r][1].abs() + a[r][2].abs();
    row(0).max(row(1)).max(row(2))
}

/// Anti-tunneling step cap while collision rootfinding is armed: the
/// root function is only sampled at CVODE's internal steps, so a step
/// must not be able to carry one body clear through another. Cap:
/// smallest positive feature size among paired bodies over twice the
/// largest achievable pairwise surface speed. `span` bounds the cap
/// itself; `growth` is the horizon until the next refresh (an output
/// interval), over which speeds are allowed to GROW:
///
/// - linear speed can grow by acceleration from pairwise gravity (at
///   the current configuration), uniform gravity, the E field and
///   external forces — so a body released FROM REST above a thin plate
///   still gets a finite cap (magnetic forces are ⟂ v and never grow
///   speed);
/// - angular speed is bounded via `|ω| = |R I⁻¹ Rᵀ L| ≤ √3‖I⁻¹‖∞·|L|`
///   with `|L|` allowed to grow by external + magnetic torque — this
///   covers torque-free tumbling exactly (L is conserved, the polhode
///   can still spike |ω| up to |L|/I_min mid-interval).
///
/// Returns 0.0 (= no cap in CVODE semantics) only when nothing can
/// move at all.
fn collision_hmax(
    system: &PhysicalObjectSystem,
    pairs: &[(usize, usize)],
    span: f64,
    growth: f64,
) -> f64 {
    let feature = |o: &physical_object| -> f64 {
        match o.get_boundary() {
            Boundary::Point => f64::INFINITY,
            Boundary::Sphere { radius } => radius,
            Boundary::Cuboid { half_extents } => {
                half_extents[0].min(half_extents[1]).min(half_extents[2])
            }
            // Thinnest crossable feature: the torus tube, the cylinder's
            // smaller of radius/half-height. The ideal disk has zero
            // thickness, so the cap comes from the radius of whatever
            // ball approaches it (the pairwise min picks the smaller).
            Boundary::Torus { tube_radius, .. } => tube_radius,
            Boundary::Disk { radius } => radius,
            Boundary::Cylinder { radius, half_height } => radius.min(half_height),
            Boundary::Dumbbell { r1, r2, rod_radius, .. } => r1.min(r2).min(rod_radius),
        }
    };
    let growth = growth.max(0.0);
    let grav = system.compute_accelerations();
    let e_field = system.e_field.norm();
    let b_field = system.b_field.norm();
    // Achievable linear-speed growth of body k over `growth` seconds.
    let accel = |k: usize| -> f64 {
        let o = &system.objects[k];
        grav[k].norm()
            + system.uniform_gravity.norm()
            + (o.get_charge().abs() * e_field + system.external_forces[k].norm())
                * o.get_inverse_mass()
    };
    // Achievable |ω| of body k over `growth` seconds (see doc above).
    let omega_bound = |k: usize| -> f64 {
        let o = &system.objects[k];
        let inv_inertia = mat_inf_norm(&o.get_inverse_inertia_tensor());
        if inv_inertia == 0.0 {
            return 0.0; // cannot rotate (points, static walls)
        }
        let torque = system.external_torques[k].norm()
            + mat_inf_norm(&o.get_magnetic_moment_tensor()) * b_field;
        3.0f64.sqrt() * inv_inertia * (o.get_angular_momentum().norm() + growth * torque)
    };
    let mut min_feature = f64::INFINITY;
    let mut vmax = 0.0f64;
    for &(i, j) in pairs {
        let a = &system.objects[i];
        let b = &system.objects[j];
        min_feature = min_feature.min(feature(a).min(feature(b)));
        // Fastest surface-point approach: relative center speed (plus
        // what the accelerations can add before the next refresh) plus
        // each body's spin bound times its bounding radius (a rotating
        // edge can sweep into contact without the centers moving).
        let rel = (a.get_velocity() - b.get_velocity()).norm()
            + (accel(i) + accel(j)) * growth;
        let spin = omega_bound(i) * boundary::bounding_radius(&a.get_boundary())
            + omega_bound(j) * boundary::bounding_radius(&b.get_boundary());
        vmax = vmax.max(rel + spin);
    }
    if !(min_feature.is_finite() && vmax > 0.0) {
        return 0.0;
    }
    (min_feature / (2.0 * vmax)).clamp(1e-12, span.max(1e-12))
}

fn snapshot(system: &PhysicalObjectSystem, t: f64) -> Snapshot {
    Snapshot {
        t,
        energy: system.total_energy(),
        total_momentum: system.total_momentum(),
        total_angular_momentum: system.total_angular_momentum(Vec3::zeros()),
        center_of_mass: system.center_of_mass(),
    }
}

/// Renormalizes every quaternion block in a packed state vector;
/// returns the worst norm deviation seen.
fn renormalize_quats(y: &mut [f64], n: usize) -> f64 {
    let mut worst = 0.0f64;
    for i in 0..n {
        let b = VARS_PER_OBJECT * i;
        let q = Quat::new(y[b + 6], y[b + 7], y[b + 8], y[b + 9]);
        worst = worst.max((q.norm() - 1.0).abs());
        let qn = q.normalize();
        y[b + 6] = qn.w;
        y[b + 7] = qn.x;
        y[b + 8] = qn.y;
        y[b + 9] = qn.z;
    }
    worst
}

/// Integrates `system` from its current `time` to `t_end` with the
/// configured [`Method`], recording `nout` evenly spaced outputs.
/// Object states and `system.time` are updated in place.
pub fn run(
    system: &mut PhysicalObjectSystem,
    t_end: f64,
    nout: usize,
) -> Result<RunReport, String> {
    system.contacts.clear();
    /* A constraint is a promise about the trajectory. Only the DAE path
     * can keep it, so an explicit method that cannot is refused by name
     * rather than silently integrating the unconstrained system. */
    if !system.constraints.is_empty() && !matches!(system.method, Method::Ida) {
        return Err(format!(
            "this system has {} rigid constraint(s), which only the DAE integrator can hold: \
             use METHOD IDA (or remove them with CONSTRAIN OFF). The current method is {:?}",
            system.constraints.len(),
            system.method
        ));
    }
    match system.method.clone() {
        Method::Adams => run_cvode(system, t_end, nout, CV_ADAMS),
        Method::Bdf => run_cvode(system, t_end, nout, CV_BDF),
        Method::Sprk { table, dt } => run_sprk(system, t_end, nout, &table, dt),
        Method::Ida => run_ida(system, t_end, nout),
    }
}

/// Advances the system by a single interval `dt` (one output).
pub fn step(system: &mut PhysicalObjectSystem, dt: f64) -> Result<RunReport, String> {
    let t_end = system.time + dt;
    run(system, t_end, 1)
}

/// CVODE path (Adams or BDF + Newton + dense DQ Jacobian) over the full
/// 13N state — the `solar_system.rs` driving pattern.
fn run_cvode(
    system: &mut PhysicalObjectSystem,
    t_end: f64,
    nout: usize,
    lmm: i32,
) -> Result<RunReport, String> {
    let t0 = system.time;
    if t_end <= t0 {
        return Err(format!("t_end ({t_end}) must be greater than current time ({t0})"));
    }
    let nout = nout.max(1);
    let n = system.objects.len();
    if n == 0 {
        system.time = t_end;
        return Ok(RunReport::default());
    }
    let neq = system.state_len();

    /* 7.8.0 context creation is the C two-step: an out-parameter plus a
     * SUNErrCode, not a returned handle. */
    let mut sunctx_out: Option<SUNContext> = None;
    let retval_ctx = SUNContext_Create(SUN_COMM_NULL, &mut sunctx_out);
    if retval_ctx != 0 {
        return Err(format!("SUNContext_Create failed: {retval_ctx}"));
    }
    let sunctx = sunctx_out.ok_or_else(|| "SUNContext_Create returned NULL".to_string())?;

    let y = N_VNew_Serial(neq as i64, &sunctx)
        .ok_or_else(|| format!("N_VNew_Serial({neq}) returned NULL"))?;
    with_data_mut(&y, |d| d.copy_from_slice(&system.pack_state()))
        .ok_or_else(|| no_array("y"))?;

    let cvode_mem = CVodeCreate(lmm, &sunctx)
        .ok_or_else(|| format!("CVodeCreate(lmm = {lmm}) returned NULL"))?;

    let mut retval = CVodeInit(&cvode_mem, rhs_full, t0, &y);
    if retval != CV_SUCCESS {
        return Err(format!("CVodeInit failed: {retval}"));
    }
    retval = CVodeSStolerances(&cvode_mem, system.rtol, system.atol);
    if retval != CV_SUCCESS {
        return Err(format!("CVodeSStolerances failed: {retval}"));
    }
    let a = SUNDenseMatrix(neq as i64, neq as i64, &sunctx)
        .ok_or_else(|| format!("SUNDenseMatrix({neq}, {neq}) returned NULL"))?;
    let ls = SUNLinSol_Dense(&y, &a, &sunctx)
        .ok_or_else(|| "SUNLinSol_Dense returned NULL".to_string())?;
    retval = CVodeSetLinearSolver(&cvode_mem, &ls, Some(&a));
    if retval != CV_SUCCESS {
        return Err(format!("CVodeSetLinearSolver failed: {retval}"));
    }
    retval = CVodeSetMaxNumSteps(&cvode_mem, 500_000);
    if retval != CV_SUCCESS {
        return Err(format!("CVodeSetMaxNumSteps failed: {retval}"));
    }
    /* Collision event detection (ARCHITECTURE.md §3.8): arm sundials
     * rootfinding on the pairwise signed separations. With roots armed
     * but never firing, CVODE's step selection is untouched (the root
     * check runs after each completed internal step, on the
     * interpolant only), so systems with no collidable pairs take
     * exactly the historical code path. */
    let pairs = collide::collidable_pairs(system);
    let armed = system.collide_enabled && !pairs.is_empty();
    let mut params = RhsParams::from_system(system);
    if armed {
        params.pairs = pairs.clone();
    }
    retval = CVodeSetUserData(&cvode_mem, Some(Box::new(params)));
    if retval != CV_SUCCESS {
        return Err(format!("CVodeSetUserData failed: {retval}"));
    }
    if armed {
        retval = CVodeRootInit(&cvode_mem, pairs.len() as i32, Some(g_contacts));
        if retval != CV_SUCCESS {
            return Err(format!("CVodeRootInit failed: {retval}"));
        }
        /* Only downward crossings (approach) are contact events. */
        let dirs = vec![-1i32; pairs.len()];
        retval = CVodeSetRootDirection(&cvode_mem, &dirs);
        if retval != CV_SUCCESS {
            return Err(format!("CVodeSetRootDirection failed: {retval}"));
        }
        retval = CVodeSetNoInactiveRootWarn(&cvode_mem);
        if retval != CV_SUCCESS {
            return Err(format!("CVodeSetNoInactiveRootWarn failed: {retval}"));
        }
        let hmax = collision_hmax(system, &pairs, t_end - t0, (t_end - t0) / nout.max(1) as f64);
        retval = CVodeSetMaxStep(&cvode_mem, hmax);
        if retval != CV_SUCCESS {
            return Err(format!("CVodeSetMaxStep failed: {retval}"));
        }
    }

    let mut report = RunReport::default();
    let accumulate_stats = |mem: &CVodeMem, rep: &mut RunReport| {
        let (mut nst, mut nfe, mut nni, mut netf, mut nge) = (0i64, 0i64, 0i64, 0i64, 0i64);
        CVodeGetNumSteps(mem, &mut nst);
        CVodeGetNumRhsEvals(mem, &mut nfe);
        CVodeGetNumNonlinSolvIters(mem, &mut nni);
        CVodeGetNumErrTestFails(mem, &mut netf);
        CVodeGetNumGEvals(mem, &mut nge);
        rep.nst += nst;
        rep.nfe += nfe;
        rep.nni += nni;
        rep.netf += netf;
        rep.nge += nge;
    };

    let mut t = t0;
    let span = t_end - t0;
    let mut roots_armed = armed;
    /* Zeno burst state. This lives across output intervals on purpose:
     * counting per interval is what made the physics depend on how
     * often output was requested. */
    let mut burst = 0usize;
    let mut last_event_t = f64::NEG_INFINITY;
    for k in 1..=nout {
        let tout = t0 + span * (k as f64) / (nout as f64);

        /* Re-arm rootfinding if the Zeno guard disarmed it during the
         * previous output interval. */
        if armed && !roots_armed {
            retval = CVodeRootInit(&cvode_mem, pairs.len() as i32, Some(g_contacts));
            if retval != CV_SUCCESS {
                return Err(format!("CVodeRootInit (re-arm) failed: {retval}"));
            }
            let dirs = vec![-1i32; pairs.len()];
            let r = CVodeSetRootDirection(&cvode_mem, &dirs);
            if r != CV_SUCCESS {
                return Err(format!("CVodeSetRootDirection (re-arm) failed: {r}"));
            }
            let r = CVodeSetNoInactiveRootWarn(&cvode_mem);
            if r != CV_SUCCESS {
                return Err(format!("CVodeSetNoInactiveRootWarn (re-arm) failed: {r}"));
            }
            roots_armed = true;
        }

        /* Refresh the anti-tunneling cap at every interval start: a cap
         * computed at an event near the previous tout must not go stale
         * (velocities may also have changed since arm time). */
        if roots_armed {
            let hmax = collision_hmax(system, &pairs, t_end - t, tout - t);
            let r = CVodeSetMaxStep(&cvode_mem, hmax);
            if r != CV_SUCCESS {
                return Err(format!("CVodeSetMaxStep (interval) failed: {r}"));
            }
        }

        /* Event loop: integrate toward tout; every CV_ROOT_RETURN is a
         * contact event at the interpolated time of impact — resolve
         * impulses, re-initialize, continue toward the same tout
         * (the cvRocket_dns.rs pattern). */
        loop {
            retval = CVode(&cvode_mem, tout, &y, &mut t, CV_NORMAL);
            if retval < 0 {
                return Err(format!("CVode failed with retval = {retval} at t = {t}"));
            }
            if retval != CV_ROOT_RETURN {
                break; // CV_SUCCESS: tout reached
            }
            let mut roots = vec![0i32; pairs.len()];
            let r = CVodeGetRootInfo(&cvode_mem, &mut roots);
            if r != CV_SUCCESS {
                return Err(format!("CVodeGetRootInfo failed: {r}"));
            }
            with_data(&y, |d| system.unpack_state(d)).ok_or_else(|| no_array("y"))?;
            system.time = t;
            let flagged: Vec<bool> = roots.iter().map(|ri| *ri != 0).collect();
            /* Zeno accounting is by BURST, not by output interval: an
             * event that follows the previous one after a real flight
             * starts the count again, however many have happened since
             * the last snapshot. */
            if collide::same_burst(t, last_event_t) {
                burst += 1;
            } else {
                burst = 1;
            }
            last_event_t = t;
            let force_plastic = burst > collide::MAX_EVENTS_IN_BURST;
            let contacts = collide::resolve_impulses(system, &pairs, &flagged, force_plastic)?;
            report.ncollisions += contacts.len() as u64;
            system.collision_count += contacts.len() as u64;
            collide::record_contacts(system, contacts);

            if burst > 2 * collide::MAX_EVENTS_IN_BURST && roots_armed {
                /* Zeno guard tier 2: chattering contact — project out
                 * any penetration and disarm rootfinding for the rest
                 * of this output interval. */
                let extra = collide::resolve_penetrations(system, true)?;
                report.ncollisions += extra.len() as u64;
                system.collision_count += extra.len() as u64;
                collide::record_contacts(system, extra);
                let r = CVodeRootInit(&cvode_mem, 0, None);
                if r != CV_SUCCESS {
                    return Err(format!("CVodeRootInit (disarm) failed: {r}"));
                }
                roots_armed = false;
            }

            with_data_mut(&y, |d| d.copy_from_slice(&system.pack_state()))
                .ok_or_else(|| no_array("y"))?;
            accumulate_stats(&cvode_mem, &mut report);
            let r = CVodeReInit(&cvode_mem, t, &y);
            if r != CV_SUCCESS {
                return Err(format!("CVodeReInit failed: {r}"));
            }
            if roots_armed {
                /* Velocities changed: refresh the anti-tunneling cap.
                 * The clamp span must be the REMAINING RUN (t_end − t,
                 * as at arm time), never the remaining output interval:
                 * an event landing exactly on tout would collapse
                 * tout − t to 0, pin hmax at the 1e-12 clamp floor, and
                 * starve every later interval into CV_TOO_MUCH_WORK. */
                let hmax = collision_hmax(system, &pairs, t_end - t, tout - t);
                let r = CVodeSetMaxStep(&cvode_mem, hmax);
                if r != CV_SUCCESS {
                    return Err(format!("CVodeSetMaxStep failed: {r}"));
                }
            }
            /* An event can land (numerically) on tout itself — e.g. a
             * plastic pair that keeps grazing contact. The state is
             * already at tout; asking CVODE to integrate the remaining
             * zero-length interval would fail with CV_TOO_CLOSE. */
            if (tout - t).abs() <= 1e-12 * tout.abs().max(1.0) {
                break;
            }
        }

        /* Renormalize quaternion drift; a renormalization mutates y, so
         * the multistep history must be re-initialized (accumulating
         * stats first — CVodeReInit zeroes the counters). Rootfinding
         * stays armed across CVodeReInit. */
        let mut y_check = with_data(&y, |d| d.to_vec()).ok_or_else(|| no_array("y"))?;
        let drift = renormalize_quats(&mut y_check, n);
        if drift > QUAT_RENORM_TOL {
            with_data_mut(&y, |d| d.copy_from_slice(&y_check)).ok_or_else(|| no_array("y"))?;
            accumulate_stats(&cvode_mem, &mut report);
            let r = CVodeReInit(&cvode_mem, t, &y);
            if r != CV_SUCCESS {
                return Err(format!("CVodeReInit failed: {r}"));
            }
        }

        with_data(&y, |d| system.unpack_state(d)).ok_or_else(|| no_array("y"))?;
        system.time = t;

        /* End-of-interval safety net: deep initial overlaps and
         * Zeno-disarmed intervals can leave real penetration behind —
         * detect it read-only first so the common (clean) case does
         * not perturb the solver state at all. */
        if armed {
            let mut needs_sweep = false;
            for &(i, j) in &pairs {
                let a = &system.objects[i];
                let b = &system.objects[j];
                if collide::aabb_overlap(a, b, system.contact_slop)
                    && collide::pair_separation(a, b) < -system.contact_slop
                {
                    needs_sweep = true;
                    break;
                }
            }
            if needs_sweep {
                let extra = collide::resolve_penetrations(system, false)?;
                report.ncollisions += extra.len() as u64;
                system.collision_count += extra.len() as u64;
                collide::record_contacts(system, extra);
                with_data_mut(&y, |d| d.copy_from_slice(&system.pack_state()))
                    .ok_or_else(|| no_array("y"))?;
                accumulate_stats(&cvode_mem, &mut report);
                let r = CVodeReInit(&cvode_mem, t, &y);
                if r != CV_SUCCESS {
                    return Err(format!("CVodeReInit failed: {r}"));
                }
            }
        }

        report.snapshots.push(snapshot(system, t));
    }

    accumulate_stats(&cvode_mem, &mut report);

    /* Teardown in the C example order: integrator, linear solver,
     * matrix, vectors, context. 7.8.0 destructors either take ownership
     * or blank an `Option`, so each handle is consumed exactly once. */
    let mut cvode_mem = Some(cvode_mem);
    CVodeFree(&mut cvode_mem);
    SUNLinSolFree(Some(ls));
    SUNMatDestroy(a);
    N_VDestroy(y);
    let mut sunctx = Some(sunctx);
    SUNContext_Free(&mut sunctx);
    Ok(report)
}

/// Parameters snapshot for the separable (SPRK) right-hand sides.
#[derive(Clone, Debug)]
struct SprkParams {
    n: usize,
    g: f64,
    softening: f64,
    uniform_gravity: Vec3,
    e_field: Vec3,
    masses: Vec<f64>,
    inverse_masses: Vec<f64>,
    charges: Vec<f64>,
    ext_force: Vec<Vec3>,
    /// Collidable pairs (root-function order; empty when unarmed).
    pairs: Vec<(usize, usize)>,
    /// Boundary of every object.
    boundaries: Vec<Boundary>,
    /// Orientation snapshot — SPRK bodies cannot spin (separability
    /// gate), so orientations are constant over the run.
    orientations: Vec<Quat>,
}

/// ARKODE root function for the SPRK `[q(3N) | p(3N)]` layout: signed
/// pairwise separations, with orientations from the (constant)
/// snapshot in the params.
fn g_contacts_sprk(
    _t: sunrealtype,
    y: &N_Vector,
    gout: &mut [sunrealtype],
    user_data: &mut UserData,
) -> i32 {
    let params = match user_data.as_mut().and_then(|b| b.downcast_mut::<SprkParams>()) {
        Some(p) => p,
        None => return -1,
    };
    let d = match N_VGetArrayPointer(y) {
        Some(d) => d,
        None => return -1,
    };
    let d = &d[..];
    for (k, &(i, j)) in params.pairs.iter().enumerate() {
        gout[k] = collide::separation_at(
            &params.boundaries[i],
            read_vec3(d, 3 * i),
            params.orientations[i],
            &params.boundaries[j],
            read_vec3(d, 3 * j),
            params.orientations[j],
        );
    }
    0
}

/// SPRK force RHS (`f1`): writes `dp/dt` into the second half of
/// `[q(3N) | p(3N)]` (the `ark_kepler.rs` layout).
fn sprk_force(_t: sunrealtype, y: &N_Vector, ydot: &N_Vector, user_data: &mut UserData) -> i32 {
    let params = match user_data.as_mut().and_then(|b| b.downcast_mut::<SprkParams>()) {
        Some(p) => p,
        None => return -1,
    };
    let n = params.n;
    let d = match N_VGetArrayPointer(y) {
        Some(d) => d,
        None => return -1,
    };
    let mut out_guard = match N_VGetArrayPointer(ydot) {
        Some(d) => d,
        None => return -1,
    };
    let d = &d[..];
    let out = &mut out_guard[..];
    for i in 0..n {
        let pos_i = read_vec3(d, 3 * i);
        let mut force = Vec3::zeros();
        for j in 0..n {
            if i == j {
                continue;
            }
            let r_vec = read_vec3(d, 3 * j) - pos_i;
            let dist_sq = r_vec.norm_squared() + params.softening * params.softening;
            let dist = dist_sq.sqrt();
            force += (params.g * params.masses[i] * params.masses[j] / (dist_sq * dist)) * r_vec;
        }
        force += params.masses[i] * params.uniform_gravity;
        force += params.charges[i] * params.e_field;
        force += params.ext_force[i];
        write_vec3(out, 3 * (n + i), force);
    }
    0
}

/// SPRK velocity RHS (`f2`): writes `dq/dt = p m⁻¹` into the first half.
fn sprk_velocity(_t: sunrealtype, y: &N_Vector, ydot: &N_Vector, user_data: &mut UserData) -> i32 {
    let params = match user_data.as_mut().and_then(|b| b.downcast_mut::<SprkParams>()) {
        Some(p) => p,
        None => return -1,
    };
    let n = params.n;
    let d = match N_VGetArrayPointer(y) {
        Some(d) => d,
        None => return -1,
    };
    let mut out_guard = match N_VGetArrayPointer(ydot) {
        Some(d) => d,
        None => return -1,
    };
    let d = &d[..];
    let out = &mut out_guard[..];
    for i in 0..n {
        let mom_i = read_vec3(d, 3 * (n + i));
        write_vec3(out, 3 * i, mom_i * params.inverse_masses[i]);
    }
    0
}

/// ARKODE SPRKStep path — symplectic, fixed step. Requires a separable
/// Hamiltonian: point-mass translational dynamics only.
fn run_sprk(
    system: &mut PhysicalObjectSystem,
    t_end: f64,
    nout: usize,
    table: &str,
    dt: f64,
) -> Result<RunReport, String> {
    let t0 = system.time;
    if t_end <= t0 {
        return Err(format!("t_end ({t_end}) must be greater than current time ({t0})"));
    }
    // Written as `<=` plus an explicit NaN check rather than `!(dt >
    // 0.0)`: the negated form silently relies on NaN comparing false,
    // which is correct but invisible to a reader.
    if dt.is_nan() || dt <= 0.0 {
        return Err(format!("SPRK requires a positive fixed step dt (got {dt})"));
    }
    let nout = nout.max(1);
    let n = system.objects.len();
    if n == 0 {
        system.time = t_end;
        return Ok(RunReport::default());
    }

    /* Separability gate: no velocity-dependent forces, no rotational
     * dynamics. Report exactly which feature blocks SPRK. */
    if system.b_field != Vec3::zeros() {
        return Err("SPRK method requires a separable Hamiltonian: magnetic field B must be zero \
                    (the Lorentz force q v x B is velocity-dependent); use METHOD ADAMS or BDF"
            .to_string());
    }
    for (i, o) in system.objects.iter().enumerate() {
        if o.get_inverse_inertia_tensor() != Mat3::zeros()
            && o.get_angular_momentum() != Vec3::zeros()
        {
            return Err(format!(
                "SPRK method integrates translational dynamics only: object {i} has spinning \
                 rigid-body state (nonzero angular momentum and invertible inertia tensor); \
                 use METHOD ADAMS or BDF"
            ));
        }
        if o.get_magnetic_moment_tensor() != Mat3::zeros() {
            return Err(format!(
                "SPRK method requires zero magnetic moment tensor (object {i}); \
                 use METHOD ADAMS or BDF"
            ));
        }
    }
    for (i, tq) in system.external_torques.iter().enumerate() {
        if *tq != Vec3::zeros() {
            return Err(format!(
                "SPRK method cannot apply external torques (object {i}); use METHOD ADAMS or BDF"
            ));
        }
    }

    let mut sunctx_out: Option<SUNContext> = None;
    let retval_ctx = SUNContext_Create(SUN_COMM_NULL, &mut sunctx_out);
    if retval_ctx != 0 {
        return Err(format!("SUNContext_Create failed: {retval_ctx}"));
    }
    let sunctx = sunctx_out.ok_or_else(|| "SUNContext_Create returned NULL".to_string())?;

    let neq = 6 * n;
    let y = N_VNew_Serial(neq as i64, &sunctx)
        .ok_or_else(|| format!("N_VNew_Serial({neq}) returned NULL"))?;
    with_data_mut(&y, |d| {
        for (i, o) in system.objects.iter().enumerate() {
            write_vec3(d, 3 * i, o.get_position());
            write_vec3(d, 3 * (n + i), o.get_momentum());
        }
    })
    .ok_or_else(|| no_array("y"))?;

    /* 7.8.0 `SPRKStepCreate` takes the two right-hand sides by value
     * (`ARKRhsFn`, not `Option<ARKRhsFn>`) — C's mandatory f1/f2. */
    let am = SPRKStepCreate(sprk_force, sprk_velocity, t0, &y, &sunctx)
        .ok_or_else(|| "SPRKStepCreate returned NULL".to_string())?;

    let mut retval = SPRKStepSetMethodName(&am, table);
    if retval < 0 {
        return Err(format!("SPRKStepSetMethodName({table:?}) failed: {retval}"));
    }
    retval = ARKodeSetFixedStep(&am, dt);
    if retval < 0 {
        return Err(format!("ARKodeSetFixedStep failed: {retval}"));
    }
    let max_steps = (((t_end - t0) / dt).ceil() as i64 + 16).max(1000) * 2;
    retval = ARKodeSetMaxNumSteps(&am, max_steps);
    if retval < 0 {
        return Err(format!("ARKodeSetMaxNumSteps failed: {retval}"));
    }
    /* Collision events on the SPRK path: same design as CVODE, but the
     * root check samples at the fixed step dt, so the anti-tunneling
     * bound is the user's own dt (documented). */
    let pairs = collide::collidable_pairs(system);
    let armed = system.collide_enabled && !pairs.is_empty();
    let mut params = SprkParams {
        n,
        g: system.g_constant,
        softening: system.softening,
        uniform_gravity: system.uniform_gravity,
        e_field: system.e_field,
        masses: system.objects.iter().map(|o| o.get_mass()).collect(),
        inverse_masses: system.objects.iter().map(|o| o.get_inverse_mass()).collect(),
        charges: system.objects.iter().map(|o| o.get_charge()).collect(),
        ext_force: system.external_forces.clone(),
        pairs: Vec::new(),
        boundaries: system.objects.iter().map(|o| o.get_boundary()).collect(),
        orientations: system.objects.iter().map(|o| o.get_orientation()).collect(),
    };
    if armed {
        params.pairs = pairs.clone();
    }
    retval = ARKodeSetUserData(&am, Some(Box::new(params)));
    if retval < 0 {
        return Err(format!("ARKodeSetUserData failed: {retval}"));
    }
    if armed {
        retval = ARKodeRootInit(&am, pairs.len() as i32, Some(g_contacts_sprk));
        if retval < 0 {
            return Err(format!("ARKodeRootInit failed: {retval}"));
        }
        let dirs = vec![-1i32; pairs.len()];
        retval = ARKodeSetRootDirection(&am, &dirs);
        if retval < 0 {
            return Err(format!("ARKodeSetRootDirection failed: {retval}"));
        }
    }

    let mut report = RunReport::default();
    let mut t = t0;
    let span = t_end - t0;
    let mut roots_armed = armed;
    /* Zeno burst state. This lives across output intervals on purpose:
     * counting per interval is what made the physics depend on how
     * often output was requested. */
    let mut burst = 0usize;
    let mut last_event_t = f64::NEG_INFINITY;
    /* These take the raw payload slice, not the `N_Vector`: the caller
     * owns the `RefMut` guard (via `with_data`/`with_data_mut`) and drops
     * it before the next solver call. */
    let write_back = |system: &mut PhysicalObjectSystem, d: &[f64]| {
        let n = system.objects.len();
        for (i, o) in system.objects.iter_mut().enumerate() {
            o.set_position(read_vec3(d, 3 * i));
            o.set_momentum(read_vec3(d, 3 * (n + i)));
        }
    };
    let repack = |system: &PhysicalObjectSystem, d: &mut [f64]| {
        let n = system.objects.len();
        for (i, o) in system.objects.iter().enumerate() {
            write_vec3(d, 3 * i, o.get_position());
            write_vec3(d, 3 * (n + i), o.get_momentum());
        }
    };
    for k in 1..=nout {
        let tout = t0 + span * (k as f64) / (nout as f64);
        if armed && !roots_armed {
            retval = ARKodeRootInit(&am, pairs.len() as i32, Some(g_contacts_sprk));
            if retval < 0 {
                return Err(format!("ARKodeRootInit (re-arm) failed: {retval}"));
            }
            let dirs = vec![-1i32; pairs.len()];
            let r = ARKodeSetRootDirection(&am, &dirs);
            if r < 0 {
                return Err(format!("ARKodeSetRootDirection (re-arm) failed: {r}"));
            }
            roots_armed = true;
        }
        loop {
            retval = ARKodeEvolve(&am, tout, &y, &mut t, ARK_NORMAL);
            if retval < 0 {
                return Err(format!("ARKodeEvolve failed with retval = {retval} at t = {t}"));
            }
            if retval != ARK_ROOT_RETURN {
                break;
            }
            let mut roots = vec![0i32; pairs.len()];
            let r = ARKodeGetRootInfo(&am, &mut roots);
            if r < 0 {
                return Err(format!("ARKodeGetRootInfo failed: {r}"));
            }
            with_data(&y, |d| write_back(system, d)).ok_or_else(|| no_array("y"))?;
            system.time = t;
            let flagged: Vec<bool> = roots.iter().map(|ri| *ri != 0).collect();
            if collide::same_burst(t, last_event_t) {
                burst += 1;
            } else {
                burst = 1;
            }
            last_event_t = t;
            let force_plastic = burst > collide::MAX_EVENTS_IN_BURST;
            let contacts = collide::resolve_impulses(system, &pairs, &flagged, force_plastic)?;
            report.ncollisions += contacts.len() as u64;
            system.collision_count += contacts.len() as u64;
            collide::record_contacts(system, contacts);
            if burst > 2 * collide::MAX_EVENTS_IN_BURST && roots_armed {
                let extra = collide::resolve_penetrations(system, true)?;
                report.ncollisions += extra.len() as u64;
                system.collision_count += extra.len() as u64;
                collide::record_contacts(system, extra);
                let r = ARKodeRootInit(&am, 0, None);
                if r < 0 {
                    return Err(format!("ARKodeRootInit (disarm) failed: {r}"));
                }
                roots_armed = false;
            }
            with_data_mut(&y, |d| repack(system, d)).ok_or_else(|| no_array("y"))?;
            let r = ARKodeReset(&am, t, &y);
            if r < 0 {
                return Err(format!("ARKodeReset failed: {r}"));
            }
            if (tout - t).abs() <= 1e-12 * tout.abs().max(1.0) {
                break;
            }
        }
        with_data(&y, |d| write_back(system, d)).ok_or_else(|| no_array("y"))?;
        system.time = t;
        if armed {
            let mut needs_sweep = false;
            for &(i, j) in &pairs {
                let a = &system.objects[i];
                let b = &system.objects[j];
                if collide::aabb_overlap(a, b, system.contact_slop)
                    && collide::pair_separation(a, b) < -system.contact_slop
                {
                    needs_sweep = true;
                    break;
                }
            }
            if needs_sweep {
                let extra = collide::resolve_penetrations(system, false)?;
                report.ncollisions += extra.len() as u64;
                system.collision_count += extra.len() as u64;
                collide::record_contacts(system, extra);
                with_data_mut(&y, |d| repack(system, d)).ok_or_else(|| no_array("y"))?;
                let r = ARKodeReset(&am, t, &y);
                if r < 0 {
                    return Err(format!("ARKodeReset failed: {r}"));
                }
            }
        }
        report.snapshots.push(snapshot(system, t));
    }
    report.nst = ((t - t0) / dt).round() as i64;

    let mut slot = Some(am);
    ARKodeFree(&mut slot);
    N_VDestroy(y);
    let mut sunctx = Some(sunctx);
    SUNContext_Free(&mut sunctx);
    Ok(report)
}

/// Propagates a single object under constant external force and torque
/// for `dt` — the sundials-backed replacement for the legacy
/// `RigidBody::integrate` / `RigidBody3D::integrate` Euler steppers.
pub fn propagate_single(
    obj: &mut physical_object,
    force: Vec3,
    torque: Vec3,
    dt: f64,
) -> Result<(), String> {
    let mut sys = PhysicalObjectSystem::new(vec![obj.clone()], 0.0);
    sys.external_forces[0] = force;
    sys.external_torques[0] = torque;
    sys.method = Method::Adams;
    step(&mut sys, dt)?;
    *obj = sys.objects.remove(0);
    Ok(())
}

/*=================================================================*/
/* Constrained dynamics: the GGL index-2 DAE, driven by ida_rs      */
/*=================================================================*/

/// Parameters snapshot for the DAE residual.
///
/// Unlike the CVODE path this carries the *whole* rigid-body problem —
/// the DAE state is the same 13-per-object packing `system.rs` defines,
/// so joints can grip orientation as well as position.
#[derive(Clone, Debug)]
struct DaeParams {
    n: usize,
    m: usize,
    g: f64,
    softening: f64,
    uniform_gravity: Vec3,
    e_field: Vec3,
    b_field: Vec3,
    masses: Vec<f64>,
    inverse_masses: Vec<f64>,
    charges: Vec<f64>,
    inverse_inertia: Vec<Mat3>,
    magnetic: Vec<Mat3>,
    ext_force: Vec<Vec3>,
    ext_torque: Vec<Vec3>,
    anchors: Anchors,
    constraints: ConstraintSet,
}

impl DaeParams {
    fn from_system(s: &PhysicalObjectSystem) -> Self {
        Self {
            n: s.objects.len(),
            m: s.constraints.len(),
            g: s.g_constant,
            softening: s.softening,
            uniform_gravity: s.uniform_gravity,
            e_field: s.e_field,
            b_field: s.b_field,
            masses: s.objects.iter().map(|o| o.get_mass()).collect(),
            inverse_masses: s.objects.iter().map(|o| o.get_inverse_mass()).collect(),
            charges: s.objects.iter().map(|o| o.get_charge()).collect(),
            inverse_inertia: s.objects.iter().map(|o| o.get_inverse_inertia_tensor()).collect(),
            magnetic: s.objects.iter().map(|o| o.get_magnetic_moment_tensor()).collect(),
            ext_force: s.external_forces.clone(),
            ext_torque: s.external_torques.clone(),
            anchors: Anchors::of(s),
            constraints: s.constraints.clone(),
        }
    }

    /// Poses, linear and angular velocities read out of a packed 13N
    /// state — the three things the constraint algebra works in.
    fn kinematics(&self, y: &[f64]) -> (Vec<Pose>, Vec<Vec3>, Vec<Vec3>) {
        let mut pose = Vec::with_capacity(self.n);
        let mut v = Vec::with_capacity(self.n);
        let mut w = Vec::with_capacity(self.n);
        for i in 0..self.n {
            let b = VARS_PER_OBJECT * i;
            let q = Quat::new(y[b + 6], y[b + 7], y[b + 8], y[b + 9]).normalize();
            let r = q.to_rotation_matrix();
            pose.push(Pose { position: read_vec3(y, b), orientation: q });
            v.push(read_vec3(y, b + 3) * self.inverse_masses[i]);
            w.push(r * self.inverse_inertia[i] * r.transpose() * read_vec3(y, b + 10));
        }
        (pose, v, w)
    }

    /// Applied force and torque on every body — the same expressions, in
    /// the same arithmetic order, as [`rhs_full`] (hard rule 3).
    fn applied(&self, y: &[f64], pose: &[Pose], v: &[Vec3]) -> (Vec<Vec3>, Vec<Vec3>) {
        let mut force = vec![Vec3::zeros(); self.n];
        let mut torque = vec![Vec3::zeros(); self.n];
        for i in 0..self.n {
            let pos_i = pose[i].position;
            let mut f = Vec3::zeros();
            for j in 0..self.n {
                if i == j {
                    continue;
                }
                let r_vec = pose[j].position - pos_i;
                let dist_sq = r_vec.norm_squared() + self.softening * self.softening;
                let dist = dist_sq.sqrt();
                f += (self.g * self.masses[i] * self.masses[j] / (dist_sq * dist)) * r_vec;
            }
            f += self.masses[i] * self.uniform_gravity;
            f += self.charges[i] * (self.e_field + v[i].cross(self.b_field));
            f += self.ext_force[i];
            force[i] = f;

            let r = pose[i].orientation.to_rotation_matrix();
            torque[i] = self.ext_torque[i] + r * self.magnetic[i] * r.transpose() * self.b_field;
            let _ = y;
        }
        (force, torque)
    }
}

/// The GGL-stabilized index-2 residual `F(t, y, ẏ) = 0` over the full
/// rigid state `y = [pos, momentum, quat, angmom]ⁿ ⧺ λ(m) ⧺ μ(m)`:
///
/// ```text
///   0 = q̇   - (v - J_vᵀμ)              (position, GGL-projected)
///   0 = Q̇   - ½(0, ω - J_ωᵀμ) ⊗ Q      (orientation, likewise)
///   0 = ṗ   - F + J_vᵀλ                 (linear momentum balance)
///   0 = L̇   - τ + J_ωᵀλ                 (angular momentum balance)
///   0 = g(q, Q)                          (the joints)
///   0 = J·u                              (and their rates)
/// ```
///
/// Two things are worth pointing at. First, the multipliers act on
/// **velocity** — `J_vᵀμ` corrects `v`, `J_ωᵀμ` corrects `ω` *before* it
/// drives the quaternion. That is the GGL projection expressed in the
/// chart the constraint Jacobian is already written in, which is what
/// lets one formulation cover rods and hinges alike.
///
/// Second, carrying **both** `g` and `ġ` as algebraic equations is what
/// pins them at roundoff. Plain index-1 (acceleration-level) constraints
/// satisfy only the second and let `g` drift quadratically — and nothing
/// fails loudly when it does.
///
/// An anchored body gets `0 = ṗ` (or `0 = L̇`) instead of its balance,
/// so it never moves and absorbs any reaction.
fn dae_residual(
    _t: sunrealtype,
    yy: &N_Vector,
    yp: &N_Vector,
    rr: &N_Vector,
    user_data: &mut UserData,
) -> i32 {
    let p = match user_data.as_mut().and_then(|b| b.downcast_mut::<DaeParams>()) {
        Some(p) => p,
        None => return -1,
    };
    let y = match N_VGetArrayPointer(yy) {
        Some(d) => d,
        None => return -1,
    };
    let ypv = match N_VGetArrayPointer(yp) {
        Some(d) => d,
        None => return -1,
    };
    let mut rg = match N_VGetArrayPointer(rr) {
        Some(d) => d,
        None => return -1,
    };
    let (n, m) = (p.n, p.m);
    let (y, ypv, r) = (&y[..], &ypv[..], &mut rg[..]);
    let base = VARS_PER_OBJECT * n;

    let (pose, v, w) = p.kinematics(y);
    let (force, torque) = p.applied(y, &pose, &v);

    /* Jᵀλ is a wrench — a force and a torque per body — and goes
     * straight into the momentum balances.
     *
     * Jᵀμ is a wrench too, and that is the point: the GGL projection is
     * `q̇ = v - M⁻¹Jᵀμ`, NOT `v - Jᵀμ`. The mass metric is not optional
     * bookkeeping. `J_v` is dimensionless but `J_ω` carries the
     * attachment arm, so `J_ωᵀμ` has units of length × μ; subtracting it
     * from an angular velocity is dimensionally wrong by a factor of
     * length². For translation alone `M⁻¹` is a single scalar and
     * omitting it merely rescales μ, which is why rods never noticed —
     * add a hinge and the same omission breaks the integration outright
     * (a body spinning at 1e-3 rad/s was enough). */
    let (mut fc, mut tc) = (vec![Vec3::zeros(); n], vec![Vec3::zeros(); n]);
    let (mut dv, mut dw) = (vec![Vec3::zeros(); n], vec![Vec3::zeros(); n]);
    if m > 0 {
        let lam = &y[base..base + m];
        let mu = &y[base + m..base + 2 * m];
        p.constraints.add_jacobian_transpose(&pose, &p.anchors, lam, &mut fc, &mut tc);
        p.constraints.add_jacobian_transpose(&pose, &p.anchors, mu, &mut dv, &mut dw);
        for i in 0..n {
            dv[i] = dv[i] * p.inverse_masses[i];
            let r = pose[i].orientation.to_rotation_matrix();
            dw[i] = r * p.inverse_inertia[i] * r.transpose() * dw[i];
        }
    }

    for i in 0..n {
        let b = VARS_PER_OBJECT * i;
        /* --- translation --- */
        if p.anchors.translation_fixed[i] {
            for k in 0..3 {
                r[b + k] = ypv[b + k];
                r[b + 3 + k] = ypv[b + 3 + k];
            }
        } else {
            write_vec3_from(r, b, |k| ypv[b + k] - (v[i] - dv[i]).to_array()[k]);
            write_vec3_from(r, b + 3, |k| ypv[b + 3 + k] - force[i].to_array()[k] + fc[i].to_array()[k]);
        }
        /* --- rotation --- */
        if p.anchors.rotation_fixed[i] {
            for k in 0..4 {
                r[b + 6 + k] = ypv[b + 6 + k];
            }
            for k in 0..3 {
                r[b + 10 + k] = ypv[b + 10 + k];
            }
        } else {
            let qdot = (Quat::pure(w[i] - dw[i]) * pose[i].orientation) * 0.5;
            r[b + 6] = ypv[b + 6] - qdot.w;
            r[b + 7] = ypv[b + 7] - qdot.x;
            r[b + 8] = ypv[b + 8] - qdot.y;
            r[b + 9] = ypv[b + 9] - qdot.z;
            write_vec3_from(r, b + 10, |k| {
                ypv[b + 10 + k] - torque[i].to_array()[k] + tc[i].to_array()[k]
            });
        }
    }

    /* --- 0 = g(q, Q) and 0 = J·u --- */
    if m > 0 {
        let (_, tail) = r.split_at_mut(base);
        let (gblk, gdblk) = tail.split_at_mut(m);
        p.constraints.residual(&pose, gblk);
        p.constraints.velocity_residual(&pose, &v, &w, gdblk);
    }
    0
}

fn write_vec3_from(d: &mut [f64], at: usize, f: impl Fn(usize) -> f64) {
    for k in 0..3 {
        d[at + k] = f(k);
    }
}

/// Solves for the Lagrange multipliers at the starting configuration —
/// the forces and torques the joints must carry so that `g̈ = 0`.
///
/// **This is not an optimisation, it is required.** The GGL system
/// carries `g` and `ġ` as equations but *not* `g̈`, so at an instant
/// where every body is at rest, `ġ = 0` holds no matter what the
/// accelerations are: free fall satisfies the residual exactly, with
/// `λ = 0`. `IDACalcIC` therefore has nothing to solve and leaves the
/// derivative in free fall — after which BDF spends its first step
/// discovering that a hinge is attached, and the step size collapses to
/// `1e-15`. Differentiating once more is what pins the accelerations:
///
/// ```text
///   g̈ = J u̇ + (dJ/dt) u = 0,     u̇ = u̇_applied - M⁻¹Jᵀλ
///   ⟹  (J M⁻¹ Jᵀ) λ = J u̇_applied + (dJ/dt) u
/// ```
///
/// `M⁻¹` is block diagonal: `1/m` on the linear part, and `A = R I⁻¹ Rᵀ`
/// on the angular part. The angular acceleration is
/// `ω̇ = A(L̇ - ω × L)` — differentiating `ω = A L` gives
/// `Ȧ = [ω]ₓA - A[ω]ₓ`, and `ω × ω` vanishes, leaving exactly that. The
/// `ω × L` term is the gyroscopic one; a spinning body's joint carries
/// it, and dropping it makes the initial acceleration wrong by that
/// much.
///
/// `(dJ/dt)u` is taken as a central difference of `J·u` along the motion
/// at fixed `u`, which is what it means.
fn seed_multipliers(
    p: &DaeParams,
    pose: &[Pose],
    v: &[Vec3],
    w: &[Vec3],
    angmom: &[Vec3],
    force: &[Vec3],
    torque: &[Vec3],
) -> Vec<f64> {
    let m = p.m;
    if m == 0 {
        return Vec::new();
    }
    let n = p.n;
    /* A_i = R I⁻¹ Rᵀ, the world-frame inverse inertia. */
    let a: Vec<Mat3> = (0..n)
        .map(|i| {
            let r = pose[i].orientation.to_rotation_matrix();
            r * p.inverse_inertia[i] * r.transpose()
        })
        .collect();

    /* Generalized inverse mass applied to a wrench. */
    let apply_minv = |f: &[Vec3], t: &[Vec3]| -> (Vec<Vec3>, Vec<Vec3>) {
        let lin = (0..n)
            .map(|i| {
                if p.anchors.translation_fixed[i] {
                    Vec3::zeros()
                } else {
                    f[i] * p.inverse_masses[i]
                }
            })
            .collect();
        let ang = (0..n)
            .map(|i| {
                if p.anchors.rotation_fixed[i] {
                    Vec3::zeros()
                } else {
                    a[i] * t[i]
                }
            })
            .collect();
        (lin, ang)
    };
    /* J · (lin, ang) */
    let apply_j = |lin: &[Vec3], ang: &[Vec3]| -> Vec<f64> {
        let mut out = vec![0.0; m];
        p.constraints.for_each_block(pose, |row, b| {
            out[row] += b.jv.dot(lin[b.body]) + b.jw.dot(ang[b.body]);
        });
        out
    };

    /* Right-hand side: J u̇_applied + (dJ/dt) u. */
    let gyro: Vec<Vec3> = (0..n).map(|i| torque[i] - w[i].cross(angmom[i])).collect();
    let (lin_app, ang_app) = apply_minv(force, &gyro);
    let mut rhs = apply_j(&lin_app, &ang_app);

    let h = 1e-6;
    let advance = |dt: f64| -> Vec<Pose> {
        pose.iter()
            .enumerate()
            .map(|(k, q)| Pose {
                position: q.position + dt * v[k],
                orientation: (q.orientation + (Quat::pure(w[k]) * q.orientation) * (0.5 * dt))
                    .normalize(),
            })
            .collect()
    };
    let (mut gp, mut gm) = (vec![0.0; m], vec![0.0; m]);
    p.constraints.velocity_residual(&advance(h), v, w, &mut gp);
    p.constraints.velocity_residual(&advance(-h), v, w, &mut gm);
    for k in 0..m {
        rhs[k] += (gp[k] - gm[k]) / (2.0 * h);
    }

    /* S = J M⁻¹ Jᵀ, one column at a time. `m` is the number of joint
     * rows — a handful — so a dense build and solve is the cheap way. */
    let mut mat = vec![0.0; m * m];
    let mut e = vec![0.0; m];
    for l in 0..m {
        e[l] = 1.0;
        let (mut f, mut t) = (vec![Vec3::zeros(); n], vec![Vec3::zeros(); n]);
        p.constraints.add_jacobian_transpose(pose, &p.anchors, &e, &mut f, &mut t);
        let (lin, ang) = apply_minv(&f, &t);
        let col = apply_j(&lin, &ang);
        for (k, val) in col.into_iter().enumerate() {
            mat[k * m + l] = val;
        }
        e[l] = 0.0;
    }
    solve_dense(&mut mat, &mut rhs, m);

    rhs
}

/// Gaussian elimination with partial pivoting, in place. A singular
/// system leaves the affected multipliers at zero rather than producing
/// infinities: an over-constrained mechanism is reported by IDA with a
/// message the caller can act on, and a NaN here would only obscure it.
fn solve_dense(mat: &mut [f64], b: &mut [f64], m: usize) {
    for col in 0..m {
        let mut piv = col;
        for r in col + 1..m {
            if mat[r * m + col].abs() > mat[piv * m + col].abs() {
                piv = r;
            }
        }
        if mat[piv * m + col].abs() < 1e-14 {
            b[col] = 0.0;
            continue;
        }
        if piv != col {
            for k in 0..m {
                mat.swap(col * m + k, piv * m + k);
            }
            b.swap(col, piv);
        }
        let d = mat[col * m + col];
        for r in col + 1..m {
            let factor = mat[r * m + col] / d;
            if factor == 0.0 {
                continue;
            }
            for k in col..m {
                mat[r * m + k] -= factor * mat[col * m + k];
            }
            b[r] -= factor * b[col];
        }
    }
    for col in (0..m).rev() {
        let d = mat[col * m + col];
        if d.abs() < 1e-14 {
            b[col] = 0.0;
            continue;
        }
        let mut acc = b[col];
        for k in col + 1..m {
            acc -= mat[col * m + k] * b[k];
        }
        b[col] = acc / d;
    }
}

/// IDA path — the only integrator that honours rigid joints.
///
/// The state is the ordinary 13-per-object packing (`system.rs`), so this
/// integrates the *same* rigid-body dynamics as the CVODE path, plus the
/// joints. With no joints at all it is simply a BDF integration of that
/// system, which is how the cross-check in the tests works.
fn run_ida(
    system: &mut PhysicalObjectSystem,
    t_end: f64,
    nout: usize,
) -> Result<RunReport, String> {
    let t0 = system.time;
    if t_end <= t0 {
        return Err(format!("t_end ({t_end}) must be greater than current time ({t0})"));
    }
    let nout = nout.max(1);
    let n = system.objects.len();
    if n == 0 {
        system.time = t_end;
        return Ok(RunReport::default());
    }
    let velocity_correction = project_initial_velocities(system)?;
    let m = system.constraints.len();
    let base = system.state_len();
    let neq = base + 2 * m;

    let mut sunctx_out: Option<SUNContext> = None;
    let retval_ctx = SUNContext_Create(SUN_COMM_NULL, &mut sunctx_out);
    if retval_ctx != 0 {
        return Err(format!("SUNContext_Create failed: {retval_ctx}"));
    }
    let sunctx = sunctx_out.ok_or_else(|| "SUNContext_Create returned NULL".to_string())?;

    let yy = N_VNew_Serial(neq as i64, &sunctx)
        .ok_or_else(|| format!("N_VNew_Serial({neq}) returned NULL"))?;
    let yp = N_VNew_Serial(neq as i64, &sunctx)
        .ok_or_else(|| format!("N_VNew_Serial({neq}) returned NULL"))?;
    let id = N_VNew_Serial(neq as i64, &sunctx)
        .ok_or_else(|| format!("N_VNew_Serial({neq}) returned NULL"))?;

    let params = DaeParams::from_system(system);

    /* y0 = [packed state | λ₀ | 0]; the multipliers are solved for below
     * (`seed_multipliers`), because IDACalcIC cannot find them — see the
     * comment on that function. The configuration itself is consistent by
     * construction: every joint is built from the pose the bodies are
     * already in. */
    with_data_mut(&yy, |d| {
        d[0..base].copy_from_slice(&system.pack_state());
        for k in base..neq {
            d[k] = 0.0;
        }
    })
    .ok_or_else(|| no_array("yy"))?;

    /* yp0 from the UNCONSTRAINED dynamics — the derivative IDACalcIC
     * refines. */
    let y0: Vec<f64> = with_data(&yy, |d| d.to_vec()).ok_or_else(|| no_array("yy"))?;
    let (pose0, v0, w0) = params.kinematics(&y0);
    let (f0, tq0) = params.applied(&y0, &pose0, &v0);
    let angmom0: Vec<Vec3> = (0..n)
        .map(|i| read_vec3(&y0, VARS_PER_OBJECT * i + 10))
        .collect();
    let lam0 = seed_multipliers(&params, &pose0, &v0, &w0, &angmom0, &f0, &tq0);
    let (mut fc0, mut tc0) = (vec![Vec3::zeros(); n], vec![Vec3::zeros(); n]);
    if m > 0 {
        params
            .constraints
            .add_jacobian_transpose(&pose0, &params.anchors, &lam0, &mut fc0, &mut tc0);
        with_data_mut(&yy, |d| d[base..base + m].copy_from_slice(&lam0))
            .ok_or_else(|| no_array("yy"))?;
    }
    with_data_mut(&yp, |d| {
        for i in 0..n {
            let b = VARS_PER_OBJECT * i;
            let trans_free = !params.anchors.translation_fixed[i];
            let rot_free = !params.anchors.rotation_fixed[i];
            write_vec3(d, b, if trans_free { v0[i] } else { Vec3::zeros() });
            write_vec3(d, b + 3, if trans_free { f0[i] - fc0[i] } else { Vec3::zeros() });
            let qdot = if rot_free {
                (Quat::pure(w0[i]) * pose0[i].orientation) * 0.5
            } else {
                Quat::new(0.0, 0.0, 0.0, 0.0)
            };
            d[b + 6] = qdot.w;
            d[b + 7] = qdot.x;
            d[b + 8] = qdot.y;
            d[b + 9] = qdot.z;
            write_vec3(d, b + 10, if rot_free { tq0[i] - tc0[i] } else { Vec3::zeros() });
        }
        for k in base..neq {
            d[k] = 0.0;
        }
    })
    .ok_or_else(|| no_array("yp"))?;

    /* id: 1 = differential (the whole 13N state), 0 = algebraic (λ, μ).
     * IDA needs this to know which components IDACalcIC may move and
     * which the error test should ignore. */
    with_data_mut(&id, |d| {
        for k in 0..base {
            d[k] = 1.0;
        }
        for k in base..neq {
            d[k] = 0.0;
        }
    })
    .ok_or_else(|| no_array("id"))?;

    let ida_mem = IDACreate(&sunctx).ok_or_else(|| "IDACreate returned NULL".to_string())?;
    let mut retval = IDAInit(&ida_mem, dae_residual, t0, &yy, &yp);
    if retval != IDA_SUCCESS {
        return Err(format!("IDAInit failed: {retval}"));
    }
    /* An orientation-gripping DAE cannot be integrated to an arbitrarily
     * tight tolerance — see ROT_JOINT_RTOL_FLOOR. Raise it, and record
     * that it was raised. */
    let (rtol, atol) = if system.constraints.has_rotational() {
        (
            system.rtol.max(ROT_JOINT_RTOL_FLOOR),
            system.atol.max(ROT_JOINT_ATOL_FLOOR),
        )
    } else {
        (system.rtol, system.atol)
    };
    let floored = rtol != system.rtol || atol != system.atol;
    retval = IDASStolerances(&ida_mem, rtol, atol);
    if retval != IDA_SUCCESS {
        return Err(format!("IDASStolerances failed: {retval}"));
    }
    retval = IDASetUserData(&ida_mem, Some(Box::new(params.clone())));
    if retval != IDA_SUCCESS {
        return Err(format!("IDASetUserData failed: {retval}"));
    }
    retval = IDASetId(&ida_mem, Some(&id));
    if retval != IDA_SUCCESS {
        return Err(format!("IDASetId failed: {retval}"));
    }
    /* The multipliers are algebraic; keeping them out of the local error
     * test is what stops IDA chasing their (meaningless) accuracy. */
    retval = IDASetSuppressAlg(&ida_mem, true);
    if retval != IDA_SUCCESS {
        return Err(format!("IDASetSuppressAlg failed: {retval}"));
    }
    retval = IDASetMaxNumSteps(&ida_mem, 500_000);
    if retval != IDA_SUCCESS {
        return Err(format!("IDASetMaxNumSteps failed: {retval}"));
    }
    let a = SUNDenseMatrix(neq as i64, neq as i64, &sunctx)
        .ok_or_else(|| format!("SUNDenseMatrix({neq}, {neq}) returned NULL"))?;
    let ls = SUNLinSol_Dense(&yy, &a, &sunctx)
        .ok_or_else(|| "SUNLinSol_Dense returned NULL".to_string())?;
    retval = IDASetLinearSolver(&ida_mem, &ls, Some(&a));
    if retval != IDA_SUCCESS {
        return Err(format!("IDASetLinearSolver failed: {retval}"));
    }

    let span = t_end - t0;

    let mut report = RunReport::default();
    let mut t = t0;
    for k in 1..=nout {
        let tout = t0 + span * (k as f64) / (nout as f64);
        let r = IDASolve(&ida_mem, tout, &mut t, &yy, &yp, IDA_NORMAL);
        if r < 0 {
            let hint = if system.constraints.has_rotational() {
                " — an orientation-gripping DAE is index 2 and does not converge at every \
                 tolerance; try a looser one (`set system.rtol = 1e-5`)"
            } else {
                ""
            };
            return Err(format!("IDASolve failed with retval = {r} at t = {t}{hint}"));
        }
        with_data(&yy, |d| system.unpack_state(&d[0..base])).ok_or_else(|| no_array("yy"))?;
        system.time = t;
        report.snapshots.push(snapshot(system, t));
    }

    let (mut nst, mut nre, mut nni, mut netf) = (0i64, 0i64, 0i64, 0i64);
    IDAGetNumSteps(&ida_mem, &mut nst);
    IDAGetNumResEvals(&ida_mem, &mut nre);
    IDAGetNumNonlinSolvIters(&ida_mem, &mut nni);
    IDAGetNumErrTestFails(&ida_mem, &mut netf);
    report.nst = nst;
    report.nfe = nre;
    report.nni = nni;
    report.netf = netf;
    report.constraint_drift = system.constraints.drift(system);
    report.tolerance_floored = floored;
    report.initial_velocity_projected = velocity_correction;

    let mut ida_mem = Some(ida_mem);
    IDAFree(&mut ida_mem);
    SUNLinSolFree(Some(ls));
    SUNMatDestroy(a);
    N_VDestroy(yy);
    N_VDestroy(yp);
    N_VDestroy(id);
    let mut sunctx = Some(sunctx);
    SUNContext_Free(&mut sunctx);
    Ok(report)
}

/// Makes the starting velocities **consistent with the joints**, and
/// returns how big a correction that took.
///
/// This is the whole answer to what looked for a long time like an
/// index-2 wall. A ball joint says the two bodies share a point, so at
/// the velocity level it says
///
/// ```text
///   v_i + ω_i × r_i = v_j + ω_j × r_j
/// ```
///
/// — a body turning about a pivot offset from its centre **must have its
/// centre moving**. Give it `ω` and leave `v` at zero and `ġ = J·u ≠ 0`:
/// the initial condition violates the constraint, and IDA is being asked
/// to integrate a state that is not on the manifold. It fails on the
/// first step, at every tolerance, which is exactly what was observed.
///
/// A rod has `J_ω = 0`, so spin never enters its `ġ` — which is why rods
/// tolerated spinning bodies from the start and hid the problem.
///
/// The fix is the standard impulsive projection: find the smallest
/// (mass-weighted) velocity change that lands on the manifold,
///
/// ```text
///   minimise ½ δuᵀ M δu   subject to   J(u + δu) = 0
///   ⟹  (J M⁻¹ Jᵀ) ν = J·u,     δu = -M⁻¹Jᵀν
/// ```
///
/// and the momentum-level form is simply `δ(p, L) = -Jᵀν`, because
/// `M·M⁻¹Jᵀν = Jᵀν`. Anchors have `M⁻¹ = 0` and are untouched.
///
/// Physically this is the impulse the joint delivers the instant it is
/// engaged — exactly what a real coupling does when you clutch it to a
/// spinning shaft. It is reported rather than done silently, because it
/// changes the state the caller handed in.
fn project_initial_velocities(system: &mut PhysicalObjectSystem) -> Result<f64, String> {
    let m = system.constraints.len();
    if m == 0 {
        return Ok(0.0);
    }
    let n = system.objects.len();
    let p = DaeParams::from_system(system);
    let pose = ConstraintSet::poses(system);
    let a: Vec<Mat3> = (0..n)
        .map(|i| {
            let r = pose[i].orientation.normalize().to_rotation_matrix();
            r * p.inverse_inertia[i] * r.transpose()
        })
        .collect();
    let apply_minv = |f: &[Vec3], t: &[Vec3]| -> (Vec<Vec3>, Vec<Vec3>) {
        (
            (0..n)
                .map(|i| {
                    if p.anchors.translation_fixed[i] {
                        Vec3::zeros()
                    } else {
                        f[i] * p.inverse_masses[i]
                    }
                })
                .collect(),
            (0..n)
                .map(|i| {
                    if p.anchors.rotation_fixed[i] {
                        Vec3::zeros()
                    } else {
                        a[i] * t[i]
                    }
                })
                .collect(),
        )
    };
    let apply_j = |lin: &[Vec3], ang: &[Vec3]| -> Vec<f64> {
        let mut out = vec![0.0; m];
        p.constraints.for_each_block(&pose, |row, b| {
            out[row] += b.jv.dot(lin[b.body]) + b.jw.dot(ang[b.body]);
        });
        out
    };

    let v: Vec<Vec3> = system.objects.iter().map(|o| o.get_velocity()).collect();
    let w: Vec<Vec3> = system.objects.iter().map(crate::constrain::angular_velocity).collect();
    let mut rhs = vec![0.0; m];
    p.constraints.velocity_residual(&pose, &v, &w, &mut rhs);
    let worst = rhs.iter().fold(0.0f64, |acc, x| acc.max(x.abs()));
    if worst <= 1.0e-12 {
        return Ok(0.0);
    }

    /* S = J M⁻¹ Jᵀ, the same matrix `seed_multipliers` builds. */
    let mut mat = vec![0.0; m * m];
    let mut e = vec![0.0; m];
    for l in 0..m {
        e[l] = 1.0;
        let (mut f, mut t) = (vec![Vec3::zeros(); n], vec![Vec3::zeros(); n]);
        p.constraints.add_jacobian_transpose(&pose, &p.anchors, &e, &mut f, &mut t);
        let (lin, ang) = apply_minv(&f, &t);
        for (k, val) in apply_j(&lin, &ang).into_iter().enumerate() {
            mat[k * m + l] = val;
        }
        e[l] = 0.0;
    }
    solve_dense(&mut mat, &mut rhs, m);

    /* δ(p, L) = -Jᵀν, applied through the setters (hard rule 4). */
    let (mut dp, mut dl) = (vec![Vec3::zeros(); n], vec![Vec3::zeros(); n]);
    p.constraints
        .add_jacobian_transpose(&pose, &p.anchors, &rhs, &mut dp, &mut dl);
    let mut correction = 0.0f64;
    for i in 0..n {
        if !p.anchors.translation_fixed[i] {
            let before = system.objects[i].get_velocity();
            let momentum = system.objects[i].get_momentum();
            system.objects[i].set_momentum(momentum - dp[i]);
            correction = correction.max((system.objects[i].get_velocity() - before).norm());
        }
        if !p.anchors.rotation_fixed[i] {
            let before = crate::constrain::angular_velocity(&system.objects[i]);
            let angmom = system.objects[i].get_angular_momentum();
            system.objects[i].set_angular_momentum(angmom - dl[i]);
            correction = correction
                .max((crate::constrain::angular_velocity(&system.objects[i]) - before).norm());
        }
    }

    /* It must have worked: if the residual survives, the constraint set
     * is singular (an over-constrained mechanism), and saying so beats
     * integrating something that is not on the manifold. */
    let v2: Vec<Vec3> = system.objects.iter().map(|o| o.get_velocity()).collect();
    let w2: Vec<Vec3> = system.objects.iter().map(crate::constrain::angular_velocity).collect();
    let mut after = vec![0.0; m];
    p.constraints.velocity_residual(&pose, &v2, &w2, &mut after);
    let left = after.iter().fold(0.0f64, |acc, x| acc.max(x.abs()));
    if left > 1.0e-9 * worst.max(1.0) {
        return Err(format!(
            "the starting velocities cannot be made consistent with the joints \
             (|J·u| = {worst:e} before the projection, {left:e} after). That means the joint \
             set is singular — usually two joints competing for the same freedom, or a \
             mechanism with no assembly at all. CONSTRAINTS lists them"
        ));
    }
    Ok(correction)
}
