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
    match system.method.clone() {
        Method::Adams => run_cvode(system, t_end, nout, CV_ADAMS),
        Method::Bdf => run_cvode(system, t_end, nout, CV_BDF),
        Method::Sprk { table, dt } => run_sprk(system, t_end, nout, &table, dt),
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
