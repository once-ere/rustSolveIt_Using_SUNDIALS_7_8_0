//! Static equilibrium — where does this system come to rest?
//!
//! Integration answers "what happens next". This module answers a
//! different question: **is there a configuration where nothing wants to
//! move, and what is it?** That is not an initial-value problem at all,
//! it is a nonlinear algebraic system, and it is what `kinsol_rs`
//! (KINSOL) is for.
//!
//! # What is solved
//!
//! Unknowns `u = [q(3N) | λ(m)]` — every body position, plus one Lagrange
//! multiplier per rigid constraint. The residual is
//!
//! ```text
//!   F_i(u) = applied force on body i  -  (Gᵀλ)_i        (free body)
//!   F_i(u) = q_i - q_i⁰                                  (anchor: stay put)
//!   F_λ(u) = g(q)                                        (constraints hold)
//! ```
//!
//! `F(u) = 0` says precisely: every free body has zero net force, every
//! anchor is where it started, and every rod is the right length. KINSOL
//! finds it with Newton's method plus a line search, using a dense
//! difference-quotient Jacobian.
//!
//! Velocities are zero at equilibrium, so the magnetic term `q v × B`
//! vanishes identically and the residual is a pure function of position.
//! The solver therefore cannot be fooled by a velocity-dependent force —
//! there are none to have.
//!
//! # What it does not tell you
//!
//! **Nothing here says the equilibrium is stable.** A pencil balanced on
//! its point is an equilibrium. KINSOL will happily converge to a
//! maximum of the potential if you start it near one. The honest test is
//! to perturb the answer and integrate: a stable equilibrium comes back,
//! an unstable one runs away. `EQUILIBRIUM` reports the residual norm it
//! achieved, not a stability claim.

use std::any::Any;

use crate::constrain::{Anchors, ConstraintSet, Pose};
use crate::linalg::{Quat, Vec3};
use crate::system::PhysicalObjectSystem;

use sundials_core::nvector_serial::N_VNew_Serial;
use sundials_core::sundials_context::{SUNContext, SUNContext_Create, SUNContext_Free};
use sundials_core::sundials_linearsolver::SUNLinSolFree;
use sundials_core::sundials_matrix::SUNMatDestroy;
use sundials_core::sundials_nvector::{N_VConst, N_VDestroy, N_VGetArrayPointer, N_Vector};
use sundials_core::sundials_types::{sunrealtype, SUN_COMM_NULL};
use sundials_core::sunlinsol_dense::SUNLinSol_Dense;
use sundials_core::sunmatrix_dense::SUNDenseMatrix;

use kinsol_rs::kinsol::{KINCreate, KINFree, KINInit, KINSol};
use kinsol_rs::kinsol_impl::{KINMem, KIN_LINESEARCH, KIN_SUCCESS};
use kinsol_rs::kinsol_io::{
    KINGetFuncNorm, KINGetNumFuncEvals, KINGetNumNonlinSolvIters, KINSetFuncNormTol,
    KINSetNumMaxIters, KINSetUserData,
};
use kinsol_rs::kinsol_ls::KINSetLinearSolver;

/// What the solve achieved. `residual_norm` is the scaled norm of `F(u)`
/// KINSOL stopped at — the number that says how well "nothing wants to
/// move" actually holds.
#[derive(Clone, Debug, Default)]
pub struct EquilibriumReport {
    pub iterations: i64,
    pub func_evals: i64,
    pub residual_norm: f64,
    /// Worst `|g|` over the constraint set at the solution.
    pub constraint_error: f64,
    /// Largest net force left on any single free body — the physical
    /// reading of "converged", in force units rather than solver units.
    pub max_net_force: f64,
}

/// Snapshot handed to the KINSOL residual. Positions are the unknowns,
/// so they are *not* here; everything else is.
#[derive(Clone, Debug)]
struct KinParams {
    n: usize,
    m: usize,
    g: f64,
    softening: f64,
    uniform_gravity: Vec3,
    e_field: Vec3,
    masses: Vec<f64>,
    charges: Vec<f64>,
    ext_force: Vec<Vec3>,
    anchors: Anchors,
    anchor_positions: Vec<f64>,
    /// Orientations are FIXED here: equilibrium solves for positions
    /// only, so a joint that grips orientation is refused up front.
    orientations: Vec<Quat>,
    constraints: ConstraintSet,
}

impl KinParams {
    /// Poses at a candidate configuration: the unknown positions with the
    /// (fixed) starting orientations.
    fn poses(&self, q: &[f64]) -> Vec<Pose> {
        (0..self.n)
            .map(|i| Pose { position: read3(q, 3 * i), orientation: self.orientations[i] })
            .collect()
    }
}

impl KinParams {
    /// Applied force on body `i` at configuration `q`, at rest.
    ///
    /// Same arithmetic order as `integrate::rhs_full`'s force block, with
    /// the `v × B` term dropped because `v = 0` here — dropping it is not
    /// an approximation, the cross product is exactly zero.
    fn force(&self, q: &[f64], i: usize) -> Vec3 {
        let pos_i = read3(q, 3 * i);
        let mut force = Vec3::zeros();
        for j in 0..self.n {
            if i == j {
                continue;
            }
            let r_vec = read3(q, 3 * j) - pos_i;
            let dist_sq = r_vec.norm_squared() + self.softening * self.softening;
            let dist = dist_sq.sqrt();
            force += (self.g * self.masses[i] * self.masses[j] / (dist_sq * dist)) * r_vec;
        }
        force += self.masses[i] * self.uniform_gravity;
        force += self.charges[i] * self.e_field;
        force += self.ext_force[i];
        force
    }
}

/// The KINSOL system function `F(u)`.
fn kin_residual(uu: &N_Vector, fval: &N_Vector, user_data: &mut Option<Box<dyn Any>>) -> i32 {
    let p = match user_data.as_mut().and_then(|b| b.downcast_mut::<KinParams>()) {
        Some(p) => p,
        None => return -1,
    };
    let u = match N_VGetArrayPointer(uu) {
        Some(d) => d,
        None => return -1,
    };
    let mut fg = match N_VGetArrayPointer(fval) {
        Some(d) => d,
        None => return -1,
    };
    let (n, m) = (p.n, p.m);
    let (u, f) = (&u[..], &mut fg[..]);
    let q = &u[0..3 * n];
    let lam = &u[3 * n..3 * n + m];

    for i in 0..n {
        let b = 3 * i;
        if p.anchors.translation_fixed[i] {
            /* An anchor is not free to move: pin it to where it was. */
            f[b] = q[b] - p.anchor_positions[b];
            f[b + 1] = q[b + 1] - p.anchor_positions[b + 1];
            f[b + 2] = q[b + 2] - p.anchor_positions[b + 2];
            continue;
        }
        let force = p.force(q, i);
        f[b] = force.x;
        f[b + 1] = force.y;
        f[b + 2] = force.z;
    }
    /* Subtract the constraint force Gᵀλ; anchors are skipped inside, and
     * their rows above already pin them, so the two never fight. */
    let pose = p.poses(q);
    let (mut gf, mut gt) = (vec![Vec3::zeros(); n], vec![Vec3::zeros(); n]);
    p.constraints
        .add_jacobian_transpose(&pose, &p.anchors, lam, &mut gf, &mut gt);
    for i in 0..n {
        f[3 * i] -= gf[i].x;
        f[3 * i + 1] -= gf[i].y;
        f[3 * i + 2] -= gf[i].z;
    }

    if m > 0 {
        let (_, tail) = f.split_at_mut(3 * n);
        p.constraints.residual(&pose, &mut tail[0..m]);
    }
    0
}

/// Moves `system` to a static equilibrium, in place.
///
/// Every body's position is updated through the setters (hard rule 4) and
/// every velocity is zeroed — at equilibrium, by definition. The system's
/// `time` is untouched: this is not an integration.
///
/// The initial guess is wherever the bodies already are, so `EQUILIBRIUM`
/// finds *the nearby* rest state, not some global one. That is the useful
/// behaviour: drop a chain roughly into place and let it settle.
pub fn solve(system: &mut PhysicalObjectSystem) -> Result<EquilibriumReport, String> {
    let n = system.objects.len();
    if n == 0 {
        return Ok(EquilibriumReport::default());
    }
    if system.objects.iter().all(|o| o.get_inverse_mass() == 0.0) {
        return Err(
            "every object has inverse_mass = 0 — there is nothing free to move, so equilibrium \
             is whatever you already have"
                .to_string(),
        );
    }
    if system.constraints.has_rotational() {
        let kinds: Vec<&str> = system
            .constraints
            .joints
            .iter()
            .filter(|j| j.is_rotational())
            .map(|j| j.kind())
            .collect();
        return Err(format!(
            "EQUILIBRIUM solves for positions only, and this system has orientation-gripping \
             joint(s): {kinds:?}. Finding the rest pose of a mechanism means solving for \
             orientations too, which this does not do — integrate it with METHOD IDA and let \
             it settle, or drop the joint"
        ));
    }
    let m = system.constraints.len();
    let neq = 3 * n + m;

    let mut sunctx_out: Option<SUNContext> = None;
    let rc = SUNContext_Create(SUN_COMM_NULL, &mut sunctx_out);
    if rc != 0 {
        return Err(format!("SUNContext_Create failed: {rc}"));
    }
    let sunctx = sunctx_out.ok_or_else(|| "SUNContext_Create returned NULL".to_string())?;

    let u = N_VNew_Serial(neq as i64, &sunctx)
        .ok_or_else(|| format!("N_VNew_Serial({neq}) returned NULL"))?;
    let u_scale = N_VNew_Serial(neq as i64, &sunctx)
        .ok_or_else(|| format!("N_VNew_Serial({neq}) returned NULL"))?;
    let f_scale = N_VNew_Serial(neq as i64, &sunctx)
        .ok_or_else(|| format!("N_VNew_Serial({neq}) returned NULL"))?;
    N_VConst(1.0, &u_scale);
    N_VConst(1.0, &f_scale);

    let anchor_positions: Vec<f64> = {
        let mut v = vec![0.0; 3 * n];
        for (i, o) in system.objects.iter().enumerate() {
            let p = o.get_position();
            v[3 * i] = p.x;
            v[3 * i + 1] = p.y;
            v[3 * i + 2] = p.z;
        }
        v
    };
    let params = KinParams {
        n,
        m,
        g: system.g_constant,
        softening: system.softening,
        uniform_gravity: system.uniform_gravity,
        e_field: system.e_field,
        masses: system.objects.iter().map(|o| o.get_mass()).collect(),
        charges: system.objects.iter().map(|o| o.get_charge()).collect(),
        ext_force: system.external_forces.clone(),
        anchors: Anchors::of(system),
        orientations: system.objects.iter().map(|o| o.get_orientation().normalize()).collect(),
        anchor_positions: anchor_positions.clone(),
        constraints: system.constraints.clone(),
    };

    /* Initial guess. Positions: where the bodies already are.
     *
     * Multipliers: NOT zero. With λ = 0 the residual's derivative with
     * respect to position is ∂F/∂q alone, and for a body in uniform
     * gravity that is the zero matrix — the Newton matrix is singular at
     * the starting point and KINSOL fails its very first setup with
     * KIN_LSETUP_FAIL. Seeding λ from the force balance along the rod,
     *
     *     F·d = 2λ|d|²      so      λ ≈ (F·d) / (2|d|²),
     *
     * puts the -2λI block on the diagonal and the iteration starts on a
     * nonsingular matrix. It is also simply the right answer for a
     * single rod, so the first Newton step lands close. */
    let lam0 = initial_multipliers(&params, &anchor_positions);
    with_data_mut(&u, |d| {
        d[0..3 * n].copy_from_slice(&anchor_positions);
        d[3 * n..neq].copy_from_slice(&lam0);
    })
    .ok_or_else(|| no_array("u"))?;

    let kin_mem = KINCreate(&sunctx).ok_or_else(|| "KINCreate returned NULL".to_string())?;
    let mut rc = KINInit(&kin_mem, kin_residual, &u);
    if rc != KIN_SUCCESS {
        return Err(format!("KINInit failed: {rc}"));
    }
    rc = KINSetUserData(&kin_mem, Some(Box::new(params)));
    if rc != KIN_SUCCESS {
        return Err(format!("KINSetUserData failed: {rc}"));
    }
    rc = KINSetFuncNormTol(&kin_mem, 1.0e-12);
    if rc != KIN_SUCCESS {
        return Err(format!("KINSetFuncNormTol failed: {rc}"));
    }
    rc = KINSetNumMaxIters(&kin_mem, 500);
    if rc != KIN_SUCCESS {
        return Err(format!("KINSetNumMaxIters failed: {rc}"));
    }
    let a = SUNDenseMatrix(neq as i64, neq as i64, &sunctx)
        .ok_or_else(|| format!("SUNDenseMatrix({neq}, {neq}) returned NULL"))?;
    let ls = SUNLinSol_Dense(&u, &a, &sunctx)
        .ok_or_else(|| "SUNLinSol_Dense returned NULL".to_string())?;
    rc = KINSetLinearSolver(&kin_mem, &ls, Some(&a));
    if rc != KIN_SUCCESS {
        return Err(format!("KINSetLinearSolver failed: {rc}"));
    }

    /* KIN_LINESEARCH rather than KIN_NONE: a bare Newton step from a
     * rough guess routinely overshoots into a configuration where two
     * bodies have swapped sides, and the line search is what keeps the
     * iteration in the basin the user aimed at. */
    let rc = KINSol(&kin_mem, &u, KIN_LINESEARCH, &u_scale, &f_scale);
    if rc < 0 {
        let hint = if system.objects.iter().all(|o| o.get_inverse_mass() != 0.0) {
            " Every body here is free to move, so the whole system can be translated bodily \
             without changing any force — the equilibrium is not isolated and the Newton matrix \
             is singular. Pin one body with `set objN.inverse_mass = 0` to fix the frame."
        } else {
            " Move the bodies closer to where you expect them to rest, or check that a rest \
             state exists at all (a single body in uniform gravity has none)."
        };
        return Err(format!(
            "KINSol failed: {rc} — no equilibrium was found near the current configuration.{hint}"
        ));
    }

    let mut report = EquilibriumReport::default();
    KINGetNumNonlinSolvIters(&kin_mem, &mut report.iterations);
    KINGetNumFuncEvals(&kin_mem, &mut report.func_evals);
    KINGetFuncNorm(&kin_mem, &mut report.residual_norm);

    /* Write the answer back through the setters, at rest.
     *
     * An anchor is restored to its EXACT starting position rather than to
     * whatever the solver's pin row converged to. The pin equation is
     * `q_i - q_i⁰ = 0`, so the residual leaves roundoff of order 1e-27
     * behind — harmless numerically, but "a wall never moves" is a
     * property callers rely on bit-for-bit (the collision suite asserts
     * it), and it costs nothing to make it exact. */
    let anchors = Anchors::of(system);
    with_data(&u, |d| {
        for (i, o) in system.objects.iter_mut().enumerate() {
            if anchors.translation_fixed[i] {
                o.set_position(read3(&anchor_positions, 3 * i));
            } else {
                o.set_position(read3(d, 3 * i));
            }
            o.set_velocity(Vec3::zeros());
        }
    })
    .ok_or_else(|| no_array("u"))?;

    /* Report the physical residual, not only the solver's scaled one. */
    let params2 = KinParams {
        n,
        m,
        g: system.g_constant,
        softening: system.softening,
        uniform_gravity: system.uniform_gravity,
        e_field: system.e_field,
        masses: system.objects.iter().map(|o| o.get_mass()).collect(),
        charges: system.objects.iter().map(|o| o.get_charge()).collect(),
        ext_force: system.external_forces.clone(),
        anchors: Anchors::of(system),
        orientations: system.objects.iter().map(|o| o.get_orientation().normalize()).collect(),
        anchor_positions,
        constraints: system.constraints.clone(),
    };
    let q: Vec<f64> = with_data(&u, |d| d[0..3 * n].to_vec()).ok_or_else(|| no_array("u"))?;
    let lam: Vec<f64> =
        with_data(&u, |d| d[3 * n..3 * n + m].to_vec()).ok_or_else(|| no_array("u"))?;
    let pose2 = params2.poses(&q);
    let (mut gf, mut gt2) = (vec![Vec3::zeros(); n], vec![Vec3::zeros(); n]);
    params2
        .constraints
        .add_jacobian_transpose(&pose2, &params2.anchors, &lam, &mut gf, &mut gt2);
    let mut worst = 0.0f64;
    for i in 0..n {
        if params2.anchors.translation_fixed[i] {
            continue;
        }
        worst = worst.max((params2.force(&q, i) - gf[i]).norm());
    }
    report.max_net_force = worst;
    report.constraint_error = system.constraints.drift(system).0;

    let mut kin_mem: Option<KINMem> = Some(kin_mem);
    KINFree(&mut kin_mem);
    SUNLinSolFree(Some(ls));
    SUNMatDestroy(a);
    N_VDestroy(u);
    N_VDestroy(u_scale);
    N_VDestroy(f_scale);
    let mut sunctx = Some(sunctx);
    SUNContext_Free(&mut sunctx);
    Ok(report)
}

/// Force-balance estimate of each Lagrange multiplier at the starting
/// configuration — see the comment at the call site for why zero is not
/// an option. Uses whichever end of the rod is free; if both are, their
/// average, which is exact for a symmetric pair.
/// Force-balance estimate of each Lagrange multiplier at the starting
/// configuration — see the comment at the call site for why zero is not
/// an option.
///
/// Only rods reach here: [`solve`] refuses orientation-gripping joints
/// up front, so every joint is a single row whose Jacobian is `±d̂`.
fn initial_multipliers(p: &KinParams, q: &[f64]) -> Vec<f64> {
    p.constraints
        .joints
        .iter()
        .map(|joint| {
            let (i, j) = joint.bodies();
            let d = read3(q, 3 * j) - read3(q, 3 * i);
            let len = d.norm();
            if len == 0.0 {
                return 0.0;
            }
            let dhat = d * (1.0 / len);
            // dg/dq_i = -d̂ and dg/dq_j = +d̂, so the two ends read the
            // same tension with OPPOSITE signs. At rest the balance is
            // F_j·d̂ = λ on one end and F_i·d̂ = -λ on the other.
            let wi = if p.anchors.translation_fixed[i] { 0.0 } else { 1.0 / p.masses[i] };
            let wj = if p.anchors.translation_fixed[j] { 0.0 } else { 1.0 / p.masses[j] };
            if wi + wj == 0.0 {
                return 0.0;
            }
            let fi = if p.anchors.translation_fixed[i] { Vec3::zeros() } else { p.force(q, i) };
            let fj = if p.anchors.translation_fixed[j] { Vec3::zeros() } else { p.force(q, j) };
            (fj.dot(dhat) * wj - fi.dot(dhat) * wi) / (wi + wj)
        })
        .collect()
}

fn read3(d: &[f64], at: usize) -> Vec3 {
    Vec3::new(d[at], d[at + 1], d[at + 2])
}

fn with_data<R>(v: &N_Vector, f: impl FnOnce(&[f64]) -> R) -> Option<R> {
    let d = N_VGetArrayPointer(v)?;
    Some(f(&d))
}

fn with_data_mut<R>(v: &N_Vector, f: impl FnOnce(&mut [f64]) -> R) -> Option<R> {
    let mut d = N_VGetArrayPointer(v)?;
    Some(f(&mut d))
}

fn no_array(what: &str) -> String {
    format!("N_VGetArrayPointer returned NULL for {what} (not a serial N_Vector)")
}

/// Silences the unused-type warning on `sunrealtype` when the module is
/// compiled without the solver paths that name it.
const _: Option<sunrealtype> = None;
