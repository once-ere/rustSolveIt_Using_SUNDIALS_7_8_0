//! Forward sensitivity analysis — how much does the answer depend on the
//! input?
//!
//! Integrating tells you where the bodies end up. This module tells you
//! the **derivative** of that answer with respect to a number you fed in:
//! `∂y(T)/∂p`. If Jupiter were 1 % heavier, how far would Pluto move? The
//! honest way to ask is not to run it twice and subtract — that answer is
//! swamped by the difference of two nearly equal numbers — but to
//! integrate the sensitivity equations alongside the state, which is what
//! `cvodes_rs` (CVODES) and `idas_rs` (IDAS) do.
//!
//! # Which solver
//!
//! | system | solver | why |
//! |---|---|---|
//! | unconstrained | **CVODES** | ODE plus its sensitivity equations |
//! | has `CONSTRAIN` rods | **IDAS** | the same, on the DAE |
//!
//! The choice is automatic: [`run`] picks IDAS exactly when
//! `system.constraints` is non-empty, because that is the only case where
//! the state equations are a DAE at all.
//!
//! # How the derivative is obtained
//!
//! The parameters live in a **shared vector** (`SensParams` —
//! `Rc<RefCell<Vec<f64>>>`), and the right-hand side reads every value it
//! needs out of that vector rather than from a captured copy. That
//! indirection is the whole trick: the solver perturbs an entry, calls
//! the same right-hand side, and differences the result, so the
//! sensitivity equations never have to be written out by hand. Pass
//! `fS: None` and CVODES/IDAS supply them.
//!
//! `plist` selects which entries are differentiated, so the parameter
//! vector can hold everything differentiable while `Ns` stays as small as
//! the user's question.
//!
//! # Scope
//!
//! Like [`crate::constrain`], this runs the **translational** dynamics:
//! positions and velocities, no orientation. [`gate`] refuses a spinning
//! body by name rather than silently differentiating a different problem.

use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

use crate::constrain::{Anchors, ConstraintSet, Pose};
use crate::linalg::{Mat3, Vec3};
use crate::system::PhysicalObjectSystem;

use sundials_core::nvector_serial::N_VNew_Serial;
use sundials_core::sundials_context::{SUNContext, SUNContext_Create, SUNContext_Free};
use sundials_core::sundials_linearsolver::SUNLinSolFree;
use sundials_core::sundials_matrix::SUNMatDestroy;
use sundials_core::sundials_nvector::{
    N_VCloneVectorArray, N_VConst, N_VDestroy, N_VDestroyVectorArray, N_VGetArrayPointer, N_Vector,
};
use sundials_core::sundials_types::{sunrealtype, SUN_COMM_NULL};
use sundials_core::sunlinsol_dense::SUNLinSol_Dense;
use sundials_core::sunmatrix_dense::SUNDenseMatrix;

/// A scalar input the run can be differentiated with respect to.
///
/// Spelled in the language exactly as the `Display` impl prints it:
/// `g_constant`, `mass 2`, `charge 0`, `gravity.y`, `e_field.x`,
/// `b_field.z`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SensParam {
    GConstant,
    Mass(usize),
    Charge(usize),
    Gravity(usize),
    EField(usize),
    BField(usize),
}

impl std::fmt::Display for SensParam {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        const AXIS: [char; 3] = ['x', 'y', 'z'];
        match self {
            SensParam::GConstant => write!(f, "g_constant"),
            SensParam::Mass(k) => write!(f, "mass {k}"),
            SensParam::Charge(k) => write!(f, "charge {k}"),
            SensParam::Gravity(a) => write!(f, "gravity.{}", AXIS[*a]),
            SensParam::EField(a) => write!(f, "e_field.{}", AXIS[*a]),
            SensParam::BField(a) => write!(f, "b_field.{}", AXIS[*a]),
        }
    }
}

/// Slot of `p` this parameter occupies. The layout is fixed so the
/// right-hand side can read every value positionally:
/// `[g | gravity(3) | e_field(3) | b_field(3) | mass(N) | charge(N)]`.
const P_G: usize = 0;
const P_GRAV: usize = 1;
const P_E: usize = 4;
const P_B: usize = 7;
const P_FIXED: usize = 10;

impl SensParam {
    fn slot(&self, n: usize) -> usize {
        match self {
            SensParam::GConstant => P_G,
            SensParam::Gravity(a) => P_GRAV + a,
            SensParam::EField(a) => P_E + a,
            SensParam::BField(a) => P_B + a,
            SensParam::Mass(k) => P_FIXED + k,
            SensParam::Charge(k) => P_FIXED + n + k,
        }
    }

    /// Parses the language spelling. Returns a message naming every
    /// accepted form on failure — a wrong parameter name is the most
    /// likely user error here.
    pub fn parse(text: &str, n: usize) -> Result<Self, String> {
        let t = text.trim().to_ascii_lowercase();
        let axis = |s: &str| match s {
            "x" => Some(0usize),
            "y" => Some(1),
            "z" => Some(2),
            _ => None,
        };
        let indexed = |rest: &str| -> Result<usize, String> {
            let k: usize = rest
                .trim()
                .trim_start_matches("obj")
                .parse()
                .map_err(|_| format!("`{text}` needs an object index, e.g. `mass 0`"))?;
            if k >= n {
                return Err(format!(
                    "`{text}` names obj{k}, but there are only {n} object(s)"
                ));
            }
            Ok(k)
        };
        if t == "g_constant" || t == "g" {
            return Ok(SensParam::GConstant);
        }
        if let Some(rest) = t.strip_prefix("mass") {
            return Ok(SensParam::Mass(indexed(rest)?));
        }
        if let Some(rest) = t.strip_prefix("charge") {
            return Ok(SensParam::Charge(indexed(rest)?));
        }
        for (prefix, make) in [
            ("gravity.", SensParam::Gravity as fn(usize) -> SensParam),
            ("uniform_gravity.", SensParam::Gravity as fn(usize) -> SensParam),
            ("e_field.", SensParam::EField as fn(usize) -> SensParam),
            ("b_field.", SensParam::BField as fn(usize) -> SensParam),
        ] {
            if let Some(rest) = t.strip_prefix(prefix) {
                if let Some(a) = axis(rest) {
                    return Ok(make(a));
                }
            }
        }
        Err(format!(
            "unknown sensitivity parameter `{text}` — expected g_constant, mass <n>, \
             charge <n>, gravity.<x|y|z>, e_field.<x|y|z> or b_field.<x|y|z>"
        ))
    }
}

/// `∂y(T)/∂p` for one parameter, split into the parts a user asks about.
#[derive(Clone, Debug)]
pub struct ParamSensitivity {
    pub param: SensParam,
    /// `∂position_i/∂p` for every object.
    pub d_position: Vec<Vec3>,
    /// `∂velocity_i/∂p` for every object.
    pub d_velocity: Vec<Vec3>,
}

impl ParamSensitivity {
    /// The largest single position derivative — the one-number answer to
    /// "does this parameter matter?".
    pub fn max_position_sensitivity(&self) -> f64 {
        self.d_position.iter().fold(0.0f64, |a, v| a.max(v.norm()))
    }
}

/// Everything a sensitivity run produces.
#[derive(Clone, Debug)]
pub struct SensitivityReport {
    pub t: f64,
    pub per_param: Vec<ParamSensitivity>,
    /// Which solver actually ran — `"CVODES"` or `"IDAS"`.
    pub solver: &'static str,
    pub nst: i64,
}

/// Refuses what the translational sensitivity path cannot express.
pub fn gate(system: &PhysicalObjectSystem) -> Result<(), String> {
    for (k, o) in system.objects.iter().enumerate() {
        if o.get_angular_momentum() != Vec3::zeros()
            && o.get_inverse_inertia_tensor() != Mat3::zeros()
        {
            return Err(format!(
                "sensitivity analysis is translational only: obj{k} has spinning rigid-body \
                 state (nonzero angular momentum and invertible inertia). Zero the spin, or ask \
                 for the trajectory instead of its derivative"
            ));
        }
    }
    if system.constraints.has_rotational() {
        return Err(
            "sensitivity analysis runs the translational dynamics, and this system has a joint \
             that grips orientation (BALL/HINGE/UNIVERSAL). Ask for the trajectory with \
             METHOD IDA instead, or drop the joint"
                .to_string(),
        );
    }
    for (k, tq) in system.external_torques.iter().enumerate() {
        if *tq != Vec3::zeros() {
            return Err(format!(
                "sensitivity analysis cannot apply external torques (obj{k})"
            ));
        }
    }
    Ok(())
}

/// The parameter vector every right-hand side here reads from.
#[derive(Clone, Debug)]
struct SensCommon {
    n: usize,
    m: usize,
    softening: f64,
    ext_force: Vec<Vec3>,
    anchors: Anchors,
    orientations: Vec<crate::linalg::Quat>,
    constraints: ConstraintSet,
    p: Rc<RefCell<Vec<f64>>>,
}

impl SensCommon {
    /// Applied force on body `i`, with **every** parameter read live out
    /// of the shared vector. Nothing is captured: that is what makes the
    /// difference-quotient sensitivity correct.
    ///
    /// Arithmetic order matches `integrate::rhs_full` exactly (hard
    /// rule 3).
    fn force(&self, q: &[f64], v: &[f64], i: usize) -> Vec3 {
        let p = self.p.borrow();
        let n = self.n;
        let g = p[P_G];
        let grav = Vec3::new(p[P_GRAV], p[P_GRAV + 1], p[P_GRAV + 2]);
        let ef = Vec3::new(p[P_E], p[P_E + 1], p[P_E + 2]);
        let bf = Vec3::new(p[P_B], p[P_B + 1], p[P_B + 2]);
        let mass = |k: usize| p[P_FIXED + k];
        let charge = |k: usize| p[P_FIXED + n + k];

        let pos_i = read3(q, 3 * i);
        let v_i = read3(v, 3 * i);
        let mut force = Vec3::zeros();
        for j in 0..n {
            if i == j {
                continue;
            }
            let r_vec = read3(q, 3 * j) - pos_i;
            let dist_sq = r_vec.norm_squared() + self.softening * self.softening;
            let dist = dist_sq.sqrt();
            force += (g * mass(i) * mass(j) / (dist_sq * dist)) * r_vec;
        }
        force += mass(i) * grav;
        force += charge(i) * (ef + v_i.cross(bf));
        force += self.ext_force[i];
        force
    }

    fn mass(&self, i: usize) -> f64 {
        self.p.borrow()[P_FIXED + i]
    }
}

/// Translational ODE right-hand side for the CVODES path.
/// Layout `[q(3N) | v(3N)]`.
fn sens_rhs(
    _t: sunrealtype,
    y: &N_Vector,
    ydot: &N_Vector,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    let c = match user_data.as_mut().and_then(|b| b.downcast_mut::<SensCommon>()) {
        Some(c) => c,
        None => return -1,
    };
    let yv = match N_VGetArrayPointer(y) {
        Some(d) => d,
        None => return -1,
    };
    let mut og = match N_VGetArrayPointer(ydot) {
        Some(d) => d,
        None => return -1,
    };
    let n = c.n;
    let (yv, out) = (&yv[..], &mut og[..]);
    let (q, v) = (&yv[0..3 * n], &yv[3 * n..6 * n]);
    for k in 0..3 * n {
        out[k] = v[k];
    }
    for i in 0..n {
        let a = if c.anchors.translation_fixed[i] {
            Vec3::zeros()
        } else {
            c.force(q, v, i) * (1.0 / c.mass(i))
        };
        write3(out, 3 * (n + i), a);
    }
    0
}

/// GGL index-2 residual for the IDAS path — the same equations as
/// `integrate::dae_residual`, with the parameters read live from `p`.
fn sens_residual(
    _t: sunrealtype,
    yy: &N_Vector,
    yp: &N_Vector,
    rr: &N_Vector,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    let c = match user_data.as_mut().and_then(|b| b.downcast_mut::<SensCommon>()) {
        Some(c) => c,
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
    let (n, m) = (c.n, c.m);
    let (y, ypv, r) = (&y[..], &ypv[..], &mut rg[..]);
    let q = &y[0..3 * n];
    let v = &y[3 * n..6 * n];
    let lam = &y[6 * n..6 * n + m];
    let mu = &y[6 * n + m..6 * n + 2 * m];

    /* Sensitivity runs the translational dynamics, so every body's
     * orientation is the (fixed) one it started with — `gate` has
     * already refused an orientation-gripping joint. */
    let pose: Vec<Pose> = (0..n)
        .map(|i| Pose { position: read3(q, 3 * i), orientation: c.orientations[i] })
        .collect();
    let vel: Vec<Vec3> = (0..n).map(|i| read3(v, 3 * i)).collect();
    let zero_w = vec![Vec3::zeros(); n];
    let (mut fmu, mut tmu) = (vec![Vec3::zeros(); n], vec![Vec3::zeros(); n]);
    let (mut flam, mut tlam) = (vec![Vec3::zeros(); n], vec![Vec3::zeros(); n]);
    if m > 0 {
        c.constraints.add_jacobian_transpose(&pose, &c.anchors, mu, &mut fmu, &mut tmu);
        c.constraints.add_jacobian_transpose(&pose, &c.anchors, lam, &mut flam, &mut tlam);
    }
    for i in 0..n {
        /* GGL projection is mass-weighted: q̇ = v - M⁻¹J_vᵀμ. With one
         * mass it is only a rescaling of μ, but it is what the equation
         * says — see the note in `integrate::dae_residual`. */
        let d = fmu[i] * (1.0 / c.mass(i));
        r[3 * i] = ypv[3 * i] - (v[3 * i] - d.x);
        r[3 * i + 1] = ypv[3 * i + 1] - (v[3 * i + 1] - d.y);
        r[3 * i + 2] = ypv[3 * i + 2] - (v[3 * i + 2] - d.z);
    }
    for i in 0..n {
        let b = 3 * n + 3 * i;
        if c.anchors.translation_fixed[i] {
            r[b] = ypv[b];
            r[b + 1] = ypv[b + 1];
            r[b + 2] = ypv[b + 2];
            continue;
        }
        let f = c.force(q, v, i);
        let mi = c.mass(i);
        r[b] = mi * ypv[b] - f.x + flam[i].x;
        r[b + 1] = mi * ypv[b + 1] - f.y + flam[i].y;
        r[b + 2] = mi * ypv[b + 2] - f.z + flam[i].z;
    }
    if m > 0 {
        let (_, tail) = r.split_at_mut(6 * n);
        let (gblk, gdblk) = tail.split_at_mut(m);
        c.constraints.residual(&pose, gblk);
        c.constraints.velocity_residual(&pose, &vel, &zero_w, gdblk);
    }
    0
}

/// Builds the full parameter vector and the `plist` that selects the
/// requested subset.
fn build_params(
    system: &PhysicalObjectSystem,
    wanted: &[SensParam],
) -> (Rc<RefCell<Vec<f64>>>, Vec<i32>, Vec<f64>) {
    let n = system.objects.len();
    let mut p = vec![0.0; P_FIXED + 2 * n];
    p[P_G] = system.g_constant;
    for (a, val) in [
        system.uniform_gravity.x,
        system.uniform_gravity.y,
        system.uniform_gravity.z,
    ]
    .into_iter()
    .enumerate()
    {
        p[P_GRAV + a] = val;
    }
    for (a, val) in [system.e_field.x, system.e_field.y, system.e_field.z]
        .into_iter()
        .enumerate()
    {
        p[P_E + a] = val;
    }
    for (a, val) in [system.b_field.x, system.b_field.y, system.b_field.z]
        .into_iter()
        .enumerate()
    {
        p[P_B + a] = val;
    }
    for (k, o) in system.objects.iter().enumerate() {
        p[P_FIXED + k] = o.get_mass();
        p[P_FIXED + n + k] = o.get_charge();
    }
    let plist: Vec<i32> = wanted.iter().map(|w| w.slot(n) as i32).collect();
    /* pbar scales each parameter to O(1) so the internal difference
     * quotient has a sensible step. A parameter that is exactly zero has
     * no scale of its own; 1 is the only defensible choice. */
    let pbar: Vec<f64> = plist
        .iter()
        .map(|&i| {
            let v = p[i as usize].abs();
            if v > 0.0 {
                v
            } else {
                1.0
            }
        })
        .collect();
    (Rc::new(RefCell::new(p)), plist, pbar)
}

fn common(system: &PhysicalObjectSystem, p: Rc<RefCell<Vec<f64>>>) -> SensCommon {
    SensCommon {
        n: system.objects.len(),
        m: system.constraints.len(),
        softening: system.softening,
        ext_force: system.external_forces.clone(),
        anchors: Anchors::of(system),
        orientations: system.objects.iter().map(|o| o.get_orientation().normalize()).collect(),
        constraints: system.constraints.clone(),
        p,
    }
}

/// Integrates `system` to `t_end` **and** the sensitivity of the final
/// state with respect to each parameter in `wanted`.
///
/// Picks IDAS when the system is constrained and CVODES when it is not.
/// The system itself is advanced in place, exactly as [`crate::integrate::run`]
/// would advance it — a sensitivity run is a real run that also carries
/// derivatives.
pub fn run(
    system: &mut PhysicalObjectSystem,
    t_end: f64,
    wanted: &[SensParam],
) -> Result<SensitivityReport, String> {
    if wanted.is_empty() {
        return Err(
            "SENSITIVITY needs at least one parameter, e.g. `SENSITIVITY g_constant`".to_string(),
        );
    }
    gate(system)?;
    let t0 = system.time;
    if t_end <= t0 {
        return Err(format!("t_end ({t_end}) must be greater than current time ({t0})"));
    }
    let n = system.objects.len();
    if n == 0 {
        return Err("SENSITIVITY needs at least one object".to_string());
    }
    if system.constraints.is_empty() {
        run_cvodes(system, t_end, wanted)
    } else {
        /* Rotational joints were refused by `gate` above; what is left
         * is rods, which the translational DAE holds. */
        run_idas(system, t_end, wanted)
    }
}

fn run_cvodes(
    system: &mut PhysicalObjectSystem,
    t_end: f64,
    wanted: &[SensParam],
) -> Result<SensitivityReport, String> {
    use cvodes_rs::cvodes::{
        CVode, CVodeCreate, CVodeFree, CVodeGetSens, CVodeInit, CVodeSStolerances,
        CVodeSensEEtolerances, CVodeSensInit,
    };
    use cvodes_rs::cvodes_impl::{CV_BDF, CV_NORMAL, CV_SIMULTANEOUS, CV_SUCCESS};
    use cvodes_rs::cvodes_io::{
        CVodeGetNumSteps, CVodeSetMaxNumSteps, CVodeSetSensParams, CVodeSetUserData,
    };
    use cvodes_rs::cvodes_ls::CVodeSetLinearSolver;

    let n = system.objects.len();
    let neq = 6 * n;
    let t0 = system.time;
    let ns = wanted.len();

    let mut ctx_out: Option<SUNContext> = None;
    let rc = SUNContext_Create(SUN_COMM_NULL, &mut ctx_out);
    if rc != 0 {
        return Err(format!("SUNContext_Create failed: {rc}"));
    }
    let sunctx = ctx_out.ok_or_else(|| "SUNContext_Create returned NULL".to_string())?;

    let y = N_VNew_Serial(neq as i64, &sunctx)
        .ok_or_else(|| format!("N_VNew_Serial({neq}) returned NULL"))?;
    with_data_mut(&y, |d| {
        for (i, o) in system.objects.iter().enumerate() {
            write3(d, 3 * i, o.get_position());
            write3(d, 3 * (n + i), o.get_velocity());
        }
    })
    .ok_or_else(|| no_array("y"))?;

    let (p, plist, pbar) = build_params(system, wanted);
    let cvode_mem =
        CVodeCreate(CV_BDF, &sunctx).ok_or_else(|| "CVodeCreate returned NULL".to_string())?;
    let mut rc = CVodeInit(&cvode_mem, sens_rhs, t0, &y);
    if rc != CV_SUCCESS {
        return Err(format!("CVodeInit failed: {rc}"));
    }
    rc = CVodeSStolerances(&cvode_mem, system.rtol, system.atol);
    if rc != CV_SUCCESS {
        return Err(format!("CVodeSStolerances failed: {rc}"));
    }
    rc = CVodeSetUserData(&cvode_mem, Some(Box::new(common(system, Rc::clone(&p)))));
    if rc != CV_SUCCESS {
        return Err(format!("CVodeSetUserData failed: {rc}"));
    }
    rc = CVodeSetMaxNumSteps(&cvode_mem, 500_000);
    if rc != CV_SUCCESS {
        return Err(format!("CVodeSetMaxNumSteps failed: {rc}"));
    }
    let a = SUNDenseMatrix(neq as i64, neq as i64, &sunctx)
        .ok_or_else(|| "SUNDenseMatrix returned NULL".to_string())?;
    let ls = SUNLinSol_Dense(&y, &a, &sunctx)
        .ok_or_else(|| "SUNLinSol_Dense returned NULL".to_string())?;
    rc = CVodeSetLinearSolver(&cvode_mem, &ls, Some(&a));
    if rc != CV_SUCCESS {
        return Err(format!("CVodeSetLinearSolver failed: {rc}"));
    }

    /* Sensitivity vectors start at zero: the initial state does not
     * depend on any of these parameters. */
    let yS = N_VCloneVectorArray(ns as i32, &y)
        .ok_or_else(|| "N_VCloneVectorArray returned NULL".to_string())?;
    for v in &yS {
        N_VConst(0.0, v);
    }
    /* fS = None -> CVODES forms the sensitivity right-hand side itself by
     * difference quotients, perturbing `p` and re-calling `sens_rhs`. */
    rc = CVodeSensInit(&cvode_mem, ns as i32, CV_SIMULTANEOUS, None, &yS);
    if rc != CV_SUCCESS {
        return Err(format!("CVodeSensInit failed: {rc}"));
    }
    rc = CVodeSensEEtolerances(&cvode_mem);
    if rc != CV_SUCCESS {
        return Err(format!("CVodeSensEEtolerances failed: {rc}"));
    }
    rc = CVodeSetSensParams(&cvode_mem, Some(Rc::clone(&p)), Some(&pbar), Some(&plist));
    if rc != CV_SUCCESS {
        return Err(format!("CVodeSetSensParams failed: {rc}"));
    }

    let mut t = t0;
    let rc = CVode(&cvode_mem, t_end, &y, &mut t, CV_NORMAL);
    if rc < 0 {
        return Err(format!("CVode failed with retval = {rc} at t = {t}"));
    }
    let mut tret = t;
    let rc = CVodeGetSens(&cvode_mem, &mut tret, &yS);
    if rc != CV_SUCCESS {
        return Err(format!("CVodeGetSens failed: {rc}"));
    }

    with_data(&y, |d| {
        for (i, o) in system.objects.iter_mut().enumerate() {
            o.set_position(read3(d, 3 * i));
            o.set_velocity(read3(d, 3 * (n + i)));
        }
    })
    .ok_or_else(|| no_array("y"))?;
    system.time = t;

    let per_param = collect(&yS, wanted, n)?;
    let mut nst = 0i64;
    CVodeGetNumSteps(&cvode_mem, &mut nst);

    let mut cvode_mem = Some(cvode_mem);
    CVodeFree(&mut cvode_mem);
    SUNLinSolFree(Some(ls));
    SUNMatDestroy(a);
    N_VDestroyVectorArray(yS, ns as i32);
    N_VDestroy(y);
    let mut sunctx = Some(sunctx);
    SUNContext_Free(&mut sunctx);
    Ok(SensitivityReport { t, per_param, solver: "CVODES", nst })
}

fn run_idas(
    system: &mut PhysicalObjectSystem,
    t_end: f64,
    wanted: &[SensParam],
) -> Result<SensitivityReport, String> {
    use idas_rs::idas::{
        IDACreate, IDAFree, IDAGetSens, IDAInit, IDASStolerances, IDASensEEtolerances,
        IDASensInit, IDASolve,
    };
    use idas_rs::idas_ic::IDACalcIC;
    use idas_rs::idas_impl::{IDA_NORMAL, IDA_SIMULTANEOUS, IDA_SUCCESS, IDA_YA_YDP_INIT};
    use idas_rs::idas_io::{
        IDAGetNumSteps, IDASetId, IDASetMaxNumSteps, IDASetSensParams, IDASetSuppressAlg,
        IDASetUserData,
    };
    use idas_rs::idas_ls::IDASetLinearSolver;

    let n = system.objects.len();
    let m = system.constraints.len();
    let neq = 6 * n + 2 * m;
    let t0 = system.time;
    let ns = wanted.len();

    let mut ctx_out: Option<SUNContext> = None;
    let rc = SUNContext_Create(SUN_COMM_NULL, &mut ctx_out);
    if rc != 0 {
        return Err(format!("SUNContext_Create failed: {rc}"));
    }
    let sunctx = ctx_out.ok_or_else(|| "SUNContext_Create returned NULL".to_string())?;

    let yy = N_VNew_Serial(neq as i64, &sunctx)
        .ok_or_else(|| format!("N_VNew_Serial({neq}) returned NULL"))?;
    let yp = N_VNew_Serial(neq as i64, &sunctx)
        .ok_or_else(|| format!("N_VNew_Serial({neq}) returned NULL"))?;
    let id = N_VNew_Serial(neq as i64, &sunctx)
        .ok_or_else(|| format!("N_VNew_Serial({neq}) returned NULL"))?;

    let (p, plist, pbar) = build_params(system, wanted);
    let c = common(system, Rc::clone(&p));

    with_data_mut(&yy, |d| {
        for (i, o) in system.objects.iter().enumerate() {
            write3(d, 3 * i, o.get_position());
            write3(d, 3 * (n + i), o.get_velocity());
        }
        for k in 6 * n..neq {
            d[k] = 0.0;
        }
    })
    .ok_or_else(|| no_array("yy"))?;
    let q0: Vec<f64> = with_data(&yy, |d| d[0..3 * n].to_vec()).ok_or_else(|| no_array("yy"))?;
    let v0: Vec<f64> = with_data(&yy, |d| d[3 * n..6 * n].to_vec()).ok_or_else(|| no_array("yy"))?;
    with_data_mut(&yp, |d| {
        d[0..3 * n].copy_from_slice(&v0);
        for i in 0..n {
            let a = if c.anchors.translation_fixed[i] {
                Vec3::zeros()
            } else {
                c.force(&q0, &v0, i) * (1.0 / c.mass(i))
            };
            write3(d, 3 * (n + i), a);
        }
        for k in 6 * n..neq {
            d[k] = 0.0;
        }
    })
    .ok_or_else(|| no_array("yp"))?;
    with_data_mut(&id, |d| {
        for k in 0..6 * n {
            d[k] = 1.0;
        }
        for k in 6 * n..neq {
            d[k] = 0.0;
        }
    })
    .ok_or_else(|| no_array("id"))?;

    let ida_mem = IDACreate(&sunctx).ok_or_else(|| "IDACreate returned NULL".to_string())?;
    let mut rc = IDAInit(&ida_mem, sens_residual, t0, &yy, &yp);
    if rc != IDA_SUCCESS {
        return Err(format!("IDAInit failed: {rc}"));
    }
    rc = IDASStolerances(&ida_mem, system.rtol, system.atol);
    if rc != IDA_SUCCESS {
        return Err(format!("IDASStolerances failed: {rc}"));
    }
    rc = IDASetUserData(&ida_mem, Some(Box::new(c)));
    if rc != IDA_SUCCESS {
        return Err(format!("IDASetUserData failed: {rc}"));
    }
    rc = IDASetId(&ida_mem, Some(&id));
    if rc != IDA_SUCCESS {
        return Err(format!("IDASetId failed: {rc}"));
    }
    rc = IDASetSuppressAlg(&ida_mem, true);
    if rc != IDA_SUCCESS {
        return Err(format!("IDASetSuppressAlg failed: {rc}"));
    }
    rc = IDASetMaxNumSteps(&ida_mem, 500_000);
    if rc != IDA_SUCCESS {
        return Err(format!("IDASetMaxNumSteps failed: {rc}"));
    }
    let a = SUNDenseMatrix(neq as i64, neq as i64, &sunctx)
        .ok_or_else(|| "SUNDenseMatrix returned NULL".to_string())?;
    let ls = SUNLinSol_Dense(&yy, &a, &sunctx)
        .ok_or_else(|| "SUNLinSol_Dense returned NULL".to_string())?;
    rc = IDASetLinearSolver(&ida_mem, &ls, Some(&a));
    if rc != IDA_SUCCESS {
        return Err(format!("IDASetLinearSolver failed: {rc}"));
    }

    let yS = N_VCloneVectorArray(ns as i32, &yy)
        .ok_or_else(|| "N_VCloneVectorArray returned NULL".to_string())?;
    let ypS = N_VCloneVectorArray(ns as i32, &yy)
        .ok_or_else(|| "N_VCloneVectorArray returned NULL".to_string())?;
    for v in &yS {
        N_VConst(0.0, v);
    }
    for v in &ypS {
        N_VConst(0.0, v);
    }
    rc = IDASensInit(&ida_mem, ns as i32, IDA_SIMULTANEOUS, None, &yS, &ypS);
    if rc != IDA_SUCCESS {
        return Err(format!("IDASensInit failed: {rc}"));
    }
    rc = IDASensEEtolerances(&ida_mem);
    if rc != IDA_SUCCESS {
        return Err(format!("IDASensEEtolerances failed: {rc}"));
    }
    rc = IDASetSensParams(&ida_mem, Some(Rc::clone(&p)), Some(&pbar), Some(&plist));
    if rc != IDA_SUCCESS {
        return Err(format!("IDASetSensParams failed: {rc}"));
    }

    rc = IDACalcIC(&ida_mem, IDA_YA_YDP_INIT, t_end);
    if rc != IDA_SUCCESS {
        return Err(format!(
            "IDACalcIC failed: {rc} — the initial configuration is inconsistent with the \
             constraints"
        ));
    }

    let mut t = t0;
    let rc = IDASolve(&ida_mem, t_end, &mut t, &yy, &yp, IDA_NORMAL);
    if rc < 0 {
        return Err(format!("IDASolve failed with retval = {rc} at t = {t}"));
    }
    let mut tret = t;
    let rc = IDAGetSens(&ida_mem, &mut tret, &yS);
    if rc != IDA_SUCCESS {
        return Err(format!("IDAGetSens failed: {rc}"));
    }

    with_data(&yy, |d| {
        for (i, o) in system.objects.iter_mut().enumerate() {
            o.set_position(read3(d, 3 * i));
            o.set_velocity(read3(d, 3 * (n + i)));
        }
    })
    .ok_or_else(|| no_array("yy"))?;
    system.time = t;

    let per_param = collect(&yS, wanted, n)?;
    let mut nst = 0i64;
    IDAGetNumSteps(&ida_mem, &mut nst);

    let mut ida_mem = Some(ida_mem);
    IDAFree(&mut ida_mem);
    SUNLinSolFree(Some(ls));
    SUNMatDestroy(a);
    N_VDestroyVectorArray(yS, ns as i32);
    N_VDestroyVectorArray(ypS, ns as i32);
    N_VDestroy(yy);
    N_VDestroy(yp);
    N_VDestroy(id);
    let mut sunctx = Some(sunctx);
    SUNContext_Free(&mut sunctx);
    Ok(SensitivityReport { t, per_param, solver: "IDAS", nst })
}

/// Reads the sensitivity vectors into per-object derivatives. The
/// multiplier tail of the DAE layout is simply not read — `∂λ/∂p` is a
/// real number but not one anybody asked for.
fn collect(
    yS: &[N_Vector],
    wanted: &[SensParam],
    n: usize,
) -> Result<Vec<ParamSensitivity>, String> {
    let mut out = Vec::with_capacity(wanted.len());
    for (k, param) in wanted.iter().enumerate() {
        let (dp, dv) = with_data(&yS[k], |d| {
            let dp: Vec<Vec3> = (0..n).map(|i| read3(d, 3 * i)).collect();
            let dv: Vec<Vec3> = (0..n).map(|i| read3(d, 3 * (n + i))).collect();
            (dp, dv)
        })
        .ok_or_else(|| no_array("yS"))?;
        out.push(ParamSensitivity { param: *param, d_position: dp, d_velocity: dv });
    }
    Ok(out)
}

fn read3(d: &[f64], at: usize) -> Vec3 {
    Vec3::new(d[at], d[at + 1], d[at + 2])
}

fn write3(d: &mut [f64], at: usize, v: Vec3) {
    d[at] = v.x;
    d[at + 1] = v.y;
    d[at + 2] = v.z;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parameter_names_round_trip_and_bad_ones_explain_themselves() {
        assert_eq!(SensParam::parse("g_constant", 2).unwrap(), SensParam::GConstant);
        assert_eq!(SensParam::parse("mass 1", 2).unwrap(), SensParam::Mass(1));
        assert_eq!(SensParam::parse("mass obj1", 2).unwrap(), SensParam::Mass(1));
        assert_eq!(SensParam::parse("GRAVITY.Y", 2).unwrap(), SensParam::Gravity(1));
        assert_eq!(SensParam::parse("b_field.z", 2).unwrap(), SensParam::BField(2));
        assert_eq!(SensParam::Mass(1).to_string(), "mass 1");
        assert_eq!(SensParam::Gravity(1).to_string(), "gravity.y");

        let e = SensParam::parse("mass 7", 2).unwrap_err();
        assert!(e.contains("only 2 object"), "{e}");
        let e = SensParam::parse("wobble", 2).unwrap_err();
        assert!(e.contains("expected g_constant"), "{e}");
    }

    /// The slot layout must be injective — two parameters sharing a slot
    /// would silently differentiate the wrong thing.
    #[test]
    fn every_parameter_gets_its_own_slot() {
        let n = 3;
        let mut seen = std::collections::BTreeSet::new();
        let mut all = vec![SensParam::GConstant];
        for a in 0..3 {
            all.push(SensParam::Gravity(a));
            all.push(SensParam::EField(a));
            all.push(SensParam::BField(a));
        }
        for k in 0..n {
            all.push(SensParam::Mass(k));
            all.push(SensParam::Charge(k));
        }
        for p in &all {
            assert!(seen.insert(p.slot(n)), "slot collision at {p}");
        }
        assert_eq!(seen.len(), all.len());
        assert!(*seen.iter().max().unwrap() < P_FIXED + 2 * n);
    }
}
