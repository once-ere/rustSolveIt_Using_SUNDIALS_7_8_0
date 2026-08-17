//! Port of `src/arkode/arkode_mristep.c` with `src/arkode/arkode_mristep_impl.h`
//! folded in (module = C file base name; the impl header's data structures,
//! constants and error messages live here and every other MRIStep part —
//! `arkode_mristep_io.rs`, `arkode_mristep_nls.rs`,
//! `arkode_mristep_controller.rs` — reaches them through
//! `use crate::arkode_mristep::*;`).
//!
//! Reference build configuration: SUNDIALS_LOGGING_LEVEL = 2, so every
//! `SUNLogInfo` / `SUNLogInfoIf` / `SUNLogDebug` / `SUNLogExtraDebug*` call
//! site is omitted at translation time (MRIStep queues no `ARK_WARNING`
//! messages). Profiling off, error checks off (`SUNAssert` / `SUNCheck*` are
//! no-ops), monitoring on, serial branches only.
//!
//! Binding notes specific to this module:
//!  * The MRIStep content record lives BY VALUE in `ark_mem.step_mem`
//!    (`Option<Box<dyn Any>>` = C `void* step_mem`) and is reached through
//!    the single downcast helper [`mriStep_mem_mut`]. That guard IS a borrow
//!    of `ark_mem`: it is never held across `arkProcessError`, a user
//!    callback, an `N_Vector` / linear-solver / nonlinear-solver operation,
//!    an inner-stepper call, or a second borrow of the same mem.
//!  * C `step_mem->lmem` lives in `ark_mem.ark_lmem` (contract §4), so
//!    `mriStep_GetLmem` is a presence probe and `mriStep_AttachLinsol` moves
//!    the ARKLS record into `ark_mem.ark_lmem`.
//!  * C `step_mem->jcur` is the shared [`ARKJcurPtr`] cell so that a
//!    preconditioner-setup routine reached re-entrantly through
//!    `arkLsSetup` writes through the same flag `mriStep_GetGammas` handed
//!    out (contract §"THE jcur SEAM").
//!  * The fused-op scratch arrays keep the C shape: `cvals` is a
//!    `Vec<sunrealtype>` sized `nfusedopvecs` and `Xvecs` a
//!    `Vec<Option<N_Vector>>` of the same length (C `calloc` leaves NULL
//!    slots, which map to `None`). [`mriStep_xvecs`] materialises the dense
//!    `&[N_Vector]` the fused vector kernels take.
//!  * Rust-forced renames: `step_mem->crate` is `crate_` and
//!    `MRIC->type` is `type_` (both are Rust keywords); `ark_mem->fn` is
//!    `ark_mem.fn_` (contract).

use std::any::Any;
use std::cell::{Cell, RefCell, RefMut};
use std::rc::Rc;

use crate::arkode::*;
use crate::arkode_impl::*;
use crate::arkode_io::*;
use crate::arkode_ls::*;
use crate::arkode_mri_tables::*;
use crate::arkode_mristep_io::*;
use crate::arkode_mristep_nls::*;
use sundials_core::sundials_adaptcontroller::*;
use sundials_core::sundials_context::SUNContext;
use sundials_core::sundials_errors::*;
use sundials_core::sundials_linearsolver::SUNLinearSolver_Type;
use sundials_core::sundials_math::*;
use sundials_core::sundials_nonlinearsolver::*;
use sundials_core::sundials_nvector::*;
use sundials_core::sundials_stepper::*;
use sundials_core::sundials_types::*;
use sundials_core::sundials_utils::*;
use sundials_core::sunnonlinsol_newton::SUNNonlinSol_Newton;

/*===============================================================
  MRIStep constants (arkode_mristep_impl.h)
  ===============================================================*/

/* Stage type identifiers */
pub const MRISTAGE_FIRST: i32 = -2;
pub const MRISTAGE_STIFF_ACC: i32 = -1;
pub const MRISTAGE_ERK_FAST: i32 = 0;
pub const MRISTAGE_ERK_NOFAST: i32 = 1;
pub const MRISTAGE_DIRK_NOFAST: i32 = 2;
pub const MRISTAGE_DIRK_FAST: i32 = 3;

/* Default MRI coupling tables, by method order and type
   (`include/arkode/arkode_mristep.h` lines 101-123). The table IDs
   themselves live in `crate::arkode_mri_tables`. */
pub const MRISTEP_DEFAULT_EXPL_1: i32 = ARKODE_MRI_GARK_FORWARD_EULER;
pub const MRISTEP_DEFAULT_EXPL_2: i32 = ARKODE_MRI_GARK_ERK22b;
pub const MRISTEP_DEFAULT_EXPL_3: i32 = ARKODE_MIS_KW3;
pub const MRISTEP_DEFAULT_EXPL_4: i32 = ARKODE_MRI_GARK_ERK45a;

pub const MRISTEP_DEFAULT_EXPL_2_AD: i32 = ARKODE_MRI_GARK_ERK22b;
pub const MRISTEP_DEFAULT_EXPL_3_AD: i32 = ARKODE_MRI_GARK_ERK33a;
pub const MRISTEP_DEFAULT_EXPL_4_AD: i32 = ARKODE_MRI_GARK_ERK45a;
pub const MRISTEP_DEFAULT_EXPL_5_AD: i32 = ARKODE_MERK54;

pub const MRISTEP_DEFAULT_IMPL_SD_1: i32 = ARKODE_MRI_GARK_BACKWARD_EULER;
pub const MRISTEP_DEFAULT_IMPL_SD_2: i32 = ARKODE_MRI_GARK_IRK21a;
pub const MRISTEP_DEFAULT_IMPL_SD_3: i32 = ARKODE_MRI_GARK_ESDIRK34a;
pub const MRISTEP_DEFAULT_IMPL_SD_4: i32 = ARKODE_MRI_GARK_ESDIRK46a;

pub const MRISTEP_DEFAULT_IMEX_SD_1: i32 = ARKODE_IMEX_MRI_GARK_EULER;
pub const MRISTEP_DEFAULT_IMEX_SD_2: i32 = ARKODE_IMEX_MRI_GARK_TRAPEZOIDAL;
pub const MRISTEP_DEFAULT_IMEX_SD_3: i32 = ARKODE_IMEX_MRI_GARK3b;
pub const MRISTEP_DEFAULT_IMEX_SD_4: i32 = ARKODE_IMEX_MRI_GARK4;

pub const MRISTEP_DEFAULT_IMEX_SD_2_AD: i32 = ARKODE_IMEX_MRI_SR21;
pub const MRISTEP_DEFAULT_IMEX_SD_3_AD: i32 = ARKODE_IMEX_MRI_SR32;
pub const MRISTEP_DEFAULT_IMEX_SD_4_AD: i32 = ARKODE_IMEX_MRI_SR43;

/* The implicit-solver constants MAXCOR / CRDOWN / DGMAX / RDIV / MSBP /
   NLSCOEF are byte-identical duplicates of the ARKStep ones and live in
   `arkode_impl.rs` (contract §7). */

/*===============================================================
  Reusable MRIStep Error Messages (arkode_mristep_impl.h)
  ===============================================================*/

pub const MSG_MRISTEP_NO_MEM: &str = "Time step module memory is NULL.";
pub const MSG_NLS_INIT_FAIL: &str = "The nonlinear solver's init routine failed.";
pub const MSG_MRISTEP_NO_COUPLING: &str = "The MRIStepCoupling is NULL.";

/*===============================================================
  MRIStep user-supplied function types (arkode/arkode_mristep.h)
  ===============================================================*/

pub type MRIStepPreInnerFn = fn(
    t: sunrealtype,
    f_1d: &[N_Vector],
    nvecs: i32,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32;

pub type MRIStepPostInnerFn =
    fn(t: sunrealtype, y: &N_Vector, user_data: &mut Option<Box<dyn Any>>) -> i32;

/*===============================================================
  MRIStep inner-stepper function types (arkode/arkode_mristep.h)
  ===============================================================*/

pub type MRIStepInnerEvolveFn = fn(
    stepper: &MRIStepInnerStepper,
    t0: sunrealtype,
    tout: sunrealtype,
    y: &N_Vector,
) -> i32;

pub type MRIStepInnerFullRhsFn = fn(
    stepper: &MRIStepInnerStepper,
    t: sunrealtype,
    y: &N_Vector,
    f: &N_Vector,
    mode: i32,
) -> i32;

pub type MRIStepInnerResetFn =
    fn(stepper: &MRIStepInnerStepper, tR: sunrealtype, yR: &N_Vector) -> i32;

pub type MRIStepInnerGetAccumulatedError =
    fn(stepper: &MRIStepInnerStepper, accum_error: &mut sunrealtype) -> i32;

pub type MRIStepInnerResetAccumulatedError = fn(stepper: &MRIStepInnerStepper) -> i32;

pub type MRIStepInnerSetRTol = fn(stepper: &MRIStepInnerStepper, rtol: sunrealtype) -> i32;

/*===============================================================
  MRI inner time stepper data structure (arkode_mristep_impl.h)
  ===============================================================*/

#[derive(Default, Clone)]
pub struct _MRIStepInnerStepper_Ops {
    pub evolve: Option<MRIStepInnerEvolveFn>,
    pub fullrhs: Option<MRIStepInnerFullRhsFn>,
    pub reset: Option<MRIStepInnerResetFn>,
    pub geterror: Option<MRIStepInnerGetAccumulatedError>,
    pub reseterror: Option<MRIStepInnerResetAccumulatedError>,
    pub setrtol: Option<MRIStepInnerSetRTol>,
}

pub type MRIStepInnerStepper_Ops = _MRIStepInnerStepper_Ops;

/// C `struct _MRIStepInnerStepper`. Handle model: `Rc` clone = C pointer
/// copy, `Rc::ptr_eq` = C pointer equality; every mutable field carries its
/// own `Cell`/`RefCell` so a holder of the handle can write through it
/// exactly as C writes through the pointer.
///
/// `content` is C `void* content`: every in-tree producer stores a SUNDIALS
/// handle there (`ARKodeMem` from `ARKodeCreateMRIStepInnerStepper`,
/// `SUNStepper` from `MRIStepInnerStepper_CreateFromSUNStepper`), so it is a
/// `Box<dyn Any>` exactly as `SUNStepper_::content` is;
/// [`MRIStepInnerStepper_GetContentAs`] clones the handle out of it.
pub struct _MRIStepInnerStepper {
    /* stepper specific content and operations */
    pub content: RefCell<Option<Box<dyn Any>>>,
    /// C `void* python` (Python bindings are out of scope); always `None`.
    pub python: RefCell<Option<Box<dyn Any>>>,
    pub ops: RefCell<MRIStepInnerStepper_Ops>,

    /* stepper context */
    pub sunctx: RefCell<SUNContext>,

    /* base class data */
    pub forcing: RefCell<Vec<N_Vector>>, /* array of forcing vectors            */
    pub nforcing: RefCell<i32>,          /* number of forcing vectors active    */
    pub nforcing_allocated: RefCell<i32>, /* number of forcing vectors allocated */
    pub last_flag: RefCell<i32>,         /* last stepper return flag            */
    pub tshift: RefCell<sunrealtype>,    /* time normalization shift            */
    pub tscale: RefCell<sunrealtype>,    /* time normalization scaling          */

    /* fused op workspace */
    pub vals: RefCell<Vec<sunrealtype>>,
    pub vecs: RefCell<Vec<Option<N_Vector>>>,

    /* Space requirements */
    pub lrw1: RefCell<sunindextype>,
    pub liw1: RefCell<sunindextype>,
    pub lrw: RefCell<i64>,
    pub liw: RefCell<i64>,
}

pub type MRIStepInnerStepper = Rc<_MRIStepInnerStepper>;

impl _MRIStepInnerStepper {
    /// C `malloc` + `memset(*stepper, 0, sizeof(**stepper))` in
    /// `MRIStepInnerStepper_Create` (which then assigns `last_flag`,
    /// `sunctx` and `python`).
    pub fn zeroed(sunctx: SUNContext) -> _MRIStepInnerStepper {
        _MRIStepInnerStepper {
            content: RefCell::new(None),
            python: RefCell::new(None),
            ops: RefCell::new(MRIStepInnerStepper_Ops::default()),
            sunctx: RefCell::new(sunctx),
            forcing: RefCell::new(Vec::new()),
            nforcing: RefCell::new(0),
            nforcing_allocated: RefCell::new(0),
            last_flag: RefCell::new(0),
            tshift: RefCell::new(ZERO),
            tscale: RefCell::new(ZERO),
            vals: RefCell::new(Vec::new()),
            vecs: RefCell::new(Vec::new()),
            lrw1: RefCell::new(0),
            liw1: RefCell::new(0),
            lrw: RefCell::new(0),
            liw: RefCell::new(0),
        }
    }
}

/*===============================================================
  MRI time step module data structure (arkode_mristep_impl.h)
  ===============================================================*/

/// C `struct ARKodeMRIStepMemRec`, held BY VALUE in `ark_mem.step_mem`.
///
/// Deviations from the C layout, all forced and all documented at their use
/// sites: `crate` -> `crate_` (Rust keyword); `void* lmem` is gone (the ARKLS
/// record lives in `ark_mem.ark_lmem`, contract §4); `sunbooleantype jcur` is
/// the shared [`ARKJcurPtr`] cell; the `N_Vector*` arrays `Fse` / `Fsi` /
/// `forcing` are `Vec<N_Vector>` (empty == C `NULL`); the `int*` /
/// `sunrealtype*` arrays are `Vec` (empty == C `NULL`); `Xvecs` is
/// `Vec<Option<N_Vector>>` (C `calloc`'d NULL slots).
pub struct ARKodeMRIStepMemRec {
    /* MRI problem specification */
    pub fse: Option<ARKRhsFn>, /* y' = fse(t,y) + fsi(t,y) + ff(t,y) */
    pub fsi: Option<ARKRhsFn>,
    pub linear: sunbooleantype,         /* SUNTRUE if fi is linear        */
    pub linear_timedep: sunbooleantype, /* SUNTRUE if dfi/dy depends on t */
    pub explicit_rhs: sunbooleantype,   /* SUNTRUE if fse is provided     */
    pub implicit_rhs: sunbooleantype,   /* SUNTRUE if fsi is provided     */
    pub deduce_rhs: sunbooleantype,     /* SUNTRUE if fi is deduced after
                                        a nonlinear solve              */

    /* Outer RK method storage and parameters */
    pub Fse: Vec<N_Vector>,       /* explicit RHS at each stage               */
    pub Fsi: Vec<N_Vector>,       /* implicit RHS at each stage               */
    pub unify_Fs: sunbooleantype, /* Fse and Fsi point at the same memory     */
    pub fse_is_current: sunbooleantype,
    pub fsi_is_current: sunbooleantype,
    pub MRIC: Option<MRIStepCoupling>, /* slow->fast coupling table           */
    pub q: i32,                        /* method order                        */
    pub p: i32,                        /* embedding order                     */
    pub stages: i32,                   /* total number of stages              */
    pub nstages_active: i32,           /* number of active stage RHS vectors  */
    pub nstages_allocated: i32,        /* number of stage RHS vectors alloc'd */
    pub stage_map: Vec<i32>,           /* index map for stage RHS vectors     */
    pub stagetypes: Vec<i32>,          /* type flags for stages               */
    pub Ae_row: Vec<sunrealtype>,      /* equivalent explicit RK coeffs       */
    pub Ai_row: Vec<sunrealtype>,      /* equivalent implicit RK coeffs       */

    /* Algebraic solver data and parameters */
    pub sdata: Option<N_Vector>,          /* old stage data in residual        */
    pub zpred: Option<N_Vector>,          /* predicted stage solution          */
    pub zcor: Option<N_Vector>,           /* stage correction                  */
    pub NLS: Option<SUNNonlinearSolver>,  /* generic SUNNonlinearSolver object */
    pub ownNLS: sunbooleantype,           /* flag indicating ownership of NLS  */
    pub nls_fsi: Option<ARKRhsFn>,        /* fsi(t,y) used in the nonlin solver*/
    pub gamma: sunrealtype,               /* gamma = h * A(i,i)                */
    pub gammap: sunrealtype,              /* gamma at the last setup call      */
    pub gamrat: sunrealtype,              /* gamma / gammap                    */
    pub dgmax: sunrealtype,               /* call lsetup if |gamma/gammap-1| >= dgmax */
    pub predictor: i32,                   /* implicit prediction method to use */
    pub crdown: sunrealtype,              /* nonlin conv rate estimation const */
    pub rdiv: sunrealtype,                /* divergence if delnrm/delnrm_p > rdiv */
    /// C `step_mem->crate` (estimated nonlinear convergence rate); `crate`
    /// is a Rust keyword and cannot even be written raw.
    pub crate_: sunrealtype,
    pub delnrm_p: sunrealtype, /* norm of previous nonlinear solver update */
    pub delnrm: sunrealtype,   /* norm of current nonlinear solver update  */
    pub eRNrm: sunrealtype,    /* estimated residual norm, used in nonlin
                               and linear solver convergence tests      */
    pub nlscoef: sunrealtype,  /* coefficient in nonlin. convergence test  */
    pub msbp: i32,             /* positive => max # steps between lsetup
                               negative => call at each Newton iter     */
    pub nstlp: i64,            /* step number of last setup call           */
    pub maxcor: i32,           /* max num iterations for solving the
                               nonlinear equation                       */
    pub convfail: i32,         /* NLS fail flag (for interface routines)   */
    /// C `sunbooleantype jcur` — is the Jacobian info for the linear solver
    /// current? Shared cell so `step_getgammas` can hand out the same flag
    /// `arkLsSetup` / `arkLsPSetup` write through.
    pub jcur: ARKJcurPtr,
    pub stage_predict: Option<ARKStagePredictFn>, /* User-supplied stage predictor */
    pub istage: i32,                              /* stage index in nonlinear solve */

    /* Informational output for mriStep_GetStageIndex -- note that this
       may differ from istage, since that is used internally by the
       nonlinear solver, and it is manually modified during embedding
       stages to match the last internal stage index. */
    pub cur_stage: i32,

    /* Linear Solver Data (C `void* lmem` lives in `ark_mem.ark_lmem`) */
    pub linit: Option<ARKLinsolInitFn>,
    pub lsetup: Option<ARKLinsolSetupFn>,
    pub lsolve: Option<ARKLinsolSolveFn>,
    pub lfree: Option<ARKLinsolFreeFn>,

    /* Inner stepper */
    pub stepper: Option<MRIStepInnerStepper>,

    /* User-supplied pre and post inner evolve functions */
    pub pre_inner_evolve: Option<MRIStepPreInnerFn>,
    pub post_inner_evolve: Option<MRIStepPostInnerFn>,

    /* MRI adaptivity parameters */
    pub inner_rtol_factor: sunrealtype, /* prev control parameter               */
    pub inner_dsm: sunrealtype,         /* prev inner stepper accumulated error */
    pub inner_rtol_factor_new: sunrealtype, /* upcoming control parameter       */

    /* Counters */
    pub nfse: i64,        /* num fse calls                    */
    pub nfsi: i64,        /* num fsi calls                    */
    pub nsetups: i64,     /* num linear solver setup calls    */
    pub nls_iters: i64,   /* num nonlinear solver iters       */
    pub nls_fails: i64,   /* num nonlinear solver fails       */
    pub inner_fails: i64, /* num recov. inner solver fails    */
    pub nfusedopvecs: i32, /* length of cvals and Xvecs arrays */

    /* Data for using MRIStep with external polynomial forcing */
    pub expforcing: sunbooleantype, /* add forcing to explicit RHS */
    pub impforcing: sunbooleantype, /* add forcing to implicit RHS */
    pub tshift: sunrealtype,        /* time normalization shift    */
    pub tscale: sunrealtype,        /* time normalization scaling  */
    pub forcing: Vec<N_Vector>,     /* array of forcing vectors    */
    pub nforcing: i32,              /* number of forcing vectors   */

    /* Reusable arrays for fused vector operations */
    pub cvals: Vec<sunrealtype>,
    pub Xvecs: Vec<Option<N_Vector>>,
}

impl ARKodeMRIStepMemRec {
    /// C `calloc(1, sizeof(*step_mem))` in `MRIStepCreate`.
    pub fn zeroed() -> ARKodeMRIStepMemRec {
        ARKodeMRIStepMemRec {
            fse: None,
            fsi: None,
            linear: SUNFALSE,
            linear_timedep: SUNFALSE,
            explicit_rhs: SUNFALSE,
            implicit_rhs: SUNFALSE,
            deduce_rhs: SUNFALSE,
            Fse: Vec::new(),
            Fsi: Vec::new(),
            unify_Fs: SUNFALSE,
            fse_is_current: SUNFALSE,
            fsi_is_current: SUNFALSE,
            MRIC: None,
            q: 0,
            p: 0,
            stages: 0,
            nstages_active: 0,
            nstages_allocated: 0,
            stage_map: Vec::new(),
            stagetypes: Vec::new(),
            Ae_row: Vec::new(),
            Ai_row: Vec::new(),
            sdata: None,
            zpred: None,
            zcor: None,
            NLS: None,
            ownNLS: SUNFALSE,
            nls_fsi: None,
            gamma: ZERO,
            gammap: ZERO,
            gamrat: ZERO,
            dgmax: ZERO,
            predictor: 0,
            crdown: ZERO,
            rdiv: ZERO,
            crate_: ZERO,
            delnrm_p: ZERO,
            delnrm: ZERO,
            eRNrm: ZERO,
            nlscoef: ZERO,
            msbp: 0,
            nstlp: 0,
            maxcor: 0,
            convfail: 0,
            jcur: Rc::new(Cell::new(SUNFALSE)),
            stage_predict: None,
            istage: 0,
            cur_stage: 0,
            linit: None,
            lsetup: None,
            lsolve: None,
            lfree: None,
            stepper: None,
            pre_inner_evolve: None,
            post_inner_evolve: None,
            inner_rtol_factor: ZERO,
            inner_dsm: ZERO,
            inner_rtol_factor_new: ZERO,
            nfse: 0,
            nfsi: 0,
            nsetups: 0,
            nls_iters: 0,
            nls_fails: 0,
            inner_fails: 0,
            nfusedopvecs: 0,
            expforcing: SUNFALSE,
            impforcing: SUNFALSE,
            tshift: ZERO,
            tscale: ZERO,
            forcing: Vec::new(),
            nforcing: 0,
            cvals: Vec::new(),
            Xvecs: Vec::new(),
        }
    }
}

/// Downcast helper: view `ark_mem.step_mem` as the MRIStep memory record.
///
/// Panics if no stepper memory is attached or it is not an MRIStep record
/// (C would blindly cast the `void*` — UB maps to a deterministic panic).
/// NEVER hold the returned guard across `arkProcessError`, a user callback,
/// an `N_Vector` / matrix / linear-solver / nonlinear-solver operation, an
/// inner-stepper call, or another borrow of the same `ark_mem`.
pub fn mriStep_mem_mut(ark_mem: &ARKodeMem) -> RefMut<'_, ARKodeMRIStepMemRec> {
    RefMut::map(ark_mem.borrow_mut(), |m| {
        m.step_mem
            .as_mut()
            .expect("step_mem set")
            .downcast_mut::<ARKodeMRIStepMemRec>()
            .expect("MRIStep step memory")
    })
}

/// C `mriStep_AccessStepMem(ark_mem, fname, &step_mem)` reduced to its
/// presence check; the record itself is reached with [`mriStep_mem_mut`] at
/// each use site (contract §3).
fn mriStep_step_mem_ok(ark_mem: &ARKodeMem, fname: &str) -> i32 {
    let missing = ark_mem.borrow().step_mem.is_none();
    if missing {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            fname,
            file!(),
            MSG_MRISTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }
    ARK_SUCCESS
}

/// Materialise the first `nvec` entries of the fused-op vector scratch as
/// the dense `&[N_Vector]` the `N_V*` kernels take (C hands the calloc'd
/// `step_mem->Xvecs` array over directly).
pub fn mriStep_xvecs(step_mem: &ARKodeMRIStepMemRec, nvec: i32) -> Vec<N_Vector> {
    step_mem.Xvecs[..nvec as usize]
        .iter()
        .map(|v| v.clone().expect("Xvecs entry set"))
        .collect()
}

/*===============================================================
  Callback invocation helpers

  C `void* user_data` is `Option<Box<dyn Any>>`: the box is taken out of
  the mem for the duration of the call and restored on EVERY path, and no
  borrow of `ark_mem` is held across the callback.
  ===============================================================*/

/// C `step_mem->fse(t, y, ydot, ark_mem->user_data)`.
pub fn mriStep_CallFse(
    ark_mem: &ARKodeMem,
    t: sunrealtype,
    y: &N_Vector,
    ydot: &N_Vector,
) -> i32 {
    let fse = { mriStep_mem_mut(ark_mem).fse }.expect("fse set");
    let mut user_data = ark_mem.borrow_mut().user_data.take();
    let retval = fse(t, y, ydot, &mut user_data);
    ark_mem.borrow_mut().user_data = user_data;
    retval
}

/// C `step_mem->fsi(t, y, ydot, ark_mem->user_data)`.
pub fn mriStep_CallFsi(
    ark_mem: &ARKodeMem,
    t: sunrealtype,
    y: &N_Vector,
    ydot: &N_Vector,
) -> i32 {
    let fsi = { mriStep_mem_mut(ark_mem).fsi }.expect("fsi set");
    let mut user_data = ark_mem.borrow_mut().user_data.take();
    let retval = fsi(t, y, ydot, &mut user_data);
    ark_mem.borrow_mut().user_data = user_data;
    retval
}

/// C `ark_mem->PreRhsFn(t, y, ark_mem->user_data)`.
pub fn mriStep_CallPreRhsFn(ark_mem: &ARKodeMem, t: sunrealtype, y: &N_Vector) -> i32 {
    let f = ark_mem.borrow().PreRhsFn.expect("PreRhsFn set");
    let mut user_data = ark_mem.borrow_mut().user_data.take();
    let retval = f(t, y, &mut user_data);
    ark_mem.borrow_mut().user_data = user_data;
    retval
}

/// C `ark_mem->PostProcessStageFn(t, y, ark_mem->user_data)`.
pub fn mriStep_CallPostProcessStageFn(
    ark_mem: &ARKodeMem,
    t: sunrealtype,
    y: &N_Vector,
) -> i32 {
    let f = ark_mem.borrow().PostProcessStageFn.expect("PostProcessStageFn set");
    let mut user_data = ark_mem.borrow_mut().user_data.take();
    let retval = f(t, y, &mut user_data);
    ark_mem.borrow_mut().user_data = user_data;
    retval
}

/// C `ark_mem->PostProcessStepFn(t, y, ark_mem->user_data)`.
pub fn mriStep_CallPostProcessStepFn(
    ark_mem: &ARKodeMem,
    t: sunrealtype,
    y: &N_Vector,
) -> i32 {
    let f = ark_mem.borrow().PostProcessStepFn.expect("PostProcessStepFn set");
    let mut user_data = ark_mem.borrow_mut().user_data.take();
    let retval = f(t, y, &mut user_data);
    ark_mem.borrow_mut().user_data = user_data;
    retval
}

/// C `step_mem->stage_predict(t, zpred, ark_mem->user_data)`.
pub fn mriStep_CallStagePredict(
    ark_mem: &ARKodeMem,
    t: sunrealtype,
    zpred: &N_Vector,
) -> i32 {
    let f = { mriStep_mem_mut(ark_mem).stage_predict }.expect("stage_predict set");
    let mut user_data = ark_mem.borrow_mut().user_data.take();
    let retval = f(t, zpred, &mut user_data);
    ark_mem.borrow_mut().user_data = user_data;
    retval
}

/*===============================================================
  Exported functions
  ===============================================================*/

pub fn MRIStepCreate(
    fse: Option<ARKRhsFn>,
    fsi: Option<ARKRhsFn>,
    t0: sunrealtype,
    y0: &N_Vector,
    stepper: &MRIStepInnerStepper,
    sunctx: &SUNContext,
) -> Option<ARKodeMem> {
    let mut retval: i32;

    /* Check that at least one of fse, fsi is supplied and is to be used*/
    if fse.is_none() && fsi.is_none() {
        arkProcessError(
            None,
            ARK_ILL_INPUT,
            line!() as i32,
            "MRIStepCreate",
            file!(),
            MSG_ARK_NULL_F,
        );
        return None;
    }

    /* Check that y0 is supplied: handled by the type system */
    /* Check that stepper is supplied: handled by the type system */
    /* Check that context is supplied: handled by the type system */

    /* Create ark_mem structure and set default values */
    let ark_mem = match arkCreate(sunctx) {
        Some(m) => m,
        None => {
            arkProcessError(
                None,
                ARK_MEM_NULL,
                line!() as i32,
                "MRIStepCreate",
                file!(),
                MSG_ARK_NO_MEM,
            );
            return None;
        }
    };

    /* Allocate ARKodeMRIStepMem structure, and initialize to zero */
    let step_mem = ARKodeMRIStepMemRec::zeroed();

    /* Attach step_mem structure and function pointers to ark_mem */
    {
        let mut m = ark_mem.borrow_mut();
        m.step_attachlinsol = Some(mriStep_AttachLinsol);
        m.step_disablelsetup = Some(mriStep_DisableLSetup);
        m.step_getlinmem = Some(mriStep_GetLmem);
        m.step_getimplicitrhs = Some(mriStep_GetImplicitRHS);
        m.step_getgammas = Some(mriStep_GetGammas);
        m.step_init = Some(mriStep_Init);
        m.step_fullrhs = Some(mriStep_FullRHS);
        m.step = Some(mriStep_TakeStepMRIGARK);
        m.step_setuserdata = Some(mriStep_SetUserData);
        m.step_printallstats = Some(mriStep_PrintAllStats);
        m.step_writeparameters = Some(mriStep_WriteParameters);
        m.step_setusecompensatedsums = None;
        m.step_resize = Some(mriStep_Resize);
        m.step_reset = Some(mriStep_Reset);
        m.step_free = Some(mriStep_Free);
        m.step_printmem = Some(mriStep_PrintMem);
        m.step_setdefaults = Some(mriStep_SetDefaults);
        m.step_computestate = Some(mriStep_ComputeState);
        m.step_setoptions = Some(mriStep_SetOptions);
        m.step_setorder = Some(mriStep_SetOrder);
        m.step_setnonlinearsolver = Some(mriStep_SetNonlinearSolver);
        m.step_setlinear = Some(mriStep_SetLinear);
        m.step_setnonlinear = Some(mriStep_SetNonlinear);
        m.step_setnlsrhsfn = Some(mriStep_SetNlsRhsFn);
        m.step_setdeduceimplicitrhs = Some(mriStep_SetDeduceImplicitRhs);
        m.step_setnonlincrdown = Some(mriStep_SetNonlinCRDown);
        m.step_setnonlinrdiv = Some(mriStep_SetNonlinRDiv);
        m.step_setdeltagammamax = Some(mriStep_SetDeltaGammaMax);
        m.step_setlsetupfrequency = Some(mriStep_SetLSetupFrequency);
        m.step_setpredictormethod = Some(mriStep_SetPredictorMethod);
        m.step_setmaxnonliniters = Some(mriStep_SetMaxNonlinIters);
        m.step_setnonlinconvcoef = Some(mriStep_SetNonlinConvCoef);
        m.step_setstagepredictfn = Some(mriStep_SetStagePredictFn);
        m.step_getnumrhsevals = Some(mriStep_GetNumRhsEvals);
        m.step_getnumlinsolvsetups = Some(mriStep_GetNumLinSolvSetups);
        m.step_getcurrentgamma = Some(mriStep_GetCurrentGamma);
        m.step_setadaptcontroller = Some(mriStep_SetAdaptController);
        m.step_getestlocalerrors = Some(mriStep_GetEstLocalErrors);
        m.step_getnonlinearsystemdata = Some(mriStep_GetNonlinearSystemData);
        m.step_getnumnonlinsolviters = Some(mriStep_GetNumNonlinSolvIters);
        m.step_getnumnonlinsolvconvfails = Some(mriStep_GetNumNonlinSolvConvFails);
        m.step_getnonlinsolvstats = Some(mriStep_GetNonlinSolvStats);
        m.step_setforcing = Some(mriStep_SetInnerForcing);
        m.step_getstageindex = Some(mriStep_GetStageIndex);
        m.step_supports_adaptive = SUNTRUE;
        m.step_supports_implicit = SUNTRUE;
        m.step_mem = Some(Box::new(step_mem));
    }

    /* Set default values for optional inputs */
    retval = mriStep_SetDefaults(&ark_mem);
    if retval != ARK_SUCCESS {
        arkProcessError(
            Some(&ark_mem),
            retval,
            line!() as i32,
            "MRIStepCreate",
            file!(),
            "Error setting default solver options",
        );
        let mut mem = Some(ark_mem.clone());
        ARKodeFree(&mut mem);
        return None;
    }

    /* Allocate the general MRI stepper vectors using y0 as a template */
    /* NOTE: Fse, Fsi, inner_forcing, sdata, zpred and zcor will be allocated
       later on (based on the MRI method) */

    /* Copy the slow RHS functions into stepper memory */
    {
        let mut s = mriStep_mem_mut(&ark_mem);
        s.fse = fse;
        s.fsi = fsi;
        s.fse_is_current = SUNFALSE;
        s.fsi_is_current = SUNFALSE;

        /* Set implicit/explicit problem based on function pointers */
        s.explicit_rhs = fse.is_some();
        s.implicit_rhs = fsi.is_some();
    }

    /* Update the ARKODE workspace requirements */
    {
        let mut m = ark_mem.borrow_mut();
        m.liw += 49; /* fcn/data ptr, int, long int, sunindextype, sunbooleantype */
        m.lrw += 14;
    }

    /* Create a default Newton NLS object (just in case; will be deleted if
       the user attaches a nonlinear solver) */
    let implicit_rhs = {
        let mut s = mriStep_mem_mut(&ark_mem);
        s.NLS = None;
        s.ownNLS = SUNFALSE;
        s.implicit_rhs
    };

    if implicit_rhs {
        let ark_sunctx = ark_mem.borrow().sunctx.clone();
        let NLS = match SUNNonlinSol_Newton(y0, &ark_sunctx) {
            Some(nls) => nls,
            None => {
                arkProcessError(
                    Some(&ark_mem),
                    ARK_MEM_FAIL,
                    line!() as i32,
                    "MRIStepCreate",
                    file!(),
                    "Error creating default Newton solver",
                );
                let mut mem = Some(ark_mem.clone());
                ARKodeFree(&mut mem);
                return None;
            }
        };
        retval = ARKodeSetNonlinearSolver(&ark_mem, &NLS);
        if retval != ARK_SUCCESS {
            arkProcessError(
                Some(&ark_mem),
                ARK_MEM_FAIL,
                line!() as i32,
                "MRIStepCreate",
                file!(),
                "Error attaching default Newton solver",
            );
            let mut mem = Some(ark_mem.clone());
            ARKodeFree(&mut mem);
            return None;
        }
        mriStep_mem_mut(&ark_mem).ownNLS = SUNTRUE;
    }

    /* Set the linear solver addresses to NULL (we check != NULL later) */
    {
        let mut s = mriStep_mem_mut(&ark_mem);
        s.linit = None;
        s.lsetup = None;
        s.lsolve = None;
        s.lfree = None;

        /* Initialize error norm  */
        s.eRNrm = ONE;

        /* Initialize all the counters */
        s.nfse = 0;
        s.nfsi = 0;
        s.nsetups = 0;
        s.nstlp = 0;
        s.nls_iters = 0;
        s.nls_fails = 0;
        s.inner_fails = 0;
    }
    /* C `step_mem->lmem = NULL`: the ARKLS record lives in ark_mem (§4) */
    ark_mem.borrow_mut().ark_lmem = None;

    /* Initialize fused op work space with sufficient storage for at least filling
       the full RHS on an ImEx problem -- must be allocate here as the full RHS
       is called before mriStep_Init when nesting MRI methods.
       The C calloc-failure branches are unreachable: a Rust allocation failure
       aborts rather than returning NULL. */
    let nfusedopvecs = {
        let mut s = mriStep_mem_mut(&ark_mem);
        s.nfusedopvecs = 3;
        s.cvals = vec![ZERO; s.nfusedopvecs as usize];
        s.Xvecs = vec![None; s.nfusedopvecs as usize];
        s.nfusedopvecs
    };
    {
        let mut m = ark_mem.borrow_mut();
        m.lrw += nfusedopvecs as i64;
        m.liw += nfusedopvecs as i64;
    }

    {
        let mut s = mriStep_mem_mut(&ark_mem);

        /* Initialize adaptivity parameters */
        s.inner_rtol_factor = ONE;
        s.inner_dsm = ONE;
        s.inner_rtol_factor_new = ONE;

        /* Initialize pre and post inner evolve functions */
        s.pre_inner_evolve = None;
        s.post_inner_evolve = None;

        /* Initialize external polynomial forcing data */
        s.expforcing = SUNFALSE;
        s.impforcing = SUNFALSE;
        s.forcing = Vec::new();
        s.nforcing = 0;
    }

    /* Initialize main ARKODE infrastructure (allocates vectors) */
    retval = arkInit(&ark_mem, t0, y0, FIRST_INIT);
    if retval != ARK_SUCCESS {
        arkProcessError(
            Some(&ark_mem),
            retval,
            line!() as i32,
            "MRIStepCreate",
            file!(),
            "Unable to initialize main ARKODE infrastructure",
        );
        let mut mem = Some(ark_mem.clone());
        ARKodeFree(&mut mem);
        return None;
    }

    /* Attach the inner stepper memory */
    mriStep_mem_mut(&ark_mem).stepper = Some(stepper.clone());

    /* Check for required stepper functions */
    retval = mriStepInnerStepper_HasRequiredOps(stepper);
    if retval != ARK_SUCCESS {
        arkProcessError(
            Some(&ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "MRIStepCreate",
            file!(),
            "A required inner stepper function is NULL",
        );
        let mut mem = Some(ark_mem.clone());
        ARKodeFree(&mut mem);
        return None;
    }

    /* return ARKODE memory */
    Some(ark_mem)
}

/*---------------------------------------------------------------
  MRIStepReInit:

  This routine re-initializes the MRIStep module to solve a new
  problem of the same size as was previously solved (all counter
  values are set to 0).

  NOTE: the inner stepper needs to be reinitialized before
  calling this function.
  ---------------------------------------------------------------*/
pub fn MRIStepReInit(
    arkode_mem: &ARKodeMem,
    fse: Option<ARKRhsFn>,
    fsi: Option<ARKRhsFn>,
    t0: sunrealtype,
    y0: &N_Vector,
) -> i32 {
    let mut retval: i32;

    /* access ARKodeMem and ARKodeMRIStepMem structures */
    retval = mriStep_step_mem_ok(arkode_mem, "MRIStepReInit");
    if retval != ARK_SUCCESS {
        return retval;
    }
    let ark_mem = arkode_mem;

    /* Check if ark_mem was allocated */
    if !ark_mem.borrow().MallocDone {
        arkProcessError(
            Some(ark_mem),
            ARK_NO_MALLOC,
            line!() as i32,
            "MRIStepReInit",
            file!(),
            MSG_ARK_NO_MALLOC,
        );
        return ARK_NO_MALLOC;
    }

    /* Check that at least one of fse, fsi is supplied and is to be used */
    if fse.is_none() && fsi.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "MRIStepReInit",
            file!(),
            MSG_ARK_NULL_F,
        );
        return ARK_ILL_INPUT;
    }

    /* Check that y0 is supplied: handled by the type system */

    /* Set implicit/explicit problem based on function pointers */
    let (implicit_rhs, have_nls) = {
        let mut s = mriStep_mem_mut(ark_mem);
        s.explicit_rhs = fse.is_some();
        s.implicit_rhs = fsi.is_some();
        (s.implicit_rhs, s.NLS.is_some())
    };

    /* Create a default Newton NLS object (just in case; will be deleted if
       the user attaches a nonlinear solver) */
    if implicit_rhs && !have_nls {
        let ark_sunctx = ark_mem.borrow().sunctx.clone();
        let NLS = match SUNNonlinSol_Newton(y0, &ark_sunctx) {
            Some(nls) => nls,
            None => {
                arkProcessError(
                    Some(ark_mem),
                    ARK_MEM_FAIL,
                    line!() as i32,
                    "MRIStepReInit",
                    file!(),
                    "Error creating default Newton solver",
                );
                let mut mem = Some(ark_mem.clone());
                ARKodeFree(&mut mem);
                return ARK_MEM_FAIL;
            }
        };
        retval = ARKodeSetNonlinearSolver(ark_mem, &NLS);
        if retval != ARK_SUCCESS {
            arkProcessError(
                Some(ark_mem),
                ARK_MEM_FAIL,
                line!() as i32,
                "MRIStepReInit",
                file!(),
                "Error attaching default Newton solver",
            );
            let mut mem = Some(ark_mem.clone());
            ARKodeFree(&mut mem);
            return ARK_MEM_FAIL;
        }
        mriStep_mem_mut(ark_mem).ownNLS = SUNTRUE;
    }

    /* ReInitialize main ARKODE infrastructure */
    retval = arkInit(ark_mem, t0, y0, FIRST_INIT);
    if retval != ARK_SUCCESS {
        arkProcessError(
            Some(ark_mem),
            retval,
            line!() as i32,
            "MRIStepReInit",
            file!(),
            "Unable to reinitialize main ARKODE infrastructure",
        );
        return retval;
    }

    /* Copy the input parameters into ARKODE state */
    {
        let mut s = mriStep_mem_mut(ark_mem);
        s.fse = fse;
        s.fsi = fsi;
        s.fse_is_current = SUNFALSE;
        s.fsi_is_current = SUNFALSE;

        /* Initialize all the counters */
        s.nfse = 0;
        s.nfsi = 0;
        s.nsetups = 0;
        s.nstlp = 0;
        s.nls_iters = 0;
        s.nls_fails = 0;
        s.inner_fails = 0;
    }

    /* C `if (step_mem->lmem) { arkLsInitializeCounters(step_mem->lmem); }` */
    let have_lmem = ark_mem.borrow().ark_lmem.is_some();
    if have_lmem {
        let mut arkls_mem = arkls_mem_mut(ark_mem);
        arkLsInitializeCounters(&mut arkls_mem);
    }

    ARK_SUCCESS
}

/*===============================================================
  Interface routines supplied to ARKODE
  ===============================================================*/

/*---------------------------------------------------------------
  mriStep_Resize:

  This routine resizes the memory within the MRIStep module.
  ---------------------------------------------------------------*/
pub fn mriStep_Resize(
    ark_mem: &ARKodeMem,
    y0: &N_Vector,
    _hscale: sunrealtype,
    _t0: sunrealtype,
    resize: Option<ARKVecResizeFn>,
    resize_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    let mut retval: i32;

    /* access ARKodeMRIStepMem structure */
    retval = mriStep_step_mem_ok(ark_mem, "mriStep_Resize");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* Determine change in vector sizes */
    let mut lrw1: sunindextype = 0;
    let mut liw1: sunindextype = 0;
    if y0.ops.borrow().nvspace.is_some() {
        N_VSpace(y0, &mut lrw1, &mut liw1);
    }
    let (lrw_diff, liw_diff) = {
        let mut m = ark_mem.borrow_mut();
        let lrw_diff = lrw1 - m.lrw1;
        let liw_diff = liw1 - m.liw1;
        m.lrw1 = lrw1;
        m.liw1 = liw1;
        (lrw_diff, liw_diff)
    };

    /* Resize Fse */
    let (have_Fse, nstages_allocated) = {
        let s = mriStep_mem_mut(ark_mem);
        (!s.Fse.is_empty(), s.nstages_allocated)
    };
    if have_Fse {
        let mut Fse = { std::mem::take(&mut mriStep_mem_mut(ark_mem).Fse) };
        let (mut lrw, mut liw) = {
            let m = ark_mem.borrow();
            (m.lrw, m.liw)
        };
        let ok = arkResizeVecArray(
            resize,
            resize_data,
            nstages_allocated,
            y0,
            &mut Fse,
            lrw_diff,
            &mut lrw,
            liw_diff,
            &mut liw,
        );
        {
            let mut m = ark_mem.borrow_mut();
            m.lrw = lrw;
            m.liw = liw;
        }
        {
            let mut s = mriStep_mem_mut(ark_mem);
            s.Fse = Fse;
        }
        if !ok {
            arkProcessError(
                Some(ark_mem),
                ARK_MEM_FAIL,
                line!() as i32,
                "mriStep_Resize",
                file!(),
                "Unable to resize vector",
            );
            return ARK_MEM_FAIL;
        }
        let mut s = mriStep_mem_mut(ark_mem);
        if s.unify_Fs {
            s.Fsi = s.Fse.clone();
        }
    }

    /* Resize Fsi */
    let resize_Fsi = {
        let s = mriStep_mem_mut(ark_mem);
        !s.Fsi.is_empty() && !s.unify_Fs
    };
    if resize_Fsi {
        let mut Fsi = { std::mem::take(&mut mriStep_mem_mut(ark_mem).Fsi) };
        let (mut lrw, mut liw) = {
            let m = ark_mem.borrow();
            (m.lrw, m.liw)
        };
        let ok = arkResizeVecArray(
            resize,
            resize_data,
            nstages_allocated,
            y0,
            &mut Fsi,
            lrw_diff,
            &mut lrw,
            liw_diff,
            &mut liw,
        );
        {
            let mut m = ark_mem.borrow_mut();
            m.lrw = lrw;
            m.liw = liw;
        }
        {
            let mut s = mriStep_mem_mut(ark_mem);
            s.Fsi = Fsi;
        }
        if !ok {
            arkProcessError(
                Some(ark_mem),
                ARK_MEM_FAIL,
                line!() as i32,
                "mriStep_Resize",
                file!(),
                "Unable to resize vector",
            );
            return ARK_MEM_FAIL;
        }
    }

    /* Resize the nonlinear solver interface vectors (if applicable) */
    {
        let mut sdata = { mriStep_mem_mut(ark_mem).sdata.clone() };
        if sdata.is_some() {
            let ok = arkResizeVec(
                ark_mem,
                resize,
                resize_data,
                lrw_diff,
                liw_diff,
                y0,
                &mut sdata,
            );
            mriStep_mem_mut(ark_mem).sdata = sdata;
            if !ok {
                arkProcessError(
                    Some(ark_mem),
                    ARK_MEM_FAIL,
                    line!() as i32,
                    "mriStep_Resize",
                    file!(),
                    "Unable to resize vector",
                );
                return ARK_MEM_FAIL;
            }
        }
    }
    {
        let mut zpred = { mriStep_mem_mut(ark_mem).zpred.clone() };
        if zpred.is_some() {
            let ok = arkResizeVec(
                ark_mem,
                resize,
                resize_data,
                lrw_diff,
                liw_diff,
                y0,
                &mut zpred,
            );
            mriStep_mem_mut(ark_mem).zpred = zpred;
            if !ok {
                arkProcessError(
                    Some(ark_mem),
                    ARK_MEM_FAIL,
                    line!() as i32,
                    "mriStep_Resize",
                    file!(),
                    "Unable to resize vector",
                );
                return ARK_MEM_FAIL;
            }
        }
    }
    {
        let mut zcor = { mriStep_mem_mut(ark_mem).zcor.clone() };
        if zcor.is_some() {
            let ok = arkResizeVec(
                ark_mem,
                resize,
                resize_data,
                lrw_diff,
                liw_diff,
                y0,
                &mut zcor,
            );
            mriStep_mem_mut(ark_mem).zcor = zcor;
            if !ok {
                arkProcessError(
                    Some(ark_mem),
                    ARK_MEM_FAIL,
                    line!() as i32,
                    "mriStep_Resize",
                    file!(),
                    "Unable to resize vector",
                );
                return ARK_MEM_FAIL;
            }
        }
    }

    /* If a NLS object was previously used, destroy and recreate default Newton
       NLS object (can be replaced by user-defined object if desired) */
    let recreate_nls = {
        let s = mriStep_mem_mut(ark_mem);
        s.NLS.is_some() && s.ownNLS
    };
    if recreate_nls {
        /* destroy existing NLS object */
        let old_nls = { mriStep_mem_mut(ark_mem).NLS.take() };
        retval = SUNNonlinSolFree(old_nls);
        if retval != ARK_SUCCESS {
            return retval;
        }
        {
            let mut s = mriStep_mem_mut(ark_mem);
            s.NLS = None;
            s.ownNLS = SUNFALSE;
        }

        /* create new Newton NLS object */
        let ark_sunctx = ark_mem.borrow().sunctx.clone();
        let NLS = match SUNNonlinSol_Newton(y0, &ark_sunctx) {
            Some(nls) => nls,
            None => {
                arkProcessError(
                    Some(ark_mem),
                    ARK_MEM_FAIL,
                    line!() as i32,
                    "mriStep_Resize",
                    file!(),
                    "Error creating default Newton solver",
                );
                return ARK_MEM_FAIL;
            }
        };

        /* attach new Newton NLS object */
        retval = ARKodeSetNonlinearSolver(ark_mem, &NLS);
        if retval != ARK_SUCCESS {
            arkProcessError(
                Some(ark_mem),
                ARK_MEM_FAIL,
                line!() as i32,
                "mriStep_Resize",
                file!(),
                "Error attaching default Newton solver",
            );
            return ARK_MEM_FAIL;
        }
        mriStep_mem_mut(ark_mem).ownNLS = SUNTRUE;
    }

    /* Resize the inner stepper vectors */
    let stepper = { mriStep_mem_mut(ark_mem).stepper.clone() }.expect("stepper set");
    retval = mriStepInnerStepper_Resize(&stepper, resize, resize_data, lrw_diff, liw_diff, y0);
    if retval != ARK_SUCCESS {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_FAIL,
            line!() as i32,
            "mriStep_Resize",
            file!(),
            "Unable to resize vector",
        );
        return ARK_MEM_FAIL;
    }

    /* reset nonlinear solver counters */
    {
        let mut s = mriStep_mem_mut(ark_mem);
        if s.NLS.is_some() {
            s.nsetups = 0;
        }
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  mriStep_Reset:

  This routine resets the MRIStep module state to solve the same
  problem from the given time with the input state (all counter
  values are retained).  It is called after the main ARKODE
  infrastructure is reset.
  ---------------------------------------------------------------*/
pub fn mriStep_Reset(ark_mem: &ARKodeMem, tR: sunrealtype, yR: &N_Vector) -> i32 {
    /* access ARKodeMRIStepMem structure */
    let retval = mriStep_step_mem_ok(ark_mem, "mriStep_Reset");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* Reset the inner integrator with this same state */
    let stepper = { mriStep_mem_mut(ark_mem).stepper.clone() }.expect("stepper set");
    let retval = mriStepInnerStepper_Reset(&stepper, tR, yR);
    if retval != ARK_SUCCESS {
        arkProcessError(
            Some(ark_mem),
            ARK_INNERSTEP_FAIL,
            line!() as i32,
            "mriStep_Reset",
            file!(),
            "Unable to reset the inner stepper",
        );
        return ARK_INNERSTEP_FAIL;
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  mriStep_ComputeState:

  Computes y based on the current prediction and given correction.
  ---------------------------------------------------------------*/
pub fn mriStep_ComputeState(ark_mem: &ARKodeMem, zcor: &N_Vector, z: &N_Vector) -> i32 {
    /* access ARKodeMRIStepMem structure */
    let retval = mriStep_step_mem_ok(ark_mem, "mriStep_ComputeState");
    if retval != ARK_SUCCESS {
        return retval;
    }

    let zpred = { mriStep_mem_mut(ark_mem).zpred.clone() }.expect("zpred set");
    N_VLinearSum(ONE, &zpred, ONE, zcor, z);

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  mriStep_Free frees all MRIStep memory.
  ---------------------------------------------------------------*/
pub fn mriStep_Free(ark_mem: &ARKodeMem) {
    /* nothing to do if ark_mem is already NULL: handled by the type system */

    /* conditional frees on non-NULL MRIStep module */
    if ark_mem.borrow().step_mem.is_none() {
        return;
    }

    /* free the coupling structure and derived quantities */
    let MRIC = { mriStep_mem_mut(ark_mem).MRIC.take() };
    if let Some(MRIC) = MRIC {
        let mut Cliw: sunindextype = 0;
        let mut Clrw: sunindextype = 0;
        MRIStepCoupling_Space(Some(&MRIC), &mut Cliw, &mut Clrw);
        MRIStepCoupling_Free(Some(MRIC));
        /* `step_mem->MRIC = NULL` performed by the `take()` above */
        {
            let mut m = ark_mem.borrow_mut();
            m.liw -= Cliw;
            m.lrw -= Clrw;
        }
        let stages = { mriStep_mem_mut(ark_mem).stages };
        let have_stagetypes = { !mriStep_mem_mut(ark_mem).stagetypes.is_empty() };
        if have_stagetypes {
            mriStep_mem_mut(ark_mem).stagetypes = Vec::new();
            ark_mem.borrow_mut().liw -= (stages + 1) as i64;
        }
        let have_stage_map = { !mriStep_mem_mut(ark_mem).stage_map.is_empty() };
        if have_stage_map {
            mriStep_mem_mut(ark_mem).stage_map = Vec::new();
            ark_mem.borrow_mut().liw -= stages as i64;
        }
        let have_Ae_row = { !mriStep_mem_mut(ark_mem).Ae_row.is_empty() };
        if have_Ae_row {
            mriStep_mem_mut(ark_mem).Ae_row = Vec::new();
            ark_mem.borrow_mut().lrw -= stages as i64;
        }
        let have_Ai_row = { !mriStep_mem_mut(ark_mem).Ai_row.is_empty() };
        if have_Ai_row {
            mriStep_mem_mut(ark_mem).Ai_row = Vec::new();
            ark_mem.borrow_mut().lrw -= stages as i64;
        }
    }

    /* free the nonlinear solver memory (if applicable) */
    let free_nls = {
        let s = mriStep_mem_mut(ark_mem);
        s.NLS.is_some() && s.ownNLS
    };
    if free_nls {
        let nls = { mriStep_mem_mut(ark_mem).NLS.clone() };
        let _ = SUNNonlinSolFree(nls);
        mriStep_mem_mut(ark_mem).ownNLS = SUNFALSE;
    }
    mriStep_mem_mut(ark_mem).NLS = None;

    /* free the linear solver memory */
    let lfree = { mriStep_mem_mut(ark_mem).lfree };
    if let Some(lfree) = lfree {
        let _ = lfree(ark_mem);
        /* C `step_mem->lmem = NULL` (the record lives in ark_mem, §4) */
        ark_mem.borrow_mut().ark_lmem = None;
    }

    /* free the sdata, zpred and zcor vectors */
    {
        let mut sdata = { mriStep_mem_mut(ark_mem).sdata.take() };
        if sdata.is_some() {
            arkFreeVec(ark_mem, &mut sdata);
            mriStep_mem_mut(ark_mem).sdata = None;
        }
    }
    {
        let mut zpred = { mriStep_mem_mut(ark_mem).zpred.take() };
        if zpred.is_some() {
            arkFreeVec(ark_mem, &mut zpred);
            mriStep_mem_mut(ark_mem).zpred = None;
        }
    }
    {
        let mut zcor = { mriStep_mem_mut(ark_mem).zcor.take() };
        if zcor.is_some() {
            arkFreeVec(ark_mem, &mut zcor);
            mriStep_mem_mut(ark_mem).zcor = None;
        }
    }

    /* free the RHS vectors */
    let (nstages_allocated, have_Fse) = {
        let s = mriStep_mem_mut(ark_mem);
        (s.nstages_allocated, !s.Fse.is_empty())
    };
    if have_Fse {
        let (lrw1, liw1, mut lrw, mut liw) = {
            let m = ark_mem.borrow();
            (m.lrw1, m.liw1, m.lrw, m.liw)
        };
        let mut Fse = { std::mem::take(&mut mriStep_mem_mut(ark_mem).Fse) };
        arkFreeVecArray(nstages_allocated, &mut Fse, lrw1, &mut lrw, liw1, &mut liw);
        {
            let mut m = ark_mem.borrow_mut();
            m.lrw = lrw;
            m.liw = liw;
        }
        let mut s = mriStep_mem_mut(ark_mem);
        s.Fse = Fse;
        if s.unify_Fs {
            s.Fsi = Vec::new();
        }
    }

    let have_Fsi = { !mriStep_mem_mut(ark_mem).Fsi.is_empty() };
    if have_Fsi {
        let (lrw1, liw1, mut lrw, mut liw) = {
            let m = ark_mem.borrow();
            (m.lrw1, m.liw1, m.lrw, m.liw)
        };
        let mut Fsi = { std::mem::take(&mut mriStep_mem_mut(ark_mem).Fsi) };
        arkFreeVecArray(nstages_allocated, &mut Fsi, lrw1, &mut lrw, liw1, &mut liw);
        {
            let mut m = ark_mem.borrow_mut();
            m.lrw = lrw;
            m.liw = liw;
        }
        mriStep_mem_mut(ark_mem).Fsi = Fsi;
    }

    /* free the reusable arrays for fused vector interface */
    let (have_cvals, have_Xvecs, nfusedopvecs) = {
        let s = mriStep_mem_mut(ark_mem);
        (!s.cvals.is_empty(), !s.Xvecs.is_empty(), s.nfusedopvecs)
    };
    if have_cvals {
        mriStep_mem_mut(ark_mem).cvals = Vec::new();
        ark_mem.borrow_mut().lrw -= nfusedopvecs as i64;
    }
    if have_Xvecs {
        mriStep_mem_mut(ark_mem).Xvecs = Vec::new();
        ark_mem.borrow_mut().liw -= nfusedopvecs as i64;
    }
    mriStep_mem_mut(ark_mem).nfusedopvecs = 0;

    /* free the time stepper module itself */
    ark_mem.borrow_mut().step_mem = None;
}

/*---------------------------------------------------------------
  mriStep_PrintMem:

  This routine outputs the memory from the MRIStep structure to
  a specified file pointer (useful when debugging).
  ---------------------------------------------------------------*/
pub fn mriStep_PrintMem(ark_mem: &ARKodeMem, outfile: &SUNFile) {
    /* access ARKodeMRIStepMem structure */
    let retval = mriStep_step_mem_ok(ark_mem, "mriStep_PrintMem");
    if retval != ARK_SUCCESS {
        return;
    }

    /* output integer quantities */
    let (q, p, istage, cur_stage, stages, maxcor, msbp, predictor, convfail, stagetypes) = {
        let s = mriStep_mem_mut(ark_mem);
        (
            s.q,
            s.p,
            s.istage,
            s.cur_stage,
            s.stages,
            s.maxcor,
            s.msbp,
            s.predictor,
            s.convfail,
            s.stagetypes.clone(),
        )
    };
    outfile.write_str(&format!("MRIStep: q = {q}\n"));
    outfile.write_str(&format!("MRIStep: p = {p}\n"));
    outfile.write_str(&format!("MRIStep: istage = {istage}\n"));
    outfile.write_str(&format!("MRIStep: cur_stage = {cur_stage}\n"));
    outfile.write_str(&format!("MRIStep: stages = {stages}\n"));
    outfile.write_str(&format!("MRIStep: maxcor = {maxcor}\n"));
    outfile.write_str(&format!("MRIStep: msbp = {msbp}\n"));
    outfile.write_str(&format!("MRIStep: predictor = {predictor}\n"));
    outfile.write_str(&format!("MRIStep: convfail = {convfail}\n"));
    outfile.write_str("MRIStep: stagetypes =");
    for i in 0..=stages {
        outfile.write_str(&format!(" {}", stagetypes[i as usize]));
    }
    outfile.write_str("\n");

    /* output long integer quantities */
    let (nfse, nfsi, nsetups, nstlp, nls_iters, nls_fails, inner_fails) = {
        let s = mriStep_mem_mut(ark_mem);
        (
            s.nfse,
            s.nfsi,
            s.nsetups,
            s.nstlp,
            s.nls_iters,
            s.nls_fails,
            s.inner_fails,
        )
    };
    outfile.write_str(&format!("MRIStep: nfse = {nfse}\n"));
    outfile.write_str(&format!("MRIStep: nfsi = {nfsi}\n"));
    outfile.write_str(&format!("MRIStep: nsetups = {nsetups}\n"));
    outfile.write_str(&format!("MRIStep: nstlp = {nstlp}\n"));
    outfile.write_str(&format!("MRIStep: nls_iters = {nls_iters}\n"));
    outfile.write_str(&format!("MRIStep: nls_fails = {nls_fails}\n"));
    outfile.write_str(&format!("MRIStep: inner_fails = {inner_fails}\n"));

    /* output boolean quantities */
    let (linear, linear_timedep, explicit_rhs, implicit_rhs, jcur, ownNLS) = {
        let s = mriStep_mem_mut(ark_mem);
        (
            s.linear,
            s.linear_timedep,
            s.explicit_rhs,
            s.implicit_rhs,
            s.jcur.get(),
            s.ownNLS,
        )
    };
    outfile.write_str(&format!("MRIStep: user_linear = {}\n", linear as i32));
    outfile.write_str(&format!(
        "MRIStep: user_linear_timedep = {}\n",
        linear_timedep as i32
    ));
    outfile.write_str(&format!("MRIStep: user_explicit = {}\n", explicit_rhs as i32));
    outfile.write_str(&format!("MRIStep: user_implicit = {}\n", implicit_rhs as i32));
    outfile.write_str(&format!("MRIStep: jcur = {}\n", jcur as i32));
    outfile.write_str(&format!("MRIStep: ownNLS = {}\n", ownNLS as i32));

    /* output sunrealtype quantities */
    outfile.write_str("MRIStep: Coupling structure:\n");
    let MRIC = { mriStep_mem_mut(ark_mem).MRIC.clone() };
    if let Some(MRIC) = MRIC {
        MRIStepCoupling_Write(Some(&MRIC), outfile);
    }

    let (gamma, gammap, gamrat, crate_, delnrm_p, eRNrm, nlscoef, crdown, rdiv, dgmax) = {
        let s = mriStep_mem_mut(ark_mem);
        (
            s.gamma, s.gammap, s.gamrat, s.crate_, s.delnrm_p, s.eRNrm, s.nlscoef, s.crdown,
            s.rdiv, s.dgmax,
        )
    };
    outfile.write_str(&format!("MRIStep: gamma = {}\n", sun_format_g(gamma)));
    outfile.write_str(&format!("MRIStep: gammap = {}\n", sun_format_g(gammap)));
    outfile.write_str(&format!("MRIStep: gamrat = {}\n", sun_format_g(gamrat)));
    outfile.write_str(&format!("MRIStep: crate = {}\n", sun_format_g(crate_)));
    outfile.write_str(&format!("MRIStep: delnrm_p = {}\n", sun_format_g(delnrm_p)));
    outfile.write_str(&format!("MRIStep: eRNrm = {}\n", sun_format_g(eRNrm)));
    outfile.write_str(&format!("MRIStep: nlscoef = {}\n", sun_format_g(nlscoef)));
    outfile.write_str(&format!("MRIStep: crdown = {}\n", sun_format_g(crdown)));
    outfile.write_str(&format!("MRIStep: rdiv = {}\n", sun_format_g(rdiv)));
    outfile.write_str(&format!("MRIStep: dgmax = {}\n", sun_format_g(dgmax)));

    let (nstages_active, Ae_row, Ai_row) = {
        let s = mriStep_mem_mut(ark_mem);
        (s.nstages_active, s.Ae_row.clone(), s.Ai_row.clone())
    };
    outfile.write_str("MRIStep: Ae_row =");
    for i in 0..nstages_active {
        outfile.write_str(&format!(" {}", sun_format_g(Ae_row[i as usize])));
    }
    outfile.write_str("\n");
    outfile.write_str("MRIStep: Ai_row =");
    for i in 0..nstages_active {
        outfile.write_str(&format!(" {}", sun_format_g(Ai_row[i as usize])));
    }
    outfile.write_str("\n");

    /* SUNDIALS_DEBUG_PRINTVEC vector output is not enabled in this build */

    /* print the inner stepper memory */
    let stepper = { mriStep_mem_mut(ark_mem).stepper.clone() }.expect("stepper set");
    mriStepInnerStepper_PrintMem(&stepper, outfile);
}

/*---------------------------------------------------------------
  mriStep_AttachLinsol:

  This routine attaches the various set of system linear solver
  interface routines, data structure, and solver type to the
  MRIStep module.
  ---------------------------------------------------------------*/
pub fn mriStep_AttachLinsol(
    ark_mem: &ARKodeMem,
    linit: Option<ARKLinsolInitFn>,
    lsetup: Option<ARKLinsolSetupFn>,
    lsolve: Option<ARKLinsolSolveFn>,
    lfree: Option<ARKLinsolFreeFn>,
    _lsolve_type: SUNLinearSolver_Type,
    lmem: Option<Box<dyn Any>>,
) -> i32 {
    /* access ARKodeMRIStepMem structure */
    let retval = mriStep_step_mem_ok(ark_mem, "mriStep_AttachLinsol");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* free any existing system solver */
    let old_lfree = { mriStep_mem_mut(ark_mem).lfree };
    if let Some(old_lfree) = old_lfree {
        let _ = old_lfree(ark_mem);
    }

    /* Attach the provided routines, data structure and solve type */
    {
        let mut s = mriStep_mem_mut(ark_mem);
        s.linit = linit;
        s.lsetup = lsetup;
        s.lsolve = lsolve;
        s.lfree = lfree;

        /* Reset all linear solver counters */
        s.nsetups = 0;
        s.nstlp = 0;
    }
    /* C `step_mem->lmem = lmem`: the record is owned by ark_mem (§4) */
    ark_mem.borrow_mut().ark_lmem = lmem;

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  mriStep_DisableLSetup:

  This routine NULLifies the lsetup function pointer in the
  MRIStep module.
  ---------------------------------------------------------------*/
pub fn mriStep_DisableLSetup(ark_mem: &ARKodeMem) {
    /* access ARKodeMRIStepMem structure */
    let retval = mriStep_step_mem_ok(ark_mem, "mriStep_DisableLSetup");
    if retval != ARK_SUCCESS {
        return;
    }

    /* nullify the lsetup function pointer */
    mriStep_mem_mut(ark_mem).lsetup = None;
}

/*---------------------------------------------------------------
  mriStep_GetLmem:

  This routine returns the system linear solver interface memory
  structure, lmem.

  Seam (§4): the ARKLS record is stored BY VALUE in `ark_mem.ark_lmem`, so
  this reports PRESENCE; `arkls_mem_mut(ark_mem)` reaches the record.
  ---------------------------------------------------------------*/
pub fn mriStep_GetLmem(ark_mem: &ARKodeMem) -> sunbooleantype {
    /* access ARKodeMRIStepMem structure, and return lmem */
    let retval = mriStep_step_mem_ok(ark_mem, "mriStep_GetLmem");
    if retval != ARK_SUCCESS {
        return SUNFALSE;
    }
    ark_mem.borrow().ark_lmem.is_some()
}

/*---------------------------------------------------------------
  mriStep_GetImplicitRHS:

  This routine returns the implicit RHS function pointer, fi.
  ---------------------------------------------------------------*/
pub fn mriStep_GetImplicitRHS(ark_mem: &ARKodeMem) -> Option<ARKRhsFn> {
    /* access ARKodeMRIStepMem structure, and return fi */
    let retval = mriStep_step_mem_ok(ark_mem, "mriStep_GetImplicitRHS");
    if retval != ARK_SUCCESS {
        return None;
    }
    let s = mriStep_mem_mut(ark_mem);
    if s.implicit_rhs {
        s.fsi
    } else {
        None
    }
}

/*---------------------------------------------------------------
  mriStep_GetGammas:

  This routine fills the current value of gamma, and states
  whether the gamma ratio fails the dgmax criteria.
  ---------------------------------------------------------------*/
pub fn mriStep_GetGammas(
    ark_mem: &ARKodeMem,
    gamma: &mut sunrealtype,
    gamrat: &mut sunrealtype,
    jcur: &mut Option<ARKJcurPtr>,
    dgamma_fail: &mut sunbooleantype,
) -> i32 {
    /* access ARKodeMRIStepMem structure */
    let retval = mriStep_step_mem_ok(ark_mem, "mriStep_GetGammas");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* set outputs */
    let (s_gamma, s_gamrat, s_jcur, s_dgmax) = {
        let s = mriStep_mem_mut(ark_mem);
        (s.gamma, s.gamrat, s.jcur.clone(), s.dgmax)
    };
    *gamma = s_gamma;
    *gamrat = s_gamrat;
    *jcur = Some(s_jcur);
    *dgamma_fail = SUNRabs(*gamrat - ONE) >= s_dgmax;

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  mriStep_Init:

  This routine is called just prior to performing internal time
  steps (after all user "set" routines have been called) from
  within arkInitialSetup.

  With initialization type RESET_INIT, this routine does nothing.

  For other initialization types, this routine:
  - initializes and sets up the linear and nonlinear solvers
    (if applicable)
  - initializes and sets up the nonlinear solver (if applicable)
  - performs timestep adaptivity checks and initial setup,
    including setting the initial time step size if needed
  - sets the relevant TakeStep routine based on the current
    problem configuration
  - sets/checks the coefficient tables to be used
  - allocates any internal memory that depends on the MRI method
    structure or solver options

  With other initialization types, this routine does nothing.
  ---------------------------------------------------------------*/
pub fn mriStep_Init(ark_mem: &ARKodeMem, init_type: i32) -> i32 {
    let mut retval: i32;

    /* access ARKodeMRIStepMem structure */
    retval = mriStep_step_mem_ok(ark_mem, "mriStep_Init");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* immediately return if reset */
    if init_type == RESET_INIT {
        return ARK_SUCCESS;
    }

    /* initializations/checks for (re-)initialization call */
    if init_type == FIRST_INIT {
        /* enforce use of arkEwtSmallReal if using a fixed step size for
           an explicit method, an internal error weight function, and not performing
           accumulated temporal error estimation */
        let mut reset_efun: sunbooleantype = SUNTRUE;
        let implicit_rhs = { mriStep_mem_mut(ark_mem).implicit_rhs };
        if implicit_rhs {
            reset_efun = SUNFALSE;
        }
        let (fixedstep, user_efun, accum_type) = {
            let m = ark_mem.borrow();
            (m.fixedstep, m.user_efun, m.AccumErrorType)
        };
        if !fixedstep {
            reset_efun = SUNFALSE;
        }
        if user_efun {
            reset_efun = SUNFALSE;
        }
        if accum_type != ARK_ACCUMERROR_NONE {
            reset_efun = SUNFALSE;
        }
        if reset_efun {
            {
                let mut m = ark_mem.borrow_mut();
                m.user_efun = SUNFALSE;
                m.efun = Some(arkEwtSetSmallReal);
            }
            /* C `ark_mem->e_data = ark_mem`: a boxed handle clone playing the
               same role (the Rc cycle is broken in ARKodeFree) */
            let token: Box<dyn Any> = Box::new(ark_mem.clone());
            ark_mem.borrow_mut().e_data = Some(token);
        }

        /* Create coupling structure (if not already set) */
        retval = mriStep_SetCoupling(ark_mem);
        if retval != ARK_SUCCESS {
            arkProcessError(
                Some(ark_mem),
                ARK_ILL_INPUT,
                line!() as i32,
                "mriStep_Init",
                file!(),
                "Could not create coupling table",
            );
            return ARK_ILL_INPUT;
        }

        /* Check that coupling structure is OK */
        retval = mriStep_CheckCoupling(ark_mem);
        if retval != ARK_SUCCESS {
            arkProcessError(
                Some(ark_mem),
                ARK_ILL_INPUT,
                line!() as i32,
                "mriStep_Init",
                file!(),
                "Error in coupling table",
            );
            return ARK_ILL_INPUT;
        }

        let MRIC = { mriStep_mem_mut(ark_mem).MRIC.clone() }.expect("MRIC set");

        /* Attach correct TakeStep routine for this coupling table.
           (The C `default:` "Unknown method type" branch is unreachable:
           MRISTEP_METHOD_TYPE is a Rust enum.) */
        let mric_type = { MRIC.borrow().type_ };
        match mric_type {
            MRISTEP_EXPLICIT => ark_mem.borrow_mut().step = Some(mriStep_TakeStepMRIGARK),
            MRISTEP_IMPLICIT => ark_mem.borrow_mut().step = Some(mriStep_TakeStepMRIGARK),
            MRISTEP_IMEX => ark_mem.borrow_mut().step = Some(mriStep_TakeStepMRIGARK),
            MRISTEP_MERK => ark_mem.borrow_mut().step = Some(mriStep_TakeStepMERK),
            MRISTEP_SR => ark_mem.borrow_mut().step = Some(mriStep_TakeStepMRISR),
        }

        /* Request arkode ensure that ycur==yn upon entry to TakeStep function */
        ark_mem.borrow_mut().ensure_ycur = SUNTRUE;

        /* Retrieve/store method and embedding orders now that tables are finalized */
        let (mric_stages, mric_q, mric_p, mric_nmat) = {
            let c = MRIC.borrow();
            (c.stages, c.q, c.p, c.nmat)
        };
        {
            let mut s = mriStep_mem_mut(ark_mem);
            s.stages = mric_stages;
            s.q = mric_q;
            s.p = mric_p;
        }
        {
            let mut m = ark_mem.borrow_mut();
            let hadapt_mem = m.hadapt_mem.as_mut().expect("hadapt_mem set");
            hadapt_mem.q = mric_q;
            hadapt_mem.p = mric_p;
        }

        /* Ensure that if adaptivity or error accumulation is enabled, then
           method includes embedding coefficients */
        let (fixedstep, accum_type) = {
            let m = ark_mem.borrow();
            (m.fixedstep, m.AccumErrorType)
        };
        let p = { mriStep_mem_mut(ark_mem).p };
        if (!fixedstep || (accum_type != ARK_ACCUMERROR_NONE)) && (p <= 0) {
            arkProcessError(
                Some(ark_mem),
                ARK_ILL_INPUT,
                line!() as i32,
                "mriStep_Init",
                file!(),
                "Temporal error estimation cannot be performed without embedding coefficients",
            );
            return ARK_ILL_INPUT;
        }

        /* allocate/fill derived quantities from MRIC structure */

        /* stage map */
        let (have_stage_map, stages) = {
            let s = mriStep_mem_mut(ark_mem);
            (!s.stage_map.is_empty(), s.stages)
        };
        if have_stage_map {
            mriStep_mem_mut(ark_mem).stage_map = Vec::new();
            ark_mem.borrow_mut().liw -= stages as i64;
        }
        mriStep_mem_mut(ark_mem).stage_map = vec![0i32; mric_stages as usize];
        ark_mem.borrow_mut().liw += mric_stages as i64;
        {
            let (mut stage_map, mut nstages_active) = {
                let mut s = mriStep_mem_mut(ark_mem);
                (std::mem::take(&mut s.stage_map), s.nstages_active)
            };
            retval = mriStepCoupling_GetStageMap(Some(&MRIC), &mut stage_map, &mut nstages_active);
            let mut s = mriStep_mem_mut(ark_mem);
            s.stage_map = stage_map;
            s.nstages_active = nstages_active;
        }
        if retval != ARK_SUCCESS {
            arkProcessError(
                Some(ark_mem),
                ARK_ILL_INPUT,
                line!() as i32,
                "mriStep_Init",
                file!(),
                "Error in coupling table",
            );
            return ARK_ILL_INPUT;
        }

        /* stage types */
        let have_stagetypes = { !mriStep_mem_mut(ark_mem).stagetypes.is_empty() };
        if have_stagetypes {
            mriStep_mem_mut(ark_mem).stagetypes = Vec::new();
            ark_mem.borrow_mut().liw -= stages as i64;
        }
        mriStep_mem_mut(ark_mem).stagetypes = vec![0i32; (mric_stages + 1) as usize];
        ark_mem.borrow_mut().liw += (mric_stages + 1) as i64;
        for j in 0..=mric_stages {
            let stagetype = mriStepCoupling_GetStageType(&MRIC, j);
            mriStep_mem_mut(ark_mem).stagetypes[j as usize] = stagetype;
        }

        /* explicit RK coefficient row */
        let have_Ae_row = { !mriStep_mem_mut(ark_mem).Ae_row.is_empty() };
        if have_Ae_row {
            mriStep_mem_mut(ark_mem).Ae_row = Vec::new();
            ark_mem.borrow_mut().lrw -= stages as i64;
        }
        mriStep_mem_mut(ark_mem).Ae_row = vec![ZERO; mric_stages as usize];
        ark_mem.borrow_mut().lrw += mric_stages as i64;

        /* implicit RK coefficient row */
        let have_Ai_row = { !mriStep_mem_mut(ark_mem).Ai_row.is_empty() };
        if have_Ai_row {
            mriStep_mem_mut(ark_mem).Ai_row = Vec::new();
            ark_mem.borrow_mut().lrw -= stages as i64;
        }
        mriStep_mem_mut(ark_mem).Ai_row = vec![ZERO; mric_stages as usize];
        ark_mem.borrow_mut().lrw += mric_stages as i64;

        /* Allocate reusable arrays for fused vector operations */
        let nforcing = { mriStep_mem_mut(ark_mem).nforcing };
        let fused_workspace_size: i32 = SUNMAX(3, 2 * mric_stages + 2 + nforcing);

        let nfusedopvecs = { mriStep_mem_mut(ark_mem).nfusedopvecs };
        if nfusedopvecs < fused_workspace_size {
            let (have_cvals, have_Xvecs) = {
                let s = mriStep_mem_mut(ark_mem);
                (!s.cvals.is_empty(), !s.Xvecs.is_empty())
            };
            if have_cvals {
                mriStep_mem_mut(ark_mem).cvals = Vec::new();
                ark_mem.borrow_mut().lrw -= nfusedopvecs as i64;
            }
            if have_Xvecs {
                mriStep_mem_mut(ark_mem).Xvecs = Vec::new();
                ark_mem.borrow_mut().liw -= nfusedopvecs as i64;
            }
            {
                let mut s = mriStep_mem_mut(ark_mem);
                s.nfusedopvecs = 0;

                /* The C calloc-failure branches are unreachable: a Rust
                   allocation failure aborts rather than returning NULL. */
                s.cvals = vec![ZERO; fused_workspace_size as usize];
                s.Xvecs = vec![None; fused_workspace_size as usize];
                s.nfusedopvecs = fused_workspace_size;
            }
            let mut m = ark_mem.borrow_mut();
            m.lrw += fused_workspace_size as i64;
            m.liw += fused_workspace_size as i64;
        }

        /* Retrieve/store method and embedding orders now that tables are finalized */
        {
            let mut s = mriStep_mem_mut(ark_mem);
            s.stages = mric_stages;
            s.q = mric_q;
            s.p = mric_p;

            /* If an MRISR method is applied to a non-ImEx problem, we "unify"
               the Fse and Fsi vectors to point at the same memory */
            s.unify_Fs = SUNFALSE;
            if (mric_type == MRISTEP_SR)
                && ((s.explicit_rhs && !s.implicit_rhs) || (!s.explicit_rhs && s.implicit_rhs))
            {
                s.unify_Fs = SUNTRUE;
            }
        }

        /* Allocate MRI RHS vector memory, update storage requirements */
        /*   Allocate Fse[0] ... Fse[nstages_active - 1] and           */
        /*   Fsi[0] ... Fsi[nstages_active - 1] if needed              */
        let (nstages_allocated, nstages_active, explicit_rhs, implicit_rhs, unify_Fs) = {
            let s = mriStep_mem_mut(ark_mem);
            (
                s.nstages_allocated,
                s.nstages_active,
                s.explicit_rhs,
                s.implicit_rhs,
                s.unify_Fs,
            )
        };
        if nstages_allocated < nstages_active {
            if nstages_allocated != 0 {
                if explicit_rhs {
                    let (lrw1, liw1, mut lrw, mut liw) = {
                        let m = ark_mem.borrow();
                        (m.lrw1, m.liw1, m.lrw, m.liw)
                    };
                    let mut Fse = { std::mem::take(&mut mriStep_mem_mut(ark_mem).Fse) };
                    arkFreeVecArray(nstages_allocated, &mut Fse, lrw1, &mut lrw, liw1, &mut liw);
                    {
                        let mut m = ark_mem.borrow_mut();
                        m.lrw = lrw;
                        m.liw = liw;
                    }
                    let mut s = mriStep_mem_mut(ark_mem);
                    s.Fse = Fse;
                    if s.unify_Fs {
                        s.Fsi = Vec::new();
                    }
                }
                if implicit_rhs {
                    let (lrw1, liw1, mut lrw, mut liw) = {
                        let m = ark_mem.borrow();
                        (m.lrw1, m.liw1, m.lrw, m.liw)
                    };
                    let mut Fsi = { std::mem::take(&mut mriStep_mem_mut(ark_mem).Fsi) };
                    arkFreeVecArray(nstages_allocated, &mut Fsi, lrw1, &mut lrw, liw1, &mut liw);
                    {
                        let mut m = ark_mem.borrow_mut();
                        m.lrw = lrw;
                        m.liw = liw;
                    }
                    let mut s = mriStep_mem_mut(ark_mem);
                    s.Fsi = Fsi;
                    if s.unify_Fs {
                        s.Fse = Vec::new();
                    }
                }
            }
            let ewt = ark_mem.borrow().ewt.clone().expect("ewt set");
            if explicit_rhs && !unify_Fs {
                let (lrw1, liw1, mut lrw, mut liw) = {
                    let m = ark_mem.borrow();
                    (m.lrw1, m.liw1, m.lrw, m.liw)
                };
                let mut Fse = { std::mem::take(&mut mriStep_mem_mut(ark_mem).Fse) };
                let ok = arkAllocVecArray(
                    nstages_active,
                    &ewt,
                    &mut Fse,
                    lrw1,
                    &mut lrw,
                    liw1,
                    &mut liw,
                );
                {
                    let mut m = ark_mem.borrow_mut();
                    m.lrw = lrw;
                    m.liw = liw;
                }
                mriStep_mem_mut(ark_mem).Fse = Fse;
                if !ok {
                    return ARK_MEM_FAIL;
                }
            }
            if implicit_rhs && !unify_Fs {
                let (lrw1, liw1, mut lrw, mut liw) = {
                    let m = ark_mem.borrow();
                    (m.lrw1, m.liw1, m.lrw, m.liw)
                };
                let mut Fsi = { std::mem::take(&mut mriStep_mem_mut(ark_mem).Fsi) };
                let ok = arkAllocVecArray(
                    nstages_active,
                    &ewt,
                    &mut Fsi,
                    lrw1,
                    &mut lrw,
                    liw1,
                    &mut liw,
                );
                {
                    let mut m = ark_mem.borrow_mut();
                    m.lrw = lrw;
                    m.liw = liw;
                }
                mriStep_mem_mut(ark_mem).Fsi = Fsi;
                if !ok {
                    return ARK_MEM_FAIL;
                }
            }
            if unify_Fs {
                let (lrw1, liw1, mut lrw, mut liw) = {
                    let m = ark_mem.borrow();
                    (m.lrw1, m.liw1, m.lrw, m.liw)
                };
                let mut Fse = { std::mem::take(&mut mriStep_mem_mut(ark_mem).Fse) };
                let ok = arkAllocVecArray(
                    nstages_active,
                    &ewt,
                    &mut Fse,
                    lrw1,
                    &mut lrw,
                    liw1,
                    &mut liw,
                );
                {
                    let mut m = ark_mem.borrow_mut();
                    m.lrw = lrw;
                    m.liw = liw;
                }
                mriStep_mem_mut(ark_mem).Fse = Fse;
                if !ok {
                    return ARK_MEM_FAIL;
                }
                let mut s = mriStep_mem_mut(ark_mem);
                s.Fsi = s.Fse.clone();
            }

            mriStep_mem_mut(ark_mem).nstages_allocated = nstages_active;
        }

        /* if any slow stage is implicit, allocate sdata, zpred, zcor vectors;
           if all stages explicit, free default NLS object, and detach all
           linear solver routines.  Note: step_mem->implicit_rhs will only equal
           SUNTRUE if an implicit table has been user-provided. */
        if implicit_rhs {
            let ewt = ark_mem.borrow().ewt.clone().expect("ewt set");
            {
                let mut sdata = { mriStep_mem_mut(ark_mem).sdata.clone() };
                let ok = arkAllocVec(ark_mem, &ewt, &mut sdata);
                mriStep_mem_mut(ark_mem).sdata = sdata;
                if !ok {
                    return ARK_MEM_FAIL;
                }
            }
            {
                let mut zpred = { mriStep_mem_mut(ark_mem).zpred.clone() };
                let ok = arkAllocVec(ark_mem, &ewt, &mut zpred);
                mriStep_mem_mut(ark_mem).zpred = zpred;
                if !ok {
                    return ARK_MEM_FAIL;
                }
            }
            {
                let mut zcor = { mriStep_mem_mut(ark_mem).zcor.clone() };
                let ok = arkAllocVec(ark_mem, &ewt, &mut zcor);
                mriStep_mem_mut(ark_mem).zcor = zcor;
                if !ok {
                    return ARK_MEM_FAIL;
                }
            }
        } else {
            let free_nls = {
                let s = mriStep_mem_mut(ark_mem);
                s.NLS.is_some() && s.ownNLS
            };
            if free_nls {
                let nls = { mriStep_mem_mut(ark_mem).NLS.take() };
                let _ = SUNNonlinSolFree(nls);
                let mut s = mriStep_mem_mut(ark_mem);
                s.NLS = None;
                s.ownNLS = SUNFALSE;
            }
            {
                let mut s = mriStep_mem_mut(ark_mem);
                s.linit = None;
                s.lsetup = None;
                s.lsolve = None;
                s.lfree = None;
            }
            /* C `step_mem->lmem = NULL` (the record lives in ark_mem, §4) */
            ark_mem.borrow_mut().ark_lmem = None;
        }

        /* Allocate inner stepper data */
        let stepper = { mriStep_mem_mut(ark_mem).stepper.clone() }.expect("stepper set");
        let ewt = ark_mem.borrow().ewt.clone().expect("ewt set");
        retval = mriStepInnerStepper_AllocVecs(&stepper, mric_nmat, &ewt);
        if retval != ARK_SUCCESS {
            arkProcessError(
                Some(ark_mem),
                ARK_ILL_INPUT,
                line!() as i32,
                "mriStep_Init",
                file!(),
                "Error allocating inner stepper memory",
            );
            return ARK_MEM_FAIL;
        }

        /* Override the interpolant degree (if needed), used in arkInitialSetup */
        let q = { mriStep_mem_mut(ark_mem).q };
        let interp_degree = ark_mem.borrow().interp_degree;
        if q > 1 && interp_degree > (q - 1) {
            /* Limit max degree to at most one less than the method global order */
            ark_mem.borrow_mut().interp_degree = q - 1;
        } else if q == 1 && interp_degree > 1 {
            /* Allow for linear interpolant with first order methods to ensure
               solution values are returned at the time interval end points */
            ark_mem.borrow_mut().interp_degree = 1;
        }

        /* Higher-order predictors require interpolation */
        let interp_type = ark_mem.borrow().interp_type;
        let predictor = { mriStep_mem_mut(ark_mem).predictor };
        if interp_type == ARK_INTERP_NONE && predictor != 0 {
            arkProcessError(
                Some(ark_mem),
                ARK_ILL_INPUT,
                line!() as i32,
                "mriStep_Init",
                file!(),
                "Non-trival predictors require an interpolation module",
            );
            return ARK_ILL_INPUT;
        }
    }

    /* Call linit (if it exists) */
    let linit = { mriStep_mem_mut(ark_mem).linit };
    if let Some(linit) = linit {
        retval = linit(ark_mem);
        if retval != 0 {
            arkProcessError(
                Some(ark_mem),
                ARK_LINIT_FAIL,
                line!() as i32,
                "mriStep_Init",
                file!(),
                MSG_ARK_LINIT_FAIL,
            );
            return ARK_LINIT_FAIL;
        }
    }

    /* Initialize the nonlinear solver object (if it exists) */
    let have_nls = { mriStep_mem_mut(ark_mem).NLS.is_some() };
    if have_nls {
        retval = mriStep_NlsInit(ark_mem);
        if retval != ARK_SUCCESS {
            arkProcessError(
                Some(ark_mem),
                ARK_NLS_INIT_FAIL,
                line!() as i32,
                "mriStep_Init",
                file!(),
                "Unable to initialize SUNNonlinearSolver object",
            );
            return ARK_NLS_INIT_FAIL;
        }
    }

    /* get timestep adaptivity type */
    let hcontroller = ark_mem
        .borrow()
        .hadapt_mem
        .as_ref()
        .expect("hadapt_mem set")
        .hcontroller
        .clone()
        .expect("hcontroller set");
    let adapt_type = SUNAdaptController_GetType(&hcontroller);

    let fixedstep = ark_mem.borrow().fixedstep;
    if fixedstep {
        /* Fixed step sizes: user must supply the initial step size */
        if ark_mem.borrow().hin == ZERO {
            arkProcessError(
                Some(ark_mem),
                ARK_ILL_INPUT,
                line!() as i32,
                "mriStep_Init",
                file!(),
                "Timestep adaptivity disabled, but missing user-defined fixed stepsize",
            );
            return ARK_ILL_INPUT;
        }
    } else {
        /* ensure that a compatible adaptivity controller is provided */
        if (adapt_type != SUN_ADAPTCONTROLLER_MRI_H_TOL) && (adapt_type != SUN_ADAPTCONTROLLER_H)
        {
            arkProcessError(
                Some(ark_mem),
                ARK_ILL_INPUT,
                line!() as i32,
                "mriStep_Init",
                file!(),
                "SUNAdaptController type is unsupported by MRIStep",
            );
            return ARK_ILL_INPUT;
        }

        /* Controller provides adaptivity (at least at the slow time scale):
           - verify that the MRI method includes an embedding, and
           - estimate initial slow step size (store in ark_mem->hin) */
        let mric_p = {
            let MRIC = { mriStep_mem_mut(ark_mem).MRIC.clone() }.expect("MRIC set");
            let p = MRIC.borrow().p;
            p
        };
        if mric_p <= 0 {
            arkProcessError(
                Some(ark_mem),
                ARK_ILL_INPUT,
                line!() as i32,
                "mriStep_Init",
                file!(),
                "Timestep adaptivity enabled, but non-embedded MRI table specified",
            );
            return ARK_ILL_INPUT;
        }
    }

    /* Perform additional setup for (H,tol) controller */
    if adapt_type == SUN_ADAPTCONTROLLER_MRI_H_TOL {
        /* Verify that adaptivity type is supported by inner stepper */
        let stepper = { mriStep_mem_mut(ark_mem).stepper.clone() }.expect("stepper set");
        if !mriStepInnerStepper_SupportsRTolAdaptivity(&stepper) {
            arkProcessError(
                Some(ark_mem),
                ARK_ILL_INPUT,
                line!() as i32,
                "mriStep_Init",
                file!(),
                "MRI H-TOL SUNAdaptController provided, but unsupported by inner stepper",
            );
            return ARK_ILL_INPUT;
        }

        /* initialize fast stepper to use the same relative tolerance as MRIStep */
        mriStep_mem_mut(ark_mem).inner_rtol_factor = ONE;
    }

    ARK_SUCCESS
}

/*------------------------------------------------------------------------------
  mriStep_ComputeH0:

  This utility routine computes the initial slow step size for MRI methods.

  It is assumed that the IVP is defined by multiple RHS functions,
     y'(t) = f(t,y) = fs(t,y)  + ff(t,y),
  where fs corresponds to dynamics that should be evolved directly by MRIStep,
  and ff corresponds to dynamics that will be evolved by an inner stepper.
  ----------------------------------------------------------------------------*/
pub fn mriStep_ComputeH0(ark_mem: &ARKodeMem, tout: sunrealtype, hin: &mut sunrealtype) -> i32 {
    let mut retval: i32;

    /*   tempv1 = fs(t0, y0) */
    let (tn, yn, tempv1) = {
        let m = ark_mem.borrow();
        (
            m.tn,
            m.yn.clone().expect("yn set"),
            m.tempv1.clone().expect("tempv1 set"),
        )
    };
    if mriStep_SlowRHS(ark_mem, tn, &yn, &tempv1, ARK_FULLRHS_START) != ARK_SUCCESS {
        arkProcessError(
            Some(ark_mem),
            ARK_RHSFUNC_FAIL,
            line!() as i32,
            "mriStep_ComputeH0",
            file!(),
            "error calling slow RHS function(s)",
        );
        return ARK_RHSFUNC_FAIL;
    }
    retval = mriStep_Hin(ark_mem, tn, tout, &tempv1, hin);
    if retval != ARK_SUCCESS {
        retval = arkHandleFailure(ark_mem, retval);
        return retval;
    }

    ARK_SUCCESS
}

/*------------------------------------------------------------------------------
  mriStep_FullRHS:

  This is just a wrapper to call the user-supplied RHS functions,
  f(t,y) = fse(t,y) + fsi(t,y)  + ff(t,y).

  Note: this relies on the utility routine mriStep_UpdateF0 to update Fse[0]
  and Fsi[0] as appropriate (i.e., leveraging previous evaluations, etc.), and
  merely combines the resulting values together with ff to construct the output.

  However, in ARK_FULLRHS_OTHER mode, this routine must call all slow RHS
  functions directly, since that mode cannot reuse internally stored values.

   ARK_FULLRHS_OTHER -> called in the following circumstances:
                        (a) when estimating the initial time step size,
                        (b) for high-order dense output with the Hermite
                            interpolation module,
                        (c) by an "outer" stepper when MRIStep is used as an
                            inner solver), or
                        (d) when a high-order implicit predictor is requested from
                            the Hermite interpolation module within the time step
                            t_{n} \to t_{n+1}.

                        While instances (a)-(c) will occur in-between MRIStep time
                        steps, instance (d) can occur at the start of each internal
                        MRIStep stage.  Since the (t,y) input does not correspond
                        to an "official" time step, thus the RHS functions should
                        always be evaluated, and the values should *not* be stored
                        anywhere that will interfere with other reused MRIStep data
                        from one stage to the next (but it may use nonlinear solver
                        scratch space).

  Note that this routine always calls the fast RHS function, ff(t,y), in
  ARK_FULLRHS_OTHER mode.
  ----------------------------------------------------------------------------*/
pub fn mriStep_FullRHS(
    ark_mem: &ARKodeMem,
    t: sunrealtype,
    y: &N_Vector,
    f: &N_Vector,
    mode: i32,
) -> i32 {
    let mut nvec: i32;
    let mut retval: i32;

    /* access ARKodeMRIStepMem structure */
    retval = mriStep_step_mem_ok(ark_mem, "mriStep_FullRHS");
    if retval != ARK_SUCCESS {
        return retval;
    }

    let stepper = { mriStep_mem_mut(ark_mem).stepper.clone() }.expect("stepper set");

    /* ensure that inner stepper provides fullrhs function */
    let has_fullrhs = stepper.ops.borrow().fullrhs.is_some();
    if !has_fullrhs {
        arkProcessError(
            Some(ark_mem),
            ARK_RHSFUNC_FAIL,
            line!() as i32,
            "mriStep_FullRHS",
            file!(),
            MSG_ARK_MISSING_FULLRHS,
        );
        return ARK_RHSFUNC_FAIL;
    }

    /* perform RHS functions contingent on 'mode' argument */
    if mode == ARK_FULLRHS_START || mode == ARK_FULLRHS_END {
        /* update the internal storage for Fse[0] and Fsi[0] */
        retval = mriStep_UpdateF0(ark_mem, t, y, mode);
        if retval != 0 {
            arkProcessError(
                Some(ark_mem),
                ARK_RHSFUNC_FAIL,
                line!() as i32,
                "mriStep_FullRHS",
                file!(),
                &MSG_ARK_RHSFUNC_FAILED(t),
            );
            return ARK_RHSFUNC_FAIL;
        }

        /* evaluate fast component */
        retval = mriStepInnerStepper_FullRhs(&stepper, t, y, f, ARK_FULLRHS_OTHER);
        if retval != ARK_SUCCESS {
            arkProcessError(
                Some(ark_mem),
                ARK_RHSFUNC_FAIL,
                line!() as i32,
                "mriStep_FullRHS",
                file!(),
                &MSG_ARK_RHSFUNC_FAILED(t),
            );
            return ARK_RHSFUNC_FAIL;
        }

        /* combine RHS vectors into output */
        let (explicit_rhs, implicit_rhs) = {
            let s = mriStep_mem_mut(ark_mem);
            (s.explicit_rhs, s.implicit_rhs)
        };
        if explicit_rhs && implicit_rhs {
            /* ImEx */
            let (cvals, Xvecs) = {
                let mut s = mriStep_mem_mut(ark_mem);
                s.cvals[0] = ONE;
                s.Xvecs[0] = Some(f.clone());
                s.cvals[1] = ONE;
                let v = s.Fse[0].clone();
                s.Xvecs[1] = Some(v);
                s.cvals[2] = ONE;
                let v = s.Fsi[0].clone();
                s.Xvecs[2] = Some(v);
                (s.cvals.clone(), mriStep_xvecs(&s, 3))
            };
            nvec = 3;
            let _ = N_VLinearCombination(nvec, &cvals, &Xvecs, f);
        } else if implicit_rhs {
            /* implicit */
            let v = { mriStep_mem_mut(ark_mem).Fsi[0].clone() };
            N_VLinearSum(ONE, &v, ONE, f, f);
        } else {
            /* explicit */
            let v = { mriStep_mem_mut(ark_mem).Fse[0].clone() };
            N_VLinearSum(ONE, &v, ONE, f, f);
        }
    } else if mode == ARK_FULLRHS_OTHER {
        /* compute the fast component (force new RHS computation) */
        nvec = 0;
        retval = mriStepInnerStepper_FullRhs(&stepper, t, y, f, ARK_FULLRHS_OTHER);
        if retval != ARK_SUCCESS {
            arkProcessError(
                Some(ark_mem),
                ARK_RHSFUNC_FAIL,
                line!() as i32,
                "mriStep_FullRHS",
                file!(),
                &MSG_ARK_RHSFUNC_FAILED(t),
            );
            return ARK_RHSFUNC_FAIL;
        }
        {
            let mut s = mriStep_mem_mut(ark_mem);
            s.cvals[nvec as usize] = ONE;
            s.Xvecs[nvec as usize] = Some(f.clone());
        }
        nvec += 1;

        /* call the user-supplied pre-RHS function (if supplied) */
        if ark_mem.borrow().PreRhsFn.is_some() {
            retval = mriStep_CallPreRhsFn(ark_mem, t, y);
            if retval != 0 {
                return ARK_PRERHSFN_FAIL;
            }
        }

        let (explicit_rhs, implicit_rhs) = {
            let s = mriStep_mem_mut(ark_mem);
            (s.explicit_rhs, s.implicit_rhs)
        };

        /* compute the implicit component and store in sdata */
        if implicit_rhs {
            let sdata = { mriStep_mem_mut(ark_mem).sdata.clone() }.expect("sdata set");
            retval = mriStep_CallFsi(ark_mem, t, y, &sdata);
            mriStep_mem_mut(ark_mem).nfsi += 1;
            if retval != 0 {
                arkProcessError(
                    Some(ark_mem),
                    ARK_RHSFUNC_FAIL,
                    line!() as i32,
                    "mriStep_FullRHS",
                    file!(),
                    &MSG_ARK_RHSFUNC_FAILED(t),
                );
                return ARK_RHSFUNC_FAIL;
            }
            {
                let mut s = mriStep_mem_mut(ark_mem);
                s.cvals[nvec as usize] = ONE;
                s.Xvecs[nvec as usize] = Some(sdata);
            }
            nvec += 1;
        }

        /* compute the explicit component and store in ark_tempv2 */
        if explicit_rhs {
            let tempv2 = ark_mem.borrow().tempv2.clone().expect("tempv2 set");
            retval = mriStep_CallFse(ark_mem, t, y, &tempv2);
            mriStep_mem_mut(ark_mem).nfse += 1;
            if retval != 0 {
                arkProcessError(
                    Some(ark_mem),
                    ARK_RHSFUNC_FAIL,
                    line!() as i32,
                    "mriStep_FullRHS",
                    file!(),
                    &MSG_ARK_RHSFUNC_FAILED(t),
                );
                return ARK_RHSFUNC_FAIL;
            }
            {
                let mut s = mriStep_mem_mut(ark_mem);
                s.cvals[nvec as usize] = ONE;
                s.Xvecs[nvec as usize] = Some(tempv2);
            }
            nvec += 1;
        }

        /* Add external forcing components to linear combination */
        let (expforcing, impforcing) = {
            let s = mriStep_mem_mut(ark_mem);
            (s.expforcing, s.impforcing)
        };
        if expforcing || impforcing {
            let mut s = mriStep_mem_mut(ark_mem);
            mriStep_ApplyForcing(&mut s, t, ONE, &mut nvec);
        }

        /* combine RHS vectors into output */
        let (cvals, Xvecs) = {
            let s = mriStep_mem_mut(ark_mem);
            (s.cvals.clone(), mriStep_xvecs(&s, nvec))
        };
        let _ = N_VLinearCombination(nvec, &cvals, &Xvecs, f);
    } else {
        /* return with RHS failure if unknown mode is passed */
        arkProcessError(
            Some(ark_mem),
            ARK_RHSFUNC_FAIL,
            line!() as i32,
            "mriStep_FullRHS",
            file!(),
            "Unknown full RHS mode",
        );
        return ARK_RHSFUNC_FAIL;
    }

    ARK_SUCCESS
}

/*------------------------------------------------------------------------------
  mriStep_UpdateF0:

  This routine is called by mriStep_FullRHS to update the internal storage for
  Fse[0] and Fsi[0], incorporating forcing from a slower time scale as necessary.
  This supports the ARK_FULLRHS_START and ARK_FULLRHS_END "mode" values
  provided to mriStep_FullRHS, and contains all internal logic regarding whether
  RHS functions must be called, versus if the relevant data can just be copied.

  (See the C source for the full ARK_FULLRHS_START / ARK_FULLRHS_END commentary.)

  The C `ARKodeMRIStepMem step_mem` parameter is dropped: the record lives
  inside `ark_mem` and is reached through `mriStep_mem_mut` (an `&mut` to it
  could not coexist with the `&ARKodeMem` this routine also needs).
  ----------------------------------------------------------------------------*/
pub fn mriStep_UpdateF0(
    ark_mem: &ARKodeMem,
    t: sunrealtype,
    y: &N_Vector,
    mode: i32,
) -> i32 {
    let mut nvec: i32;
    let mut retval: i32;

    /* perform RHS functions contingent on 'mode' argument */
    if mode == ARK_FULLRHS_START {
        /* update the RHS components */

        let (fse_is_current, fsi_is_current, explicit_rhs, implicit_rhs, expforcing, impforcing) = {
            let s = mriStep_mem_mut(ark_mem);
            (
                s.fse_is_current,
                s.fsi_is_current,
                s.explicit_rhs,
                s.implicit_rhs,
                s.expforcing,
                s.impforcing,
            )
        };
        let fn_is_current = ark_mem.borrow().fn_is_current;

        /* call the user-supplied pre-RHS function (if supplied) */
        if ark_mem.borrow().PreRhsFn.is_some()
            && ((!fse_is_current || !fn_is_current) || (!fsi_is_current || !fn_is_current))
        {
            retval = mriStep_CallPreRhsFn(ark_mem, t, y);
            if retval != 0 {
                return ARK_PRERHSFN_FAIL;
            }
        }

        /*   implicit component */
        if implicit_rhs {
            /* if either ARKODE or MRIStep consider Fsi[0] stale, then recompute */
            if !fsi_is_current || !fn_is_current {
                let Fsi0 = { mriStep_mem_mut(ark_mem).Fsi[0].clone() };
                retval = mriStep_CallFsi(ark_mem, t, y, &Fsi0);
                mriStep_mem_mut(ark_mem).nfsi += 1;
                if retval != 0 {
                    arkProcessError(
                        Some(ark_mem),
                        ARK_RHSFUNC_FAIL,
                        line!() as i32,
                        "mriStep_UpdateF0",
                        file!(),
                        &MSG_ARK_RHSFUNC_FAILED(t),
                    );
                    return ARK_RHSFUNC_FAIL;
                }
                mriStep_mem_mut(ark_mem).fsi_is_current = SUNTRUE;

                /* Add external forcing, if applicable */
                if impforcing {
                    let (cvals, Xvecs) = {
                        let mut s = mriStep_mem_mut(ark_mem);
                        s.cvals[0] = ONE;
                        let v = s.Fsi[0].clone();
                        s.Xvecs[0] = Some(v);
                        nvec = 1;
                        mriStep_ApplyForcing(&mut s, t, ONE, &mut nvec);
                        (s.cvals.clone(), mriStep_xvecs(&s, nvec))
                    };
                    let _ = N_VLinearCombination(Xvecs.len() as i32, &cvals, &Xvecs, &Fsi0);
                }
            }
        }

        /*   explicit component */
        if explicit_rhs {
            /* if either ARKODE or MRIStep consider Fse[0] stale, then recompute */
            if !fse_is_current || !fn_is_current {
                let Fse0 = { mriStep_mem_mut(ark_mem).Fse[0].clone() };
                retval = mriStep_CallFse(ark_mem, t, y, &Fse0);
                mriStep_mem_mut(ark_mem).nfse += 1;
                if retval != 0 {
                    arkProcessError(
                        Some(ark_mem),
                        ARK_RHSFUNC_FAIL,
                        line!() as i32,
                        "mriStep_UpdateF0",
                        file!(),
                        &MSG_ARK_RHSFUNC_FAILED(t),
                    );
                    return ARK_RHSFUNC_FAIL;
                }
                mriStep_mem_mut(ark_mem).fse_is_current = SUNTRUE;

                /* Add external forcing, if applicable */
                if expforcing {
                    let (cvals, Xvecs) = {
                        let mut s = mriStep_mem_mut(ark_mem);
                        s.cvals[0] = ONE;
                        let v = s.Fse[0].clone();
                        s.Xvecs[0] = Some(v);
                        nvec = 1;
                        mriStep_ApplyForcing(&mut s, t, ONE, &mut nvec);
                        (s.cvals.clone(), mriStep_xvecs(&s, nvec))
                    };
                    let _ = N_VLinearCombination(Xvecs.len() as i32, &cvals, &Xvecs, &Fse0);
                }
            }
        }
    } else if mode == ARK_FULLRHS_END {
        /* compute the full RHS */
        if !ark_mem.borrow().fn_is_current {
            /* call the user-supplied pre-RHS function (if supplied) */
            if ark_mem.borrow().PreRhsFn.is_some() {
                retval = mriStep_CallPreRhsFn(ark_mem, t, y);
                if retval != 0 {
                    return ARK_PRERHSFN_FAIL;
                }
            }

            let (explicit_rhs, implicit_rhs, expforcing, impforcing) = {
                let s = mriStep_mem_mut(ark_mem);
                (s.explicit_rhs, s.implicit_rhs, s.expforcing, s.impforcing)
            };

            /* compute the implicit component */
            if implicit_rhs {
                let Fsi0 = { mriStep_mem_mut(ark_mem).Fsi[0].clone() };
                retval = mriStep_CallFsi(ark_mem, t, y, &Fsi0);
                mriStep_mem_mut(ark_mem).nfsi += 1;
                if retval != 0 {
                    arkProcessError(
                        Some(ark_mem),
                        ARK_RHSFUNC_FAIL,
                        line!() as i32,
                        "mriStep_UpdateF0",
                        file!(),
                        &MSG_ARK_RHSFUNC_FAILED(t),
                    );
                    return ARK_RHSFUNC_FAIL;
                }
                mriStep_mem_mut(ark_mem).fsi_is_current = SUNTRUE;

                /* Add external forcing, as appropriate */
                if impforcing {
                    let (cvals, Xvecs) = {
                        let mut s = mriStep_mem_mut(ark_mem);
                        s.cvals[0] = ONE;
                        let v = s.Fsi[0].clone();
                        s.Xvecs[0] = Some(v);
                        nvec = 1;
                        mriStep_ApplyForcing(&mut s, t, ONE, &mut nvec);
                        (s.cvals.clone(), mriStep_xvecs(&s, nvec))
                    };
                    let _ = N_VLinearCombination(Xvecs.len() as i32, &cvals, &Xvecs, &Fsi0);
                }
            }

            /* compute the explicit component */
            if explicit_rhs {
                let Fse0 = { mriStep_mem_mut(ark_mem).Fse[0].clone() };
                retval = mriStep_CallFse(ark_mem, t, y, &Fse0);
                mriStep_mem_mut(ark_mem).nfse += 1;
                if retval != 0 {
                    arkProcessError(
                        Some(ark_mem),
                        ARK_RHSFUNC_FAIL,
                        line!() as i32,
                        "mriStep_UpdateF0",
                        file!(),
                        &MSG_ARK_RHSFUNC_FAILED(t),
                    );
                    return ARK_RHSFUNC_FAIL;
                }
                mriStep_mem_mut(ark_mem).fse_is_current = SUNTRUE;

                /* Add external forcing, as appropriate */
                if expforcing {
                    let (cvals, Xvecs) = {
                        let mut s = mriStep_mem_mut(ark_mem);
                        s.cvals[0] = ONE;
                        let v = s.Fse[0].clone();
                        s.Xvecs[0] = Some(v);
                        nvec = 1;
                        mriStep_ApplyForcing(&mut s, t, ONE, &mut nvec);
                        (s.cvals.clone(), mriStep_xvecs(&s, nvec))
                    };
                    let _ = N_VLinearCombination(Xvecs.len() as i32, &cvals, &Xvecs, &Fse0);
                }
            }
        }
    } else {
        /* return with RHS failure if unknown mode is requested */
        arkProcessError(
            Some(ark_mem),
            ARK_RHSFUNC_FAIL,
            line!() as i32,
            "mriStep_UpdateF0",
            file!(),
            "Unknown full RHS mode",
        );
        return ARK_RHSFUNC_FAIL;
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  mriStep_TakeStepMRIGARK:

  This routine serves the primary purpose of the MRIStep module:
  it performs a single MRI step (with embedding, if possible).

  Both the vectors ark_mem->yn and ark_mem->ycur hold the previous
  time-step solution on input, and the vector ark_mem->ycur should
  hold the result of this step on output.

  If timestep adaptivity is enabled, this routine also computes
  the error estimate y-ytilde, where ytilde is the
  embedded solution, and the norm weights come from ark_ewt.
  This estimate is stored in ark_mem->tempv1, in case the calling
  routine wishes to examine the error locations.

  The output variable dsmPtr should contain a scalar-valued
  estimate of the temporal error from this step, ||y-ytilde||_WRMS
  if timestep adaptivity is enabled; otherwise it should be 0.

  The input/output variable nflagPtr is used to gauge convergence
  of any algebraic solvers within the step.  At the start of a new
  time step, this will initially have the value FIRST_CALL.  On
  return from this function, nflagPtr should have a value:
            0 => algebraic solve completed successfully
           >0 => solve did not converge at this step size
                 (but may with a smaller stepsize)
           <0 => solve encountered an unrecoverable failure
  Since the fast-scale evolution could be considered a different
  type of "algebraic solver", we similarly report any fast-scale
  evolution error as a recoverable nflagPtr value.

  The return value from this routine is:
            0 => step completed successfully
           >0 => step encountered recoverable failure;
                 reduce step and retry (if possible)
           <0 => step encountered unrecoverable failure
  ---------------------------------------------------------------*/
pub fn mriStep_TakeStepMRIGARK(
    ark_mem: &ARKodeMem,
    dsmPtr: &mut sunrealtype,
    nflagPtr: &mut i32,
) -> i32 {
    let mut is: i32; /* current stage index        */
    /* Like C, `retval` is one reused local: the `_ => {}` match arms below
       (C `switch` cases with no `default:`) deliberately leave the previous
       value in place. */
    let mut retval: i32;
    let mut t0: sunrealtype;
    let mut tf: sunrealtype; /* start/end of each stage    */
    let mut calc_fslow: sunbooleantype;
    let mut need_inner_dsm: sunbooleantype;
    let do_embedding: sunbooleantype;
    let nested_mri: sunbooleantype;
    let mut nvec: i32;

    /* access the MRIStep mem structure */
    retval = mriStep_step_mem_ok(ark_mem, "mriStep_TakeStepMRIGARK");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* initialize algebraic solver convergence flag to success;
       error estimate to zero */
    *nflagPtr = ARK_SUCCESS;
    *dsmPtr = ZERO;

    /* determine whether embedding stage is needed */
    let (fixedstep, accum_type) = {
        let m = ark_mem.borrow();
        (m.fixedstep, m.AccumErrorType)
    };
    do_embedding = !fixedstep || (accum_type != ARK_ACCUMERROR_NONE);

    /* initialize the current stage index */
    {
        let mut s = mriStep_mem_mut(ark_mem);
        s.istage = 0;
        s.cur_stage = 0;
    }

    let stepper = { mriStep_mem_mut(ark_mem).stepper.clone() }.expect("stepper set");

    /* if MRI adaptivity is enabled: reset fast accumulated error,
       and send appropriate control parameter to the fast integrator */
    let hcontroller = ark_mem
        .borrow()
        .hadapt_mem
        .as_ref()
        .expect("hadapt_mem set")
        .hcontroller
        .clone()
        .expect("hcontroller set");
    let adapt_type = SUNAdaptController_GetType(&hcontroller);
    need_inner_dsm = SUNFALSE;
    if adapt_type == SUN_ADAPTCONTROLLER_MRI_H_TOL {
        need_inner_dsm = SUNTRUE;
        mriStep_mem_mut(ark_mem).inner_dsm = ZERO;
        retval = mriStepInnerStepper_ResetAccumulatedError(&stepper);
        if retval != ARK_SUCCESS {
            arkProcessError(
                Some(ark_mem),
                ARK_INNERSTEP_FAIL,
                line!() as i32,
                "mriStep_TakeStepMRIGARK",
                file!(),
                "Unable to reset the inner stepper error estimate",
            );
            return ARK_INNERSTEP_FAIL;
        }
        let inner_rtol_factor = { mriStep_mem_mut(ark_mem).inner_rtol_factor };
        let reltol = ark_mem.borrow().reltol;
        retval = mriStepInnerStepper_SetRTol(&stepper, inner_rtol_factor * reltol);
        if retval != ARK_SUCCESS {
            arkProcessError(
                Some(ark_mem),
                ARK_INNERSTEP_FAIL,
                line!() as i32,
                "mriStep_TakeStepMRIGARK",
                file!(),
                "Unable to set the inner stepper tolerance",
            );
            return ARK_INNERSTEP_FAIL;
        }
    }

    /* for adaptive computations, reset the inner integrator to the beginning of this step */
    if !fixedstep {
        let (tcur, ycur) = {
            let m = ark_mem.borrow();
            (m.tcur, m.ycur.clone().expect("ycur set"))
        };
        retval = mriStepInnerStepper_Reset(&stepper, tcur, &ycur);
        if retval != ARK_SUCCESS {
            arkProcessError(
                Some(ark_mem),
                ARK_INNERSTEP_FAIL,
                line!() as i32,
                "mriStep_TakeStepMRIGARK",
                file!(),
                "Unable to reset the inner stepper",
            );
            return ARK_INNERSTEP_FAIL;
        }
    }

    /* call nonlinear solver setup if it exists */
    let NLS = { mriStep_mem_mut(ark_mem).NLS.clone() };
    if let Some(NLS) = NLS {
        if NLS.ops.borrow().setup.is_some() {
            let tempv3 = ark_mem.borrow().tempv3.clone().expect("tempv3 set");
            N_VConst(ZERO, &tempv3); /* set guess to 0 */
            let mut nls_mem: Option<Box<dyn Any>> = Some(Box::new(ark_mem.clone()));
            retval = SUNNonlinSolSetup(&NLS, &tempv3, &mut nls_mem);
            if retval < 0 {
                return ARK_NLS_SETUP_FAIL;
            }
            if retval > 0 {
                return ARK_NLS_SETUP_RECVR;
            }
        }
    }

    /* Evaluate the slow RHS functions if needed. NOTE: we decide between calling the
       full RHS function (if ark_mem->fn is non-NULL and MRIStep is not an inner
       integrator) versus just updating the stored values of Fse[0] and Fsi[0].  In
       either case, we use ARK_FULLRHS_START mode because MRIGARK methods do not
       evaluate the RHS functions at the end of the time step (so nothing can be
       leveraged). */
    let (expforcing, impforcing) = {
        let s = mriStep_mem_mut(ark_mem);
        (s.expforcing, s.impforcing)
    };
    nested_mri = expforcing || impforcing;
    let fn_is_null = ark_mem.borrow().fn_.is_none();
    if fn_is_null || nested_mri {
        let (tcur, ycur) = {
            let m = ark_mem.borrow();
            (m.tcur, m.ycur.clone().expect("ycur set"))
        };
        retval = mriStep_UpdateF0(ark_mem, tcur, &ycur, ARK_FULLRHS_START);
        if retval != 0 {
            return ARK_RHSFUNC_FAIL;
        }

        /* For a nested MRI configuration we might still need fn to create a predictor
           but it should be fn only for the current nesting level which is why we use
           UpdateF0 in this case rather than FullRHS */
        let fn_v = ark_mem.borrow().fn_.clone();
        let (explicit_rhs, implicit_rhs) = {
            let s = mriStep_mem_mut(ark_mem);
            (s.explicit_rhs, s.implicit_rhs)
        };
        if fn_v.is_some() && nested_mri && implicit_rhs {
            let fn_v = fn_v.expect("fn set");
            if implicit_rhs && explicit_rhs {
                let (Fsi0, Fse0) = {
                    let s = mriStep_mem_mut(ark_mem);
                    (s.Fsi[0].clone(), s.Fse[0].clone())
                };
                N_VLinearSum(ONE, &Fsi0, ONE, &Fse0, &fn_v);
            } else {
                let Fsi0 = { mriStep_mem_mut(ark_mem).Fsi[0].clone() };
                N_VScale(ONE, &Fsi0, &fn_v);
            }
        }
    } else if !fn_is_null && !ark_mem.borrow().fn_is_current {
        let (tcur, ycur, fn_v) = {
            let m = ark_mem.borrow();
            (
                m.tcur,
                m.ycur.clone().expect("ycur set"),
                m.fn_.clone().expect("fn set"),
            )
        };
        retval = mriStep_FullRHS(ark_mem, tcur, &ycur, &fn_v, ARK_FULLRHS_START);
        if retval != 0 {
            return ARK_RHSFUNC_FAIL;
        }
    }
    ark_mem.borrow_mut().fn_is_current = SUNTRUE;

    /* The first stage is the previous time-step solution, so its RHS
       is the [already-computed] slow RHS from the start of the step */

    let stages = { mriStep_mem_mut(ark_mem).stages };
    let MRIC = { mriStep_mem_mut(ark_mem).MRIC.clone() }.expect("MRIC set");

    /* Loop over remaining internal stages */
    is = 1;
    while is < stages - 1 {
        /* Set relevant stage times (including desired stage time for implicit solves)
           and stage index */
        let (tn, h) = {
            let m = ark_mem.borrow();
            (m.tn, m.h)
        };
        let (c_prev, c_cur) = {
            let c = MRIC.borrow();
            (c.c[(is - 1) as usize], c.c[is as usize])
        };
        t0 = tn + c_prev * h;
        tf = tn + c_cur * h;
        ark_mem.borrow_mut().tcur = tf;
        {
            let mut s = mriStep_mem_mut(ark_mem);
            s.istage = is;
            s.cur_stage = is;
        }

        /* Determine current stage type, and call corresponding routine; the
           vector ark_mem->ycur stores the previous stage solution on input, and
           should store the result of this stage solution on output. */
        let stagetype = { mriStep_mem_mut(ark_mem).stagetypes[is as usize] };
        match stagetype {
            MRISTAGE_ERK_FAST => {
                retval = mriStep_ComputeInnerForcing(ark_mem, is, t0, tf);
                if retval != ARK_SUCCESS {
                    return retval;
                }
                let (ycur, tempv2) = {
                    let m = ark_mem.borrow();
                    (
                        m.ycur.clone().expect("ycur set"),
                        m.tempv2.clone().expect("tempv2 set"),
                    )
                };
                retval = mriStep_StageERKFast(ark_mem, t0, tf, &ycur, &tempv2, need_inner_dsm);
                if retval != ARK_SUCCESS {
                    *nflagPtr = CONV_FAIL;
                }
            }
            MRISTAGE_ERK_NOFAST => {
                retval = mriStep_StageERKNoFast(ark_mem, is);
            }
            MRISTAGE_DIRK_NOFAST => {
                retval = mriStep_StageDIRKNoFast(ark_mem, is, nflagPtr);
            }
            MRISTAGE_DIRK_FAST => {
                retval = mriStep_StageDIRKFast(ark_mem, is, nflagPtr);
            }
            MRISTAGE_STIFF_ACC => {
                retval = ARK_SUCCESS;
            }
            _ => {}
        }
        if retval != ARK_SUCCESS {
            return retval;
        }

        /* apply user-supplied stage postprocessing function (if supplied) */
        if ark_mem.borrow().PostProcessStageFn.is_some() && (stagetype != MRISTAGE_STIFF_ACC) {
            let (tcur, ycur) = {
                let m = ark_mem.borrow();
                (m.tcur, m.ycur.clone().expect("ycur set"))
            };
            retval = mriStep_CallPostProcessStageFn(ark_mem, tcur, &ycur);
            if retval != 0 {
                return ARK_POSTPROCESS_STAGE_FAIL;
            }
        }

        /* conditionally reset the inner integrator with the modified stage solution */
        if stagetype != MRISTAGE_STIFF_ACC {
            let have_postprocess = ark_mem.borrow().PostProcessStageFn.is_some();
            if (stagetype != MRISTAGE_ERK_FAST) || have_postprocess {
                let ycur = ark_mem.borrow().ycur.clone().expect("ycur set");
                retval = mriStepInnerStepper_Reset(&stepper, tf, &ycur);
                if retval != ARK_SUCCESS {
                    arkProcessError(
                        Some(ark_mem),
                        ARK_INNERSTEP_FAIL,
                        line!() as i32,
                        "mriStep_TakeStepMRIGARK",
                        file!(),
                        "Unable to reset the inner stepper",
                    );
                    return ARK_INNERSTEP_FAIL;
                }
            }
        }

        /* Compute updated slow RHS, except:
           1. if the stage is excluded from stage_map
           2. if the next stage has "STIFF_ACC" type, and temporal estimation is disabled */
        calc_fslow = SUNTRUE;
        let (stage_map_is, next_stagetype) = {
            let s = mriStep_mem_mut(ark_mem);
            (s.stage_map[is as usize], s.stagetypes[(is + 1) as usize])
        };
        if stage_map_is == -1 {
            calc_fslow = SUNFALSE;
        }
        if !do_embedding && (next_stagetype == MRISTAGE_STIFF_ACC) {
            calc_fslow = SUNFALSE;
        }
        if calc_fslow {
            let (explicit_rhs, implicit_rhs, deduce_rhs, expforcing, impforcing) = {
                let s = mriStep_mem_mut(ark_mem);
                (
                    s.explicit_rhs,
                    s.implicit_rhs,
                    s.deduce_rhs,
                    s.expforcing,
                    s.impforcing,
                )
            };

            /* call the user-supplied pre-RHS function (if supplied) */
            if ark_mem.borrow().PreRhsFn.is_some() {
                if explicit_rhs
                    || (implicit_rhs
                        && (!deduce_rhs || (stagetype != MRISTAGE_DIRK_NOFAST)))
                {
                    let (tcur, ycur) = {
                        let m = ark_mem.borrow();
                        (m.tcur, m.ycur.clone().expect("ycur set"))
                    };
                    retval = mriStep_CallPreRhsFn(ark_mem, tcur, &ycur);
                    if retval != 0 {
                        return ARK_PRERHSFN_FAIL;
                    }
                }
            }

            /* store implicit slow rhs  */
            if implicit_rhs {
                if !deduce_rhs || (stagetype != MRISTAGE_DIRK_NOFAST) {
                    let (tcur, ycur) = {
                        let m = ark_mem.borrow();
                        (m.tcur, m.ycur.clone().expect("ycur set"))
                    };
                    let Fsi_is =
                        { mriStep_mem_mut(ark_mem).Fsi[stage_map_is as usize].clone() };
                    retval = mriStep_CallFsi(ark_mem, tcur, &ycur, &Fsi_is);
                    mriStep_mem_mut(ark_mem).nfsi += 1;

                    if retval < 0 {
                        return ARK_RHSFUNC_FAIL;
                    }
                    if retval > 0 {
                        return ARK_UNREC_RHSFUNC_ERR;
                    }

                    /* Add external forcing to Fsi, if applicable */
                    if impforcing {
                        let (cvals, Xvecs) = {
                            let mut s = mriStep_mem_mut(ark_mem);
                            s.cvals[0] = ONE;
                            let v = s.Fsi[stage_map_is as usize].clone();
                            s.Xvecs[0] = Some(v);
                            nvec = 1;
                            mriStep_ApplyForcing(&mut s, tf, ONE, &mut nvec);
                            (s.cvals.clone(), mriStep_xvecs(&s, nvec))
                        };
                        let _ = N_VLinearCombination(
                            Xvecs.len() as i32,
                            &cvals,
                            &Xvecs,
                            &Fsi_is,
                        );
                    }
                } else {
                    let (gamma, zcor, sdata, Fsi_is) = {
                        let s = mriStep_mem_mut(ark_mem);
                        (
                            s.gamma,
                            s.zcor.clone().expect("zcor set"),
                            s.sdata.clone().expect("sdata set"),
                            s.Fsi[stage_map_is as usize].clone(),
                        )
                    };
                    N_VLinearSum(ONE / gamma, &zcor, -ONE / gamma, &sdata, &Fsi_is);
                }
            }

            /* store explicit slow rhs */
            if explicit_rhs {
                let (tcur, ycur) = {
                    let m = ark_mem.borrow();
                    (m.tcur, m.ycur.clone().expect("ycur set"))
                };
                let Fse_is = { mriStep_mem_mut(ark_mem).Fse[stage_map_is as usize].clone() };
                retval = mriStep_CallFse(ark_mem, tcur, &ycur, &Fse_is);
                mriStep_mem_mut(ark_mem).nfse += 1;

                if retval < 0 {
                    return ARK_RHSFUNC_FAIL;
                }
                if retval > 0 {
                    return ARK_UNREC_RHSFUNC_ERR;
                }

                /* Add external forcing to Fse, if applicable */
                if expforcing {
                    let (cvals, Xvecs) = {
                        let mut s = mriStep_mem_mut(ark_mem);
                        s.cvals[0] = ONE;
                        let v = s.Fse[stage_map_is as usize].clone();
                        s.Xvecs[0] = Some(v);
                        nvec = 1;
                        mriStep_ApplyForcing(&mut s, tf, ONE, &mut nvec);
                        (s.cvals.clone(), mriStep_xvecs(&s, nvec))
                    };
                    let _ =
                        N_VLinearCombination(Xvecs.len() as i32, &cvals, &Xvecs, &Fse_is);
                }
            }
        } /* compute slow RHS */

        is += 1;
    } /* loop over stages */

    /* perform embedded stage (if needed) */
    if do_embedding {
        is = stages;
        {
            let mut s = mriStep_mem_mut(ark_mem);
            s.istage = is;
            s.cur_stage = is;
        }

        /* Temporarily swap ark_mem->ycur and ark_mem->tempv4 pointers, copying
           data so that both hold the current ark_mem->ycur value.  This ensures
           that during this embedding "stage":
             - ark_mem->ycur will be the correct initial condition for the final stage.
             - ark_mem->tempv4 will hold the embedded solution vector. */
        {
            let (ycur, tempv4) = {
                let m = ark_mem.borrow();
                (
                    m.ycur.clone().expect("ycur set"),
                    m.tempv4.clone().expect("tempv4 set"),
                )
            };
            N_VScale(ONE, &ycur, &tempv4);
        }
        {
            let mut m = ark_mem.borrow_mut();
            let tmp = m.ycur.take();
            m.ycur = m.tempv4.take();
            m.tempv4 = tmp;
        }

        /* Reset ark_mem->tcur as the time value corresponding with the end of the step */
        /* Set relevant stage times (including desired stage time for implicit solves) */
        let (tn, h) = {
            let m = ark_mem.borrow();
            (m.tn, m.h)
        };
        let c_im2 = { MRIC.borrow().c[(is - 2) as usize] };
        t0 = tn + c_im2 * h;
        tf = tn + h;
        ark_mem.borrow_mut().tcur = tf;

        /* Determine embedding stage type, and call corresponding routine; the
           vector ark_mem->ycur stores the previous stage solution on input, and
           should store the result of this stage solution on output. */
        let stagetype = { mriStep_mem_mut(ark_mem).stagetypes[is as usize] };
        match stagetype {
            MRISTAGE_ERK_FAST => {
                retval = mriStep_ComputeInnerForcing(ark_mem, is, t0, tf);
                if retval != ARK_SUCCESS {
                    return retval;
                }
                let (ycur, tempv2) = {
                    let m = ark_mem.borrow();
                    (
                        m.ycur.clone().expect("ycur set"),
                        m.tempv2.clone().expect("tempv2 set"),
                    )
                };
                retval = mriStep_StageERKFast(ark_mem, t0, tf, &ycur, &tempv2, SUNFALSE);
                if retval != ARK_SUCCESS {
                    *nflagPtr = CONV_FAIL;
                }
            }
            MRISTAGE_ERK_NOFAST => {
                retval = mriStep_StageERKNoFast(ark_mem, is);
            }
            MRISTAGE_DIRK_NOFAST => {
                retval = mriStep_StageDIRKNoFast(ark_mem, is, nflagPtr);
            }
            MRISTAGE_DIRK_FAST => {
                retval = mriStep_StageDIRKFast(ark_mem, is, nflagPtr);
            }
            _ => {}
        }
        if retval != ARK_SUCCESS {
            return retval;
        }

        /* Swap back ark_mem->ycur with ark_mem->tempv4, and reset the inner integrator */
        {
            let mut m = ark_mem.borrow_mut();
            let tmp = m.ycur.take();
            m.ycur = m.tempv4.take();
            m.tempv4 = tmp;
        }
        let ycur = ark_mem.borrow().ycur.clone().expect("ycur set");
        retval = mriStepInnerStepper_Reset(&stepper, t0, &ycur);
        if retval != ARK_SUCCESS {
            arkProcessError(
                Some(ark_mem),
                ARK_INNERSTEP_FAIL,
                line!() as i32,
                "mriStep_TakeStepMRIGARK",
                file!(),
                "Unable to reset the inner stepper",
            );
            return ARK_INNERSTEP_FAIL;
        }
    }

    /* Compute final stage (for evolved solution), along with error estimate */
    {
        is = stages - 1;
        {
            let mut s = mriStep_mem_mut(ark_mem);
            s.istage = is;
            s.cur_stage = is;
        }

        /* Set relevant stage times (including desired stage time for implicit solves) */
        let (tn, h) = {
            let m = ark_mem.borrow();
            (m.tn, m.h)
        };
        let c_im1 = { MRIC.borrow().c[(is - 1) as usize] };
        t0 = tn + c_im1 * h;
        tf = tn + h;
        ark_mem.borrow_mut().tcur = tf;

        /* Determine final stage type, and call corresponding routine; the
           vector ark_mem->ycur stores the previous stage solution on input, and
           should store the result of this stage solution on output. */
        let stagetype = { mriStep_mem_mut(ark_mem).stagetypes[is as usize] };
        match stagetype {
            MRISTAGE_ERK_FAST => {
                retval = mriStep_ComputeInnerForcing(ark_mem, is, t0, tf);
                if retval != ARK_SUCCESS {
                    return retval;
                }
                let (ycur, tempv2) = {
                    let m = ark_mem.borrow();
                    (
                        m.ycur.clone().expect("ycur set"),
                        m.tempv2.clone().expect("tempv2 set"),
                    )
                };
                retval = mriStep_StageERKFast(ark_mem, t0, tf, &ycur, &tempv2, need_inner_dsm);
                if retval != ARK_SUCCESS {
                    *nflagPtr = CONV_FAIL;
                }
            }
            MRISTAGE_ERK_NOFAST => {
                retval = mriStep_StageERKNoFast(ark_mem, is);
            }
            MRISTAGE_DIRK_NOFAST => {
                retval = mriStep_StageDIRKNoFast(ark_mem, is, nflagPtr);
            }
            MRISTAGE_DIRK_FAST => {
                retval = mriStep_StageDIRKFast(ark_mem, is, nflagPtr);
            }
            MRISTAGE_STIFF_ACC => {
                retval = ARK_SUCCESS;
            }
            _ => {}
        }
        if retval != ARK_SUCCESS {
            return retval;
        }

        /* apply user-supplied step postprocessing function (if supplied) */
        if ark_mem.borrow().PostProcessStepFn.is_some() && (stagetype != MRISTAGE_STIFF_ACC) {
            let (tcur, ycur) = {
                let m = ark_mem.borrow();
                (m.tcur, m.ycur.clone().expect("ycur set"))
            };
            retval = mriStep_CallPostProcessStepFn(ark_mem, tcur, &ycur);
            if retval != 0 {
                return ARK_POSTPROCESS_STEP_FAIL;
            }
        }

        /* conditionally reset the inner integrator with the modified stage solution */
        if stagetype != MRISTAGE_STIFF_ACC {
            let have_postprocess = ark_mem.borrow().PostProcessStepFn.is_some();
            if (stagetype != MRISTAGE_ERK_FAST) || have_postprocess {
                let ycur = ark_mem.borrow().ycur.clone().expect("ycur set");
                retval = mriStepInnerStepper_Reset(&stepper, tf, &ycur);
                if retval != ARK_SUCCESS {
                    arkProcessError(
                        Some(ark_mem),
                        ARK_INNERSTEP_FAIL,
                        line!() as i32,
                        "mriStep_TakeStepMRIGARK",
                        file!(),
                        "Unable to reset the inner stepper",
                    );
                    return ARK_INNERSTEP_FAIL;
                }
            }
        }

        /* Compute temporal error estimate via difference between step
           solution and embedding, store in ark_mem->tempv1, and take norm. */
        if do_embedding {
            let (tempv4, ycur, tempv1, ewt) = {
                let m = ark_mem.borrow();
                (
                    m.tempv4.clone().expect("tempv4 set"),
                    m.ycur.clone().expect("ycur set"),
                    m.tempv1.clone().expect("tempv1 set"),
                    m.ewt.clone().expect("ewt set"),
                )
            };
            N_VLinearSum(ONE, &tempv4, -ONE, &ycur, &tempv1);
            *dsmPtr = N_VWrmsNorm(&tempv1, &ewt);
        }
    } /* loop over stages */

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  mriStep_TakeStepMRISR:

  This routine performs a single MRISR step.

  Both the vectors ark_mem->yn and ark_mem->ycur hold the previous
  time-step solution on input, and the vector ark_mem->ycur should
  hold the result of this step on output.

  If timestep adaptivity is enabled, this routine also computes
  the error estimate y-ytilde, where ytilde is the
  embedded solution, and the norm weights come from ark_ewt.
  This estimate is stored in ark_mem->tempv1, in case the calling
  routine wishes to examine the error locations.

  The output variable dsmPtr should contain a scalar-valued
  estimate of the temporal error from this step, ||y-ytilde||_WRMS
  if timestep adaptivity is enabled; otherwise it should be 0.

  The input/output variable nflagPtr is used to gauge convergence
  of any algebraic solvers within the step.  At the start of a new
  time step, this will initially have the value FIRST_CALL.  On
  return from this function, nflagPtr should have a value:
            0 => algebraic solve completed successfully
           >0 => solve did not converge at this step size
                 (but may with a smaller stepsize)
           <0 => solve encountered an unrecoverable failure
  Since the fast-scale evolution could be considered a different
  type of "algebraic solver", we similarly report any fast-scale
  evolution error as a recoverable nflagPtr value.

  The return value from this routine is:
            0 => step completed successfully
           >0 => step encountered recoverable failure;
                 reduce step and retry (if possible)
           <0 => step encountered unrecoverable failure
  ---------------------------------------------------------------*/
pub fn mriStep_TakeStepMRISR(
    ark_mem: &ARKodeMem,
    dsmPtr: &mut sunrealtype,
    nflagPtr: &mut i32,
) -> i32 {
    let mut stage: i32;
    let mut retval: i32; /* reusable return flag       */
    let mut embedding: sunbooleantype; /* flag indicating embedding  */
    let mut solution: sunbooleantype; /*   or solution stages       */
    let mut impl_corr: sunbooleantype; /* is slow correct. implicit? */
    let mut need_inner_dsm: sunbooleantype;
    let nested_mri: sunbooleantype;
    let mut nvec: i32;
    let tol: sunrealtype = 100.0 * SUN_UNIT_ROUNDOFF;

    /* access the MRIStep mem structure */
    retval = mriStep_step_mem_ok(ark_mem, "mriStep_TakeStepMRISR");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* initialize algebraic solver convergence flag to success;
       error estimate to zero */
    *nflagPtr = ARK_SUCCESS;
    *dsmPtr = ZERO;

    /* set N_Vector shortcuts */
    let (ytilde, ytemp) = {
        let m = ark_mem.borrow();
        (
            m.tempv4.clone().expect("tempv4 set"),
            m.tempv2.clone().expect("tempv2 set"),
        )
    };

    /* initialize the current stage index */
    {
        let mut s = mriStep_mem_mut(ark_mem);
        s.istage = 0;
        s.cur_stage = 0;
    }

    let stepper = { mriStep_mem_mut(ark_mem).stepper.clone() }.expect("stepper set");

    /* if MRI adaptivity is enabled: reset fast accumulated error,
       and send appropriate control parameter to the fast integrator */
    let hcontroller = ark_mem
        .borrow()
        .hadapt_mem
        .as_ref()
        .expect("hadapt_mem set")
        .hcontroller
        .clone()
        .expect("hcontroller set");
    let adapt_type = SUNAdaptController_GetType(&hcontroller);
    need_inner_dsm = SUNFALSE;
    if adapt_type == SUN_ADAPTCONTROLLER_MRI_H_TOL {
        need_inner_dsm = SUNTRUE;
        mriStep_mem_mut(ark_mem).inner_dsm = ZERO;
        retval = mriStepInnerStepper_ResetAccumulatedError(&stepper);
        if retval != ARK_SUCCESS {
            arkProcessError(
                Some(ark_mem),
                ARK_INNERSTEP_FAIL,
                line!() as i32,
                "mriStep_TakeStepMRISR",
                file!(),
                "Unable to reset the inner stepper error estimate",
            );
            return ARK_INNERSTEP_FAIL;
        }
        let inner_rtol_factor = { mriStep_mem_mut(ark_mem).inner_rtol_factor };
        let reltol = ark_mem.borrow().reltol;
        retval = mriStepInnerStepper_SetRTol(&stepper, inner_rtol_factor * reltol);
        if retval != ARK_SUCCESS {
            arkProcessError(
                Some(ark_mem),
                ARK_INNERSTEP_FAIL,
                line!() as i32,
                "mriStep_TakeStepMRISR",
                file!(),
                "Unable to set the inner stepper tolerance",
            );
            return ARK_INNERSTEP_FAIL;
        }
    }

    /* for adaptive computations, reset the inner integrator to the beginning of this step */
    let fixedstep = ark_mem.borrow().fixedstep;
    if !fixedstep {
        let (tcur, ycur) = {
            let m = ark_mem.borrow();
            (m.tcur, m.ycur.clone().expect("ycur set"))
        };
        retval = mriStepInnerStepper_Reset(&stepper, tcur, &ycur);
        if retval != ARK_SUCCESS {
            arkProcessError(
                Some(ark_mem),
                ARK_INNERSTEP_FAIL,
                line!() as i32,
                "mriStep_TakeStepMRISR",
                file!(),
                "Unable to reset the inner stepper",
            );
            return ARK_INNERSTEP_FAIL;
        }
    }

    /* call nonlinear solver setup if it exists */
    let NLS = { mriStep_mem_mut(ark_mem).NLS.clone() };
    if let Some(NLS) = NLS {
        if NLS.ops.borrow().setup.is_some() {
            let tempv3 = ark_mem.borrow().tempv3.clone().expect("tempv3 set");
            N_VConst(ZERO, &tempv3); /* set guess to 0 */
            let mut nls_mem: Option<Box<dyn Any>> = Some(Box::new(ark_mem.clone()));
            retval = SUNNonlinSolSetup(&NLS, &tempv3, &mut nls_mem);
            if retval < 0 {
                return ARK_NLS_SETUP_FAIL;
            }
            if retval > 0 {
                return ARK_NLS_SETUP_RECVR;
            }
        }
    }

    /* Evaluate the slow RHS functions if needed. NOTE: we decide between calling the
       full RHS function (if ark_mem->fn is non-NULL and MRIStep is not an inner
       integrator) versus just updating the stored values of Fse[0] and Fsi[0].  In
       either case, we use ARK_FULLRHS_START mode because MRISR methods do not
       evaluate the RHS functions at the end of the time step (so nothing can be
       leveraged). */
    let (expforcing, impforcing) = {
        let s = mriStep_mem_mut(ark_mem);
        (s.expforcing, s.impforcing)
    };
    nested_mri = expforcing || impforcing;
    let fn_is_null = ark_mem.borrow().fn_.is_none();
    if fn_is_null || nested_mri {
        let (tcur, ycur) = {
            let m = ark_mem.borrow();
            (m.tcur, m.ycur.clone().expect("ycur set"))
        };
        retval = mriStep_UpdateF0(ark_mem, tcur, &ycur, ARK_FULLRHS_START);
        if retval != 0 {
            return ARK_RHSFUNC_FAIL;
        }

        /* For a nested MRI configuration we might still need fn to create a predictor
           but it should be fn only for the current nesting level which is why we use
           UpdateF0 in this case rather than FullRHS */
        let fn_v = ark_mem.borrow().fn_.clone();
        let (explicit_rhs, implicit_rhs) = {
            let s = mriStep_mem_mut(ark_mem);
            (s.explicit_rhs, s.implicit_rhs)
        };
        if fn_v.is_some() && nested_mri && implicit_rhs {
            let fn_v = fn_v.expect("fn set");
            if implicit_rhs && explicit_rhs {
                let (Fsi0, Fse0) = {
                    let s = mriStep_mem_mut(ark_mem);
                    (s.Fsi[0].clone(), s.Fse[0].clone())
                };
                N_VLinearSum(ONE, &Fsi0, ONE, &Fse0, &fn_v);
            } else {
                let Fsi0 = { mriStep_mem_mut(ark_mem).Fsi[0].clone() };
                N_VScale(ONE, &Fsi0, &fn_v);
            }
        }
    }
    if !fn_is_null && !ark_mem.borrow().fn_is_current {
        let (tcur, ycur, fn_v) = {
            let m = ark_mem.borrow();
            (
                m.tcur,
                m.ycur.clone().expect("ycur set"),
                m.fn_.clone().expect("fn set"),
            )
        };
        retval = mriStep_FullRHS(ark_mem, tcur, &ycur, &fn_v, ARK_FULLRHS_START);
        if retval != 0 {
            return ARK_RHSFUNC_FAIL;
        }
    }
    ark_mem.borrow_mut().fn_is_current = SUNTRUE;

    /* combine both RHS into FSE for ImEx problems, since MRISR fast forcing function
       only depends on Omega coefficients  */
    let (explicit_rhs, implicit_rhs) = {
        let s = mriStep_mem_mut(ark_mem);
        (s.explicit_rhs, s.implicit_rhs)
    };
    if implicit_rhs && explicit_rhs {
        let (Fse0, Fsi0) = {
            let s = mriStep_mem_mut(ark_mem);
            (s.Fse[0].clone(), s.Fsi[0].clone())
        };
        N_VLinearSum(ONE, &Fse0, ONE, &Fsi0, &Fse0);
    }

    /* The first stage is the previous time-step solution, so its RHS
       is the [already-computed] slow RHS from the start of the step */

    let stages = { mriStep_mem_mut(ark_mem).stages };
    let MRIC = { mriStep_mem_mut(ark_mem).MRIC.clone() }.expect("MRIC set");
    let accum_type = ark_mem.borrow().AccumErrorType;

    /* Determine how many stages will be needed */
    let max_stages: i32 = if fixedstep && (accum_type == ARK_ACCUMERROR_NONE) {
        stages
    } else {
        stages + 1
    };

    /* Loop over stages */
    stage = 1;
    while stage < max_stages {
        /* Determine if this is an "embedding" or "solution" stage */
        solution = stage == stages - 1;
        embedding = stage == stages;

        /* Set initial condition for this stage (all but first stage) */
        if stage > 1 {
            let (yn, ycur) = {
                let m = ark_mem.borrow();
                (
                    m.yn.clone().expect("yn set"),
                    m.ycur.clone().expect("ycur set"),
                )
            };
            N_VScale(ONE, &yn, &ycur);
        }

        /* Set current stage abscissa */
        let cstage: sunrealtype = if embedding {
            ONE
        } else {
            MRIC.borrow().c[stage as usize]
        };

        /* Set current stage time and index */
        let (tn, h) = {
            let m = ark_mem.borrow();
            (m.tn, m.h)
        };
        let tcur = tn + cstage * h;
        ark_mem.borrow_mut().tcur = tcur;
        {
            let mut s = mriStep_mem_mut(ark_mem);
            s.istage = stage;
            s.cur_stage = stage;
        }

        /* Compute forcing function for inner solver */
        retval = mriStep_ComputeInnerForcing(ark_mem, stage, tn, tcur);
        if retval != ARK_SUCCESS {
            return retval;
        }

        /* Reset the inner stepper on all but the first stage due to
           "stage-restart" structure */
        if stage > 1 {
            let ycur = ark_mem.borrow().ycur.clone().expect("ycur set");
            retval = mriStepInnerStepper_Reset(&stepper, tn, &ycur);
            if retval != ARK_SUCCESS {
                arkProcessError(
                    Some(ark_mem),
                    ARK_INNERSTEP_FAIL,
                    line!() as i32,
                    "mriStep_TakeStepMRISR",
                    file!(),
                    "Unable to reset the inner stepper",
                );
                return ARK_INNERSTEP_FAIL;
            }
        }

        /* Evolve fast IVP for this stage, potentially get inner dsm on
           all non-embedding stages */
        {
            let ycur = ark_mem.borrow().ycur.clone().expect("ycur set");
            retval = mriStep_StageERKFast(
                ark_mem,
                tn,
                tcur,
                &ycur,
                &ytemp,
                need_inner_dsm && !embedding,
            );
        }
        if retval != ARK_SUCCESS {
            *nflagPtr = CONV_FAIL;
            return retval;
        }

        /* perform MRISR slow/implicit correction */
        impl_corr = SUNFALSE;
        if implicit_rhs {
            /* determine whether implicit RHS correction will require an implicit solve */
            /* `MRIStepCoupling_Alloc` gives each G matrix `stages+1` ROWS but
               only `stages` COLUMNS (arkode_mri_tables.c:181-203), so on the
               embedding iteration (`stage == stages`) upstream's
               `G[0][stage][stage]` (arkode_mristep.c:2592) reads one element
               past the end of a calloc'd row. The embedding row has no
               diagonal entry by construction -- every in-bounds use of that
               row is `G[0][stage][j]` for `j < stage` -- so the value the
               calloc'd storage yields, and the one upstream relies on, is
               ZERO (i.e. `impl_corr` false, and the `gamma` update below is
               then unreachable). Reproduce that deterministically instead of
               panicking; see ARCHITECTURE.md deviation class 5, named
               exception. */
            let g_ss = {
                let C = MRIC.borrow();
                let row = &C.G[0][stage as usize];
                if (stage as usize) < row.len() {
                    row[stage as usize]
                } else {
                    ZERO
                }
            };
            impl_corr = SUNRabs(g_ss) > tol;

            /* perform implicit solve for correction */
            if impl_corr {
                /* update stage index for prediction and nonlinear solver if this is an "embedded" stage */
                if embedding {
                    mriStep_mem_mut(ark_mem).istage = stage - 1;
                }

                /* Call predictor for current stage solution (result placed in zpred) */
                let (istage, zpred) = {
                    let s = mriStep_mem_mut(ark_mem);
                    (s.istage, s.zpred.clone().expect("zpred set"))
                };
                retval = mriStep_Predict(ark_mem, istage, &zpred);
                if retval != ARK_SUCCESS {
                    return retval;
                }

                /* If a user-supplied predictor routine is provided, call that here
                   Note that mriStep_Predict is *still* called, so this user-supplied
                   routine can just "clean up" the built-in prediction, if desired. */
                let have_stage_predict = { mriStep_mem_mut(ark_mem).stage_predict.is_some() };
                if have_stage_predict {
                    retval = mriStep_CallStagePredict(ark_mem, tcur, &zpred);
                    if retval < 0 {
                        return ARK_USER_PREDICT_FAIL;
                    }
                    if retval > 0 {
                        return TRY_AGAIN;
                    }
                }

                /* fill sdata with explicit contributions to correction */
                let ycur = ark_mem.borrow().ycur.clone().expect("ycur set");
                let (cvals, Xvecs, sdata) = {
                    let mut s = mriStep_mem_mut(ark_mem);
                    s.cvals[0] = ONE;
                    s.Xvecs[0] = Some(ycur);
                    s.cvals[1] = -ONE;
                    let v = s.zpred.clone().expect("zpred set");
                    s.Xvecs[1] = Some(v);
                    for j in 0..stage {
                        let g = MRIC.borrow().G[0][stage as usize][j as usize];
                        s.cvals[(j + 2) as usize] = h * g;
                        let v = s.Fsi[j as usize].clone();
                        s.Xvecs[(j + 2) as usize] = Some(v);
                    }
                    let sdata = s.sdata.clone().expect("sdata set");
                    (s.cvals.clone(), mriStep_xvecs(&s, stage + 2), sdata)
                };
                retval = N_VLinearCombination(stage + 2, &cvals, &Xvecs, &sdata);
                if retval != 0 {
                    return ARK_VECTOROP_ERR;
                }

                /* Update gamma for implicit solver */
                let firststage = ark_mem.borrow().firststage;
                {
                    let mut s = mriStep_mem_mut(ark_mem);
                    s.gamma = h * g_ss;
                    if firststage {
                        s.gammap = s.gamma;
                    }
                    s.gamrat = if firststage { ONE } else { s.gamma / s.gammap };
                }

                /* perform implicit solve (result is stored in ark_mem->ycur); return
                   with positive value on anything but success */
                *nflagPtr = mriStep_Nls(ark_mem, *nflagPtr);
                if *nflagPtr != ARK_SUCCESS {
                    return TRY_AGAIN;
                }
            }
            /* perform explicit update for correction */
            else {
                let ycur = ark_mem.borrow().ycur.clone().expect("ycur set");
                let (cvals, Xvecs) = {
                    let mut s = mriStep_mem_mut(ark_mem);
                    s.cvals[0] = ONE;
                    s.Xvecs[0] = Some(ycur.clone());
                    for j in 0..stage {
                        let g = MRIC.borrow().G[0][stage as usize][j as usize];
                        s.cvals[(j + 1) as usize] = h * g;
                        let v = s.Fsi[j as usize].clone();
                        s.Xvecs[(j + 1) as usize] = Some(v);
                    }
                    (s.cvals.clone(), mriStep_xvecs(&s, stage + 1))
                };
                retval = N_VLinearCombination(stage + 1, &cvals, &Xvecs, &ycur);
                if retval != 0 {
                    return ARK_VECTOROP_ERR;
                }
            }
        }

        /* apply user-supplied stage or step postprocessing function (if supplied),
           and reset the inner integrator with the modified stage solution */
        let (have_post_stage, have_post_step) = {
            let m = ark_mem.borrow();
            (m.PostProcessStageFn.is_some(), m.PostProcessStepFn.is_some())
        };
        if !solution && !embedding && have_post_stage {
            let (tcur_now, ycur) = {
                let m = ark_mem.borrow();
                (m.tcur, m.ycur.clone().expect("ycur set"))
            };
            retval = mriStep_CallPostProcessStageFn(ark_mem, tcur_now, &ycur);
            if retval != 0 {
                return ARK_POSTPROCESS_STAGE_FAIL;
            }
            let (tcur_now, ycur) = {
                let m = ark_mem.borrow();
                (m.tcur, m.ycur.clone().expect("ycur set"))
            };
            retval = mriStepInnerStepper_Reset(&stepper, tcur_now, &ycur);
            if retval != ARK_SUCCESS {
                arkProcessError(
                    Some(ark_mem),
                    ARK_INNERSTEP_FAIL,
                    line!() as i32,
                    "mriStep_TakeStepMRISR",
                    file!(),
                    "Unable to reset the inner stepper",
                );
                return ARK_INNERSTEP_FAIL;
            }
        } else if solution && have_post_step {
            let (tcur_now, ycur) = {
                let m = ark_mem.borrow();
                (m.tcur, m.ycur.clone().expect("ycur set"))
            };
            retval = mriStep_CallPostProcessStepFn(ark_mem, tcur_now, &ycur);
            if retval != 0 {
                return ARK_POSTPROCESS_STEP_FAIL;
            }
            let (tcur_now, ycur) = {
                let m = ark_mem.borrow();
                (m.tcur, m.ycur.clone().expect("ycur set"))
            };
            retval = mriStepInnerStepper_Reset(&stepper, tcur_now, &ycur);
            if retval != ARK_SUCCESS {
                arkProcessError(
                    Some(ark_mem),
                    ARK_INNERSTEP_FAIL,
                    line!() as i32,
                    "mriStep_TakeStepMRISR",
                    file!(),
                    "Unable to reset the inner stepper",
                );
                return ARK_INNERSTEP_FAIL;
            }
        }

        /* Compute updated slow RHS (except for final solution or embedding) */
        if !solution && !embedding {
            let deduce_rhs = { mriStep_mem_mut(ark_mem).deduce_rhs };

            /* call the user-supplied pre-RHS function (if supplied) */
            if ark_mem.borrow().PreRhsFn.is_some() {
                if explicit_rhs || (implicit_rhs && (!deduce_rhs || !impl_corr)) {
                    let (tcur_now, ycur) = {
                        let m = ark_mem.borrow();
                        (m.tcur, m.ycur.clone().expect("ycur set"))
                    };
                    retval = mriStep_CallPreRhsFn(ark_mem, tcur_now, &ycur);
                    if retval != 0 {
                        return ARK_PRERHSFN_FAIL;
                    }
                }
            }

            /* store implicit slow rhs */
            if implicit_rhs {
                if !deduce_rhs || !impl_corr {
                    let (tcur_now, ycur) = {
                        let m = ark_mem.borrow();
                        (m.tcur, m.ycur.clone().expect("ycur set"))
                    };
                    let Fsi_stage = { mriStep_mem_mut(ark_mem).Fsi[stage as usize].clone() };
                    retval = mriStep_CallFsi(ark_mem, tcur_now, &ycur, &Fsi_stage);
                    mriStep_mem_mut(ark_mem).nfsi += 1;

                    if retval < 0 {
                        return ARK_RHSFUNC_FAIL;
                    }
                    if retval > 0 {
                        return ARK_UNREC_RHSFUNC_ERR;
                    }

                    /* Add external forcing to Fsi[stage], if applicable */
                    if impforcing {
                        let (cvals, Xvecs) = {
                            let mut s = mriStep_mem_mut(ark_mem);
                            s.cvals[0] = ONE;
                            let v = s.Fsi[stage as usize].clone();
                            s.Xvecs[0] = Some(v);
                            nvec = 1;
                            mriStep_ApplyForcing(&mut s, tcur_now, ONE, &mut nvec);
                            (s.cvals.clone(), mriStep_xvecs(&s, nvec))
                        };
                        let _ = N_VLinearCombination(
                            Xvecs.len() as i32,
                            &cvals,
                            &Xvecs,
                            &Fsi_stage,
                        );
                    }
                } else {
                    let (gamma, zcor, sdata, Fsi_stage) = {
                        let s = mriStep_mem_mut(ark_mem);
                        (
                            s.gamma,
                            s.zcor.clone().expect("zcor set"),
                            s.sdata.clone().expect("sdata set"),
                            s.Fsi[stage as usize].clone(),
                        )
                    };
                    N_VLinearSum(ONE / gamma, &zcor, -ONE / gamma, &sdata, &Fsi_stage);
                }
            }

            /* store explicit slow rhs */
            if explicit_rhs {
                let (tcur_now, ycur) = {
                    let m = ark_mem.borrow();
                    (m.tcur, m.ycur.clone().expect("ycur set"))
                };
                let Fse_stage = { mriStep_mem_mut(ark_mem).Fse[stage as usize].clone() };
                retval = mriStep_CallFse(ark_mem, tcur_now, &ycur, &Fse_stage);
                mriStep_mem_mut(ark_mem).nfse += 1;

                if retval < 0 {
                    return ARK_RHSFUNC_FAIL;
                }
                if retval > 0 {
                    return ARK_UNREC_RHSFUNC_ERR;
                }

                /* Add external forcing to Fse[stage], if applicable */
                if expforcing {
                    let (cvals, Xvecs) = {
                        let mut s = mriStep_mem_mut(ark_mem);
                        s.cvals[0] = ONE;
                        let v = s.Fse[stage as usize].clone();
                        s.Xvecs[0] = Some(v);
                        nvec = 1;
                        mriStep_ApplyForcing(&mut s, tcur_now, ONE, &mut nvec);
                        (s.cvals.clone(), mriStep_xvecs(&s, nvec))
                    };
                    let _ =
                        N_VLinearCombination(Xvecs.len() as i32, &cvals, &Xvecs, &Fse_stage);
                }
            }

            /* combine both RHS into Fse for ImEx problems since
               fast forcing function only depends on Omega coefficients */
            if implicit_rhs && explicit_rhs {
                let (Fse_stage, Fsi_stage) = {
                    let s = mriStep_mem_mut(ark_mem);
                    (s.Fse[stage as usize].clone(), s.Fsi[stage as usize].clone())
                };
                N_VLinearSum(ONE, &Fse_stage, ONE, &Fsi_stage, &Fse_stage);
            }
        }

        /* If this is the solution stage, archive for error estimation */
        if solution {
            let ycur = ark_mem.borrow().ycur.clone().expect("ycur set");
            N_VScale(ONE, &ycur, &ytilde);
        }

        stage += 1;
    } /* loop over stages */

    /* if temporal error estimation is enabled: compute estimate via difference between
       step solution and embedding, store in ark_mem->tempv1, store norm in dsmPtr, and
       copy solution back to ycur */
    if !fixedstep || (accum_type != ARK_ACCUMERROR_NONE) {
        let (ycur, tempv1, ewt) = {
            let m = ark_mem.borrow();
            (
                m.ycur.clone().expect("ycur set"),
                m.tempv1.clone().expect("tempv1 set"),
                m.ewt.clone().expect("ewt set"),
            )
        };
        N_VLinearSum(ONE, &ytilde, -ONE, &ycur, &tempv1);
        *dsmPtr = N_VWrmsNorm(&tempv1, &ewt);
        N_VScale(ONE, &ytilde, &ycur);
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  mriStep_TakeStepMERK:

  This routine performs a single MERK step.

  Both the vectors ark_mem->yn and ark_mem->ycur hold the previous
  time-step solution on input, and the vector ark_mem->ycur should
  hold the result of this step on output.

  If timestep adaptivity is enabled, this routine also computes
  the error estimate y-ytilde, where ytilde is the
  embedded solution, and the norm weights come from ark_ewt.
  This estimate is stored in ark_mem->tempv1, in case the calling
  routine wishes to examine the error locations.

  The output variable dsmPtr should contain a scalar-valued
  estimate of the temporal error from this step, ||y-ytilde||_WRMS
  if timestep adaptivity is enabled; otherwise it should be 0.

  The input/output variable nflagPtr is used to gauge convergence
  of any algebraic solvers within the step.  At the start of a new
  time step, this will initially have the value FIRST_CALL.  On
  return from this function, nflagPtr should have a value:
            0 => algebraic solve completed successfully
           >0 => solve did not converge at this step size
                 (but may with a smaller stepsize)
           <0 => solve encountered an unrecoverable failure
  Since the fast-scale evolution could be considered a different
  type of "algebraic solver", we similarly report any fast-scale
  evolution error as a recoverable nflagPtr value.

  The return value from this routine is:
            0 => step completed successfully
           >0 => step encountered recoverable failure;
                 reduce step and retry (if possible)
           <0 => step encountered unrecoverable failure
  ---------------------------------------------------------------*/
pub fn mriStep_TakeStepMERK(
    ark_mem: &ARKodeMem,
    dsmPtr: &mut sunrealtype,
    nflagPtr: &mut i32,
) -> i32 {
    /* access the MRIStep mem structure */
    let retval = mriStep_AccessStepMem(ark_mem, "mriStep_TakeStepMERK");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* initialize algebraic solver convergence flag to success;
       error estimate to zero */
    *nflagPtr = ARK_SUCCESS;
    *dsmPtr = ZERO;

    /* set N_Vector shortcuts */
    let (ytilde, ytemp) = {
        let m = ark_mem.borrow();
        (
            m.tempv4.clone().expect("tempv4"),
            m.tempv2.clone().expect("tempv2"),
        )
    };

    /* initial time for step */
    /* dead store in the C source too (arkode_mristep.c:2919): `t0` is
     * first read only after the re-assignment inside the stage loop.
     * Kept for fidelity. */
    #[allow(unused_assignments)]
    let mut t0 = ark_mem.borrow().tn;

    /* initialize the current stage index */
    {
        let mut step_mem = mriStep_mem_mut(ark_mem);
        step_mem.cur_stage = 0;
        step_mem.istage = step_mem.cur_stage;
    }

    /* handles that do not change during the step */
    let stepper = mriStep_mem_mut(ark_mem).stepper.clone().expect("stepper");
    let mric = mriStep_mem_mut(ark_mem).MRIC.clone().expect("MRIC");
    let stages = mriStep_mem_mut(ark_mem).stages;

    /* if MRI adaptivity is enabled: reset fast accumulated error,
       and send appropriate control parameter to the fast integrator */
    let hcontroller = ark_mem
        .borrow()
        .hadapt_mem
        .as_ref()
        .expect("hadapt_mem")
        .hcontroller
        .clone();
    let adapt_type = match hcontroller.as_ref() {
        Some(C) => SUNAdaptController_GetType(C),
        None => SUN_ADAPTCONTROLLER_NONE,
    };
    let mut need_inner_dsm = SUNFALSE;
    if adapt_type == SUN_ADAPTCONTROLLER_MRI_H_TOL {
        need_inner_dsm = SUNTRUE;
        mriStep_mem_mut(ark_mem).inner_dsm = ZERO;
        let retval = mriStepInnerStepper_ResetAccumulatedError(&stepper);
        if retval != ARK_SUCCESS {
            arkProcessError(
                Some(ark_mem),
                ARK_INNERSTEP_FAIL,
                line!() as i32,
                "mriStep_TakeStepMERK",
                file!(),
                "Unable to reset the inner stepper error estimate",
            );
            return ARK_INNERSTEP_FAIL;
        }
        let inner_rtol_factor = mriStep_mem_mut(ark_mem).inner_rtol_factor;
        let reltol = ark_mem.borrow().reltol;
        let retval = mriStepInnerStepper_SetRTol(&stepper, inner_rtol_factor * reltol);
        if retval != ARK_SUCCESS {
            arkProcessError(
                Some(ark_mem),
                ARK_INNERSTEP_FAIL,
                line!() as i32,
                "mriStep_TakeStepMERK",
                file!(),
                "Unable to set the inner stepper tolerance",
            );
            return ARK_INNERSTEP_FAIL;
        }
    }

    /* for adaptive computations, reset the inner integrator to the beginning of this step */
    let fixedstep = ark_mem.borrow().fixedstep;
    if !fixedstep {
        let (tcur, ycur) = {
            let m = ark_mem.borrow();
            (m.tcur, m.ycur.clone().expect("ycur"))
        };
        let retval = mriStepInnerStepper_Reset(&stepper, tcur, &ycur);
        if retval != ARK_SUCCESS {
            arkProcessError(
                Some(ark_mem),
                ARK_INNERSTEP_FAIL,
                line!() as i32,
                "mriStep_TakeStepMERK",
                file!(),
                "Unable to reset the inner stepper",
            );
            return ARK_INNERSTEP_FAIL;
        }
    }

    /* Evaluate the slow RHS function if needed. NOTE: we decide between calling the
       full RHS function (if ark_mem->fn is non-NULL and MRIStep is not an inner
       integrator) versus just updating the stored value of Fse[0]. In either case,
       we use ARK_FULLRHS_START mode because MERK methods do not evaluate Fse at the
       end of the time step (so nothing can be leveraged). */
    let nested_mri = {
        let step_mem = mriStep_mem_mut(ark_mem);
        step_mem.expforcing || step_mem.impforcing
    };
    let fn_is_null = ark_mem.borrow().fn_.is_none();
    let fn_is_current = ark_mem.borrow().fn_is_current;
    if fn_is_null || nested_mri {
        let (tcur, ycur) = {
            let m = ark_mem.borrow();
            (m.tcur, m.ycur.clone().expect("ycur"))
        };
        let retval = mriStep_UpdateF0(ark_mem, tcur, &ycur, ARK_FULLRHS_START);
        if retval != 0 {
            return ARK_RHSFUNC_FAIL;
        }
    } else if !fn_is_null && !fn_is_current {
        let (tcur, ycur, fn_) = {
            let m = ark_mem.borrow();
            (
                m.tcur,
                m.ycur.clone().expect("ycur"),
                m.fn_.clone().expect("fn"),
            )
        };
        let retval = mriStep_FullRHS(ark_mem, tcur, &ycur, &fn_, ARK_FULLRHS_START);
        if retval != 0 {
            return ARK_RHSFUNC_FAIL;
        }
    }
    ark_mem.borrow_mut().fn_is_current = SUNTRUE;

    /* The first stage is the previous time-step solution, so its RHS
       is the [already-computed] slow RHS from the start of the step */

    /* Loop over stage groups */
    let ngroup = mric.borrow().ngroup;
    for ig in 0..ngroup {
        /* Find the lowest stage number in this group. The stages in a group are not
           necessarily in increasing order e.g., in MERK43 stage 3 is before stage 2
           in time. Since all the stages in a group share the same forcing vectors
           and the tables must be lower triangular, only stages up to one less than
           the lowest stage index in the group can be used in the forcing. Using the
           lowest stage number in the group prevents unintentionally including stage
           RHS values that have not been computed yet. */
        let mut lowest_stage: i32;
        {
            let C = mric.borrow();
            lowest_stage = C.group[ig as usize][0];
            for il in 1..C.stages {
                if C.group[ig as usize][il as usize] < 0 {
                    break;
                }
                lowest_stage = SUNMIN(lowest_stage, C.group[ig as usize][il as usize]);
            }
        }

        /* Set up fast RHS for this stage group */
        let (tn, h) = {
            let m = ark_mem.borrow();
            (m.tn, m.h)
        };
        let retval = mriStep_ComputeInnerForcing(ark_mem, lowest_stage, tn, tn + h);
        if retval != ARK_SUCCESS {
            return retval;
        }

        /* Set initial condition for this stage group (all but first group) */
        if ig > 0 {
            let (yn, ycur) = {
                let m = ark_mem.borrow();
                (m.yn.clone().expect("yn"), m.ycur.clone().expect("ycur"))
            };
            N_VScale(ONE, &yn, &ycur);
        }
        t0 = ark_mem.borrow().tn;

        /* Evolve fast IVP over each subinterval in stage group */
        for is in 0..stages {
            /* Get stage index from group; skip to the next group if
               we've reached the end of this one */
            let stage = mric.borrow().group[ig as usize][is as usize];
            {
                let mut step_mem = mriStep_mem_mut(ark_mem);
                step_mem.cur_stage = stage;
                step_mem.istage = step_mem.cur_stage;
            }
            if stage < 0 {
                break;
            }
            let mut nextstage = -1;
            if stage < stages {
                nextstage = mric.borrow().group[ig as usize][(is + 1) as usize];
            }

            /* Determine if this is an "embedding" or "solution" stage */
            let mut embedding = SUNFALSE;
            let mut solution = SUNFALSE;
            let ngroup = mric.borrow().ngroup;
            if ig == ngroup - 2 {
                if (stage >= 0) && (nextstage < 0) {
                    embedding = SUNTRUE;
                }
            }
            if ig == ngroup - 1 {
                if (stage >= 0) && (nextstage < 0) {
                    solution = SUNTRUE;
                }
            }

            /* Skip the embedding if we're using fixed time-stepping and
               temporal error estimation is disabled */
            let (fixedstep, accum_type) = {
                let m = ark_mem.borrow();
                (m.fixedstep, m.AccumErrorType)
            };
            if fixedstep && embedding && (accum_type == ARK_ACCUMERROR_NONE) {
                break;
            }

            /* Set current stage abscissa */
            let cstage = if stage >= stages {
                ONE
            } else {
                mric.borrow().c[stage as usize]
            };

            /* Set desired output time for subinterval */
            let (tn, h) = {
                let m = ark_mem.borrow();
                (m.tn, m.h)
            };
            let tf = tn + cstage * h;

            /* Reset the inner stepper on the first stage within all but the
               first stage group due to "stage-restart" structure */
            if (stage > 1) && (is == 0) {
                let ycur = ark_mem.borrow().ycur.clone().expect("ycur");
                let retval = mriStepInnerStepper_Reset(&stepper, t0, &ycur);
                if retval != ARK_SUCCESS {
                    arkProcessError(
                        Some(ark_mem),
                        ARK_INNERSTEP_FAIL,
                        line!() as i32,
                        "mriStep_TakeStepMERK",
                        file!(),
                        "Unable to reset the inner stepper",
                    );
                    return ARK_INNERSTEP_FAIL;
                }
            }

            /* Evolve fast IVP for this stage, potentially get inner dsm on all
               non-embedding stages */
            let ycur = ark_mem.borrow().ycur.clone().expect("ycur");
            let retval = mriStep_StageERKFast(
                ark_mem,
                t0,
                tf,
                &ycur,
                &ytemp,
                need_inner_dsm && !embedding,
            );
            if retval != ARK_SUCCESS {
                *nflagPtr = CONV_FAIL;
                return retval;
            }

            /* Update "initial time" for next stage in group */
            t0 = tf;

            /* set current stage time for postprocessing and RHS calls */
            ark_mem.borrow_mut().tcur = tf;

            /* apply user-supplied stage postprocessing function (if supplied),
               and reset the inner integrator with the modified stage solution */
            let (PostProcessStageFn, PostProcessStepFn) = {
                let m = ark_mem.borrow();
                (m.PostProcessStageFn, m.PostProcessStepFn)
            };
            if !solution && !embedding && PostProcessStageFn.is_some() {
                let (tcur, ycur) = {
                    let m = ark_mem.borrow();
                    (m.tcur, m.ycur.clone().expect("ycur"))
                };
                let mut user_data = ark_mem.borrow_mut().user_data.take();
                let retval =
                    PostProcessStageFn.expect("PostProcessStageFn")(tcur, &ycur, &mut user_data);
                ark_mem.borrow_mut().user_data = user_data;
                if retval != 0 {
                    return ARK_POSTPROCESS_STAGE_FAIL;
                }

                let retval = mriStepInnerStepper_Reset(&stepper, tcur, &ycur);
                if retval != ARK_SUCCESS {
                    arkProcessError(
                        Some(ark_mem),
                        ARK_INNERSTEP_FAIL,
                        line!() as i32,
                        "mriStep_TakeStepMERK",
                        file!(),
                        "Unable to reset the inner stepper",
                    );
                    return ARK_INNERSTEP_FAIL;
                }
            } else if solution && PostProcessStepFn.is_some() {
                let (tcur, ycur) = {
                    let m = ark_mem.borrow();
                    (m.tcur, m.ycur.clone().expect("ycur"))
                };
                let mut user_data = ark_mem.borrow_mut().user_data.take();
                let retval =
                    PostProcessStepFn.expect("PostProcessStepFn")(tcur, &ycur, &mut user_data);
                ark_mem.borrow_mut().user_data = user_data;
                if retval != 0 {
                    return ARK_POSTPROCESS_STEP_FAIL;
                }

                let retval = mriStepInnerStepper_Reset(&stepper, tcur, &ycur);
                if retval != ARK_SUCCESS {
                    arkProcessError(
                        Some(ark_mem),
                        ARK_INNERSTEP_FAIL,
                        line!() as i32,
                        "mriStep_TakeStepMERK",
                        file!(),
                        "Unable to reset the inner stepper",
                    );
                    return ARK_INNERSTEP_FAIL;
                }
            }

            /* Compute updated slow RHS (except for final solution or embedding) */
            if !solution && !embedding {
                /* call the user-supplied pre-RHS function (if supplied) */
                let PreRhsFn = ark_mem.borrow().PreRhsFn;
                if let Some(PreRhsFn) = PreRhsFn {
                    let (tcur, ycur) = {
                        let m = ark_mem.borrow();
                        (m.tcur, m.ycur.clone().expect("ycur"))
                    };
                    let mut user_data = ark_mem.borrow_mut().user_data.take();
                    let retval = PreRhsFn(tcur, &ycur, &mut user_data);
                    ark_mem.borrow_mut().user_data = user_data;
                    if retval != 0 {
                        return ARK_PRERHSFN_FAIL;
                    }
                }

                /* store explicit slow rhs */
                let (tcur, ycur) = {
                    let m = ark_mem.borrow();
                    (m.tcur, m.ycur.clone().expect("ycur"))
                };
                let (fse, Fse_stage) = {
                    let step_mem = mriStep_mem_mut(ark_mem);
                    (
                        step_mem.fse.expect("fse"),
                        step_mem.Fse[stage as usize].clone(),
                    )
                };
                let mut user_data = ark_mem.borrow_mut().user_data.take();
                let retval = fse(tcur, &ycur, &Fse_stage, &mut user_data);
                ark_mem.borrow_mut().user_data = user_data;
                mriStep_mem_mut(ark_mem).nfse += 1;

                if retval < 0 {
                    return ARK_RHSFUNC_FAIL;
                }
                if retval > 0 {
                    return ARK_UNREC_RHSFUNC_ERR;
                }

                /* Add external forcing to Fse[stage], if applicable */
                let expforcing = mriStep_mem_mut(ark_mem).expforcing;
                if expforcing {
                    let mut nvec = 1;
                    let (cvals, Xvecs) = {
                        let mut step_mem = mriStep_mem_mut(ark_mem);
                        step_mem.cvals[0] = ONE;
                        step_mem.Xvecs[0] = Some(Fse_stage.clone());
                        mriStep_ApplyForcing(&mut step_mem, tcur, ONE, &mut nvec);
                        (step_mem.cvals.clone(), mriStep_xvecs(&step_mem, nvec))
                    };
                    N_VLinearCombination(nvec, &cvals, &Xvecs, &Fse_stage);
                }
            }

            /* If this is the embedding stage, archive solution for error estimation */
            if embedding {
                let ycur = ark_mem.borrow().ycur.clone().expect("ycur");
                N_VScale(ONE, &ycur, &ytilde);
            }
        } /* loop over stages */
    } /* loop over stage groups */

    /* if temporal error estimation is enabled: compute estimate via difference between
       step solution and embedding, store in ark_mem->tempv1, and store norm in dsmPtr */
    let (fixedstep, accum_type) = {
        let m = ark_mem.borrow();
        (m.fixedstep, m.AccumErrorType)
    };
    if !fixedstep || (accum_type != ARK_ACCUMERROR_NONE) {
        let (ycur, tempv1, ewt) = {
            let m = ark_mem.borrow();
            (
                m.ycur.clone().expect("ycur"),
                m.tempv1.clone().expect("tempv1"),
                m.ewt.clone().expect("ewt"),
            )
        };
        N_VLinearSum(ONE, &ytilde, -ONE, &ycur, &tempv1);
        *dsmPtr = N_VWrmsNorm(&tempv1, &ewt);
    }

    ARK_SUCCESS
}

/*===============================================================
  Internal utility routines
  ===============================================================*/

/*---------------------------------------------------------------
  mriStep_AccessARKODEStepMem:

  Shortcut routine to unpack ark_mem and step_mem structures from
  void* pointer.  If either is missing it returns ARK_MEM_NULL.

  Port note (frozen seam spec, section 3): handles are never NULL in
  Rust, so the C out-params `ARKodeMem* ark_mem` /
  `ARKodeMRIStepMem* step_mem` disappear and this collapses to the
  step-memory PRESENCE CHECK.  Use `mriStep_mem_mut(ark_mem)` at each
  use site to reach the record itself.
  ---------------------------------------------------------------*/
pub fn mriStep_AccessARKODEStepMem(arkode_mem: &ARKodeMem, fname: &str) -> i32 {
    /* access ARKodeMem structure: `&ARKodeMem` is never NULL */

    /* access ARKodeMRIStepMem structure */
    if arkode_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(arkode_mem),
            ARK_MEM_NULL,
            line!() as i32,
            fname,
            file!(),
            MSG_MRISTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }
    ARK_SUCCESS
}

/*---------------------------------------------------------------
  mriStep_AccessStepMem:

  Shortcut routine to unpack step_mem structure from ark_mem.
  If missing it returns ARK_MEM_NULL.

  Port note: presence check only (see mriStep_AccessARKODEStepMem).
  ---------------------------------------------------------------*/
pub fn mriStep_AccessStepMem(ark_mem: &ARKodeMem, fname: &str) -> i32 {
    /* access ARKodeMRIStepMem structure */
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            fname,
            file!(),
            MSG_MRISTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }
    ARK_SUCCESS
}

/*---------------------------------------------------------------
  mriStep_SetCoupling

  This routine determines the MRI method to use, based on the
  desired accuracy and fixed/adaptive time stepping choice.
  ---------------------------------------------------------------*/
pub fn mriStep_SetCoupling(ark_mem: &ARKodeMem) -> i32 {
    let mut Cliw: sunindextype = 0;
    let mut Clrw: sunindextype = 0;
    let mut table_id: i32 = ARKODE_MRI_NONE;

    /* access ARKodeMRIStepMem structure */
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "mriStep_SetCoupling",
            file!(),
            MSG_MRISTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    /* if coupling has already been specified, just return */
    let have_coupling = mriStep_mem_mut(ark_mem).MRIC.is_some();
    if have_coupling {
        return ARK_SUCCESS;
    }

    let (implicit_rhs, explicit_rhs, q) = {
        let step_mem = mriStep_mem_mut(ark_mem);
        (step_mem.implicit_rhs, step_mem.explicit_rhs, step_mem.q)
    };
    let fixedstep = ark_mem.borrow().fixedstep;

    /* select method based on order and type */
    if fixedstep
    /**** fixed-step methods ****/
    {
        if implicit_rhs && explicit_rhs
        /**** ImEx methods ****/
        {
            match q {
                1 => table_id = MRISTEP_DEFAULT_IMEX_SD_1,
                2 => table_id = MRISTEP_DEFAULT_IMEX_SD_2,
                3 => table_id = MRISTEP_DEFAULT_IMEX_SD_3,
                4 => table_id = MRISTEP_DEFAULT_IMEX_SD_4,
                _ => {}
            }
        } else if implicit_rhs
        /**** implicit methods ****/
        {
            match q {
                1 => table_id = MRISTEP_DEFAULT_IMPL_SD_1,
                2 => table_id = MRISTEP_DEFAULT_IMPL_SD_2,
                3 => table_id = MRISTEP_DEFAULT_IMPL_SD_3,
                4 => table_id = MRISTEP_DEFAULT_IMPL_SD_4,
                _ => {}
            }
        } else
        /**** explicit methods ****/
        {
            match q {
                1 => table_id = MRISTEP_DEFAULT_EXPL_1,
                2 => table_id = MRISTEP_DEFAULT_EXPL_2,
                3 => table_id = MRISTEP_DEFAULT_EXPL_3,
                4 => table_id = MRISTEP_DEFAULT_EXPL_4,
                5 => table_id = MRISTEP_DEFAULT_EXPL_5_AD,
                _ => {}
            }
        }
    } else
    /**** adaptive methods ****/
    {
        if implicit_rhs && explicit_rhs
        /**** ImEx methods ****/
        {
            match q {
                2 => table_id = MRISTEP_DEFAULT_IMEX_SD_2_AD,
                3 => table_id = MRISTEP_DEFAULT_IMEX_SD_3_AD,
                4 => table_id = MRISTEP_DEFAULT_IMEX_SD_4_AD,
                _ => {}
            }
        } else if implicit_rhs
        /**** implicit methods ****/
        {
            match q {
                2 => table_id = MRISTEP_DEFAULT_IMPL_SD_2,
                3 => table_id = MRISTEP_DEFAULT_IMPL_SD_3,
                4 => table_id = MRISTEP_DEFAULT_IMPL_SD_4,
                _ => {}
            }
        } else
        /**** explicit methods ****/
        {
            match q {
                2 => table_id = MRISTEP_DEFAULT_EXPL_2_AD,
                3 => table_id = MRISTEP_DEFAULT_EXPL_3_AD,
                4 => table_id = MRISTEP_DEFAULT_EXPL_4_AD,
                5 => table_id = MRISTEP_DEFAULT_EXPL_5_AD,
                _ => {}
            }
        }
    }
    if table_id == ARKODE_MRI_NONE {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "mriStep_SetCoupling",
            file!(),
            "No MRI method is available for the requested configuration.",
        );
        return ARK_ILL_INPUT;
    }

    mriStep_mem_mut(ark_mem).MRIC = MRIStepCoupling_LoadTable(table_id);
    let MRIC = mriStep_mem_mut(ark_mem).MRIC.clone();
    if MRIC.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_INVALID_TABLE,
            line!() as i32,
            "mriStep_SetCoupling",
            file!(),
            "An error occurred in constructing coupling table.",
        );
        return ARK_INVALID_TABLE;
    }
    let MRIC = MRIC.expect("MRIC");

    /* note coupling structure space requirements */
    MRIStepCoupling_Space(Some(&MRIC), &mut Cliw, &mut Clrw);
    {
        let mut m = ark_mem.borrow_mut();
        m.liw += Cliw;
        m.lrw += Clrw;
    }

    /* set [redundant] stored values for stage numbers and
       method/embedding orders */
    {
        let C = MRIC.borrow();
        let mut step_mem = mriStep_mem_mut(ark_mem);
        step_mem.stages = C.stages;
        step_mem.q = C.q;
        step_mem.p = C.p;
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  mriStep_CheckCoupling

  This routine runs through the MRI coupling structure to ensure
  that it meets all necessary requirements, including:
    sorted abscissae, with c[0] = 0 and c[end] = 1
    lower-triangular (i.e., ERK or DIRK)
    all DIRK stages are solve-decoupled [temporarily]
    method order q > 0 (all)
    stages > 0 (all)

  Returns ARK_SUCCESS if it passes, ARK_INVALID_TABLE otherwise.
  ---------------------------------------------------------------*/
pub fn mriStep_CheckCoupling(ark_mem: &ARKodeMem) -> i32 {
    let mut okay: sunbooleantype;
    let mut Gabs: sunrealtype;
    let mut Wabs: sunrealtype;
    let tol: sunrealtype = 100.0 * SUN_UNIT_ROUNDOFF;

    /* access ARKodeMRIStepMem structure */
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "mriStep_CheckCoupling",
            file!(),
            MSG_MRISTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    let (mric, implicit_rhs, explicit_rhs) = {
        let step_mem = mriStep_mem_mut(ark_mem);
        (
            step_mem.MRIC.clone().expect("MRIC"),
            step_mem.implicit_rhs,
            step_mem.explicit_rhs,
        )
    };
    let fixedstep = ark_mem.borrow().fixedstep;

    let C = mric.borrow();

    /* check that stages > 0 */
    if C.stages < 1 {
        arkProcessError(
            Some(ark_mem),
            ARK_INVALID_TABLE,
            line!() as i32,
            "mriStep_CheckCoupling",
            file!(),
            "stages < 1!",
        );
        return ARK_INVALID_TABLE;
    }

    /* check that method order q > 0 */
    if C.q < 1 {
        arkProcessError(
            Some(ark_mem),
            ARK_INVALID_TABLE,
            line!() as i32,
            "mriStep_CheckCoupling",
            file!(),
            "method order < 1",
        );
        return ARK_INVALID_TABLE;
    }

    /* check that embedding order p > 0 (if adaptive) */
    if (C.p < 1) && (!fixedstep) {
        arkProcessError(
            Some(ark_mem),
            ARK_INVALID_TABLE,
            line!() as i32,
            "mriStep_CheckCoupling",
            file!(),
            "embedding order < 1, but ARKodeSetFixedStep was not called",
        );
        return ARK_INVALID_TABLE;
    }

    /* Check that coupling table has compatible type */
    if implicit_rhs && explicit_rhs && (C.type_ != MRISTEP_IMEX) && (C.type_ != MRISTEP_SR) {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "mriStep_CheckCoupling",
            file!(),
            "Invalid coupling table for an IMEX problem!",
        );
        return ARK_ILL_INPUT;
    }
    if explicit_rhs
        && (C.type_ != MRISTEP_EXPLICIT)
        && (C.type_ != MRISTEP_IMEX)
        && (C.type_ != MRISTEP_MERK)
        && (C.type_ != MRISTEP_SR)
    {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "mriStep_CheckCoupling",
            file!(),
            "Invalid coupling table for an explicit problem!",
        );
        return ARK_ILL_INPUT;
    }
    if implicit_rhs
        && (C.type_ != MRISTEP_IMPLICIT)
        && (C.type_ != MRISTEP_IMEX)
        && (C.type_ != MRISTEP_SR)
    {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "mriStep_CheckCoupling",
            file!(),
            "Invalid coupling table for an implicit problem!",
        );
        return ARK_ILL_INPUT;
    }

    /* Check that the matrices are defined appropriately */
    if (C.type_ == MRISTEP_IMEX) || (C.type_ == MRISTEP_SR) {
        /* ImEx */
        if C.W.is_empty() || C.G.is_empty() {
            arkProcessError(
                Some(ark_mem),
                ARK_ILL_INPUT,
                line!() as i32,
                "mriStep_CheckCoupling",
                file!(),
                "Invalid coupling table for an IMEX problem!",
            );
            return ARK_ILL_INPUT;
        }
    } else if (C.type_ == MRISTEP_EXPLICIT) || (C.type_ == MRISTEP_MERK) {
        /* Explicit */
        if C.W.is_empty() || !C.G.is_empty() {
            arkProcessError(
                Some(ark_mem),
                ARK_ILL_INPUT,
                line!() as i32,
                "mriStep_CheckCoupling",
                file!(),
                "Invalid coupling table for an explicit problem!",
            );
            return ARK_ILL_INPUT;
        }
    } else if C.type_ == MRISTEP_IMPLICIT {
        /* Implicit */
        if !C.W.is_empty() || C.G.is_empty() {
            arkProcessError(
                Some(ark_mem),
                ARK_ILL_INPUT,
                line!() as i32,
                "mriStep_CheckCoupling",
                file!(),
                "Invalid coupling table for an implicit problem!",
            );
            return ARK_ILL_INPUT;
        }
    }

    /* Check that W tables are strictly lower triangular */
    if !C.W.is_empty() {
        Wabs = 0.0;
        for k in 0..C.nmat {
            for i in 0..C.stages {
                for j in i..C.stages {
                    Wabs += SUNRabs(C.W[k as usize][i as usize][j as usize]);
                }
            }
        }
        if Wabs > tol {
            arkProcessError(
                Some(ark_mem),
                ARK_INVALID_TABLE,
                line!() as i32,
                "mriStep_CheckCoupling",
                file!(),
                "Coupling can be up to ERK (at most)!",
            );
            return ARK_INVALID_TABLE;
        }
    }

    /* Check that G tables are lower triangular */
    if !C.G.is_empty() {
        Gabs = 0.0;
        for k in 0..C.nmat {
            for i in 0..C.stages {
                for j in (i + 1)..C.stages {
                    Gabs += SUNRabs(C.G[k as usize][i as usize][j as usize]);
                }
            }
        }
        if Gabs > tol {
            arkProcessError(
                Some(ark_mem),
                ARK_INVALID_TABLE,
                line!() as i32,
                "mriStep_CheckCoupling",
                file!(),
                "Coupling can be up to DIRK (at most)!",
            );
            return ARK_INVALID_TABLE;
        }
    }

    /* Check that MERK "groups" are structured appropriately */
    if C.type_ == MRISTEP_MERK {
        let mut group_counter: Vec<i32> = vec![0; (C.stages + 1) as usize];
        for i in 0..C.ngroup {
            for j in 0..C.stages {
                let k = C.group[i as usize][j as usize];
                if k == -1 {
                    break;
                }
                if (k < 0) || (k > C.stages) {
                    arkProcessError(
                        Some(ark_mem),
                        ARK_INVALID_TABLE,
                        line!() as i32,
                        "mriStep_CheckCoupling",
                        file!(),
                        "Invalid MERK group index!",
                    );
                    return ARK_INVALID_TABLE;
                }
                group_counter[k as usize] += 1;
            }
        }
        for i in 1..=C.stages {
            if (group_counter[i as usize] == 0) || (group_counter[i as usize] > 1) {
                arkProcessError(
                    Some(ark_mem),
                    ARK_INVALID_TABLE,
                    line!() as i32,
                    "mriStep_CheckCoupling",
                    file!(),
                    "Duplicated/missing stages from MERK groups!",
                );
                return ARK_INVALID_TABLE;
            }
        }
    }

    /* Check that no stage has MRISTAGE_DIRK_FAST type (for now) */
    let stages = C.stages;
    drop(C);
    okay = SUNTRUE;
    for i in 0..stages {
        if mriStepCoupling_GetStageType(&mric, i) == MRISTAGE_DIRK_FAST {
            okay = SUNFALSE;
        }
    }
    if !okay {
        arkProcessError(
            Some(ark_mem),
            ARK_INVALID_TABLE,
            line!() as i32,
            "mriStep_CheckCoupling",
            file!(),
            "solve-coupled DIRK stages not currently supported",
        );
        return ARK_INVALID_TABLE;
    }
    let C = mric.borrow();

    /* check that MRI-GARK stage times are sorted */
    if (C.type_ == MRISTEP_IMPLICIT) || (C.type_ == MRISTEP_EXPLICIT) || (C.type_ == MRISTEP_IMEX)
    {
        okay = SUNTRUE;
        for i in 1..C.stages {
            if (C.c[i as usize] - C.c[(i - 1) as usize]) < -tol {
                okay = SUNFALSE;
            }
        }
        if !okay {
            arkProcessError(
                Some(ark_mem),
                ARK_INVALID_TABLE,
                line!() as i32,
                "mriStep_CheckCoupling",
                file!(),
                "Stage times must be sorted.",
            );
            return ARK_INVALID_TABLE;
        }
    }

    /* check that the first stage is just the old step solution */
    Gabs = SUNRabs(C.c[0]);
    for k in 0..C.nmat {
        for j in 0..C.stages {
            if !C.W.is_empty() {
                Gabs += SUNRabs(C.W[k as usize][0][j as usize]);
            }
            if !C.G.is_empty() {
                Gabs += SUNRabs(C.G[k as usize][0][j as usize]);
            }
        }
    }
    if Gabs > tol {
        arkProcessError(
            Some(ark_mem),
            ARK_INVALID_TABLE,
            line!() as i32,
            "mriStep_CheckCoupling",
            file!(),
            "First stage must equal old solution.",
        );
        return ARK_INVALID_TABLE;
    }

    /* check that the last stage is at the final time */
    if SUNRabs(ONE - C.c[(C.stages - 1) as usize]) > tol {
        arkProcessError(
            Some(ark_mem),
            ARK_INVALID_TABLE,
            line!() as i32,
            "mriStep_CheckCoupling",
            file!(),
            "Final stage time must be equal 1.",
        );
        return ARK_INVALID_TABLE;
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  mriStep_StageERKFast

  This routine performs a single MRI stage, is, with explicit
  slow time scale and fast time scale that requires evolution.

  On input, ycur is the initial condition for the fast IVP at t0.
  On output, ycur is the solution of the fast IVP at tf.
  The vector ytemp is only used if temporal adaptivity is enabled,
  and the fast error is not provided by the fast integrator.

  get_inner_dsm indicates whether this stage is one that should
  accumulate an inner temporal error estimate.

  Port note: the C `ARKodeMRIStepMem step_mem` parameter is dropped;
  the record is re-acquired granularly through `mriStep_mem_mut`,
  because this routine re-enters `ark_mem` (inner-stepper evolve,
  user callbacks, arkProcessError) and no borrow may be live then.
  ---------------------------------------------------------------*/
pub fn mriStep_StageERKFast(
    ark_mem: &ARKodeMem,
    t0: sunrealtype,
    tf: sunrealtype,
    ycur: &N_Vector,
    ytemp: &N_Vector,
    get_inner_dsm: sunbooleantype,
) -> i32 {
    let _ = ytemp; /* SUNDIALS_MAYBE_UNUSED */

    let stepper = mriStep_mem_mut(ark_mem).stepper.clone().expect("stepper");

    /* pre inner evolve function (if supplied) */
    let pre_inner_evolve = mriStep_mem_mut(ark_mem).pre_inner_evolve;
    if let Some(pre_inner_evolve) = pre_inner_evolve {
        let forcing = stepper.forcing.borrow().clone();
        let nforcing = *stepper.nforcing.borrow();
        let mut user_data = ark_mem.borrow_mut().user_data.take();
        let retval = pre_inner_evolve(t0, &forcing, nforcing, &mut user_data);
        ark_mem.borrow_mut().user_data = user_data;
        if retval != 0 {
            return ARK_OUTERTOINNER_FAIL;
        }
    }

    /* Get the adaptivity type (if applicable) */
    let adapt_type = if get_inner_dsm {
        let hcontroller = ark_mem
            .borrow()
            .hadapt_mem
            .as_ref()
            .expect("hadapt_mem")
            .hcontroller
            .clone();
        match hcontroller.as_ref() {
            Some(C) => SUNAdaptController_GetType(C),
            None => SUN_ADAPTCONTROLLER_NONE,
        }
    } else {
        SUN_ADAPTCONTROLLER_NONE
    };

    /* advance inner method in time */
    let retval = mriStepInnerStepper_Evolve(&stepper, t0, tf, ycur);

    if retval < 0 {
        arkProcessError(
            Some(ark_mem),
            ARK_INNERSTEP_FAIL,
            line!() as i32,
            "mriStep_StageERKFast",
            file!(),
            "Failure when evolving the inner stepper",
        );
        return ARK_INNERSTEP_FAIL;
    }
    if retval > 0 {
        /* increment stepper-specific counter, and decrement ARKODE-level nonlinear
           solver counter (since that will be incremented automatically by ARKODE).
           Return with "TRY_AGAIN" which should cause ARKODE to cut the step size
           and retry the step. */
        mriStep_mem_mut(ark_mem).inner_fails += 1;
        ark_mem.borrow_mut().ncfn -= 1;
        return TRY_AGAIN;
    }

    /* for normal stages (i.e., not the embedding) with MRI adaptivity enabled, get an
       estimate for the fast time scale error */
    if get_inner_dsm {
        /* if the fast integrator uses adaptive steps, retrieve the error estimate */
        if adapt_type == SUN_ADAPTCONTROLLER_MRI_H_TOL {
            /* C passes &step_mem->inner_dsm; mirror the write back into the field */
            let mut inner_dsm: sunrealtype = mriStep_mem_mut(ark_mem).inner_dsm;
            let retval = mriStepInnerStepper_GetAccumulatedError(&stepper, &mut inner_dsm);
            mriStep_mem_mut(ark_mem).inner_dsm = inner_dsm;
            if retval != ARK_SUCCESS {
                arkProcessError(
                    Some(ark_mem),
                    ARK_INNERSTEP_FAIL,
                    line!() as i32,
                    "mriStep_StageERKFast",
                    file!(),
                    "Unable to get accumulated error from the inner stepper",
                );
                return ARK_INNERSTEP_FAIL;
            }

            /* scale the error estimate by 1/rtol to account for different inner/outer tolerances */
            let reltol = ark_mem.borrow().reltol;
            mriStep_mem_mut(ark_mem).inner_dsm /= reltol;
        }
    }

    /* post inner evolve function (if supplied) */
    let post_inner_evolve = mriStep_mem_mut(ark_mem).post_inner_evolve;
    if let Some(post_inner_evolve) = post_inner_evolve {
        let mut user_data = ark_mem.borrow_mut().user_data.take();
        let retval = post_inner_evolve(tf, ycur, &mut user_data);
        ark_mem.borrow_mut().user_data = user_data;
        if retval != 0 {
            return ARK_INNERTOOUTER_FAIL;
        }
    }

    /* return with success */
    ARK_SUCCESS
}

/*---------------------------------------------------------------
  mriStep_StageERKNoFast

  This routine performs a single MRI stage with explicit slow
  time scale only (no fast time scale evolution).

  Port note: the C `ARKodeMRIStepMem step_mem` parameter is dropped
  (see mriStep_StageERKFast).
  ---------------------------------------------------------------*/
pub fn mriStep_StageERKNoFast(ark_mem: &ARKodeMem, is: i32) -> i32 {
    let mric = mriStep_mem_mut(ark_mem).MRIC.clone().expect("MRIC");

    /* determine effective ERK coefficients (store in Ae_row and Ai_row) */
    let retval = {
        let mut guard = mriStep_mem_mut(ark_mem);
        let step_mem = &mut *guard;
        mriStep_RKCoeffs(
            &mric,
            is,
            &step_mem.stage_map,
            &mut step_mem.Ae_row,
            &mut step_mem.Ai_row,
        )
    };
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* call fused vector operation to perform ERK update -- bound on
       j needs "SUNMIN" to handle the case of an "embedding" stage */
    let (h, ycur) = {
        let m = ark_mem.borrow();
        (m.h, m.ycur.clone().expect("ycur"))
    };
    let mut cvals: Vec<sunrealtype> = Vec::new();
    let mut Xvecs: Vec<N_Vector> = Vec::new();
    cvals.push(ONE);
    Xvecs.push(ycur.clone());
    let mut nvec = 1;
    {
        let step_mem = mriStep_mem_mut(ark_mem);
        for j in 0..SUNMIN(is, step_mem.stages) {
            if step_mem.explicit_rhs && step_mem.stage_map[j as usize] > -1 {
                cvals.push(h * step_mem.Ae_row[step_mem.stage_map[j as usize] as usize]);
                Xvecs.push(step_mem.Fse[step_mem.stage_map[j as usize] as usize].clone());
                nvec += 1;
            }
            if step_mem.implicit_rhs && step_mem.stage_map[j as usize] > -1 {
                cvals.push(h * step_mem.Ai_row[step_mem.stage_map[j as usize] as usize]);
                Xvecs.push(step_mem.Fsi[step_mem.stage_map[j as usize] as usize].clone());
                nvec += 1;
            }
        }
    }
    /* Is there a case where we have an explicit update with Fsi? */

    let retval = N_VLinearCombination(nvec, &cvals, &Xvecs, &ycur);
    if retval != 0 {
        return ARK_VECTOROP_ERR;
    }
    ARK_SUCCESS
}

/*---------------------------------------------------------------
  mriStep_StageDIRKFast

  This routine performs a single stage of a "solve coupled"
  MRI method, i.e. a stage that is DIRK on the slow time scale
  and involves evolution of the fast time scale, in a
  fully-coupled fashion.

  Port note: the C `ARKodeMRIStepMem step_mem` parameter is dropped
  (see mriStep_StageERKFast).
  ---------------------------------------------------------------*/
pub fn mriStep_StageDIRKFast(ark_mem: &ARKodeMem, is: i32, nflagPtr: &mut i32) -> i32 {
    let _ = is; /* SUNDIALS_MAYBE_UNUSED */
    let _ = nflagPtr; /* SUNDIALS_MAYBE_UNUSED */

    /* this is not currently implemented */
    arkProcessError(
        Some(ark_mem),
        ARK_INVALID_TABLE,
        line!() as i32,
        "mriStep_StageDIRKFast",
        file!(),
        "This routine is not yet implemented.",
    );
    ARK_INVALID_TABLE
}

/*---------------------------------------------------------------
  mriStep_StageDIRKNoFast

  This routine performs a single MRI stage with implicit slow
  time scale only (no fast time scale evolution).

  Port note: the C `ARKodeMRIStepMem step_mem` parameter is dropped
  (see mriStep_StageERKFast).
  ---------------------------------------------------------------*/
pub fn mriStep_StageDIRKNoFast(ark_mem: &ARKodeMem, is: i32, nflagPtr: &mut i32) -> i32 {
    /* store current stage index (for an "embedded" stage, subtract 1) */
    let istage = {
        let mut step_mem = mriStep_mem_mut(ark_mem);
        step_mem.istage = if is == step_mem.stages { is - 1 } else { is };
        step_mem.istage
    };

    /* Call predictor for current stage solution (result placed in zpred) */
    let zpred = mriStep_mem_mut(ark_mem).zpred.clone().expect("zpred");
    let retval = mriStep_Predict(ark_mem, istage, &zpred);
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* If a user-supplied predictor routine is provided, call that here
       Note that mriStep_Predict is *still* called, so this user-supplied
       routine can just "clean up" the built-in prediction, if desired. */
    let stage_predict = mriStep_mem_mut(ark_mem).stage_predict;
    if let Some(stage_predict) = stage_predict {
        let tcur = ark_mem.borrow().tcur;
        let mut user_data = ark_mem.borrow_mut().user_data.take();
        let retval = stage_predict(tcur, &zpred, &mut user_data);
        ark_mem.borrow_mut().user_data = user_data;
        if retval < 0 {
            return ARK_USER_PREDICT_FAIL;
        }
        if retval > 0 {
            return TRY_AGAIN;
        }
    }

    /* determine effective DIRK coefficients (store in cvals) */
    let mric = mriStep_mem_mut(ark_mem).MRIC.clone().expect("MRIC");
    let retval = {
        let mut guard = mriStep_mem_mut(ark_mem);
        let step_mem = &mut *guard;
        mriStep_RKCoeffs(
            &mric,
            is,
            &step_mem.stage_map,
            &mut step_mem.Ae_row,
            &mut step_mem.Ai_row,
        )
    };
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* Set up data for evaluation of DIRK stage residual (data stored in sdata) */
    let retval = mriStep_StageSetup(ark_mem);
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* perform implicit solve (result is stored in ark_mem->ycur); return
       with positive value on anything but success */
    *nflagPtr = mriStep_Nls(ark_mem, *nflagPtr);
    if *nflagPtr != ARK_SUCCESS {
        return TRY_AGAIN;
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  mriStep_ComputeInnerForcing

  Constructs the 'coefficient' vectors for the forcing polynomial
  for a 'fast' outer MRI-GARK stage i:

  p_i(theta) = sum_{k=0}^{n-1} forcing[k] * theta^k

  where theta = (t - t0) / (tf-t0) is the mapped 'time' for
  each 'fast' MRIStep evolution, with:
  * t0 -- the start of this outer MRIStep stage
  * tf-t0, the temporal width of this MRIStep stage
  * n -- shorthand for MRIC->nmat

  Defining cdiff = (tf-t0)/h, explicit and solve-decoupled
  implicit or IMEX MRI-based methods define this forcing polynomial
  for each outer stage i > 0:

  p_i(theta) = w_i,0(theta) * fse_0 + ... + w_i,{i-1}(theta) * fse_{i-1}
             + g_i,0(theta) * fsi_0 + ... + g_i,{i-1}(theta) * fsi_{i-1}

  where

  w_i,j(theta) = w_0,i,j + w_1,i,j * theta + ... + w_n,i,j * theta^{n-1},
  w_k,i,j = 1/cdiff * MRIC->W[k][i][j]

  and

  g_i,j(theta) = g_0,i,j + g_1,i,j * theta + ... + g_n,i,j * theta^{n-1},
  g_k,i,j = 1/cdiff * MRIC->G[k][i][j]

  Converting to the appropriate form, we have

  p_i(theta) = ( w_0,i,0 * fse_0 + ... + w_0,i,{i-1} * fse_{i-1} +
                 g_0,i,0 * fsi_0 + ... + g_0,i,{i-1} * fsi_{i-1} ) * theta^0
             + ( w_1,i,0 * fse_0 + ... + w_1,i,{i-1} * fse_{i-1} +
                 g_1,i,0 * fsi_0 + ... + g_1,i,{i-1} * fsi_{i-1} ) * theta^1
                                    .
                                    .
                                    .
             + ( w_n,i,0 * fse_0 + ... + w_n,i,{i-1} * fse_{i-1} +
                 g_n,i,0 * fsi_0 + ... + g_n,i,{i-1} * fsi_{i-1} ) * theta^{n-1}

  Thus we define the forcing vectors for k = 0,...,nmat - 1

  forcing[k] = w_k,i,0 * fse_0 + ... + w_k,i,{i-1} * fse_{i-1}
             + g_k,i,0 * fsi_0 + ... + g_k,i,{i-1} * fsi_{i-1}

             = 1 / cdiff *
               ( W[k][i][0] * fse_0 + ... + W[k][i][i-1] * fse_{i-1} +
               ( G[k][i][0] * fsi_0 + ... + G[k][i][i-1] * fsi_{i-1} )

  We may use an identical formula for MERK methods, so long as we set t0=tn,
  tf=tn+h, stage_map[j]=j (identity map), and implicit_rhs=SUNFALSE.
  With this configuration: tf-t0=h, theta = (t-tn)/h, and cdiff=1.  MERK methods
  define the forcing polynomial for each outer stage i > 0 as:

  p_i(theta) = w_i,0(theta) * fse_0 + ... + w_i,{i-1}(theta) * fse_{i-1}

  where

  w_i,j(theta) = w_0,i,j + w_1,i,j * theta + ... + w_n,i,j * theta^{n-1},
  w_k,i,j = MRIC->W[k][i][j]

  which is equivalent to the formula above.

  We may use a similar formula for MRISR methods, so long as we set t0=tn,
  tf=tn+h*ci, stage_map[j]=j (identity map), and implicit_rhs=SUNFALSE.
  With this configuration: tf-t0=ci*h, theta = (t-tn)/(ci*h), and cdiff=1/ci.
  MRISR methods define the forcing polynomial for each outer stage i > 0 as:

  p_i(theta) = w_i,0(theta) * fs_0 + ... + w_i,{i-1}(theta) * fs_{i-1}

  where fs_j = fse_j + fsi_j and

  w_i,j(theta) = w_0,i,j + w_1,i,j * theta + ... + w_n,i,j * theta^{n-1},
  w_k,i,j = 1/ci * MRIC->W[k][i][j]

  which is equivalent to the formula above, so long as the stage RHS vectors
  Fse[j] are repurposed to instead store (fse_j + fsi_j).

  This routine additionally returns a success/failure flag:
     ARK_SUCCESS -- successful evaluation

  Port note: the C `ARKodeMRIStepMem step_mem` parameter is dropped
  (see mriStep_StageERKFast); `cvals`/`Xvecs` are function-local
  rebuilds of the step_mem scratch arrays (locked house pattern).
  ---------------------------------------------------------------*/
pub fn mriStep_ComputeInnerForcing(
    ark_mem: &ARKodeMem,
    stage: i32,
    t0: sunrealtype,
    tf: sunrealtype,
) -> i32 {
    let (stepper, mric, mut implicit_rhs, mut explicit_rhs, stages) = {
        let step_mem = mriStep_mem_mut(ark_mem);
        (
            step_mem.stepper.clone().expect("stepper"),
            step_mem.MRIC.clone().expect("MRIC"),
            step_mem.implicit_rhs,
            step_mem.explicit_rhs,
            step_mem.stages,
        )
    };

    /* Set inner forcing time normalization constants */
    *stepper.tshift.borrow_mut() = t0;
    *stepper.tscale.borrow_mut() = tf - t0;

    /* Adjust implicit/explicit RHS flags for MRISR methods, since these
       ignore the G coefficients in the forcing function */
    let is_mrisr = mric.borrow().type_ == MRISTEP_SR;
    if is_mrisr {
        implicit_rhs = SUNFALSE;
        explicit_rhs = SUNTRUE;
    }

    /* compute inner forcing vectors (assumes cdiff != 0) */
    let mut Xvecs: Vec<N_Vector> = Vec::new();
    {
        let step_mem = mriStep_mem_mut(ark_mem);
        for j in 0..SUNMIN(stage, stages) {
            if explicit_rhs && step_mem.stage_map[j as usize] > -1 {
                Xvecs.push(step_mem.Fse[step_mem.stage_map[j as usize] as usize].clone());
            }
            if implicit_rhs && step_mem.stage_map[j as usize] > -1 {
                Xvecs.push(step_mem.Fsi[step_mem.stage_map[j as usize] as usize].clone());
            }
        }
    }

    let nmat = mric.borrow().nmat;
    let rcdiff = ark_mem.borrow().h / (tf - t0);

    for k in 0..nmat {
        let mut cvals: Vec<sunrealtype> = Vec::new();
        let mut nstore = 0;
        {
            let C = mric.borrow();
            let step_mem = mriStep_mem_mut(ark_mem);
            for j in 0..SUNMIN(stage, stages) {
                if step_mem.stage_map[j as usize] > -1 {
                    if explicit_rhs && implicit_rhs {
                        /* ImEx */
                        cvals.push(rcdiff * C.W[k as usize][stage as usize][j as usize]);
                        nstore += 1;
                        cvals.push(rcdiff * C.G[k as usize][stage as usize][j as usize]);
                        nstore += 1;
                    } else if explicit_rhs {
                        /* explicit only */
                        cvals.push(rcdiff * C.W[k as usize][stage as usize][j as usize]);
                        nstore += 1;
                    } else {
                        /* implicit only */
                        cvals.push(rcdiff * C.G[k as usize][stage as usize][j as usize]);
                        nstore += 1;
                    }
                }
            }
        }

        let forcing_k = stepper.forcing.borrow()[k as usize].clone();
        let retval = N_VLinearCombination(nstore, &cvals, &Xvecs, &forcing_k);
        if retval != 0 {
            return ARK_VECTOROP_ERR;
        }
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  Compute/return the effective RK coefficients for a "nofast"
  stage.  We may assume that "A" has already been allocated.
  ---------------------------------------------------------------*/
pub fn mriStep_RKCoeffs(
    MRIC: &MRIStepCoupling,
    is: i32,
    stage_map: &[i32],
    Ae_row: &mut [sunrealtype],
    Ai_row: &mut [sunrealtype],
) -> i32 {
    let C = MRIC.borrow();

    if is < 1 || is > C.stages || stage_map.is_empty() || Ae_row.is_empty() || Ai_row.is_empty() {
        return ARK_INVALID_TABLE;
    }

    /* initialize RK coefficient array */
    for j in 0..C.stages {
        Ae_row[j as usize] = ZERO;
        Ai_row[j as usize] = ZERO;
    }

    /* compute RK coefficients -- note that bounds on j need
       "SUNMIN" to handle the case of an "embedding" stage */
    for k in 0..C.nmat {
        let kconst = ONE / (k as sunrealtype + ONE);
        if !C.W.is_empty() {
            for j in 0..SUNMIN(is, C.stages - 1) {
                if stage_map[j as usize] > -1 {
                    Ae_row[stage_map[j as usize] as usize] +=
                        C.W[k as usize][is as usize][j as usize] * kconst;
                }
            }
        }
        if !C.G.is_empty() {
            for j in 0..=SUNMIN(is, C.stages - 1) {
                if stage_map[j as usize] > -1 {
                    Ai_row[stage_map[j as usize] as usize] +=
                        C.G[k as usize][is as usize][j as usize] * kconst;
                }
            }
        }
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  mriStep_Predict

  This routine computes the prediction for a specific internal
  stage solution, storing the result in yguess.  The
  prediction is done using the interpolation structure in
  extrapolation mode, hence stages "far" from the previous time
  interval are predicted using lower order polynomials than the
  "nearby" stages.
  ---------------------------------------------------------------*/
pub fn mriStep_Predict(ark_mem: &ARKodeMem, istage: i32, yguess: &N_Vector) -> i32 {
    /* access ARKodeMRIStepMem structure */
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            None,
            ARK_MEM_NULL,
            line!() as i32,
            "mriStep_Predict",
            file!(),
            MSG_MRISTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    let predictor = mriStep_mem_mut(ark_mem).predictor;

    /* verify that interpolation structure is provided */
    let no_interp = ark_mem.borrow().interp.is_none();
    if no_interp && (predictor > 0) {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "mriStep_Predict",
            file!(),
            "Interpolation structure is NULL",
        );
        return ARK_MEM_NULL;
    }

    /* local shortcuts for use with fused vector operations */
    let mut cvals: Vec<sunrealtype> = Vec::new();
    let mut Xvecs: Vec<N_Vector> = Vec::new();

    /* if the first step (or if resized), use initial condition as guess */
    let initsetup = ark_mem.borrow().initsetup;
    if initsetup {
        let yn = ark_mem.borrow().yn.clone().expect("yn");
        N_VScale(ONE, &yn, yguess);
        return ARK_SUCCESS;
    }

    let mric = mriStep_mem_mut(ark_mem).MRIC.clone().expect("MRIC");

    /* set evaluation time tau as relative shift from previous successful time */
    let mut tau = {
        let m = ark_mem.borrow();
        mric.borrow().c[istage as usize] * m.h / m.hold
    };

    /* use requested predictor formula */
    match predictor {
        1 => {
            /***** Interpolatory Predictor 1 -- all to max order *****/
            let retval = arkPredict_MaximumOrder(ark_mem, tau, yguess);
            if retval != ARK_ILL_INPUT {
                return retval;
            }
        }

        2 => {
            /***** Interpolatory Predictor 2 -- decrease order w/ increasing level of extrapolation *****/
            let retval = arkPredict_VariableOrder(ark_mem, tau, yguess);
            if retval != ARK_ILL_INPUT {
                return retval;
            }
        }

        3 => {
            /***** Cutoff predictor: max order interpolatory output for stages "close"
                   to previous step, first-order predictor for subsequent stages *****/
            let retval = arkPredict_CutoffOrder(ark_mem, tau, yguess);
            if retval != ARK_ILL_INPUT {
                return retval;
            }
        }

        4 => {
            /***** Bootstrap predictor: if any previous stage in step has nonzero c_i,
                   construct a quadratic Hermite interpolant for prediction; otherwise
                   use the trivial predictor.  The actual calculations are performed in
                   arkPredict_Bootstrap, but here we need to determine the appropriate
                   stage, c_j, to use. *****/

            /* determine if any previous stages in step meet criteria */
            let mut jstage: i32 = -1;
            {
                let C = mric.borrow();
                for i in 0..istage {
                    jstage = if C.c[i as usize] != ZERO { i } else { jstage };
                }
            }

            /* if using the trivial predictor, break */
            if jstage != -1 {
                /* find the "optimal" previous stage to use */
                {
                    let C = mric.borrow();
                    let step_mem = mriStep_mem_mut(ark_mem);
                    for i in 0..istage {
                        if (C.c[i as usize] > C.c[jstage as usize])
                            && (C.c[i as usize] != ZERO)
                            && step_mem.stage_map[i as usize] > -1
                        {
                            jstage = i;
                        }
                    }
                }

                /* set stage time, stage RHS and interpolation values */
                let ark_h = ark_mem.borrow().h;
                let h = ark_h * mric.borrow().c[jstage as usize];
                tau = ark_h * mric.borrow().c[istage as usize];
                let mut nvec = 0;
                {
                    let step_mem = mriStep_mem_mut(ark_mem);
                    if step_mem.implicit_rhs {
                        /* Implicit piece */
                        cvals.push(ONE);
                        Xvecs.push(
                            step_mem.Fsi[step_mem.stage_map[jstage as usize] as usize].clone(),
                        );
                        nvec += 1;
                    }
                    if step_mem.explicit_rhs {
                        /* Explicit piece */
                        cvals.push(ONE);
                        Xvecs.push(
                            step_mem.Fse[step_mem.stage_map[jstage as usize] as usize].clone(),
                        );
                        nvec += 1;
                    }
                }

                /* call predictor routine */
                let retval = arkPredict_Bootstrap(ark_mem, h, tau, nvec, &mut cvals, &mut Xvecs, yguess);
                if retval != ARK_ILL_INPUT {
                    return retval;
                }
            }
        }

        _ => {}
    }

    /* if we made it here, use the trivial predictor (previous step solution) */
    let yn = ark_mem.borrow().yn.clone().expect("yn");
    N_VScale(ONE, &yn, yguess);
    ARK_SUCCESS
}

/*---------------------------------------------------------------
  mriStep_StageSetup

  This routine sets up the stage data for computing the
  solve-decoupled MRI stage residual, along with the step- and
  method-related factors gamma, gammap and gamrat.

  At the ith stage, we compute the residual vector for
  z=z_i=zp+zc:
    r = z - z_{i-1} - h*sum_{j=0}^{i} A(i,j)*F(z_j)
    r = (zp + zc) - z_{i-1} - h*sum_{j=0}^{i} A(i,j)*F(z_j)
    r = (zc - gamma*F(z)) - data,
  where data = (z_{i-1} - zp + h*sum_{j=0}^{i-1} A(i,j)*F(z_j))
  corresponds to existing information.  This routine computes
  this 'data' vector and stores in step_mem->sdata.

  Note: on input, this row A(i,:) is already stored in rkcoeffs.
  ---------------------------------------------------------------*/
pub fn mriStep_StageSetup(ark_mem: &ARKodeMem) -> i32 {
    /* access ARKodeMRIStepMem structure */
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            None,
            ARK_MEM_NULL,
            line!() as i32,
            "mriStep_StageSetup",
            file!(),
            MSG_MRISTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    let (h, firststage, ycur) = {
        let m = ark_mem.borrow();
        (m.h, m.firststage, m.ycur.clone().expect("ycur"))
    };

    /* Set shortcut to current stage index */
    let i = mriStep_mem_mut(ark_mem).istage;

    /* local shortcuts for fused vector operations */
    let mut cvals: Vec<sunrealtype> = Vec::new();
    let mut Xvecs: Vec<N_Vector> = Vec::new();

    let sdata;
    let mut nvec;
    {
        let mut step_mem = mriStep_mem_mut(ark_mem);

        /* Update gamma (if the method contains an implicit component) */
        step_mem.gamma = h * step_mem.Ai_row[step_mem.stage_map[i as usize] as usize];

        if firststage {
            step_mem.gammap = step_mem.gamma;
        }
        step_mem.gamrat = if firststage {
            ONE
        } else {
            step_mem.gamma / step_mem.gammap
        };

        /* set cvals and Xvecs for setting stage data */
        cvals.push(ONE);
        Xvecs.push(ycur.clone());
        cvals.push(-ONE);
        Xvecs.push(step_mem.zpred.clone().expect("zpred"));
        nvec = 2;

        for j in 0..i {
            if step_mem.explicit_rhs && step_mem.stage_map[j as usize] > -1 {
                cvals.push(h * step_mem.Ae_row[step_mem.stage_map[j as usize] as usize]);
                Xvecs.push(step_mem.Fse[step_mem.stage_map[j as usize] as usize].clone());
                nvec += 1;
            }
            if step_mem.implicit_rhs && step_mem.stage_map[j as usize] > -1 {
                cvals.push(h * step_mem.Ai_row[step_mem.stage_map[j as usize] as usize]);
                Xvecs.push(step_mem.Fsi[step_mem.stage_map[j as usize] as usize].clone());
                nvec += 1;
            }
        }

        sdata = step_mem.sdata.clone().expect("sdata");
    }

    /* call fused vector operation to do the work */
    let retval = N_VLinearCombination(nvec, &cvals, &Xvecs, &sdata);
    if retval != 0 {
        return ARK_VECTOROP_ERR;
    }

    /* return with success */
    ARK_SUCCESS
}

/*---------------------------------------------------------------
  mriStep_SlowRHS:

  Wrapper routine to call the user-supplied slow RHS functions,
  f(t,y) = fse(t,y) + fsi(t,y), with API matching
  ARKTimestepFullRHSFn.  This is only used to determine an
  initial slow time-step size to use when one is not specified
  by the user (i.e., mode should correspond with
  ARK_FULLRHS_START.
  ---------------------------------------------------------------*/
pub fn mriStep_SlowRHS(
    ark_mem: &ARKodeMem,
    t: sunrealtype,
    y: &N_Vector,
    f: &N_Vector,
    mode: i32,
) -> i32 {
    let _ = mode; /* SUNDIALS_MAYBE_UNUSED */

    /* access ARKodeMRIStepMem structure */
    let retval = mriStep_AccessStepMem(ark_mem, "mriStep_SlowRHS");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* call the user-supplied pre-RHS function (if supplied) */
    let PreRhsFn = ark_mem.borrow().PreRhsFn;
    if let Some(PreRhsFn) = PreRhsFn {
        let mut user_data = ark_mem.borrow_mut().user_data.take();
        let retval = PreRhsFn(t, y, &mut user_data);
        ark_mem.borrow_mut().user_data = user_data;
        if retval != 0 {
            return ARK_PRERHSFN_FAIL;
        }
    }

    let (implicit_rhs, explicit_rhs) = {
        let step_mem = mriStep_mem_mut(ark_mem);
        (step_mem.implicit_rhs, step_mem.explicit_rhs)
    };

    /* call fsi if the problem has an implicit component */
    if implicit_rhs {
        let (fsi, Fsi0) = {
            let step_mem = mriStep_mem_mut(ark_mem);
            (step_mem.fsi.expect("fsi"), step_mem.Fsi[0].clone())
        };
        let mut user_data = ark_mem.borrow_mut().user_data.take();
        let retval = fsi(t, y, &Fsi0, &mut user_data);
        ark_mem.borrow_mut().user_data = user_data;
        {
            let mut step_mem = mriStep_mem_mut(ark_mem);
            step_mem.nfsi += 1;
            step_mem.fsi_is_current = SUNTRUE;
        }
        if retval != 0 {
            arkProcessError(
                Some(ark_mem),
                ARK_RHSFUNC_FAIL,
                line!() as i32,
                "mriStep_SlowRHS",
                file!(),
                &MSG_ARK_RHSFUNC_FAILED(t),
            );
            return ARK_RHSFUNC_FAIL;
        }

        /* Add external forcing, if applicable */
        let impforcing = mriStep_mem_mut(ark_mem).impforcing;
        if impforcing {
            let mut nvec = 1;
            let (cvals, Xvecs) = {
                let mut step_mem = mriStep_mem_mut(ark_mem);
                step_mem.cvals[0] = ONE;
                step_mem.Xvecs[0] = Some(Fsi0.clone());
                mriStep_ApplyForcing(&mut step_mem, t, ONE, &mut nvec);
                (step_mem.cvals.clone(), mriStep_xvecs(&step_mem, nvec))
            };
            N_VLinearCombination(nvec, &cvals, &Xvecs, &Fsi0);
        }
    }

    /* call fse if the problem has an explicit component */
    if explicit_rhs {
        let (fse, Fse0) = {
            let step_mem = mriStep_mem_mut(ark_mem);
            (step_mem.fse.expect("fse"), step_mem.Fse[0].clone())
        };
        let mut user_data = ark_mem.borrow_mut().user_data.take();
        let retval = fse(t, y, &Fse0, &mut user_data);
        ark_mem.borrow_mut().user_data = user_data;
        {
            let mut step_mem = mriStep_mem_mut(ark_mem);
            step_mem.nfse += 1;
            step_mem.fse_is_current = SUNTRUE;
        }
        if retval != 0 {
            arkProcessError(
                Some(ark_mem),
                ARK_RHSFUNC_FAIL,
                line!() as i32,
                "mriStep_SlowRHS",
                file!(),
                &MSG_ARK_RHSFUNC_FAILED(t),
            );
            return ARK_RHSFUNC_FAIL;
        }

        /* Add external forcing, if applicable */
        let expforcing = mriStep_mem_mut(ark_mem).expforcing;
        if expforcing {
            let mut nvec = 1;
            let (cvals, Xvecs) = {
                let mut step_mem = mriStep_mem_mut(ark_mem);
                step_mem.cvals[0] = ONE;
                step_mem.Xvecs[0] = Some(Fse0.clone());
                mriStep_ApplyForcing(&mut step_mem, t, ONE, &mut nvec);
                (step_mem.cvals.clone(), mriStep_xvecs(&step_mem, nvec))
            };
            N_VLinearCombination(nvec, &cvals, &Xvecs, &Fse0);
        }
    }

    /* combine RHS vectors into output */
    if explicit_rhs && implicit_rhs
    /* ImEx */
    {
        let (Fse0, Fsi0) = {
            let step_mem = mriStep_mem_mut(ark_mem);
            (step_mem.Fse[0].clone(), step_mem.Fsi[0].clone())
        };
        N_VLinearSum(ONE, &Fse0, ONE, &Fsi0, f);
    } else if implicit_rhs {
        let Fsi0 = mriStep_mem_mut(ark_mem).Fsi[0].clone();
        N_VScale(ONE, &Fsi0, f);
    } else {
        let Fse0 = mriStep_mem_mut(ark_mem).Fse[0].clone();
        N_VScale(ONE, &Fse0, f);
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  mriStep_Hin

  This routine computes a tentative initial step size h0.  This
  employs the same safeguards as ARKODE's arkHin utility routine,
  but employs a simpler algorithm that estimates the first step
  such that an explicit Euler step (for only the slow RHS
  routine(s)) would be within user-specified tolerances of the
  initial condition.
  ---------------------------------------------------------------*/
pub fn mriStep_Hin(
    ark_mem: &ARKodeMem,
    tcur: sunrealtype,
    tout: sunrealtype,
    fcur: &N_Vector,
    h: &mut sunrealtype,
) -> i32 {
    /* If tout is too close to tn, give up */
    let tdiff = tout - tcur;
    if tdiff == ZERO {
        return ARK_TOO_CLOSE;
    }
    let sign: i32 = if tdiff > ZERO { 1 } else { -1 };
    let tdist = SUNRabs(tdiff);
    let tround = ark_mem.borrow().uround * SUNMAX(SUNRabs(tcur), SUNRabs(tout));
    if tdist < TWO * tround {
        return ARK_TOO_CLOSE;
    }

    /* h0 should bound the change due to a forward Euler step, and
       include safeguard against "too-small" ||f(t0,y0)||: */
    let ewt = ark_mem.borrow().ewt.clone().expect("ewt");
    let fnorm = N_VWrmsNorm(fcur, &ewt) / H0_BIAS;
    let h0_inv = SUNMAX(ONE / H0_UBFACTOR / tdist, fnorm);
    *h = (sign as sunrealtype) / h0_inv;
    ARK_SUCCESS
}

/*===============================================================
  User-callable functions for a custom inner integrator
  ===============================================================*/

/// C `MRIStepInnerStepper_Create(SUNContext sunctx, MRIStepInnerStepper* stepper)`.
///
/// The C `!sunctx` guard is unreachable through `&SUNContext`.
pub fn MRIStepInnerStepper_Create(
    sunctx: &SUNContext,
    stepper: &mut Option<MRIStepInnerStepper>,
) -> i32 {
    *stepper = None;

    /* malloc + memset(0) of the record and of its ops table */
    *stepper = Some(Rc::new(_MRIStepInnerStepper {
        content: RefCell::new(None),
        python: RefCell::new(None),
        ops: RefCell::new(MRIStepInnerStepper_Ops::default()),
        sunctx: RefCell::new(sunctx.clone()),
        forcing: RefCell::new(Vec::new()),
        nforcing: RefCell::new(0),
        nforcing_allocated: RefCell::new(0),
        /* initialize stepper data */
        last_flag: RefCell::new(ARK_SUCCESS),
        tshift: RefCell::new(ZERO),
        tscale: RefCell::new(ZERO),
        vals: RefCell::new(Vec::new()),
        vecs: RefCell::new(Vec::new()),
        lrw1: RefCell::new(0),
        liw1: RefCell::new(0),
        lrw: RefCell::new(0),
        liw: RefCell::new(0),
    }));

    ARK_SUCCESS
}

pub fn MRIStepInnerStepper_CreateFromSUNStepper(
    sunstepper: &SUNStepper,
    stepper: &mut Option<MRIStepInnerStepper>,
) -> i32 {
    let sunctx = sunstepper.sunctx.borrow().clone();
    let retval = MRIStepInnerStepper_Create(&sunctx, stepper);
    if retval != ARK_SUCCESS {
        return retval;
    }

    let this = stepper.clone().expect("stepper");

    let retval = MRIStepInnerStepper_SetContent(&this, Some(Box::new(sunstepper.clone())));
    if retval != ARK_SUCCESS {
        return retval;
    }

    let retval = MRIStepInnerStepper_SetEvolveFn(&this, Some(mriStepInnerStepper_EvolveSUNStepper));
    if retval != ARK_SUCCESS {
        return retval;
    }

    let retval =
        MRIStepInnerStepper_SetFullRhsFn(&this, Some(mriStepInnerStepper_FullRhsSUNStepper));
    if retval != ARK_SUCCESS {
        return retval;
    }

    let retval = MRIStepInnerStepper_SetResetFn(&this, Some(mriStepInnerStepper_ResetSUNStepper));
    if retval != ARK_SUCCESS {
        return retval;
    }

    ARK_SUCCESS
}

/// C `MRIStepInnerStepper_Free(MRIStepInnerStepper* stepper)`.
///
/// Dropping the `Rc` replaces C's `free(ops)` / `free(*stepper)`; storage
/// survives while other clones of the handle do (C would leave those
/// dangling).
pub fn MRIStepInnerStepper_Free(stepper: &mut Option<MRIStepInnerStepper>) -> i32 {
    if stepper.is_none() {
        return ARK_SUCCESS;
    }

    {
        let this = stepper.as_ref().expect("stepper");

        /* free the inner forcing and fused op workspace vector */
        mriStepInnerStepper_FreeVecs(this);

        /* free operations structure: released together with the handle */

        /* free python data (SUNDIALS_ENABLE_PYTHON not built) */
        *this.python.borrow_mut() = None;
    }

    /* free inner stepper mem */
    *stepper = None;

    ARK_SUCCESS
}

/// C `MRIStepInnerStepper_SetContent(stepper, void* content)`; `None` is
/// C's `NULL`.
pub fn MRIStepInnerStepper_SetContent(
    stepper: &MRIStepInnerStepper,
    content: Option<Box<dyn Any>>,
) -> i32 {
    /* C `stepper == NULL` guard is unreachable through `&MRIStepInnerStepper` */
    *stepper.content.borrow_mut() = content;

    ARK_SUCCESS
}

/// C `MRIStepInnerStepper_GetContent(stepper, void** content)`.
///
/// A safe-Rust `Box<dyn Any>` token cannot be aliased, so the stored box is
/// SWAPPED with `content` (deviation class 6, as `SUNStepper_GetContent`):
/// the caller MUST hand it back on every return path before anything else
/// touches the stepper's content. Implementation modules should instead use
/// [`MRIStepInnerStepper_GetContentAs`], which clones the handle exactly as
/// C's pointer copy does.
pub fn MRIStepInnerStepper_GetContent(
    stepper: &MRIStepInnerStepper,
    content: &mut Option<Box<dyn Any>>,
) -> i32 {
    std::mem::swap(&mut *stepper.content.borrow_mut(), content);

    ARK_SUCCESS
}

/// Port-only, borrow-safe companion to [`MRIStepInnerStepper_GetContent`]
/// for the common case where the C `void* content` is a SUNDIALS handle
/// (the ARKODE case: `ARKodeMem`; the SUNStepper case: `SUNStepper`). The
/// stepper keeps its content; nothing has to be handed back. A content type
/// mismatch is C UB (a bad cast) and panics here (deviation class 5).
pub fn MRIStepInnerStepper_GetContentAs<T: Any + Clone>(
    stepper: &MRIStepInnerStepper,
    content: &mut Option<T>,
) -> i32 {
    *content = Some(
        stepper
            .content
            .borrow()
            .as_ref()
            .expect("MRIStepInnerStepper content")
            .downcast_ref::<T>()
            .expect("MRIStepInnerStepper content")
            .clone(),
    );

    ARK_SUCCESS
}

pub fn MRIStepInnerStepper_SetEvolveFn(
    stepper: &MRIStepInnerStepper,
    fn_: Option<MRIStepInnerEvolveFn>,
) -> i32 {
    /* C `stepper == NULL` / `stepper->ops == NULL` guards are unreachable */
    stepper.ops.borrow_mut().evolve = fn_;

    ARK_SUCCESS
}

pub fn MRIStepInnerStepper_SetFullRhsFn(
    stepper: &MRIStepInnerStepper,
    fn_: Option<MRIStepInnerFullRhsFn>,
) -> i32 {
    stepper.ops.borrow_mut().fullrhs = fn_;

    ARK_SUCCESS
}

pub fn MRIStepInnerStepper_SetResetFn(
    stepper: &MRIStepInnerStepper,
    fn_: Option<MRIStepInnerResetFn>,
) -> i32 {
    stepper.ops.borrow_mut().reset = fn_;

    ARK_SUCCESS
}

pub fn MRIStepInnerStepper_SetAccumulatedErrorGetFn(
    stepper: &MRIStepInnerStepper,
    fn_: Option<MRIStepInnerGetAccumulatedError>,
) -> i32 {
    stepper.ops.borrow_mut().geterror = fn_;

    ARK_SUCCESS
}

pub fn MRIStepInnerStepper_SetAccumulatedErrorResetFn(
    stepper: &MRIStepInnerStepper,
    fn_: Option<MRIStepInnerResetAccumulatedError>,
) -> i32 {
    stepper.ops.borrow_mut().reseterror = fn_;

    ARK_SUCCESS
}

pub fn MRIStepInnerStepper_SetRTolFn(
    stepper: &MRIStepInnerStepper,
    fn_: Option<MRIStepInnerSetRTol>,
) -> i32 {
    stepper.ops.borrow_mut().setrtol = fn_;

    ARK_SUCCESS
}

pub fn MRIStepInnerStepper_AddForcing(
    stepper: &MRIStepInnerStepper,
    t: sunrealtype,
    f: &N_Vector,
) -> i32 {
    /* C `stepper == NULL` guard is unreachable through the handle type */

    /* `vals`/`vecs` are rebuilt as locals here (an N_Vector array cannot be
    left uninitialised in safe Rust); the values, the `nvec` argument and
    therefore the arithmetic are identical to C's in-place scratch. */
    let mut vals: Vec<sunrealtype> = Vec::new();
    let mut vecs: Vec<N_Vector> = Vec::new();

    /* always append the constant forcing term */
    vals.push(ONE);
    vecs.push(f.clone());

    /* compute normalized time tau and initialize tau^i */
    let tau = (t - *stepper.tshift.borrow()) / (*stepper.tscale.borrow());
    let mut taui = ONE;

    let nforcing = *stepper.nforcing.borrow();
    for i in 0..nforcing {
        vals.push(taui);
        vecs.push(stepper.forcing.borrow()[i as usize].clone());
        taui *= tau;
    }

    N_VLinearCombination(nforcing + 1, &vals, &vecs, f);

    ARK_SUCCESS
}

/// C `MRIStepInnerStepper_GetForcingData(stepper, tshift, tscale, N_Vector**
/// forcing, nforcing)`. The C out-param hands back the internal array
/// pointer; the port hands back a `Vec` of clones of the same `N_Vector`
/// handles (C pointer copies), so the vectors themselves still alias.
pub fn MRIStepInnerStepper_GetForcingData(
    stepper: &MRIStepInnerStepper,
    tshift: &mut sunrealtype,
    tscale: &mut sunrealtype,
    forcing: &mut Vec<N_Vector>,
    nforcing: &mut i32,
) -> i32 {
    *tshift = *stepper.tshift.borrow();
    *tscale = *stepper.tscale.borrow();
    *forcing = stepper.forcing.borrow().clone();
    *nforcing = *stepper.nforcing.borrow();

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  Internal inner integrator functions
  ---------------------------------------------------------------*/

/* Check for required operations */
pub fn mriStepInnerStepper_HasRequiredOps(stepper: &MRIStepInnerStepper) -> i32 {
    /* C NULL guards on `stepper` and `stepper->ops` are unreachable */

    if stepper.ops.borrow().evolve.is_some() {
        ARK_SUCCESS
    } else {
        ARK_ILL_INPUT
    }
}

/* Check whether stepper supports fast/slow tolerance adaptivity */
pub fn mriStepInnerStepper_SupportsRTolAdaptivity(stepper: &MRIStepInnerStepper) -> sunbooleantype {
    let ops = stepper.ops.borrow();
    if ops.geterror.is_some() && ops.reseterror.is_some() && ops.setrtol.is_some() {
        SUNTRUE
    } else {
        SUNFALSE
    }
}

/* Evolve the inner (fast) ODE */
pub fn mriStepInnerStepper_Evolve(
    stepper: &MRIStepInnerStepper,
    t0: sunrealtype,
    tout: sunrealtype,
    y: &N_Vector,
) -> i32 {
    let evolve = stepper.ops.borrow().evolve;
    if evolve.is_none() {
        return ARK_ILL_INPUT;
    }
    let evolve = evolve.expect("evolve");

    let last_flag = evolve(stepper, t0, tout, y);
    *stepper.last_flag.borrow_mut() = last_flag;

    last_flag
}

pub fn mriStepInnerStepper_EvolveSUNStepper(
    stepper: &MRIStepInnerStepper,
    t0: sunrealtype,
    tout: sunrealtype,
    y: &N_Vector,
) -> i32 {
    let _ = t0; /* SUNDIALS_MAYBE_UNUSED */

    let sunstepper: SUNStepper = stepper
        .content
        .borrow()
        .as_ref()
        .expect("SUNStepper content")
        .downcast_ref::<SUNStepper>()
        .expect("SUNStepper content")
        .clone();
    let mut tret: sunrealtype = ZERO;

    let (tshift, tscale, forcing, nforcing) = (
        *stepper.tshift.borrow(),
        *stepper.tscale.borrow(),
        stepper.forcing.borrow().clone(),
        *stepper.nforcing.borrow(),
    );
    let mut err = SUNStepper_SetForcing(&sunstepper, tshift, tscale, &forcing, nforcing);
    *stepper.last_flag.borrow_mut() = *sunstepper.last_flag.borrow();
    if err != SUN_SUCCESS {
        return ARK_SUNSTEPPER_ERR;
    }

    err = SUNStepper_SetStopTime(&sunstepper, tout);
    *stepper.last_flag.borrow_mut() = *sunstepper.last_flag.borrow();
    if err != SUN_SUCCESS {
        return ARK_SUNSTEPPER_ERR;
    }

    err = SUNStepper_Evolve(&sunstepper, tout, y, &mut tret);
    *stepper.last_flag.borrow_mut() = *sunstepper.last_flag.borrow();
    if err != SUN_SUCCESS {
        return ARK_SUNSTEPPER_ERR;
    }

    err = SUNStepper_SetForcing(&sunstepper, ZERO, ONE, &[], 0);
    *stepper.last_flag.borrow_mut() = *sunstepper.last_flag.borrow();
    if err != SUN_SUCCESS {
        return ARK_SUNSTEPPER_ERR;
    }

    ARK_SUCCESS
}

/* Compute the full RHS for inner (fast) time scale TODO(DJG): This function can
   be made optional when fullrhs is not called unconditionally by the ARKODE
   infrastructure e.g., in arkInitialSetup, arkYddNorm, and arkCompleteStep. */
pub fn mriStepInnerStepper_FullRhs(
    stepper: &MRIStepInnerStepper,
    t: sunrealtype,
    y: &N_Vector,
    f: &N_Vector,
    mode: i32,
) -> i32 {
    let fullrhs = stepper.ops.borrow().fullrhs;
    if fullrhs.is_none() {
        return ARK_ILL_INPUT;
    }
    let fullrhs = fullrhs.expect("fullrhs");

    let last_flag = fullrhs(stepper, t, y, f, mode);
    *stepper.last_flag.borrow_mut() = last_flag;
    last_flag
}

pub fn mriStepInnerStepper_FullRhsSUNStepper(
    stepper: &MRIStepInnerStepper,
    t: sunrealtype,
    y: &N_Vector,
    f: &N_Vector,
    ark_mode: i32,
) -> i32 {
    let sunstepper: SUNStepper = stepper
        .content
        .borrow()
        .as_ref()
        .expect("SUNStepper content")
        .downcast_ref::<SUNStepper>()
        .expect("SUNStepper content")
        .clone();

    let mode: SUNFullRhsMode = match ark_mode {
        ARK_FULLRHS_START => SUN_FULLRHS_START,
        ARK_FULLRHS_END => SUN_FULLRHS_END,
        _ => SUN_FULLRHS_OTHER,
    };

    let err = SUNStepper_FullRhs(&sunstepper, t, y, f, mode);
    *stepper.last_flag.borrow_mut() = *sunstepper.last_flag.borrow();
    if err != SUN_SUCCESS {
        return ARK_SUNSTEPPER_ERR;
    }
    ARK_SUCCESS
}

/* Reset the inner (fast) stepper state */
pub fn mriStepInnerStepper_Reset(
    stepper: &MRIStepInnerStepper,
    tR: sunrealtype,
    yR: &N_Vector,
) -> i32 {
    let reset = stepper.ops.borrow().reset;

    if let Some(reset) = reset {
        let last_flag = reset(stepper, tR, yR);
        *stepper.last_flag.borrow_mut() = last_flag;
        last_flag
    } else {
        /* assume stepper uses input state and does not need to be reset */
        ARK_SUCCESS
    }
}

/* Gets the inner (fast) stepper accumulated error */
pub fn mriStepInnerStepper_GetAccumulatedError(
    stepper: &MRIStepInnerStepper,
    accum_error: &mut sunrealtype,
) -> i32 {
    let geterror = stepper.ops.borrow().geterror;

    if let Some(geterror) = geterror {
        let last_flag = geterror(stepper, accum_error);
        *stepper.last_flag.borrow_mut() = last_flag;
        last_flag
    } else {
        ARK_INNERSTEP_FAIL
    }
}

/* Resets the inner (fast) stepper accumulated error */
pub fn mriStepInnerStepper_ResetAccumulatedError(stepper: &MRIStepInnerStepper) -> i32 {
    /* NOTE: upstream tests `ops->geterror` here but calls `ops->reseterror`;
    the quirk is preserved (a set geterror with an unset reseterror is a NULL
    call in C and a panic here -- deviation class 5). */
    let (geterror, reseterror) = {
        let ops = stepper.ops.borrow();
        (ops.geterror, ops.reseterror)
    };

    if geterror.is_some() {
        let last_flag = reseterror.expect("reseterror")(stepper);
        *stepper.last_flag.borrow_mut() = last_flag;
        last_flag
    } else {
        /* assume stepper provides exact solution and needs no reset */
        ARK_SUCCESS
    }
}

/* Sets the inner (fast) stepper relative tolerance scaling factor */
pub fn mriStepInnerStepper_SetRTol(stepper: &MRIStepInnerStepper, rtol: sunrealtype) -> i32 {
    let setrtol = stepper.ops.borrow().setrtol;

    if let Some(setrtol) = setrtol {
        let last_flag = setrtol(stepper, rtol);
        *stepper.last_flag.borrow_mut() = last_flag;
        last_flag
    } else {
        /* assume stepper provides exact solution */
        ARK_SUCCESS
    }
}

pub fn mriStepInnerStepper_ResetSUNStepper(
    stepper: &MRIStepInnerStepper,
    tR: sunrealtype,
    yR: &N_Vector,
) -> i32 {
    let sunstepper: SUNStepper = stepper
        .content
        .borrow()
        .as_ref()
        .expect("SUNStepper content")
        .downcast_ref::<SUNStepper>()
        .expect("SUNStepper content")
        .clone();
    let err = SUNStepper_Reset(&sunstepper, tR, yR);
    *stepper.last_flag.borrow_mut() = *sunstepper.last_flag.borrow();
    if err != SUN_SUCCESS {
        return ARK_SUNSTEPPER_ERR;
    }
    ARK_SUCCESS
}

/* Allocate MRI forcing and fused op workspace vectors if necessary */
pub fn mriStepInnerStepper_AllocVecs(
    stepper: &MRIStepInnerStepper,
    count: i32,
    tmpl: &N_Vector,
) -> i32 {
    let mut lrw1: sunindextype = 0;
    let mut liw1: sunindextype = 0;

    /* Set space requirements for one N_Vector */
    let has_nvspace = tmpl.ops.borrow().nvspace.is_some();
    if has_nvspace {
        N_VSpace(tmpl, &mut lrw1, &mut liw1);
    } else {
        lrw1 = 0;
        liw1 = 0;
    }
    *stepper.lrw1.borrow_mut() = lrw1;
    *stepper.liw1.borrow_mut() = liw1;

    /* Set the number of forcing vectors and allocate vectors */
    *stepper.nforcing.borrow_mut() = count;

    let nforcing_allocated = *stepper.nforcing_allocated.borrow();
    let nforcing = *stepper.nforcing.borrow();
    if nforcing_allocated < nforcing {
        let mut forcing = std::mem::take(&mut *stepper.forcing.borrow_mut());
        let mut lrw = *stepper.lrw.borrow();
        let mut liw = *stepper.liw.borrow();
        if nforcing_allocated != 0 {
            arkFreeVecArray(
                nforcing_allocated,
                &mut forcing,
                lrw1,
                &mut lrw,
                liw1,
                &mut liw,
            );
        }
        let ok = arkAllocVecArray(nforcing, tmpl, &mut forcing, lrw1, &mut lrw, liw1, &mut liw);
        *stepper.forcing.borrow_mut() = forcing;
        *stepper.lrw.borrow_mut() = lrw;
        *stepper.liw.borrow_mut() = liw;
        if !ok {
            mriStepInnerStepper_FreeVecs(stepper);
            return ARK_MEM_FAIL;
        }
        *stepper.nforcing_allocated.borrow_mut() = nforcing;
    }

    /* Allocate fused operation workspace arrays. `vecs` is N_Vector handle
    scratch that MRIStepInnerStepper_AddForcing rebuilds on demand (an
    N_Vector array cannot be left uninitialised in safe Rust), so only
    `vals` is materialised; the C NULL-return failure branches are
    unreachable because Vec allocation aborts rather than returning NULL. */
    let vals_empty = stepper.vals.borrow().is_empty();
    if vals_empty {
        *stepper.vals.borrow_mut() = vec![ZERO; (count + 1) as usize];
    }

    ARK_SUCCESS
}

/* Resize MRI forcing and fused op workspace vectors if necessary */
pub fn mriStepInnerStepper_Resize(
    stepper: &MRIStepInnerStepper,
    resize: Option<ARKVecResizeFn>,
    resize_data: &mut Option<Box<dyn Any>>,
    lrw_diff: sunindextype,
    liw_diff: sunindextype,
    tmpl: &N_Vector,
) -> i32 {
    let nforcing_allocated = *stepper.nforcing_allocated.borrow();
    let mut forcing = std::mem::take(&mut *stepper.forcing.borrow_mut());
    let mut lrw = *stepper.lrw.borrow();
    let mut liw = *stepper.liw.borrow();

    let ok = arkResizeVecArray(
        resize,
        resize_data,
        nforcing_allocated,
        tmpl,
        &mut forcing,
        lrw_diff,
        &mut lrw,
        liw_diff,
        &mut liw,
    );

    *stepper.forcing.borrow_mut() = forcing;
    *stepper.lrw.borrow_mut() = lrw;
    *stepper.liw.borrow_mut() = liw;

    if !ok {
        return ARK_MEM_FAIL;
    }

    ARK_SUCCESS
}

/* Free MRI forcing and fused op workspace vectors if necessary */
pub fn mriStepInnerStepper_FreeVecs(stepper: &MRIStepInnerStepper) -> i32 {
    let nforcing_allocated = *stepper.nforcing_allocated.borrow();
    let lrw1 = *stepper.lrw1.borrow();
    let liw1 = *stepper.liw1.borrow();
    let mut forcing = std::mem::take(&mut *stepper.forcing.borrow_mut());
    let mut lrw = *stepper.lrw.borrow();
    let mut liw = *stepper.liw.borrow();

    arkFreeVecArray(
        nforcing_allocated,
        &mut forcing,
        lrw1,
        &mut lrw,
        liw1,
        &mut liw,
    );

    *stepper.forcing.borrow_mut() = forcing;
    *stepper.lrw.borrow_mut() = lrw;
    *stepper.liw.borrow_mut() = liw;

    let vecs_alloc = !stepper.vecs.borrow().is_empty();
    if vecs_alloc {
        *stepper.vecs.borrow_mut() = Vec::new();
    }

    let vals_alloc = !stepper.vals.borrow().is_empty();
    if vals_alloc {
        *stepper.vals.borrow_mut() = Vec::new();
    }

    ARK_SUCCESS
}

/* Print forcing vectors to output file */
pub fn mriStepInnerStepper_PrintMem(stepper: &MRIStepInnerStepper, outfile: &SUNFile) {
    /* output data from the inner stepper */
    outfile.write_str("MRIStepInnerStepper Mem:\n");
    outfile.write_str(&format!(
        "MRIStepInnerStepper: inner_nforcing = {}\n",
        *stepper.nforcing.borrow()
    ));
}

/*---------------------------------------------------------------
  Utility routines for MRIStep to serve as an MRIStepInnerStepper
  ---------------------------------------------------------------*/

/*------------------------------------------------------------------------------
  mriStep_ApplyForcing

  Determines the linear combination coefficients and vectors to apply forcing
  at a given value of the independent variable (t).  This occurs through
  appending coefficients and N_Vector pointers to the underlying cvals and Xvecs
  arrays in the step_mem structure.  The dereferenced input *nvec should indicate
  the next available entry in the cvals/Xvecs arrays.  The input 's' is a
  scaling factor that should be applied to each of these coefficients.

  Port note: the C `N_Vector*` workspace `step_mem->Xvecs` is
  `Vec<Option<N_Vector>>` (a `calloc`'d NULL slot is `None`); the filled
  prefix is materialised for the `N_V*` kernels by `mriStep_xvecs`.
  ----------------------------------------------------------------------------*/
pub fn mriStep_ApplyForcing(
    step_mem: &mut ARKodeMRIStepMemRec,
    t: sunrealtype,
    s: sunrealtype,
    nvec: &mut i32,
) {
    /* always append the constant forcing term */
    step_mem.cvals[*nvec as usize] = s;
    step_mem.Xvecs[*nvec as usize] = Some(step_mem.forcing[0].clone());
    *nvec += 1;

    /* compute normalized time tau and initialize tau^i */
    let tau = (t - step_mem.tshift) / (step_mem.tscale);
    let mut taui = tau;
    for i in 1..step_mem.nforcing {
        step_mem.cvals[*nvec as usize] = s * taui;
        step_mem.Xvecs[*nvec as usize] = Some(step_mem.forcing[i as usize].clone());
        taui *= tau;
        *nvec += 1;
    }
}

/*------------------------------------------------------------------------------
  mriStep_SetInnerForcing

  Sets an array of coefficient vectors for a time-dependent external polynomial
  forcing term in the ODE RHS i.e., y' = f(t,y) + p(t). This function is
  primarily intended for using MRIStep as an inner integrator within another
  [outer] instance of MRIStep, where this instance is used to solve a
  modified ODE at a fast time scale. The polynomial is of the form

  p(t) = sum_{i = 0}^{nvecs - 1} forcing[i] * ((t - tshift) / (tscale))^i

  where tshift and tscale are used to normalize the time t (e.g., with MRIGARK
  methods).
  ----------------------------------------------------------------------------*/
pub fn mriStep_SetInnerForcing(
    ark_mem: &ARKodeMem,
    tshift: sunrealtype,
    tscale: sunrealtype,
    forcing: &[N_Vector],
    nvecs: i32,
) -> i32 {
    /* access ARKodeMRIStepMem structure */
    let retval = mriStep_AccessStepMem(ark_mem, "mriStep_SetInnerForcing");
    if retval != ARK_SUCCESS {
        return retval;
    }

    if nvecs > 0 {
        /* enable forcing, and signal that the corresponding pre-existing RHS
           vector is no longer current, since it has a stale forcing function */
        {
            let mut step_mem = mriStep_mem_mut(ark_mem);
            if step_mem.explicit_rhs {
                step_mem.expforcing = SUNTRUE;
                step_mem.impforcing = SUNFALSE;
                step_mem.fse_is_current = SUNFALSE;
            } else {
                step_mem.expforcing = SUNFALSE;
                step_mem.impforcing = SUNTRUE;
                step_mem.fsi_is_current = SUNFALSE;
            }
            step_mem.tshift = tshift;
            step_mem.tscale = tscale;
            step_mem.forcing = forcing.to_vec();
            step_mem.nforcing = nvecs;
        }

        /* Signal that any pre-existing RHS vector is no longer current, since it
           has a stale forcing function */
        ark_mem.borrow_mut().fn_is_current = SUNFALSE;

        /* If the coupling table is NULL, then mriStep_Init has not been called and
           the number of stages has not been set yet. In this case, the workspace
           arrays for fused vector operations will be re-allocated in mriStep_Init
           if necessary to account the value of nforcing. On subsequent calls we
           check if enough space has already been allocated in case nforcing has
           increased since the original allocation. */
        let mric = mriStep_mem_mut(ark_mem).MRIC.clone();
        if let Some(mric) = mric {
            let mric_stages = mric.borrow().stages;

            /* check if there are enough reusable arrays for fused operations */
            let (nfusedopvecs, have_cvals, have_Xvecs) = {
                let step_mem = mriStep_mem_mut(ark_mem);
                /* empty `Vec` == C `NULL` for both workspace arrays */
                (
                    step_mem.nfusedopvecs,
                    !step_mem.cvals.is_empty(),
                    !step_mem.Xvecs.is_empty(),
                )
            };
            if (nfusedopvecs - nvecs) < (2 * mric_stages + 2) {
                /* free current work space */
                if have_cvals {
                    mriStep_mem_mut(ark_mem).cvals = Vec::new();
                    ark_mem.borrow_mut().lrw -= nfusedopvecs as i64;
                }
                if have_Xvecs {
                    mriStep_mem_mut(ark_mem).Xvecs = Vec::new();
                    ark_mem.borrow_mut().liw -= nfusedopvecs as i64;
                }

                /* allocate reusable arrays for fused vector operations */
                let new_nfusedopvecs = 2 * mric_stages + 2 + nvecs;
                {
                    let mut step_mem = mriStep_mem_mut(ark_mem);
                    step_mem.nfusedopvecs = new_nfusedopvecs;
                    step_mem.cvals = vec![ZERO; new_nfusedopvecs as usize];
                }
                ark_mem.borrow_mut().lrw += new_nfusedopvecs as i64;
                {
                    let mut step_mem = mriStep_mem_mut(ark_mem);
                    step_mem.Xvecs = vec![None; new_nfusedopvecs as usize];
                }
                ark_mem.borrow_mut().liw += new_nfusedopvecs as i64;
            }
        }
    } else {
        /* disable forcing */
        let mut step_mem = mriStep_mem_mut(ark_mem);
        step_mem.expforcing = SUNFALSE;
        step_mem.impforcing = SUNFALSE;
        step_mem.tshift = ZERO;
        step_mem.tscale = ONE;
        step_mem.forcing = Vec::new();
        step_mem.nforcing = 0;
    }

    ARK_SUCCESS
}
