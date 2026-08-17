//! Port of `src/arkode/arkode_impl.h` + the constants/typedefs of
//! `include/arkode/arkode.h`, plus `src/arkode/arkode_adapt_impl.h`,
//! `src/arkode/arkode_root_impl.h` and `src/arkode/arkode_relaxation_impl.h`
//! (folded here because `arkode_impl.h` `#include`s all three and
//! `ARKodeMemRec` embeds their records — the same treatment
//! `cvode_proj_impl.h` gets inside `cvode_impl.rs`).
//!
//! This is the FROZEN SHARED CONTRACT for the whole `arkode_rs` crate.
//! Every time-stepper module (ARKStep, ERKStep, MRIStep, LSRKStep,
//! SPRKStep, SplittingStep, ForcingStep) defines its own
//! `ARKode<X>StepMemRec` content struct in its own file and plugs into the
//! `step_*` function-pointer table below. Nothing stepper-specific lives
//! here.
//!
//! `arkProcessError` (defined in `arkode.c` upstream) is relocated here so
//! every arkode module shares one definition; C varargs map to a
//! pre-formatted `msg` (call sites use the `MSG_ARK_*` constants/builders
//! below). Parameterized messages are functions producing the exact C
//! `printf` expansion (`SUN_FORMAT_G` = `%.15g` via `sun_format_g`).
//!
//! Handle model: `ARKodeMem = Rc<RefCell<ARKodeMemRec>>`. Internal
//! functions take `&ARKodeMem` and use granular borrows (never hold a
//! borrow across a callback, an N_Vector op on user vectors, a stepper
//! `step_*` call, or a linear/nonlinear solver call — all can re-enter the
//! mem).
//!
//! Naming deviations forced by Rust (there are exactly two):
//!  * C `ark_mem->fn` (the full-RHS vector) is `ark_mem.fn_` — `fn` is a
//!    Rust keyword.
//!  * The ARKLS system/mass records live in `ark_mem.ark_lmem` /
//!    `ark_mem.ark_mass_mem` instead of inside the active stepper's
//!    `step_mem` (see the field docs).

use std::any::Any;
use std::cell::{Cell, RefCell};
use std::rc::Rc;

use sundials_core::sundials_adaptcontroller::SUNAdaptController;
use sundials_core::sundials_adjointcheckpointscheme::SUNAdjointCheckpointScheme;
use sundials_core::sundials_context::{SUNContext, SUNContext_GetLastError};
use sundials_core::sundials_errors::{SUNGlobalFallbackErrHandler, SUNHandleErrWithMsg};
use sundials_core::sundials_linearsolver::SUNLinearSolver_Type;
use sundials_core::sundials_logger::{SUNLogger_QueueMsg, SUN_LOGLEVEL_WARNING};
use sundials_core::sundials_nonlinearsolver::SUNNonlinearSolver;
use sundials_core::sundials_nvector::N_Vector;
use sundials_core::sundials_types::*;
use sundials_core::sundials_utils::{sun_format_g, sunCombineFileAndLine, SUNFile};

/* =================================================================
 * Public constants (include/arkode/arkode.h)
 * =================================================================*/

/* usage modes (itask) */
pub const ARK_NORMAL: i32 = 1;
pub const ARK_ONE_STEP: i32 = 2;

/* adaptivity module flags */
pub const ARK_ADAPT_CUSTOM: i32 = -1;
pub const ARK_ADAPT_PID: i32 = 0;
pub const ARK_ADAPT_PI: i32 = 1;
pub const ARK_ADAPT_I: i32 = 2;
pub const ARK_ADAPT_EXP_GUS: i32 = 3;
pub const ARK_ADAPT_IMP_GUS: i32 = 4;
pub const ARK_ADAPT_IMEX_GUS: i32 = 5;

/* Constants for evaluating the full RHS */
pub const ARK_FULLRHS_START: i32 = 0;
pub const ARK_FULLRHS_END: i32 = 1;
pub const ARK_FULLRHS_OTHER: i32 = 2;

/* interpolation module flags */

/*    max allowed degree */
pub const ARK_INTERP_MAX_DEGREE: i32 = 5;

/*    interpolation module types */
pub const ARK_INTERP_NONE: i32 = -1;
pub const ARK_INTERP_HERMITE: i32 = 0;
pub const ARK_INTERP_LAGRANGE: i32 = 1;

/* return values */

pub const ARK_SUCCESS: i32 = 0;
pub const ARK_TSTOP_RETURN: i32 = 1;
pub const ARK_ROOT_RETURN: i32 = 2;

pub const ARK_WARNING: i32 = 99;

pub const ARK_TOO_MUCH_WORK: i32 = -1;
pub const ARK_TOO_MUCH_ACC: i32 = -2;
pub const ARK_ERR_FAILURE: i32 = -3;
pub const ARK_CONV_FAILURE: i32 = -4;

pub const ARK_LINIT_FAIL: i32 = -5;
pub const ARK_LSETUP_FAIL: i32 = -6;
pub const ARK_LSOLVE_FAIL: i32 = -7;
pub const ARK_RHSFUNC_FAIL: i32 = -8;
pub const ARK_FIRST_RHSFUNC_ERR: i32 = -9;
pub const ARK_REPTD_RHSFUNC_ERR: i32 = -10;
pub const ARK_UNREC_RHSFUNC_ERR: i32 = -11;
pub const ARK_RTFUNC_FAIL: i32 = -12;
pub const ARK_LFREE_FAIL: i32 = -13;
pub const ARK_MASSINIT_FAIL: i32 = -14;
pub const ARK_MASSSETUP_FAIL: i32 = -15;
pub const ARK_MASSSOLVE_FAIL: i32 = -16;
pub const ARK_MASSFREE_FAIL: i32 = -17;
pub const ARK_MASSMULT_FAIL: i32 = -18;

pub const ARK_CONSTR_FAIL: i32 = -19;
pub const ARK_MEM_FAIL: i32 = -20;
pub const ARK_MEM_NULL: i32 = -21;
pub const ARK_ILL_INPUT: i32 = -22;
pub const ARK_NO_MALLOC: i32 = -23;
pub const ARK_BAD_K: i32 = -24;
pub const ARK_BAD_T: i32 = -25;
pub const ARK_BAD_DKY: i32 = -26;
pub const ARK_TOO_CLOSE: i32 = -27;

pub const ARK_VECTOROP_ERR: i32 = -28;

pub const ARK_NLS_INIT_FAIL: i32 = -29;
pub const ARK_NLS_SETUP_FAIL: i32 = -30;
pub const ARK_NLS_SETUP_RECVR: i32 = -31;
pub const ARK_NLS_OP_ERR: i32 = -32;

pub const ARK_INNERSTEP_ATTACH_ERR: i32 = -33;
pub const ARK_INNERSTEP_FAIL: i32 = -34;
pub const ARK_OUTERTOINNER_FAIL: i32 = -35;
pub const ARK_INNERTOOUTER_FAIL: i32 = -36;

/* ARK_POSTPROCESS_FAIL equals ARK_POSTPROCESS_STEP_FAIL
   for backwards compatibility. */
pub const ARK_POSTPROCESS_FAIL: i32 = -37;
pub const ARK_POSTPROCESS_STEP_FAIL: i32 = -37;
pub const ARK_POSTPROCESS_STAGE_FAIL: i32 = -38;
pub const ARK_PRESTEPFN_FAIL: i32 = -39;
pub const ARK_POSTSTEPFN_FAIL: i32 = -40;
pub const ARK_PRERHSFN_FAIL: i32 = -41;

pub const ARK_USER_PREDICT_FAIL: i32 = -42;
pub const ARK_INTERP_FAIL: i32 = -43;

pub const ARK_INVALID_TABLE: i32 = -44;

pub const ARK_CONTEXT_ERR: i32 = -45;

pub const ARK_RELAX_FAIL: i32 = -46;
pub const ARK_RELAX_MEM_NULL: i32 = -47;
pub const ARK_RELAX_FUNC_FAIL: i32 = -48;
pub const ARK_RELAX_JAC_FAIL: i32 = -49;

pub const ARK_CONTROLLER_ERR: i32 = -50;

pub const ARK_STEPPER_UNSUPPORTED: i32 = -51;

pub const ARK_DOMEIG_FAIL: i32 = -52;
pub const ARK_MAX_STAGE_LIMIT_FAIL: i32 = -53;

pub const ARK_SUNSTEPPER_ERR: i32 = -54;
pub const ARK_STEP_DIRECTION_ERR: i32 = -55;

pub const ARK_ADJ_CHECKPOINT_FAIL: i32 = -56;
pub const ARK_ADJ_RECOMPUTE_FAIL: i32 = -57;
pub const ARK_SUNADJSTEPPER_ERR: i32 = -58;

pub const ARK_DEE_FAIL: i32 = -59;

pub const ARK_STEP_H0_FAIL: i32 = -60;

pub const ARK_UNRECOGNIZED_ERROR: i32 = -99;

/* ------------------------------
 * User-Supplied Function Types
 * ------------------------------
 *
 * C `void* user_data` maps to `&mut Option<Box<dyn Any>>`: the integrator
 * `Option::take`s the box out of the mem record around each callback
 * invocation, so the callback gets exclusive access without re-borrowing
 * the mem. Never change one of these signatures without updating every
 * example in the workspace. */

pub type ARKRhsFn =
    fn(t: sunrealtype, y: &N_Vector, ydot: &N_Vector, user_data: &mut Option<Box<dyn Any>>) -> i32;

pub type ARKRootFn = fn(
    t: sunrealtype,
    y: &N_Vector,
    gout: &mut [sunrealtype],
    user_data: &mut Option<Box<dyn Any>>,
) -> i32;

pub type ARKEwtFn = fn(y: &N_Vector, ewt: &N_Vector, user_data: &mut Option<Box<dyn Any>>) -> i32;

pub type ARKRwtFn = fn(y: &N_Vector, rwt: &N_Vector, user_data: &mut Option<Box<dyn Any>>) -> i32;

pub type ARKAdaptFn = fn(
    y: &N_Vector,
    t: sunrealtype,
    h1: sunrealtype,
    h2: sunrealtype,
    h3: sunrealtype,
    e1: sunrealtype,
    e2: sunrealtype,
    e3: sunrealtype,
    q: i32,
    p: i32,
    hnew: &mut sunrealtype,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32;

pub type ARKExpStabFn = fn(
    y: &N_Vector,
    t: sunrealtype,
    hstab: &mut sunrealtype,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32;

pub type ARKVecResizeFn = fn(
    y: &N_Vector,
    ytemplate: &N_Vector,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32;

pub type ARKPreStepFn = fn(
    t: sunrealtype,
    y: &N_Vector,
    step: i64,
    attempt: i32,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32;

pub type ARKPostStepFn =
    fn(t: sunrealtype, y: &N_Vector, step: i64, user_data: &mut Option<Box<dyn Any>>) -> i32;

pub type ARKPostProcessFn =
    fn(t: sunrealtype, y: &N_Vector, user_data: &mut Option<Box<dyn Any>>) -> i32;

pub type ARKPreRhsFn =
    fn(t: sunrealtype, y: &N_Vector, user_data: &mut Option<Box<dyn Any>>) -> i32;

pub type ARKStagePredictFn =
    fn(t: sunrealtype, zpred: &N_Vector, user_data: &mut Option<Box<dyn Any>>) -> i32;

pub type ARKRelaxFn =
    fn(y: &N_Vector, r: &mut sunrealtype, user_data: &mut Option<Box<dyn Any>>) -> i32;

pub type ARKRelaxJacFn =
    fn(y: &N_Vector, J: &N_Vector, user_data: &mut Option<Box<dyn Any>>) -> i32;

/* --------------------------
 * Relaxation Solver Options
 * -------------------------- */

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ARKRelaxSolver {
    #[default]
    ARK_RELAX_BRENT,
    ARK_RELAX_NEWTON,
}
pub use ARKRelaxSolver::*;

/* --------------------------
 * Error Accumulation Options
 * -------------------------- */

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ARKAccumError {
    #[default]
    ARK_ACCUMERROR_NONE,
    ARK_ACCUMERROR_MAX,
    ARK_ACCUMERROR_SUM,
    ARK_ACCUMERROR_AVG,
}
pub use ARKAccumError::*;

/* =================================================================
 * ARKODE Private Constants (arkode_impl.h)
 * =================================================================*/

/* Basic ARKODE defaults */
/*   method order */
pub const Q_DEFAULT: i32 = 4;
/*   max steps between returns */
pub const MXSTEP_DEFAULT: i64 = 500;
/*   max number of error failures */
pub const MAXNEF: i32 = 7;
/*   max number of convergence failures */
pub const MAXNCF: i32 = 10;
/*   max number of constraint failures */
pub const MAXCONSTRFAILS: i32 = 10;
/*   max number of t+h==h warnings */
pub const MXHNIL: i32 = 10;
/*   max number of attempts to recover in DQ J*v */
pub const MAX_DQITERS: i32 = 3;

/* Numeric constants */
pub const ZERO: sunrealtype = 0.0;
pub const TINY: sunrealtype = 1.0e-10;
pub const TENTH: sunrealtype = 0.1;
pub const HALF: sunrealtype = 0.5;
pub const ONE: sunrealtype = 1.0;
pub const TWO: sunrealtype = 2.0;
pub const THREE: sunrealtype = 3.0;
pub const FOUR: sunrealtype = 4.0;
pub const FIVE: sunrealtype = 5.0;

/* Control constants for tolerances */
pub const ARK_SS: i32 = 0;
pub const ARK_SV: i32 = 1;
pub const ARK_WF: i32 = 2;

/*---------------------------------------------------------------
  Initialization types
  ---------------------------------------------------------------*/
pub const FIRST_INIT: i32 = 0; /* first step (re-)initialization */
pub const RESET_INIT: i32 = 1; /* reset initialization           */
pub const RESIZE_INIT: i32 = 2; /* resize initialization          */

/*---------------------------------------------------------------
  Control constants for lower-level time-stepping functions
  ---------------------------------------------------------------*/
pub const PREDICT_AGAIN: i32 = 3;
pub const CONV_FAIL: i32 = 4;
pub const TRY_AGAIN: i32 = 5;
pub const FIRST_CALL: i32 = 6;
pub const PREV_CONV_FAIL: i32 = 7;
pub const PREV_ERR_FAIL: i32 = 8;
pub const RHSFUNC_RECVR: i32 = 9;
pub const CONSTR_RECVR: i32 = 10;
pub const ARK_RETRY_STEP: i32 = 11;

/*---------------------------------------------------------------
  Return values for lower-level rootfinding functions
  ---------------------------------------------------------------*/
pub const RTFOUND: i32 = 1;
pub const CLOSERT: i32 = 3;

/*---------------------------------------------------------------
  Algorithmic constants
  ---------------------------------------------------------------
  ARKodeGetDky and arkStep:  FUZZ_FACTOR

  arkHin:  H0_LBFACTOR, H0_UBFACTOR, H0_BIAS and H0_ITERS

  time comparison factors:
     ONEPSM      safety factor for floating point comparisons
     ONEMSM      safety factor for floating point comparisons
  ---------------------------------------------------------------*/
pub const FUZZ_FACTOR: sunrealtype = 100.0;

pub const H0_LBFACTOR: sunrealtype = 100.0;
pub const H0_UBFACTOR: sunrealtype = 0.1;
pub const H0_BIAS: sunrealtype = HALF;
pub const H0_ITERS: i32 = 4;

pub const ONEPSM: sunrealtype = 1.000001;
pub const ONEMSM: sunrealtype = 0.999999;

/*---------------------------------------------------------------
  Input flag to linear solver setup routine:  CONVFAIL
  --------------------------------------------------------------*/
pub const ARK_NO_FAILURES: i32 = 0;
pub const ARK_FAIL_BAD_J: i32 = 1;
pub const ARK_FAIL_OTHER: i32 = 2;

/* =================================================================
 * Implicit-solver constants shared by ARKStep and MRIStep
 * (`arkode_arkstep_impl.h`, duplicated verbatim in
 * `arkode_mristep_impl.h`). Hoisted here so all split parts of each
 * stepper — arkode_arkstep{,_io,_nls}.rs and
 * arkode_mristep{,_io,_nls,_controller}.rs — share one definition.
 * =================================================================*/

/* max number of nonlinear iterations */
pub const MAXCOR: i32 = 3;
/* constant to estimate the convergence rate for the nonlinear equation */
pub const CRDOWN: sunrealtype = 0.3;
/* if |gamma/gammap-1| > DGMAX then call lsetup */
pub const DGMAX: sunrealtype = 0.2;
/* declare divergence if ratio delnrm/delnrm_p > RDIV */
pub const RDIV: sunrealtype = 2.3;
/* max no. of steps between lsetup calls */
pub const MSBP: i32 = 20;
/* Default solver tolerance factor */
pub const NLSCOEF: sunrealtype = 0.1;

/* =================================================================
 * ARKODE Time Step Adaptivity constants (arkode_adapt_impl.h)
 * =================================================================*/

/* size constants for the adaptivity memory structure */
pub const ARK_ADAPT_LRW: i64 = 10;
pub const ARK_ADAPT_LIW: i64 = 7; /* includes function/data pointers */

/* Time step controller default values */
pub const CFLFAC: sunrealtype = 0.5;
pub const SAFETY: sunrealtype = 0.9; /* CVODE uses 1.0  */
pub const GROWTH: sunrealtype = 20.0; /* CVODE uses 10.0 */
pub const HFIXED_LB: sunrealtype = 1.0; /* CVODE uses 1.0  */
pub const HFIXED_UB: sunrealtype = 1.0; /* CVODE uses 1.5  */

/* maximum step size change on first step */
pub const ETAMX1: sunrealtype = 10000.0;
/* step size reduction factor on multiple error test failures
   (multiple implies >= SMALL_NEF) */
pub const ETAMXF: sunrealtype = 0.3;
/* smallest allowable step size reduction factor on an error test failure */
pub const ETAMIN: sunrealtype = 0.1;
/* step size reduction factor on nonlinear convergence failure */
pub const ETACF: sunrealtype = 0.25;
/* if an error failure occurs and SMALL_NEF <= nef, then reset
   eta = MIN(eta, ETAMXF) */
pub const SMALL_NEF: i32 = 2;
/* order to use for controller:
     0=embedding,
     1=method,
     otherwise min(method,embedding) */
pub const PQ: i32 = 0;
/* adjustment to apply within controller to method order of accuracy */
pub const ADJUST: i32 = 0;

/* =================================================================
 * ARKODE Root-finding constants (arkode_root_impl.h)
 * =================================================================*/

pub const ARK_ROOT_LRW: i64 = 5;
pub const ARK_ROOT_LIW: i64 = 12;

/* Numeric constants */
pub const HUND: sunrealtype = 100.0;

/* =================================================================
 * Relaxation constants (arkode_relaxation_impl.h)
 * =================================================================*/

pub const ARK_RELAX_DEFAULT_MAX_FAILS: i32 = 10;
pub const ARK_RELAX_DEFAULT_RES_TOL: sunrealtype = 10.0 * SUN_UNIT_ROUNDOFF;
pub const ARK_RELAX_DEFAULT_REL_TOL: sunrealtype = 4.0 * SUN_UNIT_ROUNDOFF;
pub const ARK_RELAX_DEFAULT_ABS_TOL: sunrealtype = 1.0e-14;
pub const ARK_RELAX_DEFAULT_MAX_ITERS: i32 = 10;
pub const ARK_RELAX_DEFAULT_LOWER_BOUND: sunrealtype = 0.8;
pub const ARK_RELAX_DEFAULT_UPPER_BOUND: sunrealtype = 1.2;
pub const ARK_RELAX_DEFAULT_ETA_FAIL: sunrealtype = 0.25;

/* Relaxation private return values (public values live in arkode.h) */
pub const ARK_RELAX_FUNC_RECV: i32 = 1;
pub const ARK_RELAX_JAC_RECV: i32 = 2;
pub const ARK_RELAX_SOLVE_RECV: i32 = 3;

/* =================================================================
 * ARKODE Interface function definitions (arkode_impl.h)
 * =================================================================*/

/// C `sunbooleantype*` alias into a stepper's internal `jcur` flag.
///
/// `step_getgammas` hands out `&step_mem->jcur`, and `arkLsSetup` is
/// handed the *same* address as `jcurPtr`; a user preconditioner setup
/// routine reached re-entrantly through `SUNLinSolSetup` -> `arkLsPSetup`
/// writes through it and `arkLsSetup` reads the result afterwards. A
/// plain `&mut sunbooleantype` cannot model that (it would either be a
/// stale copy or an `ark_mem` borrow held across a re-entrant call), so
/// the flag itself is a shared cell: ARKStep/MRIStep store
/// `pub jcur: ARKJcurPtr` in their content struct, and every reader uses
/// `.get()` / every writer `.set()`.
pub type ARKJcurPtr = Rc<Cell<sunbooleantype>>;

/* linear solver interface functions */
pub type ARKLinsolInitFn = fn(ark_mem: &ARKodeMem) -> i32;

pub type ARKLinsolSetupFn = fn(
    ark_mem: &ARKodeMem,
    convfail: i32,
    tpred: sunrealtype,
    ypred: &N_Vector,
    fpred: &N_Vector,
    jcurPtr: &Cell<sunbooleantype>,
    vtemp1: &N_Vector,
    vtemp2: &N_Vector,
    vtemp3: &N_Vector,
) -> i32;

pub type ARKLinsolSolveFn = fn(
    ark_mem: &ARKodeMem,
    b: &N_Vector,
    tcur: sunrealtype,
    ycur: &N_Vector,
    fcur: &N_Vector,
    client_tol: sunrealtype,
    mnewt: i32,
) -> i32;

pub type ARKLinsolFreeFn = fn(ark_mem: &ARKodeMem) -> i32;

/* mass matrix solver interface functions */
pub type ARKMassInitFn = fn(ark_mem: &ARKodeMem) -> i32;

pub type ARKMassSetupFn = fn(
    ark_mem: &ARKodeMem,
    t: sunrealtype,
    vtemp1: &N_Vector,
    vtemp2: &N_Vector,
    vtemp3: &N_Vector,
) -> i32;

pub type ARKMassMultFn = fn(arkode_mem: &ARKodeMem, v: &N_Vector, Mv: &N_Vector) -> i32;

pub type ARKMassSolveFn =
    fn(ark_mem: &ARKodeMem, b: &N_Vector, client_tol: sunrealtype) -> i32;

pub type ARKMassFreeFn = fn(ark_mem: &ARKodeMem) -> i32;

/* time stepper interface functions -- general */
pub type ARKTimestepInitFn = fn(ark_mem: &ARKodeMem, init_type: i32) -> i32;

pub type ARKTimestepFullRHSFn =
    fn(ark_mem: &ARKodeMem, t: sunrealtype, y: &N_Vector, f: &N_Vector, mode: i32) -> i32;

pub type ARKTimestepStepFn =
    fn(ark_mem: &ARKodeMem, dsm: &mut sunrealtype, nflag: &mut i32) -> i32;

/// C `(ARKodeMem, void* user_data)`. `ARKodeSetUserData` stores the box in
/// `ark_mem.user_data` and then `Option::take`s it back out for the
/// duration of this call, so the hook receives exclusive access to the
/// very same box (never a clone).
pub type ARKTimetepSetUserDataFn =
    fn(ark_mem: &ARKodeMem, user_data: &mut Option<Box<dyn Any>>) -> i32;

pub type ARKTimestepPrintAllStats =
    fn(ark_mem: &ARKodeMem, outfile: &SUNFile, fmt: SUNOutputFormat) -> i32;

pub type ARKTimestepWriteParameters = fn(ark_mem: &ARKodeMem, fp: &SUNFile) -> i32;

pub type ARKTimestepResize = fn(
    ark_mem: &ARKodeMem,
    ynew: &N_Vector,
    hscale: sunrealtype,
    t0: sunrealtype,
    resize: Option<ARKVecResizeFn>,
    resize_data: &mut Option<Box<dyn Any>>,
) -> i32;

pub type ARKTimestepReset = fn(ark_mem: &ARKodeMem, tR: sunrealtype, yR: &N_Vector) -> i32;

pub type ARKTimestepFree = fn(ark_mem: &ARKodeMem);

pub type ARKTimestepPrintMem = fn(ark_mem: &ARKodeMem, outfile: &SUNFile);

pub type ARKTimestepSetDefaults = fn(ark_mem: &ARKodeMem) -> i32;

pub type ARKTimestepSetOrder = fn(ark_mem: &ARKodeMem, maxord: i32) -> i32;

pub type ARKTimestepGetNumRhsEvals =
    fn(ark_mem: &ARKodeMem, partition_index: i32, num_rhs_evals: &mut i64) -> i32;

pub type ARKTimestepSetStepDirection = fn(ark_mem: &ARKodeMem, stepdir: sunrealtype) -> i32;

pub type ARKTimestepSetUseCompensatedSums =
    fn(ark_mem: &ARKodeMem, onoff: sunbooleantype) -> i32;

pub type ARKTimestepSetOptions = fn(
    ark_mem: &ARKodeMem,
    argidx: &mut i32,
    argv: &[String],
    offset: usize,
    arg_used: &mut sunbooleantype,
) -> i32;

pub type ARKTimestepGetStageIndex =
    fn(ark_mem: &ARKodeMem, stage: &mut i32, max_stages: &mut i32) -> i32;

/* time stepper interface functions -- temporal adaptivity */
pub type ARKTimestepComputeH0 =
    fn(ark_mem: &ARKodeMem, tout: sunrealtype, hin: &mut sunrealtype) -> i32;

pub type ARKTimestepGetEstLocalErrors = fn(ark_mem: &ARKodeMem, ele: &N_Vector) -> i32;

/// C `SUNAdaptController C` may legitimately be NULL here (it resets the
/// stepper to its default controller), hence `Option<&_>`.
pub type ARKSetAdaptControllerFn =
    fn(ark_mem: &ARKodeMem, C: Option<&SUNAdaptController>) -> i32;

/* time stepper interface functions -- relaxation */
pub type ARKTimestepSetRelaxFn =
    fn(ark_mem: &ARKodeMem, rfn: Option<ARKRelaxFn>, rjac: Option<ARKRelaxJacFn>) -> i32;

/* time stepper interface functions -- implicit solvers */

/// C `(..., void* lmem)`: the ARKLS system record. The box is moved into
/// `ark_mem.ark_lmem` (see `ARKodeMemRec::ark_lmem`); the stepper records
/// the fn pointers, `lsolve_type`, and the fact that a linear solver is
/// attached.
pub type ARKTimestepAttachLinsolFn = fn(
    ark_mem: &ARKodeMem,
    linit: Option<ARKLinsolInitFn>,
    lsetup: Option<ARKLinsolSetupFn>,
    lsolve: Option<ARKLinsolSolveFn>,
    lfree: Option<ARKLinsolFreeFn>,
    lsolve_type: SUNLinearSolver_Type,
    lmem: Option<Box<dyn Any>>,
) -> i32;

pub type ARKTimestepDisableLSetup = fn(ark_mem: &ARKodeMem);

/// C `void* (*)(ARKodeMem)` returning `step_mem->lmem`. Because the ARKLS
/// record is stored BY VALUE in `ark_mem.ark_lmem`, the Rust seam reports
/// *presence* instead of handing out the pointer: `SUNTRUE` iff this
/// stepper has an attached system linear solver. C call sites written as
/// `(ARKLsMem) ark_mem->step_getlinmem(ark_mem)` become
/// `arkls_mem_mut(ark_mem)` (arkode_ls.rs).
pub type ARKTimestepGetLinMemFn = fn(ark_mem: &ARKodeMem) -> sunbooleantype;

pub type ARKTimestepGetImplicitRHSFn = fn(ark_mem: &ARKodeMem) -> Option<ARKRhsFn>;

/// C `sunbooleantype** jcur` is an out-param yielding `&step_mem->jcur`;
/// the Rust out-param yields a clone of the shared `ARKJcurPtr` cell
/// (`None` where C would return NULL).
pub type ARKTimestepGetGammasFn = fn(
    ark_mem: &ARKodeMem,
    gamma: &mut sunrealtype,
    gamrat: &mut sunrealtype,
    jcur: &mut Option<ARKJcurPtr>,
    dgamma_fail: &mut sunbooleantype,
) -> i32;

pub type ARKTimestepComputeState =
    fn(ark_mem: &ARKodeMem, zcor: &N_Vector, z: &N_Vector) -> i32;

pub type ARKTimestepSetNonlinearSolver =
    fn(ark_mem: &ARKodeMem, NLS: &SUNNonlinearSolver) -> i32;

pub type ARKTimestepSetLinear = fn(ark_mem: &ARKodeMem, timedepend: i32) -> i32;

pub type ARKTimestepSetNonlinear = fn(ark_mem: &ARKodeMem) -> i32;

pub type ARKTimestepSetAutonomous =
    fn(ark_mem: &ARKodeMem, autonomous: sunbooleantype) -> i32;

pub type ARKTimestepSetNlsRhsFn = fn(ark_mem: &ARKodeMem, nls_fi: Option<ARKRhsFn>) -> i32;

pub type ARKTimestepSetDeduceImplicitRhs =
    fn(ark_mem: &ARKodeMem, deduce: sunbooleantype) -> i32;

pub type ARKTimestepSetNonlinCRDown = fn(ark_mem: &ARKodeMem, crdown: sunrealtype) -> i32;

pub type ARKTimestepSetNonlinRDiv = fn(ark_mem: &ARKodeMem, rdiv: sunrealtype) -> i32;

pub type ARKTimestepSetDeltaGammaMax = fn(ark_mem: &ARKodeMem, dgmax: sunrealtype) -> i32;

pub type ARKTimestepSetLSetupFrequency = fn(ark_mem: &ARKodeMem, msbp: i32) -> i32;

pub type ARKTimestepSetPredictorMethod = fn(ark_mem: &ARKodeMem, method: i32) -> i32;

pub type ARKTimestepSetMaxNonlinIters = fn(ark_mem: &ARKodeMem, maxcor: i32) -> i32;

pub type ARKTimestepSetNonlinConvCoef = fn(ark_mem: &ARKodeMem, nlscoef: sunrealtype) -> i32;

pub type ARKTimestepSetStagePredictFn =
    fn(ark_mem: &ARKodeMem, PredictStage: Option<ARKStagePredictFn>) -> i32;

pub type ARKTimestepGetNumLinSolvSetups =
    fn(ark_mem: &ARKodeMem, nlinsetups: &mut i64) -> i32;

pub type ARKTimestepGetCurrentGamma = fn(ark_mem: &ARKodeMem, gamma: &mut sunrealtype) -> i32;

/// C `void** user_data` out-param: the box is SWAPPED with the caller's
/// out-param (accepted deviation class 6); the caller must hand it back
/// before the integrator next invokes a user callback.
pub type ARKTimestepGetNonlinearSystemData = fn(
    ark_mem: &ARKodeMem,
    tcur: &mut sunrealtype,
    zpred: &mut Option<N_Vector>,
    z: &mut Option<N_Vector>,
    Fi: &mut Option<N_Vector>,
    gamma: &mut sunrealtype,
    sdata: &mut Option<N_Vector>,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32;

pub type ARKTimestepGetNumNonlinSolvIters = fn(ark_mem: &ARKodeMem, nniters: &mut i64) -> i32;

pub type ARKTimestepGetNumNonlinSolvConvFails =
    fn(ark_mem: &ARKodeMem, nnfails: &mut i64) -> i32;

pub type ARKTimestepGetNonlinSolvStats =
    fn(ark_mem: &ARKodeMem, nniters: &mut i64, nnfails: &mut i64) -> i32;

/* time stepper interface functions -- non-identity mass matrices */

/// C `(..., void* mass_mem)`: the ARKLS mass record. The box is moved
/// into `ark_mem.ark_mass_mem` (see `ARKodeMemRec::ark_mass_mem`).
pub type ARKTimestepAttachMasssolFn = fn(
    ark_mem: &ARKodeMem,
    minit: Option<ARKMassInitFn>,
    msetup: Option<ARKMassSetupFn>,
    mmult: Option<ARKMassMultFn>,
    msolve: Option<ARKMassSolveFn>,
    mfree: Option<ARKMassFreeFn>,
    time_dep: sunbooleantype,
    msolve_type: SUNLinearSolver_Type,
    mass_mem: Option<Box<dyn Any>>,
) -> i32;

pub type ARKTimestepDisableMSetup = fn(ark_mem: &ARKodeMem);

/// Presence probe, exactly as `ARKTimestepGetLinMemFn`: `SUNTRUE` iff
/// this stepper has an attached mass-matrix linear solver
/// (`arkls_mass_mem_mut(ark_mem)` reaches the record itself).
pub type ARKTimestepGetMassMemFn = fn(ark_mem: &ARKodeMem) -> sunbooleantype;

/* time stepper interface functions -- forcing */
pub type ARKTimestepSetForcingFn = fn(
    ark_mem: &ARKodeMem,
    tshift: sunrealtype,
    tscale: sunrealtype,
    f: &[N_Vector],
    nvecs: i32,
) -> i32;

/* =================================================================
 * Stepper-supplied relaxation functions (arkode_relaxation_impl.h)
 * =================================================================*/

/* Compute the estimated change in entropy for this step delta_e */
pub type ARKRelaxDeltaEFn = fn(
    ark_mem: &ARKodeMem,
    relax_jac_fn: Option<ARKRelaxJacFn>,
    evals_out: &mut i64,
    delta_e_out: &mut sunrealtype,
) -> i32;

/* Get the method order */
pub type ARKRelaxGetOrderFn = fn(ark_mem: &ARKodeMem) -> i32;

/* =================================================================
 * ARKODE interpolation module definition (arkode_impl.h)
 * =================================================================*/

/* Structure containing function pointers to interpolation operations */
#[derive(Default, Clone)]
pub struct _generic_ARKInterpOps {
    pub resize: Option<
        fn(
            ark_mem: &ARKodeMem,
            interp: &ARKInterp,
            resize: Option<ARKVecResizeFn>,
            resize_data: &mut Option<Box<dyn Any>>,
            lrw_diff: sunindextype,
            liw_diff: sunindextype,
            tmpl: &N_Vector,
        ) -> i32,
    >,
    pub free: Option<fn(ark_mem: &ARKodeMem, interp: &ARKInterp)>,
    pub print: Option<fn(interp: &ARKInterp, outfile: &SUNFile)>,
    pub setdegree: Option<fn(ark_mem: &ARKodeMem, interp: &ARKInterp, degree: i32) -> i32>,
    pub init: Option<fn(ark_mem: &ARKodeMem, interp: &ARKInterp, tnew: sunrealtype) -> i32>,
    pub update: Option<fn(ark_mem: &ARKodeMem, interp: &ARKInterp, tnew: sunrealtype) -> i32>,
    pub evaluate: Option<
        fn(
            ark_mem: &ARKodeMem,
            interp: &ARKInterp,
            tau: sunrealtype,
            d: i32,
            order: i32,
            yout: &N_Vector,
        ) -> i32,
    >,
}

pub type ARKInterpOps = _generic_ARKInterpOps;

/* An interpolation module consists of an implementation-dependent
   'content' structure, and a structure of implementation-dependent
   operations. `arkode_interp.rs` owns both concrete contents
   (`_ARKInterpContent_Hermite`, `_ARKInterpContent_Lagrange`) and their
   `content_mut` downcast helpers. */
pub struct _generic_ARKInterp {
    pub content: RefCell<Box<dyn Any>>,
    pub ops: RefCell<ARKInterpOps>,
}

pub type ARKInterp = Rc<_generic_ARKInterp>;

/* =================================================================
 * ARKODE data structures
 * =================================================================*/

/*---------------------------------------------------------------
  Types : struct ARKodeMassMemRec, ARKodeMassMem
  ---------------------------------------------------------------
  This structure contains data pertaining to the use of a
  non-identity mass matrix.
  ---------------------------------------------------------------*/
pub struct ARKodeMassMemRec {
    /* mass matrix linear solver interface function pointers */
    pub minit: Option<ARKMassInitFn>,
    pub msetup: Option<ARKMassSetupFn>,
    pub mmult: Option<ARKMassMultFn>,
    pub msolve: Option<ARKMassSolveFn>,
    pub mfree: Option<ARKMassFreeFn>,
    pub sol_mem: Option<Box<dyn Any>>, /* mass matrix solver interface data */
    pub msolve_type: i32,              /* mass matrix interface type:
                                       0=iterative; 1=direct; 2=custom */
}

pub type ARKodeMassMem = Box<ARKodeMassMemRec>;

/*---------------------------------------------------------------
  Types : struct ARKodeHAdaptMemRec, ARKodeHAdaptMem
  (arkode_adapt_impl.h)
  ---------------------------------------------------------------*/
pub struct ARKodeHAdaptMemRec {
    pub etamax: sunrealtype, /* eta <= etamax                              */
    pub etamx1: sunrealtype, /* max step size change on first step         */
    pub etamxf: sunrealtype, /* h reduction factor on multiple error fails */
    pub etamin: sunrealtype, /* eta >= etamin on error test fail           */
    pub small_nef: i32,      /* bound to determine 'multiple' above        */
    pub etacf: sunrealtype,  /* h reduction factor on nonlinear conv fail  */
    pub cfl: sunrealtype,    /* cfl safety factor                          */
    pub safety: sunrealtype, /* accuracy safety factor on h                */
    pub growth: sunrealtype, /* maximum step growth safety factor          */
    pub lbound: sunrealtype, /* eta lower bound to leave h unchanged       */
    pub ubound: sunrealtype, /* eta upper bound to leave h unchanged       */
    pub p: i32,              /* embedding order                            */
    pub q: i32,              /* method order                               */
    pub pq: i32,             /* decision flag for controller order         */
    pub adjust: i32,         /* controller order adjustment factor         */

    pub hcontroller: Option<SUNAdaptController>, /* temporal error controller     */
    pub owncontroller: sunbooleantype, /* flag indicating hcontroller ownership   */

    pub expstab: Option<ARKExpStabFn>, /* step stability function          */
    pub estab_data: Option<Box<dyn Any>>, /* user pointer passed to expstab */

    pub nst_acc: i64, /* num accuracy-limited internal steps        */
    pub nst_exp: i64, /* num stability-limited internal steps       */
}

pub type ARKodeHAdaptMem = Box<ARKodeHAdaptMemRec>;

/*---------------------------------------------------------------
  Types : struct ARKodeRootMemRec, ARKodeRootMem
  (arkode_root_impl.h)
  ---------------------------------------------------------------*/
pub struct ARKodeRootMemRec {
    pub gfun: Option<ARKRootFn>, /* function g for roots sought                  */
    pub nrtfn: i32,              /* number of components of g                    */
    pub iroots: Vec<i32>,        /* array for root information                   */
    pub rootdir: Vec<i32>,       /* array specifying direction of zero-crossing  */
    pub tlo: sunrealtype,        /* nearest endpoint of interval in root search  */
    pub thi: sunrealtype,        /* farthest endpoint of interval in root search */
    pub trout: sunrealtype,      /* t value returned by rootfinding routine      */
    pub glo: Vec<sunrealtype>,   /* saved array of g values at t = tlo           */
    pub ghi: Vec<sunrealtype>,   /* saved array of g values at t = thi           */
    pub grout: Vec<sunrealtype>, /* array of g values at t = trout               */
    pub ttol: sunrealtype,       /* tolerance on root location                   */
    pub irfnd: i32,              /* flag showing whether last step had a root    */
    pub nge: i64,                /* counter for g evaluations                    */
    pub gactive: Vec<sunbooleantype>, /* array with active/inactive event fns    */
    pub mxgnull: i32,            /* num. warning messages about possible g==0    */

    /// C `root_data = ark_mem->user_data` is a raw pointer snapshot; a
    /// `Box` cannot alias, so (accepted deviation class 6) this field
    /// stays `None` and `arkRootCheck*`/`arkRootfind` pass the CURRENT
    /// `ark_mem.user_data` box to `gfun` at call time.
    pub root_data: Option<Box<dyn Any>>,
}

pub type ARKodeRootMem = Box<ARKodeRootMemRec>;

/*---------------------------------------------------------------
  Types : struct ARKodeRelaxMemRec, ARKodeRelaxMem
  (arkode_relaxation_impl.h)
  ---------------------------------------------------------------*/
pub struct ARKodeRelaxMemRec {
    /* user-supplied and stepper supplied functions */
    pub relax_fn: Option<ARKRelaxFn>, /* user relaxation function ("entropy") */
    pub relax_jac_fn: Option<ARKRelaxJacFn>, /* user relaxation Jacobian      */
    pub delta_e_fn: Option<ARKRelaxDeltaEFn>, /* get delta entropy from stepper */
    pub get_order_fn: Option<ARKRelaxGetOrderFn>, /* get the method order     */

    /* relaxation variables */
    pub max_fails: i32,                /* max allowed relax fails in a step   */
    pub num_relax_fn_evals: i64,       /* counter for total function evals    */
    pub num_relax_jac_evals: i64,      /* counter for total jacobian evals    */
    pub num_fails: i64,                /* counter for total relaxation fails  */
    pub e_old: sunrealtype,            /* entropy at start of step y(t_{n-1}) */
    pub delta_e: sunrealtype,          /* change in entropy                   */
    pub res: sunrealtype,              /* relaxation residual value           */
    pub jac: sunrealtype,              /* relaxation Jacobian value           */
    pub relax_param: sunrealtype,      /* current relaxation parameter value  */
    pub relax_param_prev: sunrealtype, /* previous relaxation parameter value */
    pub lower_bound: sunrealtype,      /* smallest allowed relaxation value   */
    pub upper_bound: sunrealtype,      /* largest allowed relaxation value    */
    pub eta_fail: sunrealtype,         /* failed relaxation step size factor  */

    /* nonlinear solver settings */
    pub solver: ARKRelaxSolver, /* choice of relaxation solver          */
    pub res_tol: sunrealtype,   /* nonlinear residual solve tolerance   */
    pub rel_tol: sunrealtype,   /* nonlinear iterate relative tolerance */
    pub abs_tol: sunrealtype,   /* nonlinear iterate absolute tolerance */
    pub max_iters: i32,         /* nonlinear solve max iterations       */
    pub nls_iters: i64,         /* total nonlinear iterations           */
    pub nls_fails: i64,         /* number of nonlinear solver fails     */
    pub bound_fails: i64,       /* number of relax param bound fails    */
}

pub type ARKodeRelaxMem = Box<ARKodeRelaxMemRec>;

/*---------------------------------------------------------------
  Types : struct ARKodeMemRec, ARKodeMem
  ---------------------------------------------------------------
  This structure contains fields to keep track of problem state.
  ---------------------------------------------------------------*/
pub struct ARKodeMemRec {
    pub sunctx: SUNContext,

    /// C `void* python` (only populated by the Python bindings, which are
    /// out of scope for this port); always `None`.
    pub python: Option<Box<dyn Any>>,

    pub uround: sunrealtype, /* machine unit roundoff */

    /* Problem specification data */
    pub user_data: Option<Box<dyn Any>>, /* user ptr passed to supplied functions */
    pub itol: i32,                       /* itol = ARK_SS (scalar, default),
                                         ARK_SV (vector),
                                         ARK_WF (user weight function)  */
    pub ritol: i32,                      /* itol = ARK_SS (scalar, default),
                                         ARK_SV (vector),
                                         ARK_WF (user weight function)  */
    pub reltol: sunrealtype,             /* relative tolerance                    */
    pub Sabstol: sunrealtype,            /* scalar absolute solution tolerance    */
    pub Vabstol: Option<N_Vector>,       /* vector absolute solution tolerance    */
    pub atolmin0: sunbooleantype,        /* flag indicating that min(abstol) = 0  */
    pub SRabstol: sunrealtype,           /* scalar absolute residual tolerance    */
    pub VRabstol: Option<N_Vector>,      /* vector absolute residual tolerance    */
    pub Ratolmin0: sunbooleantype,       /* flag indicating that min(Rabstol) = 0 */
    pub user_efun: sunbooleantype,       /* SUNTRUE if user sets efun             */
    pub efun: Option<ARKEwtFn>,          /* function to set ewt                   */
    pub e_data: Option<Box<dyn Any>>,    /* user pointer passed to efun           */
    pub user_rfun: sunbooleantype,       /* SUNTRUE if user sets rfun             */
    pub rfun: Option<ARKRwtFn>,          /* function to set rwt                   */
    pub r_data: Option<Box<dyn Any>>,    /* user pointer passed to rfun           */

    /* Time stepper module -- general.
       `step_mem` holds the ACTIVE stepper's content struct BY VALUE
       (`Box<dyn Any>` = C `void* step_mem`); each stepper module defines
       `pub fn <x>Step_mem_mut(ark_mem: &ARKodeMem) -> RefMut<'_, ARKode<X>StepMemRec>`
       to reach it. */
    pub step_mem: Option<Box<dyn Any>>,
    pub step_init: Option<ARKTimestepInitFn>,
    pub step_fullrhs: Option<ARKTimestepFullRHSFn>,
    pub step: Option<ARKTimestepStepFn>,
    pub step_setuserdata: Option<ARKTimetepSetUserDataFn>,
    pub step_printallstats: Option<ARKTimestepPrintAllStats>,
    pub step_writeparameters: Option<ARKTimestepWriteParameters>,
    pub step_resize: Option<ARKTimestepResize>,
    pub step_reset: Option<ARKTimestepReset>,
    pub step_free: Option<ARKTimestepFree>,
    pub step_printmem: Option<ARKTimestepPrintMem>,
    pub step_setdefaults: Option<ARKTimestepSetDefaults>,
    pub step_setorder: Option<ARKTimestepSetOrder>,
    pub step_getnumrhsevals: Option<ARKTimestepGetNumRhsEvals>,
    pub step_setstepdirection: Option<ARKTimestepSetStepDirection>,
    pub step_setusecompensatedsums: Option<ARKTimestepSetUseCompensatedSums>,
    pub step_setoptions: Option<ARKTimestepSetOptions>,
    pub step_getstageindex: Option<ARKTimestepGetStageIndex>,

    /* Time stepper module -- temporal adaptivity */
    pub step_supports_adaptive: sunbooleantype,
    pub step_H0: Option<ARKTimestepComputeH0>,
    pub step_setadaptcontroller: Option<ARKSetAdaptControllerFn>,
    pub step_getestlocalerrors: Option<ARKTimestepGetEstLocalErrors>,

    /* Time stepper module -- relaxation */
    pub step_supports_relaxation: sunbooleantype,
    pub step_setrelaxfn: Option<ARKTimestepSetRelaxFn>,

    /* Time stepper module -- implicit solvers */
    pub step_supports_implicit: sunbooleantype,
    pub step_attachlinsol: Option<ARKTimestepAttachLinsolFn>,
    pub step_disablelsetup: Option<ARKTimestepDisableLSetup>,
    pub step_getlinmem: Option<ARKTimestepGetLinMemFn>,
    pub step_getimplicitrhs: Option<ARKTimestepGetImplicitRHSFn>,
    pub step_getgammas: Option<ARKTimestepGetGammasFn>,
    pub step_computestate: Option<ARKTimestepComputeState>,
    pub step_setnonlinearsolver: Option<ARKTimestepSetNonlinearSolver>,
    pub step_setlinear: Option<ARKTimestepSetLinear>,
    pub step_setautonomous: Option<ARKTimestepSetAutonomous>,
    pub step_setnonlinear: Option<ARKTimestepSetNonlinear>,
    pub step_setnlsrhsfn: Option<ARKTimestepSetNlsRhsFn>,
    pub step_setdeduceimplicitrhs: Option<ARKTimestepSetDeduceImplicitRhs>,
    pub step_setnonlincrdown: Option<ARKTimestepSetNonlinCRDown>,
    pub step_setnonlinrdiv: Option<ARKTimestepSetNonlinRDiv>,
    pub step_setdeltagammamax: Option<ARKTimestepSetDeltaGammaMax>,
    pub step_setlsetupfrequency: Option<ARKTimestepSetLSetupFrequency>,
    pub step_setpredictormethod: Option<ARKTimestepSetPredictorMethod>,
    pub step_setmaxnonliniters: Option<ARKTimestepSetMaxNonlinIters>,
    pub step_setnonlinconvcoef: Option<ARKTimestepSetNonlinConvCoef>,
    pub step_setstagepredictfn: Option<ARKTimestepSetStagePredictFn>,
    pub step_getnumlinsolvsetups: Option<ARKTimestepGetNumLinSolvSetups>,
    pub step_getcurrentgamma: Option<ARKTimestepGetCurrentGamma>,
    pub step_getnonlinearsystemdata: Option<ARKTimestepGetNonlinearSystemData>,
    pub step_getnumnonlinsolviters: Option<ARKTimestepGetNumNonlinSolvIters>,
    pub step_getnumnonlinsolvconvfails: Option<ARKTimestepGetNumNonlinSolvConvFails>,
    pub step_getnonlinsolvstats: Option<ARKTimestepGetNonlinSolvStats>,

    /* Time stepper module -- non-identity mass matrices */
    pub step_supports_massmatrix: sunbooleantype,
    pub step_attachmasssol: Option<ARKTimestepAttachMasssolFn>,
    pub step_disablemsetup: Option<ARKTimestepDisableMSetup>,
    pub step_getmassmem: Option<ARKTimestepGetMassMemFn>,
    pub step_mmult: Option<ARKMassMultFn>,

    /* Time stepper module -- forcing */
    pub step_setforcing: Option<ARKTimestepSetForcingFn>,

    /// ARKLS system linear-solver record (`ARKLsMemRec`) held BY VALUE.
    ///
    /// In C this box lives in `step_mem->lmem` and is reached through
    /// `step_getlinmem`; a `Box<dyn Any>` cannot be handed out that way
    /// without moving it, and `arkode_ls.rs` must reach it without
    /// knowing the active stepper's concrete type. It therefore lives
    /// here, with `pub fn arkls_mem_mut(ark_mem: &ARKodeMem) ->
    /// RefMut<'_, ARKLsMemRec>` defined in `arkode_ls.rs`. Ownership,
    /// lifetime and the `step_attachlinsol` / `step_getlinmem` /
    /// `arkLsFree` call sequence are otherwise unchanged from C.
    pub ark_lmem: Option<Box<dyn Any>>,

    /// ARKLS mass-matrix linear-solver record (`ARKLsMassMemRec`) held BY
    /// VALUE — same rationale as `ark_lmem`; the accessor is
    /// `pub fn arkls_mass_mem_mut(ark_mem: &ARKodeMem) ->
    /// RefMut<'_, ARKLsMassMemRec>` in `arkode_ls.rs`.
    pub ark_mass_mem: Option<Box<dyn Any>>,

    /* N_Vector storage */
    pub ewt: Option<N_Vector>, /* error weight vector                        */
    pub rwt: Option<N_Vector>, /* residual weight vector                     */
    pub rwt_is_ewt: sunbooleantype, /* SUNTRUE if rwt is a pointer to ewt    */
    pub ycur: Option<N_Vector>, /* pointer to user-provided solution memory;
                               used as evolving solution by the time stepper
                               modules (aliases the user's `yout` during
                               ARKodeEvolve -- copy back at EVERY return) */
    pub ensure_ycur: sunbooleantype, /* SUNTRUE if stepper expects ycur=yn on
                                     entry to its takestep routine */
    pub yn: Option<N_Vector>, /* solution from the last successful step     */
    /// C `ark_mem->fn` (full IVP right-hand side from last step); renamed
    /// because `fn` is a Rust keyword.
    pub fn_: Option<N_Vector>,
    pub fn_is_current: sunbooleantype, /* SUNTRUE if fn has been evaluated at yn */
    pub tempv1: Option<N_Vector>,      /* temporary storage vectors (for local use */
    pub tempv2: Option<N_Vector>,      /* and by time-stepping modules)            */
    pub tempv3: Option<N_Vector>,
    pub tempv4: Option<N_Vector>,
    pub tempv5: Option<N_Vector>,

    /* Temporal interpolation module */
    pub interp: Option<ARKInterp>,
    pub interp_type: i32,
    pub interp_degree: i32,

    /* Tstop information */
    pub tstopset: sunbooleantype,
    pub tstopinterp: sunbooleantype,
    pub tstop: sunrealtype,

    /* Time step data */
    pub hin: sunrealtype,      /* initial step size                        */
    pub h: sunrealtype,        /* current step size                        */
    pub hmin: sunrealtype,     /* |h| >= hmin                              */
    pub hmax_inv: sunrealtype, /* |h| <= 1/hmax_inv                        */
    pub hprime: sunrealtype,   /* next actual step size to be used         */
    pub next_h: sunrealtype,   /* next dynamical step size (only used in
                               getCurrentStep); note that this could
                               overtake tstop */
    pub eta: sunrealtype,      /* eta = hprime / h                         */
    pub tcur: sunrealtype,     /* current internal value of t
                               (changes with each stage)                  */
    pub tretlast: sunrealtype, /* value of tret last returned by ARKODE    */
    pub fixedstep: sunbooleantype, /* flag to disable temporal adaptivity  */
    pub hadapt_mem: Option<ARKodeHAdaptMem>, /* time step adaptivity structure */

    /* Limits and various solver parameters */
    pub mxstep: i64, /* max number of internal steps for one user call */
    pub mxhnil: i32, /* max number of warning messages issued to the
                     user that t+h == t for the next internal step  */
    pub maxnef: i32, /* max error test fails in one step               */
    pub maxncf: i32, /* max num alg. solver conv. fails in one step    */

    /* Counters */
    pub nst_attempts: i64, /* number of attempted steps                  */
    pub nst: i64,          /* number of internal steps taken             */
    pub nhnil: i32,        /* number of messages issued to the user that
                           t+h == t for the next iternal step          */
    pub ncfn: i64,         /* num corrector convergence failures         */
    pub netf: i64,         /* num error test failures                    */

    /* Space requirements for ARKODE */
    pub lrw1: sunindextype, /* no. of sunrealtype words in 1 N_Vector       */
    pub liw1: sunindextype, /* no. of integer words in 1 N_Vector           */
    pub lrw: i64,           /* no. of sunrealtype words in ARKODE work vectors */
    pub liw: i64,           /* no. of integer words in ARKODE work vectors  */

    /* Saved Values */
    pub h0u: sunrealtype,   /* actual initial stepsize                     */
    pub tn: sunrealtype,    /* time of last successful step                */
    pub terr: sunrealtype,  /* error in tn for compensated sums            */
    pub hold: sunrealtype,  /* last successful h value used                */
    pub tolsf: sunrealtype, /* tolerance scale factor (suggestion to user) */
    pub AccumErrorType: ARKAccumError, /* accumulated error estimation type */
    pub AccumErrorStart: sunrealtype, /* time of last accumulated error reset */
    pub AccumError: sunrealtype, /* accumulated error estimate             */
    pub VabstolMallocDone: sunbooleantype,
    pub VRabstolMallocDone: sunbooleantype,
    pub MallocDone: sunbooleantype,
    pub initsetup: sunbooleantype, /* denotes a call to InitialSetup is needed  */
    pub init_type: i32,            /* initialization type (see constants above) */
    pub firststage: sunbooleantype, /* denotes first stage in simulation        */
    pub initialized: sunbooleantype, /* denotes arkInitialSetup has been done   */
    pub call_fullrhs: sunbooleantype, /* denotes the full RHS fn will be called */
    pub preallocated: sunbooleantype, /* SUNTRUE if ARKodeInit has been
                                      called to preallocate data
                                      prior to ARKodeEvolve */

    /* Rootfinding Data */
    pub root_mem: Option<ARKodeRootMem>, /* root-finding structure */

    /* Inequality Constraints Data */
    pub constraints: Option<N_Vector>, /* vector of constraint flags     */
    pub nconstrfails: i64,             /* total constraint failures      */
    pub maxconstrfails: i32,           /* max failures allowed in a step */

    /* Relaxation Data */
    pub relax_enabled: sunbooleantype, /* is relaxation enabled?    */
    pub relax_mem: Option<ARKodeRelaxMem>, /* relaxation data structure */

    /* User-supplied step solution pre/post-processing functions */
    pub PreStepFn: Option<ARKPreStepFn>,
    pub PostStepFn: Option<ARKPostStepFn>,

    /* User-supplied RHS function pre-processing function */
    pub PreRhsFn: Option<ARKPreRhsFn>,

    /* User-supplied stage and step solution post-processing function */
    pub PostProcessStepFn: Option<ARKPostProcessFn>,
    pub PostProcessStageFn: Option<ARKPostProcessFn>,

    pub use_compensated_sums: sunbooleantype,

    /* Adjoint solver data */
    pub load_checkpoint_fail: sunbooleantype,
    pub do_adjoint: sunbooleantype,
    pub adj_stage_idx: suncountertype, /* current stage index (only valid in
                                       adjoint context) */
    pub adj_step_idx: suncountertype, /* current step index (only valid in
                                      adjoint context) */

    /* Checkpointing data */
    pub checkpoint_scheme: Option<SUNAdjointCheckpointScheme>,
    pub checkpoint_step_idx: suncountertype, /* the step number for checkpointing */

    /* XBraid interface variables */
    pub force_pass: sunbooleantype, /* when true the step attempt loop will ignore
                                    the return value (kflag) from
                                    arkCheckTemporalError and set
                                    kflag = ARK_SUCCESS to force the step attempt
                                    to always pass (if a solver failure did not
                                    occur before the error test). */
    pub last_kflag: i32, /* last value of the return flag (kflag) from a call
                         to arkCheckTemporalError. This is only set when
                         force_pass is true and is used by the XBraid
                         interface to determine if a time step passed or
                         failed the time step error test.  */
}

pub type ARKodeMem = Rc<RefCell<ARKodeMemRec>>;

impl ARKodeMemRec {
    /// All-zero/None baseline, mirroring C `arkCreate`'s
    /// `malloc` + `memset(ark_mem, 0, sizeof(struct ARKodeMemRec))`.
    /// `arkCreate` then assigns `sunctx`, `python = NULL`,
    /// `uround = SUN_UNIT_ROUNDOFF`, NULLs the whole `step_*` table, and
    /// installs every explicit default, so the baseline values below are
    /// never observable.
    pub fn zeroed(sunctx: SUNContext) -> ARKodeMemRec {
        ARKodeMemRec {
            sunctx,
            python: None,
            uround: 0.0,
            user_data: None,
            itol: 0,
            ritol: 0,
            reltol: 0.0,
            Sabstol: 0.0,
            Vabstol: None,
            atolmin0: SUNFALSE,
            SRabstol: 0.0,
            VRabstol: None,
            Ratolmin0: SUNFALSE,
            user_efun: SUNFALSE,
            efun: None,
            e_data: None,
            user_rfun: SUNFALSE,
            rfun: None,
            r_data: None,
            step_mem: None,
            step_init: None,
            step_fullrhs: None,
            step: None,
            step_setuserdata: None,
            step_printallstats: None,
            step_writeparameters: None,
            step_resize: None,
            step_reset: None,
            step_free: None,
            step_printmem: None,
            step_setdefaults: None,
            step_setorder: None,
            step_getnumrhsevals: None,
            step_setstepdirection: None,
            step_setusecompensatedsums: None,
            step_setoptions: None,
            step_getstageindex: None,
            step_supports_adaptive: SUNFALSE,
            step_H0: None,
            step_setadaptcontroller: None,
            step_getestlocalerrors: None,
            step_supports_relaxation: SUNFALSE,
            step_setrelaxfn: None,
            step_supports_implicit: SUNFALSE,
            step_attachlinsol: None,
            step_disablelsetup: None,
            step_getlinmem: None,
            step_getimplicitrhs: None,
            step_getgammas: None,
            step_computestate: None,
            step_setnonlinearsolver: None,
            step_setlinear: None,
            step_setautonomous: None,
            step_setnonlinear: None,
            step_setnlsrhsfn: None,
            step_setdeduceimplicitrhs: None,
            step_setnonlincrdown: None,
            step_setnonlinrdiv: None,
            step_setdeltagammamax: None,
            step_setlsetupfrequency: None,
            step_setpredictormethod: None,
            step_setmaxnonliniters: None,
            step_setnonlinconvcoef: None,
            step_setstagepredictfn: None,
            step_getnumlinsolvsetups: None,
            step_getcurrentgamma: None,
            step_getnonlinearsystemdata: None,
            step_getnumnonlinsolviters: None,
            step_getnumnonlinsolvconvfails: None,
            step_getnonlinsolvstats: None,
            step_supports_massmatrix: SUNFALSE,
            step_attachmasssol: None,
            step_disablemsetup: None,
            step_getmassmem: None,
            step_mmult: None,
            step_setforcing: None,
            ark_lmem: None,
            ark_mass_mem: None,
            ewt: None,
            rwt: None,
            rwt_is_ewt: SUNFALSE,
            ycur: None,
            ensure_ycur: SUNFALSE,
            yn: None,
            fn_: None,
            fn_is_current: SUNFALSE,
            tempv1: None,
            tempv2: None,
            tempv3: None,
            tempv4: None,
            tempv5: None,
            interp: None,
            interp_type: 0,
            interp_degree: 0,
            tstopset: SUNFALSE,
            tstopinterp: SUNFALSE,
            tstop: 0.0,
            hin: 0.0,
            h: 0.0,
            hmin: 0.0,
            hmax_inv: 0.0,
            hprime: 0.0,
            next_h: 0.0,
            eta: 0.0,
            tcur: 0.0,
            tretlast: 0.0,
            fixedstep: SUNFALSE,
            hadapt_mem: None,
            mxstep: 0,
            mxhnil: 0,
            maxnef: 0,
            maxncf: 0,
            nst_attempts: 0,
            nst: 0,
            nhnil: 0,
            ncfn: 0,
            netf: 0,
            lrw1: 0,
            liw1: 0,
            lrw: 0,
            liw: 0,
            h0u: 0.0,
            tn: 0.0,
            terr: 0.0,
            hold: 0.0,
            tolsf: 0.0,
            AccumErrorType: ARK_ACCUMERROR_NONE,
            AccumErrorStart: 0.0,
            AccumError: 0.0,
            VabstolMallocDone: SUNFALSE,
            VRabstolMallocDone: SUNFALSE,
            MallocDone: SUNFALSE,
            initsetup: SUNFALSE,
            init_type: 0,
            firststage: SUNFALSE,
            initialized: SUNFALSE,
            call_fullrhs: SUNFALSE,
            preallocated: SUNFALSE,
            root_mem: None,
            constraints: None,
            nconstrfails: 0,
            maxconstrfails: 0,
            relax_enabled: SUNFALSE,
            relax_mem: None,
            PreStepFn: None,
            PostStepFn: None,
            PreRhsFn: None,
            PostProcessStepFn: None,
            PostProcessStageFn: None,
            use_compensated_sums: SUNFALSE,
            load_checkpoint_fail: SUNFALSE,
            do_adjoint: SUNFALSE,
            adj_stage_idx: 0,
            adj_step_idx: 0,
            checkpoint_scheme: None,
            checkpoint_step_idx: 0,
            force_pass: SUNFALSE,
            last_kflag: 0,
        }
    }
}

/* =================================================================
 * High level error handler (relocated from arkode.c; C varargs map to a
 * pre-formatted msg — call sites use the MSG_ARK_* builders below)
 * =================================================================*/

pub fn arkProcessError(
    ark_mem: Option<&ARKodeMem>,
    error_code: i32,
    line: i32,
    func: &str,
    file: &str,
    msg: &str,
) {
    match ark_mem {
        None => {
            SUNGlobalFallbackErrHandler(line, func, file, msg, error_code);
        }
        Some(ark_mem) => {
            let sunctx = ark_mem.borrow().sunctx.clone();

            if error_code == ARK_WARNING {
                /* SUNDIALS_LOGGING_LEVEL >= WARNING in the reference build */
                let file_and_line = sunCombineFileAndLine(line, file);
                let logger = sunctx.borrow().logger.clone();
                if let Some(logger) = logger {
                    SUNLogger_QueueMsg(&logger, SUN_LOGLEVEL_WARNING, &file_and_line, func, msg);
                }
                return;
            }

            /* Call the SUNDIALS main error handler */
            SUNHandleErrWithMsg(line, func, file, msg, error_code, &sunctx);

            /* Clear the error now */
            let _ = SUNContext_GetLastError(&sunctx);
        }
    }
}

/* =================================================================
 * Reusable ARKODE Error Messages (arkode_impl.h). Parameter-less
 * messages are consts; parameterized ones are builders producing the
 * exact C expansion (`SUN_FORMAT_G` = `%.15g`).
 * =================================================================*/

/* MSG_TIME fragments */
pub fn MSG_TIME(t: sunrealtype) -> String {
    format!("t = {}", sun_format_g(t))
}

pub fn MSG_TIME_H(t: sunrealtype, h: sunrealtype) -> String {
    format!("t = {} and h = {}", sun_format_g(t), sun_format_g(h))
}

pub fn MSG_TIME_INT(t: sunrealtype, t0: sunrealtype, t1: sunrealtype) -> String {
    format!(
        "t = {} is not between tcur - hold = {} and tcur = {}",
        sun_format_g(t),
        sun_format_g(t0),
        sun_format_g(t1)
    )
}

pub fn MSG_TIME_TOUT(tout: sunrealtype) -> String {
    format!("tout = {}", sun_format_g(tout))
}

pub fn MSG_TIME_TSTOP(tstop: sunrealtype) -> String {
    format!("tstop = {}", sun_format_g(tstop))
}

/* Initialization and I/O error messages */
pub const MSG_ARK_NO_MEM: &str = "arkode_mem = NULL illegal.";
pub const MSG_ARK_ARKMEM_FAIL: &str = "Allocation of arkode_mem failed.";
pub const MSG_ARK_MEM_FAIL: &str = "A memory request failed.";
pub const MSG_ARK_NO_MALLOC: &str = "Attempt to call before ARKODE initialized.";
pub const MSG_ARK_BAD_HMIN_HMAX: &str = "Inconsistent step size limits: hmin > hmax.";
pub const MSG_ARK_BAD_RELTOL: &str = "reltol < 0 illegal.";
pub const MSG_ARK_BAD_ABSTOL: &str = "abstol has negative component(s) (illegal).";
pub const MSG_ARK_NULL_ABSTOL: &str = "abstol = NULL illegal.";
pub const MSG_ARK_BAD_RABSTOL: &str = "rabstol has negative component(s) (illegal).";
pub const MSG_ARK_NULL_RABSTOL: &str = "rabstol = NULL illegal.";
pub const MSG_ARK_NULL_Y0: &str = "y0 = NULL illegal.";
pub const MSG_ARK_Y0_FAIL_CONSTR: &str = "y0 fails to satisfy constraints.";
pub const MSG_ARK_NULL_F: &str = "Must specify at least one of fe, fi (both NULL).";
pub const MSG_ARK_NULL_G: &str = "g = NULL illegal.";
pub const MSG_ARK_BAD_NVECTOR: &str = "A required vector operation is not implemented.";
pub const MSG_ARK_BAD_CONSTR: &str = "Illegal values in constraints vector.";
pub const MSG_ARK_NULL_DKY: &str = "dky = NULL illegal.";

pub fn MSG_ARK_BAD_T(t: sunrealtype, t0: sunrealtype, t1: sunrealtype) -> String {
    format!("Illegal value for t. {}", MSG_TIME_INT(t, t0, t1))
}

pub const MSG_ARK_NO_ROOT: &str = "Rootfinding was not initialized.";

/* ARKODE Error Messages */
pub const MSG_ARK_YOUT_NULL: &str = "yout = NULL illegal.";
pub const MSG_ARK_TRET_NULL: &str = "tret = NULL illegal.";
pub const MSG_ARK_BAD_EWT: &str = "Initial ewt has component(s) equal to zero (illegal).";

pub fn MSG_ARK_EWT_NOW_BAD(t: sunrealtype) -> String {
    format!("At {}, a component of ewt has become <= 0.", MSG_TIME(t))
}

pub const MSG_ARK_BAD_RWT: &str = "Initial rwt has component(s) equal to zero (illegal).";

pub fn MSG_ARK_RWT_NOW_BAD(t: sunrealtype) -> String {
    format!("At {}, a component of rwt has become <= 0.", MSG_TIME(t))
}

pub const MSG_ARK_BAD_ITASK: &str = "Illegal value for itask.";
pub const MSG_ARK_BAD_H0: &str = "h0 and tout - t0 inconsistent.";

pub fn MSG_ARK_BAD_TOUT(tout: sunrealtype) -> String {
    format!(
        "Trouble interpolating at {}. tout too far back in direction of integration",
        MSG_TIME_TOUT(tout)
    )
}

pub const MSG_ARK_EWT_FAIL: &str = "The user-provide EwtSet function failed.";

pub fn MSG_ARK_EWT_NOW_FAIL(t: sunrealtype) -> String {
    format!("At {}, the user-provide EwtSet function failed.", MSG_TIME(t))
}

pub const MSG_ARK_RWT_FAIL: &str = "The user-provide RwtSet function failed.";

pub fn MSG_ARK_RWT_NOW_FAIL(t: sunrealtype) -> String {
    format!("At {}, the user-provide RwtSet function failed.", MSG_TIME(t))
}

pub const MSG_ARK_LINIT_FAIL: &str = "The linear solver's init routine failed.";
pub const MSG_ARK_HNIL_DONE: &str = "The above warning has been issued mxhnil times and will not \
                                     be issued again for this problem.";
pub const MSG_ARK_TOO_CLOSE: &str = "tout too close to t0 to start integration.";

pub fn MSG_ARK_MAX_STEPS(t: sunrealtype) -> String {
    format!("At {}, mxstep steps taken before reaching tout.", MSG_TIME(t))
}

pub fn MSG_ARK_TOO_MUCH_ACC(t: sunrealtype) -> String {
    format!("At {}, too much accuracy requested.", MSG_TIME(t))
}

pub fn MSG_ARK_HNIL(t: sunrealtype, h: sunrealtype) -> String {
    format!(
        "Internal {} are such that t + h = t on the next step. The solver will continue anyway.",
        MSG_TIME_H(t, h)
    )
}

pub fn MSG_ARK_ERR_FAILS(t: sunrealtype, h: sunrealtype) -> String {
    format!(
        "At {}, the error test failed repeatedly or with |h| = hmin.",
        MSG_TIME_H(t, h)
    )
}

pub fn MSG_ARK_CONV_FAILS(t: sunrealtype, h: sunrealtype) -> String {
    format!(
        "At {}, the solver convergence test failed repeatedly or with |h| = hmin.",
        MSG_TIME_H(t, h)
    )
}

pub fn MSG_ARK_SETUP_FAILED(t: sunrealtype) -> String {
    format!(
        "At {}, the setup routine failed in an unrecoverable manner.",
        MSG_TIME(t)
    )
}

pub fn MSG_ARK_SOLVE_FAILED(t: sunrealtype) -> String {
    format!(
        "At {}, the solve routine failed in an unrecoverable manner.",
        MSG_TIME(t)
    )
}

pub fn MSG_ARK_FAILED_CONSTR(t: sunrealtype) -> String {
    format!("At {}, unable to satisfy inequality constraints.", MSG_TIME(t))
}

pub fn MSG_ARK_RHSFUNC_FAILED(t: sunrealtype) -> String {
    format!(
        "At {}, the right-hand side routine failed in an unrecoverable manner.",
        MSG_TIME(t)
    )
}

pub fn MSG_ARK_RHSFUNC_UNREC(t: sunrealtype) -> String {
    format!(
        "At {}, the right-hand side failed in a recoverable manner, but no recovery is possible.",
        MSG_TIME(t)
    )
}

pub fn MSG_ARK_RHSFUNC_REPTD(t: sunrealtype) -> String {
    format!(
        "At {} repeated recoverable right-hand side function errors.",
        MSG_TIME(t)
    )
}

pub fn MSG_ARK_RTFUNC_FAILED(t: sunrealtype) -> String {
    format!(
        "At {}, the rootfinding routine failed in an unrecoverable manner.",
        MSG_TIME(t)
    )
}

pub fn MSG_ARK_CLOSE_ROOTS(t: sunrealtype) -> String {
    format!("Root found at and very near {}.", MSG_TIME(t))
}

pub fn MSG_ARK_BAD_TSTOP(tstop: sunrealtype, t: sunrealtype) -> String {
    format!(
        "The value {} is behind current {} in the direction of integration.",
        MSG_TIME_TSTOP(tstop),
        MSG_TIME(t)
    )
}

pub const MSG_ARK_INACTIVE_ROOTS: &str = "At the end of the first step, there are still some root \
                                          functions identically 0. This warning will not be \
                                          issued again.";
pub const MSG_ARK_RESIZE_FAIL: &str = "Error in user-supplied resize() function.";
pub const MSG_ARK_MASSINIT_FAIL: &str = "The mass matrix solver's init routine failed.";
pub const MSG_ARK_MASSSETUP_FAIL: &str = "The mass matrix solver's setup routine failed.";
pub const MSG_ARK_MASSSOLVE_FAIL: &str = "The mass matrix solver failed.";

pub fn MSG_ARK_NLS_FAIL(t: sunrealtype) -> String {
    format!(
        "At {} the nonlinear solver failed in an unrecoverable manner.",
        MSG_TIME(t)
    )
}

pub fn MSG_ARK_USER_PREDICT_FAIL(t: sunrealtype) -> String {
    format!(
        "At {} the user-supplied predictor failed in an unrecoverable manner.",
        MSG_TIME(t)
    )
}

pub const MSG_ARKADAPT_NO_MEM: &str = "Adaptivity memory structure not allocated.";

pub fn MSG_ARK_VECTOROP_ERR(t: sunrealtype) -> String {
    format!("At {}, a vector operation failed.", MSG_TIME(t))
}

pub fn MSG_ARK_INNERSTEP_FAILED(t: sunrealtype) -> String {
    format!(
        "At {}, the inner stepper failed in an unrecoverable manner.",
        MSG_TIME(t)
    )
}

pub fn MSG_ARK_PRESTEPFN_FAIL(t: sunrealtype) -> String {
    format!(
        "At {}, the pre-step function failed in an unrecoverable manner.",
        MSG_TIME(t)
    )
}

pub fn MSG_ARK_POSTSTEPFN_FAIL(t: sunrealtype) -> String {
    format!(
        "At {}, the post-step function failed in an unrecoverable manner.",
        MSG_TIME(t)
    )
}

pub fn MSG_ARK_POSTPROCESS_STEP_FAIL(t: sunrealtype) -> String {
    format!(
        "At {}, the step postprocessing routine failed in an unrecoverable manner.",
        MSG_TIME(t)
    )
}

pub fn MSG_ARK_POSTPROCESS_STAGE_FAIL(t: sunrealtype) -> String {
    format!(
        "At {}, the stage postprocessing routine failed in an unrecoverable manner.",
        MSG_TIME(t)
    )
}

pub fn MSG_ARK_PRERHSFN_FAIL(t: sunrealtype) -> String {
    format!(
        "At {}, the pre-RHS function failed in an unrecoverable manner.",
        MSG_TIME(t)
    )
}

pub const MSG_ARK_NULL_SUNCTX: &str = "sunctx = NULL illegal.";
pub const MSG_ARK_CONTEXT_MISMATCH: &str = "Outer and inner steppers have different contexts.";
pub const MSG_ARK_MISSING_FULLRHS: &str = "Time-stepping module missing fullrhs routine \
                                           (required by requested solver configuration).";

pub fn MSG_ARK_INTERPOLATION_FAIL(t: sunrealtype) -> String {
    format!("At {}, interpolating the solution failed.", MSG_TIME(t))
}

pub const MSG_ARK_ADJOINT_BAD_VECTOR: &str =
    "JacPFn or JPvpFn was provided, but the number of subvectors in y is not 2. To perform ASA \
     w.r.t. parameters, one subvector should be the state vector, and the other should be the \
     parameter vector.";

/* Relaxation error message (arkode_relaxation_impl.h) */
pub const MSG_RELAX_MEM_NULL: &str = "Relaxation memory is NULL.";
