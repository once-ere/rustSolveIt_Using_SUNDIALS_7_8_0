//! Port of `src/cvodes/cvodes_impl.h` + the constants/typedefs of
//! `include/cvodes/cvodes.h`, plus `src/cvodes/cvodes_proj_impl.h` (folded
//! here because `cvodes_impl.h` includes it and `CVodeMemRec` embeds the
//! projection memory), plus the adjoint (ASA) memory records that
//! `cvodes_impl.h` defines (`CVadjMemRec`, `CVckpntMemRec`,
//! `CVdtpntMemRec`, `CVodeBMemRec`, interpolation data records).
//!
//! `cvProcessError` (defined in `cvodes.c` upstream) is relocated here so
//! every cvodes module shares one definition; C varargs map to a
//! pre-formatted `msg` (call sites use the `MSGCV_*` constants/builders
//! below). Parameterized messages are functions producing the exact
//! C `printf` expansion (`SUN_FORMAT_G` = `%.15g` via `sun_format_g`).
//!
//! Fragment-file protocol: `cvodes.c` is ported as several fragment
//! modules; ALL module-scope constants `cvodes.c` defines (ZERO … HUNDRED,
//! RTFOUND/CLOSERT, CENTERED1/2, FORWARD1/2, CV_ONESENS/CV_ALLSENS,
//! CV_NN/CV_SS/CV_SV/CV_WF/CV_EE, FUZZ_FACTOR, HLB_FACTOR, HUB_FACTOR,
//! H_BIAS, MAX_ITERS, CORTES) live HERE so every fragment shares one
//! definition.
//!
//! Handle model: `CVodeMem = Rc<RefCell<CVodeMemRec>>`. Internal
//! functions take `&CVodeMem` and use granular borrows (never hold a
//! borrow across a callback, N_Vector op on user vectors, or
//! linear/nonlinear solver call — all can re-enter the mem). The adjoint
//! records use the same handle model (`CVadjMem`, `CVckpntMem`,
//! `CVdtpntMem`, `CVodeBMem` are `Rc<RefCell<…>>`); the C intrusive
//! linked lists (`ck_next`, `cv_next`) become `Vec`s ordered exactly as
//! the C traversal order (index 0 = list head = most recently inserted;
//! C `for (p = head; p; p = p->next)` ≡ `for p in vec.iter()`).

use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

use sundials_core::sundials_context::{SUNContext, SUNContext_GetLastError};
use sundials_core::sundials_errors::{SUNGlobalFallbackErrHandler, SUNHandleErrWithMsg};
use sundials_core::sundials_logger::{SUNLogger_QueueMsg, SUN_LOGLEVEL_WARNING};
use sundials_core::sundials_nonlinearsolver::SUNNonlinearSolver;
use sundials_core::sundials_nvector::N_Vector;
use sundials_core::sundials_types::*;
use sundials_core::sundials_utils::{sun_format_g, sunCombineFileAndLine};

/* =================================================================
 * Public constants (include/cvodes/cvodes.h)
 * =================================================================*/

/* lmm */
pub const CV_ADAMS: i32 = 1;
pub const CV_BDF: i32 = 2;

/* itask */
pub const CV_NORMAL: i32 = 1;
pub const CV_ONE_STEP: i32 = 2;

/* ism */
pub const CV_SIMULTANEOUS: i32 = 1;
pub const CV_STAGGERED: i32 = 2;
pub const CV_STAGGERED1: i32 = 3;

/* DQtype */
pub const CV_CENTERED: i32 = 1;
pub const CV_FORWARD: i32 = 2;

/* interp */
pub const CV_HERMITE: i32 = 1;
pub const CV_POLYNOMIAL: i32 = 2;

/* return values */
pub const CV_SUCCESS: i32 = 0;
pub const CV_TSTOP_RETURN: i32 = 1;
pub const CV_ROOT_RETURN: i32 = 2;

pub const CV_WARNING: i32 = 99;

pub const CV_TOO_MUCH_WORK: i32 = -1;
pub const CV_TOO_MUCH_ACC: i32 = -2;
pub const CV_ERR_FAILURE: i32 = -3;
pub const CV_CONV_FAILURE: i32 = -4;

pub const CV_LINIT_FAIL: i32 = -5;
pub const CV_LSETUP_FAIL: i32 = -6;
pub const CV_LSOLVE_FAIL: i32 = -7;
pub const CV_RHSFUNC_FAIL: i32 = -8;
pub const CV_FIRST_RHSFUNC_ERR: i32 = -9;
pub const CV_REPTD_RHSFUNC_ERR: i32 = -10;
pub const CV_UNREC_RHSFUNC_ERR: i32 = -11;
pub const CV_RTFUNC_FAIL: i32 = -12;
pub const CV_NLS_INIT_FAIL: i32 = -13;
pub const CV_NLS_SETUP_FAIL: i32 = -14;
pub const CV_CONSTR_FAIL: i32 = -15;
pub const CV_NLS_FAIL: i32 = -16;

pub const CV_MEM_FAIL: i32 = -20;
pub const CV_MEM_NULL: i32 = -21;
pub const CV_ILL_INPUT: i32 = -22;
pub const CV_NO_MALLOC: i32 = -23;
pub const CV_BAD_K: i32 = -24;
pub const CV_BAD_T: i32 = -25;
pub const CV_BAD_DKY: i32 = -26;
pub const CV_TOO_CLOSE: i32 = -27;
pub const CV_VECTOROP_ERR: i32 = -28;

pub const CV_NO_QUAD: i32 = -30;
pub const CV_QRHSFUNC_FAIL: i32 = -31;
pub const CV_FIRST_QRHSFUNC_ERR: i32 = -32;
pub const CV_REPTD_QRHSFUNC_ERR: i32 = -33;
pub const CV_UNREC_QRHSFUNC_ERR: i32 = -34;

pub const CV_NO_SENS: i32 = -40;
pub const CV_SRHSFUNC_FAIL: i32 = -41;
pub const CV_FIRST_SRHSFUNC_ERR: i32 = -42;
pub const CV_REPTD_SRHSFUNC_ERR: i32 = -43;
pub const CV_UNREC_SRHSFUNC_ERR: i32 = -44;

pub const CV_BAD_IS: i32 = -45;

pub const CV_NO_QUADSENS: i32 = -50;
pub const CV_QSRHSFUNC_FAIL: i32 = -51;
pub const CV_FIRST_QSRHSFUNC_ERR: i32 = -52;
pub const CV_REPTD_QSRHSFUNC_ERR: i32 = -53;
pub const CV_UNREC_QSRHSFUNC_ERR: i32 = -54;

pub const CV_CONTEXT_ERR: i32 = -55;

pub const CV_PROJ_MEM_NULL: i32 = -56;
pub const CV_PROJFUNC_FAIL: i32 = -57;
pub const CV_REPTD_PROJFUNC_ERR: i32 = -58;

pub const CV_BAD_TINTERP: i32 = -59;

pub const CV_UNRECOGNIZED_ERR: i32 = -99;

/* adjoint return values */
pub const CV_NO_ADJ: i32 = -101;
pub const CV_NO_FWD: i32 = -102;
pub const CV_NO_BCK: i32 = -103;
pub const CV_BAD_TB0: i32 = -104;
pub const CV_REIFWD_FAIL: i32 = -105;
pub const CV_FWD_FAIL: i32 = -106;
pub const CV_GETY_BADT: i32 = -107;

/* ------------------------------
 * User-Supplied Function Types
 * ------------------------------ */

pub type CVRhsFn =
    fn(t: sunrealtype, y: &N_Vector, ydot: &N_Vector, user_data: &mut Option<Box<dyn Any>>) -> i32;

pub type CVRootFn = fn(
    t: sunrealtype,
    y: &N_Vector,
    gout: &mut [sunrealtype],
    user_data: &mut Option<Box<dyn Any>>,
) -> i32;

pub type CVEwtFn = fn(y: &N_Vector, ewt: &N_Vector, user_data: &mut Option<Box<dyn Any>>) -> i32;

/* DEPRECATION NOTICE: this will be removed in v8.0.0 */
pub type CVMonitorFn = fn(cvode_mem: &CVodeMem, user_data: &mut Option<Box<dyn Any>>) -> i32;

pub type CVQuadRhsFn = fn(
    t: sunrealtype,
    y: &N_Vector,
    yQdot: &N_Vector,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32;

pub type CVSensRhsFn = fn(
    Ns: i32,
    t: sunrealtype,
    y: &N_Vector,
    ydot: &N_Vector,
    yS: &[N_Vector],
    ySdot: &[N_Vector],
    user_data: &mut Option<Box<dyn Any>>,
    tmp1: &N_Vector,
    tmp2: &N_Vector,
) -> i32;

pub type CVSensRhs1Fn = fn(
    Ns: i32,
    t: sunrealtype,
    y: &N_Vector,
    ydot: &N_Vector,
    iS: i32,
    yS: &N_Vector,
    ySdot: &N_Vector,
    user_data: &mut Option<Box<dyn Any>>,
    tmp1: &N_Vector,
    tmp2: &N_Vector,
) -> i32;

pub type CVQuadSensRhsFn = fn(
    Ns: i32,
    t: sunrealtype,
    y: &N_Vector,
    yS: &[N_Vector],
    yQdot: &N_Vector,
    yQSdot: &[N_Vector],
    user_data: &mut Option<Box<dyn Any>>,
    tmp: &N_Vector,
    tmpQ: &N_Vector,
) -> i32;

pub type CVRhsFnB = fn(
    t: sunrealtype,
    y: &N_Vector,
    yB: &N_Vector,
    yBdot: &N_Vector,
    user_dataB: &mut Option<Box<dyn Any>>,
) -> i32;

pub type CVRhsFnBS = fn(
    t: sunrealtype,
    y: &N_Vector,
    yS: &[N_Vector],
    yB: &N_Vector,
    yBdot: &N_Vector,
    user_dataB: &mut Option<Box<dyn Any>>,
) -> i32;

pub type CVQuadRhsFnB = fn(
    t: sunrealtype,
    y: &N_Vector,
    yB: &N_Vector,
    qBdot: &N_Vector,
    user_dataB: &mut Option<Box<dyn Any>>,
) -> i32;

pub type CVQuadRhsFnBS = fn(
    t: sunrealtype,
    y: &N_Vector,
    yS: &[N_Vector],
    yB: &N_Vector,
    qBdot: &N_Vector,
    user_dataB: &mut Option<Box<dyn Any>>,
) -> i32;

/* include/cvodes/cvodes_proj.h */
pub type CVProjFn = fn(
    t: sunrealtype,
    ycur: &N_Vector,
    corr: &N_Vector,
    epsProj: sunrealtype,
    err: Option<&N_Vector>,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32;

/* =================================================================
 * Internal constants (cvodes_impl.h)
 * =================================================================*/

/* Basic constants */
pub const ADAMS_Q_MAX: usize = 12; /* max value of q for lmm == ADAMS */
pub const BDF_Q_MAX: usize = 5; /* max value of q for lmm == BDF */
pub const Q_MAX: usize = ADAMS_Q_MAX; /* max value of q for either lmm */
pub const L_MAX: usize = Q_MAX + 1; /* max value of L for either lmm */
pub const NUM_TESTS: usize = 5; /* number of error test quantities */

pub const HMIN_DEFAULT: sunrealtype = 0.0;
pub const HMAX_INV_DEFAULT: sunrealtype = 0.0;
pub const MXHNIL_DEFAULT: i32 = 10;
pub const MXSTEP_DEFAULT: i64 = 500;

pub const MSBP_DEFAULT: i64 = 20; /* max steps between lsetup calls */
pub const DGMAX_LSETUP_DEFAULT: sunrealtype = 0.3; /* gamma threshold to call lsetup */

/* Step size change constants */
pub const ETA_MIN_FX_DEFAULT: sunrealtype = 0.0;
pub const ETA_MAX_FX_DEFAULT: sunrealtype = 1.5;
pub const ETA_MAX_FS_DEFAULT: sunrealtype = 10000.0;
pub const ETA_MAX_ES_DEFAULT: sunrealtype = 10.0;
pub const ETA_MAX_GS_DEFAULT: sunrealtype = 10.0;
pub const ETA_MIN_DEFAULT: sunrealtype = 0.1;
pub const ETA_MAX_EF_DEFAULT: sunrealtype = 0.2;
pub const ETA_MIN_EF_DEFAULT: sunrealtype = 0.1;
pub const ETA_CF_DEFAULT: sunrealtype = 0.25;
pub const SMALL_NST_DEFAULT: i64 = 10;
pub const SMALL_NEF_DEFAULT: i32 = 2;
pub const ONEPSM: sunrealtype = 1.000001;

/* Step size controller constants */
pub const ADDON: sunrealtype = 0.000001;
pub const BIAS1: sunrealtype = 6.0;
pub const BIAS2: sunrealtype = 6.0;
pub const BIAS3: sunrealtype = 10.0;

/* Order selection constants */
pub const LONG_WAIT: i32 = 10;

/* Failure limits */
pub const MXNCF: i32 = 10;
pub const MXNEF: i32 = 7;
pub const MXNEF1: i32 = 3;
pub const MAX_CONSTRAINT_FAILS: i32 = 10;

/* Control constants for lower-level functions used by cvStep */
pub const DO_ERROR_TEST: i32 = 2;
pub const PREDICT_AGAIN: i32 = 3;

pub const TRY_AGAIN: i32 = 5;
pub const FIRST_CALL: i32 = 6;
pub const PREV_CONV_FAIL: i32 = 7;
pub const PREV_PROJ_FAIL: i32 = 8;
pub const PREV_ERR_FAIL: i32 = 9;

pub const RHSFUNC_RECVR: i32 = 10;
pub const CONSTRFUNC_RECVR: i32 = 11;
pub const PROJFUNC_RECVR: i32 = 12;

pub const QRHSFUNC_RECVR: i32 = 13;
pub const SRHSFUNC_RECVR: i32 = 14;
pub const QSRHSFUNC_RECVR: i32 = 15;

/* nonlinear solver constants
   NLS_MAXCOR  maximum no. of corrector iterations for the nonlinear solver
   CRDOWN      constant used in estimation of the convergence rate (crate)
   RDIV        declare divergence if ratio delnrm/delnrm_p > RDIV          */
pub const NLS_MAXCOR: i32 = 3;
pub const CRDOWN: sunrealtype = 0.3;
pub const RDIV: sunrealtype = 2.0;

/* Constants for convfail (input to cv_lsetup) */
pub const CV_NO_FAILURES: i32 = 0;
pub const CV_FAIL_BAD_J: i32 = 1;
pub const CV_FAIL_OTHER: i32 = 2;

/* =================================================================
 * cvodes.c module-scope constants (fragment-file protocol: every
 * fragment of cvodes.c shares these single definitions)
 * =================================================================*/

/* CVODE Private Constants */
pub const ZERO: sunrealtype = 0.0; /* real 0.0     */
pub const TINY: sunrealtype = 1.0e-10; /* small number */
pub const PT1: sunrealtype = 0.1; /* real 0.1     */
pub const POINT2: sunrealtype = 0.2; /* real 0.2     */
pub const FOURTH: sunrealtype = 0.25; /* real 0.25    */
pub const HALF: sunrealtype = 0.5; /* real 0.5     */
pub const PT9: sunrealtype = 0.9; /* real 0.9     */
pub const ONE: sunrealtype = 1.0; /* real 1.0     */
pub const ONEPT5: sunrealtype = 1.50; /* real 1.5     */
pub const TWO: sunrealtype = 2.0; /* real 2.0     */
pub const THREE: sunrealtype = 3.0; /* real 3.0     */
pub const FOUR: sunrealtype = 4.0; /* real 4.0     */
pub const FIVE: sunrealtype = 5.0; /* real 5.0     */
pub const TWELVE: sunrealtype = 12.0; /* real 12.0    */
pub const HUNDRED: sunrealtype = 100.0; /* real 100.0   */

/* Control constants for lower-level rootfinding functions */
pub const RTFOUND: i32 = 1;
pub const CLOSERT: i32 = 3;

/* Control constants for sensitivity DQ */
pub const CENTERED1: i32 = 1;
pub const CENTERED2: i32 = 2;
pub const FORWARD1: i32 = 3;
pub const FORWARD2: i32 = 4;

/* Control constants for type of sensitivity RHS */
pub const CV_ONESENS: i32 = 1;
pub const CV_ALLSENS: i32 = 2;

/* Control constants for tolerances */
pub const CV_NN: i32 = 0;
pub const CV_SS: i32 = 1;
pub const CV_SV: i32 = 2;
pub const CV_WF: i32 = 3;
pub const CV_EE: i32 = 4;

/* Algorithmic constants */
pub const FUZZ_FACTOR: sunrealtype = 100.0;

pub const HLB_FACTOR: sunrealtype = 100.0;
pub const HUB_FACTOR: sunrealtype = 0.1;
pub const H_BIAS: sunrealtype = HALF;
pub const MAX_ITERS: i32 = 4;

pub const CORTES: sunrealtype = 0.1;

/* =============================================================================
 * Default Projection Constants (cvodes_proj_impl.h)
 * ===========================================================================*/

pub const PROJ_MAX_FAILS: i32 = 10;
pub const PROJ_EPS: sunrealtype = 0.1;
pub const PROJ_FAIL_ETA: sunrealtype = 0.25;

/* -----------------------------------------------------------------------------
 * Types : struct CVodeProjMemRec, CVodeProjMem (cvodes_proj_impl.h)
 * ---------------------------------------------------------------------------*/

pub struct CVodeProjMemRec {
    pub internal_proj: sunbooleantype, /* use the internal projection algorithm? */
    pub err_proj: sunbooleantype,      /* is error projection enabled?           */
    pub first_proj: sunbooleantype,    /* is this the first time we project?     */

    pub freq: i64,    /* projection frequency           */
    pub nstlprj: i64, /* step number of last projection */

    pub max_fails: i32, /* maximum number of projection failures */

    pub pfun: Option<CVProjFn>, /* function to perform projection */

    pub eps_proj: sunrealtype,  /* projection solve tolerance               */
    pub eta_pfail: sunrealtype, /* projection failure step reduction factor */

    pub nproj: i64,   /* number of projections performed */
    pub npfails: i64, /* number of projection failures   */
}

pub type CVodeProjMem = Box<CVodeProjMemRec>;

/* =================================================================
 * Sensitivity parameter array (C `sunrealtype* cv_p`)
 * =================================================================*/

/// Shared handle on the problem parameter array `p` of `f(t,y,p)`.
///
/// `CVodeSetSensParams` stores the caller's POINTER in `cv_mem->cv_p`
/// (`cvodes_io.c`: `cv_mem->cv_p = p;`), and the internal difference-quotient
/// sensitivity routines perturb `cv_p[which]` IN PLACE around each call to the
/// user's `f` / `fQ`. The user callback, reading the same memory through
/// `user_data`, therefore sees the perturbed parameter — that aliasing IS the
/// DQ mechanism. The port reproduces the shared pointer with the workspace
/// handle model: the caller keeps a clone of this `Rc` inside its user data
/// and hands an identical clone to `CVodeSetSensParams`.
///
/// Borrow discipline (same rule as every other RefCell in the port): never
/// hold a borrow of this cell across a user callback — write the perturbed
/// value, drop the borrow, call, then re-borrow to restore.
pub type SensParams = Rc<RefCell<Vec<sunrealtype>>>;

/* =================================================================
 * Main integrator memory block
 * =================================================================*/

pub struct CVodeMemRec {
    pub cv_sunctx: SUNContext,

    pub cv_uround: sunrealtype, /* machine unit roundoff */

    /*--------------------------
    Problem Specification Data
    --------------------------*/
    pub cv_f: Option<CVRhsFn>,              /* y' = f(t,y(t))                */
    pub cv_user_data: Option<Box<dyn Any>>, /* user pointer passed to f      */
    pub cv_lmm: i32,                        /* lmm = CV_ADAMS or CV_BDF      */
    pub cv_itol: i32,                       /* itol = CV_SS, CV_SV, CV_WF, CV_NN */

    pub cv_reltol: sunrealtype,          /* relative tolerance            */
    pub cv_Sabstol: sunrealtype,         /* scalar absolute tolerance     */
    pub cv_Vabstol: Option<N_Vector>,    /* vector absolute tolerance     */
    pub cv_atolmin0: sunbooleantype,     /* flag: min(abstol) = 0         */
    pub cv_user_efun: sunbooleantype,    /* SUNTRUE if user sets efun     */
    pub cv_efun: Option<CVEwtFn>,        /* function to set ewt           */
    pub cv_e_data: Option<Box<dyn Any>>, /* user pointer passed to efun   */

    /*-----------------------
    Quadrature Related Data
    -----------------------*/
    pub cv_quadr: sunbooleantype, /* SUNTRUE if integrating quadratures            */

    pub cv_fQ: Option<CVQuadRhsFn>, /* q' = fQ(t, y(t))                              */

    pub cv_errconQ: sunbooleantype, /* SUNTRUE if quadrs. are included in error test */

    pub cv_itolQ: i32,                 /* itolQ = CV_SS or CV_SV                        */
    pub cv_reltolQ: sunrealtype,       /* relative tolerance for quadratures            */
    pub cv_SabstolQ: sunrealtype,      /* scalar absolute tolerance for quadratures     */
    pub cv_VabstolQ: Option<N_Vector>, /* vector absolute tolerance for quadratures     */
    pub cv_atolQmin0: sunbooleantype,  /* flag indicating that min(abstolQ) = 0         */

    /*------------------------
    Sensitivity Related Data
    ------------------------*/
    pub cv_sensi: sunbooleantype, /* SUNTRUE if computing sensitivities           */

    pub cv_Ns: i32, /* Number of sensitivities                      */

    pub cv_ism: i32, /* ism = SIMULTANEOUS or STAGGERED              */

    pub cv_fS: Option<CVSensRhsFn>,       /* fS = (df/dy)*yS + (df/dp)              */
    pub cv_fS1: Option<CVSensRhs1Fn>,     /* fS1 = (df/dy)*yS_i + (df/dp)           */
    pub cv_fS_data: Option<Box<dyn Any>>, /* data pointer passed to fS (holds a
                                          CVodeMem clone when cv_fSDQ)            */
    pub cv_fSDQ: sunbooleantype,          /* SUNTRUE if using internal DQ functions */
    pub cv_ifS: i32,                      /* ifS = ALLSENS or ONESENS               */

    pub cv_p: Option<SensParams>,  /* parameters in f(t,y,p) (SHARED with the
                                   caller, as C shares the pointer; None = C
                                   NULL)                                    */
    pub cv_pbar: Vec<sunrealtype>, /* scale factors for parameters               */
    pub cv_plist: Vec<i32>,        /* list of sensitivities                      */
    pub cv_DQtype: i32,            /* central/forward finite differences         */
    pub cv_DQrhomax: sunrealtype,  /* cut-off value for separate/simultaneous FD */

    pub cv_errconS: sunbooleantype, /* SUNTRUE if yS are considered in err. control */

    pub cv_itolS: i32,
    pub cv_reltolS: sunrealtype,           /* relative tolerance for sensitivities      */
    pub cv_SabstolS: Vec<sunrealtype>,     /* scalar absolute tolerances for sensi.     */
    pub cv_VabstolS: Vec<N_Vector>,        /* vector absolute tolerances for sensi.     */
    pub cv_atolSmin0: Vec<sunbooleantype>, /* flags indicating that min(abstolS[i]) = 0 */

    /*-----------------------------------
    Quadrature Sensitivity Related Data
    -----------------------------------*/
    pub cv_quadr_sensi: sunbooleantype, /* SUNTRUE if computing sensitivities of quadrs. */

    pub cv_fQS: Option<CVQuadSensRhsFn>,   /* fQS = (dfQ/dy)*yS + (dfQ/dp)           */
    pub cv_fQS_data: Option<Box<dyn Any>>, /* data pointer passed to fQS (holds a
                                           CVodeMem clone when cv_fQSDQ)           */
    pub cv_fQSDQ: sunbooleantype,          /* SUNTRUE if using internal DQ functions */

    pub cv_errconQS: sunbooleantype, /* SUNTRUE if yQS are considered in err. con.   */

    pub cv_itolQS: i32,
    pub cv_reltolQS: sunrealtype,           /* relative tolerance for yQS                 */
    pub cv_SabstolQS: Vec<sunrealtype>,     /* scalar absolute tolerances for yQS         */
    pub cv_VabstolQS: Vec<N_Vector>,        /* vector absolute tolerances for yQS         */
    pub cv_atolQSmin0: Vec<sunbooleantype>, /* flags indicating that min(abstolQS[i]) = 0 */

    /*-----------------------
    Nordsieck History Array
    -----------------------*/
    pub cv_zn: [Option<N_Vector>; L_MAX],

    /*-------------------
    Vectors of length N
    -------------------*/
    pub cv_ewt: Option<N_Vector>, /* error weight vector             */
    pub cv_y: Option<N_Vector>,   /* temp storage; aliases the user's
                                  yout during CVode (copy-back!)   */
    pub cv_acor: Option<N_Vector>,
    pub cv_tempv: Option<N_Vector>,
    pub cv_ftemp: Option<N_Vector>,
    pub cv_vtemp1: Option<N_Vector>,
    pub cv_vtemp2: Option<N_Vector>,
    pub cv_vtemp3: Option<N_Vector>,

    /*--------------------------
    Quadrature Related Vectors
    --------------------------*/
    pub cv_znQ: [Option<N_Vector>; L_MAX], /* Nordsieck arrays for quadratures          */
    pub cv_ewtQ: Option<N_Vector>,         /* error weight vector for quadratures       */
    pub cv_yQ: Option<N_Vector>,           /* Unlike y, yQ is not allocated by the user */
    pub cv_acorQ: Option<N_Vector>,        /* acorQ = yQ_n(m) - yQ_n(0)                 */
    pub cv_tempvQ: Option<N_Vector>,       /* temporary storage vector (~ tempv)        */

    /*---------------------------
    Sensitivity Related Vectors
    ---------------------------*/
    pub cv_znS: [Vec<N_Vector>; L_MAX], /* Nordsieck arrays for sensitivities     */
    pub cv_ewtS: Vec<N_Vector>,         /* error weight vectors for sensitivities */
    pub cv_yS: Vec<N_Vector>,           /* yS=yS0 (allocated by the user)         */
    pub cv_acorS: Vec<N_Vector>,        /* acorS = yS_n(m) - yS_n(0)              */
    pub cv_tempvS: Vec<N_Vector>,       /* temporary storage vector (~ tempv)     */
    pub cv_ftempS: Vec<N_Vector>,       /* temporary storage vector (~ ftemp)     */

    pub cv_stgr1alloc: sunbooleantype, /* Did we allocate ncfS1, ncfnS1, and nniS1? */

    /*--------------------------------------
    Quadrature Sensitivity Related Vectors
    --------------------------------------*/
    pub cv_znQS: [Vec<N_Vector>; L_MAX], /* Nordsieck arrays for quadr. sensitivities   */
    pub cv_ewtQS: Vec<N_Vector>,         /* error weight vectors for sensitivities      */
    pub cv_yQS: Vec<N_Vector>,           /* Unlike yS, yQS is not allocated by the user */
    pub cv_acorQS: Vec<N_Vector>,        /* acorQS = yQS_n(m) - yQS_n(0)                */
    pub cv_tempvQS: Vec<N_Vector>,       /* temporary storage vector (~ tempv)          */
    pub cv_ftempQ: Option<N_Vector>,     /* temporary storage vector (~ ftemp)          */

    /*-----------------
    Tstop information
    -----------------*/
    pub cv_tstopset: sunbooleantype,
    pub cv_tstopinterp: sunbooleantype,
    pub cv_tstop: sunrealtype,

    /*---------
    Step Data
    ---------*/
    pub cv_q: i32,      /* current order                   */
    pub cv_qprime: i32, /* order to be used on next step   */
    pub cv_next_q: i32,
    pub cv_qwait: i32, /* steps to wait before order change */
    pub cv_L: i32,     /* L = q + 1                       */

    pub cv_hin: sunrealtype,
    pub cv_h: sunrealtype,
    pub cv_hprime: sunrealtype,
    pub cv_next_h: sunrealtype,
    pub cv_eta: sunrealtype,
    pub cv_hscale: sunrealtype,
    pub cv_tn: sunrealtype,
    pub cv_tretlast: sunrealtype,

    pub cv_tau: [sunrealtype; L_MAX + 1],
    pub cv_tq: [sunrealtype; NUM_TESTS + 1],
    pub cv_l: [sunrealtype; L_MAX],

    pub cv_rl1: sunrealtype,
    pub cv_gamma: sunrealtype,
    pub cv_gammap: sunrealtype,
    pub cv_gamrat: sunrealtype,

    pub cv_crate: sunrealtype,        /* estimated corrector convergence rate        */
    pub cv_crateS: sunrealtype,       /* estimated corrector convergence rate (Stgr) */
    pub cv_delp: sunrealtype,         /* norm of previous nonlinear solver update    */
    pub cv_delnrm: sunrealtype,       /* norm of current nonlinear solver update     */
    pub cv_delnrmS: sunrealtype,      /* norm of current NLS update (Sens)           */
    pub cv_acnrm: sunrealtype,        /* | acor |                                    */
    pub cv_acnrmcur: sunbooleantype,  /* is | acor | current?                        */
    pub cv_acnrmQ: sunrealtype,       /* | acorQ |                                   */
    pub cv_acnrmS: sunrealtype,       /* | acorS |                                   */
    pub cv_acnrmScur: sunbooleantype, /* is | acorS | current?                       */
    pub cv_acnrmQS: sunrealtype,      /* | acorQS |                                  */
    pub cv_nlscoef: sunrealtype,      /* coefficient in nonlinear convergence test   */
    pub cv_ncfS1: Vec<i32>,           /* Array of Ns local counters for conv.
                                       * failures (used in CVStep for STAGGERED1)    */

    /*------
    Limits
    ------*/
    pub cv_qmax: i32,
    pub cv_mxstep: i64,
    pub cv_mxhnil: i32,
    pub cv_maxnef: i32,
    pub cv_maxncf: i32,

    pub cv_hmin: sunrealtype,
    pub cv_hmax_inv: sunrealtype,
    pub cv_etamax: sunrealtype,
    pub cv_eta_min_fx: sunrealtype,
    pub cv_eta_max_fx: sunrealtype,
    pub cv_eta_max_fs: sunrealtype,
    pub cv_eta_max_es: sunrealtype,
    pub cv_eta_max_gs: sunrealtype,
    pub cv_eta_min: sunrealtype,
    pub cv_eta_min_ef: sunrealtype,
    pub cv_eta_max_ef: sunrealtype,
    pub cv_eta_cf: sunrealtype,

    pub cv_small_nst: i64,
    pub cv_small_nef: i32,

    /*--------
    Counters
    --------*/
    pub cv_nst: i64, /* number of internal steps taken   */

    pub cv_nfe: i64,   /* number of f calls                */
    pub cv_nfQe: i64,  /* number of fQ calls               */
    pub cv_nfSe: i64,  /* number of fS calls               */
    pub cv_nfeS: i64,  /* number of f calls from sensi DQ  */
    pub cv_nfQSe: i64, /* number of fQS calls              */
    pub cv_nfQeS: i64, /* number of fQ calls from sensi DQ */

    pub cv_ncfn: i64,        /* number of corrector convergence failures    */
    pub cv_ncfnS: i64,       /* number of total sensi. corr. conv. failures */
    pub cv_ncfnS1: Vec<i64>, /* number of sensi. corrector conv. failures   */

    pub cv_nni: i64,        /* number of nonlinear iterations performed    */
    pub cv_nniS: i64,       /* number of total sensi. nonlinear iterations */
    pub cv_nniS1: Vec<i64>, /* number of sensi. nonlinear iterations       */

    pub cv_nnf: i64,        /* number of nonlinear convergence failures     */
    pub cv_nnfS: i64,       /* number of total sensi. nonlinear conv. fails */
    pub cv_nnfS1: Vec<i64>, /* number of sensi. nonlinear conv. fails       */

    pub cv_netf: i64,   /* number of error test failures               */
    pub cv_netfQ: i64,  /* number of quadr. error test failures        */
    pub cv_netfS: i64,  /* number of sensi. error test failures        */
    pub cv_netfQS: i64, /* number of quadr. sensi. error test failures */

    pub cv_nsetups: i64,  /* number of setup calls                      */
    pub cv_nsetupsS: i64, /* number of setup calls due to sensitivities */

    pub cv_nhnil: i32, /* number of messages issued to the user that
                       t + h == t for the next internal step        */

    /*----------------
    Step size ratios
    ----------------*/
    pub cv_etaqm1: sunrealtype,
    pub cv_etaq: sunrealtype,
    pub cv_etaqp1: sunrealtype,

    /*------------------
    Space requirements
    ------------------*/
    pub cv_lrw1: sunindextype,
    pub cv_liw1: sunindextype,
    pub cv_lrw1Q: sunindextype,
    pub cv_liw1Q: sunindextype,
    pub cv_lrw: i64,
    pub cv_liw: i64,

    /*---------------------
    Nonlinear Solver Data
    ---------------------*/
    pub NLS: Option<SUNNonlinearSolver>, /* nonlinear solver object     */
    pub ownNLS: sunbooleantype,          /* flag indicating NLS ownership */

    pub NLSsim: Option<SUNNonlinearSolver>, /* NLS object for the simultaneous corrector */
    pub ownNLSsim: sunbooleantype,          /* flag indicating NLS ownership             */

    pub NLSstg: Option<SUNNonlinearSolver>, /* NLS object for the staggered corrector */
    pub ownNLSstg: sunbooleantype,          /* flag indicating NLS ownership          */

    pub NLSstg1: Option<SUNNonlinearSolver>, /* NLS object for the staggered1 corrector */
    pub ownNLSstg1: sunbooleantype,          /* flag indicating NLS ownership           */
    pub sens_solve_idx: i32,                 /* index of the current staggered1 solve   */
    pub nnip: i64,                           /* previous total number of iterations     */

    pub sens_solve: sunbooleantype, /* flag indicating if the current solve is a
                                    staggered or staggered1 sensitivity solve  */
    pub nls_f: Option<CVRhsFn>,     /* f(t,y(t)) used in the nonlinear solver    */
    pub convfail: i32,              /* flag to indicate when a Jacobian update
                                    may be needed                              */

    /* The following vectors are NVector wrappers for use with the simultaneous
    and staggered corrector methods:

      Simultaneous: zn0Sim  = [cv_zn[0], cv_znS[0]]
                    ycorSim = [cv_acor,  cv_acorS]
                    ewtSim  = [cv_ewt,   cv_ewtS]

      Staggered: zn0Stg  = cv_znS[0]
                 ycorStg = cv_acorS
                 ewtStg  = cv_ewtS
    */
    pub zn0Sim: Option<N_Vector>,
    pub ycorSim: Option<N_Vector>,
    pub ewtSim: Option<N_Vector>,
    pub zn0Stg: Option<N_Vector>,
    pub ycorStg: Option<N_Vector>,
    pub ewtStg: Option<N_Vector>,

    /* flags indicating if vector wrappers for the simultaneous and staggered
    correctors have been allocated */
    pub simMallocDone: sunbooleantype,
    pub stgMallocDone: sunbooleantype,

    /*------------------
    Linear Solver Data
    ------------------*/
    pub cv_linit: Option<fn(cv_mem: &CVodeMem) -> i32>,
    pub cv_lreinit: Option<fn(cv_mem: &CVodeMem) -> i32>,
    pub cv_lsetup: Option<
        fn(
            cv_mem: &CVodeMem,
            convfail: i32,
            ypred: &N_Vector,
            fpred: &N_Vector,
            jcurPtr: &mut sunbooleantype,
            vtemp1: &N_Vector,
            vtemp2: &N_Vector,
            vtemp3: &N_Vector,
        ) -> i32,
    >,
    pub cv_lsolve: Option<
        fn(
            cv_mem: &CVodeMem,
            b: &N_Vector,
            weight: &N_Vector,
            ycur: &N_Vector,
            fcur: &N_Vector,
        ) -> i32,
    >,
    pub cv_lfree: Option<fn(cv_mem: &CVodeMem) -> i32>,

    /* Linear Solver specific memory */
    pub cv_lmem: Option<Box<dyn Any>>,
    pub cv_msbp: i64,
    pub cv_dgmax_lsetup: sunrealtype,
    pub cv_forceSetup: sunbooleantype, /* flag to request a call to the setup routine */

    /*------------
    Saved Values
    ------------*/
    pub cv_qu: i32,
    pub cv_nstlp: i64,
    pub cv_h0u: sunrealtype,
    pub cv_hu: sunrealtype,
    pub cv_saved_tq5: sunrealtype,
    pub cv_jcur: sunbooleantype,
    pub cv_convfail: i32, /* flag storing previous solver failure mode */
    pub cv_tolsf: sunrealtype,
    pub cv_qmax_alloc: i32,   /* value of qmax used when allocating mem     */
    pub cv_qmax_allocQ: i32,  /* qmax used when allocating quad. mem        */
    pub cv_qmax_allocS: i32,  /* qmax used when allocating sensi. mem       */
    pub cv_qmax_allocQS: i32, /* qmax used when allocating quad. sensi. mem */
    pub cv_indx_acor: i32,

    /*--------------------------------------------------------------------
    Flags turned ON by CVodeInit, CVodeSensMalloc, and CVodeQuadMalloc
    and read by CVodeReInit, CVodeSensReInit, and CVodeQuadReInit
    --------------------------------------------------------------------*/
    pub cv_VabstolMallocDone: sunbooleantype,
    pub cv_MallocDone: sunbooleantype,
    pub cv_constraintsMallocDone: sunbooleantype,

    pub cv_VabstolQMallocDone: sunbooleantype,
    pub cv_QuadMallocDone: sunbooleantype,

    pub cv_VabstolSMallocDone: sunbooleantype,
    pub cv_SabstolSMallocDone: sunbooleantype,
    pub cv_SensMallocDone: sunbooleantype,

    pub cv_VabstolQSMallocDone: sunbooleantype,
    pub cv_SabstolQSMallocDone: sunbooleantype,
    pub cv_QuadSensMallocDone: sunbooleantype,

    /*-------------------------------------------
    User access function
    -------------------------------------------*/
    pub cv_monitorfun: Option<CVMonitorFn>,
    pub cv_monitor_interval: i64,

    /*-------------------------
    Stability Limit Detection
    -------------------------*/
    pub cv_sldeton: sunbooleantype,
    pub cv_ssdat: [[sunrealtype; 4]; 6],
    pub cv_nscon: i32,
    pub cv_nor: i64,

    /*----------------
    Rootfinding Data
    ----------------*/
    pub cv_gfun: Option<CVRootFn>,
    pub cv_nrtfn: i32,
    pub cv_iroots: Vec<i32>,
    pub cv_rootdir: Vec<i32>,
    pub cv_tlo: sunrealtype,
    pub cv_thi: sunrealtype,
    pub cv_trout: sunrealtype,
    pub cv_glo: Vec<sunrealtype>,
    pub cv_ghi: Vec<sunrealtype>,
    pub cv_grout: Vec<sunrealtype>,
    pub cv_ttol: sunrealtype,
    pub cv_irfnd: i32,
    pub cv_nge: i64,
    pub cv_gactive: Vec<sunbooleantype>,
    pub cv_mxgnull: i32,

    /*---------------------------
    Inequality Constraints Data
    ---------------------------*/
    pub cv_constraints: Option<N_Vector>,
    pub constraint_corrections: i64,
    pub constraint_fails: i64,
    pub max_constraint_fails: i32,

    /*---------------
    Projection Data
    ---------------*/
    pub proj_mem: Option<CVodeProjMem>,
    pub proj_enabled: sunbooleantype,
    pub proj_applied: sunbooleantype,
    pub proj_p: [sunrealtype; L_MAX],

    /*-----------------------
    Fused Vector Operations
    -----------------------*/
    pub cv_cvals: Vec<sunrealtype>, /* array of scalars */
    pub cv_Xvecs: Vec<N_Vector>,    /* array of vectors */
    pub cv_Zvecs: Vec<N_Vector>,    /* array of vectors */

    /*----------------
    Resizing History
    ----------------*/
    pub first_step_after_resize: sunbooleantype,

    /*------------------------
    Adjoint sensitivity data
    ------------------------*/
    pub cv_adj: sunbooleantype, /* SUNTRUE if performing ASA */

    pub cv_adj_mem: Option<CVadjMem>, /* Pointer to adjoint memory structure */

    pub cv_adjMallocDone: sunbooleantype,
}

pub type CVodeMem = Rc<RefCell<CVodeMemRec>>;

impl CVodeMemRec {
    /// All-zero/None baseline (the C `malloc` block before `CVodeCreate`
    /// assigns its explicit defaults; every field the C code reads is
    /// explicitly set there, so the baseline values are never observable).
    pub fn zeroed(sunctx: SUNContext) -> CVodeMemRec {
        CVodeMemRec {
            cv_sunctx: sunctx,
            cv_uround: 0.0,
            cv_f: None,
            cv_user_data: None,
            cv_lmm: 0,
            cv_itol: 0,
            cv_reltol: 0.0,
            cv_Sabstol: 0.0,
            cv_Vabstol: None,
            cv_atolmin0: SUNFALSE,
            cv_user_efun: SUNFALSE,
            cv_efun: None,
            cv_e_data: None,
            cv_quadr: SUNFALSE,
            cv_fQ: None,
            cv_errconQ: SUNFALSE,
            cv_itolQ: 0,
            cv_reltolQ: 0.0,
            cv_SabstolQ: 0.0,
            cv_VabstolQ: None,
            cv_atolQmin0: SUNFALSE,
            cv_sensi: SUNFALSE,
            cv_Ns: 0,
            cv_ism: 0,
            cv_fS: None,
            cv_fS1: None,
            cv_fS_data: None,
            cv_fSDQ: SUNFALSE,
            cv_ifS: 0,
            cv_p: None,
            cv_pbar: Vec::new(),
            cv_plist: Vec::new(),
            cv_DQtype: 0,
            cv_DQrhomax: 0.0,
            cv_errconS: SUNFALSE,
            cv_itolS: 0,
            cv_reltolS: 0.0,
            cv_SabstolS: Vec::new(),
            cv_VabstolS: Vec::new(),
            cv_atolSmin0: Vec::new(),
            cv_quadr_sensi: SUNFALSE,
            cv_fQS: None,
            cv_fQS_data: None,
            cv_fQSDQ: SUNFALSE,
            cv_errconQS: SUNFALSE,
            cv_itolQS: 0,
            cv_reltolQS: 0.0,
            cv_SabstolQS: Vec::new(),
            cv_VabstolQS: Vec::new(),
            cv_atolQSmin0: Vec::new(),
            cv_zn: Default::default(),
            cv_ewt: None,
            cv_y: None,
            cv_acor: None,
            cv_tempv: None,
            cv_ftemp: None,
            cv_vtemp1: None,
            cv_vtemp2: None,
            cv_vtemp3: None,
            cv_znQ: Default::default(),
            cv_ewtQ: None,
            cv_yQ: None,
            cv_acorQ: None,
            cv_tempvQ: None,
            cv_znS: Default::default(),
            cv_ewtS: Vec::new(),
            cv_yS: Vec::new(),
            cv_acorS: Vec::new(),
            cv_tempvS: Vec::new(),
            cv_ftempS: Vec::new(),
            cv_stgr1alloc: SUNFALSE,
            cv_znQS: Default::default(),
            cv_ewtQS: Vec::new(),
            cv_yQS: Vec::new(),
            cv_acorQS: Vec::new(),
            cv_tempvQS: Vec::new(),
            cv_ftempQ: None,
            cv_tstopset: SUNFALSE,
            cv_tstopinterp: SUNFALSE,
            cv_tstop: 0.0,
            cv_q: 0,
            cv_qprime: 0,
            cv_next_q: 0,
            cv_qwait: 0,
            cv_L: 0,
            cv_hin: 0.0,
            cv_h: 0.0,
            cv_hprime: 0.0,
            cv_next_h: 0.0,
            cv_eta: 0.0,
            cv_hscale: 0.0,
            cv_tn: 0.0,
            cv_tretlast: 0.0,
            cv_tau: [0.0; L_MAX + 1],
            cv_tq: [0.0; NUM_TESTS + 1],
            cv_l: [0.0; L_MAX],
            cv_rl1: 0.0,
            cv_gamma: 0.0,
            cv_gammap: 0.0,
            cv_gamrat: 0.0,
            cv_crate: 0.0,
            cv_crateS: 0.0,
            cv_delp: 0.0,
            cv_delnrm: 0.0,
            cv_delnrmS: 0.0,
            cv_acnrm: 0.0,
            cv_acnrmcur: SUNFALSE,
            cv_acnrmQ: 0.0,
            cv_acnrmS: 0.0,
            cv_acnrmScur: SUNFALSE,
            cv_acnrmQS: 0.0,
            cv_nlscoef: 0.0,
            cv_ncfS1: Vec::new(),
            cv_qmax: 0,
            cv_mxstep: 0,
            cv_mxhnil: 0,
            cv_maxnef: 0,
            cv_maxncf: 0,
            cv_hmin: 0.0,
            cv_hmax_inv: 0.0,
            cv_etamax: 0.0,
            cv_eta_min_fx: 0.0,
            cv_eta_max_fx: 0.0,
            cv_eta_max_fs: 0.0,
            cv_eta_max_es: 0.0,
            cv_eta_max_gs: 0.0,
            cv_eta_min: 0.0,
            cv_eta_min_ef: 0.0,
            cv_eta_max_ef: 0.0,
            cv_eta_cf: 0.0,
            cv_small_nst: 0,
            cv_small_nef: 0,
            cv_nst: 0,
            cv_nfe: 0,
            cv_nfQe: 0,
            cv_nfSe: 0,
            cv_nfeS: 0,
            cv_nfQSe: 0,
            cv_nfQeS: 0,
            cv_ncfn: 0,
            cv_ncfnS: 0,
            cv_ncfnS1: Vec::new(),
            cv_nni: 0,
            cv_nniS: 0,
            cv_nniS1: Vec::new(),
            cv_nnf: 0,
            cv_nnfS: 0,
            cv_nnfS1: Vec::new(),
            cv_netf: 0,
            cv_netfQ: 0,
            cv_netfS: 0,
            cv_netfQS: 0,
            cv_nsetups: 0,
            cv_nsetupsS: 0,
            cv_nhnil: 0,
            cv_etaqm1: 0.0,
            cv_etaq: 0.0,
            cv_etaqp1: 0.0,
            cv_lrw1: 0,
            cv_liw1: 0,
            cv_lrw1Q: 0,
            cv_liw1Q: 0,
            cv_lrw: 0,
            cv_liw: 0,
            NLS: None,
            ownNLS: SUNFALSE,
            NLSsim: None,
            ownNLSsim: SUNFALSE,
            NLSstg: None,
            ownNLSstg: SUNFALSE,
            NLSstg1: None,
            ownNLSstg1: SUNFALSE,
            sens_solve_idx: 0,
            nnip: 0,
            sens_solve: SUNFALSE,
            nls_f: None,
            convfail: 0,
            zn0Sim: None,
            ycorSim: None,
            ewtSim: None,
            zn0Stg: None,
            ycorStg: None,
            ewtStg: None,
            simMallocDone: SUNFALSE,
            stgMallocDone: SUNFALSE,
            cv_linit: None,
            cv_lreinit: None,
            cv_lsetup: None,
            cv_lsolve: None,
            cv_lfree: None,
            cv_lmem: None,
            cv_msbp: 0,
            cv_dgmax_lsetup: 0.0,
            cv_forceSetup: SUNFALSE,
            cv_qu: 0,
            cv_nstlp: 0,
            cv_h0u: 0.0,
            cv_hu: 0.0,
            cv_saved_tq5: 0.0,
            cv_jcur: SUNFALSE,
            cv_convfail: 0,
            cv_tolsf: 0.0,
            cv_qmax_alloc: 0,
            cv_qmax_allocQ: 0,
            cv_qmax_allocS: 0,
            cv_qmax_allocQS: 0,
            cv_indx_acor: 0,
            cv_VabstolMallocDone: SUNFALSE,
            cv_MallocDone: SUNFALSE,
            cv_constraintsMallocDone: SUNFALSE,
            cv_VabstolQMallocDone: SUNFALSE,
            cv_QuadMallocDone: SUNFALSE,
            cv_VabstolSMallocDone: SUNFALSE,
            cv_SabstolSMallocDone: SUNFALSE,
            cv_SensMallocDone: SUNFALSE,
            cv_VabstolQSMallocDone: SUNFALSE,
            cv_SabstolQSMallocDone: SUNFALSE,
            cv_QuadSensMallocDone: SUNFALSE,
            cv_monitorfun: None,
            cv_monitor_interval: 0,
            cv_sldeton: SUNFALSE,
            cv_ssdat: [[0.0; 4]; 6],
            cv_nscon: 0,
            cv_nor: 0,
            cv_gfun: None,
            cv_nrtfn: 0,
            cv_iroots: Vec::new(),
            cv_rootdir: Vec::new(),
            cv_tlo: 0.0,
            cv_thi: 0.0,
            cv_trout: 0.0,
            cv_glo: Vec::new(),
            cv_ghi: Vec::new(),
            cv_grout: Vec::new(),
            cv_ttol: 0.0,
            cv_irfnd: 0,
            cv_nge: 0,
            cv_gactive: Vec::new(),
            cv_mxgnull: 0,
            cv_constraints: None,
            constraint_corrections: 0,
            constraint_fails: 0,
            max_constraint_fails: 0,
            proj_mem: None,
            proj_enabled: SUNFALSE,
            proj_applied: SUNFALSE,
            proj_p: [0.0; L_MAX],
            cv_cvals: Vec::new(),
            cv_Xvecs: Vec::new(),
            cv_Zvecs: Vec::new(),
            first_step_after_resize: SUNFALSE,
            cv_adj: SUNFALSE,
            cv_adj_mem: None,
            cv_adjMallocDone: SUNFALSE,
        }
    }
}

/* =================================================================
 * Adjoint module memory block (cvodes_impl.h)
 * =================================================================*/

/* -----------------------------------------------------------------
 * Types : struct CVckpntMemRec, CVckpntMem
 * -----------------------------------------------------------------
 * Information at a check point needed to 'hot' start cvodes. The C
 * intrusive list link `ck_next` is replaced by the enclosing
 * `CVadjMemRec::ck_mem` Vec (index 0 = C list head = most recent
 * checkpoint; C `ck = ck->ck_next` ≡ next Vec index).
 * -----------------------------------------------------------------*/

pub struct CVckpntMemRec {
    /* Integration limits */
    pub ck_t0: sunrealtype,
    pub ck_t1: sunrealtype,

    /* Nordsieck History Array */
    pub ck_zn: [Option<N_Vector>; L_MAX],

    /* Do we need to carry quadratures? */
    pub ck_quadr: sunbooleantype,

    /* Nordsieck History Array for quadratures */
    pub ck_znQ: [Option<N_Vector>; L_MAX],

    /* Do we need to carry sensitivities? */
    pub ck_sensi: sunbooleantype,

    /* number of sensitivities */
    pub ck_Ns: i32,

    /* Nordsieck History Array for sensitivities */
    pub ck_znS: [Vec<N_Vector>; L_MAX],

    /* Do we need to carry quadrature sensitivities? */
    pub ck_quadr_sensi: sunbooleantype,

    /* Nordsieck History Array for quadrature sensitivities */
    pub ck_znQS: [Vec<N_Vector>; L_MAX],

    /* Was ck_zn[qmax] allocated?
    ck_zqm = 0    - no
    ck_zqm = qmax - yes      */
    pub ck_zqm: i32,

    /* Step data */
    pub ck_nst: i64,
    pub ck_tretlast: sunrealtype,
    pub ck_q: i32,
    pub ck_qprime: i32,
    pub ck_qwait: i32,
    pub ck_L: i32,
    pub ck_gammap: sunrealtype,
    pub ck_h: sunrealtype,
    pub ck_hprime: sunrealtype,
    pub ck_hscale: sunrealtype,
    pub ck_eta: sunrealtype,
    pub ck_etamax: sunrealtype,
    pub ck_tau: [sunrealtype; L_MAX + 1],
    pub ck_tq: [sunrealtype; NUM_TESTS + 1],
    pub ck_l: [sunrealtype; L_MAX],

    /* Saved values */
    pub ck_saved_tq5: sunrealtype,
}

pub type CVckpntMem = Rc<RefCell<CVckpntMemRec>>;

impl CVckpntMemRec {
    /// All-zero/None baseline (mirrors the C malloc before explicit init).
    pub fn zeroed() -> CVckpntMemRec {
        CVckpntMemRec {
            ck_t0: 0.0,
            ck_t1: 0.0,
            ck_zn: Default::default(),
            ck_quadr: SUNFALSE,
            ck_znQ: Default::default(),
            ck_sensi: SUNFALSE,
            ck_Ns: 0,
            ck_znS: Default::default(),
            ck_quadr_sensi: SUNFALSE,
            ck_znQS: Default::default(),
            ck_zqm: 0,
            ck_nst: 0,
            ck_tretlast: 0.0,
            ck_q: 0,
            ck_qprime: 0,
            ck_qwait: 0,
            ck_L: 0,
            ck_gammap: 0.0,
            ck_h: 0.0,
            ck_hprime: 0.0,
            ck_hscale: 0.0,
            ck_eta: 0.0,
            ck_etamax: 0.0,
            ck_tau: [0.0; L_MAX + 1],
            ck_tq: [0.0; NUM_TESTS + 1],
            ck_l: [0.0; L_MAX],
            ck_saved_tq5: 0.0,
        }
    }
}

/* -----------------------------------------------------------------
 * Types for functions provided by an interpolation module
 * -----------------------------------------------------------------
 * cvaIMMallocFn: initializes the content field of the structures in
 *                the dt array
 * cvaIMFreeFn:   deallocates the content field of the structures in
 *                the dt array
 * cvaIMGetYFn:   returns the interpolated forward solution (a C NULL
 *                `yS` maps to an empty slice)
 * cvaIMStorePntFn: stores a new point in the structure d
 * -----------------------------------------------------------------*/

pub type cvaIMMallocFn = fn(cv_mem: &CVodeMem) -> sunbooleantype;
pub type cvaIMFreeFn = fn(cv_mem: &CVodeMem);
pub type cvaIMGetYFn =
    fn(cv_mem: &CVodeMem, t: sunrealtype, y: &N_Vector, yS: &[N_Vector]) -> i32;
pub type cvaIMStorePntFn = fn(cv_mem: &CVodeMem, d: &CVdtpntMem) -> i32;

/* -----------------------------------------------------------------
 * Type : struct CVdtpntMemRec
 * -----------------------------------------------------------------
 * Information at a data point needed to interpolate the solution of
 * forward simulations. `content` holds a `CVhermiteDataMemRec` or a
 * `CVpolynomialDataMemRec` BY VALUE depending on IMtype (C void*).
 * -----------------------------------------------------------------*/

pub struct CVdtpntMemRec {
    pub t: sunrealtype,                 /* time */
    pub content: Option<Box<dyn Any>>,  /* IMtype-dependent content */
}

pub type CVdtpntMem = Rc<RefCell<CVdtpntMemRec>>;

/* Data for cubic Hermite interpolation */
pub struct CVhermiteDataMemRec {
    pub y: Option<N_Vector>,
    pub yd: Option<N_Vector>,
    pub yS: Vec<N_Vector>,
    pub ySd: Vec<N_Vector>,
}

/* Data for polynomial interpolation */
pub struct CVpolynomialDataMemRec {
    pub y: Option<N_Vector>,
    pub yS: Vec<N_Vector>,
    pub order: i32,
}

/* -----------------------------------------------------------------
 * Type : struct CVodeBMemRec
 * -----------------------------------------------------------------
 * Information for ONE backward problem. The C intrusive list link
 * `cv_next` is replaced by the enclosing `CVadjMemRec::cvB_mem` Vec
 * (index 0 = C list head = most recently created backward problem).
 * -----------------------------------------------------------------*/

pub struct CVodeBMemRec {
    /* Index of this backward problem */
    pub cv_index: i32,

    /* Time at which the backward problem is initialized */
    pub cv_t0: sunrealtype,

    /* CVODES memory for this backward problem */
    pub cv_mem: Option<CVodeMem>,

    /* Flags to indicate that this backward problem's RHS or quad RHS
     * require forward sensitivities */
    pub cv_f_withSensi: sunbooleantype,
    pub cv_fQ_withSensi: sunbooleantype,

    /* Right hand side function for backward run */
    pub cv_f: Option<CVRhsFnB>,
    pub cv_fs: Option<CVRhsFnBS>,

    /* Right hand side quadrature function for backward run */
    pub cv_fQ: Option<CVQuadRhsFnB>,
    pub cv_fQs: Option<CVQuadRhsFnBS>,

    /* User user_data */
    pub cv_user_data: Option<Box<dyn Any>>,

    /* Memory block for a linear solver's interface to CVODEA */
    pub cv_lmem: Option<Box<dyn Any>>,

    /* Function to free any memory allocated by the linear solver */
    pub cv_lfree: Option<fn(cvB_mem: &CVodeBMem) -> i32>,

    /* Memory block for a preconditioner's module interface to CVODEA */
    pub cv_pmem: Option<Box<dyn Any>>,

    /* Function to free any memory allocated by the preconditioner module */
    pub cv_pfree: Option<fn(cvB_mem: &CVodeBMem) -> i32>,

    /* Time at which to extract solution / quadratures */
    pub cv_tout: sunrealtype,

    /* Workspace Nvector */
    pub cv_y: Option<N_Vector>,
}

pub type CVodeBMem = Rc<RefCell<CVodeBMemRec>>;

impl CVodeBMemRec {
    /// All-zero/None baseline (mirrors the C malloc before explicit init).
    pub fn zeroed() -> CVodeBMemRec {
        CVodeBMemRec {
            cv_index: 0,
            cv_t0: 0.0,
            cv_mem: None,
            cv_f_withSensi: SUNFALSE,
            cv_fQ_withSensi: SUNFALSE,
            cv_f: None,
            cv_fs: None,
            cv_fQ: None,
            cv_fQs: None,
            cv_user_data: None,
            cv_lmem: None,
            cv_lfree: None,
            cv_pmem: None,
            cv_pfree: None,
            cv_tout: 0.0,
            cv_y: None,
        }
    }
}

/* -----------------------------------------------------------------
 * Type : struct CVadjMemRec
 * -----------------------------------------------------------------
 * All information necessary for adjoint sensitivity analysis.
 * -----------------------------------------------------------------*/

pub struct CVadjMemRec {
    /* --------------------
     * Forward problem data
     * -------------------- */

    /* Integration interval */
    pub ca_tinitial: sunrealtype,
    pub ca_tfinal: sunrealtype,

    /* Flag for first call to CVodeF */
    pub ca_firstCVodeFcall: sunbooleantype,

    /* Flag if CVodeF was called with TSTOP */
    pub ca_tstopCVodeFcall: sunbooleantype,
    pub ca_tstopCVodeF: sunrealtype,

    /* Flag if CVodeF was called in CV_NORMAL_MODE and encountered a
    root after tout */
    pub ca_rootret: sunbooleantype,
    pub ca_troot: sunrealtype,

    /* ----------------------
     * Backward problems data
     * ---------------------- */

    /* Storage for backward problems (C linked list head = index 0) */
    pub cvB_mem: Vec<CVodeBMem>,

    /* Number of backward problems */
    pub ca_nbckpbs: i32,

    /* Address of current backward problem */
    pub ca_bckpbCrt: Option<CVodeBMem>,

    /* Flag for first call to CVodeB */
    pub ca_firstCVodeBcall: sunbooleantype,

    /* ----------------
     * Check point data
     * ---------------- */

    /* Storage for check point information (C linked list head = index 0,
    i.e. most recent checkpoint first, t0 checkpoint last) */
    pub ck_mem: Vec<CVckpntMem>,

    /* Number of check points */
    pub ca_nckpnts: i32,

    /* address of the check point structure for which data is available */
    pub ca_ckpntData: Option<CVckpntMem>,

    /* ------------------
     * Interpolation data
     * ------------------ */

    /* Number of steps between 2 check points */
    pub ca_nsteps: i64,

    /* Last index used in CVAfindIndex */
    pub ca_ilast: i64,

    /* Storage for data from forward runs */
    pub dt_mem: Vec<CVdtpntMem>,

    /* Actual number of data points in dt_mem (typically np=nsteps+1) */
    pub ca_np: i64,

    /* Interpolation type */
    pub ca_IMtype: i32,

    /* Functions set by the interpolation module */
    pub ca_IMmalloc: Option<cvaIMMallocFn>,
    pub ca_IMfree: Option<cvaIMFreeFn>,
    pub ca_IMstore: Option<cvaIMStorePntFn>, /* store a new interpolation point */
    pub ca_IMget: Option<cvaIMGetYFn>,       /* interpolate forward solution    */

    /* Flags controlling the interpolation module */
    pub ca_IMmallocDone: sunbooleantype,  /* IM initialized?              */
    pub ca_IMnewData: sunbooleantype,     /* new data available in dt_mem? */
    pub ca_IMstoreSensi: sunbooleantype,  /* store sensitivities?         */
    pub ca_IMinterpSensi: sunbooleantype, /* interpolate sensitivities?   */

    /* Workspace for the interpolation module */
    pub ca_Y: [Option<N_Vector>; L_MAX], /* pointers to zn[i]  */
    pub ca_YS: [Vec<N_Vector>; L_MAX],   /* pointers to znS[i] */
    pub ca_T: [sunrealtype; L_MAX],

    /* -------------------------------
     * Workspace for wrapper functions
     * ------------------------------- */
    pub ca_ytmp: Option<N_Vector>,

    pub ca_yStmp: Vec<N_Vector>,
}

pub type CVadjMem = Rc<RefCell<CVadjMemRec>>;

impl CVadjMemRec {
    /// All-zero/None baseline (mirrors the C malloc before explicit init).
    pub fn zeroed() -> CVadjMemRec {
        CVadjMemRec {
            ca_tinitial: 0.0,
            ca_tfinal: 0.0,
            ca_firstCVodeFcall: SUNFALSE,
            ca_tstopCVodeFcall: SUNFALSE,
            ca_tstopCVodeF: 0.0,
            ca_rootret: SUNFALSE,
            ca_troot: 0.0,
            cvB_mem: Vec::new(),
            ca_nbckpbs: 0,
            ca_bckpbCrt: None,
            ca_firstCVodeBcall: SUNFALSE,
            ck_mem: Vec::new(),
            ca_nckpnts: 0,
            ca_ckpntData: None,
            ca_nsteps: 0,
            ca_ilast: 0,
            dt_mem: Vec::new(),
            ca_np: 0,
            ca_IMtype: 0,
            ca_IMmalloc: None,
            ca_IMfree: None,
            ca_IMstore: None,
            ca_IMget: None,
            ca_IMmallocDone: SUNFALSE,
            ca_IMnewData: SUNFALSE,
            ca_IMstoreSensi: SUNFALSE,
            ca_IMinterpSensi: SUNFALSE,
            ca_Y: Default::default(),
            ca_YS: Default::default(),
            ca_T: [0.0; L_MAX],
            ca_ytmp: None,
            ca_yStmp: Vec::new(),
        }
    }
}

/* -----------------------------------------------------------------
 * CVadjCheckPointRec (public, include/cvodes/cvodes.h) — the C void*
 * checkpoint addresses map to checkpoint handles (Rc identity).
 * -----------------------------------------------------------------*/

pub struct CVadjCheckPointRec {
    pub my_addr: Option<CVckpntMem>,
    pub next_addr: Option<CVckpntMem>,
    pub t0: sunrealtype,
    pub t1: sunrealtype,
    pub nstep: i64,
    pub order: i32,
    pub step: sunrealtype,
}

/* =================================================================
 * High level error handler (relocated from cvodes.c; C varargs map to a
 * pre-formatted msg — call sites use the MSGCV_* builders below)
 * =================================================================*/

pub fn cvProcessError(
    cv_mem: Option<&CVodeMem>,
    error_code: i32,
    line: i32,
    func: &str,
    file: &str,
    msg: &str,
) {
    match cv_mem {
        None => {
            SUNGlobalFallbackErrHandler(line, func, file, msg, error_code);
        }
        Some(cv_mem) => {
            let sunctx = cv_mem.borrow().cv_sunctx.clone();

            if error_code == CV_WARNING {
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

            /* Clear the last error value */
            let _ = SUNContext_GetLastError(&sunctx);
        }
    }
}

/* =================================================================
 * Error messages (cvodes_impl.h). Parameter-less messages are consts;
 * parameterized ones are builders producing the exact C expansion.
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
pub const MSGCV_NO_MEM: &str = "cvode_mem = NULL illegal.";
pub const MSGCV_CVMEM_FAIL: &str = "Allocation of cvode_mem failed.";
pub const MSGCV_MEM_FAIL: &str = "A memory request failed.";
pub const MSGCV_BAD_LMM: &str =
    "Illegal value for lmm. The legal values are CV_ADAMS and CV_BDF.";
pub const MSGCV_NULL_SUNCTX: &str = "sunctx = NULL illegal.";
pub const MSGCV_NO_MALLOC: &str = "Attempt to call before CVodeInit.";
pub const MSGCV_NEG_MAXORD: &str = "maxord <= 0 illegal.";
pub const MSGCV_BAD_MAXORD: &str = "Illegal attempt to increase maximum method order.";
pub const MSGCV_SET_SLDET: &str =
    "Attempt to use stability limit detection with the CV_ADAMS method illegal.";
pub const MSGCV_NEG_HMIN: &str = "hmin < 0 illegal.";
pub const MSGCV_NEG_HMAX: &str = "hmax < 0 illegal.";
pub const MSGCV_BAD_HMIN_HMAX: &str = "Inconsistent step size limits: hmin > hmax.";
pub const MSGCV_BAD_RELTOL: &str = "reltol < 0 illegal.";
pub const MSGCV_BAD_ABSTOL: &str = "abstol has negative component(s) (illegal).";
pub const MSGCV_NULL_ABSTOL: &str = "abstol = NULL illegal.";
pub const MSGCV_NULL_Y0: &str = "y0 = NULL illegal.";
pub const MSGCV_Y0_FAIL_CONSTR: &str = "y0 fails to satisfy constraints.";
pub const MSGCV_BAD_ISM_CONSTR: &str = "Constraints can not be enforced while forward \
                                        sensitivity is used with simultaneous method";
pub const MSGCV_NULL_F: &str = "f = NULL illegal.";
pub const MSGCV_NULL_G: &str = "g = NULL illegal.";
pub const MSGCV_BAD_NVECTOR: &str = "A required vector operation is not implemented.";
pub const MSGCV_BAD_CONSTR: &str = "Illegal values in constraints vector.";
pub const MSGCV_BAD_K: &str = "Illegal value for k.";
pub const MSGCV_NULL_DKY: &str = "dky = NULL illegal.";

pub fn MSGCV_BAD_T(t: sunrealtype, t0: sunrealtype, t1: sunrealtype) -> String {
    format!("Illegal value for t.{}", MSG_TIME_INT(t, t0, t1))
}

pub const MSGCV_NO_ROOT: &str = "Rootfinding was not initialized.";
pub const MSGCV_NLS_INIT_FAIL: &str = "The nonlinear solver's init routine failed.";

pub const MSGCV_NO_QUAD: &str = "Quadrature integration not activated.";
pub const MSGCV_BAD_ITOLQ: &str =
    "Illegal value for itolQ. The legal values are CV_SS and CV_SV.";
pub const MSGCV_NULL_ABSTOLQ: &str = "abstolQ = NULL illegal.";
pub const MSGCV_BAD_RELTOLQ: &str = "reltolQ < 0 illegal.";
pub const MSGCV_BAD_ABSTOLQ: &str = "abstolQ has negative component(s) (illegal).";

pub const MSGCV_SENSINIT_2: &str = "Sensitivity analysis already initialized.";
pub const MSGCV_NO_SENSI: &str = "Forward sensitivity analysis not activated.";
pub const MSGCV_BAD_ITOLS: &str =
    "Illegal value for itolS. The legal values are CV_SS, CV_SV, and CV_EE.";
pub const MSGCV_NULL_ABSTOLS: &str = "abstolS = NULL illegal.";
pub const MSGCV_BAD_RELTOLS: &str = "reltolS < 0 illegal.";
pub const MSGCV_BAD_ABSTOLS: &str = "abstolS has negative component(s) (illegal).";
pub const MSGCV_BAD_PBAR: &str = "pbar has zero component(s) (illegal).";
pub const MSGCV_BAD_PLIST: &str = "plist has negative component(s) (illegal).";
pub const MSGCV_BAD_NS: &str = "NS <= 0 illegal.";
pub const MSGCV_NULL_YS0: &str = "yS0 = NULL illegal.";
pub const MSGCV_BAD_ISM: &str = "Illegal value for ism. Legal values are: CV_SIMULTANEOUS, \
                                 CV_STAGGERED and CV_STAGGERED1.";
pub const MSGCV_BAD_IFS: &str =
    "Illegal value for ifS. Legal values are: CV_ALLSENS and CV_ONESENS.";
pub const MSGCV_BAD_ISM_IFS: &str = "Illegal ism = CV_STAGGERED1 for CVodeSensInit.";
pub const MSGCV_BAD_IS: &str = "Illegal value for is.";
pub const MSGCV_NULL_DKYA: &str = "dkyA = NULL illegal.";
pub const MSGCV_BAD_DQTYPE: &str =
    "Illegal value for DQtype. Legal values are: CV_CENTERED and CV_FORWARD.";
pub const MSGCV_BAD_DQRHO: &str = "DQrhomax < 0 illegal.";

pub const MSGCV_BAD_ITOLQS: &str =
    "Illegal value for itolQS. The legal values are CV_SS, CV_SV, and CV_EE.";
pub const MSGCV_NULL_ABSTOLQS: &str = "abstolQS = NULL illegal.";
pub const MSGCV_BAD_RELTOLQS: &str = "reltolQS < 0 illegal.";
pub const MSGCV_BAD_ABSTOLQS: &str = "abstolQS has negative component(s) (illegal).";
pub const MSGCV_NO_QUADSENSI: &str =
    "Forward sensitivity analysis for quadrature variables not activated.";
pub const MSGCV_NULL_YQS0: &str = "yQS0 = NULL illegal.";

/* CVode Error Messages */
pub const MSGCV_NO_TOL: &str = "No integration tolerances have been specified.";
pub const MSGCV_LSOLVE_NULL: &str = "The linear solver's solve routine is NULL.";
pub const MSGCV_YOUT_NULL: &str = "yout = NULL illegal.";
pub const MSGCV_TRET_NULL: &str = "tret = NULL illegal.";
pub const MSGCV_BAD_EWT: &str = "Initial ewt has component(s) equal to zero (illegal).";

pub fn MSGCV_EWT_NOW_BAD(t: sunrealtype) -> String {
    format!("At {}, a component of ewt has become <= 0.", MSG_TIME(t))
}

pub const MSGCV_BAD_ITASK: &str = "Illegal value for itask.";
pub const MSGCV_BAD_H0: &str = "h0 and tout - t0 inconsistent.";

pub fn MSGCV_BAD_TOUT(tout: sunrealtype) -> String {
    format!(
        "Trouble interpolating at {}. tout too far back in direction of integration",
        MSG_TIME_TOUT(tout)
    )
}

pub const MSGCV_EWT_FAIL: &str = "The user-provide EwtSet function failed.";

pub fn MSGCV_EWT_NOW_FAIL(t: sunrealtype) -> String {
    format!("At {}, the user-provide EwtSet function failed.", MSG_TIME(t))
}

pub const MSGCV_LINIT_FAIL: &str = "The linear solver's init routine failed.";
pub const MSGCV_HNIL_DONE: &str = "The above warning has been issued mxhnil times and will not be \
                                   issued again for this problem.";
pub const MSGCV_TOO_CLOSE: &str = "tout too close to t0 to start integration.";

pub fn MSGCV_MAX_STEPS(t: sunrealtype) -> String {
    format!("At {}, mxstep steps taken before reaching tout.", MSG_TIME(t))
}

pub fn MSGCV_TOO_MUCH_ACC(t: sunrealtype) -> String {
    format!("At {}, too much accuracy requested.", MSG_TIME(t))
}

pub fn MSGCV_HNIL(t: sunrealtype, h: sunrealtype) -> String {
    format!(
        "Internal {} are such that t + h = t on the next step. The solver will continue anyway.",
        MSG_TIME_H(t, h)
    )
}

pub fn MSGCV_ERR_FAILS(t: sunrealtype, h: sunrealtype) -> String {
    format!(
        "At {}, the error test failed repeatedly or with |h| = hmin.",
        MSG_TIME_H(t, h)
    )
}

pub fn MSGCV_CONV_FAILS(t: sunrealtype, h: sunrealtype) -> String {
    format!(
        "At {}, the corrector convergence test failed repeatedly or with |h| = hmin.",
        MSG_TIME_H(t, h)
    )
}

pub fn MSGCV_SETUP_FAILED(t: sunrealtype) -> String {
    format!(
        "At {}, the setup routine failed in an unrecoverable manner.",
        MSG_TIME(t)
    )
}

pub fn MSGCV_SOLVE_FAILED(t: sunrealtype) -> String {
    format!(
        "At {}, the solve routine failed in an unrecoverable manner.",
        MSG_TIME(t)
    )
}

pub fn MSGCV_FAILED_CONSTR(t: sunrealtype) -> String {
    format!("At {}, unable to satisfy inequality constraints.", MSG_TIME(t))
}

pub fn MSGCV_RHSFUNC_FAILED(t: sunrealtype) -> String {
    format!(
        "At {}, the right-hand side routine failed in an unrecoverable manner.",
        MSG_TIME(t)
    )
}

pub fn MSGCV_RHSFUNC_UNREC(t: sunrealtype) -> String {
    format!(
        "At {}, the right-hand side failed in a recoverable manner, but no recovery is possible.",
        MSG_TIME(t)
    )
}

pub fn MSGCV_RHSFUNC_REPTD(t: sunrealtype) -> String {
    format!(
        "At {} repeated recoverable right-hand side function errors.",
        MSG_TIME(t)
    )
}

pub const MSGCV_RHSFUNC_FIRST: &str = "The right-hand side routine failed at the first call.";

pub fn MSGCV_RTFUNC_FAILED(t: sunrealtype) -> String {
    format!(
        "At {}, the rootfinding routine failed in an unrecoverable manner.",
        MSG_TIME(t)
    )
}

pub fn MSGCV_CLOSE_ROOTS(t: sunrealtype) -> String {
    format!("Root found at and very near {}.", MSG_TIME(t))
}

pub fn MSGCV_BAD_TSTOP(tstop: sunrealtype, t: sunrealtype) -> String {
    format!(
        "The value {} is behind current {} in the direction of integration.",
        MSG_TIME_TSTOP(tstop),
        MSG_TIME(t)
    )
}

pub const MSGCV_INACTIVE_ROOTS: &str = "At the end of the first step, there are still some root \
                                        functions identically 0. This warning will not be issued \
                                        again.";

pub fn MSGCV_NLS_SETUP_FAILED(t: sunrealtype) -> String {
    format!("At {}, the nonlinear solver setup failed unrecoverably.", MSG_TIME(t))
}

pub fn MSGCV_NLS_INPUT_NULL(t: sunrealtype) -> String {
    format!("At {}, the nonlinear solver was passed a NULL input.", MSG_TIME(t))
}

pub fn MSGCV_NLS_FAIL(t: sunrealtype) -> String {
    format!(
        "At {}, the nonlinear solver failed in an unrecoverable manner.",
        MSG_TIME(t)
    )
}

/* CVode Projection Error Messages */
pub const MSG_CV_MEM_NULL: &str = "cvode_mem = NULL illegal.";
pub const MSG_CV_MEM_FAIL: &str = "A memory request failed.";
pub const MSG_CV_PROJ_MEM_NULL: &str = "proj_mem = NULL illegal.";

pub fn MSG_CV_PROJFUNC_FAIL(t: sunrealtype) -> String {
    format!(
        "At {} the projection function failed with an unrecoverable error.",
        MSG_TIME(t)
    )
}

pub fn MSG_CV_REPTD_PROJFUNC_ERR(t: sunrealtype) -> String {
    format!(
        "At {} the projection function had repeated recoverable errors.",
        MSG_TIME(t)
    )
}

/* Quadrature CVode Error Messages */
pub const MSGCV_NO_TOLQ: &str =
    "No integration tolerances for quadrature variables have been specified.";
pub const MSGCV_BAD_EWTQ: &str = "Initial ewtQ has component(s) equal to zero (illegal).";

pub fn MSGCV_EWTQ_NOW_BAD(t: sunrealtype) -> String {
    format!("At {}, a component of ewtQ has become <= 0.", MSG_TIME(t))
}

pub fn MSGCV_QRHSFUNC_FAILED(t: sunrealtype) -> String {
    format!(
        "At {}, the quadrature right-hand side routine failed in an unrecoverable manner.",
        MSG_TIME(t)
    )
}

pub fn MSGCV_QRHSFUNC_UNREC(t: sunrealtype) -> String {
    format!(
        "At {}, the quadrature right-hand side failed in a recoverable manner, but no recovery \
         is possible.",
        MSG_TIME(t)
    )
}

pub fn MSGCV_QRHSFUNC_REPTD(t: sunrealtype) -> String {
    format!(
        "At {} repeated recoverable quadrature right-hand side function errors.",
        MSG_TIME(t)
    )
}

pub const MSGCV_QRHSFUNC_FIRST: &str =
    "The quadrature right-hand side routine failed at the first call.";

/* Sensitivity CVode Error Messages */
pub const MSGCV_NO_TOLS: &str =
    "No integration tolerances for sensitivity variables have been specified.";
pub const MSGCV_NULL_P: &str = "p = NULL when using internal DQ for sensitivity RHS illegal.";
pub const MSGCV_BAD_EWTS: &str = "Initial ewtS has component(s) equal to zero (illegal).";

pub fn MSGCV_EWTS_NOW_BAD(t: sunrealtype) -> String {
    format!("At {}, a component of ewtS has become <= 0.", MSG_TIME(t))
}

pub fn MSGCV_SRHSFUNC_FAILED(t: sunrealtype) -> String {
    format!(
        "At {}, the sensitivity right-hand side routine failed in an unrecoverable manner.",
        MSG_TIME(t)
    )
}

pub fn MSGCV_SRHSFUNC_UNREC(t: sunrealtype) -> String {
    format!(
        "At {}, the sensitivity right-hand side failed in a recoverable manner, but no recovery \
         is possible.",
        MSG_TIME(t)
    )
}

pub fn MSGCV_SRHSFUNC_REPTD(t: sunrealtype) -> String {
    format!(
        "At {} repeated recoverable sensitivity right-hand side function errors.",
        MSG_TIME(t)
    )
}

pub const MSGCV_SRHSFUNC_FIRST: &str =
    "The sensitivity right-hand side routine failed at the first call.";

/* Quadrature Sensitivity CVode Error Messages */
pub const MSGCV_NULL_FQ: &str = "CVODES is expected to use DQ to evaluate the RHS of quad. \
                                 sensi., but quadratures were not initialized.";
pub const MSGCV_NO_TOLQS: &str =
    "No integration tolerances for quadrature sensitivity variables have been specified.";
pub const MSGCV_BAD_EWTQS: &str = "Initial ewtQS has component(s) equal to zero (illegal).";

pub fn MSGCV_EWTQS_NOW_BAD(t: sunrealtype) -> String {
    format!("At {}, a component of ewtQS has become <= 0.", MSG_TIME(t))
}

pub fn MSGCV_QSRHSFUNC_FAILED(t: sunrealtype) -> String {
    format!(
        "At {}, the quadrature sensitivity right-hand side routine failed in an unrecoverable \
         manner.",
        MSG_TIME(t)
    )
}

pub fn MSGCV_QSRHSFUNC_UNREC(t: sunrealtype) -> String {
    format!(
        "At {}, the quadrature sensitivity right-hand side failed in a recoverable manner, but \
         no recovery is possible.",
        MSG_TIME(t)
    )
}

pub fn MSGCV_QSRHSFUNC_REPTD(t: sunrealtype) -> String {
    format!(
        "At {} repeated recoverable quadrature sensitivity right-hand side function errors.",
        MSG_TIME(t)
    )
}

pub const MSGCV_QSRHSFUNC_FIRST: &str =
    "The quadrature sensitivity right-hand side routine failed at the first call.";

/* =================================================================
 * Adjoint error messages
 * =================================================================*/

pub const MSGCV_NO_ADJ: &str = "Illegal attempt to call before calling CVodeAdjMalloc.";
pub const MSGCV_BAD_STEPS: &str = "Steps nonpositive illegal.";
pub const MSGCV_BAD_INTERP: &str = "Illegal value for interp.";
pub const MSGCV_BAD_WHICH: &str = "Illegal value for which.";
pub const MSGCV_NO_BCK: &str = "No backward problems have been defined yet.";
pub const MSGCV_NO_FWD: &str = "Illegal attempt to call before calling CVodeF.";

pub fn MSGCV_BAD_TB0(which: i32) -> String {
    format!(
        "The initial time tB0 for problem {} is outside the interval over which the forward \
         problem was solved.",
        which
    )
}

pub const MSGCV_BAD_SENSI: &str = "At least one backward problem requires sensitivities, but \
                                   they were not stored for interpolation.";
pub const MSGCV_BAD_ITASKB: &str =
    "Illegal value for itaskB. Legal values are CV_NORMAL and CV_ONE_STEP.";
pub const MSGCV_BAD_TBOUT: &str = "The final time tBout is outside the interval over which the \
                                   forward problem was solved.";

pub fn MSGCV_BACK_ERROR(which: i32) -> String {
    format!("Error occurred while integrating backward problem # {}", which)
}

pub fn MSGCV_BAD_TINTERP(t: sunrealtype) -> String {
    format!("Bad t = {} for interpolation.", sun_format_g(t))
}

pub const MSGCV_WRONG_INTERP: &str =
    "This function cannot be called for the specified interp type.";
