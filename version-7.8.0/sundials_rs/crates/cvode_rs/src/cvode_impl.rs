//! Port of `src/cvode/cvode_impl.h` + the constants/typedefs of
//! `include/cvode/cvode.h`, plus `src/cvode/cvode_proj_impl.h` (folded
//! here because `cvode_impl.h` includes it and `CVodeMemRec` embeds the
//! projection memory).
//!
//! `cvProcessError` (defined in `cvode.c` upstream) is relocated here so
//! every cvode module shares one definition; C varargs map to a
//! pre-formatted `msg` (call sites use the `MSGCV_*` constants/builders
//! below). Parameterized messages are functions producing the exact
//! C `printf` expansion (`SUN_FORMAT_G` = `%.15g` via `sun_format_g`).
//!
//! Handle model: `CVodeMem = Rc<RefCell<CVodeMemRec>>`. Internal
//! functions take `&CVodeMem` and use granular borrows (never hold a
//! borrow across a callback, N_Vector op on user vectors, or
//! linear/nonlinear solver call — all can re-enter the mem).

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
 * Public constants (include/cvode/cvode.h)
 * =================================================================*/

/* lmm */
pub const CV_ADAMS: i32 = 1;
pub const CV_BDF: i32 = 2;

/* itask */
pub const CV_NORMAL: i32 = 1;
pub const CV_ONE_STEP: i32 = 2;

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

pub const CV_PROJ_MEM_NULL: i32 = -29;
pub const CV_PROJFUNC_FAIL: i32 = -30;
pub const CV_REPTD_PROJFUNC_ERR: i32 = -31;

pub const CV_CONTEXT_ERR: i32 = -32;

pub const CV_UNRECOGNIZED_ERR: i32 = -99;

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

/* include/cvode/cvode_proj.h */
pub type CVProjFn = fn(
    t: sunrealtype,
    ycur: &N_Vector,
    corr: &N_Vector,
    epsProj: sunrealtype,
    err: Option<&N_Vector>,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32;

/* =================================================================
 * Internal constants (cvode_impl.h)
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

/* Constants for convfail (input to cv_lsetup) */
pub const CV_NO_FAILURES: i32 = 0;
pub const CV_FAIL_BAD_J: i32 = 1;
pub const CV_FAIL_OTHER: i32 = 2;

/* =============================================================================
 * Default Projection Constants (cvode_proj_impl.h)
 * ===========================================================================*/

pub const PROJ_MAX_FAILS: i32 = 10;
pub const PROJ_EPS: sunrealtype = 0.1;
pub const PROJ_FAIL_ETA: sunrealtype = 0.25;

/* -----------------------------------------------------------------------------
 * Types : struct CVodeProjMemRec, CVodeProjMem (cvode_proj_impl.h)
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
 * Main integrator memory block
 * =================================================================*/

pub struct CVodeMemRec {
    pub cv_sunctx: SUNContext,

    pub cv_uround: sunrealtype, /* machine unit roundoff */

    /*--------------------------
    Problem Specification Data
    --------------------------*/
    pub cv_f: Option<CVRhsFn>,               /* y' = f(t,y(t))                */
    pub cv_user_data: Option<Box<dyn Any>>,  /* user pointer passed to f      */
    pub cv_lmm: i32,                         /* lmm = CV_ADAMS or CV_BDF      */
    pub cv_itol: i32,                        /* itol = CV_SS, CV_SV, CV_WF, CV_NN */

    pub cv_reltol: sunrealtype,              /* relative tolerance            */
    pub cv_Sabstol: sunrealtype,             /* scalar absolute tolerance     */
    pub cv_Vabstol: Option<N_Vector>,        /* vector absolute tolerance     */
    pub cv_atolmin0: sunbooleantype,         /* flag: min(abstol) = 0         */
    pub cv_user_efun: sunbooleantype,        /* SUNTRUE if user sets efun     */
    pub cv_efun: Option<CVEwtFn>,            /* function to set ewt           */
    pub cv_e_data: Option<Box<dyn Any>>,     /* user pointer passed to efun   */

    /*-----------------------
    Nordsieck History Array
    -----------------------*/
    pub cv_zn: [Option<N_Vector>; L_MAX],

    /*-------------------
    Vectors of length N
    -------------------*/
    pub cv_ewt: Option<N_Vector>,    /* error weight vector             */
    pub cv_y: Option<N_Vector>,      /* temp storage; aliases the user's
                                     yout during CVode (copy-back!)   */
    pub cv_acor: Option<N_Vector>,
    pub cv_tempv: Option<N_Vector>,
    pub cv_ftemp: Option<N_Vector>,
    pub cv_vtemp1: Option<N_Vector>,
    pub cv_vtemp2: Option<N_Vector>,
    pub cv_vtemp3: Option<N_Vector>,

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
    pub cv_qwait: i32,  /* steps to wait before order change */
    pub cv_L: i32,      /* L = q + 1                       */

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

    pub cv_crate: sunrealtype,
    pub cv_delp: sunrealtype,
    pub cv_delnrm: sunrealtype,
    pub cv_acnrm: sunrealtype,
    pub cv_acnrmcur: sunbooleantype,
    pub cv_nlscoef: sunrealtype,

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
    pub cv_nst: i64,
    pub cv_nfe: i64,
    pub cv_ncfn: i64,
    pub cv_nni: i64,
    pub cv_nnf: i64,
    pub cv_netf: i64,
    pub cv_nsetups: i64,
    pub cv_nhnil: i32,

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
    pub cv_lrw: i64,
    pub cv_liw: i64,

    /*---------------------
    Nonlinear Solver Data
    ---------------------*/
    pub NLS: Option<SUNNonlinearSolver>,
    pub ownNLS: sunbooleantype,
    pub nls_f: Option<CVRhsFn>,
    pub convfail: i32,

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

    /*------------
    Saved Values
    ------------*/
    pub cv_qu: i32,
    pub cv_nstlp: i64,
    pub cv_h0u: sunrealtype,
    pub cv_hu: sunrealtype,
    pub cv_saved_tq5: sunrealtype,
    pub cv_jcur: sunbooleantype,
    pub cv_tolsf: sunrealtype,
    pub cv_qmax_alloc: i32,
    pub cv_indx_acor: i32,

    /*--------------------------------------------------------------------
    Flags turned ON by CVodeInit and read by CVodeReInit
    --------------------------------------------------------------------*/
    pub cv_VabstolMallocDone: sunbooleantype,
    pub cv_MallocDone: sunbooleantype,

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
    pub cv_cvals: [sunrealtype; L_MAX],
    pub cv_Xvecs: Vec<N_Vector>, /* scratch handle array */

    pub cv_usefused: sunbooleantype,

    /*----------------
    Resizing History
    ----------------*/
    pub first_step_after_resize: sunbooleantype,
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
            cv_zn: Default::default(),
            cv_ewt: None,
            cv_y: None,
            cv_acor: None,
            cv_tempv: None,
            cv_ftemp: None,
            cv_vtemp1: None,
            cv_vtemp2: None,
            cv_vtemp3: None,
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
            cv_delp: 0.0,
            cv_delnrm: 0.0,
            cv_acnrm: 0.0,
            cv_acnrmcur: SUNFALSE,
            cv_nlscoef: 0.0,
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
            cv_ncfn: 0,
            cv_nni: 0,
            cv_nnf: 0,
            cv_netf: 0,
            cv_nsetups: 0,
            cv_nhnil: 0,
            cv_etaqm1: 0.0,
            cv_etaq: 0.0,
            cv_etaqp1: 0.0,
            cv_lrw1: 0,
            cv_liw1: 0,
            cv_lrw: 0,
            cv_liw: 0,
            NLS: None,
            ownNLS: SUNFALSE,
            nls_f: None,
            convfail: 0,
            cv_linit: None,
            cv_lreinit: None,
            cv_lsetup: None,
            cv_lsolve: None,
            cv_lfree: None,
            cv_lmem: None,
            cv_msbp: 0,
            cv_dgmax_lsetup: 0.0,
            cv_qu: 0,
            cv_nstlp: 0,
            cv_h0u: 0.0,
            cv_hu: 0.0,
            cv_saved_tq5: 0.0,
            cv_jcur: SUNFALSE,
            cv_tolsf: 0.0,
            cv_qmax_alloc: 0,
            cv_indx_acor: 0,
            cv_VabstolMallocDone: SUNFALSE,
            cv_MallocDone: SUNFALSE,
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
            cv_cvals: [0.0; L_MAX],
            cv_Xvecs: Vec::new(),
            cv_usefused: SUNFALSE,
            first_step_after_resize: SUNFALSE,
        }
    }
}

/* =================================================================
 * High level error handler (relocated from cvode.c; C varargs map to a
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
 * Error messages (cvode_impl.h). Parameter-less messages are consts;
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
