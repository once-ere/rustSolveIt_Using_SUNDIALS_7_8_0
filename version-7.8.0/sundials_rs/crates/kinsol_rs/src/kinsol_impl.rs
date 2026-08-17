//! Port of `src/kinsol/kinsol_impl.h` + the constants/typedefs of
//! `include/kinsol/kinsol.h`.
//!
//! `KINProcessError` and `KINPrintInfo` (defined in `kinsol.c` upstream)
//! are relocated here so every kinsol module shares one definition; C
//! varargs map to a pre-formatted `msg` (call sites use the `MSG_*` /
//! `INFO_*` constants/builders below). Parameterized messages are
//! functions producing the exact C `printf` expansion
//! (`SUN_FORMAT_E` = `% .15e` via `sun_format_e`, `SUN_FORMAT_G` =
//! `%.15g` via `sun_format_g`).
//!
//! Because `KINPrintInfo` lives here, the base `PRNT_*` keys (defined at
//! the top of `kinsol.c` upstream, keys 2..13 only when
//! `SUNDIALS_LOGGING_LEVEL >= INFO`) are hosted here as well;
//! `kinsol_ls_impl.h` adds its own module-local keys (`PRNT_NLI` = 101,
//! `PRNT_EPS` = 102) and formats (`INFO_NLI`, `INFO_EPS`) in
//! `kinsol_ls.rs`. Every `KINPrintInfo` call site in C is guarded by
//! `#if SUNDIALS_LOGGING_LEVEL >= SUNDIALS_LOGGING_INFO`, so at the
//! reference logging level (2) the calls compile away and are omitted at
//! translation time — no kinsol serial reference output contains logger
//! info lines (the `nni = ...` lines in e.g. `kinFerTron_dns.out` are the
//! examples' own `printf` statistics).
//!
//! Upstream `kinsol_impl.h` also declares `void KINInfoHandler(...)`,
//! which has no definition anywhere in the 7.8.0 sources (dead
//! declaration) — it is omitted, not stubbed.
//!
//! Handle model: `KINMem = Rc<RefCell<KINMemRec>>`. Internal functions
//! take `&KINMem` and use granular borrows (never hold a borrow across a
//! callback, N_Vector op on user vectors, or linear solver call — all
//! can re-enter the mem).

use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

use sundials_core::sundials_context::{SUNContext, SUNContext_GetLastError};
use sundials_core::sundials_errors::{SUNGlobalFallbackErrHandler, SUNHandleErrWithMsg};
use sundials_core::sundials_iterative::{SUNQRAddFn, SUNQRData};
use sundials_core::sundials_logger::{SUNLogger_QueueMsg, SUN_LOGLEVEL_INFO, SUN_LOGLEVEL_WARNING};
use sundials_core::sundials_nvector::N_Vector;
use sundials_core::sundials_types::*;
use sundials_core::sundials_utils::{sun_format_e, sun_format_g, sunCombineFileAndLine};

/* =================================================================
 * Public constants (include/kinsol/kinsol.h)
 * =================================================================*/

/* return values */
pub const KIN_SUCCESS: i32 = 0;
pub const KIN_INITIAL_GUESS_OK: i32 = 1;
pub const KIN_STEP_LT_STPTOL: i32 = 2;

pub const KIN_WARNING: i32 = 99;

pub const KIN_MEM_NULL: i32 = -1;
pub const KIN_ILL_INPUT: i32 = -2;
pub const KIN_NO_MALLOC: i32 = -3;
pub const KIN_MEM_FAIL: i32 = -4;
pub const KIN_LINESEARCH_NONCONV: i32 = -5;
pub const KIN_MAXITER_REACHED: i32 = -6;
pub const KIN_MXNEWT_5X_EXCEEDED: i32 = -7;
pub const KIN_LINESEARCH_BCFAIL: i32 = -8;
pub const KIN_LINSOLV_NO_RECOVERY: i32 = -9;
pub const KIN_LINIT_FAIL: i32 = -10;
pub const KIN_LSETUP_FAIL: i32 = -11;
pub const KIN_LSOLVE_FAIL: i32 = -12;
pub const KIN_SYSFUNC_FAIL: i32 = -13;
pub const KIN_FIRST_SYSFUNC_ERR: i32 = -14;
pub const KIN_REPTD_SYSFUNC_ERR: i32 = -15;
pub const KIN_VECTOROP_ERR: i32 = -16;
pub const KIN_CONTEXT_ERR: i32 = -17;
pub const KIN_DAMPING_FN_ERR: i32 = -18;
pub const KIN_DEPTH_FN_ERR: i32 = -19;

/* Anderson Acceleration Orthogonalization Choice */
pub const KIN_ORTH_MGS: i32 = 0;
pub const KIN_ORTH_ICWY: i32 = 1;
pub const KIN_ORTH_CGS2: i32 = 2;
pub const KIN_ORTH_DCGS2: i32 = 3;

/* Enumeration for eta choice */
pub const KIN_ETACHOICE1: i32 = 1;
pub const KIN_ETACHOICE2: i32 = 2;
pub const KIN_ETACONSTANT: i32 = 3;

/* Enumeration for global strategy */
pub const KIN_NONE: i32 = 0;
pub const KIN_LINESEARCH: i32 = 1;
pub const KIN_PICARD: i32 = 2;
pub const KIN_FP: i32 = 3;

/* ------------------------------
 * User-Supplied Function Types
 * ------------------------------ */

pub type KINSysFn =
    fn(uu: &N_Vector, fval: &N_Vector, user_data: &mut Option<Box<dyn Any>>) -> i32;

/// C `KINDampingFn`. `qt_fn_1d` is NULL (`None`) at the FP/Picard
/// non-accelerated call sites and `gamma` (= Q^T fv, length `depth`)
/// inside `AndersonAcc`.
pub type KINDampingFn = fn(
    iter: i64,
    u_val: &N_Vector,
    g_val: &N_Vector,
    qt_fn_1d: Option<&mut [sunrealtype]>,
    depth: i64,
    user_data: &mut Option<Box<dyn Any>>,
    damping_factor_ptr: &mut sunrealtype,
) -> i32;

/// C `KINDepthFn`. `remove_indices_1d` is NULL (`None`) at the single
/// upstream call site in `AndersonAcc`.
pub type KINDepthFn = fn(
    iter: i64,
    u_val: &N_Vector,
    g_val: &N_Vector,
    f_val: &N_Vector,
    df_1d: &[N_Vector],
    R_mat_1d: &mut [sunrealtype],
    depth: i64,
    user_data: &mut Option<Box<dyn Any>>,
    new_depth_ptr: &mut i64,
    remove_indices_1d: Option<&mut [sunbooleantype]>,
) -> i32;

/* =================================================================
 * Internal constants (kinsol_impl.h)
 * =================================================================*/

/* KINSOL default constants */
pub const MXITER_DEFAULT: i64 = 200;
pub const MXNBCF_DEFAULT: i64 = 10;
pub const MSBSET_DEFAULT: i64 = 10;
pub const MSBSET_SUB_DEFAULT: i64 = 5;

pub const OMEGA_MIN: sunrealtype = 0.00001;
pub const OMEGA_MAX: sunrealtype = 0.9;

/* =================================================================
 * Main solver memory block
 * =================================================================*/

pub struct KINMemRec {
    pub kin_sunctx: SUNContext,

    /// C `void* python` (function table for the excluded Python
    /// bindings; always NULL in this port — KINCreate zeroes it and
    /// nothing else touches it).
    pub python: Option<Box<dyn Any>>,

    pub kin_uround: sunrealtype, /* machine epsilon (or unit roundoff error) */

    /* problem specification data */
    pub kin_func: Option<KINSysFn>,          /* nonlinear system function implementation     */
    pub kin_user_data: Option<Box<dyn Any>>, /* work space available to func routine         */
    pub kin_fnormtol: sunrealtype,           /* stopping tolerance on L2-norm of function
                                             value                                        */
    pub kin_scsteptol: sunrealtype,          /* scaled step length tolerance                 */
    pub kin_globalstrategy: i32,             /* choices are KIN_NONE, KIN_LINESEARCH,
                                             KIN_PICARD and KIN_FP                        */
    pub kin_mxiter: i64,                     /* maximum number of nonlinear iterations       */
    pub kin_msbset: i64,                     /* maximum number of nonlinear iterations that
                                             may be performed between calls to the
                                             linear solver setup routine (lsetup)         */
    pub kin_msbset_sub: i64,                 /* subinterval length for residual monitoring   */
    pub kin_mxnbcf: i64,                     /* maximum number of beta condition failures    */
    pub kin_etaflag: i32,                    /* choices are KIN_ETACONSTANT, KIN_ETACHOICE1
                                             and KIN_ETACHOICE2                           */
    pub kin_noMinEps: sunbooleantype,        /* flag controlling whether or not the value
                                             of eps is bounded below                      */
    pub kin_constraintsSet: sunbooleantype,  /* flag indicating if constraints are being
                                             used                                         */
    pub kin_jacCurrent: sunbooleantype,      /* flag indicating if the Jacobian info. used
                                             by the linear solver is current              */
    pub kin_callForcingTerm: sunbooleantype, /* flag set if using either KIN_ETACHOICE1
                                             or KIN_ETACHOICE2                            */
    pub kin_noResMon: sunbooleantype,        /* flag indicating if the nonlinear residual
                                             monitoring scheme should be used             */
    pub kin_retry_nni: sunbooleantype,       /* flag indicating if nonlinear iteration
                                             should be retried (set by residual
                                             monitoring algorithm)                        */
    pub kin_update_fnorm_sub: sunbooleantype, /* flag indicating if the fnorm associated
                                              with the subinterval needs to be updated
                                              (set by residual monitoring algorithm)      */

    pub kin_mxnewtstep: sunrealtype,   /* maximum allowable scaled step length         */
    pub kin_mxnstepin: sunrealtype,    /* input (or preset) value for mxnewtstep       */
    pub kin_sqrt_relfunc: sunrealtype, /* relative error bound for func(u)             */
    pub kin_stepl: sunrealtype,        /* scaled length of current step                */
    pub kin_stepmul: sunrealtype,      /* step scaling factor                          */
    pub kin_eps: sunrealtype,          /* current value of eps                         */
    pub kin_eta: sunrealtype,          /* current value of eta                         */
    pub kin_eta_gamma: sunrealtype,    /* gamma value used in eta calculation
                                       (choice #2)                                   */
    pub kin_eta_alpha: sunrealtype,    /* alpha value used in eta calculation
                                       (choice #2)                                   */
    pub kin_noInitSetup: sunbooleantype, /* flag controlling whether or not the KINSol
                                         routine makes an initial call to the
                                         linear solver setup routine (lsetup)        */
    pub kin_sthrsh: sunrealtype,       /* threshold value for calling the linear
                                       solver setup routine                         */

    /* counters */
    pub kin_nni: i64,         /* number of nonlinear iterations               */
    pub kin_nfe: i64,         /* number of calls made to func routine         */
    pub kin_nnilset: i64,     /* value of nni counter when the linear solver
                              setup was last called                        */
    pub kin_nnilset_sub: i64, /* value of nni counter when the linear solver
                              setup was last called (subinterval)          */
    pub kin_nbcf: i64,        /* number of times the beta-condition could not
                              be met in KINLineSearch                      */
    pub kin_nbktrk: i64,      /* number of backtracks performed by
                              KINLineSearch                                */
    pub kin_ncscmx: i64,      /* number of consecutive steps of size
                              mxnewtstep taken                             */

    /* vectors */
    pub kin_uu: Option<N_Vector>,     /* solution vector/current iterate (initially
                                      contains initial guess, but holds approximate
                                      solution upon completion if no errors
                                      occurred); aliases the user's `u` during
                                      KINSol                                        */
    pub kin_unew: Option<N_Vector>,   /* next iterate (unew = uu+pp)                  */
    pub kin_fval: Option<N_Vector>,   /* vector containing result of nonlinear system
                                      function evaluated at a given iterate
                                      (fval = func(uu))                             */
    pub kin_gval: Option<N_Vector>,   /* vector containing result of the fixed point
                                      function evaluated at a given iterate; used
                                      in KIN_PICARD strategy only.
                                      (gval = uu - L^{-1}fval(uu))                  */
    pub kin_uscale: Option<N_Vector>, /* iterate scaling vector                       */
    pub kin_fscale: Option<N_Vector>, /* fval scaling vector                          */
    pub kin_pp: Option<N_Vector>,     /* incremental change vector (pp = unew-uu)     */
    pub kin_constraints: Option<N_Vector>, /* constraints vector                      */
    pub kin_vtemp1: Option<N_Vector>, /* scratch vector #1                            */
    pub kin_vtemp2: Option<N_Vector>, /* scratch vector #2                            */
    pub kin_vtemp3: Option<N_Vector>, /* scratch vector #3                            */

    /* fixed point and Picard options */
    pub kin_ret_newest: sunbooleantype, /* return the newest FP iteration     */
    pub kin_damping: sunbooleantype,    /* flag to apply damping in FP/Picard */
    pub kin_beta: sunrealtype,          /* damping parameter for FP/Picard    */

    /* space requirements for AA, Broyden and NLEN */
    pub kin_fold_aa: Option<N_Vector>, /* vector needed for AA, Broyden, and NLEN      */
    pub kin_gold_aa: Option<N_Vector>, /* vector needed for AA, Broyden, and NLEN      */
    pub kin_df_aa: Vec<N_Vector>,      /* vector array needed for AA, Broyden, and NLEN */
    pub kin_dg_aa: Vec<N_Vector>,      /* vector array needed for AA, Broyden and NLEN */
    pub kin_q_aa: Vec<N_Vector>,       /* vector array needed for AA                   */
    pub kin_beta_aa: sunrealtype,      /* beta damping parameter for AA                */
    pub kin_gamma_aa: Vec<sunrealtype>, /* array of size maa used in AA                */
    pub kin_R_aa: Vec<sunrealtype>,    /* array of size maa*maa used in AA             */
    pub kin_T_aa: Vec<sunrealtype>,    /* array of size maa*maa used in AA with ICWY MGS */
    pub kin_m_aa: i64,                 /* parameter for AA, Broyden or NLEN            */
    pub kin_m_aa_alloc: i64,           /* depth (m) used for AA memory allocations     */
    pub kin_delay_aa: i64,             /* number of iterations to delay AA             */
    pub kin_current_depth: i64,        /* current Anderson acceleration space size     */
    pub kin_damping_fn: Option<KINDampingFn>, /* function to determine the damping factor */
    pub kin_depth_fn: Option<KINDepthFn>, /* function to determine the depth with AA   */
    pub kin_orth_aa: i32,              /* parameter for AA determining orthogonalization
                                       routine
                                       0 - Modified Gram Schmidt (standard)
                                       1 - ICWY Modified Gram Schmidt (Bjorck)
                                       2 - CGS2 (Hernandez)
                                       3 - Delayed CGS2 (Hernandez)                  */
    pub kin_orth_aa_alloc: i64,        /* depth (m) used for orthogonalization memory
                                       allocations                                  */
    pub kin_qr_func: Option<SUNQRAddFn>, /* QRAdd function for AA orthogonalization   */
    pub kin_qr_data: Option<Box<SUNQRData>>, /* Additional parameters required for
                                             QRAdd routine set for AA               */
    pub kin_damping_aa: sunbooleantype, /* flag to apply damping in AA                 */
    pub kin_dot_prod_sb: sunbooleantype, /* use single buffer dot product              */
    pub kin_cv: Vec<sunrealtype>,      /* scalar array for fused vector operations     */
    pub kin_Xv: Vec<N_Vector>,         /* vector array for fused vector operations     */

    /* space requirements for vector storage */
    pub kin_lrw1: sunindextype, /* number of sunrealtype-sized memory blocks needed
                                for a single N_Vector                           */
    pub kin_liw1: sunindextype, /* number of int-sized memory blocks needed for
                                a single N_Vector                               */
    pub kin_lrw: i64,           /* total number of sunrealtype-sized memory blocks
                                needed for all KINSOL work vectors              */
    pub kin_liw: i64,           /* total number of int-sized memory blocks needed
                                for all KINSOL work vectors                     */

    /* linear solver data (function prototypes and lmem) */
    pub kin_linit: Option<fn(kin_mem: &KINMem) -> i32>,
    pub kin_lsetup: Option<fn(kin_mem: &KINMem) -> i32>,
    pub kin_lsolve: Option<
        fn(
            kin_mem: &KINMem,
            xx: &N_Vector,
            bb: &N_Vector,
            sJpnorm: &mut sunrealtype,
            sFdotJp: &mut sunrealtype,
        ) -> i32,
    >,
    pub kin_lfree: Option<fn(kin_mem: &KINMem) -> i32>,

    pub kin_inexact_ls: sunbooleantype, /* flag set by the linear solver module
                                        (in linit) indicating whether this is an
                                        iterative linear solver (SUNTRUE), or a
                                        direct linear solver (SUNFALSE)          */

    pub kin_lmem: Option<Box<dyn Any>>, /* pointer to linear solver memory block   */

    pub kin_fnorm: sunrealtype,   /* value of L2-norm of fscale*fval               */
    pub kin_f1norm: sunrealtype,  /* f1norm = 0.5*(fnorm)^2                        */
    pub kin_sFdotJp: sunrealtype, /* value of scaled F(u) vector (fscale*fval)
                                  dotted with scaled J(u)*pp vector (set by
                                  lsolve)                                        */
    pub kin_sJpnorm: sunrealtype, /* value of L2-norm of fscale*(J(u)*pp)
                                  (set by lsolve)                                */

    pub kin_fnorm_sub: sunrealtype,     /* value of L2-norm of fscale*fval (subinterval) */
    pub kin_eval_omega: sunbooleantype, /* flag indicating that omega must be evaluated. */
    pub kin_omega: sunrealtype,         /* constant value for real scalar used in test to
                                        determine if reduction of norm of nonlinear
                                        residual is sufficient. Unless a valid constant
                                        value is specified by the user, omega is
                                        estimated from omega_min and omega_max at each
                                        iteration.                                    */
    pub kin_omega_min: sunrealtype,     /* lower bound on omega                          */
    pub kin_omega_max: sunrealtype,     /* upper bound on omega                          */

    /*
     * -----------------------------------------------------------------
     * Note: The KINLineSearch subroutine scales the values of the
     * variables sFdotJp and sJpnorm by a factor rl (lambda) that is
     * chosen by the line search algorithm such that the scaled Newton
     * step satisfies the following conditions:
     *
     *  F(u_k+1) <= F(u_k) + alpha*(F(u_k)^T * J(u_k))*p*rl
     *
     *  F(u_k+1) >= F(u_k) + beta*(F(u_k)^T * J(u_k))*p*rl
     *
     * where alpha = 1.0e-4, beta = 0.9, u_k+1 = u_k + rl*p,
     * 0 < rl <= 1, J denotes the system Jacobian, and F represents
     * the nonlinear system function.
     * -----------------------------------------------------------------
     */
    pub kin_MallocDone: sunbooleantype, /* flag indicating if KINMalloc has been
                                        called yet                                    */
}

pub type KINMem = Rc<RefCell<KINMemRec>>;

impl KINMemRec {
    /// All-zero/None baseline (the C `malloc` + `memset(0)` block at the
    /// top of `KINCreate`, before it assigns its explicit defaults;
    /// every field the C code reads is explicitly set there or in
    /// KINInit/KINSolInit, so the baseline values are never observable).
    pub fn zeroed(sunctx: SUNContext) -> KINMemRec {
        KINMemRec {
            kin_sunctx: sunctx,
            python: None,
            kin_uround: 0.0,
            kin_func: None,
            kin_user_data: None,
            kin_fnormtol: 0.0,
            kin_scsteptol: 0.0,
            kin_globalstrategy: 0,
            kin_mxiter: 0,
            kin_msbset: 0,
            kin_msbset_sub: 0,
            kin_mxnbcf: 0,
            kin_etaflag: 0,
            kin_noMinEps: SUNFALSE,
            kin_constraintsSet: SUNFALSE,
            kin_jacCurrent: SUNFALSE,
            kin_callForcingTerm: SUNFALSE,
            kin_noResMon: SUNFALSE,
            kin_retry_nni: SUNFALSE,
            kin_update_fnorm_sub: SUNFALSE,
            kin_mxnewtstep: 0.0,
            kin_mxnstepin: 0.0,
            kin_sqrt_relfunc: 0.0,
            kin_stepl: 0.0,
            kin_stepmul: 0.0,
            kin_eps: 0.0,
            kin_eta: 0.0,
            kin_eta_gamma: 0.0,
            kin_eta_alpha: 0.0,
            kin_noInitSetup: SUNFALSE,
            kin_sthrsh: 0.0,
            kin_nni: 0,
            kin_nfe: 0,
            kin_nnilset: 0,
            kin_nnilset_sub: 0,
            kin_nbcf: 0,
            kin_nbktrk: 0,
            kin_ncscmx: 0,
            kin_uu: None,
            kin_unew: None,
            kin_fval: None,
            kin_gval: None,
            kin_uscale: None,
            kin_fscale: None,
            kin_pp: None,
            kin_constraints: None,
            kin_vtemp1: None,
            kin_vtemp2: None,
            kin_vtemp3: None,
            kin_ret_newest: SUNFALSE,
            kin_damping: SUNFALSE,
            kin_beta: 0.0,
            kin_fold_aa: None,
            kin_gold_aa: None,
            kin_df_aa: Vec::new(),
            kin_dg_aa: Vec::new(),
            kin_q_aa: Vec::new(),
            kin_beta_aa: 0.0,
            kin_gamma_aa: Vec::new(),
            kin_R_aa: Vec::new(),
            kin_T_aa: Vec::new(),
            kin_m_aa: 0,
            kin_m_aa_alloc: 0,
            kin_delay_aa: 0,
            kin_current_depth: 0,
            kin_damping_fn: None,
            kin_depth_fn: None,
            kin_orth_aa: 0,
            kin_orth_aa_alloc: 0,
            kin_qr_func: None,
            kin_qr_data: None,
            kin_damping_aa: SUNFALSE,
            kin_dot_prod_sb: SUNFALSE,
            kin_cv: Vec::new(),
            kin_Xv: Vec::new(),
            kin_lrw1: 0,
            kin_liw1: 0,
            kin_lrw: 0,
            kin_liw: 0,
            kin_linit: None,
            kin_lsetup: None,
            kin_lsolve: None,
            kin_lfree: None,
            kin_inexact_ls: SUNFALSE,
            kin_lmem: None,
            kin_fnorm: 0.0,
            kin_f1norm: 0.0,
            kin_sFdotJp: 0.0,
            kin_sJpnorm: 0.0,
            kin_fnorm_sub: 0.0,
            kin_eval_omega: SUNFALSE,
            kin_omega: 0.0,
            kin_omega_min: 0.0,
            kin_omega_max: 0.0,
            kin_MallocDone: SUNFALSE,
        }
    }
}

/* =================================================================
 * High level error handler (relocated from kinsol.c; C varargs map to a
 * pre-formatted msg — call sites use the MSG_* constants below or
 * literal strings where kinsol.c passes literals)
 * =================================================================*/

pub fn KINProcessError(
    kin_mem: Option<&KINMem>,
    error_code: i32,
    line: i32,
    func: &str,
    file: &str,
    msg: &str,
) {
    match kin_mem {
        None => {
            SUNGlobalFallbackErrHandler(line, func, file, msg, error_code);
        }
        Some(kin_mem) => {
            let sunctx = kin_mem.borrow().kin_sunctx.clone();

            if error_code == KIN_WARNING {
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
 * High level info handler (relocated from kinsol.c)
 * =================================================================*/

/* Keys for KINPrintInfo (upstream: top of kinsol.c; keys 2..13 exist
 * only when SUNDIALS_LOGGING_LEVEL >= INFO). kinsol_ls.rs defines its
 * own PRNT_NLI = 101 / PRNT_EPS = 102 module-locally. */
pub const PRNT_RETVAL: i32 = 1;
pub const PRNT_NNI: i32 = 2;
pub const PRNT_TOL: i32 = 3;
pub const PRNT_FMAX: i32 = 4;
pub const PRNT_PNORM: i32 = 5;
pub const PRNT_PNORM1: i32 = 6;
pub const PRNT_FNORM: i32 = 7;
pub const PRNT_LAM: i32 = 8;
pub const PRNT_ALPHA: i32 = 9;
pub const PRNT_BETA: i32 = 10;
pub const PRNT_ALPHABETA: i32 = 11;
pub const PRNT_ADJ: i32 = 12;
pub const PRNT_OTHER: i32 = 13;

/// C `KINPrintInfo` — composes the info message and queues it on the
/// context logger at INFO level (scope = `fname`, label = `"KINSOL"`).
///
/// C varargs map to the pre-formatted `msg` (call sites use the
/// `INFO_*` builders below). The C `PRNT_RETVAL` branch composes
/// `"<msgfmt(ret)> (<decoded flag name>)"` internally; that composition
/// lives in the [`INFO_RETVAL`] builder here, so every `info_code`
/// takes the single queue path below with byte-identical output.
/// `module` is unused in the C body as well (the label is hardcoded).
///
/// NOTE: every upstream call site is guarded by
/// `#if SUNDIALS_LOGGING_LEVEL >= SUNDIALS_LOGGING_INFO`, so at the
/// reference logging level (2) no calls exist — port agents omit the
/// guarded call sites at translation time.
pub fn KINPrintInfo(kin_mem: &KINMem, info_code: i32, module: &str, fname: &str, msg: &str) {
    /* C marks kin_mem/module unused (SUNDIALS_MAYBE_UNUSED) and reads
    info_code only to select the PRNT_RETVAL composition, which the
    INFO_RETVAL builder performs here. */
    let _ = (info_code, module);

    let sunctx = kin_mem.borrow().kin_sunctx.clone();

    /* Call QueueMsg directly rather than using the SUNLogInfo macro in
    order to use the passed in function name */
    let logger = sunctx.borrow().logger.clone();
    if let Some(logger) = logger {
        let _ = SUNLogger_QueueMsg(&logger, SUN_LOGLEVEL_INFO, fname, "KINSOL", msg);
    }
}

/// The `PRNT_RETVAL` decode switch inside C `KINPrintInfo` (`retstr`).
/// The C switch has no default: an unlisted value reads uninitialized
/// stack memory (UB) — mapped to a deterministic panic at the same site.
pub fn kinPrintInfoRetvalString(ret: i32) -> &'static str {
    match ret {
        KIN_SUCCESS => "KIN_SUCCESS",
        KIN_SYSFUNC_FAIL => "KIN_SYSFUNC_FAIL",
        KIN_REPTD_SYSFUNC_ERR => "KIN_REPTD_SYSFUNC_ERR",
        KIN_STEP_LT_STPTOL => "KIN_STEP_LT_STPTOL",
        KIN_LINESEARCH_NONCONV => "KIN_LINESEARCH_NONCONV",
        KIN_LINESEARCH_BCFAIL => "KIN_LINESEARCH_BCFAIL",
        KIN_MAXITER_REACHED => "KIN_MAXITER_REACHED",
        KIN_MXNEWT_5X_EXCEEDED => "KIN_MXNEWT_5X_EXCEEDED",
        KIN_LINSOLV_NO_RECOVERY => "KIN_LINSOLV_NO_RECOVERY",
        KIN_LSETUP_FAIL => "KIN_PRECONDSET_FAILURE",
        KIN_LSOLVE_FAIL => "KIN_PRECONDSOLVE_FAILURE",
        _ => panic!("KINPrintInfo: PRNT_RETVAL passed undecodable ret {ret} (uninitialized retstr in C)"),
    }
}

/* =================================================================
 * Error messages (kinsol_impl.h). All are parameter-less consts; the
 * few parameterized error texts in kinsol.c are call-site literals
 * there ("The damping function failed.", etc.) and stay module-local.
 * =================================================================*/

pub const MSG_MEM_FAIL: &str = "A memory request failed.";
pub const MSG_NO_MEM: &str = "kinsol_mem = NULL illegal.";
pub const MSG_NULL_SUNCTX: &str = "sunctx = NULL illegal.";
pub const MSG_BAD_NVECTOR: &str = "A required vector operation is not implemented.";
pub const MSG_FUNC_NULL: &str = "func = NULL illegal.";
pub const MSG_NO_MALLOC: &str = "Attempt to call before KINMalloc illegal.";

pub const MSG_BAD_MXITER: &str = "Illegal value for mxiter.";
pub const MSG_BAD_MSBSET: &str = "Illegal msbset < 0.";
pub const MSG_BAD_MSBSETSUB: &str = "Illegal msbsetsub < 0.";
pub const MSG_BAD_ETACHOICE: &str = "Illegal value for etachoice.";
pub const MSG_BAD_ETACONST: &str = "eta out of range.";
pub const MSG_BAD_GAMMA: &str = "gamma out of range.";
pub const MSG_BAD_ALPHA: &str = "alpha out of range.";
pub const MSG_BAD_MXNEWTSTEP: &str = "Illegal mxnewtstep < 0.";
pub const MSG_BAD_RELFUNC: &str = "relfunc < 0 illegal.";
pub const MSG_BAD_FNORMTOL: &str = "fnormtol < 0 illegal.";
pub const MSG_BAD_SCSTEPTOL: &str = "scsteptol < 0 illegal.";
pub const MSG_BAD_MXNBCF: &str = "mxbcf < 0 illegal.";
pub const MSG_BAD_CONSTRAINTS: &str = "Illegal values in constraints vector.";
pub const MSG_BAD_OMEGA: &str = "scalars < 0 illegal.";
pub const MSG_BAD_MAA: &str = "maa < 0 illegal.";
pub const MSG_BAD_ORTHAA: &str = "Illegal value for orthaa.";
pub const MSG_ZERO_MAA: &str = "maa = 0 illegal.";

pub const MSG_LSOLV_NO_MEM: &str = "The linear solver memory pointer is NULL.";
pub const MSG_UU_NULL: &str = "uu = NULL illegal.";
pub const MSG_BAD_GLSTRAT: &str = "Illegal value for global strategy.";
pub const MSG_BAD_USCALE: &str = "uscale = NULL illegal.";
pub const MSG_USCALE_NONPOSITIVE: &str = "uscale has nonpositive elements.";
pub const MSG_BAD_FSCALE: &str = "fscale = NULL illegal.";
pub const MSG_FSCALE_NONPOSITIVE: &str = "fscale has nonpositive elements.";
pub const MSG_CONSTRAINTS_NOTOK: &str =
    "Constraints not allowed with fixed point or Picard iterations";
pub const MSG_INITIAL_CNSTRNT: &str = "Initial guess does NOT meet constraints.";
pub const MSG_LINIT_FAIL: &str = "The linear solver's init routine failed.";

pub const MSG_SYSFUNC_FAILED: &str = "The system function failed in an unrecoverable manner.";
pub const MSG_SYSFUNC_FIRST: &str = "The system function failed at the first call.";
pub const MSG_LSETUP_FAILED: &str =
    "The linear solver's setup function failed in an unrecoverable manner.";
pub const MSG_LSOLVE_FAILED: &str =
    "The linear solver's solve function failed in an unrecoverable manner.";
pub const MSG_LINSOLV_NO_RECOVERY: &str = "The linear solver's solve function failed recoverably, \
                                           but the Jacobian data is already current.";
pub const MSG_LINESEARCH_NONCONV: &str = "The line search algorithm was unable to find an iterate \
                                          sufficiently distinct from the current iterate.";
pub const MSG_LINESEARCH_BCFAIL: &str = "The line search algorithm was unable to satisfy the \
                                         beta-condition for nbcfails iterations.";
pub const MSG_MAXITER_REACHED: &str =
    "The maximum number of iterations was reached before convergence.";
pub const MSG_MXNEWT_5X_EXCEEDED: &str = "Five consecutive steps have been taken that satisfy a \
                                          scaled step length test.";
pub const MSG_SYSFUNC_REPTD: &str = "Unable to correct repeated recoverable system function errors.";
pub const MSG_NOL_FAIL: &str = "Unable to find user's Linear Jacobian, which is required for the \
                                KIN_PICARD Strategy";

/* =================================================================
 * Info messages (kinsol_impl.h). Parameterized formats become builders
 * producing the exact C printf expansion (SUN_FORMAT_E / SUN_FORMAT_G).
 * =================================================================*/

/* INFO_IVAR: "%s = %d" */
pub fn INFO_IVAR(name: &str, val: i32) -> String {
    format!("{} = {}", name, val)
}

/* INFO_LIVAR: "%s = %ld" */
pub fn INFO_LIVAR(name: &str, val: i64) -> String {
    format!("{} = {}", name, val)
}

/// C `INFO_RETVAL` ("Return value: %d") composed with the decoded flag
/// name exactly as the `PRNT_RETVAL` branch of C `KINPrintInfo` does:
/// `sprintf(msg, "%s (%s)", msg1, retstr)`.
pub fn INFO_RETVAL(ret: i32) -> String {
    format!("Return value: {} ({})", ret, kinPrintInfoRetvalString(ret))
}

/* INFO_ADJ: "no. of lambda adjustments = %ld" */
pub fn INFO_ADJ(nbktrk: i64) -> String {
    format!("no. of lambda adjustments = {}", nbktrk)
}

/* INFO_RVAR: "%s = " SUN_FORMAT_G */
pub fn INFO_RVAR(name: &str, val: sunrealtype) -> String {
    format!("{} = {}", name, sun_format_g(val))
}

/* INFO_NNI: "nni = %4ld, nfe = %6ld, fnorm = " SUN_FORMAT_G */
pub fn INFO_NNI(nni: i64, nfe: i64, fnorm: sunrealtype) -> String {
    format!("nni = {:4}, nfe = {:6}, fnorm = {}", nni, nfe, sun_format_g(fnorm))
}

/* INFO_TOL: "scsteptol = " SUN_FORMAT_G ", fnormtol = " SUN_FORMAT_G */
pub fn INFO_TOL(scsteptol: sunrealtype, fnormtol: sunrealtype) -> String {
    format!(
        "scsteptol = {}, fnormtol = {}",
        sun_format_g(scsteptol),
        sun_format_g(fnormtol)
    )
}

/* INFO_FMAX: "scaled f norm (for stopping) = " SUN_FORMAT_G */
pub fn INFO_FMAX(fmax: sunrealtype) -> String {
    format!("scaled f norm (for stopping) = {}", sun_format_g(fmax))
}

/* INFO_PNORM: "pnorm = " SUN_FORMAT_E */
pub fn INFO_PNORM(pnorm: sunrealtype) -> String {
    format!("pnorm = {}", sun_format_e(pnorm))
}

/* INFO_PNORM1: "(ivio=1) pnorm = " SUN_FORMAT_E */
pub fn INFO_PNORM1(pnorm: sunrealtype) -> String {
    format!("(ivio=1) pnorm = {}", sun_format_e(pnorm))
}

/* INFO_FNORM: "fnorm(L2) = " SUN_FORMAT_E */
pub fn INFO_FNORM(fnorm: sunrealtype) -> String {
    format!("fnorm(L2) = {}", sun_format_e(fnorm))
}

/* INFO_LAM: "min_lam = " E ", f1norm = " E ", pnorm = " E */
pub fn INFO_LAM(rlmin: sunrealtype, f1norm: sunrealtype, pnorm: sunrealtype) -> String {
    format!(
        "min_lam = {}, f1norm = {}, pnorm = {}",
        sun_format_e(rlmin),
        sun_format_e(f1norm),
        sun_format_e(pnorm)
    )
}

/* INFO_ALPHA: "fnorm = " E ", f1norm = " E ", alpha_cond = " E ", lam = " E */
pub fn INFO_ALPHA(
    fnorm: sunrealtype,
    f1norm: sunrealtype,
    alpha_cond: sunrealtype,
    lam: sunrealtype,
) -> String {
    format!(
        "fnorm = {}, f1norm = {}, alpha_cond = {}, lam = {}",
        sun_format_e(fnorm),
        sun_format_e(f1norm),
        sun_format_e(alpha_cond),
        sun_format_e(lam)
    )
}

/* INFO_BETA: "f1norm = " E ", beta_cond = " E ", lam = " E */
pub fn INFO_BETA(f1norm: sunrealtype, beta_cond: sunrealtype, lam: sunrealtype) -> String {
    format!(
        "f1norm = {}, beta_cond = {}, lam = {}",
        sun_format_e(f1norm),
        sun_format_e(beta_cond),
        sun_format_e(lam)
    )
}

/* INFO_ALPHABETA:
 * "f1norm = " E ", alpha_cond = " E ", beta_cond = " E ", lam = " E */
pub fn INFO_ALPHABETA(
    f1norm: sunrealtype,
    alpha_cond: sunrealtype,
    beta_cond: sunrealtype,
    lam: sunrealtype,
) -> String {
    format!(
        "f1norm = {}, alpha_cond = {}, beta_cond = {}, lam = {}",
        sun_format_e(f1norm),
        sun_format_e(alpha_cond),
        sun_format_e(beta_cond),
        sun_format_e(lam)
    )
}
