//! Port of `src/ida/ida_impl.h` + the constants/typedefs of
//! `include/ida/ida.h`.
//!
//! `IDAProcessError` (defined in `ida.c` upstream) is relocated here so
//! every ida module shares one definition; C varargs map to a
//! pre-formatted `msg` (call sites use the `MSG_*` constants/builders
//! below). Parameterized messages are functions producing the exact
//! C `printf` expansion (`SUN_FORMAT_G` = `%.15g` via `sun_format_g`).
//!
//! Fragment protocol: module-scope `#define` constants that upstream
//! repeats at the top of `ida.c` / `ida_ic.c` / `ida_io.c` live here
//! (one shared definition) because `ida.c` is ported in fragments; the
//! porting modules `use crate::ida_impl::*;` instead of redefining them.
//!
//! Handle model: `IDAMem = Rc<RefCell<IDAMemRec>>`. Internal functions
//! take `&IDAMem` and use granular borrows (never hold a borrow across a
//! callback, N_Vector op on user vectors, or linear/nonlinear solver
//! call — all can re-enter the mem).

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
 * Public constants (include/ida/ida.h)
 * =================================================================*/

/* itask */
pub const IDA_NORMAL: i32 = 1;
pub const IDA_ONE_STEP: i32 = 2;

/* icopt */
pub const IDA_YA_YDP_INIT: i32 = 1;
pub const IDA_Y_INIT: i32 = 2;

/* return values */
pub const IDA_SUCCESS: i32 = 0;
pub const IDA_TSTOP_RETURN: i32 = 1;
pub const IDA_ROOT_RETURN: i32 = 2;

pub const IDA_WARNING: i32 = 99;

pub const IDA_TOO_MUCH_WORK: i32 = -1;
pub const IDA_TOO_MUCH_ACC: i32 = -2;
pub const IDA_ERR_FAIL: i32 = -3;
pub const IDA_CONV_FAIL: i32 = -4;

pub const IDA_LINIT_FAIL: i32 = -5;
pub const IDA_LSETUP_FAIL: i32 = -6;
pub const IDA_LSOLVE_FAIL: i32 = -7;
pub const IDA_RES_FAIL: i32 = -8;
pub const IDA_REP_RES_ERR: i32 = -9;
pub const IDA_RTFUNC_FAIL: i32 = -10;
pub const IDA_CONSTR_FAIL: i32 = -11;

pub const IDA_FIRST_RES_FAIL: i32 = -12;
pub const IDA_LINESEARCH_FAIL: i32 = -13;
pub const IDA_NO_RECOVERY: i32 = -14;
pub const IDA_NLS_INIT_FAIL: i32 = -15;
pub const IDA_NLS_SETUP_FAIL: i32 = -16;
pub const IDA_NLS_FAIL: i32 = -17;

pub const IDA_MEM_NULL: i32 = -20;
pub const IDA_MEM_FAIL: i32 = -21;
pub const IDA_ILL_INPUT: i32 = -22;
pub const IDA_NO_MALLOC: i32 = -23;
pub const IDA_BAD_EWT: i32 = -24;
pub const IDA_BAD_K: i32 = -25;
pub const IDA_BAD_T: i32 = -26;
pub const IDA_BAD_DKY: i32 = -27;
pub const IDA_VECTOROP_ERR: i32 = -28;

pub const IDA_CONTEXT_ERR: i32 = -29;

pub const IDA_TOO_CLOSE: i32 = -60;

pub const IDA_UNRECOGNIZED_ERROR: i32 = -99;

/* ------------------------------
 * User-Supplied Function Types
 * ------------------------------ */

pub type IDAResFn = fn(
    tt: sunrealtype,
    yy: &N_Vector,
    yp: &N_Vector,
    rr: &N_Vector,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32;

pub type IDARootFn = fn(
    t: sunrealtype,
    y: &N_Vector,
    yp: &N_Vector,
    gout: &mut [sunrealtype],
    user_data: &mut Option<Box<dyn Any>>,
) -> i32;

pub type IDAEwtFn = fn(y: &N_Vector, ewt: &N_Vector, user_data: &mut Option<Box<dyn Any>>) -> i32;

/* =================================================================
 * Internal constants (ida_impl.h)
 * =================================================================*/

/* Basic IDA constants */
pub const HMAX_INV_DEFAULT: sunrealtype = 0.0; /* hmax_inv default value          */
pub const HMIN_DEFAULT: sunrealtype = 0.0; /* hmin default value              */
pub const MAXORD_DEFAULT: usize = 5; /* maxord default value            */
pub const MXORDP1: usize = 6; /* max. number of N_Vectors in phi */
pub const MXSTEP_DEFAULT: i64 = 500; /* mxstep default value            */

pub const ETA_MAX_FX_DEFAULT: sunrealtype = 2.0; /* threshold to increase step size   */
pub const ETA_MIN_FX_DEFAULT: sunrealtype = 1.0; /* threshold to decrease step size   */
pub const ETA_MAX_DEFAULT: sunrealtype = 2.0; /* max step size increase factor     */
pub const ETA_MIN_DEFAULT: sunrealtype = 0.5; /* min step size decrease factor     */
pub const ETA_LOW_DEFAULT: sunrealtype = 0.9; /* upper bound on decrease factor    */
pub const ETA_MIN_EF_DEFAULT: sunrealtype = 0.25; /* err test fail min decrease factor */
pub const ETA_CF_DEFAULT: sunrealtype = 0.25; /* NLS failure decrease factor       */

pub const DCJ_DEFAULT: sunrealtype = 0.25; /* constant for updating Jacobian/preconditioner */

pub const MAX_CONSTRAINT_FAILS: i32 = 10;

/* Return values for lower level routines used by IDASolve and functions
   provided to the nonlinear solver */

pub const IDA_RES_RECVR: i32 = 1;
pub const IDA_LSETUP_RECVR: i32 = 2;
pub const IDA_LSOLVE_RECVR: i32 = 3;
pub const IDA_NLS_SETUP_RECVR: i32 = 4;

/* =================================================================
 * IDA private constants (module-scope in ida.c; shared here because
 * ida.c is ported in fragments — fragment protocol)
 * =================================================================*/

pub const ZERO: sunrealtype = 0.0; /* real 0.0    */
pub const HALF: sunrealtype = 0.5; /* real 0.5    */
pub const TWOTHIRDS: sunrealtype = 0.667; /* real 2/3    */
pub const ONE: sunrealtype = 1.0; /* real 1.0    */
pub const ONEPT5: sunrealtype = 1.5; /* real 1.5    */
pub const TWO: sunrealtype = 2.0; /* real 2.0    */
pub const FOUR: sunrealtype = 4.0; /* real 4.0    */
pub const FIVE: sunrealtype = 5.0; /* real 5.0    */
pub const TEN: sunrealtype = 10.0; /* real 10.0   */
pub const TWENTY: sunrealtype = 20.0; /* real 20.0   */
pub const HUNDRED: sunrealtype = 100.0; /* real 100.0  */
pub const PT9: sunrealtype = 0.9; /* real 0.9    */
pub const PT1: sunrealtype = 0.1; /* real 0.1    */
pub const PT01: sunrealtype = 0.01; /* real 0.01   */
pub const PT001: sunrealtype = 0.001; /* real 0.001  */
pub const PT0001: sunrealtype = 0.0001; /* real 0.0001 */
pub const PT99: sunrealtype = 0.99; /* real 0.99 (ida_ic.c) */
pub const TWOPT5: sunrealtype = 2.5; /* real 2.5  (ida_io.c) */

/* real 1 + epsilon used in testing if the step size is below its bound */
pub const ONEPSM: sunrealtype = 1.000001;

/* IDAStep control constants */
pub const PREDICT_AGAIN: i32 = 20;

/* Return values for lower level routines used by IDASolve */
pub const CONTINUE_STEPS: i32 = 99;

/* IDACompleteStep constants */
pub const UNSET: i32 = -1;
pub const LOWER: i32 = 1;
pub const RAISE: i32 = 2;
pub const MAINTAIN: i32 = 3;

/* IDATestError constants */
pub const ERROR_TEST_FAIL: i32 = 7;

/* Control constants for lower-level rootfinding functions */
pub const RTFOUND: i32 = 1;
pub const CLOSERT: i32 = 3;

/* Control constants for tolerances */
pub const IDA_NN: i32 = 0;
pub const IDA_SS: i32 = 1;
pub const IDA_SV: i32 = 2;
pub const IDA_WF: i32 = 3;

/* =================================================================
 * IDACalcIC constants (module-scope in ida_ic.c; shared here —
 * fragment protocol)
 * =================================================================*/

pub const ICRATEMAX: sunrealtype = 0.9; /* max. Newton conv. rate */
pub const ALPHALS: sunrealtype = 0.0001; /* alpha in linesearch conv. test */

/* IDACalcIC control constants */
pub const IC_FAIL_RECOV: i32 = 1;
pub const IC_CONSTR_FAILED: i32 = 2;
pub const IC_LINESRCH_FAILED: i32 = 3;
pub const IC_CONV_FAIL: i32 = 4;
pub const IC_SLOW_CONVRG: i32 = 5;

/* =================================================================
 * Main integrator memory block
 * =================================================================*/

pub struct IDAMemRec {
    pub ida_sunctx: SUNContext,

    pub ida_uround: sunrealtype, /* machine unit roundoff */

    /*--------------------------
    Problem Specification Data
    --------------------------*/
    pub ida_res: Option<IDAResFn>,           /* F(t,y(t),y'(t))=0; the function F     */
    pub ida_user_data: Option<Box<dyn Any>>, /* user pointer passed to res            */

    pub ida_itol: i32,                 /* itol = IDA_SS, IDA_SV, IDA_WF, IDA_NN */
    pub ida_rtol: sunrealtype,         /* relative tolerance                    */
    pub ida_Satol: sunrealtype,        /* scalar absolute tolerance             */
    pub ida_Vatol: Option<N_Vector>,   /* vector absolute tolerance             */
    pub ida_atolmin0: sunbooleantype,  /* flag indicating that min(atol) = 0    */
    pub ida_user_efun: sunbooleantype, /* SUNTRUE if user provides efun         */
    pub ida_efun: Option<IDAEwtFn>,    /* function to set ewt                   */
    pub ida_edata: Option<Box<dyn Any>>, /* user pointer passed to efun           */

    pub ida_suppressalg: sunbooleantype, /* SUNTRUE means suppress algebraic vars
                                         in local error tests                  */

    /*-----------------------------------------------
    Divided differences array and associated arrays
    -----------------------------------------------*/
    pub ida_phi: [Option<N_Vector>; MXORDP1], /* phi = (maxord+1) arrays of divided differences */

    pub ida_psi: [sunrealtype; MXORDP1], /* differences in t (sums of recent step sizes)   */
    pub ida_alpha: [sunrealtype; MXORDP1], /* ratios of current stepsize to psi values       */
    pub ida_beta: [sunrealtype; MXORDP1], /* ratios of current to previous product of psi's */
    pub ida_sigma: [sunrealtype; MXORDP1], /* product successive alpha values and factorial  */
    pub ida_gamma: [sunrealtype; MXORDP1], /* sum of reciprocals of psi values               */

    /*-------------------------
    N_Vectors for integration
    -------------------------*/
    pub ida_ewt: Option<N_Vector>, /* error weight vector                            */
    pub ida_yy: Option<N_Vector>,  /* work space for y vector (= user's yret;
                                   copy-back at every IDASolve return path!)      */
    pub ida_yp: Option<N_Vector>,  /* work space for y' vector (= user's ypret;
                                   copy-back at every IDASolve return path!)      */
    pub ida_yypredict: Option<N_Vector>, /* predicted y vector                             */
    pub ida_yppredict: Option<N_Vector>, /* predicted y' vector                            */
    pub ida_delta: Option<N_Vector>, /* residual vector                                */
    pub ida_id: Option<N_Vector>,  /* bit vector for diff./algebraic components      */
    pub ida_savres: Option<N_Vector>, /* saved residual vector                          */
    pub ida_ee: Option<N_Vector>,  /* accumulated corrections to y vector, but
                                   set equal to estimated local errors upon
                                   successful return                              */
    pub ida_tempv1: Option<N_Vector>, /* work space vector                              */
    pub ida_tempv2: Option<N_Vector>, /* work space vector                              */
    pub ida_tempv3: Option<N_Vector>, /* work space vector                              */
    pub ida_ynew: Option<N_Vector>, /* work vector for y in IDACalcIC (= tempv2)      */
    pub ida_ypnew: Option<N_Vector>, /* work vector for yp in IDACalcIC (= ee)         */
    pub ida_delnew: Option<N_Vector>, /* work vector for delta in IDACalcIC (= phi[2])  */
    pub ida_dtemp: Option<N_Vector>, /* work vector in IDACalcIC (= phi[3])            */

    /*------------------------------
    Variables for use by IDACalcIC
    ------------------------------*/
    pub ida_t0: sunrealtype,       /* initial t                                      */
    pub ida_yy0: Option<N_Vector>, /* initial y vector (user-supplied).              */
    pub ida_yp0: Option<N_Vector>, /* initial y' vector (user-supplied).             */

    pub ida_icopt: i32,            /* IC calculation user option                     */
    pub ida_lsoff: sunbooleantype, /* IC calculation linesearch turnoff option       */
    pub ida_maxnh: i32,            /* max. number of h tries in IC calculation       */
    pub ida_maxnj: i32,            /* max. number of J tries in IC calculation       */
    pub ida_maxnit: i32,           /* max. number of Netwon iterations in IC calc.   */
    pub ida_nbacktr: i32,          /* number of IC linesearch backtrack operations   */
    pub ida_sysindex: i32,         /* computed system index (0 or 1)                 */
    pub ida_maxbacks: i32,         /* max backtracks per Newton step                 */
    pub ida_epiccon: sunrealtype,  /* IC nonlinear convergence test constant         */
    pub ida_steptol: sunrealtype,  /* minimum Newton step size in IC calculation     */
    pub ida_tscale: sunrealtype,   /* time scale factor = abs(tout1 - t0)            */

    /* Tstop information */
    pub ida_tstopset: sunbooleantype,
    pub ida_tstop: sunrealtype,

    /* Step Data */
    pub ida_kk: i32,    /* current BDF method order                              */
    pub ida_kused: i32, /* method order used on last successful step             */
    pub ida_knew: i32,  /* order for next step from order decrease decision      */
    pub ida_phase: i32, /* flag to trigger step doubling in first few steps      */
    pub ida_ns: i32,    /* counts steps at fixed stepsize and order              */

    pub ida_hin: sunrealtype,      /* initial step                                      */
    pub ida_h0u: sunrealtype,      /* actual initial stepsize                           */
    pub ida_hh: sunrealtype,       /* current step size h                               */
    pub ida_hused: sunrealtype,    /* step size used on last successful step            */
    pub ida_eta: sunrealtype,      /* eta = hnext / hused                               */
    pub ida_tn: sunrealtype,       /* current internal value of t                       */
    pub ida_tretlast: sunrealtype, /* value of tret previously returned by IDASolve     */
    pub ida_cj: sunrealtype,       /* current value of scalar (-alphas/hh) in Jacobian  */
    pub ida_cjlast: sunrealtype,   /* cj value saved from last successful step          */
    pub ida_cjold: sunrealtype,    /* cj value saved from last call to lsetup           */
    pub ida_cjratio: sunrealtype,  /* ratio of cj values: cj/cjold                      */
    pub ida_ss: sunrealtype,       /* scalar used in Newton iteration convergence test  */
    pub ida_delnrm: sunrealtype,   /* norm of current nonlinear solver update           */
    pub ida_oldnrm: sunrealtype,   /* norm of previous nonlinear solver update          */
    pub ida_epsNewt: sunrealtype,  /* test constant in Newton convergence test          */
    pub ida_epcon: sunrealtype,    /* coefficient of the Newton convergence test        */
    pub ida_toldel: sunrealtype,   /* tolerance in direct test on Newton corrections    */

    /*------
    Limits
    ------*/
    pub ida_maxncf: i32, /* max number of convergence failures                */
    pub ida_maxnef: i32, /* max number of error test failures                 */

    pub ida_maxord: i32,          /* max value of method order k:                      */
    pub ida_maxord_alloc: i32,    /* value of maxord used when allocating memory       */
    pub ida_mxstep: i64,          /* max number of internal steps for one user call    */
    pub ida_hmax_inv: sunrealtype, /* inverse of max. step size hmax (default = 0.0)    */
    pub ida_hmin: sunrealtype,    /* min step size hmin (default = 0.0)                */

    pub ida_eta_max_fx: sunrealtype, /* threshold to increase step size */
    pub ida_eta_min_fx: sunrealtype, /* threshold to decrease step size */
    pub ida_eta_max: sunrealtype,    /* max step size increase factor   */
    pub ida_eta_min: sunrealtype,    /* min step size decrease factor   */
    pub ida_eta_low: sunrealtype,    /* upper bound on decrease factor  */
    pub ida_eta_min_ef: sunrealtype, /* eta >= eta_min_ef after an error test failure */
    pub ida_eta_cf: sunrealtype,     /* eta on a nonlinear solver convergence failure */

    /*--------
    Counters
    --------*/
    pub ida_nst: i64,     /* number of internal steps taken                    */
    pub ida_nre: i64,     /* number of function (res) calls                    */
    pub ida_ncfn: i64,    /* number of corrector convergence failures          */
    pub ida_netf: i64,    /* number of error test failures                     */
    pub ida_nni: i64,     /* number of Newton iterations performed             */
    pub ida_nnf: i64,     /* number of Newton convergence failures             */
    pub ida_nsetups: i64, /* number of lsetup calls                            */

    /*------------------
    Space requirements
    ------------------*/
    pub ida_lrw1: sunindextype, /* no. of sunrealtype words in 1 N_Vector            */
    pub ida_liw1: sunindextype, /* no. of integer words in 1 N_Vector                */
    pub ida_lrw: i64,           /* number of sunrealtype words in IDA work vectors   */
    pub ida_liw: i64,           /* no. of integer words in IDA work vectors          */

    pub ida_tolsf: sunrealtype, /* tolerance scale factor (saved value)              */

    /* Flags to verify correct calling sequence */
    pub ida_SetupDone: sunbooleantype, /* set to SUNFALSE by IDAMalloc and IDAReInit
                                       set to SUNTRUE by IDACalcIC or IDASolve      */

    pub ida_VatolMallocDone: sunbooleantype,
    pub ida_idMallocDone: sunbooleantype,

    pub ida_MallocDone: sunbooleantype, /* set to SUNFALSE by IDACreate
                                        set to SUNTRUE by IDAMAlloc
                                        tested by IDAReInit and IDASolve             */

    /*---------------------
    Nonlinear Solver Data
    ---------------------*/
    pub NLS: Option<SUNNonlinearSolver>, /* nonlinear solver object */
    pub ownNLS: sunbooleantype,          /* flag indicating NLS ownership */
    pub nls_res: Option<IDAResFn>,       /* F(t,y(t),y'(t))=0; used in the nonlinear
                                         solver */

    /*------------------
    Linear Solver Data
    ------------------*/

    /* Linear Solver functions to be called */
    pub ida_linit: Option<fn(idamem: &IDAMem) -> i32>,

    pub ida_lsetup: Option<
        fn(
            idamem: &IDAMem,
            yyp: &N_Vector,
            ypp: &N_Vector,
            resp: &N_Vector,
            tempv1: &N_Vector,
            tempv2: &N_Vector,
            tempv3: &N_Vector,
        ) -> i32,
    >,

    pub ida_lsolve: Option<
        fn(
            idamem: &IDAMem,
            b: &N_Vector,
            weight: &N_Vector,
            ycur: &N_Vector,
            ypcur: &N_Vector,
            rescur: &N_Vector,
        ) -> i32,
    >,

    pub ida_lperf: Option<fn(idamem: &IDAMem, perftask: i32) -> i32>,

    pub ida_lfree: Option<fn(idamem: &IDAMem) -> i32>,

    /* Linear Solver specific memory */
    pub ida_lmem: Option<Box<dyn Any>>, /* linear solver interface structure */
    pub ida_dcj: sunrealtype, /* parameter that determines cj ratio thresholds for calling
                              * the linear solver setup function */

    /* Flag to indicate successful ida_linit call */
    pub ida_linitOK: sunbooleantype,

    /*----------------
    Rootfinding Data
    ----------------*/
    pub ida_gfun: Option<IDARootFn>, /* Function g for roots sought                     */
    pub ida_nrtfn: i32,              /* number of components of g                       */
    pub ida_iroots: Vec<i32>,        /* array for root information                      */
    pub ida_rootdir: Vec<i32>,       /* array specifying direction of zero-crossing     */
    pub ida_tlo: sunrealtype,        /* nearest endpoint of interval in root search     */
    pub ida_thi: sunrealtype,        /* farthest endpoint of interval in root search    */
    pub ida_trout: sunrealtype,      /* t return value from rootfinder routine          */
    pub ida_glo: Vec<sunrealtype>,   /* saved array of g values at t = tlo              */
    pub ida_ghi: Vec<sunrealtype>,   /* saved array of g values at t = thi              */
    pub ida_grout: Vec<sunrealtype>, /* array of g values at t = trout                  */
    pub ida_ttol: sunrealtype,       /* tolerance on root location                      */
    pub ida_irfnd: i32,              /* flag showing whether last step had a root       */
    pub ida_nge: i64,                /* counter for g evaluations                       */
    pub ida_gactive: Vec<sunbooleantype>, /* array with active/inactive event functions      */
    pub ida_mxgnull: i32,            /* number of warning messages about possible g==0  */

    /*---------------------------
    Inequality Constraints Data
    ---------------------------*/
    pub ida_constraints: Option<N_Vector>, /* vector of inequality constraint flags */
    pub constraint_corrections: i64,       /* total constraint corrections   */
    pub constraint_fails: i64,             /* total constraint failures             */
    pub max_constraint_fails: i32,         /* max failures allowed in a step        */

    /* Arrays for Fused Vector Operations */

    /* scalar arrays */
    pub ida_cvals: [sunrealtype; MXORDP1],
    pub ida_dvals: [sunrealtype; MAXORD_DEFAULT],

    /* vector arrays (scratch handle arrays) */
    pub ida_Xvecs: Vec<N_Vector>,
    pub ida_Zvecs: Vec<N_Vector>,
}

pub type IDAMem = Rc<RefCell<IDAMemRec>>;

impl IDAMemRec {
    /// All-zero/None baseline (the C `malloc` block in `IDACreate` before
    /// its explicit default assignments; every field the C code reads is
    /// explicitly set there, so the baseline values are never observable).
    pub fn zeroed(sunctx: SUNContext) -> IDAMemRec {
        IDAMemRec {
            ida_sunctx: sunctx,
            ida_uround: 0.0,
            ida_res: None,
            ida_user_data: None,
            ida_itol: 0,
            ida_rtol: 0.0,
            ida_Satol: 0.0,
            ida_Vatol: None,
            ida_atolmin0: SUNFALSE,
            ida_user_efun: SUNFALSE,
            ida_efun: None,
            ida_edata: None,
            ida_suppressalg: SUNFALSE,
            ida_phi: Default::default(),
            ida_psi: [0.0; MXORDP1],
            ida_alpha: [0.0; MXORDP1],
            ida_beta: [0.0; MXORDP1],
            ida_sigma: [0.0; MXORDP1],
            ida_gamma: [0.0; MXORDP1],
            ida_ewt: None,
            ida_yy: None,
            ida_yp: None,
            ida_yypredict: None,
            ida_yppredict: None,
            ida_delta: None,
            ida_id: None,
            ida_savres: None,
            ida_ee: None,
            ida_tempv1: None,
            ida_tempv2: None,
            ida_tempv3: None,
            ida_ynew: None,
            ida_ypnew: None,
            ida_delnew: None,
            ida_dtemp: None,
            ida_t0: 0.0,
            ida_yy0: None,
            ida_yp0: None,
            ida_icopt: 0,
            ida_lsoff: SUNFALSE,
            ida_maxnh: 0,
            ida_maxnj: 0,
            ida_maxnit: 0,
            ida_nbacktr: 0,
            ida_sysindex: 0,
            ida_maxbacks: 0,
            ida_epiccon: 0.0,
            ida_steptol: 0.0,
            ida_tscale: 0.0,
            ida_tstopset: SUNFALSE,
            ida_tstop: 0.0,
            ida_kk: 0,
            ida_kused: 0,
            ida_knew: 0,
            ida_phase: 0,
            ida_ns: 0,
            ida_hin: 0.0,
            ida_h0u: 0.0,
            ida_hh: 0.0,
            ida_hused: 0.0,
            ida_eta: 0.0,
            ida_tn: 0.0,
            ida_tretlast: 0.0,
            ida_cj: 0.0,
            ida_cjlast: 0.0,
            ida_cjold: 0.0,
            ida_cjratio: 0.0,
            ida_ss: 0.0,
            ida_delnrm: 0.0,
            ida_oldnrm: 0.0,
            ida_epsNewt: 0.0,
            ida_epcon: 0.0,
            ida_toldel: 0.0,
            ida_maxncf: 0,
            ida_maxnef: 0,
            ida_maxord: 0,
            ida_maxord_alloc: 0,
            ida_mxstep: 0,
            ida_hmax_inv: 0.0,
            ida_hmin: 0.0,
            ida_eta_max_fx: 0.0,
            ida_eta_min_fx: 0.0,
            ida_eta_max: 0.0,
            ida_eta_min: 0.0,
            ida_eta_low: 0.0,
            ida_eta_min_ef: 0.0,
            ida_eta_cf: 0.0,
            ida_nst: 0,
            ida_nre: 0,
            ida_ncfn: 0,
            ida_netf: 0,
            ida_nni: 0,
            ida_nnf: 0,
            ida_nsetups: 0,
            ida_lrw1: 0,
            ida_liw1: 0,
            ida_lrw: 0,
            ida_liw: 0,
            ida_tolsf: 0.0,
            ida_SetupDone: SUNFALSE,
            ida_VatolMallocDone: SUNFALSE,
            ida_idMallocDone: SUNFALSE,
            ida_MallocDone: SUNFALSE,
            NLS: None,
            ownNLS: SUNFALSE,
            nls_res: None,
            ida_linit: None,
            ida_lsetup: None,
            ida_lsolve: None,
            ida_lperf: None,
            ida_lfree: None,
            ida_lmem: None,
            ida_dcj: 0.0,
            ida_linitOK: SUNFALSE,
            ida_gfun: None,
            ida_nrtfn: 0,
            ida_iroots: Vec::new(),
            ida_rootdir: Vec::new(),
            ida_tlo: 0.0,
            ida_thi: 0.0,
            ida_trout: 0.0,
            ida_glo: Vec::new(),
            ida_ghi: Vec::new(),
            ida_grout: Vec::new(),
            ida_ttol: 0.0,
            ida_irfnd: 0,
            ida_nge: 0,
            ida_gactive: Vec::new(),
            ida_mxgnull: 0,
            ida_constraints: None,
            constraint_corrections: 0,
            constraint_fails: 0,
            max_constraint_fails: 0,
            ida_cvals: [0.0; MXORDP1],
            ida_dvals: [0.0; MAXORD_DEFAULT],
            ida_Xvecs: Vec::new(),
            ida_Zvecs: Vec::new(),
        }
    }
}

/* =================================================================
 * High level error handler (relocated from ida.c; C varargs map to a
 * pre-formatted msg — call sites use the MSG_* builders below)
 *
 * `line`/`file` come from Rust `line!()`/`file!()` at every call site
 * where C passes `__LINE__`/`__FILE__`; they only reach the logger
 * scope field. See accepted deviation A in lib.rs.
 * =================================================================*/

pub fn IDAProcessError(
    IDA_mem: Option<&IDAMem>,
    error_code: i32,
    line: i32,
    func: &str,
    file: &str,
    msg: &str,
) {
    match IDA_mem {
        None => {
            SUNGlobalFallbackErrHandler(line, func, file, msg, error_code);
        }
        Some(IDA_mem) => {
            let sunctx = IDA_mem.borrow().ida_sunctx.clone();

            if error_code == IDA_WARNING {
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
 * Error messages (ida_impl.h). Parameter-less messages are consts;
 * parameterized ones are builders producing the exact C expansion
 * (including the C header's missing-separator quirks, e.g.
 * `"At " MSG_TIME "the user-provide EwtSet function failed."` has no
 * space/comma after the time value — preserved byte-for-byte).
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

/* General errors */
pub const MSG_MEM_FAIL: &str = "A memory request failed.";
pub const MSG_NULL_SUNCTX: &str = "sunctx = NULL illegal.";
pub const MSG_NO_MEM: &str = "ida_mem = NULL illegal.";
pub const MSG_NO_MALLOC: &str = "Attempt to call before IDAMalloc.";
pub const MSG_BAD_NVECTOR: &str = "A required vector operation is not implemented.";

/* Initialization errors */
pub const MSG_Y0_NULL: &str = "y0 = NULL illegal.";
pub const MSG_YP0_NULL: &str = "yp0 = NULL illegal.";
pub const MSG_BAD_ITOL: &str =
    "Illegal value for itol. The legal values are IDA_SS, IDA_SV, and IDA_WF.";
pub const MSG_RES_NULL: &str = "res = NULL illegal.";
pub const MSG_BAD_RTOL: &str = "rtol < 0 illegal.";
pub const MSG_ATOL_NULL: &str = "atol = NULL illegal.";
pub const MSG_BAD_ATOL: &str = "Some atol component < 0.0 illegal.";
pub const MSG_ROOT_FUNC_NULL: &str = "g = NULL illegal.";

pub const MSG_MISSING_ID: &str = "id = NULL but suppressalg option on.";
pub const MSG_NO_TOLS: &str = "No integration tolerances have been specified.";
pub const MSG_FAIL_EWT: &str = "The user-provide EwtSet function failed.";
pub const MSG_BAD_EWT: &str = "Some initial ewt component = 0.0 illegal.";
pub const MSG_Y0_FAIL_CONSTR: &str = "y0 fails to satisfy constraints.";
pub const MSG_LSOLVE_NULL: &str = "The linear solver's solve routine is NULL.";
pub const MSG_LINIT_FAIL: &str = "The linear solver's init routine failed.";
pub const MSG_NLS_INIT_FAIL: &str = "The nonlinear solver's init routine failed.";

/* IDACalcIC error messages */
pub const MSG_IC_BAD_ICOPT: &str = "icopt has an illegal value.";
pub const MSG_IC_BAD_MAXBACKS: &str = "maxbacks <= 0 illegal.";
pub const MSG_IC_MISSING_ID: &str = "id = NULL conflicts with icopt.";
pub const MSG_IC_TOO_CLOSE: &str =
    "tout1 too close to t0 to attempt initial condition calculation.";
pub const MSG_IC_BAD_ID: &str = "id has illegal values.";
pub const MSG_IC_BAD_EWT: &str = "Some initial ewt component = 0.0 illegal.";
pub const MSG_IC_RES_NONREC: &str = "The residual function failed unrecoverably. ";
pub const MSG_IC_RES_FAIL: &str = "The residual function failed at the first call. ";
pub const MSG_IC_SETUP_FAIL: &str = "The linear solver setup failed unrecoverably.";
pub const MSG_IC_SOLVE_FAIL: &str = "The linear solver solve failed unrecoverably.";
pub const MSG_IC_NO_RECOVERY: &str = "The residual routine or the linear setup or solve routine \
                                      had a recoverable error, but IDACalcIC was unable to \
                                      recover.";
pub const MSG_IC_FAIL_CONSTR: &str = "Unable to satisfy the inequality constraints.";
pub const MSG_IC_FAILED_LINS: &str =
    "The linesearch algorithm failed: step too small or too many backtracks.";
pub const MSG_IC_CONV_FAILED: &str = "Newton/Linesearch algorithm failed to converge.";

/* IDASolve error messages */
pub const MSG_YRET_NULL: &str = "yret = NULL illegal.";
pub const MSG_YPRET_NULL: &str = "ypret = NULL illegal.";
pub const MSG_TRET_NULL: &str = "tret = NULL illegal.";
pub const MSG_BAD_ITASK: &str = "itask has an illegal value.";
pub const MSG_TOO_CLOSE: &str = "tout too close to t0 to start integration.";
pub const MSG_BAD_HINIT: &str = "Initial step is not towards tout.";

pub fn MSG_BAD_TSTOP(tstop: sunrealtype, t: sunrealtype) -> String {
    format!(
        "The value {} is behind current {}in the direction of integration.",
        MSG_TIME_TSTOP(tstop),
        MSG_TIME(t)
    )
}

pub fn MSG_CLOSE_ROOTS(t: sunrealtype) -> String {
    format!("Root found at and very near {}.", MSG_TIME(t))
}

pub fn MSG_MAX_STEPS(t: sunrealtype) -> String {
    format!("At {}, mxstep steps taken before reaching tout.", MSG_TIME(t))
}

pub fn MSG_EWT_NOW_FAIL(t: sunrealtype) -> String {
    format!("At {}the user-provide EwtSet function failed.", MSG_TIME(t))
}

pub fn MSG_EWT_NOW_BAD(t: sunrealtype) -> String {
    format!("At {}some ewt component has become <= 0.0.", MSG_TIME(t))
}

pub fn MSG_TOO_MUCH_ACC(t: sunrealtype) -> String {
    format!("At {}too much accuracy requested.", MSG_TIME(t))
}

pub const MSG_BAD_K: &str = "Illegal value for k.";
pub const MSG_NULL_DKY: &str = "dky = NULL illegal.";
pub const MSG_NULL_DKYP: &str = "dkyp = NULL illegal.";

pub fn MSG_BAD_T(t: sunrealtype, t0: sunrealtype, t1: sunrealtype) -> String {
    format!("Illegal value for t.{}", MSG_TIME_INT(t, t0, t1))
}

pub fn MSG_BAD_TOUT(tout: sunrealtype) -> String {
    format!(
        "Trouble interpolating at {}. tout too far back in direction of integration.",
        MSG_TIME_TOUT(tout)
    )
}

pub fn MSG_ERR_FAILS(t: sunrealtype, h: sunrealtype) -> String {
    format!(
        "At {}, the error test failed repeatedly or with |h| = hmin.",
        MSG_TIME_H(t, h)
    )
}

pub fn MSG_CONV_FAILS(t: sunrealtype, h: sunrealtype) -> String {
    format!(
        "At {}, the corrector convergence failed repeatedly or with |h| = hmin.",
        MSG_TIME_H(t, h)
    )
}

pub fn MSG_SETUP_FAILED(t: sunrealtype) -> String {
    format!("At {}, the linear solver setup failed unrecoverably.", MSG_TIME(t))
}

pub fn MSG_SOLVE_FAILED(t: sunrealtype) -> String {
    format!("At {}, the linear solver solve failed unrecoverably.", MSG_TIME(t))
}

pub fn MSG_REP_RES_ERR(t: sunrealtype) -> String {
    format!("At {} repeated recoverable residual errors.", MSG_TIME(t))
}

pub fn MSG_RES_NONRECOV(t: sunrealtype) -> String {
    format!("At {}, the residual function failed unrecoverably.", MSG_TIME(t))
}

pub fn MSG_FAILED_CONSTR(t: sunrealtype) -> String {
    format!("At {}, unable to satisfy inequality constraints.", MSG_TIME(t))
}

pub fn MSG_RTFUNC_FAILED(t: sunrealtype) -> String {
    format!(
        "At {}, the rootfinding routine failed in an unrecoverable manner.",
        MSG_TIME(t)
    )
}

pub const MSG_NO_ROOT: &str = "Rootfinding was not initialized.";
pub const MSG_INACTIVE_ROOTS: &str = "At the end of the first step, there are still some root \
                                      functions identically 0. This warning will not be issued \
                                      again.";

pub fn MSG_NLS_INPUT_NULL(t: sunrealtype) -> String {
    format!("At {}, the nonlinear solver was passed a NULL input.", MSG_TIME(t))
}

pub fn MSG_NLS_SETUP_FAILED(t: sunrealtype) -> String {
    format!("At {}, the nonlinear solver setup failed unrecoverably.", MSG_TIME(t))
}

pub fn MSG_NLS_FAIL(t: sunrealtype) -> String {
    format!(
        "At {}, the nonlinear solver failed in an unrecoverable manner.",
        MSG_TIME(t)
    )
}

/* IDASet* / IDAGet* error messages */
pub const MSG_NEG_MAXORD: &str = "maxord <= 0 illegal.";
pub const MSG_BAD_MAXORD: &str = "Illegal attempt to increase maximum order.";
pub const MSG_NEG_HMAX: &str = "hmax < 0 illegal.";
pub const MSG_NEG_HMIN: &str = "hmin < 0 illegal.";
pub const MSG_NEG_EPCON: &str = "epcon <= 0.0 illegal.";
pub const MSG_BAD_CONSTR: &str = "Illegal values in constraints vector.";
pub const MSG_BAD_EPICCON: &str = "epiccon <= 0.0 illegal.";
pub const MSG_BAD_MAXNH: &str = "maxnh <= 0 illegal.";
pub const MSG_BAD_MAXNJ: &str = "maxnj <= 0 illegal.";
pub const MSG_BAD_MAXNIT: &str = "maxnit <= 0 illegal.";
pub const MSG_BAD_STEPTOL: &str = "steptol <= 0.0 illegal.";

pub const MSG_TOO_LATE: &str = "IDAGetConsistentIC can only be called before IDASolve.";
