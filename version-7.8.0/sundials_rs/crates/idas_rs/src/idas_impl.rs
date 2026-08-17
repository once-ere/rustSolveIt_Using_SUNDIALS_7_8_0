//! Port of `src/idas/idas_impl.h` + the constants/typedefs of
//! `include/idas/idas.h`, plus the adjoint (ASA) memory records that
//! `idas_impl.h` defines (`IDAadjMemRec`, `IDAckpntMemRec`,
//! `IDAdtpntMemRec`, `IDABMemRec`, interpolation data records).
//!
//! IDAS is a strict superset of IDA: every convention pinned by
//! `ida_rs::ida_impl` is preserved verbatim here — field names identical
//! to the C (`ida_*`), `MXORDP1`-sized `phi`/`psi`/`alpha`/`beta`/
//! `sigma`/`gamma` arrays, message constants named `MSG_*` with the exact
//! C names (the IDA family uses bare `MSG_*`, unlike CVODE's `MSGCV_*`),
//! and `IDAProcessError` shaped exactly as in `ida_impl.rs`. The
//! sensitivity / quadrature / adjoint additions follow the rendering
//! already fixed by `cvodes_rs::cvodes_impl`.
//!
//! `IDAProcessError` (defined in `idas.c` upstream) is relocated here so
//! every idas module shares one definition; C varargs map to a
//! pre-formatted `msg` (call sites use the `MSG_*` / `MSGAM_*` constants
//! and builders below). Parameterized messages are functions producing
//! the exact C `printf` expansion (`SUN_FORMAT_G` = `%.15g` via
//! `sun_format_g`), including the C header's missing-separator quirks
//! (e.g. `"At " MSG_TIME "too much accuracy requested."` has no comma
//! after the time value — preserved byte-for-byte).
//!
//! Fragment-file protocol: `idas.c` (8923 lines), `idaa.c` (3842),
//! `idas_ic.c`, `idas_io.c`, `idas_ls.c` and the three `idas_nls*.c`
//! files are ported as several fragment modules; ALL module-scope
//! `#define` constants those files declare live HERE (one shared
//! definition) so every fragment agrees. Porting modules write
//! `use crate::idas_impl::*;` instead of redefining them.
//!
//! Handle model: `IDAMem = Rc<RefCell<IDAMemRec>>`. Internal functions
//! take `&IDAMem` and use granular borrows (never hold a borrow across a
//! callback, an N_Vector op on a user vector, or a linear/nonlinear
//! solver call — all can re-enter the mem). The adjoint records use the
//! same handle model (`IDAadjMem`, `IDAckpntMem`, `IDAdtpntMem`,
//! `IDABMem` are `Rc<RefCell<…>>`); the C intrusive linked lists
//! (`ck_next`, `ida_next`) become `Vec`s ordered exactly as the C
//! traversal order (index 0 = list head = most recently inserted;
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
 * Public constants (include/idas/idas.h)
 * =================================================================*/

/* itask */
pub const IDA_NORMAL: i32 = 1;
pub const IDA_ONE_STEP: i32 = 2;

/* icopt */
pub const IDA_YA_YDP_INIT: i32 = 1;
pub const IDA_Y_INIT: i32 = 2;

/* ism */
pub const IDA_SIMULTANEOUS: i32 = 1;
pub const IDA_STAGGERED: i32 = 2;

/* DQtype */
pub const IDA_CENTERED: i32 = 1;
pub const IDA_FORWARD: i32 = 2;

/* interp */
pub const IDA_HERMITE: i32 = 1;
pub const IDA_POLYNOMIAL: i32 = 2;

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

pub const IDA_NO_QUAD: i32 = -30;
pub const IDA_QRHS_FAIL: i32 = -31;
pub const IDA_FIRST_QRHS_ERR: i32 = -32;
pub const IDA_REP_QRHS_ERR: i32 = -33;

pub const IDA_NO_SENS: i32 = -40;
pub const IDA_SRES_FAIL: i32 = -41;
pub const IDA_REP_SRES_ERR: i32 = -42;
pub const IDA_BAD_IS: i32 = -43;

pub const IDA_NO_QUADSENS: i32 = -50;
pub const IDA_QSRHS_FAIL: i32 = -51;
pub const IDA_FIRST_QSRHS_ERR: i32 = -52;
pub const IDA_REP_QSRHS_ERR: i32 = -53;

pub const IDA_TOO_CLOSE: i32 = -60;

pub const IDA_UNRECOGNIZED_ERROR: i32 = -99;

/* adjoint return values */
pub const IDA_NO_ADJ: i32 = -101;
pub const IDA_NO_FWD: i32 = -102;
pub const IDA_NO_BCK: i32 = -103;
pub const IDA_BAD_TB0: i32 = -104;
pub const IDA_REIFWD_FAIL: i32 = -105;
pub const IDA_FWD_FAIL: i32 = -106;
pub const IDA_GETY_BADT: i32 = -107;

/* ------------------------------
 * User-Supplied Function Types
 * ------------------------------
 * Plain `fn` pointers matching the C signature argument-for-argument
 * (same names, same order). C `void* user_data` becomes
 * `&mut Option<Box<dyn Any>>`; C `N_Vector*` vector arrays become
 * `&[N_Vector]` (the Python-binding `_1d` name suffixes are dropped).
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

pub type IDAQuadRhsFn = fn(
    tres: sunrealtype,
    yy: &N_Vector,
    yp: &N_Vector,
    rrQ: &N_Vector,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32;

pub type IDASensResFn = fn(
    Ns: i32,
    t: sunrealtype,
    yy: &N_Vector,
    yp: &N_Vector,
    resval: &N_Vector,
    yyS: &[N_Vector],
    ypS: &[N_Vector],
    resvalS: &[N_Vector],
    user_data: &mut Option<Box<dyn Any>>,
    tmp1: &N_Vector,
    tmp2: &N_Vector,
    tmp3: &N_Vector,
) -> i32;

pub type IDAQuadSensRhsFn = fn(
    Ns: i32,
    t: sunrealtype,
    yy: &N_Vector,
    yp: &N_Vector,
    yyS: &[N_Vector],
    ypS: &[N_Vector],
    rrQ: &N_Vector,
    rhsvalQS: &[N_Vector],
    user_data: &mut Option<Box<dyn Any>>,
    yytmp: &N_Vector,
    yptmp: &N_Vector,
    tmpQS: &N_Vector,
) -> i32;

pub type IDAResFnB = fn(
    tt: sunrealtype,
    yy: &N_Vector,
    yp: &N_Vector,
    yyB: &N_Vector,
    ypB: &N_Vector,
    rrB: &N_Vector,
    user_dataB: &mut Option<Box<dyn Any>>,
) -> i32;

pub type IDAResFnBS = fn(
    t: sunrealtype,
    yy: &N_Vector,
    yp: &N_Vector,
    yyS: &[N_Vector],
    ypS: &[N_Vector],
    yyB: &N_Vector,
    ypB: &N_Vector,
    rrBS: &N_Vector,
    user_dataB: &mut Option<Box<dyn Any>>,
) -> i32;

pub type IDAQuadRhsFnB = fn(
    tt: sunrealtype,
    yy: &N_Vector,
    yp: &N_Vector,
    yyB: &N_Vector,
    ypB: &N_Vector,
    rhsvalBQ: &N_Vector,
    user_dataB: &mut Option<Box<dyn Any>>,
) -> i32;

pub type IDAQuadRhsFnBS = fn(
    t: sunrealtype,
    yy: &N_Vector,
    yp: &N_Vector,
    yyS: &[N_Vector],
    ypS: &[N_Vector],
    yyB: &N_Vector,
    ypB: &N_Vector,
    rhsvalBQS: &N_Vector,
    user_dataB: &mut Option<Box<dyn Any>>,
) -> i32;

/* =================================================================
 * Internal constants (idas_impl.h)
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

pub const IDA_QRHS_RECVR: i32 = 10;
pub const IDA_SRES_RECVR: i32 = 11;
pub const IDA_QSRHS_RECVR: i32 = 12;

/* itol */
pub const IDA_NN: i32 = 0;
pub const IDA_SS: i32 = 1;
pub const IDA_SV: i32 = 2;
pub const IDA_WF: i32 = 3;
pub const IDA_EE: i32 = 4;

/* =================================================================
 * idas.c module-scope constants (fragment-file protocol: every
 * fragment of idas.c shares these single definitions)
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

/* Control constants for sensitivity DQ */
pub const CENTERED1: i32 = 1;
pub const CENTERED2: i32 = 2;
pub const FORWARD1: i32 = 3;
pub const FORWARD2: i32 = 4;

/* IDACreate default values */
pub const MXNCF: i32 = 10; /* max number of convergence failures allowed */
pub const MXNEF: i32 = 10; /* max number of error test failures allowed  */
pub const MAXNH: i32 = 5; /* max. number of h tries in IC calc. */
pub const MAXNJ: i32 = 4; /* max. number of J tries in IC calc. */
pub const MAXNI: i32 = 10; /* max. Newton iterations in IC calc. */
pub const EPCON: sunrealtype = 0.33; /* Newton convergence test constant */
pub const MAXBACKS: i32 = 100; /* max backtracks per Newton step in IDACalcIC */

/* =================================================================
 * IDACalcIC constants (module-scope in idas_ic.c; shared here —
 * fragment protocol. ZERO/HALF/ONE/TWO/PT1/PT001 are identical to the
 * idas.c definitions above.)
 * =================================================================*/

pub const PT99: sunrealtype = 0.99; /* real 0.99 */

pub const ICRATEMAX: sunrealtype = 0.9; /* max. Newton conv. rate */
pub const ALPHALS: sunrealtype = 0.0001; /* alpha in linesearch conv. test */

/* IDACalcIC control constants */
pub const IC_FAIL_RECOV: i32 = 1;
pub const IC_CONSTR_FAILED: i32 = 2;
pub const IC_LINESRCH_FAILED: i32 = 3;
pub const IC_CONV_FAIL: i32 = 4;
pub const IC_SLOW_CONVRG: i32 = 5;

/* =================================================================
 * idas_io.c module-scope constant (ZERO/HALF/ONE as above)
 * =================================================================*/

pub const TWOPT5: sunrealtype = 2.5;

/* =================================================================
 * idaa.c module-scope constant (ZERO/ONE/TWO/HUNDRED as above)
 * =================================================================*/

pub const FUZZ_FACTOR: sunrealtype = 1000000.0; /* fuzz factor for IDAAgetY */

/* =================================================================
 * idas_nls.c / idas_nls_sim.c / idas_nls_stg.c shared constants
 * (PT0001/ONE/TWENTY as above)
 * =================================================================*/

pub const MAXIT: i32 = 4; /* default max number of nonlinear iterations    */
pub const RATEMAX: sunrealtype = 0.9; /* max convergence rate used in divergence check */

/* =================================================================
 * idas_ls.c module-scope constants (ZERO/PT9/ONE/TWO as above)
 * =================================================================*/

pub const MAX_ITERS: i32 = 3; /* max. number of attempts to recover in DQ J*v */
pub const PT25: sunrealtype = 0.25;
pub const PT05: sunrealtype = 0.05;

/* =================================================================
 * Sensitivity parameter array (C `sunrealtype* ida_p`)
 * =================================================================*/

/// Shared handle on the problem parameter array `p` of `F(t,y,y',p)`.
///
/// `IDASetSensParams` stores the caller's POINTER in `IDA_mem->ida_p`
/// (`idas_io.c`: `IDA_mem->ida_p = p;`), and the internal difference-quotient
/// sensitivity routines (`IDASensRes1DQ`, `IDAQuadSensRhs1InternalDQ`) perturb
/// `ida_p[which]` IN PLACE around each call to the user's `res` / `rhsQ`. The
/// user callback, reading the same memory through `user_data`, therefore sees
/// the perturbed parameter — that aliasing IS the DQ mechanism. The port
/// reproduces the shared pointer with the workspace handle model (identical to
/// `cvodes_impl::SensParams`, ARCHITECTURE §8): the caller keeps a clone of
/// this `Rc` inside its user data and hands an identical clone to
/// `IDASetSensParams`.
///
/// Borrow discipline (same rule as every other RefCell in the port): never
/// hold a borrow of this cell across a user callback — write the perturbed
/// value, drop the borrow, call, then re-borrow to restore.
pub type SensParams = Rc<RefCell<Vec<sunrealtype>>>;

/* =================================================================
 * Main integrator memory block (idas_impl.h)
 * =================================================================*/

pub struct IDAMemRec {
    pub ida_sunctx: SUNContext,

    /* C `void* python` is omitted — Python bindings are out of scope */
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

    /*-----------------------
    Quadrature Related Data
    -----------------------*/
    pub ida_quadr: sunbooleantype,

    pub ida_rhsQ: Option<IDAQuadRhsFn>,
    pub ida_user_dataQ: Option<Box<dyn Any>>,

    pub ida_errconQ: sunbooleantype,

    pub ida_itolQ: i32,
    pub ida_rtolQ: sunrealtype,
    pub ida_SatolQ: sunrealtype,       /* scalar absolute tolerance for quadratures  */
    pub ida_VatolQ: Option<N_Vector>,  /* vector absolute tolerance for quadratures  */
    pub ida_atolQmin0: sunbooleantype, /* flag indicating that min(atolQ) = 0        */

    /*------------------------
    Sensitivity Related Data
    ------------------------*/
    pub ida_sensi: sunbooleantype,
    pub ida_Ns: i32,
    pub ida_ism: i32,

    pub ida_resS: Option<IDASensResFn>,
    pub ida_user_dataS: Option<Box<dyn Any>>, /* data pointer passed to resS (holds an
                                              IDAMem clone when ida_resSDQ)          */
    pub ida_resSDQ: sunbooleantype,

    pub ida_p: Option<SensParams>, /* parameters in F(t,y,y',p) (SHARED with the
                                   caller's user data; `None` = C NULL)      */
    pub ida_pbar: Vec<sunrealtype>,
    pub ida_plist: Vec<i32>,
    pub ida_DQtype: i32,
    pub ida_DQrhomax: sunrealtype,

    pub ida_errconS: sunbooleantype, /* SUNTRUE if sensitivities in err. control  */

    pub ida_itolS: i32,
    pub ida_rtolS: sunrealtype,            /* relative tolerance for sensitivities    */
    pub ida_SatolS: Vec<sunrealtype>,      /* scalar absolute tolerances for sensi.   */
    pub ida_VatolS: Vec<N_Vector>,         /* vector absolute tolerances for sensi.   */
    pub ida_atolSmin0: Vec<sunbooleantype>, /* flag indicating that min(atolS[is]) = 0 */

    /*-----------------------------------
    Quadrature Sensitivity Related Data
    -----------------------------------*/
    pub ida_quadr_sensi: sunbooleantype, /* SUNTRUE if computing sensitivities of quadrs. */

    pub ida_rhsQS: Option<IDAQuadSensRhsFn>, /* fQS = (dfQ/dy)*yS + (dfQ/dp)          */
    pub ida_user_dataQS: Option<Box<dyn Any>>, /* data pointer passed to fQS (holds an
                                               IDAMem clone when ida_rhsQSDQ)        */
    pub ida_rhsQSDQ: sunbooleantype,         /* SUNTRUE if using internal DQ functions */

    pub ida_errconQS: sunbooleantype, /* SUNTRUE if yQS are considered in err. con.    */

    pub ida_itolQS: i32,
    pub ida_rtolQS: sunrealtype,        /* relative tolerance for yQS                */
    pub ida_SatolQS: Vec<sunrealtype>,  /* scalar absolute tolerances for yQS        */
    pub ida_VatolQS: Vec<N_Vector>,     /* vector absolute tolerances for yQS        */
    pub ida_atolQSmin0: Vec<sunbooleantype>, /* flag indicating that min(atolQS[is]) = 0  */

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

    /*----------------------------
    Quadrature Related N_Vectors
    ----------------------------*/
    pub ida_phiQ: [Option<N_Vector>; MXORDP1],
    pub ida_yyQ: Option<N_Vector>,
    pub ida_ypQ: Option<N_Vector>,
    pub ida_ewtQ: Option<N_Vector>,
    pub ida_eeQ: Option<N_Vector>,

    /*---------------------------
    Sensitivity Related Vectors
    ---------------------------*/
    pub ida_phiS: [Vec<N_Vector>; MXORDP1],
    pub ida_ewtS: Vec<N_Vector>,

    pub ida_eeS: Vec<N_Vector>, /* cumulative sensitivity corrections            */

    pub ida_yyS: Vec<N_Vector>,        /* allocated and used for:                       */
    pub ida_ypS: Vec<N_Vector>,        /*                 ism = SIMULTANEOUS            */
    pub ida_yySpredict: Vec<N_Vector>, /*                 ism = STAGGERED               */
    pub ida_ypSpredict: Vec<N_Vector>,
    pub ida_deltaS: Vec<N_Vector>,

    pub ida_tmpS1: Option<N_Vector>, /* work space vectors  | tmpS1 = tempv1          */
    pub ida_tmpS2: Option<N_Vector>, /* for resS            | tmpS2 = tempv2          */
    pub ida_tmpS3: Option<N_Vector>, /*                     | tmpS3 = allocated       */

    pub ida_savresS: Vec<N_Vector>, /* work vector in IDACalcIC for stg (= phiS[2])  */
    pub ida_delnewS: Vec<N_Vector>, /* work vector in IDACalcIC for stg (= phiS[3])  */

    pub ida_yyS0: Vec<N_Vector>, /* initial yS, ypS vectors allocated and         */
    pub ida_ypS0: Vec<N_Vector>, /* deallocated in IDACalcIC function             */

    pub ida_yyS0new: Vec<N_Vector>, /* work vector in IDASensLineSrch   (= phiS[4])  */
    pub ida_ypS0new: Vec<N_Vector>, /* work vector in IDASensLineSrch   (= eeS)      */

    /*--------------------------------------
    Quadrature Sensitivity Related Vectors
    --------------------------------------*/
    pub ida_phiQS: [Vec<N_Vector>; MXORDP1], /* Mod. div. diffs. for quadr. sensitivities   */
    pub ida_ewtQS: Vec<N_Vector>,            /* error weight vectors for sensitivities      */

    pub ida_eeQS: Vec<N_Vector>, /* cumulative quadr.sensi.corrections          */

    pub ida_yyQS: Vec<N_Vector>,     /* Unlike yS, yQS is not allocated by the user */
    pub ida_tempvQS: Vec<N_Vector>,  /* temporary storage vector (~ tempv)          */
    pub ida_savrhsQ: Option<N_Vector>, /* saved quadr. rhs (needed for rhsQS calls)   */

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

    pub ida_ssS: sunrealtype,     /* scalar ss for staggered sensitivities             */
    pub ida_delnrmS: sunrealtype, /* norm of current staggered sensitivity update      */

    /*------
    Limits
    ------*/
    pub ida_maxncf: i32, /* max number of convergence failures                */
    pub ida_maxnef: i32, /* max number of error test failures                 */

    pub ida_maxord: i32,           /* max value of method order k:                      */
    pub ida_maxord_alloc: i32,     /* value of maxord used when allocating memory       */
    pub ida_mxstep: i64,           /* max number of internal steps for one user call    */
    pub ida_hmax_inv: sunrealtype, /* inverse of max. step size hmax (default = 0.0)    */
    pub ida_hmin: sunrealtype,     /* min step size hmin (default = 0.0)                */

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
    pub ida_nst: i64, /* number of internal steps taken                    */

    pub ida_nre: i64, /* number of function (res) calls                    */
    pub ida_nrQe: i64,
    pub ida_nrSe: i64,
    pub ida_nrQSe: i64, /* number of fQS calls                               */
    pub ida_nreS: i64,
    pub ida_nrQeS: i64, /* number of fQ calls from sensi DQ                  */

    pub ida_ncfn: i64, /* number of corrector convergence failures          */
    pub ida_ncfnQ: i64,
    pub ida_ncfnS: i64,

    pub ida_netf: i64, /* number of error test failures                     */
    pub ida_netfQ: i64,
    pub ida_netfS: i64,
    pub ida_netfQS: i64, /* number of quadr. sensi. error test failures  */

    pub ida_nni: i64, /* number of Newton iterations performed             */
    pub ida_nniS: i64,

    pub ida_nnf: i64, /* number of Newton convergence failures             */
    pub ida_nnfS: i64,

    pub ida_nsetups: i64, /* number of lsetup calls                            */
    pub ida_nsetupsS: i64,

    /*------------------
    Space requirements
    ------------------*/
    pub ida_lrw1: sunindextype, /* no. of sunrealtype words in 1 N_Vector            */
    pub ida_liw1: sunindextype, /* no. of integer words in 1 N_Vector                */
    pub ida_lrw1Q: sunindextype,
    pub ida_liw1Q: sunindextype,
    pub ida_lrw: i64, /* number of sunrealtype words in IDA work vectors   */
    pub ida_liw: i64, /* no. of integer words in IDA work vectors          */

    pub ida_tolsf: sunrealtype, /* tolerance scale factor (saved value)              */

    /* Flags to verify correct calling sequence */
    pub ida_SetupDone: sunbooleantype, /* set to SUNFALSE by IDAMalloc and IDAReInit
                                       set to SUNTRUE by IDACalcIC or IDASolve      */

    pub ida_VatolMallocDone: sunbooleantype,
    pub ida_idMallocDone: sunbooleantype,

    pub ida_MallocDone: sunbooleantype, /* set to SUNFALSE by IDACreate
                                        set to SUNTRUE by IDAMAlloc
                                        tested by IDAReInit and IDASolve             */

    pub ida_VatolQMallocDone: sunbooleantype,
    pub ida_quadMallocDone: sunbooleantype,

    pub ida_VatolSMallocDone: sunbooleantype,
    pub ida_SatolSMallocDone: sunbooleantype,
    pub ida_sensMallocDone: sunbooleantype,

    pub ida_VatolQSMallocDone: sunbooleantype,
    pub ida_SatolQSMallocDone: sunbooleantype,
    pub ida_quadSensMallocDone: sunbooleantype,

    /*---------------------
    Nonlinear Solver Data
    ---------------------*/
    pub NLS: Option<SUNNonlinearSolver>, /* nonlinear solver object */
    pub ownNLS: sunbooleantype,          /* flag indicating NLS ownership */

    pub NLSsim: Option<SUNNonlinearSolver>, /* nonlinear solver object for DAE+Sens solves
                                            with the simultaneous corrector option */
    pub ownNLSsim: sunbooleantype,          /* flag indicating NLS ownership */

    pub NLSstg: Option<SUNNonlinearSolver>, /* nonlinear solver object for DAE+Sens solves
                                            with the staggered corrector option */
    pub ownNLSstg: sunbooleantype,          /* flag indicating NLS ownership */

    /* The following vectors are NVector wrappers for use with the simultaneous
    and staggered corrector methods:

      Simult:  ypredictSim = [ida_delta, ida_deltaS]
               ycorSim     = [ida_ee,    ida_eeS]
               ewtSim      = [ida_ewt,   ida_ewtS]

      Stagger: ypredictStg = ida_deltaS
               ycorStg     = ida_eeS
               ewtStg      = ida_ewtS
    */
    pub ypredictSim: Option<N_Vector>,
    pub ycorSim: Option<N_Vector>,
    pub ewtSim: Option<N_Vector>,
    pub ypredictStg: Option<N_Vector>,
    pub ycorStg: Option<N_Vector>,
    pub ewtStg: Option<N_Vector>,

    /* flags indicating if vector wrappers for the simultaneous and staggered
    correctors have been allocated */
    pub simMallocDone: sunbooleantype,
    pub stgMallocDone: sunbooleantype,

    pub nls_res: Option<IDAResFn>, /* F(t,y(t),y'(t))=0; used in the nonlinear
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

    /* Flag to request a call to the setup routine */
    pub ida_forceSetup: sunbooleantype,

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

    /* scalar arrays (ida_cvals is heap-allocated in C: MXORDP1 entries,
    grown to Ns*MXORDP1 by IDASensInit) */
    pub ida_cvals: Vec<sunrealtype>,
    pub ida_dvals: [sunrealtype; MAXORD_DEFAULT],

    /* vector arrays (scratch handle arrays; same sizing as ida_cvals) */
    pub ida_Xvecs: Vec<N_Vector>,
    pub ida_Zvecs: Vec<N_Vector>,

    /*------------------------
    Adjoint sensitivity data
    ------------------------*/
    pub ida_adj: sunbooleantype, /* SUNTRUE if performing ASA              */

    pub ida_adj_mem: Option<IDAadjMem>, /* Pointer to adjoint memory structure    */

    pub ida_adjMallocDone: sunbooleantype,
}

pub type IDAMem = Rc<RefCell<IDAMemRec>>;

impl IDAMemRec {
    /// All-zero/None baseline (the C `memset(IDA_mem, 0, …)` in
    /// `IDACreate` before its explicit default assignments; every field
    /// the C code reads is explicitly set there, so the baseline values
    /// are never observable).
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
            ida_quadr: SUNFALSE,
            ida_rhsQ: None,
            ida_user_dataQ: None,
            ida_errconQ: SUNFALSE,
            ida_itolQ: 0,
            ida_rtolQ: 0.0,
            ida_SatolQ: 0.0,
            ida_VatolQ: None,
            ida_atolQmin0: SUNFALSE,
            ida_sensi: SUNFALSE,
            ida_Ns: 0,
            ida_ism: 0,
            ida_resS: None,
            ida_user_dataS: None,
            ida_resSDQ: SUNFALSE,
            ida_p: None,
            ida_pbar: Vec::new(),
            ida_plist: Vec::new(),
            ida_DQtype: 0,
            ida_DQrhomax: 0.0,
            ida_errconS: SUNFALSE,
            ida_itolS: 0,
            ida_rtolS: 0.0,
            ida_SatolS: Vec::new(),
            ida_VatolS: Vec::new(),
            ida_atolSmin0: Vec::new(),
            ida_quadr_sensi: SUNFALSE,
            ida_rhsQS: None,
            ida_user_dataQS: None,
            ida_rhsQSDQ: SUNFALSE,
            ida_errconQS: SUNFALSE,
            ida_itolQS: 0,
            ida_rtolQS: 0.0,
            ida_SatolQS: Vec::new(),
            ida_VatolQS: Vec::new(),
            ida_atolQSmin0: Vec::new(),
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
            ida_phiQ: Default::default(),
            ida_yyQ: None,
            ida_ypQ: None,
            ida_ewtQ: None,
            ida_eeQ: None,
            ida_phiS: Default::default(),
            ida_ewtS: Vec::new(),
            ida_eeS: Vec::new(),
            ida_yyS: Vec::new(),
            ida_ypS: Vec::new(),
            ida_yySpredict: Vec::new(),
            ida_ypSpredict: Vec::new(),
            ida_deltaS: Vec::new(),
            ida_tmpS1: None,
            ida_tmpS2: None,
            ida_tmpS3: None,
            ida_savresS: Vec::new(),
            ida_delnewS: Vec::new(),
            ida_yyS0: Vec::new(),
            ida_ypS0: Vec::new(),
            ida_yyS0new: Vec::new(),
            ida_ypS0new: Vec::new(),
            ida_phiQS: Default::default(),
            ida_ewtQS: Vec::new(),
            ida_eeQS: Vec::new(),
            ida_yyQS: Vec::new(),
            ida_tempvQS: Vec::new(),
            ida_savrhsQ: None,
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
            ida_ssS: 0.0,
            ida_delnrmS: 0.0,
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
            ida_nrQe: 0,
            ida_nrSe: 0,
            ida_nrQSe: 0,
            ida_nreS: 0,
            ida_nrQeS: 0,
            ida_ncfn: 0,
            ida_ncfnQ: 0,
            ida_ncfnS: 0,
            ida_netf: 0,
            ida_netfQ: 0,
            ida_netfS: 0,
            ida_netfQS: 0,
            ida_nni: 0,
            ida_nniS: 0,
            ida_nnf: 0,
            ida_nnfS: 0,
            ida_nsetups: 0,
            ida_nsetupsS: 0,
            ida_lrw1: 0,
            ida_liw1: 0,
            ida_lrw1Q: 0,
            ida_liw1Q: 0,
            ida_lrw: 0,
            ida_liw: 0,
            ida_tolsf: 0.0,
            ida_SetupDone: SUNFALSE,
            ida_VatolMallocDone: SUNFALSE,
            ida_idMallocDone: SUNFALSE,
            ida_MallocDone: SUNFALSE,
            ida_VatolQMallocDone: SUNFALSE,
            ida_quadMallocDone: SUNFALSE,
            ida_VatolSMallocDone: SUNFALSE,
            ida_SatolSMallocDone: SUNFALSE,
            ida_sensMallocDone: SUNFALSE,
            ida_VatolQSMallocDone: SUNFALSE,
            ida_SatolQSMallocDone: SUNFALSE,
            ida_quadSensMallocDone: SUNFALSE,
            NLS: None,
            ownNLS: SUNFALSE,
            NLSsim: None,
            ownNLSsim: SUNFALSE,
            NLSstg: None,
            ownNLSstg: SUNFALSE,
            ypredictSim: None,
            ycorSim: None,
            ewtSim: None,
            ypredictStg: None,
            ycorStg: None,
            ewtStg: None,
            simMallocDone: SUNFALSE,
            stgMallocDone: SUNFALSE,
            nls_res: None,
            ida_linit: None,
            ida_lsetup: None,
            ida_lsolve: None,
            ida_lperf: None,
            ida_lfree: None,
            ida_lmem: None,
            ida_dcj: 0.0,
            ida_forceSetup: SUNFALSE,
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
            ida_cvals: Vec::new(),
            ida_dvals: [0.0; MAXORD_DEFAULT],
            ida_Xvecs: Vec::new(),
            ida_Zvecs: Vec::new(),
            ida_adj: SUNFALSE,
            ida_adj_mem: None,
            ida_adjMallocDone: SUNFALSE,
        }
    }
}

/* =================================================================
 * Adjoint module memory block (idas_impl.h)
 * =================================================================*/

/* -----------------------------------------------------------------
 * Types for functions provided by an interpolation module
 * -----------------------------------------------------------------
 * IDAAMMallocFn:  initializes the content field of the structures in
 *                 the dt array
 * IDAAMFreeFn:    deallocates the content field of the structures in
 *                 the dt array
 * IDAAGetYFn:     returns the interpolated forward solution (a C NULL
 *                 `yyS`/`ypS` maps to an empty slice)
 * IDAAStorePntFn: stores a new point in the structure d
 * -----------------------------------------------------------------*/

pub type IDAAMMallocFn = fn(IDA_mem: &IDAMem) -> sunbooleantype;
pub type IDAAMFreeFn = fn(IDA_mem: &IDAMem);
pub type IDAAGetYFn = fn(
    IDA_mem: &IDAMem,
    t: sunrealtype,
    yy: &N_Vector,
    yp: &N_Vector,
    yyS: &[N_Vector],
    ypS: &[N_Vector],
) -> i32;
pub type IDAAStorePntFn = fn(IDA_mem: &IDAMem, d: &IDAdtpntMem) -> i32;

/* -----------------------------------------------------------------
 * Types : struct IDAckpntMemRec, IDAckpntMem
 * -----------------------------------------------------------------
 * Information at a check point needed to 'hot' start IDAS. The C
 * intrusive list link `ck_next` is replaced by the enclosing
 * `IDAadjMemRec::ck_mem` Vec (index 0 = C list head = most recent
 * checkpoint; C `ck = ck->ck_next` ≡ next Vec index).
 * -----------------------------------------------------------------*/

pub struct IDAckpntMemRec {
    /* Integration limits */
    pub ck_t0: sunrealtype,
    pub ck_t1: sunrealtype,

    /* Modified divided difference array */
    pub ck_phi: [Option<N_Vector>; MXORDP1],

    /* Do we need to carry quadratures? */
    pub ck_quadr: sunbooleantype,

    /* Modified divided difference array for quadratures */
    pub ck_phiQ: [Option<N_Vector>; MXORDP1],

    /* Do we need to carry sensitivities? */
    pub ck_sensi: sunbooleantype,

    /* number of sensitivities */
    pub ck_Ns: i32,

    /* Modified divided difference array for sensitivities */
    pub ck_phiS: [Vec<N_Vector>; MXORDP1],

    /* Do we need to carry quadrature sensitivities? */
    pub ck_quadr_sensi: sunbooleantype,

    /* Modified divided difference array for quadrature sensitivities */
    pub ck_phiQS: [Vec<N_Vector>; MXORDP1],

    /* Step data */
    pub ck_nst: i64,
    pub ck_tretlast: sunrealtype,
    pub ck_ns: i32,
    pub ck_kk: i32,
    pub ck_kused: i32,
    pub ck_knew: i32,
    pub ck_phase: i32,

    pub ck_hh: sunrealtype,
    pub ck_hused: sunrealtype,
    pub ck_eta: sunrealtype,
    pub ck_cj: sunrealtype,
    pub ck_cjlast: sunrealtype,
    pub ck_cjold: sunrealtype,
    pub ck_cjratio: sunrealtype,
    pub ck_ss: sunrealtype,
    pub ck_ssS: sunrealtype,

    pub ck_psi: [sunrealtype; MXORDP1],
    pub ck_alpha: [sunrealtype; MXORDP1],
    pub ck_beta: [sunrealtype; MXORDP1],
    pub ck_sigma: [sunrealtype; MXORDP1],
    pub ck_gamma: [sunrealtype; MXORDP1],

    /* How many phi, phiS, phiQ and phiQS were allocated? */
    pub ck_phi_alloc: i32,
}

pub type IDAckpntMem = Rc<RefCell<IDAckpntMemRec>>;

impl IDAckpntMemRec {
    /// All-zero/None baseline (mirrors the C malloc before explicit init).
    pub fn zeroed() -> IDAckpntMemRec {
        IDAckpntMemRec {
            ck_t0: 0.0,
            ck_t1: 0.0,
            ck_phi: Default::default(),
            ck_quadr: SUNFALSE,
            ck_phiQ: Default::default(),
            ck_sensi: SUNFALSE,
            ck_Ns: 0,
            ck_phiS: Default::default(),
            ck_quadr_sensi: SUNFALSE,
            ck_phiQS: Default::default(),
            ck_nst: 0,
            ck_tretlast: 0.0,
            ck_ns: 0,
            ck_kk: 0,
            ck_kused: 0,
            ck_knew: 0,
            ck_phase: 0,
            ck_hh: 0.0,
            ck_hused: 0.0,
            ck_eta: 0.0,
            ck_cj: 0.0,
            ck_cjlast: 0.0,
            ck_cjold: 0.0,
            ck_cjratio: 0.0,
            ck_ss: 0.0,
            ck_ssS: 0.0,
            ck_psi: [0.0; MXORDP1],
            ck_alpha: [0.0; MXORDP1],
            ck_beta: [0.0; MXORDP1],
            ck_sigma: [0.0; MXORDP1],
            ck_gamma: [0.0; MXORDP1],
            ck_phi_alloc: 0,
        }
    }
}

/* -----------------------------------------------------------------
 * Type : struct IDAdtpntMemRec
 * -----------------------------------------------------------------
 * Information at a data point needed to interpolate the solution of
 * forward simulations. `content` holds an `IDAhermiteDataMemRec` or an
 * `IDApolynomialDataMemRec` BY VALUE depending on ia_interpType
 * (C `void* content`).
 * -----------------------------------------------------------------*/

pub struct IDAdtpntMemRec {
    pub t: sunrealtype,                /* time */
    pub content: Option<Box<dyn Any>>, /* interpType-dependent content */
}

pub type IDAdtpntMem = Rc<RefCell<IDAdtpntMemRec>>;

/* Data for cubic Hermite interpolation */
pub struct IDAhermiteDataMemRec {
    pub y: Option<N_Vector>,
    pub yd: Option<N_Vector>,
    pub yS: Vec<N_Vector>,
    pub ySd: Vec<N_Vector>,
}

/* Data for polynomial interpolation */
pub struct IDApolynomialDataMemRec {
    pub y: Option<N_Vector>,
    pub yS: Vec<N_Vector>,

    /* yd and ySd store the derivative(s) only for the first dt
    point. None/empty otherwise. */
    pub yd: Option<N_Vector>,
    pub ySd: Vec<N_Vector>,
    pub order: i32,
}

/* -----------------------------------------------------------------
 * Type : struct IDABMemRec
 * -----------------------------------------------------------------
 * Information for ONE backward problem. The C intrusive list link
 * `ida_next` is replaced by the enclosing `IDAadjMemRec::IDAB_mem` Vec
 * (index 0 = C list head = most recently created backward problem).
 * -----------------------------------------------------------------*/

pub struct IDABMemRec {
    /* Index of this backward problem */
    pub ida_index: i32,

    /* Time at which the backward problem is initialized. */
    pub ida_t0: sunrealtype,

    /* Memory for this backward problem */
    pub IDA_mem: Option<IDAMem>,

    /* Flags to indicate that this backward problem's RHS or quad RHS
     * require forward sensitivities */
    pub ida_res_withSensi: sunbooleantype,
    pub ida_rhsQ_withSensi: sunbooleantype,

    /* Residual function for backward run */
    pub ida_res: Option<IDAResFnB>,
    pub ida_resS: Option<IDAResFnBS>,

    /* Right hand side quadrature function (fQB) for backward run */
    pub ida_rhsQ: Option<IDAQuadRhsFnB>,
    pub ida_rhsQS: Option<IDAQuadRhsFnBS>,

    /* User user_data */
    pub ida_user_data: Option<Box<dyn Any>>,

    /* Memory block for a linear solver's interface to IDAA */
    pub ida_lmem: Option<Box<dyn Any>>,

    /* Function to free any memory allocated by the linear solver */
    pub ida_lfree: Option<fn(IDAB_mem: &IDABMem) -> i32>,

    /* Memory block for a preconditioner's module interface to IDAA */
    pub ida_pmem: Option<Box<dyn Any>>,

    /* Function to free any memory allocated by the preconditioner module */
    pub ida_pfree: Option<fn(IDAB_mem: &IDABMem) -> i32>,

    /* Time at which to extract solution / quadratures */
    pub ida_tout: sunrealtype,

    /* Workspace Nvectors */
    pub ida_yy: Option<N_Vector>,
    pub ida_yp: Option<N_Vector>,
}

pub type IDABMem = Rc<RefCell<IDABMemRec>>;

impl IDABMemRec {
    /// All-zero/None baseline (mirrors the C malloc before explicit init).
    pub fn zeroed() -> IDABMemRec {
        IDABMemRec {
            ida_index: 0,
            ida_t0: 0.0,
            IDA_mem: None,
            ida_res_withSensi: SUNFALSE,
            ida_rhsQ_withSensi: SUNFALSE,
            ida_res: None,
            ida_resS: None,
            ida_rhsQ: None,
            ida_rhsQS: None,
            ida_user_data: None,
            ida_lmem: None,
            ida_lfree: None,
            ida_pmem: None,
            ida_pfree: None,
            ida_tout: 0.0,
            ida_yy: None,
            ida_yp: None,
        }
    }
}

/* -----------------------------------------------------------------
 * Type : struct IDAadjMemRec
 * -----------------------------------------------------------------
 * All information necessary for adjoint sensitivity analysis.
 * -----------------------------------------------------------------*/

pub struct IDAadjMemRec {
    /* --------------------
     * Forward problem data
     * -------------------- */

    /* Integration interval */
    pub ia_tinitial: sunrealtype,
    pub ia_tfinal: sunrealtype,

    /* Flag for first call to IDASolveF */
    pub ia_firstIDAFcall: sunbooleantype,

    /* Flag if IDASolveF was called with TSTOP */
    pub ia_tstopIDAFcall: sunbooleantype,
    pub ia_tstopIDAF: sunrealtype,

    /* Flag if IDASolveF was called in IDA_NORMAL_MODE and encountered
    a root after tout */
    pub ia_rootret: sunbooleantype,
    pub ia_troot: sunrealtype,

    /* ----------------------
     * Backward problems data
     * ---------------------- */

    /* Storage for backward problems (C linked list head = index 0) */
    pub IDAB_mem: Vec<IDABMem>,

    /* Number of backward problems. */
    pub ia_nbckpbs: i32,

    /* Address of current backward problem (iterator). */
    pub ia_bckpbCrt: Option<IDABMem>,

    /* Flag for first call to IDASolveB */
    pub ia_firstIDABcall: sunbooleantype,

    /* ----------------
     * Check point data
     * ---------------- */

    /* Storage for check point information (C linked list head = index 0,
    i.e. most recent checkpoint first, t0 checkpoint last) */
    pub ck_mem: Vec<IDAckpntMem>,

    /* address of the check point structure for which data is available */
    pub ia_ckpntData: Option<IDAckpntMem>,

    /* Number of checkpoints. */
    pub ia_nckpnts: i32,

    /* ------------------
     * Interpolation data
     * ------------------ */

    /* Number of steps between 2 check points */
    pub ia_nsteps: i64,

    /* Last index used in IDAAfindIndex */
    pub ia_ilast: i64,

    /* Storage for data from forward runs */
    pub dt_mem: Vec<IDAdtpntMem>,

    /* Actual number of data points saved in current dt_mem */
    /* Commonly, np = nsteps+1                              */
    pub ia_np: i64,

    /* Interpolation type */
    pub ia_interpType: i32,

    /* Functions set by the interpolation module */
    pub ia_storePnt: Option<IDAAStorePntFn>, /* store a new interpolation point */
    pub ia_getY: Option<IDAAGetYFn>,         /* interpolate forward solution    */
    pub ia_malloc: Option<IDAAMMallocFn>,    /* allocate new data point         */
    pub ia_free: Option<IDAAMFreeFn>,        /* destroys data point             */

    /* Flags controlling the interpolation module */
    pub ia_mallocDone: sunbooleantype,  /* IM initialized?                */
    pub ia_newData: sunbooleantype,     /* new data available in dt_mem?  */
    pub ia_storeSensi: sunbooleantype,  /* store sensitivities?           */
    pub ia_interpSensi: sunbooleantype, /* interpolate sensitivities?     */

    pub ia_noInterp: sunbooleantype, /* interpolations are temporarily */
    /* disabled ( IDACalcICB )        */

    /* Workspace for polynomial interpolation */
    pub ia_Y: [Option<N_Vector>; MXORDP1], /* pointers  phi[i]               */
    pub ia_YS: [Vec<N_Vector>; MXORDP1],   /* pointers phiS[i]               */
    pub ia_T: [sunrealtype; MXORDP1],

    /* Workspace for wrapper functions */
    pub ia_yyTmp: Option<N_Vector>,
    pub ia_ypTmp: Option<N_Vector>,
    pub ia_yySTmp: Vec<N_Vector>,
    pub ia_ypSTmp: Vec<N_Vector>,
}

pub type IDAadjMem = Rc<RefCell<IDAadjMemRec>>;

impl IDAadjMemRec {
    /// All-zero/None baseline (mirrors the C malloc before explicit init).
    pub fn zeroed() -> IDAadjMemRec {
        IDAadjMemRec {
            ia_tinitial: 0.0,
            ia_tfinal: 0.0,
            ia_firstIDAFcall: SUNFALSE,
            ia_tstopIDAFcall: SUNFALSE,
            ia_tstopIDAF: 0.0,
            ia_rootret: SUNFALSE,
            ia_troot: 0.0,
            IDAB_mem: Vec::new(),
            ia_nbckpbs: 0,
            ia_bckpbCrt: None,
            ia_firstIDABcall: SUNFALSE,
            ck_mem: Vec::new(),
            ia_ckpntData: None,
            ia_nckpnts: 0,
            ia_nsteps: 0,
            ia_ilast: 0,
            dt_mem: Vec::new(),
            ia_np: 0,
            ia_interpType: 0,
            ia_storePnt: None,
            ia_getY: None,
            ia_malloc: None,
            ia_free: None,
            ia_mallocDone: SUNFALSE,
            ia_newData: SUNFALSE,
            ia_storeSensi: SUNFALSE,
            ia_interpSensi: SUNFALSE,
            ia_noInterp: SUNFALSE,
            ia_Y: Default::default(),
            ia_YS: Default::default(),
            ia_T: [0.0; MXORDP1],
            ia_yyTmp: None,
            ia_ypTmp: None,
            ia_yySTmp: Vec::new(),
            ia_ypSTmp: Vec::new(),
        }
    }
}

/* -----------------------------------------------------------------
 * IDAadjCheckPointRec (public, include/idas/idas.h) — the C void*
 * checkpoint addresses map to checkpoint handles (Rc identity).
 * -----------------------------------------------------------------*/

pub struct IDAadjCheckPointRec {
    pub my_addr: Option<IDAckpntMem>,
    pub next_addr: Option<IDAckpntMem>,
    pub t0: sunrealtype,
    pub t1: sunrealtype,
    pub nstep: i64,
    pub order: i32,
    pub step: sunrealtype,
}

/* =================================================================
 * High level error handler (relocated from idas.c; C varargs map to a
 * pre-formatted msg — call sites use the MSG_* builders below)
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
 * Error messages (idas_impl.h). Parameter-less messages are consts;
 * parameterized ones are builders producing the exact C expansion
 * (including the C header's missing-separator quirks, e.g.
 * `"At " MSG_TIME "the user-provide EwtSet function failed."` has no
 * space/comma after the time value — preserved byte-for-byte).
 *
 * NOTE: the linear-solver interface messages (`MSG_LS_*`, defined in
 * `idas_ls_impl.h`) and the BBD preconditioner messages (`MSGBBD_*`,
 * `idas_bbdpre_impl.h`) fold into `idas_ls.rs` / `idas_bbdpre.rs`
 * respectively, per the impl-header-folds-into-matching-module rule.
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
pub const MSG_BAD_ISM_CONSTR: &str = "Constraints can not be enforced while forward sensitivity \
                                      is used with simultaneous method.";
pub const MSG_LSOLVE_NULL: &str = "The linear solver's solve routine is NULL.";
pub const MSG_LINIT_FAIL: &str = "The linear solver's init routine failed.";
pub const MSG_NLS_INIT_FAIL: &str = "The nonlinear solver's init routine failed.";

pub const MSG_NO_QUAD: &str = "Illegal attempt to call before calling IDAQuadInit.";
pub const MSG_BAD_EWTQ: &str = "Initial ewtQ has component(s) equal to zero (illegal).";
pub const MSG_BAD_ITOLQ: &str =
    "Illegal value for itolQ. The legal values are IDA_SS and IDA_SV.";
pub const MSG_NO_TOLQ: &str =
    "No integration tolerances for quadrature variables have been specified.";
pub const MSG_NULL_ATOLQ: &str = "atolQ = NULL illegal.";
pub const MSG_BAD_RTOLQ: &str = "rtolQ < 0 illegal.";
pub const MSG_BAD_ATOLQ: &str = "atolQ has negative component(s) (illegal).";

pub const MSG_NO_SENSI: &str = "Illegal attempt to call before calling IDASensInit.";
pub const MSG_BAD_EWTS: &str = "Initial ewtS has component(s) equal to zero (illegal).";
pub const MSG_BAD_ITOLS: &str =
    "Illegal value for itolS. The legal values are IDA_SS, IDA_SV, and IDA_EE.";
pub const MSG_NULL_ATOLS: &str = "atolS = NULL illegal.";
pub const MSG_BAD_RTOLS: &str = "rtolS < 0 illegal.";
pub const MSG_BAD_ATOLS: &str = "atolS has negative component(s) (illegal).";
pub const MSG_BAD_PBAR: &str = "pbar has zero component(s) (illegal).";
pub const MSG_BAD_PLIST: &str = "plist has negative component(s) (illegal).";
pub const MSG_BAD_NS: &str = "NS <= 0 illegal.";
pub const MSG_NULL_YYS0: &str = "yyS0 = NULL illegal.";
pub const MSG_NULL_YPS0: &str = "ypS0 = NULL illegal.";
pub const MSG_BAD_ISM: &str =
    "Illegal value for ism. Legal values are: IDA_SIMULTANEOUS and IDA_STAGGERED.";
pub const MSG_BAD_IS: &str = "Illegal value for is.";
pub const MSG_NULL_DKYA: &str = "dkyA = NULL illegal.";
pub const MSG_BAD_DQTYPE: &str =
    "Illegal value for DQtype. Legal values are: IDA_CENTERED and IDA_FORWARD.";
pub const MSG_BAD_DQRHO: &str = "DQrhomax < 0 illegal.";

pub const MSG_NULL_ABSTOLQS: &str = "abstolQS = NULL illegal parameter.";
pub const MSG_BAD_RELTOLQS: &str = "reltolQS < 0 illegal parameter.";
pub const MSG_BAD_ABSTOLQS: &str = "abstolQS has negative component(s) (illegal).";
pub const MSG_NO_QUADSENSI: &str =
    "Forward sensitivity analysis for quadrature variables was not activated.";
pub const MSG_NULL_YQS0: &str = "yQS0 = NULL illegal parameter.";

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

/* Quadrature error messages */
pub fn MSG_EWTQ_NOW_BAD(t: sunrealtype) -> String {
    format!("At {}, a component of ewtQ has become <= 0.", MSG_TIME(t))
}

pub fn MSG_QRHSFUNC_FAILED(t: sunrealtype) -> String {
    format!(
        "At {}, the quadrature right-hand side routine failed in an unrecoverable manner.",
        MSG_TIME(t)
    )
}

pub fn MSG_QRHSFUNC_UNREC(t: sunrealtype) -> String {
    format!(
        "At {}, the quadrature right-hand side failed in a recoverable manner, but no recovery \
         is possible.",
        MSG_TIME(t)
    )
}

pub fn MSG_QRHSFUNC_REPTD(t: sunrealtype) -> String {
    format!(
        "At {}repeated recoverable quadrature right-hand side function errors.",
        MSG_TIME(t)
    )
}

pub const MSG_QRHSFUNC_FIRST: &str =
    "The quadrature right-hand side routine failed at the first call.";

/* Sensitivity error messages */
pub const MSG_NULL_P: &str =
    "p = NULL when using internal DQ for sensitivity residual is illegal.";

pub fn MSG_EWTS_NOW_BAD(t: sunrealtype) -> String {
    format!("At {}, a component of ewtS has become <= 0.", MSG_TIME(t))
}

pub fn MSG_SRES_FAILED(t: sunrealtype) -> String {
    format!(
        "At {}, the sensitivity residual routine failed in an unrecoverable manner.",
        MSG_TIME(t)
    )
}

pub fn MSG_SRES_UNREC(t: sunrealtype) -> String {
    format!(
        "At {}, the sensitivity residual failed in a recoverable manner, but no recovery is \
         possible.",
        MSG_TIME(t)
    )
}

pub fn MSG_SRES_REPTD(t: sunrealtype) -> String {
    format!(
        "At {}repeated recoverable sensitivity residual function errors.",
        MSG_TIME(t)
    )
}

/* Quadrature sensitivity error messages */
pub const MSG_NO_TOLQS: &str = "No integration tolerances for quadrature sensitivity variables \
                                have been specified.";
pub const MSG_NULL_RHSQ: &str = "IDAS is expected to use DQ to evaluate the RHS of quad. sensi., \
                                 but quadratures were not initialized.";
pub const MSG_BAD_EWTQS: &str = "Initial ewtQS has component(s) equal to zero (illegal).";

pub fn MSG_EWTQS_NOW_BAD(t: sunrealtype) -> String {
    format!("At {}, a component of ewtQS has become <= 0.", MSG_TIME(t))
}

pub fn MSG_QSRHSFUNC_FAILED(t: sunrealtype) -> String {
    format!(
        "At {}, the sensitivity quadrature right-hand side routine failed in an unrecoverable \
         manner.",
        MSG_TIME(t)
    )
}

pub fn MSG_QSRHSFUNC_REPTD(t: sunrealtype) -> String {
    format!(
        "At {}repeated recoverable sensitivity quadrature right-hand side function errors.",
        MSG_TIME(t)
    )
}

pub const MSG_QSRHSFUNC_FIRST: &str =
    "The quadrature right-hand side routine failed at the first call.";

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

/* =================================================================
 * IDAA error messages (idas_impl.h)
 * =================================================================*/

pub const MSGAM_NULL_IDAMEM: &str = "ida_mem = NULL illegal.";
pub const MSGAM_NO_ADJ: &str = "Illegal attempt to call before calling IDAadjInit.";
pub const MSGAM_BAD_INTERP: &str = "Illegal value for interp.";
pub const MSGAM_BAD_STEPS: &str = "Steps nonpositive illegal.";
pub const MSGAM_BAD_WHICH: &str = "Illegal value for which.";
pub const MSGAM_NO_BCK: &str = "No backward problems have been defined yet.";
pub const MSGAM_NO_FWD: &str = "Illegal attempt to call before calling IDASolveF.";
pub const MSGAM_BAD_TB0: &str = "The initial time tB0 is outside the interval over which the \
                                 forward problem was solved.";
pub const MSGAM_BAD_SENSI: &str = "At least one backward problem requires sensitivities, but they \
                                   were not stored for interpolation.";
pub const MSGAM_BAD_ITASKB: &str =
    "Illegal value for itaskB. Legal values are IDA_NORMAL and IDA_ONE_STEP.";
pub const MSGAM_BAD_TBOUT: &str = "The final time tBout is outside the interval over which the \
                                   forward problem was solved.";

pub fn MSGAM_BACK_ERROR(which: i32) -> String {
    format!("Error occurred while integrating backward problem # {}", which)
}

pub fn MSGAM_BAD_TINTERP(t: sunrealtype) -> String {
    format!("Bad t = {} for interpolation.", sun_format_g(t))
}

pub const MSGAM_BAD_T: &str = "Bad t for interpolation.";
pub const MSGAM_WRONG_INTERP: &str =
    "This function cannot be called for the specified interp type.";
pub const MSGAM_MEM_FAIL: &str = "A memory request failed.";
pub const MSGAM_NO_INITBS: &str = "Illegal attempt to call before calling IDAInitBS.";
