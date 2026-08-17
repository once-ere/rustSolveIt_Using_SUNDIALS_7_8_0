//! Port of `src/cvode/cvode.c` (+ headers folded).
//!
//! Main CVODE integrator: creation/initialization, tolerance functions,
//! rootfinding initialization, the `CVode` driver, `cvStep` and all its
//! helpers, dense output (`CVodeGetDky`), the internal error-weight
//! functions, rootfinding (`cvRcheck1/2/3`, `cvRootfind`), BDF stability
//! limit detection (`cvBDFStab`, `cvSLdet`), and `CVodeFree`.
//!
//! `cvProcessError` lives in `cvode_impl` (shared by all cvode modules).
//! Reference build configuration: SUNDIALS_LOGGING_LEVEL = 2 (SUNLogInfo/
//! SUNLogDebug/SUNLogExtraDebug* call sites omitted; CV_WARNING paths
//! kept), profiling off, error checks off, monitoring ON, fused kernels
//! OFF (the unfused branch is the live code), serial MPI branch.

use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

use crate::cvode_impl::*;
use sundials_core::sundials_context::SUNContext;
use sundials_core::sundials_errors::{SUN_ERR_ARG_CORRUPT, SUN_SUCCESS};
use sundials_core::sundials_math::*;
use sundials_core::sundials_nonlinearsolver::*;
use sundials_core::sundials_nvector::*;
use sundials_core::sundials_types::*;
use sundials_core::sunnonlinsol_newton::SUNNonlinSol_Newton;

/*=================================================================*/
/* CVODE Private Constants                                         */
/*=================================================================*/

const ZERO: sunrealtype = 0.0; /* real 0.0     */
const TINY: sunrealtype = 1.0e-10; /* small number */
const PT1: sunrealtype = 0.1; /* real 0.1     */
const POINT2: sunrealtype = 0.2; /* real 0.2     */
const FOURTH: sunrealtype = 0.25; /* real 0.25    */
const HALF: sunrealtype = 0.5; /* real 0.5     */
const PT9: sunrealtype = 0.9; /* real 0.9     */
const ONE: sunrealtype = 1.0; /* real 1.0     */
const ONEPT5: sunrealtype = 1.50; /* real 1.5     */
const TWO: sunrealtype = 2.0; /* real 2.0     */
const THREE: sunrealtype = 3.0; /* real 3.0     */
const FOUR: sunrealtype = 4.0; /* real 4.0     */
const FIVE: sunrealtype = 5.0; /* real 5.0     */
const TWELVE: sunrealtype = 12.0; /* real 12.0    */
const HUNDRED: sunrealtype = 100.0; /* real 100.0   */

/*=================================================================*/
/* CVODE Routine-Specific Constants                                */
/*=================================================================*/

/* Control constants for lower-level rootfinding functions */
pub const RTFOUND: i32 = 1;
pub const CLOSERT: i32 = 3;

/* Control constants for tolerances */
pub const CV_NN: i32 = 0;
pub const CV_SS: i32 = 1;
pub const CV_SV: i32 = 2;
pub const CV_WF: i32 = 3;

/* Algorithmic constants */
const FUZZ_FACTOR: sunrealtype = 100.0;

const HLB_FACTOR: sunrealtype = 100.0;
const HUB_FACTOR: sunrealtype = 0.1;
const H_BIAS: sunrealtype = HALF;
const MAX_ITERS: i32 = 4;

const CORTES: sunrealtype = 0.1;

/*
 * =================================================================
 * Callback invocation helpers (granular borrow discipline: the box
 * token is taken out of the mem around every user callback call and
 * restored afterwards; no mem borrow is held across the call)
 * =================================================================
 */

/// Invoke the user RHS function `f` (C: `cv_mem->cv_f(t, y, ydot, cv_mem->cv_user_data)`).
fn cv_call_f(cv_mem: &CVodeMem, t: sunrealtype, y: &N_Vector, ydot: &N_Vector) -> i32 {
    let f = cv_mem.borrow().cv_f.expect("cv_f set");
    let mut user_data = cv_mem.borrow_mut().cv_user_data.take();
    let retval = f(t, y, ydot, &mut user_data);
    cv_mem.borrow_mut().cv_user_data = user_data;
    retval
}

/// Invoke the error weight function (C: `cv_mem->cv_efun(ycur, weight, cv_mem->cv_e_data)`).
///
/// In C, `cv_e_data` aliases `cv_user_data` when the user supplied `efun`
/// (`e_data = user_data` in `cvInitialSetup`) and points to `cv_mem` for
/// the default `cvEwtSet`. Box aliasing is impossible in safe Rust, so
/// user-efun call sites pass `cv_user_data` directly; the observable
/// behavior is identical.
fn cv_call_efun(cv_mem: &CVodeMem, ycur: &N_Vector, weight: &N_Vector) -> i32 {
    let (efun, user_efun) = {
        let m = cv_mem.borrow();
        (m.cv_efun, m.cv_user_efun)
    };
    let efun = efun.expect("cv_efun set");
    if user_efun {
        let mut data = cv_mem.borrow_mut().cv_user_data.take();
        let retval = efun(ycur, weight, &mut data);
        cv_mem.borrow_mut().cv_user_data = data;
        retval
    } else {
        let mut data = cv_mem.borrow_mut().cv_e_data.take();
        let retval = efun(ycur, weight, &mut data);
        cv_mem.borrow_mut().cv_e_data = data;
        retval
    }
}

/// Invoke the user root function `g` (C: `cv_mem->cv_gfun(t, y, gout, cv_mem->cv_user_data)`).
fn cv_call_gfun(cv_mem: &CVodeMem, t: sunrealtype, y: &N_Vector, gout: &mut [sunrealtype]) -> i32 {
    let gfun = cv_mem.borrow().cv_gfun.expect("cv_gfun set");
    let mut user_data = cv_mem.borrow_mut().cv_user_data.take();
    let retval = gfun(t, y, gout, &mut user_data);
    cv_mem.borrow_mut().cv_user_data = user_data;
    retval
}

/*
 * =================================================================
 * Exported Functions Implementation
 * =================================================================
 */

/*
 * -----------------------------------------------------------------
 * Creation, allocation and re-initialization functions
 * -----------------------------------------------------------------
 */

/*
 * CVodeCreate
 *
 * CVodeCreate creates an internal memory block for a problem to
 * be solved by CVODE.
 * If successful, CVodeCreate returns a pointer to the problem memory.
 * This pointer should be passed to CVodeInit.
 * If an initialization error occurs, CVodeCreate prints an error
 * message to standard err and returns NULL.
 */

pub fn CVodeCreate(lmm: i32, sunctx: &SUNContext) -> Option<CVodeMem> {
    /* Test inputs */

    if (lmm != CV_ADAMS) && (lmm != CV_BDF) {
        cvProcessError(None, 0, line!() as i32, "CVodeCreate", file!(), MSGCV_BAD_LMM);
        return None;
    }

    /* NULL sunctx check: handled by type system */

    /* malloc failure branch: allocation cannot fail observably in Rust */
    let mut cv_mem = CVodeMemRec::zeroed(sunctx.clone());

    let maxord: i32 = if lmm == CV_ADAMS {
        ADAMS_Q_MAX as i32
    } else {
        BDF_Q_MAX as i32
    };

    /* Copy input parameters into cv_mem */
    cv_mem.cv_lmm = lmm;

    /* Set uround */
    cv_mem.cv_uround = SUN_UNIT_ROUNDOFF;

    /* Set default values for integrator optional inputs */
    cv_mem.cv_f = None;
    cv_mem.cv_user_data = None;
    cv_mem.cv_itol = CV_NN;
    cv_mem.cv_atolmin0 = SUNTRUE;
    cv_mem.cv_user_efun = SUNFALSE;
    cv_mem.cv_efun = None;
    cv_mem.cv_e_data = None;
    cv_mem.cv_monitorfun = None;
    cv_mem.cv_monitor_interval = 0;
    cv_mem.cv_qmax = maxord;
    cv_mem.cv_mxstep = MXSTEP_DEFAULT;
    cv_mem.cv_mxhnil = MXHNIL_DEFAULT;
    cv_mem.cv_sldeton = SUNFALSE;
    cv_mem.cv_hin = ZERO;
    cv_mem.cv_hmin = HMIN_DEFAULT;
    cv_mem.cv_hmax_inv = HMAX_INV_DEFAULT;
    cv_mem.cv_eta_min_fx = ETA_MIN_FX_DEFAULT;
    cv_mem.cv_eta_max_fx = ETA_MAX_FX_DEFAULT;
    cv_mem.cv_eta_max_fs = ETA_MAX_FS_DEFAULT;
    cv_mem.cv_eta_max_es = ETA_MAX_ES_DEFAULT;
    cv_mem.cv_eta_max_gs = ETA_MAX_GS_DEFAULT;
    cv_mem.cv_eta_min = ETA_MIN_DEFAULT;
    cv_mem.cv_eta_min_ef = ETA_MIN_EF_DEFAULT;
    cv_mem.cv_eta_max_ef = ETA_MAX_EF_DEFAULT;
    cv_mem.cv_eta_cf = ETA_CF_DEFAULT;
    cv_mem.cv_small_nst = SMALL_NST_DEFAULT;
    cv_mem.cv_small_nef = SMALL_NEF_DEFAULT;
    cv_mem.cv_tstopset = SUNFALSE;
    cv_mem.cv_tstopinterp = SUNFALSE;
    cv_mem.cv_maxnef = MXNEF;
    cv_mem.cv_maxncf = MXNCF;
    cv_mem.cv_nlscoef = CORTES;
    cv_mem.cv_msbp = MSBP_DEFAULT;
    cv_mem.cv_dgmax_lsetup = DGMAX_LSETUP_DEFAULT;
    cv_mem.convfail = CV_NO_FAILURES;

    /* Initialize inequality constraint variables */
    cv_mem.cv_constraints = None;
    cv_mem.constraint_corrections = 0;
    cv_mem.constraint_fails = 0;
    cv_mem.max_constraint_fails = MAX_CONSTRAINT_FAILS;

    /* Initialize root finding variables */

    cv_mem.cv_glo = Vec::new();
    cv_mem.cv_ghi = Vec::new();
    cv_mem.cv_grout = Vec::new();
    cv_mem.cv_iroots = Vec::new();
    cv_mem.cv_rootdir = Vec::new();
    cv_mem.cv_gfun = None;
    cv_mem.cv_nrtfn = 0;
    cv_mem.cv_gactive = Vec::new();
    cv_mem.cv_mxgnull = 1;

    /* Initialize projection variables */
    cv_mem.proj_mem = None;
    cv_mem.proj_enabled = SUNFALSE;
    cv_mem.proj_applied = SUNFALSE;

    /* Initialize resize variables */
    cv_mem.first_step_after_resize = SUNFALSE;

    /* Set the saved value for qmax_alloc */

    cv_mem.cv_qmax_alloc = maxord;

    /* Initialize lrw and liw */

    cv_mem.cv_lrw = (58 + 2 * L_MAX + NUM_TESTS) as i64;
    cv_mem.cv_liw = 40;

    /* No mallocs have been done yet */

    cv_mem.cv_VabstolMallocDone = SUNFALSE;
    cv_mem.cv_MallocDone = SUNFALSE;

    /* Initialize nonlinear solver variables */
    cv_mem.NLS = None;
    cv_mem.ownNLS = SUNFALSE;

    /* Initialize fused operations variable */
    cv_mem.cv_usefused = SUNFALSE;

    /* Return pointer to CVODE memory block */

    Some(Rc::new(RefCell::new(cv_mem)))
}

/*-----------------------------------------------------------------*/

/*
 * CVodeInit
 *
 * CVodeInit allocates and initializes memory for a problem. All
 * problem inputs are checked for errors. If any error occurs during
 * initialization, it is reported to the file whose file pointer is
 * errfp and an error flag is returned. Otherwise, it returns CV_SUCCESS
 */

pub fn CVodeInit(cvode_mem: &CVodeMem, f: CVRhsFn, t0: sunrealtype, y0: &N_Vector) -> i32 {
    /* NULL-mem check: handled by type system */
    let cv_mem = cvode_mem;

    /* Check for legal input parameters */

    /* NULL y0 check: handled by type system */
    /* NULL f check: handled by type system */

    /* Test if all required vector operations are implemented */

    let nvectorOK = cvCheckNvector(y0);
    if !nvectorOK {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeInit",
            file!(),
            MSGCV_BAD_NVECTOR,
        );
        return CV_ILL_INPUT;
    }

    /* Set space requirements for one N_Vector */

    let mut lrw1: sunindextype = 0;
    let mut liw1: sunindextype = 0;
    if y0.ops.borrow().nvspace.is_some() {
        N_VSpace(y0, &mut lrw1, &mut liw1);
    } else {
        lrw1 = 0;
        liw1 = 0;
    }
    {
        let mut m = cv_mem.borrow_mut();
        m.cv_lrw1 = lrw1;
        m.cv_liw1 = liw1;
    }

    /* Allocate the vectors (using y0 as a template) */

    let allocOK = cvAllocVectors(cv_mem, y0);
    if !allocOK {
        cvProcessError(
            Some(cv_mem),
            CV_MEM_FAIL,
            line!() as i32,
            "CVodeInit",
            file!(),
            MSGCV_MEM_FAIL,
        );
        return CV_MEM_FAIL;
    }

    /* Input checks complete at this point and history array allocated */

    /* Copy the input parameters into CVODE state */
    {
        let mut m = cv_mem.borrow_mut();
        m.cv_f = Some(f);
        m.cv_tn = t0;
    }

    /* Initialize zn[0] in the history array */
    let zn0 = cv_mem.borrow().cv_zn[0].clone().unwrap();
    N_VScale(ONE, y0, &zn0);

    /* create a Newton nonlinear solver object by default */
    let sunctx = cv_mem.borrow().cv_sunctx.clone();
    let NLS = SUNNonlinSol_Newton(y0, &sunctx);

    /* check that nonlinear solver is non-NULL */
    let NLS = match NLS {
        Some(nls) => nls,
        None => {
            cvProcessError(
                Some(cv_mem),
                CV_MEM_FAIL,
                line!() as i32,
                "CVodeInit",
                file!(),
                MSGCV_MEM_FAIL,
            );
            cvFreeVectors(cv_mem);
            return CV_MEM_FAIL;
        }
    };

    /* attach the nonlinear solver to the CVODE memory */
    let retval = crate::cvode_nls::CVodeSetNonlinearSolver(cv_mem, &NLS);

    /* check that the nonlinear solver was successfully attached */
    if retval != CV_SUCCESS {
        cvProcessError(
            Some(cv_mem),
            retval,
            line!() as i32,
            "CVodeInit",
            file!(),
            "Setting the nonlinear solver failed",
        );
        cvFreeVectors(cv_mem);
        let _ = SUNNonlinSolFree(Some(NLS));
        return CV_MEM_FAIL;
    }

    /* set ownership flag */
    cv_mem.borrow_mut().ownNLS = SUNTRUE;

    /* All error checking is complete at this point */

    {
        let mut m = cv_mem.borrow_mut();

        /* Set step parameters */

        m.cv_q = 1;
        m.cv_L = 2;
        m.cv_qwait = m.cv_L;
        m.cv_etamax = m.cv_eta_max_fs;

        m.cv_qu = 0;
        m.cv_hu = ZERO;
        m.cv_tolsf = ONE;

        /* Set the linear solver addresses to NULL.
        (We check != NULL later, in CVode) */

        m.cv_linit = None;
        m.cv_lreinit = None;
        m.cv_lsetup = None;
        m.cv_lsolve = None;
        m.cv_lfree = None;
        m.cv_lmem = None;

        /* Initialize all the counters */

        m.cv_nst = 0;
        m.cv_nfe = 0;
        m.cv_ncfn = 0;
        m.cv_netf = 0;
        m.cv_nni = 0;
        m.cv_nnf = 0;
        m.cv_nsetups = 0;
        m.cv_nhnil = 0;
        m.cv_nstlp = 0;
        m.cv_nscon = 0;
        m.cv_nge = 0;

        m.cv_irfnd = 0;

        /* Initialize other integrator optional outputs */

        m.cv_h0u = ZERO;
        m.cv_next_h = ZERO;
        m.cv_next_q = 0;

        /* Initialize Stablilty Limit Detection data */
        /* NOTE: We do this even if stab lim det was not
        turned on yet. This way, the user can turn it
        on at any time */

        m.cv_nor = 0;
        for i in 1..=5usize {
            for k in 1..=3usize {
                m.cv_ssdat[i - 1][k - 1] = ZERO;
            }
        }

        /* Problem has been successfully initialized */

        m.cv_MallocDone = SUNTRUE;
    }

    CV_SUCCESS
}

/*-----------------------------------------------------------------*/

/*
 * CVodeReInit
 *
 * CVodeReInit re-initializes CVODE's memory for a problem, assuming
 * it has already been allocated in a prior CVodeInit call.
 * All problem specification inputs are checked for errors.
 * If any error occurs during initialization, it is reported to the
 * file whose file pointer is errfp.
 * The return value is CV_SUCCESS = 0 if no errors occurred, or
 * a negative value otherwise.
 */

pub fn CVodeReInit(cvode_mem: &CVodeMem, t0: sunrealtype, y0: &N_Vector) -> i32 {
    /* NULL-mem check: handled by type system */
    let cv_mem = cvode_mem;

    /* Check if cvode_mem was allocated */

    if !cv_mem.borrow().cv_MallocDone {
        cvProcessError(
            Some(cv_mem),
            CV_NO_MALLOC,
            line!() as i32,
            "CVodeReInit",
            file!(),
            MSGCV_NO_MALLOC,
        );
        return CV_NO_MALLOC;
    }

    /* Check for legal input parameters */

    /* NULL y0 check: handled by type system */

    {
        let mut m = cv_mem.borrow_mut();

        /* Copy the input parameters into CVODE state */

        m.cv_tn = t0;

        /* Set step parameters */

        m.cv_q = 1;
        m.cv_L = 2;
        m.cv_qwait = m.cv_L;
        m.cv_etamax = m.cv_eta_max_fs;

        m.cv_qu = 0;
        m.cv_hu = ZERO;
        m.cv_tolsf = ONE;
    }

    /* Initialize zn[0] in the history array */

    let zn0 = cv_mem.borrow().cv_zn[0].clone().unwrap();
    N_VScale(ONE, y0, &zn0);

    {
        let mut m = cv_mem.borrow_mut();

        /* Initialize all the counters */

        m.cv_nst = 0;
        m.cv_nfe = 0;
        m.cv_ncfn = 0;
        m.cv_netf = 0;
        m.cv_nni = 0;
        m.cv_nnf = 0;
        m.cv_nsetups = 0;
        m.cv_nhnil = 0;
        m.cv_nstlp = 0;
        m.cv_nscon = 0;
        m.cv_nge = 0;

        m.cv_irfnd = 0;

        m.constraint_corrections = 0;
        m.constraint_fails = 0;
    }

    let lreinit = cv_mem.borrow().cv_lreinit;
    if let Some(lreinit) = lreinit {
        let _ = lreinit(cv_mem);
    }

    {
        let mut m = cv_mem.borrow_mut();

        /* Initialize other integrator optional outputs */

        m.cv_h0u = ZERO;
        m.cv_next_h = ZERO;
        m.cv_next_q = 0;

        /* Initialize Stablilty Limit Detection data */

        m.cv_nor = 0;
        for i in 1..=5usize {
            for k in 1..=3usize {
                m.cv_ssdat[i - 1][k - 1] = ZERO;
            }
        }
    }

    /* Problem has been successfully re-initialized */

    CV_SUCCESS
}

/*-----------------------------------------------------------------*/

/*
 * CVodeSStolerances
 * CVodeSVtolerances
 * CVodeWFtolerances
 *
 * These functions specify the integration tolerances. One of them
 * MUST be called before the first call to CVode.
 *
 * CVodeSStolerances specifies scalar relative and absolute tolerances.
 * CVodeSVtolerances specifies scalar relative tolerance and a vector
 *   absolute tolerance (a potentially different absolute tolerance
 *   for each vector component).
 * CVodeWFtolerances specifies a user-provides function (of type CVEwtFn)
 *   which will be called to set the error weight vector.
 */

pub fn CVodeSStolerances(cvode_mem: &CVodeMem, reltol: sunrealtype, abstol: sunrealtype) -> i32 {
    /* NULL-mem check: handled by type system */
    let cv_mem = cvode_mem;

    if !cv_mem.borrow().cv_MallocDone {
        cvProcessError(
            Some(cv_mem),
            CV_NO_MALLOC,
            line!() as i32,
            "CVodeSStolerances",
            file!(),
            MSGCV_NO_MALLOC,
        );
        return CV_NO_MALLOC;
    }

    /* Check inputs */

    if reltol < ZERO {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeSStolerances",
            file!(),
            MSGCV_BAD_RELTOL,
        );
        return CV_ILL_INPUT;
    }

    if abstol < ZERO {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeSStolerances",
            file!(),
            MSGCV_BAD_ABSTOL,
        );
        return CV_ILL_INPUT;
    }

    /* Copy tolerances into memory */

    let mut m = cv_mem.borrow_mut();
    m.cv_reltol = reltol;
    m.cv_Sabstol = abstol;
    m.cv_atolmin0 = abstol == ZERO;

    m.cv_itol = CV_SS;

    m.cv_user_efun = SUNFALSE;
    m.cv_efun = Some(cvEwtSet);
    m.cv_e_data = None; /* will be set to cvode_mem in InitialSetup */

    CV_SUCCESS
}

pub fn CVodeSVtolerances(cvode_mem: &CVodeMem, reltol: sunrealtype, abstol: &N_Vector) -> i32 {
    /* NULL-mem check: handled by type system */
    let cv_mem = cvode_mem;

    if !cv_mem.borrow().cv_MallocDone {
        cvProcessError(
            Some(cv_mem),
            CV_NO_MALLOC,
            line!() as i32,
            "CVodeSVtolerances",
            file!(),
            MSGCV_NO_MALLOC,
        );
        return CV_NO_MALLOC;
    }

    /* Check inputs */

    if reltol < ZERO {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeSVtolerances",
            file!(),
            MSGCV_BAD_RELTOL,
        );
        return CV_ILL_INPUT;
    }

    if abstol.ops.borrow().nvmin.is_none() {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeSVtolerances",
            file!(),
            "Missing N_VMin routine from N_Vector",
        );
        return CV_ILL_INPUT;
    }
    let atolmin = N_VMin(abstol);
    if atolmin < ZERO {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeSVtolerances",
            file!(),
            MSGCV_BAD_ABSTOL,
        );
        return CV_ILL_INPUT;
    }

    /* Copy tolerances into memory */

    if !cv_mem.borrow().cv_VabstolMallocDone {
        let ewt = cv_mem.borrow().cv_ewt.clone().unwrap();
        let vabstol = N_VClone(&ewt).unwrap();
        let mut m = cv_mem.borrow_mut();
        m.cv_Vabstol = Some(vabstol);
        m.cv_lrw += m.cv_lrw1;
        m.cv_liw += m.cv_liw1;
        m.cv_VabstolMallocDone = SUNTRUE;
    }

    cv_mem.borrow_mut().cv_reltol = reltol;
    let vabstol = cv_mem.borrow().cv_Vabstol.clone().unwrap();
    N_VScale(ONE, abstol, &vabstol);
    {
        let mut m = cv_mem.borrow_mut();
        m.cv_atolmin0 = atolmin == ZERO;

        m.cv_itol = CV_SV;

        m.cv_user_efun = SUNFALSE;
        m.cv_efun = Some(cvEwtSet);
        m.cv_e_data = None; /* will be set to cvode_mem in InitialSetup */
    }

    CV_SUCCESS
}

pub fn CVodeWFtolerances(cvode_mem: &CVodeMem, efun: CVEwtFn) -> i32 {
    /* NULL-mem check: handled by type system */
    let cv_mem = cvode_mem;

    if !cv_mem.borrow().cv_MallocDone {
        cvProcessError(
            Some(cv_mem),
            CV_NO_MALLOC,
            line!() as i32,
            "CVodeWFtolerances",
            file!(),
            MSGCV_NO_MALLOC,
        );
        return CV_NO_MALLOC;
    }

    let mut m = cv_mem.borrow_mut();
    m.cv_itol = CV_WF;

    m.cv_user_efun = SUNTRUE;
    m.cv_efun = Some(efun);
    m.cv_e_data = None; /* will be set to user_data in InitialSetup */

    CV_SUCCESS
}

/*-----------------------------------------------------------------*/

/*
 * CVodeRootInit
 *
 * CVodeRootInit initializes a rootfinding problem to be solved
 * during the integration of the ODE system.  It loads the root
 * function pointer and the number of root functions, and allocates
 * workspace memory.  The return value is CV_SUCCESS = 0 if no errors
 * occurred, or a negative value otherwise.
 */

pub fn CVodeRootInit(cvode_mem: &CVodeMem, nrtfn: i32, g: Option<CVRootFn>) -> i32 {
    /* NULL-mem check: handled by type system */
    let cv_mem = cvode_mem;

    let nrt = if nrtfn < 0 { 0 } else { nrtfn };

    /* If rerunning CVodeRootInit() with a different number of root
    functions (changing number of gfun components), then free
    currently held memory resources */
    {
        let mut m = cv_mem.borrow_mut();
        if (nrt != m.cv_nrtfn) && (m.cv_nrtfn > 0) {
            m.cv_glo = Vec::new();
            m.cv_ghi = Vec::new();
            m.cv_grout = Vec::new();
            m.cv_iroots = Vec::new();
            m.cv_rootdir = Vec::new();
            m.cv_gactive = Vec::new();

            m.cv_lrw -= 3 * (m.cv_nrtfn as i64);
            m.cv_liw -= 3 * (m.cv_nrtfn as i64);
        }
    }

    /* If CVodeRootInit() was called with nrtfn == 0, then set cv_nrtfn to
    zero and cv_gfun to NULL before returning */
    if nrt == 0 {
        let mut m = cv_mem.borrow_mut();
        m.cv_nrtfn = nrt;
        m.cv_gfun = None;
        return CV_SUCCESS;
    }

    /* If rerunning CVodeRootInit() with the same number of root functions
    (not changing number of gfun components), then check if the root
    function argument has changed */
    /* If g != NULL then return as currently reserved memory resources
    will suffice */
    if nrt == cv_mem.borrow().cv_nrtfn {
        let mut m = cv_mem.borrow_mut();
        /* C compares the root-fn pointers by identity; fn-pointer identity
        in Rust carries the same caveats as C across translation units */
        #[allow(unpredictable_function_pointer_comparisons)]
        if g != m.cv_gfun {
            if g.is_none() {
                m.cv_glo = Vec::new();
                m.cv_ghi = Vec::new();
                m.cv_grout = Vec::new();
                m.cv_iroots = Vec::new();
                m.cv_rootdir = Vec::new();
                m.cv_gactive = Vec::new();

                m.cv_lrw -= 3 * (nrt as i64);
                m.cv_liw -= 3 * (nrt as i64);

                drop(m);
                cvProcessError(
                    Some(cv_mem),
                    CV_ILL_INPUT,
                    line!() as i32,
                    "CVodeRootInit",
                    file!(),
                    MSGCV_NULL_G,
                );
                return CV_ILL_INPUT;
            } else {
                m.cv_gfun = g;
                return CV_SUCCESS;
            }
        } else {
            return CV_SUCCESS;
        }
    }

    /* Set variable values in CVode memory block */
    cv_mem.borrow_mut().cv_nrtfn = nrt;
    if g.is_none() {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeRootInit",
            file!(),
            MSGCV_NULL_G,
        );
        return CV_ILL_INPUT;
    } else {
        cv_mem.borrow_mut().cv_gfun = g;
    }

    /* Allocate necessary memory and return (C allocation-failure branches
    are unreachable in Rust: Vec allocation aborts rather than returning
    NULL) */
    {
        let mut m = cv_mem.borrow_mut();
        m.cv_glo = vec![ZERO; nrt as usize];
        m.cv_ghi = vec![ZERO; nrt as usize];
        m.cv_grout = vec![ZERO; nrt as usize];
        m.cv_iroots = vec![0; nrt as usize];

        /* Set default values for rootdir (both directions) */
        m.cv_rootdir = vec![0; nrt as usize];

        /* Set default values for gactive (all active) */
        m.cv_gactive = vec![SUNTRUE; nrt as usize];

        m.cv_lrw += 3 * (nrt as i64);
        m.cv_liw += 3 * (nrt as i64);
    }

    CV_SUCCESS
}

/*
 * -----------------------------------------------------------------
 * Main solver function
 * -----------------------------------------------------------------
 */

/*
 * CVode
 *
 * This routine is the main driver of the CVODE package.
 *
 * It integrates over a time interval defined by the user, by calling
 * cvStep to do internal time steps.
 *
 * The first time that CVode is called for a successfully initialized
 * problem, it computes a tentative initial step size h.
 *
 * CVode supports two modes, specified by itask: CV_NORMAL, CV_ONE_STEP.
 * In the CV_NORMAL mode, the solver steps until it reaches or passes tout
 * and then interpolates to obtain y(tout).
 * In the CV_ONE_STEP mode, it takes one internal step and returns.
 */

pub fn CVode(
    cvode_mem: &CVodeMem,
    tout: sunrealtype,
    yout: &N_Vector,
    tret: &mut sunrealtype,
    itask: i32,
) -> i32 {
    /*
     * -------------------------------------
     * 1. Check and process inputs
     * -------------------------------------
     */

    /* NULL-mem check: handled by type system */
    let cv_mem = cvode_mem;

    /* Check if cvode_mem was allocated */
    if !cv_mem.borrow().cv_MallocDone {
        cvProcessError(
            Some(cv_mem),
            CV_NO_MALLOC,
            line!() as i32,
            "CVode",
            file!(),
            MSGCV_NO_MALLOC,
        );
        return CV_NO_MALLOC;
    }

    /* Check for yout != NULL (cv_y aliases the user's yout: the Rc clone
    shares the underlying data exactly as the C pointer copy does).
    NULL yout check: handled by type system */
    cv_mem.borrow_mut().cv_y = Some(yout.clone());

    /* NULL tret check: handled by type system */

    /* Check for valid itask */
    if (itask != CV_NORMAL) && (itask != CV_ONE_STEP) {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVode",
            file!(),
            MSGCV_BAD_ITASK,
        );
        return CV_ILL_INPUT;
    }

    /*
     * ----------------------------------------
     * 2. Initializations performed only at
     *    the first step (nst=0):
     *    - initial setup
     *    - initialize Nordsieck history array
     *    - compute initial step size
     *    - check for approach to tstop
     *    - check for approach to a root
     *    Or initializations performed after
     *    resizing the integrator
     *    - check constraints
     *    - initialize linear solver
     *    - initialize nonlinear solver
     * ----------------------------------------
     */

    if cv_mem.borrow().cv_nst == 0 {
        {
            let mut m = cv_mem.borrow_mut();
            m.cv_tretlast = m.cv_tn;
            *tret = m.cv_tn;
        }

        /* Check inputs for correctness */

        let ier = cvInitialSetup(cv_mem, tout);
        if ier != CV_SUCCESS {
            return ier;
        }

        /* Call f at (t0,y0), set zn[1] = y'(t0). */

        let (zn0, zn1, tn) = {
            let m = cv_mem.borrow();
            (
                m.cv_zn[0].clone().unwrap(),
                m.cv_zn[1].clone().unwrap(),
                m.cv_tn,
            )
        };
        let retval = cv_call_f(cv_mem, tn, &zn0, &zn1);
        cv_mem.borrow_mut().cv_nfe += 1;
        if retval < 0 {
            cvProcessError(
                Some(cv_mem),
                CV_RHSFUNC_FAIL,
                line!() as i32,
                "CVode",
                file!(),
                &MSGCV_RHSFUNC_FAILED(tn),
            );
            return CV_RHSFUNC_FAIL;
        }
        if retval > 0 {
            cvProcessError(
                Some(cv_mem),
                CV_FIRST_RHSFUNC_ERR,
                line!() as i32,
                "CVode",
                file!(),
                MSGCV_RHSFUNC_FIRST,
            );
            return CV_FIRST_RHSFUNC_ERR;
        }

        /* Test input tstop for legality. */

        {
            let m = cv_mem.borrow();
            if m.cv_tstopset && (m.cv_tstop - m.cv_tn) * (tout - m.cv_tn) <= ZERO {
                let (tstop, tn) = (m.cv_tstop, m.cv_tn);
                drop(m);
                cvProcessError(
                    Some(cv_mem),
                    CV_ILL_INPUT,
                    line!() as i32,
                    "CVode",
                    file!(),
                    &MSGCV_BAD_TSTOP(tstop, tn),
                );
                return CV_ILL_INPUT;
            }
        }

        /* Set initial h (from H0 or cvHin). */

        {
            let mut m = cv_mem.borrow_mut();
            m.cv_h = m.cv_hin;
            if (m.cv_h != ZERO) && ((tout - m.cv_tn) * m.cv_h < ZERO) {
                drop(m);
                cvProcessError(
                    Some(cv_mem),
                    CV_ILL_INPUT,
                    line!() as i32,
                    "CVode",
                    file!(),
                    MSGCV_BAD_H0,
                );
                return CV_ILL_INPUT;
            }
        }
        if cv_mem.borrow().cv_h == ZERO {
            let mut tout_hin = tout;
            {
                let m = cv_mem.borrow();
                if m.cv_tstopset && (tout - m.cv_tn) * (tout - m.cv_tstop) > ZERO {
                    tout_hin = m.cv_tstop;
                }
            }
            let hflag = cvHin(cv_mem, tout_hin);
            if hflag != CV_SUCCESS {
                let istate = cvHandleFailure(cv_mem, hflag);
                return istate;
            }
        }

        {
            let mut m = cv_mem.borrow_mut();

            /* Enforce hmax and hmin */

            let rh = SUNRabs(m.cv_h) * m.cv_hmax_inv;
            if rh > ONE {
                m.cv_h /= rh;
            }
            if SUNRabs(m.cv_h) < m.cv_hmin {
                m.cv_h *= m.cv_hmin / SUNRabs(m.cv_h);
            }

            /* Check for approach to tstop */

            if m.cv_tstopset && (m.cv_tn + m.cv_h - m.cv_tstop) * m.cv_h > ZERO {
                m.cv_h = (m.cv_tstop - m.cv_tn) * (ONE - FOUR * m.cv_uround);
            }

            /* Scale zn[1] by h.*/

            m.cv_hscale = m.cv_h;
            m.cv_h0u = m.cv_h;
            m.cv_hprime = m.cv_h;
        }

        let (h, zn1) = {
            let m = cv_mem.borrow();
            (m.cv_h, m.cv_zn[1].clone().unwrap())
        };
        N_VScale(h, &zn1, &zn1);

        /* Check for zeros of root function g at and near t0. */

        if cv_mem.borrow().cv_nrtfn > 0 {
            let retval = cvRcheck1(cv_mem);

            if retval == CV_RTFUNC_FAIL {
                let tn = cv_mem.borrow().cv_tn;
                cvProcessError(
                    Some(cv_mem),
                    CV_RTFUNC_FAIL,
                    line!() as i32,
                    "CVode",
                    file!(),
                    &MSGCV_RTFUNC_FAILED(tn),
                );
                return CV_RTFUNC_FAIL;
            }
        }

        /* end of first call block */
    } else if cv_mem.borrow().first_step_after_resize {
        /* Check if the resized y satisfies the constraints */
        let constraints = cv_mem.borrow().cv_constraints.clone();
        if let Some(constraints) = constraints {
            let (zn0, tempv) = {
                let m = cv_mem.borrow();
                (m.cv_zn[0].clone().unwrap(), m.cv_tempv.clone().unwrap())
            };
            let conOK = N_VConstrMask(&constraints, &zn0, &tempv);
            if !conOK {
                cvProcessError(
                    Some(cv_mem),
                    CV_ILL_INPUT,
                    line!() as i32,
                    "CVode",
                    file!(),
                    "y does not satisfy the constraints",
                );
                return CV_ILL_INPUT;
            }
        }

        /* Initialize the linear solver */
        let linit = cv_mem.borrow().cv_linit;
        if let Some(linit) = linit {
            let ier = linit(cv_mem);
            if ier != 0 {
                cvProcessError(
                    Some(cv_mem),
                    CV_LINIT_FAIL,
                    line!() as i32,
                    "CVode",
                    file!(),
                    MSGCV_LINIT_FAIL,
                );
                return CV_LINIT_FAIL;
            }
        }

        /* Initialize the nonlinear solver (must occur after linear solver is
        initialized) so the lsetup and lsolve pointers have been set */
        let ier = crate::cvode_nls::cvNlsInit(cv_mem);
        if ier != 0 {
            cvProcessError(
                Some(cv_mem),
                CV_NLS_INIT_FAIL,
                line!() as i32,
                "CVode",
                file!(),
                MSGCV_NLS_INIT_FAIL,
            );
            return CV_NLS_INIT_FAIL;
        }
    }

    /*
     * ------------------------------------------------------
     * 3. At following steps, perform stop tests:
     *    - check for root in last step
     *    - check if we passed tstop
     *    - check if we passed tout (NORMAL mode)
     *    - check if current tn was returned (ONE_STEP mode)
     *    - check if we are close to tstop
     *      (adjust step size if needed)
     * -------------------------------------------------------
     */

    if cv_mem.borrow().cv_nst > 0 {
        /* Estimate an infinitesimal time interval to be used as
        a roundoff for time quantities (based on current time
        and step size) */
        let troundoff = {
            let m = cv_mem.borrow();
            FUZZ_FACTOR * m.cv_uround * (SUNRabs(m.cv_tn) + SUNRabs(m.cv_h))
        };

        /* First, check for a root in the last step taken, other than the
        last root found, if any.  If itask = CV_ONE_STEP and y(tn) was not
        returned because of an intervening root, return y(tn) now.     */
        if cv_mem.borrow().cv_nrtfn > 0 {
            let irfndp = cv_mem.borrow().cv_irfnd;

            let retval = cvRcheck2(cv_mem);

            if retval == CLOSERT {
                let tlo = cv_mem.borrow().cv_tlo;
                cvProcessError(
                    Some(cv_mem),
                    CV_ILL_INPUT,
                    line!() as i32,
                    "CVode",
                    file!(),
                    &MSGCV_CLOSE_ROOTS(tlo),
                );
                return CV_ILL_INPUT;
            } else if retval == CV_RTFUNC_FAIL {
                let tlo = cv_mem.borrow().cv_tlo;
                cvProcessError(
                    Some(cv_mem),
                    CV_RTFUNC_FAIL,
                    line!() as i32,
                    "CVode",
                    file!(),
                    &MSGCV_RTFUNC_FAILED(tlo),
                );
                return CV_RTFUNC_FAIL;
            } else if retval == RTFOUND {
                let mut m = cv_mem.borrow_mut();
                m.cv_tretlast = m.cv_tlo;
                *tret = m.cv_tlo;
                return CV_ROOT_RETURN;
            }

            /* If tn is distinct from tretlast (within roundoff),
            check remaining interval for roots */
            let distinct = {
                let m = cv_mem.borrow();
                SUNRabs(m.cv_tn - m.cv_tretlast) > troundoff
            };
            if distinct {
                let retval = cvRcheck3(cv_mem, tout, itask);

                if retval == CV_SUCCESS {
                    /* no root found */
                    cv_mem.borrow_mut().cv_irfnd = 0;
                    if (irfndp == 1) && (itask == CV_ONE_STEP) {
                        let (tn, zn0) = {
                            let mut m = cv_mem.borrow_mut();
                            m.cv_tretlast = m.cv_tn;
                            (m.cv_tn, m.cv_zn[0].clone().unwrap())
                        };
                        *tret = tn;
                        N_VScale(ONE, &zn0, yout);
                        return CV_SUCCESS;
                    }
                } else if retval == RTFOUND {
                    /* a new root was found */
                    let mut m = cv_mem.borrow_mut();
                    m.cv_irfnd = 1;
                    m.cv_tretlast = m.cv_tlo;
                    *tret = m.cv_tlo;
                    return CV_ROOT_RETURN;
                } else if retval == CV_RTFUNC_FAIL {
                    /* g failed */
                    let tlo = cv_mem.borrow().cv_tlo;
                    cvProcessError(
                        Some(cv_mem),
                        CV_RTFUNC_FAIL,
                        line!() as i32,
                        "CVode",
                        file!(),
                        &MSGCV_RTFUNC_FAILED(tlo),
                    );
                    return CV_RTFUNC_FAIL;
                }
            }
        } /* end of root stop check */

        /* Test for tn at tstop or near tstop */
        if cv_mem.borrow().cv_tstopset {
            let (tn, tstop, h) = {
                let m = cv_mem.borrow();
                (m.cv_tn, m.cv_tstop, m.cv_h)
            };
            /* Test for tn at tstop */
            if SUNRabs(tn - tstop) <= troundoff {
                /* Ensure tout >= tstop, otherwise check for tout return below */
                if (tout - tstop) * h >= ZERO || SUNRabs(tout - tstop) <= troundoff {
                    if cv_mem.borrow().cv_tstopinterp {
                        let ier = CVodeGetDky(cv_mem, tstop, 0, yout);
                        if ier != CV_SUCCESS {
                            cvProcessError(
                                Some(cv_mem),
                                CV_ILL_INPUT,
                                line!() as i32,
                                "CVode",
                                file!(),
                                &MSGCV_BAD_TSTOP(tstop, tn),
                            );
                            return CV_ILL_INPUT;
                        }
                    } else {
                        let zn0 = cv_mem.borrow().cv_zn[0].clone().unwrap();
                        N_VScale(ONE, &zn0, yout);
                    }
                    let mut m = cv_mem.borrow_mut();
                    m.cv_tretlast = m.cv_tstop;
                    *tret = m.cv_tstop;
                    m.cv_tstopset = SUNFALSE;
                    return CV_TSTOP_RETURN;
                }
            }
            /* If next step would overtake tstop, adjust stepsize */
            else if (tn + cv_mem.borrow().cv_hprime - tstop) * h > ZERO {
                let mut m = cv_mem.borrow_mut();
                m.cv_hprime = (m.cv_tstop - m.cv_tn) * (ONE - FOUR * m.cv_uround);
                m.cv_eta = m.cv_hprime / m.cv_h;
            }
        }

        /* In CV_NORMAL mode, test if tout was reached */
        if (itask == CV_NORMAL) && {
            let m = cv_mem.borrow();
            (m.cv_tn - tout) * m.cv_h >= ZERO
        } {
            {
                let mut m = cv_mem.borrow_mut();
                m.cv_tretlast = tout;
            }
            *tret = tout;
            let ier = CVodeGetDky(cv_mem, tout, 0, yout);
            if ier != CV_SUCCESS {
                cvProcessError(
                    Some(cv_mem),
                    CV_ILL_INPUT,
                    line!() as i32,
                    "CVode",
                    file!(),
                    &MSGCV_BAD_TOUT(tout),
                );
                return CV_ILL_INPUT;
            }
            return CV_SUCCESS;
        }

        /* In CV_ONE_STEP mode, test if tn was returned */
        if itask == CV_ONE_STEP && {
            let m = cv_mem.borrow();
            SUNRabs(m.cv_tn - m.cv_tretlast) > troundoff
        } {
            let (tn, zn0) = {
                let mut m = cv_mem.borrow_mut();
                m.cv_tretlast = m.cv_tn;
                (m.cv_tn, m.cv_zn[0].clone().unwrap())
            };
            *tret = tn;
            N_VScale(ONE, &zn0, yout);
            return CV_SUCCESS;
        }
    } /* end stopping tests block */

    /*
     * --------------------------------------------------
     * 4. Looping point for internal steps
     *
     *    4.1. check for errors (too many steps, too much
     *         accuracy requested, step size too small)
     *    4.2. take a new step (call cvStep)
     *    4.3. stop on error
     *    4.4. perform stop tests:
     *         - check for root in last step
     *         - check if tout was passed
     *         - check if close to tstop
     *         - check if in ONE_STEP mode (must return)
     * --------------------------------------------------
     */

    let mut nstloc: i64 = 0;
    let istate: i32;
    loop {
        {
            let mut m = cv_mem.borrow_mut();
            m.cv_next_h = m.cv_h;
            m.cv_next_q = m.cv_q;
        }

        /* Reset and check ewt */
        if cv_mem.borrow().cv_nst > 0 {
            let (zn0, ewt) = {
                let m = cv_mem.borrow();
                (m.cv_zn[0].clone().unwrap(), m.cv_ewt.clone().unwrap())
            };
            let ewtsetOK = cv_call_efun(cv_mem, &zn0, &ewt);

            if ewtsetOK != 0 {
                let (itol, tn) = {
                    let m = cv_mem.borrow();
                    (m.cv_itol, m.cv_tn)
                };
                if itol == CV_WF {
                    cvProcessError(
                        Some(cv_mem),
                        CV_ILL_INPUT,
                        line!() as i32,
                        "CVode",
                        file!(),
                        &MSGCV_EWT_NOW_FAIL(tn),
                    );
                } else {
                    cvProcessError(
                        Some(cv_mem),
                        CV_ILL_INPUT,
                        line!() as i32,
                        "CVode",
                        file!(),
                        &MSGCV_EWT_NOW_BAD(tn),
                    );
                }

                istate = CV_ILL_INPUT;
                cv_mem.borrow_mut().cv_tretlast = tn;
                *tret = tn;
                N_VScale(ONE, &zn0, yout);
                break;
            }
        }

        /* Check for too many steps */
        {
            let (mxstep, tn, zn0) = {
                let m = cv_mem.borrow();
                (m.cv_mxstep, m.cv_tn, m.cv_zn[0].clone().unwrap())
            };
            if (mxstep > 0) && (nstloc >= mxstep) {
                cvProcessError(
                    Some(cv_mem),
                    CV_TOO_MUCH_WORK,
                    line!() as i32,
                    "CVode",
                    file!(),
                    &MSGCV_MAX_STEPS(tn),
                );
                istate = CV_TOO_MUCH_WORK;
                cv_mem.borrow_mut().cv_tretlast = tn;
                *tret = tn;
                N_VScale(ONE, &zn0, yout);
                break;
            }
        }

        /* Check for too much accuracy requested */
        {
            let (zn0, ewt, uround) = {
                let m = cv_mem.borrow();
                (
                    m.cv_zn[0].clone().unwrap(),
                    m.cv_ewt.clone().unwrap(),
                    m.cv_uround,
                )
            };
            let nrm = N_VWrmsNorm(&zn0, &ewt);
            cv_mem.borrow_mut().cv_tolsf = uround * nrm;
            if cv_mem.borrow().cv_tolsf > ONE {
                let tn = cv_mem.borrow().cv_tn;
                cvProcessError(
                    Some(cv_mem),
                    CV_TOO_MUCH_ACC,
                    line!() as i32,
                    "CVode",
                    file!(),
                    &MSGCV_TOO_MUCH_ACC(tn),
                );
                istate = CV_TOO_MUCH_ACC;
                cv_mem.borrow_mut().cv_tretlast = tn;
                *tret = tn;
                N_VScale(ONE, &zn0, yout);
                cv_mem.borrow_mut().cv_tolsf *= TWO;
                break;
            } else {
                cv_mem.borrow_mut().cv_tolsf = ONE;
            }
        }

        /* Check for h below roundoff level in tn */
        {
            let (tn, h) = {
                let m = cv_mem.borrow();
                (m.cv_tn, m.cv_h)
            };
            if tn + h == tn {
                cv_mem.borrow_mut().cv_nhnil += 1;
                let (nhnil, mxhnil) = {
                    let m = cv_mem.borrow();
                    (m.cv_nhnil, m.cv_mxhnil)
                };
                if nhnil <= mxhnil {
                    cvProcessError(
                        Some(cv_mem),
                        CV_WARNING,
                        line!() as i32,
                        "CVode",
                        file!(),
                        &MSGCV_HNIL(tn, h),
                    );
                }
                if nhnil == mxhnil {
                    cvProcessError(
                        Some(cv_mem),
                        CV_WARNING,
                        line!() as i32,
                        "CVode",
                        file!(),
                        MSGCV_HNIL_DONE,
                    );
                }
            }
        }

        /* Call cvStep to take a step */
        let kflag = cvStep(cv_mem);

        /* Process failed step cases, and exit loop */
        if kflag != CV_SUCCESS {
            istate = cvHandleFailure(cv_mem, kflag);
            let (tn, zn0) = {
                let mut m = cv_mem.borrow_mut();
                m.cv_tretlast = m.cv_tn;
                (m.cv_tn, m.cv_zn[0].clone().unwrap())
            };
            *tret = tn;
            N_VScale(ONE, &zn0, yout);
            break;
        }

        nstloc += 1;

        /* If tstop is set and was reached, reset tn = tstop */
        if cv_mem.borrow().cv_tstopset {
            let mut m = cv_mem.borrow_mut();
            let troundoff = FUZZ_FACTOR * m.cv_uround * (SUNRabs(m.cv_tn) + SUNRabs(m.cv_h));
            if SUNRabs(m.cv_tn - m.cv_tstop) <= troundoff {
                m.cv_tn = m.cv_tstop;
            }
        }

        /* Check for root in last step taken. */
        if cv_mem.borrow().cv_nrtfn > 0 {
            let retval = cvRcheck3(cv_mem, tout, itask);

            if retval == RTFOUND {
                /* A new root was found */
                let mut m = cv_mem.borrow_mut();
                m.cv_irfnd = 1;
                istate = CV_ROOT_RETURN;
                m.cv_tretlast = m.cv_tlo;
                *tret = m.cv_tlo;
                break;
            } else if retval == CV_RTFUNC_FAIL {
                /* g failed */
                let tlo = cv_mem.borrow().cv_tlo;
                cvProcessError(
                    Some(cv_mem),
                    CV_RTFUNC_FAIL,
                    line!() as i32,
                    "CVode",
                    file!(),
                    &MSGCV_RTFUNC_FAILED(tlo),
                );
                istate = CV_RTFUNC_FAIL;
                break;
            }

            /* If we are at the end of the first step and we still have
             * some event functions that are inactive, issue a warning
             * as this may indicate a user error in the implementation
             * of the root function. */

            if cv_mem.borrow().cv_nst == 1 {
                let (inactive_roots, mxgnull) = {
                    let m = cv_mem.borrow();
                    let mut inactive_roots = SUNFALSE;
                    for ir in 0..m.cv_nrtfn as usize {
                        if !m.cv_gactive[ir] {
                            inactive_roots = SUNTRUE;
                            break;
                        }
                    }
                    (inactive_roots, m.cv_mxgnull)
                };
                if (mxgnull > 0) && inactive_roots {
                    cvProcessError(
                        Some(cv_mem),
                        CV_WARNING,
                        line!() as i32,
                        "CVode",
                        file!(),
                        MSGCV_INACTIVE_ROOTS,
                    );
                }
            }
        }

        /* Check if tn is at tstop or near tstop */
        if cv_mem.borrow().cv_tstopset {
            let (tn, tstop, h, troundoff) = {
                let m = cv_mem.borrow();
                (
                    m.cv_tn,
                    m.cv_tstop,
                    m.cv_h,
                    FUZZ_FACTOR * m.cv_uround * (SUNRabs(m.cv_tn) + SUNRabs(m.cv_h)),
                )
            };

            /* Test for tn at tstop */
            if SUNRabs(tn - tstop) <= troundoff {
                /* Ensure tout >= tstop, otherwise check for tout return below */
                if (tout - tstop) * h >= ZERO || SUNRabs(tout - tstop) <= troundoff {
                    if cv_mem.borrow().cv_tstopinterp {
                        let _ = CVodeGetDky(cv_mem, tstop, 0, yout);
                    } else {
                        let zn0 = cv_mem.borrow().cv_zn[0].clone().unwrap();
                        N_VScale(ONE, &zn0, yout);
                    }
                    let mut m = cv_mem.borrow_mut();
                    m.cv_tretlast = m.cv_tstop;
                    *tret = m.cv_tstop;
                    m.cv_tstopset = SUNFALSE;
                    istate = CV_TSTOP_RETURN;
                    break;
                }
            }
            /* If next step would overtake tstop, adjust stepsize */
            else if (tn + cv_mem.borrow().cv_hprime - tstop) * h > ZERO {
                let mut m = cv_mem.borrow_mut();
                m.cv_hprime = (m.cv_tstop - m.cv_tn) * (ONE - FOUR * m.cv_uround);
                m.cv_eta = m.cv_hprime / m.cv_h;
            }
        }

        /* In NORMAL mode, check if tout reached */
        if (itask == CV_NORMAL) && {
            let m = cv_mem.borrow();
            (m.cv_tn - tout) * m.cv_h >= ZERO
        } {
            istate = CV_SUCCESS;
            cv_mem.borrow_mut().cv_tretlast = tout;
            *tret = tout;
            let _ = CVodeGetDky(cv_mem, tout, 0, yout);
            let mut m = cv_mem.borrow_mut();
            m.cv_next_q = m.cv_qprime;
            m.cv_next_h = m.cv_hprime;
            break;
        }

        /* In ONE_STEP mode, copy y and exit loop */
        if itask == CV_ONE_STEP {
            istate = CV_SUCCESS;
            let (tn, zn0) = {
                let mut m = cv_mem.borrow_mut();
                m.cv_tretlast = m.cv_tn;
                (m.cv_tn, m.cv_zn[0].clone().unwrap())
            };
            *tret = tn;
            N_VScale(ONE, &zn0, yout);
            let mut m = cv_mem.borrow_mut();
            m.cv_next_q = m.cv_qprime;
            m.cv_next_h = m.cv_hprime;
            break;
        }
    } /* end looping for internal steps */

    istate
}

/*
 * -----------------------------------------------------------------
 * Interpolated output and extraction functions
 * -----------------------------------------------------------------
 */

/*
 * CVodeGetDky
 *
 * This routine computes the k-th derivative of the interpolating
 * polynomial at the time t and stores the result in the vector dky.
 * The formula is:
 *         q
 *  dky = SUM c(j,k) * (t - tn)^(j-k) * h^(-j) * zn[j] ,
 *        j=k
 * where c(j,k) = j*(j-1)*...*(j-k+1), q is the current order, and
 * zn[j] is the j-th column of the Nordsieck history array.
 *
 * This function is called by CVode with k = 0 and t = tout, but
 * may also be called directly by the user.
 */

pub fn CVodeGetDky(cvode_mem: &CVodeMem, t: sunrealtype, k: i32, dky: &N_Vector) -> i32 {
    /* Check all inputs for legality */

    /* NULL-mem check: handled by type system */
    let cv_mem = cvode_mem;

    /* NULL dky check: handled by type system */

    if (k < 0) || (k > cv_mem.borrow().cv_q) {
        cvProcessError(
            Some(cv_mem),
            CV_BAD_K,
            line!() as i32,
            "CVodeGetDky",
            file!(),
            MSGCV_BAD_K,
        );
        return CV_BAD_K;
    }

    /* Allow for some slack */
    let (uround, tn, hu, h, q) = {
        let m = cv_mem.borrow();
        (m.cv_uround, m.cv_tn, m.cv_hu, m.cv_h, m.cv_q)
    };
    let mut tfuzz = FUZZ_FACTOR * uround * (SUNRabs(tn) + SUNRabs(hu));
    if hu < ZERO {
        tfuzz = -tfuzz;
    }
    let tp = tn - hu - tfuzz;
    let tn1 = tn + tfuzz;
    if (t - tp) * (t - tn1) > ZERO {
        cvProcessError(
            Some(cv_mem),
            CV_BAD_T,
            line!() as i32,
            "CVodeGetDky",
            file!(),
            &MSGCV_BAD_T(t, tn - hu, tn),
        );
        return CV_BAD_T;
    }

    /* Sum the differentiated interpolating polynomial */
    let mut nvec: usize = 0;

    let s = (t - tn) / h;
    let mut cvals = [ZERO; L_MAX];
    let mut Xvecs: Vec<N_Vector> = Vec::new();
    {
        let m = cv_mem.borrow();
        let mut j = q;
        while j >= k {
            cvals[nvec] = ONE;
            let mut i = j;
            while i >= j - k + 1 {
                cvals[nvec] *= i as sunrealtype;
                i -= 1;
            }
            for _ in 0..(j - k) {
                cvals[nvec] *= s;
            }
            Xvecs.push(m.cv_zn[j as usize].clone().unwrap());
            nvec += 1;
            j -= 1;
        }
    }
    let ier = N_VLinearCombination(nvec as i32, &cvals, &Xvecs, dky);
    if ier != CV_SUCCESS {
        return CV_VECTOROP_ERR;
    }

    if k == 0 {
        return CV_SUCCESS;
    }
    let r = SUNRpowerI(h, -k);
    N_VScale(r, dky, dky);

    CV_SUCCESS
}

/*
 * CVodeComputeState
 *
 * Computes y based on the current prediction and given correction.
 */

pub fn CVodeComputeState(cvode_mem: &CVodeMem, ycor: &N_Vector, y: &N_Vector) -> i32 {
    /* NULL-mem check: handled by type system */
    let cv_mem = cvode_mem;

    let zn0 = cv_mem.borrow().cv_zn[0].clone().unwrap();
    N_VLinearSum(ONE, &zn0, ONE, ycor, y);

    CV_SUCCESS
}

/*
 * CVodeFree
 *
 * This routine frees the problem memory allocated by CVodeInit.
 * Such memory includes all the vectors allocated by cvAllocVectors,
 * and the memory lmem for the linear solver (deallocated by a call
 * to lfree).
 */

pub fn CVodeFree(cvode_mem: &mut Option<CVodeMem>) {
    if cvode_mem.is_none() {
        return;
    }

    let cv_mem = cvode_mem.as_ref().unwrap().clone();

    cvFreeVectors(&cv_mem);

    /* if CVODE created the nonlinear solver object then free it */
    if cv_mem.borrow().ownNLS {
        let nls = {
            let mut m = cv_mem.borrow_mut();
            m.ownNLS = SUNFALSE;
            m.NLS.take()
        };
        let _ = SUNNonlinSolFree(nls);
    }

    let lfree = cv_mem.borrow().cv_lfree;
    if let Some(lfree) = lfree {
        let _ = lfree(&cv_mem);
    }

    if cv_mem.borrow().cv_nrtfn > 0 {
        let mut m = cv_mem.borrow_mut();
        m.cv_glo = Vec::new();
        m.cv_ghi = Vec::new();
        m.cv_grout = Vec::new();
        m.cv_iroots = Vec::new();
        m.cv_rootdir = Vec::new();
        m.cv_gactive = Vec::new();
    }

    if cv_mem.borrow().proj_mem.is_some() {
        /* cvProjFree: dropping the projection memory frees it */
        cv_mem.borrow_mut().proj_mem = None;
    }

    /* C frees the mem struct wholesale; the Rust handle is dropped by the
    caller, so break the Rc cycle the default-efun e_data token creates
    (cv_e_data holds a CVodeMem clone pointing back at this record) */
    cv_mem.borrow_mut().cv_e_data = None;

    *cvode_mem = None;
}

/*
 * =================================================================
 *  Private Functions Implementation
 * =================================================================
 */

/*
 * cvCheckNvector
 * This routine checks if all required vector operations are present.
 * If any of them is missing it returns SUNFALSE.
 */

fn cvCheckNvector(tmpl: &N_Vector) -> sunbooleantype {
    let ops = tmpl.ops.borrow();
    if ops.nvclone.is_none()
        || ops.nvdestroy.is_none()
        || ops.nvlinearsum.is_none()
        || ops.nvconst.is_none()
        || ops.nvprod.is_none()
        || ops.nvdiv.is_none()
        || ops.nvscale.is_none()
        || ops.nvabs.is_none()
        || ops.nvinv.is_none()
        || ops.nvaddconst.is_none()
        || ops.nvmaxnorm.is_none()
        || ops.nvwrmsnorm.is_none()
    {
        SUNFALSE
    } else {
        SUNTRUE
    }
}

/*
 * -----------------------------------------------------------------
 * Memory allocation/deallocation
 * -----------------------------------------------------------------
 */

/*
 * cvAllocVectors
 *
 * This routine allocates the CVODE vectors ewt, acor, tempv, ftemp, and
 * zn[0], ..., zn[maxord].
 * If all memory allocations are successful, cvAllocVectors returns SUNTRUE.
 * Otherwise all allocated memory is freed and cvAllocVectors returns SUNFALSE.
 * This routine also sets the optional outputs lrw and liw, which are
 * (respectively) the lengths of the real and integer work spaces
 * allocated here.
 */

fn cvAllocVectors(cv_mem: &CVodeMem, tmpl: &N_Vector) -> sunbooleantype {
    /* Allocate ewt, acor, tempv, ftemp */

    let ewt = match N_VClone(tmpl) {
        Some(v) => v,
        None => return SUNFALSE,
    };

    let acor = match N_VClone(tmpl) {
        Some(v) => v,
        None => {
            N_VDestroy(ewt);
            return SUNFALSE;
        }
    };

    let tempv = match N_VClone(tmpl) {
        Some(v) => v,
        None => {
            N_VDestroy(ewt);
            N_VDestroy(acor);
            return SUNFALSE;
        }
    };

    let ftemp = match N_VClone(tmpl) {
        Some(v) => v,
        None => {
            N_VDestroy(tempv);
            N_VDestroy(ewt);
            N_VDestroy(acor);
            return SUNFALSE;
        }
    };

    let vtemp1 = match N_VClone(tmpl) {
        Some(v) => v,
        None => {
            N_VDestroy(ftemp);
            N_VDestroy(tempv);
            N_VDestroy(ewt);
            N_VDestroy(acor);
            return SUNFALSE;
        }
    };

    let vtemp2 = match N_VClone(tmpl) {
        Some(v) => v,
        None => {
            N_VDestroy(vtemp1);
            N_VDestroy(ftemp);
            N_VDestroy(tempv);
            N_VDestroy(ewt);
            N_VDestroy(acor);
            return SUNFALSE;
        }
    };

    let vtemp3 = match N_VClone(tmpl) {
        Some(v) => v,
        None => {
            N_VDestroy(vtemp2);
            N_VDestroy(vtemp1);
            N_VDestroy(ftemp);
            N_VDestroy(tempv);
            N_VDestroy(ewt);
            N_VDestroy(acor);
            return SUNFALSE;
        }
    };

    {
        let mut m = cv_mem.borrow_mut();
        m.cv_ewt = Some(ewt);
        m.cv_acor = Some(acor);
        m.cv_tempv = Some(tempv);
        m.cv_ftemp = Some(ftemp);
        m.cv_vtemp1 = Some(vtemp1);
        m.cv_vtemp2 = Some(vtemp2);
        m.cv_vtemp3 = Some(vtemp3);
    }

    /* Allocate zn[0] ... zn[qmax] */

    let qmax = cv_mem.borrow().cv_qmax;
    for j in 0..=qmax as usize {
        match N_VClone(tmpl) {
            Some(v) => {
                cv_mem.borrow_mut().cv_zn[j] = Some(v);
            }
            None => {
                let mut m = cv_mem.borrow_mut();
                if let Some(v) = m.cv_ewt.take() {
                    N_VDestroy(v);
                }
                if let Some(v) = m.cv_acor.take() {
                    N_VDestroy(v);
                }
                if let Some(v) = m.cv_tempv.take() {
                    N_VDestroy(v);
                }
                if let Some(v) = m.cv_ftemp.take() {
                    N_VDestroy(v);
                }
                if let Some(v) = m.cv_vtemp1.take() {
                    N_VDestroy(v);
                }
                if let Some(v) = m.cv_vtemp2.take() {
                    N_VDestroy(v);
                }
                if let Some(v) = m.cv_vtemp3.take() {
                    N_VDestroy(v);
                }
                for i in 0..j {
                    if let Some(v) = m.cv_zn[i].take() {
                        N_VDestroy(v);
                    }
                }
                return SUNFALSE;
            }
        }
    }

    {
        let mut m = cv_mem.borrow_mut();

        /* Update solver workspace lengths  */
        m.cv_lrw += (m.cv_qmax as i64 + 8) * m.cv_lrw1;
        m.cv_liw += (m.cv_qmax as i64 + 8) * m.cv_liw1;

        /* Store the value of qmax used here */
        m.cv_qmax_alloc = m.cv_qmax;
    }

    SUNTRUE
}

/*
 * cvFreeVectors
 *
 * This routine frees the vectors allocated in cvAllocVectors.
 */

fn cvFreeVectors(cv_mem: &CVodeMem) {
    let mut m = cv_mem.borrow_mut();

    let maxord = m.cv_qmax_alloc;

    if let Some(v) = m.cv_ewt.take() {
        N_VDestroy(v);
    }
    if let Some(v) = m.cv_acor.take() {
        N_VDestroy(v);
    }
    if let Some(v) = m.cv_tempv.take() {
        N_VDestroy(v);
    }
    if let Some(v) = m.cv_ftemp.take() {
        N_VDestroy(v);
    }
    if let Some(v) = m.cv_vtemp1.take() {
        N_VDestroy(v);
    }
    if let Some(v) = m.cv_vtemp2.take() {
        N_VDestroy(v);
    }
    if let Some(v) = m.cv_vtemp3.take() {
        N_VDestroy(v);
    }
    for j in 0..=maxord as usize {
        if let Some(v) = m.cv_zn[j].take() {
            N_VDestroy(v);
        }
    }

    m.cv_lrw -= (maxord as i64 + 8) * m.cv_lrw1;
    m.cv_liw -= (maxord as i64 + 8) * m.cv_liw1;

    if m.cv_VabstolMallocDone {
        if let Some(v) = m.cv_Vabstol.take() {
            N_VDestroy(v);
        }
        m.cv_lrw -= m.cv_lrw1;
        m.cv_liw -= m.cv_liw1;
    }

    if m.cv_constraints.is_some() {
        if let Some(v) = m.cv_constraints.take() {
            N_VDestroy(v);
        }
        m.cv_lrw -= m.cv_lrw1;
        m.cv_liw -= m.cv_liw1;
    }
}

/*
 * -----------------------------------------------------------------
 * Initial setup
 * -----------------------------------------------------------------
 */

/*
 * cvInitialSetup
 *
 * This routine performs input consistency checks at the first step.
 * If needed, it also checks the linear solver module and calls the
 * linear solver initialization routine.
 */

fn cvInitialSetup(cv_mem: &CVodeMem, tout: sunrealtype) -> i32 {
    /* Is tout too close to tn? */
    let (tn, uround) = {
        let m = cv_mem.borrow();
        (m.cv_tn, m.cv_uround)
    };
    let tdist = SUNRabs(tout - tn);
    let tround = uround * SUNMAX(SUNRabs(tn), SUNRabs(tout));

    if tdist == ZERO || tdist < TWO * tround {
        cvProcessError(
            Some(cv_mem),
            CV_TOO_CLOSE,
            line!() as i32,
            "cvInitialSetup",
            file!(),
            MSGCV_TOO_CLOSE,
        );
        return CV_TOO_CLOSE;
    }

    /* Did the user specify tolerances? */
    if cv_mem.borrow().cv_itol == CV_NN {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "cvInitialSetup",
            file!(),
            MSGCV_NO_TOL,
        );
        return CV_ILL_INPUT;
    }

    /* If using a built-in routine for error weights with abstol==0,
    ensure that N_VMin is available */
    let (user_efun, atolmin0) = {
        let m = cv_mem.borrow();
        (m.cv_user_efun, m.cv_atolmin0)
    };
    if !user_efun && atolmin0 {
        let tempv = cv_mem.borrow().cv_tempv.clone().unwrap();
        if tempv.ops.borrow().nvmin.is_none() {
            cvProcessError(
                Some(cv_mem),
                CV_ILL_INPUT,
                line!() as i32,
                "cvInitialSetup",
                file!(),
                "Missing N_VMin routine from N_Vector",
            );
            return CV_ILL_INPUT;
        }
    }

    /* Set data for efun */
    if user_efun {
        /* C: cv_e_data = cv_user_data (pointer alias); efun call sites pass
        cv_user_data directly instead (box aliasing impossible) */
        cv_mem.borrow_mut().cv_e_data = None;
    } else {
        let token: Box<dyn Any> = Box::new(cv_mem.clone());
        cv_mem.borrow_mut().cv_e_data = Some(token);
    }

    /* Check to see if y0 satisfies constraints */
    let constraints = cv_mem.borrow().cv_constraints.clone();
    if let Some(constraints) = constraints {
        let (zn0, tempv) = {
            let m = cv_mem.borrow();
            (m.cv_zn[0].clone().unwrap(), m.cv_tempv.clone().unwrap())
        };
        let conOK = N_VConstrMask(&constraints, &zn0, &tempv);
        if !conOK {
            cvProcessError(
                Some(cv_mem),
                CV_ILL_INPUT,
                line!() as i32,
                "cvInitialSetup",
                file!(),
                MSGCV_Y0_FAIL_CONSTR,
            );
            return CV_ILL_INPUT;
        }
    }

    /* Load initial error weights */
    let (zn0, ewt) = {
        let m = cv_mem.borrow();
        (m.cv_zn[0].clone().unwrap(), m.cv_ewt.clone().unwrap())
    };
    let ier = cv_call_efun(cv_mem, &zn0, &ewt);
    if ier != 0 {
        if cv_mem.borrow().cv_itol == CV_WF {
            cvProcessError(
                Some(cv_mem),
                CV_ILL_INPUT,
                line!() as i32,
                "cvInitialSetup",
                file!(),
                MSGCV_EWT_FAIL,
            );
        } else {
            cvProcessError(
                Some(cv_mem),
                CV_ILL_INPUT,
                line!() as i32,
                "cvInitialSetup",
                file!(),
                MSGCV_BAD_EWT,
            );
        }
        return CV_ILL_INPUT;
    }

    /* Call linit function (if it exists) */
    let linit = cv_mem.borrow().cv_linit;
    if let Some(linit) = linit {
        let ier = linit(cv_mem);
        if ier != 0 {
            cvProcessError(
                Some(cv_mem),
                CV_LINIT_FAIL,
                line!() as i32,
                "cvInitialSetup",
                file!(),
                MSGCV_LINIT_FAIL,
            );
            return CV_LINIT_FAIL;
        }
    }

    /* Initialize the nonlinear solver (must occur after linear solver is
    initialized) so that lsetup and lsolve pointer have been set */
    let ier = crate::cvode_nls::cvNlsInit(cv_mem);
    if ier != 0 {
        cvProcessError(
            Some(cv_mem),
            CV_NLS_INIT_FAIL,
            line!() as i32,
            "cvInitialSetup",
            file!(),
            MSGCV_NLS_INIT_FAIL,
        );
        return CV_NLS_INIT_FAIL;
    }

    /* Initialize projection data */
    if cv_mem.borrow().proj_enabled && cv_mem.borrow().proj_mem.is_none() {
        cvProcessError(
            Some(cv_mem),
            CV_PROJ_MEM_NULL,
            line!() as i32,
            "cvInitialSetup",
            file!(),
            MSG_CV_PROJ_MEM_NULL,
        );
        return CV_PROJ_MEM_NULL;
    }

    if cv_mem.borrow().proj_mem.is_some() {
        let ier = {
            let mut m = cv_mem.borrow_mut();
            crate::cvode_proj::cvProjInit(m.proj_mem.as_mut().unwrap())
        };
        if ier != CV_SUCCESS {
            cvProcessError(
                Some(cv_mem),
                CV_MEM_FAIL,
                line!() as i32,
                "cvInitialSetup",
                file!(),
                MSGCV_MEM_FAIL,
            );
            return CV_MEM_FAIL;
        }
        cv_mem.borrow_mut().proj_applied = SUNFALSE;
    }

    /* Initial setup complete */
    CV_SUCCESS
}

/*
 * -----------------------------------------------------------------
 * Initial stepsize calculation
 * -----------------------------------------------------------------
 */

/*
 * cvHin
 *
 * This routine computes a tentative initial step size h0. Note that here tout
 * is either the value passed to CVode at the first call or the value of tstop
 * (if tstop is enabled and it is closer to t0=tn than tout). If the RHS
 * function fails unrecoverably, cvHin returns CV_RHSFUNC_FAIL. If the RHS
 * function fails recoverably too many times and recovery is not possible, cvHin
 * returns CV_REPTD_RHSFUNC_ERR. Otherwise, cvHin sets h to the chosen value h0
 * and returns CV_SUCCESS.
 *
 * The algorithm used seeks to find h0 as a solution of
 *       (WRMS norm of (h0^2 ydd / 2)) = 1,
 * where ydd = estimated second derivative of y.
 *
 * We start with an initial estimate equal to the geometric mean of the
 * lower and upper bounds on the step size.
 *
 * Loop up to MAX_ITERS times to find h0.
 * Stop if new and previous values differ by a factor < 2.
 * Stop if hnew/hg > 2 after one iteration, as this probably means
 * that the ydd value is bad because of cancellation error.
 *
 * For each new proposed hg, we allow MAX_ITERS attempts to
 * resolve a possible recoverable failure from f() by reducing
 * the proposed stepsize by a factor of 0.2. If a legal stepsize
 * still cannot be found, fall back on a previous value if possible,
 * or else return CV_REPTD_RHSFUNC_ERR.
 *
 * Finally, we apply a bias (0.5) and verify that h0 is within bounds.
 */

fn cvHin(cv_mem: &CVodeMem, tout: sunrealtype) -> i32 {
    /* cvInitialSetup checks for tdiff = 0 or < 2 * troundoff */
    let (tn, uround) = {
        let m = cv_mem.borrow();
        (m.cv_tn, m.cv_uround)
    };
    let tdiff = tout - tn;
    let sign: i32 = if tdiff > ZERO { 1 } else { -1 };
    let tdist = SUNRabs(tdiff);
    let tround = uround * SUNMAX(SUNRabs(tn), SUNRabs(tout));

    /*
    Set lower and upper bounds on h0, and take geometric mean
    as first trial value.
    Exit with this value if the bounds cross each other.
    */

    let hlb = HLB_FACTOR * tround;
    let hub = cvUpperBoundH0(cv_mem, tdist);

    let mut hg = SUNRsqrt(hlb * hub);

    if hub < hlb {
        if sign == -1 {
            cv_mem.borrow_mut().cv_h = -hg;
        } else {
            cv_mem.borrow_mut().cv_h = hg;
        }
        return CV_SUCCESS;
    }

    /* Outer loop */

    let mut hs = hg; /* safeguard against 'uninitialized variable' warning */
    let mut hnew = ZERO;
    let mut yddnrm = ZERO;

    for count1 in 1..=MAX_ITERS {
        /* Attempts to estimate ydd */

        let mut hgOK = SUNFALSE;

        for _count2 in 1..=MAX_ITERS {
            let hgs = hg * sign as sunrealtype;
            let retval = cvYddNorm(cv_mem, hgs, &mut yddnrm);
            /* If the RHS function failed unrecoverably, give up */
            if retval < 0 {
                return CV_RHSFUNC_FAIL;
            }
            /* If successful, we can use ydd */
            if retval == CV_SUCCESS {
                hgOK = SUNTRUE;
                break;
            }
            /* The RHS function failed recoverably; cut step size and test again */
            hg *= POINT2;
        }

        /* If the RHS function failed recoverably MAX_ITERS times */

        if !hgOK {
            /* Exit if this is the first or second pass. No recovery possible */
            if count1 <= 2 {
                return CV_REPTD_RHSFUNC_ERR;
            }
            /* We have a fall-back option. The value hs is a previous hnew which
            passed through f(). Use it and break */
            hnew = hs;
            break;
        }

        /* The proposed step size is feasible. Save it. */
        hs = hg;

        /* Propose new step size */
        hnew = if yddnrm * hub * hub > TWO {
            SUNRsqrt(TWO / yddnrm)
        } else {
            SUNRsqrt(hg * hub)
        };

        /* If last pass, stop now with hnew */
        if count1 == MAX_ITERS {
            break;
        }

        let hrat = hnew / hg;

        /* Accept hnew if it does not differ from hg by more than a factor of 2 */
        if (hrat > HALF) && (hrat < TWO) {
            break;
        }

        /* After one pass, if ydd seems to be bad, use fall-back value. */
        if (count1 > 1) && (hrat > TWO) {
            hnew = hg;
            break;
        }

        /* Send this value back through f() */
        hg = hnew;
    }

    /* Apply bounds, bias factor, and attach sign */

    let mut h0 = H_BIAS * hnew;
    if h0 < hlb {
        h0 = hlb;
    }
    if h0 > hub {
        h0 = hub;
    }
    if sign == -1 {
        h0 = -h0;
    }
    cv_mem.borrow_mut().cv_h = h0;

    CV_SUCCESS
}

/*
 * cvUpperBoundH0
 *
 * This routine sets an upper bound on abs(h0) based on
 * tdist = tn - t0 and the values of y[i]/y'[i].
 */

fn cvUpperBoundH0(cv_mem: &CVodeMem, tdist: sunrealtype) -> sunrealtype {
    /*
     * Bound based on |y0|/|y0'| -- allow at most an increase of
     * HUB_FACTOR in y0 (based on a forward Euler step). The weight
     * factor is used as a safeguard against zero components in y0.
     */

    let (temp1, temp2, zn0, zn1) = {
        let m = cv_mem.borrow();
        (
            m.cv_tempv.clone().unwrap(),
            m.cv_acor.clone().unwrap(),
            m.cv_zn[0].clone().unwrap(),
            m.cv_zn[1].clone().unwrap(),
        )
    };

    N_VAbs(&zn0, &temp2);
    let _ = cv_call_efun(cv_mem, &zn0, &temp1);
    N_VInv(&temp1, &temp1);
    N_VLinearSum(HUB_FACTOR, &temp2, ONE, &temp1, &temp1);

    N_VAbs(&zn1, &temp2);

    N_VDiv(&temp2, &temp1, &temp1);
    let hub_inv = N_VMaxNorm(&temp1);

    /*
     * bound based on tdist -- allow at most a step of magnitude
     * HUB_FACTOR * tdist
     */

    let mut hub = HUB_FACTOR * tdist;

    /* Use the smaller of the two */

    if hub * hub_inv > ONE {
        hub = ONE / hub_inv;
    }

    hub
}

/*
 * cvYddNorm
 *
 * This routine computes an estimate of the second derivative of y
 * using a difference quotient, and returns its WRMS norm.
 */

fn cvYddNorm(cv_mem: &CVodeMem, hg: sunrealtype, yddnrm: &mut sunrealtype) -> i32 {
    let (zn0, zn1, y, tempv, ewt, tn) = {
        let m = cv_mem.borrow();
        (
            m.cv_zn[0].clone().unwrap(),
            m.cv_zn[1].clone().unwrap(),
            m.cv_y.clone().unwrap(),
            m.cv_tempv.clone().unwrap(),
            m.cv_ewt.clone().unwrap(),
            m.cv_tn,
        )
    };

    N_VLinearSum(hg, &zn1, ONE, &zn0, &y);
    let retval = cv_call_f(cv_mem, tn + hg, &y, &tempv);
    cv_mem.borrow_mut().cv_nfe += 1;
    if retval < 0 {
        return CV_RHSFUNC_FAIL;
    }
    if retval > 0 {
        return RHSFUNC_RECVR;
    }

    N_VLinearSum(ONE / hg, &tempv, -ONE / hg, &zn1, &tempv);

    *yddnrm = N_VWrmsNorm(&tempv, &ewt);

    CV_SUCCESS
}

/*
 * -----------------------------------------------------------------
 * Main cvStep function
 * -----------------------------------------------------------------
 */

/*
 * cvStep
 *
 * This routine performs one internal cvode step, from tn to tn + h.
 * It calls other routines to do all the work.
 *
 * The main operations done here are as follows:
 * - preliminary adjustments if a new step size was chosen;
 * - prediction of the Nordsieck history array zn at tn + h;
 * - setting of multistep method coefficients and test quantities;
 * - solution of the nonlinear system;
 * - testing the local error;
 * - updating zn and other state data if successful;
 * - resetting stepsize and order for the next step.
 * - if SLDET is on, check for stability, reduce order if necessary.
 * On a failure in the nonlinear system solution or error test, the
 * step may be reattempted, depending on the nature of the failure.
 */

fn cvStep(cv_mem: &CVodeMem) -> i32 {
    let mut dsm: sunrealtype = ZERO; /* local truncation error estimate */

    /* Initialize failure counters for this step attempt */

    let mut ncf: i32 = 0; /* corrector failures  */
    let mut npf: i32 = 0; /* projection failures */
    let mut nef: i32 = 0; /* error test failures */
    let mut step_constraint_fails: i32 = 0;

    /* If the step size has changed, update the history array */
    {
        let (nst, hprime, h) = {
            let m = cv_mem.borrow();
            (m.cv_nst, m.cv_hprime, m.cv_h)
        };
        if (nst > 0) && (hprime != h) {
            cvAdjustParams(cv_mem);
        }
    }

    /* Check if this step should be projected */
    let mut doProjection = SUNFALSE;
    if cv_mem.borrow().proj_enabled {
        let m = cv_mem.borrow();
        let pm = m.proj_mem.as_ref().unwrap();
        doProjection = pm.freq > 0 && (m.cv_nst == 0 || (m.cv_nst >= pm.nstlprj + pm.freq));
    }

    /* Looping point for attempts to take a step */

    let saved_t = cv_mem.borrow().cv_tn; /* tn is updated in cvPredict */
    let mut nflag = FIRST_CALL;

    loop {
        cvPredict(cv_mem);
        cvSet(cv_mem);

        nflag = cvNls(cv_mem, nflag);
        let kflag = cvHandleNFlag(cv_mem, &mut nflag, saved_t, &mut ncf);

        /* Go back in loop if we need to predict again (nflag=PREV_CONV_FAIL) */
        if kflag == PREDICT_AGAIN {
            continue;
        }

        /* Return if nonlinear solve failed and recovery is not possible. */
        if kflag != DO_ERROR_TEST {
            return kflag;
        }

        /* Check inequality constraints */
        if cv_mem.borrow().cv_constraints.is_some() {
            let cflag = cvCheckConstraints(cv_mem, &mut nflag, saved_t, &mut step_constraint_fails);

            /* Go back in loop if we need to predict again (nflag=PREV_CONV_FAIL) */
            if cflag == PREDICT_AGAIN {
                continue;
            }

            /* Return if the check failed and recovery is not possible. */
            if cflag != CV_SUCCESS {
                return cflag;
            }
        }

        /* Check if a projection needs to be performed */
        cv_mem.borrow_mut().proj_applied = SUNFALSE;

        if doProjection {
            /* Perform projection (nflag=CV_SUCCESS) */
            let pflag = crate::cvode_proj::cvDoProjection(cv_mem, &mut nflag, saved_t, &mut npf);

            /* Go back in loop if we need to predict again (nflag=PREV_PROJ_FAIL) */
            if pflag == PREDICT_AGAIN {
                continue;
            }

            /* Return if projection failed and recovery is not possible */
            if pflag != CV_SUCCESS {
                return pflag;
            }
        }

        /* Perform error test (nflag=CV_SUCCESS) */
        let eflag = cvDoErrorTest(cv_mem, &mut nflag, saved_t, &mut nef, &mut dsm);

        /* Go back in loop if we need to predict again (nflag=PREV_ERR_FAIL) */
        if eflag == TRY_AGAIN {
            continue;
        }

        /* Return if error test failed and recovery is not possible. */
        if eflag != CV_SUCCESS {
            return eflag;
        }

        /* Error test passed (eflag=CV_SUCCESS), break from loop */
        break;
    }

    /* Nonlinear system solve and error test were both successful.
    Update data, and consider change of step and/or order.       */

    cvCompleteStep(cv_mem);

    cvPrepareNextStep(cv_mem, dsm);

    /* If Stablilty Limit Detection is turned on, call stability limit
    detection routine for possible order reduction. */

    if cv_mem.borrow().cv_sldeton {
        cvBDFStab(cv_mem);
    }

    {
        let mut m = cv_mem.borrow_mut();
        m.cv_etamax = if m.cv_nst <= m.cv_small_nst {
            m.cv_eta_max_es
        } else {
            m.cv_eta_max_gs
        };
    }

    /*  Finally, we rescale the acor array to be the
    estimated local error vector. */

    let (tq2, acor) = {
        let m = cv_mem.borrow();
        (m.cv_tq[2], m.cv_acor.clone().unwrap())
    };
    N_VScale(tq2, &acor, &acor);

    CV_SUCCESS
}

/*
 * -----------------------------------------------------------------
 * Function called at beginning of step
 * -----------------------------------------------------------------
 */

/*
 * cvAdjustParams
 *
 * This routine is called when a change in step size was decided upon,
 * and it handles the required adjustments to the history array zn.
 * If there is to be a change in order, we call cvAdjustOrder and reset
 * q, L = q+1, and qwait.  Then in any case, we call cvRescale, which
 * resets h and rescales the Nordsieck array.
 */

fn cvAdjustParams(cv_mem: &CVodeMem) {
    let (qprime, q) = {
        let m = cv_mem.borrow();
        (m.cv_qprime, m.cv_q)
    };
    if qprime != q {
        /* History adjustments for an order change were applied when resizing */
        if !cv_mem.borrow().first_step_after_resize {
            cvAdjustOrder(cv_mem, qprime - q);
        }
        let mut m = cv_mem.borrow_mut();
        m.cv_q = m.cv_qprime;
        m.cv_L = m.cv_q + 1;
        m.cv_qwait = m.cv_L;
    }
    cvRescale(cv_mem);
}

/*
 * cvAdjustOrder
 *
 * This routine is a high level routine which handles an order
 * change by an amount deltaq (= +1 or -1). If a decrease in order
 * is requested and q==2, then the routine returns immediately.
 * Otherwise cvAdjustAdams or cvAdjustBDF is called to handle the
 * order change (depending on the value of lmm).
 */

fn cvAdjustOrder(cv_mem: &CVodeMem, deltaq: i32) {
    let (q, lmm) = {
        let m = cv_mem.borrow();
        (m.cv_q, m.cv_lmm)
    };
    if (q == 2) && (deltaq != 1) {
        return;
    }

    match lmm {
        CV_ADAMS => cvAdjustAdams(cv_mem, deltaq),
        CV_BDF => cvAdjustBDF(cv_mem, deltaq),
        _ => {}
    }
}

/*
 * cvAdjustAdams
 *
 * This routine adjusts the history array on a change of order q by
 * deltaq, in the case that lmm == CV_ADAMS.
 */

fn cvAdjustAdams(cv_mem: &CVodeMem, deltaq: i32) {
    /* On an order increase, set new column of zn to zero and return */

    if deltaq == 1 {
        let znL = {
            let m = cv_mem.borrow();
            m.cv_zn[m.cv_L as usize].clone().unwrap()
        };
        N_VConst(ZERO, &znL);
        return;
    }

    /*
     * On an order decrease, each zn[j] is adjusted by a multiple of zn[q].
     * The coeffs. in the adjustment are the coeffs. of the polynomial:
     *        x
     * q * INT { u * ( u + xi_1 ) * ... * ( u + xi_{q-2} ) } du
     *        0
     * where xi_j = [t_n - t_(n-j)]/h => xi_0 = 0
     */

    {
        let mut m = cv_mem.borrow_mut();
        let qmax = m.cv_qmax;
        for i in 0..=qmax as usize {
            m.cv_l[i] = ZERO;
        }
        m.cv_l[1] = ONE;
        let mut hsum = ZERO;
        let q = m.cv_q;
        for j in 1..=(q - 2) {
            hsum += m.cv_tau[j as usize];
            let xi = hsum / m.cv_hscale;
            let mut i = j + 1;
            while i >= 1 {
                m.cv_l[i as usize] = m.cv_l[i as usize] * xi + m.cv_l[(i - 1) as usize];
                i -= 1;
            }
        }

        for j in 1..=(q - 2) {
            m.cv_l[(j + 1) as usize] = q as sunrealtype * (m.cv_l[j as usize] / (j + 1) as sunrealtype);
        }

        for j in 2..q {
            m.cv_cvals[(j - 2) as usize] = -m.cv_l[j as usize];
        }
    }

    let q = cv_mem.borrow().cv_q;
    if q > 2 {
        let (cvals, znq, znvec) = {
            let m = cv_mem.borrow();
            let znvec: Vec<N_Vector> = (2..m.cv_q as usize)
                .map(|j| m.cv_zn[j].clone().unwrap())
                .collect();
            (m.cv_cvals, m.cv_zn[m.cv_q as usize].clone().unwrap(), znvec)
        };
        let _ = N_VScaleAddMulti(q - 2, &cvals, &znq, &znvec, &znvec);
    }
}

/*
 * cvAdjustBDF
 *
 * This is a high level routine which handles adjustments to the
 * history array on a change of order by deltaq in the case that
 * lmm == CV_BDF.  cvAdjustBDF calls cvIncreaseBDF if deltaq = +1 and
 * cvDecreaseBDF if deltaq = -1 to do the actual work.
 */

fn cvAdjustBDF(cv_mem: &CVodeMem, deltaq: i32) {
    match deltaq {
        1 => cvIncreaseBDF(cv_mem),
        -1 => cvDecreaseBDF(cv_mem),
        _ => {}
    }
}

/*
 * cvIncreaseBDF
 *
 * This routine adjusts the history array on an increase in the
 * order q in the case that lmm == CV_BDF.
 * A new column zn[q+1] is set equal to a multiple of the saved
 * vector (= acor) in zn[indx_acor].  Then each zn[j] is adjusted by
 * a multiple of zn[q+1].  The coefficients in the adjustment are the
 * coefficients of the polynomial x*x*(x+xi_1)*...*(x+xi_j),
 * where xi_j = [t_n - t_(n-j)]/h.
 */

fn cvIncreaseBDF(cv_mem: &CVodeMem) {
    let A1;
    {
        let mut m = cv_mem.borrow_mut();
        let qmax = m.cv_qmax;
        for i in 0..=qmax as usize {
            m.cv_l[i] = ZERO;
        }
        m.cv_l[2] = ONE;
        let mut alpha1 = ONE;
        let mut prod = ONE;
        let mut xiold = ONE;
        let mut alpha0 = -ONE;
        let mut hsum = m.cv_hscale;
        let q = m.cv_q;
        if q > 1 {
            for j in 1..q {
                hsum += m.cv_tau[(j + 1) as usize];
                let xi = hsum / m.cv_hscale;
                prod *= xi;
                alpha0 -= ONE / (j + 1) as sunrealtype;
                alpha1 += ONE / xi;
                let mut i = j + 2;
                while i >= 2 {
                    m.cv_l[i as usize] = m.cv_l[i as usize] * xiold + m.cv_l[(i - 1) as usize];
                    i -= 1;
                }
                xiold = xi;
            }
        }
        A1 = (-alpha0 - alpha1) / prod;
    }
    let (zn_indx_acor, znL) = {
        let m = cv_mem.borrow();
        (
            m.cv_zn[m.cv_indx_acor as usize].clone().unwrap(),
            m.cv_zn[m.cv_L as usize].clone().unwrap(),
        )
    };
    N_VScale(A1, &zn_indx_acor, &znL);

    /* for (j=2; j <= cv_mem->cv_q; j++) */
    let q = cv_mem.borrow().cv_q;
    if q > 1 {
        let (lvals, znvec) = {
            let m = cv_mem.borrow();
            let mut lvals = [ZERO; L_MAX];
            for (idx, v) in m.cv_l[2..].iter().enumerate() {
                lvals[idx] = *v;
            }
            let znvec: Vec<N_Vector> = (2..=m.cv_q as usize)
                .map(|j| m.cv_zn[j].clone().unwrap())
                .collect();
            (lvals, znvec)
        };
        let _ = N_VScaleAddMulti(q - 1, &lvals, &znL, &znvec, &znvec);
    }
}

/*
 * cvDecreaseBDF
 *
 * This routine adjusts the history array on a decrease in the
 * order q in the case that lmm == CV_BDF.
 * Each zn[j] is adjusted by a multiple of zn[q].  The coefficients
 * in the adjustment are the coefficients of the polynomial
 *   x*x*(x+xi_1)*...*(x+xi_j), where xi_j = [t_n - t_(n-j)]/h.
 */

fn cvDecreaseBDF(cv_mem: &CVodeMem) {
    {
        let mut m = cv_mem.borrow_mut();
        let qmax = m.cv_qmax;
        for i in 0..=qmax as usize {
            m.cv_l[i] = ZERO;
        }
        m.cv_l[2] = ONE;
        let mut hsum = ZERO;
        let q = m.cv_q;
        for j in 1..=(q - 2) {
            hsum += m.cv_tau[j as usize];
            let xi = hsum / m.cv_hscale;
            let mut i = j + 2;
            while i >= 2 {
                m.cv_l[i as usize] = m.cv_l[i as usize] * xi + m.cv_l[(i - 1) as usize];
                i -= 1;
            }
        }

        for j in 2..q {
            m.cv_cvals[(j - 2) as usize] = -m.cv_l[j as usize];
        }
    }

    let q = cv_mem.borrow().cv_q;
    if q > 2 {
        let (cvals, znq, znvec) = {
            let m = cv_mem.borrow();
            let znvec: Vec<N_Vector> = (2..m.cv_q as usize)
                .map(|j| m.cv_zn[j].clone().unwrap())
                .collect();
            (m.cv_cvals, m.cv_zn[m.cv_q as usize].clone().unwrap(), znvec)
        };
        let _ = N_VScaleAddMulti(q - 2, &cvals, &znq, &znvec, &znvec);
    }
}

/*
 * cvRescale
 *
 * This routine rescales the Nordsieck array by multiplying the
 * jth column zn[j] by eta^j, j = 1, ..., q.  Then the value of
 * h is rescaled by eta, and hscale is reset to h.
 */

pub fn cvRescale(cv_mem: &CVodeMem) {
    /* compute scaling factors */
    let (q, cvals, znvec) = {
        let mut m = cv_mem.borrow_mut();
        m.cv_cvals[0] = m.cv_eta;
        let q = m.cv_q;
        for j in 1..=q as usize {
            m.cv_cvals[j] = m.cv_eta * m.cv_cvals[j - 1];
        }
        let znvec: Vec<N_Vector> = (1..=q as usize)
            .map(|j| m.cv_zn[j].clone().unwrap())
            .collect();
        (q, m.cv_cvals, znvec)
    };

    let _ = N_VScaleVectorArray(q, &cvals, &znvec, &znvec);

    let mut m = cv_mem.borrow_mut();
    m.cv_h = m.cv_hscale * m.cv_eta;
    m.cv_next_h = m.cv_h;
    m.cv_hscale = m.cv_h;
    m.cv_nscon = 0;
}

/*
 * cvPredict
 *
 * This routine advances tn by the tentative step size h, and computes
 * the predicted array z_n(0), which is overwritten on zn.  The
 * prediction of zn is done by repeated additions.
 * If tstop is enabled, it is possible for tn + h to be past tstop by roundoff,
 * and in that case, we reset tn (after incrementing by h) to tstop.
 */

fn cvPredict(cv_mem: &CVodeMem) {
    {
        let mut m = cv_mem.borrow_mut();
        m.cv_tn += m.cv_h;
        if m.cv_tstopset && (m.cv_tn - m.cv_tstop) * m.cv_h > ZERO {
            m.cv_tn = m.cv_tstop;
        }
    }

    let (q, zn) = {
        let m = cv_mem.borrow();
        let zn: Vec<N_Vector> = (0..=m.cv_q as usize)
            .map(|j| m.cv_zn[j].clone().unwrap())
            .collect();
        (m.cv_q, zn)
    };
    for k in 1..=q {
        let mut j = q;
        while j >= k {
            N_VLinearSum(
                ONE,
                &zn[(j - 1) as usize],
                ONE,
                &zn[j as usize],
                &zn[(j - 1) as usize],
            );
            j -= 1;
        }
    }
}

/*
 * cvSet
 *
 * This routine is a high level routine which calls cvSetAdams or
 * cvSetBDF to set the polynomial l, the test quantity array tq,
 * and the related variables  rl1, gamma, and gamrat.
 *
 * The array tq is loaded with constants used in the control of estimated
 * local errors and in the nonlinear convergence test.  Specifically, while
 * running at order q, the components of tq are as follows:
 *   tq[1] = a coefficient used to get the est. local error at order q-1
 *   tq[2] = a coefficient used to get the est. local error at order q
 *   tq[3] = a coefficient used to get the est. local error at order q+1
 *   tq[4] = constant used in nonlinear iteration convergence test
 *   tq[5] = coefficient used to get the order q+2 derivative vector used in
 *           the est. local error at order q+1
 */

fn cvSet(cv_mem: &CVodeMem) {
    let lmm = cv_mem.borrow().cv_lmm;
    match lmm {
        CV_ADAMS => cvSetAdams(cv_mem),
        CV_BDF => cvSetBDF(cv_mem),
        _ => {}
    }
    let mut m = cv_mem.borrow_mut();
    m.cv_rl1 = ONE / m.cv_l[1];
    m.cv_gamma = m.cv_h * m.cv_rl1;
    if m.cv_nst == 0 {
        m.cv_gammap = m.cv_gamma;
    }
    m.cv_gamrat = if m.cv_nst > 0 {
        m.cv_gamma / m.cv_gammap
    } else {
        ONE /* protect x / x != 1.0 */
    };
}

/*
 * cvSetAdams
 *
 * This routine handles the computation of l and tq for the
 * case lmm == CV_ADAMS.
 *
 * The components of the array l are the coefficients of a
 * polynomial Lambda(x) = l_0 + l_1 x + ... + l_q x^q, given by
 *                          q-1
 * (d/dx) Lambda(x) = c * PRODUCT (1 + x / xi_i) , where
 *                          i=1
 *  Lambda(-1) = 0, Lambda(0) = 1, and c is a normalization factor.
 * Here xi_i = [t_n - t_(n-i)] / h.
 *
 * The array tq is set to test quantities used in the convergence
 * test, the error test, and the selection of h at a new order.
 */

fn cvSetAdams(cv_mem: &CVodeMem) {
    let mut m_arr = [ZERO; L_MAX];
    let mut M = [ZERO; 3];

    {
        let mut m = cv_mem.borrow_mut();
        if m.cv_q == 1 {
            m.cv_l[0] = ONE;
            m.cv_l[1] = ONE;
            m.cv_tq[1] = ONE;
            m.cv_tq[5] = ONE;
            m.cv_tq[2] = HALF;
            m.cv_tq[3] = ONE / TWELVE;
            m.cv_tq[4] = m.cv_nlscoef / m.cv_tq[2]; /* = 0.1 / tq[2] */
            return;
        }
    }

    let hsum = cvAdamsStart(cv_mem, &mut m_arr);

    let q = cv_mem.borrow().cv_q;
    M[0] = cvAltSum(q - 1, &m_arr, 1);
    M[1] = cvAltSum(q - 1, &m_arr, 2);

    cvAdamsFinish(cv_mem, &mut m_arr, &mut M, hsum);
}

/*
 * cvAdamsStart
 *
 * This routine generates in m[] the coefficients of the product
 * polynomial needed for the Adams l and tq coefficients for q > 1.
 */

fn cvAdamsStart(cv_mem: &CVodeMem, m_: &mut [sunrealtype]) -> sunrealtype {
    let mut mm = cv_mem.borrow_mut();

    let mut hsum = mm.cv_h;
    m_[0] = ONE;
    for i in 1..=mm.cv_q as usize {
        m_[i] = ZERO;
    }
    let q = mm.cv_q;
    for j in 1..q {
        if (j == q - 1) && (mm.cv_qwait == 1) {
            let sum = cvAltSum(q - 2, m_, 2);
            mm.cv_tq[1] = q as sunrealtype * sum / m_[(q - 2) as usize];
        }
        let xi_inv = mm.cv_h / hsum;
        let mut i = j;
        while i >= 1 {
            m_[i as usize] += m_[(i - 1) as usize] * xi_inv;
            i -= 1;
        }
        hsum += mm.cv_tau[j as usize];
        /* The m[i] are coefficients of product(1 to j) (1 + x/xi_i) */
    }
    hsum
}

/*
 * cvAdamsFinish
 *
 * This routine completes the calculation of the Adams l and tq.
 */

fn cvAdamsFinish(cv_mem: &CVodeMem, m_: &mut [sunrealtype], M: &mut [sunrealtype], hsum: sunrealtype) {
    let mut mm = cv_mem.borrow_mut();

    let M0_inv = ONE / M[0];

    mm.cv_l[0] = ONE;
    for i in 1..=mm.cv_q as usize {
        mm.cv_l[i] = M0_inv * (m_[i - 1] / i as sunrealtype);
    }
    let xi = hsum / mm.cv_h;
    let xi_inv = ONE / xi;

    mm.cv_tq[2] = M[1] * M0_inv / xi;
    mm.cv_tq[5] = xi / mm.cv_l[mm.cv_q as usize];

    if mm.cv_qwait == 1 {
        let mut i = mm.cv_q;
        while i >= 1 {
            m_[i as usize] += m_[(i - 1) as usize] * xi_inv;
            i -= 1;
        }
        M[2] = cvAltSum(mm.cv_q, m_, 2);
        mm.cv_tq[3] = M[2] * M0_inv / mm.cv_L as sunrealtype;
    }

    mm.cv_tq[4] = mm.cv_nlscoef / mm.cv_tq[2];
}

/*
 * cvAltSum
 *
 * cvAltSum returns the value of the alternating sum
 *   sum (i= 0 ... iend) [ (-1)^i * (a[i] / (i + k)) ].
 * If iend < 0 then cvAltSum returns 0.
 * This operation is needed to compute the integral, from -1 to 0,
 * of a polynomial x^(k-1) M(x) given the coefficients of M(x).
 */

fn cvAltSum(iend: i32, a: &[sunrealtype], k: i32) -> sunrealtype {
    if iend < 0 {
        return ZERO;
    }

    let mut sum = ZERO;
    let mut sign: i32 = 1;
    for i in 0..=iend {
        sum += sign as sunrealtype * (a[i as usize] / (i + k) as sunrealtype);
        sign = -sign;
    }
    sum
}

/*
 * cvSetBDF
 *
 * This routine computes the coefficients l and tq in the case
 * lmm == CV_BDF.  cvSetBDF calls cvSetTqBDF to set the test
 * quantity array tq.
 *
 * The components of the array l are the coefficients of a
 * polynomial Lambda(x) = l_0 + l_1 x + ... + l_q x^q, given by
 *                                 q-1
 * Lambda(x) = (1 + x / xi*_q) * PRODUCT (1 + x / xi_i) , where
 *                                 i=1
 *
 * The components of the array p (for projections) are the
 * coefficients of a polynomial Phi(x) = p_0 + p_1 x + ... + p_q x^q,
 * given by
 *             q
 * Phi(x) = PRODUCT (1 + x / xi_i)
 *            i=1
 *
 * Here xi_i = [t_n - t_(n-i)] / h.
 *
 * The array tq is set to test quantities used in the convergence
 * test, the error test, and the selection of h at a new order.
 */

fn cvSetBDF(cv_mem: &CVodeMem) {
    let (hsum, alpha0, alpha0_hat, xi_inv, xistar_inv);
    {
        let mut m = cv_mem.borrow_mut();

        m.cv_l[0] = ONE;
        m.cv_l[1] = ONE;
        let mut xi_inv_l = ONE;
        let mut xistar_inv_l = ONE;
        for i in 2..=m.cv_q as usize {
            m.cv_l[i] = ZERO;
        }
        let mut alpha0_l = -ONE;
        let mut alpha0_hat_l = -ONE;
        let mut hsum_l = m.cv_h;

        if m.proj_enabled {
            for i in 0..=m.cv_q as usize {
                m.proj_p[i] = m.cv_l[i];
            }
        }

        let q = m.cv_q;
        if q > 1 {
            for j in 2..q {
                hsum_l += m.cv_tau[(j - 1) as usize];
                xi_inv_l = m.cv_h / hsum_l;
                alpha0_l -= ONE / j as sunrealtype;
                let mut i = j;
                while i >= 1 {
                    m.cv_l[i as usize] += m.cv_l[(i - 1) as usize] * xi_inv_l;
                    i -= 1;
                }
                /* The l[i] are coefficients of product(1 to j) (1 + x/xi_i) */
            }

            /* j = q */
            alpha0_l -= ONE / q as sunrealtype;
            xistar_inv_l = -m.cv_l[1] - alpha0_l;
            hsum_l += m.cv_tau[(q - 1) as usize];
            xi_inv_l = m.cv_h / hsum_l;
            alpha0_hat_l = -m.cv_l[1] - xi_inv_l;

            if m.proj_enabled {
                let mut i = q;
                while i >= 1 {
                    m.proj_p[i as usize] =
                        m.cv_l[i as usize] + m.proj_p[(i - 1) as usize] * xi_inv_l;
                    i -= 1;
                }
            }

            let mut i = q;
            while i >= 1 {
                m.cv_l[i as usize] += m.cv_l[(i - 1) as usize] * xistar_inv_l;
                i -= 1;
            }
        }

        hsum = hsum_l;
        alpha0 = alpha0_l;
        alpha0_hat = alpha0_hat_l;
        xi_inv = xi_inv_l;
        xistar_inv = xistar_inv_l;
    }

    cvSetTqBDF(cv_mem, hsum, alpha0, alpha0_hat, xi_inv, xistar_inv);
}

/*
 * cvSetTqBDF
 *
 * This routine sets the test quantity array tq in the case
 * lmm == CV_BDF.
 */

fn cvSetTqBDF(
    cv_mem: &CVodeMem,
    hsum: sunrealtype,
    alpha0: sunrealtype,
    alpha0_hat: sunrealtype,
    xi_inv: sunrealtype,
    xistar_inv: sunrealtype,
) {
    let mut m = cv_mem.borrow_mut();

    let mut hsum = hsum;
    let mut xi_inv = xi_inv;

    let A1 = ONE - alpha0_hat + alpha0;
    let A2 = ONE + m.cv_q as sunrealtype * A1;
    m.cv_tq[2] = SUNRabs(A1 / (alpha0 * A2));
    m.cv_tq[5] = SUNRabs(A2 * xistar_inv / (m.cv_l[m.cv_q as usize] * xi_inv));
    if m.cv_qwait == 1 {
        if m.cv_q > 1 {
            let C = xistar_inv / m.cv_l[m.cv_q as usize];
            let A3 = alpha0 + ONE / m.cv_q as sunrealtype;
            let A4 = alpha0_hat + xi_inv;
            let Cpinv = (ONE - A4 + A3) / A3;
            m.cv_tq[1] = SUNRabs(C * Cpinv);
        } else {
            m.cv_tq[1] = ONE;
        }
        hsum += m.cv_tau[m.cv_q as usize];
        xi_inv = m.cv_h / hsum;
        let A5 = alpha0 - (ONE / (m.cv_q + 1) as sunrealtype);
        let A6 = alpha0_hat - xi_inv;
        let Cppinv = (ONE - A6 + A5) / A2;
        m.cv_tq[3] = SUNRabs(Cppinv / (xi_inv * (m.cv_q + 2) as sunrealtype * A5));
    }
    m.cv_tq[4] = m.cv_nlscoef / m.cv_tq[2];
}

/*
 * -----------------------------------------------------------------
 * Nonlinear solver functions
 * -----------------------------------------------------------------
 */

/*
 * cvNls
 *
 * This routine attempts to solve the nonlinear system associated
 * with a single implicit step of the linear multistep method.
 */

fn cvNls(cv_mem: &CVodeMem, nflag: i32) -> i32 {
    let callSetup: sunbooleantype;
    let mut nni_inc: i64 = 0;
    let mut nnf_inc: i64 = 0;

    /* (C initializes flag = CV_SUCCESS here; every read below follows an
    assignment, so the dead store is omitted.) */

    /* Decide whether or not to call setup routine (if one exists) and */
    /* set flag convfail (input to lsetup for its evaluation decision) */
    {
        let mut m = cv_mem.borrow_mut();
        if m.cv_lsetup.is_some() {
            m.convfail = if (nflag == FIRST_CALL) || (nflag == PREV_ERR_FAIL) {
                CV_NO_FAILURES
            } else {
                CV_FAIL_OTHER
            };

            callSetup = (nflag == PREV_CONV_FAIL)
                || (nflag == PREV_ERR_FAIL)
                || (m.cv_nst == 0)
                || (m.first_step_after_resize)
                || (m.cv_nst >= m.cv_nstlp + m.cv_msbp)
                || (SUNRabs(m.cv_gamrat - ONE) > m.cv_dgmax_lsetup);
        } else {
            m.cv_crate = ONE;
            callSetup = SUNFALSE;
        }
    }

    /* initial guess for the correction to the predictor */
    let acor = cv_mem.borrow().cv_acor.clone().unwrap();
    N_VConst(ZERO, &acor);

    /* The C `void*` integrator mem handed to the nonlinear solver maps to a
    boxed handle clone (the same token shape cvode_nls.rs downcasts) */
    let NLS = cv_mem.borrow().NLS.clone().unwrap();
    let mut nls_mem: Option<Box<dyn Any>> = Some(Box::new(cv_mem.clone()));

    /* call nonlinear solver setup if it exists */
    if NLS.ops.borrow().setup.is_some() {
        let flag = SUNNonlinSolSetup(&NLS, &acor, &mut nls_mem);
        if flag < 0 {
            return CV_NLS_SETUP_FAIL;
        }
        if flag > 0 {
            return SUN_NLS_CONV_RECVR;
        }
    }

    /* solve the nonlinear system */
    let (zn0, ewt, tq4) = {
        let m = cv_mem.borrow();
        (
            m.cv_zn[0].clone().unwrap(),
            m.cv_ewt.clone().unwrap(),
            m.cv_tq[4],
        )
    };
    let flag = SUNNonlinSolSolve(&NLS, &zn0, &acor, &ewt, tq4, callSetup, &mut nls_mem);

    /* increment counters */
    let _ = SUNNonlinSolGetNumIters(&NLS, &mut nni_inc);
    cv_mem.borrow_mut().cv_nni += nni_inc;

    let _ = SUNNonlinSolGetNumConvFails(&NLS, &mut nnf_inc);
    cv_mem.borrow_mut().cv_nnf += nnf_inc;

    /* if the solve failed return */
    if flag != SUN_SUCCESS {
        return flag;
    }

    /* solve successful */

    /* update the state based on the final correction from the nonlinear solver */
    let y = cv_mem.borrow().cv_y.clone().unwrap();
    N_VLinearSum(ONE, &zn0, ONE, &acor, &y);

    /* compute acnrm if is was not already done by the nonlinear solver */
    if !cv_mem.borrow().cv_acnrmcur {
        let acnrm = N_VWrmsNorm(&acor, &ewt);
        cv_mem.borrow_mut().cv_acnrm = acnrm;
    }

    /* update Jacobian status */
    cv_mem.borrow_mut().cv_jcur = SUNFALSE;

    flag
}

/*
 * cvCheckConstraints
 *
 * This routine determines if the constraints of the problem
 * are satisfied by the proposed step
 *
 * Possible return values are:
 *
 *   CV_SUCCESS     ---> allows stepping forward
 *
 *   PREDICT_AGAIN  ---> values failed to satisfy constraints
 *
 *   CV_CONSTR_FAIL ---> values failed to satisfy constraints with hmin
 */

fn cvCheckConstraints(
    cv_mem: &CVodeMem,
    nflagPtr: &mut i32,
    saved_t: sunrealtype,
    step_constraint_fails: &mut i32,
) -> i32 {
    let (mm, tmp, constraints, y, ewt) = {
        let m = cv_mem.borrow();
        (
            m.cv_ftemp.clone().unwrap(), /* mask      */
            m.cv_tempv.clone().unwrap(), /* workspace */
            m.cv_constraints.clone().unwrap(),
            m.cv_y.clone().unwrap(),
            m.cv_ewt.clone().unwrap(),
        )
    };

    /* Get mask vector mm, 1 where constraints failed and 0 otherwise */
    let constraintsPassed = N_VConstrMask(&constraints, &y, &mm);
    if constraintsPassed {
        return CV_SUCCESS;
    }

    /* Constraints not met */

    /* Compute correction v such that y - v will satisfy the constraints
     *
     * 1. Create a mask array that is +1 where constraints are strictly greater
     *    than or less than zero (|c[i]| = 2) and 0 otherwise
     *
     * 2. Create a mask array that is +/- 2 where constraints are strictly greater
     *    than (+) or less than (-) zero and 0 otherwise
     *
     * 3. Use error weights to compute an adjustment vector for values with strict
     *    constraints, a[i] = +/- 2 * w[i] = +/- 2 * (atol * |y[i]| + rtol[i]),
     *    and is 0 otherwise
     *
     * 4. Save the adjustment vector for possible use later
     *
     * 5. Compute correction vector for all values, v[i] = y[i] - 0.1 * a[i] for
     *    strict constraints and v[i] = y[i] otherwise
     *
     * 6. Zero out entries where the constraints passed, v = mask * v
     */
    /* SUNDIALS_ENABLE_PACKAGE_FUSED_KERNELS not defined: unfused branch */
    let vtemp1 = cv_mem.borrow().cv_vtemp1.clone().unwrap();
    N_VCompare(ONEPT5, &constraints, &tmp);
    N_VProd(&tmp, &constraints, &tmp);
    N_VDiv(&tmp, &ewt, &tmp);
    N_VScale(-PT1, &tmp, &vtemp1);
    N_VLinearSum(ONE, &y, -PT1, &tmp, &tmp);
    N_VProd(&tmp, &mm, &tmp);

    let vnorm = N_VWrmsNorm(&tmp, &ewt); /* ||v|| */

    /* If constraint correction vector is small in norm (satisfies the nonlinear
    solver convergence condition with R = 1), correct and accept this step */
    if vnorm <= cv_mem.borrow().cv_tq[4] {
        /* Update constraint correction count */
        cv_mem.borrow_mut().constraint_corrections += 1;

        /* To reduce roundoff errors that can violate the constraints, split the
         * correction update, acor = acor - v, into three steps */

        let (acor, zn0) = {
            let m = cv_mem.borrow();
            (m.cv_acor.clone().unwrap(), m.cv_zn[0].clone().unwrap())
        };

        /* Zero out the correction where any constraint failed */
        N_VProd(&mm, &acor, &tmp);
        N_VLinearSum(ONE, &acor, -ONE, &tmp, &acor);

        /* Set correction to zero out the predictor where any constraint failed */
        N_VProd(&mm, &zn0, &tmp);
        N_VLinearSum(ONE, &acor, -ONE, &tmp, &acor);

        /* Update the correction where constraints failed and are strictly greater
        or less than zero to shift the state with the adjustment saved above */
        N_VProd(&mm, &vtemp1, &vtemp1);
        N_VLinearSum(ONE, &acor, -ONE, &vtemp1, &acor);

        return CV_SUCCESS;
    }

    /* update failure counts */
    *step_constraint_fails += 1;
    cv_mem.borrow_mut().constraint_fails += 1;

    /* restore zn */
    cvRestore(cv_mem, saved_t);

    /* Check for |h| == hmin */
    {
        let m = cv_mem.borrow();
        if SUNRabs(m.cv_h) <= m.cv_hmin * ONEPSM {
            return CV_CONSTR_FAIL;
        }
    }

    /* Check for max step attempt failures */
    if *step_constraint_fails == cv_mem.borrow().max_constraint_fails {
        return CV_CONSTR_FAIL;
    }

    /* Constraint correction is too large, reduce h by computing eta = h'/h */
    let zn0 = cv_mem.borrow().cv_zn[0].clone().unwrap();
    N_VLinearSum(ONE, &zn0, -ONE, &y, &tmp);
    N_VProd(&mm, &tmp, &tmp);

    /* Reduce step size; return to reattempt the step */
    let minq = N_VMinQuotient(&zn0, &tmp);
    {
        let mut m = cv_mem.borrow_mut();
        m.cv_eta = PT9 * minq;
        m.cv_eta = SUNMAX(m.cv_eta, PT1);
        m.cv_eta = SUNMAX(m.cv_eta, m.cv_hmin / SUNRabs(m.cv_h));
    }
    cvRescale(cv_mem);
    *nflagPtr = PREV_CONV_FAIL;

    PREDICT_AGAIN
}

/*
 * cvHandleNFlag
 *
 * This routine takes action on the return value nflag = *nflagPtr
 * returned by cvNls, as follows:
 *
 * If cvNls succeeded in solving the nonlinear system, then
 * cvHandleNFlag returns the constant DO_ERROR_TEST, which tells cvStep
 * to perform the error test.
 *
 * If the nonlinear system was not solved successfully, then ncfn and
 * ncf = *ncfPtr are incremented and Nordsieck array zn is restored.
 *
 * If the solution of the nonlinear system failed due to an
 * unrecoverable failure by setup, we return the value CV_LSETUP_FAIL.
 *
 * If it failed due to an unrecoverable failure in solve, then we return
 * the value CV_LSOLVE_FAIL.
 *
 * If it failed due to an unrecoverable failure in rhs, then we return
 * the value CV_RHSFUNC_FAIL.
 *
 * Otherwise, a recoverable failure occurred when solving the nonlinear system
 * (cvNls returned SUN_NLS_CONV_RECVR or RHSFUNC_RECVR).
 *
 * If ncf is now equal to maxncf or |h| = hmin, we return the value
 * CV_CONV_FAILURE (if SUN_NLS_CONV_RECVR) or
 * CV_REPTD_RHSFUNC_ERR (if RHSFUNC_RECVR).
 * Otherwise, we set *nflagPtr = PREV_CONV_FAIL and return the value
 * PREDICT_AGAIN, telling cvStep to reattempt the step.
 *
 */

fn cvHandleNFlag(
    cv_mem: &CVodeMem,
    nflagPtr: &mut i32,
    saved_t: sunrealtype,
    ncfPtr: &mut i32,
) -> i32 {
    let nflag = *nflagPtr;

    if nflag == CV_SUCCESS {
        return DO_ERROR_TEST;
    }

    /* The nonlinear soln. failed; increment ncfn and restore zn */
    cv_mem.borrow_mut().cv_ncfn += 1;
    cvRestore(cv_mem, saved_t);

    /* Return if failed unrecoverably */
    if nflag < 0 {
        if nflag == CV_LSETUP_FAIL {
            return CV_LSETUP_FAIL;
        } else if nflag == CV_LSOLVE_FAIL {
            return CV_LSOLVE_FAIL;
        } else if nflag == CV_RHSFUNC_FAIL {
            return CV_RHSFUNC_FAIL;
        } else {
            return CV_NLS_FAIL;
        }
    }

    /* At this point, a recoverable error occurred. */

    *ncfPtr += 1;
    cv_mem.borrow_mut().cv_etamax = ONE;

    /* If we had maxncf failures or |h| = hmin, return failure. */

    {
        let m = cv_mem.borrow();
        if (SUNRabs(m.cv_h) <= m.cv_hmin * ONEPSM) || (*ncfPtr == m.cv_maxncf) {
            if nflag == SUN_NLS_CONV_RECVR {
                return CV_CONV_FAILURE;
            }
            if nflag == RHSFUNC_RECVR {
                return CV_REPTD_RHSFUNC_ERR;
            }
        }
    }

    /* Reduce step size; return to reattempt the step */
    {
        let mut m = cv_mem.borrow_mut();
        m.cv_eta = SUNMAX(m.cv_eta_cf, m.cv_hmin / SUNRabs(m.cv_h));
    }
    *nflagPtr = PREV_CONV_FAIL;
    cvRescale(cv_mem);

    PREDICT_AGAIN
}

/*
 * cvRestore
 *
 * This routine restores the value of tn to saved_t and undoes the
 * prediction.  After execution of cvRestore, the Nordsieck array zn has
 * the same values as before the call to cvPredict.
 */

pub fn cvRestore(cv_mem: &CVodeMem, saved_t: sunrealtype) {
    cv_mem.borrow_mut().cv_tn = saved_t;
    let (q, zn) = {
        let m = cv_mem.borrow();
        let zn: Vec<N_Vector> = (0..=m.cv_q as usize)
            .map(|j| m.cv_zn[j].clone().unwrap())
            .collect();
        (m.cv_q, zn)
    };
    for k in 1..=q {
        let mut j = q;
        while j >= k {
            N_VLinearSum(
                ONE,
                &zn[(j - 1) as usize],
                -ONE,
                &zn[j as usize],
                &zn[(j - 1) as usize],
            );
            j -= 1;
        }
    }
}

/*
 * -----------------------------------------------------------------
 * Error Test
 * -----------------------------------------------------------------
 */

/*
 * cvDoErrorTest
 *
 * This routine performs the local error test.
 * The weighted local error norm dsm is loaded into *dsmPtr, and
 * the test dsm ?<= 1 is made.
 *
 * If the test passes, cvDoErrorTest returns CV_SUCCESS.
 *
 * If the test fails, we undo the step just taken (call cvRestore) and
 *
 *   - if maxnef error test failures have occurred or if SUNRabs(h) = hmin,
 *     we return CV_ERR_FAILURE.
 *
 *   - if more than MXNEF1 error test failures have occurred, an order
 *     reduction is forced. If already at order 1, restart by reloading
 *     zn from scratch. If f() fails we return either CV_RHSFUNC_FAIL
 *     or CV_UNREC_RHSFUNC_ERR (no recovery is possible at this stage).
 *
 *   - otherwise, set *nflagPtr to PREV_ERR_FAIL, and return TRY_AGAIN.
 *
 */

fn cvDoErrorTest(
    cv_mem: &CVodeMem,
    nflagPtr: &mut i32,
    saved_t: sunrealtype,
    nefPtr: &mut i32,
    dsmPtr: &mut sunrealtype,
) -> i32 {
    let dsm = {
        let m = cv_mem.borrow();
        m.cv_acnrm * m.cv_tq[2]
    };

    /* If est. local error norm dsm passes test, return CV_SUCCESS */
    *dsmPtr = dsm;
    if dsm <= ONE {
        return CV_SUCCESS;
    }

    /* Test failed; increment counters, set nflag, and restore zn array */
    *nefPtr += 1;
    cv_mem.borrow_mut().cv_netf += 1;
    *nflagPtr = PREV_ERR_FAIL;
    cvRestore(cv_mem, saved_t);

    /* At maxnef failures or |h| = hmin, return CV_ERR_FAILURE */
    {
        let m = cv_mem.borrow();
        if (SUNRabs(m.cv_h) <= m.cv_hmin * ONEPSM) || (*nefPtr == m.cv_maxnef) {
            return CV_ERR_FAILURE;
        }
    }

    /* Set etamax = 1 to prevent step size increase at end of this step */
    cv_mem.borrow_mut().cv_etamax = ONE;

    /* Set h ratio eta from dsm, rescale, and return for retry of step */
    if *nefPtr <= MXNEF1 {
        {
            let mut m = cv_mem.borrow_mut();
            m.cv_eta = ONE / (SUNRpowerR(BIAS2 * dsm, ONE / m.cv_L as sunrealtype) + ADDON);
            m.cv_eta = SUNMAX(
                m.cv_eta_min_ef,
                SUNMAX(m.cv_eta, m.cv_hmin / SUNRabs(m.cv_h)),
            );
            if *nefPtr >= m.cv_small_nef {
                m.cv_eta = SUNMIN(m.cv_eta, m.cv_eta_max_ef);
            }
        }

        cvRescale(cv_mem);

        return TRY_AGAIN;
    }

    /* After MXNEF1 failures, force an order reduction and retry step */
    if cv_mem.borrow().cv_q > 1 {
        {
            let mut m = cv_mem.borrow_mut();
            m.cv_eta = SUNMAX(m.cv_eta_min_ef, m.cv_hmin / SUNRabs(m.cv_h));
        }
        cvAdjustOrder(cv_mem, -1);
        {
            let mut m = cv_mem.borrow_mut();
            m.cv_L = m.cv_q;
            m.cv_q -= 1;
            m.cv_qwait = m.cv_L;
        }
        cvRescale(cv_mem);
        return TRY_AGAIN;
    }

    /* If already at order 1, restart: reload zn from scratch */

    {
        let mut m = cv_mem.borrow_mut();
        m.cv_eta = SUNMAX(m.cv_eta_min_ef, m.cv_hmin / SUNRabs(m.cv_h));
        m.cv_h *= m.cv_eta;
        m.cv_next_h = m.cv_h;
        m.cv_hscale = m.cv_h;
        m.cv_qwait = LONG_WAIT;
        m.cv_nscon = 0;
    }

    let (tn, zn0, tempv) = {
        let m = cv_mem.borrow();
        (
            m.cv_tn,
            m.cv_zn[0].clone().unwrap(),
            m.cv_tempv.clone().unwrap(),
        )
    };
    let retval = cv_call_f(cv_mem, tn, &zn0, &tempv);
    cv_mem.borrow_mut().cv_nfe += 1;
    if retval < 0 {
        return CV_RHSFUNC_FAIL;
    }
    if retval > 0 {
        return CV_UNREC_RHSFUNC_ERR;
    }

    let (h, zn1) = {
        let m = cv_mem.borrow();
        (m.cv_h, m.cv_zn[1].clone().unwrap())
    };
    N_VScale(h, &tempv, &zn1);

    TRY_AGAIN
}

/*
 * -----------------------------------------------------------------
 * Functions called after a successful step
 * -----------------------------------------------------------------
 */

/*
 * cvCompleteStep
 *
 * This routine performs various update operations when the solution
 * to the nonlinear system has passed the local error test.
 * We increment the step counter nst, record the values hu and qu,
 * update the tau array, and apply the corrections to the zn array.
 * The tau[i] are the last q values of h, with tau[1] the most recent.
 * The counter qwait is decremented, and if qwait == 1 (and q < qmax)
 * we save acor and tq[5] for a possible order increase.
 */

fn cvCompleteStep(cv_mem: &CVodeMem) {
    {
        let mut m = cv_mem.borrow_mut();
        m.cv_nst += 1;
        m.cv_nscon += 1;
        m.cv_hu = m.cv_h;
        m.cv_qu = m.cv_q;

        m.first_step_after_resize = SUNFALSE;

        let mut i = m.cv_q;
        while i >= 2 {
            m.cv_tau[i as usize] = m.cv_tau[(i - 1) as usize];
            i -= 1;
        }
        if (m.cv_q == 1) && (m.cv_nst > 1) {
            m.cv_tau[2] = m.cv_tau[1];
        }
        m.cv_tau[1] = m.cv_h;
    }

    /* Apply correction to column j of zn: l_j * Delta_n */
    let (q, lvals, acor, znvec) = {
        let m = cv_mem.borrow();
        let znvec: Vec<N_Vector> = (0..=m.cv_q as usize)
            .map(|j| m.cv_zn[j].clone().unwrap())
            .collect();
        (m.cv_q, m.cv_l, m.cv_acor.clone().unwrap(), znvec)
    };
    let _ = N_VScaleAddMulti(q + 1, &lvals, &acor, &znvec, &znvec);

    /* Apply the projection correction to column j of zn: p_j * Delta_n */
    if cv_mem.borrow().proj_applied {
        let (pvals, tempv) = {
            let m = cv_mem.borrow();
            (
                m.proj_p,
                m.cv_tempv.clone().unwrap(), /* tempv = acorP */
            )
        };
        let _ = N_VScaleAddMulti(q + 1, &pvals, &tempv, &znvec, &znvec);
    }

    cv_mem.borrow_mut().cv_qwait -= 1;
    let (qwait, q, qmax) = {
        let m = cv_mem.borrow();
        (m.cv_qwait, m.cv_q, m.cv_qmax)
    };
    if (qwait == 1) && (q != qmax) {
        let znqmax = cv_mem.borrow().cv_zn[qmax as usize].clone().unwrap();
        N_VScale(ONE, &acor, &znqmax);
        let mut m = cv_mem.borrow_mut();
        m.cv_saved_tq5 = m.cv_tq[5];
        m.cv_indx_acor = m.cv_qmax;
    }

    /* SUNDIALS_ENABLE_MONITORING defined in the reference build */
    /* If user access function was provided, call it now */
    let (monitorfun, monitor_interval) = {
        let m = cv_mem.borrow();
        (m.cv_monitorfun, m.cv_monitor_interval)
    };
    if let Some(monitorfun) = monitorfun {
        if cv_mem.borrow().cv_nst % monitor_interval == 0 {
            let mut user_data = cv_mem.borrow_mut().cv_user_data.take();
            let _ = monitorfun(cv_mem, &mut user_data);
            cv_mem.borrow_mut().cv_user_data = user_data;
        }
    }
}

/*
 * cvPrepareNextStep
 *
 * This routine handles the setting of stepsize and order for the
 * next step -- hprime and qprime.  Along with hprime, it sets the
 * ratio eta = hprime/h.  It also updates other state variables
 * related to a change of step size or order.
 */

fn cvPrepareNextStep(cv_mem: &CVodeMem, dsm: sunrealtype) {
    /* If etamax = 1, defer step size or order changes */
    if cv_mem.borrow().cv_etamax == ONE {
        let mut m = cv_mem.borrow_mut();
        m.cv_qwait = SUNMAX(m.cv_qwait, 2);
        m.cv_qprime = m.cv_q;
        m.cv_hprime = m.cv_h;
        m.cv_eta = ONE;
    } else {
        /* etaq is the ratio of new to old h at the current order */
        {
            let mut m = cv_mem.borrow_mut();
            m.cv_etaq = ONE / (SUNRpowerR(BIAS2 * dsm, ONE / m.cv_L as sunrealtype) + ADDON);
        }

        /* If no order change, adjust eta and acor in cvSetEta and return */
        if cv_mem.borrow().cv_qwait != 0 {
            {
                let mut m = cv_mem.borrow_mut();
                m.cv_eta = m.cv_etaq;
                m.cv_qprime = m.cv_q;
            }
            cvSetEta(cv_mem);
        } else {
            /* If qwait = 0, consider an order change.   etaqm1 and etaqp1 are
            the ratios of new to old h at orders q-1 and q+1, respectively.
            cvChooseEta selects the largest; cvSetEta adjusts eta and acor */
            cv_mem.borrow_mut().cv_qwait = 2;
            let etaqm1 = cvComputeEtaqm1(cv_mem);
            cv_mem.borrow_mut().cv_etaqm1 = etaqm1;
            let etaqp1 = cvComputeEtaqp1(cv_mem);
            cv_mem.borrow_mut().cv_etaqp1 = etaqp1;
            cvChooseEta(cv_mem);
            cvSetEta(cv_mem);
        }
    }
}

/*
 * cvSetEta
 *
 * This routine adjusts the value of eta according to the various
 * heuristic limits and the optional input hmax.
 */

fn cvSetEta(cv_mem: &CVodeMem) {
    let mut m = cv_mem.borrow_mut();

    if (m.cv_eta > m.cv_eta_min_fx) && (m.cv_eta < m.cv_eta_max_fx) {
        /* Eta is within the fixed step bounds, retain step size */
        m.cv_eta = ONE;
        m.cv_hprime = m.cv_h;
    } else {
        if m.cv_eta >= m.cv_eta_max_fx {
            /* Increase the step size, limit eta by etamax and hmax */
            m.cv_eta = SUNMIN(m.cv_eta, m.cv_etamax);
            m.cv_eta /= SUNMAX(ONE, SUNRabs(m.cv_h) * m.cv_hmax_inv * m.cv_eta);
        } else {
            /* Reduce the step size, limit eta by etamin and hmin */
            m.cv_eta = SUNMAX(m.cv_eta, m.cv_eta_min);
            m.cv_eta = SUNMAX(m.cv_eta, m.cv_hmin / SUNRabs(m.cv_h));
        }
        /* Set hprime */
        m.cv_hprime = m.cv_h * m.cv_eta;
        if m.cv_qprime < m.cv_q {
            m.cv_nscon = 0;
        }
    }
}

/*
 * cvComputeEtaqm1
 *
 * This routine computes and returns the value of etaqm1 for a
 * possible decrease in order by 1.
 */

fn cvComputeEtaqm1(cv_mem: &CVodeMem) -> sunrealtype {
    cv_mem.borrow_mut().cv_etaqm1 = ZERO;
    if cv_mem.borrow().cv_q > 1 {
        let (znq, ewt, tq1, q) = {
            let m = cv_mem.borrow();
            (
                m.cv_zn[m.cv_q as usize].clone().unwrap(),
                m.cv_ewt.clone().unwrap(),
                m.cv_tq[1],
                m.cv_q,
            )
        };
        let ddn = N_VWrmsNorm(&znq, &ewt) * tq1;
        cv_mem.borrow_mut().cv_etaqm1 =
            ONE / (SUNRpowerR(BIAS1 * ddn, ONE / q as sunrealtype) + ADDON);
    }
    cv_mem.borrow().cv_etaqm1
}

/*
 * cvComputeEtaqp1
 *
 * This routine computes and returns the value of etaqp1 for a
 * possible increase in order by 1.
 */

fn cvComputeEtaqp1(cv_mem: &CVodeMem) -> sunrealtype {
    cv_mem.borrow_mut().cv_etaqp1 = ZERO;
    let (q, qmax) = {
        let m = cv_mem.borrow();
        (m.cv_q, m.cv_qmax)
    };
    if q != qmax {
        if cv_mem.borrow().cv_saved_tq5 == ZERO {
            return cv_mem.borrow().cv_etaqp1;
        }
        let (cquot, znqmax, acor, tempv, ewt, tq3, L) = {
            let m = cv_mem.borrow();
            (
                (m.cv_tq[5] / m.cv_saved_tq5) * SUNRpowerI(m.cv_h / m.cv_tau[2], m.cv_L),
                m.cv_zn[m.cv_qmax as usize].clone().unwrap(),
                m.cv_acor.clone().unwrap(),
                m.cv_tempv.clone().unwrap(),
                m.cv_ewt.clone().unwrap(),
                m.cv_tq[3],
                m.cv_L,
            )
        };
        N_VLinearSum(-cquot, &znqmax, ONE, &acor, &tempv);
        let dup = N_VWrmsNorm(&tempv, &ewt) * tq3;
        cv_mem.borrow_mut().cv_etaqp1 =
            ONE / (SUNRpowerR(BIAS3 * dup, ONE / (L + 1) as sunrealtype) + ADDON);
    }
    cv_mem.borrow().cv_etaqp1
}

/*
 * cvChooseEta
 * Given etaqm1, etaq, etaqp1 (the values of eta for qprime =
 * q - 1, q, or q + 1, respectively), this routine chooses the
 * maximum eta value, sets eta to that value, and sets qprime to the
 * corresponding value of q.  If there is a tie, the preference
 * order is to (1) keep the same order, then (2) decrease the order,
 * and finally (3) increase the order.  If the maximum eta value
 * is within the fixed step bounds, the order is kept unchanged and
 * eta is set to 1.
 */

fn cvChooseEta(cv_mem: &CVodeMem) {
    let (etam, etaq, etaqm1) = {
        let m = cv_mem.borrow();
        (
            SUNMAX(m.cv_etaqm1, SUNMAX(m.cv_etaq, m.cv_etaqp1)),
            m.cv_etaq,
            m.cv_etaqm1,
        )
    };

    let within_fx = {
        let m = cv_mem.borrow();
        (etam > m.cv_eta_min_fx) && (etam < m.cv_eta_max_fx)
    };

    if within_fx {
        let mut m = cv_mem.borrow_mut();
        m.cv_eta = ONE;
        m.cv_qprime = m.cv_q;
    } else {
        if etam == etaq {
            let mut m = cv_mem.borrow_mut();
            m.cv_eta = m.cv_etaq;
            m.cv_qprime = m.cv_q;
        } else if etam == etaqm1 {
            let mut m = cv_mem.borrow_mut();
            m.cv_eta = m.cv_etaqm1;
            m.cv_qprime = m.cv_q - 1;
        } else {
            {
                let mut m = cv_mem.borrow_mut();
                m.cv_eta = m.cv_etaqp1;
                m.cv_qprime = m.cv_q + 1;
            }

            if cv_mem.borrow().cv_lmm == CV_BDF {
                /*
                 * Store Delta_n in zn[qmax] to be used in order increase
                 *
                 * This happens at the last step of order q before an increase
                 * to order q+1, so it represents Delta_n in the ELTE at q+1
                 */

                let (acor, znqmax) = {
                    let m = cv_mem.borrow();
                    (
                        m.cv_acor.clone().unwrap(),
                        m.cv_zn[m.cv_qmax as usize].clone().unwrap(),
                    )
                };
                N_VScale(ONE, &acor, &znqmax);
            }
        }
    }
}

/*
 * -----------------------------------------------------------------
 * Function to handle failures
 * -----------------------------------------------------------------
 */

/*
 * cvHandleFailure
 *
 * This routine prints error messages for all cases of failure by
 * cvHin and cvStep.
 * It returns to CVode the value that CVode is to return to the user.
 */

fn cvHandleFailure(cv_mem: &CVodeMem, flag: i32) -> i32 {
    /* Set vector of  absolute weighted local errors */
    /*
    N_VProd(acor, ewt, tempv);
    N_VAbs(tempv, tempv);
    */

    let (tn, h) = {
        let m = cv_mem.borrow();
        (m.cv_tn, m.cv_h)
    };

    /* Depending on flag, print error message and return error flag */
    match flag {
        CV_ERR_FAILURE => {
            cvProcessError(
                Some(cv_mem),
                CV_ERR_FAILURE,
                line!() as i32,
                "cvHandleFailure",
                file!(),
                &MSGCV_ERR_FAILS(tn, h),
            );
        }
        CV_CONV_FAILURE => {
            cvProcessError(
                Some(cv_mem),
                CV_CONV_FAILURE,
                line!() as i32,
                "cvHandleFailure",
                file!(),
                &MSGCV_CONV_FAILS(tn, h),
            );
        }
        CV_LSETUP_FAIL => {
            cvProcessError(
                Some(cv_mem),
                CV_LSETUP_FAIL,
                line!() as i32,
                "cvHandleFailure",
                file!(),
                &MSGCV_SETUP_FAILED(tn),
            );
        }
        CV_LSOLVE_FAIL => {
            cvProcessError(
                Some(cv_mem),
                CV_LSOLVE_FAIL,
                line!() as i32,
                "cvHandleFailure",
                file!(),
                &MSGCV_SOLVE_FAILED(tn),
            );
        }
        CV_RHSFUNC_FAIL => {
            cvProcessError(
                Some(cv_mem),
                CV_RHSFUNC_FAIL,
                line!() as i32,
                "cvHandleFailure",
                file!(),
                &MSGCV_RHSFUNC_FAILED(tn),
            );
        }
        CV_UNREC_RHSFUNC_ERR => {
            cvProcessError(
                Some(cv_mem),
                CV_UNREC_RHSFUNC_ERR,
                line!() as i32,
                "cvHandleFailure",
                file!(),
                &MSGCV_RHSFUNC_UNREC(tn),
            );
        }
        CV_REPTD_RHSFUNC_ERR => {
            cvProcessError(
                Some(cv_mem),
                CV_REPTD_RHSFUNC_ERR,
                line!() as i32,
                "cvHandleFailure",
                file!(),
                &MSGCV_RHSFUNC_REPTD(tn),
            );
        }
        CV_RTFUNC_FAIL => {
            cvProcessError(
                Some(cv_mem),
                CV_RTFUNC_FAIL,
                line!() as i32,
                "cvHandleFailure",
                file!(),
                &MSGCV_RTFUNC_FAILED(tn),
            );
        }
        CV_TOO_CLOSE => {
            cvProcessError(
                Some(cv_mem),
                CV_TOO_CLOSE,
                line!() as i32,
                "cvHandleFailure",
                file!(),
                MSGCV_TOO_CLOSE,
            );
        }
        CV_MEM_NULL => {
            cvProcessError(
                None,
                CV_MEM_NULL,
                line!() as i32,
                "cvHandleFailure",
                file!(),
                MSGCV_NO_MEM,
            );
        }
        SUN_ERR_ARG_CORRUPT => {
            cvProcessError(
                Some(cv_mem),
                CV_MEM_NULL,
                line!() as i32,
                "cvHandleFailure",
                file!(),
                &MSGCV_NLS_INPUT_NULL(tn),
            );
        }
        CV_NLS_SETUP_FAIL => {
            cvProcessError(
                Some(cv_mem),
                CV_NLS_SETUP_FAIL,
                line!() as i32,
                "cvHandleFailure",
                file!(),
                &MSGCV_NLS_SETUP_FAILED(tn),
            );
        }
        CV_CONSTR_FAIL => {
            cvProcessError(
                Some(cv_mem),
                CV_CONSTR_FAIL,
                line!() as i32,
                "cvHandleFailure",
                file!(),
                &MSGCV_FAILED_CONSTR(tn),
            );
        }
        CV_NLS_FAIL => {
            cvProcessError(
                Some(cv_mem),
                CV_NLS_FAIL,
                line!() as i32,
                "cvHandleFailure",
                file!(),
                &MSGCV_NLS_FAIL(tn),
            );
        }
        CV_PROJ_MEM_NULL => {
            cvProcessError(
                Some(cv_mem),
                CV_PROJ_MEM_NULL,
                line!() as i32,
                "cvHandleFailure",
                file!(),
                MSG_CV_PROJ_MEM_NULL,
            );
        }
        CV_PROJFUNC_FAIL => {
            cvProcessError(
                Some(cv_mem),
                CV_PROJFUNC_FAIL,
                line!() as i32,
                "cvHandleFailure",
                file!(),
                &MSG_CV_PROJFUNC_FAIL(tn),
            );
        }
        CV_REPTD_PROJFUNC_ERR => {
            cvProcessError(
                Some(cv_mem),
                CV_REPTD_PROJFUNC_ERR,
                line!() as i32,
                "cvHandleFailure",
                file!(),
                &MSG_CV_REPTD_PROJFUNC_ERR(tn),
            );
        }
        _ => {
            /* This return should never happen */
            cvProcessError(
                Some(cv_mem),
                CV_UNRECOGNIZED_ERR,
                line!() as i32,
                "cvHandleFailure",
                file!(),
                "CVODE encountered an unrecognized error. Please report this to the SUNDIALS developers at sundials-users@llnl.gov",
            );
            return CV_UNRECOGNIZED_ERR;
        }
    }

    flag
}

/*
 * -----------------------------------------------------------------
 * Functions for BDF Stability Limit Detection
 * -----------------------------------------------------------------
 */

/*
 * cvBDFStab
 *
 * This routine handles the BDF Stability Limit Detection Algorithm
 * STALD.  It is called if lmm = CV_BDF and the SLDET option is on.
 * If the order is 3 or more, the required norm data is saved.
 * If a decision to reduce order has not already been made, and
 * enough data has been saved, cvSLdet is called.  If it signals
 * a stability limit violation, the order is reduced, and the step
 * size is reset accordingly.
 */

fn cvBDFStab(cv_mem: &CVodeMem) {
    /* If order is 3 or greater, then save scaled derivative data,
    push old data down in i, then add current values to top.    */

    if cv_mem.borrow().cv_q >= 3 {
        {
            let mut m = cv_mem.borrow_mut();
            for k in 1..=3usize {
                let mut i = 5usize;
                while i >= 2 {
                    m.cv_ssdat[i][k] = m.cv_ssdat[i - 1][k];
                    i -= 1;
                }
            }
        }
        let (factorial, q, acnrm, tq5, znq, znqm1, ewt) = {
            let m = cv_mem.borrow();
            let mut factorial: i32 = 1;
            for i in 1..=(m.cv_q - 1) {
                factorial *= i;
            }
            (
                factorial,
                m.cv_q,
                m.cv_acnrm,
                m.cv_tq[5],
                m.cv_zn[m.cv_q as usize].clone().unwrap(),
                m.cv_zn[(m.cv_q - 1) as usize].clone().unwrap(),
                m.cv_ewt.clone().unwrap(),
            )
        };
        let sq = (factorial * q * (q + 1)) as sunrealtype * acnrm / SUNMAX(tq5, TINY);
        let sqm1 = (factorial * q) as sunrealtype * N_VWrmsNorm(&znq, &ewt);
        let sqm2 = factorial as sunrealtype * N_VWrmsNorm(&znqm1, &ewt);
        let mut m = cv_mem.borrow_mut();
        m.cv_ssdat[1][1] = sqm2 * sqm2;
        m.cv_ssdat[1][2] = sqm1 * sqm1;
        m.cv_ssdat[1][3] = sq * sq;
    }

    if cv_mem.borrow().cv_qprime >= cv_mem.borrow().cv_q {
        /* If order is 3 or greater, and enough ssdat has been saved,
        nscon >= q+5, then call stability limit detection routine.  */

        let sldet_check = {
            let m = cv_mem.borrow();
            (m.cv_q >= 3) && (m.cv_nscon >= m.cv_q + 5)
        };
        if sldet_check {
            let ldflag = cvSLdet(cv_mem);
            if ldflag > 3 {
                /* A stability limit violation is indicated by
                a return flag of 4, 5, or 6.
                Reduce new order.                     */
                let mut m = cv_mem.borrow_mut();
                m.cv_qprime = m.cv_q - 1;
                m.cv_eta = m.cv_etaqm1;
                m.cv_eta = SUNMIN(m.cv_eta, m.cv_etamax);
                m.cv_eta = m.cv_eta / SUNMAX(ONE, SUNRabs(m.cv_h) * m.cv_hmax_inv * m.cv_eta);
                m.cv_hprime = m.cv_h * m.cv_eta;
                m.cv_nor = m.cv_nor + 1;
            }
        }
    } else {
        /* Otherwise, let order increase happen, and
        reset stability limit counter, nscon.     */
        cv_mem.borrow_mut().cv_nscon = 0;
    }
}

/*
 * cvSLdet
 *
 * This routine detects stability limitation using stored scaled
 * derivatives data. cvSLdet returns the magnitude of the
 * dominate characteristic root, rr. The presence of a stability
 * limit is indicated by rr > "something a little less then 1.0",
 * and a positive kflag. This routine should only be called if
 * order is greater than or equal to 3, and data has been collected
 * for 5 time steps.
 *
 * Returned values:
 *    kflag = 1 -> Found stable characteristic root, normal matrix case
 *    kflag = 2 -> Found stable characteristic root, quartic solution
 *    kflag = 3 -> Found stable characteristic root, quartic solution,
 *                 with Newton correction
 *    kflag = 4 -> Found stability violation, normal matrix case
 *    kflag = 5 -> Found stability violation, quartic solution
 *    kflag = 6 -> Found stability violation, quartic solution,
 *                 with Newton correction
 *
 *    kflag < 0 -> No stability limitation,
 *                 or could not compute limitation.
 *
 *    kflag = -1 -> Min/max ratio of ssdat too small.
 *    kflag = -2 -> For normal matrix case, vmax > vrrt2*vrrt2
 *    kflag = -3 -> For normal matrix case, The three ratios
 *                  are inconsistent.
 *    kflag = -4 -> Small coefficient prevents elimination of quartics.
 *    kflag = -5 -> R value from quartics not consistent.
 *    kflag = -6 -> No corrected root passes test on qk values
 *    kflag = -7 -> Trouble solving for sigsq.
 *    kflag = -8 -> Trouble solving for B, or R via B.
 *    kflag = -9 -> R via sigsq[k] disagrees with R from data.
 */

fn cvSLdet(cv_mem: &CVodeMem) -> i32 {
    let mut kmin: usize = 0;
    let mut kflag: i32 = 0;
    let mut rat = [[ZERO; 4]; 5];
    let mut rav = [ZERO; 4];
    let mut qkr = [ZERO; 4];
    let mut sigsq = [ZERO; 4];
    let mut smax = [ZERO; 4];
    let mut ssmax = [ZERO; 4];
    let mut drr = [ZERO; 4];
    let mut rrc = [ZERO; 4];
    let mut sqmx = [ZERO; 4];
    let mut qjk = [[ZERO; 4]; 4];
    let mut vrat = [ZERO; 5];
    let mut qc = [[ZERO; 4]; 6];
    let mut qco = [[ZERO; 4]; 6];

    /* Copy the scaled-derivative data and current order out of the mem
    (cvSLdet is pure computation on this snapshot) */
    let (ssdat, q) = {
        let m = cv_mem.borrow();
        (m.cv_ssdat, m.cv_q)
    };

    /* The following are cutoffs and tolerances used by this routine */

    let rrcut: sunrealtype = 0.98;
    let vrrtol: sunrealtype = 1.0e-4;
    let vrrt2: sunrealtype = 5.0e-4;
    let sqtol: sunrealtype = 1.0e-3;
    let rrtol: sunrealtype = 1.0e-2;

    /* (C initializes rr = ZERO here; every branch below assigns rr before
    it is read, so the dead store is omitted.) */
    let mut rr;

    /*  Index k corresponds to the degree of the interpolating polynomial. */
    /*      k = 1 -> q-1          */
    /*      k = 2 -> q            */
    /*      k = 3 -> q+1          */

    /*  Index i is a backward-in-time index, i = 1 -> current time, */
    /*      i = 2 -> previous step, etc    */

    /* get maxima, minima, and variances, and form quartic coefficients  */

    for k in 1..=3usize {
        let mut smink = ssdat[1][k];
        let mut smaxk = ZERO;

        for i in 1..=5usize {
            smink = SUNMIN(smink, ssdat[i][k]);
            smaxk = SUNMAX(smaxk, ssdat[i][k]);
        }

        if smink < TINY * smaxk {
            kflag = -1;
            return kflag;
        }
        smax[k] = smaxk;
        ssmax[k] = smaxk * smaxk;

        let mut sumrat = ZERO;
        let mut sumrsq = ZERO;
        for i in 1..=4usize {
            rat[i][k] = ssdat[i][k] / ssdat[i + 1][k];
            sumrat = sumrat + rat[i][k];
            sumrsq = sumrsq + rat[i][k] * rat[i][k];
        }
        rav[k] = FOURTH * sumrat;
        vrat[k] = SUNRabs(FOURTH * sumrsq - rav[k] * rav[k]);

        qc[5][k] = ssdat[1][k] * ssdat[3][k] - ssdat[2][k] * ssdat[2][k];
        qc[4][k] = ssdat[2][k] * ssdat[3][k] - ssdat[1][k] * ssdat[4][k];
        qc[3][k] = ZERO;
        qc[2][k] = ssdat[2][k] * ssdat[5][k] - ssdat[3][k] * ssdat[4][k];
        qc[1][k] = ssdat[4][k] * ssdat[4][k] - ssdat[3][k] * ssdat[5][k];

        for i in 1..=5usize {
            qco[i][k] = qc[i][k];
        }
    } /* End of k loop */

    /* Isolate normal or nearly-normal matrix case. The three quartics will
    have a common or nearly-common root in this case.
    Return a kflag = 1 if this procedure works. If the three roots
    differ more than vrrt2, return error kflag = -3.    */

    let vmin = SUNMIN(vrat[1], SUNMIN(vrat[2], vrat[3]));
    let vmax = SUNMAX(vrat[1], SUNMAX(vrat[2], vrat[3]));

    if vmin < vrrtol * vrrtol {
        if vmax > vrrt2 * vrrt2 {
            kflag = -2;
            return kflag;
        } else {
            rr = (rav[1] + rav[2] + rav[3]) / THREE;
            let mut drrmax = ZERO;
            for k in 1..=3usize {
                let adrr = SUNRabs(rav[k] - rr);
                drrmax = SUNMAX(drrmax, adrr);
            }
            if drrmax > vrrt2 {
                kflag = -3;
                return kflag;
            }

            kflag = 1;

            /*  can compute charactistic root, drop to next section   */
        }
    } else {
        /* use the quartics to get rr. */

        if SUNRabs(qco[1][1]) < TINY * ssmax[1] {
            kflag = -4;
            return kflag;
        }

        let mut tem = qco[1][2] / qco[1][1];
        for i in 2..=5usize {
            qco[i][2] = qco[i][2] - tem * qco[i][1];
        }

        qco[1][2] = ZERO;
        tem = qco[1][3] / qco[1][1];
        for i in 2..=5usize {
            qco[i][3] = qco[i][3] - tem * qco[i][1];
        }
        qco[1][3] = ZERO;

        if SUNRabs(qco[2][2]) < TINY * ssmax[2] {
            kflag = -4;
            return kflag;
        }

        tem = qco[2][3] / qco[2][2];
        for i in 3..=5usize {
            qco[i][3] = qco[i][3] - tem * qco[i][2];
        }

        if SUNRabs(qco[4][3]) < TINY * ssmax[3] {
            kflag = -4;
            return kflag;
        }

        rr = -qco[5][3] / qco[4][3];

        if rr < TINY || rr > HUNDRED {
            kflag = -5;
            return kflag;
        }

        for k in 1..=3usize {
            qkr[k] = qc[5][k] + rr * (qc[4][k] + rr * rr * (qc[2][k] + rr * qc[1][k]));
        }

        let mut sqmax = ZERO;
        for k in 1..=3usize {
            let saqk = SUNRabs(qkr[k]) / ssmax[k];
            if saqk > sqmax {
                sqmax = saqk;
            }
        }

        if sqmax < sqtol {
            kflag = 2;

            /*  can compute charactistic root, drop to "given rr,etc"   */
        } else {
            /* do Newton corrections to improve rr.  */

            let mut sqmin = ZERO;
            for _it in 1..=3 {
                for k in 1..=3usize {
                    let qp = qc[4][k] + rr * rr * (THREE * qc[2][k] + rr * FOUR * qc[1][k]);
                    drr[k] = ZERO;
                    if SUNRabs(qp) > TINY * ssmax[k] {
                        drr[k] = -qkr[k] / qp;
                    }
                    rrc[k] = rr + drr[k];
                }

                for k in 1..=3usize {
                    let s = rrc[k];
                    let mut sqmaxk = ZERO;
                    for j in 1..=3usize {
                        qjk[j][k] = qc[5][j] + s * (qc[4][j] + s * s * (qc[2][j] + s * qc[1][j]));
                        let saqj = SUNRabs(qjk[j][k]) / ssmax[j];
                        if saqj > sqmaxk {
                            sqmaxk = saqj;
                        }
                    }
                    sqmx[k] = sqmaxk;
                }

                sqmin = sqmx[1] + ONE;
                for k in 1..=3usize {
                    if sqmx[k] < sqmin {
                        kmin = k;
                        sqmin = sqmx[k];
                    }
                }
                rr = rrc[kmin];

                if sqmin < sqtol {
                    kflag = 3;
                    /*  can compute charactistic root   */
                    /*  break out of Newton correction loop and drop to "given rr,etc" */
                    break;
                } else {
                    for j in 1..=3usize {
                        qkr[j] = qjk[j][kmin];
                    }
                }
            } /*  end of Newton correction loop  */

            if sqmin > sqtol {
                kflag = -6;
                return kflag;
            }
        } /*  end of if (sqmax < sqtol) else   */
    } /*  end of if (vmin < vrrtol*vrrtol) else, quartics to get rr. */

    /* given rr, find sigsq[k] and verify rr.  */
    /* All positive kflag drop to this section  */

    for k in 1..=3usize {
        let rsa = ssdat[1][k];
        let rsb = ssdat[2][k] * rr;
        let rsc = ssdat[3][k] * rr * rr;
        let rsd = ssdat[4][k] * rr * rr * rr;
        let rd1a = rsa - rsb;
        let rd1b = rsb - rsc;
        let rd1c = rsc - rsd;
        let rd2a = rd1a - rd1b;
        let rd2b = rd1b - rd1c;
        let rd3a = rd2a - rd2b;

        if SUNRabs(rd1b) < TINY * smax[k] {
            kflag = -7;
            return kflag;
        }

        let cest1 = -rd3a / rd1b;
        if cest1 < TINY || cest1 > FOUR {
            kflag = -7;
            return kflag;
        }
        let corr1 = (rd2b / cest1) / (rr * rr);
        sigsq[k] = ssdat[3][k] + corr1;
    }

    if sigsq[2] < TINY {
        kflag = -8;
        return kflag;
    }

    let ratp = sigsq[3] / sigsq[2];
    let ratm = sigsq[1] / sigsq[2];
    let qfac1 = FOURTH * ((q * q) as sunrealtype - ONE);
    let qfac2 = TWO / (q as sunrealtype - ONE);
    let bb = ratp * ratm - ONE - qfac1 * ratp;
    let tem = ONE - qfac2 * bb;

    if SUNRabs(tem) < TINY {
        kflag = -8;
        return kflag;
    }

    let rrb = ONE / tem;

    if SUNRabs(rrb - rr) > rrtol {
        kflag = -9;
        return kflag;
    }

    /* Check to see if rr is above cutoff rrcut  */
    if rr > rrcut {
        if kflag == 1 {
            kflag = 4;
        }
        if kflag == 2 {
            kflag = 5;
        }
        if kflag == 3 {
            kflag = 6;
        }
    }

    /* All positive kflag returned at this point  */

    kflag
}

/*
 * -----------------------------------------------------------------
 * Functions for rootfinding
 * -----------------------------------------------------------------
 */

/*
 * cvRcheck1
 *
 * This routine completes the initialization of rootfinding memory
 * information, and checks whether g has a zero both at and very near
 * the initial point of the IVP.
 *
 * This routine returns an int equal to:
 *  CV_RTFUNC_FAIL < 0 if the g function failed, or
 *  CV_SUCCESS     = 0 otherwise.
 */

fn cvRcheck1(cv_mem: &CVodeMem) -> i32 {
    {
        let mut m = cv_mem.borrow_mut();
        for i in 0..m.cv_nrtfn as usize {
            m.cv_iroots[i] = 0;
        }
        m.cv_tlo = m.cv_tn;
        m.cv_ttol = (SUNRabs(m.cv_tn) + SUNRabs(m.cv_h)) * m.cv_uround * HUNDRED;
    }

    /* Evaluate g at initial t and check for zero values. */
    let (tlo, zn0) = {
        let m = cv_mem.borrow();
        (m.cv_tlo, m.cv_zn[0].clone().unwrap())
    };
    let mut glo = std::mem::take(&mut cv_mem.borrow_mut().cv_glo);
    let retval = cv_call_gfun(cv_mem, tlo, &zn0, &mut glo);
    {
        let mut m = cv_mem.borrow_mut();
        m.cv_glo = glo;
        m.cv_nge = 1;
    }
    if retval != 0 {
        return CV_RTFUNC_FAIL;
    }

    let mut zroot = SUNFALSE;
    {
        let mut m = cv_mem.borrow_mut();
        for i in 0..m.cv_nrtfn as usize {
            if SUNRabs(m.cv_glo[i]) == ZERO {
                zroot = SUNTRUE;
                m.cv_gactive[i] = SUNFALSE;
            }
        }
    }
    if !zroot {
        return CV_SUCCESS;
    }

    /* Some g_i is zero at t0; look at g at t0+(small increment). */
    let (hratio, tplus, zn0, zn1, y) = {
        let m = cv_mem.borrow();
        let hratio = SUNMAX(m.cv_ttol / SUNRabs(m.cv_h), PT1);
        let smallh = hratio * m.cv_h;
        let tplus = m.cv_tlo + smallh;
        (
            hratio,
            tplus,
            m.cv_zn[0].clone().unwrap(),
            m.cv_zn[1].clone().unwrap(),
            m.cv_y.clone().unwrap(),
        )
    };
    N_VLinearSum(ONE, &zn0, hratio, &zn1, &y);
    let mut ghi = std::mem::take(&mut cv_mem.borrow_mut().cv_ghi);
    let retval = cv_call_gfun(cv_mem, tplus, &y, &mut ghi);
    {
        let mut m = cv_mem.borrow_mut();
        m.cv_ghi = ghi;
        m.cv_nge += 1;
    }
    if retval != 0 {
        return CV_RTFUNC_FAIL;
    }

    /* We check now only the components of g which were exactly 0.0 at t0
     * to see if we can 'activate' them. */
    {
        let mut m = cv_mem.borrow_mut();
        for i in 0..m.cv_nrtfn as usize {
            if !m.cv_gactive[i] && SUNRabs(m.cv_ghi[i]) != ZERO {
                m.cv_gactive[i] = SUNTRUE;
                m.cv_glo[i] = m.cv_ghi[i];
            }
        }
    }
    CV_SUCCESS
}

/*
 * cvRcheck2
 *
 * This routine checks for exact zeros of g at the last root found,
 * if the last return was a root.  It then checks for a close pair of
 * zeros (an error condition), and for a new root at a nearby point.
 * The array glo = g(tlo) at the left endpoint of the search interval
 * is adjusted if necessary to assure that all g_i are nonzero
 * there, before returning to do a root search in the interval.
 *
 * On entry, tlo = tretlast is the last value of tret returned by
 * CVode.  This may be the previous tn, the previous tout value,
 * or the last root location.
 *
 * This routine returns an int equal to:
 *     CV_RTFUNC_FAIL  < 0 if the g function failed, or
 *     CLOSERT         = 3 if a close pair of zeros was found, or
 *     RTFOUND         = 1 if a new zero of g was found near tlo, or
 *     CV_SUCCESS      = 0 otherwise.
 */

fn cvRcheck2(cv_mem: &CVodeMem) -> i32 {
    if cv_mem.borrow().cv_irfnd == 0 {
        return CV_SUCCESS;
    }

    let (tlo, y) = {
        let m = cv_mem.borrow();
        (m.cv_tlo, m.cv_y.clone().unwrap())
    };
    let _ = CVodeGetDky(cv_mem, tlo, 0, &y);
    let mut glo = std::mem::take(&mut cv_mem.borrow_mut().cv_glo);
    let retval = cv_call_gfun(cv_mem, tlo, &y, &mut glo);
    {
        let mut m = cv_mem.borrow_mut();
        m.cv_glo = glo;
        m.cv_nge += 1;
    }
    if retval != 0 {
        return CV_RTFUNC_FAIL;
    }

    let mut zroot = SUNFALSE;
    {
        let mut m = cv_mem.borrow_mut();
        for i in 0..m.cv_nrtfn as usize {
            m.cv_iroots[i] = 0;
        }
        for i in 0..m.cv_nrtfn as usize {
            if !m.cv_gactive[i] {
                continue;
            }
            if SUNRabs(m.cv_glo[i]) == ZERO {
                zroot = SUNTRUE;
                m.cv_iroots[i] = 1;
            }
        }
    }
    if !zroot {
        return CV_SUCCESS;
    }

    /* One or more g_i has a zero at tlo.  Check g at tlo+smallh. */
    let (smallh, tplus, beyond_tn) = {
        let mut m = cv_mem.borrow_mut();
        m.cv_ttol = (SUNRabs(m.cv_tn) + SUNRabs(m.cv_h)) * m.cv_uround * HUNDRED;
        let smallh = if m.cv_h > ZERO { m.cv_ttol } else { -m.cv_ttol };
        let tplus = m.cv_tlo + smallh;
        (smallh, tplus, (tplus - m.cv_tn) * m.cv_h >= ZERO)
    };
    if beyond_tn {
        let (hratio, zn1) = {
            let m = cv_mem.borrow();
            (smallh / m.cv_h, m.cv_zn[1].clone().unwrap())
        };
        N_VLinearSum(ONE, &y, hratio, &zn1, &y);
    } else {
        let _ = CVodeGetDky(cv_mem, tplus, 0, &y);
    }
    let mut ghi = std::mem::take(&mut cv_mem.borrow_mut().cv_ghi);
    let retval = cv_call_gfun(cv_mem, tplus, &y, &mut ghi);
    {
        let mut m = cv_mem.borrow_mut();
        m.cv_ghi = ghi;
        m.cv_nge += 1;
    }
    if retval != 0 {
        return CV_RTFUNC_FAIL;
    }

    /* Check for close roots (error return), for a new zero at tlo+smallh,
    and for a g_i that changed from zero to nonzero. */
    zroot = SUNFALSE;
    {
        let mut m = cv_mem.borrow_mut();
        for i in 0..m.cv_nrtfn as usize {
            if !m.cv_gactive[i] {
                continue;
            }
            if SUNRabs(m.cv_ghi[i]) == ZERO {
                if m.cv_iroots[i] == 1 {
                    return CLOSERT;
                }
                zroot = SUNTRUE;
                m.cv_iroots[i] = 1;
            } else {
                if m.cv_iroots[i] == 1 {
                    m.cv_glo[i] = m.cv_ghi[i];
                }
            }
        }
    }
    if zroot {
        return RTFOUND;
    }
    CV_SUCCESS
}

/*
 * cvRcheck3
 *
 * This routine interfaces to cvRootfind to look for a root of g
 * between tlo and either tn or tout, whichever comes first.
 * Only roots beyond tlo in the direction of integration are sought.
 *
 * This routine returns an int equal to:
 *     CV_RTFUNC_FAIL  < 0 if the g function failed, or
 *     RTFOUND         = 1 if a root of g was found, or
 *     CV_SUCCESS      = 0 otherwise.
 */

fn cvRcheck3(cv_mem: &CVodeMem, tout: sunrealtype, itask: i32) -> i32 {
    /* Set thi = tn or tout, whichever comes first; set y = y(thi). */
    if itask == CV_ONE_STEP {
        let (zn0, y) = {
            let mut m = cv_mem.borrow_mut();
            m.cv_thi = m.cv_tn;
            (m.cv_zn[0].clone().unwrap(), m.cv_y.clone().unwrap())
        };
        N_VScale(ONE, &zn0, &y);
    }
    if itask == CV_NORMAL {
        let beyond_tn = {
            let m = cv_mem.borrow();
            (tout - m.cv_tn) * m.cv_h >= ZERO
        };
        if beyond_tn {
            let (zn0, y) = {
                let mut m = cv_mem.borrow_mut();
                m.cv_thi = m.cv_tn;
                (m.cv_zn[0].clone().unwrap(), m.cv_y.clone().unwrap())
            };
            N_VScale(ONE, &zn0, &y);
        } else {
            let (thi, y) = {
                let mut m = cv_mem.borrow_mut();
                m.cv_thi = tout;
                (m.cv_thi, m.cv_y.clone().unwrap())
            };
            let _ = CVodeGetDky(cv_mem, thi, 0, &y);
        }
    }

    /* Set ghi = g(thi) and call cvRootfind to search (tlo,thi) for roots. */
    let (thi, y) = {
        let m = cv_mem.borrow();
        (m.cv_thi, m.cv_y.clone().unwrap())
    };
    let mut ghi = std::mem::take(&mut cv_mem.borrow_mut().cv_ghi);
    let retval = cv_call_gfun(cv_mem, thi, &y, &mut ghi);
    {
        let mut m = cv_mem.borrow_mut();
        m.cv_ghi = ghi;
        m.cv_nge += 1;
    }
    if retval != 0 {
        return CV_RTFUNC_FAIL;
    }

    {
        let mut m = cv_mem.borrow_mut();
        m.cv_ttol = (SUNRabs(m.cv_tn) + SUNRabs(m.cv_h)) * m.cv_uround * HUNDRED;
    }
    let ier = cvRootfind(cv_mem);
    if ier == CV_RTFUNC_FAIL {
        return CV_RTFUNC_FAIL;
    }
    {
        let mut m = cv_mem.borrow_mut();
        for i in 0..m.cv_nrtfn as usize {
            if !m.cv_gactive[i] && m.cv_grout[i] != ZERO {
                m.cv_gactive[i] = SUNTRUE;
            }
        }
        m.cv_tlo = m.cv_trout;
        for i in 0..m.cv_nrtfn as usize {
            m.cv_glo[i] = m.cv_grout[i];
        }
    }

    /* If no root found, return CV_SUCCESS. */
    if ier == CV_SUCCESS {
        return CV_SUCCESS;
    }

    /* If a root was found, interpolate to get y(trout) and return.  */
    let (trout, y) = {
        let m = cv_mem.borrow();
        (m.cv_trout, m.cv_y.clone().unwrap())
    };
    let _ = CVodeGetDky(cv_mem, trout, 0, &y);
    RTFOUND
}

/*
 * cvRootfind
 *
 * This routine solves for a root of g(t) between tlo and thi, if
 * one exists.  Only roots of odd multiplicity (i.e. with a change
 * of sign in one of the g_i), or exact zeros, are found.
 * Here the sign of tlo - thi is arbitrary, but if multiple roots
 * are found, the one closest to tlo is returned.
 *
 * The method used is the Illinois algorithm, a modified secant method.
 * Reference: Kathie L. Hiebert and Lawrence F. Shampine, Implicitly
 * Defined Output Points for Solutions of ODEs, Sandia National
 * Laboratory Report SAND80-0180, February 1980.
 *
 * This routine uses the following parameters for communication:
 *
 * nrtfn    = number of functions g_i, or number of components of
 *            the vector-valued function g(t).  Input only.
 *
 * gfun     = user-defined function for g(t).  Its form is
 *            (void) gfun(t, y, gt, user_data)
 *
 * rootdir  = in array specifying the direction of zero-crossings.
 *            If rootdir[i] > 0, search for roots of g_i only if
 *            g_i is increasing; if rootdir[i] < 0, search for
 *            roots of g_i only if g_i is decreasing; otherwise
 *            always search for roots of g_i.
 *
 * gactive  = array specifying whether a component of g should
 *            or should not be monitored. gactive[i] is initially
 *            set to SUNTRUE for all i=0,...,nrtfn-1, but it may be
 *            reset to SUNFALSE if at the first step g[i] is 0.0
 *            both at the I.C. and at a small perturbation of them.
 *            gactive[i] is then set back on SUNTRUE only after the
 *            corresponding g function moves away from 0.0.
 *
 * nge      = cumulative counter for gfun calls.
 *
 * ttol     = a convergence tolerance for trout.  Input only.
 *            When a root at trout is found, it is located only to
 *            within a tolerance of ttol.  Typically, ttol should
 *            be set to a value on the order of
 *               100 * UROUND * max (SUNRabs(tlo), SUNRabs(thi))
 *            where UROUND is the unit roundoff of the machine.
 *
 * tlo, thi = endpoints of the interval in which roots are sought.
 *            On input, these must be distinct, but tlo - thi may
 *            be of either sign.  The direction of integration is
 *            assumed to be from tlo to thi.  On return, tlo and thi
 *            are the endpoints of the final relevant interval.
 *
 * glo, ghi = arrays of length nrtfn containing the vectors g(tlo)
 *            and g(thi) respectively.  Input and output.  On input,
 *            none of the glo[i] should be zero.
 *
 * trout    = root location, if a root was found, or thi if not.
 *            Output only.  If a root was found other than an exact
 *            zero of g, trout is the endpoint thi of the final
 *            interval bracketing the root, with size at most ttol.
 *
 * grout    = array of length nrtfn containing g(trout) on return.
 *
 * iroots   = int array of length nrtfn with root information.
 *            Output only.  If a root was found, iroots indicates
 *            which components g_i have a root at trout.  For
 *            i = 0, ..., nrtfn-1, iroots[i] = 1 if g_i has a root
 *            and g_i is increasing, iroots[i] = -1 if g_i has a
 *            root and g_i is decreasing, and iroots[i] = 0 if g_i
 *            has no roots or g_i varies in the direction opposite
 *            to that indicated by rootdir[i].
 *
 * This routine returns an int equal to:
 *      CV_RTFUNC_FAIL  < 0 if the g function failed, or
 *      RTFOUND         = 1 if a root of g was found, or
 *      CV_SUCCESS      = 0 otherwise.
 */

fn cvRootfind(cv_mem: &CVodeMem) -> i32 {
    /* Move the rootfinding state into locals for the duration of the search
    (the user's g function is invoked inside the loop; no RefCell borrow may
    be held across it). C writes through the cv_mem fields on every return
    path; the single write-back below restores the identical state for each
    path (on the CV_RTFUNC_FAIL path the fields hold the values from the
    last completed iteration, exactly as in C). */
    let (nrtfn, ttol, y) = {
        let m = cv_mem.borrow();
        (m.cv_nrtfn as usize, m.cv_ttol, m.cv_y.clone().unwrap())
    };
    let (mut tlo, mut thi, mut trout) = {
        let m = cv_mem.borrow();
        (m.cv_tlo, m.cv_thi, m.cv_trout)
    };
    let (mut glo, mut ghi, mut grout, mut iroots, rootdir, gactive) = {
        let mut m = cv_mem.borrow_mut();
        (
            std::mem::take(&mut m.cv_glo),
            std::mem::take(&mut m.cv_ghi),
            std::mem::take(&mut m.cv_grout),
            std::mem::take(&mut m.cv_iroots),
            std::mem::take(&mut m.cv_rootdir),
            std::mem::take(&mut m.cv_gactive),
        )
    };

    let retflag = {
        let mut search = || -> i32 {
            let mut imax: usize = 0;

            /* First check for change in sign in ghi or for a zero in ghi. */
            let mut maxfrac = ZERO;
            let mut zroot = SUNFALSE;
            let mut sgnchg = SUNFALSE;
            for i in 0..nrtfn {
                if !gactive[i] {
                    continue;
                }
                if SUNRabs(ghi[i]) == ZERO {
                    if rootdir[i] as sunrealtype * glo[i] <= ZERO {
                        zroot = SUNTRUE;
                    }
                } else {
                    if SUNRdifferentsign(glo[i], ghi[i])
                        && (rootdir[i] as sunrealtype * glo[i] <= ZERO)
                    {
                        let gfrac = SUNRabs(ghi[i] / (ghi[i] - glo[i]));
                        if gfrac > maxfrac {
                            sgnchg = SUNTRUE;
                            maxfrac = gfrac;
                            imax = i;
                        }
                    }
                }
            }

            /* If no sign change was found, reset trout and grout.  Then return
            CV_SUCCESS if no zero was found, or set iroots and return RTFOUND.  */
            if !sgnchg {
                trout = thi;
                for i in 0..nrtfn {
                    grout[i] = ghi[i];
                }
                if !zroot {
                    return CV_SUCCESS;
                }
                for i in 0..nrtfn {
                    iroots[i] = 0;
                    if !gactive[i] {
                        continue;
                    }
                    if (SUNRabs(ghi[i]) == ZERO) && (rootdir[i] as sunrealtype * glo[i] <= ZERO) {
                        iroots[i] = if glo[i] > ZERO { -1 } else { 1 };
                    }
                }
                return RTFOUND;
            }

            /* Initialize alph to avoid compiler warning */
            let mut alph = ONE;

            /* A sign change was found.  Loop to locate nearest root. */

            let mut side = 0;
            let mut sideprev = -1;
            loop {
                /* Looping point */

                /* If interval size is already less than tolerance ttol, break. */
                if SUNRabs(thi - tlo) <= ttol {
                    break;
                }

                /* Set weight alph.
                On the first two passes, set alph = 1.  Thereafter, reset alph
                according to the side (low vs high) of the subinterval in which
                the sign change was found in the previous two passes.
                If the sides were opposite, set alph = 1.
                If the sides were the same, then double alph (if high side),
                or halve alph (if low side).
                The next guess tmid is the secant method value if alph = 1, but
                is closer to tlo if alph < 1, and closer to thi if alph > 1.    */

                if sideprev == side {
                    alph = if side == 2 { alph * TWO } else { alph * HALF };
                } else {
                    alph = ONE;
                }

                /* Set next root approximation tmid and get g(tmid).
                If tmid is too close to tlo or thi, adjust it inward,
                by a fractional distance that is between 0.1 and 0.5.  */
                let mut tmid = thi - (thi - tlo) * ghi[imax] / (ghi[imax] - alph * glo[imax]);
                if SUNRabs(tmid - tlo) < HALF * ttol {
                    let fracint = SUNRabs(thi - tlo) / ttol;
                    let fracsub = if fracint > FIVE { PT1 } else { HALF / fracint };
                    tmid = tlo + fracsub * (thi - tlo);
                }
                if SUNRabs(thi - tmid) < HALF * ttol {
                    let fracint = SUNRabs(thi - tlo) / ttol;
                    let fracsub = if fracint > FIVE { PT1 } else { HALF / fracint };
                    tmid = thi - fracsub * (thi - tlo);
                }

                let _ = CVodeGetDky(cv_mem, tmid, 0, &y);
                let retval = cv_call_gfun(cv_mem, tmid, &y, &mut grout);
                cv_mem.borrow_mut().cv_nge += 1;
                if retval != 0 {
                    return CV_RTFUNC_FAIL;
                }

                /* Check to see in which subinterval g changes sign, and reset imax.
                Set side = 1 if sign change is on low side, or 2 if on high side.  */
                maxfrac = ZERO;
                zroot = SUNFALSE;
                sgnchg = SUNFALSE;
                sideprev = side;
                for i in 0..nrtfn {
                    if !gactive[i] {
                        continue;
                    }
                    if SUNRabs(grout[i]) == ZERO {
                        if rootdir[i] as sunrealtype * glo[i] <= ZERO {
                            zroot = SUNTRUE;
                        }
                    } else {
                        if SUNRdifferentsign(glo[i], grout[i])
                            && (rootdir[i] as sunrealtype * glo[i] <= ZERO)
                        {
                            let gfrac = SUNRabs(grout[i] / (grout[i] - glo[i]));
                            if gfrac > maxfrac {
                                sgnchg = SUNTRUE;
                                maxfrac = gfrac;
                                imax = i;
                            }
                        }
                    }
                }
                if sgnchg {
                    /* Sign change found in (tlo,tmid); replace thi with tmid. */
                    thi = tmid;
                    for i in 0..nrtfn {
                        ghi[i] = grout[i];
                    }
                    side = 1;
                    /* Stop at root thi if converged; otherwise loop. */
                    if SUNRabs(thi - tlo) <= ttol {
                        break;
                    }
                    continue; /* Return to looping point. */
                }

                if zroot {
                    /* No sign change in (tlo,tmid), but g = 0 at tmid; return root tmid. */
                    thi = tmid;
                    for i in 0..nrtfn {
                        ghi[i] = grout[i];
                    }
                    break;
                }

                /* No sign change in (tlo,tmid), and no zero at tmid.
                Sign change must be in (tmid,thi).  Replace tlo with tmid. */
                tlo = tmid;
                for i in 0..nrtfn {
                    glo[i] = grout[i];
                }
                side = 2;
                /* Stop at root thi if converged; otherwise loop back. */
                if SUNRabs(thi - tlo) <= ttol {
                    break;
                }
            } /* End of root-search loop */

            /* Reset trout and grout, set iroots, and return RTFOUND. */
            trout = thi;
            for i in 0..nrtfn {
                grout[i] = ghi[i];
                iroots[i] = 0;
                if !gactive[i] {
                    continue;
                }
                if (SUNRabs(ghi[i]) == ZERO) && (rootdir[i] as sunrealtype * glo[i] <= ZERO) {
                    iroots[i] = if glo[i] > ZERO { -1 } else { 1 };
                }
                if SUNRdifferentsign(glo[i], ghi[i]) && (rootdir[i] as sunrealtype * glo[i] <= ZERO)
                {
                    iroots[i] = if glo[i] > ZERO { -1 } else { 1 };
                }
            }
            RTFOUND
        };
        search()
    };

    /* Write the rootfinding state back into the mem (single exit point) */
    {
        let mut m = cv_mem.borrow_mut();
        m.cv_tlo = tlo;
        m.cv_thi = thi;
        m.cv_trout = trout;
        m.cv_glo = glo;
        m.cv_ghi = ghi;
        m.cv_grout = grout;
        m.cv_iroots = iroots;
        m.cv_rootdir = rootdir;
        m.cv_gactive = gactive;
    }

    retflag
}

/*
 * =================================================================
 * Internal EWT function
 * =================================================================
 */

/*
 * cvEwtSet
 *
 * This routine is responsible for setting the error weight vector ewt,
 * according to tol_type, as follows:
 *
 * (1) ewt[i] = 1 / (reltol * SUNRabs(ycur[i]) + abstol), i=0,...,neq-1
 *     if tol_type = CV_SS
 * (2) ewt[i] = 1 / (reltol * SUNRabs(ycur[i]) + abstol[i]), i=0,...,neq-1
 *     if tol_type = CV_SV
 *
 * cvEwtSet returns 0 if ewt is successfully set as above to a
 * positive vector and -1 otherwise. In the latter case, ewt is
 * considered undefined.
 *
 * All the real work is done in the routines cvEwtSetSS, cvEwtSetSV.
 */

pub fn cvEwtSet(ycur: &N_Vector, weight: &N_Vector, data: &mut Option<Box<dyn Any>>) -> i32 {
    /* data points to cv_mem here (a boxed CVodeMem handle clone; C's cast
    of a NULL/foreign pointer is UB -> deterministic panic) */
    let cv_mem = data
        .as_mut()
        .and_then(|b| b.downcast_ref::<CVodeMem>())
        .cloned()
        .expect("cvEwtSet data holds CVodeMem");

    let itol = cv_mem.borrow().cv_itol;
    let flag: i32 = match itol {
        CV_SS => cvEwtSetSS(&cv_mem, ycur, weight),
        CV_SV => cvEwtSetSV(&cv_mem, ycur, weight),
        _ => 0,
    };

    flag
}

/*
 * cvEwtSetSS
 *
 * This routine sets ewt as described above in the case tol_type = CV_SS.
 * If the absolute tolerance is zero, it tests for non-positive components
 * before inverting. cvEwtSetSS returns 0 if ewt is successfully set to a
 * positive vector and -1 otherwise. In the latter case, ewt is considered
 * undefined.
 */

fn cvEwtSetSS(cv_mem: &CVodeMem, ycur: &N_Vector, weight: &N_Vector) -> i32 {
    /* SUNDIALS_ENABLE_PACKAGE_FUSED_KERNELS not defined: unfused branch */
    let (tempv, reltol, Sabstol, atolmin0) = {
        let m = cv_mem.borrow();
        (
            m.cv_tempv.clone().unwrap(),
            m.cv_reltol,
            m.cv_Sabstol,
            m.cv_atolmin0,
        )
    };
    N_VAbs(ycur, &tempv);
    N_VScale(reltol, &tempv, &tempv);
    N_VAddConst(&tempv, Sabstol, &tempv);
    if atolmin0 {
        if N_VMin(&tempv) <= ZERO {
            return -1;
        }
    }
    N_VInv(&tempv, weight);

    0
}

/*
 * cvEwtSetSV
 *
 * This routine sets ewt as described above in the case tol_type = CV_SV.
 * If any absolute tolerance is zero, it tests for non-positive components
 * before inverting. cvEwtSetSV returns 0 if ewt is successfully set to a
 * positive vector and -1 otherwise. In the latter case, ewt is considered
 * undefined.
 */

fn cvEwtSetSV(cv_mem: &CVodeMem, ycur: &N_Vector, weight: &N_Vector) -> i32 {
    /* SUNDIALS_ENABLE_PACKAGE_FUSED_KERNELS not defined: unfused branch */
    let (tempv, reltol, Vabstol, atolmin0) = {
        let m = cv_mem.borrow();
        (
            m.cv_tempv.clone().unwrap(),
            m.cv_reltol,
            m.cv_Vabstol.clone().unwrap(),
            m.cv_atolmin0,
        )
    };
    N_VAbs(ycur, &tempv);
    N_VLinearSum(reltol, &tempv, ONE, &Vabstol, &tempv);
    if atolmin0 {
        if N_VMin(&tempv) <= ZERO {
            return -1;
        }
    }
    N_VInv(&tempv, weight);

    0
}
