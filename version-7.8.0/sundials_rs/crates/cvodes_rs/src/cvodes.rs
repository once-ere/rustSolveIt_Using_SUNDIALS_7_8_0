//! Port of `src/cvodes/cvodes.c` (+ the public `include/cvodes/cvodes.h`
//! declarations folded into `cvodes_impl`).
//!
//! Main CVODES integrator: creation/initialization, quadrature,
//! forward-sensitivity and quadrature-sensitivity initialization and
//! tolerance functions, rootfinding initialization, the `CVode` driver,
//! `cvStep` and all its helpers, dense output (`CVodeGetDky` and the
//! quadrature/sensitivity variants), the internal error-weight functions,
//! rootfinding (`cvRcheck1/2/3`, `cvRootfind`), BDF stability limit
//! detection (`cvBDFStab`, `cvSLdet`), the combined norms, the sensitivity
//! RHS wrappers and their internal DQ approximations, and the
//! deallocation functions.
//!
//! `cvProcessError`, the `MSGCV_*`/`MSG_TIME*` message constants and
//! builders, and **every module-scope constant `cvodes.c` defines**
//! (`ZERO` … `HUNDRED`, `RTFOUND`/`CLOSERT`, `CENTERED1/2`,
//! `FORWARD1/2`, `CV_ONESENS`/`CV_ALLSENS`, `CV_NN`/`CV_SS`/`CV_SV`/
//! `CV_WF`/`CV_EE`, `FUZZ_FACTOR`, `HLB_FACTOR`, `HUB_FACTOR`, `H_BIAS`,
//! `MAX_ITERS`, `CORTES`) live in `cvodes_impl` per the frozen
//! fragment-file protocol and reach this module through the
//! `crate::cvodes_impl::*` glob import below. Fragments of this module
//! must NOT redeclare them.
//!
//! Reference build configuration: SUNDIALS_LOGGING_LEVEL = 2 (SUNLogInfo/
//! SUNLogInfoIf/SUNLogDebug/SUNLogExtraDebug* call sites omitted entirely;
//! CV_WARNING paths kept — they print through the logger), profiling off,
//! error checks off (SUNAssert/SUNCheck* are no-ops, SUNCheckCall
//! evaluates and continues), monitoring ON, fused kernels OFF (the
//! unfused branch is the live code), serial branches only.
//!
//! Handle model: `CVodeMem = Rc<RefCell<CVodeMemRec>>`. Internal
//! functions take `&CVodeMem` and use granular borrows — no borrow is
//! ever held across a user callback, an N_Vector op, or a linear/
//! nonlinear-solver call, all of which can re-enter the mem.

use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

use crate::cvodes_impl::*;
use sundials_core::sundials_context::SUNContext;
use sundials_core::sundials_errors::{SUN_ERR_ARG_CORRUPT, SUN_SUCCESS};
use sundials_core::sundials_math::*;
use sundials_core::sundials_nonlinearsolver::*;
use sundials_core::sundials_nvector::*;
use sundials_core::sundials_types::*;
use sundials_core::sunnonlinsol_newton::{SUNNonlinSol_Newton, SUNNonlinSol_NewtonSens};

/*=================================================================*/
/* CVODE Private Constants / Routine-Specific Constants            */
/*                                                                 */
/* ZERO … HUNDRED, RTFOUND/CLOSERT, CENTERED1/2, FORWARD1/2,       */
/* CV_ONESENS/CV_ALLSENS, CV_NN/CV_SS/CV_SV/CV_WF/CV_EE,           */
/* FUZZ_FACTOR, HLB_FACTOR, HUB_FACTOR, H_BIAS, MAX_ITERS and      */
/* CORTES are defined once in `cvodes_impl` (fragment-file          */
/* protocol) and are in scope here through the glob import above.  */
/*=================================================================*/

/*
 * =================================================================
 * Callback invocation helpers (granular borrow discipline: the data
 * token is taken out of the mem around every user callback call and
 * restored on every path; no mem borrow is held across the call).
 *
 * `cv_fS_data` / `cv_fQS_data` follow the C `void*` convention:
 * `Some(box)` is a module-owned token (a `CVodeMem` clone, installed
 * when the internal DQ routines are in use), `None` means "pass the
 * integrator's `cv_user_data`" (C stored the user_data pointer there).
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

/// Invoke the user quadrature RHS function `fQ`
/// (C: `cv_mem->cv_fQ(t, y, yQdot, cv_mem->cv_user_data)`).
fn cv_call_fQ(cv_mem: &CVodeMem, t: sunrealtype, y: &N_Vector, yQdot: &N_Vector) -> i32 {
    let fQ = cv_mem.borrow().cv_fQ.expect("cv_fQ set");
    let mut user_data = cv_mem.borrow_mut().cv_user_data.take();
    let retval = fQ(t, y, yQdot, &mut user_data);
    cv_mem.borrow_mut().cv_user_data = user_data;
    retval
}

/// Invoke the quadrature-sensitivity RHS function `fQS`
/// (C: `cv_mem->cv_fQS(Ns, t, y, yS, yQdot, yQSdot, cv_mem->cv_fQS_data, tmp, tmpQ)`).
#[allow(clippy::too_many_arguments)]
fn cv_call_fQS(
    cv_mem: &CVodeMem,
    Ns: i32,
    t: sunrealtype,
    y: &N_Vector,
    yS: &[N_Vector],
    yQdot: &N_Vector,
    yQSdot: &[N_Vector],
    tmp: &N_Vector,
    tmpQ: &N_Vector,
) -> i32 {
    let (fQS, owned) = {
        let m = cv_mem.borrow();
        (m.cv_fQS.expect("cv_fQS set"), m.cv_fQS_data.is_some())
    };
    if owned {
        let mut data = cv_mem.borrow_mut().cv_fQS_data.take();
        let retval = fQS(Ns, t, y, yS, yQdot, yQSdot, &mut data, tmp, tmpQ);
        cv_mem.borrow_mut().cv_fQS_data = data;
        retval
    } else {
        let mut data = cv_mem.borrow_mut().cv_user_data.take();
        let retval = fQS(Ns, t, y, yS, yQdot, yQSdot, &mut data, tmp, tmpQ);
        cv_mem.borrow_mut().cv_user_data = data;
        retval
    }
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
 * be solved by CVODES.
 * If successful, CVodeCreate returns a pointer to the problem memory.
 * This pointer should be passed to CVodeInit.
 * If an initialization error occurs, CVodeCreate prints an error
 * message to standard err and returns NULL.
 */

pub fn CVodeCreate(lmm: i32, sunctx: &SUNContext) -> Option<CVodeMem> {
    /* Test inputs */

    if (lmm != CV_ADAMS) && (lmm != CV_BDF) {
        cvProcessError(
            None,
            0,
            line!() as i32,
            "CVodeCreate",
            file!(),
            MSGCV_BAD_LMM,
        );
        return None;
    }

    /* NULL sunctx check: handled by type system */

    /* malloc failure branch: allocation cannot fail observably in Rust.
    The C `memset(cv_mem, 0, ...)` maps to `CVodeMemRec::zeroed`. */
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

    /* Set default values for quad. optional inputs */

    cv_mem.cv_quadr = SUNFALSE;
    cv_mem.cv_fQ = None;
    cv_mem.cv_errconQ = SUNFALSE;
    cv_mem.cv_itolQ = CV_NN;
    cv_mem.cv_atolQmin0 = SUNTRUE;

    /* Set default values for sensi. optional inputs */

    cv_mem.cv_sensi = SUNFALSE;
    cv_mem.cv_fS_data = None;
    cv_mem.cv_fS = Some(cvSensRhsInternalDQ);
    cv_mem.cv_fS1 = Some(cvSensRhs1InternalDQ);
    cv_mem.cv_fSDQ = SUNTRUE;
    cv_mem.cv_ifS = CV_ONESENS;
    cv_mem.cv_DQtype = CV_CENTERED;
    cv_mem.cv_DQrhomax = ZERO;
    cv_mem.cv_p = None;
    cv_mem.cv_pbar = Vec::new();
    cv_mem.cv_plist = Vec::new();
    cv_mem.cv_errconS = SUNFALSE;
    cv_mem.cv_ncfS1 = Vec::new();
    cv_mem.cv_ncfnS1 = Vec::new();
    cv_mem.cv_nniS1 = Vec::new();
    cv_mem.cv_nnfS1 = Vec::new();
    cv_mem.cv_itolS = CV_NN;
    cv_mem.cv_atolSmin0 = Vec::new();

    /* Set default values for quad. sensi. optional inputs */

    cv_mem.cv_quadr_sensi = SUNFALSE;
    cv_mem.cv_fQS = None;
    cv_mem.cv_fQS_data = None;
    cv_mem.cv_fQSDQ = SUNTRUE;
    cv_mem.cv_errconQS = SUNFALSE;
    cv_mem.cv_itolQS = CV_NN;
    cv_mem.cv_atolQSmin0 = Vec::new();

    /* Set default for ASA */

    cv_mem.cv_adj = SUNFALSE;
    cv_mem.cv_adj_mem = None;

    /* Set the saved value for qmax_alloc */

    cv_mem.cv_qmax_alloc = maxord;
    cv_mem.cv_qmax_allocQ = maxord;
    cv_mem.cv_qmax_allocS = maxord;

    /* Initialize lrw and liw */

    cv_mem.cv_lrw = (65 + 2 * L_MAX + NUM_TESTS) as i64;
    cv_mem.cv_liw = 52;

    /* No mallocs have been done yet */

    cv_mem.cv_VabstolMallocDone = SUNFALSE;
    cv_mem.cv_MallocDone = SUNFALSE;
    cv_mem.cv_constraintsMallocDone = SUNFALSE;

    cv_mem.cv_VabstolQMallocDone = SUNFALSE;
    cv_mem.cv_QuadMallocDone = SUNFALSE;

    cv_mem.cv_VabstolSMallocDone = SUNFALSE;
    cv_mem.cv_SabstolSMallocDone = SUNFALSE;
    cv_mem.cv_SensMallocDone = SUNFALSE;

    cv_mem.cv_VabstolQSMallocDone = SUNFALSE;
    cv_mem.cv_SabstolQSMallocDone = SUNFALSE;
    cv_mem.cv_QuadSensMallocDone = SUNFALSE;

    cv_mem.cv_adjMallocDone = SUNFALSE;

    /* Initialize nonlinear solver variables */
    cv_mem.NLS = None;
    cv_mem.ownNLS = SUNFALSE;

    cv_mem.NLSsim = None;
    cv_mem.ownNLSsim = SUNFALSE;
    cv_mem.zn0Sim = None;
    cv_mem.ycorSim = None;
    cv_mem.ewtSim = None;
    cv_mem.simMallocDone = SUNFALSE;

    cv_mem.NLSstg = None;
    cv_mem.ownNLSstg = SUNFALSE;
    cv_mem.zn0Stg = None;
    cv_mem.ycorStg = None;
    cv_mem.ewtStg = None;
    cv_mem.stgMallocDone = SUNFALSE;

    cv_mem.NLSstg1 = None;
    cv_mem.ownNLSstg1 = SUNFALSE;

    cv_mem.sens_solve = SUNFALSE;
    cv_mem.sens_solve_idx = -1;

    /* Return pointer to CVODES memory block */

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

    /* Allocate temporary work arrays for fused vector ops. The C code
    mallocs L_MAX slots for cvals/Xvecs/Zvecs; `cv_Xvecs`/`cv_Zvecs` are
    handle scratch that the callers rebuild on demand (an N_Vector array
    cannot be left uninitialized in safe Rust), so only `cv_cvals` is
    materialized. The C NULL-check failure branch is unreachable: Vec
    allocation aborts rather than returning NULL. */
    {
        let mut m = cv_mem.borrow_mut();
        m.cv_cvals = vec![ZERO; L_MAX];
        m.cv_Xvecs = Vec::new();
        m.cv_Zvecs = Vec::new();
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
    let retval = crate::cvodes_nls::CVodeSetNonlinearSolver(cv_mem, &NLS);

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

        /* Set forceSetup to SUNFALSE */

        m.cv_forceSetup = SUNFALSE;

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
 * CVodeReInit re-initializes CVODES's memory for a problem, assuming
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

        /* Copy the input parameters into CVODES state */

        m.cv_tn = t0;

        /* Set step parameters */

        m.cv_q = 1;
        m.cv_L = 2;
        m.cv_qwait = m.cv_L;
        m.cv_etamax = m.cv_eta_max_fs;

        m.cv_qu = 0;
        m.cv_hu = ZERO;
        m.cv_tolsf = ONE;

        /* Set forceSetup to SUNFALSE */

        m.cv_forceSetup = SUNFALSE;
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
 * CVodeQuadInit
 *
 * CVodeQuadInit allocates and initializes quadrature related
 * memory for a problem. All problem specification inputs are
 * checked for errors. If any error occurs during initialization,
 * it is reported to the file whose file pointer is errfp.
 * The return value is CV_SUCCESS = 0 if no errors occurred, or
 * a negative value otherwise.
 */

pub fn CVodeQuadInit(cvode_mem: &CVodeMem, fQ: CVQuadRhsFn, yQ0: &N_Vector) -> i32 {
    /* Check cvode_mem: NULL-mem check handled by type system */
    let cv_mem = cvode_mem;

    /* Set space requirements for one N_Vector */
    let mut lrw1Q: sunindextype = 0;
    let mut liw1Q: sunindextype = 0;
    N_VSpace(yQ0, &mut lrw1Q, &mut liw1Q);
    {
        let mut m = cv_mem.borrow_mut();
        m.cv_lrw1Q = lrw1Q;
        m.cv_liw1Q = liw1Q;
    }

    /* Allocate the vectors (using yQ0 as a template) */
    let allocOK = cvQuadAllocVectors(cv_mem, yQ0);
    if !allocOK {
        cvProcessError(
            Some(cv_mem),
            CV_MEM_FAIL,
            line!() as i32,
            "CVodeQuadInit",
            file!(),
            MSGCV_MEM_FAIL,
        );
        return CV_MEM_FAIL;
    }

    /* Initialize znQ[0] in the history array */
    let znQ0 = cv_mem.borrow().cv_znQ[0].clone().unwrap();
    N_VScale(ONE, yQ0, &znQ0);

    {
        let mut m = cv_mem.borrow_mut();

        /* Copy the input parameters into CVODES state */
        m.cv_fQ = Some(fQ);

        /* Initialize counters */
        m.cv_nfQe = 0;
        m.cv_netfQ = 0;

        /* Quadrature integration turned ON */
        m.cv_quadr = SUNTRUE;
        m.cv_QuadMallocDone = SUNTRUE;
    }

    /* Quadrature initialization was successful */
    CV_SUCCESS
}

/*-----------------------------------------------------------------*/

/*
 * CVodeQuadReInit
 *
 * CVodeQuadReInit re-initializes CVODES's quadrature related memory
 * for a problem, assuming it has already been allocated in prior
 * calls to CVodeInit and CVodeQuadInit.
 * All problem specification inputs are checked for errors.
 * If any error occurs during initialization, it is reported to the
 * file whose file pointer is errfp.
 * The return value is CV_SUCCESS = 0 if no errors occurred, or
 * a negative value otherwise.
 */

pub fn CVodeQuadReInit(cvode_mem: &CVodeMem, yQ0: &N_Vector) -> i32 {
    /* Check cvode_mem: NULL-mem check handled by type system */
    let cv_mem = cvode_mem;

    /* Check if quadrature was initialized? */
    if !cv_mem.borrow().cv_QuadMallocDone {
        cvProcessError(
            Some(cv_mem),
            CV_NO_QUAD,
            line!() as i32,
            "CVodeQuadReInit",
            file!(),
            MSGCV_NO_QUAD,
        );
        return CV_NO_QUAD;
    }

    /* Initialize znQ[0] in the history array */
    let znQ0 = cv_mem.borrow().cv_znQ[0].clone().unwrap();
    N_VScale(ONE, yQ0, &znQ0);

    {
        let mut m = cv_mem.borrow_mut();

        /* Initialize counters */
        m.cv_nfQe = 0;
        m.cv_netfQ = 0;

        /* Quadrature integration turned ON */
        m.cv_quadr = SUNTRUE;
    }

    /* Quadrature re-initialization was successful */
    CV_SUCCESS
}

/*-----------------------------------------------------------------*/

/*
 * CVodeQuadSStolerances
 * CVodeQuadSVtolerances
 *
 * These functions specify the integration tolerances for sensitivity
 * variables. One of them MUST be called before the first call to
 * CVode IF error control on the quadrature variables is enabled
 * (see CVodeSetQuadErrCon).
 *
 * CVodeQuadSStolerances specifies scalar relative and absolute tolerances.
 * CVodeQuadSVtolerances specifies scalar relative tolerance and a vector
 *   absolute toleranc (a potentially different absolute tolerance for each
 *   vector component).
 */

pub fn CVodeQuadSStolerances(
    cvode_mem: &CVodeMem,
    reltolQ: sunrealtype,
    abstolQ: sunrealtype,
) -> i32 {
    /* NULL-mem check: handled by type system */
    let cv_mem = cvode_mem;

    /* Check if quadrature was initialized? */

    if !cv_mem.borrow().cv_QuadMallocDone {
        cvProcessError(
            Some(cv_mem),
            CV_NO_QUAD,
            line!() as i32,
            "CVodeQuadSStolerances",
            file!(),
            MSGCV_NO_QUAD,
        );
        return CV_NO_QUAD;
    }

    /* Test user-supplied tolerances */

    if reltolQ < ZERO {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeQuadSStolerances",
            file!(),
            MSGCV_BAD_RELTOLQ,
        );
        return CV_ILL_INPUT;
    }

    if abstolQ < 0.0 {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeQuadSStolerances",
            file!(),
            MSGCV_BAD_ABSTOLQ,
        );
        return CV_ILL_INPUT;
    }

    /* Copy tolerances into memory */

    let mut m = cv_mem.borrow_mut();
    m.cv_itolQ = CV_SS;

    m.cv_reltolQ = reltolQ;
    m.cv_SabstolQ = abstolQ;
    m.cv_atolQmin0 = abstolQ == ZERO;

    CV_SUCCESS
}

pub fn CVodeQuadSVtolerances(
    cvode_mem: &CVodeMem,
    reltolQ: sunrealtype,
    abstolQ: &N_Vector,
) -> i32 {
    /* NULL-mem check: handled by type system */
    let cv_mem = cvode_mem;

    /* Check if quadrature was initialized? */

    if !cv_mem.borrow().cv_QuadMallocDone {
        cvProcessError(
            Some(cv_mem),
            CV_NO_QUAD,
            line!() as i32,
            "CVodeQuadSVtolerances",
            file!(),
            MSGCV_NO_QUAD,
        );
        return CV_NO_QUAD;
    }

    /* Test user-supplied tolerances */

    if reltolQ < ZERO {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeQuadSVtolerances",
            file!(),
            MSGCV_BAD_RELTOLQ,
        );
        return CV_ILL_INPUT;
    }

    /* NULL abstolQ check: handled by type system */

    if abstolQ.ops.borrow().nvmin.is_none() {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeQuadSVtolerances",
            file!(),
            "Missing N_VMin routine from N_Vector",
        );
        return CV_ILL_INPUT;
    }
    let atolmin = N_VMin(abstolQ);
    if atolmin < ZERO {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeQuadSVtolerances",
            file!(),
            MSGCV_BAD_ABSTOLQ,
        );
        return CV_ILL_INPUT;
    }

    /* Copy tolerances into memory */

    {
        let mut m = cv_mem.borrow_mut();
        m.cv_itolQ = CV_SV;

        m.cv_reltolQ = reltolQ;
    }

    if !cv_mem.borrow().cv_VabstolQMallocDone {
        let tempvQ = cv_mem.borrow().cv_tempvQ.clone().unwrap();
        let vabstolQ = N_VClone(&tempvQ).unwrap();
        let mut m = cv_mem.borrow_mut();
        m.cv_VabstolQ = Some(vabstolQ);
        m.cv_lrw += m.cv_lrw1Q;
        m.cv_liw += m.cv_liw1Q;
        m.cv_VabstolQMallocDone = SUNTRUE;
    }

    let vabstolQ = cv_mem.borrow().cv_VabstolQ.clone().unwrap();
    N_VScale(ONE, abstolQ, &vabstolQ);
    cv_mem.borrow_mut().cv_atolQmin0 = atolmin == ZERO;

    CV_SUCCESS
}

/*-----------------------------------------------------------------*/

/*
 * CVodeSensInit
 *
 * CVodeSensInit allocates and initializes sensitivity related
 * memory for a problem (using a sensitivity RHS function of type
 * CVSensRhsFn). All problem specification inputs are checked for
 * errors.
 * The return value is CV_SUCCESS = 0 if no errors occurred, or
 * a negative value otherwise.
 */

pub fn CVodeSensInit(
    cvode_mem: &CVodeMem,
    Ns: i32,
    ism: i32,
    fS: Option<CVSensRhsFn>,
    yS0: &[N_Vector],
) -> i32 {
    /* Check cvode_mem: NULL-mem check handled by type system */
    let cv_mem = cvode_mem;

    /* Check if CVodeSensInit or CVodeSensInit1 was already called */

    if cv_mem.borrow().cv_SensMallocDone {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeSensInit",
            file!(),
            MSGCV_SENSINIT_2,
        );
        return CV_ILL_INPUT;
    }

    /* Check if Ns is legal */

    if Ns <= 0 {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeSensInit",
            file!(),
            MSGCV_BAD_NS,
        );
        return CV_ILL_INPUT;
    }
    cv_mem.borrow_mut().cv_Ns = Ns;

    /* Check if ism is compatible */

    if ism == CV_STAGGERED1 {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeSensInit",
            file!(),
            MSGCV_BAD_ISM_IFS,
        );
        return CV_ILL_INPUT;
    }

    /* Check if ism is legal */

    if (ism != CV_SIMULTANEOUS) && (ism != CV_STAGGERED) {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeSensInit",
            file!(),
            MSGCV_BAD_ISM,
        );
        return CV_ILL_INPUT;
    }
    cv_mem.borrow_mut().cv_ism = ism;

    /* Check if yS0 is non-null: handled by type system */

    /* Store sensitivity RHS-related data */

    {
        let mut m = cv_mem.borrow_mut();
        m.cv_ifS = CV_ALLSENS;
        m.cv_fS1 = None;
    }

    match fS {
        None => {
            let token: Box<dyn Any> = Box::new(cv_mem.clone());
            let mut m = cv_mem.borrow_mut();
            m.cv_fSDQ = SUNTRUE;
            m.cv_fS = Some(cvSensRhsInternalDQ);
            m.cv_fS_data = Some(token);
        }
        Some(fS) => {
            let mut m = cv_mem.borrow_mut();
            m.cv_fSDQ = SUNFALSE;
            m.cv_fS = Some(fS);
            /* C: cv_fS_data = cv_user_data (pointer alias); `None` means
            "pass the integrator's cv_user_data" at call time */
            m.cv_fS_data = None;
        }
    }

    /* No memory allocation for STAGGERED1 */

    cv_mem.borrow_mut().cv_stgr1alloc = SUNFALSE;

    /* Allocate the vectors (using yS0[0] as a template) */

    let allocOK = cvSensAllocVectors(cv_mem, &yS0[0]);
    if !allocOK {
        cvProcessError(
            Some(cv_mem),
            CV_MEM_FAIL,
            line!() as i32,
            "CVodeSensInit",
            file!(),
            MSGCV_MEM_FAIL,
        );
        return CV_MEM_FAIL;
    }

    /* Check if larger temporary work arrays are needed for fused vector ops
    (the C NULL-check failure branch is unreachable: Vec allocation aborts
    rather than returning NULL) */
    if Ns as usize * L_MAX > L_MAX {
        let mut m = cv_mem.borrow_mut();
        m.cv_cvals = vec![ZERO; Ns as usize * L_MAX];
        m.cv_Xvecs = Vec::new();
        m.cv_Zvecs = Vec::new();
    }

    /*----------------------------------------------
    All error checking is complete at this point
    -----------------------------------------------*/

    /* Initialize znS[0] in the history array */

    let (cvals, znS0) = {
        let mut m = cv_mem.borrow_mut();
        for is in 0..Ns as usize {
            m.cv_cvals[is] = ONE;
        }
        let cvals = m.cv_cvals.clone();
        let znS0 = m.cv_znS[0].clone();
        (cvals, znS0)
    };

    let retval = N_VScaleVectorArray(Ns, &cvals, yS0, &znS0);
    if retval != CV_SUCCESS {
        return CV_VECTOROP_ERR;
    }

    {
        let mut m = cv_mem.borrow_mut();

        /* Initialize all sensitivity related counters */

        m.cv_nfSe = 0;
        m.cv_nfeS = 0;
        m.cv_ncfnS = 0;
        m.cv_netfS = 0;
        m.cv_nniS = 0;
        m.cv_nnfS = 0;
        m.cv_nsetupsS = 0;

        /* Set default values for plist and pbar */

        for is in 0..Ns as usize {
            m.cv_plist[is] = is as i32;
            m.cv_pbar[is] = ONE;
        }

        /* Sensitivities will be computed */

        m.cv_sensi = SUNTRUE;
        m.cv_SensMallocDone = SUNTRUE;
    }

    /* create a Newton nonlinear solver object by default */
    let (acor, sunctx) = {
        let m = cv_mem.borrow();
        (m.cv_acor.clone().unwrap(), m.cv_sunctx.clone())
    };
    let NLS = if ism == CV_SIMULTANEOUS {
        SUNNonlinSol_NewtonSens(Ns + 1, &acor, &sunctx)
    } else {
        SUNNonlinSol_NewtonSens(Ns, &acor, &sunctx)
    };

    /* check that the nonlinear solver is non-NULL */
    let NLS = match NLS {
        Some(nls) => nls,
        None => {
            cvProcessError(
                Some(cv_mem),
                CV_MEM_FAIL,
                line!() as i32,
                "CVodeSensInit",
                file!(),
                MSGCV_MEM_FAIL,
            );
            cvSensFreeVectors(cv_mem);
            return CV_MEM_FAIL;
        }
    };

    /* attach the nonlinear solver to the CVODE memory */
    let retval = if ism == CV_SIMULTANEOUS {
        crate::cvodes_nls_sim::CVodeSetNonlinearSolverSensSim(cv_mem, &NLS)
    } else {
        crate::cvodes_nls_stg::CVodeSetNonlinearSolverSensStg(cv_mem, &NLS)
    };

    /* check that the nonlinear solver was successfully attached */
    if retval != CV_SUCCESS {
        cvProcessError(
            Some(cv_mem),
            retval,
            line!() as i32,
            "CVodeSensInit",
            file!(),
            "Setting the nonlinear solver failed",
        );
        cvSensFreeVectors(cv_mem);
        let _ = SUNNonlinSolFree(Some(NLS));
        return CV_MEM_FAIL;
    }

    /* set ownership flag */
    if ism == CV_SIMULTANEOUS {
        cv_mem.borrow_mut().ownNLSsim = SUNTRUE;
    } else {
        cv_mem.borrow_mut().ownNLSstg = SUNTRUE;
    }

    /* Sensitivity initialization was successful */
    CV_SUCCESS
}

/*
 * CVodeSensInit1
 *
 * CVodeSensInit1 allocates and initializes sensitivity related
 * memory for a problem (using a sensitivity RHS function of type
 * CVSensRhs1Fn). All problem specification inputs are checked for
 * errors.
 * The return value is CV_SUCCESS = 0 if no errors occurred, or
 * a negative value otherwise.
 */

pub fn CVodeSensInit1(
    cvode_mem: &CVodeMem,
    Ns: i32,
    ism: i32,
    fS1: Option<CVSensRhs1Fn>,
    yS0: &[N_Vector],
) -> i32 {
    /* Check cvode_mem: NULL-mem check handled by type system */
    let cv_mem = cvode_mem;

    /* Check if CVodeSensInit or CVodeSensInit1 was already called */

    if cv_mem.borrow().cv_SensMallocDone {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeSensInit1",
            file!(),
            MSGCV_SENSINIT_2,
        );
        return CV_ILL_INPUT;
    }

    /* Check if Ns is legal */

    if Ns <= 0 {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeSensInit1",
            file!(),
            MSGCV_BAD_NS,
        );
        return CV_ILL_INPUT;
    }
    cv_mem.borrow_mut().cv_Ns = Ns;

    /* Check if ism is legal */

    if (ism != CV_SIMULTANEOUS) && (ism != CV_STAGGERED) && (ism != CV_STAGGERED1) {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeSensInit1",
            file!(),
            MSGCV_BAD_ISM,
        );
        return CV_ILL_INPUT;
    }
    cv_mem.borrow_mut().cv_ism = ism;

    /* Check if yS0 is non-null: handled by type system */

    /* Store sensitivity RHS-related data */

    {
        let mut m = cv_mem.borrow_mut();
        m.cv_ifS = CV_ONESENS;
        m.cv_fS = None;
    }

    match fS1 {
        None => {
            let token: Box<dyn Any> = Box::new(cv_mem.clone());
            let mut m = cv_mem.borrow_mut();
            m.cv_fSDQ = SUNTRUE;
            m.cv_fS1 = Some(cvSensRhs1InternalDQ);
            m.cv_fS_data = Some(token);
        }
        Some(fS1) => {
            let mut m = cv_mem.borrow_mut();
            m.cv_fSDQ = SUNFALSE;
            m.cv_fS1 = Some(fS1);
            /* C: cv_fS_data = cv_user_data (pointer alias); `None` means
            "pass the integrator's cv_user_data" at call time */
            m.cv_fS_data = None;
        }
    }

    /* Allocate ncfS1, ncfnS1, and nniS1 if needed (the C NULL-check
    failure branch is unreachable: Vec allocation aborts rather than
    returning NULL) */

    if ism == CV_STAGGERED1 {
        let mut m = cv_mem.borrow_mut();
        m.cv_stgr1alloc = SUNTRUE;
        m.cv_ncfS1 = vec![0; Ns as usize];
        m.cv_ncfnS1 = vec![0; Ns as usize];
        m.cv_nniS1 = vec![0; Ns as usize];
        m.cv_nnfS1 = vec![0; Ns as usize];
    } else {
        cv_mem.borrow_mut().cv_stgr1alloc = SUNFALSE;
    }

    /* Allocate the vectors (using yS0[0] as a template) */

    let allocOK = cvSensAllocVectors(cv_mem, &yS0[0]);
    if !allocOK {
        {
            let mut m = cv_mem.borrow_mut();
            if m.cv_stgr1alloc {
                m.cv_ncfS1 = Vec::new();
                m.cv_ncfnS1 = Vec::new();
                m.cv_nniS1 = Vec::new();
                m.cv_nnfS1 = Vec::new();
            }
        }
        cvProcessError(
            Some(cv_mem),
            CV_MEM_FAIL,
            line!() as i32,
            "CVodeSensInit1",
            file!(),
            MSGCV_MEM_FAIL,
        );
        return CV_MEM_FAIL;
    }

    /* Check if larger temporary work arrays are needed for fused vector ops
    (the C NULL-check failure branch is unreachable: Vec allocation aborts
    rather than returning NULL) */
    if Ns as usize * L_MAX > L_MAX {
        let mut m = cv_mem.borrow_mut();
        m.cv_cvals = vec![ZERO; Ns as usize * L_MAX];
        m.cv_Xvecs = Vec::new();
        m.cv_Zvecs = Vec::new();
    }

    /*----------------------------------------------
    All error checking is complete at this point
    -----------------------------------------------*/

    /* Initialize znS[0] in the history array */

    let (cvals, znS0) = {
        let mut m = cv_mem.borrow_mut();
        for is in 0..Ns as usize {
            m.cv_cvals[is] = ONE;
        }
        let cvals = m.cv_cvals.clone();
        let znS0 = m.cv_znS[0].clone();
        (cvals, znS0)
    };

    let retval = N_VScaleVectorArray(Ns, &cvals, yS0, &znS0);
    if retval != CV_SUCCESS {
        return CV_VECTOROP_ERR;
    }

    {
        let mut m = cv_mem.borrow_mut();

        /* Initialize all sensitivity related counters */

        m.cv_nfSe = 0;
        m.cv_nfeS = 0;
        m.cv_ncfnS = 0;
        m.cv_netfS = 0;
        m.cv_nniS = 0;
        m.cv_nnfS = 0;
        m.cv_nsetupsS = 0;
        if ism == CV_STAGGERED1 {
            for is in 0..Ns as usize {
                m.cv_ncfnS1[is] = 0;
                m.cv_nniS1[is] = 0;
                m.cv_nnfS1[is] = 0;
            }
        }

        /* Set default values for plist and pbar */

        for is in 0..Ns as usize {
            m.cv_plist[is] = is as i32;
            m.cv_pbar[is] = ONE;
        }

        /* Sensitivities will be computed */

        m.cv_sensi = SUNTRUE;
        m.cv_SensMallocDone = SUNTRUE;
    }

    /* create a Newton nonlinear solver object by default */
    let (acor, sunctx) = {
        let m = cv_mem.borrow();
        (m.cv_acor.clone().unwrap(), m.cv_sunctx.clone())
    };
    let NLS = if ism == CV_SIMULTANEOUS {
        SUNNonlinSol_NewtonSens(Ns + 1, &acor, &sunctx)
    } else if ism == CV_STAGGERED {
        SUNNonlinSol_NewtonSens(Ns, &acor, &sunctx)
    } else {
        SUNNonlinSol_Newton(&acor, &sunctx)
    };

    /* check that the nonlinear solver is non-NULL */
    let NLS = match NLS {
        Some(nls) => nls,
        None => {
            cvProcessError(
                Some(cv_mem),
                CV_MEM_FAIL,
                line!() as i32,
                "CVodeSensInit1",
                file!(),
                MSGCV_MEM_FAIL,
            );
            cvSensFreeVectors(cv_mem);
            return CV_MEM_FAIL;
        }
    };

    /* attach the nonlinear solver to the CVODE memory */
    let retval = if ism == CV_SIMULTANEOUS {
        crate::cvodes_nls_sim::CVodeSetNonlinearSolverSensSim(cv_mem, &NLS)
    } else if ism == CV_STAGGERED {
        crate::cvodes_nls_stg::CVodeSetNonlinearSolverSensStg(cv_mem, &NLS)
    } else {
        crate::cvodes_nls_stg1::CVodeSetNonlinearSolverSensStg1(cv_mem, &NLS)
    };

    /* check that the nonlinear solver was successfully attached */
    if retval != CV_SUCCESS {
        cvProcessError(
            Some(cv_mem),
            retval,
            line!() as i32,
            "CVodeSensInit1",
            file!(),
            "Setting the nonlinear solver failed",
        );
        cvSensFreeVectors(cv_mem);
        let _ = SUNNonlinSolFree(Some(NLS));
        return CV_MEM_FAIL;
    }

    /* set ownership flag */
    if ism == CV_SIMULTANEOUS {
        cv_mem.borrow_mut().ownNLSsim = SUNTRUE;
    } else if ism == CV_STAGGERED {
        cv_mem.borrow_mut().ownNLSstg = SUNTRUE;
    } else {
        cv_mem.borrow_mut().ownNLSstg1 = SUNTRUE;
    }

    /* Sensitivity initialization was successful */
    CV_SUCCESS
}

/*-----------------------------------------------------------------*/

/*
 * CVodeSensReInit
 *
 * CVodeSensReInit re-initializes CVODES's sensitivity related memory
 * for a problem, assuming it has already been allocated in prior
 * calls to CVodeInit and CVodeSensInit/CVodeSensInit1.
 * All problem specification inputs are checked for errors.
 * The number of sensitivities Ns is assumed to be unchanged since
 * the previous call to CVodeSensInit.
 * If any error occurs during initialization, it is reported to the
 * file whose file pointer is errfp.
 * The return value is CV_SUCCESS = 0 if no errors occurred, or
 * a negative value otherwise.
 */

pub fn CVodeSensReInit(cvode_mem: &CVodeMem, ism: i32, yS0: &[N_Vector]) -> i32 {
    /* Check cvode_mem: NULL-mem check handled by type system */
    let cv_mem = cvode_mem;

    /* Was sensitivity initialized? */

    if !cv_mem.borrow().cv_SensMallocDone {
        cvProcessError(
            Some(cv_mem),
            CV_NO_SENS,
            line!() as i32,
            "CVodeSensReInit",
            file!(),
            MSGCV_NO_SENSI,
        );
        return CV_NO_SENS;
    }

    /* Check if ism is compatible */

    if (cv_mem.borrow().cv_ifS == CV_ALLSENS) && (ism == CV_STAGGERED1) {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeSensReInit",
            file!(),
            MSGCV_BAD_ISM_IFS,
        );
        return CV_ILL_INPUT;
    }

    /* Check if ism is legal */

    if (ism != CV_SIMULTANEOUS) && (ism != CV_STAGGERED) && (ism != CV_STAGGERED1) {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeSensReInit",
            file!(),
            MSGCV_BAD_ISM,
        );
        return CV_ILL_INPUT;
    }
    cv_mem.borrow_mut().cv_ism = ism;

    /* Check if yS0 is non-null: handled by type system */

    /* Allocate ncfS1, ncfnS1, and nniS1 if needed (the C NULL-check
    failure branch is unreachable: Vec allocation aborts rather than
    returning NULL) */

    if (ism == CV_STAGGERED1) && !cv_mem.borrow().cv_stgr1alloc {
        let mut m = cv_mem.borrow_mut();
        let Ns = m.cv_Ns as usize;
        m.cv_stgr1alloc = SUNTRUE;
        m.cv_ncfS1 = vec![0; Ns];
        m.cv_ncfnS1 = vec![0; Ns];
        m.cv_nniS1 = vec![0; Ns];
        m.cv_nnfS1 = vec![0; Ns];
    }

    /*----------------------------------------------
    All error checking is complete at this point
    -----------------------------------------------*/

    /* Initialize znS[0] in the history array */

    let (Ns, cvals, znS0) = {
        let mut m = cv_mem.borrow_mut();
        let Ns = m.cv_Ns;
        for is in 0..Ns as usize {
            m.cv_cvals[is] = ONE;
        }
        let cvals = m.cv_cvals.clone();
        let znS0 = m.cv_znS[0].clone();
        (Ns, cvals, znS0)
    };

    let retval = N_VScaleVectorArray(Ns, &cvals, yS0, &znS0);
    if retval != CV_SUCCESS {
        return CV_VECTOROP_ERR;
    }

    {
        let mut m = cv_mem.borrow_mut();

        /* Initialize all sensitivity related counters */

        m.cv_nfSe = 0;
        m.cv_nfeS = 0;
        m.cv_ncfnS = 0;
        m.cv_netfS = 0;
        m.cv_nniS = 0;
        m.cv_nnfS = 0;
        m.cv_nsetupsS = 0;
        if ism == CV_STAGGERED1 {
            for is in 0..m.cv_Ns as usize {
                m.cv_ncfnS1[is] = 0;
                m.cv_nniS1[is] = 0;
                m.cv_nnfS1[is] = 0;
            }
        }

        /* Problem has been successfully re-initialized */

        m.cv_sensi = SUNTRUE;
    }

    /* Check if the NLS exists, create the default NLS if needed */
    let need_nls = {
        let m = cv_mem.borrow();
        (ism == CV_SIMULTANEOUS && m.NLSsim.is_none())
            || (ism == CV_STAGGERED && m.NLSstg.is_none())
            || (ism == CV_STAGGERED1 && m.NLSstg1.is_none())
    };
    if need_nls {
        /* create a Newton nonlinear solver object by default */
        let (Ns, acor, sunctx) = {
            let m = cv_mem.borrow();
            (m.cv_Ns, m.cv_acor.clone().unwrap(), m.cv_sunctx.clone())
        };
        let NLS = if ism == CV_SIMULTANEOUS {
            SUNNonlinSol_NewtonSens(Ns + 1, &acor, &sunctx)
        } else if ism == CV_STAGGERED {
            SUNNonlinSol_NewtonSens(Ns, &acor, &sunctx)
        } else {
            SUNNonlinSol_Newton(&acor, &sunctx)
        };

        /* check that the nonlinear solver is non-NULL */
        let NLS = match NLS {
            Some(nls) => nls,
            None => {
                cvProcessError(
                    Some(cv_mem),
                    CV_MEM_FAIL,
                    line!() as i32,
                    "CVodeSensReInit",
                    file!(),
                    MSGCV_MEM_FAIL,
                );
                return CV_MEM_FAIL;
            }
        };

        /* attach the nonlinear solver to the CVODES memory */
        let retval = if ism == CV_SIMULTANEOUS {
            crate::cvodes_nls_sim::CVodeSetNonlinearSolverSensSim(cv_mem, &NLS)
        } else if ism == CV_STAGGERED {
            crate::cvodes_nls_stg::CVodeSetNonlinearSolverSensStg(cv_mem, &NLS)
        } else {
            crate::cvodes_nls_stg1::CVodeSetNonlinearSolverSensStg1(cv_mem, &NLS)
        };

        /* check that the nonlinear solver was successfully attached */
        if retval != CV_SUCCESS {
            cvProcessError(
                Some(cv_mem),
                retval,
                line!() as i32,
                "CVodeSensReInit",
                file!(),
                "Setting the nonlinear solver failed",
            );
            let _ = SUNNonlinSolFree(Some(NLS));
            return CV_MEM_FAIL;
        }

        /* set ownership flag */
        if ism == CV_SIMULTANEOUS {
            cv_mem.borrow_mut().ownNLSsim = SUNTRUE;
        } else if ism == CV_STAGGERED {
            cv_mem.borrow_mut().ownNLSstg = SUNTRUE;
        } else {
            cv_mem.borrow_mut().ownNLSstg1 = SUNTRUE;
        }

        /* initialize the NLS object, this assumes that the linear solver has
        already been initialized in CVodeInit */
        let retval = if ism == CV_SIMULTANEOUS {
            crate::cvodes_nls_sim::cvNlsInitSensSim(cv_mem)
        } else if ism == CV_STAGGERED {
            crate::cvodes_nls_stg::cvNlsInitSensStg(cv_mem)
        } else {
            crate::cvodes_nls_stg1::cvNlsInitSensStg1(cv_mem)
        };

        if retval != CV_SUCCESS {
            cvProcessError(
                Some(cv_mem),
                CV_NLS_INIT_FAIL,
                line!() as i32,
                "CVodeSensReInit",
                file!(),
                MSGCV_NLS_INIT_FAIL,
            );
            return CV_NLS_INIT_FAIL;
        }
    }

    /* Sensitivity re-initialization was successful */
    CV_SUCCESS
}

/*-----------------------------------------------------------------*/

/*
 * CVodeSensSStolerances
 * CVodeSensSVtolerances
 * CVodeSensEEtolerances
 *
 * These functions specify the integration tolerances for sensitivity
 * variables. One of them MUST be called before the first call to CVode.
 *
 * CVodeSensSStolerances specifies scalar relative and absolute tolerances.
 * CVodeSensSVtolerances specifies scalar relative tolerance and a vector
 *   absolute tolerance for each sensitivity vector (a potentially different
 *   absolute tolerance for each vector component).
 * CVodeSensEEtolerances specifies that tolerances for sensitivity variables
 *   should be estimated from those provided for the state variables.
 */

pub fn CVodeSensSStolerances(
    cvode_mem: &CVodeMem,
    reltolS: sunrealtype,
    abstolS: &[sunrealtype],
) -> i32 {
    /* NULL-mem check: handled by type system */
    let cv_mem = cvode_mem;

    /* Was sensitivity initialized? */

    if !cv_mem.borrow().cv_SensMallocDone {
        cvProcessError(
            Some(cv_mem),
            CV_NO_SENS,
            line!() as i32,
            "CVodeSensSStolerances",
            file!(),
            MSGCV_NO_SENSI,
        );
        return CV_NO_SENS;
    }

    /* Test user-supplied tolerances */

    if reltolS < ZERO {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeSensSStolerances",
            file!(),
            MSGCV_BAD_RELTOLS,
        );
        return CV_ILL_INPUT;
    }

    /* NULL abstolS check: handled by type system */

    let Ns = cv_mem.borrow().cv_Ns;
    for is in 0..Ns as usize {
        if abstolS[is] < ZERO {
            cvProcessError(
                Some(cv_mem),
                CV_ILL_INPUT,
                line!() as i32,
                "CVodeSensSStolerances",
                file!(),
                MSGCV_BAD_ABSTOLS,
            );
            return CV_ILL_INPUT;
        }
    }

    /* Copy tolerances into memory */

    let mut m = cv_mem.borrow_mut();
    m.cv_itolS = CV_SS;

    m.cv_reltolS = reltolS;

    if !m.cv_SabstolSMallocDone {
        m.cv_SabstolS = vec![ZERO; Ns as usize];
        m.cv_atolSmin0 = vec![SUNFALSE; Ns as usize];
        m.cv_lrw += Ns as i64;
        m.cv_SabstolSMallocDone = SUNTRUE;
    }

    for is in 0..Ns as usize {
        m.cv_SabstolS[is] = abstolS[is];
        m.cv_atolSmin0[is] = abstolS[is] == ZERO;
    }

    CV_SUCCESS
}

pub fn CVodeSensSVtolerances(
    cvode_mem: &CVodeMem,
    reltolS: sunrealtype,
    abstolS: &[N_Vector],
) -> i32 {
    /* NULL-mem check: handled by type system */
    let cv_mem = cvode_mem;

    /* Was sensitivity initialized? */

    if !cv_mem.borrow().cv_SensMallocDone {
        cvProcessError(
            Some(cv_mem),
            CV_NO_SENS,
            line!() as i32,
            "CVodeSensSVtolerances",
            file!(),
            MSGCV_NO_SENSI,
        );
        return CV_NO_SENS;
    }

    /* Test user-supplied tolerances */

    if reltolS < ZERO {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeSensSVtolerances",
            file!(),
            MSGCV_BAD_RELTOLS,
        );
        return CV_ILL_INPUT;
    }

    /* NULL abstolS check: handled by type system */

    let (Ns, tempv) = {
        let m = cv_mem.borrow();
        (m.cv_Ns, m.cv_tempv.clone().unwrap())
    };
    if tempv.ops.borrow().nvmin.is_none() {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeSensSVtolerances",
            file!(),
            "Missing N_VMin routine from N_Vector",
        );
        return CV_ILL_INPUT;
    }
    let mut atolmin: Vec<sunrealtype> = vec![ZERO; Ns as usize];
    for is in 0..Ns as usize {
        atolmin[is] = N_VMin(&abstolS[is]);
        if atolmin[is] < ZERO {
            cvProcessError(
                Some(cv_mem),
                CV_ILL_INPUT,
                line!() as i32,
                "CVodeSensSVtolerances",
                file!(),
                MSGCV_BAD_ABSTOLS,
            );
            /* C: free(atolmin) — the Vec is dropped on return */
            return CV_ILL_INPUT;
        }
    }

    /* Copy tolerances into memory */

    {
        let mut m = cv_mem.borrow_mut();
        m.cv_itolS = CV_SV;

        m.cv_reltolS = reltolS;
    }

    if !cv_mem.borrow().cv_VabstolSMallocDone {
        let vabstolS = N_VCloneVectorArray(Ns, &tempv).unwrap();
        let mut m = cv_mem.borrow_mut();
        m.cv_VabstolS = vabstolS;
        m.cv_atolSmin0 = vec![SUNFALSE; Ns as usize];
        m.cv_lrw += Ns as i64 * m.cv_lrw1;
        m.cv_liw += Ns as i64 * m.cv_liw1;
        m.cv_VabstolSMallocDone = SUNTRUE;
    }

    let (cvals, vabstolS) = {
        let mut m = cv_mem.borrow_mut();
        for is in 0..Ns as usize {
            m.cv_cvals[is] = ONE;
            m.cv_atolSmin0[is] = atolmin[is] == ZERO;
        }
        let cvals = m.cv_cvals.clone();
        let vabstolS = m.cv_VabstolS.clone();
        (cvals, vabstolS)
    };
    drop(atolmin);

    let retval = N_VScaleVectorArray(Ns, &cvals, abstolS, &vabstolS);
    if retval != CV_SUCCESS {
        return CV_VECTOROP_ERR;
    }

    CV_SUCCESS
}

pub fn CVodeSensEEtolerances(cvode_mem: &CVodeMem) -> i32 {
    /* NULL-mem check: handled by type system */
    let cv_mem = cvode_mem;

    /* Was sensitivity initialized? */

    if !cv_mem.borrow().cv_SensMallocDone {
        cvProcessError(
            Some(cv_mem),
            CV_NO_SENS,
            line!() as i32,
            "CVodeSensEEtolerances",
            file!(),
            MSGCV_NO_SENSI,
        );
        return CV_NO_SENS;
    }

    cv_mem.borrow_mut().cv_itolS = CV_EE;

    CV_SUCCESS
}

/*-----------------------------------------------------------------*/

/*
 * CVodeQuadSensInit
 *
 */

pub fn CVodeQuadSensInit(
    cvode_mem: &CVodeMem,
    fQS: Option<CVQuadSensRhsFn>,
    yQS0: &[N_Vector],
) -> i32 {
    /* Check cvode_mem: NULL-mem check handled by type system */
    let cv_mem = cvode_mem;

    /* Check if sensitivity analysis is active */
    if !cv_mem.borrow().cv_sensi {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeQuadSensInit",
            file!(),
            MSGCV_NO_SENSI,
        );
        return CV_ILL_INPUT;
    }

    /* Check if yQS0 is non-null: handled by type system */

    /* Allocate the vectors (using yQS0[0] as a template) */
    let allocOK = cvQuadSensAllocVectors(cv_mem, &yQS0[0]);
    if !allocOK {
        cvProcessError(
            Some(cv_mem),
            CV_MEM_FAIL,
            line!() as i32,
            "CVodeQuadSensInit",
            file!(),
            MSGCV_MEM_FAIL,
        );
        return CV_MEM_FAIL;
    }

    /*----------------------------------------------
    All error checking is complete at this point
    -----------------------------------------------*/

    /* Set fQS */
    match fQS {
        None => {
            let token: Box<dyn Any> = Box::new(cv_mem.clone());
            let mut m = cv_mem.borrow_mut();
            m.cv_fQSDQ = SUNTRUE;
            m.cv_fQS = Some(cvQuadSensRhsInternalDQ);

            m.cv_fQS_data = Some(token);
        }
        Some(fQS) => {
            let mut m = cv_mem.borrow_mut();
            m.cv_fQSDQ = SUNFALSE;
            m.cv_fQS = Some(fQS);

            /* C: cv_fQS_data = cv_user_data (pointer alias); `None` means
            "pass the integrator's cv_user_data" at call time */
            m.cv_fQS_data = None;
        }
    }

    /* Initialize znQS[0] in the history array */
    let (Ns, cvals, znQS0) = {
        let mut m = cv_mem.borrow_mut();
        let Ns = m.cv_Ns;
        for is in 0..Ns as usize {
            m.cv_cvals[is] = ONE;
        }
        let cvals = m.cv_cvals.clone();
        let znQS0 = m.cv_znQS[0].clone();
        (Ns, cvals, znQS0)
    };

    let retval = N_VScaleVectorArray(Ns, &cvals, yQS0, &znQS0);
    if retval != CV_SUCCESS {
        return CV_VECTOROP_ERR;
    }

    {
        let mut m = cv_mem.borrow_mut();

        /* Initialize all sensitivity related counters */
        m.cv_nfQSe = 0;
        m.cv_nfQeS = 0;
        m.cv_netfQS = 0;

        /* Quadrature sensitivities will be computed */
        m.cv_quadr_sensi = SUNTRUE;
        m.cv_QuadSensMallocDone = SUNTRUE;
    }

    /* Sensitivity initialization was successful */
    CV_SUCCESS
}

/*
 * CVodeQuadSensReInit
 *
 */

pub fn CVodeQuadSensReInit(cvode_mem: &CVodeMem, yQS0: &[N_Vector]) -> i32 {
    /* Check cvode_mem: NULL-mem check handled by type system */
    let cv_mem = cvode_mem;

    /* Check if sensitivity analysis is active */
    if !cv_mem.borrow().cv_sensi {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeQuadSensReInit",
            file!(),
            MSGCV_NO_SENSI,
        );
        return CV_NO_SENS;
    }

    /* Was quadrature sensitivity initialized? */
    if !cv_mem.borrow().cv_QuadSensMallocDone {
        cvProcessError(
            Some(cv_mem),
            CV_NO_QUADSENS,
            line!() as i32,
            "CVodeQuadSensReInit",
            file!(),
            MSGCV_NO_QUADSENSI,
        );
        return CV_NO_QUADSENS;
    }

    /* Check if yQS0 is non-null: handled by type system */

    /*----------------------------------------------
    All error checking is complete at this point
    -----------------------------------------------*/

    /* Initialize znQS[0] in the history array */
    let (Ns, cvals, znQS0) = {
        let mut m = cv_mem.borrow_mut();
        let Ns = m.cv_Ns;
        for is in 0..Ns as usize {
            m.cv_cvals[is] = ONE;
        }
        let cvals = m.cv_cvals.clone();
        let znQS0 = m.cv_znQS[0].clone();
        (Ns, cvals, znQS0)
    };

    let retval = N_VScaleVectorArray(Ns, &cvals, yQS0, &znQS0);
    if retval != CV_SUCCESS {
        return CV_VECTOROP_ERR;
    }

    {
        let mut m = cv_mem.borrow_mut();

        /* Initialize all sensitivity related counters */
        m.cv_nfQSe = 0;
        m.cv_nfQeS = 0;
        m.cv_netfQS = 0;

        /* Quadrature sensitivities will be computed */
        m.cv_quadr_sensi = SUNTRUE;
    }

    /* Problem has been successfully re-initialized */
    CV_SUCCESS
}

/*
 * CVodeQuadSensSStolerances
 * CVodeQuadSensSVtolerances
 * CVodeQuadSensEEtolerances
 *
 * These functions specify the integration tolerances for quadrature
 * sensitivity variables. One of them MUST be called before the first
 * call to CVode IF these variables are included in the error test.
 *
 * CVodeQuadSensSStolerances specifies scalar relative and absolute tolerances.
 * CVodeQuadSensSVtolerances specifies scalar relative tolerance and a vector
 *   absolute tolerance for each quadrature sensitivity vector (a potentially
 *   different absolute tolerance for each vector component).
 * CVodeQuadSensEEtolerances specifies that tolerances for sensitivity variables
 *   should be estimated from those provided for the quadrature variables.
 *   In this case, tolerances for the quadrature variables must be
 *   specified through a call to one of CVodeQuad**tolerances.
 */

pub fn CVodeQuadSensSStolerances(
    cvode_mem: &CVodeMem,
    reltolQS: sunrealtype,
    abstolQS: &[sunrealtype],
) -> i32 {
    /* NULL-mem check: handled by type system */
    let cv_mem = cvode_mem;

    /* Check if sensitivity was initialized */

    if !cv_mem.borrow().cv_SensMallocDone {
        cvProcessError(
            Some(cv_mem),
            CV_NO_SENS,
            line!() as i32,
            "CVodeQuadSensSStolerances",
            file!(),
            MSGCV_NO_SENSI,
        );
        return CV_NO_SENS;
    }

    /* Check if quadrature sensitivity was initialized? */

    if !cv_mem.borrow().cv_QuadSensMallocDone {
        cvProcessError(
            Some(cv_mem),
            CV_NO_QUADSENS,
            line!() as i32,
            "CVodeQuadSensSStolerances",
            file!(),
            MSGCV_NO_QUADSENSI,
        );
        return CV_NO_QUAD;
    }

    /* Test user-supplied tolerances */

    if reltolQS < ZERO {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeQuadSensSStolerances",
            file!(),
            MSGCV_BAD_RELTOLQS,
        );
        return CV_ILL_INPUT;
    }

    /* NULL abstolQS check: handled by type system */

    let Ns = cv_mem.borrow().cv_Ns;
    for is in 0..Ns as usize {
        if abstolQS[is] < ZERO {
            cvProcessError(
                Some(cv_mem),
                CV_ILL_INPUT,
                line!() as i32,
                "CVodeQuadSensSStolerances",
                file!(),
                MSGCV_BAD_ABSTOLQS,
            );
            return CV_ILL_INPUT;
        }
    }

    /* Copy tolerances into memory */

    let mut m = cv_mem.borrow_mut();
    m.cv_itolQS = CV_SS;

    m.cv_reltolQS = reltolQS;

    if !m.cv_SabstolQSMallocDone {
        m.cv_SabstolQS = vec![ZERO; Ns as usize];
        m.cv_atolQSmin0 = vec![SUNFALSE; Ns as usize];
        m.cv_lrw += Ns as i64;
        m.cv_SabstolQSMallocDone = SUNTRUE;
    }

    for is in 0..Ns as usize {
        m.cv_SabstolQS[is] = abstolQS[is];
        m.cv_atolQSmin0[is] = abstolQS[is] == ZERO;
    }

    CV_SUCCESS
}

pub fn CVodeQuadSensSVtolerances(
    cvode_mem: &CVodeMem,
    reltolQS: sunrealtype,
    abstolQS: &[N_Vector],
) -> i32 {
    /* NULL-mem check: handled by type system */
    let cv_mem = cvode_mem;

    /* check if sensitivity was initialized */

    if !cv_mem.borrow().cv_SensMallocDone {
        cvProcessError(
            Some(cv_mem),
            CV_NO_SENS,
            line!() as i32,
            "CVodeQuadSensSVtolerances",
            file!(),
            MSGCV_NO_SENSI,
        );
        return CV_NO_SENS;
    }

    /* Check if quadrature sensitivity was initialized? */

    if !cv_mem.borrow().cv_QuadSensMallocDone {
        cvProcessError(
            Some(cv_mem),
            CV_NO_QUADSENS,
            line!() as i32,
            "CVodeQuadSensSVtolerances",
            file!(),
            MSGCV_NO_QUADSENSI,
        );
        return CV_NO_QUAD;
    }

    /* Test user-supplied tolerances */

    if reltolQS < ZERO {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeQuadSensSVtolerances",
            file!(),
            MSGCV_BAD_RELTOLQS,
        );
        return CV_ILL_INPUT;
    }

    /* NULL abstolQS check: handled by type system */

    let (Ns, tempv) = {
        let m = cv_mem.borrow();
        (m.cv_Ns, m.cv_tempv.clone().unwrap())
    };
    if tempv.ops.borrow().nvmin.is_none() {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeQuadSensSVtolerances",
            file!(),
            "Missing N_VMin routine from N_Vector",
        );
        return CV_ILL_INPUT;
    }
    let mut atolmin: Vec<sunrealtype> = vec![ZERO; Ns as usize];
    for is in 0..Ns as usize {
        atolmin[is] = N_VMin(&abstolQS[is]);
        if atolmin[is] < ZERO {
            cvProcessError(
                Some(cv_mem),
                CV_ILL_INPUT,
                line!() as i32,
                "CVodeQuadSensSVtolerances",
                file!(),
                MSGCV_BAD_ABSTOLQS,
            );
            /* C: free(atolmin) — the Vec is dropped on return */
            return CV_ILL_INPUT;
        }
    }

    /* Copy tolerances into memory */

    {
        let mut m = cv_mem.borrow_mut();
        m.cv_itolQS = CV_SV;

        m.cv_reltolQS = reltolQS;
    }

    if !cv_mem.borrow().cv_VabstolQSMallocDone {
        let tempvQ = cv_mem.borrow().cv_tempvQ.clone().unwrap();
        let vabstolQS = N_VCloneVectorArray(Ns, &tempvQ).unwrap();
        let mut m = cv_mem.borrow_mut();
        m.cv_VabstolQS = vabstolQS;
        m.cv_atolQSmin0 = vec![SUNFALSE; Ns as usize];
        m.cv_lrw += Ns as i64 * m.cv_lrw1Q;
        m.cv_liw += Ns as i64 * m.cv_liw1Q;
        m.cv_VabstolQSMallocDone = SUNTRUE;
    }

    let (cvals, vabstolQS) = {
        let mut m = cv_mem.borrow_mut();
        for is in 0..Ns as usize {
            m.cv_cvals[is] = ONE;
            m.cv_atolQSmin0[is] = atolmin[is] == ZERO;
        }
        let cvals = m.cv_cvals.clone();
        let vabstolQS = m.cv_VabstolQS.clone();
        (cvals, vabstolQS)
    };
    drop(atolmin);

    let retval = N_VScaleVectorArray(Ns, &cvals, abstolQS, &vabstolQS);
    if retval != CV_SUCCESS {
        return CV_VECTOROP_ERR;
    }

    CV_SUCCESS
}

pub fn CVodeQuadSensEEtolerances(cvode_mem: &CVodeMem) -> i32 {
    /* NULL-mem check: handled by type system */
    let cv_mem = cvode_mem;

    /* check if sensitivity was initialized */

    if !cv_mem.borrow().cv_SensMallocDone {
        cvProcessError(
            Some(cv_mem),
            CV_NO_SENS,
            line!() as i32,
            "CVodeQuadSensEEtolerances",
            file!(),
            MSGCV_NO_SENSI,
        );
        return CV_NO_SENS;
    }

    /* Check if quadrature sensitivity was initialized? */

    if !cv_mem.borrow().cv_QuadSensMallocDone {
        cvProcessError(
            Some(cv_mem),
            CV_NO_QUADSENS,
            line!() as i32,
            "CVodeQuadSensEEtolerances",
            file!(),
            MSGCV_NO_QUADSENSI,
        );
        return CV_NO_QUAD;
    }

    cv_mem.borrow_mut().cv_itolQS = CV_EE;

    CV_SUCCESS
}

/*-----------------------------------------------------------------*/

/*
 * CVodeSensToggleOff
 *
 * CVodeSensToggleOff deactivates sensitivity calculations.
 * It does NOT deallocate sensitivity-related memory.
 */

pub fn CVodeSensToggleOff(cvode_mem: &CVodeMem) -> i32 {
    /* Check cvode_mem: NULL-mem check handled by type system */
    let cv_mem = cvode_mem;

    /* Disable sensitivities */
    let mut m = cv_mem.borrow_mut();
    m.cv_sensi = SUNFALSE;
    m.cv_quadr_sensi = SUNFALSE;

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
    /* Check cvode_mem pointer: NULL-mem check handled by type system */
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
 * This routine is the main driver of the CVODES package.
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

    /* Check if cvode_mem exists: NULL-mem check handled by type system */
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

    /* Check for tret != NULL: handled by type system */

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

        /*
         * Call f at (t0,y0), set zn[1] = y'(t0).
         * If computing any quadratures, call fQ at (t0,y0), set znQ[1] = yQ'(t0)
         * If computing sensitivities, call fS at (t0,y0,yS0), set znS[1][is] = yS'(t0), is=1,...,Ns.
         * If computing quadr. sensi., call fQS at (t0,y0,yS0), set znQS[1][is] = yQS'(t0), is=1,...,Ns.
         */

        let (tn, zn0, zn1) = {
            let m = cv_mem.borrow();
            (
                m.cv_tn,
                m.cv_zn[0].clone().unwrap(),
                m.cv_zn[1].clone().unwrap(),
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

        if cv_mem.borrow().cv_quadr {
            let znQ1 = cv_mem.borrow().cv_znQ[1].clone().unwrap();
            let retval = cv_call_fQ(cv_mem, tn, &zn0, &znQ1);
            cv_mem.borrow_mut().cv_nfQe += 1;
            if retval < 0 {
                cvProcessError(
                    Some(cv_mem),
                    CV_QRHSFUNC_FAIL,
                    line!() as i32,
                    "CVode",
                    file!(),
                    &MSGCV_QRHSFUNC_FAILED(tn),
                );
                return CV_QRHSFUNC_FAIL;
            }
            if retval > 0 {
                cvProcessError(
                    Some(cv_mem),
                    CV_FIRST_QRHSFUNC_ERR,
                    line!() as i32,
                    "CVode",
                    file!(),
                    MSGCV_QRHSFUNC_FIRST,
                );
                return CV_FIRST_QRHSFUNC_ERR;
            }
        }

        if cv_mem.borrow().cv_sensi {
            let (znS0, znS1, tempv, ftemp) = {
                let m = cv_mem.borrow();
                (
                    m.cv_znS[0].clone(),
                    m.cv_znS[1].clone(),
                    m.cv_tempv.clone().unwrap(),
                    m.cv_ftemp.clone().unwrap(),
                )
            };
            let retval = cvSensRhsWrapper(cv_mem, tn, &zn0, &zn1, &znS0, &znS1, &tempv, &ftemp);
            if retval < 0 {
                cvProcessError(
                    Some(cv_mem),
                    CV_SRHSFUNC_FAIL,
                    line!() as i32,
                    "CVode",
                    file!(),
                    &MSGCV_SRHSFUNC_FAILED(tn),
                );
                return CV_SRHSFUNC_FAIL;
            }
            if retval > 0 {
                cvProcessError(
                    Some(cv_mem),
                    CV_FIRST_SRHSFUNC_ERR,
                    line!() as i32,
                    "CVode",
                    file!(),
                    MSGCV_SRHSFUNC_FIRST,
                );
                return CV_FIRST_SRHSFUNC_ERR;
            }
        }

        if cv_mem.borrow().cv_quadr_sensi {
            let (Ns, znS0, znQ1, znQS1, tempv, tempvQ) = {
                let m = cv_mem.borrow();
                (
                    m.cv_Ns,
                    m.cv_znS[0].clone(),
                    m.cv_znQ[1].clone().unwrap(),
                    m.cv_znQS[1].clone(),
                    m.cv_tempv.clone().unwrap(),
                    m.cv_tempvQ.clone().unwrap(),
                )
            };
            let retval = cv_call_fQS(cv_mem, Ns, tn, &zn0, &znS0, &znQ1, &znQS1, &tempv, &tempvQ);
            cv_mem.borrow_mut().cv_nfQSe += 1;
            if retval < 0 {
                cvProcessError(
                    Some(cv_mem),
                    CV_QSRHSFUNC_FAIL,
                    line!() as i32,
                    "CVode",
                    file!(),
                    &MSGCV_QSRHSFUNC_FAILED(tn),
                );
                return CV_QSRHSFUNC_FAIL;
            }
            if retval > 0 {
                cvProcessError(
                    Some(cv_mem),
                    CV_FIRST_QSRHSFUNC_ERR,
                    line!() as i32,
                    "CVode",
                    file!(),
                    MSGCV_QSRHSFUNC_FIRST,
                );
                return CV_FIRST_QSRHSFUNC_ERR;
            }
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

            /*
             * Scale zn[1] by h.
             * If computing any quadratures, scale znQ[1] by h.
             * If computing sensitivities,  scale znS[1][is] by h.
             * If computing quadrature sensitivities,  scale znQS[1][is] by h.
             */

            m.cv_hscale = m.cv_h;
            m.cv_h0u = m.cv_h;
            m.cv_hprime = m.cv_h;
        }

        let h = cv_mem.borrow().cv_h;

        N_VScale(h, &zn1, &zn1);

        if cv_mem.borrow().cv_quadr {
            let znQ1 = cv_mem.borrow().cv_znQ[1].clone().unwrap();
            N_VScale(h, &znQ1, &znQ1);
        }

        if cv_mem.borrow().cv_sensi {
            let (Ns, cvals, znS1) = {
                let mut m = cv_mem.borrow_mut();
                let Ns = m.cv_Ns;
                for is in 0..Ns as usize {
                    m.cv_cvals[is] = h;
                }
                let cvals = m.cv_cvals.clone();
                let znS1 = m.cv_znS[1].clone();
                (Ns, cvals, znS1)
            };

            let retval = N_VScaleVectorArray(Ns, &cvals, &znS1, &znS1);
            if retval != CV_SUCCESS {
                return CV_VECTOROP_ERR;
            }
        }

        if cv_mem.borrow().cv_quadr_sensi {
            let (Ns, cvals, znQS1) = {
                let mut m = cv_mem.borrow_mut();
                let Ns = m.cv_Ns;
                for is in 0..Ns as usize {
                    m.cv_cvals[is] = h;
                }
                let cvals = m.cv_cvals.clone();
                let znQS1 = m.cv_znQS[1].clone();
                (Ns, cvals, znQS1)
            };

            let retval = N_VScaleVectorArray(Ns, &cvals, &znQS1, &znQS1);
            if retval != CV_SUCCESS {
                return CV_VECTOROP_ERR;
            }
        }

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
        let ier = crate::cvodes_nls::cvNlsInit(cv_mem);
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

        /* Reset and check ewt, ewtQ, ewtS */
        if cv_mem.borrow().cv_nst > 0 {
            let (zn0, ewt) = {
                let m = cv_mem.borrow();
                (m.cv_zn[0].clone().unwrap(), m.cv_ewt.clone().unwrap())
            };
            let ier = cv_call_efun(cv_mem, &zn0, &ewt);
            if ier != 0 {
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

            let (quadr, errconQ) = {
                let m = cv_mem.borrow();
                (m.cv_quadr, m.cv_errconQ)
            };
            if quadr && errconQ {
                let (znQ0, ewtQ) = {
                    let m = cv_mem.borrow();
                    (m.cv_znQ[0].clone().unwrap(), m.cv_ewtQ.clone().unwrap())
                };
                let ier = cvQuadEwtSet(cv_mem, &znQ0, &ewtQ);
                if ier != 0 {
                    let tn = cv_mem.borrow().cv_tn;
                    cvProcessError(
                        Some(cv_mem),
                        CV_ILL_INPUT,
                        line!() as i32,
                        "CVode",
                        file!(),
                        &MSGCV_EWTQ_NOW_BAD(tn),
                    );
                    istate = CV_ILL_INPUT;
                    cv_mem.borrow_mut().cv_tretlast = tn;
                    *tret = tn;
                    N_VScale(ONE, &zn0, yout);
                    break;
                }
            }

            if cv_mem.borrow().cv_sensi {
                let (znS0, ewtS) = {
                    let m = cv_mem.borrow();
                    (m.cv_znS[0].clone(), m.cv_ewtS.clone())
                };
                let ier = cvSensEwtSet(cv_mem, &znS0, &ewtS);
                if ier != 0 {
                    let tn = cv_mem.borrow().cv_tn;
                    cvProcessError(
                        Some(cv_mem),
                        CV_ILL_INPUT,
                        line!() as i32,
                        "CVode",
                        file!(),
                        &MSGCV_EWTS_NOW_BAD(tn),
                    );
                    istate = CV_ILL_INPUT;
                    cv_mem.borrow_mut().cv_tretlast = tn;
                    *tret = tn;
                    N_VScale(ONE, &zn0, yout);
                    break;
                }
            }

            let (quadr_sensi, errconQS) = {
                let m = cv_mem.borrow();
                (m.cv_quadr_sensi, m.cv_errconQS)
            };
            if quadr_sensi && errconQS {
                let (znQS0, ewtQS) = {
                    let m = cv_mem.borrow();
                    (m.cv_znQS[0].clone(), m.cv_ewtQS.clone())
                };
                let ier = cvQuadSensEwtSet(cv_mem, &znQS0, &ewtQS);
                if ier != 0 {
                    let tn = cv_mem.borrow().cv_tn;
                    cvProcessError(
                        Some(cv_mem),
                        CV_ILL_INPUT,
                        line!() as i32,
                        "CVode",
                        file!(),
                        &MSGCV_EWTQS_NOW_BAD(tn),
                    );
                    istate = CV_ILL_INPUT;
                    cv_mem.borrow_mut().cv_tretlast = tn;
                    *tret = tn;
                    N_VScale(ONE, &zn0, yout);
                    break;
                }
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
            let (zn0, ewt) = {
                let m = cv_mem.borrow();
                (m.cv_zn[0].clone().unwrap(), m.cv_ewt.clone().unwrap())
            };
            let mut nrm = N_VWrmsNorm(&zn0, &ewt);
            let (quadr, errconQ) = {
                let m = cv_mem.borrow();
                (m.cv_quadr, m.cv_errconQ)
            };
            if quadr && errconQ {
                let (znQ0, ewtQ) = {
                    let m = cv_mem.borrow();
                    (m.cv_znQ[0].clone().unwrap(), m.cv_ewtQ.clone().unwrap())
                };
                nrm = cvQuadUpdateNorm(cv_mem, nrm, &znQ0, &ewtQ);
            }
            let (sensi, errconS) = {
                let m = cv_mem.borrow();
                (m.cv_sensi, m.cv_errconS)
            };
            if sensi && errconS {
                let (znS0, ewtS) = {
                    let m = cv_mem.borrow();
                    (m.cv_znS[0].clone(), m.cv_ewtS.clone())
                };
                nrm = cvSensUpdateNorm(cv_mem, nrm, &znS0, &ewtS);
            }
            let (quadr_sensi, errconQS) = {
                let m = cv_mem.borrow();
                (m.cv_quadr_sensi, m.cv_errconQS)
            };
            if quadr_sensi && errconQS {
                let (znQS0, ewtQS) = {
                    let m = cv_mem.borrow();
                    (m.cv_znQS[0].clone(), m.cv_ewtQS.clone())
                };
                nrm = cvQuadSensUpdateNorm(cv_mem, nrm, &znQS0, &ewtQS);
            }
            let uround = cv_mem.borrow().cv_uround;
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

    /* Load optional output */
    {
        let (sensi, ism) = {
            let m = cv_mem.borrow();
            (m.cv_sensi, m.cv_ism)
        };
        if sensi && (ism == CV_STAGGERED1) {
            let mut m = cv_mem.borrow_mut();
            m.cv_nniS = 0;
            m.cv_nnfS = 0;
            m.cv_ncfnS = 0;
            for is in 0..m.cv_Ns as usize {
                let (nniS1, nnfS1, ncfnS1) = (m.cv_nniS1[is], m.cv_nnfS1[is], m.cv_ncfnS1[is]);
                m.cv_nniS += nniS1;
                m.cv_nnfS += nnfS1;
                m.cv_ncfnS += ncfnS1;
            }
        }
    }

    istate
}

/* =====================================================================
 * cvodes.c FRAGMENT B — every function whose definition starts in
 * lines 3600..7200 of `src/cvodes/cvodes.c`.
 *
 * Concatenated into `cvodes.rs`; imports and module-scope constants come
 * from the concatenation target (`crate::cvodes_impl::*`,
 * `sundials_core::sundials_{math,nvector,nonlinearsolver,types,errors}`).
 *
 * Reference build: SUNDIALS_LOGGING_LEVEL = 2 (SUNLogInfo/SUNLogDebug/
 * SUNLogExtraDebug* omitted at translation time), profiling off, error
 * checks off, monitoring ON, fused kernels OFF, serial branches only.
 * =====================================================================*/

/*
 * =================================================================
 * Callback invocation helpers (granular borrow discipline: the box
 * token is taken out of the mem around every user callback call and
 * restored on every path).  Named `cvb_*` so this fragment never
 * collides with identically-shaped helpers in a sibling fragment; the
 * integrator may dedupe them into a single set.
 * =================================================================
 */

/// Invoke the user RHS `f` (C: `cv_mem->cv_f(t, y, ydot, cv_mem->cv_user_data)`).
fn cvb_call_f(cv_mem: &CVodeMem, t: sunrealtype, y: &N_Vector, ydot: &N_Vector) -> i32 {
    let f = cv_mem.borrow().cv_f.expect("cv_f set");
    let mut user_data = cv_mem.borrow_mut().cv_user_data.take();
    let retval = f(t, y, ydot, &mut user_data);
    cv_mem.borrow_mut().cv_user_data = user_data;
    retval
}

/// Invoke the quadrature RHS `fQ`
/// (C: `cv_mem->cv_fQ(t, y, yQdot, cv_mem->cv_user_data)`).
fn cvb_call_fQ(cv_mem: &CVodeMem, t: sunrealtype, y: &N_Vector, yQdot: &N_Vector) -> i32 {
    let fQ = cv_mem.borrow().cv_fQ.expect("cv_fQ set");
    let mut user_data = cv_mem.borrow_mut().cv_user_data.take();
    let retval = fQ(t, y, yQdot, &mut user_data);
    cv_mem.borrow_mut().cv_user_data = user_data;
    retval
}

/// Invoke the error-weight function
/// (C: `cv_mem->cv_efun(ycur, weight, cv_mem->cv_e_data)`).
///
/// C aliases `cv_e_data` with `cv_user_data` when the user supplied
/// `efun` and with `cv_mem` otherwise (`cvInitialSetup`).  Box aliasing
/// is impossible in safe Rust, so the user-efun case passes the CURRENT
/// `cv_user_data` (accepted deviation class 6) and the default case
/// passes the module-owned `cv_e_data` token.
fn cvb_call_efun(cv_mem: &CVodeMem, ycur: &N_Vector, weight: &N_Vector) -> i32 {
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

/// Invoke the quadrature-sensitivity RHS `fQS` (C:
/// `cv_mem->cv_fQS(Ns, t, y, yS, yQdot, yQSdot, cv_mem->cv_fQS_data, tmp, tmpQ)`).
///
/// `cv_fQS_data` is `Some(token)` when CVODES uses its internal DQ
/// routine (C stored `cv_mem` there) and `None` when C stored
/// `cv_user_data`; the `None` case therefore forwards the integrator's
/// `cv_user_data` box.
#[allow(clippy::too_many_arguments)]
fn cvb_call_fQS(
    cv_mem: &CVodeMem,
    Ns: i32,
    t: sunrealtype,
    y: &N_Vector,
    yS: &[N_Vector],
    yQdot: &N_Vector,
    yQSdot: &[N_Vector],
    tmp: &N_Vector,
    tmpQ: &N_Vector,
) -> i32 {
    let fQS = cv_mem.borrow().cv_fQS.expect("cv_fQS set");
    let mut token = cv_mem.borrow_mut().cv_fQS_data.take();
    if token.is_some() {
        let retval = fQS(Ns, t, y, yS, yQdot, yQSdot, &mut token, tmp, tmpQ);
        cv_mem.borrow_mut().cv_fQS_data = token;
        retval
    } else {
        let mut user_data = cv_mem.borrow_mut().cv_user_data.take();
        let retval = fQS(Ns, t, y, yS, yQdot, yQSdot, &mut user_data, tmp, tmpQ);
        let mut m = cv_mem.borrow_mut();
        m.cv_user_data = user_data;
        m.cv_fQS_data = token.take();
        retval
    }
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
 * CVodeComputeStateSens
 *
 * Computes yS based on the current prediction and given correction.
 */

pub fn CVodeComputeStateSens(cvode_mem: &CVodeMem, ycorS: &[N_Vector], yS: &[N_Vector]) -> i32 {
    /* NULL-mem check: handled by type system */
    let cv_mem = cvode_mem;

    let (Ns, znS0) = {
        let m = cv_mem.borrow();
        (m.cv_Ns, m.cv_znS[0].clone())
    };

    let retval = N_VLinearSumVectorArray(Ns, ONE, &znS0, ONE, ycorS, yS);
    if retval != CV_SUCCESS {
        return CV_VECTOROP_ERR;
    }

    CV_SUCCESS
}

/*
 * CVodeComputeStateSens1
 *
 * Computes yS[idx] based on the current prediction and given correction.
 */

pub fn CVodeComputeStateSens1(
    cvode_mem: &CVodeMem,
    idx: i32,
    ycorS1: &N_Vector,
    yS1: &N_Vector,
) -> i32 {
    /* NULL-mem check: handled by type system */
    let cv_mem = cvode_mem;

    let znS0idx = cv_mem.borrow().cv_znS[0][idx as usize].clone();
    N_VLinearSum(ONE, &znS0idx, ONE, ycorS1, yS1);

    CV_SUCCESS
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
 * CVodeGetQuad
 *
 * This routine extracts quadrature solution into yQout at the
 * time which CVode returned the solution.
 * This is just a wrapper that calls CVodeGetQuadDky with k=0.
 */

pub fn CVodeGetQuad(cvode_mem: &CVodeMem, tret: &mut sunrealtype, yQout: &N_Vector) -> i32 {
    /* NULL-mem check: handled by type system */
    let cv_mem = cvode_mem;

    let tretlast = cv_mem.borrow().cv_tretlast;
    *tret = tretlast;

    CVodeGetQuadDky(cvode_mem, tretlast, 0, yQout)
}

/*
 * CVodeGetQuadDky
 *
 * CVodeQuadDky computes the kth derivative of the yQ function at
 * time t, where tn-hu <= t <= tn, tn denotes the current
 * internal time reached, and hu is the last internal step size
 * successfully used by the solver. The user may request
 * k=0, 1, ..., qu, where qu is the current order.
 * The derivative vector is returned in dky. This vector
 * must be allocated by the caller. It is only legal to call this
 * function after a successful return from CVode with quadrature
 * computation enabled.
 */

pub fn CVodeGetQuadDky(cvode_mem: &CVodeMem, t: sunrealtype, k: i32, dkyQ: &N_Vector) -> i32 {
    /* Check all inputs for legality */

    /* NULL-mem check: handled by type system */
    let cv_mem = cvode_mem;

    if cv_mem.borrow().cv_quadr != SUNTRUE {
        cvProcessError(
            Some(cv_mem),
            CV_NO_QUAD,
            line!() as i32,
            "CVodeGetQuadDky",
            file!(),
            MSGCV_NO_QUAD,
        );
        return CV_NO_QUAD;
    }

    /* NULL dkyQ check: handled by type system */

    if (k < 0) || (k > cv_mem.borrow().cv_q) {
        cvProcessError(
            Some(cv_mem),
            CV_BAD_K,
            line!() as i32,
            "CVodeGetQuadDky",
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
        /* C passes MSGCV_BAD_T with no varargs here (the format string
        expects three); the printed text is indeterminate in C.  The port
        supplies the same three values CVodeGetDky uses.  Accepted
        deviation class 5 (C UB -> deterministic behavior); error path
        only, reachable solely with an out-of-range t. */
        cvProcessError(
            Some(cv_mem),
            CV_BAD_T,
            line!() as i32,
            "CVodeGetQuadDky",
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
            Xvecs.push(m.cv_znQ[j as usize].clone().unwrap());
            nvec += 1;
            j -= 1;
        }
    }
    let ier = N_VLinearCombination(nvec as i32, &cvals, &Xvecs, dkyQ);
    if ier != CV_SUCCESS {
        return CV_VECTOROP_ERR;
    }

    if k == 0 {
        return CV_SUCCESS;
    }
    let r = SUNRpowerI(h, -k);
    N_VScale(r, dkyQ, dkyQ);

    CV_SUCCESS
}

/*
 * CVodeGetSens
 *
 * This routine extracts sensitivity solution into ySout at the
 * time at which CVode returned the solution.
 * This is just a wrapper that calls CVodeSensDky with k=0.
 */

pub fn CVodeGetSens(cvode_mem: &CVodeMem, tret: &mut sunrealtype, ySout: &[N_Vector]) -> i32 {
    /* NULL-mem check: handled by type system */
    let cv_mem = cvode_mem;

    let tretlast = cv_mem.borrow().cv_tretlast;
    *tret = tretlast;

    CVodeGetSensDky(cvode_mem, tretlast, 0, ySout)
}

/*
 * CVodeGetSens1
 *
 * This routine extracts the is-th sensitivity solution into ySout
 * at the time at which CVode returned the solution.
 * This is just a wrapper that calls CVodeSensDky1 with k=0.
 */

pub fn CVodeGetSens1(
    cvode_mem: &CVodeMem,
    tret: &mut sunrealtype,
    is: i32,
    ySout: &N_Vector,
) -> i32 {
    /* NULL-mem check: handled by type system */
    let cv_mem = cvode_mem;

    let tretlast = cv_mem.borrow().cv_tretlast;
    *tret = tretlast;

    CVodeGetSensDky1(cvode_mem, tretlast, 0, is, ySout)
}

/*
 * CVodeGetSensDky
 *
 * If the user calls directly CVodeSensDky then s must be allocated
 * prior to this call. When CVodeSensDky is called by
 * CVodeGetSens, only ier=CV_SUCCESS, ier=CV_NO_SENS, or
 * ier=CV_BAD_T are possible.
 */

pub fn CVodeGetSensDky(cvode_mem: &CVodeMem, t: sunrealtype, k: i32, dkyS: &[N_Vector]) -> i32 {
    let mut ier = CV_SUCCESS;

    /* NULL-mem check: handled by type system */
    let cv_mem = cvode_mem;

    /* NULL dkyS check: handled by type system */

    let Ns = cv_mem.borrow().cv_Ns;
    for is in 0..Ns {
        ier = CVodeGetSensDky1(cvode_mem, t, k, is, &dkyS[is as usize]);
        if ier != CV_SUCCESS {
            break;
        }
    }

    ier
}

/*
 * CVodeGetSensDky1
 *
 * CVodeSensDky1 computes the kth derivative of the yS[is] function at
 * time t, where tn-hu <= t <= tn, tn denotes the current
 * internal time reached, and hu is the last internal step size
 * successfully used by the solver. The user may request
 * is=0, 1, ..., Ns-1 and k=0, 1, ..., qu, where qu is the current
 * order. The derivative vector is returned in dky. This vector
 * must be allocated by the caller. It is only legal to call this
 * function after a successful return from CVode with sensitivity
 * computation enabled.
 */

pub fn CVodeGetSensDky1(
    cvode_mem: &CVodeMem,
    t: sunrealtype,
    k: i32,
    is: i32,
    dkyS: &N_Vector,
) -> i32 {
    /* Check all inputs for legality */

    /* NULL-mem check: handled by type system */
    let cv_mem = cvode_mem;

    if cv_mem.borrow().cv_sensi != SUNTRUE {
        cvProcessError(
            Some(cv_mem),
            CV_NO_SENS,
            line!() as i32,
            "CVodeGetSensDky1",
            file!(),
            MSGCV_NO_SENSI,
        );
        return CV_NO_SENS;
    }

    /* NULL dkyS check: handled by type system */

    if (k < 0) || (k > cv_mem.borrow().cv_q) {
        cvProcessError(
            Some(cv_mem),
            CV_BAD_K,
            line!() as i32,
            "CVodeGetSensDky1",
            file!(),
            MSGCV_BAD_K,
        );
        return CV_BAD_K;
    }

    if (is < 0) || (is > cv_mem.borrow().cv_Ns - 1) {
        cvProcessError(
            Some(cv_mem),
            CV_BAD_IS,
            line!() as i32,
            "CVodeGetSensDky1",
            file!(),
            MSGCV_BAD_IS,
        );
        return CV_BAD_IS;
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
        /* C passes MSGCV_BAD_T with no varargs here; see CVodeGetQuadDky. */
        cvProcessError(
            Some(cv_mem),
            CV_BAD_T,
            line!() as i32,
            "CVodeGetSensDky1",
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
            Xvecs.push(m.cv_znS[j as usize][is as usize].clone());
            nvec += 1;
            j -= 1;
        }
    }
    let ier = N_VLinearCombination(nvec as i32, &cvals, &Xvecs, dkyS);
    if ier != CV_SUCCESS {
        return CV_VECTOROP_ERR;
    }

    if k == 0 {
        return CV_SUCCESS;
    }
    let r = SUNRpowerI(h, -k);
    N_VScale(r, dkyS, dkyS);

    CV_SUCCESS
}

/*
 * CVodeGetQuadSens and CVodeGetQuadSens1
 *
 * Extraction functions for all or only one of the quadrature sensitivity
 * vectors at the time at which CVode returned the ODE solution.
 */

pub fn CVodeGetQuadSens(cvode_mem: &CVodeMem, tret: &mut sunrealtype, yQSout: &[N_Vector]) -> i32 {
    /* NULL-mem check: handled by type system */
    let cv_mem = cvode_mem;

    let tretlast = cv_mem.borrow().cv_tretlast;
    *tret = tretlast;

    CVodeGetQuadSensDky(cvode_mem, tretlast, 0, yQSout)
}

pub fn CVodeGetQuadSens1(
    cvode_mem: &CVodeMem,
    tret: &mut sunrealtype,
    is: i32,
    yQSout: &N_Vector,
) -> i32 {
    /* NULL-mem check: handled by type system */
    let cv_mem = cvode_mem;

    let tretlast = cv_mem.borrow().cv_tretlast;
    *tret = tretlast;

    CVodeGetQuadSensDky1(cvode_mem, tretlast, 0, is, yQSout)
}

/*
 * CVodeGetQuadSensDky and CVodeGetQuadSensDky1
 *
 * Dense output functions for all or only one of the quadrature sensitivity
 * vectors (or derivative thereof).
 */

pub fn CVodeGetQuadSensDky(
    cvode_mem: &CVodeMem,
    t: sunrealtype,
    k: i32,
    dkyQS_all: &[N_Vector],
) -> i32 {
    let mut ier = CV_SUCCESS;

    /* NULL-mem check: handled by type system */
    let cv_mem = cvode_mem;

    /* NULL dkyQS_all check: handled by type system */

    let Ns = cv_mem.borrow().cv_Ns;
    for is in 0..Ns {
        ier = CVodeGetQuadSensDky1(cvode_mem, t, k, is, &dkyQS_all[is as usize]);
        if ier != CV_SUCCESS {
            break;
        }
    }

    ier
}

pub fn CVodeGetQuadSensDky1(
    cvode_mem: &CVodeMem,
    t: sunrealtype,
    k: i32,
    is: i32,
    dkyQS: &N_Vector,
) -> i32 {
    /* Check all inputs for legality */

    /* NULL-mem check: handled by type system */
    let cv_mem = cvode_mem;

    if cv_mem.borrow().cv_quadr_sensi != SUNTRUE {
        cvProcessError(
            Some(cv_mem),
            CV_NO_QUADSENS,
            line!() as i32,
            "CVodeGetQuadSensDky1",
            file!(),
            MSGCV_NO_QUADSENSI,
        );
        return CV_NO_QUADSENS;
    }

    /* NULL dkyQS check: handled by type system */

    if (k < 0) || (k > cv_mem.borrow().cv_q) {
        cvProcessError(
            Some(cv_mem),
            CV_BAD_K,
            line!() as i32,
            "CVodeGetQuadSensDky1",
            file!(),
            MSGCV_BAD_K,
        );
        return CV_BAD_K;
    }

    if (is < 0) || (is > cv_mem.borrow().cv_Ns - 1) {
        cvProcessError(
            Some(cv_mem),
            CV_BAD_IS,
            line!() as i32,
            "CVodeGetQuadSensDky1",
            file!(),
            MSGCV_BAD_IS,
        );
        return CV_BAD_IS;
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
        /* C passes MSGCV_BAD_T with no varargs here; see CVodeGetQuadDky. */
        cvProcessError(
            Some(cv_mem),
            CV_BAD_T,
            line!() as i32,
            "CVodeGetQuadSensDky1",
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
            Xvecs.push(m.cv_znQS[j as usize][is as usize].clone());
            nvec += 1;
            j -= 1;
        }
    }
    let ier = N_VLinearCombination(nvec as i32, &cvals, &Xvecs, dkyQS);
    if ier != CV_SUCCESS {
        return CV_VECTOROP_ERR;
    }

    if k == 0 {
        return CV_SUCCESS;
    }
    let r = SUNRpowerI(h, -k);
    N_VScale(r, dkyQS, dkyQS);

    CV_SUCCESS
}

/*
 * -----------------------------------------------------------------
 * Deallocation functions
 * -----------------------------------------------------------------
 */

/*
 * CVodeFree
 *
 * This routine frees the problem memory allocated by CVodeInit.
 * Such memory includes all the vectors allocated by cvAllocVectors,
 * and the memory lmem for the linear solver (deallocated by a call
 * to lfree), as well as (if Ns!=0) all memory allocated for
 * sensitivity computations by CVodeSensInit.
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

    CVodeQuadFree(&cv_mem);

    CVodeSensFree(&cv_mem);

    CVodeQuadSensFree(&cv_mem);

    crate::cvodea::CVodeAdjFree(&cv_mem);

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

    {
        let mut m = cv_mem.borrow_mut();
        m.cv_cvals = Vec::new();
        m.cv_Xvecs = Vec::new();
        m.cv_Zvecs = Vec::new();
    }

    if cv_mem.borrow().proj_mem.is_some() {
        /* cvProjFree: dropping the projection memory frees it */
        cv_mem.borrow_mut().proj_mem = None;
    }

    /* C frees the mem struct wholesale; the Rust handle is dropped by the
    caller, so break the Rc cycles the module-owned callback tokens create
    (cv_e_data / cv_fS_data / cv_fQS_data hold CVodeMem clones pointing
    back at this record) */
    {
        let mut m = cv_mem.borrow_mut();
        m.cv_e_data = None;
        m.cv_fS_data = None;
        m.cv_fQS_data = None;
    }

    *cvode_mem = None;
}

/*
 * CVodeQuadFree
 *
 * CVodeQuadFree frees the problem memory in cvode_mem allocated
 * for quadrature integration. Its only argument is the pointer
 * cvode_mem returned by CVodeCreate.
 */

pub fn CVodeQuadFree(cvode_mem: &CVodeMem) {
    /* NULL-mem check: handled by type system */
    let cv_mem = cvode_mem;

    if cv_mem.borrow().cv_QuadMallocDone {
        cvQuadFreeVectors(cv_mem);
        let mut m = cv_mem.borrow_mut();
        m.cv_QuadMallocDone = SUNFALSE;
        m.cv_quadr = SUNFALSE;
    }
}

/*
 * CVodeSensFree
 *
 * CVodeSensFree frees the problem memory in cvode_mem allocated
 * for sensitivity analysis. Its only argument is the pointer
 * cvode_mem returned by CVodeCreate.
 */

pub fn CVodeSensFree(cvode_mem: &CVodeMem) {
    /* NULL-mem check: handled by type system */
    let cv_mem = cvode_mem;

    if cv_mem.borrow().cv_SensMallocDone {
        if cv_mem.borrow().cv_stgr1alloc {
            let mut m = cv_mem.borrow_mut();
            m.cv_ncfS1 = Vec::new();
            m.cv_ncfnS1 = Vec::new();
            m.cv_nniS1 = Vec::new();
            m.cv_nnfS1 = Vec::new();
            m.cv_stgr1alloc = SUNFALSE;
        }
        cvSensFreeVectors(cv_mem);
        let mut m = cv_mem.borrow_mut();
        m.cv_SensMallocDone = SUNFALSE;
        m.cv_sensi = SUNFALSE;
    }

    /* free any vector wrappers */
    if cv_mem.borrow().simMallocDone {
        let (zn0Sim, ycorSim, ewtSim) = {
            let mut m = cv_mem.borrow_mut();
            let v = (m.zn0Sim.take(), m.ycorSim.take(), m.ewtSim.take());
            m.simMallocDone = SUNFALSE;
            v
        };
        if let Some(v) = zn0Sim {
            N_VDestroy(v);
        }
        if let Some(v) = ycorSim {
            N_VDestroy(v);
        }
        if let Some(v) = ewtSim {
            N_VDestroy(v);
        }
    }
    if cv_mem.borrow().stgMallocDone {
        let (zn0Stg, ycorStg, ewtStg) = {
            let mut m = cv_mem.borrow_mut();
            let v = (m.zn0Stg.take(), m.ycorStg.take(), m.ewtStg.take());
            m.stgMallocDone = SUNFALSE;
            v
        };
        if let Some(v) = zn0Stg {
            N_VDestroy(v);
        }
        if let Some(v) = ycorStg {
            N_VDestroy(v);
        }
        if let Some(v) = ewtStg {
            N_VDestroy(v);
        }
    }

    /* if CVODES created a NLS object then free it */
    if cv_mem.borrow().ownNLSsim {
        let nls = {
            let mut m = cv_mem.borrow_mut();
            m.ownNLSsim = SUNFALSE;
            m.NLSsim.take()
        };
        let _ = SUNNonlinSolFree(nls);
    }
    if cv_mem.borrow().ownNLSstg {
        let nls = {
            let mut m = cv_mem.borrow_mut();
            m.ownNLSstg = SUNFALSE;
            m.NLSstg.take()
        };
        let _ = SUNNonlinSolFree(nls);
    }
    if cv_mem.borrow().ownNLSstg1 {
        let nls = {
            let mut m = cv_mem.borrow_mut();
            m.ownNLSstg1 = SUNFALSE;
            m.NLSstg1.take()
        };
        let _ = SUNNonlinSolFree(nls);
    }

    /* free min atol array if necessary */
    if !cv_mem.borrow().cv_atolSmin0.is_empty() {
        cv_mem.borrow_mut().cv_atolSmin0 = Vec::new();
    }
}

/*
 * CVodeQuadSensFree
 *
 * CVodeQuadSensFree frees the problem memory in cvode_mem allocated
 * for quadrature sensitivity analysis. Its only argument is the pointer
 * cvode_mem returned by CVodeCreate.
 */

pub fn CVodeQuadSensFree(cvode_mem: &CVodeMem) {
    /* NULL-mem check: handled by type system */
    let cv_mem = cvode_mem;

    if cv_mem.borrow().cv_QuadSensMallocDone {
        cvQuadSensFreeVectors(cv_mem);
        let mut m = cv_mem.borrow_mut();
        m.cv_QuadSensMallocDone = SUNFALSE;
        m.cv_quadr_sensi = SUNFALSE;
    }

    /* free min atol array if necessary */
    if !cv_mem.borrow().cv_atolQSmin0.is_empty() {
        cv_mem.borrow_mut().cv_atolQSmin0 = Vec::new();
    }
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
 * This routine allocates the CVODES vectors ewt, acor, tempv, ftemp, and
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
 * CVodeQuadAllocVectors
 *
 * NOTE: Space for ewtQ is allocated even when errconQ=SUNFALSE,
 * although in this case, ewtQ is never used. The reason for this
 * decision is to allow the user to re-initialize the quadrature
 * computation with errconQ=SUNTRUE, after an initialization with
 * errconQ=SUNFALSE, without new memory allocation within
 * CVodeQuadReInit.
 */

fn cvQuadAllocVectors(cv_mem: &CVodeMem, tmpl: &N_Vector) -> sunbooleantype {
    /* Allocate ewtQ */
    let ewtQ = match N_VClone(tmpl) {
        Some(v) => v,
        None => return SUNFALSE,
    };

    /* Allocate acorQ */
    let acorQ = match N_VClone(tmpl) {
        Some(v) => v,
        None => {
            N_VDestroy(ewtQ);
            return SUNFALSE;
        }
    };

    /* Allocate yQ */
    let yQ = match N_VClone(tmpl) {
        Some(v) => v,
        None => {
            N_VDestroy(ewtQ);
            N_VDestroy(acorQ);
            return SUNFALSE;
        }
    };

    /* Allocate tempvQ */
    let tempvQ = match N_VClone(tmpl) {
        Some(v) => v,
        None => {
            N_VDestroy(ewtQ);
            N_VDestroy(acorQ);
            N_VDestroy(yQ);
            return SUNFALSE;
        }
    };

    {
        let mut m = cv_mem.borrow_mut();
        m.cv_ewtQ = Some(ewtQ);
        m.cv_acorQ = Some(acorQ);
        m.cv_yQ = Some(yQ);
        m.cv_tempvQ = Some(tempvQ);
    }

    /* Allocate zQn[0] ... zQn[maxord] */

    let qmax = cv_mem.borrow().cv_qmax;
    for j in 0..=qmax as usize {
        match N_VClone(tmpl) {
            Some(v) => {
                cv_mem.borrow_mut().cv_znQ[j] = Some(v);
            }
            None => {
                let mut m = cv_mem.borrow_mut();
                if let Some(v) = m.cv_ewtQ.take() {
                    N_VDestroy(v);
                }
                if let Some(v) = m.cv_acorQ.take() {
                    N_VDestroy(v);
                }
                if let Some(v) = m.cv_yQ.take() {
                    N_VDestroy(v);
                }
                if let Some(v) = m.cv_tempvQ.take() {
                    N_VDestroy(v);
                }
                for i in 0..j {
                    if let Some(v) = m.cv_znQ[i].take() {
                        N_VDestroy(v);
                    }
                }
                return SUNFALSE;
            }
        }
    }

    {
        let mut m = cv_mem.borrow_mut();

        /* Store the value of qmax used here */
        m.cv_qmax_allocQ = m.cv_qmax;

        /* Update solver workspace lengths */
        m.cv_lrw += (m.cv_qmax as i64 + 5) * m.cv_lrw1Q;
        m.cv_liw += (m.cv_qmax as i64 + 5) * m.cv_liw1Q;
    }

    SUNTRUE
}

/*
 * cvQuadFreeVectors
 *
 * This routine frees the CVODES vectors allocated in cvQuadAllocVectors.
 */

fn cvQuadFreeVectors(cv_mem: &CVodeMem) {
    let mut m = cv_mem.borrow_mut();

    let maxord = m.cv_qmax_allocQ;

    if let Some(v) = m.cv_ewtQ.take() {
        N_VDestroy(v);
    }
    if let Some(v) = m.cv_acorQ.take() {
        N_VDestroy(v);
    }
    if let Some(v) = m.cv_yQ.take() {
        N_VDestroy(v);
    }
    if let Some(v) = m.cv_tempvQ.take() {
        N_VDestroy(v);
    }

    for j in 0..=maxord as usize {
        if let Some(v) = m.cv_znQ[j].take() {
            N_VDestroy(v);
        }
    }

    m.cv_lrw -= (maxord as i64 + 5) * m.cv_lrw1Q;
    m.cv_liw -= (maxord as i64 + 5) * m.cv_liw1Q;

    if m.cv_VabstolQMallocDone {
        if let Some(v) = m.cv_VabstolQ.take() {
            N_VDestroy(v);
        }
        m.cv_lrw -= m.cv_lrw1Q;
        m.cv_liw -= m.cv_liw1Q;
    }

    m.cv_VabstolQMallocDone = SUNFALSE;
}

/*
 * cvSensAllocVectors
 *
 * Create (through duplication) N_Vectors used for sensitivity analysis,
 * using the N_Vector 'tmpl' as a template.
 */

fn cvSensAllocVectors(cv_mem: &CVodeMem, tmpl: &N_Vector) -> sunbooleantype {
    let (Ns, qmax) = {
        let m = cv_mem.borrow();
        (m.cv_Ns, m.cv_qmax)
    };

    /* Allocate yS */
    let yS = match N_VCloneVectorArray(Ns, tmpl) {
        Some(v) => v,
        None => return SUNFALSE,
    };

    /* Allocate ewtS */
    let ewtS = match N_VCloneVectorArray(Ns, tmpl) {
        Some(v) => v,
        None => {
            N_VDestroyVectorArray(yS, Ns);
            return SUNFALSE;
        }
    };

    /* Allocate acorS */
    let acorS = match N_VCloneVectorArray(Ns, tmpl) {
        Some(v) => v,
        None => {
            N_VDestroyVectorArray(yS, Ns);
            N_VDestroyVectorArray(ewtS, Ns);
            return SUNFALSE;
        }
    };

    /* Allocate tempvS */
    let tempvS = match N_VCloneVectorArray(Ns, tmpl) {
        Some(v) => v,
        None => {
            N_VDestroyVectorArray(yS, Ns);
            N_VDestroyVectorArray(ewtS, Ns);
            N_VDestroyVectorArray(acorS, Ns);
            return SUNFALSE;
        }
    };

    /* Allocate ftempS */
    let ftempS = match N_VCloneVectorArray(Ns, tmpl) {
        Some(v) => v,
        None => {
            N_VDestroyVectorArray(yS, Ns);
            N_VDestroyVectorArray(ewtS, Ns);
            N_VDestroyVectorArray(acorS, Ns);
            N_VDestroyVectorArray(tempvS, Ns);
            return SUNFALSE;
        }
    };

    /* Allocate znS */
    let mut znS: Vec<Vec<N_Vector>> = Vec::new();
    for _j in 0..=qmax as usize {
        match N_VCloneVectorArray(Ns, tmpl) {
            Some(v) => znS.push(v),
            None => {
                N_VDestroyVectorArray(yS, Ns);
                N_VDestroyVectorArray(ewtS, Ns);
                N_VDestroyVectorArray(acorS, Ns);
                N_VDestroyVectorArray(tempvS, Ns);
                N_VDestroyVectorArray(ftempS, Ns);
                for zj in znS {
                    N_VDestroyVectorArray(zj, Ns);
                }
                return SUNFALSE;
            }
        }
    }

    {
        let mut m = cv_mem.borrow_mut();
        m.cv_yS = yS;
        m.cv_ewtS = ewtS;
        m.cv_acorS = acorS;
        m.cv_tempvS = tempvS;
        m.cv_ftempS = ftempS;
        for (j, zj) in znS.into_iter().enumerate() {
            m.cv_znS[j] = zj;
        }

        /* Allocate space for pbar and plist (C `malloc` cannot fail here in
        safe Rust; the C failure branches are unreachable) */
        m.cv_pbar = vec![ZERO; Ns as usize];
        m.cv_plist = vec![0i32; Ns as usize];

        /* Update solver workspace lengths */
        m.cv_lrw += (m.cv_qmax as i64 + 6) * Ns as i64 * m.cv_lrw1 + Ns as i64;
        m.cv_liw += (m.cv_qmax as i64 + 6) * Ns as i64 * m.cv_liw1 + Ns as i64;

        /* Store the value of qmax used here */
        m.cv_qmax_allocS = m.cv_qmax;
    }

    SUNTRUE
}

/*
 * cvSensFreeVectors
 *
 * This routine frees the CVODES vectors allocated in cvSensAllocVectors.
 */

fn cvSensFreeVectors(cv_mem: &CVodeMem) {
    let mut m = cv_mem.borrow_mut();

    let maxord = m.cv_qmax_allocS;
    let Ns = m.cv_Ns;

    N_VDestroyVectorArray(std::mem::take(&mut m.cv_yS), Ns);
    N_VDestroyVectorArray(std::mem::take(&mut m.cv_ewtS), Ns);
    N_VDestroyVectorArray(std::mem::take(&mut m.cv_acorS), Ns);
    N_VDestroyVectorArray(std::mem::take(&mut m.cv_tempvS), Ns);
    N_VDestroyVectorArray(std::mem::take(&mut m.cv_ftempS), Ns);

    for j in 0..=maxord as usize {
        N_VDestroyVectorArray(std::mem::take(&mut m.cv_znS[j]), Ns);
    }

    m.cv_pbar = Vec::new();
    m.cv_plist = Vec::new();

    m.cv_lrw -= (maxord as i64 + 6) * Ns as i64 * m.cv_lrw1 + Ns as i64;
    m.cv_liw -= (maxord as i64 + 6) * Ns as i64 * m.cv_liw1 + Ns as i64;

    if m.cv_VabstolSMallocDone {
        N_VDestroyVectorArray(std::mem::take(&mut m.cv_VabstolS), Ns);
        m.cv_lrw -= Ns as i64 * m.cv_lrw1;
        m.cv_liw -= Ns as i64 * m.cv_liw1;
    }
    if m.cv_SabstolSMallocDone {
        m.cv_SabstolS = Vec::new();
        m.cv_lrw -= Ns as i64;
    }
    m.cv_VabstolSMallocDone = SUNFALSE;
    m.cv_SabstolSMallocDone = SUNFALSE;
}

/*
 * cvQuadSensAllocVectors
 *
 * Create (through duplication) N_Vectors used for quadrature sensitivity
 * analysis, using the N_Vector 'tmpl' as a template.
 */

fn cvQuadSensAllocVectors(cv_mem: &CVodeMem, tmpl: &N_Vector) -> sunbooleantype {
    let (Ns, qmax) = {
        let m = cv_mem.borrow();
        (m.cv_Ns, m.cv_qmax)
    };

    /* Allocate ftempQ */
    let ftempQ = match N_VClone(tmpl) {
        Some(v) => v,
        None => return SUNFALSE,
    };

    /* Allocate yQS */
    let yQS = match N_VCloneVectorArray(Ns, tmpl) {
        Some(v) => v,
        None => {
            N_VDestroy(ftempQ);
            return SUNFALSE;
        }
    };

    /* Allocate ewtQS */
    let ewtQS = match N_VCloneVectorArray(Ns, tmpl) {
        Some(v) => v,
        None => {
            N_VDestroy(ftempQ);
            N_VDestroyVectorArray(yQS, Ns);
            return SUNFALSE;
        }
    };

    /* Allocate acorQS */
    let acorQS = match N_VCloneVectorArray(Ns, tmpl) {
        Some(v) => v,
        None => {
            N_VDestroy(ftempQ);
            N_VDestroyVectorArray(yQS, Ns);
            N_VDestroyVectorArray(ewtQS, Ns);
            return SUNFALSE;
        }
    };

    /* Allocate tempvQS */
    let tempvQS = match N_VCloneVectorArray(Ns, tmpl) {
        Some(v) => v,
        None => {
            N_VDestroy(ftempQ);
            N_VDestroyVectorArray(yQS, Ns);
            N_VDestroyVectorArray(ewtQS, Ns);
            N_VDestroyVectorArray(acorQS, Ns);
            return SUNFALSE;
        }
    };

    /* Allocate znQS */
    let mut znQS: Vec<Vec<N_Vector>> = Vec::new();
    for _j in 0..=qmax as usize {
        match N_VCloneVectorArray(Ns, tmpl) {
            Some(v) => znQS.push(v),
            None => {
                N_VDestroy(ftempQ);
                N_VDestroyVectorArray(yQS, Ns);
                N_VDestroyVectorArray(ewtQS, Ns);
                N_VDestroyVectorArray(acorQS, Ns);
                N_VDestroyVectorArray(tempvQS, Ns);
                for zj in znQS {
                    N_VDestroyVectorArray(zj, Ns);
                }
                return SUNFALSE;
            }
        }
    }

    {
        let mut m = cv_mem.borrow_mut();
        m.cv_ftempQ = Some(ftempQ);
        m.cv_yQS = yQS;
        m.cv_ewtQS = ewtQS;
        m.cv_acorQS = acorQS;
        m.cv_tempvQS = tempvQS;
        for (j, zj) in znQS.into_iter().enumerate() {
            m.cv_znQS[j] = zj;
        }

        /* Update solver workspace lengths */
        m.cv_lrw += (m.cv_qmax as i64 + 5) * Ns as i64 * m.cv_lrw1Q;
        m.cv_liw += (m.cv_qmax as i64 + 5) * Ns as i64 * m.cv_liw1Q;

        /* Store the value of qmax used here */
        m.cv_qmax_allocQS = m.cv_qmax;
    }

    SUNTRUE
}

/*
 * cvQuadSensFreeVectors
 *
 * This routine frees the CVODES vectors allocated in cvQuadSensAllocVectors.
 */

fn cvQuadSensFreeVectors(cv_mem: &CVodeMem) {
    let mut m = cv_mem.borrow_mut();

    let maxord = m.cv_qmax_allocQS;
    let Ns = m.cv_Ns;

    if let Some(v) = m.cv_ftempQ.take() {
        N_VDestroy(v);
    }

    N_VDestroyVectorArray(std::mem::take(&mut m.cv_yQS), Ns);
    N_VDestroyVectorArray(std::mem::take(&mut m.cv_ewtQS), Ns);
    N_VDestroyVectorArray(std::mem::take(&mut m.cv_acorQS), Ns);
    N_VDestroyVectorArray(std::mem::take(&mut m.cv_tempvQS), Ns);

    for j in 0..=maxord as usize {
        N_VDestroyVectorArray(std::mem::take(&mut m.cv_znQS[j]), Ns);
    }

    m.cv_lrw -= (maxord as i64 + 5) * Ns as i64 * m.cv_lrw1Q;
    m.cv_liw -= (maxord as i64 + 5) * Ns as i64 * m.cv_liw1Q;

    if m.cv_VabstolQSMallocDone {
        N_VDestroyVectorArray(std::mem::take(&mut m.cv_VabstolQS), Ns);
        m.cv_lrw -= Ns as i64 * m.cv_lrw1Q;
        m.cv_liw -= Ns as i64 * m.cv_liw1Q;
    }
    if m.cv_SabstolQSMallocDone {
        m.cv_SabstolQS = Vec::new();
        m.cv_lrw -= Ns as i64;
    }
    m.cv_VabstolQSMallocDone = SUNFALSE;
    m.cv_SabstolQSMallocDone = SUNFALSE;
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
        let (sensi, ism) = {
            let m = cv_mem.borrow();
            (m.cv_sensi, m.cv_ism)
        };
        if sensi && (ism == CV_SIMULTANEOUS) {
            cvProcessError(
                Some(cv_mem),
                CV_ILL_INPUT,
                line!() as i32,
                "cvInitialSetup",
                file!(),
                MSGCV_BAD_ISM_CONSTR,
            );
            return CV_ILL_INPUT;
        }

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
    let ier = cvb_call_efun(cv_mem, &zn0, &ewt);
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

    /* Quadrature initial setup */

    let (quadr, errconQ) = {
        let m = cv_mem.borrow();
        (m.cv_quadr, m.cv_errconQ)
    };
    if quadr && errconQ {
        /* Did the user specify tolerances? */
        if cv_mem.borrow().cv_itolQ == CV_NN {
            cvProcessError(
                Some(cv_mem),
                CV_ILL_INPUT,
                line!() as i32,
                "cvInitialSetup",
                file!(),
                MSGCV_NO_TOLQ,
            );
            return CV_ILL_INPUT;
        }

        /* Load ewtQ */
        let (znQ0, ewtQ) = {
            let m = cv_mem.borrow();
            (m.cv_znQ[0].clone().unwrap(), m.cv_ewtQ.clone().unwrap())
        };
        let ier = cvQuadEwtSet(cv_mem, &znQ0, &ewtQ);
        if ier != 0 {
            cvProcessError(
                Some(cv_mem),
                CV_ILL_INPUT,
                line!() as i32,
                "cvInitialSetup",
                file!(),
                MSGCV_BAD_EWTQ,
            );
            return CV_ILL_INPUT;
        }
    }

    if !quadr {
        cv_mem.borrow_mut().cv_errconQ = SUNFALSE;
    }

    /* Forward sensitivity initial setup */

    if cv_mem.borrow().cv_sensi {
        /* Did the user specify tolerances? */
        if cv_mem.borrow().cv_itolS == CV_NN {
            cvProcessError(
                Some(cv_mem),
                CV_ILL_INPUT,
                line!() as i32,
                "cvInitialSetup",
                file!(),
                MSGCV_NO_TOLS,
            );
            return CV_ILL_INPUT;
        }

        /* If using the internal DQ functions, we must have access to the
        problem parameters */
        let (fSDQ, p_is_null) = {
            let m = cv_mem.borrow();
            (m.cv_fSDQ, m.cv_p.is_none())
        };
        if fSDQ && p_is_null {
            cvProcessError(
                Some(cv_mem),
                CV_ILL_INPUT,
                line!() as i32,
                "cvInitialSetup",
                file!(),
                MSGCV_NULL_P,
            );
            return CV_ILL_INPUT;
        }

        /* Load ewtS */
        let (znS0, ewtS) = {
            let m = cv_mem.borrow();
            (m.cv_znS[0].clone(), m.cv_ewtS.clone())
        };
        let ier = cvSensEwtSet(cv_mem, &znS0, &ewtS);
        if ier != 0 {
            cvProcessError(
                Some(cv_mem),
                CV_ILL_INPUT,
                line!() as i32,
                "cvInitialSetup",
                file!(),
                MSGCV_BAD_EWTS,
            );
            return CV_ILL_INPUT;
        }
    }

    /* FSA of quadrature variables */

    if cv_mem.borrow().cv_quadr_sensi {
        /* If using the internal DQ functions, we must have access to fQ
         * (i.e. quadrature integration must be enabled) and to the problem
         * parameters */

        if cv_mem.borrow().cv_fQSDQ {
            /* Test if quadratures are defined, so we can use fQ */
            if !cv_mem.borrow().cv_quadr {
                cvProcessError(
                    Some(cv_mem),
                    CV_ILL_INPUT,
                    line!() as i32,
                    "cvInitialSetup",
                    file!(),
                    MSGCV_NULL_FQ,
                );
                return CV_ILL_INPUT;
            }

            /* Test if we have the problem parameters */
            if cv_mem.borrow().cv_p.is_none() {
                cvProcessError(
                    Some(cv_mem),
                    CV_ILL_INPUT,
                    line!() as i32,
                    "cvInitialSetup",
                    file!(),
                    MSGCV_NULL_P,
                );
                return CV_ILL_INPUT;
            }
        }

        if cv_mem.borrow().cv_errconQS {
            /* Did the user specify tolerances? */
            if cv_mem.borrow().cv_itolQS == CV_NN {
                cvProcessError(
                    Some(cv_mem),
                    CV_ILL_INPUT,
                    line!() as i32,
                    "cvInitialSetup",
                    file!(),
                    MSGCV_NO_TOLQS,
                );
                return CV_ILL_INPUT;
            }

            /* If needed, did the user provide quadrature tolerances? */
            let (itolQS, itolQ) = {
                let m = cv_mem.borrow();
                (m.cv_itolQS, m.cv_itolQ)
            };
            if (itolQS == CV_EE) && (itolQ == CV_NN) {
                cvProcessError(
                    Some(cv_mem),
                    CV_ILL_INPUT,
                    line!() as i32,
                    "cvInitialSetup",
                    file!(),
                    MSGCV_NO_TOLQ,
                );
                return CV_ILL_INPUT;
            }

            /* Load ewtQS */
            let (znQS0, ewtQS) = {
                let m = cv_mem.borrow();
                (m.cv_znQS[0].clone(), m.cv_ewtQS.clone())
            };
            let ier = cvQuadSensEwtSet(cv_mem, &znQS0, &ewtQS);
            if ier != 0 {
                cvProcessError(
                    Some(cv_mem),
                    CV_ILL_INPUT,
                    line!() as i32,
                    "cvInitialSetup",
                    file!(),
                    MSGCV_BAD_EWTQS,
                );
                return CV_ILL_INPUT;
            }
        }
    } else {
        cv_mem.borrow_mut().cv_errconQS = SUNFALSE;
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

    /* always initialize the ODE NLS in case the user disables sensitivities */
    let ier = crate::cvodes_nls::cvNlsInit(cv_mem);
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

    if cv_mem.borrow().NLSsim.is_some() {
        let ier = crate::cvodes_nls_sim::cvNlsInitSensSim(cv_mem);
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
    }

    if cv_mem.borrow().NLSstg.is_some() {
        let ier = crate::cvodes_nls_stg::cvNlsInitSensStg(cv_mem);
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
    }

    if cv_mem.borrow().NLSstg1.is_some() {
        let ier = crate::cvodes_nls_stg1::cvNlsInitSensStg1(cv_mem);
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
            crate::cvodes_proj::cvProjInit(m.proj_mem.as_mut().unwrap())
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
 * (if tstop is enabled and it is closer to t0=tn than tout). If any RHS
 * function fails unrecoverably, cvHin returns CV_*RHSFUNC_FAIL. If any RHS
 * function fails recoverably too many times and recovery is not possible, cvHin
 * returns CV_REPTD_*RHSFUNC_ERR. Otherwise, cvHin sets h to the chosen value
 * h0 and returns CV_SUCCESS.
 *
 * The algorithm used seeks to find h0 as a solution of
 *       (WRMS norm of (h0^2 ydd / 2)) = 1,
 * where ydd = estimated second derivative of y. Here, y includes
 * all variables considered in the error test.
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
    let mut retval: i32 = 0;

    for count1 in 1..=MAX_ITERS {
        /* Attempts to estimate ydd */

        let mut hgOK = SUNFALSE;

        for _count2 in 1..=MAX_ITERS {
            let hgs = hg * sign as sunrealtype;
            retval = cvYddNorm(cv_mem, hgs, &mut yddnrm);
            /* If a RHS function failed unrecoverably, give up */
            if retval < 0 {
                return retval;
            }
            /* If successful, we can use ydd */
            if retval == CV_SUCCESS {
                hgOK = SUNTRUE;
                break;
            }
            /* A RHS function failed recoverably; cut step size and test again */
            hg *= POINT2;
        }

        /* If a RHS function failed recoverably MAX_ITERS times */

        if !hgOK {
            /* Exit if this is the first or second pass. No recovery possible */
            if count1 <= 2 {
                if retval == RHSFUNC_RECVR {
                    return CV_REPTD_RHSFUNC_ERR;
                }
                if retval == QRHSFUNC_RECVR {
                    return CV_REPTD_QRHSFUNC_ERR;
                }
                if retval == SRHSFUNC_RECVR {
                    return CV_REPTD_SRHSFUNC_ERR;
                }
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
     * Bound based on |y|/|y'| -- allow at most an increase of
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
    let _ = cvb_call_efun(cv_mem, &zn0, &temp1);
    N_VInv(&temp1, &temp1);
    N_VLinearSum(HUB_FACTOR, &temp2, ONE, &temp1, &temp1);

    N_VAbs(&zn1, &temp2);

    N_VDiv(&temp2, &temp1, &temp1);
    let mut hub_inv = N_VMaxNorm(&temp1);

    /* Bound based on |yQ|/|yQ'| */

    let (quadr, errconQ, sensi, errconS, quadr_sensi, errconQS, Ns) = {
        let m = cv_mem.borrow();
        (
            m.cv_quadr,
            m.cv_errconQ,
            m.cv_sensi,
            m.cv_errconS,
            m.cv_quadr_sensi,
            m.cv_errconQS,
            m.cv_Ns,
        )
    };

    if quadr && errconQ {
        let (tempQ1, tempQ2, znQ0, znQ1) = {
            let m = cv_mem.borrow();
            (
                m.cv_tempvQ.clone().unwrap(),
                m.cv_acorQ.clone().unwrap(),
                m.cv_znQ[0].clone().unwrap(),
                m.cv_znQ[1].clone().unwrap(),
            )
        };

        N_VAbs(&znQ0, &tempQ2);
        let _ = cvQuadEwtSet(cv_mem, &znQ0, &tempQ1);
        N_VInv(&tempQ1, &tempQ1);
        N_VLinearSum(HUB_FACTOR, &tempQ2, ONE, &tempQ1, &tempQ1);

        N_VAbs(&znQ1, &tempQ2);

        N_VDiv(&tempQ2, &tempQ1, &tempQ1);
        let hubQ_inv = N_VMaxNorm(&tempQ1);

        if hubQ_inv > hub_inv {
            hub_inv = hubQ_inv;
        }
    }

    /* Bound based on |yS|/|yS'| */

    if sensi && errconS {
        let (tempS1, znS0, znS1) = {
            let m = cv_mem.borrow();
            (m.cv_acorS.clone(), m.cv_znS[0].clone(), m.cv_znS[1].clone())
        };
        let _ = cvSensEwtSet(cv_mem, &znS0, &tempS1);

        for is in 0..Ns as usize {
            N_VAbs(&znS0[is], &temp2);
            N_VInv(&tempS1[is], &temp1);
            N_VLinearSum(HUB_FACTOR, &temp2, ONE, &temp1, &temp1);

            N_VAbs(&znS1[is], &temp2);

            N_VDiv(&temp2, &temp1, &temp1);
            let hubS_inv = N_VMaxNorm(&temp1);

            if hubS_inv > hub_inv {
                hub_inv = hubS_inv;
            }
        }
    }

    /* Bound based on |yQS|/|yQS'| */

    if quadr_sensi && errconQS {
        let (tempQ1, tempQ2, tempQS1, znQS0, znQS1) = {
            let m = cv_mem.borrow();
            (
                m.cv_tempvQ.clone().unwrap(),
                m.cv_acorQ.clone().unwrap(),
                m.cv_acorQS.clone(),
                m.cv_znQS[0].clone(),
                m.cv_znQS[1].clone(),
            )
        };
        let _ = cvQuadSensEwtSet(cv_mem, &znQS0, &tempQS1);

        for is in 0..Ns as usize {
            N_VAbs(&znQS0[is], &tempQ2);
            N_VInv(&tempQS1[is], &tempQ1);
            N_VLinearSum(HUB_FACTOR, &tempQ2, ONE, &tempQ1, &tempQ1);

            N_VAbs(&znQS1[is], &tempQ2);

            N_VDiv(&tempQ2, &tempQ1, &tempQ1);
            let hubQS_inv = N_VMaxNorm(&tempQ1);

            if hubQS_inv > hub_inv {
                hub_inv = hubQS_inv;
            }
        }
    }

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
 * This routine computes an estimate of the second derivative of Y
 * using a difference quotient, and returns its WRMS norm.
 *
 * Y contains all variables included in the error test.
 */

fn cvYddNorm(cv_mem: &CVodeMem, hg: sunrealtype, yddnrm: &mut sunrealtype) -> i32 {
    let (tn, Ns, quadr, errconQ, sensi, errconS, quadr_sensi, errconQS) = {
        let m = cv_mem.borrow();
        (
            m.cv_tn,
            m.cv_Ns,
            m.cv_quadr,
            m.cv_errconQ,
            m.cv_sensi,
            m.cv_errconS,
            m.cv_quadr_sensi,
            m.cv_errconQS,
        )
    };

    /* y <- h*y'(t) + y(t) */

    let (zn0, zn1, y, tempv, ewt) = {
        let m = cv_mem.borrow();
        (
            m.cv_zn[0].clone().unwrap(),
            m.cv_zn[1].clone().unwrap(),
            m.cv_y.clone().unwrap(),
            m.cv_tempv.clone().unwrap(),
            m.cv_ewt.clone().unwrap(),
        )
    };

    N_VLinearSum(hg, &zn1, ONE, &zn0, &y);

    if sensi && errconS {
        let (znS1, znS0, yS) = {
            let m = cv_mem.borrow();
            (m.cv_znS[1].clone(), m.cv_znS[0].clone(), m.cv_yS.clone())
        };
        let retval = N_VLinearSumVectorArray(Ns, hg, &znS1, ONE, &znS0, &yS);
        if retval != CV_SUCCESS {
            return CV_VECTOROP_ERR;
        }
    }

    /* tempv <- f(t+h, h*y'(t)+y(t)) */

    let retval = cvb_call_f(cv_mem, tn + hg, &y, &tempv);
    cv_mem.borrow_mut().cv_nfe += 1;
    if retval < 0 {
        return CV_RHSFUNC_FAIL;
    }
    if retval > 0 {
        return RHSFUNC_RECVR;
    }

    if quadr && errconQ {
        let tempvQ = cv_mem.borrow().cv_tempvQ.clone().unwrap();
        let retval = cvb_call_fQ(cv_mem, tn + hg, &y, &tempvQ);
        cv_mem.borrow_mut().cv_nfQe += 1;
        if retval < 0 {
            return CV_QRHSFUNC_FAIL;
        }
        if retval > 0 {
            return QRHSFUNC_RECVR;
        }
    }

    if sensi && errconS {
        let (wrk1, wrk2, yS, tempvS) = {
            let m = cv_mem.borrow();
            (
                m.cv_ftemp.clone().unwrap(),
                m.cv_acor.clone().unwrap(),
                m.cv_yS.clone(),
                m.cv_tempvS.clone(),
            )
        };
        let retval = cvSensRhsWrapper(cv_mem, tn + hg, &y, &tempv, &yS, &tempvS, &wrk1, &wrk2);
        if retval < 0 {
            return CV_SRHSFUNC_FAIL;
        }
        if retval > 0 {
            return SRHSFUNC_RECVR;
        }
    }

    if quadr_sensi && errconQS {
        let (wrk1, wrk2, yS, tempvQ, tempvQS) = {
            let m = cv_mem.borrow();
            (
                m.cv_ftemp.clone().unwrap(),
                m.cv_acorQ.clone().unwrap(),
                m.cv_yS.clone(),
                m.cv_tempvQ.clone().unwrap(),
                m.cv_tempvQS.clone(),
            )
        };
        let retval = cvb_call_fQS(
            cv_mem,
            Ns,
            tn + hg,
            &y,
            &yS,
            &tempvQ,
            &tempvQS,
            &wrk1,
            &wrk2,
        );

        cv_mem.borrow_mut().cv_nfQSe += 1;
        if retval < 0 {
            return CV_QSRHSFUNC_FAIL;
        }
        if retval > 0 {
            return QSRHSFUNC_RECVR;
        }
    }

    /* Load estimate of ||y''|| into tempv:
     * tempv <-  (1/h) * f(t+h, h*y'(t)+y(t)) - y'(t) */

    N_VLinearSum(ONE / hg, &tempv, -ONE / hg, &zn1, &tempv);

    *yddnrm = N_VWrmsNorm(&tempv, &ewt);

    if quadr && errconQ {
        let (tempvQ, znQ1, ewtQ) = {
            let m = cv_mem.borrow();
            (
                m.cv_tempvQ.clone().unwrap(),
                m.cv_znQ[1].clone().unwrap(),
                m.cv_ewtQ.clone().unwrap(),
            )
        };
        N_VLinearSum(ONE / hg, &tempvQ, -ONE / hg, &znQ1, &tempvQ);

        *yddnrm = cvQuadUpdateNorm(cv_mem, *yddnrm, &tempvQ, &ewtQ);
    }

    if sensi && errconS {
        let (tempvS, znS1, ewtS) = {
            let m = cv_mem.borrow();
            (m.cv_tempvS.clone(), m.cv_znS[1].clone(), m.cv_ewtS.clone())
        };
        let retval = N_VLinearSumVectorArray(Ns, ONE / hg, &tempvS, -ONE / hg, &znS1, &tempvS);
        if retval != CV_SUCCESS {
            return CV_VECTOROP_ERR;
        }

        *yddnrm = cvSensUpdateNorm(cv_mem, *yddnrm, &tempvS, &ewtS);
    }

    if quadr_sensi && errconQS {
        let (tempvQS, znQS1, ewtQS) = {
            let m = cv_mem.borrow();
            (
                m.cv_tempvQS.clone(),
                m.cv_znQS[1].clone(),
                m.cv_ewtQS.clone(),
            )
        };
        let retval = N_VLinearSumVectorArray(Ns, ONE / hg, &tempvQS, -ONE / hg, &znQS1, &tempvQS);
        if retval != CV_SUCCESS {
            return CV_VECTOROP_ERR;
        }

        *yddnrm = cvQuadSensUpdateNorm(cv_mem, *yddnrm, &tempvQS, &ewtQS);
    }

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
    let mut dsm: sunrealtype = ZERO; /* local truncation error estimate       */
    let mut dsmQ: sunrealtype = ZERO; /* quadrature error estimate             */
    let mut dsmS: sunrealtype = ZERO; /* sensitivity error estimate            */
    let mut dsmQS: sunrealtype = ZERO; /* quadrature sensitivity error estimate */

    /* Are we computing sensitivities with a staggered approach? */

    let (sensi, ism, Ns) = {
        let m = cv_mem.borrow();
        (m.cv_sensi, m.cv_ism, m.cv_Ns)
    };
    let do_sensi_stg = sensi && (ism == CV_STAGGERED);
    let do_sensi_stg1 = sensi && (ism == CV_STAGGERED1);

    /* Initialize failure counters for this step attempt */

    let mut ncf: i32 = 0; /* corrector failures  */
    let mut npf: i32 = 0; /* projection failures */
    let mut nef: i32 = 0; /* error test failures */
    let mut step_constraint_fails: i32 = 0;

    let mut ncfS: i32 = 0; /* sensitivity corrector failures          */
    let mut nefS: i32 = 0; /* sensitivity error test fails            */
    let mut nefQ: i32 = 0; /* quadrature error test fails             */
    let mut nefQS: i32 = 0; /* quadrature sensitivity error test fails */

    if do_sensi_stg1 {
        let mut m = cv_mem.borrow_mut();
        for is in 0..Ns as usize {
            m.cv_ncfS1[is] = 0;
        }
    }

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

        /* ------ Correct state variables ------ */

        nflag = cvNls(cv_mem, nflag);
        let kflag = {
            let mut ncfn = cv_mem.borrow().cv_ncfn;
            let kflag = cvHandleNFlag(cv_mem, &mut nflag, saved_t, &mut ncf, &mut ncfn);
            cv_mem.borrow_mut().cv_ncfn = ncfn;
            kflag
        };

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
            let pflag = crate::cvodes_proj::cvDoProjection(cv_mem, &mut nflag, saved_t, &mut npf);

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
        let eflag = {
            let (acnrm, mut netf) = {
                let m = cv_mem.borrow();
                (m.cv_acnrm, m.cv_netf)
            };
            let eflag = cvDoErrorTest(
                cv_mem, &mut nflag, saved_t, acnrm, &mut nef, &mut netf, &mut dsm,
            );
            cv_mem.borrow_mut().cv_netf = netf;
            eflag
        };

        /* Go back in loop if we need to predict again (nflag=PREV_ERR_FAIL) */
        if eflag == TRY_AGAIN {
            continue;
        }

        /* Return if error test failed and recovery is not possible. */
        if eflag != CV_SUCCESS {
            return eflag;
        }

        /* Error test passed (eflag=CV_SUCCESS, nflag=CV_SUCCESS), go on */

        /* ------ Correct the quadrature variables ------ */

        if cv_mem.borrow().cv_quadr {
            ncf = 0; /* reset counters for states */
            nef = 0;

            nflag = cvQuadNls(cv_mem);
            let kflag = {
                let mut ncfn = cv_mem.borrow().cv_ncfn;
                let kflag = cvHandleNFlag(cv_mem, &mut nflag, saved_t, &mut ncf, &mut ncfn);
                cv_mem.borrow_mut().cv_ncfn = ncfn;
                kflag
            };

            if kflag == PREDICT_AGAIN {
                continue;
            }
            if kflag != DO_ERROR_TEST {
                return kflag;
            }

            /* Error test on quadratures */
            if cv_mem.borrow().cv_errconQ {
                let (acorQ, ewtQ) = {
                    let m = cv_mem.borrow();
                    (m.cv_acorQ.clone().unwrap(), m.cv_ewtQ.clone().unwrap())
                };
                let acnrmQ = N_VWrmsNorm(&acorQ, &ewtQ);
                cv_mem.borrow_mut().cv_acnrmQ = acnrmQ;

                let eflag = {
                    let mut netfQ = cv_mem.borrow().cv_netfQ;
                    let eflag = cvDoErrorTest(
                        cv_mem, &mut nflag, saved_t, acnrmQ, &mut nefQ, &mut netfQ, &mut dsmQ,
                    );
                    cv_mem.borrow_mut().cv_netfQ = netfQ;
                    eflag
                };

                if eflag == TRY_AGAIN {
                    continue;
                }
                if eflag != CV_SUCCESS {
                    return eflag;
                }

                /* Set dsm = max(dsm, dsmQ) to be used in cvPrepareNextStep */
                if dsmQ > dsm {
                    dsm = dsmQ;
                }
            }
        }

        /* ------ Correct the sensitivity variables (STAGGERED or STAGGERED1) ------- */

        if do_sensi_stg || do_sensi_stg1 {
            ncf = 0; /* reset counters for states     */
            nef = 0;
            if cv_mem.borrow().cv_quadr {
                nefQ = 0; /* reset counter for quadratures */
            }

            /* Evaluate f at converged y, needed for future evaluations of sens. RHS
             * If f() fails recoverably, treat it as a convergence failure and
             * attempt the step again */

            let (tn, y, ftemp) = {
                let m = cv_mem.borrow();
                (
                    m.cv_tn,
                    m.cv_y.clone().unwrap(),
                    m.cv_ftemp.clone().unwrap(),
                )
            };
            let retval = cvb_call_f(cv_mem, tn, &y, &ftemp);
            cv_mem.borrow_mut().cv_nfe += 1;

            if retval < 0 {
                return CV_RHSFUNC_FAIL;
            }
            if retval > 0 {
                nflag = PREV_CONV_FAIL;
                continue;
            }

            let kflag;
            if do_sensi_stg {
                /* Nonlinear solve for sensitivities (all-at-once) */
                nflag = cvStgrNls(cv_mem);
                let mut ncfnS = cv_mem.borrow().cv_ncfnS;
                kflag = cvHandleNFlag(cv_mem, &mut nflag, saved_t, &mut ncfS, &mut ncfnS);
                cv_mem.borrow_mut().cv_ncfnS = ncfnS;
            } else {
                /* Nonlinear solve for sensitivities (one-by-one).  The C code
                leaves kflag at its previous value (DO_ERROR_TEST, since the
                state solve got past its own check) when the loop body does
                not run. */
                let mut kf = DO_ERROR_TEST;
                for is in 0..Ns {
                    cv_mem.borrow_mut().sens_solve_idx = is;

                    nflag = cvStgr1Nls(cv_mem, is);
                    let (mut ncfS1, mut ncfnS1) = {
                        let m = cv_mem.borrow();
                        (m.cv_ncfS1[is as usize], m.cv_ncfnS1[is as usize])
                    };
                    kf = cvHandleNFlag(cv_mem, &mut nflag, saved_t, &mut ncfS1, &mut ncfnS1);
                    {
                        let mut m = cv_mem.borrow_mut();
                        m.cv_ncfS1[is as usize] = ncfS1;
                        m.cv_ncfnS1[is as usize] = ncfnS1;
                    }
                    if kf != DO_ERROR_TEST {
                        break;
                    }
                }
                kflag = kf;
            }

            if kflag == PREDICT_AGAIN {
                continue;
            }
            if kflag != DO_ERROR_TEST {
                return kflag;
            }

            /* Error test on sensitivities */
            if cv_mem.borrow().cv_errconS {
                if !cv_mem.borrow().cv_acnrmScur {
                    let (acorS, ewtS) = {
                        let m = cv_mem.borrow();
                        (m.cv_acorS.clone(), m.cv_ewtS.clone())
                    };
                    let acnrmS = cvSensNorm(cv_mem, &acorS, &ewtS);
                    cv_mem.borrow_mut().cv_acnrmS = acnrmS;
                }

                let eflag = {
                    let (acnrmS, mut netfS) = {
                        let m = cv_mem.borrow();
                        (m.cv_acnrmS, m.cv_netfS)
                    };
                    let eflag = cvDoErrorTest(
                        cv_mem, &mut nflag, saved_t, acnrmS, &mut nefS, &mut netfS, &mut dsmS,
                    );
                    cv_mem.borrow_mut().cv_netfS = netfS;
                    eflag
                };

                if eflag == TRY_AGAIN {
                    continue;
                }
                if eflag != CV_SUCCESS {
                    return eflag;
                }

                /* Set dsm = max(dsm, dsmS) to be used in cvPrepareNextStep */
                if dsmS > dsm {
                    dsm = dsmS;
                }
            }
        }

        /* ------ Correct the quadrature sensitivity variables ------ */

        if cv_mem.borrow().cv_quadr_sensi {
            /* Reset local convergence and error test failure counters */
            ncf = 0;
            nef = 0;
            if cv_mem.borrow().cv_quadr {
                nefQ = 0;
            }
            if do_sensi_stg {
                ncfS = 0;
                nefS = 0;
            }
            if do_sensi_stg1 {
                let mut m = cv_mem.borrow_mut();
                for is in 0..Ns as usize {
                    m.cv_ncfS1[is] = 0;
                }
                drop(m);
                nefS = 0;
            }

            /* Note that ftempQ contains yQdot evaluated at the converged y
             * (stored in cvQuadNls) and can be used in evaluating fQS */

            nflag = cvQuadSensNls(cv_mem);
            let kflag = {
                let mut ncfn = cv_mem.borrow().cv_ncfn;
                let kflag = cvHandleNFlag(cv_mem, &mut nflag, saved_t, &mut ncf, &mut ncfn);
                cv_mem.borrow_mut().cv_ncfn = ncfn;
                kflag
            };

            if kflag == PREDICT_AGAIN {
                continue;
            }
            if kflag != DO_ERROR_TEST {
                return kflag;
            }

            /* Error test on quadrature sensitivities */
            if cv_mem.borrow().cv_errconQS {
                let (acorQS, ewtQS) = {
                    let m = cv_mem.borrow();
                    (m.cv_acorQS.clone(), m.cv_ewtQS.clone())
                };
                let acnrmQS = cvQuadSensNorm(cv_mem, &acorQS, &ewtQS);
                cv_mem.borrow_mut().cv_acnrmQS = acnrmQS;

                let eflag = {
                    let mut netfQS = cv_mem.borrow().cv_netfQS;
                    let eflag = cvDoErrorTest(
                        cv_mem,
                        &mut nflag,
                        saved_t,
                        acnrmQS,
                        &mut nefQS,
                        &mut netfQS,
                        &mut dsmQS,
                    );
                    cv_mem.borrow_mut().cv_netfQS = netfQS;
                    eflag
                };

                if eflag == TRY_AGAIN {
                    continue;
                }
                if eflag != CV_SUCCESS {
                    return eflag;
                }

                /* Set dsm = max(dsm, dsmQS) to be used in cvPrepareNextStep */
                if dsmQS > dsm {
                    dsm = dsmQS;
                }
            }
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

    if cv_mem.borrow().cv_quadr {
        let acorQ = cv_mem.borrow().cv_acorQ.clone().unwrap();
        N_VScale(tq2, &acorQ, &acorQ);
    }

    if cv_mem.borrow().cv_sensi {
        let cvals = vec![tq2; Ns as usize];

        let acorS = cv_mem.borrow().cv_acorS.clone();
        let retval = N_VScaleVectorArray(Ns, &cvals, &acorS, &acorS);
        if retval != CV_SUCCESS {
            return CV_VECTOROP_ERR;
        }
    }

    if cv_mem.borrow().cv_quadr_sensi {
        let cvals = vec![tq2; Ns as usize];

        let acorQS = cv_mem.borrow().cv_acorQS.clone();
        let retval = N_VScaleVectorArray(Ns, &cvals, &acorQS, &acorQS);
        if retval != CV_SUCCESS {
            return CV_VECTOROP_ERR;
        }
    }

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
    let (quadr, sensi, Ns) = {
        let m = cv_mem.borrow();
        (m.cv_quadr, m.cv_sensi, m.cv_Ns)
    };

    /* On an order increase, set new column of zn to zero and return */

    if deltaq == 1 {
        let znL = {
            let m = cv_mem.borrow();
            m.cv_zn[m.cv_L as usize].clone().unwrap()
        };
        N_VConst(ZERO, &znL);
        if quadr {
            let znQL = {
                let m = cv_mem.borrow();
                m.cv_znQ[m.cv_L as usize].clone().unwrap()
            };
            N_VConst(ZERO, &znQL);
        }
        if sensi {
            let znSL = {
                let m = cv_mem.borrow();
                m.cv_znS[m.cv_L as usize].clone()
            };
            let _ = N_VConstVectorArray(Ns, ZERO, &znSL);
        }
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

    let q = {
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
            m.cv_l[(j + 1) as usize] =
                q as sunrealtype * (m.cv_l[j as usize] / (j + 1) as sunrealtype);
        }
        q
    };

    if q > 2 {
        let (cvals, znq, znvec) = {
            let m = cv_mem.borrow();
            let cvals: Vec<sunrealtype> = (2..m.cv_q).map(|j| -m.cv_l[j as usize]).collect();
            let znvec: Vec<N_Vector> = (2..m.cv_q as usize)
                .map(|j| m.cv_zn[j].clone().unwrap())
                .collect();
            (cvals, m.cv_zn[m.cv_q as usize].clone().unwrap(), znvec)
        };

        let _ = N_VScaleAddMulti(q - 2, &cvals, &znq, &znvec, &znvec);

        if quadr {
            let (znQq, znQvec) = {
                let m = cv_mem.borrow();
                let znQvec: Vec<N_Vector> = (2..m.cv_q as usize)
                    .map(|j| m.cv_znQ[j].clone().unwrap())
                    .collect();
                (m.cv_znQ[m.cv_q as usize].clone().unwrap(), znQvec)
            };
            let _ = N_VScaleAddMulti(q - 2, &cvals, &znQq, &znQvec, &znQvec);
        }

        if sensi {
            let (znSq, znSvec) = {
                let m = cv_mem.borrow();
                let znSvec: Vec<Vec<N_Vector>> =
                    (2..m.cv_q as usize).map(|j| m.cv_znS[j].clone()).collect();
                (m.cv_znS[m.cv_q as usize].clone(), znSvec)
            };
            let _ = N_VScaleAddMultiVectorArray(Ns, q - 2, &cvals, &znSq, &znSvec, &znSvec);
        }
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
    let (quadr, sensi, quadr_sensi, Ns) = {
        let m = cv_mem.borrow();
        (m.cv_quadr, m.cv_sensi, m.cv_quadr_sensi, m.cv_Ns)
    };

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

    /*
       zn[indx_acor] contains the value Delta_n = y_n - y_n(0)
       This value was stored there at the previous successful
       step (in cvCompleteStep)

       A1 contains dbar = (1/xi* - 1/xi_q)/prod(xi_j)
    */

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
    let l2: Vec<sunrealtype> = cv_mem.borrow().cv_l[2..].to_vec();
    if q > 1 {
        let znvec: Vec<N_Vector> = {
            let m = cv_mem.borrow();
            (2..=m.cv_q as usize)
                .map(|j| m.cv_zn[j].clone().unwrap())
                .collect()
        };
        let _ = N_VScaleAddMulti(q - 1, &l2, &znL, &znvec, &znvec);
    }

    if quadr {
        let (znQ_indx_acor, znQL) = {
            let m = cv_mem.borrow();
            (
                m.cv_znQ[m.cv_indx_acor as usize].clone().unwrap(),
                m.cv_znQ[m.cv_L as usize].clone().unwrap(),
            )
        };
        N_VScale(A1, &znQ_indx_acor, &znQL);

        /* for (j=2; j <= cv_mem->cv_q; j++) */
        if q > 1 {
            let znQvec: Vec<N_Vector> = {
                let m = cv_mem.borrow();
                (2..=m.cv_q as usize)
                    .map(|j| m.cv_znQ[j].clone().unwrap())
                    .collect()
            };
            let _ = N_VScaleAddMulti(q - 1, &l2, &znQL, &znQvec, &znQvec);
        }
    }

    if sensi {
        let cvals = vec![A1; Ns as usize];

        let (znS_indx_acor, znSL) = {
            let m = cv_mem.borrow();
            (
                m.cv_znS[m.cv_indx_acor as usize].clone(),
                m.cv_znS[m.cv_L as usize].clone(),
            )
        };
        let _ = N_VScaleVectorArray(Ns, &cvals, &znS_indx_acor, &znSL);

        /* for (j=2; j <= cv_mem->cv_q; j++) */
        if q > 1 {
            let znSvec: Vec<Vec<N_Vector>> = {
                let m = cv_mem.borrow();
                (2..=m.cv_q as usize).map(|j| m.cv_znS[j].clone()).collect()
            };
            let _ = N_VScaleAddMultiVectorArray(Ns, q - 1, &l2, &znSL, &znSvec, &znSvec);
        }
    }

    if quadr_sensi {
        let cvals = vec![A1; Ns as usize];

        let (znQS_indx_acor, znQSL) = {
            let m = cv_mem.borrow();
            (
                m.cv_znQS[m.cv_indx_acor as usize].clone(),
                m.cv_znQS[m.cv_L as usize].clone(),
            )
        };
        let _ = N_VScaleVectorArray(Ns, &cvals, &znQS_indx_acor, &znQSL);

        /* for (j=2; j <= cv_mem->cv_q; j++) */
        if q > 1 {
            let znQSvec: Vec<Vec<N_Vector>> = {
                let m = cv_mem.borrow();
                (2..=m.cv_q as usize)
                    .map(|j| m.cv_znQS[j].clone())
                    .collect()
            };
            let _ = N_VScaleAddMultiVectorArray(Ns, q - 1, &l2, &znQSL, &znQSvec, &znQSvec);
        }
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
    let (quadr, sensi, quadr_sensi, Ns) = {
        let m = cv_mem.borrow();
        (m.cv_quadr, m.cv_sensi, m.cv_quadr_sensi, m.cv_Ns)
    };

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
    }

    let q = cv_mem.borrow().cv_q;
    if q > 2 {
        let (cvals, znq, znvec) = {
            let m = cv_mem.borrow();
            let cvals: Vec<sunrealtype> = (2..m.cv_q).map(|j| -m.cv_l[j as usize]).collect();
            let znvec: Vec<N_Vector> = (2..m.cv_q as usize)
                .map(|j| m.cv_zn[j].clone().unwrap())
                .collect();
            (cvals, m.cv_zn[m.cv_q as usize].clone().unwrap(), znvec)
        };

        let _ = N_VScaleAddMulti(q - 2, &cvals, &znq, &znvec, &znvec);

        if quadr {
            let (znQq, znQvec) = {
                let m = cv_mem.borrow();
                let znQvec: Vec<N_Vector> = (2..m.cv_q as usize)
                    .map(|j| m.cv_znQ[j].clone().unwrap())
                    .collect();
                (m.cv_znQ[m.cv_q as usize].clone().unwrap(), znQvec)
            };
            let _ = N_VScaleAddMulti(q - 2, &cvals, &znQq, &znQvec, &znQvec);
        }

        if sensi {
            let (znSq, znSvec) = {
                let m = cv_mem.borrow();
                let znSvec: Vec<Vec<N_Vector>> =
                    (2..m.cv_q as usize).map(|j| m.cv_znS[j].clone()).collect();
                (m.cv_znS[m.cv_q as usize].clone(), znSvec)
            };
            let _ = N_VScaleAddMultiVectorArray(Ns, q - 2, &cvals, &znSq, &znSvec, &znSvec);
        }

        if quadr_sensi {
            let (znQSq, znQSvec) = {
                let m = cv_mem.borrow();
                let znQSvec: Vec<Vec<N_Vector>> =
                    (2..m.cv_q as usize).map(|j| m.cv_znQS[j].clone()).collect();
                (m.cv_znQS[m.cv_q as usize].clone(), znQSvec)
            };
            let _ = N_VScaleAddMultiVectorArray(Ns, q - 2, &cvals, &znQSq, &znQSvec, &znQSvec);
        }
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
    let (q, Ns, eta, quadr, sensi, quadr_sensi) = {
        let m = cv_mem.borrow();
        (
            m.cv_q,
            m.cv_Ns,
            m.cv_eta,
            m.cv_quadr,
            m.cv_sensi,
            m.cv_quadr_sensi,
        )
    };

    /* compute scaling factors */
    let mut cvals = vec![ZERO; (q as usize + 1).max(1)];
    cvals[0] = eta;
    for j in 1..=q as usize {
        cvals[j] = eta * cvals[j - 1];
    }

    let znvec: Vec<N_Vector> = {
        let m = cv_mem.borrow();
        (1..=q as usize)
            .map(|j| m.cv_zn[j].clone().unwrap())
            .collect()
    };
    let _ = N_VScaleVectorArray(q, &cvals, &znvec, &znvec);

    if quadr {
        let znQvec: Vec<N_Vector> = {
            let m = cv_mem.borrow();
            (1..=q as usize)
                .map(|j| m.cv_znQ[j].clone().unwrap())
                .collect()
        };
        let _ = N_VScaleVectorArray(q, &cvals, &znQvec, &znQvec);
    }

    /* compute sensi scaling factors */
    let mut cvalsS: Vec<sunrealtype> = Vec::new();
    if sensi || quadr_sensi {
        cvalsS = vec![ZERO; ((q as usize + 1) * Ns as usize).max(1)];
        for is in 0..Ns as usize {
            cvalsS[is] = eta;
        }
        for j in 1..=q as usize {
            for is in 0..Ns as usize {
                cvalsS[j * Ns as usize + is] = eta * cvalsS[(j - 1) * Ns as usize + is];
            }
        }
    }

    if sensi {
        let Xvecs: Vec<N_Vector> = {
            let m = cv_mem.borrow();
            let mut v: Vec<N_Vector> = Vec::new();
            for j in 1..=q as usize {
                for is in 0..Ns as usize {
                    v.push(m.cv_znS[j][is].clone());
                }
            }
            v
        };

        let _ = N_VScaleVectorArray(q * Ns, &cvalsS, &Xvecs, &Xvecs);
    }

    if quadr_sensi {
        let Xvecs: Vec<N_Vector> = {
            let m = cv_mem.borrow();
            let mut v: Vec<N_Vector> = Vec::new();
            for j in 1..=q as usize {
                for is in 0..Ns as usize {
                    v.push(m.cv_znQS[j][is].clone());
                }
            }
            v
        };

        let _ = N_VScaleVectorArray(q * Ns, &cvalsS, &Xvecs, &Xvecs);
    }

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

    let (q, Ns, quadr, sensi, quadr_sensi) = {
        let m = cv_mem.borrow();
        (m.cv_q, m.cv_Ns, m.cv_quadr, m.cv_sensi, m.cv_quadr_sensi)
    };

    let zn: Vec<N_Vector> = {
        let m = cv_mem.borrow();
        (0..=q as usize)
            .map(|j| m.cv_zn[j].clone().unwrap())
            .collect()
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

    if quadr {
        let znQ: Vec<N_Vector> = {
            let m = cv_mem.borrow();
            (0..=q as usize)
                .map(|j| m.cv_znQ[j].clone().unwrap())
                .collect()
        };
        for k in 1..=q {
            let mut j = q;
            while j >= k {
                N_VLinearSum(
                    ONE,
                    &znQ[(j - 1) as usize],
                    ONE,
                    &znQ[j as usize],
                    &znQ[(j - 1) as usize],
                );
                j -= 1;
            }
        }
    }

    if sensi {
        let znS: Vec<Vec<N_Vector>> = {
            let m = cv_mem.borrow();
            (0..=q as usize).map(|j| m.cv_znS[j].clone()).collect()
        };
        for k in 1..=q {
            let mut j = q;
            while j >= k {
                let _ = N_VLinearSumVectorArray(
                    Ns,
                    ONE,
                    &znS[(j - 1) as usize],
                    ONE,
                    &znS[j as usize],
                    &znS[(j - 1) as usize],
                );
                j -= 1;
            }
        }
    }

    if quadr_sensi {
        let znQS: Vec<Vec<N_Vector>> = {
            let m = cv_mem.borrow();
            (0..=q as usize).map(|j| m.cv_znQS[j].clone()).collect()
        };
        for k in 1..=q {
            let mut j = q;
            while j >= k {
                let _ = N_VLinearSumVectorArray(
                    Ns,
                    ONE,
                    &znQS[(j - 1) as usize],
                    ONE,
                    &znQS[j as usize],
                    &znQS[(j - 1) as usize],
                );
                j -= 1;
            }
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

fn cvAdamsFinish(
    cv_mem: &CVodeMem,
    m_: &mut [sunrealtype],
    M: &mut [sunrealtype],
    hsum: sunrealtype,
) {
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

    /* Are we computing sensitivities with the CV_SIMULTANEOUS approach? */
    let (do_sensi_sim, Ns) = {
        let m = cv_mem.borrow();
        (m.cv_sensi && (m.cv_ism == CV_SIMULTANEOUS), m.cv_Ns)
    };

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

            let mut cs = (nflag == PREV_CONV_FAIL)
                || (nflag == PREV_ERR_FAIL)
                || (m.cv_nst == 0)
                || (m.first_step_after_resize)
                || (m.cv_nst >= m.cv_nstlp + m.cv_msbp)
                || (SUNRabs(m.cv_gamrat - ONE) > m.cv_dgmax_lsetup);

            /* Decide whether to force a call to setup */
            if m.cv_forceSetup {
                cs = SUNTRUE;
                m.convfail = CV_FAIL_OTHER;
            }
            callSetup = cs;
        } else {
            m.cv_crate = ONE;
            m.cv_crateS = ONE; /* if NO lsetup all conv. rates are set to ONE */
            callSetup = SUNFALSE;
        }
    }

    /* initial guess for the correction to the predictor */
    let acor = cv_mem.borrow().cv_acor.clone().unwrap();
    let ycorSim = cv_mem.borrow().ycorSim.clone();
    if do_sensi_sim {
        N_VConst(ZERO, ycorSim.as_ref().unwrap());
    } else {
        N_VConst(ZERO, &acor);
    }

    /* The C `void*` integrator mem handed to the nonlinear solver maps to a
    boxed handle clone (the same token shape cvodes_nls*.rs downcasts) */
    let NLS = cv_mem.borrow().NLS.clone().unwrap();
    let mut nls_mem: Option<Box<dyn Any>> = Some(Box::new(cv_mem.clone()));

    /* call nonlinear solver setup if it exists */
    if NLS.ops.borrow().setup.is_some() {
        let flag = if do_sensi_sim {
            SUNNonlinSolSetup(&NLS, ycorSim.as_ref().unwrap(), &mut nls_mem)
        } else {
            SUNNonlinSolSetup(&NLS, &acor, &mut nls_mem)
        };

        if flag < 0 {
            return CV_NLS_SETUP_FAIL;
        }
        if flag > 0 {
            return SUN_NLS_CONV_RECVR;
        }
    }

    /* solve the nonlinear system */
    let flag;
    if do_sensi_sim {
        let (NLSsim, zn0Sim, ewtSim, tq4) = {
            let m = cv_mem.borrow();
            (
                m.NLSsim.clone().unwrap(),
                m.zn0Sim.clone().unwrap(),
                m.ewtSim.clone().unwrap(),
                m.cv_tq[4],
            )
        };
        flag = SUNNonlinSolSolve(
            &NLSsim,
            &zn0Sim,
            ycorSim.as_ref().unwrap(),
            &ewtSim,
            tq4,
            callSetup,
            &mut nls_mem,
        );

        /* increment counters */
        let _ = SUNNonlinSolGetNumIters(&NLSsim, &mut nni_inc);
        cv_mem.borrow_mut().cv_nni += nni_inc;

        let _ = SUNNonlinSolGetNumConvFails(&NLSsim, &mut nnf_inc);
        cv_mem.borrow_mut().cv_nnf += nnf_inc;
    } else {
        let (zn0, ewt, tq4) = {
            let m = cv_mem.borrow();
            (
                m.cv_zn[0].clone().unwrap(),
                m.cv_ewt.clone().unwrap(),
                m.cv_tq[4],
            )
        };
        flag = SUNNonlinSolSolve(&NLS, &zn0, &acor, &ewt, tq4, callSetup, &mut nls_mem);

        /* increment counters */
        let _ = SUNNonlinSolGetNumIters(&NLS, &mut nni_inc);
        cv_mem.borrow_mut().cv_nni += nni_inc;

        let _ = SUNNonlinSolGetNumConvFails(&NLS, &mut nnf_inc);
        cv_mem.borrow_mut().cv_nnf += nnf_inc;
    }

    /* if the solve failed return */
    if flag != SUN_SUCCESS {
        return flag;
    }

    /* solve successful */

    /* update the state based on the final correction from the nonlinear solver */
    let (zn0, y, ewt) = {
        let m = cv_mem.borrow();
        (
            m.cv_zn[0].clone().unwrap(),
            m.cv_y.clone().unwrap(),
            m.cv_ewt.clone().unwrap(),
        )
    };
    N_VLinearSum(ONE, &zn0, ONE, &acor, &y);

    /* update the sensitivities based on the final correction from the
    nonlinear solver */
    if do_sensi_sim {
        let (znS0, acorS, yS) = {
            let m = cv_mem.borrow();
            (m.cv_znS[0].clone(), m.cv_acorS.clone(), m.cv_yS.clone())
        };
        let _ = N_VLinearSumVectorArray(Ns, ONE, &znS0, ONE, &acorS, &yS);
    }

    /* compute acnrm if is was not already done by the nonlinear solver */
    if !cv_mem.borrow().cv_acnrmcur {
        let acnrm = if do_sensi_sim && cv_mem.borrow().cv_errconS {
            let ewtSim = cv_mem.borrow().ewtSim.clone().unwrap();
            N_VWrmsNorm(ycorSim.as_ref().unwrap(), &ewtSim)
        } else {
            N_VWrmsNorm(&acor, &ewt)
        };
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

/* =====================================================================
 * FRAGMENT: `src/cvodes/cvodes.c` PART C
 * ---------------------------------------------------------------------
 * Every function whose definition starts at line 7200 or later in
 * `src/cvodes/cvodes.c`, i.e.
 *
 *   Nonlinear-solver helpers : cvQuadNls, cvQuadSensNls, cvStgrNls,
 *                              cvStgr1Nls, cvHandleNFlag, cvRestore
 *   Error test               : cvDoErrorTest
 *   After a successful step  : cvCompleteStep, cvPrepareNextStep,
 *                              cvSetEta, cvComputeEtaqm1,
 *                              cvComputeEtaqp1, cvChooseEta
 *   Failure handling         : cvHandleFailure
 *   BDF stability limit      : cvBDFStab, cvSLdet
 *   Rootfinding              : cvRcheck1, cvRcheck2, cvRcheck3,
 *                              cvRootfind
 *   Internal EWT functions   : cvEwtSet, cvEwtSetSS, cvEwtSetSV,
 *                              cvQuadEwtSet, cvQuadEwtSetSS,
 *                              cvQuadEwtSetSV, cvSensEwtSet,
 *                              cvSensEwtSetEE, cvSensEwtSetSS,
 *                              cvSensEwtSetSV, cvQuadSensEwtSet,
 *                              cvQuadSensEwtSetEE, cvQuadSensEwtSetSS,
 *                              cvQuadSensEwtSetSV
 *   Combined norms           : cvQuadUpdateNorm, cvSensNorm,
 *                              cvSensUpdateNorm, cvQuadSensNorm,
 *                              cvQuadSensUpdateNorm
 *   Sensitivity RHS wrappers : cvSensRhsWrapper, cvSensRhs1Wrapper
 *   Internal sensitivity DQ  : cvSensRhsInternalDQ,
 *                              cvSensRhs1InternalDQ,
 *                              cvQuadSensRhsInternalDQ,
 *                              cvQuadSensRhs1InternalDQ
 *
 * `cvProcessError` (cvodes.c line 10075) is NOT reproduced here: it is
 * relocated to `cvodes_impl.rs` per the frozen contract.
 *
 * Reference build configuration: SUNDIALS_LOGGING_LEVEL = 2 (SUNLogInfo/
 * SUNLogDebug/SUNLogExtraDebug* call sites omitted at translation time;
 * CV_WARNING paths kept), profiling off, error checks off, monitoring ON,
 * fused kernels OFF (the unfused branch is the live code), serial only.
 * ===================================================================== */

/* Part-C-only callback helper. `cv_call_f`, `cv_call_fQ` and
 * `cv_call_efun` are defined once, above, in Part A. */
/// Invoke the user root function `g`
/// (C: `cv_mem->cv_gfun(t, y, gout, cv_mem->cv_user_data)`).
fn cv_call_gfun(cv_mem: &CVodeMem, t: sunrealtype, y: &N_Vector, gout: &mut [sunrealtype]) -> i32 {
    let gfun = cv_mem.borrow().cv_gfun.expect("cv_gfun set");
    let mut user_data = cv_mem.borrow_mut().cv_user_data.take();
    let retval = gfun(t, y, gout, &mut user_data);
    cv_mem.borrow_mut().cv_user_data = user_data;
    retval
}
/* ------------------ END FRAGMENT-SHARED HELPERS -------------------- */

/*
 * cvQuadNls
 *
 * This routine solves for the quadrature variables at the new step.
 * It does not solve a nonlinear system, but rather updates the
 * quadrature variables. The name for this function is just for
 * uniformity purposes.
 *
 * Possible return values (interpreted by cvHandleNFlag)
 *
 *   CV_SUCCESS       -> continue with error test
 *   CV_QRHSFUNC_FAIL -> halt the integration
 *   QRHSFUNC_RECVR   -> predict again or stop if too many
 *
 */

pub(crate) fn cvQuadNls(cv_mem: &CVodeMem) -> i32 {
    let retval: i32;

    /* Save quadrature correction in acorQ */
    let (tn, y, acorQ) = {
        let m = cv_mem.borrow();
        (
            m.cv_tn,
            m.cv_y.clone().unwrap(),
            m.cv_acorQ.clone().unwrap(),
        )
    };
    retval = cv_call_fQ(cv_mem, tn, &y, &acorQ);
    cv_mem.borrow_mut().cv_nfQe += 1;
    if retval < 0 {
        return CV_QRHSFUNC_FAIL;
    }
    if retval > 0 {
        return QRHSFUNC_RECVR;
    }

    /* If needed, save the value of yQdot = fQ into ftempQ
     * for use in evaluating fQS */
    if cv_mem.borrow().cv_quadr_sensi {
        let ftempQ = cv_mem.borrow().cv_ftempQ.clone().unwrap();
        N_VScale(ONE, &acorQ, &ftempQ);
    }

    let (h, rl1, znQ1, znQ0, yQ) = {
        let m = cv_mem.borrow();
        (
            m.cv_h,
            m.cv_rl1,
            m.cv_znQ[1].clone().unwrap(),
            m.cv_znQ[0].clone().unwrap(),
            m.cv_yQ.clone().unwrap(),
        )
    };
    N_VLinearSum(h, &acorQ, -ONE, &znQ1, &acorQ);
    N_VScale(rl1, &acorQ, &acorQ);

    /* Apply correction to quadrature variables */
    N_VLinearSum(ONE, &znQ0, ONE, &acorQ, &yQ);

    CV_SUCCESS
}

/*
 * cvQuadSensNls
 *
 * This routine solves for the quadrature sensitivity variables
 * at the new step. It does not solve a nonlinear system, but
 * rather updates the quadrature variables. The name for this
 * function is just for uniformity purposes.
 *
 * Possible return values (interpreted by cvHandleNFlag)
 *
 *   CV_SUCCESS        -> continue with error test
 *   CV_QSRHSFUNC_FAIL -> halt the integration
 *   QSRHSFUNC_RECVR   -> predict again or stop if too many
 *
 */

pub(crate) fn cvQuadSensNls(cv_mem: &CVodeMem) -> i32 {
    let retval: i32;

    /* Save quadrature correction in acorQ */
    let (Ns, tn, y, yS, ftempQ, acorQS, tempv, tempvQ) = {
        let m = cv_mem.borrow();
        (
            m.cv_Ns,
            m.cv_tn,
            m.cv_y.clone().unwrap(),
            m.cv_yS.clone(),
            m.cv_ftempQ.clone().unwrap(),
            m.cv_acorQS.clone(),
            m.cv_tempv.clone().unwrap(),
            m.cv_tempvQ.clone().unwrap(),
        )
    };
    /* NOTE (faithful to upstream, cvodes.c:7328-7331): this call site
    passes `cv_user_data`, unlike the three other `cv_fQS` call sites
    (CVode cvodes.c:3075, cvYddNorm cvodes.c:5799, cvDoErrorTest
    cvodes.c:7768) which all pass `cv_fQS_data`.  This is an upstream
    defect, not a porting slip: when the user selects the internal DQ
    quadrature-sensitivity RHS (`CVodeQuadSensInit(mem, None, yQS0)`),
    C sets `cv_fQS_data = cvode_mem` (cvodes.c:2330) and
    `cvQuadSensRhsInternalDQ` casts its data argument straight back to
    `CVodeMem` (cvodes.c:9978) -- so this call hands it the user's own
    pointer and C reinterprets it, which is undefined behavior.  The
    port reproduces the call site exactly, so the same misuse turns into
    a deterministic panic in the downcast inside
    `cvQuadSensRhsInternalDQ` (accepted deviation class 5).  Do NOT
    "fix" this by passing `cv_fQS_data`: that would diverge from
    cvodes.c.  Consequence: internal-DQ quadrature sensitivities are
    unusable in upstream 7.8.0 as well; a user-supplied `fQS` (which
    every reference example provides) is unaffected because
    `cv_fQS_data` then aliases `cv_user_data` in C anyway. */
    {
        let fQS = cv_mem.borrow().cv_fQS.expect("cv_fQS set");
        let mut user_data = cv_mem.borrow_mut().cv_user_data.take();
        retval = fQS(
            Ns,
            tn,
            &y,
            &yS,
            &ftempQ,
            &acorQS,
            &mut user_data,
            &tempv,
            &tempvQ,
        );
        cv_mem.borrow_mut().cv_user_data = user_data;
    }
    cv_mem.borrow_mut().cv_nfQSe += 1;
    if retval < 0 {
        return CV_QSRHSFUNC_FAIL;
    }
    if retval > 0 {
        return QSRHSFUNC_RECVR;
    }

    let (h, rl1, znQS1, znQS0, yQS) = {
        let m = cv_mem.borrow();
        (
            m.cv_h,
            m.cv_rl1,
            m.cv_znQS[1].clone(),
            m.cv_znQS[0].clone(),
            m.cv_yQS.clone(),
        )
    };
    for is in 0..Ns as usize {
        N_VLinearSum(h, &acorQS[is], -ONE, &znQS1[is], &acorQS[is]);
        N_VScale(rl1, &acorQS[is], &acorQS[is]);
        /* Apply correction to quadrature sensitivity variables */
        N_VLinearSum(ONE, &znQS0[is], ONE, &acorQS[is], &yQS[is]);
    }

    CV_SUCCESS
}

/*
 * cvStgrNls
 *
 * This is a high-level routine that attempts to solve the
 * sensitivity linear systems using the attached nonlinear solver
 * once the states y_n were obtained and passed the error test.
 */

pub(crate) fn cvStgrNls(cv_mem: &CVodeMem) -> i32 {
    let callSetup: sunbooleantype;
    let mut nniS_inc: i64 = 0;
    let mut nnfS_inc: i64 = 0;

    callSetup = SUNFALSE;
    if cv_mem.borrow().cv_lsetup.is_none() {
        cv_mem.borrow_mut().cv_crateS = ONE;
    }

    /* initial guess for the correction to the predictor */
    let ycorStg = cv_mem.borrow().ycorStg.clone().unwrap();
    N_VConst(ZERO, &ycorStg);

    /* set sens solve flag */
    cv_mem.borrow_mut().sens_solve = SUNTRUE;

    /* solve the nonlinear system */
    let (NLSstg, zn0Stg, ewtStg, tq4) = {
        let m = cv_mem.borrow();
        (
            m.NLSstg.clone().unwrap(),
            m.zn0Stg.clone().unwrap(),
            m.ewtStg.clone().unwrap(),
            m.cv_tq[4],
        )
    };
    /* The C `void*` integrator mem handed to the nonlinear solver maps to a
    boxed handle clone (the same token shape the cvodes_nls_* modules
    downcast) */
    let mut nls_mem: Option<Box<dyn Any>> = Some(Box::new(cv_mem.clone()));
    let flag = SUNNonlinSolSolve(
        &NLSstg,
        &zn0Stg,
        &ycorStg,
        &ewtStg,
        tq4,
        callSetup,
        &mut nls_mem,
    );

    /* increment counters */
    let _ = SUNNonlinSolGetNumIters(&NLSstg, &mut nniS_inc);
    cv_mem.borrow_mut().cv_nniS += nniS_inc;

    let _ = SUNNonlinSolGetNumConvFails(&NLSstg, &mut nnfS_inc);
    cv_mem.borrow_mut().cv_nnfS += nnfS_inc;

    /* reset sens solve flag */
    cv_mem.borrow_mut().sens_solve = SUNFALSE;

    /* if the solve failed return */
    if flag != SUN_SUCCESS {
        return flag;
    }

    /* solve successful */

    /* update the sensitivities based on the final correction from the nonlinear solver */
    let (Ns, znS0, acorS, yS) = {
        let m = cv_mem.borrow();
        (
            m.cv_Ns,
            m.cv_znS[0].clone(),
            m.cv_acorS.clone(),
            m.cv_yS.clone(),
        )
    };
    N_VLinearSumVectorArray(Ns, ONE, &znS0, ONE, &acorS, &yS);

    /* update Jacobian status */
    cv_mem.borrow_mut().cv_jcur = SUNFALSE;

    flag
}

/*
 * cvStgr1Nls
 *
 * This is a high-level routine that attempts to solve the i-th
 * sensitivity linear system using the attached nonlinear solver
 * once the states y_n were obtained and passed the error test.
 */

pub(crate) fn cvStgr1Nls(cv_mem: &CVodeMem, is: i32) -> i32 {
    let callSetup: sunbooleantype;
    let mut nniS1_inc: i64 = 0;
    let mut nnfS1_inc: i64 = 0;

    callSetup = SUNFALSE;
    if cv_mem.borrow().cv_lsetup.is_none() {
        cv_mem.borrow_mut().cv_crateS = ONE;
    }

    /* initial guess for the correction to the predictor */
    let acorS_is = cv_mem.borrow().cv_acorS[is as usize].clone();
    N_VConst(ZERO, &acorS_is);

    /* set sens solve flag */
    cv_mem.borrow_mut().sens_solve = SUNTRUE;

    /* solve the nonlinear system */
    let (NLSstg1, znS0_is, ewtS_is, tq4) = {
        let m = cv_mem.borrow();
        (
            m.NLSstg1.clone().unwrap(),
            m.cv_znS[0][is as usize].clone(),
            m.cv_ewtS[is as usize].clone(),
            m.cv_tq[4],
        )
    };
    let mut nls_mem: Option<Box<dyn Any>> = Some(Box::new(cv_mem.clone()));
    let flag = SUNNonlinSolSolve(
        &NLSstg1,
        &znS0_is,
        &acorS_is,
        &ewtS_is,
        tq4,
        callSetup,
        &mut nls_mem,
    );

    /* increment counters */
    let _ = SUNNonlinSolGetNumIters(&NLSstg1, &mut nniS1_inc);
    cv_mem.borrow_mut().cv_nniS1[is as usize] += nniS1_inc;

    let _ = SUNNonlinSolGetNumConvFails(&NLSstg1, &mut nnfS1_inc);
    cv_mem.borrow_mut().cv_nnfS1[is as usize] += nnfS1_inc;

    /* reset sens solve flag */
    cv_mem.borrow_mut().sens_solve = SUNFALSE;

    /* if the solve failed return */
    if flag != SUN_SUCCESS {
        return flag;
    }

    /* solve successful */

    /* update the sensitivity with the final correction from the nonlinear solver */
    let yS_is = cv_mem.borrow().cv_yS[is as usize].clone();
    N_VLinearSum(ONE, &znS0_is, ONE, &acorS_is, &yS_is);

    /* update Jacobian status */
    cv_mem.borrow_mut().cv_jcur = SUNFALSE;

    flag
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
 * If it failed due to an unrecoverable failure in quad rhs, then we return
 * the value CV_QRHSFUNC_FAIL.
 *
 * If it failed due to an unrecoverable failure in sensi rhs, then we return
 * the value CV_SRHSFUNC_FAIL.
 *
 * If it failed due to an unrecoverable failure in sensi quad rhs, then we
 * return the value CV_QSRHSFUNC_FAIL.
 *
 * Otherwise, a recoverable failure occurred when solving the nonlinear system
 * (cvNls returned SUN_NLS_CONV_RECVR, RHSFUNC_RECVR, or SRHSFUNC_RECVR).
 *
 * If ncf is now equal to maxncf or |h| = hmin, we return the value
 * CV_CONV_FAILURE (if SUN_NLS_CONV_RECVR),
 * CV_REPTD_RHSFUNC_ERR (if RHSFUNC_RECVR), or
 * CV_REPTD_SRHSFUNC_ERR (if SRHSFUNC_RECVR).
 * Otherwise, we set *nflagPtr = PREV_CONV_FAIL and return the value
 * PREDICT_AGAIN, telling cvStep to reattempt the step.
 *
 * PORTING NOTE: C passes `ncfPtr`/`ncfnPtr` as raw pointers that may point
 * INTO the mem (`&cv_mem->cv_ncfn`, `&cv_mem->cv_ncfnS`,
 * `&cv_mem->cv_ncfS1[is]`, `&cv_mem->cv_ncfnS1[is]`). Rust cannot alias the
 * mem that way, so the CALLER copies the counter into a local, passes
 * `&mut local`, and stores it back at the call site. This is exactly
 * equivalent: nothing reachable from cvHandleNFlag reads or writes those
 * counters other than through these two pointers.
 */

pub(crate) fn cvHandleNFlag(
    cv_mem: &CVodeMem,
    nflagPtr: &mut i32,
    saved_t: sunrealtype,
    ncfPtr: &mut i32,
    ncfnPtr: &mut i64,
) -> i32 {
    let nflag: i32;

    nflag = *nflagPtr;

    if nflag == CV_SUCCESS {
        return DO_ERROR_TEST;
    }

    /* The nonlinear soln. failed; increment ncfn and restore zn */
    *ncfnPtr += 1;
    cvRestore(cv_mem, saved_t);

    /* Return if failed unrecoverably */
    if nflag < 0 {
        if nflag == CV_LSETUP_FAIL {
            return CV_LSETUP_FAIL;
        } else if nflag == CV_LSOLVE_FAIL {
            return CV_LSOLVE_FAIL;
        } else if nflag == CV_RHSFUNC_FAIL {
            return CV_RHSFUNC_FAIL;
        } else if nflag == CV_QRHSFUNC_FAIL {
            return CV_QRHSFUNC_FAIL;
        } else if nflag == CV_SRHSFUNC_FAIL {
            return CV_SRHSFUNC_FAIL;
        } else if nflag == CV_QSRHSFUNC_FAIL {
            return CV_QSRHSFUNC_FAIL;
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
            if nflag == QRHSFUNC_RECVR {
                return CV_REPTD_QRHSFUNC_ERR;
            }
            if nflag == SRHSFUNC_RECVR {
                return CV_REPTD_SRHSFUNC_ERR;
            }
            if nflag == QSRHSFUNC_RECVR {
                return CV_REPTD_QSRHSFUNC_ERR;
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

    if cv_mem.borrow().cv_quadr {
        let znQ: Vec<N_Vector> = {
            let m = cv_mem.borrow();
            (0..=q as usize)
                .map(|j| m.cv_znQ[j].clone().unwrap())
                .collect()
        };
        for k in 1..=q {
            let mut j = q;
            while j >= k {
                N_VLinearSum(
                    ONE,
                    &znQ[(j - 1) as usize],
                    -ONE,
                    &znQ[j as usize],
                    &znQ[(j - 1) as usize],
                );
                j -= 1;
            }
        }
    }

    if cv_mem.borrow().cv_sensi {
        let (Ns, znS) = {
            let m = cv_mem.borrow();
            let znS: Vec<Vec<N_Vector>> = (0..=q as usize).map(|j| m.cv_znS[j].clone()).collect();
            (m.cv_Ns, znS)
        };
        for k in 1..=q {
            let mut j = q;
            while j >= k {
                let _ = N_VLinearSumVectorArray(
                    Ns,
                    ONE,
                    &znS[(j - 1) as usize],
                    -ONE,
                    &znS[j as usize],
                    &znS[(j - 1) as usize],
                );
                j -= 1;
            }
        }
    }

    if cv_mem.borrow().cv_quadr_sensi {
        let (Ns, znQS) = {
            let m = cv_mem.borrow();
            let znQS: Vec<Vec<N_Vector>> = (0..=q as usize).map(|j| m.cv_znQS[j].clone()).collect();
            (m.cv_Ns, znQS)
        };
        for k in 1..=q {
            let mut j = q;
            while j >= k {
                let _ = N_VLinearSumVectorArray(
                    Ns,
                    ONE,
                    &znQS[(j - 1) as usize],
                    -ONE,
                    &znQS[j as usize],
                    &znQS[(j - 1) as usize],
                );
                j -= 1;
            }
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
 * This routine performs the local error test, for the state, quadrature,
 * or sensitivity variables. Its last three arguments change depending
 * on which variables the error test is to be performed on.
 *
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
 *     zn from scratch (also znQ and znS if appropriate).
 *     If f() fails, we return CV_RHSFUNC_FAIL or CV_UNREC_RHSFUNC_ERR;
 *     if fQ() fails, we return CV_QRHSFUNC_FAIL or CV_UNREC_QRHSFUNC_ERR;
 *     if cvSensRhsWrapper() fails, we return CV_SRHSFUNC_FAIL or CV_UNREC_SRHSFUNC_ERR;
 *     (no recovery is possible at this stage).
 *
 *   - otherwise, set *nflagPtr to PREV_ERR_FAIL, and return TRY_AGAIN.
 *
 * PORTING NOTE: as for cvHandleNFlag, C's `netfPtr` may point into the mem
 * (`&cv_mem->cv_netf`, `cv_netfQ`, `cv_netfS`, `cv_netfQS`) and `nefPtr`
 * into a local of cvStep. The CALLER copies the counter out, passes
 * `&mut local` and writes it back; nothing reachable from cvDoErrorTest
 * touches those counters other than through this pointer.
 */

pub(crate) fn cvDoErrorTest(
    cv_mem: &CVodeMem,
    nflagPtr: &mut i32,
    saved_t: sunrealtype,
    acor_nrm: sunrealtype,
    nefPtr: &mut i32,
    netfPtr: &mut i64,
    dsmPtr: &mut sunrealtype,
) -> i32 {
    let dsm: sunrealtype;
    let mut retval: i32;

    dsm = acor_nrm * cv_mem.borrow().cv_tq[2];

    /* If est. local error norm dsm passes test, return CV_SUCCESS */
    *dsmPtr = dsm;
    if dsm <= ONE {
        return CV_SUCCESS;
    }

    /* Test failed; increment counters, set nflag, and restore zn array */
    *nefPtr += 1;
    *netfPtr += 1;
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

    /* If already at order 1, restart: reload zn, znQ, znS, znQS from scratch */

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
    retval = cv_call_f(cv_mem, tn, &zn0, &tempv);
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

    if cv_mem.borrow().cv_quadr {
        let tempvQ = cv_mem.borrow().cv_tempvQ.clone().unwrap();
        retval = cv_call_fQ(cv_mem, tn, &zn0, &tempvQ);
        cv_mem.borrow_mut().cv_nfQe += 1;
        if retval < 0 {
            return CV_QRHSFUNC_FAIL;
        }
        if retval > 0 {
            return CV_UNREC_QRHSFUNC_ERR;
        }

        let znQ1 = cv_mem.borrow().cv_znQ[1].clone().unwrap();
        N_VScale(h, &tempvQ, &znQ1);
    }

    if cv_mem.borrow().cv_sensi {
        let (wrk1, wrk2, Ns, znS0, tempvS, znS1) = {
            let m = cv_mem.borrow();
            (
                m.cv_ftemp.clone().unwrap(),
                m.cv_ftempS[0].clone(),
                m.cv_Ns,
                m.cv_znS[0].clone(),
                m.cv_tempvS.clone(),
                m.cv_znS[1].clone(),
            )
        };

        retval = cvSensRhsWrapper(cv_mem, tn, &zn0, &tempv, &znS0, &tempvS, &wrk1, &wrk2);
        if retval < 0 {
            return CV_SRHSFUNC_FAIL;
        }
        if retval > 0 {
            return CV_UNREC_SRHSFUNC_ERR;
        }

        let mut cvals = std::mem::take(&mut cv_mem.borrow_mut().cv_cvals);
        for is in 0..Ns as usize {
            cvals[is] = h;
        }

        retval = N_VScaleVectorArray(Ns, &cvals, &tempvS, &znS1);
        cv_mem.borrow_mut().cv_cvals = cvals;
        if retval != CV_SUCCESS {
            return CV_VECTOROP_ERR;
        }
    }

    if cv_mem.borrow().cv_quadr_sensi {
        let (wrk1, wrk2, Ns, znS0, tempvQ, tempvQS, znQS1) = {
            let m = cv_mem.borrow();
            (
                m.cv_ftemp.clone().unwrap(),
                m.cv_ftempQ.clone().unwrap(),
                m.cv_Ns,
                m.cv_znS[0].clone(),
                m.cv_tempvQ.clone().unwrap(),
                m.cv_tempvQS.clone(),
                m.cv_znQS[1].clone(),
            )
        };

        /* C: `cv_mem->cv_fQS(..., cv_mem->cv_fQS_data, wrk1, wrk2)`.
        Invariant D: `Some(box)` is the module-owned token (a boxed
        `CVodeMem` handle clone when `cv_fQSDQ`), `None` means hand the
        integrator's `cv_user_data` to the callback. */
        {
            let fQS = cv_mem.borrow().cv_fQS.expect("cv_fQS set");
            let mut data = cv_mem.borrow_mut().cv_fQS_data.take();
            let from_user_data = data.is_none();
            if from_user_data {
                data = cv_mem.borrow_mut().cv_user_data.take();
            }
            retval = fQS(
                Ns, tn, &zn0, &znS0, &tempvQ, &tempvQS, &mut data, &wrk1, &wrk2,
            );
            if from_user_data {
                cv_mem.borrow_mut().cv_user_data = data;
            } else {
                cv_mem.borrow_mut().cv_fQS_data = data;
            }
        }
        cv_mem.borrow_mut().cv_nfQSe += 1;
        if retval < 0 {
            return CV_QSRHSFUNC_FAIL;
        }
        if retval > 0 {
            return CV_UNREC_QSRHSFUNC_ERR;
        }

        let mut cvals = std::mem::take(&mut cv_mem.borrow_mut().cv_cvals);
        for is in 0..Ns as usize {
            cvals[is] = h;
        }

        retval = N_VScaleVectorArray(Ns, &cvals, &tempvQS, &znQS1);
        cv_mem.borrow_mut().cv_cvals = cvals;
        if retval != CV_SUCCESS {
            return CV_VECTOROP_ERR;
        }
    }

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

pub(crate) fn cvCompleteStep(cv_mem: &CVodeMem) {
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

    if cv_mem.borrow().cv_quadr {
        let (acorQ, znQvec) = {
            let m = cv_mem.borrow();
            let znQvec: Vec<N_Vector> = (0..=q as usize)
                .map(|j| m.cv_znQ[j].clone().unwrap())
                .collect();
            (m.cv_acorQ.clone().unwrap(), znQvec)
        };
        let _ = N_VScaleAddMulti(q + 1, &lvals, &acorQ, &znQvec, &znQvec);
    }

    if cv_mem.borrow().cv_sensi {
        let (Ns, acorS, znSvec) = {
            let m = cv_mem.borrow();
            let znSvec: Vec<Vec<N_Vector>> =
                (0..=q as usize).map(|j| m.cv_znS[j].clone()).collect();
            (m.cv_Ns, m.cv_acorS.clone(), znSvec)
        };
        let _ = N_VScaleAddMultiVectorArray(Ns, q + 1, &lvals, &acorS, &znSvec, &znSvec);
    }

    if cv_mem.borrow().cv_quadr_sensi {
        let (Ns, acorQS, znQSvec) = {
            let m = cv_mem.borrow();
            let znQSvec: Vec<Vec<N_Vector>> =
                (0..=q as usize).map(|j| m.cv_znQS[j].clone()).collect();
            (m.cv_Ns, m.cv_acorQS.clone(), znQSvec)
        };
        let _ = N_VScaleAddMultiVectorArray(Ns, q + 1, &lvals, &acorQS, &znQSvec, &znQSvec);
    }

    /* If necessary, store Delta_n in zn[qmax] to be used in order increase.
     * This actually will be Delta_{n-1} in the ELTE at q+1 since it happens at
     * the next to last step of order q before a possible one at order q+1
     */

    cv_mem.borrow_mut().cv_qwait -= 1;
    let (qwait, q, qmax) = {
        let m = cv_mem.borrow();
        (m.cv_qwait, m.cv_q, m.cv_qmax)
    };
    if (qwait == 1) && (q != qmax) {
        let znqmax = cv_mem.borrow().cv_zn[qmax as usize].clone().unwrap();
        N_VScale(ONE, &acor, &znqmax);

        if cv_mem.borrow().cv_quadr {
            let (acorQ, znQqmax) = {
                let m = cv_mem.borrow();
                (
                    m.cv_acorQ.clone().unwrap(),
                    m.cv_znQ[qmax as usize].clone().unwrap(),
                )
            };
            N_VScale(ONE, &acorQ, &znQqmax);
        }

        if cv_mem.borrow().cv_sensi {
            let (Ns, acorS, znSqmax) = {
                let m = cv_mem.borrow();
                (m.cv_Ns, m.cv_acorS.clone(), m.cv_znS[qmax as usize].clone())
            };
            let mut cvals = std::mem::take(&mut cv_mem.borrow_mut().cv_cvals);
            for is in 0..Ns as usize {
                cvals[is] = ONE;
            }

            let _ = N_VScaleVectorArray(Ns, &cvals, &acorS, &znSqmax);
            cv_mem.borrow_mut().cv_cvals = cvals;
        }

        if cv_mem.borrow().cv_quadr_sensi {
            let (Ns, acorQS, znQSqmax) = {
                let m = cv_mem.borrow();
                (
                    m.cv_Ns,
                    m.cv_acorQS.clone(),
                    m.cv_znQS[qmax as usize].clone(),
                )
            };
            let mut cvals = std::mem::take(&mut cv_mem.borrow_mut().cv_cvals);
            for is in 0..Ns as usize {
                cvals[is] = ONE;
            }

            let _ = N_VScaleVectorArray(Ns, &cvals, &acorQS, &znQSqmax);
            cv_mem.borrow_mut().cv_cvals = cvals;
        }

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

pub(crate) fn cvPrepareNextStep(cv_mem: &CVodeMem, dsm: sunrealtype) {
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

pub(crate) fn cvSetEta(cv_mem: &CVodeMem) {
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

pub(crate) fn cvComputeEtaqm1(cv_mem: &CVodeMem) -> sunrealtype {
    let mut ddn: sunrealtype;

    cv_mem.borrow_mut().cv_etaqm1 = ZERO;
    if cv_mem.borrow().cv_q > 1 {
        let (znq, ewt) = {
            let m = cv_mem.borrow();
            (
                m.cv_zn[m.cv_q as usize].clone().unwrap(),
                m.cv_ewt.clone().unwrap(),
            )
        };
        ddn = N_VWrmsNorm(&znq, &ewt);

        let (quadr, errconQ, sensi, errconS, quadr_sensi, errconQS) = {
            let m = cv_mem.borrow();
            (
                m.cv_quadr,
                m.cv_errconQ,
                m.cv_sensi,
                m.cv_errconS,
                m.cv_quadr_sensi,
                m.cv_errconQS,
            )
        };

        if quadr && errconQ {
            let (znQq, ewtQ) = {
                let m = cv_mem.borrow();
                (
                    m.cv_znQ[m.cv_q as usize].clone().unwrap(),
                    m.cv_ewtQ.clone().unwrap(),
                )
            };
            ddn = cvQuadUpdateNorm(cv_mem, ddn, &znQq, &ewtQ);
        }

        if sensi && errconS {
            let (znSq, ewtS) = {
                let m = cv_mem.borrow();
                (m.cv_znS[m.cv_q as usize].clone(), m.cv_ewtS.clone())
            };
            ddn = cvSensUpdateNorm(cv_mem, ddn, &znSq, &ewtS);
        }

        if quadr_sensi && errconQS {
            let (znQSq, ewtQS) = {
                let m = cv_mem.borrow();
                (m.cv_znQS[m.cv_q as usize].clone(), m.cv_ewtQS.clone())
            };
            ddn = cvQuadSensUpdateNorm(cv_mem, ddn, &znQSq, &ewtQS);
        }

        let (tq1, q) = {
            let m = cv_mem.borrow();
            (m.cv_tq[1], m.cv_q)
        };
        ddn = ddn * tq1;
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

pub(crate) fn cvComputeEtaqp1(cv_mem: &CVodeMem) -> sunrealtype {
    let mut dup: sunrealtype;

    cv_mem.borrow_mut().cv_etaqp1 = ZERO;
    let (q, qmax) = {
        let m = cv_mem.borrow();
        (m.cv_q, m.cv_qmax)
    };
    if q != qmax {
        if cv_mem.borrow().cv_saved_tq5 == ZERO {
            return cv_mem.borrow().cv_etaqp1;
        }
        let (cquot, znqmax, acor, tempv, ewt) = {
            let m = cv_mem.borrow();
            (
                (m.cv_tq[5] / m.cv_saved_tq5) * SUNRpowerI(m.cv_h / m.cv_tau[2], m.cv_L),
                m.cv_zn[m.cv_qmax as usize].clone().unwrap(),
                m.cv_acor.clone().unwrap(),
                m.cv_tempv.clone().unwrap(),
                m.cv_ewt.clone().unwrap(),
            )
        };
        N_VLinearSum(-cquot, &znqmax, ONE, &acor, &tempv);
        dup = N_VWrmsNorm(&tempv, &ewt);

        let (quadr, errconQ, sensi, errconS, quadr_sensi, errconQS) = {
            let m = cv_mem.borrow();
            (
                m.cv_quadr,
                m.cv_errconQ,
                m.cv_sensi,
                m.cv_errconS,
                m.cv_quadr_sensi,
                m.cv_errconQS,
            )
        };

        if quadr && errconQ {
            let (znQqmax, acorQ, tempvQ, ewtQ) = {
                let m = cv_mem.borrow();
                (
                    m.cv_znQ[m.cv_qmax as usize].clone().unwrap(),
                    m.cv_acorQ.clone().unwrap(),
                    m.cv_tempvQ.clone().unwrap(),
                    m.cv_ewtQ.clone().unwrap(),
                )
            };
            N_VLinearSum(-cquot, &znQqmax, ONE, &acorQ, &tempvQ);
            dup = cvQuadUpdateNorm(cv_mem, dup, &tempvQ, &ewtQ);
        }

        if sensi && errconS {
            let (Ns, znSqmax, acorS, tempvS, ewtS) = {
                let m = cv_mem.borrow();
                (
                    m.cv_Ns,
                    m.cv_znS[m.cv_qmax as usize].clone(),
                    m.cv_acorS.clone(),
                    m.cv_tempvS.clone(),
                    m.cv_ewtS.clone(),
                )
            };
            let _ = N_VLinearSumVectorArray(Ns, -cquot, &znSqmax, ONE, &acorS, &tempvS);

            dup = cvSensUpdateNorm(cv_mem, dup, &tempvS, &ewtS);
        }

        if quadr_sensi && errconQS {
            let (Ns, znQSqmax, acorQS, tempvQS, ewtQS) = {
                let m = cv_mem.borrow();
                (
                    m.cv_Ns,
                    m.cv_znQS[m.cv_qmax as usize].clone(),
                    m.cv_acorQS.clone(),
                    m.cv_tempvQS.clone(),
                    m.cv_ewtQS.clone(),
                )
            };
            let _ = N_VLinearSumVectorArray(Ns, -cquot, &znQSqmax, ONE, &acorQS, &tempvQS);

            /* NOTE (faithful to upstream): cvodes.c calls cvSensUpdateNorm
            here, not cvQuadSensUpdateNorm. The two have identical bodies
            (cvSensNorm == cvQuadSensNorm), so this is behaviour-preserving. */
            dup = cvSensUpdateNorm(cv_mem, dup, &tempvQS, &ewtQS);
        }

        let (tq3, L) = {
            let m = cv_mem.borrow();
            (m.cv_tq[3], m.cv_L)
        };
        dup = dup * tq3;
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

pub(crate) fn cvChooseEta(cv_mem: &CVodeMem) {
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

                let (quadr, errconQ, sensi, errconS, quadr_sensi, errconQS) = {
                    let m = cv_mem.borrow();
                    (
                        m.cv_quadr,
                        m.cv_errconQ,
                        m.cv_sensi,
                        m.cv_errconS,
                        m.cv_quadr_sensi,
                        m.cv_errconQS,
                    )
                };

                if quadr && errconQ {
                    let (acorQ, znQqmax) = {
                        let m = cv_mem.borrow();
                        (
                            m.cv_acorQ.clone().unwrap(),
                            m.cv_znQ[m.cv_qmax as usize].clone().unwrap(),
                        )
                    };
                    N_VScale(ONE, &acorQ, &znQqmax);
                }

                if sensi && errconS {
                    let (Ns, acorS, znSqmax) = {
                        let m = cv_mem.borrow();
                        (
                            m.cv_Ns,
                            m.cv_acorS.clone(),
                            m.cv_znS[m.cv_qmax as usize].clone(),
                        )
                    };
                    let mut cvals = std::mem::take(&mut cv_mem.borrow_mut().cv_cvals);
                    for is in 0..Ns as usize {
                        cvals[is] = ONE;
                    }

                    let _ = N_VScaleVectorArray(Ns, &cvals, &acorS, &znSqmax);
                    cv_mem.borrow_mut().cv_cvals = cvals;
                }

                if quadr_sensi && errconQS {
                    let (Ns, acorQS, znQSqmax) = {
                        let m = cv_mem.borrow();
                        (
                            m.cv_Ns,
                            m.cv_acorQS.clone(),
                            m.cv_znQS[m.cv_qmax as usize].clone(),
                        )
                    };
                    let mut cvals = std::mem::take(&mut cv_mem.borrow_mut().cv_cvals);
                    for is in 0..Ns as usize {
                        cvals[is] = ONE;
                    }

                    let _ = N_VScaleVectorArray(Ns, &cvals, &acorQS, &znQSqmax);
                    cv_mem.borrow_mut().cv_cvals = cvals;
                }
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

pub(crate) fn cvHandleFailure(cv_mem: &CVodeMem, flag: i32) -> i32 {
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
        CV_QRHSFUNC_FAIL => {
            cvProcessError(
                Some(cv_mem),
                CV_QRHSFUNC_FAIL,
                line!() as i32,
                "cvHandleFailure",
                file!(),
                &MSGCV_QRHSFUNC_FAILED(tn),
            );
        }
        CV_UNREC_QRHSFUNC_ERR => {
            cvProcessError(
                Some(cv_mem),
                CV_UNREC_QRHSFUNC_ERR,
                line!() as i32,
                "cvHandleFailure",
                file!(),
                &MSGCV_QRHSFUNC_UNREC(tn),
            );
        }
        CV_REPTD_QRHSFUNC_ERR => {
            cvProcessError(
                Some(cv_mem),
                CV_REPTD_QRHSFUNC_ERR,
                line!() as i32,
                "cvHandleFailure",
                file!(),
                &MSGCV_QRHSFUNC_REPTD(tn),
            );
        }
        CV_SRHSFUNC_FAIL => {
            cvProcessError(
                Some(cv_mem),
                CV_SRHSFUNC_FAIL,
                line!() as i32,
                "cvHandleFailure",
                file!(),
                &MSGCV_SRHSFUNC_FAILED(tn),
            );
        }
        CV_UNREC_SRHSFUNC_ERR => {
            cvProcessError(
                Some(cv_mem),
                CV_UNREC_SRHSFUNC_ERR,
                line!() as i32,
                "cvHandleFailure",
                file!(),
                &MSGCV_SRHSFUNC_UNREC(tn),
            );
        }
        CV_REPTD_SRHSFUNC_ERR => {
            cvProcessError(
                Some(cv_mem),
                CV_REPTD_SRHSFUNC_ERR,
                line!() as i32,
                "cvHandleFailure",
                file!(),
                &MSGCV_SRHSFUNC_REPTD(tn),
            );
        }
        CV_QSRHSFUNC_FAIL => {
            cvProcessError(
                Some(cv_mem),
                CV_QSRHSFUNC_FAIL,
                line!() as i32,
                "cvHandleFailure",
                file!(),
                &MSGCV_QSRHSFUNC_FAILED(tn),
            );
        }
        CV_UNREC_QSRHSFUNC_ERR => {
            cvProcessError(
                Some(cv_mem),
                CV_UNREC_QSRHSFUNC_ERR,
                line!() as i32,
                "cvHandleFailure",
                file!(),
                &MSGCV_QSRHSFUNC_UNREC(tn),
            );
        }
        CV_REPTD_QSRHSFUNC_ERR => {
            cvProcessError(
                Some(cv_mem),
                CV_REPTD_QSRHSFUNC_ERR,
                line!() as i32,
                "cvHandleFailure",
                file!(),
                &MSGCV_QSRHSFUNC_REPTD(tn),
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
                "CVODES encountered an unrecognized error. Please report this to the SUNDIALS developers at sundials-users@llnl.gov",
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

pub(crate) fn cvBDFStab(cv_mem: &CVodeMem) {
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

    let qprime_ge_q = {
        let m = cv_mem.borrow();
        m.cv_qprime >= m.cv_q
    };
    if qprime_ge_q {
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

pub(crate) fn cvSLdet(cv_mem: &CVodeMem) -> i32 {
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

pub(crate) fn cvRcheck1(cv_mem: &CVodeMem) -> i32 {
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

pub(crate) fn cvRcheck2(cv_mem: &CVodeMem) -> i32 {
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

pub(crate) fn cvRcheck3(cv_mem: &CVodeMem, tout: sunrealtype, itask: i32) -> i32 {
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

pub(crate) fn cvRootfind(cv_mem: &CVodeMem) -> i32 {
    /* Move the mutated rootfinding state into locals for the duration of the
    search (the user's g function is invoked inside the loop; no RefCell borrow
    may be held across it). C writes through the cv_mem fields on every return
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
    /* Only glo/ghi/grout are mutated across the search (grout is written by
    the user's g), so only those move into locals; cv_rootdir and cv_gactive
    are read-only here and are CLONED, and cv_iroots is written only at the
    two terminal points (no callback in between) so it is written in place.
    That keeps every array the public API can reach -- CVodeGetRootInfo
    (cv_iroots), CVodeSetRootDirection (cv_rootdir) -- populated in the mem
    while the user's g function can re-enter, exactly as in C. */
    let (mut glo, mut ghi, mut grout, rootdir, gactive) = {
        let mut m = cv_mem.borrow_mut();
        (
            std::mem::take(&mut m.cv_glo),
            std::mem::take(&mut m.cv_ghi),
            std::mem::take(&mut m.cv_grout),
            m.cv_rootdir.clone(),
            m.cv_gactive.clone(),
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
                {
                    let mut m = cv_mem.borrow_mut();
                    for i in 0..nrtfn {
                        m.cv_iroots[i] = 0;
                        if !gactive[i] {
                            continue;
                        }
                        if (SUNRabs(ghi[i]) == ZERO) && (rootdir[i] as sunrealtype * glo[i] <= ZERO)
                        {
                            m.cv_iroots[i] = if glo[i] > ZERO { -1 } else { 1 };
                        }
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
            {
                let mut m = cv_mem.borrow_mut();
                for i in 0..nrtfn {
                    grout[i] = ghi[i];
                    m.cv_iroots[i] = 0;
                    if !gactive[i] {
                        continue;
                    }
                    if (SUNRabs(ghi[i]) == ZERO) && (rootdir[i] as sunrealtype * glo[i] <= ZERO) {
                        m.cv_iroots[i] = if glo[i] > ZERO { -1 } else { 1 };
                    }
                    if SUNRdifferentsign(glo[i], ghi[i])
                        && (rootdir[i] as sunrealtype * glo[i] <= ZERO)
                    {
                        m.cv_iroots[i] = if glo[i] > ZERO { -1 } else { 1 };
                    }
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
 *
 * NOTE: the signature must stay exactly `CVEwtFn` — `cv_mem->cv_efun` is
 * assigned `Some(cvEwtSet)` in CVodeSStolerances/CVodeSVtolerances.
 */

pub fn cvEwtSet(ycur: &N_Vector, weight: &N_Vector, data: &mut Option<Box<dyn Any>>) -> i32 {
    let flag: i32;

    /* data points to cv_mem here (a boxed CVodeMem handle clone; C's cast of
    a NULL/foreign pointer is UB -> deterministic panic) */
    let cv_mem = data
        .as_mut()
        .and_then(|b| b.downcast_ref::<CVodeMem>())
        .cloned()
        .expect("cvEwtSet data holds CVodeMem");

    let itol = cv_mem.borrow().cv_itol;
    flag = match itol {
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

pub(crate) fn cvEwtSetSS(cv_mem: &CVodeMem, ycur: &N_Vector, weight: &N_Vector) -> i32 {
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

pub(crate) fn cvEwtSetSV(cv_mem: &CVodeMem, ycur: &N_Vector, weight: &N_Vector) -> i32 {
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

/*
 * cvQuadEwtSet
 *
 */

pub(crate) fn cvQuadEwtSet(cv_mem: &CVodeMem, qcur: &N_Vector, weightQ: &N_Vector) -> i32 {
    let flag: i32;

    let itolQ = cv_mem.borrow().cv_itolQ;
    flag = match itolQ {
        CV_SS => cvQuadEwtSetSS(cv_mem, qcur, weightQ),
        CV_SV => cvQuadEwtSetSV(cv_mem, qcur, weightQ),
        _ => 0,
    };

    flag
}

/*
 * cvQuadEwtSetSS
 *
 */

pub(crate) fn cvQuadEwtSetSS(cv_mem: &CVodeMem, qcur: &N_Vector, weightQ: &N_Vector) -> i32 {
    let (tempvQ, reltolQ, SabstolQ, atolQmin0) = {
        let m = cv_mem.borrow();
        (
            m.cv_tempvQ.clone().unwrap(),
            m.cv_reltolQ,
            m.cv_SabstolQ,
            m.cv_atolQmin0,
        )
    };
    N_VAbs(qcur, &tempvQ);
    N_VScale(reltolQ, &tempvQ, &tempvQ);
    N_VAddConst(&tempvQ, SabstolQ, &tempvQ);
    if atolQmin0 {
        if N_VMin(&tempvQ) <= ZERO {
            return -1;
        }
    }
    N_VInv(&tempvQ, weightQ);
    0
}

/*
 * cvQuadEwtSetSV
 *
 */

pub(crate) fn cvQuadEwtSetSV(cv_mem: &CVodeMem, qcur: &N_Vector, weightQ: &N_Vector) -> i32 {
    let (tempvQ, reltolQ, VabstolQ, atolQmin0) = {
        let m = cv_mem.borrow();
        (
            m.cv_tempvQ.clone().unwrap(),
            m.cv_reltolQ,
            m.cv_VabstolQ.clone().unwrap(),
            m.cv_atolQmin0,
        )
    };
    N_VAbs(qcur, &tempvQ);
    N_VLinearSum(reltolQ, &tempvQ, ONE, &VabstolQ, &tempvQ);
    if atolQmin0 {
        if N_VMin(&tempvQ) <= ZERO {
            return -1;
        }
    }
    N_VInv(&tempvQ, weightQ);
    0
}

/*
 * cvSensEwtSet
 *
 */

pub(crate) fn cvSensEwtSet(cv_mem: &CVodeMem, yScur: &[N_Vector], weightS: &[N_Vector]) -> i32 {
    let flag: i32;

    let itolS = cv_mem.borrow().cv_itolS;
    flag = match itolS {
        CV_EE => cvSensEwtSetEE(cv_mem, yScur, weightS),
        CV_SS => cvSensEwtSetSS(cv_mem, yScur, weightS),
        CV_SV => cvSensEwtSetSV(cv_mem, yScur, weightS),
        _ => 0,
    };

    flag
}

/*
 * cvSensEwtSetEE
 *
 * In this case, the error weight vector for the i-th sensitivity is set to
 *
 * ewtS_i = pbar_i * efun(pbar_i*yS_i)
 *
 * In other words, the scaled sensitivity pbar_i * yS_i has the same error
 * weight vector calculation as the solution vector.
 *
 */

pub(crate) fn cvSensEwtSetEE(cv_mem: &CVodeMem, yScur: &[N_Vector], weightS: &[N_Vector]) -> i32 {
    let pyS: N_Vector;

    /* Use tempvS[0] as temporary storage for the scaled sensitivity */
    pyS = cv_mem.borrow().cv_tempvS[0].clone();

    let Ns = cv_mem.borrow().cv_Ns;
    for is in 0..Ns as usize {
        let pbar_is = cv_mem.borrow().cv_pbar[is];
        N_VScale(pbar_is, &yScur[is], &pyS);
        let flag: i32 = cv_call_efun(cv_mem, &pyS, &weightS[is]);
        if flag != 0 {
            return -1;
        }
        N_VScale(pbar_is, &weightS[is], &weightS[is]);
    }
    0
}

/*
 * cvSensEwtSetSS
 *
 */

pub(crate) fn cvSensEwtSetSS(cv_mem: &CVodeMem, yScur: &[N_Vector], weightS: &[N_Vector]) -> i32 {
    let (Ns, tempv, reltolS) = {
        let m = cv_mem.borrow();
        (m.cv_Ns, m.cv_tempv.clone().unwrap(), m.cv_reltolS)
    };

    for is in 0..Ns as usize {
        let (SabstolS_is, atolSmin0_is) = {
            let m = cv_mem.borrow();
            (m.cv_SabstolS[is], m.cv_atolSmin0[is])
        };
        N_VAbs(&yScur[is], &tempv);
        N_VScale(reltolS, &tempv, &tempv);
        N_VAddConst(&tempv, SabstolS_is, &tempv);
        if atolSmin0_is {
            if N_VMin(&tempv) <= ZERO {
                return -1;
            }
        }
        N_VInv(&tempv, &weightS[is]);
    }
    0
}

/*
 * cvSensEwtSetSV
 *
 */

pub(crate) fn cvSensEwtSetSV(cv_mem: &CVodeMem, yScur: &[N_Vector], weightS: &[N_Vector]) -> i32 {
    let (Ns, tempv, reltolS) = {
        let m = cv_mem.borrow();
        (m.cv_Ns, m.cv_tempv.clone().unwrap(), m.cv_reltolS)
    };

    for is in 0..Ns as usize {
        let (VabstolS_is, atolSmin0_is) = {
            let m = cv_mem.borrow();
            (m.cv_VabstolS[is].clone(), m.cv_atolSmin0[is])
        };
        N_VAbs(&yScur[is], &tempv);
        N_VLinearSum(reltolS, &tempv, ONE, &VabstolS_is, &tempv);
        if atolSmin0_is {
            if N_VMin(&tempv) <= ZERO {
                return -1;
            }
        }
        N_VInv(&tempv, &weightS[is]);
    }
    0
}

/*
 * cvQuadSensEwtSet
 *
 */

pub(crate) fn cvQuadSensEwtSet(
    cv_mem: &CVodeMem,
    yQScur: &[N_Vector],
    weightQS: &[N_Vector],
) -> i32 {
    let flag: i32;

    let itolQS = cv_mem.borrow().cv_itolQS;
    flag = match itolQS {
        CV_EE => cvQuadSensEwtSetEE(cv_mem, yQScur, weightQS),
        CV_SS => cvQuadSensEwtSetSS(cv_mem, yQScur, weightQS),
        CV_SV => cvQuadSensEwtSetSV(cv_mem, yQScur, weightQS),
        _ => 0,
    };

    flag
}

/*
 * cvQuadSensEwtSetEE
 *
 * In this case, the error weight vector for the i-th quadrature sensitivity
 * is set to
 *
 * ewtQS_i = pbar_i * cvQuadEwtSet(pbar_i*yQS_i)
 *
 * In other words, the scaled sensitivity pbar_i * yQS_i has the same error
 * weight vector calculation as the quadrature vector.
 *
 */
pub(crate) fn cvQuadSensEwtSetEE(
    cv_mem: &CVodeMem,
    yQScur: &[N_Vector],
    weightQS: &[N_Vector],
) -> i32 {
    let pyS: N_Vector;

    /* Use tempvQS[0] as temporary storage for the scaled sensitivity */
    pyS = cv_mem.borrow().cv_tempvQS[0].clone();

    let Ns = cv_mem.borrow().cv_Ns;
    for is in 0..Ns as usize {
        let pbar_is = cv_mem.borrow().cv_pbar[is];
        N_VScale(pbar_is, &yQScur[is], &pyS);
        let flag: i32 = cvQuadEwtSet(cv_mem, &pyS, &weightQS[is]);
        if flag != 0 {
            return -1;
        }
        N_VScale(pbar_is, &weightQS[is], &weightQS[is]);
    }
    0
}

pub(crate) fn cvQuadSensEwtSetSS(
    cv_mem: &CVodeMem,
    yQScur: &[N_Vector],
    weightQS: &[N_Vector],
) -> i32 {
    let (Ns, tempvQ, reltolQS) = {
        let m = cv_mem.borrow();
        (m.cv_Ns, m.cv_tempvQ.clone().unwrap(), m.cv_reltolQS)
    };

    for is in 0..Ns as usize {
        let (SabstolQS_is, atolQSmin0_is) = {
            let m = cv_mem.borrow();
            (m.cv_SabstolQS[is], m.cv_atolQSmin0[is])
        };
        N_VAbs(&yQScur[is], &tempvQ);
        N_VScale(reltolQS, &tempvQ, &tempvQ);
        N_VAddConst(&tempvQ, SabstolQS_is, &tempvQ);
        if atolQSmin0_is {
            if N_VMin(&tempvQ) <= ZERO {
                return -1;
            }
        }
        N_VInv(&tempvQ, &weightQS[is]);
    }
    0
}

pub(crate) fn cvQuadSensEwtSetSV(
    cv_mem: &CVodeMem,
    yQScur: &[N_Vector],
    weightQS: &[N_Vector],
) -> i32 {
    let (Ns, tempvQ, reltolQS) = {
        let m = cv_mem.borrow();
        (m.cv_Ns, m.cv_tempvQ.clone().unwrap(), m.cv_reltolQS)
    };

    for is in 0..Ns as usize {
        let (VabstolQS_is, atolQSmin0_is) = {
            let m = cv_mem.borrow();
            (m.cv_VabstolQS[is].clone(), m.cv_atolQSmin0[is])
        };
        N_VAbs(&yQScur[is], &tempvQ);
        N_VLinearSum(reltolQS, &tempvQ, ONE, &VabstolQS_is, &tempvQ);
        if atolQSmin0_is {
            if N_VMin(&tempvQ) <= ZERO {
                return -1;
            }
        }
        N_VInv(&tempvQ, &weightQS[is]);
    }
    0
}

/*
 * -----------------------------------------------------------------
 * Functions for combined norms
 * -----------------------------------------------------------------
 */

/*
 * cvQuadUpdateNorm
 *
 * Updates the norm old_nrm to account for all quadratures.
 *
 * (`cv_mem` is `SUNDIALS_MAYBE_UNUSED` in C; kept in the signature for
 * argument-list fidelity and named `_cv_mem` so the port stays
 * warning-free.)
 */

pub(crate) fn cvQuadUpdateNorm(
    _cv_mem: &CVodeMem,
    old_nrm: sunrealtype,
    xQ: &N_Vector,
    wQ: &N_Vector,
) -> sunrealtype {
    let qnrm: sunrealtype;

    qnrm = N_VWrmsNorm(xQ, wQ);
    if old_nrm > qnrm {
        old_nrm
    } else {
        qnrm
    }
}

/*
 * cvSensNorm
 *
 * This routine returns the maximum over the weighted root mean
 * square norm of xS with weight vectors wS:
 *
 *  max { wrms(xS[0],wS[0]) ... wrms(xS[Ns-1],wS[Ns-1]) }
 *
 * Called by cvSensUpdateNorm or directly in the CV_STAGGERED approach
 * during the NLS solution and before the error test.
 */

pub fn cvSensNorm(cv_mem: &CVodeMem, xS: &[N_Vector], wS: &[N_Vector]) -> sunrealtype {
    let mut nrm: sunrealtype;

    /* C scribbles the per-sensitivity norms into the mem's `cv_cvals`
    scratch array; the port takes it out and puts it back so no mem borrow
    is held across the vector-array op (callers must therefore not hold a
    borrow of the mem across this call). */
    let Ns = cv_mem.borrow().cv_Ns;
    let mut cvals = std::mem::take(&mut cv_mem.borrow_mut().cv_cvals);

    let _ = N_VWrmsNormVectorArray(Ns, xS, wS, &mut cvals);

    nrm = cvals[0];
    for is in 1..Ns as usize {
        if cvals[is] > nrm {
            nrm = cvals[is];
        }
    }

    cv_mem.borrow_mut().cv_cvals = cvals;

    nrm
}

/*
 * cvSensUpdateNorm
 *
 * Updates the norm old_nrm to account for all sensitivities.
 */

pub fn cvSensUpdateNorm(
    cv_mem: &CVodeMem,
    old_nrm: sunrealtype,
    xS: &[N_Vector],
    wS: &[N_Vector],
) -> sunrealtype {
    let snrm: sunrealtype;

    snrm = cvSensNorm(cv_mem, xS, wS);
    if old_nrm > snrm {
        old_nrm
    } else {
        snrm
    }
}

/*
 * cvQuadSensNorm
 *
 * This routine returns the maximum over the weighted root mean
 * square norm of xQS with weight vectors wQS:
 *
 *  max { wrms(xQS[0],wS[0]) ... wrms(xQS[Ns-1],wS[Ns-1]) }
 *
 * Called by cvQuadSensUpdateNorm.
 */

pub(crate) fn cvQuadSensNorm(cv_mem: &CVodeMem, xQS: &[N_Vector], wQS: &[N_Vector]) -> sunrealtype {
    let mut nrm: sunrealtype;

    let Ns = cv_mem.borrow().cv_Ns;
    let mut cvals = std::mem::take(&mut cv_mem.borrow_mut().cv_cvals);

    let _ = N_VWrmsNormVectorArray(Ns, xQS, wQS, &mut cvals);

    nrm = cvals[0];
    for is in 1..Ns as usize {
        if cvals[is] > nrm {
            nrm = cvals[is];
        }
    }

    cv_mem.borrow_mut().cv_cvals = cvals;

    nrm
}

/*
 * cvSensUpdateNorm
 *
 * Updates the norm old_nrm to account for all quadrature sensitivities.
 */

pub(crate) fn cvQuadSensUpdateNorm(
    cv_mem: &CVodeMem,
    old_nrm: sunrealtype,
    xQS: &[N_Vector],
    wQS: &[N_Vector],
) -> sunrealtype {
    let snrm: sunrealtype;

    snrm = cvQuadSensNorm(cv_mem, xQS, wQS);
    if old_nrm > snrm {
        old_nrm
    } else {
        snrm
    }
}

/*
 * -----------------------------------------------------------------
 * Wrappers for sensitivity RHS
 * -----------------------------------------------------------------
 */

/*
 * cvSensRhsWrapper
 *
 * CVSensRhs is a high level routine that returns right hand side
 * of sensitivity equations. Depending on the 'ifS' flag, it either
 * calls directly the fS routine (ifS=CV_ALLSENS) or (if ifS=CV_ONESENS)
 * calls the fS1 routine in a loop over all sensitivities.
 *
 * CVSensRhs is called:
 *  (*) by CVode at the first step
 *  (*) by cvYddNorm if errcon=SUNTRUE
 *  (*) by the nonlinear solver if ism=CV_SIMULTANEOUS
 *  (*) by cvDoErrorTest when restarting from scratch
 *  (*) in the corrector loop if ism=CV_STAGGERED
 *  (*) by cvStgrDoErrorTest when restarting from scratch
 *
 * The return value is that of the sensitivity RHS function fS,
 *
 */

pub fn cvSensRhsWrapper(
    cv_mem: &CVodeMem,
    time: sunrealtype,
    ycur: &N_Vector,
    fcur: &N_Vector,
    yScur: &[N_Vector],
    fScur: &[N_Vector],
    temp1: &N_Vector,
    temp2: &N_Vector,
) -> i32 {
    let mut retval: i32 = 0;

    let (ifS, Ns) = {
        let m = cv_mem.borrow();
        (m.cv_ifS, m.cv_Ns)
    };

    if ifS == CV_ALLSENS {
        let fS = cv_mem.borrow().cv_fS.expect("cv_fS set");
        /* C: `cv_mem->cv_fS_data` is `cv_mem` when the internal DQ RHS is in
        use and `cv_mem->cv_user_data` otherwise. Invariant D: `Some(box)` is
        the module-owned token, `None` means hand over `cv_user_data`. */
        let mut data = cv_mem.borrow_mut().cv_fS_data.take();
        let from_user_data = data.is_none();
        if from_user_data {
            data = cv_mem.borrow_mut().cv_user_data.take();
        }
        retval = fS(Ns, time, ycur, fcur, yScur, fScur, &mut data, temp1, temp2);
        if from_user_data {
            cv_mem.borrow_mut().cv_user_data = data;
        } else {
            cv_mem.borrow_mut().cv_fS_data = data;
        }
        cv_mem.borrow_mut().cv_nfSe += 1;
    } else {
        for is in 0..Ns as usize {
            let fS1 = cv_mem.borrow().cv_fS1.expect("cv_fS1 set");
            let mut data = cv_mem.borrow_mut().cv_fS_data.take();
            let from_user_data = data.is_none();
            if from_user_data {
                data = cv_mem.borrow_mut().cv_user_data.take();
            }
            retval = fS1(
                Ns, time, ycur, fcur, is as i32, &yScur[is], &fScur[is], &mut data, temp1, temp2,
            );
            if from_user_data {
                cv_mem.borrow_mut().cv_user_data = data;
            } else {
                cv_mem.borrow_mut().cv_fS_data = data;
            }
            cv_mem.borrow_mut().cv_nfSe += 1;
            if retval != 0 {
                break;
            }
        }
    }

    retval
}

/*
 * cvSensRhs1Wrapper
 *
 * cvSensRhs1Wrapper is a high level routine that returns right-hand
 * side of the is-th sensitivity equation.
 *
 * cvSensRhs1Wrapper is called only during the CV_STAGGERED1 corrector loop
 * (ifS must be CV_ONESENS, otherwise CVodeSensInit would have
 * issued an error message).
 *
 * The return value is that of the sensitivity RHS function fS1,
 */

pub fn cvSensRhs1Wrapper(
    cv_mem: &CVodeMem,
    time: sunrealtype,
    ycur: &N_Vector,
    fcur: &N_Vector,
    is: i32,
    yScur: &N_Vector,
    fScur: &N_Vector,
    temp1: &N_Vector,
    temp2: &N_Vector,
) -> i32 {
    let retval: i32;

    let (fS1, Ns) = {
        let m = cv_mem.borrow();
        (m.cv_fS1.expect("cv_fS1 set"), m.cv_Ns)
    };
    let mut data = cv_mem.borrow_mut().cv_fS_data.take();
    let from_user_data = data.is_none();
    if from_user_data {
        data = cv_mem.borrow_mut().cv_user_data.take();
    }
    retval = fS1(
        Ns, time, ycur, fcur, is, yScur, fScur, &mut data, temp1, temp2,
    );
    if from_user_data {
        cv_mem.borrow_mut().cv_user_data = data;
    } else {
        cv_mem.borrow_mut().cv_fS_data = data;
    }
    cv_mem.borrow_mut().cv_nfSe += 1;

    retval
}

/*
 * -----------------------------------------------------------------
 * Internal DQ approximations for sensitivity RHS
 * -----------------------------------------------------------------
 */

/*
 * `cv_mem->cv_p` is the caller's own parameter array (C stores the
 * POINTER; the port stores a clone of the caller's `SensParams` handle —
 * see `cvodes_impl::SensParams`), so the perturbations below are visible
 * to the user's `f`/`fQ` through their `user_data`, exactly as in C.
 *
 * Both accessors borrow the parameter cell for the duration of one
 * statement only — never across the user callback that follows.
 */

/// C: `psave = cv_mem->cv_p[which];`
fn cv_p_get(cv_mem: &CVodeMem, which: i32) -> sunrealtype {
    let p = cv_mem.borrow().cv_p.clone().expect("cv_p set");
    let psave = p.borrow()[which as usize];

    psave
}

/// C: `cv_mem->cv_p[which] = value;`
fn cv_p_set(cv_mem: &CVodeMem, which: i32, value: sunrealtype) {
    let p = cv_mem.borrow().cv_p.clone().expect("cv_p set");
    p.borrow_mut()[which as usize] = value;
}

/*
 * cvSensRhsInternalDQ   - internal CVSensRhsFn
 *
 * cvSensRhsInternalDQ computes right hand side of all sensitivity equations
 * by finite differences
 *
 * NOTE: the signature must stay exactly `CVSensRhsFn` — `cv_mem->cv_fS` is
 * assigned `Some(cvSensRhsInternalDQ)` in CVodeSensInit.
 */

pub fn cvSensRhsInternalDQ(
    Ns: i32,
    t: sunrealtype,
    y: &N_Vector,
    ydot: &N_Vector,
    yS: &[N_Vector],
    ySdot: &[N_Vector],
    cvode_mem: &mut Option<Box<dyn Any>>,
    ytemp: &N_Vector,
    ftemp: &N_Vector,
) -> i32 {
    let mut retval: i32;

    for is in 0..Ns as usize {
        retval = cvSensRhs1InternalDQ(
            Ns, t, y, ydot, is as i32, &yS[is], &ySdot[is], cvode_mem, ytemp, ftemp,
        );
        if retval != 0 {
            return retval;
        }
    }

    0
}

/*
 * cvSensRhs1InternalDQ   - internal CVSensRhs1Fn
 *
 * cvSensRhs1InternalDQ computes the right hand side of the is-th sensitivity
 * equation by finite differences
 *
 * cvSensRhs1InternalDQ returns 0 if successful. Otherwise it returns the
 * non-zero return value from f().
 *
 * NOTE: the signature must stay exactly `CVSensRhs1Fn` — `cv_mem->cv_fS1` is
 * assigned `Some(cvSensRhs1InternalDQ)` in CVodeSensInit1. `Ns` is
 * `SUNDIALS_MAYBE_UNUSED` in C, hence `_Ns` here.
 */

pub fn cvSensRhs1InternalDQ(
    _Ns: i32,
    t: sunrealtype,
    y: &N_Vector,
    ydot: &N_Vector,
    is: i32,
    yS: &N_Vector,
    ySdot: &N_Vector,
    cvode_mem: &mut Option<Box<dyn Any>>,
    ytemp: &N_Vector,
    ftemp: &N_Vector,
) -> i32 {
    let mut retval: i32;
    let method: i32;
    let mut nfel: i32 = 0;
    let which: i32;
    let psave: sunrealtype;
    let pbari: sunrealtype;
    let delta: sunrealtype;
    let rdelta: sunrealtype;
    let Deltap: sunrealtype;
    let rDeltap: sunrealtype;
    let Deltay: sunrealtype;
    let rDeltay: sunrealtype;
    let norms: sunrealtype;
    let ratio: sunrealtype;

    /* local variables for fused vector operations */
    let mut cvals: [sunrealtype; 3] = [ZERO; 3];
    let mut Xvecs: Vec<N_Vector> = Vec::with_capacity(3);

    /* cvode_mem is passed here as user data */
    let cv_mem = cvode_mem
        .as_mut()
        .and_then(|b| b.downcast_ref::<CVodeMem>())
        .cloned()
        .expect("cvSensRhs1InternalDQ cvode_mem holds CVodeMem");
    let cv_mem = &cv_mem;

    {
        let m = cv_mem.borrow();
        delta = SUNRsqrt(SUNMAX(m.cv_reltol, m.cv_uround));
    }
    rdelta = ONE / delta;

    pbari = cv_mem.borrow().cv_pbar[is as usize];

    which = cv_mem.borrow().cv_plist[is as usize];

    psave = cv_p_get(cv_mem, which);

    Deltap = pbari * delta;
    rDeltap = ONE / Deltap;
    let ewt = cv_mem.borrow().cv_ewt.clone().unwrap();
    norms = N_VWrmsNorm(yS, &ewt) * pbari;
    rDeltay = SUNMAX(norms, rdelta) / pbari;
    Deltay = ONE / rDeltay;

    let (DQrhomax, DQtype) = {
        let m = cv_mem.borrow();
        (m.cv_DQrhomax, m.cv_DQtype)
    };

    if DQrhomax == ZERO {
        /* No switching */
        method = if DQtype == CV_CENTERED {
            CENTERED1
        } else {
            FORWARD1
        };
    } else {
        /* switch between simultaneous/separate DQ */
        ratio = Deltay * rDeltap;
        if SUNMAX(ONE / ratio, ratio) <= DQrhomax {
            method = if DQtype == CV_CENTERED {
                CENTERED1
            } else {
                FORWARD1
            };
        } else {
            method = if DQtype == CV_CENTERED {
                CENTERED2
            } else {
                FORWARD2
            };
        }
    }

    match method {
        CENTERED1 => {
            let Delta = SUNMIN(Deltay, Deltap);
            let r2Delta = HALF / Delta;

            N_VLinearSum(ONE, y, Delta, yS, ytemp);
            cv_p_set(cv_mem, which, psave + Delta);

            retval = cv_call_f(cv_mem, t, ytemp, ySdot);
            nfel += 1;
            if retval != 0 {
                return retval;
            }

            N_VLinearSum(ONE, y, -Delta, yS, ytemp);
            cv_p_set(cv_mem, which, psave - Delta);

            retval = cv_call_f(cv_mem, t, ytemp, ftemp);
            nfel += 1;
            if retval != 0 {
                return retval;
            }

            N_VLinearSum(r2Delta, ySdot, -r2Delta, ftemp, ySdot);
        }

        CENTERED2 => {
            let r2Deltap = HALF / Deltap;
            let r2Deltay = HALF / Deltay;

            N_VLinearSum(ONE, y, Deltay, yS, ytemp);

            retval = cv_call_f(cv_mem, t, ytemp, ySdot);
            nfel += 1;
            if retval != 0 {
                return retval;
            }

            N_VLinearSum(ONE, y, -Deltay, yS, ytemp);

            retval = cv_call_f(cv_mem, t, ytemp, ftemp);
            nfel += 1;
            if retval != 0 {
                return retval;
            }

            N_VLinearSum(r2Deltay, ySdot, -r2Deltay, ftemp, ySdot);

            cv_p_set(cv_mem, which, psave + Deltap);
            retval = cv_call_f(cv_mem, t, y, ytemp);
            nfel += 1;
            if retval != 0 {
                return retval;
            }

            cv_p_set(cv_mem, which, psave - Deltap);
            retval = cv_call_f(cv_mem, t, y, ftemp);
            nfel += 1;
            if retval != 0 {
                return retval;
            }

            /* ySdot = ySdot + r2Deltap * ytemp - r2Deltap * ftemp */
            cvals[0] = ONE;
            Xvecs.push(ySdot.clone());
            cvals[1] = r2Deltap;
            Xvecs.push(ytemp.clone());
            cvals[2] = -r2Deltap;
            Xvecs.push(ftemp.clone());

            retval = N_VLinearCombination(3, &cvals, &Xvecs, ySdot);
            if retval != CV_SUCCESS {
                return CV_VECTOROP_ERR;
            }
        }

        FORWARD1 => {
            let Delta = SUNMIN(Deltay, Deltap);
            let rDelta = ONE / Delta;

            N_VLinearSum(ONE, y, Delta, yS, ytemp);
            cv_p_set(cv_mem, which, psave + Delta);

            retval = cv_call_f(cv_mem, t, ytemp, ySdot);
            nfel += 1;
            if retval != 0 {
                return retval;
            }

            N_VLinearSum(rDelta, ySdot, -rDelta, ydot, ySdot);
        }

        FORWARD2 => {
            N_VLinearSum(ONE, y, Deltay, yS, ytemp);

            retval = cv_call_f(cv_mem, t, ytemp, ySdot);
            nfel += 1;
            if retval != 0 {
                return retval;
            }

            N_VLinearSum(rDeltay, ySdot, -rDeltay, ydot, ySdot);

            cv_p_set(cv_mem, which, psave + Deltap);
            retval = cv_call_f(cv_mem, t, y, ytemp);
            nfel += 1;
            if retval != 0 {
                return retval;
            }

            /* ySdot = ySdot + rDeltap * ytemp - rDeltap * ydot */
            cvals[0] = ONE;
            Xvecs.push(ySdot.clone());
            cvals[1] = rDeltap;
            Xvecs.push(ytemp.clone());
            cvals[2] = -rDeltap;
            Xvecs.push(ydot.clone());

            retval = N_VLinearCombination(3, &cvals, &Xvecs, ySdot);
            if retval != CV_SUCCESS {
                return CV_VECTOROP_ERR;
            }
        }

        _ => {}
    }

    cv_p_set(cv_mem, which, psave);

    /* Increment counter nfeS */
    cv_mem.borrow_mut().cv_nfeS += nfel as i64;

    0
}

/*
 * cvQuadSensRhsInternalDQ   - internal CVQuadSensRhsFn
 *
 * cvQuadSensRhsInternalDQ computes right hand side of all quadrature
 * sensitivity equations by finite differences. All work is actually
 * done in cvQuadSensRhs1InternalDQ.
 *
 * NOTE: the signature must stay exactly `CVQuadSensRhsFn` —
 * `cv_mem->cv_fQS` is assigned `Some(cvQuadSensRhsInternalDQ)` in
 * CVodeQuadSensInit.
 */

pub(crate) fn cvQuadSensRhsInternalDQ(
    Ns: i32,
    t: sunrealtype,
    y: &N_Vector,
    yS: &[N_Vector],
    yQdot: &N_Vector,
    yQSdot: &[N_Vector],
    cvode_mem: &mut Option<Box<dyn Any>>,
    tmp: &N_Vector,
    tmpQ: &N_Vector,
) -> i32 {
    let mut retval: i32;

    /* cvode_mem is passed here as user data.  C: `cv_mem = (CVodeMem)cvode_mem;`
    (cvodes.c:9978).  Reached with the user's own data instead of the mem when
    cvQuadSensNls calls fQS -- an upstream defect (UB in C); see the note in
    cvQuadSensNls. */
    let cv_mem = cvode_mem
        .as_mut()
        .and_then(|b| b.downcast_ref::<CVodeMem>())
        .cloned()
        .expect("cvQuadSensRhsInternalDQ cvode_mem holds CVodeMem");
    let cv_mem = &cv_mem;

    for is in 0..Ns as usize {
        retval = cvQuadSensRhs1InternalDQ(
            cv_mem,
            is as i32,
            t,
            y,
            &yS[is],
            yQdot,
            &yQSdot[is],
            tmp,
            tmpQ,
        );
        if retval != 0 {
            return retval;
        }
    }

    0
}

pub(crate) fn cvQuadSensRhs1InternalDQ(
    cv_mem: &CVodeMem,
    is: i32,
    t: sunrealtype,
    y: &N_Vector,
    yS: &N_Vector,
    yQdot: &N_Vector,
    yQSdot: &N_Vector,
    tmp: &N_Vector,
    tmpQ: &N_Vector,
) -> i32 {
    let mut retval: i32;
    let method: i32;
    let mut nfel: i32 = 0;
    let which: i32;
    let psave: sunrealtype;
    let pbari: sunrealtype;
    let delta: sunrealtype;
    let rdelta: sunrealtype;
    let Deltap: sunrealtype;
    let Deltay: sunrealtype;
    let rDeltay: sunrealtype;
    let norms: sunrealtype;

    {
        let m = cv_mem.borrow();
        delta = SUNRsqrt(SUNMAX(m.cv_reltol, m.cv_uround));
    }
    rdelta = ONE / delta;

    pbari = cv_mem.borrow().cv_pbar[is as usize];

    which = cv_mem.borrow().cv_plist[is as usize];

    psave = cv_p_get(cv_mem, which);

    Deltap = pbari * delta;
    let ewt = cv_mem.borrow().cv_ewt.clone().unwrap();
    norms = N_VWrmsNorm(yS, &ewt) * pbari;
    rDeltay = SUNMAX(norms, rdelta) / pbari;
    Deltay = ONE / rDeltay;

    let DQtype = cv_mem.borrow().cv_DQtype;
    method = if DQtype == CV_CENTERED {
        CENTERED1
    } else {
        FORWARD1
    };

    match method {
        CENTERED1 => {
            let Delta = SUNMIN(Deltay, Deltap);
            let r2Delta = HALF / Delta;

            N_VLinearSum(ONE, y, Delta, yS, tmp);
            cv_p_set(cv_mem, which, psave + Delta);

            retval = cv_call_fQ(cv_mem, t, tmp, yQSdot);
            nfel += 1;
            if retval != 0 {
                return retval;
            }

            N_VLinearSum(ONE, y, -Delta, yS, tmp);
            cv_p_set(cv_mem, which, psave - Delta);

            retval = cv_call_fQ(cv_mem, t, tmp, tmpQ);
            nfel += 1;
            if retval != 0 {
                return retval;
            }

            N_VLinearSum(r2Delta, yQSdot, -r2Delta, tmpQ, yQSdot);
        }

        FORWARD1 => {
            let Delta = SUNMIN(Deltay, Deltap);
            let rDelta = ONE / Delta;

            N_VLinearSum(ONE, y, Delta, yS, tmp);
            cv_p_set(cv_mem, which, psave + Delta);

            retval = cv_call_fQ(cv_mem, t, tmp, yQSdot);
            nfel += 1;
            if retval != 0 {
                return retval;
            }

            N_VLinearSum(rDelta, yQSdot, -rDelta, yQdot, yQSdot);
        }

        _ => {}
    }

    cv_p_set(cv_mem, which, psave);

    /* Increment counter nfQeS */
    cv_mem.borrow_mut().cv_nfQeS += nfel as i64;

    0
}

/*
 * =================================================================
 * Regression tests
 * =================================================================
 */

#[cfg(test)]
mod tests {
    use super::*;
    use sundials_core::sundials_libm::SunMath;

    use crate::cvodes_io::{CVodeSetSensParams, CVodeSetUserData};
    use crate::cvodes_ls::CVodeSetLinearSolver;
    use sundials_core::nvector_serial::N_VNew_Serial;
    use sundials_core::sundials_context::SUNContext_Create;
    use sundials_core::sunlinsol_dense::SUNLinSol_Dense;
    use sundials_core::sunmatrix_dense::SUNDenseMatrix;

    /* -----------------------------------------------------------------
     * Internal-DQ forward sensitivity relies on ALIASING: C stores the
     * caller's `p` POINTER in `cv_mem->cv_p` and `cvSensRhs1InternalDQ`
     * perturbs `cv_p[which]` in place around each call to the user's `f`,
     * which reads the very same array through its `user_data`. The port
     * shares the array as a `SensParams` handle; this test is the
     * executable proof that the perturbation reaches the callback.
     *
     * Problem:  y' = -p*y,  y(0) = 1,  p = 2
     *   exact:  y(t)     = exp(-p t)
     *           dy/dp(t) = -t exp(-p t)
     * With a private copy of `p` the callback would never see the
     * perturbation, so `df/dp` would be identically zero and the computed
     * sensitivity would stay exactly 0 for all time.
     * -----------------------------------------------------------------*/

    const P0: sunrealtype = 2.0;
    const TEND: sunrealtype = 1.0;

    struct SensTestData {
        /* the shared parameter array — the solver holds a clone of this
        very handle (C: the same `sunrealtype*`) */
        p: SensParams,
        /* extreme parameter values observed by the RHS callback */
        pmin: sunrealtype,
        pmax: sunrealtype,
    }

    fn sens_test_f(
        _t: sunrealtype,
        y: &N_Vector,
        ydot: &N_Vector,
        user_data: &mut Option<Box<dyn Any>>,
    ) -> i32 {
        let data = user_data
            .as_mut()
            .and_then(|b| b.downcast_mut::<SensTestData>())
            .expect("user_data is SensTestData");

        /* read the parameter exactly as a C callback would read data->p[0];
        the borrow ends with this statement */
        let p = data.p.borrow()[0];

        if p < data.pmin {
            data.pmin = p;
        }
        if p > data.pmax {
            data.pmax = p;
        }

        let ydata = N_VGetArrayPointer(y).expect("N_VGetArrayPointer");
        let mut dydata = N_VGetArrayPointer(ydot).expect("N_VGetArrayPointer");
        dydata[0] = -p * ydata[0];

        0
    }

    #[test]
    fn internal_dq_sensitivity_sees_perturbed_parameters() {
        let mut sunctx: Option<SUNContext> = None;
        assert_eq!(SUNContext_Create(SUN_COMM_NULL, &mut sunctx), 0);
        let sunctx = sunctx.expect("SUNContext_Create");

        /* the parameter array the user owns and the solver shares */
        let p: SensParams = Rc::new(RefCell::new(vec![P0]));

        let y = N_VNew_Serial(1, &sunctx).expect("N_VNew_Serial");
        N_VGetArrayPointer(&y).expect("N_VGetArrayPointer")[0] = ONE;

        let cvode_mem = CVodeCreate(CV_BDF, &sunctx).expect("CVodeCreate");

        assert_eq!(CVodeInit(&cvode_mem, sens_test_f, ZERO, &y), CV_SUCCESS);
        assert_eq!(CVodeSStolerances(&cvode_mem, 1.0e-8, 1.0e-10), CV_SUCCESS);

        /* the user data holds a CLONE of the same handle */
        let data = SensTestData {
            p: p.clone(),
            pmin: P0,
            pmax: P0,
        };
        assert_eq!(
            CVodeSetUserData(&cvode_mem, Some(Box::new(data))),
            CV_SUCCESS
        );

        let A = SUNDenseMatrix(1, 1, &sunctx).expect("SUNDenseMatrix");
        let LS = SUNLinSol_Dense(&y, &A, &sunctx).expect("SUNLinSol_Dense");
        assert_eq!(CVodeSetLinearSolver(&cvode_mem, &LS, Some(&A)), CV_SUCCESS);

        /* one sensitivity, computed by the INTERNAL DQ routine (fS1 = None) */
        let yS = N_VCloneVectorArray(1, &y).expect("N_VCloneVectorArray");
        N_VConst(ZERO, &yS[0]);
        assert_eq!(
            CVodeSensInit1(&cvode_mem, 1, CV_SIMULTANEOUS, None, &yS),
            CV_SUCCESS
        );
        assert_eq!(CVodeSensEEtolerances(&cvode_mem), CV_SUCCESS);

        /* C: CVodeSetSensParams(cvode_mem, data->p, pbar, NULL) */
        let pbar = [P0];
        assert_eq!(
            CVodeSetSensParams(&cvode_mem, Some(p.clone()), Some(&pbar[..]), None),
            CV_SUCCESS
        );

        let mut t = ZERO;
        let flag = CVode(&cvode_mem, TEND, &y, &mut t, CV_NORMAL);
        assert_eq!(flag, CV_SUCCESS, "CVode failed with flag {flag}");
        assert_eq!(t, TEND);

        /* state: y(TEND) = exp(-p*TEND) */
        let yend = N_VGetArrayPointer(&y).expect("N_VGetArrayPointer")[0];
        let y_exact = (-P0 * TEND).sun_exp();
        assert!(
            SUNRabs(yend - y_exact) <= 1.0e-6 * y_exact,
            "state wrong: got {yend}, expected {y_exact}"
        );

        /* sensitivity: dy/dp(TEND) = -TEND*exp(-p*TEND) */
        let mut tS = ZERO;
        assert_eq!(CVodeGetSens(&cvode_mem, &mut tS, &yS), CV_SUCCESS);
        let s = N_VGetArrayPointer(&yS[0]).expect("N_VGetArrayPointer")[0];
        let s_exact = -TEND * (-P0 * TEND).sun_exp();

        /* the defect signature: with an unshared copy of `p` this is 0 */
        assert!(s != ZERO, "sensitivity is identically zero — the DQ perturbation never reached the RHS callback");
        assert!(
            SUNRabs(s - s_exact) <= 1.0e-4 * SUNRabs(s_exact),
            "sensitivity wrong: got {s}, expected {s_exact}"
        );

        /* direct proof of the aliasing: the callback saw p perturbed both
        ways (CV_CENTERED is the default DQtype) */
        let mut user_data: Option<Box<dyn Any>> = None;
        std::mem::swap(&mut cvode_mem.borrow_mut().cv_user_data, &mut user_data);
        let data = user_data
            .as_ref()
            .and_then(|b| b.downcast_ref::<SensTestData>())
            .expect("user_data is SensTestData");
        assert!(
            data.pmax > P0 && data.pmin < P0,
            "RHS never observed a perturbed parameter (saw [{}, {}], p = {P0})",
            data.pmin,
            data.pmax
        );

        /* and the array the caller still owns was restored exactly */
        assert_eq!(p.borrow()[0], P0);
    }
}
