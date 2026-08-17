//! Port of `src/cvodes/cvodes_proj.c` (+ `include/cvodes/cvodes_proj.h` folded).
//!
//! `CVProjFn`, `CVodeProjMemRec`/`CVodeProjMem`, and the default projection
//! constants (`PROJ_MAX_FAILS`, `PROJ_EPS`, `PROJ_FAIL_ETA`) live in
//! `cvodes_impl` (the header content of `cvodes_proj.h` / `cvodes_proj_impl.h`
//! is folded there because `CVodeMemRec` embeds the projection memory).
//!
//! C `cvProjFree` has no Rust counterpart: dropping the
//! `Option<CVodeProjMem>` box frees the projection memory.
//!
//! Upstream `cvodes_proj.c` is character-for-character identical to
//! `cvode_proj.c` apart from the `cvodes_impl.h` include, so this module is
//! the cvodes-scoped twin of `cvode_rs::cvode_proj`.

use crate::cvodes_impl::*;
use sundials_core::sundials_math::{SUNMAX, SUNRabs};
use sundials_core::sundials_nvector::{N_VScale, N_VWrmsNorm, N_Vector};
use sundials_core::sundials_types::*;

/* Private constants (file-scope `#define`s in cvodes_proj.c; they shadow the
crate-wide `cvodes_impl` constants of the same name and identical value) */
const ZERO: sunrealtype = 0.0; /* real 0.0 */
const ONE: sunrealtype = 1.0; /* real 1.0 */

const ONEPSM: sunrealtype = 1.000001;

/* ===========================================================================
 * Exported Functions - projection initialization
 * ===========================================================================*/

/* -----------------------------------------------------------------------------
 * CVodeSetProjFn sets a user defined projection function
 * ---------------------------------------------------------------------------*/
pub fn CVodeSetProjFn(cvode_mem: &CVodeMem, pfun: Option<CVProjFn>) -> i32 {
    /* NULL-mem check: handled by type system */

    /* Check if the projection function is NULL */
    if pfun.is_none() {
        cvProcessError(
            Some(cvode_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeSetProjFn",
            file!(),
            "The projection function is NULL.",
        );
        return CV_ILL_INPUT;
    }

    /* Check for compatible method */
    let lmm = cvode_mem.borrow().cv_lmm;
    if lmm != CV_BDF {
        cvProcessError(
            Some(cvode_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeSetProjFn",
            file!(),
            "Projection is only supported with BDF methods.",
        );
        return CV_ILL_INPUT;
    }

    /* Create the projection memory (if necessary) */
    let retval = cvProjCreate(&mut cvode_mem.borrow_mut().proj_mem);
    if retval != CV_SUCCESS {
        cvProcessError(
            Some(cvode_mem),
            CV_MEM_FAIL,
            line!() as i32,
            "CVodeSetProjFn",
            file!(),
            MSG_CV_MEM_FAIL,
        );
        return CV_MEM_FAIL;
    }

    /* Shortcut to projection memory */
    let mut mem = cvode_mem.borrow_mut();
    let proj_mem = mem.proj_mem.as_mut().unwrap();

    /* User-defined projection */
    proj_mem.internal_proj = SUNFALSE;

    /* Set the projection function */
    proj_mem.pfun = pfun;

    /* Enable projection */
    mem.proj_enabled = SUNTRUE;

    CV_SUCCESS
}

/* ===========================================================================
 * Exported Functions - projection set function
 * ===========================================================================*/

pub fn CVodeSetProjErrEst(cvode_mem: &CVodeMem, onoff: sunbooleantype) -> i32 {
    /* Access memory structures */
    let retval = cvAccessProjMem(cvode_mem, "CVodeSetProjErrEst");
    if retval != CV_SUCCESS {
        return retval;
    }

    /* Set projection error flag */
    cvode_mem.borrow_mut().proj_mem.as_mut().unwrap().err_proj = onoff;

    CV_SUCCESS
}

pub fn CVodeSetProjFrequency(cvode_mem: &CVodeMem, freq: i64) -> i32 {
    /* Access memory structures */
    let retval = cvAccessProjMem(cvode_mem, "CVodeSetProjFrequency");
    if retval != CV_SUCCESS {
        return retval;
    }

    /* Set projection frequency */
    let mut mem = cvode_mem.borrow_mut();
    if freq < 0 {
        /* Restore default */
        mem.proj_mem.as_mut().unwrap().freq = 1;
        mem.proj_enabled = SUNTRUE;
    } else if freq == 0 {
        /* Disable projection */
        mem.proj_mem.as_mut().unwrap().freq = 0;
        mem.proj_enabled = SUNFALSE;
    } else {
        /* Enable projection at given frequency */
        mem.proj_mem.as_mut().unwrap().freq = freq;
        mem.proj_enabled = SUNTRUE;
    }

    CV_SUCCESS
}

pub fn CVodeSetMaxNumProjFails(cvode_mem: &CVodeMem, max_fails: i32) -> i32 {
    /* Access memory structures */
    let retval = cvAccessProjMem(cvode_mem, "CVodeSetMaxNumProjFails");
    if retval != CV_SUCCESS {
        return retval;
    }

    /* Set maximum number of projection failures in a step attempt */
    let mut mem = cvode_mem.borrow_mut();
    if max_fails < 1 {
        /* Restore default */
        mem.proj_mem.as_mut().unwrap().max_fails = PROJ_MAX_FAILS;
    } else {
        /* Update max number of fails */
        mem.proj_mem.as_mut().unwrap().max_fails = max_fails;
    }

    CV_SUCCESS
}

pub fn CVodeSetEpsProj(cvode_mem: &CVodeMem, eps: sunrealtype) -> i32 {
    /* Access memory structures */
    let retval = cvAccessProjMem(cvode_mem, "CVodeSetEpsProj");
    if retval != CV_SUCCESS {
        return retval;
    }

    /* Set the projection tolerance */
    let mut mem = cvode_mem.borrow_mut();
    if eps <= ZERO {
        /* Restore default */
        mem.proj_mem.as_mut().unwrap().eps_proj = PROJ_EPS;
    } else {
        /* Update projection tolerance */
        mem.proj_mem.as_mut().unwrap().eps_proj = eps;
    }

    CV_SUCCESS
}

pub fn CVodeSetProjFailEta(cvode_mem: &CVodeMem, eta: sunrealtype) -> i32 {
    /* Access memory structures */
    let retval = cvAccessProjMem(cvode_mem, "CVodeSetProjFailEta");
    if retval != CV_SUCCESS {
        return retval;
    }

    /* Set the step size reduction factor for a projection failure */
    let mut mem = cvode_mem.borrow_mut();
    if (eta <= ZERO) || (eta > ONE) {
        /* Restore default */
        mem.proj_mem.as_mut().unwrap().eta_pfail = PROJ_FAIL_ETA;
    } else {
        /* Update the eta value */
        mem.proj_mem.as_mut().unwrap().eta_pfail = eta;
    }

    CV_SUCCESS
}

/* ===========================================================================
 * Exported Functions - projection get functions
 * ===========================================================================*/

pub fn CVodeGetNumProjEvals(cvode_mem: &CVodeMem, nproj: &mut i64) -> i32 {
    /* Access memory structures */
    let retval = cvAccessProjMem(cvode_mem, "CVodeGetNumProjEvals");
    if retval != CV_SUCCESS {
        return retval;
    }

    /* Get number of projection evaluations */
    *nproj = cvode_mem.borrow().proj_mem.as_ref().unwrap().nproj;

    CV_SUCCESS
}

pub fn CVodeGetNumProjFails(cvode_mem: &CVodeMem, npfails: &mut i64) -> i32 {
    /* Access memory structures */
    let retval = cvAccessProjMem(cvode_mem, "CVodeGetNumProjFails");
    if retval != CV_SUCCESS {
        return retval;
    }

    /* Get number of projection fails */
    *npfails = cvode_mem.borrow().proj_mem.as_ref().unwrap().npfails;

    CV_SUCCESS
}

/* ===========================================================================
 * Internal Functions
 * ===========================================================================*/

/*
 * cvProjection
 *
 * For user supplied projection function, use ftemp as temporary storage
 * for the current error estimate (acor) and use tempv to store the
 * accumulated correction due to projection, acorP (tempv is not touched
 * until it is potentially used in cvCompleteStep).
 */

pub fn cvDoProjection(
    cv_mem: &CVodeMem,
    nflagPtr: &mut i32,
    saved_t: sunrealtype,
    npfailPtr: &mut i32,
) -> i32 {
    /* Access projection memory */
    let have_proj_mem = cv_mem.borrow().proj_mem.is_some();
    if !have_proj_mem {
        cvProcessError(
            Some(cv_mem),
            CV_PROJ_MEM_NULL,
            line!() as i32,
            "cvDoProjection",
            file!(),
            MSG_CV_PROJ_MEM_NULL,
        );
        return CV_PROJ_MEM_NULL;
    }

    /* (C initializes retval = CV_SUCCESS here; the value is overwritten by
    the pfun call below, so the dead store is omitted.) */

    /* Copy needed scalars and clone vector handles out of the mem (granular
    borrow discipline: no borrow may be held across the user callback) */
    let (err_proj, eps_proj, pfun, tn, y, tempv, ftemp, acor) = {
        let mem = cv_mem.borrow();
        let proj_mem = mem.proj_mem.as_ref().unwrap();
        (
            proj_mem.err_proj,
            proj_mem.eps_proj,
            proj_mem.pfun,
            mem.cv_tn,
            mem.cv_y.as_ref().unwrap().clone(),
            mem.cv_tempv.as_ref().unwrap().clone(),
            mem.cv_ftemp.as_ref().unwrap().clone(),
            mem.cv_acor.as_ref().unwrap().clone(),
        )
    };

    /* Use tempv to store acorP and, if projecting the error, ftemp to store
    errP (recall that in this case we did not allocate vectors to for
    acorP and errP). */
    let acorP = tempv;
    let errP: Option<N_Vector> = if err_proj { Some(ftemp) } else { None };

    /* Copy acor into errP (if projecting the error) */
    if err_proj {
        N_VScale(ONE, &acor, errP.as_ref().unwrap());
    }

    /* Call the user projection function */
    let pfun = pfun.expect("proj_mem->pfun is NULL"); /* C UB (NULL call) -> panic */
    let mut user_data = cv_mem.borrow_mut().cv_user_data.take();
    let mut retval = pfun(tn, &y, &acorP, eps_proj, errP.as_ref(), &mut user_data);
    cv_mem.borrow_mut().cv_user_data = user_data;

    {
        let mut mem = cv_mem.borrow_mut();
        let proj_mem = mem.proj_mem.as_mut().unwrap();
        proj_mem.nproj += 1;

        /* This is not the first projection anymore */
        proj_mem.first_proj = SUNFALSE;
    }

    /* Check the return value */
    if retval == CV_SUCCESS {
        /* Recompute acnrm to be used in error test (if projecting the error) */
        if err_proj {
            let ewt = cv_mem.borrow().cv_ewt.as_ref().unwrap().clone();
            let acnrm = N_VWrmsNorm(errP.as_ref().unwrap(), &ewt);
            cv_mem.borrow_mut().cv_acnrm = acnrm;
        }

        /* The projection was successful, return now */
        cv_mem.borrow_mut().proj_applied = SUNTRUE;
        return CV_SUCCESS;
    }

    /* The projection failed, update the return value */
    if retval < 0 {
        retval = CV_PROJFUNC_FAIL;
    }
    if retval > 0 {
        retval = PROJFUNC_RECVR;
    }

    /* Increment cumulative failure count and restore zn */
    cv_mem.borrow_mut().proj_mem.as_mut().unwrap().npfails += 1;
    crate::cvodes::cvRestore(cv_mem, saved_t);

    /* Return if failed unrecoverably */
    if retval == CV_PROJFUNC_FAIL {
        return CV_PROJFUNC_FAIL;
    }

    /* Recoverable failure, increment failure count for this step attempt */
    *npfailPtr += 1;
    cv_mem.borrow_mut().cv_etamax = ONE;

    /* Check for maximum number of failures or |h| = hmin */
    {
        let mem = cv_mem.borrow();
        let proj_mem = mem.proj_mem.as_ref().unwrap();
        if (SUNRabs(mem.cv_h) <= mem.cv_hmin * ONEPSM) || (*npfailPtr == proj_mem.max_fails) {
            if retval == PROJFUNC_RECVR {
                return CV_REPTD_PROJFUNC_ERR;
            }
        }
    }

    /* Reduce step size; return to reattempt the step */
    {
        let mut mem = cv_mem.borrow_mut();
        let eta_pfail = mem.proj_mem.as_ref().unwrap().eta_pfail;
        let eta = SUNMAX(eta_pfail, mem.cv_hmin / SUNRabs(mem.cv_h));
        mem.cv_eta = eta;
    }
    *nflagPtr = PREV_PROJ_FAIL;
    crate::cvodes::cvRescale(cv_mem);

    PREDICT_AGAIN
}

pub fn cvProjInit(proj_mem: &mut CVodeProjMemRec) -> i32 {
    /* NULL proj_mem check (C returned CV_PROJ_MEM_NULL): handled by type
    system — the caller unwraps the Option<CVodeProjMem> */

    /* reset flags and counters */
    proj_mem.first_proj = SUNTRUE;
    proj_mem.nstlprj = 0;
    proj_mem.nproj = 0;
    proj_mem.npfails = 0;

    CV_SUCCESS
}

/* C cvProjFree: not ported — dropping the Option<CVodeProjMem> frees the
projection memory. */

/* ===========================================================================
 * Utility Functions
 * ===========================================================================*/

fn cvProjCreate(proj_mem: &mut Option<CVodeProjMem>) -> i32 {
    /* Allocate projection memory if necessary, otherwise return success */
    if proj_mem.is_none() {
        /* Zero out proj_mem (C malloc + memset) */
        let mut new_mem: CVodeProjMem = Box::new(CVodeProjMemRec {
            internal_proj: SUNFALSE,
            err_proj: SUNFALSE,
            first_proj: SUNFALSE,
            freq: 0,
            nstlprj: 0,
            max_fails: 0,
            pfun: None,
            eps_proj: 0.0,
            eta_pfail: 0.0,
            nproj: 0,
            npfails: 0,
        });

        /* Initialize projection variables */
        let retval = cvProjSetDefaults(&mut new_mem);
        if retval != CV_SUCCESS {
            return retval;
        }

        *proj_mem = Some(new_mem);
    }

    CV_SUCCESS
}

fn cvProjSetDefaults(proj_mem: &mut CVodeProjMemRec) -> i32 {
    /* NULL proj_mem check (C returned CV_MEM_FAIL): handled by type system */

    proj_mem.internal_proj = SUNTRUE;
    proj_mem.err_proj = SUNTRUE;
    proj_mem.first_proj = SUNTRUE;

    proj_mem.freq = 1;
    proj_mem.nstlprj = 0;

    proj_mem.max_fails = PROJ_MAX_FAILS;

    proj_mem.pfun = None;

    proj_mem.eps_proj = PROJ_EPS;
    proj_mem.eta_pfail = PROJ_FAIL_ETA;

    proj_mem.nproj = 0;
    proj_mem.npfails = 0;

    CV_SUCCESS
}

fn cvAccessProjMem(cvode_mem: &CVodeMem, fname: &str) -> i32 {
    /* Access cvode memory: NULL-mem check handled by type system */

    /* Access projection memory */
    let have_proj_mem = cvode_mem.borrow().proj_mem.is_some();
    if !have_proj_mem {
        cvProcessError(
            Some(cvode_mem),
            CV_PROJ_MEM_NULL,
            line!() as i32,
            fname,
            file!(),
            MSG_CV_PROJ_MEM_NULL,
        );
        return CV_PROJ_MEM_NULL;
    }

    CV_SUCCESS
}
