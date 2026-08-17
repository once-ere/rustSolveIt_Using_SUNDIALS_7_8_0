//! Port of `src/cvodes/cvodes_nls.c`: the CVODES nonlinear solver
//! interface (state-only corrector — the sensitivity variants live in
//! `cvodes_nls_sim.rs`, `cvodes_nls_stg.rs`, `cvodes_nls_stg1.rs`).
//!
//! The C `void* cvode_mem` handed to the SUNNonlinearSolver as the
//! integrator mem / ctest / norm / getter data maps to a boxed
//! `CVodeMem` clone inside an `Option<Box<dyn Any>>` token; each
//! callback downcasts the token back to the handle and uses granular
//! borrows (never holding a borrow across a user callback, an
//! `N_Vector` op on a user-visible vector, or a linear-solver call).
//!
//! `ONE`, `NLS_MAXCOR`, `CRDOWN` and `RDIV` come from `cvodes_impl`
//! (`cvodes_impl.h` defines the latter three; `ONE` is the shared
//! `cvodes.c` fragment constant and has the same value `1.0` as this
//! file's `#define ONE SUN_RCONST(1.0)`).

use std::any::Any;

use crate::cvodes_impl::*;
use sundials_core::sundials_errors::{SUN_ERR_ARG_CORRUPT, SUN_SUCCESS};
use sundials_core::sundials_math::{SUNMAX, SUNMIN};
use sundials_core::sundials_nonlinearsolver::*;
use sundials_core::sundials_nvector::{N_VLinearSum, N_VScale, N_VWrmsNorm, N_Vector};
use sundials_core::sundials_types::*;

/* -----------------------------------------------------------------------------
 * Exported functions
 * ---------------------------------------------------------------------------*/

pub fn CVodeSetNonlinearSolver(cvode_mem: &CVodeMem, NLS: &SUNNonlinearSolver) -> i32 {
    /* Return immediately if CVode memory is NULL: handled by the type system */
    let cv_mem = cvode_mem;
    let mut retval: i32;

    /* Return immediately if NLS memory is NULL ("NLS must be non-NULL"):
    handled by the type system */

    /* check for required nonlinear solver functions */
    {
        let ops = NLS.ops.borrow();
        if ops.gettype.is_none()
            || ops.solve.is_none()
            || (ops.setsysfn.is_none() && ops.setsysfns.is_none())
        {
            drop(ops);
            cvProcessError(
                Some(cv_mem),
                CV_ILL_INPUT,
                line!() as i32,
                "CVodeSetNonlinearSolver",
                file!(),
                "NLS does not support required operations",
            );
            return CV_ILL_INPUT;
        }
    }

    /* free any existing nonlinear solver */
    let (old_nls, own_nls) = {
        let mut m = cv_mem.borrow_mut();
        (m.NLS.take(), m.ownNLS)
    };
    if old_nls.is_some() && own_nls {
        /* C stores the result in retval but never reads it */
        let _ = SUNNonlinSolFree(old_nls);
    }

    {
        let mut m = cv_mem.borrow_mut();
        m.NLS = Some(NLS.clone());

        /* Set NLS ownership flag. If this function was called to attach the default
        NLS, CVODE will set the flag to SUNTRUE after this function returns. */
        m.ownNLS = SUNFALSE;
    }

    /* set the nonlinear system function */
    let nls = { cv_mem.borrow().NLS.clone() }.expect("NLS");
    if SUNNonlinSolGetType(&nls) == SUNNONLINEARSOLVER_ROOTFIND {
        retval = SUNNonlinSolSetSysFn(&nls, Some(cvNlsResidual));
    } else if SUNNonlinSolGetType(&nls) == SUNNONLINEARSOLVER_FIXEDPOINT {
        retval = SUNNonlinSolSetSysFn(&nls, Some(cvNlsFPFunction));
    } else if SUNNonlinSolGetType(&nls) == SUNNONLINEARSOLVER_HYBRID {
        retval = SUNNonlinSolSetSysFns(&nls, Some(cvNlsResidual), Some(cvNlsFPFunction));
    } else {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeSetNonlinearSolver",
            file!(),
            "Invalid nonlinear solver type",
        );
        return CV_ILL_INPUT;
    }

    if retval != CV_SUCCESS {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeSetNonlinearSolver",
            file!(),
            "Setting nonlinear system function failed",
        );
        return CV_ILL_INPUT;
    }

    retval = SUNNonlinSolSetConvTestFn(&nls, Some(cvNlsConvTest), Some(Box::new(cv_mem.clone())));
    if retval != CV_SUCCESS {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeSetNonlinearSolver",
            file!(),
            "Setting convergence test function failed",
        );
        return CV_ILL_INPUT;
    }

    retval = SUNNonlinSolSetNormFn(&nls, Some(cvNlsNorm), Some(Box::new(cv_mem.clone())));
    if retval != CV_SUCCESS {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeSetNonlinearSolver",
            file!(),
            "Setting convergence-test norm function failed",
        );
        return CV_ILL_INPUT;
    }

    retval = SUNNonlinSolSetGetUpdateNormFn(
        &nls,
        Some(cvNlsGetUpdateNorm),
        Some(Box::new(cv_mem.clone())),
    );
    if retval != CV_SUCCESS {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeSetNonlinearSolver",
            file!(),
            "Setting update-norm getter failed",
        );
        return CV_ILL_INPUT;
    }

    retval =
        SUNNonlinSolSetGetConvRateFn(&nls, Some(cvNlsGetConvRate), Some(Box::new(cv_mem.clone())));
    if retval != CV_SUCCESS {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeSetNonlinearSolver",
            file!(),
            "Setting convergence-rate getter failed",
        );
        return CV_ILL_INPUT;
    }

    retval = SUNNonlinSolSetMaxIters(&nls, NLS_MAXCOR);
    if retval != CV_SUCCESS {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeSetNonlinearSolver",
            file!(),
            "Setting maximum number of nonlinear iterations failed",
        );
        return CV_ILL_INPUT;
    }

    cv_mem.borrow_mut().cv_acnrmcur = SUNFALSE;

    /* Set the nonlinear system RHS function */
    if cv_mem.borrow().cv_f.is_none() {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeSetNonlinearSolver",
            file!(),
            "The ODE RHS function is NULL",
        );
        return CV_ILL_INPUT;
    }
    {
        let mut m = cv_mem.borrow_mut();
        m.nls_f = m.cv_f;
    }

    CV_SUCCESS
}

/*---------------------------------------------------------------
  CVodeSetNlsRhsFn:

  This routine sets an alternative user-supplied ODE right-hand
  side function to use in the evaluation of nonlinear system
  functions.
  ---------------------------------------------------------------*/
pub fn CVodeSetNlsRhsFn(cvode_mem: &CVodeMem, f: Option<CVRhsFn>) -> i32 {
    /* NULL-mem check: handled by the type system */
    let mut cv_mem = cvode_mem.borrow_mut();

    if let Some(f) = f {
        cv_mem.nls_f = Some(f);
    } else {
        cv_mem.nls_f = cv_mem.cv_f;
    }

    CV_SUCCESS
}

/*---------------------------------------------------------------
  CVodeGetNonlinearSystemData:

  This routine provides access to the relevant data needed to
  compute the nonlinear system function.

  C hands out the raw `user_data` pointer; boxes cannot alias, so
  the token is SWAPPED with the caller's out-param (same convention
  as `CVodeGetUserData` in `cvodes_io.rs`) — a caller that uses it
  must hand it back (via CVodeSetUserData or a second swap) before
  the integrator next invokes a user callback.
  ---------------------------------------------------------------*/
#[allow(clippy::too_many_arguments)]
pub fn CVodeGetNonlinearSystemData(
    cvode_mem: &CVodeMem,
    tcur: &mut sunrealtype,
    ypred: &mut Option<N_Vector>,
    yn: &mut Option<N_Vector>,
    fn_: &mut Option<N_Vector>,
    gamma: &mut sunrealtype,
    rl1: &mut sunrealtype,
    zn1: &mut Option<N_Vector>,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    /* NULL-mem check: handled by the type system */
    let mut cv_mem = cvode_mem.borrow_mut();

    *tcur = cv_mem.cv_tn;
    *ypred = cv_mem.cv_zn[0].clone();
    *yn = cv_mem.cv_y.clone();
    *fn_ = cv_mem.cv_ftemp.clone();
    *gamma = cv_mem.cv_gamma;
    *rl1 = cv_mem.cv_rl1;
    *zn1 = cv_mem.cv_zn[1].clone();
    /* C copies the raw pointer; the box is swapped out instead — the
    caller must hand it back before the next user-callback invocation. */
    std::mem::swap(&mut cv_mem.cv_user_data, user_data);

    CV_SUCCESS
}

/* -----------------------------------------------------------------------------
 * Private functions
 * ---------------------------------------------------------------------------*/

pub fn cvNlsInit(cvode_mem: &CVodeMem) -> i32 {
    let mut retval: i32;

    let nls = { cvode_mem.borrow().NLS.clone() }.expect("NLS");

    /* set the linear solver setup wrapper function */
    let has_lsetup = cvode_mem.borrow().cv_lsetup.is_some();
    if has_lsetup {
        retval = SUNNonlinSolSetLSetupFn(&nls, Some(cvNlsLSetup));
    } else {
        retval = SUNNonlinSolSetLSetupFn(&nls, None);
    }

    if retval != CV_SUCCESS {
        cvProcessError(
            Some(cvode_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "cvNlsInit",
            file!(),
            "Setting the linear solver setup function failed",
        );
        return CV_NLS_INIT_FAIL;
    }

    /* set the linear solver solve wrapper function */
    let has_lsolve = cvode_mem.borrow().cv_lsolve.is_some();
    if has_lsolve {
        retval = SUNNonlinSolSetLSolveFn(&nls, Some(cvNlsLSolve));
    } else {
        retval = SUNNonlinSolSetLSolveFn(&nls, None);
    }

    if retval != CV_SUCCESS {
        cvProcessError(
            Some(cvode_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "cvNlsInit",
            file!(),
            "Setting linear solver solve function failed",
        );
        return CV_NLS_INIT_FAIL;
    }

    /* initialize nonlinear solver */
    retval = SUNNonlinSolInitialize(&nls);

    if retval != CV_SUCCESS {
        cvProcessError(
            Some(cvode_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "cvNlsInit",
            file!(),
            MSGCV_NLS_INIT_FAIL,
        );
        return CV_NLS_INIT_FAIL;
    }

    CV_SUCCESS
}

fn cvNlsLSetup(
    jbad: sunbooleantype,
    jcur: &mut sunbooleantype,
    cvode_mem: &mut Option<Box<dyn Any>>,
) -> i32 {
    let cv_mem = match cvode_mem
        .as_mut()
        .and_then(|b| b.downcast_ref::<CVodeMem>())
        .cloned()
    {
        Some(cv_mem) => cv_mem,
        None => {
            cvProcessError(
                None,
                CV_MEM_NULL,
                line!() as i32,
                "cvNlsLSetup",
                file!(),
                MSGCV_NO_MEM,
            );
            return CV_MEM_NULL;
        }
    };

    /* if the nonlinear solver marked the Jacobian as bad update convfail */
    if jbad {
        cv_mem.borrow_mut().convfail = CV_FAIL_BAD_J;
    }

    /* setup the linear solver (C passes &cv_mem->cv_jcur directly; the port
    copies the flag out, hands the local to lsetup, and writes it back) */
    let (lsetup, convfail, y, ftemp, vtemp1, vtemp2, vtemp3);
    let mut jcur_local;
    {
        let m = cv_mem.borrow();
        lsetup = m.cv_lsetup.expect("cv_lsetup");
        convfail = m.convfail;
        y = m.cv_y.as_ref().expect("cv_y").clone();
        ftemp = m.cv_ftemp.as_ref().expect("cv_ftemp").clone();
        jcur_local = m.cv_jcur;
        vtemp1 = m.cv_vtemp1.as_ref().expect("cv_vtemp1").clone();
        vtemp2 = m.cv_vtemp2.as_ref().expect("cv_vtemp2").clone();
        vtemp3 = m.cv_vtemp3.as_ref().expect("cv_vtemp3").clone();
    }
    let retval = lsetup(
        &cv_mem,
        convfail,
        &y,
        &ftemp,
        &mut jcur_local,
        &vtemp1,
        &vtemp2,
        &vtemp3,
    );
    {
        let mut m = cv_mem.borrow_mut();
        m.cv_jcur = jcur_local;
        m.cv_nsetups += 1;

        /* update Jacobian status */
        *jcur = m.cv_jcur;

        m.cv_forceSetup = SUNFALSE;
        m.cv_gamrat = ONE;
        m.cv_gammap = m.cv_gamma;
        m.cv_crate = ONE;
        m.cv_crateS = ONE;
        m.cv_delnrm = 0.0; /* SUN_RCONST(0.0) */
        m.cv_delnrmS = 0.0; /* SUN_RCONST(0.0) */
        m.cv_nstlp = m.cv_nst;
    }

    if retval < 0 {
        return CV_LSETUP_FAIL;
    }
    if retval > 0 {
        return SUN_NLS_CONV_RECVR;
    }

    CV_SUCCESS
}

fn cvNlsLSolve(delta: &N_Vector, cvode_mem: &mut Option<Box<dyn Any>>) -> i32 {
    let cv_mem = match cvode_mem
        .as_mut()
        .and_then(|b| b.downcast_ref::<CVodeMem>())
        .cloned()
    {
        Some(cv_mem) => cv_mem,
        None => {
            cvProcessError(
                None,
                CV_MEM_NULL,
                line!() as i32,
                "cvNlsLSolve",
                file!(),
                MSGCV_NO_MEM,
            );
            return CV_MEM_NULL;
        }
    };

    let (lsolve, ewt, y, ftemp);
    {
        let m = cv_mem.borrow();
        lsolve = m.cv_lsolve.expect("cv_lsolve");
        ewt = m.cv_ewt.as_ref().expect("cv_ewt").clone();
        y = m.cv_y.as_ref().expect("cv_y").clone();
        ftemp = m.cv_ftemp.as_ref().expect("cv_ftemp").clone();
    }
    let retval = lsolve(&cv_mem, delta, &ewt, &y, &ftemp);

    if retval < 0 {
        return CV_LSOLVE_FAIL;
    }
    if retval > 0 {
        return SUN_NLS_CONV_RECVR;
    }

    CV_SUCCESS
}

fn cvNlsConvTest(
    NLS: &SUNNonlinearSolver,
    ycor: &N_Vector,
    delta: &N_Vector,
    tol: sunrealtype,
    ewt: &N_Vector,
    cvode_mem: &mut Option<Box<dyn Any>>,
) -> i32 {
    let cv_mem = match cvode_mem
        .as_mut()
        .and_then(|b| b.downcast_ref::<CVodeMem>())
        .cloned()
    {
        Some(cv_mem) => cv_mem,
        None => {
            cvProcessError(
                None,
                CV_MEM_NULL,
                line!() as i32,
                "cvNlsConvTest",
                file!(),
                MSGCV_NO_MEM,
            );
            return CV_MEM_NULL;
        }
    };

    /* compute the norm of the correction (C writes through
    &cv_mem->cv_delnrm; ported as copy-out / call / copy-back) */
    let mut delnrm = cv_mem.borrow().cv_delnrm;
    let nrm_retval = cvNlsNorm(delta, ewt, &mut delnrm, cvode_mem);
    cv_mem.borrow_mut().cv_delnrm = delnrm;
    if nrm_retval != SUN_SUCCESS {
        /* unreachable: cvNlsNorm always succeeds (as in C); the C call site
        passes no time argument for the MSG_TIME format — use cv_tn */
        let tn = cv_mem.borrow().cv_tn;
        cvProcessError(
            Some(&cv_mem),
            CV_NLS_FAIL,
            line!() as i32,
            "cvNlsConvTest",
            file!(),
            &MSGCV_NLS_FAIL(tn),
        );
        return CV_NLS_FAIL;
    }

    /* get the current nonlinear solver iteration count */
    let mut m: i32 = 0;
    let retval = SUNNonlinSolGetCurIter(NLS, &mut m);
    if retval != CV_SUCCESS {
        return CV_MEM_NULL;
    }

    /* Test for convergence. If m > 0, an estimate of the convergence
       rate constant is stored in crate, and used in the test.        */
    let dcon: sunrealtype;
    {
        let mut mm = cv_mem.borrow_mut();
        if m > 0 {
            mm.cv_crate = SUNMAX(CRDOWN * mm.cv_crate, mm.cv_delnrm / mm.cv_delp);
        }
        dcon = mm.cv_delnrm * SUNMIN(ONE, mm.cv_crate) / tol;
    }

    if dcon <= ONE {
        if m == 0 {
            let mut mm = cv_mem.borrow_mut();
            mm.cv_acnrm = mm.cv_delnrm;
            mm.cv_acnrmcur = SUNTRUE;
        } else {
            let acnrm = N_VWrmsNorm(ycor, ewt);
            let mut mm = cv_mem.borrow_mut();
            mm.cv_acnrm = acnrm;
            mm.cv_acnrmcur = SUNTRUE;
        }
        return CV_SUCCESS; /* Nonlinear system was solved successfully */
    }

    /* check if the iteration seems to be diverging */
    {
        let mm = cv_mem.borrow();
        if (m >= 1) && (mm.cv_delnrm > RDIV * mm.cv_delp) {
            return SUN_NLS_CONV_RECVR;
        }
    }

    /* Save norm of correction and loop again */
    {
        let mut mm = cv_mem.borrow_mut();
        mm.cv_delp = mm.cv_delnrm;
    }

    /* Not yet converged */
    SUN_NLS_CONTINUE
}

fn cvNlsNorm(
    delta: &N_Vector,
    ewt: &N_Vector,
    delnrm: &mut sunrealtype,
    _cvode_mem: &mut Option<Box<dyn Any>>,
) -> SUNErrCode {
    *delnrm = N_VWrmsNorm(delta, ewt);
    SUN_SUCCESS
}

fn cvNlsGetUpdateNorm(
    delnrm: &mut sunrealtype,
    cvode_mem: &mut Option<Box<dyn Any>>,
) -> SUNErrCode {
    let cv_mem = match cvode_mem
        .as_mut()
        .and_then(|b| b.downcast_ref::<CVodeMem>())
        .cloned()
    {
        Some(cv_mem) => cv_mem,
        None => return SUN_ERR_ARG_CORRUPT,
    };

    *delnrm = cv_mem.borrow().cv_delnrm;
    SUN_SUCCESS
}

fn cvNlsGetConvRate(crate_: &mut sunrealtype, cvode_mem: &mut Option<Box<dyn Any>>) -> SUNErrCode {
    let cv_mem = match cvode_mem
        .as_mut()
        .and_then(|b| b.downcast_ref::<CVodeMem>())
        .cloned()
    {
        Some(cv_mem) => cv_mem,
        None => return SUN_ERR_ARG_CORRUPT,
    };

    *crate_ = cv_mem.borrow().cv_crate;
    SUN_SUCCESS
}

fn cvNlsResidual(ycor: &N_Vector, res: &N_Vector, cvode_mem: &mut Option<Box<dyn Any>>) -> i32 {
    let cv_mem = match cvode_mem
        .as_mut()
        .and_then(|b| b.downcast_ref::<CVodeMem>())
        .cloned()
    {
        Some(cv_mem) => cv_mem,
        None => {
            cvProcessError(
                None,
                CV_MEM_NULL,
                line!() as i32,
                "cvNlsResidual",
                file!(),
                MSGCV_NO_MEM,
            );
            return CV_MEM_NULL;
        }
    };

    let (zn0, y, tn, nls_f, ftemp);
    {
        let m = cv_mem.borrow();
        zn0 = m.cv_zn[0].as_ref().expect("cv_zn[0]").clone();
        y = m.cv_y.as_ref().expect("cv_y").clone();
        tn = m.cv_tn;
        nls_f = m.nls_f.expect("nls_f");
        ftemp = m.cv_ftemp.as_ref().expect("cv_ftemp").clone();
    }

    /* update the state based on the current correction */
    N_VLinearSum(ONE, &zn0, ONE, ycor, &y);

    /* evaluate the rhs function */
    let mut user_data = cv_mem.borrow_mut().cv_user_data.take();
    let retval = nls_f(tn, &y, &ftemp, &mut user_data);
    {
        let mut m = cv_mem.borrow_mut();
        m.cv_user_data = user_data;
        m.cv_nfe += 1;
    }
    if retval < 0 {
        return CV_RHSFUNC_FAIL;
    }
    if retval > 0 {
        return RHSFUNC_RECVR;
    }

    /* compute the resiudal */
    let (rl1, gamma, zn1);
    {
        let m = cv_mem.borrow();
        rl1 = m.cv_rl1;
        gamma = m.cv_gamma;
        zn1 = m.cv_zn[1].as_ref().expect("cv_zn[1]").clone();
    }
    N_VLinearSum(rl1, &zn1, ONE, ycor, res);
    N_VLinearSum(-gamma, &ftemp, ONE, res, res);

    CV_SUCCESS
}

fn cvNlsFPFunction(ycor: &N_Vector, res: &N_Vector, cvode_mem: &mut Option<Box<dyn Any>>) -> i32 {
    let cv_mem = match cvode_mem
        .as_mut()
        .and_then(|b| b.downcast_ref::<CVodeMem>())
        .cloned()
    {
        Some(cv_mem) => cv_mem,
        None => {
            cvProcessError(
                None,
                CV_MEM_NULL,
                line!() as i32,
                "cvNlsFPFunction",
                file!(),
                MSGCV_NO_MEM,
            );
            return CV_MEM_NULL;
        }
    };

    let (zn0, y, tn, nls_f);
    {
        let m = cv_mem.borrow();
        zn0 = m.cv_zn[0].as_ref().expect("cv_zn[0]").clone();
        y = m.cv_y.as_ref().expect("cv_y").clone();
        tn = m.cv_tn;
        nls_f = m.nls_f.expect("nls_f");
    }

    /* update the state based on the current correction */
    N_VLinearSum(ONE, &zn0, ONE, ycor, &y);

    /* evaluate the rhs function */
    let mut user_data = cv_mem.borrow_mut().cv_user_data.take();
    let retval = nls_f(tn, &y, res, &mut user_data);
    {
        let mut m = cv_mem.borrow_mut();
        m.cv_user_data = user_data;
        m.cv_nfe += 1;
    }
    if retval < 0 {
        return CV_RHSFUNC_FAIL;
    }
    if retval > 0 {
        return RHSFUNC_RECVR;
    }

    let (h, rl1, zn1);
    {
        let m = cv_mem.borrow();
        h = m.cv_h;
        rl1 = m.cv_rl1;
        zn1 = m.cv_zn[1].as_ref().expect("cv_zn[1]").clone();
    }
    N_VLinearSum(h, res, -ONE, &zn1, res);
    N_VScale(rl1, res, res);

    CV_SUCCESS
}
