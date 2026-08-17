//! Port of `src/cvodes/cvodes_nls_stg1.c`: the CVODES nonlinear solver
//! interface for the STAGGERED1 sensitivity corrector (one sensitivity
//! system at a time, selected by `cv_mem->sens_solve_idx`).
//!
//! The C `void* cvode_mem` handed to the SUNNonlinearSolver as the
//! integrator mem / ctest / norm / getter data maps to a boxed
//! `CVodeMem` clone inside an `Option<Box<dyn Any>>` token; each callback
//! downcasts the token back to the handle and uses granular borrows
//! (never holding a borrow across a user callback, an `N_Vector` op on a
//! user-visible vector, a linear-solver call, or `cvProcessError`).
//!
//! Unlike the STAGGERED corrector, STAGGERED1 works on plain `N_Vector`s
//! (no SensWrapper) and allocates no vector wrappers.
//!
//! `ONE` (the file-local `#define`) is the shared `cvodes_impl::ONE`.

use std::any::Any;

use crate::cvodes::cvSensRhs1Wrapper;
use crate::cvodes_impl::*;
use sundials_core::sundials_errors::{SUN_ERR_ARG_CORRUPT, SUN_SUCCESS};
use sundials_core::sundials_math::{SUNMAX, SUNMIN};
use sundials_core::sundials_nonlinearsolver::*;
use sundials_core::sundials_nvector::{N_VLinearSum, N_VScale, N_VWrmsNorm, N_Vector};
use sundials_core::sundials_types::*;

/* -----------------------------------------------------------------------------
 * Exported functions
 * ---------------------------------------------------------------------------*/

pub fn CVodeSetNonlinearSolverSensStg1(cvode_mem: &CVodeMem, NLS: &SUNNonlinearSolver) -> i32 {
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
            cvProcessError(
                Some(cv_mem),
                CV_ILL_INPUT,
                line!() as i32,
                "CVodeSetNonlinearSolverSensStg1",
                file!(),
                "NLS does not support required operations",
            );
            return CV_ILL_INPUT;
        }
    }

    /* check that sensitivities were initialized */
    if !(cv_mem.borrow().cv_sensi) {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeSetNonlinearSolverSensStg1",
            file!(),
            MSGCV_NO_SENSI,
        );
        return CV_ILL_INPUT;
    }

    /* check that staggered corrector was selected */
    if cv_mem.borrow().cv_ism != CV_STAGGERED1 {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeSetNonlinearSolverSensStg1",
            file!(),
            "Sensitivity solution method is not CV_STAGGERED1",
        );
        return CV_ILL_INPUT;
    }

    /* free any existing nonlinear solver */
    let (old_nls, own_nls) = {
        let mut m = cv_mem.borrow_mut();
        (m.NLSstg1.take(), m.ownNLSstg1)
    };
    if old_nls.is_some() && own_nls {
        /* C stores the result in retval but never reads it */
        let _ = SUNNonlinSolFree(old_nls);
    }

    {
        let mut m = cv_mem.borrow_mut();
        /* set SUNNonlinearSolver pointer */
        m.NLSstg1 = Some(NLS.clone());

        /* Set NLS ownership flag. If this function was called to attach the default
        NLS, CVODE will set the flag to SUNTRUE after this function returns. */
        m.ownNLSstg1 = SUNFALSE;
    }

    let nls = { cv_mem.borrow().NLSstg1.clone() }.expect("NLSstg1");

    /* set the nonlinear system function */
    if SUNNonlinSolGetType(NLS) == SUNNONLINEARSOLVER_ROOTFIND {
        retval = SUNNonlinSolSetSysFn(&nls, Some(cvNlsResidualSensStg1));
    } else if SUNNonlinSolGetType(NLS) == SUNNONLINEARSOLVER_FIXEDPOINT {
        retval = SUNNonlinSolSetSysFn(&nls, Some(cvNlsFPFunctionSensStg1));
    } else {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeSetNonlinearSolverSensStg1",
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
            "CVodeSetNonlinearSolverSensStg1",
            file!(),
            "Setting nonlinear system function failed",
        );
        return CV_ILL_INPUT;
    }

    /* set convergence test function */
    retval = SUNNonlinSolSetConvTestFn(
        &nls,
        Some(cvNlsConvTestSensStg1),
        Some(Box::new(cv_mem.clone())),
    );
    if retval != CV_SUCCESS {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeSetNonlinearSolverSensStg1",
            file!(),
            "Setting convergence test function failed",
        );
        return CV_ILL_INPUT;
    }

    retval = SUNNonlinSolSetNormFn(
        &nls,
        Some(cvNlsNormSensStg1),
        Some(Box::new(cv_mem.clone())),
    );
    if retval != CV_SUCCESS {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeSetNonlinearSolverSensStg1",
            file!(),
            "Setting convergence-test norm function failed",
        );
        return CV_ILL_INPUT;
    }

    retval = SUNNonlinSolSetGetUpdateNormFn(
        &nls,
        Some(cvNlsGetUpdateNormSensStg1),
        Some(Box::new(cv_mem.clone())),
    );
    if retval != CV_SUCCESS {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeSetNonlinearSolverSensStg1",
            file!(),
            "Setting update-norm getter failed",
        );
        return CV_ILL_INPUT;
    }

    retval = SUNNonlinSolSetGetConvRateFn(
        &nls,
        Some(cvNlsGetConvRateSensStg1),
        Some(Box::new(cv_mem.clone())),
    );
    if retval != CV_SUCCESS {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeSetNonlinearSolverSensStg1",
            file!(),
            "Setting convergence-rate getter failed",
        );
        return CV_ILL_INPUT;
    }

    /* set max allowed nonlinear iterations */
    retval = SUNNonlinSolSetMaxIters(&nls, NLS_MAXCOR);
    if retval != CV_SUCCESS {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeSetNonlinearSolverSensStg1",
            file!(),
            "Setting maximum number of nonlinear iterations failed",
        );
        return CV_ILL_INPUT;
    }

    /* Reset the acnrmScur flag to SUNFALSE (always false for stg1) */
    cv_mem.borrow_mut().cv_acnrmScur = SUNFALSE;

    CV_SUCCESS
}

/* -----------------------------------------------------------------------------
 * Private functions
 * ---------------------------------------------------------------------------*/

pub fn cvNlsInitSensStg1(cvode_mem: &CVodeMem) -> i32 {
    let mut retval: i32;

    let nls = { cvode_mem.borrow().NLSstg1.clone() }.expect("NLSstg1");

    /* set the linear solver setup wrapper function */
    let has_lsetup = cvode_mem.borrow().cv_lsetup.is_some();
    if has_lsetup {
        retval = SUNNonlinSolSetLSetupFn(&nls, Some(cvNlsLSetupSensStg1));
    } else {
        retval = SUNNonlinSolSetLSetupFn(&nls, None);
    }

    if retval != CV_SUCCESS {
        cvProcessError(
            Some(cvode_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "cvNlsInitSensStg1",
            file!(),
            "Setting the linear solver setup function failed",
        );
        return CV_NLS_INIT_FAIL;
    }

    /* set the linear solver solve wrapper function */
    let has_lsolve = cvode_mem.borrow().cv_lsolve.is_some();
    if has_lsolve {
        retval = SUNNonlinSolSetLSolveFn(&nls, Some(cvNlsLSolveSensStg1));
    } else {
        retval = SUNNonlinSolSetLSolveFn(&nls, None);
    }

    if retval != CV_SUCCESS {
        cvProcessError(
            Some(cvode_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "cvNlsInitSensStg1",
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
            "cvNlsInitSensStg1",
            file!(),
            MSGCV_NLS_INIT_FAIL,
        );
        return CV_NLS_INIT_FAIL;
    }

    /* reset previous iteration count for updating nniS1 */
    cvode_mem.borrow_mut().nnip = 0;

    CV_SUCCESS
}

fn cvNlsLSetupSensStg1(
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
                "cvNlsLSetupSensStg1",
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

    /* setup the linear solver (C passes &cv_mem->cv_jcur, so the callee's
    writes land in the mem; copy out / call / copy back reproduces that
    aliasing, and *jcur is re-read from the mem afterwards) */
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
        m.cv_nsetupsS += 1;

        /* update Jacobian status */
        *jcur = m.cv_jcur;

        m.cv_gamrat = ONE;
        m.cv_gammap = m.cv_gamma;
        m.cv_crate = ONE;
        m.cv_crateS = ONE;
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

fn cvNlsLSolveSensStg1(delta: &N_Vector, cvode_mem: &mut Option<Box<dyn Any>>) -> i32 {
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
                "cvNlsLSolveSensStg1",
                file!(),
                MSGCV_NO_MEM,
            );
            return CV_MEM_NULL;
        }
    };

    /* solve the sensitivity linear systems */
    let (lsolve, ewtS, y, ftemp);
    {
        let m = cv_mem.borrow();
        /* get index of current sensitivity solve */
        let is = m.sens_solve_idx as usize;
        lsolve = m.cv_lsolve.expect("cv_lsolve");
        ewtS = m.cv_ewtS[is].clone();
        y = m.cv_y.as_ref().expect("cv_y").clone();
        ftemp = m.cv_ftemp.as_ref().expect("cv_ftemp").clone();
    }
    let retval = lsolve(&cv_mem, delta, &ewtS, &y, &ftemp);

    if retval < 0 {
        return CV_LSOLVE_FAIL;
    }
    if retval > 0 {
        return SUN_NLS_CONV_RECVR;
    }

    CV_SUCCESS
}

fn cvNlsConvTestSensStg1(
    NLS: &SUNNonlinearSolver,
    _ycor: &N_Vector,
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
                "cvNlsConvTestSensStg1",
                file!(),
                MSGCV_NO_MEM,
            );
            return CV_MEM_NULL;
        }
    };

    /* compute the norm of the state and sensitivity corrections (C writes
    through &cv_mem->cv_delnrm; ported as copy-out / call / copy-back) */
    let mut delnrm = cv_mem.borrow().cv_delnrm;
    let nrm_retval = cvNlsNormSensStg1(delta, ewt, &mut delnrm, cvode_mem);
    cv_mem.borrow_mut().cv_delnrm = delnrm;
    if nrm_retval != SUN_SUCCESS {
        /* unreachable: cvNlsNormSensStg1 always succeeds (as in C); the C call
        site passes no time argument for the MSG_TIME format — use cv_tn */
        let tn = cv_mem.borrow().cv_tn;
        cvProcessError(
            Some(&cv_mem),
            CV_NLS_FAIL,
            line!() as i32,
            "cvNlsConvTestSensStg1",
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
       rate constant is stored in crate, and used in the test.
    */
    let dcon: sunrealtype;
    {
        let mut mm = cv_mem.borrow_mut();
        if m > 0 {
            mm.cv_crateS = SUNMAX(CRDOWN * mm.cv_crateS, mm.cv_delnrm / mm.cv_delp);
        }
        dcon = mm.cv_delnrm * SUNMIN(ONE, mm.cv_crateS) / tol;
    }

    /* check if nonlinear system was solved successfully */
    if dcon <= ONE {
        return CV_SUCCESS;
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

fn cvNlsNormSensStg1(
    delta: &N_Vector,
    ewt: &N_Vector,
    delnrm: &mut sunrealtype,
    _cvode_mem: &mut Option<Box<dyn Any>>,
) -> SUNErrCode {
    *delnrm = N_VWrmsNorm(delta, ewt);
    SUN_SUCCESS
}

fn cvNlsGetUpdateNormSensStg1(
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

fn cvNlsGetConvRateSensStg1(
    crate_: &mut sunrealtype,
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

    *crate_ = cv_mem.borrow().cv_crateS;
    SUN_SUCCESS
}

fn cvNlsResidualSensStg1(
    ycor: &N_Vector,
    res: &N_Vector,
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
                "cvNlsResidualSensStg1",
                file!(),
                MSGCV_NO_MEM,
            );
            return CV_MEM_NULL;
        }
    };

    /* get index of current sensitivity solve */
    let is = cv_mem.borrow().sens_solve_idx as usize;

    /* update sensitivity based on the current correction */
    let (znS0is, ySis) = {
        let m = cv_mem.borrow();
        (m.cv_znS[0][is].clone(), m.cv_yS[is].clone())
    };
    N_VLinearSum(ONE, &znS0is, ONE, ycor, &ySis);

    /* evaluate the sensitivity rhs function */
    let (tn, y, ftemp, ftempSis, vtemp1, vtemp2) = {
        let m = cv_mem.borrow();
        (
            m.cv_tn,
            m.cv_y.as_ref().expect("cv_y").clone(),
            m.cv_ftemp.as_ref().expect("cv_ftemp").clone(),
            m.cv_ftempS[is].clone(),
            m.cv_vtemp1.as_ref().expect("cv_vtemp1").clone(),
            m.cv_vtemp2.as_ref().expect("cv_vtemp2").clone(),
        )
    };
    let retval = cvSensRhs1Wrapper(
        &cv_mem, tn, &y, &ftemp, is as i32, &ySis, &ftempSis, &vtemp1, &vtemp2,
    );

    if retval < 0 {
        return CV_SRHSFUNC_FAIL;
    }
    if retval > 0 {
        return SRHSFUNC_RECVR;
    }

    /* compute the sensitivity resiudal (C re-reads the mem fields here,
    after the RHS call) */
    let (rl1, znS1is, gamma, ftempSis) = {
        let m = cv_mem.borrow();
        (
            m.cv_rl1,
            m.cv_znS[1][is].clone(),
            m.cv_gamma,
            m.cv_ftempS[is].clone(),
        )
    };
    N_VLinearSum(rl1, &znS1is, ONE, ycor, res);
    N_VLinearSum(-gamma, &ftempSis, ONE, res, res);

    CV_SUCCESS
}

fn cvNlsFPFunctionSensStg1(
    ycor: &N_Vector,
    res: &N_Vector,
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
                "cvNlsFPFunctionSensStg1",
                file!(),
                MSGCV_NO_MEM,
            );
            return CV_MEM_NULL;
        }
    };

    /* get index of current sensitivity solve */
    let is = cv_mem.borrow().sens_solve_idx as usize;

    /* update the sensitivities based on the current correction */
    let (znS0is, ySis) = {
        let m = cv_mem.borrow();
        (m.cv_znS[0][is].clone(), m.cv_yS[is].clone())
    };
    N_VLinearSum(ONE, &znS0is, ONE, ycor, &ySis);

    /* evaluate the sensitivity rhs function */
    let (tn, y, ftemp, vtemp1, vtemp2) = {
        let m = cv_mem.borrow();
        (
            m.cv_tn,
            m.cv_y.as_ref().expect("cv_y").clone(),
            m.cv_ftemp.as_ref().expect("cv_ftemp").clone(),
            m.cv_vtemp1.as_ref().expect("cv_vtemp1").clone(),
            m.cv_vtemp2.as_ref().expect("cv_vtemp2").clone(),
        )
    };
    let retval = cvSensRhs1Wrapper(
        &cv_mem, tn, &y, &ftemp, is as i32, &ySis, res, &vtemp1, &vtemp2,
    );

    if retval < 0 {
        return CV_SRHSFUNC_FAIL;
    }
    if retval > 0 {
        return SRHSFUNC_RECVR;
    }

    /* evaluate sensitivity fixed point function */
    let (h, rl1, znS1is) = {
        let m = cv_mem.borrow();
        (m.cv_h, m.cv_rl1, m.cv_znS[1][is].clone())
    };
    N_VLinearSum(h, res, -ONE, &znS1is, res);
    N_VScale(rl1, res, res);

    CV_SUCCESS
}
