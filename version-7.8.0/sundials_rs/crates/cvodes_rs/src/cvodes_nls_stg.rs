//! Port of `src/cvodes/cvodes_nls_stg.c`: the CVODES nonlinear solver
//! interface for the STAGGERED sensitivity corrector.
//!
//! The C `void* cvode_mem` handed to the SUNNonlinearSolver as the
//! integrator mem / ctest / norm / getter data maps to a boxed
//! `CVodeMem` clone inside an `Option<Box<dyn Any>>` token; each callback
//! downcasts the token back to the handle and uses granular borrows
//! (never holding a borrow across a user callback, an `N_Vector` op on a
//! user-visible vector, a linear-solver call, or `cvProcessError`).
//!
//! The C `NV_VECS_SW(v)` macro yields the wrapper's `N_Vector*` array;
//! here it becomes a `Vec<N_Vector>` of handle clones, which alias the
//! very same vectors (`Rc` clone == C pointer copy), so writing through
//! an element writes through the wrapper exactly as in C.
//!
//! `ONE` (the file-local `#define`) is the shared `cvodes_impl::ONE`.

use std::any::Any;

use crate::cvodes::{cvSensNorm, cvSensRhsWrapper};
use crate::cvodes_impl::*;
use sundials_core::sundials_errors::{SUN_ERR_ARG_CORRUPT, SUN_SUCCESS};
use sundials_core::sundials_math::{SUNMAX, SUNMIN};
use sundials_core::sundials_nonlinearsolver::*;
use sundials_core::sundials_nvector::{
    N_VDestroy, N_VLinearCombinationVectorArray, N_VLinearSum, N_VLinearSumVectorArray, N_VScale,
    N_Vector,
};
use sundials_core::sundials_nvector_senswrapper::{
    NV_VEC_SW_set, N_VNewEmpty_SensWrapper, NV_NVECS_SW, NV_VEC_SW,
};
use sundials_core::sundials_types::*;

/* C macro `NV_VECS_SW(v)`: the wrapper's subvector array. */
fn NV_VECS_SW(v: &N_Vector) -> Vec<N_Vector> {
    let nvecs = NV_NVECS_SW(v);
    (0..nvecs).map(|i| NV_VEC_SW(v, i)).collect()
}

/* -----------------------------------------------------------------------------
 * Exported functions
 * ---------------------------------------------------------------------------*/

pub fn CVodeSetNonlinearSolverSensStg(cvode_mem: &CVodeMem, NLS: &SUNNonlinearSolver) -> i32 {
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
                "CVodeSetNonlinearSolverSensStg",
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
            "CVodeSetNonlinearSolverSensStg",
            file!(),
            MSGCV_NO_SENSI,
        );
        return CV_ILL_INPUT;
    }

    /* check that staggered corrector was selected */
    if cv_mem.borrow().cv_ism != CV_STAGGERED {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeSetNonlinearSolverSensStg",
            file!(),
            "Sensitivity solution method is not CV_STAGGERED",
        );
        return CV_ILL_INPUT;
    }

    /* free any existing nonlinear solver */
    let (old_nls, own_nls) = {
        let mut m = cv_mem.borrow_mut();
        (m.NLSstg.take(), m.ownNLSstg)
    };
    if old_nls.is_some() && own_nls {
        /* C stores the result in retval but never reads it */
        let _ = SUNNonlinSolFree(old_nls);
    }

    {
        let mut m = cv_mem.borrow_mut();
        /* set SUNNonlinearSolver pointer */
        m.NLSstg = Some(NLS.clone());

        /* Set NLS ownership flag. If this function was called to attach the default
        NLS, CVODE will set the flag to SUNTRUE after this function returns. */
        m.ownNLSstg = SUNFALSE;
    }

    let nls = { cv_mem.borrow().NLSstg.clone() }.expect("NLSstg");

    /* set the nonlinear system function */
    if SUNNonlinSolGetType(NLS) == SUNNONLINEARSOLVER_ROOTFIND {
        retval = SUNNonlinSolSetSysFn(&nls, Some(cvNlsResidualSensStg));
    } else if SUNNonlinSolGetType(NLS) == SUNNONLINEARSOLVER_FIXEDPOINT {
        retval = SUNNonlinSolSetSysFn(&nls, Some(cvNlsFPFunctionSensStg));
    } else {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeSetNonlinearSolverSensStg",
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
            "CVodeSetNonlinearSolverSensStg",
            file!(),
            "Setting nonlinear system function failed",
        );
        return CV_ILL_INPUT;
    }

    /* set convergence test function */
    retval = SUNNonlinSolSetConvTestFn(
        &nls,
        Some(cvNlsConvTestSensStg),
        Some(Box::new(cv_mem.clone())),
    );
    if retval != CV_SUCCESS {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeSetNonlinearSolverSensStg",
            file!(),
            "Setting convergence test function failed",
        );
        return CV_ILL_INPUT;
    }

    retval = SUNNonlinSolSetNormFn(&nls, Some(cvNlsNormSensStg), Some(Box::new(cv_mem.clone())));
    if retval != CV_SUCCESS {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeSetNonlinearSolverSensStg",
            file!(),
            "Setting convergence-test norm function failed",
        );
        return CV_ILL_INPUT;
    }

    retval = SUNNonlinSolSetGetUpdateNormFn(
        &nls,
        Some(cvNlsGetUpdateNormSensStg),
        Some(Box::new(cv_mem.clone())),
    );
    if retval != CV_SUCCESS {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeSetNonlinearSolverSensStg",
            file!(),
            "Setting update-norm getter failed",
        );
        return CV_ILL_INPUT;
    }

    retval = SUNNonlinSolSetGetConvRateFn(
        &nls,
        Some(cvNlsGetConvRateSensStg),
        Some(Box::new(cv_mem.clone())),
    );
    if retval != CV_SUCCESS {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeSetNonlinearSolverSensStg",
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
            "CVodeSetNonlinearSolverSensStg",
            file!(),
            "Setting maximum number of nonlinear iterations failed",
        );
        return CV_ILL_INPUT;
    }

    /* create vector wrappers if necessary */
    let stgMallocDone = cv_mem.borrow().stgMallocDone;
    if stgMallocDone == SUNFALSE {
        let (Ns, sunctx) = {
            let m = cv_mem.borrow();
            (m.cv_Ns, m.cv_sunctx.clone())
        };

        let zn0Stg = N_VNewEmpty_SensWrapper(Ns, &sunctx);
        if zn0Stg.is_none() {
            cvProcessError(
                Some(cv_mem),
                CV_MEM_FAIL,
                line!() as i32,
                "CVodeSetNonlinearSolverSensStg",
                file!(),
                MSGCV_MEM_FAIL,
            );
            return CV_MEM_FAIL;
        }
        cv_mem.borrow_mut().zn0Stg = zn0Stg;

        let ycorStg = N_VNewEmpty_SensWrapper(Ns, &sunctx);
        if ycorStg.is_none() {
            let v = cv_mem.borrow_mut().zn0Stg.take();
            if let Some(v) = v {
                N_VDestroy(v);
            }
            cvProcessError(
                Some(cv_mem),
                CV_MEM_FAIL,
                line!() as i32,
                "CVodeSetNonlinearSolverSensStg",
                file!(),
                MSGCV_MEM_FAIL,
            );
            return CV_MEM_FAIL;
        }
        cv_mem.borrow_mut().ycorStg = ycorStg;

        let ewtStg = N_VNewEmpty_SensWrapper(Ns, &sunctx);
        if ewtStg.is_none() {
            let v = cv_mem.borrow_mut().zn0Stg.take();
            if let Some(v) = v {
                N_VDestroy(v);
            }
            let v = cv_mem.borrow_mut().ycorStg.take();
            if let Some(v) = v {
                N_VDestroy(v);
            }
            cvProcessError(
                Some(cv_mem),
                CV_MEM_FAIL,
                line!() as i32,
                "CVodeSetNonlinearSolverSensStg",
                file!(),
                MSGCV_MEM_FAIL,
            );
            return CV_MEM_FAIL;
        }
        cv_mem.borrow_mut().ewtStg = ewtStg;

        cv_mem.borrow_mut().stgMallocDone = SUNTRUE;
    }

    /* attach vectors to vector wrappers */
    let (Ns, zn0Stg, ycorStg, ewtStg) = {
        let m = cv_mem.borrow();
        (
            m.cv_Ns,
            m.zn0Stg.as_ref().expect("zn0Stg").clone(),
            m.ycorStg.as_ref().expect("ycorStg").clone(),
            m.ewtStg.as_ref().expect("ewtStg").clone(),
        )
    };
    for is in 0..Ns {
        let (znS0, acorS, ewtS) = {
            let m = cv_mem.borrow();
            (
                m.cv_znS[0][is as usize].clone(),
                m.cv_acorS[is as usize].clone(),
                m.cv_ewtS[is as usize].clone(),
            )
        };
        NV_VEC_SW_set(&zn0Stg, is, Some(znS0));
        NV_VEC_SW_set(&ycorStg, is, Some(acorS));
        NV_VEC_SW_set(&ewtStg, is, Some(ewtS));
    }

    /* Reset the acnrmScur flag to SUNFALSE */
    cv_mem.borrow_mut().cv_acnrmScur = SUNFALSE;

    CV_SUCCESS
}

/* -----------------------------------------------------------------------------
 * Private functions
 * ---------------------------------------------------------------------------*/

pub fn cvNlsInitSensStg(cvode_mem: &CVodeMem) -> i32 {
    let mut retval: i32;

    let nls = { cvode_mem.borrow().NLSstg.clone() }.expect("NLSstg");

    /* set the linear solver setup wrapper function */
    let has_lsetup = cvode_mem.borrow().cv_lsetup.is_some();
    if has_lsetup {
        retval = SUNNonlinSolSetLSetupFn(&nls, Some(cvNlsLSetupSensStg));
    } else {
        retval = SUNNonlinSolSetLSetupFn(&nls, None);
    }

    if retval != CV_SUCCESS {
        cvProcessError(
            Some(cvode_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "cvNlsInitSensStg",
            file!(),
            "Setting the linear solver setup function failed",
        );
        return CV_NLS_INIT_FAIL;
    }

    /* set the linear solver solve wrapper function */
    let has_lsolve = cvode_mem.borrow().cv_lsolve.is_some();
    if has_lsolve {
        retval = SUNNonlinSolSetLSolveFn(&nls, Some(cvNlsLSolveSensStg));
    } else {
        retval = SUNNonlinSolSetLSolveFn(&nls, None);
    }

    if retval != CV_SUCCESS {
        cvProcessError(
            Some(cvode_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "cvNlsInitSensStg",
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
            "cvNlsInitSensStg",
            file!(),
            MSGCV_NLS_INIT_FAIL,
        );
        return CV_NLS_INIT_FAIL;
    }

    CV_SUCCESS
}

fn cvNlsLSetupSensStg(
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
                "cvNlsLSetupSensStg",
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

fn cvNlsLSolveSensStg(deltaStg: &N_Vector, cvode_mem: &mut Option<Box<dyn Any>>) -> i32 {
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
                "cvNlsLSolveSensStg",
                file!(),
                MSGCV_NO_MEM,
            );
            return CV_MEM_NULL;
        }
    };

    /* extract sensitivity deltas from the vector wrapper */
    let deltaS = NV_VECS_SW(deltaStg);

    /* solve the sensitivity linear systems */
    let Ns = cv_mem.borrow().cv_Ns;
    for is in 0..Ns as usize {
        let (lsolve, ewtS, y, ftemp);
        {
            let m = cv_mem.borrow();
            lsolve = m.cv_lsolve.expect("cv_lsolve");
            ewtS = m.cv_ewtS[is].clone();
            y = m.cv_y.as_ref().expect("cv_y").clone();
            ftemp = m.cv_ftemp.as_ref().expect("cv_ftemp").clone();
        }
        let retval = lsolve(&cv_mem, &deltaS[is], &ewtS, &y, &ftemp);

        if retval < 0 {
            return CV_LSOLVE_FAIL;
        }
        if retval > 0 {
            return SUN_NLS_CONV_RECVR;
        }
    }

    CV_SUCCESS
}

fn cvNlsConvTestSensStg(
    NLS: &SUNNonlinearSolver,
    ycorStg: &N_Vector,
    deltaStg: &N_Vector,
    tol: sunrealtype,
    ewtStg: &N_Vector,
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
                "cvNlsConvTestSensStg",
                file!(),
                MSGCV_NO_MEM,
            );
            return CV_MEM_NULL;
        }
    };

    /* extract the current sensitivity corrections */
    let ycorS = NV_VECS_SW(ycorStg);

    /* extract the sensitivity error weights */
    let ewtS = NV_VECS_SW(ewtStg);

    /* compute the norm of the state and sensitivity corrections (C writes
    through &cv_mem->cv_delnrm; ported as copy-out / call / copy-back) */
    let mut delnrm = cv_mem.borrow().cv_delnrm;
    let nrm_retval = cvNlsNormSensStg(deltaStg, ewtStg, &mut delnrm, cvode_mem);
    cv_mem.borrow_mut().cv_delnrm = delnrm;
    if nrm_retval != SUN_SUCCESS {
        /* the C call site passes no time argument for the MSG_TIME format —
        use cv_tn */
        let tn = cv_mem.borrow().cv_tn;
        cvProcessError(
            Some(&cv_mem),
            CV_NLS_FAIL,
            line!() as i32,
            "cvNlsConvTestSensStg",
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

       Recall that, even when errconS=SUNFALSE, all variables are used in the
       convergence test. Hence, we use cv_delnrm. However, acnrm is used in the
       error test and thus it has different forms depending on errconS.
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
        let errconS = cv_mem.borrow().cv_errconS;
        if errconS {
            let acnrmS = if m == 0 {
                let cur_delnrm = cv_mem.borrow().cv_delnrm;
                cur_delnrm
            } else {
                cvSensNorm(&cv_mem, &ycorS, &ewtS)
            };
            let mut mm = cv_mem.borrow_mut();
            mm.cv_acnrmS = acnrmS;
            mm.cv_acnrmScur = SUNTRUE;
        }
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

fn cvNlsNormSensStg(
    deltaStg: &N_Vector,
    ewtStg: &N_Vector,
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

    let deltaS = NV_VECS_SW(deltaStg);
    let ewtS = NV_VECS_SW(ewtStg);
    *delnrm = cvSensNorm(&cv_mem, &deltaS, &ewtS);

    SUN_SUCCESS
}

fn cvNlsGetUpdateNormSensStg(
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

fn cvNlsGetConvRateSensStg(
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

fn cvNlsResidualSensStg(
    ycorStg: &N_Vector,
    resStg: &N_Vector,
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
                "cvNlsResidualSensStg",
                file!(),
                MSGCV_NO_MEM,
            );
            return CV_MEM_NULL;
        }
    };

    /* extract sensitivity and residual vectors from the vector wrapper */
    let ycorS = NV_VECS_SW(ycorStg);
    let resS = NV_VECS_SW(resStg);

    /* update sensitivities based on the current correction */
    let (Ns, znS0, yS) = {
        let m = cv_mem.borrow();
        (m.cv_Ns, m.cv_znS[0].clone(), m.cv_yS.clone())
    };
    let mut retval = N_VLinearSumVectorArray(Ns, ONE, &znS0, ONE, &ycorS, &yS);
    if retval != CV_SUCCESS {
        return CV_VECTOROP_ERR;
    }

    /* evaluate the sensitivity rhs function */
    let (tn, y, ftemp, ftempS, vtemp1, vtemp2) = {
        let m = cv_mem.borrow();
        (
            m.cv_tn,
            m.cv_y.as_ref().expect("cv_y").clone(),
            m.cv_ftemp.as_ref().expect("cv_ftemp").clone(),
            m.cv_ftempS.clone(),
            m.cv_vtemp1.as_ref().expect("cv_vtemp1").clone(),
            m.cv_vtemp2.as_ref().expect("cv_vtemp2").clone(),
        )
    };
    retval = cvSensRhsWrapper(&cv_mem, tn, &y, &ftemp, &yS, &ftempS, &vtemp1, &vtemp2);

    if retval < 0 {
        return CV_SRHSFUNC_FAIL;
    }
    if retval > 0 {
        return SRHSFUNC_RECVR;
    }

    /* compute the sensitivity resiudal */
    let mut cvals: [sunrealtype; 3] = [ZERO; 3];
    let mut XXvecs: Vec<Vec<N_Vector>> = vec![Vec::new(), Vec::new(), Vec::new()];

    /* C re-reads the mem fields here, after the RHS call */
    let (rl1, gamma, znS1, ftempS) = {
        let m = cv_mem.borrow();
        (
            m.cv_rl1,
            m.cv_gamma,
            m.cv_znS[1].clone(),
            m.cv_ftempS.clone(),
        )
    };

    cvals[0] = rl1;
    XXvecs[0] = znS1;
    cvals[1] = ONE;
    XXvecs[1] = ycorS;
    cvals[2] = -gamma;
    XXvecs[2] = ftempS;

    retval = N_VLinearCombinationVectorArray(Ns, 3, &cvals, &XXvecs, &resS);
    if retval != CV_SUCCESS {
        return CV_VECTOROP_ERR;
    }

    CV_SUCCESS
}

fn cvNlsFPFunctionSensStg(
    ycorStg: &N_Vector,
    resStg: &N_Vector,
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
                "cvNlsFPFunctionSensStg",
                file!(),
                MSGCV_NO_MEM,
            );
            return CV_MEM_NULL;
        }
    };

    /* extract sensitivity and residual vectors from the vector wrapper */
    let ycorS = NV_VECS_SW(ycorStg);
    let resS = NV_VECS_SW(resStg);

    /* update the sensitivities based on the current correction */
    let (Ns, znS0, yS) = {
        let m = cv_mem.borrow();
        (m.cv_Ns, m.cv_znS[0].clone(), m.cv_yS.clone())
    };
    let mut retval = N_VLinearSumVectorArray(Ns, ONE, &znS0, ONE, &ycorS, &yS);
    if retval != CV_SUCCESS {
        return CV_VECTOROP_ERR;
    }

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
    retval = cvSensRhsWrapper(&cv_mem, tn, &y, &ftemp, &yS, &resS, &vtemp1, &vtemp2);

    if retval < 0 {
        return CV_SRHSFUNC_FAIL;
    }
    if retval > 0 {
        return SRHSFUNC_RECVR;
    }

    /* evaluate sensitivity fixed point function */
    for is in 0..Ns as usize {
        let (h, rl1, znS1is) = {
            let m = cv_mem.borrow();
            (m.cv_h, m.cv_rl1, m.cv_znS[1][is].clone())
        };
        N_VLinearSum(h, &resS[is], -ONE, &znS1is, &resS[is]);
        N_VScale(rl1, &resS[is], &resS[is]);
    }

    CV_SUCCESS
}
