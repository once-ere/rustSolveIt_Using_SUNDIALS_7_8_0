//! Port of `src/idas/idas_nls_stg.c`: the IDAS nonlinear solver
//! interface for the IDA_STAGGERED sensitivity corrector.
//!
//! The staggered corrector solves only the `Ns` sensitivity systems, so
//! its `sundials_nvector_senswrapper` vectors hold exactly `Ns`
//! subvectors (no state slot, unlike the SIMULTANEOUS wrappers). The C
//! macro `NV_VECS_SW(v)` — the wrapper's whole `N_Vector*` array —
//! becomes [`NV_VECS_SW`] below, a `Vec` of handle clones aliasing the
//! very same vectors (`Rc` clone == C pointer copy), so writing through
//! an element writes through the wrapper exactly as in C.
//!
//! The C `void* ida_mem` handed to the SUNNonlinearSolver as the
//! integrator mem / ctest / update-norm data maps to a boxed `IDAMem`
//! clone inside an `Option<Box<dyn Any>>` token; each callback downcasts
//! the token back to the handle and uses granular borrows (never holding
//! a borrow across a user callback, an `N_Vector` op on a user-visible
//! vector, a linear-solver call, or `IDAProcessError`).
//!
//! `ONE`, `TWENTY`, `MAXIT` and `RATEMAX` come from `idas_impl` (the
//! shared fragment-protocol constant block). NOTE that `idas_nls_stg.c`
//! deliberately does NOT define `PT0001`: unlike the state and
//! SIMULTANEOUS convergence tests, the staggered direct test is
//! `delnrmS <= toldel` with NO `PT0001` factor — preserved verbatim.
//!
//! Counter attribution (matches the C exactly): `idaNlsLSetupSensStg`
//! bumps the SENSITIVITY setup counter `ida_nsetupsS` (and, unlike the
//! state/SIMULTANEOUS setups, does NOT clear `ida_forceSetup`);
//! `idaNlsResidualSensStg` bumps only `ida_nrSe`. The convergence test
//! drives `ida_delnrmS` / `ida_ssS` (but shares `ida_oldnrm` with the
//! state test, exactly as in C).

use std::any::Any;

use crate::idas_impl::*;
use sundials_core::sundials_errors::{SUN_ERR_ARG_CORRUPT, SUN_SUCCESS};
use sundials_core::sundials_math::SUNRpowerR;
use sundials_core::sundials_nonlinearsolver::*;
use sundials_core::sundials_nvector::{N_VDestroy, N_VLinearSumVectorArray, N_VWrmsNorm, N_Vector};
use sundials_core::sundials_nvector_senswrapper::{
    NV_VEC_SW_set, N_VNewEmpty_SensWrapper, NV_NVECS_SW, NV_VEC_SW,
};
use sundials_core::sundials_types::*;

/// C macro `NV_VECS_SW(v)`: the wrapper's subvector array.
fn NV_VECS_SW(v: &N_Vector) -> Vec<N_Vector> {
    let nvecs = NV_NVECS_SW(v);
    (0..nvecs).map(|i| NV_VEC_SW(v, i)).collect()
}

/* -----------------------------------------------------------------------------
 * Exported functions
 * ---------------------------------------------------------------------------*/

pub fn IDASetNonlinearSolverSensStg(ida_mem: &IDAMem, NLS: &SUNNonlinearSolver) -> i32 {
    /* return immediately if IDA memory is NULL: handled by the type system */
    let IDA_mem = ida_mem;
    let mut retval: i32;

    /* return immediately if NLS memory is NULL ("NLS must be non-NULL"):
    handled by the type system */

    /* check for required nonlinear solver functions */
    let ops_missing = {
        let ops = NLS.ops.borrow();
        ops.gettype.is_none() || ops.solve.is_none() || ops.setsysfn.is_none()
    };
    if ops_missing {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDASetNonlinearSolverSensStg",
            file!(),
            "NLS does not support required operations",
        );
        return IDA_ILL_INPUT;
    }

    /* check for allowed nonlinear solver types */
    if SUNNonlinSolGetType(NLS) != SUNNONLINEARSOLVER_ROOTFIND {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDASetNonlinearSolverSensStg",
            file!(),
            "NLS type must be SUNNONLINEARSOLVER_ROOTFIND",
        );
        return IDA_ILL_INPUT;
    }

    /* check that sensitivities were initialized */
    if !(IDA_mem.borrow().ida_sensi) {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDASetNonlinearSolverSensStg",
            file!(),
            MSG_NO_SENSI,
        );
        return IDA_ILL_INPUT;
    }

    /* check that the staggered corrector was selected */
    if IDA_mem.borrow().ida_ism != IDA_STAGGERED {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDASetNonlinearSolverSensStg",
            file!(),
            "Sensitivity solution method is not IDA_STAGGERED",
        );
        return IDA_ILL_INPUT;
    }

    /* free any existing nonlinear solver */
    let (old_nls, own_nls) = {
        let mut m = IDA_mem.borrow_mut();
        (m.NLSstg.take(), m.ownNLSstg)
    };
    if old_nls.is_some() && own_nls {
        /* C stores the result in retval but never reads it */
        let _ = SUNNonlinSolFree(old_nls);
    }

    {
        let mut m = IDA_mem.borrow_mut();

        /* set SUNNonlinearSolver pointer */
        m.NLSstg = Some(NLS.clone());

        /* Set NLS ownership flag. If this function was called to attach the default
        NLS, IDA will set the flag to SUNTRUE after this function returns. */
        m.ownNLSstg = SUNFALSE;
    }

    let nls = { IDA_mem.borrow().NLSstg.clone() }.expect("NLSstg");

    /* set the nonlinear residual function */
    retval = SUNNonlinSolSetSysFn(&nls, Some(idaNlsResidualSensStg));
    if retval != IDA_SUCCESS {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDASetNonlinearSolverSensStg",
            file!(),
            "Setting nonlinear system function failed",
        );
        return IDA_ILL_INPUT;
    }

    /* set convergence test function */
    retval = SUNNonlinSolSetConvTestFn(
        &nls,
        Some(idaNlsConvTestSensStg),
        Some(Box::new(IDA_mem.clone())),
    );
    if retval != IDA_SUCCESS {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDASetNonlinearSolverSensStg",
            file!(),
            "Setting convergence test function failed",
        );
        return IDA_ILL_INPUT;
    }

    retval = SUNNonlinSolSetGetUpdateNormFn(
        &nls,
        Some(idaNlsGetUpdateNormSensStg),
        Some(Box::new(IDA_mem.clone())),
    );
    if retval != IDA_SUCCESS {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDASetNonlinearSolverSensStg",
            file!(),
            "Setting update-norm getter failed",
        );
        return IDA_ILL_INPUT;
    }

    /* set max allowed nonlinear iterations */
    retval = SUNNonlinSolSetMaxIters(&nls, MAXIT);
    if retval != IDA_SUCCESS {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDASetNonlinearSolverSensStg",
            file!(),
            "Setting maximum number of nonlinear iterations failed",
        );
        return IDA_ILL_INPUT;
    }

    /* create vector wrappers if necessary */
    let stgMallocDone = IDA_mem.borrow().stgMallocDone;
    if stgMallocDone == SUNFALSE {
        let (Ns, sunctx) = {
            let m = IDA_mem.borrow();
            (m.ida_Ns, m.ida_sunctx.clone())
        };

        let ypredictStg = N_VNewEmpty_SensWrapper(Ns, &sunctx);
        let ypredictStg_null = ypredictStg.is_none();
        IDA_mem.borrow_mut().ypredictStg = ypredictStg;
        if ypredictStg_null {
            IDAProcessError(
                Some(IDA_mem),
                IDA_MEM_FAIL,
                line!() as i32,
                "IDASetNonlinearSolverSensStg",
                file!(),
                MSG_MEM_FAIL,
            );
            return IDA_MEM_FAIL;
        }

        let ycorStg = N_VNewEmpty_SensWrapper(Ns, &sunctx);
        let ycorStg_null = ycorStg.is_none();
        IDA_mem.borrow_mut().ycorStg = ycorStg;
        if ycorStg_null {
            /* C leaves the destroyed handle in the mem; the port takes it out */
            let v = IDA_mem.borrow_mut().ypredictStg.take();
            if let Some(v) = v {
                N_VDestroy(v);
            }
            IDAProcessError(
                Some(IDA_mem),
                IDA_MEM_FAIL,
                line!() as i32,
                "IDASetNonlinearSolverSensStg",
                file!(),
                MSG_MEM_FAIL,
            );
            return IDA_MEM_FAIL;
        }

        let ewtStg = N_VNewEmpty_SensWrapper(Ns, &sunctx);
        let ewtStg_null = ewtStg.is_none();
        IDA_mem.borrow_mut().ewtStg = ewtStg;
        if ewtStg_null {
            let v = IDA_mem.borrow_mut().ypredictStg.take();
            if let Some(v) = v {
                N_VDestroy(v);
            }
            let v = IDA_mem.borrow_mut().ycorStg.take();
            if let Some(v) = v {
                N_VDestroy(v);
            }
            IDAProcessError(
                Some(IDA_mem),
                IDA_MEM_FAIL,
                line!() as i32,
                "IDASetNonlinearSolverSensStg",
                file!(),
                MSG_MEM_FAIL,
            );
            return IDA_MEM_FAIL;
        }

        IDA_mem.borrow_mut().stgMallocDone = SUNTRUE;
    }

    /* attach vectors to vector wrappers */
    let (ypredictStg, ycorStg, ewtStg, Ns) = {
        let m = IDA_mem.borrow();
        (
            m.ypredictStg.as_ref().expect("ypredictStg").clone(),
            m.ycorStg.as_ref().expect("ycorStg").clone(),
            m.ewtStg.as_ref().expect("ewtStg").clone(),
            m.ida_Ns,
        )
    };
    for is in 0..Ns {
        let (yySpredict_is, eeS_is, ewtS_is) = {
            let m = IDA_mem.borrow();
            (
                m.ida_yySpredict[is as usize].clone(),
                m.ida_eeS[is as usize].clone(),
                m.ida_ewtS[is as usize].clone(),
            )
        };
        NV_VEC_SW_set(&ypredictStg, is, Some(yySpredict_is));
        NV_VEC_SW_set(&ycorStg, is, Some(eeS_is));
        NV_VEC_SW_set(&ewtStg, is, Some(ewtS_is));
    }

    IDA_SUCCESS
}

/* -----------------------------------------------------------------------------
 * Private functions
 * ---------------------------------------------------------------------------*/

pub fn idaNlsInitSensStg(IDA_mem: &IDAMem) -> i32 {
    let mut retval: i32;

    let nls = { IDA_mem.borrow().NLSstg.clone() }.expect("NLSstg");

    /* set the linear solver setup wrapper function */
    if IDA_mem.borrow().ida_lsetup.is_some() {
        retval = SUNNonlinSolSetLSetupFn(&nls, Some(idaNlsLSetupSensStg));
    } else {
        retval = SUNNonlinSolSetLSetupFn(&nls, None);
    }

    if retval != IDA_SUCCESS {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "idaNlsInitSensStg",
            file!(),
            "Setting the linear solver setup function failed",
        );
        return IDA_NLS_INIT_FAIL;
    }

    /* set the linear solver solve wrapper function */
    if IDA_mem.borrow().ida_lsolve.is_some() {
        retval = SUNNonlinSolSetLSolveFn(&nls, Some(idaNlsLSolveSensStg));
    } else {
        retval = SUNNonlinSolSetLSolveFn(&nls, None);
    }

    if retval != IDA_SUCCESS {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "idaNlsInitSensStg",
            file!(),
            "Setting linear solver solve function failed",
        );
        return IDA_NLS_INIT_FAIL;
    }

    /* initialize nonlinear solver */
    retval = SUNNonlinSolInitialize(&nls);

    if retval != IDA_SUCCESS {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "idaNlsInitSensStg",
            file!(),
            MSG_NLS_INIT_FAIL,
        );
        return IDA_NLS_INIT_FAIL;
    }

    IDA_SUCCESS
}

fn idaNlsLSetupSensStg(
    _jbad: sunbooleantype,
    jcur: &mut sunbooleantype,
    ida_mem: &mut Option<Box<dyn Any>>,
) -> i32 {
    let IDA_mem = match ida_mem
        .as_mut()
        .and_then(|b| b.downcast_ref::<IDAMem>())
        .cloned()
    {
        Some(IDA_mem) => IDA_mem,
        None => {
            IDAProcessError(
                None,
                IDA_MEM_NULL,
                line!() as i32,
                "idaNlsLSetupSensStg",
                file!(),
                MSG_NO_MEM,
            );
            return IDA_MEM_NULL;
        }
    };

    IDA_mem.borrow_mut().ida_nsetupsS += 1;

    let (lsetup, yy, yp, delta, tmpS1, tmpS2, tmpS3) = {
        let m = IDA_mem.borrow();
        (
            m.ida_lsetup.expect("ida_lsetup"),
            m.ida_yy.clone().expect("ida_yy"),
            m.ida_yp.clone().expect("ida_yp"),
            m.ida_delta.clone().expect("ida_delta"),
            m.ida_tmpS1.clone().expect("ida_tmpS1"),
            m.ida_tmpS2.clone().expect("ida_tmpS2"),
            m.ida_tmpS3.clone().expect("ida_tmpS3"),
        )
    };
    let retval = lsetup(&IDA_mem, &yy, &yp, &delta, &tmpS1, &tmpS2, &tmpS3);

    /* update Jacobian status */
    *jcur = SUNTRUE;

    /* update convergence test constants */
    {
        let mut m = IDA_mem.borrow_mut();
        m.ida_cjold = m.ida_cj;
        m.ida_cjratio = ONE;
        m.ida_ss = TWENTY;
        m.ida_ssS = TWENTY;
        m.ida_delnrmS = 0.0; /* SUN_RCONST(0.0) */
    }

    if retval < 0 {
        return IDA_LSETUP_FAIL;
    }
    if retval > 0 {
        return IDA_LSETUP_RECVR;
    }

    IDA_SUCCESS
}

fn idaNlsLSolveSensStg(deltaStg: &N_Vector, ida_mem: &mut Option<Box<dyn Any>>) -> i32 {
    let IDA_mem = match ida_mem
        .as_mut()
        .and_then(|b| b.downcast_ref::<IDAMem>())
        .cloned()
    {
        Some(IDA_mem) => IDA_mem,
        None => {
            IDAProcessError(
                None,
                IDA_MEM_NULL,
                line!() as i32,
                "idaNlsLSolveSensStg",
                file!(),
                MSG_NO_MEM,
            );
            return IDA_MEM_NULL;
        }
    };

    let Ns = IDA_mem.borrow().ida_Ns;
    for is in 0..Ns as usize {
        let (lsolve, ewtS_is, yy, yp, delta) = {
            let m = IDA_mem.borrow();
            (
                m.ida_lsolve.expect("ida_lsolve"),
                m.ida_ewtS[is].clone(),
                m.ida_yy.clone().expect("ida_yy"),
                m.ida_yp.clone().expect("ida_yp"),
                m.ida_delta.clone().expect("ida_delta"),
            )
        };
        let deltaStg_is = NV_VEC_SW(deltaStg, is as i32);
        let retval = lsolve(&IDA_mem, &deltaStg_is, &ewtS_is, &yy, &yp, &delta);

        if retval < 0 {
            return IDA_LSOLVE_FAIL;
        }
        if retval > 0 {
            return IDA_LSOLVE_RECVR;
        }
    }

    IDA_SUCCESS
}

fn idaNlsResidualSensStg(
    ycorStg: &N_Vector,
    resStg: &N_Vector,
    ida_mem: &mut Option<Box<dyn Any>>,
) -> i32 {
    let IDA_mem = match ida_mem
        .as_mut()
        .and_then(|b| b.downcast_ref::<IDAMem>())
        .cloned()
    {
        Some(IDA_mem) => IDA_mem,
        None => {
            IDAProcessError(
                None,
                IDA_MEM_NULL,
                line!() as i32,
                "idaNlsResidualSensStg",
                file!(),
                MSG_NO_MEM,
            );
            return IDA_MEM_NULL;
        }
    };

    /* update yS and ypS based on the current correction */
    let ycorS = NV_VECS_SW(ycorStg);
    let (Ns, cj, yySpredict, ypSpredict, yyS, ypS) = {
        let m = IDA_mem.borrow();
        (
            m.ida_Ns,
            m.ida_cj,
            m.ida_yySpredict.clone(),
            m.ida_ypSpredict.clone(),
            m.ida_yyS.clone(),
            m.ida_ypS.clone(),
        )
    };
    /* C discards the return values here */
    let _ = N_VLinearSumVectorArray(Ns, ONE, &yySpredict, ONE, &ycorS, &yyS);
    let _ = N_VLinearSumVectorArray(Ns, ONE, &ypSpredict, cj, &ycorS, &ypS);

    /* evaluate sens residual */
    let resS_vecs = NV_VECS_SW(resStg);
    let (resS_fn, tn, yy, yp, delta, tmpS1, tmpS2, tmpS3) = {
        let m = IDA_mem.borrow();
        (
            m.ida_resS.expect("ida_resS"),
            m.ida_tn,
            m.ida_yy.clone().expect("ida_yy"),
            m.ida_yp.clone().expect("ida_yp"),
            m.ida_delta.clone().expect("ida_delta"),
            m.ida_tmpS1.clone().expect("ida_tmpS1"),
            m.ida_tmpS2.clone().expect("ida_tmpS2"),
            m.ida_tmpS3.clone().expect("ida_tmpS3"),
        )
    };
    let mut user_dataS = IDA_mem.borrow_mut().ida_user_dataS.take();
    /* C: `ida_user_dataS` is `IDA_mem` when the internal DQ residual is in
    use and `ida_user_data` otherwise (idas.c:1359/1365). Invariant D:
    `Some(box)` is the module-owned token, `None` means hand over
    `ida_user_data`. */
    let resS_from_user_data = user_dataS.is_none();
    if resS_from_user_data {
        user_dataS = IDA_mem.borrow_mut().ida_user_data.take();
    }
    let retval = resS_fn(
        Ns,
        tn,
        &yy,
        &yp,
        &delta,
        &yyS,
        &ypS,
        &resS_vecs,
        &mut user_dataS,
        &tmpS1,
        &tmpS2,
        &tmpS3,
    );
    {
        let mut m = IDA_mem.borrow_mut();
        if resS_from_user_data {
            m.ida_user_data = user_dataS;
        } else {
            m.ida_user_dataS = user_dataS;
        }

        /* increment the number of sens residual evaluations */
        m.ida_nrSe += 1;
    }

    if retval < 0 {
        return IDA_SRES_FAIL;
    }
    if retval > 0 {
        return IDA_SRES_RECVR;
    }

    IDA_SUCCESS
}

fn idaNlsGetUpdateNormSensStg(
    delnrm: &mut sunrealtype,
    ida_mem: &mut Option<Box<dyn Any>>,
) -> SUNErrCode {
    let IDA_mem = match ida_mem
        .as_mut()
        .and_then(|b| b.downcast_ref::<IDAMem>())
        .cloned()
    {
        Some(IDA_mem) => IDA_mem,
        None => return SUN_ERR_ARG_CORRUPT,
    };

    *delnrm = IDA_mem.borrow().ida_delnrmS;
    SUN_SUCCESS
}

fn idaNlsConvTestSensStg(
    NLS: &SUNNonlinearSolver,
    _ycor: &N_Vector,
    del: &N_Vector,
    tol: sunrealtype,
    ewt: &N_Vector,
    ida_mem: &mut Option<Box<dyn Any>>,
) -> i32 {
    let IDA_mem = match ida_mem
        .as_mut()
        .and_then(|b| b.downcast_ref::<IDAMem>())
        .cloned()
    {
        Some(IDA_mem) => IDA_mem,
        None => {
            IDAProcessError(
                None,
                IDA_MEM_NULL,
                line!() as i32,
                "idaNlsConvTestSensStg",
                file!(),
                MSG_NO_MEM,
            );
            return IDA_MEM_NULL;
        }
    };

    /* compute the norm of the correction */
    let delnrmS = N_VWrmsNorm(del, ewt);
    IDA_mem.borrow_mut().ida_delnrmS = delnrmS;

    /* get the current nonlinear solver iteration count */
    let mut m: i32 = 0;
    let retval = SUNNonlinSolGetCurIter(NLS, &mut m);
    if retval != IDA_SUCCESS {
        return IDA_MEM_NULL;
    }

    /* test for convergence, first directly, then with rate estimate. */
    if m == 0 {
        let toldel = {
            let mut mm = IDA_mem.borrow_mut();
            mm.ida_oldnrm = mm.ida_delnrmS;
            mm.ida_toldel
        };
        /* NOTE: no PT0001 factor here (unlike the state / SIMULTANEOUS
        tests) — this is the C code verbatim */
        if delnrmS <= toldel {
            return SUN_SUCCESS;
        }
    } else {
        let oldnrm = IDA_mem.borrow().ida_oldnrm;
        let rate = SUNRpowerR(delnrmS / oldnrm, ONE / (m as sunrealtype));
        if rate > RATEMAX {
            return SUN_NLS_CONV_RECVR;
        }
        IDA_mem.borrow_mut().ida_ssS = rate / (ONE - rate);
    }

    if IDA_mem.borrow().ida_ssS * delnrmS <= tol {
        return SUN_SUCCESS;
    }

    /* not yet converged */
    SUN_NLS_CONTINUE
}
