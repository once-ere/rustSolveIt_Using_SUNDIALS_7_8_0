//! Port of `src/idas/idas_nls_sim.c`: the IDAS nonlinear solver
//! interface for the IDA_SIMULTANEOUS sensitivity corrector.
//!
//! The combined state+sensitivity system vector is the
//! `sundials_nvector_senswrapper` vector: slot 0 is the state, slots
//! `1 ..= Ns` are the sensitivities. C's `NV_VECS_SW(v) + 1` pointer
//! arithmetic becomes [`sens_vecs_sw`], which snapshots the subvector
//! handles (`Rc` clones) — equivalent because every use writes THROUGH
//! the handles, never re-assigning wrapper slots.
//!
//! The C `void* ida_mem` handed to the SUNNonlinearSolver as the
//! integrator mem / ctest / update-norm data maps to a boxed `IDAMem`
//! clone inside an `Option<Box<dyn Any>>` token; each callback downcasts
//! the token back to the handle and uses granular borrows (never holding
//! a borrow across a user callback, an `N_Vector` op on a user-visible
//! vector, a linear-solver call, or `IDAProcessError`).
//!
//! `PT0001`, `ONE`, `TWENTY`, `MAXIT` and `RATEMAX` come from
//! `idas_impl` (the shared fragment-protocol constant block).
//!
//! Counter attribution (matches the C exactly): the SIMULTANEOUS
//! corrector solves the combined system, so `idaNlsLSetupSensSim` bumps
//! the STATE setup counter `ida_nsetups` (not `ida_nsetupsS`) and
//! `idaNlsResidualSensSim` bumps BOTH `ida_nre` (state residual) and
//! `ida_nrSe` (sensitivity residual). The convergence test drives the
//! STATE constants `ida_delnrm` / `ida_oldnrm` / `ida_ss`.

use std::any::Any;

use crate::idas_impl::*;
use sundials_core::sundials_errors::{SUN_ERR_ARG_CORRUPT, SUN_SUCCESS};
use sundials_core::sundials_math::SUNRpowerR;
use sundials_core::sundials_nonlinearsolver::*;
use sundials_core::sundials_nvector::{
    N_VDestroy, N_VLinearSum, N_VLinearSumVectorArray, N_VScale, N_VWrmsNorm, N_Vector,
};
use sundials_core::sundials_nvector_senswrapper::{
    NV_VEC_SW_set, N_VNewEmpty_SensWrapper, NV_NVECS_SW, NV_VEC_SW,
};
use sundials_core::sundials_types::*;

/// C `NV_VECS_SW(v) + 1`: the sensitivity subvector array of a
/// sensitivity-wrapper vector (slot 0 is the state).
fn sens_vecs_sw(v: &N_Vector) -> Vec<N_Vector> {
    (1..NV_NVECS_SW(v)).map(|i| NV_VEC_SW(v, i)).collect()
}

/* -----------------------------------------------------------------------------
 * Exported functions
 * ---------------------------------------------------------------------------*/

pub fn IDASetNonlinearSolverSensSim(ida_mem: &IDAMem, NLS: &SUNNonlinearSolver) -> i32 {
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
            "IDASetNonlinearSolverSensSim",
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
            "IDASetNonlinearSolverSensSim",
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
            "IDASetNonlinearSolverSensSim",
            file!(),
            MSG_NO_SENSI,
        );
        return IDA_ILL_INPUT;
    }

    /* check that the simultaneous corrector was selected */
    if IDA_mem.borrow().ida_ism != IDA_SIMULTANEOUS {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDASetNonlinearSolverSensSim",
            file!(),
            "Sensitivity solution method is not IDA_SIMULTANEOUS",
        );
        return IDA_ILL_INPUT;
    }

    /* free any existing nonlinear solver */
    let (old_nls, own_nls) = {
        let mut m = IDA_mem.borrow_mut();
        (m.NLSsim.take(), m.ownNLSsim)
    };
    if old_nls.is_some() && own_nls {
        /* C stores the result in retval but never reads it */
        let _ = SUNNonlinSolFree(old_nls);
    }

    {
        let mut m = IDA_mem.borrow_mut();

        /* set SUNNonlinearSolver pointer */
        m.NLSsim = Some(NLS.clone());

        /* Set NLS ownership flag. If this function was called to attach the default
        NLS, IDA will set the flag to SUNTRUE after this function returns. */
        m.ownNLSsim = SUNFALSE;
    }

    let nls = { IDA_mem.borrow().NLSsim.clone() }.expect("NLSsim");

    /* set the nonlinear residual function */
    retval = SUNNonlinSolSetSysFn(&nls, Some(idaNlsResidualSensSim));
    if retval != IDA_SUCCESS {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDASetNonlinearSolverSensSim",
            file!(),
            "Setting nonlinear system function failed",
        );
        return IDA_ILL_INPUT;
    }

    /* set convergence test function */
    retval = SUNNonlinSolSetConvTestFn(
        &nls,
        Some(idaNlsConvTestSensSim),
        Some(Box::new(IDA_mem.clone())),
    );
    if retval != IDA_SUCCESS {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDASetNonlinearSolverSensSim",
            file!(),
            "Setting convergence test function failed",
        );
        return IDA_ILL_INPUT;
    }

    retval = SUNNonlinSolSetGetUpdateNormFn(
        &nls,
        Some(idaNlsGetUpdateNormSensSim),
        Some(Box::new(IDA_mem.clone())),
    );
    if retval != IDA_SUCCESS {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDASetNonlinearSolverSensSim",
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
            "IDASetNonlinearSolverSensSim",
            file!(),
            "Setting maximum number of nonlinear iterations failed",
        );
        return IDA_ILL_INPUT;
    }

    /* create vector wrappers if necessary */
    let simMallocDone = IDA_mem.borrow().simMallocDone;
    if simMallocDone == SUNFALSE {
        let (Ns, sunctx) = {
            let m = IDA_mem.borrow();
            (m.ida_Ns, m.ida_sunctx.clone())
        };

        let ypredictSim = N_VNewEmpty_SensWrapper(Ns + 1, &sunctx);
        let ypredictSim_null = ypredictSim.is_none();
        IDA_mem.borrow_mut().ypredictSim = ypredictSim;
        if ypredictSim_null {
            IDAProcessError(
                Some(IDA_mem),
                IDA_MEM_FAIL,
                line!() as i32,
                "IDASetNonlinearSolverSensSim",
                file!(),
                MSG_MEM_FAIL,
            );
            return IDA_MEM_FAIL;
        }

        let ycorSim = N_VNewEmpty_SensWrapper(Ns + 1, &sunctx);
        let ycorSim_null = ycorSim.is_none();
        IDA_mem.borrow_mut().ycorSim = ycorSim;
        if ycorSim_null {
            /* C leaves the destroyed handle in the mem; the port takes it out */
            let v = IDA_mem.borrow_mut().ypredictSim.take();
            if let Some(v) = v {
                N_VDestroy(v);
            }
            IDAProcessError(
                Some(IDA_mem),
                IDA_MEM_FAIL,
                line!() as i32,
                "IDASetNonlinearSolverSensSim",
                file!(),
                MSG_MEM_FAIL,
            );
            return IDA_MEM_FAIL;
        }

        let ewtSim = N_VNewEmpty_SensWrapper(Ns + 1, &sunctx);
        let ewtSim_null = ewtSim.is_none();
        IDA_mem.borrow_mut().ewtSim = ewtSim;
        if ewtSim_null {
            let v = IDA_mem.borrow_mut().ypredictSim.take();
            if let Some(v) = v {
                N_VDestroy(v);
            }
            let v = IDA_mem.borrow_mut().ycorSim.take();
            if let Some(v) = v {
                N_VDestroy(v);
            }
            IDAProcessError(
                Some(IDA_mem),
                IDA_MEM_FAIL,
                line!() as i32,
                "IDASetNonlinearSolverSensSim",
                file!(),
                MSG_MEM_FAIL,
            );
            return IDA_MEM_FAIL;
        }

        IDA_mem.borrow_mut().simMallocDone = SUNTRUE;
    }

    /* attach vectors to vector wrappers */
    let (ypredictSim, ycorSim, ewtSim, yypredict, ee, ewt, Ns) = {
        let m = IDA_mem.borrow();
        (
            m.ypredictSim.as_ref().expect("ypredictSim").clone(),
            m.ycorSim.as_ref().expect("ycorSim").clone(),
            m.ewtSim.as_ref().expect("ewtSim").clone(),
            m.ida_yypredict.clone(),
            m.ida_ee.clone(),
            m.ida_ewt.clone(),
            m.ida_Ns,
        )
    };
    NV_VEC_SW_set(&ypredictSim, 0, yypredict);
    NV_VEC_SW_set(&ycorSim, 0, ee);
    NV_VEC_SW_set(&ewtSim, 0, ewt);

    for is in 0..Ns {
        let (yySpredict_is, eeS_is, ewtS_is) = {
            let m = IDA_mem.borrow();
            (
                m.ida_yySpredict[is as usize].clone(),
                m.ida_eeS[is as usize].clone(),
                m.ida_ewtS[is as usize].clone(),
            )
        };
        NV_VEC_SW_set(&ypredictSim, is + 1, Some(yySpredict_is));
        NV_VEC_SW_set(&ycorSim, is + 1, Some(eeS_is));
        NV_VEC_SW_set(&ewtSim, is + 1, Some(ewtS_is));
    }

    /* Set the nonlinear system RES function */
    if IDA_mem.borrow().ida_res.is_none() {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDASetNonlinearSolverSensSim",
            file!(),
            "The DAE residual function is NULL",
        );
        return IDA_ILL_INPUT;
    }
    {
        let mut m = IDA_mem.borrow_mut();
        m.nls_res = m.ida_res;
    }

    IDA_SUCCESS
}

/*---------------------------------------------------------------
  IDAGetNonlinearSystemDataSens:

  This routine provides access to the relevant data needed to
  compute the nonlinear system function.

  The C `N_Vector**` out-parameters hand out the internal vector
  ARRAYS; the port fills the caller's `Vec` with `Rc` clones of the
  same vectors (writes through them reach the integrator's storage,
  exactly as in C). `user_data` follows the SWAP convention of
  `IDAGetUserData` — the caller must hand the box back before the
  integrator next invokes a user callback.
  ---------------------------------------------------------------*/
#[allow(clippy::too_many_arguments)]
pub fn IDAGetNonlinearSystemDataSens(
    ida_mem: &IDAMem,
    tcur: &mut sunrealtype,
    yySpred: &mut Vec<N_Vector>,
    ypSpred: &mut Vec<N_Vector>,
    yySn: &mut Vec<N_Vector>,
    ypSn: &mut Vec<N_Vector>,
    cj: &mut sunrealtype,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    /* NULL-mem check: handled by the type system */
    let mut IDA_mem = ida_mem.borrow_mut();

    *tcur = IDA_mem.ida_tn;
    *yySpred = IDA_mem.ida_yySpredict.clone();
    *ypSpred = IDA_mem.ida_ypSpredict.clone();
    *yySn = IDA_mem.ida_yyS.clone();
    *ypSn = IDA_mem.ida_ypS.clone();
    *cj = IDA_mem.ida_cj;
    /* C copies the raw pointer; the box is swapped out instead */
    std::mem::swap(&mut IDA_mem.ida_user_data, user_data);

    IDA_SUCCESS
}

/* -----------------------------------------------------------------------------
 * Private functions
 * ---------------------------------------------------------------------------*/

pub fn idaNlsInitSensSim(IDA_mem: &IDAMem) -> i32 {
    let mut retval: i32;

    let nls = { IDA_mem.borrow().NLSsim.clone() }.expect("NLSsim");

    /* set the linear solver setup wrapper function */
    if IDA_mem.borrow().ida_lsetup.is_some() {
        retval = SUNNonlinSolSetLSetupFn(&nls, Some(idaNlsLSetupSensSim));
    } else {
        retval = SUNNonlinSolSetLSetupFn(&nls, None);
    }

    if retval != IDA_SUCCESS {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "idaNlsInitSensSim",
            file!(),
            "Setting the linear solver setup function failed",
        );
        return IDA_NLS_INIT_FAIL;
    }

    /* set the linear solver solve wrapper function */
    if IDA_mem.borrow().ida_lsolve.is_some() {
        retval = SUNNonlinSolSetLSolveFn(&nls, Some(idaNlsLSolveSensSim));
    } else {
        retval = SUNNonlinSolSetLSolveFn(&nls, None);
    }

    if retval != IDA_SUCCESS {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "idaNlsInitSensSim",
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
            "idaNlsInitSensSim",
            file!(),
            MSG_NLS_INIT_FAIL,
        );
        return IDA_NLS_INIT_FAIL;
    }

    IDA_SUCCESS
}

fn idaNlsLSetupSensSim(
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
                "idaNlsLSetupSensSim",
                file!(),
                MSG_NO_MEM,
            );
            return IDA_MEM_NULL;
        }
    };

    {
        let mut m = IDA_mem.borrow_mut();
        m.ida_nsetups += 1;
        m.ida_forceSetup = SUNFALSE;
    }

    let (lsetup, yy, yp, savres, tempv1, tempv2, tempv3) = {
        let m = IDA_mem.borrow();
        (
            m.ida_lsetup.expect("ida_lsetup"),
            m.ida_yy.clone().expect("ida_yy"),
            m.ida_yp.clone().expect("ida_yp"),
            m.ida_savres.clone().expect("ida_savres"),
            m.ida_tempv1.clone().expect("ida_tempv1"),
            m.ida_tempv2.clone().expect("ida_tempv2"),
            m.ida_tempv3.clone().expect("ida_tempv3"),
        )
    };
    let retval = lsetup(&IDA_mem, &yy, &yp, &savres, &tempv1, &tempv2, &tempv3);

    /* update Jacobian status */
    *jcur = SUNTRUE;

    /* update convergence test constants */
    {
        let mut m = IDA_mem.borrow_mut();
        m.ida_cjold = m.ida_cj;
        m.ida_cjratio = ONE;
        m.ida_ss = TWENTY;
        m.ida_ssS = TWENTY;
        m.ida_delnrm = 0.0; /* SUN_RCONST(0.0) */
    }

    if retval < 0 {
        return IDA_LSETUP_FAIL;
    }
    if retval > 0 {
        return IDA_LSETUP_RECVR;
    }

    IDA_SUCCESS
}

fn idaNlsLSolveSensSim(deltaSim: &N_Vector, ida_mem: &mut Option<Box<dyn Any>>) -> i32 {
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
                "idaNlsLSolveSensSim",
                file!(),
                MSG_NO_MEM,
            );
            return IDA_MEM_NULL;
        }
    };

    /* extract state update vector from the vector wrapper */
    let delta = NV_VEC_SW(deltaSim, 0);

    /* solve the state linear system */
    let (lsolve, ewt, yy, yp, savres) = {
        let m = IDA_mem.borrow();
        (
            m.ida_lsolve.expect("ida_lsolve"),
            m.ida_ewt.clone().expect("ida_ewt"),
            m.ida_yy.clone().expect("ida_yy"),
            m.ida_yp.clone().expect("ida_yp"),
            m.ida_savres.clone().expect("ida_savres"),
        )
    };
    let retval = lsolve(&IDA_mem, &delta, &ewt, &yy, &yp, &savres);

    if retval < 0 {
        return IDA_LSOLVE_FAIL;
    }
    if retval > 0 {
        return IDA_LSOLVE_RECVR;
    }

    /* extract sensitivity deltas from the vector wrapper */
    let deltaS = sens_vecs_sw(deltaSim);

    /* solve the sensitivity linear systems */
    let Ns = IDA_mem.borrow().ida_Ns;
    for is in 0..Ns as usize {
        let (lsolve, ewtS_is, yy, yp, savres) = {
            let m = IDA_mem.borrow();
            (
                m.ida_lsolve.expect("ida_lsolve"),
                m.ida_ewtS[is].clone(),
                m.ida_yy.clone().expect("ida_yy"),
                m.ida_yp.clone().expect("ida_yp"),
                m.ida_savres.clone().expect("ida_savres"),
            )
        };
        let retval = lsolve(&IDA_mem, &deltaS[is], &ewtS_is, &yy, &yp, &savres);

        if retval < 0 {
            return IDA_LSOLVE_FAIL;
        }
        if retval > 0 {
            return IDA_LSOLVE_RECVR;
        }
    }

    IDA_SUCCESS
}

fn idaNlsResidualSensSim(
    ycorSim: &N_Vector,
    resSim: &N_Vector,
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
                "idaNlsResidualSensSim",
                file!(),
                MSG_NO_MEM,
            );
            return IDA_MEM_NULL;
        }
    };

    /* extract state and residual vectors from the vector wrapper */
    let ycor = NV_VEC_SW(ycorSim, 0);
    let res = NV_VEC_SW(resSim, 0);

    let (yypredict, yppredict, yy, yp, cj, tn, nls_res, savres) = {
        let m = IDA_mem.borrow();
        (
            m.ida_yypredict.clone().expect("ida_yypredict"),
            m.ida_yppredict.clone().expect("ida_yppredict"),
            m.ida_yy.clone().expect("ida_yy"),
            m.ida_yp.clone().expect("ida_yp"),
            m.ida_cj,
            m.ida_tn,
            m.nls_res.expect("nls_res"),
            m.ida_savres.clone().expect("ida_savres"),
        )
    };

    /* update yy and yp based on the current correction */
    N_VLinearSum(ONE, &yypredict, ONE, &ycor, &yy);
    N_VLinearSum(ONE, &yppredict, cj, &ycor, &yp);

    /* evaluate residual */
    let mut user_data = IDA_mem.borrow_mut().ida_user_data.take();
    let retval = nls_res(tn, &yy, &yp, &res, &mut user_data);
    {
        let mut m = IDA_mem.borrow_mut();
        m.ida_user_data = user_data;

        /* increment the number of residual evaluations */
        m.ida_nre += 1;
    }

    /* save a copy of the residual vector in savres */
    N_VScale(ONE, &res, &savres);

    if retval < 0 {
        return IDA_RES_FAIL;
    }
    if retval > 0 {
        return IDA_RES_RECVR;
    }

    /* extract sensitivity and residual vectors from the vector wrapper */
    let ycorS = sens_vecs_sw(ycorSim);
    let resS = sens_vecs_sw(resSim);

    /* update yS and ypS based on the current correction */
    let (Ns, yySpredict, ypSpredict, yyS, ypS) = {
        let m = IDA_mem.borrow();
        (
            m.ida_Ns,
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
    let (resS_fn, tmpS1, tmpS2, tmpS3) = {
        let m = IDA_mem.borrow();
        (
            m.ida_resS.expect("ida_resS"),
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
        &res,
        &yyS,
        &ypS,
        &resS,
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

fn idaNlsGetUpdateNormSensSim(
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

    *delnrm = IDA_mem.borrow().ida_delnrm;
    SUN_SUCCESS
}

fn idaNlsConvTestSensSim(
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
                "idaNlsConvTestSensSim",
                file!(),
                MSG_NO_MEM,
            );
            return IDA_MEM_NULL;
        }
    };

    /* compute the norm of the correction */
    let delnrm = N_VWrmsNorm(del, ewt);
    IDA_mem.borrow_mut().ida_delnrm = delnrm;

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
            mm.ida_oldnrm = mm.ida_delnrm;
            mm.ida_toldel
        };
        if delnrm <= PT0001 * toldel {
            return SUN_SUCCESS;
        }
    } else {
        let oldnrm = IDA_mem.borrow().ida_oldnrm;
        let rate = SUNRpowerR(delnrm / oldnrm, ONE / (m as sunrealtype));
        if rate > RATEMAX {
            return SUN_NLS_CONV_RECVR;
        }
        IDA_mem.borrow_mut().ida_ss = rate / (ONE - rate);
    }

    if IDA_mem.borrow().ida_ss * delnrm <= tol {
        return SUN_SUCCESS;
    }

    /* not yet converged */
    SUN_NLS_CONTINUE
}
