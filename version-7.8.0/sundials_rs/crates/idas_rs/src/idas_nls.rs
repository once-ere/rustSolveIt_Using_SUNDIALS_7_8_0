//! Port of `src/idas/idas_nls.c`: the IDAS nonlinear solver interface for
//! the state (non-sensitivity) nonlinear system.
//!
//! The C `void* ida_mem` handed to the SUNNonlinearSolver as the
//! convergence-test / update-norm data maps to a boxed `IDAMem` clone
//! inside an `Option<Box<dyn Any>>` token; the same token shape carries
//! the integrator mem through `SUNNonlinSolSolve` into the Sys / LSetup
//! / LSolve wrappers. Each callback downcasts the token back to the
//! handle and uses granular borrows (never holding a borrow across a
//! user callback, an `N_Vector` op on a user-visible vector, a
//! linear-solver call, or `IDAProcessError`).
//!
//! `IDASetNonlinearSolver` takes the solver **by reference**: the C
//! function stores the caller's pointer and leaves ownership with the
//! caller on every failure path, so the Rust port clones the handle only
//! once the input checks have passed.
//!
//! `PT0001`, `ONE`, `TWENTY`, `MAXIT` and `RATEMAX` come from
//! `idas_impl` (the shared fragment-protocol constant block) and are NOT
//! redefined here.
//!
//! IDAS deltas vs. the IDA twin (`ida_nls.c`), preserved verbatim:
//! `idaNlsLSetup` additionally clears `ida_forceSetup` *before* calling
//! `ida_lsetup`, and additionally resets the staggered-sensitivity
//! convergence constant `ida_ssS` to `TWENTY`.

use std::any::Any;

use crate::idas_impl::*;
use sundials_core::sundials_errors::{SUN_ERR_ARG_CORRUPT, SUN_SUCCESS};
use sundials_core::sundials_math::SUNRpowerR;
use sundials_core::sundials_nonlinearsolver::*;
use sundials_core::sundials_nvector::{N_VLinearSum, N_VScale, N_VWrmsNorm, N_Vector};
use sundials_core::sundials_types::*;

/* -----------------------------------------------------------------------------
 * Exported functions
 * ---------------------------------------------------------------------------*/

pub fn IDASetNonlinearSolver(ida_mem: &IDAMem, NLS: &SUNNonlinearSolver) -> i32 {
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
            "IDASetNonlinearSolver",
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
            "IDASetNonlinearSolver",
            file!(),
            "NLS type must be SUNNONLINEARSOLVER_ROOTFIND",
        );
        return IDA_ILL_INPUT;
    }

    /* free any existing nonlinear solver */
    let (old_nls, own_nls) = {
        let mut m = IDA_mem.borrow_mut();
        (m.NLS.take(), m.ownNLS)
    };
    if old_nls.is_some() && own_nls {
        /* C stores the result in retval but never reads it */
        let _ = SUNNonlinSolFree(old_nls);
    }

    {
        let mut m = IDA_mem.borrow_mut();

        /* set SUNNonlinearSolver pointer */
        m.NLS = Some(NLS.clone());

        /* Set NLS ownership flag. If this function was called to attach the default
        NLS, IDA will set the flag to SUNTRUE after this function returns. */
        m.ownNLS = SUNFALSE;
    }

    let nls = { IDA_mem.borrow().NLS.clone() }.expect("NLS");

    /* set the nonlinear residual function */
    retval = SUNNonlinSolSetSysFn(&nls, Some(idaNlsResidual));
    if retval != IDA_SUCCESS {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDASetNonlinearSolver",
            file!(),
            "Setting nonlinear system function failed",
        );
        return IDA_ILL_INPUT;
    }

    /* set convergence test function */
    retval = SUNNonlinSolSetConvTestFn(&nls, Some(idaNlsConvTest), Some(Box::new(IDA_mem.clone())));
    if retval != IDA_SUCCESS {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDASetNonlinearSolver",
            file!(),
            "Setting convergence test function failed",
        );
        return IDA_ILL_INPUT;
    }

    retval = SUNNonlinSolSetGetUpdateNormFn(
        &nls,
        Some(idaNlsGetUpdateNorm),
        Some(Box::new(IDA_mem.clone())),
    );
    if retval != IDA_SUCCESS {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDASetNonlinearSolver",
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
            "IDASetNonlinearSolver",
            file!(),
            "Setting maximum number of nonlinear iterations failed",
        );
        return IDA_ILL_INPUT;
    }

    /* Set the nonlinear system RES function */
    if IDA_mem.borrow().ida_res.is_none() {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDASetNonlinearSolver",
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
  IDASetNlsResFn:

  This routine sets an alternative user-supplied DAE residual
  function to use in the evaluation of nonlinear system functions.
  ---------------------------------------------------------------*/
pub fn IDASetNlsResFn(ida_mem: &IDAMem, res: Option<IDAResFn>) -> i32 {
    /* NULL-mem check: handled by the type system */
    let mut IDA_mem = ida_mem.borrow_mut();

    if let Some(res) = res {
        IDA_mem.nls_res = Some(res);
    } else {
        IDA_mem.nls_res = IDA_mem.ida_res;
    }

    IDA_SUCCESS
}

/*---------------------------------------------------------------
  IDAGetNonlinearSystemData:

  This routine provides access to the relevant data needed to
  compute the nonlinear system function.

  C hands out the raw `user_data` pointer; boxes cannot alias, so
  the token is SWAPPED with the caller's out-param (the locked
  `void*`-getter convention) — a caller that uses it must hand it
  back (via IDASetUserData or a second swap) before the integrator
  next invokes a user callback.
  ---------------------------------------------------------------*/
#[allow(clippy::too_many_arguments)]
pub fn IDAGetNonlinearSystemData(
    ida_mem: &IDAMem,
    tcur: &mut sunrealtype,
    yypred: &mut Option<N_Vector>,
    yppred: &mut Option<N_Vector>,
    yyn: &mut Option<N_Vector>,
    ypn: &mut Option<N_Vector>,
    res: &mut Option<N_Vector>,
    cj: &mut sunrealtype,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    /* NULL-mem check: handled by the type system */
    let mut IDA_mem = ida_mem.borrow_mut();

    *tcur = IDA_mem.ida_tn;
    *yypred = IDA_mem.ida_yypredict.clone();
    *yppred = IDA_mem.ida_yppredict.clone();
    *yyn = IDA_mem.ida_yy.clone();
    *ypn = IDA_mem.ida_yp.clone();
    *res = IDA_mem.ida_savres.clone();
    *cj = IDA_mem.ida_cj;
    /* C copies the raw pointer; the box is swapped out instead — the
    caller must hand it back before the next user-callback invocation. */
    std::mem::swap(&mut IDA_mem.ida_user_data, user_data);

    IDA_SUCCESS
}

/* -----------------------------------------------------------------------------
 * Private functions
 * ---------------------------------------------------------------------------*/

pub fn idaNlsInit(IDA_mem: &IDAMem) -> i32 {
    let mut retval: i32;

    let nls = { IDA_mem.borrow().NLS.clone() }.expect("NLS");

    /* set the linear solver setup wrapper function */
    if IDA_mem.borrow().ida_lsetup.is_some() {
        retval = SUNNonlinSolSetLSetupFn(&nls, Some(idaNlsLSetup));
    } else {
        retval = SUNNonlinSolSetLSetupFn(&nls, None);
    }

    if retval != IDA_SUCCESS {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "idaNlsInit",
            file!(),
            "Setting the linear solver setup function failed",
        );
        return IDA_NLS_INIT_FAIL;
    }

    /* set the linear solver solve wrapper function */
    if IDA_mem.borrow().ida_lsolve.is_some() {
        retval = SUNNonlinSolSetLSolveFn(&nls, Some(idaNlsLSolve));
    } else {
        retval = SUNNonlinSolSetLSolveFn(&nls, None);
    }

    if retval != IDA_SUCCESS {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "idaNlsInit",
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
            "idaNlsInit",
            file!(),
            MSG_NLS_INIT_FAIL,
        );
        return IDA_NLS_INIT_FAIL;
    }

    IDA_SUCCESS
}

fn idaNlsLSetup(
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
                "idaNlsLSetup",
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

fn idaNlsLSolve(delta: &N_Vector, ida_mem: &mut Option<Box<dyn Any>>) -> i32 {
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
                "idaNlsLSolve",
                file!(),
                MSG_NO_MEM,
            );
            return IDA_MEM_NULL;
        }
    };

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
    let retval = lsolve(&IDA_mem, delta, &ewt, &yy, &yp, &savres);

    if retval < 0 {
        return IDA_LSOLVE_FAIL;
    }
    if retval > 0 {
        return IDA_LSOLVE_RECVR;
    }

    IDA_SUCCESS
}

fn idaNlsResidual(ycor: &N_Vector, res: &N_Vector, ida_mem: &mut Option<Box<dyn Any>>) -> i32 {
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
                "idaNlsResidual",
                file!(),
                MSG_NO_MEM,
            );
            return IDA_MEM_NULL;
        }
    };

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
    N_VLinearSum(ONE, &yypredict, ONE, ycor, &yy);
    N_VLinearSum(ONE, &yppredict, cj, ycor, &yp);

    /* evaluate residual */
    let mut user_data = IDA_mem.borrow_mut().ida_user_data.take();
    let retval = nls_res(tn, &yy, &yp, res, &mut user_data);
    {
        let mut m = IDA_mem.borrow_mut();
        m.ida_user_data = user_data;

        /* increment the number of residual evaluations */
        m.ida_nre += 1;
    }

    /* save a copy of the residual vector in savres */
    N_VScale(ONE, res, &savres);

    if retval < 0 {
        return IDA_RES_FAIL;
    }
    if retval > 0 {
        return IDA_RES_RECVR;
    }

    IDA_SUCCESS
}

fn idaNlsGetUpdateNorm(delnrm: &mut sunrealtype, ida_mem: &mut Option<Box<dyn Any>>) -> SUNErrCode {
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

fn idaNlsConvTest(
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
                "idaNlsConvTest",
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
