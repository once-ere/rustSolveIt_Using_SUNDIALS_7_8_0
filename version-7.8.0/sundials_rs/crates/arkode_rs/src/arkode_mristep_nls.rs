//! Port of `src/arkode/arkode_mristep_nls.c`: the interface between MRIStep
//! and the `SUNNonlinearSolver` object.
//!
//! The C `void* arkode_mem` handed to the nonlinear solver as the integrator
//! mem / ctest / norm / getter data maps to a boxed `ARKodeMem` clone inside
//! an `Option<Box<dyn Any>>` token; each callback downcasts the token back to
//! the handle and uses granular borrows (never holding a borrow of the mem or
//! of the MRIStep record across a user callback, an `N_Vector` op on a
//! user-visible vector, or a linear-solver call).
//!
//! `SUNDIALS_LOGGING_LEVEL=2`: every `SUNLogInfo` / `SUNLogExtraDebugVec`
//! statement compiles away and is omitted at translation time.
//!
//! The `jcur` seam: `step_mem.jcur` is the shared `ARKJcurPtr` cell, so the
//! address C hands to `lsetup` is modelled by a clone of that cell —
//! `arkLsPSetup` writing through it is observed by `mriStep_NlsLSetup` after
//! the call, exactly as in C.

use std::any::Any;

use crate::arkode_impl::*;
use crate::arkode_mristep::*;

use sundials_core::sundials_errors::{SUN_ERR_ARG_CORRUPT, SUN_SUCCESS};
use sundials_core::sundials_math::{SUNRabs, SUNMAX, SUNMIN};
use sundials_core::sundials_nonlinearsolver::*;
use sundials_core::sundials_nvector::{
    N_VConst, N_VLinearCombination, N_VLinearSum, N_VWrmsNorm, N_Vector,
};
use sundials_core::sundials_types::*;

/// Downcast the nonlinear-solver mem token back to the ARKODE handle
/// (C `(ARKodeMem) arkode_mem`); `None` = C `arkode_mem == NULL`.
fn nls_ark_mem(arkode_mem: &mut Option<Box<dyn Any>>) -> Option<ARKodeMem> {
    arkode_mem
        .as_mut()
        .and_then(|b| b.downcast_ref::<ARKodeMem>())
        .cloned()
}

/*===============================================================
  Interface routines supplied to ARKODE
  ===============================================================*/

/*---------------------------------------------------------------
  mriStep_SetNonlinearSolver:

  This routine attaches a SUNNonlinearSolver object to the MRIStep
  module.
  ---------------------------------------------------------------*/
pub fn mriStep_SetNonlinearSolver(ark_mem: &ARKodeMem, NLS: &SUNNonlinearSolver) -> i32 {
    let mut retval: i32;

    /* access ARKodeMRIStepMem structure */
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "mriStep_SetNonlinearSolver",
            file!(),
            MSG_MRISTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    /* Return immediately if NLS input is NULL: handled by the type system */

    /* check for required nonlinear solver functions */
    {
        let ops = NLS.ops.borrow();
        if ops.gettype.is_none()
            || ops.solve.is_none()
            || (ops.setsysfn.is_none() && ops.setsysfns.is_none())
        {
            drop(ops);
            arkProcessError(
                Some(ark_mem),
                ARK_ILL_INPUT,
                line!() as i32,
                "mriStep_SetNonlinearSolver",
                file!(),
                "NLS does not support required operations",
            );
            return ARK_ILL_INPUT;
        }
    }

    /* free any existing nonlinear solver */
    let (old_nls, own_nls) = {
        let mut step_mem = mriStep_mem_mut(ark_mem);
        (step_mem.NLS.take(), step_mem.ownNLS)
    };
    if old_nls.is_some() && own_nls {
        /* C stores the result in retval but never reads it */
        let _ = SUNNonlinSolFree(old_nls);
    }

    /* set SUNNonlinearSolver pointer */
    {
        let mut step_mem = mriStep_mem_mut(ark_mem);
        step_mem.NLS = Some(NLS.clone());
        step_mem.ownNLS = SUNFALSE;
    }

    /* set the nonlinear residual/fixed-point function, based on solver type */
    let nls = { mriStep_mem_mut(ark_mem).NLS.clone() }.expect("NLS");
    if SUNNonlinSolGetType(&nls) == SUNNONLINEARSOLVER_ROOTFIND {
        retval = SUNNonlinSolSetSysFn(&nls, Some(mriStep_NlsResidual));
    } else if SUNNonlinSolGetType(&nls) == SUNNONLINEARSOLVER_FIXEDPOINT {
        retval = SUNNonlinSolSetSysFn(&nls, Some(mriStep_NlsFPFunction));
    } else if SUNNonlinSolGetType(&nls) == SUNNONLINEARSOLVER_HYBRID {
        retval = SUNNonlinSolSetSysFns(
            &nls,
            Some(mriStep_NlsResidual),
            Some(mriStep_NlsFPFunction),
        );
    } else {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "mriStep_SetNonlinearSolver",
            file!(),
            "Invalid nonlinear solver type",
        );
        return ARK_ILL_INPUT;
    }
    if retval != ARK_SUCCESS {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "mriStep_SetNonlinearSolver",
            file!(),
            "Setting nonlinear system function failed",
        );
        return ARK_ILL_INPUT;
    }

    /* set convergence test function */
    retval = SUNNonlinSolSetConvTestFn(
        &nls,
        Some(mriStep_NlsConvTest),
        Some(Box::new(ark_mem.clone())),
    );
    if retval != ARK_SUCCESS {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "mriStep_SetNonlinearSolver",
            file!(),
            "Setting convergence test function failed",
        );
        return ARK_ILL_INPUT;
    }

    retval = SUNNonlinSolSetNormFn(&nls, Some(mriStep_NlsNorm), Some(Box::new(ark_mem.clone())));
    if retval != ARK_SUCCESS {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "mriStep_SetNonlinearSolver",
            file!(),
            "Setting convergence-test norm function failed",
        );
        return ARK_ILL_INPUT;
    }

    retval = SUNNonlinSolSetGetUpdateNormFn(
        &nls,
        Some(mriStep_NlsGetUpdateNorm),
        Some(Box::new(ark_mem.clone())),
    );
    if retval != ARK_SUCCESS {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "mriStep_SetNonlinearSolver",
            file!(),
            "Setting update-norm getter failed",
        );
        return ARK_ILL_INPUT;
    }

    retval = SUNNonlinSolSetGetConvRateFn(
        &nls,
        Some(mriStep_NlsGetConvRate),
        Some(Box::new(ark_mem.clone())),
    );
    if retval != ARK_SUCCESS {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "mriStep_SetNonlinearSolver",
            file!(),
            "Setting convergence-rate getter failed",
        );
        return ARK_ILL_INPUT;
    }

    /* set default nonlinear iterations */
    let maxcor = mriStep_mem_mut(ark_mem).maxcor;
    retval = SUNNonlinSolSetMaxIters(&nls, maxcor);
    if retval != ARK_SUCCESS {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "mriStep_SetNonlinearSolver",
            file!(),
            "Setting maximum number of nonlinear iterations failed",
        );
        return ARK_ILL_INPUT;
    }

    /* set the nonlinear system RHS function */
    mriStep_mem_mut(ark_mem).nls_fsi = None;

    let (implicit_rhs, fsi) = {
        let step_mem = mriStep_mem_mut(ark_mem);
        (step_mem.implicit_rhs, step_mem.fsi)
    };
    if implicit_rhs {
        if fsi.is_none() {
            arkProcessError(
                Some(ark_mem),
                ARK_ILL_INPUT,
                line!() as i32,
                "mriStep_SetNonlinearSolver",
                file!(),
                "The implicit slow ODE RHS function is NULL",
            );
            return ARK_ILL_INPUT;
        }
        mriStep_mem_mut(ark_mem).nls_fsi = fsi;
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  mriStep_SetNlsRhsFn:

  This routine sets an alternative user-supplied slow ODE
  right-hand side function to use in the evaluation of nonlinear
  system functions.
  ---------------------------------------------------------------*/
pub fn mriStep_SetNlsRhsFn(ark_mem: &ARKodeMem, nls_fsi: Option<ARKRhsFn>) -> i32 {
    /* access ARKodeMRIStepMem structure */
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "mriStep_SetNlsRhsFn",
            file!(),
            MSG_MRISTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    {
        let mut step_mem = mriStep_mem_mut(ark_mem);
        if nls_fsi.is_some() {
            step_mem.nls_fsi = nls_fsi;
        } else {
            step_mem.nls_fsi = step_mem.fsi;
        }
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  mriStep_GetNonlinearSystemData:

  This routine provides access to the relevant data needed to
  compute the nonlinear system function.

  C hands out the raw `user_data` pointer; boxes cannot alias, so
  the token is SWAPPED with the caller's out-param (accepted
  deviation class 6) — the caller must hand it back before the
  integrator next invokes a user callback.
  ---------------------------------------------------------------*/
#[allow(clippy::too_many_arguments)]
pub fn mriStep_GetNonlinearSystemData(
    ark_mem: &ARKodeMem,
    tcur: &mut sunrealtype,
    zpred: &mut Option<N_Vector>,
    z: &mut Option<N_Vector>,
    F: &mut Option<N_Vector>,
    gamma: &mut sunrealtype,
    sdata: &mut Option<N_Vector>,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    /* access ARKodeMRIStepMem structure */
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "mriStep_GetNonlinearSystemData",
            file!(),
            MSG_MRISTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    let (zpred_v, Fsi_v, gamma_v, sdata_v) = {
        let step_mem = mriStep_mem_mut(ark_mem);
        let idx = step_mem.stage_map[step_mem.istage as usize] as usize;
        (
            step_mem.zpred.clone(),
            step_mem.Fsi[idx].clone(),
            step_mem.gamma,
            step_mem.sdata.clone(),
        )
    };

    let mut m = ark_mem.borrow_mut();
    *tcur = m.tcur;
    *zpred = zpred_v;
    *z = m.ycur.clone();
    *F = Some(Fsi_v);
    *gamma = gamma_v;
    *sdata = sdata_v;
    std::mem::swap(&mut m.user_data, user_data);

    ARK_SUCCESS
}

/*===============================================================
  Utility routines called by MRIStep
  ===============================================================*/

/*---------------------------------------------------------------
  mriStep_NlsInit:

  This routine attaches the linear solver 'setup' and 'solve'
  routines to the nonlinear solver object, and then initializes
  the nonlinear solver object itself.  This should only be
  called at the start of a simulation, after a re-init, or after
  a re-size.
  ---------------------------------------------------------------*/
pub fn mriStep_NlsInit(ark_mem: &ARKodeMem) -> i32 {
    let mut retval: i32;

    /* access ARKodeMRIStepMem structure */
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "mriStep_NlsInit",
            file!(),
            MSG_MRISTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    /* reset counters */
    {
        let mut step_mem = mriStep_mem_mut(ark_mem);
        step_mem.nls_iters = 0;
        step_mem.nls_fails = 0;
    }

    let nls = { mriStep_mem_mut(ark_mem).NLS.clone() }.expect("NLS");

    /* set the linear solver setup wrapper function */
    let has_lsetup = mriStep_mem_mut(ark_mem).lsetup.is_some();
    if has_lsetup {
        retval = SUNNonlinSolSetLSetupFn(&nls, Some(mriStep_NlsLSetup));
    } else {
        retval = SUNNonlinSolSetLSetupFn(&nls, None);
    }
    if retval != ARK_SUCCESS {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "mriStep_NlsInit",
            file!(),
            "Setting the linear solver setup function failed",
        );
        return ARK_NLS_INIT_FAIL;
    }

    /* set the linear solver solve wrapper function */
    let has_lsolve = mriStep_mem_mut(ark_mem).lsolve.is_some();
    if has_lsolve {
        retval = SUNNonlinSolSetLSolveFn(&nls, Some(mriStep_NlsLSolve));
    } else {
        retval = SUNNonlinSolSetLSolveFn(&nls, None);
    }
    if retval != ARK_SUCCESS {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "mriStep_NlsInit",
            file!(),
            "Setting linear solver solve function failed",
        );
        return ARK_NLS_INIT_FAIL;
    }

    /* initialize nonlinear solver */
    retval = SUNNonlinSolInitialize(&nls);
    if retval != ARK_SUCCESS {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "mriStep_NlsInit",
            file!(),
            MSG_NLS_INIT_FAIL,
        );
        return ARK_NLS_INIT_FAIL;
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  mriStep_Nls

  This routine attempts to solve the nonlinear system associated
  with a single solve-decoupled implicit stage. It calls the
  supplied SUNNonlinearSolver object to perform the solve.

  Upon entry, the predicted solution is held in step_mem->zpred,
  which is never changed throughout this routine.  If an initial
  attempt at solving the nonlinear system fails (e.g. due to a
  stale Jacobian), this allows for new attempts at the solution.

  Upon a successful solve, the solution is held in ark_mem->ycur.
  ---------------------------------------------------------------*/
pub fn mriStep_Nls(ark_mem: &ARKodeMem, nflag: i32) -> i32 {
    let callLSetup: sunbooleantype;
    let mut nls_iters_inc: i64 = 0;
    let mut nls_fails_inc: i64 = 0;

    /* access ARKodeMRIStepMem structure */
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "mriStep_Nls",
            file!(),
            MSG_MRISTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    /* If a linear solver 'setup' is supplied, set various flags for
       determining whether it should be called */
    let has_lsetup = mriStep_mem_mut(ark_mem).lsetup.is_some();
    if has_lsetup {
        /* Set interface 'convfail' flag for use inside lsetup */
        {
            let mut step_mem = mriStep_mem_mut(ark_mem);
            if step_mem.linear {
                step_mem.convfail = if nflag == FIRST_CALL {
                    ARK_NO_FAILURES
                } else {
                    ARK_FAIL_OTHER
                };
            } else {
                step_mem.convfail = if (nflag == FIRST_CALL) || (nflag == PREV_ERR_FAIL) {
                    ARK_NO_FAILURES
                } else {
                    ARK_FAIL_OTHER
                };
            }
        }

        /* Decide whether to recommend call to lsetup within nonlinear solver */
        let (firststage, nst) = {
            let m = ark_mem.borrow();
            (m.firststage, m.nst)
        };
        let step_mem = mriStep_mem_mut(ark_mem);
        let mut call_lsetup = firststage
            || (step_mem.msbp < 0)
            || (SUNRabs(step_mem.gamrat - ONE) > step_mem.dgmax);
        if step_mem.linear {
            /* linearly-implicit problem */
            call_lsetup = call_lsetup || step_mem.linear_timedep;
        } else {
            /* nonlinearly-implicit problem */
            call_lsetup = call_lsetup
                || (nflag == PREV_CONV_FAIL)
                || (nflag == PREV_ERR_FAIL)
                || (nst >= step_mem.nstlp + (step_mem.msbp.abs() as i64));
        }
        callLSetup = call_lsetup;
    } else {
        mriStep_mem_mut(ark_mem).crate_ = ONE;
        callLSetup = SUNFALSE;
    }

    /* set a zero guess for correction */
    let (zpred, zcor, nlscoef) = {
        let step_mem = mriStep_mem_mut(ark_mem);
        (
            step_mem.zpred.clone().expect("zpred"),
            step_mem.zcor.clone().expect("zcor"),
            step_mem.nlscoef,
        )
    };
    N_VConst(ZERO, &zcor);

    /* Reset the stored residual norm (for iterative linear solvers) */
    mriStep_mem_mut(ark_mem).eRNrm = 0.1 * nlscoef;

    /* solve the nonlinear system for the actual correction. The C `void*`
    integrator mem handed to the nonlinear solver maps to a boxed handle
    clone (the token shape the callbacks below downcast). */
    let nls = { mriStep_mem_mut(ark_mem).NLS.clone() }.expect("NLS");
    let ewt = ark_mem.borrow().ewt.clone().expect("ewt");
    let mut nls_mem: Option<Box<dyn Any>> = Some(Box::new(ark_mem.clone()));
    let retval = SUNNonlinSolSolve(&nls, &zpred, &zcor, &ewt, nlscoef, callLSetup, &mut nls_mem);

    /* increment counters */
    let _ = SUNNonlinSolGetNumIters(&nls, &mut nls_iters_inc);
    mriStep_mem_mut(ark_mem).nls_iters += nls_iters_inc;

    let _ = SUNNonlinSolGetNumConvFails(&nls, &mut nls_fails_inc);
    mriStep_mem_mut(ark_mem).nls_fails += nls_fails_inc;

    /* successful solve -- reset the jcur flag and apply correction */
    if retval == SUN_SUCCESS {
        mriStep_mem_mut(ark_mem).jcur.set(SUNFALSE);
        let ycur = ark_mem.borrow().ycur.clone().expect("ycur");
        N_VLinearSum(ONE, &zcor, ONE, &zpred, &ycur);

        return ARK_SUCCESS;
    }

    /* check for recoverable failure, return ARKODE::CONV_FAIL */
    if retval == SUN_NLS_CONV_RECVR {
        return CONV_FAIL;
    }

    retval
}

/*===============================================================
  Interface routines supplied to the SUNNonlinearSolver module
  ===============================================================*/

/*---------------------------------------------------------------
  mriStep_NlsLSetup:

  This routine wraps the ARKODE linear solver interface 'setup'
  routine for use by the nonlinear solver object.
  ---------------------------------------------------------------*/
pub fn mriStep_NlsLSetup(
    jbad: sunbooleantype,
    jcur: &mut sunbooleantype,
    arkode_mem: &mut Option<Box<dyn Any>>,
) -> i32 {
    /* access ARKodeMem and ARKodeMRIStepMem structures */
    let ark_mem = match nls_ark_mem(arkode_mem) {
        Some(ark_mem) => ark_mem,
        None => {
            arkProcessError(
                None,
                ARK_MEM_NULL,
                line!() as i32,
                "mriStep_NlsLSetup",
                file!(),
                MSG_ARK_NO_MEM,
            );
            return ARK_MEM_NULL;
        }
    };
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(&ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "mriStep_NlsLSetup",
            file!(),
            MSG_MRISTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    /* update convfail based on jbad flag */
    if jbad {
        mriStep_mem_mut(&ark_mem).convfail = ARK_FAIL_BAD_J;
    }

    /* Use ARKODE's tempv1, tempv2 and tempv3 as
       temporary vectors for the linear solver setup routine */
    let (lsetup, convfail, Fsi_v, jcur_cell) = {
        let mut step_mem = mriStep_mem_mut(&ark_mem);
        step_mem.nsetups += 1;
        let idx = step_mem.stage_map[step_mem.istage as usize] as usize;
        (
            step_mem.lsetup.expect("lsetup"),
            step_mem.convfail,
            step_mem.Fsi[idx].clone(),
            step_mem.jcur.clone(),
        )
    };
    let (tcur, ycur, tempv1, tempv2, tempv3) = {
        let m = ark_mem.borrow();
        (
            m.tcur,
            m.ycur.clone().expect("ycur"),
            m.tempv1.clone().expect("tempv1"),
            m.tempv2.clone().expect("tempv2"),
            m.tempv3.clone().expect("tempv3"),
        )
    };
    let retval = lsetup(
        &ark_mem,
        convfail,
        tcur,
        &ycur,
        &Fsi_v,
        &*jcur_cell,
        &tempv1,
        &tempv2,
        &tempv3,
    );

    /* update Jacobian status */
    *jcur = jcur_cell.get();

    /* update flags and 'gamma' values for last lsetup call */
    ark_mem.borrow_mut().firststage = SUNFALSE;
    let nst = ark_mem.borrow().nst;
    {
        let mut step_mem = mriStep_mem_mut(&ark_mem);
        step_mem.crate_ = ONE;
        step_mem.gamrat = ONE;
        step_mem.gammap = step_mem.gamma;
        step_mem.nstlp = nst;
    }

    if retval < 0 {
        return ARK_LSETUP_FAIL;
    }
    if retval > 0 {
        return CONV_FAIL;
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  mriStep_NlsLSolve:

  This routine wraps the ARKODE linear solver interface 'solve'
  routine for use by the nonlinear solver object.
  ---------------------------------------------------------------*/
pub fn mriStep_NlsLSolve(b: &N_Vector, arkode_mem: &mut Option<Box<dyn Any>>) -> i32 {
    /* access ARKodeMem and ARKodeMRIStepMem structures */
    let ark_mem = match nls_ark_mem(arkode_mem) {
        Some(ark_mem) => ark_mem,
        None => {
            arkProcessError(
                None,
                ARK_MEM_NULL,
                line!() as i32,
                "mriStep_NlsLSolve",
                file!(),
                MSG_ARK_NO_MEM,
            );
            return ARK_MEM_NULL;
        }
    };
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(&ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "mriStep_NlsLSolve",
            file!(),
            MSG_MRISTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    /* retrieve nonlinear solver iteration from module */
    let nls = { mriStep_mem_mut(&ark_mem).NLS.clone() }.expect("NLS");
    let mut nonlin_iter: i32 = 0;
    let retval = SUNNonlinSolGetCurIter(&nls, &mut nonlin_iter);
    if retval != SUN_SUCCESS {
        return ARK_NLS_OP_ERR;
    }

    /* call linear solver interface, and handle return value */
    let (lsolve, Fsi_v, eRNrm) = {
        let step_mem = mriStep_mem_mut(&ark_mem);
        let idx = step_mem.stage_map[step_mem.istage as usize] as usize;
        (
            step_mem.lsolve.expect("lsolve"),
            step_mem.Fsi[idx].clone(),
            step_mem.eRNrm,
        )
    };
    let (tcur, ycur) = {
        let m = ark_mem.borrow();
        (m.tcur, m.ycur.clone().expect("ycur"))
    };
    let retval = lsolve(&ark_mem, b, tcur, &ycur, &Fsi_v, eRNrm, nonlin_iter);

    if retval < 0 {
        return ARK_LSOLVE_FAIL;
    }
    if retval > 0 {
        return CONV_FAIL;
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  mriStep_NlsResidual:

  This routine evaluates the nonlinear residual for this
  solve-decoupled implicit MRI stage.  It assumes that any data
  from previous time steps/stages is contained in step_mem, and
  merely combines this old data with the current implicit ODE
  RHS vector to compute the nonlinear residual r.

  At the ith stage, we compute the residual vector:
     r = zc - gamma*Fsi(z) - sdata
  where the current stage solution is z = zp + zc,
     gamma = h*A(i,i),
     zc is stored in the input, zcor, and
     sdata is the old solution/stage data stored in step_mem->sdata.
  Hence we really just compute:
     z = zp + zc (stored in ark_mem->ycur)
     Fsi(z) (stored step_mem->Fsi[step_mem->istage])
     r = zc - gamma*Fsi(z) - step_mem->sdata
  ---------------------------------------------------------------*/
pub fn mriStep_NlsResidual(
    zcor: &N_Vector,
    r: &N_Vector,
    arkode_mem: &mut Option<Box<dyn Any>>,
) -> i32 {
    /* access ARKodeMem and ARKodeMRIStepMem structures */
    let ark_mem = match nls_ark_mem(arkode_mem) {
        Some(ark_mem) => ark_mem,
        None => {
            arkProcessError(
                None,
                ARK_MEM_NULL,
                line!() as i32,
                "mriStep_NlsResidual",
                file!(),
                MSG_ARK_NO_MEM,
            );
            return ARK_MEM_NULL;
        }
    };
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(&ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "mriStep_NlsResidual",
            file!(),
            MSG_MRISTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    /* update 'ycur' value as stored predictor + current corrector */
    let zpred = mriStep_mem_mut(&ark_mem).zpred.clone().expect("zpred");
    let ycur = ark_mem.borrow().ycur.clone().expect("ycur");
    N_VLinearSum(ONE, &zpred, ONE, zcor, &ycur);

    /* call the user-supplied pre-RHS function (if supplied), then call RHS */
    let PreRhsFn = ark_mem.borrow().PreRhsFn;
    if let Some(PreRhsFn) = PreRhsFn {
        let tcur = ark_mem.borrow().tcur;
        let mut user_data = ark_mem.borrow_mut().user_data.take();
        let retval = PreRhsFn(tcur, &ycur, &mut user_data);
        ark_mem.borrow_mut().user_data = user_data;
        if retval != 0 {
            return ARK_PRERHSFN_FAIL;
        }
    }
    let (nls_fsi, Fsi_v) = {
        let step_mem = mriStep_mem_mut(&ark_mem);
        let idx = step_mem.stage_map[step_mem.istage as usize] as usize;
        (step_mem.nls_fsi.expect("nls_fsi"), step_mem.Fsi[idx].clone())
    };
    let tcur = ark_mem.borrow().tcur;
    let mut user_data = ark_mem.borrow_mut().user_data.take();
    let retval = nls_fsi(tcur, &ycur, &Fsi_v, &mut user_data);
    ark_mem.borrow_mut().user_data = user_data;
    mriStep_mem_mut(&ark_mem).nfsi += 1;
    if retval < 0 {
        return ARK_RHSFUNC_FAIL;
    }
    if retval > 0 {
        return RHSFUNC_RECVR;
    }

    /* compute residual: zcor - gamma*Fsi - sdata */
    let (sdata, gamma) = {
        let step_mem = mriStep_mem_mut(&ark_mem);
        (step_mem.sdata.clone().expect("sdata"), step_mem.gamma)
    };
    let c: [sunrealtype; 3] = [ONE, -ONE, -gamma];
    let X: [N_Vector; 3] = [zcor.clone(), sdata, Fsi_v.clone()];
    let retval = N_VLinearCombination(3, &c, &X, r);
    if retval != 0 {
        return ARK_VECTOROP_ERR;
    }
    ARK_SUCCESS
}

/*---------------------------------------------------------------
  mriStep_NlsFPFunction:

  This routine evaluates the fixed point iteration function for
  this solve-decoupled implicit MRI stage.  It assumes that any
  data from previous time steps/stages is contained in step_mem,
  and merely combines this old data with the current guess and
  current slow RHS vector to compute the iteration function g.

  At the ith stage, the new stage solution z=(zc+zp) should solve:
     zc = g(zc) := gamma*Fsi(z) + sdata
  where
     gamma = h*A(i,i),
     zp is the predicted stage solution,
     zc is stored in the input, zcor, and
     sdata is the old solution/stage data stored in step_mem->sdata.
  So we really just compute:
     z = zp + zc (stored in ark_mem->ycur)
     Fsi(z) (store in step_mem->Fsi[step_mem->istage])
     g = gamma*Fsi(z) + step_mem->sdata
  ---------------------------------------------------------------*/
pub fn mriStep_NlsFPFunction(
    zcor: &N_Vector,
    g: &N_Vector,
    arkode_mem: &mut Option<Box<dyn Any>>,
) -> i32 {
    /* access ARKodeMem and ARKodeMRIStepMem structures */
    let ark_mem = match nls_ark_mem(arkode_mem) {
        Some(ark_mem) => ark_mem,
        None => {
            arkProcessError(
                None,
                ARK_MEM_NULL,
                line!() as i32,
                "mriStep_NlsFPFunction",
                file!(),
                MSG_ARK_NO_MEM,
            );
            return ARK_MEM_NULL;
        }
    };
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(&ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "mriStep_NlsFPFunction",
            file!(),
            MSG_MRISTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    /* update 'ycur' value as stored predictor + current corrector */
    let zpred = mriStep_mem_mut(&ark_mem).zpred.clone().expect("zpred");
    let ycur = ark_mem.borrow().ycur.clone().expect("ycur");
    N_VLinearSum(ONE, &zpred, ONE, zcor, &ycur);

    /* call the user-supplied pre-RHS function (if supplied), then call RHS */
    let PreRhsFn = ark_mem.borrow().PreRhsFn;
    if let Some(PreRhsFn) = PreRhsFn {
        let tcur = ark_mem.borrow().tcur;
        let mut user_data = ark_mem.borrow_mut().user_data.take();
        let retval = PreRhsFn(tcur, &ycur, &mut user_data);
        ark_mem.borrow_mut().user_data = user_data;
        if retval != 0 {
            return ARK_PRERHSFN_FAIL;
        }
    }
    let (nls_fsi, Fsi_v) = {
        let step_mem = mriStep_mem_mut(&ark_mem);
        let idx = step_mem.stage_map[step_mem.istage as usize] as usize;
        (step_mem.nls_fsi.expect("nls_fsi"), step_mem.Fsi[idx].clone())
    };
    let tcur = ark_mem.borrow().tcur;
    let mut user_data = ark_mem.borrow_mut().user_data.take();
    let retval = nls_fsi(tcur, &ycur, &Fsi_v, &mut user_data);
    ark_mem.borrow_mut().user_data = user_data;
    mriStep_mem_mut(&ark_mem).nfsi += 1;
    if retval < 0 {
        return ARK_RHSFUNC_FAIL;
    }
    if retval > 0 {
        return RHSFUNC_RECVR;
    }

    /* combine parts:  g = gamma*Fsi(z) + sdata */
    let (sdata, gamma) = {
        let step_mem = mriStep_mem_mut(&ark_mem);
        (step_mem.sdata.clone().expect("sdata"), step_mem.gamma)
    };
    N_VLinearSum(gamma, &Fsi_v, ONE, &sdata, g);

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  mriStep_NlsConvTest:

  This routine provides the nonlinear solver convergence test for
  this solve-decoupled implicit MRI stage.  We have two modes.

  Standard:
      delnrm = ||del||_WRMS
      if (m==0) crate = 1
      if (m>0)  crate = max(crdown*crate, delnrm/delnrm_p)
      dcon = min(crate, ONE) * delnrm / nlscoef
      if (dcon<=1)  return convergence
      if ((m >= 2) && (delnrm > rdiv*delnrm_p))  return divergence

  Linearly-implicit mode:
      if the user specifies that the problem is linearly
      implicit, then we just declare 'success' no matter what
      is provided.
  ---------------------------------------------------------------*/
pub fn mriStep_NlsConvTest(
    NLS: &SUNNonlinearSolver,
    _y: &N_Vector,
    del: &N_Vector,
    tol: sunrealtype,
    ewt: &N_Vector,
    arkode_mem: &mut Option<Box<dyn Any>>,
) -> i32 {
    /* access ARKodeMem and ARKodeMRIStepMem structures */
    let ark_mem = match nls_ark_mem(arkode_mem) {
        Some(ark_mem) => ark_mem,
        None => {
            arkProcessError(
                None,
                ARK_MEM_NULL,
                line!() as i32,
                "mriStep_NlsConvTest",
                file!(),
                MSG_ARK_NO_MEM,
            );
            return ARK_MEM_NULL;
        }
    };
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(&ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "mriStep_NlsConvTest",
            file!(),
            MSG_MRISTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    /* if the problem is linearly implicit, just return success */
    let linear = mriStep_mem_mut(&ark_mem).linear;
    if linear {
        return SUN_SUCCESS;
    }

    /* compute the norm of the correction (C writes through
    &step_mem->delnrm; ported as copy-out / call / copy-back) */
    let mut delnrm = mriStep_mem_mut(&ark_mem).delnrm;
    let nrm_retval = mriStep_NlsNorm(del, ewt, &mut delnrm, arkode_mem);
    mriStep_mem_mut(&ark_mem).delnrm = delnrm;
    if nrm_retval != SUN_SUCCESS {
        /* unreachable: mriStep_NlsNorm always succeeds (as in C); the C call
        site passes no time argument for the MSG_TIME format — use tcur */
        let tcur = ark_mem.borrow().tcur;
        arkProcessError(
            Some(&ark_mem),
            ARK_NLS_OP_ERR,
            line!() as i32,
            "mriStep_NlsConvTest",
            file!(),
            &MSG_ARK_NLS_FAIL(tcur),
        );
        return ARK_NLS_OP_ERR;
    }

    /* get the current nonlinear solver iteration count */
    let mut m: i32 = 0;
    let retval = SUNNonlinSolGetCurIter(NLS, &mut m);
    if retval != ARK_SUCCESS {
        return ARK_MEM_NULL;
    }

    /* update the stored estimate of the convergence rate (assumes linear convergence) */
    if m > 0 {
        let mut step_mem = mriStep_mem_mut(&ark_mem);
        step_mem.crate_ = SUNMAX(
            step_mem.crdown * step_mem.crate_,
            step_mem.delnrm / step_mem.delnrm_p,
        );
    }

    /* compute our scaled error norm for testing convergence */
    let dcon = {
        let step_mem = mriStep_mem_mut(&ark_mem);
        SUNMIN(step_mem.crate_, ONE) * step_mem.delnrm / tol
    };

    /* check for convergence; if so return with success */
    if dcon <= ONE {
        return SUN_SUCCESS;
    }

    /* check for divergence */
    {
        let step_mem = mriStep_mem_mut(&ark_mem);
        if (m >= 1) && (step_mem.delnrm > step_mem.rdiv * step_mem.delnrm_p) {
            return SUN_NLS_CONV_RECVR;
        }
    }

    /* save norm of correction for next iteration */
    {
        let mut step_mem = mriStep_mem_mut(&ark_mem);
        step_mem.delnrm_p = step_mem.delnrm;
    }

    /* return with flag that there is more work to do */
    SUN_NLS_CONTINUE
}

fn mriStep_NlsNorm(
    del: &N_Vector,
    ewt: &N_Vector,
    delnrm: &mut sunrealtype,
    _arkode_mem: &mut Option<Box<dyn Any>>,
) -> SUNErrCode {
    *delnrm = N_VWrmsNorm(del, ewt);
    SUN_SUCCESS
}

fn mriStep_NlsGetUpdateNorm(
    delnrm: &mut sunrealtype,
    arkode_mem: &mut Option<Box<dyn Any>>,
) -> SUNErrCode {
    let ark_mem = match nls_ark_mem(arkode_mem) {
        Some(ark_mem) => ark_mem,
        None => return SUN_ERR_ARG_CORRUPT,
    };
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(&ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "mriStep_NlsGetUpdateNorm",
            file!(),
            MSG_MRISTEP_NO_MEM,
        );
        return SUN_ERR_ARG_CORRUPT;
    }

    *delnrm = mriStep_mem_mut(&ark_mem).delnrm;
    SUN_SUCCESS
}

fn mriStep_NlsGetConvRate(
    crate_: &mut sunrealtype,
    arkode_mem: &mut Option<Box<dyn Any>>,
) -> SUNErrCode {
    let ark_mem = match nls_ark_mem(arkode_mem) {
        Some(ark_mem) => ark_mem,
        None => return SUN_ERR_ARG_CORRUPT,
    };
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(&ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "mriStep_NlsGetConvRate",
            file!(),
            MSG_MRISTEP_NO_MEM,
        );
        return SUN_ERR_ARG_CORRUPT;
    }

    *crate_ = mriStep_mem_mut(&ark_mem).crate_;
    SUN_SUCCESS
}

/*===============================================================
  EOF
  ===============================================================*/
