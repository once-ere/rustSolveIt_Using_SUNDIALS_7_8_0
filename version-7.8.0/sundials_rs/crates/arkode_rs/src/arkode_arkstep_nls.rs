//! Port of `src/arkode/arkode_arkstep_nls.c`: the interface between
//! ARKStep and the `SUNNonlinearSolver` object.
//!
//! The C `void* arkode_mem` handed to the nonlinear solver as the
//! integrator mem / ctest / norm / getter data maps to a boxed
//! `ARKodeMem` clone inside an `Option<Box<dyn Any>>` token; each
//! callback downcasts the token back to the handle and then uses
//! granular borrows — never holding a `borrow()`/`borrow_mut()` or an
//! `arkStep_mem_mut` guard across a user callback, an `N_Vector`
//! operation on a user-visible vector, a mass-matrix product/solve, or
//! a linear-solver call (all of them re-enter the mem).
//!
//! `step_mem->jcur` is the shared `ARKJcurPtr` cell (see
//! `arkode_impl.rs`): `arkStep_NlsLSetup` clones the cell, drops the
//! step-memory borrow, hands `&jcur_cell` to `lsetup` (which may write
//! through it re-entrantly from a user preconditioner setup routine),
//! and reads the result back afterwards for the nonlinear solver's
//! `jcur` out-param.
//!
//! Build configuration: `SUNDIALS_LOGGING_LEVEL=2`, so every
//! `SUNLogInfo`/`SUNLogExtraDebugVec` call compiles away and is omitted
//! at translation time.

use std::any::Any;

use sundials_core::sundials_errors::{SUN_ERR_ARG_CORRUPT, SUN_SUCCESS};
use sundials_core::sundials_math::{SUNMAX, SUNMIN, SUNRabs};
use sundials_core::sundials_nonlinearsolver::*;
use sundials_core::sundials_nvector::{
    N_VConst, N_VLinearCombination, N_VLinearSum, N_VScale, N_VWrmsNorm, N_Vector,
};
use sundials_core::sundials_types::*;

use crate::arkode_arkstep::{
    arkStep_AccessARKODEStepMem, arkStep_AccessStepMem, arkStep_mem_mut, MASS_FIXED,
    MASS_IDENTITY, MASS_TIMEDEP, MSG_ARKSTEP_NO_MEM, MSG_NLS_INIT_FAIL,
};
use crate::arkode_impl::*;

/* -----------------------------------------------------------------
 * Adapter helper: recover the ARKodeMem handle from the token the
 * SUNNonlinearSolver hands back to these callbacks (C: the raw
 * `void* arkode_mem`). A missing/mistyped token corresponds to C
 * passing a garbage pointer, so `None` here maps to the same
 * `ARK_MEM_NULL` / `SUN_ERR_ARG_CORRUPT` reports C would produce for a
 * NULL `arkode_mem`.
 * ----------------------------------------------------------------- */

fn nlsARKodeMem(arkode_mem: &mut Option<Box<dyn Any>>) -> Option<ARKodeMem> {
    arkode_mem
        .as_mut()
        .and_then(|b| b.downcast_ref::<ARKodeMem>())
        .cloned()
}

/*===============================================================
  Interface routines supplied to ARKODE
  ===============================================================*/

/*---------------------------------------------------------------
  arkStep_SetNonlinearSolver:

  This routine attaches a SUNNonlinearSolver object to the ARKStep
  module.
  ---------------------------------------------------------------*/
pub fn arkStep_SetNonlinearSolver(ark_mem: &ARKodeMem, NLS: &SUNNonlinearSolver) -> i32 {
    let mut retval: i32;

    /* access ARKodeARKStepMem structure */
    retval = arkStep_AccessStepMem(ark_mem, "arkStep_SetNonlinearSolver");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* Return immediately if NLS input is NULL: handled by the type system
    (`&SUNNonlinearSolver` is never NULL) */

    /* check for required nonlinear solver functions */
    let missing_ops = {
        let ops = NLS.ops.borrow();
        ops.gettype.is_none()
            || ops.solve.is_none()
            || (ops.setsysfn.is_none() && ops.setsysfns.is_none())
    };
    if missing_ops {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "arkStep_SetNonlinearSolver",
            file!(),
            "NLS does not support required operations",
        );
        return ARK_ILL_INPUT;
    }

    /* free any existing nonlinear solver */
    let (old_nls, own_nls) = {
        let mut step_mem = arkStep_mem_mut(ark_mem);
        (step_mem.NLS.take(), step_mem.ownNLS)
    };
    if old_nls.is_some() && own_nls {
        /* C stores the result in retval but never reads it */
        let _ = SUNNonlinSolFree(old_nls);
    }

    /* set SUNNonlinearSolver pointer */
    {
        let mut step_mem = arkStep_mem_mut(ark_mem);
        step_mem.NLS = Some(NLS.clone());
        step_mem.ownNLS = SUNFALSE;
    }

    /* set default convergence test function */
    retval = SUNNonlinSolSetConvTestFn(
        NLS,
        Some(arkStep_NlsConvTest),
        Some(Box::new(ark_mem.clone())),
    );
    if retval != ARK_SUCCESS {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "arkStep_SetNonlinearSolver",
            file!(),
            "Setting convergence test function failed",
        );
        return ARK_ILL_INPUT;
    }

    retval = SUNNonlinSolSetNormFn(NLS, Some(arkStep_NlsNorm), Some(Box::new(ark_mem.clone())));
    if retval != ARK_SUCCESS {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "arkStep_SetNonlinearSolver",
            file!(),
            "Setting convergence-test norm function failed",
        );
        return ARK_ILL_INPUT;
    }

    retval = SUNNonlinSolSetGetUpdateNormFn(
        NLS,
        Some(arkStep_NlsGetUpdateNorm),
        Some(Box::new(ark_mem.clone())),
    );
    if retval != ARK_SUCCESS {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "arkStep_SetNonlinearSolver",
            file!(),
            "Setting update-norm getter failed",
        );
        return ARK_ILL_INPUT;
    }

    retval = SUNNonlinSolSetGetConvRateFn(
        NLS,
        Some(arkStep_NlsGetConvRate),
        Some(Box::new(ark_mem.clone())),
    );
    if retval != ARK_SUCCESS {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "arkStep_SetNonlinearSolver",
            file!(),
            "Setting convergence-rate getter failed",
        );
        return ARK_ILL_INPUT;
    }

    /* set default nonlinear iterations */
    let maxcor = arkStep_mem_mut(ark_mem).maxcor;
    retval = SUNNonlinSolSetMaxIters(NLS, maxcor);
    if retval != ARK_SUCCESS {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "arkStep_SetNonlinearSolver",
            file!(),
            "Setting maximum number of nonlinear iterations failed",
        );
        return ARK_ILL_INPUT;
    }

    /* set the nonlinear system RHS function */
    let fi = arkStep_mem_mut(ark_mem).fi;
    if fi.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "arkStep_SetNonlinearSolver",
            file!(),
            "The implicit ODE RHS function is NULL",
        );
        return ARK_ILL_INPUT;
    }
    arkStep_mem_mut(ark_mem).nls_fi = fi;

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  arkStep_SetNlsRhsFn:

  This routine sets an alternative user-supplied implicit ODE
  right-hand side function to use in the evaluation of nonlinear
  system functions.
  ---------------------------------------------------------------*/
pub fn arkStep_SetNlsRhsFn(ark_mem: &ARKodeMem, nls_fi: Option<ARKRhsFn>) -> i32 {
    /* access ARKodeARKStepMem structure */
    let retval = arkStep_AccessStepMem(ark_mem, "arkStep_SetNlsRhsFn");
    if retval != ARK_SUCCESS {
        return retval;
    }

    {
        let mut step_mem = arkStep_mem_mut(ark_mem);
        if nls_fi.is_some() {
            step_mem.nls_fi = nls_fi;
        } else {
            step_mem.nls_fi = step_mem.fi;
        }
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  arkStep_SetNlsSysFn:

  This routine sets the appropriate version of the nonlinear
  system function based on the current settings.
  ---------------------------------------------------------------*/
pub fn arkStep_SetNlsSysFn(ark_mem: &ARKodeMem) -> i32 {
    let root_fn: Option<SUNNonlinSolSysFn>;
    let fixedpoint_fn: Option<SUNNonlinSolSysFn>;
    let retval: i32;

    /* access ARKodeARKStepMem structure */
    let access = arkStep_AccessStepMem(ark_mem, "arkStep_SetNlsSysFn");
    if access != ARK_SUCCESS {
        return access;
    }

    /* determine residual/fixed-point functions based on current settings */
    let (mass_type, predictor, autonomous) = {
        let step_mem = arkStep_mem_mut(ark_mem);
        (step_mem.mass_type, step_mem.predictor, step_mem.autonomous)
    };
    if mass_type == MASS_IDENTITY {
        if predictor == 0 && autonomous {
            root_fn = Some(arkStep_NlsResidual_MassIdent_TrivialPredAutonomous);
            fixedpoint_fn = Some(arkStep_NlsFPFunction_MassIdent_TrivialPredAutonomous);
        } else {
            root_fn = Some(arkStep_NlsResidual_MassIdent);
            fixedpoint_fn = Some(arkStep_NlsFPFunction_MassIdent);
        }
    } else if mass_type == MASS_FIXED {
        if predictor == 0 && autonomous {
            root_fn = Some(arkStep_NlsResidual_MassFixed_TrivialPredAutonomous);
            fixedpoint_fn = Some(arkStep_NlsFPFunction_MassFixed_TrivialPredAutonomous);
        } else {
            root_fn = Some(arkStep_NlsResidual_MassFixed);
            fixedpoint_fn = Some(arkStep_NlsFPFunction_MassFixed);
        }
    } else if mass_type == MASS_TIMEDEP {
        root_fn = Some(arkStep_NlsResidual_MassTDep);
        fixedpoint_fn = Some(arkStep_NlsFPFunction_MassTDep);
    } else {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "arkStep_SetNlsSysFn",
            file!(),
            "Invalid mass matrix type",
        );
        return ARK_ILL_INPUT;
    }

    /* set the nonlinear residual/fixed-point function, based on solver type */
    let NLS = { arkStep_mem_mut(ark_mem).NLS.clone() }.expect("NLS");
    if SUNNonlinSolGetType(&NLS) == SUNNONLINEARSOLVER_ROOTFIND {
        retval = SUNNonlinSolSetSysFn(&NLS, root_fn);
    } else if SUNNonlinSolGetType(&NLS) == SUNNONLINEARSOLVER_FIXEDPOINT {
        retval = SUNNonlinSolSetSysFn(&NLS, fixedpoint_fn);
    } else if SUNNonlinSolGetType(&NLS) == SUNNONLINEARSOLVER_HYBRID {
        retval = SUNNonlinSolSetSysFns(&NLS, root_fn, fixedpoint_fn);
    } else {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "arkStep_SetNlsSysFn",
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
            "arkStep_SetNlsSysFn",
            file!(),
            "Setting nonlinear system function failed",
        );
        return ARK_ILL_INPUT;
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  arkStep_GetNonlinearSystemData:

  This routine provides access to the relevant data needed to
  compute the nonlinear system function.
  ---------------------------------------------------------------*/
#[allow(clippy::too_many_arguments)]
pub fn arkStep_GetNonlinearSystemData(
    ark_mem: &ARKodeMem,
    tcur: &mut sunrealtype,
    zpred: &mut Option<N_Vector>,
    z: &mut Option<N_Vector>,
    Fi: &mut Option<N_Vector>,
    gamma: &mut sunrealtype,
    sdata: &mut Option<N_Vector>,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    /* access ARKodeARKStepMem structure */
    let retval = arkStep_AccessStepMem(ark_mem, "arkStep_GetNonlinearSystemData");
    if retval != ARK_SUCCESS {
        return retval;
    }

    {
        let m = ark_mem.borrow();
        *tcur = m.tcur;
        *z = m.ycur.clone();
    }
    {
        let step_mem = arkStep_mem_mut(ark_mem);
        *zpred = step_mem.zpred.clone();
        *Fi = Some(step_mem.Fi[step_mem.istage as usize].clone());
        *gamma = step_mem.gamma;
        *sdata = step_mem.sdata.clone();
    }
    /* C copies the raw `user_data` pointer; the box is SWAPPED with the
    caller's out-param instead (accepted deviation class 6) — the caller
    must hand it back before the integrator next invokes a user
    callback. */
    std::mem::swap(&mut ark_mem.borrow_mut().user_data, user_data);

    ARK_SUCCESS
}

/*===============================================================
  Utility routines called by ARKStep
  ===============================================================*/

/*---------------------------------------------------------------
  arkStep_NlsInit:

  This routine attaches the linear solver 'setup' and 'solve'
  routines to the nonlinear solver object, and then initializes
  the nonlinear solver object itself.  This should only be
  called at the start of a simulation, after a re-init, or after
  a re-size.
  ---------------------------------------------------------------*/
pub fn arkStep_NlsInit(ark_mem: &ARKodeMem) -> i32 {
    let mut retval: i32;

    /* access ARKodeARKStepMem structure */
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "arkStep_NlsInit",
            file!(),
            MSG_ARKSTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    /* reset counters */
    let (has_lsetup, has_lsolve, NLS) = {
        let mut step_mem = arkStep_mem_mut(ark_mem);
        step_mem.nls_iters = 0;
        step_mem.nls_fails = 0;
        (
            step_mem.lsetup.is_some(),
            step_mem.lsolve.is_some(),
            step_mem.NLS.clone(),
        )
    };
    let NLS = NLS.expect("NLS");

    /* set the linear solver setup wrapper function */
    if has_lsetup {
        retval = SUNNonlinSolSetLSetupFn(&NLS, Some(arkStep_NlsLSetup));
    } else {
        retval = SUNNonlinSolSetLSetupFn(&NLS, None);
    }
    if retval != ARK_SUCCESS {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "arkStep_NlsInit",
            file!(),
            "Setting the linear solver setup function failed",
        );
        return ARK_NLS_INIT_FAIL;
    }

    /* set the linear solver solve wrapper function */
    if has_lsolve {
        retval = SUNNonlinSolSetLSolveFn(&NLS, Some(arkStep_NlsLSolve));
    } else {
        retval = SUNNonlinSolSetLSolveFn(&NLS, None);
    }
    if retval != ARK_SUCCESS {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "arkStep_NlsInit",
            file!(),
            "Setting linear solver solve function failed",
        );
        return ARK_NLS_INIT_FAIL;
    }

    retval = arkStep_SetNlsSysFn(ark_mem);
    if retval != ARK_SUCCESS {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "arkStep_NlsInit",
            file!(),
            "Setting nonlinear system function failed",
        );
        return ARK_ILL_INPUT;
    }

    /* initialize nonlinear solver */
    retval = SUNNonlinSolInitialize(&NLS);
    if retval != ARK_SUCCESS {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "arkStep_NlsInit",
            file!(),
            MSG_NLS_INIT_FAIL,
        );
        return ARK_NLS_INIT_FAIL;
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  arkStep_Nls

  This routine attempts to solve the nonlinear system associated
  with a single implicit stage.  It calls the supplied
  SUNNonlinearSolver object to perform the solve.

  Upon entry, the predicted solution is held in step_mem->zpred,
  which is never changed throughout this routine.  If an initial
  attempt at solving the nonlinear system fails (e.g. due to a
  stale Jacobian), this allows for new attempts at the solution.

  Upon a successful solve, the solution is held in ark_mem->ycur.
  ---------------------------------------------------------------*/
pub fn arkStep_Nls(ark_mem: &ARKodeMem, nflag: i32) -> i32 {
    let callLSetup: sunbooleantype;
    let mut nls_iters_inc: i64 = 0;
    let mut nls_fails_inc: i64 = 0;

    /* access ARKodeARKStepMem structure */
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "arkStep_Nls",
            file!(),
            MSG_ARKSTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    /* If a linear solver 'setup' is supplied, set various flags for
       determining whether it should be called */
    let (has_lsetup, linear) = {
        let step_mem = arkStep_mem_mut(ark_mem);
        (step_mem.lsetup.is_some(), step_mem.linear)
    };
    if has_lsetup {
        /* Set interface 'convfail' flag for use inside lsetup */
        {
            let mut step_mem = arkStep_mem_mut(ark_mem);
            if linear {
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
        let (msbp, gamrat, dgmax, linear_timedep, nstlp) = {
            let step_mem = arkStep_mem_mut(ark_mem);
            (
                step_mem.msbp,
                step_mem.gamrat,
                step_mem.dgmax,
                step_mem.linear_timedep,
                step_mem.nstlp,
            )
        };
        let mut call_lsetup = firststage || (msbp < 0) || (SUNRabs(gamrat - ONE) > dgmax);
        if linear {
            /* linearly-implicit problem */
            call_lsetup = call_lsetup || linear_timedep;
        } else {
            /* nonlinearly-implicit problem */
            call_lsetup = call_lsetup
                || (nflag == PREV_CONV_FAIL)
                || (nflag == PREV_ERR_FAIL)
                || (nst >= nstlp + (msbp.abs() as i64));
        }
        callLSetup = call_lsetup;
    } else {
        arkStep_mem_mut(ark_mem).crate_ = ONE;
        callLSetup = SUNFALSE;
    }

    /* set a zero guess for correction */
    let zcor = { arkStep_mem_mut(ark_mem).zcor.clone() }.expect("zcor");
    N_VConst(ZERO, &zcor);

    /* Reset the stored residual norm (for iterative linear solvers) */
    {
        let mut step_mem = arkStep_mem_mut(ark_mem);
        step_mem.eRNrm = 0.1 * step_mem.nlscoef; /* SUN_RCONST(0.1) */
    }

    /* solve the nonlinear system for the actual correction */
    let (NLS, zpred, nlscoef) = {
        let step_mem = arkStep_mem_mut(ark_mem);
        (
            step_mem.NLS.clone().expect("NLS"),
            step_mem.zpred.clone().expect("zpred"),
            step_mem.nlscoef,
        )
    };
    let ewt = { ark_mem.borrow().ewt.clone() }.expect("ewt");
    let mut nls_mem: Option<Box<dyn Any>> = Some(Box::new(ark_mem.clone()));
    let retval = SUNNonlinSolSolve(
        &NLS,
        &zpred,
        &zcor,
        &ewt,
        nlscoef,
        callLSetup,
        &mut nls_mem,
    );

    /* increment counters */
    let _ = SUNNonlinSolGetNumIters(&NLS, &mut nls_iters_inc);
    arkStep_mem_mut(ark_mem).nls_iters += nls_iters_inc;

    let _ = SUNNonlinSolGetNumConvFails(&NLS, &mut nls_fails_inc);
    arkStep_mem_mut(ark_mem).nls_fails += nls_fails_inc;

    /* successful solve -- reset jcur flag and apply correction */
    if retval == SUN_SUCCESS {
        let jcur = { arkStep_mem_mut(ark_mem).jcur.clone() };
        jcur.set(SUNFALSE);
        let ycur = { ark_mem.borrow().ycur.clone() }.expect("ycur");
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
  arkStep_NlsLSetup:

  This routine wraps the ARKODE linear solver interface 'setup'
  routine for use by the nonlinear solver object.
  ---------------------------------------------------------------*/
pub fn arkStep_NlsLSetup(
    jbad: sunbooleantype,
    jcur: &mut sunbooleantype,
    arkode_mem: &mut Option<Box<dyn Any>>,
) -> i32 {
    /* access ARKodeMem and ARKodeARKStepMem structures */
    let ark_mem = match nlsARKodeMem(arkode_mem) {
        Some(ark_mem) => ark_mem,
        None => {
            arkProcessError(
                None,
                ARK_MEM_NULL,
                line!() as i32,
                "arkStep_NlsLSetup",
                file!(),
                MSG_ARK_NO_MEM,
            );
            return ARK_MEM_NULL;
        }
    };
    let access = arkStep_AccessARKODEStepMem(&ark_mem, "arkStep_NlsLSetup");
    if access != ARK_SUCCESS {
        return access;
    }

    /* update convfail based on jbad flag */
    if jbad {
        arkStep_mem_mut(&ark_mem).convfail = ARK_FAIL_BAD_J;
    }

    /* Use ARKODE's tempv1, tempv2 and tempv3 as
       temporary vectors for the linear solver setup routine */
    let (lsetup, convfail, fpred, jcur_cell) = {
        let mut step_mem = arkStep_mem_mut(&ark_mem);
        step_mem.nsetups += 1;
        let istage = step_mem.istage as usize;
        (
            step_mem.lsetup.expect("lsetup"),
            step_mem.convfail,
            step_mem.Fi[istage].clone(),
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
        &ark_mem, convfail, tcur, &ycur, &fpred, &jcur_cell, &tempv1, &tempv2, &tempv3,
    );

    /* update Jacobian status (C reads back through &step_mem->jcur, which
    a re-entrant psetup may have written) */
    *jcur = jcur_cell.get();

    /* update flags and 'gamma' values for last lsetup call */
    let nst = {
        let mut m = ark_mem.borrow_mut();
        m.firststage = SUNFALSE;
        m.nst
    };
    {
        let mut step_mem = arkStep_mem_mut(&ark_mem);
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
  arkStep_NlsLSolve:

  This routine wraps the ARKODE linear solver interface 'solve'
  routine for use by the nonlinear solver object.
  ---------------------------------------------------------------*/
pub fn arkStep_NlsLSolve(b: &N_Vector, arkode_mem: &mut Option<Box<dyn Any>>) -> i32 {
    /* access ARKodeMem and ARKodeARKStepMem structures */
    let ark_mem = match nlsARKodeMem(arkode_mem) {
        Some(ark_mem) => ark_mem,
        None => {
            arkProcessError(
                None,
                ARK_MEM_NULL,
                line!() as i32,
                "arkStep_NlsLSolve",
                file!(),
                MSG_ARK_NO_MEM,
            );
            return ARK_MEM_NULL;
        }
    };
    let access = arkStep_AccessARKODEStepMem(&ark_mem, "arkStep_NlsLSolve");
    if access != ARK_SUCCESS {
        return access;
    }

    /* retrieve nonlinear solver iteration from module */
    let NLS = { arkStep_mem_mut(&ark_mem).NLS.clone() }.expect("NLS");
    let mut nonlin_iter: i32 = 0;
    let retval = SUNNonlinSolGetCurIter(&NLS, &mut nonlin_iter);
    if retval != SUN_SUCCESS {
        return ARK_NLS_OP_ERR;
    }

    /* call linear solver interface, and handle return value */
    let (lsolve, fcur, eRNrm) = {
        let step_mem = arkStep_mem_mut(&ark_mem);
        let istage = step_mem.istage as usize;
        (
            step_mem.lsolve.expect("lsolve"),
            step_mem.Fi[istage].clone(),
            step_mem.eRNrm,
        )
    };
    let (tcur, ycur) = {
        let m = ark_mem.borrow();
        (m.tcur, m.ycur.clone().expect("ycur"))
    };
    let retval = lsolve(&ark_mem, b, tcur, &ycur, &fcur, eRNrm, nonlin_iter);

    if retval < 0 {
        return ARK_LSOLVE_FAIL;
    }
    if retval > 0 {
        return CONV_FAIL;
    }

    ARK_SUCCESS
}

/* -----------------------------------------------------------------
 * Shared helper for the residual / fixed-point functions: run the
 * user-supplied pre-RHS hook (if supplied) and then the implicit ODE
 * RHS, exactly as the C blocks do. The `user_data` box is taken out of
 * the mem for the duration of each callback and restored on EVERY path
 * (including the early error return). Returns `ARK_SUCCESS` when the
 * caller should continue, or the flag the C code returns.
 * ----------------------------------------------------------------- */
fn arkStep_NlsCallRhs(ark_mem: &ARKodeMem, ycur: &N_Vector, fi_istage: &N_Vector) -> i32 {
    /* call the user-supplied pre-RHS function (if supplied), then call RHS */
    let (PreRhsFn, tcur) = {
        let m = ark_mem.borrow();
        (m.PreRhsFn, m.tcur)
    };
    if let Some(pre_rhs_fn) = PreRhsFn {
        let mut user_data = ark_mem.borrow_mut().user_data.take();
        let retval = pre_rhs_fn(tcur, ycur, &mut user_data);
        ark_mem.borrow_mut().user_data = user_data;
        if retval != 0 {
            return ARK_PRERHSFN_FAIL;
        }
    }
    let nls_fi = { arkStep_mem_mut(ark_mem).nls_fi }.expect("nls_fi");
    let mut user_data = ark_mem.borrow_mut().user_data.take();
    let retval = nls_fi(tcur, ycur, fi_istage, &mut user_data);
    ark_mem.borrow_mut().user_data = user_data;
    arkStep_mem_mut(ark_mem).nfi += 1;
    if retval < 0 {
        return ARK_RHSFUNC_FAIL;
    }
    if retval > 0 {
        return RHSFUNC_RECVR;
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  arkStep_NlsResidual_MassIdent
  arkStep_NlsResidual_MassIdent_TrivialPredAutonomous

  This routine evaluates the nonlinear residual for the additive
  Runge-Kutta method.  It assumes that any data from previous
  time steps/stages is contained in step_mem, and merely combines
  this old data with the current implicit ODE RHS vector to
  compute the nonlinear residual r.

  This version assumes an identity mass matrix.

  At the ith stage, we compute the residual vector:
     r = z - yn - h*sum_{j=0}^{i-1} Ae(i,j)*Fe(j)
           - h*sum_{j=0}^{i} Ai(i,j)*Fi(j)
  <=>
     r = zp + zc - yn - h*sum_{j=0}^{i-1} Ae(i,j)*Fe(j)
            - h*sum_{j=0}^{i} Ai(i,j)*Fi(j)
  <=>
     r = (zc - gamma*Fi(z)) - (yn - zp + data)
  where the current stage solution z = zp + zc, and where
     zc is stored in the input, zcor
     (yn-zp+data) is stored in step_mem->sdata,
  so we really just compute:
     z = zp + zc (stored in ark_mem->ycur)
     Fi(z) (stored step_mem->Fi[step_mem->istage])
     r = zc - gamma*Fi(z) - step_mem->sdata

  The "TrivialPredAutonomous" version reuses the implicit RHS
  evaluation at the beginning of the step in the initial residual
  evaluation.
  ---------------------------------------------------------------*/
pub fn arkStep_NlsResidual_MassIdent(
    zcor: &N_Vector,
    r: &N_Vector,
    arkode_mem: &mut Option<Box<dyn Any>>,
) -> i32 {
    /* access ARKodeMem and ARKodeARKStepMem structures */
    let ark_mem = match nlsARKodeMem(arkode_mem) {
        Some(ark_mem) => ark_mem,
        None => {
            arkProcessError(
                None,
                ARK_MEM_NULL,
                line!() as i32,
                "arkStep_NlsResidual_MassIdent",
                file!(),
                MSG_ARK_NO_MEM,
            );
            return ARK_MEM_NULL;
        }
    };
    let access = arkStep_AccessARKODEStepMem(&ark_mem, "arkStep_NlsResidual_MassIdent");
    if access != ARK_SUCCESS {
        return access;
    }

    /* update 'ycur' value as stored predictor + current corrector */
    let zpred = { arkStep_mem_mut(&ark_mem).zpred.clone() }.expect("zpred");
    let ycur = { ark_mem.borrow().ycur.clone() }.expect("ycur");
    N_VLinearSum(ONE, &zpred, ONE, zcor, &ycur);

    let fi_istage = {
        let step_mem = arkStep_mem_mut(&ark_mem);
        let istage = step_mem.istage as usize;
        step_mem.Fi[istage].clone()
    };
    let retval = arkStep_NlsCallRhs(&ark_mem, &ycur, &fi_istage);
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* compute residual via linear combination */
    let (gamma, sdata) = {
        let step_mem = arkStep_mem_mut(&ark_mem);
        (step_mem.gamma, step_mem.sdata.clone().expect("sdata"))
    };
    let c: [sunrealtype; 3] = [ONE, -ONE, -gamma];
    let X: [N_Vector; 3] = [zcor.clone(), sdata, fi_istage];
    let retval = N_VLinearCombination(3, &c, &X, r);
    if retval != 0 {
        return ARK_VECTOROP_ERR;
    }

    ARK_SUCCESS
}

pub fn arkStep_NlsResidual_MassIdent_TrivialPredAutonomous(
    zcor: &N_Vector,
    r: &N_Vector,
    arkode_mem: &mut Option<Box<dyn Any>>,
) -> i32 {
    /* access ARKodeMem and ARKodeARKStepMem structures */
    let ark_mem = match nlsARKodeMem(arkode_mem) {
        Some(ark_mem) => ark_mem,
        None => {
            arkProcessError(
                None,
                ARK_MEM_NULL,
                line!() as i32,
                "arkStep_NlsResidual_MassIdent_TrivialPredAutonomous",
                file!(),
                MSG_ARK_NO_MEM,
            );
            return ARK_MEM_NULL;
        }
    };
    let access = arkStep_AccessARKODEStepMem(
        &ark_mem,
        "arkStep_NlsResidual_MassIdent_TrivialPredAutonomous",
    );
    if access != ARK_SUCCESS {
        return access;
    }

    /* update 'ycur' value as stored predictor + current corrector */
    let zpred = { arkStep_mem_mut(&ark_mem).zpred.clone() }.expect("zpred");
    let ycur = { ark_mem.borrow().ycur.clone() }.expect("ycur");
    N_VLinearSum(ONE, &zpred, ONE, zcor, &ycur);

    /* compute implicit RHS if not already available */
    let NLS = { arkStep_mem_mut(&ark_mem).NLS.clone() }.expect("NLS");
    let mut nls_iter: i32 = 0;
    let retval = SUNNonlinSolGetCurIter(&NLS, &mut nls_iter);
    if retval != ARK_SUCCESS {
        return ARK_NLS_OP_ERR;
    }

    let (fn_implicit, fi_istage) = {
        let step_mem = arkStep_mem_mut(&ark_mem);
        let istage = step_mem.istage as usize;
        (step_mem.fn_implicit.clone(), step_mem.Fi[istage].clone())
    };
    if nls_iter == 0 && fn_implicit.is_some() {
        N_VScale(ONE, fn_implicit.as_ref().expect("fn_implicit"), &fi_istage);
    } else {
        let retval = arkStep_NlsCallRhs(&ark_mem, &ycur, &fi_istage);
        if retval != ARK_SUCCESS {
            return retval;
        }
    }

    /* compute residual via linear combination */
    let (gamma, sdata) = {
        let step_mem = arkStep_mem_mut(&ark_mem);
        (step_mem.gamma, step_mem.sdata.clone().expect("sdata"))
    };
    let c: [sunrealtype; 3] = [ONE, -ONE, -gamma];
    let X: [N_Vector; 3] = [zcor.clone(), sdata, fi_istage];
    let retval = N_VLinearCombination(3, &c, &X, r);
    if retval != 0 {
        return ARK_VECTOROP_ERR;
    }
    ARK_SUCCESS
}

/*---------------------------------------------------------------
  arkStep_NlsResidual_MassFixed
  arkStep_NlsResidual_MassFixed_TrivialPredAutonomous

  This routine evaluates the nonlinear residual for the additive
  Runge-Kutta method.  It assumes that any data from previous
  time steps/stages is contained in step_mem, and merely combines
  this old data with the current implicit ODE RHS vector to
  compute the nonlinear residual r.

  This version assumes a fixed mass matrix.

  At the ith stage, we compute the residual vector:
     r = M*z - M*yn - h*sum_{j=0}^{i-1} Ae(i,j)*Fe(j)
                    - h*sum_{j=0}^{i} Ai(i,j)*Fi(j)
  <=>
     r = M*zp + M*zc - M*yn - h*sum_{j=0}^{i-1} Ae(i,j)*Fe(j)
                            - h*sum_{j=0}^{i} Ai(i,j)*Fi(j)
  <=>
     r = (M*zc - gamma*Fi(z)) - (M*yn - M*zp + data)
  where the current stage solution z = zp + zc, and where
     zc is stored in the input, zcor
     (M*yn-M*zp+data) is stored in step_mem->sdata,
  so we really just compute:
     z = zp + zc (stored in ark_mem->ycur)
     Fi(z) (stored step_mem->Fi[step_mem->istage])
     r = M*zc - gamma*Fi(z) - step_mem->sdata

  The "TrivialPredAutonomous" version reuses the implicit RHS
  evaluation at the beginning of the step in the initial residual
  evaluation.
  ---------------------------------------------------------------*/
pub fn arkStep_NlsResidual_MassFixed(
    zcor: &N_Vector,
    r: &N_Vector,
    arkode_mem: &mut Option<Box<dyn Any>>,
) -> i32 {
    /* access ARKodeMem and ARKodeARKStepMem structures */
    let ark_mem = match nlsARKodeMem(arkode_mem) {
        Some(ark_mem) => ark_mem,
        None => {
            arkProcessError(
                None,
                ARK_MEM_NULL,
                line!() as i32,
                "arkStep_NlsResidual_MassFixed",
                file!(),
                MSG_ARK_NO_MEM,
            );
            return ARK_MEM_NULL;
        }
    };
    let access = arkStep_AccessARKODEStepMem(&ark_mem, "arkStep_NlsResidual_MassFixed");
    if access != ARK_SUCCESS {
        return access;
    }

    /* update 'ycur' value as stored predictor + current corrector */
    let zpred = { arkStep_mem_mut(&ark_mem).zpred.clone() }.expect("zpred");
    let ycur = { ark_mem.borrow().ycur.clone() }.expect("ycur");
    N_VLinearSum(ONE, &zpred, ONE, zcor, &ycur);

    let fi_istage = {
        let step_mem = arkStep_mem_mut(&ark_mem);
        let istage = step_mem.istage as usize;
        step_mem.Fi[istage].clone()
    };
    let retval = arkStep_NlsCallRhs(&ark_mem, &ycur, &fi_istage);
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* put M*zcor in r */
    let mmult = { arkStep_mem_mut(&ark_mem).mmult }.expect("mmult");
    let retval = mmult(&ark_mem, zcor, r);
    if retval != ARK_SUCCESS {
        return ARK_MASSMULT_FAIL;
    }

    /* compute residual via linear combination */
    let (gamma, sdata) = {
        let step_mem = arkStep_mem_mut(&ark_mem);
        (step_mem.gamma, step_mem.sdata.clone().expect("sdata"))
    };
    let c: [sunrealtype; 3] = [ONE, -ONE, -gamma];
    let X: [N_Vector; 3] = [r.clone(), sdata, fi_istage];
    let retval = N_VLinearCombination(3, &c, &X, r);
    if retval != 0 {
        return ARK_VECTOROP_ERR;
    }
    ARK_SUCCESS
}

pub fn arkStep_NlsResidual_MassFixed_TrivialPredAutonomous(
    zcor: &N_Vector,
    r: &N_Vector,
    arkode_mem: &mut Option<Box<dyn Any>>,
) -> i32 {
    /* access ARKodeMem and ARKodeARKStepMem structures */
    let ark_mem = match nlsARKodeMem(arkode_mem) {
        Some(ark_mem) => ark_mem,
        None => {
            arkProcessError(
                None,
                ARK_MEM_NULL,
                line!() as i32,
                "arkStep_NlsResidual_MassFixed_TrivialPredAutonomous",
                file!(),
                MSG_ARK_NO_MEM,
            );
            return ARK_MEM_NULL;
        }
    };
    let access = arkStep_AccessARKODEStepMem(
        &ark_mem,
        "arkStep_NlsResidual_MassFixed_TrivialPredAutonomous",
    );
    if access != ARK_SUCCESS {
        return access;
    }

    /* update 'ycur' value as stored predictor + current corrector */
    let zpred = { arkStep_mem_mut(&ark_mem).zpred.clone() }.expect("zpred");
    let ycur = { ark_mem.borrow().ycur.clone() }.expect("ycur");
    N_VLinearSum(ONE, &zpred, ONE, zcor, &ycur);

    /* compute implicit RHS if not already available */
    let NLS = { arkStep_mem_mut(&ark_mem).NLS.clone() }.expect("NLS");
    let mut nls_iter: i32 = 0;
    let retval = SUNNonlinSolGetCurIter(&NLS, &mut nls_iter);
    if retval != ARK_SUCCESS {
        return ARK_NLS_OP_ERR;
    }

    let (fn_implicit, fi_istage) = {
        let step_mem = arkStep_mem_mut(&ark_mem);
        let istage = step_mem.istage as usize;
        (step_mem.fn_implicit.clone(), step_mem.Fi[istage].clone())
    };
    if nls_iter == 0 && fn_implicit.is_some() {
        N_VScale(ONE, fn_implicit.as_ref().expect("fn_implicit"), &fi_istage);
    } else {
        let retval = arkStep_NlsCallRhs(&ark_mem, &ycur, &fi_istage);
        if retval != ARK_SUCCESS {
            return retval;
        }
    }

    /* put M*zcor in r */
    let mmult = { arkStep_mem_mut(&ark_mem).mmult }.expect("mmult");
    let retval = mmult(&ark_mem, zcor, r);
    if retval != ARK_SUCCESS {
        return ARK_MASSMULT_FAIL;
    }

    /* compute residual via linear combination */
    let (gamma, sdata) = {
        let step_mem = arkStep_mem_mut(&ark_mem);
        (step_mem.gamma, step_mem.sdata.clone().expect("sdata"))
    };
    let c: [sunrealtype; 3] = [ONE, -ONE, -gamma];
    let X: [N_Vector; 3] = [r.clone(), sdata, fi_istage];
    let retval = N_VLinearCombination(3, &c, &X, r);
    if retval != 0 {
        return ARK_VECTOROP_ERR;
    }
    ARK_SUCCESS
}

/*---------------------------------------------------------------
  arkStep_NlsResidual_MassTDep:

  This routine evaluates the nonlinear residual for the additive
  Runge-Kutta method.  It assumes that any data from previous
  time steps/stages is contained in step_mem, and merely combines
  this old data with the current implicit ODE RHS vector to
  compute the nonlinear residual r.

  This version assumes a time-dependent mass matrix.

  At the ith stage, we compute the residual vector:
     r = M(ti)*(z - yn) - M(ti)*h*sum_{j=0}^{i-1} Ae(i,j)*M(tj)^{-1}*Fe(j)
                        - M(ti)*h*sum_{j=0}^{i} Ai(i,j)*M(tj)^{-1}*Fi(j)
  <=>
     r = M(ti)*[zc + zp - yn - h*sum_{j=0}^{i-1} (Ai(i,j)*M(tj)^{-1}*Fi(j)
                                                + Ae(i,j)*M(tj)^{-1}*Fe(j))]
         - M(ti)*gamma*M(ti)^{-1}*Fi(i)
  <=>
     r = M(ti)*(zc - data) - gamma*Fi(z)
  where the current stage solution z = zp + zc, and where
     zc is stored in the input, zcor
     yn - zp + h*sum_{j=0}^{i-1} (Ai(i,j)*M(tj)^{-1}*Fi(j)
        + Ae(i,j)*M(tj)^{-1}*Fe(j)) stored in step_mem->sdata,
  so we really just compute:
     z = zp + zc (stored in ark_mem->ycur)
     tmp = zc - data (stored in Fi[istage])
     M(t)*tmp (stored in r)
     Fi(z) (stored step_mem->Fi[istage])
     r = r - gamma*Fi(z)
  ---------------------------------------------------------------*/
pub fn arkStep_NlsResidual_MassTDep(
    zcor: &N_Vector,
    r: &N_Vector,
    arkode_mem: &mut Option<Box<dyn Any>>,
) -> i32 {
    /* access ARKodeMem and ARKodeARKStepMem structures */
    let ark_mem = match nlsARKodeMem(arkode_mem) {
        Some(ark_mem) => ark_mem,
        None => {
            arkProcessError(
                None,
                ARK_MEM_NULL,
                line!() as i32,
                "arkStep_NlsResidual_MassTDep",
                file!(),
                MSG_ARK_NO_MEM,
            );
            return ARK_MEM_NULL;
        }
    };
    let access = arkStep_AccessARKODEStepMem(&ark_mem, "arkStep_NlsResidual_MassTDep");
    if access != ARK_SUCCESS {
        return access;
    }

    /* update 'ycur' value as stored predictor + current corrector */
    let zpred = { arkStep_mem_mut(&ark_mem).zpred.clone() }.expect("zpred");
    let ycur = { ark_mem.borrow().ycur.clone() }.expect("ycur");
    N_VLinearSum(ONE, &zpred, ONE, zcor, &ycur);

    /* put M*(zcor - sdata) in r (use Fi[is] as temporary storage) */
    let (sdata, fi_istage) = {
        let step_mem = arkStep_mem_mut(&ark_mem);
        let istage = step_mem.istage as usize;
        (
            step_mem.sdata.clone().expect("sdata"),
            step_mem.Fi[istage].clone(),
        )
    };
    N_VLinearSum(ONE, zcor, -ONE, &sdata, &fi_istage);
    let mmult = { arkStep_mem_mut(&ark_mem).mmult }.expect("mmult");
    let retval = mmult(&ark_mem, &fi_istage, r);
    if retval != ARK_SUCCESS {
        return ARK_MASSMULT_FAIL;
    }

    let retval = arkStep_NlsCallRhs(&ark_mem, &ycur, &fi_istage);
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* compute residual via linear sum */
    let gamma = { arkStep_mem_mut(&ark_mem).gamma };
    N_VLinearSum(ONE, r, -gamma, &fi_istage, r);
    ARK_SUCCESS
}

/*---------------------------------------------------------------
  arkStep_NlsFPFunction_MassIdent
  arkStep_NlsFPFunction_MassIdent_TrivialPredAutonomous

  This routine evaluates the fixed point iteration function for
  the additive Runge-Kutta method.  It assumes that any data from
  previous time steps/stages is contained in step_mem, and
  merely combines this old data with the current guess and
  implicit ODE RHS vector to compute the iteration function g.

  This version assumes an identity mass matrix.

  At the ith stage, the new stage solution z should solve:
     z = yn + h*sum_{j=0}^{i-1} Ae(i,j)*Fe(j)
            + h*sum_{j=0}^{i} Ai(i,j)*Fi(j)
  <=>
     z = yn + gamma*Fi(z) + h*sum_{j=0}^{i-1} ( Ae(i,j)*Fe(j)
                                              + Ai(i,j)*Fi(j) )
  <=>
     z = yn + gamma*Fi(z) + data
  <=>
     zc = -zp + yn + gamma*Fi(zp+zc) + data
  Where zp is the predicted stage and zc is the correction to
  the prediction.

  Our fixed-point problem is zc=g(zc), so the FP function is just:
     g(z) = gamma*Fi(z) + (yn - zp + data)
  where the current nonlinear guess is z = zp + zc, and where
     z is stored in ycur,
     zp is stored in step_mem->zpred,
     (yn-zp+data) is stored in step_mem->sdata,
  so we really just compute:
     Fi(z) (store in step_mem->Fi[step_mem->istage])
     g = gamma*Fi(z) + step_mem->sdata

  The "TrivialPredAutonomous" version reuses the implicit RHS
  evaluation at the beginning of the step in the initial FP
  function evaluation.
  ---------------------------------------------------------------*/
pub fn arkStep_NlsFPFunction_MassIdent(
    zcor: &N_Vector,
    g: &N_Vector,
    arkode_mem: &mut Option<Box<dyn Any>>,
) -> i32 {
    /* access ARKodeMem and ARKodeARKStepMem structures */
    let ark_mem = match nlsARKodeMem(arkode_mem) {
        Some(ark_mem) => ark_mem,
        None => {
            arkProcessError(
                None,
                ARK_MEM_NULL,
                line!() as i32,
                "arkStep_NlsFPFunction_MassIdent",
                file!(),
                MSG_ARK_NO_MEM,
            );
            return ARK_MEM_NULL;
        }
    };
    let access = arkStep_AccessARKODEStepMem(&ark_mem, "arkStep_NlsFPFunction_MassIdent");
    if access != ARK_SUCCESS {
        return access;
    }

    /* update 'ycur' value as stored predictor + current corrector */
    let zpred = { arkStep_mem_mut(&ark_mem).zpred.clone() }.expect("zpred");
    let ycur = { ark_mem.borrow().ycur.clone() }.expect("ycur");
    N_VLinearSum(ONE, &zpred, ONE, zcor, &ycur);

    let fi_istage = {
        let step_mem = arkStep_mem_mut(&ark_mem);
        let istage = step_mem.istage as usize;
        step_mem.Fi[istage].clone()
    };
    let retval = arkStep_NlsCallRhs(&ark_mem, &ycur, &fi_istage);
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* combine parts:  g = gamma*Fi(z) + sdata */
    let (gamma, sdata) = {
        let step_mem = arkStep_mem_mut(&ark_mem);
        (step_mem.gamma, step_mem.sdata.clone().expect("sdata"))
    };
    N_VLinearSum(gamma, &fi_istage, ONE, &sdata, g);

    ARK_SUCCESS
}

pub fn arkStep_NlsFPFunction_MassIdent_TrivialPredAutonomous(
    zcor: &N_Vector,
    g: &N_Vector,
    arkode_mem: &mut Option<Box<dyn Any>>,
) -> i32 {
    /* access ARKodeMem and ARKodeARKStepMem structures */
    let ark_mem = match nlsARKodeMem(arkode_mem) {
        Some(ark_mem) => ark_mem,
        None => {
            arkProcessError(
                None,
                ARK_MEM_NULL,
                line!() as i32,
                "arkStep_NlsFPFunction_MassIdent_TrivialPredAutonomous",
                file!(),
                MSG_ARK_NO_MEM,
            );
            return ARK_MEM_NULL;
        }
    };
    let access = arkStep_AccessARKODEStepMem(
        &ark_mem,
        "arkStep_NlsFPFunction_MassIdent_TrivialPredAutonomous",
    );
    if access != ARK_SUCCESS {
        return access;
    }

    /* update 'ycur' value as stored predictor + current corrector */
    let zpred = { arkStep_mem_mut(&ark_mem).zpred.clone() }.expect("zpred");
    let ycur = { ark_mem.borrow().ycur.clone() }.expect("ycur");
    N_VLinearSum(ONE, &zpred, ONE, zcor, &ycur);

    /* compute implicit RHS if not already available */
    let NLS = { arkStep_mem_mut(&ark_mem).NLS.clone() }.expect("NLS");
    let mut nls_iter: i32 = 0;
    let retval = SUNNonlinSolGetCurIter(&NLS, &mut nls_iter);
    if retval != ARK_SUCCESS {
        return ARK_NLS_OP_ERR;
    }

    let (fn_implicit, fi_istage) = {
        let step_mem = arkStep_mem_mut(&ark_mem);
        let istage = step_mem.istage as usize;
        (step_mem.fn_implicit.clone(), step_mem.Fi[istage].clone())
    };
    if nls_iter == 0 && fn_implicit.is_some() {
        N_VScale(ONE, fn_implicit.as_ref().expect("fn_implicit"), &fi_istage);
    } else {
        let retval = arkStep_NlsCallRhs(&ark_mem, &ycur, &fi_istage);
        if retval != ARK_SUCCESS {
            return retval;
        }
    }

    /* combine parts:  g = gamma*Fi(z) + sdata */
    let (gamma, sdata) = {
        let step_mem = arkStep_mem_mut(&ark_mem);
        (step_mem.gamma, step_mem.sdata.clone().expect("sdata"))
    };
    N_VLinearSum(gamma, &fi_istage, ONE, &sdata, g);

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  arkStep_NlsFPFunction_MassFixed
  arkStep_NlsFPFunction_MassFixed_TrivialPredAutonomous

  This routine evaluates the fixed point iteration function for
  the additive Runge-Kutta method.  It assumes that any data from
  previous time steps/stages is contained in step_mem, and
  merely combines this old data with the current guess and
  implicit ODE RHS vector to compute the iteration function g.

  This version assumes a fixed mass matrix.

  At the ith stage, the new stage solution z should solve:
     M*z = M*yn + h*sum_{j=0}^{i-1} Ae(i,j)*Fe(j)
                + h*sum_{j=0}^{i} Ai(i,j)*Fi(j)
  <=>
     M*z = M*yn + gamma*Fi(z) + h*sum_{j=0}^{i-1} ( Ae(i,j)*Fe(j)
                                                  + Ai(i,j)*Fi(j) )
  <=>
     z = yn + M^{-1}*(gamma*Fi(z) + data)
  <=>
     zc = M^{-1}*(gamma*Fi(zp+zc) + M*yn - M*zp + data)
  Where zp is the predicted stage and zc is the correction to
  the prediction.

  Our fixed-point problem is zc=g(zc), so the FP function is just:
     g(z) = M^{-1}*(gamma*Fi(z) + M*yn - M*zp + data)
  where the current nonlinear guess is z = zp + zc, and where
     z is stored in ycur,
     zp is stored in step_mem->zpred,
     (M*yn-M*zp+data) is stored in step_mem->sdata,
  so we really just compute:
     Fi(z) (store in step_mem->Fi[step_mem->istage])
     g = gamma*Fi(z) + step_mem->sdata
     g = M^{-1}*g

  The "TrivialPredAutonomous" version reuses the implicit RHS
  evaluation at the beginning of the step in the initial FP
  function evaluation.
  ---------------------------------------------------------------*/
pub fn arkStep_NlsFPFunction_MassFixed(
    zcor: &N_Vector,
    g: &N_Vector,
    arkode_mem: &mut Option<Box<dyn Any>>,
) -> i32 {
    /* access ARKodeMem and ARKodeARKStepMem structures */
    let ark_mem = match nlsARKodeMem(arkode_mem) {
        Some(ark_mem) => ark_mem,
        None => {
            arkProcessError(
                None,
                ARK_MEM_NULL,
                line!() as i32,
                "arkStep_NlsFPFunction_MassFixed",
                file!(),
                MSG_ARK_NO_MEM,
            );
            return ARK_MEM_NULL;
        }
    };
    let access = arkStep_AccessARKODEStepMem(&ark_mem, "arkStep_NlsFPFunction_MassFixed");
    if access != ARK_SUCCESS {
        return access;
    }

    /* update 'ycur' value as stored predictor + current corrector */
    let zpred = { arkStep_mem_mut(&ark_mem).zpred.clone() }.expect("zpred");
    let ycur = { ark_mem.borrow().ycur.clone() }.expect("ycur");
    N_VLinearSum(ONE, &zpred, ONE, zcor, &ycur);

    let fi_istage = {
        let step_mem = arkStep_mem_mut(&ark_mem);
        let istage = step_mem.istage as usize;
        step_mem.Fi[istage].clone()
    };
    let retval = arkStep_NlsCallRhs(&ark_mem, &ycur, &fi_istage);
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* combine parts:  g = gamma*Fi(z) + sdata */
    let (gamma, sdata) = {
        let step_mem = arkStep_mem_mut(&ark_mem);
        (step_mem.gamma, step_mem.sdata.clone().expect("sdata"))
    };
    N_VLinearSum(gamma, &fi_istage, ONE, &sdata, g);

    /* perform mass matrix solve */
    let (msolve, nlscoef) = {
        let step_mem = arkStep_mem_mut(&ark_mem);
        (step_mem.msolve.expect("msolve"), step_mem.nlscoef)
    };
    let retval = msolve(&ark_mem, g, nlscoef);
    if retval < 0 {
        return ARK_RHSFUNC_FAIL;
    }
    if retval > 0 {
        return RHSFUNC_RECVR;
    }

    ARK_SUCCESS
}

pub fn arkStep_NlsFPFunction_MassFixed_TrivialPredAutonomous(
    zcor: &N_Vector,
    g: &N_Vector,
    arkode_mem: &mut Option<Box<dyn Any>>,
) -> i32 {
    /* access ARKodeMem and ARKodeARKStepMem structures */
    let ark_mem = match nlsARKodeMem(arkode_mem) {
        Some(ark_mem) => ark_mem,
        None => {
            arkProcessError(
                None,
                ARK_MEM_NULL,
                line!() as i32,
                "arkStep_NlsFPFunction_MassFixed_TrivialPredAutonomous",
                file!(),
                MSG_ARK_NO_MEM,
            );
            return ARK_MEM_NULL;
        }
    };
    let access = arkStep_AccessARKODEStepMem(
        &ark_mem,
        "arkStep_NlsFPFunction_MassFixed_TrivialPredAutonomous",
    );
    if access != ARK_SUCCESS {
        return access;
    }

    /* update 'ycur' value as stored predictor + current corrector */
    let zpred = { arkStep_mem_mut(&ark_mem).zpred.clone() }.expect("zpred");
    let ycur = { ark_mem.borrow().ycur.clone() }.expect("ycur");
    N_VLinearSum(ONE, &zpred, ONE, zcor, &ycur);

    /* compute implicit RHS if not already available */
    let NLS = { arkStep_mem_mut(&ark_mem).NLS.clone() }.expect("NLS");
    let mut nls_iter: i32 = 0;
    let retval = SUNNonlinSolGetCurIter(&NLS, &mut nls_iter);
    if retval != ARK_SUCCESS {
        return ARK_NLS_OP_ERR;
    }

    let (fn_implicit, fi_istage) = {
        let step_mem = arkStep_mem_mut(&ark_mem);
        let istage = step_mem.istage as usize;
        (step_mem.fn_implicit.clone(), step_mem.Fi[istage].clone())
    };
    if nls_iter == 0 && fn_implicit.is_some() {
        N_VScale(ONE, fn_implicit.as_ref().expect("fn_implicit"), &fi_istage);
    } else {
        let retval = arkStep_NlsCallRhs(&ark_mem, &ycur, &fi_istage);
        if retval != ARK_SUCCESS {
            return retval;
        }
    }

    /* combine parts:  g = gamma*Fi(z) + sdata */
    let (gamma, sdata) = {
        let step_mem = arkStep_mem_mut(&ark_mem);
        (step_mem.gamma, step_mem.sdata.clone().expect("sdata"))
    };
    N_VLinearSum(gamma, &fi_istage, ONE, &sdata, g);

    /* perform mass matrix solve */
    let (msolve, nlscoef) = {
        let step_mem = arkStep_mem_mut(&ark_mem);
        (step_mem.msolve.expect("msolve"), step_mem.nlscoef)
    };
    let retval = msolve(&ark_mem, g, nlscoef);
    if retval < 0 {
        return ARK_RHSFUNC_FAIL;
    }
    if retval > 0 {
        return RHSFUNC_RECVR;
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  arkStep_NlsFPFunction_MassTDep:

  This routine evaluates the fixed point iteration function for
  the additive Runge-Kutta method.  It assumes that any data from
  previous time steps/stages is contained in step_mem, and
  merely combines this old data with the current guess and
  implicit ODE RHS vector to compute the iteration function g.

  This version assumes a time-dependent mass matrix.

  At the ith stage, the new stage solution z should solve:
     z = yn + h*sum_{j=0}^{i-1} Ae(i,j)*M(tj)^{-1}*Fe(j)
            + h*sum_{j=0}^{i} Ai(i,j)*M(tj)^{-1}*Fi(j)
  <=>
     z = yn + gamma*M(ti)^{-1}*Fi(z)
            + h*sum_{j=0}^{i-1} ( Ae(i,j)*M(tj)^{-1}*Fe(j)
                                + Ai(i,j)*M(tj)^{-1}*Fi(j) )
  <=>
     z = yn + M(ti)^{-1}*gamma*Fi(z) + data
  <=>
     zc = yn - zp + data + M(ti)^{-1}*gamma*Fi(z)
  Where zp is the predicted stage and zc is the correction to
  the prediction.

  Our fixed-point problem is zc=g(zc), so the FP function is just:
     g(z) = yn - zp + data + M(ti)^{-1}*gamma*Fi(z)
  where the current nonlinear guess is z = zp + zc, and where
     z is stored in ycur,
     zp is stored in step_mem->zpred,
     (yn-zp+data) is stored in step_mem->sdata,
  so we really just compute:
     Fi(z) (store in step_mem->Fi[step_mem->istage])
     g = M(ti)^{-1}*(gamma*Fi(z))
     g = g + step_mem->sdata
  ---------------------------------------------------------------*/
pub fn arkStep_NlsFPFunction_MassTDep(
    zcor: &N_Vector,
    g: &N_Vector,
    arkode_mem: &mut Option<Box<dyn Any>>,
) -> i32 {
    /* access ARKodeMem and ARKodeARKStepMem structures */
    let ark_mem = match nlsARKodeMem(arkode_mem) {
        Some(ark_mem) => ark_mem,
        None => {
            arkProcessError(
                None,
                ARK_MEM_NULL,
                line!() as i32,
                "arkStep_NlsFPFunction_MassTDep",
                file!(),
                MSG_ARK_NO_MEM,
            );
            return ARK_MEM_NULL;
        }
    };
    let access = arkStep_AccessARKODEStepMem(&ark_mem, "arkStep_NlsFPFunction_MassTDep");
    if access != ARK_SUCCESS {
        return access;
    }

    /* update 'ycur' value as stored predictor + current corrector */
    let zpred = { arkStep_mem_mut(&ark_mem).zpred.clone() }.expect("zpred");
    let ycur = { ark_mem.borrow().ycur.clone() }.expect("ycur");
    N_VLinearSum(ONE, &zpred, ONE, zcor, &ycur);

    let fi_istage = {
        let step_mem = arkStep_mem_mut(&ark_mem);
        let istage = step_mem.istage as usize;
        step_mem.Fi[istage].clone()
    };
    let retval = arkStep_NlsCallRhs(&ark_mem, &ycur, &fi_istage);
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* copy step_mem->gamma*Fi into g */
    let gamma = { arkStep_mem_mut(&ark_mem).gamma };
    N_VScale(gamma, &fi_istage, g);

    /* perform mass matrix solve */
    let (msolve, nlscoef) = {
        let step_mem = arkStep_mem_mut(&ark_mem);
        (step_mem.msolve.expect("msolve"), step_mem.nlscoef)
    };
    let retval = msolve(&ark_mem, g, nlscoef);
    if retval < 0 {
        return ARK_RHSFUNC_FAIL;
    }
    if retval > 0 {
        return RHSFUNC_RECVR;
    }

    /* combine parts:  g = g + sdata */
    let sdata = { arkStep_mem_mut(&ark_mem).sdata.clone() }.expect("sdata");
    N_VLinearSum(ONE, g, ONE, &sdata, g);

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  arkStep_NlsConvTest:

  This routine provides the nonlinear solver convergence test for
  the additive Runge-Kutta method.  We have two modes.

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
pub fn arkStep_NlsConvTest(
    NLS: &SUNNonlinearSolver,
    _y: &N_Vector,
    del: &N_Vector,
    tol: sunrealtype,
    ewt: &N_Vector,
    arkode_mem: &mut Option<Box<dyn Any>>,
) -> i32 {
    /* access ARKodeMem and ARKodeARKStepMem structures */
    let ark_mem = match nlsARKodeMem(arkode_mem) {
        Some(ark_mem) => ark_mem,
        None => {
            arkProcessError(
                None,
                ARK_MEM_NULL,
                line!() as i32,
                "arkStep_NlsConvTest",
                file!(),
                MSG_ARK_NO_MEM,
            );
            return ARK_MEM_NULL;
        }
    };
    let access = arkStep_AccessARKODEStepMem(&ark_mem, "arkStep_NlsConvTest");
    if access != ARK_SUCCESS {
        return access;
    }

    /* if the problem is linearly implicit, just return success */
    let linear = arkStep_mem_mut(&ark_mem).linear;
    if linear {
        return SUN_SUCCESS;
    }

    /* compute the norm of the correction (C writes through
    &step_mem->delnrm; ported as copy-out / call / copy-back) */
    let mut delnrm = { arkStep_mem_mut(&ark_mem).delnrm };
    let nrm_retval = arkStep_NlsNorm(del, ewt, &mut delnrm, arkode_mem);
    arkStep_mem_mut(&ark_mem).delnrm = delnrm;
    if nrm_retval != SUN_SUCCESS {
        /* unreachable: arkStep_NlsNorm always succeeds (as in C). The C call
        site passes MSG_ARK_NLS_FAIL with no time argument for its
        MSG_TIME conversion (a varargs bug); the builder is fed tcur. */
        let tcur = ark_mem.borrow().tcur;
        arkProcessError(
            Some(&ark_mem),
            ARK_NLS_OP_ERR,
            line!() as i32,
            "arkStep_NlsConvTest",
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
        let mut step_mem = arkStep_mem_mut(&ark_mem);
        step_mem.crate_ = SUNMAX(
            step_mem.crdown * step_mem.crate_,
            step_mem.delnrm / step_mem.delnrm_p,
        );
    }

    /* compute our scaled error norm for testing convergence */
    let dcon = {
        let step_mem = arkStep_mem_mut(&ark_mem);
        SUNMIN(step_mem.crate_, ONE) * step_mem.delnrm / tol
    };

    /* check for convergence; if so return with success */
    if dcon <= ONE {
        return SUN_SUCCESS;
    }

    /* check for divergence */
    {
        let step_mem = arkStep_mem_mut(&ark_mem);
        if (m >= 1) && (step_mem.delnrm > step_mem.rdiv * step_mem.delnrm_p) {
            return SUN_NLS_CONV_RECVR;
        }
    }

    /* save norm of correction for next iteration */
    {
        let mut step_mem = arkStep_mem_mut(&ark_mem);
        step_mem.delnrm_p = step_mem.delnrm;
    }

    /* return with flag that there is more work to do */
    SUN_NLS_CONTINUE
}

fn arkStep_NlsNorm(
    del: &N_Vector,
    ewt: &N_Vector,
    delnrm: &mut sunrealtype,
    _arkode_mem: &mut Option<Box<dyn Any>>,
) -> SUNErrCode {
    *delnrm = N_VWrmsNorm(del, ewt);
    SUN_SUCCESS
}

fn arkStep_NlsGetUpdateNorm(
    delnrm: &mut sunrealtype,
    arkode_mem: &mut Option<Box<dyn Any>>,
) -> SUNErrCode {
    let ark_mem = match nlsARKodeMem(arkode_mem) {
        Some(ark_mem) => ark_mem,
        None => return SUN_ERR_ARG_CORRUPT,
    };
    let retval = arkStep_AccessStepMem(&ark_mem, "arkStep_NlsGetUpdateNorm");
    if retval != ARK_SUCCESS {
        return SUN_ERR_ARG_CORRUPT;
    }

    *delnrm = arkStep_mem_mut(&ark_mem).delnrm;
    SUN_SUCCESS
}

fn arkStep_NlsGetConvRate(
    crate_: &mut sunrealtype,
    arkode_mem: &mut Option<Box<dyn Any>>,
) -> SUNErrCode {
    let ark_mem = match nlsARKodeMem(arkode_mem) {
        Some(ark_mem) => ark_mem,
        None => return SUN_ERR_ARG_CORRUPT,
    };
    let retval = arkStep_AccessARKODEStepMem(&ark_mem, "arkStep_NlsGetConvRate");
    if retval != ARK_SUCCESS {
        return SUN_ERR_ARG_CORRUPT;
    }

    *crate_ = arkStep_mem_mut(&ark_mem).crate_;
    SUN_SUCCESS
}

/*===============================================================
  EOF
  ===============================================================*/
