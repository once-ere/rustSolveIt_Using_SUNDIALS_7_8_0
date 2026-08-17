//! Port of `src/cvodes/cvodes_nls_sim.c`: the CVODES nonlinear solver
//! interface for the CV_SIMULTANEOUS corrector.
//!
//! When sensitivities are computed using the CV_SIMULTANEOUS approach and the
//! Newton solver is selected the iteraiton is a  quasi-Newton method on the
//! combined system (by approximating the Jacobian matrix by its block diagonal)
//! and thus only solve linear systems with multiple right hand sides (all
//! sharing the same coefficient matrix - whatever iteration matrix we decide on)
//! we set-up the linear solver to handle N equations at a time.
//!
//! The combined state+sensitivity system vector is the
//! `sundials_nvector_senswrapper` vector: slot 0 is the state, slots
//! `1 ..= Ns` are the sensitivities. C's `NV_VECS_SW(v) + 1` pointer
//! arithmetic becomes [`sens_vecs_sw`], which snapshots the subvector
//! handles (`Rc` clones) — equivalent because every use writes THROUGH
//! the handles, never re-assigning wrapper slots.
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

use crate::cvodes::{cvSensRhsWrapper, cvSensUpdateNorm};
use crate::cvodes_impl::*;
use sundials_core::sundials_errors::{SUN_ERR_ARG_CORRUPT, SUN_SUCCESS};
use sundials_core::sundials_math::{SUNMAX, SUNMIN};
use sundials_core::sundials_nonlinearsolver::*;
use sundials_core::sundials_nvector::{
    N_VDestroy, N_VLinearCombinationVectorArray, N_VLinearSum, N_VLinearSumVectorArray, N_VScale,
    N_VWrmsNorm, N_Vector,
};
use sundials_core::sundials_nvector_senswrapper::{
    NV_NVECS_SW, NV_VEC_SW, NV_VEC_SW_set, N_VNewEmpty_SensWrapper,
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

pub fn CVodeSetNonlinearSolverSensSim(cvode_mem: &CVodeMem, NLS: &SUNNonlinearSolver) -> i32 {
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
                "CVodeSetNonlinearSolverSensSim",
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
            "CVodeSetNonlinearSolverSensSim",
            file!(),
            MSGCV_NO_SENSI,
        );
        return CV_ILL_INPUT;
    }

    /* check that simultaneous corrector was selected */
    if cv_mem.borrow().cv_ism != CV_SIMULTANEOUS {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeSetNonlinearSolverSensSim",
            file!(),
            "Sensitivity solution method is not CV_SIMULTANEOUS",
        );
        return CV_ILL_INPUT;
    }

    /* free any existing nonlinear solver */
    let (old_nls, own_nls) = {
        let mut m = cv_mem.borrow_mut();
        (m.NLSsim.take(), m.ownNLSsim)
    };
    if old_nls.is_some() && own_nls {
        /* C stores the result in retval but never reads it */
        let _ = SUNNonlinSolFree(old_nls);
    }

    {
        let mut m = cv_mem.borrow_mut();

        /* set SUNNonlinearSolver pointer */
        m.NLSsim = Some(NLS.clone());

        /* Set NLS ownership flag. If this function was called to attach the default
        NLS, CVODE will set the flag to SUNTRUE after this function returns. */
        m.ownNLSsim = SUNFALSE;
    }

    /* set the nonlinear system function */
    let nls = { cv_mem.borrow().NLSsim.clone() }.expect("NLSsim");
    if SUNNonlinSolGetType(&nls) == SUNNONLINEARSOLVER_ROOTFIND {
        retval = SUNNonlinSolSetSysFn(&nls, Some(cvNlsResidualSensSim));
    } else if SUNNonlinSolGetType(&nls) == SUNNONLINEARSOLVER_FIXEDPOINT {
        retval = SUNNonlinSolSetSysFn(&nls, Some(cvNlsFPFunctionSensSim));
    } else {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeSetNonlinearSolverSensSim",
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
            "CVodeSetNonlinearSolverSensSim",
            file!(),
            "Setting nonlinear system function failed",
        );
        return CV_ILL_INPUT;
    }

    /* set convergence test function */
    retval =
        SUNNonlinSolSetConvTestFn(&nls, Some(cvNlsConvTestSensSim), Some(Box::new(cv_mem.clone())));
    if retval != CV_SUCCESS {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeSetNonlinearSolverSensSim",
            file!(),
            "Setting convergence test function failed",
        );
        return CV_ILL_INPUT;
    }

    retval = SUNNonlinSolSetNormFn(&nls, Some(cvNlsNormSensSim), Some(Box::new(cv_mem.clone())));
    if retval != CV_SUCCESS {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeSetNonlinearSolverSensSim",
            file!(),
            "Setting convergence-test norm function failed",
        );
        return CV_ILL_INPUT;
    }

    retval = SUNNonlinSolSetGetUpdateNormFn(
        &nls,
        Some(cvNlsGetUpdateNormSensSim),
        Some(Box::new(cv_mem.clone())),
    );
    if retval != CV_SUCCESS {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeSetNonlinearSolverSensSim",
            file!(),
            "Setting update-norm getter failed",
        );
        return CV_ILL_INPUT;
    }

    retval = SUNNonlinSolSetGetConvRateFn(
        &nls,
        Some(cvNlsGetConvRateSensSim),
        Some(Box::new(cv_mem.clone())),
    );
    if retval != CV_SUCCESS {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeSetNonlinearSolverSensSim",
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
            "CVodeSetNonlinearSolverSensSim",
            file!(),
            "Setting maximum number of nonlinear iterations failed",
        );
        return CV_ILL_INPUT;
    }

    /* create vector wrappers if necessary */
    let simMallocDone = cv_mem.borrow().simMallocDone;
    if simMallocDone == SUNFALSE {
        let (Ns, sunctx) = {
            let m = cv_mem.borrow();
            (m.cv_Ns, m.cv_sunctx.clone())
        };

        let zn0Sim = N_VNewEmpty_SensWrapper(Ns + 1, &sunctx);
        let zn0Sim_null = zn0Sim.is_none();
        cv_mem.borrow_mut().zn0Sim = zn0Sim;
        if zn0Sim_null {
            cvProcessError(
                Some(cv_mem),
                CV_MEM_FAIL,
                line!() as i32,
                "CVodeSetNonlinearSolverSensSim",
                file!(),
                MSGCV_MEM_FAIL,
            );
            return CV_MEM_FAIL;
        }

        let ycorSim = N_VNewEmpty_SensWrapper(Ns + 1, &sunctx);
        let ycorSim_null = ycorSim.is_none();
        cv_mem.borrow_mut().ycorSim = ycorSim;
        if ycorSim_null {
            /* C leaves the destroyed handle in the mem; the port takes it out */
            let v = cv_mem.borrow_mut().zn0Sim.take();
            if let Some(v) = v {
                N_VDestroy(v);
            }
            cvProcessError(
                Some(cv_mem),
                CV_MEM_FAIL,
                line!() as i32,
                "CVodeSetNonlinearSolverSensSim",
                file!(),
                MSGCV_MEM_FAIL,
            );
            return CV_MEM_FAIL;
        }

        let ewtSim = N_VNewEmpty_SensWrapper(Ns + 1, &sunctx);
        let ewtSim_null = ewtSim.is_none();
        cv_mem.borrow_mut().ewtSim = ewtSim;
        if ewtSim_null {
            let v = cv_mem.borrow_mut().zn0Sim.take();
            if let Some(v) = v {
                N_VDestroy(v);
            }
            let v = cv_mem.borrow_mut().ycorSim.take();
            if let Some(v) = v {
                N_VDestroy(v);
            }
            cvProcessError(
                Some(cv_mem),
                CV_MEM_FAIL,
                line!() as i32,
                "CVodeSetNonlinearSolverSensSim",
                file!(),
                MSGCV_MEM_FAIL,
            );
            return CV_MEM_FAIL;
        }

        cv_mem.borrow_mut().simMallocDone = SUNTRUE;
    }

    /* attach vectors to vector wrappers */
    let (zn0Sim, ycorSim, ewtSim, zn0, acor, ewt, Ns) = {
        let m = cv_mem.borrow();
        (
            m.zn0Sim.as_ref().expect("zn0Sim").clone(),
            m.ycorSim.as_ref().expect("ycorSim").clone(),
            m.ewtSim.as_ref().expect("ewtSim").clone(),
            m.cv_zn[0].clone(),
            m.cv_acor.clone(),
            m.cv_ewt.clone(),
            m.cv_Ns,
        )
    };
    NV_VEC_SW_set(&zn0Sim, 0, zn0);
    NV_VEC_SW_set(&ycorSim, 0, acor);
    NV_VEC_SW_set(&ewtSim, 0, ewt);

    for is in 0..Ns {
        let (znS0_is, acorS_is, ewtS_is) = {
            let m = cv_mem.borrow();
            (
                m.cv_znS[0][is as usize].clone(),
                m.cv_acorS[is as usize].clone(),
                m.cv_ewtS[is as usize].clone(),
            )
        };
        NV_VEC_SW_set(&zn0Sim, is + 1, Some(znS0_is));
        NV_VEC_SW_set(&ycorSim, is + 1, Some(acorS_is));
        NV_VEC_SW_set(&ewtSim, is + 1, Some(ewtS_is));
    }

    /* Reset the acnrmcur flag to SUNFALSE */
    cv_mem.borrow_mut().cv_acnrmcur = SUNFALSE;

    /* Set the nonlinear system RHS function */
    if cv_mem.borrow().cv_f.is_none() {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeSetNonlinearSolverSensSim",
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
  CVodeGetNonlinearSystemDataSens:

  This routine provides access to the relevant data needed to
  compute the nonlinear system function.

  The C `N_Vector**` out-parameters hand out the internal vector
  ARRAYS; the port fills the caller's `Vec` with `Rc` clones of the
  same vectors (writes through them reach the integrator's storage,
  exactly as in C). `user_data` follows the SWAP convention of
  `CVodeGetUserData` — the caller must hand the box back before the
  integrator next invokes a user callback.
  ---------------------------------------------------------------*/
#[allow(clippy::too_many_arguments)]
pub fn CVodeGetNonlinearSystemDataSens(
    cvode_mem: &CVodeMem,
    tcur: &mut sunrealtype,
    ySpred: &mut Vec<N_Vector>,
    ySn: &mut Vec<N_Vector>,
    gamma: &mut sunrealtype,
    rl1: &mut sunrealtype,
    znS1: &mut Vec<N_Vector>,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    /* NULL-mem check: handled by the type system */
    let mut cv_mem = cvode_mem.borrow_mut();

    *tcur = cv_mem.cv_tn;
    *ySpred = cv_mem.cv_znS[0].clone();
    *ySn = cv_mem.cv_yS.clone();
    *gamma = cv_mem.cv_gamma;
    *rl1 = cv_mem.cv_rl1;
    *znS1 = cv_mem.cv_znS[1].clone();
    /* C copies the raw pointer; the box is swapped out instead */
    std::mem::swap(&mut cv_mem.cv_user_data, user_data);

    CV_SUCCESS
}

/* -----------------------------------------------------------------------------
 * Private functions
 * ---------------------------------------------------------------------------*/

pub fn cvNlsInitSensSim(cvode_mem: &CVodeMem) -> i32 {
    let mut retval: i32;

    let nls = { cvode_mem.borrow().NLSsim.clone() }.expect("NLSsim");

    /* set the linear solver setup wrapper function */
    let has_lsetup = cvode_mem.borrow().cv_lsetup.is_some();
    if has_lsetup {
        retval = SUNNonlinSolSetLSetupFn(&nls, Some(cvNlsLSetupSensSim));
    } else {
        retval = SUNNonlinSolSetLSetupFn(&nls, None);
    }

    if retval != CV_SUCCESS {
        cvProcessError(
            Some(cvode_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "cvNlsInitSensSim",
            file!(),
            "Setting the linear solver setup function failed",
        );
        return CV_NLS_INIT_FAIL;
    }

    /* set the linear solver solve wrapper function */
    let has_lsolve = cvode_mem.borrow().cv_lsolve.is_some();
    if has_lsolve {
        retval = SUNNonlinSolSetLSolveFn(&nls, Some(cvNlsLSolveSensSim));
    } else {
        retval = SUNNonlinSolSetLSolveFn(&nls, None);
    }

    if retval != CV_SUCCESS {
        cvProcessError(
            Some(cvode_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "cvNlsInitSensSim",
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
            "cvNlsInitSensSim",
            file!(),
            MSGCV_NLS_INIT_FAIL,
        );
        return CV_NLS_INIT_FAIL;
    }

    CV_SUCCESS
}

fn cvNlsLSetupSensSim(
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
                "cvNlsLSetupSensSim",
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

fn cvNlsLSolveSensSim(deltaSim: &N_Vector, cvode_mem: &mut Option<Box<dyn Any>>) -> i32 {
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
                "cvNlsLSolveSensSim",
                file!(),
                MSGCV_NO_MEM,
            );
            return CV_MEM_NULL;
        }
    };

    /* extract state delta from the vector wrapper */
    let delta = NV_VEC_SW(deltaSim, 0);

    /* solve the state linear system */
    let (lsolve, ewt, y, ftemp);
    {
        let m = cv_mem.borrow();
        lsolve = m.cv_lsolve.expect("cv_lsolve");
        ewt = m.cv_ewt.as_ref().expect("cv_ewt").clone();
        y = m.cv_y.as_ref().expect("cv_y").clone();
        ftemp = m.cv_ftemp.as_ref().expect("cv_ftemp").clone();
    }
    let retval = lsolve(&cv_mem, &delta, &ewt, &y, &ftemp);

    if retval < 0 {
        return CV_LSOLVE_FAIL;
    }
    if retval > 0 {
        return SUN_NLS_CONV_RECVR;
    }

    /* extract sensitivity deltas from the vector wrapper */
    let deltaS = sens_vecs_sw(deltaSim);

    /* solve the sensitivity linear systems */
    let Ns = cv_mem.borrow().cv_Ns;
    for is in 0..Ns as usize {
        let (lsolve, ewtS_is, y, ftemp);
        {
            let m = cv_mem.borrow();
            lsolve = m.cv_lsolve.expect("cv_lsolve");
            ewtS_is = m.cv_ewtS[is].clone();
            y = m.cv_y.as_ref().expect("cv_y").clone();
            ftemp = m.cv_ftemp.as_ref().expect("cv_ftemp").clone();
        }
        let retval = lsolve(&cv_mem, &deltaS[is], &ewtS_is, &y, &ftemp);

        if retval < 0 {
            return CV_LSOLVE_FAIL;
        }
        if retval > 0 {
            return SUN_NLS_CONV_RECVR;
        }
    }

    CV_SUCCESS
}

fn cvNlsConvTestSensSim(
    NLS: &SUNNonlinearSolver,
    ycorSim: &N_Vector,
    deltaSim: &N_Vector,
    tol: sunrealtype,
    ewtSim: &N_Vector,
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
                "cvNlsConvTestSensSim",
                file!(),
                MSGCV_NO_MEM,
            );
            return CV_MEM_NULL;
        }
    };

    /* extract the current state and sensitivity corrections */
    let ycor = NV_VEC_SW(ycorSim, 0);

    /* extract the current error weight vector */
    let ewt = NV_VEC_SW(ewtSim, 0);

    /* compute the norm of the sensitivity corrections (C writes through
    &cv_mem->cv_delnrmS; ported as copy-out / call / copy-back) */
    let mut delnrmS = cv_mem.borrow().cv_delnrmS;
    let nrm_retval = cvNlsNormSensSim(deltaSim, ewtSim, &mut delnrmS, cvode_mem);
    cv_mem.borrow_mut().cv_delnrmS = delnrmS;
    if nrm_retval != SUN_SUCCESS {
        /* the C call site passes no time argument for the MSG_TIME
        format — use cv_tn */
        let tn = cv_mem.borrow().cv_tn;
        cvProcessError(
            Some(&cv_mem),
            CV_NLS_FAIL,
            line!() as i32,
            "cvNlsConvTestSensSim",
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
       convergence test. Hence, we use cv_delnrmS (and not cv_delnrm). However, acnrm is
       used in the error test and thus it has different forms depending on
       errconS (and this explains why we still carry around cv_delnrm).
    */
    let dcon: sunrealtype;
    {
        let mut mm = cv_mem.borrow_mut();
        if m > 0 {
            mm.cv_crate = SUNMAX(CRDOWN * mm.cv_crate, mm.cv_delnrmS / mm.cv_delp);
        }
        dcon = mm.cv_delnrmS * SUNMIN(ONE, mm.cv_crate) / tol;
    }

    /* check if nonlinear system was solved successfully */
    if dcon <= ONE {
        if m == 0 {
            let mut mm = cv_mem.borrow_mut();
            mm.cv_acnrm = if mm.cv_errconS {
                mm.cv_delnrmS
            } else {
                mm.cv_delnrm
            };
        } else {
            let errconS = cv_mem.borrow().cv_errconS;
            let acnrm = if errconS {
                N_VWrmsNorm(ycorSim, ewtSim)
            } else {
                N_VWrmsNorm(&ycor, &ewt)
            };
            cv_mem.borrow_mut().cv_acnrm = acnrm;
        }
        cv_mem.borrow_mut().cv_acnrmcur = SUNTRUE;
        return CV_SUCCESS;
    }

    /* check if the iteration seems to be diverging */
    {
        let mm = cv_mem.borrow();
        if (m >= 1) && (mm.cv_delnrmS > RDIV * mm.cv_delp) {
            return SUN_NLS_CONV_RECVR;
        }
    }

    /* Save norm of correction and loop again */
    {
        let mut mm = cv_mem.borrow_mut();
        mm.cv_delp = mm.cv_delnrmS;
    }

    /* Not yet converged */
    SUN_NLS_CONTINUE
}

fn cvNlsNormSensSim(
    deltaSim: &N_Vector,
    ewtSim: &N_Vector,
    delnrmS: &mut sunrealtype,
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

    /* extract state and sensitivity deltas */
    let delta = NV_VEC_SW(deltaSim, 0);
    let deltaS = sens_vecs_sw(deltaSim);

    /* extract state and sensitivity error weights */
    let ewt = NV_VEC_SW(ewtSim, 0);
    let ewtS = sens_vecs_sw(ewtSim);

    /* compute and save the norm of the state corrections */
    let delnrm = N_VWrmsNorm(&delta, &ewt);
    cv_mem.borrow_mut().cv_delnrm = delnrm;

    /* compute the norm of the sensitivity corrections */
    let nrmS = cvSensUpdateNorm(&cv_mem, delnrm, &deltaS, &ewtS);
    cv_mem.borrow_mut().cv_delnrmS = nrmS;

    *delnrmS = nrmS;

    SUN_SUCCESS
}

fn cvNlsGetUpdateNormSensSim(
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

    *delnrm = cv_mem.borrow().cv_delnrmS;
    SUN_SUCCESS
}

fn cvNlsGetConvRateSensSim(
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

    *crate_ = cv_mem.borrow().cv_crate;
    SUN_SUCCESS
}

fn cvNlsResidualSensSim(
    ycorSim: &N_Vector,
    resSim: &N_Vector,
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
                "cvNlsResidualSensSim",
                file!(),
                MSGCV_NO_MEM,
            );
            return CV_MEM_NULL;
        }
    };

    /* extract state and residual vectors from the vector wrapper */
    let ycor = NV_VEC_SW(ycorSim, 0);
    let res = NV_VEC_SW(resSim, 0);

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
    N_VLinearSum(ONE, &zn0, ONE, &ycor, &y);

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
    N_VLinearSum(rl1, &zn1, ONE, &ycor, &res);
    N_VLinearSum(-gamma, &ftemp, ONE, &res, &res);

    /* extract sensitivity and residual vectors from the vector wrapper */
    let ycorS = sens_vecs_sw(ycorSim);
    let resS = sens_vecs_sw(resSim);

    /* update sensitivities based on the current correction */
    let (Ns, znS0, yS);
    {
        let m = cv_mem.borrow();
        Ns = m.cv_Ns;
        znS0 = m.cv_znS[0].clone();
        yS = m.cv_yS.clone();
    }
    let retval = N_VLinearSumVectorArray(Ns, ONE, &znS0, ONE, &ycorS, &yS);
    if retval != CV_SUCCESS {
        return CV_VECTOROP_ERR;
    }

    /* evaluate the sensitivity rhs function */
    let (ftempS, vtemp1, vtemp2);
    {
        let m = cv_mem.borrow();
        ftempS = m.cv_ftempS.clone();
        vtemp1 = m.cv_vtemp1.as_ref().expect("cv_vtemp1").clone();
        vtemp2 = m.cv_vtemp2.as_ref().expect("cv_vtemp2").clone();
    }
    let retval = cvSensRhsWrapper(&cv_mem, tn, &y, &ftemp, &yS, &ftempS, &vtemp1, &vtemp2);

    if retval < 0 {
        return CV_SRHSFUNC_FAIL;
    }
    if retval > 0 {
        return SRHSFUNC_RECVR;
    }

    /* compute the sensitivity resiudal
       (C: cvals[0]=rl1, XXvecs[0]=znS[1]; cvals[1]=ONE, XXvecs[1]=ycorS;
           cvals[2]=-gamma, XXvecs[2]=ftempS) */
    let (rl1, gamma, znS1);
    {
        let m = cv_mem.borrow();
        rl1 = m.cv_rl1;
        gamma = m.cv_gamma;
        znS1 = m.cv_znS[1].clone();
    }
    let cvals: [sunrealtype; 3] = [rl1, ONE, -gamma];
    let XXvecs: [Vec<N_Vector>; 3] = [znS1, ycorS, ftempS];

    let retval = N_VLinearCombinationVectorArray(Ns, 3, &cvals, &XXvecs, &resS);
    if retval != CV_SUCCESS {
        return CV_VECTOROP_ERR;
    }

    CV_SUCCESS
}

fn cvNlsFPFunctionSensSim(
    ycorSim: &N_Vector,
    resSim: &N_Vector,
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
                "cvNlsFPFunctionSensSim",
                file!(),
                MSGCV_NO_MEM,
            );
            return CV_MEM_NULL;
        }
    };

    /* extract state and residual vectors from the vector wrapper */
    let ycor = NV_VEC_SW(ycorSim, 0);
    let res = NV_VEC_SW(resSim, 0);

    let (zn0, y, tn, nls_f);
    {
        let m = cv_mem.borrow();
        zn0 = m.cv_zn[0].as_ref().expect("cv_zn[0]").clone();
        y = m.cv_y.as_ref().expect("cv_y").clone();
        tn = m.cv_tn;
        nls_f = m.nls_f.expect("nls_f");
    }

    /* update the state based on the current correction */
    N_VLinearSum(ONE, &zn0, ONE, &ycor, &y);

    /* evaluate the rhs function */
    let mut user_data = cv_mem.borrow_mut().cv_user_data.take();
    let retval = nls_f(tn, &y, &res, &mut user_data);
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

    /* evaluate fixed point function */
    let (h, rl1, zn1);
    {
        let m = cv_mem.borrow();
        h = m.cv_h;
        rl1 = m.cv_rl1;
        zn1 = m.cv_zn[1].as_ref().expect("cv_zn[1]").clone();
    }
    N_VLinearSum(h, &res, -ONE, &zn1, &res);
    N_VScale(rl1, &res, &res);

    /* extract sensitivity and residual vectors from the vector wrapper */
    let ycorS = sens_vecs_sw(ycorSim);
    let resS = sens_vecs_sw(resSim);

    /* update the sensitivities based on the current correction */
    let (Ns, znS0, yS);
    {
        let m = cv_mem.borrow();
        Ns = m.cv_Ns;
        znS0 = m.cv_znS[0].clone();
        yS = m.cv_yS.clone();
    }
    /* C discards the return value here */
    let _ = N_VLinearSumVectorArray(Ns, ONE, &znS0, ONE, &ycorS, &yS);

    /* evaluate the sensitivity rhs function */
    let (vtemp1, vtemp2);
    {
        let m = cv_mem.borrow();
        vtemp1 = m.cv_vtemp1.as_ref().expect("cv_vtemp1").clone();
        vtemp2 = m.cv_vtemp2.as_ref().expect("cv_vtemp2").clone();
    }
    let retval = cvSensRhsWrapper(&cv_mem, tn, &y, &res, &yS, &resS, &vtemp1, &vtemp2);

    if retval < 0 {
        return CV_SRHSFUNC_FAIL;
    }
    if retval > 0 {
        return SRHSFUNC_RECVR;
    }

    /* evaluate sensitivity fixed point function */
    for is in 0..Ns as usize {
        let (h, rl1, znS1_is);
        {
            let m = cv_mem.borrow();
            h = m.cv_h;
            rl1 = m.cv_rl1;
            znS1_is = m.cv_znS[1][is].clone();
        }
        N_VLinearSum(h, &resS[is], -ONE, &znS1_is, &resS[is]);
        N_VScale(rl1, &resS[is], &resS[is]);
    }

    CV_SUCCESS
}
