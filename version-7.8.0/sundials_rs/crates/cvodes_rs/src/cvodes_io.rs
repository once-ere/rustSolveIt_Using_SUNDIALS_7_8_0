//! Port of `src/cvodes/cvodes_io.c` (+ headers folded) — the optional
//! input and output functions for the CVODES solver, including the
//! quadrature, forward-sensitivity and quadrature-sensitivity getters
//! and setters.
//!
//! Reference build configuration:
//! - `SUNDIALS_ENABLE_MONITORING` is DEFINED: `CVodeSetMonitorFn` /
//!   `CVodeSetMonitorFrequency` port the enabled branch (the `#else`
//!   error branch is dead code).
//! - `SUNDIALS_LOGGING_LEVEL = 2`, profiling OFF, error checks OFF.
//!
//! `void*` out-params (`CVodeGetUserData` here; `CVodeGetNonlinearSystemData`
//! in `cvodes_nls.rs` follows the same convention): C returns the stored
//! pointer without ownership transfer; the safe-Rust `Option<Box<dyn Any>>`
//! token cannot be aliased, so these functions SWAP the stored token with the
//! caller's out-param. Callers must hand the box back (via `CVodeSetUserData`
//! or a second swap) before the integrator next invokes a user callback.
//!
//! `CVodeSetSensParams` stores the caller's parameter array as a SHARED
//! handle (`cv_p: Option<SensParams>` = `Option<Rc<RefCell<Vec<…>>>>`),
//! mirroring C's stored pointer, so the internal DQ routines' in-place
//! perturbations are visible to the user callbacks — see the note on that
//! function.

use std::any::Any;

use sundials_core::sundials_math::{SUNRabs, SUNMIN};
use sundials_core::sundials_nonlinearsolver::*;
use sundials_core::sundials_nvector::*;
use sundials_core::sundials_types::*;
use sundials_core::sundials_utils::*;

use crate::cvodes_impl::*;
use crate::cvodes_ls::CVLsMemRec;

const ZERO: sunrealtype = 0.0;
const HALF: sunrealtype = 0.5;
const ONE: sunrealtype = 1.0;
const TWOPT5: sunrealtype = 2.5;

/*
 * =================================================================
 * CVODES optional input functions
 * =================================================================
 */

/*
 * CVodeSetDeltaGammaMaxLSetup
 *
 * Specifies the gamma ratio threshold to signal for a linear solver setup
 */

pub fn CVodeSetDeltaGammaMaxLSetup(cvode_mem: &CVodeMem, dgmax_lsetup: sunrealtype) -> i32 {
    /* NULL-mem check: handled by type system */
    let mut cv_mem = cvode_mem.borrow_mut();

    /* Set value or use default */
    if dgmax_lsetup < ZERO {
        cv_mem.cv_dgmax_lsetup = DGMAX_LSETUP_DEFAULT;
    } else {
        cv_mem.cv_dgmax_lsetup = dgmax_lsetup;
    }

    CV_SUCCESS
}

/*
 * CVodeSetUserData
 *
 * Specifies the user data pointer for f
 */

pub fn CVodeSetUserData(cvode_mem: &CVodeMem, user_data: Option<Box<dyn Any>>) -> i32 {
    /* NULL-mem check: handled by type system */
    cvode_mem.borrow_mut().cv_user_data = user_data;

    CV_SUCCESS
}

/*
 * CVodeSetMonitorFn
 *
 * Specifies the user function to call for monitoring
 * the solution and/or integrator statistics.
 */

pub fn CVodeSetMonitorFn(cvode_mem: &CVodeMem, fn_: Option<CVMonitorFn>) -> i32 {
    /* NULL-mem check: handled by type system */
    /* SUNDIALS_ENABLE_MONITORING is defined in the reference build */
    cvode_mem.borrow_mut().cv_monitorfun = fn_;
    CV_SUCCESS
}

/*
 * CVodeSetMonitorFrequency
 *
 * Specifies the frequency with which to call the user function.
 */

pub fn CVodeSetMonitorFrequency(cvode_mem: &CVodeMem, nst: i64) -> i32 {
    /* NULL-mem check: handled by type system */
    if nst < 0 {
        cvProcessError(
            None,
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeSetMonitorFrequency",
            file!(),
            "step interval must be >= 0\n",
        );
        return CV_ILL_INPUT;
    }

    /* SUNDIALS_ENABLE_MONITORING is defined in the reference build */
    cvode_mem.borrow_mut().cv_monitor_interval = nst;
    CV_SUCCESS
}

/*
 * CVodeSetMaxOrd
 *
 * Specifies the maximum method order
 */

pub fn CVodeSetMaxOrd(cvode_mem: &CVodeMem, maxord: i32) -> i32 {
    /* NULL-mem check: handled by type system */
    if maxord <= 0 {
        cvProcessError(
            Some(cvode_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeSetMaxOrd",
            file!(),
            MSGCV_NEG_MAXORD,
        );
        return CV_ILL_INPUT;
    }

    /* Cannot increase maximum order beyond the value that
    was used when allocating memory */
    let qmax_alloc = {
        let cv_mem = cvode_mem.borrow();
        let mut qmax_alloc = cv_mem.cv_qmax_alloc;
        qmax_alloc = SUNMIN(qmax_alloc, cv_mem.cv_qmax_allocQ);
        qmax_alloc = SUNMIN(qmax_alloc, cv_mem.cv_qmax_allocS);
        qmax_alloc
    };

    if maxord > qmax_alloc {
        cvProcessError(
            Some(cvode_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeSetMaxOrd",
            file!(),
            MSGCV_BAD_MAXORD,
        );
        return CV_ILL_INPUT;
    }

    cvode_mem.borrow_mut().cv_qmax = maxord;

    CV_SUCCESS
}

/*
 * CVodeSetMaxNumSteps
 *
 * Specifies the maximum number of integration steps
 */

pub fn CVodeSetMaxNumSteps(cvode_mem: &CVodeMem, mxsteps: i64) -> i32 {
    /* NULL-mem check: handled by type system */
    let mut cv_mem = cvode_mem.borrow_mut();

    /* Passing mxsteps=0 sets the default. Passing mxsteps<0 disables the test. */
    if mxsteps == 0 {
        cv_mem.cv_mxstep = MXSTEP_DEFAULT;
    } else {
        cv_mem.cv_mxstep = mxsteps;
    }

    CV_SUCCESS
}

/*
 * CVodeSetMaxHnilWarns
 *
 * Specifies the maximum number of warnings for small h
 */

pub fn CVodeSetMaxHnilWarns(cvode_mem: &CVodeMem, mxhnil: i32) -> i32 {
    /* NULL-mem check: handled by type system */
    cvode_mem.borrow_mut().cv_mxhnil = mxhnil;

    CV_SUCCESS
}

/*
 *CVodeSetStabLimDet
 *
 * Turns on/off the stability limit detection algorithm
 */

pub fn CVodeSetStabLimDet(cvode_mem: &CVodeMem, sldet: sunbooleantype) -> i32 {
    /* NULL-mem check: handled by type system */
    let lmm = cvode_mem.borrow().cv_lmm;
    if sldet && (lmm != CV_BDF) {
        cvProcessError(
            Some(cvode_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeSetStabLimDet",
            file!(),
            MSGCV_SET_SLDET,
        );
        return CV_ILL_INPUT;
    }

    cvode_mem.borrow_mut().cv_sldeton = sldet;

    CV_SUCCESS
}

/*
 * CVodeSetInitStep
 *
 * Specifies the initial step size
 */

pub fn CVodeSetInitStep(cvode_mem: &CVodeMem, hin: sunrealtype) -> i32 {
    /* NULL-mem check: handled by type system */
    cvode_mem.borrow_mut().cv_hin = hin;

    CV_SUCCESS
}

/*
 * CVodeSetMinStep
 *
 * Specifies the minimum step size
 */

pub fn CVodeSetMinStep(cvode_mem: &CVodeMem, hmin: sunrealtype) -> i32 {
    /* NULL-mem check: handled by type system */
    if hmin < ZERO {
        cvProcessError(
            Some(cvode_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeSetMinStep",
            file!(),
            MSGCV_NEG_HMIN,
        );
        return CV_ILL_INPUT;
    }

    /* Passing 0 sets hmin = zero */
    if hmin == ZERO {
        cvode_mem.borrow_mut().cv_hmin = HMIN_DEFAULT;
        return CV_SUCCESS;
    }

    let hmax_inv = cvode_mem.borrow().cv_hmax_inv;
    if hmin * hmax_inv > ONE {
        cvProcessError(
            Some(cvode_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeSetMinStep",
            file!(),
            MSGCV_BAD_HMIN_HMAX,
        );
        return CV_ILL_INPUT;
    }

    cvode_mem.borrow_mut().cv_hmin = hmin;

    CV_SUCCESS
}

/*
 * CVodeSetMaxStep
 *
 * Specifies the maximum step size
 */

pub fn CVodeSetMaxStep(cvode_mem: &CVodeMem, hmax: sunrealtype) -> i32 {
    /* NULL-mem check: handled by type system */
    if hmax < ZERO {
        cvProcessError(
            Some(cvode_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeSetMaxStep",
            file!(),
            MSGCV_NEG_HMAX,
        );
        return CV_ILL_INPUT;
    }

    /* Passing 0 sets hmax = infinity */
    if hmax == ZERO {
        cvode_mem.borrow_mut().cv_hmax_inv = HMAX_INV_DEFAULT;
        return CV_SUCCESS;
    }

    let hmax_inv = ONE / hmax;
    let hmin = cvode_mem.borrow().cv_hmin;
    if hmax_inv * hmin > ONE {
        cvProcessError(
            Some(cvode_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeSetMaxStep",
            file!(),
            MSGCV_BAD_HMIN_HMAX,
        );
        return CV_ILL_INPUT;
    }

    cvode_mem.borrow_mut().cv_hmax_inv = hmax_inv;

    CV_SUCCESS
}

/*
 * CVodeSetEtaFixedStepBounds
 *
 * Specifies the bounds for retaining the current step size
 */

pub fn CVodeSetEtaFixedStepBounds(
    cvode_mem: &CVodeMem,
    eta_min_fx: sunrealtype,
    eta_max_fx: sunrealtype,
) -> i32 {
    /* NULL-mem check: handled by type system */
    let mut cv_mem = cvode_mem.borrow_mut();

    /* set allowed value or use default */
    if eta_min_fx >= ZERO && eta_min_fx <= ONE {
        cv_mem.cv_eta_min_fx = eta_min_fx;
    } else {
        cv_mem.cv_eta_min_fx = ETA_MIN_FX_DEFAULT;
    }

    if eta_max_fx >= ONE {
        cv_mem.cv_eta_max_fx = eta_max_fx;
    } else {
        cv_mem.cv_eta_max_fx = ETA_MAX_FX_DEFAULT;
    }

    CV_SUCCESS
}

/*
 * CVodeSetEtaMaxFirstStep
 *
 * Specifies the maximum step size change on the first step
 */

pub fn CVodeSetEtaMaxFirstStep(cvode_mem: &CVodeMem, eta_max_fs: sunrealtype) -> i32 {
    /* NULL-mem check: handled by type system */
    let mut cv_mem = cvode_mem.borrow_mut();

    /* set allowed value or use default */
    if eta_max_fs <= ONE {
        cv_mem.cv_eta_max_fs = ETA_MAX_FS_DEFAULT;
    } else {
        cv_mem.cv_eta_max_fs = eta_max_fs;
    }

    CV_SUCCESS
}

/*
 * CVodeSetEtaMaxEarlyStep
 *
 * Specifies the maximum step size change on steps early in the integration
 * when nst <= small_nst
 */

pub fn CVodeSetEtaMaxEarlyStep(cvode_mem: &CVodeMem, eta_max_es: sunrealtype) -> i32 {
    /* NULL-mem check: handled by type system */
    let mut cv_mem = cvode_mem.borrow_mut();

    /* set allowed value or use default */
    if eta_max_es <= ONE {
        cv_mem.cv_eta_max_es = ETA_MAX_ES_DEFAULT;
    } else {
        cv_mem.cv_eta_max_es = eta_max_es;
    }

    CV_SUCCESS
}

/*
 * CVodeSetNumStepsEtaMaxEarlyStep
 *
 * Specifies the maximum number of steps for using the early integration change
 * factor
 */

pub fn CVodeSetNumStepsEtaMaxEarlyStep(cvode_mem: &CVodeMem, small_nst: i64) -> i32 {
    /* NULL-mem check: handled by type system */
    let mut cv_mem = cvode_mem.borrow_mut();

    /* set allowed value or use default */
    if small_nst < 0 {
        cv_mem.cv_small_nst = SMALL_NST_DEFAULT;
    } else {
        cv_mem.cv_small_nst = small_nst;
    }

    CV_SUCCESS
}

/*
 * CVodeSetEtaMax
 *
 * Specifies the maximum step size change on a general steps (nst > small_nst)
 */

pub fn CVodeSetEtaMax(cvode_mem: &CVodeMem, eta_max_gs: sunrealtype) -> i32 {
    /* NULL-mem check: handled by type system */
    let mut cv_mem = cvode_mem.borrow_mut();

    /* set allowed value or use default */
    if eta_max_gs <= ONE {
        cv_mem.cv_eta_max_gs = ETA_MAX_GS_DEFAULT;
    } else {
        cv_mem.cv_eta_max_gs = eta_max_gs;
    }

    CV_SUCCESS
}

/*
 * CVodeSetEtaMin
 *
 * Specifies the minimum change on a general steps
 */

pub fn CVodeSetEtaMin(cvode_mem: &CVodeMem, eta_min: sunrealtype) -> i32 {
    /* NULL-mem check: handled by type system */
    let mut cv_mem = cvode_mem.borrow_mut();

    /* set allowed value or use default */
    if eta_min <= ZERO || eta_min >= ONE {
        cv_mem.cv_eta_min = ETA_MIN_DEFAULT;
    } else {
        cv_mem.cv_eta_min = eta_min;
    }

    CV_SUCCESS
}

/*
 * CVodeSetEtaMinErrFail
 *
 * Specifies the minimum step size change after an error test failure
 */

pub fn CVodeSetEtaMinErrFail(cvode_mem: &CVodeMem, eta_min_ef: sunrealtype) -> i32 {
    /* NULL-mem check: handled by type system */
    let mut cv_mem = cvode_mem.borrow_mut();

    /* set allowed value or use default */
    if eta_min_ef <= ZERO || eta_min_ef >= ONE {
        cv_mem.cv_eta_min_ef = ETA_MIN_EF_DEFAULT;
    } else {
        cv_mem.cv_eta_min_ef = eta_min_ef;
    }

    CV_SUCCESS
}

/*
 * CVodeSetEtaMaxErrFail
 *
 * Specifies the maximum step size change after multiple (>= small_nef) error
 * test failures
 */

pub fn CVodeSetEtaMaxErrFail(cvode_mem: &CVodeMem, eta_max_ef: sunrealtype) -> i32 {
    /* NULL-mem check: handled by type system */
    let mut cv_mem = cvode_mem.borrow_mut();

    /* set allowed value or use default */
    if eta_max_ef <= ZERO || eta_max_ef >= ONE {
        cv_mem.cv_eta_max_ef = ETA_MAX_EF_DEFAULT;
    } else {
        cv_mem.cv_eta_max_ef = eta_max_ef;
    }

    CV_SUCCESS
}

/*
 * CVodeSetNumFailsEtaMaxErrFail
 *
 * Specifies the maximum number of error test failures necessary to enforce
 * eta_max_ef
 */

pub fn CVodeSetNumFailsEtaMaxErrFail(cvode_mem: &CVodeMem, small_nef: i32) -> i32 {
    /* NULL-mem check: handled by type system */
    let mut cv_mem = cvode_mem.borrow_mut();

    /* set allowed value or use default */
    if small_nef < 0 {
        cv_mem.cv_small_nef = SMALL_NEF_DEFAULT;
    } else {
        cv_mem.cv_small_nef = small_nef;
    }

    CV_SUCCESS
}

/*
 * CVodeSetEtaConvFail
 *
 * Specifies the step size change after a nonlinear solver failure
 */

pub fn CVodeSetEtaConvFail(cvode_mem: &CVodeMem, eta_cf: sunrealtype) -> i32 {
    /* NULL-mem check: handled by type system */
    let mut cv_mem = cvode_mem.borrow_mut();

    /* set allowed value or use default */
    if eta_cf <= ZERO || eta_cf >= ONE {
        cv_mem.cv_eta_cf = ETA_CF_DEFAULT;
    } else {
        cv_mem.cv_eta_cf = eta_cf;
    }

    CV_SUCCESS
}

/*
 * CVodeSetStopTime
 *
 * Specifies the time beyond which the integration is not to proceed.
 */

pub fn CVodeSetStopTime(cvode_mem: &CVodeMem, tstop: sunrealtype) -> i32 {
    /* NULL-mem check: handled by type system */

    /* If CVode was called at least once, test if tstop is legal
     * (i.e. if it was not already passed).
     * If CVodeSetStopTime is called before the first call to CVode,
     * tstop will be checked in CVode. */
    let (nst, tn, h) = {
        let cv_mem = cvode_mem.borrow();
        (cv_mem.cv_nst, cv_mem.cv_tn, cv_mem.cv_h)
    };
    if nst > 0 {
        if (tstop - tn) * h < ZERO {
            cvProcessError(
                Some(cvode_mem),
                CV_ILL_INPUT,
                line!() as i32,
                "CVodeSetStopTime",
                file!(),
                &MSGCV_BAD_TSTOP(tstop, tn),
            );
            return CV_ILL_INPUT;
        }
    }

    let mut cv_mem = cvode_mem.borrow_mut();
    cv_mem.cv_tstop = tstop;
    cv_mem.cv_tstopset = SUNTRUE;

    CV_SUCCESS
}

/*
 * CVodeSetInterpolateStopTime
 *
 * Specifies to use interpolation to fill the output solution at
 * the stop time (instead of a copy).
 */

pub fn CVodeSetInterpolateStopTime(cvode_mem: &CVodeMem, interp: sunbooleantype) -> i32 {
    /* NULL-mem check: handled by type system */
    cvode_mem.borrow_mut().cv_tstopinterp = interp;

    CV_SUCCESS
}

/*
 * CVodeClearStopTime
 *
 * Disable the stop time.
 */

pub fn CVodeClearStopTime(cvode_mem: &CVodeMem) -> i32 {
    /* NULL-mem check: handled by type system */
    cvode_mem.borrow_mut().cv_tstopset = SUNFALSE;

    CV_SUCCESS
}

/*
 * CVodeSetMaxErrTestFails
 *
 * Specifies the maximum number of error test failures during one
 * step try.
 */

pub fn CVodeSetMaxErrTestFails(cvode_mem: &CVodeMem, maxnef: i32) -> i32 {
    /* NULL-mem check: handled by type system */
    cvode_mem.borrow_mut().cv_maxnef = maxnef;

    CV_SUCCESS
}

/*
 * CVodeSetMaxConvFails
 *
 * Specifies the maximum number of nonlinear convergence failures
 * during one step try.
 */

pub fn CVodeSetMaxConvFails(cvode_mem: &CVodeMem, maxncf: i32) -> i32 {
    /* NULL-mem check: handled by type system */
    cvode_mem.borrow_mut().cv_maxncf = maxncf;

    CV_SUCCESS
}

/*
 * CVodeSetMaxNonlinIters
 *
 * Specifies the maximum number of nonlinear iterations during
 * one solve.
 */

pub fn CVodeSetMaxNonlinIters(cvode_mem: &CVodeMem, maxcor: i32) -> i32 {
    /* NULL-mem check: handled by type system */

    /* Are we computing sensitivities with the simultaneous approach? */
    let sensi_sim = {
        let cv_mem = cvode_mem.borrow();
        cv_mem.cv_sensi && (cv_mem.cv_ism == CV_SIMULTANEOUS)
    };

    if sensi_sim {
        /* check that the NLS is non-NULL */
        let NLSsim = cvode_mem.borrow().NLSsim.clone();
        match NLSsim {
            None => {
                cvProcessError(
                    None,
                    CV_MEM_FAIL,
                    line!() as i32,
                    "CVodeSetMaxNonlinIters",
                    file!(),
                    MSGCV_MEM_FAIL,
                );
                CV_MEM_FAIL
            }
            Some(NLSsim) => SUNNonlinSolSetMaxIters(&NLSsim, maxcor),
        }
    } else {
        /* check that the NLS is non-NULL */
        let NLS = cvode_mem.borrow().NLS.clone();
        match NLS {
            None => {
                cvProcessError(
                    None,
                    CV_MEM_FAIL,
                    line!() as i32,
                    "CVodeSetMaxNonlinIters",
                    file!(),
                    MSGCV_MEM_FAIL,
                );
                CV_MEM_FAIL
            }
            Some(NLS) => SUNNonlinSolSetMaxIters(&NLS, maxcor),
        }
    }
}

/*
 * CVodeSetNonlinConvCoef
 *
 * Specifies the coefficient in the nonlinear solver convergence
 * test
 */

pub fn CVodeSetNonlinConvCoef(cvode_mem: &CVodeMem, nlscoef: sunrealtype) -> i32 {
    /* NULL-mem check: handled by type system */
    cvode_mem.borrow_mut().cv_nlscoef = nlscoef;

    CV_SUCCESS
}

/*
 * CVodeSetLSetupFrequency
 *
 * Specifies the frequency for calling the linear solver setup function to
 * recompute the Jacobian matrix and/or preconditioner
 */

pub fn CVodeSetLSetupFrequency(cvode_mem: &CVodeMem, msbp: i64) -> i32 {
    /* NULL-mem check: handled by type system */

    /* check for a valid input */
    if msbp < 0 {
        cvProcessError(
            Some(cvode_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeSetLSetupFrequency",
            file!(),
            "A negative setup frequency was provided",
        );
        return CV_ILL_INPUT;
    }

    /* use default or user provided value */
    cvode_mem.borrow_mut().cv_msbp = if msbp == 0 { MSBP_DEFAULT } else { msbp };

    CV_SUCCESS
}

/*
 * CVodeSetRootDirection
 *
 * Specifies the direction of zero-crossings to be monitored.
 * The default is to monitor both crossings.
 */

pub fn CVodeSetRootDirection(cvode_mem: &CVodeMem, rootdir: &[i32]) -> i32 {
    /* NULL-mem check: handled by type system */
    let nrt = cvode_mem.borrow().cv_nrtfn;
    if nrt == 0 {
        cvProcessError(
            Some(cvode_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeSetRootDirection",
            file!(),
            MSGCV_NO_ROOT,
        );
        return CV_ILL_INPUT;
    }

    let mut cv_mem = cvode_mem.borrow_mut();
    for i in 0..nrt {
        cv_mem.cv_rootdir[i as usize] = rootdir[i as usize];
    }

    CV_SUCCESS
}

/*
 * CVodeSetNoInactiveRootWarn
 *
 * Disables issuing a warning if some root function appears
 * to be identically zero at the beginning of the integration
 */

pub fn CVodeSetNoInactiveRootWarn(cvode_mem: &CVodeMem) -> i32 {
    /* NULL-mem check: handled by type system */
    cvode_mem.borrow_mut().cv_mxgnull = 0;

    CV_SUCCESS
}

/*
 * CVodeSetConstraints
 *
 * Setup for constraint handling feature
 */

pub fn CVodeSetConstraints(cvode_mem: &CVodeMem, constraints: Option<&N_Vector>) -> i32 {
    /* NULL-mem check: handled by type system */

    /* Disable constraints */
    let constraints = match constraints {
        None => {
            /* C destroys the stored vector but leaves the pointer dangling
            (upstream bug: any later use is UB); the safe port takes the
            handle out, so later checks see "no constraints". */
            let taken = cvode_mem.borrow_mut().cv_constraints.take();
            if let Some(c) = taken {
                N_VDestroy(c);
                let mut cv_mem = cvode_mem.borrow_mut();
                let lrw1 = cv_mem.cv_lrw1;
                let liw1 = cv_mem.cv_liw1;
                cv_mem.cv_lrw -= lrw1;
                cv_mem.cv_liw -= liw1;
            }
            return CV_SUCCESS;
        }
        Some(c) => c,
    };

    /* Test if required vector ops. are defined */
    {
        let ops = constraints.ops.borrow();
        if ops.nvdiv.is_none()
            || ops.nvmaxnorm.is_none()
            || ops.nvcompare.is_none()
            || ops.nvconstrmask.is_none()
            || ops.nvminquotient.is_none()
        {
            drop(ops);
            cvProcessError(
                Some(cvode_mem),
                CV_ILL_INPUT,
                line!() as i32,
                "CVodeSetConstraints",
                file!(),
                MSGCV_BAD_NVECTOR,
            );
            return CV_ILL_INPUT;
        }
    }

    /* Check the constraints vector */
    let temptest = N_VMaxNorm(constraints);
    if (temptest > TWOPT5) || (temptest < HALF) {
        cvProcessError(
            Some(cvode_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeSetConstraints",
            file!(),
            MSGCV_BAD_CONSTR,
        );
        return CV_ILL_INPUT;
    }

    /* Enable constraints */
    if cvode_mem.borrow().cv_constraints.is_none() {
        let cloned = match N_VClone(constraints) {
            Some(c) => c,
            None => {
                cvProcessError(
                    None,
                    CV_MEM_FAIL,
                    line!() as i32,
                    "CVodeSetConstraints",
                    file!(),
                    MSGCV_MEM_FAIL,
                );
                return CV_MEM_FAIL;
            }
        };
        let mut cv_mem = cvode_mem.borrow_mut();
        cv_mem.cv_constraints = Some(cloned);
        let lrw1 = cv_mem.cv_lrw1;
        let liw1 = cv_mem.cv_liw1;
        cv_mem.cv_lrw += lrw1;
        cv_mem.cv_liw += liw1;
    }

    /* Load the constraints vector */
    let target = cvode_mem
        .borrow()
        .cv_constraints
        .as_ref()
        .expect("cv_constraints")
        .clone();
    N_VScale(ONE, constraints, &target);

    CV_SUCCESS
}

/*
 * CVodeSetMaxNumConstraintFails
 *
 * Set the maximum number of constraint failure allowed in a step
 */

pub fn CVodeSetMaxNumConstraintFails(cvode_mem: &CVodeMem, max_fails: i32) -> i32 {
    /* NULL-mem check: handled by type system */
    let mut cv_mem = cvode_mem.borrow_mut();

    if max_fails <= 0 {
        cv_mem.max_constraint_fails = MAX_CONSTRAINT_FAILS;
    } else {
        cv_mem.max_constraint_fails = max_fails;
    }

    CV_SUCCESS
}

/*
 * CVodeGetNumConstraintFails
 *
 * Get the number of failed steps due to constraint violation
 */

pub fn CVodeGetNumConstraintFails(cvode_mem: &CVodeMem, num_fails_out: &mut i64) -> i32 {
    /* NULL-mem check: handled by type system */
    *num_fails_out = cvode_mem.borrow().constraint_fails;

    CV_SUCCESS
}

/*
 * CVodeGetNumConstraintCorrections
 *
 * Get the number of constraint corrections
 */

pub fn CVodeGetNumConstraintCorrections(
    cvode_mem: &CVodeMem,
    num_corrections_out: &mut i64,
) -> i32 {
    /* NULL-mem check: handled by type system */
    *num_corrections_out = cvode_mem.borrow().constraint_corrections;

    CV_SUCCESS
}

/*
 * =================================================================
 * Quadrature optional input functions
 * =================================================================
 */

pub fn CVodeSetQuadErrCon(cvode_mem: &CVodeMem, errconQ: sunbooleantype) -> i32 {
    /* NULL-mem check: handled by type system */
    cvode_mem.borrow_mut().cv_errconQ = errconQ;

    CV_SUCCESS
}

/*
 * =================================================================
 * FSA optional input functions
 * =================================================================
 */

pub fn CVodeSetSensDQMethod(cvode_mem: &CVodeMem, DQtype: i32, DQrhomax: sunrealtype) -> i32 {
    /* NULL-mem check: handled by type system */
    if (DQtype != CV_CENTERED) && (DQtype != CV_FORWARD) {
        cvProcessError(
            Some(cvode_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeSetSensDQMethod",
            file!(),
            MSGCV_BAD_DQTYPE,
        );
        return CV_ILL_INPUT;
    }

    if DQrhomax < ZERO {
        cvProcessError(
            Some(cvode_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeSetSensDQMethod",
            file!(),
            MSGCV_BAD_DQRHO,
        );
        return CV_ILL_INPUT;
    }

    let mut cv_mem = cvode_mem.borrow_mut();
    cv_mem.cv_DQtype = DQtype;
    cv_mem.cv_DQrhomax = DQrhomax;

    CV_SUCCESS
}

/*-----------------------------------------------------------------*/

pub fn CVodeSetSensErrCon(cvode_mem: &CVodeMem, errconS: sunbooleantype) -> i32 {
    /* NULL-mem check: handled by type system */
    cvode_mem.borrow_mut().cv_errconS = errconS;

    CV_SUCCESS
}

/*-----------------------------------------------------------------*/

pub fn CVodeSetSensMaxNonlinIters(cvode_mem: &CVodeMem, maxcorS: i32) -> i32 {
    /* NULL-mem check: handled by type system */

    /* Are we computing sensitivities with a staggered approach? */
    let sensi_stg = {
        let cv_mem = cvode_mem.borrow();
        cv_mem.cv_sensi && (cv_mem.cv_ism == CV_STAGGERED)
    };

    if sensi_stg {
        /* check that the NLS is non-NULL */
        let NLSstg = cvode_mem.borrow().NLSstg.clone();
        match NLSstg {
            None => {
                cvProcessError(
                    None,
                    CV_MEM_FAIL,
                    line!() as i32,
                    "CVodeSetSensMaxNonlinIters",
                    file!(),
                    MSGCV_MEM_FAIL,
                );
                CV_MEM_FAIL
            }
            Some(NLSstg) => SUNNonlinSolSetMaxIters(&NLSstg, maxcorS),
        }
    } else {
        /* check that the NLS is non-NULL */
        let NLSstg1 = cvode_mem.borrow().NLSstg1.clone();
        match NLSstg1 {
            None => {
                cvProcessError(
                    None,
                    CV_MEM_FAIL,
                    line!() as i32,
                    "CVodeSetSensMaxNonlinIters",
                    file!(),
                    MSGCV_MEM_FAIL,
                );
                CV_MEM_FAIL
            }
            Some(NLSstg1) => SUNNonlinSolSetMaxIters(&NLSstg1, maxcorS),
        }
    }
}

/*-----------------------------------------------------------------*/

/// C stores the caller's `p` POINTER in `cv_mem->cv_p`, so the internal DQ
/// sensitivity RHS perturbs the user's own parameter array in place and the
/// user's `f` (reading the same memory through `user_data`) observes it —
/// that aliasing IS the DQ mechanism. The port reproduces it with the
/// `SensParams` shared handle (`Rc<RefCell<Vec<sunrealtype>>>`): pass a
/// CLONE of the very handle the user data holds, and the perturbations
/// reach the callback exactly as in C. `None` is C's `NULL` (legal only
/// when the sensitivity RHS is user-supplied).
///
/// `pbar` and `plist` stay borrowed slices: C copies them element-wise into
/// `cv_mem`'s own arrays and never writes back, so an owned copy is faithful.
pub fn CVodeSetSensParams(
    cvode_mem: &CVodeMem,
    p: Option<SensParams>,
    pbar: Option<&[sunrealtype]>,
    plist: Option<&[i32]>,
) -> i32 {
    /* NULL-mem check: handled by type system */

    /* Was sensitivity initialized? */

    if cvode_mem.borrow().cv_SensMallocDone == SUNFALSE {
        cvProcessError(
            Some(cvode_mem),
            CV_NO_SENS,
            line!() as i32,
            "CVodeSetSensParams",
            file!(),
            MSGCV_NO_SENSI,
        );
        return CV_NO_SENS;
    }

    let Ns = cvode_mem.borrow().cv_Ns;

    /* Parameters */

    cvode_mem.borrow_mut().cv_p = p;

    /* pbar */

    match pbar {
        Some(pbar) => {
            for is in 0..Ns {
                if pbar[is as usize] == ZERO {
                    cvProcessError(
                        Some(cvode_mem),
                        CV_ILL_INPUT,
                        line!() as i32,
                        "CVodeSetSensParams",
                        file!(),
                        MSGCV_BAD_PBAR,
                    );
                    return CV_ILL_INPUT;
                }
                cvode_mem.borrow_mut().cv_pbar[is as usize] = SUNRabs(pbar[is as usize]);
            }
        }
        None => {
            let mut cv_mem = cvode_mem.borrow_mut();
            for is in 0..Ns {
                cv_mem.cv_pbar[is as usize] = ONE;
            }
        }
    }

    /* plist */

    match plist {
        Some(plist) => {
            for is in 0..Ns {
                if plist[is as usize] < 0 {
                    cvProcessError(
                        Some(cvode_mem),
                        CV_ILL_INPUT,
                        line!() as i32,
                        "CVodeSetSensParams",
                        file!(),
                        MSGCV_BAD_PLIST,
                    );
                    return CV_ILL_INPUT;
                }
                cvode_mem.borrow_mut().cv_plist[is as usize] = plist[is as usize];
            }
        }
        None => {
            let mut cv_mem = cvode_mem.borrow_mut();
            for is in 0..Ns {
                cv_mem.cv_plist[is as usize] = is;
            }
        }
    }

    CV_SUCCESS
}

/*-----------------------------------------------------------------*/

pub fn CVodeSetQuadSensErrCon(cvode_mem: &CVodeMem, errconQS: sunbooleantype) -> i32 {
    /* NULL-mem check: handled by type system */

    /* Was sensitivity initialized? */

    if cvode_mem.borrow().cv_SensMallocDone == SUNFALSE {
        cvProcessError(
            Some(cvode_mem),
            CV_NO_SENS,
            line!() as i32,
            "CVodeSetQuadSensErrCon",
            file!(),
            MSGCV_NO_SENSI,
        );
        return CV_NO_SENS;
    }

    /* Check if quadrature sensitivity was initialized? */

    if cvode_mem.borrow().cv_QuadSensMallocDone == SUNFALSE {
        cvProcessError(
            Some(cvode_mem),
            CV_NO_QUADSENS,
            line!() as i32,
            "CVodeSetQuadSensErrCon",
            file!(),
            MSGCV_NO_QUADSENSI,
        );
        /* upstream returns CV_NO_QUAD here (not CV_NO_QUADSENS) */
        return CV_NO_QUAD;
    }

    cvode_mem.borrow_mut().cv_errconQS = errconQS;

    CV_SUCCESS
}

/*
 * =================================================================
 * CVODES optional output functions
 * =================================================================
 */

/*
 * CVodeGetNumSteps
 *
 * Returns the current number of integration steps
 */

pub fn CVodeGetNumSteps(cvode_mem: &CVodeMem, nsteps: &mut i64) -> i32 {
    /* NULL-mem check: handled by type system */
    *nsteps = cvode_mem.borrow().cv_nst;

    CV_SUCCESS
}

/*
 * CVodeGetNumRhsEvals
 *
 * Returns the current number of calls to f
 */

pub fn CVodeGetNumRhsEvals(cvode_mem: &CVodeMem, nfevals: &mut i64) -> i32 {
    /* NULL-mem check: handled by type system */
    *nfevals = cvode_mem.borrow().cv_nfe;

    CV_SUCCESS
}

/*
 * CVodeGetNumLinSolvSetups
 *
 * Returns the current number of calls to the linear solver setup routine
 */

pub fn CVodeGetNumLinSolvSetups(cvode_mem: &CVodeMem, nlinsetups: &mut i64) -> i32 {
    /* NULL-mem check: handled by type system */
    *nlinsetups = cvode_mem.borrow().cv_nsetups;

    CV_SUCCESS
}

/*
 * CVodeGetNumErrTestFails
 *
 * Returns the current number of error test failures
 */

pub fn CVodeGetNumErrTestFails(cvode_mem: &CVodeMem, netfails: &mut i64) -> i32 {
    /* NULL-mem check: handled by type system */
    *netfails = cvode_mem.borrow().cv_netf;

    CV_SUCCESS
}

/*
 * CVodeGetLastOrder
 *
 * Returns the order on the last successful step
 */

pub fn CVodeGetLastOrder(cvode_mem: &CVodeMem, qlast: &mut i32) -> i32 {
    /* NULL-mem check: handled by type system */
    *qlast = cvode_mem.borrow().cv_qu;

    CV_SUCCESS
}

/*
 * CVodeGetCurrentOrder
 *
 * Returns the order to be attempted on the next step
 */

pub fn CVodeGetCurrentOrder(cvode_mem: &CVodeMem, qcur: &mut i32) -> i32 {
    /* NULL-mem check: handled by type system */
    *qcur = cvode_mem.borrow().cv_next_q;

    CV_SUCCESS
}

/*
 * CVodeGetCurrentGamma
 *
 * Returns the value of gamma for the current step.
 */

pub fn CVodeGetCurrentGamma(cvode_mem: &CVodeMem, gamma: &mut sunrealtype) -> i32 {
    /* NULL-mem check: handled by type system */
    *gamma = cvode_mem.borrow().cv_gamma;

    CV_SUCCESS
}

/*
 * CVodeGetNumStabLimOrderReds
 *
 * Returns the number of order reductions triggered by the stability
 * limit detection algorithm
 */

pub fn CVodeGetNumStabLimOrderReds(cvode_mem: &CVodeMem, nslred: &mut i64) -> i32 {
    /* NULL-mem check: handled by type system */
    let cv_mem = cvode_mem.borrow();

    if cv_mem.cv_sldeton == SUNFALSE {
        *nslred = 0;
    } else {
        *nslred = cv_mem.cv_nor;
    }

    CV_SUCCESS
}

/*
 * CVodeGetActualInitStep
 *
 * Returns the step size used on the first step
 */

pub fn CVodeGetActualInitStep(cvode_mem: &CVodeMem, hinused: &mut sunrealtype) -> i32 {
    /* NULL-mem check: handled by type system */
    *hinused = cvode_mem.borrow().cv_h0u;

    CV_SUCCESS
}

/*
 * CVodeGetLastStep
 *
 * Returns the step size used on the last successful step
 */

pub fn CVodeGetLastStep(cvode_mem: &CVodeMem, hlast: &mut sunrealtype) -> i32 {
    /* NULL-mem check: handled by type system */
    *hlast = cvode_mem.borrow().cv_hu;

    CV_SUCCESS
}

/*
 * CVodeGetCurrentStep
 *
 * Returns the step size to be attempted on the next step
 */

pub fn CVodeGetCurrentStep(cvode_mem: &CVodeMem, hcur: &mut sunrealtype) -> i32 {
    /* NULL-mem check: handled by type system */
    *hcur = cvode_mem.borrow().cv_next_h;

    CV_SUCCESS
}

/*
 * CVodeGetCurrentState
 *
 * Returns the current state vector
 */

pub fn CVodeGetCurrentState(cvode_mem: &CVodeMem, y: &mut Option<N_Vector>) -> i32 {
    /* NULL-mem check: handled by type system */
    /* handle clone out = C pointer copy */
    *y = cvode_mem.borrow().cv_y.clone();

    CV_SUCCESS
}

/*
 * CVodeGetCurrentStateSens
 *
 * Returns the current sensitivity state vector array
 */

pub fn CVodeGetCurrentStateSens(cvode_mem: &CVodeMem, yS: &mut Vec<N_Vector>) -> i32 {
    /* NULL-mem check: handled by type system */
    /* handle clones out = C array-pointer copy (the vectors themselves are
    shared; only re-seating an array slot fails to propagate) */
    *yS = cvode_mem.borrow().cv_yS.clone();

    CV_SUCCESS
}

/*
 * CVodeGetCurrentSensSolveIndex
 *
 * Returns the current index of the sensitivity solve when using
 * the staggered1 nonlinear solver.
 */

pub fn CVodeGetCurrentSensSolveIndex(cvode_mem: &CVodeMem, index: &mut i32) -> i32 {
    /* NULL-mem check: handled by type system */
    *index = cvode_mem.borrow().sens_solve_idx;

    CV_SUCCESS
}

/*
 * CVodeGetCurrentTime
 *
 * Returns the current value of the independent variable
 */

pub fn CVodeGetCurrentTime(cvode_mem: &CVodeMem, tcur: &mut sunrealtype) -> i32 {
    /* NULL-mem check: handled by type system */
    *tcur = cvode_mem.borrow().cv_tn;

    CV_SUCCESS
}

/*
 * CVodeGetTolScaleFactor
 *
 * Returns a suggested factor for scaling tolerances
 */

pub fn CVodeGetTolScaleFactor(cvode_mem: &CVodeMem, tolsfact: &mut sunrealtype) -> i32 {
    /* NULL-mem check: handled by type system */
    *tolsfact = cvode_mem.borrow().cv_tolsf;

    CV_SUCCESS
}

/*
 * CVodeGetErrWeights
 *
 * This routine returns the current weight vector.
 */

pub fn CVodeGetErrWeights(cvode_mem: &CVodeMem, eweight: &N_Vector) -> i32 {
    /* NULL-mem check: handled by type system */
    let ewt = cvode_mem.borrow().cv_ewt.as_ref().expect("cv_ewt").clone();

    N_VScale(ONE, &ewt, eweight);

    CV_SUCCESS
}

/*
 * CVodeGetEstLocalErrors
 *
 * Returns an estimate of the local error
 */

pub fn CVodeGetEstLocalErrors(cvode_mem: &CVodeMem, ele: &N_Vector) -> i32 {
    /* NULL-mem check: handled by type system */
    let acor = cvode_mem
        .borrow()
        .cv_acor
        .as_ref()
        .expect("cv_acor")
        .clone();

    N_VScale(ONE, &acor, ele);

    CV_SUCCESS
}

/*
 * CVodeGetWorkSpace
 *
 * Returns integrator work space requirements
 */

pub fn CVodeGetWorkSpace(cvode_mem: &CVodeMem, lenrw: &mut i64, leniw: &mut i64) -> i32 {
    /* NULL-mem check: handled by type system */
    let cv_mem = cvode_mem.borrow();

    *leniw = cv_mem.cv_liw;
    *lenrw = cv_mem.cv_lrw;

    CV_SUCCESS
}

/*
 * CVodeGetIntegratorStats
 *
 * Returns integrator statistics
 */

#[allow(clippy::too_many_arguments)]
pub fn CVodeGetIntegratorStats(
    cvode_mem: &CVodeMem,
    nsteps: &mut i64,
    nfevals: &mut i64,
    nlinsetups: &mut i64,
    netfails: &mut i64,
    qlast: &mut i32,
    qcur: &mut i32,
    hinused: &mut sunrealtype,
    hlast: &mut sunrealtype,
    hcur: &mut sunrealtype,
    tcur: &mut sunrealtype,
) -> i32 {
    /* NULL-mem check: handled by type system */
    let cv_mem = cvode_mem.borrow();

    *nsteps = cv_mem.cv_nst;
    *nfevals = cv_mem.cv_nfe;
    *nlinsetups = cv_mem.cv_nsetups;
    *netfails = cv_mem.cv_netf;
    *qlast = cv_mem.cv_qu;
    *qcur = cv_mem.cv_next_q;
    *hinused = cv_mem.cv_h0u;
    *hlast = cv_mem.cv_hu;
    *hcur = cv_mem.cv_next_h;
    *tcur = cv_mem.cv_tn;

    CV_SUCCESS
}

/*
 * CVodeGetNumGEvals
 *
 * Returns the current number of calls to g (for rootfinding)
 */

pub fn CVodeGetNumGEvals(cvode_mem: &CVodeMem, ngevals: &mut i64) -> i32 {
    /* NULL-mem check: handled by type system */
    *ngevals = cvode_mem.borrow().cv_nge;

    CV_SUCCESS
}

/*
 * CVodeGetRootInfo
 *
 * Returns pointer to array rootsfound showing roots found
 */

pub fn CVodeGetRootInfo(cvode_mem: &CVodeMem, rootsfound: &mut [i32]) -> i32 {
    /* NULL-mem check: handled by type system */
    let cv_mem = cvode_mem.borrow();

    let nrt = cv_mem.cv_nrtfn;

    for i in 0..nrt {
        rootsfound[i as usize] = cv_mem.cv_iroots[i as usize];
    }

    CV_SUCCESS
}

/*
 * CVodeGetNumNonlinSolvIters
 *
 * Returns the current number of iterations in the nonlinear solver
 */

pub fn CVodeGetNumNonlinSolvIters(cvode_mem: &CVodeMem, nniters: &mut i64) -> i32 {
    /* NULL-mem check: handled by type system */
    *nniters = cvode_mem.borrow().cv_nni;

    CV_SUCCESS
}

/*
 * CVodeGetNumNonlinSolvConvFails
 *
 * Returns the current number of convergence failures in the
 * nonlinear solver
 */

pub fn CVodeGetNumNonlinSolvConvFails(cvode_mem: &CVodeMem, nnfails: &mut i64) -> i32 {
    /* NULL-mem check: handled by type system */
    *nnfails = cvode_mem.borrow().cv_nnf;

    CV_SUCCESS
}

/*
 * CVodeGetNonlinSolvStats
 *
 * Returns nonlinear solver statistics
 */

pub fn CVodeGetNonlinSolvStats(cvode_mem: &CVodeMem, nniters: &mut i64, nnfails: &mut i64) -> i32 {
    /* NULL-mem check: handled by type system */
    let cv_mem = cvode_mem.borrow();

    *nniters = cv_mem.cv_nni;
    *nnfails = cv_mem.cv_nnf;

    CV_SUCCESS
}

/*
 * CVodeGetNumStepSolveFails
 *
 * Returns the current number of failed steps due to a nonlinear solver
 * convergence failure
 */

pub fn CVodeGetNumStepSolveFails(cvode_mem: &CVodeMem, nncfails: &mut i64) -> i32 {
    /* NULL-mem check: handled by type system */
    *nncfails = cvode_mem.borrow().cv_ncfn;

    CV_SUCCESS
}

/*
 * =================================================================
 * Quadrature optional output functions
 * =================================================================
 */

/*-----------------------------------------------------------------*/

pub fn CVodeGetQuadNumRhsEvals(cvode_mem: &CVodeMem, nfQevals: &mut i64) -> i32 {
    /* NULL-mem check: handled by type system */
    if cvode_mem.borrow().cv_quadr == SUNFALSE {
        cvProcessError(
            Some(cvode_mem),
            CV_NO_QUAD,
            line!() as i32,
            "CVodeGetQuadNumRhsEvals",
            file!(),
            MSGCV_NO_QUAD,
        );
        return CV_NO_QUAD;
    }

    *nfQevals = cvode_mem.borrow().cv_nfQe;

    CV_SUCCESS
}

/*-----------------------------------------------------------------*/

pub fn CVodeGetQuadNumErrTestFails(cvode_mem: &CVodeMem, nQetfails: &mut i64) -> i32 {
    /* NULL-mem check: handled by type system */
    if cvode_mem.borrow().cv_quadr == SUNFALSE {
        cvProcessError(
            Some(cvode_mem),
            CV_NO_QUAD,
            line!() as i32,
            "CVodeGetQuadNumErrTestFails",
            file!(),
            MSGCV_NO_QUAD,
        );
        return CV_NO_QUAD;
    }

    *nQetfails = cvode_mem.borrow().cv_netfQ;

    CV_SUCCESS
}

/*-----------------------------------------------------------------*/

pub fn CVodeGetQuadErrWeights(cvode_mem: &CVodeMem, eQweight: &N_Vector) -> i32 {
    /* NULL-mem check: handled by type system */
    if cvode_mem.borrow().cv_quadr == SUNFALSE {
        cvProcessError(
            Some(cvode_mem),
            CV_NO_QUAD,
            line!() as i32,
            "CVodeGetQuadErrWeights",
            file!(),
            MSGCV_NO_QUAD,
        );
        return CV_NO_QUAD;
    }

    let ewtQ = {
        let cv_mem = cvode_mem.borrow();
        if cv_mem.cv_errconQ {
            Some(cv_mem.cv_ewtQ.as_ref().expect("cv_ewtQ").clone())
        } else {
            None
        }
    };
    if let Some(ewtQ) = ewtQ {
        N_VScale(ONE, &ewtQ, eQweight);
    }

    CV_SUCCESS
}

/*-----------------------------------------------------------------*/

pub fn CVodeGetQuadStats(cvode_mem: &CVodeMem, nfQevals: &mut i64, nQetfails: &mut i64) -> i32 {
    /* NULL-mem check: handled by type system */
    if cvode_mem.borrow().cv_quadr == SUNFALSE {
        cvProcessError(
            Some(cvode_mem),
            CV_NO_QUAD,
            line!() as i32,
            "CVodeGetQuadStats",
            file!(),
            MSGCV_NO_QUAD,
        );
        return CV_NO_QUAD;
    }

    let cv_mem = cvode_mem.borrow();
    *nfQevals = cv_mem.cv_nfQe;
    *nQetfails = cv_mem.cv_netfQ;

    CV_SUCCESS
}

/*
 * =================================================================
 * Quadrature FSA optional output functions
 * =================================================================
 */

/*-----------------------------------------------------------------*/

pub fn CVodeGetQuadSensNumRhsEvals(cvode_mem: &CVodeMem, nfQSevals: &mut i64) -> i32 {
    /* NULL-mem check: handled by type system */
    if cvode_mem.borrow().cv_quadr_sensi == SUNFALSE {
        cvProcessError(
            Some(cvode_mem),
            CV_NO_QUADSENS,
            line!() as i32,
            "CVodeGetQuadSensNumRhsEvals",
            file!(),
            MSGCV_NO_QUADSENSI,
        );
        return CV_NO_QUADSENS;
    }

    *nfQSevals = cvode_mem.borrow().cv_nfQSe;

    CV_SUCCESS
}

/*-----------------------------------------------------------------*/

pub fn CVodeGetQuadSensNumErrTestFails(cvode_mem: &CVodeMem, nQSetfails: &mut i64) -> i32 {
    /* NULL-mem check: handled by type system */
    if cvode_mem.borrow().cv_quadr_sensi == SUNFALSE {
        cvProcessError(
            Some(cvode_mem),
            CV_NO_QUADSENS,
            line!() as i32,
            "CVodeGetQuadSensNumErrTestFails",
            file!(),
            MSGCV_NO_QUADSENSI,
        );
        return CV_NO_QUADSENS;
    }

    *nQSetfails = cvode_mem.borrow().cv_netfQS;

    CV_SUCCESS
}

/*-----------------------------------------------------------------*/

pub fn CVodeGetQuadSensErrWeights(cvode_mem: &CVodeMem, eQSweight: &[N_Vector]) -> i32 {
    /* NULL-mem check: handled by type system */
    if cvode_mem.borrow().cv_quadr_sensi == SUNFALSE {
        cvProcessError(
            Some(cvode_mem),
            CV_NO_QUADSENS,
            line!() as i32,
            "CVodeGetQuadSensErrWeights",
            file!(),
            MSGCV_NO_QUADSENSI,
        );
        return CV_NO_QUADSENS;
    }
    let Ns = cvode_mem.borrow().cv_Ns;

    let ewtQS = {
        let cv_mem = cvode_mem.borrow();
        if cv_mem.cv_errconQS {
            Some(cv_mem.cv_ewtQS.clone())
        } else {
            None
        }
    };
    if let Some(ewtQS) = ewtQS {
        for is in 0..Ns {
            N_VScale(ONE, &ewtQS[is as usize], &eQSweight[is as usize]);
        }
    }

    CV_SUCCESS
}

/*-----------------------------------------------------------------*/

pub fn CVodeGetQuadSensStats(
    cvode_mem: &CVodeMem,
    nfQSevals: &mut i64,
    nQSetfails: &mut i64,
) -> i32 {
    /* NULL-mem check: handled by type system */
    if cvode_mem.borrow().cv_quadr_sensi == SUNFALSE {
        cvProcessError(
            Some(cvode_mem),
            CV_NO_QUADSENS,
            line!() as i32,
            "CVodeGetQuadSensStats",
            file!(),
            MSGCV_NO_QUADSENSI,
        );
        return CV_NO_QUADSENS;
    }

    let cv_mem = cvode_mem.borrow();
    *nfQSevals = cv_mem.cv_nfQSe;
    *nQSetfails = cv_mem.cv_netfQS;

    CV_SUCCESS
}

/*
 * =================================================================
 * FSA optional output functions
 * =================================================================
 */

/*-----------------------------------------------------------------*/

pub fn CVodeGetSensNumRhsEvals(cvode_mem: &CVodeMem, nfSevals: &mut i64) -> i32 {
    /* NULL-mem check: handled by type system */
    if cvode_mem.borrow().cv_sensi == SUNFALSE {
        cvProcessError(
            Some(cvode_mem),
            CV_NO_SENS,
            line!() as i32,
            "CVodeGetSensNumRhsEvals",
            file!(),
            MSGCV_NO_SENSI,
        );
        return CV_NO_SENS;
    }

    *nfSevals = cvode_mem.borrow().cv_nfSe;

    CV_SUCCESS
}

/*-----------------------------------------------------------------*/

pub fn CVodeGetNumRhsEvalsSens(cvode_mem: &CVodeMem, nfevalsS: &mut i64) -> i32 {
    /* NULL-mem check: handled by type system */
    if cvode_mem.borrow().cv_sensi == SUNFALSE {
        cvProcessError(
            Some(cvode_mem),
            CV_NO_SENS,
            line!() as i32,
            "CVodeGetNumRhsEvalsSens",
            file!(),
            MSGCV_NO_SENSI,
        );
        return CV_NO_SENS;
    }

    *nfevalsS = cvode_mem.borrow().cv_nfeS;

    CV_SUCCESS
}

/*-----------------------------------------------------------------*/

pub fn CVodeGetSensNumErrTestFails(cvode_mem: &CVodeMem, nSetfails: &mut i64) -> i32 {
    /* NULL-mem check: handled by type system */
    if cvode_mem.borrow().cv_sensi == SUNFALSE {
        cvProcessError(
            Some(cvode_mem),
            CV_NO_SENS,
            line!() as i32,
            "CVodeGetSensNumErrTestFails",
            file!(),
            MSGCV_NO_SENSI,
        );
        return CV_NO_SENS;
    }

    *nSetfails = cvode_mem.borrow().cv_netfS;

    CV_SUCCESS
}

/*-----------------------------------------------------------------*/

pub fn CVodeGetSensNumLinSolvSetups(cvode_mem: &CVodeMem, nlinsetupsS: &mut i64) -> i32 {
    /* NULL-mem check: handled by type system */
    if cvode_mem.borrow().cv_sensi == SUNFALSE {
        cvProcessError(
            Some(cvode_mem),
            CV_NO_SENS,
            line!() as i32,
            "CVodeGetSensNumLinSolvSetups",
            file!(),
            MSGCV_NO_SENSI,
        );
        return CV_NO_SENS;
    }

    *nlinsetupsS = cvode_mem.borrow().cv_nsetupsS;

    CV_SUCCESS
}

/*-----------------------------------------------------------------*/

pub fn CVodeGetSensErrWeights(cvode_mem: &CVodeMem, eSweight: &[N_Vector]) -> i32 {
    /* NULL-mem check: handled by type system */
    if cvode_mem.borrow().cv_sensi == SUNFALSE {
        cvProcessError(
            Some(cvode_mem),
            CV_NO_SENS,
            line!() as i32,
            "CVodeGetSensErrWeights",
            file!(),
            MSGCV_NO_SENSI,
        );
        return CV_NO_SENS;
    }

    let (Ns, ewtS) = {
        let cv_mem = cvode_mem.borrow();
        (cv_mem.cv_Ns, cv_mem.cv_ewtS.clone())
    };

    for is in 0..Ns {
        N_VScale(ONE, &ewtS[is as usize], &eSweight[is as usize]);
    }

    CV_SUCCESS
}

/*-----------------------------------------------------------------*/

pub fn CVodeGetSensStats(
    cvode_mem: &CVodeMem,
    nfSevals: &mut i64,
    nfevalsS: &mut i64,
    nSetfails: &mut i64,
    nlinsetupsS: &mut i64,
) -> i32 {
    /* NULL-mem check: handled by type system */
    if cvode_mem.borrow().cv_sensi == SUNFALSE {
        cvProcessError(
            Some(cvode_mem),
            CV_NO_SENS,
            line!() as i32,
            "CVodeGetSensStats",
            file!(),
            MSGCV_NO_SENSI,
        );
        return CV_NO_SENS;
    }

    let cv_mem = cvode_mem.borrow();
    *nfSevals = cv_mem.cv_nfSe;
    *nfevalsS = cv_mem.cv_nfeS;
    *nSetfails = cv_mem.cv_netfS;
    *nlinsetupsS = cv_mem.cv_nsetupsS;

    CV_SUCCESS
}

/*-----------------------------------------------------------------*/

pub fn CVodeGetSensNumNonlinSolvIters(cvode_mem: &CVodeMem, nSniters: &mut i64) -> i32 {
    /* NULL-mem check: handled by type system */
    if cvode_mem.borrow().cv_sensi == SUNFALSE {
        cvProcessError(
            Some(cvode_mem),
            CV_NO_SENS,
            line!() as i32,
            "CVodeGetSensNumNonlinSolvIters",
            file!(),
            MSGCV_NO_SENSI,
        );
        return CV_NO_SENS;
    }

    *nSniters = cvode_mem.borrow().cv_nniS;

    CV_SUCCESS
}

/*-----------------------------------------------------------------*/

pub fn CVodeGetSensNumNonlinSolvConvFails(cvode_mem: &CVodeMem, nSnfails: &mut i64) -> i32 {
    /* NULL-mem check: handled by type system */
    if cvode_mem.borrow().cv_sensi == SUNFALSE {
        cvProcessError(
            Some(cvode_mem),
            CV_NO_SENS,
            line!() as i32,
            "CVodeGetSensNumNonlinSolvConvFails",
            file!(),
            MSGCV_NO_SENSI,
        );
        return CV_NO_SENS;
    }

    *nSnfails = cvode_mem.borrow().cv_nnfS;

    CV_SUCCESS
}

/*-----------------------------------------------------------------*/

pub fn CVodeGetSensNonlinSolvStats(
    cvode_mem: &CVodeMem,
    nSniters: &mut i64,
    nSnfails: &mut i64,
) -> i32 {
    /* NULL-mem check: handled by type system */
    if cvode_mem.borrow().cv_sensi == SUNFALSE {
        cvProcessError(
            Some(cvode_mem),
            CV_NO_SENS,
            line!() as i32,
            "CVodeGetSensNonlinSolvStats",
            file!(),
            MSGCV_NO_SENSI,
        );
        return CV_NO_SENS;
    }

    let cv_mem = cvode_mem.borrow();
    *nSniters = cv_mem.cv_nniS;
    *nSnfails = cv_mem.cv_nnfS;

    CV_SUCCESS
}

pub fn CVodeGetNumStepSensSolveFails(cvode_mem: &CVodeMem, nSncfails: &mut i64) -> i32 {
    /* NULL-mem check: handled by type system */
    if cvode_mem.borrow().cv_sensi == SUNFALSE {
        cvProcessError(
            Some(cvode_mem),
            CV_NO_SENS,
            line!() as i32,
            "CVodeGetNumStepSensSolveFails",
            file!(),
            MSGCV_NO_SENSI,
        );
        return CV_NO_SENS;
    }

    *nSncfails = cvode_mem.borrow().cv_ncfnS;

    CV_SUCCESS
}

/*-----------------------------------------------------------------*/

pub fn CVodeGetStgrSensNumNonlinSolvIters(cvode_mem: &CVodeMem, nSTGR1niters: &mut [i64]) -> i32 {
    /* NULL-mem check: handled by type system */
    let Ns = cvode_mem.borrow().cv_Ns;

    if cvode_mem.borrow().cv_sensi == SUNFALSE {
        cvProcessError(
            Some(cvode_mem),
            CV_NO_SENS,
            line!() as i32,
            "CVodeGetStgrSensNumNonlinSolvIters",
            file!(),
            MSGCV_NO_SENSI,
        );
        return CV_NO_SENS;
    }

    let cv_mem = cvode_mem.borrow();
    if cv_mem.cv_ism == CV_STAGGERED1 {
        for is in 0..Ns {
            nSTGR1niters[is as usize] = cv_mem.cv_nniS1[is as usize];
        }
    }

    CV_SUCCESS
}

/*-----------------------------------------------------------------*/

pub fn CVodeGetStgrSensNumNonlinSolvConvFails(
    cvode_mem: &CVodeMem,
    nSTGR1nfails: &mut [i64],
) -> i32 {
    /* NULL-mem check: handled by type system */
    let Ns = cvode_mem.borrow().cv_Ns;

    if cvode_mem.borrow().cv_sensi == SUNFALSE {
        cvProcessError(
            Some(cvode_mem),
            CV_NO_SENS,
            line!() as i32,
            "CVodeGetStgrSensNumNonlinSolvConvFails",
            file!(),
            MSGCV_NO_SENSI,
        );
        return CV_NO_SENS;
    }

    let cv_mem = cvode_mem.borrow();
    if cv_mem.cv_ism == CV_STAGGERED1 {
        for is in 0..Ns {
            nSTGR1nfails[is as usize] = cv_mem.cv_nnfS1[is as usize];
        }
    }

    CV_SUCCESS
}

/*-----------------------------------------------------------------*/

pub fn CVodeGetStgrSensNonlinSolvStats(
    cvode_mem: &CVodeMem,
    nSTGR1niters: &mut [i64],
    nSTGR1nfails: &mut [i64],
) -> i32 {
    /* NULL-mem check: handled by type system */
    let Ns = cvode_mem.borrow().cv_Ns;

    if cvode_mem.borrow().cv_sensi == SUNFALSE {
        cvProcessError(
            Some(cvode_mem),
            CV_NO_SENS,
            line!() as i32,
            "CVodeGetStgrSensNonlinSolvStats",
            file!(),
            MSGCV_NO_SENSI,
        );
        return CV_NO_SENS;
    }

    let cv_mem = cvode_mem.borrow();
    if cv_mem.cv_ism == CV_STAGGERED1 {
        for is in 0..Ns {
            nSTGR1niters[is as usize] = cv_mem.cv_nniS1[is as usize];
        }
        for is in 0..Ns {
            nSTGR1nfails[is as usize] = cv_mem.cv_nnfS1[is as usize];
        }
    }

    CV_SUCCESS
}

/*-----------------------------------------------------------------*/

pub fn CVodeGetNumStepStgrSensSolveFails(cvode_mem: &CVodeMem, nSTGR1ncfails: &mut [i64]) -> i32 {
    /* NULL-mem check: handled by type system */
    let Ns = cvode_mem.borrow().cv_Ns;

    if cvode_mem.borrow().cv_sensi == SUNFALSE {
        cvProcessError(
            Some(cvode_mem),
            CV_NO_SENS,
            line!() as i32,
            "CVodeGetNumStepStgrSensSolveFails",
            file!(),
            MSGCV_NO_SENSI,
        );
        return CV_NO_SENS;
    }

    let cv_mem = cvode_mem.borrow();
    if cv_mem.cv_ism == CV_STAGGERED1 {
        for is in 0..Ns {
            nSTGR1ncfails[is as usize] = cv_mem.cv_ncfnS1[is as usize];
        }
    }

    CV_SUCCESS
}

/*
 * CVodePrintAllStats
 *
 * Print all integrator statistics
 */

pub fn CVodePrintAllStats(cvode_mem: &CVodeMem, outfile: &SUNFile, fmt: SUNOutputFormat) -> i32 {
    /* NULL-mem check: handled by type system */
    if fmt != SUNOutputFormat::SUN_OUTPUTFORMAT_TABLE
        && fmt != SUNOutputFormat::SUN_OUTPUTFORMAT_CSV
    {
        cvProcessError(
            Some(cvode_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodePrintAllStats",
            file!(),
            "Invalid formatting option.",
        );
        return CV_ILL_INPUT;
    }

    /* Copy every statistic out under one borrow (printing must not hold a
    borrow of the mem), then print in the exact C order. */
    let (
        tn,
        nst,
        netf,
        ncfn,
        constraint_fails,
        constraint_corrections,
        h0u,
        hu,
        next_h,
        qu,
        next_q,
        nor,
        nfe,
        nni,
        nnf,
        nsetups,
        nge,
    );
    let ls_stats: Option<(i64, i64, i64, i64, i64, i64, i64, i64)>;
    let proj_stats: Option<(i64, i64)>;
    let quadr_stats: Option<(i64, i64)>;
    let sensi_stats: Option<(i64, i64, i64, i64)>;
    let sensi_nls_stats: Option<(i64, i64, i64)>;
    let sensi_stgr1_stats: Option<(Vec<i64>, Vec<i64>, Vec<i64>)>;
    let quadr_sensi_stats: Option<(i64, i64)>;
    {
        let cv_mem = cvode_mem.borrow();
        tn = cv_mem.cv_tn;
        nst = cv_mem.cv_nst;
        netf = cv_mem.cv_netf;
        ncfn = cv_mem.cv_ncfn;
        constraint_fails = cv_mem.constraint_fails;
        constraint_corrections = cv_mem.constraint_corrections;
        h0u = cv_mem.cv_h0u;
        hu = cv_mem.cv_hu;
        next_h = cv_mem.cv_next_h;
        qu = cv_mem.cv_qu;
        next_q = cv_mem.cv_next_q;
        nor = cv_mem.cv_nor;
        nfe = cv_mem.cv_nfe;
        nni = cv_mem.cv_nni;
        nnf = cv_mem.cv_nnf;
        nsetups = cv_mem.cv_nsetups;
        nge = cv_mem.cv_nge;

        ls_stats = cv_mem.cv_lmem.as_ref().map(|lmem| {
            let cvls_mem = lmem
                .downcast_ref::<CVLsMemRec>()
                .expect("cv_lmem holds CVLsMemRec");
            (
                cvls_mem.nje,
                cvls_mem.nfeDQ,
                cvls_mem.npe,
                cvls_mem.nps,
                cvls_mem.nli,
                cvls_mem.ncfl,
                cvls_mem.njtsetup,
                cvls_mem.njtimes,
            )
        });

        proj_stats = cv_mem
            .proj_mem
            .as_ref()
            .map(|cvproj_mem| (cvproj_mem.nproj, cvproj_mem.npfails));

        quadr_stats = if cv_mem.cv_quadr {
            Some((cv_mem.cv_nfQe, cv_mem.cv_netfQ))
        } else {
            None
        };

        if cv_mem.cv_sensi {
            let Ns = cv_mem.cv_Ns as usize;
            sensi_stats = Some((
                cv_mem.cv_nfSe,
                cv_mem.cv_nfeS,
                cv_mem.cv_netfS,
                cv_mem.cv_nsetupsS,
            ));
            sensi_nls_stats = if cv_mem.cv_ism != CV_SIMULTANEOUS {
                Some((cv_mem.cv_nniS, cv_mem.cv_nnfS, cv_mem.cv_ncfnS))
            } else {
                None
            };
            sensi_stgr1_stats = if cv_mem.cv_ism == CV_STAGGERED1 {
                Some((
                    cv_mem.cv_nniS1[..Ns].to_vec(),
                    cv_mem.cv_nnfS1[..Ns].to_vec(),
                    cv_mem.cv_ncfnS1[..Ns].to_vec(),
                ))
            } else {
                None
            };
        } else {
            sensi_stats = None;
            sensi_nls_stats = None;
            sensi_stgr1_stats = None;
        }

        quadr_sensi_stats = if cv_mem.cv_quadr_sensi {
            Some((cv_mem.cv_nfQSe, cv_mem.cv_netfQS))
        } else {
            None
        };
    }

    /* step and method stats */
    sunfprintf_real(outfile, fmt, SUNTRUE, "Current time", tn);
    sunfprintf_long(outfile, fmt, SUNFALSE, "Steps", nst);
    sunfprintf_long(outfile, fmt, SUNFALSE, "Error test fails", netf);
    sunfprintf_long(outfile, fmt, SUNFALSE, "NLS step fails", ncfn);
    sunfprintf_long(outfile, fmt, SUNFALSE, "Constraint fails", constraint_fails);
    sunfprintf_long(
        outfile,
        fmt,
        SUNFALSE,
        "Constraint corrections",
        constraint_corrections,
    );
    sunfprintf_real(outfile, fmt, SUNFALSE, "Initial step size", h0u);
    sunfprintf_real(outfile, fmt, SUNFALSE, "Last step size", hu);
    sunfprintf_real(outfile, fmt, SUNFALSE, "Current step size", next_h);
    sunfprintf_long(outfile, fmt, SUNFALSE, "Last method order", qu as i64);
    sunfprintf_long(
        outfile,
        fmt,
        SUNFALSE,
        "Current method order",
        next_q as i64,
    );
    sunfprintf_long(outfile, fmt, SUNFALSE, "Stab. lim. order reductions", nor);
    /* function evaluations */
    sunfprintf_long(outfile, fmt, SUNFALSE, "RHS fn evals", nfe);
    /* nonlinear solver stats */
    sunfprintf_long(outfile, fmt, SUNFALSE, "NLS iters", nni);
    sunfprintf_long(outfile, fmt, SUNFALSE, "NLS fails", nnf);
    if nst > 0 {
        sunfprintf_real(
            outfile,
            fmt,
            SUNFALSE,
            "NLS iters per step",
            nni as sunrealtype / nst as sunrealtype,
        );
    }

    /* linear solver stats */
    sunfprintf_long(outfile, fmt, SUNFALSE, "LS setups", nsetups);
    if let Some((nje, nfeDQ, npe, nps, nli, ncfl, njtsetup, njtimes)) = ls_stats {
        sunfprintf_long(outfile, fmt, SUNFALSE, "Jac fn evals", nje);
        sunfprintf_long(outfile, fmt, SUNFALSE, "LS RHS fn evals", nfeDQ);
        sunfprintf_long(outfile, fmt, SUNFALSE, "Prec setup evals", npe);
        sunfprintf_long(outfile, fmt, SUNFALSE, "Prec solves", nps);
        sunfprintf_long(outfile, fmt, SUNFALSE, "LS iters", nli);
        sunfprintf_long(outfile, fmt, SUNFALSE, "LS fails", ncfl);
        sunfprintf_long(outfile, fmt, SUNFALSE, "Jac-times setups", njtsetup);
        sunfprintf_long(outfile, fmt, SUNFALSE, "Jac-times evals", njtimes);
        if nni > 0 {
            sunfprintf_real(
                outfile,
                fmt,
                SUNFALSE,
                "LS iters per NLS iter",
                nli as sunrealtype / nni as sunrealtype,
            );
            sunfprintf_real(
                outfile,
                fmt,
                SUNFALSE,
                "Jac evals per NLS iter",
                nje as sunrealtype / nni as sunrealtype,
            );
            sunfprintf_real(
                outfile,
                fmt,
                SUNFALSE,
                "Prec evals per NLS iter",
                npe as sunrealtype / nni as sunrealtype,
            );
        }
    }

    /* rootfinding stats */
    sunfprintf_long(outfile, fmt, SUNFALSE, "Root fn evals", nge);

    /* projection stats */
    if let Some((nproj, npfails)) = proj_stats {
        sunfprintf_long(outfile, fmt, SUNFALSE, "Projection fn evals", nproj);
        sunfprintf_long(outfile, fmt, SUNFALSE, "Projection fails", npfails);
    }

    /* quadrature stats */
    if let Some((nfQe, netfQ)) = quadr_stats {
        sunfprintf_long(outfile, fmt, SUNFALSE, "Quad fn evals", nfQe);
        sunfprintf_long(outfile, fmt, SUNFALSE, "Quad error test fails", netfQ);
    }

    /* sensitivity stats */
    if let Some((nfSe, nfeS, netfS, nsetupsS)) = sensi_stats {
        sunfprintf_long(outfile, fmt, SUNFALSE, "Sens fn evals", nfSe);
        sunfprintf_long(outfile, fmt, SUNFALSE, "Sens RHS fn evals", nfeS);
        sunfprintf_long(outfile, fmt, SUNFALSE, "Sens error test fails", netfS);
        if let Some((nniS, nnfS, ncfnS)) = sensi_nls_stats {
            sunfprintf_long(outfile, fmt, SUNFALSE, "Sens NLS iters", nniS);
            sunfprintf_long(outfile, fmt, SUNFALSE, "Sens NLS fails", nnfS);
            sunfprintf_long(outfile, fmt, SUNFALSE, "Sens NLS step fails", ncfnS);
        }
        if let Some((nniS1, nnfS1, ncfnS1)) = sensi_stgr1_stats {
            sunfprintf_long_array(outfile, fmt, SUNFALSE, "Sens stgr1 NLS iters", &nniS1);
            sunfprintf_long_array(outfile, fmt, SUNFALSE, "Sens stgr1 NLS fails", &nnfS1);
            sunfprintf_long_array(outfile, fmt, SUNFALSE, "Sens stgr1 NLS step fails", &ncfnS1);
        }
        sunfprintf_long(outfile, fmt, SUNFALSE, "Sens LS setups", nsetupsS);
    }

    /* quadrature-sensitivity stats */
    if let Some((nfQSe, netfQS)) = quadr_sensi_stats {
        sunfprintf_long(outfile, fmt, SUNFALSE, "QuadSens fn evals", nfQSe);
        sunfprintf_long(outfile, fmt, SUNFALSE, "QuadSens error test fails", netfQS);
    }

    CV_SUCCESS
}

/*-----------------------------------------------------------------*/

/// C `CVodeGetUserData` returns the stored `void*` without ownership
/// transfer. The safe-Rust token cannot be aliased, so the stored box is
/// SWAPPED with `user_data`; the caller must hand it back (via
/// `CVodeSetUserData` or a second swap) before the integrator next
/// invokes a user callback.
pub fn CVodeGetUserData(cvode_mem: &CVodeMem, user_data: &mut Option<Box<dyn Any>>) -> i32 {
    /* NULL-mem check: handled by type system */
    std::mem::swap(&mut cvode_mem.borrow_mut().cv_user_data, user_data);

    CV_SUCCESS
}

/*-----------------------------------------------------------------*/

/// C returns a `malloc`'d `char*`; the Rust port returns `String`.
pub fn CVodeGetReturnFlagName(flag: i64) -> String {
    let name = match flag {
        f if f == CV_SUCCESS as i64 => "CV_SUCCESS",
        f if f == CV_TSTOP_RETURN as i64 => "CV_TSTOP_RETURN",
        f if f == CV_ROOT_RETURN as i64 => "CV_ROOT_RETURN",
        f if f == CV_TOO_MUCH_WORK as i64 => "CV_TOO_MUCH_WORK",
        f if f == CV_TOO_MUCH_ACC as i64 => "CV_TOO_MUCH_ACC",
        f if f == CV_ERR_FAILURE as i64 => "CV_ERR_FAILURE",
        f if f == CV_CONV_FAILURE as i64 => "CV_CONV_FAILURE",
        f if f == CV_LINIT_FAIL as i64 => "CV_LINIT_FAIL",
        f if f == CV_LSETUP_FAIL as i64 => "CV_LSETUP_FAIL",
        f if f == CV_LSOLVE_FAIL as i64 => "CV_LSOLVE_FAIL",
        f if f == CV_RHSFUNC_FAIL as i64 => "CV_RHSFUNC_FAIL",
        f if f == CV_FIRST_RHSFUNC_ERR as i64 => "CV_FIRST_RHSFUNC_ERR",
        f if f == CV_REPTD_RHSFUNC_ERR as i64 => "CV_REPTD_RHSFUNC_ERR",
        f if f == CV_UNREC_RHSFUNC_ERR as i64 => "CV_UNREC_RHSFUNC_ERR",
        f if f == CV_RTFUNC_FAIL as i64 => "CV_RTFUNC_FAIL",
        f if f == CV_MEM_FAIL as i64 => "CV_MEM_FAIL",
        f if f == CV_MEM_NULL as i64 => "CV_MEM_NULL",
        f if f == CV_ILL_INPUT as i64 => "CV_ILL_INPUT",
        f if f == CV_NO_MALLOC as i64 => "CV_NO_MALLOC",
        f if f == CV_BAD_K as i64 => "CV_BAD_K",
        f if f == CV_BAD_T as i64 => "CV_BAD_T",
        f if f == CV_BAD_DKY as i64 => "CV_BAD_DKY",
        f if f == CV_NO_QUAD as i64 => "CV_NO_QUAD",
        f if f == CV_QRHSFUNC_FAIL as i64 => "CV_QRHSFUNC_FAIL",
        f if f == CV_FIRST_QRHSFUNC_ERR as i64 => "CV_FIRST_QRHSFUNC_ERR",
        f if f == CV_REPTD_QRHSFUNC_ERR as i64 => "CV_REPTD_QRHSFUNC_ERR",
        f if f == CV_UNREC_QRHSFUNC_ERR as i64 => "CV_UNREC_QRHSFUNC_ERR",
        f if f == CV_BAD_IS as i64 => "CV_BAD_IS",
        f if f == CV_NO_SENS as i64 => "CV_NO_SENS",
        f if f == CV_SRHSFUNC_FAIL as i64 => "CV_SRHSFUNC_FAIL",
        f if f == CV_FIRST_SRHSFUNC_ERR as i64 => "CV_FIRST_SRHSFUNC_ERR",
        f if f == CV_REPTD_SRHSFUNC_ERR as i64 => "CV_REPTD_SRHSFUNC_ERR",
        f if f == CV_UNREC_SRHSFUNC_ERR as i64 => "CV_UNREC_SRHSFUNC_ERR",
        f if f == CV_TOO_CLOSE as i64 => "CV_TOO_CLOSE",
        f if f == CV_NLS_INIT_FAIL as i64 => "CV_NLS_INIT_FAIL",
        /* upstream typo preserved: */
        f if f == CV_NLS_SETUP_FAIL as i64 => "CV_NLS_SETUPT_FAIL",
        f if f == CV_NO_ADJ as i64 => "CV_NO_ADJ",
        f if f == CV_NO_FWD as i64 => "CV_NO_FWD",
        f if f == CV_NO_BCK as i64 => "CV_NO_BCK",
        f if f == CV_BAD_TB0 as i64 => "CV_BAD_TB0",
        f if f == CV_REIFWD_FAIL as i64 => "CV_REIFWD_FAIL",
        f if f == CV_FWD_FAIL as i64 => "CV_FWD_FAIL",
        f if f == CV_GETY_BADT as i64 => "CV_GETY_BADT",
        f if f == CV_NLS_FAIL as i64 => "CV_NLS_FAIL",
        f if f == CV_PROJ_MEM_NULL as i64 => "CV_PROJ_MEM_NULL",
        f if f == CV_PROJFUNC_FAIL as i64 => "CV_PROJFUNC_FAIL",
        f if f == CV_REPTD_PROJFUNC_ERR as i64 => "CV_REPTD_PROJFUNC_ERR",
        _ => "NONE",
    };

    name.to_string()
}
