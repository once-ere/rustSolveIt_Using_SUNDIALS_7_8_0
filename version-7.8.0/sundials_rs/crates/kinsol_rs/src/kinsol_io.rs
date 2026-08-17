//! Port of `src/kinsol/kinsol_io.c` — the optional input and output
//! functions for the KINSOL solver.
//!
//! Reference build configuration notes:
//! - SUNDIALS 7.8.0 has **no** `KINSetPrintLevel` / `KINSetInfoFile` /
//!   `KINSetErrHandlerFn` / `KINSetErrFile`: the user-facing "print
//!   level" machinery was removed when KINSOL moved its informational
//!   messages onto the `SUNLogger` (`KINPrintInfo`, guarded by
//!   `SUNDIALS_LOGGING_LEVEL >= INFO`, which the reference build at
//!   level 2 compiles out). There is therefore nothing to port for
//!   those entry points and no KINSOL reference `.out` file contains
//!   logger output.
//! - `KINGetWorkSpace` is `SUNDIALS_DEPRECATED` upstream but still
//!   compiled and still called by `kinFoodWeb_kry`, so it is ported.
//!
//! Conventions applied here (see ARCHITECTURE.md):
//! - The C `kinmem == NULL` guard at the top of every function is
//!   handled by the type system (`&KINMem` cannot be NULL); the
//!   `KIN_MEM_NULL` return therefore has no reachable path and the
//!   check is dropped, exactly as the CVODE port does.
//! - Nullable C function-pointer parameters become `Option<FnType>`;
//!   nullable `N_Vector` parameters become `Option<&N_Vector>`.
//! - `void**` out-params (`KINGetUserData`) SWAP the stored
//!   `Option<Box<dyn Any>>` token with the caller's out-param, since a
//!   safe token cannot be aliased (accepted deviation class 6). The
//!   caller must hand the box back (via `KINSetUserData` or a second
//!   swap) before KINSOL next invokes a user callback.
//! - `SUNRsqrt` / `SUNRpowerR` come from `sundials_core::sundials_math`
//!   (deterministic `pow`), never `f64::powf`.
//! - Granular borrows: no `kinmem` borrow is held across
//!   `KINProcessError`, an `N_Vector` op, or a `SUNFile` write.

use std::any::Any;

use sundials_core::sundials_math::{SUNRpowerR, SUNRsqrt};
use sundials_core::sundials_nvector::{N_VClone, N_VDestroy, N_VMaxNorm, N_VScale, N_Vector};
use sundials_core::sundials_types::*;
use sundials_core::sundials_utils::{sunfprintf_long, sunfprintf_real, SUNFile};

use crate::kinsol_impl::*;
use crate::kinsol_ls::KINLsMemRec;

const ZERO: sunrealtype = 0.0;
const POINT1: sunrealtype = 0.1;
const ONETHIRD: sunrealtype = 0.3333333333333333;
const TWOTHIRDS: sunrealtype = 0.6666666666666667;
const POINT9: sunrealtype = 0.9;
const ONE: sunrealtype = 1.0;
const TWO: sunrealtype = 2.0;
const TWOPT5: sunrealtype = 2.5;

/*
 * =================================================================
 * KINSOL optional input functions
 * =================================================================
 */

/*
 * -----------------------------------------------------------------
 * Function : KINSetUserData
 * -----------------------------------------------------------------
 */

pub fn KINSetUserData(kinmem: &KINMem, user_data: Option<Box<dyn Any>>) -> i32 {
    /* NULL-mem check: handled by type system */
    kinmem.borrow_mut().kin_user_data = user_data;

    KIN_SUCCESS
}

/*
 * -----------------------------------------------------------------
 * Function : KINSetDamping
 * -----------------------------------------------------------------
 */

pub fn KINSetDamping(kinmem: &KINMem, beta: sunrealtype) -> i32 {
    /* NULL-mem check: handled by type system */

    /* check for illegal input value */
    if beta <= ZERO {
        KINProcessError(
            Some(kinmem),
            KIN_ILL_INPUT,
            line!() as i32,
            "KINSetDamping",
            file!(),
            "beta <= 0 illegal",
        );
        return KIN_ILL_INPUT;
    }

    let mut kin_mem = kinmem.borrow_mut();

    if beta < ONE {
        /* enable damping */
        kin_mem.kin_beta = beta;
        kin_mem.kin_damping = SUNTRUE;
    } else {
        /* disable damping */
        kin_mem.kin_beta = ONE;
        kin_mem.kin_damping = SUNFALSE;
    }

    KIN_SUCCESS
}

/*
 * -----------------------------------------------------------------
 * Function : KINSetMAA
 * -----------------------------------------------------------------
 */

pub fn KINSetMAA(kinmem: &KINMem, maa: i64) -> i32 {
    /* NULL-mem check: handled by type system */

    if maa < 0 {
        KINProcessError(
            Some(kinmem),
            KIN_ILL_INPUT,
            line!() as i32,
            "KINSetMAA",
            file!(),
            MSG_BAD_MAA,
        );
        return KIN_ILL_INPUT;
    }

    // To allow for setting the depth and max number of iterations in any order we
    // do not limit maa here and instead enforce maa < mxiter in the AA
    // initialization function (KINInitAA)
    kinmem.borrow_mut().kin_m_aa = maa;

    KIN_SUCCESS
}

/*
 * -----------------------------------------------------------------
 * Function : KINSetDelayAA
 * -----------------------------------------------------------------
 */

pub fn KINSetDelayAA(kinmem: &KINMem, delay: i64) -> i32 {
    /* NULL-mem check: handled by type system */

    /* check for illegal input value */
    if delay < 0 {
        KINProcessError(
            Some(kinmem),
            KIN_ILL_INPUT,
            line!() as i32,
            "KINSetDelayAA",
            file!(),
            "delay < 0 illegal",
        );
        return KIN_ILL_INPUT;
    }

    kinmem.borrow_mut().kin_delay_aa = delay;

    KIN_SUCCESS
}

/*
 * -----------------------------------------------------------------
 * Function : KINSetOrthAA
 * -----------------------------------------------------------------
 */

pub fn KINSetOrthAA(kinmem: &KINMem, orthaa: i32) -> i32 {
    /* NULL-mem check: handled by type system */

    if (orthaa < KIN_ORTH_MGS) || (orthaa > KIN_ORTH_DCGS2) {
        KINProcessError(
            Some(kinmem),
            KIN_ILL_INPUT,
            line!() as i32,
            "KINSetOrthAA",
            file!(),
            MSG_BAD_ORTHAA,
        );
        return KIN_ILL_INPUT;
    }

    kinmem.borrow_mut().kin_orth_aa = orthaa;

    KIN_SUCCESS
}

/*
 * -----------------------------------------------------------------
 * Function : KINSetDampingAA
 * -----------------------------------------------------------------
 */

pub fn KINSetDampingAA(kinmem: &KINMem, beta: sunrealtype) -> i32 {
    /* NULL-mem check: handled by type system */

    /* check for illegal input value */
    if beta <= ZERO {
        KINProcessError(
            Some(kinmem),
            KIN_ILL_INPUT,
            line!() as i32,
            "KINSetDampingAA",
            file!(),
            "beta <= 0 illegal",
        );
        return KIN_ILL_INPUT;
    }

    let mut kin_mem = kinmem.borrow_mut();

    if beta < ONE {
        /* enable damping */
        kin_mem.kin_beta_aa = beta;
        kin_mem.kin_damping_aa = SUNTRUE;
    } else {
        /* disable damping */
        kin_mem.kin_beta_aa = ONE;
        kin_mem.kin_damping_aa = SUNFALSE;
    }

    KIN_SUCCESS
}

/*
 * -----------------------------------------------------------------
 * Function : KINSetDampingFn
 * -----------------------------------------------------------------
 */

pub fn KINSetDampingFn(kinmem: &KINMem, damping_fn: Option<KINDampingFn>) -> i32 {
    /* NULL-mem check: handled by type system */
    kinmem.borrow_mut().kin_damping_fn = damping_fn;

    KIN_SUCCESS
}

/*
 * -----------------------------------------------------------------
 * Function : KINSetDepthFn
 * -----------------------------------------------------------------
 */

pub fn KINSetDepthFn(kinmem: &KINMem, depth_fn: Option<KINDepthFn>) -> i32 {
    /* NULL-mem check: handled by type system */
    kinmem.borrow_mut().kin_depth_fn = depth_fn;

    KIN_SUCCESS
}

/*
 * -----------------------------------------------------------------
 * Function : KINSetReturnNewest
 * -----------------------------------------------------------------
 */

pub fn KINSetReturnNewest(kinmem: &KINMem, ret_newest: sunbooleantype) -> i32 {
    /* NULL-mem check: handled by type system */
    kinmem.borrow_mut().kin_ret_newest = ret_newest;

    KIN_SUCCESS
}

/*
 * -----------------------------------------------------------------
 * Function : KINSetNumMaxIters
 * -----------------------------------------------------------------
 */

pub fn KINSetNumMaxIters(kinmem: &KINMem, mxiter: i64) -> i32 {
    /* NULL-mem check: handled by type system */

    if mxiter < 0 {
        KINProcessError(
            Some(kinmem),
            KIN_ILL_INPUT,
            line!() as i32,
            "KINSetNumMaxIters",
            file!(),
            MSG_BAD_MXITER,
        );
        return KIN_ILL_INPUT;
    }

    if mxiter == 0 {
        kinmem.borrow_mut().kin_mxiter = MXITER_DEFAULT;
    } else {
        kinmem.borrow_mut().kin_mxiter = mxiter;
    }

    KIN_SUCCESS
}

/*
 * -----------------------------------------------------------------
 * Function : KINSetNoInitSetup
 * -----------------------------------------------------------------
 */

pub fn KINSetNoInitSetup(kinmem: &KINMem, noInitSetup: sunbooleantype) -> i32 {
    /* NULL-mem check: handled by type system */
    kinmem.borrow_mut().kin_noInitSetup = noInitSetup;

    KIN_SUCCESS
}

/*
 * -----------------------------------------------------------------
 * Function : KINSetNoResMon
 * -----------------------------------------------------------------
 */

pub fn KINSetNoResMon(kinmem: &KINMem, noResMon: sunbooleantype) -> i32 {
    /* NULL-mem check: handled by type system */
    kinmem.borrow_mut().kin_noResMon = noResMon;

    KIN_SUCCESS
}

/*
 * -----------------------------------------------------------------
 * Function : KINSetMaxSetupCalls
 * -----------------------------------------------------------------
 */

pub fn KINSetMaxSetupCalls(kinmem: &KINMem, msbset: i64) -> i32 {
    /* NULL-mem check: handled by type system */

    if msbset < 0 {
        KINProcessError(
            Some(kinmem),
            KIN_ILL_INPUT,
            line!() as i32,
            "KINSetMaxSetupCalls",
            file!(),
            MSG_BAD_MSBSET,
        );
        return KIN_ILL_INPUT;
    }

    if msbset == 0 {
        kinmem.borrow_mut().kin_msbset = MSBSET_DEFAULT;
    } else {
        kinmem.borrow_mut().kin_msbset = msbset;
    }

    KIN_SUCCESS
}

/*
 * -----------------------------------------------------------------
 * Function : KINSetMaxSubSetupCalls
 * -----------------------------------------------------------------
 */

pub fn KINSetMaxSubSetupCalls(kinmem: &KINMem, msbsetsub: i64) -> i32 {
    /* NULL-mem check: handled by type system */

    if msbsetsub < 0 {
        KINProcessError(
            Some(kinmem),
            KIN_ILL_INPUT,
            line!() as i32,
            "KINSetMaxSubSetupCalls",
            file!(),
            MSG_BAD_MSBSETSUB,
        );
        return KIN_ILL_INPUT;
    }

    if msbsetsub == 0 {
        kinmem.borrow_mut().kin_msbset_sub = MSBSET_SUB_DEFAULT;
    } else {
        kinmem.borrow_mut().kin_msbset_sub = msbsetsub;
    }

    KIN_SUCCESS
}

/*
 * -----------------------------------------------------------------
 * Function : KINSetEtaForm
 * -----------------------------------------------------------------
 */

pub fn KINSetEtaForm(kinmem: &KINMem, etachoice: i32) -> i32 {
    /* NULL-mem check: handled by type system */

    if (etachoice != KIN_ETACONSTANT)
        && (etachoice != KIN_ETACHOICE1)
        && (etachoice != KIN_ETACHOICE2)
    {
        KINProcessError(
            Some(kinmem),
            KIN_ILL_INPUT,
            line!() as i32,
            "KINSetEtaForm",
            file!(),
            MSG_BAD_ETACHOICE,
        );
        return KIN_ILL_INPUT;
    }

    kinmem.borrow_mut().kin_etaflag = etachoice;

    KIN_SUCCESS
}

/*
 * -----------------------------------------------------------------
 * Function : KINSetEtaConstValue
 * -----------------------------------------------------------------
 */

pub fn KINSetEtaConstValue(kinmem: &KINMem, eta: sunrealtype) -> i32 {
    /* NULL-mem check: handled by type system */

    if (eta < ZERO) || (eta > ONE) {
        KINProcessError(
            Some(kinmem),
            KIN_ILL_INPUT,
            line!() as i32,
            "KINSetEtaConstValue",
            file!(),
            MSG_BAD_ETACONST,
        );
        return KIN_ILL_INPUT;
    }

    if eta == ZERO {
        kinmem.borrow_mut().kin_eta = POINT1;
    } else {
        kinmem.borrow_mut().kin_eta = eta;
    }

    KIN_SUCCESS
}

/*
 * -----------------------------------------------------------------
 * Function : KINSetEtaParams
 * -----------------------------------------------------------------
 */

pub fn KINSetEtaParams(kinmem: &KINMem, egamma: sunrealtype, ealpha: sunrealtype) -> i32 {
    /* NULL-mem check: handled by type system */

    if (ealpha <= ONE) || (ealpha > TWO) {
        if ealpha != ZERO {
            KINProcessError(
                Some(kinmem),
                KIN_ILL_INPUT,
                line!() as i32,
                "KINSetEtaParams",
                file!(),
                MSG_BAD_ALPHA,
            );
            return KIN_ILL_INPUT;
        }
    }

    if ealpha == ZERO {
        kinmem.borrow_mut().kin_eta_alpha = TWO;
    } else {
        kinmem.borrow_mut().kin_eta_alpha = ealpha;
    }

    if (egamma <= ZERO) || (egamma > ONE) {
        if egamma != ZERO {
            KINProcessError(
                Some(kinmem),
                KIN_ILL_INPUT,
                line!() as i32,
                "KINSetEtaParams",
                file!(),
                MSG_BAD_GAMMA,
            );
            return KIN_ILL_INPUT;
        }
    }

    if egamma == ZERO {
        kinmem.borrow_mut().kin_eta_gamma = POINT9;
    } else {
        kinmem.borrow_mut().kin_eta_gamma = egamma;
    }

    KIN_SUCCESS
}

/*
 * -----------------------------------------------------------------
 * Function : KINSetResMonParams
 * -----------------------------------------------------------------
 */

pub fn KINSetResMonParams(kinmem: &KINMem, omegamin: sunrealtype, omegamax: sunrealtype) -> i32 {
    /* NULL-mem check: handled by type system */

    /* check omegamin */

    if omegamin < ZERO {
        KINProcessError(
            Some(kinmem),
            KIN_ILL_INPUT,
            line!() as i32,
            "KINSetResMonParams",
            file!(),
            MSG_BAD_OMEGA,
        );
        return KIN_ILL_INPUT;
    }

    if omegamin == ZERO {
        kinmem.borrow_mut().kin_omega_min = OMEGA_MIN;
    } else {
        kinmem.borrow_mut().kin_omega_min = omegamin;
    }

    /* check omegamax */

    if omegamax < ZERO {
        KINProcessError(
            Some(kinmem),
            KIN_ILL_INPUT,
            line!() as i32,
            "KINSetResMonParams",
            file!(),
            MSG_BAD_OMEGA,
        );
        return KIN_ILL_INPUT;
    }

    let omega_min = kinmem.borrow().kin_omega_min;

    if omegamax == ZERO {
        if omega_min > OMEGA_MAX {
            KINProcessError(
                Some(kinmem),
                KIN_ILL_INPUT,
                line!() as i32,
                "KINSetResMonParams",
                file!(),
                MSG_BAD_OMEGA,
            );
            return KIN_ILL_INPUT;
        } else {
            kinmem.borrow_mut().kin_omega_max = OMEGA_MAX;
        }
    } else {
        if omega_min > omegamax {
            KINProcessError(
                Some(kinmem),
                KIN_ILL_INPUT,
                line!() as i32,
                "KINSetResMonParams",
                file!(),
                MSG_BAD_OMEGA,
            );
            return KIN_ILL_INPUT;
        } else {
            kinmem.borrow_mut().kin_omega_max = omegamax;
        }
    }

    KIN_SUCCESS
}

/*
 * -----------------------------------------------------------------
 * Function : KINSetResMonConstValue
 * -----------------------------------------------------------------
 */

pub fn KINSetResMonConstValue(kinmem: &KINMem, omegaconst: sunrealtype) -> i32 {
    /* NULL-mem check: handled by type system */

    /* check omegaconst */

    if omegaconst < ZERO {
        KINProcessError(
            Some(kinmem),
            KIN_ILL_INPUT,
            line!() as i32,
            "KINSetResMonConstValue",
            file!(),
            MSG_BAD_OMEGA,
        );
        return KIN_ILL_INPUT;
    }

    /* Load omega value. A value of 0 will force using omega_min and omega_max */
    kinmem.borrow_mut().kin_omega = omegaconst;

    KIN_SUCCESS
}

/*
 * -----------------------------------------------------------------
 * Function : KINSetNoMinEps
 * -----------------------------------------------------------------
 */

pub fn KINSetNoMinEps(kinmem: &KINMem, noMinEps: sunbooleantype) -> i32 {
    /* NULL-mem check: handled by type system */
    kinmem.borrow_mut().kin_noMinEps = noMinEps;

    KIN_SUCCESS
}

/*
 * -----------------------------------------------------------------
 * Function : KINSetMaxNewtonStep
 * -----------------------------------------------------------------
 */

pub fn KINSetMaxNewtonStep(kinmem: &KINMem, mxnewtstep: sunrealtype) -> i32 {
    /* NULL-mem check: handled by type system */

    if mxnewtstep < ZERO {
        KINProcessError(
            Some(kinmem),
            KIN_ILL_INPUT,
            line!() as i32,
            "KINSetMaxNewtonStep",
            file!(),
            MSG_BAD_MXNEWTSTEP,
        );
        return KIN_ILL_INPUT;
    }

    /* Note: passing a value of 0.0 will use the default
    value (computed in KINSolInit) */

    kinmem.borrow_mut().kin_mxnstepin = mxnewtstep;

    KIN_SUCCESS
}

/*
 * -----------------------------------------------------------------
 * Function : KINSetMaxBetaFails
 * -----------------------------------------------------------------
 */

pub fn KINSetMaxBetaFails(kinmem: &KINMem, mxnbcf: i64) -> i32 {
    /* NULL-mem check: handled by type system */

    if mxnbcf < 0 {
        KINProcessError(
            Some(kinmem),
            KIN_ILL_INPUT,
            line!() as i32,
            "KINSetMaxBetaFails",
            file!(),
            MSG_BAD_MXNBCF,
        );
        return KIN_ILL_INPUT;
    }

    if mxnbcf == 0 {
        kinmem.borrow_mut().kin_mxnbcf = MXNBCF_DEFAULT;
    } else {
        kinmem.borrow_mut().kin_mxnbcf = mxnbcf;
    }

    KIN_SUCCESS
}

/*
 * -----------------------------------------------------------------
 * Function : KINSetRelErrFunc
 * -----------------------------------------------------------------
 */

pub fn KINSetRelErrFunc(kinmem: &KINMem, relfunc: sunrealtype) -> i32 {
    /* NULL-mem check: handled by type system */
    let uround: sunrealtype;

    if relfunc < ZERO {
        KINProcessError(
            Some(kinmem),
            KIN_ILL_INPUT,
            line!() as i32,
            "KINSetRelErrFunc",
            file!(),
            MSG_BAD_RELFUNC,
        );
        return KIN_ILL_INPUT;
    }

    if relfunc == ZERO {
        uround = kinmem.borrow().kin_uround;
        kinmem.borrow_mut().kin_sqrt_relfunc = SUNRsqrt(uround);
    } else {
        kinmem.borrow_mut().kin_sqrt_relfunc = SUNRsqrt(relfunc);
    }

    KIN_SUCCESS
}

/*
 * -----------------------------------------------------------------
 * Function : KINSetFuncNormTol
 * -----------------------------------------------------------------
 */

pub fn KINSetFuncNormTol(kinmem: &KINMem, fnormtol: sunrealtype) -> i32 {
    /* NULL-mem check: handled by type system */
    let uround: sunrealtype;

    if fnormtol < ZERO {
        KINProcessError(
            Some(kinmem),
            KIN_ILL_INPUT,
            line!() as i32,
            "KINSetFuncNormTol",
            file!(),
            MSG_BAD_FNORMTOL,
        );
        return KIN_ILL_INPUT;
    }

    if fnormtol == ZERO {
        uround = kinmem.borrow().kin_uround;
        kinmem.borrow_mut().kin_fnormtol = SUNRpowerR(uround, ONETHIRD);
    } else {
        kinmem.borrow_mut().kin_fnormtol = fnormtol;
    }

    KIN_SUCCESS
}

/*
 * -----------------------------------------------------------------
 * Function : KINSetScaledStepTol
 * -----------------------------------------------------------------
 */

pub fn KINSetScaledStepTol(kinmem: &KINMem, scsteptol: sunrealtype) -> i32 {
    /* NULL-mem check: handled by type system */
    let uround: sunrealtype;

    if scsteptol < ZERO {
        KINProcessError(
            Some(kinmem),
            KIN_ILL_INPUT,
            line!() as i32,
            "KINSetScaledStepTol",
            file!(),
            MSG_BAD_SCSTEPTOL,
        );
        return KIN_ILL_INPUT;
    }

    if scsteptol == ZERO {
        uround = kinmem.borrow().kin_uround;
        kinmem.borrow_mut().kin_scsteptol = SUNRpowerR(uround, TWOTHIRDS);
    } else {
        kinmem.borrow_mut().kin_scsteptol = scsteptol;
    }

    KIN_SUCCESS
}

/*
 * -----------------------------------------------------------------
 * Function : KINSetConstraints
 * -----------------------------------------------------------------
 */

pub fn KINSetConstraints(kinmem: &KINMem, constraints: Option<&N_Vector>) -> i32 {
    /* NULL-mem check: handled by type system */

    let constraints = match constraints {
        None => {
            /* C destroys the stored vector but leaves kin_constraints
            dangling (upstream: only kin_constraintsSet guards later use);
            the safe port takes the handle out as well. */
            let constraintsSet = kinmem.borrow().kin_constraintsSet;
            if constraintsSet {
                let taken = kinmem.borrow_mut().kin_constraints.take();
                if let Some(c) = taken {
                    N_VDestroy(c);
                }
                let mut kin_mem = kinmem.borrow_mut();
                let lrw1 = kin_mem.kin_lrw1;
                let liw1 = kin_mem.kin_liw1;
                kin_mem.kin_lrw -= lrw1;
                kin_mem.kin_liw -= liw1;
            }
            kinmem.borrow_mut().kin_constraintsSet = SUNFALSE;
            return KIN_SUCCESS;
        }
        Some(c) => c,
    };

    /* Check the constraints vector */

    let temptest = N_VMaxNorm(constraints);
    if temptest > TWOPT5 {
        KINProcessError(
            Some(kinmem),
            KIN_ILL_INPUT,
            line!() as i32,
            "KINSetConstraints",
            file!(),
            MSG_BAD_CONSTRAINTS,
        );
        return KIN_ILL_INPUT;
    }

    let constraintsSet = kinmem.borrow().kin_constraintsSet;
    if !constraintsSet {
        /* C does not check the N_VClone result; a NULL clone would be
        dereferenced by the N_VScale below (C UB -> deterministic panic,
        accepted deviation class 5). */
        let cloned = N_VClone(constraints);
        let mut kin_mem = kinmem.borrow_mut();
        kin_mem.kin_constraints = cloned;
        let lrw1 = kin_mem.kin_lrw1;
        let liw1 = kin_mem.kin_liw1;
        kin_mem.kin_lrw += lrw1;
        kin_mem.kin_liw += liw1;
        kin_mem.kin_constraintsSet = SUNTRUE;
    }

    /* Load the constraint vector */

    let target = kinmem
        .borrow()
        .kin_constraints
        .as_ref()
        .expect("kin_constraints (C: N_VClone returned NULL)")
        .clone();
    N_VScale(ONE, constraints, &target);

    KIN_SUCCESS
}

/*
 * -----------------------------------------------------------------
 * Function : KINSetSysFunc
 * -----------------------------------------------------------------
 */

pub fn KINSetSysFunc(kinmem: &KINMem, func: Option<KINSysFn>) -> i32 {
    /* NULL-mem check: handled by type system */

    if func.is_none() {
        KINProcessError(
            Some(kinmem),
            KIN_ILL_INPUT,
            line!() as i32,
            "KINSetSysFunc",
            file!(),
            MSG_FUNC_NULL,
        );
        return KIN_ILL_INPUT;
    }

    kinmem.borrow_mut().kin_func = func;

    KIN_SUCCESS
}

/*
 * =================================================================
 * KINSOL optional output functions
 * =================================================================
 */

/*
 * -----------------------------------------------------------------
 * Function : KINGetWorkSpace
 * -----------------------------------------------------------------
 */

pub fn KINGetWorkSpace(kinmem: &KINMem, lenrw: &mut i64, leniw: &mut i64) -> i32 {
    /* NULL-mem check: handled by type system */
    let kin_mem = kinmem.borrow();

    *lenrw = kin_mem.kin_lrw;
    *leniw = kin_mem.kin_liw;

    KIN_SUCCESS
}

/*
 * -----------------------------------------------------------------
 * Function : KINGetNumNonlinSolvIters
 * -----------------------------------------------------------------
 */

pub fn KINGetNumNonlinSolvIters(kinmem: &KINMem, nniters: &mut i64) -> i32 {
    /* NULL-mem check: handled by type system */
    *nniters = kinmem.borrow().kin_nni;

    KIN_SUCCESS
}

/*
 * -----------------------------------------------------------------
 * Function : KINGetNumFuncEvals
 * -----------------------------------------------------------------
 */

pub fn KINGetNumFuncEvals(kinmem: &KINMem, nfevals: &mut i64) -> i32 {
    /* NULL-mem check: handled by type system */
    *nfevals = kinmem.borrow().kin_nfe;

    KIN_SUCCESS
}

/*
 * -----------------------------------------------------------------
 * Function : KINGetNumBetaCondFails
 * -----------------------------------------------------------------
 */

pub fn KINGetNumBetaCondFails(kinmem: &KINMem, nbcfails: &mut i64) -> i32 {
    /* NULL-mem check: handled by type system */
    *nbcfails = kinmem.borrow().kin_nbcf;

    KIN_SUCCESS
}

/*
 * -----------------------------------------------------------------
 * Function : KINGetNumBacktrackOps
 * -----------------------------------------------------------------
 */

pub fn KINGetNumBacktrackOps(kinmem: &KINMem, nbacktr: &mut i64) -> i32 {
    /* NULL-mem check: handled by type system */
    *nbacktr = kinmem.borrow().kin_nbktrk;

    KIN_SUCCESS
}

/*
 * -----------------------------------------------------------------
 * Function : KINGetFuncNorm
 * -----------------------------------------------------------------
 */

pub fn KINGetFuncNorm(kinmem: &KINMem, funcnorm: &mut sunrealtype) -> i32 {
    /* NULL-mem check: handled by type system */
    *funcnorm = kinmem.borrow().kin_fnorm;

    KIN_SUCCESS
}

/*
 * -----------------------------------------------------------------
 * Function : KINGetStepLength
 * -----------------------------------------------------------------
 */

pub fn KINGetStepLength(kinmem: &KINMem, steplength: &mut sunrealtype) -> i32 {
    /* NULL-mem check: handled by type system */
    *steplength = kinmem.borrow().kin_stepl;

    KIN_SUCCESS
}

/*
 * -----------------------------------------------------------------
 * Function : KINGetUserData
 * -----------------------------------------------------------------
 */

/// C `KINGetUserData` returns the stored `void*` without ownership
/// transfer. The safe-Rust token cannot be aliased, so the stored box is
/// SWAPPED with `user_data`; the caller must hand it back (via
/// `KINSetUserData` or a second swap) before KINSOL next invokes a user
/// callback (accepted deviation class 6).
pub fn KINGetUserData(kinmem: &KINMem, user_data: &mut Option<Box<dyn Any>>) -> i32 {
    /* NULL-mem check: handled by type system */
    std::mem::swap(&mut kinmem.borrow_mut().kin_user_data, user_data);

    KIN_SUCCESS
}

/*
 * -----------------------------------------------------------------
 * Function : KINPrintAllStats
 * -----------------------------------------------------------------
 */

pub fn KINPrintAllStats(kinmem: &KINMem, outfile: &SUNFile, fmt: SUNOutputFormat) -> i32 {
    /* NULL-mem check: handled by type system */

    if fmt != SUNOutputFormat::SUN_OUTPUTFORMAT_TABLE
        && fmt != SUNOutputFormat::SUN_OUTPUTFORMAT_CSV
    {
        KINProcessError(
            Some(kinmem),
            KIN_ILL_INPUT,
            line!() as i32,
            "KINPrintAllStats",
            file!(),
            "Invalid formatting option.",
        );
        return KIN_ILL_INPUT;
    }

    /* Copy every statistic out under one borrow (printing must not hold a
    borrow of the mem), then print in the exact C order. */
    let (nni, nfe, nbcf, nbktrk, fnorm, stepl);
    let ls_stats: Option<(i64, i64, i64, i64, i64, i64, i64)>;
    {
        let kin_mem = kinmem.borrow();
        nni = kin_mem.kin_nni;
        nfe = kin_mem.kin_nfe;
        nbcf = kin_mem.kin_nbcf;
        nbktrk = kin_mem.kin_nbktrk;
        fnorm = kin_mem.kin_fnorm;
        stepl = kin_mem.kin_stepl;

        ls_stats = kin_mem.kin_lmem.as_ref().map(|lmem| {
            let kinls_mem = lmem
                .downcast_ref::<KINLsMemRec>()
                .expect("kin_lmem holds KINLsMemRec");
            (
                kinls_mem.nje,
                kinls_mem.nfeDQ,
                kinls_mem.npe,
                kinls_mem.nps,
                kinls_mem.nli,
                kinls_mem.ncfl,
                kinls_mem.njtimes,
            )
        });
    }

    sunfprintf_long(outfile, fmt, SUNTRUE, "Nonlinear iters", nni);
    sunfprintf_long(outfile, fmt, SUNFALSE, "Nonlinear fn evals", nfe);
    sunfprintf_long(outfile, fmt, SUNFALSE, "Beta condition fails", nbcf);
    sunfprintf_long(outfile, fmt, SUNFALSE, "Backtrack operations", nbktrk);
    sunfprintf_real(outfile, fmt, SUNFALSE, "Nonlinear fn norm", fnorm);
    sunfprintf_real(outfile, fmt, SUNFALSE, "Step length", stepl);

    /* linear solver stats */
    if let Some((nje, nfeDQ, npe, nps, nli, ncfl, njtimes)) = ls_stats {
        sunfprintf_long(outfile, fmt, SUNFALSE, "Jac fn evals", nje);
        sunfprintf_long(outfile, fmt, SUNFALSE, "LS Nonlinear fn evals", nfeDQ);
        sunfprintf_long(outfile, fmt, SUNFALSE, "Prec setup evals", npe);
        sunfprintf_long(outfile, fmt, SUNFALSE, "Prec solves", nps);
        sunfprintf_long(outfile, fmt, SUNFALSE, "LS iters", nli);
        sunfprintf_long(outfile, fmt, SUNFALSE, "LS fails", ncfl);
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

    KIN_SUCCESS
}

/*
 * -----------------------------------------------------------------
 * Function : KINGetReturnFlagName
 * -----------------------------------------------------------------
 */

/// C returns a `malloc`'d `char*`; the Rust port returns `String`.
///
/// Note the upstream switch has no `KIN_DEPTH_FN_ERR` case — that flag
/// decodes to `"NONE"` via the `default` branch, preserved here.
pub fn KINGetReturnFlagName(flag: i64) -> String {
    let name = match flag {
        f if f == KIN_SUCCESS as i64 => "KIN_SUCCESS",
        f if f == KIN_INITIAL_GUESS_OK as i64 => "KIN_INITIAL_GUESS_OK",
        f if f == KIN_STEP_LT_STPTOL as i64 => "KIN_STEP_LT_STPTOL",
        f if f == KIN_WARNING as i64 => "KIN_WARNING",
        f if f == KIN_MEM_NULL as i64 => "KIN_MEM_NULL",
        f if f == KIN_ILL_INPUT as i64 => "KIN_ILL_INPUT",
        f if f == KIN_NO_MALLOC as i64 => "KIN_NO_MALLOC",
        f if f == KIN_MEM_FAIL as i64 => "KIN_MEM_FAIL",
        f if f == KIN_LINESEARCH_NONCONV as i64 => "KIN_LINESEARCH_NONCONV",
        f if f == KIN_MAXITER_REACHED as i64 => "KIN_MAXITER_REACHED",
        f if f == KIN_MXNEWT_5X_EXCEEDED as i64 => "KIN_MXNEWT_5X_EXCEEDED",
        f if f == KIN_LINESEARCH_BCFAIL as i64 => "KIN_LINESEARCH_BCFAIL",
        f if f == KIN_LINSOLV_NO_RECOVERY as i64 => "KIN_LINSOLV_NO_RECOVERY",
        f if f == KIN_LINIT_FAIL as i64 => "KIN_LINIT_FAIL",
        f if f == KIN_LSETUP_FAIL as i64 => "KIN_LSETUP_FAIL",
        f if f == KIN_LSOLVE_FAIL as i64 => "KIN_LSOLVE_FAIL",
        f if f == KIN_SYSFUNC_FAIL as i64 => "KIN_SYSFUNC_FAIL",
        f if f == KIN_FIRST_SYSFUNC_ERR as i64 => "KIN_FIRST_SYSFUNC_ERR",
        f if f == KIN_REPTD_SYSFUNC_ERR as i64 => "KIN_REPTD_SYSFUNC_ERR",
        f if f == KIN_VECTOROP_ERR as i64 => "KIN_VECTOROP_ERR",
        f if f == KIN_CONTEXT_ERR as i64 => "KIN_CONTEXT_ERR",
        f if f == KIN_DAMPING_FN_ERR as i64 => "KIN_DAMPING_FN_ERR",
        _ => "NONE",
    };

    name.to_string()
}
