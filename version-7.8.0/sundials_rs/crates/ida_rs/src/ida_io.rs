//! Port of `src/ida/ida_io.c` — the optional input and output functions
//! for the IDA solver.
//!
//! Fragment protocol (see `ida_impl.rs`): the module-scope `#define`s that
//! `ida_io.c` repeats (`ZERO`, `HALF`, `ONE`, `TWOPT5`) are NOT redefined
//! here; they come from `crate::ida_impl::*`.
//!
//! Every C entry point takes `void* ida_mem` and opens with a NULL check
//! emitting `MSG_NO_MEM`. The Rust port takes `&IDAMem`, so that branch is
//! unreachable and is elided at translation time (noted at each site).
//!
//! `void*` out-params (`IDAGetUserData`): C returns the stored pointer
//! without ownership transfer; the safe-Rust `Option<Box<dyn Any>>` token
//! cannot be aliased, so the function SWAPS the stored token with the
//! caller's out-param. Callers must hand the box back (via `IDASetUserData`
//! or a second swap) before the integrator next invokes a user callback.
//!
//! Borrow discipline: a `RefCell` borrow of the mem is never held across
//! `IDAProcessError`, an `N_Vector` op, or a nonlinear-solver call — each
//! such site copies the needed fields into locals, drops the guard, then
//! calls.

use std::any::Any;

use sundials_core::sundials_math::SUNMIN;
use sundials_core::sundials_nonlinearsolver::SUNNonlinSolSetMaxIters;
use sundials_core::sundials_nvector::*;
use sundials_core::sundials_types::*;
use sundials_core::sundials_utils::*;

use crate::ida_impl::*;
use crate::ida_ls::IDALsMemRec;

/*
 * =================================================================
 * IDA optional input functions
 * =================================================================
 */

pub fn IDASetDeltaCjLSetup(ida_mem: &IDAMem, dcj: sunrealtype) -> i32 {
    /* NULL-mem check: handled by type system */
    let mut IDA_mem = ida_mem.borrow_mut();

    if dcj < ZERO || dcj >= ONE {
        IDA_mem.ida_dcj = DCJ_DEFAULT;
    } else {
        IDA_mem.ida_dcj = dcj;
    }

    IDA_SUCCESS
}

/*-----------------------------------------------------------------*/

pub fn IDASetUserData(ida_mem: &IDAMem, user_data: Option<Box<dyn Any>>) -> i32 {
    /* NULL-mem check: handled by type system */
    ida_mem.borrow_mut().ida_user_data = user_data;

    IDA_SUCCESS
}

/*-----------------------------------------------------------------*/

pub fn IDASetEtaFixedStepBounds(
    ida_mem: &IDAMem,
    eta_min_fx: sunrealtype,
    eta_max_fx: sunrealtype,
) -> i32 {
    /* NULL-mem check: handled by type system */
    let mut IDA_mem = ida_mem.borrow_mut();

    /* set allowed value or use default */
    if eta_min_fx >= ZERO && eta_min_fx <= ONE {
        IDA_mem.ida_eta_min_fx = eta_min_fx;
    } else {
        IDA_mem.ida_eta_min_fx = ETA_MIN_FX_DEFAULT;
    }

    if eta_max_fx >= ONE {
        IDA_mem.ida_eta_max_fx = eta_max_fx;
    } else {
        IDA_mem.ida_eta_max_fx = ETA_MAX_FX_DEFAULT;
    }

    IDA_SUCCESS
}

/*-----------------------------------------------------------------*/

pub fn IDASetEtaMax(ida_mem: &IDAMem, eta_max: sunrealtype) -> i32 {
    /* NULL-mem check: handled by type system */
    let mut IDA_mem = ida_mem.borrow_mut();

    /* set allowed value or use default */
    if eta_max <= ONE {
        IDA_mem.ida_eta_max = ETA_MAX_DEFAULT;
    } else {
        IDA_mem.ida_eta_max = eta_max;
    }

    IDA_SUCCESS
}

/*-----------------------------------------------------------------*/

pub fn IDASetEtaMin(ida_mem: &IDAMem, eta_min: sunrealtype) -> i32 {
    /* NULL-mem check: handled by type system */
    let mut IDA_mem = ida_mem.borrow_mut();

    /* set allowed value or use default */
    if eta_min <= ZERO || eta_min >= ONE {
        IDA_mem.ida_eta_min = ETA_MIN_DEFAULT;
    } else {
        IDA_mem.ida_eta_min = eta_min;
    }

    IDA_SUCCESS
}

/*-----------------------------------------------------------------*/

pub fn IDASetEtaLow(ida_mem: &IDAMem, eta_low: sunrealtype) -> i32 {
    /* NULL-mem check: handled by type system */
    let mut IDA_mem = ida_mem.borrow_mut();

    /* set allowed value or use default */
    if eta_low <= ZERO || eta_low >= ONE {
        IDA_mem.ida_eta_low = ETA_LOW_DEFAULT;
    } else {
        IDA_mem.ida_eta_low = eta_low;
    }

    IDA_SUCCESS
}

/*-----------------------------------------------------------------*/

pub fn IDASetEtaMinErrFail(ida_mem: &IDAMem, eta_min_ef: sunrealtype) -> i32 {
    /* NULL-mem check: handled by type system */
    let mut IDA_mem = ida_mem.borrow_mut();

    /* set allowed value or use default */
    if eta_min_ef <= ZERO || eta_min_ef >= ONE {
        IDA_mem.ida_eta_min_ef = ETA_MIN_EF_DEFAULT;
    } else {
        IDA_mem.ida_eta_min_ef = eta_min_ef;
    }

    IDA_SUCCESS
}

/*-----------------------------------------------------------------*/

pub fn IDASetEtaConvFail(ida_mem: &IDAMem, eta_cf: sunrealtype) -> i32 {
    /* NULL-mem check: handled by type system */
    let mut IDA_mem = ida_mem.borrow_mut();

    /* set allowed value or use default */
    if eta_cf <= ZERO || eta_cf >= ONE {
        IDA_mem.ida_eta_cf = ETA_CF_DEFAULT;
    } else {
        IDA_mem.ida_eta_cf = eta_cf;
    }

    IDA_SUCCESS
}

/*-----------------------------------------------------------------*/

pub fn IDASetMaxOrd(ida_mem: &IDAMem, maxord: i32) -> i32 {
    /* NULL-mem check: handled by type system */
    if maxord <= 0 {
        IDAProcessError(
            Some(ida_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDASetMaxOrd",
            file!(),
            MSG_NEG_MAXORD,
        );
        return IDA_ILL_INPUT;
    }

    /* Cannot increase maximum order beyond the value that
    was used when allocating memory */
    let maxord_alloc = ida_mem.borrow().ida_maxord_alloc;

    if maxord > maxord_alloc {
        IDAProcessError(
            Some(ida_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDASetMaxOrd",
            file!(),
            MSG_BAD_MAXORD,
        );
        return IDA_ILL_INPUT;
    }

    ida_mem.borrow_mut().ida_maxord = SUNMIN(maxord, MAXORD_DEFAULT as i32);

    IDA_SUCCESS
}

/*-----------------------------------------------------------------*/

pub fn IDASetMaxNumSteps(ida_mem: &IDAMem, mxsteps: i64) -> i32 {
    /* NULL-mem check: handled by type system */
    let mut IDA_mem = ida_mem.borrow_mut();

    /* Passing mxsteps=0 sets the default. Passing mxsteps<0 disables the test. */

    if mxsteps == 0 {
        IDA_mem.ida_mxstep = MXSTEP_DEFAULT;
    } else {
        IDA_mem.ida_mxstep = mxsteps;
    }

    IDA_SUCCESS
}

/*-----------------------------------------------------------------*/

pub fn IDASetInitStep(ida_mem: &IDAMem, hin: sunrealtype) -> i32 {
    /* NULL-mem check: handled by type system */
    ida_mem.borrow_mut().ida_hin = hin;

    IDA_SUCCESS
}

/*-----------------------------------------------------------------*/

pub fn IDASetMaxStep(ida_mem: &IDAMem, hmax: sunrealtype) -> i32 {
    /* NULL-mem check: handled by type system */
    if hmax < ZERO {
        IDAProcessError(
            Some(ida_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDASetMaxStep",
            file!(),
            MSG_NEG_HMAX,
        );
        return IDA_ILL_INPUT;
    }

    /* Passing 0 sets hmax = infinity */
    if hmax == ZERO {
        ida_mem.borrow_mut().ida_hmax_inv = HMAX_INV_DEFAULT;
        return IDA_SUCCESS;
    }

    ida_mem.borrow_mut().ida_hmax_inv = ONE / hmax;

    IDA_SUCCESS
}

/*-----------------------------------------------------------------*/

pub fn IDASetMinStep(ida_mem: &IDAMem, hmin: sunrealtype) -> i32 {
    /* NULL-mem check: handled by type system */
    if hmin < ZERO {
        IDAProcessError(
            Some(ida_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDASetMinStep",
            file!(),
            MSG_NEG_HMIN,
        );
        return IDA_ILL_INPUT;
    }

    /* Passing 0 sets hmin = zero */
    if hmin == ZERO {
        ida_mem.borrow_mut().ida_hmin = HMIN_DEFAULT;
        return IDA_SUCCESS;
    }

    ida_mem.borrow_mut().ida_hmin = hmin;

    IDA_SUCCESS
}

/*-----------------------------------------------------------------*/

pub fn IDASetStopTime(ida_mem: &IDAMem, tstop: sunrealtype) -> i32 {
    /* NULL-mem check: handled by type system */

    /* If IDASolve was called at least once, test if tstop is legal
     * (i.e. if it was not already passed).
     * If IDASetStopTime is called before the first call to IDASolve,
     * tstop will be checked in IDASolve. */
    let (nst, tn, hh) = {
        let IDA_mem = ida_mem.borrow();
        (IDA_mem.ida_nst, IDA_mem.ida_tn, IDA_mem.ida_hh)
    };
    if nst > 0 {
        if (tstop - tn) * hh < ZERO {
            IDAProcessError(
                Some(ida_mem),
                IDA_ILL_INPUT,
                line!() as i32,
                "IDASetStopTime",
                file!(),
                &MSG_BAD_TSTOP(tstop, tn),
            );
            return IDA_ILL_INPUT;
        }
    }

    let mut IDA_mem = ida_mem.borrow_mut();
    IDA_mem.ida_tstop = tstop;
    IDA_mem.ida_tstopset = SUNTRUE;

    IDA_SUCCESS
}

/*-----------------------------------------------------------------*/

pub fn IDAClearStopTime(ida_mem: &IDAMem) -> i32 {
    /* NULL-mem check: handled by type system */
    ida_mem.borrow_mut().ida_tstopset = SUNFALSE;

    IDA_SUCCESS
}

/*-----------------------------------------------------------------*/

pub fn IDASetNonlinConvCoef(ida_mem: &IDAMem, epcon: sunrealtype) -> i32 {
    /* NULL-mem check: handled by type system */
    if epcon <= ZERO {
        IDAProcessError(
            Some(ida_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDASetNonlinConvCoef",
            file!(),
            MSG_NEG_EPCON,
        );
        return IDA_ILL_INPUT;
    }

    ida_mem.borrow_mut().ida_epcon = epcon;

    IDA_SUCCESS
}

/*-----------------------------------------------------------------*/

pub fn IDASetMaxErrTestFails(ida_mem: &IDAMem, maxnef: i32) -> i32 {
    /* NULL-mem check: handled by type system */
    ida_mem.borrow_mut().ida_maxnef = maxnef;

    IDA_SUCCESS
}

/*-----------------------------------------------------------------*/

pub fn IDASetMaxConvFails(ida_mem: &IDAMem, maxncf: i32) -> i32 {
    /* NULL-mem check: handled by type system */
    ida_mem.borrow_mut().ida_maxncf = maxncf;

    IDA_SUCCESS
}

/*-----------------------------------------------------------------*/

pub fn IDASetMaxNonlinIters(ida_mem: &IDAMem, maxcor: i32) -> i32 {
    /* NULL-mem check: handled by type system */

    /* check that the NLS is non-NULL */
    let NLS = ida_mem.borrow().NLS.clone();

    let NLS = match NLS {
        Some(NLS) => NLS,
        None => {
            IDAProcessError(
                None,
                IDA_MEM_FAIL,
                line!() as i32,
                "IDASetMaxNonlinIters",
                file!(),
                MSG_MEM_FAIL,
            );
            return IDA_MEM_FAIL;
        }
    };

    SUNNonlinSolSetMaxIters(&NLS, maxcor)
}

/*-----------------------------------------------------------------*/

pub fn IDASetSuppressAlg(ida_mem: &IDAMem, suppressalg: sunbooleantype) -> i32 {
    /* NULL-mem check: handled by type system */
    ida_mem.borrow_mut().ida_suppressalg = suppressalg;

    IDA_SUCCESS
}

/*-----------------------------------------------------------------*/

pub fn IDASetId(ida_mem: &IDAMem, id: Option<&N_Vector>) -> i32 {
    /* NULL-mem check: handled by type system */

    let id = match id {
        None => {
            if ida_mem.borrow().ida_idMallocDone {
                /* C destroys the stored vector but leaves the pointer
                dangling (upstream bug: any later use is UB); the safe port
                takes the handle out, so later checks see "no id".
                Accepted deviation C in lib.rs. */
                let taken = ida_mem.borrow_mut().ida_id.take();
                if let Some(v) = taken {
                    N_VDestroy(v);
                }
                let mut IDA_mem = ida_mem.borrow_mut();
                let lrw1 = IDA_mem.ida_lrw1;
                let liw1 = IDA_mem.ida_liw1;
                IDA_mem.ida_lrw -= lrw1;
                IDA_mem.ida_liw -= liw1;
            }
            ida_mem.borrow_mut().ida_idMallocDone = SUNFALSE;
            return IDA_SUCCESS;
        }
        Some(id) => id,
    };

    if !ida_mem.borrow().ida_idMallocDone {
        /* C does not check the clone for NULL (a NULL return would be
        dereferenced by the N_VScale below: UB) -> deterministic panic. */
        let cloned = N_VClone(id).expect("N_VClone(id)");
        let mut IDA_mem = ida_mem.borrow_mut();
        IDA_mem.ida_id = Some(cloned);
        let lrw1 = IDA_mem.ida_lrw1;
        let liw1 = IDA_mem.ida_liw1;
        IDA_mem.ida_lrw += lrw1;
        IDA_mem.ida_liw += liw1;
        IDA_mem.ida_idMallocDone = SUNTRUE;
    }

    /* Load the id vector */

    let target = ida_mem.borrow().ida_id.as_ref().expect("ida_id").clone();
    N_VScale(ONE, id, &target);

    IDA_SUCCESS
}

/*-----------------------------------------------------------------*/

pub fn IDASetConstraints(ida_mem: &IDAMem, constraints: Option<&N_Vector>) -> i32 {
    /* NULL-mem check: handled by type system */

    let constraints = match constraints {
        None => {
            /* C destroys the stored vector but leaves the pointer dangling
            (upstream bug: any later use is UB); the safe port takes the
            handle out, so later checks see "no constraints".
            Accepted deviation C in lib.rs. */
            let taken = ida_mem.borrow_mut().ida_constraints.take();
            if let Some(c) = taken {
                N_VDestroy(c);
                let mut IDA_mem = ida_mem.borrow_mut();
                let lrw1 = IDA_mem.ida_lrw1;
                let liw1 = IDA_mem.ida_liw1;
                IDA_mem.ida_lrw -= lrw1;
                IDA_mem.ida_liw -= liw1;
            }
            return IDA_SUCCESS;
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
            IDAProcessError(
                Some(ida_mem),
                IDA_ILL_INPUT,
                line!() as i32,
                "IDASetConstraints",
                file!(),
                MSG_BAD_NVECTOR,
            );
            return IDA_ILL_INPUT;
        }
    }

    /*  Check the constraints vector */

    let temptest = N_VMaxNorm(constraints);
    if (temptest > TWOPT5) || (temptest < HALF) {
        IDAProcessError(
            Some(ida_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDASetConstraints",
            file!(),
            MSG_BAD_CONSTR,
        );
        return IDA_ILL_INPUT;
    }

    if ida_mem.borrow().ida_constraints.is_none() {
        let cloned = match N_VClone(constraints) {
            Some(c) => c,
            None => {
                IDAProcessError(
                    Some(ida_mem),
                    IDA_MEM_NULL,
                    line!() as i32,
                    "IDASetConstraints",
                    file!(),
                    MSG_MEM_FAIL,
                );
                return IDA_MEM_NULL;
            }
        };
        let mut IDA_mem = ida_mem.borrow_mut();
        IDA_mem.ida_constraints = Some(cloned);
        let lrw1 = IDA_mem.ida_lrw1;
        let liw1 = IDA_mem.ida_liw1;
        IDA_mem.ida_lrw += lrw1;
        IDA_mem.ida_liw += liw1;
    }

    /* Load the constraints vector */

    let target = ida_mem
        .borrow()
        .ida_constraints
        .as_ref()
        .expect("ida_constraints")
        .clone();
    N_VScale(ONE, constraints, &target);

    IDA_SUCCESS
}

/*
 * IDASetMaxNumConstraintFails
 *
 * Set the maximum number of constraint failure allowed in a step
 */

pub fn IDASetMaxNumConstraintFails(ida_mem: &IDAMem, max_fails: i32) -> i32 {
    /* NULL-mem check: handled by type system */
    let mut IDA_mem = ida_mem.borrow_mut();

    if max_fails <= 0 {
        IDA_mem.max_constraint_fails = MAX_CONSTRAINT_FAILS;
    } else {
        IDA_mem.max_constraint_fails = max_fails;
    }

    IDA_SUCCESS
}

/*
 * IDAGetNumConstraintFails
 *
 * Get the number of failed steps due to constraint violation
 */

pub fn IDAGetNumConstraintFails(ida_mem: &IDAMem, num_fails_out: &mut i64) -> i32 {
    /* NULL-mem check: handled by type system */
    *num_fails_out = ida_mem.borrow().constraint_fails;

    IDA_SUCCESS
}

/*
 * IDAGetNumConstraintCorrections
 *
 * Get the number of constraint corrections
 */

pub fn IDAGetNumConstraintCorrections(ida_mem: &IDAMem, num_corrections_out: &mut i64) -> i32 {
    /* NULL-mem check: handled by type system */
    *num_corrections_out = ida_mem.borrow().constraint_corrections;

    IDA_SUCCESS
}

/*
 * IDASetRootDirection
 *
 * Specifies the direction of zero-crossings to be monitored.
 * The default is to monitor both crossings.
 */

pub fn IDASetRootDirection(ida_mem: &IDAMem, rootdir: &[i32]) -> i32 {
    /* NULL-mem check: handled by type system */

    let nrt = ida_mem.borrow().ida_nrtfn;
    if nrt == 0 {
        /* C passes NULL (not IDA_mem) here — preserved. */
        IDAProcessError(
            None,
            IDA_ILL_INPUT,
            line!() as i32,
            "IDASetRootDirection",
            file!(),
            MSG_NO_ROOT,
        );
        return IDA_ILL_INPUT;
    }

    let mut IDA_mem = ida_mem.borrow_mut();
    for i in 0..nrt {
        IDA_mem.ida_rootdir[i as usize] = rootdir[i as usize];
    }

    IDA_SUCCESS
}

/*
 * IDASetNoInactiveRootWarn
 *
 * Disables issuing a warning if some root function appears
 * to be identically zero at the beginning of the integration
 */

pub fn IDASetNoInactiveRootWarn(ida_mem: &IDAMem) -> i32 {
    /* NULL-mem check: handled by type system */
    ida_mem.borrow_mut().ida_mxgnull = 0;

    IDA_SUCCESS
}

/*
 * =================================================================
 * IDA IC optional input functions
 * =================================================================
 */

pub fn IDASetNonlinConvCoefIC(ida_mem: &IDAMem, epiccon: sunrealtype) -> i32 {
    /* NULL-mem check: handled by type system */
    if epiccon <= ZERO {
        IDAProcessError(
            Some(ida_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDASetNonlinConvCoefIC",
            file!(),
            MSG_BAD_EPICCON,
        );
        return IDA_ILL_INPUT;
    }

    ida_mem.borrow_mut().ida_epiccon = epiccon;

    IDA_SUCCESS
}

/*-----------------------------------------------------------------*/

pub fn IDASetMaxNumStepsIC(ida_mem: &IDAMem, maxnh: i32) -> i32 {
    /* NULL-mem check: handled by type system */
    if maxnh <= 0 {
        IDAProcessError(
            Some(ida_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDASetMaxNumStepsIC",
            file!(),
            MSG_BAD_MAXNH,
        );
        return IDA_ILL_INPUT;
    }

    ida_mem.borrow_mut().ida_maxnh = maxnh;

    IDA_SUCCESS
}

/*-----------------------------------------------------------------*/

pub fn IDASetMaxNumJacsIC(ida_mem: &IDAMem, maxnj: i32) -> i32 {
    /* NULL-mem check: handled by type system */
    if maxnj <= 0 {
        IDAProcessError(
            Some(ida_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDASetMaxNumJacsIC",
            file!(),
            MSG_BAD_MAXNJ,
        );
        return IDA_ILL_INPUT;
    }

    ida_mem.borrow_mut().ida_maxnj = maxnj;

    IDA_SUCCESS
}

/*-----------------------------------------------------------------*/

pub fn IDASetMaxNumItersIC(ida_mem: &IDAMem, maxnit: i32) -> i32 {
    /* NULL-mem check: handled by type system */
    if maxnit <= 0 {
        IDAProcessError(
            Some(ida_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDASetMaxNumItersIC",
            file!(),
            MSG_BAD_MAXNIT,
        );
        return IDA_ILL_INPUT;
    }

    ida_mem.borrow_mut().ida_maxnit = maxnit;

    IDA_SUCCESS
}

/*-----------------------------------------------------------------*/

pub fn IDASetMaxBacksIC(ida_mem: &IDAMem, maxbacks: i32) -> i32 {
    /* NULL-mem check: handled by type system */
    if maxbacks <= 0 {
        IDAProcessError(
            Some(ida_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDASetMaxBacksIC",
            file!(),
            MSG_IC_BAD_MAXBACKS,
        );
        return IDA_ILL_INPUT;
    }

    ida_mem.borrow_mut().ida_maxbacks = maxbacks;

    IDA_SUCCESS
}

/*-----------------------------------------------------------------*/

pub fn IDASetLineSearchOffIC(ida_mem: &IDAMem, lsoff: sunbooleantype) -> i32 {
    /* NULL-mem check: handled by type system */
    ida_mem.borrow_mut().ida_lsoff = lsoff;

    IDA_SUCCESS
}

/*-----------------------------------------------------------------*/

pub fn IDASetStepToleranceIC(ida_mem: &IDAMem, steptol: sunrealtype) -> i32 {
    /* NULL-mem check: handled by type system */
    if steptol <= ZERO {
        IDAProcessError(
            Some(ida_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDASetStepToleranceIC",
            file!(),
            MSG_BAD_STEPTOL,
        );
        return IDA_ILL_INPUT;
    }

    ida_mem.borrow_mut().ida_steptol = steptol;

    IDA_SUCCESS
}

/*
 * =================================================================
 * IDA optional output functions
 * =================================================================
 */

pub fn IDAGetNumSteps(ida_mem: &IDAMem, nsteps: &mut i64) -> i32 {
    /* NULL-mem check: handled by type system */
    *nsteps = ida_mem.borrow().ida_nst;

    IDA_SUCCESS
}

/*-----------------------------------------------------------------*/

pub fn IDAGetNumResEvals(ida_mem: &IDAMem, nrevals: &mut i64) -> i32 {
    /* NULL-mem check: handled by type system */
    *nrevals = ida_mem.borrow().ida_nre;

    IDA_SUCCESS
}

/*-----------------------------------------------------------------*/

pub fn IDAGetNumLinSolvSetups(ida_mem: &IDAMem, nlinsetups: &mut i64) -> i32 {
    /* NULL-mem check: handled by type system */
    *nlinsetups = ida_mem.borrow().ida_nsetups;

    IDA_SUCCESS
}

/*-----------------------------------------------------------------*/

pub fn IDAGetNumErrTestFails(ida_mem: &IDAMem, netfails: &mut i64) -> i32 {
    /* NULL-mem check: handled by type system */
    *netfails = ida_mem.borrow().ida_netf;

    IDA_SUCCESS
}

/*-----------------------------------------------------------------*/

pub fn IDAGetNumBacktrackOps(ida_mem: &IDAMem, nbacktracks: &mut i64) -> i32 {
    /* NULL-mem check: handled by type system */
    *nbacktracks = ida_mem.borrow().ida_nbacktr as i64;

    IDA_SUCCESS
}

/*-----------------------------------------------------------------*/

pub fn IDAGetConsistentIC(ida_mem: &IDAMem, yy0: Option<&N_Vector>, yp0: Option<&N_Vector>) -> i32 {
    /* NULL-mem check: handled by type system */
    let kused = ida_mem.borrow().ida_kused;

    if kused != 0 {
        IDAProcessError(
            Some(ida_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDAGetConsistentIC",
            file!(),
            MSG_TOO_LATE,
        );
        return IDA_ILL_INPUT;
    }

    if let Some(yy0) = yy0 {
        let phi0 = ida_mem.borrow().ida_phi[0]
            .as_ref()
            .expect("ida_phi[0]")
            .clone();
        N_VScale(ONE, &phi0, yy0);
    }
    if let Some(yp0) = yp0 {
        let phi1 = ida_mem.borrow().ida_phi[1]
            .as_ref()
            .expect("ida_phi[1]")
            .clone();
        N_VScale(ONE, &phi1, yp0);
    }

    IDA_SUCCESS
}

/*-----------------------------------------------------------------*/

pub fn IDAGetLastOrder(ida_mem: &IDAMem, klast: &mut i32) -> i32 {
    /* NULL-mem check: handled by type system */
    *klast = ida_mem.borrow().ida_kused;

    IDA_SUCCESS
}

/*-----------------------------------------------------------------*/

pub fn IDAGetCurrentOrder(ida_mem: &IDAMem, kcur: &mut i32) -> i32 {
    /* NULL-mem check: handled by type system */
    *kcur = ida_mem.borrow().ida_kk;

    IDA_SUCCESS
}

/*-----------------------------------------------------------------*/

pub fn IDAGetCurrentCj(ida_mem: &IDAMem, cj: &mut sunrealtype) -> i32 {
    /* NULL-mem check: handled by type system */
    *cj = ida_mem.borrow().ida_cj;

    IDA_SUCCESS
}

/*-----------------------------------------------------------------*/

pub fn IDAGetCurrentY(ida_mem: &IDAMem, ycur: &mut Option<N_Vector>) -> i32 {
    /* NULL-mem check: handled by type system */
    /* handle clone out = C pointer copy */
    *ycur = ida_mem.borrow().ida_yy.clone();

    IDA_SUCCESS
}

/*-----------------------------------------------------------------*/

pub fn IDAGetCurrentYp(ida_mem: &IDAMem, ypcur: &mut Option<N_Vector>) -> i32 {
    /* NULL-mem check: handled by type system */
    /* handle clone out = C pointer copy */
    *ypcur = ida_mem.borrow().ida_yp.clone();

    IDA_SUCCESS
}

/*-----------------------------------------------------------------*/

pub fn IDAGetActualInitStep(ida_mem: &IDAMem, hinused: &mut sunrealtype) -> i32 {
    /* NULL-mem check: handled by type system */
    *hinused = ida_mem.borrow().ida_h0u;

    IDA_SUCCESS
}

/*-----------------------------------------------------------------*/

pub fn IDAGetLastStep(ida_mem: &IDAMem, hlast: &mut sunrealtype) -> i32 {
    /* NULL-mem check: handled by type system */
    *hlast = ida_mem.borrow().ida_hused;

    IDA_SUCCESS
}

/*-----------------------------------------------------------------*/

pub fn IDAGetCurrentStep(ida_mem: &IDAMem, hcur: &mut sunrealtype) -> i32 {
    /* NULL-mem check: handled by type system */
    *hcur = ida_mem.borrow().ida_hh;

    IDA_SUCCESS
}

/*-----------------------------------------------------------------*/

pub fn IDAGetCurrentTime(ida_mem: &IDAMem, tcur: &mut sunrealtype) -> i32 {
    /* NULL-mem check: handled by type system */
    *tcur = ida_mem.borrow().ida_tn;

    IDA_SUCCESS
}

/*-----------------------------------------------------------------*/

pub fn IDAGetTolScaleFactor(ida_mem: &IDAMem, tolsfact: &mut sunrealtype) -> i32 {
    /* NULL-mem check: handled by type system */
    *tolsfact = ida_mem.borrow().ida_tolsf;

    IDA_SUCCESS
}

/*-----------------------------------------------------------------*/

pub fn IDAGetErrWeights(ida_mem: &IDAMem, eweight: &N_Vector) -> i32 {
    /* NULL-mem check: handled by type system */
    let ewt = ida_mem.borrow().ida_ewt.as_ref().expect("ida_ewt").clone();

    N_VScale(ONE, &ewt, eweight);

    IDA_SUCCESS
}

/*-----------------------------------------------------------------*/

pub fn IDAGetEstLocalErrors(ida_mem: &IDAMem, ele: &N_Vector) -> i32 {
    /* NULL-mem check: handled by type system */
    let ee = ida_mem.borrow().ida_ee.as_ref().expect("ida_ee").clone();

    N_VScale(ONE, &ee, ele);

    IDA_SUCCESS
}

/*-----------------------------------------------------------------*/

pub fn IDAGetWorkSpace(ida_mem: &IDAMem, lenrw: &mut i64, leniw: &mut i64) -> i32 {
    /* NULL-mem check: handled by type system */
    let IDA_mem = ida_mem.borrow();

    *leniw = IDA_mem.ida_liw;
    *lenrw = IDA_mem.ida_lrw;

    IDA_SUCCESS
}

/*-----------------------------------------------------------------*/

#[allow(clippy::too_many_arguments)]
pub fn IDAGetIntegratorStats(
    ida_mem: &IDAMem,
    nsteps: &mut i64,
    nrevals: &mut i64,
    nlinsetups: &mut i64,
    netfails: &mut i64,
    klast: &mut i32,
    kcur: &mut i32,
    hinused: &mut sunrealtype,
    hlast: &mut sunrealtype,
    hcur: &mut sunrealtype,
    tcur: &mut sunrealtype,
) -> i32 {
    /* NULL-mem check: handled by type system */
    let IDA_mem = ida_mem.borrow();

    *nsteps = IDA_mem.ida_nst;
    *nrevals = IDA_mem.ida_nre;
    *nlinsetups = IDA_mem.ida_nsetups;
    *netfails = IDA_mem.ida_netf;
    *klast = IDA_mem.ida_kused;
    *kcur = IDA_mem.ida_kk;
    *hinused = IDA_mem.ida_h0u;
    *hlast = IDA_mem.ida_hused;
    *hcur = IDA_mem.ida_hh;
    *tcur = IDA_mem.ida_tn;

    IDA_SUCCESS
}

/*-----------------------------------------------------------------*/

pub fn IDAGetNumGEvals(ida_mem: &IDAMem, ngevals: &mut i64) -> i32 {
    /* NULL-mem check: handled by type system */
    *ngevals = ida_mem.borrow().ida_nge;

    IDA_SUCCESS
}

/*-----------------------------------------------------------------*/

pub fn IDAGetRootInfo(ida_mem: &IDAMem, rootsfound: &mut [i32]) -> i32 {
    /* NULL-mem check: handled by type system */
    let IDA_mem = ida_mem.borrow();

    let nrt = IDA_mem.ida_nrtfn;

    for i in 0..nrt {
        rootsfound[i as usize] = IDA_mem.ida_iroots[i as usize];
    }

    IDA_SUCCESS
}

/*-----------------------------------------------------------------*/

pub fn IDAGetNumNonlinSolvIters(ida_mem: &IDAMem, nniters: &mut i64) -> i32 {
    /* NULL-mem check: handled by type system */
    *nniters = ida_mem.borrow().ida_nni;

    IDA_SUCCESS
}

/*-----------------------------------------------------------------*/

pub fn IDAGetNumNonlinSolvConvFails(ida_mem: &IDAMem, nnfails: &mut i64) -> i32 {
    /* NULL-mem check: handled by type system */
    *nnfails = ida_mem.borrow().ida_nnf;

    IDA_SUCCESS
}

/*-----------------------------------------------------------------*/

pub fn IDAGetNonlinSolvStats(ida_mem: &IDAMem, nniters: &mut i64, nnfails: &mut i64) -> i32 {
    /* NULL-mem check: handled by type system */
    let IDA_mem = ida_mem.borrow();

    *nniters = IDA_mem.ida_nni;
    *nnfails = IDA_mem.ida_nnf;

    IDA_SUCCESS
}

/*-----------------------------------------------------------------*/

pub fn IDAGetNumStepSolveFails(ida_mem: &IDAMem, nncfails: &mut i64) -> i32 {
    /* NULL-mem check: handled by type system */
    *nncfails = ida_mem.borrow().ida_ncfn;

    IDA_SUCCESS
}

/*-----------------------------------------------------------------*/

/// C `IDAGetUserData` returns the stored `void*` without ownership
/// transfer. The safe-Rust token cannot be aliased, so the stored box is
/// SWAPPED with `user_data`; the caller must hand it back (via
/// `IDASetUserData` or a second swap) before the integrator next invokes a
/// user callback.
pub fn IDAGetUserData(ida_mem: &IDAMem, user_data: &mut Option<Box<dyn Any>>) -> i32 {
    /* NULL-mem check: handled by type system */
    std::mem::swap(&mut ida_mem.borrow_mut().ida_user_data, user_data);

    IDA_SUCCESS
}

/*-----------------------------------------------------------------*/

pub fn IDAPrintAllStats(ida_mem: &IDAMem, outfile: &SUNFile, fmt: SUNOutputFormat) -> i32 {
    /* NULL-mem check: handled by type system */
    if fmt != SUNOutputFormat::SUN_OUTPUTFORMAT_TABLE
        && fmt != SUNOutputFormat::SUN_OUTPUTFORMAT_CSV
    {
        IDAProcessError(
            Some(ida_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDAPrintAllStats",
            file!(),
            "Invalid formatting option.",
        );
        return IDA_ILL_INPUT;
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
        hused,
        hh,
        kused,
        kk,
        nre,
        nbacktr,
        nni,
        nnf,
        nsetups,
        nge,
    );
    let ls_stats: Option<(i64, i64, i64, i64, i64, i64, i64, i64)>;
    {
        let IDA_mem = ida_mem.borrow();
        tn = IDA_mem.ida_tn;
        nst = IDA_mem.ida_nst;
        netf = IDA_mem.ida_netf;
        ncfn = IDA_mem.ida_ncfn;
        constraint_fails = IDA_mem.constraint_fails;
        constraint_corrections = IDA_mem.constraint_corrections;
        h0u = IDA_mem.ida_h0u;
        hused = IDA_mem.ida_hused;
        hh = IDA_mem.ida_hh;
        kused = IDA_mem.ida_kused;
        kk = IDA_mem.ida_kk;
        nre = IDA_mem.ida_nre;
        nbacktr = IDA_mem.ida_nbacktr;
        nni = IDA_mem.ida_nni;
        nnf = IDA_mem.ida_nnf;
        nsetups = IDA_mem.ida_nsetups;
        nge = IDA_mem.ida_nge;

        ls_stats = IDA_mem.ida_lmem.as_ref().map(|lmem| {
            let idals_mem = lmem
                .downcast_ref::<IDALsMemRec>()
                .expect("ida_lmem holds IDALsMemRec");
            (
                idals_mem.nje,
                idals_mem.nreDQ,
                idals_mem.npe,
                idals_mem.nps,
                idals_mem.nli,
                idals_mem.ncfl,
                idals_mem.njtsetup,
                idals_mem.njtimes,
            )
        });
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
    sunfprintf_real(outfile, fmt, SUNFALSE, "Last step size", hused);
    sunfprintf_real(outfile, fmt, SUNFALSE, "Current step size", hh);
    sunfprintf_long(outfile, fmt, SUNFALSE, "Last method order", kused as i64);
    sunfprintf_long(outfile, fmt, SUNFALSE, "Current method order", kk as i64);

    /* function evaluations */
    sunfprintf_long(outfile, fmt, SUNFALSE, "Residual fn evals", nre);

    /* IC calculation stats */
    sunfprintf_long(
        outfile,
        fmt,
        SUNFALSE,
        "IC linesearch backtrack ops",
        nbacktr as i64,
    );

    /* nonlinear solver stats */
    sunfprintf_long(outfile, fmt, SUNFALSE, "NLS iters", nni);
    sunfprintf_long(outfile, fmt, SUNFALSE, "NLS fails", nnf);
    if nst > 0 {
        /* upstream uses ida_nre (not ida_nni) for this ratio — preserved */
        sunfprintf_real(
            outfile,
            fmt,
            SUNFALSE,
            "NLS iters per step",
            nre as sunrealtype / nst as sunrealtype,
        );
    }

    /* linear solver stats */
    sunfprintf_long(outfile, fmt, SUNFALSE, "LS setups", nsetups);
    if let Some((nje, nreDQ, npe, nps, nli, ncfl, njtsetup, njtimes)) = ls_stats {
        sunfprintf_long(outfile, fmt, SUNFALSE, "Jac fn evals", nje);
        sunfprintf_long(outfile, fmt, SUNFALSE, "LS residual fn evals", nreDQ);
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

    IDA_SUCCESS
}

/*-----------------------------------------------------------------*/

/// C returns a `malloc`'d `char*`; the Rust port returns `String`.
pub fn IDAGetReturnFlagName(flag: i64) -> String {
    let name = match flag {
        f if f == IDA_SUCCESS as i64 => "IDA_SUCCESS",
        f if f == IDA_TSTOP_RETURN as i64 => "IDA_TSTOP_RETURN",
        f if f == IDA_ROOT_RETURN as i64 => "IDA_ROOT_RETURN",
        f if f == IDA_TOO_MUCH_WORK as i64 => "IDA_TOO_MUCH_WORK",
        f if f == IDA_TOO_MUCH_ACC as i64 => "IDA_TOO_MUCH_ACC",
        f if f == IDA_ERR_FAIL as i64 => "IDA_ERR_FAIL",
        f if f == IDA_CONV_FAIL as i64 => "IDA_CONV_FAIL",
        f if f == IDA_LINIT_FAIL as i64 => "IDA_LINIT_FAIL",
        f if f == IDA_LSETUP_FAIL as i64 => "IDA_LSETUP_FAIL",
        f if f == IDA_LSOLVE_FAIL as i64 => "IDA_LSOLVE_FAIL",
        f if f == IDA_CONSTR_FAIL as i64 => "IDA_CONSTR_FAIL",
        f if f == IDA_RES_FAIL as i64 => "IDA_RES_FAIL",
        f if f == IDA_FIRST_RES_FAIL as i64 => "IDA_FIRST_RES_FAIL",
        f if f == IDA_REP_RES_ERR as i64 => "IDA_REP_RES_ERR",
        f if f == IDA_RTFUNC_FAIL as i64 => "IDA_RTFUNC_FAIL",
        f if f == IDA_MEM_FAIL as i64 => "IDA_MEM_FAIL",
        f if f == IDA_MEM_NULL as i64 => "IDA_MEM_NULL",
        f if f == IDA_ILL_INPUT as i64 => "IDA_ILL_INPUT",
        f if f == IDA_NO_MALLOC as i64 => "IDA_NO_MALLOC",
        f if f == IDA_BAD_T as i64 => "IDA_BAD_T",
        f if f == IDA_BAD_K as i64 => "IDA_BAD_K",
        f if f == IDA_BAD_DKY as i64 => "IDA_BAD_DKY",
        f if f == IDA_BAD_EWT as i64 => "IDA_BAD_EWT",
        f if f == IDA_NO_RECOVERY as i64 => "IDA_NO_RECOVERY",
        f if f == IDA_LINESEARCH_FAIL as i64 => "IDA_LINESEARCH_FAIL",
        f if f == IDA_NLS_SETUP_FAIL as i64 => "IDA_NLS_SETUP_FAIL",
        f if f == IDA_NLS_FAIL as i64 => "IDA_NLS_FAIL",
        _ => "NONE",
    };

    name.to_string()
}
