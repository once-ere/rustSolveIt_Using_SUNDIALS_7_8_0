//! Port of `src/arkode/arkode_sunstepper.c`: ARKODE's interfacing with the
//! generic `SUNStepper` base class (an ARKODE integrator wrapped as a
//! `SUNStepper`).
//!
//! Binding notes:
//! * The C `void* content` of the wrapping `SUNStepper` IS the `ARKodeMem`
//!   handle, so every `SUNStepper_GetContent(stepper, &arkode_mem)` maps to
//!   `SUNStepper_GetContentAs::<ARKodeMem>` -- which clones the `Rc` (exactly
//!   C's pointer copy) and leaves the stepper's content in place. The
//!   swap-based `SUNStepper_GetContent` is deliberately not used here.
//! * `stepper->last_flag = …` writes the public `last_flag` cell; the flag is
//!   also kept in a local so the following test reads the same value C does.
//! * `SUNFunctionBegin`, `SUNCheckCall` and `SUNAssert` are no-ops in this
//!   build configuration: call sites evaluate the call and continue.
//! * The C `default: ark_mode = -1` arm of the `SUNFullRhsMode` switch is
//!   unreachable -- the Rust enum has exactly the three variants -- so the
//!   `match` is exhaustive without it.
//! * `arkSUNStepperSelfDestruct` calls `ARKodeFree` on the cloned handle: the
//!   integrator's resources are released exactly where C releases them, and
//!   the (now empty) `ARKodeMemRec` shell goes away with the stepper's own
//!   content when the last handle drops. C instead leaves `stepper->content`
//!   dangling at that point.

use crate::arkode::{ARKodeEvolve, ARKodeFree, ARKodeReset};
use crate::arkode_impl::*;
use crate::arkode_io::{
    ARKodeSetAdjointCheckpointIndex, ARKodeSetStepDirection, ARKodeSetStopTime,
};
use sundials_core::sundials_errors::{SUN_ERR_OP_FAIL, SUN_SUCCESS};
use sundials_core::sundials_nvector::N_Vector;
use sundials_core::sundials_stepper::*;
use sundials_core::sundials_types::*;

fn arkSUNStepperEvolveHelper(
    stepper: &SUNStepper,
    tout: sunrealtype,
    y: &N_Vector,
    tret: &mut sunrealtype,
    mode: i32,
) -> SUNErrCode {
    /* extract the ARKODE memory struct */
    let mut arkode_mem: Option<ARKodeMem> = None;
    let _ = SUNStepper_GetContentAs(stepper, &mut arkode_mem);
    let arkode_mem = arkode_mem.expect("ARKodeMem SUNStepper content");

    /* evolve inner ODE */
    let last_flag = ARKodeEvolve(&arkode_mem, tout, y, tret, mode);
    *stepper.last_flag.borrow_mut() = last_flag;
    if last_flag < 0 {
        return SUN_ERR_OP_FAIL;
    }

    SUN_SUCCESS
}

fn arkSUNStepperEvolve(
    stepper: &SUNStepper,
    tout: sunrealtype,
    y: &N_Vector,
    tret: &mut sunrealtype,
) -> SUNErrCode {
    arkSUNStepperEvolveHelper(stepper, tout, y, tret, ARK_NORMAL)
}

fn arkSUNStepperOneStep(
    stepper: &SUNStepper,
    tout: sunrealtype,
    y: &N_Vector,
    tret: &mut sunrealtype,
) -> SUNErrCode {
    arkSUNStepperEvolveHelper(stepper, tout, y, tret, ARK_ONE_STEP)
}

/*------------------------------------------------------------------------------
  Implementation of SUNStepperFullRhsFn to compute the full inner
  (fast) ODE IVP RHS.
  ----------------------------------------------------------------------------*/

fn arkSUNStepperFullRhs(
    stepper: &SUNStepper,
    t: sunrealtype,
    y: &N_Vector,
    f: &N_Vector,
    mode: SUNFullRhsMode,
) -> SUNErrCode {
    /* extract the ARKODE memory struct */
    let mut arkode_mem: Option<ARKodeMem> = None;
    let _ = SUNStepper_GetContentAs(stepper, &mut arkode_mem);
    let ark_mem = arkode_mem.expect("ARKodeMem SUNStepper content");

    let ark_mode = match mode {
        SUN_FULLRHS_START => ARK_FULLRHS_START,
        SUN_FULLRHS_END => ARK_FULLRHS_END,
        SUN_FULLRHS_OTHER => ARK_FULLRHS_OTHER,
    };

    let step_fullrhs = ark_mem.borrow().step_fullrhs.expect("step_fullrhs");
    let last_flag = step_fullrhs(&ark_mem, t, y, f, ark_mode);
    *stepper.last_flag.borrow_mut() = last_flag;
    if last_flag != ARK_SUCCESS {
        return SUN_ERR_OP_FAIL;
    }

    SUN_SUCCESS
}

/*------------------------------------------------------------------------------
  Implementation of SUNStepperResetFn to reset the stepper state.
  ----------------------------------------------------------------------------*/

fn arkSUNStepperReset(stepper: &SUNStepper, tR: sunrealtype, yR: &N_Vector) -> SUNErrCode {
    /* extract the ARKODE memory struct */
    let mut arkode_mem: Option<ARKodeMem> = None;
    let _ = SUNStepper_GetContentAs(stepper, &mut arkode_mem);
    let arkode_mem = arkode_mem.expect("ARKodeMem SUNStepper content");

    let last_flag = ARKodeReset(&arkode_mem, tR, yR);
    *stepper.last_flag.borrow_mut() = last_flag;
    if last_flag != ARK_SUCCESS {
        return SUN_ERR_OP_FAIL;
    }

    SUN_SUCCESS
}

/*------------------------------------------------------------------------------
  Implementation of SUNStepperResetCheckpointIndexFn.
  ----------------------------------------------------------------------------*/

fn arkSUNStepperResetCheckpointIndex(
    stepper: &SUNStepper,
    ckptIdxR: suncountertype,
) -> SUNErrCode {
    /* extract the ARKODE memory struct */
    let mut arkode_mem: Option<ARKodeMem> = None;
    let _ = SUNStepper_GetContentAs(stepper, &mut arkode_mem);
    let arkode_mem = arkode_mem.expect("ARKodeMem SUNStepper content");

    let last_flag = ARKodeSetAdjointCheckpointIndex(&arkode_mem, ckptIdxR);
    *stepper.last_flag.borrow_mut() = last_flag;
    if last_flag != ARK_SUCCESS {
        return SUN_ERR_OP_FAIL;
    }

    SUN_SUCCESS
}

/*------------------------------------------------------------------------------
  Implementation of SUNStepperStopTimeFn to set the tstop time
  ----------------------------------------------------------------------------*/

fn arkSUNStepperSetStopTime(stepper: &SUNStepper, tstop: sunrealtype) -> SUNErrCode {
    /* extract the ARKODE memory struct */
    let mut arkode_mem: Option<ARKodeMem> = None;
    let _ = SUNStepper_GetContentAs(stepper, &mut arkode_mem);
    let arkode_mem = arkode_mem.expect("ARKodeMem SUNStepper content");

    let last_flag = ARKodeSetStopTime(&arkode_mem, tstop);
    *stepper.last_flag.borrow_mut() = last_flag;
    if last_flag != ARK_SUCCESS {
        return SUN_ERR_OP_FAIL;
    }

    SUN_SUCCESS
}

fn arkSUNStepperSetStepDirection(stepper: &SUNStepper, stepdir: sunrealtype) -> SUNErrCode {
    /* extract the ARKODE memory struct */
    let mut arkode_mem: Option<ARKodeMem> = None;
    let _ = SUNStepper_GetContentAs(stepper, &mut arkode_mem);
    let arkode_mem = arkode_mem.expect("ARKodeMem SUNStepper content");

    let last_flag = ARKodeSetStepDirection(&arkode_mem, stepdir);
    *stepper.last_flag.borrow_mut() = last_flag;
    if last_flag != ARK_SUCCESS {
        return SUN_ERR_OP_FAIL;
    }

    SUN_SUCCESS
}

fn arkSUNStepperSetForcing(
    stepper: &SUNStepper,
    tshift: sunrealtype,
    tscale: sunrealtype,
    forcing: &[N_Vector],
    nforcing: i32,
) -> SUNErrCode {
    /* extract the ARKODE memory struct */
    let mut arkode_mem: Option<ARKodeMem> = None;
    let _ = SUNStepper_GetContentAs(stepper, &mut arkode_mem);
    let ark_mem = arkode_mem.expect("ARKodeMem SUNStepper content");

    let step_setforcing = ark_mem.borrow().step_setforcing.expect("step_setforcing");
    let last_flag = step_setforcing(&ark_mem, tshift, tscale, forcing, nforcing);
    *stepper.last_flag.borrow_mut() = last_flag;
    if last_flag != ARK_SUCCESS {
        return SUN_ERR_OP_FAIL;
    }

    SUN_SUCCESS
}

pub fn arkSUNStepperSelfDestruct(stepper: &SUNStepper) -> SUNErrCode {
    /* This function is useful when we create a ARKodeMem/SUNStepper internally,
       and want it to be destroyed with the SUNStepper. */
    let mut ark_mem: Option<ARKodeMem> = None;

    let errcode = SUNStepper_GetContentAs(stepper, &mut ark_mem);
    if errcode != 0 {
        return errcode;
    }

    ARKodeFree(&mut ark_mem);

    SUN_SUCCESS
}

fn arkSUNStepperGetNumSteps(stepper: &SUNStepper, nst: &mut suncountertype) -> SUNErrCode {
    let mut ark_mem: Option<ARKodeMem> = None;

    let errcode = SUNStepper_GetContentAs(stepper, &mut ark_mem);
    if errcode != 0 {
        return errcode;
    }
    let ark_mem = ark_mem.expect("ARKodeMem SUNStepper content");

    *nst = ark_mem.borrow().nst;

    SUN_SUCCESS
}

pub fn ARKodeCreateSUNStepper(arkode_mem: &ARKodeMem, stepper: &mut Option<SUNStepper>) -> i32 {
    /* unpack ark_mem: the C `arkode_mem == NULL` branch is unrepresentable */
    let ark_mem = arkode_mem;

    let sunctx = ark_mem.borrow().sunctx.clone();
    let err = SUNStepper_Create(&sunctx, stepper);
    if err != SUN_SUCCESS {
        arkProcessError(
            Some(ark_mem),
            ARK_SUNSTEPPER_ERR,
            line!() as i32,
            "ARKodeCreateSUNStepper",
            file!(),
            "Failed to create SUNStepper",
        );
        return ARK_SUNSTEPPER_ERR;
    }
    /* C `*stepper`; cloning the handle is the C pointer copy */
    let s = stepper.clone().expect("SUNStepper");

    let err = SUNStepper_SetContent(&s, Box::new(arkode_mem.clone()));
    if err != SUN_SUCCESS {
        arkProcessError(
            Some(ark_mem),
            ARK_SUNSTEPPER_ERR,
            line!() as i32,
            "ARKodeCreateSUNStepper",
            file!(),
            "Failed to set SUNStepper content",
        );
        return ARK_SUNSTEPPER_ERR;
    }

    let err = SUNStepper_SetEvolveFn(&s, Some(arkSUNStepperEvolve));
    if err != SUN_SUCCESS {
        arkProcessError(
            Some(ark_mem),
            ARK_SUNSTEPPER_ERR,
            line!() as i32,
            "ARKodeCreateSUNStepper",
            file!(),
            "Failed to set SUNStepper evolve function",
        );
        return ARK_SUNSTEPPER_ERR;
    }

    let err = SUNStepper_SetOneStepFn(&s, Some(arkSUNStepperOneStep));
    if err != SUN_SUCCESS {
        arkProcessError(
            Some(ark_mem),
            ARK_SUNSTEPPER_ERR,
            line!() as i32,
            "ARKodeCreateSUNStepper",
            file!(),
            "Failed to set SUNStepper one step function",
        );
        return ARK_SUNSTEPPER_ERR;
    }

    let err = SUNStepper_SetFullRhsFn(&s, Some(arkSUNStepperFullRhs));
    if err != SUN_SUCCESS {
        arkProcessError(
            Some(ark_mem),
            ARK_SUNSTEPPER_ERR,
            line!() as i32,
            "ARKodeCreateSUNStepper",
            file!(),
            "Failed to set SUNStepper full RHS function",
        );
        return ARK_SUNSTEPPER_ERR;
    }

    let err = SUNStepper_SetResetFn(&s, Some(arkSUNStepperReset));
    if err != SUN_SUCCESS {
        arkProcessError(
            Some(ark_mem),
            ARK_SUNSTEPPER_ERR,
            line!() as i32,
            "ARKodeCreateSUNStepper",
            file!(),
            "Failed to set SUNStepper reset function",
        );
        return ARK_SUNSTEPPER_ERR;
    }

    let err = SUNStepper_SetResetCheckpointIndexFn(&s, Some(arkSUNStepperResetCheckpointIndex));
    if err != SUN_SUCCESS {
        arkProcessError(
            Some(ark_mem),
            ARK_SUNSTEPPER_ERR,
            line!() as i32,
            "ARKodeCreateSUNStepper",
            file!(),
            "Failed to set SUNStepper reset function",
        );
        return ARK_SUNSTEPPER_ERR;
    }

    let err = SUNStepper_SetStopTimeFn(&s, Some(arkSUNStepperSetStopTime));
    if err != SUN_SUCCESS {
        arkProcessError(
            Some(ark_mem),
            ARK_SUNSTEPPER_ERR,
            line!() as i32,
            "ARKodeCreateSUNStepper",
            file!(),
            "Failed to set SUNStepper stop time function",
        );
        return ARK_SUNSTEPPER_ERR;
    }

    let err = SUNStepper_SetStepDirectionFn(&s, Some(arkSUNStepperSetStepDirection));
    if err != SUN_SUCCESS {
        return ARK_SUNSTEPPER_ERR;
    }

    if ark_mem.borrow().step_setforcing.is_some() {
        let err = SUNStepper_SetForcingFn(&s, Some(arkSUNStepperSetForcing));
        if err != SUN_SUCCESS {
            arkProcessError(
                Some(ark_mem),
                ARK_SUNSTEPPER_ERR,
                line!() as i32,
                "ARKodeCreateSUNStepper",
                file!(),
                "Failed to set SUNStepper forcing function",
            );
            return ARK_SUNSTEPPER_ERR;
        }
    }

    let err = SUNStepper_SetGetNumStepsFn(&s, Some(arkSUNStepperGetNumSteps));
    if err != SUN_SUCCESS {
        arkProcessError(
            Some(ark_mem),
            ARK_SUNSTEPPER_ERR,
            line!() as i32,
            "ARKodeCreateSUNStepper",
            file!(),
            "Failed to set SUNStepper get number of steps function",
        );
        return ARK_SUNSTEPPER_ERR;
    }

    ARK_SUCCESS
}
