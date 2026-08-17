//! Port of `src/arkode/arkode_forcingstep.c` (+ `arkode_forcingstep_impl.h`
//! and `include/arkode/arkode_forcingstep.h`).
//!
//! ARKODE's forcing method: partition 1 is evolved on its own over the outer
//! step, its tendency `(ycur - yn)/h` is handed to partition 2 as a constant
//! forcing term, and partition 2 is then evolved over the same interval.
//!
//! Binding notes (all forced by the frozen contract in `arkode_impl.rs`):
//! * `ARKodeForcingStepMemRec` is stored BY VALUE in
//!   `ark_mem.step_mem: Option<Box<dyn Any>>`; `forcingStep_mem_mut` is the
//!   single module-local downcast helper, and its guard is never held across
//!   `arkProcessError`, an `N_Vector` operation, or a `SUNStepper_*` call.
//! * `forcingStep_InitStepMem` keeps C's shape (it takes the record itself,
//!   not `ark_mem`), because C calls it both before the record is attached
//!   (`ForcingStepCreate`) and after (`ForcingStepReInit`).
//! * `SUNStepper_SetForcing(s1, ZERO, ZERO, &ark_mem->tempv1, 1)` passes the
//!   address of a single `N_Vector` field as a one-element array, so the port
//!   passes a one-element slice; the clearing call's `(NULL, 0)` becomes
//!   `(&[], 0)`, exactly as the contract specifies for `step_setforcing`.
//!
//! Logging: `SUNLogInfo`/`SUNLogExtraDebugVec` compile away at
//! `SUNDIALS_LOGGING_LEVEL=2` and are omitted; every `SUNErrCode err` they
//! consumed is still tested for `SUN_SUCCESS`.

use std::cell::RefMut;

use crate::arkode::{arkCreate, arkInit, ARKodeFree};
use crate::arkode_impl::*;
use crate::arkode_io::ARKodeSetInterpolantType;
use sundials_core::sundials_context::SUNContext;
use sundials_core::sundials_errors::SUN_SUCCESS;
use sundials_core::sundials_nvector::{N_VLinearSum, N_Vector};
use sundials_core::sundials_stepper::*;
use sundials_core::sundials_types::*;
use sundials_core::sundials_utils::{sunfprintf_long, SUNFile};

/* arkode_forcingstep_impl.h */
pub const NUM_PARTITIONS: i32 = 2;

/*---------------------------------------------------------------
  Types : struct ARKodeForcingStepMemRec, ARKodeForcingStepMem
  (arkode_forcingstep_impl.h)
  ---------------------------------------------------------------*/
pub struct ARKodeForcingStepMemRec {
    pub stepper: [SUNStepper; NUM_PARTITIONS as usize],
    pub n_stepper_evolves: [i64; NUM_PARTITIONS as usize],
}

/// Downcast helper: view `ark_mem.step_mem` as the ForcingStep memory record.
/// Panics if no step memory is attached or it is not this stepper's record
/// (C would blindly cast the `void*` — UB maps to a panic, deviation
/// class 5). NEVER hold the guard across `arkProcessError`, a callback, an
/// `N_Vector` operation, or a `SUNStepper_*` call.
pub fn forcingStep_mem_mut(ark_mem: &ARKodeMem) -> RefMut<'_, ARKodeForcingStepMemRec> {
    RefMut::map(ark_mem.borrow_mut(), |m| {
        m.step_mem
            .as_mut()
            .expect("step_mem set")
            .downcast_mut::<ARKodeForcingStepMemRec>()
            .expect("ForcingStep step memory")
    })
}

/*------------------------------------------------------------------------------
  Shortcut routine to unpack step_mem structure from ark_mem. If missing it
  returns ARK_MEM_NULL.
  ----------------------------------------------------------------------------*/
fn forcingStep_AccessStepMem(ark_mem: &ARKodeMem, fname: &str) -> i32 {
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            fname,
            file!(),
            "Time step module memory is NULL.",
        );
        return ARK_MEM_NULL;
    }
    ARK_SUCCESS
}

/*------------------------------------------------------------------------------
  Shortcut routine to unpack ark_mem and step_mem structures from void* pointer.
  If either is missing it returns ARK_MEM_NULL.
  ----------------------------------------------------------------------------*/
fn forcingStep_AccessARKODEStepMem(ark_mem: &ARKodeMem, fname: &str) -> i32 {
    /* access ARKodeMem structure: C's `arkode_mem == NULL` branch is
       unrepresentable with `&ARKodeMem` */
    let _ = fname;

    forcingStep_AccessStepMem(ark_mem, "forcingStep_AccessARKODEStepMem")
}

/*------------------------------------------------------------------------------
  This routine is called just prior to performing internal time steps (after
  all user "set" routines have been called) from within arkInitialSetup.
  ----------------------------------------------------------------------------*/
fn forcingStep_Init(ark_mem: &ARKodeMem, init_type: i32) -> i32 {
    let retval = forcingStep_AccessStepMem(ark_mem, "forcingStep_Init");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* assume fixed outer step size */
    let fixedstep = ark_mem.borrow().fixedstep;
    if !fixedstep {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "forcingStep_Init",
            file!(),
            "Adaptive outer time stepping is not currently supported",
        );
        return ARK_ILL_INPUT;
    }

    let interp_type = ark_mem.borrow().interp_type;
    let (stepper0, stepper1) = {
        let step_mem = forcingStep_mem_mut(ark_mem);
        (step_mem.stepper[0].clone(), step_mem.stepper[1].clone())
    };
    if interp_type == ARK_INTERP_HERMITE
        && (stepper0.ops.borrow().fullrhs.is_none() || stepper1.ops.borrow().fullrhs.is_none())
    {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "forcingStep_Init",
            file!(),
            "The SUNSteppers must implement SUNStepper_FullRhs when using Hermite interpolation",
        );
        return ARK_ILL_INPUT;
    }

    /* immediately return if resize or reset */
    if init_type == RESIZE_INIT || init_type == RESET_INIT {
        return ARK_SUCCESS;
    }

    /* On first initialization, make the SUNStepper consistent with the current
     * state in case a user provided a different initial condition for the
     * ForcingStep integrator and SUNStepper. */
    let (tn, yn) = {
        let m = ark_mem.borrow();
        (m.tn, m.yn.clone().expect("yn"))
    };
    let stepper1 = forcingStep_mem_mut(ark_mem).stepper[1].clone();
    let err = SUNStepper_Reset(&stepper1, tn, &yn);
    if err != SUN_SUCCESS {
        arkProcessError(
            Some(ark_mem),
            ARK_SUNSTEPPER_ERR,
            line!() as i32,
            "forcingStep_Init",
            file!(),
            "Resetting the second partition SUNStepper failed",
        );
        return ARK_SUNSTEPPER_ERR;
    }

    ark_mem.borrow_mut().interp_degree = 1;

    ARK_SUCCESS
}

/*------------------------------------------------------------------------------
  This routine resets the ForcingStep integrator by resetting the partition
  integrators
  ----------------------------------------------------------------------------*/
fn forcingStep_Reset(ark_mem: &ARKodeMem, tR: sunrealtype, yR: &N_Vector) -> i32 {
    let retval = forcingStep_AccessStepMem(ark_mem, "forcingStep_Reset");
    if retval != ARK_SUCCESS {
        return retval;
    }

    let stepper0 = forcingStep_mem_mut(ark_mem).stepper[0].clone();
    let err = SUNStepper_Reset(&stepper0, tR, yR);
    if err != SUN_SUCCESS {
        arkProcessError(
            Some(ark_mem),
            ARK_SUNSTEPPER_ERR,
            line!() as i32,
            "forcingStep_Reset",
            file!(),
            "Resetting the first partition SUNStepper failed",
        );
        return ARK_SUNSTEPPER_ERR;
    }

    let stepper1 = forcingStep_mem_mut(ark_mem).stepper[1].clone();
    let err = SUNStepper_Reset(&stepper1, tR, yR);
    if err != SUN_SUCCESS {
        arkProcessError(
            Some(ark_mem),
            ARK_SUNSTEPPER_ERR,
            line!() as i32,
            "forcingStep_Reset",
            file!(),
            "Resetting the second partition SUNStepper failed",
        );
        return ARK_SUNSTEPPER_ERR;
    }

    ARK_SUCCESS
}

/*------------------------------------------------------------------------------
  This routine sets the step direction of the partition integrators and is
  called once the ForcingStep integrator has updated its step direction.
  ----------------------------------------------------------------------------*/
fn forcingStep_SetStepDirection(ark_mem: &ARKodeMem, stepdir: sunrealtype) -> i32 {
    let retval = forcingStep_AccessStepMem(ark_mem, "forcingStep_SetStepDirection");
    if retval != ARK_SUCCESS {
        return retval;
    }

    let stepper0 = forcingStep_mem_mut(ark_mem).stepper[0].clone();
    let err = SUNStepper_SetStepDirection(&stepper0, stepdir);
    if err != SUN_SUCCESS {
        arkProcessError(
            Some(ark_mem),
            ARK_SUNSTEPPER_ERR,
            line!() as i32,
            "forcingStep_SetStepDirection",
            file!(),
            "Setting the step direction for the first partition SUNStepper failed",
        );
        return ARK_SUNSTEPPER_ERR;
    }

    let stepper1 = forcingStep_mem_mut(ark_mem).stepper[1].clone();
    let err = SUNStepper_SetStepDirection(&stepper1, stepdir);
    if err != SUN_SUCCESS {
        arkProcessError(
            Some(ark_mem),
            ARK_SUNSTEPPER_ERR,
            line!() as i32,
            "forcingStep_SetStepDirection",
            file!(),
            "Setting the step direction for the second partition SUNStepper failed",
        );
        return ARK_SUNSTEPPER_ERR;
    }

    ARK_SUCCESS
}

/*------------------------------------------------------------------------------
  This is just a wrapper to call the user-supplied RHS function,
  f^1(t,y) + f^2(t,y).

  This will be called in one of three 'modes':

     ARK_FULLRHS_START -> called at the beginning of a simulation i.e., at
                          (tn, yn) = (t0, y0) or (tR, yR)

     ARK_FULLRHS_END   -> called at the end of a successful step i.e, at
                          (tcur, ycur) or the start of the subsequent step i.e.,
                          at (tn, yn) = (tcur, ycur) from the end of the last
                          step

     ARK_FULLRHS_OTHER -> called elsewhere (e.g. for dense output)

  The stepper for partition 1 has a state that is inconsistent with the
  ForcingStep integrator, so we cannot pass it the SUN_FULLRHS_END option. For
  partition 2, the state should be consistent, and we can use SUN_FULLRHS_END.
  ----------------------------------------------------------------------------*/
fn forcingStep_FullRHS(
    ark_mem: &ARKodeMem,
    t: sunrealtype,
    y: &N_Vector,
    f: &N_Vector,
    mode: i32,
) -> i32 {
    let retval = forcingStep_AccessStepMem(ark_mem, "forcingStep_FullRHS");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* TODO(SBR): Possible optimization in FULLRHS_START mode. Currently that
     * mode is not forwarded to the SUNSteppers */
    let stepper0 = forcingStep_mem_mut(ark_mem).stepper[0].clone();
    let tempv1 = ark_mem.borrow().tempv1.clone().expect("tempv1");
    let err = SUNStepper_FullRhs(&stepper0, t, y, &tempv1, SUN_FULLRHS_OTHER);
    if err != SUN_SUCCESS {
        arkProcessError(
            Some(ark_mem),
            ARK_RHSFUNC_FAIL,
            line!() as i32,
            "forcingStep_FullRHS",
            file!(),
            &MSG_ARK_RHSFUNC_FAILED(t),
        );
        return ARK_RHSFUNC_FAIL;
    }

    let stepper1 = forcingStep_mem_mut(ark_mem).stepper[1].clone();
    let err = SUNStepper_FullRhs(
        &stepper1,
        t,
        y,
        f,
        if mode == ARK_FULLRHS_END {
            SUN_FULLRHS_END
        } else {
            SUN_FULLRHS_OTHER
        },
    );
    if err != SUN_SUCCESS {
        arkProcessError(
            Some(ark_mem),
            ARK_RHSFUNC_FAIL,
            line!() as i32,
            "forcingStep_FullRHS",
            file!(),
            &MSG_ARK_RHSFUNC_FAILED(t),
        );
        return ARK_RHSFUNC_FAIL;
    }
    let tempv1 = ark_mem.borrow().tempv1.clone().expect("tempv1");
    N_VLinearSum(1.0, f, 1.0, &tempv1, f);

    ARK_SUCCESS
}

/*------------------------------------------------------------------------------
  This routine performs a single step of the forcing method.
  ----------------------------------------------------------------------------*/
fn forcingStep_TakeStep(ark_mem: &ARKodeMem, dsmPtr: &mut sunrealtype, nflagPtr: &mut i32) -> i32 {
    let retval = forcingStep_AccessStepMem(ark_mem, "forcingStep_TakeStep");
    if retval != ARK_SUCCESS {
        return retval;
    }

    *nflagPtr = ARK_SUCCESS; /* No algebraic solver */
    *dsmPtr = ZERO; /* No error estimate */

    let s0 = forcingStep_mem_mut(ark_mem).stepper[0].clone();
    let tout = {
        let m = ark_mem.borrow();
        m.tn + m.h
    };
    let mut tret: sunrealtype = ZERO;

    /* Evolve stepper 0 on its own */
    let (tn, yn) = {
        let m = ark_mem.borrow();
        (m.tn, m.yn.clone().expect("yn"))
    };
    let err = SUNStepper_Reset(&s0, tn, &yn);
    if err != SUN_SUCCESS {
        return ARK_SUNSTEPPER_ERR;
    }

    let err = SUNStepper_SetStopTime(&s0, tout);
    if err != SUN_SUCCESS {
        return ARK_SUNSTEPPER_ERR;
    }

    let ycur = ark_mem.borrow().ycur.clone().expect("ycur");
    let err = SUNStepper_Evolve(&s0, tout, &ycur, &mut tret);
    if err != SUN_SUCCESS {
        return ARK_SUNSTEPPER_ERR;
    }
    forcingStep_mem_mut(ark_mem).n_stepper_evolves[0] += 1;

    let s1 = forcingStep_mem_mut(ark_mem).stepper[1].clone();
    /* A reset is not needed because steeper 1's state is consistent with the
     * forcing method */
    let err = SUNStepper_SetStopTime(&s1, tout);
    if err != SUN_SUCCESS {
        return ARK_SUNSTEPPER_ERR;
    }

    /* Write tendency (ycur - yn)/h into stepper 1 forcing */
    let hinv = {
        let m = ark_mem.borrow();
        1.0 / m.h
    };
    let (ycur, yn, tempv1) = {
        let m = ark_mem.borrow();
        (
            m.ycur.clone().expect("ycur"),
            m.yn.clone().expect("yn"),
            m.tempv1.clone().expect("tempv1"),
        )
    };
    N_VLinearSum(hinv, &ycur, -hinv, &yn, &tempv1);
    let err = SUNStepper_SetForcing(&s1, ZERO, ZERO, &[tempv1], 1);
    if err != SUN_SUCCESS {
        return ARK_SUNSTEPPER_ERR;
    }

    /* Evolve stepper 1 with the forcing */
    let ycur = ark_mem.borrow().ycur.clone().expect("ycur");
    let err = SUNStepper_Evolve(&s1, tout, &ycur, &mut tret);
    if err != SUN_SUCCESS {
        return ARK_SUNSTEPPER_ERR;
    }
    forcingStep_mem_mut(ark_mem).n_stepper_evolves[1] += 1;

    /* Clear the forcing so it doesn't get included in a fullRhs call */
    let err = SUNStepper_SetForcing(&s1, ZERO, ZERO, &[], 0);
    if err != SUN_SUCCESS {
        return ARK_SUNSTEPPER_ERR;
    }

    ARK_SUCCESS
}

/*------------------------------------------------------------------------------
  Prints integrator statistics
  ----------------------------------------------------------------------------*/
fn forcingStep_PrintAllStats(ark_mem: &ARKodeMem, outfile: &SUNFile, fmt: SUNOutputFormat) -> i32 {
    let retval = forcingStep_AccessStepMem(ark_mem, "forcingStep_PrintAllStats");
    if retval != ARK_SUCCESS {
        return retval;
    }

    let n_stepper_evolves = forcingStep_mem_mut(ark_mem).n_stepper_evolves;
    sunfprintf_long(
        outfile,
        fmt,
        SUNFALSE,
        "Partition 1 evolves",
        n_stepper_evolves[0],
    );
    sunfprintf_long(
        outfile,
        fmt,
        SUNFALSE,
        "Partition 2 evolves",
        n_stepper_evolves[1],
    );

    ARK_SUCCESS
}

/*------------------------------------------------------------------------------
  Frees all ForcingStep memory.
  ----------------------------------------------------------------------------*/
fn forcingStep_Free(ark_mem: &ARKodeMem) {
    ark_mem.borrow_mut().step_mem = None;
}

/*------------------------------------------------------------------------------
  This routine outputs the memory from the ForcingStep structure to a specified
  file pointer (useful when debugging).
  ----------------------------------------------------------------------------*/
fn forcingStep_PrintMem(ark_mem: &ARKodeMem, outfile: &SUNFile) {
    let retval = forcingStep_AccessStepMem(ark_mem, "forcingStep_PrintMem");
    if retval != ARK_SUCCESS {
        return;
    }

    /* output long integer quantities */
    let n_stepper_evolves = forcingStep_mem_mut(ark_mem).n_stepper_evolves;
    for k in 0..NUM_PARTITIONS {
        let value = n_stepper_evolves[k as usize];
        outfile.write_str(&format!(
            "ForcingStep: partition {k}: n_stepper_evolves = {value}\n"
        ));
    }
}

/*------------------------------------------------------------------------------
  This routine checks if all required SUNStepper operations are present. If any
  of them are missing it return SUNFALSE.
  ----------------------------------------------------------------------------*/
fn forcingStep_CheckSUNStepper(stepper: &SUNStepper, needs_forcing: sunbooleantype) -> sunbooleantype {
    let ops = stepper.ops.borrow();
    ops.evolve.is_some()
        && ops.reset.is_some()
        && ops.setstoptime.is_some()
        && (!needs_forcing || ops.setforcing.is_some())
}

/*------------------------------------------------------------------------------
  This routine validates arguments when (re)initializing a ForcingStep
  integrator
  ----------------------------------------------------------------------------*/
fn forcingStep_CheckArgs(
    ark_mem: Option<&ARKodeMem>,
    stepper1: &SUNStepper,
    stepper2: &SUNStepper,
    y0: &N_Vector,
) -> i32 {
    /* C: `stepper1 == NULL`, `stepper2 == NULL` and `y0 == NULL` are
       unrepresentable with `&SUNStepper` / `&N_Vector` */
    let _ = y0;

    if !forcingStep_CheckSUNStepper(stepper1, SUNFALSE) {
        arkProcessError(
            ark_mem,
            ARK_ILL_INPUT,
            line!() as i32,
            "forcingStep_CheckArgs",
            file!(),
            "stepper1 does not implement the required operations.",
        );
        return ARK_ILL_INPUT;
    }

    if !forcingStep_CheckSUNStepper(stepper2, SUNTRUE) {
        arkProcessError(
            ark_mem,
            ARK_ILL_INPUT,
            line!() as i32,
            "forcingStep_CheckArgs",
            file!(),
            "stepper2 does not implement the required operations.",
        );
        return ARK_ILL_INPUT;
    }

    ARK_SUCCESS
}

/*------------------------------------------------------------------------------
  This routine initializes the step memory and resets the statistics
  ----------------------------------------------------------------------------*/
fn forcingStep_InitStepMem(
    step_mem: &mut ARKodeForcingStepMemRec,
    stepper1: &SUNStepper,
    stepper2: &SUNStepper,
) {
    step_mem.stepper[0] = stepper1.clone();
    step_mem.stepper[1] = stepper2.clone();
    step_mem.n_stepper_evolves[0] = 0;
    step_mem.n_stepper_evolves[1] = 0;
}

/*------------------------------------------------------------------------------
  Creates the ForcingStep integrator
  ----------------------------------------------------------------------------*/
pub fn ForcingStepCreate(
    stepper1: &SUNStepper,
    stepper2: &SUNStepper,
    t0: sunrealtype,
    y0: &N_Vector,
    sunctx: &SUNContext,
) -> Option<ARKodeMem> {
    let mut retval = forcingStep_CheckArgs(None, stepper1, stepper2, y0);
    if retval != ARK_SUCCESS {
        return None;
    }

    /* C: `sunctx == NULL` (MSG_ARK_NULL_SUNCTX) is unrepresentable here */

    /* Create ark_mem structure and set default values */
    let ark_mem = match arkCreate(sunctx) {
        Some(ark_mem) => ark_mem,
        None => {
            arkProcessError(
                None,
                ARK_MEM_NULL,
                line!() as i32,
                "ForcingStepCreate",
                file!(),
                MSG_ARK_NO_MEM,
            );
            return None;
        }
    };

    /* C `malloc`s an indeterminate record and lets `forcingStep_InitStepMem`
       fill every field; Rust must build a value first, so the arrays are
       seeded with the very handles the helper then assigns. */
    let mut step_mem = ARKodeForcingStepMemRec {
        stepper: [stepper1.clone(), stepper2.clone()],
        n_stepper_evolves: [0; NUM_PARTITIONS as usize],
    };
    forcingStep_InitStepMem(&mut step_mem, stepper1, stepper2);

    /* Attach step_mem structure and function pointers to ark_mem */
    {
        let mut m = ark_mem.borrow_mut();
        m.step_init = Some(forcingStep_Init);
        m.step_fullrhs = Some(forcingStep_FullRHS);
        m.step_reset = Some(forcingStep_Reset);
        m.step_setstepdirection = Some(forcingStep_SetStepDirection);
        m.step = Some(forcingStep_TakeStep);
        m.step_printallstats = Some(forcingStep_PrintAllStats);
        m.step_free = Some(forcingStep_Free);
        m.step_printmem = Some(forcingStep_PrintMem);
        m.step_mem = Some(Box::new(step_mem));
    }

    /* C tests `retval` again here; it is still the ARK_SUCCESS returned by
       forcingStep_CheckArgs, so this branch is dead upstream too. */
    if retval != ARK_SUCCESS {
        arkProcessError(
            Some(&ark_mem),
            retval,
            line!() as i32,
            "ForcingStepCreate",
            file!(),
            "Error setting default solver options",
        );
        let mut ark_mem = Some(ark_mem);
        ARKodeFree(&mut ark_mem);
        return None;
    }

    /* Initialize main ARKODE infrastructure */
    retval = arkInit(&ark_mem, t0, y0, FIRST_INIT);
    if retval != ARK_SUCCESS {
        arkProcessError(
            Some(&ark_mem),
            retval,
            line!() as i32,
            "ForcingStepCreate",
            file!(),
            "Unable to initialize main ARKODE infrastructure",
        );
        let mut ark_mem = Some(ark_mem);
        ARKodeFree(&mut ark_mem);
        return None;
    }

    let _ = ARKodeSetInterpolantType(&ark_mem, ARK_INTERP_LAGRANGE);

    Some(ark_mem)
}

/*------------------------------------------------------------------------------
  This routine re-initializes the ForcingStep module to solve a new problem of
  the same size as was previously solved. This routine should also be called
  when the problem dynamics or desired solvers have changed dramatically, so
  that the problem integration should resume as if started from scratch.

  Note all internal counters are set to 0 on re-initialization.
  ----------------------------------------------------------------------------*/
pub fn ForcingStepReInit(
    arkode_mem: &ARKodeMem,
    stepper1: &SUNStepper,
    stepper2: &SUNStepper,
    t0: sunrealtype,
    y0: &N_Vector,
) -> i32 {
    let ark_mem = arkode_mem;

    let retval = forcingStep_AccessARKODEStepMem(ark_mem, "ForcingStepReInit");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* Check if ark_mem was allocated */
    let MallocDone = ark_mem.borrow().MallocDone;
    if MallocDone == SUNFALSE {
        arkProcessError(
            Some(ark_mem),
            ARK_NO_MALLOC,
            line!() as i32,
            "ForcingStepReInit",
            file!(),
            MSG_ARK_NO_MALLOC,
        );
        return ARK_NO_MALLOC;
    }

    let retval = forcingStep_CheckArgs(Some(ark_mem), stepper1, stepper2, y0);
    if retval != ARK_SUCCESS {
        return retval;
    }

    {
        let mut step_mem = forcingStep_mem_mut(ark_mem);
        forcingStep_InitStepMem(&mut step_mem, stepper1, stepper2);
    }

    /* Initialize main ARKODE infrastructure */
    let retval = arkInit(ark_mem, t0, y0, FIRST_INIT);
    if retval != ARK_SUCCESS {
        arkProcessError(
            Some(ark_mem),
            retval,
            line!() as i32,
            "ForcingStepReInit",
            file!(),
            "Unable to initialize main ARKODE infrastructure",
        );
        return retval;
    }

    ARK_SUCCESS
}

/*------------------------------------------------------------------------------
  Accesses the number of times a given partition was evolved
  ----------------------------------------------------------------------------*/
pub fn ForcingStepGetNumEvolves(arkode_mem: &ARKodeMem, partition: i32, evolves: &mut i64) -> i32 {
    let ark_mem = arkode_mem;

    let retval = forcingStep_AccessARKODEStepMem(ark_mem, "ForcingStepGetNumEvolves");
    if retval != ARK_SUCCESS {
        return retval;
    }

    if partition >= NUM_PARTITIONS {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "ForcingStepGetNumEvolves",
            file!(),
            &format!("The partition index is {partition} but there are only 2 partitions"),
        );
        return ARK_ILL_INPUT;
    }

    let n_stepper_evolves = forcingStep_mem_mut(ark_mem).n_stepper_evolves;
    if partition < 0 {
        *evolves = n_stepper_evolves[0] + n_stepper_evolves[1];
    } else {
        *evolves = n_stepper_evolves[partition as usize];
    }

    ARK_SUCCESS
}
