//! Port of `src/arkode/arkode_splittingstep.c` (+ `arkode_splittingstep_impl.h`
//! and the SplittingStep integrator declarations of
//! `include/arkode/arkode_splittingstep.h`; the coefficient structure and its
//! constructors live in `arkode_splittingstep_coefficients.rs`, matching the
//! upstream file split).
//!
//! ARKODE's operator-splitting module: the outer integrator advances a
//! sequence of `SUNStepper` partitions over subintervals dictated by the
//! splitting coefficients, then forms a weighted sum of the sequential
//! methods' results.
//!
//! Binding notes (all forced by the frozen contract in `arkode_impl.rs`):
//! * `ARKodeSplittingStepMemRec` is stored BY VALUE in
//!   `ark_mem.step_mem: Option<Box<dyn Any>>`; `splittingStep_mem_mut` is the
//!   single module-local downcast helper. The guard it returns IS a borrow of
//!   the mem, so it is never held across `arkProcessError`, an `N_Vector`
//!   operation, or any `SUNStepper_*` call — every such site copies the fields
//!   it needs into locals in a scoped block, drops the guard, and only then
//!   calls out.
//! * C `SUNStepper*` (array) becomes `&[SUNStepper]`; the record owns a
//!   `Vec<SUNStepper>` exactly as C owns its `malloc`ed copy of the array.
//! * The private helpers that C threads `ARKodeSplittingStepMem step_mem`
//!   through (`splittingStep_SetCoefficients`, `splittingStep_SequentialMethod`,
//!   `splittingStep_InitStepMem`) drop that parameter: the record lives inside
//!   `ark_mem`, so it cannot be passed alongside a borrow of the mem. Each
//!   reaches it through `splittingStep_mem_mut` at the point of use, which is
//!   the seam the contract prescribes for `splittingStep_AccessStepMem`.
//! * `splittingStep_AccessStepMem` survives as a presence check returning the
//!   C flag (and emitting the C message); `splittingStep_AccessARKODEStepMem`
//!   loses only its `arkode_mem == NULL` branch (unrepresentable).
//!
//! Logging: `SUNLogInfo`/`SUNLogExtraDebugVec` compile away at
//! `SUNDIALS_LOGGING_LEVEL=2` and are omitted; the `ARK_WARNING` in
//! `splittingStep_SetCoefficients` does print and is kept.

use std::cell::RefMut;

use crate::arkode::{arkCreate, arkInit, ARKodeFree};
use crate::arkode_impl::*;
use crate::arkode_io::ARKodeSetInterpolantType;
use crate::arkode_splittingstep_coefficients::*;
use sundials_core::sundials_context::SUNContext;
use sundials_core::sundials_errors::SUN_SUCCESS;
use sundials_core::sundials_math::{SUNMAX, SUNMIN};
use sundials_core::sundials_nvector::{N_VLinearSum, N_VScale, N_Vector};
use sundials_core::sundials_stepper::*;
use sundials_core::sundials_types::*;
use sundials_core::sundials_utils::{sunfprintf_long, SUNFile};

/*---------------------------------------------------------------
  Types : struct ARKodeSplittingStepMemRec, ARKodeSplittingStepMem
  (arkode_splittingstep_impl.h)
  ---------------------------------------------------------------*/
pub struct ARKodeSplittingStepMemRec {
    /// C `SUNStepper* steppers` (`malloc`ed copy of the caller's array;
    /// `NULL` == empty here)
    pub steppers: Vec<SUNStepper>,
    pub coefficients: Option<SplittingStepCoefficients>,
    /// C `long int* n_stepper_evolves` (`NULL` == empty here)
    pub n_stepper_evolves: Vec<i64>,

    pub istage: i32,
    pub partitions: i32,
    pub order: i32,
}

/// Downcast helper: view `ark_mem.step_mem` as the SplittingStep memory
/// record. Panics if no step memory is attached or it is not this stepper's
/// record (C would blindly cast the `void*` — UB maps to a panic,
/// deviation class 5). NEVER hold the guard across `arkProcessError`, a
/// callback, an `N_Vector` operation, or a `SUNStepper_*` call.
pub fn splittingStep_mem_mut(ark_mem: &ARKodeMem) -> RefMut<'_, ARKodeSplittingStepMemRec> {
    RefMut::map(ark_mem.borrow_mut(), |m| {
        m.step_mem
            .as_mut()
            .expect("step_mem set")
            .downcast_mut::<ARKodeSplittingStepMemRec>()
            .expect("SplittingStep step memory")
    })
}

/*------------------------------------------------------------------------------
  Shortcut routine to unpack step_mem structure from ark_mem. If missing it
  returns ARK_MEM_NULL.
  ----------------------------------------------------------------------------*/
fn splittingStep_AccessStepMem(ark_mem: &ARKodeMem, fname: &str) -> i32 {
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
fn splittingStep_AccessARKODEStepMem(ark_mem: &ARKodeMem, fname: &str) -> i32 {
    /* access ARKodeMem structure: C's `arkode_mem == NULL` branch is
       unrepresentable with `&ARKodeMem` */
    let _ = fname;

    splittingStep_AccessStepMem(ark_mem, "splittingStep_AccessARKODEStepMem")
}

/*------------------------------------------------------------------------------
  This routine determines the splitting coefficients to use based on the desired
  accuracy.
  ----------------------------------------------------------------------------*/
fn splittingStep_SetCoefficients(ark_mem: &ARKodeMem) -> i32 {
    let (have_coefficients, order, partitions) = {
        let step_mem = splittingStep_mem_mut(ark_mem);
        (
            step_mem.coefficients.is_some(),
            step_mem.order,
            step_mem.partitions,
        )
    };

    if have_coefficients {
        return ARK_SUCCESS;
    }

    let coefficients = if order <= 1 {
        /* Lie-Trotter is the default (order < 1) */
        SplittingStepCoefficients_LieTrotter(partitions)
    } else if order == 3 {
        SplittingStepCoefficients_ThirdOrderSuzuki(partitions)
    } else if order % 2 == 0 {
        /* Triple jump only works for even order */
        SplittingStepCoefficients_TripleJump(partitions, order)
    } else {
        /* Bump the order up to be even but with a warning */
        let new_order = order + 1;
        arkProcessError(
            Some(ark_mem),
            ARK_WARNING,
            line!() as i32,
            "splittingStep_SetCoefficients",
            file!(),
            &format!("No splitting method at requested order, using q={new_order}."),
        );
        SplittingStepCoefficients_TripleJump(partitions, new_order)
    };

    if coefficients.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_FAIL,
            line!() as i32,
            "splittingStep_SetCoefficients",
            file!(),
            "Failed to allocate splitting coefficients",
        );
        return ARK_MEM_FAIL;
    }

    splittingStep_mem_mut(ark_mem).coefficients = coefficients;

    ARK_SUCCESS
}

/*-----------------------------------------------------------------------------
  This routine is called just prior to performing internal time steps (after all
  user "set" routines have been called) from within arkInitialSetup.

  With initialization types FIRST_INIT this routine:
  - sets/checks the splitting coefficients to be used

  With other initialization types, this routine does nothing.
  ----------------------------------------------------------------------------*/
fn splittingStep_Init(ark_mem: &ARKodeMem, init_type: i32) -> i32 {
    let retval = splittingStep_AccessStepMem(ark_mem, "splittingStep_Init");
    if retval != ARK_SUCCESS {
        return retval;
    }

    let interp_type = ark_mem.borrow().interp_type;
    if interp_type == ARK_INTERP_HERMITE {
        let partitions = splittingStep_mem_mut(ark_mem).partitions;
        for i in 0..partitions {
            let stepper = splittingStep_mem_mut(ark_mem).steppers[i as usize].clone();
            if stepper.ops.borrow().fullrhs.is_none() {
                arkProcessError(
                    Some(ark_mem),
                    ARK_ILL_INPUT,
                    line!() as i32,
                    "splittingStep_Init",
                    file!(),
                    &format!(
                        "steppers[{i}] must implement SUNStepper_FullRhs when using Hermite \
                         interpolation"
                    ),
                );
                return ARK_ILL_INPUT;
            }
        }
    }

    /* inform arkode to ensure that ycur==yn upon entry to TakeStep function */
    ark_mem.borrow_mut().ensure_ycur = SUNTRUE;

    /* immediately return if resize or reset */
    if init_type == RESIZE_INIT || init_type == RESET_INIT {
        return ARK_SUCCESS;
    }

    /* assume fixed step size */
    let fixedstep = ark_mem.borrow().fixedstep;
    if !fixedstep {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "splittingStep_Init",
            file!(),
            "Adaptive outer time stepping is not currently supported",
        );
        return ARK_ILL_INPUT;
    }

    let retval = splittingStep_SetCoefficients(ark_mem);
    if retval != ARK_SUCCESS {
        return retval;
    }

    let coefficients = {
        splittingStep_mem_mut(ark_mem)
            .coefficients
            .as_ref()
            .expect("SplittingStep coefficients")
            .clone()
    };
    let order = coefficients.borrow().order;
    let interp_degree = ark_mem.borrow().interp_degree;
    ark_mem.borrow_mut().interp_degree = SUNMAX(1, SUNMIN(order - 1, interp_degree));

    ARK_SUCCESS
}

/*------------------------------------------------------------------------------
  This is just a wrapper to call the user-supplied RHS function,
  f^1(t,y) + f^2(t,y) + ... + f^P(t,y).

  This will be called in one of three 'modes':

     ARK_FULLRHS_START -> called at the beginning of a simulation i.e., at
                          (tn, yn) = (t0, y0) or (tR, yR)

     ARK_FULLRHS_END   -> called at the end of a successful step i.e, at
                          (tcur, ycur) or the start of the subsequent step i.e.,
                          at (tn, yn) = (tcur, ycur) from the end of the last
                          step

     ARK_FULLRHS_OTHER -> called elsewhere (e.g. for dense output)

  In SplittingStep, we accumulate the RHS functions in ARK_FULLRHS_OTHER mode.
  Generally, inner steppers will not have the correct yn when this function is
  called and will not be able to reuse a function evaluation since their state
  resets at the next SUNStepper_Evolve call.
  ----------------------------------------------------------------------------*/
fn splittingStep_FullRHS(
    ark_mem: &ARKodeMem,
    t: sunrealtype,
    y: &N_Vector,
    f: &N_Vector,
    mode: i32,
) -> i32 {
    let retval = splittingStep_AccessStepMem(ark_mem, "splittingStep_FullRHS");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* C: SUNDIALS_MAYBE_UNUSED int mode */
    let _ = mode;

    let partitions = splittingStep_mem_mut(ark_mem).partitions;
    for i in 0..partitions {
        let stepper = splittingStep_mem_mut(ark_mem).steppers[i as usize].clone();
        let tempv1 = ark_mem.borrow().tempv1.clone().expect("tempv1");
        let err = SUNStepper_FullRhs(
            &stepper,
            t,
            y,
            if i == 0 { f } else { &tempv1 },
            SUN_FULLRHS_OTHER,
        );
        if err != SUN_SUCCESS {
            arkProcessError(
                Some(ark_mem),
                ARK_RHSFUNC_FAIL,
                line!() as i32,
                "splittingStep_FullRHS",
                file!(),
                &MSG_ARK_RHSFUNC_FAILED(t),
            );
            return ARK_RHSFUNC_FAIL;
        }
        if i > 0 {
            N_VLinearSum(ONE, f, ONE, &tempv1, f);
        }
    }

    ARK_SUCCESS
}

/*------------------------------------------------------------------------------
  This routine performs a sequential operator splitting method
  ----------------------------------------------------------------------------*/
fn splittingStep_SequentialMethod(ark_mem: &ARKodeMem, i: i32, y: &N_Vector) -> i32 {
    let coefficients = {
        splittingStep_mem_mut(ark_mem)
            .coefficients
            .as_ref()
            .expect("SplittingStep coefficients")
            .clone()
    };

    let (stages, partitions) = {
        let c = coefficients.borrow();
        (c.stages, c.partitions)
    };

    for j in 0..stages {
        for k in 0..partitions {
            let (beta_start, beta_end) = {
                let c = coefficients.borrow();
                (
                    c.beta[i as usize][j as usize][k as usize],
                    c.beta[i as usize][(j + 1) as usize][k as usize],
                )
            };

            if beta_start == beta_end {
                continue;
            }

            let (tn, h) = {
                let m = ark_mem.borrow();
                (m.tn, m.h)
            };
            let t_start = tn + beta_start * h;
            let t_end = tn + beta_end * h;

            let stepper = splittingStep_mem_mut(ark_mem).steppers[k as usize].clone();
            /* TODO(SBR): A potential future optimization is removing this reset and
             * a call to SUNStepper_SetStopTime later for methods that start a step
             * evolving the same partition the last step ended with (essentially a
             * FSAL property). Care is needed when a reset occurs, the step direction
             * changes, the coefficients change, etc. */
            let err = SUNStepper_Reset(&stepper, t_start, y);
            if err != SUN_SUCCESS {
                return ARK_SUNSTEPPER_ERR;
            }

            let err = SUNStepper_SetStepDirection(&stepper, t_end - t_start);
            if err != SUN_SUCCESS {
                return ARK_SUNSTEPPER_ERR;
            }

            let err = SUNStepper_SetStopTime(&stepper, t_end);
            if err != SUN_SUCCESS {
                return ARK_SUNSTEPPER_ERR;
            }

            let mut tret: sunrealtype = ZERO;
            let err = SUNStepper_Evolve(&stepper, t_end, y, &mut tret);
            if err != SUN_SUCCESS {
                return ARK_SUNSTEPPER_ERR;
            }
            splittingStep_mem_mut(ark_mem).n_stepper_evolves[k as usize] += 1;
        }
    }

    ARK_SUCCESS
}

/*------------------------------------------------------------------------------
  This routine performs a single step of the splitting method.
  ----------------------------------------------------------------------------*/
fn splittingStep_TakeStep(
    ark_mem: &ARKodeMem,
    dsmPtr: &mut sunrealtype,
    nflagPtr: &mut i32,
) -> i32 {
    let retval = splittingStep_AccessStepMem(ark_mem, "splittingStep_TakeStep");
    if retval != ARK_SUCCESS {
        return retval;
    }

    *nflagPtr = ARK_SUCCESS; /* No algebraic solver */
    *dsmPtr = ZERO; /* No error estimate */

    let coefficients = {
        splittingStep_mem_mut(ark_mem)
            .coefficients
            .as_ref()
            .expect("SplittingStep coefficients")
            .clone()
    };

    splittingStep_mem_mut(ark_mem).istage = 0;
    let ycur = ark_mem.borrow().ycur.clone().expect("ycur");
    let retval = splittingStep_SequentialMethod(ark_mem, 0, &ycur);
    if retval != ARK_SUCCESS {
        return retval;
    }

    let alpha0 = coefficients.borrow().alpha[0];
    if alpha0 != ONE {
        let ycur = ark_mem.borrow().ycur.clone().expect("ycur");
        N_VScale(alpha0, &ycur, &ycur);
    }

    let sequential_methods = coefficients.borrow().sequential_methods;
    for i in 1..sequential_methods {
        splittingStep_mem_mut(ark_mem).istage = i;

        let (yn, tempv1) = {
            let m = ark_mem.borrow();
            (
                m.yn.clone().expect("yn"),
                m.tempv1.clone().expect("tempv1"),
            )
        };
        N_VScale(ONE, &yn, &tempv1);
        let retval = splittingStep_SequentialMethod(ark_mem, i, &tempv1);
        if retval != ARK_SUCCESS {
            return retval;
        }
        let alpha_i = coefficients.borrow().alpha[i as usize];
        let (ycur, tempv1) = {
            let m = ark_mem.borrow();
            (
                m.ycur.clone().expect("ycur"),
                m.tempv1.clone().expect("tempv1"),
            )
        };
        N_VLinearSum(ONE, &ycur, alpha_i, &tempv1, &ycur);
    }

    ARK_SUCCESS
}

/*------------------------------------------------------------------------------
  Prints integrator statistics
  ----------------------------------------------------------------------------*/
fn splittingStep_PrintAllStats(
    ark_mem: &ARKodeMem,
    outfile: &SUNFile,
    fmt: SUNOutputFormat,
) -> i32 {
    let retval = splittingStep_AccessStepMem(ark_mem, "splittingStep_PrintAllStats");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* C builds each label with `snprintf(name_buf, SUN_TABLE_WIDTH, ...)`;
       "Partition <k+1> evolves" cannot reach the 28-character truncation
       point for any representable `int` partition count. */
    let partitions = splittingStep_mem_mut(ark_mem).partitions;
    for k in 0..partitions {
        let name_buf = format!("Partition {} evolves", k + 1);
        let n_stepper_evolves = splittingStep_mem_mut(ark_mem).n_stepper_evolves[k as usize];
        sunfprintf_long(outfile, fmt, SUNFALSE, &name_buf, n_stepper_evolves);
    }

    ARK_SUCCESS
}

/*------------------------------------------------------------------------------
  Outputs all solver parameters to the provided file pointer.
  ----------------------------------------------------------------------------*/
fn splittingStep_WriteParameters(ark_mem: &ARKodeMem, fp: &SUNFile) -> i32 {
    let retval = splittingStep_AccessStepMem(ark_mem, "splittingStep_WriteParameters");
    if retval != ARK_SUCCESS {
        return retval;
    }

    let order = splittingStep_mem_mut(ark_mem).order;
    fp.write_str(&format!(
        "SplittingStep time step module parameters:\n  Method order {order}\n\n"
    ));

    ARK_SUCCESS
}

/*------------------------------------------------------------------------------
  Frees all SplittingStep memory.
  ----------------------------------------------------------------------------*/
fn splittingStep_Free(ark_mem: &ARKodeMem) {
    /* C frees the steppers array, the evolve counters, the coefficients, and
       the record itself; dropping the boxed record does all of it. */
    ark_mem.borrow_mut().step_mem = None;
}

/*------------------------------------------------------------------------------
  This routine outputs the memory from the SplittingStep structure to a
  specified file pointer (useful when debugging).
  ----------------------------------------------------------------------------*/
fn splittingStep_PrintMem(ark_mem: &ARKodeMem, outfile: &SUNFile) {
    let retval = splittingStep_AccessStepMem(ark_mem, "splittingStep_PrintMem");
    if retval != ARK_SUCCESS {
        return;
    }

    /* output integer quantities */
    let (istage, partitions, order) = {
        let step_mem = splittingStep_mem_mut(ark_mem);
        (step_mem.istage, step_mem.partitions, step_mem.order)
    };
    outfile.write_str(&format!("SplittingStep: istage = {istage}\n"));
    outfile.write_str(&format!("SplittingStep: partitions = {partitions}\n"));
    outfile.write_str(&format!("SplittingStep: order = {order}\n"));

    /* output long integer quantities */
    for k in 0..partitions {
        let n_stepper_evolves = splittingStep_mem_mut(ark_mem).n_stepper_evolves[k as usize];
        outfile.write_str(&format!(
            "SplittingStep: partition {k}: n_stepper_evolves = {n_stepper_evolves}\n"
        ));
    }

    /* output sunrealtype quantities */
    outfile.write_str("SplittingStep: Coefficients:\n");
    let coefficients = splittingStep_mem_mut(ark_mem).coefficients.clone();
    SplittingStepCoefficients_Write(coefficients.as_ref(), outfile);
}

/*------------------------------------------------------------------------------
  Specifies the method order
  ----------------------------------------------------------------------------*/
fn splittingStep_SetOrder(ark_mem: &ARKodeMem, order: i32) -> i32 {
    let retval = splittingStep_AccessStepMem(ark_mem, "splittingStep_SetOrder");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* set user-provided value, or default, depending on argument */
    {
        let mut step_mem = splittingStep_mem_mut(ark_mem);
        step_mem.order = SUNMAX(1, order);

        SplittingStepCoefficients_Destroy(&mut step_mem.coefficients);
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  Returns the current stage index and number of stages
  ---------------------------------------------------------------*/
fn splittingStep_GetStageIndex(
    ark_mem: &ARKodeMem,
    istage: &mut i32,
    num_stages: &mut i32,
) -> i32 {
    let retval = splittingStep_AccessStepMem(ark_mem, "splittingStep_GetStageIndex");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* if coefficients structure is not yet available, return defaults */
    let coefficients = splittingStep_mem_mut(ark_mem).coefficients.clone();
    match coefficients {
        None => {
            *istage = -1;
            *num_stages = -1;
            arkProcessError(
                Some(ark_mem),
                ARK_MEM_NULL,
                line!() as i32,
                "splittingStep_GetStageIndex",
                file!(),
                "coefficient table not allocated",
            );
            /* C returns `retval`, which is still ARK_SUCCESS here */
            return retval;
        }
        Some(coefficients) => {
            *istage = splittingStep_mem_mut(ark_mem).istage;
            *num_stages = coefficients.borrow().sequential_methods;
        }
    }

    ARK_SUCCESS
}

/*------------------------------------------------------------------------------
  Routine to set SplittingStep options
  ----------------------------------------------------------------------------*/
fn splittingStep_SetOptions(
    ark_mem: &ARKodeMem,
    argidx: &mut i32,
    argv: &[String],
    offset: usize,
    arg_used: &mut sunbooleantype,
) -> i32 {
    /* The only SplittingStep-specific "Set" routine takes a custom set of
       coefficients; however, these may be specified by name, so here we'll support
       a key to specify the SplittingStepCoefficients by name,
       create the coefficients with that name, attach it to SplittingStep (who copies its
       values), and then frees the coefficients. */
    if &argv[*argidx as usize][offset..] == "splitting_coefficients_name" {
        *argidx += 1;
        let mut Coefficients =
            SplittingStepCoefficients_LoadCoefficientsByName(&argv[*argidx as usize]);
        if Coefficients.is_none() {
            arkProcessError(
                Some(ark_mem),
                ARK_ILL_INPUT,
                line!() as i32,
                "splittingStep_SetOptions",
                file!(),
                &format!(
                    "error setting key {} {} (invalid coefficients name)",
                    argv[(*argidx - 1) as usize],
                    argv[*argidx as usize]
                ),
            );
            return ARK_ILL_INPUT;
        }
        let retval = SplittingStepSetCoefficients(ark_mem, Coefficients.as_ref());
        SplittingStepCoefficients_Destroy(&mut Coefficients);
        if retval != ARK_SUCCESS {
            arkProcessError(
                Some(ark_mem),
                retval,
                line!() as i32,
                "splittingStep_SetOptions",
                file!(),
                &format!(
                    "error setting key {} {} (SetCoefficients failed)",
                    argv[(*argidx - 1) as usize],
                    argv[*argidx as usize]
                ),
            );
            return retval;
        }
        *arg_used = SUNTRUE;
        return ARK_SUCCESS;
    }

    ARK_SUCCESS
}

/*------------------------------------------------------------------------------
  Resets all SplittingStep optional inputs to their default values. Does not
  change problem-defining function pointers or user_data pointer.
  ----------------------------------------------------------------------------*/
fn splittingStep_SetDefaults(ark_mem: &ARKodeMem) -> i32 {
    let retval = splittingStep_AccessStepMem(ark_mem, "splittingStep_SetDefaults");
    if retval != ARK_SUCCESS {
        return retval;
    }

    splittingStep_SetOrder(ark_mem, 0)
}

/*------------------------------------------------------------------------------
  This routine checks if all required SUNStepper operations are present. If any
  of them are missing it return SUNFALSE.
  ----------------------------------------------------------------------------*/
fn splittingStep_CheckSUNStepper(stepper: &SUNStepper) -> sunbooleantype {
    let ops = stepper.ops.borrow();
    ops.evolve.is_some()
        && ops.reset.is_some()
        && ops.setstoptime.is_some()
        && ops.setstepdirection.is_some()
}

/*------------------------------------------------------------------------------
  This routine validates arguments when (re)initializing a SplittingStep
  integrator
  ----------------------------------------------------------------------------*/
fn splittingStep_CheckArgs(
    ark_mem: Option<&ARKodeMem>,
    steppers: &[SUNStepper],
    partitions: i32,
    y0: &N_Vector,
) -> i32 {
    /* C: `steppers == NULL`, `steppers[i] == NULL` and `y0 == NULL` are
       unrepresentable with `&[SUNStepper]` / `&N_Vector` */
    let _ = y0;

    if partitions <= 1 {
        arkProcessError(
            ark_mem,
            ARK_ILL_INPUT,
            line!() as i32,
            "splittingStep_CheckArgs",
            file!(),
            "The number of partitions must be greater than one",
        );
        return ARK_ILL_INPUT;
    }

    for i in 0..partitions {
        if !splittingStep_CheckSUNStepper(&steppers[i as usize]) {
            arkProcessError(
                ark_mem,
                ARK_ILL_INPUT,
                line!() as i32,
                "splittingStep_CheckArgs",
                file!(),
                &format!("stepper[{i}] does not implement the required operations."),
            );
            return ARK_ILL_INPUT;
        }
    }

    ARK_SUCCESS
}

/*------------------------------------------------------------------------------
  This routine initializes the step memory and resets the statistics
  ----------------------------------------------------------------------------*/
fn splittingStep_InitStepMem(
    ark_mem: &ARKodeMem,
    steppers: &[SUNStepper],
    partitions: i32,
) -> i32 {
    /* C frees any previous arrays and `malloc`s/`calloc`s new ones, reporting
       ARK_MEM_FAIL if the steppers array cannot be allocated; allocation
       cannot fail here. */
    let mut step_mem = splittingStep_mem_mut(ark_mem);
    step_mem.steppers = steppers[..partitions as usize].to_vec();
    step_mem.n_stepper_evolves = vec![0; partitions as usize];

    /* If the number of partitions changed, the coefficients are no longer
     * compatible and must be cleared. If a user previously called ARKodeSetOrder
     * that will still be respected at the next call to ARKodeEvolve */
    if step_mem.partitions != partitions {
        SplittingStepCoefficients_Destroy(&mut step_mem.coefficients);
    }
    step_mem.partitions = partitions;

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  Creates the SplittingStep integrator
  ---------------------------------------------------------------*/
pub fn SplittingStepCreate(
    steppers: &[SUNStepper],
    partitions: i32,
    t0: sunrealtype,
    y0: &N_Vector,
    sunctx: &SUNContext,
) -> Option<ARKodeMem> {
    let retval = splittingStep_CheckArgs(None, steppers, partitions, y0);
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
                "SplittingStepCreate",
                file!(),
                MSG_ARK_NO_MEM,
            );
            return None;
        }
    };

    /* C `malloc`s the record, sets the six fields below, calls
       `splittingStep_InitStepMem` on the detached record, and only then
       attaches it. `splittingStep_InitStepMem` reaches the record through
       `ark_mem`, so the port attaches first — observationally identical
       (an InitStepMem failure would free `ark_mem` either way, and C leaks
       the unattached record in that case). */
    let step_mem = ARKodeSplittingStepMemRec {
        steppers: Vec::new(),
        coefficients: None,
        n_stepper_evolves: Vec::new(),
        istage: 0,
        partitions,
        order: 0,
    };
    ark_mem.borrow_mut().step_mem = Some(Box::new(step_mem));

    let retval = splittingStep_InitStepMem(&ark_mem, steppers, partitions);
    if retval != ARK_SUCCESS {
        let mut ark_mem = Some(ark_mem);
        ARKodeFree(&mut ark_mem);
        return None;
    }

    /* Attach step_mem structure and function pointers to ark_mem */
    {
        let mut m = ark_mem.borrow_mut();
        m.step_init = Some(splittingStep_Init);
        m.step_fullrhs = Some(splittingStep_FullRHS);
        m.step = Some(splittingStep_TakeStep);
        m.step_printallstats = Some(splittingStep_PrintAllStats);
        m.step_writeparameters = Some(splittingStep_WriteParameters);
        m.step_free = Some(splittingStep_Free);
        m.step_printmem = Some(splittingStep_PrintMem);
        m.step_setoptions = Some(splittingStep_SetOptions);
        m.step_setdefaults = Some(splittingStep_SetDefaults);
        m.step_setorder = Some(splittingStep_SetOrder);
        m.step_getstageindex = Some(splittingStep_GetStageIndex);
    }

    /* Set default values for ARKStep optional inputs */
    let retval = splittingStep_SetDefaults(&ark_mem);
    if retval != ARK_SUCCESS {
        arkProcessError(
            Some(&ark_mem),
            retval,
            line!() as i32,
            "SplittingStepCreate",
            file!(),
            "Error setting default solver options",
        );
        let mut ark_mem = Some(ark_mem);
        ARKodeFree(&mut ark_mem);
        return None;
    }

    /* Initialize main ARKODE infrastructure */
    let retval = arkInit(&ark_mem, t0, y0, FIRST_INIT);
    if retval != ARK_SUCCESS {
        arkProcessError(
            Some(&ark_mem),
            retval,
            line!() as i32,
            "SplittingStepCreate",
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
  This routine re-initializes the SplittingStep module to solve a new problem of
  the same size as was previously solved. This routine should also be called
  when the problem dynamics or desired solvers have changed dramatically, so
  that the problem integration should resume as if started from scratch.

  Note all internal counters are set to 0 on re-initialization.
  ----------------------------------------------------------------------------*/
pub fn SplittingStepReInit(
    arkode_mem: &ARKodeMem,
    steppers: &[SUNStepper],
    partitions: i32,
    t0: sunrealtype,
    y0: &N_Vector,
) -> i32 {
    let ark_mem = arkode_mem;

    let retval = splittingStep_AccessARKODEStepMem(ark_mem, "SplittingStepReInit");
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
            "SplittingStepReInit",
            file!(),
            MSG_ARK_NO_MALLOC,
        );
        return ARK_NO_MALLOC;
    }

    let retval = splittingStep_CheckArgs(Some(ark_mem), steppers, partitions, y0);
    if retval != ARK_SUCCESS {
        return retval;
    }

    splittingStep_InitStepMem(ark_mem, steppers, partitions);

    /* Initialize main ARKODE infrastructure */
    let retval = arkInit(ark_mem, t0, y0, FIRST_INIT);
    if retval != ARK_SUCCESS {
        arkProcessError(
            Some(ark_mem),
            retval,
            line!() as i32,
            "SplittingStepReInit",
            file!(),
            "Unable to initialize main ARKODE infrastructure",
        );
        return retval;
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  Sets the SplittingStep coefficients.
  ---------------------------------------------------------------*/
pub fn SplittingStepSetCoefficients(
    arkode_mem: &ARKodeMem,
    coefficients: Option<&SplittingStepCoefficients>,
) -> i32 {
    let ark_mem = arkode_mem;

    let retval = splittingStep_AccessARKODEStepMem(ark_mem, "SplittingStepSetCoefficients");
    if retval != ARK_SUCCESS {
        return retval;
    }

    let coefficients = match coefficients {
        None => {
            arkProcessError(
                Some(ark_mem),
                ARK_ILL_INPUT,
                line!() as i32,
                "SplittingStepSetCoefficients",
                file!(),
                "Splitting coefficients must be non-NULL",
            );
            return ARK_ILL_INPUT;
        }
        Some(coefficients) => coefficients,
    };

    let step_partitions = splittingStep_mem_mut(ark_mem).partitions;
    let coef_partitions = coefficients.borrow().partitions;
    if step_partitions != coef_partitions {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "SplittingStepSetCoefficients",
            file!(),
            &format!(
                "The splitting method has {step_partitions} partitions but the coefficients have \
                 {coef_partitions}."
            ),
        );
        return ARK_ILL_INPUT;
    }

    {
        let mut step_mem = splittingStep_mem_mut(ark_mem);
        SplittingStepCoefficients_Destroy(&mut step_mem.coefficients);
        step_mem.coefficients = SplittingStepCoefficients_Copy(coefficients);
    }
    let copied = splittingStep_mem_mut(ark_mem).coefficients.is_some();
    if !copied {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_FAIL,
            line!() as i32,
            "SplittingStepSetCoefficients",
            file!(),
            "Failed to copy splitting coefficients",
        );
        /* C reports ARK_MEM_FAIL but returns ARK_MEM_NULL */
        return ARK_MEM_NULL;
    }

    ARK_SUCCESS
}

/*------------------------------------------------------------------------------
  Accesses the number of times a given partition was evolved
  ----------------------------------------------------------------------------*/
pub fn SplittingStepGetNumEvolves(
    arkode_mem: &ARKodeMem,
    partition: i32,
    evolves: &mut i64,
) -> i32 {
    let ark_mem = arkode_mem;

    let retval = splittingStep_AccessARKODEStepMem(ark_mem, "SplittingStepGetNumEvolves");
    if retval != ARK_SUCCESS {
        return retval;
    }

    let partitions = splittingStep_mem_mut(ark_mem).partitions;
    if partition >= partitions {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "SplittingStepGetNumEvolves",
            file!(),
            &format!(
                "The partition index is {partition} but there are only {partitions} partitions"
            ),
        );
        return ARK_ILL_INPUT;
    }

    if partition < 0 {
        *evolves = 0;
        for k in 0..partitions {
            *evolves += splittingStep_mem_mut(ark_mem).n_stepper_evolves[k as usize];
        }
    } else {
        *evolves = splittingStep_mem_mut(ark_mem).n_stepper_evolves[partition as usize];
    }

    ARK_SUCCESS
}
