//! Port of `src/arkode/arkode_erkstep_io.c`: the optional input and output
//! functions for the ARKODE ERKStep time stepper module.
//!
//! Binding notes: the stepper content record is reached through
//! [`erkStep_mem_mut`] (defined in [`crate::arkode_erkstep`]) — never held
//! across `arkProcessError`, a callback, an N_Vector operation, or another
//! borrow of the same mem. C `erkStep_Access[ARKODE]StepMem` becomes a
//! `step_mem.is_some()` presence check plus `erkStep_mem_mut(...)` at each
//! use site; the `arkode_mem == NULL` half is handled by the type system.
//!
//! The C key table for `erkStep_SetOptions` stores the public
//! `ERKStepSetTableName` setter directly, which receives the raw
//! `void* arkode_mem` forwarded by `sunCheckAndSetCharArgs`. Here the table
//! entry is a small adapter matching `sundials_core::sundials_cli`'s setter
//! fn type: it downcasts the token (an `Option<Box<dyn Any>>` holding an
//! `ARKodeMem` clone) back to the handle and forwards to the real setter —
//! the same shape as `cvode_cli.rs`.

use std::any::Any;

use sundials_core::sundials_adaptcontroller::SUNAdaptController;
use sundials_core::sundials_cli::{sunCheckAndSetCharArgs, sunKeyCharPair};
use sundials_core::sundials_nvector::{N_VScale, N_Vector};
use sundials_core::sundials_types::*;
use sundials_core::sundials_utils::{sunfprintf_long, SUNFile};

use crate::arkode::{
    ARKodeEvolve, ARKodeFree, ARKodeGetDky, ARKodePrintMem, ARKodeReset, ARKodeResize,
    ARKodeSStolerances, ARKodeSVtolerances, ARKodeWFtolerances,
};
use crate::arkode_butcher::{
    ARKodeButcherTable, ARKodeButcherTable_Copy, ARKodeButcherTable_Space,
    ARKodeButcherTable_Write,
};
use crate::arkode_butcher_erk::{
    arkButcherTableERKNameToID, ARKodeButcherTable_LoadERK, ARKODE_ERKTableID,
    ARKODE_MAX_ERK_NUM, ARKODE_MIN_ERK_NUM,
};
use crate::arkode_erkstep::{erkStep_GetOrder, erkStep_RelaxDeltaE, erkStep_mem_mut, MSG_ERKSTEP_NO_MEM};
use crate::arkode_impl::*;
use crate::arkode_io::{
    arkReplaceAdaptController, arkSetAdaptivityFn, arkSetAdaptivityMethod, ARKodeClearStopTime,
    ARKodeGetActualInitStep, ARKodeGetCurrentStep, ARKodeGetCurrentTime, ARKodeGetErrWeights,
    ARKodeGetEstLocalErrors, ARKodeGetLastStep, ARKodeGetNumAccSteps, ARKodeGetNumConstrFails,
    ARKodeGetNumErrTestFails, ARKodeGetNumExpSteps, ARKodeGetNumGEvals, ARKodeGetNumRhsEvals,
    ARKodeGetNumSteps, ARKodeGetNumStepAttempts, ARKodeGetReturnFlagName, ARKodeGetRootInfo,
    ARKodeGetStepStats, ARKodeGetTolScaleFactor, ARKodeGetUserData, ARKodeGetWorkSpace,
    ARKodePrintAllStats, ARKodeSetAdaptController, ARKodeSetAdaptivityAdjustment,
    ARKodeSetCFLFraction, ARKodeSetConstraints, ARKodeSetDefaults, ARKodeSetErrorBias,
    ARKodeSetFixedStep, ARKodeSetFixedStepBounds, ARKodeSetInitStep, ARKodeSetInterpolantDegree,
    ARKodeSetInterpolantType, ARKodeSetInterpolateStopTime, ARKodeSetMaxEFailGrowth,
    ARKodeSetMaxErrTestFails, ARKodeSetMaxFirstGrowth, ARKodeSetMaxGrowth, ARKodeSetMaxHnilWarns,
    ARKodeSetMaxNumConstrFails, ARKodeSetMaxNumSteps, ARKodeSetMaxStep, ARKodeSetMinReduction,
    ARKodeSetMinStep, ARKodeSetNoInactiveRootWarn, ARKodeSetOrder, ARKodeSetPostprocessStageFn,
    ARKodeSetPostprocessStepFn, ARKodeSetRootDirection, ARKodeSetSafetyFactor,
    ARKodeSetSmallNumEFails, ARKodeSetStabilityFn, ARKodeSetStopTime, ARKodeSetUserData,
    ARKodeWriteParameters,
};
use crate::arkode_relaxation::{
    arkRelaxCreate, ARKodeGetNumRelaxBoundFails, ARKodeGetNumRelaxFails, ARKodeGetNumRelaxFnEvals,
    ARKodeGetNumRelaxJacEvals, ARKodeGetNumRelaxSolveFails, ARKodeGetNumRelaxSolveIters,
    ARKodeSetRelaxEtaFail, ARKodeSetRelaxFn, ARKodeSetRelaxLowerBound, ARKodeSetRelaxMaxFails,
    ARKodeSetRelaxMaxIters, ARKodeSetRelaxResTol, ARKodeSetRelaxSolver, ARKodeSetRelaxTol,
    ARKodeSetRelaxUpperBound,
};
use crate::arkode_root::ARKodeRootInit;

/*===============================================================
  Exported optional input functions.
  ===============================================================*/

/*---------------------------------------------------------------
  ERKStepSetTable:

  Specifies to use a customized Butcher table for the explicit
  portion of the system.

  If d==NULL, then the method is automatically flagged as a
  fixed-step method; a user MUST also call either
  ERKStepSetFixedStep or ERKStepSetInitStep to set the desired
  time step size.
  ---------------------------------------------------------------*/
pub fn ERKStepSetTable(arkode_mem: &ARKodeMem, B: &ARKodeButcherTable) -> i32 {
    /* access ARKodeMem and ARKodeERKStepMem structures */
    if arkode_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(arkode_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "ERKStepSetTable",
            file!(),
            MSG_ERKSTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }
    let ark_mem = arkode_mem;

    /* check for legal inputs: the `B == NULL` test is handled by the type
       system */

    /* clear any existing parameters and Butcher tables */
    {
        let mut step_mem = erkStep_mem_mut(ark_mem);
        step_mem.stages = 0;
        step_mem.q = 0;
        step_mem.p = 0;
    }

    let oldB = erkStep_mem_mut(ark_mem).B.clone();
    let mut Bliw: sunindextype = 0;
    let mut Blrw: sunindextype = 0;
    ARKodeButcherTable_Space(oldB.as_ref(), &mut Bliw, &mut Blrw);
    /* C: ARKodeButcherTable_Free(step_mem->B); step_mem->B = NULL; */
    erkStep_mem_mut(ark_mem).B = None;
    drop(oldB);
    {
        let mut m = ark_mem.borrow_mut();
        m.liw -= Bliw;
        m.lrw -= Blrw;
    }

    /* set the relevant parameters */
    let (Bstages, Bq, Bp) = {
        let B = B.borrow();
        (B.stages, B.q, B.p)
    };
    {
        let mut step_mem = erkStep_mem_mut(ark_mem);
        step_mem.stages = Bstages;
        step_mem.q = Bq;
        step_mem.p = Bp;
    }

    /* copy the table into step memory */
    let newB = ARKodeButcherTable_Copy(Some(B));
    if newB.is_none() {
        erkStep_mem_mut(ark_mem).B = None;
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "ERKStepSetTable",
            file!(),
            MSG_ARK_NO_MEM,
        );
        return ARK_MEM_NULL;
    }
    erkStep_mem_mut(ark_mem).B = newB;

    let newB = erkStep_mem_mut(ark_mem).B.clone();
    ARKodeButcherTable_Space(newB.as_ref(), &mut Bliw, &mut Blrw);
    {
        let mut m = ark_mem.borrow_mut();
        m.liw += Bliw;
        m.lrw += Blrw;
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  ERKStepSetTableNum:

  Specifies to use a pre-existing Butcher table for the problem,
  based on the integer flag passed to ARKodeButcherTable_LoadERK()
  within the file arkode_butcher_erk.c.
  ---------------------------------------------------------------*/
pub fn ERKStepSetTableNum(arkode_mem: &ARKodeMem, etable: ARKODE_ERKTableID) -> i32 {
    /* access ARKodeMem and ARKodeERKStepMem structures */
    if arkode_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(arkode_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "ERKStepSetTableNum",
            file!(),
            MSG_ERKSTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }
    let ark_mem = arkode_mem;

    /* check that argument specifies an explicit table */
    if etable < ARKODE_MIN_ERK_NUM || etable > ARKODE_MAX_ERK_NUM {
        /* C reports ARK_MEM_NULL here but returns ARK_ILL_INPUT */
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "ERKStepSetTableNum",
            file!(),
            "Illegal ERK table number",
        );
        return ARK_ILL_INPUT;
    }

    /* clear any existing parameters and Butcher tables */
    {
        let mut step_mem = erkStep_mem_mut(ark_mem);
        step_mem.stages = 0;
        step_mem.q = 0;
        step_mem.p = 0;
    }

    let oldB = erkStep_mem_mut(ark_mem).B.clone();
    let mut Bliw: sunindextype = 0;
    let mut Blrw: sunindextype = 0;
    ARKodeButcherTable_Space(oldB.as_ref(), &mut Bliw, &mut Blrw);
    /* C: ARKodeButcherTable_Free(step_mem->B); step_mem->B = NULL; */
    erkStep_mem_mut(ark_mem).B = None;
    drop(oldB);
    {
        let mut m = ark_mem.borrow_mut();
        m.liw -= Bliw;
        m.lrw -= Blrw;
    }

    /* fill in table based on argument */
    let newB = ARKodeButcherTable_LoadERK(etable);
    if newB.is_none() {
        erkStep_mem_mut(ark_mem).B = None;
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "ERKStepSetTableNum",
            file!(),
            "Error setting table with that index",
        );
        return ARK_ILL_INPUT;
    }
    erkStep_mem_mut(ark_mem).B = newB;

    let newB = erkStep_mem_mut(ark_mem).B.clone().expect("Butcher table set");
    let (Bstages, Bq, Bp) = {
        let B = newB.borrow();
        (B.stages, B.q, B.p)
    };
    {
        let mut step_mem = erkStep_mem_mut(ark_mem);
        step_mem.stages = Bstages;
        step_mem.q = Bq;
        step_mem.p = Bp;
    }

    ARKodeButcherTable_Space(Some(&newB), &mut Bliw, &mut Blrw);
    {
        let mut m = ark_mem.borrow_mut();
        m.liw += Bliw;
        m.lrw += Blrw;
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  ERKStepSetTableName:

  Specifies to use a pre-existing Butcher table for the problem,
  based on the string passed to ARKodeButcherTable_LoadERKByNmae()
  within the file arkode_butcher_erk.c.
  ---------------------------------------------------------------*/
pub fn ERKStepSetTableName(arkode_mem: &ARKodeMem, etable: &str) -> i32 {
    ERKStepSetTableNum(arkode_mem, arkButcherTableERKNameToID(etable))
}

/*===============================================================
  Exported optional output functions.
  ===============================================================*/

/*---------------------------------------------------------------
  erkStep_GetNumRhsEvals:

  Returns the current number of RHS calls
  ---------------------------------------------------------------*/
pub fn erkStep_GetNumRhsEvals(
    ark_mem: &ARKodeMem,
    partition_index: i32,
    rhs_evals: &mut i64,
) -> i32 {
    /* access ARKodeERKStepMem structure */
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "erkStep_GetNumRhsEvals",
            file!(),
            MSG_ERKSTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    /* the `rhs_evals == NULL` check is handled by the type system */

    if partition_index > 0 {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "erkStep_GetNumRhsEvals",
            file!(),
            "Invalid partition index",
        );
        return ARK_ILL_INPUT;
    }

    *rhs_evals = erkStep_mem_mut(ark_mem).nfe;

    ARK_SUCCESS
}

pub fn ERKStepGetNumRhsEvals(arkode_mem: &ARKodeMem, fevals: &mut i64) -> i32 {
    ARKodeGetNumRhsEvals(arkode_mem, 0, fevals)
}

/*---------------------------------------------------------------
  ERKStepGetCurrentButcherTable:

  Sets pointers to the Butcher table currently in use.
  ---------------------------------------------------------------*/
pub fn ERKStepGetCurrentButcherTable(
    arkode_mem: &ARKodeMem,
    B: &mut Option<ARKodeButcherTable>,
) -> i32 {
    /* access ARKodeMem and ARKodeERKStepMem structures */
    if arkode_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(arkode_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "ERKStepGetCurrentButcherTable",
            file!(),
            MSG_ERKSTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    /* get tables from step_mem */
    *B = erkStep_mem_mut(arkode_mem).B.clone();
    ARK_SUCCESS
}

/*---------------------------------------------------------------
  ERKStepGetTimestepperStats:

  Returns integrator statistics
  ---------------------------------------------------------------*/
pub fn ERKStepGetTimestepperStats(
    arkode_mem: &ARKodeMem,
    expsteps: &mut i64,
    accsteps: &mut i64,
    attempts: &mut i64,
    fevals: &mut i64,
    netfails: &mut i64,
) -> i32 {
    /* access ARKodeMem and ARKodeERKStepMem structures */
    if arkode_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(arkode_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "ERKStepGetTimestepperStats",
            file!(),
            MSG_ERKSTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }
    let ark_mem = arkode_mem;

    /* set expsteps and accsteps from adaptivity structure */
    {
        let m = ark_mem.borrow();
        let hadapt_mem = m.hadapt_mem.as_ref().expect("hadapt_mem set");
        *expsteps = hadapt_mem.nst_exp;
        *accsteps = hadapt_mem.nst_acc;

        /* set remaining outputs */
        *attempts = m.nst_attempts;
        *netfails = m.netf;
    }
    *fevals = erkStep_mem_mut(ark_mem).nfe;

    ARK_SUCCESS
}

/*===============================================================
  Private functions attached to ARKODE
  ===============================================================*/

/// Recover the `ARKodeMem` handle from a CLI token (C: the raw
/// `void* arkode_mem` passed through `sunCheckAndSet*Args`). A
/// missing/mistyped token corresponds to C passing a garbage pointer to the
/// setter (UB) and maps to a deterministic panic.
fn cliARKodeMem(mem: &mut Option<Box<dyn Any>>) -> ARKodeMem {
    mem.as_mut()
        .and_then(|b| b.downcast_ref::<ARKodeMem>())
        .cloned()
        .expect("arkode_mem token")
}

fn cliERKStepSetTableName(mem: &mut Option<Box<dyn Any>>, arg: &str) -> i32 {
    let ark_mem = cliARKodeMem(mem);
    ERKStepSetTableName(&ark_mem, arg)
}

/*---------------------------------------------------------------
  erkStep_SetOption:

  Provides string-based control over ERKStep-specific "set" routines.
  ---------------------------------------------------------------*/
pub fn erkStep_SetOptions(
    ark_mem: &ARKodeMem,
    argidx: &mut i32,
    argv: &[String],
    offset: usize,
    arg_used: &mut sunbooleantype,
) -> i32 {
    /* Set lists of keys, and the corresponding set routines */
    static char_pairs: [sunKeyCharPair; 1] = [sunKeyCharPair {
        key: "table_name",
        set: cliERKStepSetTableName,
    }];
    let num_char_keys: i32 = char_pairs.len() as i32;

    /* the CLI helpers receive C's `void* arkode_mem` as a boxed handle clone */
    let mut mem: Option<Box<dyn Any>> = Some(Box::new(ark_mem.clone()));

    /* check all "char" keys */
    let mut j: i32 = 0;
    let retval = sunCheckAndSetCharArgs(
        &mut mem,
        argidx,
        argv,
        offset,
        &char_pairs,
        num_char_keys,
        arg_used,
        &mut j,
    );
    if retval != ARK_SUCCESS {
        arkProcessError(
            Some(ark_mem),
            retval,
            line!() as i32,
            "erkStep_SetOptions",
            file!(),
            &format!("error setting key: {}", char_pairs[j as usize].key),
        );
        return retval;
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  erkStep_SetRelaxFn:

  Sets up the relaxation module using ERKStep's utility routines.
  ---------------------------------------------------------------*/
pub fn erkStep_SetRelaxFn(
    ark_mem: &ARKodeMem,
    rfn: Option<ARKRelaxFn>,
    rjac: Option<ARKRelaxJacFn>,
) -> i32 {
    arkRelaxCreate(
        ark_mem,
        rfn,
        rjac,
        Some(erkStep_RelaxDeltaE),
        Some(erkStep_GetOrder),
    )
}

/*---------------------------------------------------------------
  erkStep_SetDefaults:

  Resets all ERKStep optional inputs to their default values.
  Does not change problem-defining function pointers or
  user_data pointer.
  ---------------------------------------------------------------*/
pub fn erkStep_SetDefaults(ark_mem: &ARKodeMem) -> i32 {
    /* access ARKodeERKStepMem structure */
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "erkStep_SetDefaults",
            file!(),
            MSG_ERKSTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    /* Set default values for integrator optional inputs */
    {
        let mut step_mem = erkStep_mem_mut(ark_mem);
        step_mem.q = Q_DEFAULT; /* method order */
        step_mem.p = 0; /* embedding order */
        step_mem.stages = 0; /* no stages */
    }
    {
        let mut m = ark_mem.borrow_mut();
        let hadapt_mem = m.hadapt_mem.as_mut().expect("hadapt_mem set");
        hadapt_mem.etamxf = 0.3; /* max change on error-failed step */
        hadapt_mem.safety = 0.99; /* step adaptivity safety factor  */
        hadapt_mem.growth = 25.0; /* step adaptivity growth factor */
    }

    /* Remove pre-existing Butcher table */
    let oldB = erkStep_mem_mut(ark_mem).B.clone();
    if oldB.is_some() {
        let mut Bliw: sunindextype = 0;
        let mut Blrw: sunindextype = 0;
        ARKodeButcherTable_Space(oldB.as_ref(), &mut Bliw, &mut Blrw);
        {
            let mut m = ark_mem.borrow_mut();
            m.liw -= Bliw;
            m.lrw -= Blrw;
        }
        /* C: ARKodeButcherTable_Free(step_mem->B) */
    }
    erkStep_mem_mut(ark_mem).B = None;
    drop(oldB);

    /* Load the default SUNAdaptController */
    let retval = arkReplaceAdaptController(ark_mem, None, SUNTRUE);
    if retval != 0 {
        return retval;
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  erkStep_SetOrder:

  Specifies the method order
  ---------------------------------------------------------------*/
pub fn erkStep_SetOrder(ark_mem: &ARKodeMem, ord: i32) -> i32 {
    /* access ARKodeERKStepMem structure */
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "erkStep_SetOrder",
            file!(),
            MSG_ERKSTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    /* set user-provided value, or default, depending on argument */
    if ord <= 0 {
        erkStep_mem_mut(ark_mem).q = Q_DEFAULT;
    } else {
        erkStep_mem_mut(ark_mem).q = ord;
    }

    /* clear Butcher tables, since user is requesting a change in method
       or a reset to defaults.  Tables will be set in ARKInitialSetup. */
    {
        let mut step_mem = erkStep_mem_mut(ark_mem);
        step_mem.stages = 0;
        step_mem.p = 0;
    }

    let oldB = erkStep_mem_mut(ark_mem).B.clone();
    let mut Bliw: sunindextype = 0;
    let mut Blrw: sunindextype = 0;
    ARKodeButcherTable_Space(oldB.as_ref(), &mut Bliw, &mut Blrw);
    /* C: ARKodeButcherTable_Free(step_mem->B); step_mem->B = NULL; */
    erkStep_mem_mut(ark_mem).B = None;
    drop(oldB);
    {
        let mut m = ark_mem.borrow_mut();
        m.liw -= Bliw;
        m.lrw -= Blrw;
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  erkStep_GetEstLocalErrors: Returns the current local truncation
  error estimate vector
  ---------------------------------------------------------------*/
pub fn erkStep_GetEstLocalErrors(ark_mem: &ARKodeMem, ele: &N_Vector) -> i32 {
    /* access ARKodeERKStepMem structure */
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "erkStep_GetEstLocalErrors",
            file!(),
            MSG_ERKSTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    /* return an error if local truncation error is not computed */
    let (fixedstep, AccumErrorType) = {
        let m = ark_mem.borrow();
        (m.fixedstep, m.AccumErrorType)
    };
    let p = erkStep_mem_mut(ark_mem).p;
    if (fixedstep && (AccumErrorType == ARK_ACCUMERROR_NONE)) || (p <= 0) {
        return ARK_STEPPER_UNSUPPORTED;
    }

    /* otherwise, copy local truncation error vector to output */
    let tempv1 = ark_mem.borrow().tempv1.clone().expect("tempv1 set");
    N_VScale(ONE, &tempv1, ele);
    ARK_SUCCESS
}

/*---------------------------------------------------------------
  erkStep_GetStageIndex:

  Returns the current stage index and number of stages
  ---------------------------------------------------------------*/
pub fn erkStep_GetStageIndex(ark_mem: &ARKodeMem, stage: &mut i32, max_stages: &mut i32) -> i32 {
    /* access ARKodeARKStepMem structure */
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "erkStep_GetStageIndex",
            file!(),
            MSG_ERKSTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    let step_mem = erkStep_mem_mut(ark_mem);
    *stage = step_mem.istage;
    *max_stages = step_mem.stages;

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  erkStep_PrintAllStats:

  Prints integrator statistics
  ---------------------------------------------------------------*/
pub fn erkStep_PrintAllStats(
    ark_mem: &ARKodeMem,
    outfile: &SUNFile,
    fmt: SUNOutputFormat,
) -> i32 {
    /* access ARKodeERKStepMem structure */
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "erkStep_PrintAllStats",
            file!(),
            MSG_ERKSTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    let nfe = erkStep_mem_mut(ark_mem).nfe;
    sunfprintf_long(outfile, fmt, SUNFALSE, "RHS fn evals", nfe);

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  erkStep_WriteParameters:

  Outputs all solver parameters to the provided file pointer.
  ---------------------------------------------------------------*/
pub fn erkStep_WriteParameters(ark_mem: &ARKodeMem, fp: &SUNFile) -> i32 {
    /* access ARKodeERKStepMem structure */
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "erkStep_WriteParameters",
            file!(),
            MSG_ERKSTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    let q = erkStep_mem_mut(ark_mem).q;

    /* print integrator parameters to file */
    fp.write_str("ERKStep time step module parameters:\n");
    fp.write_str(&format!("  Method order {q}\n"));
    fp.write_str("\n");

    ARK_SUCCESS
}

/*===============================================================
  Exported-but-deprecated user-callable functions.
  ===============================================================*/

pub fn ERKStepResize(
    arkode_mem: &ARKodeMem,
    y0: &N_Vector,
    hscale: sunrealtype,
    t0: sunrealtype,
    resize: Option<ARKVecResizeFn>,
    resize_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    ARKodeResize(arkode_mem, y0, hscale, t0, resize, resize_data)
}

pub fn ERKStepReset(arkode_mem: &ARKodeMem, tR: sunrealtype, yR: &N_Vector) -> i32 {
    ARKodeReset(arkode_mem, tR, yR)
}

pub fn ERKStepSStolerances(
    arkode_mem: &ARKodeMem,
    reltol: sunrealtype,
    abstol: sunrealtype,
) -> i32 {
    ARKodeSStolerances(arkode_mem, reltol, abstol)
}

pub fn ERKStepSVtolerances(
    arkode_mem: &ARKodeMem,
    reltol: sunrealtype,
    abstol: &N_Vector,
) -> i32 {
    ARKodeSVtolerances(arkode_mem, reltol, abstol)
}

pub fn ERKStepWFtolerances(arkode_mem: &ARKodeMem, efun: ARKEwtFn) -> i32 {
    ARKodeWFtolerances(arkode_mem, efun)
}

pub fn ERKStepRootInit(arkode_mem: &ARKodeMem, nrtfn: i32, g: Option<ARKRootFn>) -> i32 {
    ARKodeRootInit(arkode_mem, nrtfn, g)
}

pub fn ERKStepSetDefaults(arkode_mem: &ARKodeMem) -> i32 {
    ARKodeSetDefaults(arkode_mem)
}

pub fn ERKStepSetOrder(arkode_mem: &ARKodeMem, ord: i32) -> i32 {
    ARKodeSetOrder(arkode_mem, ord)
}

pub fn ERKStepSetInterpolantType(arkode_mem: &ARKodeMem, itype: i32) -> i32 {
    ARKodeSetInterpolantType(arkode_mem, itype)
}

pub fn ERKStepSetInterpolantDegree(arkode_mem: &ARKodeMem, degree: i32) -> i32 {
    ARKodeSetInterpolantDegree(arkode_mem, degree)
}

pub fn ERKStepSetDenseOrder(arkode_mem: &ARKodeMem, dord: i32) -> i32 {
    ARKodeSetInterpolantDegree(arkode_mem, dord)
}

pub fn ERKStepSetAdaptController(arkode_mem: &ARKodeMem, C: Option<&SUNAdaptController>) -> i32 {
    ARKodeSetAdaptController(arkode_mem, C)
}

pub fn ERKStepSetAdaptivityAdjustment(arkode_mem: &ARKodeMem, adjust: i32) -> i32 {
    ARKodeSetAdaptivityAdjustment(arkode_mem, adjust)
}

pub fn ERKStepSetCFLFraction(arkode_mem: &ARKodeMem, cfl_frac: sunrealtype) -> i32 {
    ARKodeSetCFLFraction(arkode_mem, cfl_frac)
}

pub fn ERKStepSetSafetyFactor(arkode_mem: &ARKodeMem, safety: sunrealtype) -> i32 {
    ARKodeSetSafetyFactor(arkode_mem, safety)
}

pub fn ERKStepSetErrorBias(arkode_mem: &ARKodeMem, bias: sunrealtype) -> i32 {
    ARKodeSetErrorBias(arkode_mem, bias)
}

pub fn ERKStepSetMaxGrowth(arkode_mem: &ARKodeMem, mx_growth: sunrealtype) -> i32 {
    ARKodeSetMaxGrowth(arkode_mem, mx_growth)
}

pub fn ERKStepSetMinReduction(arkode_mem: &ARKodeMem, eta_min: sunrealtype) -> i32 {
    ARKodeSetMinReduction(arkode_mem, eta_min)
}

pub fn ERKStepSetFixedStepBounds(
    arkode_mem: &ARKodeMem,
    lb: sunrealtype,
    ub: sunrealtype,
) -> i32 {
    ARKodeSetFixedStepBounds(arkode_mem, lb, ub)
}

pub fn ERKStepSetAdaptivityMethod(
    arkode_mem: &ARKodeMem,
    imethod: i32,
    idefault: i32,
    pq: i32,
    adapt_params: Option<&[sunrealtype; 3]>,
) -> i32 {
    arkSetAdaptivityMethod(arkode_mem, imethod, idefault, pq, adapt_params)
}

pub fn ERKStepSetAdaptivityFn(
    arkode_mem: &ARKodeMem,
    hfun: Option<ARKAdaptFn>,
    h_data: Option<Box<dyn Any>>,
) -> i32 {
    arkSetAdaptivityFn(arkode_mem, hfun, h_data)
}

pub fn ERKStepSetMaxFirstGrowth(arkode_mem: &ARKodeMem, etamx1: sunrealtype) -> i32 {
    ARKodeSetMaxFirstGrowth(arkode_mem, etamx1)
}

pub fn ERKStepSetMaxEFailGrowth(arkode_mem: &ARKodeMem, etamxf: sunrealtype) -> i32 {
    ARKodeSetMaxEFailGrowth(arkode_mem, etamxf)
}

pub fn ERKStepSetSmallNumEFails(arkode_mem: &ARKodeMem, small_nef: i32) -> i32 {
    ARKodeSetSmallNumEFails(arkode_mem, small_nef)
}

pub fn ERKStepSetStabilityFn(
    arkode_mem: &ARKodeMem,
    EStab: Option<ARKExpStabFn>,
    estab_data: Option<Box<dyn Any>>,
) -> i32 {
    ARKodeSetStabilityFn(arkode_mem, EStab, estab_data)
}

pub fn ERKStepSetMaxErrTestFails(arkode_mem: &ARKodeMem, maxnef: i32) -> i32 {
    ARKodeSetMaxErrTestFails(arkode_mem, maxnef)
}

pub fn ERKStepSetConstraints(arkode_mem: &ARKodeMem, constraints: Option<&N_Vector>) -> i32 {
    ARKodeSetConstraints(arkode_mem, constraints)
}

pub fn ERKStepSetMaxNumSteps(arkode_mem: &ARKodeMem, mxsteps: i64) -> i32 {
    ARKodeSetMaxNumSteps(arkode_mem, mxsteps)
}

pub fn ERKStepSetMaxHnilWarns(arkode_mem: &ARKodeMem, mxhnil: i32) -> i32 {
    ARKodeSetMaxHnilWarns(arkode_mem, mxhnil)
}

pub fn ERKStepSetInitStep(arkode_mem: &ARKodeMem, hin: sunrealtype) -> i32 {
    ARKodeSetInitStep(arkode_mem, hin)
}

pub fn ERKStepSetMinStep(arkode_mem: &ARKodeMem, hmin: sunrealtype) -> i32 {
    ARKodeSetMinStep(arkode_mem, hmin)
}

pub fn ERKStepSetMaxStep(arkode_mem: &ARKodeMem, hmax: sunrealtype) -> i32 {
    ARKodeSetMaxStep(arkode_mem, hmax)
}

pub fn ERKStepSetInterpolateStopTime(arkode_mem: &ARKodeMem, interp: sunbooleantype) -> i32 {
    ARKodeSetInterpolateStopTime(arkode_mem, interp)
}

pub fn ERKStepSetStopTime(arkode_mem: &ARKodeMem, tstop: sunrealtype) -> i32 {
    ARKodeSetStopTime(arkode_mem, tstop)
}

pub fn ERKStepClearStopTime(arkode_mem: &ARKodeMem) -> i32 {
    ARKodeClearStopTime(arkode_mem)
}

pub fn ERKStepSetFixedStep(arkode_mem: &ARKodeMem, hfixed: sunrealtype) -> i32 {
    ARKodeSetFixedStep(arkode_mem, hfixed)
}

pub fn ERKStepSetMaxNumConstrFails(arkode_mem: &ARKodeMem, maxfails: i32) -> i32 {
    ARKodeSetMaxNumConstrFails(arkode_mem, maxfails)
}

pub fn ERKStepSetRootDirection(arkode_mem: &ARKodeMem, rootdir: &[i32]) -> i32 {
    ARKodeSetRootDirection(arkode_mem, rootdir)
}

pub fn ERKStepSetNoInactiveRootWarn(arkode_mem: &ARKodeMem) -> i32 {
    ARKodeSetNoInactiveRootWarn(arkode_mem)
}

pub fn ERKStepSetUserData(arkode_mem: &ARKodeMem, user_data: Option<Box<dyn Any>>) -> i32 {
    ARKodeSetUserData(arkode_mem, user_data)
}

pub fn ERKStepSetPostprocessStepFn(
    arkode_mem: &ARKodeMem,
    ProcessStep: Option<ARKPostProcessFn>,
) -> i32 {
    ARKodeSetPostprocessStepFn(arkode_mem, ProcessStep)
}

pub fn ERKStepSetPostprocessStageFn(
    arkode_mem: &ARKodeMem,
    ProcessStage: Option<ARKPostProcessFn>,
) -> i32 {
    ARKodeSetPostprocessStageFn(arkode_mem, ProcessStage)
}

pub fn ERKStepEvolve(
    arkode_mem: &ARKodeMem,
    tout: sunrealtype,
    yout: &N_Vector,
    tret: &mut sunrealtype,
    itask: i32,
) -> i32 {
    ARKodeEvolve(arkode_mem, tout, yout, tret, itask)
}

pub fn ERKStepGetDky(arkode_mem: &ARKodeMem, t: sunrealtype, k: i32, dky: &N_Vector) -> i32 {
    ARKodeGetDky(arkode_mem, t, k, dky)
}

pub fn ERKStepGetNumExpSteps(arkode_mem: &ARKodeMem, nsteps: &mut i64) -> i32 {
    ARKodeGetNumExpSteps(arkode_mem, nsteps)
}

pub fn ERKStepGetNumAccSteps(arkode_mem: &ARKodeMem, nsteps: &mut i64) -> i32 {
    ARKodeGetNumAccSteps(arkode_mem, nsteps)
}

pub fn ERKStepGetNumStepAttempts(arkode_mem: &ARKodeMem, nstep_attempts: &mut i64) -> i32 {
    ARKodeGetNumStepAttempts(arkode_mem, nstep_attempts)
}

pub fn ERKStepGetNumErrTestFails(arkode_mem: &ARKodeMem, netfails: &mut i64) -> i32 {
    ARKodeGetNumErrTestFails(arkode_mem, netfails)
}

pub fn ERKStepGetEstLocalErrors(arkode_mem: &ARKodeMem, ele: &N_Vector) -> i32 {
    ARKodeGetEstLocalErrors(arkode_mem, ele)
}

pub fn ERKStepGetWorkSpace(arkode_mem: &ARKodeMem, lenrw: &mut i64, leniw: &mut i64) -> i32 {
    ARKodeGetWorkSpace(arkode_mem, lenrw, leniw)
}

pub fn ERKStepGetNumSteps(arkode_mem: &ARKodeMem, nsteps: &mut i64) -> i32 {
    ARKodeGetNumSteps(arkode_mem, nsteps)
}

pub fn ERKStepGetActualInitStep(arkode_mem: &ARKodeMem, hinused: &mut sunrealtype) -> i32 {
    ARKodeGetActualInitStep(arkode_mem, hinused)
}

pub fn ERKStepGetLastStep(arkode_mem: &ARKodeMem, hlast: &mut sunrealtype) -> i32 {
    ARKodeGetLastStep(arkode_mem, hlast)
}

pub fn ERKStepGetCurrentStep(arkode_mem: &ARKodeMem, hcur: &mut sunrealtype) -> i32 {
    ARKodeGetCurrentStep(arkode_mem, hcur)
}

pub fn ERKStepGetCurrentTime(arkode_mem: &ARKodeMem, tcur: &mut sunrealtype) -> i32 {
    ARKodeGetCurrentTime(arkode_mem, tcur)
}

pub fn ERKStepGetTolScaleFactor(arkode_mem: &ARKodeMem, tolsfact: &mut sunrealtype) -> i32 {
    ARKodeGetTolScaleFactor(arkode_mem, tolsfact)
}

pub fn ERKStepGetErrWeights(arkode_mem: &ARKodeMem, eweight: &N_Vector) -> i32 {
    ARKodeGetErrWeights(arkode_mem, eweight)
}

pub fn ERKStepGetNumGEvals(arkode_mem: &ARKodeMem, ngevals: &mut i64) -> i32 {
    ARKodeGetNumGEvals(arkode_mem, ngevals)
}

pub fn ERKStepGetRootInfo(arkode_mem: &ARKodeMem, rootsfound: &mut [i32]) -> i32 {
    ARKodeGetRootInfo(arkode_mem, rootsfound)
}

pub fn ERKStepGetNumConstrFails(arkode_mem: &ARKodeMem, nconstrfails: &mut i64) -> i32 {
    ARKodeGetNumConstrFails(arkode_mem, nconstrfails)
}

pub fn ERKStepGetUserData(
    arkode_mem: &ARKodeMem,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    ARKodeGetUserData(arkode_mem, user_data)
}

pub fn ERKStepPrintAllStats(
    arkode_mem: &ARKodeMem,
    outfile: &SUNFile,
    fmt: SUNOutputFormat,
) -> i32 {
    ARKodePrintAllStats(arkode_mem, outfile, fmt)
}

pub fn ERKStepGetReturnFlagName(flag: i64) -> String {
    ARKodeGetReturnFlagName(flag)
}

pub fn ERKStepWriteParameters(arkode_mem: &ARKodeMem, fp: &SUNFile) -> i32 {
    ARKodeWriteParameters(arkode_mem, fp)
}

pub fn ERKStepWriteButcher(arkode_mem: &ARKodeMem, fp: &SUNFile) -> i32 {
    /* access ARKodeMem and ARKodeERKStepMem structures */
    if arkode_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(arkode_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "ERKStepWriteButcher",
            file!(),
            MSG_ERKSTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }
    let ark_mem = arkode_mem;

    /* check that Butcher table is non-NULL (otherwise report error) */
    let (B, stages) = {
        let step_mem = erkStep_mem_mut(ark_mem);
        (step_mem.B.clone(), step_mem.stages)
    };
    if B.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "ERKStepWriteButcher",
            file!(),
            "Butcher table memory is NULL",
        );
        return ARK_MEM_NULL;
    }

    /* print Butcher table to file */
    fp.write_str(&format!("\nERKStep Butcher table (stages = {stages}):\n"));
    ARKodeButcherTable_Write(B.as_ref(), fp);
    fp.write_str("\n");

    ARK_SUCCESS
}

pub fn ERKStepGetStepStats(
    arkode_mem: &ARKodeMem,
    nsteps: &mut i64,
    hinused: &mut sunrealtype,
    hlast: &mut sunrealtype,
    hcur: &mut sunrealtype,
    tcur: &mut sunrealtype,
) -> i32 {
    ARKodeGetStepStats(arkode_mem, nsteps, hinused, hlast, hcur, tcur)
}

pub fn ERKStepFree(arkode_mem: &mut Option<ARKodeMem>) {
    ARKodeFree(arkode_mem)
}

pub fn ERKStepPrintMem(arkode_mem: &ARKodeMem, outfile: &SUNFile) {
    ARKodePrintMem(arkode_mem, outfile)
}

pub fn ERKStepSetRelaxFn(
    arkode_mem: &ARKodeMem,
    rfn: Option<ARKRelaxFn>,
    rjac: Option<ARKRelaxJacFn>,
) -> i32 {
    ARKodeSetRelaxFn(arkode_mem, rfn, rjac)
}

pub fn ERKStepSetRelaxEtaFail(arkode_mem: &ARKodeMem, eta_rf: sunrealtype) -> i32 {
    ARKodeSetRelaxEtaFail(arkode_mem, eta_rf)
}

pub fn ERKStepSetRelaxLowerBound(arkode_mem: &ARKodeMem, lower: sunrealtype) -> i32 {
    ARKodeSetRelaxLowerBound(arkode_mem, lower)
}

pub fn ERKStepSetRelaxMaxFails(arkode_mem: &ARKodeMem, max_fails: i32) -> i32 {
    ARKodeSetRelaxMaxFails(arkode_mem, max_fails)
}

pub fn ERKStepSetRelaxMaxIters(arkode_mem: &ARKodeMem, max_iters: i32) -> i32 {
    ARKodeSetRelaxMaxIters(arkode_mem, max_iters)
}

pub fn ERKStepSetRelaxSolver(arkode_mem: &ARKodeMem, solver: ARKRelaxSolver) -> i32 {
    ARKodeSetRelaxSolver(arkode_mem, solver)
}

pub fn ERKStepSetRelaxResTol(arkode_mem: &ARKodeMem, res_tol: sunrealtype) -> i32 {
    ARKodeSetRelaxResTol(arkode_mem, res_tol)
}

pub fn ERKStepSetRelaxTol(
    arkode_mem: &ARKodeMem,
    rel_tol: sunrealtype,
    abs_tol: sunrealtype,
) -> i32 {
    ARKodeSetRelaxTol(arkode_mem, rel_tol, abs_tol)
}

pub fn ERKStepSetRelaxUpperBound(arkode_mem: &ARKodeMem, upper: sunrealtype) -> i32 {
    ARKodeSetRelaxUpperBound(arkode_mem, upper)
}

pub fn ERKStepGetNumRelaxFnEvals(arkode_mem: &ARKodeMem, r_evals: &mut i64) -> i32 {
    ARKodeGetNumRelaxFnEvals(arkode_mem, r_evals)
}

pub fn ERKStepGetNumRelaxJacEvals(arkode_mem: &ARKodeMem, J_evals: &mut i64) -> i32 {
    ARKodeGetNumRelaxJacEvals(arkode_mem, J_evals)
}

pub fn ERKStepGetNumRelaxFails(arkode_mem: &ARKodeMem, relax_fails: &mut i64) -> i32 {
    ARKodeGetNumRelaxFails(arkode_mem, relax_fails)
}

pub fn ERKStepGetNumRelaxBoundFails(arkode_mem: &ARKodeMem, fails: &mut i64) -> i32 {
    ARKodeGetNumRelaxBoundFails(arkode_mem, fails)
}

pub fn ERKStepGetNumRelaxSolveFails(arkode_mem: &ARKodeMem, fails: &mut i64) -> i32 {
    ARKodeGetNumRelaxSolveFails(arkode_mem, fails)
}

pub fn ERKStepGetNumRelaxSolveIters(arkode_mem: &ARKodeMem, iters: &mut i64) -> i32 {
    ARKodeGetNumRelaxSolveIters(arkode_mem, iters)
}

/*===============================================================
  EOF
  ===============================================================*/
