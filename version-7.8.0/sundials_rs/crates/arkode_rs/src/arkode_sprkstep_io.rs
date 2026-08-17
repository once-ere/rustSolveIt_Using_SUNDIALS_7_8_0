//! Port of `src/arkode/arkode_sprkstep_io.c` (optional input/output
//! functions for the ARKODE SPRKStep time stepper).
//!
//! Split exactly as upstream: the user-callable `SPRKStep*` setters and
//! getters, the private `sprkStep_*` routines attached to the ARKODE
//! `step_*` table, and the block of deprecated `SPRKStep*` wrappers that
//! forward to the shared `ARKode*` routines.
//!
//! Borrow discipline is the same as `arkode_sprkstep.rs`: the
//! `sprkStep_mem_mut` guard is a borrow of `ark_mem` and is never held
//! across `arkProcessError`, an `N_Vector` op, or a core `ark*` call.
//!
//! CLI: `sprkStep_SetOptions`'s key table holds
//! `sundials_core::sundials_cli` setter adapters. C hands the raw
//! `void* ark_mem` through `sunCheckAndSetCharArgs`; here the token is
//! an `Option<Box<dyn Any>>` holding an `ARKodeMem` clone, which the
//! adapter downcasts back to the handle (the locked `cvode_cli`
//! pattern).

use std::any::Any;

use sundials_core::sundials_cli::{sunCheckAndSetCharArgs, sunKeyCharPair};
use sundials_core::sundials_nvector::{N_VConst, N_Vector};
use sundials_core::sundials_types::*;
use sundials_core::sundials_utils::{sunfprintf_long, SUNFile};

use crate::arkode::{arkAllocVec, ARKodeEvolve, ARKodeFree, ARKodeGetDky, ARKodeReset};
use crate::arkode_impl::*;
use crate::arkode_io::{
    ARKodeGetCurrentState, ARKodeGetCurrentStep, ARKodeGetCurrentTime, ARKodeGetLastStep,
    ARKodeGetNumRhsEvals, ARKodeGetNumStepAttempts, ARKodeGetNumSteps, ARKodeGetReturnFlagName,
    ARKodeGetRootInfo, ARKodeGetStepStats, ARKodeGetUserData, ARKodePrintAllStats,
    ARKodeSetDefaults, ARKodeSetFixedStep, ARKodeSetInterpolantDegree, ARKodeSetInterpolantType,
    ARKodeSetMaxNumSteps, ARKodeSetNoInactiveRootWarn, ARKodeSetOrder,
    ARKodeSetPostprocessStageFn, ARKodeSetPostprocessStepFn, ARKodeSetRootDirection,
    ARKodeSetStopTime, ARKodeSetUserData, ARKodeWriteParameters,
};
use crate::arkode_root::ARKodeRootInit;
use crate::arkode_sprk::{
    ARKodeSPRKTable, ARKodeSPRKTable_Copy, ARKodeSPRKTable_Free, ARKodeSPRKTable_LoadByName,
};
use crate::arkode_sprkstep::{
    sprkStep_AccessARKODEStepMem, sprkStep_AccessStepMem, sprkStep_mem_mut, sprkStep_TakeStep,
    sprkStep_TakeStep_Compensated,
};

/*===============================================================
  Exported optional input functions.
  ===============================================================*/

/*---------------------------------------------------------------
  SPRKStepSetUseCompensatedSums:

  Turns on/off compensated summation in SPRKStep and ARKODE.
  ---------------------------------------------------------------*/
pub fn SPRKStepSetUseCompensatedSums(arkode_mem: &ARKodeMem, onoff: sunbooleantype) -> i32 {
    /* access ARKodeMem and ARKodeSPRKStepMem structures */
    let retval = sprkStep_AccessARKODEStepMem(arkode_mem, "SPRKStepSetUseCompensatedSums");
    if retval != ARK_SUCCESS {
        return retval;
    }
    let ark_mem = arkode_mem;

    if onoff {
        ark_mem.borrow_mut().use_compensated_sums = SUNTRUE;
    } else {
        ark_mem.borrow_mut().use_compensated_sums = SUNFALSE;
    }

    let retval = sprkStep_SetUseCompensatedSums(arkode_mem, onoff);

    retval
}

/*---------------------------------------------------------------
  SPRKStepSetMethod:

  Specifies the SPRK method

  ** Note in documentation that this should not be called along
  with ARKodeSetOrder. **
  ---------------------------------------------------------------*/
pub fn SPRKStepSetMethod(arkode_mem: &ARKodeMem, sprk_storage: &ARKodeSPRKTable) -> i32 {
    /* access ARKodeMem and ARKodeSPRKStepMem structures */
    let retval = sprkStep_AccessARKODEStepMem(arkode_mem, "SPRKStepSetMethod");
    if retval != ARK_SUCCESS {
        return retval;
    }
    let ark_mem = arkode_mem;

    if sprkStep_mem_mut(ark_mem).method.is_some() {
        let method = sprkStep_mem_mut(ark_mem).method.take();
        ARKodeSPRKTable_Free(method);
        sprkStep_mem_mut(ark_mem).method = None;
    }

    let method = ARKodeSPRKTable_Copy(sprk_storage);
    sprkStep_mem_mut(ark_mem).method = method;

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  SPRKStepSetMethodName:

  Specifies the SPRK method.
  ---------------------------------------------------------------*/
pub fn SPRKStepSetMethodName(arkode_mem: &ARKodeMem, method: &str) -> i32 {
    /* access ARKodeMem and ARKodeSPRKStepMem structures */
    let retval = sprkStep_AccessARKODEStepMem(arkode_mem, "SPRKStepSetMethodName");
    if retval != ARK_SUCCESS {
        return retval;
    }
    let ark_mem = arkode_mem;

    if sprkStep_mem_mut(ark_mem).method.is_some() {
        let old = sprkStep_mem_mut(ark_mem).method.take();
        ARKodeSPRKTable_Free(old);
        sprkStep_mem_mut(ark_mem).method = None;
    }

    let loaded = ARKodeSPRKTable_LoadByName(method);
    sprkStep_mem_mut(ark_mem).method = loaded;

    if sprkStep_mem_mut(ark_mem).method.is_some() {
        ARK_SUCCESS
    } else {
        ARK_ILL_INPUT
    }
}

/*===============================================================
  Exported optional output functions.
  ===============================================================*/

/*---------------------------------------------------------------
  SPRKStepGetCurrentMethod:

  Returns the stepper method structure.
  ---------------------------------------------------------------*/
pub fn SPRKStepGetCurrentMethod(
    arkode_mem: &ARKodeMem,
    sprk_storage: &mut Option<ARKodeSPRKTable>,
) -> i32 {
    /* access ARKodeMem and ARKodeSPRKStepMem structures */
    let retval = sprkStep_AccessARKODEStepMem(arkode_mem, "SPRKStepGetCurrentMethod");
    if retval != ARK_SUCCESS {
        return retval;
    }
    let ark_mem = arkode_mem;

    /* C hands out the internal pointer (a borrowed reference); the `Rc`
       clone is that same shared handle. */
    *sprk_storage = sprkStep_mem_mut(ark_mem).method.clone();

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  sprkStep_GetNumRhsEvals:

  Returns the current number of RHS calls
  ---------------------------------------------------------------*/
pub fn sprkStep_GetNumRhsEvals(
    ark_mem: &ARKodeMem,
    partition_index: i32,
    rhs_evals: &mut i64,
) -> i32 {
    /* access ARKodeSPRKStepMem structure */
    let retval = sprkStep_AccessStepMem(ark_mem, "sprkStep_GetNumRhsEvals");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* NULL rhs_evals check: handled by the type system */

    if partition_index > 1 {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "sprkStep_GetNumRhsEvals",
            file!(),
            "Invalid partition index",
        );
        return ARK_ILL_INPUT;
    }

    let (nf1, nf2) = {
        let step_mem = sprkStep_mem_mut(ark_mem);
        (step_mem.nf1, step_mem.nf2)
    };

    match partition_index {
        0 => *rhs_evals = nf1,
        1 => *rhs_evals = nf2,
        _ => *rhs_evals = nf1 + nf2,
    }

    ARK_SUCCESS
}

pub fn SPRKStepGetNumRhsEvals(arkode_mem: &ARKodeMem, nf1: &mut i64, nf2: &mut i64) -> i32 {
    let retval = ARKodeGetNumRhsEvals(arkode_mem, 0, nf1);
    if retval != ARK_SUCCESS {
        return retval;
    }

    let retval = ARKodeGetNumRhsEvals(arkode_mem, 1, nf2);
    if retval != ARK_SUCCESS {
        return retval;
    }

    ARK_SUCCESS
}

/*===============================================================
  Private functions attached to ARKODE
  ===============================================================*/

/*---------------------------------------------------------------
  sprkStep_SetOption:

  Provides string-based control over SPRKStep-specific "set"
  routines.
  ---------------------------------------------------------------*/
pub fn sprkStep_SetOptions(
    ark_mem: &ARKodeMem,
    argidx: &mut i32,
    argv: &[String],
    offset: usize,
    arg_used: &mut sunbooleantype,
) -> i32 {
    /* Set lists of keys, and the corresponding set routines */
    static char_pairs: [sunKeyCharPair; 1] = [sunKeyCharPair {
        key: "method_name",
        set: cliSPRKStepSetMethodName,
    }];
    let num_char_keys: i32 = char_pairs.len() as i32;

    /* the CLI helpers receive C's `void* ark_mem` as a boxed handle clone */
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
            "sprkStep_SetOptions",
            file!(),
            &format!("error setting key: {}", char_pairs[j as usize].key),
        );
        return retval;
    }
    if *arg_used {
        return ARK_SUCCESS;
    }

    ARK_SUCCESS
}

/// Adapter matching `sundials_cli::sunCharSetFn`: recovers the
/// `ARKodeMem` handle from the CLI token (C: the raw `void* ark_mem`
/// forwarded by `sunCheckAndSetCharArgs`) and calls the real setter. A
/// missing/mistyped token corresponds to C passing a garbage pointer
/// (UB) and maps to a deterministic panic.
fn cliSPRKStepSetMethodName(mem: &mut Option<Box<dyn Any>>, arg: &str) -> i32 {
    let ark_mem: ARKodeMem = mem
        .as_mut()
        .and_then(|b| b.downcast_ref::<ARKodeMem>())
        .cloned()
        .expect("arkode_mem token");
    SPRKStepSetMethodName(&ark_mem, arg)
}

/*---------------------------------------------------------------
  sprkStep_SetDefaults:

  Resets all SPRKStep optional inputs to their default values.
  Does not change problem-defining function pointers or
  user_data pointer.  Also leaves alone any data
  structures/options related to the ARKODE infrastructure itself
  (e.g., root-finding and post-process step).
  ---------------------------------------------------------------*/
pub fn sprkStep_SetDefaults(ark_mem: &ARKodeMem) -> i32 {
    /* use the default method order */
    sprkStep_SetOrder(ark_mem, 0)
}

/*---------------------------------------------------------------
  sprkStep_SetOrder:

  Specifies the method order
  ---------------------------------------------------------------*/
pub fn sprkStep_SetOrder(ark_mem: &ARKodeMem, ord: i32) -> i32 {
    let mut ord = ord;

    /* access ARKodeSPRKStepMem structure */
    let retval = sprkStep_AccessStepMem(ark_mem, "sprkStep_SetOrder");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* Invalid orders result in the default order being used. */
    if ord == 7 || ord == 9 || ord > 10 {
        ord = -1;
    }

    /* set user-provided value, or default, depending on argument */
    if ord <= 0 {
        sprkStep_mem_mut(ark_mem).q = 4;
    } else {
        sprkStep_mem_mut(ark_mem).q = ord;
    }

    if sprkStep_mem_mut(ark_mem).method.is_some() {
        let method = sprkStep_mem_mut(ark_mem).method.take();
        ARKodeSPRKTable_Free(method);
        sprkStep_mem_mut(ark_mem).method = None;
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  sprkStep_GetStageIndex:

  Returns the current stage index and number of stages
  ---------------------------------------------------------------*/
pub fn sprkStep_GetStageIndex(ark_mem: &ARKodeMem, stage: &mut i32, max_stages: &mut i32) -> i32 {
    /* access ARKodeSPRKStepMem structure */
    let retval = sprkStep_AccessStepMem(ark_mem, "sprkStep_GetStageIndex");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* if table is not yet set, return defaults */
    if sprkStep_mem_mut(ark_mem).method.is_none() {
        *stage = -1;
        *max_stages = -1;
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "sprkStep_GetStageIndex",
            file!(),
            "method structure not allocated",
        );
        /* C returns `retval`, which is still ARK_SUCCESS here */
        return retval;
    } else {
        let (istage, method) = {
            let step_mem = sprkStep_mem_mut(ark_mem);
            (
                step_mem.istage,
                step_mem.method.clone().expect("method set"),
            )
        };
        *stage = istage;
        *max_stages = method.borrow().stages;
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  sprkStep_PrintAllStats:

  Prints integrator statistics
  ---------------------------------------------------------------*/
pub fn sprkStep_PrintAllStats(
    ark_mem: &ARKodeMem,
    outfile: &SUNFile,
    fmt: SUNOutputFormat,
) -> i32 {
    /* access ARKodeSPRKStepMem structure */
    let retval = sprkStep_AccessStepMem(ark_mem, "sprkStep_PrintAllStats");
    if retval != ARK_SUCCESS {
        return retval;
    }

    let (nf1, nf2) = {
        let step_mem = sprkStep_mem_mut(ark_mem);
        (step_mem.nf1, step_mem.nf2)
    };

    sunfprintf_long(outfile, fmt, SUNFALSE, "f1 RHS fn evals", nf1);
    sunfprintf_long(outfile, fmt, SUNFALSE, "f2 RHS fn evals", nf2);

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  sprkStep_WriteParameters:

  Outputs all solver parameters to the provided file pointer.
  ---------------------------------------------------------------*/
pub fn sprkStep_WriteParameters(ark_mem: &ARKodeMem, fp: &SUNFile) -> i32 {
    /* access ARKodeSPRKStepMem structure */
    let retval = sprkStep_AccessStepMem(ark_mem, "sprkStep_WriteParameters");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* C dereferences `step_mem->method` unconditionally (UB if unset) */
    let method = sprkStep_mem_mut(ark_mem).method.clone().expect("method set");
    let (q, stages) = {
        let m = method.borrow();
        (m.q, m.stages)
    };

    /* print integrator parameters to file */
    fp.write_str("SPRKStep time step module parameters:\n");
    fp.write_str(&format!("  Method order {}\n", q));
    fp.write_str(&format!("  Method stages {}\n", stages));

    ARK_SUCCESS
}

pub fn sprkStep_SetUseCompensatedSums(ark_mem: &ARKodeMem, onoff: sunbooleantype) -> i32 {
    /* access ARKodeSPRKStepMem structure */
    let retval = sprkStep_AccessStepMem(ark_mem, "sprkStep_SetUseCompensatedSums");
    if retval != ARK_SUCCESS {
        return retval;
    }

    if onoff {
        ark_mem.borrow_mut().step = Some(sprkStep_TakeStep_Compensated);
        if sprkStep_mem_mut(ark_mem).yerr.is_none() {
            let yn = ark_mem.borrow().yn.clone().expect("yn set");
            let mut yerr: Option<N_Vector> = None;
            if !arkAllocVec(ark_mem, &yn, &mut yerr) {
                return ARK_MEM_FAIL;
            }
            let yerr_v = yerr.as_ref().expect("yerr allocated").clone();
            sprkStep_mem_mut(ark_mem).yerr = yerr;
            /* Zero yerr for compensated summation */
            N_VConst(ZERO, &yerr_v);
        }
    } else {
        ark_mem.borrow_mut().step = Some(sprkStep_TakeStep);
    }

    retval
}

/*===============================================================
  Exported-but-deprecated user-callable functions.
  ===============================================================*/

pub fn SPRKStepReset(arkode_mem: &ARKodeMem, tR: sunrealtype, yR: &N_Vector) -> i32 {
    ARKodeReset(arkode_mem, tR, yR)
}

pub fn SPRKStepRootInit(arkode_mem: &ARKodeMem, nrtfn: i32, g: Option<ARKRootFn>) -> i32 {
    ARKodeRootInit(arkode_mem, nrtfn, g)
}

pub fn SPRKStepSetRootDirection(arkode_mem: &ARKodeMem, rootdir: &[i32]) -> i32 {
    ARKodeSetRootDirection(arkode_mem, rootdir)
}

pub fn SPRKStepSetNoInactiveRootWarn(arkode_mem: &ARKodeMem) -> i32 {
    ARKodeSetNoInactiveRootWarn(arkode_mem)
}

pub fn SPRKStepSetDefaults(arkode_mem: &ARKodeMem) -> i32 {
    ARKodeSetDefaults(arkode_mem)
}

pub fn SPRKStepSetOrder(arkode_mem: &ARKodeMem, ord: i32) -> i32 {
    ARKodeSetOrder(arkode_mem, ord)
}

pub fn SPRKStepSetInterpolantType(arkode_mem: &ARKodeMem, itype: i32) -> i32 {
    ARKodeSetInterpolantType(arkode_mem, itype)
}

pub fn SPRKStepSetInterpolantDegree(arkode_mem: &ARKodeMem, degree: i32) -> i32 {
    ARKodeSetInterpolantDegree(arkode_mem, degree)
}

pub fn SPRKStepSetMaxNumSteps(arkode_mem: &ARKodeMem, mxsteps: i64) -> i32 {
    ARKodeSetMaxNumSteps(arkode_mem, mxsteps)
}

pub fn SPRKStepSetStopTime(arkode_mem: &ARKodeMem, tstop: sunrealtype) -> i32 {
    ARKodeSetStopTime(arkode_mem, tstop)
}

pub fn SPRKStepSetFixedStep(arkode_mem: &ARKodeMem, hfixed: sunrealtype) -> i32 {
    ARKodeSetFixedStep(arkode_mem, hfixed)
}

pub fn SPRKStepSetUserData(arkode_mem: &ARKodeMem, user_data: Option<Box<dyn Any>>) -> i32 {
    ARKodeSetUserData(arkode_mem, user_data)
}

pub fn SPRKStepSetPostprocessStepFn(
    arkode_mem: &ARKodeMem,
    ProcessStep: Option<ARKPostProcessFn>,
) -> i32 {
    ARKodeSetPostprocessStepFn(arkode_mem, ProcessStep)
}

pub fn SPRKStepSetPostprocessStageFn(
    arkode_mem: &ARKodeMem,
    ProcessStage: Option<ARKPostProcessFn>,
) -> i32 {
    ARKodeSetPostprocessStageFn(arkode_mem, ProcessStage)
}

pub fn SPRKStepEvolve(
    arkode_mem: &ARKodeMem,
    tout: sunrealtype,
    yout: &N_Vector,
    tret: &mut sunrealtype,
    itask: i32,
) -> i32 {
    ARKodeEvolve(arkode_mem, tout, yout, tret, itask)
}

pub fn SPRKStepGetDky(arkode_mem: &ARKodeMem, t: sunrealtype, k: i32, dky: &N_Vector) -> i32 {
    ARKodeGetDky(arkode_mem, t, k, dky)
}

pub fn SPRKStepGetReturnFlagName(flag: i64) -> String {
    ARKodeGetReturnFlagName(flag)
}

pub fn SPRKStepGetCurrentState(arkode_mem: &ARKodeMem, state: &mut Option<N_Vector>) -> i32 {
    ARKodeGetCurrentState(arkode_mem, state)
}

pub fn SPRKStepGetCurrentStep(arkode_mem: &ARKodeMem, hcur: &mut sunrealtype) -> i32 {
    ARKodeGetCurrentStep(arkode_mem, hcur)
}

pub fn SPRKStepGetCurrentTime(arkode_mem: &ARKodeMem, tcur: &mut sunrealtype) -> i32 {
    ARKodeGetCurrentTime(arkode_mem, tcur)
}

pub fn SPRKStepGetLastStep(arkode_mem: &ARKodeMem, hlast: &mut sunrealtype) -> i32 {
    ARKodeGetLastStep(arkode_mem, hlast)
}

pub fn SPRKStepGetNumStepAttempts(arkode_mem: &ARKodeMem, nstep_attempts: &mut i64) -> i32 {
    ARKodeGetNumStepAttempts(arkode_mem, nstep_attempts)
}

pub fn SPRKStepGetNumSteps(arkode_mem: &ARKodeMem, nsteps: &mut i64) -> i32 {
    ARKodeGetNumSteps(arkode_mem, nsteps)
}

pub fn SPRKStepGetRootInfo(arkode_mem: &ARKodeMem, rootsfound: &mut [i32]) -> i32 {
    ARKodeGetRootInfo(arkode_mem, rootsfound)
}

pub fn SPRKStepGetUserData(
    arkode_mem: &ARKodeMem,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    ARKodeGetUserData(arkode_mem, user_data)
}

pub fn SPRKStepPrintAllStats(
    arkode_mem: &ARKodeMem,
    outfile: &SUNFile,
    fmt: SUNOutputFormat,
) -> i32 {
    ARKodePrintAllStats(arkode_mem, outfile, fmt)
}

pub fn SPRKStepWriteParameters(arkode_mem: &ARKodeMem, fp: &SUNFile) -> i32 {
    ARKodeWriteParameters(arkode_mem, fp)
}

pub fn SPRKStepGetStepStats(
    arkode_mem: &ARKodeMem,
    nsteps: &mut i64,
    hinused: &mut sunrealtype,
    hlast: &mut sunrealtype,
    hcur: &mut sunrealtype,
    tcur: &mut sunrealtype,
) -> i32 {
    ARKodeGetStepStats(arkode_mem, nsteps, hinused, hlast, hcur, tcur)
}

pub fn SPRKStepFree(arkode_mem: &mut Option<ARKodeMem>) {
    ARKodeFree(arkode_mem)
}

/*===============================================================
  EOF
  ===============================================================*/
