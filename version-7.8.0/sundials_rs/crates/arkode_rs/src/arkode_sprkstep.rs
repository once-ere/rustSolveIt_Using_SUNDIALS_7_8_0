//! Port of `src/arkode/arkode_sprkstep.c` (+ `src/arkode/arkode_sprkstep_impl.h`
//! and the constants of `include/arkode/arkode_sprkstep.h` folded in).
//!
//! ARKODE's symplectic-partitioned Runge-Kutta time stepper. The
//! problem is split as `p' = f1(t,q)` / `q' = f2(t,p)` and advanced with
//! the `a`/`ahat` coefficient pair of an `ARKodeSPRKTable`
//! (`arkode_sprk.rs`); SPRKStep supports neither temporal adaptivity
//! nor implicit solvers, so only the general part of the `step_*` table
//! is populated.
//!
//! Storage model (frozen contract §3): the stepper content struct lives
//! BY VALUE in `ark_mem.step_mem` (`Option<Box<dyn Any>>`) and is
//! reached through `sprkStep_mem_mut`, which returns a `RefMut` derived
//! from `ark_mem.borrow_mut()`. That guard IS a borrow of the mem: it is
//! never held across `arkProcessError`, a user callback, an `N_Vector`
//! operation, or an `ark*`/`ARKode*` core call. C's
//! `sprkStep_AccessStepMem` / `sprkStep_AccessARKODEStepMem` therefore
//! become presence checks followed by `sprkStep_mem_mut` at each use
//! site.
//!
//! `user_data`: each user callback invocation `Option::take`s the box
//! out of `ark_mem.user_data`, passes `&mut Option<Box<dyn Any>>`, and
//! restores it on every path (including the error returns).
//!
//! Signature deviations forced by the storage model: C's
//! `sprkStep_f1`/`sprkStep_f2` take `ARKodeSPRKStepMem step_mem` as
//! their first argument; here they take `&ARKodeMem` and reach the step
//! memory through it.
//!
//! `sprkStep_SetUserData` and `sprkStep_PrintMem` are declared in
//! `arkode_sprkstep_impl.h` but have no definition anywhere in the
//! upstream tree, so nothing is ported for them (stubbing is
//! forbidden).
//!
//! Logging: built at `SUNDIALS_LOGGING_LEVEL=2`, so every
//! `SUNLogInfo`/`SUNLogExtraDebugVec` in the C source compiles away and
//! is omitted here; the `ARK_WARNING` `arkProcessError` in
//! `sprkStep_Init` does print and is kept.

use std::any::Any;
use std::cell::RefMut;

use sundials_core::sundials_context::SUNContext;
use sundials_core::sundials_math::SUNRabs;
use sundials_core::sundials_nvector::{N_VConst, N_VLinearSum, N_VSpace, N_Vector};
use sundials_core::sundials_types::*;

use crate::arkode::{arkAllocVec, arkCreate, arkFreeVec, arkInit, arkResizeVec, ARKodeFree};
use crate::arkode_impl::*;
use crate::arkode_io::ARKodeSetInterpolantType;
use crate::arkode_sprk::*;
use crate::arkode_sprkstep_io::{
    sprkStep_GetNumRhsEvals, sprkStep_GetStageIndex, sprkStep_PrintAllStats, sprkStep_SetDefaults,
    sprkStep_SetOptions, sprkStep_SetOrder, sprkStep_SetUseCompensatedSums,
    sprkStep_WriteParameters,
};

/*===============================================================
  SPRKStep Constants (include/arkode/arkode_sprkstep.h)
  ===============================================================*/

pub const SPRKSTEP_DEFAULT_1: i32 = ARKODE_SPRK_EULER_1_1;
pub const SPRKSTEP_DEFAULT_2: i32 = ARKODE_SPRK_LEAPFROG_2_2;
pub const SPRKSTEP_DEFAULT_3: i32 = ARKODE_SPRK_MCLACHLAN_3_3;
pub const SPRKSTEP_DEFAULT_4: i32 = ARKODE_SPRK_MCLACHLAN_4_4;
pub const SPRKSTEP_DEFAULT_5: i32 = ARKODE_SPRK_MCLACHLAN_5_6;
pub const SPRKSTEP_DEFAULT_6: i32 = ARKODE_SPRK_YOSHIDA_6_8;
pub const SPRKSTEP_DEFAULT_8: i32 = ARKODE_SPRK_SUZUKI_UMENO_8_16;
pub const SPRKSTEP_DEFAULT_10: i32 = ARKODE_SPRK_SOFRONIOU_10_36;

/*===============================================================
  Reusable SPRKStep Error Messages (arkode_sprkstep_impl.h)
  ===============================================================*/

pub const MSG_SPRKSTEP_NO_MEM: &str = "Time step module memory is NULL.";

/*===============================================================
  SPRK time step module data structure (arkode_sprkstep_impl.h)
  ===============================================================*/

/// C `struct ARKodeSPRKStepMemRec`. Held BY VALUE in
/// `ark_mem.step_mem` (C `void* step_mem`).
pub struct ARKodeSPRKStepMemRec {
    /* SPRK method and storage */
    pub method: Option<ARKodeSPRKTable>, /* method spec  */
    pub q: i32,                          /* method order */
    pub sdata: Option<N_Vector>,         /* persisted stage data */
    pub yerr: Option<N_Vector>,          /* error vector for compensated summation */

    /* SPRK problem specification */
    pub f1: Option<ARKRhsFn>, /* p' = f1(t,q) = - dV(t,q)/dq  */
    pub f2: Option<ARKRhsFn>, /* q' = f2(t,p) =   dT(t,p)/dp  */

    /* Counters */
    pub nf1: i64, /* number of calls to f1        */
    pub nf2: i64, /* number of calls to f2        */
    pub istage: i32,
}

impl ARKodeSPRKStepMemRec {
    /// All-zero/None baseline, mirroring C's
    /// `malloc` + `memset(step_mem, 0, sizeof(struct ARKodeSPRKStepMemRec))`.
    pub fn zeroed() -> ARKodeSPRKStepMemRec {
        ARKodeSPRKStepMemRec {
            method: None,
            q: 0,
            sdata: None,
            yerr: None,
            f1: None,
            f2: None,
            nf1: 0,
            nf2: 0,
            istage: 0,
        }
    }
}

/// Downcast helper: view `ark_mem.step_mem` as the SPRKStep memory
/// record. Panics if no step memory is attached or it is not SPRKStep's
/// record (C would blindly cast the `void*` -- UB maps to a panic).
/// NEVER hold the returned guard across `arkProcessError`, a user
/// callback, an `N_Vector` operation, or a core `ark*` call.
pub fn sprkStep_mem_mut(ark_mem: &ARKodeMem) -> RefMut<'_, ARKodeSPRKStepMemRec> {
    RefMut::map(ark_mem.borrow_mut(), |m| {
        m.step_mem
            .as_mut()
            .expect("step_mem set")
            .downcast_mut::<ARKodeSPRKStepMemRec>()
            .expect("SPRKStep step memory")
    })
}

/*===============================================================
  Exported functions
  ===============================================================*/

pub fn SPRKStepCreate(
    f1: ARKRhsFn,
    f2: ARKRhsFn,
    t0: sunrealtype,
    y0: &N_Vector,
    sunctx: &SUNContext,
) -> Option<ARKodeMem> {
    /* Check that f1 and f2 are supplied: handled by the type system */

    /* Check for legal input parameters */
    /* NULL y0 check: handled by the type system */

    /* NULL sunctx check: handled by the type system */

    /* Create ark_mem structure and set default values */
    let ark_mem = arkCreate(sunctx);
    if ark_mem.is_none() {
        arkProcessError(
            None,
            ARK_MEM_NULL,
            line!() as i32,
            "SPRKStepCreate",
            file!(),
            MSG_ARK_NO_MEM,
        );
        return None;
    }
    let ark_mem = ark_mem.expect("arkCreate");

    /* Allocate ARKodeSPRKStepMem structure, and initialize to zero
       (the C malloc-failure branch cannot be observed in Rust) */
    let step_mem = ARKodeSPRKStepMemRec::zeroed();

    /* Attach stepper memory early so generic cleanup can handle partial setup. */
    {
        let mut m = ark_mem.borrow_mut();
        m.step_free = Some(sprkStep_Free);
        m.step_mem = Some(Box::new(step_mem));
    }

    /* Allocate vectors in stepper mem */
    let mut sdata: Option<N_Vector> = None;
    if !arkAllocVec(&ark_mem, y0, &mut sdata) {
        let mut arkode_mem = Some(ark_mem);
        ARKodeFree(&mut arkode_mem);
        return None;
    }
    sprkStep_mem_mut(&ark_mem).sdata = sdata;

    let use_compensated_sums = ark_mem.borrow().use_compensated_sums;
    if use_compensated_sums {
        let mut yerr: Option<N_Vector> = None;
        if !arkAllocVec(&ark_mem, y0, &mut yerr) {
            let mut arkode_mem = Some(ark_mem);
            ARKodeFree(&mut arkode_mem);
            return None;
        }
        let yerr_v = yerr.as_ref().expect("yerr allocated").clone();
        sprkStep_mem_mut(&ark_mem).yerr = yerr;
        /* Zero yerr for compensated summation */
        N_VConst(ZERO, &yerr_v);
    } else {
        sprkStep_mem_mut(&ark_mem).yerr = None;
    }
    {
        let mut m = ark_mem.borrow_mut();
        m.step_init = Some(sprkStep_Init);
        m.step_fullrhs = Some(sprkStep_FullRHS);
        m.step = Some(sprkStep_TakeStep);
        m.step_printallstats = Some(sprkStep_PrintAllStats);
        m.step_writeparameters = Some(sprkStep_WriteParameters);
        m.step_setusecompensatedsums = Some(sprkStep_SetUseCompensatedSums);
        m.step_setoptions = Some(sprkStep_SetOptions);
        m.step_resize = Some(sprkStep_Resize);
        m.step_setdefaults = Some(sprkStep_SetDefaults);
        m.step_setorder = Some(sprkStep_SetOrder);
        m.step_getnumrhsevals = Some(sprkStep_GetNumRhsEvals);
        m.step_getstageindex = Some(sprkStep_GetStageIndex);
    }

    /* Set default values for optional inputs */
    let retval = sprkStep_SetDefaults(&ark_mem);
    if retval != ARK_SUCCESS {
        arkProcessError(
            Some(&ark_mem),
            retval,
            line!() as i32,
            "SPRKStepCreate",
            file!(),
            "Error setting default solver options",
        );
        let mut arkode_mem = Some(ark_mem);
        ARKodeFree(&mut arkode_mem);
        return None;
    }

    /* Copy the input parameters into ARKODE state */
    {
        let mut step_mem = sprkStep_mem_mut(&ark_mem);
        step_mem.f1 = Some(f1);
        step_mem.f2 = Some(f2);

        /* Initialize the counters */
        step_mem.nf1 = 0;
        step_mem.nf2 = 0;
        step_mem.istage = 0;
    }

    /* SPRKStep uses Lagrange interpolation by default, since Hermite is
       less compatible with these methods. */
    let _ = ARKodeSetInterpolantType(&ark_mem, ARK_INTERP_LAGRANGE);

    /* Initialize main ARKODE infrastructure */
    let retval = arkInit(&ark_mem, t0, y0, FIRST_INIT);
    if retval != ARK_SUCCESS {
        arkProcessError(
            Some(&ark_mem),
            retval,
            line!() as i32,
            "SPRKStepCreate",
            file!(),
            "Unable to initialize main ARKODE infrastructure",
        );
        let mut arkode_mem = Some(ark_mem);
        ARKodeFree(&mut arkode_mem);
        return None;
    }

    Some(ark_mem)
}

/*---------------------------------------------------------------
  SPRKStepReInit:

  This routine re-initializes the SPRKStep module to solve a new
  problem of the same size as was previously solved. This routine
  should also be called when the problem dynamics or desired solvers
  have changed dramatically, so that the problem integration should
  resume as if started from scratch.

  Note all internal counters are set to 0 on re-initialization.
  ---------------------------------------------------------------*/
pub fn SPRKStepReInit(
    arkode_mem: &ARKodeMem,
    f1: ARKRhsFn,
    f2: ARKRhsFn,
    t0: sunrealtype,
    y0: &N_Vector,
) -> i32 {
    /* access ARKodeMem and ARKodeSPRKStepMem structures */
    let retval = sprkStep_AccessARKODEStepMem(arkode_mem, "SPRKStepReInit");
    if retval != ARK_SUCCESS {
        return retval;
    }
    let ark_mem = arkode_mem;

    /* Check if ark_mem was allocated */
    if ark_mem.borrow().MallocDone == SUNFALSE {
        arkProcessError(
            Some(ark_mem),
            ARK_NO_MALLOC,
            line!() as i32,
            "SPRKStepReInit",
            file!(),
            MSG_ARK_NO_MALLOC,
        );
        return ARK_NO_MALLOC;
    }

    /* Check that f1 and f2 are supplied: handled by the type system */

    /* Check that y0 is supplied: handled by the type system */

    /* Copy the input parameters into ARKODE state */
    {
        let mut step_mem = sprkStep_mem_mut(ark_mem);
        step_mem.f1 = Some(f1);
        step_mem.f2 = Some(f2);
    }

    /* Initialize main ARKODE infrastructure */
    let retval = arkInit(ark_mem, t0, y0, FIRST_INIT);
    if retval != ARK_SUCCESS {
        arkProcessError(
            Some(ark_mem),
            retval,
            line!() as i32,
            "SPRKStepReInit",
            file!(),
            "Unable to reinitialize main ARKODE infrastructure",
        );
        return retval;
    }

    /* Initialize the counters */
    {
        let mut step_mem = sprkStep_mem_mut(ark_mem);
        step_mem.nf1 = 0;
        step_mem.nf2 = 0;
        step_mem.istage = 0;
    }

    /* Zero yerr for compensated summation */
    if ark_mem.borrow().use_compensated_sums {
        let yerr = sprkStep_mem_mut(ark_mem).yerr.clone().expect("yerr set");
        N_VConst(ZERO, &yerr);
    }

    ARK_SUCCESS
}

/*===============================================================
  Interface routines supplied to ARKODE
  ===============================================================*/

/*---------------------------------------------------------------
  sprkStep_Resize:

  This routine resizes the memory within the SPRKStep module.
  ---------------------------------------------------------------*/
pub fn sprkStep_Resize(
    ark_mem: &ARKodeMem,
    y0: &N_Vector,
    hscale: sunrealtype,
    t0: sunrealtype,
    resize: Option<ARKVecResizeFn>,
    resize_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    let _ = hscale; /* SUNDIALS_MAYBE_UNUSED */
    let _ = t0; /* SUNDIALS_MAYBE_UNUSED */

    /* access ARKodeSPRKStepMem structure */
    let retval = sprkStep_AccessStepMem(ark_mem, "sprkStep_Resize");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* Determine change in vector sizes */
    let mut lrw1: sunindextype = 0;
    let mut liw1: sunindextype = 0;
    if y0.ops.borrow().nvspace.is_some() {
        N_VSpace(y0, &mut lrw1, &mut liw1);
    }
    let lrw_diff;
    let liw_diff;
    {
        let mut m = ark_mem.borrow_mut();
        lrw_diff = lrw1 - m.lrw1;
        liw_diff = liw1 - m.liw1;
        m.lrw1 = lrw1;
        m.liw1 = liw1;
    }

    /* Resize the local vectors */
    let mut sdata = sprkStep_mem_mut(ark_mem).sdata.take();
    let ok = arkResizeVec(
        ark_mem,
        resize,
        resize_data,
        lrw_diff,
        liw_diff,
        y0,
        &mut sdata,
    );
    sprkStep_mem_mut(ark_mem).sdata = sdata;
    if !ok {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_FAIL,
            line!() as i32,
            "sprkStep_Resize",
            file!(),
            "Unable to resize vector",
        );
        return ARK_MEM_FAIL;
    }

    if sprkStep_mem_mut(ark_mem).yerr.is_some() {
        let mut yerr = sprkStep_mem_mut(ark_mem).yerr.take();
        let ok = arkResizeVec(
            ark_mem,
            resize,
            resize_data,
            lrw_diff,
            liw_diff,
            y0,
            &mut yerr,
        );
        let yerr_v = yerr.clone();
        sprkStep_mem_mut(ark_mem).yerr = yerr;
        if !ok {
            arkProcessError(
                Some(ark_mem),
                ARK_MEM_FAIL,
                line!() as i32,
                "sprkStep_Resize",
                file!(),
                "Unable to resize vector",
            );
            return ARK_MEM_FAIL;
        }
        /* Zero yerr for compensated summation */
        N_VConst(ZERO, yerr_v.as_ref().expect("yerr resized"));
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  sprkStep_Reset:

  This routine resets the SPRKStep module state to solve the same
  problem from the given time with the input state (all counter
  values are retained).
  ---------------------------------------------------------------*/
pub fn sprkStep_Reset(ark_mem: &ARKodeMem, tR: sunrealtype, yR: &N_Vector) -> i32 {
    let _ = tR; /* SUNDIALS_MAYBE_UNUSED */
    let _ = yR; /* SUNDIALS_MAYBE_UNUSED */

    /* access ARKodeSPRKStepMem structure */
    let retval = sprkStep_AccessStepMem(ark_mem, "sprkStep_Reset");
    if retval != ARK_SUCCESS {
        return retval;
    }

    if ark_mem.borrow().use_compensated_sums {
        let yerr = sprkStep_mem_mut(ark_mem).yerr.clone().expect("yerr set");
        N_VConst(0.0, &yerr);
    }
    ARK_SUCCESS
}

/*---------------------------------------------------------------
  sprkStep_Free frees all SPRKStep memory.
  ---------------------------------------------------------------*/
pub fn sprkStep_Free(ark_mem: &ARKodeMem) {
    /* nothing to do if ark_mem is already NULL: handled by the type system */

    /* conditional frees on non-NULL SPRKStep module */
    if ark_mem.borrow().step_mem.is_some() {
        let mut sdata = sprkStep_mem_mut(ark_mem).sdata.take();
        if sdata.is_some() {
            arkFreeVec(ark_mem, &mut sdata);
        }

        let mut yerr = sprkStep_mem_mut(ark_mem).yerr.take();
        if yerr.is_some() {
            arkFreeVec(ark_mem, &mut yerr);
        }

        let method = sprkStep_mem_mut(ark_mem).method.take();
        ARKodeSPRKTable_Free(method);

        ark_mem.borrow_mut().step_mem = None;
    }
}

/*---------------------------------------------------------------
  sprkStep_Init:

  This routine is called just prior to performing internal time
  steps (after all user "set" routines have been called) from
  within arkInitialSetup.

  For all initialization types, this routine sets the relevant
  TakeStep routine based on the current problem configuration.

  With initialization type FIRST_INIT or RESIZE_INIT, this routine
  this routines loads the default method of the selected order
  if necessary.

  With initialization type RESET_INIT, this routine does nothing.
  ---------------------------------------------------------------*/
pub fn sprkStep_Init(ark_mem: &ARKodeMem, init_type: i32) -> i32 {
    /* access ARKodeSPRKStepMem structure */
    let retval = sprkStep_AccessStepMem(ark_mem, "sprkStep_Init");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* immediately return if reset */
    if init_type == RESET_INIT {
        return ARK_SUCCESS;
    }

    /* initializations/checks for (re-)initialization call */
    if init_type == FIRST_INIT {
        if sprkStep_mem_mut(ark_mem).method.is_none() {
            let q = sprkStep_mem_mut(ark_mem).q;
            let method = match q {
                1 => ARKodeSPRKTable_Load(SPRKSTEP_DEFAULT_1),
                2 => ARKodeSPRKTable_Load(SPRKSTEP_DEFAULT_2),
                3 => ARKodeSPRKTable_Load(SPRKSTEP_DEFAULT_3),
                4 => ARKodeSPRKTable_Load(SPRKSTEP_DEFAULT_4),
                5 => ARKodeSPRKTable_Load(SPRKSTEP_DEFAULT_5),
                6 => ARKodeSPRKTable_Load(SPRKSTEP_DEFAULT_6),
                7 | 8 => ARKodeSPRKTable_Load(SPRKSTEP_DEFAULT_8),
                9 | 10 => ARKodeSPRKTable_Load(SPRKSTEP_DEFAULT_10),
                _ => {
                    arkProcessError(
                        Some(ark_mem),
                        ARK_WARNING,
                        line!() as i32,
                        "sprkStep_Init",
                        file!(),
                        "No SPRK method at requested order, using q=4.",
                    );
                    ARKodeSPRKTable_Load(SPRKSTEP_DEFAULT_4)
                }
            };
            sprkStep_mem_mut(ark_mem).method = method;
        }
    }

    /* Override the interpolant degree (if needed), used in arkInitialSetup */
    let method = sprkStep_mem_mut(ark_mem).method.clone().expect("method set");
    let method_q = method.borrow().q;
    let interp_degree = ark_mem.borrow().interp_degree;
    if method_q > 1 && interp_degree > (method_q - 1) {
        /* Limit max degree to at most one less than the method global order */
        ark_mem.borrow_mut().interp_degree = method_q - 1;
    } else if method_q == 1 && interp_degree > 1 {
        /* Allow for linear interpolant with first order methods to ensure
           solution values are returned at the time interval end points */
        ark_mem.borrow_mut().interp_degree = 1;
    }

    /* Zero yerr for compensated summation */
    if ark_mem.borrow().use_compensated_sums {
        let yerr = sprkStep_mem_mut(ark_mem).yerr.clone().expect("yerr set");
        N_VConst(ZERO, &yerr);
    }

    ARK_SUCCESS
}

/// Utility to call f1 and increment the counter.
///
/// C takes `ARKodeSPRKStepMem step_mem`; the step memory lives inside
/// `ark_mem` here, so the handle is passed instead. The fn pointer is
/// copied out before the call so no borrow of `ark_mem` is live while
/// the user callback runs.
pub fn sprkStep_f1(
    ark_mem: &ARKodeMem,
    tcur: sunrealtype,
    ycur: &N_Vector,
    f1: &N_Vector,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    let f1_fn = sprkStep_mem_mut(ark_mem).f1.expect("f1 set");
    let retval = f1_fn(tcur, ycur, f1, user_data);
    sprkStep_mem_mut(ark_mem).nf1 += 1;
    retval
}

/// Utility to call f2 and increment the counter (see `sprkStep_f1`).
pub fn sprkStep_f2(
    ark_mem: &ARKodeMem,
    tcur: sunrealtype,
    ycur: &N_Vector,
    f2: &N_Vector,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    let f2_fn = sprkStep_mem_mut(ark_mem).f2.expect("f2 set");
    let retval = f2_fn(tcur, ycur, f2, user_data);
    sprkStep_mem_mut(ark_mem).nf2 += 1;
    retval
}

/*------------------------------------------------------------------------------
  sprkStep_FullRHS:

  This is just a wrapper to call the user-supplied RHS,
  f1(t,y) + f2(t,y).

  This will be called in one of three 'modes':

     ARK_FULLRHS_START -> called at the beginning of a simulation i.e., at
                          (tn, yn) = (t0, y0) or (tR, yR)

     ARK_FULLRHS_END   -> called at the end of a successful step i.e, at
                          (tcur, ycur) or the start of the subsequent step i.e.,
                          at (tn, yn) = (tcur, ycur) from the end of the last
                          step

     ARK_FULLRHS_OTHER -> called elsewhere (e.g. for dense output)

  Since RHS values are not stored in SPRKStep we evaluate the RHS functions for
  all modes.
  ----------------------------------------------------------------------------*/
pub fn sprkStep_FullRHS(
    ark_mem: &ARKodeMem,
    t: sunrealtype,
    y: &N_Vector,
    f: &N_Vector,
    mode: i32,
) -> i32 {
    /* access ARKodeSPRKStepMem structure */
    let retval = sprkStep_AccessStepMem(ark_mem, "sprkStep_FullRHS");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* perform RHS functions contingent on 'mode' argument */
    match mode {
        ARK_FULLRHS_START | ARK_FULLRHS_END | ARK_FULLRHS_OTHER => {
            /* call the user-supplied pre-RHS function (if supplied) */
            let PreRhsFn = ark_mem.borrow().PreRhsFn;
            if let Some(PreRhsFn) = PreRhsFn {
                let mut user_data = ark_mem.borrow_mut().user_data.take();
                let retval = PreRhsFn(t, y, &mut user_data);
                ark_mem.borrow_mut().user_data = user_data;
                if retval != 0 {
                    return ARK_PRERHSFN_FAIL;
                }
            }

            /* Since f1 and f2 do not have overlapping outputs and so the f vector is
               passed to both RHS functions. */

            let mut user_data = ark_mem.borrow_mut().user_data.take();
            let retval = sprkStep_f1(ark_mem, t, y, f, &mut user_data);
            ark_mem.borrow_mut().user_data = user_data;
            if retval != 0 {
                arkProcessError(
                    Some(ark_mem),
                    ARK_RHSFUNC_FAIL,
                    line!() as i32,
                    "sprkStep_FullRHS",
                    file!(),
                    &MSG_ARK_RHSFUNC_FAILED(t),
                );
                return ARK_RHSFUNC_FAIL;
            }

            let mut user_data = ark_mem.borrow_mut().user_data.take();
            let retval = sprkStep_f2(ark_mem, t, y, f, &mut user_data);
            ark_mem.borrow_mut().user_data = user_data;
            if retval != 0 {
                arkProcessError(
                    Some(ark_mem),
                    ARK_RHSFUNC_FAIL,
                    line!() as i32,
                    "sprkStep_FullRHS",
                    file!(),
                    &MSG_ARK_RHSFUNC_FAILED(t),
                );
                return ARK_RHSFUNC_FAIL;
            }
        }

        _ => {
            /* return with RHS failure if unknown mode is passed */
            arkProcessError(
                Some(ark_mem),
                ARK_RHSFUNC_FAIL,
                line!() as i32,
                "sprkStep_FullRHS",
                file!(),
                "Unknown full RHS mode",
            );
            return ARK_RHSFUNC_FAIL;
        }
    }

    ARK_SUCCESS
}

/* Standard formulation of SPRK.
   This requires only 2 vectors in principle, but we use three
   since we persist the stage data. Only the stage data vector
   belongs to SPRKStep, the other two are reused from the ARKODE core. */
pub fn sprkStep_TakeStep(
    ark_mem: &ARKodeMem,
    dsmPtr: &mut sunrealtype,
    nflagPtr: &mut i32,
) -> i32 {
    let mut ci: sunrealtype = 0.0;
    let mut chati: sunrealtype = 0.0;

    /* access ARKodeSPRKStepMem structure */
    let retval = sprkStep_AccessStepMem(ark_mem, "sprkStep_TakeStep");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* C dereferences `step_mem->method` / `step_mem->sdata` throughout the
       loop below; neither can change during a step, so the handles are
       copied out once here. */
    let (method, sdata) = {
        let step_mem = sprkStep_mem_mut(ark_mem);
        (
            step_mem.method.clone().expect("method set"),
            step_mem.sdata.clone().expect("sdata set"),
        )
    };
    let stages = method.borrow().stages;

    let mut prev_stage = ark_mem.borrow().yn.clone().expect("yn set");
    let curr_stage = ark_mem.borrow().ycur.clone().expect("ycur set");

    let mut is: i32 = 0;
    while is < stages {
        /* load/compute coefficients */
        let (ai, ahati) = {
            let m = method.borrow();
            (m.a[is as usize], m.ahat[is as usize])
        };

        ci += ai;
        chati += ahati;

        /* store current stage index */
        sprkStep_mem_mut(ark_mem).istage = is;

        /* evaluate p' with the previous velocity */
        if SUNRabs(ahati) > TINY {
            N_VConst(ZERO, &sdata); /* either have to do this or ask user to
                                    set other outputs to zero */

            /* call the user-supplied pre-RHS function (if supplied) */
            let PreRhsFn = ark_mem.borrow().PreRhsFn;
            if let Some(PreRhsFn) = PreRhsFn {
                let t = {
                    let m = ark_mem.borrow();
                    m.tn + chati * m.h
                };
                let mut user_data = ark_mem.borrow_mut().user_data.take();
                let retval = PreRhsFn(t, &prev_stage, &mut user_data);
                ark_mem.borrow_mut().user_data = user_data;
                if retval != 0 {
                    return ARK_PRERHSFN_FAIL;
                }
            }

            /* evaluate p' */
            let t = {
                let m = ark_mem.borrow();
                m.tn + chati * m.h
            };
            let mut user_data = ark_mem.borrow_mut().user_data.take();
            let retval = sprkStep_f1(ark_mem, t, &prev_stage, &sdata, &mut user_data);
            ark_mem.borrow_mut().user_data = user_data;

            if retval != 0 {
                return ARK_RHSFUNC_FAIL;
            }
        }

        /* position update */
        let h = ark_mem.borrow().h;
        N_VLinearSum(ONE, &prev_stage, h * ahati, &sdata, &curr_stage);

        /* set current stage time(s) */
        let tcur = {
            let m = ark_mem.borrow();
            m.tn + chati * m.h
        };
        ark_mem.borrow_mut().tcur = tcur;

        /* evaluate q' with the current positions and update velocity */
        if SUNRabs(ai) > TINY {
            N_VConst(ZERO, &sdata); /* either have to do this or ask user to
                                    set other outputs to zero */

            /* call the user-supplied pre-RHS function (if supplied) */
            let PreRhsFn = ark_mem.borrow().PreRhsFn;
            if let Some(PreRhsFn) = PreRhsFn {
                let t = {
                    let m = ark_mem.borrow();
                    m.tn + ci * m.h
                };
                let mut user_data = ark_mem.borrow_mut().user_data.take();
                let retval = PreRhsFn(t, &curr_stage, &mut user_data);
                ark_mem.borrow_mut().user_data = user_data;
                if retval != 0 {
                    return ARK_PRERHSFN_FAIL;
                }
            }

            /* evaluate q' */
            let t = {
                let m = ark_mem.borrow();
                m.tn + ci * m.h
            };
            let mut user_data = ark_mem.borrow_mut().user_data.take();
            let retval = sprkStep_f2(ark_mem, t, &curr_stage, &sdata, &mut user_data);
            ark_mem.borrow_mut().user_data = user_data;

            if retval != 0 {
                return ARK_RHSFUNC_FAIL;
            }

            /* velocity update */
            let h = ark_mem.borrow().h;
            N_VLinearSum(ONE, &curr_stage, h * ai, &sdata, &curr_stage);
        }

        /* apply user-supplied stage postprocessing function (if supplied) */
        let (PostProcessStageFn, PostProcessStepFn) = {
            let m = ark_mem.borrow();
            (m.PostProcessStageFn, m.PostProcessStepFn)
        };
        if is < stages - 1 && PostProcessStageFn.is_some() {
            let PostProcessStageFn = PostProcessStageFn.expect("PostProcessStageFn set");
            let (tcur, ycur) = {
                let m = ark_mem.borrow();
                (m.tcur, m.ycur.clone().expect("ycur set"))
            };
            let mut user_data = ark_mem.borrow_mut().user_data.take();
            let retval = PostProcessStageFn(tcur, &ycur, &mut user_data);
            ark_mem.borrow_mut().user_data = user_data;
            if retval != 0 {
                return ARK_POSTPROCESS_STAGE_FAIL;
            }
        } else if is == stages - 1 && PostProcessStepFn.is_some() {
            let PostProcessStepFn = PostProcessStepFn.expect("PostProcessStepFn set");
            let (tcur, ycur) = {
                let m = ark_mem.borrow();
                (m.tcur, m.ycur.clone().expect("ycur set"))
            };
            let mut user_data = ark_mem.borrow_mut().user_data.take();
            let retval = PostProcessStepFn(tcur, &ycur, &mut user_data);
            ark_mem.borrow_mut().user_data = user_data;
            if retval != 0 {
                return ARK_POSTPROCESS_STEP_FAIL;
            }
        }

        prev_stage = curr_stage.clone();

        is += 1;
    }

    *nflagPtr = 0;
    *dsmPtr = 0.0;

    ARK_SUCCESS
}

/* Increment SPRK algorithm with compensated summation.
   This algorithm requires 6 vectors, but 5 of them are reused
   from the ARKODE core. */
pub fn sprkStep_TakeStep_Compensated(
    ark_mem: &ARKodeMem,
    dsmPtr: &mut sunrealtype,
    nflagPtr: &mut i32,
) -> i32 {
    let mut ci: sunrealtype = 0.0;
    let mut chati: sunrealtype = 0.0;

    /* access ARKodeSPRKStepMem structure */
    let retval = sprkStep_AccessStepMem(ark_mem, "sprkStep_TakeStep_Compensated");
    if retval != ARK_SUCCESS {
        return retval;
    }

    let (method, sdata, yerr) = {
        let step_mem = sprkStep_mem_mut(ark_mem);
        (
            step_mem.method.clone().expect("method set"),
            step_mem.sdata.clone().expect("sdata set"),
            step_mem.yerr.clone().expect("yerr set"),
        )
    };
    let stages = method.borrow().stages;

    /* Vector shortcuts */
    let (delta_Yi, yn_plus_delta_Yi, diff, yn, ycur) = {
        let m = ark_mem.borrow();
        (
            m.tempv1.clone().expect("tempv1 set"),
            m.tempv2.clone().expect("tempv2 set"),
            m.tempv3.clone().expect("tempv3 set"),
            m.yn.clone().expect("yn set"),
            m.ycur.clone().expect("ycur set"),
        )
    };

    /* [ \Delta P_0 ] = [ 0 ]
       [ \Delta Q_0 ] = [ 0 ] */
    N_VConst(ZERO, &delta_Yi);

    /* if user-supplied stage preprocessing or postprocessing functions,
     * we error out since those won't work with the increment form */
    let has_processing = {
        let m = ark_mem.borrow();
        m.PreRhsFn.is_some() || m.PostProcessStageFn.is_some() || m.PostProcessStepFn.is_some()
    };
    if has_processing {
        arkProcessError(
            Some(ark_mem),
            ARK_POSTPROCESS_STAGE_FAIL,
            line!() as i32,
            "sprkStep_TakeStep_Compensated",
            file!(),
            "Compensated summation is not compatible with stage Pre- or PostProcessing!\n",
        );
        return ARK_POSTPROCESS_STAGE_FAIL;
    }

    /* loop over internal stages to the step */
    let mut is: i32 = 0;
    while is < stages {
        /* load/compute coefficients */
        let (ai, ahati) = {
            let m = method.borrow();
            (m.a[is as usize], m.ahat[is as usize])
        };

        ci += ai;
        chati += ahati;

        /* store current stage index */
        sprkStep_mem_mut(ark_mem).istage = is;

        /* [     ] + [            ]
           [ q_n ] + [ \Delta Q_i ] */
        N_VLinearSum(ONE, &yn, ONE, &delta_Yi, &yn_plus_delta_Yi);

        if SUNRabs(ahati) > TINY {
            /* Evaluate p' with the previous velocity */
            N_VConst(ZERO, &sdata); /* either have to do this or ask user to
                                    set other outputs to zero */
            let t = {
                let m = ark_mem.borrow();
                m.tn + chati * m.h
            };
            let mut user_data = ark_mem.borrow_mut().user_data.take();
            let retval = sprkStep_f1(ark_mem, t, &yn_plus_delta_Yi, &sdata, &mut user_data);
            ark_mem.borrow_mut().user_data = user_data;

            if retval != 0 {
                return ARK_RHSFUNC_FAIL;
            }

            /* Incremental position update:
               [ \Delta P_i ] = [ \Delta P_{i-1} ] + [ sdata ]
               [            ] = [                ] + [       ] */
            let h = ark_mem.borrow().h;
            N_VLinearSum(ONE, &delta_Yi, h * ahati, &sdata, &delta_Yi);
        }

        /* [ p_n ] + [ \Delta P_i ]
           [     ] + [            ] */
        N_VLinearSum(ONE, &yn, ONE, &delta_Yi, &yn_plus_delta_Yi);

        /* set current stage time(s) */
        let tcur = {
            let m = ark_mem.borrow();
            m.tn + chati * m.h
        };
        ark_mem.borrow_mut().tcur = tcur;

        if SUNRabs(ai) > TINY {
            /* Evaluate q' with the current positions */
            N_VConst(ZERO, &sdata); /* either have to do this or ask user to
                                    set other outputs to zero */
            let t = {
                let m = ark_mem.borrow();
                m.tn + ci * m.h
            };
            let mut user_data = ark_mem.borrow_mut().user_data.take();
            let retval = sprkStep_f2(ark_mem, t, &yn_plus_delta_Yi, &sdata, &mut user_data);
            ark_mem.borrow_mut().user_data = user_data;

            if retval != 0 {
                return ARK_RHSFUNC_FAIL;
            }

            /* Incremental velocity update:
               [            ] = [                ] + [       ]
               [ \Delta Q_i ] = [ \Delta Q_{i-1} ] + [ sdata ] */
            let h = ark_mem.borrow().h;
            N_VLinearSum(ONE, &delta_Yi, h * ai, &sdata, &delta_Yi);
        }

        is += 1;
    }

    /*
      Now we compute the step solution via compensated summation.
       [ p_{n+1} ] = [ p_n ] + [ \Delta P_i ]
       [ q_{n+1} ] = [ q_n ] + [ \Delta Q_i ] */
    N_VLinearSum(ONE, &delta_Yi, -ONE, &yerr, &delta_Yi);
    N_VLinearSum(ONE, &yn, ONE, &delta_Yi, &ycur);
    N_VLinearSum(ONE, &ycur, -ONE, &yn, &diff);
    N_VLinearSum(ONE, &diff, -ONE, &delta_Yi, &yerr);

    *nflagPtr = 0;
    *dsmPtr = 0.0;

    0
}

/*===============================================================
  Internal utility routines
  ===============================================================*/

/*---------------------------------------------------------------
  sprkStep_AccessARKODEStepMem:

  Shortcut routine to unpack ark_mem and step_mem structures from
  void* pointer.  If either is missing it returns ARK_MEM_NULL.

  Rust: the `void*` unpacking is handled by the type system, so only
  the step-memory presence check survives; the record itself is reached
  with `sprkStep_mem_mut` at each use site (frozen contract §3).
  `fname` is the caller's `__func__`.
  ---------------------------------------------------------------*/
pub fn sprkStep_AccessARKODEStepMem(arkode_mem: &ARKodeMem, fname: &str) -> i32 {
    /* access ARKodeMem structure: handled by the type system */
    let ark_mem = arkode_mem;

    /* access ARKodeSPRKStepMem structure */
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            fname,
            file!(),
            MSG_SPRKSTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }
    ARK_SUCCESS
}

/*---------------------------------------------------------------
  sprkStep_AccessStepMem:

  Shortcut routine to unpack step_mem structure from ark_mem.
  If missing it returns ARK_MEM_NULL.
  ---------------------------------------------------------------*/
pub fn sprkStep_AccessStepMem(ark_mem: &ARKodeMem, fname: &str) -> i32 {
    /* access ARKodeSPRKStepMem structure */
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            fname,
            file!(),
            MSG_SPRKSTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }
    ARK_SUCCESS
}

/*===============================================================
  EOF
  ===============================================================*/
