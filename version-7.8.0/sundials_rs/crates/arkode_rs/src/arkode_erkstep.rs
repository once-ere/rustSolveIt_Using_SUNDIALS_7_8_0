//! Port of `src/arkode/arkode_erkstep.c`, with
//! `src/arkode/arkode_erkstep_impl.h` and the constants of
//! `include/arkode/arkode_erkstep.h` folded in: ARKODE's explicit
//! Runge-Kutta (ERK) time stepper.
//!
//! Binding notes (all locked by `arkode_impl.rs`, the frozen contract):
//!
//! * The stepper content record `ARKodeERKStepMemRec` lives BY VALUE in
//!   `ark_mem.step_mem` (`Option<Box<dyn Any>>` = C `void* step_mem`) and is
//!   reached through [`erkStep_mem_mut`], the module's single downcast
//!   helper. The returned guard IS a `borrow_mut()` of the mem: it is never
//!   held across `arkProcessError`, a user callback, an N_Vector operation,
//!   or another borrow of the same mem — every such site copies the fields
//!   it needs into locals in a scoped block, drops the guard, and then
//!   calls.
//! * C `erkStep_AccessStepMem(ark_mem, fname, &step_mem)` /
//!   `erkStep_AccessARKODEStepMem(arkode_mem, fname, &ark_mem, &step_mem)`
//!   become a presence check (`step_mem.is_some()`, else `MSG_ERKSTEP_NO_MEM`
//!   + `ARK_MEM_NULL`) followed by `erkStep_mem_mut(...)` at each use site;
//!   the `arkode_mem == NULL` half is handled by the type system.
//! * The reusable fused-operation arrays are `cvals: Vec<sunrealtype>` and
//!   `Xvecs: Vec<Option<N_Vector>>` — a `calloc`'d `N_Vector*` array holds
//!   NULL slots, which `Vec<N_Vector>` cannot represent. An EMPTY `Vec` is C
//!   `NULL` for every array field here (`F`, `cvals`, `Xvecs`, `forcing`,
//!   `stage_times`, `stage_coefs`), exactly as an empty `d` means "no
//!   embedding" for a Butcher table.
//! * `SUNDIALS_LOGGING_LEVEL=2`: every `SUNLogInfo`/`SUNLogDebug`/
//!   `SUNLogExtraDebug*` call compiles away and is omitted here.
//!   `arkProcessError(..., ARK_WARNING, ...)` still reaches the logger.
//!
//! * The discrete-adjoint cluster (`erkStep_TakeStep_Adjoint`,
//!   `erkStep_fe_Adj`, `erkStep_SUNStepperReInit`,
//!   `ERKStepCreateAdjointStepper`; upstream lines 1043-1943) IS translated
//!   here, and `erkStep_Init` selects `erkStep_TakeStep_Adjoint` when
//!   `ark_mem.do_adjoint` is set exactly as upstream `:518` does. The flag is
//!   only ever set by `ERKStepCreateAdjointStepper`. No reference example
//!   exercises the path (see ARCHITECTURE.md item 12).

use std::any::Any;
use std::cell::RefMut;

use sundials_core::nvector_manyvector::N_VGetSubvector_ManyVector;
use sundials_core::sundials_adjointcheckpointscheme::{
    SUNAdjointCheckpointScheme_InsertVector, SUNAdjointCheckpointScheme_LoadVector,
    SUNAdjointCheckpointScheme_NeedsSaving,
};
use sundials_core::sundials_adjointstepper::{
    SUNAdjRhsFn, SUNAdjointStepper, SUNAdjointStepper_Create, SUNAdjointStepper_RecomputeFwd,
    SUNAdjointStepper_SetUserData,
};
use sundials_core::sundials_context::SUNContext;
use sundials_core::sundials_errors::{SUN_ERR_CHECKPOINT_NOT_FOUND, SUN_ERR_OP_FAIL, SUN_SUCCESS};
use sundials_core::sundials_math::SUNRabs;
use sundials_core::sundials_nvector::{
    N_VConst, N_VDotProd, N_VDotProdLocal, N_VDotProdMultiAllReduce, N_VGetVectorID,
    N_VLinearCombination, N_VScale, N_VSpace, N_VWrmsNorm, N_Vector, SUNDIALS_NVEC_MANYVECTOR,
};
use sundials_core::sundials_stepper::{
    SUNStepper, SUNStepper_GetContentAs, SUNStepper_SetDestroyFn, SUNStepper_SetReInitFn,
};
use sundials_core::sundials_types::*;
use sundials_core::sundials_utils::SUNFile;

use crate::arkode::{
    arkAllocVecArray, arkCreate, arkEwtSetSmallReal, arkFreeVecArray, arkInit, arkResizeVec,
    ARKodeFree,
};
use crate::arkode_butcher::{
    ARKodeButcherTable, ARKodeButcherTable_IsStifflyAccurate, ARKodeButcherTable_Space,
    ARKodeButcherTable_Write,
};
use crate::arkode_butcher_erk::{
    ARKodeButcherTable_LoadERK, ARKODE_BOGACKI_SHAMPINE_4_2_3, ARKODE_FORWARD_EULER_1_1,
    ARKODE_RALSTON_3_1_2, ARKODE_SOFRONIOU_SPALETTA_5_3_4, ARKODE_TSITOURAS_7_4_5,
    ARKODE_VERNER_10_6_7, ARKODE_VERNER_13_7_8, ARKODE_VERNER_16_8_9, ARKODE_VERNER_9_5_6,
};
use crate::arkode_impl::*;
use crate::arkode_io::{
    ARKodeGetNumSteps, ARKodeSetAdjointCheckpointScheme, ARKodeSetFixedStep, ARKodeSetMaxNumSteps,
    ARKodeSetUserData,
};
use crate::arkode_sunstepper::{arkSUNStepperSelfDestruct, ARKodeCreateSUNStepper};

/*===============================================================
  ERKStep Constants (include/arkode/arkode_erkstep.h)
  ===============================================================*/

/* Default Butcher tables for each order (C: `static const int`) */
pub const ERKSTEP_DEFAULT_1: i32 = ARKODE_FORWARD_EULER_1_1;
pub const ERKSTEP_DEFAULT_2: i32 = ARKODE_RALSTON_3_1_2;
pub const ERKSTEP_DEFAULT_3: i32 = ARKODE_BOGACKI_SHAMPINE_4_2_3;
pub const ERKSTEP_DEFAULT_4: i32 = ARKODE_SOFRONIOU_SPALETTA_5_3_4;
pub const ERKSTEP_DEFAULT_5: i32 = ARKODE_TSITOURAS_7_4_5;
pub const ERKSTEP_DEFAULT_6: i32 = ARKODE_VERNER_9_5_6;
pub const ERKSTEP_DEFAULT_7: i32 = ARKODE_VERNER_10_6_7;
pub const ERKSTEP_DEFAULT_8: i32 = ARKODE_VERNER_13_7_8;
pub const ERKSTEP_DEFAULT_9: i32 = ARKODE_VERNER_16_8_9;

/*===============================================================
  Reusable ERKStep Error Messages (arkode_erkstep_impl.h)
  ===============================================================*/

/* Initialization and I/O error messages */
pub const MSG_ERKSTEP_NO_MEM: &str = "Time step module memory is NULL.";

/*===============================================================
  ERK time step module data structure (arkode_erkstep_impl.h)
  ===============================================================*/

/// C `struct ARKodeERKStepMemRec`; the C `ARKodeERKStepMem` pointer typedef
/// has no Rust counterpart — the record is owned by `ark_mem.step_mem` and
/// reached through [`erkStep_mem_mut`].
pub struct ARKodeERKStepMemRec {
    /* ERK problem specification */
    pub f: Option<ARKRhsFn>, /* y' = f(t,y)                */

    /* Adjoint problem specification */
    pub adj_f: Option<SUNAdjRhsFn>,

    /* ARK method storage and parameters */
    pub F: Vec<N_Vector>,              /* explicit RHS at each stage */
    pub q: i32,                        /* method order               */
    pub p: i32,                        /* embedding order            */
    pub istage: i32,                   /* current stage              */
    pub stages: i32,                   /* number of stages           */
    pub B: Option<ARKodeButcherTable>, /* ERK Butcher table          */

    /* Counters */
    pub nfe: i64, /* num fe calls               */

    /* Reusable arrays for fused vector operations */
    pub cvals: Vec<sunrealtype>,
    pub Xvecs: Vec<Option<N_Vector>>,
    pub nfusedopvecs: i32, /* length of cvals and Xvecs arrays */

    /* Data for using ERKStep with external polynomial forcing */
    pub tshift: sunrealtype,           /* time normalization shift       */
    pub tscale: sunrealtype,           /* time normalization scaling     */
    pub forcing: Vec<N_Vector>,        /* array of forcing vectors       */
    pub nforcing: i32,                 /* number of forcing vectors      */
    pub stage_times: Vec<sunrealtype>, /* workspace for applying forcing */
    pub stage_coefs: Vec<sunrealtype>, /* workspace for applying forcing */
}

impl ARKodeERKStepMemRec {
    /// C `malloc` + `memset(step_mem, 0, sizeof(struct ARKodeERKStepMemRec))`
    /// in `ERKStepCreate`.
    pub fn zeroed() -> ARKodeERKStepMemRec {
        ARKodeERKStepMemRec {
            f: None,
            adj_f: None,
            F: Vec::new(),
            q: 0,
            p: 0,
            istage: 0,
            stages: 0,
            B: None,
            nfe: 0,
            cvals: Vec::new(),
            Xvecs: Vec::new(),
            nfusedopvecs: 0,
            tshift: 0.0,
            tscale: 0.0,
            forcing: Vec::new(),
            nforcing: 0,
            stage_times: Vec::new(),
            stage_coefs: Vec::new(),
        }
    }
}

/// Downcast helper: view `ark_mem.step_mem` as the ERKStep memory record.
///
/// Panics if no step memory is attached or it is not an ERKStep record (C
/// would blindly cast the `void*` — UB maps to a deterministic panic).
/// NEVER hold the returned guard across `arkProcessError`, a user callback,
/// an N_Vector operation, or another borrow of the same `ark_mem`.
pub fn erkStep_mem_mut(ark_mem: &ARKodeMem) -> RefMut<'_, ARKodeERKStepMemRec> {
    RefMut::map(ark_mem.borrow_mut(), |m| {
        m.step_mem
            .as_mut()
            .expect("step_mem set")
            .downcast_mut::<ARKodeERKStepMemRec>()
            .expect("ERKStep step memory")
    })
}

/*===============================================================
  Local helpers for the C aliasing idioms
  ===============================================================*/

/// The first `nvec` entries of the reusable `Xvecs` array. C hands
/// `N_VLinearCombination` the whole `N_Vector*` block and the callee reads
/// only `[0, nvec)`.
fn erkStep_Xvecs(Xvecs: &[Option<N_Vector>], nvec: i32) -> Vec<N_Vector> {
    Xvecs[..nvec as usize]
        .iter()
        .map(|v| v.clone().expect("Xvecs entry set"))
        .collect()
}

/// C `N_VLinearCombination(nvec, step_mem->cvals, step_mem->Xvecs, z)`: the
/// reusable arrays live in `step_mem`, so `cvals` is moved out (and restored
/// on every path) instead of borrowed across the vector operation, which
/// touches user-visible vectors.
fn erkStep_LinearCombination(ark_mem: &ARKodeMem, nvec: i32, z: &N_Vector) -> i32 {
    let (cvals, Xvecs) = {
        let mut step_mem = erkStep_mem_mut(ark_mem);
        let Xvecs = erkStep_Xvecs(&step_mem.Xvecs, nvec);
        (std::mem::take(&mut step_mem.cvals), Xvecs)
    };
    let retval = N_VLinearCombination(nvec, &cvals, &Xvecs, z);
    erkStep_mem_mut(ark_mem).cvals = cvals;
    retval
}

/// Invoke the user RHS function (C:
/// `step_mem->f(t, y, ydot, ark_mem->user_data)`). The `user_data` box is
/// taken out of the mem for the duration of the call and restored on every
/// path; no borrow is held across it. The caller increments `nfe`, exactly
/// as at each C call site.
fn erkStep_call_f(ark_mem: &ARKodeMem, t: sunrealtype, y: &N_Vector, ydot: &N_Vector) -> i32 {
    let f = erkStep_mem_mut(ark_mem).f.expect("step_mem->f set");
    let mut user_data = ark_mem.borrow_mut().user_data.take();
    let retval = f(t, y, ydot, &mut user_data);
    ark_mem.borrow_mut().user_data = user_data;
    retval
}

/// C `ark_mem->PreRhsFn(t, y, ark_mem->user_data)` (presence checked by the
/// caller, which holds no borrow).
fn erkStep_call_prerhsfn(
    ark_mem: &ARKodeMem,
    PreRhsFn: ARKPreRhsFn,
    t: sunrealtype,
    y: &N_Vector,
) -> i32 {
    let mut user_data = ark_mem.borrow_mut().user_data.take();
    let retval = PreRhsFn(t, y, &mut user_data);
    ark_mem.borrow_mut().user_data = user_data;
    retval
}

/// C `ark_mem->PostProcess{Step,Stage}Fn(t, y, ark_mem->user_data)`.
fn erkStep_call_postprocessfn(
    ark_mem: &ARKodeMem,
    PostProcessFn: ARKPostProcessFn,
    t: sunrealtype,
    y: &N_Vector,
) -> i32 {
    let mut user_data = ark_mem.borrow_mut().user_data.take();
    let retval = PostProcessFn(t, y, &mut user_data);
    ark_mem.borrow_mut().user_data = user_data;
    retval
}

/*===============================================================
  Exported functions
  ===============================================================*/

/// C `void* ERKStepCreate(ARKRhsFn f, sunrealtype t0, N_Vector y0,
/// SUNContext sunctx)`. The `f == NULL`, `y0 == NULL` and `!sunctx` checks
/// are handled by the type system.
pub fn ERKStepCreate(
    f: ARKRhsFn,
    t0: sunrealtype,
    y0: &N_Vector,
    sunctx: &SUNContext,
) -> Option<ARKodeMem> {
    /* Check that f is supplied: handled by the type system */

    /* Check for legal input parameters: handled by the type system */

    /* Create ark_mem structure and set default values */
    let ark_mem = match arkCreate(sunctx) {
        Some(ark_mem) => ark_mem,
        None => {
            arkProcessError(
                None,
                ARK_MEM_NULL,
                line!() as i32,
                "ERKStepCreate",
                file!(),
                MSG_ARK_NO_MEM,
            );
            return None;
        }
    };

    /* Allocate ARKodeERKStepMem structure, and initialize to zero
       (the C malloc-failure branch cannot be observed in Rust) */
    let step_mem = ARKodeERKStepMemRec::zeroed();

    /* Attach step_mem structure and function pointers to ark_mem */
    {
        let mut m = ark_mem.borrow_mut();
        m.step_init = Some(erkStep_Init);
        m.step_fullrhs = Some(erkStep_FullRHS);
        m.step = Some(erkStep_TakeStep);
        m.step_printallstats = Some(crate::arkode_erkstep_io::erkStep_PrintAllStats);
        m.step_writeparameters = Some(crate::arkode_erkstep_io::erkStep_WriteParameters);
        m.step_setusecompensatedsums = None;
        m.step_resize = Some(erkStep_Resize);
        m.step_free = Some(erkStep_Free);
        m.step_printmem = Some(erkStep_PrintMem);
        m.step_setoptions = Some(crate::arkode_erkstep_io::erkStep_SetOptions);
        m.step_setdefaults = Some(crate::arkode_erkstep_io::erkStep_SetDefaults);
        m.step_setrelaxfn = Some(crate::arkode_erkstep_io::erkStep_SetRelaxFn);
        m.step_setorder = Some(crate::arkode_erkstep_io::erkStep_SetOrder);
        m.step_getnumrhsevals = Some(crate::arkode_erkstep_io::erkStep_GetNumRhsEvals);
        m.step_getestlocalerrors = Some(crate::arkode_erkstep_io::erkStep_GetEstLocalErrors);
        m.step_setforcing = Some(erkStep_SetInnerForcing);
        m.step_getstageindex = Some(crate::arkode_erkstep_io::erkStep_GetStageIndex);
        m.step_supports_adaptive = SUNTRUE;
        m.step_supports_relaxation = SUNTRUE;
        m.step_mem = Some(Box::new(step_mem));
    }

    /* Set default values for optional inputs */
    let retval = crate::arkode_erkstep_io::erkStep_SetDefaults(&ark_mem);
    if retval != ARK_SUCCESS {
        arkProcessError(
            Some(&ark_mem),
            retval,
            line!() as i32,
            "ERKStepCreate",
            file!(),
            "Error setting default solver options",
        );
        let mut arkode_mem = Some(ark_mem);
        ARKodeFree(&mut arkode_mem);
        return None;
    }

    /* Allocate the general ERK stepper vectors using y0 as a template */
    /* NOTE: F, cvals and Xvecs will be allocated later on
       (based on the number of ERK stages) */

    /* Copy the input parameters into ARKODE state */
    erkStep_mem_mut(&ark_mem).f = Some(f);

    /* Update the ARKODE workspace requirements -- UPDATE */
    {
        let mut m = ark_mem.borrow_mut();
        m.liw += 41; /* fcn/data ptr, int, long int, sunindextype, sunbooleantype */
        m.lrw += 10;
    }

    /* Initialize all the counters */
    erkStep_mem_mut(&ark_mem).nfe = 0;

    /* Initialize fused op work space */
    {
        let mut step_mem = erkStep_mem_mut(&ark_mem);
        step_mem.cvals = Vec::new();
        step_mem.Xvecs = Vec::new();
        step_mem.nfusedopvecs = 0;
    }

    /* Initialize external polynomial forcing data */
    {
        let mut step_mem = erkStep_mem_mut(&ark_mem);
        step_mem.forcing = Vec::new();
        step_mem.nforcing = 0;
    }

    /* Initialize main ARKODE infrastructure */
    let retval = arkInit(&ark_mem, t0, y0, FIRST_INIT);
    if retval != ARK_SUCCESS {
        arkProcessError(
            Some(&ark_mem),
            retval,
            line!() as i32,
            "ERKStepCreate",
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
  ERKStepReInit:

  This routine re-initializes the ERKStep module to solve a new
  problem of the same size as was previously solved. This routine
  should also be called when the problem dynamics or desired solvers
  have changed dramatically, so that the problem integration should
  resume as if started from scratch.

  Note all internal counters are set to 0 on re-initialization.
  ---------------------------------------------------------------*/
pub fn ERKStepReInit(
    arkode_mem: &ARKodeMem,
    f: ARKRhsFn,
    t0: sunrealtype,
    y0: &N_Vector,
) -> i32 {
    /* access ARKodeERKStepMem structure */
    if arkode_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(arkode_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "ERKStepReInit",
            file!(),
            MSG_ERKSTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }
    let ark_mem = arkode_mem;

    /* Check if ark_mem was allocated */
    if !ark_mem.borrow().MallocDone {
        arkProcessError(
            Some(ark_mem),
            ARK_NO_MALLOC,
            line!() as i32,
            "ERKStepReInit",
            file!(),
            MSG_ARK_NO_MALLOC,
        );
        return ARK_NO_MALLOC;
    }

    /* Check that f is supplied: handled by the type system */

    /* Check for legal input parameters: handled by the type system */

    /* Copy the input parameters into ARKODE state */
    erkStep_mem_mut(ark_mem).f = Some(f);

    /* Initialize main ARKODE infrastructure */
    let retval = arkInit(arkode_mem, t0, y0, FIRST_INIT);
    if retval != ARK_SUCCESS {
        arkProcessError(
            Some(ark_mem),
            retval,
            line!() as i32,
            "ERKStepReInit",
            file!(),
            "Unable to initialize main ARKODE infrastructure",
        );
        return retval;
    }

    /* Initialize all the counters */
    erkStep_mem_mut(ark_mem).nfe = 0;

    ARK_SUCCESS
}

/*===============================================================
  Interface routines supplied to ARKODE
  ===============================================================*/

/*---------------------------------------------------------------
  erkStep_Resize:

  This routine resizes the memory within the ERKStep module.
  ---------------------------------------------------------------*/
pub fn erkStep_Resize(
    ark_mem: &ARKodeMem,
    y0: &N_Vector,
    hscale: sunrealtype,
    t0: sunrealtype,
    resize: Option<ARKVecResizeFn>,
    resize_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    /* C: SUNDIALS_MAYBE_UNUSED hscale, t0 */
    let _ = hscale;
    let _ = t0;

    /* access ARKodeERKStepMem structure */
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "erkStep_Resize",
            file!(),
            MSG_ERKSTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    /* Determine change in vector sizes */
    let mut lrw1: sunindextype = 0;
    let mut liw1: sunindextype = 0;
    if y0.ops.borrow().nvspace.is_some() {
        N_VSpace(y0, &mut lrw1, &mut liw1);
    }
    let (lrw_diff, liw_diff) = {
        let mut m = ark_mem.borrow_mut();
        let lrw_diff = lrw1 - m.lrw1;
        let liw_diff = liw1 - m.liw1;
        m.lrw1 = lrw1;
        m.liw1 = liw1;
        (lrw_diff, liw_diff)
    };

    /* Resize the RHS vectors */
    if !erkStep_mem_mut(ark_mem).F.is_empty() {
        let stages = erkStep_mem_mut(ark_mem).stages;
        let mut F = std::mem::take(&mut erkStep_mem_mut(ark_mem).F);
        for i in 0..stages {
            let mut v: Option<N_Vector> = Some(F[i as usize].clone());
            let ok = arkResizeVec(
                ark_mem,
                resize,
                resize_data,
                lrw_diff,
                liw_diff,
                y0,
                &mut v,
            );
            /* on failure C leaves `F[i]` NULL; a Vec cannot hold NULL, so the
               previous handle stays in place -- unobservable, the caller
               propagates ARK_MEM_FAIL */
            if let Some(v) = v {
                F[i as usize] = v;
            }
            if !ok {
                erkStep_mem_mut(ark_mem).F = F;
                arkProcessError(
                    Some(ark_mem),
                    ARK_MEM_FAIL,
                    line!() as i32,
                    "erkStep_Resize",
                    file!(),
                    "Unable to resize vector",
                );
                return ARK_MEM_FAIL;
            }
        }
        erkStep_mem_mut(ark_mem).F = F;
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  erkStep_Free frees all ERKStep memory.
  ---------------------------------------------------------------*/
pub fn erkStep_Free(ark_mem: &ARKodeMem) {
    /* nothing to do if ark_mem is already NULL: handled by the type system */

    /* conditional frees on non-NULL ERKStep module */
    if ark_mem.borrow().step_mem.is_none() {
        return;
    }

    /* free the Butcher table */
    let B = erkStep_mem_mut(ark_mem).B.clone();
    if B.is_some() {
        let mut Bliw: sunindextype = 0;
        let mut Blrw: sunindextype = 0;
        ARKodeButcherTable_Space(B.as_ref(), &mut Bliw, &mut Blrw);
        /* C: ARKodeButcherTable_Free(step_mem->B); step_mem->B = NULL; */
        erkStep_mem_mut(ark_mem).B = None;
        drop(B);
        let mut m = ark_mem.borrow_mut();
        m.liw -= Bliw;
        m.lrw -= Blrw;
    }

    /* free the RHS vectors */
    if !erkStep_mem_mut(ark_mem).F.is_empty() {
        let stages = erkStep_mem_mut(ark_mem).stages;
        let mut F = std::mem::take(&mut erkStep_mem_mut(ark_mem).F);
        let (lrw1, liw1, mut lrw, mut liw) = {
            let m = ark_mem.borrow();
            (m.lrw1, m.liw1, m.lrw, m.liw)
        };
        arkFreeVecArray(stages, &mut F, lrw1, &mut lrw, liw1, &mut liw);
        {
            let mut m = ark_mem.borrow_mut();
            m.lrw = lrw;
            m.liw = liw;
        }
        erkStep_mem_mut(ark_mem).F = F;
    }

    /* free the reusable arrays for fused vector interface */
    let nfusedopvecs = erkStep_mem_mut(ark_mem).nfusedopvecs;
    if !erkStep_mem_mut(ark_mem).cvals.is_empty() {
        erkStep_mem_mut(ark_mem).cvals = Vec::new();
        ark_mem.borrow_mut().lrw -= nfusedopvecs as i64;
    }
    if !erkStep_mem_mut(ark_mem).Xvecs.is_empty() {
        erkStep_mem_mut(ark_mem).Xvecs = Vec::new();
        ark_mem.borrow_mut().liw -= nfusedopvecs as i64;
    }
    erkStep_mem_mut(ark_mem).nfusedopvecs = 0;

    /* free work arrays for MRI forcing */
    let stages = erkStep_mem_mut(ark_mem).stages;
    if !erkStep_mem_mut(ark_mem).stage_times.is_empty() {
        erkStep_mem_mut(ark_mem).stage_times = Vec::new();
        ark_mem.borrow_mut().lrw -= stages as i64;
    }

    if !erkStep_mem_mut(ark_mem).stage_coefs.is_empty() {
        erkStep_mem_mut(ark_mem).stage_coefs = Vec::new();
        ark_mem.borrow_mut().lrw -= stages as i64;
    }

    /* free the time stepper module itself */
    ark_mem.borrow_mut().step_mem = None;
}

/*---------------------------------------------------------------
  erkStep_PrintMem:

  This routine outputs the memory from the ERKStep structure to
  a specified file pointer (useful when debugging).
  ---------------------------------------------------------------*/
pub fn erkStep_PrintMem(ark_mem: &ARKodeMem, outfile: &SUNFile) {
    /* access ARKodeERKStepMem structure */
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "erkStep_PrintMem",
            file!(),
            MSG_ERKSTEP_NO_MEM,
        );
        return;
    }

    let (q, p, istage, stages, nfe, B) = {
        let step_mem = erkStep_mem_mut(ark_mem);
        (
            step_mem.q,
            step_mem.p,
            step_mem.istage,
            step_mem.stages,
            step_mem.nfe,
            step_mem.B.clone(),
        )
    };

    /* output integer quantities */
    outfile.write_str(&format!("ERKStep: q = {q}\n"));
    outfile.write_str(&format!("ERKStep: p = {p}\n"));
    outfile.write_str(&format!("ERKStep: istage = {istage}\n"));
    outfile.write_str(&format!("ERKStep: stages = {stages}\n"));

    /* output long integer quantities */
    outfile.write_str(&format!("ERKStep: nfe = {nfe}\n"));

    /* output sunrealtype quantities */
    outfile.write_str("ERKStep: Butcher table:\n");
    ARKodeButcherTable_Write(B.as_ref(), outfile);

    /* SUNDIALS_DEBUG_PRINTVEC is not defined in the reference build */
}

/*---------------------------------------------------------------
  erkStep_Init:

  This routine is called just prior to performing internal time
  steps (after all user "set" routines have been called) from
  within arkInitialSetup.

  With initialization types FIRST_INIT this routine:
  - sets/checks the ARK Butcher tables to be used
  - allocates any memory that depends on the number of
    stages, method order, or solver options
  - sets the call_fullrhs flag

  With other initialization types, this routine does nothing.
  ---------------------------------------------------------------*/
pub fn erkStep_Init(ark_mem: &ARKodeMem, init_type: i32) -> i32 {
    /* access ARKodeERKStepMem structure */
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "erkStep_Init",
            file!(),
            MSG_ERKSTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    /* immediately return if resize or reset */
    if init_type == RESIZE_INIT || init_type == RESET_INIT {
        return ARK_SUCCESS;
    }

    /* enforce use of arkEwtSmallReal if using a fixed step size,
       an internal error weight function, and not performing accumulated
       temporal error estimation */
    let mut reset_efun: sunbooleantype = SUNTRUE;
    {
        let m = ark_mem.borrow();
        if !m.fixedstep {
            reset_efun = SUNFALSE;
        }
        if m.user_efun {
            reset_efun = SUNFALSE;
        }
        if m.AccumErrorType != ARK_ACCUMERROR_NONE {
            reset_efun = SUNFALSE;
        }
    }
    if reset_efun {
        /* C `ark_mem->e_data = ark_mem`: the token is a handle clone, exactly
           as CVODE's default-efun `cv_e_data` (ARKodeFree clears the field to
           break the Rc cycle). `arkEwtSetSmallReal` never reads it. */
        let e_data: Box<dyn Any> = Box::new(ark_mem.clone());
        let mut m = ark_mem.borrow_mut();
        m.user_efun = SUNFALSE;
        m.efun = Some(arkEwtSetSmallReal);
        m.e_data = Some(e_data);
    }

    /* Create Butcher table (if not already set) */
    let retval = erkStep_SetButcherTable(ark_mem);
    if retval != ARK_SUCCESS {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "erkStep_Init",
            file!(),
            "Could not create Butcher table",
        );
        return ARK_ILL_INPUT;
    }

    /* Check that Butcher table are OK */
    let retval = erkStep_CheckButcherTable(ark_mem);
    if retval != ARK_SUCCESS {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "erkStep_Init",
            file!(),
            "Error in Butcher table",
        );
        return ARK_ILL_INPUT;
    }

    /* Retrieve/store method and embedding orders now that table is finalized */
    let B = erkStep_mem_mut(ark_mem).B.clone().expect("Butcher table set");
    let (Bq, Bp) = {
        let B = B.borrow();
        (B.q, B.p)
    };
    {
        let mut m = ark_mem.borrow_mut();
        let hadapt_mem = m.hadapt_mem.as_mut().expect("hadapt_mem set");
        hadapt_mem.q = Bq;
        hadapt_mem.p = Bp;
    }
    {
        let mut step_mem = erkStep_mem_mut(ark_mem);
        step_mem.q = Bq;
        step_mem.p = Bp;
    }

    /* Ensure that if adaptivity or error accumulation is enabled, then
         method includes embedding coefficients */
    let (fixedstep, AccumErrorType) = {
        let m = ark_mem.borrow();
        (m.fixedstep, m.AccumErrorType)
    };
    let p = erkStep_mem_mut(ark_mem).p;
    if (!fixedstep || (AccumErrorType != ARK_ACCUMERROR_NONE)) && (p == 0) {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "erkStep_Init",
            file!(),
            "Temporal error estimation cannot be performed without embedding coefficients",
        );
        return ARK_ILL_INPUT;
    }

    /* Allocate RHS vector memory, update storage requirements */
    /*   Allocate F[0] ... F[stages-1] if needed */
    let stages = erkStep_mem_mut(ark_mem).stages;
    let (ewt, lrw1, liw1, mut lrw, mut liw) = {
        let m = ark_mem.borrow();
        (
            m.ewt.clone().expect("ewt set"),
            m.lrw1,
            m.liw1,
            m.lrw,
            m.liw,
        )
    };
    let mut F = std::mem::take(&mut erkStep_mem_mut(ark_mem).F);
    let ok = arkAllocVecArray(stages, &ewt, &mut F, lrw1, &mut lrw, liw1, &mut liw);
    {
        let mut m = ark_mem.borrow_mut();
        m.lrw = lrw;
        m.liw = liw;
    }
    erkStep_mem_mut(ark_mem).F = F;
    if !ok {
        return ARK_MEM_FAIL;
    }

    /* Allocate reusable arrays for fused vector interface
       (the C calloc-failure branches cannot be observed in Rust) */
    let nforcing = erkStep_mem_mut(ark_mem).nforcing;
    let nfusedopvecs = 2 * stages + 2 + nforcing;
    erkStep_mem_mut(ark_mem).nfusedopvecs = nfusedopvecs;
    if erkStep_mem_mut(ark_mem).cvals.is_empty() {
        erkStep_mem_mut(ark_mem).cvals = vec![ZERO; nfusedopvecs as usize];
        ark_mem.borrow_mut().lrw += nfusedopvecs as i64;
    }
    if erkStep_mem_mut(ark_mem).Xvecs.is_empty() {
        erkStep_mem_mut(ark_mem).Xvecs = vec![None; nfusedopvecs as usize];
        ark_mem.borrow_mut().liw += nfusedopvecs as i64; /* pointers */
    }

    /* Allocate workspace for MRI forcing -- need to allocate here as the
       number of stages may not be set before this point */
    if erkStep_mem_mut(ark_mem).stage_times.is_empty() {
        erkStep_mem_mut(ark_mem).stage_times = vec![ZERO; stages as usize];
        ark_mem.borrow_mut().lrw += stages as i64;
    }

    if erkStep_mem_mut(ark_mem).stage_coefs.is_empty() {
        erkStep_mem_mut(ark_mem).stage_coefs = vec![ZERO; stages as usize];
        ark_mem.borrow_mut().lrw += stages as i64;
    }

    /* Override the interpolant degree (if needed), used in arkInitialSetup */
    let q = erkStep_mem_mut(ark_mem).q;
    {
        let mut m = ark_mem.borrow_mut();
        if q > 1 && m.interp_degree > (q - 1) {
            /* Limit max degree to at most one less than the method global order */
            m.interp_degree = q - 1;
        } else if q == 1 && m.interp_degree > 1 {
            /* Allow for linear interpolant with first order methods to ensure
               solution values are returned at the time interval end points */
            m.interp_degree = 1;
        }
    }

    /* set appropriate TakeStep routine based on problem configuration */
    {
        let mut m = ark_mem.borrow_mut();
        if m.do_adjoint {
            m.step = Some(erkStep_TakeStep_Adjoint);
        } else {
            m.step = Some(erkStep_TakeStep);
        }
    }

    /* Signal to shared arkode module that full RHS evaluations are required */
    ark_mem.borrow_mut().call_fullrhs = SUNTRUE;

    ARK_SUCCESS
}

/*------------------------------------------------------------------------------
  erkStep_FullRHS:

  This is just a wrapper to call the user-supplied RHS function, f(t,y).

  This will be called in one of three 'modes':

     ARK_FULLRHS_START -> called in the following circumstances:
                          (a) at the beginning of a simulation i.e., at
                              (tn, yn) = (t0, y0) or (tR, yR),
                          (b) when transitioning between time steps t_{n-1}
                              \to t_{n} to fill f_{n-1} within the Hermite
                              interpolation module, or
                          (c) by ERKStep at the start of the first internal step.

                          In each case, we may check the fn_is_current flag to
                          know whether the values stored in F[0] are up-to-date,
                          allowing us to copy those values instead of recomputing.
                          If these values are not current, then the RHS should be
                          stored in F[0] for reuse later, before copying the values
                          into the output vector.

     ARK_FULLRHS_END   -> called in the following circumstances:
                          (a) when temporal root-finding is enabled, this will be
                              called in-between steps t_{n-1} \to t_{n} to fill f_{n},
                          (b) when high-order dense output is requested from the
                              Hermite interpolation module in-between steps t_{n-1}
                              \to t_{n} to fill f_{n}, or
                          (c) by ERKStep when starting a time step t_{n} \to t_{n+1}
                              and when using an FSAL method.

                          Again, we may check the fn_is_current flag to know whether
                          ARKODE believes that the values stored in F[0] are
                          up-to-date, and may just be copied.  If the values stored
                          in F[0] are not current, then the only instance where
                          recomputation is not needed is (c), since the values in
                          F[stages - 1] may be copied into F[0].  In all other cases,
                          the RHS should be recomputed and stored in F[0] for reuse
                          later, before copying the values into the output vector.

     ARK_FULLRHS_OTHER -> called in the following circumstances:
                          (a) when estimating the initial time step size,
                          (b) for high-order dense output with the Hermite
                              interpolation module, or
                          (c) by an "outer" stepper when ERKStep is used as an
                              inner solver).

                          All of these instances will occur in-between ERKStep time
                          steps, but the (t,y) input does not correspond to an
                          "official" time step, thus the RHS should always be
                          evaluated, with the values *not* stored in F[0].
  ----------------------------------------------------------------------------*/
pub fn erkStep_FullRHS(
    ark_mem: &ARKodeMem,
    t: sunrealtype,
    y: &N_Vector,
    f: &N_Vector,
    mode: i32,
) -> i32 {
    let stage_coefs: sunrealtype = ONE;

    /* access ARKodeERKStepMem structure */
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "erkStep_FullRHS",
            file!(),
            MSG_ERKSTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    /* local shortcuts for use with fused vector operations: `cvals` and
       `Xvecs` live in step_mem and are reached through erkStep_mem_mut */

    /* perform RHS functions contingent on 'mode' argument */
    match mode {
        ARK_FULLRHS_START => {
            /* compute the RHS if needed */
            if !ark_mem.borrow().fn_is_current {
                /* call the user-supplied pre-RHS function (if supplied) */
                let PreRhsFn = ark_mem.borrow().PreRhsFn;
                if let Some(PreRhsFn) = PreRhsFn {
                    let retval = erkStep_call_prerhsfn(ark_mem, PreRhsFn, t, y);
                    if retval != 0 {
                        return ARK_PRERHSFN_FAIL;
                    }
                }

                /* call f */
                let F0 = erkStep_mem_mut(ark_mem).F[0].clone();
                let retval = erkStep_call_f(ark_mem, t, y, &F0);
                erkStep_mem_mut(ark_mem).nfe += 1;
                if retval != 0 {
                    arkProcessError(
                        Some(ark_mem),
                        ARK_RHSFUNC_FAIL,
                        line!() as i32,
                        "erkStep_FullRHS",
                        file!(),
                        &MSG_ARK_RHSFUNC_FAILED(t),
                    );
                    return ARK_RHSFUNC_FAIL;
                }
            }

            /* copy RHS into output */
            let F0 = erkStep_mem_mut(ark_mem).F[0].clone();
            N_VScale(ONE, &F0, f);

            /* apply external polynomial forcing */
            if erkStep_mem_mut(ark_mem).nforcing > 0 {
                let mut nvec: i32;
                {
                    let mut step_mem = erkStep_mem_mut(ark_mem);
                    step_mem.cvals[0] = ONE;
                    step_mem.Xvecs[0] = Some(f.clone());
                    nvec = 1;
                    erkStep_ApplyForcing(
                        &mut step_mem,
                        std::slice::from_ref(&t),
                        std::slice::from_ref(&stage_coefs),
                        1,
                        &mut nvec,
                    );
                }
                let _ = erkStep_LinearCombination(ark_mem, nvec, f);
            }
        }

        ARK_FULLRHS_END => {
            /* determine if RHS function needs to be recomputed */
            if !ark_mem.borrow().fn_is_current {
                let B = erkStep_mem_mut(ark_mem).B.clone().expect("Butcher table set");
                let mut recomputeRHS = !ARKodeButcherTable_IsStifflyAccurate(Some(&B));

                /* First Same As Last methods are not FSAL when relaxation is enabled */
                if ark_mem.borrow().relax_enabled {
                    recomputeRHS = SUNTRUE;
                }

                /* base RHS call on recomputeRHS argument */
                if recomputeRHS {
                    /* call the user-supplied pre-RHS function (if supplied) */
                    let PreRhsFn = ark_mem.borrow().PreRhsFn;
                    if let Some(PreRhsFn) = PreRhsFn {
                        let retval = erkStep_call_prerhsfn(ark_mem, PreRhsFn, t, y);
                        if retval != 0 {
                            return ARK_PRERHSFN_FAIL;
                        }
                    }

                    /* call f */
                    let F0 = erkStep_mem_mut(ark_mem).F[0].clone();
                    let retval = erkStep_call_f(ark_mem, t, y, &F0);
                    erkStep_mem_mut(ark_mem).nfe += 1;
                    if retval != 0 {
                        arkProcessError(
                            Some(ark_mem),
                            ARK_RHSFUNC_FAIL,
                            line!() as i32,
                            "erkStep_FullRHS",
                            file!(),
                            &MSG_ARK_RHSFUNC_FAILED(t),
                        );
                        return ARK_RHSFUNC_FAIL;
                    }
                } else {
                    let (Flast, F0) = {
                        let step_mem = erkStep_mem_mut(ark_mem);
                        (
                            step_mem.F[(step_mem.stages - 1) as usize].clone(),
                            step_mem.F[0].clone(),
                        )
                    };
                    N_VScale(ONE, &Flast, &F0);
                }

                /* copy RHS vector into output */
                let F0 = erkStep_mem_mut(ark_mem).F[0].clone();
                N_VScale(ONE, &F0, f);

                /* apply external polynomial forcing */
                if erkStep_mem_mut(ark_mem).nforcing > 0 {
                    let mut nvec: i32;
                    {
                        let mut step_mem = erkStep_mem_mut(ark_mem);
                        step_mem.cvals[0] = ONE;
                        step_mem.Xvecs[0] = Some(f.clone());
                        nvec = 1;
                        erkStep_ApplyForcing(
                            &mut step_mem,
                            std::slice::from_ref(&t),
                            std::slice::from_ref(&stage_coefs),
                            1,
                            &mut nvec,
                        );
                    }
                    let _ = erkStep_LinearCombination(ark_mem, nvec, f);
                }
            }
        }

        ARK_FULLRHS_OTHER => {
            /* call the user-supplied pre-RHS function (if supplied) */
            let PreRhsFn = ark_mem.borrow().PreRhsFn;
            if let Some(PreRhsFn) = PreRhsFn {
                let retval = erkStep_call_prerhsfn(ark_mem, PreRhsFn, t, y);
                if retval != 0 {
                    return ARK_PRERHSFN_FAIL;
                }
            }

            /* call f */
            let retval = erkStep_call_f(ark_mem, t, y, f);
            erkStep_mem_mut(ark_mem).nfe += 1;
            if retval != 0 {
                arkProcessError(
                    Some(ark_mem),
                    ARK_RHSFUNC_FAIL,
                    line!() as i32,
                    "erkStep_FullRHS",
                    file!(),
                    &MSG_ARK_RHSFUNC_FAILED(t),
                );
                return ARK_RHSFUNC_FAIL;
            }
            /* apply external polynomial forcing */
            if erkStep_mem_mut(ark_mem).nforcing > 0 {
                let mut nvec: i32;
                {
                    let mut step_mem = erkStep_mem_mut(ark_mem);
                    step_mem.cvals[0] = ONE;
                    step_mem.Xvecs[0] = Some(f.clone());
                    nvec = 1;
                    erkStep_ApplyForcing(
                        &mut step_mem,
                        std::slice::from_ref(&t),
                        std::slice::from_ref(&stage_coefs),
                        1,
                        &mut nvec,
                    );
                }
                let _ = erkStep_LinearCombination(ark_mem, nvec, f);
            }
        }

        _ => {
            /* return with RHS failure if unknown mode is passed */
            arkProcessError(
                Some(ark_mem),
                ARK_RHSFUNC_FAIL,
                line!() as i32,
                "erkStep_FullRHS",
                file!(),
                "Unknown full RHS mode",
            );
            return ARK_RHSFUNC_FAIL;
        }
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  erkStep_TakeStep:

  This routine serves the primary purpose of the ERKStep module:
  it performs a single ERK step (with embedding, if possible).

  The output variable dsmPtr should contain estimate of the
  weighted local error if an embedding is present; otherwise it
  should be 0.

  The input/output variable nflagPtr is used to gauge convergence
  of any algebraic solvers within the step.  As this routine
  involves no algebraic solve, it is set to 0 (success).

  The return value from this routine is:
            0 => step completed successfully
           >0 => step encountered recoverable failure;
                 reduce step and retry (if possible)
           <0 => step encountered unrecoverable failure
  ---------------------------------------------------------------*/
pub fn erkStep_TakeStep(ark_mem: &ARKodeMem, dsmPtr: &mut sunrealtype, nflagPtr: &mut i32) -> i32 {
    /* initialize algebraic solver convergence flag to success */
    *nflagPtr = ARK_SUCCESS;

    /* access ARKodeERKStepMem structure */
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "erkStep_TakeStep",
            file!(),
            MSG_ERKSTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    /* determine if method has fsal property */
    let B = erkStep_mem_mut(ark_mem).B.clone().expect("Butcher table set");
    let fsal = ARKodeButcherTable_IsStifflyAccurate(Some(&B));

    /* local shortcuts for fused vector operations: `cvals` and `Xvecs` live
       in step_mem and are reached through erkStep_mem_mut */

    /* initialize the current stage index */
    erkStep_mem_mut(ark_mem).istage = 0;

    /* Call the full RHS if needed. If this is the first step then we may need to
       evaluate or copy the RHS values from an earlier evaluation (e.g., to
       compute h0). For subsequent steps treat this RHS evaluation as an
       evaluation at the end of the just completed step to potentially reuse
       (FSAL methods) RHS evaluations from the end of the last step. */

    if !ark_mem.borrow().fn_is_current {
        let (initsetup, step_fullrhs, tn, yn, fn_) = {
            let m = ark_mem.borrow();
            (
                m.initsetup,
                m.step_fullrhs.expect("step_fullrhs set"),
                m.tn,
                m.yn.clone().expect("yn set"),
                m.fn_.clone().expect("fn set"),
            )
        };
        let mode = if initsetup {
            ARK_FULLRHS_START
        } else {
            ARK_FULLRHS_END
        };
        let retval = step_fullrhs(ark_mem, tn, &yn, &fn_, mode);
        if retval != 0 {
            return ARK_RHSFUNC_FAIL;
        }
        ark_mem.borrow_mut().fn_is_current = SUNTRUE;
    }

    let checkpoint_scheme = ark_mem.borrow().checkpoint_scheme.clone();
    if let Some(checkpoint_scheme) = &checkpoint_scheme {
        let mut do_save: sunbooleantype = SUNFALSE;
        let (checkpoint_step_idx, tn, yn) = {
            let m = ark_mem.borrow();
            (m.checkpoint_step_idx, m.tn, m.yn.clone().expect("yn set"))
        };
        let errcode = SUNAdjointCheckpointScheme_NeedsSaving(
            checkpoint_scheme,
            checkpoint_step_idx,
            0,
            tn,
            &mut do_save,
        );
        if errcode != 0 {
            arkProcessError(
                Some(ark_mem),
                ARK_ADJ_CHECKPOINT_FAIL,
                line!() as i32,
                "erkStep_TakeStep",
                file!(),
                &format!("SUNAdjointCheckpointScheme_NeedsSaving returned {errcode}"),
            );
            return ARK_ADJ_CHECKPOINT_FAIL;
        }

        if do_save {
            let errcode = SUNAdjointCheckpointScheme_InsertVector(
                checkpoint_scheme,
                checkpoint_step_idx,
                0,
                tn,
                &yn,
            );

            if errcode != 0 {
                arkProcessError(
                    Some(ark_mem),
                    ARK_ADJ_CHECKPOINT_FAIL,
                    line!() as i32,
                    "erkStep_TakeStep",
                    file!(),
                    &format!("SUNAdjointCheckpointScheme_InsertVector returned {errcode}"),
                );
                return ARK_ADJ_CHECKPOINT_FAIL;
            }
        }
    }

    /* Loop over internal stages to the step; since the method is explicit
       the first stage RHS is just the full RHS from the start of the step */
    let stages = erkStep_mem_mut(ark_mem).stages;
    for is in 1..stages {
        /* Set current stage time and index */
        let (tn, h, yn, ycur) = {
            let m = ark_mem.borrow();
            (
                m.tn,
                m.h,
                m.yn.clone().expect("yn set"),
                m.ycur.clone().expect("ycur set"),
            )
        };
        let c_is = B.borrow().c[is as usize];
        ark_mem.borrow_mut().tcur = tn + c_is * h;
        erkStep_mem_mut(ark_mem).istage = is;

        /* Set ycur to current stage solution */
        let mut nvec: i32 = 0;
        {
            let mut step_mem = erkStep_mem_mut(ark_mem);
            let Bref = B.borrow();
            for js in 0..is {
                step_mem.cvals[nvec as usize] = h * Bref.A[is as usize][js as usize];
                let Fjs = step_mem.F[js as usize].clone();
                step_mem.Xvecs[nvec as usize] = Some(Fjs);
                nvec += 1;
            }
            step_mem.cvals[nvec as usize] = ONE;
            step_mem.Xvecs[nvec as usize] = Some(yn.clone());
            nvec += 1;
        }

        /* apply external polynomial forcing */
        if erkStep_mem_mut(ark_mem).nforcing > 0 {
            let mut step_mem = erkStep_mem_mut(ark_mem);
            let Bref = B.borrow();
            let mut stage_times = std::mem::take(&mut step_mem.stage_times);
            let mut stage_coefs = std::mem::take(&mut step_mem.stage_coefs);
            for js in 0..is {
                stage_times[js as usize] = tn + Bref.c[js as usize] * h;
                stage_coefs[js as usize] = h * Bref.A[is as usize][js as usize];
            }
            erkStep_ApplyForcing(&mut step_mem, &stage_times, &stage_coefs, is, &mut nvec);
            step_mem.stage_times = stage_times;
            step_mem.stage_coefs = stage_coefs;
        }

        /*   call fused vector operation to do the work */
        let retval = erkStep_LinearCombination(ark_mem, nvec, &ycur);
        if retval != 0 {
            return ARK_VECTOROP_ERR;
        }

        /* apply user-supplied stage postprocessing function (if supplied) unless
           this is the last stage of a FSAL method, then apply the user-supplied
           step postprocessing function (if supplied) */
        let (tcur, PostProcessStepFn, PostProcessStageFn) = {
            let m = ark_mem.borrow();
            (m.tcur, m.PostProcessStepFn, m.PostProcessStageFn)
        };
        if is == stages - 1 && fsal && PostProcessStepFn.is_some() {
            let retval = erkStep_call_postprocessfn(
                ark_mem,
                PostProcessStepFn.expect("PostProcessStepFn set"),
                tcur,
                &ycur,
            );
            if retval != 0 {
                return ARK_POSTPROCESS_STEP_FAIL;
            }
        } else if let Some(PostProcessStageFn) = PostProcessStageFn {
            let retval = erkStep_call_postprocessfn(ark_mem, PostProcessStageFn, tcur, &ycur);
            if retval != 0 {
                return ARK_POSTPROCESS_STAGE_FAIL;
            }
        }

        /* call the user-supplied pre-RHS function (if supplied) */
        let PreRhsFn = ark_mem.borrow().PreRhsFn;
        if let Some(PreRhsFn) = PreRhsFn {
            let retval = erkStep_call_prerhsfn(ark_mem, PreRhsFn, tcur, &ycur);
            if retval != 0 {
                return ARK_PRERHSFN_FAIL;
            }
        }

        /* compute updated RHS */
        let Fis = erkStep_mem_mut(ark_mem).F[is as usize].clone();
        let retval = erkStep_call_f(ark_mem, tcur, &ycur, &Fis);
        erkStep_mem_mut(ark_mem).nfe += 1;

        if retval < 0 {
            return ARK_RHSFUNC_FAIL;
        }
        if retval > 0 {
            return ARK_UNREC_RHSFUNC_ERR;
        }

        /* checkpoint stage for adjoint (if necessary) */
        let checkpoint_scheme = ark_mem.borrow().checkpoint_scheme.clone();
        if let Some(checkpoint_scheme) = &checkpoint_scheme {
            let mut do_save: sunbooleantype = SUNFALSE;
            let (checkpoint_step_idx, tcur) = {
                let m = ark_mem.borrow();
                (m.checkpoint_step_idx, m.tcur)
            };
            let errcode = SUNAdjointCheckpointScheme_NeedsSaving(
                checkpoint_scheme,
                checkpoint_step_idx,
                is as suncountertype,
                tcur,
                &mut do_save,
            );
            if errcode != 0 {
                arkProcessError(
                    Some(ark_mem),
                    ARK_ADJ_CHECKPOINT_FAIL,
                    line!() as i32,
                    "erkStep_TakeStep",
                    file!(),
                    &format!("SUNAdjointCheckpointScheme_NeedsSaving returned {errcode}"),
                );
                return ARK_ADJ_CHECKPOINT_FAIL;
            }

            if do_save {
                let errcode = SUNAdjointCheckpointScheme_InsertVector(
                    checkpoint_scheme,
                    checkpoint_step_idx,
                    is as suncountertype,
                    tcur,
                    &ycur,
                );

                if errcode != 0 {
                    arkProcessError(
                        Some(ark_mem),
                        ARK_ADJ_CHECKPOINT_FAIL,
                        line!() as i32,
                        "erkStep_TakeStep",
                        file!(),
                        &format!("SUNAdjointCheckpointScheme_InsertVector returned {errcode}"),
                    );
                    return ARK_ADJ_CHECKPOINT_FAIL;
                }
            }
        }
    } /* loop over stages */

    /* compute time-evolved solution (in ark_ycur), error estimate (in dsm) */
    {
        let mut m = ark_mem.borrow_mut();
        let tn = m.tn;
        let h = m.h;
        m.tcur = tn + h;
    }

    let retval = erkStep_ComputeSolutions(ark_mem, dsmPtr);
    if retval < 0 {
        return retval;
    }

    let checkpoint_scheme = ark_mem.borrow().checkpoint_scheme.clone();
    if let Some(checkpoint_scheme) = &checkpoint_scheme {
        let mut do_save: sunbooleantype = SUNFALSE;
        let (checkpoint_step_idx, tn, h, ycur) = {
            let m = ark_mem.borrow();
            (
                m.checkpoint_step_idx,
                m.tn,
                m.h,
                m.ycur.clone().expect("ycur set"),
            )
        };
        let Bstages = B.borrow().stages;
        let errcode = SUNAdjointCheckpointScheme_NeedsSaving(
            checkpoint_scheme,
            checkpoint_step_idx,
            Bstages as suncountertype,
            tn + h,
            &mut do_save,
        );
        if errcode != 0 {
            arkProcessError(
                Some(ark_mem),
                ARK_ADJ_CHECKPOINT_FAIL,
                line!() as i32,
                "erkStep_TakeStep",
                file!(),
                &format!("SUNAdjointCheckpointScheme_NeedsSaving returned {errcode}"),
            );
            return ARK_ADJ_CHECKPOINT_FAIL;
        }

        if do_save {
            let errcode = SUNAdjointCheckpointScheme_InsertVector(
                checkpoint_scheme,
                checkpoint_step_idx,
                Bstages as suncountertype,
                tn + h,
                &ycur,
            );

            if errcode != 0 {
                arkProcessError(
                    Some(ark_mem),
                    ARK_ADJ_CHECKPOINT_FAIL,
                    line!() as i32,
                    "erkStep_TakeStep",
                    file!(),
                    &format!("SUNAdjointCheckpointScheme_InsertVector returned {errcode}"),
                );
                return ARK_ADJ_CHECKPOINT_FAIL;
            }
        }
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  erkStep_TakeStep_Adjoint:

  This routine performs a single backwards step of the discrete
  adjoint of the ERK method.

  Since we are not doing error control during the adjoint integration,
  the output variable dsmPtr should should be 0.

  The input/output variable nflagPtr is used to gauge convergence
  of any algebraic solvers within the step. In this case, it should
  always be 0 since we do not do any algebraic solves.

  The return value from this routine is:
            0 => step completed successfully
           >0 => step encountered recoverable failure;
                 reduce step and retry (if possible)
           <0 => step encountered unrecoverable failure
  ---------------------------------------------------------------*/
pub fn erkStep_TakeStep_Adjoint(
    ark_mem: &ARKodeMem,
    dsmPtr: &mut sunrealtype,
    nflagPtr: &mut i32,
) -> i32 {
    let mut retval: i32;

    /* access ARKodeERKStepMem structure */
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "erkStep_TakeStep_Adjoint",
            file!(),
            MSG_ERKSTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    /* local shortcuts for readability (`cvals`/`Xvecs` stay in step_mem and
       are reached through erkStep_mem_mut, as everywhere else here) */
    let adj_stepper: SUNAdjointStepper = ark_mem
        .borrow()
        .user_data
        .as_ref()
        .and_then(|b| b.downcast_ref::<SUNAdjointStepper>())
        .cloned()
        .expect("SUNAdjointStepper user_data");
    let (sens_np1, sens_n, sens_tmp, nst) = {
        let m = ark_mem.borrow();
        (
            m.yn.clone().expect("yn set"),
            m.ycur.clone().expect("ycur set"),
            m.tempv2.clone().expect("tempv2 set"),
            m.nst,
        )
    };
    let sens_tmp_Lambda = N_VGetSubvector_ManyVector(&sens_tmp, 0);
    let sens_np1_lambda = N_VGetSubvector_ManyVector(&sens_np1, 0);
    let (stage_values, stages, B) = {
        let step_mem = erkStep_mem_mut(ark_mem);
        (
            step_mem.F.clone(),
            step_mem.stages,
            step_mem.B.clone().expect("Butcher table set"),
        )
    };

    /* which adjoint step is being processed */
    ark_mem.borrow_mut().adj_step_idx = adj_stepper.final_step_idx.get() - nst;

    /* determine if method has fsal property */
    let fsal: sunbooleantype =
        (SUNRabs(B.borrow().A[0][0]) <= TINY) && ARKodeButcherTable_IsStifflyAccurate(Some(&B));

    /* For FSAL ERK methods, A[s-1][s-1] == b[s-1] = 0 so F[s-1] is always zero */
    if fsal {
        N_VConst(0.0, &stage_values[(stages - 1) as usize]);
    }

    /* Loop over stages */
    let mut is: i32 = stages - if fsal { 2 } else { 1 };
    while is >= 0 {
        /* Consider solving a forward IVP from t0 to tf, tf > t0.
           The adjoint ODE is solved backwards in time with step size h' = -h
           where h is the forward time step used. So at this point in the
           code ark_mem->h is h', however, the adjoint formulae need h. */
        let h = ark_mem.borrow().h;
        let adj_h: sunrealtype = -h;

        /* which stage is being processed -- needed for loading checkpoints */
        ark_mem.borrow_mut().adj_stage_idx = is as suncountertype;

        /* Set current stage time(s) and index */
        {
            let c_is = B.borrow().c[is as usize];
            let mut m = ark_mem.borrow_mut();
            let tn = m.tn;
            let h = m.h;
            m.tcur = tn + h * (1.0 - c_is);
        }

        /*
         * Compute partial current stage value \Lambda
         */
        /* the ManyVector subvector handles are read before the step_mem guard
           is taken, so no mem borrow is held across a vector op */
        let Lambda_js: Vec<N_Vector> = ((is + 1)..stages)
            .map(|js| N_VGetSubvector_ManyVector(&stage_values[js as usize], 0))
            .collect();
        let mut nvec: i32 = 0;
        {
            let mut step_mem = erkStep_mem_mut(ark_mem);
            let Bref = B.borrow();
            for js in (is + 1)..stages {
                /* h sum_{j=i}^{s} A_{ji} \Lambda_{j} */
                step_mem.cvals[nvec as usize] = adj_h * Bref.A[js as usize][is as usize];
                step_mem.Xvecs[nvec as usize] = Some(Lambda_js[(js - is - 1) as usize].clone());
                nvec += 1;
            }
            step_mem.cvals[nvec as usize] = adj_h * Bref.b[is as usize];
            step_mem.Xvecs[nvec as usize] = Some(sens_np1_lambda.clone());
            nvec += 1;
        }

        /* h b_i \lambda_{n+1} + h sum_{j=i}^{s} A_{ji} \Lambda_{j} */
        retval = erkStep_LinearCombination(ark_mem, nvec, &sens_tmp_Lambda);
        if retval != 0 {
            return ARK_VECTOROP_ERR;
        }

        /* Compute the stages \Lambda_i and \nu_i by evaluating f_{y}^*(t_i, z_i, p) and
           f_{p}^*(t_i, z_i, p) and applying them to sens_tmp_Lambda (in sens_tmp). This is
           done in fe which retrieves z_i from the checkpoint data */
        let tcur = ark_mem.borrow().tcur;
        retval = erkStep_call_f(ark_mem, tcur, &sens_tmp, &stage_values[is as usize]);
        erkStep_mem_mut(ark_mem).nfe += 1;

        /* The checkpoint was not found, so we need to recompute at least
           this step forward in time. We first seek the last checkpointed step
           solution, then recompute from there. */
        if ark_mem.borrow().load_checkpoint_fail {
            let tempv3 = ark_mem.borrow().tempv3.clone().expect("tempv3 set");
            let mut checkpoint = N_VGetSubvector_ManyVector(&tempv3, 0);
            let curr_step: suncountertype = ark_mem.borrow().adj_step_idx;
            let mut start_step: suncountertype = curr_step;

            let checkpoint_scheme = ark_mem
                .borrow()
                .checkpoint_scheme
                .clone()
                .expect("checkpoint_scheme set");
            let mut errcode: SUNErrCode = SUN_ERR_CHECKPOINT_NOT_FOUND;
            let mut i: suncountertype = 0;
            while i <= curr_step {
                let mut checkpoint_t: sunrealtype = 0.0;
                errcode = SUNAdjointCheckpointScheme_LoadVector(
                    &checkpoint_scheme,
                    start_step,
                    stages as suncountertype,
                    /*peek=*/ SUNTRUE,
                    &mut checkpoint,
                    &mut checkpoint_t,
                );
                if errcode == SUN_SUCCESS {
                    /* OK, now we have the last checkpoint that stored as (start_step, stages).
                       This represents the last step solution that was checkpointed. As such, we
                       want to recompute from start_step+1 to stop_step. */
                    start_step += 1;
                    let t0 = checkpoint_t;
                    let tf = ark_mem.borrow().tn;
                    errcode = SUNAdjointStepper_RecomputeFwd(
                        &adj_stepper,
                        start_step,
                        t0,
                        &checkpoint,
                        tf,
                    );
                    if errcode != SUN_SUCCESS {
                        arkProcessError(
                            Some(ark_mem),
                            ARK_ADJ_RECOMPUTE_FAIL,
                            line!() as i32,
                            "erkStep_TakeStep_Adjoint",
                            file!(),
                            &format!("SUNAdjointStepper_RecomputeFwd returned {errcode}"),
                        );
                        return ARK_ADJ_RECOMPUTE_FAIL;
                    }
                    return erkStep_TakeStep_Adjoint(ark_mem, dsmPtr, nflagPtr);
                }
                i += 1;
                start_step -= 1;
            }
            if errcode != SUN_SUCCESS {
                arkProcessError(
                    Some(ark_mem),
                    ARK_ADJ_RECOMPUTE_FAIL,
                    line!() as i32,
                    "erkStep_TakeStep_Adjoint",
                    file!(),
                    "Could not load or recompute missing step",
                );
                return ARK_ADJ_RECOMPUTE_FAIL;
            }
        } else if retval > 0 {
            return ARK_UNREC_RHSFUNC_ERR;
        } else if retval < 0 {
            arkProcessError(
                Some(ark_mem),
                ARK_RHSFUNC_FAIL,
                line!() as i32,
                "erkStep_TakeStep_Adjoint",
                file!(),
                &format!("The right hand side function failed returned {retval}"),
            );
            return ARK_RHSFUNC_FAIL;
        }

        is -= 1;
    }

    /* Throw away the step solution */
    let mut checkpoint_t: sunrealtype = ZERO;
    let tempv2 = ark_mem.borrow().tempv2.clone().expect("tempv2 set");
    let mut checkpoint = N_VGetSubvector_ManyVector(&tempv2, 0);
    let (checkpoint_scheme, adj_step_idx) = {
        let m = ark_mem.borrow();
        (
            m.checkpoint_scheme.clone().expect("checkpoint_scheme set"),
            m.adj_step_idx,
        )
    };
    let errcode = SUNAdjointCheckpointScheme_LoadVector(
        &checkpoint_scheme,
        adj_step_idx,
        0,
        /*peek=*/ SUNFALSE,
        &mut checkpoint,
        &mut checkpoint_t,
    );
    if errcode != SUN_SUCCESS {
        arkProcessError(
            Some(ark_mem),
            ARK_ADJ_CHECKPOINT_FAIL,
            line!() as i32,
            "erkStep_TakeStep_Adjoint",
            file!(),
            &format!("SUNAdjointCheckpointScheme_LoadVector returned {errcode}"),
        );
        return ARK_ADJ_CHECKPOINT_FAIL;
    }

    /* Now compute the time step solution. We cannot use erkStep_ComputeSolutions because the
       adjoint calculation for the time step solution is different than the forward case. */

    let mut nvec: i32 = 0;
    {
        let mut step_mem = erkStep_mem_mut(ark_mem);
        for j in 0..stages {
            step_mem.cvals[nvec as usize] = ONE;
            /* this needs to be the stage values [Lambda_i, nu_i] */
            step_mem.Xvecs[nvec as usize] = Some(stage_values[j as usize].clone());
            nvec += 1;
        }
        step_mem.cvals[nvec as usize] = ONE;
        step_mem.Xvecs[nvec as usize] = Some(sens_np1.clone());
        nvec += 1;
    }

    /* \lambda_n = \lambda_{n+1} + \sum_{j=1}^{s} \Lambda_j
       \mu_n     = \mu_{n+1} + \sum_{j=1}^{s} \nu_j */
    retval = erkStep_LinearCombination(ark_mem, nvec, &sens_n);
    if retval != 0 {
        return ARK_VECTOROP_ERR;
    }

    *dsmPtr = ZERO;
    *nflagPtr = 0;

    ARK_SUCCESS
}

/*===============================================================
  Internal utility routines
  ===============================================================*/

/*---------------------------------------------------------------
  erkStep_SetButcherTable

  This routine determines the ERK method to use, based on the
  desired accuracy.
  ---------------------------------------------------------------*/
pub fn erkStep_SetButcherTable(ark_mem: &ARKodeMem) -> i32 {
    /* access ARKodeERKStepMem structure */
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "erkStep_SetButcherTable",
            file!(),
            MSG_ERKSTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    /* if table has already been specified, just return */
    if erkStep_mem_mut(ark_mem).B.is_some() {
        return ARK_SUCCESS;
    }

    /* initialize table number to illegal values */
    /* (every arm of the switch below assigns, so C's `etable = -1` initializer
       is dead; the `etable > -1` guard is preserved verbatim) */

    /* select method based on order */
    let q = erkStep_mem_mut(ark_mem).q;
    let etable: i32 = match q {
        1 => ERKSTEP_DEFAULT_1,
        2 => ERKSTEP_DEFAULT_2,
        3 => ERKSTEP_DEFAULT_3,
        4 => ERKSTEP_DEFAULT_4,
        5 => ERKSTEP_DEFAULT_5,
        6 => ERKSTEP_DEFAULT_6,
        7 => ERKSTEP_DEFAULT_7,
        8 => ERKSTEP_DEFAULT_8,
        9 => ERKSTEP_DEFAULT_9,
        _ => {
            /* no available method, set default */
            arkProcessError(
                Some(ark_mem),
                ARK_WARNING,
                line!() as i32,
                "erkStep_SetButcherTable",
                file!(),
                "No explicit method at requested order, using q=9.",
            );
            ERKSTEP_DEFAULT_9
        }
    };

    if etable > -1 {
        let B = ARKodeButcherTable_LoadERK(etable);
        erkStep_mem_mut(ark_mem).B = B;
    }

    /* note Butcher table space requirements */
    let B = erkStep_mem_mut(ark_mem).B.clone();
    let mut Bliw: sunindextype = 0;
    let mut Blrw: sunindextype = 0;
    ARKodeButcherTable_Space(B.as_ref(), &mut Bliw, &mut Blrw);
    {
        let mut m = ark_mem.borrow_mut();
        m.liw += Bliw;
        m.lrw += Blrw;
    }

    /* set [redundant] stored values for stage numbers and method orders */
    if let Some(B) = &B {
        let (Bstages, Bq, Bp) = {
            let B = B.borrow();
            (B.stages, B.q, B.p)
        };
        let mut step_mem = erkStep_mem_mut(ark_mem);
        step_mem.stages = Bstages;
        step_mem.q = Bq;
        step_mem.p = Bp;
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  erkStep_CheckButcherTable

  This routine runs through the explicit Butcher table to ensure
  that it meets all necessary requirements, including:
    strictly lower-triangular (ERK)
    method order q > 0 (all)
    embedding order q > 0 (all -- if adaptive time-stepping enabled)
    stages > 0 (all)

  Returns ARK_SUCCESS if tables pass, ARK_INVALID_TABLE otherwise.
  ---------------------------------------------------------------*/
pub fn erkStep_CheckButcherTable(ark_mem: &ARKodeMem) -> i32 {
    let tol: sunrealtype = 1.0e-12;

    /* access ARKodeERKStepMem structure */
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "erkStep_CheckButcherTable",
            file!(),
            MSG_ERKSTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    /* `B` stays an Option here: C only dereferences it after the `stages`,
       `q` and `p` guards below have passed */
    let (stages, q, p, B) = {
        let step_mem = erkStep_mem_mut(ark_mem);
        (step_mem.stages, step_mem.q, step_mem.p, step_mem.B.clone())
    };
    let fixedstep = ark_mem.borrow().fixedstep;

    /* check that stages > 0 */
    if stages < 1 {
        arkProcessError(
            Some(ark_mem),
            ARK_INVALID_TABLE,
            line!() as i32,
            "erkStep_CheckButcherTable",
            file!(),
            "stages < 1!",
        );
        return ARK_INVALID_TABLE;
    }

    /* check that method order q > 0 */
    if q < 1 {
        arkProcessError(
            Some(ark_mem),
            ARK_INVALID_TABLE,
            line!() as i32,
            "erkStep_CheckButcherTable",
            file!(),
            "method order < 1!",
        );
        return ARK_INVALID_TABLE;
    }

    /* check that embedding order p > 0 */
    if (p < 1) && (!fixedstep) {
        arkProcessError(
            Some(ark_mem),
            ARK_INVALID_TABLE,
            line!() as i32,
            "erkStep_CheckButcherTable",
            file!(),
            "embedding order < 1, but ARKodeSetFixedStep was not called!",
        );
        return ARK_INVALID_TABLE;
    }

    /* check that embedding exists */
    if (p > 0) && (!fixedstep) {
        if B.as_ref().expect("Butcher table set").borrow().d.is_empty() {
            arkProcessError(
                Some(ark_mem),
                ARK_INVALID_TABLE,
                line!() as i32,
                "erkStep_CheckButcherTable",
                file!(),
                "no embedding, but ARKodeSetFixedStep was not called!",
            );
            return ARK_INVALID_TABLE;
        }
    }

    /* check that ERK table is strictly lower triangular */
    let mut okay: sunbooleantype = SUNTRUE;
    {
        let B = B.as_ref().expect("Butcher table set").borrow();
        for i in 0..stages {
            for j in i..stages {
                if SUNRabs(B.A[i as usize][j as usize]) > tol {
                    okay = SUNFALSE;
                }
            }
        }
    }
    if !okay {
        arkProcessError(
            Some(ark_mem),
            ARK_INVALID_TABLE,
            line!() as i32,
            "erkStep_CheckButcherTable",
            file!(),
            "Ae Butcher table is implicit!",
        );
        return ARK_INVALID_TABLE;
    }

    /* check if all b values are positive for relaxation */
    if ark_mem.borrow().relax_enabled {
        if q < 2 {
            arkProcessError(
                Some(ark_mem),
                ARK_INVALID_TABLE,
                line!() as i32,
                "erkStep_CheckButcherTable",
                file!(),
                "The Butcher table must be at least second order when using relaxation!",
            );
            return ARK_INVALID_TABLE;
        }

        for i in 0..stages {
            let bi = B.as_ref().expect("Butcher table set").borrow().b[i as usize];
            if bi < ZERO {
                arkProcessError(
                    Some(ark_mem),
                    ARK_INVALID_TABLE,
                    line!() as i32,
                    "erkStep_CheckButcherTable",
                    file!(),
                    "The Butcher table has a negative b value but relaxation enabled!",
                );
                return ARK_INVALID_TABLE;
            }
        }
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  erkStep_ComputeSolutions

  This routine calculates the final RK solution using the existing
  data.  This solution is placed directly in ark_ycur.  This routine
  also computes the error estimate ||y-ytilde||_WRMS, where ytilde
  is the embedded solution, and the norm weights come from
  ark_ewt.  This norm value is returned.  The vector form of this
  estimated error (y-ytilde) is stored in ark_tempv1, in case the
  calling routine wishes to examine the error locations.

  Note: at this point in the step, the vector ark_tempv1 may be
  used as a temporary vector.
  ---------------------------------------------------------------*/
pub fn erkStep_ComputeSolutions(ark_mem: &ARKodeMem, dsmPtr: &mut sunrealtype) -> i32 {
    /* access ARKodeERKStepMem structure */
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "erkStep_ComputeSolutions",
            file!(),
            MSG_ERKSTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    /* set N_Vector shortcuts */
    let (y, yerr, yn, ewt, tn, h, fixedstep, AccumErrorType) = {
        let m = ark_mem.borrow();
        (
            m.ycur.clone().expect("ycur set"),
            m.tempv1.clone().expect("tempv1 set"),
            m.yn.clone().expect("yn set"),
            m.ewt.clone().expect("ewt set"),
            m.tn,
            m.h,
            m.fixedstep,
            m.AccumErrorType,
        )
    };

    /* local shortcuts for fused vector operations: `cvals` and `Xvecs` live
       in step_mem and are reached through erkStep_mem_mut */

    /* initialize output */
    *dsmPtr = ZERO;

    /* determine if method has fsal property */
    let (stages, B) = {
        let step_mem = erkStep_mem_mut(ark_mem);
        (
            step_mem.stages,
            step_mem.B.clone().expect("Butcher table set"),
        )
    };
    let fsal = ARKodeButcherTable_IsStifflyAccurate(Some(&B));

    /* Compute time step solution. For FSAL methods, ycur already contains the new
       solution. */
    if !fsal {
        /* set arrays for fused vector operation */
        let mut nvec: i32 = 0;
        {
            let mut step_mem = erkStep_mem_mut(ark_mem);
            let Bref = B.borrow();
            for j in 0..stages {
                step_mem.cvals[nvec as usize] = h * Bref.b[j as usize];
                let Fj = step_mem.F[j as usize].clone();
                step_mem.Xvecs[nvec as usize] = Some(Fj);
                nvec += 1;
            }
            step_mem.cvals[nvec as usize] = ONE;
            step_mem.Xvecs[nvec as usize] = Some(yn.clone());
            nvec += 1;
        }

        /* apply external polynomial forcing */
        if erkStep_mem_mut(ark_mem).nforcing > 0 {
            let mut step_mem = erkStep_mem_mut(ark_mem);
            let Bref = B.borrow();
            let mut stage_times = std::mem::take(&mut step_mem.stage_times);
            let mut stage_coefs = std::mem::take(&mut step_mem.stage_coefs);
            for j in 0..stages {
                stage_times[j as usize] = tn + Bref.c[j as usize] * h;
                stage_coefs[j as usize] = h * Bref.b[j as usize];
            }
            erkStep_ApplyForcing(&mut step_mem, &stage_times, &stage_coefs, stages, &mut nvec);
            step_mem.stage_times = stage_times;
            step_mem.stage_coefs = stage_coefs;
        }

        /* call fused vector operation to do the work */
        let retval = erkStep_LinearCombination(ark_mem, nvec, &y);
        if retval != 0 {
            return ARK_VECTOROP_ERR;
        }

        /* apply user-supplied step postprocessing function (if supplied) */
        let (tcur, PostProcessStepFn, ycur) = {
            let m = ark_mem.borrow();
            (
                m.tcur,
                m.PostProcessStepFn,
                m.ycur.clone().expect("ycur set"),
            )
        };
        if let Some(PostProcessStepFn) = PostProcessStepFn {
            let retval = erkStep_call_postprocessfn(ark_mem, PostProcessStepFn, tcur, &ycur);
            if retval != 0 {
                return ARK_POSTPROCESS_STEP_FAIL;
            }
        }
    }

    /* Compute yerr (if step adaptivity or error accumulation enabled) */
    if !fixedstep || (AccumErrorType != ARK_ACCUMERROR_NONE) {
        /* set arrays for fused vector operation */
        let mut nvec: i32 = 0;
        {
            let mut step_mem = erkStep_mem_mut(ark_mem);
            let Bref = B.borrow();
            for j in 0..stages {
                step_mem.cvals[nvec as usize] = h * (Bref.b[j as usize] - Bref.d[j as usize]);
                let Fj = step_mem.F[j as usize].clone();
                step_mem.Xvecs[nvec as usize] = Some(Fj);
                nvec += 1;
            }
        }

        /* apply external polynomial forcing */
        if erkStep_mem_mut(ark_mem).nforcing > 0 {
            let mut step_mem = erkStep_mem_mut(ark_mem);
            let Bref = B.borrow();
            let mut stage_times = std::mem::take(&mut step_mem.stage_times);
            let mut stage_coefs = std::mem::take(&mut step_mem.stage_coefs);
            for j in 0..stages {
                stage_times[j as usize] = tn + Bref.c[j as usize] * h;
                stage_coefs[j as usize] = h * (Bref.b[j as usize] - Bref.d[j as usize]);
            }
            erkStep_ApplyForcing(&mut step_mem, &stage_times, &stage_coefs, stages, &mut nvec);
            step_mem.stage_times = stage_times;
            step_mem.stage_coefs = stage_coefs;
        }

        /* call fused vector operation to do the work */
        let retval = erkStep_LinearCombination(ark_mem, nvec, &yerr);
        if retval != 0 {
            return ARK_VECTOROP_ERR;
        }

        /* fill error norm */
        *dsmPtr = N_VWrmsNorm(&yerr, &ewt);
    }

    ARK_SUCCESS
}

/*===============================================================
  Internal utility routines for relaxation
  ===============================================================*/

/* -----------------------------------------------------------------------------
 * erkStep_RelaxDeltaE
 *
 * Computes the change in the relaxation functions for use in relaxation methods
 * delta_e = h * sum_i b_i * <rjac(z_i), f_i>
 * ---------------------------------------------------------------------------*/
pub fn erkStep_RelaxDeltaE(
    ark_mem: &ARKodeMem,
    relax_jac_fn: Option<ARKRelaxJacFn>,
    num_relax_jac_evals: &mut i64,
    delta_e_out: &mut sunrealtype,
) -> i32 {
    let (z_stage, J_relax, yn, h) = {
        let m = ark_mem.borrow();
        (
            m.tempv2.clone().expect("tempv2 set"),
            m.tempv3.clone().expect("tempv3 set"),
            m.yn.clone().expect("yn set"),
            m.h,
        )
    };

    /* Access the stepper memory structure */
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "erkStep_RelaxDeltaE",
            file!(),
            MSG_ERKSTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    /* C would dereference a NULL function pointer here (UB -> panic) */
    let relax_jac_fn = relax_jac_fn.expect("relax_jac_fn set");

    /* Initialize output */
    *delta_e_out = ZERO;

    /* Set arrays for fused vector operation: `cvals` and `Xvecs` live in
       step_mem and are reached through erkStep_mem_mut */

    let (stages, B) = {
        let step_mem = erkStep_mem_mut(ark_mem);
        (
            step_mem.stages,
            step_mem.B.clone().expect("Butcher table set"),
        )
    };

    for i in 0..stages {
        /* Construct stages z[i] = y_n + h * sum_j Ae[i,j] Fe[j] + Ai[i,j] Fi[j] */
        let mut nvec: i32 = 0;

        {
            let mut step_mem = erkStep_mem_mut(ark_mem);
            let Bref = B.borrow();

            step_mem.cvals[nvec as usize] = ONE;
            step_mem.Xvecs[nvec as usize] = Some(yn.clone());
            nvec += 1;

            for j in 0..i {
                step_mem.cvals[nvec as usize] = h * Bref.A[i as usize][j as usize];
                let Fj = step_mem.F[j as usize].clone();
                step_mem.Xvecs[nvec as usize] = Some(Fj);
                nvec += 1;
            }
        }

        let retval = erkStep_LinearCombination(ark_mem, nvec, &z_stage);
        if retval != 0 {
            return ARK_VECTOROP_ERR;
        }

        /* Evaluate the Jacobian at z_i */
        let mut user_data = ark_mem.borrow_mut().user_data.take();
        let retval = relax_jac_fn(&z_stage, &J_relax, &mut user_data);
        ark_mem.borrow_mut().user_data = user_data;
        *num_relax_jac_evals += 1;
        if retval < 0 {
            return ARK_RELAX_JAC_FAIL;
        }
        if retval > 0 {
            return ARK_RELAX_JAC_RECV;
        }

        /* Update estimates */
        let Fi = erkStep_mem_mut(ark_mem).F[i as usize].clone();
        let bi = B.borrow().b[i as usize];
        let (nvdotprodlocal, nvdotprodmultiallreduce) = {
            let ops = J_relax.ops.borrow();
            (
                ops.nvdotprodlocal.is_some(),
                ops.nvdotprodmultiallreduce.is_some(),
            )
        };
        if nvdotprodlocal && nvdotprodmultiallreduce {
            *delta_e_out += bi * N_VDotProdLocal(&J_relax, &Fi);
        } else {
            *delta_e_out += bi * N_VDotProd(&J_relax, &Fi);
        }
    }

    let (nvdotprodlocal, nvdotprodmultiallreduce) = {
        let ops = J_relax.ops.borrow();
        (
            ops.nvdotprodlocal.is_some(),
            ops.nvdotprodmultiallreduce.is_some(),
        )
    };
    if nvdotprodlocal && nvdotprodmultiallreduce {
        let retval = N_VDotProdMultiAllReduce(1, &J_relax, std::slice::from_mut(delta_e_out));
        if retval != 0 {
            return ARK_VECTOROP_ERR;
        }
    }

    *delta_e_out *= h;

    ARK_SUCCESS
}

/* -----------------------------------------------------------------------------
 * erkStep_GetOrder
 *
 * Returns the method order
 * ---------------------------------------------------------------------------*/
pub fn erkStep_GetOrder(ark_mem: &ARKodeMem) -> i32 {
    erkStep_mem_mut(ark_mem).q
}

/*---------------------------------------------------------------
  Utility routines for interfacing with SUNAdjointStepper
  ---------------------------------------------------------------*/

/// C `erkStep_fe_Adj(sunrealtype t, N_Vector sens_partial_stage,
/// N_Vector sens_complete_stage, void* content)` — installed as ERKStep's
/// `f` for the adjoint memory, hence the [`ARKRhsFn`] shape.
///
/// `void* content` is the `SUNAdjointStepper` token stored in the adjoint
/// ARKODE memory's `user_data`.
pub fn erkStep_fe_Adj(
    t: sunrealtype,
    sens_partial_stage: &N_Vector,
    sens_complete_stage: &N_Vector,
    content: &mut Option<Box<dyn Any>>,
) -> i32 {
    let errcode: SUNErrCode;

    let adj_stepper: SUNAdjointStepper = content
        .as_ref()
        .and_then(|b| b.downcast_ref::<SUNAdjointStepper>())
        .cloned()
        .expect("SUNAdjointStepper content");
    let check_scheme = adj_stepper.checkpoint_scheme.borrow().clone();
    let adj_sunstepper = adj_stepper.adj_sunstepper.borrow().clone();
    let mut ark_mem_out: Option<ARKodeMem> = None;
    let _ = SUNStepper_GetContentAs::<ARKodeMem>(&adj_sunstepper, &mut ark_mem_out);
    let ark_mem = ark_mem_out.expect("ARKodeMem stepper content");

    /* access ARKodeERKStepMem structure */
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            None,
            ARK_MEM_NULL,
            line!() as i32,
            "erkStep_fe_Adj",
            file!(),
            MSG_ERKSTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    let adj_f = erkStep_mem_mut(&ark_mem).adj_f.expect("adj_f set");

    let tempv3 = ark_mem.borrow().tempv3.clone().expect("tempv3 set");
    let mut checkpoint = N_VGetSubvector_ManyVector(&tempv3, 0);
    let mut checkpoint_t: sunrealtype = 0.0;

    ark_mem.borrow_mut().load_checkpoint_fail = SUNFALSE;

    let (adj_step_idx, adj_stage_idx) = {
        let m = ark_mem.borrow();
        (m.adj_step_idx, m.adj_stage_idx)
    };
    errcode = SUNAdjointCheckpointScheme_LoadVector(
        &check_scheme,
        adj_step_idx,
        adj_stage_idx,
        SUNFALSE,
        &mut checkpoint,
        &mut checkpoint_t,
    );

    /* Checkpoint was not found, recompute the missing step */
    if errcode == SUN_ERR_CHECKPOINT_NOT_FOUND {
        ark_mem.borrow_mut().load_checkpoint_fail = SUNTRUE;
        return 1;
    }

    /* C: `void* user_data = adj_stepper->user_data;` aliases the FORWARD
       integrator's `user_data`, which the forward RHS also dereferences during
       `SUNAdjointStepper_RecomputeFwd`. A `Box` cannot alias and moving it
       would strand the forward RHS, so (accepted deviation class 6, the shape
       used by `arkStep_fe_Adj`) the token is taken from the adjoint stepper
       for the duration of the call and restored on every path. */
    let mut user_data = adj_stepper.user_data.borrow_mut().take();

    /* Evaluate f_{y}^*(t_i, z_i, p) \Lambda_i and f_{p}^*(t_i, z_i, p) \nu_i */
    let retval = adj_f(
        t,
        &checkpoint,
        sens_partial_stage,
        sens_complete_stage,
        &mut user_data,
    );
    *adj_stepper.user_data.borrow_mut() = user_data;
    retval
}

/// C `erkStepCompatibleWithAdjointSolver(ark_mem, step_mem, lineno, fname,
/// filename)`; the `SUNDIALS_MAYBE_UNUSED` `step_mem` argument has no Rust
/// counterpart (the record is reached through `ark_mem`).
pub fn erkStepCompatibleWithAdjointSolver(
    ark_mem: &ARKodeMem,
    lineno: i32,
    fname: &str,
    filename: &str,
) -> i32 {
    if !ark_mem.borrow().fixedstep {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            lineno,
            fname,
            filename,
            "ERKStep must be using a fixed step to work with SUNAdjointStepper",
        );
        return ARK_ILL_INPUT;
    }

    if ark_mem.borrow().relax_enabled {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            lineno,
            fname,
            filename,
            "SUNAdjointStepper is not compatible with relaxation",
        );
        return ARK_ILL_INPUT;
    }

    if ark_mem.borrow().constraints.is_some() {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            lineno,
            fname,
            filename,
            "SUNAdjointStepper is not compatible with constraints",
        );
        return ARK_ILL_INPUT;
    }

    ARK_SUCCESS
}

/// C `static SUNErrCode erkStep_SUNStepperReInit(SUNStepper stepper,
/// sunrealtype t0, N_Vector y0)`.
fn erkStep_SUNStepperReInit(stepper: &SUNStepper, t0: sunrealtype, y0: &N_Vector) -> SUNErrCode {
    let mut arkode_mem: Option<ARKodeMem> = None;
    let _ = SUNStepper_GetContentAs::<ARKodeMem>(stepper, &mut arkode_mem);
    let arkode_mem = match arkode_mem {
        Some(arkode_mem) => arkode_mem,
        None => {
            arkProcessError(
                None,
                ARK_ILL_INPUT,
                line!() as i32,
                "erkStep_SUNStepperReInit",
                file!(),
                "The ARKStep memory pointer is NULL",
            );
            return ARK_ILL_INPUT;
        }
    };
    let ark_mem = &arkode_mem;

    /* access ARKodeMem and ARKodeERKStepMem structures (C:
       erkStep_AccessARKODEStepMem(arkode_mem, "erkStepSUNStepperReInit", ...)) */
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "erkStepSUNStepperReInit",
            file!(),
            MSG_ERKSTEP_NO_MEM,
        );
        arkProcessError(
            None,
            ARK_ILL_INPUT,
            line!() as i32,
            "erkStep_SUNStepperReInit",
            file!(),
            "The ARKStep memory pointer is NULL",
        );
        return ARK_ILL_INPUT;
    }

    /* C passes `step_mem->f` straight through; `ERKStepReInit` takes a
       non-nullable `ARKRhsFn` (its C NULL check is handled by the type
       system), so a missing `f` panics here instead of returning
       ARK_ILL_INPUT from the callee */
    let f = erkStep_mem_mut(ark_mem).f.expect("step_mem->f set");

    let last_flag = ERKStepReInit(&arkode_mem, f, t0, y0);
    *stepper.last_flag.borrow_mut() = last_flag;
    if last_flag != ARK_SUCCESS {
        arkProcessError(
            Some(ark_mem),
            last_flag,
            line!() as i32,
            "erkStep_SUNStepperReInit",
            file!(),
            "ERKStepReInit return an error\n",
        );
        return SUN_ERR_OP_FAIL;
    }

    SUN_SUCCESS
}

/// C `ERKStepCreateAdjointStepper(void* arkode_mem, SUNAdjRhsFn adj_f,
/// sunrealtype tf, N_Vector sf, SUNContext sunctx,
/// SUNAdjointStepper* adj_stepper_ptr)`.
pub fn ERKStepCreateAdjointStepper(
    arkode_mem: &ARKodeMem,
    adj_f: Option<SUNAdjRhsFn>,
    tf: sunrealtype,
    sf: &N_Vector,
    sunctx: &SUNContext,
    adj_stepper_ptr: &mut Option<SUNAdjointStepper>,
) -> i32 {
    /* access ARKodeMem and ARKodeERKStepMem structures (C:
       erkStep_AccessARKODEStepMem(arkode_mem, "ERKStepCreateAdjointStepper",
       ...)) */
    if arkode_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(arkode_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "ERKStepCreateAdjointStepper",
            file!(),
            MSG_ERKSTEP_NO_MEM,
        );
        arkProcessError(
            None,
            ARK_ILL_INPUT,
            line!() as i32,
            "ERKStepCreateAdjointStepper",
            file!(),
            "The ERKStep memory pointer is NULL",
        );
        return ARK_ILL_INPUT;
    }
    let ark_mem = arkode_mem;

    if erkStepCompatibleWithAdjointSolver(
        ark_mem,
        line!() as i32,
        "ERKStepCreateAdjointStepper",
        file!(),
    ) != 0
    {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "ERKStepCreateAdjointStepper",
            file!(),
            "ark_mem provided is not compatible with adjoint calculation",
        );
        return ARK_ILL_INPUT;
    }

    if adj_f.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "ERKStepCreateAdjointStepper",
            file!(),
            "adj_fe cannot be NULL.",
        );
        return ARK_ILL_INPUT;
    }

    if N_VGetVectorID(sf) != SUNDIALS_NVEC_MANYVECTOR {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "ERKStepCreateAdjointStepper",
            file!(),
            "Incompatible vector type provided for adjoint calculation",
        );
        return ARK_ILL_INPUT;
    }

    /*
      Create and configure the ERKStep stepper for the adjoint system
    */
    let mut nst: i64 = 0;
    let mut retval = ARKodeGetNumSteps(arkode_mem, &mut nst);
    if retval != 0 {
        arkProcessError(
            Some(ark_mem),
            retval,
            line!() as i32,
            "ERKStepCreateAdjointStepper",
            file!(),
            "ARKodeGetNumSteps failed",
        );
        return retval;
    }

    let sunctx_fwd = ark_mem.borrow().sunctx.clone();
    let arkode_mem_adj = match ERKStepCreate(erkStep_fe_Adj, tf, sf, &sunctx_fwd) {
        Some(arkode_mem_adj) => arkode_mem_adj,
        None => {
            arkProcessError(
                Some(ark_mem),
                ARK_MEM_NULL,
                line!() as i32,
                "ERKStepCreateAdjointStepper",
                file!(),
                "ERKStepCreate returned NULL\n",
            );
            return ARK_MEM_NULL;
        }
    };
    let ark_mem_adj = &arkode_mem_adj;

    erkStep_mem_mut(ark_mem_adj).adj_f = adj_f;
    ark_mem_adj.borrow_mut().do_adjoint = SUNTRUE;

    let h = ark_mem.borrow().h;
    retval = ARKodeSetFixedStep(&arkode_mem_adj, -h);
    if retval != 0 {
        arkProcessError(
            Some(ark_mem),
            retval,
            line!() as i32,
            "ERKStepCreateAdjointStepper",
            file!(),
            "ARKodeSetFixedStep failed",
        );
        return retval;
    }

    let B = erkStep_mem_mut(ark_mem).B.clone().expect("Butcher table set");
    retval = crate::arkode_erkstep_io::ERKStepSetTable(&arkode_mem_adj, &B);
    if retval != 0 {
        arkProcessError(
            Some(ark_mem),
            retval,
            line!() as i32,
            "ERKStepCreateAdjointStepper",
            file!(),
            "ERKStepSetTables failed",
        );
        return retval;
    }

    retval = ARKodeSetMaxNumSteps(&arkode_mem_adj, nst);
    if retval != 0 {
        arkProcessError(
            Some(ark_mem),
            retval,
            line!() as i32,
            "ERKStepCreateAdjointStepper",
            file!(),
            "ARKodeSetMaxNumSteps failed",
        );
        return retval;
    }

    let checkpoint_scheme = ark_mem.borrow().checkpoint_scheme.clone();
    retval = ARKodeSetAdjointCheckpointScheme(&arkode_mem_adj, checkpoint_scheme.as_ref());
    if retval != 0 {
        arkProcessError(
            Some(ark_mem),
            retval,
            line!() as i32,
            "ERKStepCreateAdjointStepper",
            file!(),
            "ARKodeSetAdjointCheckpointScheme failed",
        );
        return retval;
    }

    let mut errcode: SUNErrCode;

    let mut fwd_stepper: Option<SUNStepper> = None;
    retval = ARKodeCreateSUNStepper(arkode_mem, &mut fwd_stepper);
    if retval != 0 {
        arkProcessError(
            Some(ark_mem),
            retval,
            line!() as i32,
            "ERKStepCreateAdjointStepper",
            file!(),
            "ARKodeCreateSUNStepper failed",
        );
        return retval;
    }
    let fwd_stepper = fwd_stepper.expect("fwd_stepper");

    errcode = SUNStepper_SetReInitFn(&fwd_stepper, Some(erkStep_SUNStepperReInit));
    if errcode != SUN_SUCCESS {
        retval = ARK_SUNSTEPPER_ERR;
        arkProcessError(
            Some(ark_mem),
            retval,
            line!() as i32,
            "ERKStepCreateAdjointStepper",
            file!(),
            "SUNStepper_SetReInitFn failed",
        );
        return retval;
    }

    let mut adj_stepper: Option<SUNStepper> = None;
    retval = ARKodeCreateSUNStepper(&arkode_mem_adj, &mut adj_stepper);
    if retval != 0 {
        arkProcessError(
            Some(ark_mem),
            retval,
            line!() as i32,
            "ERKStepCreateAdjointStepper",
            file!(),
            "ARKodeCreateSUNStepper failed",
        );
        return retval;
    }
    let adj_stepper = adj_stepper.expect("adj_stepper");

    errcode = SUNStepper_SetReInitFn(&adj_stepper, Some(erkStep_SUNStepperReInit));
    if errcode != SUN_SUCCESS {
        retval = ARK_SUNSTEPPER_ERR;
        arkProcessError(
            Some(ark_mem),
            retval,
            line!() as i32,
            "ERKStepCreateAdjointStepper",
            file!(),
            "SUNStepper_SetReInitFn failed",
        );
        return retval;
    }

    /* Setting this ensures that the ARKodeMem underneath the adj_stepper
       is destroyed with the SUNStepper_Destroy call. */
    errcode = SUNStepper_SetDestroyFn(&adj_stepper, Some(arkSUNStepperSelfDestruct));
    if errcode != SUN_SUCCESS {
        retval = ARK_SUNSTEPPER_ERR;
        arkProcessError(
            Some(ark_mem),
            retval,
            line!() as i32,
            "ERKStepCreateAdjointStepper",
            file!(),
            "SUNStepper_SetDestroyFn failed",
        );
        return retval;
    }

    /* SUNAdjointStepper will own the SUNSteppers and destroy them */
    errcode = SUNAdjointStepper_Create(
        fwd_stepper,
        SUNTRUE,
        adj_stepper,
        SUNTRUE,
        nst - 1,
        tf,
        sf,
        checkpoint_scheme.expect("checkpoint_scheme set"),
        sunctx,
        adj_stepper_ptr,
    );
    if errcode != SUN_SUCCESS {
        retval = ARK_SUNADJSTEPPER_ERR;
        arkProcessError(
            Some(ark_mem),
            retval,
            line!() as i32,
            "ERKStepCreateAdjointStepper",
            file!(),
            "SUNAdjointStepper_Create failed",
        );
        return retval;
    }

    /* C: SUNAdjointStepper_SetUserData(*adj_stepper_ptr, ark_mem->user_data)
       ALIASES the forward integrator's `user_data` into the adjoint stepper --
       both `adj_f` (through `erkStep_fe_Adj`) and the forward RHS (during
       SUNAdjointStepper_RecomputeFwd) dereference it. A `Box` cannot alias and
       moving it would strand the forward RHS, so (deviation class 6) the token
       is left with the forward memory; the example/integration layer must hand
       the adjoint stepper its own copy with SUNAdjointStepper_SetUserData. */
    errcode =
        SUNAdjointStepper_SetUserData(adj_stepper_ptr.as_ref().expect("adj_stepper_ptr"), None);
    if errcode != SUN_SUCCESS {
        retval = ARK_SUNADJSTEPPER_ERR;
        arkProcessError(
            Some(ark_mem),
            retval,
            line!() as i32,
            "ERKStepCreateAdjointStepper",
            file!(),
            "SUNAdjointStepper_SetUserData failed",
        );
        return retval;
    }

    /* We need access to the adjoint solver to access the parameter Jacobian inside of ERKStep's
       backwards integration of the the adjoint problem. */
    retval = ARKodeSetUserData(
        &arkode_mem_adj,
        Some(Box::new(
            adj_stepper_ptr.as_ref().expect("adj_stepper_ptr").clone(),
        )),
    );
    if retval != 0 {
        arkProcessError(
            Some(ark_mem),
            retval,
            line!() as i32,
            "ERKStepCreateAdjointStepper",
            file!(),
            "ARKodeSetUserData failed",
        );
        return retval;
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  Utility routines for ERKStep to serve as an MRIStepInnerStepper
  ---------------------------------------------------------------*/

/*------------------------------------------------------------------------------
  erkStep_ApplyForcing

  Determines the linear combination coefficients and vectors to apply forcing
  at a given value of the independent variable (t).  This occurs through
  appending coefficients and N_Vector pointers to the underlying cvals and Xvecs
  arrays in the step_mem structure.  The dereferenced input *nvec should indicate
  the next available entry in the cvals/Xvecs arrays.  The input 's' is a
  scaling factor that should be applied to each of these coefficients.
  ----------------------------------------------------------------------------*/

pub fn erkStep_ApplyForcing(
    step_mem: &mut ARKodeERKStepMemRec,
    stage_times: &[sunrealtype],
    stage_coefs: &[sunrealtype],
    jmax: i32,
    nvec: &mut i32,
) {
    /* Shortcuts to step_mem data (`vals`/`vecs` alias step_mem->cvals /
       step_mem->Xvecs; the fields are written in place below) */
    let tshift = step_mem.tshift;
    let tscale = step_mem.tscale;
    let nforcing = step_mem.nforcing;

    /* Offset into vals and vecs arrays */
    let offset = *nvec;

    /* Initialize scaling values, set vectors */
    for k in 0..nforcing {
        step_mem.cvals[(offset + k) as usize] = ZERO;
        let forcing_k = step_mem.forcing[k as usize].clone();
        step_mem.Xvecs[(offset + k) as usize] = Some(forcing_k);
    }

    for j in 0..jmax {
        let tau = (stage_times[j as usize] - tshift) / tscale;
        let mut taui = ONE;

        for k in 0..nforcing {
            step_mem.cvals[(offset + k) as usize] += stage_coefs[j as usize] * taui;
            taui *= tau;
        }
    }

    /* Update vector count for linear combination */
    *nvec += nforcing;
}

/*------------------------------------------------------------------------------
  erkStep_SetInnerForcing

  Sets an array of coefficient vectors for a time-dependent external polynomial
  forcing term in the ODE RHS i.e., y' = f(t,y) + p(t). This function is
  primarily intended for use with multirate integration methods (e.g., MRIStep)
  where ERKStep is used to solve a modified ODE at a fast time scale. The
  polynomial is of the form

  p(t) = sum_{i = 0}^{nvecs - 1} forcing[i] * ((t - tshift) / (tscale))^i

  where tshift and tscale are used to normalize the time t (e.g., with MRIGARK
  methods).
  ----------------------------------------------------------------------------*/

pub fn erkStep_SetInnerForcing(
    ark_mem: &ARKodeMem,
    tshift: sunrealtype,
    tscale: sunrealtype,
    forcing: &[N_Vector],
    nvecs: i32,
) -> i32 {
    /* access ARKodeERKStepMem structure */
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "erkStep_SetInnerForcing",
            file!(),
            MSG_ERKSTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    if nvecs > 0 {
        /* store forcing inputs */
        {
            let mut step_mem = erkStep_mem_mut(ark_mem);
            step_mem.tshift = tshift;
            step_mem.tscale = tscale;
            step_mem.forcing = forcing.to_vec();
            step_mem.nforcing = nvecs;
        }

        /* If cvals and Xvecs are not allocated then erkStep_Init has not been
           called and the number of stages has not been set yet. These arrays will
           be allocated in erkStep_Init and take into account the value of nforcing.
           On subsequent calls will check if enough space has allocated in case
           nforcing has increased since the original allocation. */
        let (cvals_alloc, Xvecs_alloc, nfusedopvecs, stages) = {
            let step_mem = erkStep_mem_mut(ark_mem);
            (
                !step_mem.cvals.is_empty(),
                !step_mem.Xvecs.is_empty(),
                step_mem.nfusedopvecs,
                step_mem.stages,
            )
        };
        if cvals_alloc && Xvecs_alloc {
            /* check if there are enough reusable arrays for fused operations */
            if (nfusedopvecs - nvecs) < (stages + 1) {
                /* free current work space */
                if cvals_alloc {
                    erkStep_mem_mut(ark_mem).cvals = Vec::new();
                    ark_mem.borrow_mut().lrw -= nfusedopvecs as i64;
                }
                if Xvecs_alloc {
                    erkStep_mem_mut(ark_mem).Xvecs = Vec::new();
                    ark_mem.borrow_mut().liw -= nfusedopvecs as i64;
                }

                /* allocate reusable arrays for fused vector operations
                   (the C calloc-failure branches cannot be observed in Rust) */
                let nfusedopvecs = stages + 1 + nvecs;
                erkStep_mem_mut(ark_mem).nfusedopvecs = nfusedopvecs;

                erkStep_mem_mut(ark_mem).cvals = vec![ZERO; nfusedopvecs as usize];
                ark_mem.borrow_mut().lrw += nfusedopvecs as i64;

                erkStep_mem_mut(ark_mem).Xvecs = vec![None; nfusedopvecs as usize];
                ark_mem.borrow_mut().liw += nfusedopvecs as i64;
            }
        }
    } else {
        /* disable forcing */
        let mut step_mem = erkStep_mem_mut(ark_mem);
        step_mem.tshift = ZERO;
        step_mem.tscale = ONE;
        step_mem.forcing = Vec::new();
        step_mem.nforcing = 0;
    }

    ARK_SUCCESS
}

/*===============================================================
  EOF
  ===============================================================*/
