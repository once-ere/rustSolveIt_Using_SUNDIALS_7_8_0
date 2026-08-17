//! Port of `src/arkode/arkode_lsrkstep.c` together with
//! `src/arkode/arkode_lsrkstep_impl.h` and the constants/typedefs of
//! `include/arkode/arkode_lsrkstep.h` (the public header folds into the
//! matching module, per the workspace module-naming rule). The optional
//! input/output routines of `src/arkode/arkode_lsrkstep_io.c` live in
//! `arkode_lsrkstep_io.rs`.
//!
//! Build config notes: `SUNDIALS_LOGGING_LEVEL=2`, so every
//! `SUNLogInfo`/`SUNLogInfoIf`/`SUNLogExtraDebugVec` compiles away and is
//! omitted at translation time; profiling is off; `SUNAssert`/`SUNCheck*`
//! are release no-ops.
//!
//! Binding notes:
//!  * `step_mem` holds `ARKodeLSRKStepMemRec` BY VALUE inside
//!    `ARKodeMemRec::step_mem`; `lsrkStep_mem_mut` is the single
//!    module-local downcast accessor. The guard it returns IS a
//!    `borrow_mut` of the mem, so it is never held across a user
//!    callback, an N_Vector operation, a `SUNDomEigEstimator` call, or a
//!    second borrow of the same mem.
//!  * C `void* user_data` is `Option<Box<dyn Any>>`: every invoker
//!    `Option::take`s the box, calls, and restores it on **every** path.
//!  * The reusable fused-op scratch arrays (`step_mem->cvals` /
//!    `step_mem->Xvecs`) keep their allocation and `lrw`/`liw` accounting
//!    in `lsrkStep_Init`/`lsrkStep_Free`, but each fused-op site fills
//!    equivalent locals: C writes the entries and consumes them inside one
//!    statement group, never reading them across calls, and the step_mem
//!    borrow cannot be held across `N_VLinearCombination`.
//!  * C `int` NULL-pointer guards that the Rust type system makes
//!    impossible (`rhs`, `y0`, `sunctx`, `ark_mem`, and every `T*`
//!    out-parameter) are dropped; the remaining ones are kept verbatim.

use sundials_core::sundials_libm::SunMath;
use std::any::Any;
use std::cell::RefMut;

use sundials_core::sundials_context::SUNContext;
use sundials_core::sundials_domeigestimator::{
    SUNDomEigEstimator, SUNDomEigEstimator_Estimate, SUNDomEigEstimator_GetNumIters,
    SUNDomEigEstimator_Initialize, SUNDomEigEstimator_SetNumPreprocessIters,
    SUNDomEigEstimator_Write,
};
use sundials_core::sundials_errors::SUN_SUCCESS;
use sundials_core::sundials_math::{SUNMAX, SUNRabs, SUNRceil, SUNRround, SUNRsqrt, SUNSQR};
use sundials_core::sundials_nvector::{
    N_VLinearCombination, N_VLinearSum, N_VScale, N_VWrmsNorm, N_Vector,
};
use sundials_core::sundials_types::*;
use sundials_core::sundials_utils::{sun_format_g, sun_format_sg, SUNFile};

use crate::arkode::{arkCreate, arkEwtSetSmallReal, arkInit, ARKodeFree};
use crate::arkode_impl::*;
use crate::arkode_io::ARKodeSetInterpolantType;
use crate::arkode_lsrkstep_io::{
    lsrkStep_GetEstLocalErrors, lsrkStep_GetNumRhsEvals, lsrkStep_GetStageIndex,
    lsrkStep_PrintAllStats, lsrkStep_SetDefaults, lsrkStep_SetOptions, lsrkStep_WriteParameters,
    LSRKStepSetSSPMethod, LSRKStepSetSTSMethod,
};

/*===============================================================
  LSRKStep module constants (arkode_lsrkstep_impl.h)
  ===============================================================*/

pub const STAGE_MAX_LIMIT_DEFAULT: i32 = 200;
pub const DOM_EIG_SAFETY_DEFAULT: sunrealtype = 1.01;
pub const RKC_DAMPING_DEFAULT: sunrealtype = 2.0 / 13.0;
pub const DOM_EIG_FREQ_DEFAULT: i64 = 25;
pub const DOM_EIG_NUM_WARMUPS_DEFAULT: i32 = 0;
/* use DEE's default value */
pub const DOM_EIG_NUM_INIT_WARMUPS_DEFAULT: i32 = -1;

/// `SIX` comes from `arkode_interp_impl.h`, which `arkode_lsrkstep.c`
/// `#include`s; the interpolation module keeps its own copy (contract §7),
/// so this one is module-local too.
pub const SIX: sunrealtype = 6.0;

/*===============================================================
  LSRK time step module private math function macros
  (arkode_lsrkstep_impl.h)
  ===============================================================*/

/// C `SUNRlog(x)` = `log(x)` in double precision.
pub fn SUNRlog(x: sunrealtype) -> sunrealtype {
    x.sun_ln()
}

/// C `SUNRsinh(x)` = `sinh(x)` in double precision.
pub fn SUNRsinh(x: sunrealtype) -> sunrealtype {
    x.sun_sinh()
}

/// C `SUNRcosh(x)` = `cosh(x)` in double precision.
pub fn SUNRcosh(x: sunrealtype) -> sunrealtype {
    x.sun_cosh()
}

/// C `SUNRacosh(x)` = `acosh(x)` in double precision.
pub fn SUNRacosh(x: sunrealtype) -> sunrealtype {
    x.sun_acosh()
}

/*===============================================================
  LSRKStep constants (include/arkode/arkode_lsrkstep.h)
  ===============================================================*/

/// C `ARKDomEigFn`. `fn` is a Rust keyword, so the third parameter is
/// `fn_` (the same rename `ark_mem->fn` gets in the contract).
pub type ARKDomEigFn = fn(
    t: sunrealtype,
    y: &N_Vector,
    fn_: &N_Vector,
    lambdaR: &mut sunrealtype,
    lambdaI: &mut sunrealtype,
    user_data: &mut Option<Box<dyn Any>>,
    temp1: &N_Vector,
    temp2: &N_Vector,
    temp3: &N_Vector,
) -> i32;

/// C `enum ARKODE_LSRKMethodType` (values 0..4; `calloc` zero =
/// `ARKODE_LSRK_RKC_2`, hence `Default`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ARKODE_LSRKMethodType {
    #[default]
    ARKODE_LSRK_RKC_2,
    ARKODE_LSRK_RKL_2,
    ARKODE_LSRK_SSP_S_2,
    ARKODE_LSRK_SSP_S_3,
    ARKODE_LSRK_SSP_10_4,
}
pub use ARKODE_LSRKMethodType::*;

/*===============================================================
  Reusable LSRKStep Error Messages
  ===============================================================*/

pub const MSG_LSRKSTEP_NO_MEM: &str = "Time step module memory is NULL.";

/*===============================================================
  LSRK time step module data structure
  ===============================================================*/

/*---------------------------------------------------------------
  Types : struct ARKodeLSRKStepMemRec, ARKodeLSRKStepMem
  ---------------------------------------------------------------
  This structure contains fields to perform an explicit
  Runge-Kutta time step.
  ---------------------------------------------------------------*/
pub struct ARKodeLSRKStepMemRec {
    /* LSRK problem specification */
    pub fe: Option<ARKRhsFn>,
    pub dom_eig_fn: Option<ARKDomEigFn>,

    pub q: i32, /* method order               */
    pub p: i32, /* embedding order            */

    pub istage: i32,     /* current stage            */
    pub req_stages: i32, /* number of stages in step */

    pub LSRKmethod: ARKODE_LSRKMethodType,

    /* Counters and stats*/
    pub nfe: i64,               /* num fe calls       */
    pub nfeDQ: i64,             /* num fe calls for difference quotient approximation */
    pub dom_eig_num_evals: i64, /* num of dom_eig computations   */
    pub stage_max: i32,         /* num of max stages used      */
    pub stage_max_limit: i32,   /* max allowed num of stages     */
    pub dom_eig_nst: i64, /* num of step at which the last domainant eigenvalue was computed  */
    pub step_nst: i64,      /* The number of successful steps. */
    pub num_dee_iters: i64, /* number of iterations in the DEE estimates */

    /* Spectral info */
    pub lambdaR: sunrealtype,             /* Real part of the dominated eigenvalue*/
    pub lambdaI: sunrealtype,             /* Imaginary part of the dominated eigenvalue*/
    pub spectral_radius: sunrealtype,     /* spectral radius*/
    pub spectral_radius_max: sunrealtype, /* max spectral radius*/
    pub spectral_radius_min: sunrealtype, /* min spectral radius*/
    pub dom_eig_safety: sunrealtype, /* some safety factor for the user provided dom_eig*/
    pub rkc_damping: sunrealtype,    /* damping parameter for RKC methods*/
    pub dom_eig_freq: i64, /* indicates dom_eig update after dom_eig_freq successful steps*/
    pub num_init_warmups: i32, /* number of warm-ups in the first DEE estimates */
    pub num_warmups: i32,  /* number of warm-ups in succeeding DEE estimates */

    pub DEE: Option<SUNDomEigEstimator>, /* DomEig estimator*/

    /* Flags */
    pub dom_eig_update: sunbooleantype, /* flag indicating new dom_eig is needed */
    pub const_Jac: sunbooleantype,      /* flag indicating Jacobian is constant */
    pub dom_eig_is_current: sunbooleantype, /* SUNTRUE if dom_eig has been evaluated at tn */
    pub use_ellipse: sunbooleantype, /* flag indicating whether to use ellipse or exact stability region for stability checks */
    pub is_SSP: sunbooleantype,      /* flag indicating SSP method*/
    pub init_warmup: sunbooleantype, /* flag indicating initial warm-up*/

    /// Reusable fused vector operation array (C `sunrealtype* cvals`);
    /// empty == C `NULL`.
    pub cvals: Vec<sunrealtype>,
    /// Reusable fused vector operation array (C `N_Vector* Xvecs`);
    /// empty == C `NULL`.
    pub Xvecs: Vec<Option<N_Vector>>,
    pub nfusedopvecs: i32, /* length of cvals and Xvecs arrays */
}

impl ARKodeLSRKStepMemRec {
    /// C `calloc(1, sizeof(*step_mem))` in `lsrkStep_Create_Commons`.
    pub fn zeroed() -> ARKodeLSRKStepMemRec {
        ARKodeLSRKStepMemRec {
            fe: None,
            dom_eig_fn: None,
            q: 0,
            p: 0,
            istage: 0,
            req_stages: 0,
            LSRKmethod: ARKODE_LSRK_RKC_2,
            nfe: 0,
            nfeDQ: 0,
            dom_eig_num_evals: 0,
            stage_max: 0,
            stage_max_limit: 0,
            dom_eig_nst: 0,
            step_nst: 0,
            num_dee_iters: 0,
            lambdaR: 0.0,
            lambdaI: 0.0,
            spectral_radius: 0.0,
            spectral_radius_max: 0.0,
            spectral_radius_min: 0.0,
            dom_eig_safety: 0.0,
            rkc_damping: 0.0,
            dom_eig_freq: 0,
            num_init_warmups: 0,
            num_warmups: 0,
            DEE: None,
            dom_eig_update: SUNFALSE,
            const_Jac: SUNFALSE,
            dom_eig_is_current: SUNFALSE,
            use_ellipse: SUNFALSE,
            is_SSP: SUNFALSE,
            init_warmup: SUNFALSE,
            cvals: Vec::new(),
            Xvecs: Vec::new(),
            nfusedopvecs: 0,
        }
    }
}

/// Downcast accessor for the LSRKStep content held by value in
/// `ark_mem.step_mem` (C would blindly cast the `void*`; misuse maps to a
/// deterministic panic). NEVER hold the returned guard across a callback,
/// an N_Vector operation, a `SUNDomEigEstimator` call, or another borrow
/// of the same mem.
pub fn lsrkStep_mem_mut(ark_mem: &ARKodeMem) -> RefMut<'_, ARKodeLSRKStepMemRec> {
    RefMut::map(ark_mem.borrow_mut(), |m| {
        m.step_mem
            .as_mut()
            .expect("step_mem set")
            .downcast_mut::<ARKodeLSRKStepMemRec>()
            .expect("LSRKStep step memory")
    })
}

/* =================================================================
 * User-callback invokers (the `user_data` token is taken out of the mem
 * around every call and restored afterwards; no mem borrow is held
 * across the call)
 * =================================================================*/

/// C `step_mem->fe(t, y, ydot, ark_mem->user_data)`.
fn lsrk_call_fe(ark_mem: &ARKodeMem, t: sunrealtype, y: &N_Vector, ydot: &N_Vector) -> i32 {
    let fe = lsrkStep_mem_mut(ark_mem).fe.expect("fe set");
    let mut user_data = ark_mem.borrow_mut().user_data.take();
    let retval = fe(t, y, ydot, &mut user_data);
    ark_mem.borrow_mut().user_data = user_data;
    retval
}

/// One `(t, y, user_data)` hook call: `ARKPreRhsFn`, `ARKPostProcessFn`
/// and `ARKStagePredictFn` all share this signature in C and in Rust.
fn lsrk_call_processfn(
    ark_mem: &ARKodeMem,
    f: ARKPostProcessFn,
    t: sunrealtype,
    y: &N_Vector,
) -> i32 {
    let mut user_data = ark_mem.borrow_mut().user_data.take();
    let retval = f(t, y, &mut user_data);
    ark_mem.borrow_mut().user_data = user_data;
    retval
}

/// C `step_mem->dom_eig_fn(tn, yn, fn, &lambdaR, &lambdaI, user_data,
/// tempv1, tempv2, tempv3)`. `lambdaR`/`lambdaI` are C's addresses of the
/// step_mem fields: the values are copied out, handed to the callback as
/// `&mut` locals, and written back by the caller.
#[allow(clippy::too_many_arguments)]
fn lsrk_call_dom_eig_fn(
    ark_mem: &ARKodeMem,
    f: ARKDomEigFn,
    t: sunrealtype,
    y: &N_Vector,
    fn_: &N_Vector,
    lambdaR: &mut sunrealtype,
    lambdaI: &mut sunrealtype,
    temp1: &N_Vector,
    temp2: &N_Vector,
    temp3: &N_Vector,
) -> i32 {
    let mut user_data = ark_mem.borrow_mut().user_data.take();
    let retval = f(t, y, fn_, lambdaR, lambdaI, &mut user_data, temp1, temp2, temp3);
    ark_mem.borrow_mut().user_data = user_data;
    retval
}

/*===============================================================
  Exported functions
  ===============================================================*/

pub fn LSRKStepCreateSTS(
    rhs: ARKRhsFn,
    t0: sunrealtype,
    y0: &N_Vector,
    sunctx: &SUNContext,
) -> Option<ARKodeMem> {
    /* Create shared LSRKStep memory structure */
    let ark_mem = lsrkStep_Create_Commons(rhs, t0, y0, sunctx);

    /* C continues with a NULL `ark_mem`, so that `LSRKStepSetSTSMethod`
       reports MSG_ARK_NO_MEM and `lsrkStep_Free(NULL)` returns at once;
       both are unreachable through `&ARKodeMem`. */
    let ark_mem = ark_mem?;

    /* set default ARKODE_LSRK_RKC_2 method */
    let retval = LSRKStepSetSTSMethod(&ark_mem, ARKODE_LSRK_RKC_2);
    if retval != ARK_SUCCESS {
        lsrkStep_Free(&ark_mem);
        return None;
    }

    Some(ark_mem)
}

pub fn LSRKStepCreateSSP(
    rhs: ARKRhsFn,
    t0: sunrealtype,
    y0: &N_Vector,
    sunctx: &SUNContext,
) -> Option<ARKodeMem> {
    /* Create shared LSRKStep memory structure */
    let ark_mem = lsrkStep_Create_Commons(rhs, t0, y0, sunctx);

    let ark_mem = ark_mem?;

    /* set default ARKODE_LSRK_SSP_S_2 method */
    let retval = LSRKStepSetSSPMethod(&ark_mem, ARKODE_LSRK_SSP_S_2);
    if retval != ARK_SUCCESS {
        lsrkStep_Free(&ark_mem);
        return None;
    }

    Some(ark_mem)
}

/*---------------------------------------------------------------
  LSRKStepReInitSTS:

  This routine re-initializes the LSRK STS module to solve a new
  problem of the same size as was previously solved. This routine
  should also be called when the problem dynamics or desired solvers
  have changed dramatically, so that the problem integration should
  resume as if started from scratch.

  Note all internal counters are set to 0 on re-initialization.
  ---------------------------------------------------------------*/
pub fn LSRKStepReInitSTS(
    arkode_mem: &ARKodeMem,
    rhs: ARKRhsFn,
    t0: sunrealtype,
    y0: &N_Vector,
) -> i32 {
    lsrkStep_ReInit_Commons(arkode_mem, rhs, t0, y0)
}

/*---------------------------------------------------------------
  LSRKStepReInitSSP:

  This routine re-initializes the LSRK SSP module to solve a new
  problem of the same size as was previously solved. This routine
  should also be called when the problem dynamics or desired solvers
  have changed dramatically, so that the problem integration should
  resume as if started from scratch.

  Note all internal counters are set to 0 on re-initialization.
  ---------------------------------------------------------------*/
pub fn LSRKStepReInitSSP(
    arkode_mem: &ARKodeMem,
    rhs: ARKRhsFn,
    t0: sunrealtype,
    y0: &N_Vector,
) -> i32 {
    lsrkStep_ReInit_Commons(arkode_mem, rhs, t0, y0)
}

/*===============================================================
  Interface routines supplied to ARKODE
  ===============================================================*/

/*---------------------------------------------------------------
  lsrkStep_Create_Commons:

  A submodule for creating the common features of
  LSRKStepCreateSTS and LSRKStepCreateSSP.
  ---------------------------------------------------------------*/
pub fn lsrkStep_Create_Commons(
    rhs: ARKRhsFn,
    t0: sunrealtype,
    y0: &N_Vector,
    sunctx: &SUNContext,
) -> Option<ARKodeMem> {
    /* C checks `rhs == NULL` (MSG_ARK_NULL_F), `y0 == NULL`
       (MSG_ARK_NULL_Y0) and `sunctx == NULL` (MSG_ARK_NULL_SUNCTX); all
       three are impossible through the Rust parameter types. */

    /* Create ark_mem structure and set default values */
    let ark_mem = match arkCreate(sunctx) {
        Some(ark_mem) => ark_mem,
        None => {
            arkProcessError(
                None,
                ARK_MEM_NULL,
                line!() as i32,
                "lsrkStep_Create_Commons",
                file!(),
                MSG_ARK_NO_MEM,
            );
            return None;
        }
    };

    /* Allocate ARKodeLSRKStepMem structure, and initialize to zero */
    let step_mem = ARKodeLSRKStepMemRec::zeroed();

    /* Attach step_mem structure and function pointers to ark_mem */
    {
        let mut m = ark_mem.borrow_mut();
        m.step_init = Some(lsrkStep_Init);
        m.step_fullrhs = Some(lsrkStep_FullRHS);
        m.step = Some(lsrkStep_TakeStepRKC);
        m.step_printallstats = Some(lsrkStep_PrintAllStats);
        m.step_writeparameters = Some(lsrkStep_WriteParameters);
        m.step_free = Some(lsrkStep_Free);
        m.step_printmem = Some(lsrkStep_PrintMem);
        m.step_setoptions = Some(lsrkStep_SetOptions);
        m.step_setdefaults = Some(lsrkStep_SetDefaults);
        m.step_getnumrhsevals = Some(lsrkStep_GetNumRhsEvals);
        m.step_getestlocalerrors = Some(lsrkStep_GetEstLocalErrors);
        m.step_getstageindex = Some(lsrkStep_GetStageIndex);
        m.step_mem = Some(Box::new(step_mem));
        m.step_supports_adaptive = SUNTRUE;
    }

    /* Set default values for optional inputs */
    let retval = lsrkStep_SetDefaults(&ark_mem);
    if retval != ARK_SUCCESS {
        arkProcessError(
            Some(&ark_mem),
            retval,
            line!() as i32,
            "lsrkStep_Create_Commons",
            file!(),
            "Error setting default solver options",
        );
        let mut handle = Some(ark_mem);
        ARKodeFree(&mut handle);
        return None;
    }

    {
        let mut step_mem = lsrkStep_mem_mut(&ark_mem);

        /* Copy the input parameters into ARKODE state */
        step_mem.fe = Some(rhs);

        /* Initialize spectral radius info */
        step_mem.lambdaR = ZERO;
        step_mem.lambdaI = ZERO;
        step_mem.spectral_radius = ZERO;
        step_mem.spectral_radius_max = ZERO;
        step_mem.spectral_radius_min = ZERO;

        /* Initialize flags */
        step_mem.dom_eig_update = SUNTRUE;
        step_mem.dom_eig_is_current = SUNFALSE;
        step_mem.is_SSP = SUNFALSE;
        step_mem.init_warmup = SUNTRUE;

        /* Set NULL for dom_eig_fn */
        step_mem.dom_eig_fn = None;

        /* Set NULL for DEE */
        step_mem.DEE = None;

        /* Initialize all the counters */
        step_mem.nfe = 0;
        step_mem.nfeDQ = 0;
        step_mem.stage_max = 0;
        step_mem.dom_eig_num_evals = 0;
        step_mem.stage_max_limit = STAGE_MAX_LIMIT_DEFAULT;
        step_mem.dom_eig_nst = 0;
        step_mem.num_dee_iters = 0;
    }

    /* Initialize main ARKODE infrastructure */
    let retval = arkInit(&ark_mem, t0, y0, FIRST_INIT);
    if retval != ARK_SUCCESS {
        arkProcessError(
            Some(&ark_mem),
            retval,
            line!() as i32,
            "lsrkStep_Create_Commons",
            file!(),
            "Unable to initialize main ARKODE infrastructure",
        );
        let mut handle = Some(ark_mem);
        ARKodeFree(&mut handle);
        return None;
    }

    /* Specify preferred interpolation type */
    let _ = ARKodeSetInterpolantType(&ark_mem, ARK_INTERP_LAGRANGE);

    Some(ark_mem)
}

/*---------------------------------------------------------------
  lsrkStep_ReInit_Commons:

  A submodule designed to reinitialize the common features of
  LSRKStepCreateSTS and LSRKStepCreateSSP.
  ---------------------------------------------------------------*/
pub fn lsrkStep_ReInit_Commons(
    arkode_mem: &ARKodeMem,
    rhs: ARKRhsFn,
    t0: sunrealtype,
    y0: &N_Vector,
) -> i32 {
    /* access ARKodeLSRKStepMem structure */
    let retval = lsrkStep_AccessARKODEStepMem(arkode_mem, "lsrkStep_ReInit_Commons");
    if retval != ARK_SUCCESS {
        return retval;
    }
    let ark_mem = arkode_mem;

    /* Check if ark_mem was allocated */
    if !ark_mem.borrow().MallocDone {
        arkProcessError(
            Some(ark_mem),
            ARK_NO_MALLOC,
            line!() as i32,
            "lsrkStep_ReInit_Commons",
            file!(),
            MSG_ARK_NO_MALLOC,
        );
        return ARK_NO_MALLOC;
    }

    /* C checks `rhs == NULL` (MSG_ARK_NULL_F) and `y0 == NULL`
       (MSG_ARK_NULL_Y0); both impossible through the Rust types. */

    /* Copy the input parameters into ARKODE state */
    lsrkStep_mem_mut(ark_mem).fe = Some(rhs);

    /* Initialize main ARKODE infrastructure */
    let retval = arkInit(arkode_mem, t0, y0, FIRST_INIT);
    if retval != ARK_SUCCESS {
        arkProcessError(
            Some(ark_mem),
            retval,
            line!() as i32,
            "lsrkStep_ReInit_Commons",
            file!(),
            "Unable to initialize main ARKODE infrastructure",
        );
        return retval;
    }

    /* Initialize all the counters, flags and stats */
    {
        let mut step_mem = lsrkStep_mem_mut(ark_mem);
        step_mem.nfe = 0;
        step_mem.nfeDQ = 0;
        step_mem.dom_eig_num_evals = 0;
        step_mem.stage_max = 0;
        step_mem.lambdaR = ZERO;
        step_mem.lambdaI = ZERO;
        step_mem.spectral_radius = ZERO;
        step_mem.spectral_radius_max = 0.0;
        step_mem.spectral_radius_min = 0.0;
        step_mem.dom_eig_nst = 0;
        step_mem.num_dee_iters = 0;
        step_mem.dom_eig_update = SUNTRUE;
        step_mem.dom_eig_is_current = SUNFALSE;
        step_mem.init_warmup = SUNTRUE;
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  lsrkStep_Init:

  This routine is called just prior to performing internal time
  steps (after all user "set" routines have been called) from
  within arkInitialSetup.

  With initialization types FIRST_INIT this routine performs
  setup and allocations needed for the method and sets
  the call_fullrhs flag.

  With other initialization types, this routine does nothing.
  ---------------------------------------------------------------*/
pub fn lsrkStep_Init(ark_mem: &ARKodeMem, init_type: i32) -> i32 {
    /* access ARKodeLSRKStepMem structure */
    let retval = lsrkStep_AccessStepMem(ark_mem, "lsrkStep_Init");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* immediately return if resize or reset */
    if init_type == RESIZE_INIT || init_type == RESET_INIT {
        return ARK_SUCCESS;
    }

    /* enforce use of arkEwtSmallReal if using a fixed step size
       and an internal error weight function */
    let (fixedstep, user_efun) = {
        let m = ark_mem.borrow();
        (m.fixedstep, m.user_efun)
    };
    if fixedstep && !user_efun {
        /* C stores the raw `ark_mem` self-pointer in `e_data`; the port
           boxes a handle clone (the locked CVODE `cv_e_data` rendering).
           `ARKodeFree` must clear `e_data` to break the Rc cycle. */
        let token: Box<dyn Any> = Box::new(ark_mem.clone());
        let mut m = ark_mem.borrow_mut();
        m.user_efun = SUNFALSE;
        m.efun = Some(arkEwtSetSmallReal);
        m.e_data = Some(token);
    }

    /* Check if user has provided dom_eig_fn or DEE */
    let (is_SSP, has_dom_eig_fn, has_DEE) = {
        let step_mem = lsrkStep_mem_mut(ark_mem);
        (
            step_mem.is_SSP,
            step_mem.dom_eig_fn.is_some(),
            step_mem.DEE.is_some(),
        )
    };
    if !is_SSP && !has_dom_eig_fn && !has_DEE {
        arkProcessError(
            Some(ark_mem),
            ARK_DOMEIG_FAIL,
            line!() as i32,
            "lsrkStep_Init",
            file!(),
            "STS methods require either a user provided dominant eigenvalue function or a \
             SUNDomEigEstimator",
        );
        return ARK_DOMEIG_FAIL;
    }

    /* Initialize the DEE */
    let DEE = lsrkStep_mem_mut(ark_mem).DEE.clone();
    if let Some(DEE) = DEE {
        let retval = SUNDomEigEstimator_Initialize(&DEE);
        if retval != SUN_SUCCESS {
            arkProcessError(
                Some(ark_mem),
                ARK_DEE_FAIL,
                line!() as i32,
                "lsrkStep_Init",
                file!(),
                "SUNDomEigEstimator_Initialize failed",
            );
            return ARK_DEE_FAIL;
        }

        /* Set number of DEE preprocessing iterations for the initial estimate */
        let num_init_warmups = lsrkStep_mem_mut(ark_mem).num_init_warmups;
        let retval = SUNDomEigEstimator_SetNumPreprocessIters(&DEE, num_init_warmups);
        if retval != SUN_SUCCESS {
            arkProcessError(
                Some(ark_mem),
                ARK_DEE_FAIL,
                line!() as i32,
                "lsrkStep_Init",
                file!(),
                "SUNDomEigEstimator_SetNumPreprocessIters failed",
            );
            return ARK_DEE_FAIL;
        }
    }

    /* Allocate reusable arrays for fused vector interface */
    let nfusedopvecs = lsrkStep_mem_mut(ark_mem).nfusedopvecs;
    {
        let mut m = ark_mem.borrow_mut();
        let alloc_cvals;
        let alloc_Xvecs;
        {
            let step_mem = m
                .step_mem
                .as_mut()
                .expect("step_mem set")
                .downcast_mut::<ARKodeLSRKStepMemRec>()
                .expect("LSRKStep step memory");
            alloc_cvals = step_mem.cvals.is_empty();
            if alloc_cvals {
                step_mem.cvals = vec![ZERO; nfusedopvecs as usize];
            }
            alloc_Xvecs = step_mem.Xvecs.is_empty();
            if alloc_Xvecs {
                step_mem.Xvecs = vec![None; nfusedopvecs as usize];
            }
        }
        if alloc_cvals {
            m.lrw += nfusedopvecs as i64;
        }
        if alloc_Xvecs {
            m.liw += nfusedopvecs as i64; /* pointers */
        }
    }

    /* While LSRKStep does not currently call the full RHS function directly (later
       optimizations might) we do need the fn vector to always be allocated. Signaling
       to shared arkode module that full RHS evaluations are required will ensure
       fn is always allocated. */
    ark_mem.borrow_mut().call_fullrhs = SUNTRUE;

    ARK_SUCCESS
}

/*------------------------------------------------------------------------------
  lsrkStep_FullRHS:

  This is just a wrapper to call the user-supplied RHS function, f(t,y).

  This will be called in one of three 'modes':

     ARK_FULLRHS_START -> called at the beginning of a simulation i.e., at
                          (tn, yn) = (t0, y0) or (tR, yR)

     ARK_FULLRHS_END   -> called at the end of a successful step i.e, at
                          (tcur, ycur) or the start of the subsequent step i.e.,
                          at (tn, yn) = (tcur, ycur) from the end of the last
                          step

     ARK_FULLRHS_OTHER -> called elsewhere (e.g. for dense output)

  If this function is called in ARK_FULLRHS_START the RHS function is always
  evaluated.

  In ARK_FULLRHS_END mode we evaluate the RHS if an SSP method is being
  used otherwise we copy the RHS evaluation from the end of the STS step.

  ARK_FULLRHS_OTHER mode is only called for dense output in-between steps, or
  when estimating the initial time step size, so we strive to store the
  intermediate parts so that they do not interfere with the other two modes.
  ----------------------------------------------------------------------------*/
pub fn lsrkStep_FullRHS(
    ark_mem: &ARKodeMem,
    t: sunrealtype,
    y: &N_Vector,
    f: &N_Vector,
    mode: i32,
) -> i32 {
    /* access ARKodeLSRKStepMem structure */
    let retval = lsrkStep_AccessStepMem(ark_mem, "lsrkStep_FullRHS");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* perform RHS functions contingent on 'mode' argument */
    match mode {
        ARK_FULLRHS_START => {
            /* compute the RHS */
            if !ark_mem.borrow().fn_is_current {
                /* call the user-supplied pre-rhs function (if supplied) */
                let PreRhsFn = ark_mem.borrow().PreRhsFn;
                if let Some(PreRhsFn) = PreRhsFn {
                    let retval = lsrk_call_processfn(ark_mem, PreRhsFn, t, y);
                    if retval != 0 {
                        return ARK_PRERHSFN_FAIL;
                    }
                }
                let retval = lsrk_call_fe(ark_mem, t, y, f);
                lsrkStep_mem_mut(ark_mem).nfe += 1;
                if retval != 0 {
                    arkProcessError(
                        Some(ark_mem),
                        ARK_RHSFUNC_FAIL,
                        line!() as i32,
                        "lsrkStep_FullRHS",
                        file!(),
                        &MSG_ARK_RHSFUNC_FAILED(t),
                    );
                    return ARK_RHSFUNC_FAIL;
                }
            }
        }

        ARK_FULLRHS_END => {
            /* No further action is needed if STS since the currently available STS methods
               evaluate the RHS at the end of each time step. If the stepper is an SSP, fn is
               updated and reused at the beginning of the step unless
               ark_mem->fn_is_current is changed by ARKODE. */
            if lsrkStep_mem_mut(ark_mem).is_SSP {
                /* call the user-supplied pre-rhs function (if supplied) */
                let PreRhsFn = ark_mem.borrow().PreRhsFn;
                if let Some(PreRhsFn) = PreRhsFn {
                    let retval = lsrk_call_processfn(ark_mem, PreRhsFn, t, y);
                    if retval != 0 {
                        return ARK_PRERHSFN_FAIL;
                    }
                }
                let fn_ = ark_mem.borrow().fn_.clone().expect("fn");
                let retval = lsrk_call_fe(ark_mem, t, y, &fn_);
                lsrkStep_mem_mut(ark_mem).nfe += 1;
                if retval != 0 {
                    arkProcessError(
                        Some(ark_mem),
                        ARK_RHSFUNC_FAIL,
                        line!() as i32,
                        "lsrkStep_FullRHS",
                        file!(),
                        &MSG_ARK_RHSFUNC_FAILED(t),
                    );
                    return ARK_RHSFUNC_FAIL;
                }
                ark_mem.borrow_mut().fn_is_current = SUNTRUE;
            }
            let fn_ = ark_mem.borrow().fn_.clone().expect("fn");
            N_VScale(ONE, &fn_, f);
        }

        ARK_FULLRHS_OTHER => {
            /* call the user-supplied pre-rhs function (if supplied) */
            let PreRhsFn = ark_mem.borrow().PreRhsFn;
            if let Some(PreRhsFn) = PreRhsFn {
                let retval = lsrk_call_processfn(ark_mem, PreRhsFn, t, y);
                if retval != 0 {
                    return ARK_PRERHSFN_FAIL;
                }
            }

            /* call f */
            let retval = lsrk_call_fe(ark_mem, t, y, f);
            lsrkStep_mem_mut(ark_mem).nfe += 1;
            if retval != 0 {
                arkProcessError(
                    Some(ark_mem),
                    ARK_RHSFUNC_FAIL,
                    line!() as i32,
                    "lsrkStep_FullRHS",
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
                "lsrkStep_FullRHS",
                file!(),
                "Unknown full RHS mode",
            );
            return ARK_RHSFUNC_FAIL;
        }
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  lsrkStep_TakeStepRKC:

  This routine serves the primary purpose of the LSRKStepRKC module:
  it performs a single RKC step (with embedding, if possible).

  The output variable dsmPtr should contain estimate of the
  weighted local error if an embedding is present; otherwise it
  should be 0.

  The variables (ark_mem->tcur, ark_mem->ycur) should
  contain the current time and solution at the end of this time step.

  The input/output variable nflagPtr is generally used in ARKODE
  to gauge the convergence of any algebraic solvers. However, since
  the STS step routines do not involve an algebraic solve, this variable
  instead serves to identify possible ARK_RETRY_STEP returns within this
  routine.

  The return value from this routine is:
            0 => step completed successfully
           >0 => step encountered recoverable failure;
                 reduce step and retry (if possible)
                 ARK_RETRY_STEP indicates that the required stage
                 number has reached the stage_max_limit with the
                 current value of h. The step is then returned to
                 adjust the step size.
           <0 => step encountered unrecoverable failure
  ---------------------------------------------------------------*/
pub fn lsrkStep_TakeStepRKC(
    ark_mem: &ARKodeMem,
    dsmPtr: &mut sunrealtype,
    nflagPtr: &mut i32,
) -> i32 {
    let mut stability_norm: sunrealtype = ZERO;

    let p8: sunrealtype = 0.8;
    let p4: sunrealtype = 0.4;

    /* initialize algebraic solver convergence flag to success,
       temporal error estimate to zero */
    *nflagPtr = ARK_SUCCESS;
    *dsmPtr = ZERO;

    /* access ARKodeLSRKStepMem structure */
    let retval = lsrkStep_AccessStepMem(ark_mem, "lsrkStep_TakeStepRKC");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* C: `cvals`/`Xvecs` alias the step_mem scratch arrays here; each
       fused-op site below fills equivalent locals (see the module docs).
       `tmp1`/`tmp2` start as ARKODE's tempv1/tempv2 and are swapped; the
       original handles stay available for the direct `ark_mem->tempv1` /
       `ark_mem->tempv2` uses after the stage loop. */
    let tempv1 = ark_mem.borrow().tempv1.clone().expect("tempv1");
    let tempv2 = ark_mem.borrow().tempv2.clone().expect("tempv2");
    let mut tmp1 = tempv1.clone();
    let mut tmp2 = tempv2.clone();

    /* C re-reads ark_mem->h, ->tn and the yn/ycur/fn/ewt handles at each
       use; none of them is modified anywhere inside a step attempt. */
    let (h, tn, yn, ycur, fn_, ewt) = {
        let m = ark_mem.borrow();
        (
            m.h,
            m.tn,
            m.yn.clone().expect("yn"),
            m.ycur.clone().expect("ycur"),
            m.fn_.clone().expect("fn"),
            m.ewt.clone().expect("ewt"),
        )
    };

    let (rkc_damping, stage_max_limit) = {
        let step_mem = lsrkStep_mem_mut(ark_mem);
        (step_mem.rkc_damping, step_mem.stage_max_limit)
    };

    let coefz = THREE / TWO / (ONE - TWO / 15.0 * rkc_damping);

    /* Initialize the current stage index */
    {
        let mut step_mem = lsrkStep_mem_mut(ark_mem);
        step_mem.istage = 0;
        let stage_max_limit = step_mem.stage_max_limit;
        step_mem.req_stages = stage_max_limit;
    }

    /* Compute dominant eigenvalue and update stats */
    let dom_eig_update = lsrkStep_mem_mut(ark_mem).dom_eig_update;
    if dom_eig_update {
        let retval = lsrkStep_ComputeNewDomEig(ark_mem);
        if retval != ARK_SUCCESS {
            return retval;
        }
    }

    /* Compute the number of stages based on the current step size and dominant
       eigenvalue using Eq. 2.7 in Verwer et al. (2004)
       https://doi.org/10.1016/j.jcp.2004.05.002

       Note beta(s) in Eq. 2.7 is positive (i.e., beta = -zR = h * lambdaR assuming
       that h * lambdaR < 0) and we have incorporated the minus sign on zR below. We
       use the minimum number of stages (ss = 2) when zR > 0. */
    let (lambdaR, lambdaI) = {
        let step_mem = lsrkStep_mem_mut(ark_mem);
        (step_mem.lambdaR, step_mem.lambdaI)
    };
    let zR = h * lambdaR;
    let zI = h * lambdaI;
    let mut ss: i32 = if zR > ZERO {
        2
    } else {
        SUNRceil(SUNRsqrt(ONE - coefz * zR)) as i32
    };
    ss = SUNMAX(ss, 2);

    /* Check if number of stages exceeds maximum allowed.
       If so, and if adaptive stepping is enabled, reduce step size
       and return ARK_RETRY_STEP. If fixed step size, return
       ARK_MAX_STAGE_LIMIT_FAIL error. */
    if ss > stage_max_limit {
        if !ark_mem.borrow().fixedstep {
            let safety = ark_mem.borrow().hadapt_mem.as_ref().expect("hadapt_mem").safety;
            let hmax = safety * ((stage_max_limit * stage_max_limit) as sunrealtype - ONE)
                / (coefz * SUNRabs(lambdaR));
            ark_mem.borrow_mut().eta = hmax / SUNRabs(h);
            *nflagPtr = ARK_RETRY_STEP;
            ark_mem.borrow_mut().hadapt_mem.as_mut().expect("hadapt_mem").nst_exp += 1;
            return ARK_RETRY_STEP;
        } else {
            arkProcessError(
                Some(ark_mem),
                ARK_MAX_STAGE_LIMIT_FAIL,
                line!() as i32,
                "lsrkStep_TakeStepRKC",
                file!(),
                "Unable to achieve stable results: Either reduce the step size or increase the \
                 stage_max_limit",
            );
            return ARK_MAX_STAGE_LIMIT_FAIL;
        }
    }

    /* Copy ss in case it is needed for falling back to step size adaptivity below */
    let mut req_stages: i32 = ss;

    if zR < -SUN_UNIT_ROUNDOFF {
        /* We first check whether the combination of ss, step size, and dominant
          eigenvalue, is stable.  If not, we then check whether it would be stable
          when using ss = stage_max_limit -- if so, we increase ss until stability is
          obtained. Otherwise, we reject the step, resulting either in method failure
          when using fixed step sizes, or time step reduction when using adaptive
          steps. */
        let retval =
            lsrkStep_RKC_CheckStabilityNorm(ark_mem, req_stages, h, &mut stability_norm);
        if retval != ARK_SUCCESS {
            return retval;
        }

        if stability_norm > ONE - SUN_UNIT_ROUNDOFF {
            let initial_stability_norm = stability_norm;
            let mut max_stage_is_stable = SUNFALSE;

            if req_stages < stage_max_limit {
                let retval = lsrkStep_RKC_CheckStabilityNorm(
                    ark_mem,
                    stage_max_limit,
                    h,
                    &mut stability_norm,
                );
                if retval != ARK_SUCCESS {
                    return retval;
                }

                max_stage_is_stable = stability_norm <= ONE - SUN_UNIT_ROUNDOFF;
                stability_norm = initial_stability_norm;
            }

            if max_stage_is_stable {
                while (stability_norm > ONE - SUN_UNIT_ROUNDOFF) && (req_stages < stage_max_limit)
                {
                    req_stages += 1;
                    let retval =
                        lsrkStep_RKC_CheckStabilityNorm(ark_mem, req_stages, h, &mut stability_norm);
                    if retval != ARK_SUCCESS {
                        return retval;
                    }
                }
            }

            if stability_norm > ONE - SUN_UNIT_ROUNDOFF {
                if !ark_mem.borrow().fixedstep {
                    /* For adaptive simulations, we adjust the step size by the ellipse approximation */
                    let a = (TWO / THREE) * ((ss * ss) as sunrealtype - ONE)
                        * (ONE - TWO / 15.0 * rkc_damping)
                        / TWO;
                    let b = a / (if ss == 2 { 0.6 } else { 1.825 * ss as sunrealtype });

                    let safety =
                        ark_mem.borrow().hadapt_mem.as_ref().expect("hadapt_mem").safety;
                    ark_mem.borrow_mut().eta = safety * (-TWO * a * b * b * zR)
                        / (SUNSQR(b * zR) + SUNSQR(a * zI));
                    *nflagPtr = ARK_RETRY_STEP;
                    ark_mem.borrow_mut().hadapt_mem.as_mut().expect("hadapt_mem").nst_exp += 1;
                    return ARK_RETRY_STEP;
                } else {
                    arkProcessError(
                        Some(ark_mem),
                        ARK_MAX_STAGE_LIMIT_FAIL,
                        line!() as i32,
                        "lsrkStep_TakeStepRKC",
                        file!(),
                        "Unable to achieve stable results: Either reduce the step size or \
                         increase the stage_max_limit",
                    );
                    return ARK_MAX_STAGE_LIMIT_FAIL;
                }
            }
        }
    }

    {
        let mut step_mem = lsrkStep_mem_mut(ark_mem);
        step_mem.req_stages = req_stages;
        let stage_max = step_mem.stage_max;
        step_mem.stage_max = SUNMAX(step_mem.req_stages, stage_max);
    }

    /* Compute RHS function for the start of the step, if necessary. */
    let need_rhs = {
        let m = ark_mem.borrow();
        let step_nst = m
            .step_mem
            .as_ref()
            .expect("step_mem set")
            .downcast_ref::<ARKodeLSRKStepMemRec>()
            .expect("LSRKStep step memory")
            .step_nst;
        (!m.fn_is_current && m.initsetup) || (step_nst != m.nst)
    };
    if need_rhs {
        /* call the user-supplied pre-rhs function (if supplied) */
        let PreRhsFn = ark_mem.borrow().PreRhsFn;
        if let Some(PreRhsFn) = PreRhsFn {
            let retval = lsrk_call_processfn(ark_mem, PreRhsFn, tn, &yn);
            if retval != 0 {
                return ARK_PRERHSFN_FAIL;
            }
        }

        /* call fe */
        let retval = lsrk_call_fe(ark_mem, tn, &yn, &fn_);
        lsrkStep_mem_mut(ark_mem).nfe += 1;
        if retval != ARK_SUCCESS {
            return ARK_RHSFUNC_FAIL;
        }
        ark_mem.borrow_mut().fn_is_current = SUNTRUE;
    }

    /* Track the number of successful steps to determine if the previous step failed. */
    {
        let nst = ark_mem.borrow().nst;
        lsrkStep_mem_mut(ark_mem).step_nst = nst + 1;
    }

    /* Initialize constants */
    let w0 = ONE + rkc_damping / SUNSQR(req_stages as sunrealtype);
    let temp1 = SUNSQR(w0) - ONE;
    let temp2 = SUNRsqrt(temp1);
    let arg = req_stages as sunrealtype * SUNRlog(w0 + temp2);
    let w1 = SUNRsinh(arg) * temp1
        / (SUNRcosh(arg) * req_stages as sunrealtype * temp2 - w0 * SUNRsinh(arg));
    let mut bjm1 = ONE / SUNSQR(TWO * w0);
    let mut bjm2 = bjm1;
    let mut mus = w1 * bjm1;

    /* Begin stage 1 (store in tmp2) and initialize embedding */
    let mut tcur = tn + h * mus;
    ark_mem.borrow_mut().tcur = tcur;
    lsrkStep_mem_mut(ark_mem).istage = 1;
    N_VLinearSum(ONE, &yn, h * mus, &fn_, &tmp2);
    N_VScale(ONE, &yn, &tmp1);

    /* apply user-supplied stage postprocessing function (if supplied) */
    let PostProcessStageFn = ark_mem.borrow().PostProcessStageFn;
    if let Some(PostProcessStageFn) = PostProcessStageFn {
        let retval = lsrk_call_processfn(ark_mem, PostProcessStageFn, tcur, &tmp2);
        if retval != 0 {
            return ARK_POSTPROCESS_STAGE_FAIL;
        }
    }

    /* Initialize constants for stage loop */
    let mut thjm2 = ZERO;
    let mut thjm1 = mus;
    let mut zjm1 = w0;
    let mut zjm2 = ONE;
    let mut dzjm1 = ONE;
    let mut dzjm2 = ZERO;
    let mut d2zjm1 = ZERO;
    let mut d2zjm2 = ZERO;

    /* Evaluate stages j = 2,...,step_mem->req_stages */
    for j in 2..=req_stages {
        /* Complete the previous stage (evaluate the RHS and store it in ycur) */

        /* call the user-supplied pre-RHS function (if supplied) */
        let PreRhsFn = ark_mem.borrow().PreRhsFn;
        if let Some(PreRhsFn) = PreRhsFn {
            let retval = lsrk_call_processfn(ark_mem, PreRhsFn, tcur, &tmp2);
            if retval != 0 {
                return ARK_PRERHSFN_FAIL;
            }
        }

        let retval = lsrk_call_fe(ark_mem, tcur, &tmp2, &ycur);
        lsrkStep_mem_mut(ark_mem).nfe += 1;

        if retval < 0 {
            return ARK_RHSFUNC_FAIL;
        }
        if retval > 0 {
            return RHSFUNC_RECVR;
        }

        /* Begin stage j (store in ycur) */
        let zj = TWO * w0 * zjm1 - zjm2;
        let dzj = TWO * w0 * dzjm1 - dzjm2 + TWO * zjm1;
        let d2zj = TWO * w0 * d2zjm1 - d2zjm2 + FOUR * dzjm1;
        let bj = d2zj / SUNSQR(dzj);
        let ajm1 = ONE - zjm1 * bjm1;
        let mu = TWO * w0 * bj / bjm1;
        let nu = -bj / bjm2;
        mus = mu * w1 / w0;
        let thj = mu * thjm1 + nu * thjm2 + mus * (ONE - ajm1);
        tcur = tn + h * thj;
        ark_mem.borrow_mut().tcur = tcur;
        lsrkStep_mem_mut(ark_mem).istage = j;
        let cvals = [
            mus * h,
            nu,
            ONE - mu - nu,
            mu,
            -mus * ajm1 * h,
        ];
        let Xvecs = [ycur.clone(), tmp1.clone(), yn.clone(), tmp2.clone(), fn_.clone()];
        let retval = N_VLinearCombination(5, &cvals, &Xvecs, &ycur);
        if retval != 0 {
            return ARK_VECTOROP_ERR;
        }

        /* apply user-supplied stage or step postprocessing function (if supplied) */
        let PostProcessStageFn = ark_mem.borrow().PostProcessStageFn;
        let PostProcessStepFn = ark_mem.borrow().PostProcessStepFn;
        if j < req_stages && PostProcessStageFn.is_some() {
            let f = PostProcessStageFn.expect("PostProcessStageFn");
            let retval = lsrk_call_processfn(ark_mem, f, tcur, &ycur);
            if retval != 0 {
                return ARK_POSTPROCESS_STAGE_FAIL;
            }
        } else if j == req_stages && PostProcessStepFn.is_some() {
            let f = PostProcessStepFn.expect("PostProcessStepFn");
            let retval = lsrk_call_processfn(ark_mem, f, tcur, &ycur);
            if retval != 0 {
                return ARK_POSTPROCESS_STEP_FAIL;
            }
        }

        /* Shift the data for the next stage */
        if j < req_stages {
            /* Swap tempv1 and tempv2 pointers to handle two-previous-stage logic */
            std::mem::swap(&mut tmp1, &mut tmp2);

            N_VScale(ONE, &ycur, &tmp2);

            /* Update coefficients to handle the two-previous stage logic */
            thjm2 = thjm1;
            thjm1 = thj;
            bjm2 = bjm1;
            bjm1 = bj;
            zjm2 = zjm1;
            zjm1 = zj;
            dzjm2 = dzjm1;
            dzjm1 = dzj;
            d2zjm2 = d2zjm1;
            d2zjm1 = d2zj;
        }
    }

    /* final stage processing */
    tcur = tn + h;
    ark_mem.borrow_mut().tcur = tcur;

    /* call the user-supplied pre-RHS function (if supplied) */
    let PreRhsFn = ark_mem.borrow().PreRhsFn;
    if let Some(PreRhsFn) = PreRhsFn {
        let retval = lsrk_call_processfn(ark_mem, PreRhsFn, tcur, &ycur);
        if retval != 0 {
            return ARK_PRERHSFN_FAIL;
        }
    }

    let retval = lsrk_call_fe(ark_mem, tcur, &ycur, &tempv2);
    lsrkStep_mem_mut(ark_mem).nfe += 1;

    if retval < 0 {
        return ARK_RHSFUNC_FAIL;
    }
    if retval > 0 {
        return RHSFUNC_RECVR;
    }

    /* Compute yerr (if step adaptivity enabled) */
    if !ark_mem.borrow().fixedstep {
        /* Estimate the local error and compute its weighted RMS norm */
        let cvals = [p8, -p8, p4 * h, p4 * h];
        let Xvecs = [yn.clone(), ycur.clone(), fn_.clone(), tempv2.clone()];

        let retval = N_VLinearCombination(4, &cvals, &Xvecs, &tempv1);
        if retval != 0 {
            return ARK_VECTOROP_ERR;
        }
        *dsmPtr = N_VWrmsNorm(&tempv1, &ewt);
    }
    lsrkStep_DomEigUpdateLogic(ark_mem, *dsmPtr, &tempv2);

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  lsrkStep_TakeStepRKL:

  This routine serves the primary purpose of the LSRKStepRKL module:
  it performs a single RKL step (with embedding, if possible).

  The output variable dsmPtr should contain estimate of the
  weighted local error if an embedding is present; otherwise it
  should be 0.

  The variables (ark_mem->tcur, ark_mem->ycur) should
  contain the current time and solution at the end of this time step.

  The input/output variable nflagPtr is generally used in ARKODE
  to gauge the convergence of any algebraic solvers. However, since
  the STS step routines do not involve an algebraic solve, this variable
  instead serves to identify possible ARK_RETRY_STEP returns within this
  routine.

  The return value from this routine is:
            0 => step completed successfully
           >0 => step encountered recoverable failure;
                 reduce step and retry (if possible)
                 ARK_RETRY_STEP indicates that the required stage
                 number has reached the stage_max_limit with the
                 current value of h. The step is then returned to
                 adjust the step size.
           <0 => step encountered unrecoverable failure
  ---------------------------------------------------------------*/
pub fn lsrkStep_TakeStepRKL(
    ark_mem: &ARKodeMem,
    dsmPtr: &mut sunrealtype,
    nflagPtr: &mut i32,
) -> i32 {
    let p8: sunrealtype = 0.8;
    let p4: sunrealtype = 0.4;
    let mut stability_norm: sunrealtype = ZERO;

    /* initialize algebraic solver convergence flag to success,
       temporal error estimate to zero */
    *nflagPtr = ARK_SUCCESS;
    *dsmPtr = ZERO;

    /* access ARKodeLSRKStepMem structure */
    let retval = lsrkStep_AccessStepMem(ark_mem, "lsrkStep_TakeStepRKL");
    if retval != ARK_SUCCESS {
        return retval;
    }

    let tempv1 = ark_mem.borrow().tempv1.clone().expect("tempv1");
    let tempv2 = ark_mem.borrow().tempv2.clone().expect("tempv2");
    let mut tmp1 = tempv1.clone();
    let mut tmp2 = tempv2.clone();

    let (h, tn, yn, ycur, fn_, ewt) = {
        let m = ark_mem.borrow();
        (
            m.h,
            m.tn,
            m.yn.clone().expect("yn"),
            m.ycur.clone().expect("ycur"),
            m.fn_.clone().expect("fn"),
            m.ewt.clone().expect("ewt"),
        )
    };

    let stage_max_limit = lsrkStep_mem_mut(ark_mem).stage_max_limit;

    /* Initialize the current stage index */
    {
        let mut step_mem = lsrkStep_mem_mut(ark_mem);
        step_mem.istage = 0;
        let stage_max_limit = step_mem.stage_max_limit;
        step_mem.req_stages = stage_max_limit;
    }

    /* Compute dominant eigenvalue and update stats */
    let dom_eig_update = lsrkStep_mem_mut(ark_mem).dom_eig_update;
    if dom_eig_update {
        let retval = lsrkStep_ComputeNewDomEig(ark_mem);
        if retval != ARK_SUCCESS {
            return retval;
        }
    }

    /* Compute the number of stages based on the current step size and dominant
       eigenvalue using Eq. 19 in Meyer et al. (2014)
       https://doi.org/10.1016/j.jcp.2013.08.021

       Using delta t_expl = 2 / lambda_max, note tau_max * lambda_max in Eq. 19 is
       positive (i.e., tau_max * lambda_max = -zR = h * lambdaR assuming that
       h * lambdaR < 0) and we have incorporated the minus sign on zR below. We
       use the minimum number of stages (ss = 2) for zR > 0. */
    let (lambdaR, lambdaI) = {
        let step_mem = lsrkStep_mem_mut(ark_mem);
        (step_mem.lambdaR, step_mem.lambdaI)
    };
    let zR = h * lambdaR;
    let zI = h * lambdaI;
    let zRabs = SUNRabs(zR);
    let mut ss: i32 = if zR > ZERO {
        2
    } else {
        SUNRceil((SUNRsqrt(9.0 + 8.0 * zRabs) - ONE) / TWO) as i32
    };

    ss = SUNMAX(ss, 2);

    /* Check if number of stages exceeds maximum allowed.
       If so, and if adaptive stepping is enabled, reduce step size
       and return ARK_RETRY_STEP. If fixed step size, return
       ARK_MAX_STAGE_LIMIT_FAIL error. */
    if ss > stage_max_limit {
        if !ark_mem.borrow().fixedstep {
            let safety = ark_mem.borrow().hadapt_mem.as_ref().expect("hadapt_mem").safety;
            let hmax = safety
                * ((stage_max_limit * stage_max_limit + stage_max_limit) as sunrealtype - TWO)
                / (TWO * SUNRabs(lambdaR));
            ark_mem.borrow_mut().eta = hmax / SUNRabs(h);
            *nflagPtr = ARK_RETRY_STEP;
            ark_mem.borrow_mut().hadapt_mem.as_mut().expect("hadapt_mem").nst_exp += 1;
            return ARK_RETRY_STEP;
        } else {
            arkProcessError(
                Some(ark_mem),
                ARK_MAX_STAGE_LIMIT_FAIL,
                line!() as i32,
                "lsrkStep_TakeStepRKL",
                file!(),
                "Unable to achieve stable results: Either reduce the step size or increase the \
                 stage_max_limit",
            );
            return ARK_MAX_STAGE_LIMIT_FAIL;
        }
    }

    /* Copy ss in case it is needed for falling back to step size adaptivity below */
    let mut req_stages: i32 = ss;

    if zR < -SUN_UNIT_ROUNDOFF {
        /* To check stability, we evaluate the analytic stability function or an
          inscribed ellipse approximation. If the stability norm is greater than
          one, first check whether the method is stable at stage_max_limit. If so,
          increase the number of stages until stability is obtained. Otherwise,
          keep the existing fixed-step error and adaptive-step eta update logic. */
        let retval =
            lsrkStep_RKL_CheckStabilityNorm(ark_mem, req_stages, h, &mut stability_norm);
        if retval != ARK_SUCCESS {
            return retval;
        }

        if stability_norm > ONE - SUN_UNIT_ROUNDOFF {
            let initial_stability_norm = stability_norm;
            let mut max_stage_is_stable = SUNFALSE;

            if req_stages < stage_max_limit {
                let retval = lsrkStep_RKL_CheckStabilityNorm(
                    ark_mem,
                    stage_max_limit,
                    h,
                    &mut stability_norm,
                );
                if retval != ARK_SUCCESS {
                    return retval;
                }

                max_stage_is_stable = stability_norm <= ONE - SUN_UNIT_ROUNDOFF;
                stability_norm = initial_stability_norm;
            }

            if max_stage_is_stable {
                while (stability_norm > ONE - SUN_UNIT_ROUNDOFF) && (req_stages < stage_max_limit)
                {
                    req_stages += 1;
                    let retval =
                        lsrkStep_RKL_CheckStabilityNorm(ark_mem, req_stages, h, &mut stability_norm);
                    if retval != ARK_SUCCESS {
                        return retval;
                    }
                }
            }

            if stability_norm > ONE - SUN_UNIT_ROUNDOFF {
                if !ark_mem.borrow().fixedstep {
                    let ssr = ss as sunrealtype;
                    let aspect_ratio: [sunrealtype; 7] = [
                        0.3 * ssr,   /* s = 2 */
                        0.75 * ssr,  /* s = 3 */
                        0.665 * ssr, /* s = 4 */
                        0.665 * ssr, /* s = 5 */
                        0.635 * ssr, /* s = 6 to 20 */
                        0.6 * ssr,   /* s >= 20 and odd */
                        0.53 * ssr,  /* s >= 20 and even */
                    ];
                    let a = ((TWO * ssr + ONE) * (TWO * ssr + ONE) - 9.0) / 16.0;
                    let b: sunrealtype;

                    if ss < 7 {
                        b = a / aspect_ratio[(req_stages - 2) as usize];
                    } else if ss <= 20 {
                        b = a / aspect_ratio[4];
                    } else {
                        b = a / aspect_ratio[(6 - req_stages % 2) as usize];
                    }

                    let safety =
                        ark_mem.borrow().hadapt_mem.as_ref().expect("hadapt_mem").safety;
                    ark_mem.borrow_mut().eta = safety * (-TWO * a * b * b * zR)
                        / (SUNSQR(b * zR) + SUNSQR(a * zI));
                    *nflagPtr = ARK_RETRY_STEP;
                    ark_mem.borrow_mut().hadapt_mem.as_mut().expect("hadapt_mem").nst_exp += 1;
                    return ARK_RETRY_STEP;
                } else {
                    arkProcessError(
                        Some(ark_mem),
                        ARK_MAX_STAGE_LIMIT_FAIL,
                        line!() as i32,
                        "lsrkStep_TakeStepRKL",
                        file!(),
                        "Unable to achieve stable results: Either reduce the step size or \
                         increase the stage_max_limit",
                    );
                    return ARK_MAX_STAGE_LIMIT_FAIL;
                }
            }
        }
    }

    {
        let mut step_mem = lsrkStep_mem_mut(ark_mem);
        step_mem.req_stages = req_stages;
        let stage_max = step_mem.stage_max;
        step_mem.stage_max = SUNMAX(step_mem.req_stages, stage_max);
    }

    /* Compute RHS function for the start of the step, if necessary. */
    let need_rhs = {
        let m = ark_mem.borrow();
        let step_nst = m
            .step_mem
            .as_ref()
            .expect("step_mem set")
            .downcast_ref::<ARKodeLSRKStepMemRec>()
            .expect("LSRKStep step memory")
            .step_nst;
        (!m.fn_is_current && m.initsetup) || (step_nst != m.nst)
    };
    if need_rhs {
        /* call the user-supplied pre-RHS function (if supplied) */
        let PreRhsFn = ark_mem.borrow().PreRhsFn;
        if let Some(PreRhsFn) = PreRhsFn {
            let retval = lsrk_call_processfn(ark_mem, PreRhsFn, tn, &yn);
            if retval != 0 {
                return ARK_PRERHSFN_FAIL;
            }
        }
        let retval = lsrk_call_fe(ark_mem, tn, &yn, &fn_);
        lsrkStep_mem_mut(ark_mem).nfe += 1;
        if retval != ARK_SUCCESS {
            return ARK_RHSFUNC_FAIL;
        }
        ark_mem.borrow_mut().fn_is_current = SUNTRUE;
    }

    /* Track the number of successful steps to determine if the previous step failed. */
    {
        let nst = ark_mem.borrow().nst;
        lsrkStep_mem_mut(ark_mem).step_nst = nst + 1;
    }

    /* Initialize constants */
    let rs = req_stages as sunrealtype;
    let w1 = FOUR / ((rs + TWO) * (rs - ONE));
    let mut bjm2 = ONE / THREE;
    let mut bjm1 = bjm2;
    let mut mus = w1 * bjm1;

    /* Begin stage 1 (store in tmp2) and initialize embedding */
    let mut tcur = tn + h * mus;
    ark_mem.borrow_mut().tcur = tcur;
    lsrkStep_mem_mut(ark_mem).istage = 1;
    N_VLinearSum(ONE, &yn, h * mus, &fn_, &tmp2);
    N_VScale(ONE, &yn, &tmp1);

    /* apply user-supplied stage postprocessing function (if supplied) */
    let PostProcessStageFn = ark_mem.borrow().PostProcessStageFn;
    if let Some(PostProcessStageFn) = PostProcessStageFn {
        let retval = lsrk_call_processfn(ark_mem, PostProcessStageFn, tcur, &tmp2);
        if retval != 0 {
            return ARK_POSTPROCESS_STAGE_FAIL;
        }
    }

    /* Evaluate stages j = 2,...,step_mem->req_stages */
    for j in 2..=req_stages {
        /* Complete the previous stage (evaluate the RHS and store it in ycur) */

        /* call the user-supplied pre-RHS function (if supplied) */
        let PreRhsFn = ark_mem.borrow().PreRhsFn;
        if let Some(PreRhsFn) = PreRhsFn {
            let retval = lsrk_call_processfn(ark_mem, PreRhsFn, tcur, &tmp2);
            if retval != 0 {
                return ARK_PRERHSFN_FAIL;
            }
        }

        let retval = lsrk_call_fe(ark_mem, tcur, &tmp2, &ycur);
        lsrkStep_mem_mut(ark_mem).nfe += 1;

        if retval < 0 {
            return ARK_RHSFUNC_FAIL;
        }
        if retval > 0 {
            return RHSFUNC_RECVR;
        }

        /* Begin stage j (store in ycur) */
        let jr = j as sunrealtype;
        let temj = (jr + TWO) * (jr - ONE);
        let bj = temj / (TWO * jr * (jr + ONE));
        let ajm1 = ONE - bjm1;
        let mu = (TWO * jr - ONE) / jr * (bj / bjm1);
        let nu = -(jr - ONE) / jr * (bj / bjm2);
        mus = w1 * mu;
        let cj = temj * w1 / FOUR;
        tcur = tn + h * cj;
        ark_mem.borrow_mut().tcur = tcur;
        lsrkStep_mem_mut(ark_mem).istage = j;
        let cvals = [mus * h, nu, ONE - mu - nu, mu, -mus * ajm1 * h];
        let Xvecs = [ycur.clone(), tmp1.clone(), yn.clone(), tmp2.clone(), fn_.clone()];
        let retval = N_VLinearCombination(5, &cvals, &Xvecs, &ycur);
        if retval != 0 {
            return ARK_VECTOROP_ERR;
        }

        /* apply user-supplied stage or step postprocessing function (if supplied) */
        let PostProcessStageFn = ark_mem.borrow().PostProcessStageFn;
        let PostProcessStepFn = ark_mem.borrow().PostProcessStepFn;
        if j < req_stages && PostProcessStageFn.is_some() {
            let f = PostProcessStageFn.expect("PostProcessStageFn");
            let retval = lsrk_call_processfn(ark_mem, f, tcur, &ycur);
            if retval != 0 {
                return ARK_POSTPROCESS_STAGE_FAIL;
            }
        } else if j == req_stages && PostProcessStepFn.is_some() {
            let f = PostProcessStepFn.expect("PostProcessStepFn");
            let retval = lsrk_call_processfn(ark_mem, f, tcur + h * cj, &ycur);
            if retval != 0 {
                return ARK_POSTPROCESS_STEP_FAIL;
            }
        }

        /* Shift the data for the next stage */
        if j < req_stages {
            /* To avoid two data copies we swap ARKODE's tempv1 and tempv2 pointers*/
            std::mem::swap(&mut tmp1, &mut tmp2);

            N_VScale(ONE, &ycur, &tmp2);

            bjm2 = bjm1;
            bjm1 = bj;
        }
    }

    /* final stage processing */
    tcur = tn + h;
    ark_mem.borrow_mut().tcur = tcur;

    /* call the user-supplied pre-RHS function (if supplied) */
    let PreRhsFn = ark_mem.borrow().PreRhsFn;
    if let Some(PreRhsFn) = PreRhsFn {
        let retval = lsrk_call_processfn(ark_mem, PreRhsFn, tcur, &ycur);
        if retval != 0 {
            return ARK_PRERHSFN_FAIL;
        }
    }
    let retval = lsrk_call_fe(ark_mem, tcur, &ycur, &tempv2);
    lsrkStep_mem_mut(ark_mem).nfe += 1;

    if retval < 0 {
        return ARK_RHSFUNC_FAIL;
    }
    if retval > 0 {
        return RHSFUNC_RECVR;
    }

    /* Compute yerr (if step adaptivity enabled) */
    if !ark_mem.borrow().fixedstep {
        /* Estimate the local error and compute its weighted RMS norm */
        let cvals = [p8, -p8, p4 * h, p4 * h];
        let Xvecs = [yn.clone(), ycur.clone(), fn_.clone(), tempv2.clone()];
        let retval = N_VLinearCombination(4, &cvals, &Xvecs, &tempv1);
        if retval != 0 {
            return ARK_VECTOROP_ERR;
        }
        *dsmPtr = N_VWrmsNorm(&tempv1, &ewt);
        lsrkStep_DomEigUpdateLogic(ark_mem, *dsmPtr, &tempv2);
    } else {
        lsrkStep_DomEigUpdateLogic(ark_mem, *dsmPtr, &tempv2);
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  lsrkStep_TakeStepSSPs2:

  This routine serves the primary purpose of the LSRKStepSSPs2 module:
  it performs a single SSPs2 step (with embedding).

  The output variable dsmPtr should contain estimate of the
  weighted local error if an embedding is present; otherwise it
  should be 0.

  The variables (ark_mem->tcur, ark_mem->ycur) should
  contain the current time and solution at the end of this time step.

  The input/output variable nflagPtr is used to gauge convergence
  of any algebraic solvers within the step.  As this routine
  involves no algebraic solve, it is set to 0 (success).

  The return value from this routine is:
            0 => step completed successfully
           >0 => step encountered recoverable failure;
                 reduce step and retry (if possible)
           <0 => step encountered unrecoverable failure
  ---------------------------------------------------------------*/
pub fn lsrkStep_TakeStepSSPs2(
    ark_mem: &ARKodeMem,
    dsmPtr: &mut sunrealtype,
    nflagPtr: &mut i32,
) -> i32 {
    /* initialize algebraic solver convergence flag to success,
       temporal error estimate to zero */
    *nflagPtr = ARK_SUCCESS;
    *dsmPtr = ZERO;

    /* access ARKodeLSRKStepMem structure */
    let retval = lsrkStep_AccessStepMem(ark_mem, "lsrkStep_TakeStepSSPs2");
    if retval != ARK_SUCCESS {
        return retval;
    }

    let (h, tn, yn, ycur, fn_, ewt, tempv1, tempv2) = {
        let m = ark_mem.borrow();
        (
            m.h,
            m.tn,
            m.yn.clone().expect("yn"),
            m.ycur.clone().expect("ycur"),
            m.fn_.clone().expect("fn"),
            m.ewt.clone().expect("ewt"),
            m.tempv1.clone().expect("tempv1"),
            m.tempv2.clone().expect("tempv2"),
        )
    };

    /* Initialize the current stage index */
    lsrkStep_mem_mut(ark_mem).istage = 0;

    /* Initialize method coefficients */
    let req_stages = lsrkStep_mem_mut(ark_mem).req_stages;
    let rs = req_stages as sunrealtype;
    let sm1inv = ONE / (rs - ONE);
    let hsm1inv = h * sm1inv;
    let rsinv = ONE / rs;
    let hrsinv = h * rsinv;
    let hbt1: sunrealtype;
    let hbt2: sunrealtype;
    let hbt3: sunrealtype;

    /* Embedding coefficients differ when req_stages == 2 */
    if req_stages == 2 {
        /* from https://doi.org/10.1016/j.cam.2022.114325 pg 5 */
        hbt1 = h * 0.694021459207626;
        hbt2 = ZERO;
        hbt3 = h - hbt1;
    } else {
        hbt1 = hrsinv * (ONE + rsinv);
        hbt2 = hrsinv;
        hbt3 = hrsinv * (ONE - rsinv);
    }

    /* Begin stage 0 */

    /* The method is not FSAL. Therefore, fn is computed at the beginning
       of the step unless the previous step failed or ARKODE updated fn. */
    if !ark_mem.borrow().fn_is_current {
        /* call the user-supplied pre-RHS function (if supplied) */
        let PreRhsFn = ark_mem.borrow().PreRhsFn;
        if let Some(PreRhsFn) = PreRhsFn {
            let retval = lsrk_call_processfn(ark_mem, PreRhsFn, tn, &yn);
            if retval != 0 {
                return ARK_PRERHSFN_FAIL;
            }
        }

        let retval = lsrk_call_fe(ark_mem, tn, &yn, &fn_);
        lsrkStep_mem_mut(ark_mem).nfe += 1;
        if retval != ARK_SUCCESS {
            return ARK_RHSFUNC_FAIL;
        }
        ark_mem.borrow_mut().fn_is_current = SUNTRUE;
    }

    /* Begin stage 1 and accumulate embedding into tempv1 */
    let mut tcur = tn + hsm1inv;
    ark_mem.borrow_mut().tcur = tcur;
    lsrkStep_mem_mut(ark_mem).istage = 1;
    N_VLinearSum(ONE, &yn, hsm1inv, &fn_, &ycur);
    if !ark_mem.borrow().fixedstep {
        N_VLinearSum(ONE, &yn, hbt1, &fn_, &tempv1);
    }

    /* apply user-supplied stage postprocessing function (if supplied) */
    let PostProcessStageFn = ark_mem.borrow().PostProcessStageFn;
    if let Some(PostProcessStageFn) = PostProcessStageFn {
        let retval = lsrk_call_processfn(ark_mem, PostProcessStageFn, tcur, &ycur);
        if retval != 0 {
            return ARK_POSTPROCESS_STAGE_FAIL;
        }
    }

    /* Evaluate stages j = 2,...,step_mem->req_stages - 1 */
    for j in 2..req_stages {
        /* Complete the previous stage (evaluate the RHS and store it in tempv2) */

        /* apply user-supplied stage preprocessing function (if supplied) */
        let PreRhsFn = ark_mem.borrow().PreRhsFn;
        if let Some(PreRhsFn) = PreRhsFn {
            let retval = lsrk_call_processfn(ark_mem, PreRhsFn, tcur, &ycur);
            if retval != 0 {
                return ARK_PRERHSFN_FAIL;
            }
        }

        let retval = lsrk_call_fe(ark_mem, tcur, &ycur, &tempv2);
        lsrkStep_mem_mut(ark_mem).nfe += 1;

        if retval < 0 {
            return ARK_RHSFUNC_FAIL;
        }
        if retval > 0 {
            return RHSFUNC_RECVR;
        }

        /* Begin stage j (update the state and embedding) */
        lsrkStep_mem_mut(ark_mem).istage = j;
        tcur = tn + j as sunrealtype * hsm1inv;
        ark_mem.borrow_mut().tcur = tcur;
        N_VLinearSum(ONE, &ycur, hsm1inv, &tempv2, &ycur);
        if !ark_mem.borrow().fixedstep {
            N_VLinearSum(ONE, &tempv1, hbt2, &tempv2, &tempv1);
        }

        /* apply user-supplied stage postprocessing function (if supplied) */
        let PostProcessStageFn = ark_mem.borrow().PostProcessStageFn;
        if let Some(PostProcessStageFn) = PostProcessStageFn {
            let retval = lsrk_call_processfn(ark_mem, PostProcessStageFn, tcur, &ycur);
            if retval != 0 {
                return ARK_POSTPROCESS_STAGE_FAIL;
            }
        }
    }

    /* Complete the next-to-last stage by evaluating the RHS and storing it in tempv2 */
    let PreRhsFn = ark_mem.borrow().PreRhsFn;
    if let Some(PreRhsFn) = PreRhsFn {
        let retval = lsrk_call_processfn(ark_mem, PreRhsFn, tcur, &ycur);
        if retval != 0 {
            return ARK_PRERHSFN_FAIL;
        }
    }
    let retval = lsrk_call_fe(ark_mem, tcur, &ycur, &tempv2);
    lsrkStep_mem_mut(ark_mem).nfe += 1;

    if retval < 0 {
        return ARK_RHSFUNC_FAIL;
    }
    if retval > 0 {
        return RHSFUNC_RECVR;
    }

    /* Compute the step solution */
    tcur = tn + h;
    ark_mem.borrow_mut().tcur = tcur;
    lsrkStep_mem_mut(ark_mem).istage = req_stages;
    let cvals = [ONE / (sm1inv * rs), rsinv, hrsinv];
    let Xvecs = [ycur.clone(), yn.clone(), tempv2.clone()];
    let retval = N_VLinearCombination(3, &cvals, &Xvecs, &ycur);
    if retval != 0 {
        return ARK_VECTOROP_ERR;
    }

    /* apply user-supplied step postprocessing function (if supplied) */
    let PostProcessStepFn = ark_mem.borrow().PostProcessStepFn;
    if let Some(PostProcessStepFn) = PostProcessStepFn {
        let retval = lsrk_call_processfn(ark_mem, PostProcessStepFn, tcur, &ycur);
        if retval != 0 {
            return ARK_POSTPROCESS_STEP_FAIL;
        }
    }

    /* Compute yerr (if step adaptivity enabled) */
    if !ark_mem.borrow().fixedstep {
        N_VLinearSum(ONE, &tempv1, hbt3, &tempv2, &tempv1);
        N_VLinearSum(ONE, &ycur, -ONE, &tempv1, &tempv1);
        *dsmPtr = N_VWrmsNorm(&tempv1, &ewt);
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  lsrkStep_TakeStepSSPs3:

  This routine serves the primary purpose of the LSRKStepSSPs3 module:
  it performs a single SSPs3 step (with embedding).

  The SSP3 method differs significantly when s = 4. Therefore, the case
  where num_of_stages = 4 is considered separately to avoid unnecessary
  boolean checks and improve computational efficiency.

  The output variable dsmPtr should contain estimate of the
  weighted local error if an embedding is present; otherwise it
  should be 0.

  The variables (ark_mem->tcur, ark_mem->ycur) should
  contain the current time and solution at the end of this time step.

  The input/output variable nflagPtr is used to gauge convergence
  of any algebraic solvers within the step.  As this routine
  involves no algebraic solve, it is set to 0 (success).

  The return value from this routine is:
            0 => step completed successfully
           >0 => step encountered recoverable failure;
                 reduce step and retry (if possible)
           <0 => step encountered unrecoverable failure
  ---------------------------------------------------------------*/
pub fn lsrkStep_TakeStepSSPs3(
    ark_mem: &ARKodeMem,
    dsmPtr: &mut sunrealtype,
    nflagPtr: &mut i32,
) -> i32 {
    /* initialize algebraic solver convergence flag to success,
       temporal error estimate to zero */
    *nflagPtr = ARK_SUCCESS;
    *dsmPtr = ZERO;

    /* access ARKodeLSRKStepMem structure */
    let retval = lsrkStep_AccessStepMem(ark_mem, "lsrkStep_TakeStepSSPs3");
    if retval != ARK_SUCCESS {
        return retval;
    }

    let (h, tn, yn, ycur, fn_, ewt, tempv1, tempv2, tempv3) = {
        let m = ark_mem.borrow();
        (
            m.h,
            m.tn,
            m.yn.clone().expect("yn"),
            m.ycur.clone().expect("ycur"),
            m.fn_.clone().expect("fn"),
            m.ewt.clone().expect("ewt"),
            m.tempv1.clone().expect("tempv1"),
            m.tempv2.clone().expect("tempv2"),
            m.tempv3.clone().expect("tempv3"),
        )
    };

    /* Initialize the current stage index */
    lsrkStep_mem_mut(ark_mem).istage = 0;

    /* Initialize method coefficients */
    let req_stages = lsrkStep_mem_mut(ark_mem).req_stages;
    let rs = req_stages as sunrealtype;
    let rn = SUNRsqrt(rs);
    let hrat = h / (rs - rn);
    let hrsinv = h / rs;
    /* C `in`; `in` is a Rust keyword */
    let in_ = SUNRround(rn) as i32;

    /* Begin stage 0 */

    /* The method is not FSAL. Therefore, fn is computed at the beginning
       of the step unless ARKODE updated fn. */
    if !ark_mem.borrow().fn_is_current {
        /* call the user-supplied pre-RHS function (if supplied) */
        let PreRhsFn = ark_mem.borrow().PreRhsFn;
        if let Some(PreRhsFn) = PreRhsFn {
            let retval = lsrk_call_processfn(ark_mem, PreRhsFn, tn, &yn);
            if retval != 0 {
                return ARK_PRERHSFN_FAIL;
            }
        }

        let retval = lsrk_call_fe(ark_mem, tn, &yn, &fn_);
        lsrkStep_mem_mut(ark_mem).nfe += 1;
        if retval != ARK_SUCCESS {
            return ARK_RHSFUNC_FAIL;
        }
        ark_mem.borrow_mut().fn_is_current = SUNTRUE;
    }

    /* Begin stage 1 and accumulate embedding into tempv1 */
    let mut tcur = tn + hrat;
    ark_mem.borrow_mut().tcur = tcur;
    lsrkStep_mem_mut(ark_mem).istage = 1;
    N_VLinearSum(ONE, &yn, hrat, &fn_, &ycur);
    if !ark_mem.borrow().fixedstep {
        N_VLinearSum(ONE, &yn, hrsinv, &fn_, &tempv1);
    }

    /* apply user-supplied stage postprocessing function (if supplied) */
    let PostProcessStageFn = ark_mem.borrow().PostProcessStageFn;
    if let Some(PostProcessStageFn) = PostProcessStageFn {
        let retval = lsrk_call_processfn(ark_mem, PostProcessStageFn, tcur, &ycur);
        if retval != 0 {
            return ARK_POSTPROCESS_STAGE_FAIL;
        }
    }

    /* Evaluate first stage group */
    for j in 2..=((in_ - 1) * (in_ - 2) / 2) {
        /* Complete the previous stage (evaluate the RHS and store it in tempv3) */

        /* apply user-supplied stage preprocessing function (if supplied) */
        let PreRhsFn = ark_mem.borrow().PreRhsFn;
        if let Some(PreRhsFn) = PreRhsFn {
            let retval = lsrk_call_processfn(ark_mem, PreRhsFn, tcur, &ycur);
            if retval != 0 {
                return ARK_PRERHSFN_FAIL;
            }
        }

        let retval = lsrk_call_fe(ark_mem, tcur, &ycur, &tempv3);
        lsrkStep_mem_mut(ark_mem).nfe += 1;

        if retval < 0 {
            return ARK_RHSFUNC_FAIL;
        }
        if retval > 0 {
            return RHSFUNC_RECVR;
        }

        /* Begin stage j (update the state and embedding) */
        tcur = tn + j as sunrealtype * hrat;
        ark_mem.borrow_mut().tcur = tcur;
        lsrkStep_mem_mut(ark_mem).istage = j;
        N_VLinearSum(ONE, &ycur, hrat, &tempv3, &ycur);
        if !ark_mem.borrow().fixedstep {
            N_VLinearSum(ONE, &tempv1, hrsinv, &tempv3, &tempv1);
        }

        /* apply user-supplied stage postprocessing function (if supplied) */
        let PostProcessStageFn = ark_mem.borrow().PostProcessStageFn;
        if let Some(PostProcessStageFn) = PostProcessStageFn {
            let retval = lsrk_call_processfn(ark_mem, PostProcessStageFn, tcur, &ycur);
            if retval != 0 {
                return ARK_POSTPROCESS_STAGE_FAIL;
            }
        }
    }

    /* Copy ycur into tempv2 before looping over second stage group */
    N_VScale(ONE, &ycur, &tempv2);

    /* Evaluate second stage group */
    for j in ((in_ - 1) * (in_ - 2) / 2 + 1)..=(in_ * (in_ + 1) / 2 - 1) {
        /* Complete the previous stage (evaluate the RHS and store it in tempv3) */

        /* apply user-supplied stage preprocessing function (if supplied) */
        let PreRhsFn = ark_mem.borrow().PreRhsFn;
        if let Some(PreRhsFn) = PreRhsFn {
            let retval = lsrk_call_processfn(ark_mem, PreRhsFn, tcur, &ycur);
            if retval != 0 {
                return ARK_PRERHSFN_FAIL;
            }
        }

        let retval = lsrk_call_fe(ark_mem, tcur, &ycur, &tempv3);
        lsrkStep_mem_mut(ark_mem).nfe += 1;

        if retval < 0 {
            return ARK_RHSFUNC_FAIL;
        }
        if retval > 0 {
            return RHSFUNC_RECVR;
        }

        /* Begin stage j (update the state and embedding) */
        tcur = tn + j as sunrealtype * hrat;
        ark_mem.borrow_mut().tcur = tcur;
        lsrkStep_mem_mut(ark_mem).istage = j;
        N_VLinearSum(ONE, &ycur, hrat, &tempv3, &ycur);
        if !ark_mem.borrow().fixedstep {
            N_VLinearSum(ONE, &tempv1, hrsinv, &tempv3, &tempv1);
        }

        /* apply user-supplied stage postprocessing function (if supplied) */
        let PostProcessStageFn = ark_mem.borrow().PostProcessStageFn;
        if let Some(PostProcessStageFn) = PostProcessStageFn {
            let retval = lsrk_call_processfn(ark_mem, PostProcessStageFn, tcur, &ycur);
            if retval != 0 {
                return ARK_POSTPROCESS_STAGE_FAIL;
            }
        }
    }

    /* apply user-supplied stage preprocessing function (if supplied) */
    let PreRhsFn = ark_mem.borrow().PreRhsFn;
    if let Some(PreRhsFn) = PreRhsFn {
        let retval = lsrk_call_processfn(ark_mem, PreRhsFn, tcur, &ycur);
        if retval != 0 {
            return ARK_PRERHSFN_FAIL;
        }
    }

    let retval = lsrk_call_fe(ark_mem, tcur, &ycur, &tempv3);
    lsrkStep_mem_mut(ark_mem).nfe += 1;

    if retval < 0 {
        return ARK_RHSFUNC_FAIL;
    }
    if retval > 0 {
        return RHSFUNC_RECVR;
    }

    /* Begin the next stage before final stage group */
    tcur = tn + (in_ * (in_ - 1) / 2) as sunrealtype * hrat;
    ark_mem.borrow_mut().tcur = tcur;
    lsrkStep_mem_mut(ark_mem).istage = in_ * (in_ + 1) / 2;
    let cvals = [
        (rn - ONE) / (TWO * rn - ONE),
        rn / (TWO * rn - ONE),
        (rn - ONE) * hrat / (TWO * rn - ONE),
    ];
    let Xvecs = [ycur.clone(), tempv2.clone(), tempv3.clone()];
    let retval = N_VLinearCombination(3, &cvals, &Xvecs, &ycur);
    if retval != 0 {
        return ARK_VECTOROP_ERR;
    }
    if !ark_mem.borrow().fixedstep {
        N_VLinearSum(ONE, &tempv1, hrsinv, &tempv3, &tempv1);
    }

    /* apply user-supplied stage postprocessing function (if supplied) */
    let PostProcessStageFn = ark_mem.borrow().PostProcessStageFn;
    if let Some(PostProcessStageFn) = PostProcessStageFn {
        let retval = lsrk_call_processfn(ark_mem, PostProcessStageFn, tcur, &ycur);
        if retval != 0 {
            return ARK_POSTPROCESS_STAGE_FAIL;
        }
    }

    /* Evaluate final stage group */
    for j in (in_ * (in_ + 1) / 2 + 1)..=req_stages {
        /* Complete the previous stage (evaluate the RHS and store it in tempv3) */

        /* apply user-supplied stage preprocessing function (if supplied) */
        let PreRhsFn = ark_mem.borrow().PreRhsFn;
        if let Some(PreRhsFn) = PreRhsFn {
            let retval = lsrk_call_processfn(ark_mem, PreRhsFn, tcur, &ycur);
            if retval != 0 {
                return ARK_PRERHSFN_FAIL;
            }
        }

        let retval = lsrk_call_fe(ark_mem, tcur, &ycur, &tempv3);
        lsrkStep_mem_mut(ark_mem).nfe += 1;

        if retval < 0 {
            return ARK_RHSFUNC_FAIL;
        }
        if retval > 0 {
            return RHSFUNC_RECVR;
        }

        /* Begin stage j (update the state and embedding) */
        tcur = tn + (j - in_) as sunrealtype * hrat;
        ark_mem.borrow_mut().tcur = tcur;
        lsrkStep_mem_mut(ark_mem).istage = j;
        N_VLinearSum(ONE, &ycur, hrat, &tempv3, &ycur);
        if !ark_mem.borrow().fixedstep {
            N_VLinearSum(ONE, &tempv1, hrsinv, &tempv3, &tempv1);
        }

        /* apply user-supplied stage or step postprocessing function (if supplied) */
        let PostProcessStageFn = ark_mem.borrow().PostProcessStageFn;
        let PostProcessStepFn = ark_mem.borrow().PostProcessStepFn;
        if j < req_stages && PostProcessStageFn.is_some() {
            let f = PostProcessStageFn.expect("PostProcessStageFn");
            let retval = lsrk_call_processfn(ark_mem, f, tcur, &ycur);
            if retval != 0 {
                return ARK_POSTPROCESS_STAGE_FAIL;
            }
        } else if j == req_stages && PostProcessStepFn.is_some() {
            let f = PostProcessStepFn.expect("PostProcessStepFn");
            let retval = lsrk_call_processfn(ark_mem, f, tn + h, &ycur);
            if retval != 0 {
                return ARK_POSTPROCESS_STEP_FAIL;
            }
        }
    }

    /* Compute yerr (if step adaptivity enabled) */
    if !ark_mem.borrow().fixedstep {
        N_VLinearSum(ONE, &ycur, -ONE, &tempv1, &tempv1);
        *dsmPtr = N_VWrmsNorm(&tempv1, &ewt);
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  lsrkStep_TakeStepSSP43:

  This routine serves the primary purpose of the LSRKStepSSP43 module:
  it performs a single SSP43 step (with embedding).

  The SSP3 method differs significantly when s = 4. Therefore, the case
  where num_of_stages = 4 is considered separately to avoid unnecessary
  boolean checks and improve computational efficiency.

  The output variable dsmPtr should contain estimate of the
  weighted local error if an embedding is present; otherwise it
  should be 0.

  The variables (ark_mem->tcur, ark_mem->ycur) should
  contain the current time and solution at the end of this time step.

  The input/output variable nflagPtr is used to gauge convergence
  of any algebraic solvers within the step.  As this routine
  involves no algebraic solve, it is set to 0 (success).

  The return value from this routine is:
            0 => step completed successfully
           >0 => step encountered recoverable failure;
                 reduce step and retry (if possible)
           <0 => step encountered unrecoverable failure
  ---------------------------------------------------------------*/
pub fn lsrkStep_TakeStepSSP43(
    ark_mem: &ARKodeMem,
    dsmPtr: &mut sunrealtype,
    nflagPtr: &mut i32,
) -> i32 {
    /* initialize algebraic solver convergence flag to success,
       temporal error estimate to zero */
    *nflagPtr = ARK_SUCCESS;
    *dsmPtr = ZERO;

    /* access ARKodeLSRKStepMem structure */
    let retval = lsrkStep_AccessStepMem(ark_mem, "lsrkStep_TakeStepSSP43");
    if retval != ARK_SUCCESS {
        return retval;
    }

    let (h, tn, yn, ycur, fn_, ewt, tempv1, tempv3) = {
        let m = ark_mem.borrow();
        (
            m.h,
            m.tn,
            m.yn.clone().expect("yn"),
            m.ycur.clone().expect("ycur"),
            m.fn_.clone().expect("fn"),
            m.ewt.clone().expect("ewt"),
            m.tempv1.clone().expect("tempv1"),
            m.tempv3.clone().expect("tempv3"),
        )
    };

    /* Initialize the current stage index */
    lsrkStep_mem_mut(ark_mem).istage = 0;

    /* Initialize method coefficients */
    let rs: sunrealtype = 4.0;
    let hp5 = h * 0.5;
    let hrsinv = h / rs;

    /* Begin stage 0 */

    /* The method is not FSAL. Therefore, fn is computed at the beginning
       of the step unless ARKODE updated fn. */
    if !ark_mem.borrow().fn_is_current {
        /* call the user-supplied pre-RHS function (if supplied) */
        let PreRhsFn = ark_mem.borrow().PreRhsFn;
        if let Some(PreRhsFn) = PreRhsFn {
            let retval = lsrk_call_processfn(ark_mem, PreRhsFn, tn, &yn);
            if retval != 0 {
                return ARK_PRERHSFN_FAIL;
            }
        }

        let retval = lsrk_call_fe(ark_mem, tn, &yn, &fn_);
        lsrkStep_mem_mut(ark_mem).nfe += 1;
        if retval != ARK_SUCCESS {
            return ARK_RHSFUNC_FAIL;
        }
        ark_mem.borrow_mut().fn_is_current = SUNTRUE;
    }

    /* Begin stage 1 and accumulate embedding into tempv1 */
    let mut tcur = tn + hp5;
    ark_mem.borrow_mut().tcur = tcur;
    lsrkStep_mem_mut(ark_mem).istage = 1;
    N_VLinearSum(ONE, &yn, hp5, &fn_, &ycur);
    if !ark_mem.borrow().fixedstep {
        N_VLinearSum(ONE, &yn, hrsinv, &fn_, &tempv1);
    }

    /* apply user-supplied stage postprocessing function (if supplied) */
    let PostProcessStageFn = ark_mem.borrow().PostProcessStageFn;
    if let Some(PostProcessStageFn) = PostProcessStageFn {
        let retval = lsrk_call_processfn(ark_mem, PostProcessStageFn, tcur, &ycur);
        if retval != 0 {
            return ARK_POSTPROCESS_STAGE_FAIL;
        }
    }

    /* call the user-supplied pre-RHS function (if supplied) */
    let PreRhsFn = ark_mem.borrow().PreRhsFn;
    if let Some(PreRhsFn) = PreRhsFn {
        let retval = lsrk_call_processfn(ark_mem, PreRhsFn, tcur, &ycur);
        if retval != 0 {
            return ARK_PRERHSFN_FAIL;
        }
    }

    /* Evaluate stage RHS */
    let retval = lsrk_call_fe(ark_mem, tcur, &ycur, &tempv3);
    lsrkStep_mem_mut(ark_mem).nfe += 1;

    if retval < 0 {
        return ARK_RHSFUNC_FAIL;
    }
    if retval > 0 {
        return RHSFUNC_RECVR;
    }

    /* Begin stage 2 */
    tcur = tn + h;
    ark_mem.borrow_mut().tcur = tcur;
    lsrkStep_mem_mut(ark_mem).istage = 2;
    N_VLinearSum(ONE, &ycur, hp5, &tempv3, &ycur);
    if !ark_mem.borrow().fixedstep {
        N_VLinearSum(ONE, &tempv1, hrsinv, &tempv3, &tempv1);
    }

    /* apply user-supplied stage postprocessing function (if supplied) */
    let PostProcessStageFn = ark_mem.borrow().PostProcessStageFn;
    if let Some(PostProcessStageFn) = PostProcessStageFn {
        let retval = lsrk_call_processfn(ark_mem, PostProcessStageFn, tcur, &ycur);
        if retval != 0 {
            return ARK_POSTPROCESS_STAGE_FAIL;
        }
    }

    /* Evaluate stage RHS */

    /* apply user-supplied stage preprocessing function (if supplied) */
    let PreRhsFn = ark_mem.borrow().PreRhsFn;
    if let Some(PreRhsFn) = PreRhsFn {
        let retval = lsrk_call_processfn(ark_mem, PreRhsFn, tcur, &ycur);
        if retval != 0 {
            return ARK_PRERHSFN_FAIL;
        }
    }

    let retval = lsrk_call_fe(ark_mem, tcur, &ycur, &tempv3);
    lsrkStep_mem_mut(ark_mem).nfe += 1;

    if retval < 0 {
        return ARK_RHSFUNC_FAIL;
    }
    if retval > 0 {
        return RHSFUNC_RECVR;
    }

    /* Begin stage 3 */
    tcur = tn + hp5;
    ark_mem.borrow_mut().tcur = tcur;
    lsrkStep_mem_mut(ark_mem).istage = 3;
    let cvals = [ONE / THREE, TWO / THREE, ONE / SIX * h];
    let Xvecs = [ycur.clone(), yn.clone(), tempv3.clone()];
    let retval = N_VLinearCombination(3, &cvals, &Xvecs, &ycur);
    if retval != 0 {
        return ARK_VECTOROP_ERR;
    }
    if !ark_mem.borrow().fixedstep {
        N_VLinearSum(ONE, &tempv1, hrsinv, &tempv3, &tempv1);
    }

    /* apply user-supplied stage postprocessing function (if supplied) */
    let PostProcessStageFn = ark_mem.borrow().PostProcessStageFn;
    if let Some(PostProcessStageFn) = PostProcessStageFn {
        let retval = lsrk_call_processfn(ark_mem, PostProcessStageFn, tcur, &ycur);
        if retval != 0 {
            return ARK_POSTPROCESS_STAGE_FAIL;
        }
    }

    /* Evaluate stage RHS */

    /* apply user-supplied stage preprocessing function (if supplied) */
    let PreRhsFn = ark_mem.borrow().PreRhsFn;
    if let Some(PreRhsFn) = PreRhsFn {
        let retval = lsrk_call_processfn(ark_mem, PreRhsFn, tcur, &ycur);
        if retval != 0 {
            return ARK_PRERHSFN_FAIL;
        }
    }

    let retval = lsrk_call_fe(ark_mem, tcur, &ycur, &tempv3);
    lsrkStep_mem_mut(ark_mem).nfe += 1;

    if retval < 0 {
        return ARK_RHSFUNC_FAIL;
    }
    if retval > 0 {
        return RHSFUNC_RECVR;
    }

    /* Compute the time step solution and embedding */
    tcur = tn + h;
    ark_mem.borrow_mut().tcur = tcur;
    lsrkStep_mem_mut(ark_mem).istage = 4;
    N_VLinearSum(ONE, &ycur, hp5, &tempv3, &ycur);

    /* apply user-supplied step postprocessing function (if supplied) */
    let PostProcessStepFn = ark_mem.borrow().PostProcessStepFn;
    if let Some(PostProcessStepFn) = PostProcessStepFn {
        let retval = lsrk_call_processfn(ark_mem, PostProcessStepFn, tcur, &ycur);
        if retval != 0 {
            return ARK_POSTPROCESS_STEP_FAIL;
        }
    }

    /* Compute yerr (if step adaptivity enabled) */
    if !ark_mem.borrow().fixedstep {
        N_VLinearSum(ONE, &tempv1, hrsinv, &tempv3, &tempv1);

        N_VLinearSum(ONE, &ycur, -ONE, &tempv1, &tempv1);
        *dsmPtr = N_VWrmsNorm(&tempv1, &ewt);
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  lsrkStep_TakeStepSSP104:

  This routine serves the primary purpose of the LSRKStepSSP104 module:
  it performs a single SSP104 step (with embedding).

  The output variable dsmPtr should contain estimate of the
  weighted local error if an embedding is present; otherwise it
  should be 0.

  The variables (ark_mem->tcur, ark_mem->ycur) should
  contain the current time and solution at the end of this time step.

  The input/output variable nflagPtr is used to gauge convergence
  of any algebraic solvers within the step.  As this routine
  involves no algebraic solve, it is set to 0 (success).

  The return value from this routine is:
            0 => step completed successfully
           >0 => step encountered recoverable failure;
                 reduce step and retry (if possible)
           <0 => step encountered unrecoverable failure
  ---------------------------------------------------------------*/
pub fn lsrkStep_TakeStepSSP104(
    ark_mem: &ARKodeMem,
    dsmPtr: &mut sunrealtype,
    nflagPtr: &mut i32,
) -> i32 {
    /* initialize algebraic solver convergence flag to success,
       temporal error estimate to zero */
    *nflagPtr = ARK_SUCCESS;
    *dsmPtr = ZERO;

    /* access ARKodeLSRKStepMem structure */
    let retval = lsrkStep_AccessStepMem(ark_mem, "lsrkStep_TakeStepSSP104");
    if retval != ARK_SUCCESS {
        return retval;
    }

    let (h, tn, yn, ycur, fn_, ewt, tempv1, tempv2, tempv3) = {
        let m = ark_mem.borrow();
        (
            m.h,
            m.tn,
            m.yn.clone().expect("yn"),
            m.ycur.clone().expect("ycur"),
            m.fn_.clone().expect("fn"),
            m.ewt.clone().expect("ewt"),
            m.tempv1.clone().expect("tempv1"),
            m.tempv2.clone().expect("tempv2"),
            m.tempv3.clone().expect("tempv3"),
        )
    };

    /* Initialize the current stage index */
    lsrkStep_mem_mut(ark_mem).istage = 0;

    /* Initialize method coefficients */
    let hsixth = h / SIX;
    let hfifth = h / FIVE;

    /* Begin stage 0 */

    /* The method is not FSAL. Therefore, fn is computed at the beginning
       of the step unless ARKODE updated fn. */
    if !ark_mem.borrow().fn_is_current {
        /* call the user-supplied pre-RHS function (if supplied) */
        let PreRhsFn = ark_mem.borrow().PreRhsFn;
        if let Some(PreRhsFn) = PreRhsFn {
            let retval = lsrk_call_processfn(ark_mem, PreRhsFn, tn, &yn);
            if retval != 0 {
                return ARK_PRERHSFN_FAIL;
            }
        }

        let retval = lsrk_call_fe(ark_mem, tn, &yn, &fn_);
        lsrkStep_mem_mut(ark_mem).nfe += 1;
        if retval != ARK_SUCCESS {
            return ARK_RHSFUNC_FAIL;
        }
        ark_mem.borrow_mut().fn_is_current = SUNTRUE;
    }

    /* Copy yn into tempv2 for use in later stages */
    N_VScale(ONE, &yn, &tempv2);

    /* Begin stage 1 and accumulate embedding into tempv1 */
    let mut tcur = tn + hsixth;
    ark_mem.borrow_mut().tcur = tcur;
    lsrkStep_mem_mut(ark_mem).istage = 1;
    N_VLinearSum(ONE, &yn, hsixth, &fn_, &ycur);
    if !ark_mem.borrow().fixedstep {
        N_VLinearSum(ONE, &yn, hfifth, &fn_, &tempv1);
    }

    /* Evaluate stages j = 2,...,5 */
    for j in 2..=5 {
        /* Complete the previous stage (postprocess the stage, evaluate the RHS, and
           store it in tempv3) */

        /* apply user-supplied stage postprocessing function (if supplied) */
        let PostProcessStageFn = ark_mem.borrow().PostProcessStageFn;
        if let Some(PostProcessStageFn) = PostProcessStageFn {
            let retval = lsrk_call_processfn(ark_mem, PostProcessStageFn, tcur, &ycur);
            if retval != 0 {
                return ARK_POSTPROCESS_STAGE_FAIL;
            }
        }

        /* apply user-supplied stage preprocessing function (if supplied) */
        let PreRhsFn = ark_mem.borrow().PreRhsFn;
        if let Some(PreRhsFn) = PreRhsFn {
            let retval = lsrk_call_processfn(ark_mem, PreRhsFn, tcur, &ycur);
            if retval != 0 {
                return ARK_PRERHSFN_FAIL;
            }
        }

        let retval = lsrk_call_fe(ark_mem, tcur, &ycur, &tempv3);
        lsrkStep_mem_mut(ark_mem).nfe += 1;

        if retval < 0 {
            return ARK_RHSFUNC_FAIL;
        }
        if retval > 0 {
            return RHSFUNC_RECVR;
        }

        /* Begin stage j (update the state and embedding) */
        if j == 5 {
            tcur = tn + TWO * hsixth;
        } else {
            tcur = tn + j as sunrealtype * hsixth;
        }
        ark_mem.borrow_mut().tcur = tcur;
        lsrkStep_mem_mut(ark_mem).istage = j;
        N_VLinearSum(ONE, &ycur, hsixth, &tempv3, &ycur);
        if j == 4 && !ark_mem.borrow().fixedstep {
            N_VLinearSum(ONE, &tempv1, 0.3 * h, &tempv3, &tempv1);
        }
    }

    /* no need to call RHS preprocessing here, since the stage does not require
       a RHS function evaluation */

    /* Finish stage 5 by preparing for the final stage group */
    N_VLinearSum(1.0 / 25.0, &tempv2, 9.0 / 25.0, &ycur, &tempv2);
    N_VLinearSum(15.0, &tempv2, -5.0, &ycur, &ycur);

    /* apply user-supplied stage postprocessing function (if supplied) */
    let PostProcessStageFn = ark_mem.borrow().PostProcessStageFn;
    if let Some(PostProcessStageFn) = PostProcessStageFn {
        let retval = lsrk_call_processfn(ark_mem, PostProcessStageFn, tcur, &ycur);
        if retval != 0 {
            return ARK_POSTPROCESS_STAGE_FAIL;
        }
    }

    /* Evaluate stages j = 6,...,9 */
    for j in 6..=9 {
        /* Complete the previous stage (evaluate the RHS and store in tempv3) */

        /* apply user-supplied stage preprocessing function (if supplied) */
        let PreRhsFn = ark_mem.borrow().PreRhsFn;
        if let Some(PreRhsFn) = PreRhsFn {
            let retval = lsrk_call_processfn(ark_mem, PreRhsFn, tcur, &ycur);
            if retval != 0 {
                return ARK_PRERHSFN_FAIL;
            }
        }

        let retval = lsrk_call_fe(ark_mem, tcur, &ycur, &tempv3);
        lsrkStep_mem_mut(ark_mem).nfe += 1;

        if retval < 0 {
            return ARK_RHSFUNC_FAIL;
        }
        if retval > 0 {
            return RHSFUNC_RECVR;
        }

        /* Begin stage j (update the state and embedding) */
        tcur = tn + (j - 3) as sunrealtype * hsixth;
        ark_mem.borrow_mut().tcur = tcur;
        lsrkStep_mem_mut(ark_mem).istage = j;
        N_VLinearSum(ONE, &ycur, hsixth, &tempv3, &ycur);

        if j == 7 && !ark_mem.borrow().fixedstep {
            N_VLinearSum(ONE, &tempv1, hfifth, &tempv3, &tempv1);
        }
        if j == 9 && !ark_mem.borrow().fixedstep {
            N_VLinearSum(ONE, &tempv1, 0.3 * h, &tempv3, &tempv1);
        }

        /* apply user-supplied stage postprocessing function (if supplied) */
        let PostProcessStageFn = ark_mem.borrow().PostProcessStageFn;
        if let Some(PostProcessStageFn) = PostProcessStageFn {
            let retval = lsrk_call_processfn(ark_mem, PostProcessStageFn, tcur, &ycur);
            if retval != 0 {
                return ARK_POSTPROCESS_STAGE_FAIL;
            }
        }
    }

    /* Complete the previous stage (evaluate the RHS and store it in tempv3) */

    /* apply user-supplied stage preprocessing function (if supplied) */
    let PreRhsFn = ark_mem.borrow().PreRhsFn;
    if let Some(PreRhsFn) = PreRhsFn {
        let retval = lsrk_call_processfn(ark_mem, PreRhsFn, tcur, &ycur);
        if retval != 0 {
            return ARK_PRERHSFN_FAIL;
        }
    }

    let retval = lsrk_call_fe(ark_mem, tcur, &ycur, &tempv3);
    lsrkStep_mem_mut(ark_mem).nfe += 1;

    if retval < 0 {
        return ARK_RHSFUNC_FAIL;
    }
    if retval > 0 {
        return RHSFUNC_RECVR;
    }

    /* Compute the final time step solution */
    lsrkStep_mem_mut(ark_mem).istage = 10;
    let cvals = [0.6, ONE, 0.1 * h];
    let Xvecs = [ycur.clone(), tempv2.clone(), tempv3.clone()];
    let retval = N_VLinearCombination(3, &cvals, &Xvecs, &ycur);
    if retval != 0 {
        return ARK_VECTOROP_ERR;
    }

    /* apply user-supplied step postprocessing function (if supplied) */
    let PostProcessStepFn = ark_mem.borrow().PostProcessStepFn;
    if let Some(PostProcessStepFn) = PostProcessStepFn {
        let retval = lsrk_call_processfn(ark_mem, PostProcessStepFn, tcur, &ycur);
        if retval != 0 {
            return ARK_POSTPROCESS_STEP_FAIL;
        }
    }

    /* Compute yerr (if step adaptivity enabled) */
    if !ark_mem.borrow().fixedstep {
        N_VLinearSum(ONE, &ycur, -ONE, &tempv1, &tempv1);
        *dsmPtr = N_VWrmsNorm(&tempv1, &ewt);
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  lsrkStep_Free frees all LSRKStep memory.
  ---------------------------------------------------------------*/
pub fn lsrkStep_Free(ark_mem: &ARKodeMem) {
    /* C also returns immediately for a NULL ark_mem; unreachable here */

    let mut m = ark_mem.borrow_mut();

    /* conditional frees on non-NULL LSRKStep module */
    if m.step_mem.is_none() {
        return;
    }

    let (free_cvals, free_Xvecs, nfusedopvecs) = {
        let step_mem = m
            .step_mem
            .as_mut()
            .expect("step_mem set")
            .downcast_mut::<ARKodeLSRKStepMemRec>()
            .expect("LSRKStep step memory");

        /* free the reusable arrays for fused vector interface */
        let free_cvals = !step_mem.cvals.is_empty();
        if free_cvals {
            step_mem.cvals = Vec::new();
        }
        let free_Xvecs = !step_mem.Xvecs.is_empty();
        if free_Xvecs {
            step_mem.Xvecs = Vec::new();
        }
        (free_cvals, free_Xvecs, step_mem.nfusedopvecs as i64)
    };
    if free_cvals {
        m.lrw -= nfusedopvecs;
    }
    if free_Xvecs {
        m.liw -= nfusedopvecs;
    }

    /* free the time stepper module itself */
    m.step_mem = None;
}

/*---------------------------------------------------------------
  lsrkStep_PrintMem:

  This routine outputs the memory from the LSRKStep structure to
  a specified file pointer (useful when debugging).
  ---------------------------------------------------------------*/
pub fn lsrkStep_PrintMem(ark_mem: &ARKodeMem, outfile: &SUNFile) {
    /* access ARKodeLSRKStepMem structure */
    let retval = lsrkStep_AccessStepMem(ark_mem, "lsrkStep_PrintMem");
    if retval != ARK_SUCCESS {
        return;
    }

    /* print integrator memory to file. C's `default:` arm ("Invalid
       method option.") is unreachable: the Rust enum has exactly the five
       upstream values. */
    let LSRKmethod = lsrkStep_mem_mut(ark_mem).LSRKmethod;
    match LSRKmethod {
        ARKODE_LSRK_RKC_2 => outfile.write_str("LSRKStep RKC time step module memory:\n"),
        ARKODE_LSRK_RKL_2 => outfile.write_str("LSRKStep RKL time step module memory:\n"),
        ARKODE_LSRK_SSP_S_2 => outfile.write_str("LSRKStep SSP(s,2) time step module memory:\n"),
        ARKODE_LSRK_SSP_S_3 => outfile.write_str("LSRKStep SSP(s,3) time step module memory:\n"),
        ARKODE_LSRK_SSP_10_4 => outfile.write_str("LSRKStep SSP(10,4) time step module memory:\n"),
    }

    let (q, p, istage, req_stages, is_SSP) = {
        let step_mem = lsrkStep_mem_mut(ark_mem);
        (
            step_mem.q,
            step_mem.p,
            step_mem.istage,
            step_mem.req_stages,
            step_mem.is_SSP,
        )
    };

    outfile.write_str(&format!("LSRKStep: q                   = {q}\n"));
    outfile.write_str(&format!("LSRKStep: p                   = {p}\n"));
    outfile.write_str(&format!("LSRKStep: istage              = {istage}\n"));
    outfile.write_str(&format!("LSRKStep: req_stages          = {req_stages}\n"));

    /* C's trailing `else` arm ("Invalid method type.") is unreachable for
       a `sunbooleantype`. */
    if is_SSP {
        let nfe = lsrkStep_mem_mut(ark_mem).nfe;
        outfile.write_str(&format!("LSRKStep: nfe                 = {nfe}\n"));
    } else {
        let (
            dom_eig_nst,
            stage_max,
            stage_max_limit,
            dom_eig_freq,
            num_init_warmups,
            num_warmups,
            nfe,
            nfeDQ,
            num_dee_iters,
            dom_eig_num_evals,
            lambdaR,
            lambdaI,
            spectral_radius,
            spectral_radius_max,
            spectral_radius_min,
            dom_eig_safety,
            rkc_damping,
            dom_eig_update,
            dom_eig_is_current,
            use_ellipse,
            DEE,
        ) = {
            let step_mem = lsrkStep_mem_mut(ark_mem);
            (
                step_mem.dom_eig_nst,
                step_mem.stage_max,
                step_mem.stage_max_limit,
                step_mem.dom_eig_freq,
                step_mem.num_init_warmups,
                step_mem.num_warmups,
                step_mem.nfe,
                step_mem.nfeDQ,
                step_mem.num_dee_iters,
                step_mem.dom_eig_num_evals,
                step_mem.lambdaR,
                step_mem.lambdaI,
                step_mem.spectral_radius,
                step_mem.spectral_radius_max,
                step_mem.spectral_radius_min,
                step_mem.dom_eig_safety,
                step_mem.rkc_damping,
                step_mem.dom_eig_update,
                step_mem.dom_eig_is_current,
                step_mem.use_ellipse,
                step_mem.DEE.clone(),
            )
        };

        /* output integer quantities */
        outfile.write_str(&format!("LSRKStep: dom_eig_nst           = {dom_eig_nst}\n"));
        outfile.write_str(&format!("LSRKStep: stage_max             = {stage_max}\n"));
        outfile.write_str(&format!(
            "LSRKStep: stage_max_limit       = {stage_max_limit}\n"
        ));
        outfile.write_str(&format!("LSRKStep: dom_eig_freq          = {dom_eig_freq}\n"));
        outfile.write_str(&format!(
            "LSRKStep: num_init_warmups      = {num_init_warmups}\n"
        ));
        outfile.write_str(&format!("LSRKStep: num_warmups           = {num_warmups}\n"));

        /* output long integer quantities */
        outfile.write_str(&format!("LSRKStep: nfe                   = {nfe}\n"));
        if DEE.is_some() {
            outfile.write_str(&format!("LSRKStep: nfeDQ               = {nfeDQ}\n"));
            outfile.write_str(&format!("LSRKStep: num_iters           = {num_dee_iters}\n"));
        }
        outfile.write_str(&format!(
            "LSRKStep: dom_eig_num_evals     = {dom_eig_num_evals}\n"
        ));

        /* output sunrealtype quantities */
        // TODO(SRB): temporary fix for complex numbers
        outfile.write_str(&format!(
            "LSRKStep: dom_eig               = {}{}i\n",
            sun_format_g(lambdaR),
            sun_format_sg(lambdaI)
        ));
        outfile.write_str(&format!(
            "LSRKStep: spectral_radius       = {}\n",
            sun_format_g(spectral_radius)
        ));
        outfile.write_str(&format!(
            "LSRKStep: spectral_radius_max   = {}\n",
            sun_format_g(spectral_radius_max)
        ));
        outfile.write_str(&format!(
            "LSRKStep: spectral_radius_min   = {}\n",
            sun_format_g(spectral_radius_min)
        ));
        outfile.write_str(&format!(
            "LSRKStep: dom_eig_safety        = {}\n",
            sun_format_g(dom_eig_safety)
        ));
        outfile.write_str(&format!(
            "LSRKStep: rkc_damping           = {}\n",
            sun_format_g(rkc_damping)
        ));

        /* output sunbooleantype quantities */
        outfile.write_str(&format!(
            "LSRKStep: dom_eig_update        = {}\n",
            dom_eig_update as i32
        ));
        outfile.write_str(&format!(
            "LSRKStep: dom_eig_is_current    = {}\n",
            dom_eig_is_current as i32
        ));
        outfile.write_str(&format!(
            "LSRKStep: use_ellipse          = {}\n",
            use_ellipse as i32
        ));

        if let Some(DEE) = DEE {
            let retval = SUNDomEigEstimator_Write(&DEE, outfile);
            if retval != SUN_SUCCESS {
                arkProcessError(
                    Some(ark_mem),
                    ARK_DEE_FAIL,
                    line!() as i32,
                    "lsrkStep_PrintMem",
                    file!(),
                    "SUNDomEigEstimator_Write failed",
                );
            }
        }
    }
}

/*===============================================================
  Internal utility routines
  ===============================================================*/

/*---------------------------------------------------------------
  lsrkStep_AccessARKODEStepMem:

  Shortcut routine to unpack both ark_mem and step_mem structures
  from void* pointer.  If either is missing it returns ARK_MEM_NULL.

  The Rust seam is a presence check (contract §3): `arkode_mem` is a
  non-null handle by construction, and callers reach the content with
  `lsrkStep_mem_mut` at each use site.
  ---------------------------------------------------------------*/
pub fn lsrkStep_AccessARKODEStepMem(arkode_mem: &ARKodeMem, fname: &str) -> i32 {
    /* access ARKodeLSRKStepMem structure */
    if arkode_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(arkode_mem),
            ARK_MEM_NULL,
            line!() as i32,
            fname,
            file!(),
            MSG_LSRKSTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }
    ARK_SUCCESS
}

/*---------------------------------------------------------------
  lsrkStep_AccessStepMem:

  Shortcut routine to unpack the step_mem structure from
  ark_mem.  If missing it returns ARK_MEM_NULL.
  ---------------------------------------------------------------*/
pub fn lsrkStep_AccessStepMem(ark_mem: &ARKodeMem, fname: &str) -> i32 {
    /* access ARKodeLSRKStepMem structure */
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            fname,
            file!(),
            MSG_LSRKSTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }
    ARK_SUCCESS
}

/*---------------------------------------------------------------
  lsrkStep_DomEigUpdateLogic:

  This routine checks if the step is accepted or not and reassigns
  the dom_eig update flags accordingly.
  ---------------------------------------------------------------*/
pub fn lsrkStep_DomEigUpdateLogic(ark_mem: &ARKodeMem, dsm: sunrealtype, fnew: &N_Vector) {
    if dsm <= ONE {
        let fn_ = ark_mem.borrow().fn_.clone().expect("fn");
        N_VScale(ONE, fnew, &fn_);
        ark_mem.borrow_mut().fn_is_current = SUNTRUE;

        let nst = ark_mem.borrow().nst;
        let mut step_mem = lsrkStep_mem_mut(ark_mem);
        let const_Jac = step_mem.const_Jac;
        step_mem.dom_eig_is_current = const_Jac == SUNTRUE;

        step_mem.dom_eig_update = SUNFALSE;
        if nst + 1 >= step_mem.dom_eig_nst + step_mem.dom_eig_freq {
            let dom_eig_is_current = step_mem.dom_eig_is_current;
            step_mem.dom_eig_update = !dom_eig_is_current;
        }
    } else {
        let mut step_mem = lsrkStep_mem_mut(ark_mem);
        let dom_eig_is_current = step_mem.dom_eig_is_current;
        step_mem.dom_eig_update = !dom_eig_is_current;
    }
}

/*---------------------------------------------------------------
  lsrkStep_ComputeNewDomEig:

  This routine computes new dom_eig and returns SUN_SUCCESS.
  ---------------------------------------------------------------*/
pub fn lsrkStep_ComputeNewDomEig(ark_mem: &ARKodeMem) -> i32 {
    /* C initializes `retval = SUN_SUCCESS`; every path below assigns it
       before the final `return retval`, so the binding is left uninit. */
    let mut retval: i32;

    /* C aliases &step_mem->lambdaR / &step_mem->lambdaI into the
       estimator/callback; the port hands over locals and writes them back
       immediately after each call. */
    let (mut lambdaR, mut lambdaI) = {
        let step_mem = lsrkStep_mem_mut(ark_mem);
        (step_mem.lambdaR, step_mem.lambdaI)
    };

    let DEE = lsrkStep_mem_mut(ark_mem).DEE.clone();
    let dom_eig_fn = lsrkStep_mem_mut(ark_mem).dom_eig_fn;

    if let Some(DEE) = DEE {
        retval = SUNDomEigEstimator_Estimate(&DEE, &mut lambdaR, &mut lambdaI);
        {
            let mut step_mem = lsrkStep_mem_mut(ark_mem);
            step_mem.lambdaR = lambdaR;
            step_mem.lambdaI = lambdaI;
            step_mem.dom_eig_num_evals += 1;
        }
        if retval != SUN_SUCCESS {
            arkProcessError(
                Some(ark_mem),
                ARK_DEE_FAIL,
                line!() as i32,
                "lsrkStep_ComputeNewDomEig",
                file!(),
                "SUNDomEigEstimator_Estimate failed",
            );
            return ARK_DEE_FAIL;
        }

        let mut num_iters: i64 = 0;
        retval = SUNDomEigEstimator_GetNumIters(&DEE, &mut num_iters);
        if retval != SUN_SUCCESS {
            arkProcessError(
                Some(ark_mem),
                ARK_DEE_FAIL,
                line!() as i32,
                "lsrkStep_ComputeNewDomEig",
                file!(),
                "SUNDomEigEstimator_GetNumIters failed",
            );
            return ARK_DEE_FAIL;
        }
        lsrkStep_mem_mut(ark_mem).num_dee_iters += num_iters;

        /* After the first call to SUNDomEigEstimator_Estimate, the number of warmups is set to
           num_warmups, this allows the successive calls to
           SUNDomEigEstimator_Estimate to use a different number of warmups. */
        let init_warmup = lsrkStep_mem_mut(ark_mem).init_warmup;
        if init_warmup {
            let num_warmups = lsrkStep_mem_mut(ark_mem).num_warmups;
            retval = SUNDomEigEstimator_SetNumPreprocessIters(&DEE, num_warmups);
            if retval != SUN_SUCCESS {
                arkProcessError(
                    Some(ark_mem),
                    ARK_DEE_FAIL,
                    line!() as i32,
                    "lsrkStep_ComputeNewDomEig",
                    file!(),
                    "SUNDomEigEstimator_SetNumPreprocessIters failed",
                );
                return ARK_DEE_FAIL;
            }
            lsrkStep_mem_mut(ark_mem).init_warmup = SUNFALSE;
        }
    } else if let Some(dom_eig_fn) = dom_eig_fn {
        let (tn, yn, fn_, tempv1, tempv2, tempv3) = {
            let m = ark_mem.borrow();
            (
                m.tn,
                m.yn.clone().expect("yn"),
                m.fn_.clone().expect("fn"),
                m.tempv1.clone().expect("tempv1"),
                m.tempv2.clone().expect("tempv2"),
                m.tempv3.clone().expect("tempv3"),
            )
        };
        retval = lsrk_call_dom_eig_fn(
            ark_mem,
            dom_eig_fn,
            tn,
            &yn,
            &fn_,
            &mut lambdaR,
            &mut lambdaI,
            &tempv1,
            &tempv2,
            &tempv3,
        );
        {
            let mut step_mem = lsrkStep_mem_mut(ark_mem);
            step_mem.lambdaR = lambdaR;
            step_mem.lambdaI = lambdaI;
            step_mem.dom_eig_num_evals += 1;
        }
        if retval != ARK_SUCCESS {
            arkProcessError(
                Some(ark_mem),
                ARK_DOMEIG_FAIL,
                line!() as i32,
                "lsrkStep_ComputeNewDomEig",
                file!(),
                "Unable to estimate the dominant eigenvalue",
            );
            return ARK_DOMEIG_FAIL;
        }
    } else {
        arkProcessError(
            Some(ark_mem),
            ARK_DOMEIG_FAIL,
            line!() as i32,
            "lsrkStep_ComputeNewDomEig",
            file!(),
            "Unable to estimate the dominant eigenvalue: Either a user provided function or a \
             SUNDomEigEstimator is required",
        );
        return ARK_DOMEIG_FAIL;
    }

    let h = ark_mem.borrow().h;
    let (lambdaR, lambdaI) = {
        let step_mem = lsrkStep_mem_mut(ark_mem);
        (step_mem.lambdaR, step_mem.lambdaI)
    };

    if lambdaR * h > SUNRsqrt(SUN_UNIT_ROUNDOFF) {
        arkProcessError(
            None,
            ARK_DOMEIG_FAIL,
            line!() as i32,
            "lsrkStep_ComputeNewDomEig",
            file!(),
            "lambdaR*h must be nonpositive",
        );
        return ARK_DOMEIG_FAIL;
    } else if lambdaR == 0.0 && SUNRabs(lambdaI) > 0.0 {
        arkProcessError(
            None,
            ARK_DOMEIG_FAIL,
            line!() as i32,
            "lsrkStep_ComputeNewDomEig",
            file!(),
            "DomEig cannot be purely imaginary",
        );
        return ARK_DOMEIG_FAIL;
    }

    let nst = ark_mem.borrow().nst;
    {
        let mut step_mem = lsrkStep_mem_mut(ark_mem);
        let dom_eig_safety = step_mem.dom_eig_safety;
        step_mem.lambdaR *= dom_eig_safety;
        step_mem.lambdaI *= dom_eig_safety;
        let lambdaR = step_mem.lambdaR;
        let lambdaI = step_mem.lambdaI;
        step_mem.spectral_radius = SUNRsqrt(SUNSQR(lambdaR) + SUNSQR(lambdaI));

        step_mem.dom_eig_is_current = SUNTRUE;
        step_mem.dom_eig_nst = nst;

        let spectral_radius = step_mem.spectral_radius;
        let spectral_radius_max = step_mem.spectral_radius_max;
        step_mem.spectral_radius_max = if spectral_radius > spectral_radius_max {
            spectral_radius
        } else {
            spectral_radius_max
        };

        if spectral_radius < step_mem.spectral_radius_min || nst == 0 {
            step_mem.spectral_radius_min = spectral_radius;
        }

        step_mem.dom_eig_update = SUNFALSE;
    }

    retval
}

/*---------------------------------------------------------------
  lsrkStep_RKC_CheckStabilityNorm:

  This routine computes the stability norm for RKC methods.
  If use_ellipse is SUNTRUE, we use a heuristic that approximates the stability region by an ellipse.
  If use_ellipse is SUNFALSE, we compute the stability norm directly from the stability function using
  the Chebyshev polynomial.

  C takes `step_mem` directly; the port reaches the same record through
  `ark_mem` (the guard is dropped before `lsrkStep_cheb_T_complex`).
  ---------------------------------------------------------------*/
pub fn lsrkStep_RKC_CheckStabilityNorm(
    ark_mem: &ARKodeMem,
    num_stages: i32,
    h: sunrealtype,
    stability_norm: &mut sunrealtype,
) -> i32 {
    let ss = num_stages as sunrealtype;
    let (use_ellipse, rkc_damping, lambdaR, lambdaI) = {
        let step_mem = lsrkStep_mem_mut(ark_mem);
        (
            step_mem.use_ellipse,
            step_mem.rkc_damping,
            step_mem.lambdaR,
            step_mem.lambdaI,
        )
    };
    let zR = h * lambdaR;
    let zI = h * lambdaI;

    if use_ellipse {
        /* The stability region of the damped RKC method is approximated by an ellipse
        centered at (-a,0), with horizontal semiaxis a and vertical semiaxis b, so
        that its vertices are located at (0,0), (-2a,0), and (-a,+/-b). These
        quantities depend on the damping parameter. Also, b is estimated
        heuristically from the ellipse aspect ratio, taken as approximately 1.825s,
        where s is the number of stages (for s=2, the ratio is approximated as 0.6).
        This heuristic reflects the observed near-linear growth of the imaginary
        extent with the number of stages. The numerical factors (1.825 and 0.6)
        were obtained empirically from stability-region plots using the default
        damping parameter and may change if the damping is modified. */
        let a = (TWO / THREE) * (SUNSQR(ss) - ONE) * (ONE - TWO / 15.0 * rkc_damping) / TWO;
        let b = a / (if num_stages == 2 { 0.6 } else { 1.825 * ss });

        *stability_norm = SUNRsqrt(SUNSQR((zR + a) / a) + SUNSQR(zI / b));
    } else {
        let w0 = ONE + rkc_damping / (ss * ss);
        let th = SUNRacosh(w0);
        let sh = SUNRsinh(th);
        let ch = SUNRcosh(th);

        let Ts = SUNRcosh(ss * th);
        let Ts_p = ss * SUNRsinh(ss * th) / sh;
        let Ts_pp = (ss * ss * SUNRcosh(ss * th) / (sh * sh))
            - ss * ch * SUNRsinh(ss * th) / (sh * sh * sh);

        let b_s = Ts_pp / (Ts_p * Ts_p);
        let a_s = ONE - b_s * Ts;
        let w1 = Ts_p / Ts_pp;

        let wr = w0 + w1 * zR;
        let wi = w1 * zI;

        let mut TsR: sunrealtype = ZERO;
        let mut TsI: sunrealtype = ZERO;
        let retval = lsrkStep_cheb_T_complex(num_stages, wr, wi, &mut TsR, &mut TsI);
        if retval != ARK_SUCCESS {
            return retval;
        }

        let Ps_ZR = a_s + b_s * TsR;
        let Ps_ZI = b_s * TsI;

        *stability_norm = SUNRsqrt(SUNSQR(Ps_ZR) + SUNSQR(Ps_ZI));
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  lsrkStep_RKL_CheckStabilityNorm:

  This routine computes the stability norm for RKL methods.
  If use_ellipse is SUNTRUE, we use a heuristic that approximates the stability region by an ellipse.
  If use_ellipse is SUNFALSE, we compute the stability norm directly from the stability function using
  the Chebyshev polynomial.
  ---------------------------------------------------------------*/
pub fn lsrkStep_RKL_CheckStabilityNorm(
    ark_mem: &ARKodeMem,
    num_stages: i32,
    h: sunrealtype,
    stability_norm: &mut sunrealtype,
) -> i32 {
    let ss = num_stages as sunrealtype;
    let (use_ellipse, lambdaR, lambdaI) = {
        let step_mem = lsrkStep_mem_mut(ark_mem);
        (step_mem.use_ellipse, step_mem.lambdaR, step_mem.lambdaI)
    };
    let zR = h * lambdaR;
    let zI = h * lambdaI;

    if use_ellipse {
        /* The stability region of the RKL method is approximated by an ellipse
           centered at (-a,0), with horizontal semiaxis a and vertical semiaxis b,
           so that its vertices are located at (0,0), (-2a,0), and (-a,+/-b).
           The half-height b is estimated heuristically from the ellipse aspect
           ratio a/b based on the number of stages as follows:
             s = 2 -> 0.3 s
             s = 3 -> 0.75 s
             s = 4 -> 0.665 s
             s = 5 -> 0.665 s
             s = 6 to 20 -> 0.635 s
             s >= 20 and odd -> 0.6 s
             s >= 20 and even -> 0.53 s */
        let aspect_ratio: [sunrealtype; 7] = [
            0.3 * ss,   /* s = 2 */
            0.75 * ss,  /* s = 3 */
            0.665 * ss, /* s = 4 */
            0.665 * ss, /* s = 5 */
            0.635 * ss, /* s = 6 to 20 */
            0.6 * ss,   /* s >= 20 and odd */
            0.53 * ss,  /* s >= 20 and even */
        ];
        let a = ((TWO * ss + ONE) * (TWO * ss + ONE) - 9.0) / 16.0;
        let b: sunrealtype;

        if num_stages < 7 {
            b = a / (aspect_ratio[(num_stages - 2) as usize]);
        } else if num_stages <= 20 {
            b = a / (aspect_ratio[4]);
        } else {
            b = a / (aspect_ratio[(6 - num_stages % 2) as usize]);
        }

        *stability_norm = SUNRsqrt(SUNSQR((zR + a) / a) + SUNSQR(zI / b));
    } else {
        let b_s = (ss * ss + ss - TWO) / (TWO * ss * (ss + ONE));
        let a_s = ONE - b_s;
        let w1 = FOUR / (ss * ss + ss - TWO); /* Eq.(15) in Meyer et al. (2014) */
        let wr = ONE + w1 * zR;
        let wi = w1 * zI;

        let mut PsR: sunrealtype = ZERO;
        let mut PsI: sunrealtype = ZERO;
        let retval = lsrkStep_legendre_P_complex(num_stages, wr, wi, &mut PsR, &mut PsI);
        if retval != ARK_SUCCESS {
            return retval;
        }

        let Ps_ZR = a_s + b_s * PsR;
        let Ps_ZI = b_s * PsI;

        *stability_norm = SUNRsqrt(SUNSQR(Ps_ZR) + SUNSQR(Ps_ZI));
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  lsrkStep_cheb_T_complex:

  This routine computes the Chebyshev polynomial of the first kind
  T_s(z) for complex argument z = zR + i*zI using the
  recurrence relation:
    T_0(z) = 1
    T_1(z) = z
    T_{k+1}(z) = 2*z*T_k(z) - T_{k-1}(z),  k = 1,...,s-1
  ---------------------------------------------------------------*/
pub fn lsrkStep_cheb_T_complex(
    s: i32,
    zR: sunrealtype,
    zI: sunrealtype,
    TsR: &mut sunrealtype,
    TsI: &mut sunrealtype,
) -> i32 {
    if s < 0 {
        arkProcessError(
            None,
            ARK_ILL_INPUT,
            line!() as i32,
            "lsrkStep_cheb_T_complex",
            file!(),
            "s cannot be negative",
        );
        return ARK_ILL_INPUT;
    } else if s == 0 {
        *TsR = ONE;
        *TsI = ZERO;
        return ARK_SUCCESS;
    } else if s == 1 {
        *TsR = zR;
        *TsI = zI;
        return ARK_SUCCESS;
    } else {
        let mut Tkm1R = ONE; /* T_0(z) */
        let mut Tkm1I = ZERO;
        let mut TkR = zR; /* T_1(z) */
        let mut TkI = zI;
        for _k in 1..s {
            let Tkp1R = 2.0 * (zR * TkR - zI * TkI) - Tkm1R;
            let Tkp1I = 2.0 * (zR * TkI + zI * TkR) - Tkm1I;
            Tkm1R = TkR;
            Tkm1I = TkI;
            TkR = Tkp1R;
            TkI = Tkp1I;
        }
        *TsR = TkR;
        *TsI = TkI;
    }
    ARK_SUCCESS
}

/*---------------------------------------------------------------
  lsrkStep_legendre_P_complex:

  This routine computes the Legendre polynomial P_s(z) for complex
  argument z = zR + i*zI using the recurrence relation:
    P_0(z) = 1
    P_1(z) = z
    P_{k+1}(z) = ((2*k+1)*z*P_k(z) - k*P_{k-1}(z))/(k+1),  k = 1,...,s-1
  ---------------------------------------------------------------*/
pub fn lsrkStep_legendre_P_complex(
    s: i32,
    zR: sunrealtype,
    zI: sunrealtype,
    PsR: &mut sunrealtype,
    PsI: &mut sunrealtype,
) -> i32 {
    if s < 0 {
        arkProcessError(
            None,
            ARK_ILL_INPUT,
            line!() as i32,
            "lsrkStep_legendre_P_complex",
            file!(),
            "s cannot be negative",
        );
        return ARK_ILL_INPUT;
    } else if s == 0 {
        *PsR = ONE;
        *PsI = ZERO;
        return ARK_SUCCESS;
    } else if s == 1 {
        *PsR = zR;
        *PsI = zI;
        return ARK_SUCCESS;
    } else {
        let mut Pkm1R = ONE; /* P_0(z) */
        let mut Pkm1I = ZERO;
        let mut PkR = zR; /* P_1(z) */
        let mut PkI = zI;
        for k in 1..s {
            let kr = k as sunrealtype;
            let Pkp1R = ((TWO * kr + ONE) * (zR * PkR - zI * PkI) - kr * Pkm1R) / (kr + ONE);
            let Pkp1I = ((TWO * kr + ONE) * (zR * PkI + zI * PkR) - kr * Pkm1I) / (kr + ONE);
            Pkm1R = PkR;
            Pkm1I = PkI;
            PkR = Pkp1R;
            PkI = Pkp1I;
        }
        *PsR = PkR;
        *PsI = PkI;
    }
    ARK_SUCCESS
}

/*---------------------------------------------------------------
  lsrkStep_DQJtimes:

  This routine generates a difference quotient approximation to
  the Jacobian-vector product f_y(t,y) * v. The approximation is
  Jv = [f(y + v*sig) - f(y)]/sig, where sig = 1 / ||v||_WRMS,
  i.e. the WRMS norm of v*sig is 1.

  Attached to the SUNDomEigEstimator as its `SUNATimesFn`, so C's
  `void* arkode_mem` is the estimator's `A_data` token (holding an
  `ARKodeMem` handle clone).
  ---------------------------------------------------------------*/
pub fn lsrkStep_DQJtimes(
    arkode_mem: &mut Option<Box<dyn Any>>,
    v: &N_Vector,
    Jv: &N_Vector,
) -> i32 {
    /* access ARKodeLSRKStepMem structure (C casts the void*; a missing or
       mistyped token is C's UB and maps to a deterministic panic) */
    let ark_mem: ARKodeMem = arkode_mem
        .as_mut()
        .and_then(|b| b.downcast_ref::<ARKodeMem>())
        .cloned()
        .expect("lsrkStep_DQJtimes arkode_mem token");
    let ark_mem = &ark_mem;
    let retval = lsrkStep_AccessARKODEStepMem(ark_mem, "lsrkStep_DQJtimes");
    if retval != ARK_SUCCESS {
        return retval;
    }

    let (t, y, work, ewt, fn_) = {
        let m = ark_mem.borrow();
        (
            m.tn,
            m.yn.clone().expect("yn"),
            m.tempv3.clone().expect("tempv3"),
            m.ewt.clone().expect("ewt"),
            m.fn_.clone().expect("fn"),
        )
    };

    /* Compute RHS function, if necessary. */
    let need_rhs = {
        let m = ark_mem.borrow();
        let step_nst = m
            .step_mem
            .as_ref()
            .expect("step_mem set")
            .downcast_ref::<ARKodeLSRKStepMemRec>()
            .expect("LSRKStep step memory")
            .step_nst;
        (!m.fn_is_current && m.initsetup) || (step_nst != m.nst)
    };
    if need_rhs {
        /* call the user-supplied pre-RHS function (if supplied) */
        let PreRhsFn = ark_mem.borrow().PreRhsFn;
        if let Some(PreRhsFn) = PreRhsFn {
            let retval = lsrk_call_processfn(ark_mem, PreRhsFn, t, &y);
            if retval != 0 {
                return ARK_PRERHSFN_FAIL;
            }
        }

        let retval = lsrk_call_fe(ark_mem, t, &y, &fn_);
        lsrkStep_mem_mut(ark_mem).nfeDQ += 1;
        if retval != ARK_SUCCESS {
            return ARK_RHSFUNC_FAIL;
        }
        ark_mem.borrow_mut().fn_is_current = SUNTRUE;
    }

    /* Initialize perturbation to 1/||v|| */
    let mut sig = ONE / N_VWrmsNorm(v, &ewt);

    /* C leaves `retval` holding the last `fe` return after the loop; the
       loop always runs at least once, so the seed value is never read. */
    let mut retval: i32 = 0;
    for _iter in 0..MAX_DQITERS {
        /* Set work = y + sig*v */
        N_VLinearSum(sig, v, ONE, &y, &work);

        /* call the user-supplied pre-RHS function (if supplied) */
        let PreRhsFn = ark_mem.borrow().PreRhsFn;
        if let Some(PreRhsFn) = PreRhsFn {
            let rv = lsrk_call_processfn(ark_mem, PreRhsFn, t, &work);
            if rv != 0 {
                return ARK_PRERHSFN_FAIL;
            }
        }
        /* Set Jv = f(tn, y+sig*v) */
        retval = lsrk_call_fe(ark_mem, t, &work, Jv);
        lsrkStep_mem_mut(ark_mem).nfeDQ += 1;
        if retval == 0 {
            break;
        }
        if retval < 0 {
            return -1;
        }

        /* If f failed recoverably, shrink sig and retry */
        sig *= 0.25;
    }

    /* If retval still isn't 0, return with a recoverable failure */
    if retval > 0 {
        return 1;
    }

    /* Replace Jv by (Jv - fn)/sig */
    let siginv = ONE / sig;
    N_VLinearSum(siginv, Jv, -siginv, &fn_, Jv);

    ARK_SUCCESS
}

/*===============================================================
  EOF
  ===============================================================*/
