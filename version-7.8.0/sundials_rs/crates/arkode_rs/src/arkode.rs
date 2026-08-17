//! Port of `src/arkode/arkode.c` — the main ARKODE infrastructure,
//! independent of the ARKODE time-step module, nonlinear solver, linear
//! solver and vector modules in use.
//!
//! `arkode_impl.h` (+ the constants/typedefs of `include/arkode/arkode.h`,
//! `arkode_adapt_impl.h`, `arkode_root_impl.h` and
//! `arkode_relaxation_impl.h`) is folded into `crate::arkode_impl`, which
//! also owns `arkProcessError` and every `MSG_ARK_*` message so that all
//! arkode modules share one definition.
//!
//! Reference build configuration: SUNDIALS_LOGGING_LEVEL = 2
//! (`SUNLogInfo`/`SUNLogInfoIf`/`SUNLogDebug`/`SUNLogExtraDebug*` call
//! sites omitted at translation time; `ARK_WARNING` paths are kept because
//! they print through the logger), profiling OFF
//! (`SUNDIALS_MARK_FUNCTION_BEGIN/END` omitted), error checks OFF
//! (`SUNAssert`/`SUNCheck*` are no-ops), monitoring ENABLED, serial
//! branches only. `SUNDIALS_DEBUG_PRINTVEC` is not defined, so the vector
//! dump inside `ARKodePrintMem` is dead code and is omitted.
//! `SUNDIALS_ENABLE_PYTHON` is not defined, so
//! `arkode_user_supplied_fn_table_destroy` is not called in `ARKodeFree`
//! (the `ark_mem->python = NULL` assignment is kept).
//!
//! Handle model: `ARKodeMem = Rc<RefCell<ARKodeMemRec>>`; `ark_mem->ycur`
//! is an `Rc` clone of the caller's `yout`/`y0`, so it aliases the user
//! buffer exactly as the C pointer copy does and no explicit copy-back is
//! required.
//!
//! `arkExpStab` is declared in `arkode_impl.h` but never defined anywhere
//! in the upstream C tree (dead prototype); it is therefore not ported.

use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

use crate::arkode_adapt::{arkAdapt, arkAdaptInit, arkPrintAdaptMem};
use crate::arkode_impl::*;
use crate::arkode_interp::{
    arkInterpCreate_Hermite, arkInterpCreate_Lagrange, arkInterpEvaluate, arkInterpFree,
    arkInterpInit, arkInterpPrintMem, arkInterpResize, arkInterpSetDegree, arkInterpUpdate,
};
use crate::arkode_io::{
    ARKodeGetAccumulatedError, ARKodeResetAccumulatedError, ARKodeSetDefaults, ARKodeSetStopTime,
};
use crate::arkode_mristep::{
    MRIStepInnerStepper, MRIStepInnerStepper_Create, MRIStepInnerStepper_GetForcingData,
    MRIStepInnerStepper_SetAccumulatedErrorGetFn,
    MRIStepInnerStepper_SetAccumulatedErrorResetFn, MRIStepInnerStepper_SetContent,
    MRIStepInnerStepper_SetEvolveFn, MRIStepInnerStepper_SetFullRhsFn,
    MRIStepInnerStepper_SetRTolFn, MRIStepInnerStepper_SetResetFn,
};
use crate::arkode_relaxation::{arkRelax, arkRelaxDestroy};
use crate::arkode_root::{arkPrintRootMem, arkRootCheck1, arkRootCheck2, arkRootCheck3, arkRootFree};

use sundials_core::sundials_adaptcontroller::*;
use sundials_core::sundials_context::SUNContext;
use sundials_core::sundials_errors::SUN_SUCCESS;
use sundials_core::sundials_math::*;
use sundials_core::sundials_nvector::*;
use sundials_core::sundials_types::*;
use sundials_core::sundials_utils::*;

/*===============================================================
  Module-local numeric constants

  `arkode.c` has no file-scope numeric `#define`s of its own: every
  named constant it uses (ZERO, TENTH, HALF, ONE, TWO, FOUR, FUZZ_FACTOR,
  H0_LBFACTOR, H0_UBFACTOR, H0_BIAS, H0_ITERS, ONEPSM, ...) comes from
  `arkode_impl.h`, which is folded into `crate::arkode_impl` and glob-
  imported above.  Per the frozen contract (section 7) those must NOT be
  redefined here.  The only bare literals in the C file
  (`SUN_RCONST(0.2)` in `arkHin`, `SUN_RCONST(0.75)` in
  `arkPredict_VariableOrder`, `SUN_RCONST(0.9)` in `arkCheckConstraints`)
  are written inline exactly where C writes them.
  ===============================================================*/

/*===============================================================
  Callback invocation helpers

  Granular borrow discipline: the `Box<dyn Any>` data token is taken out
  of the mem around every user callback call and restored afterwards on
  every path, and no mem borrow is held across the call.
  ===============================================================*/

/// Invoke the error-weight function
/// (C: `ark_mem->efun(y, ewt, ark_mem->e_data)`).
///
/// In C, `e_data` is `ark_mem` for the built-in `arkEwtSetSS`/`arkEwtSetSV`
/// and an alias of `user_data` when the user supplied `efun` through
/// `ARKodeWFtolerances` (`ARKodeSetUserData` keeps the alias in sync).
/// A `Box` cannot alias, so the port stores a boxed `ARKodeMem` handle in
/// `e_data` for the built-in case and passes the CURRENT `user_data` box
/// when `user_efun` is set (accepted deviation class 6).
fn ark_call_efun(ark_mem: &ARKodeMem, y: &N_Vector, ewt: &N_Vector) -> i32 {
    let (efun, user_efun) = {
        let m = ark_mem.borrow();
        (m.efun, m.user_efun)
    };
    let efun = efun.expect("efun set");
    if user_efun {
        let mut data = ark_mem.borrow_mut().user_data.take();
        let retval = efun(y, ewt, &mut data);
        ark_mem.borrow_mut().user_data = data;
        retval
    } else {
        let mut data = ark_mem.borrow_mut().e_data.take();
        let retval = efun(y, ewt, &mut data);
        ark_mem.borrow_mut().e_data = data;
        retval
    }
}

/// Invoke the residual-weight function
/// (C: `ark_mem->rfun(y, rwt, ark_mem->r_data)`); same `r_data` treatment
/// as `ark_call_efun`.
fn ark_call_rfun(ark_mem: &ARKodeMem, y: &N_Vector, rwt: &N_Vector) -> i32 {
    let (rfun, user_rfun) = {
        let m = ark_mem.borrow();
        (m.rfun, m.user_rfun)
    };
    let rfun = rfun.expect("rfun set");
    if user_rfun {
        let mut data = ark_mem.borrow_mut().user_data.take();
        let retval = rfun(y, rwt, &mut data);
        ark_mem.borrow_mut().user_data = data;
        retval
    } else {
        let mut data = ark_mem.borrow_mut().r_data.take();
        let retval = rfun(y, rwt, &mut data);
        ark_mem.borrow_mut().r_data = data;
        retval
    }
}

/// Invoke the user pre-step function
/// (C: `ark_mem->PreStepFn(t, y, step, attempt, ark_mem->user_data)`).
fn ark_call_prestepfn(
    ark_mem: &ARKodeMem,
    t: sunrealtype,
    y: &N_Vector,
    step: i64,
    attempt: i32,
) -> i32 {
    let f = ark_mem.borrow().PreStepFn.expect("PreStepFn set");
    let mut user_data = ark_mem.borrow_mut().user_data.take();
    let retval = f(t, y, step, attempt, &mut user_data);
    ark_mem.borrow_mut().user_data = user_data;
    retval
}

/*===============================================================
  Exported functions
  ===============================================================*/

/*---------------------------------------------------------------
  ARKodeResize:

  ARKodeResize re-initializes ARKODE's memory for a problem with a
  changing vector size.  It is assumed that the problem dynamics
  before and after the vector resize will be comparable, so that
  all time-stepping heuristics prior to calling ARKodeResize
  remain valid after the call.  If instead the dynamics should be
  re-calibrated, the ARKODE memory structure should be deleted
  with a call to ARKodeFree, and re-created with a call to
  *StepCreate.

  To aid in the vector-resize operation, the user can supply a
  vector resize function, that will take as input an N_Vector with
  the previous size, and return as output a corresponding vector
  of the new size.  If this function (of type ARKVecResizeFn) is
  not supplied (i.e. is set to NULL), then all existing N_Vectors
  will be destroyed and re-cloned from the input vector.

  In the case that the dynamical time scale should be modified
  slightly from the previous time scale, an input "hscale" is
  allowed, that will re-scale the upcoming time step by the
  specified factor.  If a value <= 0 is specified, the default of
  1.0 will be used.

  Other arguments:
  ark_mem          Existing ARKODE memory data structure.
  y0               The newly-sized solution vector, holding
                   the current dependent variable values.
  t0               The current value of the independent
                   variable.
  resize_data      User-supplied data structure that will be
                   passed to the supplied resize function.

  The return value is ARK_SUCCESS = 0 if no errors occurred, or
  a negative value otherwise.
  ---------------------------------------------------------------*/
pub fn ARKodeResize(
    arkode_mem: &ARKodeMem,
    y0: &N_Vector,
    hscale: sunrealtype,
    t0: sunrealtype,
    resize: Option<ARKVecResizeFn>,
    resize_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    let mut hscale = hscale;

    /* Check ark_mem: NULL-mem check handled by the type system */
    let ark_mem = arkode_mem;

    /* Check if ark_mem was allocated */
    if !ark_mem.borrow().MallocDone {
        arkProcessError(
            Some(ark_mem),
            ARK_NO_MALLOC,
            line!() as i32,
            "ARKodeResize",
            file!(),
            MSG_ARK_NO_MALLOC,
        );
        return ARK_NO_MALLOC;
    }

    /* Check for legal input parameters (NULL y0 handled by the type
    system; the Rc clone aliases the caller's vector exactly as the C
    pointer copy does) */
    ark_mem.borrow_mut().ycur = Some(y0.clone());

    /* Copy the input parameters into ARKODE state */
    {
        let mut m = ark_mem.borrow_mut();
        m.tcur = t0;
        m.tn = t0;
    }

    /* Update time-stepping parameters */
    /*   adjust upcoming step size depending on hscale */
    if hscale <= ZERO {
        hscale = ONE;
    }
    if hscale != ONE {
        let mut m = ark_mem.borrow_mut();

        /* Encode hscale into ark_mem structure */
        m.eta = hscale;
        m.hprime *= hscale;

        /* If next step would overtake tstop, adjust stepsize */
        if m.tstopset && (m.tcur + m.hprime - m.tstop) * m.hprime > ZERO {
            m.hprime = (m.tstop - m.tcur) * (ONE - FOUR * m.uround);
            m.eta = m.hprime / m.h;
        }
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

    /* Disable constraints, the user will need to set a new constraint vector for
       the updated problem size */
    {
        let mut constraints = ark_mem.borrow_mut().constraints.take();
        arkFreeVec(ark_mem, &mut constraints);
        ark_mem.borrow_mut().constraints = constraints;
    }

    /* Resize the solver vectors (using y0 as a template) */
    let resizeOK = arkResizeVectors(ark_mem, resize, resize_data, lrw_diff, liw_diff, y0);
    if !resizeOK {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_FAIL,
            line!() as i32,
            "ARKodeResize",
            file!(),
            "Unable to resize vector",
        );
        return ARK_MEM_FAIL;
    }

    /* Resize the interpolation structure memory */
    let interp = ark_mem.borrow().interp.clone();
    if let Some(interp) = interp {
        let retval = arkInterpResize(
            ark_mem,
            Some(&interp),
            resize,
            resize_data,
            lrw_diff,
            liw_diff,
            y0,
        );
        if retval != ARK_SUCCESS {
            arkProcessError(
                Some(ark_mem),
                retval,
                line!() as i32,
                "ARKodeResize",
                file!(),
                "Interpolation module resize failure",
            );
            return retval;
        }
    }

    /* Copy y0 into ark_yn to set the current solution */
    let yn = ark_mem.borrow().yn.clone().expect("yn allocated");
    N_VScale(ONE, y0, &yn);

    {
        let mut m = ark_mem.borrow_mut();
        m.fn_is_current = SUNFALSE;

        /* Indicate that problem needs to be initialized */
        m.initsetup = SUNTRUE;
        m.init_type = RESIZE_INIT;
        m.firststage = SUNTRUE;
    }

    /* Call the stepper-specific resize (if provided) */
    let step_resize = ark_mem.borrow().step_resize;
    if let Some(step_resize) = step_resize {
        return step_resize(ark_mem, y0, hscale, t0, resize, resize_data);
    }

    /* Problem has been successfully re-sized */
    ARK_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeReset:

  This routine resets an ARKode module to solve the same
  problem from the given time with the input state (all counter
  values are retained).
  ---------------------------------------------------------------*/
pub fn ARKodeReset(arkode_mem: &ARKodeMem, tR: sunrealtype, yR: &N_Vector) -> i32 {
    /* NULL-mem check: handled by the type system */
    let ark_mem = arkode_mem;

    /* Reset main ARKODE infrastructure */
    let retval = arkInit(ark_mem, tR, yR, RESET_INIT);
    if retval != ARK_SUCCESS {
        arkProcessError(
            Some(ark_mem),
            retval,
            line!() as i32,
            "ARKodeReset",
            file!(),
            "ARKode reset failure",
        );
        return retval;
    }

    /* Call stepper routine to perform remaining reset operations (if provided) */
    let step_reset = ark_mem.borrow().step_reset;
    if let Some(step_reset) = step_reset {
        return step_reset(ark_mem, tR, yR);
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeSStolerances, ARKodeSVtolerances, ARKodeWFtolerances:

  These functions specify the integration tolerances. One of them
  SHOULD be called before the first call to ARKodeEvolve; otherwise
  default values of reltol=1e-4 and abstol=1e-9 will be used,
  which may be entirely incorrect for a specific problem.

  ARKodeSStolerances specifies scalar relative and absolute
  tolerances.

  ARKodeSVtolerances specifies scalar relative tolerance and a
  vector absolute tolerance (a potentially different absolute
  tolerance for each vector component).

  ARKodeWFtolerances specifies a user-provides function (of type
  ARKEwtFn) which will be called to set the error weight vector.
  ---------------------------------------------------------------*/
pub fn ARKodeSStolerances(arkode_mem: &ARKodeMem, reltol: sunrealtype, abstol: sunrealtype) -> i32 {
    /* unpack ark_mem: NULL-mem check handled by the type system */
    let ark_mem = arkode_mem;

    /* Check inputs */
    if !ark_mem.borrow().MallocDone {
        arkProcessError(
            Some(ark_mem),
            ARK_NO_MALLOC,
            line!() as i32,
            "ARKodeSStolerances",
            file!(),
            MSG_ARK_NO_MALLOC,
        );
        return ARK_NO_MALLOC;
    }
    if reltol < ZERO {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "ARKodeSStolerances",
            file!(),
            MSG_ARK_BAD_RELTOL,
        );
        return ARK_ILL_INPUT;
    }
    if abstol < ZERO {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "ARKodeSStolerances",
            file!(),
            MSG_ARK_BAD_ABSTOL,
        );
        return ARK_ILL_INPUT;
    }

    /* Ensure that vector supports N_VAddConst */
    let has_nvaddconst = {
        let tempv1 = ark_mem.borrow().tempv1.clone().expect("tempv1 allocated");
        let has = tempv1.ops.borrow().nvaddconst.is_some();
        has
    };
    if !has_nvaddconst {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "ARKodeSStolerances",
            file!(),
            "N_VAddConst unimplemented (required for scalar abstol)",
        );
        return ARK_ILL_INPUT;
    }

    let mut m = ark_mem.borrow_mut();

    /* Set flag indicating whether abstol == 0 */
    m.atolmin0 = abstol == ZERO;

    /* Copy tolerances into memory */
    m.reltol = reltol;
    m.Sabstol = abstol;
    m.itol = ARK_SS;

    /* enforce use of arkEwtSetSS */
    m.user_efun = SUNFALSE;
    m.efun = Some(arkEwtSetSS);
    /* C: e_data = ark_mem -- the built-in error-weight function reaches
    the integrator through a boxed handle clone */
    m.e_data = Some(Box::new(ark_mem.clone()));

    ARK_SUCCESS
}

pub fn ARKodeSVtolerances(arkode_mem: &ARKodeMem, reltol: sunrealtype, abstol: &N_Vector) -> i32 {
    /* unpack ark_mem: NULL-mem check handled by the type system */
    let ark_mem = arkode_mem;

    /* Check inputs */
    if !ark_mem.borrow().MallocDone {
        arkProcessError(
            Some(ark_mem),
            ARK_NO_MALLOC,
            line!() as i32,
            "ARKodeSVtolerances",
            file!(),
            MSG_ARK_NO_MALLOC,
        );
        return ARK_NO_MALLOC;
    }
    if reltol < ZERO {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "ARKodeSVtolerances",
            file!(),
            MSG_ARK_BAD_RELTOL,
        );
        return ARK_ILL_INPUT;
    }
    /* NULL abstol check: handled by the type system */
    if abstol.ops.borrow().nvmin.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "ARKodeSVtolerances",
            file!(),
            "Missing N_VMin routine from N_Vector",
        );
        return ARK_ILL_INPUT;
    }
    let abstolmin = N_VMin(abstol);
    if abstolmin < ZERO {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "ARKodeSVtolerances",
            file!(),
            MSG_ARK_BAD_ABSTOL,
        );
        return ARK_ILL_INPUT;
    }

    /* Set flag indicating whether min(abstol) == 0 */
    ark_mem.borrow_mut().atolmin0 = abstolmin == ZERO;

    /* Copy tolerances into memory */
    if !ark_mem.borrow().VabstolMallocDone {
        let ewt = ark_mem.borrow().ewt.clone().expect("ewt allocated");
        let mut Vabstol = ark_mem.borrow_mut().Vabstol.take();
        let allocOK = arkAllocVec(ark_mem, &ewt, &mut Vabstol);
        ark_mem.borrow_mut().Vabstol = Vabstol;
        if !allocOK {
            arkProcessError(
                Some(ark_mem),
                ARK_MEM_FAIL,
                line!() as i32,
                "ARKodeSVtolerances",
                file!(),
                MSG_ARK_ARKMEM_FAIL,
            );
            return ARK_ILL_INPUT;
        }
        ark_mem.borrow_mut().VabstolMallocDone = SUNTRUE;
    }
    let Vabstol = ark_mem.borrow().Vabstol.clone().expect("Vabstol allocated");
    N_VScale(ONE, abstol, &Vabstol);

    let mut m = ark_mem.borrow_mut();
    m.reltol = reltol;
    m.itol = ARK_SV;

    /* enforce use of arkEwtSetSV */
    m.user_efun = SUNFALSE;
    m.efun = Some(arkEwtSetSV);
    /* C: e_data = ark_mem (see ARKodeSStolerances) */
    m.e_data = Some(Box::new(ark_mem.clone()));

    ARK_SUCCESS
}

pub fn ARKodeWFtolerances(arkode_mem: &ARKodeMem, efun: ARKEwtFn) -> i32 {
    /* unpack ark_mem: NULL-mem check handled by the type system */
    let ark_mem = arkode_mem;

    if !ark_mem.borrow().MallocDone {
        arkProcessError(
            Some(ark_mem),
            ARK_NO_MALLOC,
            line!() as i32,
            "ARKodeWFtolerances",
            file!(),
            MSG_ARK_NO_MALLOC,
        );
        return ARK_NO_MALLOC;
    }

    /* Copy tolerance data into memory */
    let mut m = ark_mem.borrow_mut();
    m.itol = ARK_WF;
    m.user_efun = SUNTRUE;
    m.efun = Some(efun);
    /* C: e_data = ark_mem->user_data -- a raw pointer snapshot that a
    `Box` cannot reproduce (accepted deviation class 6).  The token is
    cleared here and `ark_call_efun` passes the CURRENT `user_data` box
    whenever `user_efun` is set. */
    m.e_data = None;

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeResStolerance, ARKodeResVtolerance, ARKodeResFtolerance:

  These functions specify the absolute residual tolerance.
  Specification of the absolute residual tolerance is only
  necessary for problems with non-identity mass matrices in which
  the units of the solution vector y dramatically differ from the
  units of the ODE right-hand side f(t,y).  If this occurs, one
  of these routines SHOULD be called before the first call to
  ARKODE; otherwise the default value of rabstol=1e-9 will be
  used, which may be entirely incorrect for a specific problem.

  ARKodeResStolerances specifies a scalar residual tolerance.

  ARKodeResVtolerances specifies a vector residual tolerance
  (a potentially different absolute residual tolerance for
  each vector component).

  ARKodeResFtolerances specifies a user-provides function (of
  type ARKRwtFn) which will be called to set the residual
  weight vector.
  ---------------------------------------------------------------*/
pub fn ARKodeResStolerance(arkode_mem: &ARKodeMem, rabstol: sunrealtype) -> i32 {
    /* unpack ark_mem: NULL-mem check handled by the type system */
    let ark_mem = arkode_mem;

    /* Guard against use for time steppers that do not support mass matrices */
    if !ark_mem.borrow().step_supports_massmatrix {
        arkProcessError(
            Some(ark_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeResStolerance",
            file!(),
            "time-stepping module does not support non-identity mass matrices",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    /* Check inputs */
    if !ark_mem.borrow().MallocDone {
        arkProcessError(
            Some(ark_mem),
            ARK_NO_MALLOC,
            line!() as i32,
            "ARKodeResStolerance",
            file!(),
            MSG_ARK_NO_MALLOC,
        );
        return ARK_NO_MALLOC;
    }
    if rabstol < ZERO {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "ARKodeResStolerance",
            file!(),
            MSG_ARK_BAD_RABSTOL,
        );
        return ARK_ILL_INPUT;
    }

    /* Ensure that vector supports N_VAddConst */
    let has_nvaddconst = {
        let tempv1 = ark_mem.borrow().tempv1.clone().expect("tempv1 allocated");
        let has = tempv1.ops.borrow().nvaddconst.is_some();
        has
    };
    if !has_nvaddconst {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "ARKodeResStolerance",
            file!(),
            "N_VAddConst unimplemented (required for scalar rabstol)",
        );
        return ARK_ILL_INPUT;
    }

    /* Set flag indicating whether rabstol == 0 */
    ark_mem.borrow_mut().Ratolmin0 = rabstol == ZERO;

    /* Allocate space for rwt if necessary */
    if ark_mem.borrow().rwt_is_ewt {
        ark_mem.borrow_mut().rwt = None;
        let ewt = ark_mem.borrow().ewt.clone().expect("ewt allocated");
        let mut rwt = ark_mem.borrow_mut().rwt.take();
        let allocOK = arkAllocVec(ark_mem, &ewt, &mut rwt);
        ark_mem.borrow_mut().rwt = rwt;
        if !allocOK {
            arkProcessError(
                Some(ark_mem),
                ARK_MEM_FAIL,
                line!() as i32,
                "ARKodeResStolerance",
                file!(),
                MSG_ARK_ARKMEM_FAIL,
            );
            return ARK_ILL_INPUT;
        }
        ark_mem.borrow_mut().rwt_is_ewt = SUNFALSE;
    }

    /* Copy tolerances into memory */
    let mut m = ark_mem.borrow_mut();
    m.SRabstol = rabstol;
    m.ritol = ARK_SS;

    /* enforce use of arkRwtSet
       (upstream really does clear `user_efun` and not `user_rfun` here --
       preserved verbatim) */
    m.user_efun = SUNFALSE;
    m.rfun = Some(arkRwtSet);
    /* C: r_data = ark_mem */
    m.r_data = Some(Box::new(ark_mem.clone()));

    ARK_SUCCESS
}

pub fn ARKodeResVtolerance(arkode_mem: &ARKodeMem, rabstol: &N_Vector) -> i32 {
    /* unpack ark_mem: NULL-mem check handled by the type system */
    let ark_mem = arkode_mem;

    /* Guard against use for time steppers that do not support mass matrices */
    if !ark_mem.borrow().step_supports_massmatrix {
        arkProcessError(
            Some(ark_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeResVtolerance",
            file!(),
            "time-stepping module does not support non-identity mass matrices",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    /* Check inputs */
    if !ark_mem.borrow().MallocDone {
        arkProcessError(
            Some(ark_mem),
            ARK_NO_MALLOC,
            line!() as i32,
            "ARKodeResVtolerance",
            file!(),
            MSG_ARK_NO_MALLOC,
        );
        return ARK_NO_MALLOC;
    }
    /* NULL rabstol check: handled by the type system */
    if rabstol.ops.borrow().nvmin.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "ARKodeResVtolerance",
            file!(),
            "Missing N_VMin routine from N_Vector",
        );
        return ARK_ILL_INPUT;
    }
    let rabstolmin = N_VMin(rabstol);
    if rabstolmin < ZERO {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "ARKodeResVtolerance",
            file!(),
            MSG_ARK_BAD_RABSTOL,
        );
        return ARK_ILL_INPUT;
    }

    /* Set flag indicating whether min(abstol) == 0 */
    ark_mem.borrow_mut().Ratolmin0 = rabstolmin == ZERO;

    /* Allocate space for rwt if necessary */
    if ark_mem.borrow().rwt_is_ewt {
        ark_mem.borrow_mut().rwt = None;
        let ewt = ark_mem.borrow().ewt.clone().expect("ewt allocated");
        let mut rwt = ark_mem.borrow_mut().rwt.take();
        let allocOK = arkAllocVec(ark_mem, &ewt, &mut rwt);
        ark_mem.borrow_mut().rwt = rwt;
        if !allocOK {
            arkProcessError(
                Some(ark_mem),
                ARK_MEM_FAIL,
                line!() as i32,
                "ARKodeResVtolerance",
                file!(),
                MSG_ARK_ARKMEM_FAIL,
            );
            return ARK_ILL_INPUT;
        }
        ark_mem.borrow_mut().rwt_is_ewt = SUNFALSE;
    }

    /* Copy tolerances into memory */
    if !ark_mem.borrow().VRabstolMallocDone {
        let rwt = ark_mem.borrow().rwt.clone().expect("rwt allocated");
        let mut VRabstol = ark_mem.borrow_mut().VRabstol.take();
        let allocOK = arkAllocVec(ark_mem, &rwt, &mut VRabstol);
        ark_mem.borrow_mut().VRabstol = VRabstol;
        if !allocOK {
            arkProcessError(
                Some(ark_mem),
                ARK_MEM_FAIL,
                line!() as i32,
                "ARKodeResVtolerance",
                file!(),
                MSG_ARK_ARKMEM_FAIL,
            );
            return ARK_ILL_INPUT;
        }
        ark_mem.borrow_mut().VRabstolMallocDone = SUNTRUE;
    }
    let VRabstol = ark_mem
        .borrow()
        .VRabstol
        .clone()
        .expect("VRabstol allocated");
    N_VScale(ONE, rabstol, &VRabstol);

    let mut m = ark_mem.borrow_mut();
    m.ritol = ARK_SV;

    /* enforce use of arkRwtSet (see the note in ARKodeResStolerance) */
    m.user_efun = SUNFALSE;
    m.rfun = Some(arkRwtSet);
    /* C: r_data = ark_mem */
    m.r_data = Some(Box::new(ark_mem.clone()));

    ARK_SUCCESS
}

pub fn ARKodeResFtolerance(arkode_mem: &ARKodeMem, rfun: ARKRwtFn) -> i32 {
    /* unpack ark_mem: NULL-mem check handled by the type system */
    let ark_mem = arkode_mem;

    /* Guard against use for time steppers that do not support mass matrices */
    if !ark_mem.borrow().step_supports_massmatrix {
        arkProcessError(
            Some(ark_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeResFtolerance",
            file!(),
            "time-stepping module does not support non-identity mass matrices",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    if !ark_mem.borrow().MallocDone {
        arkProcessError(
            Some(ark_mem),
            ARK_NO_MALLOC,
            line!() as i32,
            "ARKodeResFtolerance",
            file!(),
            MSG_ARK_NO_MALLOC,
        );
        return ARK_NO_MALLOC;
    }

    /* Allocate space for rwt if necessary */
    if ark_mem.borrow().rwt_is_ewt {
        ark_mem.borrow_mut().rwt = None;
        let ewt = ark_mem.borrow().ewt.clone().expect("ewt allocated");
        let mut rwt = ark_mem.borrow_mut().rwt.take();
        let allocOK = arkAllocVec(ark_mem, &ewt, &mut rwt);
        ark_mem.borrow_mut().rwt = rwt;
        if !allocOK {
            arkProcessError(
                Some(ark_mem),
                ARK_MEM_FAIL,
                line!() as i32,
                "ARKodeResFtolerance",
                file!(),
                MSG_ARK_ARKMEM_FAIL,
            );
            return ARK_ILL_INPUT;
        }
        ark_mem.borrow_mut().rwt_is_ewt = SUNFALSE;
    }

    /* Copy tolerance data into memory */
    let mut m = ark_mem.borrow_mut();
    m.ritol = ARK_WF;
    m.user_rfun = SUNTRUE;
    m.rfun = Some(rfun);
    /* C: r_data = ark_mem->user_data -- pointer snapshot, see
    ARKodeWFtolerances (accepted deviation class 6) */
    m.r_data = None;

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeEvolve:

  This routine is the main driver of ARKODE-based integrators.

  It integrates over a time interval defined by the user, by
  calling the time step module to do internal time steps.

  The first time that ARKodeEvolve is called for a successfully
  initialized problem, it computes a tentative initial step size.

  ARKodeEvolve supports two modes as specified by itask: ARK_NORMAL and
  ARK_ONE_STEP.  In the ARK_NORMAL mode, the solver steps until
  it reaches or passes tout and then interpolates to obtain
  y(tout).  In the ARK_ONE_STEP mode, it takes one internal step
  and returns.  The behavior of both modes can be over-rided
  through user-specification of ark_tstop (through the
  *StepSetStopTime function), in which case if a solver step
  would pass tstop, the step is shortened so that it stops at
  exactly the specified stop time, and hence interpolation of
  y(tout) is not required.
  ---------------------------------------------------------------*/
pub fn ARKodeEvolve(
    arkode_mem: &ARKodeMem,
    tout: sunrealtype,
    yout: &N_Vector,
    tret: &mut sunrealtype,
    itask: i32,
) -> i32 {
    /* C leaves `istate` uninitialized; every path that leaves the internal
    step loop assigns it exactly once before its `break`, so the Rust
    declaration is likewise deferred-initialization */
    let istate: i32;

    /* Check and process inputs */

    /* Check if ark_mem exists: handled by the type system */
    let ark_mem = arkode_mem;

    /* Check if ark_mem was allocated */
    if !ark_mem.borrow().MallocDone {
        arkProcessError(
            Some(ark_mem),
            ARK_NO_MALLOC,
            line!() as i32,
            "ARKodeEvolve",
            file!(),
            MSG_ARK_NO_MALLOC,
        );
        return ARK_NO_MALLOC;
    }

    /* Check for yout != NULL (handled by the type system; ark_mem->ycur
    aliases the user's yout -- the Rc clone shares the underlying data
    exactly as the C pointer copy does) */
    ark_mem.borrow_mut().ycur = Some(yout.clone());

    /* Check for tret != NULL: handled by the type system */

    /* Check for valid itask */
    if (itask != ARK_NORMAL) && (itask != ARK_ONE_STEP) {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "ARKodeEvolve",
            file!(),
            MSG_ARK_BAD_ITASK,
        );
        return ARK_ILL_INPUT;
    }

    /* start profiler: profiling disabled in the reference build */

    /* perform first-step-specific initializations:
       - initialize tret values to initialization time
       - perform initial integrator setup  */
    if ark_mem.borrow().initsetup {
        {
            let mut m = ark_mem.borrow_mut();
            m.tretlast = m.tcur;
            *tret = m.tcur;
        }
        let retval = arkInitialSetup(ark_mem, tout);
        if retval != ARK_SUCCESS {
            return retval;
        }
    }

    /* perform stopping tests */
    if !ark_mem.borrow().initsetup {
        let mut retval: i32 = ARK_SUCCESS;
        if arkStopTests(ark_mem, tout, yout, tret, itask, &mut retval) != 0 {
            return retval;
        }
    }

    /* fill current independent variable (and optionally ycur with yn) */
    {
        let mut m = ark_mem.borrow_mut();
        m.tcur = m.tn;
    }
    {
        let (ensure_ycur, yn, ycur) = {
            let m = ark_mem.borrow();
            (m.ensure_ycur, m.yn.clone(), m.ycur.clone())
        };
        if ensure_ycur {
            N_VScale(
                ONE,
                &yn.expect("yn allocated"),
                &ycur.expect("ycur attached"),
            );
        }
    }

    /*--------------------------------------------------
      Looping point for successful internal steps

      - update the ewt/rwt vectors for upcoming step
      - check for errors (too many steps, too much
        accuracy requested, step size too small)
      - loop over attempts at a new step:
        * try to take step (via time stepper module),
          handle solver convergence or other failures
        * if the stepper requests ARK_RETRY_STEP, we
          retry the step without accumulating failures.
          A stepper should never request this multiple
          times in a row.
        * perform constraint-handling (if selected)
        * check temporal error
        * if all of the above pass, complete step by
          updating current time, solution, error &
          stepsize history arrays.
      - perform stop tests:
        * check for root in last step taken
        * check if tout was passed
        * check if close to tstop
        * check if in ONE_STEP mode (must return)
      --------------------------------------------------*/
    let mut nstloc: i64 = 0;
    loop {
        {
            let mut m = ark_mem.borrow_mut();
            m.next_h = m.h;
        }

        /* Reset and check ewt and rwt */
        if !ark_mem.borrow().initsetup {
            let (yn, ewt) = {
                let m = ark_mem.borrow();
                (
                    m.yn.clone().expect("yn allocated"),
                    m.ewt.clone().expect("ewt allocated"),
                )
            };
            let ewtsetOK = ark_call_efun(ark_mem, &yn, &ewt);
            if ewtsetOK != 0 {
                let (itol, tcur) = {
                    let m = ark_mem.borrow();
                    (m.itol, m.tcur)
                };
                if itol == ARK_WF {
                    arkProcessError(
                        Some(ark_mem),
                        ARK_ILL_INPUT,
                        line!() as i32,
                        "ARKodeEvolve",
                        file!(),
                        &MSG_ARK_EWT_NOW_FAIL(tcur),
                    );
                } else {
                    arkProcessError(
                        Some(ark_mem),
                        ARK_ILL_INPUT,
                        line!() as i32,
                        "ARKodeEvolve",
                        file!(),
                        &MSG_ARK_EWT_NOW_BAD(tcur),
                    );
                }

                istate = ARK_ILL_INPUT;
                ark_mem.borrow_mut().tretlast = tcur;
                *tret = tcur;
                N_VScale(ONE, &yn, yout);
                break;
            }

            if !ark_mem.borrow().rwt_is_ewt {
                let rwt = ark_mem.borrow().rwt.clone().expect("rwt allocated");
                let ewtsetOK = ark_call_rfun(ark_mem, &yn, &rwt);
                if ewtsetOK != 0 {
                    let (itol, tcur) = {
                        let m = ark_mem.borrow();
                        (m.itol, m.tcur)
                    };
                    if itol == ARK_WF {
                        arkProcessError(
                            Some(ark_mem),
                            ARK_ILL_INPUT,
                            line!() as i32,
                            "ARKodeEvolve",
                            file!(),
                            &MSG_ARK_RWT_NOW_FAIL(tcur),
                        );
                    } else {
                        arkProcessError(
                            Some(ark_mem),
                            ARK_ILL_INPUT,
                            line!() as i32,
                            "ARKodeEvolve",
                            file!(),
                            &MSG_ARK_RWT_NOW_BAD(tcur),
                        );
                    }

                    istate = ARK_ILL_INPUT;
                    ark_mem.borrow_mut().tretlast = tcur;
                    *tret = tcur;
                    N_VScale(ONE, &yn, yout);
                    break;
                }
            }
        }

        /* Check for too many steps */
        {
            let (mxstep, tcur) = {
                let m = ark_mem.borrow();
                (m.mxstep, m.tcur)
            };
            if (mxstep > 0) && (nstloc >= mxstep) {
                arkProcessError(
                    Some(ark_mem),
                    ARK_TOO_MUCH_WORK,
                    line!() as i32,
                    "ARKodeEvolve",
                    file!(),
                    &MSG_ARK_MAX_STEPS(tcur),
                );
                istate = ARK_TOO_MUCH_WORK;
                ark_mem.borrow_mut().tretlast = tcur;
                *tret = tcur;
                let yn = ark_mem.borrow().yn.clone().expect("yn allocated");
                N_VScale(ONE, &yn, yout);
                break;
            }
        }

        /* Check for too much accuracy requested */
        {
            let (yn, ewt, uround) = {
                let m = ark_mem.borrow();
                (
                    m.yn.clone().expect("yn allocated"),
                    m.ewt.clone().expect("ewt allocated"),
                    m.uround,
                )
            };
            let nrm = N_VWrmsNorm(&yn, &ewt);
            ark_mem.borrow_mut().tolsf = uround * nrm;
            let (tolsf, fixedstep, tcur) = {
                let m = ark_mem.borrow();
                (m.tolsf, m.fixedstep, m.tcur)
            };
            if tolsf > ONE && !fixedstep {
                arkProcessError(
                    Some(ark_mem),
                    ARK_TOO_MUCH_ACC,
                    line!() as i32,
                    "ARKodeEvolve",
                    file!(),
                    &MSG_ARK_TOO_MUCH_ACC(tcur),
                );
                istate = ARK_TOO_MUCH_ACC;
                ark_mem.borrow_mut().tretlast = tcur;
                *tret = tcur;
                N_VScale(ONE, &yn, yout);
                ark_mem.borrow_mut().tolsf *= TWO;
                break;
            } else {
                ark_mem.borrow_mut().tolsf = ONE;
            }
        }

        /* Check for h below roundoff level in tn */
        {
            let (tcur, h) = {
                let m = ark_mem.borrow();
                (m.tcur, m.h)
            };
            if tcur + h == tcur {
                ark_mem.borrow_mut().nhnil += 1;
                let (nhnil, mxhnil) = {
                    let m = ark_mem.borrow();
                    (m.nhnil, m.mxhnil)
                };
                if nhnil <= mxhnil {
                    arkProcessError(
                        Some(ark_mem),
                        ARK_WARNING,
                        line!() as i32,
                        "ARKodeEvolve",
                        file!(),
                        &MSG_ARK_HNIL(tcur, h),
                    );
                }
                if nhnil == mxhnil {
                    arkProcessError(
                        Some(ark_mem),
                        ARK_WARNING,
                        line!() as i32,
                        "ARKodeEvolve",
                        file!(),
                        MSG_ARK_HNIL_DONE,
                    );
                }
            }
        }

        /* Update parameter for upcoming step size */
        {
            let mut m = ark_mem.borrow_mut();
            if m.hprime != m.h {
                m.h = m.h * m.eta;
                m.next_h = m.h;
            }
            if m.fixedstep {
                m.h = m.hin;
                m.next_h = m.h;

                /* patch for 'fixedstep' + 'tstop' use case:
                   limit fixed step size if step would overtake tstop */
                if m.tstopset && (m.tcur + m.h - m.tstop) * m.h > ZERO {
                    m.h = (m.tstop - m.tcur) * (ONE - FOUR * m.uround);
                }
            }
        }

        /* Looping point for step attempts */
        let mut dsm: sunrealtype = ZERO;
        let mut kflag: i32 = ARK_SUCCESS;
        let mut relax_fails: i32 = 0;
        let mut nflag: i32 = FIRST_CALL;
        let mut attempts: i32 = 0;
        let mut ncf: i32 = 0;
        let mut nef: i32 = 0;
        let mut constrfails: i32 = 0;
        ark_mem.borrow_mut().last_kflag = 0;
        loop {
            /* increment attempt counters
               Note: kflag can only equal ARK_RETRY_STEP if the stepper rejected
               the current step size before performing calculations. Thus, we do
               not include those when keeping track of step "attempts". */
            if kflag != ARK_RETRY_STEP {
                attempts += 1;
                ark_mem.borrow_mut().nst_attempts += 1;
            }

            /* fill tcur with the last accepted step time */
            {
                let mut m = ark_mem.borrow_mut();
                m.tcur = m.tn;
            }

            /* call the user-supplied pre-step function (if it exists) */
            if ark_mem.borrow().PreStepFn.is_some() {
                let (ensure_ycur, tcur, nst, ycur, yn) = {
                    let m = ark_mem.borrow();
                    (m.ensure_ycur, m.tcur, m.nst, m.ycur.clone(), m.yn.clone())
                };
                let retval = if ensure_ycur {
                    ark_call_prestepfn(
                        ark_mem,
                        tcur,
                        &ycur.expect("ycur attached"),
                        nst,
                        attempts,
                    )
                } else {
                    ark_call_prestepfn(ark_mem, tcur, &yn.expect("yn allocated"), nst, attempts)
                };
                if retval != 0 {
                    return ARK_PRESTEPFN_FAIL;
                }
            }

            /* Call time stepper module to attempt a step:
                  0 => step completed successfully
                 >0 => step encountered recoverable failure; reduce step if possible
                 <0 => step encountered unrecoverable failure */
            let step = ark_mem.borrow().step.expect("step set");
            kflag = step(ark_mem, &mut dsm, &mut nflag);
            if kflag < 0 {
                break;
            }

            /* handle solver convergence failures */
            kflag = arkCheckConvergence(ark_mem, &mut nflag, &mut ncf);

            if kflag < 0 {
                break;
            }

            /* Perform relaxation:
                 - computes relaxation parameter
                 - on success, updates ycur, h, and dsm
                 - on recoverable failure, updates eta and signals to retry step
                 - on fatal error, returns negative error flag */
            if ark_mem.borrow().relax_enabled && (kflag == ARK_SUCCESS) {
                kflag = arkRelax(ark_mem, &mut relax_fails, &mut dsm);

                if kflag < 0 {
                    break;
                }
            }

            /* perform constraint-handling (if selected, and if solver check passed) */
            if ark_mem.borrow().constraints.is_some() && (kflag == ARK_SUCCESS) {
                kflag = arkCheckConstraints(ark_mem, &mut constrfails, &mut nflag);

                if kflag < 0 {
                    break;
                }
            }

            /* when fixed time-stepping is enabled, 'success' == successful stage solves
               (checked in previous block), so just enforce no step size change */
            if ark_mem.borrow().fixedstep {
                ark_mem.borrow_mut().eta = ONE;
                break;
            }

            /* check temporal error (if checks above passed) */
            if kflag == ARK_SUCCESS {
                kflag = arkCheckTemporalError(ark_mem, &mut nflag, &mut nef, dsm);

                if kflag < 0 {
                    break;
                }
            }

            /* if ignoring temporal error test result (XBraid) force step to pass */
            if ark_mem.borrow().force_pass {
                ark_mem.borrow_mut().last_kflag = kflag;
                kflag = ARK_SUCCESS;
                break;
            }

            /* break attempt loop on successful step */
            if kflag == ARK_SUCCESS {
                break;
            }

            /* unsuccessful step, if |h| = hmin, return ARK_ERR_FAILURE */
            {
                let m = ark_mem.borrow();
                if SUNRabs(m.h) <= m.hmin * ONEPSM {
                    return ARK_ERR_FAILURE;
                }
            }

            /* update h, hprime and next_h for next iteration */
            {
                let mut m = ark_mem.borrow_mut();
                m.h *= m.eta;
                m.hprime = m.h;
                m.next_h = m.hprime;

                /* reset tcur to last saved internal time before reattempting step
                   (and optionally ycur to yn ) */
                m.tcur = m.tn;
            }
            {
                let (ensure_ycur, yn, ycur) = {
                    let m = ark_mem.borrow();
                    (m.ensure_ycur, m.yn.clone(), m.ycur.clone())
                };
                if ensure_ycur {
                    N_VScale(
                        ONE,
                        &yn.expect("yn allocated"),
                        &ycur.expect("ycur attached"),
                    );
                }
            }
        } /* end looping for step attempts */

        /* If step attempt loop succeeded, complete step (update current time, solution,
           error stepsize history arrays; call user-supplied step postprocessing function)
           (added stuff from arkStep_PrepareNextStep -- revisit) */
        if kflag == ARK_SUCCESS {
            kflag = arkCompleteStep(ark_mem, dsm);
        }

        /* If step attempt loop failed, process flag and return to user */
        if kflag != ARK_SUCCESS {
            istate = arkHandleFailure(ark_mem, kflag);
            let (tcur, yn) = {
                let mut m = ark_mem.borrow_mut();
                m.tretlast = m.tcur;
                (m.tcur, m.yn.clone().expect("yn allocated"))
            };
            *tret = tcur;
            N_VScale(ONE, &yn, yout);
            break;
        }

        nstloc += 1;

        /* Check for root in last step taken. */
        if ark_mem.borrow().root_mem.is_some() {
            let nrtfn = ark_mem.borrow().root_mem.as_ref().expect("root_mem").nrtfn;
            if nrtfn > 0 {
                let retval = arkRootCheck3(ark_mem, tout, itask);
                if retval == RTFOUND {
                    /* A new root was found */
                    let tlo = {
                        let mut m = ark_mem.borrow_mut();
                        let root_mem = m.root_mem.as_mut().expect("root_mem");
                        root_mem.irfnd = 1;
                        root_mem.tlo
                    };
                    istate = ARK_ROOT_RETURN;
                    ark_mem.borrow_mut().tretlast = tlo;
                    *tret = tlo;
                    break;
                } else if retval == ARK_RTFUNC_FAIL {
                    /* g failed */
                    let tlo = ark_mem.borrow().root_mem.as_ref().expect("root_mem").tlo;
                    arkProcessError(
                        Some(ark_mem),
                        ARK_RTFUNC_FAIL,
                        line!() as i32,
                        "ARKodeEvolve",
                        file!(),
                        &MSG_ARK_RTFUNC_FAILED(tlo),
                    );
                    istate = ARK_RTFUNC_FAIL;
                    break;
                }

                /* If we are at the end of the first step and we still have
                   some event functions that are inactive, issue a warning
                   as this may indicate a user error in the implementation
                   of the root function. */
                if ark_mem.borrow().nst == 1 {
                    let (inactive_roots, mxgnull) = {
                        let m = ark_mem.borrow();
                        let root_mem = m.root_mem.as_ref().expect("root_mem");
                        let mut inactive_roots = SUNFALSE;
                        for ir in 0..root_mem.nrtfn as usize {
                            if !root_mem.gactive[ir] {
                                inactive_roots = SUNTRUE;
                                break;
                            }
                        }
                        (inactive_roots, root_mem.mxgnull)
                    };
                    if (mxgnull > 0) && inactive_roots {
                        arkProcessError(
                            Some(ark_mem),
                            ARK_WARNING,
                            line!() as i32,
                            "ARKodeEvolve",
                            file!(),
                            MSG_ARK_INACTIVE_ROOTS,
                        );
                    }
                }
            }
        }

        /* Check if tn is at tstop or near tstop */
        if ark_mem.borrow().tstopset {
            let (tcur, h, hprime, tstop, tstopinterp, has_interp, troundoff) = {
                let m = ark_mem.borrow();
                (
                    m.tcur,
                    m.h,
                    m.hprime,
                    m.tstop,
                    m.tstopinterp,
                    m.interp.is_some(),
                    FUZZ_FACTOR * m.uround * (SUNRabs(m.tcur) + SUNRabs(m.h)),
                )
            };

            if SUNRabs(tcur - tstop) <= troundoff {
                /* Ensure tout >= tstop, otherwise check for tout return below */
                if (tout - tstop) * h >= ZERO || SUNRabs(tout - tstop) <= troundoff {
                    if tstopinterp && has_interp {
                        let retval = ARKodeGetDky(ark_mem, tstop, 0, yout);
                        if retval != ARK_SUCCESS {
                            arkProcessError(
                                Some(ark_mem),
                                retval,
                                line!() as i32,
                                "ARKodeEvolve",
                                file!(),
                                &MSG_ARK_INTERPOLATION_FAIL(tstop),
                            );
                            istate = retval;
                            break;
                        }
                    } else {
                        let yn = ark_mem.borrow().yn.clone().expect("yn allocated");
                        N_VScale(ONE, &yn, yout);
                    }
                    {
                        let mut m = ark_mem.borrow_mut();
                        m.tretlast = m.tstop;
                        *tret = m.tstop;
                        m.tstopset = SUNFALSE;
                    }
                    istate = ARK_TSTOP_RETURN;
                    break;
                }
            }
            /* limit upcoming step if it will overcome tstop */
            else if (tcur + hprime - tstop) * h > ZERO {
                let mut m = ark_mem.borrow_mut();
                m.hprime = (m.tstop - m.tcur) * (ONE - FOUR * m.uround);
                m.eta = m.hprime / m.h;
            }
        }

        /* In NORMAL mode, check if tout reached */
        {
            let (tcur, h, has_interp) = {
                let m = ark_mem.borrow();
                (m.tcur, m.h, m.interp.is_some())
            };
            if (itask == ARK_NORMAL) && (tcur - tout) * h >= ZERO {
                if has_interp {
                    let retval = ARKodeGetDky(ark_mem, tout, 0, yout);
                    if retval != ARK_SUCCESS {
                        arkProcessError(
                            Some(ark_mem),
                            retval,
                            line!() as i32,
                            "ARKodeEvolve",
                            file!(),
                            &MSG_ARK_INTERPOLATION_FAIL(tout),
                        );
                        istate = retval;
                        break;
                    }
                    ark_mem.borrow_mut().tretlast = tout;
                    *tret = tout;
                } else {
                    let yn = ark_mem.borrow().yn.clone().expect("yn allocated");
                    N_VScale(ONE, &yn, yout);
                    let mut m = ark_mem.borrow_mut();
                    m.tretlast = m.tcur;
                    *tret = m.tcur;
                }
                {
                    let mut m = ark_mem.borrow_mut();
                    m.next_h = m.hprime;
                }
                istate = ARK_SUCCESS;
                break;
            }
        }

        /* In ONE_STEP mode, exit loop (arkCompleteStep already copied yn to ycur, an alias to yout) */
        if itask == ARK_ONE_STEP {
            istate = ARK_SUCCESS;
            let mut m = ark_mem.borrow_mut();
            m.tretlast = m.tcur;
            *tret = m.tcur;
            m.next_h = m.hprime;
            break;
        }
    } /* end looping for internal steps */

    /* stop profiler and return: profiling disabled in the reference build */
    istate
}

/*---------------------------------------------------------------
  ARKodeGetDky:

  This routine computes the k-th derivative of the interpolating
  polynomial at the time t and stores the result in the vector
  dky. This routine internally calls arkInterpEvaluate to perform
  the interpolation.  We have the restriction that 0 <= k <= 3.
  This routine uses an interpolating polynomial of degree
  max(deg, k), i.e. it will form a polynomial of the degree
  available by the interpolation module and/or requested by
  the user through deg, unless higher-order derivatives are
  requested.

  This function is called by ARKodeEvolve with k=0 and t=tout to
  perform interpolation of outputs, but may also be called
  indirectly by the user via time step module *StepGetDky calls.
  Note: in all cases it will be called after ark_tcur has been
  updated to correspond with the end time of the last successful
  step.
  ---------------------------------------------------------------*/
pub fn ARKodeGetDky(arkode_mem: &ARKodeMem, t: sunrealtype, k: i32, dky: &N_Vector) -> i32 {
    /* Check if ark_mem exists: handled by the type system */
    let ark_mem = arkode_mem;

    /* Check all inputs for legality (NULL dky handled by the type system) */
    let interp = ark_mem.borrow().interp.clone();
    let interp = match interp {
        None => {
            arkProcessError(
                Some(ark_mem),
                ARK_MEM_NULL,
                line!() as i32,
                "ARKodeGetDky",
                file!(),
                "Missing interpolation structure",
            );
            return ARK_MEM_NULL;
        }
        Some(interp) => interp,
    };

    /* Allow for some slack */
    let (tcur, hold, h, uround) = {
        let m = ark_mem.borrow();
        (m.tcur, m.hold, m.h, m.uround)
    };
    let mut tfuzz = FUZZ_FACTOR * uround * (SUNRabs(tcur) + SUNRabs(hold));
    if hold < ZERO {
        tfuzz = -tfuzz;
    }
    let tp = tcur - hold - tfuzz;
    let tn1 = tcur + tfuzz;
    if (t - tp) * (t - tn1) > ZERO {
        arkProcessError(
            Some(ark_mem),
            ARK_BAD_T,
            line!() as i32,
            "ARKodeGetDky",
            file!(),
            &MSG_ARK_BAD_T(t, tcur - hold, tcur),
        );
        return ARK_BAD_T;
    }

    /* call arkInterpEvaluate to evaluate result */
    let s = (t - tcur) / h;
    let retval = arkInterpEvaluate(ark_mem, Some(&interp), s, k, ARK_INTERP_MAX_DEGREE, dky);
    if retval != ARK_SUCCESS {
        arkProcessError(
            Some(ark_mem),
            retval,
            line!() as i32,
            "ARKodeGetDky",
            file!(),
            "Error calling arkInterpEvaluate",
        );
        return retval;
    }
    ARK_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeFree:

  This routine frees the ARKODE infrastructure memory.
  ---------------------------------------------------------------*/
pub fn ARKodeFree(arkode_mem: &mut Option<ARKodeMem>) {
    if arkode_mem.is_none() {
        return;
    }

    let ark_mem = arkode_mem.as_ref().expect("arkode_mem").clone();

    /* free the time-stepper module memory (if provided) */
    let step_free = ark_mem.borrow().step_free;
    if let Some(step_free) = step_free {
        step_free(&ark_mem);
    }

    /* free vector storage */
    arkFreeVectors(&ark_mem);

    /* free the time step adaptivity module */
    if ark_mem.borrow().hadapt_mem.is_some() {
        let owncontroller = ark_mem
            .borrow()
            .hadapt_mem
            .as_ref()
            .expect("hadapt_mem")
            .owncontroller;
        if owncontroller {
            let hcontroller = ark_mem
                .borrow_mut()
                .hadapt_mem
                .as_mut()
                .expect("hadapt_mem")
                .hcontroller
                .take();
            let _ = SUNAdaptController_Destroy(hcontroller);
            ark_mem
                .borrow_mut()
                .hadapt_mem
                .as_mut()
                .expect("hadapt_mem")
                .owncontroller = SUNFALSE;
        }
        ark_mem.borrow_mut().hadapt_mem = None;
    }

    /* free the interpolation module */
    let interp = ark_mem.borrow().interp.clone();
    if let Some(interp) = interp {
        arkInterpFree(&ark_mem, Some(&interp));
        ark_mem.borrow_mut().interp = None;
    }

    /* free the root-finding module */
    if ark_mem.borrow().root_mem.is_some() {
        let _ = arkRootFree(&ark_mem);
        ark_mem.borrow_mut().root_mem = None;
    }

    /* free the relaxation module */
    if ark_mem.borrow().relax_mem.is_some() {
        let relax_mem = ark_mem.borrow_mut().relax_mem.take();
        let _ = arkRelaxDestroy(relax_mem);
    }

    /* SUNDIALS_ENABLE_PYTHON is not defined:
       arkode_user_supplied_fn_table_destroy(ark_mem->python) is not called */
    ark_mem.borrow_mut().python = None;

    /* C frees the mem struct wholesale; the Rust handle is dropped by the
    caller, so break the Rc cycles the built-in ewt/rwt data tokens create
    (e_data / r_data hold an ARKodeMem clone pointing back at this record) */
    {
        let mut m = ark_mem.borrow_mut();
        m.e_data = None;
        m.r_data = None;
    }

    *arkode_mem = None;
}

/*---------------------------------------------------------------
  ARKodePrintMem:

  This routine outputs the ark_mem structure to a specified file
  pointer.
  ---------------------------------------------------------------*/
pub fn ARKodePrintMem(arkode_mem: &ARKodeMem, outfile: &SUNFile) {
    /* Check if ark_mem exists: handled by the type system */
    let ark_mem = arkode_mem;

    /* if outfile==NULL, set it to stdout */
    let stdout_ = SUNFile::Stdout;
    let outfile = if outfile.is_null() { &stdout_ } else { outfile };

    {
        let m = ark_mem.borrow();

        /* output general values */
        outfile.write_str(&format!("itol = {}\n", m.itol));
        outfile.write_str(&format!("ritol = {}\n", m.ritol));
        outfile.write_str(&format!("mxhnil = {}\n", m.mxhnil));
        outfile.write_str(&format!("mxstep = {}\n", m.mxstep));
        outfile.write_str(&format!("lrw1 = {}\n", m.lrw1));
        outfile.write_str(&format!("liw1 = {}\n", m.liw1));
        outfile.write_str(&format!("lrw = {}\n", m.lrw));
        outfile.write_str(&format!("liw = {}\n", m.liw));
        outfile.write_str(&format!("user_efun = {}\n", m.user_efun as i32));
        outfile.write_str(&format!("tstopset = {}\n", m.tstopset as i32));
        outfile.write_str(&format!("tstopinterp = {}\n", m.tstopinterp as i32));
        outfile.write_str(&format!("tstop = {}\n", sun_format_g(m.tstop)));
        outfile.write_str(&format!(
            "VabstolMallocDone = {}\n",
            m.VabstolMallocDone as i32
        ));
        outfile.write_str(&format!("MallocDone = {}\n", m.MallocDone as i32));
        outfile.write_str(&format!("initsetup = {}\n", m.initsetup as i32));
        outfile.write_str(&format!("init_type = {}\n", m.init_type));
        outfile.write_str(&format!("firststage = {}\n", m.firststage as i32));
        outfile.write_str(&format!("uround = {}\n", sun_format_g(m.uround)));
        outfile.write_str(&format!("reltol = {}\n", sun_format_g(m.reltol)));
        outfile.write_str(&format!("Sabstol = {}\n", sun_format_g(m.Sabstol)));
        outfile.write_str(&format!("fixedstep = {}\n", m.fixedstep as i32));
        outfile.write_str(&format!("tolsf = {}\n", sun_format_g(m.tolsf)));
        outfile.write_str(&format!("call_fullrhs = {}\n", m.call_fullrhs as i32));
        outfile.write_str(&format!("do_adjoint = {}\n", m.do_adjoint as i32));
        outfile.write_str(&format!("ensure_ycur = {}\n", m.ensure_ycur as i32));

        /* output counters */
        outfile.write_str(&format!("nhnil = {}\n", m.nhnil));
        outfile.write_str(&format!("nst_attempts = {}\n", m.nst_attempts));
        outfile.write_str(&format!("nst = {}\n", m.nst));
        outfile.write_str(&format!("ncfn = {}\n", m.ncfn));
        outfile.write_str(&format!("netf = {}\n", m.netf));

        /* output time-stepping values */
        outfile.write_str(&format!("hin = {}\n", sun_format_g(m.hin)));
        outfile.write_str(&format!("h = {}\n", sun_format_g(m.h)));
        outfile.write_str(&format!("hprime = {}\n", sun_format_g(m.hprime)));
        outfile.write_str(&format!("next_h = {}\n", sun_format_g(m.next_h)));
        outfile.write_str(&format!("eta = {}\n", sun_format_g(m.eta)));
        outfile.write_str(&format!("tcur = {}\n", sun_format_g(m.tcur)));
        outfile.write_str(&format!("tretlast = {}\n", sun_format_g(m.tretlast)));
        outfile.write_str(&format!("hmin = {}\n", sun_format_g(m.hmin)));
        outfile.write_str(&format!("hmax_inv = {}\n", sun_format_g(m.hmax_inv)));
        outfile.write_str(&format!("h0u = {}\n", sun_format_g(m.h0u)));
        outfile.write_str(&format!("tn = {}\n", sun_format_g(m.tn)));
        outfile.write_str(&format!("hold = {}\n", sun_format_g(m.hold)));
        outfile.write_str(&format!("maxnef = {}\n", m.maxnef));
        outfile.write_str(&format!("maxncf = {}\n", m.maxncf));

        /* output time-stepping adaptivity structure */
        outfile.write_str("timestep adaptivity structure:\n");
        arkPrintAdaptMem(m.hadapt_mem.as_deref(), outfile);

        /* output inequality constraints quantities */
        outfile.write_str(&format!("maxconstrfails = {}\n", m.maxconstrfails));
    }

    /* output root-finding quantities */
    if ark_mem.borrow().root_mem.is_some() {
        let _ = arkPrintRootMem(ark_mem, outfile);
    }

    /* output interpolation quantities */
    let interp = ark_mem.borrow().interp.clone();
    if let Some(interp) = interp {
        arkInterpPrintMem(Some(&interp), outfile);
    } else {
        outfile.write_str("interpolation = NULL\n");
    }

    /* SUNDIALS_DEBUG_PRINTVEC is not defined: the vector dump is omitted */

    /* Call stepper PrintMem function (if provided) */
    let step_printmem = ark_mem.borrow().step_printmem;
    if let Some(step_printmem) = step_printmem {
        step_printmem(ark_mem, outfile);
    }
}

/*------------------------------------------------------------------------------
  ARKodeCreateMRIStepInnerStepper

  Wraps an ARKODE integrator as an MRIStep inner stepper.
  ----------------------------------------------------------------------------*/

pub fn ARKodeCreateMRIStepInnerStepper(
    inner_arkode_mem: &ARKodeMem,
    stepper: &mut Option<MRIStepInnerStepper>,
) -> i32 {
    /* Check if ark_mem exists: handled by the type system */
    let ark_mem = inner_arkode_mem;

    /* return with an error if the ARKODE solver does not support forcing */
    if ark_mem.borrow().step_setforcing.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeCreateMRIStepInnerStepper",
            file!(),
            "time-stepping module does not support forcing",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    let sunctx = ark_mem.borrow().sunctx.clone();
    let retval = MRIStepInnerStepper_Create(&sunctx, stepper);
    if retval != ARK_SUCCESS {
        return retval;
    }
    let inner = stepper.as_ref().expect("stepper created").clone();

    let retval = MRIStepInnerStepper_SetContent(&inner, Some(Box::new(ark_mem.clone()) as Box<dyn Any>));
    if retval != ARK_SUCCESS {
        return retval;
    }

    let retval = MRIStepInnerStepper_SetEvolveFn(&inner, Some(ark_MRIStepInnerEvolve));
    if retval != ARK_SUCCESS {
        return retval;
    }

    let retval = MRIStepInnerStepper_SetFullRhsFn(&inner, Some(ark_MRIStepInnerFullRhs));
    if retval != ARK_SUCCESS {
        return retval;
    }

    let retval = MRIStepInnerStepper_SetResetFn(&inner, Some(ark_MRIStepInnerReset));
    if retval != ARK_SUCCESS {
        return retval;
    }

    let retval = MRIStepInnerStepper_SetAccumulatedErrorGetFn(
        &inner,
        Some(ark_MRIStepInnerGetAccumulatedError),
    );
    if retval != ARK_SUCCESS {
        return retval;
    }

    let retval = MRIStepInnerStepper_SetAccumulatedErrorResetFn(
        &inner,
        Some(ark_MRIStepInnerResetAccumulatedError),
    );
    if retval != ARK_SUCCESS {
        return retval;
    }

    let retval = MRIStepInnerStepper_SetRTolFn(&inner, Some(ark_MRIStepInnerSetRTol));
    if retval != ARK_SUCCESS {
        return retval;
    }

    ARK_SUCCESS
}

/*===============================================================
  Private internal functions
  ===============================================================*/

/*---------------------------------------------------------------
  arkCreate:

  arkCreate creates an internal memory block for a problem to
  be solved by a time step module built on ARKODE.  If successful,
  arkCreate returns a pointer to the problem memory. If an
  initialization error occurs, arkCreate prints an error message
  to standard err and returns NULL.
  ---------------------------------------------------------------*/
pub fn arkCreate(sunctx: &SUNContext) -> Option<ARKodeMem> {
    /* NULL sunctx check: handled by the type system */

    /* malloc failure branch: allocation cannot fail observably in Rust.
    `ARKodeMemRec::zeroed` is C's malloc + memset(ark_mem, 0, ...) and also
    sets the context. */
    let ark_mem: ARKodeMem = Rc::new(RefCell::new(ARKodeMemRec::zeroed(sunctx.clone())));

    {
        let mut m = ark_mem.borrow_mut();

        /* Set the Python context to NULL */
        m.python = None;

        /* Set uround */
        m.uround = SUN_UNIT_ROUNDOFF;

        /* Initialize time step module to NULL */
        m.step_attachlinsol = None;
        m.step_attachmasssol = None;
        m.step_disablelsetup = None;
        m.step_disablemsetup = None;
        m.step_getlinmem = None;
        m.step_getmassmem = None;
        m.step_getimplicitrhs = None;
        m.step_mmult = None;
        m.step_getgammas = None;
        m.step_init = None;
        m.step_fullrhs = None;
        m.step = None;
        m.step_setuserdata = None;
        m.step_printallstats = None;
        m.step_writeparameters = None;
        m.step_resize = None;
        m.step_reset = None;
        m.step_free = None;
        m.step_printmem = None;
        m.step_setdefaults = None;
        m.step_computestate = None;
        m.step_setrelaxfn = None;
        m.step_setorder = None;
        m.step_setnonlinearsolver = None;
        m.step_setlinear = None;
        m.step_setnonlinear = None;
        m.step_setautonomous = None;
        m.step_setnlsrhsfn = None;
        m.step_setdeduceimplicitrhs = None;
        m.step_setnonlincrdown = None;
        m.step_setnonlinrdiv = None;
        m.step_setdeltagammamax = None;
        m.step_setlsetupfrequency = None;
        m.step_setpredictormethod = None;
        m.step_setmaxnonliniters = None;
        m.step_setnonlinconvcoef = None;
        m.step_setstagepredictfn = None;
        m.step_getnumrhsevals = None;
        m.step_setstepdirection = None;
        m.step_setoptions = None;
        m.step_getnumlinsolvsetups = None;
        m.step_H0 = None;
        m.step_setadaptcontroller = None;
        m.step_getestlocalerrors = None;
        m.step_getcurrentgamma = None;
        m.step_getnonlinearsystemdata = None;
        m.step_getnumnonlinsolviters = None;
        m.step_getnumnonlinsolvconvfails = None;
        m.step_getnonlinsolvstats = None;
        m.step_getstageindex = None;
        m.step_setforcing = None;
        m.step_mem = None;
        m.step_supports_adaptive = SUNFALSE;
        m.step_supports_implicit = SUNFALSE;
        m.step_supports_massmatrix = SUNFALSE;
        m.step_supports_relaxation = SUNFALSE;

        /* Initialize root finding variables */
        m.root_mem = None;

        /* Initialize inequality constraints variables */
        m.constraints = None;

        /* Initialize relaxation variables */
        m.relax_enabled = SUNFALSE;
        m.relax_mem = None;

        /* Initialize lrw and liw */
        m.lrw = 18;
        m.liw = 53; /* fcn/data ptr, int, long int, sunindextype, sunbooleantype */

        /* No mallocs have been done yet */
        m.VabstolMallocDone = SUNFALSE;
        m.VRabstolMallocDone = SUNFALSE;
        m.MallocDone = SUNFALSE;

        /* No user-supplied pre- or post-step functions yet */
        m.PreStepFn = None;
        m.PostStepFn = None;

        /* No user-supplied pre-RHS function yet */
        m.PreRhsFn = None;

        /* No user-supplied stage/step post-processing functions yet */
        m.PostProcessStepFn = None;
        m.PostProcessStageFn = None;

        /* No user_data pointer yet */
        m.user_data = None;
    }

    /* Allocate step adaptivity structure and note storage */
    let hadapt_mem = arkAdaptInit();
    if hadapt_mem.is_none() {
        arkProcessError(
            None,
            ARK_MEM_FAIL,
            line!() as i32,
            "arkCreate",
            file!(),
            "Allocation of step adaptivity structure failed",
        );
        let mut ark_mem = Some(ark_mem);
        ARKodeFree(&mut ark_mem);
        return None;
    }
    {
        let mut m = ark_mem.borrow_mut();
        m.hadapt_mem = hadapt_mem;
        m.lrw += ARK_ADAPT_LRW;
        m.liw += ARK_ADAPT_LIW;

        /* Initialize the interpolation structure to NULL */
        m.interp = None;
        m.interp_type = ARK_INTERP_HERMITE;
        m.interp_degree = ARK_INTERP_MAX_DEGREE;

        /* Initially, rwt should point to ewt */
        m.rwt_is_ewt = SUNTRUE;

        /* Indicate that calling the full RHS function is not required, this flag is
           updated to SUNTRUE by the interpolation module initialization function
           and/or the stepper initialization function in arkInitialSetup */
        m.call_fullrhs = SUNFALSE;

        /* Indicate that the problem needs to be initialized */
        m.initsetup = SUNTRUE;
        m.init_type = FIRST_INIT;
        m.firststage = SUNTRUE;
        m.initialized = SUNFALSE;

        /* Initial step size has not been determined yet */
        m.h = ZERO;
        m.h0u = ZERO;

        /* Accumulated error estimation strategy */
        m.AccumErrorType = ARK_ACCUMERROR_NONE;
        m.AccumError = ZERO;

        /* Default to having stepper initialize ycur during evolution */
        m.ensure_ycur = SUNFALSE;
    }

    /* Set default values for integrator and stepper optional inputs */
    let iret = ARKodeSetDefaults(&ark_mem);
    if iret != ARK_SUCCESS {
        arkProcessError(
            None,
            0,
            line!() as i32,
            "arkCreate",
            file!(),
            "Error setting default solver options",
        );
        let mut ark_mem = Some(ark_mem);
        ARKodeFree(&mut ark_mem);
        return None;
    }

    {
        let mut m = ark_mem.borrow_mut();
        m.load_checkpoint_fail = SUNFALSE;
        m.do_adjoint = SUNFALSE;
    }

    /* Return pointer to ARKODE memory block */
    Some(ark_mem)
}

/*---------------------------------------------------------------
  arkRwtSet

  This routine is responsible for setting the residual weight
  vector rwt, according to tol_type, as follows:

  (1) rwt[i] = 1 / (reltol * SUNRabs(M*ycur[i]) + rabstol), i=0,...,neq-1
      if tol_type = ARK_SS
  (2) rwt[i] = 1 / (reltol * SUNRabs(M*ycur[i]) + rabstol[i]), i=0,...,neq-1
      if tol_type = ARK_SV
  (3) unset if tol_type is any other value (occurs rwt=ewt)

  arkRwtSet returns 0 if rwt is successfully set as above to a
  positive vector and -1 otherwise. In the latter case, rwt is
  considered undefined.

  All the real work is done in the routines arkRwtSetSS, arkRwtSetSV.
  ---------------------------------------------------------------*/
pub fn arkRwtSet(y: &N_Vector, weight: &N_Vector, data: &mut Option<Box<dyn Any>>) -> i32 {
    /* data points to ark_mem here (a boxed ARKodeMem handle clone; C's cast
    of a NULL/foreign pointer is UB -> deterministic panic) */
    let ark_mem = data
        .as_mut()
        .and_then(|b| b.downcast_ref::<ARKodeMem>())
        .cloned()
        .expect("arkRwtSet data holds ARKodeMem");

    let mut flag: i32 = 0;

    /* return if rwt is just ewt */
    if ark_mem.borrow().rwt_is_ewt {
        return 0;
    }

    /* put M*y into ark_tempv1 */
    let My = ark_mem.borrow().tempv1.clone().expect("tempv1 allocated");
    let step_mmult = ark_mem.borrow().step_mmult;
    if let Some(step_mmult) = step_mmult {
        flag = step_mmult(&ark_mem, y, &My);
        if flag != ARK_SUCCESS {
            return ARK_MASSMULT_FAIL;
        }
    } else {
        /* this condition should not apply, but just in case */
        N_VScale(ONE, y, &My);
    }

    /* call appropriate routine to fill rwt */
    let ritol = ark_mem.borrow().ritol;
    match ritol {
        ARK_SS => flag = arkRwtSetSS(&ark_mem, &My, weight),
        ARK_SV => flag = arkRwtSetSV(&ark_mem, &My, weight),
        _ => {}
    }

    flag
}

/*---------------------------------------------------------------
  arkInit:

  arkInit allocates and initializes memory for a problem. All
  inputs are checked for errors. If any error occurs during
  initialization, an error flag is returned. Otherwise, it
  returns ARK_SUCCESS.

  This routine should only be called by
  (a) ARKodeReset (with the input init_type == RESET_INIT),
  (b) an ARKODE timestepper module creation routine (with
      init_type == FIRST_INIT), or
  (c) an ARKODE timestepper module re-initialization routine
      (with init_type == FIRST_INIT).
  This should never be called by the user.

  The initialization type indicates if the values of internal
  counters should be reinitialized (FIRST_INIT) or retained
  (RESET_INIT).

  This routine must be called prior to calling ARKodeEvolve
  to evolve the problem.
  ---------------------------------------------------------------*/
pub fn arkInit(ark_mem: &ARKodeMem, t0: sunrealtype, y0: &N_Vector, init_type: i32) -> i32 {
    let mut init_type = init_type;

    /* Check ark_mem: NULL-mem check handled by the type system */

    /* Check for legal input parameters (NULL y0 handled by the type
    system; the Rc clone aliases the caller's vector) */
    ark_mem.borrow_mut().ycur = Some(y0.clone());

    /* Check if reset was called before the first Evolve call */
    if init_type == RESET_INIT && !ark_mem.borrow().initialized {
        init_type = FIRST_INIT;
    }

    /* Check if allocations have been done i.e., is this first init call */
    if !ark_mem.borrow().MallocDone {
        /* Test if all required time stepper operations are implemented */
        let stepperOK = arkCheckTimestepper(ark_mem);
        if !stepperOK {
            arkProcessError(
                Some(ark_mem),
                ARK_ILL_INPUT,
                line!() as i32,
                "arkInit",
                file!(),
                "Time stepper module is missing required functionality",
            );
            return ARK_ILL_INPUT;
        }

        /* Test if all required vector operations are implemented */
        let nvectorOK = arkCheckNvectorRequired(y0);
        if !nvectorOK {
            arkProcessError(
                Some(ark_mem),
                ARK_ILL_INPUT,
                line!() as i32,
                "arkInit",
                file!(),
                MSG_ARK_BAD_NVECTOR,
            );
            return ARK_ILL_INPUT;
        }

        /* Set space requirements for one N_Vector */
        let mut lrw1: sunindextype = 0;
        let mut liw1: sunindextype = 0;
        if y0.ops.borrow().nvspace.is_some() {
            N_VSpace(y0, &mut lrw1, &mut liw1);
        } else {
            lrw1 = 0;
            liw1 = 0;
        }
        {
            let mut m = ark_mem.borrow_mut();
            m.lrw1 = lrw1;
            m.liw1 = liw1;
        }

        /* Allocate the solver vectors (using y0 as a template) */
        let allocOK = arkAllocVectors(ark_mem, y0);
        if !allocOK {
            arkProcessError(
                Some(ark_mem),
                ARK_MEM_FAIL,
                line!() as i32,
                "arkInit",
                file!(),
                MSG_ARK_MEM_FAIL,
            );
            return ARK_MEM_FAIL;
        }

        /* All allocations are complete */
        ark_mem.borrow_mut().MallocDone = SUNTRUE;
    }

    /* All allocation and error checking is complete at this point */

    /* Copy the input parameters into ARKODE state */
    {
        let mut m = ark_mem.borrow_mut();
        m.tcur = t0;
        m.tn = t0;
    }

    /* Initialize yn */
    let yn = ark_mem.borrow().yn.clone().expect("yn allocated");
    N_VScale(ONE, y0, &yn);
    {
        let mut m = ark_mem.borrow_mut();
        m.fn_is_current = SUNFALSE;

        /* Clear any previous 'tstop' */
        m.tstopset = SUNFALSE;
    }

    /* Initializations on (re-)initialization call, skip on reset */
    if init_type == FIRST_INIT {
        {
            let mut m = ark_mem.borrow_mut();

            /* Counters */
            m.nst_attempts = 0;
            m.nst = 0;
            m.nhnil = 0;
            m.ncfn = 0;
            m.netf = 0;
            m.nconstrfails = 0;

            /* Initial, old, and next step sizes */
            m.h0u = ZERO;
            m.hold = ZERO;
            m.next_h = ZERO;

            /* Tolerance scale factor */
            m.tolsf = ONE;
        }

        /* Reset error controller object */
        let hcontroller = ark_mem
            .borrow()
            .hadapt_mem
            .as_ref()
            .expect("hadapt_mem")
            .hcontroller
            .clone();
        if let Some(hcontroller) = hcontroller {
            let retval = SUNAdaptController_Reset(&hcontroller);
            if retval != SUN_SUCCESS {
                arkProcessError(
                    Some(ark_mem),
                    ARK_CONTROLLER_ERR,
                    line!() as i32,
                    "arkInit",
                    file!(),
                    "Unable to reset error controller object",
                );
                return ARK_CONTROLLER_ERR;
            }
        }

        let mut m = ark_mem.borrow_mut();

        /* Adaptivity counters */
        {
            let hadapt_mem = m.hadapt_mem.as_mut().expect("hadapt_mem");
            hadapt_mem.nst_acc = 0;
            hadapt_mem.nst_exp = 0;
        }

        /* Accumulated error estimate */
        m.AccumError = ZERO;

        /* Indicate that calling the full RHS function is not required, this flag is
           updated to SUNTRUE by the interpolation module initialization function
           and/or the stepper initialization function in arkInitialSetup */
        m.call_fullrhs = SUNFALSE;

        /* Adjoint related */
        m.checkpoint_step_idx = 0;

        /* Indicate that initialization has not been done before */
        m.initialized = SUNFALSE;
    }

    /* Indicate initialization is needed */
    {
        let mut m = ark_mem.borrow_mut();
        m.initsetup = SUNTRUE;
        m.init_type = init_type;
        m.firststage = SUNTRUE;
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  arkCheckTimestepper:

  This routine checks if all required time stepper function
  pointers have been supplied.  If any of them is missing it
  returns SUNFALSE.
  ---------------------------------------------------------------*/
pub fn arkCheckTimestepper(ark_mem: &ARKodeMem) -> sunbooleantype {
    let m = ark_mem.borrow();
    if m.step_init.is_none() || m.step.is_none() || m.step_mem.is_none() {
        return SUNFALSE;
    }
    SUNTRUE
}

/*---------------------------------------------------------------
  arkCheckNvectorRequired:

  This routine checks if all absolutely-required vector
  operations are present.  If any of them is missing it returns
  SUNFALSE.
  ---------------------------------------------------------------*/
pub fn arkCheckNvectorRequired(tmpl: &N_Vector) -> sunbooleantype {
    let ops = tmpl.ops.borrow();
    if ops.nvclone.is_none()
        || ops.nvdestroy.is_none()
        || ops.nvlinearsum.is_none()
        || ops.nvconst.is_none()
        || ops.nvdiv.is_none()
        || ops.nvscale.is_none()
        || ops.nvabs.is_none()
        || ops.nvinv.is_none()
        || ops.nvwrmsnorm.is_none()
    {
        SUNFALSE
    } else {
        SUNTRUE
    }
}

/*---------------------------------------------------------------
  arkCheckNvectorOptional:

  This routine perform conditional checks on required vector
  operations are present (i.e., if the current ARKODE
  configuration requires additional N_Vector routines).  If any
  of them is missing it returns SUNFALSE.
  ---------------------------------------------------------------*/
pub fn arkCheckNvectorOptional(ark_mem: &ARKodeMem) -> sunbooleantype {
    let (user_efun, atolmin0, user_rfun, rwt_is_ewt, Ratolmin0, h0u, hin, itol, ritol, tempv1) = {
        let m = ark_mem.borrow();
        (
            m.user_efun,
            m.atolmin0,
            m.user_rfun,
            m.rwt_is_ewt,
            m.Ratolmin0,
            m.h0u,
            m.hin,
            m.itol,
            m.ritol,
            m.tempv1.clone().expect("tempv1 allocated"),
        )
    };
    let (has_nvmin, has_nvdiv, has_nvmaxnorm, has_nvaddconst) = {
        let ops = tempv1.ops.borrow();
        (
            ops.nvmin.is_some(),
            ops.nvdiv.is_some(),
            ops.nvmaxnorm.is_some(),
            ops.nvaddconst.is_some(),
        )
    };

    /* If using a built-in routine for error/residual weights with abstol==0,
       ensure that N_VMin is available */
    if !user_efun && atolmin0 && !has_nvmin {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "arkCheckNvectorOptional",
            file!(),
            "N_VMin unimplemented (required by error-weight function)",
        );
        return SUNFALSE;
    }
    if !user_rfun && !rwt_is_ewt && Ratolmin0 && !has_nvmin {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "arkCheckNvectorOptional",
            file!(),
            "N_VMin unimplemented (required by residual-weight function)",
        );
        return SUNFALSE;
    }

    /* If the user has not specified a step size (and it will be estimated
       internally), ensure that N_VDiv and N_VMaxNorm are available */
    if (h0u == ZERO) && (hin == ZERO) && !has_nvdiv {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "arkCheckNvectorOptional",
            file!(),
            "N_VDiv unimplemented (required for initial step estimation)",
        );
        return SUNFALSE;
    }
    if (h0u == ZERO) && (hin == ZERO) && !has_nvmaxnorm {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "arkCheckNvectorOptional",
            file!(),
            "N_VMaxNorm unimplemented (required for initial step estimation)",
        );
        return SUNFALSE;
    }

    /* If using a scalar-valued absolute tolerance (for either the state or
       residual), then ensure that N_VAddConst is available */
    if (itol == ARK_SS) && !has_nvaddconst {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "arkCheckNvectorOptional",
            file!(),
            "N_VAddConst unimplemented (required for scalar abstol)",
        );
        return SUNFALSE;
    }
    if !rwt_is_ewt && (ritol == ARK_SS) && !has_nvaddconst {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "arkCheckNvectorOptional",
            file!(),
            "N_VAddConst unimplemented (required for scalar rabstol)",
        );
        return SUNFALSE;
    }

    /* If we made it here, then the vector is sufficient */
    SUNTRUE
}

/* =================================================================
   FRAGMENT: src/arkode/arkode.c, PART B (every function whose C
   definition begins at line 2000 or later).  This file contains ONLY
   function definitions; `arkode.rs` supplies the module doc comment,
   the `use` statements and any module-scope constants.  Every constant
   used below (ZERO, ONE, TWO, FOUR, TENTH, ONEPSM, FUZZ_FACTOR,
   H0_LBFACTOR, H0_UBFACTOR, H0_BIAS, H0_ITERS, ARK_* return codes,
   PREDICT_AGAIN/CONV_FAIL/TRY_AGAIN/..., MSG_ARK_*) comes from the
   frozen contract `crate::arkode_impl`.

   Reference build: SUNDIALS_LOGGING_LEVEL = 2, so every SUNLogInfo /
   SUNLogInfoIf / SUNLogDebug / SUNLogExtraDebug call site in the C is
   omitted at translation time; ARK_WARNING messages (none in this
   part) would still go through arkProcessError.
   ================================================================= */

/*---------------------------------------------------------------
  arkInitialSetup

  This routine performs all necessary items to prepare ARKODE for
  the first internal step after initialization, reinitialization,
  a reset() call, or a resize() call, including:
  - input consistency checks
  - (re)initializes the stepper
  - computes error and residual weights
  - (re)initialize the interpolation structure
  - checks for valid initial step input or estimates first step
  - checks for approach to tstop
  - checks for root near t0
  ---------------------------------------------------------------*/
pub fn arkInitialSetup(ark_mem: &ARKodeMem, tout: sunrealtype) -> i32 {
    /* Is tout too close to tn? */
    let (tcur, uround) = {
        let m = ark_mem.borrow();
        (m.tcur, m.uround)
    };
    let tdist = SUNRabs(tout - tcur);
    let tround = uround * SUNMAX(SUNRabs(tcur), SUNRabs(tout));

    if tdist == ZERO || tdist < TWO * tround {
        arkProcessError(
            Some(ark_mem),
            ARK_TOO_CLOSE,
            line!() as i32,
            "arkInitialSetup",
            file!(),
            MSG_ARK_TOO_CLOSE,
        );
        return ARK_TOO_CLOSE;
    }

    /* Check that user has supplied an initial step size if fixedstep mode is on */
    let (fixedstep, hin) = {
        let m = ark_mem.borrow();
        (m.fixedstep, m.hin)
    };
    if fixedstep && (hin == ZERO) {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "arkInitialSetup",
            file!(),
            "Fixed step mode enabled, but no step size set",
        );
        return ARK_ILL_INPUT;
    }

    /* Perform additional N_Vector checks here, now that ARKODE has been
    fully configured by the user */
    if !arkCheckNvectorOptional(ark_mem) {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "arkInitialSetup",
            file!(),
            MSG_ARK_BAD_NVECTOR,
        );
        return ARK_ILL_INPUT;
    }

    /* Test input tstop for legality (correct direction of integration) */
    if ark_mem.borrow().tstopset {
        let (h, tstop, tcur) = {
            let m = ark_mem.borrow();
            (m.h, m.tstop, m.tcur)
        };
        let htmp = if h == ZERO { tout - tcur } else { h };
        if (tstop - tcur) * htmp <= ZERO {
            arkProcessError(
                Some(ark_mem),
                ARK_ILL_INPUT,
                line!() as i32,
                "arkInitialSetup",
                file!(),
                &MSG_ARK_BAD_TSTOP(tstop, tcur),
            );
            return ARK_ILL_INPUT;
        }
    }

    /* Check to see if y0 satisfies constraints */
    let constraints = ark_mem.borrow().constraints.clone();
    if let Some(constraints) = constraints {
        let (yn, tempv1) = {
            let m = ark_mem.borrow();
            (
                m.yn.clone().expect("yn allocated"),
                m.tempv1.clone().expect("tempv1 allocated"),
            )
        };
        let conOK = N_VConstrMask(&constraints, &yn, &tempv1);
        if !conOK {
            arkProcessError(
                Some(ark_mem),
                ARK_ILL_INPUT,
                line!() as i32,
                "arkInitialSetup",
                file!(),
                MSG_ARK_Y0_FAIL_CONSTR,
            );
            return ARK_ILL_INPUT;
        }
    }

    /* Load initial error weights.

    C: `ark_mem->efun(ark_mem->yn, ark_mem->ewt, ark_mem->e_data)`, where
    `e_data` aliases `ark_mem` for the built-in weight functions and
    `ark_mem->user_data` when the user supplied `efun`.  A `Box` token
    cannot alias (deviation class 6), so the user-efun case passes the
    CURRENT `user_data` box and the built-in case passes `e_data` (which
    holds a boxed `ARKodeMem` handle clone).  The box is taken out of the
    mem around the call and restored on every path. */
    let (yn, ewt) = {
        let m = ark_mem.borrow();
        (
            m.yn.clone().expect("yn allocated"),
            m.ewt.clone().expect("ewt allocated"),
        )
    };
    let (efun, user_efun) = {
        let m = ark_mem.borrow();
        (m.efun, m.user_efun)
    };
    let efun = efun.expect("efun set");
    let retval = if user_efun {
        let mut data = ark_mem.borrow_mut().user_data.take();
        let r = efun(&yn, &ewt, &mut data);
        ark_mem.borrow_mut().user_data = data;
        r
    } else {
        let mut data = ark_mem.borrow_mut().e_data.take();
        let r = efun(&yn, &ewt, &mut data);
        ark_mem.borrow_mut().e_data = data;
        r
    };
    if retval != 0 {
        if ark_mem.borrow().itol == ARK_WF {
            arkProcessError(
                Some(ark_mem),
                ARK_ILL_INPUT,
                line!() as i32,
                "arkInitialSetup",
                file!(),
                MSG_ARK_EWT_FAIL,
            );
        } else {
            arkProcessError(
                Some(ark_mem),
                ARK_ILL_INPUT,
                line!() as i32,
                "arkInitialSetup",
                file!(),
                MSG_ARK_BAD_EWT,
            );
        }
        return ARK_ILL_INPUT;
    }

    /* Set up the time stepper module if not done so already */
    if !ark_mem.borrow().preallocated {
        let step_init = ark_mem.borrow().step_init;
        let step_init = match step_init {
            None => {
                arkProcessError(
                    Some(ark_mem),
                    ARK_ILL_INPUT,
                    line!() as i32,
                    "arkInitialSetup",
                    file!(),
                    "Time stepper module is missing",
                );
                return ARK_ILL_INPUT;
            }
            Some(f) => f,
        };
        let init_type = ark_mem.borrow().init_type;
        let retval = step_init(ark_mem, init_type);
        if retval != ARK_SUCCESS {
            arkProcessError(
                Some(ark_mem),
                retval,
                line!() as i32,
                "arkInitialSetup",
                file!(),
                "Error in initialization of time stepper module",
            );
            return retval;
        }
    }

    /* Load initial residual weights */
    if ark_mem.borrow().rwt_is_ewt {
        /* update pointer to ewt */
        let ewt = ark_mem.borrow().ewt.clone();
        ark_mem.borrow_mut().rwt = ewt;
    } else {
        let (yn, rwt) = {
            let m = ark_mem.borrow();
            (
                m.yn.clone().expect("yn allocated"),
                m.rwt.clone().expect("rwt allocated"),
            )
        };
        let (rfun, user_rfun) = {
            let m = ark_mem.borrow();
            (m.rfun, m.user_rfun)
        };
        let rfun = rfun.expect("rfun set");
        let retval = if user_rfun {
            let mut data = ark_mem.borrow_mut().user_data.take();
            let r = rfun(&yn, &rwt, &mut data);
            ark_mem.borrow_mut().user_data = data;
            r
        } else {
            let mut data = ark_mem.borrow_mut().r_data.take();
            let r = rfun(&yn, &rwt, &mut data);
            ark_mem.borrow_mut().r_data = data;
            r
        };
        if retval != 0 {
            if ark_mem.borrow().itol == ARK_WF {
                arkProcessError(
                    Some(ark_mem),
                    ARK_ILL_INPUT,
                    line!() as i32,
                    "arkInitialSetup",
                    file!(),
                    MSG_ARK_RWT_FAIL,
                );
            } else {
                arkProcessError(
                    Some(ark_mem),
                    ARK_ILL_INPUT,
                    line!() as i32,
                    "arkInitialSetup",
                    file!(),
                    MSG_ARK_BAD_RWT,
                );
            }
            return ARK_ILL_INPUT;
        }
    }

    /* Create default interpolation module (if needed) */
    let (interp_type, interp_present) = {
        let m = ark_mem.borrow();
        (m.interp_type, m.interp.is_some())
    };
    if interp_type != ARK_INTERP_NONE && !interp_present {
        let interp_degree = ark_mem.borrow().interp_degree;
        let interp = if interp_type == ARK_INTERP_LAGRANGE {
            arkInterpCreate_Lagrange(ark_mem, interp_degree)
        } else {
            arkInterpCreate_Hermite(ark_mem, interp_degree)
        };
        let is_none = interp.is_none();
        ark_mem.borrow_mut().interp = interp;
        if is_none {
            arkProcessError(
                Some(ark_mem),
                ARK_MEM_FAIL,
                line!() as i32,
                "arkInitialSetup",
                file!(),
                "Unable to allocate interpolation module",
            );
            return ARK_MEM_FAIL;
        }
    }

    /* Fill initial interpolation data (if needed) */
    let interp = ark_mem.borrow().interp.clone();
    if let Some(interp) = interp.as_ref() {
        /* Stepper init may have limited the interpolation degree */
        let interp_degree = ark_mem.borrow().interp_degree;
        if arkInterpSetDegree(ark_mem, Some(interp), interp_degree) != 0 {
            arkProcessError(
                Some(ark_mem),
                ARK_ILL_INPUT,
                line!() as i32,
                "arkInitialSetup",
                file!(),
                "Unable to update interpolation polynomial degree",
            );
            return ARK_ILL_INPUT;
        }

        let tcur = ark_mem.borrow().tcur;
        if arkInterpInit(ark_mem, Some(interp), tcur) != 0 {
            arkProcessError(
                Some(ark_mem),
                ARK_ILL_INPUT,
                line!() as i32,
                "arkInitialSetup",
                file!(),
                "Unable to initialize interpolation module",
            );
            return ARK_ILL_INPUT;
        }
    }

    /* Check if the configuration requires interpolation */
    let (root_present, interp_present, tstopinterp) = {
        let m = ark_mem.borrow();
        (m.root_mem.is_some(), m.interp.is_some(), m.tstopinterp)
    };
    if root_present && !interp_present {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "arkInitialSetup",
            file!(),
            "Rootfinding requires an interpolation module",
        );
        return ARK_ILL_INPUT;
    }

    if tstopinterp && !interp_present {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "arkInitialSetup",
            file!(),
            "Stop time interpolation requires an interpolation module",
        );
        return ARK_ILL_INPUT;
    }

    /* Call stepper-provided initial step size estimation routine to fill
    ark_mem->hin, if applicable. */
    let (h0u, hin, fixedstep, step_H0) = {
        let m = ark_mem.borrow();
        (m.h0u, m.hin, m.fixedstep, m.step_H0)
    };
    if h0u == ZERO && hin == ZERO && !fixedstep && step_H0.is_some() {
        let step_H0 = step_H0.expect("step_H0 set");
        /* C passes `&(ark_mem->hin)` straight into the stepper; the port
        copies the field out, calls, and writes it back on every path
        (binding invariant B). */
        let mut hin_out = ark_mem.borrow().hin;
        let retval = step_H0(ark_mem, tout, &mut hin_out);
        ark_mem.borrow_mut().hin = hin_out;
        if retval != 0 {
            arkProcessError(
                Some(ark_mem),
                ARK_STEP_H0_FAIL,
                line!() as i32,
                "arkInitialSetup",
                file!(),
                "Failure in timestepping module h0 calculation",
            );
            return ARK_STEP_H0_FAIL;
        }
    }

    /* If fullrhs will be called (to estimate initial step, explicit steppers, Hermite
    interpolation module, and possibly (but not always) arkRootCheck1), then
    ensure that it is provided, and space is allocated for fn.  Otherwise,
    we should free ark_mem->fn if it is allocated. */
    let (call_fullrhs, h0u, hin, root_present) = {
        let m = ark_mem.borrow();
        (m.call_fullrhs, m.h0u, m.hin, m.root_mem.is_some())
    };
    if call_fullrhs || (h0u == ZERO && hin == ZERO) || root_present {
        if ark_mem.borrow().step_fullrhs.is_none() {
            arkProcessError(
                Some(ark_mem),
                ARK_ILL_INPUT,
                line!() as i32,
                "arkInitialSetup",
                file!(),
                MSG_ARK_MISSING_FULLRHS,
            );
            return ARK_ILL_INPUT;
        }

        let yn = ark_mem.borrow().yn.clone().expect("yn allocated");
        let mut fn_ = ark_mem.borrow_mut().fn_.take();
        let allocOK = arkAllocVec(ark_mem, &yn, &mut fn_);
        ark_mem.borrow_mut().fn_ = fn_;
        if !allocOK {
            arkProcessError(
                Some(ark_mem),
                ARK_MEM_FAIL,
                line!() as i32,
                "arkInitialSetup",
                file!(),
                MSG_ARK_MEM_FAIL,
            );
            return ARK_MEM_FAIL;
        }
    } else if ark_mem.borrow().fn_.is_some() {
        let mut fn_ = ark_mem.borrow_mut().fn_.take();
        arkFreeVec(ark_mem, &mut fn_);
        ark_mem.borrow_mut().fn_ = fn_;
    }

    /* initialization complete */
    ark_mem.borrow_mut().initialized = SUNTRUE;

    /* Set initial step size */
    if ark_mem.borrow().h0u == ZERO {
        /* Check input h for validity */
        {
            let mut guard = ark_mem.borrow_mut();
            let m = &mut *guard;
            m.h = m.hin;
        }
        let (h, tcur) = {
            let m = ark_mem.borrow();
            (m.h, m.tcur)
        };
        if (h != ZERO) && ((tout - tcur) * h < ZERO) {
            arkProcessError(
                Some(ark_mem),
                ARK_ILL_INPUT,
                line!() as i32,
                "arkInitialSetup",
                file!(),
                MSG_ARK_BAD_H0,
            );
            return ARK_ILL_INPUT;
        }

        /* Estimate initial h if not set */
        if h == ZERO {
            /* If necessary, temporarily set h as it is used to compute the tolerance
            in a potential mass matrix solve when computing the full rhs */
            {
                let mut guard = ark_mem.borrow_mut();
                let m = &mut *guard;
                m.h = SUNRabs(tout - m.tcur);
                if m.h == ZERO {
                    m.h = ONE;
                }
            }

            /* Estimate the first step size */
            let mut tout_hin = tout;
            let (tstopset, tstop, tcur) = {
                let m = ark_mem.borrow();
                (m.tstopset, m.tstop, m.tcur)
            };
            if tstopset && (tout - tcur) * (tout - tstop) > ZERO {
                tout_hin = tstop;
            }
            let hflag = arkHin(ark_mem, tout_hin);
            if hflag != ARK_SUCCESS {
                let istate = arkHandleFailure(ark_mem, hflag);
                return istate;
            }

            /* Use first step growth factor for estimated h */
            {
                let mut guard = ark_mem.borrow_mut();
                let m = &mut *guard;
                let hadapt_mem = m.hadapt_mem.as_mut().expect("hadapt_mem allocated");
                let etamx1 = hadapt_mem.etamx1;
                hadapt_mem.etamax = etamx1;
            }
        } else if ark_mem.borrow().nst == 0 {
            /* Use first step growth factor for user defined h */
            let mut guard = ark_mem.borrow_mut();
            let m = &mut *guard;
            let hadapt_mem = m.hadapt_mem.as_mut().expect("hadapt_mem allocated");
            let etamx1 = hadapt_mem.etamx1;
            hadapt_mem.etamax = etamx1;
        } else {
            /* Use standard growth factor (e.g., for reset) */
            let mut guard = ark_mem.borrow_mut();
            let m = &mut *guard;
            let hadapt_mem = m.hadapt_mem.as_mut().expect("hadapt_mem allocated");
            let growth = hadapt_mem.growth;
            hadapt_mem.etamax = growth;
        }

        /* Enforce step size bounds */
        {
            let mut guard = ark_mem.borrow_mut();
            let m = &mut *guard;
            let rh = SUNRabs(m.h) * m.hmax_inv;
            if rh > ONE {
                m.h /= rh;
            }
            let habs = SUNRabs(m.h);
            if habs < m.hmin {
                let scale = m.hmin / habs;
                m.h *= scale;
            }
        }

        /* Check for approach to tstop */
        {
            let mut guard = ark_mem.borrow_mut();
            let m = &mut *guard;
            if m.tstopset && ((m.tcur + m.h - m.tstop) * m.h > ZERO) {
                m.h = (m.tstop - m.tcur) * (ONE - FOUR * m.uround);
            }
        }

        /* Set initial time step factors */
        {
            let mut guard = ark_mem.borrow_mut();
            let m = &mut *guard;
            m.h0u = m.h;
            m.eta = ONE;
            m.hprime = m.h;
        }
    } else {
        /* If next step would overtake tstop, adjust stepsize */
        let mut guard = ark_mem.borrow_mut();
        let m = &mut *guard;
        if m.tstopset && ((m.tcur + m.hprime - m.tstop) * m.h > ZERO) {
            m.hprime = (m.tstop - m.tcur) * (ONE - FOUR * m.uround);
            m.eta = m.hprime / m.h;
        }
    }

    /* Check for zeros of root function g at and near t0. */
    let nrtfn = ark_mem
        .borrow()
        .root_mem
        .as_ref()
        .map(|r| r.nrtfn)
        .unwrap_or(0);
    if ark_mem.borrow().root_mem.is_some() && nrtfn > 0 {
        let retval = arkRootCheck1(ark_mem);
        if retval != ARK_SUCCESS {
            return retval;
        }
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  arkStopTests

  This routine performs relevant stopping tests:
  - check for root in last step
  - check if we passed tstop
  - check if we passed tout (NORMAL mode)
  - check if current tn was returned (ONE_STEP mode)
  - check if we are close to tstop
  (adjust step size if needed)
  ---------------------------------------------------------------*/
pub fn arkStopTests(
    ark_mem: &ARKodeMem,
    tout: sunrealtype,
    yout: &N_Vector,
    tret: &mut sunrealtype,
    itask: i32,
    ier: &mut i32,
) -> i32 {
    /* Estimate an infinitesimal time interval to be used as
    a roundoff for time quantities (based on current time
    and step size) */
    let troundoff = {
        let m = ark_mem.borrow();
        FUZZ_FACTOR * m.uround * (SUNRabs(m.tcur) + SUNRabs(m.h))
    };

    /* First, check for a root in the last step taken, other than the
    last root found, if any.  If itask = ARK_ONE_STEP and y(tn) was not
    returned because of an intervening root, return y(tn) now.     */
    if ark_mem.borrow().root_mem.is_some() {
        let nrtfn = ark_mem
            .borrow()
            .root_mem
            .as_ref()
            .expect("root_mem allocated")
            .nrtfn;
        if nrtfn > 0 {
            /* Shortcut to roots found in previous step */
            let irfndp = ark_mem
                .borrow()
                .root_mem
                .as_ref()
                .expect("root_mem allocated")
                .irfnd;

            /* If the full RHS was not computed in the last call to arkCompleteStep
            and roots were found in the previous step, then compute the full rhs
            for possible use in arkRootCheck2 (not always necessary) */
            let fn_is_current = ark_mem.borrow().fn_is_current;
            if !fn_is_current && irfndp != 0 {
                let (step_fullrhs, tn, yn, fn_) = {
                    let m = ark_mem.borrow();
                    (
                        m.step_fullrhs,
                        m.tn,
                        m.yn.clone().expect("yn allocated"),
                        m.fn_.clone().expect("fn allocated"),
                    )
                };
                let step_fullrhs = step_fullrhs.expect("step_fullrhs set");
                let retval = step_fullrhs(ark_mem, tn, &yn, &fn_, ARK_FULLRHS_END);
                if retval != 0 {
                    /* NOTE: upstream C passes MSG_ARK_RHSFUNC_FAILED (which carries a
                    SUN_FORMAT_G conversion) with NO argument here -- undefined
                    behavior in C.  The port supplies ark_mem->tcur, the value every
                    other MSG_ARK_RHSFUNC_FAILED call site uses. */
                    let tcur = ark_mem.borrow().tcur;
                    arkProcessError(
                        Some(ark_mem),
                        ARK_RHSFUNC_FAIL,
                        line!() as i32,
                        "arkStopTests",
                        file!(),
                        &MSG_ARK_RHSFUNC_FAILED(tcur),
                    );
                    *ier = ARK_RHSFUNC_FAIL;
                    return 1;
                }
                ark_mem.borrow_mut().fn_is_current = SUNTRUE;
            }

            let retval = arkRootCheck2(ark_mem);

            if retval == CLOSERT {
                let tlo = ark_mem
                    .borrow()
                    .root_mem
                    .as_ref()
                    .expect("root_mem allocated")
                    .tlo;
                arkProcessError(
                    Some(ark_mem),
                    ARK_ILL_INPUT,
                    line!() as i32,
                    "arkStopTests",
                    file!(),
                    &MSG_ARK_CLOSE_ROOTS(tlo),
                );
                *ier = ARK_ILL_INPUT;
                return 1;
            } else if retval == ARK_RTFUNC_FAIL {
                let tlo = ark_mem
                    .borrow()
                    .root_mem
                    .as_ref()
                    .expect("root_mem allocated")
                    .tlo;
                arkProcessError(
                    Some(ark_mem),
                    ARK_RTFUNC_FAIL,
                    line!() as i32,
                    "arkStopTests",
                    file!(),
                    &MSG_ARK_RTFUNC_FAILED(tlo),
                );
                *ier = ARK_RTFUNC_FAIL;
                return 1;
            } else if retval == RTFOUND {
                let tlo = ark_mem
                    .borrow()
                    .root_mem
                    .as_ref()
                    .expect("root_mem allocated")
                    .tlo;
                ark_mem.borrow_mut().tretlast = tlo;
                *tret = tlo;
                *ier = ARK_ROOT_RETURN;
                return 1;
            }

            /* If tn is distinct from tretlast (within roundoff),
            check remaining interval for roots */
            let (tcur, tretlast) = {
                let m = ark_mem.borrow();
                (m.tcur, m.tretlast)
            };
            if SUNRabs(tcur - tretlast) > troundoff {
                let retval = arkRootCheck3(ark_mem, tout, itask);

                if retval == ARK_SUCCESS {
                    /* no root found */
                    ark_mem
                        .borrow_mut()
                        .root_mem
                        .as_mut()
                        .expect("root_mem allocated")
                        .irfnd = 0;
                    if (irfndp == 1) && (itask == ARK_ONE_STEP) {
                        let (tcur, yn) = {
                            let m = ark_mem.borrow();
                            (m.tcur, m.yn.clone().expect("yn allocated"))
                        };
                        ark_mem.borrow_mut().tretlast = tcur;
                        *tret = tcur;
                        N_VScale(ONE, &yn, yout);
                        *ier = ARK_SUCCESS;
                        return 1;
                    }
                } else if retval == RTFOUND {
                    /* a new root was found */
                    let tlo = {
                        let mut m = ark_mem.borrow_mut();
                        let root_mem = m.root_mem.as_mut().expect("root_mem allocated");
                        root_mem.irfnd = 1;
                        root_mem.tlo
                    };
                    ark_mem.borrow_mut().tretlast = tlo;
                    *tret = tlo;
                    *ier = ARK_ROOT_RETURN;
                    return 1;
                } else if retval == ARK_RTFUNC_FAIL {
                    /* g failed */
                    let tlo = ark_mem
                        .borrow()
                        .root_mem
                        .as_ref()
                        .expect("root_mem allocated")
                        .tlo;
                    arkProcessError(
                        Some(ark_mem),
                        ARK_RTFUNC_FAIL,
                        line!() as i32,
                        "arkStopTests",
                        file!(),
                        &MSG_ARK_RTFUNC_FAILED(tlo),
                    );
                    *ier = ARK_RTFUNC_FAIL;
                    return 1;
                }
            }
        } /* end of root stop check */
    }

    /* Test for tn at tstop or near tstop */
    if ark_mem.borrow().tstopset {
        let (tcur, tstop, h) = {
            let m = ark_mem.borrow();
            (m.tcur, m.tstop, m.h)
        };
        if SUNRabs(tcur - tstop) <= troundoff {
            /* Ensure tout >= tstop, otherwise check for tout return below */
            if (tout - tstop) * h >= ZERO || SUNRabs(tout - tstop) <= troundoff {
                let (tstopinterp, interp_present) = {
                    let m = ark_mem.borrow();
                    (m.tstopinterp, m.interp.is_some())
                };
                if tstopinterp && interp_present {
                    *ier = ARKodeGetDky(ark_mem, tstop, 0, yout);
                    if *ier != ARK_SUCCESS {
                        arkProcessError(
                            Some(ark_mem),
                            ARK_ILL_INPUT,
                            line!() as i32,
                            "arkStopTests",
                            file!(),
                            &MSG_ARK_BAD_TSTOP(tstop, tcur),
                        );
                        *ier = ARK_ILL_INPUT;
                        return 1;
                    }
                } else {
                    let yn = ark_mem.borrow().yn.clone().expect("yn allocated");
                    N_VScale(ONE, &yn, yout);
                }
                {
                    let mut m = ark_mem.borrow_mut();
                    m.tretlast = tstop;
                    m.tstopset = SUNFALSE;
                }
                *tret = tstop;
                *ier = ARK_TSTOP_RETURN;
                return 1;
            }
        }
        /* If next step would overtake tstop, adjust stepsize */
        else {
            let mut guard = ark_mem.borrow_mut();
            let m = &mut *guard;
            if (m.tcur + m.hprime - m.tstop) * m.h > ZERO {
                m.hprime = (m.tstop - m.tcur) * (ONE - FOUR * m.uround);
                m.eta = m.hprime / m.h;
            }
        }
    }

    /* In ARK_NORMAL mode, test if tout was reached */
    let (tcur, h) = {
        let m = ark_mem.borrow();
        (m.tcur, m.h)
    };
    if (itask == ARK_NORMAL) && ((tcur - tout) * h >= ZERO) {
        if ark_mem.borrow().interp.is_some() {
            *ier = ARKodeGetDky(ark_mem, tout, 0, yout);
            if *ier != ARK_SUCCESS {
                arkProcessError(
                    Some(ark_mem),
                    ARK_ILL_INPUT,
                    line!() as i32,
                    "arkStopTests",
                    file!(),
                    &MSG_ARK_BAD_TOUT(tout),
                );
                *ier = ARK_ILL_INPUT;
                return 1;
            }
            ark_mem.borrow_mut().tretlast = tout;
            *tret = tout;
        } else {
            let yn = ark_mem.borrow().yn.clone().expect("yn allocated");
            N_VScale(ONE, &yn, yout);
            ark_mem.borrow_mut().tretlast = tcur;
            *tret = tcur;
        }
        *ier = ARK_SUCCESS;
        return 1;
    }

    /* In ARK_ONE_STEP mode, test if tn was returned */
    let tretlast = ark_mem.borrow().tretlast;
    if itask == ARK_ONE_STEP && SUNRabs(tcur - tretlast) > troundoff {
        let yn = ark_mem.borrow().yn.clone().expect("yn allocated");
        ark_mem.borrow_mut().tretlast = tcur;
        *tret = tcur;
        N_VScale(ONE, &yn, yout);
        *ier = ARK_SUCCESS;
        return 1;
    }

    0
}

/*---------------------------------------------------------------
  arkHin

  This routine computes a tentative initial step size h0.
  Note that here tout is either the value passed to ARKodeEvolve
  at the first call or the value of tstop (if tstop is enabled and
  it is closer to t0=tn than tout). If the RHS function fails
  unrecoverably, arkHin returns ARK_RHSFUNC_FAIL. If the RHS
  function fails recoverably too many times and recovery is not
  possible, arkHin returns ARK_REPTD_RHSFUNC_ERR. Otherwise, arkHin
  sets h to the chosen value h0 and returns ARK_SUCCESS.

  The algorithm used seeks to find h0 as a solution of
  (WRMS norm of (h0^2 ydd / 2)) = 1,
  where ydd = estimated second derivative of y.

  We start with an initial estimate equal to the geometric mean
  of the lower and upper bounds on the step size.

  Loop up to H0_ITERS times to find h0.
  Stop if new and previous values differ by a factor < 2.
  Stop if hnew/hg > 2 after one iteration, as this probably
  means that the ydd value is bad because of cancellation error.

  For each new proposed hg, we allow H0_ITERS attempts to
  resolve a possible recoverable failure from f() by reducing
  the proposed stepsize by a factor of 0.2. If a legal stepsize
  still cannot be found, fall back on a previous value if
  possible, or else return ARK_REPTD_RHSFUNC_ERR.

  Finally, we apply a bias (0.5) and verify that h0 is within
  bounds.
  ---------------------------------------------------------------*/
pub fn arkHin(ark_mem: &ARKodeMem, tout: sunrealtype) -> i32 {
    /* arkInitialSetup checks for tdiff = 0 or < 2 * troundoff */
    let (tcur, uround) = {
        let m = ark_mem.borrow();
        (m.tcur, m.uround)
    };
    let tdiff = tout - tcur;
    let sign: i32 = if tdiff > ZERO { 1 } else { -1 };
    let tdist = SUNRabs(tdiff);
    let tround = uround * SUNMAX(SUNRabs(tcur), SUNRabs(tout));

    /* call full RHS if needed */
    if !ark_mem.borrow().fn_is_current {
        /* NOTE: The step size (h) is used in setting the tolerance in a potential
        mass matrix solve when computing the full RHS. Before calling arkHin, h
        is set to |tout - tcur| or 1 and so we do not need to guard against
        h == 0 here before calling the full RHS. */
        let (step_fullrhs, tn, yn, fn_) = {
            let m = ark_mem.borrow();
            (
                m.step_fullrhs,
                m.tn,
                m.yn.clone().expect("yn allocated"),
                m.fn_.clone().expect("fn allocated"),
            )
        };
        let step_fullrhs = step_fullrhs.expect("step_fullrhs set");
        let retval = step_fullrhs(ark_mem, tn, &yn, &fn_, ARK_FULLRHS_START);
        if retval != 0 {
            return ARK_RHSFUNC_FAIL;
        }
        ark_mem.borrow_mut().fn_is_current = SUNTRUE;
    }

    /* Set lower and upper bounds on h0, and take geometric mean
    as first trial value.
    Exit with this value if the bounds cross each other. */
    let hlb = H0_LBFACTOR * tround;
    let hub = arkUpperBoundH0(ark_mem, tdist);

    let mut hg = SUNRsqrt(hlb * hub);

    if hub < hlb {
        if sign == -1 {
            ark_mem.borrow_mut().h = -hg;
        } else {
            ark_mem.borrow_mut().h = hg;
        }
        return ARK_SUCCESS;
    }

    /* Outer loop */
    let mut hs = hg; /* safeguard against 'uninitialized variable' warning */
    let mut hnew = ZERO;
    let mut yddnrm = ZERO;
    for count1 in 1..=H0_ITERS {
        /* Attempts to estimate ydd */
        let mut hgOK = SUNFALSE;

        for _count2 in 1..=H0_ITERS {
            let hgs = hg * sign as sunrealtype;
            let retval = arkYddNorm(ark_mem, hgs, &mut yddnrm);
            /* If f() failed unrecoverably, give up */
            if retval < 0 {
                return ARK_RHSFUNC_FAIL;
            }
            /* If successful, we can use ydd */
            if retval == ARK_SUCCESS {
                hgOK = SUNTRUE;
                break;
            }
            /* f() failed recoverably; cut step size and test it again */
            hg *= 0.2;
        }

        /* If f() failed recoverably H0_ITERS times */
        if !hgOK {
            /* Exit if this is the first or second pass. No recovery possible */
            if count1 <= 2 {
                return ARK_REPTD_RHSFUNC_ERR;
            }
            /* We have a fall-back option. The value hs is a previous hnew which
            passed through f(). Use it and break */
            hnew = hs;
            break;
        }

        /* The proposed step size is feasible. Save it. */
        hs = hg;

        /* Propose new step size */
        hnew = if yddnrm * hub * hub > TWO {
            SUNRsqrt(TWO / yddnrm)
        } else {
            SUNRsqrt(hg * hub)
        };

        /* If last pass, stop now with hnew */
        if count1 == H0_ITERS {
            break;
        }

        let hrat = hnew / hg;

        /* Accept hnew if it does not differ from hg by more than a factor of 2 */
        if (hrat > HALF) && (hrat < TWO) {
            break;
        }

        /* After one pass, if ydd seems to be bad, use fall-back value. */
        if (count1 > 1) && (hrat > TWO) {
            hnew = hg;
            break;
        }

        /* Send this value back through f() */
        hg = hnew;
    }

    /* Apply bounds, bias factor, and attach sign */
    let mut h0 = H0_BIAS * hnew;
    if h0 < hlb {
        h0 = hlb;
    }
    if h0 > hub {
        h0 = hub;
    }
    if sign == -1 {
        h0 = -h0;
    }
    ark_mem.borrow_mut().h = h0;

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  arkUpperBoundH0

  This routine sets an upper bound on abs(h0) based on
  tdist = tn - t0 and the values of y[i]/y'[i].
  ---------------------------------------------------------------*/
pub fn arkUpperBoundH0(ark_mem: &ARKodeMem, tdist: sunrealtype) -> sunrealtype {
    /* Bound based on |y0|/|y0'| -- allow at most an increase of
     * H0_UBFACTOR in y0 (based on a forward Euler step). The weight
     * factor is used as a safeguard against zero components in y0. */
    let (temp1, temp2, yn, fn_) = {
        let m = ark_mem.borrow();
        (
            m.tempv1.clone().expect("tempv1 allocated"),
            m.tempv2.clone().expect("tempv2 allocated"),
            m.yn.clone().expect("yn allocated"),
            m.fn_.clone().expect("fn allocated"),
        )
    };

    N_VAbs(&yn, &temp2);

    /* C: ark_mem->efun(ark_mem->yn, temp1, ark_mem->e_data); return ignored */
    {
        let (efun, user_efun) = {
            let m = ark_mem.borrow();
            (m.efun, m.user_efun)
        };
        let efun = efun.expect("efun set");
        if user_efun {
            let mut data = ark_mem.borrow_mut().user_data.take();
            let _ = efun(&yn, &temp1, &mut data);
            ark_mem.borrow_mut().user_data = data;
        } else {
            let mut data = ark_mem.borrow_mut().e_data.take();
            let _ = efun(&yn, &temp1, &mut data);
            ark_mem.borrow_mut().e_data = data;
        }
    }

    N_VInv(&temp1, &temp1);
    N_VLinearSum(H0_UBFACTOR, &temp2, ONE, &temp1, &temp1);

    N_VAbs(&fn_, &temp2);

    N_VDiv(&temp2, &temp1, &temp1);
    let hub_inv = N_VMaxNorm(&temp1);

    /* bound based on tdist -- allow at most a step of magnitude
     * H0_UBFACTOR * tdist */
    let mut hub = H0_UBFACTOR * tdist;

    /* Use the smaller of the two */
    if hub * hub_inv > ONE {
        hub = ONE / hub_inv;
    }

    hub
}

/*---------------------------------------------------------------
  arkYddNorm

  This routine computes an estimate of the second derivative of y
  using a difference quotient, and returns its WRMS norm.
  ---------------------------------------------------------------*/
pub fn arkYddNorm(ark_mem: &ARKodeMem, hg: sunrealtype, yddnrm: &mut sunrealtype) -> i32 {
    let (fn_, yn, ycur, tempv1, ewt, tcur) = {
        let m = ark_mem.borrow();
        (
            m.fn_.clone().expect("fn allocated"),
            m.yn.clone().expect("yn allocated"),
            m.ycur.clone().expect("ycur set"),
            m.tempv1.clone().expect("tempv1 allocated"),
            m.ewt.clone().expect("ewt allocated"),
            m.tcur,
        )
    };

    /* increment y with a multiple of f */
    N_VLinearSum(hg, &fn_, ONE, &yn, &ycur);

    /* compute y', via the ODE RHS routine */
    let step_fullrhs = ark_mem.borrow().step_fullrhs.expect("step_fullrhs set");
    let retval = step_fullrhs(ark_mem, tcur + hg, &ycur, &tempv1, ARK_FULLRHS_OTHER);
    if retval != 0 {
        return ARK_RHSFUNC_FAIL;
    }

    /* difference new f and original f to estimate y'' */
    N_VLinearSum(ONE / hg, &tempv1, -ONE / hg, &fn_, &tempv1);

    /* reset ycur to equal yn (unnecessary?) */
    N_VScale(ONE, &yn, &ycur);

    /* compute norm of y'' */
    *yddnrm = N_VWrmsNorm(&tempv1, &ewt);

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  arkCompleteStep

  This routine performs various update operations when the step
  solution is complete.  It is assumed that the timestepper
  module has stored the time-evolved solution in ark_mem->ycur,
  and the step that gave rise to this solution in ark_mem->h.
  We update the current time (tn), the current solution (yn),
  increment the overall step counter nst, record the values hold
  and tnew, allow for user-provided postprocessing, and update
  the interpolation structure.
  ---------------------------------------------------------------*/
pub fn arkCompleteStep(ark_mem: &ARKodeMem, dsm: sunrealtype) -> i32 {
    /* Set current time to the end of the step (in case the last
    stage time does not coincide with the step solution time).
    If tstop is enabled, it is possible for tn + h to be past
    tstop by roundoff, and in that case, we reset tn (after
    incrementing by h) to tstop. */

    /* During long-time integration, roundoff can creep into tcur.
    Compensated summation fixes this but with increased cost, so it is optional. */
    {
        let mut guard = ark_mem.borrow_mut();
        let m = &mut *guard;
        if m.use_compensated_sums {
            sundials_core::sundials_utils::sunCompensatedSum(
                m.tn,
                m.h,
                &mut m.tcur,
                &mut m.terr,
            );
        } else {
            m.tcur = m.tn + m.h;
        }

        if m.tstopset {
            let troundoff = FUZZ_FACTOR * m.uround * (SUNRabs(m.tcur) + SUNRabs(m.h));
            if SUNRabs(m.tcur - m.tstop) <= troundoff {
                m.tcur = m.tstop;
            }
        }

        /* store this step's contribution to accumulated temporal error */
        if m.AccumErrorType != ARK_ACCUMERROR_NONE {
            if m.AccumErrorType == ARK_ACCUMERROR_MAX {
                m.AccumError = SUNMAX(dsm, m.AccumError);
            } else if m.AccumErrorType == ARK_ACCUMERROR_SUM {
                m.AccumError += dsm;
            } else
            /* ARK_ACCUMERROR_AVG */
            {
                m.AccumError += dsm * m.h;
            }
        }
    }

    /* call the user-supplied post-step function (if supplied) */
    let PostStepFn = ark_mem.borrow().PostStepFn;
    if let Some(PostStepFn) = PostStepFn {
        let (tcur, ycur, nst) = {
            let m = ark_mem.borrow();
            (m.tcur, m.ycur.clone().expect("ycur set"), m.nst)
        };
        let mut user_data = ark_mem.borrow_mut().user_data.take();
        let retval = PostStepFn(tcur, &ycur, nst, &mut user_data);
        ark_mem.borrow_mut().user_data = user_data;
        if retval != 0 {
            return ARK_POSTSTEPFN_FAIL;
        }
    }

    /* update interpolation structure

    NOTE: This must be called before updating yn with ycur as the interpolation
    module may need to save tn, yn from the start of this step. */
    let interp = ark_mem.borrow().interp.clone();
    if let Some(interp) = interp.as_ref() {
        let tcur = ark_mem.borrow().tcur;
        let retval = arkInterpUpdate(ark_mem, Some(interp), tcur);
        if retval != ARK_SUCCESS {
            return retval;
        }
    }

    /* update yn to current solution */
    let (ycur, yn) = {
        let m = ark_mem.borrow();
        (
            m.ycur.clone().expect("ycur set"),
            m.yn.clone().expect("yn allocated"),
        )
    };
    N_VScale(ONE, &ycur, &yn);
    ark_mem.borrow_mut().fn_is_current = SUNFALSE;

    /* Notify time step controller object of successful step */
    let hcontroller = ark_mem
        .borrow()
        .hadapt_mem
        .as_ref()
        .expect("hadapt_mem allocated")
        .hcontroller
        .clone();
    if let Some(hcontroller) = hcontroller.as_ref() {
        let h = ark_mem.borrow().h;
        let retval =
            sundials_core::sundials_adaptcontroller::SUNAdaptController_UpdateH(hcontroller, h, dsm);
        if retval != sundials_core::sundials_errors::SUN_SUCCESS {
            arkProcessError(
                Some(ark_mem),
                ARK_CONTROLLER_ERR,
                line!() as i32,
                "arkCompleteStep",
                file!(),
                "Failure updating controller object",
            );
            return ARK_CONTROLLER_ERR;
        }
    }

    /* update scalar quantities */
    {
        let mut guard = ark_mem.borrow_mut();
        let m = &mut *guard;
        m.nst += 1;
        m.checkpoint_step_idx += 1;
        m.hold = m.h;
        m.tn = m.tcur;
        m.hprime = m.h * m.eta;

        /* Reset growth factor for subsequent time step */
        let hadapt_mem = m.hadapt_mem.as_mut().expect("hadapt_mem allocated");
        let growth = hadapt_mem.growth;
        hadapt_mem.etamax = growth;
    }

    /* Turn off flag indicating initial step and first stage */
    {
        let mut m = ark_mem.borrow_mut();
        m.initsetup = SUNFALSE;
        m.firststage = SUNFALSE;
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  arkHandleFailure

  This routine prints error messages for all cases of failure by
  arkHin and ark_step. It returns to ARKODE the value that ARKODE
  is to return to the user.
  ---------------------------------------------------------------*/
pub fn arkHandleFailure(ark_mem: &ARKodeMem, flag: i32) -> i32 {
    let (tcur, h) = {
        let m = ark_mem.borrow();
        (m.tcur, m.h)
    };

    /* Depending on flag, print error message and return error flag */
    match flag {
        ARK_ERR_FAILURE => {
            arkProcessError(
                Some(ark_mem),
                ARK_ERR_FAILURE,
                line!() as i32,
                "arkHandleFailure",
                file!(),
                &MSG_ARK_ERR_FAILS(tcur, h),
            );
        }
        ARK_CONV_FAILURE => {
            arkProcessError(
                Some(ark_mem),
                ARK_CONV_FAILURE,
                line!() as i32,
                "arkHandleFailure",
                file!(),
                &MSG_ARK_CONV_FAILS(tcur, h),
            );
        }
        ARK_LSETUP_FAIL => {
            arkProcessError(
                Some(ark_mem),
                ARK_LSETUP_FAIL,
                line!() as i32,
                "arkHandleFailure",
                file!(),
                &MSG_ARK_SETUP_FAILED(tcur),
            );
        }
        ARK_LSOLVE_FAIL => {
            arkProcessError(
                Some(ark_mem),
                ARK_LSOLVE_FAIL,
                line!() as i32,
                "arkHandleFailure",
                file!(),
                &MSG_ARK_SOLVE_FAILED(tcur),
            );
        }
        ARK_RHSFUNC_FAIL => {
            arkProcessError(
                Some(ark_mem),
                ARK_RHSFUNC_FAIL,
                line!() as i32,
                "arkHandleFailure",
                file!(),
                &MSG_ARK_RHSFUNC_FAILED(tcur),
            );
        }
        ARK_UNREC_RHSFUNC_ERR => {
            arkProcessError(
                Some(ark_mem),
                ARK_UNREC_RHSFUNC_ERR,
                line!() as i32,
                "arkHandleFailure",
                file!(),
                &MSG_ARK_RHSFUNC_UNREC(tcur),
            );
        }
        ARK_REPTD_RHSFUNC_ERR => {
            arkProcessError(
                Some(ark_mem),
                ARK_REPTD_RHSFUNC_ERR,
                line!() as i32,
                "arkHandleFailure",
                file!(),
                &MSG_ARK_RHSFUNC_REPTD(tcur),
            );
        }
        ARK_RTFUNC_FAIL => {
            arkProcessError(
                Some(ark_mem),
                ARK_RTFUNC_FAIL,
                line!() as i32,
                "arkHandleFailure",
                file!(),
                &MSG_ARK_RTFUNC_FAILED(tcur),
            );
        }
        ARK_TOO_CLOSE => {
            arkProcessError(
                Some(ark_mem),
                ARK_TOO_CLOSE,
                line!() as i32,
                "arkHandleFailure",
                file!(),
                MSG_ARK_TOO_CLOSE,
            );
        }
        ARK_CONSTR_FAIL => {
            arkProcessError(
                Some(ark_mem),
                ARK_CONSTR_FAIL,
                line!() as i32,
                "arkHandleFailure",
                file!(),
                &MSG_ARK_FAILED_CONSTR(tcur),
            );
        }
        ARK_MASSSOLVE_FAIL => {
            arkProcessError(
                Some(ark_mem),
                ARK_MASSSOLVE_FAIL,
                line!() as i32,
                "arkHandleFailure",
                file!(),
                MSG_ARK_MASSSOLVE_FAIL,
            );
        }
        ARK_NLS_SETUP_FAIL => {
            arkProcessError(
                Some(ark_mem),
                ARK_NLS_SETUP_FAIL,
                line!() as i32,
                "arkHandleFailure",
                file!(),
                &format!(
                    "At t = {} the nonlinear solver setup failed unrecoverably",
                    sundials_core::sundials_utils::sun_format_g(tcur)
                ),
            );
        }
        ARK_VECTOROP_ERR => {
            arkProcessError(
                Some(ark_mem),
                ARK_VECTOROP_ERR,
                line!() as i32,
                "arkHandleFailure",
                file!(),
                &MSG_ARK_VECTOROP_ERR(tcur),
            );
        }
        ARK_INNERSTEP_FAIL => {
            arkProcessError(
                Some(ark_mem),
                ARK_INNERSTEP_FAIL,
                line!() as i32,
                "arkHandleFailure",
                file!(),
                &MSG_ARK_INNERSTEP_FAILED(tcur),
            );
        }
        ARK_NLS_OP_ERR => {
            arkProcessError(
                Some(ark_mem),
                ARK_NLS_OP_ERR,
                line!() as i32,
                "arkHandleFailure",
                file!(),
                &MSG_ARK_NLS_FAIL(tcur),
            );
        }
        ARK_USER_PREDICT_FAIL => {
            arkProcessError(
                Some(ark_mem),
                ARK_USER_PREDICT_FAIL,
                line!() as i32,
                "arkHandleFailure",
                file!(),
                &MSG_ARK_USER_PREDICT_FAIL(tcur),
            );
        }
        ARK_POSTPROCESS_STEP_FAIL => {
            arkProcessError(
                Some(ark_mem),
                ARK_POSTPROCESS_STEP_FAIL,
                line!() as i32,
                "arkHandleFailure",
                file!(),
                &MSG_ARK_POSTPROCESS_STEP_FAIL(tcur),
            );
        }
        ARK_POSTPROCESS_STAGE_FAIL => {
            arkProcessError(
                Some(ark_mem),
                ARK_POSTPROCESS_STAGE_FAIL,
                line!() as i32,
                "arkHandleFailure",
                file!(),
                &MSG_ARK_POSTPROCESS_STAGE_FAIL(tcur),
            );
        }
        ARK_PRESTEPFN_FAIL => {
            arkProcessError(
                Some(ark_mem),
                ARK_PRESTEPFN_FAIL,
                line!() as i32,
                "arkHandleFailure",
                file!(),
                &MSG_ARK_PRESTEPFN_FAIL(tcur),
            );
        }
        ARK_POSTSTEPFN_FAIL => {
            arkProcessError(
                Some(ark_mem),
                ARK_POSTSTEPFN_FAIL,
                line!() as i32,
                "arkHandleFailure",
                file!(),
                &MSG_ARK_POSTSTEPFN_FAIL(tcur),
            );
        }
        ARK_PRERHSFN_FAIL => {
            arkProcessError(
                Some(ark_mem),
                ARK_PRERHSFN_FAIL,
                line!() as i32,
                "arkHandleFailure",
                file!(),
                &MSG_ARK_PRERHSFN_FAIL(tcur),
            );
        }
        ARK_INTERP_FAIL => {
            arkProcessError(
                Some(ark_mem),
                ARK_INTERP_FAIL,
                line!() as i32,
                "arkHandleFailure",
                file!(),
                &format!(
                    "At t = {} the interpolation module failed unrecoverably",
                    sundials_core::sundials_utils::sun_format_g(tcur)
                ),
            );
        }
        ARK_INVALID_TABLE => {
            arkProcessError(
                Some(ark_mem),
                ARK_INVALID_TABLE,
                line!() as i32,
                "arkHandleFailure",
                file!(),
                "ARKODE was provided an invalid method table",
            );
        }
        ARK_RELAX_FAIL => {
            arkProcessError(
                Some(ark_mem),
                ARK_RELAX_FAIL,
                line!() as i32,
                "arkHandleFailure",
                file!(),
                &format!(
                    "At t = {} the relaxation module failed",
                    sundials_core::sundials_utils::sun_format_g(tcur)
                ),
            );
        }
        ARK_RELAX_MEM_NULL => {
            arkProcessError(
                Some(ark_mem),
                ARK_RELAX_MEM_NULL,
                line!() as i32,
                "arkHandleFailure",
                file!(),
                "The ARKODE relaxation module memory is NULL",
            );
        }
        ARK_RELAX_FUNC_FAIL => {
            arkProcessError(
                Some(ark_mem),
                ARK_RELAX_FUNC_FAIL,
                line!() as i32,
                "arkHandleFailure",
                file!(),
                "The relaxation function failed unrecoverably",
            );
        }
        ARK_RELAX_JAC_FAIL => {
            arkProcessError(
                Some(ark_mem),
                ARK_RELAX_JAC_FAIL,
                line!() as i32,
                "arkHandleFailure",
                file!(),
                "The relaxation Jacobian failed unrecoverably",
            );
        }
        ARK_ADJ_RECOMPUTE_FAIL => {
            arkProcessError(
                Some(ark_mem),
                ARK_ADJ_RECOMPUTE_FAIL,
                line!() as i32,
                "arkHandleFailure",
                file!(),
                "The forward recomputation of step failed unrecoverably",
            );
        }
        ARK_ADJ_CHECKPOINT_FAIL => {
            arkProcessError(
                Some(ark_mem),
                ARK_ADJ_CHECKPOINT_FAIL,
                line!() as i32,
                "arkHandleFailure",
                file!(),
                "A checkpoint operation failed unrecoverably",
            );
        }
        ARK_SUNADJSTEPPER_ERR => {
            arkProcessError(
                Some(ark_mem),
                ARK_SUNADJSTEPPER_ERR,
                line!() as i32,
                "arkHandleFailure",
                file!(),
                "A SUNAdjStepper operation failed unrecoverably",
            );
        }
        ARK_DOMEIG_FAIL => {
            arkProcessError(
                Some(ark_mem),
                ARK_DOMEIG_FAIL,
                line!() as i32,
                "arkHandleFailure",
                file!(),
                "The dominant eigenvalue function failed unrecoverably",
            );
        }
        ARK_MAX_STAGE_LIMIT_FAIL => {
            arkProcessError(
                Some(ark_mem),
                ARK_MAX_STAGE_LIMIT_FAIL,
                line!() as i32,
                "arkHandleFailure",
                file!(),
                "The max stage limit failed unrecoverably",
            );
        }
        ARK_SUNSTEPPER_ERR => {
            arkProcessError(
                Some(ark_mem),
                ARK_SUNSTEPPER_ERR,
                line!() as i32,
                "arkHandleFailure",
                file!(),
                "An inner SUNStepper error occurred",
            );
        }
        _ => {
            /* This return should never happen */
            arkProcessError(
                Some(ark_mem),
                ARK_UNRECOGNIZED_ERROR,
                line!() as i32,
                "arkHandleFailure",
                file!(),
                "ARKODE encountered an unrecognized error. Please report this to the Sundials developers at sundials-users@llnl.gov",
            );
            return ARK_UNRECOGNIZED_ERROR;
        }
    }

    flag
}

/*---------------------------------------------------------------
  arkEwtSetSS

  This routine is responsible for setting the error weight vector
  ewt as follows:

  ewt[i] = 1 / (reltol * SUNRabs(ycur[i]) + abstol), i=0,...,neq-1

  When the absolute tolerance is zero, it tests for non-positive
  components before inverting. arkEwtSetSS returns 0 if ewt is
  successfully set to a positive vector and -1 otherwise. In the
  latter case, ewt is considered undefined.
  ---------------------------------------------------------------*/
pub fn arkEwtSetSS(
    ycur: &N_Vector,
    weight: &N_Vector,
    arkode_mem: &mut Option<Box<dyn std::any::Any>>,
) -> i32 {
    /* arkode_mem points to ark_mem here (a boxed ARKodeMem handle clone;
    C's cast of a NULL/foreign pointer is UB -> deterministic panic) */
    let ark_mem = arkode_mem
        .as_ref()
        .and_then(|b| b.downcast_ref::<ARKodeMem>())
        .cloned()
        .expect("arkEwtSetSS data holds ARKodeMem");

    let (tempv1, reltol, Sabstol, atolmin0) = {
        let m = ark_mem.borrow();
        (
            m.tempv1.clone().expect("tempv1 allocated"),
            m.reltol,
            m.Sabstol,
            m.atolmin0,
        )
    };
    N_VAbs(ycur, &tempv1);
    N_VScale(reltol, &tempv1, &tempv1);
    N_VAddConst(&tempv1, Sabstol, &tempv1);
    if atolmin0 && N_VMin(&tempv1) <= ZERO {
        return -1;
    }
    N_VInv(&tempv1, weight);
    0
}

/*---------------------------------------------------------------
  arkEwtSetSV

  This routine is responsible for setting the error weight vector
  ewt as follows:

  ewt[i] = 1 / (reltol * SUNRabs(ycur[i]) + abstol[i]), i=0,...,neq-1

  When any absolute tolerance is zero, it tests for non-positive
  components before inverting. arkEwtSetSV returns 0 if ewt is
  successfully set to a positive vector and -1 otherwise. In the
  latter case, ewt is considered undefined.
  ---------------------------------------------------------------*/
pub fn arkEwtSetSV(
    ycur: &N_Vector,
    weight: &N_Vector,
    arkode_mem: &mut Option<Box<dyn std::any::Any>>,
) -> i32 {
    let ark_mem = arkode_mem
        .as_ref()
        .and_then(|b| b.downcast_ref::<ARKodeMem>())
        .cloned()
        .expect("arkEwtSetSV data holds ARKodeMem");

    let (tempv1, reltol, Vabstol, atolmin0) = {
        let m = ark_mem.borrow();
        (
            m.tempv1.clone().expect("tempv1 allocated"),
            m.reltol,
            m.Vabstol.clone().expect("Vabstol allocated"),
            m.atolmin0,
        )
    };
    N_VAbs(ycur, &tempv1);
    N_VLinearSum(reltol, &tempv1, ONE, &Vabstol, &tempv1);
    if atolmin0 && N_VMin(&tempv1) <= ZERO {
        return -1;
    }
    N_VInv(&tempv1, weight);
    0
}

/*---------------------------------------------------------------
  arkEwtSetSmallReal

  This routine is responsible for setting the error weight vector
  ewt as follows:

  ewt[i] = SUN_SMALL_REAL

  This is routine is only used with explicit time stepping with
  a fixed step size to avoid a potential too much error return
  to the user.
  ---------------------------------------------------------------*/
pub fn arkEwtSetSmallReal(
    _ycur: &N_Vector, /* SUNDIALS_MAYBE_UNUSED in C */
    weight: &N_Vector,
    _arkode_mem: &mut Option<Box<dyn std::any::Any>>, /* SUNDIALS_MAYBE_UNUSED in C */
) -> i32 {
    N_VConst(SUN_SMALL_REAL, weight);
    ARK_SUCCESS
}

/*---------------------------------------------------------------
  arkRwtSetSS

  This routine sets rwt as described above in the case tol_type = ARK_SS.
  When the absolute tolerance is zero, it tests for non-positive
  components before inverting. arkRwtSetSS returns 0 if rwt is
  successfully set to a positive vector and -1 otherwise. In the
  latter case, rwt is considered undefined.
  ---------------------------------------------------------------*/
pub fn arkRwtSetSS(ark_mem: &ARKodeMem, My: &N_Vector, weight: &N_Vector) -> i32 {
    let (tempv1, reltol, SRabstol, Ratolmin0) = {
        let m = ark_mem.borrow();
        (
            m.tempv1.clone().expect("tempv1 allocated"),
            m.reltol,
            m.SRabstol,
            m.Ratolmin0,
        )
    };
    N_VAbs(My, &tempv1);
    N_VScale(reltol, &tempv1, &tempv1);
    N_VAddConst(&tempv1, SRabstol, &tempv1);
    if Ratolmin0 && N_VMin(&tempv1) <= ZERO {
        return -1;
    }
    N_VInv(&tempv1, weight);
    0
}

/*---------------------------------------------------------------
  arkRwtSetSV

  This routine sets rwt as described above in the case tol_type = ARK_SV.
  When any absolute tolerance is zero, it tests for non-positive
  components before inverting. arkRwtSetSV returns 0 if rwt is
  successfully set to a positive vector and -1 otherwise. In the
  latter case, rwt is considered undefined.
  ---------------------------------------------------------------*/
pub fn arkRwtSetSV(ark_mem: &ARKodeMem, My: &N_Vector, weight: &N_Vector) -> i32 {
    let (tempv1, reltol, VRabstol, Ratolmin0) = {
        let m = ark_mem.borrow();
        (
            m.tempv1.clone().expect("tempv1 allocated"),
            m.reltol,
            m.VRabstol.clone().expect("VRabstol allocated"),
            m.Ratolmin0,
        )
    };
    N_VAbs(My, &tempv1);
    N_VLinearSum(reltol, &tempv1, ONE, &VRabstol, &tempv1);
    if Ratolmin0 && N_VMin(&tempv1) <= ZERO {
        return -1;
    }
    N_VInv(&tempv1, weight);
    0
}

/*---------------------------------------------------------------
  arkPredict_MaximumOrder

  This routine predicts the nonlinear implicit stage solution
  using the ARKode interpolation module.  This uses the
  highest-degree interpolant supported by the module (stored
  in the interpolation module).
  ---------------------------------------------------------------*/
pub fn arkPredict_MaximumOrder(
    ark_mem: &ARKodeMem,
    tau: sunrealtype,
    yguess: &N_Vector,
) -> i32 {
    /* verify that ark_mem and interpolation structure are provided.
    The C `ark_mem == NULL` guard cannot fire: the Rust handle is a
    `&ARKodeMem`. */
    let interp = ark_mem.borrow().interp.clone();
    let interp = match interp {
        None => {
            arkProcessError(
                Some(ark_mem),
                ARK_MEM_NULL,
                line!() as i32,
                "arkPredict_MaximumOrder",
                file!(),
                "ARKodeInterpMem structure is NULL",
            );
            return ARK_MEM_NULL;
        }
        Some(i) => i,
    };

    /* call the interpolation module to do the work */
    arkInterpEvaluate(
        ark_mem,
        Some(&interp),
        tau,
        0,
        ARK_INTERP_MAX_DEGREE,
        yguess,
    )
}

/*---------------------------------------------------------------
  arkPredict_VariableOrder

  This routine predicts the nonlinear implicit stage solution
  using the ARKODE interpolation module.  The degree of the
  interpolant is based on the level of extrapolation outside the
  preceding time step.
  ---------------------------------------------------------------*/
pub fn arkPredict_VariableOrder(
    ark_mem: &ARKodeMem,
    tau: sunrealtype,
    yguess: &N_Vector,
) -> i32 {
    let ord: i32;
    let tau_tol: sunrealtype = HALF;
    let tau_tol2: sunrealtype = 0.75;

    /* verify that ark_mem and interpolation structure are provided */
    let interp = ark_mem.borrow().interp.clone();
    let interp = match interp {
        None => {
            arkProcessError(
                Some(ark_mem),
                ARK_MEM_NULL,
                line!() as i32,
                "arkPredict_VariableOrder",
                file!(),
                "ARKodeInterpMem structure is NULL",
            );
            return ARK_MEM_NULL;
        }
        Some(i) => i,
    };

    /* set the polynomial order based on tau input */
    if tau <= tau_tol {
        ord = 3;
    } else if tau <= tau_tol2 {
        ord = 2;
    } else {
        ord = 1;
    }

    /* call the interpolation module to do the work */
    arkInterpEvaluate(ark_mem, Some(&interp), tau, 0, ord, yguess)
}

/*---------------------------------------------------------------
  arkPredict_CutoffOrder

  This routine predicts the nonlinear implicit stage solution
  using the ARKODE interpolation module.  If the level of
  extrapolation is small enough, it uses the maximum degree
  polynomial available (stored in the interpolation module
  structure); otherwise it uses a linear polynomial.
  ---------------------------------------------------------------*/
pub fn arkPredict_CutoffOrder(ark_mem: &ARKodeMem, tau: sunrealtype, yguess: &N_Vector) -> i32 {
    let ord: i32;
    let tau_tol: sunrealtype = HALF;

    /* verify that ark_mem and interpolation structure are provided */
    let interp = ark_mem.borrow().interp.clone();
    let interp = match interp {
        None => {
            arkProcessError(
                Some(ark_mem),
                ARK_MEM_NULL,
                line!() as i32,
                "arkPredict_CutoffOrder",
                file!(),
                "ARKodeInterpMem structure is NULL",
            );
            return ARK_MEM_NULL;
        }
        Some(i) => i,
    };

    /* set the polynomial order based on tau input */
    if tau <= tau_tol {
        ord = ARK_INTERP_MAX_DEGREE;
    } else {
        ord = 1;
    }

    /* call the interpolation module to do the work */
    arkInterpEvaluate(ark_mem, Some(&interp), tau, 0, ord, yguess)
}

/*---------------------------------------------------------------
  arkPredict_Bootstrap

  This routine predicts the nonlinear implicit stage solution
  using a quadratic Hermite interpolating polynomial, based on
  the data {y_n, f(t_n,y_n), f(t_n+hj,z_j)}.

  Note: we assume that ftemp = f(t_n+hj,z_j) can be computed via
     N_VLinearCombination(nvec, cvals, Xvecs, ftemp),
  i.e. the inputs cvals[0:nvec-1] and Xvecs[0:nvec-1] may be
  combined to form f(t_n+hj,z_j).

  PORT NOTE (call-site contract): C requires the caller's `cvals` and
  `Xvecs` scratch arrays to hold at least `nvec + 2` slots (steppers size
  them with `nfusedopvecs`).  `cvals` is therefore `&mut [sunrealtype]`
  and MUST already be that long.  `Xvecs` is the locked "handle scratch
  rebuilt on demand" `Vec<N_Vector>` (an `N_Vector` array cannot be left
  uninitialized in safe Rust), so it is taken as `&mut Vec<N_Vector>` and
  grown here to `nvec + 2` if the caller pushed only `nvec` handles; every
  slot 0..nvec+2 is written before use, so the filler is never observable.
  The in-place forward shift is transcribed literally, including its
  self-overwriting behavior for `i >= 2` (unreachable: `nvec <= 2` at both
  upstream call sites).
  ---------------------------------------------------------------*/
pub fn arkPredict_Bootstrap(
    ark_mem: &ARKodeMem,
    hj: sunrealtype,
    tau: sunrealtype,
    nvec: i32,
    cvals: &mut Vec<sunrealtype>,
    Xvecs: &mut Vec<N_Vector>,
    yguess: &N_Vector,
) -> i32 {
    /* verify that ark_mem and interpolation structure are provided */
    if ark_mem.borrow().interp.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "arkPredict_Bootstrap",
            file!(),
            "ARKodeInterpMem structure is NULL",
        );
        return ARK_MEM_NULL;
    }

    let (yn, fn_) = {
        let m = ark_mem.borrow();
        (
            m.yn.clone().expect("yn allocated"),
            m.fn_.clone().expect("fn allocated"),
        )
    };

    /* set coefficients for Hermite interpolant */
    let a0 = ONE;
    let a2 = tau * tau / TWO / hj;
    let a1 = tau - a2;

    /* set arrays for fused vector operation; shift inputs for
    f(t_n+hj,z_j) to end of queue */
    /* C passes the stepper's `cvals`/`Xvecs` fused-op workspace, which is
    always allocated with at least `nvec+2` slots; the Rust call sites build
    the queue as a `Vec` holding exactly `nvec` entries, so grow it here. */
    let n = nvec as usize;
    if cvals.len() < n + 2 {
        cvals.resize(n + 2, ZERO);
    }
    if Xvecs.len() < n + 2 {
        Xvecs.resize(n + 2, yn.clone());
    }
    for i in 0..n {
        cvals[2 + i] = a2 * cvals[i];
        Xvecs[2 + i] = Xvecs[i].clone();
    }
    cvals[0] = a0;
    Xvecs[0] = yn;
    cvals[1] = a1;
    Xvecs[1] = fn_;

    /* call fused vector operation to compute prediction */
    let retval = N_VLinearCombination(nvec + 2, cvals, &Xvecs[..], yguess);
    if retval != 0 {
        return ARK_VECTOROP_ERR;
    }
    ARK_SUCCESS
}

/*---------------------------------------------------------------
  arkCheckConvergence

  This routine checks the return flag from the time-stepper's
  "step" routine for algebraic solver convergence issues.

  Returns ARK_SUCCESS (0) if successful, PREDICT_AGAIN (>0)
  on a recoverable convergence failure, or a relevant
  nonrecoverable failure flag (<0).
  --------------------------------------------------------------*/
pub fn arkCheckConvergence(ark_mem: &ARKodeMem, nflagPtr: &mut i32, ncfPtr: &mut i32) -> i32 {
    /* If nonlinear solver succeeded, return with ARK_SUCCESS */
    if *nflagPtr == ARK_SUCCESS {
        return ARK_SUCCESS;
    }
    /* Returns with an ARK_RETRY_STEP flag occur at a stage well before
    any algebraic solvers are involved. On the other hand,
    the arkCheckConvergence function handles the results from algebraic
    solvers, which never take place with an ARK_RETRY_STEP flag.
    Therefore, we immediately return from arkCheckConvergence,
    as it is irrelevant in the case of an ARK_RETRY_STEP */
    if *nflagPtr == ARK_RETRY_STEP {
        return ARK_RETRY_STEP;
    }

    /* The nonlinear soln. failed; increment ncfn */
    ark_mem.borrow_mut().ncfn += 1;

    /* If fixed time stepping, then return with convergence failure */
    if ark_mem.borrow().fixedstep {
        return ARK_CONV_FAILURE;
    }

    /* Otherwise, access adaptivity structure */
    if ark_mem.borrow().hadapt_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "arkCheckConvergence",
            file!(),
            MSG_ARKADAPT_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    /* Return if lsetup, lsolve, or rhs failed unrecoverably */
    if *nflagPtr < 0 {
        if *nflagPtr == ARK_LSETUP_FAIL {
            return ARK_LSETUP_FAIL;
        } else if *nflagPtr == ARK_LSOLVE_FAIL {
            return ARK_LSOLVE_FAIL;
        } else if *nflagPtr == ARK_RHSFUNC_FAIL {
            return ARK_RHSFUNC_FAIL;
        } else {
            return ARK_NLS_OP_ERR;
        }
    }

    /* At this point, nflag = CONV_FAIL or RHSFUNC_RECVR; increment ncf */
    *ncfPtr += 1;
    {
        let mut guard = ark_mem.borrow_mut();
        let m = &mut *guard;
        let hadapt_mem = m.hadapt_mem.as_mut().expect("hadapt_mem allocated");
        hadapt_mem.etamax = ONE;
    }

    /* If we had maxncf failures, or if |h| = hmin,
    return ARK_CONV_FAILURE or ARK_REPTD_RHSFUNC_ERR. */
    let (maxncf, h, hmin) = {
        let m = ark_mem.borrow();
        (m.maxncf, m.h, m.hmin)
    };
    if (*ncfPtr == maxncf) || (SUNRabs(h) <= hmin * ONEPSM) {
        if *nflagPtr == CONV_FAIL {
            return ARK_CONV_FAILURE;
        }
        if *nflagPtr == RHSFUNC_RECVR {
            return ARK_REPTD_RHSFUNC_ERR;
        }
    }

    /* Reduce step size due to convergence failure */
    {
        let mut guard = ark_mem.borrow_mut();
        let m = &mut *guard;
        let etacf = m.hadapt_mem.as_ref().expect("hadapt_mem allocated").etacf;
        m.eta = etacf;
    }

    /* Signal for Jacobian/preconditioner setup */
    *nflagPtr = PREV_CONV_FAIL;

    /* Return to reattempt the step */
    PREDICT_AGAIN
}

/*---------------------------------------------------------------
  arkCheckConstraints

  This routine determines if the constraints of the problem
  are satisfied by the proposed step

  Returns ARK_SUCCESS if successful, otherwise CONSTR_RECVR
  --------------------------------------------------------------*/
pub fn arkCheckConstraints(ark_mem: &ARKodeMem, constrfails: &mut i32, nflag: &mut i32) -> i32 {
    let (mm, tmp, constraints, ycur, yn) = {
        let m = ark_mem.borrow();
        (
            m.tempv4.clone().expect("tempv4 allocated"),
            m.tempv3.clone().expect("tempv3 allocated"),
            m.constraints.clone().expect("constraints set"),
            m.ycur.clone().expect("ycur set"),
            m.yn.clone().expect("yn allocated"),
        )
    };

    /* Check constraints and get mask vector mm for where constraints failed */
    let constraintsPassed = N_VConstrMask(&constraints, &ycur, &mm);
    if constraintsPassed {
        return ARK_SUCCESS;
    }

    /* Constraints not met */

    /* Update total fails and fails in current step */
    ark_mem.borrow_mut().nconstrfails += 1;
    *constrfails += 1;

    /* Return with error if reached max fails in a step */
    if *constrfails == ark_mem.borrow().maxconstrfails {
        return ARK_CONSTR_FAIL;
    }

    /* Return with error if using fixed step sizes */
    if ark_mem.borrow().fixedstep {
        return ARK_CONSTR_FAIL;
    }

    /* Return with error if |h| == hmin */
    let (h, hmin) = {
        let m = ark_mem.borrow();
        (m.h, m.hmin)
    };
    if SUNRabs(h) <= hmin * ONEPSM {
        return ARK_CONSTR_FAIL;
    }

    /* Reduce h by computing eta = h'/h */
    N_VLinearSum(ONE, &yn, -ONE, &ycur, &tmp);
    N_VProd(&mm, &tmp, &tmp);
    let eta = 0.9 * N_VMinQuotient(&yn, &tmp);
    let eta = SUNMAX(eta, TENTH);
    ark_mem.borrow_mut().eta = eta;

    /* Signal for Jacobian/preconditioner setup */
    *nflag = PREV_CONV_FAIL;

    /* Return to reattempt the step */
    CONSTR_RECVR
}

/*---------------------------------------------------------------
  arkCheckTemporalError

  This routine performs the local error test for the method.
  The weighted local error norm dsm is passed in.  This value is
  used to predict the next step to attempt based on dsm.
  The test dsm <= 1 is made, and if this fails then additional
  checks are performed based on the number of successive error
  test failures.

  Returns ARK_SUCCESS if the test passes.

  If the test fails:
    - if maxnef error test failures have occurred or if
      SUNRabs(h) = hmin, we return ARK_ERR_FAILURE.
    - otherwise: set *nflagPtr to PREV_ERR_FAIL, and
      return TRY_AGAIN.
  --------------------------------------------------------------*/
pub fn arkCheckTemporalError(
    ark_mem: &ARKodeMem,
    nflagPtr: &mut i32,
    nefPtr: &mut i32,
    dsm: sunrealtype,
) -> i32 {
    /* Access hadapt_mem structure */
    if ark_mem.borrow().hadapt_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "arkCheckTemporalError",
            file!(),
            MSG_ARKADAPT_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    /* consider change of step size for next step attempt (may be
    larger/smaller than current step, depending on dsm) */
    let (tn, h, ycur) = {
        let m = ark_mem.borrow();
        (m.tn, m.h, m.ycur.clone().expect("ycur set"))
    };
    let ttmp = if dsm <= ONE { tn + h } else { tn };
    let retval = arkAdapt(ark_mem, &ycur, ttmp, h, dsm);
    if retval != ARK_SUCCESS {
        return ARK_ERR_FAILURE;
    }

    /* if we've made it here then no nonrecoverable failures occurred; someone above
    has recommended an 'eta' value for the next step -- enforce bounds on that value
    and set upcoming step size */
    {
        let mut guard = ark_mem.borrow_mut();
        let m = &mut *guard;
        let etamax = m.hadapt_mem.as_ref().expect("hadapt_mem allocated").etamax;
        m.eta = SUNMIN(m.eta, etamax);
        m.eta = SUNMAX(m.eta, m.hmin / SUNRabs(m.h));
        let denom = SUNMAX(ONE, SUNRabs(m.h) * m.hmax_inv * m.eta);
        m.eta /= denom;
    }

    /* If est. local error norm dsm passes test, return ARK_SUCCESS */
    if dsm <= ONE {
        return ARK_SUCCESS;
    }

    /* Test failed; increment counters, set nflag */
    *nefPtr += 1;
    ark_mem.borrow_mut().netf += 1;
    *nflagPtr = PREV_ERR_FAIL;

    /* At maxnef failures, return ARK_ERR_FAILURE */
    if *nefPtr == ark_mem.borrow().maxnef {
        return ARK_ERR_FAILURE;
    }

    /* Set etamax=1 to prevent step size increase at end of this step */
    {
        let mut guard = ark_mem.borrow_mut();
        let m = &mut *guard;
        let hadapt_mem = m.hadapt_mem.as_mut().expect("hadapt_mem allocated");
        hadapt_mem.etamax = ONE;
    }

    /* Enforce failure bounds on eta */
    {
        let mut guard = ark_mem.borrow_mut();
        let m = &mut *guard;
        let (small_nef, etamxf) = {
            let hadapt_mem = m.hadapt_mem.as_ref().expect("hadapt_mem allocated");
            (hadapt_mem.small_nef, hadapt_mem.etamxf)
        };
        if *nefPtr >= small_nef {
            m.eta = SUNMIN(m.eta, etamxf);
        }

        /* Enforce min/max step bounds once again due to adjustments above */
        let etamax = m.hadapt_mem.as_ref().expect("hadapt_mem allocated").etamax;
        m.eta = SUNMIN(m.eta, etamax);
        m.eta = SUNMAX(m.eta, m.hmin / SUNRabs(m.h));
        let denom = SUNMAX(ONE, SUNRabs(m.h) * m.hmax_inv * m.eta);
        m.eta /= denom;
    }

    TRY_AGAIN
}

/*---------------------------------------------------------------
  arkAllocVec and arkAllocVecArray:

  These routines allocate (respectively) single vector or a vector
  array based on a template vector.  If the target vector or vector
  array already exists it is left alone; otherwise it is allocated
  by cloning the input vector.

  This routine also updates the optional outputs lrw and liw, which
  are (respectively) the lengths of the overall ARKODE real and
  integer work spaces.

  SUNTRUE is returned if the allocation is successful (or if the
  target vector or vector array already exists) otherwise SUNFALSE
  is returned.

  PORT NOTE: C passes `&ark_mem->ewt` etc. -- an interior pointer into
  the mem that these routines also mutate.  Rust call sites must
  `Option::take` the field out of the mem, call, and store the result
  back (the failure path in C leaves `*v == NULL`, so restoring `None`
  is equivalent).
  ---------------------------------------------------------------*/
pub fn arkAllocVec(
    ark_mem: &ARKodeMem,
    tmpl: &N_Vector,
    v: &mut Option<N_Vector>,
) -> sunbooleantype {
    /* return failure if N_VClone or N_VDestroy is not implemented */
    {
        let ops = tmpl.ops.borrow();
        if ops.nvclone.is_none() || ops.nvdestroy.is_none() {
            return SUNFALSE;
        }
    }

    /* allocate the new vector if necessary */
    if v.is_none() {
        *v = N_VClone(tmpl);
        if v.is_none() {
            arkFreeVectors(ark_mem);
            return SUNFALSE;
        } else {
            let mut guard = ark_mem.borrow_mut();
            let m = &mut *guard;
            m.lrw += m.lrw1;
            m.liw += m.liw1;
        }
    }
    SUNTRUE
}

pub fn arkAllocVecArray(
    count: i32,
    tmpl: &N_Vector,
    v: &mut Vec<N_Vector>,
    lrw1: sunindextype,
    lrw: &mut i64,
    liw1: sunindextype,
    liw: &mut i64,
) -> sunbooleantype {
    /* allocate the new vector array if necessary */
    if v.is_empty() {
        match N_VCloneVectorArray(count, tmpl) {
            None => return SUNFALSE,
            Some(vs) => *v = vs,
        }
        *lrw += count as i64 * lrw1;
        *liw += count as i64 * liw1;
    }
    SUNTRUE
}

/*---------------------------------------------------------------
  arkFreeVec and arkFreeVecArray:

  These routines (respectively) free a single vector or a vector
  array. If the target vector or vector array is already NULL it
  is left alone; otherwise it is freed and the optional outputs
  lrw and liw are updated accordingly.
  ---------------------------------------------------------------*/
pub fn arkFreeVec(ark_mem: &ARKodeMem, v: &mut Option<N_Vector>) {
    if v.is_some() {
        N_VDestroy(v.take().expect("vector present"));
        let mut guard = ark_mem.borrow_mut();
        let m = &mut *guard;
        m.lrw -= m.lrw1;
        m.liw -= m.liw1;
    }
}

pub fn arkFreeVecArray(
    count: i32,
    v: &mut Vec<N_Vector>,
    lrw1: sunindextype,
    lrw: &mut i64,
    liw1: sunindextype,
    liw: &mut i64,
) {
    if !v.is_empty() {
        N_VDestroyVectorArray(std::mem::take(v), count);
        *lrw -= count as i64 * lrw1;
        *liw -= count as i64 * liw1;
    }
}

/*---------------------------------------------------------------
  arkResizeVec and arkResizeVecArray:

  This routines (respectively) resize a single vector or a vector
  array based on a template vector. If the ARKVecResizeFn function
  is non-NULL, then it calls that routine to perform the resize;
  otherwise it deallocates and reallocates the target vector or
  vector array based on the template vector. These routines also
  updates the optional outputs lrw and liw, which are
  (respectively) the lengths of the overall ARKODE real and
  integer work spaces.

  SUNTRUE is returned if the resize is successful otherwise
  SUNFALSE is returned.
  ---------------------------------------------------------------*/
pub fn arkResizeVec(
    ark_mem: &ARKodeMem,
    resize: Option<ARKVecResizeFn>,
    resize_data: &mut Option<Box<dyn std::any::Any>>,
    lrw_diff: sunindextype,
    liw_diff: sunindextype,
    tmpl: &N_Vector,
    v: &mut Option<N_Vector>,
) -> sunbooleantype {
    if v.is_some() {
        match resize {
            None => {
                N_VDestroy(v.take().expect("vector present"));
                *v = None;
                *v = N_VClone(tmpl);
                if v.is_none() {
                    arkProcessError(
                        Some(ark_mem),
                        ARK_MEM_FAIL,
                        line!() as i32,
                        "arkResizeVec",
                        file!(),
                        "Unable to clone vector",
                    );
                    return SUNFALSE;
                }
            }
            Some(resize) => {
                let vv = v.as_ref().expect("vector present").clone();
                if resize(&vv, tmpl, resize_data) != 0 {
                    arkProcessError(
                        Some(ark_mem),
                        ARK_MEM_FAIL,
                        line!() as i32,
                        "arkResizeVec",
                        file!(),
                        MSG_ARK_RESIZE_FAIL,
                    );
                    return SUNFALSE;
                }
            }
        }
        let mut guard = ark_mem.borrow_mut();
        let m = &mut *guard;
        m.lrw += lrw_diff;
        m.liw += liw_diff;
    }
    SUNTRUE
}

pub fn arkResizeVecArray(
    resize: Option<ARKVecResizeFn>,
    resize_data: &mut Option<Box<dyn std::any::Any>>,
    count: i32,
    tmpl: &N_Vector,
    v: &mut Vec<N_Vector>,
    lrw_diff: sunindextype,
    lrw: &mut i64,
    liw_diff: sunindextype,
    liw: &mut i64,
) -> sunbooleantype {
    if !v.is_empty() {
        match resize {
            None => {
                N_VDestroyVectorArray(std::mem::take(v), count);
                match N_VCloneVectorArray(count, tmpl) {
                    None => return SUNFALSE,
                    Some(vs) => *v = vs,
                }
            }
            Some(resize) => {
                for i in 0..count as usize {
                    let vi = v[i].clone();
                    if resize(&vi, tmpl, resize_data) != 0 {
                        return SUNFALSE;
                    }
                }
            }
        }
        *lrw += count as i64 * lrw_diff;
        *liw += count as i64 * liw_diff;
    }
    SUNTRUE
}

/*---------------------------------------------------------------
  arkAllocVectors:

  This routine allocates the ARKODE vectors ewt, yn, tempv* and
  ftemp. If any of these vectors already exist, they are left
  alone. Otherwise, it will allocate each vector by cloning the
  input vector. This routine also updates the optional outputs
  lrw and liw, which are (respectively) the lengths of the real
  and integer work spaces.

  If all memory allocations are successful, arkAllocVectors
  returns SUNTRUE, otherwise it returns SUNFALSE.
  ---------------------------------------------------------------*/
pub fn arkAllocVectors(ark_mem: &ARKodeMem, tmpl: &N_Vector) -> sunbooleantype {
    /* Allocate ewt if needed */
    let mut v = ark_mem.borrow_mut().ewt.take();
    let ok = arkAllocVec(ark_mem, tmpl, &mut v);
    ark_mem.borrow_mut().ewt = v;
    if !ok {
        return SUNFALSE;
    }

    /* Set rwt to point at ewt */
    if ark_mem.borrow().rwt_is_ewt {
        let ewt = ark_mem.borrow().ewt.clone();
        ark_mem.borrow_mut().rwt = ewt;
    }

    /* Allocate yn if needed */
    let mut v = ark_mem.borrow_mut().yn.take();
    let ok = arkAllocVec(ark_mem, tmpl, &mut v);
    ark_mem.borrow_mut().yn = v;
    if !ok {
        return SUNFALSE;
    }

    /* Allocate tempv1 if needed */
    let mut v = ark_mem.borrow_mut().tempv1.take();
    let ok = arkAllocVec(ark_mem, tmpl, &mut v);
    ark_mem.borrow_mut().tempv1 = v;
    if !ok {
        return SUNFALSE;
    }

    /* Allocate tempv2 if needed */
    let mut v = ark_mem.borrow_mut().tempv2.take();
    let ok = arkAllocVec(ark_mem, tmpl, &mut v);
    ark_mem.borrow_mut().tempv2 = v;
    if !ok {
        return SUNFALSE;
    }

    /* Allocate tempv3 if needed */
    let mut v = ark_mem.borrow_mut().tempv3.take();
    let ok = arkAllocVec(ark_mem, tmpl, &mut v);
    ark_mem.borrow_mut().tempv3 = v;
    if !ok {
        return SUNFALSE;
    }

    /* Allocate tempv4 if needed */
    let mut v = ark_mem.borrow_mut().tempv4.take();
    let ok = arkAllocVec(ark_mem, tmpl, &mut v);
    ark_mem.borrow_mut().tempv4 = v;
    if !ok {
        return SUNFALSE;
    }

    SUNTRUE
}

/*---------------------------------------------------------------
  arkResizeVectors:

  This routine resizes all ARKODE vectors if they exist,
  otherwise they are left alone. If a resize function is provided
  it is called to resize the vectors otherwise the vector is
  freed and a new vector is created by cloning in input vector.
  This routine also updates the optional outputs lrw and liw,
  which are (respectively) the lengths of the real and integer
  work spaces.

  If all memory allocations are successful, arkResizeVectors
  returns SUNTRUE, otherwise it returns SUNFALSE.
  ---------------------------------------------------------------*/
pub fn arkResizeVectors(
    ark_mem: &ARKodeMem,
    resize: Option<ARKVecResizeFn>,
    resize_data: &mut Option<Box<dyn std::any::Any>>,
    lrw_diff: sunindextype,
    liw_diff: sunindextype,
    tmpl: &N_Vector,
) -> sunbooleantype {
    /* Vabstol */
    let mut v = ark_mem.borrow_mut().Vabstol.take();
    let ok = arkResizeVec(ark_mem, resize, resize_data, lrw_diff, liw_diff, tmpl, &mut v);
    ark_mem.borrow_mut().Vabstol = v;
    if !ok {
        return SUNFALSE;
    }

    /* VRabstol */
    let mut v = ark_mem.borrow_mut().VRabstol.take();
    let ok = arkResizeVec(ark_mem, resize, resize_data, lrw_diff, liw_diff, tmpl, &mut v);
    ark_mem.borrow_mut().VRabstol = v;
    if !ok {
        return SUNFALSE;
    }

    /* ewt */
    let mut v = ark_mem.borrow_mut().ewt.take();
    let ok = arkResizeVec(ark_mem, resize, resize_data, lrw_diff, liw_diff, tmpl, &mut v);
    ark_mem.borrow_mut().ewt = v;
    if !ok {
        return SUNFALSE;
    }

    /* rwt  */
    if ark_mem.borrow().rwt_is_ewt {
        /* update pointer to ewt */
        let ewt = ark_mem.borrow().ewt.clone();
        ark_mem.borrow_mut().rwt = ewt;
    } else {
        /* resize if distinct from ewt */
        let mut v = ark_mem.borrow_mut().rwt.take();
        let ok = arkResizeVec(ark_mem, resize, resize_data, lrw_diff, liw_diff, tmpl, &mut v);
        ark_mem.borrow_mut().rwt = v;
        if !ok {
            return SUNFALSE;
        }
    }

    /* yn */
    let mut v = ark_mem.borrow_mut().yn.take();
    let ok = arkResizeVec(ark_mem, resize, resize_data, lrw_diff, liw_diff, tmpl, &mut v);
    ark_mem.borrow_mut().yn = v;
    if !ok {
        return SUNFALSE;
    }

    /* fn */
    let mut v = ark_mem.borrow_mut().fn_.take();
    let ok = arkResizeVec(ark_mem, resize, resize_data, lrw_diff, liw_diff, tmpl, &mut v);
    ark_mem.borrow_mut().fn_ = v;
    if !ok {
        return SUNFALSE;
    }

    /* tempv* */
    let mut v = ark_mem.borrow_mut().tempv1.take();
    let ok = arkResizeVec(ark_mem, resize, resize_data, lrw_diff, liw_diff, tmpl, &mut v);
    ark_mem.borrow_mut().tempv1 = v;
    if !ok {
        return SUNFALSE;
    }

    let mut v = ark_mem.borrow_mut().tempv2.take();
    let ok = arkResizeVec(ark_mem, resize, resize_data, lrw_diff, liw_diff, tmpl, &mut v);
    ark_mem.borrow_mut().tempv2 = v;
    if !ok {
        return SUNFALSE;
    }

    let mut v = ark_mem.borrow_mut().tempv3.take();
    let ok = arkResizeVec(ark_mem, resize, resize_data, lrw_diff, liw_diff, tmpl, &mut v);
    ark_mem.borrow_mut().tempv3 = v;
    if !ok {
        return SUNFALSE;
    }

    let mut v = ark_mem.borrow_mut().tempv4.take();
    let ok = arkResizeVec(ark_mem, resize, resize_data, lrw_diff, liw_diff, tmpl, &mut v);
    ark_mem.borrow_mut().tempv4 = v;
    if !ok {
        return SUNFALSE;
    }

    let mut v = ark_mem.borrow_mut().tempv5.take();
    let ok = arkResizeVec(ark_mem, resize, resize_data, lrw_diff, liw_diff, tmpl, &mut v);
    ark_mem.borrow_mut().tempv5 = v;
    if !ok {
        return SUNFALSE;
    }

    SUNTRUE
}

/*---------------------------------------------------------------
  arkFreeVectors

  This routine frees the ARKODE vectors allocated in both
  arkAllocVectors and arkAllocRKVectors.

  PORT NOTE: exactly as in C, `rwt` is NOT cleared when it aliases
  `ewt` (C leaves a dangling alias; the port leaves the extra handle
  clone, which keeps the buffer alive until `rwt` is overwritten -- not
  observable, and `lrw`/`liw` accounting is unchanged).
  ---------------------------------------------------------------*/
pub fn arkFreeVectors(ark_mem: &ARKodeMem) {
    let mut v = ark_mem.borrow_mut().ewt.take();
    arkFreeVec(ark_mem, &mut v);
    ark_mem.borrow_mut().ewt = v;

    if !ark_mem.borrow().rwt_is_ewt {
        let mut v = ark_mem.borrow_mut().rwt.take();
        arkFreeVec(ark_mem, &mut v);
        ark_mem.borrow_mut().rwt = v;
    }

    let mut v = ark_mem.borrow_mut().tempv1.take();
    arkFreeVec(ark_mem, &mut v);
    ark_mem.borrow_mut().tempv1 = v;

    let mut v = ark_mem.borrow_mut().tempv2.take();
    arkFreeVec(ark_mem, &mut v);
    ark_mem.borrow_mut().tempv2 = v;

    let mut v = ark_mem.borrow_mut().tempv3.take();
    arkFreeVec(ark_mem, &mut v);
    ark_mem.borrow_mut().tempv3 = v;

    let mut v = ark_mem.borrow_mut().tempv4.take();
    arkFreeVec(ark_mem, &mut v);
    ark_mem.borrow_mut().tempv4 = v;

    let mut v = ark_mem.borrow_mut().tempv5.take();
    arkFreeVec(ark_mem, &mut v);
    ark_mem.borrow_mut().tempv5 = v;

    let mut v = ark_mem.borrow_mut().yn.take();
    arkFreeVec(ark_mem, &mut v);
    ark_mem.borrow_mut().yn = v;

    let mut v = ark_mem.borrow_mut().fn_.take();
    arkFreeVec(ark_mem, &mut v);
    ark_mem.borrow_mut().fn_ = v;

    let mut v = ark_mem.borrow_mut().Vabstol.take();
    arkFreeVec(ark_mem, &mut v);
    ark_mem.borrow_mut().Vabstol = v;

    let mut v = ark_mem.borrow_mut().constraints.take();
    arkFreeVec(ark_mem, &mut v);
    ark_mem.borrow_mut().constraints = v;
}

/*---------------------------------------------------------------
  arkAccessHAdaptMem:

  Shortcut routine to unpack ark_mem and hadapt_mem structures from
  void* pointer.  If either is missing it returns ARK_MEM_NULL.

  PORT NOTE: `ARKodeHAdaptMem` is a `Box` owned by `ark_mem` and cannot
  be handed out, so -- exactly like `step_getlinmem` in the frozen
  contract -- this becomes a PRESENCE CHECK.  On `ARK_SUCCESS` the
  caller reaches the record through
  `ark_mem.borrow[_mut]().hadapt_mem.as_[mut_]ref().expect(...)`.
  The C `arkode_mem == NULL` branch (the only user of `fname`) cannot
  fire because the Rust handle is a `&ARKodeMem`.
  ---------------------------------------------------------------*/
pub fn arkAccessHAdaptMem(ark_mem: &ARKodeMem, fname: &str) -> i32 {
    let _ = fname; /* used only by C's unreachable NULL-handle branch */
    if ark_mem.borrow().hadapt_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "arkAccessHAdaptMem",
            file!(),
            MSG_ARKADAPT_NO_MEM,
        );
        return ARK_MEM_NULL;
    }
    ARK_SUCCESS
}

/*---------------------------------------------------------------
  Utility routines for ARKODE to serve as an MRIStepInnerStepper
  ---------------------------------------------------------------*/

/*------------------------------------------------------------------------------
  ark_MRIStepInnerEvolve

  Implementation of MRIStepInnerStepperEvolveFn to advance the inner (fast)
  ODE IVP.  Since the raw return value from an MRIStepInnerStepper is
  meaningless, aside from whether it is 0 (success), >0 (recoverable failure),
  and <0 (unrecoverable failure), we map various ARKODE return values
  accordingly.
  ----------------------------------------------------------------------------*/
pub fn ark_MRIStepInnerEvolve(
    stepper: &crate::arkode_mristep::MRIStepInnerStepper,
    _t0: sunrealtype, /* SUNDIALS_MAYBE_UNUSED in C */
    tout: sunrealtype,
    y: &N_Vector,
) -> i32 {
    /* extract the ARKODE memory struct */
    let mut arkode_mem: Option<ARKodeMem> = None;
    let retval =
        crate::arkode_mristep::MRIStepInnerStepper_GetContentAs::<ARKodeMem>(
            stepper,
            &mut arkode_mem,
        );
    if retval != ARK_SUCCESS {
        return -1;
    }
    let ark_mem = match arkode_mem {
        None => {
            arkProcessError(
                None,
                ARK_MEM_NULL,
                line!() as i32,
                "ark_MRIStepInnerEvolve",
                file!(),
                MSG_ARK_NO_MEM,
            );
            return -1;
        }
        Some(m) => m,
    };

    /* get the forcing data */
    let mut tshift: sunrealtype = ZERO;
    let mut tscale: sunrealtype = ZERO;
    let mut forcing: Vec<N_Vector> = Vec::new();
    let mut nforcing: i32 = 0;
    let retval = MRIStepInnerStepper_GetForcingData(
        stepper,
        &mut tshift,
        &mut tscale,
        &mut forcing,
        &mut nforcing,
    );
    if retval != ARK_SUCCESS {
        return -1;
    }

    /* set the inner forcing data */
    let step_setforcing = ark_mem.borrow().step_setforcing.expect("step_setforcing set");
    let retval = step_setforcing(&ark_mem, tshift, tscale, &forcing, nforcing);
    if retval != ARK_SUCCESS {
        return -1;
    }

    /* set the stop time */
    let retval = ARKodeSetStopTime(&ark_mem, tout);
    if retval != ARK_SUCCESS {
        return -1;
    }

    /* evolve inner ODE, consider all positive return values as 'success' */
    let mut tret: sunrealtype = ZERO;
    let mut retval = ARKodeEvolve(&ark_mem, tout, y, &mut tret, ARK_NORMAL);
    if retval > 0 {
        retval = 0;
    }

    /* set a recoverable failure for a few ARKODE failure modes;
    on other ARKODE errors return with an unrecoverable failure */
    if retval < 0 {
        if (retval == ARK_TOO_MUCH_WORK)
            || (retval == ARK_CONV_FAILURE)
            || (retval == ARK_ERR_FAILURE)
        {
            retval = 1;
        } else {
            return -1;
        }
    }

    /* disable inner forcing */
    let step_setforcing = ark_mem.borrow().step_setforcing.expect("step_setforcing set");
    if step_setforcing(&ark_mem, ZERO, ONE, &[], 0) != ARK_SUCCESS {
        return -1;
    }

    retval
}

/*------------------------------------------------------------------------------
  ark_MRIStepInnerFullRhs

  Implementation of MRIStepInnerStepperFullRhsFn to compute the full inner
  (fast) ODE IVP RHS.
  ----------------------------------------------------------------------------*/
pub fn ark_MRIStepInnerFullRhs(
    stepper: &crate::arkode_mristep::MRIStepInnerStepper,
    t: sunrealtype,
    y: &N_Vector,
    f: &N_Vector,
    mode: i32,
) -> i32 {
    let mut arkode_mem: Option<ARKodeMem> = None;
    let retval =
        crate::arkode_mristep::MRIStepInnerStepper_GetContentAs::<ARKodeMem>(
            stepper,
            &mut arkode_mem,
        );
    if retval != ARK_SUCCESS {
        return -1;
    }
    let ark_mem = match arkode_mem {
        None => {
            arkProcessError(
                None,
                ARK_MEM_NULL,
                line!() as i32,
                "ark_MRIStepInnerFullRhs",
                file!(),
                MSG_ARK_NO_MEM,
            );
            return -1;
        }
        Some(m) => m,
    };
    let step_fullrhs = ark_mem.borrow().step_fullrhs.expect("step_fullrhs set");
    let retval = step_fullrhs(&ark_mem, t, y, f, mode);
    if retval == ARK_SUCCESS {
        return 0;
    }
    -1
}

/*------------------------------------------------------------------------------
  ark_MRIStepInnerReset

  Implementation of MRIStepInnerStepperResetFn to reset the inner (fast) stepper
  state.
  ----------------------------------------------------------------------------*/
pub fn ark_MRIStepInnerReset(
    stepper: &crate::arkode_mristep::MRIStepInnerStepper,
    tR: sunrealtype,
    yR: &N_Vector,
) -> i32 {
    let mut arkode_mem: Option<ARKodeMem> = None;
    let retval =
        crate::arkode_mristep::MRIStepInnerStepper_GetContentAs::<ARKodeMem>(
            stepper,
            &mut arkode_mem,
        );
    if retval != ARK_SUCCESS {
        return -1;
    }
    /* C hands the (possibly NULL) void* straight to ARKodeReset, which then
    returns ARK_MEM_NULL -> -1; the Rust handle model reaches the same
    result without the intermediate call. */
    let ark_mem = match arkode_mem {
        None => return -1,
        Some(m) => m,
    };
    let retval = ARKodeReset(&ark_mem, tR, yR);
    if retval == ARK_SUCCESS {
        return 0;
    }
    -1
}

/*------------------------------------------------------------------------------
  ark_MRIStepInnerGetAccumulatedError

  Implementation of MRIStepInnerGetAccumulatedError to retrieve the accumulated
  temporal error estimate from the inner (fast) stepper.
  ----------------------------------------------------------------------------*/
pub fn ark_MRIStepInnerGetAccumulatedError(
    stepper: &crate::arkode_mristep::MRIStepInnerStepper,
    accum_error: &mut sunrealtype,
) -> i32 {
    let mut arkode_mem: Option<ARKodeMem> = None;
    let retval =
        crate::arkode_mristep::MRIStepInnerStepper_GetContentAs::<ARKodeMem>(
            stepper,
            &mut arkode_mem,
        );
    if retval != ARK_SUCCESS {
        return -1;
    }
    let ark_mem = match arkode_mem {
        None => return -1,
        Some(m) => m,
    };
    let retval = ARKodeGetAccumulatedError(&ark_mem, accum_error);
    if retval == ARK_SUCCESS {
        return 0;
    }
    if retval > 0 {
        return 1;
    }
    -1
}

/*------------------------------------------------------------------------------
  ark_MRIStepInnerResetAccumulatedError

  Implementation of MRIStepInnerResetAccumulatedError to reset the accumulated
  temporal error estimator in the inner (fast) stepper.
  ----------------------------------------------------------------------------*/
pub fn ark_MRIStepInnerResetAccumulatedError(
    stepper: &crate::arkode_mristep::MRIStepInnerStepper,
) -> i32 {
    let mut arkode_mem: Option<ARKodeMem> = None;
    let retval =
        crate::arkode_mristep::MRIStepInnerStepper_GetContentAs::<ARKodeMem>(
            stepper,
            &mut arkode_mem,
        );
    if retval != ARK_SUCCESS {
        return -1;
    }
    let ark_mem = match arkode_mem {
        None => return -1,
        Some(m) => m,
    };
    let retval = ARKodeResetAccumulatedError(&ark_mem);
    if retval == ARK_SUCCESS {
        return 0;
    }
    -1
}

/*------------------------------------------------------------------------------
  ark_MRIStepInnerSetRTol

  Implementation of MRIStepInnerSetRTol to set a relative tolerance for the
  upcoming evolution using the inner (fast) stepper.
  ----------------------------------------------------------------------------*/
pub fn ark_MRIStepInnerSetRTol(
    stepper: &crate::arkode_mristep::MRIStepInnerStepper,
    rtol: sunrealtype,
) -> i32 {
    let mut arkode_mem: Option<ARKodeMem> = None;
    let retval =
        crate::arkode_mristep::MRIStepInnerStepper_GetContentAs::<ARKodeMem>(
            stepper,
            &mut arkode_mem,
        );
    if retval != ARK_SUCCESS {
        return -1;
    }
    let ark_mem = match arkode_mem {
        None => {
            arkProcessError(
                None,
                ARK_MEM_NULL,
                line!() as i32,
                "ark_MRIStepInnerSetRTol",
                file!(),
                MSG_ARK_NO_MEM,
            );
            return -1;
        }
        Some(m) => m,
    };
    if rtol > ZERO {
        ark_mem.borrow_mut().reltol = rtol;
        0
    } else {
        -1
    }
}
