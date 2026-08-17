//! Port of `src/sundials/sundials_stepper.c` +
//! `src/sundials/sundials_stepper_impl.h` +
//! `include/sundials/sundials_stepper.h` (the generic SUNStepper base class).
//!
//! Handle model (ARCHITECTURE.md): `SUNStepper = Rc<SUNStepper_>`; `content`
//! is the C `void*` (`RefCell<Box<dyn Any>>`), `ops` is the C ops table
//! (`RefCell` because `SUNStepper_Set*Fn` overwrite op slots in place through
//! a shared handle), and `last_flag`/`python` are the remaining C fields.
//!
//! Reference-build configuration: `SUNDIALS_ENABLE_ERROR_CHECKS` off, so the
//! `SUNFunctionBegin` / `SUNCheck` / `SUNAssert` lines compile away; the
//! `SUNDIALS_ENABLE_PYTHON` branch in `SUNStepper_Destroy`
//! (`SUNStepperFunctionTable_Destroy`) is not built and is omitted, but the
//! unconditional `python` field itself is kept (always `None` here).
//!
//! Deviations from the C, all documented at their sites:
//! * `SUNStepper_Create` leaves `ops->reinit`, `ops->resetcheckpointindex`
//!   and `ops->getnumsteps` in freshly `malloc`ed (indeterminate) memory in
//!   C — reading them is UB. The port zero-initialises the whole ops table
//!   (`Default`), i.e. every unset op is `None` (deviation class 5).
//! * `SUNStepper_GetContent` swaps rather than aliases the content box
//!   (deviation class 6); see its doc comment and
//!   [`SUNStepper_GetContentAs`].
//! * `SUNStepper_Destroy` drops the `Rc`; storage is released only when the
//!   last clone goes away, where C `free`s immediately and leaves every other
//!   copy of the pointer dangling.

use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

use crate::sundials_context::SUNContext;
use crate::sundials_errors::{
    SUN_ERR_ARG_CORRUPT, SUN_ERR_MALLOC_FAIL, SUN_ERR_NOT_IMPLEMENTED, SUN_SUCCESS,
};
use crate::sundials_matrix::SUNMatrix;
use crate::sundials_nvector::N_Vector;
use crate::sundials_types::*;

/* -----------------------------------------------------------------
 * Types from include/sundials/sundials_stepper.h
 * ----------------------------------------------------------------- */

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SUNFullRhsMode {
    SUN_FULLRHS_START,
    SUN_FULLRHS_END,
    SUN_FULLRHS_OTHER,
}
pub use SUNFullRhsMode::*;

/// C `SUNRhsJacFn` — generic right-hand-side Jacobian callback.
pub type SUNRhsJacFn = fn(
    t: sunrealtype,
    y: &N_Vector,
    fy: &N_Vector,
    Jac: &SUNMatrix,
    user_data: &mut Option<Box<dyn Any>>,
    tmp1: &N_Vector,
    tmp2: &N_Vector,
    tmp3: &N_Vector,
) -> i32;

/// C `SUNRhsJacTimesFn` — generic Jacobian-times-vector callback.
pub type SUNRhsJacTimesFn = fn(
    v: &N_Vector,
    Jv: &N_Vector,
    t: sunrealtype,
    y: &N_Vector,
    fy: &N_Vector,
    user_data: &mut Option<Box<dyn Any>>,
    tmp: &N_Vector,
) -> i32;

pub type SUNStepperEvolveFn = fn(
    stepper: &SUNStepper,
    tout: sunrealtype,
    vret: &N_Vector,
    tret: &mut sunrealtype,
) -> SUNErrCode;

pub type SUNStepperOneStepFn = fn(
    stepper: &SUNStepper,
    tout: sunrealtype,
    vret: &N_Vector,
    tret: &mut sunrealtype,
) -> SUNErrCode;

pub type SUNStepperFullRhsFn = fn(
    stepper: &SUNStepper,
    t: sunrealtype,
    v: &N_Vector,
    f: &N_Vector,
    mode: SUNFullRhsMode,
) -> SUNErrCode;

pub type SUNStepperReInitFn =
    fn(stepper: &SUNStepper, t0: sunrealtype, v0: &N_Vector) -> SUNErrCode;

pub type SUNStepperResetFn = fn(stepper: &SUNStepper, tR: sunrealtype, vR: &N_Vector) -> SUNErrCode;

pub type SUNStepperResetCheckpointIndexFn =
    fn(stepper: &SUNStepper, ckptIdxR: suncountertype) -> SUNErrCode;

pub type SUNStepperSetStopTimeFn = fn(stepper: &SUNStepper, tstop: sunrealtype) -> SUNErrCode;

pub type SUNStepperSetStepDirectionFn =
    fn(stepper: &SUNStepper, stepdir: sunrealtype) -> SUNErrCode;

/// C `SUNStepperSetForcingFn`; the C pair `(N_Vector* forcing_1d, int
/// nforcing)` keeps both arguments (ARCHITECTURE: `N_Vector*` → `&[N_Vector]`).
pub type SUNStepperSetForcingFn = fn(
    stepper: &SUNStepper,
    tshift: sunrealtype,
    tscale: sunrealtype,
    forcing_1d: &[N_Vector],
    nforcing: i32,
) -> SUNErrCode;

pub type SUNStepperGetNumStepsFn = fn(stepper: &SUNStepper, nst: &mut suncountertype) -> SUNErrCode;

pub type SUNStepperDestroyFn = fn(stepper: &SUNStepper) -> SUNErrCode;

/* -----------------------------------------------------------------
 * Types from src/sundials/sundials_stepper_impl.h
 * ----------------------------------------------------------------- */

/// C `struct SUNStepper_Ops_`.
#[derive(Default, Clone)]
pub struct SUNStepper_Ops_ {
    pub evolve: Option<SUNStepperEvolveFn>,
    pub onestep: Option<SUNStepperOneStepFn>,
    pub fullrhs: Option<SUNStepperFullRhsFn>,
    pub reinit: Option<SUNStepperReInitFn>,
    pub reset: Option<SUNStepperResetFn>,
    pub resetcheckpointindex: Option<SUNStepperResetCheckpointIndexFn>,
    pub setstoptime: Option<SUNStepperSetStopTimeFn>,
    pub setstepdirection: Option<SUNStepperSetStepDirectionFn>,
    pub setforcing: Option<SUNStepperSetForcingFn>,
    pub getnumsteps: Option<SUNStepperGetNumStepsFn>,
    pub destroy: Option<SUNStepperDestroyFn>,
}

pub type SUNStepper_Ops = SUNStepper_Ops_;

/// C `struct SUNStepper_`.
pub struct SUNStepper_ {
    /// C `void* python` — python interface specific content. The Python
    /// interface is not part of this port, so this stays `None`; the field is
    /// kept because C declares and clears it unconditionally.
    pub python: RefCell<Option<Box<dyn Any>>>,

    /* stepper specific content and operations */
    /// C `void* content`; the empty box `Box::new(())` is C's `NULL`.
    pub content: RefCell<Box<dyn Any>>,
    pub ops: RefCell<SUNStepper_Ops_>,

    /* stepper context */
    pub sunctx: RefCell<SUNContext>,

    /* last stepper return flag */
    pub last_flag: RefCell<i32>,
}

pub type SUNStepper = Rc<SUNStepper_>;

/* -----------------------------------------------------------------
 * Constructor / destructor
 * ----------------------------------------------------------------- */

/// C `SUNStepper_Create`.
///
/// C leaves `ops->reinit`, `ops->resetcheckpointindex` and `ops->getnumsteps`
/// uninitialised (only eight of the eleven slots are NULLed after the
/// `malloc`); the port initialises the whole table to `None`.
pub fn SUNStepper_Create(sunctx: &SUNContext, stepper_ptr: &mut Option<SUNStepper>) -> SUNErrCode {
    /* SUNCheck(stepper_ptr, SUN_ERR_ARG_CORRUPT): `&mut` is never NULL.
    SUNAssert(stepper, SUN_ERR_MALLOC_FAIL): allocation cannot fail here. */
    let stepper = Rc::new(SUNStepper_ {
        python: RefCell::new(None),
        content: RefCell::new(Box::new(())),
        ops: RefCell::new(SUNStepper_Ops_::default()),
        sunctx: RefCell::new(sunctx.clone()),
        last_flag: RefCell::new(SUN_SUCCESS),
    });

    *stepper_ptr = Some(stepper);

    SUN_SUCCESS
}

/// C `SUNStepper_Destroy`.
///
/// C tests `stepper_ptr != NULL` but then dereferences `*stepper_ptr`
/// unconditionally; a NULL handle is UB there and a deterministic panic here
/// (deviation class 5). Dropping the `Rc` replaces C's `free(ops)` /
/// `free(*stepper_ptr)`, so the storage survives while other clones of the
/// handle do (C would leave those dangling).
pub fn SUNStepper_Destroy(stepper_ptr: &mut Option<SUNStepper>) -> SUNErrCode {
    {
        let stepper = stepper_ptr
            .as_ref()
            .expect("SUNStepper_Destroy: NULL SUNStepper");

        let destroy = stepper.ops.borrow().destroy;
        if let Some(f) = destroy {
            /* C discards the return value */
            let _ = f(stepper);
        }

        /* C: free(ops); [python table destroy]; (*stepper_ptr)->python = NULL */
        *stepper.python.borrow_mut() = None;
    }

    *stepper_ptr = None;

    SUN_SUCCESS
}

/* -----------------------------------------------------------------
 * Operations
 * ----------------------------------------------------------------- */

pub fn SUNStepper_Evolve(
    stepper: &SUNStepper,
    tout: sunrealtype,
    y: &N_Vector,
    tret: &mut sunrealtype,
) -> SUNErrCode {
    let f = stepper.ops.borrow().evolve;
    if let Some(f) = f {
        return f(stepper, tout, y, tret);
    }
    SUN_ERR_NOT_IMPLEMENTED
}

pub fn SUNStepper_OneStep(
    stepper: &SUNStepper,
    tout: sunrealtype,
    y: &N_Vector,
    tret: &mut sunrealtype,
) -> SUNErrCode {
    let f = stepper.ops.borrow().onestep;
    if let Some(f) = f {
        return f(stepper, tout, y, tret);
    }
    SUN_ERR_NOT_IMPLEMENTED
}

pub fn SUNStepper_FullRhs(
    stepper: &SUNStepper,
    t: sunrealtype,
    v: &N_Vector,
    f: &N_Vector,
    mode: SUNFullRhsMode,
) -> SUNErrCode {
    let op = stepper.ops.borrow().fullrhs;
    if let Some(op) = op {
        return op(stepper, t, v, f, mode);
    }
    SUN_ERR_NOT_IMPLEMENTED
}

pub fn SUNStepper_ReInit(stepper: &SUNStepper, t0: sunrealtype, y0: &N_Vector) -> SUNErrCode {
    let f = stepper.ops.borrow().reinit;
    if let Some(f) = f {
        return f(stepper, t0, y0);
    }
    SUN_ERR_NOT_IMPLEMENTED
}

pub fn SUNStepper_Reset(stepper: &SUNStepper, tR: sunrealtype, yR: &N_Vector) -> SUNErrCode {
    let f = stepper.ops.borrow().reset;
    if let Some(f) = f {
        return f(stepper, tR, yR);
    }
    SUN_ERR_NOT_IMPLEMENTED
}

pub fn SUNStepper_ResetCheckpointIndex(
    stepper: &SUNStepper,
    ckptIdxR: suncountertype,
) -> SUNErrCode {
    let f = stepper.ops.borrow().resetcheckpointindex;
    if let Some(f) = f {
        return f(stepper, ckptIdxR);
    }
    SUN_ERR_NOT_IMPLEMENTED
}

pub fn SUNStepper_SetStopTime(stepper: &SUNStepper, tstop: sunrealtype) -> SUNErrCode {
    let f = stepper.ops.borrow().setstoptime;
    if let Some(f) = f {
        return f(stepper, tstop);
    }
    SUN_ERR_NOT_IMPLEMENTED
}

pub fn SUNStepper_SetStepDirection(stepper: &SUNStepper, stepdir: sunrealtype) -> SUNErrCode {
    let f = stepper.ops.borrow().setstepdirection;
    if let Some(f) = f {
        return f(stepper, stepdir);
    }
    SUN_ERR_NOT_IMPLEMENTED
}

pub fn SUNStepper_SetForcing(
    stepper: &SUNStepper,
    tshift: sunrealtype,
    tscale: sunrealtype,
    forcing: &[N_Vector],
    nforcing: i32,
) -> SUNErrCode {
    let f = stepper.ops.borrow().setforcing;
    if let Some(f) = f {
        return f(stepper, tshift, tscale, forcing, nforcing);
    }
    SUN_ERR_NOT_IMPLEMENTED
}

/* -----------------------------------------------------------------
 * Content, counters, and last flag
 * ----------------------------------------------------------------- */

/// C `SUNStepper_SetContent`.
///
/// C stores a non-owning `void*`; the port takes ownership of the box. Use
/// `Box::new(())` for C's `NULL`.
pub fn SUNStepper_SetContent(stepper: &SUNStepper, content: Box<dyn Any>) -> SUNErrCode {
    *stepper.content.borrow_mut() = content;
    SUN_SUCCESS
}

/// C `SUNStepper_GetContent` hands back the raw `void* content` without
/// transferring ownership. A safe-Rust `Box<dyn Any>` token cannot be
/// aliased, so the stored box is SWAPPED with `content` (deviation class 6,
/// as `CVodeGetUserData`): the caller MUST hand it back — via
/// `SUNStepper_SetContent` or a second swap — on every return path, before
/// any other code (including `SUNStepper_Destroy` and any op) touches the
/// stepper's content.
///
/// Implementation modules should NOT use this. When the C content pointer is
/// a SUNDIALS handle (the ARKODE case: `ARKodeMem`), use
/// [`SUNStepper_GetContentAs`], which clones the `Rc` — exactly C's pointer
/// copy — and leaves the stepper's content in place. Otherwise follow the
/// locked `content_mut` pattern and borrow `stepper.content` directly.
pub fn SUNStepper_GetContent(stepper: &SUNStepper, content: &mut Box<dyn Any>) -> SUNErrCode {
    std::mem::swap(&mut *stepper.content.borrow_mut(), content);
    SUN_SUCCESS
}

/// Port-only, borrow-safe companion to [`SUNStepper_GetContent`] for the
/// common case where the C `void* content` is a SUNDIALS handle (any `Clone`
/// type — `Rc<…>` clones are C pointer copies). The stepper keeps its
/// content; nothing has to be handed back.
///
/// A content type mismatch is C UB (a bad cast) and panics here
/// (deviation class 5), matching the locked `content_mut` helpers.
pub fn SUNStepper_GetContentAs<T: Any + Clone>(
    stepper: &SUNStepper,
    content: &mut Option<T>,
) -> SUNErrCode {
    *content = Some(
        stepper
            .content
            .borrow()
            .downcast_ref::<T>()
            .expect("SUNStepper content")
            .clone(),
    );
    SUN_SUCCESS
}

pub fn SUNStepper_GetNumSteps(stepper: &SUNStepper, nst: &mut suncountertype) -> SUNErrCode {
    let f = stepper.ops.borrow().getnumsteps;
    if let Some(f) = f {
        return f(stepper, nst);
    }
    SUN_ERR_NOT_IMPLEMENTED
}

pub fn SUNStepper_SetLastFlag(stepper: &SUNStepper, last_flag: i32) -> SUNErrCode {
    *stepper.last_flag.borrow_mut() = last_flag;
    SUN_SUCCESS
}

pub fn SUNStepper_GetLastFlag(stepper: &SUNStepper, last_flag: &mut i32) -> SUNErrCode {
    *last_flag = *stepper.last_flag.borrow();
    SUN_SUCCESS
}

/* -----------------------------------------------------------------
 * Attach operations
 * ----------------------------------------------------------------- */

pub fn SUNStepper_SetEvolveFn(stepper: &SUNStepper, fn_: Option<SUNStepperEvolveFn>) -> SUNErrCode {
    stepper.ops.borrow_mut().evolve = fn_;
    SUN_SUCCESS
}

pub fn SUNStepper_SetOneStepFn(
    stepper: &SUNStepper,
    fn_: Option<SUNStepperOneStepFn>,
) -> SUNErrCode {
    stepper.ops.borrow_mut().onestep = fn_;
    SUN_SUCCESS
}

pub fn SUNStepper_SetFullRhsFn(
    stepper: &SUNStepper,
    fn_: Option<SUNStepperFullRhsFn>,
) -> SUNErrCode {
    stepper.ops.borrow_mut().fullrhs = fn_;
    SUN_SUCCESS
}

pub fn SUNStepper_SetReInitFn(stepper: &SUNStepper, fn_: Option<SUNStepperReInitFn>) -> SUNErrCode {
    stepper.ops.borrow_mut().reinit = fn_;
    SUN_SUCCESS
}

pub fn SUNStepper_SetResetFn(stepper: &SUNStepper, fn_: Option<SUNStepperResetFn>) -> SUNErrCode {
    stepper.ops.borrow_mut().reset = fn_;
    SUN_SUCCESS
}

pub fn SUNStepper_SetResetCheckpointIndexFn(
    stepper: &SUNStepper,
    fn_: Option<SUNStepperResetCheckpointIndexFn>,
) -> SUNErrCode {
    stepper.ops.borrow_mut().resetcheckpointindex = fn_;
    SUN_SUCCESS
}

pub fn SUNStepper_SetStopTimeFn(
    stepper: &SUNStepper,
    fn_: Option<SUNStepperSetStopTimeFn>,
) -> SUNErrCode {
    stepper.ops.borrow_mut().setstoptime = fn_;
    SUN_SUCCESS
}

pub fn SUNStepper_SetStepDirectionFn(
    stepper: &SUNStepper,
    fn_: Option<SUNStepperSetStepDirectionFn>,
) -> SUNErrCode {
    stepper.ops.borrow_mut().setstepdirection = fn_;
    SUN_SUCCESS
}

pub fn SUNStepper_SetForcingFn(
    stepper: &SUNStepper,
    fn_: Option<SUNStepperSetForcingFn>,
) -> SUNErrCode {
    stepper.ops.borrow_mut().setforcing = fn_;
    SUN_SUCCESS
}

pub fn SUNStepper_SetGetNumStepsFn(
    stepper: &SUNStepper,
    fn_: Option<SUNStepperGetNumStepsFn>,
) -> SUNErrCode {
    stepper.ops.borrow_mut().getnumsteps = fn_;
    SUN_SUCCESS
}

pub fn SUNStepper_SetDestroyFn(
    stepper: &SUNStepper,
    fn_: Option<SUNStepperDestroyFn>,
) -> SUNErrCode {
    stepper.ops.borrow_mut().destroy = fn_;
    SUN_SUCCESS
}

/* C `SUNCheck`/`SUNAssert` codes referenced by the compiled-out checks in
`SUNStepper_Create` (see there). */
const _: SUNErrCode = SUN_ERR_ARG_CORRUPT;
const _: SUNErrCode = SUN_ERR_MALLOC_FAIL;
