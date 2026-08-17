//! Port of `src/sundials/sundials_adjointstepper.c` +
//! `src/sundials/sundials_adjointstepper_impl.h` +
//! `include/sundials/sundials_adjointstepper.h` (SUNAdjointStepper class).
//!
//! Handle model (ARCHITECTURE.md): `SUNAdjointStepper = Rc<SUNAdjointStepper_>`.
//! Unlike the other SUNDIALS base classes this one has **no ops table** in C
//! (no virtual methods) — `struct SUNAdjointStepper_` is plain data that the
//! ARKODE adjoint interfaces read directly (`adj_stepper->final_step_idx`,
//! `->checkpoint_scheme`, `->adj_sunstepper`, `->user_data`). The fields
//! therefore carry their own interior mutability: `Cell` for the C scalar
//! fields (`suncountertype` / `sunrealtype` / `sunbooleantype`) and `RefCell`
//! for the handle and `void*` fields. Sub-handles are cloned out of their
//! `RefCell` (a C pointer copy) *before* each `SUNStepper_*` /
//! `SUNAdjointCheckpointScheme_*` dispatch, so no borrow is ever held across a
//! call that can re-enter this object — ARKODE calls
//! `SUNAdjointStepper_RecomputeFwd` from inside the adjoint stepper's own RHS.
//!
//! Release-build fidelity: the reference build has
//! `SUNDIALS_ENABLE_ERROR_CHECKS` **off**, so `SUNCheckCall(x)` is `(void)x`
//! and `SUNAssert(...)` is removed entirely. `Evolve`, `OneStep` and
//! `RecomputeFwd` consequently return `SUN_SUCCESS` unconditionally in C — a
//! sub-stepper error code is evaluated and discarded, never propagated — and
//! this port reproduces that exactly with `let _ = ...`. Only
//! `SUNAdjointStepper_GetNumSteps` forwards a sub-call return value, because
//! C returns it directly rather than wrapping it in `SUNCheckCall`.
//!
//! `void* user_data` is `Option<Box<dyn Any>>` and `void* content` (which the
//! C constructor leaves *uninitialized* and nothing ever reads) is the usual
//! `RefCell<Box<dyn Any>>` seeded with the empty `()` placeholder.

use std::any::Any;
use std::cell::{Cell, RefCell};
use std::rc::Rc;

use crate::sundials_adjointcheckpointscheme::{
    SUNAdjointCheckpointScheme, SUNAdjointCheckpointScheme_EnableDense,
};
use crate::sundials_context::SUNContext;
use crate::sundials_errors::SUN_SUCCESS;
use crate::sundials_nvector::N_Vector;
use crate::sundials_stepper::{
    SUNStepper, SUNStepper_Destroy, SUNStepper_Evolve, SUNStepper_GetNumSteps, SUNStepper_OneStep,
    SUNStepper_ReInit, SUNStepper_Reset, SUNStepper_ResetCheckpointIndex, SUNStepper_SetStopTime,
};
use crate::sundials_types::*;
use crate::sundials_utils::{sunfprintf_long, SUNFile};

/* -----------------------------------------------------------------
 * User-supplied function types (include/sundials/sundials_adjointstepper.h)
 * ----------------------------------------------------------------- */

/// C `SUNAdjRhsFn`: `int (*)(sunrealtype t, N_Vector y, N_Vector sens,
/// N_Vector sens_dot, void* user_data)`.
pub type SUNAdjRhsFn = fn(
    t: sunrealtype,
    y: &N_Vector,
    sens: &N_Vector,
    sens_dot: &N_Vector,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32;

/* -----------------------------------------------------------------
 * struct SUNAdjointStepper_ (src/sundials/sundials_adjointstepper_impl.h)
 * ----------------------------------------------------------------- */

pub struct SUNAdjointStepper_ {
    pub nrecompute: Cell<suncountertype>,
    pub final_step_idx: Cell<suncountertype>,

    pub adj_sunstepper: RefCell<SUNStepper>,
    pub fwd_sunstepper: RefCell<SUNStepper>,
    pub own_adj_sunstepper: Cell<sunbooleantype>,
    pub own_fwd_sunstepper: Cell<sunbooleantype>,
    pub checkpoint_scheme: RefCell<SUNAdjointCheckpointScheme>,

    /// C `void* user_data` — taken/restored around user callbacks.
    pub user_data: RefCell<Option<Box<dyn Any>>>,
    /// C `void* content` — never written by the C class (left uninitialized
    /// by `SUNAdjointStepper_Create`) and never read; kept for parity.
    pub content: RefCell<Box<dyn Any>>,
    pub sunctx: RefCell<SUNContext>,

    pub tf: Cell<sunrealtype>,
}

pub type SUNAdjointStepper = Rc<SUNAdjointStepper_>;

/* -----------------------------------------------------------------
 * exported functions
 * ----------------------------------------------------------------- */

/// The two `SUNStepper` handles are stored by value (C pointer copies whose
/// ownership is governed by `own_fwd`/`own_adj`), as is `checkpoint_scheme`
/// (never owned, never freed here). `sf` is `SUNDIALS_MAYBE_UNUSED` in C and
/// is likewise unused here.
pub fn SUNAdjointStepper_Create(
    fwd_sunstepper: SUNStepper,
    own_fwd: sunbooleantype,
    adj_sunstepper: SUNStepper,
    own_adj: sunbooleantype,
    final_step_idx: suncountertype,
    tf: sunrealtype,
    sf: &N_Vector,
    checkpoint_scheme: SUNAdjointCheckpointScheme,
    sunctx: &SUNContext,
    adj_stepper_ptr: &mut Option<SUNAdjointStepper>,
) -> SUNErrCode {
    /* C: SUNDIALS_MAYBE_UNUSED N_Vector sf */
    let _ = sf;

    /* C: malloc + SUNAssert(adj_stepper, SUN_ERR_MALLOC_FAIL) — the assert is
    removed in the reference build and `Rc::new` cannot fail. */
    let adj_stepper: SUNAdjointStepper = Rc::new(SUNAdjointStepper_ {
        nrecompute: Cell::new(0),
        final_step_idx: Cell::new(final_step_idx),

        adj_sunstepper: RefCell::new(adj_sunstepper),
        fwd_sunstepper: RefCell::new(fwd_sunstepper),
        own_adj_sunstepper: Cell::new(own_adj),
        own_fwd_sunstepper: Cell::new(own_fwd),
        checkpoint_scheme: RefCell::new(checkpoint_scheme),

        user_data: RefCell::new(None),
        content: RefCell::new(Box::new(())),
        sunctx: RefCell::new(sunctx.clone()),

        tf: Cell::new(tf),
    });

    *adj_stepper_ptr = Some(adj_stepper);

    SUN_SUCCESS
}

pub fn SUNAdjointStepper_ReInit(
    self_: &SUNAdjointStepper,
    t0: sunrealtype,
    y0: &N_Vector,
    tf: sunrealtype,
    sf: &N_Vector,
) -> SUNErrCode {
    self_.tf.set(tf);
    self_.nrecompute.set(0);
    /* C discards both SUNStepper_ReInit return codes (no SUNCheckCall). */
    let adj_sunstepper = self_.adj_sunstepper.borrow().clone();
    let _ = SUNStepper_ReInit(&adj_sunstepper, tf, sf);
    let fwd_sunstepper = self_.fwd_sunstepper.borrow().clone();
    let _ = SUNStepper_ReInit(&fwd_sunstepper, t0, y0);
    SUN_SUCCESS
}

pub fn SUNAdjointStepper_Evolve(
    self_: &SUNAdjointStepper,
    tout: sunrealtype,
    sens: &N_Vector,
    tret: &mut sunrealtype,
) -> SUNErrCode {
    let adj_sunstepper = self_.adj_sunstepper.borrow().clone();
    /* C: SUNCheckCall(...) → (void) in the reference build */
    let _ = SUNStepper_Evolve(&adj_sunstepper, tout, sens, tret);
    SUN_SUCCESS
}

pub fn SUNAdjointStepper_OneStep(
    self_: &SUNAdjointStepper,
    tout: sunrealtype,
    sens: &N_Vector,
    tret: &mut sunrealtype,
) -> SUNErrCode {
    let adj_sunstepper = self_.adj_sunstepper.borrow().clone();
    /* C: SUNCheckCall(...) → (void) in the reference build */
    let _ = SUNStepper_OneStep(&adj_sunstepper, tout, sens, tret);
    SUN_SUCCESS
}

pub fn SUNAdjointStepper_RecomputeFwd(
    self_: &SUNAdjointStepper,
    start_idx: suncountertype,
    t0: sunrealtype,
    y0: &N_Vector,
    tf: sunrealtype,
) -> SUNErrCode {
    let retcode: SUNErrCode = SUN_SUCCESS;

    let mut fwd_t = t0;
    let fwd_stepper = self_.fwd_sunstepper.borrow().clone();
    let _ = SUNStepper_Reset(&fwd_stepper, t0, y0);
    let _ = SUNStepper_ResetCheckpointIndex(&fwd_stepper, start_idx);

    let checkpoint_scheme = self_.checkpoint_scheme.borrow().clone();
    let _ = SUNAdjointCheckpointScheme_EnableDense(&checkpoint_scheme, SUNTRUE);

    let _ = SUNStepper_SetStopTime(&fwd_stepper, tf);

    /* C leaves nst_before/nst_after uninitialized; they are unconditionally
    written by SUNStepper_GetNumSteps on every path a valid stepper takes. */
    let mut nst_before: suncountertype = 0;
    let mut nst_after: suncountertype = 0;
    let _ = SUNStepper_GetNumSteps(&fwd_stepper, &mut nst_before);
    let _ = SUNStepper_Evolve(&fwd_stepper, tf, y0, &mut fwd_t);
    let _ = SUNStepper_GetNumSteps(&fwd_stepper, &mut nst_after);
    self_
        .nrecompute
        .set(self_.nrecompute.get() + (nst_after - nst_before));

    let _ = SUNAdjointCheckpointScheme_EnableDense(&checkpoint_scheme, SUNFALSE);

    retcode
}

/// C `SUNAdjointStepper_Destroy(SUNAdjointStepper* self_ptr)`.
///
/// `SUNAdjointStepper self = *self_ptr;` followed by `self->own_fwd_sunstepper`
/// dereferences a NULL handle in C when `*self_ptr` is NULL — deviation class 5
/// (deterministic panic at the same site). The owned sub-steppers are handed to
/// `SUNStepper_Destroy` through a temporary `Option` (C passes `&self->field`
/// and has it nulled); the field itself is not nulled because `self` is freed
/// on the next line, so no reader can observe the difference.
pub fn SUNAdjointStepper_Destroy(self_ptr: &mut Option<SUNAdjointStepper>) -> SUNErrCode {
    let self_ = self_ptr
        .as_ref()
        .expect("SUNAdjointStepper_Destroy: NULL SUNAdjointStepper")
        .clone();
    if self_.own_fwd_sunstepper.get() {
        let mut fwd_sunstepper = Some(self_.fwd_sunstepper.borrow().clone());
        let _ = SUNStepper_Destroy(&mut fwd_sunstepper);
    }
    if self_.own_adj_sunstepper.get() {
        let mut adj_sunstepper = Some(self_.adj_sunstepper.borrow().clone());
        let _ = SUNStepper_Destroy(&mut adj_sunstepper);
    }
    drop(self_); /* C: free(self) */
    *self_ptr = None;
    SUN_SUCCESS
}

/// C `void* user_data` becomes an owned `Option<Box<dyn Any>>` token; ARKODE
/// takes it out of `self_.user_data` and restores it around every adjoint RHS
/// invocation (deviation class 6: the C pointer aliases the caller's
/// `user_data`, a `Box` cannot).
pub fn SUNAdjointStepper_SetUserData(
    self_: &SUNAdjointStepper,
    user_data: Option<Box<dyn Any>>,
) -> SUNErrCode {
    *self_.user_data.borrow_mut() = user_data;

    SUN_SUCCESS
}

pub fn SUNAdjointStepper_GetNumSteps(
    self_: &SUNAdjointStepper,
    num_steps: &mut suncountertype,
) -> SUNErrCode {
    let adj_sunstepper = self_.adj_sunstepper.borrow().clone();
    SUNStepper_GetNumSteps(&adj_sunstepper, num_steps)
}

pub fn SUNAdjointStepper_GetNumRecompute(
    self_: &SUNAdjointStepper,
    num_recompute: &mut suncountertype,
) -> SUNErrCode {
    *num_recompute = self_.nrecompute.get();
    SUN_SUCCESS
}

pub fn SUNAdjointStepper_PrintAllStats(
    self_: &SUNAdjointStepper,
    outfile: &SUNFile,
    fmt: SUNOutputFormat,
) -> SUNErrCode {
    let mut nst: suncountertype = 0;
    let adj_sunstepper = self_.adj_sunstepper.borrow().clone();
    /* C: SUNCheckCall(...) → (void) in the reference build */
    let _ = SUNStepper_GetNumSteps(&adj_sunstepper, &mut nst);
    sunfprintf_long(outfile, fmt, SUNTRUE, "Num backwards steps", nst);
    sunfprintf_long(
        outfile,
        fmt,
        SUNFALSE,
        "Num recompute steps",
        self_.nrecompute.get(),
    );

    SUN_SUCCESS
}
