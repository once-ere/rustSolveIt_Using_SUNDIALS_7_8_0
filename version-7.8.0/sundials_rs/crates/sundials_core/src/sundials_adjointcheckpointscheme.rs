//! Port of `src/sundials/sundials_adjointcheckpointscheme.c` +
//! `src/sundials/sundials_adjointcheckpointscheme_impl.h` +
//! `include/sundials/sundials_adjointcheckpointscheme.h`
//! (the generic SUNAdjointCheckpointScheme base class: an ops table for
//! insert/load/remove of checkpoint vectors).
//!
//! Handle model: `SUNAdjointCheckpointScheme = Rc<SUNAdjointCheckpointScheme_>`
//! where the struct holds `content: RefCell<Box<dyn Any>>` (C `void* content`,
//! NULL = the `Box::new(())` placeholder installed by `NewEmpty`), the ops
//! table of plain `Option<fn>` pointers taking `&SUNAdjointCheckpointScheme`
//! (identical call shape to C), and the `sunctx` handle. Cloning the `Rc` is
//! the C pointer copy; `Rc::ptr_eq` is C pointer equality; dropping frees.
//! Implementations (the Fixed scheme) install their ops by assigning through
//! `check_scheme.ops.borrow_mut()`, exactly as C assigns
//! `check_scheme->ops->needssaving = ...`, store their content through
//! `check_scheme.content.borrow_mut()`, and reach it again through a private
//! `content_mut()` downcast helper.
//!
//! Mapping notes:
//! - `SUNFunctionBegin(...)` only declares a local `SUNContext` (error checks
//!   are off) and profiling is off, so `SUNDIALS_MARK_FUNCTION_BEGIN/END`
//!   compile away; both are omitted at translation time. The one observable
//!   part of `SUNFunctionBegin((*check_scheme_ptr)->sunctx)` in `Destroy` is
//!   its dereference of a possibly-NULL handle — kept as a panic at the same
//!   site (accepted deviation class 5).
//! - `SUNAssert` is a no-op in the reference build and a Rust allocation
//!   failure aborts rather than returning, so `NewEmpty` always reports
//!   `SUN_SUCCESS`, exactly as the reference build does.
//! - C `SUNAdjointCheckpointScheme*` (pointer-to-handle, used as both an
//!   out-param and a destroy in/out-param) becomes
//!   `&mut Option<SUNAdjointCheckpointScheme>`: NULL handle = `None`, same
//!   argument position and name as in C.
//! - Every dispatcher copies the `Option<fn>` out of the ops `RefCell` and
//!   drops that borrow BEFORE calling it: an implementation op (notably
//!   `destroy`) may re-enter this object or drop it outright.
//! - `SUNAdjointCheckpointScheme_GetContent` follows the deviation-class-6
//!   swap protocol for `void**` getters (see `CVodeGetUserData`): it swaps
//!   the content box with the caller's out-param, and the caller must hand
//!   the box back (via `SetContent`, or a second `GetContent`) before any op
//!   runs on this object again.
//!
//! The C++ view header `include/sundials/sundials_adjointcheckpointscheme.hpp`
//! contains only a `std::unique_ptr` deleter calling
//! `SUNAdjointCheckpointScheme_Destroy`; `Rc`'s own `Drop` covers it, so
//! nothing from that header is ported.

use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

use crate::sundials_context::SUNContext;
use crate::sundials_errors::{SUN_ERR_NOT_IMPLEMENTED, SUN_SUCCESS};
use crate::sundials_nvector::N_Vector;
use crate::sundials_types::*;

/* -----------------------------------------------------------------
 * Checkpoint scheme operation function types
 * (include/sundials/sundials_adjointcheckpointscheme.h)
 * ----------------------------------------------------------------- */

pub type SUNAdjointCheckpointSchemeNeedsSavingFn = fn(
    check_scheme: &SUNAdjointCheckpointScheme,
    step_num: suncountertype,
    stage_num: suncountertype,
    t: sunrealtype,
    yes_or_no: &mut sunbooleantype,
) -> SUNErrCode;

pub type SUNAdjointCheckpointSchemeInsertVectorFn = fn(
    check_scheme: &SUNAdjointCheckpointScheme,
    step_num: suncountertype,
    stage_num: suncountertype,
    t: sunrealtype,
    y: &N_Vector,
) -> SUNErrCode;

/// C `N_Vector* yout` is a pointer to the caller's own vector handle; the
/// checkpoint data is copied into `*yout` (see the Fixed scheme's
/// `SUNDataNode_GetDataNvector(solution_node, *yout, tout)`), so this is a
/// plain `&mut N_Vector` in the same position and name.
pub type SUNAdjointCheckpointSchemeLoadVectorFn = fn(
    check_scheme: &SUNAdjointCheckpointScheme,
    step_num: suncountertype,
    stage_num: suncountertype,
    peek: sunbooleantype,
    yout: &mut N_Vector,
    tout: &mut sunrealtype,
) -> SUNErrCode;

pub type SUNAdjointCheckpointSchemeDestroyFn =
    fn(check_scheme: &mut Option<SUNAdjointCheckpointScheme>) -> SUNErrCode;

pub type SUNAdjointCheckpointSchemeEnableDenseFn =
    fn(check_scheme: &SUNAdjointCheckpointScheme, on_or_off: sunbooleantype) -> SUNErrCode;

/* -----------------------------------------------------------------
 * SUNAdjointCheckpointScheme private class definition
 * (src/sundials/sundials_adjointcheckpointscheme_impl.h)
 * ----------------------------------------------------------------- */

/// C `struct SUNAdjointCheckpointScheme_Ops_` — every member is NULL
/// (`None`) until an implementation or a `Set*Fn` call installs it.
#[derive(Default, Clone)]
pub struct SUNAdjointCheckpointScheme_Ops_ {
    pub needssaving: Option<SUNAdjointCheckpointSchemeNeedsSavingFn>,
    pub insertvector: Option<SUNAdjointCheckpointSchemeInsertVectorFn>,
    pub loadvector: Option<SUNAdjointCheckpointSchemeLoadVectorFn>,
    pub destroy: Option<SUNAdjointCheckpointSchemeDestroyFn>,
    pub enableDense: Option<SUNAdjointCheckpointSchemeEnableDenseFn>,
}

/// C `typedef struct SUNAdjointCheckpointScheme_Ops_* ...` — the ops table
/// lives inline in the handle here, so the alias names the struct itself.
pub type SUNAdjointCheckpointScheme_Ops = SUNAdjointCheckpointScheme_Ops_;

/// C `struct SUNAdjointCheckpointScheme_`.
pub struct SUNAdjointCheckpointScheme_ {
    pub ops: RefCell<SUNAdjointCheckpointScheme_Ops_>,
    /// C `void* content` — `Box::new(())` stands for NULL.
    pub content: RefCell<Box<dyn Any>>,
    pub sunctx: RefCell<SUNContext>,
}

pub type SUNAdjointCheckpointScheme = Rc<SUNAdjointCheckpointScheme_>;

/* -----------------------------------------------------------------
 * "static" base class methods
 * ----------------------------------------------------------------- */

pub fn SUNAdjointCheckpointScheme_NewEmpty(
    sunctx: &SUNContext,
    check_scheme_ptr: &mut Option<SUNAdjointCheckpointScheme>,
) -> SUNErrCode {
    let self_ = Rc::new(SUNAdjointCheckpointScheme_ {
        /* ops->needssaving = ops->insertvector = ops->loadvector
        = ops->enableDense = ops->destroy = NULL */
        ops: RefCell::new(SUNAdjointCheckpointScheme_Ops_::default()),
        /* self->content = NULL */
        content: RefCell::new(Box::new(())),
        sunctx: RefCell::new(sunctx.clone()),
    });

    *check_scheme_ptr = Some(self_);

    SUN_SUCCESS
}

/* -----------------------------------------------------------------
 * Virtual (overridable) base class methods
 * ----------------------------------------------------------------- */

pub fn SUNAdjointCheckpointScheme_NeedsSaving(
    self_: &SUNAdjointCheckpointScheme,
    step_num: suncountertype,
    stage_num: suncountertype,
    t: sunrealtype,
    yes_or_no: &mut sunbooleantype,
) -> SUNErrCode {
    let f = self_.ops.borrow().needssaving;
    if let Some(f) = f {
        let err = f(self_, step_num, stage_num, t, yes_or_no);
        return err;
    }
    SUN_ERR_NOT_IMPLEMENTED
}

pub fn SUNAdjointCheckpointScheme_InsertVector(
    self_: &SUNAdjointCheckpointScheme,
    step_num: suncountertype,
    stage_num: suncountertype,
    t: sunrealtype,
    state: &N_Vector,
) -> SUNErrCode {
    let f = self_.ops.borrow().insertvector;
    if let Some(f) = f {
        let err = f(self_, step_num, stage_num, t, state);
        return err;
    }
    SUN_ERR_NOT_IMPLEMENTED
}

pub fn SUNAdjointCheckpointScheme_LoadVector(
    self_: &SUNAdjointCheckpointScheme,
    step_num: suncountertype,
    stage_num: suncountertype,
    peek: sunbooleantype,
    out: &mut N_Vector,
    tout: &mut sunrealtype,
) -> SUNErrCode {
    let f = self_.ops.borrow().loadvector;
    if let Some(f) = f {
        let err = f(self_, step_num, stage_num, peek, out, tout);
        return err;
    }
    SUN_ERR_NOT_IMPLEMENTED
}

pub fn SUNAdjointCheckpointScheme_Destroy(
    check_scheme_ptr: &mut Option<SUNAdjointCheckpointScheme>,
) -> SUNErrCode {
    /* C's SUNFunctionBegin((*check_scheme_ptr)->sunctx) dereferences the
    handle unconditionally, so a NULL handle is a NULL dereference there —
    a deterministic panic at the same site here. The ops borrow ends with
    this statement: `destroy` drops the object. */
    let f = check_scheme_ptr
        .as_ref()
        .expect("SUNAdjointCheckpointScheme_Destroy: NULL check_scheme")
        .ops
        .borrow()
        .destroy;

    if let Some(f) = f {
        let err = f(check_scheme_ptr);
        return err;
    } else if check_scheme_ptr.is_some() {
        /* C: free((*check_scheme_ptr)->ops); free(*check_scheme_ptr);
        (the content is assumed to be empty already, and C leaves the
        caller's now-dangling pointer alone). Dropping the handle releases
        ops, content and self together, and clears the caller's handle. */
        *check_scheme_ptr = None;
    }
    SUN_SUCCESS
}

pub fn SUNAdjointCheckpointScheme_EnableDense(
    self_: &SUNAdjointCheckpointScheme,
    on_or_off: sunbooleantype,
) -> SUNErrCode {
    let f = self_.ops.borrow().enableDense;
    if let Some(f) = f {
        let err = f(self_, on_or_off);
        return err;
    }
    SUN_ERR_NOT_IMPLEMENTED
}

/* -----------------------------------------------------------------
 * Base class methods
 * ----------------------------------------------------------------- */

/// C `SUNAdjointCheckpointScheme_SetContent(self, void* content)`.
///
/// The implementation content is handed over as an owned `Box<dyn Any>`
/// (C NULL = `Box::new(())`), replacing whatever the handle held.
pub fn SUNAdjointCheckpointScheme_SetContent(
    self_: &SUNAdjointCheckpointScheme,
    content: Box<dyn Any>,
) -> SUNErrCode {
    *self_.content.borrow_mut() = content;
    SUN_SUCCESS
}

/// C `SUNAdjointCheckpointScheme_GetContent(self, void** content)`.
///
/// A `Box<dyn Any>` cannot be aliased, so the content box is SWAPPED with
/// the caller's out-param (deviation class 6, as `CVodeGetUserData`). The
/// caller must hand the box back (`SUNAdjointCheckpointScheme_SetContent`,
/// or a second `GetContent` call) before any op runs on this object again.
pub fn SUNAdjointCheckpointScheme_GetContent(
    self_: &SUNAdjointCheckpointScheme,
    content: &mut Box<dyn Any>,
) -> SUNErrCode {
    std::mem::swap(&mut *self_.content.borrow_mut(), content);
    SUN_SUCCESS
}

pub fn SUNAdjointCheckpointScheme_SetNeedsSavingFn(
    self_: &SUNAdjointCheckpointScheme,
    fn_: Option<SUNAdjointCheckpointSchemeNeedsSavingFn>,
) -> SUNErrCode {
    self_.ops.borrow_mut().needssaving = fn_;
    SUN_SUCCESS
}

pub fn SUNAdjointCheckpointScheme_SetInsertVectorFn(
    self_: &SUNAdjointCheckpointScheme,
    fn_: Option<SUNAdjointCheckpointSchemeInsertVectorFn>,
) -> SUNErrCode {
    self_.ops.borrow_mut().insertvector = fn_;
    SUN_SUCCESS
}

pub fn SUNAdjointCheckpointScheme_SetLoadVectorFn(
    self_: &SUNAdjointCheckpointScheme,
    fn_: Option<SUNAdjointCheckpointSchemeLoadVectorFn>,
) -> SUNErrCode {
    self_.ops.borrow_mut().loadvector = fn_;
    SUN_SUCCESS
}

pub fn SUNAdjointCheckpointScheme_SetDestroyFn(
    self_: &SUNAdjointCheckpointScheme,
    fn_: Option<SUNAdjointCheckpointSchemeDestroyFn>,
) -> SUNErrCode {
    self_.ops.borrow_mut().destroy = fn_;
    SUN_SUCCESS
}

pub fn SUNAdjointCheckpointScheme_SetEnableDenseFn(
    self_: &SUNAdjointCheckpointScheme,
    fn_: Option<SUNAdjointCheckpointSchemeEnableDenseFn>,
) -> SUNErrCode {
    self_.ops.borrow_mut().enableDense = fn_;
    SUN_SUCCESS
}
