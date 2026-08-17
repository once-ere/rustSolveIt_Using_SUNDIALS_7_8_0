//! Port of `src/arkode/arkode_user_controller.c` +
//! `src/arkode/arkode_user_controller.h`: the `ARKUserControl`
//! `SUNAdaptController` implementation, which provides backwards
//! compatibility for ARKODE's previous `ARKAdaptFn`.
//!
//! Binding notes:
//! * The `SC_*` accessor macros become the house-standard `content_mut`
//!   downcast guard. The guard is never held across the user's `ARKAdaptFn`
//!   or across a borrow of `ark_mem`.
//! * `content->ark_mem` is a NON-OWNING back-pointer in C (the controller is
//!   owned by `ark_mem->hadapt_mem->hcontroller`), so it maps to
//!   `Weak<RefCell<ARKodeMemRec>>` -- exactly the treatment
//!   `sundatanode_inmem.rs` gives `content->parent` -- and an owning `Rc`
//!   clone would make an uncollectable cycle. It is upgraded at each use;
//!   a dead handle is C's dangling pointer and panics here (deviation
//!   class 5).
//! * `hadapt_data` is the C `void*` callback token (`Option<Box<dyn Any>>`):
//!   `EstimateStep` `Option::take`s it, calls, and restores it on every path.
//! * `arkode_mem == NULL` / `sunctx == NULL` guards in `ARKUserControl` are
//!   unrepresentable; the `hadapt == NULL` guard survives as `Option`.
//!
//! Accepted deviation (class 5, unobservable): `SUNAdaptController_Write_…`
//! cannot reproduce C's `%p` rendering of `hadapt_data` and prints a fixed
//! placeholder instead.

use std::any::Any;
use std::cell::{RefCell, RefMut};
use std::rc::{Rc, Weak};

use crate::arkode_impl::*;
use sundials_core::sundials_adaptcontroller::*;
use sundials_core::sundials_context::SUNContext;
use sundials_core::sundials_errors::{SUN_ERR_USER_FCN_FAIL, SUN_SUCCESS};
use sundials_core::sundials_types::*;
use sundials_core::sundials_utils::{sun_format_g, SUNFile};

/* ---------------------------------------------------
 * ARKUserControl implementation of SUNAdaptController
 * --------------------------------------------------- */

pub struct _ARKUserControlContent {
    pub hp: sunrealtype,  /* h from previous step */
    pub hpp: sunrealtype, /* h from 2 steps ago */
    pub ep: sunrealtype,  /* error from previous step */
    pub epp: sunrealtype, /* error from 2 steps ago */
    /// C `ARKodeMem ark_mem` -- main ARKODE memory structure, a non-owning
    /// back-pointer (see the module docs).
    pub ark_mem: Weak<RefCell<ARKodeMemRec>>,
    pub hadapt: Option<ARKAdaptFn>, /* user-provided adaptivity fn */
    pub hadapt_data: Option<Box<dyn Any>>, /* user-provided data pointer */
}

pub type ARKUserControlContent = _ARKUserControlContent;

/* ---------------
 * Macro accessors
 * --------------- */

/// C `SC_CONTENT(C)` / `SC_HP(C)` / ... Panics if the controller is not an
/// `ARKUserControl` (C would blindly cast the `void*`; deviation class 5).
/// NEVER hold the guard across the user's `ARKAdaptFn` or a borrow of the
/// ARKODE memory.
fn content_mut(C: &SUNAdaptController) -> RefMut<'_, _ARKUserControlContent> {
    RefMut::map(C.content.borrow_mut(), |c| {
        c.downcast_mut::<_ARKUserControlContent>()
            .expect("ARKUserControl SUNAdaptController content")
    })
}

/* -----------------------------------------------------------------
 * exported functions
 * ----------------------------------------------------------------- */

/* -----------------------------------------------------------------
 * Function to create a new ARKUserControl controller
 * ----------------------------------------------------------------- */

pub fn ARKUserControl(
    sunctx: &SUNContext,
    arkode_mem: &ARKodeMem,
    hadapt: Option<ARKAdaptFn>,
    hadapt_data: Option<Box<dyn Any>>,
) -> Option<SUNAdaptController> {
    /* Return with failure if hadapt, arkode_mem, or context are NULL
    (only the `hadapt` test is representable) */
    if hadapt.is_none() {
        return None;
    }

    /* Create an empty controller object */
    let C = SUNAdaptController_NewEmpty(sunctx)?;

    /* Attach operations */
    {
        let mut ops = C.ops.borrow_mut();
        ops.gettype = Some(SUNAdaptController_GetType_ARKUserControl);
        ops.estimatestep = Some(SUNAdaptController_EstimateStep_ARKUserControl);
        ops.reset = Some(SUNAdaptController_Reset_ARKUserControl);
        ops.write = Some(SUNAdaptController_Write_ARKUserControl);
        ops.updateh = Some(SUNAdaptController_UpdateH_ARKUserControl);
        ops.space = Some(SUNAdaptController_Space_ARKUserControl);
    }

    /* Create content (C `malloc` cannot fail here) and attach content */
    /* Attach ARKODE memory structure */
    /* Attach user-provided adaptivity function and data */
    *C.content.borrow_mut() = Box::new(_ARKUserControlContent {
        hp: 0.0,
        hpp: 0.0,
        ep: 0.0,
        epp: 0.0,
        ark_mem: Rc::downgrade(arkode_mem),
        hadapt,
        hadapt_data,
    });

    /* Fill content with default/reset values */
    let _ = SUNAdaptController_Reset_ARKUserControl(&C);

    Some(C)
}

/* -----------------------------------------------------------------
 * implementation of controller operations
 * ----------------------------------------------------------------- */

pub fn SUNAdaptController_GetType_ARKUserControl(
    _C: &SUNAdaptController,
) -> SUNAdaptController_Type {
    SUN_ADAPTCONTROLLER_H
}

pub fn SUNAdaptController_EstimateStep_ARKUserControl(
    C: &SUNAdaptController,
    h: sunrealtype,
    _p: i32,
    dsm: sunrealtype,
    hnew: &mut sunrealtype,
) -> SUNErrCode {
    /* call user-provided function to compute new step */
    let (ark_mem, hadapt, hp, hpp, ep, epp) = {
        let c = content_mut(C);
        (
            c.ark_mem.upgrade().expect("ARKUserControl ark_mem"),
            c.hadapt.expect("ARKUserControl hadapt"),
            c.hp,
            c.hpp,
            c.ep,
            c.epp,
        )
    };
    let (ttmp, ycur, hadapt_q, hadapt_p) = {
        let m = ark_mem.borrow();
        let ttmp = if dsm <= ONE { m.tn + m.h } else { m.tn };
        let hadapt_mem = m.hadapt_mem.as_ref().expect("hadapt_mem");
        (
            ttmp,
            m.ycur.clone().expect("ycur"),
            hadapt_mem.q,
            hadapt_mem.p,
        )
    };
    let mut hadapt_data = content_mut(C).hadapt_data.take();
    let retval = hadapt(
        &ycur,
        ttmp,
        h,
        hp,
        hpp,
        dsm,
        ep,
        epp,
        hadapt_q,
        hadapt_p,
        hnew,
        &mut hadapt_data,
    );
    content_mut(C).hadapt_data = hadapt_data;
    if retval != SUN_SUCCESS {
        return SUN_ERR_USER_FCN_FAIL;
    }
    SUN_SUCCESS
}

pub fn SUNAdaptController_Reset_ARKUserControl(C: &SUNAdaptController) -> SUNErrCode {
    let mut c = content_mut(C);
    c.ep = 1.0;
    c.epp = 1.0;
    c.hp = 0.0;
    c.hpp = 0.0;
    SUN_SUCCESS
}

pub fn SUNAdaptController_Write_ARKUserControl(
    C: &SUNAdaptController,
    fptr: &SUNFile,
) -> SUNErrCode {
    let c = content_mut(C);
    fptr.write_str("ARKUserControl module:\n");
    fptr.write_str(&format!("  hp = {}\n", sun_format_g(c.hp)));
    fptr.write_str(&format!("  hpp = {}\n", sun_format_g(c.hpp)));
    fptr.write_str(&format!("  ep = {}\n", sun_format_g(c.ep)));
    fptr.write_str(&format!("  epp = {}\n", sun_format_g(c.epp)));
    /* C prints the raw `void* hadapt_data` with "%p"; a `Box<dyn Any>` has no
    reproducible textual address, so a fixed placeholder is printed instead
    (deviation class 5). */
    fptr.write_str(&format!(
        "  hadapt_data = {}\n",
        if c.hadapt_data.is_some() {
            "(data)"
        } else {
            "(nil)"
        }
    ));
    SUN_SUCCESS
}

pub fn SUNAdaptController_UpdateH_ARKUserControl(
    C: &SUNAdaptController,
    h: sunrealtype,
    dsm: sunrealtype,
) -> SUNErrCode {
    let mut c = content_mut(C);
    c.hpp = c.hp;
    c.hp = h;
    c.epp = c.ep;
    c.ep = dsm;
    SUN_SUCCESS
}

pub fn SUNAdaptController_Space_ARKUserControl(
    _C: &SUNAdaptController,
    lenrw: &mut i64,
    leniw: &mut i64,
) -> SUNErrCode {
    *lenrw = 4;
    *leniw = 2;
    SUN_SUCCESS
}
