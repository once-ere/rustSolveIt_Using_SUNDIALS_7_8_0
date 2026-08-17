//! Port of `src/sundials/sundials_context.c` +
//! `include/sundials/sundials_context.h` +
//! `include/sundials/priv/sundials_context_impl.h`.
//!
//! Reference-build configuration: `SUNDIALS_LOGGING_LEVEL = 2` (logger
//! created from the environment) and profiling disabled (no profiler).
//! The C singly-linked error handler list is a Rust `Vec` used as a stack
//! (last element = most recently pushed handler).

use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

use crate::sundials_errors::{
    SUNErrHandlerFn, SUNLogErrHandlerFn, SUN_ERR_CORRUPT, SUN_ERR_SUNCTX_CORRUPT, SUN_SUCCESS,
};
use crate::sundials_logger::{SUNLogger, SUNLogger_CreateFromEnv};
use crate::sundials_profiler::SUNProfiler;
use crate::sundials_types::*;

/// One node of the C `SUNErrHandler_` list (`previous` is implied by the
/// stack position in `SUNContext_::err_handler`).
pub struct SUNErrHandler_ {
    pub call: SUNErrHandlerFn,
    pub data: Option<Box<dyn Any>>,
}

pub struct SUNContext_ {
    pub profiler: Option<SUNProfiler>,
    pub own_profiler: sunbooleantype,
    pub logger: Option<SUNLogger>,
    pub own_logger: sunbooleantype,
    pub last_err: SUNErrCode,
    pub err_handler: Vec<SUNErrHandler_>,
    pub comm: SUNComm,
}

pub type SUNContext = Rc<RefCell<SUNContext_>>;

pub fn SUNContext_Create(comm: SUNComm, sunctx_out: &mut Option<SUNContext>) -> SUNErrCode {
    *sunctx_out = None;

    let mut logger: Option<SUNLogger> = None;

    /* SUNDIALS_LOGGING_LEVEL > 0, non-MPI branch */
    let err = SUNLogger_CreateFromEnv(SUN_COMM_NULL, &mut logger);
    if err != SUN_SUCCESS {
        return err;
    }

    /* profiling disabled: no profiler */

    let eh = SUNErrHandler_ {
        call: SUNLogErrHandlerFn,
        data: None,
    };

    let own_logger = logger.is_some();
    let sunctx = Rc::new(RefCell::new(SUNContext_ {
        profiler: None,
        own_profiler: SUNFALSE,
        logger,
        own_logger,
        last_err: SUN_SUCCESS,
        err_handler: vec![eh],
        comm,
    }));

    *sunctx_out = Some(sunctx);
    SUN_SUCCESS
}

pub fn SUNContext_GetLastError(sunctx: &SUNContext) -> SUNErrCode {
    let mut ctx = sunctx.borrow_mut();
    let err = ctx.last_err;
    ctx.last_err = SUN_SUCCESS;
    err
}

pub fn SUNContext_PeekLastError(sunctx: &SUNContext) -> SUNErrCode {
    sunctx.borrow().last_err
}

pub fn SUNContext_PushErrHandler(
    sunctx: &SUNContext,
    err_fn: SUNErrHandlerFn,
    err_user_data: Option<Box<dyn Any>>,
) -> SUNErrCode {
    let new_err_handler = SUNErrHandler_ {
        call: err_fn,
        data: err_user_data,
    };
    sunctx.borrow_mut().err_handler.push(new_err_handler);
    SUN_SUCCESS
}

pub fn SUNContext_PopErrHandler(sunctx: &SUNContext) -> SUNErrCode {
    sunctx.borrow_mut().err_handler.pop();
    SUN_SUCCESS
}

pub fn SUNContext_ClearErrHandlers(sunctx: &SUNContext) -> SUNErrCode {
    while !sunctx.borrow().err_handler.is_empty() {
        let _ = SUNContext_PopErrHandler(sunctx);
    }
    SUN_SUCCESS
}

pub fn SUNContext_GetProfiler(sunctx: &SUNContext, profiler: &mut Option<SUNProfiler>) -> SUNErrCode {
    /* profiling disabled in the reference configuration */
    let _ = sunctx;
    *profiler = None;
    SUN_SUCCESS
}

pub fn SUNContext_SetProfiler(sunctx: &SUNContext, profiler: Option<SUNProfiler>) -> SUNErrCode {
    /* silence warnings when profiling is disabled */
    let _ = sunctx;
    let _ = profiler;
    SUN_SUCCESS
}

pub fn SUNContext_GetLogger(sunctx: &SUNContext, logger: &mut Option<SUNLogger>) -> SUNErrCode {
    *logger = sunctx.borrow().logger.clone();
    SUN_SUCCESS
}

pub fn SUNContext_SetLogger(sunctx: &SUNContext, logger: Option<SUNLogger>) -> SUNErrCode {
    let mut ctx = sunctx.borrow_mut();
    /* free any existing logger (Rust: drop our handle) */
    if ctx.logger.is_some() && ctx.own_logger {
        ctx.logger = None;
    }
    ctx.logger = logger;
    ctx.own_logger = SUNFALSE;
    SUN_SUCCESS
}

pub fn SUNContext_Free(sunctx: &mut Option<SUNContext>) -> SUNErrCode {
    if let Some(ctx) = sunctx.as_ref() {
        SUNContext_ClearErrHandlers(ctx);
    }
    *sunctx = None;
    SUN_SUCCESS
}

const _: SUNErrCode = SUN_ERR_SUNCTX_CORRUPT;
const _: SUNErrCode = SUN_ERR_CORRUPT;
