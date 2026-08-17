//! Port of `src/sundials/sundials_errors.c` +
//! `include/sundials/sundials_errors.h` +
//! `include/sundials/priv/sundials_errors_impl.h`.
//!
//! Release-build note: the reference outputs come from builds without
//! `SUNDIALS_ENABLE_ERROR_CHECKS`, so the `SUNCheck*`/`SUNAssert` macros
//! compile to nothing (`SUNCheckCall(x)` → `(void)x`). Ported call sites
//! therefore just evaluate the call. The live machinery below is the error
//! handler stack invoked through `SUNHandleErrWithMsg` (used by the solver
//! `ProcessError` paths) and the default `SUNLogErrHandlerFn`.

use std::any::Any;

use crate::sundials_context::{SUNContext, SUNErrHandler_};
use crate::sundials_logger::{SUNLogger_Flush, SUNLogger_QueueMsg, SUN_LOGLEVEL_ALL, SUN_LOGLEVEL_ERROR};
use crate::sundials_types::*;
use crate::sundials_utils::sunCombineFileAndLine;

pub const SUN_SUCCESS: SUNErrCode = 0;
pub const SUN_ERR_MINIMUM: SUNErrCode = -10000;
pub const SUN_ERR_ARG_CORRUPT: SUNErrCode = -9999;
pub const SUN_ERR_ARG_INCOMPATIBLE: SUNErrCode = -9998;
pub const SUN_ERR_ARG_OUTOFRANGE: SUNErrCode = -9997;
pub const SUN_ERR_ARG_WRONGTYPE: SUNErrCode = -9996;
pub const SUN_ERR_ARG_DIMSMISMATCH: SUNErrCode = -9995;
pub const SUN_ERR_GENERIC: SUNErrCode = -9994;
pub const SUN_ERR_CORRUPT: SUNErrCode = -9993;
pub const SUN_ERR_OUTOFRANGE: SUNErrCode = -9992;
pub const SUN_ERR_FILE_OPEN: SUNErrCode = -9991;
pub const SUN_ERR_OP_FAIL: SUNErrCode = -9990;
pub const SUN_ERR_MEM_FAIL: SUNErrCode = -9989;
pub const SUN_ERR_MALLOC_FAIL: SUNErrCode = -9988;
pub const SUN_ERR_EXT_FAIL: SUNErrCode = -9987;
pub const SUN_ERR_DESTROY_FAIL: SUNErrCode = -9986;
pub const SUN_ERR_NOT_IMPLEMENTED: SUNErrCode = -9985;
pub const SUN_ERR_USER_FCN_FAIL: SUNErrCode = -9984;
pub const SUN_ERR_DATANODE_NODENOTFOUND: SUNErrCode = -9983;
pub const SUN_ERR_PROFILER_MAPFULL: SUNErrCode = -9982;
pub const SUN_ERR_PROFILER_MAPGET: SUNErrCode = -9981;
pub const SUN_ERR_PROFILER_MAPINSERT: SUNErrCode = -9980;
pub const SUN_ERR_PROFILER_MAPKEYNOTFOUND: SUNErrCode = -9979;
pub const SUN_ERR_PROFILER_MAPSORT: SUNErrCode = -9978;
pub const SUN_ERR_ADJOINT_STEPPERFAILED: SUNErrCode = -9977;
pub const SUN_ERR_ADJOINT_STEPPERINVALIDSTOP: SUNErrCode = -9976;
pub const SUN_ERR_CHECKPOINT_NOT_FOUND: SUNErrCode = -9975;
pub const SUN_ERR_CHECKPOINT_MISMATCH: SUNErrCode = -9974;
pub const SUN_ERR_SUNCTX_CORRUPT: SUNErrCode = -9973;
pub const SUN_ERR_MPI_FAIL: SUNErrCode = -9972;
pub const SUN_ERR_UNREACHABLE: SUNErrCode = -9971;
pub const SUN_ERR_UNKNOWN: SUNErrCode = -9970;
pub const SUN_ERR_MAXIMUM: SUNErrCode = -1000;

/// C `SUNErrHandlerFn` (`err_user_data` is the handler node's own data).
pub type SUNErrHandlerFn = fn(
    line: i32,
    func: &str,
    file: &str,
    msg: &str,
    err_code: SUNErrCode,
    err_user_data: &mut Option<Box<dyn Any>>,
    sunctx: &SUNContext,
);

pub fn SUNGetErrMsg(code: SUNErrCode) -> &'static str {
    match code {
        SUN_ERR_ARG_CORRUPT => "argument provided is NULL or corrupted",
        SUN_ERR_ARG_INCOMPATIBLE => "argument provided is not compatible",
        SUN_ERR_ARG_OUTOFRANGE => "argument is out of the valid range",
        SUN_ERR_ARG_WRONGTYPE => "argument provided is not the right type",
        SUN_ERR_ARG_DIMSMISMATCH => "argument dimensions do not agree",
        SUN_ERR_GENERIC => "an error occurred",
        SUN_ERR_CORRUPT => "value is NULL or corrupt",
        SUN_ERR_OUTOFRANGE => "Value is out of the expected range",
        SUN_ERR_FILE_OPEN => "Unable to open file",
        SUN_ERR_OP_FAIL => "an operation failed",
        SUN_ERR_MEM_FAIL => "a memory operation failed",
        SUN_ERR_MALLOC_FAIL => "malloc returned NULL",
        SUN_ERR_EXT_FAIL => "a failure occurred in an external library",
        SUN_ERR_DESTROY_FAIL => "a destroy function returned an error",
        SUN_ERR_NOT_IMPLEMENTED => "operation is not implemented: function pointer is NULL",
        SUN_ERR_USER_FCN_FAIL => "the user provided callback function failed",
        SUN_ERR_DATANODE_NODENOTFOUND => "the data node could not be found",
        SUN_ERR_PROFILER_MAPFULL => {
            "the number of profiler entries exceeded SUNPROFILER_MAX_ENTRIES"
        }
        SUN_ERR_PROFILER_MAPGET => "unknown error getting SUNProfiler timer",
        SUN_ERR_PROFILER_MAPINSERT => "unknown error inserting SUNProfiler timer",
        SUN_ERR_PROFILER_MAPKEYNOTFOUND => "timer was not found in SUNProfiler",
        SUN_ERR_PROFILER_MAPSORT => "error sorting SUNProfiler map",
        SUN_ERR_ADJOINT_STEPPERFAILED => {
            "SUNStepper stopped without successfully reaching the requested \
             output time when solving the adjoint system"
        }
        SUN_ERR_ADJOINT_STEPPERINVALIDSTOP => {
            "SUNStepper stopped with a flag not supported by the adjoint solver"
        }
        SUN_ERR_CHECKPOINT_NOT_FOUND => "the requested checkpoint was not found",
        SUN_ERR_CHECKPOINT_MISMATCH => {
            "the expected time for the checkpoint and the stored time do not match"
        }
        SUN_ERR_SUNCTX_CORRUPT => "SUNContext is NULL or corrupt",
        SUN_ERR_MPI_FAIL => "an MPI call returned something other than MPI_SUCCESS",
        SUN_ERR_UNREACHABLE => {
            "Reached code that should be unreachable: open an issue at: \
             https://github.com/LLNL/sundials"
        }
        SUN_ERR_UNKNOWN => {
            "Unknown error occurred: open an issue at https://github.com/LLNL/sundials"
        }
        _ => "unknown error",
    }
}

pub fn SUNLogErrHandlerFn(
    line: i32,
    func: &str,
    file: &str,
    msg: &str,
    err_code: SUNErrCode,
    _err_user_data: &mut Option<Box<dyn Any>>,
    sunctx: &SUNContext,
) {
    let file_and_line = sunCombineFileAndLine(line, file);
    let msg_owned;
    let msg = if msg.is_empty() {
        msg_owned = SUNGetErrMsg(err_code).to_string();
        &msg_owned
    } else {
        msg
    };
    let logger = sunctx.borrow().logger.clone();
    if let Some(logger) = logger {
        SUNLogger_QueueMsg(&logger, SUN_LOGLEVEL_ERROR, &file_and_line, func, msg);
    }
}

pub fn SUNAbortErrHandlerFn(
    line: i32,
    func: &str,
    file: &str,
    msg: &str,
    err_code: SUNErrCode,
    _err_user_data: &mut Option<Box<dyn Any>>,
    sunctx: &SUNContext,
) {
    /* C signature is void (the abort never returns); declaring `()` keeps
    the fn item coercible to SUNErrHandlerFn at user call sites */
    /* Flush all buffered logging messages now before we abort */
    let logger = sunctx.borrow().logger.clone();
    if let Some(logger) = logger.as_ref() {
        SUNLogger_Flush(logger, SUN_LOGLEVEL_ALL);
    }

    let file_and_line = sunCombineFileAndLine(line, file);
    let msg_owned;
    let msg = if msg.is_empty() {
        msg_owned = SUNGetErrMsg(err_code).to_string();
        &msg_owned
    } else {
        msg
    };
    if let Some(logger) = logger.as_ref() {
        SUNLogger_QueueMsg(&logger.clone(), SUN_LOGLEVEL_ERROR, &file_and_line, func, msg);
        let file_and_line = sunCombineFileAndLine(line!() as i32 + 1, file!());
        SUNLogger_QueueMsg(
            &logger.clone(),
            SUN_LOGLEVEL_ERROR,
            &file_and_line,
            "SUNAbortErrHandlerFn",
            "SUNAbortErrHandler: Calling abort now, use a different error handler to \
             avoid program termination.\n",
        );
    }
    std::process::abort();
}

/// C `SUNGlobalFallbackErrHandler`: stderr fallback when the context is
/// NULL/corrupt (unreachable through the safe Rust API; kept for parity).
pub fn SUNGlobalFallbackErrHandler(
    line: i32,
    func: &str,
    file: &str,
    msgfmt: &str,
    err_code: SUNErrCode,
) {
    let file_and_line = sunCombineFileAndLine(line!() as i32, file!());
    eprintln!(
        "[ERROR][rank 0][{file_and_line}][SUNGlobalFallbackErrHandler] The SUNDIALS \
         SUNContext was corrupt or NULL when an error occurred. As such, error \
         messages have been printed to stderr."
    );
    let file_and_line = sunCombineFileAndLine(line, file);
    let msg = if msgfmt.is_empty() {
        SUNGetErrMsg(err_code)
    } else {
        msgfmt
    };
    eprintln!("[ERROR][rank 0][{file_and_line}][{func}] {msg}");
}

/// C `SUNHandleErrWithMsg` (priv impl header): set `last_err`, then call
/// every registered handler from newest to oldest.
pub fn SUNHandleErrWithMsg(
    line: i32,
    func: &str,
    file: &str,
    msg: &str,
    code: SUNErrCode,
    sunctx: &SUNContext,
) {
    sunctx.borrow_mut().last_err = code;
    /* Take the handler stack out so handlers can use the context freely. */
    let mut handlers: Vec<SUNErrHandler_> = std::mem::take(&mut sunctx.borrow_mut().err_handler);
    for eh in handlers.iter_mut().rev() {
        (eh.call)(line, func, file, msg, code, &mut eh.data, sunctx);
    }
    let mut ctx = sunctx.borrow_mut();
    /* Handlers pushed during handling (none in practice) would be lost;
    match C by restoring the original stack plus any additions. */
    let added = std::mem::take(&mut ctx.err_handler);
    ctx.err_handler = handlers;
    ctx.err_handler.extend(added);
}
