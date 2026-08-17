//! Port of `src/sundials/sundials_logger.c` +
//! `include/sundials/sundials_logger.h` + `sundials_logger_impl.h` +
//! `include/sundials/priv/sundials_logger_macros.h`.
//!
//! The reference builds use `SUNDIALS_LOGGING_LEVEL = 2` (errors +
//! warnings), so `SUNLogInfo`/`SUNLogDebug`/`SUNLogExtraDebug*` call sites
//! compile away and are omitted at translation time. C's variadic
//! `SUNLogger_QueueMsg(fmt, ...)` maps to a pre-formatted `msg_txt` (Rust
//! callers use `format!` with the `sundials_utils` fmt helpers).

use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

use crate::sundials_errors::{SUN_ERR_ARG_CORRUPT, SUN_ERR_UNREACHABLE, SUN_SUCCESS};
use crate::sundials_hashmap::*;
use crate::sundials_types::*;
use crate::sundials_utils::{sunIsNullOrEmpty, SUNFile};

/// Compile-time logging level of the reference build
/// (CMake default: `SUNDIALS_LOGGING_LEVEL 2`).
pub const SUNDIALS_LOGGING_LEVEL: i32 = 2;

pub const SUNDIALS_LOGGING_ERROR: i32 = 1;
pub const SUNDIALS_LOGGING_WARNING: i32 = 2;
pub const SUNDIALS_LOGGING_INFO: i32 = 3;
pub const SUNDIALS_LOGGING_DEBUG: i32 = 4;

pub type SUNLogLevel = i32;
pub const SUN_LOGLEVEL_ALL: SUNLogLevel = -1;
pub const SUN_LOGLEVEL_NONE: SUNLogLevel = 0;
pub const SUN_LOGLEVEL_ERROR: SUNLogLevel = 1;
pub const SUN_LOGLEVEL_WARNING: SUNLogLevel = 2;
pub const SUN_LOGLEVEL_INFO: SUNLogLevel = 3;
pub const SUN_LOGLEVEL_DEBUG: SUNLogLevel = 4;

pub type SUNLoggerQueueMsgFn = fn(
    logger: &SUNLogger,
    lvl: SUNLogLevel,
    prefix: &str,
    rank: i32,
    scope: &str,
    label: &str,
    payload: &str,
) -> SUNErrCode;

pub type SUNLoggerFlushMsgFn = fn(logger: &SUNLogger, lvl: SUNLogLevel) -> SUNErrCode;

pub struct SUNLogger_ {
    /* MPI information */
    pub comm: SUNComm,
    pub output_rank: i32,

    /* Output files */
    pub debug_fp: SUNFile,
    pub warning_fp: SUNFile,
    pub info_fp: SUNFile,
    pub error_fp: SUNFile,

    /* Hashmap used to store filename, FILE* pairs */
    pub filenames: Option<SUNHashMap<SUNFile>>,

    /* Content for custom implementations */
    pub content: Option<Box<dyn Any>>,

    /* Overridable operations */
    pub queue_msg: Option<SUNLoggerQueueMsgFn>,
    pub flush_msg: Option<SUNLoggerFlushMsgFn>,
}

pub type SUNLogger = Rc<RefCell<SUNLogger_>>;

/// C `sunCreateLogMessage` output format:
/// `[<prefix>][rank <rank>][<scope>][<label>] <payload>\n`.
fn sunCreateLogMessage(prefix: &str, rank: i32, scope: &str, label: &str, payload: &str) -> String {
    format!("[{prefix}][rank {rank}][{scope}][{label}] {payload}\n")
}

/// default number of files that we allocate space for
const SUN_DEFAULT_LOGFILE_HANDLES_: i64 = 8;

fn sunOpenLogFile(fname: &str, mode: &str) -> SUNFile {
    if fname == "stdout" {
        SUNFile::Stdout
    } else if fname == "stderr" {
        SUNFile::Stderr
    } else {
        SUNFile::fopen(fname, mode)
    }
}

fn sunLoggerGetFilePointer(logger: &SUNLogger_, lvl: SUNLogLevel) -> Result<SUNFile, SUNErrCode> {
    match lvl {
        SUN_LOGLEVEL_DEBUG => Ok(logger.debug_fp.clone()),
        SUN_LOGLEVEL_WARNING => Ok(logger.warning_fp.clone()),
        SUN_LOGLEVEL_INFO => Ok(logger.info_fp.clone()),
        SUN_LOGLEVEL_ERROR => Ok(logger.error_fp.clone()),
        _ => Err(SUN_ERR_UNREACHABLE),
    }
}

fn sunLoggerIsOutputRank(_logger: &SUNLogger_, rank_ref: Option<&mut i32>) -> bool {
    if let Some(r) = rank_ref {
        *r = 0;
    }
    true
}

fn sunQueueLogMessage(
    logger: &SUNLogger,
    lvl: SUNLogLevel,
    prefix: &str,
    rank: i32,
    scope: &str,
    label: &str,
    payload: &str,
) -> SUNErrCode {
    let fp = sunLoggerGetFilePointer(&logger.borrow(), lvl);
    match fp {
        Ok(fp) => {
            if !fp.is_null() {
                let log_msg = sunCreateLogMessage(prefix, rank, scope, label, payload);
                fp.write_str(&log_msg);
            }
            SUN_SUCCESS
        }
        Err(code) => code,
    }
}

fn sunFlushLogMessage(logger: &SUNLogger, lvl: SUNLogLevel) -> SUNErrCode {
    let l = logger.borrow();
    if lvl == SUN_LOGLEVEL_ALL {
        if !l.debug_fp.is_null() {
            l.debug_fp.fflush();
        }
        if !l.warning_fp.is_null() {
            l.warning_fp.fflush();
        }
        if !l.info_fp.is_null() {
            l.info_fp.fflush();
        }
        if !l.error_fp.is_null() {
            l.error_fp.fflush();
        }
        SUN_SUCCESS
    } else {
        match sunLoggerGetFilePointer(&l, lvl) {
            Ok(fp) => {
                if !fp.is_null() {
                    fp.fflush();
                }
                SUN_SUCCESS
            }
            Err(code) => code,
        }
    }
}

/// C `sunLoggerSetFilename`: resolves through the filename→FILE* hashmap so
/// the same file used at several levels is opened once.
fn sunLoggerSetFilename(
    logger: &mut SUNLogger_,
    filename: Option<&str>,
    which: fn(&mut SUNLogger_) -> &mut SUNFile,
) -> SUNErrCode {
    if !sunLoggerIsOutputRank(logger, None) {
        return SUN_SUCCESS;
    }

    /* An empty or NULL filename disables output for this stream. */
    if sunIsNullOrEmpty(filename) {
        *which(logger) = SUNFile::Null;
        return SUN_SUCCESS;
    }
    let filename = filename.expect("checked non-empty above");

    let map = logger.filenames.as_mut().expect("filenames map exists at logging level > 0");
    let (err, existing) = SUNHashMap_GetValue(map, filename);
    if err == SUNHASHMAP_ERROR {
        return crate::sundials_errors::SUN_ERR_FILE_OPEN;
    } else if err == SUNHASHMAP_KEYNOTFOUND {
        let fp = sunOpenLogFile(filename, "w+");
        if fp.is_null() {
            return crate::sundials_errors::SUN_ERR_FILE_OPEN;
        }
        let ierr = SUNHashMap_Insert(map, filename, fp.clone());
        if ierr != 0 {
            return crate::sundials_errors::SUN_ERR_FILE_OPEN;
        }
        *which(logger) = fp;
    } else {
        let fp = existing.expect("hit implies value").clone();
        *which(logger) = fp;
    }

    SUN_SUCCESS
}

pub fn SUNLogger_Create(comm: SUNComm, output_rank: i32, logger_ptr: &mut Option<SUNLogger>) -> SUNErrCode {
    *logger_ptr = None;

    if comm != SUN_COMM_NULL {
        return SUN_ERR_ARG_CORRUPT;
    }

    let logger = SUNLogger_ {
        comm: SUN_COMM_NULL,
        output_rank,
        content: None,
        /* use default routines */
        queue_msg: Some(sunQueueLogMessage as SUNLoggerQueueMsgFn),
        flush_msg: Some(sunFlushLogMessage as SUNLoggerFlushMsgFn),
        /* set the output file handles */
        filenames: match SUNHashMap_New(SUN_DEFAULT_LOGFILE_HANDLES_) {
            Ok(map) => Some(map),
            Err(_) => None,
        },
        error_fp: SUNFile::Stderr,
        warning_fp: SUNFile::Stdout,
        debug_fp: SUNFile::Stdout,
        info_fp: SUNFile::Stdout,
    };

    *logger_ptr = Some(Rc::new(RefCell::new(logger)));
    SUN_SUCCESS
}

pub fn SUNLogger_CreateFromEnv(comm: SUNComm, logger_out: &mut Option<SUNLogger>) -> SUNErrCode {
    let mut err = SUN_SUCCESS;
    let mut logger: Option<SUNLogger> = None;

    let output_rank_env = std::env::var("SUNLOGGER_OUTPUT_RANK").ok();
    let output_rank: i32 = output_rank_env
        .as_deref()
        .map(|s| crate::sundials_utils::atoi(s))
        .unwrap_or(0);
    let error_fname_env = std::env::var("SUNLOGGER_ERROR_FILENAME").ok();
    let warning_fname_env = std::env::var("SUNLOGGER_WARNING_FILENAME").ok();
    let info_fname_env = std::env::var("SUNLOGGER_INFO_FILENAME").ok();
    let debug_fname_env = std::env::var("SUNLOGGER_DEBUG_FILENAME").ok();

    if SUNLogger_Create(comm, output_rank, &mut logger) != SUN_SUCCESS {
        return crate::sundials_errors::SUN_ERR_CORRUPT;
    }
    let lg = logger.expect("created above");

    loop {
        /* Only override the default logging if the env var is defined */
        if let Some(f) = error_fname_env.as_deref() {
            err = SUNLogger_SetErrorFilename(&lg, Some(f));
            if err != SUN_SUCCESS {
                break;
            }
        }
        if let Some(f) = warning_fname_env.as_deref() {
            err = SUNLogger_SetWarningFilename(&lg, Some(f));
            if err != SUN_SUCCESS {
                break;
            }
        }
        if let Some(f) = debug_fname_env.as_deref() {
            err = SUNLogger_SetDebugFilename(&lg, Some(f));
            if err != SUN_SUCCESS {
                break;
            }
        }
        if let Some(f) = info_fname_env.as_deref() {
            err = SUNLogger_SetInfoFilename(&lg, Some(f));
            if err != SUN_SUCCESS {
                break;
            }
        }
        break;
    }

    if err != SUN_SUCCESS {
        /* SUNLogger_Destroy */
        *logger_out = None;
    } else {
        *logger_out = Some(lg);
    }

    err
}

pub fn SUNLogger_SetErrorFilename(logger: &SUNLogger, error_filename: Option<&str>) -> SUNErrCode {
    if SUNDIALS_LOGGING_LEVEL >= SUNDIALS_LOGGING_ERROR {
        sunLoggerSetFilename(&mut logger.borrow_mut(), error_filename, |l| &mut l.error_fp)
    } else {
        SUN_SUCCESS
    }
}

pub fn SUNLogger_SetErrorFile(logger: &SUNLogger, error_fp: SUNFile) -> SUNErrCode {
    if SUNDIALS_LOGGING_LEVEL >= SUNDIALS_LOGGING_ERROR {
        logger.borrow_mut().error_fp = error_fp;
    }
    SUN_SUCCESS
}

pub fn SUNLogger_GetErrorFile(logger: &SUNLogger, error_fp: &mut SUNFile) -> SUNErrCode {
    *error_fp = logger.borrow().error_fp.clone();
    SUN_SUCCESS
}

pub fn SUNLogger_SetWarningFilename(
    logger: &SUNLogger,
    warning_filename: Option<&str>,
) -> SUNErrCode {
    if SUNDIALS_LOGGING_LEVEL >= SUNDIALS_LOGGING_WARNING {
        sunLoggerSetFilename(&mut logger.borrow_mut(), warning_filename, |l| {
            &mut l.warning_fp
        })
    } else {
        SUN_SUCCESS
    }
}

pub fn SUNLogger_SetWarningFile(logger: &SUNLogger, warning_fp: SUNFile) -> SUNErrCode {
    if SUNDIALS_LOGGING_LEVEL >= SUNDIALS_LOGGING_WARNING {
        logger.borrow_mut().warning_fp = warning_fp;
    }
    SUN_SUCCESS
}

pub fn SUNLogger_GetWarningFile(logger: &SUNLogger, warning_fp: &mut SUNFile) -> SUNErrCode {
    *warning_fp = logger.borrow().warning_fp.clone();
    SUN_SUCCESS
}

pub fn SUNLogger_SetInfoFilename(logger: &SUNLogger, info_filename: Option<&str>) -> SUNErrCode {
    if SUNDIALS_LOGGING_LEVEL >= SUNDIALS_LOGGING_INFO {
        sunLoggerSetFilename(&mut logger.borrow_mut(), info_filename, |l| &mut l.info_fp)
    } else {
        let _ = info_filename;
        SUN_SUCCESS
    }
}

pub fn SUNLogger_SetInfoFile(logger: &SUNLogger, info_fp: SUNFile) -> SUNErrCode {
    if SUNDIALS_LOGGING_LEVEL >= SUNDIALS_LOGGING_INFO {
        logger.borrow_mut().info_fp = info_fp;
    }
    SUN_SUCCESS
}

pub fn SUNLogger_GetInfoFile(logger: &SUNLogger, info_fp: &mut SUNFile) -> SUNErrCode {
    *info_fp = logger.borrow().info_fp.clone();
    SUN_SUCCESS
}

pub fn SUNLogger_SetDebugFilename(logger: &SUNLogger, debug_filename: Option<&str>) -> SUNErrCode {
    if SUNDIALS_LOGGING_LEVEL >= SUNDIALS_LOGGING_DEBUG {
        sunLoggerSetFilename(&mut logger.borrow_mut(), debug_filename, |l| &mut l.debug_fp)
    } else {
        let _ = debug_filename;
        SUN_SUCCESS
    }
}

pub fn SUNLogger_SetDebugFile(logger: &SUNLogger, debug_fp: SUNFile) -> SUNErrCode {
    if SUNDIALS_LOGGING_LEVEL >= SUNDIALS_LOGGING_DEBUG {
        logger.borrow_mut().debug_fp = debug_fp;
    }
    SUN_SUCCESS
}

pub fn SUNLogger_GetDebugFile(logger: &SUNLogger, debug_fp: &mut SUNFile) -> SUNErrCode {
    *debug_fp = logger.borrow().debug_fp.clone();
    SUN_SUCCESS
}

pub fn SUNLogger_SetQueueAndFlushMsgFns(
    logger: &SUNLogger,
    queue_msg: Option<SUNLoggerQueueMsgFn>,
    flush_msg: Option<SUNLoggerFlushMsgFn>,
    lptr: Option<Box<dyn Any>>,
) -> SUNErrCode {
    let mut l = logger.borrow_mut();
    if let Some(q) = queue_msg {
        l.queue_msg = Some(q);
        l.flush_msg = flush_msg;
        l.content = lptr;
    } else {
        l.queue_msg = Some(sunQueueLogMessage as SUNLoggerQueueMsgFn);
        l.flush_msg = Some(sunFlushLogMessage as SUNLoggerFlushMsgFn);
        l.content = None;
    }
    SUN_SUCCESS
}

/// C `SUNLogger_QueueMsg` — `msg_txt` is the already-formatted payload.
pub fn SUNLogger_QueueMsg(
    logger: &SUNLogger,
    lvl: SUNLogLevel,
    scope: &str,
    label: &str,
    msg_txt: &str,
) -> SUNErrCode {
    let mut retval = SUN_SUCCESS;

    let (queue_fn, rank) = {
        let l = logger.borrow();
        if l.queue_msg.is_none() {
            return retval;
        }
        let mut rank = 0;
        if !sunLoggerIsOutputRank(&l, Some(&mut rank)) {
            return retval;
        }
        match sunLoggerGetFilePointer(&l, lvl) {
            Ok(fp) => {
                if fp.is_null() {
                    return retval;
                }
            }
            Err(code) => return code,
        }
        (l.queue_msg.expect("checked above"), rank)
    };

    let prefix = if lvl == SUN_LOGLEVEL_DEBUG {
        "DEBUG"
    } else if lvl == SUN_LOGLEVEL_WARNING {
        "WARNING"
    } else if lvl == SUN_LOGLEVEL_INFO {
        "INFO"
    } else if lvl == SUN_LOGLEVEL_ERROR {
        "ERROR"
    } else {
        ""
    };

    retval = queue_fn(logger, lvl, prefix, rank, scope, label, msg_txt);
    retval
}

pub fn SUNLogger_Flush(logger: &SUNLogger, lvl: SUNLogLevel) -> SUNErrCode {
    let flush_fn = {
        let l = logger.borrow();
        if !sunLoggerIsOutputRank(&l, None) {
            return SUN_SUCCESS;
        }
        l.flush_msg
    };
    match flush_fn {
        Some(f) => f(logger, lvl),
        None => SUN_SUCCESS,
    }
}

pub fn SUNLogger_GetOutputRank(logger: &SUNLogger, output_rank: &mut i32) -> SUNErrCode {
    *output_rank = logger.borrow().output_rank;
    SUN_SUCCESS
}

pub fn SUNLogger_Destroy(logger_ptr: &mut Option<SUNLogger>) -> SUNErrCode {
    *logger_ptr = None;
    SUN_SUCCESS
}
