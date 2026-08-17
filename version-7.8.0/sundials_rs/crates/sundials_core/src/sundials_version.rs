//! Port of `src/sundials/sundials_version.c` +
//! `include/sundials/sundials_version.h`.

use crate::sundials_errors::SUN_SUCCESS;
use crate::sundials_types::*;

pub const SUNDIALS_VERSION: &str = "7.8.0";
pub const SUNDIALS_VERSION_MAJOR: i32 = 7;
pub const SUNDIALS_VERSION_MINOR: i32 = 8;
pub const SUNDIALS_VERSION_PATCH: i32 = 0;
pub const SUNDIALS_VERSION_LABEL: &str = "";
/// Release tarball builds carry no git metadata.
pub const SUNDIALS_GIT_VERSION: &str = "";

/// C `SUNDIALSGetVersion(char* version, int len)` — the `len` bound guards
/// a C buffer; the Rust port writes into a `String` out-param with the same
/// length check semantics.
pub fn SUNDIALSGetVersion(version: &mut String, len: i32) -> SUNErrCode {
    if SUNDIALS_VERSION.len() >= len as usize {
        return crate::sundials_errors::SUN_ERR_ARG_OUTOFRANGE;
    }
    *version = SUNDIALS_VERSION.to_string();
    SUN_SUCCESS
}

/// C `SUNDIALSGetVersionNumber`.
pub fn SUNDIALSGetVersionNumber(
    major: &mut i32,
    minor: &mut i32,
    patch: &mut i32,
    label: &mut String,
    len: i32,
) -> SUNErrCode {
    if SUNDIALS_VERSION_LABEL.len() >= len as usize {
        return crate::sundials_errors::SUN_ERR_ARG_OUTOFRANGE;
    }
    *major = SUNDIALS_VERSION_MAJOR;
    *minor = SUNDIALS_VERSION_MINOR;
    *patch = SUNDIALS_VERSION_PATCH;
    *label = SUNDIALS_VERSION_LABEL.to_string();
    SUN_SUCCESS
}
