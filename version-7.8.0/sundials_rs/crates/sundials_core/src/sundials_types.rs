//! Port of `include/sundials/sundials_types.h`.
//!
//! `SUN_RCONST(x)` is just a literal in Rust. The forward-declared handle
//! types (`SUNContext`, `SUNErrHandler`, `SUNProfiler`, `SUNLogger`) live
//! in their own modules. `SUN_FORMAT_E/G/SG` printf formats map to
//! `sundials_utils::{sun_format_e, sun_format_g, sun_format_sg}`.

pub type sunrealtype = f64;
pub type sunindextype = i64;
pub type suncountertype = i64;
pub type sunbooleantype = bool;
pub type SUNErrCode = i32;
pub type SUNComm = i32;

pub const SUNFALSE: sunbooleantype = false;
pub const SUNTRUE: sunbooleantype = true;

pub const SUN_BIG_REAL: sunrealtype = f64::MAX;
pub const SUN_SMALL_REAL: sunrealtype = f64::MIN_POSITIVE;
pub const SUN_UNIT_ROUNDOFF: sunrealtype = f64::EPSILON;

pub const SUN_COMM_NULL: SUNComm = 0;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SUNOutputFormat {
    SUN_OUTPUTFORMAT_TABLE,
    SUN_OUTPUTFORMAT_CSV,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SUNDataIOMode {
    SUNDATAIOMODE_INMEM,
}
