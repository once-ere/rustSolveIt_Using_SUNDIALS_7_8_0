//! Port of `src/sundials/sundials_futils.c` +
//! `include/sundials/sundials_futils.h`.
//!
//! C `FILE*` maps to `crate::sundials_utils::SUNFile` (`SUNFile::Null` is a
//! NULL `FILE*`), so the C `FILE**` in-out parameter maps to `&mut SUNFile`
//! (not `&mut Option<SUNFile>`, which would encode NULL twice):
//! `SUNFileOpen` reads the incoming `*fp_out` first — a NULL `filename`
//! leaves it unchanged — and `SUNFileClose` resets a closed `File` handle to
//! `SUNFile::Null` (the safe equivalent of C's dangling pointer after
//! `fclose`) while leaving `stdout`/`stderr` untouched, exactly as in C.

use crate::sundials_errors::{SUN_ERR_FILE_OPEN, SUN_SUCCESS};
use crate::sundials_types::SUNErrCode;
use crate::sundials_utils::SUNFile;

/// Create a file pointer with the given file name and mode.
///
/// The special filenames `"stdout"` and `"stderr"` map to the standard
/// streams (`mode` is ignored for them, as in C). Otherwise the file is
/// opened via `fopen(filename, mode)`; a failed open (or a NULL result with
/// a NULL `filename` and NULL incoming `*fp_out`) yields `SUN_ERR_FILE_OPEN`.
pub fn SUNFileOpen(filename: Option<&str>, mode: &str, fp_out: &mut SUNFile) -> SUNErrCode {
    let mut err: SUNErrCode = SUN_SUCCESS;
    let mut fp = fp_out.clone();

    if let Some(filename) = filename {
        if filename == "stdout" {
            fp = SUNFile::Stdout;
        } else if filename == "stderr" {
            fp = SUNFile::Stderr;
        } else {
            fp = SUNFile::fopen(filename, mode);
        }
    }

    if fp.is_null() {
        err = SUN_ERR_FILE_OPEN;
    }

    *fp_out = fp;
    err
}

/// Deprecated in C (`"Use SUNFileOpen"`); kept as a plain wrapper (no
/// `#[deprecated]` attribute, which would break the zero-warning build).
pub fn SUNDIALSFileOpen(filename: Option<&str>, mode: &str, fp_out: &mut SUNFile) -> SUNErrCode {
    SUNFileOpen(filename, mode, fp_out)
}

/// Close a file pointer with the given file name.
///
/// C first checks `if (!fp_ptr)`; a `&mut` reference cannot be NULL, so that
/// guard has no Rust counterpart. `fclose(fp)` maps to dropping the handle
/// (the OS file closes when the last `Rc` clone is dropped); `*fp_ptr` is
/// reset to `SUNFile::Null` in that case, and `stdout`/`stderr`/NULL are
/// left as-is, matching C (which never closes or nulls them).
pub fn SUNFileClose(fp_ptr: &mut SUNFile) -> SUNErrCode {
    if !fp_ptr.is_null()
        && !fp_ptr.ptr_eq(&SUNFile::Stdout)
        && !fp_ptr.ptr_eq(&SUNFile::Stderr)
    {
        /* fclose(fp) */
        *fp_ptr = SUNFile::Null;
    }
    SUN_SUCCESS
}

/// Deprecated in C (`"Use SUNFileClose"`); kept as a plain wrapper (no
/// `#[deprecated]` attribute, which would break the zero-warning build).
pub fn SUNDIALSFileClose(fp_ptr: &mut SUNFile) -> SUNErrCode {
    SUNFileClose(fp_ptr)
}
