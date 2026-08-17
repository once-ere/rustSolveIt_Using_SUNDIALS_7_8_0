//! Port of `src/sundials/sundials_domeigestimator.c` +
//! `include/sundials/sundials_domeigestimator.h` (generic
//! SUNDomEigEstimator — the dominant-eigenvalue estimator base class used
//! by the ARKODE LSRK stepper).
//!
//! `A_data` / `rhs_data` (C `void*`) are `Option<Box<dyn Any>>` tokens
//! handed to the implementation, which stores them in its content and
//! `Option::take`s them around callback invocations (restoring on every
//! return path) — the locked `SUNLinSolSetATimes` pattern. C `FILE*` maps
//! to `crate::sundials_utils::SUNFile`.
//!
//! Handle arguments are `&SUNDomEigEstimator` (non-null by construction),
//! so C's `DEE == NULL` guard in `SetOptions` vanishes. `Destroy` keeps
//! C's `SUNDomEigEstimator*` shape as `&mut Option<SUNDomEigEstimator>`
//! so the NULL-out of the caller's handle on return is preserved
//! (examples do `SUNDomEigEstimator_Destroy(&DEE)`).
//!
//! Build config: profiling OFF, so the `SUNDIALS_MARK_FUNCTION_BEGIN/END`
//! pairs and `getSUNProfiler` are omitted; `SUNDIALS_ENABLE_PYTHON` OFF,
//! so the C `void* python` field is dropped exactly as in
//! `sundials_linearsolver`.

use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

use crate::sundials_context::SUNContext;
use crate::sundials_errors::{SUN_ERR_ARG_INCOMPATIBLE, SUN_ERR_NOT_IMPLEMENTED, SUN_SUCCESS};
use crate::sundials_linearsolver::SUNATimesFn;
use crate::sundials_math::SUNStrToReal;
use crate::sundials_nvector::N_Vector;
use crate::sundials_types::*;
use crate::sundials_utils::{atoi, atol, SUNFile};

/// C `SUNRhsFn` (declared in `sundials_domeigestimator.h`): the ODE
/// right-hand side handed to an estimator that linearizes it internally.
/// Argument-for-argument identical to `cvode_rs::cvode_impl::CVRhsFn`, so
/// an ARKODE/CVODE RHS function can be passed straight through.
pub type SUNRhsFn =
    fn(t: sunrealtype, y: &N_Vector, ydot: &N_Vector, user_data: &mut Option<Box<dyn Any>>) -> i32;

/* -----------------------------------------------------------------
 * Generic definition of SUNDomEigEstimator (DEE)
 * ----------------------------------------------------------------- */

/* Structure containing function pointers to estimator operations */
#[derive(Default, Clone)]
pub struct SUNDomEigEstimator_Ops_ {
    pub setatimes:
        Option<fn(&SUNDomEigEstimator, Option<Box<dyn Any>>, Option<SUNATimesFn>) -> SUNErrCode>,
    pub setrhs:
        Option<fn(&SUNDomEigEstimator, Option<Box<dyn Any>>, Option<SUNRhsFn>) -> SUNErrCode>,
    pub setrhslinearizationpoint:
        Option<fn(&SUNDomEigEstimator, sunrealtype, &N_Vector) -> SUNErrCode>,
    pub setoptions:
        Option<fn(&SUNDomEigEstimator, Option<&str>, Option<&str>, &[String]) -> SUNErrCode>,
    pub setmaxiters: Option<fn(&SUNDomEigEstimator, i64) -> SUNErrCode>,
    pub setnumpreprocessiters: Option<fn(&SUNDomEigEstimator, i32) -> SUNErrCode>,
    pub setreltol: Option<fn(&SUNDomEigEstimator, sunrealtype) -> SUNErrCode>,
    pub setinitialguess: Option<fn(&SUNDomEigEstimator, &N_Vector) -> SUNErrCode>,
    pub initialize: Option<fn(&SUNDomEigEstimator) -> SUNErrCode>,
    pub estimate: Option<fn(&SUNDomEigEstimator, &mut sunrealtype, &mut sunrealtype) -> SUNErrCode>,
    pub getres: Option<fn(&SUNDomEigEstimator, &mut sunrealtype) -> SUNErrCode>,
    pub getnumiters: Option<fn(&SUNDomEigEstimator, &mut i64) -> SUNErrCode>,
    pub getnumrhsevals: Option<fn(&SUNDomEigEstimator, &mut i64) -> SUNErrCode>,
    pub getnumatimescalls: Option<fn(&SUNDomEigEstimator, &mut i64) -> SUNErrCode>,
    pub write: Option<fn(&SUNDomEigEstimator, &SUNFile) -> SUNErrCode>,
    pub destroy: Option<fn(&mut Option<SUNDomEigEstimator>) -> SUNErrCode>,
}

pub type SUNDomEigEstimator_Ops = SUNDomEigEstimator_Ops_;

/* An estimator is a structure with an implementation-dependent
'content' field, and a pointer to a structure of estimator
operations corresponding to that implementation. */
pub struct SUNDomEigEstimator_ {
    pub content: RefCell<Box<dyn Any>>,
    pub ops: RefCell<SUNDomEigEstimator_Ops_>,
    pub sunctx: RefCell<SUNContext>,
}

pub type SUNDomEigEstimator = Rc<SUNDomEigEstimator_>;

/* -----------------------------------------------------------------
 * Create a new empty SUNDomEigEstimator object
 * ----------------------------------------------------------------- */

pub fn SUNDomEigEstimator_NewEmpty(sunctx: &SUNContext) -> Option<SUNDomEigEstimator> {
    Some(Rc::new(SUNDomEigEstimator_ {
        content: RefCell::new(Box::new(())),
        ops: RefCell::new(SUNDomEigEstimator_Ops_::default()),
        sunctx: RefCell::new(sunctx.clone()),
    }))
}

/* -----------------------------------------------------------------
 * Free a generic SUNDomEigEstimator (assumes content is already empty)
 * ----------------------------------------------------------------- */

pub fn SUNDomEigEstimator_FreeEmpty(DEE: Option<SUNDomEigEstimator>) {
    /* C frees the ops struct and then the object itself; dropping the
    handle releases both (and the content box). NULL is tolerated. */
    drop(DEE);
}

/* -----------------------------------------------------------------
 * internal utility routines
 * ----------------------------------------------------------------- */

/// C `sunDEESetFromCommandLine`: processes `<Did>.max_iters <long>`,
/// `<Did>.num_preprocess_iters <int>` and `<Did>.rel_tol <real>` tokens
/// (default prefix `sundomeigestimator.`).
fn sunDEESetFromCommandLine(
    DEE: &SUNDomEigEstimator,
    Did: Option<&str>,
    argv: &[String],
) -> SUNErrCode {
    /* Prefix for options to set */
    let default_id = "sundomeigestimator";
    let id = match Did {
        Some(s) if !s.is_empty() => s,
        _ => default_id,
    };
    let prefix = format!("{id}.");

    let mut idx = 1;
    while idx < argv.len() {
        /* skip command-line arguments that do not begin with correct prefix */
        if !argv[idx].starts_with(&prefix) {
            idx += 1;
            continue;
        }
        let key = &argv[idx][prefix.len()..];

        /* control over SetMaxIters function */
        if key == "max_iters" {
            idx += 1;
            let large = atol(&argv[idx]);
            let retval = SUNDomEigEstimator_SetMaxIters(DEE, large);
            if retval != SUN_SUCCESS {
                return retval;
            }
            idx += 1;
            continue;
        }

        /* control over SetNumPreprocessIters function */
        if key == "num_preprocess_iters" {
            idx += 1;
            let iarg = atoi(&argv[idx]);
            let retval = SUNDomEigEstimator_SetNumPreprocessIters(DEE, iarg);
            if retval != SUN_SUCCESS {
                return retval;
            }
            idx += 1;
            continue;
        }

        /* control over SetRelTol function */
        if key == "rel_tol" {
            idx += 1;
            let rarg = SUNStrToReal(&argv[idx]);
            let retval = SUNDomEigEstimator_SetRelTol(DEE, rarg);
            if retval != SUN_SUCCESS {
                return retval;
            }
            idx += 1;
            continue;
        }

        idx += 1;
    }

    SUN_SUCCESS
}

/* -----------------------------------------------------------------
 * Functions in the 'ops' structure
 * -----------------------------------------------------------------*/

pub fn SUNDomEigEstimator_SetATimes(
    DEE: &SUNDomEigEstimator,
    A_data: Option<Box<dyn Any>>,
    ATimes: Option<SUNATimesFn>,
) -> SUNErrCode {
    let f = DEE.ops.borrow().setatimes;
    match f {
        Some(f) => f(DEE, A_data, ATimes),
        None => SUN_SUCCESS,
    }
}

pub fn SUNDomEigEstimator_SetRhs(
    DEE: &SUNDomEigEstimator,
    rhs_data: Option<Box<dyn Any>>,
    RHSfn: Option<SUNRhsFn>,
) -> SUNErrCode {
    let f = DEE.ops.borrow().setrhs;
    match f {
        Some(f) => f(DEE, rhs_data, RHSfn),
        None => SUN_SUCCESS,
    }
}

pub fn SUNDomEigEstimator_SetRhsLinearizationPoint(
    DEE: &SUNDomEigEstimator,
    t: sunrealtype,
    v: &N_Vector,
) -> SUNErrCode {
    let f = DEE.ops.borrow().setrhslinearizationpoint;
    match f {
        Some(f) => f(DEE, t, v),
        None => SUN_SUCCESS,
    }
}

pub fn SUNDomEigEstimator_SetOptions(
    DEE: &SUNDomEigEstimator,
    Did: Option<&str>,
    file_name: Option<&str>,
    argv: &[String],
) -> SUNErrCode {
    /* File-based option control is currently unimplemented */
    if !(file_name.is_none() || file_name == Some("")) {
        return SUN_ERR_ARG_INCOMPATIBLE;
    }

    /* First, process all base-class options */
    if !argv.is_empty() {
        let ier = sunDEESetFromCommandLine(DEE, Did, argv);
        if ier != SUN_SUCCESS {
            return ier;
        }
    }

    /* Second, ask the implementation to process any remaining options */
    let f = DEE.ops.borrow().setoptions;
    match f {
        Some(f) => f(DEE, Did, file_name, argv),
        None => SUN_SUCCESS,
    }
}

pub fn SUNDomEigEstimator_SetMaxIters(DEE: &SUNDomEigEstimator, max_iters: i64) -> SUNErrCode {
    let f = DEE.ops.borrow().setmaxiters;
    match f {
        Some(f) => f(DEE, max_iters),
        None => SUN_SUCCESS,
    }
}

pub fn SUNDomEigEstimator_SetNumPreprocessIters(
    DEE: &SUNDomEigEstimator,
    num_iters: i32,
) -> SUNErrCode {
    let f = DEE.ops.borrow().setnumpreprocessiters;
    match f {
        Some(f) => f(DEE, num_iters),
        None => SUN_SUCCESS,
    }
}

pub fn SUNDomEigEstimator_SetRelTol(DEE: &SUNDomEigEstimator, rel_tol: sunrealtype) -> SUNErrCode {
    let f = DEE.ops.borrow().setreltol;
    match f {
        Some(f) => f(DEE, rel_tol),
        None => SUN_SUCCESS,
    }
}

pub fn SUNDomEigEstimator_SetInitialGuess(DEE: &SUNDomEigEstimator, q: &N_Vector) -> SUNErrCode {
    let f = DEE.ops.borrow().setinitialguess;
    match f {
        Some(f) => f(DEE, q),
        None => SUN_SUCCESS,
    }
}

pub fn SUNDomEigEstimator_Initialize(DEE: &SUNDomEigEstimator) -> SUNErrCode {
    let f = DEE.ops.borrow().initialize;
    match f {
        Some(f) => f(DEE),
        None => SUN_SUCCESS,
    }
}

pub fn SUNDomEigEstimator_Estimate(
    DEE: &SUNDomEigEstimator,
    lambdaR: &mut sunrealtype,
    lambdaI: &mut sunrealtype,
) -> SUNErrCode {
    let f = DEE.ops.borrow().estimate;
    match f {
        Some(f) => f(DEE, lambdaR, lambdaI),
        None => SUN_ERR_NOT_IMPLEMENTED,
    }
}

pub fn SUNDomEigEstimator_GetRes(DEE: &SUNDomEigEstimator, res: &mut sunrealtype) -> SUNErrCode {
    let f = DEE.ops.borrow().getres;
    match f {
        Some(f) => f(DEE, res),
        None => {
            *res = 0.0;
            SUN_SUCCESS
        }
    }
}

pub fn SUNDomEigEstimator_GetNumIters(DEE: &SUNDomEigEstimator, num_iters: &mut i64) -> SUNErrCode {
    let f = DEE.ops.borrow().getnumiters;
    match f {
        Some(f) => f(DEE, num_iters),
        None => {
            *num_iters = 0;
            SUN_SUCCESS
        }
    }
}

pub fn SUNDomEigEstimator_GetNumRhsEvals(
    DEE: &SUNDomEigEstimator,
    num_rhs_evals: &mut i64,
) -> SUNErrCode {
    let f = DEE.ops.borrow().getnumrhsevals;
    match f {
        Some(f) => f(DEE, num_rhs_evals),
        None => {
            *num_rhs_evals = 0;
            SUN_SUCCESS
        }
    }
}

pub fn SUNDomEigEstimator_GetNumATimesCalls(
    DEE: &SUNDomEigEstimator,
    num_ATimes: &mut i64,
) -> SUNErrCode {
    let f = DEE.ops.borrow().getnumatimescalls;
    match f {
        Some(f) => f(DEE, num_ATimes),
        None => {
            *num_ATimes = 0;
            SUN_SUCCESS
        }
    }
}

pub fn SUNDomEigEstimator_Write(DEE: &SUNDomEigEstimator, outfile: &SUNFile) -> SUNErrCode {
    let f = DEE.ops.borrow().write;
    match f {
        Some(f) => f(DEE, outfile),
        None => SUN_SUCCESS,
    }
}

pub fn SUNDomEigEstimator_Destroy(DEEptr: &mut Option<SUNDomEigEstimator>) -> SUNErrCode {
    let ier = SUN_SUCCESS;
    /* C also guards `DEEptr == NULL`; a `&mut` reference is never null. */
    let f = match DEEptr.as_ref() {
        None => return ier,
        Some(DEE) => DEE.ops.borrow().destroy,
    };
    match f {
        Some(f) => f(DEEptr),
        None => {
            SUNDomEigEstimator_FreeEmpty(DEEptr.take());
            ier
        }
    }
}
