//! Port of `src/sundials/sundials_linearsolver.c` +
//! `include/sundials/sundials_linearsolver.h` (generic SUNLinearSolver).
//!
//! `A_data`/`P_data` (C `void*`) are `Box<dyn Any>` tokens stored in the
//! implementation content; solve routines `Option::take` them around
//! callback invocations (restoring on every return path) so a callback
//! gets exclusive access without re-borrowing the solver content.

use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

use crate::sundials_context::SUNContext;
use crate::sundials_errors::{SUN_ERR_ARG_CORRUPT, SUN_ERR_ARG_INCOMPATIBLE, SUN_SUCCESS};
use crate::sundials_matrix::SUNMatrix;
use crate::sundials_nvector::N_Vector;
use crate::sundials_types::*;

/* ----- types from sundials_iterative.h (callback signatures) ----- */

pub type SUNATimesFn = fn(A_data: &mut Option<Box<dyn Any>>, v: &N_Vector, z: &N_Vector) -> i32;
pub type SUNPSetupFn = fn(P_data: &mut Option<Box<dyn Any>>) -> i32;
pub type SUNPSolveFn = fn(
    P_data: &mut Option<Box<dyn Any>>,
    r: &N_Vector,
    z: &N_Vector,
    tol: sunrealtype,
    lr: i32,
) -> i32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SUNLinearSolver_Type {
    SUNLINEARSOLVER_DIRECT,
    SUNLINEARSOLVER_ITERATIVE,
    SUNLINEARSOLVER_MATRIX_ITERATIVE,
    SUNLINEARSOLVER_MATRIX_EMBEDDED,
}
pub use SUNLinearSolver_Type::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SUNLinearSolver_ID {
    SUNLINEARSOLVER_BAND,
    SUNLINEARSOLVER_DENSE,
    SUNLINEARSOLVER_KLU,
    SUNLINEARSOLVER_LAPACKBAND,
    SUNLINEARSOLVER_LAPACKDENSE,
    SUNLINEARSOLVER_PCG,
    SUNLINEARSOLVER_SPBCGS,
    SUNLINEARSOLVER_SPFGMR,
    SUNLINEARSOLVER_SPGMR,
    SUNLINEARSOLVER_SPTFQMR,
    SUNLINEARSOLVER_SUPERLUDIST,
    SUNLINEARSOLVER_SUPERLUMT,
    SUNLINEARSOLVER_CUSOLVERSP_BATCHQR,
    SUNLINEARSOLVER_MAGMADENSE,
    SUNLINEARSOLVER_ONEMKLDENSE,
    SUNLINEARSOLVER_GINKGO,
    SUNLINEARSOLVER_GINKGOBATCH,
    SUNLINEARSOLVER_KOKKOSDENSE,
    SUNLINEARSOLVER_CUSTOM,
}
pub use SUNLinearSolver_ID::*;

/* SUNLinearSolver return values */
pub const SUNLS_ATIMES_NULL: i32 = -804;
pub const SUNLS_ATIMES_FAIL_UNREC: i32 = -805;
pub const SUNLS_PSET_FAIL_UNREC: i32 = -806;
pub const SUNLS_PSOLVE_NULL: i32 = -807;
pub const SUNLS_PSOLVE_FAIL_UNREC: i32 = -808;
pub const SUNLS_GS_FAIL: i32 = -810;
pub const SUNLS_QRSOL_FAIL: i32 = -811;

pub const SUNLS_RECOV_FAILURE: i32 = 800;
pub const SUNLS_RES_REDUCED: i32 = 801;
pub const SUNLS_CONV_FAIL: i32 = 802;
pub const SUNLS_ATIMES_FAIL_REC: i32 = 803;
pub const SUNLS_PSET_FAIL_REC: i32 = 804;
pub const SUNLS_PSOLVE_FAIL_REC: i32 = 805;
pub const SUNLS_PACKAGE_FAIL_REC: i32 = 806;
pub const SUNLS_QRFACT_FAIL: i32 = 807;
pub const SUNLS_LUFACT_FAIL: i32 = 808;

#[derive(Default, Clone)]
pub struct _generic_SUNLinearSolver_Ops {
    pub gettype: Option<fn(&SUNLinearSolver) -> SUNLinearSolver_Type>,
    pub getid: Option<fn(&SUNLinearSolver) -> SUNLinearSolver_ID>,
    pub setatimes:
        Option<fn(&SUNLinearSolver, Option<Box<dyn Any>>, Option<SUNATimesFn>) -> SUNErrCode>,
    pub setpreconditioner: Option<
        fn(
            &SUNLinearSolver,
            Option<Box<dyn Any>>,
            Option<SUNPSetupFn>,
            Option<SUNPSolveFn>,
        ) -> SUNErrCode,
    >,
    pub setscalingvectors:
        Option<fn(&SUNLinearSolver, Option<&N_Vector>, Option<&N_Vector>) -> SUNErrCode>,
    pub setoptions:
        Option<fn(&SUNLinearSolver, Option<&str>, Option<&str>, &[String]) -> SUNErrCode>,
    pub setzeroguess: Option<fn(&SUNLinearSolver, sunbooleantype) -> SUNErrCode>,
    pub initialize: Option<fn(&SUNLinearSolver) -> SUNErrCode>,
    pub setup: Option<fn(&SUNLinearSolver, Option<&SUNMatrix>) -> i32>,
    pub solve:
        Option<fn(&SUNLinearSolver, Option<&SUNMatrix>, &N_Vector, &N_Vector, sunrealtype) -> i32>,
    pub numiters: Option<fn(&SUNLinearSolver) -> i32>,
    pub resnorm: Option<fn(&SUNLinearSolver) -> sunrealtype>,
    pub lastflag: Option<fn(&SUNLinearSolver) -> sunindextype>,
    pub space: Option<fn(&SUNLinearSolver, &mut i64, &mut i64) -> SUNErrCode>,
    pub resid: Option<fn(&SUNLinearSolver) -> Option<N_Vector>>,
    pub free: Option<fn(&SUNLinearSolver) -> SUNErrCode>,
}

pub type SUNLinearSolver_Ops = _generic_SUNLinearSolver_Ops;

pub struct _generic_SUNLinearSolver {
    pub content: RefCell<Box<dyn Any>>,
    pub ops: RefCell<_generic_SUNLinearSolver_Ops>,
    pub sunctx: RefCell<SUNContext>,
}

pub type SUNLinearSolver = Rc<_generic_SUNLinearSolver>;

pub fn SUNLinSolNewEmpty(sunctx: &SUNContext) -> Option<SUNLinearSolver> {
    Some(Rc::new(_generic_SUNLinearSolver {
        content: RefCell::new(Box::new(())),
        ops: RefCell::new(_generic_SUNLinearSolver_Ops::default()),
        sunctx: RefCell::new(sunctx.clone()),
    }))
}

pub fn SUNLinSolFreeEmpty(S: SUNLinearSolver) {
    drop(S);
}

/// C `sunlsSetFromCommandLine`: processes `<LSid>.zero_guess <int>` tokens.
fn sunlsSetFromCommandLine(S: &SUNLinearSolver, LSid: Option<&str>, argv: &[String]) -> SUNErrCode {
    let default_id = "sunlinearsolver";
    let id = match LSid {
        Some(s) if !s.is_empty() => s,
        _ => default_id,
    };
    let prefix = format!("{id}.");

    let mut idx = 1;
    while idx < argv.len() {
        if !argv[idx].starts_with(&prefix) {
            idx += 1;
            continue;
        }
        let key = &argv[idx][prefix.len()..];
        if key == "zero_guess" {
            idx += 1;
            let iarg: i32 = crate::sundials_utils::atoi(&argv[idx]);
            let retval = SUNLinSolSetZeroGuess(S, iarg != 0);
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

pub fn SUNLinSolGetType(S: &SUNLinearSolver) -> SUNLinearSolver_Type {
    let f = S.ops.borrow().gettype.expect("gettype");
    f(S)
}

pub fn SUNLinSolGetID(S: &SUNLinearSolver) -> SUNLinearSolver_ID {
    let f = S.ops.borrow().getid;
    match f {
        Some(f) => f(S),
        None => SUNLINEARSOLVER_CUSTOM,
    }
}

pub fn SUNLinSolSetATimes(
    S: &SUNLinearSolver,
    A_data: Option<Box<dyn Any>>,
    ATimes: Option<SUNATimesFn>,
) -> SUNErrCode {
    let f = S.ops.borrow().setatimes;
    match f {
        Some(f) => f(S, A_data, ATimes),
        None => SUN_SUCCESS,
    }
}

pub fn SUNLinSolSetPreconditioner(
    S: &SUNLinearSolver,
    P_data: Option<Box<dyn Any>>,
    Pset: Option<SUNPSetupFn>,
    Psol: Option<SUNPSolveFn>,
) -> SUNErrCode {
    let f = S.ops.borrow().setpreconditioner;
    match f {
        Some(f) => f(S, P_data, Pset, Psol),
        None => SUN_SUCCESS,
    }
}

pub fn SUNLinSolSetScalingVectors(
    S: &SUNLinearSolver,
    s1: Option<&N_Vector>,
    s2: Option<&N_Vector>,
) -> SUNErrCode {
    let f = S.ops.borrow().setscalingvectors;
    match f {
        Some(f) => f(S, s1, s2),
        None => SUN_SUCCESS,
    }
}

pub fn SUNLinSolSetOptions(
    S: &SUNLinearSolver,
    LSid: Option<&str>,
    file_name: Option<&str>,
    argv: &[String],
) -> SUNErrCode {
    /* File-based option control is currently unimplemented */
    if !(file_name.is_none() || file_name == Some("")) {
        return SUN_ERR_ARG_INCOMPATIBLE;
    }

    /* First, process all base-class options */
    if !argv.is_empty() {
        let ier = sunlsSetFromCommandLine(S, LSid, argv);
        if ier != SUN_SUCCESS {
            return ier;
        }
    }

    /* Second, ask the implementation to process any remaining options */
    let f = S.ops.borrow().setoptions;
    match f {
        Some(f) => f(S, LSid, file_name, argv),
        None => SUN_SUCCESS,
    }
}

pub fn SUNLinSolSetZeroGuess(S: &SUNLinearSolver, onoff: sunbooleantype) -> SUNErrCode {
    let f = S.ops.borrow().setzeroguess;
    match f {
        Some(f) => f(S, onoff),
        None => SUN_SUCCESS,
    }
}

pub fn SUNLinSolInitialize(S: &SUNLinearSolver) -> SUNErrCode {
    let f = S.ops.borrow().initialize;
    match f {
        Some(f) => f(S),
        None => SUN_SUCCESS,
    }
}

pub fn SUNLinSolSetup(S: &SUNLinearSolver, A: Option<&SUNMatrix>) -> i32 {
    let f = S.ops.borrow().setup;
    match f {
        Some(f) => f(S, A),
        None => SUN_SUCCESS,
    }
}

pub fn SUNLinSolSolve(
    S: &SUNLinearSolver,
    A: Option<&SUNMatrix>,
    x: &N_Vector,
    b: &N_Vector,
    tol: sunrealtype,
) -> i32 {
    let f = S.ops.borrow().solve.expect("solve");
    f(S, A, x, b, tol)
}

pub fn SUNLinSolNumIters(S: &SUNLinearSolver) -> i32 {
    let f = S.ops.borrow().numiters;
    match f {
        Some(f) => f(S),
        None => 0,
    }
}

pub fn SUNLinSolResNorm(S: &SUNLinearSolver) -> sunrealtype {
    let f = S.ops.borrow().resnorm;
    match f {
        Some(f) => f(S),
        None => 0.0,
    }
}

pub fn SUNLinSolResid(S: &SUNLinearSolver) -> Option<N_Vector> {
    let f = S.ops.borrow().resid;
    match f {
        Some(f) => f(S),
        None => None,
    }
}

pub fn SUNLinSolLastFlag(S: &SUNLinearSolver) -> sunindextype {
    let f = S.ops.borrow().lastflag;
    match f {
        Some(f) => f(S),
        None => 0,
    }
}

pub fn SUNLinSolSpace(S: &SUNLinearSolver, lenrwLS: &mut i64, leniwLS: &mut i64) -> SUNErrCode {
    let f = S.ops.borrow().space;
    match f {
        Some(f) => f(S, lenrwLS, leniwLS),
        None => {
            *lenrwLS = 0;
            *leniwLS = 0;
            SUN_SUCCESS
        }
    }
}

pub fn SUNLinSolFree(S: Option<SUNLinearSolver>) -> SUNErrCode {
    match S {
        None => SUN_SUCCESS,
        Some(S) => {
            let f = S.ops.borrow().free;
            if let Some(f) = f {
                return f(&S);
            }
            drop(S);
            SUN_SUCCESS
        }
    }
}

const _: SUNErrCode = SUN_ERR_ARG_CORRUPT;
