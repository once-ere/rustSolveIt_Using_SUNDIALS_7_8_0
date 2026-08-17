//! Port of `src/sundials/sundials_nonlinearsolver.c` +
//! `include/sundials/sundials_nonlinearsolver.h` (generic
//! SUNNonlinearSolver layer).
//!
//! The integrator `void* mem` passed through Setup/Solve and into the
//! integrator-supplied callbacks maps to `&mut Option<Box<dyn Any>>`
//! (a token owned by the caller for the duration of the call).

use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

use crate::sundials_context::SUNContext;
use crate::sundials_errors::{SUN_ERR_ARG_CORRUPT, SUN_ERR_ARG_INCOMPATIBLE, SUN_SUCCESS};
use crate::sundials_nvector::N_Vector;
use crate::sundials_types::*;

/* integrator supplied function types */
pub type SUNNonlinSolSysFn =
    fn(y: &N_Vector, F: &N_Vector, mem: &mut Option<Box<dyn Any>>) -> i32;
pub type SUNNonlinSolLSetupFn = fn(
    jbad: sunbooleantype,
    jcur: &mut sunbooleantype,
    mem: &mut Option<Box<dyn Any>>,
) -> i32;
pub type SUNNonlinSolLSolveFn = fn(b: &N_Vector, mem: &mut Option<Box<dyn Any>>) -> i32;
pub type SUNNonlinSolConvTestFn = fn(
    NLS: &SUNNonlinearSolver,
    y: &N_Vector,
    del: &N_Vector,
    tol: sunrealtype,
    ewt: &N_Vector,
    ctest_data: &mut Option<Box<dyn Any>>,
) -> i32;
pub type SUNNonlinSolNormFn = fn(
    del: &N_Vector,
    w: &N_Vector,
    delnrm: &mut sunrealtype,
    mem: &mut Option<Box<dyn Any>>,
) -> SUNErrCode;
pub type SUNNonlinSolGetUpdateNormFn =
    fn(delnrm: &mut sunrealtype, mem: &mut Option<Box<dyn Any>>) -> SUNErrCode;
pub type SUNNonlinSolGetConvRateFn =
    fn(crate_: &mut sunrealtype, mem: &mut Option<Box<dyn Any>>) -> SUNErrCode;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SUNNonlinearSolver_Type {
    SUNNONLINEARSOLVER_ROOTFIND,
    SUNNONLINEARSOLVER_FIXEDPOINT,
    SUNNONLINEARSOLVER_HYBRID,
}
pub use SUNNonlinearSolver_Type::*;

/* SUNNonlinearSolver return values (recoverable) */
pub const SUN_NLS_CONTINUE: i32 = 901;
pub const SUN_NLS_CONV_RECVR: i32 = 902;
pub const SUN_NLS_SWITCH: i32 = 903;

#[derive(Default, Clone)]
pub struct _generic_SUNNonlinearSolver_Ops {
    pub gettype: Option<fn(&SUNNonlinearSolver) -> SUNNonlinearSolver_Type>,
    pub initialize: Option<fn(&SUNNonlinearSolver) -> SUNErrCode>,
    pub setup:
        Option<fn(&SUNNonlinearSolver, &N_Vector, &mut Option<Box<dyn Any>>) -> i32>,
    pub solve: Option<
        fn(
            &SUNNonlinearSolver,
            &N_Vector,
            &N_Vector,
            &N_Vector,
            sunrealtype,
            sunbooleantype,
            &mut Option<Box<dyn Any>>,
        ) -> i32,
    >,
    pub free: Option<fn(&SUNNonlinearSolver) -> SUNErrCode>,
    pub setsysfn: Option<fn(&SUNNonlinearSolver, Option<SUNNonlinSolSysFn>) -> SUNErrCode>,
    pub setsysfns: Option<
        fn(
            &SUNNonlinearSolver,
            Option<SUNNonlinSolSysFn>,
            Option<SUNNonlinSolSysFn>,
        ) -> SUNErrCode,
    >,
    pub setlsetupfn:
        Option<fn(&SUNNonlinearSolver, Option<SUNNonlinSolLSetupFn>) -> SUNErrCode>,
    pub setlsolvefn:
        Option<fn(&SUNNonlinearSolver, Option<SUNNonlinSolLSolveFn>) -> SUNErrCode>,
    pub setctestfn: Option<
        fn(
            &SUNNonlinearSolver,
            Option<SUNNonlinSolConvTestFn>,
            Option<Box<dyn Any>>,
        ) -> SUNErrCode,
    >,
    pub setnormfn: Option<
        fn(&SUNNonlinearSolver, Option<SUNNonlinSolNormFn>, Option<Box<dyn Any>>) -> SUNErrCode,
    >,
    pub setgetupdatenormfn: Option<
        fn(
            &SUNNonlinearSolver,
            Option<SUNNonlinSolGetUpdateNormFn>,
            Option<Box<dyn Any>>,
        ) -> SUNErrCode,
    >,
    pub setgetconvratefn: Option<
        fn(
            &SUNNonlinearSolver,
            Option<SUNNonlinSolGetConvRateFn>,
            Option<Box<dyn Any>>,
        ) -> SUNErrCode,
    >,
    pub setoptions:
        Option<fn(&SUNNonlinearSolver, Option<&str>, Option<&str>, &[String]) -> SUNErrCode>,
    pub setmaxiters: Option<fn(&SUNNonlinearSolver, i32) -> SUNErrCode>,
    pub getnumiters: Option<fn(&SUNNonlinearSolver, &mut i64) -> SUNErrCode>,
    pub getcuriter: Option<fn(&SUNNonlinearSolver, &mut i32) -> SUNErrCode>,
    pub getnumconvfails: Option<fn(&SUNNonlinearSolver, &mut i64) -> SUNErrCode>,
}

pub type SUNNonlinearSolver_Ops = _generic_SUNNonlinearSolver_Ops;

pub struct _generic_SUNNonlinearSolver {
    pub content: RefCell<Box<dyn Any>>,
    pub ops: RefCell<_generic_SUNNonlinearSolver_Ops>,
    pub sunctx: RefCell<SUNContext>,
}

pub type SUNNonlinearSolver = Rc<_generic_SUNNonlinearSolver>;

pub fn SUNNonlinSolNewEmpty(sunctx: &SUNContext) -> Option<SUNNonlinearSolver> {
    Some(Rc::new(_generic_SUNNonlinearSolver {
        content: RefCell::new(Box::new(())),
        ops: RefCell::new(_generic_SUNNonlinearSolver_Ops::default()),
        sunctx: RefCell::new(sunctx.clone()),
    }))
}

pub fn SUNNonlinSolFreeEmpty(NLS: SUNNonlinearSolver) {
    drop(NLS);
}

pub fn SUNNonlinSolGetType(NLS: &SUNNonlinearSolver) -> SUNNonlinearSolver_Type {
    let f = NLS.ops.borrow().gettype.expect("gettype");
    f(NLS)
}

pub fn SUNNonlinSolInitialize(NLS: &SUNNonlinearSolver) -> SUNErrCode {
    let f = NLS.ops.borrow().initialize;
    match f {
        Some(f) => f(NLS),
        None => SUN_SUCCESS,
    }
}

pub fn SUNNonlinSolSetup(
    NLS: &SUNNonlinearSolver,
    y: &N_Vector,
    mem: &mut Option<Box<dyn Any>>,
) -> i32 {
    let f = NLS.ops.borrow().setup;
    match f {
        Some(f) => f(NLS, y, mem),
        None => SUN_SUCCESS,
    }
}

pub fn SUNNonlinSolSolve(
    NLS: &SUNNonlinearSolver,
    y0: &N_Vector,
    y: &N_Vector,
    w: &N_Vector,
    tol: sunrealtype,
    callLSetup: sunbooleantype,
    mem: &mut Option<Box<dyn Any>>,
) -> i32 {
    let f = NLS.ops.borrow().solve.expect("solve");
    f(NLS, y0, y, w, tol, callLSetup, mem)
}

pub fn SUNNonlinSolFree(NLS: Option<SUNNonlinearSolver>) -> SUNErrCode {
    match NLS {
        None => SUN_SUCCESS,
        Some(NLS) => {
            let f = NLS.ops.borrow().free;
            if let Some(f) = f {
                return f(&NLS);
            }
            drop(NLS);
            SUN_SUCCESS
        }
    }
}

fn sunnlsSetFromCommandLine(
    NLS: &SUNNonlinearSolver,
    NLSid: Option<&str>,
    argv: &[String],
) -> SUNErrCode {
    let default_id = "sunnonlinearsolver";
    let id = match NLSid {
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
        if key == "max_iters" {
            idx += 1;
            let iarg: i32 = crate::sundials_utils::atoi(&argv[idx]);
            let retval = SUNNonlinSolSetMaxIters(NLS, iarg);
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

pub fn SUNNonlinSolSetSysFn(
    NLS: &SUNNonlinearSolver,
    SysFn: Option<SUNNonlinSolSysFn>,
) -> SUNErrCode {
    let f = NLS.ops.borrow().setsysfn.expect("setsysfn");
    f(NLS, SysFn)
}

pub fn SUNNonlinSolSetSysFns(
    NLS: &SUNNonlinearSolver,
    root_fn: Option<SUNNonlinSolSysFn>,
    fixed_point_fn: Option<SUNNonlinSolSysFn>,
) -> SUNErrCode {
    let f = NLS.ops.borrow().setsysfns;
    match f {
        Some(f) => f(NLS, root_fn, fixed_point_fn),
        None => SUN_SUCCESS,
    }
}

pub fn SUNNonlinSolSetLSetupFn(
    NLS: &SUNNonlinearSolver,
    LSetupFn: Option<SUNNonlinSolLSetupFn>,
) -> SUNErrCode {
    let f = NLS.ops.borrow().setlsetupfn;
    match f {
        Some(f) => f(NLS, LSetupFn),
        None => SUN_SUCCESS,
    }
}

pub fn SUNNonlinSolSetLSolveFn(
    NLS: &SUNNonlinearSolver,
    LSolveFn: Option<SUNNonlinSolLSolveFn>,
) -> SUNErrCode {
    let f = NLS.ops.borrow().setlsolvefn;
    match f {
        Some(f) => f(NLS, LSolveFn),
        None => SUN_SUCCESS,
    }
}

pub fn SUNNonlinSolSetConvTestFn(
    NLS: &SUNNonlinearSolver,
    CTestFn: Option<SUNNonlinSolConvTestFn>,
    ctest_data: Option<Box<dyn Any>>,
) -> SUNErrCode {
    let f = NLS.ops.borrow().setctestfn;
    match f {
        Some(f) => f(NLS, CTestFn, ctest_data),
        None => SUN_SUCCESS,
    }
}

pub fn SUNNonlinSolSetNormFn(
    NLS: &SUNNonlinearSolver,
    NormFn: Option<SUNNonlinSolNormFn>,
    norm_fn_data: Option<Box<dyn Any>>,
) -> SUNErrCode {
    let f = NLS.ops.borrow().setnormfn;
    match f {
        Some(f) => f(NLS, NormFn, norm_fn_data),
        None => SUN_SUCCESS,
    }
}

pub fn SUNNonlinSolSetGetUpdateNormFn(
    NLS: &SUNNonlinearSolver,
    GetUpdateNormFn: Option<SUNNonlinSolGetUpdateNormFn>,
    getupdatenorm_data: Option<Box<dyn Any>>,
) -> SUNErrCode {
    let f = NLS.ops.borrow().setgetupdatenormfn;
    match f {
        Some(f) => f(NLS, GetUpdateNormFn, getupdatenorm_data),
        None => SUN_SUCCESS,
    }
}

pub fn SUNNonlinSolSetGetConvRateFn(
    NLS: &SUNNonlinearSolver,
    GetConvRateFn: Option<SUNNonlinSolGetConvRateFn>,
    getconvrate_data: Option<Box<dyn Any>>,
) -> SUNErrCode {
    let f = NLS.ops.borrow().setgetconvratefn;
    match f {
        Some(f) => f(NLS, GetConvRateFn, getconvrate_data),
        None => SUN_SUCCESS,
    }
}

pub fn SUNNonlinSolSetOptions(
    NLS: &SUNNonlinearSolver,
    NLSid: Option<&str>,
    file_name: Option<&str>,
    argv: &[String],
) -> SUNErrCode {
    /* File-based option control is currently unimplemented */
    if !(file_name.is_none() || file_name == Some("")) {
        return SUN_ERR_ARG_INCOMPATIBLE;
    }

    /* First, process all base-class options */
    if !argv.is_empty() {
        let ier = sunnlsSetFromCommandLine(NLS, NLSid, argv);
        if ier != SUN_SUCCESS {
            return ier;
        }
    }

    /* Second, ask the implementation to process any remaining options */
    let f = NLS.ops.borrow().setoptions;
    match f {
        Some(f) => f(NLS, NLSid, file_name, argv),
        None => SUN_SUCCESS,
    }
}

pub fn SUNNonlinSolSetMaxIters(NLS: &SUNNonlinearSolver, maxiters: i32) -> SUNErrCode {
    let f = NLS.ops.borrow().setmaxiters;
    match f {
        Some(f) => f(NLS, maxiters),
        None => SUN_SUCCESS,
    }
}

pub fn SUNNonlinSolGetNumIters(NLS: &SUNNonlinearSolver, niters: &mut i64) -> SUNErrCode {
    let f = NLS.ops.borrow().getnumiters;
    match f {
        Some(f) => f(NLS, niters),
        None => {
            *niters = 0;
            SUN_SUCCESS
        }
    }
}

pub fn SUNNonlinSolGetCurIter(NLS: &SUNNonlinearSolver, iter: &mut i32) -> SUNErrCode {
    let f = NLS.ops.borrow().getcuriter;
    match f {
        Some(f) => f(NLS, iter),
        None => {
            *iter = -1;
            SUN_SUCCESS
        }
    }
}

pub fn SUNNonlinSolGetNumConvFails(NLS: &SUNNonlinearSolver, nconvfails: &mut i64) -> SUNErrCode {
    let f = NLS.ops.borrow().getnumconvfails;
    match f {
        Some(f) => f(NLS, nconvfails),
        None => {
            *nconvfails = 0;
            SUN_SUCCESS
        }
    }
}

const _: SUNErrCode = SUN_ERR_ARG_CORRUPT;
