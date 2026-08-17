//! Port of `src/sunnonlinsol/auto/sunnonlinsol_auto.c` +
//! `include/sunnonlinsol/sunnonlinsol_auto.h` (nonlinear solver that
//! automatically switches between Newton and fixed-point iterations).
//!
//! The C module mallocs one `SUNNonlinSolAutoConvTestData` and registers the
//! same pointer with both sub-solvers; here it is an `Rc<RefCell<..>>` whose
//! `auto_nls` back-pointer is a `Weak` (breaks the auto -> sub-solver ->
//! ctest-data -> auto reference cycle). C also hands the same
//! `norm_fn_data`/`getupdatenorm_data` pointer to both sub-solvers; a
//! `Box<dyn Any>` cannot be duplicated, so Newton (the only sub-solver that
//! dereferences it at logging level 2) receives the box and fixed-point
//! receives `None`. `SUNNonlinSolAutoType_ToString` only feeds omitted
//! `SUNLogInfo` calls and is not ported.

use std::any::Any;
use std::cell::{RefCell, RefMut};
use std::rc::{Rc, Weak};

use crate::sundials_context::SUNContext;
use crate::sundials_errors::{
    SUN_ERR_ARG_CORRUPT, SUN_ERR_ARG_INCOMPATIBLE, SUN_ERR_NOT_IMPLEMENTED, SUN_SUCCESS,
};
use crate::sundials_math::SUNStrToReal;
use crate::sundials_nonlinearsolver::*;
use crate::sundials_nvector::N_Vector;
use crate::sundials_types::*;
use crate::sunnonlinsol_fixedpoint::SUNNonlinSol_FixedPoint;
use crate::sunnonlinsol_newton::{
    SUNNonlinSol_Newton, SUNNonlinSolGetStiffnessRatio_Newton,
    SUNNonlinSolSetComputeStiffnessRatio_Newton,
};

const ZERO: sunrealtype = 0.0;

/* this is effectively the maximum number of times we would allow
   switching to happen within a single solve. It is unlikely to
   ever be reached, but it prevents an infinite switchign loop from
   being possible. */
const SUNNLS_AUTO_MAX_SOLVE_ATTEMPTS: i64 = 3;

/* Default switching parameters
   2.0 and 0.8 come from the numerical experiments in Norsett & Thomsen 1986,
   as does the Newton to fixed-point switch delay of 10 solves.
   The idea behind setting the fixed-point to Newton switch delay to 1 is to
   allow the integrator to see at least one failed convergence test, cut the time step,
   and try fixed-point again before switching to Newton.
*/
const SUNNLS_AUTO_DEFAULT_NEWT_TO_FP_THRESHOLD: sunrealtype = 2.0;
const SUNNLS_AUTO_DEFAULT_FP_TO_NEWT_THRESHOLD: sunrealtype = 0.8;
const SUNNLS_AUTO_DEFAULT_NEWT_TO_FP_DELAY: i64 = 10;
const SUNNLS_AUTO_DEFAULT_FP_TO_NEWT_DELAY: i64 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SUNNonlinSolAutoType {
    SUNNONLINSOL_AUTO_FIXEDPOINT = 0,
    SUNNONLINSOL_AUTO_NEWTON = 1,
}
pub use SUNNonlinSolAutoType::*;

pub struct SUNNonlinSolAutoConvTestData {
    pub auto_nls: Weak<_generic_SUNNonlinearSolver>,
    pub user_ctest_fn: Option<SUNNonlinSolConvTestFn>,
    pub user_ctest_data: Option<Box<dyn Any>>,
}

pub struct SUNNonlinearSolverContent_Auto_ {
    pub active_solver_type: SUNNonlinSolAutoType,
    pub fp_solver: Option<SUNNonlinearSolver>,
    pub newton_solver: Option<SUNNonlinearSolver>,
    pub getconvrate_fn: Option<SUNNonlinSolGetConvRateFn>,
    pub getconvrate_data: Option<Box<dyn Any>>,
    pub fp_to_newt_delay: i64,
    pub newt_to_fp_delay: i64,
    pub num_solves_since_switch: i64,
    pub newt_to_fp_threshold: sunrealtype,
    pub fp_to_newt_threshold: sunrealtype,
    pub num_iters: i64,
    pub num_conv_fails: i64,
    pub switch_count: i64,
    pub fp_num_iters_total: i64,
    pub newton_num_iters_total: i64,
    pub fp_num_conv_fails_total: i64,
    pub newton_num_conv_fails_total: i64,
    pub auto_ctest_data: Option<Rc<RefCell<SUNNonlinSolAutoConvTestData>>>,
}

pub type SUNNonlinearSolverContent_Auto = SUNNonlinearSolverContent_Auto_;

fn content_mut(NLS: &SUNNonlinearSolver) -> RefMut<'_, SUNNonlinearSolverContent_Auto_> {
    RefMut::map(NLS.content.borrow_mut(), |c| {
        c.downcast_mut::<SUNNonlinearSolverContent_Auto_>()
            .expect("Auto SUNNonlinearSolver content")
    })
}

pub fn SUNNonlinSol_Auto(
    y: &N_Vector,
    m: i32,
    initial_solver_type: SUNNonlinSolAutoType,
    sunctx: &SUNContext,
) -> Option<SUNNonlinearSolver> {
    /* Create an empty nonlinear solver object */
    let NLS = SUNNonlinSolNewEmpty(sunctx)?;

    /* Attach operations */
    {
        let mut ops = NLS.ops.borrow_mut();
        ops.gettype = Some(SUNNonlinSolGetType_Auto);
        ops.initialize = Some(SUNNonlinSolInitialize_Auto);
        ops.solve = Some(SUNNonlinSolSolve_Auto);
        ops.free = Some(SUNNonlinSolFree_Auto);
        ops.setsysfns = Some(SUNNonlinSolSetSysFns_Auto);
        ops.setctestfn = Some(SUNNonlinSolSetConvTestFn_Auto);
        ops.setnormfn = Some(SUNNonlinSolSetNormFn_Auto);
        ops.setgetupdatenormfn = Some(SUNNonlinSolSetGetUpdateNormFn_Auto);
        ops.setgetconvratefn = Some(SUNNonlinSolSetGetConvRateFn_Auto);
        ops.setlsetupfn = Some(SUNNonlinSolSetLSetupFn_Auto);
        ops.setlsolvefn = Some(SUNNonlinSolSetLSolveFn_Auto);
        ops.setoptions = Some(SUNNonlinSolSetOptions_Auto);
        ops.setmaxiters = Some(SUNNonlinSolSetMaxIters_Auto);
        ops.getnumiters = Some(SUNNonlinSolGetNumIters_Auto);
        ops.getcuriter = Some(SUNNonlinSolGetCurIter_Auto);
        ops.getnumconvfails = Some(SUNNonlinSolGetNumConvFails_Auto);
    }

    /* Create the wrapped sub-solvers (C fills content->fp_solver and
       content->newton_solver; scalar fields are assigned with the content
       struct below) */
    let fp_solver = SUNNonlinSol_FixedPoint(y, m, sunctx);
    let newton_solver = SUNNonlinSol_Newton(y, sunctx);
    let (fp_solver, newton_solver) = match (fp_solver, newton_solver) {
        (Some(fp_solver), Some(newton_solver)) => (fp_solver, newton_solver),
        (fp_solver, newton_solver) => {
            let _ = SUNNonlinSolFree(fp_solver);
            let _ = SUNNonlinSolFree(newton_solver);
            SUNNonlinSolFreeEmpty(NLS);
            return None;
        }
    };

    /* Shared convergence-test trampoline data (C mallocs one struct and
       registers the same pointer with both sub-solvers) */
    let auto_ctest_data = Rc::new(RefCell::new(SUNNonlinSolAutoConvTestData {
        auto_nls: Rc::downgrade(&NLS),
        user_ctest_fn: None,
        user_ctest_data: None,
    }));

    if SUNNonlinSolSetConvTestFn(
        &fp_solver,
        Some(SUNNonlinSolConvTest_Auto),
        Some(Box::new(auto_ctest_data.clone())),
    ) != SUN_SUCCESS
        || SUNNonlinSolSetConvTestFn(
            &newton_solver,
            Some(SUNNonlinSolConvTest_Auto),
            Some(Box::new(auto_ctest_data.clone())),
        ) != SUN_SUCCESS
    {
        let _ = SUNNonlinSolFree(Some(fp_solver));
        let _ = SUNNonlinSolFree(Some(newton_solver));
        SUNNonlinSolFreeEmpty(NLS);
        return None;
    }

    if SUNNonlinSolSetComputeStiffnessRatio_Newton(&newton_solver, SUNTRUE) != SUN_SUCCESS {
        let _ = SUNNonlinSolFree(Some(fp_solver));
        let _ = SUNNonlinSolFree(Some(newton_solver));
        SUNNonlinSolFreeEmpty(NLS);
        return None;
    }

    /* Attach and fill content */
    *NLS.content.borrow_mut() = Box::new(SUNNonlinearSolverContent_Auto_ {
        active_solver_type: initial_solver_type,
        fp_solver: Some(fp_solver),
        newton_solver: Some(newton_solver),
        getconvrate_fn: None,
        getconvrate_data: None,
        fp_to_newt_delay: SUNNLS_AUTO_DEFAULT_FP_TO_NEWT_DELAY,
        newt_to_fp_delay: SUNNLS_AUTO_DEFAULT_NEWT_TO_FP_DELAY,
        num_solves_since_switch: 0,
        newt_to_fp_threshold: SUNNLS_AUTO_DEFAULT_NEWT_TO_FP_THRESHOLD,
        fp_to_newt_threshold: SUNNLS_AUTO_DEFAULT_FP_TO_NEWT_THRESHOLD,
        num_iters: 0,
        num_conv_fails: 0,
        switch_count: 0,
        fp_num_iters_total: 0,
        newton_num_iters_total: 0,
        fp_num_conv_fails_total: 0,
        newton_num_conv_fails_total: 0,
        auto_ctest_data: Some(auto_ctest_data),
    });

    Some(NLS)
}

pub fn SUNNonlinSolGetType_Auto(_NLS: &SUNNonlinearSolver) -> SUNNonlinearSolver_Type {
    SUNNONLINEARSOLVER_HYBRID
}

pub fn SUNNonlinSolInitialize_Auto(NLS: &SUNNonlinearSolver) -> SUNErrCode {
    let fp_solver = content_mut(NLS).fp_solver.as_ref().expect("fp_solver").clone();
    let retval = SUNNonlinSolInitialize(&fp_solver);
    if retval != SUN_SUCCESS {
        return retval;
    }
    let newton_solver = content_mut(NLS)
        .newton_solver
        .as_ref()
        .expect("newton_solver")
        .clone();
    let retval = SUNNonlinSolInitialize(&newton_solver);
    if retval != SUN_SUCCESS {
        return retval;
    }
    SUN_SUCCESS
}

pub fn SUNNonlinSolSolve_Auto(
    NLS: &SUNNonlinearSolver,
    y0: &N_Vector,
    ycor: &N_Vector,
    w: &N_Vector,
    tol: sunrealtype,
    callSetup: sunbooleantype,
    mem: &mut Option<Box<dyn Any>>,
) -> i32 {
    {
        let mut C = content_mut(NLS);
        C.num_iters = 0;
        C.num_conv_fails = 0;
    }

    for _attempts in 0..SUNNLS_AUTO_MAX_SOLVE_ATTEMPTS {
        /* re-read the active solver every attempt: the convergence-test
           wrapper may flip it during the sub-solve */
        let (solve_solver_type, subsolver) = {
            let C = content_mut(NLS);
            let solve_solver_type = C.active_solver_type;
            let subsolver = if solve_solver_type == SUNNONLINSOL_AUTO_FIXEDPOINT {
                C.fp_solver.as_ref().expect("fp_solver").clone()
            } else {
                C.newton_solver.as_ref().expect("newton_solver").clone()
            };
            (solve_solver_type, subsolver)
        };

        let retval = SUNNonlinSolSolve(&subsolver, y0, ycor, w, tol, callSetup, mem);

        let mut iters: i64 = 0;
        if SUNNonlinSolGetNumIters(&subsolver, &mut iters) == SUN_SUCCESS {
            let mut C = content_mut(NLS);
            C.num_iters += iters;
            if solve_solver_type == SUNNONLINSOL_AUTO_FIXEDPOINT {
                C.fp_num_iters_total += iters;
            } else {
                C.newton_num_iters_total += iters;
            }
        }

        let mut nconvfails: i64 = 0;
        if SUNNonlinSolGetNumConvFails(&subsolver, &mut nconvfails) == SUN_SUCCESS {
            let mut C = content_mut(NLS);
            C.num_conv_fails += nconvfails;
            if solve_solver_type == SUNNONLINSOL_AUTO_FIXEDPOINT {
                C.fp_num_conv_fails_total += nconvfails;
            } else {
                C.newton_num_conv_fails_total += nconvfails;
            }
        }

        if retval == SUN_NLS_SWITCH {
            continue;
        }

        content_mut(NLS).num_solves_since_switch += 1;

        return retval;
    }

    SUN_SUCCESS
}

fn SUNNonlinSolConvTest_Auto(
    sub_nls: &SUNNonlinearSolver,
    y: &N_Vector,
    del: &N_Vector,
    tol: sunrealtype,
    ewt: &N_Vector,
    mem: &mut Option<Box<dyn Any>>,
) -> i32 {
    let data = match mem
        .as_mut()
        .and_then(|d| d.downcast_mut::<Rc<RefCell<SUNNonlinSolAutoConvTestData>>>())
    {
        Some(d) => d.clone(),
        None => return SUN_ERR_ARG_CORRUPT,
    };
    if data.borrow().user_ctest_fn.is_none() {
        return SUN_ERR_ARG_CORRUPT;
    }
    let auto_nls = data.borrow().auto_nls.upgrade().expect("auto NLS alive");

    let user_ctest_fn = data.borrow().user_ctest_fn.expect("user_ctest_fn");
    let mut user_ctest_data = data.borrow_mut().user_ctest_data.take();
    let retval = user_ctest_fn(sub_nls, y, del, tol, ewt, &mut user_ctest_data);
    data.borrow_mut().user_ctest_data = user_ctest_data;

    /* Return early if convergence test error is unrecoverable */
    if retval < 0 {
        return retval;
    }

    /* We follow the switching strategy outlined in
    Norsett, Syvert P., and Per G. Thomsen. "Switching between modified Newton
    and fix-point iteration for implicit ODE-solvers." BIT Numerical Mathematics
    26, no. 3 (1986): 339-348. https://doi.org/10.1007/BF01933714. */

    if content_mut(&auto_nls).active_solver_type == SUNNONLINSOL_AUTO_FIXEDPOINT {
        /* If the integrator-provided convergence test passed, exit with success and
        don't consider switching, since fixed-point is still converging fine. */
        if retval == SUN_SUCCESS {
            return SUN_SUCCESS;
        }

        /* Get the convergence rate from the user-provided function */
        let getconvrate_fn = match content_mut(&auto_nls).getconvrate_fn {
            Some(f) => f,
            None => return SUN_ERR_NOT_IMPLEMENTED,
        };
        let mut crate_: sunrealtype = ZERO;
        let mut getconvrate_data = content_mut(&auto_nls).getconvrate_data.take();
        let crate_retval = getconvrate_fn(&mut crate_, &mut getconvrate_data);
        content_mut(&auto_nls).getconvrate_data = getconvrate_data;
        if crate_retval != SUN_SUCCESS {
            return crate_retval;
        }

        let mut C = content_mut(&auto_nls);
        let diverging: sunbooleantype = crate_ >= C.fp_to_newt_threshold;
        let dont_delay: sunbooleantype = C.num_solves_since_switch >= C.fp_to_newt_delay;

        if diverging && dont_delay {
            C.num_solves_since_switch = 0;
            C.active_solver_type = SUNNONLINSOL_AUTO_NEWTON;
            C.switch_count += 1;
            /* Return SUN_NLS_SWITCH so that the solver loop continues but with Newton */
            return SUN_NLS_SWITCH;
        }
    } else {
        /* Since Newton is active, check if we should switch to fixed-point regardless
        of if the convergence test passed. */

        /* Get the stiffness ratio from the Newton solver */
        let newton_solver = content_mut(&auto_nls)
            .newton_solver
            .as_ref()
            .expect("newton_solver")
            .clone();
        let mut stiffr: sunrealtype = ZERO;
        let stiffr_retval = SUNNonlinSolGetStiffnessRatio_Newton(&newton_solver, &mut stiffr);
        if stiffr_retval != SUN_SUCCESS {
            return stiffr_retval;
        }

        let mut C = content_mut(&auto_nls);
        let contraction: sunbooleantype = stiffr < C.newt_to_fp_threshold;
        let dont_delay: sunbooleantype = C.num_solves_since_switch >= C.newt_to_fp_delay;

        if contraction && dont_delay {
            C.num_solves_since_switch = 0;
            C.active_solver_type = SUNNONLINSOL_AUTO_FIXEDPOINT;
            C.switch_count += 1;

            /* If the integrator-provided convergence test passed, then we return with success
            so that the solve loop is stopped. The switch to fixed-point will happen on
            the next time step. If the convergence test failed, we return SUN_NLS_SWITCH
            so that the solver loop continues but with fixed-point iteration. */
            if retval == SUN_SUCCESS {
                return SUN_SUCCESS;
            } else {
                return SUN_NLS_SWITCH;
            }
        }
    }

    retval
}

pub fn SUNNonlinSolFree_Auto(NLS: &SUNNonlinearSolver) -> SUNErrCode {
    /* Handle-model port: dropping the handle releases everything; mirror the
       C cleanup by explicitly releasing the shared ctest data and the wrapped
       sub-solvers. */
    let (fp_solver, newton_solver) = {
        let mut content = content_mut(NLS);
        content.auto_ctest_data = None;
        (content.fp_solver.take(), content.newton_solver.take())
    };
    let _ = SUNNonlinSolFree(fp_solver);
    let _ = SUNNonlinSolFree(newton_solver);
    SUN_SUCCESS
}

pub fn SUNNonlinSolSetSysFns_Auto(
    NLS: &SUNNonlinearSolver,
    root_sys_fn: Option<SUNNonlinSolSysFn>,
    fixed_point_fn: Option<SUNNonlinSolSysFn>,
) -> SUNErrCode {
    let newton_solver = content_mut(NLS)
        .newton_solver
        .as_ref()
        .expect("newton_solver")
        .clone();
    let retval = SUNNonlinSolSetSysFn(&newton_solver, root_sys_fn);
    if retval != SUN_SUCCESS {
        return retval;
    }
    let fp_solver = content_mut(NLS).fp_solver.as_ref().expect("fp_solver").clone();
    let retval = SUNNonlinSolSetSysFn(&fp_solver, fixed_point_fn);
    if retval != SUN_SUCCESS {
        return retval;
    }
    SUN_SUCCESS
}

pub fn SUNNonlinSolSetConvTestFn_Auto(
    NLS: &SUNNonlinearSolver,
    CTestFn: Option<SUNNonlinSolConvTestFn>,
    ctest_data: Option<Box<dyn Any>>,
) -> SUNErrCode {
    if CTestFn.is_none() {
        return SUN_ERR_ARG_CORRUPT;
    }

    let data = content_mut(NLS)
        .auto_ctest_data
        .as_ref()
        .expect("auto_ctest_data")
        .clone();
    let mut data = data.borrow_mut();
    data.auto_nls = Rc::downgrade(NLS);
    data.user_ctest_fn = CTestFn;
    data.user_ctest_data = ctest_data;
    SUN_SUCCESS
}

pub fn SUNNonlinSolSetLSetupFn_Auto(
    NLS: &SUNNonlinearSolver,
    LSetupFn: Option<SUNNonlinSolLSetupFn>,
) -> SUNErrCode {
    let newton_solver = content_mut(NLS)
        .newton_solver
        .as_ref()
        .expect("newton_solver")
        .clone();
    let retval = SUNNonlinSolSetLSetupFn(&newton_solver, LSetupFn);
    if retval != SUN_SUCCESS {
        return retval;
    }
    SUN_SUCCESS
}

pub fn SUNNonlinSolSetLSolveFn_Auto(
    NLS: &SUNNonlinearSolver,
    LSolveFn: Option<SUNNonlinSolLSolveFn>,
) -> SUNErrCode {
    let newton_solver = content_mut(NLS)
        .newton_solver
        .as_ref()
        .expect("newton_solver")
        .clone();
    let retval = SUNNonlinSolSetLSolveFn(&newton_solver, LSolveFn);
    if retval != SUN_SUCCESS {
        return retval;
    }
    SUN_SUCCESS
}

pub fn SUNNonlinSolSetNormFn_Auto(
    NLS: &SUNNonlinearSolver,
    NormFn: Option<SUNNonlinSolNormFn>,
    norm_fn_data: Option<Box<dyn Any>>,
) -> SUNErrCode {
    /* C hands the same `norm_fn_data` pointer to both sub-solvers; the box
       cannot be duplicated, and at the reference logging level the
       fixed-point solver never invokes its norm function, so Newton (which
       does) keeps the data. */
    let fp_solver = content_mut(NLS).fp_solver.as_ref().expect("fp_solver").clone();
    let retval = SUNNonlinSolSetNormFn(&fp_solver, NormFn, None);
    if retval != SUN_SUCCESS {
        return retval;
    }
    let newton_solver = content_mut(NLS)
        .newton_solver
        .as_ref()
        .expect("newton_solver")
        .clone();
    let retval = SUNNonlinSolSetNormFn(&newton_solver, NormFn, norm_fn_data);
    if retval != SUN_SUCCESS {
        return retval;
    }
    SUN_SUCCESS
}

pub fn SUNNonlinSolSetGetUpdateNormFn_Auto(
    NLS: &SUNNonlinearSolver,
    GetUpdateNormFn: Option<SUNNonlinSolGetUpdateNormFn>,
    getupdatenorm_data: Option<Box<dyn Any>>,
) -> SUNErrCode {
    /* Same single-owner mapping as SUNNonlinSolSetNormFn_Auto. */
    let fp_solver = content_mut(NLS).fp_solver.as_ref().expect("fp_solver").clone();
    let retval = SUNNonlinSolSetGetUpdateNormFn(&fp_solver, GetUpdateNormFn, None);
    if retval != SUN_SUCCESS {
        return retval;
    }
    let newton_solver = content_mut(NLS)
        .newton_solver
        .as_ref()
        .expect("newton_solver")
        .clone();
    let retval =
        SUNNonlinSolSetGetUpdateNormFn(&newton_solver, GetUpdateNormFn, getupdatenorm_data);
    if retval != SUN_SUCCESS {
        return retval;
    }
    SUN_SUCCESS
}

pub fn SUNNonlinSolSetGetConvRateFn_Auto(
    NLS: &SUNNonlinearSolver,
    GetConvRateFn: Option<SUNNonlinSolGetConvRateFn>,
    getconvrate_data: Option<Box<dyn Any>>,
) -> SUNErrCode {
    let mut content = content_mut(NLS);
    content.getconvrate_fn = GetConvRateFn;
    content.getconvrate_data = getconvrate_data;
    SUN_SUCCESS
}

pub fn SUNNonlinSolSetMaxIters_Auto(NLS: &SUNNonlinearSolver, maxiters: i32) -> SUNErrCode {
    let fp_solver = content_mut(NLS).fp_solver.as_ref().expect("fp_solver").clone();
    let retval = SUNNonlinSolSetMaxIters(&fp_solver, maxiters);
    if retval != SUN_SUCCESS {
        return retval;
    }
    let newton_solver = content_mut(NLS)
        .newton_solver
        .as_ref()
        .expect("newton_solver")
        .clone();
    let retval = SUNNonlinSolSetMaxIters(&newton_solver, maxiters);
    if retval != SUN_SUCCESS {
        return retval;
    }
    SUN_SUCCESS
}

pub fn SUNNonlinSolSetSwitchingParameters_Auto(
    NLS: &SUNNonlinearSolver,
    newt_to_fp_threshold: sunrealtype,
    newt_to_fp_delay: i64,
    fp_to_newt_threshold: sunrealtype,
    fp_to_newt_delay: i64,
) -> SUNErrCode {
    let mut content = content_mut(NLS);

    content.newt_to_fp_threshold = if newt_to_fp_threshold < ZERO {
        SUNNLS_AUTO_DEFAULT_NEWT_TO_FP_THRESHOLD
    } else {
        newt_to_fp_threshold
    };
    content.newt_to_fp_delay = if newt_to_fp_delay < 0 {
        SUNNLS_AUTO_DEFAULT_NEWT_TO_FP_DELAY
    } else {
        newt_to_fp_delay
    };
    content.fp_to_newt_threshold = if fp_to_newt_threshold < ZERO {
        SUNNLS_AUTO_DEFAULT_FP_TO_NEWT_THRESHOLD
    } else {
        fp_to_newt_threshold
    };
    content.fp_to_newt_delay = if fp_to_newt_delay < 0 {
        SUNNLS_AUTO_DEFAULT_FP_TO_NEWT_DELAY
    } else {
        fp_to_newt_delay
    };

    SUN_SUCCESS
}

pub fn SUNNonlinSolGetNumIters_Auto(NLS: &SUNNonlinearSolver, niters: &mut i64) -> SUNErrCode {
    *niters = content_mut(NLS).num_iters;
    SUN_SUCCESS
}

pub fn SUNNonlinSolGetFixedPointSolver_Auto(
    NLS: &SUNNonlinearSolver,
    fp_nls: &mut Option<SUNNonlinearSolver>,
) -> SUNErrCode {
    *fp_nls = content_mut(NLS).fp_solver.clone();
    SUN_SUCCESS
}

pub fn SUNNonlinSolGetNewtonSolver_Auto(
    NLS: &SUNNonlinearSolver,
    newton_nls: &mut Option<SUNNonlinearSolver>,
) -> SUNErrCode {
    *newton_nls = content_mut(NLS).newton_solver.clone();
    SUN_SUCCESS
}

pub fn SUNNonlinSolGetActiveSolverType_Auto(
    NLS: &SUNNonlinearSolver,
    active_solver_type: &mut SUNNonlinSolAutoType,
) -> SUNErrCode {
    *active_solver_type = content_mut(NLS).active_solver_type;
    SUN_SUCCESS
}

pub fn SUNNonlinSolGetSwitchCount_Auto(
    NLS: &SUNNonlinearSolver,
    switch_count: &mut i64,
) -> SUNErrCode {
    *switch_count = content_mut(NLS).switch_count;
    SUN_SUCCESS
}

pub fn SUNNonlinSolGetTotalNumItersByType_Auto(
    NLS: &SUNNonlinearSolver,
    fp_iters: &mut i64,
    newt_iters: &mut i64,
) -> SUNErrCode {
    let content = content_mut(NLS);
    *fp_iters = content.fp_num_iters_total;
    *newt_iters = content.newton_num_iters_total;

    SUN_SUCCESS
}

pub fn SUNNonlinSolGetCurIter_Auto(NLS: &SUNNonlinearSolver, iter: &mut i32) -> SUNErrCode {
    if content_mut(NLS).active_solver_type == SUNNONLINSOL_AUTO_FIXEDPOINT {
        let fp_solver = content_mut(NLS).fp_solver.as_ref().expect("fp_solver").clone();
        SUNNonlinSolGetCurIter(&fp_solver, iter)
    } else {
        let newton_solver = content_mut(NLS)
            .newton_solver
            .as_ref()
            .expect("newton_solver")
            .clone();
        SUNNonlinSolGetCurIter(&newton_solver, iter)
    }
}

pub fn SUNNonlinSolGetNumConvFails_Auto(
    NLS: &SUNNonlinearSolver,
    nconvfails: &mut i64,
) -> SUNErrCode {
    *nconvfails = content_mut(NLS).num_conv_fails;
    SUN_SUCCESS
}

pub fn SUNNonlinSolGetTotalNumConvFailsByType_Auto(
    NLS: &SUNNonlinearSolver,
    fp_nconvfails: &mut i64,
    newt_nconvfails: &mut i64,
) -> SUNErrCode {
    let content = content_mut(NLS);
    *fp_nconvfails = content.fp_num_conv_fails_total;
    *newt_nconvfails = content.newton_num_conv_fails_total;
    SUN_SUCCESS
}

pub fn SUNNonlinSolSetOptions_Auto(
    NLS: &SUNNonlinearSolver,
    NLSid: Option<&str>,
    file_name: Option<&str>,
    argv: &[String],
) -> SUNErrCode {
    if !(file_name.is_none() || file_name == Some("")) {
        return SUN_ERR_ARG_INCOMPATIBLE;
    }

    if !argv.is_empty() {
        let retval = setFromCommandLine_Auto(NLS, NLSid, argv);
        if retval != SUN_SUCCESS {
            return retval;
        }
    }

    SUN_SUCCESS
}

fn setFromCommandLine_Auto(
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
        if key == "switching_parameters" {
            if idx + 4 >= argv.len() {
                return SUN_ERR_ARG_INCOMPATIBLE;
            }
            let retval = SUNNonlinSolSetSwitchingParameters_Auto(
                NLS,
                SUNStrToReal(argv[idx + 1].trim()),
                crate::sundials_utils::atol(&argv[idx + 2]),
                SUNStrToReal(argv[idx + 3].trim()),
                crate::sundials_utils::atol(&argv[idx + 4]),
            );
            if retval != SUN_SUCCESS {
                return retval;
            }
            idx += 5;
            continue;
        }
        idx += 1;
    }

    SUN_SUCCESS
}
