//! Port of `src/sunnonlinsol/newton/sunnonlinsol_newton.c` +
//! `include/sunnonlinsol/sunnonlinsol_newton.h` (Newton iteration).
//!
//! Callback-data boxes (`ctest_data`, `norm_fn_data`,
//! `getupdatenorm_data`) are taken out of the content around each
//! invocation; the integrator `mem` token passes straight through.

use std::any::Any;
use std::cell::RefMut;

use crate::sundials_context::SUNContext;
use crate::sundials_errors::{SUN_ERR_ARG_CORRUPT, SUN_SUCCESS};
use crate::sundials_nonlinearsolver::*;
use crate::sundials_nvector::*;
use crate::sundials_nvector_senswrapper::N_VNew_SensWrapper;
use crate::sundials_types::*;

const ZERO: sunrealtype = 0.0;
const ONE: sunrealtype = 1.0;

pub struct SUNNonlinearSolverContent_Newton_ {
    pub Sys: Option<SUNNonlinSolSysFn>,
    pub LSetup: Option<SUNNonlinSolLSetupFn>,
    pub LSolve: Option<SUNNonlinSolLSolveFn>,
    pub CTest: Option<SUNNonlinSolConvTestFn>,
    pub norm_fn: Option<SUNNonlinSolNormFn>,
    pub norm_fn_data: Option<Box<dyn Any>>,
    pub getupdatenorm_fn: Option<SUNNonlinSolGetUpdateNormFn>,
    pub getupdatenorm_data: Option<Box<dyn Any>>,

    pub delta: Option<N_Vector>,
    pub jcur: sunbooleantype,
    pub curiter: i32,
    pub maxiters: i32,
    pub niters: i64,
    pub nconvfails: i64,
    pub compute_stiffr: sunbooleantype,
    pub stiffr: sunrealtype,
    pub delnrm: sunrealtype,
    pub ctest_data: Option<Box<dyn Any>>,
}

pub type SUNNonlinearSolverContent_Newton = SUNNonlinearSolverContent_Newton_;

fn content_mut(NLS: &SUNNonlinearSolver) -> RefMut<'_, SUNNonlinearSolverContent_Newton_> {
    RefMut::map(NLS.content.borrow_mut(), |c| {
        c.downcast_mut::<SUNNonlinearSolverContent_Newton_>()
            .expect("Newton SUNNonlinearSolver content")
    })
}

/// C `GetUpdateNorm_Newton` (retrieve or compute the update norm).
fn GetUpdateNorm_Newton(NLS: &SUNNonlinearSolver, delta: &N_Vector, w: &N_Vector) -> SUNErrCode {
    let getupdatenorm_fn = content_mut(NLS).getupdatenorm_fn;
    if let Some(f) = getupdatenorm_fn {
        let mut data = content_mut(NLS).getupdatenorm_data.take();
        let mut delnrm = content_mut(NLS).delnrm;
        let ier = f(&mut delnrm, &mut data);
        let mut content = content_mut(NLS);
        content.getupdatenorm_data = data;
        content.delnrm = delnrm;
        return ier;
    }

    let norm_fn = content_mut(NLS).norm_fn;
    if let Some(f) = norm_fn {
        let mut data = content_mut(NLS).norm_fn_data.take();
        let mut delnrm = content_mut(NLS).delnrm;
        let ier = f(delta, w, &mut delnrm, &mut data);
        let mut content = content_mut(NLS);
        content.norm_fn_data = data;
        content.delnrm = delnrm;
        return ier;
    }

    content_mut(NLS).delnrm = N_VWrmsNorm(delta, w);
    SUN_SUCCESS
}

pub fn SUNNonlinSol_Newton(y: &N_Vector, sunctx: &SUNContext) -> Option<SUNNonlinearSolver> {
    /* Check that the supplied N_Vector supports all required operations */
    {
        let ops = y.ops.borrow();
        if ops.nvclone.is_none() || ops.nvscale.is_none() || ops.nvlinearsum.is_none() {
            return None;
        }
    }

    /* Create an empty nonlinear linear solver object */
    let NLS = SUNNonlinSolNewEmpty(sunctx)?;

    /* Attach operations */
    {
        let mut ops = NLS.ops.borrow_mut();
        ops.gettype = Some(SUNNonlinSolGetType_Newton);
        ops.initialize = Some(SUNNonlinSolInitialize_Newton);
        ops.solve = Some(SUNNonlinSolSolve_Newton);
        ops.free = Some(SUNNonlinSolFree_Newton);
        ops.setsysfn = Some(SUNNonlinSolSetSysFn_Newton);
        ops.setlsetupfn = Some(SUNNonlinSolSetLSetupFn_Newton);
        ops.setlsolvefn = Some(SUNNonlinSolSetLSolveFn_Newton);
        ops.setctestfn = Some(SUNNonlinSolSetConvTestFn_Newton);
        ops.setnormfn = Some(SUNNonlinSolSetNormFn_Newton);
        ops.setgetupdatenormfn = Some(SUNNonlinSolSetGetUpdateNormFn_Newton);
        ops.setmaxiters = Some(SUNNonlinSolSetMaxIters_Newton);
        ops.getnumiters = Some(SUNNonlinSolGetNumIters_Newton);
        ops.getcuriter = Some(SUNNonlinSolGetCurIter_Newton);
        ops.getnumconvfails = Some(SUNNonlinSolGetNumConvFails_Newton);
    }

    /* Create, attach, and fill content */
    let delta = N_VClone(y)?;
    *NLS.content.borrow_mut() = Box::new(SUNNonlinearSolverContent_Newton_ {
        Sys: None,
        LSetup: None,
        LSolve: None,
        CTest: None,
        norm_fn: None,
        norm_fn_data: None,
        getupdatenorm_fn: None,
        getupdatenorm_data: None,
        jcur: SUNFALSE,
        curiter: 0,
        maxiters: 3,
        niters: 0,
        nconvfails: 0,
        compute_stiffr: SUNFALSE,
        stiffr: 0.0,
        delnrm: 0.0,
        ctest_data: None,
        delta: Some(delta),
    });

    Some(NLS)
}

/// Constructor wrapper to create a new Newton solver for sensitivity solvers.
pub fn SUNNonlinSol_NewtonSens(
    count: i32,
    y: &N_Vector,
    sunctx: &SUNContext,
) -> Option<SUNNonlinearSolver> {
    /* create sensitivity vector wrapper */
    let w = N_VNew_SensWrapper(count, y)?;

    /* create nonlinear solver using sensitivity vector wrapper */
    let NLS = SUNNonlinSol_Newton(&w, sunctx)?;

    /* free sensitivity vector wrapper */
    N_VDestroy(w);

    Some(NLS)
}

pub fn SUNNonlinSolGetType_Newton(_NLS: &SUNNonlinearSolver) -> SUNNonlinearSolver_Type {
    SUNNONLINEARSOLVER_ROOTFIND
}

pub fn SUNNonlinSolInitialize_Newton(NLS: &SUNNonlinearSolver) -> SUNErrCode {
    let mut content = content_mut(NLS);

    /* check that all required function pointers have been set */
    if content.Sys.is_none() || content.CTest.is_none() || content.LSolve.is_none() {
        return SUN_ERR_ARG_CORRUPT;
    }

    /* reset the total number of iterations and convergence failures */
    content.niters = 0;
    content.nconvfails = 0;
    content.stiffr = 0.0;

    /* reset the Jacobian status */
    content.jcur = SUNFALSE;

    SUN_SUCCESS
}

pub fn SUNNonlinSolSolve_Newton(
    NLS: &SUNNonlinearSolver,
    _y0: &N_Vector,
    ycor: &N_Vector,
    w: &N_Vector,
    tol: sunrealtype,
    callLSetup: sunbooleantype,
    mem: &mut Option<Box<dyn Any>>,
) -> i32 {
    let mut callLSetup = callLSetup;

    /* set local shortcut variables */
    let (delta, Sys, LSetup, LSolve, CTest);
    {
        let content = content_mut(NLS);
        delta = content.delta.as_ref().expect("delta").clone();
        Sys = content.Sys.expect("Sys set");
        LSetup = content.LSetup;
        LSolve = content.LSolve.expect("LSolve set");
        CTest = content.CTest.expect("CTest set");
    }

    /* assume the Jacobian is good */
    let mut jbad = SUNFALSE;

    /* initialize iteration and convergence fail counters for this solve */
    {
        let mut content = content_mut(NLS);
        content.niters = 0;
        content.nconvfails = 0;
    }

    let mut retval: i32;

    /* looping point for attempts at solution of the nonlinear system */
    'setup_loop: loop {
        /* initialize current iteration counter for this solve attempt */
        content_mut(NLS).curiter = 0;

        /* compute the nonlinear residual, store in delta */
        retval = Sys(ycor, &delta, mem);
        if retval != SUN_SUCCESS {
            /* C breaks straight out of the outer loop here
            (sunnonlinsol_newton.c:258-259): initial-residual failures
            must NOT reach the jbad-retry block below */
            break 'setup_loop;
        }
        if retval == SUN_SUCCESS {
            /* if indicated, setup the linear system */
            if callLSetup {
                let mut jcur = content_mut(NLS).jcur;
                retval = (LSetup.expect("LSetup set"))(jbad, &mut jcur, mem);
                content_mut(NLS).jcur = jcur;
                if retval != SUN_SUCCESS {
                    /* C direct exit (sunnonlinsol_newton.c:266) */
                    break 'setup_loop;
                }
            }

            if retval == SUN_SUCCESS {
                /* looping point for Newton iteration. Break out on any error. */
                loop {
                    /* increment nonlinear solver iteration counter */
                    content_mut(NLS).niters += 1;

                    /* compute the negative of the residual for the linear
                    system rhs */
                    N_VScale(-ONE, &delta, &delta);

                    /* solve the linear system to get Newton update delta */
                    retval = LSolve(&delta, mem);
                    if retval != SUN_SUCCESS {
                        break;
                    }

                    /* update the Newton iterate */
                    N_VLinearSum(ONE, ycor, ONE, &delta, ycor);

                    /* test for convergence */
                    let mut ctest_data = content_mut(NLS).ctest_data.take();
                    retval = CTest(NLS, ycor, &delta, tol, w, &mut ctest_data);
                    content_mut(NLS).ctest_data = ctest_data;

                    content_mut(NLS).curiter += 1;

                    let ierr = GetUpdateNorm_Newton(NLS, &delta, w);
                    if ierr != SUN_SUCCESS {
                        return ierr;
                    }

                    /* if successful update Jacobian status and return */
                    if retval == SUN_SUCCESS {
                        content_mut(NLS).jcur = SUNFALSE;
                        return SUN_SUCCESS;
                    } else if retval == SUN_NLS_SWITCH {
                        return SUN_NLS_SWITCH;
                    }

                    /* check if the iteration should continue */
                    if retval != SUN_NLS_CONTINUE {
                        break;
                    }

                    /* not yet converged, test for max allowed iterations */
                    let (curiter, maxiters) = {
                        let content = content_mut(NLS);
                        (content.curiter, content.maxiters)
                    };
                    if curiter >= maxiters {
                        retval = SUN_NLS_CONV_RECVR;
                        break;
                    }

                    /* compute the nonlinear residual, store in delta */
                    retval = Sys(ycor, &delta, mem);
                    if retval != SUN_SUCCESS {
                        break;
                    }

                    if content_mut(NLS).compute_stiffr {
                        let delnrm = content_mut(NLS).delnrm;
                        let mut resnrm;

                        let norm_fn = content_mut(NLS).norm_fn;
                        if let Some(f) = norm_fn {
                            let mut data = content_mut(NLS).norm_fn_data.take();
                            resnrm = ZERO;
                            retval = f(&delta, w, &mut resnrm, &mut data);
                            content_mut(NLS).norm_fn_data = data;
                            if retval != SUN_SUCCESS {
                                break;
                            }
                        } else {
                            resnrm = N_VWrmsNorm(&delta, w);
                        }

                        /* Norsett's switching metric compares the next residual
                        to the previous Newton update norm. */
                        content_mut(NLS).stiffr =
                            if delnrm > ZERO { resnrm / delnrm } else { ZERO };
                    }
                } /* end of Newton iteration loop */
            }
        }

        /* all errors go here */

        /* If there is a recoverable convergence failure and the
        Jacobian-related data appears not to be current, increment the
        convergence failure count, reset the initial correction to zero, and
        loop again with a call to lsetup in which jbad is TRUE. */
        if retval > 0 && !content_mut(NLS).jcur && LSetup.is_some() {
            content_mut(NLS).nconvfails += 1;
            callLSetup = SUNTRUE;
            jbad = SUNTRUE;
            N_VConst(ZERO, ycor);
            continue 'setup_loop;
        } else {
            break 'setup_loop;
        }
    } /* end of setup loop */

    /* increment number of convergence failures */
    content_mut(NLS).nconvfails += 1;

    /* all error returns exit here */
    retval
}

pub fn SUNNonlinSolFree_Newton(_NLS: &SUNNonlinearSolver) -> SUNErrCode {
    SUN_SUCCESS
}

pub fn SUNNonlinSolSetSysFn_Newton(
    NLS: &SUNNonlinearSolver,
    SysFn: Option<SUNNonlinSolSysFn>,
) -> SUNErrCode {
    if SysFn.is_none() {
        return SUN_ERR_ARG_CORRUPT;
    }
    content_mut(NLS).Sys = SysFn;
    SUN_SUCCESS
}

pub fn SUNNonlinSolSetLSetupFn_Newton(
    NLS: &SUNNonlinearSolver,
    LSetupFn: Option<SUNNonlinSolLSetupFn>,
) -> SUNErrCode {
    content_mut(NLS).LSetup = LSetupFn;
    SUN_SUCCESS
}

pub fn SUNNonlinSolSetLSolveFn_Newton(
    NLS: &SUNNonlinearSolver,
    LSolveFn: Option<SUNNonlinSolLSolveFn>,
) -> SUNErrCode {
    if LSolveFn.is_none() {
        return SUN_ERR_ARG_CORRUPT;
    }
    content_mut(NLS).LSolve = LSolveFn;
    SUN_SUCCESS
}

pub fn SUNNonlinSolSetConvTestFn_Newton(
    NLS: &SUNNonlinearSolver,
    CTestFn: Option<SUNNonlinSolConvTestFn>,
    ctest_data: Option<Box<dyn Any>>,
) -> SUNErrCode {
    if CTestFn.is_none() {
        return SUN_ERR_ARG_CORRUPT;
    }
    let mut content = content_mut(NLS);
    content.CTest = CTestFn;
    content.ctest_data = ctest_data;
    SUN_SUCCESS
}

pub fn SUNNonlinSolSetNormFn_Newton(
    NLS: &SUNNonlinearSolver,
    NormFn: Option<SUNNonlinSolNormFn>,
    norm_fn_data: Option<Box<dyn Any>>,
) -> SUNErrCode {
    let mut content = content_mut(NLS);
    content.norm_fn = NormFn;
    content.norm_fn_data = norm_fn_data;
    SUN_SUCCESS
}

pub fn SUNNonlinSolSetGetUpdateNormFn_Newton(
    NLS: &SUNNonlinearSolver,
    GetUpdateNormFn: Option<SUNNonlinSolGetUpdateNormFn>,
    getupdatenorm_data: Option<Box<dyn Any>>,
) -> SUNErrCode {
    let mut content = content_mut(NLS);
    content.getupdatenorm_fn = GetUpdateNormFn;
    content.getupdatenorm_data = getupdatenorm_data;
    SUN_SUCCESS
}

pub fn SUNNonlinSolSetMaxIters_Newton(NLS: &SUNNonlinearSolver, maxiters: i32) -> SUNErrCode {
    content_mut(NLS).maxiters = maxiters;
    SUN_SUCCESS
}

pub fn SUNNonlinSolSetComputeStiffnessRatio_Newton(
    NLS: &SUNNonlinearSolver,
    onoff: sunbooleantype,
) -> SUNErrCode {
    let mut content = content_mut(NLS);
    content.compute_stiffr = onoff;
    if !onoff {
        content.stiffr = 0.0;
    }
    SUN_SUCCESS
}

pub fn SUNNonlinSolGetNumIters_Newton(NLS: &SUNNonlinearSolver, niters: &mut i64) -> SUNErrCode {
    *niters = content_mut(NLS).niters;
    SUN_SUCCESS
}

pub fn SUNNonlinSolGetCurIter_Newton(NLS: &SUNNonlinearSolver, iter: &mut i32) -> SUNErrCode {
    *iter = content_mut(NLS).curiter;
    SUN_SUCCESS
}

pub fn SUNNonlinSolGetNumConvFails_Newton(
    NLS: &SUNNonlinearSolver,
    nconvfails: &mut i64,
) -> SUNErrCode {
    *nconvfails = content_mut(NLS).nconvfails;
    SUN_SUCCESS
}

pub fn SUNNonlinSolGetSysFn_Newton(
    NLS: &SUNNonlinearSolver,
    SysFn: &mut Option<SUNNonlinSolSysFn>,
) -> SUNErrCode {
    *SysFn = content_mut(NLS).Sys;
    SUN_SUCCESS
}

pub fn SUNNonlinSolGetStiffnessRatio_Newton(
    NLS: &SUNNonlinearSolver,
    stiffr: &mut sunrealtype,
) -> SUNErrCode {
    *stiffr = content_mut(NLS).stiffr;
    SUN_SUCCESS
}
