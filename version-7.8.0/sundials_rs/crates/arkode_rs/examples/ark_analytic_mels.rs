/*-----------------------------------------------------------------
 * Rust port of examples/arkode/C_serial/ark_analytic_mels.c
 * Programmer(s): Daniel R. Reynolds @ UMBC
 *---------------------------------------------------------------
 * Example problem:
 *
 * The following is a simple example problem with analytical
 * solution,
 *    dy/dt = lambda*y + 1/(1+t^2) - lambda*atan(t)
 * for t in the interval [0.0, 10.0], with initial condition: y=0.
 *
 * The stiffness of the problem is directly proportional to the
 * value of "lambda".  The value of lambda should be negative to
 * result in a well-posed ODE; for values with magnitude larger
 * than 100 the problem becomes quite stiff.
 *
 * This program solves the problem with the DIRK method and a
 * custom 'matrix-embedded' SUNLinearSolver. Output is printed
 * every 1.0 units of time (10 total).  Run statistics (optional
 * outputs) are printed at the end.
 *-----------------------------------------------------------------*/

#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

use arkode_rs::prelude::*;

use std::any::Any;

/* Main Program */
fn main() {
    /* general problem parameters */
    let T0: sunrealtype = 0.0; /* initial time */
    let Tf: sunrealtype = 10.0; /* final time */
    let dTout: sunrealtype = 1.0; /* time between outputs */
    let NEQ: sunindextype = 1; /* number of dependent vars. */
    let reltol: sunrealtype = 1.0e-5; /* tolerances */
    let abstol: sunrealtype = 1.0e-10;
    let lambda: sunrealtype = -100.0; /* stiffness parameter */

    /* general problem variables */
    let mut retval: i32; /* reusable error-checking flag */
    let (
        mut nst,
        mut nst_a,
        mut nfe,
        mut nfi,
        mut nsetups,
        mut nje,
        mut nfeLS,
        mut nni,
        mut ncfn,
        mut netf,
    ) = (0i64, 0i64, 0i64, 0i64, 0i64, 0i64, 0i64, 0i64, 0i64, 0i64);

    /* Create the SUNDIALS context object for this simulation */
    let mut ctx_opt: Option<SUNContext> = None;
    retval = SUNContext_Create(SUN_COMM_NULL, &mut ctx_opt);
    if check_retval(Some(retval), "SUNContext_Create", 1) != 0 {
        std::process::exit(1);
    }
    let ctx = ctx_opt.as_ref().expect("ctx").clone();

    /* Initial diagnostics output */
    print!("\nAnalytical ODE test problem:\n");
    print!("   lambda = {}\n", fmt_g(lambda, 6));
    print!("   reltol = {}\n", fmt_e(reltol, 1));
    print!("   abstol = {}\n\n", fmt_e(abstol, 1));

    /* Initialize data structures */
    let y = N_VNew_Serial(NEQ, &ctx); /* Create serial vector for solution */
    if check_retval(y.as_ref().map(|_| 0), "N_VNew_Serial", 0) != 0 {
        std::process::exit(1);
    }
    let y = y.expect("y");
    N_VConst(0.0, &y); /* Specify initial condition */

    /* Call ARKStepCreate to initialize the ARK timestepper module and
    specify the right-hand side function in y'=f(t,y), the initial time
    T0, and the initial dependent variable vector y.  Note: since this
    problem is fully implicit, we set f_E to NULL and f_I to f. */
    let mut arkode_mem_opt = ARKStepCreate(None, Some(f), T0, &y, &ctx);
    if check_retval(arkode_mem_opt.as_ref().map(|_| 0), "ARKStepCreate", 0) != 0 {
        std::process::exit(1);
    }
    let arkode_mem = arkode_mem_opt.as_ref().expect("arkode_mem").clone();

    /* Set routines */
    retval = ARKodeSetUserData(&arkode_mem, Some(Box::new(lambda))); /* Pass lambda to user functions */
    if check_retval(Some(retval), "ARKodeSetUserData", 1) != 0 {
        std::process::exit(1);
    }
    retval = ARKodeSStolerances(&arkode_mem, reltol, abstol); /* Specify tolerances */
    if check_retval(Some(retval), "ARKodeSStolerances", 1) != 0 {
        std::process::exit(1);
    }

    /* Initialize custom matrix-embedded linear solver */
    let LS = MatrixEmbeddedLS(&arkode_mem, &ctx);
    if check_retval(LS.as_ref().map(|_| 0), "MatrixEmbeddedLS", 0) != 0 {
        std::process::exit(1);
    }
    let LS = LS.expect("LS");
    retval = ARKodeSetLinearSolver(&arkode_mem, &LS, None); /* Attach linear solver */
    if check_retval(Some(retval), "ARKodeSetLinearSolver", 1) != 0 {
        std::process::exit(1);
    }

    /* Specify linearly implicit RHS, with non-time-dependent Jacobian */
    retval = ARKodeSetLinear(&arkode_mem, 0);
    if check_retval(Some(retval), "ARKodeSetLinear", 1) != 0 {
        std::process::exit(1);
    }

    /* Main time-stepping loop: calls ARKodeEvolve to perform the integration, then
    prints results.  Stops when the final time has been reached. */
    let mut t: sunrealtype = T0;
    let mut tout: sunrealtype = T0 + dTout;
    print!("        t           u\n");
    print!("   ---------------------\n");
    while Tf - t > 1.0e-15 {
        retval = ARKodeEvolve(&arkode_mem, tout, &y, &mut t, ARK_NORMAL); /* call integrator */
        if check_retval(Some(retval), "ARKodeEvolve", 1) != 0 {
            break;
        }
        {
            let u = NV_DATA_S(&y)[0]; /* access/print solution */
            print!("  {}  {}\n", fmt_fw(t, 10, 6), fmt_fw(u, 10, 6));
        }
        if retval >= 0 {
            /* successful solve: update time */
            tout += dTout;
            tout = if tout > Tf { Tf } else { tout };
        } else {
            /* unsuccessful solve: break */
            eprint!("Solver failure, stopping integration\n");
            break;
        }
    }
    print!("   ---------------------\n");

    /* Get/print some final statistics on how the solve progressed */
    retval = ARKodeGetNumSteps(&arkode_mem, &mut nst);
    check_retval(Some(retval), "ARKodeGetNumSteps", 1);
    retval = ARKodeGetNumStepAttempts(&arkode_mem, &mut nst_a);
    check_retval(Some(retval), "ARKodeGetNumStepAttempts", 1);
    retval = ARKodeGetNumRhsEvals(&arkode_mem, 0, &mut nfe);
    check_retval(Some(retval), "ARKodeGetNumRhsEvals", 1);
    retval = ARKodeGetNumRhsEvals(&arkode_mem, 1, &mut nfi);
    check_retval(Some(retval), "ARKodeGetNumRhsEvals", 1);
    retval = ARKodeGetNumLinSolvSetups(&arkode_mem, &mut nsetups);
    check_retval(Some(retval), "ARKodeGetNumLinSolvSetups", 1);
    retval = ARKodeGetNumErrTestFails(&arkode_mem, &mut netf);
    check_retval(Some(retval), "ARKodeGetNumErrTestFails", 1);
    retval = ARKodeGetNumNonlinSolvIters(&arkode_mem, &mut nni);
    check_retval(Some(retval), "ARKodeGetNumNonlinSolvIters", 1);
    retval = ARKodeGetNumNonlinSolvConvFails(&arkode_mem, &mut ncfn);
    check_retval(Some(retval), "ARKodeGetNumNonlinSolvConvFails", 1);
    retval = ARKodeGetNumJacEvals(&arkode_mem, &mut nje);
    check_retval(Some(retval), "ARKodeGetNumJacEvals", 1);
    retval = ARKodeGetNumLinRhsEvals(&arkode_mem, &mut nfeLS);
    check_retval(Some(retval), "ARKodeGetNumLinRhsEvals", 1);

    print!("\nFinal Solver Statistics:\n");
    print!(
        "   Internal solver steps = {} (attempted = {})\n",
        nst, nst_a
    );
    print!("   Total RHS evals:  Fe = {},  Fi = {}\n", nfe, nfi);
    print!("   Total linear solver setups = {}\n", nsetups);
    print!(
        "   Total RHS evals for setting up the linear system = {}\n",
        nfeLS
    );
    print!("   Total number of Jacobian evaluations = {}\n", nje);
    print!("   Total number of Newton iterations = {}\n", nni);
    print!(
        "   Total number of linear solver convergence failures = {}\n",
        ncfn
    );
    print!("   Total number of error test failures = {}\n\n", netf);

    /* check the solution error */
    let retval = check_ans(&y, t, reltol, abstol);

    /* Clean up and return */
    N_VDestroy(y); /* Free y vector */
    ARKodeFree(&mut arkode_mem_opt); /* Free integrator memory */
    SUNLinSolFree(Some(LS)); /* Free linear solver */
    SUNContext_Free(&mut ctx_opt); /* Free the SUNContext */

    std::process::exit(retval);
}

/*-------------------------------
 * Functions called by the solver
 *-------------------------------*/

/* f routine to compute the ODE RHS function f(t,y). */
fn f(t: sunrealtype, y: &N_Vector, ydot: &N_Vector, user_data: &mut Option<Box<dyn Any>>) -> i32 {
    /* C casts user_data to sunrealtype* and reads rdata[0] */
    let lambda = *user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<sunrealtype>())
        .expect("user_data"); /* set shortcut for stiffness parameter */
    let u = NV_DATA_S(y)[0]; /* access current solution value */

    /* fill in the RHS function */
    NV_DATA_S(ydot)[0] = lambda * u + 1.0 / (1.0 + t * t) - lambda * t.sun_atan();

    0 /* return with success */
}

/*-------------------------------------
 * Custom matrix-embedded linear solver
 *-------------------------------------*/

/* constructor */
fn MatrixEmbeddedLS(arkode_mem: &ARKodeMem, ctx: &SUNContext) -> Option<SUNLinearSolver> {
    /* Create an empty linear solver */
    let LS = SUNLinSolNewEmpty(ctx)?;

    /* Attach operations */
    {
        let mut ops = LS.ops.borrow_mut();
        ops.gettype = Some(MatrixEmbeddedLSType);
        ops.solve = Some(MatrixEmbeddedLSSolve);
        ops.free = Some(MatrixEmbeddedLSFree);
    }

    /* Set content pointer to ARKODE memory */
    *LS.content.borrow_mut() = Box::new(arkode_mem.clone());

    /* Return solver */
    Some(LS)
}

/* type descriptor */
fn MatrixEmbeddedLSType(_S: &SUNLinearSolver) -> SUNLinearSolver_Type {
    SUNLINEARSOLVER_MATRIX_EMBEDDED
}

/* linear solve routine */
fn MatrixEmbeddedLSSolve(
    LS: &SUNLinearSolver,
    _A: Option<&SUNMatrix>,
    x: &N_Vector,
    b: &N_Vector,
    _tol: sunrealtype,
) -> i32 {
    /* temporary variables */
    let mut tcur: sunrealtype = 0.0;
    let mut gamma: sunrealtype = 0.0;
    let mut zpred: Option<N_Vector> = None;
    let mut z: Option<N_Vector> = None;
    let mut Fi: Option<N_Vector> = None;
    let mut sdata: Option<N_Vector> = None;
    let mut user_data: Option<Box<dyn Any>> = None;

    /* retrieve implicit system data from ARKODE (LS->content) */
    let arkode_mem = LS
        .content
        .borrow()
        .downcast_ref::<ARKodeMem>()
        .expect("MatrixEmbeddedLS content")
        .clone();
    let retval = ARKodeGetNonlinearSystemData(
        &arkode_mem,
        &mut tcur,
        &mut zpred,
        &mut z,
        &mut Fi,
        &mut gamma,
        &mut sdata,
        &mut user_data,
    );
    if check_retval(Some(retval), "ARKodeGetNonlinearSystemData", 1) != 0 {
        return -1;
    }

    /* extract stiffness parameter from user_data */
    let lambda = *user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<sunrealtype>())
        .expect("user_data");

    /* perform linear solve: (1-gamma*lambda)*x = b */
    let bval = NV_DATA_S(b)[0];
    NV_DATA_S(x)[0] = bval / (1.0 - gamma * lambda);

    /* ARKodeGetNonlinearSystemData SWAPPED the user_data box out of the
    integrator; a second call swaps it back in (leaving the local as
    None), so the integrator owns the box again before we return. */
    let retval = ARKodeGetNonlinearSystemData(
        &arkode_mem,
        &mut tcur,
        &mut zpred,
        &mut z,
        &mut Fi,
        &mut gamma,
        &mut sdata,
        &mut user_data,
    );
    if check_retval(Some(retval), "ARKodeGetNonlinearSystemData", 1) != 0 {
        return -1;
    }
    debug_assert!(user_data.is_none());

    /* return with success */
    SUN_SUCCESS
}

/* destructor */
fn MatrixEmbeddedLSFree(LS: &SUNLinearSolver) -> SUNErrCode {
    /* LS->content = NULL: drop the stored ARKodeMem handle; the caller
    (SUNLinSolFree) then drops the handle itself (SUNLinSolFreeEmpty) */
    *LS.content.borrow_mut() = Box::new(());
    SUN_SUCCESS
}

/*-------------------------------
 * Private helper functions
 *-------------------------------*/

/* Check function return value...
    opt == 0 means SUNDIALS function allocates memory so check if
             returned NULL pointer (represented as `None`)
    opt == 1 means SUNDIALS function returns a flag so check if
             flag >= 0
    opt == 2 means function allocates memory so check if returned
             NULL pointer
*/
fn check_retval(returnvalue: Option<i32>, funcname: &str, opt: i32) -> i32 {
    /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
    if opt == 0 && returnvalue.is_none() {
        eprint!(
            "\nSUNDIALS_ERROR: {}() failed - returned NULL pointer\n\n",
            funcname
        );
        return 1;
    }
    /* Check if flag < 0 */
    else if opt == 1 {
        let retval = returnvalue.expect("retval");
        if retval < 0 {
            eprint!(
                "\nSUNDIALS_ERROR: {}() failed with flag = {}\n\n",
                funcname, retval
            );
            return 1;
        }
    }
    /* Check if function returned NULL pointer - no memory allocated */
    else if opt == 2 && returnvalue.is_none() {
        eprint!(
            "\nMEMORY_ERROR: {}() failed - returned NULL pointer\n\n",
            funcname
        );
        return 1;
    }

    0
}

/* check the computed solution */
fn check_ans(y: &N_Vector, t: sunrealtype, rtol: sunrealtype, atol: sunrealtype) -> i32 {
    /* compute solution error */
    let ans = t.sun_atan();
    let ewt = 1.0 / (rtol * SUNRabs(ans) + atol);
    let err = ewt * SUNRabs(NV_DATA_S(y)[0] - ans);

    /* The local errors accumulate from step to step so that the global error is
     * not quite within the local error tolerances. This factor accounts for
     * this. */
    let global_bound: sunrealtype = 1.5;

    /* is the solution within the tolerances? */
    let passfail = if err < global_bound { 0 } else { 1 };

    if passfail != 0 {
        print!("\nSUNDIALS_WARNING: check_ans error={}\n\n", fmt_g(err, 6));
    }

    passfail
}

/*---- end of file ----*/
