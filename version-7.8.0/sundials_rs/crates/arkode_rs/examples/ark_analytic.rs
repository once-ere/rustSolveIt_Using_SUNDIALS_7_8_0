/*-----------------------------------------------------------------
 * Rust port of examples/arkode/C_serial/ark_analytic.c
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
 * This program solves the problem with the DIRK method,
 * Newton iteration with the dense SUNLinearSolver, and a
 * user-supplied Jacobian routine.
 * Output is printed every 1.0 units of time (10 total).
 * Run statistics (optional outputs) are printed at the end.
 *-----------------------------------------------------------------*/

#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

use arkode_rs::prelude::*;

use std::any::Any;
use std::fs::File;
use std::io::Write;

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

    /* command-line arguments (C: argc, argv) */
    let argv: Vec<String> = std::env::args().collect();
    let argc: i32 = argv.len() as i32;

    /* general problem variables */
    let mut flag: i32; /* reusable error-checking flag */
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
    flag = SUNContext_Create(SUN_COMM_NULL, &mut ctx_opt);
    if check_flag(Some(flag), "SUNContext_Create", 1) != 0 {
        std::process::exit(1);
    }
    let ctx = ctx_opt.as_ref().expect("ctx").clone();

    /* Initial diagnostics output */
    print!("\nAnalytical ODE test problem:\n");
    print!("   lambda = {}\n\n", fmt_g(lambda, 6));

    /* Initialize data structures */
    let y = N_VNew_Serial(NEQ, &ctx); /* Create serial vector for solution */
    if check_flag(y.as_ref().map(|_| 0), "N_VNew_Serial", 0) != 0 {
        std::process::exit(1);
    }
    let y = y.expect("y");
    N_VConst(0.0, &y); /* Specify initial condition */

    /* Call ARKStepCreate to initialize the ARK timestepper module and
    specify the right-hand side function in y'=f(t,y), the initial time
    T0, and the initial dependent variable vector y.  Note: since this
    problem is fully implicit, we set f_E to NULL and f_I to f. */
    let mut arkode_mem_opt = ARKStepCreate(None, Some(f), T0, &y, &ctx);
    if check_flag(arkode_mem_opt.as_ref().map(|_| 0), "ARKStepCreate", 0) != 0 {
        std::process::exit(1);
    }
    let arkode_mem = arkode_mem_opt.as_ref().expect("arkode_mem").clone();

    /* Set routines */
    flag = ARKodeSetUserData(&arkode_mem, Some(Box::new(lambda))); /* Pass lambda to user functions */
    if check_flag(Some(flag), "ARKodeSetUserData", 1) != 0 {
        std::process::exit(1);
    }
    flag = ARKodeSStolerances(&arkode_mem, reltol, abstol); /* Specify tolerances */
    if check_flag(Some(flag), "ARKodeSStolerances", 1) != 0 {
        std::process::exit(1);
    }

    /* Initialize dense matrix data structure and solver */
    let A = SUNDenseMatrix(NEQ, NEQ, &ctx);
    if check_flag(A.as_ref().map(|_| 0), "SUNDenseMatrix", 0) != 0 {
        std::process::exit(1);
    }
    let A = A.expect("A");
    let LS = SUNLinSol_Dense(&y, &A, &ctx);
    if check_flag(LS.as_ref().map(|_| 0), "SUNLinSol_Dense", 0) != 0 {
        std::process::exit(1);
    }
    let LS = LS.expect("LS");

    /* Linear solver interface */
    flag = ARKodeSetLinearSolver(&arkode_mem, &LS, Some(&A)); /* Attach matrix and linear solver */
    if check_flag(Some(flag), "ARKodeSetLinearSolver", 1) != 0 {
        std::process::exit(1);
    }
    flag = ARKodeSetJacFn(&arkode_mem, Some(Jac)); /* Set Jacobian routine */
    if check_flag(Some(flag), "ARKodeSetJacFn", 1) != 0 {
        std::process::exit(1);
    }

    /* Specify linearly implicit RHS, with non-time-dependent Jacobian */
    flag = ARKodeSetLinear(&arkode_mem, 0);
    if check_flag(Some(flag), "ARKodeSetLinear", 1) != 0 {
        std::process::exit(1);
    }

    /* Override any current settings with command-line options */
    flag = ARKodeSetOptions(&arkode_mem, None, None, argc, &argv);
    if check_flag(Some(flag), "ARKodeSetOptions", 1) != 0 {
        std::process::exit(1);
    }

    /* Output current ARKODE options */
    flag = ARKodeWriteParameters(&arkode_mem, &SUNFile::Stdout);
    if check_flag(Some(flag), "ARKodeWriteParameters", 1) != 0 {
        std::process::exit(1);
    }

    /* Open output stream for results, output comment line */
    let mut UFID = File::create("solution.txt").expect("solution.txt");
    let _ = write!(UFID, "# t u\n");

    /* output initial condition to disk */
    {
        let u = NV_DATA_S(&y)[0];
        let _ = write!(UFID, " {} {}\n", fmt_e(T0, 16), fmt_e(u, 16));
    }

    /* Main time-stepping loop: calls ARKodeEvolve to perform the integration, then
    prints results.  Stops when the final time has been reached */
    let mut t: sunrealtype = T0;
    let mut tout: sunrealtype = T0 + dTout;
    print!("        t           u\n");
    print!("   ---------------------\n");
    while Tf - t > 1.0e-15 {
        flag = ARKodeEvolve(&arkode_mem, tout, &y, &mut t, ARK_NORMAL); /* call integrator */
        if check_flag(Some(flag), "ARKodeEvolve", 1) != 0 {
            break;
        }
        {
            let u = NV_DATA_S(&y)[0]; /* access/print solution */
            print!("  {}  {}\n", fmt_fw(t, 10, 6), fmt_fw(u, 10, 6));
            let _ = write!(UFID, " {} {}\n", fmt_e(t, 16), fmt_e(u, 16));
        }
        if flag >= 0 {
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
    drop(UFID);

    /* Get/print some final statistics on how the solve progressed */
    flag = ARKodeGetNumSteps(&arkode_mem, &mut nst);
    check_flag(Some(flag), "ARKodeGetNumSteps", 1);
    flag = ARKodeGetNumStepAttempts(&arkode_mem, &mut nst_a);
    check_flag(Some(flag), "ARKodeGetNumStepAttempts", 1);
    flag = ARKodeGetNumRhsEvals(&arkode_mem, 0, &mut nfe);
    check_flag(Some(flag), "ARKodeGetNumRhsEvals", 1);
    flag = ARKodeGetNumRhsEvals(&arkode_mem, 1, &mut nfi);
    check_flag(Some(flag), "ARKodeGetNumRhsEvals", 1);
    flag = ARKodeGetNumLinSolvSetups(&arkode_mem, &mut nsetups);
    check_flag(Some(flag), "ARKodeGetNumLinSolvSetups", 1);
    flag = ARKodeGetNumErrTestFails(&arkode_mem, &mut netf);
    check_flag(Some(flag), "ARKodeGetNumErrTestFails", 1);
    flag = ARKodeGetNumNonlinSolvIters(&arkode_mem, &mut nni);
    check_flag(Some(flag), "ARKodeGetNumNonlinSolvIters", 1);
    flag = ARKodeGetNumNonlinSolvConvFails(&arkode_mem, &mut ncfn);
    check_flag(Some(flag), "ARKodeGetNumNonlinSolvConvFails", 1);
    flag = ARKodeGetNumJacEvals(&arkode_mem, &mut nje);
    check_flag(Some(flag), "ARKodeGetNumJacEvals", 1);
    flag = ARKodeGetNumLinRhsEvals(&arkode_mem, &mut nfeLS);
    check_flag(Some(flag), "ARKodeGetNumLinRhsEvals", 1);

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
    let flag = check_ans(&y, t, reltol, abstol);

    /* Clean up and return */
    N_VDestroy(y); /* Free y vector */
    ARKodeFree(&mut arkode_mem_opt); /* Free integrator memory */
    SUNLinSolFree(Some(LS)); /* Free linear solver */
    SUNMatDestroy(A); /* Free A matrix */
    SUNContext_Free(&mut ctx_opt); /* Free context */

    std::process::exit(flag);
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

/* Jacobian routine to compute J(t,y) = df/dy. */
fn Jac(
    _t: sunrealtype,
    _y: &N_Vector,
    _fy: &N_Vector,
    J: &SUNMatrix,
    user_data: &mut Option<Box<dyn Any>>,
    _tmp1: &N_Vector,
    _tmp2: &N_Vector,
    _tmp3: &N_Vector,
) -> i32 {
    /* C casts user_data to sunrealtype* and reads rdata[0] */
    let lambda = *user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<sunrealtype>())
        .expect("user_data"); /* set shortcut for stiffness parameter */

    /* Fill in Jacobian of f: set the first entry of the data array to set the (0,0) entry */
    SUNDenseMatrix_Data(J)[0] = lambda;

    0 /* return with success */
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
fn check_flag(flagvalue: Option<i32>, funcname: &str, opt: i32) -> i32 {
    /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
    if opt == 0 && flagvalue.is_none() {
        eprint!(
            "\nSUNDIALS_ERROR: {}() failed - returned NULL pointer\n\n",
            funcname
        );
        return 1;
    }
    /* Check if flag < 0 */
    else if opt == 1 {
        let errflag = flagvalue.expect("errflag");
        if errflag < 0 {
            eprint!(
                "\nSUNDIALS_ERROR: {}() failed with flag = {}\n\n",
                funcname, errflag
            );
            return 1;
        }
    }
    /* Check if function returned NULL pointer - no memory allocated */
    else if opt == 2 && flagvalue.is_none() {
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
