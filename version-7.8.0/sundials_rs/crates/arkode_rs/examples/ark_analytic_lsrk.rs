/*-----------------------------------------------------------------
 * Rust port of examples/arkode/C_serial/ark_analytic_lsrk.c
 * Programmer(s): Mustafa Aggul @ UMBC
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
 * This program solves the problem with the LSRK method.
 * Output is printed every 1.0 units of time (10 total).
 * Run statistics (optional outputs) are printed at the end.
 *-----------------------------------------------------------------*/

#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

use arkode_rs::prelude::*;

use arkode_rs::sundials_futils::SUNFileClose;
use std::any::Any;

/* Main Program */
fn main() {
    /* general problem parameters */
    let T0: sunrealtype = 0.0; /* initial time */
    let Tf: sunrealtype = 10.0; /* final time */
    let dTout: sunrealtype = 1.0; /* time between outputs */
    let NEQ: sunindextype = 1; /* number of dependent vars. */

    let reltol: sunrealtype = 1.0e-8; /* tolerances */
    let abstol: sunrealtype = 1.0e-8;
    let lambda: sunrealtype = -1000000.0; /* stiffness parameter */

    /* general problem variables */
    let mut flag: i32; /* reusable error-checking flag */

    /* Create the SUNDIALS context object for this simulation */
    let mut sunctx: Option<SUNContext> = None;
    flag = SUNContext_Create(SUN_COMM_NULL, &mut sunctx);
    if check_flag(Some(flag), "SUNContext_Create", 1) != 0 {
        std::process::exit(1);
    }
    let ctx = sunctx.as_ref().expect("sunctx").clone();

    /* Initial diagnostics output */
    print!("\nAnalytical ODE test problem:\n");
    print!("    lambda = {}\n", fmt_g(lambda, 6));
    print!("   reltol = {}\n", fmt_e(reltol, 1));
    print!("   abstol = {}\n\n", fmt_e(abstol, 1));

    /* Initialize data structures */
    let y = N_VNew_Serial(NEQ, &ctx); /* Create serial vector for solution */
    if check_flag(y.as_ref().map(|_| 0), "N_VNew_Serial", 0) != 0 {
        std::process::exit(1);
    }
    let y = y.expect("y");
    N_VConst(0.0, &y); /* Specify initial condition */

    /* Call LSRKStepCreateSTS to initialize the ARK timestepper module and
    specify the right-hand side function in y'=f(t,y), the initial time
    T0, and the initial dependent variable vector y. */
    let mut arkode_mem_opt = LSRKStepCreateSTS(f, T0, &y, &ctx);
    if check_flag(arkode_mem_opt.as_ref().map(|_| 0), "LSRKStepCreateSTS", 0) != 0 {
        std::process::exit(1);
    }
    let arkode_mem = arkode_mem_opt.as_ref().expect("arkode_mem").clone();

    /* Set routines */
    flag = ARKodeSetUserData(&arkode_mem, Some(Box::new(lambda))); /* Pass lambda to user functions */
    if check_flag(Some(flag), "ARKodeSetUserData", 1) != 0 {
        std::process::exit(1);
    }

    /* Specify tolerances */
    flag = ARKodeSStolerances(&arkode_mem, reltol, abstol);
    if check_flag(Some(flag), "ARKodeSStolerances", 1) != 0 {
        std::process::exit(1);
    }

    /* Specify user provided spectral radius */
    flag = LSRKStepSetDomEigFn(&arkode_mem, Some(dom_eig));
    if check_flag(Some(flag), "LSRKStepSetDomEigFn", 1) != 0 {
        std::process::exit(1);
    }

    /* Specify after how many successful steps dom_eig is recomputed
    Note that nsteps = 0 refers to constant dominant eigenvalue */
    flag = LSRKStepSetDomEigFrequency(&arkode_mem, 0);
    if check_flag(Some(flag), "LSRKStepSetDomEigFrequency", 1) != 0 {
        std::process::exit(1);
    }

    /* Specify max number of stages allowed */
    flag = LSRKStepSetMaxNumStages(&arkode_mem, 200);
    if check_flag(Some(flag), "LSRKStepSetMaxNumStages", 1) != 0 {
        std::process::exit(1);
    }

    /* Specify max number of steps allowed */
    flag = ARKodeSetMaxNumSteps(&arkode_mem, 1000);
    if check_flag(Some(flag), "ARKodeSetMaxNumSteps", 1) != 0 {
        std::process::exit(1);
    }

    /* Specify safety factor for user provided dom_eig */
    flag = LSRKStepSetDomEigSafetyFactor(&arkode_mem, 1.01);
    if check_flag(Some(flag), "LSRKStepSetDomEigSafetyFactor", 1) != 0 {
        std::process::exit(1);
    }

    /* Specify the Runge--Kutta--Legendre LSRK method */
    flag = LSRKStepSetSTSMethod(&arkode_mem, ARKODE_LSRK_RKL_2);
    if check_flag(Some(flag), "LSRKStepSetSTSMethod", 1) != 0 {
        std::process::exit(1);
    }

    /* Open output stream for results, output comment line */
    let mut UFID = SUNFile::fopen("solution.txt", "w");
    UFID.write_str("# t u\n");

    /* output initial condition to disk */
    UFID.write_str(&format!(
        " {} {}\n",
        fmt_e(T0, 16),
        fmt_e(NV_DATA_S(&y)[0], 16)
    ));

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
        /* access/print solution */
        print!(
            "  {}  {}\n",
            fmt_fw(t, 10, 6),
            fmt_fw(NV_DATA_S(&y)[0], 10, 6)
        );
        UFID.write_str(&format!(
            " {} {}\n",
            fmt_e(t, 16),
            fmt_e(NV_DATA_S(&y)[0], 16)
        ));
        if flag < 0 {
            /* unsuccessful solve: break */
            eprint!("Solver failure, stopping integration\n");
            break;
        } else {
            /* successful solve: update time */
            tout += dTout;
            tout = if tout > Tf { Tf } else { tout };
        }
    }
    print!("   ---------------------\n");
    SUNFileClose(&mut UFID);

    /* Print final statistics */
    print!("\nFinal Statistics:\n");
    let _ = ARKodePrintAllStats(
        &arkode_mem,
        &SUNFile::Stdout,
        SUNOutputFormat::SUN_OUTPUTFORMAT_TABLE,
    );

    /* Print final statistics to a file in CSV format */
    let mut FID = SUNFile::fopen("ark_analytic_nonlin_stats.csv", "w");
    let _ = ARKodePrintAllStats(&arkode_mem, &FID, SUNOutputFormat::SUN_OUTPUTFORMAT_CSV);
    SUNFileClose(&mut FID);

    /* check the solution error */
    let _ = check_ans(&y, t, reltol, abstol);
    flag = compute_error(&y, t);

    /* Clean up and return */
    N_VDestroy(y); /* Free y vector */
    ARKodeFree(&mut arkode_mem_opt); /* Free integrator memory */
    SUNContext_Free(&mut sunctx); /* Free context */

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

/* dom_eig routine to estimate the dominated eigenvalue */
fn dom_eig(
    _t: sunrealtype,
    _y: &N_Vector,
    _fn_: &N_Vector,
    lambdaR: &mut sunrealtype,
    lambdaI: &mut sunrealtype,
    user_data: &mut Option<Box<dyn Any>>,
    _temp1: &N_Vector,
    _temp2: &N_Vector,
    _temp3: &N_Vector,
) -> i32 {
    /* C casts user_data to sunrealtype* and reads rdata[0] */
    let lambda = *user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<sunrealtype>())
        .expect("user_data"); /* set shortcut for stiffness parameter */
    *lambdaR = lambda;
    *lambdaI = 0.0;
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
        let errflag = flagvalue.expect("flagvalue");
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

    /* is the solution within the tolerances? */
    let passfail = if err < 1.0 { 0 } else { 1 };

    if passfail != 0 {
        print!("\nSUNDIALS_WARNING: check_ans error={}\n\n", fmt_g(err, 6));
    }

    passfail
}

/* check the error */
fn compute_error(y: &N_Vector, t: sunrealtype) -> i32 {
    /* compute solution error */
    let ans = t.sun_atan();
    let err = SUNRabs(NV_DATA_S(y)[0] - ans);

    print!("\nACCURACY at the final time   = {}\n", fmt_g(err, 6));
    0
}

/*---- end of file ----*/
