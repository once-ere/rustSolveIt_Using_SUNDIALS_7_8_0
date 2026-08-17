/*-----------------------------------------------------------------
 * Rust port of examples/arkode/C_serial/ark_analytic_nonlin.c
 * Programmer(s): Daniel R. Reynolds @ UMBC
 *---------------------------------------------------------------
 * Example problem:
 *
 * The following is a simple example problem with analytical
 * solution,
 *     dy/dt = (t+1)*exp(-y)
 * for t in the interval [0.0, 10.0], with initial condition: y=0.
 * This has analytical solution
 *      y(t) = log(0.5*t^2 + t + 1)
 *
 * This program solves the problem with the ERK method.
 * Output is printed every 1.0 units of time (10 total).
 * Run statistics (optional outputs) are printed at the end.
 *-----------------------------------------------------------------*/

#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

use arkode_rs::prelude::*;

use arkode_rs::sundials_futils::SUNFileClose;
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
    let reltol: sunrealtype = 1.0e-6; /* tolerances */
    let abstol: sunrealtype = 1.0e-10;

    /* general problem variables */
    let mut flag: i32; /* reusable error-checking flag */

    /* Create the SUNDIALS context object for this simulation */
    let mut ctx_opt: Option<SUNContext> = None;
    flag = SUNContext_Create(SUN_COMM_NULL, &mut ctx_opt);
    if check_flag(Some(flag), "SUNContext_Create", 1) != 0 {
        std::process::exit(1);
    }
    let ctx = ctx_opt.as_ref().expect("ctx").clone();

    /* Initial problem output */
    print!("\nAnalytical ODE test problem:\n");
    print!("   reltol = {}\n", fmt_e(reltol, 1));
    print!("   abstol = {}\n\n", fmt_e(abstol, 1));

    /* Initialize data structures */
    let y = N_VNew_Serial(NEQ, &ctx); /* Create serial vector for solution */
    if check_flag(y.as_ref().map(|_| 0), "N_VNew_Serial", 0) != 0 {
        std::process::exit(1);
    }
    let y = y.expect("y");
    NV_DATA_S(&y)[0] = 0.0; /* Specify initial condition */

    /* Call ERKStepCreate to initialize the ERK timestepper module and
    specify the right-hand side function in y'=f(t,y), the initial time
    T0, and the initial dependent variable vector y. */
    let mut arkode_mem_opt = ERKStepCreate(f, T0, &y, &ctx);
    if check_flag(arkode_mem_opt.as_ref().map(|_| 0), "ERKStepCreate", 0) != 0 {
        std::process::exit(1);
    }
    let arkode_mem = arkode_mem_opt.as_ref().expect("arkode_mem").clone();

    /* Specify tolerances */
    flag = ARKodeSStolerances(&arkode_mem, reltol, abstol);
    if check_flag(Some(flag), "ARKodeSStolerances", 1) != 0 {
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

    /* Print final statistics */
    print!("\nFinal Statistics:\n");
    /* C stores the return in `flag` and never reads it again */
    let _ = ARKodePrintAllStats(
        &arkode_mem,
        &SUNFile::Stdout,
        SUNOutputFormat::SUN_OUTPUTFORMAT_TABLE,
    );

    /* Print final statistics to a file in CSV format */
    let mut FID = SUNFile::fopen("ark_analytic_nonlin_stats.csv", "w");
    let _ = ARKodePrintAllStats(&arkode_mem, &FID, SUNOutputFormat::SUN_OUTPUTFORMAT_CSV);
    SUNFileClose(&mut FID);

    /* Clean up and return with successful completion */
    N_VDestroy(y); /* Free y vector */
    ARKodeFree(&mut arkode_mem_opt); /* Free integrator memory */
    SUNContext_Free(&mut ctx_opt); /* Free context */

    std::process::exit(0);
}

/*-------------------------------
 * Functions called by the solver
 *-------------------------------*/

/* f routine to compute the ODE RHS function f(t,y). */
fn f(t: sunrealtype, y: &N_Vector, ydot: &N_Vector, _user_data: &mut Option<Box<dyn Any>>) -> i32 {
    let u = NV_DATA_S(y)[0];
    NV_DATA_S(ydot)[0] = (t + 1.0) * SUNRexp(-u);
    0
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

/*---- end of file ----*/
