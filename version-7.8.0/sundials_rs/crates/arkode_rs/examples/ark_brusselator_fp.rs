/*-----------------------------------------------------------------
 * Programmer(s): Daniel R. Reynolds @ UMBC
 *---------------------------------------------------------------
 * Port of `examples/arkode/C_serial/ark_brusselator_fp.c`.
 *---------------------------------------------------------------
 * Example problem:
 *
 * The following test simulates a brusselator problem from chemical
 * kinetics.  This is an ODE system with 3 components, Y = [u,v,w],
 * satisfying the equations,
 *    du/dt = a - (w+1)*u + v*u^2
 *    dv/dt = w*u - v*u^2
 *    dw/dt = (b-w)/ep - w*u
 * for t in the interval [0.0, 10.0], with initial conditions
 * Y0 = [u0,v0,w0].
 *
 * We have 3 different testing scenarios:
 *
 * Test 1:  u0=3.9,  v0=1.1,  w0=2.8,  a=1.2,  b=2.5,  ep=1.0e-5
 * Test 2:  u0=1.2,  v0=3.1,  w0=3,  a=1,  b=3.5,  ep=5.0e-6
 * Test 3:  u0=3,  v0=3,  w0=3.5,  a=0.5,  b=3,  ep=5.0e-4
 *
 * This file is hard-coded to use test 3.
 *
 * This program solves the problem with the ARK method, using an
 * accelerated fixed-point iteration for the nonlinear solver.
 *
 * 100 outputs are printed at equal intervals, and run statistics
 * are printed at the end.
 *-----------------------------------------------------------------*/

#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

use arkode_rs::prelude::*;
use arkode_rs::sundials_logger::{
    SUNLogger, SUNLogger_Create, SUNLogger_Destroy, SUNLogger_SetInfoFilename,
};

use std::any::Any;
use std::fs::File;
use std::io::Write;

/* accessor helpers for the serial N_Vector (C macro `NV_Ith_S(v,i)`) */

fn NV_Ith_S(v: &N_Vector, i: usize) -> sunrealtype {
    N_VGetArrayPointer(v).expect("N_VGetArrayPointer")[i]
}

fn NV_Ith_S_set(v: &N_Vector, i: usize, x: sunrealtype) {
    N_VGetArrayPointer(v).expect("N_VGetArrayPointer")[i] = x;
}

/* Main Program */
fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let argc: usize = argv.len();

    /* general problem parameters */
    let T0: sunrealtype = 0.0; /* initial time */
    let Tf: sunrealtype = 10.0; /* final time */
    let dTout: sunrealtype = 1.0; /* time between outputs */
    let NEQ: sunindextype = 3; /* number of dependent vars. */
    let Nt: i32 = (Tf / dTout).ceil() as i32; /* number of output times */
    let test: i32 = 3; /* test problem to run */
    let reltol: sunrealtype = 1.0e-6; /* tolerances */
    let abstol: sunrealtype = 1.0e-10;
    let fp_m: i32 = 3; /* dimension of acceleration subspace */
    let maxcor: i32 = 10; /* maximum # of nonlinear iterations/step */
    let a: sunrealtype;
    let b: sunrealtype;
    let ep: sunrealtype;
    let u0: sunrealtype;
    let v0: sunrealtype;
    let w0: sunrealtype;
    let mut rdata: [sunrealtype; 3] = [0.0; 3];
    let mut monitor: i32 = 0; /* turn on/off monitoring */

    /* general problem variables */
    let info_fname: &str = "ark_brusselator_fp-info.txt";
    let mut flag: i32; /* reusable error-checking flag */
    let mut t: sunrealtype;
    let mut tout: sunrealtype;
    let mut nst: i64 = 0;
    let mut nst_a: i64 = 0;
    let mut nfe: i64 = 0;
    let mut nfi: i64 = 0;
    let mut nni: i64 = 0;
    let mut ncfn: i64 = 0;
    let mut netf: i64 = 0;

    /* read inputs */
    if argc == 2 {
        monitor = atoi(&argv[1]);
    }

    /* create SUNDIALS context and a logger which will record
    nonlinear solver info (e.g., residual) amongst other things. */

    let mut ctx_opt: Option<SUNContext> = None;
    flag = SUNContext_Create(SUN_COMM_NULL, &mut ctx_opt);
    if check_flag(flag, "SUNContext_Create") != 0 {
        std::process::exit(1);
    }
    let ctx = ctx_opt.clone().expect("SUNContext_Create");

    let mut logger_opt: Option<SUNLogger> = None;
    flag = SUNLogger_Create(SUN_COMM_NULL, 0, &mut logger_opt);
    if check_flag(flag, "SUNLogger_Create") != 0 {
        std::process::exit(1);
    }
    let logger = logger_opt.clone().expect("SUNLogger_Create");

    flag = SUNLogger_SetInfoFilename(
        &logger,
        if monitor != 0 { Some(info_fname) } else { None },
    );
    if check_flag(flag, "SUNLogger_SetInfoFilename") != 0 {
        std::process::exit(1);
    }

    flag = SUNContext_SetLogger(&ctx, Some(logger.clone()));
    if check_flag(flag, "SUNContext_SetLogger") != 0 {
        std::process::exit(1);
    }

    /* set up the test problem according to the desired test */
    if test == 1 {
        u0 = 3.9;
        v0 = 1.1;
        w0 = 2.8;
        a = 1.2;
        b = 2.5;
        ep = 1.0e-5;
    } else if test == 3 {
        u0 = 3.0;
        v0 = 3.0;
        w0 = 3.5;
        a = 0.5;
        b = 3.0;
        ep = 5.0e-4;
    } else {
        u0 = 1.2;
        v0 = 3.1;
        w0 = 3.0;
        a = 1.0;
        b = 3.5;
        ep = 5.0e-6;
    }

    /* Initial problem output */
    print!("\nBrusselator ODE test problem, fixed-point solver:\n");
    print!(
        "    initial conditions:  u0 = {},  v0 = {},  w0 = {}\n",
        fmt_g(u0, 6),
        fmt_g(v0, 6),
        fmt_g(w0, 6)
    );
    print!(
        "    problem parameters:  a = {},  b = {},  ep = {}\n",
        fmt_g(a, 6),
        fmt_g(b, 6),
        fmt_g(ep, 6)
    );
    print!(
        "    reltol = {},  abstol = {}\n\n",
        fmt_e(reltol, 1),
        fmt_e(abstol, 1)
    );

    /* Initialize data structures */
    rdata[0] = a; /* set user data  */
    rdata[1] = b;
    rdata[2] = ep;
    let y = N_VNew_Serial(NEQ, &ctx); /* Create serial vector for solution */
    if check_flag_ptr(&y, "N_VNew_Serial", 0) != 0 {
        std::process::exit(1);
    }
    let y = y.expect("N_VNew_Serial");
    NV_Ith_S_set(&y, 0, u0); /* Set initial conditions */
    NV_Ith_S_set(&y, 1, v0);
    NV_Ith_S_set(&y, 2, w0);

    /* Call ARKStepCreate to initialize the ARK timestepper module and
    specify the right-hand side functions in y'=fe(t,y)+fi(t,y),
    the initial time T0, and the initial dependent variable vector y. */
    let arkode_mem = ARKStepCreate(Some(fe), Some(fi), T0, &y, &ctx);
    if check_flag_ptr(&arkode_mem, "ARKStepCreate", 0) != 0 {
        std::process::exit(1);
    }
    let mut arkode_mem_opt = arkode_mem;
    let arkode_mem = arkode_mem_opt.clone().expect("ARKStepCreate");

    /* Initialize fixed-point nonlinear solver and attach to ARKODE */
    let NLS = SUNNonlinSol_FixedPoint(&y, fp_m, &ctx);
    if check_flag_ptr(&NLS, "SUNNonlinSol_FixedPoint", 0) != 0 {
        std::process::exit(1);
    }
    let NLS = NLS.expect("SUNNonlinSol_FixedPoint");

    flag = ARKodeSetNonlinearSolver(&arkode_mem, &NLS);
    if check_flag(flag, "ARKodeSetNonlinearSolver") != 0 {
        std::process::exit(1);
    }

    /* Set routines */
    flag = ARKodeSetUserData(&arkode_mem, Some(Box::new(rdata))); /* Pass rdata to user functions */
    if check_flag(flag, "ARKodeSetUserData") != 0 {
        std::process::exit(1);
    }

    flag = ARKodeSStolerances(&arkode_mem, reltol, abstol); /* Specify tolerances */
    if check_flag(flag, "ARKodeSStolerances") != 0 {
        std::process::exit(1);
    }

    flag = ARKodeSetMaxNonlinIters(&arkode_mem, maxcor); /* Increase default iterations */
    if check_flag(flag, "ARKodeSetMaxNonlinIters") != 0 {
        std::process::exit(1);
    }

    flag = ARKodeSetAutonomous(&arkode_mem, SUNTRUE);
    if check_flag(flag, "ARKodeSetAutonomous") != 0 {
        std::process::exit(1);
    }

    /* Open output stream for results, output comment line */
    let mut UFID = File::create("solution.txt").expect("output file");
    let _ = write!(UFID, "# t u v w\n");

    /* output initial condition to disk */
    let _ = write!(
        UFID,
        " {} {} {} {}\n",
        fmt_e(T0, 16),
        fmt_e(NV_Ith_S(&y, 0), 16),
        fmt_e(NV_Ith_S(&y, 1), 16),
        fmt_e(NV_Ith_S(&y, 2), 16)
    );

    /* Main time-stepping loop: calls ARKodeEvolve to perform the integration,
    then prints results.  Stops when the final time has been reached */
    t = T0;
    tout = T0 + dTout;
    print!("        t           u           v           w\n");
    print!("   ----------------------------------------------\n");
    for _iout in 0..Nt {
        flag = ARKodeEvolve(&arkode_mem, tout, &y, &mut t, ARK_NORMAL); /* call integrator */
        if check_flag(flag, "ARKodeEvolve") != 0 {
            break;
        }
        print!(
            "  {}  {}  {}  {}\n", /* access/print solution */
            fmt_fw(t, 10, 6),
            fmt_fw(NV_Ith_S(&y, 0), 10, 6),
            fmt_fw(NV_Ith_S(&y, 1), 10, 6),
            fmt_fw(NV_Ith_S(&y, 2), 10, 6)
        );
        let _ = write!(
            UFID,
            " {} {} {} {}\n",
            fmt_e(t, 16),
            fmt_e(NV_Ith_S(&y, 0), 16),
            fmt_e(NV_Ith_S(&y, 1), 16),
            fmt_e(NV_Ith_S(&y, 2), 16)
        );
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
    print!("   ----------------------------------------------\n");
    drop(UFID);

    /* Print some final statistics */
    flag = ARKodeGetNumSteps(&arkode_mem, &mut nst);
    check_flag(flag, "ARKodeGetNumSteps");
    flag = ARKodeGetNumStepAttempts(&arkode_mem, &mut nst_a);
    check_flag(flag, "ARKodeGetNumStepAttempts");
    flag = ARKodeGetNumRhsEvals(&arkode_mem, 0, &mut nfe);
    check_flag(flag, "ARKodeGetNumRhsEvals");
    flag = ARKodeGetNumRhsEvals(&arkode_mem, 1, &mut nfi);
    check_flag(flag, "ARKodeGetNumRhsEvals");
    flag = ARKodeGetNumErrTestFails(&arkode_mem, &mut netf);
    check_flag(flag, "ARKodeGetNumErrTestFails");
    flag = ARKodeGetNumNonlinSolvIters(&arkode_mem, &mut nni);
    check_flag(flag, "ARKodeGetNumNonlinSolvIters");
    flag = ARKodeGetNumNonlinSolvConvFails(&arkode_mem, &mut ncfn);
    check_flag(flag, "ARKodeGetNumNonlinSolvConvFails");

    print!("\nFinal Solver Statistics:\n");
    print!(
        "   Internal solver steps = {} (attempted = {})\n",
        nst, nst_a
    );
    print!("   Total RHS evals:  Fe = {},  Fi = {}\n", nfe, nfi);
    print!("   Total number of fixed-point iterations = {}\n", nni);
    print!(
        "   Total number of nonlinear solver convergence failures = {}\n",
        ncfn
    );
    print!("   Total number of error test failures = {}\n\n", netf);

    /* Clean up and return with successful completion */
    N_VDestroy(y);
    ARKodeFree(&mut arkode_mem_opt);
    let _ = SUNNonlinSolFree(Some(NLS));
    let _ = SUNLogger_Destroy(&mut logger_opt);
    let _ = SUNContext_Free(&mut ctx_opt);
}

/*-------------------------------
 * Functions called by the solver
 *-------------------------------*/

/* fi routine to compute the implicit portion of the ODE RHS. */
fn fi(
    _t: sunrealtype,
    y: &N_Vector,
    ydot: &N_Vector,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    let rdata = user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<[sunrealtype; 3]>())
        .expect("user_data is [sunrealtype; 3]"); /* cast user_data to sunrealtype */
    let b = rdata[1]; /* access data entries */
    let ep = rdata[2];
    let w = NV_Ith_S(y, 2); /* access solution values */

    /* fill in the RHS function */
    NV_Ith_S_set(ydot, 0, 0.0);
    NV_Ith_S_set(ydot, 1, 0.0);
    NV_Ith_S_set(ydot, 2, (b - w) / ep);

    0 /* Return with success */
}

/* fe routine to compute the explicit portion of the ODE RHS. */
fn fe(
    _t: sunrealtype,
    y: &N_Vector,
    ydot: &N_Vector,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    let rdata = user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<[sunrealtype; 3]>())
        .expect("user_data is [sunrealtype; 3]"); /* cast user_data to sunrealtype */
    let a = rdata[0]; /* access data entries */
    let u = NV_Ith_S(y, 0); /* access solution values */
    let v = NV_Ith_S(y, 1);
    let w = NV_Ith_S(y, 2);

    /* fill in the RHS function */
    NV_Ith_S_set(ydot, 0, a - (w + 1.0) * u + v * u * u);
    NV_Ith_S_set(ydot, 1, w * u - v * u * u);
    NV_Ith_S_set(ydot, 2, -w * u);

    0 /* Return with success */
}

/*-------------------------------
 * Private helper functions
 *-------------------------------*/

/* Check function return value...
    opt == 0 means SUNDIALS function allocates memory so check if
             returned NULL pointer
    opt == 1 means SUNDIALS function returns a flag so check if
             flag >= 0
    opt == 2 means function allocates memory so check if returned
             NULL pointer

   The C `check_flag(void*, name, opt)` polymorphism splits into two typed
   helpers with identical message text. */

fn check_flag_ptr<T>(flagvalue: &Option<T>, funcname: &str, opt: i32) -> i32 {
    if flagvalue.is_none() {
        if opt == 0 {
            /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
            eprint!(
                "\nSUNDIALS_ERROR: {}() failed - returned NULL pointer\n\n",
                funcname
            );
        } else {
            /* Check if function returned NULL pointer - no memory allocated */
            eprint!(
                "\nMEMORY_ERROR: {}() failed - returned NULL pointer\n\n",
                funcname
            );
        }
        return 1;
    }
    0
}

fn check_flag(flagvalue: i32, funcname: &str) -> i32 {
    /* Check if flag < 0 */
    if flagvalue < 0 {
        eprint!(
            "\nSUNDIALS_ERROR: {}() failed with flag = {}\n\n",
            funcname, flagvalue
        );
        return 1;
    }
    0
}

/*---- end of file ----*/
