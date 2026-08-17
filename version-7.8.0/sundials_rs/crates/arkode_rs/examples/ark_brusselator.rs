/*-----------------------------------------------------------------
 * Programmer(s): Daniel R. Reynolds @ UMBC
 *---------------------------------------------------------------
 * Port of `examples/arkode/C_serial/ark_brusselator.c`.
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
 * This file is hard-coded to use test 2.
 *
 * This program solves the problem with the DIRK method, using a
 * Newton iteration with the SUNDENSE dense linear solver, and a
 * user-supplied Jacobian routine.
 *
 * 100 outputs are printed at equal intervals, and run statistics
 * are printed at the end.
 *-----------------------------------------------------------------*/

#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

use arkode_rs::prelude::*;

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
    /* general problem parameters */
    let T0: sunrealtype = 0.0; /* initial time */
    let Tf: sunrealtype = 10.0; /* final time */
    let dTout: sunrealtype = 1.0; /* time between outputs */
    let NEQ: sunindextype = 3; /* number of dependent vars. */
    let Nt: i32 = (Tf / dTout).ceil() as i32; /* number of output times */
    let test: i32 = 2; /* test problem to run */
    let reltol: sunrealtype = 1.0e-6; /* tolerances */
    let abstol: sunrealtype = 1.0e-10;
    let a: sunrealtype;
    let b: sunrealtype;
    let ep: sunrealtype;
    let u0: sunrealtype;
    let v0: sunrealtype;
    let w0: sunrealtype;

    /* general problem variables */
    let mut flag: i32; /* reusable error-checking flag */
    let mut rdata: [sunrealtype; 3] = [0.0; 3];
    let mut t: sunrealtype;
    let mut tout: sunrealtype;
    let mut nst: i64 = 0;
    let mut nst_a: i64 = 0;
    let mut nfe: i64 = 0;
    let mut nfi: i64 = 0;
    let mut nsetups: i64 = 0;
    let mut nje: i64 = 0;
    let mut nfeLS: i64 = 0;
    let mut nni: i64 = 0;
    let mut nnf: i64 = 0;
    let mut ncfn: i64 = 0;
    let mut netf: i64 = 0;

    /* Create the SUNDIALS context object for this simulation */
    let mut ctx_opt: Option<SUNContext> = None;
    flag = SUNContext_Create(SUN_COMM_NULL, &mut ctx_opt);
    if check_flag(flag, "SUNContext_Create") != 0 {
        std::process::exit(1);
    }
    let ctx = ctx_opt.clone().expect("SUNContext_Create");

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
    print!("\nBrusselator ODE test problem:\n");
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
    specify the right-hand side function in y'=f(t,y), the initial time
    T0, and the initial dependent variable vector y.  Note: since this
    problem is fully implicit, we set f_E to NULL and f_I to f. */
    let arkode_mem = ARKStepCreate(None, Some(f), T0, &y, &ctx);
    if check_flag_ptr(&arkode_mem, "ARKStepCreate", 0) != 0 {
        std::process::exit(1);
    }
    let mut arkode_mem_opt = arkode_mem;
    let arkode_mem = arkode_mem_opt.clone().expect("ARKStepCreate");

    /* Set routines */
    flag = ARKodeSetUserData(&arkode_mem, Some(Box::new(rdata))); /* Pass rdata to user functions */
    if check_flag(flag, "ARKodeSetUserData") != 0 {
        std::process::exit(1);
    }

    flag = ARKodeSStolerances(&arkode_mem, reltol, abstol); /* Specify tolerances */
    if check_flag(flag, "ARKodeSStolerances") != 0 {
        std::process::exit(1);
    }

    flag = ARKodeSetInterpolantType(&arkode_mem, ARK_INTERP_LAGRANGE); /* Specify stiff interpolant */
    if check_flag(flag, "ARKodeSetInterpolantType") != 0 {
        std::process::exit(1);
    }

    flag = ARKodeSetDeduceImplicitRhs(&arkode_mem, true); /* Avoid eval of f after stage */
    if check_flag(flag, "ARKodeSetDeduceImplicitRhs") != 0 {
        std::process::exit(1);
    }

    /* Initialize dense matrix data structure and solver */
    let A = SUNDenseMatrix(NEQ, NEQ, &ctx);
    if check_flag_ptr(&A, "SUNDenseMatrix", 0) != 0 {
        std::process::exit(1);
    }
    let A = A.expect("SUNDenseMatrix");

    let LS = SUNLinSol_Dense(&y, &A, &ctx);
    if check_flag_ptr(&LS, "SUNLinSol_Dense", 0) != 0 {
        std::process::exit(1);
    }
    let LS = LS.expect("SUNLinSol_Dense");

    /* Linear solver interface */
    flag = ARKodeSetLinearSolver(&arkode_mem, &LS, Some(&A)); /* Attach matrix and linear solver */
    if check_flag(flag, "ARKodeSetLinearSolver") != 0 {
        std::process::exit(1);
    }

    flag = ARKodeSetJacFn(&arkode_mem, Some(Jac)); /* Set Jacobian routine */
    if check_flag(flag, "ARKodeSetJacFn") != 0 {
        std::process::exit(1);
    }

    /* Signal that this problem does not explicitly depend on time. */
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
    print!("   -------------------------------------------\n");
    print!(
        "  {}  {}  {}  {}\n",
        fmt_fw(t, 10, 6),
        fmt_fw(NV_Ith_S(&y, 0), 10, 6),
        fmt_fw(NV_Ith_S(&y, 1), 10, 6),
        fmt_fw(NV_Ith_S(&y, 2), 10, 6)
    );

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
    print!("   -------------------------------------------\n");
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
    flag = ARKodeGetNumLinSolvSetups(&arkode_mem, &mut nsetups);
    check_flag(flag, "ARKodeGetNumLinSolvSetups");
    flag = ARKodeGetNumErrTestFails(&arkode_mem, &mut netf);
    check_flag(flag, "ARKodeGetNumErrTestFails");
    flag = ARKodeGetNumStepSolveFails(&arkode_mem, &mut ncfn);
    check_flag(flag, "ARKodeGetNumStepSolveFails");
    flag = ARKodeGetNumNonlinSolvIters(&arkode_mem, &mut nni);
    check_flag(flag, "ARKodeGetNumNonlinSolvIters");
    flag = ARKodeGetNumNonlinSolvConvFails(&arkode_mem, &mut nnf);
    check_flag(flag, "ARKodeGetNumNonlinSolvConvFails");
    flag = ARKodeGetNumJacEvals(&arkode_mem, &mut nje);
    check_flag(flag, "ARKodeGetNumJacEvals");
    flag = ARKodeGetNumLinRhsEvals(&arkode_mem, &mut nfeLS);
    check_flag(flag, "ARKodeGetNumLinRhsEvals");

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
        "   Total number of nonlinear solver convergence failures = {}\n",
        nnf
    );
    print!("   Total number of error test failures = {}\n", netf);
    print!(
        "   Total number of failed steps from solver failure = {}\n",
        ncfn
    );

    /* Clean up and return with successful completion */
    N_VDestroy(y); /* Free y vector */
    ARKodeFree(&mut arkode_mem_opt); /* Free integrator memory */
    let _ = SUNLinSolFree(Some(LS)); /* Free linear solver */
    SUNMatDestroy(A); /* Free A matrix */
    let _ = SUNContext_Free(&mut ctx_opt); /* Free context */
}

/*-------------------------------
 * Functions called by the solver
 *-------------------------------*/

/* f routine to compute the ODE RHS function f(t,y). */
fn f(
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
    let b = rdata[1];
    let ep = rdata[2];
    let u = NV_Ith_S(y, 0); /* access solution values */
    let v = NV_Ith_S(y, 1);
    let w = NV_Ith_S(y, 2);

    /* fill in the RHS function */
    NV_Ith_S_set(ydot, 0, a - (w + 1.0) * u + v * u * u);
    NV_Ith_S_set(ydot, 1, w * u - v * u * u);
    NV_Ith_S_set(ydot, 2, (b - w) / ep - w * u);

    0 /* Return with success */
}

/* Jacobian routine to compute J(t,y) = df/dy. */
fn Jac(
    _t: sunrealtype,
    y: &N_Vector,
    _fy: &N_Vector,
    J: &SUNMatrix,
    user_data: &mut Option<Box<dyn Any>>,
    _tmp1: &N_Vector,
    _tmp2: &N_Vector,
    _tmp3: &N_Vector,
) -> i32 {
    let rdata = user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<[sunrealtype; 3]>())
        .expect("user_data is [sunrealtype; 3]"); /* cast user_data to sunrealtype */
    let ep = rdata[2]; /* access data entries */
    let u = NV_Ith_S(y, 0); /* access solution values */
    let v = NV_Ith_S(y, 1);
    let w = NV_Ith_S(y, 2);

    /* fill in the Jacobian via SUNDenseMatrix macro, SM_ELEMENT_D */
    SM_ELEMENT_D_set(J, 0, 0, -(w + 1.0) + 2.0 * u * v);
    SM_ELEMENT_D_set(J, 0, 1, u * u);
    SM_ELEMENT_D_set(J, 0, 2, -u);

    SM_ELEMENT_D_set(J, 1, 0, w - 2.0 * u * v);
    SM_ELEMENT_D_set(J, 1, 1, -u * u);
    SM_ELEMENT_D_set(J, 1, 2, u);

    SM_ELEMENT_D_set(J, 2, 0, -w);
    SM_ELEMENT_D_set(J, 2, 1, 0.0);
    SM_ELEMENT_D_set(J, 2, 2, -1.0 / ep - u);

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
