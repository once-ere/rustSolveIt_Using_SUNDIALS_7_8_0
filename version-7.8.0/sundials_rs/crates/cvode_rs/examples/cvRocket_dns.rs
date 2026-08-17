/* -----------------------------------------------------------------
 * Rust port of examples/cvode/serial/cvRocket_dns.c
 * -----------------------------------------------------------------
 * Example problem:
 *
 * The problem is a simpliflied model of a rocket, ascending
 * vertically, with mass decreasing over time. The system (of size 2)
 * is given by
 *    y_1 = rocket height H, y_1(0) = 0,
 *    y_2 = rocket velocity v, y_2(0) = 0,
 *    dH/dt = v,
 *    dv/dt = a(t,v).
 * The engine force is reset to 0 when the fuel mass reaches 0, or
 * when H reaches a preset height H_c, whichever happens first.
 * Rootfinding is used to locate the time at which M_f = 0 or H =
 * H_c, and also the time at which the rocket reaches its maximum
 * height, given by the condition v = 0, t > 0.
 *
 * The problem is solved with the BDF method and Dense linear solver.
 * -----------------------------------------------------------------*/
#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

use cvode_rs::prelude::*;

use std::any::Any;

/* Problem Constants */

const NEQ: i64 = 2; /* number of equations  */

const Force: sunrealtype = 2200.0; /* engine force */
const massr: sunrealtype = 10.0; /* rocket mass (empty) */
const massf0: sunrealtype = 1.0; /* initial fuel mass */
const brate: sunrealtype = 0.1; /* fuel burn rate */
const Drag: sunrealtype = 0.3; /* Drag coefficient */
const grav: sunrealtype = 32.0; /* acceleration due to gravity */
const Hcut: sunrealtype = 4000.0; /* height of engine cutoff */

const Y1: sunrealtype = 0.0; /* initial y components */
const Y2: sunrealtype = 0.0;
const RTOL: sunrealtype = 1.0e-5; /* scalar relative tolerance            */
const ATOL1: sunrealtype = 1.0e-2; /* vector absolute tolerance components */
const ATOL2: sunrealtype = 1.0e-1;
const T0: sunrealtype = 0.0; /* initial time           */
const T1: sunrealtype = 1.0; /* first output time      */
const TINC: sunrealtype = 1.0; /* output time increment  */
const NOUT: i32 = 70; /* number of output times */

const ZERO: sunrealtype = 0.0;

/*
 *-------------------------------
 * Main Program
 *-------------------------------
 */

fn main() {
    let mut t: sunrealtype = 0.0;
    let mut rootsfound = [0i32; 2];

    /* Create the SUNDIALS context */
    let mut sunctx_opt: Option<SUNContext> = None;
    let mut retval = SUNContext_Create(SUN_COMM_NULL, &mut sunctx_opt);
    if check_retval_int(retval, "SUNContext_Create") {
        std::process::exit(1);
    }
    let sunctx = sunctx_opt.unwrap();

    /* Create serial vector of length NEQ for I.C. and abstol */
    let y = N_VNew_Serial(NEQ, &sunctx);
    if check_retval_null(&y, "N_VNew_Serial") {
        std::process::exit(1);
    }
    let y = y.unwrap();
    let abstol = N_VNew_Serial(NEQ, &sunctx);
    if check_retval_null(&abstol, "N_VNew_Serial") {
        std::process::exit(1);
    }
    let abstol = abstol.unwrap();

    /* Initialize y */
    {
        let mut ydata = N_VGetArrayPointer(&y).expect("vector data");
        ydata[0] = Y1;
        ydata[1] = Y2;
    }

    /* Set the scalar relative tolerance */
    let reltol = RTOL;
    /* Set the vector absolute tolerance */
    {
        let mut adata = N_VGetArrayPointer(&abstol).expect("vector data");
        adata[0] = ATOL1;
        adata[1] = ATOL2;
    }

    /* Call CVodeCreate to create the solver memory and specify the Backward Differentiation Formula */
    let cvode_mem = CVodeCreate(CV_BDF, &sunctx);
    if check_retval_null(&cvode_mem, "CVodeCreate") {
        std::process::exit(1);
    }
    let cvode_mem = cvode_mem.unwrap();

    /* Call CVodeInit to initialize the integrator memory and specify the right-hand side function in
     * y'=f(t,y), the initial time T0, and the initial dependent variable vector y. */
    retval = CVodeInit(&cvode_mem, f, T0, &y);
    if check_retval_int(retval, "CVodeInit") {
        std::process::exit(1);
    }

    /* Call CVodeSVtolerances to specify the scalar relative tolerance and vector absolute tolerances */
    retval = CVodeSVtolerances(&cvode_mem, reltol, &abstol);
    if check_retval_int(retval, "CVodeSVtolerances") {
        std::process::exit(1);
    }

    /* Provide sunbooleantype engine_on as user data for use in f and g routines
     * (the C example passes &engine_on; here the flag is stored in the user_data
     * box and re-set whenever main changes it). */
    retval = CVodeSetUserData(&cvode_mem, Some(Box::new(SUNTRUE)));
    if check_retval_int(retval, "CVodeSetUserData") {
        std::process::exit(1);
    }

    /* Call CVodeRootInit to specify the root function g with 2 components */
    retval = CVodeRootInit(&cvode_mem, 2, Some(g));
    if check_retval_int(retval, "CVodeRootInit") {
        std::process::exit(1);
    }

    /* Create dense SUNMatrix for use in linear solves */
    let A = SUNDenseMatrix(NEQ, NEQ, &sunctx);
    if check_retval_null(&A, "SUNDenseMatrix") {
        std::process::exit(1);
    }
    let A = A.unwrap();

    /* Create dense SUNLinearSolver object for use by CVode */
    let LS = SUNLinSol_Dense(&y, &A, &sunctx);
    if check_retval_null(&LS, "SUNLinSol_Dense") {
        std::process::exit(1);
    }
    let LS = LS.unwrap();

    /* Call CVodeSetLinearSolver to attach the matrix and linear solver to CVode */
    retval = CVodeSetLinearSolver(&cvode_mem, &LS, Some(&A));
    if check_retval_int(retval, "CVodeSetLinearSolver") {
        std::process::exit(1);
    }

    /* Set the user-supplied Jacobian routine Jac */
    retval = CVodeSetJacFn(&cvode_mem, Some(Jac));
    if check_retval_int(retval, "CVodeSetJacFn") {
        std::process::exit(1);
    }

    /* In loop, call CVode, print results, check for root stops, and test for error.  On the first
    root return, restart with engine turned off. Break out of loop when NOUT preset output times
    have been reached, or when the returned value of H is negative.  */
    print!(" \nAccelerating rocket problem\n\n");

    let mut iout: i32 = 0;
    let mut tout = T1;
    let mut engine_on: sunbooleantype = SUNTRUE;
    let mut numroot: i32 = 2;
    loop {
        retval = CVode(&cvode_mem, tout, &y, &mut t, CV_NORMAL);
        if check_retval_int(retval, "CVode") {
            std::process::exit(1);
        }

        {
            let ydata = N_VGetArrayPointer(&y).expect("vector data");
            PrintOutput(t, ydata[0], ydata[1]);
        }

        if engine_on && (retval == CV_ROOT_RETURN) {
            /* engine cutoff */
            let retvalr = CVodeGetRootInfo(&cvode_mem, &mut rootsfound);
            if check_retval_int(retvalr, "CVodeGetRootInfo") {
                std::process::exit(1);
            }
            PrintRootInfo(rootsfound[0], rootsfound[1], numroot);
            engine_on = SUNFALSE;
            numroot = 1;
            /* Propagate the engine_on change into the CVODE user data
             * (C writes through the aliased &engine_on pointer). */
            let _ = CVodeSetUserData(&cvode_mem, Some(Box::new(engine_on)));
            /* Call CVodeRootInit to specify the root function g with 1 component */
            retval = CVodeRootInit(&cvode_mem, 1, Some(g));
            if check_retval_int(retval, "CVodeRootInit") {
                std::process::exit(1);
            }
            /* Reinitialize the solver with current t and y values. */
            retval = CVodeReInit(&cvode_mem, t, &y);
            if check_retval_int(retval, "CVodeReInit") {
                std::process::exit(1);
            }
        } else if (!engine_on) && (retval == CV_ROOT_RETURN) {
            /* max.  height */
            let retvalr = CVodeGetRootInfo(&cvode_mem, &mut rootsfound);
            if check_retval_int(retvalr, "CVodeGetRootInfo") {
                std::process::exit(1);
            }
            PrintRootInfo(rootsfound[0], rootsfound[1], numroot);
        }

        if retval == CV_SUCCESS {
            iout += 1;
            tout += TINC;
        }

        if iout == NOUT {
            break;
        }
        {
            let ydata = N_VGetArrayPointer(&y).expect("vector data");
            if ydata[0] < ZERO {
                break;
            }
        }
    }

    /* Print some final statistics */
    PrintFinalStats(&cvode_mem);

    /* Free y and abstol vectors */
    N_VDestroy(y);
    N_VDestroy(abstol);

    /* Free integrator memory */
    let mut cvode_mem = Some(cvode_mem);
    CVodeFree(&mut cvode_mem);

    /* Free the linear solver memory */
    SUNLinSolFree(Some(LS));

    /* Free the matrix memory */
    SUNMatDestroy(A);

    /* Free the SUNDIALS context */
    let mut sunctx = Some(sunctx);
    SUNContext_Free(&mut sunctx);

    std::process::exit(retval);
}

/*
 *-------------------------------
 * Functions called by the solver
 *-------------------------------
 */

/*
 * f routine. Compute function f(t,y).
 */

fn f(t: sunrealtype, y: &N_Vector, ydot: &N_Vector, user_data: &mut Option<Box<dyn Any>>) -> i32 {
    let engine_on = *user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<sunbooleantype>())
        .expect("user data");
    let ydata = N_VGetArrayPointer(y).expect("vector data");
    let mut yddata = N_VGetArrayPointer(ydot).expect("vector data");

    let v = ydata[1];
    yddata[0] = v;

    let acc = if engine_on {
        Force / (massr + massf0 - brate * t)
    } else {
        ZERO
    };

    yddata[1] = acc - Drag * v - grav;

    0
}

/*
 * Jacobian routine. Compute J(t,y) = df/dy. *
 */

fn Jac(
    _t: sunrealtype,
    _y: &N_Vector,
    _fy: &N_Vector,
    J: &SUNMatrix,
    _user_data: &mut Option<Box<dyn Any>>,
    _tmp1: &N_Vector,
    _tmp2: &N_Vector,
    _tmp3: &N_Vector,
) -> i32 {
    /* Jdata is column-major */
    let mut Jdata = SUNDenseMatrix_Data(J);

    Jdata[1] = 1.0;
    Jdata[3] = -Drag;

    0
}

/*
 * g routine. Compute functions g_i(t,y).
 */

fn g(
    t: sunrealtype,
    y: &N_Vector,
    gout: &mut [sunrealtype],
    user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    let engine_on = *user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<sunbooleantype>())
        .expect("user data");

    let ydata = N_VGetArrayPointer(y).expect("vector data");

    if engine_on {
        gout[0] = massf0 - brate * t;
        let H = ydata[0];
        gout[1] = H - Hcut;
    } else {
        let v = ydata[1];
        gout[0] = v;
    }

    0
}

/*
 *-------------------------------
 * Private helper functions
 *-------------------------------
 */

fn PrintOutput(t: sunrealtype, y1: sunrealtype, y2: sunrealtype) {
    print!(
        "At t = {}      y ={}  {}\n",
        fmt_e(t, 4),
        fmt_ew(y1, 14, 6),
        fmt_ew(y2, 14, 6)
    );
}

fn PrintRootInfo(root_f1: i32, root_f2: i32, numroot: i32) {
    if numroot == 2 {
        print!("    rootsfound[] = {:>3} {:>3}\n", root_f1, root_f2);
    }
    if numroot == 1 {
        print!("    rootsfound[] = {:>3}\n", root_f1);
    }
}

/*
 * Get and print some final statistics
 */

fn PrintFinalStats(cvode_mem: &CVodeMem) {
    let mut nst: i64 = 0;
    let mut nfe: i64 = 0;
    let mut nsetups: i64 = 0;
    let mut nje: i64 = 0;
    let mut nfeLS: i64 = 0;
    let mut nni: i64 = 0;
    let mut ncfn: i64 = 0;
    let mut netf: i64 = 0;
    let mut nge: i64 = 0;

    let mut retval = CVodeGetNumSteps(cvode_mem, &mut nst);
    check_retval_int(retval, "CVodeGetNumSteps");
    retval = CVodeGetNumRhsEvals(cvode_mem, &mut nfe);
    check_retval_int(retval, "CVodeGetNumRhsEvals");
    retval = CVodeGetNumLinSolvSetups(cvode_mem, &mut nsetups);
    check_retval_int(retval, "CVodeGetNumLinSolvSetups");
    retval = CVodeGetNumErrTestFails(cvode_mem, &mut netf);
    check_retval_int(retval, "CVodeGetNumErrTestFails");
    retval = CVodeGetNumNonlinSolvIters(cvode_mem, &mut nni);
    check_retval_int(retval, "CVodeGetNumNonlinSolvIters");
    retval = CVodeGetNumNonlinSolvConvFails(cvode_mem, &mut ncfn);
    check_retval_int(retval, "CVodeGetNumNonlinSolvConvFails");

    retval = CVodeGetNumJacEvals(cvode_mem, &mut nje);
    check_retval_int(retval, "CVodeGetNumJacEvals");
    retval = CVodeGetNumLinRhsEvals(cvode_mem, &mut nfeLS);
    check_retval_int(retval, "CVodeGetNumLinRhsEvals");

    retval = CVodeGetNumGEvals(cvode_mem, &mut nge);
    check_retval_int(retval, "CVodeGetNumGEvals");

    print!("\nFinal Statistics:\n");
    /* C format "nje = % ld": space flag prefixes non-negative values */
    let nje_s = if nje >= 0 {
        format!(" {}", nje)
    } else {
        format!("{}", nje)
    };
    print!(
        "nst = {:<6} nfe  = {:<6} nsetups = {:<6} nfeLS = {:<6} nje = {}\n",
        nst, nfe, nsetups, nfeLS, nje_s
    );
    print!(
        "nni = {:<6} ncfn = {:<6} netf = {:<6} nge = {}\n \n",
        nni, ncfn, netf, nge
    );
}

/* Check function return value (C check_retval, opt 1) */
fn check_retval_int(retval: i32, funcname: &str) -> bool {
    if retval < 0 {
        eprint!(
            "\nSUNDIALS_ERROR: {}() failed with retval = {}\n\n",
            funcname, retval
        );
        return true;
    }
    false
}

/* Check function return value (C check_retval, opt 0) */
fn check_retval_null<T>(returnvalue: &Option<T>, funcname: &str) -> bool {
    if returnvalue.is_none() {
        eprint!(
            "\nSUNDIALS_ERROR: {}() failed - returned NULL pointer\n\n ",
            funcname
        );
        return true;
    }
    false
}
