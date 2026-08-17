/* -----------------------------------------------------------------
 * Ported from: examples/cvode/serial/cvDisc_dns.c
 * -----------------------------------------------------------------
 * Simple 1D example to illustrate integrating over discontinuities:
 *
 * A) Discontinuity in solution
 *       y' = -y   ; y(0) = 1    ; t = [0,1]
 *       y' = -y   ; y(1) = 1    ; t = [1,2]
 *
 * B) Discontinuity in RHS (y')
 *       y' = -y   ; y(0) = 1    ; t = [0,1]
 *       z' = -5*z ; z(1) = y(1) ; t = [1,2]
 *    This case is solved twice, first by explicitly treating the
 *    discontinuity point and secondly by letting the integrator
 *    deal with the discontinuity.
 * -----------------------------------------------------------------
 */

#![allow(non_snake_case, non_upper_case_globals)]

use std::any::Any;

use cvode_rs::prelude::*;

/* Problem Constants */
const NEQ: sunindextype = 1; /* number of equations */

const RHS1: i32 = 1;
const RHS2: i32 = 2;

fn main() {
    let mut retval: i32;
    let mut t: sunrealtype;
    let mut nst1: i64 = 0;
    let mut nst2: i64 = 0;
    let mut nst: i64;

    let reltol: sunrealtype = 1.0e-3;
    let abstol: sunrealtype = 1.0e-4;

    let t0: sunrealtype = 0.0;
    let t1: sunrealtype = 1.0;
    let t2: sunrealtype = 2.0;

    /* Create the SUNDIALS context */
    let mut sunctx: Option<SUNContext> = None;
    retval = SUNContext_Create(SUN_COMM_NULL, &mut sunctx);
    if check_retval_flag(retval, "SUNContext_Create") != 0 {
        std::process::exit(1);
    }
    let sunctx_h = sunctx.as_ref().expect("SUNContext_Create").clone();

    /* Allocate the vector of initial conditions */
    let y = N_VNew_Serial(NEQ, &sunctx_h).expect("N_VNew_Serial");

    /* Set initial condition */
    N_VGetArrayPointer(&y).expect("N_VGetArrayPointer")[0] = 1.0;

    /*
     * ------------------------------------------------------------
     *  Shared initialization and setup
     * ------------------------------------------------------------
     */

    /* Call CVodeCreate to create CVODE memory block and specify the
     * Backward Differentiaion Formula */
    let cvode_mem = CVodeCreate(CV_BDF, &sunctx_h);
    if check_retval_null(&cvode_mem, "CVodeCreate") != 0 {
        std::process::exit(1);
    }
    let cvode_mem = cvode_mem.expect("CVodeCreate");

    /* Call CVodeInit to initialize integrator memory and specify the
     * user's right hand side function y'=f(t,y), the initial time T0
     * and the initial condiition vector y. */
    retval = CVodeInit(&cvode_mem, f, t0, &y);
    if check_retval_flag(retval, "CVodeInit") != 0 {
        std::process::exit(1);
    }

    /* Call CVodeSStolerances to specify integration tolereances,
     * specifically the scalar relative and absolute tolerance. */
    retval = CVodeSStolerances(&cvode_mem, reltol, abstol);
    if check_retval_flag(retval, "CVodeSStolerances") != 0 {
        std::process::exit(1);
    }

    /* Provide RHS flag as user data which can be access in user provided routines */
    retval = CVodeSetUserData(&cvode_mem, Some(Box::new(RHS1)));
    if check_retval_flag(retval, "CVodeSetUserData") != 0 {
        std::process::exit(1);
    }

    /* Create dense SUNMatrix for use in linear solver */
    let A = SUNDenseMatrix(NEQ, NEQ, &sunctx_h);
    if check_retval_null(&A, "SUNDenseMatrix") != 0 {
        std::process::exit(1);
    }
    let A = A.expect("SUNDenseMatrix");

    /* Create dense linear solver for use by CVode */
    let LS = SUNLinSol_Dense(&y, &A, &sunctx_h);
    if check_retval_null(&LS, "SUNLinSol_Dense") != 0 {
        std::process::exit(1);
    }
    let LS = LS.expect("SUNLinSol_Dense");

    /* Attach the linear solver and matrix to CVode by calling CVodeSetLinearSolver */
    retval = CVodeSetLinearSolver(&cvode_mem, &LS, Some(&A));
    if check_retval_flag(retval, "CVodeSetLinearSolver") != 0 {
        std::process::exit(1);
    }

    /*
     * ---------------------------------------------------------------
     * Discontinuity in the solution
     *
     * 1) Integrate to the discontinuity
     * 2) Integrate from the discontinuity
     * ---------------------------------------------------------------
     */

    /* ---- Integrate to the discontinuity */

    print!("\nDiscontinuity in solution\n\n");

    /* set TSTOP (max time solution proceeds to) - this is not required */
    retval = CVodeSetStopTime(&cvode_mem, t1);
    if check_retval_flag(retval, "CVodeSetStopTime") != 0 {
        std::process::exit(1);
    }

    /* use -y for RHS */
    let _ = CVodeSetUserData(&cvode_mem, Some(Box::new(RHS1)));
    t = t0; /* set the integrator start time */

    print_t_y(t, &y);
    while t < t1 {
        /* advance solver just one internal step */
        retval = CVode(&cvode_mem, t1, &y, &mut t, CV_ONE_STEP);
        if check_retval_flag(retval, "CVode") != 0 {
            std::process::exit(1);
        }
        print_t_y(t, &y);
    }
    /* Get the number of steps the solver took to get to the discont. */
    retval = CVodeGetNumSteps(&cvode_mem, &mut nst1);
    if check_retval_flag(retval, "CvodeGetNumSteps") != 0 {
        std::process::exit(1);
    }

    /* ---- Integrate from the discontinuity */

    /* Include discontinuity */
    N_VGetArrayPointer(&y).expect("N_VGetArrayPointer")[0] = 1.0;

    /* Reinitialize the solver */
    retval = CVodeReInit(&cvode_mem, t1, &y);
    if check_retval_flag(retval, "CVodeReInit") != 0 {
        std::process::exit(1);
    }

    /* set TSTOP (max time solution proceeds to) - this is not required */
    retval = CVodeSetStopTime(&cvode_mem, t2);
    if check_retval_flag(retval, "CVodeSetStopTime") != 0 {
        std::process::exit(1);
    }

    /* use -y for RHS */
    let _ = CVodeSetUserData(&cvode_mem, Some(Box::new(RHS1)));
    t = t1; /* set the integrator start time */

    print_t_y(t, &y);

    while t < t2 {
        /* advance solver just one internal step */
        retval = CVode(&cvode_mem, t2, &y, &mut t, CV_ONE_STEP);
        if check_retval_flag(retval, "CVode") != 0 {
            std::process::exit(1);
        }
        print_t_y(t, &y);
    }

    /* Get the number of steps the solver took after the discont. */
    retval = CVodeGetNumSteps(&cvode_mem, &mut nst2);
    if check_retval_flag(retval, "CvodeGetNumSteps") != 0 {
        std::process::exit(1);
    }

    /* Print statistics */
    nst = nst1 + nst2;
    print!("\nNumber of steps: {} + {} = {}\n", nst1, nst2, nst);

    /*
     * ---------------------------------------------------------------
     * Discontinuity in RHS: Case 1 - explicit treatment
     * Note that it is not required to set TSTOP, but without it
     * we would have to find y(t1) to reinitialize the solver.
     * ---------------------------------------------------------------
     */

    print!("\nDiscontinuity in RHS: Case 1 - explicit treatment\n\n");

    /* Set initial condition */
    N_VGetArrayPointer(&y).expect("N_VGetArrayPointer")[0] = 1.0;

    /* Reinitialize the solver. CVodeReInit does not reallocate memory
     * so it can only be used when the new problem size is the same as
     * the problem size when CVodeCreate was called. */
    retval = CVodeReInit(&cvode_mem, t0, &y);
    if check_retval_flag(retval, "CVodeReInit") != 0 {
        std::process::exit(1);
    }

    /* ---- Integrate to the discontinuity */

    /* Set TSTOP (max time solution proceeds to) to location of discont. */
    retval = CVodeSetStopTime(&cvode_mem, t1);
    if check_retval_flag(retval, "CVodeSetStopTime") != 0 {
        std::process::exit(1);
    }

    /* use -y for RHS */
    let _ = CVodeSetUserData(&cvode_mem, Some(Box::new(RHS1)));
    t = t0; /* set the integrator start time */

    print_t_y(t, &y);
    while t < t1 {
        /* advance solver just one internal step */
        retval = CVode(&cvode_mem, t1, &y, &mut t, CV_ONE_STEP);
        if check_retval_flag(retval, "CVode") != 0 {
            std::process::exit(1);
        }
        print_t_y(t, &y);
    }

    /* Get the number of steps the solver took to get to the discont. */
    retval = CVodeGetNumSteps(&cvode_mem, &mut nst1);
    if check_retval_flag(retval, "CvodeGetNumSteps") != 0 {
        std::process::exit(1);
    }

    /* If TSTOP was not set, we'd need to find y(t1): */
    /* CVodeGetDky(cvode_mem, t1, 0, y); */

    /* ---- Integrate from the discontinuity */

    /* Reinitialize solver */
    let _ = CVodeReInit(&cvode_mem, t1, &y);

    /* set TSTOP (max time solution proceeds to) - this is not required */
    retval = CVodeSetStopTime(&cvode_mem, t2);
    if check_retval_flag(retval, "CVodeSetStopTime") != 0 {
        std::process::exit(1);
    }

    /* use -5y for RHS */
    let _ = CVodeSetUserData(&cvode_mem, Some(Box::new(RHS2)));
    t = t1; /* set the integrator start time */

    print_t_y(t, &y);

    while t < t2 {
        /* advance solver just one internal step */
        retval = CVode(&cvode_mem, t2, &y, &mut t, CV_ONE_STEP);
        if check_retval_flag(retval, "CVode") != 0 {
            std::process::exit(1);
        }
        print_t_y(t, &y);
    }

    /* Get the number of steps the solver took after the discont. */
    retval = CVodeGetNumSteps(&cvode_mem, &mut nst2);
    if check_retval_flag(retval, "CvodeGetNumSteps") != 0 {
        std::process::exit(1);
    }

    /* Print statistics */
    nst = nst1 + nst2;
    print!("\nNumber of steps: {} + {} = {}\n", nst1, nst2, nst);

    /*
     * ---------------------------------------------------------------
     * Discontinuity in RHS: Case 2 - let CVODE deal with it
     * Note that here we MUST set TSTOP to ensure that the
     * change in the RHS happens at the appropriate time
     * ---------------------------------------------------------------
     */

    print!("\nDiscontinuity in RHS: Case 2 - let CVODE deal with it\n\n");

    /* Set initial condition */
    N_VGetArrayPointer(&y).expect("N_VGetArrayPointer")[0] = 1.0;

    /* Reinitialize the solver. CVodeReInit does not reallocate memory
     * so it can only be used when the new problem size is the same as
     * the problem size when CVodeCreate was called. */
    retval = CVodeReInit(&cvode_mem, t0, &y);
    if check_retval_flag(retval, "CVodeReInit") != 0 {
        std::process::exit(1);
    }

    /* ---- Integrate to the discontinuity */

    /* Set TSTOP (max time solution proceeds to) to location of discont. */
    retval = CVodeSetStopTime(&cvode_mem, t1);
    if check_retval_flag(retval, "CVodeSetStopTime") != 0 {
        std::process::exit(1);
    }

    /* use -y for RHS */
    let _ = CVodeSetUserData(&cvode_mem, Some(Box::new(RHS1)));
    t = t0; /* set the integrator start time */

    print_t_y(t, &y);
    while t < t1 {
        /* advance solver just one internal step */
        retval = CVode(&cvode_mem, t1, &y, &mut t, CV_ONE_STEP);
        if check_retval_flag(retval, "CVode") != 0 {
            std::process::exit(1);
        }
        print_t_y(t, &y);
    }

    /* Get the number of steps the solver took to get to the discont. */
    retval = CVodeGetNumSteps(&cvode_mem, &mut nst1);
    if check_retval_flag(retval, "CvodeGetNumSteps") != 0 {
        std::process::exit(1);
    }

    /* ---- Integrate from the discontinuity */

    /* set TSTOP (max time solution proceeds to) - this is not required */
    retval = CVodeSetStopTime(&cvode_mem, t2);
    if check_retval_flag(retval, "CVodeSetStopTime") != 0 {
        std::process::exit(1);
    }

    /* use -5y for RHS */
    let _ = CVodeSetUserData(&cvode_mem, Some(Box::new(RHS2)));
    t = t1; /* set the integrator start time */

    print_t_y(t, &y);

    while t < t2 {
        /* advance solver just one internal step */
        retval = CVode(&cvode_mem, t2, &y, &mut t, CV_ONE_STEP);
        if check_retval_flag(retval, "CVode") != 0 {
            std::process::exit(1);
        }
        print_t_y(t, &y);
    }

    /* Get the number of steps the solver took after the discont. */
    retval = CVodeGetNumSteps(&cvode_mem, &mut nst);
    if check_retval_flag(retval, "CvodeGetNumSteps") != 0 {
        std::process::exit(1);
    }

    /* Print statistics */
    nst2 = nst - nst1;
    print!("\nNumber of steps: {} + {} = {}\n", nst1, nst2, nst);

    /* Free memory */
    N_VDestroy(y);
    SUNMatDestroy(A);
    let _ = SUNLinSolFree(Some(LS));
    CVodeFree(&mut Some(cvode_mem));
    let _ = SUNContext_Free(&mut sunctx);
}

/* Print one output line: C `printf("%12.8e  %12.8e\n", t, NV_Ith_S(y, 0))` */
fn print_t_y(t: sunrealtype, y: &N_Vector) {
    let y0 = N_VGetArrayPointer(y).expect("N_VGetArrayPointer")[0];
    print!("{}  {}\n", fmt_ew(t, 12, 8), fmt_ew(y0, 12, 8));
}

/*
 * RHS function
 * The form of the RHS function is controlled by the flag passed as f_data:
 *   flag = RHS1 -> y' = -y
 *   flag = RHS2 -> y' = -5*y
 */

fn f(_t: sunrealtype, y: &N_Vector, ydot: &N_Vector, f_data: &mut Option<Box<dyn Any>>) -> i32 {
    let flag = f_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<i32>())
        .expect("f_data is the RHS flag");

    let y_data = N_VGetArrayPointer(y).expect("N_VGetArrayPointer");
    let mut ydot_data = N_VGetArrayPointer(ydot).expect("N_VGetArrayPointer");

    match *flag {
        RHS1 => ydot_data[0] = -y_data[0],
        RHS2 => ydot_data[0] = -5.0 * y_data[0],
        _ => {}
    }

    0
}

/*
 * Check function return value...
 *   opt == 0 means SUNDIALS function allocates memory so check if
 *            returned NULL pointer
 *   opt == 1 means SUNDIALS function returns a flag so check if
 *            flag >= 0
 */

/* C check_retval with opt == 1 (flag check) */
fn check_retval_flag(retval: i32, funcname: &str) -> i32 {
    if retval < 0 {
        eprint!(
            "\nSUNDIALS_ERROR: {}() failed with flag = {}\n\n",
            funcname, retval
        );
        1
    } else {
        0
    }
}

/* C check_retval with opt == 0 (NULL pointer check) */
fn check_retval_null<T>(value: &Option<T>, funcname: &str) -> i32 {
    if value.is_none() {
        eprint!(
            "\nSUNDIALS_ERROR: {}() failed - returned NULL pointer\n\n",
            funcname
        );
        1
    } else {
        0
    }
}
