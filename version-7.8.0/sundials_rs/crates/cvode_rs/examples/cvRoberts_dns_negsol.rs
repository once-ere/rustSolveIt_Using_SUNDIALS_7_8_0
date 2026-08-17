/* -----------------------------------------------------------------
 * Rust port of examples/cvode/serial/cvRoberts_dns_negsol.c
 * -----------------------------------------------------------------
 * Modification of the CVODE example cvRoberts_dns to illustrate
 * the treatment of unphysical solution components through the RHS
 * function return retval.
 *
 * Note that, to make possible negative solution components, the
 * absolute tolerances had to be loosened a bit from their values
 * in cvRoberts_dns.
 *
 * The following is a simple example problem, with the coding
 * needed for its solution by CVODE. The problem is from
 * chemical kinetics, and consists of the following three rate
 * equations:
 *    dy1/dt = -.04*y1 + 1.e4*y2*y3
 *    dy2/dt = .04*y1 - 1.e4*y2*y3 - 3.e7*(y2)^2
 *    dy3/dt = 3.e7*(y2)^2
 * on the interval from t = 0.0 to t = 4.e10, with initial
 * conditions: y1 = 1.0, y2 = y3 = 0. The problem is stiff.
 * -----------------------------------------------------------------*/

#![allow(non_snake_case, non_upper_case_globals)]

use cvode_rs::prelude::*;

use std::any::Any;

/* NV_Ith_S(v,i) accessor helpers (0-based, exactly as the C macro) */

fn NV_Ith(v: &N_Vector, i: usize) -> sunrealtype {
    N_VGetArrayPointer(v).expect("N_VGetArrayPointer")[i]
}

fn NV_Ith_set(v: &N_Vector, i: usize, x: sunrealtype) {
    N_VGetArrayPointer(v).expect("N_VGetArrayPointer")[i] = x;
}

/* Problem Constants */

const NEQ: sunindextype = 3; /* number of equations  */
const Y1: sunrealtype = 1.0; /* initial y components */
const Y2: sunrealtype = 0.0;
const Y3: sunrealtype = 0.0;
const RTOL: sunrealtype = 1.0e-4; /* scalar relative tolerance            */
const ATOL1: sunrealtype = 1.0e-7; /* vector absolute tolerance components */
const ATOL2: sunrealtype = 1.0e-13;
const ATOL3: sunrealtype = 1.0e-5;
const T0: sunrealtype = 0.0; /* initial time           */
const T1: sunrealtype = 0.4; /* first output time      */
const TMULT: sunrealtype = 10.0; /* output time factor     */
const NOUT: i32 = 14; /* number of output times */

/*
 *-------------------------------
 * Main Program
 *-------------------------------
 */

fn main() {
    let mut sunctx: Option<SUNContext> = None;
    let mut t: sunrealtype = 0.0;

    /* Create the SUNDIALS context */
    let retval = SUNContext_Create(SUN_COMM_NULL, &mut sunctx);
    if check_retval_flag(retval, "SUNContext_Create") != 0 {
        std::process::exit(1);
    }
    let ctx = sunctx.as_ref().expect("SUNContext").clone();

    /* Initial conditions */
    let y = N_VNew_Serial(NEQ, &ctx);
    if check_retval_ptr(&y, "N_VNew_Serial") != 0 {
        std::process::exit(1);
    }
    let y = y.expect("N_VNew_Serial");

    /* Initialize y */
    NV_Ith_set(&y, 0, Y1);
    NV_Ith_set(&y, 1, Y2);
    NV_Ith_set(&y, 2, Y3);

    /* Set the vector absolute tolerance */
    let abstol = N_VNew_Serial(NEQ, &ctx);
    if check_retval_ptr(&abstol, "N_VNew_Serial") != 0 {
        std::process::exit(1);
    }
    let abstol = abstol.expect("N_VNew_Serial");

    NV_Ith_set(&abstol, 0, ATOL1);
    NV_Ith_set(&abstol, 1, ATOL2);
    NV_Ith_set(&abstol, 2, ATOL3);

    /* Call CVodeCreate to create the solver memory and specify the
     * Backward Differentiation Formula */
    let mut cvode_mem = CVodeCreate(CV_BDF, &ctx);
    if check_retval_ptr(&cvode_mem, "CVodeCreate") != 0 {
        std::process::exit(1);
    }
    let cv = cvode_mem.as_ref().expect("CVodeCreate").clone();

    /* Call CVodeInit to initialize the integrator memory and specify the
     * user's right hand side function in y'=f(t,y), the initial time T0, and
     * the initial dependent variable vector y. */
    let retval = CVodeInit(&cv, f, T0, &y);
    if check_retval_flag(retval, "CVodeInit") != 0 {
        std::process::exit(1);
    }

    /* Call CVodeSVtolerances to specify the scalar relative tolerance
     * and vector absolute tolerances */
    let retval = CVodeSVtolerances(&cv, RTOL, &abstol);
    if check_retval_flag(retval, "CVodeSVtolerances") != 0 {
        std::process::exit(1);
    }

    /* Call CVodeSetUserData to pass the check negative retval as user data
       (the C code passes &check_negative; here the flag lives in user_data
       and each C assignment to check_negative becomes a CVodeSetUserData) */
    let retval = CVodeSetUserData(&cv, Some(Box::new(SUNFALSE)));
    if check_retval_flag(retval, "CVodeSetUserData") != 0 {
        std::process::exit(1);
    }

    /* Create dense SUNMatrix for use in linear solves */
    let A = SUNDenseMatrix(NEQ, NEQ, &ctx);
    if check_retval_ptr(&A, "SUNDenseMatrix") != 0 {
        std::process::exit(1);
    }
    let A = A.expect("SUNDenseMatrix");

    /* Create dense SUNLinearSolver object for use by CVode */
    let LS = SUNLinSol_Dense(&y, &A, &ctx);
    if check_retval_ptr(&LS, "SUNLinSol_Dense") != 0 {
        std::process::exit(1);
    }
    let LS = LS.expect("SUNLinSol_Dense");

    /* Attach the matrix and linear solver */
    let retval = CVodeSetLinearSolver(&cv, &LS, Some(&A));
    if check_retval_flag(retval, "CVodeSetLinearSolver") != 0 {
        std::process::exit(1);
    }

    /* Case 1: ignore negative solution components */
    print!("Ignore negative solution components\n\n");
    /* check_negative = SUNFALSE */
    let _ = CVodeSetUserData(&cv, Some(Box::new(SUNFALSE)));
    /* In loop, call CVode in CV_NORMAL mode */
    let mut iout: i32 = 0;
    let mut tout = T1;
    loop {
        CVode(&cv, tout, &y, &mut t, CV_NORMAL);
        PrintOutput(t, NV_Ith(&y, 0), NV_Ith(&y, 1), NV_Ith(&y, 2));
        iout += 1;
        tout *= TMULT;
        if iout == NOUT {
            break;
        }
    }
    /* Print some final statistics */
    PrintFinalStats(&cv);

    /* Case 2: intercept negative solution components */
    print!("Intercept negative solution components\n\n");
    /* check_negative = SUNTRUE */
    let _ = CVodeSetUserData(&cv, Some(Box::new(SUNTRUE)));
    /* Reinitialize solver */
    NV_Ith_set(&y, 0, Y1);
    NV_Ith_set(&y, 1, Y2);
    NV_Ith_set(&y, 2, Y3);
    let _ = CVodeReInit(&cv, T0, &y);
    /* In loop, call CVode in CV_NORMAL mode */
    let mut iout: i32 = 0;
    let mut tout = T1;
    loop {
        CVode(&cv, tout, &y, &mut t, CV_NORMAL);
        PrintOutput(t, NV_Ith(&y, 0), NV_Ith(&y, 1), NV_Ith(&y, 2));
        iout += 1;
        tout *= TMULT;
        if iout == NOUT {
            break;
        }
    }
    /* Print some final statistics */
    PrintFinalStats(&cv);

    /* Free memory */
    N_VDestroy(y); /* Free y vector */
    N_VDestroy(abstol); /* Free abstol vector */
    CVodeFree(&mut cvode_mem); /* Free CVODE memory */
    SUNLinSolFree(Some(LS)); /* Free the linear solver memory */
    SUNMatDestroy(A); /* Free the matrix memory */
    let _ = SUNContext_Free(&mut sunctx); /* Free the SUNDIALS context */
}

/*
 *-------------------------------
 * Functions called by the solver
 *-------------------------------
 */

/*
 * f routine. Compute function f(t,y).
 */

fn f(_t: sunrealtype, y: &N_Vector, ydot: &N_Vector, user_data: &mut Option<Box<dyn Any>>) -> i32 {
    let check_negative = *user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<sunbooleantype>())
        .expect("user_data is the check_negative flag");

    let y1 = NV_Ith(y, 0);
    let y2 = NV_Ith(y, 1);
    let y3 = NV_Ith(y, 2);

    if check_negative && (y1 < 0.0 || y2 < 0.0 || y3 < 0.0) {
        return 1;
    }

    let yd1 = -0.04 * y1 + 1.0e4 * y2 * y3;
    NV_Ith_set(ydot, 0, yd1);
    let yd3 = 3.0e7 * y2 * y2;
    NV_Ith_set(ydot, 2, yd3);
    NV_Ith_set(ydot, 1, -yd1 - yd3);

    0
}

/*
 *-------------------------------
 * Private helper functions
 *-------------------------------
 */

fn PrintOutput(t: sunrealtype, y1: sunrealtype, y2: sunrealtype, y3: sunrealtype) {
    print!(
        "At t = {}      y ={}  {}  {}\n",
        fmt_e(t, 4),
        fmt_ew(y1, 14, 6),
        fmt_ew(y2, 14, 6),
        fmt_ew(y3, 14, 6)
    );
}

fn PrintFinalStats(cvode_mem: &CVodeMem) {
    let (mut nst, mut nfe, mut nsetups, mut nje, mut nfeLS) = (0i64, 0i64, 0i64, 0i64, 0i64);
    let (mut nni, mut nnf, mut ncfn, mut netf) = (0i64, 0i64, 0i64, 0i64);

    let retval = CVodeGetNumSteps(cvode_mem, &mut nst);
    check_retval_flag(retval, "CVodeGetNumSteps");
    let retval = CVodeGetNumRhsEvals(cvode_mem, &mut nfe);
    check_retval_flag(retval, "CVodeGetNumRhsEvals");
    let retval = CVodeGetNumLinSolvSetups(cvode_mem, &mut nsetups);
    check_retval_flag(retval, "CVodeGetNumLinSolvSetups");
    let retval = CVodeGetNumErrTestFails(cvode_mem, &mut netf);
    check_retval_flag(retval, "CVodeGetNumErrTestFails");
    let retval = CVodeGetNumNonlinSolvIters(cvode_mem, &mut nni);
    check_retval_flag(retval, "CVodeGetNumNonlinSolvIters");
    let retval = CVodeGetNumNonlinSolvConvFails(cvode_mem, &mut nnf);
    check_retval_flag(retval, "CVodeGetNumNonlinSolvConvFails");
    let retval = CVodeGetNumStepSolveFails(cvode_mem, &mut ncfn);
    check_retval_flag(retval, "CVodeGetNumStepSolveFails");

    let retval = CVodeGetNumJacEvals(cvode_mem, &mut nje);
    check_retval_flag(retval, "CVodeGetNumJacEvals");
    let retval = CVodeGetNumLinRhsEvals(cvode_mem, &mut nfeLS);
    check_retval_flag(retval, "CVodeGetNumLinRhsEvals");

    print!("\nFinal Statistics:\n");
    print!(
        "nst = {:<6} nfe = {:<6} nsetups = {:<6} nfeLS = {:<6} nje = {}\n",
        nst, nfe, nsetups, nfeLS, nje
    );
    print!(
        "nni = {:<6} nnf = {:<6} netf = {:<6}    ncfn = {:<6}\n\n",
        nni, nnf, netf, ncfn
    );
}

/*
 * Check function return value... (ports of the C check_retval helper)
 *   check_retval_ptr:  opt == 0 — SUNDIALS function allocates memory so
 *                      check if returned NULL pointer (None)
 *   check_retval_flag: opt == 1 — SUNDIALS function returns an integer
 *                      value so check if retval < 0
 */

fn check_retval_ptr<T>(returnvalue: &Option<T>, funcname: &str) -> i32 {
    if returnvalue.is_none() {
        eprint!(
            "\nSUNDIALS_ERROR: {}() failed - returned NULL pointer\n\n",
            funcname
        );
        return 1;
    }
    0
}

fn check_retval_flag(retval: i32, funcname: &str) -> i32 {
    if retval < 0 {
        eprint!(
            "\nSUNDIALS_ERROR: {}() failed with retval = {}\n\n",
            funcname, retval
        );
        return 1;
    }
    0
}
