/* -----------------------------------------------------------------
 * Rust port of examples/cvode/serial/cvRoberts_dnsL.c
 * (LAPACK dense variant; this port substitutes the native
 * SUNLinSol_Dense for SUNLinSol_LapackDense).
 * -----------------------------------------------------------------
 * Example problem:
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
 * While integrating the system, we also use the rootfinding
 * feature to find the points at which y1 = 1e-4 or at which
 * y3 = 0.01. This program solves the problem with the BDF method,
 * Newton iteration with the dense linear solver, and a
 * user-supplied Jacobian routine.
 * It uses a scalar relative tolerance and a vector absolute
 * tolerance. Output is printed in decades from t = .4 to t = 4.e10.
 * Run statistics (optional outputs) are printed at the end.
 * -----------------------------------------------------------------*/

#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

use std::any::Any;

use cvode_rs::prelude::*;

/* User-defined vector and matrix accessor helpers: Ith, IJth */

/* Ith(v,i) references the ith component of the vector v, where i is in
   the range [1..NEQ]. IJth(A,i,j) references the (i,j)th element of the
   dense matrix A, where i and j are in the range [1..NEQ]; both use the
   same 1-based offset arithmetic as the C macros. */

fn Ith(v: &N_Vector, i: usize) -> sunrealtype {
    N_VGetArrayPointer(v).expect("N_VGetArrayPointer")[i - 1]
}

fn Ith_set(v: &N_Vector, i: usize, val: sunrealtype) {
    N_VGetArrayPointer(v).expect("N_VGetArrayPointer")[i - 1] = val;
}

fn IJth_set(A: &SUNMatrix, i: sunindextype, j: sunindextype, val: sunrealtype) {
    SM_ELEMENT_D_set(A, i - 1, j - 1, val);
}

/* Problem Constants */

const NEQ: sunindextype = 3; /* number of equations  */
const Y1: sunrealtype = 1.0; /* initial y components */
const Y2: sunrealtype = 0.0;
const Y3: sunrealtype = 0.0;
const RTOL: sunrealtype = 1.0e-4; /* scalar relative tolerance            */
const ATOL1: sunrealtype = 1.0e-8; /* vector absolute tolerance components */
const ATOL2: sunrealtype = 1.0e-14;
const ATOL3: sunrealtype = 1.0e-6;
const T0: sunrealtype = 0.0; /* initial time           */
const T1: sunrealtype = 0.4; /* first output time      */
const TMULT: sunrealtype = 10.0; /* output time factor     */
const NOUT: i32 = 12; /* number of output times */

const ZERO: sunrealtype = 0.0;

/*
 *-------------------------------
 * Main Program
 *-------------------------------
 */

fn main() {
    let mut t: sunrealtype = 0.0;

    let mut sunctx: Option<SUNContext> = None;
    let mut rootsfound = [0i32; 2];

    /* Create the SUNDIALS context */
    let retval = SUNContext_Create(SUN_COMM_NULL, &mut sunctx);
    if check_retval_int(retval, "SUNContext_Create") != 0 {
        std::process::exit(1);
    }
    let ctx = sunctx.as_ref().expect("SUNContext").clone();

    /* Initial conditions */
    let y = N_VNew_Serial(NEQ, &ctx);
    if check_retval_ptr(&y, "N_VNew_Serial", 0) != 0 {
        std::process::exit(1);
    }
    let y = y.expect("N_VNew_Serial");

    /* Initialize y */
    Ith_set(&y, 1, Y1);
    Ith_set(&y, 2, Y2);
    Ith_set(&y, 3, Y3);

    /* Set the vector absolute tolerance */
    let abstol = N_VNew_Serial(NEQ, &ctx);
    if check_retval_ptr(&abstol, "N_VNew_Serial", 0) != 0 {
        std::process::exit(1);
    }
    let abstol = abstol.expect("N_VNew_Serial");

    Ith_set(&abstol, 1, ATOL1);
    Ith_set(&abstol, 2, ATOL2);
    Ith_set(&abstol, 3, ATOL3);

    /* Call CVodeCreate to create the solver memory and specify the
     * Backward Differentiation Formula */
    let cvode_mem = CVodeCreate(CV_BDF, &ctx);
    if check_retval_ptr(&cvode_mem, "CVodeCreate", 0) != 0 {
        std::process::exit(1);
    }
    let cvode_mem = cvode_mem.expect("CVodeCreate");

    /* Call CVodeInit to initialize the integrator memory and specify the
     * user's right hand side function in y'=f(t,y), the initial time T0, and
     * the initial dependent variable vector y. */
    let retval = CVodeInit(&cvode_mem, f, T0, &y);
    if check_retval_int(retval, "CVodeInit") != 0 {
        std::process::exit(1);
    }

    /* Call CVodeSVtolerances to specify the scalar relative tolerance
     * and vector absolute tolerances */
    let retval = CVodeSVtolerances(&cvode_mem, RTOL, &abstol);
    if check_retval_int(retval, "CVodeSVtolerances") != 0 {
        std::process::exit(1);
    }

    /* Call CVodeRootInit to specify the root function g with 2 components */
    let retval = CVodeRootInit(&cvode_mem, 2, Some(g));
    if check_retval_int(retval, "CVodeRootInit") != 0 {
        std::process::exit(1);
    }

    /* Create dense SUNMatrix for use in linear solves */
    let A = SUNDenseMatrix(NEQ, NEQ, &ctx);
    if check_retval_ptr(&A, "SUNDenseMatrix", 0) != 0 {
        std::process::exit(1);
    }
    let A = A.expect("SUNDenseMatrix");

    /* Create SUNLinSol_Dense solver object for use by CVode */
    let LS = SUNLinSol_Dense(&y, &A, &ctx);
    if check_retval_ptr(&LS, "SUNLinSol_Dense", 0) != 0 {
        std::process::exit(1);
    }
    let LS = LS.expect("SUNLinSol_Dense");

    /* Attach the matrix and linear solver */
    let retval = CVodeSetLinearSolver(&cvode_mem, &LS, Some(&A));
    if check_retval_int(retval, "CVodeSetLinearSolver") != 0 {
        std::process::exit(1);
    }

    /* Set the user-supplied Jacobian routine Jac */
    let retval = CVodeSetJacFn(&cvode_mem, Some(Jac));
    if check_retval_int(retval, "CVodeSetJacFn") != 0 {
        std::process::exit(1);
    }

    /* In loop, call CVode, print results, and test for error.
    Break out of loop when NOUT preset output times have been reached.  */
    print!(" \n3-species kinetics problem\n\n");

    let mut iout = 0;
    let mut tout = T1;
    loop {
        let retval = CVode(&cvode_mem, tout, &y, &mut t, CV_NORMAL);
        PrintOutput(t, Ith(&y, 1), Ith(&y, 2), Ith(&y, 3));

        if retval == CV_ROOT_RETURN {
            let retvalr = CVodeGetRootInfo(&cvode_mem, &mut rootsfound);
            if check_retval_int(retvalr, "CVodeGetRootInfo") != 0 {
                std::process::exit(1);
            }
            PrintRootInfo(rootsfound[0], rootsfound[1]);
        }

        if check_retval_int(retval, "CVode") != 0 {
            break;
        }
        if retval == CV_SUCCESS {
            iout += 1;
            tout *= TMULT;
        }

        if iout == NOUT {
            break;
        }
    }

    /* Print some final statistics */
    PrintFinalStats(&cvode_mem);

    /* Free memory */
    N_VDestroy(y); /* Free y vector */
    N_VDestroy(abstol); /* Free abstol vector */
    let mut cvode_mem = Some(cvode_mem);
    CVodeFree(&mut cvode_mem); /* Free CVODE memory */
    SUNLinSolFree(Some(LS)); /* Free the linear solver memory */
    SUNMatDestroy(A); /* Free the matrix memory */
    SUNContext_Free(&mut sunctx); /* Free the SUNDIALS context */
}

/*
 *-------------------------------
 * Functions called by the solver
 *-------------------------------
 */

/*
 * f routine. Compute function f(t,y).
 */

fn f(_t: sunrealtype, y: &N_Vector, ydot: &N_Vector, _user_data: &mut Option<Box<dyn Any>>) -> i32 {
    let y1 = Ith(y, 1);
    let y2 = Ith(y, 2);
    let y3 = Ith(y, 3);

    let yd1 = -0.04 * y1 + 1.0e4 * y2 * y3;
    Ith_set(ydot, 1, yd1);
    let yd3 = 3.0e7 * y2 * y2;
    Ith_set(ydot, 3, yd3);
    Ith_set(ydot, 2, -yd1 - yd3);

    0
}

/*
 * g routine. Compute functions g_i(t,y) for i = 0,1.
 */

fn g(
    _t: sunrealtype,
    y: &N_Vector,
    gout: &mut [sunrealtype],
    _user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    let y1 = Ith(y, 1);
    let y3 = Ith(y, 3);
    gout[0] = y1 - 0.0001;
    gout[1] = y3 - 0.01;

    0
}

/*
 * Jacobian routine. Compute J(t,y) = df/dy. *
 */

fn Jac(
    _t: sunrealtype,
    y: &N_Vector,
    _fy: &N_Vector,
    J: &SUNMatrix,
    _user_data: &mut Option<Box<dyn Any>>,
    _tmp1: &N_Vector,
    _tmp2: &N_Vector,
    _tmp3: &N_Vector,
) -> i32 {
    let y2 = Ith(y, 2);
    let y3 = Ith(y, 3);

    IJth_set(J, 1, 1, -0.04);
    IJth_set(J, 1, 2, 1.0e4 * y3);
    IJth_set(J, 1, 3, 1.0e4 * y2);

    IJth_set(J, 2, 1, 0.04);
    IJth_set(J, 2, 2, -1.0e4 * y3 - 6.0e7 * y2);
    IJth_set(J, 2, 3, -1.0e4 * y2);

    IJth_set(J, 3, 1, ZERO);
    IJth_set(J, 3, 2, 6.0e7 * y2);
    IJth_set(J, 3, 3, ZERO);

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

fn PrintRootInfo(root_f1: i32, root_f2: i32) {
    print!("    rootsfound[] = {:>3} {:>3}\n", root_f1, root_f2);
}

/*
 * Get and print some final statistics
 */

fn PrintFinalStats(cvode_mem: &CVodeMem) {
    let (mut nst, mut nfe, mut nsetups, mut nje, mut nfeLS) = (0i64, 0i64, 0i64, 0i64, 0i64);
    let (mut nni, mut nnf, mut ncfn, mut netf, mut nge) = (0i64, 0i64, 0i64, 0i64, 0i64);

    let retval = CVodeGetNumSteps(cvode_mem, &mut nst);
    check_retval_int(retval, "CVodeGetNumSteps");
    let retval = CVodeGetNumRhsEvals(cvode_mem, &mut nfe);
    check_retval_int(retval, "CVodeGetNumRhsEvals");
    let retval = CVodeGetNumLinSolvSetups(cvode_mem, &mut nsetups);
    check_retval_int(retval, "CVodeGetNumLinSolvSetups");
    let retval = CVodeGetNumErrTestFails(cvode_mem, &mut netf);
    check_retval_int(retval, "CVodeGetNumErrTestFails");
    let retval = CVodeGetNumNonlinSolvIters(cvode_mem, &mut nni);
    check_retval_int(retval, "CVodeGetNumNonlinSolvIters");
    let retval = CVodeGetNumNonlinSolvConvFails(cvode_mem, &mut nnf);
    check_retval_int(retval, "CVodeGetNumNonlinSolvConvFails");
    let retval = CVodeGetNumStepSolveFails(cvode_mem, &mut ncfn);
    check_retval_int(retval, "CVodeGetNumStepSolveFails");

    let retval = CVodeGetNumJacEvals(cvode_mem, &mut nje);
    check_retval_int(retval, "CVodeGetNumJacEvals");
    let retval = CVodeGetNumLinRhsEvals(cvode_mem, &mut nfeLS);
    check_retval_int(retval, "CVodeGetNumLinRhsEvals");

    let retval = CVodeGetNumGEvals(cvode_mem, &mut nge);
    check_retval_int(retval, "CVodeGetNumGEvals");

    print!("\nFinal Statistics:\n");
    print!(
        "nst = {:<6} nfe = {:<6} nsetups = {:<6} nfeLS = {:<6} nje = {}\n",
        nst, nfe, nsetups, nfeLS, nje
    );
    print!(
        "nni = {:<6} nnf = {:<6} netf = {:<6}    ncfn = {:<6}  nge = {}\n\n",
        nni, nnf, netf, ncfn, nge
    );
}

/*
 * Check function return value...
 *   opt == 0 means SUNDIALS function allocates memory so check if
 *            returned NULL pointer
 *   opt == 1 means SUNDIALS function returns an integer value so check if
 *            retval < 0 (see check_retval_int)
 *   opt == 2 means function allocates memory so check if returned
 *            NULL pointer
 */

fn check_retval_ptr<T>(returnvalue: &Option<T>, funcname: &str, opt: i32) -> i32 {
    /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
    if opt == 0 && returnvalue.is_none() {
        eprint!(
            "\nSUNDIALS_ERROR: {}() failed - returned NULL pointer\n\n",
            funcname
        );
        return 1;
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

fn check_retval_int(retval: i32, funcname: &str) -> i32 {
    /* Check if retval < 0 */
    if retval < 0 {
        eprint!(
            "\nSUNDIALS_ERROR: {}() failed with retval = {}\n\n",
            funcname, retval
        );
        return 1;
    }

    0
}
