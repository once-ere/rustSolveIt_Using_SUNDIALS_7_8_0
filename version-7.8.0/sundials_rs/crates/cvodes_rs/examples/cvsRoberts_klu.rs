//! Port of `examples/cvodes/serial/cvsRoberts_klu.c`.
//!
//! Example problem:
//!
//! The following is a simple example problem, with the coding
//! needed for its solution by CVODE. The problem is from
//! chemical kinetics, and consists of the following three rate
//! equations:
//!    dy1/dt = -.04*y1 + 1.e4*y2*y3
//!    dy2/dt = .04*y1 - 1.e4*y2*y3 - 3.e7*(y2)^2
//!    dy3/dt = 3.e7*(y2)^2
//! on the interval from t = 0.0 to t = 4.e10, with initial
//! conditions: y1 = 1.0, y2 = y3 = 0. The problem is stiff.
//! While integrating the system, we also use the rootfinding
//! feature to find the points at which y1 = 1e-4 or at which
//! y3 = 0.01. This program solves the problem with the BDF method,
//! Newton iteration with the KLU sparse direct linear solver, and a
//! user-supplied Jacobian routine.
//! It uses a scalar relative tolerance and a vector absolute
//! tolerance. Output is printed in decades from t = .4 to t = 4.e10.
//! Run statistics (optional outputs) are printed at the end.
//!
//! **Solver note.** `SUNLinSol_KLU` here is
//! [`sundials_core::sunlinsol_klu`], whose factorization is the
//! independent pure-Rust sparse LU in `sundials_core::sundials_sparse_lu`
//! rather than SuiteSparse KLU (LGPL, and unreachable without FFI). The
//! printed digits therefore differ from the C original; see
//! `differences/ATTRIBUTION.md`.

#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

use std::any::Any;

use cvodes_rs::prelude::*;

/* User-defined vector accessor helper: Ith
   (C macro `Ith(v,i)` = `NV_Ith_S(v,i-1)`; i is 1-based). */

fn Ith(v: &N_Vector, i: usize) -> sunrealtype {
    N_VGetArrayPointer(v).expect("N_VGetArrayPointer")[i - 1]
}

fn Ith_set(v: &N_Vector, i: usize, x: sunrealtype) {
    N_VGetArrayPointer(v).expect("N_VGetArrayPointer")[i - 1] = x;
}

/* Problem Constants */

const NEQ: sunindextype = 3; /* number of equations  */
const NNZ: sunindextype = 7; /* number of non-zero entries in the Jacobian */
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

/*
 *-------------------------------
 * Main Program
 *-------------------------------
 */

fn main() {
    let mut rootsfound = [0i32; 2];

    /* Create the SUNDIALS context */
    let mut sunctx: Option<SUNContext> = None;
    let retval = SUNContext_Create(SUN_COMM_NULL, &mut sunctx);
    if check_retval(retval, "SUNContext_Create") != 0 {
        std::process::exit(1);
    }
    let ctx = sunctx.clone().unwrap();

    /* Initial conditions */
    let y = N_VNew_Serial(NEQ, &ctx);
    if check_retval_null(&y, "N_VNew_Serial") != 0 {
        std::process::exit(1);
    }
    let y = y.unwrap();

    /* Initialize y */
    Ith_set(&y, 1, Y1);
    Ith_set(&y, 2, Y2);
    Ith_set(&y, 3, Y3);

    /* Set the vector absolute tolerance */
    let abstol = N_VNew_Serial(NEQ, &ctx);
    if check_retval_null(&abstol, "N_VNew_Serial") != 0 {
        std::process::exit(1);
    }
    let abstol = abstol.unwrap();

    Ith_set(&abstol, 1, ATOL1);
    Ith_set(&abstol, 2, ATOL2);
    Ith_set(&abstol, 3, ATOL3);

    /* Call CVodeCreate to create the solver memory and specify the
     * Backward Differentiation Formula */
    let cvode_mem = CVodeCreate(CV_BDF, &ctx);
    if check_retval_null(&cvode_mem, "CVodeCreate") != 0 {
        std::process::exit(1);
    }
    let cv = cvode_mem.clone().unwrap();

    /* Call CVodeInit to initialize the integrator memory and specify the
     * user's right hand side function in y'=f(t,y), the initial time T0, and
     * the initial dependent variable vector y. */
    let retval = CVodeInit(&cv, f, T0, &y);
    if check_retval(retval, "CVodeInit") != 0 {
        std::process::exit(1);
    }

    /* Call CVodeSVtolerances to specify the scalar relative tolerance
     * and vector absolute tolerances */
    let retval = CVodeSVtolerances(&cv, RTOL, &abstol);
    if check_retval(retval, "CVodeSVtolerances") != 0 {
        std::process::exit(1);
    }

    /* Call CVodeRootInit to specify the root function g with 2 components */
    let retval = CVodeRootInit(&cv, 2, Some(g));
    if check_retval(retval, "CVodeRootInit") != 0 {
        std::process::exit(1);
    }

    /* Create sparse SUNMatrix for use in linear solves */
    let A = SUNSparseMatrix(NEQ, NEQ, NNZ, SUN_CSC_MAT, &ctx);
    if check_retval_null(&A, "SUNSparseMatrix") != 0 {
        std::process::exit(1);
    }
    let A = A.unwrap();

    /* Create KLU solver object for use by CVode */
    let LS = SUNLinSol_KLU(&y, &A, &ctx);
    if check_retval_null(&LS, "SUNLinSol_KLU") != 0 {
        std::process::exit(1);
    }
    let LS = LS.unwrap();

    /* Attach the matrix and linear solver */
    let retval = CVodeSetLinearSolver(&cv, &LS, Some(&A));
    if check_retval(retval, "CVodeSetLinearSolver") != 0 {
        std::process::exit(1);
    }

    /* Set the user-supplied Jacobian routine Jac */
    let retval = CVodeSetJacFn(&cv, Some(Jac));
    if check_retval(retval, "CVodeSetJacFn") != 0 {
        std::process::exit(1);
    }

    /* In loop, call CVode, print results, and test for error.
    Break out of loop when NOUT preset output times have been reached.  */
    print!(" \n3-species kinetics problem\n\n");

    let mut iout = 0;
    let mut tout = T1;
    let mut t: sunrealtype = 0.0;
    loop {
        let retval = CVode(&cv, tout, &y, &mut t, CV_NORMAL);
        PrintOutput(t, Ith(&y, 1), Ith(&y, 2), Ith(&y, 3));

        if retval == CV_ROOT_RETURN {
            let retvalr = CVodeGetRootInfo(&cv, &mut rootsfound);
            if check_retval(retvalr, "CVodeGetRootInfo") != 0 {
                std::process::exit(1);
            }
            PrintRootInfo(rootsfound[0], rootsfound[1]);
        }

        if check_retval(retval, "CVode") != 0 {
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
    PrintFinalStats(&cv);

    /* Free memory (the Rust port frees on drop; the calls are kept so the
    translation reads like its parent) */
    N_VDestroy(y);
    N_VDestroy(abstol);
    let mut cvode_mem = cvode_mem;
    CVodeFree(&mut cvode_mem); /* Free CVODES memory */
    SUNLinSolFree(Some(LS));
    SUNMatDestroy(A);
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
    /* State at which to evaluate the Jacobian */
    let yval = N_VGetArrayPointer(y).expect("N_VGetArrayPointer");

    /* J is stored in CSC format:
    data    = non-zero matrix entries stored column-wise (length NNZ)
    rowvals = row index for each non-zero matrix entry (length NNZ)
    colptrs = i-th entry is the index in data where the first non-zero matrix
              entry of the i-th column is stored (length NEQ + 1)

    The C takes three pointers into the same matrix; here they are three
    fields behind one borrow, because taking them separately would be a
    second mutable borrow of the same content. */
    let mut m = SUNSparseMatrix_Content(J);
    let m = &mut *m;
    let (data, rowvals, colptrs) = (&mut m.data, &mut m.indexvals, &mut m.indexptrs);

    /* first column entries start at data[0], two entries (rows 0 and 1) */
    colptrs[0] = 0;

    rowvals[0] = 0;
    data[0] = -0.04;

    rowvals[1] = 1;
    data[1] = 0.04;

    /* second column entries start at data[2], three entries (rows 0, 1, and 2) */
    colptrs[1] = 2;

    rowvals[2] = 0;
    data[2] = 1.0e4 * yval[2];

    rowvals[3] = 1;
    data[3] = (-1.0e4 * yval[2]) - (6.0e7 * yval[1]);

    rowvals[4] = 2;
    data[4] = 6.0e7 * yval[1];

    /* third column entries start at data[5], two entries (rows 0 and 1) */
    colptrs[2] = 5;

    rowvals[5] = 0;
    data[5] = 1.0e4 * yval[1];

    rowvals[6] = 1;
    data[6] = -1.0e4 * yval[1];

    /* number of non-zeros */
    colptrs[3] = 7;

    0
}

/*
 *-------------------------------
 * Private helper functions
 *-------------------------------
 */

fn PrintOutput(t: sunrealtype, y1: sunrealtype, y2: sunrealtype, y3: sunrealtype) {
    /* C: printf("At t = %0.4e      y =%14.6e  %14.6e  %14.6e\n", t, y1, y2, y3) */
    print!(
        "At t = {}      y ={}  {}  {}\n",
        fmt_e(t, 4),
        fmt_ew(y1, 14, 6),
        fmt_ew(y2, 14, 6),
        fmt_ew(y3, 14, 6)
    );
}

fn PrintRootInfo(root_f1: i32, root_f2: i32) {
    /* C: printf("    rootsfound[] = %3d %3d\n", root_f1, root_f2) */
    print!("    rootsfound[] = {:>3} {:>3}\n", root_f1, root_f2);
}

/*
 * Get and print some final statistics
 */

fn PrintFinalStats(cvode_mem: &CVodeMem) {
    let mut nst: i64 = 0;
    let mut nfe: i64 = 0;
    let mut nsetups: i64 = 0;
    let mut nje: i64 = 0;
    let mut nni: i64 = 0;
    let mut nnf: i64 = 0;
    let mut ncfn: i64 = 0;
    let mut netf: i64 = 0;
    let mut nge: i64 = 0;

    let retval = CVodeGetNumSteps(cvode_mem, &mut nst);
    check_retval(retval, "CVodeGetNumSteps");
    let retval = CVodeGetNumRhsEvals(cvode_mem, &mut nfe);
    check_retval(retval, "CVodeGetNumRhsEvals");
    let retval = CVodeGetNumLinSolvSetups(cvode_mem, &mut nsetups);
    check_retval(retval, "CVodeGetNumLinSolvSetups");
    let retval = CVodeGetNumErrTestFails(cvode_mem, &mut netf);
    check_retval(retval, "CVodeGetNumErrTestFails");
    let retval = CVodeGetNumNonlinSolvIters(cvode_mem, &mut nni);
    check_retval(retval, "CVodeGetNumNonlinSolvIters");
    let retval = CVodeGetNumNonlinSolvConvFails(cvode_mem, &mut nnf);
    check_retval(retval, "CVodeGetNumNonlinSolvConvFails");
    let retval = CVodeGetNumStepSolveFails(cvode_mem, &mut ncfn);
    check_retval(retval, "CVodeGetNumStepSolveFails");

    let retval = CVodeGetNumJacEvals(cvode_mem, &mut nje);
    check_retval(retval, "CVodeGetNumJacEvals");

    let retval = CVodeGetNumGEvals(cvode_mem, &mut nge);
    check_retval(retval, "CVodeGetNumGEvals");

    print!("\nFinal Statistics:\n");
    /* C: printf("nst = %-6ld nfe = %-6ld nsetups = %-6ld nje = %ld\n", ...) */
    print!(
        "nst = {:<6} nfe = {:<6} nsetups = {:<6} nje = {}\n",
        nst, nfe, nsetups, nje
    );
    /* C: printf("nni = %-6ld nnf = %-6ld netf = %-6ld    ncfn = %-6ld nge = %ld\n\n", ...) */
    print!(
        "nni = {:<6} nnf = {:<6} netf = {:<6}    ncfn = {:<6} nge = {}\n\n",
        nni, nnf, netf, ncfn, nge
    );
}

/*
 * Check function return value... (C check_retval; the C void-pointer/opt
 * dispatch splits into two typed helpers here)
 */

fn check_retval(retval: i32, funcname: &str) -> i32 {
    if retval < 0 {
        eprint!(
            "\nSUNDIALS_ERROR: {}() failed with retval = {}\n\n",
            funcname, retval
        );
        return 1;
    }
    0
}

fn check_retval_null<T>(returnvalue: &Option<T>, funcname: &str) -> i32 {
    if returnvalue.is_none() {
        eprint!(
            "\nSUNDIALS_ERROR: {}() failed - returned NULL pointer\n\n",
            funcname
        );
        return 1;
    }
    0
}
