//! Port of `examples/cvode/serial/cvRoberts_block_klu.c`.
//!
//! Example problem:
//!
//! The following is a simple example problem, with the coding needed
//! for its solution by CVODE. The problem is from chemical kinetics,
//! and consists of the following three rate equations:
//!    dy1/dt = -.04*y1 + 1.e4*y2*y3
//!    dy2/dt = .04*y1 - 1.e4*y2*y3 - 3.e7*(y2)^2
//!    dy3/dt = 3.e7*(y2)^2
//! on the interval from t = 0.0 to t = 4.e10, with initial
//! conditions: y1 = 1.0, y2 = y3 = 0. The problem is stiff.
//! This program solves the problem with the BDF method, Newton
//! iteration with the KLU sparse direct linear solver, and a
//! user-supplied Jacobian routine. It uses a scalar relative tolerance
//! and a vector absolute tolerance. Output is printed in decades from
//! t = .4 to t = 4.e10. Run statistics (optional outputs) are printed
//! at the end.
//!
//! The problem is solved simultaneously for a group of independent
//! systems; the number of groups is the first command-line argument
//! (default 1000).
//!
//! **Solver note.** `SUNLinSol_KLU` here is
//! [`sundials_core::sunlinsol_klu`], whose factorization is the
//! independent pure-Rust sparse LU in `sundials_core::sundials_sparse_lu`
//! rather than SuiteSparse KLU (LGPL, and unreachable without FFI). The
//! printed digits therefore differ from the C original; see
//! `differences/ATTRIBUTION.md`.

#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

use std::any::Any;

use cvode_rs::prelude::*;

/* User-defined vector accessor helper: Ith
   (C macro `Ith(v,i)` = `NV_Ith_S(v,i-1)`; i is 1-based). */

fn Ith(v: &N_Vector, i: usize) -> sunrealtype {
    N_VGetArrayPointer(v).expect("N_VGetArrayPointer")[i - 1]
}

fn Ith_set(v: &N_Vector, i: usize, x: sunrealtype) {
    N_VGetArrayPointer(v).expect("N_VGetArrayPointer")[i - 1] = x;
}

/* Problem Constants */

const GROUPSIZE: sunindextype = 3; /* number of equations per group */
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

/* Type : UserData */

struct UserData {
    ngroups: sunindextype,
    neq: sunindextype,
}

/*
 *-------------------------------
 * Main Program
 *-------------------------------
 */

fn main() {
    let args: Vec<String> = std::env::args().collect();

    /* Parse command line arguments */
    let ngroups: sunindextype = if args.len() > 1 {
        atoi(&args[1]) as sunindextype
    } else {
        1000
    };
    let neq = ngroups * GROUPSIZE;

    let udata = UserData { ngroups, neq };

    let mut sunctx: Option<SUNContext> = None;
    let retval = SUNContext_Create(SUN_COMM_NULL, &mut sunctx);
    if check_retval(retval, "CVodeInit") != 0 {
        std::process::exit(1);
    }
    let ctx = sunctx.clone().unwrap();

    /* Create serial vector of length neq for I.C. and abstol */
    let y = N_VNew_Serial(neq, &ctx);
    if check_retval_null(&y, "N_VNew_Serial") != 0 {
        std::process::exit(1);
    }
    let y = y.unwrap();
    let abstol = N_VNew_Serial(neq, &ctx);
    if check_retval_null(&abstol, "N_VNew_Serial") != 0 {
        std::process::exit(1);
    }
    let abstol = abstol.unwrap();

    /* Initialize y */
    let mut groupj = 0;
    while groupj < neq {
        Ith_set(&y, (1 + groupj) as usize, Y1);
        Ith_set(&y, (2 + groupj) as usize, Y2);
        Ith_set(&y, (3 + groupj) as usize, Y3);
        groupj += GROUPSIZE;
    }

    /* Set the scalar relative tolerance */
    let reltol = RTOL;

    /* Set the vector absolute tolerance */
    let mut groupj = 0;
    while groupj < neq {
        Ith_set(&abstol, (1 + groupj) as usize, ATOL1);
        Ith_set(&abstol, (2 + groupj) as usize, ATOL2);
        Ith_set(&abstol, (3 + groupj) as usize, ATOL3);
        groupj += GROUPSIZE;
    }

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

    /* Call CVodeSetUserData to attach the user data structure */
    let retval = CVodeSetUserData(&cv, Some(Box::new(udata)));
    if check_retval(retval, "CVodeSetUserData") != 0 {
        std::process::exit(1);
    }

    /* Call CVodeSVtolerances to specify the scalar relative tolerance
     * and vector absolute tolerances */
    let retval = CVodeSVtolerances(&cv, reltol, &abstol);
    if check_retval(retval, "CVodeSVtolerances") != 0 {
        std::process::exit(1);
    }

    /* Create sparse SUNMatrix for use in linear solves */
    let nnz = GROUPSIZE * GROUPSIZE * ngroups;
    let A = SUNSparseMatrix(neq, neq, nnz, SUN_CSR_MAT, &ctx);
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

    /* Call CVodeSetLinearSolver to attach the matrix and linear solver to CVode */
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
    print!(" \nGroup of independent 3-species kinetics problems\n\n");
    print!("number of groups = {}\n\n", ngroups);

    let mut iout = 0;
    let mut tout = T1;
    let mut t: sunrealtype = 0.0;
    loop {
        let retval = CVode(&cv, tout, &y, &mut t, CV_NORMAL);

        for groupj in 0..1 {
            print!("group {}: ", groupj);
            PrintOutput(
                t,
                Ith(&y, (1 + GROUPSIZE * groupj) as usize),
                Ith(&y, (2 + GROUPSIZE * groupj) as usize),
                Ith(&y, (3 + GROUPSIZE * groupj) as usize),
            );
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
    CVodeFree(&mut cvode_mem);
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

fn f(_t: sunrealtype, y: &N_Vector, ydot: &N_Vector, user_data: &mut Option<Box<dyn Any>>) -> i32 {
    let neq = user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<UserData>())
        .expect("UserData")
        .neq;

    let mut groupj = 0;
    while groupj < neq {
        let y1 = Ith(y, (1 + groupj) as usize);
        let y2 = Ith(y, (2 + groupj) as usize);
        let y3 = Ith(y, (3 + groupj) as usize);

        let yd1 = -0.04 * y1 + 1.0e4 * y2 * y3;
        Ith_set(ydot, (1 + groupj) as usize, yd1);
        let yd3 = 3.0e7 * y2 * y2;
        Ith_set(ydot, (3 + groupj) as usize, yd3);
        Ith_set(ydot, (2 + groupj) as usize, -yd1 - yd3);

        groupj += GROUPSIZE;
    }

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
    user_data: &mut Option<Box<dyn Any>>,
    _tmp1: &N_Vector,
    _tmp2: &N_Vector,
    _tmp3: &N_Vector,
) -> i32 {
    let ngroups = user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<UserData>())
        .expect("UserData")
        .ngroups;
    let ydata = N_VGetArrayPointer(y).expect("N_VGetArrayPointer");
    let nnzper = GROUPSIZE * GROUPSIZE;

    SUNMatZero(J);

    /* One borrow for all three arrays: taking them separately would be a
    second mutable borrow of the same matrix content. The C advances
    `rowptrs` past its first element instead; here the offset is written
    out as `1 + ...`. */
    let mut m = SUNSparseMatrix_Content(J);
    let m = &mut *m;
    let (data, colvals, rowptrs) = (&mut m.data, &mut m.indexvals, &mut m.indexptrs);

    rowptrs[0] = 0;
    for groupj in 0..ngroups {
        /* get y values */
        let y2 = ydata[(GROUPSIZE * groupj + 1) as usize];
        let y3 = ydata[(GROUPSIZE * groupj + 2) as usize];

        let r = (1 + GROUPSIZE * groupj) as usize;
        let d = (nnzper * groupj) as usize;
        let c0 = GROUPSIZE * groupj;

        /* there are 3 entries per row */
        rowptrs[r] = 3 + nnzper * groupj;
        rowptrs[r + 1] = 6 + nnzper * groupj;
        rowptrs[r + 2] = 9 + nnzper * groupj;

        /* first row of block */
        data[d] = -0.04;
        data[d + 1] = 1.0e4 * y3;
        data[d + 2] = 1.0e4 * y2;
        colvals[d] = c0;
        colvals[d + 1] = c0 + 1;
        colvals[d + 2] = c0 + 2;

        /* second row of block */
        data[d + 3] = 0.04;
        data[d + 4] = (-1.0e4 * y3) - (6.0e7 * y2);
        data[d + 5] = -1.0e4 * y2;
        colvals[d + 3] = c0;
        colvals[d + 4] = c0 + 1;
        colvals[d + 5] = c0 + 2;

        /* third row of block */
        data[d + 6] = ZERO;
        data[d + 7] = 6.0e7 * y2;
        data[d + 8] = ZERO;
        colvals[d + 6] = c0;
        colvals[d + 7] = c0 + 1;
        colvals[d + 8] = c0 + 2;
    }

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
