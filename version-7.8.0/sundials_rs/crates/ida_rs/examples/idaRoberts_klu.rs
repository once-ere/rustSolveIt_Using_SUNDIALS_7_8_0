//! Port of `examples/ida/serial/idaRoberts_klu.c`.
//!
//! **Solver note.** `SUNLinSol_KLU` here is
//! [`sundials_core::sunlinsol_klu`], backed by the independent pure-Rust
//! sparse LU rather than SuiteSparse KLU. The Jacobian is `SUN_CSR_MAT`, so
//! the solve goes through the transpose path. See
//! `differences/ATTRIBUTION.md`.
//!
//! This simple example problem for IDA, due to Robertson,
//! is from chemical kinetics, and consists of the following three
//! equations:
//!
//!      dy1/dt = -.04*y1 + 1.e4*y2*y3
//!      dy2/dt = .04*y1 - 1.e4*y2*y3 - 3.e7*y2**2
//!         0   = y1 + y2 + y3 - 1
//!
//! on the interval from t = 0.0 to t = 4.e10, with initial
//! conditions: y1 = 1, y2 = y3 = 0.
//!
//! While integrating the system, we also use the rootfinding
//! feature to find the points at which y1 = 1e-4 or at which
//! y3 = 0.01.
//!
//! The problem is solved with IDA using the DENSE linear
//! solver, with a user-supplied Jacobian. Output is printed at
//! t = .4, 4, 40, ..., 4e10.

#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

use std::any::Any;

use ida_rs::prelude::*;

/* Problem Constants */

const NEQ: sunindextype = 3;
const NOUT: i32 = 12;

const ZERO: sunrealtype = 0.0;
const ONE: sunrealtype = 1.0;


/*
 *--------------------------------------------------------------------
 * Main Program
 *--------------------------------------------------------------------
 */

fn main() {
    let mut retval: i32;
    let mut rootsfound: [i32; 2] = [0; 2];

    /* Create SUNDIALS context */
    let mut sunctx: Option<SUNContext> = None;
    retval = SUNContext_Create(SUN_COMM_NULL, &mut sunctx);
    if check_retval(Some(retval), "SUNContext_Create", 1) != 0 {
        std::process::exit(1);
    }
    let ctx = sunctx.as_ref().expect("sunctx").clone();

    /* Allocate N-vectors. */
    let yy = N_VNew_Serial(NEQ, &ctx);
    if check_retval(yy.as_ref().map(|_| 0), "N_VNew_Serial", 0) != 0 {
        std::process::exit(1);
    }
    let yy = yy.expect("yy");
    let yp = N_VClone(&yy);
    if check_retval(yp.as_ref().map(|_| 0), "N_VNew_Serial", 0) != 0 {
        std::process::exit(1);
    }
    let yp = yp.expect("yp");
    let avtol = N_VClone(&yy);
    if check_retval(avtol.as_ref().map(|_| 0), "N_VNew_Serial", 0) != 0 {
        std::process::exit(1);
    }
    let avtol = avtol.expect("avtol");

    /* Create and initialize  y, y', and absolute tolerance vectors. */
    {
        let mut yval = N_VGetArrayPointer(&yy).expect("N_VGetArrayPointer");
        yval[0] = ONE;
        yval[1] = ZERO;
        yval[2] = ZERO;
    }

    {
        let mut ypval = N_VGetArrayPointer(&yp).expect("N_VGetArrayPointer");
        ypval[0] = -0.04;
        ypval[1] = 0.04;
        ypval[2] = ZERO;
    }

    let rtol: sunrealtype = 1.0e-4;

    {
        let mut atval = N_VGetArrayPointer(&avtol).expect("N_VGetArrayPointer");
        atval[0] = 1.0e-8;
        atval[1] = 1.0e-6;
        atval[2] = 1.0e-6;
    }

    /* Integration limits */
    let t0: sunrealtype = ZERO;
    let tout1: sunrealtype = 0.4;

    PrintHeader(rtol, &avtol, &yy);

    /* Call IDACreate and IDAInit to initialize IDA memory */
    let mut mem_opt = IDACreate(&ctx);
    if check_retval(mem_opt.as_ref().map(|_| 0), "IDACreate", 0) != 0 {
        std::process::exit(1);
    }
    let mem = mem_opt.as_ref().expect("mem").clone();
    retval = IDAInit(&mem, resrob, t0, &yy, &yp);
    if check_retval(Some(retval), "IDAInit", 1) != 0 {
        std::process::exit(1);
    }
    /* Call IDASVtolerances to set tolerances */
    retval = IDASVtolerances(&mem, rtol, &avtol);
    if check_retval(Some(retval), "IDASVtolerances", 1) != 0 {
        std::process::exit(1);
    }

    /* Free avtol */
    N_VDestroy(avtol);

    /* Call IDARootInit to specify the root function grob with 2 components */
    retval = IDARootInit(&mem, 2, Some(grob));
    if check_retval(Some(retval), "IDARootInit", 1) != 0 {
        std::process::exit(1);
    }

    /* Create dense SUNMatrix for use in linear solves */
    let nnz: sunindextype = NEQ * NEQ;
    let A = SUNSparseMatrix(NEQ, NEQ, nnz, SUN_CSR_MAT, &ctx);
    if check_retval(A.as_ref().map(|_| 0), "SUNSparseMatrix", 0) != 0 {
        std::process::exit(1);
    }
    let A = A.expect("A");

    /* Create dense SUNLinearSolver object */
    let LS = SUNLinSol_KLU(&yy, &A, &ctx);
    if check_retval(LS.as_ref().map(|_| 0), "SUNLinSol_KLU", 0) != 0 {
        std::process::exit(1);
    }
    let LS = LS.expect("LS");

    /* Attach the matrix and linear solver */
    retval = IDASetLinearSolver(&mem, &LS, Some(&A));
    if check_retval(Some(retval), "IDASetLinearSolver", 1) != 0 {
        std::process::exit(1);
    }

    /* Set the user-supplied Jacobian routine */
    retval = IDASetJacFn(&mem, Some(jacrobCSR));
    if check_retval(Some(retval), "IDASetJacFn", 1) != 0 {
        std::process::exit(1);
    }

    /* In loop, call IDASolve, print results, and test for error.
    Break out of loop when NOUT preset output times have been reached. */

    let mut iout: i32 = 0;
    let mut tout: sunrealtype = tout1;
    let mut tret: sunrealtype = 0.0;
    loop {
        retval = IDASolve(&mem, tout, &mut tret, &yy, &yp, IDA_NORMAL);

        PrintOutput(&mem, tret, &yy);

        if check_retval(Some(retval), "IDASolve", 1) != 0 {
            std::process::exit(1);
        }

        if retval == IDA_ROOT_RETURN {
            let retvalr = IDAGetRootInfo(&mem, &mut rootsfound);
            check_retval(Some(retvalr), "IDAGetRootInfo", 1);
            PrintRootInfo(rootsfound[0], rootsfound[1]);
        }

        if retval == IDA_SUCCESS {
            iout += 1;
            tout *= 10.0;
        }

        if iout == NOUT {
            break;
        }
    }

    PrintFinalStats(&mem);

    /* Free memory */
    IDAFree(&mut mem_opt);
    let _ = SUNLinSolFree(Some(LS));
    SUNMatDestroy(A);
    N_VDestroy(yy);
    N_VDestroy(yp);
    let _ = SUNContext_Free(&mut sunctx);

}

/*
 *--------------------------------------------------------------------
 * Functions called by IDA
 *--------------------------------------------------------------------
 */

/*
 * Define the system residual function.
 */

fn resrob(
    _tres: sunrealtype,
    yy: &N_Vector,
    yp: &N_Vector,
    rr: &N_Vector,
    _user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    let yval = N_VGetArrayPointer(yy).expect("N_VGetArrayPointer");
    let ypval = N_VGetArrayPointer(yp).expect("N_VGetArrayPointer");
    let mut rval = N_VGetArrayPointer(rr).expect("N_VGetArrayPointer");

    rval[0] = -0.04 * yval[0] + 1.0e4 * yval[1] * yval[2];
    rval[1] = -rval[0] - 3.0e7 * yval[1] * yval[1] - ypval[1];
    rval[0] -= ypval[0];
    rval[2] = yval[0] + yval[1] + yval[2] - ONE;

    0
}

/*
 * Root function routine. Compute functions g_i(t,y) for i = 0,1.
 */

fn grob(
    _t: sunrealtype,
    yy: &N_Vector,
    _yp: &N_Vector,
    gout: &mut [sunrealtype],
    _user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    let yval = N_VGetArrayPointer(yy).expect("N_VGetArrayPointer");
    let y1 = yval[0];
    let y3 = yval[2];
    gout[0] = y1 - 0.0001;
    gout[1] = y3 - 0.01;

    0
}

/*
 * Define the Jacobian function, in compressed sparse row form.
 */

fn jacrobCSR(
    _tt: sunrealtype,
    cj: sunrealtype,
    yy: &N_Vector,
    _yp: &N_Vector,
    _resvec: &N_Vector,
    JJ: &SUNMatrix,
    _user_data: &mut Option<Box<dyn Any>>,
    _tempv1: &N_Vector,
    _tempv2: &N_Vector,
    _tempv3: &N_Vector,
) -> i32 {
    let yval = N_VGetArrayPointer(yy).expect("N_VGetArrayPointer");

    SUNMatZero(JJ);

    /* One borrow for all three arrays: taking them separately would be a
    second mutable borrow of the same matrix content. */
    let mut m = SUNSparseMatrix_Content(JJ);
    let m = &mut *m;
    let (data, colvals, rowptrs) = (&mut m.data, &mut m.indexvals, &mut m.indexptrs);

    rowptrs[0] = 0;
    rowptrs[1] = 3;
    rowptrs[2] = 6;
    rowptrs[3] = 9;

    /* row 0 */
    data[0] = -0.04 - cj;
    colvals[0] = 0;
    data[1] = 1.0e4 * yval[2];
    colvals[1] = 1;
    data[2] = 1.0e4 * yval[1];
    colvals[2] = 2;

    /* row 1 */
    data[3] = 0.04;
    colvals[3] = 0;
    data[4] = (-1.0e4 * yval[2]) - (6.0e7 * yval[1]) - cj;
    colvals[4] = 1;
    data[5] = -1.0e4 * yval[1];
    colvals[5] = 2;

    /* row 2 */
    data[6] = ONE;
    colvals[6] = 0;
    data[7] = ONE;
    colvals[7] = 1;
    data[8] = ONE;
    colvals[8] = 2;

    0
}

/*
 *--------------------------------------------------------------------
 * Private functions
 *--------------------------------------------------------------------
 */

/*
 * Print first lines of output (problem description)
 */

fn PrintHeader(rtol: sunrealtype, avtol: &N_Vector, y: &N_Vector) {
    let atval = N_VGetArrayPointer(avtol).expect("N_VGetArrayPointer");
    let yval = N_VGetArrayPointer(y).expect("N_VGetArrayPointer");

    print!("\nidaRoberts_klu: Robertson kinetics DAE serial example problem for IDA.\n");
    print!("               Three equation chemical kinetics problem.\n\n");
    print!("Linear solver: KLU, with user-supplied Jacobian.\n");
    print!(
        "Tolerance parameters:  rtol = {}   atol = {} {} {} \n",
        fmt_g(rtol, 6),
        fmt_g(atval[0], 6),
        fmt_g(atval[1], 6),
        fmt_g(atval[2], 6)
    );
    print!(
        "Initial conditions y0 = ({} {} {})\n",
        fmt_g(yval[0], 6),
        fmt_g(yval[1], 6),
        fmt_g(yval[2], 6)
    );
    print!("Constraints and id not used.\n\n");
    print!("-----------------------------------------------------------------------\n");
    print!("  t             y1           y2           y3");
    print!("      | nst  k      h\n");
    print!("-----------------------------------------------------------------------\n");
}

/*
 * Print Output
 */

fn PrintOutput(mem: &IDAMem, t: sunrealtype, y: &N_Vector) {
    let mut retval: i32;
    let mut kused: i32 = 0;
    let mut nst: i64 = 0;
    let mut hused: sunrealtype = 0.0;

    /* C keeps the raw `yval` pointer live across the IDAGet* calls; the
    port snapshots the three components instead so no vector borrow is
    held across a library call. */
    let yval: [sunrealtype; 3] = {
        let d = N_VGetArrayPointer(y).expect("N_VGetArrayPointer");
        [d[0], d[1], d[2]]
    };

    retval = IDAGetLastOrder(mem, &mut kused);
    check_retval(Some(retval), "IDAGetLastOrder", 1);
    retval = IDAGetNumSteps(mem, &mut nst);
    check_retval(Some(retval), "IDAGetNumSteps", 1);
    retval = IDAGetLastStep(mem, &mut hused);
    check_retval(Some(retval), "IDAGetLastStep", 1);

    print!(
        "{} {} {} {} | {:3}  {:1} {}\n",
        fmt_ew(t, 10, 4),
        fmt_ew(yval[0], 12, 4),
        fmt_ew(yval[1], 12, 4),
        fmt_ew(yval[2], 12, 4),
        nst,
        kused,
        fmt_ew(hused, 12, 4)
    );
}

fn PrintRootInfo(root_f1: i32, root_f2: i32) {
    print!("    rootsfound[] = {:3} {:3}\n", root_f1, root_f2);
}

/*
 * Check function return value...
 *   opt == 0 means SUNDIALS function allocates memory so check if
 *            returned NULL pointer
 *   opt == 1 means SUNDIALS function returns an integer value so check if
 *            retval < 0
 *   opt == 2 means function allocates memory so check if returned
 *            NULL pointer
 */

fn check_retval(returnvalue: Option<i32>, funcname: &str, opt: i32) -> i32 {
    /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
    if opt == 0 && returnvalue.is_none() {
        eprint!(
            "\nSUNDIALS_ERROR: {}() failed - returned NULL pointer\n\n",
            funcname
        );
        return 1;
    } else if opt == 1 {
        /* Check if retval < 0 */
        let retval = returnvalue.expect("retval");
        if retval < 0 {
            eprint!(
                "\nSUNDIALS_ERROR: {}() failed with retval = {}\n\n",
                funcname, retval
            );
            return 1;
        }
    } else if opt == 2 && returnvalue.is_none() {
        /* Check if function returned NULL pointer - no memory allocated */
        eprint!(
            "\nMEMORY_ERROR: {}() failed - returned NULL pointer\n\n",
            funcname
        );
        return 1;
    }

    0
}

/*
 * Print final run statistics.
 */

fn PrintFinalStats(mem: &IDAMem) {
    let mut nst: i64 = 0;
    let mut nni: i64 = 0;
    let mut nnf: i64 = 0;
    let mut nje: i64 = 0;
    let mut nre: i64 = 0;
    let mut netf: i64 = 0;
    let mut ncfn: i64 = 0;
    let mut nge: i64 = 0;

    let retval = IDAGetNumSteps(mem, &mut nst);
    check_retval(Some(retval), "IDAGetNumSteps", 1);
    let retval = IDAGetNumResEvals(mem, &mut nre);
    check_retval(Some(retval), "IDAGetNumResEvals", 1);
    let retval = IDAGetNumJacEvals(mem, &mut nje);
    check_retval(Some(retval), "IDAGetNumJacEvals", 1);
    let retval = IDAGetNumNonlinSolvIters(mem, &mut nni);
    check_retval(Some(retval), "IDAGetNumNonlinSolvIters", 1);
    let retval = IDAGetNumErrTestFails(mem, &mut netf);
    check_retval(Some(retval), "IDAGetNumErrTestFails", 1);
    let retval = IDAGetNumNonlinSolvConvFails(mem, &mut nnf);
    check_retval(Some(retval), "IDAGetNumNonlinSolvConvFails", 1);
    let retval = IDAGetNumStepSolveFails(mem, &mut ncfn);
    check_retval(Some(retval), "IDAGetNumStepSolveFails", 1);
    let retval = IDAGetNumGEvals(mem, &mut nge);
    check_retval(Some(retval), "IDAGetNumGEvals", 1);

    print!("\nFinal Run Statistics: \n\n");
    print!("Number of steps                    = {}\n", nst);
    print!("Number of residual evaluations     = {}\n", nre);
    print!("Number of Jacobian evaluations     = {}\n", nje);
    print!("Number of nonlinear iterations     = {}\n", nni);
    print!("Number of error test failures      = {}\n", netf);
    print!("Number of nonlinear conv. failures = {}\n", nnf);
    print!("Number of step solver failures     = {}\n", ncfn);
    print!("Number of root fn. evaluations     = {}\n", nge);
}
