/* -----------------------------------------------------------------
 * Rust port of examples/cvodes/serial/cvsRoberts_dns_constraints.c
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
 * user-supplied Jacobian routine. It uses a scalar relative tolerance
 * and a vector absolute tolerance. The constraint y_i >= 0 is
 * posed for all components. Output is printed in decades
 * from t = .4 to t = 4.e10. Run statistics (optional outputs)
 * are printed at the end.
 * -----------------------------------------------------------------*/

#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

use std::any::Any;

use cvodes_rs::prelude::*;

/* User-defined vector and matrix accessor helpers: Ith, IJth
   (C macros `Ith(v,i)` = `NV_Ith_S(v,i-1)` and
   `IJth(A,i,j)` = `SM_ELEMENT_D(A,i-1,j-1)`; i, j are 1-based). */

fn Ith(v: &N_Vector, i: usize) -> sunrealtype {
    N_VGetArrayPointer(v).expect("N_VGetArrayPointer")[i - 1]
}

fn Ith_set(v: &N_Vector, i: usize, x: sunrealtype) {
    N_VGetArrayPointer(v).expect("N_VGetArrayPointer")[i - 1] = x;
}

fn IJth_set(A: &SUNMatrix, i: sunindextype, j: sunindextype, x: sunrealtype) {
    SM_ELEMENT_D_set(A, i - 1, j - 1, x);
}

/* Problem Constants */

const NEQ: sunindextype = 3; /* number of equations  */
const Y1: sunrealtype = 1.0; /* initial y components */
const Y2: sunrealtype = 0.0;
const Y3: sunrealtype = 0.0;
const RTOL: sunrealtype = 1.0e-4; /* scalar relative tolerance            */
const ATOL1: sunrealtype = 1.0e-6; /* vector absolute tolerance components */
const ATOL2: sunrealtype = 1.0e-11;
const ATOL3: sunrealtype = 1.0e-5;
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
    let mut rootsfound = [0i32; 2];

    /* Create the SUNDIALS context */
    let mut sunctx: Option<SUNContext> = None;
    let retval = SUNContext_Create(SUN_COMM_NULL, &mut sunctx);
    if check_retval(retval, "SUNContext_Create") != 0 {
        std::process::exit(1);
    }
    let ctx = sunctx.as_ref().expect("SUNContext").clone();

    /* Initial conditions */
    let y = N_VNew_Serial(NEQ, &ctx);
    if check_retval_null(&y, "N_VNew_Serial") != 0 {
        std::process::exit(1);
    }
    let y = y.expect("N_VNew_Serial");

    /* Initialize y */
    Ith_set(&y, 1, Y1);
    Ith_set(&y, 2, Y2);
    Ith_set(&y, 3, Y3);

    /* Set the vector absolute tolerance */
    let abstol = N_VNew_Serial(NEQ, &ctx);
    if check_retval_null(&abstol, "N_VNew_Serial") != 0 {
        std::process::exit(1);
    }
    let abstol = abstol.expect("N_VNew_Serial");

    Ith_set(&abstol, 1, ATOL1);
    Ith_set(&abstol, 2, ATOL2);
    Ith_set(&abstol, 3, ATOL3);

    /* Set constraints to all 1's for nonnegative solution values. */
    let constraints = N_VNew_Serial(NEQ, &ctx);
    if check_retval_null(&constraints, "N_VNew_Serial") != 0 {
        std::process::exit(1);
    }
    let constraints = constraints.expect("N_VNew_Serial");

    N_VConst(ONE, &constraints);

    /* Call CVodeCreate to create the solver memory and specify the
     * Backward Differentiation Formula */
    let mut cvode_mem = CVodeCreate(CV_BDF, &ctx);
    if check_retval_null(&cvode_mem, "CVodeCreate") != 0 {
        std::process::exit(1);
    }
    let cv = cvode_mem.clone().expect("CVodeCreate");

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

    /* Create dense SUNMatrix */
    let A = SUNDenseMatrix(NEQ, NEQ, &ctx);
    if check_retval_null(&A, "SUNDenseMatrix") != 0 {
        std::process::exit(1);
    }
    let A = A.expect("SUNDenseMatrix");

    /* Create dense SUNLinearSolver */
    let LS = SUNLinSol_Dense(&y, &A, &ctx);
    if check_retval_null(&LS, "SUNLinSol_Dense") != 0 {
        std::process::exit(1);
    }
    let LS = LS.expect("SUNLinSol_Dense");

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

    /* Call CVodeSetConstraints to initialize constraints */
    let retval = CVodeSetConstraints(&cv, Some(&constraints));
    if check_retval(retval, "CVodeSetConstraints") != 0 {
        std::process::exit(1);
    }
    N_VDestroy(constraints);

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

    /* check the solution error */
    let retval = check_ans(&y, t, RTOL, &abstol);

    /* Free memory */
    N_VDestroy(y); /* Free y vector */
    N_VDestroy(abstol); /* Free abstol vector */
    CVodeFree(&mut cvode_mem); /* Free CVODES memory */
    let _ = SUNLinSolFree(Some(LS)); /* Free the linear solver memory */
    SUNMatDestroy(A); /* Free the matrix memory */
    let _ = SUNContext_Free(&mut sunctx); /* Free the SUNDIALS context */

    if retval != 0 {
        std::process::exit(retval);
    }
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
    let mut nfeLS: i64 = 0;
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
    let retval = CVodeGetNumLinRhsEvals(cvode_mem, &mut nfeLS);
    check_retval(retval, "CVodeGetNumLinRhsEvals");

    let retval = CVodeGetNumGEvals(cvode_mem, &mut nge);
    check_retval(retval, "CVodeGetNumGEvals");

    print!("\nFinal Statistics:\n");
    /* C: printf("nst = %-6ld nfe = %-6ld nsetups = %-6ld nfeLS = %-6ld nje = %ld\n",
                 nst, nfe, nsetups, nfeLS, nje) */
    print!(
        "nst = {:<6} nfe = {:<6} nsetups = {:<6} nfeLS = {:<6} nje = {}\n",
        nst, nfe, nsetups, nfeLS, nje
    );
    /* C: printf("nni = %-6ld nnf = %-6ld netf = %-6ld    ncfn = %-6ld  nge = %ld\n\n",
                 nni, nnf, netf, ncfn, nge) */
    print!(
        "nni = {:<6} nnf = {:<6} netf = {:<6}    ncfn = {:<6}  nge = {}\n\n",
        nni, nnf, netf, ncfn, nge
    );
}

/*
 * Check function return value... (C check_retval; the C void-pointer/opt
 * polymorphism splits into two typed helpers with identical messages:
 *   check_retval_null = opt == 0 (NULL-pointer check)
 *   check_retval      = opt == 1 (retval < 0 check)
 */

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

/* compare the solution at the final time 4e10s to a reference solution computed
using a relative tolerance of 1e-8 and absolute tolerance of 1e-14 */
fn check_ans(y: &N_Vector, _t: sunrealtype, rtol: sunrealtype, atol: &N_Vector) -> i32 {
    /* create reference solution and error weight vectors */
    let refsol = N_VClone(y).expect("N_VClone");
    let ewt = N_VClone(y).expect("N_VClone");

    /* set the reference solution data */
    {
        let mut rd = N_VGetArrayPointer(&refsol).expect("N_VGetArrayPointer");
        rd[0] = 5.2083495894337328e-08;
        rd[1] = 2.0833399429795671e-13;
        rd[2] = 9.9999994791629776e-01;
    }

    /* compute the error weight vector, loosen atol */
    N_VAbs(&refsol, &ewt);
    N_VLinearSum(rtol, &ewt, 10.0, atol, &ewt);
    if N_VMin(&ewt) <= ZERO {
        eprint!("\nSUNDIALS_ERROR: check_ans failed - ewt <= 0\n\n");
        return -1;
    }
    N_VInv(&ewt, &ewt);

    /* compute the solution error */
    N_VLinearSum(ONE, y, -ONE, &refsol, &refsol);
    let err = N_VWrmsNorm(&refsol, &ewt);

    /* is the solution within the tolerances? */
    let passfail = if err < ONE { 0 } else { 1 };

    if passfail != 0 {
        /* C: fprintf(stdout, "\nSUNDIALS_WARNING: check_ans error=%g\n\n", err) */
        print!("\nSUNDIALS_WARNING: check_ans error={}\n\n", fmt_g(err, 6));
    }

    /* Free vectors */
    N_VDestroy(refsol);
    N_VDestroy(ewt);

    passfail
}
