//! Port of `examples/ida/serial/idaAnalytic_mels.c`.
//!
//! The following is a simple example problem with analytical
//! solution adapted from example 10.2 of Ascher & Petzold, "Computer
//! Methods for Ordinary Differential Equations and
//! Differential-Algebraic Equations," SIAM, 1998, page 267:
//!    x1'(t) = (1-alpha)/(t-2)*x1 - x1 + (alpha-1)*x2 + 2*exp(t)
//!         0 = (t+2)*x1 - (t+2)*exp(t)
//! for t in the interval [0.0, 1.0], with initial condition:
//!    x1(0) = 1   and   x2(0) = -1/2.
//! The problem has true solution
//!    x1(t) = exp(t)  and  x2(t) = exp(t)/(t-2)
//!
//! This program solves the problem with IDA using a custom
//! 'matrix-embedded' SUNLinearSolver. Output is printed
//! every 0.1 units of time (10 total).  Run statistics (optional
//! outputs) are printed at the end.

#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

use std::any::Any;

use ida_rs::prelude::*;

/* Main Program */
fn main() {
    /* general problem parameters */
    let T0: sunrealtype = 0.0; /* initial time */
    let Tf: sunrealtype = 1.0; /* final time */
    let dTout: sunrealtype = 0.1; /* time between outputs */
    let NEQ: sunindextype = 2; /* number of dependent vars. */
    let reltol: sunrealtype = 1.0e-4; /* tolerances */
    let abstol: sunrealtype = 1.0e-9;
    let alpha: sunrealtype = 10.0; /* stiffness parameter */

    /* general problem variables */
    let mut retval: i32; /* reusable error-checking flag */
    let (mut nst, mut nre, mut nni, mut netf, mut ncfn, mut nreLS) =
        (0i64, 0i64, 0i64, 0i64, 0i64, 0i64);

    /* C `main(int argc, char* argv[])` */
    let argv: Vec<String> = std::env::args().collect();
    let argc = argv.len() as i32;

    /* Initial diagnostics output */
    print!("\nAnalytical DAE test problem:\n");
    print!("    alpha = {}\n\n", fmt_g(alpha, 6));

    /* Create the SUNDIALS context object for this simulation */
    let mut sunctx: Option<SUNContext> = None;
    retval = SUNContext_Create(SUN_COMM_NULL, &mut sunctx);
    if check_retval(Some(retval), "SUNContext_Create", 1) != 0 {
        std::process::exit(1);
    }
    let ctx = sunctx.as_ref().expect("sunctx").clone();

    /* Initialize data structures */
    let yy = N_VNew_Serial(NEQ, &ctx); /* Create serial vector for solution */
    if check_retval(yy.as_ref().map(|_| 0), "N_VNew_Serial", 0) != 0 {
        std::process::exit(1);
    }
    let yy = yy.expect("yy");
    let yp = N_VClone(&yy); /* Create serial vector for solution derivative */
    if check_retval(yp.as_ref().map(|_| 0), "N_VClone", 0) != 0 {
        std::process::exit(1);
    }
    let yp = yp.expect("yp");
    analytical_solution(T0, &yy, &yp); /* Specify initial conditions */

    /* Call IDACreate and IDAInit to initialize IDA memory */
    let mut ida_mem_opt = IDACreate(&ctx);
    if check_retval(ida_mem_opt.as_ref().map(|_| 0), "IDACreate", 0) != 0 {
        std::process::exit(1);
    }
    let ida_mem = ida_mem_opt.as_ref().expect("ida_mem").clone();
    retval = IDAInit(&ida_mem, fres, T0, &yy, &yp);
    if check_retval(Some(retval), "IDAInit", 1) != 0 {
        std::process::exit(1);
    }

    /* Set routines */
    retval = IDASetUserData(&ida_mem, Some(Box::new(alpha)));
    if check_retval(Some(retval), "IDASetUserData", 1) != 0 {
        std::process::exit(1);
    }
    retval = IDASStolerances(&ida_mem, reltol, abstol);
    if check_retval(Some(retval), "IDASStolerances", 1) != 0 {
        std::process::exit(1);
    }

    /* Create custom matrix-embedded linear solver */
    let LS = MatrixEmbeddedLS(&ida_mem, &ctx);
    if check_retval(LS.as_ref().map(|_| 0), "MatrixEmbeddedLS", 0) != 0 {
        std::process::exit(1);
    }
    let LS = LS.expect("LS");

    /* Attach the linear solver */
    retval = IDASetLinearSolver(&ida_mem, &LS, None);
    if check_retval(Some(retval), "IDASetLinearSolver", 1) != 0 {
        std::process::exit(1);
    }

    /* Override any current settings with command-line options */
    retval = IDASetOptions(&ida_mem, None, None, argc, &argv);
    if check_retval(Some(retval), "IDASetOptions", 1) != 0 {
        std::process::exit(1);
    }

    /* In loop, call IDASolve, print results, and test for error.
    Stops when the final time has been reached. */
    let mut t: sunrealtype = T0;
    let mut tout: sunrealtype = T0 + dTout;
    print!("        t          x1         x2\n");
    print!("   ----------------------------------\n");
    while Tf - t > 1.0e-15 {
        retval = IDASolve(&ida_mem, tout, &mut t, &yy, &yp, IDA_NORMAL); /* call integrator */
        if check_retval(Some(retval), "IDASolve", 1) != 0 {
            std::process::exit(1);
        }
        /* access/print solution */
        {
            let yydata = NV_DATA_S(&yy);
            print!(
                "  {}  {}  {}\n",
                fmt_fw(t, 10, 6),
                fmt_fw(yydata[0], 10, 6),
                fmt_fw(yydata[1], 10, 6)
            );
        }
        if retval >= 0 {
            /* successful solve: update time */
            tout += dTout;
            tout = if tout > Tf { Tf } else { tout };
        } else {
            /* unsuccessful solve: break */
            eprint!("Solver failure, stopping integration\n");
            break;
        }
    }
    print!("   ----------------------------------\n");

    /* Get/print some final statistics on how the solve progressed */
    retval = IDAGetNumSteps(&ida_mem, &mut nst);
    check_retval(Some(retval), "IDAGetNumSteps", 1);
    retval = IDAGetNumResEvals(&ida_mem, &mut nre);
    check_retval(Some(retval), "IDAGetNumResEvals", 1);
    retval = IDAGetNumNonlinSolvIters(&ida_mem, &mut nni);
    check_retval(Some(retval), "IDAGetNumNonlinSolvIters", 1);
    retval = IDAGetNumErrTestFails(&ida_mem, &mut netf);
    check_retval(Some(retval), "IDAGetNumErrTestFails", 1);
    retval = IDAGetNumNonlinSolvConvFails(&ida_mem, &mut ncfn);
    check_retval(Some(retval), "IDAGetNumNonlinSolvConvFails", 1);
    retval = IDAGetNumLinResEvals(&ida_mem, &mut nreLS);
    check_retval(Some(retval), "IDAGetNumLinResEvals", 1);

    print!("\nFinal Solver Statistics: \n\n");
    print!("Number of steps                    = {}\n", nst);
    print!("Number of residual evaluations     = {}\n", nre + nreLS);
    print!("Number of nonlinear iterations     = {}\n", nni);
    print!("Number of error test failures      = {}\n", netf);
    print!("Number of nonlinear conv. failures = {}\n", ncfn);

    /* check the solution error */
    let retval = check_ans(&yy, t, reltol, abstol);

    /* Clean up and return */
    IDAFree(&mut ida_mem_opt);
    let _ = SUNLinSolFree(Some(LS));
    N_VDestroy(yy);
    N_VDestroy(yp);
    let _ = SUNContext_Free(&mut sunctx);

    std::process::exit(retval);
}

/*-------------------------------
 * Functions called by the solver
 *-------------------------------*/

/* System residual function:
     0 = (1-alpha)/(t-2)*x1 - x1 + (alpha-1)*x2 + 2*exp(t) - x1'(t)
     0 = (t+2)*x1 - (t+2)*exp(t)
*/
fn fres(
    t: sunrealtype,
    yy: &N_Vector,
    yp: &N_Vector,
    rr: &N_Vector,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    /* C casts user_data to sunrealtype* and reads rdata[0] */
    let alpha = *user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<sunrealtype>())
        .expect("user_data"); /* set shortcut for stiffness parameter */
    let (x1, x2) = {
        let yydata = NV_DATA_S(yy); /* access current solution values */
        (yydata[0], yydata[1])
    };
    let x1p = NV_DATA_S(yp)[0]; /* access current derivative values */

    let mut rrdata = NV_DATA_S(rr);
    rrdata[0] = (ONE - alpha) / (t - TWO) * x1 - x1 + (alpha - ONE) * x2 + TWO * t.sun_exp() - x1p;
    rrdata[1] = (t + TWO) * x1 - (t + TWO) * SUNRexp(t);

    0
}

/*-------------------------------------
 * Custom matrix-embedded linear solver
 *-------------------------------------*/

/* constructor */
fn MatrixEmbeddedLS(ida_mem: &IDAMem, ctx: &SUNContext) -> Option<SUNLinearSolver> {
    /* Create an empty linear solver */
    let LS = SUNLinSolNewEmpty(ctx)?;

    /* Attach operations */
    {
        let mut ops = LS.ops.borrow_mut();
        ops.gettype = Some(MatrixEmbeddedLSType);
        ops.solve = Some(MatrixEmbeddedLSSolve);
        ops.free = Some(MatrixEmbeddedLSFree);
    }

    /* Set content pointer to IDA memory */
    *LS.content.borrow_mut() = Box::new(ida_mem.clone());

    /* Return solver */
    Some(LS)
}

/* type descriptor */
fn MatrixEmbeddedLSType(_S: &SUNLinearSolver) -> SUNLinearSolver_Type {
    SUNLINEARSOLVER_MATRIX_EMBEDDED
}

/* linear solve routine */
fn MatrixEmbeddedLSSolve(
    LS: &SUNLinearSolver,
    _A: Option<&SUNMatrix>,
    x: &N_Vector,
    b: &N_Vector,
    _tol: sunrealtype,
) -> i32 {
    /* temporary variables */
    let mut tcur: sunrealtype = 0.0;
    let mut cj: sunrealtype = 0.0;
    let mut yypred: Option<N_Vector> = None;
    let mut yppred: Option<N_Vector> = None;
    let mut yyn: Option<N_Vector> = None;
    let mut ypn: Option<N_Vector> = None;
    let mut res: Option<N_Vector> = None;
    let mut user_data: Option<Box<dyn Any>> = None;

    /* retrieve implicit system data from IDA (LS->content) */
    let ida_mem = LS
        .content
        .borrow()
        .downcast_ref::<IDAMem>()
        .expect("MatrixEmbeddedLS content")
        .clone();
    let retval = IDAGetNonlinearSystemData(
        &ida_mem,
        &mut tcur,
        &mut yypred,
        &mut yppred,
        &mut yyn,
        &mut ypn,
        &mut res,
        &mut cj,
        &mut user_data,
    );
    if check_retval(Some(retval), "IDAGetNonlinearSystemData", 1) != 0 {
        return -1;
    }

    /* extract stiffness parameter from user_data */
    let alpha = *user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<sunrealtype>())
        .expect("user_data");

    /* perform linear solve: A*x=b
           A = df/dy + cj*df/dyp
        =>
           A = [ - cj - (alpha - 1)/(t - 2) - 1, alpha - 1]
               [                          t + 2,         0]
    */
    let a11 = -cj - (alpha - ONE) / (tcur - TWO) - ONE;
    let a12 = alpha - ONE;
    let a21 = tcur + TWO;
    let (b1, b2) = {
        let bdata = NV_DATA_S(b);
        (bdata[0], bdata[1])
    };
    {
        let mut xdata = NV_DATA_S(x);
        xdata[0] = b2 / a21;
        xdata[1] = -(a11 * b2 - a21 * b1) / (a12 * a21);
    }

    /* IDAGetNonlinearSystemData SWAPPED the user_data box out of the
    integrator; a second call swaps it back in (leaving the local as
    None), so the integrator owns the box again before we return. */
    let retval = IDAGetNonlinearSystemData(
        &ida_mem,
        &mut tcur,
        &mut yypred,
        &mut yppred,
        &mut yyn,
        &mut ypn,
        &mut res,
        &mut cj,
        &mut user_data,
    );
    if check_retval(Some(retval), "IDAGetNonlinearSystemData", 1) != 0 {
        return -1;
    }
    debug_assert!(user_data.is_none());

    /* return with success */
    SUN_SUCCESS
}

/* destructor */
fn MatrixEmbeddedLSFree(LS: &SUNLinearSolver) -> SUNErrCode {
    /* LS->content = NULL: drop the stored IDAMem handle; the caller
    (SUNLinSolFree) then drops the handle itself (SUNLinSolFreeEmpty) */
    *LS.content.borrow_mut() = Box::new(());
    SUN_SUCCESS
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
*/

fn check_retval(returnvalue: Option<i32>, funcname: &str, opt: i32) -> i32 {
    /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
    if opt == 0 && returnvalue.is_none() {
        eprint!(
            "\nSUNDIALS_ERROR: {}() failed - returned NULL pointer\n\n",
            funcname
        );
        return 1;
    }
    /* Check if flag < 0 */
    else if opt == 1 {
        let retval = returnvalue.expect("retval");
        if retval < 0 {
            eprint!(
                "\nSUNDIALS_ERROR: {}() failed with flag = {}\n\n",
                funcname, retval
            );
            return 1;
        }
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

/* routine to fill analytical solution and its derivative */
fn analytical_solution(t: sunrealtype, y: &N_Vector, yp: &N_Vector) {
    {
        let mut ydata = NV_DATA_S(y);
        ydata[0] = SUNRexp(t);
        ydata[1] = SUNRexp(t) / (t - 2.0);
    }
    {
        let mut ypdata = NV_DATA_S(yp);
        ypdata[0] = SUNRexp(t);
        ypdata[1] = SUNRexp(t) / (t - 2.0) - SUNRexp(t) / (t - 2.0) / (t - 2.0);
    }
}

/* check the computed solution */
fn check_ans(y: &N_Vector, t: sunrealtype, rtol: sunrealtype, atol: sunrealtype) -> i32 {
    /* create solution and error weight vectors */
    let ytrue = N_VClone(y).expect("N_VClone");
    let ewt = N_VClone(y).expect("N_VClone");
    let abstol = N_VClone(y).expect("N_VClone");

    /* set the solution data */
    analytical_solution(t, &ytrue, &abstol);

    /* compute the error weight vector, loosen atol */
    N_VConst(atol, &abstol);
    N_VAbs(&ytrue, &ewt);
    N_VLinearSum(rtol, &ewt, 10.0, &abstol, &ewt);
    if N_VMin(&ewt) <= 0.0 {
        eprint!("\nSUNDIALS_ERROR: check_ans failed - ewt <= 0\n\n");
        return -1;
    }
    N_VInv(&ewt, &ewt);

    /* compute the solution error */
    N_VLinearSum(ONE, y, -ONE, &ytrue, &ytrue);
    let err = N_VWrmsNorm(&ytrue, &ewt);

    /* is the solution within the tolerances? */
    let passfail = if err < ONE { 0 } else { 1 };

    if passfail != 0 {
        print!("\nSUNDIALS_WARNING: check_ans error={}\n\n", fmt_g(err, 6));
    }

    /* Free vectors */
    N_VDestroy(ytrue);
    N_VDestroy(abstol);
    N_VDestroy(ewt);

    passfail
}

/*---- end of file ----*/
