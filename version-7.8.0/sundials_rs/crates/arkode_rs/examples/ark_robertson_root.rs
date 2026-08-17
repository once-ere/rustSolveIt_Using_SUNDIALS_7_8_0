/*---------------------------------------------------------------
 * Programmer(s): Daniel R. Reynolds @ UMBC
 *---------------------------------------------------------------
 * Rust port of `examples/arkode/C_serial/ark_robertson_root.c`.
 *---------------------------------------------------------------
 * Example problem:
 *
 * The following test simulates the Robertson problem,
 * corresponding to the kinetics of an autocatalytic reaction.
 * This is an ODE system with 3 components, Y = [u,v,w], satisfying
 * the equations,
 *    du/dt = -0.04*u + 1e4*v*w
 *    dv/dt = 0.04*u - 1e4*v*w - 3e7*v^2
 *    dw/dt = 3e7*v^2
 * for t in the interval [0.0, 1e11], with initial conditions
 * Y0 = [1,0,0].
 *
 * While integrating the system, we use the rootfinding feature
 * to find the times at which either u=1e-4 or w=1e-2.
 *
 * This program solves the problem with one of the solvers, ERK,
 * DIRK or ARK.  For DIRK and ARK, implicit subsystems are solved
 * using a Newton iteration with the dense SUNLinearSolver, and a
 * user-supplied Jacobian routine.
 *
 * 100 outputs are printed at equal intervals, and run statistics
 * are printed at the end.
 *---------------------------------------------------------------*/

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
    let T1: sunrealtype = 0.4; /* first output time */
    let TMult: sunrealtype = 10.0; /* output time multiplication factor */
    let Nt: i32 = 12; /* total number of output times */
    let NEQ: sunindextype = 3; /* number of dependent vars. */

    /* rootfinding variables */
    let mut rootsfound: [i32; 2] = [0; 2];
    /* C `int rtflag;` (root info flag) is declared at the top of main but only
    ever live inside the ARK_ROOT_RETURN branch, so it is bound there */

    /* general problem variables */
    let mut flag: i32; /* reusable error-checking flag */
    let mut t: sunrealtype;
    let mut tout: sunrealtype;
    let mut iout: i32;
    let (
        mut nst,
        mut nst_a,
        mut nfe,
        mut nfi,
        mut nsetups,
        mut nje,
        mut nfeLS,
        mut nni,
        mut nnf,
        mut ncfn,
        mut netf,
        mut nge,
    ) = (
        0i64, 0i64, 0i64, 0i64, 0i64, 0i64, 0i64, 0i64, 0i64, 0i64, 0i64, 0i64,
    );

    /* set up the initial conditions */
    let u0: sunrealtype = 1.0;
    let v0: sunrealtype = 0.0;
    let w0: sunrealtype = 0.0;
    let reltol: sunrealtype = 1.0e-4;

    /* Create the SUNDIALS context object for this simulation */
    let mut ctx_opt: Option<SUNContext> = None;
    flag = SUNContext_Create(SUN_COMM_NULL, &mut ctx_opt);
    if check_flag(Some(flag), "SUNContext_Create", 1) != 0 {
        std::process::exit(1);
    }
    let ctx = ctx_opt.as_ref().expect("ctx").clone();

    /* Initial problem output */
    print!("\nRobertson ODE test problem (with rootfinding):\n");
    print!(
        "    initial conditions:  u0 = {},  v0 = {},  w0 = {}\n",
        fmt_g(u0, 6),
        fmt_g(v0, 6),
        fmt_g(w0, 6)
    );

    /* Initialize data structures */
    let y = N_VNew_Serial(NEQ, &ctx); /* Create serial vector for solution */
    if check_flag(y.as_ref().map(|_| 0), "N_VNew_Serial", 0) != 0 {
        std::process::exit(1);
    }
    let y = y.expect("y");
    NV_Ith_S_set(&y, 0, u0); /* Set initial conditions into y */
    NV_Ith_S_set(&y, 1, v0);
    NV_Ith_S_set(&y, 2, w0);

    let atols = N_VClone(&y); /* Create serial vector absolute tolerances */
    if check_flag(atols.as_ref().map(|_| 0), "N_VNew_Serial", 0) != 0 {
        std::process::exit(1);
    }
    let atols = atols.expect("atols");

    /* Set absolute tolerances */
    NV_Ith_S_set(&atols, 0, 1.0e-8);
    NV_Ith_S_set(&atols, 1, 1.0e-11);
    NV_Ith_S_set(&atols, 2, 1.0e-8);

    /* Call ARKStepCreate to initialize the ARK timestepper module and
    specify the right-hand side function in y'=f(t,y), the initial time
    T0, and the initial dependent variable vector y.  Note: since this
    problem is fully implicit, we set f_E to NULL and f_I to f. */
    let mut arkode_mem_opt = ARKStepCreate(None, Some(f), T0, &y, &ctx);
    if check_flag(arkode_mem_opt.as_ref().map(|_| 0), "ARKStepCreate", 0) != 0 {
        std::process::exit(1);
    }
    let arkode_mem = arkode_mem_opt.as_ref().expect("arkode_mem").clone();

    /* Set routines */
    flag = ARKodeSetMaxErrTestFails(&arkode_mem, 20); /* Increase max error test fails */
    if check_flag(Some(flag), "ARKodeSetMaxErrTestFails", 1) != 0 {
        std::process::exit(1);
    }
    flag = ARKodeSetMaxNonlinIters(&arkode_mem, 8); /* Increase max nonlin iters  */
    if check_flag(Some(flag), "ARKodeSetMaxNonlinIters", 1) != 0 {
        std::process::exit(1);
    }
    flag = ARKodeSetNonlinConvCoef(&arkode_mem, 1.0e-7); /* Set nonlinear convergence coeff. */
    if check_flag(Some(flag), "ARKodeSetNonlinConvCoef", 1) != 0 {
        std::process::exit(1);
    }
    flag = ARKodeSetMaxNumSteps(&arkode_mem, 100000); /* Increase max num steps */
    if check_flag(Some(flag), "ARKodeSetMaxNumSteps", 1) != 0 {
        std::process::exit(1);
    }
    flag = ARKodeSetPredictorMethod(&arkode_mem, 1); /* Specify maximum-order predictor */
    if check_flag(Some(flag), "ARKodeSetPredictorMethod", 1) != 0 {
        std::process::exit(1);
    }
    flag = ARKodeSVtolerances(&arkode_mem, reltol, &atols); /* Specify tolerances */
    if check_flag(Some(flag), "ARKodeSStolerances", 1) != 0 {
        std::process::exit(1);
    }

    /* Specify the root-finding function, having 2 equations */
    flag = ARKodeRootInit(&arkode_mem, 2, Some(g));
    if check_flag(Some(flag), "ARKodeRootInit", 1) != 0 {
        std::process::exit(1);
    }

    /* Initialize dense matrix data structure and solver */
    let A = SUNDenseMatrix(NEQ, NEQ, &ctx);
    if check_flag(A.as_ref().map(|_| 0), "SUNDenseMatrix", 0) != 0 {
        std::process::exit(1);
    }
    let A = A.expect("A");
    let LS = SUNLinSol_Dense(&y, &A, &ctx);
    if check_flag(LS.as_ref().map(|_| 0), "SUNLinSol_Dense", 0) != 0 {
        std::process::exit(1);
    }
    let LS = LS.expect("LS");

    /* Linear solver interface */
    flag = ARKodeSetLinearSolver(&arkode_mem, &LS, Some(&A)); /* Attach matrix and linear solver */
    if check_flag(Some(flag), "ARKodeSetLinearSolver", 1) != 0 {
        std::process::exit(1);
    }
    flag = ARKodeSetJacFn(&arkode_mem, Some(Jac)); /* Set the Jacobian routine */
    if check_flag(Some(flag), "ARKodeSetJacFn", 1) != 0 {
        std::process::exit(1);
    }

    /* Open output stream for results, output comment line */
    let mut UFID = File::create("solution.txt").expect("solution.txt");
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

    /* Main time-stepping loop: calls ARKodeEvolve to perform the integration, then
    prints results.  Stops when the final time has been reached */
    t = T0;
    print!("        t             u             v             w\n");
    print!("   -----------------------------------------------------\n");
    print!(
        "  {}  {}  {}  {}\n",
        fmt_ew(t, 12, 5),
        fmt_ew(NV_Ith_S(&y, 0), 12, 5),
        fmt_ew(NV_Ith_S(&y, 1), 12, 5),
        fmt_ew(NV_Ith_S(&y, 2), 12, 5)
    );
    tout = T1;
    iout = 0;
    loop {
        flag = ARKodeEvolve(&arkode_mem, tout, &y, &mut t, ARK_NORMAL); /* call integrator */
        if check_flag(Some(flag), "ARKodeEvolve", 1) != 0 {
            break;
        }
        print!(
            "  {}  {}  {}  {}\n", /* access/print solution */
            fmt_ew(t, 12, 5),
            fmt_ew(NV_Ith_S(&y, 0), 12, 5),
            fmt_ew(NV_Ith_S(&y, 1), 12, 5),
            fmt_ew(NV_Ith_S(&y, 2), 12, 5)
        );
        let _ = write!(
            UFID,
            " {} {} {} {}\n",
            fmt_e(t, 16),
            fmt_e(NV_Ith_S(&y, 0), 16),
            fmt_e(NV_Ith_S(&y, 1), 16),
            fmt_e(NV_Ith_S(&y, 2), 16)
        );
        if flag == ARK_ROOT_RETURN {
            /* check if a root was found */
            let rtflag = ARKodeGetRootInfo(&arkode_mem, &mut rootsfound);
            if check_flag(Some(rtflag), "ARKodeGetRootInfo", 1) != 0 {
                std::process::exit(1);
            }
            print!(
                "      rootsfound[] = {:3} {:3}\n",
                rootsfound[0], rootsfound[1]
            );
        }
        if flag >= 0 {
            /* successful solve: update output time */
            iout += 1;
            tout *= TMult;
        } else {
            /* unsuccessful solve: break */
            eprint!("Solver failure, stopping integration\n");
            break;
        }
        if iout == Nt {
            break; /* stop after enough outputs */
        }
    }
    print!("   -----------------------------------------------------\n");
    drop(UFID);

    /* Print some final statistics */
    flag = ARKodeGetNumSteps(&arkode_mem, &mut nst);
    check_flag(Some(flag), "ARKodeGetNumSteps", 1);
    flag = ARKodeGetNumStepAttempts(&arkode_mem, &mut nst_a);
    check_flag(Some(flag), "ARKodeGetNumStepAttempts", 1);
    flag = ARKodeGetNumRhsEvals(&arkode_mem, 0, &mut nfe);
    check_flag(Some(flag), "ARKodeGetNumRhsEvals", 1);
    flag = ARKodeGetNumRhsEvals(&arkode_mem, 1, &mut nfi);
    check_flag(Some(flag), "ARKodeGetNumRhsEvals", 1);
    flag = ARKodeGetNumLinSolvSetups(&arkode_mem, &mut nsetups);
    check_flag(Some(flag), "ARKodeGetNumLinSolvSetups", 1);
    flag = ARKodeGetNumErrTestFails(&arkode_mem, &mut netf);
    check_flag(Some(flag), "ARKodeGetNumErrTestFails", 1);
    flag = ARKodeGetNumStepSolveFails(&arkode_mem, &mut ncfn);
    check_flag(Some(flag), "ARKodeGetNumStepSolveFails", 1);
    flag = ARKodeGetNumNonlinSolvIters(&arkode_mem, &mut nni);
    check_flag(Some(flag), "ARKodeGetNumNonlinSolvIters", 1);
    flag = ARKodeGetNumNonlinSolvConvFails(&arkode_mem, &mut nnf);
    check_flag(Some(flag), "ARKodeGetNumNonlinSolvConvFails", 1);
    flag = ARKodeGetNumJacEvals(&arkode_mem, &mut nje);
    check_flag(Some(flag), "ARKodeGetNumJacEvals", 1);
    flag = ARKodeGetNumLinRhsEvals(&arkode_mem, &mut nfeLS);
    check_flag(Some(flag), "ARKodeGetNumLinRhsEvals", 1);
    flag = ARKodeGetNumGEvals(&arkode_mem, &mut nge);
    check_flag(Some(flag), "ARKodeGetNumGEvals", 1);

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
    print!("   Total root-function g evals = {}\n", nge);
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
    N_VDestroy(atols); /* Free atols vector */
    ARKodeFree(&mut arkode_mem_opt); /* Free integrator memory */
    SUNLinSolFree(Some(LS)); /* Free linear solver */
    SUNMatDestroy(A); /* Free A matrix */
    SUNContext_Free(&mut ctx_opt); /* Free context */
}

/*-------------------------------
 * Functions called by the solver
 *-------------------------------*/

/* f routine to compute the ODE RHS function f(t,y). */
fn f(
    _t: sunrealtype,
    y: &N_Vector,
    ydot: &N_Vector,
    _user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    let u = NV_Ith_S(y, 0); /* access current solution */
    let v = NV_Ith_S(y, 1);
    let w = NV_Ith_S(y, 2);

    /* Fill in ODE RHS function */
    NV_Ith_S_set(ydot, 0, -0.04 * u + 1.0e4 * v * w);
    NV_Ith_S_set(ydot, 1, 0.04 * u - 1.0e4 * v * w - 3.0e7 * v * v);
    NV_Ith_S_set(ydot, 2, 3.0e7 * v * v);

    0 /* Return with success */
}

/* Jacobian routine to compute J(t,y) = df/dy. */
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
    let v = NV_Ith_S(y, 1); /* access current solution */
    let w = NV_Ith_S(y, 2);
    SUNMatZero(J); /* initialize Jacobian to zero */

    /* Fill in the Jacobian of the ODE RHS function */
    SM_ELEMENT_D_set(J, 0, 0, -0.04);
    SM_ELEMENT_D_set(J, 0, 1, 1.0e4 * w);
    SM_ELEMENT_D_set(J, 0, 2, 1.0e4 * v);

    SM_ELEMENT_D_set(J, 1, 0, 0.04);
    SM_ELEMENT_D_set(J, 1, 1, -1.0e4 * w - 6.0e7 * v);
    SM_ELEMENT_D_set(J, 1, 2, -1.0e4 * v);

    SM_ELEMENT_D_set(J, 2, 1, 6.0e7 * v);

    0 /* Return with success */
}

/* g routine to compute the root-finding function g(t,y). */
fn g(
    _t: sunrealtype,
    y: &N_Vector,
    gout: &mut [sunrealtype],
    _user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    let u = NV_Ith_S(y, 0); /* access current solution */
    let w = NV_Ith_S(y, 2);

    gout[0] = u - 0.0001; /* check for u == 1e-4 */
    gout[1] = w - 0.01; /* check for w == 1e-2 */

    0 /* Return with success */
}

/*-------------------------------
 * Private helper functions
 *-------------------------------*/

/* Check function return value...
    opt == 0 means SUNDIALS function allocates memory so check if
             returned NULL pointer (represented as `None`)
    opt == 1 means SUNDIALS function returns a flag so check if
             flag >= 0
    opt == 2 means function allocates memory so check if returned
             NULL pointer
*/
fn check_flag(flagvalue: Option<i32>, funcname: &str, opt: i32) -> i32 {
    /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
    if opt == 0 && flagvalue.is_none() {
        eprint!(
            "\nSUNDIALS_ERROR: {}() failed - returned NULL pointer\n\n",
            funcname
        );
        return 1;
    }
    /* Check if flag < 0 */
    else if opt == 1 {
        let errflag = flagvalue.expect("errflag");
        if errflag < 0 {
            eprint!(
                "\nSUNDIALS_ERROR: {}() failed with flag = {}\n\n",
                funcname, errflag
            );
            return 1;
        }
    }
    /* Check if function returned NULL pointer - no memory allocated */
    else if opt == 2 && flagvalue.is_none() {
        eprint!(
            "\nMEMORY_ERROR: {}() failed - returned NULL pointer\n\n",
            funcname
        );
        return 1;
    }

    0
}

/*---- end of file ----*/
