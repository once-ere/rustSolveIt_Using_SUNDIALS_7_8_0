//! Port of `examples/arkode/C_serial/ark_heat1D.c`.
//!
//! Example problem:
//!
//! The following test simulates a simple 1D heat equation,
//!    u_t = k*u_xx + f
//! for t in [0, 10], x in [0, 1], with initial conditions
//!    u(0,x) =  0
//! Dirichlet boundary conditions, i.e.
//!    u_t(t,0) = u_t(t,1) = 0,
//! and a point-source heating term,
//!    f = 0.01 for x=0.5.
//!
//! The spatial derivatives are computed using second-order
//! centered differences, with the data distributed over N points
//! on a uniform spatial grid.
//!
//! This program solves the problem with either an ERK or DIRK
//! method.  For the DIRK method, we use a Newton iteration with
//! the SUNLinSol_PCG linear solver, and a user-supplied Jacobian-vector
//! product routine.
//!
//! 100 outputs are printed at equal intervals, and run statistics
//! are printed at the end.

#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

use std::any::Any;
use std::fs::File;
use std::io::Write;

use arkode_rs::prelude::*;

/* user data structure */
struct UserData {
    N: sunindextype,  /* number of intervals   */
    dx: sunrealtype,  /* mesh spacing          */
    k: sunrealtype,   /* diffusion coefficient */
}

/* Main Program */
fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let argc = argv.len() as i32;

    /* general problem parameters */
    let T0: sunrealtype = 0.0; /* initial time */
    let Tf: sunrealtype = 1.0; /* final time */
    let Nt: i32 = 10; /* total number of output times */
    let rtol: sunrealtype = 1.0e-6; /* relative tolerance */
    let atol: sunrealtype = 1.0e-10; /* absolute tolerance */
    let N: sunindextype = 201; /* spatial mesh size */
    let k: sunrealtype = 0.5; /* heat conductivity */

    /* general problem variables */
    let mut flag: i32; /* reusable error-checking flag */
    let mut t: sunrealtype;
    let dTout: sunrealtype;
    let mut tout: sunrealtype;
    let mut nst: i64 = 0;
    let mut nst_a: i64 = 0;
    let mut nfe: i64 = 0;
    let mut nfi: i64 = 0;
    let mut nsetups: i64 = 0;
    let mut nli: i64 = 0;
    let mut nJv: i64 = 0;
    let mut nlcf: i64 = 0;
    let mut nni: i64 = 0;
    let mut ncfn: i64 = 0;
    let mut netf: i64 = 0;

    /* Create the SUNDIALS context object for this simulation */
    let mut sunctx: Option<SUNContext> = None;
    flag = SUNContext_Create(SUN_COMM_NULL, &mut sunctx);
    if check_flag_int(flag, "SUNContext_Create") != 0 {
        std::process::exit(1);
    }
    let ctx = sunctx.as_ref().expect("SUNContext").clone();

    /* allocate and fill udata structure */
    let udata = UserData {
        N,
        k,
        dx: 1.0 / ((N - 1) as sunrealtype), /* mesh spacing */
    };

    /* Initial problem output */
    print!("\n1D Heat PDE test problem:\n");
    print!("  N = {}\n", udata.N);
    print!("  diffusion coefficient:  k = {}\n", fmt_g(udata.k, 6));

    /* Initialize data structures */
    let y = N_VNew_Serial(N, &ctx); /* Create serial vector for solution */
    if check_flag_null(&y, "N_VNew_Serial") != 0 {
        std::process::exit(1);
    }
    let y = y.expect("N_VNew_Serial");
    N_VConst(0.0, &y); /* Set initial conditions */

    /* Call ARKStepCreate to initialize the ARK timestepper module and
    specify the right-hand side function in y'=f(t,y), the initial time
    T0, and the initial dependent variable vector y.  Note: since this
    problem is fully implicit, we set f_E to NULL and f_I to f. */
    let mut arkode_mem = ARKStepCreate(None, Some(f), T0, &y, &ctx);
    if check_flag_null(&arkode_mem, "ARKStepCreate") != 0 {
        std::process::exit(1);
    }
    let ark = arkode_mem.as_ref().expect("ARKStepCreate").clone();

    /* Set routines */
    flag = ARKodeSetUserData(&ark, Some(Box::new(udata))); /* Pass udata to user functions */
    if check_flag_int(flag, "ARKodeSetUserData") != 0 {
        std::process::exit(1);
    }
    flag = ARKodeSetMaxNumSteps(&ark, 10000); /* Increase max num steps  */
    if check_flag_int(flag, "ARKodeSetMaxNumSteps") != 0 {
        std::process::exit(1);
    }
    flag = ARKodeSetPredictorMethod(&ark, 1); /* Specify maximum-order predictor */
    if check_flag_int(flag, "ARKodeSetPredictorMethod") != 0 {
        std::process::exit(1);
    }
    flag = ARKodeSStolerances(&ark, rtol, atol); /* Specify tolerances */
    if check_flag_int(flag, "ARKodeSStolerances") != 0 {
        std::process::exit(1);
    }

    /* Initialize PCG solver -- no preconditioning, with up to N iterations  */
    let LS = SUNLinSol_PCG(&y, SUN_PREC_NONE, N as i32, &ctx);
    if check_flag_null(&LS, "SUNLinSol_PCG") != 0 {
        std::process::exit(1);
    }
    let LS = LS.expect("SUNLinSol_PCG");

    /* Linear solver interface -- set user-supplied J*v routine (no 'jtsetup' required) */
    flag = ARKodeSetLinearSolver(&ark, &LS, None); /* Attach linear solver to ARKODE */
    if check_flag_int(flag, "ARKodeSetLinearSolver") != 0 {
        std::process::exit(1);
    }
    flag = ARKodeSetJacTimes(&ark, None, Some(Jac)); /* Set the Jacobian routine */
    if check_flag_int(flag, "ARKodeSetJacTimes") != 0 {
        std::process::exit(1);
    }

    /* Specify linearly implicit RHS, with non-time-dependent Jacobian */
    flag = ARKodeSetLinear(&ark, 0);
    if check_flag_int(flag, "ARKodeSetLinear") != 0 {
        std::process::exit(1);
    }

    /* Override any current settings with command-line options */
    flag = ARKodeSetOptions(&ark, None, None, argc, &argv);
    if check_flag_int(flag, "ARKodeSetOptions") != 0 {
        std::process::exit(1);
    }
    flag = SUNLinSolSetOptions(&LS, None, None, &argv);
    if check_flag_int(flag, "SUNLinSolSetOptions") != 0 {
        std::process::exit(1);
    }

    /* recover the mesh spacing for the on-disk mesh output (the udata box now
    lives inside the integrator; C keeps its own pointer to the same struct) */
    let dx = 1.0 / ((N - 1) as sunrealtype);

    /* output mesh to disk */
    let mut FID = File::create("heat_mesh.txt").expect("heat_mesh.txt");
    for i in 0..N {
        let _ = write!(FID, "  {}\n", fmt_e(dx * (i as sunrealtype), 16));
    }
    drop(FID);

    /* Open output stream for results, access data array */
    let mut UFID = File::create("heat1D.txt").expect("heat1D.txt");

    /* output initial condition to disk */
    {
        let data = N_VGetArrayPointer(&y).expect("N_VGetArrayPointer");
        for i in 0..N {
            let _ = write!(UFID, " {}", fmt_e(data[i as usize], 16));
        }
    }
    let _ = write!(UFID, "\n");

    /* Main time-stepping loop: calls ARKodeEvolve to perform the integration, then
    prints results.  Stops when the final time has been reached */
    t = T0;
    dTout = (Tf - T0) / (Nt as sunrealtype);
    tout = T0 + dTout;
    print!("        t      ||u||_rms\n");
    print!("   -------------------------\n");
    print!(
        "  {}  {}\n",
        fmt_fw(t, 10, 6),
        fmt_fw((N_VDotProd(&y, &y) / (N as sunrealtype)).sqrt(), 10, 6)
    );
    for _iout in 0..Nt {
        flag = ARKodeEvolve(&ark, tout, &y, &mut t, ARK_NORMAL); /* call integrator */
        if check_flag_int(flag, "ARKodeEvolve") != 0 {
            break;
        }
        print!(
            "  {}  {}\n",
            fmt_fw(t, 10, 6),
            fmt_fw((N_VDotProd(&y, &y) / (N as sunrealtype)).sqrt(), 10, 6)
        ); /* print solution stats */
        if flag >= 0 {
            /* successful solve: update output time */
            tout += dTout;
            tout = if tout > Tf { Tf } else { tout };
        } else {
            /* unsuccessful solve: break */
            eprint!("Solver failure, stopping integration\n");
            break;
        }

        /* output results to disk */
        {
            let data = N_VGetArrayPointer(&y).expect("N_VGetArrayPointer");
            for i in 0..N {
                let _ = write!(UFID, " {}", fmt_e(data[i as usize], 16));
            }
        }
        let _ = write!(UFID, "\n");
    }
    print!("   -------------------------\n");
    drop(UFID);

    /* Print some final statistics */
    flag = ARKodeGetNumSteps(&ark, &mut nst);
    check_flag_int(flag, "ARKodeGetNumSteps");
    flag = ARKodeGetNumStepAttempts(&ark, &mut nst_a);
    check_flag_int(flag, "ARKodeGetNumStepAttempts");
    flag = ARKodeGetNumRhsEvals(&ark, 0, &mut nfe);
    check_flag_int(flag, "ARKodeGetNumRhsEvals");
    flag = ARKodeGetNumRhsEvals(&ark, 1, &mut nfi);
    check_flag_int(flag, "ARKodeGetNumRhsEvals");
    flag = ARKodeGetNumLinSolvSetups(&ark, &mut nsetups);
    check_flag_int(flag, "ARKodeGetNumLinSolvSetups");
    flag = ARKodeGetNumErrTestFails(&ark, &mut netf);
    check_flag_int(flag, "ARKodeGetNumErrTestFails");
    flag = ARKodeGetNumNonlinSolvIters(&ark, &mut nni);
    check_flag_int(flag, "ARKodeGetNumNonlinSolvIters");
    flag = ARKodeGetNumNonlinSolvConvFails(&ark, &mut ncfn);
    check_flag_int(flag, "ARKodeGetNumNonlinSolvConvFails");
    flag = ARKodeGetNumLinIters(&ark, &mut nli);
    check_flag_int(flag, "ARKodeGetNumLinIters");
    flag = ARKodeGetNumJtimesEvals(&ark, &mut nJv);
    check_flag_int(flag, "ARKodeGetNumJtimesEvals");
    flag = ARKodeGetNumLinConvFails(&ark, &mut nlcf);
    check_flag_int(flag, "ARKodeGetNumLinConvFails");

    print!("\nFinal Solver Statistics:\n");
    print!(
        "   Internal solver steps = {} (attempted = {})\n",
        nst, nst_a
    );
    print!("   Total RHS evals:  Fe = {},  Fi = {}\n", nfe, nfi);
    print!("   Total linear solver setups = {}\n", nsetups);
    print!("   Total linear iterations = {}\n", nli);
    print!("   Total number of Jacobian-vector products = {}\n", nJv);
    print!(
        "   Total number of linear solver convergence failures = {}\n",
        nlcf
    );
    print!("   Total number of Newton iterations = {}\n", nni);
    print!(
        "   Total number of nonlinear solver convergence failures = {}\n",
        ncfn
    );
    print!("   Total number of error test failures = {}\n", netf);

    /* Clean up and return with successful completion */
    N_VDestroy(y); /* Free vectors */
    /* the udata struct is owned by the integrator's `user_data` box (C `free(udata)`) */
    ARKodeFree(&mut arkode_mem); /* Free integrator memory */
    let _ = SUNLinSolFree(Some(LS)); /* Free linear solver */
    let _ = SUNContext_Free(&mut sunctx); /* Free context */

    std::process::exit(0);
}

/*--------------------------------
 * Functions called by the solver
 *--------------------------------*/

/* f routine to compute the ODE RHS function f(t,y). */
fn f(_t: sunrealtype, y: &N_Vector, ydot: &N_Vector, user_data: &mut Option<Box<dyn Any>>) -> i32 {
    let udata = user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<UserData>())
        .expect("user_data is UserData"); /* access problem data */
    let N = udata.N; /* set variable shortcuts */
    let k = udata.k;
    let dx = udata.dx;

    /* access data arrays */
    if check_flag_null(&N_VGetArrayPointer(y), "N_VGetArrayPointer") != 0 {
        return 1;
    }
    if check_flag_null(&N_VGetArrayPointer(ydot), "N_VGetArrayPointer") != 0 {
        return 1;
    }
    N_VConst(0.0, ydot); /* Initialize ydot to zero */

    /* iterate over domain, computing all equations */
    let c1 = k / dx / dx;
    let c2 = -2.0 * k / dx / dx;
    let isource = N / 2;

    let Y = N_VGetArrayPointer(y).expect("N_VGetArrayPointer");
    let mut Ydot = N_VGetArrayPointer(ydot).expect("N_VGetArrayPointer");
    Ydot[0] = 0.0; /* left boundary condition */
    for i in 1..(N - 1) {
        let i = i as usize;
        Ydot[i] = c1 * Y[i - 1] + c2 * Y[i] + c1 * Y[i + 1];
    }
    Ydot[(N - 1) as usize] = 0.0; /* right boundary condition */
    Ydot[isource as usize] += 0.01 / dx; /* source term */

    0 /* Return with success */
}

/* Jacobian routine to compute J(t,y) = df/dy. */
fn Jac(
    v: &N_Vector,
    Jv: &N_Vector,
    _t: sunrealtype,
    _y: &N_Vector,
    _fy: &N_Vector,
    user_data: &mut Option<Box<dyn Any>>,
    _tmp: &N_Vector,
) -> i32 {
    let udata = user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<UserData>())
        .expect("user_data is UserData"); /* variable shortcuts */
    let N = udata.N;
    let k = udata.k;
    let dx = udata.dx;

    /* access data arrays */
    if check_flag_null(&N_VGetArrayPointer(v), "N_VGetArrayPointer") != 0 {
        return 1;
    }
    if check_flag_null(&N_VGetArrayPointer(Jv), "N_VGetArrayPointer") != 0 {
        return 1;
    }
    N_VConst(0.0, Jv); /* initialize Jv product to zero */

    /* iterate over domain, computing all Jacobian-vector products */
    let c1 = k / dx / dx;
    let c2 = -2.0 * k / dx / dx;

    let V = N_VGetArrayPointer(v).expect("N_VGetArrayPointer");
    let mut JV = N_VGetArrayPointer(Jv).expect("N_VGetArrayPointer");
    JV[0] = 0.0;
    for i in 1..(N - 1) {
        let i = i as usize;
        JV[i] = c1 * V[i - 1] + c2 * V[i] + c1 * V[i + 1];
    }
    JV[(N - 1) as usize] = 0.0;

    0 /* Return with success */
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

   The C void-pointer/opt polymorphism splits into two typed helpers with
   identical message text:
     check_flag_null = opt == 0
     check_flag_int  = opt == 1
*/

fn check_flag_null<T>(flagvalue: &Option<T>, funcname: &str) -> i32 {
    /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
    if flagvalue.is_none() {
        eprint!(
            "\nSUNDIALS_ERROR: {}() failed - returned NULL pointer\n\n",
            funcname
        );
        return 1;
    }
    0
}

fn check_flag_int(flagvalue: i32, funcname: &str) -> i32 {
    /* Check if flag < 0 */
    if flagvalue < 0 {
        eprint!(
            "\nSUNDIALS_ERROR: {}() failed with flag = {}\n\n",
            funcname, flagvalue
        );
        return 1;
    }
    0
}

/*---- end of file ----*/
