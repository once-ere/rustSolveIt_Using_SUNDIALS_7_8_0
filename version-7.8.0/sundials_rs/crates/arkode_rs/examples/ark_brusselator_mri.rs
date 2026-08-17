//! Port of `examples/arkode/C_serial/ark_brusselator_mri.c`.
//!
//! Example problem:
//!
//! The following test simulates a brusselator problem from chemical
//! kinetics. This is an ODE system with 3 components, Y = [u,v,w],
//! satisfying the equations,
//!
//!    du/dt = a - (w+1)*u + v*u^2
//!    dv/dt = w*u - v*u^2
//!    dw/dt = (b-w)/ep - w*u
//!
//! for t in the interval [0.0, 2.0], with parameter values a=1,
//! b=3.5, and ep=1.0e-2. The initial conditions Y0 = [u0,v0,w0] are
//! u0=1.2, v0=3.1, and w0=3.
//!
//! This program solves the problem with the MRI stepper. Outputs are
//! printed at equal intervals of 0.1 and run statistics are printed
//! at the end.

#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

use std::any::Any;
use std::fs::File;
use std::io::Write;

use arkode_rs::prelude::*;

/* C `NV_Ith_S(v,i)` accessors (0-based), each holding the RefMut guard for
   exactly one statement. */

fn NV_Ith_S(v: &N_Vector, i: usize) -> sunrealtype {
    NV_DATA_S(v)[i]
}

fn NV_Ith_S_set(v: &N_Vector, i: usize, x: sunrealtype) {
    NV_DATA_S(v)[i] = x;
}

/* Main Program */
fn main() {
    /* general problem parameters */
    let T0: sunrealtype = 0.0; /* initial time */
    let Tf: sunrealtype = 2.0; /* final time */
    let dTout: sunrealtype = 0.1; /* time between outputs */
    let NEQ: sunindextype = 3; /* number of dependent vars. */
    let Nt: i32 = (Tf / dTout).ceil() as i32; /* number of output times */
    let hs: sunrealtype = 0.025; /* slow step size */
    let hf: sunrealtype = 0.001; /* fast step size */
    let a: sunrealtype;
    let b: sunrealtype;
    let ep: sunrealtype; /* ODE parameters */
    let u0: sunrealtype;
    let v0: sunrealtype;
    let w0: sunrealtype; /* initial conditions */
    let mut rdata: [sunrealtype; 3] = [0.0; 3]; /* user data */

    /* general problem variables */
    let mut retval: i32; /* reusable error-checking flag */
    let mut inner_stepper: Option<MRIStepInnerStepper> = None; /* inner stepper */
    let UFID: File;
    let mut t: sunrealtype;
    let mut tout: sunrealtype;
    let mut nsts: i64 = 0;
    let mut nstf: i64 = 0;
    let mut nfse: i64 = 0;
    let mut nff: i64 = 0;

    /*
     * Initialization
     */

    /* Create the SUNDIALS context object for this simulation */
    let mut ctx: Option<SUNContext> = None;
    retval = SUNContext_Create(SUN_COMM_NULL, &mut ctx);
    if check_retval_int(retval, "SUNContext_Create") != 0 {
        std::process::exit(1);
    }
    let ctx_h = ctx.clone().unwrap();

    /* Set up the test problem parameters */
    a = 1.0;
    b = 3.5;
    ep = 1.0e-2;

    /* Set the initial contions */
    u0 = 1.2;
    v0 = 3.1;
    w0 = 3.0;

    /* Initial problem output */
    print!("\nBrusselator ODE test problem:\n");
    print!(
        "    initial conditions:  u0 = {},  v0 = {},  w0 = {}\n",
        fmt_g(u0, 6),
        fmt_g(v0, 6),
        fmt_g(w0, 6)
    );
    print!(
        "    problem parameters:  a = {},  b = {},  ep = {}\n",
        fmt_g(a, 6),
        fmt_g(b, 6),
        fmt_g(ep, 6)
    );
    print!("    hs = {},  hf = {}\n\n", fmt_g(hs, 6), fmt_g(hf, 6));

    /* Set parameters in user data */
    rdata[0] = a;
    rdata[1] = b;
    rdata[2] = ep;

    /* Create and initialize serial vector for the solution */
    let y = N_VNew_Serial(NEQ, &ctx_h);
    if check_retval_null(&y, "N_VNew_Serial") != 0 {
        std::process::exit(1);
    }
    let y = y.unwrap();
    NV_Ith_S_set(&y, 0, u0);
    NV_Ith_S_set(&y, 1, v0);
    NV_Ith_S_set(&y, 2, w0);

    /*
     * Create the fast integrator and set options
     */

    /* Initialize the fast integrator. Specify the explicit fast right-hand side
    function in y'=fe(t,y)+fi(t,y)+ff(t,y), the initial time T0, and the
    initial dependent variable vector y. */
    let mut inner_arkode_mem = ARKStepCreate(Some(ff), None, T0, &y, &ctx_h);
    if check_retval_null(&inner_arkode_mem, "ARKStepCreate") != 0 {
        std::process::exit(1);
    }
    let inner = inner_arkode_mem.clone().unwrap();

    /* Attach user data to fast integrator */
    retval = ARKodeSetUserData(&inner, Some(Box::new(rdata)));
    if check_retval_int(retval, "ARKodeSetUserData") != 0 {
        std::process::exit(1);
    }

    /* Set the fast method */
    retval = ARKStepSetTableNum(&inner, -1, ARKODE_KNOTH_WOLKE_3_3);
    if check_retval_int(retval, "ARKStepSetTableNum") != 0 {
        std::process::exit(1);
    }

    /* Set the fast step size */
    retval = ARKodeSetFixedStep(&inner, hf);
    if check_retval_int(retval, "ARKodeSetFixedStep") != 0 {
        std::process::exit(1);
    }

    /* Create inner stepper */
    retval = ARKodeCreateMRIStepInnerStepper(&inner, &mut inner_stepper);
    if check_retval_int(retval, "ARKodeCreateMRIStepInnerStepper") != 0 {
        std::process::exit(1);
    }
    let stepper = inner_stepper.clone().unwrap();

    /*
     * Create the slow integrator and set options
     */

    /* Initialize the slow integrator. Specify the explicit slow right-hand side
    function in y'=fe(t,y)+fi(t,y)+ff(t,y), the initial time T0, the
    initial dependent variable vector y, and the fast integrator. */
    let mut arkode_mem = MRIStepCreate(Some(fs), None, T0, &y, &stepper, &ctx_h);
    if check_retval_null(&arkode_mem, "MRIStepCreate") != 0 {
        std::process::exit(1);
    }
    let ark = arkode_mem.clone().unwrap();

    /* Pass rdata to user functions */
    retval = ARKodeSetUserData(&ark, Some(Box::new(rdata)));
    if check_retval_int(retval, "ARKodeSetUserData") != 0 {
        std::process::exit(1);
    }

    /* Set the slow step size */
    retval = ARKodeSetFixedStep(&ark, hs);
    if check_retval_int(retval, "ARKodeSetFixedStep") != 0 {
        std::process::exit(1);
    }

    /*
     * Integrate ODE
     */

    /* Open output stream for results, output comment line */
    UFID = File::create("ark_brusselator_mri_solution.txt").expect("output file");
    let mut UFID = UFID;
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

    /* Main time-stepping loop: calls ARKodeEvolve to perform the
    integration, then prints results. Stops when the final time
    has been reached */
    t = T0;
    tout = T0 + dTout;
    print!("        t           u           v           w\n");
    print!("   ----------------------------------------------\n");
    print!(
        "  {}  {}  {}  {}\n",
        fmt_fw(t, 10, 6),
        fmt_fw(NV_Ith_S(&y, 0), 10, 6),
        fmt_fw(NV_Ith_S(&y, 1), 10, 6),
        fmt_fw(NV_Ith_S(&y, 2), 10, 6)
    );

    for _iout in 0..Nt {
        /* call integrator */
        retval = ARKodeEvolve(&ark, tout, &y, &mut t, ARK_NORMAL);
        if check_retval_int(retval, "ARKodeEvolve") != 0 {
            break;
        }

        /* access/print solution */
        print!(
            "  {}  {}  {}  {}\n",
            fmt_fw(t, 10, 6),
            fmt_fw(NV_Ith_S(&y, 0), 10, 6),
            fmt_fw(NV_Ith_S(&y, 1), 10, 6),
            fmt_fw(NV_Ith_S(&y, 2), 10, 6)
        );
        let _ = write!(
            UFID,
            " {} {} {} {}\n",
            fmt_e(t, 16),
            fmt_e(NV_Ith_S(&y, 0), 16),
            fmt_e(NV_Ith_S(&y, 1), 16),
            fmt_e(NV_Ith_S(&y, 2), 16)
        );

        /* successful solve: update time */
        tout += dTout;
        tout = if tout > Tf { Tf } else { tout };
    }
    print!("   ----------------------------------------------\n");
    drop(UFID);

    /*
     * Finalize
     */

    /* Get some slow integrator statistics */
    retval = ARKodeGetNumSteps(&ark, &mut nsts);
    check_retval_int(retval, "ARKodeGetNumSteps");
    retval = ARKodeGetNumRhsEvals(&ark, 0, &mut nfse);
    check_retval_int(retval, "ARKodeGetNumRhsEvals");

    /* Get some fast integrator statistics */
    retval = ARKodeGetNumSteps(&inner, &mut nstf);
    check_retval_int(retval, "ARKodeGetNumSteps");
    retval = ARKodeGetNumRhsEvals(&inner, 0, &mut nff);
    check_retval_int(retval, "ARKodeGetNumRhsEvals");

    /* Print some final statistics */
    print!("\nFinal Solver Statistics:\n");
    print!("   Steps: nsts = {}, nstf = {}\n", nsts, nstf);
    print!("   Total RHS evals:  Fs = {},  Ff = {}\n", nfse, nff);

    /* Clean up and return */
    N_VDestroy(y); /* Free y vector */
    ARKodeFree(&mut inner_arkode_mem); /* Free integrator memory */
    let _ = MRIStepInnerStepper_Free(&mut inner_stepper); /* Free inner stepper */
    ARKodeFree(&mut arkode_mem); /* Free integrator memory */
    let _ = SUNContext_Free(&mut ctx); /* Free context */
}

/* ------------------------------
 * Functions called by the solver
 * ------------------------------*/

/* ff routine to compute the fast portion of the ODE RHS. */
fn ff(
    _t: sunrealtype,
    y: &N_Vector,
    ydot: &N_Vector,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    let rdata = user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<[sunrealtype; 3]>())
        .expect("user_data"); /* cast user_data to sunrealtype */
    let b = rdata[1]; /* access data entries */
    let ep = rdata[2];
    let w = NV_Ith_S(y, 2); /* access solution values */

    /* fill in the RHS function */
    NV_Ith_S_set(ydot, 0, 0.0);
    NV_Ith_S_set(ydot, 1, 0.0);
    NV_Ith_S_set(ydot, 2, (b - w) / ep);

    /* Return with success */
    0
}

/* fs routine to compute the slow portion of the ODE RHS. */
fn fs(
    _t: sunrealtype,
    y: &N_Vector,
    ydot: &N_Vector,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    let rdata = user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<[sunrealtype; 3]>())
        .expect("user_data"); /* cast user_data to sunrealtype */
    let a = rdata[0]; /* access data entries */
    let u = NV_Ith_S(y, 0); /* access solution values */
    let v = NV_Ith_S(y, 1);
    let w = NV_Ith_S(y, 2);

    /* fill in the RHS function */
    NV_Ith_S_set(ydot, 0, a - (w + 1.0) * u + v * u * u);
    NV_Ith_S_set(ydot, 1, w * u - v * u * u);
    NV_Ith_S_set(ydot, 2, -w * u);

    /* Return with success */
    0
}

/* ------------------------------
 * Private helper functions
 * ------------------------------*/

/* Check function return value...
   opt == 0 means SUNDIALS function allocates memory so check if
            returned NULL pointer
   opt == 1 means SUNDIALS function returns a retval so check if
            retval < 0
   opt == 2 means function allocates memory so check if returned
            NULL pointer

   The C void-pointer/opt polymorphism splits into two typed helpers with
   identical message text:
     check_retval_null = opt == 0
     check_retval_int  = opt == 1
*/

fn check_retval_null<T>(returnvalue: &Option<T>, funcname: &str) -> i32 {
    /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
    if returnvalue.is_none() {
        eprint!(
            "\nSUNDIALS_ERROR: {}() failed - returned NULL pointer\n\n",
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

/*---- end of file ----*/
