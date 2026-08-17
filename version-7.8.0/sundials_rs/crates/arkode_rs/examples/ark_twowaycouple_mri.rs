//! Port of `examples/arkode/C_serial/ark_twowaycouple_mri.c`.
//!
//! Example problem:
//!
//! This example simulates an ODE system with 3 components,
//! Y = [u,v,w], given by the equations,
//!
//!   du/dt =  100v+w
//!   dv/dt = -100u
//!   dw/dt = -w+u
//!
//! for t in the interval [0.0, 2.0] with initial conditions
//! u(0)=9001/10001, v(0)=-1e-5/10001, and w(0)=1000. In this problem
//! the slow (w) and fast (u and v) components depend on one another.
//!
//! This program solves the problem with the MRI stepper. Outputs are
//! printed at equal intervals of 0.1 and run statistics are printed
//! at the end.

#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

use std::any::Any;
use std::fs::File;
use std::io::Write;

use arkode_rs::prelude::*;

/* C macro `NV_Ith_S(v,i)` (0-based). The RefMut guard lives only for the
statement that uses it, per the workspace granular-borrow rule. */

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
    let hs: sunrealtype = 0.001; /* slow step size */
    let hf: sunrealtype = 0.00002; /* fast step size */
    let u0: sunrealtype;
    let v0: sunrealtype;
    let w0: sunrealtype; /* initial conditions */

    /* general problem variables */
    let mut retval: i32; /* reusable error-checking flag */

    let mut nsts: i64 = 0;
    let mut nstf: i64 = 0;
    let mut nfse: i64 = 0;
    let mut nff: i64 = 0;

    /* Create the SUNDIALS context object for this simulation */
    let mut ctx: Option<SUNContext> = None;
    retval = SUNContext_Create(SUN_COMM_NULL, &mut ctx);
    if check_retval(retval, "SUNContext_Create") != 0 {
        std::process::exit(1);
    }
    let sunctx = ctx.clone().unwrap();

    /*
     * Initialization
     */

    /* Set the initial contions */
    u0 = 9001.0 / 10001.0;
    v0 = -1.0e5 / 10001.0;
    w0 = 1000.0;

    /* Initial problem output */
    print!("\nTwo way coupling ODE test problem:\n");
    print!(
        "    initial conditions:  u0 = {},  v0 = {},  w0 = {}\n",
        fmt_g(u0, 6),
        fmt_g(v0, 6),
        fmt_g(w0, 6)
    );
    print!("    hs = {},  hf = {}\n\n", fmt_g(hs, 6), fmt_g(hf, 6));

    /* Create and initialize serial vector for the solution */
    let y = N_VNew_Serial(NEQ, &sunctx);
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
    let mut inner_arkode_mem = ARKStepCreate(Some(ff), None, T0, &y, &sunctx);
    if check_retval_null(&inner_arkode_mem, "ARKStepCreate") != 0 {
        std::process::exit(1);
    }
    let inner_mem = inner_arkode_mem.clone().unwrap();

    /* Set the fast method */
    retval = ARKStepSetTableNum(&inner_mem, -1, ARKODE_KNOTH_WOLKE_3_3);
    if check_retval(retval, "ARKStepSetTableNum") != 0 {
        std::process::exit(1);
    }

    /* Set the fast step size */
    retval = ARKodeSetFixedStep(&inner_mem, hf);
    if check_retval(retval, "ARKodeSetFixedStep") != 0 {
        std::process::exit(1);
    }

    /* Create inner stepper */
    let mut inner_stepper: Option<MRIStepInnerStepper> = None;
    retval = ARKodeCreateMRIStepInnerStepper(&inner_mem, &mut inner_stepper);
    if check_retval(retval, "ARKodeCreateMRIStepInnerStepper") != 0 {
        std::process::exit(1);
    }
    let stepper = inner_stepper.clone().unwrap();

    /*
     * Create the slow integrator and set options
     */

    /* Initialize the slow integrator. Specify the explicit slow right-hand side
    function in y'=fe(t,y)+fi(t,y)+ff(t,y), the initial time T0, the
    initial dependent variable vector y, and the fast integrator. */
    let mut arkode_mem = MRIStepCreate(Some(fs), None, T0, &y, &stepper, &sunctx);
    if check_retval_null(&arkode_mem, "MRIStepCreate") != 0 {
        std::process::exit(1);
    }
    let mri_mem = arkode_mem.clone().unwrap();

    /* Set the slow step size */
    retval = ARKodeSetFixedStep(&mri_mem, hs);
    if check_retval(retval, "ARKodeSetFixedStep") != 0 {
        std::process::exit(1);
    }

    /*
     * Integrate ODE
     */

    /* Open output stream for results, output comment line */
    let mut UFID = File::create("ark_twowaycouple_mri_solution.txt").expect("output file");
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
    let mut t: sunrealtype = T0;
    let mut tout: sunrealtype = T0 + dTout;
    print!("        t           u           v           w\n");
    print!("   -----------------------------------------------\n");
    print!(
        "  {}  {}  {}  {}\n",
        fmt_fw(t, 10, 6),
        fmt_fw(NV_Ith_S(&y, 0), 10, 6),
        fmt_fw(NV_Ith_S(&y, 1), 10, 6),
        fmt_fw(NV_Ith_S(&y, 2), 10, 6)
    );

    for _iout in 0..Nt {
        /* call integrator */
        retval = ARKodeEvolve(&mri_mem, tout, &y, &mut t, ARK_NORMAL);
        if check_retval(retval, "ARKodeEvolve") != 0 {
            break;
        }

        /* access/print solution and error */
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
    print!("   -----------------------------------------------\n");
    drop(UFID);

    /*
     * Finalize
     */

    /* Get some slow integrator statistics */
    retval = ARKodeGetNumSteps(&mri_mem, &mut nsts);
    check_retval(retval, "ARKodeGetNumSteps");
    retval = ARKodeGetNumRhsEvals(&mri_mem, 0, &mut nfse);
    check_retval(retval, "ARKodeGetNumRhsEvals");

    /* Get some fast integrator statistics */
    retval = ARKodeGetNumSteps(&inner_mem, &mut nstf);
    check_retval(retval, "ARKodeGetNumSteps");
    retval = ARKodeGetNumRhsEvals(&inner_mem, 0, &mut nff);
    check_retval(retval, "ARKodeGetNumRhsEvals");

    /* Print some final statistics */
    print!("\nFinal Solver Statistics:\n");
    print!("   Steps: nsts = {}, nstf = {}\n", nsts, nstf);
    print!("   Total RHS evals:  Fs = {},  Ff = {}\n", nfse, nff);

    /* Clean up and return */
    N_VDestroy(y); /* Free y vector */
    ARKodeFree(&mut inner_arkode_mem); /* Free integrator memory */
    MRIStepInnerStepper_Free(&mut inner_stepper); /* Free inner stepper */
    ARKodeFree(&mut arkode_mem); /* Free integrator memory */
    let _ = SUNContext_Free(&mut ctx); /* Free context */
}

/* ------------------------------
 * Functions called by the solver
 * ------------------------------*/

/* ff routine to compute the fast portion of the ODE RHS. */
fn ff(_t: sunrealtype, y: &N_Vector, ydot: &N_Vector, _user_data: &mut Option<Box<dyn Any>>) -> i32 {
    let c1: sunrealtype = 100.0; /* problem constant */
    let u = NV_Ith_S(y, 0); /* access solution values */
    let v = NV_Ith_S(y, 1);

    /* fill in the RHS function */
    NV_Ith_S_set(ydot, 0, c1 * v);
    NV_Ith_S_set(ydot, 1, -c1 * u);
    NV_Ith_S_set(ydot, 2, u);

    /* Return with success */
    0
}

/* fs routine to compute the slow portion of the ODE RHS. */
fn fs(_t: sunrealtype, y: &N_Vector, ydot: &N_Vector, _user_data: &mut Option<Box<dyn Any>>) -> i32 {
    let w = NV_Ith_S(y, 2); /* access solution values */

    /* fill in the RHS function */
    NV_Ith_S_set(ydot, 0, w);
    NV_Ith_S_set(ydot, 1, 0.0);
    NV_Ith_S_set(ydot, 2, -w);

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

/*---- end of file ----*/
