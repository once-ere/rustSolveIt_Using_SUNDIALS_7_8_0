/* ------------------------------------------------------------------
 * Rust port of examples/arkode/C_serial/ark_reaction_diffusion_mri.c
 * Programmer(s): David J. Gardner @ LLNL
 * ------------------------------------------------------------------
 * Based an example program by Rujeko Chinomona @ UMBC.
 * ------------------------------------------------------------------
 * Example problem:
 *
 * The following test simulates a simple 1D reaction-diffusion
 * equation,
 *
 *   y_t = k * y_xx + y^2 * (1-y)
 *
 * for t in [0, 3], x in [0, L] with boundary conditions,
 *
 *   y_x(0,t) = y_x(L,t) = 0
 *
 * and initial condition,
 *
 *   y(x,0) = (1 + exp(lambda*(x-1))^(-1),
 *
 * with parameter k = 1e-4/ep, lambda = 0.5*sqrt(2*ep*1e4),
 * ep = 1e-2, and L = 5.
 *
 * The spatial derivatives are computed using second-order
 * centered differences, with the data distributed over N points
 * on a uniform spatial grid.
 *
 * This program solves the problem with the MRI stepper. Outputs are
 * printed at equal intervals of 0.1 and run statistics are printed
 * at the end.
 * ----------------------------------------------------------------*/

#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

use arkode_rs::prelude::*;

use arkode_rs::sundials_futils::SUNFileClose;
use std::any::Any;
use std::fs::File;
use std::io::Write;

/* user data structure */
#[derive(Clone)]
struct UserData {
    N: sunindextype,  /* number of intervals   */
    dx: sunrealtype,  /* mesh spacing          */
    k: sunrealtype,   /* diffusion coefficient */
    lam: sunrealtype, /* initial-condition steepness */
}

/* Main Program */
fn main() {
    /* general problem parameters */
    let T0: sunrealtype = 0.0; /* initial time */
    let Tf: sunrealtype = 3.0; /* final time */
    let mut dTout: sunrealtype = 0.1; /* time between outputs */
    let Nt: i32 = (Tf / dTout).ceil() as i32; /* number of output times */
    let hs: sunrealtype = 0.001; /* slow step size */
    let hf: sunrealtype = 0.00002; /* fast step size */
    let udata: UserData; /* user data */

    let L: sunrealtype = 5.0; /* domain length */
    let N: sunindextype = 1001; /* number of mesh points */
    let ep: sunrealtype = 1e-2;
    let mut i: sunindextype;

    /* general problem variables */
    let mut retval: i32; /* reusable error-checking flag */
    let mut inner_stepper: Option<MRIStepInnerStepper> = None; /* inner stepper */
    let mut t: sunrealtype;
    let mut tout: sunrealtype;

    /* Create the SUNDIALS context object for this simulation */
    let mut ctx: Option<SUNContext> = None;
    retval = SUNContext_Create(SUN_COMM_NULL, &mut ctx);
    if check_retval_int(retval, "SUNContext_Create") != 0 {
        std::process::exit(1);
    }
    let sunctx = ctx.clone().expect("SUNContext");

    /*
     * Initialization
     */

    /* allocate and fill user data structure (C: malloc; the Rust
    allocation cannot fail) */
    udata = UserData {
        N,
        dx: L / (1.0 * N as sunrealtype - 1.0),
        k: 1e-4 / ep,
        lam: 0.5 * (2.0 * ep * 1e4).sqrt(),
    };

    /* Initial problem output */
    print!("\n1D reaction-diffusion PDE test problem:\n");
    print!("  N = {}\n", udata.N as i64);
    print!("  diffusion coefficient:  k = {}\n", fmt_g(udata.k, 6));

    /* Create and initialize serial vector for the solution */
    let y = N_VNew_Serial(N, &sunctx);
    if check_retval_null(&y, "N_VNew_Serial") != 0 {
        std::process::exit(1);
    }
    let y = y.expect("N_VNew_Serial");

    retval = SetInitialCondition(&y, &udata);
    if check_retval_int(retval, "SetInitialCondition") != 0 {
        std::process::exit(1);
    }

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
    let inner_mem = inner_arkode_mem.clone().expect("ARKStepCreate");

    /* Attach user data to fast integrator */
    retval = ARKodeSetUserData(&inner_mem, Some(Box::new(udata.clone())));
    if check_retval_int(retval, "ARKodeSetUserData") != 0 {
        std::process::exit(1);
    }

    /* Set the fast method */
    retval = ARKStepSetTableNum(&inner_mem, -1, ARKODE_KNOTH_WOLKE_3_3);
    if check_retval_int(retval, "ARKStepSetTableNum") != 0 {
        std::process::exit(1);
    }

    /* Set the fast step size */
    retval = ARKodeSetFixedStep(&inner_mem, hf);
    if check_retval_int(retval, "ARKodeSetFixedStep") != 0 {
        std::process::exit(1);
    }

    /* Create inner stepper */
    retval = ARKodeCreateMRIStepInnerStepper(&inner_mem, &mut inner_stepper);
    if check_retval_int(retval, "ARKodeCreateMRIStepInnerStepper") != 0 {
        std::process::exit(1);
    }
    let stepper = inner_stepper.clone().expect("MRIStepInnerStepper");

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
    let mri_mem = arkode_mem.clone().expect("MRIStepCreate");

    /* Pass udata to user functions */
    retval = ARKodeSetUserData(&mri_mem, Some(Box::new(udata.clone())));
    if check_retval_int(retval, "ARKodeSetUserData") != 0 {
        std::process::exit(1);
    }

    /* Set the slow step size */
    retval = ARKodeSetFixedStep(&mri_mem, hs);
    if check_retval_int(retval, "ARKodeSetFixedStep") != 0 {
        std::process::exit(1);
    }

    /* Increase max num steps  */
    retval = ARKodeSetMaxNumSteps(&mri_mem, 10000);
    if check_retval_int(retval, "ARKodeSetMaxNumSteps") != 0 {
        std::process::exit(1);
    }

    /*
     * Integrate ODE
     */

    /* output mesh to disk */
    let mut FID = File::create("heat_mesh.txt").expect("heat_mesh.txt");
    i = 0;
    while i < N {
        let _ = write!(FID, "  {}\n", fmt_e(udata.dx * i as sunrealtype, 16));
        i += 1;
    }
    drop(FID);

    /* Open output stream for results, access data array */
    let mut UFID = File::create("heat1D.txt").expect("heat1D.txt");

    /* output initial condition to disk */
    {
        let data = N_VGetArrayPointer(&y).expect("N_VGetArrayPointer");
        i = 0;
        while i < N {
            let _ = write!(UFID, " {}", fmt_e(data[i as usize], 16));
            i += 1;
        }
    }
    let _ = write!(UFID, "\n");

    /* Main time-stepping loop: calls ARKodeEvolve to perform the integration, then
    prints results. Stops when the final time has been reached */
    t = T0;
    dTout = (Tf - T0) / Nt as sunrealtype;
    tout = T0 + dTout;
    print!("        t      ||u||_rms\n");
    print!("   -------------------------\n");
    print!(
        "  {}  {}\n",
        fmt_fw(t, 10, 6),
        fmt_fw((N_VDotProd(&y, &y) / N as sunrealtype).sqrt(), 10, 6)
    );
    for _iout in 0..Nt {
        /* call integrator */
        retval = ARKodeEvolve(&mri_mem, tout, &y, &mut t, ARK_NORMAL);
        if check_retval_int(retval, "ARKodeEvolve") != 0 {
            break;
        }

        /* print solution stats and output results to disk */
        print!(
            "  {}  {}\n",
            fmt_fw(t, 10, 6),
            fmt_fw((N_VDotProd(&y, &y) / N as sunrealtype).sqrt(), 10, 6)
        );
        {
            let data = N_VGetArrayPointer(&y).expect("N_VGetArrayPointer");
            i = 0;
            while i < N {
                let _ = write!(UFID, " {}", fmt_e(data[i as usize], 16));
                i += 1;
            }
        }
        let _ = write!(UFID, "\n");

        /* successful solve: update output time */
        tout += dTout;
        tout = if tout > Tf { Tf } else { tout };
    }
    print!("   -------------------------\n");
    drop(UFID);

    /* Print final statistics to the screen */
    print!("\nFinal Slow Statistics:\n");
    let _ = ARKodePrintAllStats(
        &mri_mem,
        &SUNFile::Stdout,
        SUNOutputFormat::SUN_OUTPUTFORMAT_TABLE,
    );
    print!("\nFinal Fast Statistics:\n");
    let _ = ARKodePrintAllStats(
        &inner_mem,
        &SUNFile::Stdout,
        SUNOutputFormat::SUN_OUTPUTFORMAT_TABLE,
    );

    /* Print final statistics to a file in CSV format */
    let mut FID = SUNFile::fopen("ark_reaction_diffusion_mri_slow_stats.csv", "w");
    let _ = ARKodePrintAllStats(&mri_mem, &FID, SUNOutputFormat::SUN_OUTPUTFORMAT_CSV);
    SUNFileClose(&mut FID);
    let mut FID = SUNFile::fopen("ark_reaction_diffusion_mri_fast_stats.csv", "w");
    let _ = ARKodePrintAllStats(&inner_mem, &FID, SUNOutputFormat::SUN_OUTPUTFORMAT_CSV);
    SUNFileClose(&mut FID);

    /* Clean up and return */
    N_VDestroy(y); /* Free y vector */
    ARKodeFree(&mut inner_arkode_mem); /* Free integrator memory */
    let _ = MRIStepInnerStepper_Free(&mut inner_stepper); /* Free inner stepper */
    ARKodeFree(&mut arkode_mem); /* Free integrator memory */
    drop(udata); /* Free user data */
    let _ = SUNContext_Free(&mut ctx); /* Free context */
}

/* ------------------------------
 * Functions called by the solver
 * ------------------------------*/

/* ff routine to compute the fast portion of the ODE RHS. */
fn ff(_t: sunrealtype, y: &N_Vector, ydot: &N_Vector, user_data: &mut Option<Box<dyn Any>>) -> i32 {
    let udata = user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<UserData>())
        .expect("user_data"); /* access problem data */
    let N = udata.N; /* set variable shortcuts */
    let mut i: sunindextype;

    /* access state array data */
    let Y = N_VGetArrayPointer(y);
    if check_retval_null(&Y, "N_VGetArrayPointer") != 0 {
        return 1;
    }
    let Y = Y.expect("N_VGetArrayPointer");

    /* access RHS array data */
    let Ydot = N_VGetArrayPointer(ydot);
    if check_retval_null(&Ydot, "N_VGetArrayPointer") != 0 {
        return 1;
    }
    let mut Ydot = Ydot.expect("N_VGetArrayPointer");

    /* iterate over domain, computing reaction term */
    i = 0;
    while i < N {
        let iu = i as usize;
        Ydot[iu] = Y[iu] * Y[iu] * (1.0 - Y[iu]);
        i += 1;
    }

    /* Return with success */
    0
}

/* fs routine to compute the slow portion of the ODE RHS. */
fn fs(_t: sunrealtype, y: &N_Vector, ydot: &N_Vector, user_data: &mut Option<Box<dyn Any>>) -> i32 {
    let udata = user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<UserData>())
        .expect("user_data"); /* access problem data */
    let N = udata.N; /* set variable shortcuts */
    let k = udata.k;
    let dx = udata.dx;
    let c1: sunrealtype;
    let c2: sunrealtype;
    let mut i: sunindextype;

    /* access state array data */
    let Y = N_VGetArrayPointer(y);
    if check_retval_null(&Y, "N_VGetArrayPointer") != 0 {
        return 1;
    }
    let Y = Y.expect("N_VGetArrayPointer");

    /* access RHS array data */
    let Ydot = N_VGetArrayPointer(ydot);
    if check_retval_null(&Ydot, "N_VGetArrayPointer") != 0 {
        return 1;
    }
    let mut Ydot = Ydot.expect("N_VGetArrayPointer");

    /* iterate over domain, computing diffusion term */
    c1 = k / dx / dx;
    c2 = 2.0 * k / dx / dx;

    /* left boundary condition */
    Ydot[0] = c2 * (Y[1] - Y[0]);

    /* interior points */
    i = 1;
    while i < N - 1 {
        let iu = i as usize;
        Ydot[iu] = c1 * Y[iu - 1] - c2 * Y[iu] + c1 * Y[iu + 1];
        i += 1;
    }

    /* right boundary condition */
    Ydot[(N - 1) as usize] = c2 * (Y[(N - 2) as usize] - Y[(N - 1) as usize]);

    /* Return with success */
    0
}

/* -----------------------------------------
 * Private function to set initial condition
 * -----------------------------------------*/

fn SetInitialCondition(y: &N_Vector, user_data: &UserData) -> i32 {
    let udata = user_data; /* access problem data */
    let N = udata.N; /* set variable shortcuts */
    let lam = udata.lam;
    let dx = udata.dx;
    let mut i: sunindextype;

    /* access state array data */
    let Y = N_VGetArrayPointer(y);
    if check_retval_null(&Y, "N_VGetArrayPointer") != 0 {
        return -1;
    }
    let mut Y = Y.expect("N_VGetArrayPointer");

    /* set initial condition */
    i = 0;
    while i < N {
        Y[i as usize] = 1.0 / (1.0 + (lam * (i as sunrealtype * dx - 1.0)).sun_exp());
        i += 1;
    }

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

fn check_retval_int(retval: i32, funcname: &str) -> i32 {
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
