/* -----------------------------------------------------------------------------
 * Rust port of examples/cvode/serial/cvParticle_dns.c
 * -----------------------------------------------------------------------------
 * This example solves the equation for a particle moving conterclockwise with
 * velocity alpha on the unit circle in the xy-plane. The ODE system is given by
 *
 *   x' = -alpha * y
 *   y' =  alpha * x
 *
 * where x and y are subject to the constraint
 *
 *   x^2 + y^2 - 1 = 0
 *
 * with initial condition x = 1 and y = 0 at t = 0.
 * ---------------------------------------------------------------------------*/
#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

use cvode_rs::prelude::*;

use std::any::Any;
use std::fs::File;
use std::io::Write;

/* Problem Constants */
const PI: sunrealtype = 3.141592653589793238462643383279502884197169;
const ZERO: sunrealtype = 0.0;
const ONE: sunrealtype = 1.0;
const TWO: sunrealtype = 2.0;

/* User-defined data structure */
#[derive(Clone)]
struct UserData {
    alpha: sunrealtype, /* particle velocity */

    orbits: i32,         /* number of orbits */
    torbit: sunrealtype, /* orbit time       */

    rtol: sunrealtype, /* integration tolerances */
    atol: sunrealtype,

    proj: i32,    /* enable/disable solution projection */
    projerr: i32, /* enable/disable error projection */

    tstop: i32, /* use tstop mode */
    nout: i32,  /* number of outputs per orbit */
}

/* -----------------------------------------------------------------------------
 * Main Program
 * ---------------------------------------------------------------------------*/

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut retval: i32; /* reusable return flag       */
    let mut t: sunrealtype = ZERO; /* current integration time   */
    let mut ec: sunrealtype = ZERO; /* constraint error           */

    /* Create the SUNDIALS context */
    let mut sunctx_opt: Option<SUNContext> = None;
    retval = SUNContext_Create(SUN_COMM_NULL, &mut sunctx_opt);
    if check_retval_int(retval, "SUNContext_Create") {
        std::process::exit(1);
    }
    let sunctx = sunctx_opt.unwrap();

    /* Allocate and initialize user data structure */
    let mut udata = UserData {
        alpha: ZERO,
        orbits: 0,
        torbit: ZERO,
        rtol: ZERO,
        atol: ZERO,
        proj: 0,
        projerr: 0,
        tstop: 0,
        nout: 0,
    };

    retval = InitUserData(&args, &mut udata);
    if check_retval_int(retval, "InitUserData") {
        std::process::exit(1);
    }

    /* Create serial vector to store the solution */
    let y = N_VNew_Serial(2, &sunctx);
    if check_retval_null(&y, "N_VNew_Serial") {
        std::process::exit(1);
    }
    let y = y.unwrap();

    /* Set initial contion */
    {
        let mut ydata = N_VGetArrayPointer(&y).expect("vector data");
        ydata[0] = ONE;
        ydata[1] = ZERO;
    }

    /* Create serial vector to store the solution error */
    let e = N_VClone(&y);
    if check_retval_null(&e, "N_VClone") {
        std::process::exit(1);
    }
    let e = e.unwrap();

    /* Set initial error */
    N_VConst(ZERO, &e);

    /* Create CVODE memory */
    let cvode_mem = CVodeCreate(CV_BDF, &sunctx);
    if check_retval_null(&cvode_mem, "CVodeCreate") {
        std::process::exit(1);
    }
    let cvode_mem = cvode_mem.unwrap();

    /* Initialize CVODE */
    retval = CVodeInit(&cvode_mem, f, t, &y);
    if check_retval_int(retval, "CVodeInit") {
        std::process::exit(1);
    }

    /* Attach user-defined data structure to CVODE */
    retval = CVodeSetUserData(&cvode_mem, Some(Box::new(udata.clone())));
    if check_retval_int(retval, "CVodeSetUserData") {
        std::process::exit(1);
    }

    /* Set integration tolerances */
    retval = CVodeSStolerances(&cvode_mem, udata.rtol, udata.atol);
    if check_retval_int(retval, "CVodeSStolerances") {
        std::process::exit(1);
    }

    /* Create dense SUNMatrix for use in linear solves */
    let A = SUNDenseMatrix(2, 2, &sunctx);
    if check_retval_null(&A, "SUNDenseMatrix") {
        std::process::exit(1);
    }
    let A = A.unwrap();

    /* Create dense SUNLinearSolver object */
    let LS = SUNLinSol_Dense(&y, &A, &sunctx);
    if check_retval_null(&LS, "SUNLinSol_Dense") {
        std::process::exit(1);
    }
    let LS = LS.unwrap();

    /* Attach the matrix and linear solver to CVODE */
    retval = CVodeSetLinearSolver(&cvode_mem, &LS, Some(&A));
    if check_retval_int(retval, "CVodeSetLinearSolver") {
        std::process::exit(1);
    }

    /* Set a user-supplied Jacobian function */
    retval = CVodeSetJacFn(&cvode_mem, Some(Jac));
    if check_retval_int(retval, "CVodeSetJacFn") {
        std::process::exit(1);
    }

    /* Set a user-supplied projection function */
    if udata.proj != 0 {
        retval = CVodeSetProjFn(&cvode_mem, Some(Proj));
        if check_retval_int(retval, "CVodeSetProjFn") {
            std::process::exit(1);
        }

        retval = CVodeSetProjErrEst(&cvode_mem, udata.projerr != 0);
        if check_retval_int(retval, "CVodeSetProjErrEst") {
            std::process::exit(1);
        }
    }

    /* Set max steps between outputs */
    retval = CVodeSetMaxNumSteps(&cvode_mem, 100000);
    if check_retval_int(retval, "CVodeSetMaxNumSteps") {
        std::process::exit(1);
    }

    /* Output problem setup */
    retval = PrintUserData(&udata);
    if check_retval_int(retval, "PrintUserData") {
        std::process::exit(1);
    }

    /* Output initial condition */
    print!("\n     t            x              y");
    print!("             err x          err y       err constr\n");
    let _ = WriteOutput(t, &y, &e, ec, 0, &mut None, &mut None);

    let mut YFID: Option<File> = None; /* solution output file */
    let mut EFID: Option<File> = None; /* error output file    */

    if udata.nout > 0 {
        YFID = File::create("cvParticle_solution.txt").ok();
        EFID = File::create("cvParticle_error.txt").ok();
        let _ = WriteOutput(t, &y, &e, ec, 1, &mut YFID, &mut EFID);
    }

    /* Integrate in time and periodically output the solution and error */
    let totalout: i32; /* output counter */
    let dtout: sunrealtype; /* output spacing */
    if udata.nout > 0 {
        totalout = udata.orbits * udata.nout;
        dtout = udata.torbit / udata.nout as sunrealtype;
    } else {
        totalout = 1;
        dtout = udata.torbit * udata.orbits as sunrealtype;
    }
    let mut tout = dtout; /* next output time */

    for out in 0..totalout {
        /* Stop at output time (do not interpolate output) */
        if udata.tstop != 0 || udata.nout == 0 {
            retval = CVodeSetStopTime(&cvode_mem, tout);
            if check_retval_int(retval, "CVodeSetStopTime") {
                std::process::exit(1);
            }
        }

        /* Advance in time */
        retval = CVode(&cvode_mem, tout, &y, &mut t, CV_NORMAL);
        if check_retval_int(retval, "CVode") {
            break;
        }

        /* Output solution and error */
        if udata.nout > 0 {
            retval = ComputeError(t, &y, &e, &mut ec, &udata);
            if check_retval_int(retval, "ComputeError") {
                break;
            }

            let _ = WriteOutput(t, &y, &e, ec, 1, &mut YFID, &mut EFID);
            if check_retval_int(retval, "WriteOutput") {
                break;
            }
        }

        /* Update output time */
        if out < totalout - 1 {
            tout += dtout;
        } else {
            tout = udata.torbit * udata.orbits as sunrealtype;
        }
    }

    /* Close output files */
    if udata.nout > 0 {
        drop(YFID.take());
        drop(EFID.take());
    }

    /* Output final solution and error to screen */
    let _ = ComputeError(t, &y, &e, &mut ec, &udata);
    if check_retval_int(retval, "ComputeError") {
        std::process::exit(1);
    }

    let _ = WriteOutput(t, &y, &e, ec, 0, &mut None, &mut None);
    if check_retval_int(retval, "WriteOutput") {
        std::process::exit(1);
    }

    /* Print some final statistics */
    let _ = PrintStats(&cvode_mem);

    /* Free memory */
    N_VDestroy(y);
    N_VDestroy(e);
    SUNMatDestroy(A);
    SUNLinSolFree(Some(LS));
    let mut cvode_mem = Some(cvode_mem);
    CVodeFree(&mut cvode_mem);
    let mut sunctx = Some(sunctx);
    SUNContext_Free(&mut sunctx);
}

/* -----------------------------------------------------------------------------
 * Functions provided to CVODE
 * ---------------------------------------------------------------------------*/

/* Compute the right-hand side function, y' = f(t,y) */
fn f(_t: sunrealtype, y: &N_Vector, ydot: &N_Vector, user_data: &mut Option<Box<dyn Any>>) -> i32 {
    let udata = user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<UserData>())
        .expect("user data");
    let ydata = N_VGetArrayPointer(y).expect("vector data");
    let mut fdata = N_VGetArrayPointer(ydot).expect("vector data");

    fdata[0] = -(udata.alpha) * ydata[1];
    fdata[1] = (udata.alpha) * ydata[0];

    0
}

/* Compute the Jacobian of the right-hand side function, J(t,y) = df/dy */
fn Jac(
    _t: sunrealtype,
    _y: &N_Vector,
    _fy: &N_Vector,
    J: &SUNMatrix,
    user_data: &mut Option<Box<dyn Any>>,
    _tmp1: &N_Vector,
    _tmp2: &N_Vector,
    _tmp3: &N_Vector,
) -> i32 {
    let udata = user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<UserData>())
        .expect("user data");
    let mut Jdata = SUNDenseMatrix_Data(J);

    Jdata[0] = ZERO;
    Jdata[1] = -(udata.alpha);
    Jdata[2] = udata.alpha;
    Jdata[3] = ZERO;

    0
}

/* Project the solution onto the constraint manifold */
fn Proj(
    _t: sunrealtype,
    ycur: &N_Vector,
    corr: &N_Vector,
    _epsProj: sunrealtype,
    err: Option<&N_Vector>,
    _user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    let (x, y) = {
        let ydata = N_VGetArrayPointer(ycur).expect("vector data");
        (ydata[0], ydata[1])
    };

    /* project onto the unit circle */
    let r = (x * x + y * y).sqrt();

    let xp = x / r;
    let yp = y / r;

    /* correction to the unprojected solution */
    {
        let mut cdata = N_VGetArrayPointer(corr).expect("vector data");
        cdata[0] = xp - x;
        cdata[1] = yp - y;
    }

    /* project the error */
    if let Some(err) = err {
        let mut edata = N_VGetArrayPointer(err).expect("vector data");

        let errxp = edata[0] * yp * yp - edata[1] * xp * yp;
        let erryp = -edata[0] * xp * yp + edata[1] * xp * xp;

        edata[0] = errxp;
        edata[1] = erryp;
    }

    0
}

/* -----------------------------------------------------------------------------
 * Private helper functions
 * ---------------------------------------------------------------------------*/

fn InitUserData(args: &[String], udata: &mut UserData) -> i32 {
    let mut arg_idx = 1usize;

    /* set default values */
    udata.alpha = ONE;

    udata.orbits = 100;
    udata.torbit = (TWO * PI) / udata.alpha;

    udata.rtol = 1.0e-4;
    udata.atol = 1.0e-9;

    udata.proj = 1;
    udata.projerr = 0;

    udata.tstop = 0;
    udata.nout = 0;

    /* check for input args */
    while arg_idx < args.len() {
        if args[arg_idx] == "--alpha" {
            arg_idx += 1;
            udata.alpha = SUNStrToReal(&args[arg_idx]);
            arg_idx += 1;
            udata.torbit = (TWO * PI) / udata.alpha;
        } else if args[arg_idx] == "--orbits" {
            arg_idx += 1;
            udata.orbits = atoi(&args[arg_idx]);
            arg_idx += 1;
        } else if args[arg_idx] == "--rtol" {
            arg_idx += 1;
            udata.rtol = SUNStrToReal(&args[arg_idx]);
            arg_idx += 1;
        } else if args[arg_idx] == "--atol" {
            arg_idx += 1;
            udata.atol = SUNStrToReal(&args[arg_idx]);
            arg_idx += 1;
        } else if args[arg_idx] == "--proj" {
            arg_idx += 1;
            udata.proj = atoi(&args[arg_idx]);
            arg_idx += 1;
        } else if args[arg_idx] == "--projerr" {
            arg_idx += 1;
            udata.projerr = atoi(&args[arg_idx]);
            arg_idx += 1;
        } else if args[arg_idx] == "--nout" {
            arg_idx += 1;
            udata.nout = atoi(&args[arg_idx]);
            arg_idx += 1;
        } else if args[arg_idx] == "--tstop" {
            arg_idx += 1;
            udata.tstop = 1;
        } else if args[arg_idx] == "--help" {
            InputHelp();
            return -1;
        } else {
            eprint!("ERROR: Invalid input {}", args[arg_idx]);
            InputHelp();
            return -1;
        }
    }

    /* If projection is disabled then disable error projection */
    if udata.proj == 0 {
        udata.projerr = 0;
    }

    0
}

fn PrintUserData(udata: &UserData) -> i32 {
    print!("\nParticle traveling on the unit circle example\n");
    print!("---------------------------------------------\n");
    print!("alpha      = {}\n", fmt_e(udata.alpha, 4));
    print!("num orbits = {}\n", udata.orbits);
    print!("---------------------------------------------\n");
    print!("rtol       = {}\n", fmt_g(udata.rtol, 6));
    print!("atol       = {}\n", fmt_g(udata.atol, 6));
    print!("proj sol   = {}\n", udata.proj);
    print!("proj err   = {}\n", udata.projerr);
    print!("nout       = {}\n", udata.nout);
    print!("tstop      = {}\n", udata.tstop);
    print!("---------------------------------------------\n");

    0
}

/* Print command line options */
fn InputHelp() {
    print!("\nCommand line options:\n");
    print!("  --alpha <vel>      : particle velocity\n");
    print!("  --orbits <orbits>  : number of orbits to perform\n");
    print!("  --rtol <rtol>      : relative tolerance\n");
    print!("  --atol <atol>      : absolute tolerance\n");
    print!("  --proj <1 or 0>    : enable (1) / disable (0) projection\n");
    print!("  --projerr <1 or 0> : enable (1) / disable (0) error projection\n");
    print!("  --nout <nout>      : outputs per period\n");
    print!("  --tstop            : stop at output time (do not interpolate)\n");
}

/* Compute the analytical solution */
fn ComputeSolution(t: sunrealtype, y: &N_Vector, udata: &UserData) -> i32 {
    let mut ydata = N_VGetArrayPointer(y).expect("vector data");

    ydata[0] = ((udata.alpha) * t).sun_cos();
    ydata[1] = ((udata.alpha) * t).sun_sin();

    0
}

/* Compute the error in the solution and constraint */
fn ComputeError(
    t: sunrealtype,
    y: &N_Vector,
    e: &N_Vector,
    ec: &mut sunrealtype,
    udata: &UserData,
) -> i32 {
    /* solution error */
    let retval = ComputeSolution(t, e, udata);
    if check_retval_int(retval, "ComputeSolution") {
        return 1;
    }
    N_VLinearSum(ONE, y, -ONE, e, e);

    /* constraint error */
    let ydata = N_VGetArrayPointer(y).expect("vector data");
    *ec = ydata[0] * ydata[0] + ydata[1] * ydata[1] - ONE;

    0
}

/* Output the solution to the screen or disk */
fn WriteOutput(
    t: sunrealtype,
    y: &N_Vector,
    e: &N_Vector,
    ec: sunrealtype,
    screenfile: i32,
    YFID: &mut Option<File>,
    EFID: &mut Option<File>,
) -> i32 {
    let ydata = N_VGetArrayPointer(y).expect("vector data");
    let edata = N_VGetArrayPointer(e).expect("vector data");

    if screenfile == 0 {
        /* output solution and error to screen */
        print!(
            "{} {} {} {} {} {}\n",
            fmt_e(t, 4),
            fmt_ew(ydata[0], 14, 6),
            fmt_ew(ydata[1], 14, 6),
            fmt_ew(edata[0], 14, 6),
            fmt_ew(edata[1], 14, 6),
            fmt_ew(ec, 14, 6)
        );
    } else {
        /* check file pointers */
        if YFID.is_none() || EFID.is_none() {
            return 1;
        }

        /* output solution to disk */
        let yf = YFID.as_mut().unwrap();
        let _ = write!(
            yf,
            "{} {} {}\n",
            fmt_ew(t, 24, 16),
            fmt_ew(ydata[0], 24, 16),
            fmt_ew(ydata[1], 24, 16)
        );

        /* output error to disk */
        let ef = EFID.as_mut().unwrap();
        let _ = write!(
            ef,
            "{} {} {} {}\n",
            fmt_ew(t, 24, 16),
            fmt_ew(edata[0], 24, 16),
            fmt_ew(edata[1], 24, 16),
            fmt_ew(ec, 24, 16)
        );
    }

    0
}

/* Print final statistics */
fn PrintStats(cvode_mem: &CVodeMem) -> i32 {
    let mut nst: i64 = 0;
    let mut nfe: i64 = 0;
    let mut nsetups: i64 = 0;
    let mut nje: i64 = 0;
    let mut nni: i64 = 0;
    let mut ncfn: i64 = 0;
    let mut netf: i64 = 0;

    let mut retval = CVodeGetNumSteps(cvode_mem, &mut nst);
    check_retval_int(retval, "CVodeGetNumSteps");
    retval = CVodeGetNumRhsEvals(cvode_mem, &mut nfe);
    check_retval_int(retval, "CVodeGetNumRhsEvals");
    retval = CVodeGetNumLinSolvSetups(cvode_mem, &mut nsetups);
    check_retval_int(retval, "CVodeGetNumLinSolvSetups");
    retval = CVodeGetNumErrTestFails(cvode_mem, &mut netf);
    check_retval_int(retval, "CVodeGetNumErrTestFails");
    retval = CVodeGetNumNonlinSolvIters(cvode_mem, &mut nni);
    check_retval_int(retval, "CVodeGetNumNonlinSolvIters");
    retval = CVodeGetNumNonlinSolvConvFails(cvode_mem, &mut ncfn);
    check_retval_int(retval, "CVodeGetNumNonlinSolvConvFails");

    retval = CVodeGetNumJacEvals(cvode_mem, &mut nje);
    check_retval_int(retval, "CVodeGetNumJacEvals");

    print!("\nIntegration Statistics:\n");

    print!("Number of steps taken = {:<6}\n", nst);
    print!("Number of function evaluations = {:<6}\n", nfe);

    print!("Number of linear solver setups = {:<6}\n", nsetups);
    print!("Number of Jacobian evaluations = {:<6}\n", nje);

    print!("Number of nonlinear solver iterations = {:<6}\n", nni);
    print!("Number of convergence failures = {:<6}\n", ncfn);
    print!("Number of error test failures = {:<6}\n", netf);

    0
}

/* Check function return value (C check_retval, opt 1) */
fn check_retval_int(retval: i32, funcname: &str) -> bool {
    if retval < 0 {
        eprint!("\nERROR: {}() returned = {}\n\n", funcname, retval);
        return true;
    }
    false
}

/* Check function return value (C check_retval, opt 0) */
fn check_retval_null<T>(returnvalue: &Option<T>, funcname: &str) -> bool {
    if returnvalue.is_none() {
        eprint!("\nERROR: {}() returned a NULL pointer\n\n", funcname);
        return true;
    }
    false
}
