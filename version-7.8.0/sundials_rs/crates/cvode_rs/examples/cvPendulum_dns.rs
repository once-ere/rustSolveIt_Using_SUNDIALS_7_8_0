/* -----------------------------------------------------------------------------
 * Rust port of examples/cvode/serial/cvPendulum_dns.c
 * -----------------------------------------------------------------------------
 * This example solves a simple pendulum equation in Cartesian coordinates where
 * the pendulum bob has mass 1 and is suspended from the origin with a rod of
 * length 1. The governing equations are
 *
 * x'  = vx
 * y'  = vy
 * vx' = -x * T
 * vy' = -y * T - g
 *
 * with the constraints
 *
 * x^2 + y^2 - 1 = 0
 * x * vx + y * vy = 0
 *
 * The Cartesian formulation is run to a final time tf (default 30) with and
 * without projection for various integration tolerances.
 * ---------------------------------------------------------------------------*/
#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

use cvode_rs::prelude::*;

use std::any::Any;
use std::fs::File;
use std::io::Write;

/* Problem Constants */
const ZERO: sunrealtype = 0.0;
const ONE: sunrealtype = 1.0;
const GRAV: sunrealtype = 13.750371636040745654980191559621114395801712;

/* -----------------------------------------------------------------------------
 * Main Program
 * ---------------------------------------------------------------------------*/

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut retval: i32; /* reusable return flag    */
    let mut nout: i32 = 1; /* number of outputs       */
    let mut rtol: sunrealtype = 1.0e-5; /* base relative tolerance */
    let mut atol: sunrealtype = 1.0e-5; /* base absolute tolerance */
    let mut tf: sunrealtype = 30.0; /* final integration time  */
    let mut projerr: sunbooleantype = SUNTRUE; /* enable error projection */

    /* Create the SUNDIALS context */
    let mut sunctx_opt: Option<SUNContext> = None;
    retval = SUNContext_Create(SUN_COMM_NULL, &mut sunctx_opt);
    if check_retval_int(retval, "SUNContext_Create") {
        std::process::exit(1);
    }
    let sunctx = sunctx_opt.unwrap();

    /* Read command line inputs */
    retval = ReadInputs(
        &args,
        &mut rtol,
        &mut atol,
        &mut tf,
        &mut nout,
        &mut projerr,
    );
    if check_retval_int(retval, "ReadInputs") {
        std::process::exit(1);
    }

    /* Compute reference solution */
    let yref = N_VNew_Serial(4, &sunctx).unwrap();

    retval = RefSol(tf, &yref, nout, &sunctx);
    if check_retval_int(retval, "RefSol") {
        std::process::exit(1);
    }

    /* Create serial vector to store the initial condition */
    let yy0 = N_VNew_Serial(4, &sunctx);
    if check_retval_null(&yy0, "N_VNew_Serial") {
        std::process::exit(1);
    }
    let yy0 = yy0.unwrap();

    /* Set the initial condition values */
    {
        let mut yy0data = N_VGetArrayPointer(&yy0).expect("vector data");

        yy0data[0] = ONE; /* x  */
        yy0data[1] = ZERO; /* y  */
        yy0data[2] = ZERO; /* xd */
        yy0data[3] = ZERO; /* yd */
    }

    /* Create CVODE memory */
    let cvode_mem = CVodeCreate(CV_BDF, &sunctx);
    if check_retval_null(&cvode_mem, "CVodeCreate") {
        std::process::exit(1);
    }
    let cvode_mem = cvode_mem.unwrap();

    /* Initialize CVODE */
    retval = CVodeInit(&cvode_mem, f, ZERO, &yy0);
    if check_retval_int(retval, "CVodeInit") {
        std::process::exit(1);
    }

    /* Create dense SUNMatrix for use in linear solves */
    let A = SUNDenseMatrix(4, 4, &sunctx);
    if check_retval_null(&A, "SUNDenseMatrix") {
        std::process::exit(1);
    }
    let A = A.unwrap();

    /* Create dense SUNLinearSolver object */
    let LS = SUNLinSol_Dense(&yy0, &A, &sunctx);
    if check_retval_null(&LS, "SUNLinSol_Dense") {
        std::process::exit(1);
    }
    let LS = LS.unwrap();

    /* Attach the matrix and linear solver to CVODE */
    retval = CVodeSetLinearSolver(&cvode_mem, &LS, Some(&A));
    if check_retval_int(retval, "CVodeSetLinearSolver") {
        std::process::exit(1);
    }

    /* Set a user-supplied projection function */
    retval = CVodeSetProjFn(&cvode_mem, Some(proj));
    if check_retval_int(retval, "CVodeSetProjFn") {
        std::process::exit(1);
    }

    /* Set maximum number of steps between outputs */
    retval = CVodeSetMaxNumSteps(&cvode_mem, 50000);
    if check_retval_int(retval, "CVodeSetMaxNumSteps") {
        std::process::exit(1);
    }

    /* Compute the solution with various tolerances */
    for _i in 0..5 {
        /* Output tolerance and output header for this run */
        print!(
            "\n\nrtol = {}, atol = {}\n",
            fmt_ew(rtol, 8, 2),
            fmt_ew(atol, 8, 2)
        );
        print!("Project    x         y");
        print!("         x'        y'     |     g      |    ");
        print!("nst     rhs eval    setups (J eval)  |   cf   ef\n");

        /* Compute solution with projection */
        retval = GetSol(
            &cvode_mem, &yy0, rtol, atol, tf, nout, SUNTRUE, projerr, &yref, &sunctx,
        );
        if check_retval_int(retval, "GetSol") {
            std::process::exit(1);
        }

        /* Compute solution without projection */
        retval = GetSol(
            &cvode_mem, &yy0, rtol, atol, tf, nout, SUNFALSE, SUNFALSE, &yref, &sunctx,
        );
        if check_retval_int(retval, "GetSol") {
            std::process::exit(1);
        }

        /* Reduce tolerance for next run */
        rtol /= 10.0;
        atol /= 10.0;
    }

    /* Free memory */
    N_VDestroy_Serial(yref);
    N_VDestroy_Serial(yy0);
    SUNMatDestroy(A);
    SUNLinSolFree(Some(LS));
    let mut cvode_mem = Some(cvode_mem);
    CVodeFree(&mut cvode_mem);
    let mut sunctx = Some(sunctx);
    SUNContext_Free(&mut sunctx);
}

/* -----------------------------------------------------------------------------
 * Functions to integrate the Cartesian and reference systems
 * ---------------------------------------------------------------------------*/

/* Compute the Cartesian system solution */
#[allow(clippy::too_many_arguments)]
fn GetSol(
    cvode_mem: &CVodeMem,
    yy0: &N_Vector,
    rtol: sunrealtype,
    atol: sunrealtype,
    tf: sunrealtype,
    nout: i32,
    proj: sunbooleantype,
    projerr: sunbooleantype,
    yref: &N_Vector,
    sunctx: &SUNContext,
) -> i32 {
    let mut retval: i32; /* reusable return flag */

    /* Enable or disable projection */
    if proj {
        print!("  YES   ");
        retval = CVodeSetProjFrequency(cvode_mem, 1);
        if check_retval_int(retval, "CVodeSetProjFrequency") {
            return 1;
        }

        /* Enable or disable error projection */
        retval = CVodeSetProjErrEst(cvode_mem, projerr);
        if check_retval_int(retval, "CVodeSetProjErrEst") {
            return 1;
        }
    } else {
        retval = CVodeSetProjFrequency(cvode_mem, 0);
        if check_retval_int(retval, "CVodeSetProjFrequency") {
            return 1;
        }
        print!("  NO    ");
    }

    /* Create vector to store the solution */
    let yy = N_VNew_Serial(4, sunctx).unwrap();

    /* Copy initial condition into solution vector */
    N_VScale(ONE, yy0, &yy);

    /* Reinitialize CVODE for this run */
    retval = CVodeReInit(cvode_mem, ZERO, yy0);
    if check_retval_int(retval, "CVodeReInit") {
        N_VDestroy_Serial(yy);
        return retval;
    }

    /* Set integration tolerances for this run */
    retval = CVodeSStolerances(cvode_mem, rtol, atol);
    if check_retval_int(retval, "CVodeSStolerances") {
        N_VDestroy_Serial(yy);
        return retval;
    }

    /* Open output file */
    let outname = if proj {
        format!(
            "cvPendulum_dns_rtol_{}_atol_{}_proj.txt",
            fmt_ew(rtol, 3, 2),
            fmt_ew(atol, 3, 2)
        )
    } else {
        format!(
            "cvPendulum_dns_rtol_{}_atol_{}.txt",
            fmt_ew(rtol, 3, 2),
            fmt_ew(atol, 3, 2)
        )
    };
    let mut FID = File::create(&outname).expect("output file");

    /* Output initial condition */
    {
        let yydata = N_VGetArrayPointer(&yy).expect("vector data");
        let _ = write!(
            FID,
            "{} {} {} {} {}\n",
            fmt_ew(ZERO, 24, 16),
            fmt_ew(yydata[0], 24, 16),
            fmt_ew(yydata[1], 24, 16),
            fmt_ew(yydata[2], 24, 16),
            fmt_ew(yydata[3], 24, 16)
        );
    }

    /* Integrate to tf and peridoically output the solution */
    let dtout = tf / nout as sunrealtype; /* output frequency */
    let mut tout = dtout; /* output time      */
    let mut t: sunrealtype = ZERO; /* return time      */

    for out in 0..nout {
        /* Set stop time (do not interpolate output) */
        retval = CVodeSetStopTime(cvode_mem, tout);
        if check_retval_int(retval, "CVodeSetStopTime") {
            N_VDestroy_Serial(yy);
            drop(FID);
            return retval;
        }

        /* Integrate to tout */
        retval = CVode(cvode_mem, tout, &yy, &mut t, CV_NORMAL);
        if check_retval_int(retval, "CVode") {
            N_VDestroy_Serial(yy);
            drop(FID);
            return retval;
        }

        /* Write output */
        {
            let yydata = N_VGetArrayPointer(&yy).expect("vector data");
            let _ = write!(
                FID,
                "{} {} {} {} {}\n",
                fmt_ew(t, 24, 16),
                fmt_ew(yydata[0], 24, 16),
                fmt_ew(yydata[1], 24, 16),
                fmt_ew(yydata[2], 24, 16),
                fmt_ew(yydata[3], 24, 16)
            );
        }

        /* Update output time */
        if out < nout - 1 {
            tout += dtout;
        } else {
            tout = tf;
        }
    }

    /* Close output file */
    drop(FID);

    /* Compute the constraint violation */
    let (x, y) = {
        let yydata = N_VGetArrayPointer(&yy).expect("vector data");
        (yydata[0], yydata[1])
    };
    let g = (x * x + y * y - ONE).abs();

    /* Compute the absolute error compared to the reference solution */
    N_VLinearSum(ONE, &yy, -ONE, yref, &yy);
    N_VAbs(&yy, &yy);

    let (x, y, xd, yd) = {
        let yydata = N_VGetArrayPointer(&yy).expect("vector data");
        (yydata[0], yydata[1], yydata[2], yydata[3])
    };

    /* Output errors */
    print!(
        "{}  {}  {}  {}  |  {}  |",
        fmt_ew(x, 8, 2),
        fmt_ew(y, 8, 2),
        fmt_ew(xd, 8, 2),
        fmt_ew(yd, 8, 2),
        fmt_ew(g, 8, 2)
    );

    /* Free solution vector */
    N_VDestroy_Serial(yy);

    /* Integrator stats */
    let mut nst: i64 = 0;
    let mut nfe: i64 = 0;
    let mut nsetups: i64 = 0;
    let mut nje: i64 = 0;
    let mut nfeLS: i64 = 0;
    let mut ncfn: i64 = 0;
    let mut netf: i64 = 0;

    /* Get integrator stats */
    retval = CVodeGetNumSteps(cvode_mem, &mut nst);
    if check_retval_int(retval, "CVodeGetNumSteps") {
        return retval;
    }

    retval = CVodeGetNumRhsEvals(cvode_mem, &mut nfe);
    if check_retval_int(retval, "CVodeGetNumFctEvals") {
        return retval;
    }

    retval = CVodeGetNumLinSolvSetups(cvode_mem, &mut nsetups);
    if check_retval_int(retval, "CVodeGetNumLinSolvSetups") {
        return retval;
    }

    retval = CVodeGetNumErrTestFails(cvode_mem, &mut netf);
    if check_retval_int(retval, "CVodeGetNumErrTestFails") {
        return retval;
    }

    retval = CVodeGetNumNonlinSolvConvFails(cvode_mem, &mut ncfn);
    if check_retval_int(retval, "CVodeGetNumNonlinSolvConvFails") {
        return retval;
    }

    retval = CVodeGetNumJacEvals(cvode_mem, &mut nje);
    if check_retval_int(retval, "CVodeGetNumJacEvals") {
        return retval;
    }

    retval = CVodeGetNumLinRhsEvals(cvode_mem, &mut nfeLS);
    if check_retval_int(retval, "CVodeGetNumLinRhsEvals") {
        return retval;
    }

    /* Output stats */
    print!(
        " {:>6}   {:>6}+{:<4}     {:>4} ({:>3})     |  {:>3}  {:>3}\n",
        nst, nfe, nfeLS, nsetups, nje, ncfn, netf
    );

    0
}

/* Compute the reference system solution */
fn RefSol(tf: sunrealtype, yref: &N_Vector, nout: i32, sunctx: &SUNContext) -> i32 {
    let tol: sunrealtype = 1.0e-14; /* integration tolerance */

    /* Create the solution vector */
    let yy = N_VNew_Serial(2, sunctx);
    if check_retval_null(&yy, "N_VNew_Serial") {
        return -1;
    }
    let yy = yy.unwrap();

    /* Set the initial condition */
    {
        let mut yydata = N_VGetArrayPointer(&yy).expect("vector data");

        yydata[0] = ZERO; /* theta  */
        yydata[1] = ZERO; /* theta' */
    }

    /* Create CVODE memory */
    let cvode_mem = CVodeCreate(CV_BDF, sunctx);
    if check_retval_null(&cvode_mem, "CVodeCreate") {
        return 1;
    }
    let cvode_mem = cvode_mem.unwrap();

    /* Initialize CVODE */
    let mut retval = CVodeInit(&cvode_mem, fref, ZERO, &yy);
    if check_retval_int(retval, "CVodeInit") {
        return 1;
    }

    /* Set integration tolerances */
    retval = CVodeSStolerances(&cvode_mem, tol, tol);
    if check_retval_int(retval, "CVodeSStolerances") {
        return 1;
    }

    /* Create dense SUNMatrix for use in linear solves */
    let A = SUNDenseMatrix(2, 2, sunctx);
    if check_retval_null(&A, "SUNDenseMatrix") {
        return 1;
    }
    let A = A.unwrap();

    /* Create dense SUNLinearSolver object */
    let LS = SUNLinSol_Dense(&yy, &A, sunctx);
    if check_retval_null(&LS, "SUNLinSol_Dense") {
        return 1;
    }
    let LS = LS.unwrap();

    /* Attach the matrix and linear solver to CVODE */
    retval = CVodeSetLinearSolver(&cvode_mem, &LS, Some(&A));
    if check_retval_int(retval, "CVodeSetLinearSolver") {
        return 1;
    }

    /* Set CVODE optional inputs */
    retval = CVodeSetMaxNumSteps(&cvode_mem, 100000);
    if check_retval_int(retval, "CVodeSetMaxNumSteps") {
        return 1;
    }

    retval = CVodeSetStopTime(&cvode_mem, tf);
    if check_retval_int(retval, "CVodeSetStopTime") {
        return 1;
    }

    /* Open output file */
    let mut FID = File::create("cvPendulum_dns_ref.txt").expect("output file");

    /* Output initial condition */
    {
        let yydata = N_VGetArrayPointer(&yy).expect("vector data");
        let th = yydata[0];
        let thd = yydata[1];
        let _ = write!(
            FID,
            "{} {} {} {} {}\n",
            fmt_ew(ZERO, 24, 16),
            fmt_ew(th.sun_cos(), 24, 16),
            fmt_ew(th.sun_sin(), 24, 16),
            fmt_ew(-thd * th.sun_sin(), 24, 16),
            fmt_ew(thd * th.sun_cos(), 24, 16)
        );
    }

    /* Integrate to tf and periodically output the solution */
    let dtout = tf / nout as sunrealtype; /* output frequency */
    let mut tout = dtout; /* output time      */
    let mut t: sunrealtype = ZERO; /* return time      */

    for out in 0..nout {
        /* Set stop time (do not interpolate output) */
        retval = CVodeSetStopTime(&cvode_mem, tout);
        if check_retval_int(retval, "CVodeSetStopTime") {
            N_VDestroy_Serial(yy);
            SUNMatDestroy(A);
            SUNLinSolFree(Some(LS));
            CVodeFree(&mut Some(cvode_mem));
            drop(FID);
            return retval;
        }

        /* Integrate to tout */
        retval = CVode(&cvode_mem, tf, &yy, &mut t, CV_NORMAL);
        if check_retval_int(retval, "CVode") {
            N_VDestroy_Serial(yy);
            SUNMatDestroy(A);
            SUNLinSolFree(Some(LS));
            CVodeFree(&mut Some(cvode_mem));
            drop(FID);
            return retval;
        }

        /* Write output */
        {
            let yydata = N_VGetArrayPointer(&yy).expect("vector data");
            let th = yydata[0];
            let thd = yydata[1];
            let _ = write!(
                FID,
                "{} {} {} {} {}\n",
                fmt_ew(t, 24, 16),
                fmt_ew(th.sun_cos(), 24, 16),
                fmt_ew(th.sun_sin(), 24, 16),
                fmt_ew(-thd * th.sun_sin(), 24, 16),
                fmt_ew(thd * th.sun_cos(), 24, 16)
            );
        }

        /* Update output time */
        if out < nout - 1 {
            tout += dtout;
        } else {
            tout = tf;
        }
    }

    /* Close output file */
    drop(FID);

    /* Get solution components */
    let (th, thd) = {
        let yydata = N_VGetArrayPointer(&yy).expect("vector data");
        (yydata[0], yydata[1])
    };

    /* Convert to Cartesian reference solution */
    {
        let mut yydata = N_VGetArrayPointer(yref).expect("vector data");

        yydata[0] = th.sun_cos();
        yydata[1] = th.sun_sin();
        yydata[2] = -thd * th.sun_sin();
        yydata[3] = thd * th.sun_cos();
    }

    /* Free memory */
    N_VDestroy_Serial(yy);
    SUNMatDestroy(A);
    SUNLinSolFree(Some(LS));
    CVodeFree(&mut Some(cvode_mem));

    0
}

/* -----------------------------------------------------------------------------
 * Functions provided to CVODE
 * ---------------------------------------------------------------------------*/

/* ODE RHS function for the reference system */
fn fref(_t: sunrealtype, yy: &N_Vector, fy: &N_Vector, _f_data: &mut Option<Box<dyn Any>>) -> i32 {
    /* Get vector array pointers */
    let yydata = N_VGetArrayPointer(yy).expect("vector data");
    let mut fydata = N_VGetArrayPointer(fy).expect("vector data");

    fydata[0] = yydata[1]; /* theta'          */
    fydata[1] = -GRAV * (yydata[0]).sun_cos(); /* -g * cos(theta) */
    0
}

/* ODE RHS function for the Cartesian system */
fn f(_t: sunrealtype, yy: &N_Vector, fy: &N_Vector, _f_data: &mut Option<Box<dyn Any>>) -> i32 {
    /* Get vector array pointers */
    let yydata = N_VGetArrayPointer(yy).expect("vector data");
    let mut fydata = N_VGetArrayPointer(fy).expect("vector data");

    /* Get vector components */
    let x = yydata[0];
    let y = yydata[1];
    let xd = yydata[2];
    let yd = yydata[3];

    /* Compute tension */
    let tmp = xd * xd + yd * yd - GRAV * y;

    /* Compute RHS */
    fydata[0] = xd;
    fydata[1] = yd;
    fydata[2] = -x * tmp;
    fydata[3] = -y * tmp - GRAV;

    0
}

/* Projection function */
fn proj(
    _t: sunrealtype,
    yy: &N_Vector,
    corr: &N_Vector,
    _epsProj: sunrealtype,
    err: Option<&N_Vector>,
    _pdata: &mut Option<Box<dyn Any>>,
) -> i32 {
    /* Extract current solution */
    let (x, y, xd, yd) = {
        let yydata = N_VGetArrayPointer(yy).expect("vector data");
        (yydata[0], yydata[1], yydata[2], yydata[3])
    };

    /* Project positions */

    let R = (x * x + y * y).sqrt();

    let x_new = x / R;
    let y_new = y / R;

    /* Project velocities
     *
     *        +-            -+  +-    -+
     *        |  y*y    -x*y |  |  xd  |
     *  P v = |              |  |      |
     *        | -x*y     x*x |  |  yd  |
     *        +-            -+  +-    -+
     */

    let xd_new = xd * y_new * y_new - yd * x_new * y_new;
    let yd_new = -xd * x_new * y_new + yd * x_new * x_new;

    /* Return position and velocity corrections */
    {
        let mut cdata = N_VGetArrayPointer(corr).expect("vector data");
        cdata[0] = x_new - x;
        cdata[1] = y_new - y;
        cdata[2] = xd_new - xd;
        cdata[3] = yd_new - yd;
    }

    /* Project error P * err */
    if let Some(err) = err {
        let mut edata = N_VGetArrayPointer(err).expect("vector data");

        let e1 = edata[0];
        let e2 = edata[1];
        let e3 = edata[2];
        let e4 = edata[3];

        let e1_new = y_new * y_new * e1 - x_new * y_new * e2;
        let e2_new = -x_new * y_new * e1 + x_new * x_new * e2;

        let e3_new = y_new * y_new * e3 - x_new * y_new * e4;
        let e4_new = -x_new * y_new * e3 + x_new * x_new * e4;

        edata[0] = e1_new;
        edata[1] = e2_new;
        edata[2] = e3_new;
        edata[3] = e4_new;
    }

    0
}

/* -----------------------------------------------------------------------------
 * Private helper functions
 * ---------------------------------------------------------------------------*/

/* Read command line unputs */
fn ReadInputs(
    args: &[String],
    rtol: &mut sunrealtype,
    atol: &mut sunrealtype,
    tf: &mut sunrealtype,
    nout: &mut i32,
    projerr: &mut sunbooleantype,
) -> i32 {
    let mut arg_idx = 1usize;

    /* check for input args */
    while arg_idx < args.len() {
        if args[arg_idx] == "--tol" {
            arg_idx += 1;
            *rtol = SUNStrToReal(&args[arg_idx]);
            arg_idx += 1;
            *atol = SUNStrToReal(&args[arg_idx]);
            arg_idx += 1;
        } else if args[arg_idx] == "--tf" {
            arg_idx += 1;
            *tf = SUNStrToReal(&args[arg_idx]);
            arg_idx += 1;
        } else if args[arg_idx] == "--nout" {
            arg_idx += 1;
            *nout = atoi(&args[arg_idx]);
            arg_idx += 1;
        } else if args[arg_idx] == "--noerrproj" {
            arg_idx += 1;
            *projerr = SUNFALSE;
        } else if args[arg_idx] == "--help" {
            InputHelp();
            return -1;
        } else {
            eprint!("ERROR: Invalid input {}", args[arg_idx]);
            InputHelp();
            return -1;
        }
    }

    0
}

/* Print command line options */
fn InputHelp() {
    print!("\nCommand line options:\n");
    print!("  --tol <rtol> <atol> : relative and absolute tolerance\n");
    print!("  --tf <time>         : final simulation time\n");
    print!("  --nout <outputs>    : number of outputs\n");
    print!("  --noerrproj         : disable error projection\n");
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
        eprint!("\nERROR: {}() returned NULL pointer\n\n", funcname);
        return true;
    }
    false
}
