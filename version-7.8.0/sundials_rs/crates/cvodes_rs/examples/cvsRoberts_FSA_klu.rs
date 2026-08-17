/* -----------------------------------------------------------------
 * Rust port of examples/cvodes/serial/cvsRoberts_FSA_dns.c
 * -----------------------------------------------------------------
 * Example problem:
 *
 * The following is a simple example problem, with the coding
 * needed for its solution by CVODES for Forward Sensitivity
 * Analysis. The problem is from chemical kinetics, and consists
 * of the following three rate equations:
 *    dy1/dt = -p1*y1 + p2*y2*y3
 *    dy2/dt =  p1*y1 - p2*y2*y3 - p3*(y2)^2
 *    dy3/dt =  p3*(y2)^2
 * on the interval from t = 0.0 to t = 4.e10, with initial
 * conditions: y1 = 1.0, y2 = y3 = 0. The problem is stiff.
 * The reaction rates are: p1=0.04, p2=1e4, and p3=3e7.
 * This program solves the problem with the BDF method, Newton
 * iteration with the dense linear solver, and a
 * user-supplied Jacobian routine.
 * It uses a scalar relative tolerance and a vector absolute
 * tolerance. Output is printed in decades from t = .4 to t = 4.e10.
 * Run statistics (optional outputs) are printed at the end.
 *
 * Optionally, CVODES can compute sensitivities with respect to the
 * problem parameters p1, p2, and p3.
 * The sensitivity right hand side is given analytically through the
 * user routine fS (of type SensRhs1Fn).
 * Any of three sensitivity methods (SIMULTANEOUS, STAGGERED, and
 * STAGGERED1) can be used and sensitivities may be included in the
 * error test or not (error control set on SUNTRUE or SUNFALSE,
 * respectively).
 *
 * Execution:
 *
 * If no sensitivities are desired:
 *    % cvsRoberts_FSA_dns -nosensi
 * If sensitivities are to be computed:
 *    % cvsRoberts_FSA_dns -sensi sensi_meth err_con
 * where sensi_meth is one of {sim, stg, stg1} and err_con is one of
 * {t, f}.
 * -----------------------------------------------------------------*/

#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

use std::any::Any;

use cvodes_rs::prelude::*;

/* User-defined vector accessor helpers: Ith
   (C macro `Ith(v,i)` = `NV_Ith_S(v,i-1)`; i is 1-based). */

fn Ith(v: &N_Vector, i: usize) -> sunrealtype {
    N_VGetArrayPointer(v).expect("N_VGetArrayPointer")[i - 1]
}

fn Ith_set(v: &N_Vector, i: usize, x: sunrealtype) {
    N_VGetArrayPointer(v).expect("N_VGetArrayPointer")[i - 1] = x;
}

/* Problem Constants */

const NEQ: sunindextype = 3; /* number of equations  */
const NNZ: sunindextype = 7; /* number of non-zero entries in the Jacobian */
const Y1: sunrealtype = 1.0; /* initial y components */
const Y2: sunrealtype = 0.0;
const Y3: sunrealtype = 0.0;
const RTOL: sunrealtype = 1.0e-4; /* scalar relative tolerance            */
const ATOL1: sunrealtype = 1.0e-8; /* vector absolute tolerance components */
const ATOL2: sunrealtype = 1.0e-14;
const ATOL3: sunrealtype = 1.0e-6;
const T0: sunrealtype = 0.0; /* initial time           */
const T1: sunrealtype = 0.4; /* first output time      */
const TMULT: sunrealtype = 10.0; /* output time factor     */
const NOUT: i32 = 12; /* number of output times */

const NS: i32 = 3; /* number of sensitivities computed */

const ZERO: sunrealtype = 0.0;

/* Type : UserData */

struct UserData {
    p: [sunrealtype; 3], /* problem parameters */
}

/*
 *-------------------------------
 * Main Program
 *-------------------------------
 */

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let argc = argv.len();

    let mut pbar = [ZERO; NS as usize];
    let mut sensi: sunbooleantype = SUNFALSE;
    let mut err_con: sunbooleantype = SUNFALSE;
    let mut sensi_meth: i32 = -1;

    /* Process arguments */
    ProcessArgs(argc, &argv, &mut sensi, &mut sensi_meth, &mut err_con);

    /* User data structure */
    let data: Option<Box<UserData>> = Some(Box::new(UserData { p: [ZERO; 3] }));
    if check_retval_ptr(&data, "malloc", 2) != 0 {
        std::process::exit(1);
    }
    let mut data = data.expect("malloc");

    /* Initialize sensitivity variables (reaction rates for this problem) */
    data.p[0] = 0.04;
    data.p[1] = 1.0e4;
    data.p[2] = 3.0e7;

    /* Create the SUNDIALS context that all SUNDIALS objects require */
    let mut sunctx: Option<SUNContext> = None;
    let retval = SUNContext_Create(SUN_COMM_NULL, &mut sunctx);
    if check_retval_int(retval, "SUNContext_Create") != 0 {
        std::process::exit(1);
    }
    let ctx = sunctx.as_ref().expect("SUNContext").clone();

    /* Initial conditions */
    let y = N_VNew_Serial(NEQ, &ctx);
    if check_retval_ptr(&y, "N_VNew_Serial", 0) != 0 {
        std::process::exit(1);
    }
    let y = y.expect("N_VNew_Serial");

    /* Initialize y */
    Ith_set(&y, 1, Y1);
    Ith_set(&y, 2, Y2);
    Ith_set(&y, 3, Y3);

    /* Call CVodeCreate to create the solver memory and specify the
     * Backward Differentiation Formula */
    let cvode_mem = CVodeCreate(CV_BDF, &ctx);
    if check_retval_ptr(&cvode_mem, "CVodeCreate", 0) != 0 {
        std::process::exit(1);
    }
    let cv = cvode_mem.expect("CVodeCreate");

    /* Call CVodeInit to initialize the integrator memory and specify the
     * user's right hand side function in y'=f(t,y), the initial time T0, and
     * the initial dependent variable vector y. */
    let retval = CVodeInit(&cv, f, T0, &y);
    if check_retval_int(retval, "CVodeInit") != 0 {
        std::process::exit(1);
    }

    /* Call CVodeWFtolerances to specify a user-supplied function ewt that sets
     * the multiplicative error weights w_i for use in the weighted RMS norm */
    let retval = CVodeWFtolerances(&cv, ewt);
    if check_retval_int(retval, "CVodeWFtolerances") != 0 {
        std::process::exit(1);
    }

    /* C keeps `data` alive alongside the integrator and reads data->p again
       below to fill pbar; the Rust box moves into the mem record, so snapshot
       the parameter values here (identical values, same point in the flow). */
    let p_saved = data.p;

    /* Attach user data */
    let data: Box<dyn Any> = data;
    let retval = CVodeSetUserData(&cv, Some(data));
    if check_retval_int(retval, "CVodeSetUserData") != 0 {
        std::process::exit(1);
    }

    /* Create dense SUNMatrix for use in linear solves */
    let A = SUNSparseMatrix(NEQ, NEQ, NNZ, SUN_CSC_MAT, &ctx);
    if check_retval_ptr(&A, "SUNSparseMatrix", 0) != 0 {
        std::process::exit(1);
    }
    let A = A.expect("SUNSparseMatrix");

    /* Create dense SUNLinearSolver object */
    let LS = SUNLinSol_KLU(&y, &A, &ctx);
    if check_retval_ptr(&LS, "SUNLinSol_KLU", 0) != 0 {
        std::process::exit(1);
    }
    let LS = LS.expect("SUNLinSol_KLU");

    /* Attach the matrix and linear solver */
    let retval = CVodeSetLinearSolver(&cv, &LS, Some(&A));
    if check_retval_int(retval, "CVodeSetLinearSolver") != 0 {
        std::process::exit(1);
    }

    /* Set the user-supplied Jacobian routine Jac */
    let retval = CVodeSetJacFn(&cv, Some(Jac));
    if check_retval_int(retval, "CVodeSetJacFn") != 0 {
        std::process::exit(1);
    }

    print!(" \n3-species kinetics problem\n");

    /* Sensitivity-related settings */
    let mut yS: Option<Vec<N_Vector>> = None;
    if sensi {
        /* Set parameter scaling factor */
        pbar[0] = p_saved[0];
        pbar[1] = p_saved[1];
        pbar[2] = p_saved[2];

        /* Set sensitivity initial conditions */
        yS = N_VCloneVectorArray(NS, &y);
        if check_retval_ptr(&yS, "N_VCloneVectorArray", 0) != 0 {
            std::process::exit(1);
        }
        {
            let yS = yS.as_ref().expect("N_VCloneVectorArray");
            for is in 0..NS as usize {
                N_VConst(ZERO, &yS[is]);
            }
        }

        /* Call CVodeSensInit1 to activate forward sensitivity computations
         * and allocate internal memory for COVEDS related to sensitivity
         * calculations. Computes the right-hand sides of the sensitivity
         * ODE, one at a time */
        let retval = CVodeSensInit1(
            &cv,
            NS,
            sensi_meth,
            Some(fS),
            yS.as_ref().expect("N_VCloneVectorArray"),
        );
        if check_retval_int(retval, "CVodeSensInit") != 0 {
            std::process::exit(1);
        }

        /* Call CVodeSensEEtolerances to estimate tolerances for sensitivity
         * variables based on the rolerances supplied for states variables and
         * the scaling factor pbar */
        let retval = CVodeSensEEtolerances(&cv);
        if check_retval_int(retval, "CVodeSensEEtolerances") != 0 {
            std::process::exit(1);
        }

        /* Set sensitivity analysis optional inputs */
        /* Call CVodeSetSensErrCon to specify the error control strategy for
         * sensitivity variables */
        let retval = CVodeSetSensErrCon(&cv, err_con);
        if check_retval_int(retval, "CVodeSetSensErrCon") != 0 {
            std::process::exit(1);
        }

        /* Call CVodeSetSensParams to specify problem parameter information for
         * sensitivity calculations */
        let retval = CVodeSetSensParams(&cv, None, Some(&pbar[..]), None);
        if check_retval_int(retval, "CVodeSetSensParams") != 0 {
            std::process::exit(1);
        }

        print!("Sensitivity: YES ");
        if sensi_meth == CV_SIMULTANEOUS {
            print!("( SIMULTANEOUS +");
        } else if sensi_meth == CV_STAGGERED {
            print!("( STAGGERED +");
        } else {
            print!("( STAGGERED1 +");
        }
        if err_con {
            print!(" FULL ERROR CONTROL )");
        } else {
            print!(" PARTIAL ERROR CONTROL )");
        }
    } else {
        print!("Sensitivity: NO ");
    }

    /* In loop, call CVode, print results, and test for error.
    Break out of loop when NOUT preset output times have been reached.  */

    print!("\n\n");
    print!("===========================================");
    print!("============================\n");
    print!("     T     Q       H      NST           y1");
    print!("           y2           y3    \n");
    print!("===========================================");
    print!("============================\n");

    let mut t: sunrealtype = 0.0;
    let mut iout = 1;
    let mut tout = T1;
    while iout <= NOUT {
        let retval = CVode(&cv, tout, &y, &mut t, CV_NORMAL);
        if check_retval_int(retval, "CVode") != 0 {
            break;
        }

        PrintOutput(&cv, t, &y);

        /* Call CVodeGetSens to get the sensitivity solution vector after a
         * successful return from CVode */
        if sensi {
            let yS = yS.as_ref().expect("N_VCloneVectorArray");
            let retval = CVodeGetSens(&cv, &mut t, yS);
            if check_retval_int(retval, "CVodeGetSens") != 0 {
                break;
            }
            PrintOutputS(yS);
        }
        print!("-----------------------------------------");
        print!("------------------------------\n");

        iout += 1;
        tout *= TMULT;
    }

    /* Print some final statistics */
    PrintFinalStats(&cv, sensi);

    /* Free memory */
    N_VDestroy(y); /* Free y vector */
    if sensi {
        /* Free yS vector */
        N_VDestroyVectorArray(yS.take().expect("N_VCloneVectorArray"), NS);
    }
    /* C `free(data)`: the user-data box is owned by the integrator memory
       and is released with it below. */
    let mut cvode_mem = Some(cv);
    CVodeFree(&mut cvode_mem); /* Free CVODES memory */
    let _ = SUNLinSolFree(Some(LS)); /* Free the linear solver memory */
    SUNMatDestroy(A); /* Free the matrix memory */
    let _ = SUNContext_Free(&mut sunctx); /* Free the SUNDIALS context */
}

/*
 *-------------------------------
 * Functions called by the solver
 *-------------------------------
 */

/*
 * f routine. Compute function f(t,y).
 */

fn f(_t: sunrealtype, y: &N_Vector, ydot: &N_Vector, user_data: &mut Option<Box<dyn Any>>) -> i32 {
    let y1 = Ith(y, 1);
    let y2 = Ith(y, 2);
    let y3 = Ith(y, 3);
    let data = user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<UserData>())
        .expect("UserData");
    let p1 = data.p[0];
    let p2 = data.p[1];
    let p3 = data.p[2];

    let yd1 = -p1 * y1 + p2 * y2 * y3;
    Ith_set(ydot, 1, yd1);
    let yd3 = p3 * y2 * y2;
    Ith_set(ydot, 3, yd3);
    Ith_set(ydot, 2, -yd1 - yd3);

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
    user_data: &mut Option<Box<dyn Any>>,
    _tmp1: &N_Vector,
    _tmp2: &N_Vector,
    _tmp3: &N_Vector,
) -> i32 {
    /* State at which to evaluate the Jacobian */
    let yval = N_VGetArrayPointer(y).expect("N_VGetArrayPointer");
    let ud = user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<UserData>())
        .expect("UserData");
    let p1 = ud.p[0];
    let p2 = ud.p[1];
    let p3 = ud.p[2];

    /* J is stored in CSC format. The C takes three pointers into the same
    matrix; here they are three fields behind one borrow, because taking
    them separately would be a second mutable borrow of the same content. */
    let mut m = SUNSparseMatrix_Content(J);
    let m = &mut *m;
    let (data, rowvals, colptrs) = (&mut m.data, &mut m.indexvals, &mut m.indexptrs);

    /* first column entries start at data[0], two entries (rows 0 and 1) */
    colptrs[0] = 0;

    rowvals[0] = 0;
    data[0] = -p1;

    rowvals[1] = 1;
    data[1] = p1;

    /* second column entries start at data[2], three entries (rows 0, 1, and 2) */
    colptrs[1] = 2;

    rowvals[2] = 0;
    data[2] = p2 * yval[2];

    rowvals[3] = 1;
    data[3] = (-p2 * yval[2]) - (2.0 * p3 * yval[1]);

    rowvals[4] = 2;
    data[4] = 2.0 * p3 * yval[1];

    /* third column entries start at data[5], two entries (rows 0 and 1) */
    colptrs[2] = 5;

    rowvals[5] = 0;
    data[5] = p2 * yval[1];

    rowvals[6] = 1;
    data[6] = -p2 * yval[1];

    /* number of non-zeros */
    colptrs[3] = 7;

    0
}

/*
 * fS routine. Compute sensitivity r.h.s.
 */

fn fS(
    _Ns: i32,
    _t: sunrealtype,
    y: &N_Vector,
    _ydot: &N_Vector,
    iS: i32,
    yS: &N_Vector,
    ySdot: &N_Vector,
    user_data: &mut Option<Box<dyn Any>>,
    _tmp1: &N_Vector,
    _tmp2: &N_Vector,
) -> i32 {
    let data = user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<UserData>())
        .expect("UserData");
    let p1 = data.p[0];
    let p2 = data.p[1];
    let p3 = data.p[2];

    let y1 = Ith(y, 1);
    let y2 = Ith(y, 2);
    let y3 = Ith(y, 3);
    let s1 = Ith(yS, 1);
    let s2 = Ith(yS, 2);
    let s3 = Ith(yS, 3);

    let mut sd1 = -p1 * s1 + p2 * y3 * s2 + p2 * y2 * s3;
    let mut sd3 = 2.0 * p3 * y2 * s2;
    let mut sd2 = -sd1 - sd3;

    match iS {
        0 => {
            sd1 += -y1;
            sd2 += y1;
        }
        1 => {
            sd1 += y2 * y3;
            sd2 += -y2 * y3;
        }
        2 => {
            sd2 += -y2 * y2;
            sd3 += y2 * y2;
        }
        _ => {}
    }

    Ith_set(ySdot, 1, sd1);
    Ith_set(ySdot, 2, sd2);
    Ith_set(ySdot, 3, sd3);

    0
}

/*
 * EwtSet function. Computes the error weights at the current solution.
 */

fn ewt(y: &N_Vector, w: &N_Vector, _user_data: &mut Option<Box<dyn Any>>) -> i32 {
    let rtol = RTOL;
    let atol = [ATOL1, ATOL2, ATOL3];

    for i in 1..=3usize {
        let yy = Ith(y, i);
        let ww = rtol * yy.abs() + atol[i - 1];
        if ww <= 0.0 {
            return -1;
        }
        Ith_set(w, i, 1.0 / ww);
    }

    0
}

/*
 *-------------------------------
 * Private helper functions
 *-------------------------------
 */

/*
 * Process and verify arguments to cvsfwddenx.
 */

fn ProcessArgs(
    argc: usize,
    argv: &[String],
    sensi: &mut sunbooleantype,
    sensi_meth: &mut i32,
    err_con: &mut sunbooleantype,
) {
    *sensi = SUNFALSE;
    *sensi_meth = -1;
    *err_con = SUNFALSE;

    if argc < 2 {
        WrongArgs(&argv[0]);
    }

    if argv[1] == "-nosensi" {
        *sensi = SUNFALSE;
    } else if argv[1] == "-sensi" {
        *sensi = SUNTRUE;
    } else {
        WrongArgs(&argv[0]);
    }

    if *sensi {
        if argc != 4 {
            WrongArgs(&argv[0]);
        }

        if argv[2] == "sim" {
            *sensi_meth = CV_SIMULTANEOUS;
        } else if argv[2] == "stg" {
            *sensi_meth = CV_STAGGERED;
        } else if argv[2] == "stg1" {
            *sensi_meth = CV_STAGGERED1;
        } else {
            WrongArgs(&argv[0]);
        }

        if argv[3] == "t" {
            *err_con = SUNTRUE;
        } else if argv[3] == "f" {
            *err_con = SUNFALSE;
        } else {
            WrongArgs(&argv[0]);
        }
    }
}

fn WrongArgs(name: &str) -> ! {
    print!("\nUsage: {} [-nosensi] [-sensi sensi_meth err_con]\n", name);
    print!("         sensi_meth = sim, stg, or stg1\n");
    print!("         err_con    = t or f\n");

    std::process::exit(0);
}

/*
 * Print current t, step count, order, stepsize, and solution.
 */

fn PrintOutput(cvode_mem: &CVodeMem, t: sunrealtype, u: &N_Vector) {
    let mut nst: i64 = 0;
    let mut qu: i32 = 0;
    let mut hu: sunrealtype = 0.0;

    let udata: [sunrealtype; 3] = {
        let d = N_VGetArrayPointer(u).expect("N_VGetArrayPointer");
        [d[0], d[1], d[2]]
    };

    let retval = CVodeGetNumSteps(cvode_mem, &mut nst);
    check_retval_int(retval, "CVodeGetNumSteps");
    let retval = CVodeGetLastOrder(cvode_mem, &mut qu);
    check_retval_int(retval, "CVodeGetLastOrder");
    let retval = CVodeGetLastStep(cvode_mem, &mut hu);
    check_retval_int(retval, "CVodeGetLastStep");

    /* C: printf("%8.3e %2d  %8.3e %5ld\n", t, qu, hu, nst) */
    print!(
        "{} {:>2}  {} {:>5}\n",
        fmt_ew(t, 8, 3),
        qu,
        fmt_ew(hu, 8, 3),
        nst
    );

    print!("                  Solution       ");

    /* C: printf("%12.4e %12.4e %12.4e \n", udata[0], udata[1], udata[2]) */
    print!(
        "{} {} {} \n",
        fmt_ew(udata[0], 12, 4),
        fmt_ew(udata[1], 12, 4),
        fmt_ew(udata[2], 12, 4)
    );
}

/*
 * Print sensitivities.
 */

fn PrintOutputS(uS: &[N_Vector]) {
    let sdata: [sunrealtype; 3] = {
        let d = N_VGetArrayPointer(&uS[0]).expect("N_VGetArrayPointer");
        [d[0], d[1], d[2]]
    };
    print!("                  Sensitivity 1  ");

    print!(
        "{} {} {} \n",
        fmt_ew(sdata[0], 12, 4),
        fmt_ew(sdata[1], 12, 4),
        fmt_ew(sdata[2], 12, 4)
    );

    let sdata: [sunrealtype; 3] = {
        let d = N_VGetArrayPointer(&uS[1]).expect("N_VGetArrayPointer");
        [d[0], d[1], d[2]]
    };
    print!("                  Sensitivity 2  ");

    print!(
        "{} {} {} \n",
        fmt_ew(sdata[0], 12, 4),
        fmt_ew(sdata[1], 12, 4),
        fmt_ew(sdata[2], 12, 4)
    );

    let sdata: [sunrealtype; 3] = {
        let d = N_VGetArrayPointer(&uS[2]).expect("N_VGetArrayPointer");
        [d[0], d[1], d[2]]
    };
    print!("                  Sensitivity 3  ");

    print!(
        "{} {} {} \n",
        fmt_ew(sdata[0], 12, 4),
        fmt_ew(sdata[1], 12, 4),
        fmt_ew(sdata[2], 12, 4)
    );
}

/*
 * Print some final statistics from the CVODES memory.
 */

fn PrintFinalStats(cvode_mem: &CVodeMem, sensi: sunbooleantype) {
    let mut nst: i64 = 0;
    let mut nfe: i64 = 0;
    let mut nsetups: i64 = 0;
    let mut nje: i64 = 0;
    let mut nni: i64 = 0;
    let mut nnf: i64 = 0;
    let mut ncfn: i64 = 0;
    let mut netf: i64 = 0;
    let mut nfSe: i64 = 0;
    let mut nfeS: i64 = 0;
    let mut nsetupsS: i64 = 0;
    let mut nniS: i64 = 0;
    let mut nnfS: i64 = 0;
    let mut ncfnS: i64 = 0;
    let mut netfS: i64 = 0;

    let retval = CVodeGetNumSteps(cvode_mem, &mut nst);
    check_retval_int(retval, "CVodeGetNumSteps");
    let retval = CVodeGetNumRhsEvals(cvode_mem, &mut nfe);
    check_retval_int(retval, "CVodeGetNumRhsEvals");
    let retval = CVodeGetNumLinSolvSetups(cvode_mem, &mut nsetups);
    check_retval_int(retval, "CVodeGetNumLinSolvSetups");
    let retval = CVodeGetNumErrTestFails(cvode_mem, &mut netf);
    check_retval_int(retval, "CVodeGetNumErrTestFails");
    let retval = CVodeGetNumNonlinSolvIters(cvode_mem, &mut nni);
    check_retval_int(retval, "CVodeGetNumNonlinSolvIters");
    let retval = CVodeGetNumNonlinSolvConvFails(cvode_mem, &mut nnf);
    check_retval_int(retval, "CVodeGetNumNonlinSolvConvFails");
    let retval = CVodeGetNumStepSolveFails(cvode_mem, &mut ncfn);
    check_retval_int(retval, "CVodeGetNumStepSolveFails");

    if sensi {
        let retval = CVodeGetSensNumRhsEvals(cvode_mem, &mut nfSe);
        check_retval_int(retval, "CVodeGetSensNumRhsEvals");
        let retval = CVodeGetNumRhsEvalsSens(cvode_mem, &mut nfeS);
        check_retval_int(retval, "CVodeGetNumRhsEvalsSens");
        let retval = CVodeGetSensNumLinSolvSetups(cvode_mem, &mut nsetupsS);
        check_retval_int(retval, "CVodeGetSensNumLinSolvSetups");
        let retval = CVodeGetSensNumErrTestFails(cvode_mem, &mut netfS);
        check_retval_int(retval, "CVodeGetSensNumErrTestFails");
        let retval = CVodeGetSensNumNonlinSolvIters(cvode_mem, &mut nniS);
        check_retval_int(retval, "CVodeGetSensNumNonlinSolvIters");
        let retval = CVodeGetSensNumNonlinSolvConvFails(cvode_mem, &mut nnfS);
        check_retval_int(retval, "CVodeGetSensNumNonlinSolvConvFails");
        let retval = CVodeGetNumStepSensSolveFails(cvode_mem, &mut ncfnS);
        check_retval_int(retval, "CVodeGetNumStepSensSolveFails");
    }

    let retval = CVodeGetNumJacEvals(cvode_mem, &mut nje);
    check_retval_int(retval, "CVodeGetNumJacEvals");

    print!("\nFinal Statistics:\n");
    /* C: printf("nst = %-6ld nfe = %-6ld nsetups = %-6ld nje = %ld\n", ...) */
    print!(
        "nst = {:<6} nfe = {:<6} nsetups = {:<6} nje = {}\n",
        nst, nfe, nsetups, nje
    );
    /* C: printf("nni = %-6ld nnf = %-6ld netf = %-6ld    ncfn = %-6ld\n\n", ...) */
    print!(
        "nni = {:<6} nnf = {:<6} netf = {:<6}    ncfn = {:<6}\n\n",
        nni, nnf, netf, ncfn
    );

    if sensi {
        /* C: printf("nfSe = %-6ld nfeS = %-6ld nsetupsS = %-6ld\n", ...) */
        print!(
            "nfSe = {:<6} nfeS = {:<6} nsetupsS = {:<6}\n",
            nfSe, nfeS, nsetupsS
        );
        /* C: printf("nniS = %-6ld nnfS = %-6ld netfS = %-6ld ncfnS = %-6ld\n\n", ...) */
        print!(
            "nniS = {:<6} nnfS = {:<6} netfS = {:<6} ncfnS = {:<6}\n\n",
            nniS, nnfS, netfS, ncfnS
        );
    }
}

/*
 * Check function return value...
 *   opt == 0 means SUNDIALS function allocates memory so check if
 *            returned NULL pointer
 *   opt == 1 means SUNDIALS function returns an integer value so check if
 *            retval < 0 (see check_retval_int)
 *   opt == 2 means function allocates memory so check if returned
 *            NULL pointer
 */

fn check_retval_ptr<T>(returnvalue: &Option<T>, funcname: &str, opt: i32) -> i32 {
    /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
    if opt == 0 && returnvalue.is_none() {
        eprint!(
            "\nSUNDIALS_ERROR: {}() failed - returned NULL pointer\n\n",
            funcname
        );
        return 1;
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
