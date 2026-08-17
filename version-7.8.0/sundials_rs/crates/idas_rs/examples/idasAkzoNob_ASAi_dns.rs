//! Port of `examples/idas/serial/idasAkzoNob_ASAi_dns.c`.
//!
//! Adjoint sensitivity example problem
//!
//! This IVP is a stiff system of 6 non-linear DAEs of index 1. The
//! problem originates from Akzo Nobel Central research in Arnhern,
//! The Netherlands, and describes a chemical process in which 2
//! species are mixed, while carbon dioxide is continuously added.
//! See <http://pitagora.dm.uniba.it/~testset/report/chemakzo.pdf>
//!
//! IDAS also computes the sensitivities of the integral
//!   G = int_t0^tf y1 dt
//! with respect to the initial values of the first components of y
//! (the differential components). These sensitivities are the first
//! five components of the solution of the adjoint system, at t = 0.

#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

use std::any::Any;

use idas_rs::prelude::*;

/* Accessor macros
C: `Ith(v, i)` = `NV_Ith_S(v, i - 1)` (i is 1-based). */

fn Ith(v: &N_Vector, i: usize) -> sunrealtype {
    N_VGetArrayPointer(v).expect("N_VGetArrayPointer")[i - 1]
}

fn Ith_set(v: &N_Vector, i: usize, x: sunrealtype) {
    N_VGetArrayPointer(v).expect("N_VGetArrayPointer")[i - 1] = x;
}

/* Problem Constants */
const NEQ: sunindextype = 6;
const T0: sunrealtype = 0.0;

const TF: sunrealtype = 180.0; /* Final time. */

const RTOL: sunrealtype = 1.0e-08;
const ATOL: sunrealtype = 1.0e-10;
const RTOLB: sunrealtype = 1.0e-06;
const ATOLB: sunrealtype = 1.0e-08;
const RTOLQ: sunrealtype = 1.0e-10;
const ATOLQ: sunrealtype = 1.0e-12;

const ZERO: sunrealtype = 0.0;
const HALF: sunrealtype = 0.5;
const ONE: sunrealtype = 1.0;
const TWO: sunrealtype = 2.0;

const STEPS: i64 = 150;

#[derive(Clone, Copy)]
struct UserDataRec {
    k1: sunrealtype,
    k2: sunrealtype,
    k3: sunrealtype,
    k4: sunrealtype,
    K: sunrealtype,
    klA: sunrealtype,
    Ks: sunrealtype,
    pCO2: sunrealtype,
    H: sunrealtype,
}

/* Main program */
fn main() {
    let mut retval: i32;
    let mut ncheck: i32 = 0;
    let mut time: sunrealtype = 0.0;
    let mut nst: i64 = 0;
    let mut nstB: i64 = 0;
    let mut indexB: i32 = 0;

    /* Consistent IC for  y, y'. */
    let y01: sunrealtype = 0.444;
    let y02: sunrealtype = 0.00123;
    let y03: sunrealtype = 0.0;
    let y04: sunrealtype = 0.007;
    let y05: sunrealtype = 0.0;

    print!("\nAdjoint Sensitivity Example for Akzo-Nobel Chemical Kinetics\n");
    print!("-------------------------------------------------------------\n");
    print!("Sensitivity of G = int_t0^tf (y1) dt with respect to IC.\n");
    print!("-------------------------------------------------------------\n\n");

    /* Create the SUNDIALS context object for this simulation */
    let mut sunctx: Option<SUNContext> = None;
    retval = SUNContext_Create(SUN_COMM_NULL, &mut sunctx);
    if check_retval(Some(retval), "SUNContext_Create", 1) != 0 {
        std::process::exit(1);
    }
    let ctx = sunctx.as_ref().expect("sunctx").clone();

    /* Allocate user data. */
    /* Fill user's data with the appropriate values for coefficients. */
    let data = UserDataRec {
        k1: 18.7,
        k2: 0.58,
        k3: 0.09,
        k4: 0.42,
        K: 34.4,
        klA: 3.3,
        Ks: 115.83,
        pCO2: 0.9,
        H: 737.0,
    };
    let mut user_data: Option<Box<dyn Any>> = Some(Box::new(data));

    /* Allocate N-vectors. */
    let yy = N_VNew_Serial(NEQ, &ctx);
    if check_retval(yy.as_ref().map(|_| 0), "N_VNew_Serial", 0) != 0 {
        std::process::exit(1);
    }
    let yy = yy.expect("yy");
    let yp = N_VClone(&yy);
    if check_retval(yp.as_ref().map(|_| 0), "N_VNew_Serial", 0) != 0 {
        std::process::exit(1);
    }
    let yp = yp.expect("yp");

    /* Set IC */
    Ith_set(&yy, 1, y01);
    Ith_set(&yy, 2, y02);
    Ith_set(&yy, 3, y03);
    Ith_set(&yy, 4, y04);
    Ith_set(&yy, 5, y05);
    Ith_set(&yy, 6, data.Ks * y01 * y04);

    /* Get y' = - res(t0, y, 0) */
    N_VConst(ZERO, &yp);

    let rr = N_VClone(&yy).expect("N_VClone");
    res(T0, &yy, &yp, &rr, &mut user_data);
    N_VScale(-ONE, &rr, &yp);
    N_VDestroy(rr);

    /* Create and initialize q0 for quadratures. */
    let q = N_VNew_Serial(1, &ctx);
    if check_retval(q.as_ref().map(|_| 0), "N_VNew_Serial", 0) != 0 {
        std::process::exit(1);
    }
    let q = q.expect("q");
    Ith_set(&q, 1, ZERO);

    /* Call IDACreate and IDAInit to initialize IDA memory */
    let mut mem_opt = IDACreate(&ctx);
    if check_retval(mem_opt.as_ref().map(|_| 0), "IDACreate", 0) != 0 {
        std::process::exit(1);
    }
    let mem = mem_opt.as_ref().expect("mem").clone();

    retval = IDAInit(&mem, res, T0, &yy, &yp);
    if check_retval(Some(retval), "IDAInit", 1) != 0 {
        std::process::exit(1);
    }

    /* Set tolerances. */
    retval = IDASStolerances(&mem, RTOL, ATOL);
    if check_retval(Some(retval), "IDASStolerances", 1) != 0 {
        std::process::exit(1);
    }

    /* Attach user data. */
    retval = IDASetUserData(&mem, user_data);
    if check_retval(Some(retval), "IDASetUserData", 1) != 0 {
        std::process::exit(1);
    }

    /* Create dense SUNMatrix for use in linear solves */
    let A = SUNDenseMatrix(NEQ, NEQ, &ctx);
    if check_retval(A.as_ref().map(|_| 0), "SUNDenseMatrix", 0) != 0 {
        std::process::exit(1);
    }
    let A = A.expect("A");

    /* Create dense SUNLinearSolver object */
    let LS = SUNLinSol_Dense(&yy, &A, &ctx);
    if check_retval(LS.as_ref().map(|_| 0), "SUNLinSol_Dense", 0) != 0 {
        std::process::exit(1);
    }
    let LS = LS.expect("LS");

    /* Attach the matrix and linear solver */
    retval = IDASetLinearSolver(&mem, &LS, Some(&A));
    if check_retval(Some(retval), "IDASetLinearSolver", 1) != 0 {
        std::process::exit(1);
    }

    /* Initialize QUADRATURE(S). */
    retval = IDAQuadInit(&mem, rhsQ, &q);
    if check_retval(Some(retval), "IDAQuadInit", 1) != 0 {
        std::process::exit(1);
    }

    /* Set tolerances and error control for quadratures. */
    retval = IDAQuadSStolerances(&mem, RTOLQ, ATOLQ);
    if check_retval(Some(retval), "IDAQuadSStolerances", 1) != 0 {
        std::process::exit(1);
    }

    retval = IDASetQuadErrCon(&mem, SUNTRUE);
    if check_retval(Some(retval), "IDASetQuadErrCon", 1) != 0 {
        std::process::exit(1);
    }

    /* Prepare ADJOINT. */
    retval = IDAAdjInit(&mem, STEPS, IDA_HERMITE);
    if check_retval(Some(retval), "IDAAdjInit", 1) != 0 {
        std::process::exit(1);
    }

    /* FORWARD run. */
    print!("Forward integration ... ");
    retval = IDASolveF(&mem, TF, &mut time, &yy, &yp, IDA_NORMAL, &mut ncheck);
    if check_retval(Some(retval), "IDASolveF", 1) != 0 {
        std::process::exit(1);
    }

    retval = IDAGetNumSteps(&mem, &mut nst);
    if check_retval(Some(retval), "IDAGetNumSteps", 1) != 0 {
        std::process::exit(1);
    }

    print!("done ( nst = {} )\n", nst);

    retval = IDAGetQuad(&mem, &mut time, &q);
    if check_retval(Some(retval), "IDAGetQuad", 1) != 0 {
        std::process::exit(1);
    }

    print!("G:          {} \n", fmt_fw(Ith(&q, 1), 24, 16));
    print!("--------------------------------------------------------\n\n");

    /* BACKWARD run */

    /* Initialize yB */
    let yB = N_VClone(&yy);
    if check_retval(yB.as_ref().map(|_| 0), "N_VNew_Serial", 0) != 0 {
        std::process::exit(1);
    }
    let yB = yB.expect("yB");
    N_VConst(ZERO, &yB);

    let ypB = N_VClone(&yB);
    if check_retval(ypB.as_ref().map(|_| 0), "N_VNew_Serial", 0) != 0 {
        std::process::exit(1);
    }
    let ypB = ypB.expect("ypB");
    N_VConst(ZERO, &ypB);
    Ith_set(&ypB, 1, -ONE);

    retval = IDACreateB(&mem, &mut indexB);
    if check_retval(Some(retval), "IDACreateB", 1) != 0 {
        std::process::exit(1);
    }

    retval = IDAInitB(&mem, indexB, resB, TF, &yB, &ypB);
    if check_retval(Some(retval), "IDAInitB", 1) != 0 {
        std::process::exit(1);
    }

    retval = IDASStolerancesB(&mem, indexB, RTOLB, ATOLB);
    if check_retval(Some(retval), "IDASStolerancesB", 1) != 0 {
        std::process::exit(1);
    }

    retval = IDASetUserDataB(&mem, indexB, Some(Box::new(data)));
    if check_retval(Some(retval), "IDASetUserDataB", 1) != 0 {
        std::process::exit(1);
    }

    retval = IDASetMaxNumStepsB(&mem, indexB, 1000);
    let _ = retval;

    /* Create dense SUNMatrix for use in linear solves */
    let AB = SUNDenseMatrix(NEQ, NEQ, &ctx);
    if check_retval(AB.as_ref().map(|_| 0), "SUNDenseMatrix", 0) != 0 {
        std::process::exit(1);
    }
    let AB = AB.expect("AB");

    /* Create dense SUNLinearSolver object */
    let LSB = SUNLinSol_Dense(&yB, &AB, &ctx);
    if check_retval(LSB.as_ref().map(|_| 0), "SUNLinSol_Dense", 0) != 0 {
        std::process::exit(1);
    }
    let LSB = LSB.expect("LSB");

    /* Attach the matrix and linear solver */
    retval = IDASetLinearSolverB(&mem, indexB, &LSB, Some(&AB));
    if check_retval(Some(retval), "IDASetLinearSolverB", 1) != 0 {
        std::process::exit(1);
    }

    print!("Backward integration ... ");

    retval = IDASolveB(&mem, T0, IDA_NORMAL);
    if check_retval(Some(retval), "IDASolveB", 1) != 0 {
        std::process::exit(1);
    }

    let bmem = IDAGetAdjIDABmem(&mem, indexB).expect("IDAGetAdjIDABmem");
    let _ = IDAGetNumSteps(&bmem, &mut nstB);
    print!("done ( nst = {} )\n", nstB);

    retval = IDAGetB(&mem, indexB, &mut time, &yB, &ypB);
    if check_retval(Some(retval), "IDAGetB", 1) != 0 {
        std::process::exit(1);
    }

    PrintOutput(time, &yB, &ypB);

    IDAFree(&mut mem_opt);
    let _ = SUNLinSolFree(Some(LS));
    SUNMatDestroy(A);
    let _ = SUNLinSolFree(Some(LSB));
    SUNMatDestroy(AB);
    N_VDestroy(yy);
    N_VDestroy(yp);
    N_VDestroy(yB);
    N_VDestroy(ypB);
    N_VDestroy(q);
    let _ = SUNContext_Free(&mut sunctx);

    std::process::exit(0);
}

fn res(
    _t: sunrealtype,
    yy: &N_Vector,
    yd: &N_Vector,
    resval: &N_Vector,
    userdata: &mut Option<Box<dyn Any>>,
) -> i32 {
    let data = *userdata
        .as_mut()
        .and_then(|b| b.downcast_mut::<UserDataRec>())
        .expect("UserDataRec");
    let k1 = data.k1;
    let k2 = data.k2;
    let k3 = data.k3;
    let k4 = data.k4;
    let K = data.K;
    let klA = data.klA;
    let Ks = data.Ks;
    let pCO2 = data.pCO2;
    let H = data.H;

    let y1 = Ith(yy, 1);
    let y2 = Ith(yy, 2);
    let y3 = Ith(yy, 3);
    let y4 = Ith(yy, 4);
    let y5 = Ith(yy, 5);
    let y6 = Ith(yy, 6);

    let yd1 = Ith(yd, 1);
    let yd2 = Ith(yd, 2);
    let yd3 = Ith(yd, 3);
    let yd4 = Ith(yd, 4);
    let yd5 = Ith(yd, 5);

    let r1 = k1 * SUNRpowerI(y1, 4) * y2.sqrt();
    let r2 = k2 * y3 * y4;
    let r3 = k2 / K * y1 * y5;
    let r4 = k3 * y1 * y4 * y4;
    let r5 = k4 * y6 * y6 * y2.sqrt();
    let Fin = klA * (pCO2 / H - y2);

    Ith_set(resval, 1, yd1 + TWO * r1 - r2 + r3 + r4);
    Ith_set(resval, 2, yd2 + HALF * r1 + r4 + HALF * r5 - Fin);
    Ith_set(resval, 3, yd3 - r1 + r2 - r3);
    Ith_set(resval, 4, yd4 + r2 - r3 + TWO * r4);
    Ith_set(resval, 5, yd5 - r2 + r3 - r5);
    Ith_set(resval, 6, Ks * y1 * y4 - y6);

    0
}

/*
 * rhsQ routine. Computes quadrature(t,y).
 */

fn rhsQ(
    _t: sunrealtype,
    yy: &N_Vector,
    _yp: &N_Vector,
    qdot: &N_Vector,
    _user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    Ith_set(qdot, 1, Ith(yy, 1));

    0
}

const QUARTER: sunrealtype = 0.25;
const FOUR: sunrealtype = 4.0;
const EIGHT: sunrealtype = 8.0;

/*
 * resB routine. Residual for adjoint system.
 */
fn resB(
    _tt: sunrealtype,
    yy: &N_Vector,
    _yp: &N_Vector,
    yyB: &N_Vector,
    ypB: &N_Vector,
    rrB: &N_Vector,
    user_dataB: &mut Option<Box<dyn Any>>,
) -> i32 {
    let data = *user_dataB
        .as_mut()
        .and_then(|b| b.downcast_mut::<UserDataRec>())
        .expect("UserDataRec");
    let k1 = data.k1;
    let k2 = data.k2;
    let k3 = data.k3;
    let k4 = data.k4;
    let K = data.K;
    let klA = data.klA;
    let Ks = data.Ks;

    let y1 = Ith(yy, 1);
    let y2 = Ith(yy, 2);
    let y3 = Ith(yy, 3);
    let y4 = Ith(yy, 4);
    let y5 = Ith(yy, 5);
    let y6 = Ith(yy, 6);

    let yB1 = Ith(yyB, 1);
    let yB2 = Ith(yyB, 2);
    let yB3 = Ith(yyB, 3);
    let yB4 = Ith(yyB, 4);
    let yB5 = Ith(yyB, 5);
    let yB6 = Ith(yyB, 6);

    let ypB1 = Ith(ypB, 1);
    let ypB2 = Ith(ypB, 2);
    let ypB3 = Ith(ypB, 3);
    let ypB4 = Ith(ypB, 4);
    let ypB5 = Ith(ypB, 5);

    let y2tohalf = y2.sqrt();
    let y1to3 = y1 * y1 * y1;
    let k2overK = k2 / K;

    let mut tmp1 = k1 * y1to3 * y2tohalf;
    let mut tmp2 = k3 * y4 * y4;
    Ith_set(
        rrB,
        1,
        1.0 + ypB1 - (EIGHT * tmp1 + k2overK * y5 + tmp2) * yB1 - (TWO * tmp1 + tmp2) * yB2
            + (FOUR * tmp1 + k2overK * y5) * yB3
            + k2overK * y5 * (yB4 - yB5)
            - TWO * tmp2 * yB4
            + Ks * y4 * yB6,
    );

    tmp1 = k1 * y1 * y1to3 * (y2tohalf / y2);
    tmp2 = k4 * y6 * y6 * (y2tohalf / y2);
    Ith_set(
        rrB,
        2,
        ypB2 - tmp1 * yB1 - (QUARTER * tmp1 + QUARTER * tmp2 + klA) * yB2
            + HALF * tmp1 * yB3
            + HALF * tmp2 * yB5,
    );

    Ith_set(rrB, 3, ypB3 + k2 * y4 * (yB1 - yB3 - yB4 + yB5));

    tmp1 = k3 * y1 * y4;
    tmp2 = k2 * y3;
    Ith_set(
        rrB,
        4,
        ypB4 + (tmp2 - TWO * tmp1) * yB1
            - TWO * tmp1 * yB2
            - tmp2 * yB3
            - (tmp2 + FOUR * tmp1) * yB4
            + tmp2 * yB5
            + Ks * y1 * yB6,
    );

    Ith_set(rrB, 5, ypB5 - k2overK * y1 * (yB1 - yB3 - yB4 + yB5));

    Ith_set(rrB, 6, k4 * y6 * y2tohalf * (2.0 * yB5 - yB2) - yB6);

    0
}

/*
 * Print results after backward integration
 */
fn PrintOutput(_tfinal: sunrealtype, yB: &N_Vector, _ypB: &N_Vector) {
    print!(
        "dG/dy0: \t{}\n\t\t{}\n\t\t{}\n\t\t{}\n\t\t{}\n",
        fmt_ew(Ith(yB, 1), 12, 4),
        fmt_ew(Ith(yB, 2), 12, 4),
        fmt_ew(Ith(yB, 3), 12, 4),
        fmt_ew(Ith(yB, 4), 12, 4),
        fmt_ew(Ith(yB, 5), 12, 4)
    );
    print!("--------------------------------------------------------\n\n");
}

/*
 * Check function return value.
 *    opt == 0 means SUNDIALS function allocates memory so check if
 *             returned NULL pointer
 *    opt == 1 means SUNDIALS function returns an integer value so check if
 *             retval < 0
 *    opt == 2 means function allocates memory so check if returned
 *             NULL pointer
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
    /* Check if retval < 0 */
    else if opt == 1 {
        let retval = returnvalue.expect("retval");
        if retval < 0 {
            eprint!(
                "\nSUNDIALS_ERROR: {}() failed with retval = {}\n\n",
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
