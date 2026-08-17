//! Port of `examples/idas/serial/idasAkzoNob_dns.c`.
//!
//! Adjoint sensitivity example problem
//!
//! This IVP is a stiff system of 6 non-linear DAEs of index 1. The
//! problem originates from Akzo Nobel Central research in Arnhern,
//! The Netherlands, and describes a chemical process in which 2
//! species are mixed, while carbon dioxide is continuously added.
//! See <http://pitagora.dm.uniba.it/~testset/report/chemakzo.pdf>

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
const T1: sunrealtype = 1e-8; /* first time for output */

const TF: sunrealtype = 180.0; /* Final time. */
const NF: i32 = 25; /* Total number of outputs. */

const RTOL: sunrealtype = 1.0e-08;
const ATOL: sunrealtype = 1.0e-10;
const RTOLQ: sunrealtype = 1.0e-10;
const ATOLQ: sunrealtype = 1.0e-12;

const ZERO: sunrealtype = 0.0;
const HALF: sunrealtype = 0.5;
const ONE: sunrealtype = 1.0;
const TWO: sunrealtype = 2.0;

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

    /* Consistent IC for  y, y'. */
    let y01: sunrealtype = 0.444;
    let y02: sunrealtype = 0.00123;
    let y03: sunrealtype = 0.0;
    let y04: sunrealtype = 0.007;
    let y05: sunrealtype = 0.0;

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

    PrintHeader(RTOL, ATOL, &yy);
    /* Print initial states */
    PrintOutput(&mem, 0.0, &yy);

    let mut time: sunrealtype = 0.0;
    let mut tout: sunrealtype = T1;
    let mut nout: i32 = 0;
    let incr: sunrealtype = SUNRpowerR(TF / T1, ONE / NF as sunrealtype);

    /* FORWARD run. */
    loop {
        retval = IDASolve(&mem, tout, &mut time, &yy, &yp, IDA_NORMAL);
        if check_retval(Some(retval), "IDASolve", 1) != 0 {
            std::process::exit(1);
        }

        PrintOutput(&mem, time, &yy);

        nout += 1;
        tout *= incr;

        if nout > NF {
            break;
        }
    }

    retval = IDAGetQuad(&mem, &mut time, &q);
    if check_retval(Some(retval), "IDAGetQuad", 1) != 0 {
        std::process::exit(1);
    }

    print!("\n--------------------------------------------------------\n");
    print!("G:          {} \n", fmt_fw(Ith(&q, 1), 24, 16));
    print!("--------------------------------------------------------\n\n");

    retval = PrintFinalStats(&mem);
    if check_retval(Some(retval), "PrintFinalStats", 1) != 0 {
        std::process::exit(1);
    }

    IDAFree(&mut mem_opt);
    let _ = SUNLinSolFree(Some(LS));
    SUNMatDestroy(A);
    N_VDestroy(yy);
    N_VDestroy(yp);
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

    let r1 = k1 * SUNRpowerI(y1, 4) * SUNRsqrt(y2);
    let r2 = k2 * y3 * y4;
    let r3 = k2 / K * y1 * y5;
    let r4 = k3 * y1 * y4 * y4;
    let r5 = k4 * y6 * y6 * SUNRsqrt(y2);
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

fn PrintHeader(rtol: sunrealtype, avtol: sunrealtype, _y: &N_Vector) {
    print!(
        "\nidasAkzoNob_dns: Akzo Nobel chemical kinetics DAE serial example \
         problem for IDAS\n"
    );
    print!("Linear solver: DENSE, Jacobian is computed by IDAS.\n");
    print!(
        "Tolerance parameters:  rtol = {}   atol = {}\n",
        fmt_g(rtol, 6),
        fmt_g(avtol, 6)
    );
    print!(
        "---------------------------------------------------------------------\
         ------------\n"
    );
    print!("   t        y1        y2       y3       y4       y5");
    print!("      y6    | nst  k      h\n");
    print!(
        "---------------------------------------------------------------------\
         ------------\n"
    );
}

fn PrintOutput(mem: &IDAMem, t: sunrealtype, y: &N_Vector) {
    let mut retval: i32;
    let mut kused: i32 = 0;
    let mut nst: i64 = 0;
    let mut hused: sunrealtype = 0.0;

    /* C keeps the raw `yval` pointer live across the IDAGet* calls; the port
    snapshots the six components so no vector borrow is held across a
    library call. */
    let yval: [sunrealtype; 6] = {
        let d = N_VGetArrayPointer(y).expect("N_VGetArrayPointer");
        [d[0], d[1], d[2], d[3], d[4], d[5]]
    };

    retval = IDAGetLastOrder(mem, &mut kused);
    check_retval(Some(retval), "IDAGetLastOrder", 1);
    retval = IDAGetNumSteps(mem, &mut nst);
    check_retval(Some(retval), "IDAGetNumSteps", 1);
    retval = IDAGetLastStep(mem, &mut hused);
    check_retval(Some(retval), "IDAGetLastStep", 1);

    print!(
        "{} {} {} {} {} {} {} | {:3}  {:1} {}\n",
        fmt_ew(t, 8, 2),
        fmt_ew(yval[0], 8, 2),
        fmt_ew(yval[1], 8, 2),
        fmt_ew(yval[2], 8, 2),
        fmt_ew(yval[3], 8, 2),
        fmt_ew(yval[4], 8, 2),
        fmt_ew(yval[5], 8, 2),
        nst,
        kused,
        fmt_ew(hused, 8, 2)
    );
}

fn PrintFinalStats(mem: &IDAMem) -> i32 {
    let mut retval: i32;
    let mut nst: i64 = 0;
    let mut nni: i64 = 0;
    let mut nje: i64 = 0;
    let mut nre: i64 = 0;
    let mut nreLS: i64 = 0;
    let mut netf: i64 = 0;
    let mut ncfn: i64 = 0;

    retval = IDAGetNumSteps(mem, &mut nst);
    let _ = retval;
    retval = IDAGetNumResEvals(mem, &mut nre);
    let _ = retval;
    retval = IDAGetNumJacEvals(mem, &mut nje);
    let _ = retval;
    retval = IDAGetNumNonlinSolvIters(mem, &mut nni);
    let _ = retval;
    retval = IDAGetNumErrTestFails(mem, &mut netf);
    let _ = retval;
    retval = IDAGetNumNonlinSolvConvFails(mem, &mut ncfn);
    let _ = retval;
    retval = IDAGetNumLinResEvals(mem, &mut nreLS);

    print!("\nFinal Run Statistics: \n\n");
    print!("Number of steps                    = {}\n", nst);
    print!("Number of residual evaluations     = {}\n", nre + nreLS);
    print!("Number of Jacobian evaluations     = {}\n", nje);
    print!("Number of nonlinear iterations     = {}\n", nni);
    print!("Number of error test failures      = {}\n", netf);
    print!("Number of nonlinear conv. failures = {}\n", ncfn);

    retval
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
