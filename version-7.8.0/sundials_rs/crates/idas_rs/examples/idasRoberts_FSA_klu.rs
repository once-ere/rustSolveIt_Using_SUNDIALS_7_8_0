//! Port of `examples/idas/serial/idasRoberts_FSA_klu.c`.
//!
//! **Solver note.** `SUNLinSol_KLU` here is
//! [`sundials_core::sunlinsol_klu`], backed by the independent pure-Rust
//! sparse LU rather than SuiteSparse KLU. See
//! `differences/ATTRIBUTION.md`.
//!
//! Example problem:
//!
//! This simple example problem for IDA, due to Robertson,
//! is from chemical kinetics, and consists of the following three
//! equations:
//!
//!      dy1/dt = -p1*y1 + p2*y2*y3
//!      dy2/dt = p1*y1 - p2*y2*y3 - p3*y2**2
//!         0   = y1 + y2 + y3 - 1
//!
//! on the interval from t = 0.0 to t = 4.e10, with initial
//! conditions: y1 = 1, y2 = y3 = 0.The reaction rates are: p1=0.04,
//! p2=1e4, and p3=3e7
//!
//! Optionally, IDAS can compute sensitivities with respect to the
//! problem parameters p1, p2, and p3.
//! The sensitivity right hand side is given analytically through the
//! user routine fS (of type SensRhs1Fn).
//! Any of two sensitivity methods (SIMULTANEOUS and STAGGERED can be
//! used and sensitivities may be included in the error test or not
//! (error control set on SUNTRUE or SUNFALSE, respectively).
//!
//! Execution:
//!
//! If no sensitivities are desired:
//!    % idasRoberts_FSA_dns -nosensi
//! If sensitivities are to be computed:
//!    % idasRoberts_FSA_dns -sensi sensi_meth err_con
//! where sensi_meth is one of {sim, stg} and err_con is one of
//! {t, f}.

#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

use idas_rs::prelude::*;

/* Accessor macros

C: `Ith(v,i)` = `NV_Ith_S(v,i-1)` -- i-th vector component i=1..NEQ.
The guard returned by `N_VGetArrayPointer` is taken and dropped inside
these helpers, so it is never held across a library call. */

fn Ith(v: &N_Vector, i: usize) -> sunrealtype {
    N_VGetArrayPointer(v).expect("N_VGetArrayPointer")[i - 1]
}

fn Ith_set(v: &N_Vector, i: usize, x: sunrealtype) {
    N_VGetArrayPointer(v).expect("N_VGetArrayPointer")[i - 1] = x;
}

/* Problem Constants */

const NEQ: sunindextype = 3; /* number of equations  */
const T0: sunrealtype = 0.0; /* initial time */
const T1: sunrealtype = 0.4; /* first output time */
const TMULT: sunrealtype = 10.0; /* output time factor */
const NOUT: i32 = 12; /* number of output times */

const NS: i32 = 3; /* number of sensitivities computed */

const ZERO: sunrealtype = 0.0;
const ONE: sunrealtype = 1.0;

/* Type : UserData */

/* C hands `data->p` (the caller's own array) to IDASetSensParams, which
stores the POINTER; the internal DQ quadrature-sensitivity RHS perturbs
that very array around each `rhsQ` call.  The port shares it as a
`SensParams` handle (ARCHITECTURE §8) and hands IDASetSensParams a clone
of the same handle. */
struct UserData {
    p: SensParams, /* problem parameters */
    coef: sunrealtype,
}

/*
 *--------------------------------------------------------------------
 * MAIN PROGRAM
 *--------------------------------------------------------------------
 */

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let argc = argv.len();

    let mut retval: i32;

    let mut pbar: [sunrealtype; NS as usize] = [ZERO; NS as usize];
    let mut yS: Vec<N_Vector> = Vec::new();
    let mut ypS: Vec<N_Vector> = Vec::new();

    let mut sensi: sunbooleantype = SUNFALSE;
    let mut err_con: sunbooleantype = SUNFALSE;
    let mut sensi_meth: i32 = -1;

    /* Process arguments */
    ProcessArgs(argc, &argv, &mut sensi, &mut sensi_meth, &mut err_con);

    /* Create the SUNDIALS context object for this simulation */
    let mut sunctx: Option<SUNContext> = None;
    retval = SUNContext_Create(SUN_COMM_NULL, &mut sunctx);
    if check_retval(Some(retval), "SUNContext_Create", 1) != 0 {
        std::process::exit(1);
    }
    let ctx = sunctx.as_ref().expect("SUNContext").clone();

    /* User data structure */
    let data: Option<UserData> = Some(UserData {
        p: Rc::new(RefCell::new(vec![ZERO; 3])),
        coef: ZERO,
    });
    if check_retval(data.as_ref().map(|_| 0), "malloc", 2) != 0 {
        std::process::exit(1);
    }
    let mut data = data.expect("malloc");
    data.p.borrow_mut()[0] = 0.040;
    data.p.borrow_mut()[1] = 1.0e4;
    data.p.borrow_mut()[2] = 3.0e7;
    data.coef = 0.5;

    /* Initial conditions */
    let y = N_VNew_Serial(NEQ, &ctx);
    if check_retval(y.as_ref().map(|_| 0), "N_VNew_Serial", 0) != 0 {
        std::process::exit(1);
    }
    let y = y.expect("N_VNew_Serial");

    Ith_set(&y, 1, ONE);
    Ith_set(&y, 2, ZERO);
    Ith_set(&y, 3, ZERO);

    let yp = N_VClone(&y);
    if check_retval(yp.as_ref().map(|_| 0), "N_VNew_Serial", 0) != 0 {
        std::process::exit(1);
    }
    let yp = yp.expect("N_VClone");

    /* These initial conditions are NOT consistent. See IDACalcIC below. */
    Ith_set(&yp, 1, 0.1);
    Ith_set(&yp, 2, ZERO);
    Ith_set(&yp, 3, ZERO);

    /* Create IDAS object */
    let ida_mem = IDACreate(&ctx);
    if check_retval(ida_mem.as_ref().map(|_| 0), "IDACreate", 0) != 0 {
        std::process::exit(1);
    }
    let mut ida_mem_opt = ida_mem;
    let ida_mem = ida_mem_opt.as_ref().expect("IDACreate").clone();

    /* Allocate space for IDAS */
    retval = IDAInit(&ida_mem, res, T0, &y, &yp);
    if check_retval(Some(retval), "IDAInit", 1) != 0 {
        std::process::exit(1);
    }

    /* Specify scalar relative tol. and vector absolute tol. */
    let reltol: sunrealtype = 1.0e-6;
    let abstol = N_VClone(&y).expect("N_VClone");
    Ith_set(&abstol, 1, 1.0e-8);
    Ith_set(&abstol, 2, 1.0e-14);
    Ith_set(&abstol, 3, 1.0e-6);
    retval = IDASVtolerances(&ida_mem, reltol, &abstol);
    if check_retval(Some(retval), "IDASVtolerances", 1) != 0 {
        std::process::exit(1);
    }

    /* Set ID vector */
    let id = N_VClone(&y).expect("N_VClone");
    Ith_set(&id, 1, 1.0);
    Ith_set(&id, 2, 1.0);
    Ith_set(&id, 3, 0.0);
    retval = IDASetId(&ida_mem, Some(&id));
    if check_retval(Some(retval), "IDASetId", 1) != 0 {
        std::process::exit(1);
    }

    /* C keeps `data` alive next to the integrator and re-reads `data->p`
    below to fill `pbar` and to hand the parameter array to
    IDASetSensParams; the Rust box moves into the mem record, so keep a
    clone of the SHARED parameter handle here — the same array `res`,
    `resS` and the internal DQ routines see. */
    let p_saved = Rc::clone(&data.p);

    /* Attach user data */
    let data_box: Box<dyn Any> = Box::new(data);
    retval = IDASetUserData(&ida_mem, Some(data_box));
    if check_retval(Some(retval), "IDASetUserData", 1) != 0 {
        std::process::exit(1);
    }

    /* Create dense SUNMatrix for use in linear solves */
    let nnz: sunindextype = NEQ * NEQ;
    let A = SUNSparseMatrix(NEQ, NEQ, nnz, SUN_CSC_MAT, &ctx);
    if check_retval(A.as_ref().map(|_| 0), "SUNSparseMatrix", 0) != 0 {
        std::process::exit(1);
    }
    let A = A.expect("SUNSparseMatrix");

    /* Create dense SUNLinearSolver object */
    let LS = SUNLinSol_KLU(&y, &A, &ctx);
    if check_retval(LS.as_ref().map(|_| 0), "SUNLinSol_KLU", 0) != 0 {
        std::process::exit(1);
    }
    let LS = LS.expect("SUNLinSol_KLU");

    /* Attach the matrix and linear solver */
    retval = IDASetLinearSolver(&ida_mem, &LS, Some(&A));
    if check_retval(Some(retval), "IDASetLinearSolver", 1) != 0 {
        std::process::exit(1);
    }

    /* Set the user-supplied Jacobian routine */
    retval = IDASetJacFn(&ida_mem, Some(Jac));
    if check_retval(Some(retval), "IDASetJacFn", 1) != 0 {
        std::process::exit(1);
    }

    print!("\n3-species chemical kinetics problem\n");

    /* Sensitivity-related settings */
    if sensi {
        pbar[0] = p_saved.borrow()[0];
        pbar[1] = p_saved.borrow()[1];
        pbar[2] = p_saved.borrow()[2];

        let yS_opt = N_VCloneVectorArray(NS, &y);
        if check_retval(yS_opt.as_ref().map(|_| 0), "N_VCloneVectorArray", 0) != 0 {
            std::process::exit(1);
        }
        yS = yS_opt.expect("N_VCloneVectorArray");
        for is in 0..NS as usize {
            N_VConst(ZERO, &yS[is]);
        }

        let ypS_opt = N_VCloneVectorArray(NS, &y);
        if check_retval(ypS_opt.as_ref().map(|_| 0), "N_VCloneVectorArray", 0) != 0 {
            std::process::exit(1);
        }
        ypS = ypS_opt.expect("N_VCloneVectorArray");
        for is in 0..NS as usize {
            N_VConst(ZERO, &ypS[is]);
        }

        /*
         * Only non-zero sensitivity I.C. are ypS[0]:
         * - Ith(ypS[0],1) = -ONE;
         * - Ith(ypS[0],2) =  ONE;
         *
         * They are not set. IDACalcIC also computes consistent IC for sensitivities.
         */

        retval = IDASensInit(&ida_mem, NS, sensi_meth, Some(resS), &yS, &ypS);
        if check_retval(Some(retval), "IDASensInit", 1) != 0 {
            std::process::exit(1);
        }

        retval = IDASensEEtolerances(&ida_mem);
        if check_retval(Some(retval), "IDASensEEtolerances", 1) != 0 {
            std::process::exit(1);
        }

        retval = IDASetSensErrCon(&ida_mem, err_con);
        if check_retval(Some(retval), "IDASetSensErrCon", 1) != 0 {
            std::process::exit(1);
        }

        retval = IDASetSensParams(&ida_mem, Some(Rc::clone(&p_saved)), Some(&pbar[..]), None);
        if check_retval(Some(retval), "IDASetSensParams", 1) != 0 {
            std::process::exit(1);
        }

        print!("Sensitivity: YES ");
        if sensi_meth == IDA_SIMULTANEOUS {
            print!("( SIMULTANEOUS +");
        } else {
            print!("( STAGGERED +");
        }
        if err_con {
            print!(" FULL ERROR CONTROL )");
        } else {
            print!(" PARTIAL ERROR CONTROL )");
        }
    } else {
        print!("Sensitivity: NO ");
    }

    /*----------------------------------------------------------
     *               Q U A D R A T U R E S
     * ---------------------------------------------------------*/
    let yQ = N_VNew_Serial(2, &ctx).expect("N_VNew_Serial");

    Ith_set(&yQ, 1, 0.0);
    Ith_set(&yQ, 2, 0.0);

    let _ = IDAQuadInit(&ida_mem, rhsQ, &yQ);

    let yQS = N_VCloneVectorArray(NS, &yQ).expect("N_VCloneVectorArray");
    for is in 0..NS as usize {
        N_VConst(ZERO, &yQS[is]);
    }

    let _ = IDAQuadSensInit(&ida_mem, None, &yQS);

    /* Call IDACalcIC to compute consistent initial conditions. If sensitivity is
    enabled, this function also try to find consistent IC for the sensitivities. */

    retval = IDACalcIC(&ida_mem, IDA_YA_YDP_INIT, T1);
    if check_retval(Some(retval), "IDACalcIC", 1) != 0 {
        std::process::exit(1);
    }

    retval = IDAGetConsistentIC(&ida_mem, Some(&y), Some(&yp));
    if check_retval(Some(retval), "IDAGetConsistentIC", 1) != 0 {
        std::process::exit(1);
    }

    PrintIC(&y, &yp);

    if sensi {
        let _ = IDAGetSensConsistentIC(&ida_mem, Some(&yS[..]), Some(&ypS[..]));
        PrintSensIC(&y, &yp, &yS, &ypS);
    }

    /* In loop over output points, call IDA, print results, test for error */

    print!("\n\n");
    print!("===========================================");
    print!("============================\n");
    print!("     T     Q       H      NST           y1");
    print!("           y2           y3    \n");
    print!("===========================================");
    print!("============================\n");

    let mut t: sunrealtype = 0.0;
    let mut tout: sunrealtype = T1;
    let mut iout: i32 = 1;
    while iout <= NOUT {
        retval = IDASolve(&ida_mem, tout, &mut t, &y, &yp, IDA_NORMAL);
        if check_retval(Some(retval), "IDASolve", 1) != 0 {
            break;
        }

        PrintOutput(&ida_mem, t, &y);

        if sensi {
            retval = IDAGetSens(&ida_mem, &mut t, &yS);
            if check_retval(Some(retval), "IDAGetSens", 1) != 0 {
                break;
            }
            PrintSensOutput(&yS);
        }
        print!("-----------------------------------------");
        print!("------------------------------\n");

        iout += 1;
        tout *= TMULT;
    }

    print!("\nQuadrature:\n");
    let _ = IDAGetQuad(&ida_mem, &mut t, &yQ);
    print!("G:      {}\n", fmt_ew(Ith(&yQ, 1), 10, 4));

    if sensi {
        let _ = IDAGetQuadSens(&ida_mem, &mut t, &yQS);
        print!("\nSensitivities at t={}:\n", fmt_g(t, 6));
        print!("dG/dp1: {}\n", fmt_ew(Ith(&yQS[0], 1), 11, 4));
        print!("dG/dp1: {}\n", fmt_ew(Ith(&yQS[1], 1), 11, 4));
        print!("dG/dp1: {}\n", fmt_ew(Ith(&yQS[2], 1), 11, 4));
    }

    /* Print final statistics */
    PrintFinalStats(&ida_mem, sensi);

    /* Free memory */
    N_VDestroy(y);
    N_VDestroy(yp);
    N_VDestroy(abstol);
    N_VDestroy(id);
    N_VDestroy(yQ);
    if sensi {
        N_VDestroyVectorArray(yS, NS);
        N_VDestroyVectorArray(ypS, NS);
        N_VDestroyVectorArray(yQS, NS);
    }
    /* `data` is owned by the integrator (C: `free(data)`) */
    IDAFree(&mut ida_mem_opt);
    let _ = SUNLinSolFree(Some(LS));
    SUNMatDestroy(A);
    let _ = SUNContext_Free(&mut sunctx);

    std::process::exit(0);
}

/*
 *--------------------------------------------------------------------
 * FUNCTIONS CALLED BY IDAS
 *--------------------------------------------------------------------
 */

/*
 * Residual routine. Compute F(t,y,y',p).
 */
fn res(
    _t: sunrealtype,
    yy: &N_Vector,
    yp: &N_Vector,
    resval: &N_Vector,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    let data = user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<UserData>())
        .expect("user_data is UserData");
    let p1 = data.p.borrow()[0];
    let p2 = data.p.borrow()[1];
    let p3 = data.p.borrow()[2];

    let y1 = Ith(yy, 1);
    let y2 = Ith(yy, 2);
    let y3 = Ith(yy, 3);

    let yp1 = Ith(yp, 1);
    let yp2 = Ith(yp, 2);

    Ith_set(resval, 1, yp1 + p1 * y1 - p2 * y2 * y3);
    Ith_set(resval, 2, yp2 - p1 * y1 + p2 * y2 * y3 + p3 * y2 * y2);
    Ith_set(resval, 3, y1 + y2 + y3 - ONE);

    0
}

/*
 * resS routine. Compute sensitivity r.h.s.
 */

#[allow(clippy::too_many_arguments)]
fn resS(
    _Ns: i32,
    _t: sunrealtype,
    yy: &N_Vector,
    _yp: &N_Vector,
    _resval: &N_Vector,
    yyS: &[N_Vector],
    ypS: &[N_Vector],
    resvalS: &[N_Vector],
    user_data: &mut Option<Box<dyn Any>>,
    _tmp1: &N_Vector,
    _tmp2: &N_Vector,
    _tmp3: &N_Vector,
) -> i32 {
    let data = user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<UserData>())
        .expect("user_data is UserData");
    let p1 = data.p.borrow()[0];
    let p2 = data.p.borrow()[1];
    let p3 = data.p.borrow()[2];

    let y1 = Ith(yy, 1);
    let y2 = Ith(yy, 2);
    let y3 = Ith(yy, 3);

    for is in 0..NS as usize {
        let s1 = Ith(&yyS[is], 1);
        let s2 = Ith(&yyS[is], 2);
        let s3 = Ith(&yyS[is], 3);

        let sd1 = Ith(&ypS[is], 1);
        let sd2 = Ith(&ypS[is], 2);

        let mut rs1 = sd1 + p1 * s1 - p2 * y3 * s2 - p2 * y2 * s3;
        let mut rs2 = sd2 - p1 * s1 + p2 * y3 * s2 + p2 * y2 * s3 + 2.0 * p3 * y2 * s2;
        let rs3 = s1 + s2 + s3;

        match is {
            0 => {
                rs1 += y1;
                rs2 -= y1;
            }
            1 => {
                rs1 -= y2 * y3;
                rs2 += y2 * y3;
            }
            2 => {
                rs2 += y2 * y2;
            }
            _ => {}
        }

        Ith_set(&resvalS[is], 1, rs1);
        Ith_set(&resvalS[is], 2, rs2);
        Ith_set(&resvalS[is], 3, rs3);
    }

    0
}

fn rhsQ(
    _t: sunrealtype,
    y: &N_Vector,
    _yp: &N_Vector,
    ypQ: &N_Vector,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    let data = user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<UserData>())
        .expect("user_data is UserData");

    Ith_set(ypQ, 1, Ith(y, 3));

    Ith_set(
        ypQ,
        2,
        data.coef * (Ith(y, 1) * Ith(y, 1) + Ith(y, 2) * Ith(y, 2) + Ith(y, 3) * Ith(y, 3)),
    );

    0
}

/*
 *--------------------------------------------------------------------
 * PRIVATE FUNCTIONS
 *--------------------------------------------------------------------
 */

/*
 * Process and verify arguments to idasfwddenx.
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
            *sensi_meth = IDA_SIMULTANEOUS;
        } else if argv[2] == "stg" {
            *sensi_meth = IDA_STAGGERED;
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
    print!("         sensi_meth = sim or stg\n");
    print!("         err_con    = t or f\n");

    std::process::exit(0);
}

fn PrintIC(y: &N_Vector, yp: &N_Vector) {
    let data: [sunrealtype; 3] = {
        let d = N_VGetArrayPointer(y).expect("N_VGetArrayPointer");
        [d[0], d[1], d[2]]
    };
    print!("\n\nConsistent IC:\n");
    print!("\ty = ");
    print!(
        "{} {} {} \n",
        fmt_ew(data[0], 12, 4),
        fmt_ew(data[1], 12, 4),
        fmt_ew(data[2], 12, 4)
    );

    let data: [sunrealtype; 3] = {
        let d = N_VGetArrayPointer(yp).expect("N_VGetArrayPointer");
        [d[0], d[1], d[2]]
    };
    print!("\typ= ");
    print!(
        "{} {} {} \n",
        fmt_ew(data[0], 12, 4),
        fmt_ew(data[1], 12, 4),
        fmt_ew(data[2], 12, 4)
    );
}

fn PrintSensIC(_y: &N_Vector, _yp: &N_Vector, yS: &[N_Vector], ypS: &[N_Vector]) {
    let sdata = sens_row(&yS[0]);
    print!("                  Sensitivity 1  ");

    print!("\n\ts1 = ");
    print!(
        "{} {} {} \n",
        fmt_ew(sdata[0], 12, 4),
        fmt_ew(sdata[1], 12, 4),
        fmt_ew(sdata[2], 12, 4)
    );
    let sdata = sens_row(&ypS[0]);
    print!("\ts1'= ");
    print!(
        "{} {} {} \n",
        fmt_ew(sdata[0], 12, 4),
        fmt_ew(sdata[1], 12, 4),
        fmt_ew(sdata[2], 12, 4)
    );

    print!("                  Sensitivity 2  ");
    let sdata = sens_row(&yS[1]);
    print!("\n\ts2 = ");
    print!(
        "{} {} {} \n",
        fmt_ew(sdata[0], 12, 4),
        fmt_ew(sdata[1], 12, 4),
        fmt_ew(sdata[2], 12, 4)
    );
    let sdata = sens_row(&ypS[1]);
    print!("\ts2'= ");
    print!(
        "{} {} {} \n",
        fmt_ew(sdata[0], 12, 4),
        fmt_ew(sdata[1], 12, 4),
        fmt_ew(sdata[2], 12, 4)
    );

    print!("                  Sensitivity 3  ");
    let sdata = sens_row(&yS[2]);
    print!("\n\ts3 = ");
    print!(
        "{} {} {} \n",
        fmt_ew(sdata[0], 12, 4),
        fmt_ew(sdata[1], 12, 4),
        fmt_ew(sdata[2], 12, 4)
    );
    let sdata = sens_row(&ypS[2]);
    print!("\ts3'= ");
    print!(
        "{} {} {} \n",
        fmt_ew(sdata[0], 12, 4),
        fmt_ew(sdata[1], 12, 4),
        fmt_ew(sdata[2], 12, 4)
    );
}

/* C holds a raw `sunrealtype*` into the vector data; the port snapshots
the three components so no borrow is held across a library call. */
fn sens_row(v: &N_Vector) -> [sunrealtype; 3] {
    let d = N_VGetArrayPointer(v).expect("N_VGetArrayPointer");
    [d[0], d[1], d[2]]
}

/*
 * Print current t, step count, order, stepsize, and solution.
 */

fn PrintOutput(ida_mem: &IDAMem, t: sunrealtype, u: &N_Vector) {
    let mut nst: i64 = 0;
    let mut qu: i32 = 0;
    let mut hu: sunrealtype = 0.0;
    let mut retval: i32;

    let udata = sens_row(u);

    retval = IDAGetNumSteps(ida_mem, &mut nst);
    check_retval(Some(retval), "IDAGetNumSteps", 1);
    retval = IDAGetLastOrder(ida_mem, &mut qu);
    check_retval(Some(retval), "IDAGetLastOrder", 1);
    retval = IDAGetLastStep(ida_mem, &mut hu);
    check_retval(Some(retval), "IDAGetLastStep", 1);

    print!(
        "{} {:2}  {} {:5}\n",
        fmt_ew(t, 8, 3),
        qu,
        fmt_ew(hu, 8, 3),
        nst
    );

    print!("                  Solution       ");

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

fn PrintSensOutput(uS: &[N_Vector]) {
    let sdata = sens_row(&uS[0]);
    print!("                  Sensitivity 1  ");

    print!(
        "{} {} {} \n",
        fmt_ew(sdata[0], 12, 4),
        fmt_ew(sdata[1], 12, 4),
        fmt_ew(sdata[2], 12, 4)
    );

    let sdata = sens_row(&uS[1]);
    print!("                  Sensitivity 2  ");

    print!(
        "{} {} {} \n",
        fmt_ew(sdata[0], 12, 4),
        fmt_ew(sdata[1], 12, 4),
        fmt_ew(sdata[2], 12, 4)
    );

    let sdata = sens_row(&uS[2]);
    print!("                  Sensitivity 3  ");

    print!(
        "{} {} {} \n",
        fmt_ew(sdata[0], 12, 4),
        fmt_ew(sdata[1], 12, 4),
        fmt_ew(sdata[2], 12, 4)
    );
}

/*
 * Jacobian routine, in compressed sparse column form.
 */

fn Jac(
    _t: sunrealtype,
    cj: sunrealtype,
    yy: &N_Vector,
    _yp: &N_Vector,
    _resvec: &N_Vector,
    JJ: &SUNMatrix,
    user_data: &mut Option<Box<dyn Any>>,
    _tmp1: &N_Vector,
    _tmp2: &N_Vector,
    _tmp3: &N_Vector,
) -> i32 {
    let yval = N_VGetArrayPointer(yy).expect("N_VGetArrayPointer");
    let ud = user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<UserData>())
        .expect("UserData");
    let p1 = ud.p.borrow()[0];
    let p2 = ud.p.borrow()[1];
    let p3 = ud.p.borrow()[2];

    SUNMatZero(JJ);

    /* One borrow for all three arrays: taking them separately would be a
    second mutable borrow of the same matrix content. */
    let mut m = SUNSparseMatrix_Content(JJ);
    let m = &mut *m;
    let (data, rowvals, colptrs) = (&mut m.data, &mut m.indexvals, &mut m.indexptrs);

    colptrs[0] = 0;
    colptrs[1] = 3;
    colptrs[2] = 6;
    colptrs[3] = 9;

    data[0] = p1 + cj;
    rowvals[0] = 0;
    data[1] = -p1;
    rowvals[1] = 1;
    data[2] = ONE;
    rowvals[2] = 2;

    data[3] = -p2 * yval[2];
    rowvals[3] = 0;
    data[4] = p2 * yval[2] + 2.0 * p3 * yval[1] + cj;
    rowvals[4] = 1;
    data[5] = ONE;
    rowvals[5] = 2;

    data[6] = -p2 * yval[1];
    rowvals[6] = 0;
    data[7] = p2 * yval[1];
    rowvals[7] = 1;
    data[8] = ONE;
    rowvals[8] = 2;

    0
}

/*
 * Print some final statistics from the IDAS memory.
 */

fn PrintFinalStats(ida_mem: &IDAMem, sensi: sunbooleantype) {
    let mut nst: i64 = 0;
    let mut nfe: i64 = 0;
    let mut nsetups: i64 = 0;
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
    let mut nje: i64 = 0;
    let mut nfeLS: i64 = 0;

    let retval = IDAGetNumSteps(ida_mem, &mut nst);
    check_retval(Some(retval), "IDAGetNumSteps", 1);
    let retval = IDAGetNumResEvals(ida_mem, &mut nfe);
    check_retval(Some(retval), "IDAGetNumRhsEvals", 1);
    let retval = IDAGetNumLinSolvSetups(ida_mem, &mut nsetups);
    check_retval(Some(retval), "IDAGetNumLinSolvSetups", 1);
    let retval = IDAGetNumErrTestFails(ida_mem, &mut netf);
    check_retval(Some(retval), "IDAGetNumErrTestFails", 1);
    let retval = IDAGetNumNonlinSolvIters(ida_mem, &mut nni);
    check_retval(Some(retval), "IDAGetNumNonlinSolvIters", 1);
    let retval = IDAGetNumNonlinSolvConvFails(ida_mem, &mut nnf);
    check_retval(Some(retval), "IDAGetNumNonlinSolvConvFails", 1);
    let retval = IDAGetNumStepSolveFails(ida_mem, &mut ncfn);
    check_retval(Some(retval), "IDAGetNumStepSolveFails", 1);

    if sensi {
        let retval = IDAGetSensNumResEvals(ida_mem, &mut nfSe);
        check_retval(Some(retval), "IDAGetSensNumRhsEvals", 1);
        let retval = IDAGetNumResEvalsSens(ida_mem, &mut nfeS);
        check_retval(Some(retval), "IDAGetNumResEvalsSens", 1);
        let retval = IDAGetSensNumLinSolvSetups(ida_mem, &mut nsetupsS);
        check_retval(Some(retval), "IDAGetSensNumLinSolvSetups", 1);
        let retval = IDAGetSensNumErrTestFails(ida_mem, &mut netfS);
        check_retval(Some(retval), "IDAGetSensNumErrTestFails", 1);
        let retval = IDAGetSensNumNonlinSolvIters(ida_mem, &mut nniS);
        check_retval(Some(retval), "IDAGetSensNumNonlinSolvIters", 1);
        let retval = IDAGetSensNumNonlinSolvConvFails(ida_mem, &mut nnfS);
        check_retval(Some(retval), "IDAGetSensNumNonlinSolvConvFails", 1);
        let retval = IDAGetNumStepSensSolveFails(ida_mem, &mut ncfnS);
        check_retval(Some(retval), "IDAGetNumStepSolveFails", 1);
    }

    let retval = IDAGetNumJacEvals(ida_mem, &mut nje);
    check_retval(Some(retval), "IDAGetNumJacEvals", 1);
    let retval = IDAGetNumLinResEvals(ida_mem, &mut nfeLS);
    check_retval(Some(retval), "IDAGetNumLinResEvals", 1);

    print!("\nFinal Statistics\n\n");
    /* C: printf("nst     = %5ld\n\n", nst) */
    print!("nst     = {:>5}\n\n", nst);
    /* C: printf("nfe     = %5ld\n", nfe) */
    print!("nfe     = {:>5}\n", nfe);
    /* C: printf("netf    = %5ld    nsetups  = %5ld\n", netf, nsetups) */
    print!("netf    = {:>5}    nsetups  = {:>5}\n", netf, nsetups);
    /* C: printf("nni     = %5ld    nnf      = %5ld\n", nni, nnf) */
    print!("nni     = {:>5}    nnf      = {:>5}\n", nni, nnf);
    /* C: printf("ncfn    = %5ld\n", ncfn) */
    print!("ncfn    = {:>5}\n", ncfn);

    if sensi {
        print!("\n");
        /* C: printf("nfSe    = %5ld    nfeS     = %5ld\n", nfSe, nfeS) */
        print!("nfSe    = {:>5}    nfeS     = {:>5}\n", nfSe, nfeS);
        /* C: printf("netfs   = %5ld    nsetupsS = %5ld\n", netfS, nsetupsS) */
        print!("netfs   = {:>5}    nsetupsS = {:>5}\n", netfS, nsetupsS);
        /* C: printf("nniS    = %5ld    nnfS     = %5ld\n", nniS, nnfS) */
        print!("nniS    = {:>5}    nnfS     = {:>5}\n", nniS, nnfS);
        /* C: printf("ncfnS   = %5ld\n", ncfnS) */
        print!("ncfnS   = {:>5}\n", ncfnS);
    }

    print!("\n");
    /* C: printf("nje    = %5ld    nfeLS     = %5ld\n", nje, nfeLS) */
    print!("nje    = {:>5}    nfeLS     = {:>5}\n", nje, nfeLS);
}

/*
 * Check function return value...
 *   opt == 0 means SUNDIALS function allocates memory so check if
 *            returned NULL pointer
 *   opt == 1 means SUNDIALS function returns an integer value so check if
 *            retval < 0
 *   opt == 2 means function allocates memory so check if returned
 *            NULL pointer
 */

fn check_retval(returnvalue: Option<i32>, funcname: &str, opt: i32) -> i32 {
    /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
    if opt == 0 && returnvalue.is_none() {
        eprint!(
            "\nSUNDIALS_ERROR: {}() failed - returned NULL pointer\n\n",
            funcname
        );
        return 1;
    } else if opt == 1 {
        /* Check if retval < 0 */
        let retval = returnvalue.expect("retval");
        if retval < 0 {
            eprint!(
                "\nSUNDIALS_ERROR: {}() failed with retval = {}\n\n",
                funcname, retval
            );
            return 1;
        }
    } else if opt == 2 && returnvalue.is_none() {
        /* Check if function returned NULL pointer - no memory allocated */
        eprint!(
            "\nMEMORY_ERROR: {}() failed - returned NULL pointer\n\n",
            funcname
        );
        return 1;
    }

    0
}
