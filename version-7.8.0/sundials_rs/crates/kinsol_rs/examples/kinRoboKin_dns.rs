//! Port of `examples/kinsol/serial/kinRoboKin_dns.c`.
//!
//! This example solves a nonlinear system from robot kinematics.
//!
//! Source: "Handbook of Test Problems in Local and Global Optimization",
//!             C.A. Floudas, P.M. Pardalos et al.
//!             Kluwer Academic Publishers, 1999.
//! Test problem 6 from Section 14.1, Chapter 14
//!
//! The nonlinear system is solved by KINSOL using the DENSE linear
//! solver.
//!
//! Constraints are imposed to make all components of the solution
//! be within [-1,1].

#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

use std::any::Any;

use kinsol_rs::prelude::*;
use kinsol_rs::sundials_futils::SUNFileClose;

/* Problem Constants */

const NVAR: i32 = 8; /* variables */
const NEQ: sunindextype = 3 * NVAR as sunindextype; /* equations + bounds */

const FTOL: sunrealtype = 1.0e-5; /* function tolerance */
const STOL: sunrealtype = 1.0e-5; /* step tolerance */

const ZERO: sunrealtype = 0.0;
const ONE: sunrealtype = 1.0;
const TWO: sunrealtype = 2.0;

/* C macros `Ith(v,i)` = `NV_Ith_S(v,i-1)` and
`IJth(A,i,j)` = `SM_ELEMENT_D(A,i-1,j-1)`; i, j are 1-based. */

fn Ith(v: &N_Vector, i: i32) -> sunrealtype {
    N_VGetArrayPointer(v).expect("N_VGetArrayPointer")[(i - 1) as usize]
}

fn Ith_set(v: &N_Vector, i: i32, x: sunrealtype) {
    N_VGetArrayPointer(v).expect("N_VGetArrayPointer")[(i - 1) as usize] = x;
}

fn IJth_set(A: &SUNMatrix, i: sunindextype, j: sunindextype, x: sunrealtype) {
    SM_ELEMENT_D_set(A, i - 1, j - 1, x);
}

/*
 *--------------------------------------------------------------------
 * MAIN PROGRAM
 *--------------------------------------------------------------------
 */

fn main() {
    let fnormtol: sunrealtype;
    let scsteptol: sunrealtype;
    let mset: i32;
    let mut retval: i32;
    let mut i: i32;

    print!("\nRobot Kinematics Example\n");
    print!("8 variables; -1 <= x_i <= 1\n");
    print!("KINSOL problem size: 8 + 2*8 = 24 \n\n");

    /* Create the SUNDIALS context that all SUNDIALS objects require */
    let mut sunctx: Option<SUNContext> = None;
    retval = SUNContext_Create(SUN_COMM_NULL, &mut sunctx);
    if check_retval(retval, "SUNContext_Create") != 0 {
        std::process::exit(1);
    }
    let ctx = sunctx.clone().unwrap();

    /* Create vectors for solution, scales, and constraints */

    let y = N_VNew_Serial(NEQ, &ctx);
    if check_retval_null(&y, "N_VNew_Serial") != 0 {
        std::process::exit(1);
    }
    let y = y.unwrap();

    let scale = N_VNew_Serial(NEQ, &ctx);
    if check_retval_null(&scale, "N_VNew_Serial") != 0 {
        std::process::exit(1);
    }
    let scale = scale.unwrap();

    let constraints = N_VNew_Serial(NEQ, &ctx);
    if check_retval_null(&constraints, "N_VNew_Serial") != 0 {
        std::process::exit(1);
    }
    let constraints = constraints.unwrap();

    /* Initialize and allocate memory for KINSOL */

    let mut kmem = KINCreate(&ctx);
    if check_retval_null(&kmem, "KINCreate") != 0 {
        std::process::exit(1);
    }
    let kin = kmem.clone().unwrap();

    retval = KINInit(&kin, func, &y); /* y passed as a template */
    if check_retval(retval, "KINInit") != 0 {
        std::process::exit(1);
    }

    /* Set optional inputs */

    N_VConst(ZERO, &constraints);
    i = NVAR + 1;
    while i <= NEQ as i32 {
        Ith_set(&constraints, i, ONE);
        i += 1;
    }

    retval = KINSetConstraints(&kin, Some(&constraints));
    if check_retval(retval, "KINSetConstraints") != 0 {
        std::process::exit(1);
    }

    fnormtol = FTOL;
    retval = KINSetFuncNormTol(&kin, fnormtol);
    if check_retval(retval, "KINSetFuncNormTol") != 0 {
        std::process::exit(1);
    }

    scsteptol = STOL;
    retval = KINSetScaledStepTol(&kin, scsteptol);
    if check_retval(retval, "KINSetScaledStepTol") != 0 {
        std::process::exit(1);
    }

    /* Create dense SUNMatrix */
    let J = SUNDenseMatrix(NEQ, NEQ, &ctx);
    if check_retval_null(&J, "SUNDenseMatrix") != 0 {
        std::process::exit(1);
    }
    let J = J.unwrap();

    /* Create dense SUNLinearSolver object */
    let LS = SUNLinSol_Dense(&y, &J, &ctx);
    if check_retval_null(&LS, "SUNLinSol_Dense") != 0 {
        std::process::exit(1);
    }
    let LS = LS.unwrap();

    /* Attach the matrix and linear solver to KINSOL */
    retval = KINSetLinearSolver(&kin, &LS, Some(&J));
    if check_retval(retval, "KINSetLinearSolver") != 0 {
        std::process::exit(1);
    }

    /* Set the Jacobian function */
    retval = KINSetJacFn(&kin, Some(jac));
    if check_retval(retval, "KINSetJacFn") != 0 {
        std::process::exit(1);
    }

    /* Indicate exact Newton */

    mset = 1;
    retval = KINSetMaxSetupCalls(&kin, mset as i64);
    if check_retval(retval, "KINSetMaxSetupCalls") != 0 {
        std::process::exit(1);
    }

    /* Initial guess */

    N_VConst(ONE, &y);
    i = 1;
    while i <= NVAR {
        Ith_set(&y, i, SUNRsqrt(TWO) / TWO);
        i += 1;
    }

    print!("Initial guess:\n");
    PrintOutput(&y);

    /* Call KINSol to solve problem */

    N_VConst(ONE, &scale);
    retval = KINSol(
        &kin,           /* KINSol memory block */
        &y,             /* initial guess on input; solution vector */
        KIN_LINESEARCH, /* global strategy choice */
        &scale,         /* scaling vector, for the variable cc */
        &scale,
    ); /* scaling vector for function values fval */
    if check_retval(retval, "KINSol") != 0 {
        std::process::exit(1);
    }

    print!("\nComputed solution:\n");
    PrintOutput(&y);

    /* Print final statistics to screen and file */

    print!("\nFinal statsistics:\n");
    let _ = KINPrintAllStats(
        &kin,
        &SUNFile::Stdout,
        SUNOutputFormat::SUN_OUTPUTFORMAT_TABLE,
    );

    let mut FID = SUNFile::fopen("kinRoboKin_dns_stats.csv", "w");
    let _ = KINPrintAllStats(&kin, &FID, SUNOutputFormat::SUN_OUTPUTFORMAT_CSV);
    SUNFileClose(&mut FID);

    /* free memory */
    N_VDestroy(y);
    N_VDestroy(scale);
    N_VDestroy(constraints);
    KINFree(&mut kmem);
    let _ = SUNLinSolFree(Some(LS));
    SUNMatDestroy(J);
    let _ = SUNContext_Free(&mut sunctx);
}

/*
 * System function
 */

fn func(y: &N_Vector, f: &N_Vector, _user_data: &mut Option<Box<dyn Any>>) -> i32 {
    let yd = N_VGetArrayPointer(y).expect("N_VGetArrayPointer");
    let mut fd = N_VGetArrayPointer(f).expect("N_VGetArrayPointer");

    let x1 = yd[0];
    let l1 = yd[8];
    let u1 = yd[16];
    let x2 = yd[1];
    let l2 = yd[9];
    let u2 = yd[17];
    let x3 = yd[2];
    let l3 = yd[10];
    let u3 = yd[18];
    let x4 = yd[3];
    let l4 = yd[11];
    let u4 = yd[19];
    let x5 = yd[4];
    let l5 = yd[12];
    let u5 = yd[20];
    let x6 = yd[5];
    let l6 = yd[13];
    let u6 = yd[21];
    let x7 = yd[6];
    let l7 = yd[14];
    let u7 = yd[22];
    let x8 = yd[7];
    let l8 = yd[15];
    let u8 = yd[23];

    /* Nonlinear equations */

    let eq1 = -0.1238 * x1 + x7 - 0.001637 * x2 - 0.9338 * x4 + 0.004731 * x1 * x3
        - 0.3578 * x2 * x3
        - 0.3571;
    let eq2 = 0.2638 * x1 - x7 - 0.07745 * x2 - 0.6734 * x4 + 0.2238 * x1 * x3 + 0.7623 * x2 * x3
        - 0.6022;
    let eq3 = 0.3578 * x1 + 0.004731 * x2 + x6 * x8;
    let eq4 = -0.7623 * x1 + 0.2238 * x2 + 0.3461;
    let eq5 = x1 * x1 + x2 * x2 - 1.0;
    let eq6 = x3 * x3 + x4 * x4 - 1.0;
    let eq7 = x5 * x5 + x6 * x6 - 1.0;
    let eq8 = x7 * x7 + x8 * x8 - 1.0;

    /* Lower bounds ( l_i = 1 + x_i >= 0)*/

    let lb1 = l1 - 1.0 - x1;
    let lb2 = l2 - 1.0 - x2;
    let lb3 = l3 - 1.0 - x3;
    let lb4 = l4 - 1.0 - x4;
    let lb5 = l5 - 1.0 - x5;
    let lb6 = l6 - 1.0 - x6;
    let lb7 = l7 - 1.0 - x7;
    let lb8 = l8 - 1.0 - x8;

    /* Upper bounds ( u_i = 1 - x_i >= 0)*/

    let ub1 = u1 - 1.0 + x1;
    let ub2 = u2 - 1.0 + x2;
    let ub3 = u3 - 1.0 + x3;
    let ub4 = u4 - 1.0 + x4;
    let ub5 = u5 - 1.0 + x5;
    let ub6 = u6 - 1.0 + x6;
    let ub7 = u7 - 1.0 + x7;
    let ub8 = u8 - 1.0 + x8;

    fd[0] = eq1;
    fd[8] = lb1;
    fd[16] = ub1;
    fd[1] = eq2;
    fd[9] = lb2;
    fd[17] = ub2;
    fd[2] = eq3;
    fd[10] = lb3;
    fd[18] = ub3;
    fd[3] = eq4;
    fd[11] = lb4;
    fd[19] = ub4;
    fd[4] = eq5;
    fd[12] = lb5;
    fd[20] = ub5;
    fd[5] = eq6;
    fd[13] = lb6;
    fd[21] = ub6;
    fd[6] = eq7;
    fd[14] = lb7;
    fd[22] = ub7;
    fd[7] = eq8;
    fd[15] = lb8;
    fd[23] = ub8;

    0
}

/*
 * System Jacobian
 */

fn jac(
    y: &N_Vector,
    _f: &N_Vector,
    J: &SUNMatrix,
    _user_data: &mut Option<Box<dyn Any>>,
    _tmp1: &N_Vector,
    _tmp2: &N_Vector,
) -> i32 {
    let mut i: sunindextype;
    let (x1, x2, x3, x4, x5, x6, x7, x8);
    {
        let yd = N_VGetArrayPointer(y).expect("N_VGetArrayPointer");

        x1 = yd[0];
        x2 = yd[1];
        x3 = yd[2];
        x4 = yd[3];
        x5 = yd[4];
        x6 = yd[5];
        x7 = yd[6];
        x8 = yd[7];
    }

    /* Nonlinear equations */

    /*
       - 0.1238*x1 + x7 - 0.001637*x2
       - 0.9338*x4 + 0.004731*x1*x3 - 0.3578*x2*x3 - 0.3571
    */
    IJth_set(J, 1, 1, -0.1238 + 0.004731 * x3);
    IJth_set(J, 1, 2, -0.001637 - 0.3578 * x3);
    IJth_set(J, 1, 3, 0.004731 * x1 - 0.3578 * x2);
    IJth_set(J, 1, 4, -0.9338);
    IJth_set(J, 1, 7, 1.0);

    /*
      0.2638*x1 - x7 - 0.07745*x2
      - 0.6734*x4 + 0.2238*x1*x3 + 0.7623*x2*x3 - 0.6022
    */
    IJth_set(J, 2, 1, 0.2638 + 0.2238 * x3);
    IJth_set(J, 2, 2, -0.07745 + 0.7623 * x3);
    IJth_set(J, 2, 3, 0.2238 * x1 + 0.7623 * x2);
    IJth_set(J, 2, 4, -0.6734);
    IJth_set(J, 2, 7, -1.0);

    /*
      0.3578*x1 + 0.004731*x2 + x6*x8
    */
    IJth_set(J, 3, 1, 0.3578);
    IJth_set(J, 3, 2, 0.004731);
    IJth_set(J, 3, 6, x8);
    IJth_set(J, 3, 8, x6);

    /*
      - 0.7623*x1 + 0.2238*x2 + 0.3461
    */
    IJth_set(J, 4, 1, -0.7623);
    IJth_set(J, 4, 2, 0.2238);

    /*
      x1*x1 + x2*x2 - 1
    */
    IJth_set(J, 5, 1, 2.0 * x1);
    IJth_set(J, 5, 2, 2.0 * x2);

    /*
      x3*x3 + x4*x4 - 1
    */
    IJth_set(J, 6, 3, 2.0 * x3);
    IJth_set(J, 6, 4, 2.0 * x4);

    /*
      x5*x5 + x6*x6 - 1
    */
    IJth_set(J, 7, 5, 2.0 * x5);
    IJth_set(J, 7, 6, 2.0 * x6);

    /*
      x7*x7 + x8*x8 - 1
    */
    IJth_set(J, 8, 7, 2.0 * x7);
    IJth_set(J, 8, 8, 2.0 * x8);

    /*
     Lower bounds ( l_i = 1 + x_i >= 0)
     l_i - 1.0 - x_i
    */

    i = 1;
    while i <= 8 {
        IJth_set(J, 8 + i, i, -1.0);
        IJth_set(J, 8 + i, 8 + i, 1.0);
        i += 1;
    }

    /*
     Upper bounds ( u_i = 1 - x_i >= 0)
     u_i - 1.0 + x_i
    */

    i = 1;
    while i <= 8 {
        IJth_set(J, 16 + i, i, 1.0);
        IJth_set(J, 16 + i, 16 + i, 1.0);
        i += 1;
    }

    0
}

/*
 * Print solution
 */

fn PrintOutput(y: &N_Vector) {
    print!("     l=x+1          x         u=1-x\n");
    print!("   ----------------------------------\n");

    let mut i: i32 = 1;
    while i <= NVAR {
        /* C: printf(" %10.6g   %10.6g   %10.6g\n", Ith(y, i + NVAR), Ith(y, i),
        Ith(y, i + 2 * NVAR)) */
        print!(
            " {}   {}   {}\n",
            fmt_gw(Ith(y, i + NVAR), 10, 6),
            fmt_gw(Ith(y, i), 10, 6),
            fmt_gw(Ith(y, i + 2 * NVAR), 10, 6)
        );
        i += 1;
    }
}

/*
 * Check function return value...
 *    opt == 0 means SUNDIALS function allocates memory so check if
 *             returned NULL pointer
 *    opt == 1 means SUNDIALS function returns a retval so check if
 *             retval >= 0
 *    opt == 2 means function allocates memory so check if returned
 *             NULL pointer
 *
 * The C void-pointer/opt polymorphism splits into two typed helpers with
 * identical message text (`opt == 2` is unused by this example).
 */

fn check_retval_null<T>(retvalvalue: &Option<T>, funcname: &str) -> i32 {
    /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
    if retvalvalue.is_none() {
        eprint!(
            "\nSUNDIALS_ERROR: {}() failed - returned NULL pointer\n\n",
            funcname
        );
        return 1;
    }
    0
}

fn check_retval(retval: i32, funcname: &str) -> i32 {
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
