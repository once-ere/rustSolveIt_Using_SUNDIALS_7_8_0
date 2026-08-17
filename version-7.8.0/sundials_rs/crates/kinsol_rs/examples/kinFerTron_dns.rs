//! Port of `examples/kinsol/serial/kinFerTron_dns.c`.
//!
//! Example (serial):
//!
//! This example solves a nonlinear system from.
//!
//! Source: "Handbook of Test Problems in Local and Global Optimization",
//!             C.A. Floudas, P.M. Pardalos et al.
//!             Kluwer Academic Publishers, 1999.
//! Test problem 4 from Section 14.1, Chapter 14: Ferraris and Tronconi
//!
//! This problem involves a blend of trigonometric and exponential terms.
//!    0.5 sin(x1 x2) - 0.25 x2/pi - 0.5 x1 = 0
//!    (1-0.25/pi) ( exp(2 x1)-e ) + e x2 / pi - 2 e x1 = 0
//! such that
//!    0.25 <= x1 <=1.0
//!    1.5 <= x2 <= 2 pi
//!
//! The treatment of the bound constraints on x1 and x2 is done using
//! the additional variables
//!    l1 = x1 - x1_min >= 0
//!    L1 = x1 - x1_max <= 0
//!    l2 = x2 - x2_min >= 0
//!    L2 = x2 - x2_max >= 0
//!
//! and using the constraint feature in KINSOL to impose
//!    l1 >= 0    l2 >= 0
//!    L1 <= 0    L2 <= 0
//!
//! The Ferraris-Tronconi test problem has two known solutions.
//! The nonlinear system is solved by KINSOL using different
//! combinations of globalization and Jacobian update strategies
//! and with different initial guesses (leading to one or the other
//! of the known solutions).
//!
//! Constraints are imposed to make all components of the solution
//! positive.

#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

use std::any::Any;

use kinsol_rs::prelude::*;

/* Problem Constants */

const NVAR: usize = 2;
const NEQ: sunindextype = 3 * NVAR as sunindextype;

const FTOL: sunrealtype = 1.0e-5; /* function tolerance */
const STOL: sunrealtype = 1.0e-5; /* step tolerance     */

const ZERO: sunrealtype = 0.0;
const PT25: sunrealtype = 0.25;
const PT5: sunrealtype = 0.5;
const ONE: sunrealtype = 1.0;
const ONEPT5: sunrealtype = 1.5;
const TWO: sunrealtype = 2.0;

const PI: sunrealtype = 3.1415926;
const E: sunrealtype = 2.7182818;

/* C `typedef struct { sunrealtype lb[NVAR]; sunrealtype ub[NVAR]; }* UserData` */
#[derive(Clone, Copy)]
struct UserData {
    lb: [sunrealtype; NVAR],
    ub: [sunrealtype; NVAR],
}

/*
 *--------------------------------------------------------------------
 * MAIN PROGRAM
 *--------------------------------------------------------------------
 */

fn main() {
    /* Reusable return flag */
    let mut retval: i32;

    /* Create the SUNDIALS context that all SUNDIALS objects require */
    let mut sunctx: Option<SUNContext> = None;
    retval = SUNContext_Create(SUN_COMM_NULL, &mut sunctx);
    if check_retval(retval, "SUNContext_Create") != 0 {
        std::process::exit(1);
    }
    let ctx = sunctx.clone().unwrap();

    /* User data */
    let mut data = UserData {
        lb: [ZERO; NVAR],
        ub: [ZERO; NVAR],
    };
    data.lb[0] = PT25;
    data.ub[0] = ONE;
    data.lb[1] = ONEPT5;
    data.ub[1] = TWO * PI;

    /* Create serial vectors of length NEQ */
    let u1 = N_VNew_Serial(NEQ, &ctx);
    if check_retval_null(&u1, "N_VNew_Serial") != 0 {
        std::process::exit(1);
    }
    let u1 = u1.unwrap();

    let u2 = N_VNew_Serial(NEQ, &ctx);
    if check_retval_null(&u2, "N_VNew_Serial") != 0 {
        std::process::exit(1);
    }
    let u2 = u2.unwrap();

    let u = N_VNew_Serial(NEQ, &ctx);
    if check_retval_null(&u, "N_VNew_Serial") != 0 {
        std::process::exit(1);
    }
    let u = u.unwrap();

    let s = N_VNew_Serial(NEQ, &ctx);
    if check_retval_null(&s, "N_VNew_Serial") != 0 {
        std::process::exit(1);
    }
    let s = s.unwrap();

    let c = N_VNew_Serial(NEQ, &ctx);
    if check_retval_null(&c, "N_VNew_Serial") != 0 {
        std::process::exit(1);
    }
    let c = c.unwrap();

    SetInitialGuess1(&u1, &data);
    SetInitialGuess2(&u2, &data);

    N_VConst(ONE, &s); /* no scaling */

    {
        let mut cdata = N_VGetArrayPointer(&c).expect("N_VGetArrayPointer");
        cdata[0] = ZERO; /* no constraint on x1 */
        cdata[1] = ZERO; /* no constraint on x2 */
        cdata[2] = ONE; /* l1 = x1 - x1_min >= 0 */
        cdata[3] = -ONE; /* L1 = x1 - x1_max <= 0 */
        cdata[4] = ONE; /* l2 = x2 - x2_min >= 0 */
        cdata[5] = -ONE; /* L2 = x2 - x22_min <= 0 */
    }

    let fnormtol: sunrealtype = FTOL; /* residual tolerance    */
    let scsteptol: sunrealtype = STOL; /* scaled step tolerance */

    let mut kmem = KINCreate(&ctx);
    if check_retval_null(&kmem, "KINCreate") != 0 {
        std::process::exit(1);
    }
    let kin = kmem.clone().unwrap();

    retval = KINSetUserData(&kin, Some(Box::new(data)));
    if check_retval(retval, "KINSetUserData") != 0 {
        std::process::exit(1);
    }

    retval = KINSetConstraints(&kin, Some(&c));
    if check_retval(retval, "KINSetConstraints") != 0 {
        std::process::exit(1);
    }

    retval = KINSetFuncNormTol(&kin, fnormtol);
    if check_retval(retval, "KINSetFuncNormTol") != 0 {
        std::process::exit(1);
    }

    retval = KINSetScaledStepTol(&kin, scsteptol);
    if check_retval(retval, "KINSetScaledStepTol") != 0 {
        std::process::exit(1);
    }

    retval = KINInit(&kin, func, &u);
    if check_retval(retval, "KINInit") != 0 {
        std::process::exit(1);
    }

    /* Create dense SUNMatrix */
    let J = SUNDenseMatrix(NEQ, NEQ, &ctx);
    if check_retval_null(&J, "SUNDenseMatrix") != 0 {
        std::process::exit(1);
    }
    let J = J.unwrap();

    /* Create dense SUNLinearSolver object */
    let LS = SUNLinSol_Dense(&u, &J, &ctx);
    if check_retval_null(&LS, "SUNLinSol_Dense") != 0 {
        std::process::exit(1);
    }
    let LS = LS.unwrap();

    /* Attach the matrix and linear solver to KINSOL */
    retval = KINSetLinearSolver(&kin, &LS, Some(&J));
    if check_retval(retval, "KINSetLinearSolver") != 0 {
        std::process::exit(1);
    }

    /* Print out the problem size, solution parameters, initial guess. */
    PrintHeader(fnormtol, scsteptol);

    /* --------------------------- */

    let mut glstr: i32; /* KINSOL globalization strategy flag */
    let mut mset: i32; /* KINSOL method selection flag */

    print!("\n------------------------------------------\n");
    print!("\nInitial guess on lower bounds\n");
    print!("  [x1,x2] = ");
    PrintOutput(&u1);

    N_VScale(ONE, &u1, &u);
    glstr = KIN_NONE;
    mset = 1;
    SolveIt(&kin, &u, &s, glstr, mset);

    /* --------------------------- */

    N_VScale(ONE, &u1, &u);
    glstr = KIN_LINESEARCH;
    mset = 1;
    SolveIt(&kin, &u, &s, glstr, mset);

    /* --------------------------- */

    N_VScale(ONE, &u1, &u);
    glstr = KIN_NONE;
    mset = 0;
    SolveIt(&kin, &u, &s, glstr, mset);

    /* --------------------------- */

    N_VScale(ONE, &u1, &u);
    glstr = KIN_LINESEARCH;
    mset = 0;
    SolveIt(&kin, &u, &s, glstr, mset);

    /* --------------------------- */

    print!("\n------------------------------------------\n");
    print!("\nInitial guess in middle of feasible region\n");
    print!("  [x1,x2] = ");
    PrintOutput(&u2);

    N_VScale(ONE, &u2, &u);
    glstr = KIN_NONE;
    mset = 1;
    SolveIt(&kin, &u, &s, glstr, mset);

    /* --------------------------- */

    N_VScale(ONE, &u2, &u);
    glstr = KIN_LINESEARCH;
    mset = 1;
    SolveIt(&kin, &u, &s, glstr, mset);

    /* --------------------------- */

    N_VScale(ONE, &u2, &u);
    glstr = KIN_NONE;
    mset = 0;
    SolveIt(&kin, &u, &s, glstr, mset);

    /* --------------------------- */

    N_VScale(ONE, &u2, &u);
    glstr = KIN_LINESEARCH;
    mset = 0;
    SolveIt(&kin, &u, &s, glstr, mset);

    /* Free memory */

    N_VDestroy(u1);
    N_VDestroy(u2);
    N_VDestroy(u);
    N_VDestroy(s);
    N_VDestroy(c);
    KINFree(&mut kmem);
    let _ = SUNLinSolFree(Some(LS));
    SUNMatDestroy(J);
    /* `free(data)` — the boxed copy is owned by the KINSOL memory. */
    let _ = SUNContext_Free(&mut sunctx);
}

fn SolveIt(kmem: &KINMem, u: &N_Vector, s: &N_Vector, glstr: i32, mset: i32) -> i32 {
    let mut retval: i32;

    print!("\n");

    if mset == 1 {
        print!("Exact Newton");
    } else {
        print!("Modified Newton");
    }

    if glstr == KIN_NONE {
        print!("\n");
    } else {
        print!(" with line search\n");
    }

    retval = KINSetMaxSetupCalls(kmem, mset as i64);
    if check_retval(retval, "KINSetMaxSetupCalls") != 0 {
        return 1;
    }

    retval = KINSol(kmem, u, glstr, s, s);
    if check_retval(retval, "KINSol") != 0 {
        return 1;
    }

    print!("Solution:\n  [x1,x2] = ");
    PrintOutput(u);

    PrintFinalStats(kmem);

    0
}

/*
 *--------------------------------------------------------------------
 * FUNCTIONS CALLED BY KINSOL
 *--------------------------------------------------------------------
 */

/*
 * System function for predator-prey system
 */

fn func(u: &N_Vector, f: &N_Vector, user_data: &mut Option<Box<dyn Any>>) -> i32 {
    let data = user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<UserData>())
        .expect("user_data is UserData");
    let lb = data.lb;
    let ub = data.ub;

    let udata = N_VGetArrayPointer(u).expect("N_VGetArrayPointer");
    let mut fdata = N_VGetArrayPointer(f).expect("N_VGetArrayPointer");

    let x1 = udata[0];
    let x2 = udata[1];
    let l1 = udata[2];
    let L1 = udata[3];
    let l2 = udata[4];
    let L2 = udata[5];

    fdata[0] = PT5 * (x1 * x2).sun_sin() - PT25 * x2 / PI - PT5 * x1;
    fdata[1] = (ONE - PT25 / PI) * ((TWO * x1).sun_exp() - E) + E * x2 / PI - TWO * E * x1;
    fdata[2] = l1 - x1 + lb[0];
    fdata[3] = L1 - x1 + ub[0];
    fdata[4] = l2 - x2 + lb[1];
    fdata[5] = L2 - x2 + ub[1];

    0
}

/*
 *--------------------------------------------------------------------
 * PRIVATE FUNCTIONS
 *--------------------------------------------------------------------
 */

/*
 * Initial guesses
 */

fn SetInitialGuess1(u: &N_Vector, data: &UserData) {
    let mut udata = N_VGetArrayPointer(u).expect("N_VGetArrayPointer");

    let lb = data.lb;
    let ub = data.ub;

    /* There are two known solutions for this problem */

    /* this init. guess should take us to (0.29945; 2.83693) */
    let x1 = lb[0];
    let x2 = lb[1];

    udata[0] = x1;
    udata[1] = x2;
    udata[2] = x1 - lb[0];
    udata[3] = x1 - ub[0];
    udata[4] = x2 - lb[1];
    udata[5] = x2 - ub[1];
}

fn SetInitialGuess2(u: &N_Vector, data: &UserData) {
    let mut udata = N_VGetArrayPointer(u).expect("N_VGetArrayPointer");

    let lb = data.lb;
    let ub = data.ub;

    /* There are two known solutions for this problem */

    /* this init. guess should take us to (0.5; 3.1415926) */
    let x1 = PT5 * (lb[0] + ub[0]);
    let x2 = PT5 * (lb[1] + ub[1]);

    udata[0] = x1;
    udata[1] = x2;
    udata[2] = x1 - lb[0];
    udata[3] = x1 - ub[0];
    udata[4] = x2 - lb[1];
    udata[5] = x2 - ub[1];
}

/*
 * Print first lines of output (problem description)
 */

fn PrintHeader(fnormtol: sunrealtype, scsteptol: sunrealtype) {
    print!("\nFerraris and Tronconi test problem\n");
    print!("Tolerance parameters:\n");
    /* C: printf("  fnormtol  = %10.6g\n  scsteptol = %10.6g\n", ...) */
    print!(
        "  fnormtol  = {}\n  scsteptol = {}\n",
        fmt_gw(fnormtol, 10, 6),
        fmt_gw(scsteptol, 10, 6)
    );
}

/*
 * Print solution
 */

fn PrintOutput(u: &N_Vector) {
    let udata = N_VGetArrayPointer(u).expect("N_VGetArrayPointer");
    /* C: printf(" %8.6g  %8.6g\n", udata[0], udata[1]) */
    print!(" {}  {}\n", fmt_gw(udata[0], 8, 6), fmt_gw(udata[1], 8, 6));
}

/*
 * Print final statistics contained in iopt
 */

fn PrintFinalStats(kmem: &KINMem) {
    let mut nni: i64 = 0;
    let mut nfe: i64 = 0;
    let mut nje: i64 = 0;
    let mut nfeD: i64 = 0;
    let mut retval: i32;

    retval = KINGetNumNonlinSolvIters(kmem, &mut nni);
    check_retval(retval, "KINGetNumNonlinSolvIters");
    retval = KINGetNumFuncEvals(kmem, &mut nfe);
    check_retval(retval, "KINGetNumFuncEvals");
    retval = KINGetNumJacEvals(kmem, &mut nje);
    check_retval(retval, "KINGetNumJacEvals");
    retval = KINGetNumLinFuncEvals(kmem, &mut nfeD);
    check_retval(retval, "KINGetNumLinFuncEvals");

    print!("Final Statistics:\n");
    /* C: printf("  nni = %5ld    nfe  = %5ld \n", nni, nfe) */
    print!("  nni = {:>5}    nfe  = {:>5} \n", nni, nfe);
    /* C: printf("  nje = %5ld    nfeD = %5ld \n", nje, nfeD) */
    print!("  nje = {:>5}    nfeD = {:>5} \n", nje, nfeD);
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
