//! Port of `examples/kinsol/serial/kinRoberts_fp.c`.
//!
//! Example problem:
//!
//! The following is a simple example problem, with the coding
//! needed for its solution by the accelerated fixed point solver in
//! KINSOL.
//! The problem is from chemical kinetics, and consists of solving
//! the first time step in a Backward Euler solution for the
//! following three rate equations:
//!    dy1/dt = -.04*y1 + 1.e4*y2*y3
//!    dy2/dt = .04*y1 - 1.e4*y2*y3 - 3.e2*(y2)^2
//!    dy3/dt = 3.e2*(y2)^2
//! on the interval from t = 0.0 to t = 0.1, with initial
//! conditions: y1 = 1.0, y2 = y3 = 0. The problem is stiff.
//! Run statistics (optional outputs) are printed at the end.

#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

use std::any::Any;

use kinsol_rs::prelude::*;

/* Problem Constants */

const NEQ: sunindextype = 3; /* number of equations  */
const Y10: sunrealtype = 1.0; /* initial y components */
const Y20: sunrealtype = 0.0;
const Y30: sunrealtype = 0.0;
const TOL: sunrealtype = 1.0e-10; /* function tolerance */
const DSTEP: sunrealtype = 0.1; /* Size of the single time step used */

const PRIORS: i64 = 2;

const ZERO: sunrealtype = 0.0;
const ONE: sunrealtype = 1.0;

/* User-defined vector accessor helpers: Ith
   (C macro `Ith(v,i)` = `NV_Ith_S(v,i-1)`; i is 1-based). */

fn Ith(v: &N_Vector, i: usize) -> sunrealtype {
    N_VGetArrayPointer(v).expect("N_VGetArrayPointer")[i - 1]
}

fn Ith_set(v: &N_Vector, i: usize, x: sunrealtype) {
    N_VGetArrayPointer(v).expect("N_VGetArrayPointer")[i - 1] = x;
}

fn NV_Ith_S_set(v: &N_Vector, i: usize, x: sunrealtype) {
    N_VGetArrayPointer(v).expect("N_VGetArrayPointer")[i] = x;
}

/*
 *--------------------------------------------------------------------
 * MAIN PROGRAM
 *--------------------------------------------------------------------
 */

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let argc: i32 = argv.len() as i32;

    let mut fnorm: sunrealtype = 0.0;

    /* -------------------------
     * Print problem description
     * ------------------------- */

    print!("Example problem from chemical kinetics solving\n");
    print!("the first time step in a Backward Euler solution for the\n");
    print!("following three rate equations:\n");
    print!("    dy1/dt = -.04*y1 + 1.e4*y2*y3\n");
    print!("    dy2/dt = .04*y1 - 1.e4*y2*y3 - 3.e2*(y2)^2\n");
    print!("    dy3/dt = 3.e2*(y2)^2\n");
    print!("on the interval from t = 0.0 to t = 0.1, with initial\n");
    print!("conditions: y1 = 1.0, y2 = y3 = 0.\n");
    print!("Solution method: Anderson accelerated fixed point iteration.\n");

    /* Create the SUNDIALS context that all SUNDIALS objects require */
    let mut sunctx: Option<SUNContext> = None;
    let retval = SUNContext_Create(SUN_COMM_NULL, &mut sunctx);
    if check_retval(retval, "SUNContext_Create") != 0 {
        std::process::exit(1);
    }
    let ctx = sunctx.clone().expect("SUNContext_Create");

    /* --------------------------------------
     * Create vectors for solution and scales
     * -------------------------------------- */

    let y = N_VNew_Serial(NEQ, &ctx);
    if check_retval_null(&y, "N_VNew_Serial") != 0 {
        std::process::exit(1);
    }
    let y = y.expect("N_VNew_Serial");

    let scale = N_VNew_Serial(NEQ, &ctx);
    if check_retval_null(&scale, "N_VNew_Serial") != 0 {
        std::process::exit(1);
    }
    let scale = scale.expect("N_VNew_Serial");

    /* -----------------------------------------
     * Initialize and allocate memory for KINSOL
     * ----------------------------------------- */

    let mut kmem = KINCreate(&ctx);
    if check_retval_null(&kmem, "KINCreate") != 0 {
        std::process::exit(1);
    }
    let kin = kmem.clone().expect("KINCreate");

    /* y is used as a template */

    /* Set number of prior residuals used in Anderson acceleration */
    let _retval = KINSetMAA(&kin, PRIORS);

    let retval = KINInit(&kin, funcRoberts, &y);
    if check_retval(retval, "KINInit") != 0 {
        std::process::exit(1);
    }

    /* -------------------
     * Set optional inputs
     * ------------------- */

    /* Specify stopping tolerance based on residual */

    let fnormtol = TOL;
    let retval = KINSetFuncNormTol(&kin, fnormtol);
    if check_retval(retval, "KINSetFuncNormTol") != 0 {
        std::process::exit(1);
    }

    /* Override any current settings with command-line options */

    let retval = KINSetOptions(&kin, None, None, argc, &argv);
    if check_retval(retval, "KINSetOptions") != 0 {
        std::process::exit(1);
    }

    /* -------------
     * Initial guess
     * ------------- */

    N_VConst(ZERO, &y);
    Ith_set(&y, 1, ONE);

    /* ----------------------------
     * Call KINSol to solve problem
     * ---------------------------- */

    /* No scaling used */
    N_VConst(ONE, &scale);

    /* Call main solver */
    let retval = KINSol(
        &kin,   /* KINSol memory block */
        &y,     /* initial guess on input; solution vector */
        KIN_FP, /* global strategy choice */
        &scale, /* scaling vector, for the variable cc */
        &scale,
    ); /* scaling vector for function values fval */
    if check_retval(retval, "KINSol") != 0 {
        std::process::exit(1);
    }

    /* ------------------------------------
     * Print solution and solver statistics
     * ------------------------------------ */

    /* Get scaled norm of the system function */

    let retval = KINGetFuncNorm(&kin, &mut fnorm);
    if check_retval(retval, "KINGetfuncNorm") != 0 {
        std::process::exit(1);
    }

    print!("\nComputed solution (||F|| = {}):\n\n", fmt_g(fnorm, 6));
    PrintOutput(&y);

    PrintFinalStats(&kin);

    /* check the solution error */
    let retval = check_ans(&y, 1e-4, 1e-6);

    /* -----------
     * Free memory
     * ----------- */

    N_VDestroy(y);
    N_VDestroy(scale);
    KINFree(&mut kmem);
    let _ = SUNContext_Free(&mut sunctx);

    if retval != 0 {
        std::process::exit(retval);
    }
}

/*
 *--------------------------------------------------------------------
 * PRIVATE FUNCTIONS
 *--------------------------------------------------------------------
 */

/*
 * System function
 */

fn funcRoberts(y: &N_Vector, g: &N_Vector, _user_data: &mut Option<Box<dyn Any>>) -> i32 {
    let y1 = Ith(y, 1);
    let y2 = Ith(y, 2);
    let y3 = Ith(y, 3);

    let yd1 = DSTEP * (-0.04 * y1 + 1.0e4 * y2 * y3);
    let yd3 = DSTEP * 3.0e2 * y2 * y2;

    Ith_set(g, 1, yd1 + Y10);
    Ith_set(g, 2, -yd1 - yd3 + Y20);
    Ith_set(g, 3, yd3 + Y30);

    0
}

/*
 * Print solution at selected points
 */

fn PrintOutput(y: &N_Vector) {
    let y1 = Ith(y, 1);
    let y2 = Ith(y, 2);
    let y3 = Ith(y, 3);

    /* C: printf("y =%14.6e  %14.6e  %14.6e\n", y1, y2, y3) */
    print!(
        "y ={}  {}  {}\n",
        fmt_ew(y1, 14, 6),
        fmt_ew(y2, 14, 6),
        fmt_ew(y3, 14, 6)
    );
}

/*
 * Print final statistics
 */

fn PrintFinalStats(kmem: &KINMem) {
    /* Main solver statistics */

    let mut nni: i64 = 0;
    let retval = KINGetNumNonlinSolvIters(kmem, &mut nni);
    let _ = check_retval(retval, "KINGetNumNonlinSolvIters");
    let mut nfe: i64 = 0;
    let retval = KINGetNumFuncEvals(kmem, &mut nfe);
    let _ = check_retval(retval, "KINGetNumFuncEvals");

    print!("\nFinal Statistics.. \n\n");
    print!("nni      = {:>6}    nfe     = {:>6} \n", nni, nfe);
}

/*
 * Check function return value... (C check_retval; the C void-pointer/opt
 * polymorphism splits into two typed helpers with identical messages:
 *   check_retval_null = opt == 0 (NULL-pointer check)
 *   check_retval      = opt == 1 (retval < 0 check)
 * The opt == 2 branch ("MEMORY_ERROR") has no call site in this example.
 */

fn check_retval_null<T>(retvalvalue: &Option<T>, funcname: &str) -> i32 {
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
    if retval < 0 {
        eprint!(
            "\nSUNDIALS_ERROR: {}() failed with retval = {}\n\n",
            funcname, retval
        );
        return 1;
    }
    0
}

/* compare the solution to a reference solution computed with a
tolerance of 1e-14 */
fn check_ans(u: &N_Vector, rtol: sunrealtype, atol: sunrealtype) -> i32 {
    /* create reference solution and error weight vectors */
    let r#ref = N_VClone(u).expect("N_VClone");
    let ewt = N_VClone(u).expect("N_VClone");

    /* set the reference solution data */
    NV_Ith_S_set(&r#ref, 0, 9.9678538655358029e-01);
    NV_Ith_S_set(&r#ref, 1, 2.9530060962800345e-03);
    NV_Ith_S_set(&r#ref, 2, 2.6160735013975683e-04);

    /* compute the error weight vector */
    N_VAbs(&r#ref, &ewt);
    N_VScale(rtol, &ewt, &ewt);
    N_VAddConst(&ewt, atol, &ewt);
    if N_VMin(&ewt) <= ZERO {
        eprint!("\nSUNDIALS_ERROR: check_ans failed - ewt <= 0\n\n");
        return -1;
    }
    N_VInv(&ewt, &ewt);

    /* compute the solution error */
    N_VLinearSum(ONE, u, -ONE, &r#ref, &r#ref);
    let err = N_VWrmsNorm(&r#ref, &ewt);

    /* is the solution within the tolerances? */
    let passfail = if err < ONE { 0 } else { 1 };

    if passfail != 0 {
        print!("\nSUNDIALS_WARNING: check_ans error={}\n\n", fmt_g(err, 6));
    }

    /* Free vectors */
    N_VDestroy(r#ref);
    N_VDestroy(ewt);

    passfail
}
