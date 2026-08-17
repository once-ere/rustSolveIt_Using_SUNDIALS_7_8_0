//! Port of `examples/kinsol/serial/kinAnalytic_fp.c`.
//!
//! This example solves the nonlinear system
//!
//! 3x - cos((y-1)z) - 1/2 = 0
//! x^2 - 81(y-0.9)^2 + sin(z) + 1.06 = 0
//! exp(-x(y-1)) + 20z + (10 pi - 3)/3 = 0
//!
//! using the accelerated fixed pointer solver in KINSOL. The nonlinear fixed
//! point function is
//!
//! g1(x,y,z) = 1/3 cos((y-1)yz) + 1/6
//! g2(x,y,z) = 1/9 sqrt(x^2 + sin(z) + 1.06) + 0.9
//! g3(x,y,z) = -1/20 exp(-x(y-1)) - (10 pi - 3) / 60
//!
//! This system has the analytic solution x = 1/2, y = 1, z = -pi/6.

#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

use std::any::Any;

use kinsol_rs::prelude::*;

/* problem constants */
const NEQ: sunindextype = 3; /* number of equations */

const ZERO: sunrealtype = 0.0; /* real 0.0  */
const PTONE: sunrealtype = 0.1; /* real 0.1  */
const HALF: sunrealtype = 0.5; /* real 0.5  */
const PTNINE: sunrealtype = 0.9; /* real 0.9  */
const ONE: sunrealtype = 1.0; /* real 1.0  */
const ONEPTZEROSIX: sunrealtype = 1.06; /* real 1.06 */
const THREE: sunrealtype = 3.0; /* real 3.0  */
const SIX: sunrealtype = 6.0; /* real 6.0  */
const NINE: sunrealtype = 9.0; /* real 9.0  */
const TEN: sunrealtype = 10.0; /* real 10.0 */
const TWENTY: sunrealtype = 20.0; /* real 20.0 */
const SIXTY: sunrealtype = 60.0; /* real 60.0 */
const PI: sunrealtype = 3.1415926535898; /* real pi   */

/* analytic solution */
const XTRUE: sunrealtype = HALF;
const YTRUE: sunrealtype = ONE;
const ZTRUE: sunrealtype = -PI / SIX;

/* problem options (C: `struct { ... }* UserOpt`) */
struct UserOptRec {
    tol: sunrealtype,               /* solve tolerance                  */
    maxiter: i64,                   /* max number of iterations         */
    m_aa: i64,                      /* number of acceleration vectors   */
    delay_aa: i64,                  /* number of iterations to delay AA */
    orth_aa: i32,                   /* orthogonalization method         */
    damping_fp: sunrealtype,        /* damping parameter for FP         */
    damping_aa: sunrealtype,        /* damping parameter for AA         */
    use_damping_fn: sunbooleantype, /* damping function                 */
    use_depth_fn: sunbooleantype,   /* depth function                   */
}

/* -----------------------------------------------------------------------------
 * Main program
 * ---------------------------------------------------------------------------*/
fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let argc: i32 = argv.len() as i32;

    let nni: i64; /* solver outputs      */
    let nfe: i64;

    /* Set default options */
    let mut uopt = UserOptRec {
        tol: ZERO,
        maxiter: 0,
        m_aa: 0,
        delay_aa: 0,
        orth_aa: 0,
        damping_fp: ZERO,
        damping_aa: ZERO,
        use_damping_fn: SUNFALSE,
        use_depth_fn: SUNFALSE,
    };
    let retval = SetDefaults(&mut uopt);
    if check_retval(retval, "SetDefaults") != 0 {
        std::process::exit(1);
    }

    let retval = ReadInputs(argc, &argv, &mut uopt);
    if check_retval(retval, "ReadInputs") != 0 {
        std::process::exit(1);
    }

    /* -------------------------
     * Print problem description
     * ------------------------- */

    print!("Solve the nonlinear system:\n");
    print!("    3x - cos((y-1)z) - 1/2 = 0\n");
    print!("    x^2 - 81(y-0.9)^2 + sin(z) + 1.06 = 0\n");
    print!("    exp(-x(y-1)) + 20z + (10 pi - 3)/3 = 0\n");
    print!("Analytic solution:\n");
    print!("    x = {}\n", fmt_g(XTRUE, 6));
    print!("    y = {}\n", fmt_g(YTRUE, 6));
    print!("    z = {}\n", fmt_g(ZTRUE, 6));
    print!("Solution method: Anderson accelerated fixed point iteration.\n");
    print!("    tolerance    = {}\n", fmt_g(uopt.tol, 6));
    print!("    max iters    = {}\n", uopt.maxiter);
    print!("    m_aa         = {}\n", uopt.m_aa);
    print!("    delay_aa     = {}\n", uopt.delay_aa);
    print!("    damping_aa   = {}\n", fmt_g(uopt.damping_aa, 6));
    print!("    damping_fp   = {}\n", fmt_g(uopt.damping_fp, 6));
    if uopt.use_damping_fn {
        print!("    damping_fn   = ON\n");
    } else {
        print!("    damping_fn   = OFF\n");
    }
    if uopt.use_depth_fn {
        print!("    depth_fn     = ON\n");
    } else {
        print!("    depth_fn     = OFF\n");
    }
    print!("    orth routine = {}\n", uopt.orth_aa);

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

    let u = N_VNew_Serial(NEQ, &ctx);
    if check_retval_null(&u, "N_VNew_Serial") != 0 {
        std::process::exit(1);
    }
    let u = u.expect("N_VNew_Serial");

    let scale = N_VClone(&u);
    if check_retval_null(&scale, "N_VClone") != 0 {
        std::process::exit(1);
    }
    let scale = scale.expect("N_VClone");

    /* -----------------------------------------
     * Initialize and allocate memory for KINSOL
     * ----------------------------------------- */

    let mut kmem = KINCreate(&ctx);
    if check_retval_null(&kmem, "KINCreate") != 0 {
        std::process::exit(1);
    }
    let kin = kmem.clone().expect("KINCreate");

    /* Set number of prior residuals used in Anderson acceleration */
    let _retval = KINSetMAA(&kin, uopt.m_aa);

    /* Set orthogonalization routine used in Anderson acceleration */
    let retval = KINSetOrthAA(&kin, uopt.orth_aa);
    if check_retval(retval, "KINSetOrthAA") != 0 {
        std::process::exit(1);
    }

    let retval = KINInit(&kin, FPFunction, &u);
    if check_retval(retval, "KINInit") != 0 {
        std::process::exit(1);
    }

    /* -------------------
     * Set optional inputs
     * ------------------- */

    /* Specify stopping tolerance based on residual */
    let retval = KINSetFuncNormTol(&kin, uopt.tol);
    if check_retval(retval, "KINSetFuncNormTol") != 0 {
        std::process::exit(1);
    }

    /* Set maximum number of iterations */
    let retval = KINSetNumMaxIters(&kin, uopt.maxiter);
    if check_retval(retval, "KINSetNumMaxItersFuncNormTol") != 0 {
        std::process::exit(1);
    }

    /* Set Fixed point damping parameter */
    if uopt.m_aa == 0 {
        let _retval = KINSetDamping(&kin, uopt.damping_fp);
    }

    /* Set Anderson acceleration options */
    if uopt.m_aa > 0 {
        /* Set damping parameter */
        let retval = KINSetDampingAA(&kin, uopt.damping_aa);
        if check_retval(retval, "KINSetDampingAA") != 0 {
            std::process::exit(1);
        }

        /* Set acceleration delay */
        let retval = KINSetDelayAA(&kin, uopt.delay_aa);
        if check_retval(retval, "KINSetDelayAA") != 0 {
            std::process::exit(1);
        }
    }

    if uopt.use_damping_fn {
        /* Attach user defined damping function */
        let retval = KINSetDampingFn(&kin, Some(DampingFn));
        if check_retval(retval, "KINSetDampingFn") != 0 {
            std::process::exit(1);
        }
    }

    if uopt.use_depth_fn {
        /* Attach user defined depth function */
        let retval = KINSetDepthFn(&kin, Some(DepthFn));
        if check_retval(retval, "KINSetDepthFn") != 0 {
            std::process::exit(1);
        }
    }

    /* -------------
     * Initial guess
     * ------------- */

    /* Get vector data array */
    {
        let data = N_VGetArrayPointer(&u);
        if check_retval_null(&data, "N_VGetArrayPointer") != 0 {
            std::process::exit(1);
        }
        let mut data = data.expect("N_VGetArrayPointer");

        data[0] = PTONE;
        data[1] = PTONE;
        data[2] = -PTONE;
    }

    /* ----------------------------
     * Call KINSol to solve problem
     * ---------------------------- */

    /* No scaling used */
    N_VConst(ONE, &scale);

    /* Call main solver */
    let retval = KINSol(
        &kin,   /* KINSol memory block */
        &u,     /* initial guess on input; solution vector */
        KIN_FP, /* global strategy choice */
        &scale, /* scaling vector, for the variable cc */
        &scale,
    ); /* scaling vector for function values fval */
    if check_retval(retval, "KINSol") != 0 {
        std::process::exit(1);
    }

    /* ------------------------------------
     * Get solver statistics
     * ------------------------------------ */

    /* get solver stats */
    let mut nni_out: i64 = 0;
    let retval = KINGetNumNonlinSolvIters(&kin, &mut nni_out);
    let _ = check_retval(retval, "KINGetNumNonlinSolvIters");
    nni = nni_out;

    let mut nfe_out: i64 = 0;
    let retval = KINGetNumFuncEvals(&kin, &mut nfe_out);
    let _ = check_retval(retval, "KINGetNumFuncEvals");
    nfe = nfe_out;

    print!("\nFinal Statistics:\n");
    print!("Number of nonlinear iterations: {:>6}\n", nni);
    print!("Number of function evaluations: {:>6}\n", nfe);

    /* ------------------------------------
     * Print solution and check error
     * ------------------------------------ */

    /* check solution */
    let retval = check_ans(&u, uopt.tol);

    /* -----------
     * Free memory
     * ----------- */

    N_VDestroy(u);
    N_VDestroy(scale);
    KINFree(&mut kmem);
    let _ = SUNContext_Free(&mut sunctx);

    if retval != 0 {
        std::process::exit(retval);
    }
}

/* -----------------------------------------------------------------------------
 * Nonlinear system
 *
 * 3x - cos((y-1)z) - 1/2 = 0
 * x^2 - 81(y-0.9)^2 + sin(z) + 1.06 = 0
 * exp(-x(y-1)) + 20z + (10 pi - 3)/3 = 0
 *
 * Nonlinear fixed point function
 *
 * g1(x,y,z) = 1/3 cos((y-1)z) + 1/6
 * g2(x,y,z) = 1/9 sqrt(x^2 + sin(z) + 1.06) + 0.9
 * g3(x,y,z) = -1/20 exp(-x(y-1)) - (10 pi - 3) / 60
 *
 * ---------------------------------------------------------------------------*/
fn FPFunction(u: &N_Vector, g: &N_Vector, _user_data: &mut Option<Box<dyn Any>>) -> i32 {
    /* Get vector data arrays */
    let udata = N_VGetArrayPointer(u);
    if check_retval_null(&udata, "N_VGetArrayPointer") != 0 {
        return -1;
    }
    let udata = udata.expect("N_VGetArrayPointer");

    let gdata = N_VGetArrayPointer(g);
    if check_retval_null(&gdata, "N_VGetArrayPointer") != 0 {
        return -1;
    }
    let mut gdata = gdata.expect("N_VGetArrayPointer");

    let x = udata[0];
    let y = udata[1];
    let z = udata[2];

    gdata[0] = (ONE / THREE) * ((y - ONE) * z).sun_cos() + (ONE / SIX);
    gdata[1] = (ONE / NINE) * (x * x + z.sun_sin() + ONEPTZEROSIX).sqrt() + PTNINE;
    gdata[2] = -(ONE / TWENTY) * (-x * (y - ONE)).sun_exp() - (TEN * PI - THREE) / SIXTY;

    0
}

fn DampingFn(
    _iter: i64,
    u_val: &N_Vector,
    g_val: &N_Vector,
    qt_fn: Option<&mut [sunrealtype]>,
    depth: i64,
    _user_data: &mut Option<Box<dyn Any>>,
    damping_factor: &mut sunrealtype,
) -> i32 {
    if depth == 0 {
        *damping_factor = 0.5;
    } else {
        let qt_fn = qt_fn.expect("qt_fn");

        /* Compute ||Q^T fn||^2 */
        let mut qt_fn_norm_sqr: sunrealtype = ZERO;
        for i in 0..depth {
            qt_fn_norm_sqr += qt_fn[i as usize] * qt_fn[i as usize];
        }

        /* Compute ||fn||^2 = ||G(u_n) - u_n||^2 */
        let g_data = N_VGetArrayPointer(g_val).expect("N_VGetArrayPointer");
        let u_data = N_VGetArrayPointer(u_val).expect("N_VGetArrayPointer");
        let mut fn_: [sunrealtype; 3] = [ZERO; 3];
        for i in 0..3 {
            fn_[i] = g_data[i] - u_data[i];
        }
        let mut fn_norm_sqr: sunrealtype = ZERO;
        for i in 0..3 {
            fn_norm_sqr += fn_[i] * fn_[i];
        }

        /* Compute the gain = sqrt(1 - ||Q^T fn||^2 / ||fn||^2) */
        let gain = SUNRsqrt(ONE - qt_fn_norm_sqr / fn_norm_sqr);

        *damping_factor = 0.9 - 0.5 * gain;
    }

    0
}

fn DepthFn(
    iter: i64,
    _u_val: &N_Vector,
    _g_val: &N_Vector,
    _f_val: &N_Vector,
    _df: &[N_Vector],
    _R_mat: &mut [sunrealtype],
    depth: i64,
    _user_data: &mut Option<Box<dyn Any>>,
    new_depth: &mut i64,
    _remove_index: Option<&mut [sunbooleantype]>,
) -> i32 {
    if iter < 2 {
        *new_depth = 1;
    } else {
        *new_depth = depth;
    }

    0
}

/* -----------------------------------------------------------------------------
 * Check the solution of the nonlinear system and return PASS or FAIL
 * ---------------------------------------------------------------------------*/
fn check_ans(u: &N_Vector, tol: sunrealtype) -> i32 {
    /* Get vector data array */
    let data = N_VGetArrayPointer(u);
    if check_retval_null(&data, "N_VGetArrayPointer") != 0 {
        return 1;
    }
    let data = data.expect("N_VGetArrayPointer");

    /* print the solution */
    print!("Computed solution:\n");
    print!("    x = {}\n", fmt_g(data[0], 6));
    print!("    y = {}\n", fmt_g(data[1], 6));
    print!("    z = {}\n", fmt_g(data[2], 6));

    /* solution error */
    let ex = (data[0] - XTRUE).abs();
    let ey = (data[1] - YTRUE).abs();
    let ez = (data[2] - ZTRUE).abs();

    /* print the solution error */
    print!("Solution error:\n");
    print!("    ex = {}\n", fmt_g(ex, 6));
    print!("    ey = {}\n", fmt_g(ey, 6));
    print!("    ez = {}\n", fmt_g(ez, 6));

    let tol = tol * TEN;
    if ex > tol || ey > tol || ez > tol {
        print!("FAIL\n");
        return 1;
    }

    print!("PASS\n");
    0
}

/* -----------------------------------------------------------------------------
 * Set default options
 * ---------------------------------------------------------------------------*/
fn SetDefaults(uopt: &mut UserOptRec) -> i32 {
    /* Set default options values */
    uopt.tol = 100.0 * SUNRsqrt(SUN_UNIT_ROUNDOFF);
    uopt.maxiter = 30;
    uopt.m_aa = 0; /* no acceleration */
    uopt.delay_aa = 0; /* no delay        */
    uopt.orth_aa = 0; /* MGS             */
    uopt.damping_fp = 1.0; /* no FP dampig    */
    uopt.damping_aa = 1.0; /* no AA damping   */
    uopt.use_damping_fn = SUNFALSE; /* no damping fn   */
    uopt.use_depth_fn = SUNFALSE; /* no depth fn     */

    0
}

/* -----------------------------------------------------------------------------
 * Read command line inputs
 * ---------------------------------------------------------------------------*/
fn ReadInputs(argc: i32, argv: &[String], uopt: &mut UserOptRec) -> i32 {
    let mut arg_index: usize = 1;

    while (arg_index as i32) < argc {
        if argv[arg_index] == "--tol" {
            arg_index += 1;
            uopt.tol = SUNStrToReal(&argv[arg_index]);
            arg_index += 1;
        } else if argv[arg_index] == "--maxiter" {
            arg_index += 1;
            uopt.maxiter = atoi(&argv[arg_index]) as i64;
            arg_index += 1;
        } else if argv[arg_index] == "--m_aa" {
            arg_index += 1;
            uopt.m_aa = atoi(&argv[arg_index]) as i64;
            arg_index += 1;
        } else if argv[arg_index] == "--delay_aa" {
            arg_index += 1;
            uopt.delay_aa = atoi(&argv[arg_index]) as i64;
            arg_index += 1;
        } else if argv[arg_index] == "--damping_fp" {
            arg_index += 1;
            uopt.damping_fp = SUNStrToReal(&argv[arg_index]);
            arg_index += 1;
        } else if argv[arg_index] == "--damping_aa" {
            arg_index += 1;
            uopt.damping_aa = SUNStrToReal(&argv[arg_index]);
            arg_index += 1;
        } else if argv[arg_index] == "--damping_fn" {
            arg_index += 1;
            uopt.use_damping_fn = SUNTRUE;
        } else if argv[arg_index] == "--depth_fn" {
            arg_index += 1;
            uopt.use_depth_fn = SUNTRUE;
        } else if argv[arg_index] == "--orth_aa" {
            arg_index += 1;
            uopt.orth_aa = atoi(&argv[arg_index]);
            arg_index += 1;
        } else if argv[arg_index] == "--help" {
            InputHelp();
            return -1;
        } else {
            print!(
                "Error: Invalid command line parameter {}\n",
                argv[arg_index]
            );
            InputHelp();
            return -1;
        }
    }

    0
}

/* -----------------------------------------------------------------------------
 * Print command line options
 * ---------------------------------------------------------------------------*/
fn InputHelp() {
    print!("\n");
    print!(" Command line options:\n");
    print!("   --tol        : nonlinear solver tolerance\n");
    print!("   --maxiter    : max number of nonlinear iterations\n");
    print!("   --m_aa       : number of Anderson acceleration vectors\n");
    print!("   --delay_aa   : Anderson acceleration delay\n");
    print!("   --damping_fp : fixed point damping parameter\n");
    print!("   --damping_aa : Anderson acceleration damping parameter\n");
    print!("   --orth_aa    : Anderson acceleration orthogonalization method\n");
    print!("   --damping_fn : user defined damping function\n");
    print!("   --depth_fn   : user defined depth function\n");
}

/* -----------------------------------------------------------------------------
 * Check function return value (C check_retval; the C void-pointer/opt
 * polymorphism splits into two typed helpers with identical messages:
 *   check_retval_null = opt == 0 (NULL-pointer check)
 *   check_retval      = opt == 1 (non-zero return check)
 * ---------------------------------------------------------------------------*/

fn check_retval_null<T>(returnvalue: &Option<T>, funcname: &str) -> i32 {
    if returnvalue.is_none() {
        eprint!("\nERROR: {}() failed -- returned NULL\n\n", funcname);
        return 1;
    }
    0
}

fn check_retval(retval: i32, funcname: &str) -> i32 {
    if retval != 0 {
        eprint!("\nERROR: {}() failed -- returned {}\n\n", funcname, retval);
        return 1;
    }
    0
}
