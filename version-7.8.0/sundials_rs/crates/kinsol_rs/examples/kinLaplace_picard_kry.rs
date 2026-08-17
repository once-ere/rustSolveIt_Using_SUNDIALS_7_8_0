/* -----------------------------------------------------------------
 * Programmer(s): David J. Gardner @ LLNL
 * -----------------------------------------------------------------
 * Based on kinLaplace_picard_bnd.c by Carol S. Woodward @ LLNL
 * -----------------------------------------------------------------
 * Ported from: examples/kinsol/serial/kinLaplace_picard_kry.c
 * -----------------------------------------------------------------
 * This example solves a 2D elliptic PDE
 *
 *    d^2 u / dx^2 + d^2 u / dy^2 = u^3 - u - 2.0
 *
 * subject to homogeneous Dirichlet boundary conditions.
 * The PDE is discretized on a uniform NX+2 by NY+2 grid with
 * central differencing, and with boundary values eliminated,
 * leaving a system of size NEQ = NX*NY.
 * The nonlinear system is solved by KINSOL using the Picard
 * iteration and the SPGMR linear solver.
 * -----------------------------------------------------------------
 */

#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

use std::any::Any;

use kinsol_rs::prelude::*;

/* Problem Constants */

const NX: sunindextype = 31; /* no. of points in x direction */
const NY: sunindextype = 31; /* no. of points in y direction */
const NEQ: sunindextype = NX * NY; /* problem dimension */

const SKIP: sunindextype = 3; /* no. of points skipped for printing */

/* function tolerance */
const FTOL: sunrealtype = 1.0e-12;

const ZERO: sunrealtype = 0.0;
const ONE: sunrealtype = 1.0;
const TWO: sunrealtype = 2.0;

/* IJth is defined in order to isolate the translation from the
   mathematical 2-dimensional structure of the dependent variable vector
   to the underlying 1-dimensional storage.
   IJth(vdata,i,j) references the element in the vdata array for
   u at mesh point (i,j), where 1 <= i <= NX, 1 <= j <= NY.
   The vdata array is obtained via the call vdata = N_VGetArrayPointer(v),
   where v is an N_Vector.
   The variables are ordered by the y index j, then by the x index i. */

fn IJth(i: sunindextype, j: sunindextype) -> usize {
    ((j - 1) + (i - 1) * NY) as usize
}

/*
 *--------------------------------------------------------------------
 * MAIN PROGRAM
 *--------------------------------------------------------------------
 */

fn main() {
    let fnormtol: sunrealtype;
    let mut fnorm: sunrealtype = 0.0;
    let mut retval: i32;

    /* -------------------------
     * Print problem description
     * ------------------------- */

    print!("\n2D elliptic PDE on unit square\n");
    print!("   d^2 u / dx^2 + d^2 u / dy^2 = u^3 - u + 2.0\n");
    print!(" + homogeneous Dirichlet boundary conditions\n\n");
    print!("Solution method: Anderson accelerated Picard iteration with SPGMR linear solver.\n");
    print!("Problem size: {:>2} x {:>2} = {:>4}\n", NX, NY, NEQ);

    /* Create the SUNDIALS context that all SUNDIALS objects require */
    let mut sunctx: Option<SUNContext> = None;
    retval = SUNContext_Create(SUN_COMM_NULL, &mut sunctx);
    if check_retval_int(retval, "SUNContext_Create") != 0 {
        std::process::exit(1);
    }
    let sunctx_h = sunctx.as_ref().expect("SUNContext_Create").clone();

    /* --------------------------------------
     * Create vectors for solution and scales
     * -------------------------------------- */

    let y = N_VNew_Serial(NEQ, &sunctx_h);
    if check_retval_ptr(&y, "N_VNew_Serial", 0) != 0 {
        std::process::exit(1);
    }
    let y = y.expect("N_VNew_Serial");

    let scale = N_VNew_Serial(NEQ, &sunctx_h);
    if check_retval_ptr(&scale, "N_VNew_Serial", 0) != 0 {
        std::process::exit(1);
    }
    let scale = scale.expect("N_VNew_Serial");

    /* ----------------------------------------------------------------------------------
     * Initialize and allocate memory for KINSOL, set parameters for Anderson acceleration
     * ---------------------------------------------------------------------------------- */

    let kmem = KINCreate(&sunctx_h);
    if check_retval_ptr(&kmem, "KINCreate", 0) != 0 {
        std::process::exit(1);
    }
    let kmem = kmem.expect("KINCreate");

    /* y is used as a template */

    retval = KINInit(&kmem, func, &y);
    if check_retval_int(retval, "KINInit") != 0 {
        std::process::exit(1);
    }

    /* -------------------
     * Set optional inputs
     * ------------------- */

    /* Use acceleration with up to 3 prior residuals */

    retval = KINSetMAA(&kmem, 3);
    if check_retval_int(retval, "KINSetMAA") != 0 {
        std::process::exit(1);
    }

    /* Specify stopping tolerance based on residual */

    fnormtol = FTOL;
    retval = KINSetFuncNormTol(&kmem, fnormtol);
    if check_retval_int(retval, "KINSetFuncNormTol") != 0 {
        std::process::exit(1);
    }

    /* ----------------------
     * Create SUNLinearSolver
     * ---------------------- */

    let LS = SUNLinSol_SPGMR(&y, SUN_PREC_NONE, 10, &sunctx_h);
    if check_retval_ptr(&LS, "SUNLinSol_SPGMR", 0) != 0 {
        std::process::exit(1);
    }
    let LS = LS.expect("SUNLinSol_SPGMR");

    /* --------------------
     * Attach linear solver
     * -------------------- */

    retval = KINSetLinearSolver(&kmem, &LS, None);
    if check_retval_int(retval, "KINSetLinearSolver") != 0 {
        std::process::exit(1);
    }

    /* ------------------------------------
     * Set Jacobian vector product function
     * ------------------------------------ */

    retval = KINSetJacTimesVecFn(&kmem, Some(jactimes));
    if check_retval_int(retval, "KINSetJacTimesVecFn") != 0 {
        std::process::exit(1);
    }

    /* -------------
     * Initial guess
     * ------------- */

    N_VConst(ZERO, &y);
    {
        let mut ydata = N_VGetArrayPointer(&y).expect("N_VGetArrayPointer");
        ydata[IJth(2, 2)] = ONE;
    }

    /* ----------------------------
     * Call KINSol to solve problem
     * ---------------------------- */

    /* No scaling used */
    N_VConst(ONE, &scale);

    /* Call main solver */
    retval = KINSol(
        &kmem,      /* KINSol memory block */
        &y,         /* initial guess on input; solution vector */
        KIN_PICARD, /* global strategy choice */
        &scale,     /* scaling vector, for the variable cc */
        &scale,
    ); /* scaling vector for function values fval */
    if check_retval_int(retval, "KINSol") != 0 {
        std::process::exit(1);
    }

    /* ------------------------------------
     * Print solution and solver statistics
     * ------------------------------------ */

    /* Get scaled norm of the system function */

    retval = KINGetFuncNorm(&kmem, &mut fnorm);
    if check_retval_int(retval, "KINGetfuncNorm") != 0 {
        std::process::exit(1);
    }

    print!("\nComputed solution (||F|| = {}):\n\n", fmt_g(fnorm, 6));

    PrintOutput(&y);

    PrintFinalStats(&kmem);

    /* -----------
     * Free memory
     * ----------- */

    N_VDestroy(y);
    N_VDestroy(scale);
    KINFree(&mut Some(kmem));
    let _ = SUNLinSolFree(Some(LS));
    let _ = SUNContext_Free(&mut sunctx);
}

/*
 *--------------------------------------------------------------------
 * PRIVATE FUNCTIONS
 *--------------------------------------------------------------------
 */

/*
 * System function
 */

fn func(u: &N_Vector, f: &N_Vector, _user_data: &mut Option<Box<dyn Any>>) -> i32 {
    let dx: sunrealtype;
    let dy: sunrealtype;
    let mut hdiff: sunrealtype;
    let mut vdiff: sunrealtype;
    let hdc: sunrealtype;
    let vdc: sunrealtype;
    let mut uij: sunrealtype;
    let mut udn: sunrealtype;
    let mut uup: sunrealtype;
    let mut ult: sunrealtype;
    let mut urt: sunrealtype;

    dx = ONE / ((NX + 1) as sunrealtype);
    dy = ONE / ((NY + 1) as sunrealtype);
    hdc = ONE / (dx * dx);
    vdc = ONE / (dy * dy);

    let udata = N_VGetArrayPointer(u).expect("N_VGetArrayPointer");
    let mut fdata = N_VGetArrayPointer(f).expect("N_VGetArrayPointer");

    for j in 1..=NY {
        for i in 1..=NX {
            /* Extract u at x_i, y_j and four neighboring points */

            uij = udata[IJth(i, j)];
            udn = if j == 1 { ZERO } else { udata[IJth(i, j - 1)] };
            uup = if j == NY { ZERO } else { udata[IJth(i, j + 1)] };
            ult = if i == 1 { ZERO } else { udata[IJth(i - 1, j)] };
            urt = if i == NX { ZERO } else { udata[IJth(i + 1, j)] };

            /* Evaluate diffusion components */

            hdiff = hdc * (ult - TWO * uij + urt);
            vdiff = vdc * (uup - TWO * uij + udn);

            /* Set residual at x_i, y_j */

            fdata[IJth(i, j)] = hdiff + vdiff + uij - uij * uij * uij + 2.0;
        }
    }

    0
}

/*
 * Jacobian vector product function
 */

fn jactimes(
    v: &N_Vector,
    Jv: &N_Vector,
    _u: &N_Vector,
    _new_u: &mut sunbooleantype,
    _user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    let dx: sunrealtype;
    let dy: sunrealtype;
    let mut hdiff: sunrealtype;
    let mut vdiff: sunrealtype;
    let hdc: sunrealtype;
    let vdc: sunrealtype;
    let mut vij: sunrealtype;
    let mut vdn: sunrealtype;
    let mut vup: sunrealtype;
    let mut vlt: sunrealtype;
    let mut vrt: sunrealtype;

    dx = ONE / ((NX + 1) as sunrealtype);
    dy = ONE / ((NY + 1) as sunrealtype);
    hdc = ONE / (dx * dx);
    vdc = ONE / (dy * dy);

    let vdata = N_VGetArrayPointer(v).expect("N_VGetArrayPointer");
    let mut Jvdata = N_VGetArrayPointer(Jv).expect("N_VGetArrayPointer");

    for j in 1..=NY {
        for i in 1..=NX {
            /* Extract v at x_i, y_j and four neighboring points */

            vij = vdata[IJth(i, j)];
            vdn = if j == 1 { ZERO } else { vdata[IJth(i, j - 1)] };
            vup = if j == NY { ZERO } else { vdata[IJth(i, j + 1)] };
            vlt = if i == 1 { ZERO } else { vdata[IJth(i - 1, j)] };
            vrt = if i == NX { ZERO } else { vdata[IJth(i + 1, j)] };

            /* Evaluate diffusion components */

            hdiff = hdc * (vlt - TWO * vij + vrt);
            vdiff = vdc * (vup - TWO * vij + vdn);

            /* Set Jv at x_i, y_j */

            Jvdata[IJth(i, j)] = hdiff + vdiff;
        }
    }

    0
}

/*
 * Print solution at selected points
 */

fn PrintOutput(u: &N_Vector) {
    let dx: sunrealtype;
    let dy: sunrealtype;
    let mut x: sunrealtype;
    let mut y: sunrealtype;

    dx = ONE / ((NX + 1) as sunrealtype);
    dy = ONE / ((NY + 1) as sunrealtype);

    let udata = N_VGetArrayPointer(u).expect("N_VGetArrayPointer");

    print!("            ");
    for i in (1..=NX).step_by(SKIP as usize) {
        x = (i as sunrealtype) * dx;
        print!("{:<8} ", fmt_f(x, 5));
    }
    print!("\n\n");

    for j in (1..=NY).step_by(SKIP as usize) {
        y = (j as sunrealtype) * dy;
        print!("{:<8}    ", fmt_f(y, 5));
        for i in (1..=NX).step_by(SKIP as usize) {
            print!("{:<8} ", fmt_f(udata[IJth(i, j)], 5));
        }
        print!("\n");
    }
}

/*
 * Print final statistics
 */

fn PrintFinalStats(kmem: &KINMem) {
    let mut nni: i64 = 0;
    let mut nfe: i64 = 0;
    let mut nli: i64 = 0;
    let mut npe: i64 = 0;
    let mut nps: i64 = 0;
    let mut ncfl: i64 = 0;
    let mut nfeLS: i64 = 0;
    let mut njvevals: i64 = 0;
    let mut retval: i32;

    /* Main solver statistics */

    retval = KINGetNumNonlinSolvIters(kmem, &mut nni);
    check_retval_int(retval, "KINGetNumNonlinSolvIters");
    retval = KINGetNumFuncEvals(kmem, &mut nfe);
    check_retval_int(retval, "KINGetNumFuncEvals");

    /* Linear solver statistics */

    retval = KINGetNumLinIters(kmem, &mut nli);
    check_retval_int(retval, "KINGetNumLinIters");
    retval = KINGetNumLinFuncEvals(kmem, &mut nfeLS);
    check_retval_int(retval, "KINGetNumLinFuncEvals");
    retval = KINGetNumLinConvFails(kmem, &mut ncfl);
    check_retval_int(retval, "KINGetNumLinConvFails");
    retval = KINGetNumJtimesEvals(kmem, &mut njvevals);
    check_retval_int(retval, "KINGetNumJtimesEvals");
    retval = KINGetNumPrecEvals(kmem, &mut npe);
    check_retval_int(retval, "KINGetNumPrecEvals");
    retval = KINGetNumPrecSolves(kmem, &mut nps);
    check_retval_int(retval, "KINGetNumPrecSolves");

    print!("\nFinal Statistics.. \n\n");
    print!("nni = {:>6}  nli   = {:>6}  ncfl = {:>6}\n", nni, nli, ncfl);
    print!(
        "nfe = {:>6}  nfeLS = {:>6}  njt  = {:>6}\n",
        nfe, nfeLS, njvevals
    );
    print!("npe = {:>6}  nps   = {:>6}", npe, nps);
}

/*
 * Check function return value...
 *    opt == 0 means SUNDIALS function allocates memory so check if
 *             returned NULL pointer
 *    opt == 1 means SUNDIALS function returns a retval so check if
 *             retval >= 0 (see check_retval_int)
 *    opt == 2 means function allocates memory so check if returned
 *             NULL pointer
 */

fn check_retval_ptr<T>(retvalvalue: &Option<T>, funcname: &str, opt: i32) -> i32 {
    /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
    if opt == 0 && retvalvalue.is_none() {
        eprint!(
            "\nSUNDIALS_ERROR: {}() failed - returned NULL pointer\n\n",
            funcname
        );
        return 1;
    }
    /* Check if function returned NULL pointer - no memory allocated */
    else if opt == 2 && retvalvalue.is_none() {
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
