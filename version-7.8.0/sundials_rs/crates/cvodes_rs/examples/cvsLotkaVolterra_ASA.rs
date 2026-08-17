#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

/* -----------------------------------------------------------------------------
 * Rust port of examples/cvodes/serial/cvsLotkaVolterra_ASA.c
 * -----------------------------------------------------------------------------
 * This example solves the Lotka-Volterra ODE with four parameters,
 *
 *    u' = [dx/dt] = [  p_0*x - p_1*x*y  ]
 *         [dy/dt]   [ -p_2*y + p_3*x*y ].
 *
 * The initial condition is u(t_0) = 1.0 and we use the parameters
 * p  = [1.5, 1.0, 3.0, 1.0]. The integration interval is t \in [0, 10.].
 * The implicit BDF method from CVODES is used to solve the forward problem.
 * Afterwards, the continuous adjoint sensitivity analysis capabilities of CVODES
 * are used to obtain the gradient of the cost function,
 *
 *    g(u(t_f), p) = || 1 - u(t_f, p) ||^2 / 2
 *
 * with respect to the initial condition and the parameters.
 * -----------------------------------------------------------------------------
 */

use cvodes_rs::prelude::*;

use std::any::Any;

/* Problem Constants */
const NEQ: sunindextype = 2; /* number of equations  */
const NP: sunindextype = 4; /* number of params     */
const T0: sunrealtype = 0.0; /* initial time         */
const TF: sunrealtype = 10.0; /* final time           */
const RTOL: sunrealtype = 1.0e-10; /* relative tolerance   */
const ATOL: sunrealtype = 1.0e-14; /* absolute tolerance   */
const STEPS: i64 = 5; /* checkpoint interval  */

/* C keeps a file-scope `sunrealtype params[4]` and hands its address to both
CVodeSetUserData and CVodeSetUserDataB. Boxes cannot alias in safe Rust, so
`Params` is a Copy array and an identical copy is handed to each; the array
is never written after initialization, so this is observationally
identical. */
type Params = [sunrealtype; NP as usize];

const params: Params = [1.5, 1.0, 3.0, 1.0];

fn main() {
    let mut t: sunrealtype = 0.0;
    let mut which: i32 = 0;

    let mut sunctx: Option<SUNContext> = None;
    let _ = SUNContext_Create(SUN_COMM_NULL, &mut sunctx);
    let ctx = sunctx.as_ref().expect("SUNContext_Create").clone();

    /* Allocate memory for the solution vector */
    let u = N_VNew_Serial(NEQ, &ctx);
    if check_retval_ptr(&u, "N_VNew_Serial") != 0 {
        std::process::exit(1);
    }
    let u = u.expect("N_VNew_Serial");

    /* Initialize the solution vector */
    N_VConst(1.0, &u);

    /* Set the tolerances */
    let reltol = RTOL;
    let abstol = ATOL;

    /* Create the CVODES object */
    let cvode_mem = CVodeCreate(CV_BDF, &ctx);
    if check_retval_ptr(&cvode_mem, "CVodeCreate") != 0 {
        std::process::exit(1);
    }
    let cvode_mem = cvode_mem.expect("CVodeCreate");

    /* Initialize the CVODES solver */
    let retval = CVodeInit(&cvode_mem, lotka_volterra, T0, &u);
    if check_retval_int(retval, "CVodeInit") != 0 {
        std::process::exit(1);
    }

    /* Set the user data */
    let retval = CVodeSetUserData(&cvode_mem, Some(Box::new(params)));
    if check_retval_int(retval, "CVodeSetUserData") != 0 {
        std::process::exit(1);
    }

    /* Set the tolerances */
    let retval = CVodeSStolerances(&cvode_mem, reltol, abstol);
    if check_retval_int(retval, "CVodeSStolerances") != 0 {
        std::process::exit(1);
    }

    let LS = SUNLinSol_SPGMR(&u, SUN_PREC_NONE, 3, &ctx).expect("SUNLinSol_SPGMR");

    let retval = CVodeSetLinearSolver(&cvode_mem, &LS, None);
    if check_retval_int(retval, "CVodeSetLinearSolver") != 0 {
        std::process::exit(1);
    }

    let retval = CVodeSetMaxNumSteps(&cvode_mem, 100000);
    if check_retval_int(retval, "CVodeSetMaxNumSteps") != 0 {
        std::process::exit(1);
    }

    /* Initialize ASA */
    let retval = CVodeAdjInit(&cvode_mem, STEPS, CV_HERMITE);
    if check_retval_int(retval, "CVodeAdjInit") != 0 {
        std::process::exit(1);
    }

    /* Integrate the ODE */
    let tout = TF;
    let mut ncheck: i32 = 0;
    let retval = CVodeF(&cvode_mem, tout, &u, &mut t, CV_NORMAL, &mut ncheck);
    if check_retval_int(retval, "CVode") != 0 {
        std::process::exit(1);
    }

    /* Print the final solution */
    print!("Forward Solution at t = {}:\n", fmt_g(t, 6));
    N_VPrint(&u);

    /* Allocate memory for the adjoint solution vector */
    let uB = N_VNew_Serial(NEQ, &ctx);
    if check_retval_ptr(&uB, "N_VNew_Serial") != 0 {
        std::process::exit(1);
    }
    let uB = uB.expect("N_VNew_Serial");

    /* Allocate memory for the quadrature equations and initialize it to zero */
    let qB = N_VNew_Serial(NP, &ctx).expect("N_VNew_Serial");
    N_VConst(0.0, &qB);

    /* Initialize the adjoint solution vector */
    dgdu(&u, &uB);

    print!("Adjoint terminal condition:\n");
    N_VPrint(&uB);
    N_VPrint(&qB);

    /* Create the CVODES object for the backward problem */
    /* NOTE: the C source assigns the return value to `retval` and never
    reads it back before the next assignment. */
    let _ = CVodeCreateB(&cvode_mem, CV_BDF, &mut which);

    /* Initialize the CVODES solver for the backward problem */
    let retval = CVodeInitB(&cvode_mem, which, adjoint_rhs, TF, &uB);
    if check_retval_int(retval, "CVodeInitB") != 0 {
        std::process::exit(1);
    }

    /* Set the user data for the backward problem */
    let retval = CVodeSetUserDataB(&cvode_mem, which, Some(Box::new(params)));
    if check_retval_int(retval, "CVodeSetUserDataB") != 0 {
        std::process::exit(1);
    }

    /* Set the tolerances for the backward problem */
    let retval = CVodeSStolerancesB(&cvode_mem, which, reltol, abstol);
    if check_retval_int(retval, "CVodeSStolerancesB") != 0 {
        std::process::exit(1);
    }

    /* Create the linear solver for the backward problem */
    let LSB = SUNLinSol_SPGMR(&uB, SUN_PREC_NONE, 3, &ctx).expect("SUNLinSol_SPGMR");

    let retval = CVodeSetLinearSolverB(&cvode_mem, which, &LSB, None);
    if check_retval_int(retval, "CVodeSetLinearSolver") != 0 {
        std::process::exit(1);
    }

    /* Call CVodeQuadInitB to allocate internal memory and initialize backward
    quadrature integration. This gives the sensitivities w.r.t. the parameters. */
    let retval = CVodeQuadInitB(&cvode_mem, which, quad_rhs, &qB);
    if check_retval_int(retval, "CVodeQuadInitB") != 0 {
        std::process::exit(1);
    }

    /* Call CVodeSetQuadErrCon to specify whether or not the quadrature variables
    are to be used in the step size control mechanism within CVODES. Call
    CVodeQuadSStolerances or CVodeQuadSVtolerances to specify the integration
    tolerances for the quadrature variables. */
    let retval = CVodeSetQuadErrConB(&cvode_mem, which, true);
    if check_retval_int(retval, "CVodeSetQuadErrConB") != 0 {
        std::process::exit(1);
    }

    /* Call CVodeQuadSStolerancesB to specify the scalar relative and absolute tolerances
    for the backward problem. */
    let retval = CVodeQuadSStolerancesB(&cvode_mem, which, reltol, abstol);
    if check_retval_int(retval, "CVodeQuadSStolerancesB") != 0 {
        std::process::exit(1);
    }

    /* Integrate the adjoint ODE */
    let retval = CVodeB(&cvode_mem, T0, CV_NORMAL);
    if check_retval_int(retval, "CVodeB") != 0 {
        std::process::exit(1);
    }

    /* Get the final adjoint solution */
    let retval = CVodeGetB(&cvode_mem, which, &mut t, &uB);
    if check_retval_int(retval, "CVodeGetB") != 0 {
        std::process::exit(1);
    }

    /* Call CVodeGetQuadB to get the quadrature solution vector after a
    successful return from CVodeB. */
    let retval = CVodeGetQuadB(&cvode_mem, which, &mut t, &qB);
    if check_retval_int(retval, "CVodeGetQuadB") != 0 {
        std::process::exit(1);
    }

    /* dg/dp = -qB */
    N_VScale(-1.0, &qB, &qB);

    /* Print the final adjoint solution */
    print!("Adjoint Solution at t = {}:\n", fmt_g(t, 6));
    N_VPrint(&uB);
    N_VPrint(&qB);

    /* Free memory */
    N_VDestroy(u);
    N_VDestroy(uB);
    N_VDestroy(qB);
    SUNLinSolFree(Some(LS));
    SUNLinSolFree(Some(LSB));
    let mut cvode_mem = Some(cvode_mem);
    CVodeFree(&mut cvode_mem);
    SUNContext_Free(&mut sunctx);
}

/* Function to compute the ODE right-hand side */
fn lotka_volterra(
    _t: sunrealtype,
    uvec: &N_Vector,
    udotvec: &N_Vector,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    let p = *user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<Params>())
        .expect("user_data is Params");
    let u = N_VGetArrayPointer(uvec).expect("N_VGetArrayPointer");
    let mut udot = N_VGetArrayPointer(udotvec).expect("N_VGetArrayPointer");

    udot[0] = p[0] * u[0] - p[1] * u[0] * u[1];
    udot[1] = -p[2] * u[1] + p[3] * u[0] * u[1];

    0
}

/* Function to compute v^T (df/du) */
fn vjp(
    vvec: &N_Vector,
    Jvvec: &N_Vector,
    _t: sunrealtype,
    uvec: &N_Vector,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    let p = *user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<Params>())
        .expect("user_data is Params");
    let u = N_VGetArrayPointer(uvec).expect("N_VGetArrayPointer");
    let v = N_VGetArrayPointer(vvec).expect("N_VGetArrayPointer");
    let mut Jv = N_VGetArrayPointer(Jvvec).expect("N_VGetArrayPointer");

    Jv[0] = (p[0] - p[1] * u[1]) * v[0] + p[3] * u[1] * v[1];
    Jv[1] = -p[1] * u[0] * v[0] + (-p[2] + p[3] * u[0]) * v[1];

    0
}

/* Function to compute v^T (df/dp) */
fn parameter_vjp(
    vvec: &N_Vector,
    Jvvec: &N_Vector,
    _t: sunrealtype,
    uvec: &N_Vector,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    /* C: `if (user_data != params) { return -1; }` — a pointer identity test
    against the file-scope parameter array. The port checks that the user data
    really is the `Params` array instead (safe Rust has no address to
    compare). */
    if user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<Params>())
        .is_none()
    {
        return -1;
    }

    let u = N_VGetArrayPointer(uvec).expect("N_VGetArrayPointer");
    let v = N_VGetArrayPointer(vvec).expect("N_VGetArrayPointer");
    let mut Jv = N_VGetArrayPointer(Jvvec).expect("N_VGetArrayPointer");

    Jv[0] = u[0] * v[0];
    Jv[1] = -u[0] * u[1] * v[0];
    Jv[2] = -u[1] * v[1];
    Jv[3] = u[0] * u[1] * v[1];

    0
}

/* Gradient of the cost function w.r.t to u.
The gradient w.r.t to p is zero since the cost function
does not depend on the parameters. */
fn dgdu(uvec: &N_Vector, dgvec: &N_Vector) {
    let u = N_VGetArrayPointer(uvec).expect("N_VGetArrayPointer");
    let mut dg = N_VGetArrayPointer(dgvec).expect("N_VGetArrayPointer");

    dg[0] = -1.0 + u[0];
    dg[1] = -1.0 + u[1];
}

/* Function to compute the adjoint ODE right-hand side:
   -mu^T (df/du)
*/
fn adjoint_rhs(
    t: sunrealtype,
    uvec: &N_Vector,
    lvec: &N_Vector,
    ldotvec: &N_Vector,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    vjp(lvec, ldotvec, t, uvec, user_data);
    N_VScale(-1.0, ldotvec, ldotvec);
    0
}

/* Function to compute the quadrature right-hand side:
   mu^T (df/dp)
*/
fn quad_rhs(
    t: sunrealtype,
    uvec: &N_Vector,
    muvec: &N_Vector,
    qBdotvec: &N_Vector,
    user_dataB: &mut Option<Box<dyn Any>>,
) -> i32 {
    parameter_vjp(muvec, qBdotvec, t, uvec, user_dataB);
    0
}

/* Check function return value
 *    C opt == 0 means SUNDIALS function allocates memory so check if
 *              returned NULL pointer (check_retval_ptr)
 *    C opt == 1 means SUNDIALS function returns an integer value so check
 *              if retval < 0 (check_retval_int) */

fn check_retval_ptr<T>(retval_ptr: &Option<T>, funcname: &str) -> i32 {
    /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
    if retval_ptr.is_none() {
        eprint!(
            "\nSUNDIALS_ERROR: {}() failed - returned NULL pointer\n\n",
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
