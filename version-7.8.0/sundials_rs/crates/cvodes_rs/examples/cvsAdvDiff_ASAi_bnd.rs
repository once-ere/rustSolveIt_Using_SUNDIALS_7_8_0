/* -----------------------------------------------------------------
 * Programmer(s): Radu Serban @ LLNL
 * -----------------------------------------------------------------
 * Rust port of examples/cvodes/serial/cvsAdvDiff_ASAi_bnd.c
 * -----------------------------------------------------------------
 * Adjoint sensitivity example problem:
 *
 * The following is a simple example problem with a banded Jacobian,
 * with the program for its solution by CVODES.
 * The problem is the semi-discrete form of the advection-diffusion
 * equation in 2-D:
 *   du/dt = d^2 u / dx^2 + .5 du/dx + d^2 u / dy^2
 * on the rectangle 0 <= x <= 2, 0 <= y <= 1, and the time
 * interval 0 <= t <= 1. Homogeneous Dirichlet boundary conditions
 * are posed, and the initial condition is the following:
 *   u(x,y,t=0) = x(2-x)y(1-y)exp(5xy).
 * The PDE is discretized on a uniform MX+2 by MY+2 grid with
 * central differencing, and with boundary values eliminated,
 * leaving an ODE system of size NEQ = MX*MY.
 * This program solves the problem with the BDF method, Newton
 * iteration with the BAND linear solver, and a user-supplied
 * Jacobian routine.
 * It uses scalar relative and absolute tolerances.
 * Output is printed at t = .1, .2, ..., 1.
 * Run statistics (optional outputs) are printed at the end.
 *
 * Additionally, CVODES integrates backwards in time the
 * the semi-discrete form of the adjoint PDE:
 *   d(lambda)/dt = - d^2(lambda) / dx^2 + 0.5 d(lambda) / dx
 *                  - d^2(lambda) / dy^2 - 1.0
 * with homogeneous Dirichlet boundary conditions and final
 * conditions:
 *   lambda(x,y,t=t_final) = 0.0
 * whose solution at t = 0 represents the sensitivity of
 *   G = int_0^t_final int_x int _y u(t,x,y) dx dy dt
 * with respect to the initial conditions of the original problem.
 * -----------------------------------------------------------------
 */

#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

use std::any::Any;

use cvodes_rs::prelude::*;

/* Problem Constants */

const XMAX: sunrealtype = 2.0; /* domain boundaries             */
const YMAX: sunrealtype = 1.0;
const MX: sunindextype = 40; /* mesh dimensions               */
const MY: sunindextype = 20;
const NEQ: sunindextype = MX * MY; /* number of equations           */
const ATOL: sunrealtype = 1.0e-5;
const RTOLB: sunrealtype = 1.0e-6;
const T0: sunrealtype = 0.0; /* initial time                  */
const TOUT: sunrealtype = 1.0; /* final time                    */
const NSTEP: i64 = 50; /* check point saved every NSTEP */

const ZERO: sunrealtype = 0.0;
const ONE: sunrealtype = 1.0;
const TWO: sunrealtype = 2.0;

/* User-defined vector access helper IJth */

/* IJth is defined in order to isolate the translation from the
   mathematical 2-dimensional structure of the dependent variable vector
   to the underlying 1-dimensional storage.
   IJth(i,j) is the index into the vdata array for u at mesh point (i,j),
   where 1 <= i <= MX, 1 <= j <= MY.
   The variables are ordered by the y index j, then by the x index i. */

fn IJth(i: sunindextype, j: sunindextype) -> usize {
    ((j - 1) + (i - 1) * MY) as usize
}

/* Type : UserData
   contains grid constants */

/* NOTE: the C program hands the SAME `data` pointer to CVodeSetUserData and
   CVodeSetUserDataB. Boxes cannot alias in safe Rust, so `UserData` is Copy
   and an identical copy is handed to each; the struct is read-only after
   construction, so this is observationally identical. */
#[derive(Clone, Copy)]
struct UserData {
    dx: sunrealtype,
    dy: sunrealtype,
    hdcoef: sunrealtype,
    hacoef: sunrealtype,
    vdcoef: sunrealtype,
}

/*
 *--------------------------------------------------------------------
 * MAIN PROGRAM
 *--------------------------------------------------------------------
 */

fn main() {
    let reltol: sunrealtype;
    let abstol: sunrealtype;
    let mut t: sunrealtype = 0.0;

    let mut indexB: i32 = 0;

    let reltolB: sunrealtype;
    let abstolB: sunrealtype;

    let mut retval: i32;
    let mut ncheck: i32 = 0;

    /* Allocate and initialize user data memory */

    let dx = XMAX / ((MX + 1) as sunrealtype);
    let dy = YMAX / ((MY + 1) as sunrealtype);
    let data = UserData {
        dx,
        dy,
        hdcoef: ONE / (dx * dx),
        hacoef: 1.5 / (TWO * dx),
        vdcoef: ONE / (dy * dy),
    };

    /* Set the tolerances for the forward integration */
    reltol = ZERO;
    abstol = ATOL;

    /* Create the SUNDIALS simulation context that all SUNDIALS objects require */
    let mut sunctx: Option<SUNContext> = None;
    retval = SUNContext_Create(SUN_COMM_NULL, &mut sunctx);
    if check_retval_int(retval, "SUNContext_Create") != 0 {
        std::process::exit(1);
    }
    let ctx = sunctx.as_ref().expect("SUNContext_Create").clone();

    /* Allocate u vector */
    let u = N_VNew_Serial(NEQ, &ctx);
    if check_retval_ptr(&u, "N_VNew", 0) != 0 {
        std::process::exit(1);
    }
    let u = u.expect("N_VNew_Serial");

    /* Initialize u vector */
    SetIC(&u, &data);

    /* Create and allocate CVODES memory for forward run */

    print!("\nCreate and allocate CVODES memory for forward runs\n");

    let cvode_mem = CVodeCreate(CV_BDF, &ctx);
    if check_retval_ptr(&cvode_mem, "CVodeCreate", 0) != 0 {
        std::process::exit(1);
    }
    let cvode_mem = cvode_mem.expect("CVodeCreate");

    retval = CVodeSetUserData(&cvode_mem, Some(Box::new(data)));
    if check_retval_int(retval, "CVodeSetUserData") != 0 {
        std::process::exit(1);
    }

    retval = CVodeInit(&cvode_mem, f, T0, &u);
    if check_retval_int(retval, "CVodeInit") != 0 {
        std::process::exit(1);
    }

    retval = CVodeSStolerances(&cvode_mem, reltol, abstol);
    if check_retval_int(retval, "CVodeSStolerances") != 0 {
        std::process::exit(1);
    }

    /* Create banded SUNMatrix for the forward problem */
    let A = SUNBandMatrix(NEQ, MY, MY, &ctx);
    if check_retval_ptr(&A, "SUNBandMatrix", 0) != 0 {
        std::process::exit(1);
    }
    let A = A.expect("SUNBandMatrix");

    /* Create banded SUNLinearSolver for the forward problem */
    let LS = SUNLinSol_Band(&u, &A, &ctx);
    if check_retval_ptr(&LS, "SUNLinSol_Band", 0) != 0 {
        std::process::exit(1);
    }
    let LS = LS.expect("SUNLinSol_Band");

    /* Attach the matrix and linear solver */
    retval = CVodeSetLinearSolver(&cvode_mem, &LS, Some(&A));
    if check_retval_int(retval, "CVodeSetLinearSolver") != 0 {
        std::process::exit(1);
    }

    /* Set the user-supplied Jacobian routine for the forward problem */
    retval = CVodeSetJacFn(&cvode_mem, Some(Jac));
    if check_retval_int(retval, "CVodeSetJacFn") != 0 {
        std::process::exit(1);
    }

    /* Allocate global memory */

    print!("\nAllocate global memory\n");

    retval = CVodeAdjInit(&cvode_mem, NSTEP, CV_HERMITE);
    if check_retval_int(retval, "CVodeAdjInit") != 0 {
        std::process::exit(1);
    }

    /* Perform forward run */
    print!("\nForward integration\n");
    retval = CVodeF(&cvode_mem, TOUT, &u, &mut t, CV_NORMAL, &mut ncheck);
    if check_retval_int(retval, "CVodeF") != 0 {
        std::process::exit(1);
    }

    print!("\nncheck = {}\n", ncheck);

    /* Set the tolerances for the backward integration */
    reltolB = RTOLB;
    abstolB = ATOL;

    /* Allocate uB */
    let uB = N_VNew_Serial(NEQ, &ctx);
    if check_retval_ptr(&uB, "N_VNew", 0) != 0 {
        std::process::exit(1);
    }
    let uB = uB.expect("N_VNew_Serial");
    /* Initialize uB = 0 */
    N_VConst(ZERO, &uB);

    /* Create and allocate CVODES memory for backward run */

    print!("\nCreate and allocate CVODES memory for backward run\n");

    retval = CVodeCreateB(&cvode_mem, CV_BDF, &mut indexB);
    if check_retval_int(retval, "CVodeCreateB") != 0 {
        std::process::exit(1);
    }

    retval = CVodeSetUserDataB(&cvode_mem, indexB, Some(Box::new(data)));
    if check_retval_int(retval, "CVodeSetUserDataB") != 0 {
        std::process::exit(1);
    }

    retval = CVodeInitB(&cvode_mem, indexB, fB, TOUT, &uB);
    if check_retval_int(retval, "CVodeInitB") != 0 {
        std::process::exit(1);
    }

    retval = CVodeSStolerancesB(&cvode_mem, indexB, reltolB, abstolB);
    if check_retval_int(retval, "CVodeSStolerancesB") != 0 {
        std::process::exit(1);
    }

    /* Create banded SUNMatrix for the backward problem */
    let AB = SUNBandMatrix(NEQ, MY, MY, &ctx);
    if check_retval_ptr(&AB, "SUNBandMatrix", 0) != 0 {
        std::process::exit(1);
    }
    let AB = AB.expect("SUNBandMatrix");

    /* Create banded SUNLinearSolver for the backward problem */
    let LSB = SUNLinSol_Band(&uB, &AB, &ctx);
    if check_retval_ptr(&LSB, "SUNLinSol_Band", 0) != 0 {
        std::process::exit(1);
    }
    let LSB = LSB.expect("SUNLinSol_Band");

    /* Attach the matrix and linear solver */
    retval = CVodeSetLinearSolverB(&cvode_mem, indexB, &LSB, Some(&AB));
    if check_retval_int(retval, "CVodeSetLinearSolverB") != 0 {
        std::process::exit(1);
    }

    /* Set the user-supplied Jacobian routine for the backward problem */
    retval = CVodeSetJacFnB(&cvode_mem, indexB, Some(JacB));
    if check_retval_int(retval, "CVodeSetJacFnB") != 0 {
        std::process::exit(1);
    }

    /* Perform backward integration */
    print!("\nBackward integration\n");
    retval = CVodeB(&cvode_mem, T0, CV_NORMAL);
    if check_retval_int(retval, "CVodeB") != 0 {
        std::process::exit(1);
    }

    retval = CVodeGetB(&cvode_mem, indexB, &mut t, &uB);
    if check_retval_int(retval, "CVodeGetB") != 0 {
        std::process::exit(1);
    }

    PrintOutput(&uB, &data);

    N_VDestroy(u); /* Free the u vector                      */
    N_VDestroy(uB); /* Free the uB vector                     */
    let mut cvode_mem = Some(cvode_mem);
    CVodeFree(&mut cvode_mem); /* Free the CVODE problem memory          */
    SUNLinSolFree(Some(LS)); /* Free the forward linear solver memory  */
    SUNMatDestroy(A); /* Free the forward matrix memory         */
    SUNLinSolFree(Some(LSB)); /* Free the backward linear solver memory */
    SUNMatDestroy(AB); /* Free the backward matrix memory        */
    SUNContext_Free(&mut sunctx); /* Free the SUNDIALS simulation context */
}

/*
 *--------------------------------------------------------------------
 * FUNCTIONS CALLED BY CVODES
 *--------------------------------------------------------------------
 */

/*
 * f routine. right-hand side of forward ODE.
 */

fn f(_t: sunrealtype, u: &N_Vector, udot: &N_Vector, user_data: &mut Option<Box<dyn Any>>) -> i32 {
    let mut uij: sunrealtype;
    let mut udn: sunrealtype;
    let mut uup: sunrealtype;
    let mut ult: sunrealtype;
    let mut urt: sunrealtype;
    let mut hdiff: sunrealtype;
    let mut hadv: sunrealtype;
    let mut vdiff: sunrealtype;

    let udata = N_VGetArrayPointer(u).expect("N_VGetArrayPointer");
    let mut dudata = N_VGetArrayPointer(udot).expect("N_VGetArrayPointer");

    /* Extract needed constants from data */

    let data = user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<UserData>())
        .expect("user_data is UserData");
    let hordc = data.hdcoef;
    let horac = data.hacoef;
    let verdc = data.vdcoef;

    /* Loop over all grid points. */

    for j in 1..=MY {
        for i in 1..=MX {
            /* Extract u at x_i, y_j and four neighboring points */

            uij = udata[IJth(i, j)];
            udn = if j == 1 { ZERO } else { udata[IJth(i, j - 1)] };
            uup = if j == MY { ZERO } else { udata[IJth(i, j + 1)] };
            ult = if i == 1 { ZERO } else { udata[IJth(i - 1, j)] };
            urt = if i == MX { ZERO } else { udata[IJth(i + 1, j)] };

            /* Set diffusion and advection terms and load into udot */

            hdiff = hordc * (ult - TWO * uij + urt);
            hadv = horac * (urt - ult);
            vdiff = verdc * (uup - TWO * uij + udn);
            dudata[IJth(i, j)] = hdiff + hadv + vdiff;
        }
    }

    0
}

/*
 * Jac function. Jacobian of forward ODE.
 */

fn Jac(
    _t: sunrealtype,
    _u: &N_Vector,
    _fu: &N_Vector,
    J: &SUNMatrix,
    user_data: &mut Option<Box<dyn Any>>,
    _tmp1: &N_Vector,
    _tmp2: &N_Vector,
    _tmp3: &N_Vector,
) -> i32 {
    let mut k: sunindextype;

    /*
      The components of f = udot that depend on u(i,j) are
      f(i,j), f(i-1,j), f(i+1,j), f(i,j-1), f(i,j+1), with
        df(i,j)/du(i,j) = -2 (1/dx^2 + 1/dy^2)
        df(i-1,j)/du(i,j) = 1/dx^2 + .25/dx  (if i > 1)
        df(i+1,j)/du(i,j) = 1/dx^2 - .25/dx  (if i < MX)
        df(i,j-1)/du(i,j) = 1/dy^2           (if j > 1)
        df(i,j+1)/du(i,j) = 1/dy^2           (if j < MY)
    */

    let data = user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<UserData>())
        .expect("user_data is UserData");
    let hordc = data.hdcoef;
    let horac = data.hacoef;
    let verdc = data.vdcoef;

    let s_mu = SM_SUBAND_B(J);

    for j in 1..=MY {
        for i in 1..=MX {
            k = j - 1 + (i - 1) * MY;
            let mut kthCol = SUNBandMatrix_Column(J, k);

            /* set the kth column of J */

            kthCol[SM_COLUMN_ELEMENT_IDX(k, k, s_mu)] = -TWO * (verdc + hordc);
            if i != 1 {
                kthCol[SM_COLUMN_ELEMENT_IDX(k - MY, k, s_mu)] = hordc + horac;
            }
            if i != MX {
                kthCol[SM_COLUMN_ELEMENT_IDX(k + MY, k, s_mu)] = hordc - horac;
            }
            if j != 1 {
                kthCol[SM_COLUMN_ELEMENT_IDX(k - 1, k, s_mu)] = verdc;
            }
            if j != MY {
                kthCol[SM_COLUMN_ELEMENT_IDX(k + 1, k, s_mu)] = verdc;
            }
        }
    }

    0
}

/*
 * fB function. Right-hand side of backward ODE.
 */

fn fB(
    _tB: sunrealtype,
    _u: &N_Vector,
    uB: &N_Vector,
    uBdot: &N_Vector,
    user_dataB: &mut Option<Box<dyn Any>>,
) -> i32 {
    let mut uBij: sunrealtype;
    let mut uBdn: sunrealtype;
    let mut uBup: sunrealtype;
    let mut uBlt: sunrealtype;
    let mut uBrt: sunrealtype;
    let mut hdiffB: sunrealtype;
    let mut hadvB: sunrealtype;
    let mut vdiffB: sunrealtype;

    let uBdata = N_VGetArrayPointer(uB).expect("N_VGetArrayPointer");
    let mut duBdata = N_VGetArrayPointer(uBdot).expect("N_VGetArrayPointer");

    /* Extract needed constants from data */

    let data = user_dataB
        .as_mut()
        .and_then(|b| b.downcast_mut::<UserData>())
        .expect("user_dataB is UserData");
    let hordc = data.hdcoef;
    let horac = data.hacoef;
    let verdc = data.vdcoef;

    /* Loop over all grid points. */

    for j in 1..=MY {
        for i in 1..=MX {
            /* Extract u at x_i, y_j and four neighboring points */

            uBij = uBdata[IJth(i, j)];
            uBdn = if j == 1 { ZERO } else { uBdata[IJth(i, j - 1)] };
            uBup = if j == MY { ZERO } else { uBdata[IJth(i, j + 1)] };
            uBlt = if i == 1 { ZERO } else { uBdata[IJth(i - 1, j)] };
            uBrt = if i == MX { ZERO } else { uBdata[IJth(i + 1, j)] };

            /* Set diffusion and advection terms and load into udot */

            hdiffB = hordc * (-uBlt + TWO * uBij - uBrt);
            hadvB = horac * (uBrt - uBlt);
            vdiffB = verdc * (-uBup + TWO * uBij - uBdn);
            duBdata[IJth(i, j)] = hdiffB + hadvB + vdiffB - ONE;
        }
    }

    0
}

/*
 * JacB function. Jacobian of backward ODE
 */

fn JacB(
    _tB: sunrealtype,
    _u: &N_Vector,
    _uB: &N_Vector,
    _fuB: &N_Vector,
    JB: &SUNMatrix,
    user_dataB: &mut Option<Box<dyn Any>>,
    _tmp1B: &N_Vector,
    _tmp2B: &N_Vector,
    _tmp3B: &N_Vector,
) -> i32 {
    let mut k: sunindextype;

    /* The Jacobian of the adjoint system is: JB = -J^T */

    let data = user_dataB
        .as_mut()
        .and_then(|b| b.downcast_mut::<UserData>())
        .expect("user_dataB is UserData");
    let hordc = data.hdcoef;
    let horac = data.hacoef;
    let verdc = data.vdcoef;

    let s_mu = SM_SUBAND_B(JB);

    for j in 1..=MY {
        for i in 1..=MX {
            k = j - 1 + (i - 1) * MY;
            let mut kthCol = SUNBandMatrix_Column(JB, k);

            /* set the kth column of J */

            kthCol[SM_COLUMN_ELEMENT_IDX(k, k, s_mu)] = TWO * (verdc + hordc);
            if i != 1 {
                kthCol[SM_COLUMN_ELEMENT_IDX(k - MY, k, s_mu)] = -hordc + horac;
            }
            if i != MX {
                kthCol[SM_COLUMN_ELEMENT_IDX(k + MY, k, s_mu)] = -hordc - horac;
            }
            if j != 1 {
                kthCol[SM_COLUMN_ELEMENT_IDX(k - 1, k, s_mu)] = -verdc;
            }
            if j != MY {
                kthCol[SM_COLUMN_ELEMENT_IDX(k + 1, k, s_mu)] = -verdc;
            }
        }
    }

    0
}

/*
 *--------------------------------------------------------------------
 * PRIVATE FUNCTIONS
 *--------------------------------------------------------------------
 */

/*
 * Set initial conditions in u vector
 */

fn SetIC(u: &N_Vector, data: &UserData) {
    let mut x: sunrealtype;
    let mut y: sunrealtype;

    /* Extract needed constants from data */

    let dx = data.dx;
    let dy = data.dy;

    /* Set pointer to data array in vector u. */

    let mut udata = N_VGetArrayPointer(u).expect("N_VGetArrayPointer");

    /* Load initial profile into u vector */

    for j in 1..=MY {
        y = (j as sunrealtype) * dy;
        for i in 1..=MX {
            x = (i as sunrealtype) * dx;
            udata[IJth(i, j)] = x * (XMAX - x) * y * (YMAX - y) * SUNRexp(5.0 * x * y);
        }
    }
}

/*
 * Print results after backward integration
 */

fn PrintOutput(uB: &N_Vector, data: &UserData) {
    let mut uBij: sunrealtype;
    let mut uBmax: sunrealtype;
    let mut x: sunrealtype;
    let mut y: sunrealtype;

    x = ZERO;
    y = ZERO;

    let dx = data.dx;
    let dy = data.dy;

    let uBdata = N_VGetArrayPointer(uB).expect("N_VGetArrayPointer");

    uBmax = ZERO;
    for j in 1..=MY {
        for i in 1..=MX {
            uBij = uBdata[IJth(i, j)];
            if SUNRabs(uBij) > uBmax {
                uBmax = uBij;
                x = (i as sunrealtype) * dx;
                y = (j as sunrealtype) * dy;
            }
        }
    }
    drop(uBdata);

    print!("\nMaximum sensitivity\n");
    print!("  lambda max = {}\n", fmt_e(uBmax, 6));
    print!("at\n");
    print!("  x = {}\n  y = {}\n", fmt_e(x, 6), fmt_e(y, 6));
}

/*
 * Check function return value.
 *    opt == 0 means SUNDIALS function allocates memory so check if
 *             returned NULL pointer
 *    opt == 1 means SUNDIALS function returns an integer value so check if
 *             retval < 0 (see check_retval_int)
 *    opt == 2 means function allocates memory so check if returned
 *             NULL pointer
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
