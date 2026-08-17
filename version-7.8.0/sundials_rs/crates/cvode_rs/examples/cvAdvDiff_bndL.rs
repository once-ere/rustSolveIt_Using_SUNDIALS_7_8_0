/* -----------------------------------------------------------------
 * Rust port of examples/cvode/serial/cvAdvDiff_bndL.c
 * (LAPACK band variant; this port substitutes the native
 * SUNLinSol_Band for SUNLinSol_LapackBand).
 * -----------------------------------------------------------------
 * Example problem:
 *
 * The following is a simple example problem with a banded Jacobian,
 * with the program for its solution by CVODE.
 * The problem is the semi-discrete form of the advection-diffusion
 * equation in 2-D:
 *   du/dt = d^2 u / dx^2 + .5 du/dx + d^2 u / dy^2
 * on the rectangle 0 <= x <= 2, 0 <= y <= 1, and the time
 * interval 0 <= t <= 1. Homogeneous Dirichlet boundary conditions
 * are posed, and the initial condition is
 *   u(x,y,t=0) = x(2-x)y(1-y)exp(5xy).
 * The PDE is discretized on a uniform MX+2 by MY+2 grid with
 * central differencing, and with boundary values eliminated,
 * leaving an ODE system of size NEQ = MX*MY.
 * This program solves the problem with the BDF method, Newton
 * iteration with the band linear solver, and a user-supplied
 * Jacobian routine.
 * It uses scalar relative and absolute tolerances.
 * Output is printed at t = .1, .2, ..., 1.
 * Run statistics (optional outputs) are printed at the end.
 * -----------------------------------------------------------------*/

#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

use std::any::Any;

use cvode_rs::prelude::*;

/* Problem Constants */

const XMAX: sunrealtype = 2.0; /* domain boundaries         */
const YMAX: sunrealtype = 1.0;
const MX: sunindextype = 10; /* mesh dimensions           */
const MY: sunindextype = 5;
const NEQ: sunindextype = MX * MY; /* number of equations       */
const ATOL: sunrealtype = 1.0e-5; /* scalar absolute tolerance */
const T0: sunrealtype = 0.0; /* initial time              */
const T1: sunrealtype = 0.1; /* first output time         */
const DTOUT: sunrealtype = 0.1; /* output time increment     */
const NOUT: i32 = 10; /* number of output times    */

const ZERO: sunrealtype = 0.0;
const HALF: sunrealtype = 0.5;
const ONE: sunrealtype = 1.0;
const TWO: sunrealtype = 2.0;
const FIVE: sunrealtype = 5.0;

/* User-defined vector access helper IJth */

/* IJth is defined in order to isolate the translation from the
   mathematical 2-dimensional structure of the dependent variable vector
   to the underlying 1-dimensional storage.
   IJth(vdata,i,j) references the element in the vdata array for
   u at mesh point (i,j), where 1 <= i <= MX, 1 <= j <= MY.
   The variables are ordered by the y index j, then by the x index i. */

fn IJth(vdata: &[sunrealtype], i: sunindextype, j: sunindextype) -> sunrealtype {
    vdata[((j - 1) + (i - 1) * MY) as usize]
}

fn IJth_set(vdata: &mut [sunrealtype], i: sunindextype, j: sunindextype, val: sunrealtype) {
    vdata[((j - 1) + (i - 1) * MY) as usize] = val;
}

/* Type : UserData (contains grid constants) */

struct UserData {
    dx: sunrealtype,
    dy: sunrealtype,
    hdcoef: sunrealtype,
    hacoef: sunrealtype,
    vdcoef: sunrealtype,
}

/*
 *-------------------------------
 * Main Program
 *-------------------------------
 */

fn main() {
    let mut t: sunrealtype = 0.0;
    let mut nst: i64 = 0;

    let mut sunctx: Option<SUNContext> = None;

    /* Create the SUNDIALS context */
    let retval = SUNContext_Create(SUN_COMM_NULL, &mut sunctx);
    if check_retval_int(retval, "SUNContext_Create") != 0 {
        std::process::exit(1);
    }
    let ctx = sunctx.as_ref().expect("SUNContext").clone();

    /* Create a serial vector */

    let u = N_VNew_Serial(NEQ, &ctx); /* Allocate u vector */
    if check_retval_ptr(&u, "N_VNew_Serial", 0) != 0 {
        std::process::exit(1);
    }
    let u = u.expect("N_VNew_Serial");

    let reltol = ZERO; /* Set the tolerances */
    let abstol = ATOL;

    /* Allocate data memory; set grid coefficients in data */
    let dx = XMAX / ((MX + 1) as sunrealtype);
    let dy = YMAX / ((MY + 1) as sunrealtype);
    let data = UserData {
        dx,
        dy,
        hdcoef: ONE / (dx * dx),
        hacoef: HALF / (TWO * dx),
        vdcoef: ONE / (dy * dy),
    };

    SetIC(&u, &data); /* Initialize u vector */

    /* Call CVodeCreate to create the solver memory and specify the
     * Backward Differentiation Formula */
    let cvode_mem = CVodeCreate(CV_BDF, &ctx);
    if check_retval_ptr(&cvode_mem, "CVodeCreate", 0) != 0 {
        std::process::exit(1);
    }
    let cvode_mem = cvode_mem.expect("CVodeCreate");

    /* Call CVodeInit to initialize the integrator memory and specify the
     * user's right hand side function in u'=f(t,u), the initial time T0, and
     * the initial dependent variable vector u. */
    let retval = CVodeInit(&cvode_mem, f, T0, &u);
    if check_retval_int(retval, "CVodeInit") != 0 {
        std::process::exit(1);
    }

    /* Call CVodeSStolerances to specify the scalar relative tolerance
     * and scalar absolute tolerance */
    let retval = CVodeSStolerances(&cvode_mem, reltol, abstol);
    if check_retval_int(retval, "CVodeSStolerances") != 0 {
        std::process::exit(1);
    }

    /* Set the pointer to user-defined data */
    let retval = CVodeSetUserData(&cvode_mem, Some(Box::new(data)));
    if check_retval_int(retval, "CVodeSetUserData") != 0 {
        std::process::exit(1);
    }

    /* Create banded SUNMatrix for use in linear solves -- since this will be factored,
    set the storage bandwidth to be the sum of upper and lower bandwidths */
    let A = SUNBandMatrix(NEQ, MY, MY, &ctx);
    if check_retval_ptr(&A, "SUNBandMatrix", 0) != 0 {
        std::process::exit(1);
    }
    let A = A.expect("SUNBandMatrix");

    /* Create SUNLinSol_Band solver object for use by CVode */
    let LS = SUNLinSol_Band(&u, &A, &ctx);
    if check_retval_ptr(&LS, "SUNLinSol_Band", 0) != 0 {
        std::process::exit(1);
    }
    let LS = LS.expect("SUNLinSol_Band");

    /* Call CVodeSetLinearSolver to attach the matrix and linear solver to CVode */
    let retval = CVodeSetLinearSolver(&cvode_mem, &LS, Some(&A));
    if check_retval_int(retval, "CVodeSetLinearSolver") != 0 {
        std::process::exit(1);
    }

    /* Set the user-supplied Jacobian routine Jac */
    let retval = CVodeSetJacFn(&cvode_mem, Some(Jac));
    if check_retval_int(retval, "CVodeSetJacFn") != 0 {
        std::process::exit(1);
    }

    /* In loop over output points: call CVode, print results, test for errors */

    let mut umax = N_VMaxNorm(&u);
    PrintHeader(reltol, abstol, umax);
    let mut tout = T1;
    for _iout in 1..=NOUT {
        let retval = CVode(&cvode_mem, tout, &u, &mut t, CV_NORMAL);
        if check_retval_int(retval, "CVode") != 0 {
            break;
        }
        umax = N_VMaxNorm(&u);
        let retval = CVodeGetNumSteps(&cvode_mem, &mut nst);
        check_retval_int(retval, "CVodeGetNumSteps");
        PrintOutput(t, umax, nst);
        tout += DTOUT;
    }

    PrintFinalStats(&cvode_mem); /* Print some final statistics   */

    N_VDestroy(u); /* Free the u vector */
    let mut cvode_mem = Some(cvode_mem);
    CVodeFree(&mut cvode_mem); /* Free the integrator memory */
    SUNLinSolFree(Some(LS)); /* Free linear solver memory  */
    SUNMatDestroy(A); /* Free the matrix memory     */

    SUNContext_Free(&mut sunctx);
}

/*
 *-------------------------------
 * Functions called by the solver
 *-------------------------------
 */

/* f routine. Compute f(t,u). */

fn f(_t: sunrealtype, u: &N_Vector, udot: &N_Vector, user_data: &mut Option<Box<dyn Any>>) -> i32 {
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

            let uij = IJth(&udata, i, j);
            let udn = if j == 1 { ZERO } else { IJth(&udata, i, j - 1) };
            let uup = if j == MY { ZERO } else { IJth(&udata, i, j + 1) };
            let ult = if i == 1 { ZERO } else { IJth(&udata, i - 1, j) };
            let urt = if i == MX { ZERO } else { IJth(&udata, i + 1, j) };

            /* Set diffusion and advection terms and load into udot */

            let hdiff = hordc * (ult - TWO * uij + urt);
            let hadv = horac * (urt - ult);
            let vdiff = verdc * (uup - TWO * uij + udn);
            IJth_set(&mut dudata, i, j, hdiff + hadv + vdiff);
        }
    }

    0
}

/* Jacobian routine. Compute J(t,u). */

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
    /*
     * The components of f = udot that depend on u(i,j) are
     * f(i,j), f(i-1,j), f(i+1,j), f(i,j-1), f(i,j+1), with
     *   df(i,j)/du(i,j) = -2 (1/dx^2 + 1/dy^2)
     *   df(i-1,j)/du(i,j) = 1/dx^2 + .25/dx  (if i > 1)
     *   df(i+1,j)/du(i,j) = 1/dx^2 - .25/dx  (if i < MX)
     *   df(i,j-1)/du(i,j) = 1/dy^2           (if j > 1)
     *   df(i,j+1)/du(i,j) = 1/dy^2           (if j < MY)
     */

    let data = user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<UserData>())
        .expect("user_data is UserData");
    let hordc = data.hdcoef;
    let horac = data.hacoef;
    let verdc = data.vdcoef;

    let s_mu = SM_SUBAND_B(J);

    /* set non-zero Jacobian entries */
    for j in 1..=MY {
        for i in 1..=MX {
            let k = j - 1 + (i - 1) * MY;
            let mut kthCol = SM_COLUMN_B(J, k);

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
 *-------------------------------
 * Private helper functions
 *-------------------------------
 */

/* Set initial conditions in u vector */

fn SetIC(u: &N_Vector, data: &UserData) {
    /* Extract needed constants from data */

    let dx = data.dx;
    let dy = data.dy;

    /* Set pointer to data array in vector u. */

    let mut udata = N_VGetArrayPointer(u).expect("N_VGetArrayPointer");

    /* Load initial profile into u vector */

    for j in 1..=MY {
        let y = (j as sunrealtype) * dy;
        for i in 1..=MX {
            let x = (i as sunrealtype) * dx;
            IJth_set(
                &mut udata,
                i,
                j,
                x * (XMAX - x) * y * (YMAX - y) * SUNRexp(FIVE * x * y),
            );
        }
    }
}

/* Print first lines of output (problem description) */

fn PrintHeader(reltol: sunrealtype, abstol: sunrealtype, umax: sunrealtype) {
    print!("\n2-D Advection-Diffusion Equation\n");
    print!("Mesh dimensions = {} X {}\n", MX, MY);
    print!("Total system size = {}\n", NEQ);
    print!(
        "Tolerance parameters: reltol = {}   abstol = {}\n\n",
        fmt_g(reltol, 6),
        fmt_g(abstol, 6)
    );
    print!(
        "At t = {}      max.norm(u) ={} \n",
        fmt_g(T0, 6),
        fmt_ew(umax, 14, 6)
    );
}

/* Print current value */

fn PrintOutput(t: sunrealtype, umax: sunrealtype, nst: i64) {
    print!(
        "At t = {}   max.norm(u) ={}   nst = {:>4}\n",
        fmt_fw(t, 4, 2),
        fmt_ew(umax, 14, 6),
        nst
    );
}

/* Get and print some final statistics */

fn PrintFinalStats(cvode_mem: &CVodeMem) {
    let (mut nst, mut nfe, mut nsetups, mut netf) = (0i64, 0i64, 0i64, 0i64);
    let (mut nni, mut ncfn, mut nje, mut nfeLS) = (0i64, 0i64, 0i64, 0i64);

    let retval = CVodeGetNumSteps(cvode_mem, &mut nst);
    check_retval_int(retval, "CVodeGetNumSteps");
    let retval = CVodeGetNumRhsEvals(cvode_mem, &mut nfe);
    check_retval_int(retval, "CVodeGetNumRhsEvals");
    let retval = CVodeGetNumLinSolvSetups(cvode_mem, &mut nsetups);
    check_retval_int(retval, "CVodeGetNumLinSolvSetups");
    let retval = CVodeGetNumErrTestFails(cvode_mem, &mut netf);
    check_retval_int(retval, "CVodeGetNumErrTestFails");
    let retval = CVodeGetNumNonlinSolvIters(cvode_mem, &mut nni);
    check_retval_int(retval, "CVodeGetNumNonlinSolvIters");
    let retval = CVodeGetNumNonlinSolvConvFails(cvode_mem, &mut ncfn);
    check_retval_int(retval, "CVodeGetNumNonlinSolvConvFails");

    let retval = CVodeGetNumJacEvals(cvode_mem, &mut nje);
    check_retval_int(retval, "CVodeGetNumJacEvals");
    let retval = CVodeGetNumLinRhsEvals(cvode_mem, &mut nfeLS);
    check_retval_int(retval, "CVodeGetNumLinRhsEvals");

    print!("\nFinal Statistics:\n");
    print!(
        "nst = {:<6} nfe  = {:<6} nsetups = {:<6} nfeLS = {:<6} nje = {}\n",
        nst, nfe, nsetups, nfeLS, nje
    );
    print!("nni = {:<6} ncfn = {:<6} netf = {}\n", nni, ncfn, netf);
}

/* Check function return value...
    opt == 0 means SUNDIALS function allocates memory so check if
             returned NULL pointer
    opt == 1 means SUNDIALS function returns an integer value so check if
             retval < 0 (see check_retval_int)
    opt == 2 means function allocates memory so check if returned
             NULL pointer */

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
