#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

/* -----------------------------------------------------------------
 * Ported from: examples/idas/serial/idasHeat2D_bnd.c
 * Programmer(s): Allan Taylor, Alan Hindmarsh and
 *                Radu Serban @ LLNL
 * -----------------------------------------------------------------
 * Example problem for IDA: 2D heat equation, serial, banded.
 *
 * This example solves a discretized 2D heat equation problem.
 * This version uses the band solver and IDACalcIC.
 *
 * The DAE system solved is a spatial discretization of the PDE
 *          du/dt = d^2u/dx^2 + d^2u/dy^2
 * on the unit square. The boundary condition is u = 0 on all edges.
 * Initial conditions are given by u = 16 x (1 - x) y (1 - y).
 * The PDE is treated with central differences on a uniform M x M
 * grid. The values of u at the interior points satisfy ODEs, and
 * equations u = 0 at the boundaries are appended, to form a DAE
 * system of size N = M^2. Here M = 10.
 *
 * The system is solved with IDA using the banded linear system
 * solver, half-bandwidths equal to M, and default
 * difference-quotient Jacobian. For purposes of illustration,
 * IDACalcIC is called to compute correct values at the boundary,
 * given incorrect values as input initial guesses. The constraints
 * u >= 0 are posed for all components. Output is taken at
 * t = 0, .01, .02, .04, ..., 10.24. (Output at t = 0 is for
 * IDACalcIC cost statistics only.)
 * -----------------------------------------------------------------*/

use std::any::Any;

use idas_rs::prelude::*;

/* Problem Constants */

const NOUT: i32 = 11;
const MGRID: sunindextype = 10;
const NEQ: sunindextype = MGRID * MGRID;
const ZERO: sunrealtype = 0.0;
const ONE: sunrealtype = 1.0;
const TWO: sunrealtype = 2.0;
const BVAL: sunrealtype = 0.1;

/* Type: UserData */

struct UserData {
    mm: sunindextype,
    dx: sunrealtype,
    coeff: sunrealtype,
}

/*
 *--------------------------------------------------------------------
 * MAIN PROGRAM
 *--------------------------------------------------------------------
 */

fn main() {
    let mut retval: i32;
    let mut netf: i64 = 0;
    let mut ncfn: i64 = 0;
    let mu: sunindextype;
    let ml: sunindextype;
    let rtol: sunrealtype;
    let atol: sunrealtype;
    let t0: sunrealtype;
    let t1: sunrealtype;
    let mut tout: sunrealtype;
    let mut tret: sunrealtype = 0.0;

    /* Create the SUNDIALS context object for this simulation */
    let mut ctx: Option<SUNContext> = None;
    retval = SUNContext_Create(SUN_COMM_NULL, &mut ctx);
    if check_retval(&retval, "SUNContext_Create") != 0 {
        std::process::exit(1);
    }
    let ctx = ctx.expect("SUNContext_Create");

    /* Create vectors uu, up, res, constraints, id. */
    let uu = N_VNew_Serial(NEQ, &ctx);
    if check_ptr(&uu, "N_VNew_Serial", 0) != 0 {
        std::process::exit(1);
    }
    let uu = uu.expect("N_VNew_Serial");
    let up = N_VClone(&uu);
    if check_ptr(&up, "N_VNew_Serial", 0) != 0 {
        std::process::exit(1);
    }
    let up = up.expect("N_VClone");
    let res = N_VClone(&uu);
    if check_ptr(&res, "N_VNew_Serial", 0) != 0 {
        std::process::exit(1);
    }
    let res = res.expect("N_VClone");
    let constraints = N_VClone(&uu);
    if check_ptr(&constraints, "N_VNew_Serial", 0) != 0 {
        std::process::exit(1);
    }
    let constraints = constraints.expect("N_VClone");
    let id = N_VClone(&uu);
    if check_ptr(&id, "N_VNew_Serial", 0) != 0 {
        std::process::exit(1);
    }
    let id = id.expect("N_VClone");

    /* Create and load problem data block. */
    let dx = ONE / ((MGRID - 1) as sunrealtype);
    let mut data: Option<Box<dyn Any>> = Some(Box::new(UserData {
        mm: MGRID,
        dx,
        coeff: ONE / (dx * dx),
    }));
    if check_ptr(&data, "malloc", 2) != 0 {
        std::process::exit(1);
    }

    /* Initialize uu, up, id. */
    SetInitialProfile(&mut data, &uu, &up, &id, &res);

    /* Set constraints to all 1's for nonnegative solution values. */
    N_VConst(ONE, &constraints);

    /* Set remaining input parameters. */
    t0 = ZERO;
    t1 = 0.01;
    rtol = ZERO;
    atol = 1.0e-3;

    /* Call IDACreate and IDAMalloc to initialize solution */
    let mem = IDACreate(&ctx);
    if check_ptr(&mem, "IDACreate", 0) != 0 {
        std::process::exit(1);
    }
    let mem = mem.expect("IDACreate");

    retval = IDASetUserData(&mem, data.take());
    if check_retval(&retval, "IDASetUserData") != 0 {
        std::process::exit(1);
    }

    /* Set which components are algebraic or differential */
    retval = IDASetId(&mem, Some(&id));
    if check_retval(&retval, "IDASetId") != 0 {
        std::process::exit(1);
    }

    retval = IDASetConstraints(&mem, Some(&constraints));
    if check_retval(&retval, "IDASetConstraints") != 0 {
        std::process::exit(1);
    }
    N_VDestroy(constraints);

    retval = IDAInit(&mem, heatres, t0, &uu, &up);
    if check_retval(&retval, "IDAInit") != 0 {
        std::process::exit(1);
    }

    retval = IDASStolerances(&mem, rtol, atol);
    if check_retval(&retval, "IDASStolerances") != 0 {
        std::process::exit(1);
    }

    /* Create banded SUNMatrix for use in linear solves */
    mu = MGRID;
    ml = MGRID;
    let A = SUNBandMatrix(NEQ, mu, ml, &ctx);
    if check_ptr(&A, "SUNBandMatrix", 0) != 0 {
        std::process::exit(1);
    }
    let A = A.expect("SUNBandMatrix");

    /* Create banded SUNLinearSolver object */
    let LS = SUNLinSol_Band(&uu, &A, &ctx);
    if check_ptr(&LS, "SUNLinSol_Band", 0) != 0 {
        std::process::exit(1);
    }
    let LS = LS.expect("SUNLinSol_Band");

    /* Attach the matrix and linear solver */
    retval = IDASetLinearSolver(&mem, &LS, Some(&A));
    if check_retval(&retval, "IDASetLinearSolver") != 0 {
        std::process::exit(1);
    }

    /* Call IDACalcIC to correct the initial values. */

    retval = IDACalcIC(&mem, IDA_YA_YDP_INIT, t1);
    if check_retval(&retval, "IDACalcIC") != 0 {
        std::process::exit(1);
    }

    /* Print output heading. */
    PrintHeader(rtol, atol);

    PrintOutput(&mem, t0, &uu);

    /* Loop over output times, call IDASolve, and print results. */

    tout = t1;
    let mut iout = 1;
    while iout <= NOUT {
        retval = IDASolve(&mem, tout, &mut tret, &uu, &up, IDA_NORMAL);
        if check_retval(&retval, "IDASolve") != 0 {
            std::process::exit(1);
        }

        PrintOutput(&mem, tret, &uu);

        iout += 1;
        tout *= TWO;
    }

    /* Print remaining counters and free memory. */
    retval = IDAGetNumErrTestFails(&mem, &mut netf);
    check_retval(&retval, "IDAGetNumErrTestFails");
    retval = IDAGetNumNonlinSolvConvFails(&mem, &mut ncfn);
    check_retval(&retval, "IDAGetNumNonlinSolvConvFails");
    print!("\n netf = {},   ncfn = {} \n", netf, ncfn);

    IDAFree(&mut Some(mem));
    let _ = SUNLinSolFree(Some(LS));
    SUNMatDestroy(A);
    N_VDestroy(uu);
    N_VDestroy(up);
    N_VDestroy(id);
    N_VDestroy(res);
    /* free(data) -- the UserData box is owned by the IDA memory record */

    let _ = SUNContext_Free(&mut Some(ctx));
}

/*
 *--------------------------------------------------------------------
 * FUNCTIONS CALLED BY IDA
 *--------------------------------------------------------------------
 */

/*
 * heatres: heat equation system residual function
 * This uses 5-point central differencing on the interior points, and
 * includes algebraic equations for the boundary values.
 * So for each interior point, the residual component has the form
 *    res_i = u'_i - (central difference)_i
 * while for each boundary point, it is res_i = u_i.
 */

fn heatres(
    _tres: sunrealtype,
    uu: &N_Vector,
    up: &N_Vector,
    resval: &N_Vector,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    let (mm, coeff) = {
        let data = user_data
            .as_mut()
            .and_then(|b| b.downcast_mut::<UserData>())
            .expect("user_data is UserData");
        (data.mm, data.coeff)
    };

    /* Initialize resval to uu, to take care of boundary equations. */
    N_VScale(ONE, uu, resval);

    /* Loop over interior points; set res = up - (central difference). */
    {
        let uv = N_VGetArrayPointer(uu).expect("N_VGetArrayPointer");
        let upv = N_VGetArrayPointer(up).expect("N_VGetArrayPointer");
        let mut resv = N_VGetArrayPointer(resval).expect("N_VGetArrayPointer");

        for j in 1..(mm - 1) {
            let offset = mm * j;
            for i in 1..(mm - 1) {
                let loc = (offset + i) as usize;
                resv[loc] = upv[loc]
                    - coeff
                        * (uv[loc - 1]
                            + uv[loc + 1]
                            + uv[loc - mm as usize]
                            + uv[loc + mm as usize]
                            - 4.0 * uv[loc]);
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
 * SetInitialProfile: routine to initialize u, up, and id vectors.
 */

fn SetInitialProfile(
    data: &mut Option<Box<dyn Any>>,
    uu: &N_Vector,
    up: &N_Vector,
    id: &N_Vector,
    res: &N_Vector,
) -> i32 {
    let (mm, dx) = {
        let d = data
            .as_mut()
            .and_then(|b| b.downcast_mut::<UserData>())
            .expect("user_data is UserData");
        (d.mm, d.dx)
    };
    let mm1 = mm - 1;

    /* Initialize id to 1's. */
    N_VConst(ONE, id);

    /* Initialize uu on all grid points. */
    {
        let mut udata = N_VGetArrayPointer(uu).expect("N_VGetArrayPointer");

        for j in 0..mm {
            let yfact = dx * (j as sunrealtype);
            let offset = mm * j;
            for i in 0..mm {
                let xfact = dx * (i as sunrealtype);
                let loc = (offset + i) as usize;
                udata[loc] = 16.0 * xfact * (ONE - xfact) * yfact * (ONE - yfact);
            }
        }
    }

    /* Initialize up vector to 0. */
    N_VConst(ZERO, up);

    /* heatres sets res to negative of ODE RHS values at interior points. */
    heatres(ZERO, uu, up, res, data);

    /* Copy -res into up to get correct interior initial up values. */
    N_VScale(-ONE, res, up);

    /* Finally, set values of u, up, and id at boundary points. */
    {
        let mut udata = N_VGetArrayPointer(uu).expect("N_VGetArrayPointer");
        let mut updata = N_VGetArrayPointer(up).expect("N_VGetArrayPointer");
        let mut iddata = N_VGetArrayPointer(id).expect("N_VGetArrayPointer");

        for j in 0..mm {
            let offset = mm * j;
            for i in 0..mm {
                let loc = (offset + i) as usize;
                if j == 0 || j == mm1 || i == 0 || i == mm1 {
                    udata[loc] = BVAL;
                    updata[loc] = ZERO;
                    iddata[loc] = ZERO;
                }
            }
        }
    }

    0
}

/*
 * Print first lines of output (problem description)
 */

fn PrintHeader(rtol: sunrealtype, atol: sunrealtype) {
    print!("\nidasHeat2D_bnd: Heat equation, serial example problem for IDA\n");
    print!("              Discretized heat equation on 2D unit square.\n");
    print!("              Zero boundary conditions,");
    print!(" polynomial initial conditions.\n");
    print!("              Mesh dimensions: {} x {}", MGRID, MGRID);
    print!("        Total system size: {}\n\n", NEQ);
    print!(
        "Tolerance parameters:  rtol = {}   atol = {}\n",
        fmt_g(rtol, 6),
        fmt_g(atol, 6)
    );
    print!("Constraints set to force all solution components >= 0. \n");
    print!("Linear solver: BAND, banded direct solver \n");
    print!(
        "       difference quotient Jacobian, half-bandwidths = {} \n",
        MGRID
    );
    print!(
        "IDACalcIC called with input boundary values = {} \n",
        fmt_g(BVAL, 6)
    );
    /* Print output table heading and initial line of table. */
    print!("\n   Output Summary (umax = max-norm of solution) \n\n");
    print!("  time       umax     k  nst  nni  nje   nre   nreLS    h      \n");
    print!(" .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  . \n");
}

/*
 * Print Output
 */

fn PrintOutput(mem: &IDAMem, t: sunrealtype, uu: &N_Vector) {
    let mut retval: i32;
    let mut hused: sunrealtype = 0.0;
    let mut nst: i64 = 0;
    let mut nni: i64 = 0;
    let mut nje: i64 = 0;
    let mut nre: i64 = 0;
    let mut nreLS: i64 = 0;
    let mut kused: i32 = 0;

    let umax = N_VMaxNorm(uu);

    retval = IDAGetLastOrder(mem, &mut kused);
    check_retval(&retval, "IDAGetLastOrder");
    retval = IDAGetNumSteps(mem, &mut nst);
    check_retval(&retval, "IDAGetNumSteps");
    retval = IDAGetNumNonlinSolvIters(mem, &mut nni);
    check_retval(&retval, "IDAGetNumNonlinSolvIters");
    retval = IDAGetNumResEvals(mem, &mut nre);
    check_retval(&retval, "IDAGetNumResEvals");
    retval = IDAGetLastStep(mem, &mut hused);
    check_retval(&retval, "IDAGetLastStep");
    retval = IDAGetNumJacEvals(mem, &mut nje);
    check_retval(&retval, "IDAGetNumJacEvals");
    retval = IDAGetNumLinResEvals(mem, &mut nreLS);
    check_retval(&retval, "IDAGetNumLinResEvals");

    print!(
        " {} {}  {}  {:>3}  {:>3}  {:>3}  {:>4}  {:>4}  {} \n",
        fmt_fw(t, 5, 2),
        fmt_ew(umax, 13, 5),
        kused,
        nst,
        nni,
        nje,
        nre,
        nreLS,
        fmt_ew(hused, 9, 2)
    );
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

fn check_retval(retval: &i32, funcname: &str) -> i32 {
    /* Check if retval < 0 */
    if *retval < 0 {
        eprint!(
            "\nSUNDIALS_ERROR: {}() failed with retval = {}\n\n",
            funcname, retval
        );
        return 1;
    }
    0
}

fn check_ptr<T>(returnvalue: &Option<T>, funcname: &str, opt: i32) -> i32 {
    if returnvalue.is_none() {
        if opt == 0 {
            /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
            eprint!(
                "\nSUNDIALS_ERROR: {}() failed - returned NULL pointer\n\n",
                funcname
            );
        } else {
            /* Check if function returned NULL pointer - no memory allocated */
            eprint!(
                "\nMEMORY_ERROR: {}() failed - returned NULL pointer\n\n",
                funcname
            );
        }
        return 1;
    }
    0
}
