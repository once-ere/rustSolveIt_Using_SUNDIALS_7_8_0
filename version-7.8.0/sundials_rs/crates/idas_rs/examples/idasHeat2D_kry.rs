#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

/* -----------------------------------------------------------------
 * Ported from: examples/idas/serial/idasHeat2D_kry.c
 * Programmer(s): Allan Taylor, Alan Hindmarsh and
 *                Radu Serban @ LLNL
 * -----------------------------------------------------------------
 * Example problem for IDA: 2D heat equation, serial, GMRES.
 *
 * This example solves a discretized 2D heat equation problem.
 * This version uses the Krylov solver Spgmr.
 *
 * The DAE system solved is a spatial discretization of the PDE
 *          du/dt = d^2u/dx^2 + d^2u/dy^2
 * on the unit square. The boundary condition is u = 0 on all edges.
 * Initial conditions are given by u = 16 x (1 - x) y (1 - y). The
 * PDE is treated with central differences on a uniform M x M grid.
 * The values of u at the interior points satisfy ODEs, and
 * equations u = 0 at the boundaries are appended, to form a DAE
 * system of size N = M^2. Here M = 10.
 *
 * The system is solved with IDA using the Krylov linear solver
 * SPGMR. The preconditioner uses the diagonal elements of the
 * Jacobian only. Routines for preconditioning, required by
 * SPGMR, are supplied here. The constraints u >= 0 are posed
 * for all components. Output is taken at t = 0, .01, .02, .04,
 * ..., 10.24. Two cases are run -- with the Gram-Schmidt type
 * being Modified in the first case, and Classical in the second.
 * The second run uses IDAReInit.
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
const FOUR: sunrealtype = 4.0;

/* User data type */

struct UserData {
    mm: sunindextype, /* number of grid points */
    dx: sunrealtype,
    coeff: sunrealtype,
    pp: Option<N_Vector>, /* vector of prec. diag. elements */
}

/*
 *--------------------------------------------------------------------
 * MAIN PROGRAM
 *--------------------------------------------------------------------
 */

fn main() {
    let mut retval: i32;
    let rtol: sunrealtype;
    let atol: sunrealtype;
    let t0: sunrealtype;
    let t1: sunrealtype;
    let mut tout: sunrealtype;
    let mut tret: sunrealtype = 0.0;
    let mut netf: i64 = 0;
    let mut ncfn: i64 = 0;
    let mut ncfl: i64 = 0;

    /* Create the SUNDIALS context object for this simulation */

    let mut ctx: Option<SUNContext> = None;
    retval = SUNContext_Create(SUN_COMM_NULL, &mut ctx);
    if check_retval(&retval, "SUNContext_Create") != 0 {
        std::process::exit(1);
    }
    let ctx = ctx.expect("SUNContext_Create");

    /* Allocate N-vectors and the user data structure. */

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

    let mut data: Option<Box<dyn Any>> = Some(Box::new(UserData {
        mm: 0,
        dx: ZERO,
        coeff: ZERO,
        pp: None,
    }));
    if check_ptr(&data, "malloc", 2) != 0 {
        std::process::exit(1);
    }

    /* Assign parameters in the user data structure. */

    let pp = N_VClone(&uu);
    {
        let d = data
            .as_mut()
            .and_then(|b| b.downcast_mut::<UserData>())
            .expect("user_data is UserData");
        d.mm = MGRID;
        d.dx = ONE / ((MGRID - 1) as sunrealtype);
        d.coeff = ONE / (d.dx * d.dx);
        d.pp = pp.clone();
    }
    if check_ptr(&pp, "N_VNew_Serial", 0) != 0 {
        std::process::exit(1);
    }

    /* Initialize uu, up. */

    SetInitialProfile(&mut data, &uu, &up, &res);

    /* Set constraints to all 1's for nonnegative solution values. */

    N_VConst(ONE, &constraints);

    /* Assign various parameters. */

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

    retval = IDASetConstraints(&mem, Some(&constraints));
    if check_retval(&retval, "IDASetConstraints") != 0 {
        std::process::exit(1);
    }
    N_VDestroy(constraints);

    retval = IDAInit(&mem, resHeat, t0, &uu, &up);
    if check_retval(&retval, "IDAInit") != 0 {
        std::process::exit(1);
    }

    retval = IDASStolerances(&mem, rtol, atol);
    if check_retval(&retval, "IDASStolerances") != 0 {
        std::process::exit(1);
    }

    /* Create the linear solver SUNLinSol_SPGMR with left preconditioning
    and the default Krylov dimension */
    let LS = SUNLinSol_SPGMR(&uu, SUN_PREC_LEFT, 0, &ctx);
    if check_ptr(&LS, "SUNLinSol_SPGMR", 0) != 0 {
        std::process::exit(1);
    }
    let LS = LS.expect("SUNLinSol_SPGMR");

    /* IDA recommends allowing up to 5 restarts (default is 0) */
    retval = SUNLinSol_SPGMRSetMaxRestarts(&LS, 5);
    if check_retval(&retval, "SUNLinSol_SPGMRSetMaxRestarts") != 0 {
        std::process::exit(1);
    }

    /* Attach the linear solver */
    retval = IDASetLinearSolver(&mem, &LS, None);
    if check_retval(&retval, "IDASetLinearSolver") != 0 {
        std::process::exit(1);
    }

    /* Set the preconditioner solve and setup functions */
    retval = IDASetPreconditioner(&mem, Some(PsetupHeat), Some(PsolveHeat));
    if check_retval(&retval, "IDASetPreconditioner") != 0 {
        std::process::exit(1);
    }

    /* Print output heading. */
    PrintHeader(rtol, atol);

    /*
     * -------------------------------------------------------------------------
     * CASE I
     * -------------------------------------------------------------------------
     */

    /* Print case number, output table heading, and initial line of table. */

    print!("\n\nCase 1: gsytpe = SUN_MODIFIED_GS\n");
    print!("\n   Output Summary (umax = max-norm of solution) \n\n");
    print!("  time     umax       k  nst  nni  nje   nre   nreLS    h      npe nps\n");
    print!("----------------------------------------------------------------------\n");

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

    /* Print remaining counters. */

    retval = IDAGetNumErrTestFails(&mem, &mut netf);
    check_retval(&retval, "IDAGetNumErrTestFails");

    retval = IDAGetNumNonlinSolvConvFails(&mem, &mut ncfn);
    check_retval(&retval, "IDAGetNumNonlinSolvConvFails");

    retval = IDAGetNumLinConvFails(&mem, &mut ncfl);
    check_retval(&retval, "IDAGetNumLinConvFails");

    print!("\nError test failures            = {}\n", netf);
    print!("Nonlinear convergence failures = {}\n", ncfn);
    print!("Linear convergence failures    = {}\n", ncfl);

    /*
     * -------------------------------------------------------------------------
     * CASE II
     * -------------------------------------------------------------------------
     */

    /* Re-initialize uu, up. */

    /* The user data box lives inside the IDA memory; swap it out, use it, and
    hand it straight back (see ARCHITECTURE "user_data pointer-snapshot"). */
    let _ = IDAGetUserData(&mem, &mut data);
    SetInitialProfile(&mut data, &uu, &up, &res);
    let _ = IDASetUserData(&mem, data.take());

    /* Re-initialize IDA and SPGMR */

    retval = IDAReInit(&mem, t0, &uu, &up);
    if check_retval(&retval, "IDAReInit") != 0 {
        std::process::exit(1);
    }

    retval = SUNLinSol_SPGMRSetGSType(&LS, SUN_CLASSICAL_GS);
    if check_retval(&retval, "SUNLinSol_SPGMRSetGSType") != 0 {
        std::process::exit(1);
    }

    /* Print case number, output table heading, and initial line of table. */

    print!("\n\nCase 2: gstype = SUN_CLASSICAL_GS\n");
    print!("\n   Output Summary (umax = max-norm of solution) \n\n");
    print!("  time     umax       k  nst  nni  nje   nre   nreLS    h      npe nps\n");
    print!("----------------------------------------------------------------------\n");

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

    /* Print remaining counters. */

    retval = IDAGetNumErrTestFails(&mem, &mut netf);
    check_retval(&retval, "IDAGetNumErrTestFails");

    retval = IDAGetNumNonlinSolvConvFails(&mem, &mut ncfn);
    check_retval(&retval, "IDAGetNumNonlinSolvConvFails");

    retval = IDAGetNumLinConvFails(&mem, &mut ncfl);
    check_retval(&retval, "IDAGetNumLinConvFails");

    print!("\nError test failures            = {}\n", netf);
    print!("Nonlinear convergence failures = {}\n", ncfn);
    print!("Linear convergence failures    = {}\n", ncfl);

    /* Free Memory */

    IDAFree(&mut Some(mem));
    let _ = SUNLinSolFree(Some(LS));

    N_VDestroy(uu);
    N_VDestroy(up);
    N_VDestroy(res);

    /* data->pp and the UserData box are owned by the IDA memory record */

    let _ = SUNContext_Free(&mut Some(ctx));
}

/*
 *--------------------------------------------------------------------
 * FUNCTIONS CALLED BY IDA
 *--------------------------------------------------------------------
 */

/*
 * resHeat: heat equation system residual function (user-supplied)
 * This uses 5-point central differencing on the interior points, and
 * includes algebraic equations for the boundary values.
 * So for each interior point, the residual component has the form
 *    res_i = u'_i - (central difference)_i
 * while for each boundary point, it is res_i = u_i.
 */

fn resHeat(
    _tt: sunrealtype,
    uu: &N_Vector,
    up: &N_Vector,
    rr: &N_Vector,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    let (coeff, mm) = {
        let data = user_data
            .as_mut()
            .and_then(|b| b.downcast_mut::<UserData>())
            .expect("user_data is UserData");
        (data.coeff, data.mm)
    };

    /* Initialize rr to uu, to take care of boundary equations. */
    N_VScale(ONE, uu, rr);

    /* Loop over interior points; set res = up - (central difference). */
    {
        let uu_data = N_VGetArrayPointer(uu).expect("N_VGetArrayPointer");
        let up_data = N_VGetArrayPointer(up).expect("N_VGetArrayPointer");
        let mut rr_data = N_VGetArrayPointer(rr).expect("N_VGetArrayPointer");

        for j in 1..(MGRID - 1) {
            let offset = mm * j;
            for i in 1..(mm - 1) {
                let loc = (offset + i) as usize;
                let dif1 = uu_data[loc - 1] + uu_data[loc + 1] - TWO * uu_data[loc];
                let dif2 =
                    uu_data[loc - mm as usize] + uu_data[loc + mm as usize] - TWO * uu_data[loc];
                rr_data[loc] = up_data[loc] - coeff * (dif1 + dif2);
            }
        }
    }

    0
}

/*
 * PsetupHeat: setup for diagonal preconditioner.
 *
 * The optional user-supplied functions PsetupHeat and
 * PsolveHeat together must define the left preconditioner
 * matrix P approximating the system Jacobian matrix
 *                   J = dF/du + cj*dF/du'
 * (where the DAE system is F(t,u,u') = 0), and solve the linear
 * systems P z = r.   This is done in this case by keeping only
 * the diagonal elements of the J matrix above, storing them as
 * inverses in a vector pp, when computed in PsetupHeat, for
 * subsequent use in PsolveHeat.
 *
 * In this instance, only cj and data (user data structure, with
 * pp etc.) are used from the PsetupdHeat argument list.
 */

fn PsetupHeat(
    _tt: sunrealtype,
    _uu: &N_Vector,
    _up: &N_Vector,
    _rr: &N_Vector,
    c_j: sunrealtype,
    prec_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    let (pp, mm, coeff) = {
        let data = prec_data
            .as_mut()
            .and_then(|b| b.downcast_mut::<UserData>())
            .expect("prec_data is UserData");
        (data.pp.as_ref().expect("pp").clone(), data.mm, data.coeff)
    };

    /* Initialize the entire vector to 1., then set the interior points to the
    correct value for preconditioning. */
    N_VConst(ONE, &pp);

    /* Compute the inverse of the preconditioner diagonal elements. */
    let pelinv = ONE / (c_j + FOUR * coeff);

    {
        let mut ppv = N_VGetArrayPointer(&pp).expect("N_VGetArrayPointer");

        for j in 1..(mm - 1) {
            let offset = mm * j;
            for i in 1..(mm - 1) {
                let loc = (offset + i) as usize;
                ppv[loc] = pelinv;
            }
        }
    }

    0
}

/*
 * PsolveHeat: solve preconditioner linear system.
 * This routine multiplies the input vector rvec by the vector pp
 * containing the inverse diagonal Jacobian elements (previously
 * computed in PrecondHeateq), returning the result in zvec.
 */

#[allow(clippy::too_many_arguments)]
fn PsolveHeat(
    _tt: sunrealtype,
    _uu: &N_Vector,
    _up: &N_Vector,
    _rr: &N_Vector,
    rvec: &N_Vector,
    zvec: &N_Vector,
    _c_j: sunrealtype,
    _delta: sunrealtype,
    prec_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    let pp = {
        let data = prec_data
            .as_mut()
            .and_then(|b| b.downcast_mut::<UserData>())
            .expect("prec_data is UserData");
        data.pp.as_ref().expect("pp").clone()
    };
    N_VProd(&pp, rvec, zvec);
    0
}

/*
 *--------------------------------------------------------------------
 * PRIVATE FUNCTIONS
 *--------------------------------------------------------------------
 */

/*
 * SetInitialProfile: routine to initialize u and up vectors.
 */

fn SetInitialProfile(
    data: &mut Option<Box<dyn Any>>,
    uu: &N_Vector,
    up: &N_Vector,
    res: &N_Vector,
) -> i32 {
    let (mm, dx) = {
        let d = data
            .as_mut()
            .and_then(|b| b.downcast_mut::<UserData>())
            .expect("user_data is UserData");
        (d.mm, d.dx)
    };

    /* Initialize uu on all grid points. */
    let mm1 = mm - 1;
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

    /* resHeat sets res to negative of ODE RHS values at interior points. */
    resHeat(ZERO, uu, up, res, data);

    /* Copy -res into up to get correct interior initial up values. */
    N_VScale(-ONE, res, up);

    /* Set up at boundary points to zero. */
    {
        let mut updata = N_VGetArrayPointer(up).expect("N_VGetArrayPointer");

        for j in 0..mm {
            let offset = mm * j;
            for i in 0..mm {
                let loc = (offset + i) as usize;
                if j == 0 || j == mm1 || i == 0 || i == mm1 {
                    updata[loc] = ZERO;
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
    print!("\nidasHeat2D_kry: Heat equation, serial example problem for IDA \n");
    print!("                Discretized heat equation on 2D unit square. \n");
    print!("                Zero boundary conditions,");
    print!(" polynomial initial conditions.\n");
    print!("                Mesh dimensions: {} x {}", MGRID, MGRID);
    print!("     Total system size: {}\n\n", NEQ);
    print!(
        "Tolerance parameters:  rtol = {}   atol = {}\n",
        fmt_g(rtol, 6),
        fmt_g(atol, 6)
    );
    print!("Constraints set to force all solution components >= 0. \n");
    print!("Linear solver: SPGMR, preconditioner using diagonal elements. \n");
}

/*
 * PrintOutput: print max norm of solution and current solver statistics
 */

fn PrintOutput(mem: &IDAMem, t: sunrealtype, uu: &N_Vector) {
    let mut retval: i32;
    let mut hused: sunrealtype = 0.0;
    let mut nst: i64 = 0;
    let mut nni: i64 = 0;
    let mut nje: i64 = 0;
    let mut nre: i64 = 0;
    let mut nreLS: i64 = 0;
    let mut nli: i64 = 0;
    let mut npe: i64 = 0;
    let mut nps: i64 = 0;
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
    retval = IDAGetNumJtimesEvals(mem, &mut nje);
    check_retval(&retval, "IDAGetNumJtimesEvals");
    retval = IDAGetNumLinIters(mem, &mut nli);
    check_retval(&retval, "IDAGetNumLinIters");
    retval = IDAGetNumLinResEvals(mem, &mut nreLS);
    check_retval(&retval, "IDAGetNumLinResEvals");
    retval = IDAGetNumPrecEvals(mem, &mut npe);
    check_retval(&retval, "IDAGetPrecEvals");
    retval = IDAGetNumPrecSolves(mem, &mut nps);
    check_retval(&retval, "IDAGetNumPrecSolves");

    /* nli is fetched by the C example but never printed */
    let _ = nli;

    print!(
        " {} {}  {}  {:>3}  {:>3}  {:>3}  {:>4}  {:>4}  {}  {:>3} {:>3}\n",
        fmt_fw(t, 5, 2),
        fmt_ew(umax, 13, 5),
        kused,
        nst,
        nni,
        nje,
        nre,
        nreLS,
        fmt_ew(hused, 9, 2),
        npe,
        nps
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
