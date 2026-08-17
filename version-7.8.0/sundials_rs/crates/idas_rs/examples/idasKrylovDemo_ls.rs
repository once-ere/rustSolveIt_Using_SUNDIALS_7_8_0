/* -----------------------------------------------------------------
 * Programmer(s): Allan Taylor, Alan Hindmarsh and
 *                Radu Serban @ LLNL
 * -----------------------------------------------------------------
 * Rust port of examples/idas/serial/idasKrylovDemo_ls.c
 * -----------------------------------------------------------------
 *
 * This example loops through the available iterative linear solvers:
 * SPGMR, SPBCG and SPTFQMR.
 *
 * Example problem for IDA: 2D heat equation, serial, GMRES.
 *
 * This example solves a discretized 2D heat equation problem.
 * This version loops through the Krylov solvers Spgmr, Spbcg
 * and Sptfqmr.
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
 * The system is solved with IDA using the following Krylov
 * linear solvers: SPGMR, SPBCG and SPTFQMR. The
 * preconditioner uses the diagonal elements of the Jacobian only.
 * Routines for preconditioning, required by SP*, are supplied
 * here. The constraints u >= 0 are posed for all components. Output
 * is taken at t = 0, .01, .02, .04,..., 10.24.
 * -----------------------------------------------------------------*/

#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

use std::any::Any;

use idas_rs::prelude::*;

/* Problem Constants */

const NOUT: i32 = 11;
const MGRID: i32 = 10;
const NEQ: i32 = MGRID * MGRID;
const ZERO: sunrealtype = 0.0;
const ONE: sunrealtype = 1.0;
const TWO: sunrealtype = 2.0;
const FOUR: sunrealtype = 4.0;

/* Linear Solver Loop Constants */

const USE_SPGMR: i32 = 0;
const USE_SPBCG: i32 = 1;
const USE_SPTFQMR: i32 = 2;

/* User data type */

struct UserData {
    mm: sunindextype, /* number of grid points */
    dx: sunrealtype,
    coeff: sunrealtype,
    pp: N_Vector, /* vector of prec. diag. elements */
}

/*
 *--------------------------------------------------------------------
 * MAIN PROGRAM
 *--------------------------------------------------------------------
 */

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let argc = argv.len() as i32;

    let mut retval: i32;
    let mut tret: sunrealtype = ZERO;
    let (mut netf, mut ncfn, mut ncfl) = (0i64, 0i64, 0i64);
    let mut LS: Option<SUNLinearSolver> = None;
    let mut nrmfactor: i32 = 0; /* LS norm conversion factor flag */

    /* Retrieve the command-line options */
    if argc > 1 {
        nrmfactor = argv[1].trim().parse::<i32>().unwrap_or(0);
    }

    /* Create the SUNDIALS context object for this simulation */
    let mut sunctx: Option<SUNContext> = None;
    retval = SUNContext_Create(SUN_COMM_NULL, &mut sunctx);
    if check_retval(Some(retval), "SUNContext_Create", 1) != 0 {
        std::process::exit(1);
    }
    let ctx = sunctx.as_ref().expect("sunctx").clone();

    /* Allocate N-vectors and the user data structure. */

    let uu = N_VNew_Serial(NEQ as sunindextype, &ctx);
    if check_retval(uu.as_ref().map(|_| 0), "N_VNew_Serial", 0) != 0 {
        std::process::exit(1);
    }
    let uu = uu.expect("N_VNew_Serial");

    let up = N_VClone(&uu);
    if check_retval(up.as_ref().map(|_| 0), "N_VNew_Serial", 0) != 0 {
        std::process::exit(1);
    }
    let up = up.expect("N_VClone");

    let res = N_VClone(&uu);
    if check_retval(res.as_ref().map(|_| 0), "N_VNew_Serial", 0) != 0 {
        std::process::exit(1);
    }
    let res = res.expect("N_VClone");

    let constraints = N_VClone(&uu);
    if check_retval(constraints.as_ref().map(|_| 0), "N_VNew_Serial", 0) != 0 {
        std::process::exit(1);
    }
    let constraints = constraints.expect("N_VClone");

    /* Assign parameters in the user data structure. */

    let pp = N_VClone(&uu);
    if check_retval(pp.as_ref().map(|_| 0), "N_VNew_Serial", 0) != 0 {
        std::process::exit(1);
    }
    let pp = pp.expect("N_VClone");

    let dx = ONE / ((MGRID as sunrealtype) - ONE);
    let mut data: Option<Box<dyn Any>> = Some(Box::new(UserData {
        mm: MGRID as sunindextype,
        dx,
        coeff: ONE / (dx * dx),
        pp,
    }));

    /* Initialize uu, up. */

    SetInitialProfile(&mut data, &uu, &up, &res);

    /* Set constraints to all 1's for nonnegative solution values. */

    N_VConst(ONE, &constraints);

    /* Assign various parameters. */

    let t0: sunrealtype = ZERO;
    let t1: sunrealtype = 0.01;
    let rtol: sunrealtype = ZERO;
    let atol: sunrealtype = 1.0e-3;

    /* Call IDACreate and IDAMalloc to initialize solution */

    let mut mem_opt = IDACreate(&ctx);
    if check_retval(mem_opt.as_ref().map(|_| 0), "IDACreate", 0) != 0 {
        std::process::exit(1);
    }
    let mem = mem_opt.as_ref().expect("IDACreate").clone();

    retval = IDASetUserData(&mem, data);
    if check_retval(Some(retval), "IDASetUserData", 1) != 0 {
        std::process::exit(1);
    }

    retval = IDASetConstraints(&mem, Some(&constraints));
    if check_retval(Some(retval), "IDASetConstraints", 1) != 0 {
        std::process::exit(1);
    }
    N_VDestroy(constraints);

    retval = IDAInit(&mem, resHeat, t0, &uu, &up);
    if check_retval(Some(retval), "IDAInit", 1) != 0 {
        std::process::exit(1);
    }

    retval = IDASStolerances(&mem, rtol, atol);
    if check_retval(Some(retval), "IDASStolerances", 1) != 0 {
        std::process::exit(1);
    }

    /* START: Loop through SPGMR, SPBCG and SPTFQMR linear solver modules */
    for linsolver in 0..3 {
        if linsolver != 0 {
            /* Re-initialize uu, up.

            C reaches the user data through the pointer it still owns; the
            port borrows the box back out of the solver memory for the
            duration of the call and hands it straight back. */
            {
                let mut ud: Option<Box<dyn Any>> = None;
                IDAGetUserData(&mem, &mut ud);
                SetInitialProfile(&mut ud, &uu, &up, &res);
                IDASetUserData(&mem, ud);
            }

            /* Re-initialize IDA */
            retval = IDAReInit(&mem, t0, &uu, &up);
            if check_retval(Some(retval), "IDAReInit", 1) != 0 {
                std::process::exit(1);
            }
        }

        /* Free previous linear solver and attach a new linear solver module */
        let _ = SUNLinSolFree(LS.take());

        match linsolver {
            /* (a) SPGMR */
            USE_SPGMR => {
                /* Print header */
                print!(" -------");
                print!(" \n| SPGMR |\n");
                print!(" -------\n");

                /* Call SUNLinSol_SPGMR to specify the linear solver SPGMR with
                left preconditioning and the default maximum Krylov dimension */
                LS = SUNLinSol_SPGMR(&uu, SUN_PREC_LEFT, 0, &ctx);
                if check_retval(LS.as_ref().map(|_| 0), "SUNLinSol_SPGMR", 0) != 0 {
                    std::process::exit(1);
                }

                /* Attach the linear solver */
                retval = IDASetLinearSolver(&mem, LS.as_ref().expect("LS"), None);
                if check_retval(Some(retval), "IDASetLinearSolver", 1) != 0 {
                    std::process::exit(1);
                }
            }

            /* (b) SPBCG */
            USE_SPBCG => {
                /* Print header */
                print!(" -------");
                print!(" \n| SPBCGS |\n");
                print!(" -------\n");

                /* Call SUNLinSol_SPBCGS to specify the linear solver SPBCGS with
                left preconditioning and the default maximum Krylov dimension */
                LS = SUNLinSol_SPBCGS(&uu, SUN_PREC_LEFT, 0, &ctx);
                if check_retval(LS.as_ref().map(|_| 0), "SUNLinSol_SPBCGS", 0) != 0 {
                    std::process::exit(1);
                }

                /* Attach the linear solver */
                retval = IDASetLinearSolver(&mem, LS.as_ref().expect("LS"), None);
                if check_retval(Some(retval), "IDASetLinearSolver", 1) != 0 {
                    std::process::exit(1);
                }
            }

            /* (c) SPTFQMR */
            USE_SPTFQMR => {
                /* Print header */
                print!(" ---------");
                print!(" \n| SPTFQMR |\n");
                print!(" ---------\n");

                /* Call SUNLinSol_SPTFQMR to specify the linear solver SPTFQMR with
                left preconditioning and the default maximum Krylov dimension */
                LS = SUNLinSol_SPTFQMR(&uu, SUN_PREC_LEFT, 0, &ctx);
                if check_retval(LS.as_ref().map(|_| 0), "SUNLinSol_SPTFQMR", 0) != 0 {
                    std::process::exit(1);
                }

                /* Attach the linear solver */
                retval = IDASetLinearSolver(&mem, LS.as_ref().expect("LS"), None);
                if check_retval(Some(retval), "IDASetLinearSolver", 1) != 0 {
                    std::process::exit(1);
                }
            }

            _ => {}
        }

        /* Specify preconditioner */
        retval = IDASetPreconditioner(&mem, Some(PsetupHeat), Some(PsolveHeat));
        if check_retval(Some(retval), "IDASetPreconditioner", 1) != 0 {
            std::process::exit(1);
        }

        /* Set the linear solver tolerance conversion factor */
        let nrmfac: sunrealtype; /* LS norm conversion factor */
        match nrmfactor {
            1 => {
                /* use the square root of the vector length */
                nrmfac = (NEQ as sunrealtype).sqrt();
            }
            2 => {
                /* compute with dot product */
                nrmfac = -ONE;
            }
            _ => {
                /* use the default */
                nrmfac = ZERO;
            }
        }

        retval = IDASetLSNormFactor(&mem, nrmfac);
        if check_retval(Some(retval), "IDASetLSNormFactor", 1) != 0 {
            std::process::exit(1);
        }

        /* Print output heading. */
        PrintHeader(rtol, atol, linsolver);

        /* Print output table heading, and initial line of table. */

        print!("\n   Output Summary (umax = max-norm of solution) \n\n");
        print!("  time     umax       k  nst  nni  nje   nre   nreLS    h      npe nps\n");
        print!("----------------------------------------------------------------------\n");

        /* Loop over output times, call IDASolve, and print results. */

        let mut tout = t1;
        for _iout in 1..=NOUT {
            retval = IDASolve(&mem, tout, &mut tret, &uu, &up, IDA_NORMAL);
            if check_retval(Some(retval), "IDASolve", 1) != 0 {
                std::process::exit(1);
            }
            PrintOutput(&mem, tret, &uu, linsolver);
            tout *= TWO;
        }

        /* Print remaining counters. */
        retval = IDAGetNumErrTestFails(&mem, &mut netf);
        check_retval(Some(retval), "IDAGetNumErrTestFails", 1);

        retval = IDAGetNumNonlinSolvConvFails(&mem, &mut ncfn);
        check_retval(Some(retval), "IDAGetNumNonlinSolvConvFails", 1);

        retval = IDAGetNumLinConvFails(&mem, &mut ncfl);
        check_retval(Some(retval), "IDAGetNumLinConvFails", 1);

        print!("\nError test failures            = {}\n", netf);
        print!("Nonlinear convergence failures = {}\n", ncfn);
        print!("Linear convergence failures    = {}\n", ncfl);

        if linsolver < 2 {
            print!("\n======================================================================\n\n");
        }
    } /* END: Loop through SPGMR, SPBCG and SPTFQMR linear solver modules */

    /* Free Memory */

    IDAFree(&mut mem_opt);
    let _ = SUNLinSolFree(LS.take());

    N_VDestroy(uu);
    N_VDestroy(up);
    N_VDestroy(res);

    /* data (and the preconditioner vector pp it owns) is dropped together
    with the solver memory */

    let _ = SUNContext_Free(&mut sunctx);
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

    let uu_data = N_VGetArrayPointer(uu).expect("N_VGetArrayPointer");
    let up_data = N_VGetArrayPointer(up).expect("N_VGetArrayPointer");
    let mut rr_data = N_VGetArrayPointer(rr).expect("N_VGetArrayPointer");

    /* Loop over interior points; set res = up - (central difference). */
    let mmu = mm as usize;
    for j in 1..(MGRID as sunindextype - 1) {
        let offset = mm * j;
        for i in 1..(mm - 1) {
            let loc = (offset + i) as usize;
            let dif1 = uu_data[loc - 1] + uu_data[loc + 1] - TWO * uu_data[loc];
            let dif2 = uu_data[loc - mmu] + uu_data[loc + mmu] - TWO * uu_data[loc];
            rr_data[loc] = up_data[loc] - coeff * (dif1 + dif2);
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
    user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    let data = user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<UserData>())
        .expect("user_data is UserData");
    let mm = data.mm;

    /* Initialize the entire vector to 1., then set the interior points to the
    correct value for preconditioning. */
    N_VConst(ONE, &data.pp);

    /* Compute the inverse of the preconditioner diagonal elements. */
    let pelinv = ONE / (c_j + FOUR * data.coeff);

    let mut ppv = N_VGetArrayPointer(&data.pp).expect("N_VGetArrayPointer");

    for j in 1..(mm - 1) {
        let offset = mm * j;
        for i in 1..(mm - 1) {
            let loc = (offset + i) as usize;
            ppv[loc] = pelinv;
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

fn PsolveHeat(
    _tt: sunrealtype,
    _uu: &N_Vector,
    _up: &N_Vector,
    _rr: &N_Vector,
    rvec: &N_Vector,
    zvec: &N_Vector,
    _c_j: sunrealtype,
    _delta: sunrealtype,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    let data = user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<UserData>())
        .expect("user_data is UserData");
    N_VProd(&data.pp, rvec, zvec);
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
    user_data: &mut Option<Box<dyn Any>>,
    uu: &N_Vector,
    up: &N_Vector,
    res: &N_Vector,
) -> i32 {
    let (mm, dx) = {
        let data = user_data
            .as_mut()
            .and_then(|b| b.downcast_mut::<UserData>())
            .expect("user_data is UserData");
        (data.mm, data.dx)
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
    resHeat(ZERO, uu, up, res, user_data);

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

fn PrintHeader(rtol: sunrealtype, atol: sunrealtype, linsolver: i32) {
    print!("\nidasKrylovDemo_ls: Heat equation, serial example problem for IDA\n");
    print!("                   Discretized heat equation on 2D unit square.\n");
    print!("                   Zero boundary conditions,");
    print!(" polynomial initial conditions.\n");
    print!("                   Mesh dimensions: {} x {}", MGRID, MGRID);
    print!("       Total system size: {}\n\n", NEQ);
    print!(
        "Tolerance parameters:  rtol = {}   atol = {}\n",
        fmt_g(rtol, 6),
        fmt_g(atol, 6)
    );
    print!("Constraints set to force all solution components >= 0. \n");

    match linsolver {
        USE_SPGMR => {
            print!("Linear solver: SPGMR, preconditioner using diagonal elements. \n");
        }

        USE_SPBCG => {
            print!("Linear solver: SPBCG, preconditioner using diagonal elements. \n");
        }

        USE_SPTFQMR => {
            print!("Linear solver: SPTFQMR, preconditioner using diagonal elements. \n");
        }

        _ => {}
    }
}

/*
 * PrintOutput: print max norm of solution and current solver statistics
 */

fn PrintOutput(mem: &IDAMem, t: sunrealtype, uu: &N_Vector, _linsolver: i32) {
    let mut hused: sunrealtype = ZERO;
    let (mut nst, mut nni, mut nje, mut nre, mut nreLS, mut nli, mut npe, mut nps) =
        (0i64, 0i64, 0i64, 0i64, 0i64, 0i64, 0i64, 0i64);
    let mut kused: i32 = 0;
    let mut retval: i32;

    let umax = N_VMaxNorm(uu);

    retval = IDAGetLastOrder(mem, &mut kused);
    check_retval(Some(retval), "IDAGetLastOrder", 1);
    retval = IDAGetNumSteps(mem, &mut nst);
    check_retval(Some(retval), "IDAGetNumSteps", 1);
    retval = IDAGetNumNonlinSolvIters(mem, &mut nni);
    check_retval(Some(retval), "IDAGetNumNonlinSolvIters", 1);
    retval = IDAGetNumResEvals(mem, &mut nre);
    check_retval(Some(retval), "IDAGetNumResEvals", 1);
    retval = IDAGetLastStep(mem, &mut hused);
    check_retval(Some(retval), "IDAGetLastStep", 1);

    retval = IDAGetNumJtimesEvals(mem, &mut nje);
    check_retval(Some(retval), "IDAGetNumJtimesEvals", 1);
    retval = IDAGetNumLinIters(mem, &mut nli);
    check_retval(Some(retval), "IDAGetNumLinIters", 1);
    retval = IDAGetNumLinResEvals(mem, &mut nreLS);
    check_retval(Some(retval), "IDAGetNumLinResEvals", 1);
    retval = IDAGetNumPrecEvals(mem, &mut npe);
    check_retval(Some(retval), "IDAGetNumPrecEvals", 1);
    retval = IDAGetNumPrecSolves(mem, &mut nps);
    check_retval(Some(retval), "IDAGetNumPrecSolves", 1);

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

fn check_retval(returnvalue: Option<i32>, funcname: &str, opt: i32) -> i32 {
    /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
    if opt == 0 && returnvalue.is_none() {
        eprint!(
            "\nSUNDIALS_ERROR: {}() failed - returned NULL pointer\n\n",
            funcname
        );
        return 1;
    } else if opt == 1 {
        /* Check if retval < 0 */
        let retval = returnvalue.expect("retval");
        if retval < 0 {
            eprint!(
                "\nSUNDIALS_ERROR: {}() failed with retval = {}\n\n",
                funcname, retval
            );
            return 1;
        }
    } else if opt == 2 && returnvalue.is_none() {
        /* Check if function returned NULL pointer - no memory allocated */
        eprint!(
            "\nMEMORY_ERROR: {}() failed - returned NULL pointer\n\n",
            funcname
        );
        return 1;
    }

    0
}
