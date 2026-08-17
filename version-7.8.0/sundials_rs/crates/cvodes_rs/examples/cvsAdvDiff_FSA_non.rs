#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

/* -----------------------------------------------------------------
 * Programmer(s): Scott D. Cohen, Alan C. Hindmarsh, George D. Byrne,
 *              and Radu Serban @ LLNL
 * -----------------------------------------------------------------
 * Rust port of examples/cvodes/serial/cvsAdvDiff_FSA_non.c
 * -----------------------------------------------------------------
 * Example problem:
 *
 * The following is a simple example problem, with the program for
 * its solution by CVODES. The problem is the semi-discrete form of
 * the advection-diffusion equation in 1-D:
 *   du/dt = q1 * d^2 u / dx^2 + q2 * du/dx
 * on the interval 0 <= x <= 2, and the time interval 0 <= t <= 5.
 * Homogeneous Dirichlet boundary conditions are posed, and the
 * initial condition is:
 *   u(x,y,t=0) = x(2-x)exp(2x).
 * The PDE is discretized on a uniform grid of size MX+2 with
 * central differencing, and with boundary values eliminated,
 * leaving an ODE system of size NEQ = MX.
 * This program solves the problem with the option for nonstiff
 * systems: ADAMS method and functional iteration.
 * It uses scalar relative and absolute tolerances.
 * Output is printed at t = .5, 1.0, ..., 5.
 * Run statistics (optional outputs) are printed at the end.
 *
 * Optionally, CVODES can compute sensitivities with respect to the
 * problem parameters q1 and q2.
 * Any of three sensitivity methods (SIMULTANEOUS, STAGGERED, and
 * STAGGERED1) can be used and sensitivities may be included in the
 * error test or not (error control set on FULL or PARTIAL,
 * respectively).
 *
 * Execution:
 *
 * If no sensitivities are desired:
 *    % cvsAdvDiff_FSA_non -nosensi
 * If sensitivities are to be computed:
 *    % cvsAdvDiff_FSA_non -sensi sensi_meth err_con
 * where sensi_meth is one of {sim, stg, stg1} and err_con is one of
 * {t, f}.
 * -----------------------------------------------------------------*/

use cvodes_rs::prelude::*;

use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

/* Problem Constants */
const XMAX: sunrealtype = 2.0; /* domain boundary           */
const MX: usize = 10; /* mesh dimension            */
const NEQ: usize = MX; /* number of equations       */
const ATOL: sunrealtype = 1.0e-5; /* scalar absolute tolerance */
const T0: sunrealtype = 0.0; /* initial time              */
const T1: sunrealtype = 0.5; /* first output time         */
const DTOUT: sunrealtype = 0.5; /* output time increment     */
const NOUT: i32 = 10; /* number of output times    */

const NP: usize = 2;
const NS: i32 = 2;

const ZERO: sunrealtype = 0.0;

/* Type : UserData
contains problem parameters, grid constants, work array. */

/* C stores `data->p` (the caller's own array) in `cv_mem->cv_p` via
CVodeSetSensParams, so the internal difference-quotient sensitivity RHS
perturbs the very array that `f` reads back through `user_data`. The port
shares that array as a `SensParams` handle: `main` hands CVODES a clone of
this very `Rc`, so the perturbations land here, exactly as in C. */
struct UserData {
    p: SensParams,
    dx: sunrealtype,
}

/*
 *--------------------------------------------------------------------
 * MAIN PROGRAM
 *--------------------------------------------------------------------
 */

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let argc = argv.len() as i32;

    let mut t: sunrealtype = ZERO;

    let mut sensi: sunbooleantype = false;
    let mut err_con: sunbooleantype = false;
    let mut sensi_meth: i32 = -1;

    /* Process arguments */
    ProcessArgs(argc, &argv, &mut sensi, &mut sensi_meth, &mut err_con);

    /* Create SUNDIALS context */
    let mut sunctx: Option<SUNContext> = None;
    let retval = SUNContext_Create(SUN_COMM_NULL, &mut sunctx);
    if check_retval(&retval, "SUNContext_Create") != 0 {
        std::process::exit(1);
    }
    let ctx = sunctx.as_ref().expect("SUNContext_Create").clone();

    /* Set user data */
    let data = UserData {
        p: Rc::new(RefCell::new(vec![ZERO; NP])), /* Allocate data memory */
        dx: XMAX / ((MX + 1) as sunrealtype),
    };
    if check_ptr(&Some(&data), "malloc", 2) != 0 {
        std::process::exit(1);
    }
    let dx = data.dx;
    {
        let mut p = data.p.borrow_mut();
        p[0] = 1.0;
        p[1] = 0.5;
    }

    /* Allocate and set initial states */
    let u = N_VNew_Serial(NEQ as sunindextype, &ctx);
    if check_ptr(&u, "N_VNew_Serial", 0) != 0 {
        std::process::exit(1);
    }
    let u = u.expect("N_VNew_Serial");
    SetIC(&u, dx);

    /* Set integration tolerances */
    let reltol = ZERO;
    let abstol = ATOL;

    /* Create CVODES object */
    let cvode_mem = CVodeCreate(CV_ADAMS, &ctx);
    if check_ptr(&cvode_mem, "CVodeCreate", 0) != 0 {
        std::process::exit(1);
    }
    let cvode_mem = cvode_mem.expect("CVodeCreate");

    /* Keep a handle on the parameter array before ownership of `data` moves
    into the solver memory: this clone IS `data->p` (C keeps its own `data`
    pointer and hands that same array to CVODES). */
    let p: SensParams = data.p.clone();

    let retval = CVodeSetUserData(&cvode_mem, Some(Box::new(data)));
    if check_retval(&retval, "CVodeSetUserData") != 0 {
        std::process::exit(1);
    }

    /* Allocate CVODES memory */
    let retval = CVodeInit(&cvode_mem, f, T0, &u);
    if check_retval(&retval, "CVodeInit") != 0 {
        std::process::exit(1);
    }

    let retval = CVodeSStolerances(&cvode_mem, reltol, abstol);
    if check_retval(&retval, "CVodeSStolerances") != 0 {
        std::process::exit(1);
    }

    /* create fixed point nonlinear solver object */
    let NLS = SUNNonlinSol_FixedPoint(&u, 0, &ctx);
    if check_ptr(&NLS, "SUNNonlinSol_FixedPoint", 0) != 0 {
        std::process::exit(1);
    }
    let NLS = NLS.expect("SUNNonlinSol_FixedPoint");

    /* attach nonlinear solver object to CVode */
    let retval = CVodeSetNonlinearSolver(&cvode_mem, &NLS);
    if check_retval(&retval, "CVodeSetNonlinearSolver") != 0 {
        std::process::exit(1);
    }

    print!("\n1-D advection-diffusion equation, mesh size ={:3}\n", MX);

    /* Sensitivity-related settings */
    let mut uS: Option<Vec<N_Vector>> = None;
    let mut NLSsens: Option<SUNNonlinearSolver> = None;
    if sensi {
        let mut plist: Vec<i32> = vec![0; NS as usize];
        if check_ptr(&Some(&plist), "malloc", 2) != 0 {
            std::process::exit(1);
        }
        for is in 0..NS as usize {
            plist[is] = is as i32;
        }

        let mut pbar: Vec<sunrealtype> = vec![ZERO; NS as usize];
        if check_ptr(&Some(&pbar), "malloc", 2) != 0 {
            std::process::exit(1);
        }
        for is in 0..NS as usize {
            pbar[is] = p.borrow()[plist[is] as usize];
        }

        let uSv = N_VCloneVectorArray(NS, &u);
        if check_ptr(&uSv, "N_VCloneVectorArray", 0) != 0 {
            std::process::exit(1);
        }
        let uSv = uSv.expect("N_VCloneVectorArray");
        for is in 0..NS as usize {
            N_VConst(ZERO, &uSv[is]);
        }

        let retval = CVodeSensInit1(&cvode_mem, NS, sensi_meth, None, &uSv);
        if check_retval(&retval, "CVodeSensInit1") != 0 {
            std::process::exit(1);
        }

        let retval = CVodeSensEEtolerances(&cvode_mem);
        if check_retval(&retval, "CVodeSensEEtolerances") != 0 {
            std::process::exit(1);
        }

        let retval = CVodeSetSensErrCon(&cvode_mem, err_con);
        if check_retval(&retval, "CVodeSetSensErrCon") != 0 {
            std::process::exit(1);
        }

        let retval = CVodeSetSensDQMethod(&cvode_mem, CV_CENTERED, ZERO);
        if check_retval(&retval, "CVodeSetSensDQMethod") != 0 {
            std::process::exit(1);
        }

        /* Hand CVODES a CLONE of the handle `data` keeps (C hands it the
        `data->p` pointer): the internal DQ perturbations then reach `f`
        through `user_data`. */
        let retval = CVodeSetSensParams(&cvode_mem, Some(p.clone()), Some(&pbar), Some(&plist));
        if check_retval(&retval, "CVodeSetSensParams") != 0 {
            std::process::exit(1);
        }

        /* create sensitivity fixed point nonlinear solver object */
        let NLSs = if sensi_meth == CV_SIMULTANEOUS {
            SUNNonlinSol_FixedPointSens(NS + 1, &u, 0, &ctx)
        } else if sensi_meth == CV_STAGGERED {
            SUNNonlinSol_FixedPointSens(NS, &u, 0, &ctx)
        } else {
            SUNNonlinSol_FixedPoint(&u, 0, &ctx)
        };
        /* NOTE: the C source checks NLS here, not NLSsens; kept verbatim. */
        if check_ptr(&Some(&NLS), "SUNNonlinSol_FixedPoint", 0) != 0 {
            std::process::exit(1);
        }
        let NLSs = NLSs.expect("SUNNonlinSol_FixedPoint");

        /* attach nonlinear solver object to CVode */
        let retval = if sensi_meth == CV_SIMULTANEOUS {
            CVodeSetNonlinearSolverSensSim(&cvode_mem, &NLSs)
        } else if sensi_meth == CV_STAGGERED {
            CVodeSetNonlinearSolverSensStg(&cvode_mem, &NLSs)
        } else {
            CVodeSetNonlinearSolverSensStg1(&cvode_mem, &NLSs)
        };
        if check_retval(&retval, "CVodeSetNonlinearSolver") != 0 {
            std::process::exit(1);
        }

        print!("Sensitivity: YES ");
        if sensi_meth == CV_SIMULTANEOUS {
            print!("( SIMULTANEOUS +");
        } else if sensi_meth == CV_STAGGERED {
            print!("( STAGGERED +");
        } else {
            print!("( STAGGERED1 +");
        }
        if err_con {
            print!(" FULL ERROR CONTROL )");
        } else {
            print!(" PARTIAL ERROR CONTROL )");
        }

        uS = Some(uSv);
        NLSsens = Some(NLSs);
    } else {
        print!("Sensitivity: NO ");
    }

    /* In loop over output points, call CVode, print results, test for error */

    print!("\n\n");
    print!("============================================================\n");
    print!("     T     Q       H      NST                    Max norm   \n");
    print!("============================================================\n");

    let mut tout = T1;
    for _iout in 1..=NOUT {
        let retval = CVode(&cvode_mem, tout, &u, &mut t, CV_NORMAL);
        if check_retval(&retval, "CVode") != 0 {
            break;
        }
        PrintOutput(&cvode_mem, t, &u);
        if sensi {
            let uSv = uS.as_ref().expect("uS allocated");
            let retval = CVodeGetSens(&cvode_mem, &mut t, uSv);
            if check_retval(&retval, "CVodeGetSens") != 0 {
                break;
            }
            PrintOutputS(uSv);
        }
        print!("------------------------------------------------------------\n");

        tout += DTOUT;
    }

    /* Print final statistics */
    PrintFinalStats(&cvode_mem, sensi, err_con, sensi_meth);

    /* Free memory */
    N_VDestroy(u);
    if sensi {
        N_VDestroyVectorArray(uS.expect("uS allocated"), NS);
        /* plist and pbar are dropped with their scopes */
    }
    /* data (and data->p) are dropped with the solver memory */
    let mut cvode_mem = Some(cvode_mem);
    CVodeFree(&mut cvode_mem);
    SUNNonlinSolFree(Some(NLS));
    if sensi {
        SUNNonlinSolFree(NLSsens);
    }
    SUNContext_Free(&mut sunctx);
}

/*
 *--------------------------------------------------------------------
 * FUNCTIONS CALLED BY CVODES
 *--------------------------------------------------------------------
 */

/*
 * f routine. Compute f(t,u).
 */

fn f(_t: sunrealtype, u: &N_Vector, udot: &N_Vector, user_data: &mut Option<Box<dyn Any>>) -> i32 {
    let mut ui: sunrealtype;
    let mut ult: sunrealtype;
    let mut urt: sunrealtype;
    let mut hdiff: sunrealtype;
    let mut hadv: sunrealtype;

    let udata = N_VGetArrayPointer(u).expect("N_VGetArrayPointer");
    let mut dudata = N_VGetArrayPointer(udot).expect("N_VGetArrayPointer");

    /* Extract needed problem constants from data */
    let data = user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<UserData>())
        .expect("user_data is UserData");
    let dx = data.dx;
    /* Snapshot of `data->p` as the C callback reads it (`data->p[0]`,
    `data->p[1]`). While the internal DQ sensitivity RHS is running, the
    entry for the active parameter carries the perturbation CVODES just
    wrote through the shared handle. */
    let (p0, p1) = {
        let p = data.p.borrow();
        (p[0], p[1])
    };
    let hordc = p0 / (dx * dx);
    let horac = p1 / (2.0 * dx);

    /* Loop over all grid points. */
    for i in 0..NEQ {
        /* Extract u at x_i and two neighboring points */
        ui = udata[i];
        if i != 0 {
            ult = udata[i - 1];
        } else {
            ult = ZERO;
        }
        if i != NEQ - 1 {
            urt = udata[i + 1];
        } else {
            urt = ZERO;
        }

        /* Set diffusion and advection terms and load into udot */
        hdiff = hordc * (ult - 2.0 * ui + urt);
        hadv = horac * (urt - ult);
        dudata[i] = hdiff + hadv;
    }

    0
}

/*
 *--------------------------------------------------------------------
 * PRIVATE FUNCTIONS
 *--------------------------------------------------------------------
 */

/*
 * Process and verify arguments.
 */

fn ProcessArgs(
    argc: i32,
    argv: &[String],
    sensi: &mut sunbooleantype,
    sensi_meth: &mut i32,
    err_con: &mut sunbooleantype,
) {
    *sensi = false;
    *sensi_meth = -1;
    *err_con = false;

    if argc < 2 {
        WrongArgs(&argv[0]);
    }

    if argv[1] == "-nosensi" {
        *sensi = false;
    } else if argv[1] == "-sensi" {
        *sensi = true;
    } else {
        WrongArgs(&argv[0]);
    }

    if *sensi {
        if argc != 4 {
            WrongArgs(&argv[0]);
        }

        if argv[2] == "sim" {
            *sensi_meth = CV_SIMULTANEOUS;
        } else if argv[2] == "stg" {
            *sensi_meth = CV_STAGGERED;
        } else if argv[2] == "stg1" {
            *sensi_meth = CV_STAGGERED1;
        } else {
            WrongArgs(&argv[0]);
        }

        if argv[3] == "t" {
            *err_con = true;
        } else if argv[3] == "f" {
            *err_con = false;
        } else {
            WrongArgs(&argv[0]);
        }
    }
}

fn WrongArgs(name: &str) -> ! {
    print!("\nUsage: {} [-nosensi] [-sensi sensi_meth err_con]\n", name);
    print!("         sensi_meth = sim, stg, or stg1\n");
    print!("         err_con    = t or f\n");

    std::process::exit(0);
}

/*
 * Set initial conditions in u vector.
 */

fn SetIC(u: &N_Vector, dx: sunrealtype) {
    let mut x: sunrealtype;

    /* Set pointer to data array and get local length of u. */
    let mut udata = N_VGetArrayPointer(u).expect("N_VGetArrayPointer");

    /* Load initial profile into u vector */
    for i in 0..NEQ {
        x = ((i + 1) as sunrealtype) * dx;
        udata[i] = x * (XMAX - x) * SUNRexp(2.0 * x);
    }
}

/*
 * Print current t, step count, order, stepsize, and max norm of solution
 */

fn PrintOutput(cvode_mem: &CVodeMem, t: sunrealtype, u: &N_Vector) {
    let mut nst: i64 = 0;
    let mut qu: i32 = 0;
    let mut hu: sunrealtype = ZERO;

    let retval = CVodeGetNumSteps(cvode_mem, &mut nst);
    check_retval(&retval, "CVodeGetNumSteps");
    let retval = CVodeGetLastOrder(cvode_mem, &mut qu);
    check_retval(&retval, "CVodeGetLastOrder");
    let retval = CVodeGetLastStep(cvode_mem, &mut hu);
    check_retval(&retval, "CVodeGetLastStep");

    print!(
        "{} {:2}  {} {:5}\n",
        fmt_ew(t, 8, 3),
        qu,
        fmt_ew(hu, 8, 3),
        nst
    );

    print!("                                Solution       ");

    print!("{} \n", fmt_ew(N_VMaxNorm(u), 12, 4));
}

/*
 * Print max norm of sensitivities
 */

fn PrintOutputS(uS: &[N_Vector]) {
    print!("                                Sensitivity 1  ");
    print!("{} \n", fmt_ew(N_VMaxNorm(&uS[0]), 12, 4));

    print!("                                Sensitivity 2  ");
    print!("{} \n", fmt_ew(N_VMaxNorm(&uS[1]), 12, 4));
}

/*
 * Print some final statistics located in the CVODES memory
 */

fn PrintFinalStats(
    cvode_mem: &CVodeMem,
    sensi: sunbooleantype,
    err_con: sunbooleantype,
    sensi_meth: i32,
) {
    let mut nst: i64 = 0;
    let (mut nfe, mut nsetups, mut nni, mut ncfn, mut netf): (i64, i64, i64, i64, i64) =
        (0, 0, 0, 0, 0);
    let (mut nfSe, mut nfeS, mut nsetupsS, mut nniS, mut ncfnS, mut netfS): (
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
    ) = (0, 0, 0, 0, 0, 0);

    let retval = CVodeGetNumSteps(cvode_mem, &mut nst);
    check_retval(&retval, "CVodeGetNumSteps");
    let retval = CVodeGetNumRhsEvals(cvode_mem, &mut nfe);
    check_retval(&retval, "CVodeGetNumRhsEvals");
    let retval = CVodeGetNumLinSolvSetups(cvode_mem, &mut nsetups);
    check_retval(&retval, "CVodeGetNumLinSolvSetups");
    let retval = CVodeGetNumErrTestFails(cvode_mem, &mut netf);
    check_retval(&retval, "CVodeGetNumErrTestFails");
    let retval = CVodeGetNumNonlinSolvIters(cvode_mem, &mut nni);
    check_retval(&retval, "CVodeGetNumNonlinSolvIters");
    let retval = CVodeGetNumNonlinSolvConvFails(cvode_mem, &mut ncfn);
    check_retval(&retval, "CVodeGetNumNonlinSolvConvFails");

    if sensi {
        let retval = CVodeGetSensNumRhsEvals(cvode_mem, &mut nfSe);
        check_retval(&retval, "CVodeGetSensNumRhsEvals");
        let retval = CVodeGetNumRhsEvalsSens(cvode_mem, &mut nfeS);
        check_retval(&retval, "CVodeGetNumRhsEvalsSens");
        let retval = CVodeGetSensNumLinSolvSetups(cvode_mem, &mut nsetupsS);
        check_retval(&retval, "CVodeGetSensNumLinSolvSetups");
        if err_con {
            let retval = CVodeGetSensNumErrTestFails(cvode_mem, &mut netfS);
            check_retval(&retval, "CVodeGetSensNumErrTestFails");
        } else {
            netfS = 0;
        }
        if (sensi_meth == CV_STAGGERED) || (sensi_meth == CV_STAGGERED1) {
            let retval = CVodeGetSensNumNonlinSolvIters(cvode_mem, &mut nniS);
            check_retval(&retval, "CVodeGetSensNumNonlinSolvIters");
            let retval = CVodeGetSensNumNonlinSolvConvFails(cvode_mem, &mut ncfnS);
            check_retval(&retval, "CVodeGetSensNumNonlinSolvConvFails");
        } else {
            nniS = 0;
            ncfnS = 0;
        }
    }

    print!("\nFinal Statistics\n\n");
    print!("nst     = {:5}\n\n", nst);
    print!("nfe     = {:5}\n", nfe);
    print!("netf    = {:5}    nsetups  = {:5}\n", netf, nsetups);
    print!("nni     = {:5}    ncfn     = {:5}\n", nni, ncfn);

    if sensi {
        print!("\n");
        print!("nfSe    = {:5}    nfeS     = {:5}\n", nfSe, nfeS);
        print!("netfs   = {:5}    nsetupsS = {:5}\n", netfS, nsetupsS);
        print!("nniS    = {:5}    ncfnS    = {:5}\n", nniS, ncfnS);
    }
}

/*
 * Check function return value...
 *   check_ptr (C opt == 0) means SUNDIALS function allocates memory so
 *            check if returned NULL pointer
 *   check_retval (C opt == 1) means SUNDIALS function returns an integer
 *            value so check if retval < 0
 *   check_ptr (C opt == 2) means function allocates memory so check if
 *            returned NULL pointer
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
