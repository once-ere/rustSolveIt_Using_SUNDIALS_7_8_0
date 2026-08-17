/* -----------------------------------------------------------------
 * Rust port of examples/cvodes/serial/cvsKrylovDemo_ls.c
 * -----------------------------------------------------------------
 * This example loops through the available iterative linear solvers:
 * SPGMR, SPFGMR, SPBCG and SPTFQMR.
 *
 * Example problem:
 *
 * An ODE system is generated from the following 2-species diurnal
 * kinetics advection-diffusion PDE system in 2 space dimensions.
 * The problem is solved with CVODE, with the BDF/GMRES, BDF/FGMRES
 * BDF/Bi-CGStab, and BDF/TFQMR methods (i.e. using the SUNLinSol_SPGMR,
 * SUNLinSol_SPFGMR, SUNLinSol_SPBCGS, and SUNLinSol_SPTFQMR linear
 * solvers) and the block-diagonal part of the Newton matrix as a left
 * preconditioner. A copy of the block-diagonal part of the Jacobian is
 * saved and conditionally reused within the Precond routine.
 * -----------------------------------------------------------------*/

#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

use cvodes_rs::prelude::*;
use cvodes_rs::sundials_dense::{
    SUNDlsMat_denseAddIdentity, SUNDlsMat_denseCopy, SUNDlsMat_denseGETRF, SUNDlsMat_denseGETRS,
    SUNDlsMat_denseScale,
};
use cvodes_rs::sundials_logger::{
    SUNLogger, SUNLogger_Create, SUNLogger_Destroy, SUNLogger_SetInfoFilename,
};

use std::any::Any;

/* helpful macros */

fn SQR(a: sunrealtype) -> sunrealtype {
    a * a
}

/* Problem Constants */

const ZERO: sunrealtype = 0.0;
const ONE: sunrealtype = 1.0;
const TWO: sunrealtype = 2.0;

const NUM_SPECIES: i32 = 2; /* number of species         */
const KH: sunrealtype = 4.0e-6; /* horizontal diffusivity Kh */
const VEL: sunrealtype = 0.001; /* advection velocity V      */
const KV0: sunrealtype = 1.0e-8; /* coefficient in Kv(y)      */
const Q1: sunrealtype = 1.63e-16; /* coefficients q1, q2, c3   */
const Q2: sunrealtype = 4.66e-16;
const C3: sunrealtype = 3.7e16;
const A3: sunrealtype = 22.62; /* coefficient in expression for q3(t) */
const A4: sunrealtype = 7.601; /* coefficient in expression for q4(t) */
const C1_SCALE: sunrealtype = 1.0e6; /* coefficients in initial profiles    */
const C2_SCALE: sunrealtype = 1.0e12;

const T0: sunrealtype = ZERO; /* initial time */
const NOUT: i32 = 12; /* number of output times */
const TWOHR: sunrealtype = 7200.0; /* number of seconds in two hours  */
const HALFDAY: sunrealtype = 4.32e4; /* number of seconds in a half day */
const PI: sunrealtype = 3.1415926535898; /* pi */

const XMIN: sunrealtype = ZERO; /* grid boundaries in x  */
const XMAX: sunrealtype = 20.0;
const YMIN: sunrealtype = 30.0; /* grid boundaries in y  */
const YMAX: sunrealtype = 50.0;
const XMID: sunrealtype = 10.0; /* grid midpoints in x,y */
const YMID: sunrealtype = 40.0;

const MX: i32 = 10; /* MX = number of x mesh points */
const MY: i32 = 10; /* MY = number of y mesh points */
const NSMX: i32 = 20; /* NSMX = NUM_SPECIES*MX */
const MM: i32 = MX * MY; /* MM = MX*MY */

/* CVodeInit Constants */

const RTOL: sunrealtype = 1.0e-5; /* scalar relative tolerance */
const FLOOR: sunrealtype = 100.0; /* value of C1 or C2 at which tolerances */
/* change from relative to absolute      */
const ATOL: sunrealtype = RTOL * FLOOR; /* scalar absolute tolerance */
const NEQ: i32 = NUM_SPECIES * MM; /* NEQ = number of equations */

/* Linear Solver Loop Constants */

const USE_SPGMR: i32 = 0;
const USE_SPFGMR: i32 = 1;
const USE_SPBCG: i32 = 2;
const USE_SPTFQMR: i32 = 3;

/* User-defined vector and matrix accessor helpers: IJKth, IJth.

IJKth(vdata,i,j,k) references the element in the vdata array for
species i at mesh point (j,k), where 1 <= i <= NUM_SPECIES,
0 <= j <= MX-1, 0 <= k <= MY-1. This helper returns the flat index:
vdata[i - 1 + j*NUM_SPECIES + k*NSMX].

IJth(a,i,j) references the (i,j)th entry of a small dense matrix
stored by column (a[j-1][i-1]), where 1 <= i,j <= NUM_SPECIES. */

fn IJKth(i: i32, j: i32, k: i32) -> usize {
    (i - 1 + j * NUM_SPECIES + k * NSMX) as usize
}

/* Type : UserData
contains preconditioner blocks, pivot arrays, and problem constants */

struct UserData {
    /* P[MX][MY] and Jbd[MX][MY]: 2x2 dense blocks stored by column */
    P: Vec<Vec<Vec<Vec<sunrealtype>>>>,
    Jbd: Vec<Vec<Vec<Vec<sunrealtype>>>>,
    pivot: Vec<Vec<Vec<sunindextype>>>,
    q4: sunrealtype,
    om: sunrealtype,
    dx: sunrealtype,
    dy: sunrealtype,
    hdco: sunrealtype,
    haco: sunrealtype,
    vdco: sunrealtype,
}

/*
 *-------------------------------
 * Main Program
 *-------------------------------
 */

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let abstol: sunrealtype;
    let reltol: sunrealtype;
    let mut t: sunrealtype = ZERO;
    let mut tout: sunrealtype;
    let mut retval: i32;
    let mut nrmfactor: i32 = 0; /* LS norm conversion factor flag */
    let mut nrmfac: sunrealtype; /* LS norm conversion factor      */
    let mut monitor: i32 = 0; /* LS resiudal monitoring flag    */
    let info_fname = "cvKrylovDemo_ls-info.txt";

    /* Retrieve the command-line options */
    if args.len() > 1 {
        nrmfactor = args[1].parse::<i32>().unwrap_or(0);
    }
    if args.len() > 2 {
        monitor = args[2].parse::<i32>().unwrap_or(0);
    }

    /* Create SUNDIALS context and a logger which will record
    nonlinear solver info (e.g., residual) amongst other things. */

    let mut sunctx_opt: Option<SUNContext> = None;
    retval = SUNContext_Create(SUN_COMM_NULL, &mut sunctx_opt);
    if check_retval(retval, "SUNContext_Create", 1) != 0 {
        std::process::exit(1);
    }
    let sunctx = sunctx_opt.clone().expect("SUNContext_Create");

    let mut logger_opt: Option<SUNLogger> = None;
    retval = SUNLogger_Create(SUN_COMM_NULL, 0, &mut logger_opt);
    if check_retval(retval, "SUNLogger_Create", 1) != 0 {
        std::process::exit(1);
    }
    let logger = logger_opt.clone().expect("SUNLogger_Create");

    retval = SUNLogger_SetInfoFilename(&logger, if monitor != 0 { Some(info_fname) } else { None });
    if check_retval(retval, "SUNLogger_SetInfoFilename", 1) != 0 {
        std::process::exit(1);
    }

    retval = SUNContext_SetLogger(&sunctx, Some(logger.clone()));
    if check_retval(retval, "SUNContext_SetLogger", 1) != 0 {
        std::process::exit(1);
    }

    /* Allocate memory, and set problem data, initial values, tolerances */
    let u_opt = N_VNew_Serial(NEQ as sunindextype, &sunctx);
    if check_null(&u_opt, "N_VNew_Serial", 0) != 0 {
        std::process::exit(1);
    }
    let u = u_opt.expect("N_VNew_Serial");
    let mut data = AllocUserData();
    InitUserData(&mut data);
    SetInitialProfiles(&u, data.dx, data.dy);
    abstol = ATOL;
    reltol = RTOL;

    /* Call CVodeCreate to create the solver memory and specify the
     * Backward Differentiation Formula */
    let cvode_mem_opt = CVodeCreate(CV_BDF, &sunctx);
    if check_null(&cvode_mem_opt, "CVodeCreate", 0) != 0 {
        std::process::exit(1);
    }
    let cvode_mem = cvode_mem_opt.expect("CVodeCreate");

    /* Set the pointer to user-defined data */
    retval = CVodeSetUserData(&cvode_mem, Some(Box::new(data)));
    if check_retval(retval, "CVodeSetUserData", 1) != 0 {
        std::process::exit(1);
    }

    /* Call CVodeInit to initialize the integrator memory and specify the
     * user's right hand side function in u'=f(t,u), the initial time T0, and
     * the initial dependent variable vector u. */
    retval = CVodeInit(&cvode_mem, f, T0, &u);
    if check_retval(retval, "CVodeInit", 1) != 0 {
        std::process::exit(1);
    }

    /* Call CVodeSStolerances to specify the scalar relative tolerance
     * and scalar absolute tolerances */
    retval = CVodeSStolerances(&cvode_mem, reltol, abstol);
    if check_retval(retval, "CVodeSStolerances", 1) != 0 {
        std::process::exit(1);
    }

    /* Create the SUNNonlinearSolver */
    let NLS = SUNNonlinSol_Newton(&u, &sunctx).expect("SUNNonlinSol_Newton");

    /* Call CVodeSetNonlinearSolver to attach the nonlinear solver to CVode */
    retval = CVodeSetNonlinearSolver(&cvode_mem, &NLS);
    if check_retval(retval, "CVodeSetNonlinearSolver", 1) != 0 {
        std::process::exit(1);
    }

    let mut LS: Option<SUNLinearSolver> = None;

    /* START: Loop through SPGMR, SPFGMR, SPBCG and SPTFQMR linear solver modules */
    for linsolver in 0..4 {
        if linsolver != 0 {
            /* Re-initialize user data */
            {
                let mut ud: Option<Box<dyn Any>> = None;
                CVodeGetUserData(&cvode_mem, &mut ud);
                {
                    let data = ud
                        .as_mut()
                        .and_then(|b| b.downcast_mut::<UserData>())
                        .expect("user_data downcast");
                    InitUserData(data);
                    let (dx, dy) = (data.dx, data.dy);
                    SetInitialProfiles(&u, dx, dy);
                }
                CVodeSetUserData(&cvode_mem, ud);
            }

            /* Re-initialize CVode for the solution of the same problem, but
            using a different linear solver module */
            retval = CVodeReInit(&cvode_mem, T0, &u);
            if check_retval(retval, "CVodeReInit", 1) != 0 {
                std::process::exit(1);
            }
        }

        /* Free previous linear solver and attach a new linear solver module */
        SUNLinSolFree(LS.take());

        match linsolver {
            /* (a) SPGMR */
            USE_SPGMR => {
                /* Print header */
                print!(" -------");
                print!(" \n| SPGMR |\n");
                print!(" -------\n");

                /* Call SUNLinSol_SPGMR to specify the linear solver SPGMR with
                left preconditioning and the default maximum Krylov dimension */
                LS = SUNLinSol_SPGMR(&u, SUN_PREC_LEFT, 0, &sunctx);
                if check_null(&LS, "SUNLinSol_SPGMR", 0) != 0 {
                    std::process::exit(1);
                }

                retval = CVodeSetLinearSolver(&cvode_mem, LS.as_ref().expect("LS"), None);
                if check_retval(retval, "CVodeSetLinearSolver", 1) != 0 {
                    std::process::exit(1);
                }
            }

            /* (b) SPFGMR */
            USE_SPFGMR => {
                /* Print header */
                print!(" ---------");
                print!(" \n| SPFGMR |\n");
                print!(" ---------\n");

                /* Call SUNLinSol_SPFGMR to specify the linear solver SPFGMR with
                left preconditioning and the default maximum Krylov dimension */
                LS = SUNLinSol_SPFGMR(&u, SUN_PREC_LEFT, 0, &sunctx);
                if check_null(&LS, "SUNLinSol_SPFGMR", 0) != 0 {
                    std::process::exit(1);
                }

                retval = CVodeSetLinearSolver(&cvode_mem, LS.as_ref().expect("LS"), None);
                if check_retval(retval, "CVodeSetLinearSolver", 1) != 0 {
                    std::process::exit(1);
                }
            }

            /* (c) SPBCG */
            USE_SPBCG => {
                /* Print header */
                print!(" -------");
                print!(" \n| SPBCGS |\n");
                print!(" -------\n");

                /* Call SUNLinSol_SPBCGS to specify the linear solver SPBCGS with
                left preconditioning and the default maximum Krylov dimension */
                LS = SUNLinSol_SPBCGS(&u, SUN_PREC_LEFT, 0, &sunctx);
                if check_null(&LS, "SUNLinSol_SPBCGS", 0) != 0 {
                    std::process::exit(1);
                }

                retval = CVodeSetLinearSolver(&cvode_mem, LS.as_ref().expect("LS"), None);
                if check_retval(retval, "CVodeSetLinearSolver", 1) != 0 {
                    std::process::exit(1);
                }
            }

            /* (d) SPTFQMR */
            USE_SPTFQMR => {
                /* Print header */
                print!(" ---------");
                print!(" \n| SPTFQMR |\n");
                print!(" ---------\n");

                /* Call SUNLinSol_SPTFQMR to specify the linear solver SPTFQMR with
                left preconditioning and the default maximum Krylov dimension */
                LS = SUNLinSol_SPTFQMR(&u, SUN_PREC_LEFT, 0, &sunctx);
                if check_null(&LS, "SUNLinSol_SPTFQMR", 0) != 0 {
                    std::process::exit(1);
                }

                retval = CVodeSetLinearSolver(&cvode_mem, LS.as_ref().expect("LS"), None);
                if check_retval(retval, "CVodeSetLinearSolver", 1) != 0 {
                    std::process::exit(1);
                }
            }

            _ => {}
        }

        /* Set preconditioner setup and solve routines Precond and PSolve,
        and the pointer to the user-defined block data */
        retval = CVodeSetPreconditioner(&cvode_mem, Some(Precond), Some(PSolve));
        if check_retval(retval, "CVodeSetPreconditioner", 1) != 0 {
            std::process::exit(1);
        }

        /* Set the linear solver tolerance conversion factor */
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

        retval = CVodeSetLSNormFactor(&cvode_mem, nrmfac);
        if check_retval(retval, "CVodeSetLSNormFactor", 1) != 0 {
            std::process::exit(1);
        }

        /* In loop over output points, call CVode, print results, and test for error */
        print!(" \n2-species diurnal advection-diffusion problem\n\n");
        tout = TWOHR;
        for _iout in 1..=NOUT {
            retval = CVode(&cvode_mem, tout, &u, &mut t, CV_NORMAL);
            PrintOutput(&cvode_mem, &u, t);
            if check_retval(retval, "CVode", 1) != 0 {
                break;
            }
            tout += TWOHR;
        }

        PrintFinalStats(&cvode_mem, linsolver);
    } /* END: Loop through SPGMR, SPBCG and SPTFQMR linear solver modules */

    /* Free memory (user data is dropped together with the solver memory) */
    N_VDestroy(u);
    CVodeFree(&mut Some(cvode_mem));
    SUNLinSolFree(LS.take());
    SUNNonlinSolFree(Some(NLS));
    SUNLogger_Destroy(&mut logger_opt);
    SUNContext_Free(&mut sunctx_opt);
}

/*
 *-------------------------------
 * Private helper functions
 *-------------------------------
 */

/* Allocate memory for data structure of type UserData */

fn AllocUserData() -> UserData {
    let ns = NUM_SPECIES as usize;
    UserData {
        P: vec![vec![vec![vec![ZERO; ns]; ns]; MY as usize]; MX as usize],
        Jbd: vec![vec![vec![vec![ZERO; ns]; ns]; MY as usize]; MX as usize],
        pivot: vec![vec![vec![0; ns]; MY as usize]; MX as usize],
        q4: ZERO,
        om: ZERO,
        dx: ZERO,
        dy: ZERO,
        hdco: ZERO,
        haco: ZERO,
        vdco: ZERO,
    }
}

/* Load problem constants in data */

fn InitUserData(data: &mut UserData) {
    data.om = PI / HALFDAY;
    data.dx = (XMAX - XMIN) / ((MX - 1) as sunrealtype);
    data.dy = (YMAX - YMIN) / ((MY - 1) as sunrealtype);
    data.hdco = KH / SQR(data.dx);
    data.haco = VEL / (TWO * data.dx);
    data.vdco = (ONE / SQR(data.dy)) * KV0;
}

/* Set initial conditions in u */

fn SetInitialProfiles(u: &N_Vector, dx: sunrealtype, dy: sunrealtype) {
    /* Set pointer to data array in vector u. */

    let mut udata = N_VGetArrayPointer(u).expect("N_VGetArrayPointer");

    /* Load initial profiles of c1 and c2 into u vector */

    for jy in 0..MY {
        let y = YMIN + (jy as sunrealtype) * dy;
        let mut cy = SQR(0.1 * (y - YMID));
        cy = ONE - cy + 0.5 * SQR(cy);
        for jx in 0..MX {
            let x = XMIN + (jx as sunrealtype) * dx;
            let mut cx = SQR(0.1 * (x - XMID));
            cx = ONE - cx + 0.5 * SQR(cx);
            udata[IJKth(1, jx, jy)] = C1_SCALE * cx * cy;
            udata[IJKth(2, jx, jy)] = C2_SCALE * cx * cy;
        }
    }
}

/* Print current t, step count, order, stepsize, and sampled c1,c2 values */

fn PrintOutput(cvode_mem: &CVodeMem, u: &N_Vector, t: sunrealtype) {
    let mut nst: i64 = 0;
    let mut qu: i32 = 0;
    let mut retval: i32;
    let mut hu: sunrealtype = ZERO;
    let (mxh, myh, mx1, my1) = (MX / 2 - 1, MY / 2 - 1, MX - 1, MY - 1);

    let udata = N_VGetArrayPointer(u).expect("N_VGetArrayPointer");

    retval = CVodeGetNumSteps(cvode_mem, &mut nst);
    check_retval(retval, "CVodeGetNumSteps", 1);
    retval = CVodeGetLastOrder(cvode_mem, &mut qu);
    check_retval(retval, "CVodeGetLastOrder", 1);
    retval = CVodeGetLastStep(cvode_mem, &mut hu);
    check_retval(retval, "CVodeGetLastStep", 1);

    print!(
        "t = {}   no. steps = {}   order = {}   stepsize = {}\n",
        fmt_e(t, 2),
        nst,
        qu,
        fmt_e(hu, 2)
    );
    print!(
        "c1 (bot.left/middle/top rt.) = {}  {}  {}\n",
        fmt_ew(udata[IJKth(1, 0, 0)], 12, 3),
        fmt_ew(udata[IJKth(1, mxh, myh)], 12, 3),
        fmt_ew(udata[IJKth(1, mx1, my1)], 12, 3)
    );
    print!(
        "c2 (bot.left/middle/top rt.) = {}  {}  {}\n\n",
        fmt_ew(udata[IJKth(2, 0, 0)], 12, 3),
        fmt_ew(udata[IJKth(2, mxh, myh)], 12, 3),
        fmt_ew(udata[IJKth(2, mx1, my1)], 12, 3)
    );
}

/* Get and print final statistics */

fn PrintFinalStats(cvode_mem: &CVodeMem, linsolver: i32) {
    let (mut lenrw, mut leniw): (i64, i64) = (0, 0);
    let (mut lenrwLS, mut leniwLS): (i64, i64) = (0, 0);
    let (mut nst, mut nfe, mut nsetups, mut nni, mut ncfn, mut netf): (
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
    ) = (0, 0, 0, 0, 0, 0);
    let (mut nli, mut npe, mut nps, mut ncfl, mut nfeLS): (i64, i64, i64, i64, i64) =
        (0, 0, 0, 0, 0);
    let mut retval: i32;

    retval = CVodeGetWorkSpace(cvode_mem, &mut lenrw, &mut leniw);
    check_retval(retval, "CVodeGetWorkSpace", 1);
    retval = CVodeGetNumSteps(cvode_mem, &mut nst);
    check_retval(retval, "CVodeGetNumSteps", 1);
    retval = CVodeGetNumRhsEvals(cvode_mem, &mut nfe);
    check_retval(retval, "CVodeGetNumRhsEvals", 1);
    retval = CVodeGetNumLinSolvSetups(cvode_mem, &mut nsetups);
    check_retval(retval, "CVodeGetNumLinSolvSetups", 1);
    retval = CVodeGetNumErrTestFails(cvode_mem, &mut netf);
    check_retval(retval, "CVodeGetNumErrTestFails", 1);
    retval = CVodeGetNumNonlinSolvIters(cvode_mem, &mut nni);
    check_retval(retval, "CVodeGetNumNonlinSolvIters", 1);
    retval = CVodeGetNumNonlinSolvConvFails(cvode_mem, &mut ncfn);
    check_retval(retval, "CVodeGetNumNonlinSolvConvFails", 1);

    retval = CVodeGetLinWorkSpace(cvode_mem, &mut lenrwLS, &mut leniwLS);
    check_retval(retval, "CVodeGetLinWorkSpace", 1);
    retval = CVodeGetNumLinIters(cvode_mem, &mut nli);
    check_retval(retval, "CVodeGetNumLinIters", 1);
    retval = CVodeGetNumPrecEvals(cvode_mem, &mut npe);
    check_retval(retval, "CVodeGetNumPrecEvals", 1);
    retval = CVodeGetNumPrecSolves(cvode_mem, &mut nps);
    check_retval(retval, "CVodeGetNumPrecSolves", 1);
    retval = CVodeGetNumLinConvFails(cvode_mem, &mut ncfl);
    check_retval(retval, "CVodeGetNumLinConvFails", 1);
    retval = CVodeGetNumLinRhsEvals(cvode_mem, &mut nfeLS);
    check_retval(retval, "CVodeGetNumLinRhsEvals", 1);

    print!("\nFinal Statistics.. \n\n");
    print!("lenrw   = {:5}     leniw   = {:5}\n", lenrw, leniw);
    print!("lenrwLS = {:5}     leniwLS = {:5}\n", lenrwLS, leniwLS);
    print!("nst     = {:5}\n", nst);
    print!("nfe     = {:5}     nfeLS   = {:5}\n", nfe, nfeLS);
    print!("nni     = {:5}     nli     = {:5}\n", nni, nli);
    print!("nsetups = {:5}     netf    = {:5}\n", nsetups, netf);
    print!("npe     = {:5}     nps     = {:5}\n", npe, nps);
    print!("ncfn    = {:5}     ncfl    = {:5}\n\n", ncfn, ncfl);

    if linsolver < 2 {
        print!("======================================================================\n\n");
    }
}

/* Check function return value...
opt == 0 means SUNDIALS function allocates memory so check if
         returned NULL pointer
opt == 1 means SUNDIALS function returns an integer value so check if
         retval < 0
opt == 2 means function allocates memory so check if returned
         NULL pointer */

fn check_retval(retval: i32, funcname: &str, _opt: i32) -> i32 {
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

fn check_null<T>(returnvalue: &Option<T>, funcname: &str, opt: i32) -> i32 {
    /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
    if returnvalue.is_none() {
        if opt == 0 {
            eprint!(
                "\nSUNDIALS_ERROR: {}() failed - returned NULL pointer\n\n",
                funcname
            );
        } else {
            eprint!(
                "\nMEMORY_ERROR: {}() failed - returned NULL pointer\n\n",
                funcname
            );
        }
        return 1;
    }
    0
}

/*
 *-------------------------------
 * Functions called by the solver
 *-------------------------------
 */

/* f routine. Compute RHS function f(t,u). */

fn f(t: sunrealtype, u: &N_Vector, udot: &N_Vector, user_data: &mut Option<Box<dyn Any>>) -> i32 {
    let data = user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<UserData>())
        .expect("user_data downcast");
    let udata = N_VGetArrayPointer(u).expect("N_VGetArrayPointer");
    let mut dudata = N_VGetArrayPointer(udot).expect("N_VGetArrayPointer");

    /* Set diurnal rate coefficients. */

    let s = (data.om * t).sun_sin();
    let q3: sunrealtype;
    if s > ZERO {
        q3 = (-A3 / s).sun_exp();
        data.q4 = (-A4 / s).sun_exp();
    } else {
        q3 = ZERO;
        data.q4 = ZERO;
    }

    /* Make local copies of problem variables, for efficiency. */

    let q4coef = data.q4;
    let dely = data.dy;
    let verdco = data.vdco;
    let hordco = data.hdco;
    let horaco = data.haco;

    /* Loop over all grid points. */

    for jy in 0..MY {
        /* Set vertical diffusion coefficients at jy +- 1/2 */

        let ydn = YMIN + ((jy as sunrealtype) - 0.5) * dely;
        let yup = ydn + dely;
        let cydn = verdco * (0.2 * ydn).sun_exp();
        let cyup = verdco * (0.2 * yup).sun_exp();
        let idn: i32 = if jy == 0 { 1 } else { -1 };
        let iup: i32 = if jy == MY - 1 { -1 } else { 1 };
        for jx in 0..MX {
            /* Extract c1 and c2, and set kinetic rate terms. */

            let c1 = udata[IJKth(1, jx, jy)];
            let c2 = udata[IJKth(2, jx, jy)];
            let qq1 = Q1 * c1 * C3;
            let qq2 = Q2 * c1 * c2;
            let qq3 = q3 * C3;
            let qq4 = q4coef * c2;
            let rkin1 = -qq1 - qq2 + TWO * qq3 + qq4;
            let rkin2 = qq1 - qq2 - qq4;

            /* Set vertical diffusion terms. */

            let c1dn = udata[IJKth(1, jx, jy + idn)];
            let c2dn = udata[IJKth(2, jx, jy + idn)];
            let c1up = udata[IJKth(1, jx, jy + iup)];
            let c2up = udata[IJKth(2, jx, jy + iup)];
            let vertd1 = cyup * (c1up - c1) - cydn * (c1 - c1dn);
            let vertd2 = cyup * (c2up - c2) - cydn * (c2 - c2dn);

            /* Set horizontal diffusion and advection terms. */

            let ileft: i32 = if jx == 0 { 1 } else { -1 };
            let iright: i32 = if jx == MX - 1 { -1 } else { 1 };
            let c1lt = udata[IJKth(1, jx + ileft, jy)];
            let c2lt = udata[IJKth(2, jx + ileft, jy)];
            let c1rt = udata[IJKth(1, jx + iright, jy)];
            let c2rt = udata[IJKth(2, jx + iright, jy)];
            let hord1 = hordco * (c1rt - TWO * c1 + c1lt);
            let hord2 = hordco * (c2rt - TWO * c2 + c2lt);
            let horad1 = horaco * (c1rt - c1lt);
            let horad2 = horaco * (c2rt - c2lt);

            /* Load all terms into udot. */

            dudata[IJKth(1, jx, jy)] = vertd1 + hord1 + horad1 + rkin1;
            dudata[IJKth(2, jx, jy)] = vertd2 + hord2 + horad2 + rkin2;
        }
    }

    0
}

/* Preconditioner setup routine. Generate and preprocess P. */

fn Precond(
    _tn: sunrealtype,
    u: &N_Vector,
    _fu: &N_Vector,
    jok: sunbooleantype,
    jcurPtr: &mut sunbooleantype,
    gamma: sunrealtype,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    /* Make local copies of pointers in user_data, and of pointer to u's data */

    let data = user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<UserData>())
        .expect("user_data downcast");
    let ns = NUM_SPECIES as sunindextype;
    let udata = N_VGetArrayPointer(u).expect("N_VGetArrayPointer");

    if jok {
        /* jok = SUNTRUE: Copy Jbd to P */

        for jy in 0..MY as usize {
            for jx in 0..MX as usize {
                let jcols: Vec<&mut [sunrealtype]> = data.Jbd[jx][jy]
                    .iter_mut()
                    .map(|c| c.as_mut_slice())
                    .collect();
                let mut pcols: Vec<&mut [sunrealtype]> = data.P[jx][jy]
                    .iter_mut()
                    .map(|c| c.as_mut_slice())
                    .collect();
                SUNDlsMat_denseCopy(&jcols, &mut pcols, ns, ns);
            }
        }

        *jcurPtr = SUNFALSE;
    } else {
        /* jok = SUNFALSE: Generate Jbd from scratch and copy to P */

        /* Make local copies of problem variables, for efficiency. */

        let q4coef = data.q4;
        let dely = data.dy;
        let verdco = data.vdco;
        let hordco = data.hdco;

        /* Compute 2x2 diagonal Jacobian blocks (using q4 values
        computed on the last f call).  Load into P. */

        for jy in 0..MY {
            let ydn = YMIN + ((jy as sunrealtype) - 0.5) * dely;
            let yup = ydn + dely;
            let cydn = verdco * (0.2 * ydn).sun_exp();
            let cyup = verdco * (0.2 * yup).sun_exp();
            let diag = -(cydn + cyup + TWO * hordco);
            for jx in 0..MX {
                let c1 = udata[IJKth(1, jx, jy)];
                let c2 = udata[IJKth(2, jx, jy)];
                let (jxu, jyu) = (jx as usize, jy as usize);
                {
                    /* IJth(j,i,jj) = j[jj-1][i-1] */
                    let j = &mut data.Jbd[jxu][jyu];
                    j[0][0] = (-Q1 * C3 - Q2 * c2) + diag;
                    j[1][0] = -Q2 * c1 + q4coef;
                    j[0][1] = Q1 * C3 - Q2 * c2;
                    j[1][1] = (-Q2 * c1 - q4coef) + diag;
                }
                let jcols: Vec<&mut [sunrealtype]> = data.Jbd[jxu][jyu]
                    .iter_mut()
                    .map(|c| c.as_mut_slice())
                    .collect();
                let mut pcols: Vec<&mut [sunrealtype]> = data.P[jxu][jyu]
                    .iter_mut()
                    .map(|c| c.as_mut_slice())
                    .collect();
                SUNDlsMat_denseCopy(&jcols, &mut pcols, ns, ns);
            }
        }

        *jcurPtr = SUNTRUE;
    }

    /* Scale by -gamma */

    for jy in 0..MY as usize {
        for jx in 0..MX as usize {
            let mut pcols: Vec<&mut [sunrealtype]> = data.P[jx][jy]
                .iter_mut()
                .map(|c| c.as_mut_slice())
                .collect();
            SUNDlsMat_denseScale(-gamma, &mut pcols, ns, ns);
        }
    }

    /* Add identity matrix and do LU decompositions on blocks in place. */

    for jx in 0..MX as usize {
        for jy in 0..MY as usize {
            let mut pcols: Vec<&mut [sunrealtype]> = data.P[jx][jy]
                .iter_mut()
                .map(|c| c.as_mut_slice())
                .collect();
            SUNDlsMat_denseAddIdentity(&mut pcols, ns);
            let retval = SUNDlsMat_denseGETRF(&mut pcols, ns, ns, &mut data.pivot[jx][jy]);
            if retval != 0 {
                return 1;
            }
        }
    }

    0
}

/* Preconditioner solve routine */

fn PSolve(
    _tn: sunrealtype,
    _u: &N_Vector,
    _fu: &N_Vector,
    r: &N_Vector,
    z: &N_Vector,
    _gamma: sunrealtype,
    _delta: sunrealtype,
    _lr: i32,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    /* Extract the P and pivot arrays from user_data. */

    let data = user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<UserData>())
        .expect("user_data downcast");
    let ns = NUM_SPECIES as sunindextype;

    N_VScale(ONE, r, z);

    let mut zdata = N_VGetArrayPointer(z).expect("N_VGetArrayPointer");

    /* Solve the block-diagonal system Px = r using LU factors stored
    in P and pivot data in pivot, and return the solution in z. */

    for jx in 0..MX {
        for jy in 0..MY {
            let (jxu, jyu) = (jx as usize, jy as usize);
            let mut pcols: Vec<&mut [sunrealtype]> = data.P[jxu][jyu]
                .iter_mut()
                .map(|c| c.as_mut_slice())
                .collect();
            let off = IJKth(1, jx, jy);
            let v = &mut zdata[off..off + NUM_SPECIES as usize];
            SUNDlsMat_denseGETRS(&mut pcols, ns, &data.pivot[jxu][jyu], v);
        }
    }

    0
}
