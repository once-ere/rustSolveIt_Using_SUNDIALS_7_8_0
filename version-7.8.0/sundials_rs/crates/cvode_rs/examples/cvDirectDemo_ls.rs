#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

/* -----------------------------------------------------------------
 * Rust port of examples/cvode/serial/cvDirectDemo_ls.c
 * -----------------------------------------------------------------
 * Demonstration program for CVODE - direct linear solvers.
 * Two separate problems are solved using both the CV_ADAMS and CV_BDF
 * linear multistep methods in combination with the
 * SUNNONLINSOL_FIXEDPOINT and SUNNONLINSOL_NEWTON nonlinear solver
 * modules:
 *
 * Problem 1: Van der Pol oscillator
 *   xdotdot - 3*(1 - x^2)*xdot + x = 0, x(0) = 2, xdot(0) = 0.
 *
 * Problem 2: ydot = A * y, where A is a banded lower triangular
 * matrix derived from 2-D advection PDE.
 * -----------------------------------------------------------------*/

use cvode_rs::prelude::*;

use std::any::Any;

/* Shared Problem Constants */

const ATOL: sunrealtype = 1.0e-6;
const RTOL: sunrealtype = 0.0;

const ZERO: sunrealtype = 0.0;
const ONE: sunrealtype = 1.0;
const TWO: sunrealtype = 2.0;
const THIRTY: sunrealtype = 30.0;

/* Problem #1 Constants */

const P1_NEQ: sunindextype = 2;
const P1_ETA: sunrealtype = 3.0;
const P1_NOUT: i32 = 4;
const P1_T0: sunrealtype = 0.0;
const P1_T1: sunrealtype = 1.39283880203;
const P1_DTOUT: sunrealtype = 2.214773875;
const P1_TOL_FACTOR: sunrealtype = 1.0e4;

/* Problem #2 Constants */

const P2_MESHX: sunindextype = 5;
const P2_MESHY: sunindextype = 5;
const P2_NEQ: sunindextype = P2_MESHX * P2_MESHY;
const P2_ALPH1: sunrealtype = 1.0;
const P2_ALPH2: sunrealtype = 1.0;
const P2_NOUT: i32 = 5;
const P2_ML: sunindextype = 5;
const P2_MU: sunindextype = 0;
const P2_T0: sunrealtype = 0.0;
const P2_T1: sunrealtype = 0.01;
const P2_TOUT_MULT: sunrealtype = 10.0;
const P2_TOL_FACTOR: sunrealtype = 1.0e3;

/* Linear Solver Options */

const FUNC: i32 = 0;
const DENSE_USER: i32 = 1;
const DENSE_DQ: i32 = 2;
const DIAG: i32 = 3;
const BAND_USER: i32 = 4;
const BAND_DQ: i32 = 5;

/* Implementation */

fn main() {
    let mut nerr: i32;

    nerr = Problem1();
    nerr += Problem2();
    PrintErrInfo(nerr);
}

fn Problem1() -> i32 {
    let reltol = RTOL;
    let abstol = ATOL;
    let mut t: sunrealtype = 0.0;
    let mut retval: i32;
    let mut nerr: i32 = 0;
    let mut qu: i32 = 0;
    let mut hu: sunrealtype = 0.0;

    let mut A: Option<SUNMatrix> = None;
    let mut LS: Option<SUNLinearSolver> = None;
    let mut NLS: Option<SUNNonlinearSolver> = None;

    /* Create the SUNDIALS context */
    let mut sunctx_opt: Option<SUNContext> = None;
    retval = SUNContext_Create(SUN_COMM_NULL, &mut sunctx_opt);
    if check_retval_int(retval, "SUNContext_Create") != 0 {
        return 1;
    }
    let sunctx = sunctx_opt.as_ref().unwrap().clone();

    let y_opt = N_VNew_Serial(P1_NEQ, &sunctx);
    if check_retval_null(&y_opt, "N_VNew_Serial") != 0 {
        return 1;
    }
    let y = y_opt.unwrap();
    PrintIntro1();

    let mut cvode_mem_opt = CVodeCreate(CV_ADAMS, &sunctx);
    if check_retval_null(&cvode_mem_opt, "CVodeCreate") != 0 {
        return 1;
    }
    let cvode_mem = cvode_mem_opt.as_ref().unwrap().clone();

    for miter in FUNC..=DIAG {
        let mut ero = ZERO;
        {
            let mut ydata = NV_DATA_S(&y);
            ydata[0] = TWO;
            ydata[1] = ZERO;
        }

        let firstrun = miter == FUNC;
        if firstrun {
            /* initialize CVode */
            retval = CVodeInit(&cvode_mem, f1, P1_T0, &y);
            if check_retval_int(retval, "CVodeInit") != 0 {
                return 1;
            }

            /* set scalar tolerances */
            retval = CVodeSStolerances(&cvode_mem, reltol, abstol);
            if check_retval_int(retval, "CVodeSStolerances") != 0 {
                return 1;
            }
        } else {
            /* reinitialize CVode */
            retval = CVodeReInit(&cvode_mem, P1_T0, &y);
            if check_retval_int(retval, "CVodeReInit") != 0 {
                return 1;
            }
        }

        retval = PrepareNextRun(
            &sunctx, &cvode_mem, CV_ADAMS, miter, &y, &mut A, 0, 0, &mut LS, &mut NLS,
        );
        if check_retval_int(retval, "PrepareNextRun") != 0 {
            return 1;
        }

        PrintHeader1();

        let mut tout = P1_T1;
        for iout in 1..=P1_NOUT {
            retval = CVode(&cvode_mem, tout, &y, &mut t, CV_NORMAL);
            check_retval_int(retval, "CVode");
            let temp_retval = CVodeGetLastOrder(&cvode_mem, &mut qu);
            if check_retval_int(temp_retval, "CVodeGetLastOrder") != 0 {
                nerr += 1;
            }
            let temp_retval = CVodeGetLastStep(&cvode_mem, &mut hu);
            if check_retval_int(temp_retval, "CVodeGetLastStep") != 0 {
                nerr += 1;
            }
            {
                let ydata = NV_DATA_S(&y);
                PrintOutput1(t, ydata[0], ydata[1], qu, hu);
            }
            if retval != CV_SUCCESS {
                nerr += 1;
                break;
            }
            if iout % 2 == 0 {
                let er = {
                    let ydata = NV_DATA_S(&y);
                    SUNRabs(ydata[0]) / abstol
                };
                if er > ero {
                    ero = er;
                }
                if er > P1_TOL_FACTOR {
                    nerr += 1;
                    PrintErrOutput(P1_TOL_FACTOR);
                }
            }
            tout += P1_DTOUT;
        }

        PrintFinalStats(&cvode_mem, miter, ero);
    }

    CVodeFree(&mut cvode_mem_opt);
    SUNNonlinSolFree(NLS.take());
    LS = None;
    A = None;

    let mut cvode_mem_opt = CVodeCreate(CV_BDF, &sunctx);
    if check_retval_null(&cvode_mem_opt, "CVodeCreate") != 0 {
        return 1;
    }
    let cvode_mem = cvode_mem_opt.as_ref().unwrap().clone();

    for miter in FUNC..=DIAG {
        let mut ero = ZERO;
        {
            let mut ydata = NV_DATA_S(&y);
            ydata[0] = TWO;
            ydata[1] = ZERO;
        }

        let firstrun = miter == FUNC;
        if firstrun {
            /* initialize CVode */
            retval = CVodeInit(&cvode_mem, f1, P1_T0, &y);
            if check_retval_int(retval, "CVodeInit") != 0 {
                return 1;
            }

            /* set scalar tolerances */
            retval = CVodeSStolerances(&cvode_mem, reltol, abstol);
            if check_retval_int(retval, "CVodeSStolerances") != 0 {
                return 1;
            }
        } else {
            /* reinitialize CVode */
            retval = CVodeReInit(&cvode_mem, P1_T0, &y);
            if check_retval_int(retval, "CVodeReInit") != 0 {
                return 1;
            }
        }

        retval = PrepareNextRun(
            &sunctx, &cvode_mem, CV_BDF, miter, &y, &mut A, 0, 0, &mut LS, &mut NLS,
        );
        if check_retval_int(retval, "PrepareNextRun") != 0 {
            return 1;
        }

        PrintHeader1();

        let mut tout = P1_T1;
        for iout in 1..=P1_NOUT {
            retval = CVode(&cvode_mem, tout, &y, &mut t, CV_NORMAL);
            check_retval_int(retval, "CVode");
            let temp_retval = CVodeGetLastOrder(&cvode_mem, &mut qu);
            if check_retval_int(temp_retval, "CVodeGetLastOrder") != 0 {
                nerr += 1;
            }
            let temp_retval = CVodeGetLastStep(&cvode_mem, &mut hu);
            if check_retval_int(temp_retval, "CVodeGetLastStep") != 0 {
                nerr += 1;
            }
            {
                let ydata = NV_DATA_S(&y);
                PrintOutput1(t, ydata[0], ydata[1], qu, hu);
            }
            if retval != CV_SUCCESS {
                nerr += 1;
                break;
            }
            if iout % 2 == 0 {
                let er = {
                    let ydata = NV_DATA_S(&y);
                    SUNRabs(ydata[0]) / abstol
                };
                if er > ero {
                    ero = er;
                }
                if er > P1_TOL_FACTOR {
                    nerr += 1;
                    PrintErrOutput(P1_TOL_FACTOR);
                }
            }
            tout += P1_DTOUT;
        }

        PrintFinalStats(&cvode_mem, miter, ero);
    }

    CVodeFree(&mut cvode_mem_opt);
    SUNNonlinSolFree(NLS.take());
    N_VDestroy(y);
    SUNContext_Free(&mut sunctx_opt);

    nerr
}

fn PrintIntro1() {
    print!("Demonstration program for CVODE package - direct linear solvers\n");
    print!("\n\n");
    print!("Problem 1: Van der Pol oscillator\n");
    print!(" xdotdot - 3*(1 - x^2)*xdot + x = 0, x(0) = 2, xdot(0) = 0\n");
    print!(
        " neq = {},  reltol = {},  abstol = {}",
        P1_NEQ,
        fmt_g(RTOL, 2),
        fmt_g(ATOL, 2)
    );
}

fn PrintHeader1() {
    print!("\n     t           x              xdot         qu     hu \n");
}

fn PrintOutput1(t: sunrealtype, y0: sunrealtype, y1: sunrealtype, qu: i32, hu: sunrealtype) {
    print!(
        "{}    {}   {}   {:2}    {}\n",
        fmt_fw(t, 10, 5),
        fmt_ew(y0, 12, 5),
        fmt_ew(y1, 12, 5),
        qu,
        fmt_ew(hu, 6, 4)
    );
}

fn f1(
    _t: sunrealtype,
    y: &N_Vector,
    ydot: &N_Vector,
    _user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    let (y0, y1) = {
        let ydata = NV_DATA_S(y);
        (ydata[0], ydata[1])
    };

    let mut dydata = NV_DATA_S(ydot);
    dydata[0] = y1;
    dydata[1] = (ONE - y0 * y0) * P1_ETA * y1 - y0;

    0
}

fn Jac1(
    _tn: sunrealtype,
    y: &N_Vector,
    _fy: &N_Vector,
    J: &SUNMatrix,
    _user_data: &mut Option<Box<dyn Any>>,
    _tmp1: &N_Vector,
    _tmp2: &N_Vector,
    _tmp3: &N_Vector,
) -> i32 {
    let (y0, y1) = {
        let ydata = NV_DATA_S(y);
        (ydata[0], ydata[1])
    };

    SM_ELEMENT_D_set(J, 0, 1, ONE);
    SM_ELEMENT_D_set(J, 1, 0, -TWO * P1_ETA * y0 * y1 - ONE);
    SM_ELEMENT_D_set(J, 1, 1, P1_ETA * (ONE - y0 * y0));

    0
}

fn Problem2() -> i32 {
    let reltol = RTOL;
    let abstol = ATOL;
    let mut t: sunrealtype = 0.0;
    let mut retval: i32;
    let mut nerr: i32 = 0;
    let mut qu: i32 = 0;
    let mut hu: sunrealtype = 0.0;

    let mut A: Option<SUNMatrix> = None;
    let mut LS: Option<SUNLinearSolver> = None;
    let mut NLS: Option<SUNNonlinearSolver> = None;

    /* Create SUNDIALS context */
    let mut sunctx_opt: Option<SUNContext> = None;
    retval = SUNContext_Create(SUN_COMM_NULL, &mut sunctx_opt);
    if check_retval_int(retval, "SUNContext_Create") != 0 {
        return 1;
    }
    let sunctx = sunctx_opt.as_ref().unwrap().clone();

    let y_opt = N_VNew_Serial(P2_NEQ, &sunctx);
    if check_retval_null(&y_opt, "N_VNew_Serial") != 0 {
        return 1;
    }
    let y = y_opt.unwrap();

    PrintIntro2();

    let mut cvode_mem_opt = CVodeCreate(CV_ADAMS, &sunctx);
    if check_retval_null(&cvode_mem_opt, "CVodeCreate") != 0 {
        return 1;
    }
    let cvode_mem = cvode_mem_opt.as_ref().unwrap().clone();

    for miter in FUNC..=BAND_DQ {
        if (miter == DENSE_USER) || (miter == DENSE_DQ) {
            continue;
        }
        let mut ero = ZERO;
        N_VConst(ZERO, &y);
        {
            let mut ydata = NV_DATA_S(&y);
            ydata[0] = ONE;
        }

        let firstrun = miter == FUNC;
        if firstrun {
            /* initialize CVode */
            retval = CVodeInit(&cvode_mem, f2, P2_T0, &y);
            if check_retval_int(retval, "CVodeInit") != 0 {
                return 1;
            }

            /* set scalar tolerances */
            retval = CVodeSStolerances(&cvode_mem, reltol, abstol);
            if check_retval_int(retval, "CVodeSStolerances") != 0 {
                return 1;
            }
        } else {
            /* reinitialize CVode */
            retval = CVodeReInit(&cvode_mem, P2_T0, &y);
            if check_retval_int(retval, "CVodeReInit") != 0 {
                return 1;
            }
        }

        retval = PrepareNextRun(
            &sunctx, &cvode_mem, CV_ADAMS, miter, &y, &mut A, P2_MU, P2_ML, &mut LS, &mut NLS,
        );
        if check_retval_int(retval, "PrepareNextRun") != 0 {
            return 1;
        }

        PrintHeader2();

        let mut tout = P2_T1;
        for _iout in 1..=P2_NOUT {
            retval = CVode(&cvode_mem, tout, &y, &mut t, CV_NORMAL);
            check_retval_int(retval, "CVode");
            let erm = MaxError(&y, t);
            let temp_retval = CVodeGetLastOrder(&cvode_mem, &mut qu);
            if check_retval_int(temp_retval, "CVodeGetLastOrder") != 0 {
                nerr += 1;
            }
            let temp_retval = CVodeGetLastStep(&cvode_mem, &mut hu);
            if check_retval_int(temp_retval, "CVodeGetLastStep") != 0 {
                nerr += 1;
            }
            PrintOutput2(t, erm, qu, hu);
            if retval != CV_SUCCESS {
                nerr += 1;
                break;
            }
            let er = erm / abstol;
            if er > ero {
                ero = er;
            }
            if er > P2_TOL_FACTOR {
                nerr += 1;
                PrintErrOutput(P2_TOL_FACTOR);
            }
            tout *= P2_TOUT_MULT;
        }

        PrintFinalStats(&cvode_mem, miter, ero);
    }

    CVodeFree(&mut cvode_mem_opt);
    SUNNonlinSolFree(NLS.take());
    SUNLinSolFree(LS.take());
    if let Some(a) = A.take() {
        SUNMatDestroy(a);
    }

    let mut cvode_mem_opt = CVodeCreate(CV_BDF, &sunctx);
    if check_retval_null(&cvode_mem_opt, "CVodeCreate") != 0 {
        return 1;
    }
    let cvode_mem = cvode_mem_opt.as_ref().unwrap().clone();

    for miter in FUNC..=BAND_DQ {
        if (miter == DENSE_USER) || (miter == DENSE_DQ) {
            continue;
        }
        let mut ero = ZERO;
        N_VConst(ZERO, &y);
        {
            let mut ydata = NV_DATA_S(&y);
            ydata[0] = ONE;
        }

        let firstrun = miter == FUNC;
        if firstrun {
            /* initialize CVode */
            retval = CVodeInit(&cvode_mem, f2, P2_T0, &y);
            if check_retval_int(retval, "CVodeInit") != 0 {
                return 1;
            }

            /* set scalar tolerances */
            retval = CVodeSStolerances(&cvode_mem, reltol, abstol);
            if check_retval_int(retval, "CVodeSStolerances") != 0 {
                return 1;
            }
        } else {
            /* reinitialize CVode */
            retval = CVodeReInit(&cvode_mem, P2_T0, &y);
            if check_retval_int(retval, "CVodeReInit") != 0 {
                return 1;
            }
        }

        retval = PrepareNextRun(
            &sunctx, &cvode_mem, CV_BDF, miter, &y, &mut A, P2_MU, P2_ML, &mut LS, &mut NLS,
        );
        if check_retval_int(retval, "PrepareNextRun") != 0 {
            return 1;
        }

        PrintHeader2();

        let mut tout = P2_T1;
        for _iout in 1..=P2_NOUT {
            retval = CVode(&cvode_mem, tout, &y, &mut t, CV_NORMAL);
            check_retval_int(retval, "CVode");
            let erm = MaxError(&y, t);
            let temp_retval = CVodeGetLastOrder(&cvode_mem, &mut qu);
            if check_retval_int(temp_retval, "CVodeGetLastOrder") != 0 {
                nerr += 1;
            }
            let temp_retval = CVodeGetLastStep(&cvode_mem, &mut hu);
            if check_retval_int(temp_retval, "CVodeGetLastStep") != 0 {
                nerr += 1;
            }
            PrintOutput2(t, erm, qu, hu);
            if retval != CV_SUCCESS {
                nerr += 1;
                break;
            }
            let er = erm / abstol;
            if er > ero {
                ero = er;
            }
            if er > P2_TOL_FACTOR {
                nerr += 1;
                PrintErrOutput(P2_TOL_FACTOR);
            }
            tout *= P2_TOUT_MULT;
        }

        PrintFinalStats(&cvode_mem, miter, ero);
    }

    CVodeFree(&mut cvode_mem_opt);
    SUNNonlinSolFree(NLS.take());
    SUNLinSolFree(LS.take());
    if let Some(a) = A.take() {
        SUNMatDestroy(a);
    }
    N_VDestroy(y);
    SUNContext_Free(&mut sunctx_opt);

    nerr
}

fn PrintIntro2() {
    print!("\n\n-------------------------------------------------------------");
    print!("\n-------------------------------------------------------------");
    print!("\n\nProblem 2: ydot = A * y, where A is a banded lower\n");
    print!("triangular matrix derived from 2-D advection PDE\n\n");
    print!(" neq = {}, ml = {}, mu = {}\n", P2_NEQ, P2_ML, P2_MU);
    print!(
        " itol = {}, reltol = {}, abstol = {}",
        "CV_SS",
        fmt_g(RTOL, 2),
        fmt_g(ATOL, 2)
    );
}

fn PrintHeader2() {
    print!("\n      t        max.err      qu     hu \n");
}

fn PrintOutput2(t: sunrealtype, erm: sunrealtype, qu: i32, hu: sunrealtype) {
    print!(
        "{}  {}   {:2}   {}\n",
        fmt_fw(t, 10, 3),
        fmt_ew(erm, 12, 4),
        qu,
        fmt_ew(hu, 12, 4)
    );
}

fn f2(
    _t: sunrealtype,
    y: &N_Vector,
    ydot: &N_Vector,
    _user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    let ydata = N_VGetArrayPointer(y).expect("N_VGetArrayPointer");
    let mut dydata = N_VGetArrayPointer(ydot).expect("N_VGetArrayPointer");

    /*
       Excluding boundaries,

       ydot    = f    = -2 y    + alpha1 * y      + alpha2 * y
           i,j    i,j       i,j             i-1,j             i,j-1
    */

    for j in 0..P2_MESHY {
        for i in 0..P2_MESHX {
            let k = (i + j * P2_MESHX) as usize;
            let mut d = -TWO * ydata[k];
            if i != 0 {
                d += P2_ALPH1 * ydata[k - 1];
            }
            if j != 0 {
                d += P2_ALPH2 * ydata[k - P2_MESHX as usize];
            }
            dydata[k] = d;
        }
    }

    0
}

fn Jac2(
    _tn: sunrealtype,
    _y: &N_Vector,
    _fy: &N_Vector,
    J: &SUNMatrix,
    _user_data: &mut Option<Box<dyn Any>>,
    _tmp1: &N_Vector,
    _tmp2: &N_Vector,
    _tmp3: &N_Vector,
) -> i32 {
    /*
       The components of f(t,y) which depend on y    are
                                                 i,j
       f    , f      , and f      :
        i,j    i+1,j        i,j+1

       f    = -2 y    + alpha1 * y      + alpha2 * y
        i,j       i,j             i-1,j             i,j-1

       f      = -2 y      + alpha1 * y    + alpha2 * y
        i+1,j       i+1,j             i,j             i+1,j-1

       f      = -2 y      + alpha1 * y        + alpha2 * y
        i,j+1       i,j+1             i-1,j+1             i,j
    */

    let s_mu = SM_SUBAND_B(J);

    for j in 0..P2_MESHY {
        for i in 0..P2_MESHX {
            let k = i + j * P2_MESHX;
            let mut kthCol = SM_COLUMN_B(J, k);
            kthCol[SM_COLUMN_ELEMENT_IDX(k, k, s_mu)] = -TWO;
            if i != P2_MESHX - 1 {
                kthCol[SM_COLUMN_ELEMENT_IDX(k + 1, k, s_mu)] = P2_ALPH1;
            }
            if j != P2_MESHY - 1 {
                kthCol[SM_COLUMN_ELEMENT_IDX(k + P2_MESHX, k, s_mu)] = P2_ALPH2;
            }
        }
    }

    0
}

fn MaxError(y: &N_Vector, t: sunrealtype) -> sunrealtype {
    let mut ex = ZERO;
    let mut maxError = ZERO;
    let mut jfact_inv = ONE;

    if t == ZERO {
        return ZERO;
    }

    let ydata = N_VGetArrayPointer(y).expect("N_VGetArrayPointer");
    if t <= THIRTY {
        ex = SUNRexp(-TWO * t);
    }

    for j in 0..P2_MESHY {
        let mut ifact_inv = ONE;
        for i in 0..P2_MESHX {
            let k = (i + j * P2_MESHX) as usize;
            let yt = SUNRpowerR(t, (i + j) as sunrealtype) * ex * ifact_inv * jfact_inv;
            let er = SUNRabs(ydata[k] - yt);
            if er > maxError {
                maxError = er;
            }
            ifact_inv /= (i + 1) as sunrealtype;
        }
        jfact_inv /= (j + 1) as sunrealtype;
    }
    maxError
}

#[allow(clippy::too_many_arguments)]
fn PrepareNextRun(
    sunctx: &SUNContext,
    cvode_mem: &CVodeMem,
    lmm: i32,
    miter: i32,
    y: &N_Vector,
    A: &mut Option<SUNMatrix>,
    mu: sunindextype,
    ml: sunindextype,
    LS: &mut Option<SUNLinearSolver>,
    NLS: &mut Option<SUNNonlinearSolver>,
) -> i32 {
    let mut retval: i32;

    if NLS.is_some() {
        SUNNonlinSolFree(NLS.take());
    }
    if LS.is_some() {
        SUNLinSolFree(LS.take());
    }
    if let Some(a) = A.take() {
        SUNMatDestroy(a);
    }

    print!("\n\n-------------------------------------------------------------");

    print!("\n\nLinear Multistep Method : ");
    if lmm == CV_ADAMS {
        print!("ADAMS\n");
    } else {
        print!("BDF\n");
    }

    print!("Iteration               : ");
    if miter == FUNC {
        print!("FIXEDPOINT\n");

        /* create fixed point nonlinear solver object */
        *NLS = SUNNonlinSol_FixedPoint(y, 0, sunctx);
        if check_retval_null(NLS, "SUNNonlinSol_FixedPoint") != 0 {
            return 1;
        }

        /* attach nonlinear solver object to CVode */
        retval = CVodeSetNonlinearSolver(cvode_mem, NLS.as_ref().unwrap());
        if check_retval_int(retval, "CVodeSetNonlinearSolver") != 0 {
            return 1;
        }
    } else {
        print!("NEWTON\n");

        /* create Newton nonlinear solver object */
        *NLS = SUNNonlinSol_Newton(y, sunctx);
        if check_retval_null(NLS, "SUNNonlinSol_Newton") != 0 {
            return 1;
        }

        /* attach nonlinear solver object to CVode */
        retval = CVodeSetNonlinearSolver(cvode_mem, NLS.as_ref().unwrap());
        if check_retval_int(retval, "CVodeSetNonlinearSolver") != 0 {
            return 1;
        }

        print!("Linear Solver           : ");

        match miter {
            DENSE_USER => {
                print!("Dense, User-Supplied Jacobian\n");

                /* Create dense SUNMatrix for use in linear solves */
                *A = SUNDenseMatrix(P1_NEQ, P1_NEQ, sunctx);
                if check_retval_null(A, "SUNDenseMatrix") != 0 {
                    return 1;
                }

                /* Create dense SUNLinearSolver object for use by CVode */
                *LS = SUNLinSol_Dense(y, A.as_ref().unwrap(), sunctx);
                if check_retval_null(LS, "SUNLinSol_Dense") != 0 {
                    return 1;
                }

                /* Call CVodeSetLinearSolver to attach the matrix and linear solver to CVode */
                retval = CVodeSetLinearSolver(cvode_mem, LS.as_ref().unwrap(), A.as_ref());
                if check_retval_int(retval, "CVodeSetLinearSolver") != 0 {
                    return 1;
                }

                /* Set the user-supplied Jacobian routine Jac */
                retval = CVodeSetJacFn(cvode_mem, Some(Jac1));
                if check_retval_int(retval, "CVodeSetJacFn") != 0 {
                    return 1;
                }
            }

            DENSE_DQ => {
                print!("Dense, Difference Quotient Jacobian\n");

                /* Create dense SUNMatrix for use in linear solves */
                *A = SUNDenseMatrix(P1_NEQ, P1_NEQ, sunctx);
                if check_retval_null(A, "SUNDenseMatrix") != 0 {
                    return 1;
                }

                /* Create dense SUNLinearSolver object for use by CVode */
                *LS = SUNLinSol_Dense(y, A.as_ref().unwrap(), sunctx);
                if check_retval_null(LS, "SUNLinSol_Dense") != 0 {
                    return 1;
                }

                /* Call CVodeSetLinearSolver to attach the matrix and linear solver to CVode */
                retval = CVodeSetLinearSolver(cvode_mem, LS.as_ref().unwrap(), A.as_ref());
                if check_retval_int(retval, "CVodeSetLinearSolver") != 0 {
                    return 1;
                }

                /* Use a difference quotient Jacobian */
                retval = CVodeSetJacFn(cvode_mem, None);
                if check_retval_int(retval, "CVodeSetJacFn") != 0 {
                    return 1;
                }
            }

            DIAG => {
                print!("Diagonal Jacobian\n");

                /* Call CVDiag to create/attach the CVODE-specific diagonal solver */
                retval = CVDiag(cvode_mem);
                if check_retval_int(retval, "CVDiag") != 0 {
                    return 1;
                }
            }

            BAND_USER => {
                print!("Band, User-Supplied Jacobian\n");

                /* Create band SUNMatrix for use in linear solves */
                *A = SUNBandMatrix(P2_NEQ, mu, ml, sunctx);
                if check_retval_null(A, "SUNBandMatrix") != 0 {
                    return 1;
                }

                /* Create banded SUNLinearSolver object for use by CVode */
                *LS = SUNLinSol_Band(y, A.as_ref().unwrap(), sunctx);
                if check_retval_null(LS, "SUNLinSol_Band") != 0 {
                    return 1;
                }

                /* Call CVodeSetLinearSolver to attach the matrix and linear solver to CVode */
                retval = CVodeSetLinearSolver(cvode_mem, LS.as_ref().unwrap(), A.as_ref());
                if check_retval_int(retval, "CVodeSetLinearSolver") != 0 {
                    return 1;
                }

                /* Set the user-supplied Jacobian routine Jac */
                retval = CVodeSetJacFn(cvode_mem, Some(Jac2));
                if check_retval_int(retval, "CVodeSetJacFn") != 0 {
                    return 1;
                }
            }

            BAND_DQ => {
                print!("Band, Difference Quotient Jacobian\n");

                /* Create band SUNMatrix for use in linear solves */
                *A = SUNBandMatrix(P2_NEQ, mu, ml, sunctx);
                if check_retval_null(A, "SUNBandMatrix") != 0 {
                    return 1;
                }

                /* Create banded SUNLinearSolver object for use by CVode */
                *LS = SUNLinSol_Band(y, A.as_ref().unwrap(), sunctx);
                if check_retval_null(LS, "SUNLinSol_Band") != 0 {
                    return 1;
                }

                /* Call CVodeSetLinearSolver to attach the matrix and linear solver to CVode */
                retval = CVodeSetLinearSolver(cvode_mem, LS.as_ref().unwrap(), A.as_ref());
                if check_retval_int(retval, "CVodeSetLinearSolver") != 0 {
                    return 1;
                }

                /* Use a difference quotient Jacobian */
                retval = CVodeSetJacFn(cvode_mem, None);
                if check_retval_int(retval, "CVodeSetJacFn") != 0 {
                    return 1;
                }
            }

            _ => {}
        }
    }

    retval
}

fn PrintErrOutput(tol_factor: sunrealtype) {
    print!(
        "\n\n Error exceeds {} * tolerance \n\n",
        fmt_g(tol_factor, 6)
    );
}

fn PrintFinalStats(cvode_mem: &CVodeMem, miter: i32, ero: sunrealtype) {
    let mut lenrw: i64 = 0;
    let mut leniw: i64 = 0;
    let mut lenrwLS: i64 = 0;
    let mut leniwLS: i64 = 0;
    let mut nst: i64 = 0;
    let mut nfe: i64 = 0;
    let mut nsetups: i64 = 0;
    let mut nni: i64 = 0;
    let mut ncfn: i64 = 0;
    let mut netf: i64 = 0;
    let mut nje: i64 = 0;
    let mut nfeLS: i64 = 0;
    let mut retval: i32;

    retval = CVodeGetWorkSpace(cvode_mem, &mut lenrw, &mut leniw);
    check_retval_int(retval, "CVodeGetWorkSpace");
    retval = CVodeGetNumSteps(cvode_mem, &mut nst);
    check_retval_int(retval, "CVodeGetNumSteps");
    retval = CVodeGetNumRhsEvals(cvode_mem, &mut nfe);
    check_retval_int(retval, "CVodeGetNumRhsEvals");
    retval = CVodeGetNumLinSolvSetups(cvode_mem, &mut nsetups);
    check_retval_int(retval, "CVodeGetNumLinSolvSetups");
    retval = CVodeGetNumErrTestFails(cvode_mem, &mut netf);
    check_retval_int(retval, "CVodeGetNumErrTestFails");
    retval = CVodeGetNumNonlinSolvIters(cvode_mem, &mut nni);
    check_retval_int(retval, "CVodeGetNumNonlinSolvIters");
    retval = CVodeGetNumNonlinSolvConvFails(cvode_mem, &mut ncfn);
    check_retval_int(retval, "CVodeGetNumNonlinSolvConvFails");

    print!("\n Final statistics for this run:\n\n");
    print!(" CVode real workspace length              = {:4} \n", lenrw);
    print!(" CVode integer workspace length           = {:4} \n", leniw);
    print!(" Number of steps                          = {:4} \n", nst);
    print!(" Number of f-s                            = {:4} \n", nfe);
    print!(
        " Number of setups                         = {:4} \n",
        nsetups
    );
    print!(" Number of nonlinear iterations           = {:4} \n", nni);
    print!(" Number of nonlinear convergence failures = {:4} \n", ncfn);
    print!(
        " Number of error test failures            = {:4} \n\n",
        netf
    );

    if miter != FUNC {
        if miter != DIAG {
            retval = CVodeGetNumJacEvals(cvode_mem, &mut nje);
            check_retval_int(retval, "CVodeGetNumJacEvals");
            retval = CVodeGetNumLinRhsEvals(cvode_mem, &mut nfeLS);
            check_retval_int(retval, "CVodeGetNumLinRhsEvals");
            retval = CVodeGetLinWorkSpace(cvode_mem, &mut lenrwLS, &mut leniwLS);
            check_retval_int(retval, "CVodeGetLinWorkSpace");
        } else {
            nje = nsetups;
            retval = CVDiagGetNumRhsEvals(cvode_mem, &mut nfeLS);
            check_retval_int(retval, "CVDiagGetNumRhsEvals");
            retval = CVDiagGetWorkSpace(cvode_mem, &mut lenrwLS, &mut leniwLS);
            check_retval_int(retval, "CVDiagGetWorkSpace");
        }
        print!(
            " Linear solver real workspace length      = {:4} \n",
            lenrwLS
        );
        print!(
            " Linear solver integer workspace length   = {:4} \n",
            leniwLS
        );
        print!(" Number of Jacobian evaluations           = {:4} \n", nje);
        print!(
            " Number of f evals. in linear solver      = {:4} \n\n",
            nfeLS
        );
    }

    print!(" Error overrun = {} \n", fmt_f(ero, 3));
}

fn PrintErrInfo(nerr: i32) {
    print!("\n\n-------------------------------------------------------------");
    print!("\n-------------------------------------------------------------");
    print!("\n\n Number of errors encountered = {} \n", nerr);
}

/* Check function return value...
opt == 0 means SUNDIALS function allocates memory so check if
         returned NULL pointer
opt == 1 means SUNDIALS function returns an integer value so check if
         retval < 0 */

fn check_retval_null<T>(returnvalue: &Option<T>, funcname: &str) -> i32 {
    if returnvalue.is_none() {
        eprint!(
            "\nSUNDIALS_ERROR: {}() failed - returned NULL pointer\n\n",
            funcname
        );
        return 1;
    }
    0
}

fn check_retval_int(retval: i32, funcname: &str) -> i32 {
    if retval < 0 {
        eprint!(
            "\nSUNDIALS_ERROR: {}() failed with retval = {}\n\n",
            funcname, retval
        );
        return 1;
    }
    0
}
