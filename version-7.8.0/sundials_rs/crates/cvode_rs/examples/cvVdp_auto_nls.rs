/* -----------------------------------------------------------------
 * Rust port of examples/cvode/serial/cvVdp_auto_nls.c
 * -----------------------------------------------------------------
 * We solve the classic Van der Pol problem:
 *   y'' - mu*(1 - y^2)*y' + y = 0,  y(0) = 2,  y'(0) = 0.
 * This second-order ODE is converted to a first-order system by defining
 *   y0 = y,  y1 = y'
 * giving
 *   y0' = y1
 *   y1' = mu*(1 - y0^2)*y1 - y0.
 * We use the SUNNonlinearSolver_Auto module to solve the implicit
 * system. This solver automatically switches between modified Newton
 * iteration and fixed-point iteration using a stiffness metric.
 * -----------------------------------------------------------------*/

#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

use cvode_rs::prelude::*;

use std::any::Any;

/* Problem constants */
const NEQ: sunindextype = 2;
const T0: sunrealtype = 0.0;
const TF: sunrealtype = 250.0;
const DTOUT: sunrealtype = 10.0;

#[derive(Clone, Copy)]
struct UserData {
    mu: sunrealtype,
}

fn f(_t: sunrealtype, y: &N_Vector, ydot: &N_Vector, user_data: &mut Option<Box<dyn Any>>) -> i32 {
    let mu = user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<UserData>())
        .expect("user_data")
        .mu;

    let (y0, y1) = {
        let ydata = N_VGetArrayPointer(y).expect("N_VGetArrayPointer");
        (ydata[0], ydata[1])
    };

    {
        let mut ydotdata = N_VGetArrayPointer(ydot).expect("N_VGetArrayPointer");
        ydotdata[0] = y1;
        ydotdata[1] = mu * (1.0 - y0 * y0) * y1 - y0;
    }

    0
}

fn Jac(
    _t: sunrealtype,
    y: &N_Vector,
    _fy: &N_Vector,
    J: &SUNMatrix,
    user_data: &mut Option<Box<dyn Any>>,
    _tmp1: &N_Vector,
    _tmp2: &N_Vector,
    _tmp3: &N_Vector,
) -> i32 {
    let mu = user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<UserData>())
        .expect("user_data")
        .mu;
    let (y0, y1) = {
        let ydata = N_VGetArrayPointer(y).expect("N_VGetArrayPointer");
        (ydata[0], ydata[1])
    };

    SM_ELEMENT_D_set(J, 0, 0, 0.0);
    SM_ELEMENT_D_set(J, 0, 1, 1.0);
    SM_ELEMENT_D_set(J, 1, 0, -2.0 * mu * y0 * y1 - 1.0);
    SM_ELEMENT_D_set(J, 1, 1, mu * (1.0 - y0 * y0));

    0
}

/* Check function return value (C passes NULL pointers as `None`) */
fn check_retval(retval: Option<i32>, funcname: &str, opt: i32) -> i32 {
    if opt == 0 && retval.is_none() {
        eprint!("ERROR: {}() returned NULL\n", funcname);
        return 1;
    }
    if opt == 1 {
        let err = retval.expect("retval");
        if err < 0 {
            eprint!("ERROR: {}() returned {}\n", funcname, err);
            return 1;
        }
    }
    0
}

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let argc = argv.len() as i32;

    let mut retval: i32;

    /* Problem setup */
    let user_data = UserData { mu: 100.0 };

    let y10: sunrealtype = 2.0;
    let y20: sunrealtype = 0.0;
    let reltol: sunrealtype = 1.0e-4;
    let abstol: sunrealtype = 1.0e-4;

    let mut sunctx: Option<SUNContext> = None;
    retval = SUNContext_Create(SUN_COMM_NULL, &mut sunctx);
    if check_retval(Some(retval), "SUNContext_Create", 1) != 0 {
        std::process::exit(1);
    }
    let ctx = sunctx.as_ref().expect("sunctx").clone();

    let y = N_VNew_Serial(NEQ, &ctx);
    if check_retval(y.as_ref().map(|_| 0), "N_VNew_Serial", 0) != 0 {
        std::process::exit(1);
    }
    let y = y.expect("y");

    {
        let mut ydata = N_VGetArrayPointer(&y).expect("N_VGetArrayPointer");
        ydata[0] = y10;
        ydata[1] = y20;
    }

    let mut cvode_mem_opt = CVodeCreate(CV_BDF, &ctx);
    if check_retval(cvode_mem_opt.as_ref().map(|_| 0), "CVodeCreate", 0) != 0 {
        std::process::exit(1);
    }
    let cvode_mem = cvode_mem_opt.as_ref().expect("cvode_mem").clone();

    retval = CVodeInit(&cvode_mem, f, T0, &y);
    if check_retval(Some(retval), "CVodeInit", 1) != 0 {
        std::process::exit(1);
    }

    retval = CVodeSStolerances(&cvode_mem, reltol, abstol);
    if check_retval(Some(retval), "CVodeSStolerances", 1) != 0 {
        std::process::exit(1);
    }

    retval = CVodeSetUserData(&cvode_mem, Some(Box::new(user_data)));
    if check_retval(Some(retval), "CVodeSetUserData", 1) != 0 {
        std::process::exit(1);
    }

    retval = CVodeSetMaxNumSteps(&cvode_mem, 10000);
    if check_retval(Some(retval), "CVodeSetMaxNumSteps", 1) != 0 {
        std::process::exit(1);
    }

    /* Create nonlinear solver (auto) */
    let NLS = SUNNonlinSol_Auto(&y, 0, SUNNONLINSOL_AUTO_NEWTON, &ctx);
    if check_retval(NLS.as_ref().map(|_| 0), "SUNNonlinSol_Auto", 0) != 0 {
        std::process::exit(1);
    }
    let NLS = NLS.expect("NLS");

    retval = CVodeSetNonlinearSolver(&cvode_mem, &NLS);
    if check_retval(Some(retval), "CVodeSetNonlinearSolver", 1) != 0 {
        std::process::exit(1);
    }

    /* Provide dense linear solver and Jacobian for when Newton is active */
    let A = SUNDenseMatrix(NEQ, NEQ, &ctx);
    if check_retval(A.as_ref().map(|_| 0), "SUNDenseMatrix", 0) != 0 {
        std::process::exit(1);
    }
    let A = A.expect("A");
    let LS = SUNLinSol_Dense(&y, &A, &ctx);
    if check_retval(LS.as_ref().map(|_| 0), "SUNLinSol_Dense", 0) != 0 {
        std::process::exit(1);
    }
    let LS = LS.expect("LS");

    retval = CVodeSetLinearSolver(&cvode_mem, &LS, Some(&A));
    if check_retval(Some(retval), "CVodeSetLinearSolver", 1) != 0 {
        std::process::exit(1);
    }

    retval = CVodeSetJacFn(&cvode_mem, Some(Jac));
    if check_retval(Some(retval), "CVodeSetJacFn", 1) != 0 {
        std::process::exit(1);
    }

    /* Parse any remaining command line arguments */
    retval = CVodeSetOptions(&cvode_mem, Some(""), Some(""), argc, &argv);
    if check_retval(Some(retval), "CVodeSetOptions", 1) != 0 {
        std::process::exit(1);
    }

    print!("\nVan der Pol oscillator (CVODE):\n");
    print!(
        "    initial conditions: y1 = {}, y2 = {}\n",
        fmt_f(y10, 6),
        fmt_f(y20, 6)
    );
    print!("    mu = {}\n", fmt_f(user_data.mu, 6));
    print!(
        "    reltol = {}, abstol = {}\n\n",
        fmt_e(reltol, 2),
        fmt_e(abstol, 2)
    );
    print!("        t           y1           y2\n");
    print!("   -----------------------------------\n");
    {
        let ydata = N_VGetArrayPointer(&y).expect("N_VGetArrayPointer");
        print!(
            "  {}  {}  {}\n",
            fmt_fw(T0, 10, 6),
            fmt_fw(ydata[0], 10, 6),
            fmt_fw(ydata[1], 10, 6)
        );
    }

    let Nt = SUNRceil(TF / DTOUT) as i32;
    let mut tout = T0 + DTOUT;
    for _iout in 0..Nt {
        let mut tret: sunrealtype = 0.0;
        retval = CVode(&cvode_mem, tout, &y, &mut tret, CV_NORMAL);
        {
            let ydata = N_VGetArrayPointer(&y).expect("N_VGetArrayPointer");
            print!(
                "  {}  {}  {}\n",
                fmt_fw(tret, 10, 6),
                fmt_fw(ydata[0], 10, 6),
                fmt_fw(ydata[1], 10, 6)
            );
        }

        if retval == CV_SUCCESS {
            tout += DTOUT;
            tout = SUNMIN(tout, TF);
        } else {
            print!("Solver failure, stopping integration\n");
            break;
        }
    }
    print!("   -----------------------------------\n");

    retval = CVodePrintAllStats(
        &cvode_mem,
        &SUNFile::Stdout,
        SUNOutputFormat::SUN_OUTPUTFORMAT_TABLE,
    );
    if check_retval(Some(retval), "CVodePrintAllStats", 1) != 0 {
        std::process::exit(1);
    }

    {
        let mut nfp: i64 = 0;
        let mut nnewt: i64 = 0;
        retval = SUNNonlinSolGetTotalNumItersByType_Auto(&NLS, &mut nfp, &mut nnewt);
        if check_retval(Some(retval), "SUNNonlinSolGetTotalNumItersByType_Auto", 1) != 0 {
            std::process::exit(1);
        }
        print!(
            "   Auto nonlinear solver iteration totals: newton = {}, fixed-point = {}\n",
            nnewt, nfp
        );
    }

    CVodeFree(&mut cvode_mem_opt);
    SUNNonlinSolFree(Some(NLS));
    SUNLinSolFree(Some(LS));
    SUNMatDestroy(A);
    N_VDestroy(y);
    SUNContext_Free(&mut sunctx);
}
