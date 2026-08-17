/*
 * -----------------------------------------------------------------
 * Rust port of examples/cvodes/serial/cvsRoberts_FSA_dns_Switch.c
 * -----------------------------------------------------------------
 * Modification of the cvsRoberts_FSA_dns to illustrate switching
 * on and off sensitivity computations.
 *
 * Example problem (from cvsRoberts_FSA_dns):
 *
 * The following is a simple example problem, with the coding
 * needed for its solution by CVODES for Forward Sensitivity
 * Analysis. The problem is from chemical kinetics, and consists
 * of the following three rate equations:
 *    dy1/dt = -p1*y1 + p2*y2*y3
 *    dy2/dt =  p1*y1 - p2*y2*y3 - p3*(y2)^2
 *    dy3/dt =  p3*(y2)^2
 * on the interval from t = 0.0 to t = 4.e10, with initial
 * conditions y1 = 1.0, y2 = y3 = 0. The reaction rates are: p1=0.04,
 * p2=1e4, and p3=3e7. The problem is stiff.
 * This program solves the problem with the BDF method, Newton
 * iteration with the dense linear solver, and a
 * user-supplied Jacobian routine.
 * It uses a scalar relative tolerance and a vector absolute
 * tolerance.
 * Output is printed in decades from t = .4 to t = 4.e10.
 * Run statistics (optional outputs) are printed at the end.
 *------------------------------------------------------------------
 */

#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

use cvodes_rs::prelude::*;

/* Problem Constants */
const MXSTEPS: i64 = 2000; /* max number of steps */
const NEQ: sunindextype = 3; /* number of equations */
const T0: sunrealtype = 0.0; /* initial time        */
const T1: sunrealtype = 4.0e10; /* first output time   */

const ZERO: sunrealtype = 0.0;

/* Vector accessor helpers mirroring the C `NV_Ith_S(v, i)` macro (0-based). */

fn NV_Ith_S(v: &N_Vector, i: usize) -> sunrealtype {
    N_VGetArrayPointer(v).expect("N_VGetArrayPointer")[i]
}

fn NV_Ith_S_set(v: &N_Vector, i: usize, x: sunrealtype) {
    N_VGetArrayPointer(v).expect("N_VGetArrayPointer")[i] = x;
}

/* Type : UserData
 *
 * C hands the very same `data` pointer to CVODES (`CVodeSetUserData`) and
 * keeps using it from `main` afterwards (it flips `sensi`/`errconS`/`fsDQ`/
 * `meth` and rewrites `p` between runs). The port therefore shares the whole
 * record as an `Rc<RefCell<UserData>>`: `main` keeps one handle and the
 * integrator memory owns a boxed clone.
 *
 * `p` additionally goes to `CVodeSetSensParams`, which in C stores the raw
 * array pointer so the internal difference-quotient sensitivity RHS (used by
 * the third run below, where `fS` is NULL) perturbs the very array that `f`
 * and `Jac` read back through `user_data`. That aliasing is reproduced with
 * the `SensParams = Rc<RefCell<Vec<sunrealtype>>>` handle: CVODES receives a
 * CLONE of this exact `Rc`. */

struct UserData {
    sensi: sunbooleantype,   /* turn on (T) or off (F) sensitivity analysis    */
    errconS: sunbooleantype, /* full (T) or partial error control (F)          */
    fsDQ: sunbooleantype,    /* user provided r.h.s sensitivity analysis (T/F) */
    meth: i32,               /* sensitivity method                             */
    p: SensParams,           /* sensitivity variables                          */
}

/* Snapshot of `data->p` as the C callbacks read it (`p1 = data->p[0];` …).
While the internal DQ sensitivity RHS is running, the entry for the active
parameter carries the perturbation CVODES just wrote through the shared
handle. Both borrows are released before the caller touches anything else. */

fn UserDataParams(data: &Rc<RefCell<UserData>>) -> [sunrealtype; 3] {
    let d = data.borrow();
    let p = d.p.borrow();
    [p[0], p[1], p[2]]
}

fn UserDataFromBox(user_data: &mut Option<Box<dyn Any>>) -> Rc<RefCell<UserData>> {
    user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<Rc<RefCell<UserData>>>())
        .expect("UserData")
        .clone()
}

/*
 *--------------------------------------------------------------------
 * MAIN PROGRAM
 *--------------------------------------------------------------------
 */

fn main() {
    /* Create the SUNDIALS context object for this simulation */
    let mut sunctx: Option<SUNContext> = None;
    let retval = SUNContext_Create(SUN_COMM_NULL, &mut sunctx);
    if check_retval_int(retval, "SUNContextCreate") != 0 {
        std::process::exit(1);
    }
    let ctx = sunctx.as_ref().expect("SUNContext").clone();

    /* Allocate user data structure */
    let data: Rc<RefCell<UserData>> = Rc::new(RefCell::new(UserData {
        sensi: SUNFALSE,
        errconS: SUNFALSE,
        fsDQ: SUNFALSE,
        meth: 0,
        p: Rc::new(RefCell::new(vec![ZERO; 3])),
    }));

    /* Initialize sensitivity variables (reaction rates for this problem) */
    {
        let d = data.borrow();
        let mut p = d.p.borrow_mut();
        p[0] = 0.04;
        p[1] = 1.0e4;
        p[2] = 3.0e7;
    }

    /* Allocate initial condition vector and set context */
    let y0 = N_VNew_Serial(NEQ, &ctx);
    if check_retval_ptr(&y0, "N_VNew_Serial", 0) != 0 {
        std::process::exit(1);
    }
    let y0 = y0.expect("N_VNew_Serial");

    /* Create solution and absolute tolerance vectors */
    let y = N_VClone(&y0);
    if check_retval_ptr(&y, "N_VClone", 0) != 0 {
        std::process::exit(1);
    }
    let y = y.expect("N_VClone");

    let abstol = N_VClone(&y0);
    if check_retval_ptr(&abstol, "N_VClone", 0) != 0 {
        std::process::exit(1);
    }
    let abstol = abstol.expect("N_VClone");

    /* Set initial conditions */
    NV_Ith_S_set(&y0, 0, 1.0);
    NV_Ith_S_set(&y0, 1, 0.0);
    NV_Ith_S_set(&y0, 2, 0.0);

    /* Set integration tolerances */
    let reltol: sunrealtype = 1e-6;
    NV_Ith_S_set(&abstol, 0, 1e-8);
    NV_Ith_S_set(&abstol, 1, 1e-14);
    NV_Ith_S_set(&abstol, 2, 1e-6);

    /* Call CVodeCreate to create the solver memory and specify the
     * Backward Differentiation Formula */
    let cvode_mem = CVodeCreate(CV_BDF, &ctx);
    if check_retval_ptr(&cvode_mem, "CVodeCreate", 0) != 0 {
        std::process::exit(1);
    }
    let cv = cvode_mem.expect("CVodeCreate");

    /* Call CVodeInit to initialize the integrator memory and specify the
     * user's right hand side function y'=f(t,y), the initial time T0, and
     * the initial dependenet variable vector y0. */
    let retval = CVodeInit(&cv, f, T0, &y0);
    if check_retval_int(retval, "CVodeInit") != 0 {
        std::process::exit(1);
    }

    /* Call CVodeSVTolerances to specify the scalar relative tolerance
     * and vector absolute tolereance */
    let retval = CVodeSVtolerances(&cv, reltol, &abstol);
    if check_retval_int(retval, "CVodeSVtolerances") != 0 {
        std::process::exit(1);
    }

    /* Call CVodeSetUserData so the sensitivity params can be accessed
     * from user provided routines. */
    let retval = CVodeSetUserData(&cv, Some(Box::new(data.clone())));
    if check_retval_int(retval, "CVodeSetUserData") != 0 {
        std::process::exit(1);
    }

    /* Call CVodeSetMaxNumSteps to set the maximum number of steps the
     * solver will take in an attempt to reach the next output time. */
    let retval = CVodeSetMaxNumSteps(&cv, MXSTEPS);
    if check_retval_int(retval, "CVodeSetMaxNumSteps") != 0 {
        std::process::exit(1);
    }

    /* Create dense SUNMatrix for use in linear solvers */
    let A = SUNDenseMatrix(NEQ, NEQ, &ctx);
    if check_retval_ptr(&A, "SUNDenseMatrix", 0) != 0 {
        std::process::exit(1);
    }
    let A = A.expect("SUNDenseMatrix");

    /* Create dense SUNLinearSolver object for use by CVode */
    let LS = SUNLinSol_Dense(&y, &A, &ctx);
    if check_retval_ptr(&LS, "SUNLinSol_Dense", 0) != 0 {
        std::process::exit(1);
    }
    let LS = LS.expect("SUNLinSol_Dense");

    /* Call CVodeSetLinearSolver to attach the matrix and linear solver to CVode */
    let retval = CVodeSetLinearSolver(&cv, &LS, Some(&A));
    if check_retval_int(retval, "CVodeSetLinearSolver") != 0 {
        std::process::exit(1);
    }

    /* Specify the Jacobian approximation routine to be used */
    let retval = CVodeSetJacFn(&cv, Some(Jac));
    if check_retval_int(retval, "CVodeSetJacFn") != 0 {
        std::process::exit(1);
    }

    /* Sensitivity-related settings */
    {
        let mut d = data.borrow_mut();
        d.sensi = SUNTRUE; /* sensitivity ON                */
        d.meth = CV_SIMULTANEOUS; /* simultaneous corrector method */
        d.errconS = SUNTRUE; /* full error control            */
        d.fsDQ = SUNFALSE; /* user-provided sensitivity RHS  */
    }

    let Ns: i32 = 3;

    let params = UserDataParams(&data);
    let mut pbar: Vec<sunrealtype> = vec![ZERO; Ns as usize];
    pbar[0] = params[0];
    pbar[1] = params[1];
    pbar[2] = params[2];

    let mut plist: Vec<i32> = vec![0; Ns as usize];
    for is in 0..Ns as usize {
        plist[is] = is as i32;
    }

    let yS0 = N_VCloneVectorArray(Ns, &y).expect("N_VCloneVectorArray");
    for is in 0..Ns as usize {
        N_VConst(ZERO, &yS0[is]);
    }

    let yS = N_VCloneVectorArray(Ns, &y).expect("N_VCloneVectorArray");

    let meth = data.borrow().meth;
    let retval = CVodeSensInit1(&cv, Ns, meth, Some(fS), &yS0);
    if check_retval_int(retval, "CVodeSensInit1") != 0 {
        std::process::exit(1);
    }

    /* Hand CVODES a clone of the shared parameter array (C passes `data->p`,
    the very array the callbacks read through `user_data`). */
    let p: SensParams = data.borrow().p.clone();
    let retval = CVodeSetSensParams(&cv, Some(p), Some(&pbar[..]), Some(&plist[..]));
    if check_retval_int(retval, "CVodeSetSensParams") != 0 {
        std::process::exit(1);
    }

    /*
      Sensitivities are enabled
      Set full error control
      Set user-provided sensitivity RHS
      Run CVODES
    */

    let retval = CVodeSensEEtolerances(&cv);
    if check_retval_int(retval, "CVodeSensEEtolerances") != 0 {
        std::process::exit(1);
    }

    let errconS = data.borrow().errconS;
    let retval = CVodeSetSensErrCon(&cv, errconS);
    if check_retval_int(retval, "CVodeSetSensErrCon") != 0 {
        std::process::exit(1);
    }

    let retval = runCVode(&cv, &y, &yS, &data);
    if check_retval_int(retval, "runCVode") != 0 {
        std::process::exit(1);
    }

    /*
      Change parameters
      Toggle sensitivities OFF
      Reinitialize and run CVODES
    */

    {
        let d = data.borrow();
        let mut pp = d.p.borrow_mut();
        pp[0] = 0.05;
        pp[1] = 2.0e4;
        pp[2] = 2.9e7;
    }

    data.borrow_mut().sensi = SUNFALSE;

    let retval = CVodeReInit(&cv, T0, &y0);
    if check_retval_int(retval, "CVodeReInit") != 0 {
        std::process::exit(1);
    }

    let retval = CVodeSensToggleOff(&cv);
    if check_retval_int(retval, "CVodeSensToggleOff") != 0 {
        std::process::exit(1);
    }

    let retval = runCVode(&cv, &y, &yS, &data);
    if check_retval_int(retval, "runCVode") != 0 {
        std::process::exit(1);
    }

    /*
      Change parameters
      Switch to internal DQ sensitivity RHS function
      Toggle sensitivities ON (reinitialize sensitivities)
      Reinitialize and run CVODES
    */

    {
        let d = data.borrow();
        let mut pp = d.p.borrow_mut();
        pp[0] = 0.06;
        pp[1] = 3.0e4;
        pp[2] = 2.8e7;
    }

    {
        let mut d = data.borrow_mut();
        d.sensi = SUNTRUE;
        d.fsDQ = SUNTRUE;
    }

    let retval = CVodeReInit(&cv, T0, &y0);
    if check_retval_int(retval, "CVodeReInit") != 0 {
        std::process::exit(1);
    }

    CVodeSensFree(&cv);
    let meth = data.borrow().meth;
    let retval = CVodeSensInit1(&cv, Ns, meth, None, &yS0);
    if check_retval_int(retval, "CVodeSensInit1") != 0 {
        std::process::exit(1);
    }

    let retval = runCVode(&cv, &y, &yS, &data);
    if check_retval_int(retval, "runCVode") != 0 {
        std::process::exit(1);
    }

    /*
      Switch to partial error control
      Switch back to user-provided sensitivity RHS
      Toggle sensitivities ON (reinitialize sensitivities)
      Change method to staggered
      Reinitialize and run CVODES
    */

    {
        let mut d = data.borrow_mut();
        d.sensi = SUNTRUE;
        d.errconS = SUNFALSE;
        d.fsDQ = SUNFALSE;
        d.meth = CV_STAGGERED;
    }

    let retval = CVodeReInit(&cv, T0, &y0);
    if check_retval_int(retval, "CVodeReInit") != 0 {
        std::process::exit(1);
    }

    let errconS = data.borrow().errconS;
    let retval = CVodeSetSensErrCon(&cv, errconS);
    if check_retval_int(retval, "CVodeSetSensErrCon") != 0 {
        std::process::exit(1);
    }

    CVodeSensFree(&cv);
    let meth = data.borrow().meth;
    let retval = CVodeSensInit1(&cv, Ns, meth, Some(fS), &yS0);
    if check_retval_int(retval, "CVodeSensInit1") != 0 {
        std::process::exit(1);
    }

    let retval = runCVode(&cv, &y, &yS, &data);
    if check_retval_int(retval, "runCVode") != 0 {
        std::process::exit(1);
    }

    /*
      Free sensitivity-related memory
      (CVodeSensToggle is not needed, as CVodeSensFree toggles sensitivities OFF)
      Reinitialize and run CVODES
    */

    data.borrow_mut().sensi = SUNFALSE;

    CVodeSensFree(&cv);

    let retval = CVodeReInit(&cv, T0, &y0);
    if check_retval_int(retval, "CVodeReInit") != 0 {
        std::process::exit(1);
    }

    let retval = runCVode(&cv, &y, &yS, &data);
    if check_retval_int(retval, "runCVode") != 0 {
        std::process::exit(1);
    }

    /* Free memory */

    N_VDestroy(y0); /* Free y0 vector         */
    N_VDestroy(y); /* Free y vector          */
    N_VDestroy(abstol); /* Free abstol vector     */
    N_VDestroyVectorArray(yS0, Ns); /* Free yS0 vector        */
    N_VDestroyVectorArray(yS, Ns); /* Free yS vector         */
    drop(plist); /* Free plist             */
    drop(pbar); /* Free pbar              */
    /* C `free(data)`: the shared record dies with the last handle; the
    integrator memory still owns its boxed clone until CVodeFree below. */
    let mut cvode_mem = Some(cv);
    CVodeFree(&mut cvode_mem); /* Free integrator memory */
    let _ = SUNLinSolFree(Some(LS)); /* Free solver memory     */
    SUNMatDestroy(A); /* Free the matrix memory */

    let _ = SUNContext_Free(&mut sunctx);
}

/*
 * Runs integrator and prints final statistics when complete.
 */

fn runCVode(
    cvode_mem: &CVodeMem,
    y: &N_Vector,
    _yS: &[N_Vector],
    data: &Rc<RefCell<UserData>>,
) -> i32 {
    let mut t: sunrealtype = 0.0;

    /* Print header for current run */
    PrintHeader(data);

    /* Call CVode in CV_NORMAL mode */
    let retval = CVode(cvode_mem, T1, y, &mut t, CV_NORMAL);
    if retval != 0 {
        return retval;
    }

    /* Print final statistics */
    let retval = PrintFinalStats(cvode_mem, data);
    print!("\n");

    retval
}

/*
 *--------------------------------------------------------------------
 * FUNCTIONS CALLED BY THE SOLVER
 *--------------------------------------------------------------------
 */

/*
 * f routine. Compute f(t,y).
 */

fn f(_t: sunrealtype, y: &N_Vector, ydot: &N_Vector, user_data: &mut Option<Box<dyn Any>>) -> i32 {
    let y1 = NV_Ith_S(y, 0);
    let y2 = NV_Ith_S(y, 1);
    let y3 = NV_Ith_S(y, 2);
    let data = UserDataFromBox(user_data);
    let params = UserDataParams(&data);
    let p1 = params[0];
    let p2 = params[1];
    let p3 = params[2];

    let yd1 = -p1 * y1 + p2 * y2 * y3;
    NV_Ith_S_set(ydot, 0, yd1);
    let yd3 = p3 * y2 * y2;
    NV_Ith_S_set(ydot, 2, yd3);
    NV_Ith_S_set(ydot, 1, -yd1 - yd3);

    0
}

/*
 * Jacobian routine. Compute J(t,y).
 */

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
    let y2 = NV_Ith_S(y, 1);
    let y3 = NV_Ith_S(y, 2);
    let data = UserDataFromBox(user_data);
    let params = UserDataParams(&data);
    let p1 = params[0];
    let p2 = params[1];
    let p3 = params[2];

    SM_ELEMENT_D_set(J, 0, 0, -p1);
    SM_ELEMENT_D_set(J, 0, 1, p2 * y3);
    SM_ELEMENT_D_set(J, 0, 2, p2 * y2);
    SM_ELEMENT_D_set(J, 1, 0, p1);
    SM_ELEMENT_D_set(J, 1, 1, -p2 * y3 - 2.0 * p3 * y2);
    SM_ELEMENT_D_set(J, 1, 2, -p2 * y2);
    SM_ELEMENT_D_set(J, 2, 1, 2.0 * p3 * y2);

    0
}

/*
 * fS routine. Compute sensitivity r.h.s.
 */

fn fS(
    _Ns: i32,
    _t: sunrealtype,
    y: &N_Vector,
    _ydot: &N_Vector,
    iS: i32,
    yS: &N_Vector,
    ySdot: &N_Vector,
    user_data: &mut Option<Box<dyn Any>>,
    _tmp1: &N_Vector,
    _tmp2: &N_Vector,
) -> i32 {
    let data = UserDataFromBox(user_data);
    let params = UserDataParams(&data);
    let p1 = params[0];
    let p2 = params[1];
    let p3 = params[2];

    let y1 = NV_Ith_S(y, 0);
    let y2 = NV_Ith_S(y, 1);
    let y3 = NV_Ith_S(y, 2);
    let s1 = NV_Ith_S(yS, 0);
    let s2 = NV_Ith_S(yS, 1);
    let s3 = NV_Ith_S(yS, 2);

    let mut sd1 = -p1 * s1 + p2 * y3 * s2 + p2 * y2 * s3;
    let mut sd3 = 2.0 * p3 * y2 * s2;
    let mut sd2 = -sd1 - sd3;

    match iS {
        0 => {
            sd1 += -y1;
            sd2 += y1;
        }
        1 => {
            sd1 += y2 * y3;
            sd2 += -y2 * y3;
        }
        2 => {
            sd2 += -y2 * y2;
            sd3 += y2 * y2;
        }
        _ => {}
    }

    NV_Ith_S_set(ySdot, 0, sd1);
    NV_Ith_S_set(ySdot, 1, sd2);
    NV_Ith_S_set(ySdot, 2, sd3);

    0
}

/*
 *--------------------------------------------------------------------
 * PRIVATE FUNCTIONS
 *--------------------------------------------------------------------
 */

fn PrintHeader(data: &Rc<RefCell<UserData>>) {
    let (sensi, meth, errconS, fsDQ) = {
        let d = data.borrow();
        (d.sensi, d.meth, d.errconS, d.fsDQ)
    };

    /* Print sensitivity control retvals */
    print!("Sensitivity: ");
    if sensi {
        print!("YES (");
        if meth == CV_SIMULTANEOUS {
            print!("SIMULTANEOUS + ");
        } else if meth == CV_STAGGERED {
            print!("STAGGERED + ");
        } else if meth == CV_STAGGERED1 {
            print!("STAGGERED-1 + ");
        }
        if errconS {
            print!("FULL ERROR CONTROL + ");
        } else {
            print!("PARTIAL ERROR CONTROL + ");
        }
        if fsDQ {
            print!("DQ sensitivity RHS)\n");
        } else {
            print!("user-provided sensitivity RHS)\n");
        }
    } else {
        print!("NO\n");
    }

    /* Print current problem parameters */
    let params = UserDataParams(data);
    /* C: printf("Parameters: [%8.4e  %8.4e  %8.4e]\n",
                 data->p[0], data->p[1], data->p[2]) */
    print!(
        "Parameters: [{}  {}  {}]\n",
        fmt_ew(params[0], 8, 4),
        fmt_ew(params[1], 8, 4),
        fmt_ew(params[2], 8, 4)
    );
}

/*
 * Print some final statistics from the CVODES memory.
 */

fn PrintFinalStats(cvode_mem: &CVodeMem, data: &Rc<RefCell<UserData>>) -> i32 {
    let mut nst: i64 = 0;
    let mut nfe: i64 = 0;
    let mut nsetups: i64 = 0;
    let mut nni: i64 = 0;
    let mut ncfn: i64 = 0;
    let mut netf: i64 = 0;
    let mut nfSe: i64 = 0;
    let mut nfeS: i64 = 0;
    let mut nsetupsS: i64 = 0;
    let mut nniS: i64 = 0;
    let mut ncfnS: i64 = 0;
    let mut netfS: i64 = 0;
    let mut njeD: i64 = 0;
    let mut nfeD: i64 = 0;

    let (sensi, meth, errconS) = {
        let d = data.borrow();
        (d.sensi, d.meth, d.errconS)
    };

    let _ = CVodeGetNumSteps(cvode_mem, &mut nst);
    let _ = CVodeGetNumRhsEvals(cvode_mem, &mut nfe);
    let _ = CVodeGetNumLinSolvSetups(cvode_mem, &mut nsetups);
    let _ = CVodeGetNumErrTestFails(cvode_mem, &mut netf);
    let _ = CVodeGetNumNonlinSolvIters(cvode_mem, &mut nni);
    let _ = CVodeGetNumNonlinSolvConvFails(cvode_mem, &mut ncfn);

    if sensi {
        let _ = CVodeGetSensNumRhsEvals(cvode_mem, &mut nfSe);
        let _ = CVodeGetNumRhsEvalsSens(cvode_mem, &mut nfeS);
        let _ = CVodeGetSensNumLinSolvSetups(cvode_mem, &mut nsetupsS);
        if errconS {
            let _ = CVodeGetSensNumErrTestFails(cvode_mem, &mut netfS);
        } else {
            netfS = 0;
        }
        if meth == CV_STAGGERED {
            let _ = CVodeGetSensNumNonlinSolvIters(cvode_mem, &mut nniS);
            let _ = CVodeGetSensNumNonlinSolvConvFails(cvode_mem, &mut ncfnS);
        } else {
            nniS = 0;
            ncfnS = 0;
        }
    }

    let _ = CVodeGetNumJacEvals(cvode_mem, &mut njeD);
    let retval = CVodeGetNumLinRhsEvals(cvode_mem, &mut nfeD);

    print!("Run statistics:\n");

    /* C: printf("   nst     = %5ld\n", nst) */
    print!("   nst     = {:>5}\n", nst);
    print!("   nfe     = {:>5}\n", nfe);
    print!("   netf    = {:>5}    nsetups  = {:>5}\n", netf, nsetups);
    print!("   nni     = {:>5}    ncfn     = {:>5}\n", nni, ncfn);

    print!("   njeD    = {:>5}    nfeD     = {:>5}\n", njeD, nfeD);

    if sensi {
        print!("   -----------------------------------\n"); /* simultaneous corrector method */
        print!("   nfSe    = {:>5}    nfeS     = {:>5}\n", nfSe, nfeS);
        print!("   netfs   = {:>5}    nsetupsS = {:>5}\n", netfS, nsetupsS);
        print!("   nniS    = {:>5}    ncfnS    = {:>5}\n", nniS, ncfnS);
    }

    retval
}

/*
 * Check function return value...
 *   opt == 0 means SUNDIALS function allocates memory so check if
 *            returned NULL pointer
 *   opt == 1 means SUNDIALS function returns an integer value so check if
 *            retval < 0 (see check_retval_int)
 *   opt == 2 means function allocates memory so check if returned
 *            NULL pointer
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
