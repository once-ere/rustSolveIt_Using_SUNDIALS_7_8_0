//! Port of `examples/cvodes/serial/cvsRoberts_ASAi_klu.c`.
//!
//! **Solver note.** `SUNLinSol_KLU` here is
//! [`sundials_core::sunlinsol_klu`], backed by the independent pure-Rust
//! sparse LU rather than SuiteSparse KLU. See
//! `differences/ATTRIBUTION.md`.
//!
//! Adjoint sensitivity example problem.
//! The following is a simple example problem, with the coding
//! needed for its solution by CVODES. The problem is from chemical
//! kinetics, and consists of the following three rate equations.
//!    dy1/dt = -p1*y1 + p2*y2*y3
//!    dy2/dt =  p1*y1 - p2*y2*y3 - p3*(y2)^2
//!    dy3/dt =  p3*(y2)^2
//! on the interval from t = 0.0 to t = 4.e10, with initial
//! conditions: y1 = 1.0, y2 = y3 = 0. The reaction rates are:
//! p1=0.04, p2=1e4, and p3=3e7. The problem is stiff.
//! This program solves the problem with the BDF method, Newton
//! iteration with the dense linear solver, and a user-supplied
//! Jacobian routine.
//! It uses a scalar relative tolerance and a vector absolute
//! tolerance.
//! Output is printed in decades from t = .4 to t = 4.e10.
//! Run statistics (optional outputs) are printed at the end.
//!
//! Optionally, CVODES can compute sensitivities with respect to
//! the problem parameters p1, p2, and p3 of the following quantity:
//!   G = int_t0^t1 g(t,p,y) dt
//! where
//!   g(t,p,y) = y3

#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

use std::any::Any;

use cvodes_rs::prelude::*;

/* Accessor macros
   (C macros `Ith(v,i)` = `NV_Ith_S(v,i-1)` and
   `IJth(A,i,j)` = `SM_ELEMENT_D(A,i-1,j-1)`; i, j are 1-based). */

fn Ith(v: &N_Vector, i: usize) -> sunrealtype {
    N_VGetArrayPointer(v).expect("N_VGetArrayPointer")[i - 1]
}

fn Ith_set(v: &N_Vector, i: usize, x: sunrealtype) {
    N_VGetArrayPointer(v).expect("N_VGetArrayPointer")[i - 1] = x;
}


/* Problem Constants */

const NEQ: sunindextype = 3; /* number of equations                  */

const RTOL: sunrealtype = 1e-6; /* scalar relative tolerance            */

const ATOL1: sunrealtype = 1e-8; /* vector absolute tolerance components */
const ATOL2: sunrealtype = 1e-14;
const ATOL3: sunrealtype = 1e-6;

const ATOLl: sunrealtype = 1e-8; /* absolute tolerance for adjoint vars. */
const ATOLq: sunrealtype = 1e-6; /* absolute tolerance for quadratures   */

const T0: sunrealtype = 0.0; /* initial time                         */
const TOUT: sunrealtype = 4e7; /* final time                           */

const TB1: sunrealtype = 4e7; /* starting point for adjoint problem   */
const TB2: sunrealtype = 50.0; /* starting point for adjoint problem   */
const TBout1: sunrealtype = 40.0; /* intermediate t for adjoint problem   */

const STEPS: i64 = 150; /* number of steps between check points */

const NP: sunindextype = 3; /* number of problem parameters         */

const ZERO: sunrealtype = 0.0;

/* Type : UserData */

#[derive(Clone, Copy)]
struct UserData {
    p: [sunrealtype; 3],
}

/*
 *--------------------------------------------------------------------
 * MAIN PROGRAM
 *--------------------------------------------------------------------
 */

fn main() {
    /* Print problem description */
    print!("\nAdjoint Sensitivity Example for Chemical Kinetics\n");
    print!("-------------------------------------------------\n\n");
    print!("ODE: dy1/dt = -p1*y1 + p2*y2*y3\n");
    print!("     dy2/dt =  p1*y1 - p2*y2*y3 - p3*(y2)^2\n");
    print!("     dy3/dt =  p3*(y2)^2\n\n");
    print!("Find dG/dp for\n");
    print!("     G = int_t0^tB0 g(t,p,y) dt\n");
    print!("     g(t,p,y) = y3\n\n\n");

    /* User data structure */
    let data = Some(UserData { p: [ZERO; 3] });
    if check_ptr(&data, "malloc", 2) != 0 {
        std::process::exit(1);
    }
    let mut data = data.unwrap();
    data.p[0] = 0.04;
    data.p[1] = 1.0e4;
    data.p[2] = 3.0e7;

    /* Create the SUNDIALS simulation context that all SUNDIALS objects require */
    let mut sunctx: Option<SUNContext> = None;
    let retval = SUNContext_Create(SUN_COMM_NULL, &mut sunctx);
    if check_retval(retval, "SUNContext_Create") != 0 {
        std::process::exit(1);
    }
    let ctx = sunctx.clone().unwrap();

    /* Initialize y */
    let y = N_VNew_Serial(NEQ, &ctx);
    if check_ptr(&y, "N_VNew_Serial", 0) != 0 {
        std::process::exit(1);
    }
    let y = y.unwrap();
    Ith_set(&y, 1, 1.0);
    Ith_set(&y, 2, ZERO);
    Ith_set(&y, 3, ZERO);

    /* Initialize q */
    let q = N_VNew_Serial(1, &ctx);
    if check_ptr(&q, "N_VNew_Serial", 0) != 0 {
        std::process::exit(1);
    }
    let q = q.unwrap();
    Ith_set(&q, 1, ZERO);

    /* Set the scalar relative and absolute tolerances reltolQ and abstolQ */
    let reltolQ = RTOL;
    let abstolQ = ATOLq;

    /* Create and allocate CVODES memory for forward run */
    print!("Create and allocate CVODES memory for forward runs\n");

    /* Call CVodeCreate to create the solver memory and specify the
    Backward Differentiation Formula */
    let mut cvode_mem = CVodeCreate(CV_BDF, &ctx);
    if check_ptr(&cvode_mem, "CVodeCreate", 0) != 0 {
        std::process::exit(1);
    }
    let cv = cvode_mem.clone().unwrap();

    /* Call CVodeInit to initialize the integrator memory and specify the
    user's right hand side function in y'=f(t,y), the initial time T0, and
    the initial dependent variable vector y. */
    let retval = CVodeInit(&cv, f, T0, &y);
    if check_retval(retval, "CVodeInit") != 0 {
        std::process::exit(1);
    }

    /* Call CVodeWFtolerances to specify a user-supplied function ewt that sets
    the multiplicative error weights w_i for use in the weighted RMS norm */
    let retval = CVodeWFtolerances(&cv, ewt);
    if check_retval(retval, "CVodeWFtolerances") != 0 {
        std::process::exit(1);
    }

    /* Attach user data */
    let retval = CVodeSetUserData(&cv, Some(Box::new(data)));
    if check_retval(retval, "CVodeSetUserData") != 0 {
        std::process::exit(1);
    }

    /* Create dense SUNMatrix for use in linear solves */
    let nnz: sunindextype = NEQ * NEQ; /* max no. of nonzeros entries in the Jac */
    let A = SUNSparseMatrix(NEQ, NEQ, nnz, SUN_CSC_MAT, &ctx);
    if check_ptr(&A, "SUNSparseMatrix", 0) != 0 {
        std::process::exit(1);
    }
    let A = A.unwrap();

    /* Create dense SUNLinearSolver object for use by CVode */
    let LS = SUNLinSol_KLU(&y, &A, &ctx);
    if check_ptr(&LS, "SUNLinSol_KLU", 0) != 0 {
        std::process::exit(1);
    }
    let LS = LS.unwrap();

    /* Attach the matrix and linear solver */
    let retval = CVodeSetLinearSolver(&cv, &LS, Some(&A));
    if check_retval(retval, "CVodeSetLinearSolver") != 0 {
        std::process::exit(1);
    }

    /* Set the user-supplied Jacobian routine Jac */
    let retval = CVodeSetJacFn(&cv, Some(Jac));
    if check_retval(retval, "CVodeSetJacFn") != 0 {
        std::process::exit(1);
    }

    /* Call CVodeQuadInit to allocate initernal memory and initialize
    quadrature integration*/
    let retval = CVodeQuadInit(&cv, fQ, &q);
    if check_retval(retval, "CVodeQuadInit") != 0 {
        std::process::exit(1);
    }

    /* Call CVodeSetQuadErrCon to specify whether or not the quadrature variables
    are to be used in the step size control mechanism within CVODES. Call
    CVodeQuadSStolerances or CVodeQuadSVtolerances to specify the integration
    tolerances for the quadrature variables. */
    let retval = CVodeSetQuadErrCon(&cv, SUNTRUE);
    if check_retval(retval, "CVodeSetQuadErrCon") != 0 {
        std::process::exit(1);
    }

    /* Call CVodeQuadSStolerances to specify scalar relative and absolute
    tolerances. */
    let retval = CVodeQuadSStolerances(&cv, reltolQ, abstolQ);
    if check_retval(retval, "CVodeQuadSStolerances") != 0 {
        std::process::exit(1);
    }

    /* Call CVodeSetMaxNumSteps to set the maximum number of steps the
     * solver will take in an attempt to reach the next output time
     * during forward integration. */
    let retval = CVodeSetMaxNumSteps(&cv, 2500);
    if check_retval(retval, "CVodeSetMaxNumSteps") != 0 {
        std::process::exit(1);
    }

    /* Allocate global memory */

    /* Call CVodeAdjInit to update CVODES memory block by allocating the internal
    memory needed for backward integration.*/
    let steps = STEPS; /* no. of integration steps between two consecutive checkpoints*/
    let retval = CVodeAdjInit(&cv, steps, CV_HERMITE);
    if check_retval(retval, "CVodeAdjInit") != 0 {
        std::process::exit(1);
    }

    /* Perform forward run */
    print!("Forward integration ... ");

    /* Call CVodeF to integrate the forward problem over an interval in time and
    saves checkpointing data */
    let mut time: sunrealtype = 0.0;
    let mut ncheck: i32 = 0;
    let retval = CVodeF(&cv, TOUT, &y, &mut time, CV_NORMAL, &mut ncheck);
    if check_retval(retval, "CVodeF") != 0 {
        std::process::exit(1);
    }

    let mut nst: i64 = 0;
    let retval = CVodeGetNumSteps(&cv, &mut nst);
    if check_retval(retval, "CVodeGetNumSteps") != 0 {
        std::process::exit(1);
    }

    print!("done ( nst = {} )\n", nst);
    print!("\nncheck = {}\n\n", ncheck);

    let retval = CVodeGetQuad(&cv, &mut time, &q);
    if check_retval(retval, "CVodeGetQuad") != 0 {
        std::process::exit(1);
    }

    print!("--------------------------------------------------------\n");
    /* C: printf("G:          %12.4e \n", Ith(q, 1)) */
    print!("G:          {} \n", fmt_ew(Ith(&q, 1), 12, 4));
    print!("--------------------------------------------------------\n\n");

    /* Initialize yB */
    let yB = N_VNew_Serial(NEQ, &ctx);
    if check_ptr(&yB, "N_VNew_Serial", 0) != 0 {
        std::process::exit(1);
    }
    let yB = yB.unwrap();
    Ith_set(&yB, 1, ZERO);
    Ith_set(&yB, 2, ZERO);
    Ith_set(&yB, 3, ZERO);

    /* Initialize qB */
    let qB = N_VNew_Serial(NP, &ctx);
    if check_ptr(&qB, "N_VNew", 0) != 0 {
        std::process::exit(1);
    }
    let qB = qB.unwrap();
    Ith_set(&qB, 1, ZERO);
    Ith_set(&qB, 2, ZERO);
    Ith_set(&qB, 3, ZERO);

    /* Set the scalar relative tolerance reltolB */
    let reltolB = RTOL;

    /* Set the scalar absolute tolerance abstolB */
    let abstolB = ATOLl;

    /* Set the scalar absolute tolerance abstolQB */
    let abstolQB = ATOLq;

    /* Create and allocate CVODES memory for backward run */
    print!("Create and allocate CVODES memory for backward run\n");

    /* Call CVodeCreateB to specify the solution method for the backward
    problem. */
    let mut indexB: i32 = 0;
    let retval = CVodeCreateB(&cv, CV_BDF, &mut indexB);
    if check_retval(retval, "CVodeCreateB") != 0 {
        std::process::exit(1);
    }

    /* Call CVodeInitB to allocate internal memory and initialize the
    backward problem. */
    let retval = CVodeInitB(&cv, indexB, fB, TB1, &yB);
    if check_retval(retval, "CVodeInitB") != 0 {
        std::process::exit(1);
    }

    /* Set the scalar relative and absolute tolerances. */
    let retval = CVodeSStolerancesB(&cv, indexB, reltolB, abstolB);
    if check_retval(retval, "CVodeSStolerancesB") != 0 {
        std::process::exit(1);
    }

    /* Attach the user data for backward problem. */
    let retval = CVodeSetUserDataB(&cv, indexB, Some(Box::new(data)));
    if check_retval(retval, "CVodeSetUserDataB") != 0 {
        std::process::exit(1);
    }

    /* Create dense SUNMatrix for use in linear solves */
    let AB = SUNSparseMatrix(NEQ, NEQ, nnz, SUN_CSC_MAT, &ctx);
    if check_ptr(&AB, "SUNSparseMatrix", 0) != 0 {
        std::process::exit(1);
    }
    let AB = AB.unwrap();

    /* Create dense SUNLinearSolver object */
    let LSB = SUNLinSol_KLU(&yB, &AB, &ctx);
    if check_ptr(&LSB, "SUNLinSol_KLU", 0) != 0 {
        std::process::exit(1);
    }
    let LSB = LSB.unwrap();

    /* Attach the matrix and linear solver */
    let retval = CVodeSetLinearSolverB(&cv, indexB, &LSB, Some(&AB));
    if check_retval(retval, "CVodeSetLinearSolverB") != 0 {
        std::process::exit(1);
    }

    /* Set the user-supplied Jacobian routine JacB */
    let retval = CVodeSetJacFnB(&cv, indexB, Some(JacB));
    if check_retval(retval, "CVodeSetJacFnB") != 0 {
        std::process::exit(1);
    }

    /* Call CVodeQuadInitB to allocate internal memory and initialize backward
    quadrature integration. */
    let retval = CVodeQuadInitB(&cv, indexB, fQB, &qB);
    if check_retval(retval, "CVodeQuadInitB") != 0 {
        std::process::exit(1);
    }

    /* Call CVodeSetQuadErrCon to specify whether or not the quadrature variables
    are to be used in the step size control mechanism within CVODES. Call
    CVodeQuadSStolerances or CVodeQuadSVtolerances to specify the integration
    tolerances for the quadrature variables. */
    let retval = CVodeSetQuadErrConB(&cv, indexB, SUNTRUE);
    if check_retval(retval, "CVodeSetQuadErrConB") != 0 {
        std::process::exit(1);
    }

    /* Call CVodeQuadSStolerancesB to specify the scalar relative and absolute
    tolerances for the backward problem. */
    let retval = CVodeQuadSStolerancesB(&cv, indexB, reltolB, abstolQB);
    if check_retval(retval, "CVodeQuadSStolerancesB") != 0 {
        std::process::exit(1);
    }

    /* Backward Integration */

    PrintHead(TB1);

    /* First get results at t = TBout1 */

    /* Call CVodeB to integrate the backward ODE problem. */
    let retval = CVodeB(&cv, TBout1, CV_NORMAL);
    if check_retval(retval, "CVodeB") != 0 {
        std::process::exit(1);
    }

    /* Call CVodeGetB to get yB of the backward ODE problem. */
    let retval = CVodeGetB(&cv, indexB, &mut time, &yB);
    if check_retval(retval, "CVodeGetB") != 0 {
        std::process::exit(1);
    }

    /* Call CVodeGetAdjY to get the interpolated value of the forward solution
    y during a backward integration. */
    let retval = CVodeGetAdjY(&cv, TBout1, &y);
    if check_retval(retval, "CVodeGetAdjY") != 0 {
        std::process::exit(1);
    }

    PrintOutput1(time, TBout1, &y, &yB);

    /* Then at t = T0 */

    let retval = CVodeB(&cv, T0, CV_NORMAL);
    if check_retval(retval, "CVodeB") != 0 {
        std::process::exit(1);
    }

    let retval = CVodeGetB(&cv, indexB, &mut time, &yB);
    if check_retval(retval, "CVodeGetB") != 0 {
        std::process::exit(1);
    }

    /* Call CVodeGetQuadB to get the quadrature solution vector after a
    successful return from CVodeB. */
    let retval = CVodeGetQuadB(&cv, indexB, &mut time, &qB);
    if check_retval(retval, "CVodeGetQuadB") != 0 {
        std::process::exit(1);
    }

    let retval = CVodeGetAdjY(&cv, T0, &y);
    if check_retval(retval, "CVodeGetAdjY") != 0 {
        std::process::exit(1);
    }

    let cvB = CVodeGetAdjCVodeBmem(&cv, indexB).expect("CVodeGetAdjCVodeBmem");
    let mut nstB: i64 = 0;
    let _ = CVodeGetNumSteps(&cvB, &mut nstB);
    print!("Done ( nst = {} )\n", nstB);

    PrintOutput(time, &y, &yB, &qB);

    /* Reinitialize backward phase (new tB0) */

    Ith_set(&yB, 1, ZERO);
    Ith_set(&yB, 2, ZERO);
    Ith_set(&yB, 3, ZERO);

    Ith_set(&qB, 1, ZERO);
    Ith_set(&qB, 2, ZERO);
    Ith_set(&qB, 3, ZERO);

    print!("Re-initialize CVODES memory for backward run\n");

    let retval = CVodeReInitB(&cv, indexB, TB2, &yB);
    if check_retval(retval, "CVodeReInitB") != 0 {
        std::process::exit(1);
    }

    let retval = CVodeQuadReInitB(&cv, indexB, &qB);
    if check_retval(retval, "CVodeQuadReInitB") != 0 {
        std::process::exit(1);
    }

    PrintHead(TB2);

    /* First get results at t = TBout1 */

    let retval = CVodeB(&cv, TBout1, CV_NORMAL);
    if check_retval(retval, "CVodeB") != 0 {
        std::process::exit(1);
    }

    let retval = CVodeGetB(&cv, indexB, &mut time, &yB);
    if check_retval(retval, "CVodeGetB") != 0 {
        std::process::exit(1);
    }

    let retval = CVodeGetAdjY(&cv, TBout1, &y);
    if check_retval(retval, "CVodeGetAdjY") != 0 {
        std::process::exit(1);
    }

    PrintOutput1(time, TBout1, &y, &yB);

    /* Then at t = T0 */

    let retval = CVodeB(&cv, T0, CV_NORMAL);
    if check_retval(retval, "CVodeB") != 0 {
        std::process::exit(1);
    }

    let retval = CVodeGetB(&cv, indexB, &mut time, &yB);
    if check_retval(retval, "CVodeGetB") != 0 {
        std::process::exit(1);
    }

    let retval = CVodeGetQuadB(&cv, indexB, &mut time, &qB);
    if check_retval(retval, "CVodeGetQuadB") != 0 {
        std::process::exit(1);
    }

    let retval = CVodeGetAdjY(&cv, T0, &y);
    if check_retval(retval, "CVodeGetAdjY") != 0 {
        std::process::exit(1);
    }

    let cvB = CVodeGetAdjCVodeBmem(&cv, indexB).expect("CVodeGetAdjCVodeBmem");
    let mut nstB: i64 = 0;
    let _ = CVodeGetNumSteps(&cvB, &mut nstB);
    print!("Done ( nst = {} )\n", nstB);

    PrintOutput(time, &y, &yB, &qB);

    /* Free memory */
    print!("Free memory\n\n");

    drop(cvB);
    CVodeFree(&mut cvode_mem);
    N_VDestroy(y);
    N_VDestroy(q);
    N_VDestroy(yB);
    N_VDestroy(qB);
    let _ = SUNLinSolFree(Some(LS));
    SUNMatDestroy(A);
    let _ = SUNLinSolFree(Some(LSB));
    SUNMatDestroy(AB);
    let _ = SUNContext_Free(&mut sunctx);
}

/*
 *-------------------------------
 * Functions called by the solver
 *-------------------------------
 */

/*
 * f routine. Compute function f(t,y).
 */

fn f(_t: sunrealtype, y: &N_Vector, ydot: &N_Vector, user_data: &mut Option<Box<dyn Any>>) -> i32 {
    let y1 = Ith(y, 1);
    let y2 = Ith(y, 2);
    let y3 = Ith(y, 3);
    let data = user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<UserData>())
        .expect("user_data is UserData");
    let p1 = data.p[0];
    let p2 = data.p[1];
    let p3 = data.p[2];

    let yd1 = -p1 * y1 + p2 * y2 * y3;
    Ith_set(ydot, 1, yd1);
    let yd3 = p3 * y2 * y2;
    Ith_set(ydot, 3, yd3);
    Ith_set(ydot, 2, -yd1 - yd3);

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
    let yval = N_VGetArrayPointer(y).expect("N_VGetArrayPointer");
    let ud = user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<UserData>())
        .expect("UserData");
    let p1 = ud.p[0];
    let p2 = ud.p[1];
    let p3 = ud.p[2];

    SUNMatZero(J);

    /* One borrow for all three arrays: taking them separately would be a
    second mutable borrow of the same matrix content. */
    let mut m = SUNSparseMatrix_Content(J);
    let m = &mut *m;
    let (data, rowvals, colptrs) = (&mut m.data, &mut m.indexvals, &mut m.indexptrs);

    colptrs[0] = 0;
    colptrs[1] = 3;
    colptrs[2] = 6;
    colptrs[3] = 9;

    data[0] = -p1;
    rowvals[0] = 0;
    data[1] = p1;
    rowvals[1] = 1;
    data[2] = ZERO;
    rowvals[2] = 2;

    data[3] = p2 * yval[2];
    rowvals[3] = 0;
    data[4] = -p2 * yval[2] - 2.0 * p3 * yval[1];
    rowvals[4] = 1;
    data[5] = 2.0 * yval[1];
    rowvals[5] = 2;

    data[6] = p2 * yval[1];
    rowvals[6] = 0;
    data[7] = -p2 * yval[1];
    rowvals[7] = 1;
    data[8] = ZERO;
    rowvals[8] = 2;

    0
}

/*
 * fQ routine. Compute fQ(t,y).
 */

fn fQ(
    _t: sunrealtype,
    y: &N_Vector,
    qdot: &N_Vector,
    _user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    Ith_set(qdot, 1, Ith(y, 3));

    0
}

/*
 * EwtSet function. Computes the error weights at the current solution.
 */

fn ewt(y: &N_Vector, w: &N_Vector, _user_data: &mut Option<Box<dyn Any>>) -> i32 {
    let mut atol = [0.0 as sunrealtype; 3];

    let rtol = RTOL;
    atol[0] = ATOL1;
    atol[1] = ATOL2;
    atol[2] = ATOL3;

    for i in 1..=3 {
        let yy = Ith(y, i);
        let ww = rtol * SUNRabs(yy) + atol[i - 1];
        if ww <= 0.0 {
            return -1;
        }
        Ith_set(w, i, 1.0 / ww);
    }

    0
}

/*
 * fB routine. Compute fB(t,y,yB).
 */

fn fB(
    _t: sunrealtype,
    y: &N_Vector,
    yB: &N_Vector,
    yBdot: &N_Vector,
    user_dataB: &mut Option<Box<dyn Any>>,
) -> i32 {
    let data = user_dataB
        .as_mut()
        .and_then(|b| b.downcast_mut::<UserData>())
        .expect("user_dataB is UserData");

    /* The p vector */
    let p1 = data.p[0];
    let p2 = data.p[1];
    let p3 = data.p[2];

    /* The y vector */
    let y2 = Ith(y, 2);
    let y3 = Ith(y, 3);

    /* The lambda vector */
    let l1 = Ith(yB, 1);
    let l2 = Ith(yB, 2);
    let l3 = Ith(yB, 3);

    /* Temporary variables */
    let l21 = l2 - l1;
    let l32 = l3 - l2;

    /* Load yBdot */
    Ith_set(yBdot, 1, -p1 * l21);
    Ith_set(yBdot, 2, p2 * y3 * l21 - 2.0 * p3 * y2 * l32);
    Ith_set(yBdot, 3, p2 * y2 * l21 - 1.0);

    0
}

/*
 * JacB routine. Compute JB(t,y,yB).
 */

fn JacB(
    _t: sunrealtype,
    y: &N_Vector,
    _yB: &N_Vector,
    _fyB: &N_Vector,
    JB: &SUNMatrix,
    user_dataB: &mut Option<Box<dyn Any>>,
    _tmp1B: &N_Vector,
    _tmp2B: &N_Vector,
    _tmp3B: &N_Vector,
) -> i32 {
    let yvalB = N_VGetArrayPointer(y).expect("N_VGetArrayPointer");
    let ud = user_dataB
        .as_mut()
        .and_then(|b| b.downcast_mut::<UserData>())
        .expect("UserData");
    let p1 = ud.p[0];
    let p2 = ud.p[1];
    let p3 = ud.p[2];

    SUNMatZero(JB);

    /* One borrow for all three arrays, as in Jac above. */
    let mut m = SUNSparseMatrix_Content(JB);
    let m = &mut *m;
    let (dataB, rowvalsB, colptrsB) = (&mut m.data, &mut m.indexvals, &mut m.indexptrs);

    colptrsB[0] = 0;
    colptrsB[1] = 3;
    colptrsB[2] = 6;
    colptrsB[3] = 9;

    dataB[0] = p1;
    rowvalsB[0] = 0;
    dataB[1] = -p2 * yvalB[2];
    rowvalsB[1] = 1;
    dataB[2] = -p2 * yvalB[1];
    rowvalsB[2] = 2;

    dataB[3] = -p1;
    rowvalsB[3] = 0;
    dataB[4] = p2 * yvalB[2] + 2.0 * p3 * yvalB[1];
    rowvalsB[4] = 1;
    dataB[5] = p2 * yvalB[1];
    rowvalsB[5] = 2;

    dataB[6] = ZERO;
    rowvalsB[6] = 0;
    dataB[7] = -2.0 * p3 * yvalB[1];
    rowvalsB[7] = 1;
    dataB[8] = ZERO;
    rowvalsB[8] = 2;

    0
}

/*
 * fQB routine. Compute integrand for quadratures
 */

fn fQB(
    _t: sunrealtype,
    y: &N_Vector,
    yB: &N_Vector,
    qBdot: &N_Vector,
    _user_dataB: &mut Option<Box<dyn Any>>,
) -> i32 {
    /* The y vector */
    let y1 = Ith(y, 1);
    let y2 = Ith(y, 2);
    let y3 = Ith(y, 3);

    /* The lambda vector */
    let l1 = Ith(yB, 1);
    let l2 = Ith(yB, 2);
    let l3 = Ith(yB, 3);

    /* Temporary variables */
    let l21 = l2 - l1;
    let l32 = l3 - l2;
    let y23 = y2 * y3;

    Ith_set(qBdot, 1, y1 * l21);
    Ith_set(qBdot, 2, -y23 * l21);
    Ith_set(qBdot, 3, y2 * y2 * l32);

    0
}

/*
 *--------------------------------------------------------------------
 * PRIVATE FUNCTIONS
 *--------------------------------------------------------------------
 */

/*
 * Print heading for backward integration
 */

fn PrintHead(tB0: sunrealtype) {
    /* C: printf("Backward integration from tB0 = %12.4e\n\n", tB0) */
    print!("Backward integration from tB0 = {}\n\n", fmt_ew(tB0, 12, 4));
}

/*
 * Print intermediate results during backward integration
 */

fn PrintOutput1(time: sunrealtype, t: sunrealtype, y: &N_Vector, yB: &N_Vector) {
    print!("--------------------------------------------------------\n");
    print!("returned t: {}\n", fmt_ew(time, 12, 4));
    print!("tout:       {}\n", fmt_ew(t, 12, 4));
    print!(
        "lambda(t):  {} {} {}\n",
        fmt_ew(Ith(yB, 1), 12, 4),
        fmt_ew(Ith(yB, 2), 12, 4),
        fmt_ew(Ith(yB, 3), 12, 4)
    );
    print!(
        "y(t):       {} {} {}\n",
        fmt_ew(Ith(y, 1), 12, 4),
        fmt_ew(Ith(y, 2), 12, 4),
        fmt_ew(Ith(y, 3), 12, 4)
    );
    print!("--------------------------------------------------------\n\n");
}

/*
 * Print final results of backward integration
 */

fn PrintOutput(tfinal: sunrealtype, y: &N_Vector, yB: &N_Vector, qB: &N_Vector) {
    print!("--------------------------------------------------------\n");
    print!("returned t: {}\n", fmt_ew(tfinal, 12, 4));
    print!(
        "lambda(t0): {} {} {}\n",
        fmt_ew(Ith(yB, 1), 12, 4),
        fmt_ew(Ith(yB, 2), 12, 4),
        fmt_ew(Ith(yB, 3), 12, 4)
    );
    print!(
        "y(t0):      {} {} {}\n",
        fmt_ew(Ith(y, 1), 12, 4),
        fmt_ew(Ith(y, 2), 12, 4),
        fmt_ew(Ith(y, 3), 12, 4)
    );
    print!(
        "dG/dp:      {} {} {}\n",
        fmt_ew(-Ith(qB, 1), 12, 4),
        fmt_ew(-Ith(qB, 2), 12, 4),
        fmt_ew(-Ith(qB, 3), 12, 4)
    );
    print!("--------------------------------------------------------\n\n");
}

/*
 * Check function return value...
 *   opt == 0 means SUNDIALS function allocates memory so check if
 *            returned NULL pointer
 *   opt == 1 means SUNDIALS function returns an integer value so check if
 *            retval < 0
 *   opt == 2 means function allocates memory so check if returned
 *            NULL pointer
 *
 * (The C void-pointer/opt polymorphism splits into two typed helpers with
 * identical message text: `check_ptr` covers opt 0 and 2, `check_retval`
 * covers opt 1.)
 */

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

fn check_retval(retval: i32, funcname: &str) -> i32 {
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
