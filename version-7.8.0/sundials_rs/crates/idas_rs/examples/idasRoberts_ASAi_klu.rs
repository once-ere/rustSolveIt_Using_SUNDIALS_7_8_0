//! Port of `examples/idas/serial/idasRoberts_ASAi_klu.c`.
//!
//! **Solver note.** `SUNLinSol_KLU` here is
//! [`sundials_core::sunlinsol_klu`], backed by the independent pure-Rust
//! sparse LU rather than SuiteSparse KLU. See
//! `differences/ATTRIBUTION.md`.
//!
//! Adjoint sensitivity example problem.
//!
//! This simple example problem for IDAS, due to Robertson,
//! is from chemical kinetics, and consists of the following three
//! equations:
//!
//!      dy1/dt + p1*y1 - p2*y2*y3            = 0
//!      dy2/dt - p1*y1 + p2*y2*y3 + p3*y2**2 = 0
//!                 y1  +  y2  +  y3  -  1    = 0
//!
//! on the interval from t = 0.0 to t = 4.e10, with initial
//! conditions: y1 = 1, y2 = y3 = 0.The reaction rates are: p1=0.04,
//! p2=1e4, and p3=3e7
//!
//! It uses a scalar relative tolerance and a vector absolute
//! tolerance.
//!
//! IDAS can also compute sensitivities with respect to
//! the problem parameters p1, p2, and p3 of the following quantity:
//!   G = int_t0^t1 g(t,p,y) dt
//! where
//!   g(t,p,y) = y3
//!
//! The gradient dG/dp is obtained as:
//!   dG/dp = int_t0^t1 (g_p - lambda^T F_p ) dt -
//!           lambda^T*F_y'*y_p | _t0^t1
//!         = int_t0^t1 (lambda^T*F_p) dt
//! where lambda and are solutions of the adjoint system:
//!   d(lambda^T * F_y' )/dt -lambda^T F_y = -g_y
//!
//! During the backward integration, IDAS also evaluates G as
//!   G = - phi(t0)
//! where
//!   d(phi)/dt = g(t,y,p)
//!   phi(t1) = 0

#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

use std::any::Any;

use idas_rs::prelude::*;

/* Accessor macros

C: `Ith(v,i)` = `NV_Ith_S(v,i-1)` -- i-th vector component i = 1..NEQ.
C: `IJth(A,i,j)` = `SM_ELEMENT_D(A,i-1,j-1)` -- (i,j)-th matrix component.
The `RefMut` guard is taken and dropped inside these helpers, so it is
never held across a library call. */

fn Ith(v: &N_Vector, i: usize) -> sunrealtype {
    N_VGetArrayPointer(v).expect("N_VGetArrayPointer")[i - 1]
}

fn Ith_set(v: &N_Vector, i: usize, x: sunrealtype) {
    N_VGetArrayPointer(v).expect("N_VGetArrayPointer")[i - 1] = x;
}


/* Problem Constants */

const NEQ: sunindextype = 3; /* number of equations                  */

const RTOL: sunrealtype = 1e-06; /* scalar relative tolerance            */

const ATOL1: sunrealtype = 1e-08; /* vector absolute tolerance components */
const ATOL2: sunrealtype = 1e-12;
const ATOL3: sunrealtype = 1e-08;

const ATOLA: sunrealtype = 1e-08; /* absolute tolerance for adjoint vars. */
const ATOLQ: sunrealtype = 1e-06; /* absolute tolerance for quadratures   */

const T0: sunrealtype = 0.0; /* initial time                         */
const TOUT: sunrealtype = 4e10; /* final time                           */

const TB1: sunrealtype = 50.0; /* starting point for adjoint problem   */
const TB2: sunrealtype = TOUT; /* starting point for adjoint problem   */

const T1B: sunrealtype = 49.0; /* for IDACalcICB                       */

const STEPS: i64 = 100; /* number of steps between check points */

const NP: sunindextype = 3; /* number of problem parameters         */

const ONE: sunrealtype = 1.0;
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
    let mut retval: i32;
    let mut indexB: i32 = 0;
    let mut ncheck: i32 = 0;
    let mut time: sunrealtype = 0.0;

    /* Print problem description */
    print!("\nAdjoint Sensitivity Example for Chemical Kinetics\n");
    print!("-------------------------------------------------\n\n");
    print!("DAE: dy1/dt + p1*y1 - p2*y2*y3 = 0\n");
    print!("     dy2/dt - p1*y1 + p2*y2*y3 + p3*(y2)^2 = 0\n");
    print!("               y1  +  y2  +  y3 = 0\n\n");
    print!("Find dG/dp for\n");
    print!("     G = int_t0^tB0 g(t,p,y) dt\n");
    print!("     g(t,p,y) = y3\n\n\n");

    /* Create the SUNDIALS context object for this simulation */
    let mut sunctx: Option<SUNContext> = None;
    retval = SUNContext_Create(SUN_COMM_NULL, &mut sunctx);
    if check_retval(Some(retval), "SUNContext_Create", 1) != 0 {
        std::process::exit(1);
    }
    let ctx = sunctx.as_ref().expect("SUNContext").clone();

    /* User data structure */
    let data: Option<UserData> = Some(UserData { p: [ZERO; 3] });
    if check_retval(data.as_ref().map(|_| 0), "malloc", 2) != 0 {
        std::process::exit(1);
    }
    let mut data = data.expect("malloc");
    data.p[0] = 0.04;
    data.p[1] = 1.0e4;
    data.p[2] = 3.0e7;

    /* Initialize y */
    let yy = N_VNew_Serial(NEQ, &ctx);
    if check_retval(yy.as_ref().map(|_| 0), "N_VNew_Serial", 0) != 0 {
        std::process::exit(1);
    }
    let yy = yy.expect("N_VNew_Serial");
    Ith_set(&yy, 1, ONE);
    Ith_set(&yy, 2, ZERO);
    Ith_set(&yy, 3, ZERO);

    /* Initialize yprime */
    let yp = N_VClone(&yy);
    if check_retval(yp.as_ref().map(|_| 0), "N_VNew_Serial", 0) != 0 {
        std::process::exit(1);
    }
    let yp = yp.expect("N_VClone");
    Ith_set(&yp, 1, -0.04);
    Ith_set(&yp, 2, 0.04);
    Ith_set(&yp, 3, ZERO);

    /* Initialize q */
    let q = N_VNew_Serial(1, &ctx);
    if check_retval(q.as_ref().map(|_| 0), "N_VNew_Serial", 0) != 0 {
        std::process::exit(1);
    }
    let q = q.expect("N_VNew_Serial");
    Ith_set(&q, 1, ZERO);

    /* Set the scalar relative and absolute tolerances reltolQ and abstolQ */
    let reltolQ: sunrealtype = RTOL;
    let abstolQ: sunrealtype = ATOLQ;

    /* Create and allocate IDAS memory for forward run */
    print!("Create and allocate IDAS memory for forward runs\n");

    let ida_mem = IDACreate(&ctx);
    if check_retval(ida_mem.as_ref().map(|_| 0), "IDACreate", 0) != 0 {
        std::process::exit(1);
    }
    let mut ida_mem_opt = ida_mem;
    let ida_mem = ida_mem_opt.as_ref().expect("IDACreate").clone();

    retval = IDAInit(&ida_mem, res, T0, &yy, &yp);
    if check_retval(Some(retval), "IDAInit", 1) != 0 {
        std::process::exit(1);
    }

    retval = IDAWFtolerances(&ida_mem, ewt);
    if check_retval(Some(retval), "IDAWFtolerances", 1) != 0 {
        std::process::exit(1);
    }

    retval = IDASetUserData(&ida_mem, Some(Box::new(data)));
    if check_retval(Some(retval), "IDASetUserData", 1) != 0 {
        std::process::exit(1);
    }

    /* Create dense SUNMatrix for use in linear solves */
    let nnz: sunindextype = NEQ * NEQ;
    let A = SUNSparseMatrix(NEQ, NEQ, nnz, SUN_CSC_MAT, &ctx);
    if check_retval(A.as_ref().map(|_| 0), "SUNSparseMatrix", 0) != 0 {
        std::process::exit(1);
    }
    let A = A.expect("SUNSparseMatrix");

    /* Create dense SUNLinearSolver object */
    let LS = SUNLinSol_KLU(&yy, &A, &ctx);
    if check_retval(LS.as_ref().map(|_| 0), "SUNLinSol_KLU", 0) != 0 {
        std::process::exit(1);
    }
    let LS = LS.expect("SUNLinSol_KLU");

    /* Attach the matrix and linear solver */
    retval = IDASetLinearSolver(&ida_mem, &LS, Some(&A));
    if check_retval(Some(retval), "IDASetLinearSolver", 1) != 0 {
        std::process::exit(1);
    }

    /* Set the user-supplied Jacobian routine */
    retval = IDASetJacFn(&ida_mem, Some(Jac));
    if check_retval(Some(retval), "IDASetJacFn", 1) != 0 {
        std::process::exit(1);
    }

    /* Setup quadrature integration */
    retval = IDAQuadInit(&ida_mem, rhsQ, &q);
    if check_retval(Some(retval), "IDAQuadInit", 1) != 0 {
        std::process::exit(1);
    }

    retval = IDAQuadSStolerances(&ida_mem, reltolQ, abstolQ);
    if check_retval(Some(retval), "IDAQuadSStolerances", 1) != 0 {
        std::process::exit(1);
    }

    retval = IDASetQuadErrCon(&ida_mem, SUNTRUE);
    if check_retval(Some(retval), "IDASetQuadErrCon", 1) != 0 {
        std::process::exit(1);
    }

    /* Call IDASetMaxNumSteps to set the maximum number of steps the
     * solver will take in an attempt to reach the next output time
     * during forward integration. */
    retval = IDASetMaxNumSteps(&ida_mem, 2500);
    if check_retval(Some(retval), "IDASetMaxNumSteps", 1) != 0 {
        std::process::exit(1);
    }

    /* Allocate global memory */

    let steps: i64 = STEPS;
    retval = IDAAdjInit(&ida_mem, steps, IDA_HERMITE);
    /*retval = IDAAdjInit(&ida_mem, steps, IDA_POLYNOMIAL);*/
    if check_retval(Some(retval), "IDAAdjInit", 1) != 0 {
        std::process::exit(1);
    }

    /* Perform forward run */
    print!("Forward integration ... ");

    /* Integrate till TB1 and get the solution (y, y') at that time. */
    retval = IDASolveF(&ida_mem, TB1, &mut time, &yy, &yp, IDA_NORMAL, &mut ncheck);
    if check_retval(Some(retval), "IDASolveF", 1) != 0 {
        std::process::exit(1);
    }

    let yyTB1 = N_VClone(&yy).expect("N_VClone");
    let ypTB1 = N_VClone(&yp).expect("N_VClone");
    /* Save the states at t=TB1. */
    N_VScale(ONE, &yy, &yyTB1);
    N_VScale(ONE, &yp, &ypTB1);

    /* Continue integrating till TOUT is reached. */
    retval = IDASolveF(&ida_mem, TOUT, &mut time, &yy, &yp, IDA_NORMAL, &mut ncheck);
    if check_retval(Some(retval), "IDASolveF", 1) != 0 {
        std::process::exit(1);
    }

    let mut nst: i64 = 0;
    retval = IDAGetNumSteps(&ida_mem, &mut nst);
    if check_retval(Some(retval), "IDAGetNumSteps", 1) != 0 {
        std::process::exit(1);
    }

    print!("done ( nst = {} )\n", nst);

    retval = IDAGetQuad(&ida_mem, &mut time, &q);
    if check_retval(Some(retval), "IDAGetQuad", 1) != 0 {
        std::process::exit(1);
    }

    print!("--------------------------------------------------------\n");
    print!("G:          {} \n", fmt_ew(Ith(&q, 1), 12, 4));
    print!("--------------------------------------------------------\n\n");

    /* Test check point linked list
    (uncomment next block to print check point information) */

    /*
    {
      int i;

      printf("\nList of Check Points (ncheck = %d)\n\n", ncheck);
      ckpnt = (IDAadjCheckPointRec *) malloc ( (ncheck+1)*sizeof(IDAadjCheckPointRec));
      IDAGetAdjCheckPointsInfo(ida_mem, ckpnt);
      for (i=0;i<=ncheck;i++) {
        printf("Address:       %p\n",ckpnt[i].my_addr);
        printf("Next:          %p\n",ckpnt[i].next_addr);
        printf("Time interval: %le  %le\n",ckpnt[i].t0, ckpnt[i].t1);
        printf("Step number:   %ld\n",ckpnt[i].nstep);
        printf("Order:         %d\n",ckpnt[i].order);
        printf("Step size:     %le\n",ckpnt[i].step);
        printf("\n");
      }

    }
    */
    let _ = ncheck;

    /* Create BACKWARD problem. */

    /* Allocate yB (i.e. lambda_0). */
    let yB = N_VClone(&yy);
    if check_retval(yB.as_ref().map(|_| 0), "N_VNew_Serial", 0) != 0 {
        std::process::exit(1);
    }
    let yB = yB.expect("N_VClone");

    /* Consistently initialize yB. */
    Ith_set(&yB, 1, ZERO);
    Ith_set(&yB, 2, ZERO);
    Ith_set(&yB, 3, ONE);

    /* Allocate ypB (i.e. lambda'_0). */
    let ypB = N_VClone(&yy);
    if check_retval(ypB.as_ref().map(|_| 0), "N_VNew_Serial", 0) != 0 {
        std::process::exit(1);
    }
    let ypB = ypB.expect("N_VClone");

    /* Consistently initialize ypB. */
    Ith_set(&ypB, 1, ONE);
    Ith_set(&ypB, 2, ONE);
    Ith_set(&ypB, 3, ZERO);

    /* Set the scalar relative tolerance reltolB */
    let reltolB: sunrealtype = RTOL;

    /* Set the scalar absolute tolerance abstolB */
    let abstolB: sunrealtype = ATOLA;

    /* Set the scalar absolute tolerance abstolQB */
    let abstolQB: sunrealtype = ATOLQ;

    /* Create and allocate IDAS memory for backward run */
    print!("Create and allocate IDAS memory for backward run\n");

    retval = IDACreateB(&ida_mem, &mut indexB);
    if check_retval(Some(retval), "IDACreateB", 1) != 0 {
        std::process::exit(1);
    }

    retval = IDAInitB(&ida_mem, indexB, resB, TB2, &yB, &ypB);
    if check_retval(Some(retval), "IDAInitB", 1) != 0 {
        std::process::exit(1);
    }

    retval = IDASStolerancesB(&ida_mem, indexB, reltolB, abstolB);
    if check_retval(Some(retval), "IDASStolerancesB", 1) != 0 {
        std::process::exit(1);
    }

    retval = IDASetUserDataB(&ida_mem, indexB, Some(Box::new(data)));
    if check_retval(Some(retval), "IDASetUserDataB", 1) != 0 {
        std::process::exit(1);
    }

    retval = IDASetMaxNumStepsB(&ida_mem, indexB, 1000);
    if check_retval(Some(retval), "IDASetMaxNumStepsB", 1) != 0 {
        std::process::exit(1);
    }

    /* Create dense SUNMatrix for use in linear solves */
    let AB = SUNSparseMatrix(NEQ, NEQ, nnz, SUN_CSC_MAT, &ctx);
    if check_retval(AB.as_ref().map(|_| 0), "SUNSparseMatrix", 0) != 0 {
        std::process::exit(1);
    }
    let AB = AB.expect("SUNSparseMatrix");

    /* Create dense SUNLinearSolver object */
    let LSB = SUNLinSol_KLU(&yB, &AB, &ctx);
    if check_retval(LSB.as_ref().map(|_| 0), "SUNLinSol_KLU", 0) != 0 {
        std::process::exit(1);
    }
    let LSB = LSB.expect("SUNLinSol_KLU");

    /* Attach the matrix and linear solver */
    retval = IDASetLinearSolverB(&ida_mem, indexB, &LSB, Some(&AB));
    if check_retval(Some(retval), "IDASetLinearSolverB", 1) != 0 {
        std::process::exit(1);
    }

    /* Set the user-supplied Jacobian routine */
    retval = IDASetJacFnB(&ida_mem, indexB, Some(JacB));
    if check_retval(Some(retval), "IDASetJacFnB", 1) != 0 {
        std::process::exit(1);
    }

    /* Quadrature for backward problem. */

    /* Initialize qB */
    let qB = N_VNew_Serial(NP, &ctx);
    if check_retval(qB.as_ref().map(|_| 0), "N_VNew", 0) != 0 {
        std::process::exit(1);
    }
    let qB = qB.expect("N_VNew_Serial");
    Ith_set(&qB, 1, ZERO);
    Ith_set(&qB, 2, ZERO);
    Ith_set(&qB, 3, ZERO);

    retval = IDAQuadInitB(&ida_mem, indexB, rhsQB, &qB);
    if check_retval(Some(retval), "IDAQuadInitB", 1) != 0 {
        std::process::exit(1);
    }

    retval = IDAQuadSStolerancesB(&ida_mem, indexB, reltolB, abstolQB);
    if check_retval(Some(retval), "IDAQuadSStolerancesB", 1) != 0 {
        std::process::exit(1);
    }

    /* Include quadratures in error control. */
    retval = IDASetQuadErrConB(&ida_mem, indexB, SUNTRUE);
    if check_retval(Some(retval), "IDASetQuadErrConB", 1) != 0 {
        std::process::exit(1);
    }

    /* Backward Integration */
    print!("Backward integration ... ");

    retval = IDASolveB(&ida_mem, T0, IDA_NORMAL);
    if check_retval(Some(retval), "IDASolveB", 1) != 0 {
        std::process::exit(1);
    }

    let idaB = IDAGetAdjIDABmem(&ida_mem, indexB).expect("IDAGetAdjIDABmem");
    let mut nstB: i64 = 0;
    let _ = IDAGetNumSteps(&idaB, &mut nstB);
    print!("done ( nst = {} )\n", nstB);

    retval = IDAGetB(&ida_mem, indexB, &mut time, &yB, &ypB);
    if check_retval(Some(retval), "IDAGetB", 1) != 0 {
        std::process::exit(1);
    }

    retval = IDAGetQuadB(&ida_mem, indexB, &mut time, &qB);
    if check_retval(Some(retval), "IDAGetB", 1) != 0 {
        std::process::exit(1);
    }

    PrintOutput(TB2, &yB, &ypB, &qB);

    /* Reinitialize backward phase and start from a different time (TB1). */
    print!("Re-initialize IDAS memory for backward run\n");

    /* Both algebraic part from y and the entire y' are computed by IDACalcIC. */
    Ith_set(&yB, 1, ZERO);
    Ith_set(&yB, 2, ZERO);
    Ith_set(&yB, 3, 0.50); /* not consistent */

    /* Rough guess for ypB. */
    Ith_set(&ypB, 1, 0.80);
    Ith_set(&ypB, 2, 0.75);
    Ith_set(&ypB, 3, ZERO);

    /* Initialize qB */
    Ith_set(&qB, 1, ZERO);
    Ith_set(&qB, 2, ZERO);
    Ith_set(&qB, 3, ZERO);

    retval = IDAReInitB(&ida_mem, indexB, TB1, &yB, &ypB);
    if check_retval(Some(retval), "IDAReInitB", 1) != 0 {
        std::process::exit(1);
    }

    /* Also reinitialize quadratures. */
    retval = IDAQuadReInitB(&ida_mem, indexB, &qB);
    if check_retval(Some(retval), "IDAQuadReInitB", 1) != 0 {
        std::process::exit(1);
    }

    /* Use IDACalcICB to compute consistent initial conditions
    for this backward problem. */

    let id = N_VClone(&yy).expect("N_VClone");
    Ith_set(&id, 1, 1.0);
    Ith_set(&id, 2, 1.0);
    Ith_set(&id, 3, 0.0);

    /* Specify which variables are differential (1) and which algebraic (0).*/
    retval = IDASetIdB(&ida_mem, indexB, Some(&id));
    if check_retval(Some(retval), "IDASetId", 1) != 0 {
        std::process::exit(1);
    }

    retval = IDACalcICB(&ida_mem, indexB, T1B, &yyTB1, &ypTB1);
    if check_retval(Some(retval), "IDACalcICB", 1) != 0 {
        std::process::exit(1);
    }

    /* Get the consistent IC found by IDAS. */
    retval = IDAGetConsistentICB(&ida_mem, indexB, Some(&yB), Some(&ypB));
    if check_retval(Some(retval), "IDAGetConsistentICB", 1) != 0 {
        std::process::exit(1);
    }

    print!("Backward integration ... ");

    retval = IDASolveB(&ida_mem, T0, IDA_NORMAL);
    if check_retval(Some(retval), "IDASolveB", 1) != 0 {
        std::process::exit(1);
    }

    let idaB = IDAGetAdjIDABmem(&ida_mem, indexB).expect("IDAGetAdjIDABmem");
    let mut nstB: i64 = 0;
    let _ = IDAGetNumSteps(&idaB, &mut nstB);
    print!("done ( nst = {} )\n", nstB);

    retval = IDAGetB(&ida_mem, indexB, &mut time, &yB, &ypB);
    if check_retval(Some(retval), "IDAGetB", 1) != 0 {
        std::process::exit(1);
    }

    retval = IDAGetQuadB(&ida_mem, indexB, &mut time, &qB);
    if check_retval(Some(retval), "IDAGetQuadB", 1) != 0 {
        std::process::exit(1);
    }

    PrintOutput(TB1, &yB, &ypB, &qB);

    /* Free any memory used.*/

    print!("Free memory\n\n");

    IDAFree(&mut ida_mem_opt);
    let _ = SUNLinSolFree(Some(LS));
    SUNMatDestroy(A);
    let _ = SUNLinSolFree(Some(LSB));
    SUNMatDestroy(AB);
    N_VDestroy(yy);
    N_VDestroy(yp);
    N_VDestroy(q);
    N_VDestroy(yB);
    N_VDestroy(ypB);
    N_VDestroy(qB);
    N_VDestroy(id);
    N_VDestroy(yyTB1);
    N_VDestroy(ypTB1);

    /* C: `if (ckpnt != NULL) free(ckpnt);` -- ckpnt stays NULL here.
    C: `free(data);` -- the boxed copies are owned by the integrator. */

    let _ = SUNContext_Free(&mut sunctx);

    std::process::exit(0);
}

/*
 *--------------------------------------------------------------------
 * FUNCTIONS CALLED BY IDAS
 *--------------------------------------------------------------------
 */

/*
 * f routine. Compute f(t,y).
 */

fn res(
    _t: sunrealtype,
    yy: &N_Vector,
    yp: &N_Vector,
    resval: &N_Vector,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    let y1 = Ith(yy, 1);
    let y2 = Ith(yy, 2);
    let y3 = Ith(yy, 3);
    let yp1 = Ith(yp, 1);
    let yp2 = Ith(yp, 2);

    let data = user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<UserData>())
        .expect("user_data is UserData");
    let p1 = data.p[0];
    let p2 = data.p[1];
    let p3 = data.p[2];

    let mut rval = N_VGetArrayPointer(resval).expect("N_VGetArrayPointer");

    rval[0] = p1 * y1 - p2 * y2 * y3;
    rval[1] = -rval[0] + p3 * y2 * y2 + yp2;
    rval[0] += yp1;
    rval[2] = y1 + y2 + y3 - 1.0;

    0
}

/*
 * Jacobian routine. Compute J(t,y).
 */

#[allow(clippy::too_many_arguments)]
fn Jac(
    _t: sunrealtype,
    cj: sunrealtype,
    yy: &N_Vector,
    _yp: &N_Vector,
    _resvec: &N_Vector,
    JJ: &SUNMatrix,
    user_data: &mut Option<Box<dyn Any>>,
    _tmp1: &N_Vector,
    _tmp2: &N_Vector,
    _tmp3: &N_Vector,
) -> i32 {
    let yval = N_VGetArrayPointer(yy).expect("N_VGetArrayPointer");
    let ud = user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<UserData>())
        .expect("user_data is UserData");
    let p1 = ud.p[0];
    let p2 = ud.p[1];
    let p3 = ud.p[2];

    SUNMatZero(JJ);

    /* One borrow for all three arrays: taking them separately would be a
    second mutable borrow of the same matrix content. */
    let mut m = SUNSparseMatrix_Content(JJ);
    let m = &mut *m;
    let (data, rowvals, colptrs) = (&mut m.data, &mut m.indexvals, &mut m.indexptrs);

    colptrs[0] = 0;
    colptrs[1] = 3;
    colptrs[2] = 6;
    colptrs[3] = 9;

    data[0] = p1 + cj;
    rowvals[0] = 0;
    data[1] = -p1;
    rowvals[1] = 1;
    data[2] = ONE;
    rowvals[2] = 2;

    data[3] = -p2 * yval[2];
    rowvals[3] = 0;
    data[4] = p2 * yval[2] + 2.0 * p3 * yval[1] + cj;
    rowvals[4] = 1;
    data[5] = ONE;
    rowvals[5] = 2;

    data[6] = -p2 * yval[1];
    rowvals[6] = 0;
    data[7] = p2 * yval[1];
    rowvals[7] = 1;
    data[8] = ONE;
    rowvals[8] = 2;

    0
}

/*
 * rhsQ routine. Compute fQ(t,y).
 */

fn rhsQ(
    _t: sunrealtype,
    yy: &N_Vector,
    _yp: &N_Vector,
    qdot: &N_Vector,
    _user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    Ith_set(qdot, 1, Ith(yy, 3));
    0
}

/*
 * EwtSet function. Computes the error weights at the current solution.
 */

fn ewt(y: &N_Vector, w: &N_Vector, _user_data: &mut Option<Box<dyn Any>>) -> i32 {
    let rtol: sunrealtype = RTOL;
    let mut atol: [sunrealtype; 3] = [0.0; 3];
    atol[0] = ATOL1;
    atol[1] = ATOL2;
    atol[2] = ATOL3;

    for i in 1..=3usize {
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
 * resB routine.
 */

fn resB(
    _tt: sunrealtype,
    yy: &N_Vector,
    _yp: &N_Vector,
    yyB: &N_Vector,
    ypB: &N_Vector,
    rrB: &N_Vector,
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

    /* The y  vector */
    let y2 = Ith(yy, 2);
    let y3 = Ith(yy, 3);

    /* The lambda vector */
    let l1 = Ith(yyB, 1);
    let l2 = Ith(yyB, 2);
    let l3 = Ith(yyB, 3);

    /* The lambda dot vector */
    let lp1 = Ith(ypB, 1);
    let lp2 = Ith(ypB, 2);

    /* Temporary variables */
    let l21 = l2 - l1;

    /* Load residual. */
    Ith_set(rrB, 1, lp1 + p1 * l21 - l3);
    Ith_set(rrB, 2, lp2 - p2 * y3 * l21 - 2.0 * p3 * y2 * l2 - l3);
    Ith_set(rrB, 3, -p2 * y2 * l21 - l3 + 1.0);

    0
}

/*Jacobian for backward problem. */
#[allow(clippy::too_many_arguments)]
fn JacB(
    _tt: sunrealtype,
    cj: sunrealtype,
    yy: &N_Vector,
    _yp: &N_Vector,
    _yyB: &N_Vector,
    _ypB: &N_Vector,
    _rrB: &N_Vector,
    JB: &SUNMatrix,
    user_data: &mut Option<Box<dyn Any>>,
    _tmp1B: &N_Vector,
    _tmp2B: &N_Vector,
    _tmp3B: &N_Vector,
) -> i32 {
    let yvalB = N_VGetArrayPointer(yy).expect("N_VGetArrayPointer");
    let ud = user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<UserData>())
        .expect("user_data is UserData");
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

    dataB[0] = -p1 + cj;
    rowvalsB[0] = 0;
    dataB[1] = p2 * yvalB[2];
    rowvalsB[1] = 1;
    dataB[2] = p2 * yvalB[1];
    rowvalsB[2] = 2;

    dataB[3] = p1;
    rowvalsB[3] = 0;
    dataB[4] = -(p2 * yvalB[2] + 2.0 * p3 * yvalB[1]) + cj;
    rowvalsB[4] = 1;
    dataB[5] = -p2 * yvalB[1];
    rowvalsB[5] = 2;

    dataB[6] = -ONE;
    rowvalsB[6] = 0;
    dataB[7] = -ONE;
    rowvalsB[7] = 1;
    dataB[8] = -ONE;
    rowvalsB[8] = 2;

    0
}

fn rhsQB(
    _tt: sunrealtype,
    yy: &N_Vector,
    _yp: &N_Vector,
    yyB: &N_Vector,
    _ypB: &N_Vector,
    rrQB: &N_Vector,
    _user_dataB: &mut Option<Box<dyn Any>>,
) -> i32 {
    /* The y vector */
    let y1 = Ith(yy, 1);
    let y2 = Ith(yy, 2);
    let y3 = Ith(yy, 3);

    /* The lambda vector */
    let l1 = Ith(yyB, 1);
    let l2 = Ith(yyB, 2);

    /* Temporary variables */
    let l21 = l2 - l1;

    Ith_set(rrQB, 1, y1 * l21);
    Ith_set(rrQB, 2, -y3 * y2 * l21);
    Ith_set(rrQB, 3, -y2 * y2 * l2);

    0
}

/*
 *--------------------------------------------------------------------
 * PRIVATE FUNCTIONS
 *--------------------------------------------------------------------
 */

/*
 * Print results after backward integration
 */

fn PrintOutput(tfinal: sunrealtype, yB: &N_Vector, _ypB: &N_Vector, qB: &N_Vector) {
    print!("--------------------------------------------------------\n");
    print!("tB0:        {}\n", fmt_ew(tfinal, 12, 4));
    print!(
        "dG/dp:      {} {} {}\n",
        fmt_ew(-Ith(qB, 1), 12, 4),
        fmt_ew(-Ith(qB, 2), 12, 4),
        fmt_ew(-Ith(qB, 3), 12, 4)
    );
    print!(
        "lambda(t0): {} {} {}\n",
        fmt_ew(Ith(yB, 1), 12, 4),
        fmt_ew(Ith(yB, 2), 12, 4),
        fmt_ew(Ith(yB, 3), 12, 4)
    );
    print!("--------------------------------------------------------\n\n");
}

/*
 * Check function return value.
 *    opt == 0 means SUNDIALS function allocates memory so check if
 *             returned NULL pointer
 *    opt == 1 means SUNDIALS function returns an integer value so check if
 *             retval < 0
 *    opt == 2 means function allocates memory so check if returned
 *             NULL pointer
 */

fn check_retval(returnvalue: Option<i32>, funcname: &str, opt: i32) -> i32 {
    /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
    if opt == 0 && returnvalue.is_none() {
        eprint!(
            "\nSUNDIALS_ERROR: {}() failed - returned NULL pointer\n\n",
            funcname
        );
        return 1;
    }
    /* Check if retval < 0 */
    else if opt == 1 {
        let retval = returnvalue.expect("retval");
        if retval < 0 {
            eprint!(
                "\nSUNDIALS_ERROR: {}() failed with retval = {}\n\n",
                funcname, retval
            );
            return 1;
        }
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
