//! Port of `examples/idas/serial/idasHessian_ASA_FSA.c`.
//!
//! Hessian using adjoint sensitivity example problem.
//!
//! This simple example problem for IDAS, due to Robertson,
//! is from chemical kinetics, and consists of the following three
//! equations:
//!
//!   y1' + p1 * y1 - p2 * y2 * y3             = 0
//!   y2' - p1 * y1 + p2 * y2 * y3 + p3 * y2^2 = 0
//!   y1 + y2 + y3 - 1                         = 0
//!
//!        [1]        [-p1]
//!   y(0)=[0]  y'(0)=[ p1]   p1 = 0.04   p2 = 1e4   p3 = 1e07
//!        [0]        [ 0 ]
//!
//!       80
//!      /
//!  G = | 0.5 * (y1^2 + y2^2 + y3^2) dt
//!      /
//!      0
//! Compute the gradient (using FSA and ASA) and Hessian (FSA over ASA)
//! of G with respect to parameters p1 and p2.
//!
//! Reference: D.B. Ozyurt and P.I. Barton, SISC 26(5) 1725-1743, 2005.
//!
//! Error handling was suppressed for code readability reasons.

#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

use idas_rs::prelude::*;

/* Accessor macros
   (C macro `Ith(v,i)` = `NV_Ith_S(v,i-1)`; i is 1-based). */

fn Ith(v: &N_Vector, i: usize) -> sunrealtype {
    N_VGetArrayPointer(v).expect("N_VGetArrayPointer")[i - 1]
}

fn Ith_set(v: &N_Vector, i: usize, x: sunrealtype) {
    N_VGetArrayPointer(v).expect("N_VGetArrayPointer")[i - 1] = x;
}

/* Problem Constants */
const NEQ: sunindextype = 3; /* number of equations                  */
const NP: i32 = 2; /* number of sensitivities              */

const T0: sunrealtype = 0.0; /* Initial time. */
const TF: sunrealtype = 80.0; /* Final time. */

/* Tolerances */
const RTOL: sunrealtype = 1e-08; /* scalar relative tolerance            */
const ATOL: sunrealtype = 1e-10; /* vector absolute tolerance components */
const RTOLA: sunrealtype = 1e-08; /* for adjoint integration              */
const ATOLA: sunrealtype = 1e-08; /* for adjoint integration              */

/* Parameters */
const P1: sunrealtype = 0.04;
const P2: sunrealtype = 1.0e4;
const P3: sunrealtype = 3.0e7;

/* Predefined consts */
const HALF: sunrealtype = 0.5;
const ZERO: sunrealtype = 0.0;
const ONE: sunrealtype = 1.0;
const TWO: sunrealtype = 2.0;

/* User defined struct
   (C `UserData` is a *pointer* to this struct, shared by the forward
   problem, both backward problems and the finite-difference runs, and
   mutated in place between integrations; `Rc<RefCell<..>>` reproduces
   that aliasing exactly.) */
struct UserData {
    p: [sunrealtype; 3],
}

type UserDataRc = Rc<RefCell<UserData>>;

fn data_of(user_data: &mut Option<Box<dyn Any>>) -> UserDataRc {
    user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<UserDataRc>())
        .expect("user_data is UserDataRc")
        .clone()
}

fn main() {
    /* Create the SUNDIALS context object for this simulation. */
    let mut sunctx: Option<SUNContext> = None;
    let _ = SUNContext_Create(SUN_COMM_NULL, &mut sunctx);
    let ctx = sunctx.as_ref().expect("sunctx").clone();

    /* Print problem description */
    print!("\nAdjoint Sensitivity Example for Chemical Kinetics\n");
    print!("---------------------------------------------------------\n");
    print!("DAE: dy1/dt + p1*y1 - p2*y2*y3 = 0\n");
    print!("     dy2/dt - p1*y1 + p2*y2*y3 + p3*(y2)^2 = 0\n");
    print!("               y1  +  y2  +  y3 = 0\n\n");
    print!("Find dG/dp and d^2G/dp^2, where p=[p1,p2] for\n");
    print!("     G = int_t0^tB0 g(t,p,y) dt\n");
    print!("     g(t,p,y) = y3\n\n\n");

    /* Allocate and initialize user data. */
    let data: UserDataRc = Rc::new(RefCell::new(UserData { p: [ZERO; 3] }));
    data.borrow_mut().p[0] = P1;
    data.borrow_mut().p[1] = P2;
    data.borrow_mut().p[2] = P3;

    /* Consistent IC */
    let yy = N_VNew_Serial(NEQ, &ctx).expect("N_VNew_Serial");
    let yp = N_VClone(&yy).expect("N_VClone");
    Ith_set(&yy, 1, ONE);
    Ith_set(&yy, 2, ZERO);
    Ith_set(&yy, 3, ZERO);
    Ith_set(&yp, 1, -P1);
    Ith_set(&yp, 2, P1);
    Ith_set(&yp, 3, 0.0);

    let q = N_VNew_Serial(1, &ctx).expect("N_VNew_Serial");
    N_VConst(ZERO, &q);

    let yyS = N_VCloneVectorArray(NP, &yy).expect("N_VCloneVectorArray");
    let ypS = N_VCloneVectorArray(NP, &yp).expect("N_VCloneVectorArray");
    N_VConst(ZERO, &yyS[0]);
    N_VConst(ZERO, &yyS[1]);
    N_VConst(ZERO, &ypS[0]);
    N_VConst(ZERO, &ypS[1]);

    let qS = N_VCloneVectorArray(NP, &q).expect("N_VCloneVectorArray");
    for is in 0..NP as usize {
        N_VConst(ZERO, &qS[is]);
    }

    let mut ida_mem_opt = IDACreate(&ctx);
    let ida_mem = ida_mem_opt.as_ref().expect("ida_mem").clone();

    let mut ti = T0;
    let _ = IDAInit(&ida_mem, res, ti, &yy, &yp);

    /* Forward problem's setup. */
    let _ = IDASStolerances(&ida_mem, RTOL, ATOL);

    /* Create dense SUNMatrix for use in linear solves */
    let A = SUNDenseMatrix(NEQ, NEQ, &ctx);
    if check_retval_ptr(&A, "SUNDenseMatrix", 0) != 0 {
        std::process::exit(1);
    }
    let A = A.expect("A");

    /* Create dense SUNLinearSolver object */
    let LS = SUNLinSol_Dense(&yy, &A, &ctx);
    if check_retval_ptr(&LS, "SUNLinSol_Dense", 0) != 0 {
        std::process::exit(1);
    }
    let LS = LS.expect("LS");

    /* Attach the matrix and linear solver */
    let retval = IDASetLinearSolver(&ida_mem, &LS, Some(&A));
    if check_retval_int(retval, "IDASetLinearSolver", 1) != 0 {
        std::process::exit(1);
    }

    let _ = IDASetUserData(&ida_mem, Some(Box::new(data.clone())));
    let _ = IDASetMaxNumSteps(&ida_mem, 1500);

    /* Quadrature's setup. */
    let _ = IDAQuadInit(&ida_mem, rhsQ, &q);
    let _ = IDAQuadSStolerances(&ida_mem, RTOL, ATOL);
    let _ = IDASetQuadErrCon(&ida_mem, SUNTRUE);

    /* Sensitivity's setup. */
    let _ = IDASensInit(&ida_mem, NP, IDA_SIMULTANEOUS, Some(resS), &yyS, &ypS);
    let _ = IDASensEEtolerances(&ida_mem);
    let _ = IDASetSensErrCon(&ida_mem, SUNTRUE);

    /* Setup of quadrature's sensitivities */
    let _ = IDAQuadSensInit(&ida_mem, Some(rhsQS), &qS);
    let _ = IDAQuadSensEEtolerances(&ida_mem);
    let _ = IDASetQuadSensErrCon(&ida_mem, SUNTRUE);

    /* Initialize ASA. */
    let _ = IDAAdjInit(&ida_mem, 100, IDA_HERMITE);

    print!("---------------------------------------------------------\n");
    print!("Forward integration\n");
    print!("---------------------------------------------------------\n\n");

    let mut tf = TF;
    let mut time: sunrealtype = 0.0;
    let mut nckp: i32 = 0;
    let _ = IDASolveF(&ida_mem, tf, &mut time, &yy, &yp, IDA_NORMAL, &mut nckp);

    let _ = IDAGetQuad(&ida_mem, &mut time, &q);
    let G = Ith(&q, 1);
    /* C: printf("     G:    %12.4e\n", G) */
    print!("     G:    {}\n", fmt_ew(G, 12, 4));

    /* Sensitivities are needed for IC of backward problems. */
    let _ = IDAGetSensDky(&ida_mem, tf, 0, &yyS);
    let _ = IDAGetSensDky(&ida_mem, tf, 1, &ypS);

    let _ = IDAGetQuadSens(&ida_mem, &mut time, &qS);
    /* C: printf("   dG/dp:  %12.4e %12.4e\n", Ith(qS[0],1), Ith(qS[1],1)) */
    print!(
        "   dG/dp:  {} {}\n",
        fmt_ew(Ith(&qS[0], 1), 12, 4),
        fmt_ew(Ith(&qS[1], 1), 12, 4)
    );
    print!("\n");

    /******************************
     * BACKWARD PROBLEM #1
     *******************************/

    /* Consistent IC. */
    let yyB1 = N_VNew_Serial(2 * NEQ, &ctx).expect("N_VNew_Serial");
    let ypB1 = N_VClone(&yyB1).expect("N_VClone");

    N_VConst(ZERO, &yyB1);
    Ith_set(&yyB1, 3, Ith(&yy, 3));
    Ith_set(&yyB1, 6, Ith(&yyS[0], 3));

    N_VConst(ZERO, &ypB1);
    Ith_set(&ypB1, 1, Ith(&yy, 3) - Ith(&yy, 1));
    Ith_set(&ypB1, 2, Ith(&yy, 3) - Ith(&yy, 2));
    Ith_set(&ypB1, 4, Ith(&yyS[0], 3) - Ith(&yyS[0], 1));
    Ith_set(&ypB1, 5, Ith(&yyS[0], 3) - Ith(&yyS[0], 2));

    let qB1 = N_VNew_Serial(2 * NP as sunindextype, &ctx).expect("N_VNew_Serial");
    N_VConst(ZERO, &qB1);

    let mut indexB1: i32 = 0;
    let _ = IDACreateB(&ida_mem, &mut indexB1);
    let _ = IDAInitBS(&ida_mem, indexB1, resBS1, tf, &yyB1, &ypB1);
    let _ = IDASStolerancesB(&ida_mem, indexB1, RTOLA, ATOLA);
    let _ = IDASetUserDataB(&ida_mem, indexB1, Some(Box::new(data.clone())));
    let _ = IDASetMaxNumStepsB(&ida_mem, indexB1, 5000);

    /* Create dense SUNMatrix for use in linear solves */
    let AB1 = SUNDenseMatrix(2 * NEQ, 2 * NEQ, &ctx);
    if check_retval_ptr(&AB1, "SUNDenseMatrix", 0) != 0 {
        std::process::exit(1);
    }
    let AB1 = AB1.expect("AB1");

    /* Create dense SUNLinearSolver object */
    let LSB1 = SUNLinSol_Dense(&yyB1, &AB1, &ctx);
    if check_retval_ptr(&LSB1, "SUNLinSol_Dense", 0) != 0 {
        std::process::exit(1);
    }
    let LSB1 = LSB1.expect("LSB1");

    /* Attach the matrix and linear solver */
    let retval = IDASetLinearSolverB(&ida_mem, indexB1, &LSB1, Some(&AB1));
    if check_retval_int(retval, "IDASetLinearSolverB", 1) != 0 {
        std::process::exit(1);
    }

    let _ = IDAQuadInitBS(&ida_mem, indexB1, rhsQBS1, &qB1);

    /******************************
     * BACKWARD PROBLEM #2
     *******************************/

    /* Consistent IC. */
    let yyB2 = N_VNew_Serial(2 * NEQ, &ctx).expect("N_VNew_Serial");
    let ypB2 = N_VNew_Serial(2 * NEQ, &ctx).expect("N_VNew_Serial");

    N_VConst(ZERO, &yyB2);
    Ith_set(&yyB2, 3, Ith(&yy, 3));
    Ith_set(&yyB2, 6, Ith(&yyS[1], 3));

    N_VConst(ZERO, &ypB2);
    Ith_set(&ypB2, 1, Ith(&yy, 3) - Ith(&yy, 1));
    Ith_set(&ypB2, 2, Ith(&yy, 3) - Ith(&yy, 2));
    Ith_set(&ypB2, 4, Ith(&yyS[1], 3) - Ith(&yyS[1], 1));
    Ith_set(&ypB2, 5, Ith(&yyS[1], 3) - Ith(&yyS[1], 2));

    let qB2 = N_VNew_Serial(2 * NP as sunindextype, &ctx).expect("N_VNew_Serial");
    N_VConst(ZERO, &qB2);

    let mut indexB2: i32 = 0;
    let _ = IDACreateB(&ida_mem, &mut indexB2);
    let _ = IDAInitBS(&ida_mem, indexB2, resBS2, tf, &yyB2, &ypB2);
    let _ = IDASStolerancesB(&ida_mem, indexB2, RTOLA, ATOLA);
    let _ = IDASetUserDataB(&ida_mem, indexB2, Some(Box::new(data.clone())));
    let _ = IDASetMaxNumStepsB(&ida_mem, indexB2, 2500);

    /* Create dense SUNMatrix for use in linear solves */
    let AB2 = SUNDenseMatrix(2 * NEQ, 2 * NEQ, &ctx);
    if check_retval_ptr(&AB2, "SUNDenseMatrix", 0) != 0 {
        std::process::exit(1);
    }
    let AB2 = AB2.expect("AB2");

    /* Create dense SUNLinearSolver object */
    let LSB2 = SUNLinSol_Dense(&yyB2, &AB2, &ctx);
    if check_retval_ptr(&LSB2, "SUNLinSol_Dense", 0) != 0 {
        std::process::exit(1);
    }
    let LSB2 = LSB2.expect("LSB2");

    /* Attach the matrix and linear solver */
    let retval = IDASetLinearSolverB(&ida_mem, indexB2, &LSB2, Some(&AB2));
    if check_retval_int(retval, "IDASetLinearSolverB", 1) != 0 {
        std::process::exit(1);
    }

    let _ = IDAQuadInitBS(&ida_mem, indexB2, rhsQBS2, &qB2);

    /* Integrate backward problems. */
    print!("---------------------------------------------------------\n");
    print!("Backward integration \n");
    print!("---------------------------------------------------------\n\n");

    let _ = IDASolveB(&ida_mem, ti, IDA_NORMAL);

    let _ = IDAGetB(&ida_mem, indexB1, &mut time, &yyB1, &ypB1);
    /*
       retval = IDAGetNumSteps(IDAGetAdjIDABmem(ida_mem, indexB1), &nst);
       printf("at time=%g \tpb 1 Num steps:%d\n", time, nst);
       retval = IDAGetNumSteps(IDAGetAdjIDABmem(ida_mem, indexB2), &nst);
       printf("at time=%g \tpb 2 Num steps:%d\n\n", time, nst);
    */

    let _ = IDAGetQuadB(&ida_mem, indexB1, &mut time, &qB1);
    let _ = IDAGetQuadB(&ida_mem, indexB2, &mut time, &qB2);
    /* C: printf("   dG/dp:  %12.4e %12.4e   (from backward pb. 1)\n", ...) */
    print!(
        "   dG/dp:  {} {}   (from backward pb. 1)\n",
        fmt_ew(Ith(&qB1, 1), 12, 4),
        fmt_ew(Ith(&qB1, 2), 12, 4)
    );
    print!(
        "   dG/dp:  {} {}   (from backward pb. 2)\n",
        fmt_ew(Ith(&qB2, 1), 12, 4),
        fmt_ew(Ith(&qB2, 2), 12, 4)
    );

    print!("\n");
    print!("   H = d2G/dp2:\n");
    print!("        (1)            (2)\n");
    /* C: printf("  %12.4e  %12.4e\n", ...) */
    print!(
        "  {}  {}\n",
        fmt_ew(Ith(&qB1, 3), 12, 4),
        fmt_ew(Ith(&qB2, 3), 12, 4)
    );
    print!(
        "  {}  {}\n",
        fmt_ew(Ith(&qB1, 4), 12, 4),
        fmt_ew(Ith(&qB2, 4), 12, 4)
    );

    IDAFree(&mut ida_mem_opt);
    drop(ida_mem);
    let _ = SUNLinSolFree(Some(LS));
    SUNMatDestroy(A);
    let _ = SUNLinSolFree(Some(LSB1));
    SUNMatDestroy(AB1);
    let _ = SUNLinSolFree(Some(LSB2));
    SUNMatDestroy(AB2);

    /*********************************
     * Use Finite Differences to verify
     **********************************/

    /* Perturbations are of different magnitudes as p1 and p2 are. */
    let dp1: sunrealtype = 1.0e-3;
    let dp2: sunrealtype = 2.5e+2;

    print!("\n");
    print!("---------------------------------------------------------\n");
    /* C: printf("Finite Differences ( dp1=%6.1e and dp2 = %6.1e )\n", dp1, dp2) */
    print!(
        "Finite Differences ( dp1={} and dp2 = {} )\n",
        fmt_ew(dp1, 6, 1),
        fmt_ew(dp2, 6, 1)
    );
    print!("---------------------------------------------------------\n\n");

    let mut ida_mem_opt = IDACreate(&ctx);
    let ida_mem = ida_mem_opt.as_ref().expect("ida_mem").clone();

    /********************
     * Forward FD for p1
     ********************/
    data.borrow_mut().p[0] += dp1;

    Ith_set(&yy, 1, ONE);
    Ith_set(&yy, 2, ZERO);
    Ith_set(&yy, 3, ZERO);
    Ith_set(&yp, 1, -data.borrow().p[0]);
    Ith_set(&yp, 2, -Ith(&yp, 1));
    Ith_set(&yp, 3, 0.0);
    N_VConst(ZERO, &q);
    ti = T0;
    tf = TF;

    let _ = IDAInit(&ida_mem, res, ti, &yy, &yp);

    let rtolFD: sunrealtype = 1.0e-12;
    let atolFD: sunrealtype = 1.0e-14;

    let _ = IDASStolerances(&ida_mem, rtolFD, atolFD);

    /* Create dense SUNMatrix for use in linear solves */
    let A = SUNDenseMatrix(NEQ, NEQ, &ctx);
    if check_retval_ptr(&A, "SUNDenseMatrix", 0) != 0 {
        std::process::exit(1);
    }
    let A = A.expect("A");

    /* Create dense SUNLinearSolver object */
    let LS = SUNLinSol_Dense(&yy, &A, &ctx);
    if check_retval_ptr(&LS, "SUNLinSol_Dense", 0) != 0 {
        std::process::exit(1);
    }
    let LS = LS.expect("LS");

    /* Attach the matrix and linear solver */
    let retval = IDASetLinearSolver(&ida_mem, &LS, Some(&A));
    if check_retval_int(retval, "IDASetLinearSolver", 1) != 0 {
        std::process::exit(1);
    }

    let _ = IDASetUserData(&ida_mem, Some(Box::new(data.clone())));
    let _ = IDASetMaxNumSteps(&ida_mem, 10000);

    let _ = IDAQuadInit(&ida_mem, rhsQ, &q);
    let _ = IDAQuadSStolerances(&ida_mem, rtolFD, atolFD);
    let _ = IDASetQuadErrCon(&ida_mem, SUNTRUE);

    let _ = IDASolve(&ida_mem, tf, &mut time, &yy, &yp, IDA_NORMAL);
    let _ = IDAGetQuad(&ida_mem, &mut time, &q);
    let mut Gp = Ith(&q, 1);

    /********************
     * Backward FD for p1
     ********************/
    data.borrow_mut().p[0] -= 2.0 * dp1;

    Ith_set(&yy, 1, ONE);
    Ith_set(&yy, 2, ZERO);
    Ith_set(&yy, 3, ZERO);
    Ith_set(&yp, 1, -data.borrow().p[0]);
    Ith_set(&yp, 2, -Ith(&yp, 1));
    Ith_set(&yp, 3, 0.0);
    N_VConst(ZERO, &q);

    let _ = IDAReInit(&ida_mem, ti, &yy, &yp);
    let _ = IDAQuadReInit(&ida_mem, &q);

    let _ = IDASolve(&ida_mem, tf, &mut time, &yy, &yp, IDA_NORMAL);
    let _ = IDAGetQuad(&ida_mem, &mut time, &q);
    let mut Gm = Ith(&q, 1);

    /* Compute FD for p1. */
    let mut grdG_fwd: [sunrealtype; 2] = [ZERO; 2];
    let mut grdG_bck: [sunrealtype; 2] = [ZERO; 2];
    let mut grdG_cntr: [sunrealtype; 2] = [ZERO; 2];
    grdG_fwd[0] = (Gp - G) / dp1;
    grdG_bck[0] = (G - Gm) / dp1;
    grdG_cntr[0] = (Gp - Gm) / (2.0 * dp1);
    let H11 = (Gp - 2.0 * G + Gm) / (dp1 * dp1);

    /********************
     * Forward FD for p2
     ********************/
    /*restore p1*/
    data.borrow_mut().p[0] += dp1;
    data.borrow_mut().p[1] += dp2;

    Ith_set(&yy, 1, ONE);
    Ith_set(&yy, 2, ZERO);
    Ith_set(&yy, 3, ZERO);
    Ith_set(&yp, 1, -data.borrow().p[0]);
    Ith_set(&yp, 2, -Ith(&yp, 1));
    Ith_set(&yp, 3, 0.0);
    N_VConst(ZERO, &q);

    let _ = IDAReInit(&ida_mem, ti, &yy, &yp);
    let _ = IDAQuadReInit(&ida_mem, &q);

    let _ = IDASolve(&ida_mem, tf, &mut time, &yy, &yp, IDA_NORMAL);
    let _ = IDAGetQuad(&ida_mem, &mut time, &q);
    Gp = Ith(&q, 1);

    /********************
     * Backward FD for p2
     ********************/
    data.borrow_mut().p[1] -= 2.0 * dp2;

    Ith_set(&yy, 1, ONE);
    Ith_set(&yy, 2, ZERO);
    Ith_set(&yy, 3, ZERO);
    Ith_set(&yp, 1, -data.borrow().p[0]);
    Ith_set(&yp, 2, -Ith(&yp, 1));
    Ith_set(&yp, 3, 0.0);
    N_VConst(ZERO, &q);

    let _ = IDAReInit(&ida_mem, ti, &yy, &yp);
    let _ = IDAQuadReInit(&ida_mem, &q);

    let _ = IDASolve(&ida_mem, tf, &mut time, &yy, &yp, IDA_NORMAL);
    let _ = IDAGetQuad(&ida_mem, &mut time, &q);
    Gm = Ith(&q, 1);

    /* Compute FD for p2. */
    grdG_fwd[1] = (Gp - G) / dp2;
    grdG_bck[1] = (G - Gm) / dp2;
    grdG_cntr[1] = (Gp - Gm) / (2.0 * dp2);
    let H22 = (Gp - 2.0 * G + Gm) / (dp2 * dp2);

    print!("\n");
    /* C: printf("   dG/dp:  %12.4e  %12.4e   (fwd FD)\n", ...) */
    print!(
        "   dG/dp:  {}  {}   (fwd FD)\n",
        fmt_ew(grdG_fwd[0], 12, 4),
        fmt_ew(grdG_fwd[1], 12, 4)
    );
    print!(
        "           {}  {}   (bck FD)\n",
        fmt_ew(grdG_bck[0], 12, 4),
        fmt_ew(grdG_bck[1], 12, 4)
    );
    print!(
        "           {}  {}   (cntr FD)\n",
        fmt_ew(grdG_cntr[0], 12, 4),
        fmt_ew(grdG_cntr[1], 12, 4)
    );
    print!("\n");
    print!("  H(1,1):  {}\n", fmt_ew(H11, 12, 4));
    print!("  H(2,2):  {}\n", fmt_ew(H22, 12, 4));

    IDAFree(&mut ida_mem_opt);
    drop(ida_mem);
    let _ = SUNLinSolFree(Some(LS));
    SUNMatDestroy(A);

    N_VDestroy(yyB1);
    N_VDestroy(ypB1);
    N_VDestroy(qB1);

    N_VDestroy(yyB2);
    N_VDestroy(ypB2);
    N_VDestroy(qB2);

    N_VDestroy(yy);
    N_VDestroy(yp);
    N_VDestroy(q);
    N_VDestroyVectorArray(yyS, NP);
    N_VDestroyVectorArray(ypS, NP);
    N_VDestroyVectorArray(qS, NP);

    /* free(data): the Rc drops with `data` */
    drop(data);

    let _ = SUNContext_Free(&mut sunctx);
    std::process::exit(0);
}

fn res(
    _tres: sunrealtype,
    yy: &N_Vector,
    yp: &N_Vector,
    rr: &N_Vector,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    let y1 = Ith(yy, 1);
    let y2 = Ith(yy, 2);
    let y3 = Ith(yy, 3);
    let yp1 = Ith(yp, 1);
    let yp2 = Ith(yp, 2);

    let data = data_of(user_data);
    let p1 = data.borrow().p[0];
    let p2 = data.borrow().p[1];
    let p3 = data.borrow().p[2];

    let mut rval = N_VGetArrayPointer(rr).expect("N_VGetArrayPointer");
    rval[0] = p1 * y1 - p2 * y2 * y3;
    let rval0 = rval[0];
    rval[1] = -rval0 + p3 * y2 * y2 + yp2;
    rval[0] += yp1;
    rval[2] = y1 + y2 + y3 - 1.0;

    0
}

fn resS(
    _Ns: i32,
    _t: sunrealtype,
    yy: &N_Vector,
    _yp: &N_Vector,
    _resval: &N_Vector,
    yyS: &[N_Vector],
    ypS: &[N_Vector],
    resvalS: &[N_Vector],
    user_data: &mut Option<Box<dyn Any>>,
    _tmp1: &N_Vector,
    _tmp2: &N_Vector,
    _tmp3: &N_Vector,
) -> i32 {
    let data = data_of(user_data);
    let p1 = data.borrow().p[0];
    let p2 = data.borrow().p[1];
    let p3 = data.borrow().p[2];

    let y1 = Ith(yy, 1);
    let y2 = Ith(yy, 2);
    let y3 = Ith(yy, 3);

    for is in 0..NP as usize {
        let s1 = Ith(&yyS[is], 1);
        let s2 = Ith(&yyS[is], 2);
        let s3 = Ith(&yyS[is], 3);

        let sd1 = Ith(&ypS[is], 1);
        let sd2 = Ith(&ypS[is], 2);

        let mut rs1 = sd1 + p1 * s1 - p2 * y3 * s2 - p2 * y2 * s3;
        let mut rs2 = sd2 - p1 * s1 + p2 * y3 * s2 + p2 * y2 * s3 + TWO * p3 * y2 * s2;
        let rs3 = s1 + s2 + s3;

        match is {
            0 => {
                rs1 += y1;
                rs2 -= y1;
            }
            1 => {
                rs1 -= y2 * y3;
                rs2 += y2 * y3;
            }
            _ => {}
        }

        Ith_set(&resvalS[is], 1, rs1);
        Ith_set(&resvalS[is], 2, rs2);
        Ith_set(&resvalS[is], 3, rs3);
    }

    0
}

fn rhsQ(
    _t: sunrealtype,
    yy: &N_Vector,
    _yp: &N_Vector,
    qdot: &N_Vector,
    _user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    let y1 = Ith(yy, 1);
    let y2 = Ith(yy, 2);
    let y3 = Ith(yy, 3);
    Ith_set(qdot, 1, HALF * (y1 * y1 + y2 * y2 + y3 * y3));

    0
}

fn rhsQS(
    _Ns: i32,
    _t: sunrealtype,
    yy: &N_Vector,
    _yp: &N_Vector,
    yyS: &[N_Vector],
    _ypS: &[N_Vector],
    _rrQ: &N_Vector,
    rhsvalQS: &[N_Vector],
    _user_data: &mut Option<Box<dyn Any>>,
    _yytmp: &N_Vector,
    _yptmp: &N_Vector,
    _tmpQS: &N_Vector,
) -> i32 {
    let y1 = Ith(yy, 1);
    let y2 = Ith(yy, 2);
    let y3 = Ith(yy, 3);

    /* 1st sensitivity RHS */
    let s1 = Ith(&yyS[0], 1);
    let s2 = Ith(&yyS[0], 2);
    let s3 = Ith(&yyS[0], 3);
    Ith_set(&rhsvalQS[0], 1, y1 * s1 + y2 * s2 + y3 * s3);

    /* 2nd sensitivity RHS */
    let s1 = Ith(&yyS[1], 1);
    let s2 = Ith(&yyS[1], 2);
    let s3 = Ith(&yyS[1], 3);
    Ith_set(&rhsvalQS[1], 1, y1 * s1 + y2 * s2 + y3 * s3);

    0
}

/* Residuals for adjoint model. */
fn resBS1(
    _tt: sunrealtype,
    yy: &N_Vector,
    _yp: &N_Vector,
    yyS: &[N_Vector],
    _ypS: &[N_Vector],
    yyB: &N_Vector,
    ypB: &N_Vector,
    rrBS: &N_Vector,
    user_dataB: &mut Option<Box<dyn Any>>,
) -> i32 {
    let data = data_of(user_dataB);

    /* The parameters. */
    let p1 = data.borrow().p[0];
    let p2 = data.borrow().p[1];
    let p3 = data.borrow().p[2];

    /* The y vector. */
    let y1 = Ith(yy, 1);
    let y2 = Ith(yy, 2);
    let y3 = Ith(yy, 3);

    /* The lambda vector. */
    let l1 = Ith(yyB, 1);
    let l2 = Ith(yyB, 2);
    let l3 = Ith(yyB, 3);
    /* The mu vector. */
    let m1 = Ith(yyB, 4);
    let m2 = Ith(yyB, 5);
    let m3 = Ith(yyB, 6);

    /* The lambda dot vector. */
    let lp1 = Ith(ypB, 1);
    let lp2 = Ith(ypB, 2);
    /* The mu dot vector. */
    let mp1 = Ith(ypB, 4);
    let mp2 = Ith(ypB, 5);

    /* The sensitivity with respect to p1 */
    let s1 = Ith(&yyS[0], 1);
    let s2 = Ith(&yyS[0], 2);
    let s3 = Ith(&yyS[0], 3);

    /* Temporary variables */
    let l21 = l2 - l1;

    Ith_set(rrBS, 1, lp1 + p1 * l21 - l3 + y1);
    Ith_set(rrBS, 2, lp2 - p2 * y3 * l21 - TWO * p3 * y2 * l2 - l3 + y2);
    Ith_set(rrBS, 3, -p2 * y2 * l21 - l3 + y3);

    Ith_set(rrBS, 4, mp1 + p1 * (-m1 + m2) - m3 + l21 + s1);
    Ith_set(
        rrBS,
        5,
        mp2 + p2 * y3 * m1 - (p2 * y3 + TWO * p3 * y2) * m2 - m3 + p2 * s3 * l1
            - (TWO * p3 * s2 + p2 * s3) * l2
            + s2,
    );
    Ith_set(rrBS, 6, p2 * y2 * (m1 - m2) - m3 - p2 * s2 * l21 + s3);

    0
}

fn rhsQBS1(
    _tt: sunrealtype,
    yy: &N_Vector,
    _yp: &N_Vector,
    yyS: &[N_Vector],
    _ypS: &[N_Vector],
    yyB: &N_Vector,
    _ypB: &N_Vector,
    rhsBQS: &N_Vector,
    _user_dataB: &mut Option<Box<dyn Any>>,
) -> i32 {
    /* The y vector */
    let y1 = Ith(yy, 1);
    let y2 = Ith(yy, 2);
    let y3 = Ith(yy, 3);

    /* The lambda vector. */
    let l1 = Ith(yyB, 1);
    let l2 = Ith(yyB, 2);

    /* The mu vector. */
    let m1 = Ith(yyB, 4);
    let m2 = Ith(yyB, 5);

    /* The sensitivity with respect to p1 */
    let s1 = Ith(&yyS[0], 1);
    let s2 = Ith(&yyS[0], 2);
    let s3 = Ith(&yyS[0], 3);

    /* Temporary variables */
    let l21 = l2 - l1;

    Ith_set(rhsBQS, 1, -y1 * l21);
    Ith_set(rhsBQS, 2, y2 * y3 * l21);

    Ith_set(rhsBQS, 3, y1 * (m1 - m2) - s1 * l21);
    Ith_set(rhsBQS, 4, y2 * y3 * (m2 - m1) + (y3 * s2 + y2 * s3) * l21);

    0
}

fn resBS2(
    _tt: sunrealtype,
    yy: &N_Vector,
    _yp: &N_Vector,
    yyS: &[N_Vector],
    _ypS: &[N_Vector],
    yyB: &N_Vector,
    ypB: &N_Vector,
    rrBS: &N_Vector,
    user_dataB: &mut Option<Box<dyn Any>>,
) -> i32 {
    let data = data_of(user_dataB);

    /* The parameters. */
    let p1 = data.borrow().p[0];
    let p2 = data.borrow().p[1];
    let p3 = data.borrow().p[2];

    /* The y vector. */
    let y1 = Ith(yy, 1);
    let y2 = Ith(yy, 2);
    let y3 = Ith(yy, 3);

    /* The lambda vector. */
    let l1 = Ith(yyB, 1);
    let l2 = Ith(yyB, 2);
    let l3 = Ith(yyB, 3);
    /* The mu vector. */
    let m1 = Ith(yyB, 4);
    let m2 = Ith(yyB, 5);
    let m3 = Ith(yyB, 6);

    /* The lambda dot vector. */
    let lp1 = Ith(ypB, 1);
    let lp2 = Ith(ypB, 2);

    /* The mu dot vector. */
    let mp1 = Ith(ypB, 4);
    let mp2 = Ith(ypB, 5);

    /* The sensitivity with respect to p2 */
    let s1 = Ith(&yyS[1], 1);
    let s2 = Ith(&yyS[1], 2);
    let s3 = Ith(&yyS[1], 3);

    /* Temporary variables */
    let l21 = l2 - l1;

    Ith_set(rrBS, 1, lp1 + p1 * l21 - l3 + y1);
    Ith_set(rrBS, 2, lp2 - p2 * y3 * l21 - TWO * p3 * y2 * l2 - l3 + y2);
    Ith_set(rrBS, 3, -p2 * y2 * l21 - l3 + y3);

    Ith_set(rrBS, 4, mp1 + p1 * (-m1 + m2) - m3 + s1);
    Ith_set(
        rrBS,
        5,
        mp2 + p2 * y3 * m1 - (p2 * y3 + TWO * p3 * y2) * m2 - m3 + (y3 + p2 * s3) * l1
            - (y3 + TWO * p3 * s2 + p2 * s3) * l2
            + s2,
    );
    Ith_set(
        rrBS,
        6,
        p2 * y2 * (m1 - m2) - m3 - (y2 + p2 * s2) * l21 + s3,
    );

    0
}

fn rhsQBS2(
    _tt: sunrealtype,
    yy: &N_Vector,
    _yp: &N_Vector,
    yyS: &[N_Vector],
    _ypS: &[N_Vector],
    yyB: &N_Vector,
    _ypB: &N_Vector,
    rhsBQS: &N_Vector,
    _user_dataB: &mut Option<Box<dyn Any>>,
) -> i32 {
    /* The y vector */
    let y1 = Ith(yy, 1);
    let y2 = Ith(yy, 2);
    let y3 = Ith(yy, 3);

    /* The lambda vector. */
    let l1 = Ith(yyB, 1);
    let l2 = Ith(yyB, 2);

    /* The mu vector. */
    let m1 = Ith(yyB, 4);
    let m2 = Ith(yyB, 5);

    /* The sensitivity with respect to p2 */
    let s1 = Ith(&yyS[1], 1);
    let s2 = Ith(&yyS[1], 2);
    let s3 = Ith(&yyS[1], 3);

    /* Temporary variables */
    let l21 = l2 - l1;

    Ith_set(rhsBQS, 1, -y1 * l21);
    Ith_set(rhsBQS, 2, y2 * y3 * l21);

    Ith_set(rhsBQS, 3, y1 * (m1 - m2) - s1 * l21);
    Ith_set(rhsBQS, 4, y2 * y3 * (m2 - m1) + (y3 * s2 + y2 * s3) * l21);

    0
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

fn check_retval_int(retval: i32, funcname: &str, opt: i32) -> i32 {
    /* Check if retval < 0 */
    if opt == 1 && retval < 0 {
        eprint!(
            "\nSUNDIALS_ERROR: {}() failed with retval = {}\n\n",
            funcname, retval
        );
        return 1;
    }

    0
}
