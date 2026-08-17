#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

/* -----------------------------------------------------------------
 * Rust port of examples/cvodes/serial/cvsHessian_ASA_FSA.c
 * Programmer(s): Radu Serban @ LLNL
 * -----------------------------------------------------------------
 *
 * Hessian through adjoint sensitivity example problem.
 *
 *        [ - p1 * y1^2 - y3 ]           [ 1 ]
 *   y' = [    - y2          ]    y(0) = [ 1 ]
 *        [ -p2^2 * y2 * y3  ]           [ 1 ]
 *
 *   p1 = 1.0
 *   p2 = 2.0
 *
 *           2
 *          /
 *   G(p) = |  0.5 * ( y1^2 + y2^2 + y3^2 ) dt
 *          /
 *          0
 *
 * Compute the gradient (ASA) and Hessian (FSA over ASA) of G(p).
 *
 * See D.B. Ozyurt and P.I. Barton, SISC 26(5) 1725-1743, 2005.
 *
 * -----------------------------------------------------------------*/

use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

use cvodes_rs::prelude::*;

/* Accessor macro (C: `Ith(v,i)` = `NV_Ith_S(v,i-1)`; i is 1-based) */

fn Ith(v: &N_Vector, i: usize) -> sunrealtype {
    N_VGetArrayPointer(v).expect("N_VGetArrayPointer")[i - 1]
}

fn Ith_set(v: &N_Vector, i: usize, x: sunrealtype) {
    N_VGetArrayPointer(v).expect("N_VGetArrayPointer")[i - 1] = x;
}

const ZERO: sunrealtype = 0.0;
const ONE: sunrealtype = 1.0;

/* User data structure.
 *
 * C hands one and the same `UserData` pointer to the forward
 * integrator, to both backward problems, and keeps mutating `p1`/`p2`
 * through it during the finite-difference tests. The port shares the
 * single record through `Rc<RefCell<UserDataRec>>` so every
 * `Option<Box<dyn Any>>` token and `main` itself reach the same fields. */

struct UserDataRec {
    p1: sunrealtype,
    p2: sunrealtype,
}

type UserData = Rc<RefCell<UserDataRec>>;

fn data_of(user_data: &mut Option<Box<dyn Any>>) -> UserData {
    user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<UserData>())
        .expect("user_data is UserData")
        .clone()
}

/*
 *--------------------------------------------------------------------
 * MAIN PROGRAM
 *--------------------------------------------------------------------
 */

fn main() {
    /* User data structure */

    let data: UserData = Rc::new(RefCell::new(UserDataRec { p1: ZERO, p2: ZERO }));
    data.borrow_mut().p1 = 1.0;
    data.borrow_mut().p2 = 2.0;

    /* Problem size, integration interval, and tolerances */

    let Neq: sunindextype = 3;
    let Np: i32 = 2;
    let Np2: sunindextype = 2 * Np as sunindextype;

    let t0: sunrealtype = 0.0;
    let tf: sunrealtype = 2.0;

    let reltol: sunrealtype = 1.0e-8;

    let abstol: sunrealtype = 1.0e-8;
    let abstolQ: sunrealtype = 1.0e-8;

    let abstolB: sunrealtype = 1.0e-8;
    let abstolQB: sunrealtype = 1.0e-8;

    let mut time: sunrealtype = ZERO;
    let mut ncheck: i32 = 0;
    let mut indexB1: i32 = 0;
    let mut indexB2: i32 = 0;

    let mut grdG_fwd: [sunrealtype; 2] = [ZERO; 2];
    let mut grdG_bck: [sunrealtype; 2] = [ZERO; 2];
    let mut grdG_cntr: [sunrealtype; 2] = [ZERO; 2];

    /* Create the SUNDIALS simulation context that all SUNDIALS objects require */
    let mut sunctx_opt: Option<SUNContext> = None;
    let retval = SUNContext_Create(SUN_COMM_NULL, &mut sunctx_opt);
    if check_retval(retval, "SUNContext_Create", 1) != 0 {
        std::process::exit(1);
    }
    let sunctx = sunctx_opt.as_ref().expect("SUNContext").clone();

    /* Initializations for forward problem */

    let y = N_VNew_Serial(Neq, &sunctx);
    if check_retval_ptr(&y, "N_VNew_Serial", 0) != 0 {
        std::process::exit(1);
    }
    let y = y.expect("N_VNew_Serial");
    N_VConst(ONE, &y);

    let yQ = N_VNew_Serial(1, &sunctx);
    if check_retval_ptr(&yQ, "N_VNew_Serial", 0) != 0 {
        std::process::exit(1);
    }
    let yQ = yQ.expect("N_VNew_Serial");
    N_VConst(ZERO, &yQ);

    let yS = N_VCloneVectorArray(Np, &y);
    if check_retval_ptr(&yS, "N_VCloneVectorArray", 0) != 0 {
        std::process::exit(1);
    }
    let yS = yS.expect("N_VCloneVectorArray");
    N_VConst(ZERO, &yS[0]);
    N_VConst(ZERO, &yS[1]);

    let yQS = N_VCloneVectorArray(Np, &yQ);
    if check_retval_ptr(&yQS, "N_VCloneVectorArray", 0) != 0 {
        std::process::exit(1);
    }
    let yQS = yQS.expect("N_VCloneVectorArray");
    N_VConst(ZERO, &yQS[0]);
    N_VConst(ZERO, &yQS[1]);

    /* Create and initialize forward problem */

    let cvode_mem = CVodeCreate(CV_BDF, &sunctx);
    if check_retval_ptr(&cvode_mem, "CVodeCreate", 0) != 0 {
        std::process::exit(1);
    }
    let cvode_mem = cvode_mem.expect("CVodeCreate");

    let retval = CVodeInit(&cvode_mem, f, t0, &y);
    if check_retval(retval, "CVodeInit", 1) != 0 {
        std::process::exit(1);
    }

    let retval = CVodeSStolerances(&cvode_mem, reltol, abstol);
    if check_retval(retval, "CVodeSStolerances", 1) != 0 {
        std::process::exit(1);
    }

    let retval = CVodeSetUserData(&cvode_mem, Some(Box::new(data.clone())));
    if check_retval(retval, "CVodeSetUserData", 1) != 0 {
        std::process::exit(1);
    }

    /* Create a dense SUNMatrix */
    let A = SUNDenseMatrix(Neq, Neq, &sunctx);
    if check_retval_ptr(&A, "SUNDenseMatrix", 0) != 0 {
        std::process::exit(1);
    }
    let A = A.expect("SUNDenseMatrix");

    /* Create banded SUNLinearSolver for the forward problem */
    let LS = SUNLinSol_Dense(&y, &A, &sunctx);
    if check_retval_ptr(&LS, "SUNLinSol_Dense", 0) != 0 {
        std::process::exit(1);
    }
    let LS = LS.expect("SUNLinSol_Dense");

    /* Attach the matrix and linear solver */
    let retval = CVodeSetLinearSolver(&cvode_mem, &LS, Some(&A));
    if check_retval(retval, "CVodeSetLinearSolver", 1) != 0 {
        std::process::exit(1);
    }

    let retval = CVodeQuadInit(&cvode_mem, fQ, &yQ);
    if check_retval(retval, "CVodeQuadInit", 1) != 0 {
        std::process::exit(1);
    }

    let retval = CVodeQuadSStolerances(&cvode_mem, reltol, abstolQ);
    if check_retval(retval, "CVodeQuadSStolerances", 1) != 0 {
        std::process::exit(1);
    }

    let retval = CVodeSetQuadErrCon(&cvode_mem, SUNTRUE);
    if check_retval(retval, "CVodeSetQuadErrCon", 1) != 0 {
        std::process::exit(1);
    }

    let retval = CVodeSensInit(&cvode_mem, Np, CV_SIMULTANEOUS, Some(fS), &yS);
    if check_retval(retval, "CVodeSensInit", 1) != 0 {
        std::process::exit(1);
    }

    let retval = CVodeSensEEtolerances(&cvode_mem);
    if check_retval(retval, "CVodeSensEEtolerances", 1) != 0 {
        std::process::exit(1);
    }

    let retval = CVodeSetSensErrCon(&cvode_mem, SUNTRUE);
    if check_retval(retval, "CVodeSetSensErrCon", 1) != 0 {
        std::process::exit(1);
    }

    let retval = CVodeQuadSensInit(&cvode_mem, Some(fQS), &yQS);
    if check_retval(retval, "CVodeQuadSensInit", 1) != 0 {
        std::process::exit(1);
    }

    let retval = CVodeQuadSensEEtolerances(&cvode_mem);
    if check_retval(retval, "CVodeQuadSensEEtolerances", 1) != 0 {
        std::process::exit(1);
    }

    let retval = CVodeSetQuadSensErrCon(&cvode_mem, SUNTRUE);
    if check_retval(retval, "CVodeSetQuadSensErrCon", 1) != 0 {
        std::process::exit(1);
    }

    /* Initialize ASA */

    let steps: i64 = 100;
    let retval = CVodeAdjInit(&cvode_mem, steps, CV_POLYNOMIAL);
    if check_retval(retval, "CVodeAdjInit", 1) != 0 {
        std::process::exit(1);
    }

    /* Forward integration */

    print!("-------------------\n");
    print!("Forward integration\n");
    print!("-------------------\n\n");

    let retval = CVodeF(&cvode_mem, tf, &y, &mut time, CV_NORMAL, &mut ncheck);
    if check_retval(retval, "CVodeF", 1) != 0 {
        std::process::exit(1);
    }

    let retval = CVodeGetQuad(&cvode_mem, &mut time, &yQ);
    if check_retval(retval, "CVodeGetQuad", 1) != 0 {
        std::process::exit(1);
    }

    let G = Ith(&yQ, 1);

    let retval = CVodeGetSens(&cvode_mem, &mut time, &yS);
    if check_retval(retval, "CVodeGetSens", 1) != 0 {
        std::process::exit(1);
    }

    let retval = CVodeGetQuadSens(&cvode_mem, &mut time, &yQS);
    if check_retval(retval, "CVodeGetQuadSens", 1) != 0 {
        std::process::exit(1);
    }

    print!("ncheck = {}\n", ncheck);
    print!("\n");
    print!(
        "     y:    {} {} {}",
        fmt_ew(Ith(&y, 1), 12, 4),
        fmt_ew(Ith(&y, 2), 12, 4),
        fmt_ew(Ith(&y, 3), 12, 4)
    );
    print!("     G:    {}\n", fmt_ew(Ith(&yQ, 1), 12, 4));
    print!("\n");
    print!(
        "     yS1:  {} {} {}\n",
        fmt_ew(Ith(&yS[0], 1), 12, 4),
        fmt_ew(Ith(&yS[0], 2), 12, 4),
        fmt_ew(Ith(&yS[0], 3), 12, 4)
    );
    print!(
        "     yS2:  {} {} {}\n",
        fmt_ew(Ith(&yS[1], 1), 12, 4),
        fmt_ew(Ith(&yS[1], 2), 12, 4),
        fmt_ew(Ith(&yS[1], 3), 12, 4)
    );
    print!("\n");
    print!(
        "   dG/dp:  {} {}\n",
        fmt_ew(Ith(&yQS[0], 1), 12, 4),
        fmt_ew(Ith(&yQS[1], 1), 12, 4)
    );
    print!("\n");

    print!("Final Statistics for forward pb.\n");
    print!("--------------------------------\n");
    let retval = PrintFwdStats(&cvode_mem);
    if check_retval(retval, "PrintFwdStats", 1) != 0 {
        std::process::exit(1);
    }

    /* Initializations for backward problems */

    let yB1 = N_VNew_Serial(2 * Neq, &sunctx);
    if check_retval_ptr(&yB1, "N_VNew_Serial", 0) != 0 {
        std::process::exit(1);
    }
    let yB1 = yB1.expect("N_VNew_Serial");
    N_VConst(ZERO, &yB1);

    let yQB1 = N_VNew_Serial(Np2, &sunctx);
    if check_retval_ptr(&yQB1, "N_VNew_Serial", 0) != 0 {
        std::process::exit(1);
    }
    let yQB1 = yQB1.expect("N_VNew_Serial");
    N_VConst(ZERO, &yQB1);

    let yB2 = N_VNew_Serial(2 * Neq, &sunctx);
    if check_retval_ptr(&yB2, "N_VNew_Serial", 0) != 0 {
        std::process::exit(1);
    }
    let yB2 = yB2.expect("N_VNew_Serial");
    N_VConst(ZERO, &yB2);

    let yQB2 = N_VNew_Serial(Np2, &sunctx);
    if check_retval_ptr(&yQB2, "N_VNew_Serial", 0) != 0 {
        std::process::exit(1);
    }
    let yQB2 = yQB2.expect("N_VNew_Serial");
    N_VConst(ZERO, &yQB2);

    /* Create and initialize backward problems (one for each column of the Hessian) */

    /* -------------------------
    First backward problem
    -------------------------*/

    let retval = CVodeCreateB(&cvode_mem, CV_BDF, &mut indexB1);
    if check_retval(retval, "CVodeCreateB", 1) != 0 {
        std::process::exit(1);
    }

    let retval = CVodeInitBS(&cvode_mem, indexB1, fB1, tf, &yB1);
    if check_retval(retval, "CVodeInitBS", 1) != 0 {
        std::process::exit(1);
    }

    let retval = CVodeSStolerancesB(&cvode_mem, indexB1, reltol, abstolB);
    if check_retval(retval, "CVodeSStolerancesB", 1) != 0 {
        std::process::exit(1);
    }

    let retval = CVodeSetUserDataB(&cvode_mem, indexB1, Some(Box::new(data.clone())));
    if check_retval(retval, "CVodeSetUserDataB", 1) != 0 {
        std::process::exit(1);
    }

    let retval = CVodeQuadInitBS(&cvode_mem, indexB1, fQB1, &yQB1);
    if check_retval(retval, "CVodeQuadInitBS", 1) != 0 {
        std::process::exit(1);
    }

    let retval = CVodeQuadSStolerancesB(&cvode_mem, indexB1, reltol, abstolQB);
    if check_retval(retval, "CVodeQuadSStolerancesB", 1) != 0 {
        std::process::exit(1);
    }

    let retval = CVodeSetQuadErrConB(&cvode_mem, indexB1, SUNTRUE);
    if check_retval(retval, "CVodeSetQuadErrConB", 1) != 0 {
        std::process::exit(1);
    }

    /* Create a dense SUNMatrix */
    let AB1 = SUNDenseMatrix(2 * Neq, 2 * Neq, &sunctx);
    /* NOTE: upstream C checks `A` here (copy/paste from the forward
     * problem), not `AB1`; `A` is already known non-NULL so the check
     * never fires. Reproduced verbatim with an always-Some argument. */
    if check_retval_ptr(&Some(&A), "SUNDenseMatrix", 0) != 0 {
        std::process::exit(1);
    }
    let AB1 = AB1.expect("SUNDenseMatrix");

    /* Create dense SUNLinearSolver for the forward problem */
    let LSB1 = SUNLinSol_Dense(&yB1, &AB1, &sunctx);
    if check_retval_ptr(&LSB1, "SUNLinSol_Dense", 0) != 0 {
        std::process::exit(1);
    }
    let LSB1 = LSB1.expect("SUNLinSol_Dense");

    /* Attach the matrix and linear solver */
    let retval = CVodeSetLinearSolverB(&cvode_mem, indexB1, &LSB1, Some(&AB1));
    if check_retval(retval, "CVodeSetLinearSolverB", 1) != 0 {
        std::process::exit(1);
    }

    /* -------------------------
    Second backward problem
    -------------------------*/

    let retval = CVodeCreateB(&cvode_mem, CV_BDF, &mut indexB2);
    if check_retval(retval, "CVodeCreateB", 1) != 0 {
        std::process::exit(1);
    }

    let retval = CVodeInitBS(&cvode_mem, indexB2, fB2, tf, &yB2);
    if check_retval(retval, "CVodeInitBS", 1) != 0 {
        std::process::exit(1);
    }

    let retval = CVodeSStolerancesB(&cvode_mem, indexB2, reltol, abstolB);
    if check_retval(retval, "CVodeSStolerancesB", 1) != 0 {
        std::process::exit(1);
    }

    let retval = CVodeSetUserDataB(&cvode_mem, indexB2, Some(Box::new(data.clone())));
    if check_retval(retval, "CVodeSetUserDataB", 1) != 0 {
        std::process::exit(1);
    }

    let retval = CVodeQuadInitBS(&cvode_mem, indexB2, fQB2, &yQB2);
    if check_retval(retval, "CVodeQuadInitBS", 1) != 0 {
        std::process::exit(1);
    }

    let retval = CVodeQuadSStolerancesB(&cvode_mem, indexB2, reltol, abstolQB);
    if check_retval(retval, "CVodeQuadSStolerancesB", 1) != 0 {
        std::process::exit(1);
    }

    let retval = CVodeSetQuadErrConB(&cvode_mem, indexB2, SUNTRUE);
    if check_retval(retval, "CVodeSetQuadErrConB", 1) != 0 {
        std::process::exit(1);
    }

    /* Create a dense SUNMatrix */
    let AB2 = SUNDenseMatrix(2 * Neq, 2 * Neq, &sunctx);
    if check_retval_ptr(&AB2, "SUNDenseMatrix", 0) != 0 {
        std::process::exit(1);
    }
    let AB2 = AB2.expect("SUNDenseMatrix");

    /* Create dense SUNLinearSolver for the forward problem */
    let LSB2 = SUNLinSol_Dense(&yB2, &AB2, &sunctx);
    if check_retval_ptr(&LSB2, "SUNLinSol_Dense", 0) != 0 {
        std::process::exit(1);
    }
    let LSB2 = LSB2.expect("SUNLinSol_Dense");

    /* Attach the matrix and linear solver */
    let retval = CVodeSetLinearSolverB(&cvode_mem, indexB2, &LSB2, Some(&AB2));
    if check_retval(retval, "CVodeSetLinearSolverB", 1) != 0 {
        std::process::exit(1);
    }

    /* Backward integration */

    print!("---------------------------------------------\n");
    print!("Backward integration ... (2 adjoint problems)\n");
    print!("---------------------------------------------\n\n");

    let retval = CVodeB(&cvode_mem, t0, CV_NORMAL);
    if check_retval(retval, "CVodeB", 1) != 0 {
        std::process::exit(1);
    }

    let retval = CVodeGetB(&cvode_mem, indexB1, &mut time, &yB1);
    if check_retval(retval, "CVodeGetB", 1) != 0 {
        std::process::exit(1);
    }

    let retval = CVodeGetQuadB(&cvode_mem, indexB1, &mut time, &yQB1);
    if check_retval(retval, "CVodeGetQuadB", 1) != 0 {
        std::process::exit(1);
    }

    let retval = CVodeGetB(&cvode_mem, indexB2, &mut time, &yB2);
    if check_retval(retval, "CVodeGetB", 1) != 0 {
        std::process::exit(1);
    }

    let retval = CVodeGetQuadB(&cvode_mem, indexB2, &mut time, &yQB2);
    if check_retval(retval, "CVodeGetQuadB", 1) != 0 {
        std::process::exit(1);
    }

    print!(
        "   dG/dp:  {} {}   (from backward pb. 1)\n",
        fmt_ew(-Ith(&yQB1, 1), 12, 4),
        fmt_ew(-Ith(&yQB1, 2), 12, 4)
    );
    print!(
        "           {} {}   (from backward pb. 2)\n",
        fmt_ew(-Ith(&yQB2, 1), 12, 4),
        fmt_ew(-Ith(&yQB2, 2), 12, 4)
    );
    print!("\n");
    print!("   H = d2G/dp2:\n");
    print!("        (1)            (2)\n");
    print!(
        "  {}   {}\n",
        fmt_ew(-Ith(&yQB1, 3), 12, 4),
        fmt_ew(-Ith(&yQB2, 3), 12, 4)
    );
    print!(
        "  {}   {}\n",
        fmt_ew(-Ith(&yQB1, 4), 12, 4),
        fmt_ew(-Ith(&yQB2, 4), 12, 4)
    );
    print!("\n");

    print!("Final Statistics for backward pb. 1\n");
    print!("-----------------------------------\n");
    let retval = PrintBckStats(&cvode_mem, indexB1);
    if check_retval(retval, "PrintBckStats", 1) != 0 {
        std::process::exit(1);
    }

    print!("Final Statistics for backward pb. 2\n");
    print!("-----------------------------------\n");
    let retval = PrintBckStats(&cvode_mem, indexB2);
    if check_retval(retval, "PrintBckStats", 1) != 0 {
        std::process::exit(1);
    }

    /* Free memory */

    let mut cvode_mem = Some(cvode_mem);
    CVodeFree(&mut cvode_mem);
    SUNLinSolFree(Some(LS));
    SUNMatDestroy(A);
    SUNLinSolFree(Some(LSB1));
    SUNMatDestroy(AB1);
    SUNLinSolFree(Some(LSB2));
    SUNMatDestroy(AB2);

    /* Finite difference tests */

    let dp: sunrealtype = 1.0e-2;

    print!("-----------------------\n");
    print!("Finite Difference tests\n");
    print!("-----------------------\n\n");

    print!("del_p = {}\n\n", fmt_g(dp, 6));

    let cvode_mem = CVodeCreate(CV_BDF, &sunctx).expect("CVodeCreate");

    N_VConst(ONE, &y);
    N_VConst(ZERO, &yQ);

    let retval = CVodeInit(&cvode_mem, f, t0, &y);
    if check_retval(retval, "CVodeInit", 1) != 0 {
        std::process::exit(1);
    }

    let retval = CVodeSStolerances(&cvode_mem, reltol, abstol);
    if check_retval(retval, "CVodeSStolerances", 1) != 0 {
        std::process::exit(1);
    }

    let retval = CVodeSetUserData(&cvode_mem, Some(Box::new(data.clone())));
    if check_retval(retval, "CVodeSetUserData", 1) != 0 {
        std::process::exit(1);
    }

    /* Create a dense SUNMatrix */
    let A = SUNDenseMatrix(Neq, Neq, &sunctx);
    if check_retval_ptr(&A, "SUNDenseMatrix", 0) != 0 {
        std::process::exit(1);
    }
    let A = A.expect("SUNDenseMatrix");

    /* Create dense SUNLinearSolver for the forward problem */
    let LS = SUNLinSol_Dense(&y, &A, &sunctx);
    if check_retval_ptr(&LS, "SUNLinSol_Dense", 0) != 0 {
        std::process::exit(1);
    }
    let LS = LS.expect("SUNLinSol_Dense");

    /* Attach the matrix and linear solver */
    let retval = CVodeSetLinearSolver(&cvode_mem, &LS, Some(&A));
    if check_retval(retval, "CVodeSetLinearSolver", 1) != 0 {
        std::process::exit(1);
    }

    let retval = CVodeQuadInit(&cvode_mem, fQ, &yQ);
    if check_retval(retval, "CVodeQuadInit", 1) != 0 {
        std::process::exit(1);
    }

    let retval = CVodeQuadSStolerances(&cvode_mem, reltol, abstolQ);
    if check_retval(retval, "CVodeQuadSStolerances", 1) != 0 {
        std::process::exit(1);
    }

    let retval = CVodeSetQuadErrCon(&cvode_mem, SUNTRUE);
    if check_retval(retval, "CVodeSetQuadErrCon", 1) != 0 {
        std::process::exit(1);
    }

    data.borrow_mut().p1 += dp;

    let retval = CVode(&cvode_mem, tf, &y, &mut time, CV_NORMAL);
    if check_retval(retval, "CVode", 1) != 0 {
        std::process::exit(1);
    }

    let retval = CVodeGetQuad(&cvode_mem, &mut time, &yQ);
    if check_retval(retval, "CVodeGetQuad", 1) != 0 {
        std::process::exit(1);
    }

    let mut Gp = Ith(&yQ, 1);

    print!(
        "p1+  y:   {} {} {}",
        fmt_ew(Ith(&y, 1), 12, 4),
        fmt_ew(Ith(&y, 2), 12, 4),
        fmt_ew(Ith(&y, 3), 12, 4)
    );
    print!("     G:   {}\n", fmt_ew(Ith(&yQ, 1), 12, 4));

    data.borrow_mut().p1 -= 2.0 * dp;

    N_VConst(ONE, &y);
    N_VConst(ZERO, &yQ);

    CVodeReInit(&cvode_mem, t0, &y);
    CVodeQuadReInit(&cvode_mem, &yQ);

    let retval = CVode(&cvode_mem, tf, &y, &mut time, CV_NORMAL);
    if check_retval(retval, "CVode", 1) != 0 {
        std::process::exit(1);
    }

    let retval = CVodeGetQuad(&cvode_mem, &mut time, &yQ);
    if check_retval(retval, "CVodeGetQuad", 1) != 0 {
        std::process::exit(1);
    }

    let mut Gm = Ith(&yQ, 1);

    print!(
        "p1-  y:   {} {} {}",
        fmt_ew(Ith(&y, 1), 12, 4),
        fmt_ew(Ith(&y, 2), 12, 4),
        fmt_ew(Ith(&y, 3), 12, 4)
    );
    print!("     G:   {}\n", fmt_ew(Ith(&yQ, 1), 12, 4));

    data.borrow_mut().p1 += dp;

    grdG_fwd[0] = (Gp - G) / dp;
    grdG_bck[0] = (G - Gm) / dp;
    grdG_cntr[0] = (Gp - Gm) / (2.0 * dp);
    let H11 = (Gp - 2.0 * G + Gm) / (dp * dp);

    data.borrow_mut().p2 += dp;

    N_VConst(ONE, &y);
    N_VConst(ZERO, &yQ);

    CVodeReInit(&cvode_mem, t0, &y);
    CVodeQuadReInit(&cvode_mem, &yQ);

    let retval = CVode(&cvode_mem, tf, &y, &mut time, CV_NORMAL);
    if check_retval(retval, "CVode", 1) != 0 {
        std::process::exit(1);
    }

    let retval = CVodeGetQuad(&cvode_mem, &mut time, &yQ);
    if check_retval(retval, "CVodeGetQuad", 1) != 0 {
        std::process::exit(1);
    }

    Gp = Ith(&yQ, 1);

    print!(
        "p2+  y:   {} {} {}",
        fmt_ew(Ith(&y, 1), 12, 4),
        fmt_ew(Ith(&y, 2), 12, 4),
        fmt_ew(Ith(&y, 3), 12, 4)
    );
    print!("     G:   {}\n", fmt_ew(Ith(&yQ, 1), 12, 4));

    data.borrow_mut().p2 -= 2.0 * dp;

    N_VConst(ONE, &y);
    N_VConst(ZERO, &yQ);

    CVodeReInit(&cvode_mem, t0, &y);
    CVodeQuadReInit(&cvode_mem, &yQ);

    let retval = CVode(&cvode_mem, tf, &y, &mut time, CV_NORMAL);
    if check_retval(retval, "CVode", 1) != 0 {
        std::process::exit(1);
    }

    let retval = CVodeGetQuad(&cvode_mem, &mut time, &yQ);
    if check_retval(retval, "CVodeGetQuad", 1) != 0 {
        std::process::exit(1);
    }

    Gm = Ith(&yQ, 1);

    print!(
        "p2-  y:   {} {} {}",
        fmt_ew(Ith(&y, 1), 12, 4),
        fmt_ew(Ith(&y, 2), 12, 4),
        fmt_ew(Ith(&y, 3), 12, 4)
    );
    print!("     G:   {}\n", fmt_ew(Ith(&yQ, 1), 12, 4));

    data.borrow_mut().p2 += dp;

    grdG_fwd[1] = (Gp - G) / dp;
    grdG_bck[1] = (G - Gm) / dp;
    grdG_cntr[1] = (Gp - Gm) / (2.0 * dp);
    let H22 = (Gp - 2.0 * G + Gm) / (dp * dp);

    print!("\n");

    print!(
        "   dG/dp:  {} {}   (fwd FD)\n",
        fmt_ew(grdG_fwd[0], 12, 4),
        fmt_ew(grdG_fwd[1], 12, 4)
    );
    print!(
        "           {} {}   (bck FD)\n",
        fmt_ew(grdG_bck[0], 12, 4),
        fmt_ew(grdG_bck[1], 12, 4)
    );
    print!(
        "           {} {}   (cntr FD)\n",
        fmt_ew(grdG_cntr[0], 12, 4),
        fmt_ew(grdG_cntr[1], 12, 4)
    );
    print!("\n");
    print!("  H(1,1):  {}\n", fmt_ew(H11, 12, 4));
    print!("  H(2,2):  {}\n", fmt_ew(H22, 12, 4));

    /* Free memory */

    let mut cvode_mem = Some(cvode_mem);
    CVodeFree(&mut cvode_mem);
    SUNLinSolFree(Some(LS));
    SUNMatDestroy(A);

    N_VDestroy(y);
    N_VDestroy(yQ);

    N_VDestroyVectorArray(yS, Np);
    N_VDestroyVectorArray(yQS, Np);

    N_VDestroy(yB1);
    N_VDestroy(yQB1);
    N_VDestroy(yB2);
    N_VDestroy(yQB2);

    drop(sunctx);
    SUNContext_Free(&mut sunctx_opt);
}

/*
 *--------------------------------------------------------------------
 * FUNCTIONS CALLED BY CVODES
 *--------------------------------------------------------------------
 */

fn f(_t: sunrealtype, y: &N_Vector, ydot: &N_Vector, user_data: &mut Option<Box<dyn Any>>) -> i32 {
    let data = data_of(user_data);
    let (p1, p2) = {
        let d = data.borrow();
        (d.p1, d.p2)
    };

    let y1 = Ith(y, 1);
    let y2 = Ith(y, 2);
    let y3 = Ith(y, 3);

    Ith_set(ydot, 1, -p1 * y1 * y1 - y3);
    Ith_set(ydot, 2, -y2);
    Ith_set(ydot, 3, -p2 * p2 * y2 * y3);

    0
}

fn fQ(
    _t: sunrealtype,
    y: &N_Vector,
    qdot: &N_Vector,
    _user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    let y1 = Ith(y, 1);
    let y2 = Ith(y, 2);
    let y3 = Ith(y, 3);

    Ith_set(qdot, 1, 0.5 * (y1 * y1 + y2 * y2 + y3 * y3));

    0
}

#[allow(clippy::too_many_arguments)]
fn fS(
    _Ns: i32,
    _t: sunrealtype,
    y: &N_Vector,
    _ydot: &N_Vector,
    yS: &[N_Vector],
    ySdot: &[N_Vector],
    user_data: &mut Option<Box<dyn Any>>,
    _tmp1: &N_Vector,
    _tmp2: &N_Vector,
) -> i32 {
    let data = data_of(user_data);
    let (p1, p2) = {
        let d = data.borrow();
        (d.p1, d.p2)
    };

    let y1 = Ith(y, 1);
    let y2 = Ith(y, 2);
    let y3 = Ith(y, 3);

    /* 1st sensitivity RHS */

    let s1 = Ith(&yS[0], 1);
    let s2 = Ith(&yS[0], 2);
    let s3 = Ith(&yS[0], 3);

    let fys1 = -2.0 * p1 * y1 * s1 - s3;
    let fys2 = -s2;
    let fys3 = -p2 * p2 * y3 * s2 - p2 * p2 * y2 * s3;

    Ith_set(&ySdot[0], 1, fys1 - y1 * y1);
    Ith_set(&ySdot[0], 2, fys2);
    Ith_set(&ySdot[0], 3, fys3);

    /* 2nd sensitivity RHS */

    let s1 = Ith(&yS[1], 1);
    let s2 = Ith(&yS[1], 2);
    let s3 = Ith(&yS[1], 3);

    let fys1 = -2.0 * p1 * y1 * s1 - s3;
    let fys2 = -s2;
    let fys3 = -p2 * p2 * y3 * s2 - p2 * p2 * y2 * s3;

    Ith_set(&ySdot[1], 1, fys1);
    Ith_set(&ySdot[1], 2, fys2);
    Ith_set(&ySdot[1], 3, fys3 - 2.0 * p2 * y2 * y3);

    0
}

#[allow(clippy::too_many_arguments)]
fn fQS(
    _Ns: i32,
    _t: sunrealtype,
    y: &N_Vector,
    yS: &[N_Vector],
    _yQdot: &N_Vector,
    yQSdot: &[N_Vector],
    _user_data: &mut Option<Box<dyn Any>>,
    _tmp: &N_Vector,
    _tmpQ: &N_Vector,
) -> i32 {
    let y1 = Ith(y, 1);
    let y2 = Ith(y, 2);
    let y3 = Ith(y, 3);

    /* 1st sensitivity RHS */

    let s1 = Ith(&yS[0], 1);
    let s2 = Ith(&yS[0], 2);
    let s3 = Ith(&yS[0], 3);

    Ith_set(&yQSdot[0], 1, y1 * s1 + y2 * s2 + y3 * s3);

    /* 1st sensitivity RHS */

    let s1 = Ith(&yS[1], 1);
    let s2 = Ith(&yS[1], 2);
    let s3 = Ith(&yS[1], 3);

    Ith_set(&yQSdot[1], 1, y1 * s1 + y2 * s2 + y3 * s3);

    0
}

fn fB1(
    _t: sunrealtype,
    y: &N_Vector,
    yS: &[N_Vector],
    yB: &N_Vector,
    yBdot: &N_Vector,
    user_dataB: &mut Option<Box<dyn Any>>,
) -> i32 {
    let data = data_of(user_dataB);
    let (p1, p2) = {
        let d = data.borrow();
        (d.p1, d.p2)
    };

    let y1 = Ith(y, 1); /* solution */
    let y2 = Ith(y, 2);
    let y3 = Ith(y, 3);

    let s1 = Ith(&yS[0], 1); /* sensitivity 1 */
    let s2 = Ith(&yS[0], 2);
    let s3 = Ith(&yS[0], 3);

    let l1 = Ith(yB, 1); /* lambda */
    let l2 = Ith(yB, 2);
    let l3 = Ith(yB, 3);

    let m1 = Ith(yB, 4); /* mu */
    let m2 = Ith(yB, 5);
    let m3 = Ith(yB, 6);

    Ith_set(yBdot, 1, 2.0 * p1 * y1 * l1 - y1);
    Ith_set(yBdot, 2, l2 + p2 * p2 * y3 * l3 - y2);
    Ith_set(yBdot, 3, l1 + p2 * p2 * y2 * l3 - y3);

    Ith_set(
        yBdot,
        4,
        2.0 * p1 * y1 * m1 + l1 * 2.0 * (y1 + p1 * s1) - s1,
    );
    Ith_set(yBdot, 5, m2 + p2 * p2 * y3 * m3 + l3 * p2 * p2 * s3 - s2);
    Ith_set(yBdot, 6, m1 + p2 * p2 * y2 * m3 + l3 * p2 * p2 * s2 - s3);

    0
}

fn fQB1(
    _t: sunrealtype,
    y: &N_Vector,
    yS: &[N_Vector],
    yB: &N_Vector,
    qBdot: &N_Vector,
    user_dataB: &mut Option<Box<dyn Any>>,
) -> i32 {
    let data = data_of(user_dataB);

    let p2 = data.borrow().p2;

    let y1 = Ith(y, 1); /* solution */
    let y2 = Ith(y, 2);
    let y3 = Ith(y, 3);

    let s1 = Ith(&yS[0], 1); /* sensitivity 1 */
    let s2 = Ith(&yS[0], 2);
    let s3 = Ith(&yS[0], 3);

    let l1 = Ith(yB, 1); /* lambda */
    let l3 = Ith(yB, 3);

    let m1 = Ith(yB, 4); /* mu */
    let m3 = Ith(yB, 6);

    Ith_set(qBdot, 1, -y1 * y1 * l1);
    Ith_set(qBdot, 2, -2.0 * p2 * y2 * y3 * l3);

    Ith_set(qBdot, 3, -y1 * y1 * m1 - l1 * 2.0 * y1 * s1);
    Ith_set(
        qBdot,
        4,
        -2.0 * p2 * y2 * y3 * m3 - l3 * 2.0 * (p2 * y3 * s2 + p2 * y2 * s3),
    );

    0
}

fn fB2(
    _t: sunrealtype,
    y: &N_Vector,
    yS: &[N_Vector],
    yB: &N_Vector,
    yBdot: &N_Vector,
    user_dataB: &mut Option<Box<dyn Any>>,
) -> i32 {
    let data = data_of(user_dataB);
    let (p1, p2) = {
        let d = data.borrow();
        (d.p1, d.p2)
    };

    let y1 = Ith(y, 1); /* solution */
    let y2 = Ith(y, 2);
    let y3 = Ith(y, 3);

    let s1 = Ith(&yS[1], 1); /* sensitivity 2 */
    let s2 = Ith(&yS[1], 2);
    let s3 = Ith(&yS[1], 3);

    let l1 = Ith(yB, 1); /* lambda */
    let l2 = Ith(yB, 2);
    let l3 = Ith(yB, 3);

    let m1 = Ith(yB, 4); /* mu */
    let m2 = Ith(yB, 5);
    let m3 = Ith(yB, 6);

    Ith_set(yBdot, 1, 2.0 * p1 * y1 * l1 - y1);
    Ith_set(yBdot, 2, l2 + p2 * p2 * y3 * l3 - y2);
    Ith_set(yBdot, 3, l1 + p2 * p2 * y2 * l3 - y3);

    Ith_set(yBdot, 4, 2.0 * p1 * y1 * m1 + l1 * 2.0 * p1 * s1 - s1);
    Ith_set(
        yBdot,
        5,
        m2 + p2 * p2 * y3 * m3 + l3 * (2.0 * p2 * y3 + p2 * p2 * s3) - s2,
    );
    Ith_set(
        yBdot,
        6,
        m1 + p2 * p2 * y2 * m3 + l3 * (2.0 * p2 * y2 + p2 * p2 * s2) - s3,
    );

    0
}

fn fQB2(
    _t: sunrealtype,
    y: &N_Vector,
    yS: &[N_Vector],
    yB: &N_Vector,
    qBdot: &N_Vector,
    user_dataB: &mut Option<Box<dyn Any>>,
) -> i32 {
    let data = data_of(user_dataB);

    let p2 = data.borrow().p2;

    let y1 = Ith(y, 1); /* solution */
    let y2 = Ith(y, 2);
    let y3 = Ith(y, 3);

    let s1 = Ith(&yS[1], 1); /* sensitivity 2 */
    let s2 = Ith(&yS[1], 2);
    let s3 = Ith(&yS[1], 3);

    let l1 = Ith(yB, 1); /* lambda */
    let l3 = Ith(yB, 3);

    let m1 = Ith(yB, 4); /* mu */
    let m3 = Ith(yB, 6);

    Ith_set(qBdot, 1, -y1 * y1 * l1);
    Ith_set(qBdot, 2, -2.0 * p2 * y2 * y3 * l3);

    Ith_set(qBdot, 3, -y1 * y1 * m1 - l1 * 2.0 * y1 * s1);
    Ith_set(
        qBdot,
        4,
        -2.0 * p2 * y2 * y3 * m3 - l3 * 2.0 * (p2 * y3 * s2 + p2 * y2 * s3 + y2 * y3),
    );

    0
}

/*
 *--------------------------------------------------------------------
 * PRIVATE FUNCTIONS
 *--------------------------------------------------------------------
 */

fn PrintFwdStats(cvode_mem: &CVodeMem) -> i32 {
    let mut nst: i64 = 0;
    let mut nfe: i64 = 0;
    let mut nsetups: i64 = 0;
    let mut nni: i64 = 0;
    let mut ncfn: i64 = 0;
    let mut netf: i64 = 0;
    let mut nfQe: i64 = 0;
    let mut netfQ: i64 = 0;
    let mut nfSe: i64 = 0;
    let mut nfeS: i64 = 0;
    let mut nsetupsS: i64 = 0;
    let mut netfS: i64 = 0;
    let mut nfQSe: i64 = 0;
    let mut netfQS: i64 = 0;

    let mut qlast: i32 = 0;
    let mut qcur: i32 = 0;
    let mut h0u: sunrealtype = ZERO;
    let mut hlast: sunrealtype = ZERO;
    let mut hcur: sunrealtype = ZERO;
    let mut tcur: sunrealtype = ZERO;

    let _retval = CVodeGetIntegratorStats(
        cvode_mem,
        &mut nst,
        &mut nfe,
        &mut nsetups,
        &mut netf,
        &mut qlast,
        &mut qcur,
        &mut h0u,
        &mut hlast,
        &mut hcur,
        &mut tcur,
    );

    let _retval = CVodeGetNonlinSolvStats(cvode_mem, &mut nni, &mut ncfn);

    let _retval = CVodeGetQuadStats(cvode_mem, &mut nfQe, &mut netfQ);

    let _retval = CVodeGetSensStats(cvode_mem, &mut nfSe, &mut nfeS, &mut netfS, &mut nsetupsS);

    let retval = CVodeGetQuadSensStats(cvode_mem, &mut nfQSe, &mut netfQS);

    print!(" Number steps: {:5}\n\n", nst);
    print!(" Function evaluations:\n");
    print!(
        "  f:        {:5}\n  fQ:       {:5}\n  fS:       {:5}\n  fQS:      {:5}\n",
        nfe, nfQe, nfSe, nfQSe
    );
    print!(" Error test failures:\n");
    print!(
        "  netf:     {:5}\n  netfQ:    {:5}\n  netfS:    {:5}\n  netfQS:   {:5}\n",
        netf, netfQ, netfS, netfQS
    );
    print!(" Linear solver setups:\n");
    print!("  nsetups:  {:5}\n  nsetupsS: {:5}\n", nsetups, nsetupsS);
    print!(" Nonlinear iterations:\n");
    print!("  nni:      {:5}\n", nni);
    print!(" Convergence failures:\n");
    print!("  ncfn:     {:5}\n", ncfn);

    print!("\n");

    retval
}

fn PrintBckStats(cvode_mem: &CVodeMem, idx: i32) -> i32 {
    let mut nst: i64 = 0;
    let mut nfe: i64 = 0;
    let mut nsetups: i64 = 0;
    let mut nni: i64 = 0;
    let mut ncfn: i64 = 0;
    let mut netf: i64 = 0;
    let mut nfQe: i64 = 0;
    let mut netfQ: i64 = 0;

    let mut qlast: i32 = 0;
    let mut qcur: i32 = 0;
    let mut h0u: sunrealtype = ZERO;
    let mut hlast: sunrealtype = ZERO;
    let mut hcur: sunrealtype = ZERO;
    let mut tcur: sunrealtype = ZERO;

    let cvode_mem_bck = CVodeGetAdjCVodeBmem(cvode_mem, idx).expect("CVodeGetAdjCVodeBmem");

    let _retval = CVodeGetIntegratorStats(
        &cvode_mem_bck,
        &mut nst,
        &mut nfe,
        &mut nsetups,
        &mut netf,
        &mut qlast,
        &mut qcur,
        &mut h0u,
        &mut hlast,
        &mut hcur,
        &mut tcur,
    );

    let _retval = CVodeGetNonlinSolvStats(&cvode_mem_bck, &mut nni, &mut ncfn);

    let retval = CVodeGetQuadStats(&cvode_mem_bck, &mut nfQe, &mut netfQ);

    print!(" Number steps: {:5}\n\n", nst);
    print!(" Function evaluations:\n");
    print!("  f:        {:5}\n  fQ:       {:5}\n", nfe, nfQe);
    print!(" Error test failures:\n");
    print!("  netf:     {:5}\n  netfQ:    {:5}\n", netf, netfQ);
    print!(" Linear solver setups:\n");
    print!("  nsetups:  {:5}\n", nsetups);
    print!(" Nonlinear iterations:\n");
    print!("  nni:      {:5}\n", nni);
    print!(" Convergence failures:\n");
    print!("  ncfn:     {:5}\n", ncfn);

    print!("\n");

    retval
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
    if opt == 2 && returnvalue.is_none() {
        eprint!(
            "\nMEMORY_ERROR: {}() failed - returned NULL pointer\n\n",
            funcname
        );
        return 1;
    }

    0
}
