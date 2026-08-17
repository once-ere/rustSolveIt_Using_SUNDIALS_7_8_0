/* -----------------------------------------------------------------
 * Programmer: Radu Serban and Cosmin Petra @ LLNL
 * -----------------------------------------------------------------
 * Rust port of examples/idas/serial/idasSlCrank_FSA_dns.c
 * -----------------------------------------------------------------
 * Simulation of a slider-crank mechanism modelled with 3 generalized
 * coordinates: crank angle, connecting bar angle, and slider location.
 * The mechanism moves under the action of a constant horizontal
 * force applied to the connecting rod and a spring-damper connecting
 * the crank and connecting rod.
 *
 * The equations of motion are formulated as a system of stabilized
 * index-2 DAEs (Gear-Gupta-Leimkuhler formulation).
 *
 * IDAS also computes sensitivities with respect to the problem
 * parameters k (spring constant) and c (damper constant) of the
 * kinetic energy:
 *   G = int_t0^tend g(t,y,p) dt,
 * where
 *   g(t,y,p) = 0.5*J1*v1^2 + 0.5*J2*v3^2 + 0.5*m2*v2^2
 *
 * -----------------------------------------------------------------*/

#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

use idas_rs::prelude::*;

/* Problem Constants */

const NEQ: sunindextype = 10;
const NP: usize = 2;

const TBEGIN: sunrealtype = 0.0;
const TEND: sunrealtype = 10.000;

const RTOLF: sunrealtype = 1.0e-06;
const ATOLF: sunrealtype = 1.0e-07;

const RTOLQ: sunrealtype = 1.0e-06;
const ATOLQ: sunrealtype = 1.0e-08;

const RTOLFD: sunrealtype = 1.0e-06;
const ATOLFD: sunrealtype = 1.0e-08;

const ZERO: sunrealtype = 0.00;
const HALF: sunrealtype = 0.50;
const ONE: sunrealtype = 1.00;
const TWO: sunrealtype = 2.00;
const FOUR: sunrealtype = 4.00;

/* C macro `NV_Ith_S(v,i)` (0-based) and `Ith(v,i)` = `NV_Ith_S(v,i-1)`. */

fn NV_Ith_S(v: &N_Vector, i: usize) -> sunrealtype {
    N_VGetArrayPointer(v).expect("N_VGetArrayPointer")[i]
}

fn NV_Ith_S_set(v: &N_Vector, i: usize, x: sunrealtype) {
    N_VGetArrayPointer(v).expect("N_VGetArrayPointer")[i] = x;
}

fn Ith(v: &N_Vector, i: usize) -> sunrealtype {
    NV_Ith_S(v, i - 1)
}

fn Ith_set(v: &N_Vector, i: usize, x: sunrealtype) {
    NV_Ith_S_set(v, i - 1, x)
}

/* C `typedef struct {...}* UserData;` is a heap pointer shared by `main`
(which rewrites `params` for the finite-difference runs) and by the
integrator (which reaches it through `user_data`).  The port shares the
single record through `Rc<RefCell<UserData>>`, exactly as
`cvsFoodWeb_ASAi_kry` shares its `WebData`. */

struct UserData {
    a: sunrealtype,
    J1: sunrealtype,
    J2: sunrealtype,
    #[allow(dead_code)]
    m1: sunrealtype,
    m2: sunrealtype,
    l0: sunrealtype,
    /* C passes `data->params` (the caller's own array) to
    IDASetSensParams, which stores the POINTER in `ida_mem->ida_p`, so the
    internal DQ sensitivity residual perturbs the very array that `force`
    reads back through `user_data`.  The port shares the same array as a
    `SensParams` handle (ARCHITECTURE §8) and hands IDASetSensParams a
    clone of it. */
    params: SensParams,
    F: sunrealtype,
}

type UserDataRc = Rc<RefCell<UserData>>;

fn data_of(user_data: &mut Option<Box<dyn Any>>) -> UserDataRc {
    user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<UserDataRc>())
        .expect("user_data is UserData")
        .clone()
}

/*
 *--------------------------------------------------------------------
 * Main Program
 *--------------------------------------------------------------------
 */

fn main() {
    let mut retval: i32;
    let mut tret: sunrealtype = ZERO;
    let mut pbar: [sunrealtype; 2] = [ZERO; 2];
    let mut Gm: [sunrealtype; 2] = [ZERO; 2];
    let mut Gp: [sunrealtype; 2] = [ZERO; 2];
    let mut atolS: [sunrealtype; NP] = [ZERO; NP];

    /* Create the SUNDIALS context object for this simulation */
    let mut sunctx: Option<SUNContext> = None;
    retval = SUNContext_Create(SUN_COMM_NULL, &mut sunctx);
    if check_retval(Some(retval), "SUNContext_Create", 1) != 0 {
        std::process::exit(1);
    }
    let ctx = sunctx.as_ref().expect("sunctx").clone();

    let id = N_VNew_Serial(NEQ, &ctx).expect("N_VNew_Serial");
    let yy = N_VClone(&id).expect("N_VClone");
    let yp = N_VClone(&id).expect("N_VClone");
    let q = N_VNew_Serial(1, &ctx).expect("N_VNew_Serial");

    let yyS = N_VCloneVectorArray(NP as i32, &yy).expect("N_VCloneVectorArray");
    let ypS = N_VCloneVectorArray(NP as i32, &yp).expect("N_VCloneVectorArray");
    let qS = N_VCloneVectorArray(NP as i32, &q).expect("N_VCloneVectorArray");

    let data: UserDataRc = Rc::new(RefCell::new(UserData {
        a: 0.5,  /* half-length of crank */
        J1: 1.0, /* crank moment of inertia */
        m2: 1.0, /* mass of connecting rod */
        m1: 1.0, /* */
        J2: 2.0, /* moment of inertia of connecting rod */
        params: Rc::new(RefCell::new(vec![
            1.0, /* spring constant */
            1.0, /* damper constant */
        ])),
        l0: 1.0, /* spring free length */
        F: 1.0,  /* external constant force */
    }));

    N_VConst(ONE, &id);
    NV_Ith_S_set(&id, 9, ZERO);
    NV_Ith_S_set(&id, 8, ZERO);
    NV_Ith_S_set(&id, 7, ZERO);
    NV_Ith_S_set(&id, 6, ZERO);

    print!("\nSlider-Crank example for IDAS:\n");

    /* Consistent IC*/
    setIC(&yy, &yp, &data.borrow());

    for is in 0..NP {
        N_VConst(ZERO, &yyS[is]);
        N_VConst(ZERO, &ypS[is]);
    }

    /* IDA initialization.  C assigns each flag to `retval` and never reads
    it back; `let _ =` keeps the calls and drops the values identically. */
    let mut mem_opt = IDACreate(&ctx);
    let mem = mem_opt.as_ref().expect("IDACreate").clone();
    let _ = IDAInit(&mem, ressc, TBEGIN, &yy, &yp);
    let _ = IDASStolerances(&mem, RTOLF, ATOLF);
    let _ = IDASetUserData(&mem, Some(Box::new(Rc::clone(&data))));
    let _ = IDASetId(&mem, Some(&id));
    let _ = IDASetSuppressAlg(&mem, SUNTRUE);
    let _ = IDASetMaxNumSteps(&mem, 20000);

    /* Create dense SUNMatrix for use in linear solves */
    let A = SUNDenseMatrix(NEQ, NEQ, &ctx);
    if check_retval(A.as_ref().map(|_| 0), "SUNDenseMatrix", 0) != 0 {
        std::process::exit(1);
    }
    let A = A.expect("SUNDenseMatrix");

    /* Create dense SUNLinearSolver object */
    let LS = SUNLinSol_Dense(&yy, &A, &ctx);
    if check_retval(LS.as_ref().map(|_| 0), "SUNLinSol_Dense", 0) != 0 {
        std::process::exit(1);
    }
    let LS = LS.expect("SUNLinSol_Dense");

    /* Attach the matrix and linear solver */
    retval = IDASetLinearSolver(&mem, &LS, Some(&A));
    if check_retval(Some(retval), "IDASetLinearSolver", 1) != 0 {
        std::process::exit(1);
    }

    let _ = IDASensInit(&mem, NP as i32, IDA_SIMULTANEOUS, None, &yyS, &ypS);
    pbar[0] = data.borrow().params.borrow()[0];
    pbar[1] = data.borrow().params.borrow()[1];
    {
        /* C: IDASetSensParams(mem, data->params, pbar, NULL) — the very
        array `force` reads through `user_data`. */
        let params = Rc::clone(&data.borrow().params);
        let _ = IDASetSensParams(&mem, Some(params), Some(&pbar[..]), None);
    }
    let _ = IDASensEEtolerances(&mem);
    let _ = IDASetSensErrCon(&mem, SUNTRUE);

    N_VConst(ZERO, &q);
    let _ = IDAQuadInit(&mem, rhsQ, &q);
    let _ = IDAQuadSStolerances(&mem, RTOLQ, ATOLQ);
    let _ = IDASetQuadErrCon(&mem, SUNTRUE);

    for is in 0..NP {
        N_VConst(ZERO, &qS[is]);
    }
    let _ = IDAQuadSensInit(&mem, Some(rhsQS), &qS);
    atolS[1] = ATOLQ;
    atolS[0] = atolS[1];
    let _ = IDAQuadSensSStolerances(&mem, RTOLQ, &atolS[..]);
    let _ = IDASetQuadSensErrCon(&mem, SUNTRUE);

    /* Perform forward run */
    print!("\nForward integration ... ");

    retval = IDASolve(&mem, TEND, &mut tret, &yy, &yp, IDA_NORMAL);
    if check_retval(Some(retval), "IDASolve", 1) != 0 {
        std::process::exit(1);
    }

    print!("done!\n");

    retval = PrintFinalStats(&mem);
    if check_retval(Some(retval), "PrintFinalStats", 1) != 0 {
        std::process::exit(1);
    }

    IDAGetQuad(&mem, &mut tret, &q);
    print!("--------------------------------------------\n");
    print!("  G = {}\n", fmt_fw(Ith(&q, 1), 24, 16));
    print!("--------------------------------------------\n\n");

    IDAGetQuadSens(&mem, &mut tret, &qS);
    print!("-------------F O R W A R D------------------\n");
    print!(
        "   dG/dp:  {} {}\n",
        fmt_ew(Ith(&qS[0], 1), 12, 4),
        fmt_ew(Ith(&qS[1], 1), 12, 4)
    );
    print!("--------------------------------------------\n\n");

    IDAFree(&mut mem_opt);
    let _ = SUNLinSolFree(Some(LS));
    SUNMatDestroy(A);

    /* Finite differences for dG/dp */
    let dp: sunrealtype = 0.00001;
    data.borrow().params.borrow_mut()[0] = ONE;
    data.borrow().params.borrow_mut()[1] = ONE;

    let mut mem_opt = IDACreate(&ctx);
    let mem = mem_opt.as_ref().expect("IDACreate").clone();

    setIC(&yy, &yp, &data.borrow());
    let _ = IDAInit(&mem, ressc, TBEGIN, &yy, &yp);
    let _ = IDASStolerances(&mem, RTOLFD, ATOLFD);
    let _ = IDASetUserData(&mem, Some(Box::new(Rc::clone(&data))));
    let _ = IDASetId(&mem, Some(&id));
    let _ = IDASetSuppressAlg(&mem, SUNTRUE);

    /* Create dense SUNMatrix for use in linear solves */
    let A = SUNDenseMatrix(NEQ, NEQ, &ctx);
    if check_retval(A.as_ref().map(|_| 0), "SUNDenseMatrix", 0) != 0 {
        std::process::exit(1);
    }
    let A = A.expect("SUNDenseMatrix");

    /* Create dense SUNLinearSolver object */
    let LS = SUNLinSol_Dense(&yy, &A, &ctx);
    if check_retval(LS.as_ref().map(|_| 0), "SUNLinSol_Dense", 0) != 0 {
        std::process::exit(1);
    }
    let LS = LS.expect("SUNLinSol_Dense");

    /* Attach the matrix and linear solver */
    retval = IDASetLinearSolver(&mem, &LS, Some(&A));
    if check_retval(Some(retval), "IDASetLinearSolver", 1) != 0 {
        std::process::exit(1);
    }

    N_VConst(ZERO, &q);
    IDAQuadInit(&mem, rhsQ, &q);
    IDAQuadSStolerances(&mem, RTOLQ, ATOLQ);
    IDASetQuadErrCon(&mem, SUNTRUE);

    IDASolve(&mem, TEND, &mut tret, &yy, &yp, IDA_NORMAL);

    IDAGetQuad(&mem, &mut tret, &q);
    let G = Ith(&q, 1);
    /*printf("  G  =%12.6e\n", Ith(q,1));*/

    /******************************
     * BACKWARD for k
     ******************************/
    data.borrow().params.borrow_mut()[0] -= dp;
    setIC(&yy, &yp, &data.borrow());

    IDAReInit(&mem, TBEGIN, &yy, &yp);

    N_VConst(ZERO, &q);
    IDAQuadReInit(&mem, &q);

    IDASolve(&mem, TEND, &mut tret, &yy, &yp, IDA_NORMAL);
    IDAGetQuad(&mem, &mut tret, &q);
    Gm[0] = Ith(&q, 1);
    /*printf("Gm[0]=%12.6e\n", Ith(q,1));*/

    /****************************
     * FORWARD for k *
     ****************************/
    data.borrow().params.borrow_mut()[0] += TWO * dp;
    setIC(&yy, &yp, &data.borrow());
    IDAReInit(&mem, TBEGIN, &yy, &yp);

    N_VConst(ZERO, &q);
    IDAQuadReInit(&mem, &q);

    IDASolve(&mem, TEND, &mut tret, &yy, &yp, IDA_NORMAL);
    IDAGetQuad(&mem, &mut tret, &q);
    Gp[0] = Ith(&q, 1);
    /*printf("Gp[0]=%12.6e\n", Ith(q,1));*/

    /* Backward for c */
    data.borrow().params.borrow_mut()[0] = ONE;
    data.borrow().params.borrow_mut()[1] -= dp;
    setIC(&yy, &yp, &data.borrow());
    IDAReInit(&mem, TBEGIN, &yy, &yp);

    N_VConst(ZERO, &q);
    IDAQuadReInit(&mem, &q);

    IDASolve(&mem, TEND, &mut tret, &yy, &yp, IDA_NORMAL);
    IDAGetQuad(&mem, &mut tret, &q);
    Gm[1] = Ith(&q, 1);

    /* Forward for c */
    data.borrow().params.borrow_mut()[1] += TWO * dp;
    setIC(&yy, &yp, &data.borrow());
    IDAReInit(&mem, TBEGIN, &yy, &yp);

    N_VConst(ZERO, &q);
    IDAQuadReInit(&mem, &q);

    IDASolve(&mem, TEND, &mut tret, &yy, &yp, IDA_NORMAL);
    IDAGetQuad(&mem, &mut tret, &q);
    Gp[1] = Ith(&q, 1);

    IDAFree(&mut mem_opt);
    let _ = SUNLinSolFree(Some(LS));
    SUNMatDestroy(A);

    print!("\n\n   Checking using Finite Differences \n\n");

    print!("---------------BACKWARD------------------\n");
    print!(
        "   dG/dp:  {} {}\n",
        fmt_ew((G - Gm[0]) / dp, 12, 4),
        fmt_ew((G - Gm[1]) / dp, 12, 4)
    );
    print!("-----------------------------------------\n\n");

    print!("---------------FORWARD-------------------\n");
    print!(
        "   dG/dp:  {} {}\n",
        fmt_ew((Gp[0] - G) / dp, 12, 4),
        fmt_ew((Gp[1] - G) / dp, 12, 4)
    );
    print!("-----------------------------------------\n\n");

    print!("--------------CENTERED-------------------\n");
    print!(
        "   dG/dp:  {} {}\n",
        fmt_ew((Gp[0] - Gm[0]) / (TWO * dp), 12, 4),
        fmt_ew((Gp[1] - Gm[1]) / (TWO * dp), 12, 4)
    );
    print!("-----------------------------------------\n\n");

    /* Free memory */
    /* `data` is dropped with the last handle to it */

    N_VDestroy(id);
    N_VDestroy(yy);
    N_VDestroy(yp);
    N_VDestroy(q);

    N_VDestroyVectorArray(yyS, NP as i32);
    N_VDestroyVectorArray(ypS, NP as i32);
    N_VDestroyVectorArray(qS, NP as i32);

    let _ = SUNContext_Free(&mut sunctx);
}

fn setIC(yy: &N_Vector, yp: &N_Vector, data: &UserData) {
    N_VConst(ZERO, yy);
    N_VConst(ZERO, yp);

    let pi = FOUR * ONE.sun_atan();

    let a = data.a;
    let J1 = data.J1;
    let m2 = data.m2;
    let J2 = data.J2;

    let q = pi / TWO;
    let p = (-a).sun_asin();
    let x = p.sun_cos();

    NV_Ith_S_set(yy, 0, q);
    NV_Ith_S_set(yy, 1, x);
    NV_Ith_S_set(yy, 2, p);

    let mut Q: [sunrealtype; 3] = [ZERO; 3];
    force(yy, &mut Q, data);

    NV_Ith_S_set(yp, 3, Q[0] / J1);
    NV_Ith_S_set(yp, 4, Q[1] / m2);
    NV_Ith_S_set(yp, 5, Q[2] / J2);
}

fn force(yy: &N_Vector, Q: &mut [sunrealtype; 3], data: &UserData) {
    let a = data.a;
    let k = data.params.borrow()[0];
    let c = data.params.borrow()[1];
    let l0 = data.l0;
    let F = data.F;

    let q = NV_Ith_S(yy, 0);
    let x = NV_Ith_S(yy, 1);
    let p = NV_Ith_S(yy, 2);

    let qd = NV_Ith_S(yy, 3);
    let xd = NV_Ith_S(yy, 4);
    let pd = NV_Ith_S(yy, 5);

    let s1 = q.sun_sin();
    let c1 = q.sun_cos();
    let s2 = p.sun_sin();
    let c2 = p.sun_cos();
    let s21 = s2 * c1 - c2 * s1;
    let c21 = c2 * c1 + s2 * s1;

    let l2 = x * x - x * (c2 + a * c1) + (ONE + a * a) / FOUR + a * c21 / TWO;
    let l = l2.sqrt();
    let mut ld =
        TWO * x * xd - xd * (c2 + a * c1) + x * (s2 * pd + a * s1 * qd) - a * s21 * (pd - qd) / TWO;
    ld /= TWO * l;

    let f = k * (l - l0) + c * ld;
    let fl = f / l;

    Q[0] = -fl * a * (s21 / TWO + x * s1) / TWO;
    Q[1] = fl * (c2 / TWO - x + a * c1 / TWO) + F;
    Q[2] = -fl * (x * s2 - a * s21 / TWO) / TWO - F * s2;
}

fn ressc(
    _tres: sunrealtype,
    yy: &N_Vector,
    yp: &N_Vector,
    rr: &N_Vector,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    let data_rc = data_of(user_data);
    let data = data_rc.borrow();

    let a = data.a;
    let J1 = data.J1;
    let m2 = data.m2;
    let J2 = data.J2;

    let (q, x, p, qd, xd, pd, lam1, lam2, mu1, mu2) = {
        let yval = N_VGetArrayPointer(yy).expect("N_VGetArrayPointer");
        (
            yval[0], yval[1], yval[2], yval[3], yval[4], yval[5], yval[6], yval[7], yval[8],
            yval[9],
        )
    };
    let ypval: [sunrealtype; 6] = {
        let yp_data = N_VGetArrayPointer(yp).expect("N_VGetArrayPointer");
        [
            yp_data[0], yp_data[1], yp_data[2], yp_data[3], yp_data[4], yp_data[5],
        ]
    };

    let s1 = q.sun_sin();
    let c1 = q.sun_cos();
    let s2 = p.sun_sin();
    let c2 = p.sun_cos();

    let mut Q: [sunrealtype; 3] = [ZERO; 3];
    force(yy, &mut Q, &data);

    let mut rval = N_VGetArrayPointer(rr).expect("N_VGetArrayPointer");

    rval[0] = ypval[0] - qd + a * s1 * mu1 - a * c1 * mu2;
    rval[1] = ypval[1] - xd + mu1;
    rval[2] = ypval[2] - pd + s2 * mu1 - c2 * mu2;

    rval[3] = J1 * ypval[3] - Q[0] + a * s1 * lam1 - a * c1 * lam2;
    rval[4] = m2 * ypval[4] - Q[1] + lam1;
    rval[5] = J2 * ypval[5] - Q[2] + s2 * lam1 - c2 * lam2;

    rval[6] = x - c2 - a * c1;
    rval[7] = -s2 - a * s1;

    rval[8] = a * s1 * qd + xd + s2 * pd;
    rval[9] = -a * c1 * qd - c2 * pd;

    0
}

fn rhsQ(
    _t: sunrealtype,
    yy: &N_Vector,
    _yp: &N_Vector,
    qdot: &N_Vector,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    let data_rc = data_of(user_data);
    let (J1, m2, J2) = {
        let data = data_rc.borrow();
        (data.J1, data.m2, data.J2)
    };

    let v1 = Ith(yy, 4);
    let v2 = Ith(yy, 5);
    let v3 = Ith(yy, 6);

    Ith_set(qdot, 1, HALF * (J1 * v1 * v1 + m2 * v2 * v2 + J2 * v3 * v3));

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
    user_data: &mut Option<Box<dyn Any>>,
    _yytmp: &N_Vector,
    _yptmp: &N_Vector,
    _tmpQS: &N_Vector,
) -> i32 {
    let data_rc = data_of(user_data);
    let (J1, m2, J2) = {
        let data = data_rc.borrow();
        (data.J1, data.m2, data.J2)
    };

    let v1 = Ith(yy, 4);
    let v2 = Ith(yy, 5);
    let v3 = Ith(yy, 6);

    /* Sensitivities of v. */
    let mut s1 = Ith(&yyS[0], 4);
    let mut s2 = Ith(&yyS[0], 5);
    let mut s3 = Ith(&yyS[0], 6);

    Ith_set(&rhsvalQS[0], 1, J1 * v1 * s1 + m2 * v2 * s2 + J2 * v3 * s3);

    s1 = Ith(&yyS[1], 4);
    s2 = Ith(&yyS[1], 5);
    s3 = Ith(&yyS[1], 6);

    Ith_set(&rhsvalQS[1], 1, J1 * v1 * s1 + m2 * v2 * s2 + J2 * v3 * s3);

    0
}

fn PrintFinalStats(mem: &IDAMem) -> i32 {
    let (mut nst, mut nni, mut nnf, mut nje, mut nre, mut nreLS, mut netf, mut ncfn) =
        (0i64, 0i64, 0i64, 0i64, 0i64, 0i64, 0i64, 0i64);

    /* C overwrites `retval` on every call and returns the LAST one. */
    let _ = IDAGetNumSteps(mem, &mut nst);
    let _ = IDAGetNumResEvals(mem, &mut nre);
    let _ = IDAGetNumJacEvals(mem, &mut nje);
    let _ = IDAGetNumNonlinSolvIters(mem, &mut nni);
    let _ = IDAGetNumErrTestFails(mem, &mut netf);
    let _ = IDAGetNumNonlinSolvConvFails(mem, &mut nnf);
    let _ = IDAGetNumStepSolveFails(mem, &mut ncfn);
    let retval = IDAGetNumLinResEvals(mem, &mut nreLS);

    print!("\nFinal Run Statistics: \n\n");
    print!("Number of steps                    = {}\n", nst);
    print!("Number of residual evaluations     = {}\n", nre + nreLS);
    print!("Number of Jacobian evaluations     = {}\n", nje);
    print!("Number of nonlinear iterations     = {}\n", nni);
    print!("Number of error test failures      = {}\n", netf);
    print!("Number of nonlinear conv. failures = {}\n", nnf);
    print!("Number of step solver failures     = {}\n", ncfn);

    retval
}

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
