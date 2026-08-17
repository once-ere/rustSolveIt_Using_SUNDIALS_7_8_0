/* -----------------------------------------------------------------
 * Programmer: Radu Serban and Cosmin Petra @ LLNL
 * -----------------------------------------------------------------
 * Rust port of examples/idas/serial/idasSlCrank_dns.c
 * -----------------------------------------------------------------
 * Simulation of a slider-crank mechanism modelled with 3 generalized
 * coordinates: crank angle, connecting bar angle, and slider location.
 * The mechanism moves under the action of a constant horizontal force
 * applied to the connecting rod and a spring-damper connecting the crank
 * and connecting rod.
 *
 * The equations of motion are formulated as a system of stabilized
 * index-2 DAEs (Gear-Gupta-Leimkuhler formulation).
 *
 * IDAS also computes the average kinetic energy as the quadrature:
 *   G = int_t0^tend g(t,y,p) dt,
 * where
 *   g(t,y,p) = 0.5*J1*v1^2 + 0.5*J2*v3^2 + 0.5*m2*v2^2
 *
 * -----------------------------------------------------------------
 */

#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

use std::any::Any;

use idas_rs::prelude::*;

/* Problem Constants */

const NEQ: sunindextype = 10;

const TBEGIN: sunrealtype = 0.0;
const TEND: sunrealtype = 10.0;

const NOUT: i32 = 25;

const RTOLF: sunrealtype = 1.0e-06;
const ATOLF: sunrealtype = 1.0e-07;

const RTOLQ: sunrealtype = 1.0e-06;
const ATOLQ: sunrealtype = 1.0e-08;

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

/* C: `typedef struct {...}* UserData;` — the fields are never mutated
after initialization in this example, so a plain `Copy` record boxed
into `user_data` reproduces the shared-pointer semantics exactly. */

#[derive(Clone, Copy)]
struct UserData {
    a: sunrealtype,
    J1: sunrealtype,
    J2: sunrealtype,
    #[allow(dead_code)]
    m1: sunrealtype,
    m2: sunrealtype,
    l0: sunrealtype,
    params: [sunrealtype; 2],
    F: sunrealtype,
}

/*
 *--------------------------------------------------------------------
 * Main Program
 *--------------------------------------------------------------------
 */

fn main() {
    let mut retval: i32;

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

    let data = UserData {
        a: 0.5,                                   /* half-length of crank */
        J1: 1.0,                                  /* crank moment of inertia */
        m2: 1.0,                                  /* mass of connecting rod */
        m1: 1.0,                                  /* */
        J2: 2.0,                                  /* moment of inertia of connecting rod */
        params: [1.0 /* spring constant */, 1.0], /* damper constant */
        l0: 1.0,                                  /* spring free length */
        F: 1.0,                                   /* external constant force */
    };

    N_VConst(ONE, &id);
    NV_Ith_S_set(&id, 9, ZERO);
    NV_Ith_S_set(&id, 8, ZERO);
    NV_Ith_S_set(&id, 7, ZERO);
    NV_Ith_S_set(&id, 6, ZERO);

    /* Consistent IC*/
    setIC(&yy, &yp, &data);

    /* IDAS initialization.  C assigns each flag to `retval` and never reads
    it back; `let _ =` keeps the calls and drops the values identically. */
    let mut mem_opt = IDACreate(&ctx);
    let mem = mem_opt.as_ref().expect("IDACreate").clone();
    let _ = IDAInit(&mem, ressc, TBEGIN, &yy, &yp);
    let _ = IDASStolerances(&mem, RTOLF, ATOLF);
    let _ = IDASetUserData(&mem, Some(Box::new(data)));
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

    N_VConst(ZERO, &q);
    let _ = IDAQuadInit(&mem, rhsQ, &q);
    let _ = IDAQuadSStolerances(&mem, RTOLQ, ATOLQ);
    let _ = IDASetQuadErrCon(&mem, SUNTRUE);

    PrintHeader(RTOLF, ATOLF, &yy);

    /* Print initial states */
    PrintOutput(&mem, 0.0, &yy);

    /* Perform forward run */
    let mut tret: sunrealtype = ZERO;
    let mut tout = TEND / (NOUT as sunrealtype);

    loop {
        retval = IDASolve(&mem, tout, &mut tret, &yy, &yp, IDA_NORMAL);
        if check_retval(Some(retval), "IDASolve", 1) != 0 {
            std::process::exit(1);
        }

        PrintOutput(&mem, tret, &yy);

        tout += TEND / (NOUT as sunrealtype);

        if tret > TEND {
            break;
        }
    }

    retval = PrintFinalStats(&mem);
    if check_retval(Some(retval), "PrintFinalStats", 1) != 0 {
        std::process::exit(1);
    }

    IDAGetQuad(&mem, &mut tret, &q);
    print!("--------------------------------------------\n");
    print!("  G = {}\n", fmt_fw(Ith(&q, 1), 24, 16));
    print!("--------------------------------------------\n\n");

    IDAFree(&mut mem_opt);

    /* Free memory */

    /* `data` is owned by the solver memory and is dropped with it */
    let _ = SUNLinSolFree(Some(LS));
    SUNMatDestroy(A);
    N_VDestroy(id);
    N_VDestroy(yy);
    N_VDestroy(yp);
    N_VDestroy(q);
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
    let k = data.params[0];
    let c = data.params[1];
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
    let data = *user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<UserData>())
        .expect("user_data is UserData");

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
    let data = *user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<UserData>())
        .expect("user_data is UserData");
    let J1 = data.J1;
    let m2 = data.m2;
    let J2 = data.J2;

    let v1 = Ith(yy, 4);
    let v2 = Ith(yy, 5);
    let v3 = Ith(yy, 6);

    Ith_set(qdot, 1, HALF * (J1 * v1 * v1 + m2 * v2 * v2 + J2 * v3 * v3));

    0
}

fn PrintHeader(rtol: sunrealtype, avtol: sunrealtype, _y: &N_Vector) {
    print!("\nidasSlCrank_dns: Slider-Crank DAE serial example problem for IDAS\n");
    print!("Linear solver: DENSE, Jacobian is computed by IDAS.\n");
    print!(
        "Tolerance parameters:  rtol = {}   atol = {}\n",
        fmt_g(rtol, 6),
        fmt_g(avtol, 6)
    );
    print!("-----------------------------------------------------------------------\n");
    print!("  t         y1          y2           y3");
    print!("      | nst  k      h\n");
    print!("-----------------------------------------------------------------------\n");
}

fn PrintOutput(mem: &IDAMem, t: sunrealtype, y: &N_Vector) {
    let mut retval: i32;
    let mut kused: i32 = 0;
    let mut nst: i64 = 0;
    let mut hused: sunrealtype = ZERO;

    let yval: [sunrealtype; 3] = {
        let d = N_VGetArrayPointer(y).expect("N_VGetArrayPointer");
        [d[0], d[1], d[2]]
    };

    retval = IDAGetLastOrder(mem, &mut kused);
    check_retval(Some(retval), "IDAGetLastOrder", 1);
    retval = IDAGetNumSteps(mem, &mut nst);
    check_retval(Some(retval), "IDAGetNumSteps", 1);
    retval = IDAGetLastStep(mem, &mut hused);
    check_retval(Some(retval), "IDAGetLastStep", 1);

    print!(
        "{} {} {} {} | {:>3}  {:>1} {}\n",
        fmt_fw(t, 5, 2),
        fmt_ew(yval[0], 12, 4),
        fmt_ew(yval[1], 12, 4),
        fmt_ew(yval[2], 12, 4),
        nst,
        kused,
        fmt_ew(hused, 12, 4)
    );
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

/*
 * Check function return value...
 *   opt == 0 means SUNDIALS function allocates memory so check if
 *            returned NULL pointer
 *   opt == 1 means SUNDIALS function returns an integer value so check if
 *            retval < 0
 *   opt == 2 means function allocates memory so check if returned
 *            NULL pointer
 */

fn check_retval(returnvalue: Option<i32>, funcname: &str, opt: i32) -> i32 {
    /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
    if opt == 0 && returnvalue.is_none() {
        eprint!(
            "\nSUNDIALS_ERROR: {}() failed - returned NULL pointer\n\n",
            funcname
        );
        return 1;
    } else if opt == 1 {
        /* Check if retval < 0 */
        let retval = returnvalue.expect("retval");
        if retval < 0 {
            eprint!(
                "\nSUNDIALS_ERROR: {}() failed with retval = {}\n\n",
                funcname, retval
            );
            return 1;
        }
    } else if opt == 2 && returnvalue.is_none() {
        /* Check if function returned NULL pointer - no memory allocated */
        eprint!(
            "\nMEMORY_ERROR: {}() failed - returned NULL pointer\n\n",
            funcname
        );
        return 1;
    }

    0
}
