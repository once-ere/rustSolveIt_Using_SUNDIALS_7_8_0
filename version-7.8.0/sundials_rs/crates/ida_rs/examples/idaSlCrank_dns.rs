//! Adversarial-verification harness: port of
//! `examples/ida/serial/idaSlCrank_dns.c`.

#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

use std::any::Any;

use ida_rs::prelude::*;

const NEQ: sunindextype = 10;
const TEND: sunrealtype = 10.0;
const NOUT: i32 = 41;
const ZERO: sunrealtype = 0.0;
const ONE: sunrealtype = 1.0;
const TWO: sunrealtype = 2.0;
const FOUR: sunrealtype = 4.0;

#[derive(Clone, Copy)]
struct UserData {
    a: sunrealtype,
    J1: sunrealtype,
    J2: sunrealtype,
    m2: sunrealtype,
    k: sunrealtype,
    c: sunrealtype,
    l0: sunrealtype,
    F: sunrealtype,
}

fn NV_Ith_S(v: &N_Vector, i: usize) -> sunrealtype {
    N_VGetArrayPointer(v).unwrap()[i]
}
fn NV_Ith_S_set(v: &N_Vector, i: usize, x: sunrealtype) {
    N_VGetArrayPointer(v).unwrap()[i] = x;
}

fn force(yy: &N_Vector, Q: &mut [sunrealtype; 3], data: &UserData) {
    let a = data.a;
    let k = data.k;
    let c = data.c;
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
    let mut ld = TWO * x * xd - xd * (c2 + a * c1) + x * (s2 * pd + a * s1 * qd)
        - a * s21 * (pd - qd) / TWO;
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
        .as_ref()
        .unwrap()
        .downcast_ref::<UserData>()
        .unwrap();

    let a = data.a;
    let J1 = data.J1;
    let m2 = data.m2;
    let J2 = data.J2;

    let yval = N_VGetArrayPointer(yy).unwrap().clone();
    let ypval = N_VGetArrayPointer(yp).unwrap().clone();

    let q = yval[0];
    let x = yval[1];
    let p = yval[2];
    let qd = yval[3];
    let xd = yval[4];
    let pd = yval[5];
    let lam1 = yval[6];
    let lam2 = yval[7];
    let mu1 = yval[8];
    let mu2 = yval[9];

    let s1 = q.sun_sin();
    let c1 = q.sun_cos();
    let s2 = p.sun_sin();
    let c2 = p.sun_cos();

    let mut Q = [0.0; 3];
    force(yy, &mut Q, &data);

    let mut rval = N_VGetArrayPointer(rr).unwrap();
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

    let mut Q = [0.0; 3];
    force(yy, &mut Q, data);

    NV_Ith_S_set(yp, 3, Q[0] / J1);
    NV_Ith_S_set(yp, 4, Q[1] / m2);
    NV_Ith_S_set(yp, 5, Q[2] / J2);
}

fn print_output(mem: &IDAMem, t: sunrealtype, y: &N_Vector) {
    let yval = N_VGetArrayPointer(y).unwrap().clone();
    let mut kused = 0i32;
    let _ = IDAGetLastOrder(mem, &mut kused);
    let mut nst = 0i64;
    let _ = IDAGetNumSteps(mem, &mut nst);
    let mut hused = 0.0;
    let _ = IDAGetLastStep(mem, &mut hused);
    println!(
        "{} {} {} {} {:>3}  {:>1} {}",
        fmt_ew(t, 10, 4),
        fmt_ew(yval[0], 12, 4),
        fmt_ew(yval[1], 12, 4),
        fmt_ew(yval[2], 12, 4),
        nst,
        kused,
        fmt_ew(hused, 12, 4)
    );
}

fn main() {
    let mut sunctx: Option<SUNContext> = None;
    let _ = SUNContext_Create(SUN_COMM_NULL, &mut sunctx);
    let ctx = sunctx.clone().unwrap();

    let data = UserData {
        a: 0.5,
        J1: 1.0,
        m2: 1.0,
        J2: 2.0,
        k: 1.0,
        c: 1.0,
        l0: 1.0,
        F: 1.0,
    };

    let yy = N_VNew_Serial(NEQ, &ctx).unwrap();
    let yp = N_VClone(&yy).unwrap();
    let id = N_VClone(&yy).unwrap();

    setIC(&yy, &yp, &data);

    N_VConst(ONE, &id);
    NV_Ith_S_set(&id, 6, ZERO);
    NV_Ith_S_set(&id, 7, ZERO);
    NV_Ith_S_set(&id, 8, ZERO);
    NV_Ith_S_set(&id, 9, ZERO);

    let rtol = 1.0e-6;
    let atol = 1.0e-6;

    let t0 = ZERO;
    let tf = TEND;
    let dt = (tf - t0) / ((NOUT - 1) as sunrealtype);

    let mem = IDACreate(&ctx).unwrap();
    let _ = IDAInit(&mem, ressc, t0, &yy, &yp);
    let _ = IDASStolerances(&mem, rtol, atol);
    let _ = IDASetUserData(&mem, Some(Box::new(data)));
    let _ = IDASetId(&mem, Some(&id));
    let _ = IDASetSuppressAlg(&mem, SUNTRUE);

    let A = SUNDenseMatrix(NEQ, NEQ, &ctx).unwrap();
    let LS = SUNLinSol_Dense(&yy, &A, &ctx).unwrap();
    let _ = IDASetLinearSolver(&mem, &LS, Some(&A));

    println!("\nidaSlCrank_dns: Slider-Crank DAE serial example problem for IDA");
    println!("Linear solver: DENSE, Jacobian is computed by IDA.");
    println!(
        "Tolerance parameters:  rtol = {}   atol = {}",
        fmt_g(rtol, 6),
        fmt_g(atol, 6)
    );
    println!("-----------------------------------------------------------------------");
    println!("  t            y1          y2           y3      | nst  k      h");
    println!("-----------------------------------------------------------------------");

    print_output(&mem, t0, &yy);

    let mut tret = 0.0;
    for iout in 1..NOUT {
        let tout = (iout as sunrealtype) * dt;
        let retval = IDASolve(&mem, tout, &mut tret, &yy, &yp, IDA_NORMAL);
        if retval < 0 {
            break;
        }
        print_output(&mem, tret, &yy);
    }

    let mut nst = 0i64;
    let _ = IDAGetNumSteps(&mem, &mut nst);
    let mut nre = 0i64;
    let _ = IDAGetNumResEvals(&mem, &mut nre);
    let mut nje = 0i64;
    let _ = IDAGetNumJacEvals(&mem, &mut nje);
    let mut nni = 0i64;
    let _ = IDAGetNumNonlinSolvIters(&mem, &mut nni);
    let mut netf = 0i64;
    let _ = IDAGetNumErrTestFails(&mem, &mut netf);
    let mut nnf = 0i64;
    let _ = IDAGetNumNonlinSolvConvFails(&mem, &mut nnf);
    let mut ncfn = 0i64;
    let _ = IDAGetNumStepSolveFails(&mem, &mut ncfn);
    let mut nreLS = 0i64;
    let _ = IDAGetNumLinResEvals(&mem, &mut nreLS);

    println!("\nFinal Run Statistics: \n");
    println!("Number of steps                    = {}", nst);
    println!("Number of residual evaluations     = {}", nre + nreLS);
    println!("Number of Jacobian evaluations     = {}", nje);
    println!("Number of nonlinear iterations     = {}", nni);
    println!("Number of error test failures      = {}", netf);
    println!("Number of nonlinear conv. failures = {}", nnf);
    println!("Number of step solver failures     = {}", ncfn);
}
