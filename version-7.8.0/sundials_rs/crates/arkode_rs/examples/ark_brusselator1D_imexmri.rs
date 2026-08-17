//! Port of `examples/arkode/C_serial/ark_brusselator1D_imexmri.c`.
//!
//! Example problem:
//!
//! The following test simulates a brusselator problem from chemical
//! kinetics.  This is n PDE system with 3 components, Y = [u,v,w],
//! satisfying the equations,
//!    u_t = du*u_xx - au*u_x +  a - (w+1)*u + v*u^2
//!    v_t = dv*v_xx - av*v_x +  w*u - v*u^2
//!    w_t = dw*w_xx - aw*w_x + (b-w)/ep - w*u
//! for t in [0, 10], x in [0, 1], with initial conditions
//!    u(0,x) =  a  + 0.1*sin(pi*x)
//!    v(0,x) = b/a + 0.1*sin(pi*x)
//!    w(0,x) =  b  + 0.1*sin(pi*x),
//! and with stationary boundary conditions, i.e.
//!    u_t(t,0) = u_t(t,1) = 0,
//!    v_t(t,0) = v_t(t,1) = 0,
//!    w_t(t,0) = w_t(t,1) = 0.
//!
//! This program solves the problem with multiple solvers listed below.
//! We select method to used based on solve_type input:
//! 0. MIS with third order dirk inner
//! 1. 5th order dirk method for reference solution
//! 2. MRI-GARK34a with erk inner
//! 3. MRI-GARK34a with dirk inner
//! 4. IMEX-MRI3b with erk inner
//! 5. IMEX-MRI3b with dirk inner
//! 6. IMEX-MRI4 with erk inner
//! 7. IMEX-MRI4 with dirk inner
//!
//! This program solves the problem with the MRI stepper. 10 outputs are
//! printed at equal intervals, and run statistics are printed at the end.

#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

use std::any::Any;
use std::fs::File;
use std::io::Write;

use arkode_rs::prelude::*;

/* Define some constants */
const ZERO: sunrealtype = 0.0;
const ONE: sunrealtype = 1.0;
const TWO: sunrealtype = 2.0;

/* accessor macros between (x,v) location and 1D NVector array */
fn IDX(x: sunindextype, v: sunindextype) -> sunindextype {
    3 * x + v
}

/* C `SM_ELEMENT_B(A,i,j) += v` — the C macro is an lvalue, the ported
accessors are a get/set pair. */
fn SM_ELEMENT_B_add(A: &SUNMatrix, i: sunindextype, j: sunindextype, v: sunrealtype) {
    SM_ELEMENT_B_set(A, i, j, SM_ELEMENT_B(A, i, j) + v);
}

/* C `atol` (strtol semantics): longest valid leading integer, 0 otherwise */
fn atol(s: &str) -> i64 {
    let t = s.trim_start_matches([' ', '\t', '\n', '\x0b', '\x0c', '\r']);
    let b = t.as_bytes();
    let mut i = 0usize;
    if i < b.len() && (b[i] == b'+' || b[i] == b'-') {
        i += 1;
    }
    let start = i;
    while i < b.len() && b[i].is_ascii_digit() {
        i += 1;
    }
    if i == start {
        return 0;
    }
    t[..i].parse::<i64>().unwrap_or(0)
}

/* C `atof(s)` is `strtod(s, NULL)` — `SUNStrToReal` implements exactly that */
fn atof(s: &str) -> sunrealtype {
    SUNStrToReal(s)
}

/* user data structure */
#[derive(Clone)]
struct UserData {
    N: sunindextype, /* number of intervals     */
    dx: sunrealtype, /* mesh spacing            */
    a: sunrealtype,  /* constant forcing on u   */
    b: sunrealtype,  /* steady-state value of w */
    pi: sunrealtype, /* value of pi             */
    du: sunrealtype, /* diffusion coeff for u   */
    dv: sunrealtype, /* diffusion coeff for v   */
    dw: sunrealtype, /* diffusion coeff for w   */
    au: sunrealtype, /* advection coeff for u   */
    av: sunrealtype, /* advection coeff for v   */
    aw: sunrealtype, /* advection coeff for w   */
    ep: sunrealtype, /* stiffness parameter     */
}

/* Main Program */
fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let argc = argv.len();

    /* general problem parameters */
    let T0: sunrealtype = ZERO; /* initial time                 */
    let Tf: sunrealtype = 10.0; /* final time                   */
    let Nt: i32 = 10; /* total number of output times */
    let dTout: sunrealtype = (Tf - T0) / Nt as sunrealtype; /* time between outputs */
    let Nvar: sunindextype = 3; /* number of solution fields    */
    let N: sunindextype = 101; /* spatial mesh size            */
    let hs: sunrealtype; /* slow step size       */
    let m: sunrealtype = 10.0; /* time-scale separation factor */
    let solve_type: i32; /* solver configuration */
    let dx: sunrealtype = ONE / (N - 1) as sunrealtype; /* set spatial mesh spacing */
    let a: sunrealtype = 0.6; /* problem parameters           */
    let b: sunrealtype = 2.0;
    let pi: sunrealtype = 4.0 * (ONE).sun_atan();
    let du: sunrealtype = 0.01;
    let dv: sunrealtype = 0.01;
    let dw: sunrealtype = 0.01;
    let au: sunrealtype = -0.001;
    let av: sunrealtype = -0.001;
    let aw: sunrealtype = -0.001;
    let ep: sunrealtype = 1.0e-2; /* stiffness parameter          */
    let mut reltol: sunrealtype = 1.0e-12; /* tolerances                   */
    let mut abstol: sunrealtype = 1.0e-14;

    /* general problem variables */
    let mut retval: i32; /* reusable return flag          */
    let mut inner_arkode_mem: Option<ARKodeMem> = None; /* empty ARKode memory structure */
    let mut arkode_mem: Option<ARKodeMem> = None; /* empty ARKode memory structure */
    let mut inner_stepper: Option<MRIStepInnerStepper> = None; /* inner stepper */
    let mut B: Option<ARKodeButcherTable> = None; /* fast method Butcher table     */
    let mut C: Option<MRIStepCoupling> = None; /* slow coupling coefficients    */
    let mut Af: Option<SUNMatrix> = None; /* matrix for fast solver        */
    let mut LSf: Option<SUNLinearSolver> = None; /* fast linear solver object     */
    let mut As: Option<SUNMatrix> = None; /* matrix for slow solver        */
    let mut LSs: Option<SUNLinearSolver> = None; /* slow linear solver object     */
    let mut implicit_slow: sunbooleantype;
    let mut imex_slow: sunbooleantype = SUNFALSE;
    let mut t: sunrealtype; /* current/output time data      */
    let mut tout: sunrealtype;
    let hf: sunrealtype; /* fast time step                */
    /* temp data values `u`, `v`, `w` are bound inside the output loop */
    let mut nsts: i64 = 0; /* step stats                    */
    let mut nstf: i64 = 0;
    let mut nfse: i64 = 0; /* RHS stats                     */
    let mut nfsi: i64 = 0;
    let mut nffe: i64 = 0;
    let mut nffi: i64 = 0;
    let mut nnif: i64 = 0;
    let mut nncf: i64 = 0;
    let mut njef: i64 = 0;
    let mut nnis: i64 = 0;
    let mut nncs: i64 = 0;
    let mut njes: i64 = 0;
    let NEQ: sunindextype; /* number of equations           */
    let mut i: sunindextype; /* counter                       */

    /* Create the SUNDIALS context object for this simulation. */
    let mut ctx: Option<SUNContext> = None;
    retval = SUNContext_Create(SUN_COMM_NULL, &mut ctx);
    if check_retval_int(retval, "SUNContext_Create") != 0 {
        std::process::exit(1);
    }
    let ctx_h = ctx.clone().unwrap();

    /*
     * Initialization
     */

    /* Retrieve the command-line options: solve_type h  */
    if argc < 2 {
        print!("ERROR: enter solve_type and hs \n");
        std::process::exit(-1);
    }
    solve_type = atol(&argv[1]) as i32;
    /* C reads `argv[2]` unconditionally here; with argc == 2 that is the NULL
    terminator and `atof(NULL)` is undefined. The Rust index panics instead
    (accepted deviation class 5). */
    hs = atof(&argv[2]);

    /* Check arguments for validity */
    /*   0 <= solve_type <= 7       */
    /*   h > 0                      */

    if (solve_type < 0) || (solve_type > 7) {
        print!("ERROR: solve_type be an integer in [0,7] \n");
        std::process::exit(-1);
    }
    implicit_slow = SUNFALSE;
    if solve_type > 1 {
        implicit_slow = SUNTRUE;
    }
    if solve_type > 3 {
        imex_slow = SUNTRUE;
    }
    if hs <= ZERO {
        print!("ERROR: hs must be in positive\n");
        std::process::exit(-1);
    }
    hf = hs / m;
    NEQ = N * Nvar;

    /* Initial problem output */
    print!("\n1D Advection-Diffusion-Reaction (Brusselator) test problem:\n");
    print!("    time domain:  ({},{}]\n", fmt_g(T0, 6), fmt_g(Tf, 6));
    print!("    hs = {}\n", fmt_g(hs, 6));
    print!("    hf = {}\n", fmt_g(hf, 6));
    print!("    m  = {}\n", fmt_g(m, 6));
    print!("    N  = {},  NEQ = {}\n", N, NEQ);
    print!("    dx = {}\n", fmt_g(dx, 6));
    print!(
        "    problem parameters:  a = {},  b = {},  ep = {}\n",
        fmt_g(a, 6),
        fmt_g(b, 6),
        fmt_g(ep, 6)
    );
    print!(
        "    diffusion coefficients:  du = {},  dv = {},  dw = {}\n",
        fmt_g(du, 6),
        fmt_g(dv, 6),
        fmt_g(dw, 6)
    );
    print!(
        "    advection coefficients:  au = {},  av = {},  aw = {}\n",
        fmt_g(au, 6),
        fmt_g(av, 6),
        fmt_g(aw, 6)
    );

    match solve_type {
        0 => {
            /* reltol = SUNMAX(hs*hs*hs, 1e-10); */
            /* abstol = 1e-11; */
            print!("    solver: exp-3/dirk-3 (MIS / ESDIRK-3-3)\n\n");
            print!(
                "    reltol = {},  abstol = {}\n\n",
                fmt_e(reltol, 2),
                fmt_e(abstol, 2)
            );
        }
        1 => {
            reltol = SUNMAX(hs * hs * hs * hs * hs, 1e-14);
            abstol = 1e-14;
            print!("    solver: none/dirk-5 (no slow, default 5th order dirk fast)\n\n");
            print!(
                "    reltol = {},  abstol = {}\n\n",
                fmt_e(reltol, 2),
                fmt_e(abstol, 2)
            );
        }
        2 => {
            /* reltol = SUNMAX(hs*hs*hs, 1e-10); */
            /* abstol = 1e-11; */
            print!(
                "    solver: dirk-3/exp-3 (MRI-GARK-ESDIRK34a / ERK-3-3) -- solve decoupled\n\n"
            );
            print!(
                "    reltol = {},  abstol = {}\n\n",
                fmt_e(reltol, 2),
                fmt_e(abstol, 2)
            );
        }
        3 => {
            /* reltol = SUNMAX(hs*hs*hs, 1e-10); */
            /* abstol = 1e-11; */
            print!(
                "    solver: dirk-3/dirk-3 (MRI-GARK-ESDIRK34a / ESDIRK-3-3) -- solve decoupled\n\n"
            );
            print!(
                "    reltol = {},  abstol = {}\n\n",
                fmt_e(reltol, 2),
                fmt_e(abstol, 2)
            );
        }
        4 => {
            /* reltol = SUNMAX(hs*hs*hs, 1e-14); */
            /* abstol = 1e-14; */
            print!("    solver: ars343/exp-3 (IMEX-MRI3b / ERK-3-3) -- solve decoupled\n\n");
            print!(
                "    reltol = {},  abstol = {}\n\n",
                fmt_e(reltol, 2),
                fmt_e(abstol, 2)
            );
        }
        5 => {
            /* reltol = SUNMAX(hs*hs*hs, 1e-14); */
            /* abstol = 1e-14; */
            print!("    solver: ars343/dirk-3 (IMEX-MRI3b / ESDIRK-3-3) -- solve decoupled\n\n");
            print!(
                "    reltol = {},  abstol = {}\n\n",
                fmt_e(reltol, 2),
                fmt_e(abstol, 2)
            );
        }
        6 => {
            /* reltol = SUNMAX(hs*hs*hs*hs, 1e-14); */
            /* abstol = 1e-14; */
            print!("    solver: imexark4/exp-4 (IMEX-MRI4 / ERK-4-4) -- solve decoupled\n\n");
            print!(
                "    reltol = {},  abstol = {}\n\n",
                fmt_e(reltol, 2),
                fmt_e(abstol, 2)
            );
        }
        7 => {
            /* reltol = SUNMAX(hs*hs*hs*hs, 1e-14); */
            /* abstol = 1e-14; */
            print!(
                "    solver: imexark4/dirk-4 (IMEX-MRI4 / CASH(5,3,4)-DIRK ) -- solve decoupled\n\n"
            );
            print!(
                "    reltol = {},  abstol = {}\n\n",
                fmt_e(reltol, 2),
                fmt_e(abstol, 2)
            );
        }
        _ => {}
    }

    /* allocate udata structure (C: malloc; the Rust allocation cannot fail,
    so the `check_retval(..., "malloc", 2)` guard has no counterpart) */
    let udata = UserData {
        /* store the inputs in the UserData structure */
        N,
        a,
        b,
        du,
        dv,
        dw,
        au,
        av,
        aw,
        ep,
        pi,
        dx,
    };

    /* Create solution vector */
    let y = N_VNew_Serial(NEQ, &ctx_h); /* Create vector for solution */
    if check_retval_null(&y, "N_VNew_Serial") != 0 {
        std::process::exit(1);
    }
    let y = y.unwrap();

    /* Set initial condition */
    retval = SetIC(&y, &udata);
    if check_retval_int(retval, "SetIC") != 0 {
        std::process::exit(1);
    }

    /* Create vector masks  */
    let umask = N_VClone(&y);
    if check_retval_null(&umask, "N_VNew_Serial") != 0 {
        std::process::exit(1);
    }
    let umask = umask.unwrap();

    let vmask = N_VClone(&y);
    if check_retval_null(&vmask, "N_VNew_Serial") != 0 {
        std::process::exit(1);
    }
    let vmask = vmask.unwrap();

    let wmask = N_VClone(&y);
    if check_retval_null(&wmask, "N_VNew_Serial") != 0 {
        std::process::exit(1);
    }
    let wmask = wmask.unwrap();

    /* Set mask array values for each solution component */
    N_VConst(0.0, &umask);
    {
        let data = N_VGetArrayPointer(&umask);
        if check_retval_null(&data, "N_VGetArrayPointer") != 0 {
            std::process::exit(1);
        }
        let mut data = data.unwrap();
        i = 0;
        while i < N {
            data[IDX(i, 0) as usize] = 1.0;
            i += 1;
        }
    }

    N_VConst(0.0, &vmask);
    {
        let data = N_VGetArrayPointer(&vmask);
        if check_retval_null(&data, "N_VGetArrayPointer") != 0 {
            std::process::exit(1);
        }
        let mut data = data.unwrap();
        i = 0;
        while i < N {
            data[IDX(i, 1) as usize] = 1.0;
            i += 1;
        }
    }

    N_VConst(0.0, &wmask);
    {
        let data = N_VGetArrayPointer(&wmask);
        if check_retval_null(&data, "N_VGetArrayPointer") != 0 {
            std::process::exit(1);
        }
        let mut data = data.unwrap();
        i = 0;
        while i < N {
            data[IDX(i, 2) as usize] = 1.0;
            i += 1;
        }
    }

    /*
     * Create the fast integrator and set options
     */

    /* Initialize the fast integrator. Specify the fast right-hand side
    function in y'=fs(t,y)+ff(t,y) = fse(t,y)+fsi(t,y)+ff(t,y), the initial
    time T0, and the initial dependent variable vector y. */
    match solve_type {
        /* esdirk-3-3 fast solver */
        0 | 3 | 5 => {
            inner_arkode_mem = ARKStepCreate(None, Some(ff), T0, &y, &ctx_h);
            if check_retval_null(&inner_arkode_mem, "ARKStepCreate") != 0 {
                std::process::exit(1);
            }
            B = ARKodeButcherTable_Alloc(3, SUNFALSE);
            if check_retval_null(&B, "ARKodeButcherTable_Alloc") != 0 {
                std::process::exit(1);
            }
            let beta: sunrealtype = SUNRsqrt(3.0) / 6.0 + 0.5;
            let gamma: sunrealtype = (-ONE / 8.0) * (SUNRsqrt(3.0) + ONE);
            {
                let mut Bm = B.as_ref().unwrap().borrow_mut();
                Bm.A[1][0] = 4.0 * gamma + TWO * beta;
                Bm.A[1][1] = ONE - 4.0 * gamma - TWO * beta;
                Bm.A[2][0] = 0.5 - beta - gamma;
                Bm.A[2][1] = gamma;
                Bm.A[2][2] = beta;
                Bm.b[0] = ONE / 6.0;
                Bm.b[1] = ONE / 6.0;
                Bm.b[2] = TWO / 3.0;
                Bm.c[1] = ONE;
                Bm.c[2] = 0.5;
                Bm.q = 3;
            }
            retval = ARKStepSetTables(inner_arkode_mem.as_ref().unwrap(), 3, 0, B.as_ref(), None);
            if check_retval_int(retval, "ARKStepSetTables") != 0 {
                std::process::exit(1);
            }

            /* Initialize matrix and linear solver data structures */
            Af = SUNBandMatrix(NEQ, 4, 4, &ctx_h);
            if check_retval_null(&Af, "SUNBandMatrix") != 0 {
                std::process::exit(1);
            }

            LSf = SUNLinSol_Band(&y, Af.as_ref().unwrap(), &ctx_h);
            if check_retval_null(&LSf, "SUNLinSol_Band") != 0 {
                std::process::exit(1);
            }

            /* Specify fast tolerances */
            retval = ARKodeSStolerances(inner_arkode_mem.as_ref().unwrap(), reltol, abstol);
            if check_retval_int(retval, "ARKodeSStolerances") != 0 {
                std::process::exit(1);
            }

            /* Attach matrix and linear solver */
            retval = ARKodeSetLinearSolver(
                inner_arkode_mem.as_ref().unwrap(),
                LSf.as_ref().unwrap(),
                Af.as_ref(),
            );
            if check_retval_int(retval, "ARKodeSetLinearSolver") != 0 {
                std::process::exit(1);
            }

            /* Set max number of nonlinear iters */
            retval = ARKodeSetMaxNonlinIters(inner_arkode_mem.as_ref().unwrap(), 10);
            if check_retval_int(retval, "ARKodeSetMaxNonlinIters") != 0 {
                std::process::exit(1);
            }

            /* Set the Jacobian routine */
            retval = ARKodeSetJacFn(inner_arkode_mem.as_ref().unwrap(), Some(Jf));
            if check_retval_int(retval, "ARKodeSetJacFn") != 0 {
                std::process::exit(1);
            }
        }
        /* dirk 5th order fast solver (full problem) */
        1 => {
            inner_arkode_mem = ARKStepCreate(None, Some(f), T0, &y, &ctx_h);
            if check_retval_null(&inner_arkode_mem, "ARKStepCreate") != 0 {
                std::process::exit(1);
            }

            /* Set method order to use */
            retval = ARKodeSetOrder(inner_arkode_mem.as_ref().unwrap(), 5);
            if check_retval_int(retval, "ARKodeSetOrder") != 0 {
                std::process::exit(1);
            }

            /* Initialize matrix and linear solver data structures */
            Af = SUNBandMatrix(NEQ, 4, 4, &ctx_h);
            if check_retval_null(&Af, "SUNBandMatrix") != 0 {
                std::process::exit(1);
            }

            LSf = SUNLinSol_Band(&y, Af.as_ref().unwrap(), &ctx_h);
            if check_retval_null(&LSf, "SUNLinSol_Band") != 0 {
                std::process::exit(1);
            }

            /* Specify fast tolerances */
            retval = ARKodeSStolerances(inner_arkode_mem.as_ref().unwrap(), reltol, abstol);
            if check_retval_int(retval, "ARKodeSStolerances") != 0 {
                std::process::exit(1);
            }

            /* Attach matrix and linear solver */
            retval = ARKodeSetLinearSolver(
                inner_arkode_mem.as_ref().unwrap(),
                LSf.as_ref().unwrap(),
                Af.as_ref(),
            );
            if check_retval_int(retval, "ARKodeSetLinearSolver") != 0 {
                std::process::exit(1);
            }

            /* Set the Jacobian routine */
            retval = ARKodeSetJacFn(inner_arkode_mem.as_ref().unwrap(), Some(Jac));
            if check_retval_int(retval, "ARKodeSetJacFn") != 0 {
                std::process::exit(1);
            }
        }
        /* erk-3-3 fast solver */
        2 | 4 => {
            inner_arkode_mem = ARKStepCreate(Some(ff), None, T0, &y, &ctx_h);
            if check_retval_null(&inner_arkode_mem, "ARKStepCreate") != 0 {
                std::process::exit(1);
            }
            B = ARKodeButcherTable_Alloc(3, SUNTRUE);
            if check_retval_null(&B, "ARKodeButcherTable_Alloc") != 0 {
                std::process::exit(1);
            }
            {
                let mut Bm = B.as_ref().unwrap().borrow_mut();
                Bm.A[1][0] = 0.5;
                Bm.A[2][0] = -ONE;
                Bm.A[2][1] = TWO;
                Bm.b[0] = ONE / 6.0;
                Bm.b[1] = TWO / 3.0;
                Bm.b[2] = ONE / 6.0;
                Bm.d[1] = ONE;
                Bm.c[1] = 0.5;
                Bm.c[2] = ONE;
                Bm.q = 3;
                Bm.p = 2;
            }
            retval = ARKStepSetTables(inner_arkode_mem.as_ref().unwrap(), 3, 2, None, B.as_ref());
            if check_retval_int(retval, "ARKStepSetTables") != 0 {
                std::process::exit(1);
            }
        }
        /* erk-4-4 fast solver */
        6 => {
            inner_arkode_mem = ARKStepCreate(Some(ff), None, T0, &y, &ctx_h);
            if check_retval_null(&inner_arkode_mem, "ARKStepCreate") != 0 {
                std::process::exit(1);
            }
            B = ARKodeButcherTable_Alloc(4, SUNFALSE);
            if check_retval_null(&B, "ARKodeButcherTable_Alloc") != 0 {
                std::process::exit(1);
            }
            {
                let mut Bm = B.as_ref().unwrap().borrow_mut();
                Bm.A[1][0] = 0.5;
                Bm.A[2][1] = 0.5;
                Bm.A[3][2] = ONE;
                Bm.b[0] = ONE / 6.0;
                Bm.b[1] = ONE / 3.0;
                Bm.b[2] = ONE / 3.0;
                Bm.b[3] = ONE / 6.0;
                Bm.c[1] = 0.5;
                Bm.c[2] = 0.5;
                Bm.c[3] = ONE;
                Bm.q = 4;
            }
            retval = ARKStepSetTables(inner_arkode_mem.as_ref().unwrap(), 4, 0, None, B.as_ref());
            if check_retval_int(retval, "ARKStepSetTables") != 0 {
                std::process::exit(1);
            }
        }
        /* Cash(5,3,4)-SDIRK fast solver */
        7 => {
            inner_arkode_mem = ARKStepCreate(None, Some(ff), T0, &y, &ctx_h);
            if check_retval_null(&inner_arkode_mem, "ARKStepCreate") != 0 {
                std::process::exit(1);
            }

            /* Set fast method */
            retval = ARKStepSetTableNum(inner_arkode_mem.as_ref().unwrap(), ARKODE_CASH_5_3_4, -1);
            if check_retval_int(retval, "ARKStepSetTableNum") != 0 {
                std::process::exit(1);
            }

            /* Initialize matrix and linear solver data structures */
            Af = SUNBandMatrix(NEQ, 4, 4, &ctx_h);
            if check_retval_null(&Af, "SUNBandMatrix") != 0 {
                std::process::exit(1);
            }

            LSf = SUNLinSol_Band(&y, Af.as_ref().unwrap(), &ctx_h);
            if check_retval_null(&LSf, "SUNLinSol_Band") != 0 {
                std::process::exit(1);
            }

            /* Specify fast tolerances */
            retval = ARKodeSStolerances(inner_arkode_mem.as_ref().unwrap(), reltol, abstol);
            if check_retval_int(retval, "ARKodeSStolerances") != 0 {
                std::process::exit(1);
            }

            /* Attach matrix and linear solver */
            retval = ARKodeSetLinearSolver(
                inner_arkode_mem.as_ref().unwrap(),
                LSf.as_ref().unwrap(),
                Af.as_ref(),
            );
            if check_retval_int(retval, "ARKodeSetLinearSolver") != 0 {
                std::process::exit(1);
            }

            /* Set max number of nonlinear iters */
            retval = ARKodeSetMaxNonlinIters(inner_arkode_mem.as_ref().unwrap(), 10);
            if check_retval_int(retval, "ARKodeSetMaxNonlinIters") != 0 {
                std::process::exit(1);
            }

            /* Set the Jacobian routine */
            retval = ARKodeSetJacFn(inner_arkode_mem.as_ref().unwrap(), Some(Jf));
            if check_retval_int(retval, "ARKodeSetJacFn") != 0 {
                std::process::exit(1);
            }
        }
        _ => {}
    }
    let inner = inner_arkode_mem.clone().unwrap();

    /* Attach user data to fast integrator */
    retval = ARKodeSetUserData(&inner, Some(Box::new(udata.clone())));
    if check_retval_int(retval, "ARKodeSetUserData") != 0 {
        std::process::exit(1);
    }

    /* Set the fast step size */
    retval = ARKodeSetFixedStep(&inner, hf);
    if check_retval_int(retval, "ARKodeSetFixedStep") != 0 {
        std::process::exit(1);
    }

    /* Create inner stepper */
    retval = ARKodeCreateMRIStepInnerStepper(&inner, &mut inner_stepper);
    if check_retval_int(retval, "ARKodeCreateMRIStepInnerStepper") != 0 {
        std::process::exit(1);
    }
    let stepper = inner_stepper.clone().unwrap();

    /*
     * Create the slow integrator and set options
     */

    /* Initialize the slow integrator. Specify the slow right-hand side
    function in y'=fs(t,y)+ff(t,y) = fse(t,y)+fsi(t,y)+ff(t,y), the initial
    time T0, the initial dependent variable vector y, and the fast
    integrator. */
    match solve_type {
        /* use MIS outer integrator default for MRIStep */
        0 => {
            arkode_mem = MRIStepCreate(Some(fs), None, T0, &y, &stepper, &ctx_h);
            if check_retval_null(&arkode_mem, "MRIStepCreate") != 0 {
                std::process::exit(1);
            }
        }
        /* no slow dynamics (use ERK-2-2) */
        1 => {
            arkode_mem = MRIStepCreate(Some(f0), None, T0, &y, &stepper, &ctx_h);
            if check_retval_null(&arkode_mem, "MRIStepCreate") != 0 {
                std::process::exit(1);
            }
            B = ARKodeButcherTable_Alloc(2, SUNFALSE);
            if check_retval_null(&B, "ARKodeButcherTable_Alloc") != 0 {
                std::process::exit(1);
            }
            {
                let mut Bm = B.as_ref().unwrap().borrow_mut();
                Bm.A[1][0] = TWO / 3.0;
                Bm.b[0] = 0.25;
                Bm.b[1] = 0.75;
                Bm.c[1] = TWO / 3.0;
                Bm.q = 2;
            }
            C = MRIStepCoupling_MIStoMRI(B.as_ref(), 2, 0);
            if check_retval_null(&C, "MRIStepCoupling_MIStoMRI") != 0 {
                std::process::exit(1);
            }
            retval = MRIStepSetCoupling(arkode_mem.as_ref().unwrap(), C.as_ref().unwrap());
            if check_retval_int(retval, "MRIStepSetCoupling") != 0 {
                std::process::exit(1);
            }
        }
        /* MRI-GARK-ESDIRK34a, solve-decoupled slow solver */
        2 | 3 => {
            arkode_mem = MRIStepCreate(None, Some(fs), T0, &y, &stepper, &ctx_h);
            if check_retval_null(&arkode_mem, "MRIStepCreate") != 0 {
                std::process::exit(1);
            }

            C = MRIStepCoupling_LoadTable(ARKODE_MRI_GARK_ESDIRK34a);
            if check_retval_null(&C, "MRIStepCoupling_LoadTable") != 0 {
                std::process::exit(1);
            }

            retval = MRIStepSetCoupling(arkode_mem.as_ref().unwrap(), C.as_ref().unwrap());
            if check_retval_int(retval, "MRIStepSetCoupling") != 0 {
                std::process::exit(1);
            }

            /* Initialize matrix and linear solver data structures */
            As = SUNBandMatrix(NEQ, 4, 4, &ctx_h);
            if check_retval_null(&As, "SUNBandMatrix") != 0 {
                std::process::exit(1);
            }

            LSs = SUNLinSol_Band(&y, As.as_ref().unwrap(), &ctx_h);
            if check_retval_null(&LSs, "SUNLinSol_Band") != 0 {
                std::process::exit(1);
            }

            /* Specify tolerances */
            retval = ARKodeSStolerances(arkode_mem.as_ref().unwrap(), reltol, abstol);
            if check_retval_int(retval, "ARKodeSStolerances") != 0 {
                std::process::exit(1);
            }

            /* Attach matrix and linear solver */
            retval = ARKodeSetLinearSolver(
                arkode_mem.as_ref().unwrap(),
                LSs.as_ref().unwrap(),
                As.as_ref(),
            );
            if check_retval_int(retval, "ARKodeSetLinearSolver") != 0 {
                std::process::exit(1);
            }

            /* Set the Jacobian routine */
            retval = ARKodeSetJacFn(arkode_mem.as_ref().unwrap(), Some(Js));
            if check_retval_int(retval, "ARKodeSetJacFn") != 0 {
                std::process::exit(1);
            }
        }
        /* IMEX-MRI-GARK3b, solve-decoupled slow solver */
        4 | 5 => {
            arkode_mem = MRIStepCreate(Some(fse), Some(fsi), T0, &y, &stepper, &ctx_h);
            if check_retval_null(&arkode_mem, "MRIStepCreate") != 0 {
                std::process::exit(1);
            }

            C = MRIStepCoupling_LoadTable(ARKODE_IMEX_MRI_GARK3b);
            if check_retval_null(&C, "MRIStepCoupling_LoadTable") != 0 {
                std::process::exit(1);
            }

            retval = MRIStepSetCoupling(arkode_mem.as_ref().unwrap(), C.as_ref().unwrap());
            if check_retval_int(retval, "MRIStepSetCoupling") != 0 {
                std::process::exit(1);
            }

            /* Initialize matrix and linear solver data structures */
            As = SUNBandMatrix(NEQ, 4, 4, &ctx_h);
            if check_retval_null(&As, "SUNBandMatrix") != 0 {
                std::process::exit(1);
            }

            LSs = SUNLinSol_Band(&y, As.as_ref().unwrap(), &ctx_h);
            if check_retval_null(&LSs, "SUNLinSol_Band") != 0 {
                std::process::exit(1);
            }

            /* Specify tolerances */
            retval = ARKodeSStolerances(arkode_mem.as_ref().unwrap(), reltol, abstol);
            if check_retval_int(retval, "ARKodeSStolerances") != 0 {
                std::process::exit(1);
            }

            /* Attach matrix and linear solver */
            retval = ARKodeSetLinearSolver(
                arkode_mem.as_ref().unwrap(),
                LSs.as_ref().unwrap(),
                As.as_ref(),
            );
            if check_retval_int(retval, "ARKodeSetLinearSolver") != 0 {
                std::process::exit(1);
            }

            /* Set the Jacobian routine */
            retval = ARKodeSetJacFn(arkode_mem.as_ref().unwrap(), Some(Jsi));
            if check_retval_int(retval, "ARKodeSetJacFn") != 0 {
                std::process::exit(1);
            }
        }
        /* IMEX-MRI-GARK4, solve-decoupled slow solver */
        6 | 7 => {
            arkode_mem = MRIStepCreate(Some(fse), Some(fsi), T0, &y, &stepper, &ctx_h);
            if check_retval_null(&arkode_mem, "MRIStepCreate") != 0 {
                std::process::exit(1);
            }

            C = MRIStepCoupling_LoadTable(ARKODE_IMEX_MRI_GARK4);
            if check_retval_null(&C, "MRIStepCoupling_LoadTable") != 0 {
                std::process::exit(1);
            }

            retval = MRIStepSetCoupling(arkode_mem.as_ref().unwrap(), C.as_ref().unwrap());
            if check_retval_int(retval, "MRIStepSetCoupling") != 0 {
                std::process::exit(1);
            }

            /* Initialize matrix and linear solver data structures */
            As = SUNBandMatrix(NEQ, 4, 4, &ctx_h);
            if check_retval_null(&As, "SUNBandMatrix") != 0 {
                std::process::exit(1);
            }

            LSs = SUNLinSol_Band(&y, As.as_ref().unwrap(), &ctx_h);
            if check_retval_null(&LSs, "SUNLinSol_Band") != 0 {
                std::process::exit(1);
            }

            /* Specify tolerances */
            retval = ARKodeSStolerances(arkode_mem.as_ref().unwrap(), reltol, abstol);
            if check_retval_int(retval, "ARKodeSStolerances") != 0 {
                std::process::exit(1);
            }

            /* Attach matrix and linear solver */
            retval = ARKodeSetLinearSolver(
                arkode_mem.as_ref().unwrap(),
                LSs.as_ref().unwrap(),
                As.as_ref(),
            );
            if check_retval_int(retval, "ARKodeSetLinearSolver") != 0 {
                std::process::exit(1);
            }

            /* Set the Jacobian routine */
            retval = ARKodeSetJacFn(arkode_mem.as_ref().unwrap(), Some(Jsi));
            if check_retval_int(retval, "ARKodeSetJacFn") != 0 {
                std::process::exit(1);
            }
        }
        _ => {}
    }
    let ark = arkode_mem.clone().unwrap();

    /* Pass udata to user functions */
    retval = ARKodeSetUserData(&ark, Some(Box::new(udata.clone())));
    if check_retval_int(retval, "ARKodeSetUserData") != 0 {
        std::process::exit(1);
    }

    /* Set the slow step size */
    retval = ARKodeSetFixedStep(&ark, hs);
    if check_retval_int(retval, "ARKodeSetFixedStep") != 0 {
        std::process::exit(1);
    }

    /* Set maximum number of steps taken by solver */
    retval = ARKodeSetMaxNumSteps(&ark, 1000000);
    if check_retval_int(retval, "ARKodeSetMaxNumSteps") != 0 {
        std::process::exit(1);
    }

    /*
     * Integrate ODE
     */

    /* output spatial mesh to disk */
    let mut FID = File::create("bruss1D_mesh.txt").expect("bruss1D_mesh.txt");
    i = 0;
    while i < N {
        let _ = write!(FID, "  {}\n", fmt_e(udata.dx * i as sunrealtype, 16));
        i += 1;
    }
    drop(FID);

    /* Open output stream for results, access data arrays */
    let ofname = format!("bruss1D_u_{}_{}.txt", argv[1], argv[2]);
    let mut UFID = File::create(&ofname).expect("UFID");

    let ofname = format!("bruss1D_v_{}_{}.txt", argv[1], argv[2]);
    let mut VFID = File::create(&ofname).expect("VFID");

    let ofname = format!("bruss1D_w_{}_{}.txt", argv[1], argv[2]);
    let mut WFID = File::create(&ofname).expect("WFID");

    /* output initial condition to disk */
    {
        let data = N_VGetArrayPointer(&y);
        if check_retval_null(&data, "N_VGetArrayPointer") != 0 {
            std::process::exit(1);
        }
        let data = data.unwrap();

        i = 0;
        while i < N {
            let _ = write!(UFID, " {}", fmt_e(data[IDX(i, 0) as usize], 16));
            i += 1;
        }
        i = 0;
        while i < N {
            let _ = write!(VFID, " {}", fmt_e(data[IDX(i, 1) as usize], 16));
            i += 1;
        }
        i = 0;
        while i < N {
            let _ = write!(WFID, " {}", fmt_e(data[IDX(i, 2) as usize], 16));
            i += 1;
        }
        let _ = write!(UFID, "\n");
        let _ = write!(VFID, "\n");
        let _ = write!(WFID, "\n");
    }

    /* Main time-stepping loop: calls ARKodeEvolve to perform the integration,
    then prints results.  Stops when the final time has been reached */
    t = T0;
    tout = T0 + dTout;
    print!("        t      ||u||_rms   ||v||_rms   ||w||_rms\n");
    print!("   ----------------------------------------------\n");
    for _iout in 0..Nt {
        /* call integrator */
        retval = ARKodeEvolve(&ark, tout, &y, &mut t, ARK_NORMAL);
        if check_retval_int(retval, "ARKodeEvolve") != 0 {
            break;
        }

        /* access/print solution statistics */
        let u = N_VWL2Norm(&y, &umask);
        let u = SUNRsqrt(u * u / N as sunrealtype);
        let v = N_VWL2Norm(&y, &vmask);
        let v = SUNRsqrt(v * v / N as sunrealtype);
        let w = N_VWL2Norm(&y, &wmask);
        let w = SUNRsqrt(w * w / N as sunrealtype);
        print!(
            "  {}  {}  {}  {}\n",
            fmt_fw(t, 10, 6),
            fmt_fw(u, 10, 6),
            fmt_fw(v, 10, 6),
            fmt_fw(w, 10, 6)
        );

        /* successful solve: update output time */
        tout += dTout;
        tout = if tout > Tf { Tf } else { tout };

        /* output results to disk */
        {
            let data = N_VGetArrayPointer(&y).expect("N_VGetArrayPointer");
            i = 0;
            while i < N {
                let _ = write!(UFID, " {}", fmt_e(data[IDX(i, 0) as usize], 16));
                i += 1;
            }
            i = 0;
            while i < N {
                let _ = write!(VFID, " {}", fmt_e(data[IDX(i, 1) as usize], 16));
                i += 1;
            }
            i = 0;
            while i < N {
                let _ = write!(WFID, " {}", fmt_e(data[IDX(i, 2) as usize], 16));
                i += 1;
            }
        }
        let _ = write!(UFID, "\n");
        let _ = write!(VFID, "\n");
        let _ = write!(WFID, "\n");
    }
    print!("   ----------------------------------------------\n");
    drop(UFID);
    drop(VFID);
    drop(WFID);

    /*
     * Finalize
     */

    /* Get some slow integrator statistics */
    retval = ARKodeGetNumSteps(&ark, &mut nsts);
    check_retval_int(retval, "ARKodeGetNumSteps");
    retval = ARKodeGetNumRhsEvals(&ark, 0, &mut nfse);
    check_retval_int(retval, "ARKodeGetNumRhsEvals");
    retval = ARKodeGetNumRhsEvals(&ark, 1, &mut nfsi);
    check_retval_int(retval, "ARKodeGetNumRhsEvals");

    /* Get some fast integrator statistics */
    retval = ARKodeGetNumSteps(&inner, &mut nstf);
    check_retval_int(retval, "ARKodeGetNumSteps");
    retval = ARKodeGetNumRhsEvals(&inner, 0, &mut nffe);
    check_retval_int(retval, "ARKodeGetNumRhsEvals");
    retval = ARKodeGetNumRhsEvals(&inner, 1, &mut nffi);
    check_retval_int(retval, "ARKodeGetNumRhsEvals");

    /* Print some final statistics */
    print!("\nFinal Solver Statistics:\n");
    print!("   Slow Steps: nsts = {}\n", nsts);
    print!("   Fast Steps: nstf = {}\n", nstf);
    if imex_slow {
        if (solve_type == 0)
            || (solve_type == 1)
            || (solve_type == 3)
            || (solve_type == 5)
            || (solve_type == 7)
        {
            print!(
                "   Total RHS evals:  Fse = {}, Fsi = {},  Ff = {}\n",
                nfse, nfsi, nffi
            );
        } else {
            print!(
                "   Total RHS evals:  Fse = {}, Fsi = {},  Ff = {}\n",
                nfse, nfsi, nffe
            );
        }
    } else if implicit_slow {
        if (solve_type == 0)
            || (solve_type == 1)
            || (solve_type == 3)
            || (solve_type == 5)
            || (solve_type == 7)
        {
            print!("   Total RHS evals:  Fs = {},  Ff = {}\n", nfsi, nffi);
        } else {
            print!("   Total RHS evals:  Fs = {},  Ff = {}\n", nfsi, nffe);
        }
    } else if (solve_type == 0)
        || (solve_type == 1)
        || (solve_type == 3)
        || (solve_type == 5)
        || (solve_type == 7)
    {
        print!("   Total RHS evals:  Fs = {},  Ff = {}\n", nfse, nffi);
    } else {
        print!("   Total RHS evals:  Fs = {},  Ff = {}\n", nfse, nffe);
    }

    /* Get/print slow integrator decoupled implicit solver statistics */
    if solve_type > 1 {
        retval = ARKodeGetNonlinSolvStats(&ark, &mut nnis, &mut nncs);
        check_retval_int(retval, "ARKodeGetNonlinSolvStats");
        retval = ARKodeGetNumJacEvals(&ark, &mut njes);
        check_retval_int(retval, "ARKodeGetNumJacEvals");
        print!("   Slow Newton iters = {}\n", nnis);
        print!("   Slow Newton conv fails = {}\n", nncs);
        print!("   Slow Jacobian evals = {}\n", njes);
    }

    /* Get/print fast integrator implicit solver statistics */
    if (solve_type == 0)
        || (solve_type == 1)
        || (solve_type == 3)
        || (solve_type == 5)
        || (solve_type == 7)
    {
        retval = ARKodeGetNonlinSolvStats(&inner, &mut nnif, &mut nncf);
        check_retval_int(retval, "ARKodeGetNonlinSolvStats");
        retval = ARKodeGetNumJacEvals(&inner, &mut njef);
        check_retval_int(retval, "ARKodeGetNumJacEvals");
        print!("   Fast Newton iters = {}\n", nnif);
        print!("   Fast Newton conv fails = {}\n", nncf);
        print!("   Fast Jacobian evals = {}\n", njef);
    }

    /* Clean up and return with successful completion */
    drop(udata); /* Free user data             */
    ARKodeFree(&mut inner_arkode_mem); /* Free integrator memory     */
    let _ = MRIStepInnerStepper_Free(&mut inner_stepper); /* Free inner stepper */
    ARKodeFree(&mut arkode_mem); /* Free integrator memory     */
    ARKodeButcherTable_Free(B); /* Free Butcher table         */
    MRIStepCoupling_Free(C); /* Free coupling coefficients */
    if let Some(Af) = Af {
        SUNMatDestroy(Af); /* Free fast matrix           */
    }
    let _ = SUNLinSolFree(LSf); /* Free fast linear solver    */
    let _ = SUNLinSolFree(LSs); /* Free slow linear solver    */
    if let Some(As) = As {
        SUNMatDestroy(As); /* Free slow matrix           */
    }
    N_VDestroy(y); /* Free vectors               */
    N_VDestroy(umask);
    N_VDestroy(vmask);
    N_VDestroy(wmask);
    let _ = SUNContext_Free(&mut ctx);
}

/*------------------------------------
 * Functions called by the integrator
 *------------------------------------*/

/* ff routine to compute the fast portion of the ODE RHS. */
fn ff(_t: sunrealtype, y: &N_Vector, ydot: &N_Vector, user_data: &mut Option<Box<dyn Any>>) -> i32 {
    let udata = user_data
        .as_mut()
        .and_then(|bx| bx.downcast_mut::<UserData>())
        .expect("user_data is UserData"); /* access problem data */
    let N = udata.N; /* set variable shortcuts */
    let a = udata.a;
    let b = udata.b;
    let ep = udata.ep;
    let mut i: sunindextype;

    if check_retval_null(&N_VGetArrayPointer(y), "N_VGetArrayPointer") != 0 {
        return 1;
    }
    if check_retval_null(&N_VGetArrayPointer(ydot), "N_VGetArrayPointer") != 0 {
        return 1;
    }
    N_VConst(0.0, ydot); /* initialize ydot to zero */

    let Ydata = N_VGetArrayPointer(y).expect("N_VGetArrayPointer");
    let mut dYdata = N_VGetArrayPointer(ydot).expect("N_VGetArrayPointer");

    /* iterate over domain, computing all equations */
    i = 1;
    while i < N - 1 {
        /* set shortcuts */
        let u = Ydata[IDX(i, 0) as usize];
        let v = Ydata[IDX(i, 1) as usize];
        let w = Ydata[IDX(i, 2) as usize];

        /* Fill in ODE RHS for u */
        dYdata[IDX(i, 0) as usize] = a - (w + ONE) * u + v * u * u;

        /* Fill in ODE RHS for v */
        dYdata[IDX(i, 1) as usize] = w * u - v * u * u;

        /* Fill in ODE RHS for w */
        dYdata[IDX(i, 2) as usize] = (b - w) / ep - w * u;

        i += 1;
    }

    /* enforce stationary boundaries */
    dYdata[IDX(0, 2) as usize] = 0.0;
    dYdata[IDX(0, 1) as usize] = dYdata[IDX(0, 2) as usize];
    dYdata[IDX(0, 0) as usize] = dYdata[IDX(0, 1) as usize];
    dYdata[IDX(N - 1, 2) as usize] = 0.0;
    dYdata[IDX(N - 1, 1) as usize] = dYdata[IDX(N - 1, 2) as usize];
    dYdata[IDX(N - 1, 0) as usize] = dYdata[IDX(N - 1, 1) as usize];

    /* Return with success */
    0
}

/* fse routine to compute the slow-explicit portion of the ODE RHS function. */
fn fse(
    _t: sunrealtype,
    y: &N_Vector,
    ydot: &N_Vector,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    let udata = user_data
        .as_mut()
        .and_then(|bx| bx.downcast_mut::<UserData>())
        .expect("user_data is UserData"); /* access problem data */
    let N = udata.N; /* set variable shortcuts */
    let au = udata.au;
    let av = udata.av;
    let aw = udata.aw;
    let dx = udata.dx;
    let mut i: sunindextype;

    if check_retval_null(&N_VGetArrayPointer(y), "N_VGetArrayPointer") != 0 {
        return 1;
    }
    if check_retval_null(&N_VGetArrayPointer(ydot), "N_VGetArrayPointer") != 0 {
        return 1;
    }
    N_VConst(0.0, ydot); /* initialize ydot to zero */

    let Ydata = N_VGetArrayPointer(y).expect("N_VGetArrayPointer");
    let mut dYdata = N_VGetArrayPointer(ydot).expect("N_VGetArrayPointer");

    /* iterate over domain, computing all equations */
    let auconst = -au / 2.0 / dx;
    let avconst = -av / 2.0 / dx;
    let awconst = -aw / 2.0 / dx;
    i = 1;
    while i < N - 1 {
        /* set shortcuts */
        let ul = Ydata[IDX(i - 1, 0) as usize];
        let ur = Ydata[IDX(i + 1, 0) as usize];
        let vl = Ydata[IDX(i - 1, 1) as usize];
        let vr = Ydata[IDX(i + 1, 1) as usize];
        let wl = Ydata[IDX(i - 1, 2) as usize];
        let wr = Ydata[IDX(i + 1, 2) as usize];

        /* Fill in ODE RHS for u */
        dYdata[IDX(i, 0) as usize] = (ur - ul) * auconst;

        /* Fill in ODE RHS for v */
        dYdata[IDX(i, 1) as usize] = (vr - vl) * avconst;

        /* Fill in ODE RHS for w */
        dYdata[IDX(i, 2) as usize] = (wr - wl) * awconst;

        i += 1;
    }

    /* enforce stationary boundaries */
    dYdata[IDX(0, 2) as usize] = 0.0;
    dYdata[IDX(0, 1) as usize] = dYdata[IDX(0, 2) as usize];
    dYdata[IDX(0, 0) as usize] = dYdata[IDX(0, 1) as usize];
    dYdata[IDX(N - 1, 2) as usize] = 0.0;
    dYdata[IDX(N - 1, 1) as usize] = dYdata[IDX(N - 1, 2) as usize];
    dYdata[IDX(N - 1, 0) as usize] = dYdata[IDX(N - 1, 1) as usize];

    /* Return with success */
    0
}

/* fsi routine to compute the slow-implicit portion of the  ODE RHS. */
fn fsi(
    _t: sunrealtype,
    y: &N_Vector,
    ydot: &N_Vector,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    let udata = user_data
        .as_mut()
        .and_then(|bx| bx.downcast_mut::<UserData>())
        .expect("user_data is UserData"); /* access problem data */
    let N = udata.N; /* set variable shortcuts */
    let du = udata.du;
    let dv = udata.dv;
    let dw = udata.dw;
    let dx = udata.dx;
    let mut i: sunindextype;

    if check_retval_null(&N_VGetArrayPointer(y), "N_VGetArrayPointer") != 0 {
        return 1;
    }
    if check_retval_null(&N_VGetArrayPointer(ydot), "N_VGetArrayPointer") != 0 {
        return 1;
    }
    N_VConst(0.0, ydot); /* initialize ydot to zero */

    let Ydata = N_VGetArrayPointer(y).expect("N_VGetArrayPointer");
    let mut dYdata = N_VGetArrayPointer(ydot).expect("N_VGetArrayPointer");

    /* iterate over domain, computing all equations */
    let duconst = du / dx / dx;
    let dvconst = dv / dx / dx;
    let dwconst = dw / dx / dx;
    i = 1;
    while i < N - 1 {
        /* set shortcuts */
        let u = Ydata[IDX(i, 0) as usize];
        let ul = Ydata[IDX(i - 1, 0) as usize];
        let ur = Ydata[IDX(i + 1, 0) as usize];
        let v = Ydata[IDX(i, 1) as usize];
        let vl = Ydata[IDX(i - 1, 1) as usize];
        let vr = Ydata[IDX(i + 1, 1) as usize];
        let w = Ydata[IDX(i, 2) as usize];
        let wl = Ydata[IDX(i - 1, 2) as usize];
        let wr = Ydata[IDX(i + 1, 2) as usize];

        /* Fill in ODE RHS for u */
        dYdata[IDX(i, 0) as usize] = (ul - 2.0 * u + ur) * duconst;

        /* Fill in ODE RHS for v */
        dYdata[IDX(i, 1) as usize] = (vl - 2.0 * v + vr) * dvconst;

        /* Fill in ODE RHS for w */
        dYdata[IDX(i, 2) as usize] = (wl - 2.0 * w + wr) * dwconst;

        i += 1;
    }

    /* enforce stationary boundaries */
    dYdata[IDX(0, 2) as usize] = 0.0;
    dYdata[IDX(0, 1) as usize] = dYdata[IDX(0, 2) as usize];
    dYdata[IDX(0, 0) as usize] = dYdata[IDX(0, 1) as usize];
    dYdata[IDX(N - 1, 2) as usize] = 0.0;
    dYdata[IDX(N - 1, 1) as usize] = dYdata[IDX(N - 1, 2) as usize];
    dYdata[IDX(N - 1, 0) as usize] = dYdata[IDX(N - 1, 1) as usize];

    /* Return with success */
    0
}

/* fs routine to compute the slow portion of the ODE RHS. */
fn fs(_t: sunrealtype, y: &N_Vector, ydot: &N_Vector, user_data: &mut Option<Box<dyn Any>>) -> i32 {
    let udata = user_data
        .as_mut()
        .and_then(|bx| bx.downcast_mut::<UserData>())
        .expect("user_data is UserData"); /* access problem data */
    let N = udata.N; /* set variable shortcuts */
    let du = udata.du;
    let dv = udata.dv;
    let dw = udata.dw;
    let au = udata.au;
    let av = udata.av;
    let aw = udata.aw;
    let dx = udata.dx;
    let mut i: sunindextype;

    if check_retval_null(&N_VGetArrayPointer(y), "N_VGetArrayPointer") != 0 {
        return 1;
    }
    if check_retval_null(&N_VGetArrayPointer(ydot), "N_VGetArrayPointer") != 0 {
        return 1;
    }
    N_VConst(0.0, ydot); /* initialize ydot to zero */

    let Ydata = N_VGetArrayPointer(y).expect("N_VGetArrayPointer");
    let mut dYdata = N_VGetArrayPointer(ydot).expect("N_VGetArrayPointer");

    /* iterate over domain, computing all equations */
    let duconst = du / dx / dx;
    let dvconst = dv / dx / dx;
    let dwconst = dw / dx / dx;
    let auconst = -au / TWO / dx;
    let avconst = -av / TWO / dx;
    let awconst = -aw / TWO / dx;
    i = 1;
    while i < N - 1 {
        /* set shortcuts */
        let u = Ydata[IDX(i, 0) as usize];
        let ul = Ydata[IDX(i - 1, 0) as usize];
        let ur = Ydata[IDX(i + 1, 0) as usize];
        let v = Ydata[IDX(i, 1) as usize];
        let vl = Ydata[IDX(i - 1, 1) as usize];
        let vr = Ydata[IDX(i + 1, 1) as usize];
        let w = Ydata[IDX(i, 2) as usize];
        let wl = Ydata[IDX(i - 1, 2) as usize];
        let wr = Ydata[IDX(i + 1, 2) as usize];

        /* Fill in ODE RHS for u */
        dYdata[IDX(i, 0) as usize] = (ul - TWO * u + ur) * duconst + (ur - ul) * auconst;

        /* Fill in ODE RHS for v */
        dYdata[IDX(i, 1) as usize] = (vl - TWO * v + vr) * dvconst + (vr - vl) * avconst;

        /* Fill in ODE RHS for w */
        dYdata[IDX(i, 2) as usize] = (wl - TWO * w + wr) * dwconst + (wr - wl) * awconst;

        i += 1;
    }

    /* enforce stationary boundaries */
    dYdata[IDX(0, 2) as usize] = 0.0;
    dYdata[IDX(0, 1) as usize] = dYdata[IDX(0, 2) as usize];
    dYdata[IDX(0, 0) as usize] = dYdata[IDX(0, 1) as usize];
    dYdata[IDX(N - 1, 2) as usize] = 0.0;
    dYdata[IDX(N - 1, 1) as usize] = dYdata[IDX(N - 1, 2) as usize];
    dYdata[IDX(N - 1, 0) as usize] = dYdata[IDX(N - 1, 1) as usize];

    /* Return with success */
    0
}

/* f routine to compute the full ODE RHS function. */
fn f(_t: sunrealtype, y: &N_Vector, ydot: &N_Vector, user_data: &mut Option<Box<dyn Any>>) -> i32 {
    let udata = user_data
        .as_mut()
        .and_then(|bx| bx.downcast_mut::<UserData>())
        .expect("user_data is UserData"); /* access problem data */
    let N = udata.N; /* set variable shortcuts */
    let a = udata.a;
    let b = udata.b;
    let ep = udata.ep;
    let du = udata.du;
    let dv = udata.dv;
    let dw = udata.dw;
    let au = udata.au;
    let av = udata.av;
    let aw = udata.aw;
    let dx = udata.dx;
    let mut i: sunindextype;

    if check_retval_null(&N_VGetArrayPointer(y), "N_VGetArrayPointer") != 0 {
        return 1;
    }
    if check_retval_null(&N_VGetArrayPointer(ydot), "N_VGetArrayPointer") != 0 {
        return 1;
    }
    N_VConst(0.0, ydot); /* initialize ydot to zero */

    let Ydata = N_VGetArrayPointer(y).expect("N_VGetArrayPointer");
    let mut dYdata = N_VGetArrayPointer(ydot).expect("N_VGetArrayPointer");

    /* iterate over domain, computing all equations */
    let duconst = du / dx / dx;
    let dvconst = dv / dx / dx;
    let dwconst = dw / dx / dx;
    let auconst = -au / 2.0 / dx;
    let avconst = -av / 2.0 / dx;
    let awconst = -aw / 2.0 / dx;
    i = 1;
    while i < N - 1 {
        /* set shortcuts */
        let u = Ydata[IDX(i, 0) as usize];
        let ul = Ydata[IDX(i - 1, 0) as usize];
        let ur = Ydata[IDX(i + 1, 0) as usize];
        let v = Ydata[IDX(i, 1) as usize];
        let vl = Ydata[IDX(i - 1, 1) as usize];
        let vr = Ydata[IDX(i + 1, 1) as usize];
        let w = Ydata[IDX(i, 2) as usize];
        let wl = Ydata[IDX(i - 1, 2) as usize];
        let wr = Ydata[IDX(i + 1, 2) as usize];

        /* Fill in ODE RHS for u */
        dYdata[IDX(i, 0) as usize] =
            (ul - 2.0 * u + ur) * duconst + (ur - ul) * auconst + a - (w + 1.0) * u + v * u * u;

        /* Fill in ODE RHS for v */
        dYdata[IDX(i, 1) as usize] =
            (vl - 2.0 * v + vr) * dvconst + (vr - vl) * avconst + w * u - v * u * u;

        /* Fill in ODE RHS for w */
        dYdata[IDX(i, 2) as usize] =
            (wl - 2.0 * w + wr) * dwconst + (wr - wl) * awconst + (b - w) / ep - w * u;

        i += 1;
    }

    /* enforce stationary boundaries */
    dYdata[IDX(0, 2) as usize] = 0.0;
    dYdata[IDX(0, 1) as usize] = dYdata[IDX(0, 2) as usize];
    dYdata[IDX(0, 0) as usize] = dYdata[IDX(0, 1) as usize];
    dYdata[IDX(N - 1, 2) as usize] = 0.0;
    dYdata[IDX(N - 1, 1) as usize] = dYdata[IDX(N - 1, 2) as usize];
    dYdata[IDX(N - 1, 0) as usize] = dYdata[IDX(N - 1, 1) as usize];

    /* Return with success */
    0
}

/* Placeholder function of zeroes */
fn f0(
    _t: sunrealtype,
    _y: &N_Vector,
    ydot: &N_Vector,
    _user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    N_VConst(ZERO, ydot);
    0
}

/* Jf routine to compute Jacobian of the fast portion of the ODE RHS */
fn Jf(
    _t: sunrealtype,
    y: &N_Vector,
    _fy: &N_Vector,
    J: &SUNMatrix,
    user_data: &mut Option<Box<dyn Any>>,
    _tmp1: &N_Vector,
    _tmp2: &N_Vector,
    _tmp3: &N_Vector,
) -> i32 {
    let udata = user_data
        .as_mut()
        .and_then(|bx| bx.downcast_mut::<UserData>())
        .expect("user_data is UserData"); /* access problem data         */
    let _ = SUNMatZero(J); /* Initialize Jacobian to zero */

    /* Add in the Jacobian of the reaction terms matrix */
    ReactionJac(1.0, y, J, udata);

    /* Return with success */
    0
}

/* Jsi routine to compute the Jacobian of the slow-implicit portion of the ODE RHS. */
fn Jsi(
    _t: sunrealtype,
    _y: &N_Vector,
    _fy: &N_Vector,
    J: &SUNMatrix,
    user_data: &mut Option<Box<dyn Any>>,
    _tmp1: &N_Vector,
    _tmp2: &N_Vector,
    _tmp3: &N_Vector,
) -> i32 {
    let udata = user_data
        .as_mut()
        .and_then(|bx| bx.downcast_mut::<UserData>())
        .expect("user_data is UserData"); /* access problem data */
    let _ = SUNMatZero(J); /* Initialize Jacobian to zero */

    /* Fill in the Laplace matrix */
    LaplaceMatrix(1.0, J, udata);

    /* Return with success */
    0
}

/* Js routine to compute the Jacobian of the slow portion of ODE RHS. */
fn Js(
    _t: sunrealtype,
    _y: &N_Vector,
    _fy: &N_Vector,
    J: &SUNMatrix,
    user_data: &mut Option<Box<dyn Any>>,
    _tmp1: &N_Vector,
    _tmp2: &N_Vector,
    _tmp3: &N_Vector,
) -> i32 {
    let udata = user_data
        .as_mut()
        .and_then(|bx| bx.downcast_mut::<UserData>())
        .expect("user_data is UserData"); /* access problem data         */
    let _ = SUNMatZero(J); /* Initialize Jacobian to zero */

    /* Fill in the Laplace matrix */
    LaplaceMatrix(1.0, J, udata);

    /* Add Jacobian of the advection terms  */
    AdvectionJac(1.0, J, udata);

    /* Return with success */
    0
}

/* Jac routine to compute the Jacobian of the full ODE RHS. */
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
    let udata = user_data
        .as_mut()
        .and_then(|bx| bx.downcast_mut::<UserData>())
        .expect("user_data is UserData"); /* access problem data         */
    let _ = SUNMatZero(J); /* Initialize Jacobian to zero */

    /* Fill in the Laplace matrix */
    LaplaceMatrix(1.0, J, udata);

    /* Add Jacobian of the advection terms  */
    AdvectionJac(1.0, J, udata);

    /* Add in the Jacobian of the reaction terms matrix */
    ReactionJac(1.0, y, J, udata);

    /* Return with success */
    0
}

/*-------------------------------
 * Private helper functions
 *-------------------------------*/

/* Set the initial condition */
fn SetIC(y: &N_Vector, user_data: &UserData) -> i32 {
    let udata = user_data; /* access problem data    */
    let N = udata.N; /* set variable shortcuts */
    let a = udata.a;
    let b = udata.b;
    let dx = udata.dx;
    let pi = udata.pi;
    let mut i: sunindextype;

    /* Access data array from NVector y */
    let mut data = N_VGetArrayPointer(y).expect("N_VGetArrayPointer");

    /* Set initial conditions into y */
    i = 0;
    while i < N {
        data[IDX(i, 0) as usize] = a + 0.1 * (pi * i as sunrealtype * dx).sun_sin(); /* u */
        data[IDX(i, 1) as usize] = b / a + 0.1 * (pi * i as sunrealtype * dx).sun_sin(); /* v */
        data[IDX(i, 2) as usize] = b + 0.1 * (pi * i as sunrealtype * dx).sun_sin(); /* w */
        i += 1;
    }

    /* Return  with success */
    0
}

/* Routine to compute the Jacobian matrix from fse(t,y), scaled by the factor c.
We add the result into Jac and do not erase what was already there */
fn AdvectionJac(c: sunrealtype, Jac: &SUNMatrix, udata: &UserData) -> i32 {
    /* Set shortcuts */
    let N = udata.N;
    let dx = udata.dx;
    let au = udata.au;
    let av = udata.av;
    let aw = udata.aw;
    let mut i: sunindextype;
    let auconst = -au / TWO / dx;
    let avconst = -av / TWO / dx;
    let awconst = -aw / TWO / dx;

    /* iterate over intervals, filling in Jacobian of (L*y) using SM_ELEMENT_B
    macro (see sunmatrix_band.h) */
    i = 1;
    while i < N - 1 {
        SM_ELEMENT_B_add(Jac, IDX(i, 0), IDX(i - 1, 0), -c * auconst);
        SM_ELEMENT_B_add(Jac, IDX(i, 1), IDX(i - 1, 1), -c * avconst);
        SM_ELEMENT_B_add(Jac, IDX(i, 2), IDX(i - 1, 2), -c * awconst);
        SM_ELEMENT_B_add(Jac, IDX(i, 0), IDX(i + 1, 0), c * auconst);
        SM_ELEMENT_B_add(Jac, IDX(i, 1), IDX(i + 1, 1), c * avconst);
        SM_ELEMENT_B_add(Jac, IDX(i, 2), IDX(i + 1, 2), c * awconst);
        i += 1;
    }

    /* Return with success */
    0
}

/* Routine to compute the stiffness matrix from (L*y), scaled by the factor c.
We add the result into Jac and do not erase what was already there */
fn LaplaceMatrix(c: sunrealtype, Jac: &SUNMatrix, udata: &UserData) -> i32 {
    let N = udata.N; /* set shortcuts */
    let dx = udata.dx;
    let mut i: sunindextype;

    /* iterate over intervals, filling in Jacobian of (L*y) using SM_ELEMENT_B
    macro (see sunmatrix_band.h) */
    i = 1;
    while i < N - 1 {
        SM_ELEMENT_B_add(Jac, IDX(i, 0), IDX(i - 1, 0), c * udata.du / dx / dx);
        SM_ELEMENT_B_add(Jac, IDX(i, 1), IDX(i - 1, 1), c * udata.dv / dx / dx);
        SM_ELEMENT_B_add(Jac, IDX(i, 2), IDX(i - 1, 2), c * udata.dw / dx / dx);
        SM_ELEMENT_B_add(Jac, IDX(i, 0), IDX(i, 0), -c * 2.0 * udata.du / dx / dx);
        SM_ELEMENT_B_add(Jac, IDX(i, 1), IDX(i, 1), -c * 2.0 * udata.dv / dx / dx);
        SM_ELEMENT_B_add(Jac, IDX(i, 2), IDX(i, 2), -c * 2.0 * udata.dw / dx / dx);
        SM_ELEMENT_B_add(Jac, IDX(i, 0), IDX(i + 1, 0), c * udata.du / dx / dx);
        SM_ELEMENT_B_add(Jac, IDX(i, 1), IDX(i + 1, 1), c * udata.dv / dx / dx);
        SM_ELEMENT_B_add(Jac, IDX(i, 2), IDX(i + 1, 2), c * udata.dw / dx / dx);
        i += 1;
    }

    /* Return with success */
    0
}

/* Routine to compute the Jacobian matrix from R(y), scaled by the factor c.
We add the result into Jac and do not erase what was already there */
fn ReactionJac(c: sunrealtype, y: &N_Vector, Jac: &SUNMatrix, udata: &UserData) -> i32 {
    let N = udata.N; /* set shortcuts */
    let ep = udata.ep;
    let mut i: sunindextype;
    let Ydata = N_VGetArrayPointer(y); /* access solution array */
    if check_retval_null(&Ydata, "N_VGetArrayPointer") != 0 {
        return 1;
    }
    let Ydata = Ydata.unwrap();

    /* iterate over nodes, filling in Jacobian of reaction terms */
    i = 1;
    while i < N - 1 {
        let u = Ydata[IDX(i, 0) as usize]; /* set nodal value shortcuts */
        let v = Ydata[IDX(i, 1) as usize];
        let w = Ydata[IDX(i, 2) as usize];

        /* all vars wrt u */
        SM_ELEMENT_B_add(Jac, IDX(i, 0), IDX(i, 0), c * (TWO * u * v - (w + ONE)));
        SM_ELEMENT_B_add(Jac, IDX(i, 1), IDX(i, 0), c * (w - TWO * u * v));
        SM_ELEMENT_B_add(Jac, IDX(i, 2), IDX(i, 0), c * (-w));

        /* all vars wrt v */
        SM_ELEMENT_B_add(Jac, IDX(i, 0), IDX(i, 1), c * (u * u));
        SM_ELEMENT_B_add(Jac, IDX(i, 1), IDX(i, 1), c * (-u * u));

        /* all vars wrt w */
        SM_ELEMENT_B_add(Jac, IDX(i, 0), IDX(i, 2), c * (-u));
        SM_ELEMENT_B_add(Jac, IDX(i, 1), IDX(i, 2), c * (u));
        SM_ELEMENT_B_add(Jac, IDX(i, 2), IDX(i, 2), c * (-ONE / ep - u));

        i += 1;
    }

    /* Return with success */
    0
}

/* Check function return value...
    opt == 0 means SUNDIALS function allocates memory so check if
             returned NULL pointer
    opt == 1 means SUNDIALS function returns a flag so check if
             flag >= 0
    opt == 2 means function allocates memory so check if returned
             NULL pointer

   The C void-pointer/opt polymorphism splits into two typed helpers with
   identical message text:
     check_retval_null = opt == 0
     check_retval_int  = opt == 1
   (opt == 2 guards a `malloc` that cannot fail in the Rust port.)
*/

fn check_retval_null<T>(returnvalue: &Option<T>, funcname: &str) -> i32 {
    /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
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
    /* Check if flag < 0 */
    if retval < 0 {
        eprint!(
            "\nSUNDIALS_ERROR: {}() failed with flag = {}\n\n",
            funcname, retval
        );
        return 1;
    }
    0
}

/*---- end of file ----*/
