/* -----------------------------------------------------------------------------
 * Programmer(s): David J. Gardner @ LLNL
 * -----------------------------------------------------------------------------
 * Rust port of examples/arkode/C_serial/ark_conserved_exp_entropy_erk.c
 * -----------------------------------------------------------------------------
 * This example problem is adapted from:
 *
 * H. Ranocha, M. Sayyari, L. Dalcin, M. Parsani, and D.I. Ketcheson,
 * "Relaxation Runge-Kutta Methods: Fully-Discrete Explicit Entropy-Stable
 * Schemes for the Compressible Euler and Navier-Stokes Equations," SIAM Journal
 * on Scientific Computing, 42(2), 2020, https://doi.org/10.1137/19M1263480.
 * -----------------------------------------------------------------------------
 * This example evolves system
 *
 *   du/dt = -exp(v)
 *   dv/dt =  exp(u)
 *
 * for t in the interval [0, 5] with the initial condition
 *
 *   u(0) = 1.0
 *   v(0) = 0.5
 *
 * The system has the analytic solution
 *
 *   u = log(e + e^(3/2)) - log(b)
 *   v = log(a * e^(a * t)) - log(b)
 *
 * where log is the natural logarithm, a = sqrt(e) + e, and
 * b = sqrt(e) + e^(a * t).
 *
 * The conserved exponential entropy for the system is given by
 * ent(u,v) = exp(u) + exp(v) with the Jacobian
 * ent'(u,v) = [ de/du de/dv ]^T = [ exp(u) exp(v) ]^T.
 *
 * The problem is advanced in time with an explicit relaxed
 * Runge-Kutta method from ERKStep to ensure conservation of the entropy.
 * ---------------------------------------------------------------------------*/
#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

use arkode_rs::prelude::*;

use std::any::Any;
use std::fs::File;
use std::io::Write;

/* Value of the natural number e */
const EVAL: sunrealtype = 2.718281828459045235360287471352662497757247093699959574966;

/* Convince macros for calling precision-specific math functions */
fn EXP(x: sunrealtype) -> sunrealtype {
    x.sun_exp()
}

fn SQRT(x: sunrealtype) -> sunrealtype {
    x.sqrt()
}

fn LOG(x: sunrealtype) -> sunrealtype {
    x.sun_ln()
}

/* C `atoi` (strtol semantics): longest valid leading integer, 0 otherwise */
fn atoi(s: &str) -> i32 {
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
    t[..i].parse::<i32>().unwrap_or(0)
}

/* ------------ *
 * Main Program *
 * ------------ */

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let argc = argv.len();

    /* Error-checking flag */
    let mut flag: i32;

    /* Initial and final times */
    let t0: sunrealtype = 0.0;
    let tf: sunrealtype = 5.0;

    /* Relative and absolute tolerances */
    let reltol: sunrealtype = 1.0e-6;
    let abstol: sunrealtype = 1.0e-10;

    /* Initial entropy value */
    let mut ent0: sunrealtype = 0.0;

    /* ARKODE statistics */
    let mut nst: i64 = 0;
    let mut nst_a: i64 = 0;
    let mut nfe: i64 = 0;
    let mut nrf: i64 = 0;
    let mut nrbf: i64 = 0;
    let mut nre: i64 = 0;
    let mut nrje: i64 = 0;
    let mut nrnlsi: i64 = 0;
    let mut nrnlsf: i64 = 0;
    let mut netf: i64 = 0;

    /* Output time */
    let mut t: sunrealtype;

    /* Command line options */
    let mut relax: i32 = 1; /* enable relaxation */
    let mut fixed_h: sunrealtype = 0.0; /* adaptive stepping */

    /* -------------------- *
     * Output Problem Setup *
     * -------------------- */

    if argc > 1 {
        relax = atoi(&argv[1]);
    }
    if argc > 2 {
        fixed_h = SUNStrToReal(&argv[2]);
    }

    print!("\nConserved Exponential Entropy problem:\n");
    print!("   method     = ERK\n");
    print!("   reltol     = {}\n", fmt_e(reltol, 1));
    print!("   abstol     = {}\n", fmt_e(abstol, 1));
    if fixed_h > 0.0 {
        print!("   fixed h    = {}\n", fmt_e(fixed_h, 1));
    }
    if relax != 0 {
        print!("   relaxation = ON\n");
    } else {
        print!("   relaxation = OFF\n");
    }
    print!("\n");

    /* ------------ *
     * Setup ARKODE *
     * ------------ */

    /* Create the SUNDIALS context object for this simulation */
    let mut ctx: Option<SUNContext> = None;
    flag = SUNContext_Create(SUN_COMM_NULL, &mut ctx);
    if check_flag(flag, "SUNContext_Create") {
        std::process::exit(1);
    }
    let ctx = ctx.unwrap();

    /* Create serial vector and set the initial condition values */
    let y = N_VNew_Serial(2, &ctx);
    if check_ptr(&y, "N_VNew_Serial") {
        std::process::exit(1);
    }
    let y = y.unwrap();

    {
        let ydata = N_VGetArrayPointer(&y);
        if check_ptr(&ydata, "N_VGetArrayPointer") {
            std::process::exit(1);
        }
        let mut ydata = ydata.unwrap();

        ydata[0] = 1.0;
        ydata[1] = 0.5;
    }

    let ytrue = N_VClone(&y);
    if check_ptr(&ytrue, "N_VClone") {
        std::process::exit(1);
    }
    let ytrue = ytrue.unwrap();

    {
        let ytdata = N_VGetArrayPointer(&ytrue);
        if check_ptr(&ytdata, "N_VGetArrayPointer") {
            std::process::exit(1);
        }
    }

    /* Initialize ERKStep */
    let arkode_mem = ERKStepCreate(f, t0, &y, &ctx);
    if check_ptr(&arkode_mem, "ERKStepCreate") {
        std::process::exit(1);
    }
    let arkode_mem = arkode_mem.unwrap();

    /* Set order */
    flag = ARKodeSetOrder(&arkode_mem, 2);
    if check_flag(flag, "ARKodeSetOrder") {
        std::process::exit(1);
    }

    /* Specify tolerances */
    flag = ARKodeSStolerances(&arkode_mem, reltol, abstol);
    if check_flag(flag, "ARKodeSStolerances") {
        std::process::exit(1);
    }

    if relax != 0 {
        /* Enable relaxation methods */
        flag = ARKodeSetRelaxFn(&arkode_mem, Some(Ent), Some(JacEnt));
        if check_flag(flag, "ARKodeSetRelaxFn") {
            std::process::exit(1);
        }
    }

    if fixed_h > 0.0 {
        flag = ARKodeSetFixedStep(&arkode_mem, fixed_h);
        if check_flag(flag, "ARKodeSetFixedStep") {
            std::process::exit(1);
        }
    }

    /* Open output stream for results, output comment line */
    let mut UFID = File::create("ark_conserved_exp_entropy_erk.txt").ok();
    if let Some(fp) = UFID.as_mut() {
        let _ = write!(fp, "# vars: t u v entropy u_err v_err entropy_error\n");
    }

    /* --------------- *
     * Advance in Time *
     * --------------- */

    /* Initial time */
    t = t0;

    /* Output the initial condition and entropy */
    flag = Ent(&y, &mut ent0, &mut None);
    if check_flag(flag, "Ent") {
        std::process::exit(1);
    }

    {
        let ydata = N_VGetArrayPointer(&y).expect("vector data");
        if let Some(fp) = UFID.as_mut() {
            let _ = write!(
                fp,
                "{} {} {} {} {} {} {}\n",
                fmt_ew(t0, 23, 16),
                fmt_ew(ydata[0], 23, 16),
                fmt_ew(ydata[1], 23, 16),
                fmt_ew(ent0, 23, 16),
                fmt_ew(0.0, 23, 16),
                fmt_ew(0.0, 23, 16),
                fmt_ew(0.0, 23, 16)
            );
        }

        print!(" step   t              u              v              e              delta e\n");
        print!(" -------------------------------------------------------------------------------\n");
        print!(
            "{:5} {} {} {} {} {}\n",
            0,
            fmt_ew(t, 14, 6),
            fmt_ew(ydata[0], 14, 6),
            fmt_ew(ydata[1], 14, 6),
            fmt_ew(ent0, 14, 6),
            fmt_ew(0.0, 14, 6)
        );
    }

    while t < tf {
        /* Evolve in time */
        flag = ARKodeEvolve(&arkode_mem, tf, &y, &mut t, ARK_ONE_STEP);
        if check_flag(flag, "ARKodeEvolve") {
            break;
        }

        /* Output solution and errors */
        let mut ent: sunrealtype = 0.0;
        flag = Ent(&y, &mut ent, &mut None);
        if check_flag(flag, "Ent") {
            std::process::exit(1);
        }

        flag = ans(t, &ytrue);
        if check_flag(flag, "ans") {
            std::process::exit(1);
        }

        let (u, v, ut, vt) = {
            let ydata = N_VGetArrayPointer(&y).expect("vector data");
            let ytdata = N_VGetArrayPointer(&ytrue).expect("vector data");
            (ydata[0], ydata[1], ytdata[0], ytdata[1])
        };

        let ent_err = ent - ent0;
        let u_err = u - ut;
        let v_err = v - vt;

        /* Output to the screen periodically */
        flag = ARKodeGetNumSteps(&arkode_mem, &mut nst);
        let _ = check_flag(flag, "ARKodeGetNumSteps");

        if nst % 40 == 0 {
            print!(
                "{:5} {} {} {} {} {}\n",
                nst,
                fmt_ew(t, 14, 6),
                fmt_ew(u, 14, 6),
                fmt_ew(v, 14, 6),
                fmt_ew(ent, 14, 6),
                fmt_ew(ent_err, 14, 6)
            );
        }

        /* Write all steps to file */
        if let Some(fp) = UFID.as_mut() {
            let _ = write!(
                fp,
                "{} {} {} {} {} {} {}\n",
                fmt_ew(t, 23, 16),
                fmt_ew(u, 23, 16),
                fmt_ew(v, 23, 16),
                fmt_ew(ent, 23, 16),
                fmt_ew(u_err, 23, 16),
                fmt_ew(v_err, 23, 16),
                fmt_ew(ent_err, 23, 16)
            );
        }
    }

    print!(" -------------------------------------------------------------------------------\n");
    drop(UFID);

    /* ------------ *
     * Output Stats *
     * ------------ */

    /* Get final statistics on how the solve progressed */
    flag = ARKodeGetNumSteps(&arkode_mem, &mut nst);
    let _ = check_flag(flag, "ARKodeGetNumSteps");

    flag = ARKodeGetNumStepAttempts(&arkode_mem, &mut nst_a);
    let _ = check_flag(flag, "ARKodeGetNumStepAttempts");

    flag = ARKodeGetNumErrTestFails(&arkode_mem, &mut netf);
    let _ = check_flag(flag, "ARKodeGetNumErrTestFails");

    flag = ARKodeGetNumRhsEvals(&arkode_mem, 0, &mut nfe);
    let _ = check_flag(flag, "ARKodeGetNumRhsEvals");

    print!("\nFinal Solver Statistics:\n");
    print!(
        "   Internal solver steps = {} (attempted = {})\n",
        nst, nst_a
    );
    print!("   Total number of error test failures = {}\n", netf);
    print!("   Total RHS evals = {}\n", nfe);

    if relax != 0 {
        flag = ARKodeGetNumRelaxFnEvals(&arkode_mem, &mut nre);
        let _ = check_flag(flag, "ARKodeGetNumRelaxFnEvals");

        flag = ARKodeGetNumRelaxJacEvals(&arkode_mem, &mut nrje);
        let _ = check_flag(flag, "ARKodeGetNumRelaxJacEvals");

        flag = ARKodeGetNumRelaxFails(&arkode_mem, &mut nrf);
        let _ = check_flag(flag, "ARKodeGetNumRelaxFails");

        flag = ARKodeGetNumRelaxBoundFails(&arkode_mem, &mut nrbf);
        let _ = check_flag(flag, "ARKodeGetNumRelaxBoundFails");

        flag = ARKodeGetNumRelaxSolveFails(&arkode_mem, &mut nrnlsf);
        let _ = check_flag(flag, "ARKodeGetNumRelaxSolveFails");

        flag = ARKodeGetNumRelaxSolveIters(&arkode_mem, &mut nrnlsi);
        let _ = check_flag(flag, "ARKodeGetNumRelaxSolveIters");

        print!("   Total Relaxation Fn evals    = {}\n", nre);
        print!("   Total Relaxation Jac evals   = {}\n", nrje);
        print!("   Total Relaxation fails       = {}\n", nrf);
        print!("   Total Relaxation bound fails = {}\n", nrbf);
        print!("   Total Relaxation NLS fails   = {}\n", nrnlsf);
        print!("   Total Relaxation NLS iters   = {}\n", nrnlsi);
    }
    print!("\n");

    /* -------- *
     * Clean up *
     * -------- */

    /* Free ARKode integrator and SUNDIALS objects */
    let mut arkode_mem = Some(arkode_mem);
    ARKodeFree(&mut arkode_mem);
    N_VDestroy(y);
    N_VDestroy(ytrue);
    let mut ctx = Some(ctx);
    let _ = SUNContext_Free(&mut ctx);

    std::process::exit(flag);
}

/* ----------------------- *
 * User-supplied functions *
 * ----------------------- */

/* ODE RHS function f(t,y). */
fn f(
    _t: sunrealtype,
    y: &N_Vector,
    ydot: &N_Vector,
    _user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    let ydata = N_VGetArrayPointer(y).expect("vector data");
    let mut fdata = N_VGetArrayPointer(ydot).expect("vector data");

    fdata[0] = -EXP(ydata[1]);
    fdata[1] = EXP(ydata[0]);

    0
}

/* Entropy function e(y) */
fn Ent(y: &N_Vector, e: &mut sunrealtype, _user_data: &mut Option<Box<dyn Any>>) -> i32 {
    let ydata = N_VGetArrayPointer(y).expect("vector data");

    *e = EXP(ydata[0]) + EXP(ydata[1]);

    0
}

/* Entropy function Jacobian Je(y) = de/dy */
fn JacEnt(y: &N_Vector, J: &N_Vector, _user_data: &mut Option<Box<dyn Any>>) -> i32 {
    let ydata = N_VGetArrayPointer(y).expect("vector data");
    let mut jdata = N_VGetArrayPointer(J).expect("vector data");

    jdata[0] = EXP(ydata[0]);
    jdata[1] = EXP(ydata[1]);

    0
}

/* ----------------- *
 * Utility functions *
 * ----------------- */

/* Analytic solution */
fn ans(t: sunrealtype, y: &N_Vector) -> i32 {
    let a: sunrealtype;
    let b: sunrealtype;
    let mut ydata = N_VGetArrayPointer(y).expect("vector data");

    a = SQRT(EVAL) + EVAL;
    b = SQRT(EVAL) + EXP(a * t);

    ydata[0] = LOG(EVAL + EXP(1.5)) - LOG(b);
    ydata[1] = LOG(a * EXP(a * t)) - LOG(b);

    0
}

/* Check for an unrecoverable (negative) return flag from a SUNDIALS function */
fn check_flag(flag: i32, funcname: &str) -> bool {
    if flag < 0 {
        eprint!("ERROR: {}() returned {}\n", funcname, flag);
        return true;
    }
    false
}

/* Check if a function returned a NULL pointer */
fn check_ptr<T>(ptr: &Option<T>, funcname: &str) -> bool {
    if ptr.is_none() {
        eprint!("ERROR: {}() returned NULL\n", funcname);
        return true;
    }
    false
}
