/* -----------------------------------------------------------------------------
 * Programmer(s): David J. Gardner @ LLNL
 * -----------------------------------------------------------------------------
 * Rust port of examples/arkode/C_serial/ark_dissipated_exp_entropy.c
 * -----------------------------------------------------------------------------
 * This example problem is adapted from:
 *
 * H. Ranocha, M. Sayyari, L. Dalcin, M. Parsani, and D.I. Ketcheson,
 * "Relaxation Runge-Kutta Methods: Fully-Discrete Explicit Entropy-Stable
 * Schemes for the Compressible Euler and Navier-Stokes Equations," SIAM Journal
 * on Scientific Computing, 42(2), 2020, https://doi.org/10.1137/19M1263480.
 * -----------------------------------------------------------------------------
 * This example evolves the equation du/dt = -exp(u) for t in the interval
 * [0, 5] with the initial condition u(0) = 0.5. The equation has the analytic
 * solution u(t) = -log(e^{-0.5} + t) and the dissipated exponential entropy is
 * given by ent(u) = exp(u) with Jacobian ent'(u) = de/du = exp(u).
 *
 * The problem is advanced in time with an explicit or implicit relaxed
 * Runge-Kutta method to ensure dissipation of the entropy.
 * ---------------------------------------------------------------------------*/
#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

use arkode_rs::prelude::*;

use std::any::Any;
use std::fs::File;
use std::io::Write;

/* Convince macros for calling precision-specific math functions */
fn EXP(x: sunrealtype) -> sunrealtype {
    x.sun_exp()
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

    /* Initial, current, and change in entropy value */
    let mut ent0: sunrealtype = 0.0;

    /* ARKODE statistics */
    let mut nst: i64 = 0;
    let mut nst_a: i64 = 0;
    let mut nfe: i64 = 0;
    let mut nfi: i64 = 0;
    let mut nrf: i64 = 0;
    let mut nrbf: i64 = 0;
    let mut nre: i64 = 0;
    let mut nrje: i64 = 0;
    let mut nrnlsi: i64 = 0;
    let mut nrnlsf: i64 = 0;
    let mut nsetups: i64 = 0;
    let mut nje: i64 = 0;
    let mut nfeLS: i64 = 0;
    let mut nni: i64 = 0;
    let mut ncfn: i64 = 0;
    let mut netf: i64 = 0;

    /* Output time */
    let mut t: sunrealtype;

    /* Command line options */
    let mut relax: i32 = 1; /* enable relaxation */
    let mut implicit: i32 = 1; /* implicit          */
    let mut fixed_h: sunrealtype = 0.0; /* adaptive stepping */

    /* -------------------- *
     * Output Problem Setup *
     * -------------------- */

    if argc > 1 {
        relax = atoi(&argv[1]);
    }
    if argc > 2 {
        implicit = atoi(&argv[2]);
    }
    if argc > 3 {
        fixed_h = SUNStrToReal(&argv[3]);
    }

    print!("\nDissipated Exponential Entropy problem:\n");
    if implicit != 0 {
        print!("   method     = DIRK\n");
    } else {
        print!("   method     = ERK\n");
    }
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
    let y = N_VNew_Serial(1, &ctx);
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

        ydata[0] = 0.5;
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

    /* Initialize ARKStep */
    let arkode_mem = if implicit != 0 {
        ARKStepCreate(None, Some(f), t0, &y, &ctx)
    } else {
        ARKStepCreate(Some(f), None, t0, &y, &ctx)
    };
    if check_ptr(&arkode_mem, "ARKStepCreate") {
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

    /* SUNDIALS matrix and linear solver objects */
    let mut A: Option<SUNMatrix> = None;
    let mut LS: Option<SUNLinearSolver> = None;

    if implicit != 0 {
        /* Create dense matrix and linear solver */
        let A_new = SUNDenseMatrix(1, 1, &ctx);
        if check_ptr(&A_new, "SUNDenseMatrix") {
            std::process::exit(1);
        }
        A = A_new;

        let LS_new = SUNLinSol_Dense(&y, A.as_ref().unwrap(), &ctx);
        if check_ptr(&LS_new, "SUNLinSol_Dense") {
            std::process::exit(1);
        }
        LS = LS_new;

        /* Attach the matrix and linear solver */
        flag = ARKodeSetLinearSolver(&arkode_mem, LS.as_ref().unwrap(), A.as_ref());
        if check_flag(flag, "ARKodeSetLinearSolver") {
            std::process::exit(1);
        }

        /* Set Jacobian routine */
        flag = ARKodeSetJacFn(&arkode_mem, Some(Jac));
        if check_flag(flag, "ARKodeSetJacFn") {
            std::process::exit(1);
        }

        /* Tighten nonlinear solver tolerance */
        flag = ARKodeSetNonlinConvCoef(&arkode_mem, 0.01);
        if check_flag(flag, "ARKodeSetNonlinConvCoef") {
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
    let mut UFID = File::create("ark_dissipated_exp_entropy.txt").ok();
    if let Some(fp) = UFID.as_mut() {
        let _ = write!(fp, "# vars: t u entropy u_err delta_entropy\n");
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
                "{} {} {} {} {}\n",
                fmt_ew(t0, 23, 16),
                fmt_ew(ydata[0], 23, 16),
                fmt_ew(ent0, 23, 16),
                fmt_ew(0.0, 23, 16),
                fmt_ew(0.0, 23, 16)
            );
        }

        print!(" step   t              u              e              u_err          delta e\n");
        print!(
            " -------------------------------------------------------------------------------\n"
        );
        print!(
            "{:5} {} {} {} {} {}\n",
            0,
            fmt_ew(t, 14, 6),
            fmt_ew(ydata[0], 14, 6),
            fmt_ew(ent0, 14, 6),
            fmt_ew(0.0, 14, 6),
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

        let (u, ut) = {
            let ydata = N_VGetArrayPointer(&y).expect("vector data");
            let ytdata = N_VGetArrayPointer(&ytrue).expect("vector data");
            (ydata[0], ytdata[0])
        };

        let delta_ent = ent - ent0;
        let u_err = u - ut;

        /* Output to the screen periodically */
        flag = ARKodeGetNumSteps(&arkode_mem, &mut nst);
        let _ = check_flag(flag, "ARKodeGetNumSteps");

        if nst % 40 == 0 {
            print!(
                "{:5} {} {} {} {} {}\n",
                nst,
                fmt_ew(t, 14, 6),
                fmt_ew(u, 14, 6),
                fmt_ew(ent, 14, 6),
                fmt_ew(u_err, 14, 6),
                fmt_ew(delta_ent, 14, 6)
            );
        }

        /* Write all steps to file */
        if let Some(fp) = UFID.as_mut() {
            let _ = write!(
                fp,
                "{} {} {} {} {}\n",
                fmt_ew(t, 23, 16),
                fmt_ew(u, 23, 16),
                fmt_ew(ent, 23, 16),
                fmt_ew(u_err, 23, 16),
                fmt_ew(delta_ent, 23, 16)
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

    flag = ARKodeGetNumRhsEvals(&arkode_mem, 1, &mut nfi);
    let _ = check_flag(flag, "ARKodeGetNumRhsEvals");

    print!("\nFinal Solver Statistics:\n");
    print!(
        "   Internal solver steps = {} (attempted = {})\n",
        nst, nst_a
    );
    print!("   Total number of error test failures = {}\n", netf);
    print!("   Total RHS evals:  Fe = {},  Fi = {}\n", nfe, nfi);

    if implicit != 0 {
        flag = ARKodeGetNumNonlinSolvIters(&arkode_mem, &mut nni);
        let _ = check_flag(flag, "ARKodeGetNumNonlinSolvIters");

        flag = ARKodeGetNumNonlinSolvConvFails(&arkode_mem, &mut ncfn);
        let _ = check_flag(flag, "ARKodeGetNumNonlinSolvConvFails");

        flag = ARKodeGetNumLinSolvSetups(&arkode_mem, &mut nsetups);
        let _ = check_flag(flag, "ARKodeGetNumLinSolvSetups");

        flag = ARKodeGetNumJacEvals(&arkode_mem, &mut nje);
        let _ = check_flag(flag, "ARKodeGetNumJacEvals");

        flag = ARKodeGetNumLinRhsEvals(&arkode_mem, &mut nfeLS);
        let _ = check_flag(flag, "ARKodeGetNumLinRhsEvals");

        print!("   Total number of Newton iterations = {}\n", nni);
        print!(
            "   Total number of linear solver convergence failures = {}\n",
            ncfn
        );
        print!("   Total linear solver setups = {}\n", nsetups);
        print!("   Total number of Jacobian evaluations = {}\n", nje);
        print!(
            "   Total RHS evals for setting up the linear system = {}\n",
            nfeLS
        );
    }

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

    /* Free ARKODE integrator and SUNDIALS objects */
    let mut arkode_mem = Some(arkode_mem);
    ARKodeFree(&mut arkode_mem);
    let _ = SUNLinSolFree(LS);
    if let Some(A) = A {
        SUNMatDestroy(A);
    }
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
fn f(_t: sunrealtype, y: &N_Vector, ydot: &N_Vector, _user_data: &mut Option<Box<dyn Any>>) -> i32 {
    let ydata = N_VGetArrayPointer(y).expect("vector data");
    let mut fdata = N_VGetArrayPointer(ydot).expect("vector data");

    fdata[0] = -EXP(ydata[0]);

    0
}

/* ODE RHS Jacobian function J(t,y) = df/dy. */
fn Jac(
    _t: sunrealtype,
    y: &N_Vector,
    _fy: &N_Vector,
    J: &SUNMatrix,
    _user_data: &mut Option<Box<dyn Any>>,
    _tmp1: &N_Vector,
    _tmp2: &N_Vector,
    _tmp3: &N_Vector,
) -> i32 {
    let ydata = N_VGetArrayPointer(y).expect("vector data");
    let mut Jdata = SUNDenseMatrix_Data(J);

    Jdata[0] = -EXP(ydata[0]);

    0
}

/* Entropy function e(y) */
fn Ent(y: &N_Vector, e: &mut sunrealtype, _user_data: &mut Option<Box<dyn Any>>) -> i32 {
    let ydata = N_VGetArrayPointer(y).expect("vector data");

    *e = EXP(ydata[0]);

    0
}

/* Entropy function Jacobian Je(y) = de/dy */
fn JacEnt(y: &N_Vector, J: &N_Vector, _user_data: &mut Option<Box<dyn Any>>) -> i32 {
    let ydata = N_VGetArrayPointer(y).expect("vector data");
    let mut jdata = N_VGetArrayPointer(J).expect("vector data");

    jdata[0] = EXP(ydata[0]);

    0
}

/* ----------------- *
 * Utility functions *
 * ----------------- */

/* Analytic solution */
fn ans(t: sunrealtype, y: &N_Vector) -> i32 {
    let mut ydata = N_VGetArrayPointer(y).expect("vector data");

    ydata[0] = -LOG(EXP(-0.5) + t);

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
