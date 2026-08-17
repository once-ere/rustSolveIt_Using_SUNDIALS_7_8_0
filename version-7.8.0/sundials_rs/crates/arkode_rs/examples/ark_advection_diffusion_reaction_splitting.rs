/*---------------------------------------------------------------
 * Rust port of
 * examples/arkode/C_serial/ark_advection_diffusion_reaction_splitting.c
 *---------------------------------------------------------------
 * Example problem:
 *
 * The following test simulates a simple 1D advection-diffusion-
 * reaction equation,
 *    u_t = (a/2)*(u^2)_x + b*u_xx + c*(u - u^3)
 * for t in [0, 1], x in [0, 1], with initial conditions
 *    u(0,x) = u_0
 * and Dirichlet boundary conditions at x=0 and x=1
 *    u(0,t) = u(1,t) = u_0
 *
 * This program solves the problem with an operator splitting
 * method where advection is treated with a strong stability
 * preserving ERK method, diffusion is treated with a DIRK
 * method, and reaction is treated with a different ERK method.
 *---------------------------------------------------------------*/
#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

use std::any::Any;
use std::rc::Rc;

use arkode_rs::prelude::*;
use arkode_rs::sunmatrix_band::SM_ELEMENT_B_set;

/* user data structure */
struct UserData {
    N: sunindextype,  /* number of grid points (excluding boundaries) */
    dx: sunrealtype,  /* mesh spacing */
    a: sunrealtype,   /* advection coefficient */
    b: sunrealtype,   /* diffusion coefficient */
    c: sunrealtype,   /* reaction coefficient */
    u0: sunrealtype,  /* initial and boundary values */
}

/* The C program hands the SAME `&udata` pointer to all three inner
 * integrators. `user_data` is an owned `Option<Box<dyn Any>>` here, so the
 * shared C pointer maps to an `Rc<UserData>` clone per integrator — the
 * struct is read-only in every callback, exactly as in C. */
type UserDataRef = Rc<UserData>;

fn udata_of(user_data: &mut Option<Box<dyn Any>>) -> UserDataRef {
    user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<UserDataRef>())
        .expect("user_data is UserData")
        .clone()
}

fn main() {
    /* Problem parameters */
    let T0: sunrealtype = 0.0;
    let Tf: sunrealtype = 1.0;
    let DT: sunrealtype = 0.06;
    let N: sunindextype = 128;
    let udata = UserData {
        N,
        dx: 1.0 / (N + 1) as sunrealtype,
        a: 1.0,
        b: 0.125,
        c: 4.0,
        u0: 0.1,
    };

    print!("\n1D Advection-Diffusion-Reaction PDE test problem:\n");
    print!("  N = {}\n", udata.N);
    print!("  advection coefficient = {}\n", fmt_g(udata.a, 6));
    print!("  diffusion coefficient = {}\n", fmt_g(udata.b, 6));
    print!("  reaction coefficient  = {}\n\n", fmt_g(udata.c, 6));

    let N = udata.N;
    let udata: UserDataRef = Rc::new(udata);

    /* Create the SUNDIALS context object for this simulation */
    let mut ctx_opt: Option<SUNContext> = None;
    let flag = SUNContext_Create(SUN_COMM_NULL, &mut ctx_opt);
    if check_flag(flag, "SUNContext_Create", 1) != 0 {
        std::process::exit(1);
    }
    let ctx = ctx_opt.as_ref().expect("SUNContext").clone();

    /* Initialize vector with initial condition */
    let y = N_VNew_Serial(N, &ctx);
    if check_flag_ptr(&y, "N_VNew_Serial", 0) != 0 {
        std::process::exit(1);
    }
    let y = y.expect("N_VNew_Serial");
    N_VConst(udata.u0, &y);

    /* Create advection integrator */
    let advection_mem = ERKStepCreate(f_advection, T0, &y, &ctx);
    if check_flag_ptr(&advection_mem, "ERKStepCreate", 0) != 0 {
        std::process::exit(1);
    }
    let mut advection_mem = advection_mem;

    let flag = ARKodeSetUserData(
        advection_mem.as_ref().expect("advection_mem"),
        Some(Box::new(udata.clone())),
    );
    if check_flag(flag, "ARKodeSetUserData", 1) != 0 {
        std::process::exit(1);
    }

    /* Choose a strong stability preserving method for advecton */
    let flag = ERKStepSetTableNum(
        advection_mem.as_ref().expect("advection_mem"),
        ARKODE_SHU_OSHER_3_2_3,
    );
    if check_flag(flag, "ERKStepSetTableNum", 1) != 0 {
        std::process::exit(1);
    }

    let mut advection_stepper: Option<SUNStepper> = None;
    let flag = ARKodeCreateSUNStepper(
        advection_mem.as_ref().expect("advection_mem"),
        &mut advection_stepper,
    );
    if check_flag(flag, "ARKodeCreateSUNStepper", 1) != 0 {
        std::process::exit(1);
    }

    /* Create diffusion integrator */
    let diffusion_mem = ARKStepCreate(None, Some(f_diffusion), T0, &y, &ctx);
    if check_flag_ptr(&diffusion_mem, "ARKStepCreate", 0) != 0 {
        std::process::exit(1);
    }
    let mut diffusion_mem = diffusion_mem;

    let flag = ARKodeSetUserData(
        diffusion_mem.as_ref().expect("diffusion_mem"),
        Some(Box::new(udata.clone())),
    );
    if check_flag(flag, "ARKodeSetUserData", 1) != 0 {
        std::process::exit(1);
    }

    let flag = ARKodeSetOrder(diffusion_mem.as_ref().expect("diffusion_mem"), 3);
    if check_flag(flag, "ARKStepSetOrder", 1) != 0 {
        std::process::exit(1);
    }

    let jac_mat = SUNBandMatrix(udata.N, 1, 1, &ctx);
    if check_flag_ptr(&jac_mat, "SUNBandMatrix", 0) != 0 {
        std::process::exit(1);
    }
    let jac_mat = jac_mat.expect("SUNBandMatrix");

    let ls = SUNLinSol_Band(&y, &jac_mat, &ctx);
    if check_flag_ptr(&ls, "SUNLinSol_Band", 0) != 0 {
        std::process::exit(1);
    }
    let ls = ls.expect("SUNLinSol_Band");

    let flag = ARKodeSetLinearSolver(
        diffusion_mem.as_ref().expect("diffusion_mem"),
        &ls,
        Some(&jac_mat),
    );
    if check_flag(flag, "ARKStepSetOrder", 1) != 0 {
        std::process::exit(1);
    }

    let flag = ARKodeSetJacFn(
        diffusion_mem.as_ref().expect("diffusion_mem"),
        Some(jac_diffusion),
    );
    if check_flag(flag, "ARKodeSetJacFn", 1) != 0 {
        std::process::exit(1);
    }

    let flag = ARKodeSetLinear(
        diffusion_mem.as_ref().expect("diffusion_mem"),
        SUNFALSE as i32,
    );
    if check_flag(flag, "ARKodeSetLinear", 1) != 0 {
        std::process::exit(1);
    }

    let mut diffusion_stepper: Option<SUNStepper> = None;
    let flag = ARKodeCreateSUNStepper(
        diffusion_mem.as_ref().expect("diffusion_mem"),
        &mut diffusion_stepper,
    );
    if check_flag(flag, "ARKodeCreateSUNStepper", 1) != 0 {
        std::process::exit(1);
    }

    /* Create reaction integrator */
    let reaction_mem = ERKStepCreate(f_reaction, T0, &y, &ctx);
    if check_flag_ptr(&reaction_mem, "ERKStepCreate", 0) != 0 {
        std::process::exit(1);
    }
    let mut reaction_mem = reaction_mem;

    let flag = ARKodeSetUserData(
        reaction_mem.as_ref().expect("reaction_mem"),
        Some(Box::new(udata.clone())),
    );
    if check_flag(flag, "ARKodeSetUserData", 1) != 0 {
        std::process::exit(1);
    }

    let flag = ARKodeSetOrder(reaction_mem.as_ref().expect("reaction_mem"), 3);
    if check_flag(flag, "ARKodeSetOrder", 1) != 0 {
        std::process::exit(1);
    }

    let mut reaction_stepper: Option<SUNStepper> = None;
    let flag = ARKodeCreateSUNStepper(
        reaction_mem.as_ref().expect("reaction_mem"),
        &mut reaction_stepper,
    );
    if check_flag(flag, "ARKodeCreateSUNStepper", 1) != 0 {
        std::process::exit(1);
    }

    /* Create operator splitting integrator */
    let steppers: [SUNStepper; 3] = [
        advection_stepper.as_ref().expect("advection_stepper").clone(),
        diffusion_stepper.as_ref().expect("diffusion_stepper").clone(),
        reaction_stepper.as_ref().expect("reaction_stepper").clone(),
    ];
    let arkode_mem = SplittingStepCreate(&steppers, 3, T0, &y, &ctx);
    if check_flag_ptr(&arkode_mem, "SplittingStepCreate", 0) != 0 {
        std::process::exit(1);
    }
    let mut arkode_mem = arkode_mem;

    let flag = ARKodeSetFixedStep(arkode_mem.as_ref().expect("arkode_mem"), DT);
    if check_flag(flag, "ARKodeSetFixedStep", 1) != 0 {
        std::process::exit(1);
    }

    let flag = ARKodeSetStopTime(arkode_mem.as_ref().expect("arkode_mem"), Tf);
    if check_flag(flag, "ARKodeSetStopTime", 1) != 0 {
        std::process::exit(1);
    }

    /* Evolve solution in time */
    let mut tret: sunrealtype = T0;
    print!("        t      ||u||_rms\n");
    print!("   ----------------------\n");
    print!(
        "  {}  {}\n",
        fmt_fw(tret, 10, 6),
        fmt_fw(SUNRsqrt(N_VDotProd(&y, &y) / N as sunrealtype), 10, 6)
    );
    while tret < Tf {
        let flag = ARKodeEvolve(
            arkode_mem.as_ref().expect("arkode_mem"),
            Tf,
            &y,
            &mut tret,
            ARK_ONE_STEP,
        );
        if check_flag(flag, "ARKodeEvolve", 1) != 0 {
            std::process::exit(1);
        }
        print!(
            "  {}  {}\n",
            fmt_fw(tret, 10, 6),
            fmt_fw(SUNRsqrt(N_VDotProd(&y, &y) / N as sunrealtype), 10, 6)
        );
    }
    print!("   ----------------------\n");

    /* Print statistics */
    print!("\nSplitting Stepper Statistics:\n");
    let flag = ARKodePrintAllStats(
        arkode_mem.as_ref().expect("arkode_mem"),
        &SUNFile::Stdout,
        SUNOutputFormat::SUN_OUTPUTFORMAT_TABLE,
    );
    if check_flag(flag, "ARKodePrintAllStats", 1) != 0 {
        std::process::exit(1);
    }

    print!("\nAdvection Stepper Statistics:\n");
    let flag = ARKodePrintAllStats(
        advection_mem.as_ref().expect("advection_mem"),
        &SUNFile::Stdout,
        SUNOutputFormat::SUN_OUTPUTFORMAT_TABLE,
    );
    if check_flag(flag, "ARKodePrintAllStats", 1) != 0 {
        std::process::exit(1);
    }

    print!("\nDiffusion Stepper Statistics:\n");
    let flag = ARKodePrintAllStats(
        diffusion_mem.as_ref().expect("diffusion_mem"),
        &SUNFile::Stdout,
        SUNOutputFormat::SUN_OUTPUTFORMAT_TABLE,
    );
    if check_flag(flag, "ARKodePrintAllStats", 1) != 0 {
        std::process::exit(1);
    }

    print!("\nReaction Stepper Statistics:\n");
    let flag = ARKodePrintAllStats(
        reaction_mem.as_ref().expect("reaction_mem"),
        &SUNFile::Stdout,
        SUNOutputFormat::SUN_OUTPUTFORMAT_TABLE,
    );
    if check_flag(flag, "ARKodePrintAllStats", 1) != 0 {
        std::process::exit(1);
    }

    /* Clean up and return with successful completion */
    N_VDestroy(y);
    ARKodeFree(&mut advection_mem);
    SUNStepper_Destroy(&mut advection_stepper);
    ARKodeFree(&mut diffusion_mem);
    SUNStepper_Destroy(&mut diffusion_stepper);
    ARKodeFree(&mut reaction_mem);
    SUNStepper_Destroy(&mut reaction_stepper);
    ARKodeFree(&mut arkode_mem);
    SUNLinSolFree(Some(ls));
    SUNMatDestroy(jac_mat);

    drop(steppers);
    drop(ctx);
    SUNContext_Free(&mut ctx_opt);
}

/* f routine to compute the advection RHS function. */
fn f_advection(
    _t: sunrealtype,
    y: &N_Vector,
    ydot: &N_Vector,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    let udata = udata_of(user_data);
    let Y = N_VGetArrayPointer(y); /* access data arrays */
    if check_flag_ptr(&Y, "N_VGetArrayPointer", 0) != 0 {
        return 1;
    }
    let Y = Y.expect("N_VGetArrayPointer");

    let Ydot = N_VGetArrayPointer(ydot);
    if check_flag_ptr(&Ydot, "N_VGetArrayPointer", 0) != 0 {
        return 1;
    }
    let mut Ydot = Ydot.expect("N_VGetArrayPointer");

    let coeff = udata.a / (4.0 * udata.dx);
    let u0_sqr = udata.u0 * udata.u0;

    /* Left boundary */
    Ydot[0] = coeff * (Y[1] * Y[1] - u0_sqr);
    /* Interior */
    for i in 1..(udata.N - 1) {
        let i = i as usize;
        Ydot[i] = coeff * (Y[i + 1] * Y[i + 1] - Y[i - 1] * Y[i - 1]);
    }
    /* Right boundary */
    let nm1 = (udata.N - 1) as usize;
    Ydot[nm1] = coeff * (u0_sqr - Y[nm1] * Y[nm1]);

    0
}

/* f routine to compute the diffusion RHS function. */
fn f_diffusion(
    _t: sunrealtype,
    y: &N_Vector,
    ydot: &N_Vector,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    let udata = udata_of(user_data);
    let Y = N_VGetArrayPointer(y); /* access data arrays */
    if check_flag_ptr(&Y, "N_VGetArrayPointer", 0) != 0 {
        return 1;
    }
    let Y = Y.expect("N_VGetArrayPointer");

    let Ydot = N_VGetArrayPointer(ydot);
    if check_flag_ptr(&Ydot, "N_VGetArrayPointer", 0) != 0 {
        return 1;
    }
    let mut Ydot = Ydot.expect("N_VGetArrayPointer");

    let coeff = udata.b / (udata.dx * udata.dx);

    /* Left boundary */
    Ydot[0] = coeff * (udata.u0 - 2.0 * Y[0] + Y[1]);
    /* Interior */
    for i in 1..(udata.N - 1) {
        let i = i as usize;
        Ydot[i] = coeff * (Y[i + 1] - 2.0 * Y[i] + Y[i - 1]);
    }
    /* Right boundary */
    let nm1 = (udata.N - 1) as usize;
    Ydot[nm1] = coeff * (Y[(udata.N - 2) as usize] - 2.0 * Y[nm1] + udata.u0);

    0
}

/* Routine to compute the diffusion Jacobian function. */
fn jac_diffusion(
    _t: sunrealtype,
    _y: &N_Vector,
    _fy: &N_Vector,
    Jac: &SUNMatrix,
    user_data: &mut Option<Box<dyn Any>>,
    _tmp1: &N_Vector,
    _tmp2: &N_Vector,
    _tmp3: &N_Vector,
) -> i32 {
    let udata = udata_of(user_data);
    let coeff = udata.b / (udata.dx * udata.dx);

    SM_ELEMENT_B_set(Jac, 0, 0, -2.0 * coeff);
    for i in 1..(udata.N as i32) {
        let i = i as sunindextype;
        SM_ELEMENT_B_set(Jac, i - 1, i, coeff);
        SM_ELEMENT_B_set(Jac, i, i, -2.0 * coeff);
        SM_ELEMENT_B_set(Jac, i, i - 1, coeff);
    }

    0
}

/* f routine to compute the reaction RHS function. */
fn f_reaction(
    _t: sunrealtype,
    y: &N_Vector,
    ydot: &N_Vector,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    let udata = udata_of(user_data);
    let Y = N_VGetArrayPointer(y); /* access data arrays */
    if check_flag_ptr(&Y, "N_VGetArrayPointer", 0) != 0 {
        return 1;
    }
    let Y = Y.expect("N_VGetArrayPointer");

    let Ydot = N_VGetArrayPointer(ydot);
    if check_flag_ptr(&Ydot, "N_VGetArrayPointer", 0) != 0 {
        return 1;
    }
    let mut Ydot = Ydot.expect("N_VGetArrayPointer");

    for i in 0..udata.N {
        let i = i as usize;
        Ydot[i] = udata.c * Y[i] * (1.0 - Y[i] * Y[i]);
    }

    0
}

/* Check function return value...
    opt == 0 means SUNDIALS function allocates memory so check if
             returned NULL pointer
    opt == 1 means SUNDIALS function returns a flag so check if
             flag >= 0
    opt == 2 means function allocates memory so check if returned
             NULL pointer
*/
fn check_flag(flag: i32, funcname: &str, _opt: i32) -> i32 {
    if flag < 0 {
        eprint!(
            "\nSUNDIALS_ERROR: {}() failed with flag = {}\n\n",
            funcname, flag
        );
        return 1;
    }
    0
}

fn check_flag_ptr<T>(flagvalue: &Option<T>, funcname: &str, opt: i32) -> i32 {
    if flagvalue.is_none() {
        if opt == 2 {
            eprint!(
                "\nMEMORY_ERROR: {}() failed - returned NULL pointer\n\n",
                funcname
            );
        } else {
            eprint!(
                "\nSUNDIALS_ERROR: {}() failed - returned NULL pointer\n\n",
                funcname
            );
        }
        return 1;
    }
    0
}
