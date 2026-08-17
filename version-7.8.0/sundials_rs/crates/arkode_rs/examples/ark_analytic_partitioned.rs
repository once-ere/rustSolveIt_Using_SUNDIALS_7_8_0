/*------------------------------------------------------------------------------
 * Rust port of examples/arkode/C_serial/ark_analytic_partitioned.c
 * Programmer(s): Steven B. Roberts @ LLNL
 *------------------------------------------------------------------------------
 * We consider the initial value problem
 *    y' + lambda*y = y^2, y(0) = 1
 * proposed in
 *
 * Estep, D., et al. "An a posteriori-a priori analysis of multiscale operator
 * splitting." SIAM Journal on Numerical Analysis 46.3 (2008): 1116-1146.
 *
 * The parameter lambda is positive, t is in [0, 1], and the exact solution is
 *
 *    y(t) = lambda*y / (y(0) - (y(0) - lambda)*exp(lambda*t))
 *
 * This program solves the problem with a splitting or forcing method which can
 * be specified with the command line syntax
 *
 * ./ark_analytic_partitioned <integrator> <coefficients>
 *    integrator: either 'splitting' or 'forcing'
 *    coefficients (splitting only): the SplittingStepCoefficients to load
 *
 * The linear term lambda*y and nonlinear term y^2 are treated as the two
 * partitions. The former is integrated using a time step of 5e-3, while the
 * later uses a time step of 1e-3. The overall splitting or forcing integrator
 * uses a time step of 1e-2. Once solved, the program prints the error and
 * statistics.
 *----------------------------------------------------------------------------*/

#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

use arkode_rs::prelude::*;

use std::any::Any;

struct UserData {
    lambda: sunrealtype,
}

fn main() {
    /* Parse arguments */
    let argv: Vec<String> = std::env::args().collect();
    let argc: usize = argv.len();
    let integrator_name: &str = if argc > 1 { &argv[1] } else { "splitting" };
    if integrator_name != "splitting" && integrator_name != "forcing" {
        eprint!(
            "Invalid integrator: {}\nMust be 'splitting' or 'forcing'\n",
            integrator_name
        );
        std::process::exit(1);
    }
    let coefficients_name: Option<&str> = if argc > 2 {
        Some(&argv[2])
    } else {
        None
    };

    /* Problem parameters */
    let t0: sunrealtype = 0.0; /* initial time */
    let tf: sunrealtype = 1.0; /* final time */
    let dt: sunrealtype = 0.01; /* outer time step */
    let dt_linear: sunrealtype = dt / 5.0; /* linear integrator time step */
    let dt_nonlinear: sunrealtype = dt / 10.0; /* nonlinear integrator time step */

    let user_data = UserData { lambda: 2.0 };

    /* Create the SUNDIALS context object for this simulation */
    let mut ctx_opt: Option<SUNContext> = None;
    let mut flag = SUNContext_Create(SUN_COMM_NULL, &mut ctx_opt);
    if check_flag(Some(flag), "SUNContext_Create", 1) != 0 {
        std::process::exit(1);
    }
    let ctx = ctx_opt.as_ref().expect("ctx").clone();

    /* Initialize vector with initial condition */
    let y = N_VNew_Serial(1, &ctx);
    if check_flag(y.as_ref().map(|_| 0), "N_VNew_Serial", 0) != 0 {
        std::process::exit(1);
    }
    let y = y.expect("y");
    N_VConst(1.0, &y);

    let y_exact = exact_sol(&y, tf, &user_data);

    print!("\nAnalytical ODE test problem:\n");
    print!("   integrator = {} method\n", integrator_name);
    if let Some(coefficients_name) = coefficients_name {
        print!("   coefficients = {}\n", coefficients_name);
    }
    print!("   lambda     = {}\n", fmt_g(user_data.lambda, 6));

    /* Create the integrator for the linear partition */
    let mut linear_mem_opt = ERKStepCreate(f_linear, t0, &y, &ctx);
    if check_flag(linear_mem_opt.as_ref().map(|_| 0), "N_VNew_Serial", 0) != 0 {
        std::process::exit(1);
    }
    let linear_mem = linear_mem_opt.as_ref().expect("linear_mem").clone();

    flag = ARKodeSetUserData(
        &linear_mem,
        Some(Box::new(UserData {
            lambda: user_data.lambda,
        })),
    );
    if check_flag(Some(flag), "ARKodeSetUserData", 1) != 0 {
        std::process::exit(1);
    }

    flag = ARKodeSetFixedStep(&linear_mem, dt_linear);
    if check_flag(Some(flag), "ARKodeSetFixedStep", 1) != 0 {
        std::process::exit(1);
    }

    /* Create the integrator for the nonlinear partition */
    let mut nonlinear_mem_opt = ARKStepCreate(Some(f_nonlinear), None, t0, &y, &ctx);
    if check_flag(nonlinear_mem_opt.as_ref().map(|_| 0), "N_VNew_Serial", 0) != 0 {
        std::process::exit(1);
    }
    let nonlinear_mem = nonlinear_mem_opt.as_ref().expect("nonlinear_mem").clone();

    flag = ARKodeSetFixedStep(&nonlinear_mem, dt_nonlinear);
    if check_flag(Some(flag), "ARKodeSetFixedStep", 1) != 0 {
        std::process::exit(1);
    }

    /* Create SUNSteppers out of the integrators */
    let mut stepper0: Option<SUNStepper> = None;
    let mut stepper1: Option<SUNStepper> = None;
    let _ = ARKodeCreateSUNStepper(&linear_mem, &mut stepper0);
    let _ = ARKodeCreateSUNStepper(&nonlinear_mem, &mut stepper1);
    let steppers: [SUNStepper; 2] = [
        stepper0.as_ref().expect("steppers[0]").clone(),
        stepper1.as_ref().expect("steppers[1]").clone(),
    ];

    /* Create the outer integrator */
    let mut arkode_mem_opt: Option<ARKodeMem>;
    if integrator_name == "splitting" {
        arkode_mem_opt = SplittingStepCreate(&steppers, 2, t0, &y, &ctx);
        if check_flag(
            arkode_mem_opt.as_ref().map(|_| 0),
            "SplittingStepCreate",
            0,
        ) != 0
        {
            std::process::exit(1);
        }

        if let Some(coefficients_name) = coefficients_name {
            let mut coefficients =
                SplittingStepCoefficients_LoadCoefficientsByName(coefficients_name);
            if check_flag(
                coefficients.as_ref().map(|_| 0),
                "SplittingStepCoefficients_LoadCoefficientsByName",
                0,
            ) != 0
            {
                std::process::exit(1);
            }

            flag = SplittingStepSetCoefficients(
                arkode_mem_opt.as_ref().expect("arkode_mem"),
                coefficients.as_ref(),
            );
            if check_flag(Some(flag), "SplittingStepSetCoefficients", 1) != 0 {
                std::process::exit(1);
            }

            SplittingStepCoefficients_Destroy(&mut coefficients);
        }
    } else {
        arkode_mem_opt = ForcingStepCreate(&steppers[0], &steppers[1], t0, &y, &ctx);
        if check_flag(arkode_mem_opt.as_ref().map(|_| 0), "ForcingStepCreate", 0) != 0 {
            std::process::exit(1);
        }
    }
    let arkode_mem = arkode_mem_opt.as_ref().expect("arkode_mem").clone();

    flag = ARKodeSetFixedStep(&arkode_mem, dt);
    if check_flag(Some(flag), "ARKodeSetFixedStep", 1) != 0 {
        std::process::exit(1);
    }

    /* Compute the numerical solution */
    let mut tret: sunrealtype = 0.0;
    flag = ARKodeEvolve(&arkode_mem, tf, &y, &mut tret, ARK_NORMAL);
    if check_flag(Some(flag), "ARKodeEvolve", 1) != 0 {
        std::process::exit(1);
    }

    /* Print the numerical error and statistics */
    let y_err = N_VClone(&y);
    if check_flag(y_err.as_ref().map(|_| 0), "N_VClone", 0) != 0 {
        std::process::exit(1);
    }
    let y_err = y_err.expect("y_err");
    N_VLinearSum(1.0, &y, -1.0, &y_exact, &y_err);
    print!("\nError: {}\n", fmt_g(N_VMaxNorm(&y_err), 6));

    print!("\nSplitting Stepper Statistics:\n");
    flag = ARKodePrintAllStats(
        &arkode_mem,
        &SUNFile::Stdout,
        SUNOutputFormat::SUN_OUTPUTFORMAT_TABLE,
    );
    if check_flag(Some(flag), "ARKodePrintAllStats", 1) != 0 {
        std::process::exit(1);
    }

    print!("\nLinear Stepper Statistics:\n");
    flag = ARKodePrintAllStats(
        &linear_mem,
        &SUNFile::Stdout,
        SUNOutputFormat::SUN_OUTPUTFORMAT_TABLE,
    );
    if check_flag(Some(flag), "ARKodePrintAllStats", 1) != 0 {
        std::process::exit(1);
    }

    print!("\nNonlinear Stepper Statistics:\n");
    flag = ARKodePrintAllStats(
        &nonlinear_mem,
        &SUNFile::Stdout,
        SUNOutputFormat::SUN_OUTPUTFORMAT_TABLE,
    );
    if check_flag(Some(flag), "ARKodePrintAllStats", 1) != 0 {
        std::process::exit(1);
    }

    /* Free memory */
    N_VDestroy(y);
    N_VDestroy(y_exact);
    N_VDestroy(y_err);
    drop(steppers);
    drop(arkode_mem);
    drop(linear_mem);
    drop(nonlinear_mem);
    ARKodeFree(&mut linear_mem_opt);
    SUNStepper_Destroy(&mut stepper0);
    ARKodeFree(&mut nonlinear_mem_opt);
    SUNStepper_Destroy(&mut stepper1);
    ARKodeFree(&mut arkode_mem_opt);
    SUNContext_Free(&mut ctx_opt);

    std::process::exit(0);
}

/* RHS for f^1(t, y) = -lambda * y */
fn f_linear(
    _t: sunrealtype,
    y: &N_Vector,
    ydot: &N_Vector,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    let lambda = user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<UserData>())
        .expect("user_data")
        .lambda;
    N_VScale(-lambda, y, ydot);
    0
}

/* RHS for f^2(t, y) = y^2 */
fn f_nonlinear(
    _t: sunrealtype,
    y: &N_Vector,
    ydot: &N_Vector,
    _user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    N_VProd(y, y, ydot);
    0
}

/* Compute the exact analytic solution */
fn exact_sol(y0: &N_Vector, tf: sunrealtype, user_data: &UserData) -> N_Vector {
    let sol = N_VClone(y0).expect("N_VClone");
    let y0_val = N_VGetArrayPointer(y0).expect("y0 data")[0];
    let lambda = user_data.lambda;
    N_VGetArrayPointer(&sol).expect("sol data")[0] =
        lambda * y0_val / (y0_val - (y0_val - lambda) * SUNRexp(lambda * tf));
    sol
}

/* Check function return value...
    opt == 0 means SUNDIALS function allocates memory so check if
             returned NULL pointer (represented as `None`)
    opt == 1 means SUNDIALS function returns a flag so check if
             flag >= 0
    opt == 2 means function allocates memory so check if returned
             NULL pointer
*/
fn check_flag(flagvalue: Option<i32>, funcname: &str, opt: i32) -> i32 {
    /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
    if opt == 0 && flagvalue.is_none() {
        eprint!(
            "\nSUNDIALS_ERROR: {}() failed - returned NULL pointer\n\n",
            funcname
        );
        return 1;
    }
    /* Check if flag < 0 */
    else if opt == 1 {
        let errflag = flagvalue.expect("errflag");
        if errflag < 0 {
            eprint!(
                "\nSUNDIALS_ERROR: {}() failed with flag = {}\n\n",
                funcname, errflag
            );
            return 1;
        }
    }
    /* Check if function returned NULL pointer - no memory allocated */
    else if opt == 2 && flagvalue.is_none() {
        eprint!(
            "\nMEMORY_ERROR: {}() failed - returned NULL pointer\n\n",
            funcname
        );
        return 1;
    }

    0
}

/*---- end of file ----*/
