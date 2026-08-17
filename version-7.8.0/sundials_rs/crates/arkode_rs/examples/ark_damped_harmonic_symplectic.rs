/* clang-format off */
/* ----------------------------------------------------------------------------
 * Rust port of examples/arkode/C_serial/ark_damped_harmonic_symplectic.c
 * Programmer(s): Cody J. Balos @ LLNL
 * ----------------------------------------------------------------------------
 * In this example we consider the time-dependent damped harmonic oscillator
 *    q'(t) = p(t) exp(-F(t))
 *    p'(t) = -(F(t) * p + omega^2(t) * q)
 * With the initial conditions q(0) = 1, p(0) = 0.
 * The Hamiltonian for the system is
 *    H(p,q,t) = (p^2 * exp(-F(t)))/2 + (omega^2(t) * q^2 * exp(F(t)))/2
 * where omega(t) = cos(t/2), F(t) = 0.018*sin(t/pi).
 * We simulate the problem on t = [0, 30] using the symplectic methods in
 * SPRKStep.
 *
 * This is example 7.2 from:
 * Struckmeier, J., & Riedel, C. (2002). Canonical transformations and exact
 * invariants for time-dependent Hamiltonian systems. Annalen der Physik, 11(1),
 * 15-38.
 *
 * The example has the following command line arguments:
 *   --order <int>               the order of the method to use (default 4)
 *   --dt <Real>                 the fixed-time step size to use (default 0.01)
 *   --nout <int>                the number of output times (default 100)
 *   --disable-tstop             turns off tstop mode
 *   --use-compensated-sums      turns on compensated summation in ARKODE where
 *                               applicable
 * --------------------------------------------------------------------------*/
/* clang-format on */

#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

use arkode_rs::prelude::*;

use std::any::Any;

/* ark_damped_harmonic_symplectic.h */

const PI: sunrealtype = 3.14159265358979323846264338327950;

struct ProgramArgs {
    order: i32,
    num_output_times: i32,
    use_compsums: i32,
    use_tstop: i32,
    Tf: sunrealtype,
    dt: sunrealtype,
}

fn PrintHelp() {
    eprint!(
        "ark_damped_harmonic_symplectic: an ARKODE example demonstrating \
         the SPRKStep time-stepping module solving a time-dependent \
         damped harmonic oscillator\n"
    );
    /* clang-format off */
    eprint!("  --order <int>               the order of the method to use (default 4)\n");
    eprint!("  --dt <Real>                 the fixed-time step size to use (default 0.01)\n");
    eprint!("  --nout <int>                the number of output times (default 100)\n");
    eprint!(
        "  --use-compensated-sums      turns on compensated summation in ARKODE where applicable\n"
    );
    eprint!("  --disable-tstop             turns off tstop mode\n");
    /* clang-format on */
}

fn ParseArgs(argc: usize, argv: &[String], args: &mut ProgramArgs) -> i32 {
    let mut argi: usize;

    args.order = 4;
    args.num_output_times = 8;
    args.use_compsums = 0;
    args.use_tstop = 1;
    args.Tf = 10.0 * PI;
    args.dt = 1e-3;

    argi = 1;
    while argi < argc {
        if argv[argi] == "--order" {
            argi += 1;
            args.order = atoi(&argv[argi]);
        } else if argv[argi] == "--tf" {
            argi += 1;
            args.Tf = SUNStrToReal(&argv[argi]);
        } else if argv[argi] == "--dt" {
            argi += 1;
            args.dt = SUNStrToReal(&argv[argi]);
        } else if argv[argi] == "--nout" {
            argi += 1;
            args.num_output_times = atoi(&argv[argi]);
        } else if argv[argi] == "--use-compensated-sums" {
            args.use_compsums = 1;
        } else if argv[argi] == "--disable-tstop" {
            args.use_tstop = 0;
        } else if argv[argi] == "--help" {
            PrintHelp();
            return 1;
        } else {
            eprint!("ERROR: unrecognized argument {}\n", argv[argi]);
            PrintHelp();
            return 1;
        }
        argi += 1;
    }

    0
}

/* Check function return value...
    opt == 0 means SUNDIALS function allocates memory so check if
             returned NULL pointer
    opt == 1 means SUNDIALS function returns a retval so check if
             retval < 0
    opt == 2 means function allocates memory so check if returned
             NULL pointer
*/
fn check_retval(returnvalue: Option<i32>, funcname: &str, opt: i32) -> i32 {
    /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
    if opt == 0 && returnvalue.is_none() {
        eprint!("\nERROR: {}() failed - returned NULL pointer\n\n", funcname);
        return 1;
    }
    /* Check if retval < 0 */
    else if opt == 1 {
        let retval = returnvalue.expect("retval");
        if retval < 0 {
            eprint!(
                "\nERROR: {}() failed with retval = {}\n\n",
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

/* ark_damped_harmonic_symplectic.c */

fn main() {
    let mut args = ProgramArgs {
        order: 0,
        num_output_times: 0,
        use_compsums: 0,
        use_tstop: 0,
        Tf: 0.0,
        dt: 0.0,
    };
    let mut sunctx_opt: Option<SUNContext> = None;
    let mut iout: i32;
    let mut retval: i32;
    let order: i32;
    let use_compsums: i32;
    let num_output_times: i32;
    let Tf: sunrealtype;
    let dt: sunrealtype;
    let dTout: sunrealtype;
    let T0: sunrealtype = 0.0;

    /* Parse the command line arguments */
    let argv: Vec<String> = std::env::args().collect();
    let argc: usize = argv.len();
    if ParseArgs(argc, &argv, &mut args) != 0 {
        std::process::exit(1);
    };

    /* Default integrator options */
    order = args.order;
    use_compsums = args.use_compsums;
    num_output_times = args.num_output_times;

    /* Default problem parameters */
    Tf = args.Tf;
    dt = args.dt;
    dTout = (Tf - T0) / (num_output_times as sunrealtype);

    /* Create the SUNDIALS context object for this simulation */
    retval = SUNContext_Create(SUN_COMM_NULL, &mut sunctx_opt);
    if check_retval(Some(retval), "SUNContext_Create", 1) != 0 {
        std::process::exit(1);
    }
    let sunctx = sunctx_opt.as_ref().expect("sunctx").clone();

    print!("\n   Begin time-dependent damped harmonic oscillator problem\n\n");

    /* Allocate our state vector */
    let y = N_VNew_Serial(2, &sunctx).expect("N_VNew_Serial");

    /* Fill the initial conditions */
    {
        let mut ydata = N_VGetArrayPointer(&y).expect("y data");
        ydata[0] = 0.0; /* \dot{q} = p */
        ydata[1] = 1.0; /* \ddot{q} = \dot{p} */
    }

    /* Create SPRKStep integrator */
    let mut arkode_mem_opt: Option<ARKodeMem> = SPRKStepCreate(qdot, pdot, T0, &y, &sunctx);
    let arkode_mem = arkode_mem_opt.as_ref().expect("arkode_mem").clone();

    retval = ARKodeSetOrder(&arkode_mem, order);
    if check_retval(Some(retval), "ARKodeSetOrder", 1) != 0 {
        std::process::exit(1);
    }

    retval = ARKodeSetUseCompensatedSums(&arkode_mem, use_compsums != 0);
    if check_retval(Some(retval), "ARKodeSetUseCompensatedSums", 1) != 0 {
        std::process::exit(1);
    }

    retval = ARKodeSetFixedStep(&arkode_mem, dt);
    if check_retval(Some(retval), "ARKodeSetFixedStep", 1) != 0 {
        std::process::exit(1);
    }

    retval = ARKodeSetMaxNumSteps(&arkode_mem, ((Tf / dt).ceil() as i64) + 2);
    if check_retval(Some(retval), "ARKodeSetMaxNumSteps", 1) != 0 {
        std::process::exit(1);
    }

    /* Print out starting Hamiltonian before integrating */
    let mut tret: sunrealtype = T0;
    let mut tout: sunrealtype = T0 + dTout;
    /* Output current integration status */
    {
        let ydata1 = N_VGetArrayPointer(&y).expect("y data")[1];
        print!(
            "t = {}, q(t) = {}, H = {}\n",
            fmt_f(tret, 6),
            fmt_f(ydata1, 6),
            fmt_f(Hamiltonian(&y, tret), 6)
        );
    }

    /* Do integration */
    iout = 0;
    while iout < num_output_times {
        if args.use_tstop != 0 {
            let _ = ARKodeSetStopTime(&arkode_mem, tout);
        }
        retval = ARKodeEvolve(&arkode_mem, tout, &y, &mut tret, ARK_NORMAL);

        /* Output current integration status */
        {
            let ydata1 = N_VGetArrayPointer(&y).expect("y data")[1];
            print!(
                "t = {}, q(t) = {}, H = {}\n",
                fmt_f(tret, 6),
                fmt_f(ydata1, 6),
                fmt_f(Hamiltonian(&y, tret), 6)
            );
        }

        /* Check if the solve was successful, if so, update the time and continue */
        if retval >= 0 {
            tout += dTout;
            tout = if tout > Tf { Tf } else { tout };
        } else {
            eprint!("Solver failure, stopping integration\n");
            break;
        }

        iout += 1;
    }

    print!("\n");
    drop(arkode_mem);
    let _ = ARKodePrintAllStats(
        arkode_mem_opt.as_ref().expect("arkode_mem"),
        &SUNFile::Stdout,
        SUNOutputFormat::SUN_OUTPUTFORMAT_TABLE,
    );
    N_VDestroy(y);
    ARKodeFree(&mut arkode_mem_opt);
    SUNContext_Free(&mut sunctx_opt);

    std::process::exit(0);
}

fn omega(t: sunrealtype) -> sunrealtype {
    (t / 2.0).sun_cos()
}

fn F(t: sunrealtype) -> sunrealtype {
    0.018 * (t / PI).sun_sin()
}

fn Hamiltonian(yvec: &N_Vector, t: sunrealtype) -> sunrealtype {
    let H: sunrealtype;
    let y = N_VGetArrayPointer(yvec).expect("y data");
    let p: sunrealtype = y[0];
    let q: sunrealtype = y[1];

    H = (p * p * (-F(t)).sun_exp()) / 2.0 + (omega(t) * omega(t) * q * q * F(t).sun_exp()) / 2.0;

    H
}

fn qdot(
    t: sunrealtype,
    yvec: &N_Vector,
    ydotvec: &N_Vector,
    _user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    let y = N_VGetArrayPointer(yvec).expect("y data");
    let mut ydot = N_VGetArrayPointer(ydotvec).expect("ydot data");
    let p: sunrealtype = y[0];

    ydot[1] = p * (-F(t)).sun_exp();

    0
}

fn pdot(
    t: sunrealtype,
    yvec: &N_Vector,
    ydotvec: &N_Vector,
    _user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    let y = N_VGetArrayPointer(yvec).expect("y data");
    let mut ydot = N_VGetArrayPointer(ydotvec).expect("ydot data");
    let p: sunrealtype = y[0];
    let q: sunrealtype = y[1];

    ydot[0] = -(F(t) * p + omega(t) * omega(t) * q);

    0
}

/*---- end of file ----*/
