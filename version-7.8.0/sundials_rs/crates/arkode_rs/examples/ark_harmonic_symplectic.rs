/* clang-format: off */
/* ----------------------------------------------------------------------------
 * Rust port of examples/arkode/C_serial/ark_harmonic_symplectic.c
 * Programmer(s): Cody J. Balos @ LLNL
 * ----------------------------------------------------------------------------
 * In this example we consider the simple harmonic oscillator
 *    x''(t) + omega^2*x(t) = 0.
 * We rewrite the second order ODE as the first order ODE model
 *    x'(t) = v(t)
 *    v'(t) = -omega^2*x(t).
 * With the initial conditions x(0) = x0 and v(0) = v0,
 * the analytical solution is
 *    x(t) = A*cos(t*omega + phi),
 *    v(t) = -A*omega*sin(t*omega + phi)
 * where A = sqrt(x0^2 + v0^2/omega) and tan(phi) = v0/(omega*x0).
 * The total energy (potential + kinetic) in this system is
 *    E = (v^2 + omega^2*x^2) / 2
 * E is conserved and is the system Hamiltonian.
 * We simulate the problem on t = [0, 2pi] using the symplectic methods
 * in SPRKStep. Symplectic methods will approximately conserve E.
 *
 * The example has the following command line arguments:
 *   --order <int>               the order of the method to use (default 4)
 *   --dt <Real>                 the fixed-time step size to use (default 0.01)
 *   --nout <int>                the number of output times (default 100)
 *   --use-compensated-sums      turns on compensated summation in ARKODE where
 *                               applicable
 *   --disable-tstop             turns off tstop mode
 * --------------------------------------------------------------------------*/
/* clang-format: on */

#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

use arkode_rs::prelude::*;

use std::any::Any;

/* ark_harmonic_symplectic.h */

const PI: sunrealtype = 3.14159265358979323846264338327950;

struct ProgramArgs {
    order: i32,
    num_output_times: i32,
    use_compsums: i32,
    use_tstop: i32,
    Tf: sunrealtype,
    dt: sunrealtype,
}

fn SetDefaultArgs(args: &mut ProgramArgs) {
    args.order = 4;
    args.num_output_times = 8;
    args.use_compsums = 0;
    args.use_tstop = 1;
    args.dt = 1e-3;
    args.Tf = 2.0 * PI;
}

fn PrintHelp() {
    let mut defaults = ProgramArgs {
        order: 0,
        num_output_times: 0,
        use_compsums: 0,
        use_tstop: 0,
        Tf: 0.0,
        dt: 0.0,
    };
    SetDefaultArgs(&mut defaults);
    eprint!(
        "ark_harmonic_symplectic: an ARKODE example demonstrating \
         the SPRKStep time-stepping module solving a simple harmonic \
         oscillator\n"
    );
    /* clang-format off */
    eprint!(
        "  --order <int>               the order of the method to use (default {})\n",
        defaults.order
    );
    eprint!(
        "  --dt <Real>                 the fixed-time step size to use (default {})\n",
        fmt_e(defaults.dt, 1)
    );
    eprint!(
        "  --nout <int>                the number of output times (default {})\n",
        defaults.num_output_times
    );
    eprint!(
        "  --use-compensated-sums      turns on compensated summation in ARKODE where applicable\n"
    );
    eprint!("  --disable-tstop             turns off tstop mode\n");
    /* clang-format on */
}

fn ParseArgs(argc: usize, argv: &[String], args: &mut ProgramArgs) -> i32 {
    let mut argi: usize;

    SetDefaultArgs(args);

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

/* ark_harmonic_symplectic.c */

struct UserData {
    A: sunrealtype,
    phi: sunrealtype,
    omega: sunrealtype,
}

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
    let A: sunrealtype = 10.0;
    let phi: sunrealtype = 0.0;
    let omega: sunrealtype = 1.0;

    /* Parse the command line arguments */
    let argv: Vec<String> = std::env::args().collect();
    let argc: usize = argv.len();
    if ParseArgs(argc, &argv, &mut args) != 0 {
        std::process::exit(1);
    };

    /* Default integrator options and problem parameters */
    order = args.order;
    use_compsums = args.use_compsums;
    num_output_times = args.num_output_times;
    Tf = args.Tf;
    dt = args.dt;
    dTout = (Tf - T0) / (num_output_times as sunrealtype);

    /* Default problem parameters */

    /* Create the SUNDIALS context object for this simulation */
    retval = SUNContext_Create(SUN_COMM_NULL, &mut sunctx_opt);
    if check_retval(Some(retval), "SUNContext_Create", 1) != 0 {
        std::process::exit(1);
    }
    let sunctx = sunctx_opt.as_ref().expect("sunctx").clone();

    print!("\n   Begin simple harmonic oscillator problem\n\n");

    /* Allocate and fill udata structure */
    let udata = UserData { A, phi, omega };

    /* Allocate our state vector [x, v]^T */
    let y = N_VNew_Serial(2, &sunctx).expect("N_VNew_Serial");
    let solution = N_VClone(&y).expect("N_VClone");

    /* Fill the initial conditions (x0 then v0) */
    {
        let mut ydata = N_VGetArrayPointer(&y).expect("y data");
        ydata[0] = A * phi.sun_cos();
        ydata[1] = -A * omega * phi.sun_sin();
    }

    /* Create SPRKStep integrator */
    let mut arkode_mem_opt: Option<ARKodeMem> = SPRKStepCreate(xdot, vdot, T0, &y, &sunctx);
    let arkode_mem = arkode_mem_opt.as_ref().expect("arkode_mem").clone();

    retval = ARKodeSetOrder(&arkode_mem, order);
    if check_retval(Some(retval), "ARKodeSetOrder", 1) != 0 {
        std::process::exit(1);
    }

    retval = ARKodeSetUserData(
        &arkode_mem,
        Some(Box::new(UserData {
            A: udata.A,
            phi: udata.phi,
            omega: udata.omega,
        })),
    );
    if check_retval(Some(retval), "ARKodeSetUserData", 1) != 0 {
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

    /* Print out starting energy, momentum before integrating */
    let mut tret: sunrealtype = T0;
    let mut tout: sunrealtype = T0 + dTout;
    {
        let ydata0 = N_VGetArrayPointer(&y).expect("y data")[0];
        print!(
            "t = {}, x(t) = {}, E = {}, sol. err = {}\n",
            fmt_f(tret, 6),
            fmt_f(ydata0, 6),
            fmt_f(Energy(&y, dt, &udata), 6),
            fmt_f(0.0, 6)
        );
    }

    /* Do integration */
    iout = 0;
    while iout < num_output_times {
        if args.use_tstop != 0 {
            let _ = ARKodeSetStopTime(&arkode_mem, tout);
        }
        retval = ARKodeEvolve(&arkode_mem, tout, &y, &mut tret, ARK_NORMAL);

        /* Compute the analytical solution */
        Solution(tret, &y, &solution, &udata);

        /* Compute L2 error */
        N_VLinearSum(1.0, &y, -1.0, &solution, &solution);
        let err: sunrealtype = N_VDotProd(&solution, &solution).sqrt();

        /* Output current integration status */
        {
            let ydata0 = N_VGetArrayPointer(&y).expect("y data")[0];
            print!(
                "t = {}, x(t) = {}, E = {}, sol. err = {}\n",
                fmt_f(tret, 6),
                fmt_f(ydata0, 6),
                fmt_f(Energy(&y, dt, &udata), 6),
                fmt_e(err, 16)
            );
        }

        /* Check that solution error is within tolerance */
        if err
            > SUNMAX(
                dt / SUNRpowerR(10.0, (order - 2) as sunrealtype),
                1000.0 * SUN_UNIT_ROUNDOFF,
            )
        {
            eprint!("FAILURE: solution error is too high\n");
            std::process::exit(1);
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
    N_VDestroy(y);
    N_VDestroy(solution);
    drop(arkode_mem);
    let _ = ARKodePrintAllStats(
        arkode_mem_opt.as_ref().expect("arkode_mem"),
        &SUNFile::Stdout,
        SUNOutputFormat::SUN_OUTPUTFORMAT_TABLE,
    );
    ARKodeFree(&mut arkode_mem_opt);
    SUNContext_Free(&mut sunctx_opt);

    std::process::exit(0);
}

fn Solution(t: sunrealtype, _y: &N_Vector, solvec: &N_Vector, udata: &UserData) {
    let mut sol = N_VGetArrayPointer(solvec).expect("solvec data");

    /* compute solution */
    sol[0] = udata.A * (udata.omega * t + udata.phi).sun_cos();
    sol[1] = -udata.A * udata.omega * (udata.omega * t + udata.phi).sun_sin();
}

fn Energy(yvec: &N_Vector, _dt: sunrealtype, udata: &UserData) -> sunrealtype {
    let E: sunrealtype;
    let y = N_VGetArrayPointer(yvec).expect("y data");
    let x: sunrealtype = y[0];
    let v: sunrealtype = y[1];
    let omega2: sunrealtype = udata.omega * udata.omega;

    E = (v * v + omega2 * x * x) / 2.0;

    E
}

fn xdot(
    _t: sunrealtype,
    yvec: &N_Vector,
    ydotvec: &N_Vector,
    _user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    let y = N_VGetArrayPointer(yvec).expect("y data");
    let mut ydot = N_VGetArrayPointer(ydotvec).expect("ydot data");
    let v: sunrealtype = y[1];

    ydot[0] = v;

    0
}

fn vdot(
    _t: sunrealtype,
    yvec: &N_Vector,
    ydotvec: &N_Vector,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    let udata = user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<UserData>())
        .expect("user_data");
    let y = N_VGetArrayPointer(yvec).expect("y data");
    let mut ydot = N_VGetArrayPointer(ydotvec).expect("ydot data");
    let x: sunrealtype = y[0];
    let omega2: sunrealtype = udata.omega * udata.omega;

    ydot[1] = -omega2 * x;

    0
}

/*---- end of file ----*/
