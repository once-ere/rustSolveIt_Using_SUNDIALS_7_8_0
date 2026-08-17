/* clang-format off */
/* ----------------------------------------------------------------------------
 * Rust port of examples/arkode/C_serial/ark_kepler.c
 * Programmer(s): Cody J. Balos @ LLNL
 * ----------------------------------------------------------------------------
 * We consider the Kepler problem. We choose one body to be the center of our
 * coordinate system and then we use the coordinates q = (q1, q2) to represent
 * the position of the second body relative to the first (center). This yields
 * the ODE:
 *    dq/dt = [ p1 ]
 *            [ p2 ]
 *    dp/dt = [ -q1 / (q1^2 + q2^2)^(3/2) ]
 *          = [ -q2 / (q1^2 + q2^2)^(3/2) ]
 * with the initial conditions
 *    q(0) = [ 1 - e ],  p(0) = [        0          ]
 *           [   0   ]          [ sqrt((1+e)/(1-e)) ]
 * where e = 0.6 is the eccentricity.
 *
 * The Hamiltonian for the system,
 *    H(p,q) = 1/2 * (p1^2 + p2^2) - 1/sqrt(q1^2 + q2^2)
 * is conserved as well as the angular momentum,
 *    L(p,q) = q1*p2 - q2*p1.
 *
 * By default we solve the problem by letting y = [ q, p ]^T then using a 4th
 * order symplectic integrator via the SPRKStep time-stepper of ARKODE with a
 * fixed time-step size.
 *
 * The rootfinding feature of SPRKStep is used to count the number of complete
 * orbits. This is done by defining the function,
 *    g(q) = q2
 * and providing it to SPRKStep as the function to find the roots for g(q).
 *
 * The program also accepts command line arguments to change the method
 * used and time-stepping strategy. The program has the following CLI arguments:
 *
 *   --step-mode <fixed, adapt>  should we use a fixed time-step or adaptive time-step (default fixed)
 *   --stepper <SPRK, ERK>       should we use SPRKStep or ARKStep with an ERK method (default SPRK)
 *   --method <string>           which method to use (default ARKODE_SPRK_MCLACHLAN_4_4)
 *   --use-compensated-sums      turns on compensated summation in ARKODE where applicable
 *   --disable-tstop             turns off tstop mode
 *   --dt <Real>                 the fixed-time step size to use if fixed time stepping is turned on (default 0.01)
 *   --tf <Real>                 the final time for the simulation (default 100)
 *   --nout                      number of output times
 *   --count-orbits              use rootfinding to count the number of completed orbits
 *   --check-order               compute the order of the method used and check if it is within the expected range
 *
 * References:
 *    Ernst Hairer, Christain Lubich, Gerhard Wanner
 *    Geometric Numerical Integration: Structure-Preserving
 *    Algorithms for Ordinary Differential Equations
 *    Springer, 2006,
 *    ISSN 0179-3632
 * --------------------------------------------------------------------------*/
/* clang-format on */

#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

use arkode_rs::prelude::*;

use std::any::Any;

/* ark_kepler.h */

struct ProgramArgs {
    step_mode: i32,
    stepper: i32,
    num_output_times: i32,
    use_compsums: i32,
    use_tstop: i32,
    count_orbits: i32,
    check_order: i32,
    dt: sunrealtype,
    tf: sunrealtype,
    /* C `const char* method_name`, NULL until `ParseArgs` resolves a default */
    method_name: Option<String>,
}

fn ComputeConvergence(
    num_dt: i32,
    orders: &[sunrealtype],
    _expected_order: sunrealtype,
    a11: sunrealtype,
    a12: sunrealtype,
    a21: sunrealtype,
    a22: sunrealtype,
    b1: sunrealtype,
    b2: sunrealtype,
    ord_avg: &mut sunrealtype,
    ord_max: &mut sunrealtype,
    ord_est: &mut sunrealtype,
) -> i32 {
    /* Compute/print overall estimated convergence rate */
    let mut i: i32;
    let det: sunrealtype;
    *ord_avg = 0.0;
    *ord_max = 0.0;
    *ord_est = 0.0;
    i = 1;
    while i < num_dt {
        *ord_avg += orders[(i - 1) as usize];
        *ord_max = SUNMAX(*ord_max, orders[(i - 1) as usize]);
        i += 1;
    }
    *ord_avg = *ord_avg / (num_dt as sunrealtype - 1.0);
    det = a11 * a22 - a12 * a21;
    *ord_est = (a11 * b2 - a21 * b1) / det;
    0
}

fn PrintHelp() {
    eprint!(
        "ark_kepler: an ARKODE example demonstrating the SPRKStep \
         time-stepping module solving the Kepler problem\n"
    );
    /* clang-format off */
    eprint!("  --step-mode <fixed, adapt>  should we use a fixed time-step or adaptive time-step (default fixed)\n");
    eprint!("  --stepper <SPRK, ERK>       should we use SPRKStep or ARKStep with an ERK method (default SPRK)\n");
    eprint!(
        "  --method <string>           which method to use (default ARKODE_SPRK_MCLACHLAN_4_4)\n"
    );
    eprint!(
        "  --use-compensated-sums      turns on compensated summation in ARKODE where applicable\n"
    );
    eprint!("  --disable-tstop             turns off tstop mode\n");
    eprint!("  --dt <Real>                 the fixed-time step size to use if fixed time stepping is turned on (default 0.01)\n");
    eprint!("  --tf <Real>                 the final time for the simulation (default 100)\n");
    eprint!("  --nout <int>                the number of output times (default 100)\n");
    eprint!(
        "  --count-orbits              use rootfinding to count the number of completed orbits\n"
    );
    eprint!("  --check-order               compute the order of the method used and check if it is within range of the expected\n");
    /* clang-format on */
}

fn ParseArgs(argc: usize, argv: &[String], args: &mut ProgramArgs) -> i32 {
    let mut argi: usize;

    args.step_mode = 0;
    args.stepper = 0;
    args.method_name = None;
    args.count_orbits = 0;
    args.use_compsums = 0;
    args.use_tstop = 1;
    args.dt = 1e-2;
    args.tf = 100.0;
    args.check_order = 0;
    args.num_output_times = 50;

    argi = 1;
    while argi < argc {
        if argv[argi] == "--step-mode" {
            argi += 1;
            if argv[argi] == "fixed" {
                args.step_mode = 0;
            } else if argv[argi] == "adapt" {
                args.step_mode = 1;
            } else {
                eprint!("ERROR: --step-mode must be 'fixed' or 'adapt'\n");
                return 1;
            }
        } else if argv[argi] == "--stepper" {
            argi += 1;
            if argv[argi] == "SPRK" {
                args.stepper = 0;
            } else if argv[argi] == "ERK" {
                args.stepper = 1;
            } else {
                eprint!("ERROR: --stepper must be 'SPRK' or 'ERK'\n");
                return 1;
            }
        } else if argv[argi] == "--method" {
            argi += 1;
            args.method_name = Some(argv[argi].clone());
        } else if argv[argi] == "--dt" {
            argi += 1;
            args.dt = SUNStrToReal(&argv[argi]);
        } else if argv[argi] == "--tf" {
            argi += 1;
            args.tf = SUNStrToReal(&argv[argi]);
        } else if argv[argi] == "--nout" {
            argi += 1;
            args.num_output_times = atoi(&argv[argi]);
        } else if argv[argi] == "--count-orbits" {
            args.count_orbits = 1;
        } else if argv[argi] == "--disable-tstop" {
            args.use_tstop = 0;
        } else if argv[argi] == "--use-compensated-sums" {
            args.use_compsums = 1;
        } else if argv[argi] == "--check-order" {
            args.check_order = 1;
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

    if args.method_name.is_none() {
        if args.stepper == 0 {
            args.method_name = Some(String::from("ARKODE_SPRK_MCLACHLAN_4_4"));
        } else if args.stepper == 1 {
            args.method_name = Some(String::from("ARKODE_ZONNEVELD_5_3_4"));
        }
    }

    0
}

fn PrintArgs(args: &ProgramArgs) {
    print!("Problem Arguments:\n");
    print!("  stepper:              {}\n", args.stepper);
    print!("  step mode:            {}\n", args.step_mode);
    print!("  use tstop:            {}\n", args.use_tstop);
    print!("  use compensated sums: {}\n", args.use_compsums);
    print!("  dt:                   {}\n", fmt_g(args.dt, 6));
    print!("  Tf:                   {}\n", fmt_g(args.tf, 6));
    print!("  nout:                 {}\n\n", args.num_output_times);
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

/* ark_kepler.c */

const NUM_DT: usize = 8;

struct UserData {
    /* set from the problem parameters; the callbacks never read it back,
    exactly as in C */
    #[allow(dead_code)]
    ecc: sunrealtype,
}

struct ProblemResult {
    sol: N_Vector,
    energy_error: sunrealtype,
    /* C leaves this field uninitialized and never reads it */
    #[allow(dead_code)]
    method_order: i32,
}

fn SolveProblem(args: &ProgramArgs, result: &mut ProblemResult, sunctx: &SUNContext) -> i32 {
    let mut arkode_mem_opt: Option<ARKodeMem> = None;
    /* C declares `SUNNonlinearSolver NLS = NULL;` and only ever tests it for
    NULL at the end, so the port omits it. */
    let mut num_orbits: sunrealtype = 0.0;
    let mut rootsfound: [i32; 1] = [0];
    let mut iout: i32 = 0;
    let mut retval: i32 = 0;

    let count_orbits: i32 = args.count_orbits;
    let step_mode: i32 = args.step_mode;
    let stepper: i32 = args.stepper;
    let use_compsums: i32 = args.use_compsums;
    let num_output_times: i32 = args.num_output_times;
    let method_name: &str = args.method_name.as_deref().expect("method_name");
    let dt: sunrealtype = args.dt;
    let Tf: sunrealtype = args.tf;

    /* Default problem parameters */
    let T0: sunrealtype = 0.0;
    let dTout: sunrealtype = (Tf - T0) / (num_output_times as sunrealtype);
    let ecc: sunrealtype = 0.6;

    print!("\n   Begin Kepler Problem\n\n");
    PrintArgs(args);

    /* Allocate and fill udata structure */
    let udata = UserData { ecc };

    /* Allocate our state vector */
    let y = N_VNew_Serial(4, sunctx).expect("N_VNew_Serial");

    /* Fill the initial conditions */
    InitialConditions(&y, ecc);

    /* Create SPRKStep integrator */
    if stepper == 0 {
        arkode_mem_opt = SPRKStepCreate(force, velocity, T0, &y, sunctx);
        let arkode_mem = arkode_mem_opt.as_ref().expect("arkode_mem");

        /* Optional: enable temporal root-finding */
        if count_orbits != 0 {
            /* C discards the ARKodeRootInit flag and checks the stale
            `retval` instead; reproduced verbatim. */
            let _ = ARKodeRootInit(arkode_mem, 1, Some(rootfn));
            if check_retval(Some(retval), "ARKodeRootInit", 1) != 0 {
                return 1;
            }
        }

        retval = SPRKStepSetMethodName(arkode_mem, method_name);
        if check_retval(Some(retval), "SPRKStepSetMethodName", 1) != 0 {
            return 1;
        }

        retval = ARKodeSetUseCompensatedSums(arkode_mem, use_compsums != 0);
        if check_retval(Some(retval), "ARKodeSetUseCompensatedSums", 1) != 0 {
            return 1;
        }

        if step_mode == 0 {
            retval = ARKodeSetFixedStep(arkode_mem, dt);
            if check_retval(Some(retval), "ARKodeSetFixedStep", 1) != 0 {
                return 1;
            }

            retval = ARKodeSetMaxNumSteps(arkode_mem, ((Tf / dt).ceil() as i64) + 1);
            if check_retval(Some(retval), "ARKodeSetMaxNumSteps", 1) != 0 {
                return 1;
            }
        } else {
            eprint!("ERROR: adaptive time-steps are not supported with SPRKStep\n");
            return 1;
        }

        retval = ARKodeSetUserData(arkode_mem, Some(Box::new(UserData { ecc: udata.ecc })));
        if check_retval(Some(retval), "ARKodeSetUserData", 1) != 0 {
            return 1;
        }
    } else if stepper == 1 {
        arkode_mem_opt = ARKStepCreate(Some(dydt), None, T0, &y, sunctx);
        let arkode_mem = arkode_mem_opt.as_ref().expect("arkode_mem");

        retval = ARKStepSetTableName(arkode_mem, "ARKODE_DIRK_NONE", method_name);
        if check_retval(Some(retval), "ARKStepSetTableName", 1) != 0 {
            return 1;
        }

        if count_orbits != 0 {
            /* see the SPRKStep branch: C checks the stale `retval` here */
            let _ = ARKodeRootInit(arkode_mem, 1, Some(rootfn));
            if check_retval(Some(retval), "ARKodeRootInit", 1) != 0 {
                return 1;
            }
        }

        retval = ARKodeSetUserData(arkode_mem, Some(Box::new(UserData { ecc: udata.ecc })));
        if check_retval(Some(retval), "ARKodeSetUserData", 1) != 0 {
            return 1;
        }

        retval = ARKodeSetMaxNumSteps(arkode_mem, ((Tf / dt).ceil() as i64) + 1);
        if check_retval(Some(retval), "ARKodeSetMaxNumSteps", 1) != 0 {
            return 1;
        }

        if step_mode == 0 {
            /* C assigns the flag to `retval` and never reads it back */
            let _ = ARKodeSetFixedStep(arkode_mem, dt);
        } else {
            retval = ARKodeSStolerances(arkode_mem, dt, dt);
            if check_retval(Some(retval), "ARKodeSStolerances", 1) != 0 {
                return 1;
            }
        }
    }

    let arkode_mem = arkode_mem_opt.as_ref().expect("arkode_mem").clone();

    /* Open output files */
    let conserved_fp: SUNFile;
    let solution_fp: SUNFile;
    let times_fp: SUNFile;
    if stepper == 0 {
        let fmt1 = "ark_kepler_conserved_%s-dt-%.2e.txt";
        let fmt2 = "ark_kepler_solution_%s-dt-%.2e.txt";
        let fmt3 = "ark_kepler_times_%s-dt-%.2e.txt";
        let mut fname: String;
        fname = sprintf_name(fmt1, method_name, dt);
        conserved_fp = SUNFile::fopen(&fname, "w+");
        fname = sprintf_name(fmt2, method_name, dt);
        solution_fp = SUNFile::fopen(&fname, "w+");
        fname = sprintf_name(fmt3, method_name, dt);
        times_fp = SUNFile::fopen(&fname, "w+");
    } else {
        let fmt1 = "ark_kepler_conserved_%s-dt-%.2e.txt";
        let fmt2 = "ark_kepler_solution_%s-dt-%.2e.txt";
        let fmt3 = "ark_kepler_times_%s-dt-%.2e.txt";
        let mut fname: String;
        fname = sprintf_name(fmt1, method_name, dt);
        conserved_fp = SUNFile::fopen(&fname, "w+");
        fname = sprintf_name(fmt2, method_name, dt);
        solution_fp = SUNFile::fopen(&fname, "w+");
        fname = sprintf_name(fmt3, method_name, dt);
        times_fp = SUNFile::fopen(&fname, "w+");
    }

    /* Print out starting energy, momentum before integrating */
    let mut tret: sunrealtype = T0;
    let mut tout: sunrealtype = T0 + dTout;
    let H0: sunrealtype = Hamiltonian(&y);
    let L0: sunrealtype = AngularMomentum(&y);
    print!(
        "t = {}, H(p,q) = {}, L(p,q) = {}\n",
        fmt_f(tret, 4),
        fmt_f(H0, 16),
        fmt_f(L0, 16)
    );
    times_fp.write_str(&format!("{}\n", fmt_f(tret, 16)));
    conserved_fp.write_str(&format!("{}, {}\n", fmt_f(H0, 16), fmt_f(L0, 16)));
    N_VPrintFile(&y, &solution_fp);

    /* Do integration */
    if stepper == 0 {
        while iout < num_output_times {
            /* Optional: if the stop time is not set, then its possible that the
            exact requested output time will not be hit (even with a fixed
            time-step due to roundoff error accumulation) and interpolation will
            be used to get the solution at the output time. */
            if args.use_tstop != 0 {
                let _ = ARKodeSetStopTime(&arkode_mem, tout);
            }
            retval = ARKodeEvolve(&arkode_mem, tout, &y, &mut tret, ARK_NORMAL);

            if retval == ARK_ROOT_RETURN {
                num_orbits += 0.5;

                print!("ROOT RETURN:\t");
                let _ = ARKodeGetRootInfo(&arkode_mem, &mut rootsfound);
                let (y0, y1) = {
                    let ydata = N_VGetArrayPointer(&y).expect("y data");
                    (ydata[0], ydata[1])
                };
                print!(
                    "  g[0] = {:>3}, y[0] = {}, y[1] = {}, num. orbits is now {}\n",
                    rootsfound[0],
                    fmt_gw(y0, 3, 6),
                    fmt_gw(y1, 3, 6),
                    fmt_f(num_orbits, 2)
                );
                print!(
                    "t = {}, H(p,q)-H0 = {}, L(p,q)-L0 = {}\n",
                    fmt_f(tret, 4),
                    fmt_e(Hamiltonian(&y) - H0, 16),
                    fmt_e(AngularMomentum(&y) - L0, 16)
                );
            } else if retval >= 0 {
                /* Output current integration status */
                print!(
                    "t = {}, H(p,q)-H0 = {}, L(p,q)-L0 = {}\n",
                    fmt_f(tret, 4),
                    fmt_e(Hamiltonian(&y) - H0, 16),
                    fmt_e(AngularMomentum(&y) - L0, 16)
                );
                times_fp.write_str(&format!("{}\n", fmt_f(tret, 16)));
                conserved_fp.write_str(&format!(
                    "{}, {}\n",
                    fmt_f(Hamiltonian(&y), 16),
                    fmt_f(AngularMomentum(&y), 16)
                ));

                N_VPrintFile(&y, &solution_fp);

                tout += dTout;
                tout = if tout > Tf { Tf } else { tout };
                iout += 1;
            } else {
                eprint!("Solver failure, stopping integration\n");
                break;
            }
        }
    } else {
        while iout < num_output_times {
            /* Optional: if the stop time is not set, then its possible that the
            the exact requested output time will not be hit (even with a fixed
            time-step due to roundoff error accumulation) and interpolation will
            be used to get the solution at the output time. */
            if args.use_tstop != 0 {
                let _ = ARKodeSetStopTime(&arkode_mem, tout);
            }
            retval = ARKodeEvolve(&arkode_mem, tout, &y, &mut tret, ARK_NORMAL);

            if retval == ARK_ROOT_RETURN {
                num_orbits += 0.5;

                print!("ROOT RETURN:\t");
                let _ = ARKodeGetRootInfo(&arkode_mem, &mut rootsfound);
                let (y0, y1) = {
                    let ydata = N_VGetArrayPointer(&y).expect("y data");
                    (ydata[0], ydata[1])
                };
                print!(
                    "  g[0] = {:>3}, y[0] = {}, y[1] = {}, num. orbits is now {}\n",
                    rootsfound[0],
                    fmt_gw(y0, 3, 6),
                    fmt_gw(y1, 3, 6),
                    fmt_f(num_orbits, 2)
                );
                print!(
                    "t = {}, H(p,q)-H0 = {}, L(p,q)-L0 = {}\n",
                    fmt_f(tret, 4),
                    fmt_e(Hamiltonian(&y) - H0, 16),
                    fmt_e(AngularMomentum(&y) - L0, 16)
                );
            } else if retval >= 0 {
                /* Output current integration status */
                print!(
                    "t = {}, H(p,q)-H0 = {}, L(p,q)-L0 = {}\n",
                    fmt_f(tret, 4),
                    fmt_e(Hamiltonian(&y) - H0, 16),
                    fmt_e(AngularMomentum(&y) - L0, 16)
                );
                times_fp.write_str(&format!("{}\n", fmt_f(tret, 16)));
                conserved_fp.write_str(&format!(
                    "{}, {}\n",
                    fmt_f(Hamiltonian(&y), 16),
                    fmt_f(AngularMomentum(&y), 16)
                ));

                N_VPrintFile(&y, &solution_fp);

                tout += dTout;
                tout = if tout > Tf { Tf } else { tout };
                iout += 1;
            } else {
                eprint!("Solver failure, stopping integration\n");
                break;
            }
        }
    }

    /* Copy results */
    N_VScale(1.0, &y, &result.sol);
    result.energy_error = Hamiltonian(&y) - H0;

    drop(udata);
    drop(times_fp);
    drop(conserved_fp);
    drop(solution_fp);
    N_VDestroy(y);
    drop(arkode_mem);
    let _ = ARKodePrintAllStats(
        arkode_mem_opt.as_ref().expect("arkode_mem"),
        &SUNFile::Stdout,
        SUNOutputFormat::SUN_OUTPUTFORMAT_TABLE,
    );
    ARKodeFree(&mut arkode_mem_opt);
    0
}

/* C `sprintf(fname, "ark_kepler_..._%s-dt-%.2e.txt", method_name, dt)` */
fn sprintf_name(fmt: &str, method_name: &str, dt: sunrealtype) -> String {
    fmt.replace("%s", method_name)
        .replace("%.2e", &fmt_e(dt, 2))
}

fn InitialConditions(y0vec: &N_Vector, ecc: sunrealtype) {
    let zero: sunrealtype = 0.0;
    let one: sunrealtype = 1.0;
    let mut y0 = N_VGetArrayPointer(y0vec).expect("y0 data");

    y0[0] = one - ecc;
    y0[1] = zero;
    y0[2] = zero;
    y0[3] = SUNRsqrt((one + ecc) / (one - ecc));
}

fn Hamiltonian(yvec: &N_Vector) -> sunrealtype {
    let H: sunrealtype;
    let y = N_VGetArrayPointer(yvec).expect("y data");
    let sqrt_qTq: sunrealtype = SUNRsqrt(y[0] * y[0] + y[1] * y[1]);
    let pTp: sunrealtype = y[2] * y[2] + y[3] * y[3];

    H = 0.5 * pTp - 1.0 / sqrt_qTq;

    H
}

fn AngularMomentum(yvec: &N_Vector) -> sunrealtype {
    let L: sunrealtype;
    let y = N_VGetArrayPointer(yvec).expect("y data");
    let q1: sunrealtype = y[0];
    let q2: sunrealtype = y[1];
    let p1: sunrealtype = y[2];
    let p2: sunrealtype = y[3];

    L = q1 * p2 - q2 * p1;

    L
}

fn dydt(
    t: sunrealtype,
    yvec: &N_Vector,
    ydotvec: &N_Vector,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    let mut retval: i32 = 0;

    retval += force(t, yvec, ydotvec, user_data);
    retval += velocity(t, yvec, ydotvec, user_data);

    retval
}

fn velocity(
    _t: sunrealtype,
    yvec: &N_Vector,
    ydotvec: &N_Vector,
    _user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    let y = N_VGetArrayPointer(yvec).expect("y data");
    let mut ydot = N_VGetArrayPointer(ydotvec).expect("ydot data");
    let p1: sunrealtype = y[2];
    let p2: sunrealtype = y[3];

    ydot[0] = p1;
    ydot[1] = p2;

    0
}

fn force(
    _t: sunrealtype,
    yvec: &N_Vector,
    ydotvec: &N_Vector,
    _user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    let y = N_VGetArrayPointer(yvec).expect("y data");
    let mut ydot = N_VGetArrayPointer(ydotvec).expect("ydot data");
    let q1: sunrealtype = y[0];
    let q2: sunrealtype = y[1];
    let sqrt_qTq: sunrealtype = SUNRsqrt(q1 * q1 + q2 * q2);

    ydot[2] = -q1 / SUNRpowerR(sqrt_qTq, 3.0);
    ydot[3] = -q2 / SUNRpowerR(sqrt_qTq, 3.0);

    0
}

fn rootfn(
    _t: sunrealtype,
    yvec: &N_Vector,
    gout: &mut [sunrealtype],
    _user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    let y = N_VGetArrayPointer(yvec).expect("y data");
    let q2: sunrealtype = y[1];

    gout[0] = q2;

    0
}

fn main() {
    let mut args = ProgramArgs {
        step_mode: 0,
        stepper: 0,
        num_output_times: 0,
        use_compsums: 0,
        use_tstop: 0,
        count_orbits: 0,
        check_order: 0,
        dt: 0.0,
        tf: 0.0,
        method_name: None,
    };
    let mut sunctx_opt: Option<SUNContext> = None;
    let mut retval: i32;

    /* Create the SUNDIALS context object for this simulation */
    retval = SUNContext_Create(SUN_COMM_NULL, &mut sunctx_opt);
    if check_retval(Some(retval), "SUNContext_Create", 1) != 0 {
        std::process::exit(1);
    }
    let sunctx = sunctx_opt.as_ref().expect("sunctx").clone();

    /* Parse the command line arguments */
    let argv: Vec<String> = std::env::args().collect();
    let argc: usize = argv.len();
    if ParseArgs(argc, &argv, &mut args) != 0 {
        std::process::exit(1);
    };

    /* Allocate space for result variables */
    let mut result = ProblemResult {
        sol: N_VNew_Serial(4, &sunctx).expect("N_VNew_Serial"),
        energy_error: 0.0,
        method_order: 0,
    };

    if args.check_order == 0 {
        /* SolveProblem calls a stepper to evolve the problem to Tf */
        retval = SolveProblem(&args, &mut result, &sunctx);
        if check_retval(Some(retval), "SolveProblem", 1) != 0 {
            std::process::exit(1);
        }
    } else {
        let mut i: usize;
        /* Compute the order of accuracy of the method by testing
        it with different step sizes. */
        let mut acc_orders: [sunrealtype; NUM_DT] = [0.0; NUM_DT];
        let mut con_orders: [sunrealtype; NUM_DT] = [0.0; NUM_DT];
        let mut acc_errors: [sunrealtype; NUM_DT] = [0.0; NUM_DT];
        let mut con_errors: [sunrealtype; NUM_DT] = [0.0; NUM_DT];
        let method = ARKodeSPRKTable_LoadByName(args.method_name.as_deref().expect("method_name"));
        let expected_order: i32 = method.as_ref().expect("method").borrow().q;
        let ref_sol = N_VClone(&result.sol).expect("N_VClone");
        let error = N_VClone(&result.sol).expect("N_VClone");
        let mut a11: sunrealtype = 0.0;
        let mut a12: sunrealtype = 0.0;
        let mut a21: sunrealtype = 0.0;
        let mut a22: sunrealtype = 0.0;
        let mut b1: sunrealtype = 0.0;
        let mut b2: sunrealtype = 0.0;
        let mut b1e: sunrealtype = 0.0;
        let mut b2e: sunrealtype = 0.0;
        let mut ord_max_acc: sunrealtype = 0.0;
        let mut ord_max_conv: sunrealtype = 0.0;
        let mut ord_avg: sunrealtype = 0.0;
        let mut ord_est: sunrealtype = 0.0;
        let refine: sunrealtype = 0.5;
        let dt: sunrealtype = if expected_order >= 3 { 1e-1 } else { 1e-3 };
        let mut dts: [sunrealtype; NUM_DT] = [0.0; NUM_DT];

        /* Create a reference solution using 8th order ERK with a small time step */
        let old_step_mode: i32 = args.step_mode;
        let old_stepper: i32 = args.stepper;
        let old_method_name: Option<String> = args.method_name.clone();
        args.dt = 1e-3;
        args.step_mode = 0;
        args.stepper = 1;
        args.method_name = Some(String::from("ARKODE_ARK548L2SAb_ERK_8_4_5"));

        /* Free method, we just needed it to get its order */
        ARKodeSPRKTable_Free(method);

        /* SolveProblem calls a stepper to evolve the problem to Tf */
        retval = SolveProblem(&args, &mut result, &sunctx);
        if check_retval(Some(retval), "SolveProblem", 1) != 0 {
            std::process::exit(1);
        }

        /* Store the reference solution */
        N_VScale(1.0, &result.sol, &ref_sol);

        /* Restore the program args */
        args.step_mode = old_step_mode;
        args.stepper = old_stepper;
        args.method_name = old_method_name;

        i = 0;
        while i < NUM_DT {
            dts[i] = dt * SUNRpowerR(refine, i as sunrealtype);
            i += 1;
        }

        /* Compute the error with various step sizes */
        i = 0;
        while i < NUM_DT {
            /* Set the dt to use for this solve */
            args.dt = dts[i];

            /* SolveProblem calls a stepper to evolve the problem to Tf */
            retval = SolveProblem(&args, &mut result, &sunctx);
            if check_retval(Some(retval), "SolveProblem", 1) != 0 {
                std::process::exit(1);
            }

            print!("\n");

            /* Compute the error */
            N_VLinearSum(1.0, &result.sol, -1.0, &ref_sol, &error);
            acc_errors[i] =
                SUNRsqrt(N_VDotProd(&error, &error)) / (N_VGetLength(&error) as sunrealtype);
            con_errors[i] = SUNRabs(result.energy_error);

            a11 += 1.0;
            a12 += dts[i].sun_ln();
            a21 += dts[i].sun_ln();
            a22 += dts[i].sun_ln() * dts[i].sun_ln();
            b1 += acc_errors[i].sun_ln();
            b2 += acc_errors[i].sun_ln() * dts[i].sun_ln();
            b1e += con_errors[i].sun_ln();
            b2e += con_errors[i].sun_ln() * dts[i].sun_ln();

            if i >= 1 {
                acc_orders[i - 1] =
                    (acc_errors[i] / acc_errors[i - 1]).sun_ln() / (dts[i] / dts[i - 1]).sun_ln();
                con_orders[i - 1] =
                    (con_errors[i] / con_errors[i - 1]).sun_ln() / (dts[i] / dts[i - 1]).sun_ln();
            }

            i += 1;
        }

        /* Compute the order of accuracy */
        let _ = ComputeConvergence(
            NUM_DT as i32,
            &acc_orders,
            expected_order as sunrealtype,
            a11,
            a12,
            a21,
            a22,
            b1,
            b2,
            &mut ord_avg,
            &mut ord_max_acc,
            &mut ord_est,
        );
        print!(
            "Order of accuracy wrt solution:    expected = {}, max = {},  avg = {},  overall = {}\n",
            expected_order,
            fmt_f(ord_max_acc, 4),
            fmt_f(ord_avg, 4),
            fmt_f(ord_est, 4)
        );

        /* Compute the order of accuracy with respect to conservation */
        let _ = ComputeConvergence(
            NUM_DT as i32,
            &con_orders,
            expected_order as sunrealtype,
            a11,
            a12,
            a21,
            a22,
            b1e,
            b2e,
            &mut ord_avg,
            &mut ord_max_conv,
            &mut ord_est,
        );

        print!(
            "Order of accuracy wrt Hamiltonian: expected = {}, max = {},  avg = {},  overall = {}\n",
            expected_order,
            fmt_f(ord_max_conv, 4),
            fmt_f(ord_avg, 4),
            fmt_f(ord_est, 4)
        );

        if ord_max_acc < (expected_order as sunrealtype - 0.5) {
            print!(
                ">>> FAILURE: computed order of accuracy wrt solution is below expected ({})\n",
                expected_order
            );
            std::process::exit(1);
        }

        if ord_max_conv < (expected_order as sunrealtype - 0.5) {
            print!(
                ">>> FAILURE: computed order of accuracy wrt Hamiltonian is below expected ({})\n",
                expected_order
            );
            std::process::exit(1);
        }

        N_VDestroy(ref_sol);
        N_VDestroy(error);
    }

    N_VDestroy(result.sol);
    drop(sunctx);
    SUNContext_Free(&mut sunctx_opt);

    std::process::exit(0);
}

/*---- end of file ----*/
