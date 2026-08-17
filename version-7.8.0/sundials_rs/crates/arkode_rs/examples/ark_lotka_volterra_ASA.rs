/* ------------------------------------------------------------------
 * Programmer(s): Cody J. Balos @ LLNL
 * ------------------------------------------------------------------
 * Rust port of examples/arkode/C_serial/ark_lotka_volterra_ASA.c
 * -----------------------------------------------------------------------------
 * This example solves the Lotka-Volterra ODE with four parameters,
 *
 *     u = [dx/dt] = [ p_0*x - p_1*x*y  ]
 *         [dy/dt]   [ -p_2*y + p_3*x*y ].
 *
 * The initial condition is u(t_0) = 1.0 and we use the parameters
 * p  = [1.5, 1.0, 3.0, 1.0]. The integration interval can be controlled via
 * the --tf command line argument, but by default it is t \in [0, 10.].
 * An explicit Runge--Kutta method is employed via the ARKStep time stepper
 * provided by ARKODE. After solving the forward problem, adjoint sensitivity
 * analysis (ASA) is performed using the discrete adjoint method available with
 * with ARKStep in order to obtain the gradient of the scalar cost function,
 *
 *    g(u(t_f), p) = || 1 - u(t_f, p) ||^2 / 2
 *
 * with respect to the initial condition and the parameters.
 *
 * ./ark_lotka_volterra_adj options:
 * --tf <real>         the final simulation time
 * --dt <real>         the timestep size
 * --order <int>       the order of the RK method
 * --check-freq <int>  how often to checkpoint (in steps)
 * --dont-keep         don't keep checkpoints around after loading
 * --help              print these options
 * ---------------------------------------------------------------------------*/
#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

use std::any::Any;

use arkode_rs::prelude::*;

use arkode_rs::sunadjointcheckpointscheme_fixed::SUNAdjointCheckpointScheme_Create_Fixed;
use arkode_rs::sundials_adjointcheckpointscheme::{
    SUNAdjointCheckpointScheme, SUNAdjointCheckpointScheme_Destroy,
};
use arkode_rs::sundials_adjointstepper::{
    SUNAdjointStepper, SUNAdjointStepper_Destroy, SUNAdjointStepper_Evolve,
    SUNAdjointStepper_PrintAllStats, SUNAdjointStepper_SetUserData,
};
use arkode_rs::sundials_memory::SUNMemoryHelper_Destroy;
use arkode_rs::sundials_system_memory::SUNMemoryHelper_Sys;

/* `nvector_manyvector` is not re-exported by `arkode_rs` (see the api_gaps
 * note); `sundials_core` is a direct dependency of the crate, so the example
 * reaches the module there. */
use sundials_core::nvector_manyvector::{N_VGetSubvector_ManyVector, N_VNew_ManyVector};

struct ProgramArgs {
    tf: sunrealtype,
    dt: sunrealtype,
    order: i32,
    check_freq: i32,
    keep_checks: sunbooleantype,
}

static params: [sunrealtype; 4] = [1.5, 1.0, 3.0, 1.0];

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

/* C `atof` is `strtod` with the errno reporting dropped — the workspace's
 * `SUNStrToReal` has exactly those semantics. */
fn atof(s: &str) -> sunrealtype {
    SUNStrToReal(s)
}

/* C `ceil` */
fn ceil(x: sunrealtype) -> sunrealtype {
    x.ceil()
}

/* The forward RHS and the adjoint terminal condition read the parameter array
 * through `user_data`; a `Box<dyn Any>` holds an owned copy of the C static
 * array (read-only in every callback, exactly as in C). */
fn params_of(user_data: &mut Option<Box<dyn Any>>) -> [sunrealtype; 4] {
    user_data
        .as_ref()
        .and_then(|b| b.downcast_ref::<[sunrealtype; 4]>())
        .copied()
        .expect("params user_data")
}

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let argc: i32 = argv.len() as i32;

    /* C: `int retval = 0;` — the initializer is dead (every read is preceded
    by an assignment), and keeping it would be an `unused_assignments`
    warning. */
    let mut retval: i32;
    let mut sunctx_opt: Option<SUNContext> = None;
    let _ = SUNContext_Create(SUN_COMM_NULL, &mut sunctx_opt);
    let sunctx = sunctx_opt.as_ref().expect("SUNContext").clone();

    let mut args = ProgramArgs {
        tf: 10.0,
        dt: 1e-3,
        order: 4,
        check_freq: 2,
        keep_checks: SUNTRUE,
    };
    parse_args(argc, &argv, &mut args);

    //
    // Create the initial conditions vector
    //

    let neq: sunindextype = 2;
    let u = N_VNew_Serial(neq, &sunctx).expect("N_VNew_Serial");
    let u0 = N_VClone(&u).expect("N_VClone");
    N_VConst(1.0, &u0);
    N_VConst(1.0, &u);

    //
    // Create the ARKODE stepper that will be used for the forward evolution.
    //

    let dt: sunrealtype = args.dt;
    let t0: sunrealtype = 0.0;
    let tf: sunrealtype = args.tf;
    let nsteps: i32 = ceil((tf - t0) / dt) as i32;
    let order: i32 = args.order;
    let mut arkode_mem = ARKStepCreate(Some(lotka_volterra), None, t0, &u, &sunctx);

    retval = ARKodeSetOrder(arkode_mem.as_ref().expect("arkode_mem"), order);
    if check_retval(retval, "ARKodeSetOrder", 1) != 0 {
        std::process::exit(1);
    }

    // Due to roundoff in the `t` accumulation within the integrator,
    // the integrator may actually use nsteps + 1 time steps to reach tf.
    retval = ARKodeSetMaxNumSteps(
        arkode_mem.as_ref().expect("arkode_mem"),
        (nsteps + 1) as i64,
    );
    if check_retval(retval, "ARKodeSetMaxNumSteps", 1) != 0 {
        std::process::exit(1);
    }

    // Enable checkpointing during the forward solution.
    let check_interval: i32 = args.check_freq;
    let ncheck: i32 = nsteps * order;
    let keep_check: sunbooleantype = args.keep_checks;
    let mut checkpoint_scheme: Option<SUNAdjointCheckpointScheme> = None;
    let mem_helper = SUNMemoryHelper_Sys(&sunctx).expect("SUNMemoryHelper_Sys");

    retval = SUNAdjointCheckpointScheme_Create_Fixed(
        SUNDataIOMode::SUNDATAIOMODE_INMEM,
        &mem_helper,
        check_interval as suncountertype,
        ncheck as suncountertype,
        keep_check,
        &sunctx,
        &mut checkpoint_scheme,
    );
    if check_retval(retval, "SUNAdjointCheckpointScheme_Create_Fixed", 1) != 0 {
        std::process::exit(1);
    }

    retval = ARKodeSetAdjointCheckpointScheme(
        arkode_mem.as_ref().expect("arkode_mem"),
        checkpoint_scheme.as_ref(),
    );
    if check_retval(retval, "ARKodeSetAdjointCheckpointScheme", 1) != 0 {
        std::process::exit(1);
    }

    //
    // Compute the forward solution
    //

    print!("Initial condition:\n");
    N_VPrint(&u);

    retval = ARKodeSetUserData(
        arkode_mem.as_ref().expect("arkode_mem"),
        Some(Box::new(params)),
    );
    if check_retval(retval, "ARKodeSetUserData", 1) != 0 {
        std::process::exit(1);
    }

    retval = ARKodeSetFixedStep(arkode_mem.as_ref().expect("arkode_mem"), dt);
    if check_retval(retval, "ARKodeSetFixedStep", 1) != 0 {
        std::process::exit(1);
    }

    let mut tret: sunrealtype = t0;

    retval = ARKodeEvolve(
        arkode_mem.as_ref().expect("arkode_mem"),
        tf,
        &u,
        &mut tret,
        ARK_NORMAL,
    );
    if check_retval(retval, "ARKodeEvolve", 1) != 0 {
        std::process::exit(1);
    }

    print!("Forward Solution:\n");
    N_VPrint(&u);

    print!("ARKODE Stats for Forward Solution:\n");
    retval = ARKodePrintAllStats(
        arkode_mem.as_ref().expect("arkode_mem"),
        &SUNFile::Stdout,
        SUNOutputFormat::SUN_OUTPUTFORMAT_TABLE,
    );
    if check_retval(retval, "ARKodePrintAllStats", 1) != 0 {
        std::process::exit(1);
    }
    print!("\n");

    //
    // Create the adjoint stepper
    //

    let num_params: sunindextype = 4;
    let sensu0 = N_VClone(&u).expect("N_VClone");
    let sensp = N_VNew_Serial(num_params, &sunctx).expect("N_VNew_Serial");
    let sens: [N_Vector; 2] = [sensu0.clone(), sensp.clone()];
    let sf = N_VNew_ManyVector(2, &sens, &sunctx).expect("N_VNew_ManyVector");

    // Set the terminal condition for the adjoint system, which
    // should be the the gradient of our cost function at tf.
    dgdu(&u, &sensu0, &params);
    dgdp(&u, &sensp, &params);

    print!("Adjoint terminal condition:\n");
    N_VPrint(&sf);

    let mut adj_stepper: Option<SUNAdjointStepper> = None;
    retval = ARKStepCreateAdjointStepper(
        arkode_mem.as_ref().expect("arkode_mem"),
        Some(adj_rhs),
        None,
        tf,
        &sf,
        &sunctx,
        &mut adj_stepper,
    );
    if check_retval(retval, "ARKStepCreateAdjointStepper", 1) != 0 {
        std::process::exit(1);
    }

    /* C's `ARKStepCreateAdjointStepper` ends with
     * `SUNAdjointStepper_SetUserData(*adj_stepper_ptr, ark_mem->user_data)`,
     * ALIASING the forward integrator's `user_data` pointer. A `Box` cannot
     * alias (accepted deviation class 6), so the port leaves the adjoint
     * stepper's token NULL and the example hands it its own copy of the same
     * read-only parameter array here. */
    let _ = SUNAdjointStepper_SetUserData(
        adj_stepper.as_ref().expect("adj_stepper"),
        Some(Box::new(params)),
    );

    //
    // Now compute the adjoint solution
    //

    retval = SUNAdjointStepper_Evolve(
        adj_stepper.as_ref().expect("adj_stepper"),
        t0,
        &sf,
        &mut tret,
    );
    if check_retval(retval, "SUNAdjointStepper_Evolve", 1) != 0 {
        std::process::exit(1);
    }

    print!("Adjoint Solution:\n");
    N_VPrint(&sf);

    print!("\nSUNAdjointStepper Stats:\n");
    retval = SUNAdjointStepper_PrintAllStats(
        adj_stepper.as_ref().expect("adj_stepper"),
        &SUNFile::Stdout,
        SUNOutputFormat::SUN_OUTPUTFORMAT_TABLE,
    );
    if check_retval(retval, "SUNAdjointStepper_PrintAllStats", 1) != 0 {
        std::process::exit(1);
    }
    print!("\n");

    //
    // Cleanup
    //

    drop(sens);
    N_VDestroy(sensu0);
    N_VDestroy(sensp);
    N_VDestroy(sf);
    N_VDestroy(u);
    N_VDestroy(u0);
    let _ = SUNAdjointCheckpointScheme_Destroy(&mut checkpoint_scheme);
    let _ = SUNAdjointStepper_Destroy(&mut adj_stepper);
    ARKodeFree(&mut arkode_mem);
    let _ = SUNMemoryHelper_Destroy(mem_helper);
    drop(sunctx);
    let _ = SUNContext_Free(&mut sunctx_opt);
}

fn lotka_volterra(
    _t: sunrealtype,
    uvec: &N_Vector,
    udotvec: &N_Vector,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    let p = params_of(user_data);
    let u = N_VGetArrayPointer(uvec).expect("N_VGetArrayPointer");
    let mut udot = N_VGetArrayPointer(udotvec).expect("N_VGetArrayPointer");

    udot[0] = p[0] * u[0] - p[1] * u[0] * u[1];
    udot[1] = -p[2] * u[1] + p[3] * u[0] * u[1];

    0
}

fn vjp(
    vvec: &N_Vector,
    Jvvec: &N_Vector,
    _t: sunrealtype,
    uvec: &N_Vector,
    _udotvec: Option<&N_Vector>,
    user_data: &mut Option<Box<dyn Any>>,
    _tmp: Option<&N_Vector>,
) -> i32 {
    let p = params_of(user_data);
    let u = N_VGetArrayPointer(uvec).expect("N_VGetArrayPointer");
    let v = N_VGetArrayPointer(vvec).expect("N_VGetArrayPointer");
    let mut Jv = N_VGetArrayPointer(Jvvec).expect("N_VGetArrayPointer");

    Jv[0] = (p[0] - p[1] * u[1]) * v[0] + p[3] * u[1] * v[1];
    Jv[1] = -p[1] * u[0] * v[0] + (-p[2] + p[3] * u[0]) * v[1];

    0
}

fn parameter_vjp(
    vvec: &N_Vector,
    Jvvec: &N_Vector,
    _t: sunrealtype,
    uvec: &N_Vector,
    _udotvec: Option<&N_Vector>,
    user_data: &mut Option<Box<dyn Any>>,
    _tmp: Option<&N_Vector>,
) -> i32 {
    /* C: `if (user_data != params) { return -1; }` — a pointer identity test
     * against the static parameter array. The owned `Box<dyn Any>` token
     * cannot carry that identity, so the equivalent check is that the token
     * is the parameter array (right type, same values). */
    match user_data
        .as_ref()
        .and_then(|b| b.downcast_ref::<[sunrealtype; 4]>())
    {
        Some(p) if *p == params => {}
        _ => return -1,
    }

    let u = N_VGetArrayPointer(uvec).expect("N_VGetArrayPointer");
    let v = N_VGetArrayPointer(vvec).expect("N_VGetArrayPointer");
    let mut Jv = N_VGetArrayPointer(Jvvec).expect("N_VGetArrayPointer");

    Jv[0] = u[0] * v[0];
    Jv[1] = -u[0] * u[1] * v[0];
    Jv[2] = -u[1] * v[1];
    Jv[3] = u[0] * u[1] * v[1];

    0
}

fn adj_rhs(
    t: sunrealtype,
    y: &N_Vector,
    sens: &N_Vector,
    sens_dot: &N_Vector,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    let Lambda_part = N_VGetSubvector_ManyVector(sens, 0);
    let Lambda = N_VGetSubvector_ManyVector(sens_dot, 0);
    let nu = N_VGetSubvector_ManyVector(sens_dot, 1);
    vjp(&Lambda_part, &Lambda, t, y, None, user_data, None);
    parameter_vjp(&Lambda_part, &nu, t, y, None, user_data, None);
    0
}

fn dgdu(uvec: &N_Vector, dgvec: &N_Vector, _p: &[sunrealtype]) {
    let u = N_VGetArrayPointer(uvec).expect("N_VGetArrayPointer");
    let mut dg = N_VGetArrayPointer(dgvec).expect("N_VGetArrayPointer");

    dg[0] = -1.0 + u[0];
    dg[1] = -1.0 + u[1];
}

fn dgdp(_uvec: &N_Vector, dgvec: &N_Vector, _p: &[sunrealtype]) {
    let mut dg = N_VGetArrayPointer(dgvec).expect("N_VGetArrayPointer");

    dg[0] = 0.0;
    dg[1] = 0.0;
    dg[2] = 0.0;
    dg[3] = 0.0;
}

fn print_help(_argc: i32, argv: &[String], exit_code: i32) {
    if exit_code != 0 {
        eprint!("{}: option not recognized\n", argv[0]);
    } else {
        eprint!("{} ", argv[0]);
    }
    eprint!("options:\n");
    eprint!("--tf <real>         the final simulation time\n");
    eprint!("--dt <real>         the timestep size\n");
    eprint!("--order <int>       the order of the RK method\n");
    eprint!("--check-freq <int>  how often to checkpoint (in steps)\n");
    eprint!("--dont-keep         don't keep checkpoints around after loading\n");
    eprint!("--help              print these options\n");
    std::process::exit(exit_code);
}

fn parse_args(argc: i32, argv: &[String], args: &mut ProgramArgs) {
    let mut argi: i32 = 1;
    while argi < argc {
        let arg = argv[argi as usize].as_str();
        if arg == "--tf" {
            argi += 1;
            args.tf = atof(&argv[argi as usize]);
        } else if arg == "--dt" {
            argi += 1;
            args.dt = atof(&argv[argi as usize]);
        } else if arg == "--order" {
            argi += 1;
            args.order = atoi(&argv[argi as usize]);
        } else if arg == "--check-freq" {
            argi += 1;
            args.check_freq = atoi(&argv[argi as usize]);
        } else if arg == "--dont-keep" {
            args.keep_checks = SUNFALSE;
        } else if arg == "--help" {
            print_help(argc, argv, 0);
        } else {
            print_help(argc, argv, 1);
        }
        argi += 1;
    }
}

/* Check function return value.
 *
 * C's single `check_retval(void* retval_ptr, const char* funcname, int opt)`
 * covers `opt == 0` (NULL-pointer test) and `opt == 1` (`*retval < 0`). Every
 * call site in this example passes `opt == 1`, so only that branch is
 * translated; a dead `opt == 0` helper would be a warning. */
fn check_retval(retval: i32, funcname: &str, opt: i32) -> i32 {
    /* Check if retval < 0 */
    if opt == 1 && retval < 0 {
        eprint!(
            "\nSUNDIALS_ERROR: {}() failed with retval = {}\n\n",
            funcname, retval
        );
        return 1;
    }

    0
}
