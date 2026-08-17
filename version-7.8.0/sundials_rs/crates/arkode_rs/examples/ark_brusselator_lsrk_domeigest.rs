//! Port of `examples/arkode/C_serial/ark_brusselator_lsrk_domeigest.c`.
//!
//! Example problem:
//!
//! The following test simulates a brusselator problem from chemical
//! kinetics.  This is an ODE system with 3 components, Y = [u,v,w],
//! satisfying the equations,
//!    du/dt = a - (w+1)*u + v*u^2
//!    dv/dt = w*u - v*u^2
//!    dw/dt = (b-w)/ep - w*u
//! for t in the interval [0.0, 10.0], with initial conditions
//! Y0 = [u0,v0,w0].
//!
//! We use u0=1.2,  v0=3.1,  w0=3,  a=1,  b=3.5,  ep=5.0e-6.
//!
//! In this case, w experiences a fast initial transient, jumping 0.5
//! within a few steps. All values proceed smoothly until
//! around t=6.5, when both u and v undergo a sharp transition,
//! with u increasing from around 0.5 to 5 and v decreasing
//! from around 6 to 1 in less than 0.5 time units. After this
//! transition, both u and v continue to evolve somewhat
//! rapidly for another 1.4 time units, and finish off smoothly.
//!
//! This program solves the problem with an STS method from LSRKStep using a
//! SUNDIALS dominant eigenvalue estimation (DEE) module.
//!
//! 100 outputs are printed at equal intervals, and run statistics
//! are printed at the end.

#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

use std::any::Any;
use std::cell::RefCell;

use arkode_rs::prelude::*;

/* C `NV_Ith_S(v,i)` (0-based); the RefMut guard never outlives the
statement that reads or writes the entry. */

fn NV_Ith_S(v: &N_Vector, i: usize) -> sunrealtype {
    NV_DATA_S(v)[i]
}

fn NV_Ith_S_set(v: &N_Vector, i: usize, x: sunrealtype) {
    NV_DATA_S(v)[i] = x;
}

/* -----------------------------------------------------------------
 * C library `rand()` / `RAND_MAX`.
 *
 * The example seeds nothing, so C uses the default state (identical to
 * `srand(1)`).  The reference output was produced against the glibc
 * generator, whose sequence IS output-observable here: the drawn values
 * form the initial power-iteration eigenvector, and the converged
 * "Max./Min. spectral radius" statistics depend on it.  Reproduced
 * exactly: TYPE_3 additive-feedback generator, DEG = 31, SEP = 3, the
 * Lehmer/Schrage seeding loop, 310 discarded outputs, result = (state
 * word) >> 1.
 * ----------------------------------------------------------------- */

const RAND_MAX: i32 = 2147483647;

struct CRandState {
    r: [u32; 31],
    fptr: usize,
    rptr: usize,
}

impl CRandState {
    fn new(seed: u32) -> CRandState {
        let mut r = [0u32; 31];
        r[0] = seed;
        for i in 1..31 {
            /* word = 16807 * word % 2147483647, via Schrage's trick */
            let word = r[i - 1] as i32 as i64;
            let hi = word / 127773;
            let lo = word % 127773;
            let mut w = 16807 * lo - 2836 * hi;
            if w < 0 {
                w += 2147483647;
            }
            r[i] = w as u32;
        }
        let mut state = CRandState {
            r,
            fptr: 3,
            rptr: 0,
        };
        for _ in 0..310 {
            state.next();
        }
        state
    }

    fn next(&mut self) -> i32 {
        self.r[self.fptr] = self.r[self.fptr].wrapping_add(self.r[self.rptr]);
        let result = (self.r[self.fptr] >> 1) & 0x7fff_ffff;
        self.fptr = (self.fptr + 1) % 31;
        self.rptr = (self.rptr + 1) % 31;
        result as i32
    }
}

thread_local! {
    static C_RAND_STATE: RefCell<CRandState> = RefCell::new(CRandState::new(1));
}

fn rand() -> i32 {
    C_RAND_STATE.with(|s| s.borrow_mut().next())
}

/* Main Program */
fn main() {
    /* general problem parameters */
    let T0: sunrealtype = 0.0; /* initial time */
    let Tf: sunrealtype = 10.0; /* final time */
    let dTout: sunrealtype = 1.0; /* time between outputs */
    let NEQ: sunindextype = 3; /* number of dependent vars. */
    let Nt: i32 = (Tf / dTout).ceil() as i32; /* number of output times */

    let reltol: sunrealtype = 1.0e-6; /* tolerances */
    let abstol: sunrealtype = 1.0e-10;

    /* general problem variables */

    /* Dominant Eigenvalue Estimator (DEE) pointers and variables */
    let max_iters: sunindextype = 100; /* max number of power iterations (PI)*/
    let numwarmup: sunindextype = 10; /* number of preprocessing warmups */
    let rel_tol: sunrealtype = 5.0e-3; /* relative error for PI*/

    /* Command-line arguments (C `argc` / `argv`) */
    let argv: Vec<String> = std::env::args().collect();
    let argc: i32 = argv.len() as i32;

    /* Create the SUNDIALS context object for this simulation */
    let mut sunctx: Option<SUNContext> = None;
    let flag = SUNContext_Create(SUN_COMM_NULL, &mut sunctx);
    if check_flag(flag, "SUNContext_Create") != 0 {
        std::process::exit(1);
    }
    let ctx = sunctx.clone().expect("sunctx");

    /* set up the test problem */
    let u0: sunrealtype = 1.2;
    let v0: sunrealtype = 3.1;
    let w0: sunrealtype = 3.0;
    let a: sunrealtype = 1.0;
    let b: sunrealtype = 3.5;
    let ep: sunrealtype = 5.0e-6;

    /* Initial problem output */
    print!("\nBrusselator ODE test problem:\n");
    print!(
        "    initial conditions:  u0 = {},  v0 = {},  w0 = {}\n",
        fmt_g(u0, 6),
        fmt_g(v0, 6),
        fmt_g(w0, 6)
    );
    print!(
        "    problem parameters:  a = {},  b = {},  ep = {}\n",
        fmt_g(a, 6),
        fmt_g(b, 6),
        fmt_g(ep, 6)
    );
    print!(
        "    reltol = {},  abstol = {}\n\n",
        fmt_e(reltol, 1),
        fmt_e(abstol, 1)
    );

    /* Initialize data structures */
    let rdata: [sunrealtype; 3] = [a, b, ep]; /* set user data  */
    let y = N_VNew_Serial(NEQ, &ctx); /* Create serial vector for solution */
    if check_flag_null(&y, "N_VNew_Serial") != 0 {
        std::process::exit(1);
    }
    let y = y.expect("y");

    {
        let mut ydata = N_VGetArrayPointer(&y).expect("N_VGetArrayPointer");
        ydata[0] = u0; /* Set initial conditions */
        ydata[1] = v0;
        ydata[2] = w0;
    }

    /* Call LSRKStepCreateSTS to initialize the STS timestepper module and
    specify the right-hand side function in y'=f(t,y), the initial time
    T0, and the initial dependent variable vector y. */
    let mut arkode_mem = LSRKStepCreateSTS(f, T0, &y, &ctx);
    if check_flag_null(&arkode_mem, "LSRKStepCreateSTS") != 0 {
        std::process::exit(1);
    }
    let ark = arkode_mem.clone().expect("arkode_mem");

    /* Set routines */
    let flag = ARKodeSetUserData(&ark, Some(Box::new(rdata))); /* Pass rdata to user functions */
    if check_flag(flag, "ARKodeSetUserData") != 0 {
        std::process::exit(1);
    }

    let flag = ARKodeSStolerances(&ark, reltol, abstol); /* Specify tolerances */
    if check_flag(flag, "ARKodeSStolerances") != 0 {
        std::process::exit(1);
    }

    let flag = ARKodeSetInterpolantType(&ark, ARK_INTERP_LAGRANGE); /* Specify stiff interpolant */
    if check_flag(flag, "ARKodeSetInterpolantType") != 0 {
        std::process::exit(1);
    }

    /* Set the initial random eigenvector for the DEE */
    let q = N_VClone(&y);
    if check_flag_null(&q, "N_VClone") != 0 {
        std::process::exit(1);
    }
    let q = q.expect("q");

    {
        let mut qd = N_VGetArrayPointer(&q).expect("N_VGetArrayPointer");
        for i in 0..NEQ {
            qd[i as usize] = rand() as sunrealtype / RAND_MAX as sunrealtype;
        }
    }

    /* Create power iteration dominant eigenvalue estimator (DEE) */
    let mut DEE = SUNDomEigEstimator_Power(&q, max_iters, rel_tol, &ctx);
    if check_flag_null(&DEE, "SUNDomEigEstimator_Power") != 0 {
        std::process::exit(1);
    }
    let dee = DEE.clone().expect("DEE");

    /* After the DEE creation, random q vector is no longer needed.
    It is used only to initialize the DEE */
    N_VDestroy(q);

    /* Attach the DEE to the LSRKStep module.
    There is no need to set Atimes or initialize since LSRKStep provides
    a default Atimes, and initialized the DEE, after it is attached. */
    let flag = LSRKStepSetDomEigEstimator(&ark, Some(&dee));
    if check_flag(flag, "LSRKStepSetDomEigEstimator") != 0 {
        std::process::exit(1);
    }

    /* Set the number of preprocessing warmups. The warmup
    is used to compute a "better" initial eigenvector and so an initial
    eigenvalue. The warmup is performed only once by the LSRKStep module
    internally unless LSRKStepSetNumDomEigEstPreprocessIters is called to set
    a new number of succeeding warmups that would be executed before
    every dominant eigenvalue estimation call */
    let flag = LSRKStepSetNumDomEigEstInitPreprocessIters(&ark, numwarmup as i32);
    if check_flag(flag, "LSRKStepSetNumDomEigEstInitPreprocessIters") != 0 {
        std::process::exit(1);
    }

    /* Specify max number of stages allowed */
    let flag = LSRKStepSetMaxNumStages(&ark, 200);
    if check_flag(flag, "LSRKStepSetMaxNumStages") != 0 {
        std::process::exit(1);
    }

    /* Specify max number of steps allowed */
    let flag = ARKodeSetMaxNumSteps(&ark, 2000);
    if check_flag(flag, "ARKodeSetMaxNumSteps") != 0 {
        std::process::exit(1);
    }

    /* Specify safety factor for user provided dom_eig */
    let flag = LSRKStepSetDomEigSafetyFactor(&ark, 1.01);
    if check_flag(flag, "LSRKStepSetDomEigSafetyFactor") != 0 {
        std::process::exit(1);
    }

    /* Specify the Runge--Kutta--Legendre LSRK method by name */
    let flag = LSRKStepSetSTSMethodByName(&ark, "ARKODE_LSRK_RKL_2");
    if check_flag(flag, "LSRKStepSetSTSMethodByName") != 0 {
        std::process::exit(1);
    }

    /* Override any current settings with command-line options */
    let flag = SUNDomEigEstimator_SetOptions(&dee, None, None, &argv);
    if check_flag(flag, "SUNDomEigEstimator_SetOptions") != 0 {
        std::process::exit(1);
    }

    /* Override any current settings with command-line options */
    let flag = ARKodeSetOptions(&ark, None, None, argc, &argv);
    if check_flag(flag, "ARKodeSetOptions") != 0 {
        std::process::exit(1);
    }

    /* Main time-stepping loop: calls ARKodeEvolve to perform the integration, then
    prints results.  Stops when the final time has been reached */
    let mut t: sunrealtype = T0;
    let mut tout: sunrealtype = T0 + dTout;
    print!("        t           u           v           w\n");
    print!("   -------------------------------------------\n");
    print!(
        "  {}  {}  {}  {}\n",
        fmt_fw(t, 10, 6),
        fmt_fw(NV_Ith_S(&y, 0), 10, 6),
        fmt_fw(NV_Ith_S(&y, 1), 10, 6),
        fmt_fw(NV_Ith_S(&y, 2), 10, 6)
    );

    for _iout in 0..Nt {
        let flag = ARKodeEvolve(&ark, tout, &y, &mut t, ARK_NORMAL); /* call integrator */
        if check_flag(flag, "ARKodeEvolve") != 0 {
            break;
        }
        print!(
            "  {}  {}  {}  {}\n", /* access/print solution */
            fmt_fw(t, 10, 6),
            fmt_fw(NV_Ith_S(&y, 0), 10, 6),
            fmt_fw(NV_Ith_S(&y, 1), 10, 6),
            fmt_fw(NV_Ith_S(&y, 2), 10, 6)
        );
        if flag >= 0 {
            /* successful solve: update time */
            tout += dTout;
            tout = if tout > Tf { Tf } else { tout };
        } else {
            /* unsuccessful solve: break */
            eprint!("Solver failure, stopping integration\n");
            break;
        }
    }
    print!("   -------------------------------------------\n");

    /* Print final statistics */
    print!("\nFinal Statistics:\n");
    let flag = ARKodePrintAllStats(&ark, &SUNFile::Stdout, SUNOutputFormat::SUN_OUTPUTFORMAT_TABLE);
    if check_flag(flag, "ARKodePrintAllStats") != 0 {
        std::process::exit(1);
    }

    /* Clean up and return with successful completion */
    N_VDestroy(y); /* Free y vector */
    drop(dee);
    ARKodeFree(&mut arkode_mem); /* Free integrator memory */
    SUNDomEigEstimator_Destroy(&mut DEE); /* Free DEE object */
    let _ = SUNContext_Free(&mut sunctx); /* Free context */

    std::process::exit(flag);
}

/*-------------------------------
 * Functions called by the solver
 *-------------------------------*/

/* f routine to compute the ODE RHS function f(t,y). */
fn f(
    _t: sunrealtype,
    y: &N_Vector,
    ydot: &N_Vector,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    let rdata = user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<[sunrealtype; 3]>())
        .expect("user_data"); /* cast user_data to sunrealtype */
    let a = rdata[0]; /* access data entries */
    let b = rdata[1];
    let ep = rdata[2];
    let u = NV_Ith_S(y, 0); /* access solution values */
    let v = NV_Ith_S(y, 1);
    let w = NV_Ith_S(y, 2);

    /* fill in the RHS function */
    NV_Ith_S_set(ydot, 0, a - (w + 1.0) * u + v * u * u);
    NV_Ith_S_set(ydot, 1, w * u - v * u * u);
    NV_Ith_S_set(ydot, 2, (b - w) / ep - w * u);

    0 /* Return with success */
}

/*-------------------------------
 * Private helper functions
 *-------------------------------*/

/* Check function return value...
    opt == 0 means SUNDIALS function allocates memory so check if
             returned NULL pointer
    opt == 1 means SUNDIALS function returns a flag so check if
             flag >= 0
    opt == 2 means function allocates memory so check if returned
             NULL pointer

   The C void-pointer/opt polymorphism splits into two typed helpers with
   identical message text; `opt == 2` is unused by this example.
*/

fn check_flag_null<T>(flagvalue: &Option<T>, funcname: &str) -> i32 {
    /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
    if flagvalue.is_none() {
        eprint!(
            "\nSUNDIALS_ERROR: {}() failed - returned NULL pointer\n\n",
            funcname
        );
        return 1;
    }
    0
}

fn check_flag(flagvalue: i32, funcname: &str) -> i32 {
    /* Check if flag < 0 */
    if flagvalue < 0 {
        eprint!(
            "\nSUNDIALS_ERROR: {}() failed with flag = {}\n\n",
            funcname, flagvalue
        );
        return 1;
    }
    0
}

/*---- end of file ----*/
