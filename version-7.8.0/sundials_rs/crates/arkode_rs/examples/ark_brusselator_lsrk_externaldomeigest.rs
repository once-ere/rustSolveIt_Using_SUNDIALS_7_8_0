/*-----------------------------------------------------------------
 * Rust port of
 * examples/arkode/C_serial/ark_brusselator_lsrk_externaldomeigest.c
 * Programmer(s): Mustafa Aggul @ SMU
 * Based on
 * ark_analytic_lsrk_domeigest.c by Mustafa Aggul @ SMU and
 * ark_brusselator.c by Daniel R. Reynolds @ UMBC
 *---------------------------------------------------------------
 * The following example simulates the same problem as
 * ark_brusselator_lsrk_domeigest.c but attaches a user-supplied dominate
 * eigenvalue function (dom_eig) instead of a SUNDomEigEstimator object.
 *
 * The user-supplied function wraps a SUNDomEigEstimator to demonstrate how an
 * estimator can be used in a standalone fashion to estimate the dominant
 * eigenvalues of a desired function. In particular, we note that there is no
 * requirement for the SUNDomEigEstimator to be used purely for
 * super-time-stepping methods in LSRKStep and they may be applied in other
 * settings.
 *-----------------------------------------------------------------*/

#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

use arkode_rs::prelude::*;

use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

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
 * word) >> 1.  (Same helper as `ark_brusselator_lsrk_domeigest.rs`.)
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

/* user data structure */
struct UserData_ {
    ctx: SUNContext,
    rdata: [sunrealtype; 3],
    DEE: Option<SUNDomEigEstimator>,
    rel_tol: sunrealtype,
    max_iters: sunindextype,
}

/* C hands the SAME `UserData*` to `ARKodeSetUserData` and to
`SUNDomEigEstimator_SetRhs`, and `dom_eig` WRITES `data->DEE` through it,
so both holders must observe one object.  A `Box<dyn Any>` cannot be
aliased, so — exactly as the pinned `SensParams` contract does for the
CVODES/IDAS parameter array — the shared record lives behind an
`Rc<RefCell<..>>` and each holder owns a CLONE of that handle.  Neither
callback holds the `RefCell` borrow across a call that can re-enter it
(`SUNDomEigEstimator_Estimate` invokes `f` through the estimator's own
`rhs_data` clone). */
type UserData = Rc<RefCell<UserData_>>;

/* Main Program */
fn main() {
    /* general problem parameters */
    let T0: sunrealtype = 0.0; /* initial time */
    let Tf: sunrealtype = 10.0; /* final time */
    let dTout: sunrealtype = 1.0; /* time between outputs */
    let NEQ: sunindextype = 3; /* number of dependent vars. */
    let Nt: i32 = (Tf / dTout).ceil() as i32; /* number of output times */
    let test: i32 = 2; /* test problem to run */
    let a: sunrealtype;
    let b: sunrealtype;
    let ep: sunrealtype;
    let u0: sunrealtype;
    let v0: sunrealtype;
    let w0: sunrealtype;

    let reltol: sunrealtype = 1.0e-6; /* tolerances */
    let abstol: sunrealtype = 1.0e-10;

    /* general problem variables */
    let mut flag: i32; /* reusable error-checking flag */

    /* Command-line arguments (C `argc` / `argv`) */
    let argv: Vec<String> = std::env::args().collect();
    let argc: i32 = argv.len() as i32;

    /* Create the SUNDIALS context object for this simulation */
    let mut sunctx: Option<SUNContext> = None;
    flag = SUNContext_Create(SUN_COMM_NULL, &mut sunctx);
    if check_flag(Some(flag), "SUNContext_Create", 1) != 0 {
        std::process::exit(1);
    }
    let ctx = sunctx.as_ref().expect("sunctx").clone();

    /* set up the test problem according to the desired test */
    if test == 1 {
        u0 = 3.9;
        v0 = 1.1;
        w0 = 2.8;
        a = 1.2;
        b = 2.5;
        ep = 1.0e-5;
    } else if test == 3 {
        u0 = 3.0;
        v0 = 3.0;
        w0 = 3.5;
        a = 0.5;
        b = 3.0;
        ep = 5.0e-4;
    } else {
        u0 = 1.2;
        v0 = 3.1;
        w0 = 3.0;
        a = 1.0;
        b = 3.5;
        ep = 5.0e-6;
    }

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
    let ProbData: UserData = Rc::new(RefCell::new(UserData_ {
        ctx: ctx.clone(),
        rdata: [a, b, ep], /* set user data  */
        DEE: None,
        rel_tol: 5.0e-3,
        max_iters: 100,
    }));
    let y = N_VNew_Serial(NEQ, &ctx); /* Create serial vector for solution */
    if check_flag(y.as_ref().map(|_| 0), "N_VNew_Serial", 0) != 0 {
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
    let mut arkode_mem_opt = LSRKStepCreateSTS(f, T0, &y, &ctx);
    if check_flag(arkode_mem_opt.as_ref().map(|_| 0), "LSRKStepCreateSTS", 0) != 0 {
        std::process::exit(1);
    }
    let arkode_mem = arkode_mem_opt.as_ref().expect("arkode_mem").clone();

    /* Set routines */
    flag = ARKodeSetUserData(&arkode_mem, Some(Box::new(ProbData.clone()))); /* Pass rdata to user functions */
    if check_flag(Some(flag), "ARKodeSetUserData", 1) != 0 {
        std::process::exit(1);
    }

    flag = ARKodeSStolerances(&arkode_mem, reltol, abstol); /* Specify tolerances */
    if check_flag(Some(flag), "ARKodeSStolerances", 1) != 0 {
        std::process::exit(1);
    }

    /* Specify stiff interpolant */
    flag = ARKodeSetInterpolantType(&arkode_mem, ARK_INTERP_LAGRANGE);
    if check_flag(Some(flag), "ARKodeSetInterpolantType", 1) != 0 {
        std::process::exit(1);
    }

    /* Specify user provided dominant eigenvalue function */
    flag = LSRKStepSetDomEigFn(&arkode_mem, Some(dom_eig));
    if check_flag(Some(flag), "LSRKStepSetDomEigFn", 1) != 0 {
        std::process::exit(1);
    }

    /* Specify max number of stages allowed */
    flag = LSRKStepSetMaxNumStages(&arkode_mem, 200);
    if check_flag(Some(flag), "LSRKStepSetMaxNumStages", 1) != 0 {
        std::process::exit(1);
    }

    /* Specify max number of steps allowed */
    flag = ARKodeSetMaxNumSteps(&arkode_mem, 2000);
    if check_flag(Some(flag), "ARKodeSetMaxNumSteps", 1) != 0 {
        std::process::exit(1);
    }

    /* Specify safety factor for user provided dom_eig */
    flag = LSRKStepSetDomEigSafetyFactor(&arkode_mem, 1.01);
    if check_flag(Some(flag), "LSRKStepSetDomEigSafetyFactor", 1) != 0 {
        std::process::exit(1);
    }

    /* Specify the Runge--Kutta--Legendre LSRK method by name */
    flag = LSRKStepSetSTSMethodByName(&arkode_mem, "ARKODE_LSRK_RKL_2");
    if check_flag(Some(flag), "LSRKStepSetSTSMethodByName", 1) != 0 {
        std::process::exit(1);
    }

    /* Override any current settings with command-line options */
    flag = ARKodeSetOptions(&arkode_mem, None, None, argc, &argv);
    if check_flag(Some(flag), "ARKodeSetOptions", 1) != 0 {
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
        flag = ARKodeEvolve(&arkode_mem, tout, &y, &mut t, ARK_NORMAL); /* call integrator */
        if check_flag(Some(flag), "ARKodeEvolve", 1) != 0 {
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
    flag = ARKodePrintAllStats(
        &arkode_mem,
        &SUNFile::Stdout,
        SUNOutputFormat::SUN_OUTPUTFORMAT_TABLE,
    );
    if check_flag(Some(flag), "ARKodePrintAllStats", 1) != 0 {
        std::process::exit(1);
    }

    /* Clean up and return with successful completion */
    N_VDestroy(y); /* Free y vector */
    ARKodeFree(&mut arkode_mem_opt); /* Free integrator memory */
    /* C: SUNDomEigEstimator_Destroy(&ProbData.DEE) — taking the handle out
    of the record leaves the field NULL exactly as C's Destroy does. */
    let mut DEE_opt = ProbData.borrow_mut().DEE.take();
    let _ = SUNDomEigEstimator_Destroy(&mut DEE_opt); /* Free DEE object */
    SUNContext_Free(&mut sunctx); /* Free context */

    std::process::exit(flag);
}

/*-------------------------------
 * Functions called by the solver
 *-------------------------------*/

/* f routine to compute the ODE RHS function f(t,y). */
fn f(_t: sunrealtype, y: &N_Vector, ydot: &N_Vector, user_data: &mut Option<Box<dyn Any>>) -> i32 {
    let data = user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<UserData>())
        .expect("user_data") /* cast user_data to UserData */
        .clone();
    let rdata = data.borrow().rdata; /* access rdata from UserData */
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

/* dom_eig routine to estimate the dominated eigenvalue */
fn dom_eig(
    t: sunrealtype,
    y: &N_Vector,
    _fn_: &N_Vector,
    lambdaR: &mut sunrealtype,
    lambdaI: &mut sunrealtype,
    user_data: &mut Option<Box<dyn Any>>,
    temp1: &N_Vector,
    _temp2: &N_Vector,
    _temp3: &N_Vector,
) -> i32 {
    let flag;
    let data = user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<UserData>())
        .expect("user_data") /* cast user_data to UserData */
        .clone();

    let ctx = data.borrow().ctx.clone(); /* access context from UserData */
    let DEE = data.borrow().DEE.clone(); /* access DEE from UserData */

    /* DEE is initialized to NULL, so on the first dom_eig call we need
    to create and initialize this object */
    let DEE = match DEE {
        Some(DEE) => DEE,
        None => {
            /* Create random initial vector for power iteration.  C reads the
            data pointer first and the length second; the two are pure reads
            of the same vector, and the length query is hoisted here so the
            data RefMut guard is not live across it. */
            let NEQ: sunindextype = N_VGetLength(temp1);
            {
                let mut qd = N_VGetArrayPointer(temp1).expect("N_VGetArrayPointer");
                for i in 0..NEQ {
                    qd[i as usize] = rand() as sunrealtype / RAND_MAX as sunrealtype;
                }
            }

            /* Create power iteration dominant eigenvalue estimator (DEE) */
            let (max_iters, rel_tol) = {
                let d = data.borrow();
                (d.max_iters, d.rel_tol)
            };
            let DEE = SUNDomEigEstimator_Power(temp1, max_iters, rel_tol, &ctx);
            if check_flag(DEE.as_ref().map(|_| 0), "SUNDomEigEstimator_Power", 0) != 0 {
                return -1;
            }
            let DEE = DEE.expect("DEE");
            data.borrow_mut().DEE = Some(DEE.clone());

            /* Set the ODE right-hand side function at t for the Jacobian-vector products */
            let flag =
                SUNDomEigEstimator_SetRhs(&DEE, Some(Box::new(data.clone())), Some(f as SUNRhsFn));
            if check_flag(Some(flag), "SUNDomEigEstimator_SetRhs", 1) != 0 {
                return -1;
            }

            let flag = SUNDomEigEstimator_Initialize(&DEE);
            if check_flag(Some(flag), "SUNDomEigEstimator_Initialize", 1) != 0 {
                return 1;
            }

            DEE
        }
    };

    /* Set the linearization vector and time for the Jacobian-vector products */
    flag = SUNDomEigEstimator_SetRhsLinearizationPoint(&DEE, t, y);
    if check_flag(Some(flag), "SUNDomEigEstimator_SetRhsLinearizationPoint", 1) != 0 {
        return -1;
    }

    /* Estimate the dominant eigenvalue with power iteration */
    let flag = SUNDomEigEstimator_Estimate(&DEE, lambdaR, lambdaI);
    if check_flag(Some(flag), "SUNDomEigEstimator_Estimate", 1) != 0 {
        return -1;
    }

    0 /* return with success */
}

/*-------------------------------
 * Private helper functions
 *-------------------------------*/

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
        let errflag = flagvalue.expect("flagvalue");
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
