//! Port of `examples/arkode/C_serial/ark_brusselator_1D_mri.c`.
//!
//! This program simulates 1D advection-reaction problem. The
//! brusselator problem from chemical kinetics is used for the
//! reaction terms. This is a PDE system with 3 components,
//! Y = [u,v,w], satisfying the equations,
//!
//!    u_t = -c*u_x + a - (w+1)*u + v*u^2
//!    v_t = -c*v_x + w*u - v*u^2
//!    w_t = -c*w_x + (b-w)/ep - w*u
//!
//! for t in [0, 10], x in [0, 1], with initial conditions
//!
//!    u(0,x) =  a  + 0.1*exp(-(x-0.5)^2 / 0.1)
//!    v(0,x) = b/a + 0.1*exp(-(x-0.5)^2 / 0.1)
//!    w(0,x) =  b  + 0.1*exp(-(x-0.5)^2 / 0.1),
//!
//! and with periodic boundary conditions.
//!
//! This program use the MRIStep module with an explicit slow
//! method and an implicit fast method.

#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

use std::any::Any;
use std::fs::File;
use std::io::Write;

use arkode_rs::prelude::*;

/* Define some constants */
const ZERO: sunrealtype = 0.0;
const ONE: sunrealtype = 1.0;
const TWO: sunrealtype = 2.0;

/* accessor macro between (x,v) location and 1D NVector array */
fn IDX(x: sunindextype, v: sunindextype) -> usize {
    (3 * x + v) as usize
}

/* user data structure */
#[derive(Clone)]
struct UserData {
    N: sunindextype,  /* number of intervals      */
    dx: sunrealtype,  /* mesh spacing             */
    a: sunrealtype,   /* constant forcing on u    */
    b: sunrealtype,   /* steady-state value of w  */
    c: sunrealtype,   /* advection coefficient    */
    ep: sunrealtype,  /* stiffness parameter      */
}

/* Main Program */
fn main() {
    /* general problem parameters */
    let T0: sunrealtype = 0.0; /* initial time                    */
    let Tf: sunrealtype = 10.0; /* final time                      */
    let Nt: i32 = 100; /* total number of output times    */
    let Nvar: sunindextype = 3; /* number of solution fields       */
    let N: sunindextype = 200; /* spatial mesh size (N intervals) */
    let a: sunrealtype = 1.0; /* problem parameters              */
    let b: sunrealtype = 3.5;
    let c: sunrealtype = 0.25;
    let ep: sunrealtype = 1.0e-6; /* stiffness parameter */
    let reltol: sunrealtype = 1.0e-6; /* tolerances          */
    let abstol: sunrealtype = 1.0e-10;

    /* general problem variables */
    let hs: sunrealtype; /* slow step size                 */
    let mut retval: i32; /* reusable return flag           */
    let mut inner_stepper: Option<MRIStepInnerStepper> = None; /* inner stepper */
    let mut t: sunrealtype; /* current/output time data       */
    let dTout: sunrealtype;
    let mut tout: sunrealtype;
    /* temp data values `u`, `v`, `w` are bound inside the output loop */
    let mut nsts: i64 = 0; /* step stats                     */
    let mut nstf: i64 = 0;
    let mut nstf_a: i64 = 0;
    let mut netf: i64 = 0;
    let mut nfse: i64 = 0; /* RHS stats                      */
    let mut nffi: i64 = 0;
    let mut nsetups: i64 = 0; /* linear solver stats            */
    let mut nje: i64 = 0;
    let mut nfeLS: i64 = 0;
    let mut nni: i64 = 0; /* nonlinear solver stats         */
    let mut ncfn: i64 = 0;
    let NEQ: sunindextype; /* number of equations            */
    let mut i: sunindextype; /* counter                        */

    /* Create the SUNDIALS context object for this simulation */
    let mut ctx: Option<SUNContext> = None;
    retval = SUNContext_Create(SUN_COMM_NULL, &mut ctx);
    if check_retval_int(retval, "SUNContext_Create") != 0 {
        std::process::exit(1);
    }
    let ctx_h = ctx.clone().unwrap();

    /* allocate udata structure (C: malloc; the Rust allocation cannot fail) */
    let udata = UserData {
        /* store the inputs in the UserData structure */
        N,
        a,
        b,
        c,
        ep,
        dx: 1.0 / N as sunrealtype, /* periodic BC, divide by N not N-1 */
    };

    /* set total allocated vector length */
    NEQ = Nvar * udata.N;

    /* set the slow step size */
    hs = 0.5 * (udata.dx / SUNRabs(c));

    /* Initial problem output */
    print!("\n1D Advection-Reaction example problem:\n");
    print!("    N = {},  NEQ = {}\n", udata.N, NEQ);
    print!(
        "    problem parameters:  a = {},  b = {},  ep = {}\n",
        fmt_g(udata.a, 6),
        fmt_g(udata.b, 6),
        fmt_g(udata.ep, 6)
    );
    print!("    advection coefficient:  c = {}\n", fmt_g(udata.c, 6));
    print!(
        "    reltol = {},  abstol = {}\n\n",
        fmt_e(reltol, 1),
        fmt_e(abstol, 1)
    );

    /* Create solution vector */
    let y = N_VNew_Serial(NEQ, &ctx_h); /* Create vector for solution */
    if check_retval_null(&y, "N_VNew_Serial") != 0 {
        std::process::exit(1);
    }
    let y = y.unwrap();

    /* Set initial condition */
    retval = SetIC(&y, &udata);
    if check_retval_int(retval, "SetIC") != 0 {
        std::process::exit(1);
    }

    /* Create vector masks */
    let umask = N_VClone(&y);
    if check_retval_null(&umask, "N_VClone") != 0 {
        std::process::exit(1);
    }
    let umask = umask.unwrap();

    let vmask = N_VClone(&y);
    if check_retval_null(&vmask, "N_VClone") != 0 {
        std::process::exit(1);
    }
    let vmask = vmask.unwrap();

    let wmask = N_VClone(&y);
    if check_retval_null(&wmask, "N_VClone") != 0 {
        std::process::exit(1);
    }
    let wmask = wmask.unwrap();

    /* Set mask array values for each solution component */
    N_VConst(0.0, &umask);
    {
        let data = N_VGetArrayPointer(&umask);
        if data.is_none() {
            let _ = check_retval_null(&data, "N_VGetArrayPointer");
            std::process::exit(1);
        }
        let mut data = data.unwrap();
        i = 0;
        while i < N {
            data[IDX(i, 0)] = 1.0;
            i += 1;
        }
    }

    N_VConst(0.0, &vmask);
    {
        let data = N_VGetArrayPointer(&vmask);
        if data.is_none() {
            let _ = check_retval_null(&data, "N_VGetArrayPointer");
            std::process::exit(1);
        }
        let mut data = data.unwrap();
        i = 0;
        while i < N {
            data[IDX(i, 1)] = 1.0;
            i += 1;
        }
    }

    N_VConst(0.0, &wmask);
    {
        let data = N_VGetArrayPointer(&wmask);
        if data.is_none() {
            let _ = check_retval_null(&data, "N_VGetArrayPointer");
            std::process::exit(1);
        }
        let mut data = data.unwrap();
        i = 0;
        while i < N {
            data[IDX(i, 2)] = 1.0;
            i += 1;
        }
    }

    /*
     * Create the fast integrator and set options
     */

    /* Initialize matrix and linear solver data structures */
    let A = SUNBandMatrix(NEQ, 4, 4, &ctx_h);
    if check_retval_null(&A, "SUNBandMatrix") != 0 {
        std::process::exit(1);
    }
    let A = A.unwrap();

    let LS = SUNLinSol_Band(&y, &A, &ctx_h);
    if check_retval_null(&LS, "SUNLinSol_Band") != 0 {
        std::process::exit(1);
    }
    let LS = LS.unwrap();

    /* Initialize the fast integrator. Specify the implicit fast right-hand side
    function in y'=fe(t,y)+fi(t,y)+ff(t,y), the initial time T0, and the
    initial dependent variable vector y. */
    let mut inner_arkode_mem = ARKStepCreate(None, Some(ff), T0, &y, &ctx_h);
    if check_retval_null(&inner_arkode_mem, "ARKStepCreate") != 0 {
        std::process::exit(1);
    }
    let inner = inner_arkode_mem.clone().unwrap();

    /* Attach user data to fast integrator */
    retval = ARKodeSetUserData(&inner, Some(Box::new(udata.clone())));
    if check_retval_int(retval, "ARKodeSetUserData") != 0 {
        std::process::exit(1);
    }

    /* Set the fast method */
    retval = ARKStepSetTableNum(&inner, ARKODE_ARK324L2SA_DIRK_4_2_3, -1);
    if check_retval_int(retval, "ARKStepSetTableNum") != 0 {
        std::process::exit(1);
    }

    /* Specify fast tolerances */
    retval = ARKodeSStolerances(&inner, reltol, abstol);
    if check_retval_int(retval, "ARKodeSStolerances") != 0 {
        std::process::exit(1);
    }

    /* Attach matrix and linear solver */
    retval = ARKodeSetLinearSolver(&inner, &LS, Some(&A));
    if check_retval_int(retval, "ARKodeSetLinearSolver") != 0 {
        std::process::exit(1);
    }

    /* Set the Jacobian routine */
    retval = ARKodeSetJacFn(&inner, Some(Jf));
    if check_retval_int(retval, "ARKodeSetJacFn") != 0 {
        std::process::exit(1);
    }

    /* Create inner stepper */
    retval = ARKodeCreateMRIStepInnerStepper(&inner, &mut inner_stepper);
    if check_retval_int(retval, "ARKodeCreateMRIStepInnerStepper") != 0 {
        std::process::exit(1);
    }
    let stepper = inner_stepper.clone().unwrap();

    /*
     * Create the slow integrator and set options
     */

    /* Initialize the slow integrator. Specify the explicit slow right-hand side
    function in y'=fe(t,y)+fi(t,y)+ff(t,y), the initial time T0, the
    initial dependent variable vector y, and the fast integrator. */
    let mut arkode_mem = MRIStepCreate(Some(fs), None, T0, &y, &stepper, &ctx_h);
    if check_retval_null(&arkode_mem, "MRIStepCreate") != 0 {
        std::process::exit(1);
    }
    let ark = arkode_mem.clone().unwrap();

    /* Pass udata to user functions */
    retval = ARKodeSetUserData(&ark, Some(Box::new(udata.clone())));
    if check_retval_int(retval, "ARKodeSetUserData") != 0 {
        std::process::exit(1);
    }

    /* Set the slow step size */
    retval = ARKodeSetFixedStep(&ark, hs);
    if check_retval_int(retval, "ARKodeSetFixedStep") != 0 {
        std::process::exit(1);
    }

    /* output spatial mesh to disk (add extra point for periodic BC) */
    let mut FID = File::create("mesh.txt").expect("mesh.txt");
    i = 0;
    while i < N + 1 {
        let _ = write!(FID, "  {}\n", fmt_e(udata.dx * i as sunrealtype, 16));
        i += 1;
    }
    drop(FID);

    /* Open output stream for results, access data arrays */
    let mut UFID = File::create("u.txt").expect("u.txt");
    let mut VFID = File::create("v.txt").expect("v.txt");
    let mut WFID = File::create("w.txt").expect("w.txt");

    /* output initial condition to disk (extra output for periodic BC) */
    {
        let data = N_VGetArrayPointer(&y);
        if data.is_none() {
            let _ = check_retval_null(&data, "N_VGetArrayPointer");
            std::process::exit(1);
        }
        let data = data.unwrap();

        i = 0;
        while i < N {
            let _ = write!(UFID, " {}", fmt_e(data[IDX(i, 0)], 16));
            i += 1;
        }
        let _ = write!(UFID, " {}", fmt_e(data[IDX(0, 0)], 16));
        let _ = write!(UFID, "\n");

        i = 0;
        while i < N {
            let _ = write!(VFID, " {}", fmt_e(data[IDX(i, 1)], 16));
            i += 1;
        }
        let _ = write!(VFID, " {}", fmt_e(data[IDX(0, 1)], 16));
        let _ = write!(VFID, "\n");

        i = 0;
        while i < N {
            let _ = write!(WFID, " {}", fmt_e(data[IDX(i, 2)], 16));
            i += 1;
        }
        let _ = write!(WFID, " {}", fmt_e(data[IDX(0, 2)], 16));
        let _ = write!(WFID, "\n");
    }

    /* Main time-stepping loop: calls ARKodeEvolve to perform the integration,
    then prints results.  Stops when the final time has been reached */
    t = T0;
    dTout = (Tf - T0) / Nt as sunrealtype;
    tout = T0 + dTout;
    print!("        t      ||u||_rms   ||v||_rms   ||w||_rms\n");
    print!("   ----------------------------------------------\n");
    for _iout in 0..Nt {
        /* call integrator */
        retval = ARKodeEvolve(&ark, tout, &y, &mut t, ARK_NORMAL);
        if check_retval_int(retval, "ARKodeEvolve") != 0 {
            break;
        }

        /* access/print solution statistics */
        let u = N_VWL2Norm(&y, &umask);
        let u = SUNRsqrt(u * u / N as sunrealtype);
        let v = N_VWL2Norm(&y, &vmask);
        let v = SUNRsqrt(v * v / N as sunrealtype);
        let w = N_VWL2Norm(&y, &wmask);
        let w = SUNRsqrt(w * w / N as sunrealtype);
        print!(
            "  {}  {}  {}  {}\n",
            fmt_fw(t, 10, 6),
            fmt_fw(u, 10, 6),
            fmt_fw(v, 10, 6),
            fmt_fw(w, 10, 6)
        );

        /* output results to disk (extr output for periodic BC) */
        {
            let data = N_VGetArrayPointer(&y).expect("N_VGetArrayPointer");

            i = 0;
            while i < N {
                let _ = write!(UFID, " {}", fmt_e(data[IDX(i, 0)], 16));
                i += 1;
            }
            let _ = write!(UFID, " {}", fmt_e(data[IDX(0, 0)], 16));
            let _ = write!(UFID, "\n");

            i = 0;
            while i < N {
                let _ = write!(VFID, " {}", fmt_e(data[IDX(i, 1)], 16));
                i += 1;
            }
            let _ = write!(VFID, " {}", fmt_e(data[IDX(0, 1)], 16));
            let _ = write!(VFID, "\n");

            i = 0;
            while i < N {
                let _ = write!(WFID, " {}", fmt_e(data[IDX(i, 2)], 16));
                i += 1;
            }
            let _ = write!(WFID, " {}", fmt_e(data[IDX(0, 2)], 16));
            let _ = write!(WFID, "\n");
        }

        /* successful solve: update output time */
        tout += dTout;
        tout = if tout > Tf { Tf } else { tout };
    }
    print!("   ----------------------------------------------\n");
    drop(UFID);
    drop(VFID);
    drop(WFID);

    /* Get some slow integrator statistics */
    retval = ARKodeGetNumSteps(&ark, &mut nsts);
    check_retval_int(retval, "ARKodeGetNumSteps");
    retval = ARKodeGetNumRhsEvals(&ark, 0, &mut nfse);
    check_retval_int(retval, "ARKodeGetNumRhsEvals");

    /* Get some fast integrator statistics */
    retval = ARKodeGetNumSteps(&inner, &mut nstf);
    check_retval_int(retval, "ARKodeGetNumSteps");
    retval = ARKodeGetNumStepAttempts(&inner, &mut nstf_a);
    check_retval_int(retval, "ARKodeGetNumStepAttempts");
    retval = ARKodeGetNumRhsEvals(&inner, 1, &mut nffi);
    check_retval_int(retval, "ARKodeGetNumRhsEvals");
    retval = ARKodeGetNumLinSolvSetups(&inner, &mut nsetups);
    check_retval_int(retval, "ARKodeGetNumLinSolvSetups");
    retval = ARKodeGetNumErrTestFails(&inner, &mut netf);
    check_retval_int(retval, "ARKodeGetNumErrTestFails");
    retval = ARKodeGetNumNonlinSolvIters(&inner, &mut nni);
    check_retval_int(retval, "ARKodeGetNumNonlinSolvIters");
    retval = ARKodeGetNumNonlinSolvConvFails(&inner, &mut ncfn);
    check_retval_int(retval, "ARKodeGetNumNonlinSolvConvFails");
    retval = ARKodeGetNumJacEvals(&inner, &mut nje);
    check_retval_int(retval, "ARKodeGetNumJacEvals");
    retval = ARKodeGetNumLinRhsEvals(&inner, &mut nfeLS);
    check_retval_int(retval, "ARKodeGetNumLinRhsEvals");

    /* Print some final statistics */
    print!("\nFinal Solver Statistics:\n");
    print!("   Slow Steps: nsts = {}\n", nsts);
    print!(
        "   Fast Steps: nstf = {} (attempted = {})\n",
        nstf, nstf_a
    );
    print!("   Total RHS evals:  Fs = {},  Ff = {}\n", nfse, nffi);
    print!("   Total number of fast error test failures = {}\n", netf);
    print!("   Total linear solver setups = {}\n", nsetups);
    print!(
        "   Total RHS evals for setting up the linear system = {}\n",
        nfeLS
    );
    print!("   Total number of Jacobian evaluations = {}\n", nje);
    print!("   Total number of Newton iterations = {}\n", nni);
    print!(
        "   Total number of nonlinear solver convergence failures = {}\n",
        ncfn
    );

    /* Clean up and return with successful completion */
    drop(udata); /* Free user data         */
    ARKodeFree(&mut inner_arkode_mem); /* Free integrator memory */
    let _ = MRIStepInnerStepper_Free(&mut inner_stepper); /* Free inner stepper */
    ARKodeFree(&mut arkode_mem); /* Free integrator memory */
    let _ = SUNLinSolFree(Some(LS)); /* Free linear solver     */
    SUNMatDestroy(A); /* Free matrix            */
    N_VDestroy(y); /* Free vectors           */
    N_VDestroy(umask);
    N_VDestroy(vmask);
    N_VDestroy(wmask);
    let _ = SUNContext_Free(&mut ctx); /* Free context */
}

/* -----------------------------------
 * Functions called by the integrator
 * -----------------------------------*/

/* ff routine to compute the fast portion of the ODE RHS. */
fn ff(
    _t: sunrealtype,
    y: &N_Vector,
    ydot: &N_Vector,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    let udata = user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<UserData>())
        .expect("user_data"); /* access problem data    */
    let N = udata.N; /* set variable shortcuts */
    let a = udata.a;
    let b = udata.b;
    let ep = udata.ep;
    let mut i: sunindextype;

    /* access data arrays */
    let Ydata = N_VGetArrayPointer(y);
    if check_retval_null(&Ydata, "N_VGetArrayPointer") != 0 {
        return 1;
    }
    let Ydata = Ydata.unwrap();

    let dYdata = N_VGetArrayPointer(ydot);
    if check_retval_null(&dYdata, "N_VGetArrayPointer") != 0 {
        return 1;
    }
    let mut dYdata = dYdata.unwrap();

    /* iterate over domain, computing reactions */
    i = 0;
    while i < N {
        /* set shortcuts */
        let u = Ydata[IDX(i, 0)];
        let v = Ydata[IDX(i, 1)];
        let w = Ydata[IDX(i, 2)];

        /* u_t = a - (w+1)*u + v*u^2 */
        dYdata[IDX(i, 0)] = a - (w + ONE) * u + v * u * u;

        /* v_t = w*u - v*u^2 */
        dYdata[IDX(i, 1)] = w * u - v * u * u;

        /* w_t = (b-w)/ep - w*u */
        dYdata[IDX(i, 2)] = (b - w) / ep - w * u;

        i += 1;
    }

    /* return success */
    0
}

/* fs routine to compute the slow portion of the ODE RHS. */
fn fs(
    _t: sunrealtype,
    y: &N_Vector,
    ydot: &N_Vector,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    let udata = user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<UserData>())
        .expect("user_data"); /* access problem data    */
    let N = udata.N; /* set variable shortcuts */
    let c = udata.c;
    let dx = udata.dx;
    let tmp: sunrealtype;
    let mut i: sunindextype;

    /* access data arrays */
    let Ydata = N_VGetArrayPointer(y);
    if check_retval_null(&Ydata, "N_VGetArrayPointer") != 0 {
        return 1;
    }
    let Ydata = Ydata.unwrap();

    let dYdata = N_VGetArrayPointer(ydot);
    if check_retval_null(&dYdata, "N_VGetArrayPointer") != 0 {
        return 1;
    }
    let mut dYdata = dYdata.unwrap();

    /* iterate over domain, computing advection */
    tmp = -c / dx;

    if c > ZERO {
        /*
         * right moving flow
         */

        /* left boundary Jacobian entries */
        dYdata[IDX(0, 0)] = tmp * (Ydata[IDX(0, 0)] - Ydata[IDX(N - 1, 0)]);
        dYdata[IDX(0, 1)] = tmp * (Ydata[IDX(0, 1)] - Ydata[IDX(N - 1, 1)]);
        dYdata[IDX(0, 2)] = tmp * (Ydata[IDX(0, 2)] - Ydata[IDX(N - 1, 2)]);

        /* interior Jacobian entries */
        i = 1;
        while i < N {
            dYdata[IDX(i, 0)] = tmp * (Ydata[IDX(i, 0)] - Ydata[IDX(i - 1, 0)]);
            dYdata[IDX(i, 1)] = tmp * (Ydata[IDX(i, 1)] - Ydata[IDX(i - 1, 1)]);
            dYdata[IDX(i, 2)] = tmp * (Ydata[IDX(i, 2)] - Ydata[IDX(i - 1, 2)]);
            i += 1;
        }
    } else if c < ZERO {
        /*
         * left moving flow
         */

        /* interior Jacobian entries */
        i = 0;
        while i < N - 1 {
            dYdata[IDX(i, 0)] = tmp * (Ydata[IDX(i + 1, 0)] - Ydata[IDX(i, 0)]);
            dYdata[IDX(i, 1)] = tmp * (Ydata[IDX(i + 1, 1)] - Ydata[IDX(i, 1)]);
            dYdata[IDX(i, 2)] = tmp * (Ydata[IDX(i + 1, 2)] - Ydata[IDX(i, 2)]);
            i += 1;
        }

        /* right boundary Jacobian entries */
        dYdata[IDX(N - 1, 0)] = tmp * (Ydata[IDX(N - 1, 0)] - Ydata[IDX(0, 0)]);
        dYdata[IDX(N - 1, 1)] = tmp * (Ydata[IDX(N - 1, 1)] - Ydata[IDX(0, 1)]);
        dYdata[IDX(N - 1, 2)] = tmp * (Ydata[IDX(N - 1, 2)] - Ydata[IDX(0, 2)]);
    }

    /* return success */
    0
}

/* Js routine to compute the Jacobian of the fast portion of the ODE RHS. */
fn Jf(
    _t: sunrealtype,
    y: &N_Vector,
    _fy: &N_Vector,
    Jac: &SUNMatrix,
    user_data: &mut Option<Box<dyn Any>>,
    _tmp1: &N_Vector,
    _tmp2: &N_Vector,
    _tmp3: &N_Vector,
) -> i32 {
    let udata = user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<UserData>())
        .expect("user_data"); /* access problem data */
    let N = udata.N; /* set shortcuts */
    let ep = udata.ep;
    let mut i: sunindextype;

    /* access solution array */
    let Ydata = N_VGetArrayPointer(y);
    if check_retval_null(&Ydata, "N_VGetArrayPointer") != 0 {
        return 1;
    }
    let Ydata = Ydata.unwrap();

    /* iterate over nodes, filling in Jacobian entries */
    i = 0;
    while i < N {
        /* set nodal value shortcuts (shifted index due to start at first interior node) */
        let u = Ydata[IDX(i, 0)];
        let v = Ydata[IDX(i, 1)];
        let w = Ydata[IDX(i, 2)];

        /* all vars wrt u */
        SM_ELEMENT_B_set(Jac, 3 * i, 3 * i, TWO * u * v - (w + ONE));
        SM_ELEMENT_B_set(Jac, 3 * i + 1, 3 * i, w - TWO * u * v);
        SM_ELEMENT_B_set(Jac, 3 * i + 2, 3 * i, -w);

        /* all vars wrt v */
        SM_ELEMENT_B_set(Jac, 3 * i, 3 * i + 1, u * u);
        SM_ELEMENT_B_set(Jac, 3 * i + 1, 3 * i + 1, -u * u);

        /* all vars wrt w */
        SM_ELEMENT_B_set(Jac, 3 * i, 3 * i + 2, -u);
        SM_ELEMENT_B_set(Jac, 3 * i + 1, 3 * i + 2, u);
        SM_ELEMENT_B_set(Jac, 3 * i + 2, 3 * i + 2, -ONE / ep - u);

        i += 1;
    }

    /* return success */
    0
}

/* ------------------------------
 * Private helper functions
 * ------------------------------*/

/* Set the initial condition */
fn SetIC(y: &N_Vector, user_data: &UserData) -> i32 {
    let udata = user_data; /* access problem data    */
    let N = udata.N; /* set variable shortcuts */
    let a = udata.a;
    let b = udata.b;
    let dx = udata.dx;

    let mut x: sunrealtype;
    let mut p: sunrealtype;
    let mut i: sunindextype;

    /* Access data array from NVector y */
    let mut data = N_VGetArrayPointer(y).expect("N_VGetArrayPointer");

    /* Set initial conditions into y */
    i = 0;
    while i < N {
        x = i as sunrealtype * dx;
        p = 0.1 * SUNRexp(-(SUNSQR(x - 0.5)) / 0.1);
        data[IDX(i, 0)] = a + p;
        data[IDX(i, 1)] = b / a + p;
        data[IDX(i, 2)] = b + p;
        i += 1;
    }

    /* return success */
    0
}

/* --------------------------------------------------------------
 * Function to check return values:
 *
 * opt == 0  means SUNDIALS function allocates memory so check if
 *           returned NULL pointer
 * opt == 1  means SUNDIALS function returns a flag so check if
 *           flag < 0
 * opt == 2  means function allocates memory so check if returned
 *           NULL pointer
 *
 * The C void-pointer/opt polymorphism splits into two typed helpers with
 * identical message text:
 *   check_retval_null = opt == 0
 *   check_retval_int  = opt == 1
 * --------------------------------------------------------------*/

fn check_retval_null<T>(returnvalue: &Option<T>, funcname: &str) -> i32 {
    /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
    if returnvalue.is_none() {
        eprint!(
            "\nSUNDIALS_ERROR: {}() failed - returned NULL pointer\n\n",
            funcname
        );
        return 1;
    }
    0
}

fn check_retval_int(retval: i32, funcname: &str) -> i32 {
    /* Check if flag < 0 */
    if retval < 0 {
        eprint!(
            "\nSUNDIALS_ERROR: {}() failed with flag = {}\n\n",
            funcname, retval
        );
        return 1;
    }
    0
}

/*---- end of file ----*/
