//! Port of `examples/arkode/C_serial/ark_brusselator1D.c`.
//!
//! Example problem:
//!
//! The following test simulates a brusselator problem from chemical
//! kinetics.  This is n PDE system with 3 components, Y = [u,v,w],
//! satisfying the equations,
//!    u_t = du*u_xx + a - (w+1)*u + v*u^2
//!    v_t = dv*v_xx + w*u - v*u^2
//!    w_t = dw*w_xx + (b-w)/ep - w*u
//! for t in [0, 80], x in [0, 1], with initial conditions
//!    u(0,x) =  a  + 0.1*sin(pi*x)
//!    v(0,x) = b/a + 0.1*sin(pi*x)
//!    w(0,x) =  b  + 0.1*sin(pi*x),
//! and with stationary boundary conditions, i.e.
//!    u_t(t,0) = u_t(t,1) = 0,
//!    v_t(t,0) = v_t(t,1) = 0,
//!    w_t(t,0) = w_t(t,1) = 0.
//! Note: these can also be implemented as Dirichlet boundary
//! conditions with values identical to the initial conditions.
//!
//! The spatial derivatives are computed using second-order
//! centered differences, with the data distributed over N points
//! on a uniform spatial grid.
//!
//! This program solves the problem with the DIRK method, using a
//! Newton iteration with the SUNBAND band linear solver, and a
//! user-supplied Jacobian routine.
//!
//! 100 outputs are printed at equal intervals, and run statistics
//! are printed at the end.

#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

use std::any::Any;
use std::fs::File;
use std::io::Write;

use arkode_rs::prelude::*;

/* accessor macros between (x,v) location and 1D NVector array */
fn IDX(x: sunindextype, v: sunindextype) -> sunindextype {
    3 * x + v
}

/* C `SM_ELEMENT_B(A,i,j) += v` — the C macro is an lvalue, the ported
accessors are a get/set pair. */
fn SM_ELEMENT_B_add(A: &SUNMatrix, i: sunindextype, j: sunindextype, v: sunrealtype) {
    SM_ELEMENT_B_set(A, i, j, SM_ELEMENT_B(A, i, j) + v);
}

/* user data structure */
#[derive(Clone)]
struct UserData {
    N: sunindextype, /* number of intervals     */
    dx: sunrealtype, /* mesh spacing            */
    a: sunrealtype,  /* constant forcing on u   */
    b: sunrealtype,  /* steady-state value of w */
    du: sunrealtype, /* diffusion coeff for u   */
    dv: sunrealtype, /* diffusion coeff for v   */
    dw: sunrealtype, /* diffusion coeff for w   */
    ep: sunrealtype, /* stiffness parameter     */
}

/* Main Program */
fn main() {
    /* general problem parameters */
    let T0: sunrealtype = 0.0; /* initial time */
    let Tf: sunrealtype = 10.0; /* final time */
    let Nt: i32 = 100; /* total number of output times */
    let Nvar: sunindextype = 3; /* number of solution fields */
    let N: sunindextype = 201; /* spatial mesh size */
    let a: sunrealtype = 0.6; /* problem parameters */
    let b: sunrealtype = 2.0;
    let du: sunrealtype = 0.025;
    let dv: sunrealtype = 0.025;
    let dw: sunrealtype = 0.025;
    let ep: sunrealtype = 1.0e-5; /* stiffness parameter */
    let reltol: sunrealtype = 1.0e-6; /* tolerances */
    let abstol: sunrealtype = 1.0e-10;
    let NEQ: sunindextype;
    let mut i: sunindextype;

    /* general problem variables */
    let mut flag: i32; /* reusable error-checking flag */
    let pi: sunrealtype;
    let mut t: sunrealtype;
    let dTout: sunrealtype;
    let mut tout: sunrealtype;
    /* temp data values `u`, `v`, `w` are bound inside the output loop */
    let mut nst: i64 = 0;
    let mut nst_a: i64 = 0;
    let mut nfe: i64 = 0;
    let mut nfi: i64 = 0;
    let mut nsetups: i64 = 0;
    let mut nje: i64 = 0;
    let mut nfeLS: i64 = 0;
    let mut nni: i64 = 0;
    let mut ncfn: i64 = 0;
    let mut netf: i64 = 0;

    /* Create the SUNDIALS context object for this simulation */
    let mut ctx: Option<SUNContext> = None;
    flag = SUNContext_Create(SUN_COMM_NULL, &mut ctx);
    if check_flag_int(flag, "SUNContext_Create") != 0 {
        std::process::exit(1);
    }
    let ctx_h = ctx.clone().unwrap();

    /* allocate udata structure (C: malloc; the Rust allocation cannot fail,
    so the `check_flag(..., "malloc", 2)` guard has no counterpart) */
    let mut udata = UserData {
        /* store the inputs in the UserData structure */
        N,
        /* `dx` is assigned further below, exactly where C assigns it; C
        leaves the malloc'd field indeterminate until then and never
        reads it before. */
        dx: 0.0,
        a,
        b,
        du,
        dv,
        dw,
        ep,
    };

    /* set total allocated vector length */
    NEQ = Nvar * udata.N;

    /* Initial problem output */
    print!("\n1D Brusselator PDE test problem:\n");
    print!("    N = {},  NEQ = {}\n", udata.N, NEQ);
    print!(
        "    problem parameters:  a = {},  b = {},  ep = {}\n",
        fmt_g(udata.a, 6),
        fmt_g(udata.b, 6),
        fmt_g(udata.ep, 6)
    );
    print!(
        "    diffusion coefficients:  du = {},  dv = {},  dw = {}\n",
        fmt_g(udata.du, 6),
        fmt_g(udata.dv, 6),
        fmt_g(udata.dw, 6)
    );
    print!(
        "    reltol = {},  abstol = {}\n\n",
        fmt_e(reltol, 1),
        fmt_e(abstol, 1)
    );

    /* Initialize data structures */
    let y = N_VNew_Serial(NEQ, &ctx_h); /* Create serial vector for solution */
    if check_flag_null(&y, "N_VNew_Serial") != 0 {
        std::process::exit(1);
    }
    let y = y.unwrap();

    let umask = N_VClone(&y);
    if check_flag_null(&umask, "N_VClone") != 0 {
        std::process::exit(1);
    }
    let umask = umask.unwrap();

    let vmask = N_VClone(&y);
    if check_flag_null(&vmask, "N_VClone") != 0 {
        std::process::exit(1);
    }
    let vmask = vmask.unwrap();

    let wmask = N_VClone(&y);
    if check_flag_null(&wmask, "N_VClone") != 0 {
        std::process::exit(1);
    }
    let wmask = wmask.unwrap();

    /* Set initial conditions into y */
    udata.dx = 1.0 / (N - 1) as sunrealtype; /* set spatial mesh spacing */
    {
        /* Access data array for new NVector y */
        let data = N_VGetArrayPointer(&y);
        if check_flag_null(&data, "N_VGetArrayPointer") != 0 {
            std::process::exit(1);
        }
        let mut data = data.unwrap();

        pi = 4.0 * (1.0f64).sun_atan();
        i = 0;
        while i < N {
            data[IDX(i, 0) as usize] = a + 0.1 * (pi * i as sunrealtype * udata.dx).sun_sin(); /* u */
            data[IDX(i, 1) as usize] = b / a + 0.1 * (pi * i as sunrealtype * udata.dx).sun_sin(); /* v */
            data[IDX(i, 2) as usize] = b + 0.1 * (pi * i as sunrealtype * udata.dx).sun_sin(); /* w */
            i += 1;
        }
    }

    /* Set mask array values for each solution component */
    N_VConst(0.0, &umask);
    {
        let data = N_VGetArrayPointer(&umask);
        if check_flag_null(&data, "N_VGetArrayPointer") != 0 {
            std::process::exit(1);
        }
        let mut data = data.unwrap();
        i = 0;
        while i < N {
            data[IDX(i, 0) as usize] = 1.0;
            i += 1;
        }
    }

    N_VConst(0.0, &vmask);
    {
        let data = N_VGetArrayPointer(&vmask);
        if check_flag_null(&data, "N_VGetArrayPointer") != 0 {
            std::process::exit(1);
        }
        let mut data = data.unwrap();
        i = 0;
        while i < N {
            data[IDX(i, 1) as usize] = 1.0;
            i += 1;
        }
    }

    N_VConst(0.0, &wmask);
    {
        let data = N_VGetArrayPointer(&wmask);
        if check_flag_null(&data, "N_VGetArrayPointer") != 0 {
            std::process::exit(1);
        }
        let mut data = data.unwrap();
        i = 0;
        while i < N {
            data[IDX(i, 2) as usize] = 1.0;
            i += 1;
        }
    }

    /* Call ARKStepCreate to initialize the ARK timestepper module and
    specify the right-hand side function in y'=f(t,y), the initial time
    T0, and the initial dependent variable vector y.  Note: since this
    problem is fully implicit, we set f_E to NULL and f_I to f. */
    let mut arkode_mem = ARKStepCreate(None, Some(f), T0, &y, &ctx_h);
    if check_flag_null(&arkode_mem, "ARKStepCreate") != 0 {
        std::process::exit(1);
    }
    let ark = arkode_mem.clone().unwrap();

    /* Set routines */
    flag = ARKodeSetUserData(&ark, Some(Box::new(udata.clone()))); /* Pass udata to user functions */
    if check_flag_int(flag, "ARKodeSetUserData") != 0 {
        std::process::exit(1);
    }
    flag = ARKodeSStolerances(&ark, reltol, abstol); /* Specify tolerances */
    if check_flag_int(flag, "ARKodeSStolerances") != 0 {
        std::process::exit(1);
    }

    /* Initialize band matrix data structure and solver -- A will be factored, so set smu to ml+mu */
    let A = SUNBandMatrix(NEQ, 4, 4, &ctx_h);
    if check_flag_null(&A, "SUNBandMatrix") != 0 {
        std::process::exit(1);
    }
    let A = A.unwrap();

    let LS = SUNLinSol_Band(&y, &A, &ctx_h);
    if check_flag_null(&LS, "SUNLinSol_Band") != 0 {
        std::process::exit(1);
    }
    let LS = LS.unwrap();

    /* Linear solver interface */
    flag = ARKodeSetLinearSolver(&ark, &LS, Some(&A)); /* Attach matrix and linear solver */
    if check_flag_int(flag, "ARKodeSetLinearSolver") != 0 {
        std::process::exit(1);
    }

    flag = ARKodeSetJacFn(&ark, Some(Jac)); /* Set the Jacobian routine */
    if check_flag_int(flag, "ARKodeSetJacFn") != 0 {
        std::process::exit(1);
    }

    flag = ARKodeSetAutonomous(&ark, SUNTRUE);
    if check_flag_int(flag, "ARKodeSetAutonomous") != 0 {
        std::process::exit(1);
    }

    /* output spatial mesh to disk */
    let mut FID = File::create("bruss_mesh.txt").expect("bruss_mesh.txt");
    i = 0;
    while i < N {
        let _ = write!(FID, "  {}\n", fmt_e(udata.dx * i as sunrealtype, 16));
        i += 1;
    }
    drop(FID);

    /* Open output streams for results, access data array */
    let mut UFID = File::create("bruss_u.txt").expect("bruss_u.txt");
    let mut VFID = File::create("bruss_v.txt").expect("bruss_v.txt");
    let mut WFID = File::create("bruss_w.txt").expect("bruss_w.txt");

    /* output initial condition to disk */
    {
        let data = N_VGetArrayPointer(&y);
        if check_flag_null(&data, "N_VGetArrayPointer") != 0 {
            std::process::exit(1);
        }
        let data = data.unwrap();
        i = 0;
        while i < N {
            let _ = write!(UFID, " {}", fmt_e(data[IDX(i, 0) as usize], 16));
            i += 1;
        }
        i = 0;
        while i < N {
            let _ = write!(VFID, " {}", fmt_e(data[IDX(i, 1) as usize], 16));
            i += 1;
        }
        i = 0;
        while i < N {
            let _ = write!(WFID, " {}", fmt_e(data[IDX(i, 2) as usize], 16));
            i += 1;
        }
        let _ = write!(UFID, "\n");
        let _ = write!(VFID, "\n");
        let _ = write!(WFID, "\n");
    }

    /* Main time-stepping loop: calls ARKodeEvolve to perform the integration, then
    prints results.  Stops when the final time has been reached */
    t = T0;
    dTout = (Tf - T0) / Nt as sunrealtype;
    tout = T0 + dTout;
    print!("        t      ||u||_rms   ||v||_rms   ||w||_rms\n");
    print!("   ----------------------------------------------\n");
    for _iout in 0..Nt {
        flag = ARKodeEvolve(&ark, tout, &y, &mut t, ARK_NORMAL); /* call integrator */
        if check_flag_int(flag, "ARKodeEvolve") != 0 {
            break;
        }
        let u = N_VWL2Norm(&y, &umask); /* access/print solution statistics */
        let u = (u * u / N as sunrealtype).sqrt();
        let v = N_VWL2Norm(&y, &vmask);
        let v = (v * v / N as sunrealtype).sqrt();
        let w = N_VWL2Norm(&y, &wmask);
        let w = (w * w / N as sunrealtype).sqrt();
        print!(
            "  {}  {}  {}  {}\n",
            fmt_fw(t, 10, 6),
            fmt_fw(u, 10, 6),
            fmt_fw(v, 10, 6),
            fmt_fw(w, 10, 6)
        );
        if flag >= 0 {
            /* successful solve: update output time */
            tout += dTout;
            tout = if tout > Tf { Tf } else { tout };
        } else {
            /* unsuccessful solve: break */
            eprint!("Solver failure, stopping integration\n");
            break;
        }

        /* output results to disk */
        {
            let data = N_VGetArrayPointer(&y).expect("N_VGetArrayPointer");
            i = 0;
            while i < N {
                let _ = write!(UFID, " {}", fmt_e(data[IDX(i, 0) as usize], 16));
                i += 1;
            }
            i = 0;
            while i < N {
                let _ = write!(VFID, " {}", fmt_e(data[IDX(i, 1) as usize], 16));
                i += 1;
            }
            i = 0;
            while i < N {
                let _ = write!(WFID, " {}", fmt_e(data[IDX(i, 2) as usize], 16));
                i += 1;
            }
        }
        let _ = write!(UFID, "\n");
        let _ = write!(VFID, "\n");
        let _ = write!(WFID, "\n");
    }
    print!("   ----------------------------------------------\n");
    drop(UFID);
    drop(VFID);
    drop(WFID);

    /* Print some final statistics */
    flag = ARKodeGetNumSteps(&ark, &mut nst);
    check_flag_int(flag, "ARKodeGetNumSteps");
    flag = ARKodeGetNumStepAttempts(&ark, &mut nst_a);
    check_flag_int(flag, "ARKodeGetNumStepAttempts");
    flag = ARKodeGetNumRhsEvals(&ark, 0, &mut nfe);
    check_flag_int(flag, "ARKodeGetNumRhsEvals");
    flag = ARKodeGetNumRhsEvals(&ark, 1, &mut nfi);
    check_flag_int(flag, "ARKodeGetNumRhsEvals");
    flag = ARKodeGetNumLinSolvSetups(&ark, &mut nsetups);
    check_flag_int(flag, "ARKodeGetNumLinSolvSetups");
    flag = ARKodeGetNumErrTestFails(&ark, &mut netf);
    check_flag_int(flag, "ARKodeGetNumErrTestFails");
    flag = ARKodeGetNumNonlinSolvIters(&ark, &mut nni);
    check_flag_int(flag, "ARKodeGetNumNonlinSolvIters");
    flag = ARKodeGetNumNonlinSolvConvFails(&ark, &mut ncfn);
    check_flag_int(flag, "ARKodeGetNumNonlinSolvConvFails");
    flag = ARKodeGetNumJacEvals(&ark, &mut nje);
    check_flag_int(flag, "ARKodeGetNumJacEvals");
    flag = ARKodeGetNumLinRhsEvals(&ark, &mut nfeLS);
    check_flag_int(flag, "ARKodeGetNumLinRhsEvals");

    print!("\nFinal Solver Statistics:\n");
    print!(
        "   Internal solver steps = {} (attempted = {})\n",
        nst, nst_a
    );
    print!("   Total RHS evals:  Fe = {},  Fi = {}\n", nfe, nfi);
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
    print!("   Total number of error test failures = {}\n\n", netf);

    /* Clean up and return with successful completion */
    N_VDestroy(y); /* Free vectors */
    N_VDestroy(umask);
    N_VDestroy(vmask);
    N_VDestroy(wmask);
    drop(udata); /* Free user data */
    ARKodeFree(&mut arkode_mem); /* Free integrator memory */
    let _ = SUNLinSolFree(Some(LS)); /* Free linear solver */
    SUNMatDestroy(A); /* Free A matrix */
    let _ = SUNContext_Free(&mut ctx); /* Free context */
}

/*-------------------------------
 * Functions called by the solver
 *-------------------------------*/

/* f routine to compute the ODE RHS function f(t,y). */
fn f(_t: sunrealtype, y: &N_Vector, ydot: &N_Vector, user_data: &mut Option<Box<dyn Any>>) -> i32 {
    let udata = user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<UserData>())
        .expect("user_data is UserData"); /* access problem data */
    let N = udata.N; /* set variable shortcuts */
    let a = udata.a;
    let b = udata.b;
    let ep = udata.ep;
    let du = udata.du;
    let dv = udata.dv;
    let dw = udata.dw;
    let dx = udata.dx;
    let mut i: sunindextype;

    /* access data arrays */
    if check_flag_null(&N_VGetArrayPointer(y), "N_VGetArrayPointer") != 0 {
        return 1;
    }
    if check_flag_null(&N_VGetArrayPointer(ydot), "N_VGetArrayPointer") != 0 {
        return 1;
    }
    N_VConst(0.0, ydot); /* initialize ydot to zero */

    let Ydata = N_VGetArrayPointer(y).expect("N_VGetArrayPointer");
    let mut dYdata = N_VGetArrayPointer(ydot).expect("N_VGetArrayPointer");

    /* iterate over domain, computing all equations */
    let uconst = du / dx / dx;
    let vconst = dv / dx / dx;
    let wconst = dw / dx / dx;
    i = 1;
    while i < N - 1 {
        /* set shortcuts */
        let u = Ydata[IDX(i, 0) as usize];
        let ul = Ydata[IDX(i - 1, 0) as usize];
        let ur = Ydata[IDX(i + 1, 0) as usize];
        let v = Ydata[IDX(i, 1) as usize];
        let vl = Ydata[IDX(i - 1, 1) as usize];
        let vr = Ydata[IDX(i + 1, 1) as usize];
        let w = Ydata[IDX(i, 2) as usize];
        let wl = Ydata[IDX(i - 1, 2) as usize];
        let wr = Ydata[IDX(i + 1, 2) as usize];

        /* Fill in ODE RHS for u */
        dYdata[IDX(i, 0) as usize] = (ul - 2.0 * u + ur) * uconst + a - (w + 1.0) * u + v * u * u;

        /* Fill in ODE RHS for v */
        dYdata[IDX(i, 1) as usize] = (vl - 2.0 * v + vr) * vconst + w * u - v * u * u;

        /* Fill in ODE RHS for w */
        dYdata[IDX(i, 2) as usize] = (wl - 2.0 * w + wr) * wconst + (b - w) / ep - w * u;

        i += 1;
    }

    /* enforce stationary boundaries */
    dYdata[IDX(0, 2) as usize] = 0.0;
    dYdata[IDX(0, 1) as usize] = dYdata[IDX(0, 2) as usize];
    dYdata[IDX(0, 0) as usize] = dYdata[IDX(0, 1) as usize];
    dYdata[IDX(N - 1, 2) as usize] = 0.0;
    dYdata[IDX(N - 1, 1) as usize] = dYdata[IDX(N - 1, 2) as usize];
    dYdata[IDX(N - 1, 0) as usize] = dYdata[IDX(N - 1, 1) as usize];

    0 /* Return with success */
}

/* Jacobian routine to compute J(t,y) = df/dy. */
fn Jac(
    _t: sunrealtype,
    y: &N_Vector,
    _fy: &N_Vector,
    J: &SUNMatrix,
    user_data: &mut Option<Box<dyn Any>>,
    _tmp1: &N_Vector,
    _tmp2: &N_Vector,
    _tmp3: &N_Vector,
) -> i32 {
    let udata = user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<UserData>())
        .expect("user_data is UserData"); /* access problem data */
    let _ = SUNMatZero(J); /* Initialize Jacobian to zero */

    /* Fill in the Laplace matrix */
    LaplaceMatrix(1.0, J, udata);

    /* Add in the Jacobian of the reaction terms matrix */
    ReactionJac(1.0, y, J, udata);

    0 /* Return with success */
}

/*-------------------------------
 * Private helper functions
 *-------------------------------*/

/* Routine to compute the stiffness matrix from (L*y), scaled by the factor c.
We add the result into Jac and do not erase what was already there */
fn LaplaceMatrix(c: sunrealtype, Jac: &SUNMatrix, udata: &UserData) -> i32 {
    let N = udata.N; /* set shortcuts */
    let dx = udata.dx;
    let mut i: sunindextype;

    /* iterate over intervals, filling in Jacobian of (L*y) using SM_ELEMENT_B
    macro (see sunmatrix_band.h) */
    i = 1;
    while i < N - 1 {
        SM_ELEMENT_B_add(Jac, IDX(i, 0), IDX(i - 1, 0), c * udata.du / dx / dx);
        SM_ELEMENT_B_add(Jac, IDX(i, 1), IDX(i - 1, 1), c * udata.dv / dx / dx);
        SM_ELEMENT_B_add(Jac, IDX(i, 2), IDX(i - 1, 2), c * udata.dw / dx / dx);
        SM_ELEMENT_B_add(Jac, IDX(i, 0), IDX(i, 0), -c * 2.0 * udata.du / dx / dx);
        SM_ELEMENT_B_add(Jac, IDX(i, 1), IDX(i, 1), -c * 2.0 * udata.dv / dx / dx);
        SM_ELEMENT_B_add(Jac, IDX(i, 2), IDX(i, 2), -c * 2.0 * udata.dw / dx / dx);
        SM_ELEMENT_B_add(Jac, IDX(i, 0), IDX(i + 1, 0), c * udata.du / dx / dx);
        SM_ELEMENT_B_add(Jac, IDX(i, 1), IDX(i + 1, 1), c * udata.dv / dx / dx);
        SM_ELEMENT_B_add(Jac, IDX(i, 2), IDX(i + 1, 2), c * udata.dw / dx / dx);
        i += 1;
    }

    0 /* Return with success */
}

/* Routine to compute the Jacobian matrix from R(y), scaled by the factor c.
We add the result into Jac and do not erase what was already there */
fn ReactionJac(c: sunrealtype, y: &N_Vector, Jac: &SUNMatrix, udata: &UserData) -> i32 {
    let N = udata.N; /* set shortcuts */
    let ep = udata.ep;
    let mut i: sunindextype;
    let Ydata = N_VGetArrayPointer(y); /* access solution array */
    if check_flag_null(&Ydata, "N_VGetArrayPointer") != 0 {
        return 1;
    }
    let Ydata = Ydata.unwrap();

    /* iterate over nodes, filling in Jacobian of reaction terms */
    i = 1;
    while i < N - 1 {
        let u = Ydata[IDX(i, 0) as usize]; /* set nodal value shortcuts */
        let v = Ydata[IDX(i, 1) as usize];
        let w = Ydata[IDX(i, 2) as usize];

        /* all vars wrt u */
        SM_ELEMENT_B_add(Jac, IDX(i, 0), IDX(i, 0), c * (2.0 * u * v - (w + 1.0)));
        SM_ELEMENT_B_add(Jac, IDX(i, 1), IDX(i, 0), c * (w - 2.0 * u * v));
        SM_ELEMENT_B_add(Jac, IDX(i, 2), IDX(i, 0), c * (-w));

        /* all vars wrt v */
        SM_ELEMENT_B_add(Jac, IDX(i, 0), IDX(i, 1), c * (u * u));
        SM_ELEMENT_B_add(Jac, IDX(i, 1), IDX(i, 1), c * (-u * u));

        /* all vars wrt w */
        SM_ELEMENT_B_add(Jac, IDX(i, 0), IDX(i, 2), c * (-u));
        SM_ELEMENT_B_add(Jac, IDX(i, 1), IDX(i, 2), c * (u));
        SM_ELEMENT_B_add(Jac, IDX(i, 2), IDX(i, 2), c * (-1.0 / ep - u));

        i += 1;
    }

    0 /* Return with success */
}

/* Check function return value...
    opt == 0 means SUNDIALS function allocates memory so check if
             returned NULL pointer
    opt == 1 means SUNDIALS function returns a flag so check if
             flag >= 0
    opt == 2 means function allocates memory so check if returned
             NULL pointer

   The C void-pointer/opt polymorphism splits into two typed helpers with
   identical message text:
     check_flag_null = opt == 0
     check_flag_int  = opt == 1
   (opt == 2 guards a `malloc` that cannot fail in the Rust port.)
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

fn check_flag_int(flag: i32, funcname: &str) -> i32 {
    /* Check if flag < 0 */
    if flag < 0 {
        eprint!(
            "\nSUNDIALS_ERROR: {}() failed with flag = {}\n\n",
            funcname, flag
        );
        return 1;
    }
    0
}

/*---- end of file ----*/
