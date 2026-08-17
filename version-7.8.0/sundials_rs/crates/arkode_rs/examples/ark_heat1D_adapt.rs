/*---------------------------------------------------------------
 * Programmer(s): Daniel R. Reynolds @ UMBC
 *---------------------------------------------------------------
 * Rust port of examples/arkode/C_serial/ark_heat1D_adapt.c
 *---------------------------------------------------------------
 * Example problem:
 *
 * The following test simulates a simple 1D heat equation,
 *    u_t = k*u_xx + f
 * for t in [0, 10], x in [0, 1], with initial conditions
 *    u(0,x) =  0
 * Dirichlet boundary conditions, i.e.
 *    u_t(t,0) = u_t(t,1) = 0,
 * and a heating term of the form
 *    f = 2*exp(-200*(x-0.25)*(x-0.25))
 *        - exp(-400*(x-0.7)*(x-0.7))
 *        + exp(-500*(x-0.4)*(x-0.4))
 *        - 2*exp(-600*(x-0.55)*(x-0.55));
 *
 * The spatial derivatives are computed using a three-point
 * centered stencil (second order for a uniform mesh).  The data
 * is initially uniformly distributed over N points in the interval
 * [0, 1], but as the simulation proceeds the mesh is adapted.
 *
 * This program solves the problem with a DIRK method, solved with
 * a Newton iteration and SUNLinSol_PCG linear solver, with a
 * user-supplied Jacobian-vector product routine.
 *---------------------------------------------------------------*/
#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

use arkode_rs::prelude::*;

use std::any::Any;
use std::fs::File;
use std::io::Write;

/* constants */
const ZERO: sunrealtype = 0.0;
const PT25: sunrealtype = 0.25;
const PT4: sunrealtype = 0.4;
const PT5: sunrealtype = 0.5;
const PT55: sunrealtype = 0.55;
const PT7: sunrealtype = 0.7;
const ONE: sunrealtype = 1.0;
const TWO: sunrealtype = 2.0;
const TWOHUNDRED: sunrealtype = 200.0;
const FOURHUNDRED: sunrealtype = 400.0;
const FIVEHUNDRED: sunrealtype = 500.0;
const SIXHUNDRED: sunrealtype = 600.0;

/* user data structure */
struct UserData {
    N: sunindextype,         /* current number of intervals */
    x: Vec<sunrealtype>,     /* current mesh */
    k: sunrealtype,          /* diffusion coefficient */
    refine_tol: sunrealtype, /* adaptivity tolerance */
}

/* The C program keeps its own `udata` pointer alive alongside the one handed
to ARKODE via ARKodeSetUserData.  A `Box<dyn Any>` cannot alias, so main()
borrows the box back out of the integrator -- ARKodeGetUserData SWAPS the
token with the caller's slot -- and hands it back (a second swap) before the
next solver call.  The guard is never live across an ARKODE entry point. */
fn swap_udata(ark: &ARKodeMem, slot: &mut Option<Box<dyn Any>>) {
    ARKodeGetUserData(ark, slot);
}

/* Main Program */
fn main() {
    /* general problem parameters */
    let T0: sunrealtype = 0.0; /* initial time */
    let Tf: sunrealtype = 1.0; /* final time */
    let rtol: sunrealtype = 1.0e-3; /* relative tolerance */
    let atol: sunrealtype = 1.0e-10; /* absolute tolerance */
    let hscale: sunrealtype = 1.0; /* time step change factor on resizes */
    let N: sunindextype = 21; /* initial spatial mesh size */
    let refine: sunrealtype = 3.0e-3; /* adaptivity refinement tolerance */
    let k: sunrealtype = 0.5; /* heat conductivity */
    let mut nni: i64 = 0;
    let mut nni_tot: i64 = 0;
    let mut nli: i64 = 0;
    let mut nli_tot: i64 = 0;
    let mut iout: i32 = 0;

    /* general problem variables */
    let mut flag: i32; /* reusable error-checking flag */
    let mut t: sunrealtype;
    let mut olddt: sunrealtype;
    let mut newdt: sunrealtype;
    let mut Nnew: sunindextype = 0;

    /* Create the SUNDIALS context object for this simulation */
    let mut sunctx: Option<SUNContext> = None;
    flag = SUNContext_Create(SUN_COMM_NULL, &mut sunctx);
    if check_flag_int(flag, "SUNContext_Create") != 0 {
        std::process::exit(1);
    }
    let ctx = sunctx.as_ref().expect("SUNContext").clone();

    /* allocate and fill initial udata structure */
    let mut udata = UserData {
        N,
        k,
        refine_tol: refine,
        x: vec![ZERO; N as usize],
    };
    for i in 0..N {
        udata.x[i as usize] = ONE * (i as sunrealtype) / ((N - 1) as sunrealtype);
    }

    /* Initial problem output */
    print!("\n1D adaptive Heat PDE test problem:\n");
    print!("  diffusion coefficient:  k = {}\n", fmt_g(udata.k, 6));
    print!("  initial N = {}\n", udata.N);

    /* Initialize data structures */
    let y = N_VNew_Serial(N, &ctx); /* Create initial serial vector for solution */
    if check_flag_null(&y, "N_VNew_Serial") != 0 {
        std::process::exit(1);
    }
    let mut y = y.expect("N_VNew_Serial");
    N_VConst(ZERO, &y); /* Set initial conditions */

    /* output mesh to disk */
    let mut XFID = File::create("heat_mesh.txt").expect("heat_mesh.txt");

    /* output initial mesh to disk */
    for i in 0..udata.N {
        let _ = write!(XFID, " {}", fmt_e(udata.x[i as usize], 16));
    }
    let _ = write!(XFID, "\n");

    /* Open output stream for results, access data array */
    let mut UFID = File::create("heat1D.txt").expect("heat1D.txt");

    /* output initial condition to disk */
    {
        let data = N_VGetArrayPointer(&y).expect("N_VGetArrayPointer");
        for i in 0..udata.N {
            let _ = write!(UFID, " {}", fmt_e(data[i as usize], 16));
        }
    }
    let _ = write!(UFID, "\n");

    /* Initialize the ARK timestepper */
    let mut arkode_mem = ARKStepCreate(None, Some(f), T0, &y, &ctx);
    if check_flag_null(&arkode_mem, "ARKStepCreate") != 0 {
        std::process::exit(1);
    }
    let ark = arkode_mem.as_ref().expect("ARKStepCreate").clone();

    /* Set routines */
    flag = ARKodeSetUserData(&ark, Some(Box::new(udata))); /* Pass udata to user functions */
    if check_flag_int(flag, "ARKodeSetUserData") != 0 {
        std::process::exit(1);
    }
    flag = ARKodeSetMaxNumSteps(&ark, 10000); /* Increase max num steps  */
    if check_flag_int(flag, "ARKodeSetMaxNumSteps") != 0 {
        std::process::exit(1);
    }
    flag = ARKodeSStolerances(&ark, rtol, atol); /* Specify tolerances */
    if check_flag_int(flag, "ARKodeSStolerances") != 0 {
        std::process::exit(1);
    }
    flag = ARKStepSetAdaptivityMethod(&ark, 2, 1, 0, None); /* Set adaptivity method */
    if check_flag_int(flag, "ARKodeSetAdaptivityMethod") != 0 {
        std::process::exit(1);
    }
    flag = ARKodeSetPredictorMethod(&ark, 0); /* Set predictor method */
    if check_flag_int(flag, "ARKodeSetPredictorMethod") != 0 {
        std::process::exit(1);
    }

    /* Specify linearly implicit RHS, with time-dependent Jacobian */
    flag = ARKodeSetLinear(&ark, 1);
    if check_flag_int(flag, "ARKodeSetLinear") != 0 {
        std::process::exit(1);
    }

    /* Initialize PCG solver -- no preconditioning, with up to N iterations  */
    let LS = SUNLinSol_PCG(&y, 0, N as i32, &ctx);
    if check_flag_null(&LS, "SUNLinSol_PCG") != 0 {
        std::process::exit(1);
    }
    let mut LS = LS.expect("SUNLinSol_PCG");

    /* Linear solver interface -- set user-supplied J*v routine (no 'jtsetup' required) */
    flag = ARKodeSetLinearSolver(&ark, &LS, None); /* Attach linear solver to ARKODE */
    if check_flag_int(flag, "ARKodeSetLinearSolver") != 0 {
        std::process::exit(1);
    }
    flag = ARKodeSetJacTimes(&ark, None, Some(Jac)); /* Set the Jacobian routine */
    if check_flag_int(flag, "ARKodeSetJacTimes") != 0 {
        std::process::exit(1);
    }

    /* Main time-stepping loop: calls ARKodeEvolve to perform the integration, then
    prints results.  Stops when the final time has been reached */
    t = T0;
    olddt = ZERO;
    newdt = ZERO;
    print!("  iout          dt_old                 dt_new               ||u||_rms       N   NNI  NLI\n");
    print!(" ----------------------------------------------------------------------------------------\n");
    {
        let mut slot: Option<Box<dyn Any>> = None;
        swap_udata(&ark, &mut slot);
        {
            let udata = slot
                .as_mut()
                .and_then(|b| b.downcast_mut::<UserData>())
                .expect("user_data is UserData");
            print!(
                " {:4}  {}  {}  {}  {}   {:2}  {:3}\n",
                iout,
                fmt_ew(olddt, 19, 15),
                fmt_ew(newdt, 19, 15),
                fmt_ew(
                    (N_VDotProd(&y, &y) / (udata.N as sunrealtype)).sqrt(),
                    19,
                    15
                ),
                udata.N,
                0,
                0
            );
        }
        swap_udata(&ark, &mut slot);
    }

    while t < Tf {
        /* "set" routines */
        flag = ARKodeSetStopTime(&ark, Tf);
        if check_flag_int(flag, "ARKodeSetStopTime") != 0 {
            std::process::exit(1);
        }

        /* call integrator */
        flag = ARKodeEvolve(&ark, Tf, &y, &mut t, ARK_ONE_STEP);
        if check_flag_int(flag, "ARKodeEvolve") != 0 {
            std::process::exit(1);
        }

        /* "get" routines */
        flag = ARKodeGetLastStep(&ark, &mut olddt);
        if check_flag_int(flag, "ARKodeGetLastStep") != 0 {
            std::process::exit(1);
        }
        flag = ARKodeGetCurrentStep(&ark, &mut newdt);
        if check_flag_int(flag, "ARKodeGetCurrentStep") != 0 {
            std::process::exit(1);
        }
        flag = ARKodeGetNumNonlinSolvIters(&ark, &mut nni);
        if check_flag_int(flag, "ARKodeGetNumNonlinSolvIters") != 0 {
            std::process::exit(1);
        }
        flag = ARKodeGetNumLinIters(&ark, &mut nli);
        if check_flag_int(flag, "ARKodeGetNumLinIters") != 0 {
            std::process::exit(1);
        }

        /* print current solution stats */
        iout += 1;

        /* borrow the udata box back from the integrator for the mesh work */
        let mut slot: Option<Box<dyn Any>> = None;
        swap_udata(&ark, &mut slot);

        {
            let udata = slot
                .as_mut()
                .and_then(|b| b.downcast_mut::<UserData>())
                .expect("user_data is UserData");

            print!(
                " {:4}  {}  {}  {}  {}   {:2}  {:3}\n",
                iout,
                fmt_ew(olddt, 19, 15),
                fmt_ew(newdt, 19, 15),
                fmt_ew(
                    (N_VDotProd(&y, &y) / (udata.N as sunrealtype)).sqrt(),
                    19,
                    15
                ),
                udata.N,
                nni,
                nli
            );
            nni_tot += nni;
            nli_tot += nli;

            /* output results and current mesh to disk */
            {
                let data = N_VGetArrayPointer(&y).expect("N_VGetArrayPointer");
                for i in 0..udata.N {
                    let _ = write!(UFID, " {}", fmt_e(data[i as usize], 16));
                }
            }
            let _ = write!(UFID, "\n");
            for i in 0..udata.N {
                let _ = write!(XFID, " {}", fmt_e(udata.x[i as usize], 16));
            }
            let _ = write!(XFID, "\n");

            /* adapt the spatial mesh */
            let xnew_opt = adapt_mesh(&y, &mut Nnew, udata);
            if check_flag_null(&xnew_opt, "ark_adapt") != 0 {
                std::process::exit(1);
            }
            let xnew = xnew_opt.expect("ark_adapt");

            /* create N_Vector of new length */
            let y2 = N_VNew_Serial(Nnew, &ctx);
            if check_flag_null(&y2, "N_VNew_Serial") != 0 {
                std::process::exit(1);
            }
            let y2 = y2.expect("N_VNew_Serial");

            /* project solution onto new mesh */
            flag = project(udata.N, &udata.x, &y, Nnew, &xnew, &y2);
            if check_flag_int(flag, "project") != 0 {
                std::process::exit(1);
            }

            /* delete old vector, old mesh; swap y and y2 so that y holds the
            new solution (the old mesh vector is dropped with udata.x below) */
            let yold = std::mem::replace(&mut y, y2);
            N_VDestroy(yold);

            /* swap x and xnew so that new mesh is stored in udata structure */
            udata.x = xnew;
            udata.N = Nnew; /* store size of new mesh */
        }

        /* hand the udata box back to the integrator */
        swap_udata(&ark, &mut slot);

        /* call ARKodeResize to notify integrator of change in mesh */
        flag = ARKodeResize(&ark, &y, hscale, t, None, &mut None);
        if check_flag_int(flag, "ARKodeResize") != 0 {
            std::process::exit(1);
        }

        /* destroy and re-allocate linear solver memory; reattach to ARKODE interface */
        let _ = SUNLinSolFree(Some(LS));
        let LS_new = SUNLinSol_PCG(&y, 0, N as i32, &ctx);
        if check_flag_null(&LS_new, "SUNLinSol_PCG") != 0 {
            std::process::exit(1);
        }
        LS = LS_new.expect("SUNLinSol_PCG");
        flag = ARKodeSetLinearSolver(&ark, &LS, None);
        if check_flag_int(flag, "ARKodeSetLinearSolver") != 0 {
            std::process::exit(1);
        }
        flag = ARKodeSetJacTimes(&ark, None, Some(Jac));
        if check_flag_int(flag, "ARKodeSetJacTimes") != 0 {
            std::process::exit(1);
        }
    }
    print!(" ----------------------------------------------------------------------------------------\n");

    /* print some final statistics */
    print!(" Final solver statistics:\n");
    print!("   Total number of time steps = {}\n", iout);
    print!("   Total nonlinear iterations = {}\n", nni_tot);
    print!("   Total linear iterations    = {}\n\n", nli_tot);

    /* Clean up and return with successful completion */
    drop(UFID);
    drop(XFID);
    N_VDestroy(y); /* Free vectors */
    /* the udata struct (and its mesh) is owned by the integrator's `user_data`
    box (C `free(udata->x); free(udata);`) */
    ARKodeFree(&mut arkode_mem); /* Free integrator memory */
    let _ = SUNLinSolFree(Some(LS)); /* Free linear solver */
    let _ = SUNContext_Free(&mut sunctx); /* Free context */

    std::process::exit(0);
}

/*--------------------------------
 * Functions called by the solver
 *--------------------------------*/

/* f routine to compute the ODE RHS function f(t,y). */
fn f(_t: sunrealtype, y: &N_Vector, ydot: &N_Vector, user_data: &mut Option<Box<dyn Any>>) -> i32 {
    let udata = user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<UserData>())
        .expect("user_data is UserData"); /* access problem data */
    let N = udata.N; /* set variable shortcuts */
    let k = udata.k;
    let x = &udata.x;

    /* access data arrays */
    if check_flag_null(&N_VGetArrayPointer(y), "N_VGetArrayPointer") != 0 {
        return 1;
    }
    if check_flag_null(&N_VGetArrayPointer(ydot), "N_VGetArrayPointer") != 0 {
        return 1;
    }

    /* Initialize ydot to zero - also handles boundary conditions */
    N_VConst(ZERO, ydot);

    let Y = N_VGetArrayPointer(y).expect("N_VGetArrayPointer");
    let mut Ydot = N_VGetArrayPointer(ydot).expect("N_VGetArrayPointer");

    /* iterate over domain interior, computing all equations */
    for i in 1..(N - 1) {
        /* interior */
        let i = i as usize;
        let dxL = x[i] - x[i - 1];
        let dxR = x[i + 1] - x[i];
        Ydot[i] = Y[i - 1] * k * TWO / (dxL * (dxL + dxR)) - Y[i] * k * TWO / (dxL * dxR)
            + Y[i + 1] * k * TWO / (dxR * (dxL + dxR))
            + TWO * (-TWOHUNDRED * (x[i] - PT25) * (x[i] - PT25)).sun_exp() /* source term */
            - (-FOURHUNDRED * (x[i] - PT7) * (x[i] - PT7)).sun_exp()
            + (-FIVEHUNDRED * (x[i] - PT4) * (x[i] - PT4)).sun_exp()
            - TWO * (-SIXHUNDRED * (x[i] - PT55) * (x[i] - PT55)).sun_exp();
    }

    0 /* Return with success */
}

/* Jacobian routine to compute J(t,y) = df/dy. */
fn Jac(
    v: &N_Vector,
    Jv: &N_Vector,
    _t: sunrealtype,
    _y: &N_Vector,
    _fy: &N_Vector,
    user_data: &mut Option<Box<dyn Any>>,
    _tmp: &N_Vector,
) -> i32 {
    let udata = user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<UserData>())
        .expect("user_data is UserData"); /* variable shortcuts */
    let N = udata.N;
    let k = udata.k;
    let x = &udata.x;

    /* access data arrays */
    if check_flag_null(&N_VGetArrayPointer(v), "N_VGetArrayPointer") != 0 {
        return 1;
    }
    if check_flag_null(&N_VGetArrayPointer(Jv), "N_VGetArrayPointer") != 0 {
        return 1;
    }

    /* initialize Jv product to zero - also handles boundary conditions */
    N_VConst(ZERO, Jv);

    let V = N_VGetArrayPointer(v).expect("N_VGetArrayPointer");
    let mut JV = N_VGetArrayPointer(Jv).expect("N_VGetArrayPointer");

    /* iterate over domain, computing all Jacobian-vector products */
    for i in 1..(N - 1) {
        let i = i as usize;
        let dxL = x[i] - x[i - 1];
        let dxR = x[i + 1] - x[i];
        JV[i] = V[i - 1] * k * TWO / (dxL * (dxL + dxR)) - V[i] * k * TWO / (dxL * dxR)
            + V[i + 1] * k * TWO / (dxR * (dxL + dxR));
    }

    0 /* Return with success */
}

/*-------------------------------
 * Private helper functions
 *-------------------------------*/

/* Adapts the current mesh, using a simple adaptivity strategy of
refining when an approximation of the scaled second-derivative is
too large.  We only do this in one sweep, so no attempt is made to
ensure the resulting mesh meets these same criteria after adaptivity:
   y [input] -- the current solution vector
   Nnew [output] -- the size of the new mesh
   udata [input] -- the current system information
The return for this function is the new mesh. */
fn adapt_mesh(y: &N_Vector, Nnew: &mut sunindextype, udata: &UserData) -> Option<Vec<sunrealtype>> {
    let num_refine: sunindextype;
    let N_new: sunindextype;

    /* Access current solution and mesh arrays */
    let xold = &udata.x;
    if check_flag_null(&N_VGetArrayPointer(y), "N_VGetArrayPointer") != 0 {
        return None;
    }
    let Y = N_VGetArrayPointer(y).expect("N_VGetArrayPointer");

    /* create marking array */
    let mut marks: Vec<i32> = vec![0; (udata.N - 1) as usize];

    /* perform marking:
    0 -> leave alone
    1 -> refine */
    for i in 1..(udata.N - 1) {
        let i = i as usize;

        /* approximate scaled second-derivative */
        let ydd = Y[i - 1] - TWO * Y[i] + Y[i + 1];

        /* check for refinement */
        if SUNRabs(ydd) > udata.refine_tol {
            marks[i - 1] = 1;
            marks[i] = 1;
        }
    }
    drop(Y);

    /* allocate new mesh */
    let mut nrefine: sunindextype = 0;
    for i in 0..(udata.N - 1) {
        if marks[i as usize] == 1 {
            nrefine += 1;
        }
    }
    num_refine = nrefine;
    N_new = udata.N + num_refine;
    *Nnew = N_new; /* Store new array length */
    let mut xnew: Vec<sunrealtype> = vec![ZERO; N_new as usize];

    /* fill new mesh */
    xnew[0] = xold[0]; /* store endpoints */
    xnew[(N_new - 1) as usize] = xold[(udata.N - 1) as usize];
    let mut j: sunindextype = 1;
    /* iterate over old intervals */
    for i in 0..(udata.N - 1) {
        let i = i as usize;

        /* if mark is 0, reuse old interval */
        if marks[i] == 0 {
            xnew[j as usize] = xold[i + 1];
            j += 1;
            continue;
        }

        /* if mark is 1, refine old interval */
        if marks[i] == 1 {
            xnew[j as usize] = PT5 * (xold[i] + xold[i + 1]);
            j += 1;
            xnew[j as usize] = xold[i + 1];
            j += 1;
            continue;
        }
    }

    /* verify that new mesh is legal */
    for i in 0..(N_new - 1) {
        let i = i as usize;
        if xnew[i + 1] <= xnew[i] {
            eprint!("adapt_mesh error: illegal mesh created\n");
            return None;
        }
    }

    /* the marking array is dropped here (C `free(marks)`) */
    Some(xnew) /* Return with success */
}

/* Projects one vector onto another:
Nold [input] -- the size of the old mesh
xold [input] -- the old mesh
yold [input] -- the vector defined over the old mesh
Nnew [input] -- the size of the new mesh
xnew [input] -- the new mesh
ynew [output] -- the vector defined over the new mesh
                 (allocated prior to calling project) */
fn project(
    Nold: sunindextype,
    xold: &[sunrealtype],
    yold: &N_Vector,
    Nnew: sunindextype,
    xnew: &[sunrealtype],
    ynew: &N_Vector,
) -> i32 {
    /* Access data arrays */
    if check_flag_null(&N_VGetArrayPointer(yold), "N_VGetArrayPointer") != 0 {
        return 1;
    }
    if check_flag_null(&N_VGetArrayPointer(ynew), "N_VGetArrayPointer") != 0 {
        return 1;
    }
    let Yold = N_VGetArrayPointer(yold).expect("N_VGetArrayPointer");
    let mut Ynew = N_VGetArrayPointer(ynew).expect("N_VGetArrayPointer");

    /* loop over new mesh, finding corresponding interval within old mesh,
    and perform piecewise linear interpolation from yold to ynew */
    let mut iv: sunindextype = 0;
    for i in 0..Nnew {
        let i = i as usize;

        /* find old interval, start with previous value since sorted */
        for j in iv..(Nold - 1) {
            if xnew[i] >= xold[j as usize] && xnew[i] <= xold[(j + 1) as usize] {
                iv = j;
                break;
            }
            iv = Nold - 1; /* just in case it wasn't found above */
        }

        /* perform interpolation */
        let iv0 = iv as usize;
        Ynew[i] = Yold[iv0] * (xnew[i] - xold[iv0 + 1]) / (xold[iv0] - xold[iv0 + 1])
            + Yold[iv0 + 1] * (xnew[i] - xold[iv0]) / (xold[iv0 + 1] - xold[iv0]);
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

fn check_flag_int(flagvalue: i32, funcname: &str) -> i32 {
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
