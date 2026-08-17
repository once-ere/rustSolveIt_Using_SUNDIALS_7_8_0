#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

/* -----------------------------------------------------------------
 * Ported from: examples/cvode/serial/cvDiurnal_kry.c
 * -----------------------------------------------------------------
 * Example problem:
 *
 * An ODE system is generated from the following 2-species diurnal
 * kinetics advection-diffusion PDE system in 2 space dimensions.
 * The PDE system is treated by central differences on a uniform
 * 10 x 10 mesh, with simple polynomial initial profiles.
 * The problem is solved with CVODE, with the BDF/GMRES
 * method (i.e. using the SUNLinSol_SPGMR linear solver) and the
 * block-diagonal part of the Newton matrix as a left
 * preconditioner. A copy of the block-diagonal part of the
 * Jacobian is saved and conditionally reused within the Precond
 * routine.
 * -----------------------------------------------------------------*/

use cvode_rs::prelude::*;
use cvode_rs::sundials_dense::{
    SUNDlsMat_denseAddIdentity, SUNDlsMat_denseCopy, SUNDlsMat_denseGETRF, SUNDlsMat_denseGETRS,
    SUNDlsMat_denseScale,
};
use cvode_rs::sundials_direct::dls_cols;

use std::any::Any;

/* helpful macros */

fn SQR(a: sunrealtype) -> sunrealtype {
    a * a
}

/* Problem Constants */

const ZERO: sunrealtype = 0.0;
const ONE: sunrealtype = 1.0;
const TWO: sunrealtype = 2.0;

const NUM_SPECIES: usize = 2; /* number of species         */
const KH: sunrealtype = 4.0e-6; /* horizontal diffusivity Kh */
const VEL: sunrealtype = 0.001; /* advection velocity V      */
const KV0: sunrealtype = 1.0e-8; /* coefficient in Kv(y)      */
const Q1: sunrealtype = 1.63e-16; /* coefficients q1, q2, c3   */
const Q2: sunrealtype = 4.66e-16;
const C3: sunrealtype = 3.7e16;
const A3: sunrealtype = 22.62; /* coefficient in expression for q3(t) */
const A4: sunrealtype = 7.601; /* coefficient in expression for q4(t) */
const C1_SCALE: sunrealtype = 1.0e6; /* coefficients in initial profiles    */
const C2_SCALE: sunrealtype = 1.0e12;

const T0: sunrealtype = ZERO; /* initial time */
const NOUT: i32 = 12; /* number of output times */
const TWOHR: sunrealtype = 7200.0; /* number of seconds in two hours  */
const HALFDAY: sunrealtype = 4.32e4; /* number of seconds in a half day */
const PI: sunrealtype = 3.1415926535898; /* pi */

const XMIN: sunrealtype = ZERO; /* grid boundaries in x  */
const XMAX: sunrealtype = 20.0;
const YMIN: sunrealtype = 30.0; /* grid boundaries in y  */
const YMAX: sunrealtype = 50.0;
const XMID: sunrealtype = 10.0; /* grid midpoints in x,y */
const YMID: sunrealtype = 40.0;

const MX: usize = 10; /* MX = number of x mesh points */
const MY: usize = 10; /* MY = number of y mesh points */
const NSMX: usize = 20; /* NSMX = NUM_SPECIES*MX */
const MM: usize = MX * MY; /* MM = MX*MY */

/* CVodeInit Constants */

const RTOL: sunrealtype = 1.0e-5; /* scalar relative tolerance */
const FLOOR: sunrealtype = 100.0; /* value of C1 or C2 at which tolerances */
/* change from relative to absolute      */
const ATOL: sunrealtype = RTOL * FLOOR; /* scalar absolute tolerance */
const NEQ: usize = NUM_SPECIES * MM; /* NEQ = number of equations */

/* User-defined vector and matrix accessor helpers: IJKth, IJth.

IJKth(vdata,i,j,k) references the element in the vdata array for
species i at mesh point (j,k), where 1 <= i <= NUM_SPECIES,
0 <= j <= MX-1, 0 <= k <= MY-1.

IJth(a,i,j) = a[j-1][i-1] references the (i,j)th entry of a small
dense matrix stored by column (used inline below on `dls_cols`
column views). */

fn IJKth(i: usize, jx: usize, jy: usize) -> usize {
    i - 1 + jx * NUM_SPECIES + jy * NSMX
}

/* Type : UserData
contains preconditioner blocks, pivot arrays, and problem constants.
Each 2x2 preconditioner/Jacobian block is stored flat, column-major
(equivalent to the C SUNDlsMat_newDenseMat small-matrix storage);
block (jx,jy) lives at index jx*MY + jy. */

struct UserData {
    P: Vec<[sunrealtype; NUM_SPECIES * NUM_SPECIES]>,
    Jbd: Vec<[sunrealtype; NUM_SPECIES * NUM_SPECIES]>,
    pivot: Vec<[sunindextype; NUM_SPECIES]>,
    q4: sunrealtype,
    om: sunrealtype,
    dx: sunrealtype,
    dy: sunrealtype,
    hdco: sunrealtype,
    haco: sunrealtype,
    vdco: sunrealtype,
}

/*
 *-------------------------------
 * Main Program
 *-------------------------------
 */

fn main() {
    /* Create the SUNDIALS context */
    let mut sunctx: Option<SUNContext> = None;
    let retval = SUNContext_Create(SUN_COMM_NULL, &mut sunctx);
    if check_retval(&retval, "SUNContext_Create") != 0 {
        std::process::exit(1);
    }
    let sunctx = sunctx.unwrap();

    /* Allocate memory, and set problem data, initial values, tolerances */
    let u = N_VNew_Serial(NEQ as sunindextype, &sunctx);
    if check_ptr(&u, "N_VNew_Serial", 0) != 0 {
        std::process::exit(1);
    }
    let u = u.unwrap();
    let mut data = AllocUserData();
    InitUserData(&mut data);
    SetInitialProfiles(&u, data.dx, data.dy);
    let abstol = ATOL;
    let reltol = RTOL;

    /* Call CVodeCreate to create the solver memory and specify the
     * Backward Differentiation Formula */
    let cvode_mem = CVodeCreate(CV_BDF, &sunctx);
    if check_ptr(&cvode_mem, "CVodeCreate", 0) != 0 {
        std::process::exit(1);
    }
    let cvode_mem = cvode_mem.unwrap();

    /* Set the pointer to user-defined data */
    let retval = CVodeSetUserData(&cvode_mem, Some(Box::new(data)));
    if check_retval(&retval, "CVodeSetUserData") != 0 {
        std::process::exit(1);
    }

    /* Call CVodeInit to initialize the integrator memory and specify the
     * user's right hand side function in u'=f(t,u), the initial time T0, and
     * the initial dependent variable vector u. */
    let retval = CVodeInit(&cvode_mem, f, T0, &u);
    if check_retval(&retval, "CVodeInit") != 0 {
        std::process::exit(1);
    }

    /* Call CVodeSStolerances to specify the scalar relative tolerance
     * and scalar absolute tolerances */
    let retval = CVodeSStolerances(&cvode_mem, reltol, abstol);
    if check_retval(&retval, "CVodeSStolerances") != 0 {
        std::process::exit(1);
    }

    /* Call SUNLinSol_SPGMR to specify the linear solver SPGMR
     * with left preconditioning and the default Krylov dimension */
    let LS = SUNLinSol_SPGMR(&u, SUN_PREC_LEFT, 0, &sunctx);
    if check_ptr(&LS, "SUNLinSol_SPGMR", 0) != 0 {
        std::process::exit(1);
    }
    let LS = LS.unwrap();

    /* Call CVodeSetLinearSolver to attach the linear solver to CVode */
    let retval = CVodeSetLinearSolver(&cvode_mem, &LS, None);
    if check_retval(&retval, "CVodeSetLinearSolver") != 0 {
        std::process::exit(1);
    }

    /* set the JAcobian-times-vector function */
    let retval = CVodeSetJacTimes(&cvode_mem, None, Some(jtv));
    if check_retval(&retval, "CVodeSetJacTimes") != 0 {
        std::process::exit(1);
    }

    /* Set the preconditioner solve and setup functions */
    let retval = CVodeSetPreconditioner(&cvode_mem, Some(Precond), Some(PSolve));
    if check_retval(&retval, "CVodeSetPreconditioner") != 0 {
        std::process::exit(1);
    }

    /* In loop over output points, call CVode, print results, test for error */
    print!(" \n2-species diurnal advection-diffusion problem\n\n");
    let mut t: sunrealtype = 0.0;
    let mut tout = TWOHR;
    for _iout in 1..=NOUT {
        let retval = CVode(&cvode_mem, tout, &u, &mut t, CV_NORMAL);
        PrintOutput(&cvode_mem, &u, t);
        if check_retval(&retval, "CVode") != 0 {
            break;
        }
        tout += TWOHR;
    }

    PrintFinalStats(&cvode_mem);

    /* Free memory */
    N_VDestroy(u);
    let mut cvode_mem = Some(cvode_mem);
    CVodeFree(&mut cvode_mem); /* user data is dropped with the solver memory */
    SUNLinSolFree(Some(LS));
    let mut sunctx = Some(sunctx);
    SUNContext_Free(&mut sunctx);
}

/*
 *-------------------------------
 * Private helper functions
 *-------------------------------
 */

/* Allocate memory for data structure of type UserData */

fn AllocUserData() -> UserData {
    UserData {
        P: vec![[ZERO; NUM_SPECIES * NUM_SPECIES]; MX * MY],
        Jbd: vec![[ZERO; NUM_SPECIES * NUM_SPECIES]; MX * MY],
        pivot: vec![[0; NUM_SPECIES]; MX * MY],
        q4: ZERO,
        om: ZERO,
        dx: ZERO,
        dy: ZERO,
        hdco: ZERO,
        haco: ZERO,
        vdco: ZERO,
    }
}

/* Load problem constants in data */

fn InitUserData(data: &mut UserData) {
    data.om = PI / HALFDAY;
    data.dx = (XMAX - XMIN) / ((MX - 1) as sunrealtype);
    data.dy = (YMAX - YMIN) / ((MY - 1) as sunrealtype);
    data.hdco = KH / SQR(data.dx);
    data.haco = VEL / (TWO * data.dx);
    data.vdco = (ONE / SQR(data.dy)) * KV0;
}

/* Set initial conditions in u */

fn SetInitialProfiles(u: &N_Vector, dx: sunrealtype, dy: sunrealtype) {
    /* Set pointer to data array in vector u. */

    let mut udata = N_VGetArrayPointer(u).expect("N_VGetArrayPointer");

    /* Load initial profiles of c1 and c2 into u vector */

    for jy in 0..MY {
        let y = YMIN + (jy as sunrealtype) * dy;
        let mut cy = SQR(0.1 * (y - YMID));
        cy = ONE - cy + 0.5 * SQR(cy);
        for jx in 0..MX {
            let x = XMIN + (jx as sunrealtype) * dx;
            let mut cx = SQR(0.1 * (x - XMID));
            cx = ONE - cx + 0.5 * SQR(cx);
            udata[IJKth(1, jx, jy)] = C1_SCALE * cx * cy;
            udata[IJKth(2, jx, jy)] = C2_SCALE * cx * cy;
        }
    }
}

/* Print current t, step count, order, stepsize, and sampled c1,c2 values */

fn PrintOutput(cvode_mem: &CVodeMem, u: &N_Vector, t: sunrealtype) {
    let mut nst: i64 = 0;
    let mut qu: i32 = 0;
    let mut hu: sunrealtype = ZERO;
    let (mxh, myh, mx1, my1) = (MX / 2 - 1, MY / 2 - 1, MX - 1, MY - 1);

    let retval = CVodeGetNumSteps(cvode_mem, &mut nst);
    check_retval(&retval, "CVodeGetNumSteps");
    let retval = CVodeGetLastOrder(cvode_mem, &mut qu);
    check_retval(&retval, "CVodeGetLastOrder");
    let retval = CVodeGetLastStep(cvode_mem, &mut hu);
    check_retval(&retval, "CVodeGetLastStep");

    let udata = N_VGetArrayPointer(u).expect("N_VGetArrayPointer");

    print!(
        "t = {}   no. steps = {}   order = {}   stepsize = {}\n",
        fmt_e(t, 2),
        nst,
        qu,
        fmt_e(hu, 2)
    );
    print!(
        "c1 (bot.left/middle/top rt.) = {}  {}  {}\n",
        fmt_ew(udata[IJKth(1, 0, 0)], 12, 3),
        fmt_ew(udata[IJKth(1, mxh, myh)], 12, 3),
        fmt_ew(udata[IJKth(1, mx1, my1)], 12, 3)
    );
    print!(
        "c2 (bot.left/middle/top rt.) = {}  {}  {}\n\n",
        fmt_ew(udata[IJKth(2, 0, 0)], 12, 3),
        fmt_ew(udata[IJKth(2, mxh, myh)], 12, 3),
        fmt_ew(udata[IJKth(2, mx1, my1)], 12, 3)
    );
}

/* Get and print final statistics */

fn PrintFinalStats(cvode_mem: &CVodeMem) {
    let (mut lenrw, mut leniw): (i64, i64) = (0, 0);
    let (mut lenrwLS, mut leniwLS): (i64, i64) = (0, 0);
    let (mut nst, mut nfe, mut nsetups, mut nni, mut ncfn, mut netf): (
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
    ) = (0, 0, 0, 0, 0, 0);
    let (mut nli, mut npe, mut nps, mut ncfl, mut nfeLS): (i64, i64, i64, i64, i64) =
        (0, 0, 0, 0, 0);

    let retval = CVodeGetWorkSpace(cvode_mem, &mut lenrw, &mut leniw);
    check_retval(&retval, "CVodeGetWorkSpace");
    let retval = CVodeGetNumSteps(cvode_mem, &mut nst);
    check_retval(&retval, "CVodeGetNumSteps");
    let retval = CVodeGetNumRhsEvals(cvode_mem, &mut nfe);
    check_retval(&retval, "CVodeGetNumRhsEvals");
    let retval = CVodeGetNumLinSolvSetups(cvode_mem, &mut nsetups);
    check_retval(&retval, "CVodeGetNumLinSolvSetups");
    let retval = CVodeGetNumErrTestFails(cvode_mem, &mut netf);
    check_retval(&retval, "CVodeGetNumErrTestFails");
    let retval = CVodeGetNumNonlinSolvIters(cvode_mem, &mut nni);
    check_retval(&retval, "CVodeGetNumNonlinSolvIters");
    let retval = CVodeGetNumNonlinSolvConvFails(cvode_mem, &mut ncfn);
    check_retval(&retval, "CVodeGetNumNonlinSolvConvFails");

    let retval = CVodeGetLinWorkSpace(cvode_mem, &mut lenrwLS, &mut leniwLS);
    check_retval(&retval, "CVodeGetLinWorkSpace");
    let retval = CVodeGetNumLinIters(cvode_mem, &mut nli);
    check_retval(&retval, "CVodeGetNumLinIters");
    let retval = CVodeGetNumPrecEvals(cvode_mem, &mut npe);
    check_retval(&retval, "CVodeGetNumPrecEvals");
    let retval = CVodeGetNumPrecSolves(cvode_mem, &mut nps);
    check_retval(&retval, "CVodeGetNumPrecSolves");
    let retval = CVodeGetNumLinConvFails(cvode_mem, &mut ncfl);
    check_retval(&retval, "CVodeGetNumLinConvFails");
    let retval = CVodeGetNumLinRhsEvals(cvode_mem, &mut nfeLS);
    check_retval(&retval, "CVodeGetNumLinRhsEvals");

    print!("\nFinal Statistics.. \n\n");
    print!("lenrw   = {:5}     leniw   = {:5}\n", lenrw, leniw);
    print!("lenrwLS = {:5}     leniwLS = {:5}\n", lenrwLS, leniwLS);
    print!("nst     = {:5}\n", nst);
    print!("nfe     = {:5}     nfeLS   = {:5}\n", nfe, nfeLS);
    print!("nni     = {:5}     nli     = {:5}\n", nni, nli);
    print!("nsetups = {:5}     netf    = {:5}\n", nsetups, netf);
    print!("npe     = {:5}     nps     = {:5}\n", npe, nps);
    print!("ncfn    = {:5}     ncfl    = {:5}\n\n", ncfn, ncfl);
}

/* Check function return value...
check_retval: SUNDIALS function returns an integer value so check if
              retval < 0 (C check_retval with opt == 1)
check_ptr:    function allocates memory so check if returned
              NULL pointer (C check_retval with opt == 0 or opt == 2) */

fn check_retval(retval: &i32, funcname: &str) -> i32 {
    if *retval < 0 {
        eprint!(
            "\nSUNDIALS_ERROR: {}() failed with retval = {}\n\n",
            funcname, retval
        );
        return 1;
    }
    0
}

fn check_ptr<T>(returnvalue: &Option<T>, funcname: &str, opt: i32) -> i32 {
    if returnvalue.is_none() {
        if opt == 0 {
            eprint!(
                "\nSUNDIALS_ERROR: {}() failed - returned NULL pointer\n\n",
                funcname
            );
        } else {
            eprint!(
                "\nMEMORY_ERROR: {}() failed - returned NULL pointer\n\n",
                funcname
            );
        }
        return 1;
    }
    0
}

/*
 *-------------------------------
 * Functions called by the solver
 *-------------------------------
 */

/* f routine. Compute RHS function f(t,u). */

fn f(t: sunrealtype, u: &N_Vector, udot: &N_Vector, user_data: &mut Option<Box<dyn Any>>) -> i32 {
    let data = user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<UserData>())
        .expect("user_data is UserData");
    let udata = N_VGetArrayPointer(u).expect("N_VGetArrayPointer");
    let mut dudata = N_VGetArrayPointer(udot).expect("N_VGetArrayPointer");

    /* Set diurnal rate coefficients. */

    let s = (data.om * t).sun_sin();
    let q3;
    if s > ZERO {
        q3 = (-A3 / s).sun_exp();
        data.q4 = (-A4 / s).sun_exp();
    } else {
        q3 = ZERO;
        data.q4 = ZERO;
    }

    /* Make local copies of problem variables, for efficiency. */

    let q4coef = data.q4;
    let dely = data.dy;
    let verdco = data.vdco;
    let hordco = data.hdco;
    let horaco = data.haco;

    /* Loop over all grid points. */

    for jy in 0..MY {
        /* Set vertical diffusion coefficients at jy +- 1/2 */

        let ydn = YMIN + (jy as sunrealtype - 0.5) * dely;
        let yup = ydn + dely;
        let cydn = verdco * (0.2 * ydn).sun_exp();
        let cyup = verdco * (0.2 * yup).sun_exp();
        let idn: i32 = if jy == 0 { 1 } else { -1 };
        let iup: i32 = if jy == MY - 1 { -1 } else { 1 };
        for jx in 0..MX {
            /* Extract c1 and c2, and set kinetic rate terms. */

            let c1 = udata[IJKth(1, jx, jy)];
            let c2 = udata[IJKth(2, jx, jy)];
            let qq1 = Q1 * c1 * C3;
            let qq2 = Q2 * c1 * c2;
            let qq3 = q3 * C3;
            let qq4 = q4coef * c2;
            let rkin1 = -qq1 - qq2 + TWO * qq3 + qq4;
            let rkin2 = qq1 - qq2 - qq4;

            /* Set vertical diffusion terms. */

            let c1dn = udata[IJKth(1, jx, (jy as i32 + idn) as usize)];
            let c2dn = udata[IJKth(2, jx, (jy as i32 + idn) as usize)];
            let c1up = udata[IJKth(1, jx, (jy as i32 + iup) as usize)];
            let c2up = udata[IJKth(2, jx, (jy as i32 + iup) as usize)];
            let vertd1 = cyup * (c1up - c1) - cydn * (c1 - c1dn);
            let vertd2 = cyup * (c2up - c2) - cydn * (c2 - c2dn);

            /* Set horizontal diffusion and advection terms. */

            let ileft: i32 = if jx == 0 { 1 } else { -1 };
            let iright: i32 = if jx == MX - 1 { -1 } else { 1 };
            let c1lt = udata[IJKth(1, (jx as i32 + ileft) as usize, jy)];
            let c2lt = udata[IJKth(2, (jx as i32 + ileft) as usize, jy)];
            let c1rt = udata[IJKth(1, (jx as i32 + iright) as usize, jy)];
            let c2rt = udata[IJKth(2, (jx as i32 + iright) as usize, jy)];
            let hord1 = hordco * (c1rt - TWO * c1 + c1lt);
            let hord2 = hordco * (c2rt - TWO * c2 + c2lt);
            let horad1 = horaco * (c1rt - c1lt);
            let horad2 = horaco * (c2rt - c2lt);

            /* Load all terms into udot. */

            dudata[IJKth(1, jx, jy)] = vertd1 + hord1 + horad1 + rkin1;
            dudata[IJKth(2, jx, jy)] = vertd2 + hord2 + horad2 + rkin2;
        }
    }

    0
}

/* Jacobian-times-vector routine. */

fn jtv(
    v: &N_Vector,
    Jv: &N_Vector,
    t: sunrealtype,
    u: &N_Vector,
    _fu: &N_Vector,
    user_data: &mut Option<Box<dyn Any>>,
    _tmp: &N_Vector,
) -> i32 {
    let data = user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<UserData>())
        .expect("user_data is UserData");

    let udata = N_VGetArrayPointer(u).expect("N_VGetArrayPointer");
    let vdata = N_VGetArrayPointer(v).expect("N_VGetArrayPointer");
    let mut Jvdata = N_VGetArrayPointer(Jv).expect("N_VGetArrayPointer");

    /* Set diurnal rate coefficients. */

    let s = (data.om * t).sun_sin();
    if s > ZERO {
        data.q4 = (-A4 / s).sun_exp();
    } else {
        data.q4 = ZERO;
    }

    /* Make local copies of problem variables, for efficiency. */

    let q4coef = data.q4;
    let dely = data.dy;
    let verdco = data.vdco;
    let hordco = data.hdco;
    let horaco = data.haco;

    /* Loop over all grid points. */

    for jy in 0..MY {
        /* Set vertical diffusion coefficients at jy +- 1/2 */

        let ydn = YMIN + (jy as sunrealtype - 0.5) * dely;
        let yup = ydn + dely;

        let cydn = verdco * (0.2 * ydn).sun_exp();
        let cyup = verdco * (0.2 * yup).sun_exp();

        let idn: i32 = if jy == 0 { 1 } else { -1 };
        let iup: i32 = if jy == MY - 1 { -1 } else { 1 };

        for jx in 0..MX {
            let mut Jv1 = ZERO;
            let mut Jv2 = ZERO;

            /* Extract c1 and c2 at the current location and at neighbors */

            let c1 = udata[IJKth(1, jx, jy)];
            let c2 = udata[IJKth(2, jx, jy)];

            let v1 = vdata[IJKth(1, jx, jy)];
            let v2 = vdata[IJKth(2, jx, jy)];

            let v1dn = vdata[IJKth(1, jx, (jy as i32 + idn) as usize)];
            let v2dn = vdata[IJKth(2, jx, (jy as i32 + idn) as usize)];
            let v1up = vdata[IJKth(1, jx, (jy as i32 + iup) as usize)];
            let v2up = vdata[IJKth(2, jx, (jy as i32 + iup) as usize)];

            let ileft: i32 = if jx == 0 { 1 } else { -1 };
            let iright: i32 = if jx == MX - 1 { -1 } else { 1 };

            let v1lt = vdata[IJKth(1, (jx as i32 + ileft) as usize, jy)];
            let v2lt = vdata[IJKth(2, (jx as i32 + ileft) as usize, jy)];
            let v1rt = vdata[IJKth(1, (jx as i32 + iright) as usize, jy)];
            let v2rt = vdata[IJKth(2, (jx as i32 + iright) as usize, jy)];

            /* Set kinetic rate terms. */

            Jv1 += -(Q1 * C3 + Q2 * c2) * v1 + (q4coef - Q2 * c1) * v2;
            Jv2 += (Q1 * C3 - Q2 * c2) * v1 - (q4coef + Q2 * c1) * v2;

            /* Set vertical diffusion terms. */

            Jv1 += -(cyup + cydn) * v1 + cyup * v1up + cydn * v1dn;
            Jv2 += -(cyup + cydn) * v2 + cyup * v2up + cydn * v2dn;

            /* Set horizontal diffusion and advection terms. */

            Jv1 += hordco * (v1rt - TWO * v1 + v1lt);
            Jv2 += hordco * (v2rt - TWO * v2 + v2lt);

            Jv1 += horaco * (v1rt - v1lt);
            Jv2 += horaco * (v2rt - v2lt);

            /* Load two components of J*v */

            Jvdata[IJKth(1, jx, jy)] = Jv1;
            Jvdata[IJKth(2, jx, jy)] = Jv2;
        }
    }

    0
}

/* Preconditioner setup routine. Generate and preprocess P. */

fn Precond(
    _tn: sunrealtype,
    u: &N_Vector,
    _fu: &N_Vector,
    jok: sunbooleantype,
    jcurPtr: &mut sunbooleantype,
    gamma: sunrealtype,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    /* Make local copies of pointers in user_data, and of pointer to u's data */

    let data = user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<UserData>())
        .expect("user_data is UserData");

    if jok {
        /* jok = SUNTRUE: Copy Jbd to P */

        for jy in 0..MY {
            for jx in 0..MX {
                let idx = jx * MY + jy;
                let j = dls_cols(&mut data.Jbd[idx], NUM_SPECIES as sunindextype);
                let mut a = dls_cols(&mut data.P[idx], NUM_SPECIES as sunindextype);
                SUNDlsMat_denseCopy(
                    &j,
                    &mut a,
                    NUM_SPECIES as sunindextype,
                    NUM_SPECIES as sunindextype,
                );
            }
        }

        *jcurPtr = false;
    } else {
        /* jok = SUNFALSE: Generate Jbd from scratch and copy to P */

        /* Make local copies of problem variables, for efficiency. */

        let q4coef = data.q4;
        let dely = data.dy;
        let verdco = data.vdco;
        let hordco = data.hdco;

        /* Compute 2x2 diagonal Jacobian blocks (using q4 values
        computed on the last f call).  Load into P. */

        let udata = N_VGetArrayPointer(u).expect("N_VGetArrayPointer");

        for jy in 0..MY {
            let ydn = YMIN + (jy as sunrealtype - 0.5) * dely;
            let yup = ydn + dely;
            let cydn = verdco * (0.2 * ydn).sun_exp();
            let cyup = verdco * (0.2 * yup).sun_exp();
            let diag = -(cydn + cyup + TWO * hordco);
            for jx in 0..MX {
                let c1 = udata[IJKth(1, jx, jy)];
                let c2 = udata[IJKth(2, jx, jy)];
                let idx = jx * MY + jy;
                let mut j = dls_cols(&mut data.Jbd[idx], NUM_SPECIES as sunindextype);
                let mut a = dls_cols(&mut data.P[idx], NUM_SPECIES as sunindextype);
                /* IJth(j, i, jcol) = j[jcol - 1][i - 1] */
                j[0][0] = (-Q1 * C3 - Q2 * c2) + diag; /* IJth(j, 1, 1) */
                j[1][0] = -Q2 * c1 + q4coef; /*            IJth(j, 1, 2) */
                j[0][1] = Q1 * C3 - Q2 * c2; /*            IJth(j, 2, 1) */
                j[1][1] = (-Q2 * c1 - q4coef) + diag; /*   IJth(j, 2, 2) */
                SUNDlsMat_denseCopy(
                    &j,
                    &mut a,
                    NUM_SPECIES as sunindextype,
                    NUM_SPECIES as sunindextype,
                );
            }
        }

        *jcurPtr = true;
    }

    /* Scale by -gamma */

    for jy in 0..MY {
        for jx in 0..MX {
            let idx = jx * MY + jy;
            let mut a = dls_cols(&mut data.P[idx], NUM_SPECIES as sunindextype);
            SUNDlsMat_denseScale(
                -gamma,
                &mut a,
                NUM_SPECIES as sunindextype,
                NUM_SPECIES as sunindextype,
            );
        }
    }

    /* Add identity matrix and do LU decompositions on blocks in place. */

    for jx in 0..MX {
        for jy in 0..MY {
            let idx = jx * MY + jy;
            let mut a = dls_cols(&mut data.P[idx], NUM_SPECIES as sunindextype);
            SUNDlsMat_denseAddIdentity(&mut a, NUM_SPECIES as sunindextype);
            let retval = SUNDlsMat_denseGETRF(
                &mut a,
                NUM_SPECIES as sunindextype,
                NUM_SPECIES as sunindextype,
                &mut data.pivot[idx],
            );
            if retval != 0 {
                return 1;
            }
        }
    }

    0
}

/* Preconditioner solve routine */

fn PSolve(
    _tn: sunrealtype,
    _u: &N_Vector,
    _fu: &N_Vector,
    r: &N_Vector,
    z: &N_Vector,
    _gamma: sunrealtype,
    _delta: sunrealtype,
    _lr: i32,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    /* Extract the P and pivot arrays from user_data. */

    let data = user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<UserData>())
        .expect("user_data is UserData");

    N_VScale(ONE, r, z);

    /* Solve the block-diagonal system Px = r using LU factors stored
    in P and pivot data in pivot, and return the solution in z. */

    let mut zdata = N_VGetArrayPointer(z).expect("N_VGetArrayPointer");

    for jx in 0..MX {
        for jy in 0..MY {
            let idx = jx * MY + jy;
            let base = IJKth(1, jx, jy);
            let mut a = dls_cols(&mut data.P[idx], NUM_SPECIES as sunindextype);
            SUNDlsMat_denseGETRS(
                &mut a,
                NUM_SPECIES as sunindextype,
                &data.pivot[idx],
                &mut zdata[base..base + NUM_SPECIES],
            );
        }
    }

    0
}
