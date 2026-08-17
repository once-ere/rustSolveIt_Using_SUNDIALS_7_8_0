#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

/* -----------------------------------------------------------------
 * Ported from: examples/cvodes/serial/cvsDiurnal_FSA_kry.c
 * -----------------------------------------------------------------
 * Example problem:
 *
 * An ODE system is generated from the following 2-species diurnal
 * kinetics advection-diffusion PDE system in 2 space dimensions:
 *
 * dc(i)/dt = Kh*(d/dx)^2 c(i) + V*dc(i)/dx + (d/dz)(Kv(z)*dc(i)/dz)
 *                 + Ri(c1,c2,t)      for i = 1,2,   where
 *   R1(c1,c2,t) = -q1*c1*c3 - q2*c1*c2 + 2*q3(t)*c3 + q4(t)*c2 ,
 *   R2(c1,c2,t) =  q1*c1*c3 - q2*c1*c2 - q4(t)*c2 ,
 *   Kv(z) = Kv0*exp(z/5) ,
 * Kh, V, Kv0, q1, q2, and c3 are constants, and q3(t) and q4(t)
 * vary diurnally. The problem is posed on the square
 *   0 <= x <= 20,    30 <= z <= 50   (all in km),
 * with homogeneous Neumann boundary conditions, and for time t in
 *   0 <= t <= 86400 sec (1 day).
 * The PDE system is treated by central differences on a uniform
 * 10 x 10 mesh, with simple polynomial initial profiles.
 * The problem is solved with CVODES, with the BDF/GMRES method
 * (i.e. using the SUNLinSol_SPGMR linear solver) and the block-diagonal
 * part of the Newton matrix as a left preconditioner. A copy of
 * the block-diagonal part of the Jacobian is saved and
 * conditionally reused within the Precond routine.
 *
 * Optionally, CVODES can compute sensitivities with respect to the
 * problem parameters q1 and q2.
 * Any of three sensitivity methods (SIMULTANEOUS, STAGGERED, and
 * STAGGERED1) can be used and sensitivities may be included in the
 * error test or not (error control set on FULL or PARTIAL,
 * respectively).
 *
 * Execution:
 *
 * If no sensitivities are desired:
 *    % cvsDiurnal_FSA_kry -nosensi
 * If sensitivities are to be computed:
 *    % cvsDiurnal_FSA_kry -sensi sensi_meth err_con
 * where sensi_meth is one of {sim, stg, stg1} and err_con is one of
 * {t, f}.
 * -----------------------------------------------------------------*/

use cvodes_rs::prelude::*;
use cvodes_rs::sundials_dense::{
    SUNDlsMat_denseAddIdentity, SUNDlsMat_denseCopy, SUNDlsMat_denseGETRF, SUNDlsMat_denseGETRS,
    SUNDlsMat_denseScale,
};
use cvodes_rs::sundials_direct::dls_cols;

use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

/* helpful macros */

fn SQR(A: sunrealtype) -> sunrealtype {
    A * A
}

/* Problem Constants */

const NUM_SPECIES: usize = 2; /* number of species */
const C1_SCALE: sunrealtype = 1.0e6; /* coefficients in initial profiles */
const C2_SCALE: sunrealtype = 1.0e12;

const T0: sunrealtype = 0.0; /* initial time */
const NOUT: i32 = 12; /* number of output times */
const TWOHR: sunrealtype = 7200.0; /* number of seconds in two hours  */
const HALFDAY: sunrealtype = 4.32e4; /* number of seconds in a half day */
const PI: sunrealtype = 3.1415926535898; /* pi */

const XMIN: sunrealtype = 0.0; /* grid boundaries in x  */
const XMAX: sunrealtype = 20.0;
const ZMIN: sunrealtype = 30.0; /* grid boundaries in z  */
const ZMAX: sunrealtype = 50.0;
const XMID: sunrealtype = 10.0; /* grid midpoints in x,z */
const ZMID: sunrealtype = 40.0;

const MX: usize = 15; /* MX = number of x mesh points */
const MZ: usize = 15; /* MZ = number of z mesh points */
const NSMX: usize = NUM_SPECIES * MX; /* NSMX = NUM_SPECIES*MX */
const MM: usize = MX * MZ; /* MM = MX*MZ */

/* CVodeInit Constants */
const RTOL: sunrealtype = 1.0e-5; /* scalar relative tolerance */
const FLOOR: sunrealtype = 100.0; /* value of C1 or C2 at which tolerances */
/* change from relative to absolute      */
const ATOL: sunrealtype = RTOL * FLOOR; /* scalar absolute tolerance */
const NEQ: usize = NUM_SPECIES * MM; /* NEQ = number of equations */

/* Sensitivity Constants */
const NP: usize = 8;
const NS: i32 = 2;

const ZERO: sunrealtype = 0.0;
const ONE: sunrealtype = 1.0;

/* User-defined vector and matrix accessor helpers: IJKth, IJth.

IJKth is defined in order to isolate the translation from the
mathematical 3-dimensional structure of the dependent variable vector
to the underlying 1-dimensional storage.

IJKth(vdata,i,j,k) references the element in the vdata array for
species i at mesh point (j,k), where 1 <= i <= NUM_SPECIES,
0 <= j <= MX-1, 0 <= k <= MZ-1.

IJth(a,i,j) = a[j-1][i-1] references the (i,j)th entry of a small
dense matrix stored by column (used inline below on `dls_cols`
column views). */

fn IJKth(i: usize, j: usize, k: usize) -> usize {
    i - 1 + j * NUM_SPECIES + k * NSMX
}

/* Type : UserData
contains preconditioner blocks, pivot arrays,
problem parameters, and problem constants.

Each 2x2 preconditioner/Jacobian block is stored flat, column-major
(equivalent to the C SUNDlsMat_newDenseMat small-matrix storage);
block (jx,jz) lives at index jx*MZ + jz. */

struct UserData {
    /* C passes `data->p` (the caller's own array) to CVodeSetSensParams,
    which stores the POINTER in `cv_mem->cv_p`. The internal DQ sensitivity
    RHS therefore perturbs the very array that `f` and `Precond` read back
    through `user_data`. The port shares that array as a `SensParams`
    handle: `main` hands CVODES a clone of this very `Rc`, so the
    perturbations land here, exactly as in C. */
    p: SensParams,
    P: Vec<[sunrealtype; NUM_SPECIES * NUM_SPECIES]>,
    Jbd: Vec<[sunrealtype; NUM_SPECIES * NUM_SPECIES]>,
    pivot: Vec<[sunindextype; NUM_SPECIES]>,
    q4: sunrealtype,
    om: sunrealtype,
    dx: sunrealtype,
    dz: sunrealtype,
    hdco: sunrealtype,
    haco: sunrealtype,
    vdco: sunrealtype,
}

/* Snapshot of `data->p` as the C callbacks read it (`Q1 = data->p[0];`
…). While the internal DQ sensitivity RHS is running, the entry for the
active parameter carries the perturbation CVODES just wrote through the
shared handle. The borrow is released before the caller does anything
else with `data`. */

fn UserDataParams(data: &UserData) -> [sunrealtype; NP] {
    let mut p: [sunrealtype; NP] = [ZERO; NP];
    p.copy_from_slice(&data.p.borrow());
    p
}

/*
 *--------------------------------------------------------------------
 * MAIN PROGRAM
 *--------------------------------------------------------------------
 */

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let argc = argv.len() as i32;

    let mut sensi: sunbooleantype = false;
    let mut sensi_meth: i32 = -1;
    let mut err_con: sunbooleantype = false;

    /* Process arguments */
    ProcessArgs(argc, &argv, &mut sensi, &mut sensi_meth, &mut err_con);

    /* Problem parameters */
    let mut data = AllocUserData();
    if check_ptr(&Some(&data), "AllocUserData", 2) != 0 {
        std::process::exit(1);
    }
    InitUserData(&mut data);

    /* Create the SUNDIALS simulation context that all SUNDIALS objects require */
    let mut sunctx: Option<SUNContext> = None;
    let retval = SUNContext_Create(SUN_COMM_NULL, &mut sunctx);
    if check_retval(&retval, "SUNContext_Create") != 0 {
        std::process::exit(1);
    }
    let sunctx = sunctx.unwrap();

    /* Initial states */
    let y = N_VNew_Serial(NEQ as sunindextype, &sunctx);
    if check_ptr(&y, "N_VNew_Serial", 0) != 0 {
        std::process::exit(1);
    }
    let y = y.unwrap();
    SetInitialProfiles(&y, data.dx, data.dz);

    /* Tolerances */
    let abstol = ATOL;
    let reltol = RTOL;

    /* Create CVODES object */
    let cvode_mem = CVodeCreate(CV_BDF, &sunctx);
    if check_ptr(&cvode_mem, "CVodeCreate", 0) != 0 {
        std::process::exit(1);
    }
    let cvode_mem = cvode_mem.unwrap();

    /* Keep a handle on the parameter array before ownership of `data`
    moves into the solver memory: this clone IS `data->p` (C keeps its own
    `data` pointer and hands that same array to CVODES). */
    let p: SensParams = data.p.clone();

    let retval = CVodeSetUserData(&cvode_mem, Some(Box::new(data)));
    if check_retval(&retval, "CVodeSetUserData") != 0 {
        std::process::exit(1);
    }

    let retval = CVodeSetMaxNumSteps(&cvode_mem, 2000);
    if check_retval(&retval, "CVodeSetMaxNumSteps") != 0 {
        std::process::exit(1);
    }

    /* Allocate CVODES memory */
    let retval = CVodeInit(&cvode_mem, f, T0, &y);
    if check_retval(&retval, "CVodeInit") != 0 {
        std::process::exit(1);
    }

    let retval = CVodeSStolerances(&cvode_mem, reltol, abstol);
    if check_retval(&retval, "CVodeSStolerances") != 0 {
        std::process::exit(1);
    }

    /* Create the SUNLinSol_SPGMR linear solver with left
    preconditioning and the default Krylov dimension */
    let LS = SUNLinSol_SPGMR(&y, SUN_PREC_LEFT, 0, &sunctx);
    if check_ptr(&LS, "SUNLinSol_SPGMR", 0) != 0 {
        std::process::exit(1);
    }
    let LS = LS.unwrap();

    /* Attach the linear solver */
    let retval = CVodeSetLinearSolver(&cvode_mem, &LS, None);
    if check_retval(&retval, "CVodeSetLinearSolver") != 0 {
        std::process::exit(1);
    }

    /* Set the preconditioner solve and setup functions */
    let retval = CVodeSetPreconditioner(&cvode_mem, Some(Precond), Some(PSolve));
    if check_retval(&retval, "CVodeSetPreconditioner") != 0 {
        std::process::exit(1);
    }

    print!("\n2-species diurnal advection-diffusion problem\n");

    /* Forward sensitivity analysis */
    let mut uS: Option<Vec<N_Vector>> = None;
    if sensi {
        let mut plist: Vec<i32> = vec![0; NS as usize];
        if check_ptr(&Some(&plist), "malloc", 2) != 0 {
            std::process::exit(1);
        }
        for is in 0..NS as usize {
            plist[is] = is as i32;
        }

        let mut pbar: Vec<sunrealtype> = vec![ZERO; NS as usize];
        if check_ptr(&Some(&pbar), "malloc", 2) != 0 {
            std::process::exit(1);
        }
        for is in 0..NS as usize {
            pbar[is] = p.borrow()[plist[is] as usize];
        }

        let uSv = N_VCloneVectorArray(NS, &y);
        if check_ptr(&uSv, "N_VCloneVectorArray", 0) != 0 {
            std::process::exit(1);
        }
        let uSv = uSv.unwrap();
        for is in 0..NS as usize {
            N_VConst(ZERO, &uSv[is]);
        }

        let retval = CVodeSensInit1(&cvode_mem, NS, sensi_meth, None, &uSv);
        if check_retval(&retval, "CVodeSensInit") != 0 {
            std::process::exit(1);
        }

        let retval = CVodeSensEEtolerances(&cvode_mem);
        if check_retval(&retval, "CVodeSensEEtolerances") != 0 {
            std::process::exit(1);
        }

        let retval = CVodeSetSensErrCon(&cvode_mem, err_con);
        if check_retval(&retval, "CVodeSetSensErrCon") != 0 {
            std::process::exit(1);
        }

        let retval = CVodeSetSensDQMethod(&cvode_mem, CV_CENTERED, ZERO);
        if check_retval(&retval, "CVodeSetSensDQMethod") != 0 {
            std::process::exit(1);
        }

        /* Hand CVODES a CLONE of the handle `data` keeps (C hands it the
        `data->p` pointer): the internal DQ perturbations then reach `f`
        and `Precond` through `user_data`. */
        let retval = CVodeSetSensParams(&cvode_mem, Some(p.clone()), Some(&pbar), Some(&plist));
        if check_retval(&retval, "CVodeSetSensParams") != 0 {
            std::process::exit(1);
        }

        print!("Sensitivity: YES ");
        if sensi_meth == CV_SIMULTANEOUS {
            print!("( SIMULTANEOUS +");
        } else if sensi_meth == CV_STAGGERED {
            print!("( STAGGERED +");
        } else {
            print!("( STAGGERED1 +");
        }
        if err_con {
            print!(" FULL ERROR CONTROL )");
        } else {
            print!(" PARTIAL ERROR CONTROL )");
        }

        uS = Some(uSv);
    } else {
        print!("Sensitivity: NO ");
    }

    /* In loop over output points, call CVode, print results, test for error */

    print!("\n\n");
    print!("=====================================================================");
    print!("===\n");
    print!("     T     Q       H      NST                    Bottom left  Top ");
    print!("right \n");
    print!("=====================================================================");
    print!("===\n");

    let mut t: sunrealtype = ZERO;
    let mut tout = TWOHR;
    for _iout in 1..=NOUT {
        let retval = CVode(&cvode_mem, tout, &y, &mut t, CV_NORMAL);
        if check_retval(&retval, "CVode") != 0 {
            break;
        }
        PrintOutput(&cvode_mem, t, &y);
        if sensi {
            let uSv = uS.as_ref().expect("uS allocated");
            let retval = CVodeGetSens(&cvode_mem, &mut t, uSv);
            if check_retval(&retval, "CVodeGetSens") != 0 {
                break;
            }
            PrintOutputS(uSv);
        }

        print!("-------------------------------------------------------------------");
        print!("-----\n");

        tout += TWOHR;
    }

    /* Print final statistics */
    PrintFinalStats(&cvode_mem, sensi, err_con, sensi_meth);

    /* Free memory */
    N_VDestroy(y);
    if sensi {
        N_VDestroyVectorArray(uS.unwrap(), NS);
        /* pbar and plist are dropped with their scopes */
    }
    /* FreeUserData: the user data is dropped with the solver memory */
    let mut cvode_mem = Some(cvode_mem);
    CVodeFree(&mut cvode_mem);
    SUNLinSolFree(Some(LS));
    let mut sunctx = Some(sunctx);
    SUNContext_Free(&mut sunctx);
}

/*
 *--------------------------------------------------------------------
 * FUNCTIONS CALLED BY CVODES
 *--------------------------------------------------------------------
 */

/*
 * f routine. Compute f(t,y).
 */

fn f(t: sunrealtype, y: &N_Vector, ydot: &N_Vector, user_data: &mut Option<Box<dyn Any>>) -> i32 {
    let data = user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<UserData>())
        .expect("user_data is UserData");

    let ydata = N_VGetArrayPointer(y).expect("N_VGetArrayPointer");
    let mut dydata = N_VGetArrayPointer(ydot).expect("N_VGetArrayPointer");

    /* Load problem coefficients and parameters */

    let pp = UserDataParams(data);
    let Q1 = pp[0];
    let Q2 = pp[1];
    let C3 = pp[2];
    let A3 = pp[3];
    let A4 = pp[4];

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
    let delz = data.dz;
    let verdco = data.vdco;
    let hordco = data.hdco;
    let horaco = data.haco;

    /* Loop over all grid points. */

    for jz in 0..MZ {
        /* Set vertical diffusion coefficients at jz +- 1/2 */

        let zdn = ZMIN + (jz as sunrealtype - 0.5) * delz;
        let zup = zdn + delz;
        let czdn = verdco * (0.2 * zdn).sun_exp();
        let czup = verdco * (0.2 * zup).sun_exp();
        let idn: i32 = if jz == 0 { 1 } else { -1 };
        let iup: i32 = if jz == MZ - 1 { -1 } else { 1 };
        for jx in 0..MX {
            /* Extract c1 and c2, and set kinetic rate terms. */

            let c1 = ydata[IJKth(1, jx, jz)];
            let c2 = ydata[IJKth(2, jx, jz)];
            let qq1 = Q1 * c1 * C3;
            let qq2 = Q2 * c1 * c2;
            let qq3 = q3 * C3;
            let qq4 = q4coef * c2;
            let rkin1 = -qq1 - qq2 + 2.0 * qq3 + qq4;
            let rkin2 = qq1 - qq2 - qq4;

            /* Set vertical diffusion terms. */

            let c1dn = ydata[IJKth(1, jx, (jz as i32 + idn) as usize)];
            let c2dn = ydata[IJKth(2, jx, (jz as i32 + idn) as usize)];
            let c1up = ydata[IJKth(1, jx, (jz as i32 + iup) as usize)];
            let c2up = ydata[IJKth(2, jx, (jz as i32 + iup) as usize)];
            let vertd1 = czup * (c1up - c1) - czdn * (c1 - c1dn);
            let vertd2 = czup * (c2up - c2) - czdn * (c2 - c2dn);

            /* Set horizontal diffusion and advection terms. */

            let ileft: i32 = if jx == 0 { 1 } else { -1 };
            let iright: i32 = if jx == MX - 1 { -1 } else { 1 };
            let c1lt = ydata[IJKth(1, (jx as i32 + ileft) as usize, jz)];
            let c2lt = ydata[IJKth(2, (jx as i32 + ileft) as usize, jz)];
            let c1rt = ydata[IJKth(1, (jx as i32 + iright) as usize, jz)];
            let c2rt = ydata[IJKth(2, (jx as i32 + iright) as usize, jz)];
            let hord1 = hordco * (c1rt - 2.0 * c1 + c1lt);
            let hord2 = hordco * (c2rt - 2.0 * c2 + c2lt);
            let horad1 = horaco * (c1rt - c1lt);
            let horad2 = horaco * (c2rt - c2lt);

            /* Load all terms into ydot. */

            dydata[IJKth(1, jx, jz)] = vertd1 + hord1 + horad1 + rkin1;
            dydata[IJKth(2, jx, jz)] = vertd2 + hord2 + horad2 + rkin2;
        }
    }

    0
}

/*
 * Preconditioner setup routine. Generate and preprocess P.
 */

fn Precond(
    _tn: sunrealtype,
    y: &N_Vector,
    _fy: &N_Vector,
    jok: sunbooleantype,
    jcurPtr: &mut sunbooleantype,
    gamma: sunrealtype,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    /* Make local copies of pointers in user_data, and of pointer to y's data */
    let data = user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<UserData>())
        .expect("user_data is UserData");

    /* Load problem coefficients and parameters */
    let pp = UserDataParams(data);
    let Q1 = pp[0];
    let Q2 = pp[1];
    let C3 = pp[2];

    if jok {
        /* jok = SUNTRUE: Copy Jbd to P */

        for jz in 0..MZ {
            for jx in 0..MX {
                let idx = jx * MZ + jz;
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
        let delz = data.dz;
        let verdco = data.vdco;
        let hordco = data.hdco;

        /* Compute 2x2 diagonal Jacobian blocks (using q4 values
        computed on the last f call).  Load into P. */

        let ydata = N_VGetArrayPointer(y).expect("N_VGetArrayPointer");

        for jz in 0..MZ {
            let zdn = ZMIN + (jz as sunrealtype - 0.5) * delz;
            let zup = zdn + delz;
            let czdn = verdco * (0.2 * zdn).sun_exp();
            let czup = verdco * (0.2 * zup).sun_exp();
            let diag = -(czdn + czup + 2.0 * hordco);
            for jx in 0..MX {
                let c1 = ydata[IJKth(1, jx, jz)];
                let c2 = ydata[IJKth(2, jx, jz)];
                let idx = jx * MZ + jz;
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

    for jz in 0..MZ {
        for jx in 0..MX {
            let idx = jx * MZ + jz;
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
        for jz in 0..MZ {
            let idx = jx * MZ + jz;
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

/*
 * Preconditioner solve routine
 */

fn PSolve(
    _tn: sunrealtype,
    _y: &N_Vector,
    _fy: &N_Vector,
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
        for jz in 0..MZ {
            let idx = jx * MZ + jz;
            let v = IJKth(1, jx, jz);
            let mut a = dls_cols(&mut data.P[idx], NUM_SPECIES as sunindextype);
            SUNDlsMat_denseGETRS(
                &mut a,
                NUM_SPECIES as sunindextype,
                &data.pivot[idx],
                &mut zdata[v..v + NUM_SPECIES],
            );
        }
    }

    0
}

/*
 *--------------------------------------------------------------------
 * PRIVATE FUNCTIONS
 *--------------------------------------------------------------------
 */

/*
 * Process and verify arguments to cvsfwdkryx.
 */

fn ProcessArgs(
    argc: i32,
    argv: &[String],
    sensi: &mut sunbooleantype,
    sensi_meth: &mut i32,
    err_con: &mut sunbooleantype,
) {
    *sensi = false;
    *sensi_meth = -1;
    *err_con = false;

    if argc < 2 {
        WrongArgs(&argv[0]);
    }

    if argv[1] == "-nosensi" {
        *sensi = false;
    } else if argv[1] == "-sensi" {
        *sensi = true;
    } else {
        WrongArgs(&argv[0]);
    }

    if *sensi {
        if argc != 4 {
            WrongArgs(&argv[0]);
        }

        if argv[2] == "sim" {
            *sensi_meth = CV_SIMULTANEOUS;
        } else if argv[2] == "stg" {
            *sensi_meth = CV_STAGGERED;
        } else if argv[2] == "stg1" {
            *sensi_meth = CV_STAGGERED1;
        } else {
            WrongArgs(&argv[0]);
        }

        if argv[3] == "t" {
            *err_con = true;
        } else if argv[3] == "f" {
            *err_con = false;
        } else {
            WrongArgs(&argv[0]);
        }
    }
}

fn WrongArgs(name: &str) -> ! {
    print!("\nUsage: {} [-nosensi] [-sensi sensi_meth err_con]\n", name);
    print!("         sensi_meth = sim, stg, or stg1\n");
    print!("         err_con    = t or f\n");

    std::process::exit(0);
}

/*
 * Allocate memory for data structure of type UserData
 */

fn AllocUserData() -> UserData {
    UserData {
        p: Rc::new(RefCell::new(vec![ZERO; NP])),
        P: vec![[ZERO; NUM_SPECIES * NUM_SPECIES]; MX * MZ],
        Jbd: vec![[ZERO; NUM_SPECIES * NUM_SPECIES]; MX * MZ],
        pivot: vec![[0; NUM_SPECIES]; MX * MZ],
        q4: ZERO,
        om: ZERO,
        dx: ZERO,
        dz: ZERO,
        hdco: ZERO,
        haco: ZERO,
        vdco: ZERO,
    }
}

/*
 * Load problem constants in data
 */

fn InitUserData(data: &mut UserData) {
    /* Set problem parameters */
    let Q1: sunrealtype = 1.63e-16; /* Q1  coefficients q1, q2, c3             */
    let Q2: sunrealtype = 4.66e-16; /* Q2                                      */
    let C3: sunrealtype = 3.7e16; /* C3                                      */
    let A3: sunrealtype = 22.62; /* A3  coefficient in expression for q3(t) */
    let A4: sunrealtype = 7.601; /* A4  coefficient in expression for q4(t) */
    let KH: sunrealtype = 4.0e-6; /* KH  horizontal diffusivity Kh           */
    let VEL: sunrealtype = 0.001; /* VEL advection velocity V                */
    let KV0: sunrealtype = 1.0e-8; /* KV0 coefficient in Kv(z)                */

    data.om = PI / HALFDAY;
    data.dx = (XMAX - XMIN) / ((MX - 1) as sunrealtype);
    data.dz = (ZMAX - ZMIN) / ((MZ - 1) as sunrealtype);
    data.hdco = KH / SQR(data.dx);
    data.haco = VEL / (2.0 * data.dx);
    data.vdco = (ONE / SQR(data.dz)) * KV0;

    /* One borrow of the shared parameter array for the eight stores (no
    callback can run in between). */
    {
        let mut p = data.p.borrow_mut();
        p[0] = Q1;
        p[1] = Q2;
        p[2] = C3;
        p[3] = A3;
        p[4] = A4;
        p[5] = KH;
        p[6] = VEL;
        p[7] = KV0;
    }
}

/*
 * Set initial conditions in y
 */

fn SetInitialProfiles(y: &N_Vector, dx: sunrealtype, dz: sunrealtype) {
    /* Set pointer to data array in vector y. */

    let mut ydata = N_VGetArrayPointer(y).expect("N_VGetArrayPointer");

    /* Load initial profiles of c1 and c2 into y vector */

    for jz in 0..MZ {
        let z = ZMIN + (jz as sunrealtype) * dz;
        let mut cz = SQR(0.1 * (z - ZMID));
        cz = ONE - cz + 0.5 * SQR(cz);
        for jx in 0..MX {
            let x = XMIN + (jx as sunrealtype) * dx;
            let mut cx = SQR(0.1 * (x - XMID));
            cx = ONE - cx + 0.5 * SQR(cx);
            ydata[IJKth(1, jx, jz)] = C1_SCALE * cx * cz;
            ydata[IJKth(2, jx, jz)] = C2_SCALE * cx * cz;
        }
    }
}

/*
 * Print current t, step count, order, stepsize, and sampled c1,c2 values
 */

fn PrintOutput(cvode_mem: &CVodeMem, t: sunrealtype, y: &N_Vector) {
    let mut nst: i64 = 0;
    let mut qu: i32 = 0;
    let mut hu: sunrealtype = ZERO;

    let ydata = N_VGetArrayPointer(y).expect("N_VGetArrayPointer");

    let retval = CVodeGetNumSteps(cvode_mem, &mut nst);
    check_retval(&retval, "CVodeGetNumSteps");
    let retval = CVodeGetLastOrder(cvode_mem, &mut qu);
    check_retval(&retval, "CVodeGetLastOrder");
    let retval = CVodeGetLastStep(cvode_mem, &mut hu);
    check_retval(&retval, "CVodeGetLastStep");

    print!(
        "{} {:2}  {} {:5}\n",
        fmt_ew(t, 8, 3),
        qu,
        fmt_ew(hu, 8, 3),
        nst
    );

    print!("                                Solution       ");
    print!(
        "{} {} \n",
        fmt_ew(ydata[IJKth(1, 0, 0)], 12, 4),
        fmt_ew(ydata[IJKth(1, MX - 1, MZ - 1)], 12, 4)
    );
    print!("                                               ");
    print!(
        "{} {} \n",
        fmt_ew(ydata[IJKth(2, 0, 0)], 12, 4),
        fmt_ew(ydata[IJKth(2, MX - 1, MZ - 1)], 12, 4)
    );
}

/*
 * Print sampled sensitivities
 */

fn PrintOutputS(uS: &[N_Vector]) {
    let sdata = N_VGetArrayPointer(&uS[0]).expect("N_VGetArrayPointer");

    print!("                                ");
    print!("----------------------------------------\n");
    print!("                                Sensitivity 1  ");
    print!(
        "{} {} \n",
        fmt_ew(sdata[IJKth(1, 0, 0)], 12, 4),
        fmt_ew(sdata[IJKth(1, MX - 1, MZ - 1)], 12, 4)
    );
    print!("                                               ");
    print!(
        "{} {} \n",
        fmt_ew(sdata[IJKth(2, 0, 0)], 12, 4),
        fmt_ew(sdata[IJKth(2, MX - 1, MZ - 1)], 12, 4)
    );
    drop(sdata);

    let sdata = N_VGetArrayPointer(&uS[1]).expect("N_VGetArrayPointer");

    print!("                                ");
    print!("----------------------------------------\n");
    print!("                                Sensitivity 2  ");
    print!(
        "{} {} \n",
        fmt_ew(sdata[IJKth(1, 0, 0)], 12, 4),
        fmt_ew(sdata[IJKth(1, MX - 1, MZ - 1)], 12, 4)
    );
    print!("                                               ");
    print!(
        "{} {} \n",
        fmt_ew(sdata[IJKth(2, 0, 0)], 12, 4),
        fmt_ew(sdata[IJKth(2, MX - 1, MZ - 1)], 12, 4)
    );
}

/*
 * Print final statistics contained in iopt
 */

fn PrintFinalStats(
    cvode_mem: &CVodeMem,
    sensi: sunbooleantype,
    err_con: sunbooleantype,
    sensi_meth: i32,
) {
    let mut nst: i64 = 0;
    let (mut nfe, mut nsetups, mut nni, mut ncfn, mut netf): (i64, i64, i64, i64, i64) =
        (0, 0, 0, 0, 0);
    let (mut nfSe, mut nfeS, mut nsetupsS, mut nniS, mut ncfnS, mut netfS): (
        i64,
        i64,
        i64,
        i64,
        i64,
        i64,
    ) = (0, 0, 0, 0, 0, 0);
    let (mut nli, mut ncfl, mut npe, mut nps): (i64, i64, i64, i64) = (0, 0, 0, 0);

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

    if sensi {
        let retval = CVodeGetSensNumRhsEvals(cvode_mem, &mut nfSe);
        check_retval(&retval, "CVodeGetSensNumRhsEvals");
        let retval = CVodeGetNumRhsEvalsSens(cvode_mem, &mut nfeS);
        check_retval(&retval, "CVodeGetNumRhsEvalsSens");
        let retval = CVodeGetSensNumLinSolvSetups(cvode_mem, &mut nsetupsS);
        check_retval(&retval, "CVodeGetSensNumLinSolvSetups");
        if err_con {
            let retval = CVodeGetSensNumErrTestFails(cvode_mem, &mut netfS);
            check_retval(&retval, "CVodeGetSensNumErrTestFails");
        } else {
            netfS = 0;
        }
        if (sensi_meth == CV_STAGGERED) || (sensi_meth == CV_STAGGERED1) {
            let retval = CVodeGetSensNumNonlinSolvIters(cvode_mem, &mut nniS);
            check_retval(&retval, "CVodeGetSensNumNonlinSolvIters");
            let retval = CVodeGetSensNumNonlinSolvConvFails(cvode_mem, &mut ncfnS);
            check_retval(&retval, "CVodeGetSensNumNonlinSolvConvFails");
        } else {
            nniS = 0;
            ncfnS = 0;
        }
    }

    let retval = CVodeGetNumLinIters(cvode_mem, &mut nli);
    check_retval(&retval, "CVodeGetNumLinIters");
    let retval = CVodeGetNumLinConvFails(cvode_mem, &mut ncfl);
    check_retval(&retval, "CVodeGetNumLinConvFails");
    let retval = CVodeGetNumPrecEvals(cvode_mem, &mut npe);
    check_retval(&retval, "CVodeGetNumPrecEvals");
    let retval = CVodeGetNumPrecSolves(cvode_mem, &mut nps);
    check_retval(&retval, "CVodeGetNumPrecSolves");

    print!("\nFinal Statistics\n\n");
    print!("nst     = {:5}\n\n", nst);
    print!("nfe     = {:5}\n", nfe);
    print!("netf    = {:5}    nsetups  = {:5}\n", netf, nsetups);
    print!("nni     = {:5}    ncfn     = {:5}\n", nni, ncfn);

    if sensi {
        print!("\n");
        print!("nfSe    = {:5}    nfeS     = {:5}\n", nfSe, nfeS);
        print!("netfs   = {:5}    nsetupsS = {:5}\n", netfS, nsetupsS);
        print!("nniS    = {:5}    ncfnS    = {:5}\n", nniS, ncfnS);
    }

    print!("\n");
    print!("nli     = {:5}    ncfl     = {:5}\n", nli, ncfl);
    print!("npe     = {:5}    nps      = {:5}\n", npe, nps);
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
