#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

/* -----------------------------------------------------------------
 * Rust port of examples/cvodes/serial/cvsFoodWeb_ASAp_kry.c
 * Programmer(s): Radu Serban @ LLNL
 * -----------------------------------------------------------------
 * This program solves a stiff ODE system that arises from a system
 * of partial differential equations. The PDE system is a food web
 * population model, with predator-prey interaction and diffusion on
 * the unit square in two dimensions. The dependent variable vector
 * is the following:
 *
 *        1   2        ns
 *  c = (c , c , ..., c  )
 *
 * and the PDEs are as follows:
 *
 *    i               i      i
 *  dc /dt  =  d(i)*(c    + c   )  +  f (x,y,c)  (i=1,...,ns)
 *                    xx     yy        i
 *
 * where
 *
 *                 i          ns         j
 *  f (x,y,c)  =  c *(b(i) + sum a(i,j)*c )
 *   i                       j=1
 *
 * The number of species is ns = 2*np, with the first np being prey
 * and the last np being predators. The coefficients a(i,j), b(i),
 * d(i) are:
 *
 *  a(i,i) = -a  (all i)
 *  a(i,j) = -g  (i <= np, j > np)
 *  a(i,j) =  e  (i > np, j <= np)
 *  b(i) =  b*(1 + alpha*x*y)  (i <= np)
 *  b(i) = -b*(1 + alpha*x*y)  (i > np)
 *  d(i) = Dprey  (i <= np)
 *  d(i) = Dpred  (i > np)
 *
 * The spatial domain is the unit square. The final time is 10.
 * The boundary conditions are: normal derivative = 0.
 * A polynomial in x and y is used to set the initial conditions.
 *
 * The PDEs are discretized by central differencing on an MX by MY
 * mesh. The resulting ODE system is stiff.
 *
 * The ODE system is solved by CVODES using Newton iteration and
 * the SUNLinSol_SPGMR linear solver (scaled preconditioned GMRES).
 *
 * The preconditioner matrix used is the product of two matrices:
 * (1) A matrix, only defined implicitly, based on a fixed number
 * of Gauss-Seidel iterations using the diffusion terms only.
 * (2) A block-diagonal matrix based on the partial derivatives of
 * the interaction terms f only, using block-grouping (computing
 * only a subset of the ns by ns blocks).
 *
 * Additionally, CVODES integrates backwards in time the
 * the semi-discrete form of the adjoint PDE:
 *   d(lambda)/dt = - D^T ( lambda_xx + lambda_yy )
 *                  - F_c^T lambda
 * with homogeneous Neumann boundary conditions and final conditions
 *   lambda(x,y,t=t_final) = - g_c^T(t_final)
 * whose solution at t = 0 represents the sensitivity of
 *   int_x int _y g(t_final,c) dx dy dt
 * with respect to the initial conditions of the original problem.
 *
 * In this example,
 *   g(t,c) = c(ISPEC), with ISPEC defined below.
 * -----------------------------------------------------------------
 * Reference:  Peter N. Brown and Alan C. Hindmarsh, Reduced Storage
 * Matrix Methods in Stiff ODE Systems, J. Appl. Math. & Comp., 31
 * (1989), pp. 40-91.  Also available as Lawrence Livermore National
 * Laboratory Report UCRL-95088, Rev. 1, June 1987.
 * -----------------------------------------------------------------*/

use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

use cvodes_rs::prelude::*;
use cvodes_rs::sundials_dense::{
    SUNDlsMat_denseAddIdentity, SUNDlsMat_denseGETRF, SUNDlsMat_denseGETRS,
};

/* helpful macros */

fn MAX(a: sunrealtype, b: sunrealtype) -> sunrealtype {
    if a > b {
        a
    } else {
        b
    }
}

fn SQR(a: sunrealtype) -> sunrealtype {
    a * a
}

/* Constants */

const ZERO: sunrealtype = 0.0;
const ONE: sunrealtype = 1.0;
const TWO: sunrealtype = 2.0;

/* Problem Specification Constants */

const AA: sunrealtype = ONE; /* AA = a */
const EE: sunrealtype = 1.0e4; /* EE = e */
const GG: sunrealtype = 0.5e-6; /* GG = g */
const BB: sunrealtype = ONE; /* BB = b */
const DPREY: sunrealtype = ONE;
const DPRED: sunrealtype = 0.5;
const ALPHA: sunrealtype = ONE;
const NP: i32 = 3;
const NS: i32 = 2 * NP;

/* Method Constants */

const MX: i32 = 20;
const MY: i32 = 20;
const MXNS: i32 = MX * NS;
const AX: sunrealtype = ONE;
const AY: sunrealtype = ONE;
const DX: sunrealtype = AX / (MX - 1) as sunrealtype;
const DY: sunrealtype = AY / (MY - 1) as sunrealtype;
const MP: i32 = NS;
const MQ: i32 = MX * MY;
const MXMP: i32 = MX * MP;
const NGX: i32 = 2;
const NGY: i32 = 2;
const NGRP: i32 = NGX * NGY;
const ITMAX: i32 = 5;

/* CVodeInit Constants */

const NEQ: i32 = NS * MX * MY;
const T0: sunrealtype = ZERO;
const RTOL: sunrealtype = 1.0e-5;
const ATOL: sunrealtype = 1.0e-5;

/* Output Constants */

const TOUT: sunrealtype = 10.0;

/* Note: The value for species i at mesh point (j,k) is stored in */
/* component number (i-1) + j*NS + k*NS*MX of an N_Vector,        */
/* where 1 <= i <= NS, 0 <= j < MX, 0 <= k < MY.                  */

/* Structure for user data.
 *
 * C passes one and the same `WebData` pointer to both the forward
 * integrator (CVodeSetUserData) and the backward one
 * (CVodeSetUserDataB); the port shares the single record through
 * `Rc<RefCell<WebData>>` so both `Option<Box<dyn Any>>` tokens reach
 * the same fields (fsave written by f/fB and read by Precond/PrecondB
 * exactly as in C). */

/* Some fields (mq, jgx, jgy) are kept solely for fidelity with the C
 * struct layout and are never read after initialization. */
#[allow(dead_code)]
struct WebData {
    P: Vec<Vec<Vec<sunrealtype>>>, /* NGRP dense blocks, column-major (P[ig][j] = column j) */
    pivot: Vec<Vec<sunindextype>>, /* NGRP pivot arrays */
    ns: i32,
    mxns: i32,
    mp: i32,
    mq: i32,
    mx: i32,
    my: i32,
    ngrp: i32,
    ngx: i32,
    ngy: i32,
    mxmp: i32,
    jgx: [i32; (NGX + 1) as usize],
    jgy: [i32; (NGY + 1) as usize],
    jigx: [i32; MX as usize],
    jigy: [i32; MY as usize],
    jxr: [i32; NGX as usize],
    jyr: [i32; NGY as usize],
    acoef: [[sunrealtype; NS as usize]; NS as usize],
    bcoef: [sunrealtype; NS as usize],
    diff: [sunrealtype; NS as usize],
    cox: [sunrealtype; NS as usize],
    coy: [sunrealtype; NS as usize],
    dx: sunrealtype,
    dy: sunrealtype,
    srur: sunrealtype,
    fsave: Vec<sunrealtype>,  /* C: sunrealtype fsave[NEQ]  */
    fBsave: Vec<sunrealtype>, /* C: sunrealtype fBsave[NEQ] */
    rewt: N_Vector,
    vtemp: N_Vector,
    cvode_mem: Option<CVodeMem>,
    indexB: i32,
}

type WebDataRc = Rc<RefCell<WebData>>;

fn wdata_of(user_data: &mut Option<Box<dyn Any>>) -> WebDataRc {
    user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<WebDataRc>())
        .expect("user_data is WebData")
        .clone()
}

/* Adjoint calculation constants */
/* g = int_x int_y c(ISPEC) dy dx at t = Tfinal */

const NSTEPS: i64 = 80; /* check points every NSTEPS steps */
const ISPEC: i32 = 6; /* species # in objective */

/*
 *--------------------------------------------------------------------
 * MAIN PROGRAM
 *--------------------------------------------------------------------
 */

fn main() {
    let abstol: sunrealtype = ATOL;
    let reltol: sunrealtype = RTOL;
    let mut t: sunrealtype = ZERO;

    let mut ncheck: i32 = 0;

    let mut indexB: i32 = 0;

    let reltolB: sunrealtype = RTOL;
    let abstolB: sunrealtype = ATOL;

    /* Create the SUNDIALS simulation context that all SUNDIALS objects require */
    let mut sunctx_opt: Option<SUNContext> = None;
    let retval = SUNContext_Create(SUN_COMM_NULL, &mut sunctx_opt);
    if check_retval(retval, "SUNContext_Create", 1) != 0 {
        std::process::exit(1);
    }
    let sunctx = sunctx_opt.as_ref().expect("SUNContext").clone();

    /* Allocate and initialize user data */

    let wdata = AllocUserData(&sunctx);
    if check_retval_ptr(&wdata, "AllocUserData", 2) != 0 {
        std::process::exit(1);
    }
    let mut wdata = wdata.expect("AllocUserData");
    InitUserData(&mut wdata);
    let wdata: WebDataRc = Rc::new(RefCell::new(wdata));

    /* Set-up forward problem */

    /* Initializations */
    let c = N_VNew_Serial(NEQ as sunindextype, &sunctx);
    if check_retval_ptr(&c, "N_VNew_Serial", 0) != 0 {
        std::process::exit(1);
    }
    let c = c.expect("N_VNew_Serial");
    CInit(&c, &wdata.borrow());

    /* Call CVodeCreate/CVodeInit for forward run */
    print!("\nCreate and allocate CVODES memory for forward run\n");
    let cvode_mem = CVodeCreate(CV_BDF, &sunctx);
    if check_retval_ptr(&cvode_mem, "CVodeCreate", 0) != 0 {
        std::process::exit(1);
    }
    let cvode_mem = cvode_mem.expect("CVodeCreate");
    wdata.borrow_mut().cvode_mem = Some(cvode_mem.clone()); /* Used in Precond */
    let retval = CVodeSetUserData(&cvode_mem, Some(Box::new(wdata.clone())));
    if check_retval(retval, "CVodeSetUserData", 1) != 0 {
        std::process::exit(1);
    }
    let retval = CVodeInit(&cvode_mem, f, T0, &c);
    if check_retval(retval, "CVodeInit", 1) != 0 {
        std::process::exit(1);
    }
    let retval = CVodeSStolerances(&cvode_mem, reltol, abstol);
    if check_retval(retval, "CVodeSStolerances", 1) != 0 {
        std::process::exit(1);
    }

    /* Create SUNLinSol_SPGMR linear solver for forward run */
    let LS = SUNLinSol_SPGMR(&c, SUN_PREC_LEFT, 0, &sunctx);
    if check_retval_ptr(&LS, "SUNLinSol_SPGMR", 0) != 0 {
        std::process::exit(1);
    }
    let LS = LS.expect("SUNLinSol_SPGMR");

    /* Attach the linear solver */
    let retval = CVodeSetLinearSolver(&cvode_mem, &LS, None);
    if check_retval(retval, "CVodeSetLinearSolver", 1) != 0 {
        std::process::exit(1);
    }

    /* Set the preconditioner solve and setup functions */
    let retval = CVodeSetPreconditioner(&cvode_mem, Some(Precond), Some(PSolve));
    if check_retval(retval, "CVodeSetPreconditioner", 1) != 0 {
        std::process::exit(1);
    }

    /* Call CVodeSetMaxNumSteps to set the maximum number of steps the
     * solver will take in an attempt to reach the next output time
     * during forward integration. */
    let retval = CVodeSetMaxNumSteps(&cvode_mem, 2500);
    if check_retval(retval, "CVodeSetMaxNumSteps", 1) != 0 {
        std::process::exit(1);
    }

    /* Set-up adjoint calculations */

    print!("\nAllocate global memory\n");
    let retval = CVodeAdjInit(&cvode_mem, NSTEPS, CV_HERMITE);
    if check_retval(retval, "CVodeAdjInit", 1) != 0 {
        std::process::exit(1);
    }

    /* Perform forward run */

    print!("\nForward integration\n");
    let retval = CVodeF(&cvode_mem, TOUT, &c, &mut t, CV_NORMAL, &mut ncheck);
    if check_retval(retval, "CVodeF", 1) != 0 {
        std::process::exit(1);
    }

    print!("\nncheck = {}\n", ncheck);

    {
        let cdata = N_VGetArrayPointer(&c).expect("N_VGetArrayPointer");
        let g = doubleIntgr(&cdata, ISPEC, &wdata.borrow());
        print!(
            "\n   g = int_x int_y c{}(Tfinal,x,y) dx dy = {} \n\n",
            ISPEC,
            fmt_f(g, 6)
        );
    }

    /* Set-up backward problem */

    /* Allocate cB */
    let cB = N_VNew_Serial(NEQ as sunindextype, &sunctx);
    if check_retval_ptr(&cB, "N_VNew_Serial", 0) != 0 {
        std::process::exit(1);
    }
    let cB = cB.expect("N_VNew_Serial");
    /* Initialize cB = 0 */
    CbInit(&cB, ISPEC, &wdata.borrow());

    /* Create and allocate CVODES memory for backward run */
    print!("\nCreate and allocate CVODES memory for backward run\n");
    let retval = CVodeCreateB(&cvode_mem, CV_BDF, &mut indexB);
    if check_retval(retval, "CVodeCreateB", 1) != 0 {
        std::process::exit(1);
    }
    let retval = CVodeSetUserDataB(&cvode_mem, indexB, Some(Box::new(wdata.clone())));
    if check_retval(retval, "CVodeSetUserDataB", 1) != 0 {
        std::process::exit(1);
    }
    let retval = CVodeInitB(&cvode_mem, indexB, fB, TOUT, &cB);
    if check_retval(retval, "CVodeInitB", 1) != 0 {
        std::process::exit(1);
    }
    let retval = CVodeSStolerancesB(&cvode_mem, indexB, reltolB, abstolB);
    if check_retval(retval, "CVodeSStolerancesB", 1) != 0 {
        std::process::exit(1);
    }

    wdata.borrow_mut().indexB = indexB;

    /* Create SUNLinSol_SPGMR linear solver for backward run */
    let LSB = SUNLinSol_SPGMR(&cB, SUN_PREC_LEFT, 0, &sunctx);
    if check_retval_ptr(&LSB, "SUNLinSol_SPGMR", 0) != 0 {
        std::process::exit(1);
    }
    let LSB = LSB.expect("SUNLinSol_SPGMR");

    /* Attach the linear solver */
    let retval = CVodeSetLinearSolverB(&cvode_mem, indexB, &LSB, None);
    if check_retval(retval, "CVodeSetLinearSolverB", 1) != 0 {
        std::process::exit(1);
    }

    /* Set the preconditioner solve and setup functions */
    let retval = CVodeSetPreconditionerB(&cvode_mem, indexB, Some(PrecondB), Some(PSolveB));
    if check_retval(retval, "CVodeSetPreconditionerB", 1) != 0 {
        std::process::exit(1);
    }

    /* Perform backward integration */

    print!("\nBackward integration\n");
    let retval = CVodeB(&cvode_mem, T0, CV_NORMAL);
    if check_retval(retval, "CVodeB", 1) != 0 {
        std::process::exit(1);
    }

    let retval = CVodeGetB(&cvode_mem, indexB, &mut t, &cB);
    if check_retval(retval, "CVodeGetB", 1) != 0 {
        std::process::exit(1);
    }

    PrintOutput(&cB, NS, MXNS, &wdata.borrow());

    /* Free all memory (FreeUserData: the shared WebData record is
     * released when the last Rc handle — the integrator's user_data
     * boxes and this one — is dropped) */
    let mut cvode_mem = Some(cvode_mem);
    CVodeFree(&mut cvode_mem);

    N_VDestroy(c);
    N_VDestroy(cB);
    SUNLinSolFree(Some(LS));
    SUNLinSolFree(Some(LSB));
    drop(sunctx);
    SUNContext_Free(&mut sunctx_opt);
}

/*
 *--------------------------------------------------------------------
 * FUNCTIONS CALLED BY CVODES
 *--------------------------------------------------------------------
 */

/*
 * This routine computes the right-hand side of the ODE system and
 * returns it in cdot. The interaction rates are computed by calls to WebRates,
 * and these are saved in fsave for use in preconditioning.
 */

fn f(t: sunrealtype, c: &N_Vector, cdot: &N_Vector, user_data: &mut Option<Box<dyn Any>>) -> i32 {
    let wdata_rc = wdata_of(user_data);
    let wdata = &mut *wdata_rc.borrow_mut();

    let cdata = N_VGetArrayPointer(c).expect("N_VGetArrayPointer");
    let mut cdotdata = N_VGetArrayPointer(cdot).expect("N_VGetArrayPointer");

    let mxns = wdata.mxns;
    let ns = wdata.ns;
    let cox = wdata.cox;
    let coy = wdata.coy;
    let dx = wdata.dx;
    let dy = wdata.dy;
    let acoef = wdata.acoef;
    let bcoef = wdata.bcoef;
    let fsave = &mut wdata.fsave;

    for jy in 0..MY {
        let y = jy as sunrealtype * dy;
        let iyoff = mxns * jy;
        let idyu = if jy == MY - 1 { -mxns } else { mxns };
        let idyl = if jy == 0 { -mxns } else { mxns };
        for jx in 0..MX {
            let x = jx as sunrealtype * dx;
            let ic = iyoff + ns * jx;
            /* Get interaction rates at one point (x,y). */
            WebRates(
                x,
                y,
                t,
                &cdata[ic as usize..],
                &mut fsave[ic as usize..],
                ns,
                &acoef,
                &bcoef,
            );
            let idxu = if jx == MX - 1 { -ns } else { ns };
            let idxl = if jx == 0 { -ns } else { ns };
            for i in 1..=ns {
                let ici = ic + i - 1;
                /* Do differencing in y. */
                let dcyli = cdata[ici as usize] - cdata[(ici - idyl) as usize];
                let dcyui = cdata[(ici + idyu) as usize] - cdata[ici as usize];
                /* Do differencing in x. */
                let dcxli = cdata[ici as usize] - cdata[(ici - idxl) as usize];
                let dcxui = cdata[(ici + idxu) as usize] - cdata[ici as usize];
                /* Collect terms and load cdot elements. */
                cdotdata[ici as usize] = coy[(i - 1) as usize] * (dcyui - dcyli)
                    + cox[(i - 1) as usize] * (dcxui - dcxli)
                    + fsave[ici as usize];
            }
        }
    }

    0
}

/*
 * This routine generates the block-diagonal part of the Jacobian
 * corresponding to the interaction rates, multiplies by -gamma,
 * adds the identity matrix, and calls SUNDlsMat_denseGETRF to do
 * the LU decomposition of each diagonal block. The computation of
 * the diagonal blocks uses the preset block and grouping
 * information. One block per group is computed. The Jacobian
 * elements are generated by difference quotients using calls to the
 * routine fblock.
 *
 * This routine can be regarded as a prototype for the general case
 * of a block-diagonal preconditioner. The blocks are of size mp,
 * and there are ngrp=ngx*ngy blocks computed in the block-grouping
 * scheme.
 */

fn Precond(
    t: sunrealtype,
    c: &N_Vector,
    fc: &N_Vector,
    _jok: sunbooleantype,
    jcurPtr: &mut sunbooleantype,
    gamma: sunrealtype,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    let wdata_rc = wdata_of(user_data);
    let wdata = &mut *wdata_rc.borrow_mut();

    let cvode_mem = wdata.cvode_mem.as_ref().expect("cvode_mem").clone();
    let rewt = wdata.rewt.clone();
    let retval = CVodeGetErrWeights(&cvode_mem, &rewt);
    if check_retval(retval, "CVodeGetErrWeights", 1) != 0 {
        return 1;
    }

    let uround = SUN_UNIT_ROUNDOFF;

    let mp = wdata.mp;
    let srur = wdata.srur;
    let ngrp = wdata.ngrp;
    let ngx = wdata.ngx;
    let ngy = wdata.ngy;
    let mxmp = wdata.mxmp;

    /* Make mp calls to fblock to approximate each diagonal block of Jacobian.
    Here, fsave contains the base value of the rate vector and
    r0 is a minimum increment factor for the difference quotient. */

    let vtemp = wdata.vtemp.clone();

    let mut fac = N_VWrmsNorm(fc, &rewt);
    let mut r0 = 1000.0 * SUNRabs(gamma) * uround * NEQ as sunrealtype * fac;
    if r0 == ZERO {
        r0 = ONE;
    }

    {
        let mut cdata = N_VGetArrayPointer(c).expect("N_VGetArrayPointer");
        let rewtdata = N_VGetArrayPointer(&rewt).expect("N_VGetArrayPointer");
        let mut f1 = N_VGetArrayPointer(&vtemp).expect("N_VGetArrayPointer");

        for igy in 0..ngy {
            let jy = wdata.jyr[igy as usize];
            let if00 = jy * mxmp;
            for igx in 0..ngx {
                let jx = wdata.jxr[igx as usize];
                let if0 = if00 + jx * mp;
                let ig = (igx + igy * ngx) as usize;
                /* Generate ig-th diagonal block */
                for j in 0..mp {
                    /* Generate the jth column as a difference quotient */
                    let jj = (if0 + j) as usize;
                    let save = cdata[jj];
                    let r = MAX(srur * SUNRabs(save), r0 / rewtdata[jj]);
                    cdata[jj] += r;
                    fac = -gamma / r;
                    fblock(t, &cdata[..], jx, jy, &mut f1[..], wdata);
                    for i in 0..mp as usize {
                        wdata.P[ig][j as usize][i] = (f1[i] - wdata.fsave[if0 as usize + i]) * fac;
                    }
                    cdata[jj] = save;
                }
            }
        }
    }

    /* Add identity matrix and do LU decompositions on blocks. */

    for ig in 0..ngrp as usize {
        let mut cols: Vec<&mut [sunrealtype]> = wdata.P[ig]
            .iter_mut()
            .map(|col| col.as_mut_slice())
            .collect();
        SUNDlsMat_denseAddIdentity(&mut cols, mp as sunindextype);
        let denseretval = SUNDlsMat_denseGETRF(
            &mut cols,
            mp as sunindextype,
            mp as sunindextype,
            &mut wdata.pivot[ig],
        );
        if denseretval != 0 {
            return 1;
        }
    }

    *jcurPtr = SUNTRUE;
    0
}

/*
 * This routine applies two inverse preconditioner matrices
 * to the vector r, using the interaction-only block-diagonal Jacobian
 * with block-grouping, denoted Jr, and Gauss-Seidel applied to the
 * diffusion contribution to the Jacobian, denoted Jd.
 * It first calls GSIter for a Gauss-Seidel approximation to
 * ((I - gamma*Jd)-inverse)*r, and stores the result in z.
 * Then it computes ((I - gamma*Jr)-inverse)*z, using LU factors of the
 * blocks in P, and pivot information in pivot, and returns the result in z.
 */

fn PSolve(
    _t: sunrealtype,
    _c: &N_Vector,
    _fc: &N_Vector,
    r: &N_Vector,
    z: &N_Vector,
    gamma: sunrealtype,
    _delta: sunrealtype,
    _lr: i32,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    let wdata_rc = wdata_of(user_data);
    let wdata = &mut *wdata_rc.borrow_mut();

    N_VScale(ONE, r, z);

    /* call GSIter for Gauss-Seidel iterations */

    let vtemp = wdata.vtemp.clone();
    GSIter(gamma, z, &vtemp, wdata);

    /* Do backsolves for inverse of block-diagonal preconditioner factor */

    let mx = wdata.mx;
    let my = wdata.my;
    let ngx = wdata.ngx;
    let mp = wdata.mp;

    let mut zd = N_VGetArrayPointer(z).expect("N_VGetArrayPointer");

    let mut iv: usize = 0;
    for jy in 0..my {
        let igy = wdata.jigy[jy as usize];
        for jx in 0..mx {
            let igx = wdata.jigx[jx as usize];
            let ig = (igx + igy * ngx) as usize;
            let mut cols: Vec<&mut [sunrealtype]> = wdata.P[ig]
                .iter_mut()
                .map(|col| col.as_mut_slice())
                .collect();
            SUNDlsMat_denseGETRS(
                &mut cols,
                mp as sunindextype,
                &wdata.pivot[ig],
                &mut zd[iv..iv + mp as usize],
            );
            iv += mp as usize;
        }
    }

    0
}

/*
 * This routine computes the right-hand side of the adjoint ODE system and
 * returns it in cBdot. The interaction rates are computed by calls to WebRates,
 * and these are saved in fsave for use in preconditioning. The adjoint
 * interaction rates are computed by calls to WebRatesB.
 */

fn fB(
    t: sunrealtype,
    c: &N_Vector,
    cB: &N_Vector,
    cBdot: &N_Vector,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    let wdata_rc = wdata_of(user_data);
    let wdata = &mut *wdata_rc.borrow_mut();

    let cdata = N_VGetArrayPointer(c).expect("N_VGetArrayPointer");
    let cBdata = N_VGetArrayPointer(cB).expect("N_VGetArrayPointer");
    let mut cBdotdata = N_VGetArrayPointer(cBdot).expect("N_VGetArrayPointer");

    let mxns = wdata.mxns;
    let ns = wdata.ns;
    let cox = wdata.cox;
    let coy = wdata.coy;
    let dx = wdata.dx;
    let dy = wdata.dy;
    let acoef = wdata.acoef;
    let bcoef = wdata.bcoef;
    let (fsave, fBsave) = (&mut wdata.fsave, &mut wdata.fBsave);

    for jy in 0..MY {
        let y = jy as sunrealtype * dy;
        let iyoff = mxns * jy;
        let idyu = if jy == MY - 1 { -mxns } else { mxns };
        let idyl = if jy == 0 { -mxns } else { mxns };
        for jx in 0..MX {
            let x = jx as sunrealtype * dx;
            let ic = iyoff + ns * jx;
            /* Get interaction rates at one point (x,y). */
            WebRatesB(
                x,
                y,
                t,
                &cdata[ic as usize..],
                &cBdata[ic as usize..],
                &mut fsave[ic as usize..],
                &mut fBsave[ic as usize..],
                ns,
                &acoef,
                &bcoef,
            );
            let idxu = if jx == MX - 1 { -ns } else { ns };
            let idxl = if jx == 0 { -ns } else { ns };
            for i in 1..=ns {
                let ici = ic + i - 1;
                /* Do differencing in y. */
                let dcyli = cBdata[ici as usize] - cBdata[(ici - idyl) as usize];
                let dcyui = cBdata[(ici + idyu) as usize] - cBdata[ici as usize];
                /* Do differencing in x. */
                let dcxli = cBdata[ici as usize] - cBdata[(ici - idxl) as usize];
                let dcxui = cBdata[(ici + idxu) as usize] - cBdata[ici as usize];
                /* Collect terms and load cdot elements. */
                cBdotdata[ici as usize] = -coy[(i - 1) as usize] * (dcyui - dcyli)
                    - cox[(i - 1) as usize] * (dcxui - dcxli)
                    - fBsave[ici as usize];
            }
        }
    }

    0
}

/*
 * Preconditioner setup function for the backward problem
 */

fn PrecondB(
    t: sunrealtype,
    c: &N_Vector,
    _cB: &N_Vector,
    fcB: &N_Vector,
    _jok: sunbooleantype,
    jcurPtr: &mut sunbooleantype,
    gamma: sunrealtype,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    let wdata_rc = wdata_of(user_data);
    let wdata = &mut *wdata_rc.borrow_mut();

    let fwd_mem = wdata.cvode_mem.as_ref().expect("cvode_mem").clone();
    let cvode_mem = CVodeGetAdjCVodeBmem(&fwd_mem, wdata.indexB);
    if check_retval_ptr(&cvode_mem, "CVadjGetCVodeBmem", 0) != 0 {
        return 1;
    }
    let cvode_mem = cvode_mem.expect("CVodeGetAdjCVodeBmem");
    let rewt = wdata.rewt.clone();
    let retval = CVodeGetErrWeights(&cvode_mem, &rewt);
    if check_retval(retval, "CVodeGetErrWeights", 1) != 0 {
        return 1;
    }

    let uround = SUN_UNIT_ROUNDOFF;

    let mp = wdata.mp;
    let srur = wdata.srur;
    let ngrp = wdata.ngrp;
    let ngx = wdata.ngx;
    let ngy = wdata.ngy;
    let mxmp = wdata.mxmp;

    /* Make mp calls to fblock to approximate each diagonal block of Jacobian.
    Here, fsave contains the base value of the rate vector and
    r0 is a minimum increment factor for the difference quotient. */

    let vtemp = wdata.vtemp.clone();

    let mut fac = N_VWrmsNorm(fcB, &rewt);
    let mut r0 = 1000.0 * SUNRabs(gamma) * uround * NEQ as sunrealtype * fac;
    if r0 == ZERO {
        r0 = ONE;
    }

    {
        let mut cdata = N_VGetArrayPointer(c).expect("N_VGetArrayPointer");
        let rewtdata = N_VGetArrayPointer(&rewt).expect("N_VGetArrayPointer");
        let mut f1 = N_VGetArrayPointer(&vtemp).expect("N_VGetArrayPointer");

        for igy in 0..ngy {
            let jy = wdata.jyr[igy as usize];
            let if00 = jy * mxmp;
            for igx in 0..ngx {
                let jx = wdata.jxr[igx as usize];
                let if0 = if00 + jx * mp;
                let ig = (igx + igy * ngx) as usize;
                /* Generate ig-th diagonal block */
                for j in 0..mp {
                    /* Generate the jth column as a difference quotient */
                    let jj = (if0 + j) as usize;
                    let save = cdata[jj];
                    let r = MAX(srur * SUNRabs(save), r0 / rewtdata[jj]);
                    cdata[jj] += r;
                    fac = gamma / r;
                    fblock(t, &cdata[..], jx, jy, &mut f1[..], wdata);
                    for i in 0..mp as usize {
                        wdata.P[ig][i][j as usize] = (f1[i] - wdata.fsave[if0 as usize + i]) * fac;
                    }
                    cdata[jj] = save;
                }
            }
        }
    }

    /* Add identity matrix and do LU decompositions on blocks. */

    for ig in 0..ngrp as usize {
        let mut cols: Vec<&mut [sunrealtype]> = wdata.P[ig]
            .iter_mut()
            .map(|col| col.as_mut_slice())
            .collect();
        SUNDlsMat_denseAddIdentity(&mut cols, mp as sunindextype);
        let denseretval = SUNDlsMat_denseGETRF(
            &mut cols,
            mp as sunindextype,
            mp as sunindextype,
            &mut wdata.pivot[ig],
        );
        if denseretval != 0 {
            return 1;
        }
    }

    *jcurPtr = SUNTRUE;
    0
}

/*
 * Preconditioner solve function for the backward problem
 */

#[allow(clippy::too_many_arguments)]
fn PSolveB(
    _t: sunrealtype,
    _c: &N_Vector,
    _cB: &N_Vector,
    _fcB: &N_Vector,
    r: &N_Vector,
    z: &N_Vector,
    gamma: sunrealtype,
    _delta: sunrealtype,
    _lr: i32,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    let wdata_rc = wdata_of(user_data);
    let wdata = &mut *wdata_rc.borrow_mut();

    N_VScale(ONE, r, z);

    /* call GSIter for Gauss-Seidel iterations (same routine but with gamma=-gamma) */

    let vtemp = wdata.vtemp.clone();
    GSIter(-gamma, z, &vtemp, wdata);

    /* Do backsolves for inverse of block-diagonal preconditioner factor */

    let mx = wdata.mx;
    let my = wdata.my;
    let ngx = wdata.ngx;
    let mp = wdata.mp;

    let mut zd = N_VGetArrayPointer(z).expect("N_VGetArrayPointer");

    let mut iv: usize = 0;
    for jy in 0..my {
        let igy = wdata.jigy[jy as usize];
        for jx in 0..mx {
            let igx = wdata.jigx[jx as usize];
            let ig = (igx + igy * ngx) as usize;
            let mut cols: Vec<&mut [sunrealtype]> = wdata.P[ig]
                .iter_mut()
                .map(|col| col.as_mut_slice())
                .collect();
            SUNDlsMat_denseGETRS(
                &mut cols,
                mp as sunindextype,
                &wdata.pivot[ig],
                &mut zd[iv..iv + mp as usize],
            );
            iv += mp as usize;
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
 * Allocate space for user data structure
 */

fn AllocUserData(sunctx: &SUNContext) -> Option<WebData> {
    let ns = NS as usize;
    let ngrp = NGRP as usize;

    let mut P: Vec<Vec<Vec<sunrealtype>>> = Vec::with_capacity(ngrp);
    let mut pivot: Vec<Vec<sunindextype>> = Vec::with_capacity(ngrp);
    for _i in 0..ngrp {
        P.push(vec![vec![ZERO; ns]; ns]); /* SUNDlsMat_newDenseMat(ns, ns) */
        pivot.push(vec![0; ns]); /* SUNDlsMat_newIndexArray(ns) */
    }
    let rewt = N_VNew_Serial(NEQ as sunindextype, sunctx)?;
    let vtemp = N_VNew_Serial(NEQ as sunindextype, sunctx)?;

    Some(WebData {
        P,
        pivot,
        ns: 0,
        mxns: 0,
        mp: 0,
        mq: 0,
        mx: 0,
        my: 0,
        ngrp: 0,
        ngx: 0,
        ngy: 0,
        mxmp: 0,
        jgx: [0; (NGX + 1) as usize],
        jgy: [0; (NGY + 1) as usize],
        jigx: [0; MX as usize],
        jigy: [0; MY as usize],
        jxr: [0; NGX as usize],
        jyr: [0; NGY as usize],
        acoef: [[ZERO; NS as usize]; NS as usize],
        bcoef: [ZERO; NS as usize],
        diff: [ZERO; NS as usize],
        cox: [ZERO; NS as usize],
        coy: [ZERO; NS as usize],
        dx: ZERO,
        dy: ZERO,
        srur: ZERO,
        fsave: vec![ZERO; NEQ as usize],
        fBsave: vec![ZERO; NEQ as usize],
        rewt,
        vtemp,
        cvode_mem: None,
        indexB: 0,
    })
}

/*
 * Initialize user data structure
 */

fn InitUserData(wdata: &mut WebData) {
    let np = NP as usize;

    wdata.ns = NS;
    let ns = NS as usize;

    for j in 0..ns {
        for i in 0..ns {
            wdata.acoef[i][j] = ZERO;
        }
    }
    for j in 0..np {
        for i in 0..np {
            wdata.acoef[np + i][j] = EE;
            wdata.acoef[i][np + j] = -GG;
        }
        wdata.acoef[j][j] = -AA;
        wdata.acoef[np + j][np + j] = -AA;
        wdata.bcoef[j] = BB;
        wdata.bcoef[np + j] = -BB;
        wdata.diff[j] = DPREY;
        wdata.diff[np + j] = DPRED;
    }

    /* Set remaining problem parameters */

    wdata.mxns = MXNS;
    wdata.dx = DX;
    wdata.dy = DY;
    let dx = wdata.dx;
    let dy = wdata.dy;
    for i in 0..ns {
        wdata.cox[i] = wdata.diff[i] / SQR(dx);
        wdata.coy[i] = wdata.diff[i] / SQR(dy);
    }

    /* Set remaining method parameters */

    wdata.mp = MP;
    wdata.mq = MQ;
    wdata.mx = MX;
    wdata.my = MY;
    wdata.srur = SUNRsqrt(SUN_UNIT_ROUNDOFF);
    wdata.mxmp = MXMP;
    wdata.ngrp = NGRP;
    wdata.ngx = NGX;
    wdata.ngy = NGY;
    SetGroups(MX, NGX, &mut wdata.jgx, &mut wdata.jigx, &mut wdata.jxr);
    SetGroups(MY, NGY, &mut wdata.jgy, &mut wdata.jigy, &mut wdata.jyr);
}

/*
 * This routine sets arrays jg, jig, and jr describing
 * a uniform partition of (0,1,2,...,m-1) into ng groups.
 * The arrays set are:
 *   jg    = length ng+1 array of group boundaries.
 *           Group ig has indices j = jg[ig],...,jg[ig+1]-1.
 *   jig   = length m array of group indices vs node index.
 *           Node index j is in group jig[j].
 *   jr    = length ng array of indices representing the groups.
 *           The index for group ig is j = jr[ig].
 */

fn SetGroups(m: i32, ng: i32, jg: &mut [i32], jig: &mut [i32], jr: &mut [i32]) {
    let mper = m / ng; /* does integer division */
    for ig in 0..ng {
        jg[ig as usize] = ig * mper;
    }
    jg[ng as usize] = m;

    let ngm1 = ng - 1;
    let len1 = ngm1 * mper;
    for j in 0..len1 {
        jig[j as usize] = j / mper;
    }
    for j in len1..m {
        jig[j as usize] = ngm1;
    }

    for ig in 0..ngm1 {
        jr[ig as usize] = ((2 * ig + 1) * mper - 1) / 2;
    }
    jr[ngm1 as usize] = (ngm1 * mper + m - 1) / 2;
}

/*
 * This routine computes and loads the vector of initial values.
 */

fn CInit(c: &N_Vector, wdata: &WebData) {
    let mut cdata = N_VGetArrayPointer(c).expect("N_VGetArrayPointer");
    let ns = wdata.ns;
    let mxns = wdata.mxns;
    let dx = wdata.dx;
    let dy = wdata.dy;

    let x_factor = 4.0 / SQR(AX);
    let y_factor = 4.0 / SQR(AY);
    for jy in 0..MY {
        let y = jy as sunrealtype * dy;
        let argy = SQR(y_factor * y * (AY - y));
        let iyoff = mxns * jy;
        for jx in 0..MX {
            let x = jx as sunrealtype * dx;
            let argx = SQR(x_factor * x * (AX - x));
            let ioff = iyoff + ns * jx;
            for i in 1..=ns {
                let ici = (ioff + i - 1) as usize;
                cdata[ici] = 10.0 + i as sunrealtype * argx * argy;
            }
        }
    }
}

/*
 * This function computes and loads the final values for the adjoint variables
 */

fn CbInit(c: &N_Vector, _is: i32, wdata: &WebData) {
    let mut cdata = N_VGetArrayPointer(c).expect("N_VGetArrayPointer");
    let ns = wdata.ns;
    let mxns = wdata.mxns;

    let mut gu = [ZERO; NS as usize];

    for i in 1..=ns {
        gu[(i - 1) as usize] = ZERO;
    }
    gu[(ISPEC - 1) as usize] = ONE;

    for jy in 0..MY {
        let iyoff = mxns * jy;
        for jx in 0..MX {
            let ioff = iyoff + ns * jx;
            for i in 1..=ns {
                let ici = (ioff + i - 1) as usize;
                cdata[ici] = gu[(i - 1) as usize];
            }
        }
    }
}

/*
 * This routine computes the interaction rates for the species
 * c_1, ... ,c_ns (stored in c[0],...,c[ns-1]), at one spatial point
 * and at time t. (The C version reads ns/acoef/bcoef through wdata;
 * they are passed explicitly here to satisfy disjoint borrows.)
 */

fn WebRates(
    x: sunrealtype,
    y: sunrealtype,
    _t: sunrealtype,
    c: &[sunrealtype],
    rate: &mut [sunrealtype],
    ns: i32,
    acoef: &[[sunrealtype; NS as usize]; NS as usize],
    bcoef: &[sunrealtype; NS as usize],
) {
    let ns = ns as usize;

    for i in 0..ns {
        rate[i] = ZERO;
    }

    for j in 0..ns {
        for i in 0..ns {
            rate[i] += c[j] * acoef[i][j];
        }
    }

    let fac = ONE + ALPHA * x * y;
    for i in 0..ns {
        rate[i] = c[i] * (bcoef[i] * fac + rate[i]);
    }
}

/*
 * This routine computes the interaction rates for the backward problem
 */

#[allow(clippy::too_many_arguments)]
fn WebRatesB(
    x: sunrealtype,
    y: sunrealtype,
    _t: sunrealtype,
    c: &[sunrealtype],
    cB: &[sunrealtype],
    rate: &mut [sunrealtype],
    rateB: &mut [sunrealtype],
    ns: i32,
    acoef: &[[sunrealtype; NS as usize]; NS as usize],
    bcoef: &[sunrealtype; NS as usize],
) {
    let ns = ns as usize;

    let fac = ONE + ALPHA * x * y;

    for i in 0..ns {
        rate[i] = bcoef[i] * fac;
    }

    for j in 0..ns {
        for i in 0..ns {
            rate[i] += acoef[i][j] * c[j];
        }
    }

    for i in 0..ns {
        rateB[i] = cB[i] * rate[i];
        rate[i] = c[i] * rate[i];
    }

    for j in 0..ns {
        for i in 0..ns {
            rateB[i] += acoef[j][i] * c[j] * cB[j];
        }
    }
}

/*
 * This routine computes one block of the interaction terms of the
 * system, namely block (jx,jy), for use in preconditioning.
 * Here jx and jy count from 0.
 */

fn fblock(
    t: sunrealtype,
    cdata: &[sunrealtype],
    jx: i32,
    jy: i32,
    cdotdata: &mut [sunrealtype],
    wdata: &WebData,
) {
    let iblok = jx + jy * wdata.mx;
    let y = jy as sunrealtype * wdata.dy;
    let x = jx as sunrealtype * wdata.dx;
    let ic = (wdata.ns * iblok) as usize;
    WebRates(
        x,
        y,
        t,
        &cdata[ic..],
        cdotdata,
        wdata.ns,
        &wdata.acoef,
        &wdata.bcoef,
    );
}

/*
 * This routine performs ITMAX=5 Gauss-Seidel iterations to compute an
 * approximation to (P-inverse)*z, where P = I - gamma*Jd, and
 * Jd represents the diffusion contributions to the Jacobian.
 * The answer is stored in z on return, and x is a temporary vector.
 * The dimensions below assume a global constant NS >= ns.
 * Some inner loops of length ns are implemented with the small
 * vector kernels v_sum_prods, v_prod, v_inc_by_prod.
 */

fn GSIter(gamma: sunrealtype, z: &N_Vector, x: &N_Vector, wdata: &WebData) {
    let ns = wdata.ns;
    let mx = wdata.mx;
    let my = wdata.my;
    let mxns = wdata.mxns;
    let cox = &wdata.cox;
    let coy = &wdata.coy;

    let mut beta = [ZERO; NS as usize];
    let mut beta2 = [ZERO; NS as usize];
    let mut cof1 = [ZERO; NS as usize];
    let mut gam = [ZERO; NS as usize];
    let mut gam2 = [ZERO; NS as usize];

    /* Write matrix as P = D - L - U.
    Load local arrays beta, beta2, gam, gam2, and cof1. */

    for i in 0..ns as usize {
        let temp = ONE / (ONE + TWO * gamma * (cox[i] + coy[i]));
        beta[i] = gamma * cox[i] * temp;
        beta2[i] = TWO * beta[i];
        gam[i] = gamma * coy[i] * temp;
        gam2[i] = TWO * gam[i];
        cof1[i] = temp;
    }

    /* Begin iteration loop.
    Load vector x with (D-inverse)*z for first iteration. */

    {
        let mut xd = N_VGetArrayPointer(x).expect("N_VGetArrayPointer");
        let zd = N_VGetArrayPointer(z).expect("N_VGetArrayPointer");
        for jy in 0..my {
            let iyoff = mxns * jy;
            for jx in 0..mx {
                let ic = (iyoff + ns * jx) as usize;
                /* x[ic+i] = cof1[i]z[ic+i] */
                v_prod2(&mut xd[ic..], &cof1, &zd[ic..], ns);
            }
        }
    }
    N_VConst(ZERO, z);

    /* Looping point for iterations. */

    for iter in 1..=ITMAX {
        /* Calculate (D-inverse)*U*x if not the first iteration. */

        if iter > 1 {
            let mut xd = N_VGetArrayPointer(x).expect("N_VGetArrayPointer");
            for jy in 0..my {
                let iyoff = mxns * jy;
                for jx in 0..mx {
                    /* order of loops matters */
                    let ic = iyoff + ns * jx;
                    let x_loc = if jx == 0 {
                        0
                    } else if jx == mx - 1 {
                        2
                    } else {
                        1
                    };
                    let y_loc = if jy == 0 {
                        0
                    } else if jy == my - 1 {
                        2
                    } else {
                        1
                    };
                    let ic = ic as usize;
                    let nsu = ns as usize;
                    let mxnsu = mxns as usize;
                    match 3 * y_loc + x_loc {
                        0 => {
                            /* jx == 0, jy == 0 */
                            /* x[ic+i] = beta2[i]x[ic+ns+i] + gam2[i]x[ic+mxns+i] */
                            v_sum_prods(&mut xd, ic, &beta2, ic + nsu, &gam2, ic + mxnsu, ns);
                        }
                        1 => {
                            /* 1 <= jx <= mx-2, jy == 0 */
                            /* x[ic+i] = beta[i]x[ic+ns+i] + gam2[i]x[ic+mxns+i] */
                            v_sum_prods(&mut xd, ic, &beta, ic + nsu, &gam2, ic + mxnsu, ns);
                        }
                        2 => {
                            /* jx == mx-1, jy == 0 */
                            /* x[ic+i] = gam2[i]x[ic+mxns+i] */
                            v_prod(&mut xd, ic, &gam2, ic + mxnsu, ns);
                        }
                        3 => {
                            /* jx == 0, 1 <= jy <= my-2 */
                            /* x[ic+i] = beta2[i]x[ic+ns+i] + gam[i]x[ic+mxns+i] */
                            v_sum_prods(&mut xd, ic, &beta2, ic + nsu, &gam, ic + mxnsu, ns);
                        }
                        4 => {
                            /* 1 <= jx <= mx-2, 1 <= jy <= my-2 */
                            /* x[ic+i] = beta[i]x[ic+ns+i] + gam[i]x[ic+mxns+i] */
                            v_sum_prods(&mut xd, ic, &beta, ic + nsu, &gam, ic + mxnsu, ns);
                        }
                        5 => {
                            /* jx == mx-1, 1 <= jy <= my-2 */
                            /* x[ic+i] = gam[i]x[ic+mxns+i] */
                            v_prod(&mut xd, ic, &gam, ic + mxnsu, ns);
                        }
                        6 => {
                            /* jx == 0, jy == my-1 */
                            /* x[ic+i] = beta2[i]x[ic+ns+i] */
                            v_prod(&mut xd, ic, &beta2, ic + nsu, ns);
                        }
                        7 => {
                            /* 1 <= jx <= mx-2, jy == my-1 */
                            /* x[ic+i] = beta[i]x[ic+ns+i] */
                            v_prod(&mut xd, ic, &beta, ic + nsu, ns);
                        }
                        8 => {
                            /* jx == mx-1, jy == my-1 */
                            /* x[ic+i] = ZERO */
                            v_zero(&mut xd, ic, ns);
                        }
                        _ => {}
                    }
                }
            }
        } /* end if (iter > 1) */

        /* Overwrite x with [(I - (D-inverse)*L)-inverse]*x. */

        {
            let mut xd = N_VGetArrayPointer(x).expect("N_VGetArrayPointer");
            for jy in 0..my {
                let iyoff = mxns * jy;
                for jx in 0..mx {
                    /* order of loops matters */
                    let ic = iyoff + ns * jx;
                    let x_loc = if jx == 0 {
                        0
                    } else if jx == mx - 1 {
                        2
                    } else {
                        1
                    };
                    let y_loc = if jy == 0 {
                        0
                    } else if jy == my - 1 {
                        2
                    } else {
                        1
                    };
                    let ic = ic as usize;
                    let nsu = ns as usize;
                    let mxnsu = mxns as usize;
                    match 3 * y_loc + x_loc {
                        0 => { /* jx == 0, jy == 0 */ }
                        1 => {
                            /* 1 <= jx <= mx-2, jy == 0 */
                            /* x[ic+i] += beta[i]x[ic-ns+i] */
                            v_inc_by_prod(&mut xd, ic, &beta, ic - nsu, ns);
                        }
                        2 => {
                            /* jx == mx-1, jy == 0 */
                            /* x[ic+i] += beta2[i]x[ic-ns+i] */
                            v_inc_by_prod(&mut xd, ic, &beta2, ic - nsu, ns);
                        }
                        3 => {
                            /* jx == 0, 1 <= jy <= my-2 */
                            /* x[ic+i] += gam[i]x[ic-mxns+i] */
                            v_inc_by_prod(&mut xd, ic, &gam, ic - mxnsu, ns);
                        }
                        4 => {
                            /* 1 <= jx <= mx-2, 1 <= jy <= my-2 */
                            /* x[ic+i] += beta[i]x[ic-ns+i] + gam[i]x[ic-mxns+i] */
                            v_inc_by_prod(&mut xd, ic, &beta, ic - nsu, ns);
                            v_inc_by_prod(&mut xd, ic, &gam, ic - mxnsu, ns);
                        }
                        5 => {
                            /* jx == mx-1, 1 <= jy <= my-2 */
                            /* x[ic+i] += beta2[i]x[ic-ns+i] + gam[i]x[ic-mxns+i] */
                            v_inc_by_prod(&mut xd, ic, &beta2, ic - nsu, ns);
                            v_inc_by_prod(&mut xd, ic, &gam, ic - mxnsu, ns);
                        }
                        6 => {
                            /* jx == 0, jy == my-1 */
                            /* x[ic+i] += gam2[i]x[ic-mxns+i] */
                            v_inc_by_prod(&mut xd, ic, &gam2, ic - mxnsu, ns);
                        }
                        7 => {
                            /* 1 <= jx <= mx-2, jy == my-1 */
                            /* x[ic+i] += beta[i]x[ic-ns+i] + gam2[i]x[ic-mxns+i] */
                            v_inc_by_prod(&mut xd, ic, &beta, ic - nsu, ns);
                            v_inc_by_prod(&mut xd, ic, &gam2, ic - mxnsu, ns);
                        }
                        8 => {
                            /* jx == mx-1, jy == my-1 */
                            /* x[ic+i] += beta2[i]x[ic-ns+i] + gam2[i]x[ic-mxns+i] */
                            v_inc_by_prod(&mut xd, ic, &beta2, ic - nsu, ns);
                            v_inc_by_prod(&mut xd, ic, &gam2, ic - mxnsu, ns);
                        }
                        _ => {}
                    }
                }
            }
        }

        /* Add increment x to z : z <- z+x */

        N_VLinearSum(ONE, z, ONE, x, z);
    }
}

/* Small Vector Kernels.
 * The C kernels take raw pointers that may point into the same array
 * (xd); here the same-array variants take one slice plus offsets, and
 * v_prod2 serves the distinct-array case (x and z). Arithmetic per
 * element is identical to the C kernels. */

fn v_inc_by_prod(xd: &mut [sunrealtype], u: usize, v: &[sunrealtype], w: usize, n: i32) {
    for i in 0..n as usize {
        xd[u + i] += v[i] * xd[w + i];
    }
}

fn v_sum_prods(
    xd: &mut [sunrealtype],
    u: usize,
    p: &[sunrealtype],
    q: usize,
    v: &[sunrealtype],
    w: usize,
    n: i32,
) {
    for i in 0..n as usize {
        xd[u + i] = p[i] * xd[q + i] + v[i] * xd[w + i];
    }
}

fn v_prod(xd: &mut [sunrealtype], u: usize, v: &[sunrealtype], w: usize, n: i32) {
    for i in 0..n as usize {
        xd[u + i] = v[i] * xd[w + i];
    }
}

fn v_prod2(u: &mut [sunrealtype], v: &[sunrealtype], w: &[sunrealtype], n: i32) {
    for i in 0..n as usize {
        u[i] = v[i] * w[i];
    }
}

fn v_zero(u: &mut [sunrealtype], off: usize, n: i32) {
    for i in 0..n as usize {
        u[off + i] = ZERO;
    }
}

/*
 * Print maximum sensitivity of G for each species
 */

fn PrintOutput(cB: &N_Vector, ns: i32, mxns: i32, wdata: &WebData) {
    let cdata = N_VGetArrayPointer(cB).expect("N_VGetArrayPointer");

    let mut x = ZERO;
    let mut y = ZERO;

    for i in 1..=ns {
        let mut cmax = ZERO;

        for jy in (0..MY).rev() {
            for jx in 0..MX {
                let cij = cdata[((i - 1) + jx * ns + jy * mxns) as usize];
                if SUNRabs(cij) > cmax {
                    cmax = cij;
                    x = jx as sunrealtype * wdata.dx;
                    y = jy as sunrealtype * wdata.dy;
                }
            }
        }

        print!(
            "\nMaximum sensitivity with respect to I.C. of species {}\n",
            i
        );
        print!("  mu max = {}\n", fmt_e(cmax, 6));
        print!("at\n");
        print!("  x = {}\n  y = {}\n", fmt_e(x, 6), fmt_e(y, 6));
    }
}

/*
 * Compute double space integral
 */

fn doubleIntgr(cdata: &[sunrealtype], i: i32, wdata: &WebData) -> sunrealtype {
    let ns = wdata.ns;
    let mx = wdata.mx;
    let my = wdata.my;
    let mxns = wdata.mxns;
    let dx = wdata.dx;
    let dy = wdata.dy;

    let mut jy = 0;
    let mut intgr_x = cdata[((i - 1) + jy * mxns) as usize];
    for jx in 1..mx - 1 {
        intgr_x += TWO * cdata[((i - 1) + jx * ns + jy * mxns) as usize];
    }
    intgr_x += cdata[((i - 1) + (mx - 1) * ns + jy * mxns) as usize];
    intgr_x *= 0.5 * dx;

    let mut intgr_xy = intgr_x;

    for jy in 1..my - 1 {
        let mut intgr_x = cdata[((i - 1) + jy * mxns) as usize];
        for jx in 1..mx - 1 {
            intgr_x += TWO * cdata[((i - 1) + jx * ns + jy * mxns) as usize];
        }
        intgr_x += cdata[((i - 1) + (mx - 1) * ns + jy * mxns) as usize];
        intgr_x *= 0.5 * dx;

        intgr_xy += TWO * intgr_x;
    }

    jy = my - 1;
    let mut intgr_x = cdata[((i - 1) + jy * mxns) as usize];
    for jx in 1..mx - 1 {
        intgr_x += TWO * cdata[((i - 1) + jx * ns + jy * mxns) as usize];
    }
    intgr_x += cdata[((i - 1) + (mx - 1) * ns + jy * mxns) as usize];
    intgr_x *= 0.5 * dx;

    intgr_xy += intgr_x;

    intgr_xy *= 0.5 * dy;

    intgr_xy
}

/*
 * Check function return value.
 *    opt == 0 means SUNDIALS function allocates memory so check if
 *             returned NULL pointer
 *    opt == 1 means SUNDIALS function returns an integer value so check if
 *             retval < 0
 *    opt == 2 means function allocates memory so check if returned
 *             NULL pointer
 */

fn check_retval(retval: i32, funcname: &str, _opt: i32) -> i32 {
    /* Check if retval < 0 */
    if retval < 0 {
        eprint!(
            "\nSUNDIALS_ERROR: {}() failed with retval = {}\n\n",
            funcname, retval
        );
        return 1;
    }

    0
}

fn check_retval_ptr<T>(returnvalue: &Option<T>, funcname: &str, opt: i32) -> i32 {
    /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
    if opt == 0 && returnvalue.is_none() {
        eprint!(
            "\nSUNDIALS_ERROR: {}() failed - returned NULL pointer\n\n",
            funcname
        );
        return 1;
    }

    /* Check if function returned NULL pointer - no memory allocated */
    if opt == 2 && returnvalue.is_none() {
        eprint!(
            "\nMEMORY_ERROR: {}() failed - returned NULL pointer\n\n",
            funcname
        );
        return 1;
    }

    0
}
