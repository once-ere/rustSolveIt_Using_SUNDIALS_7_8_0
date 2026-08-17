/* --------------------------------------------------------------------
 * Rust port of examples/cvodes/serial/cvsKrylovDemo_prec.c
 * --------------------------------------------------------------------
 * Demonstration program for CVODES - Krylov linear solver.
 * ODE system from ns-species interaction PDE in 2 dimensions.
 *
 * Food web problem: predator-prey interaction and diffusion on the
 * unit square. Preconditioner is a product of (1) Gauss-Seidel
 * iterations on the diffusion terms and (2) a block-diagonal matrix
 * from the interaction terms with block-grouping, using the SUNDlsMat
 * dense kernels. Four runs: jpre = SUN_PREC_LEFT / SUN_PREC_RIGHT,
 * each with SUN_MODIFIED_GS / SUN_CLASSICAL_GS.
 * --------------------------------------------------------------------
 */
#![allow(non_snake_case, non_camel_case_types, non_upper_case_globals)]

use std::any::Any;

use cvodes_rs::prelude::*;
use cvodes_rs::sundials_dense::{
    SUNDlsMat_denseAddIdentity, SUNDlsMat_denseGETRF, SUNDlsMat_denseGETRS,
};

/* Constants */

const ZERO: sunrealtype = 0.0;
const ONE: sunrealtype = 1.0;

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

const MX: i32 = 6;
const MY: i32 = 6;
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

/* Spgmr/CVLS Constants */

const MAXL: i32 = 0; /* => use default = MIN(NEQ, 5)            */
const DELTA: sunrealtype = ZERO; /* => use default = 0.05                   */

/* Output Constants */

const T1: sunrealtype = 1.0e-8;
const TOUT_MULT: sunrealtype = 10.0;
const DTOUT: sunrealtype = ONE;
const NOUT: i32 = 18;

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

/* Note: The value for species i at mesh point (j,k) is stored in */
/* component number (i-1) + j*NS + k*NS*MX of an N_Vector,        */
/* where 1 <= i <= NS, 0 <= j < MX, 0 <= k < MY.                  */

/* Structure for user data */

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
    fsave: [sunrealtype; NEQ as usize],
    tmp: N_Vector,
    rewt: N_Vector,
    cvode_mem: Option<CVodeMem>,
}

/* Implementation */

fn main() {
    let abstol: sunrealtype = ATOL;
    let reltol: sunrealtype = RTOL;
    let mut t: sunrealtype = 0.0;
    let mut tout: sunrealtype;

    /* Create the SUNDIALS context */
    let mut sunctx_opt: Option<SUNContext> = None;
    let retval = SUNContext_Create(SUN_COMM_NULL, &mut sunctx_opt);
    if check_retval(retval, "SUNContext_Create", 1) != 0 {
        std::process::exit(1);
    }
    let sunctx = sunctx_opt.as_ref().expect("SUNContext").clone();

    /* Initializations */
    let c = N_VNew_Serial(NEQ as sunindextype, &sunctx);
    if check_retval_ptr(&c, "N_VNew_Serial", 0) != 0 {
        std::process::exit(1);
    }
    let c = c.expect("N_VNew_Serial");
    let wdata = AllocUserData(&sunctx);
    if check_retval_ptr(&wdata, "AllocUserData", 2) != 0 {
        std::process::exit(1);
    }
    let mut wdata = wdata.expect("AllocUserData");
    InitUserData(&mut wdata);
    let ns = wdata.ns;
    let mxns = wdata.mxns;

    /* Print problem description */
    PrintIntro();

    /* user_data box handed to CVODE on the first run; retrieved by swap
     * (CVodeGetUserData) whenever main needs it again. */
    let mut wdata_any: Option<Box<dyn Any>> = Some(Box::new(wdata));
    let mut cvode_mem: Option<CVodeMem> = None;
    let mut LS: Option<SUNLinearSolver> = None;

    /* Loop over jpre and gstype (four cases) */
    for jpre in SUN_PREC_LEFT..=SUN_PREC_RIGHT {
        for gstype in SUN_MODIFIED_GS..=SUN_CLASSICAL_GS {
            let firstrun = (jpre == SUN_PREC_LEFT) && (gstype == SUN_MODIFIED_GS);

            /* Initialize c and print heading */
            if firstrun {
                let wdata_ref = wdata_any
                    .as_ref()
                    .and_then(|b| b.downcast_ref::<WebData>())
                    .expect("user_data is WebData");
                CInit(&c, wdata_ref);
            } else {
                let mem = cvode_mem.as_ref().expect("cvode_mem");
                CVodeGetUserData(mem, &mut wdata_any);
                {
                    let wdata_ref = wdata_any
                        .as_ref()
                        .and_then(|b| b.downcast_ref::<WebData>())
                        .expect("user_data is WebData");
                    CInit(&c, wdata_ref);
                }
                CVodeGetUserData(mem, &mut wdata_any); /* hand the box back */
            }
            PrintHeader(jpre, gstype);

            /* Call CVodeInit or CVodeReInit, then SUNLinSol_SPGMR to set up problem */

            if firstrun {
                cvode_mem = CVodeCreate(CV_BDF, &sunctx);
                if check_retval_ptr(&cvode_mem, "CVodeCreate", 0) != 0 {
                    std::process::exit(1);
                }
                let mem = cvode_mem.as_ref().expect("cvode_mem");

                wdata_any
                    .as_mut()
                    .and_then(|b| b.downcast_mut::<WebData>())
                    .expect("user_data is WebData")
                    .cvode_mem = Some(mem.clone());

                let retval = CVodeSetUserData(mem, wdata_any.take());
                if check_retval(retval, "CVodeSetUserData", 1) != 0 {
                    std::process::exit(1);
                }

                let retval = CVodeInit(mem, f, T0, &c);
                if check_retval(retval, "CVodeInit", 1) != 0 {
                    std::process::exit(1);
                }

                let retval = CVodeSStolerances(mem, reltol, abstol);
                if check_retval(retval, "CVodeSStolerances", 1) != 0 {
                    std::process::exit(1);
                }

                LS = SUNLinSol_SPGMR(&c, jpre, MAXL, &sunctx);
                if check_retval_ptr(&LS, "SUNLinSol_SPGMR", 0) != 0 {
                    std::process::exit(1);
                }
                let ls = LS.as_ref().expect("LS");

                let retval = CVodeSetLinearSolver(mem, ls, None);
                if check_retval(retval, "CVodeSetLinearSolver", 1) != 0 {
                    std::process::exit(1);
                }

                let retval = SUNLinSol_SPGMRSetGSType(ls, gstype);
                if check_retval(retval, "SUNLinSol_SPGMRSetGSType", 1) != 0 {
                    std::process::exit(1);
                }

                let retval = CVodeSetEpsLin(mem, DELTA);
                if check_retval(retval, "CVodeSetEpsLin", 1) != 0 {
                    std::process::exit(1);
                }

                let retval = CVodeSetPreconditioner(mem, Some(Precond), Some(PSolve));
                if check_retval(retval, "CVodeSetPreconditioner", 1) != 0 {
                    std::process::exit(1);
                }
            } else {
                let mem = cvode_mem.as_ref().expect("cvode_mem");
                let ls = LS.as_ref().expect("LS");

                let retval = CVodeReInit(mem, T0, &c);
                if check_retval(retval, "CVodeReInit", 1) != 0 {
                    std::process::exit(1);
                }

                let retval = SUNLinSol_SPGMRSetPrecType(ls, jpre);
                if check_retval(retval, "SUNLinSol_SPGMRSetPrecType", 1) != 0 {
                    std::process::exit(1);
                }
                let retval = SUNLinSol_SPGMRSetGSType(ls, gstype);
                if check_retval(retval, "SUNLinSol_SPGMRSetGSType", 1) != 0 {
                    std::process::exit(1);
                }
            }

            /* Print initial values */
            if firstrun {
                PrintAllSpecies(&c, ns, mxns, T0);
            }

            /* Loop over output points, call CVode, print sample solution values. */
            let mem = cvode_mem.as_ref().expect("cvode_mem");
            tout = T1;
            for iout in 1..=NOUT {
                let retval = CVode(mem, tout, &c, &mut t, CV_NORMAL);
                PrintOutput(mem, t);
                if firstrun && (iout % 3 == 0) {
                    PrintAllSpecies(&c, ns, mxns, t);
                }
                if check_retval(retval, "CVode", 1) != 0 {
                    break;
                }
                if tout > 0.9 {
                    tout += DTOUT;
                } else {
                    tout *= TOUT_MULT;
                }
            }

            /* Print final statistics, and loop for next case */
            PrintFinalStats(mem);
        }
    }

    /* Free all memory (FreeUserData: the WebData box is owned by the
     * integrator and dropped with it) */
    CVodeFree(&mut cvode_mem);
    N_VDestroy(c);
    SUNLinSolFree(LS);

    drop(sunctx);
    SUNContext_Free(&mut sunctx_opt);
}

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
    let tmp = N_VNew_Serial(NEQ as sunindextype, sunctx)?;

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
        fsave: [ZERO; NEQ as usize],
        tmp,
        rewt,
        cvode_mem: None,
    })
}

fn InitUserData(wdata: &mut WebData) {
    let np = NP as usize;

    wdata.ns = NS;
    let ns = NS as usize;

    for j in 0..ns {
        for i in 0..ns {
            wdata.acoef[i][j] = 0.;
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
 This routine sets arrays jg, jig, and jr describing
 a uniform partition of (0,1,2,...,m-1) into ng groups.
 The arrays set are:
   jg    = length ng+1 array of group boundaries.
           Group ig has indices j = jg[ig],...,jg[ig+1]-1.
   jig   = length m array of group indices vs node index.
           Node index j is in group jig[j].
   jr    = length ng array of indices representing the groups.
           The index for group ig is j = jr[ig].
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

/* This routine computes and loads the vector of initial values. */
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

fn PrintIntro() {
    print!("\n\nDemonstration program for CVODES - SPGMR linear solver\n\n");
    print!("Food web problem with ns species, ns = {}\n", NS);
    print!("Predator-prey interaction and diffusion on a 2-D square\n\n");
    print!(
        "Matrix parameters: a = {}   e = {}   g = {}\n",
        fmt_g(AA, 2),
        fmt_g(EE, 2),
        fmt_g(GG, 2)
    );
    print!("b parameter = {}\n", fmt_g(BB, 2));
    print!(
        "Diffusion coefficients: Dprey = {}   Dpred = {}\n",
        fmt_g(DPREY, 2),
        fmt_g(DPRED, 2)
    );
    print!("Rate parameter alpha = {}\n\n", fmt_g(ALPHA, 2));
    print!("Mesh dimensions (mx,my) are {}, {}.  ", MX, MY);
    print!("Total system size is neq = {} \n\n", NEQ);
    print!(
        "Tolerances: reltol = {}, abstol = {} \n\n",
        fmt_g(RTOL, 2),
        fmt_g(ATOL, 2)
    );
    print!("Preconditioning uses a product of:\n");
    print!("  (1) Gauss-Seidel iterations with ");
    print!("itmax = {} iterations, and\n", ITMAX);
    print!("  (2) interaction-only block-diagonal matrix ");
    print!("with block-grouping\n");
    print!("  Number of diagonal block groups = ngrp = {}", NGRP);
    print!("  (ngx by ngy, ngx = {}, ngy = {})\n", NGX, NGY);
    print!("\n\n--------------------------------------------------------------");
    print!("--------------\n");
}

fn PrintHeader(jpre: i32, gstype: i32) {
    if jpre == SUN_PREC_LEFT {
        print!(
            "\n\nPreconditioner type is           jpre = {}\n",
            "SUN_PREC_LEFT"
        );
    } else {
        print!(
            "\n\nPreconditioner type is           jpre = {}\n",
            "SUN_PREC_RIGHT"
        );
    }

    if gstype == SUN_MODIFIED_GS {
        print!(
            "\nGram-Schmidt method type is    gstype = {}\n\n\n",
            "SUN_MODIFIED_GS"
        );
    } else {
        print!(
            "\nGram-Schmidt method type is    gstype = {}\n\n\n",
            "SUN_CLASSICAL_GS"
        );
    }
}

fn PrintAllSpecies(c: &N_Vector, ns: i32, mxns: i32, t: sunrealtype) {
    let cdata = N_VGetArrayPointer(c).expect("N_VGetArrayPointer");
    print!("c values at t = {}:\n\n", fmt_g(t, 6));
    for i in 1..=ns {
        print!("Species {}\n", i);
        for jy in (0..MY).rev() {
            for jx in 0..MX {
                print!(
                    "{:<10}",
                    fmt_g(cdata[((i - 1) + jx * ns + jy * mxns) as usize], 6)
                );
            }
            print!("\n");
        }
        print!("\n");
    }
}

fn PrintOutput(cvode_mem: &CVodeMem, t: sunrealtype) {
    let mut nst: i64 = 0;
    let mut nfe: i64 = 0;
    let mut nni: i64 = 0;
    let mut qu: i32 = 0;
    let mut hu: sunrealtype = 0.0;

    let retval = CVodeGetNumSteps(cvode_mem, &mut nst);
    check_retval(retval, "CVodeGetNumSteps", 1);
    let retval = CVodeGetNumRhsEvals(cvode_mem, &mut nfe);
    check_retval(retval, "CVodeGetNumRhsEvals", 1);
    let retval = CVodeGetNumNonlinSolvIters(cvode_mem, &mut nni);
    check_retval(retval, "CVodeGetNumNonlinSolvIters", 1);
    let retval = CVodeGetLastOrder(cvode_mem, &mut qu);
    check_retval(retval, "CVodeGetLastOrder", 1);
    let retval = CVodeGetLastStep(cvode_mem, &mut hu);
    check_retval(retval, "CVodeGetLastStep", 1);

    print!(
        "t = {}  nst = {}  nfe = {}  nni = {}",
        fmt_ew(t, 10, 2),
        nst,
        nfe,
        nni
    );
    print!("  qu = {}  hu = {}\n\n", qu, fmt_ew(hu, 11, 2));
}

fn PrintFinalStats(cvode_mem: &CVodeMem) {
    let mut lenrw: i64 = 0;
    let mut leniw: i64 = 0;
    let mut lenrwLS: i64 = 0;
    let mut leniwLS: i64 = 0;
    let mut nst: i64 = 0;
    let mut nfe: i64 = 0;
    let mut nsetups: i64 = 0;
    let mut nni: i64 = 0;
    let mut ncfn: i64 = 0;
    let mut netf: i64 = 0;
    let mut nli: i64 = 0;
    let mut npe: i64 = 0;
    let mut nps: i64 = 0;
    let mut ncfl: i64 = 0;
    let mut nfeLS: i64 = 0;

    let retval = CVodeGetWorkSpace(cvode_mem, &mut lenrw, &mut leniw);
    check_retval(retval, "CVodeGetWorkSpace", 1);
    let retval = CVodeGetNumSteps(cvode_mem, &mut nst);
    check_retval(retval, "CVodeGetNumSteps", 1);
    let retval = CVodeGetNumRhsEvals(cvode_mem, &mut nfe);
    check_retval(retval, "CVodeGetNumRhsEvals", 1);
    let retval = CVodeGetNumLinSolvSetups(cvode_mem, &mut nsetups);
    check_retval(retval, "CVodeGetNumLinSolvSetups", 1);
    let retval = CVodeGetNumErrTestFails(cvode_mem, &mut netf);
    check_retval(retval, "CVodeGetNumErrTestFails", 1);
    let retval = CVodeGetNumNonlinSolvIters(cvode_mem, &mut nni);
    check_retval(retval, "CVodeGetNumNonlinSolvIters", 1);
    let retval = CVodeGetNumNonlinSolvConvFails(cvode_mem, &mut ncfn);
    check_retval(retval, "CVodeGetNumNonlinSolvConvFails", 1);

    let retval = CVodeGetLinWorkSpace(cvode_mem, &mut lenrwLS, &mut leniwLS);
    check_retval(retval, "CVodeGetLinWorkSpace", 1);
    let retval = CVodeGetNumLinIters(cvode_mem, &mut nli);
    check_retval(retval, "CVodeGetNumLinIters", 1);
    let retval = CVodeGetNumPrecEvals(cvode_mem, &mut npe);
    check_retval(retval, "CVodeGetNumPrecEvals", 1);
    let retval = CVodeGetNumPrecSolves(cvode_mem, &mut nps);
    check_retval(retval, "CVodeGetNumPrecSolves", 1);
    let retval = CVodeGetNumLinConvFails(cvode_mem, &mut ncfl);
    check_retval(retval, "CVodeGetNumLinConvFails", 1);
    let retval = CVodeGetNumLinRhsEvals(cvode_mem, &mut nfeLS);
    check_retval(retval, "CVodeGetNumLinRhsEvals", 1);

    print!("\n\n Final statistics for this run:\n\n");
    print!(" CVode real workspace length           = {:4} \n", lenrw);
    print!(" CVode integer workspace length        = {:4} \n", leniw);
    print!(" CVLS real workspace length            = {:4} \n", lenrwLS);
    print!(" CVLS integer workspace length         = {:4} \n", leniwLS);
    print!(" Number of steps                       = {:4} \n", nst);
    print!(" Number of f-s                         = {:4} \n", nfe);
    print!(" Number of f-s (SPGMR)                 = {:4} \n", nfeLS);
    print!(
        " Number of f-s (TOTAL)                 = {:4} \n",
        nfe + nfeLS
    );
    print!(" Number of setups                      = {:4} \n", nsetups);
    print!(" Number of nonlinear iterations        = {:4} \n", nni);
    print!(" Number of linear iterations           = {:4} \n", nli);
    print!(" Number of preconditioner evaluations  = {:4} \n", npe);
    print!(" Number of preconditioner solves       = {:4} \n", nps);
    print!(" Number of error test failures         = {:4} \n", netf);
    print!(" Number of nonlinear conv. failures    = {:4} \n", ncfn);
    print!(" Number of linear convergence failures = {:4} \n", ncfl);
    let avdim = if nni > 0 {
        nli as sunrealtype / nni as sunrealtype
    } else {
        ZERO
    };
    print!(
        " Average Krylov subspace dimension     = {} \n",
        fmt_f(avdim, 3)
    );
    print!("\n\n--------------------------------------------------------------");
    print!("--------------\n");
    print!("--------------------------------------------------------------");
    print!("--------------\n");
}

/*
 This routine computes the right-hand side of the ODE system and
 returns it in cdot. The interaction rates are computed by calls to WebRates,
 and these are saved in fsave for use in preconditioning.
*/
fn f(t: sunrealtype, c: &N_Vector, cdot: &N_Vector, user_data: &mut Option<Box<dyn Any>>) -> i32 {
    let wdata = user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<WebData>())
        .expect("user_data is WebData");

    let cdata = N_VGetArrayPointer(c).expect("N_VGetArrayPointer");
    let mut cdotdata = N_VGetArrayPointer(cdot).expect("N_VGetArrayPointer");

    let mxns = wdata.mxns;
    let ns = wdata.ns;
    let dx = wdata.dx;
    let dy = wdata.dy;

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
                &mut wdata.fsave[ic as usize..],
                ns,
                &wdata.acoef,
                &wdata.bcoef,
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
                cdotdata[ici as usize] = wdata.coy[(i - 1) as usize] * (dcyui - dcyli)
                    + wdata.cox[(i - 1) as usize] * (dcxui - dcxli)
                    + wdata.fsave[ici as usize];
            }
        }
    }

    0
}

/*
  This routine computes the interaction rates for the species
  c_1, ... ,c_ns (stored in c[0],...,c[ns-1]), at one spatial point
  and at time t. (The C version reads ns/acoef/bcoef through wdata;
  they are passed explicitly here to satisfy disjoint borrows.)
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
 This routine generates the block-diagonal part of the Jacobian
 corresponding to the interaction rates, multiplies by -gamma, adds
 the identity matrix, and calls SUNDlsMat_denseGETRF to do the LU
 decomposition of each diagonal block. The computation of the diagonal
 blocks uses the preset block and grouping information. One block per
 group is computed. The Jacobian elements are generated by difference
 quotients using calls to the routine fblock.
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
    let wdata = user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<WebData>())
        .expect("user_data is WebData");

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

    let mut fac = N_VWrmsNorm(fc, &rewt);
    let mut r0 = 1000.0 * SUNRabs(gamma) * uround * NEQ as sunrealtype * fac;
    if r0 == ZERO {
        r0 = ONE;
    }

    let tmp = wdata.tmp.clone();
    {
        let mut cdata = N_VGetArrayPointer(c).expect("N_VGetArrayPointer");
        let rewtdata = N_VGetArrayPointer(&rewt).expect("N_VGetArrayPointer");
        let mut f1 = N_VGetArrayPointer(&tmp).expect("N_VGetArrayPointer");

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
  This routine computes one block of the interaction terms of the
  system, namely block (jx,jy), for use in preconditioning.
  Here jx and jy count from 0.
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
  This routine applies two inverse preconditioner matrices
  to the vector r, using the interaction-only block-diagonal Jacobian
  with block-grouping, denoted Jr, and Gauss-Seidel applied to the
  diffusion contribution to the Jacobian, denoted Jd.
  It first calls GSIter for a Gauss-Seidel approximation to
  ((I - gamma*Jd)-inverse)*r, and stores the result in z.
  Then it computes ((I - gamma*Jr)-inverse)*z, using LU factors of the
  blocks in P, and pivot information in pivot, and returns the result in z.
*/
fn PSolve(
    _tn: sunrealtype,
    _c: &N_Vector,
    _fc: &N_Vector,
    r: &N_Vector,
    z: &N_Vector,
    gamma: sunrealtype,
    _delta: sunrealtype,
    _lr: i32,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    let wdata = user_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<WebData>())
        .expect("user_data is WebData");

    N_VScale(ONE, r, z);

    /* call GSIter for Gauss-Seidel iterations */

    let tmp = wdata.tmp.clone();
    GSIter(gamma, z, &tmp, wdata);

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
  This routine performs ITMAX=5 Gauss-Seidel iterations to compute an
  approximation to (P-inverse)*z, where P = I - gamma*Jd, and
  Jd represents the diffusion contributions to the Jacobian.
  The answer is stored in z on return, and x is a temporary vector.
  The dimensions below assume a global constant NS >= ns.
  Some inner loops of length ns are implemented with the small
  vector kernels v_sum_prods, v_prod, v_inc_by_prod.
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
        let temp = ONE / (ONE + 2.0 * gamma * (cox[i] + coy[i]));
        beta[i] = gamma * cox[i] * temp;
        beta2[i] = 2.0 * beta[i];
        gam[i] = gamma * coy[i] * temp;
        gam2[i] = 2.0 * gam[i];
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
                            /* x[ic+i] = 0.0 */
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

/* Check function return value...
opt == 0 means SUNDIALS function allocates memory so check if
         returned NULL pointer
opt == 1 means SUNDIALS function returns an integer value so check if
         retval < 0
opt == 2 means function allocates memory so check if returned
         NULL pointer */

fn check_retval(retval: i32, funcname: &str, _opt: i32) -> i32 {
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
    if returnvalue.is_none() {
        if opt == 2 {
            eprint!(
                "\nMEMORY_ERROR: {}() failed - returned NULL pointer\n\n",
                funcname
            );
        } else {
            eprint!(
                "\nSUNDIALS_ERROR: {}() failed - returned NULL pointer\n\n",
                funcname
            );
        }
        return 1;
    }
    0
}
