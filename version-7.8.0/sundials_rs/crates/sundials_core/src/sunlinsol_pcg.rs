//! Port of `src/sunlinsol/pcg/sunlinsol_pcg.c` +
//! `include/sunlinsol/sunlinsol_pcg.h` (preconditioned conjugate gradient).

use std::any::Any;
use std::cell::RefMut;

use crate::sundials_context::SUNContext;
use crate::sundials_errors::{SUN_ERR_ARG_CORRUPT, SUN_SUCCESS};
use crate::sundials_iterative::*;
use crate::sundials_linearsolver::*;
use crate::sundials_math::SUNRsqrt;
use crate::sundials_matrix::SUNMatrix;
use crate::sundials_nvector::*;
use crate::sundials_types::*;

const ZERO: sunrealtype = 0.0;
const ONE: sunrealtype = 1.0;

pub const SUNPCG_MAXL_DEFAULT: i32 = 5;

pub struct SUNLinearSolverContent_PCG_ {
    pub maxl: i32,
    pub pretype: i32,
    pub zeroguess: sunbooleantype,
    pub numiters: i32,
    pub resnorm: sunrealtype,
    pub last_flag: sunindextype,

    pub ATimes: Option<SUNATimesFn>,
    pub ATData: Option<Box<dyn Any>>,
    pub Psetup: Option<SUNPSetupFn>,
    pub Psolve: Option<SUNPSolveFn>,
    pub PData: Option<Box<dyn Any>>,

    pub s: Option<N_Vector>,
    pub r: Option<N_Vector>,
    pub p: Option<N_Vector>,
    pub z: Option<N_Vector>,
    pub Ap: Option<N_Vector>,
}

pub type SUNLinearSolverContent_PCG = SUNLinearSolverContent_PCG_;

fn content_mut(S: &SUNLinearSolver) -> RefMut<'_, SUNLinearSolverContent_PCG_> {
    RefMut::map(S.content.borrow_mut(), |c| {
        c.downcast_mut::<SUNLinearSolverContent_PCG_>()
            .expect("PCG SUNLinearSolver content")
    })
}

pub fn SUNLinSol_PCG(
    y: &N_Vector,
    pretype: i32,
    maxl: i32,
    sunctx: &SUNContext,
) -> Option<SUNLinearSolver> {
    /* check for legal pretype and maxl values; if illegal use defaults */
    let pretype = if pretype != SUN_PREC_NONE
        && pretype != SUN_PREC_LEFT
        && pretype != SUN_PREC_RIGHT
        && pretype != SUN_PREC_BOTH
    {
        SUN_PREC_NONE
    } else {
        pretype
    };
    let maxl = if maxl <= 0 { SUNPCG_MAXL_DEFAULT } else { maxl };

    let S = SUNLinSolNewEmpty(sunctx)?;

    {
        let mut ops = S.ops.borrow_mut();
        ops.gettype = Some(SUNLinSolGetType_PCG);
        ops.getid = Some(SUNLinSolGetID_PCG);
        ops.setatimes = Some(SUNLinSolSetATimes_PCG);
        ops.setoptions = Some(SUNLinSolSetOptions_PCG);
        ops.setpreconditioner = Some(SUNLinSolSetPreconditioner_PCG);
        ops.setscalingvectors = Some(SUNLinSolSetScalingVectors_PCG);
        ops.setzeroguess = Some(SUNLinSolSetZeroGuess_PCG);
        ops.initialize = Some(SUNLinSolInitialize_PCG);
        ops.setup = Some(SUNLinSolSetup_PCG);
        ops.solve = Some(SUNLinSolSolve_PCG);
        ops.numiters = Some(SUNLinSolNumIters_PCG);
        ops.resnorm = Some(SUNLinSolResNorm_PCG);
        ops.resid = Some(SUNLinSolResid_PCG);
        ops.lastflag = Some(SUNLinSolLastFlag_PCG);
        ops.space = Some(SUNLinSolSpace_PCG);
        ops.free = Some(SUNLinSolFree_PCG);
    }

    let r = N_VClone(y)?;
    let p = N_VClone(y)?;
    let z = N_VClone(y)?;
    let Ap = N_VClone(y)?;
    *S.content.borrow_mut() = Box::new(SUNLinearSolverContent_PCG_ {
        last_flag: 0,
        maxl,
        pretype,
        zeroguess: SUNFALSE,
        numiters: 0,
        resnorm: ZERO,
        r: Some(r),
        p: Some(p),
        z: Some(z),
        Ap: Some(Ap),
        s: None,
        ATimes: None,
        ATData: None,
        Psetup: None,
        Psolve: None,
        PData: None,
    });

    Some(S)
}

pub fn SUNLinSolSetOptions_PCG(
    S: &SUNLinearSolver,
    LSid: Option<&str>,
    file_name: Option<&str>,
    argv: &[String],
) -> SUNErrCode {
    if !(file_name.is_none() || file_name == Some("")) {
        return crate::sundials_errors::SUN_ERR_ARG_INCOMPATIBLE;
    }

    if !argv.is_empty() {
        let default_id = "sunlinearsolver";
        let id = match LSid {
            Some(s) if !s.is_empty() => s,
            _ => default_id,
        };
        let prefix = format!("{id}.");

        let mut idx = 1;
        while idx < argv.len() {
            if !argv[idx].starts_with(&prefix) {
                idx += 1;
                continue;
            }
            let key = &argv[idx][prefix.len()..];
            match key {
                "prec_type" => {
                    idx += 1;
                    let iarg: i32 = crate::sundials_utils::atoi(&argv[idx]);
                    let retval = SUNLinSol_PCGSetPrecType(S, iarg);
                    if retval != SUN_SUCCESS {
                        return retval;
                    }
                }
                "maxl" => {
                    idx += 1;
                    let iarg: i32 = crate::sundials_utils::atoi(&argv[idx]);
                    let retval = SUNLinSol_PCGSetMaxl(S, iarg);
                    if retval != SUN_SUCCESS {
                        return retval;
                    }
                }
                _ => {}
            }
            idx += 1;
        }
    }
    SUN_SUCCESS
}

pub fn SUNLinSol_PCGSetPrecType(S: &SUNLinearSolver, pretype: i32) -> SUNErrCode {
    content_mut(S).pretype = pretype;
    SUN_SUCCESS
}

pub fn SUNLinSol_PCGSetMaxl(S: &SUNLinearSolver, maxl: i32) -> SUNErrCode {
    /* Check for legal number of iters */
    let maxl = if maxl <= 0 { SUNPCG_MAXL_DEFAULT } else { maxl };
    content_mut(S).maxl = maxl;
    SUN_SUCCESS
}

pub fn SUNLinSolGetType_PCG(_S: &SUNLinearSolver) -> SUNLinearSolver_Type {
    SUNLINEARSOLVER_ITERATIVE
}

pub fn SUNLinSolGetID_PCG(_S: &SUNLinearSolver) -> SUNLinearSolver_ID {
    SUNLINEARSOLVER_PCG
}

pub fn SUNLinSolInitialize_PCG(S: &SUNLinearSolver) -> SUNErrCode {
    let mut content = content_mut(S);
    if content.maxl <= 0 {
        content.maxl = SUNPCG_MAXL_DEFAULT;
    }

    if content.ATimes.is_none() {
        return SUN_ERR_ARG_CORRUPT;
    }

    if content.pretype != SUN_PREC_LEFT
        && content.pretype != SUN_PREC_RIGHT
        && content.pretype != SUN_PREC_BOTH
    {
        content.pretype = SUN_PREC_NONE;
    }

    if content.pretype != SUN_PREC_NONE && content.Psolve.is_none() {
        return SUN_ERR_ARG_CORRUPT;
    }

    /* no additional memory to allocate */
    SUN_SUCCESS
}

pub fn SUNLinSolSetATimes_PCG(
    S: &SUNLinearSolver,
    ATData: Option<Box<dyn Any>>,
    ATimes: Option<SUNATimesFn>,
) -> SUNErrCode {
    let mut content = content_mut(S);
    content.ATimes = ATimes;
    content.ATData = ATData;
    SUN_SUCCESS
}

pub fn SUNLinSolSetPreconditioner_PCG(
    S: &SUNLinearSolver,
    PData: Option<Box<dyn Any>>,
    Psetup: Option<SUNPSetupFn>,
    Psolve: Option<SUNPSolveFn>,
) -> SUNErrCode {
    let mut content = content_mut(S);
    content.Psetup = Psetup;
    content.Psolve = Psolve;
    content.PData = PData;
    SUN_SUCCESS
}

pub fn SUNLinSolSetScalingVectors_PCG(
    S: &SUNLinearSolver,
    s: Option<&N_Vector>,
    _nul: Option<&N_Vector>,
) -> SUNErrCode {
    /* only use the first scaling vector */
    content_mut(S).s = s.cloned();
    SUN_SUCCESS
}

pub fn SUNLinSolSetZeroGuess_PCG(S: &SUNLinearSolver, onoff: sunbooleantype) -> SUNErrCode {
    content_mut(S).zeroguess = onoff;
    SUN_SUCCESS
}

pub fn SUNLinSolSetup_PCG(S: &SUNLinearSolver, _nul: Option<&SUNMatrix>) -> i32 {
    let Psetup = content_mut(S).Psetup;
    if let Some(Psetup) = Psetup {
        let mut PData = content_mut(S).PData.take();
        let status = Psetup(&mut PData);
        content_mut(S).PData = PData;
        if status != 0 {
            let flag = if status < 0 {
                SUNLS_PSET_FAIL_UNREC
            } else {
                SUNLS_PSET_FAIL_REC
            };
            content_mut(S).last_flag = flag as sunindextype;
            return flag;
        }
    }

    content_mut(S).last_flag = SUN_SUCCESS as sunindextype;
    SUN_SUCCESS
}

pub fn SUNLinSolSolve_PCG(
    S: &SUNLinearSolver,
    _nul: Option<&SUNMatrix>,
    x: &N_Vector,
    b: &N_Vector,
    delta: sunrealtype,
) -> i32 {
    /* Move solver state into locals (restored at the end) */
    let (l_max, r, p, z, Ap, w, mut A_data, mut P_data, atimes, psolve, pretype, mut zeroguess);
    {
        let mut content = content_mut(S);
        l_max = content.maxl;
        r = content.r.as_ref().expect("r").clone();
        p = content.p.as_ref().expect("p").clone();
        z = content.z.as_ref().expect("z").clone();
        Ap = content.Ap.as_ref().expect("Ap").clone();
        w = content.s.clone();
        A_data = content.ATData.take();
        P_data = content.PData.take();
        atimes = content.ATimes.expect("ATimes set");
        psolve = content.Psolve;
        pretype = content.pretype;
        zeroguess = content.zeroguess;
        content.numiters = 0;
    }

    let mut nli: i32 = 0;
    let mut res_norm: sunrealtype = ZERO;

    /* set flags for internal solver options */
    let UsePrec =
        pretype == SUN_PREC_BOTH || pretype == SUN_PREC_LEFT || pretype == SUN_PREC_RIGHT;
    let UseScaling = w.is_some();

    let flag = (|| -> i32 {
        let mut converged = SUNFALSE;
        let mut status;

        /* Set r to initial residual r_0 = b - A*x_0 */
        if zeroguess {
            N_VScale(ONE, b, &r);
        } else {
            status = atimes(&mut A_data, x, &r);
            if status != 0 {
                zeroguess = SUNFALSE;
                return if status < 0 {
                    SUNLS_ATIMES_FAIL_UNREC
                } else {
                    SUNLS_ATIMES_FAIL_REC
                };
            }
            N_VLinearSum(ONE, b, -ONE, &r, &r);
        }

        /* Set rho to scaled L2 norm of r, and return if small */
        if UseScaling {
            N_VProd(&r, w.as_ref().expect("w"), &Ap);
        } else {
            N_VScale(ONE, &r, &Ap);
        }
        let mut rho = N_VDotProd(&Ap, &Ap);
        rho = SUNRsqrt(rho);
        let r0_norm = rho;
        res_norm = rho;

        if rho <= delta {
            zeroguess = SUNFALSE;
            return SUN_SUCCESS;
        }

        /* Apply preconditioner and b-scaling to r = r_0 */
        if UsePrec {
            status = (psolve.expect("psolve"))(&mut P_data, &r, &z, delta, SUN_PREC_LEFT);
            if status != 0 {
                zeroguess = SUNFALSE;
                return if status < 0 {
                    SUNLS_PSOLVE_FAIL_UNREC
                } else {
                    SUNLS_PSOLVE_FAIL_REC
                };
            }
        } else {
            N_VScale(ONE, &r, &z);
        }

        /* Initialize rz to <r,z> */
        let mut rz = N_VDotProd(&r, &z);

        /* Copy z to p */
        N_VScale(ONE, &z, &p);

        /* Begin main iteration loop */
        for l in 0..l_max {
            /* increment counter */
            nli += 1;

            /* Generate Ap = A*p */
            status = atimes(&mut A_data, &p, &Ap);
            if status != 0 {
                zeroguess = SUNFALSE;
                return if status < 0 {
                    SUNLS_ATIMES_FAIL_UNREC
                } else {
                    SUNLS_ATIMES_FAIL_REC
                };
            }

            /* Calculate alpha = <r,z> / <Ap,p> */
            let mut alpha = N_VDotProd(&Ap, &p);
            alpha = rz / alpha;

            /* Update x = x + alpha*p */
            if l == 0 && zeroguess {
                N_VScale(alpha, &p, x);
            } else {
                N_VLinearSum(ONE, x, alpha, &p, x);
            }

            /* Update r = r - alpha*Ap */
            N_VLinearSum(ONE, &r, -alpha, &Ap, &r);

            /* Set rho and check convergence */
            if UseScaling {
                N_VProd(&r, w.as_ref().expect("w"), &Ap);
            } else {
                N_VScale(ONE, &r, &Ap);
            }
            rho = N_VDotProd(&Ap, &Ap);
            rho = SUNRsqrt(rho);
            res_norm = rho;

            if rho <= delta {
                converged = SUNTRUE;
                break;
            }

            /* Exit early on last iteration */
            if l == l_max - 1 {
                break;
            }

            /* Apply preconditioner: z = P^{-1}*r */
            if UsePrec {
                status = (psolve.expect("psolve"))(&mut P_data, &r, &z, delta, SUN_PREC_LEFT);
                if status != 0 {
                    zeroguess = SUNFALSE;
                    return if status < 0 {
                        SUNLS_PSOLVE_FAIL_UNREC
                    } else {
                        SUNLS_PSOLVE_FAIL_REC
                    };
                }
            } else {
                N_VScale(ONE, &r, &z);
            }

            /* update rz */
            let rz_old = rz;
            rz = N_VDotProd(&r, &z);

            /* Calculate beta = <r,z> / <r_old,z_old> */
            let beta = rz / rz_old;

            /* Update p = z + beta*p */
            N_VLinearSum(ONE, &z, beta, &p, &p);
        }

        /* Main loop finished, return with result */
        zeroguess = SUNFALSE;
        if converged == SUNTRUE {
            SUN_SUCCESS
        } else if rho < r0_norm {
            SUNLS_RES_REDUCED
        } else {
            SUNLS_CONV_FAIL
        }
    })();

    /* restore solver state and write results back */
    {
        let mut content = content_mut(S);
        content.ATData = A_data;
        content.PData = P_data;
        content.zeroguess = zeroguess;
        content.numiters = nli;
        content.resnorm = res_norm;
        content.last_flag = flag as sunindextype;
    }

    flag
}

pub fn SUNLinSolNumIters_PCG(S: &SUNLinearSolver) -> i32 {
    content_mut(S).numiters
}

pub fn SUNLinSolResNorm_PCG(S: &SUNLinearSolver) -> sunrealtype {
    content_mut(S).resnorm
}

pub fn SUNLinSolResid_PCG(S: &SUNLinearSolver) -> Option<N_Vector> {
    content_mut(S).r.clone()
}

pub fn SUNLinSolLastFlag_PCG(S: &SUNLinearSolver) -> sunindextype {
    content_mut(S).last_flag
}

pub fn SUNLinSolSpace_PCG(S: &SUNLinearSolver, lenrwLS: &mut i64, leniwLS: &mut i64) -> SUNErrCode {
    let r = content_mut(S).r.as_ref().expect("r").clone();
    let (mut lrw1, mut liw1) = (0i64, 0i64);
    N_VSpace(&r, &mut lrw1, &mut liw1);
    *lenrwLS = 1 + lrw1 * 4;
    *leniwLS = 4 + liw1 * 4;
    SUN_SUCCESS
}

pub fn SUNLinSolFree_PCG(_S: &SUNLinearSolver) -> SUNErrCode {
    SUN_SUCCESS
}
