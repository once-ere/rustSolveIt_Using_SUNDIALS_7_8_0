//! Port of `src/sunlinsol/spbcgs/sunlinsol_spbcgs.c` +
//! `include/sunlinsol/sunlinsol_spbcgs.h` (scaled preconditioned
//! BiCG-Stab). Same take/restore solve discipline as the SPGMR port.

use std::any::Any;
use std::cell::RefMut;

use crate::sundials_context::SUNContext;
use crate::sundials_errors::{SUN_ERR_ARG_CORRUPT, SUN_ERR_ARG_INCOMPATIBLE, SUN_SUCCESS};
use crate::sundials_iterative::*;
use crate::sundials_linearsolver::*;
use crate::sundials_math::SUNRsqrt;
use crate::sundials_matrix::SUNMatrix;
use crate::sundials_nvector::*;
use crate::sundials_types::*;

const ZERO: sunrealtype = 0.0;
const ONE: sunrealtype = 1.0;

pub const SUNSPBCGS_MAXL_DEFAULT: i32 = 5;

pub struct SUNLinearSolverContent_SPBCGS_ {
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

    pub s1: Option<N_Vector>,
    pub s2: Option<N_Vector>,
    pub r: Option<N_Vector>,
    pub r_star: Option<N_Vector>,
    pub p: Option<N_Vector>,
    pub q: Option<N_Vector>,
    pub u: Option<N_Vector>,
    pub Ap: Option<N_Vector>,
    pub vtemp: Option<N_Vector>,
}

pub type SUNLinearSolverContent_SPBCGS = SUNLinearSolverContent_SPBCGS_;

fn content_mut(S: &SUNLinearSolver) -> RefMut<'_, SUNLinearSolverContent_SPBCGS_> {
    RefMut::map(S.content.borrow_mut(), |c| {
        c.downcast_mut::<SUNLinearSolverContent_SPBCGS_>()
            .expect("SPBCGS SUNLinearSolver content")
    })
}

pub fn SUNLinSol_SPBCGS(
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
    let maxl = if maxl <= 0 { SUNSPBCGS_MAXL_DEFAULT } else { maxl };

    /* check that the supplied N_Vector supports all requisite operations */
    {
        let ops = y.ops.borrow();
        if ops.nvclone.is_none()
            || ops.nvlinearsum.is_none()
            || ops.nvprod.is_none()
            || ops.nvdiv.is_none()
            || ops.nvscale.is_none()
            || ops.nvdotprod.is_none()
        {
            return None;
        }
    }

    let S = SUNLinSolNewEmpty(sunctx)?;

    {
        let mut ops = S.ops.borrow_mut();
        ops.gettype = Some(SUNLinSolGetType_SPBCGS);
        ops.getid = Some(SUNLinSolGetID_SPBCGS);
        ops.setatimes = Some(SUNLinSolSetATimes_SPBCGS);
        ops.setoptions = Some(SUNLinSolSetOptions_SPBCGS);
        ops.setpreconditioner = Some(SUNLinSolSetPreconditioner_SPBCGS);
        ops.setscalingvectors = Some(SUNLinSolSetScalingVectors_SPBCGS);
        ops.setzeroguess = Some(SUNLinSolSetZeroGuess_SPBCGS);
        ops.initialize = Some(SUNLinSolInitialize_SPBCGS);
        ops.setup = Some(SUNLinSolSetup_SPBCGS);
        ops.solve = Some(SUNLinSolSolve_SPBCGS);
        ops.numiters = Some(SUNLinSolNumIters_SPBCGS);
        ops.resnorm = Some(SUNLinSolResNorm_SPBCGS);
        ops.resid = Some(SUNLinSolResid_SPBCGS);
        ops.lastflag = Some(SUNLinSolLastFlag_SPBCGS);
        ops.space = Some(SUNLinSolSpace_SPBCGS);
        ops.free = Some(SUNLinSolFree_SPBCGS);
    }

    let r_star = N_VClone(y)?;
    let r = N_VClone(y)?;
    let p = N_VClone(y)?;
    let q = N_VClone(y)?;
    let u = N_VClone(y)?;
    let Ap = N_VClone(y)?;
    let vtemp = N_VClone(y)?;
    *S.content.borrow_mut() = Box::new(SUNLinearSolverContent_SPBCGS_ {
        last_flag: 0,
        maxl,
        pretype,
        zeroguess: SUNFALSE,
        numiters: 0,
        resnorm: ZERO,
        r_star: Some(r_star),
        r: Some(r),
        p: Some(p),
        q: Some(q),
        u: Some(u),
        Ap: Some(Ap),
        vtemp: Some(vtemp),
        s1: None,
        s2: None,
        ATimes: None,
        ATData: None,
        Psetup: None,
        Psolve: None,
        PData: None,
    });

    Some(S)
}

pub fn SUNLinSolSetOptions_SPBCGS(
    S: &SUNLinearSolver,
    LSid: Option<&str>,
    file_name: Option<&str>,
    argv: &[String],
) -> SUNErrCode {
    if !(file_name.is_none() || file_name == Some("")) {
        return SUN_ERR_ARG_INCOMPATIBLE;
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
                    let retval = SUNLinSol_SPBCGSSetPrecType(S, iarg);
                    if retval != SUN_SUCCESS {
                        return retval;
                    }
                }
                "maxl" => {
                    idx += 1;
                    let iarg: i32 = crate::sundials_utils::atoi(&argv[idx]);
                    let retval = SUNLinSol_SPBCGSSetMaxl(S, iarg);
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

pub fn SUNLinSol_SPBCGSSetPrecType(S: &SUNLinearSolver, pretype: i32) -> SUNErrCode {
    content_mut(S).pretype = pretype;
    SUN_SUCCESS
}

pub fn SUNLinSol_SPBCGSSetMaxl(S: &SUNLinearSolver, maxl: i32) -> SUNErrCode {
    let maxl = if maxl <= 0 { SUNSPBCGS_MAXL_DEFAULT } else { maxl };
    content_mut(S).maxl = maxl;
    SUN_SUCCESS
}

pub fn SUNLinSolGetType_SPBCGS(_S: &SUNLinearSolver) -> SUNLinearSolver_Type {
    SUNLINEARSOLVER_ITERATIVE
}

pub fn SUNLinSolGetID_SPBCGS(_S: &SUNLinearSolver) -> SUNLinearSolver_ID {
    SUNLINEARSOLVER_SPBCGS
}

pub fn SUNLinSolInitialize_SPBCGS(S: &SUNLinearSolver) -> SUNErrCode {
    let mut content = content_mut(S);

    if content.maxl <= 0 {
        content.maxl = SUNSPBCGS_MAXL_DEFAULT;
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

pub fn SUNLinSolSetATimes_SPBCGS(
    S: &SUNLinearSolver,
    ATData: Option<Box<dyn Any>>,
    ATimes: Option<SUNATimesFn>,
) -> SUNErrCode {
    let mut content = content_mut(S);
    content.ATimes = ATimes;
    content.ATData = ATData;
    SUN_SUCCESS
}

pub fn SUNLinSolSetPreconditioner_SPBCGS(
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

pub fn SUNLinSolSetScalingVectors_SPBCGS(
    S: &SUNLinearSolver,
    s1: Option<&N_Vector>,
    s2: Option<&N_Vector>,
) -> SUNErrCode {
    let mut content = content_mut(S);
    content.s1 = s1.cloned();
    content.s2 = s2.cloned();
    SUN_SUCCESS
}

pub fn SUNLinSolSetZeroGuess_SPBCGS(S: &SUNLinearSolver, onoff: sunbooleantype) -> SUNErrCode {
    content_mut(S).zeroguess = onoff;
    SUN_SUCCESS
}

pub fn SUNLinSolSetup_SPBCGS(S: &SUNLinearSolver, _A: Option<&SUNMatrix>) -> i32 {
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

pub fn SUNLinSolSolve_SPBCGS(
    S: &SUNLinearSolver,
    _A: Option<&SUNMatrix>,
    x: &N_Vector,
    b: &N_Vector,
    delta: sunrealtype,
) -> i32 {
    /* Move solver state into locals (restored at the end) */
    let (
        l_max,
        r_star,
        r,
        p,
        q,
        u,
        Ap,
        vtemp,
        sb,
        sx,
        mut A_data,
        mut P_data,
        atimes,
        psolve,
        pretype,
        mut zeroguess,
    );
    {
        let mut content = content_mut(S);
        l_max = content.maxl;
        r_star = content.r_star.as_ref().expect("r_star").clone();
        r = content.r.as_ref().expect("r").clone();
        p = content.p.as_ref().expect("p").clone();
        q = content.q.as_ref().expect("q").clone();
        u = content.u.as_ref().expect("u").clone();
        Ap = content.Ap.as_ref().expect("Ap").clone();
        vtemp = content.vtemp.as_ref().expect("vtemp").clone();
        sb = content.s1.clone();
        sx = content.s2.clone();
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
    let preOnLeft = pretype == SUN_PREC_LEFT || pretype == SUN_PREC_BOTH;
    let preOnRight = pretype == SUN_PREC_RIGHT || pretype == SUN_PREC_BOTH;
    let scale_x = sx.is_some();
    let scale_b = sb.is_some();

    let flag = (|| -> i32 {
        let mut converged = SUNFALSE;
        let mut status;

        /* Check for unsupported use case */
        if preOnRight && !zeroguess {
            zeroguess = SUNFALSE;
            return SUN_ERR_ARG_INCOMPATIBLE;
        }

        /* Set r_star to initial (unscaled) residual r_0 = b - A*x_0 */
        if zeroguess {
            N_VScale(ONE, b, &r_star);
        } else {
            status = atimes(&mut A_data, x, &r_star);
            if status != 0 {
                zeroguess = SUNFALSE;
                return if status < 0 {
                    SUNLS_ATIMES_FAIL_UNREC
                } else {
                    SUNLS_ATIMES_FAIL_REC
                };
            }
            N_VLinearSum(ONE, b, -ONE, &r_star, &r_star);
        }

        /* Apply left preconditioner and b-scaling to r_star = r_0 */
        if preOnLeft {
            status = (psolve.expect("psolve"))(&mut P_data, &r_star, &r, delta, SUN_PREC_LEFT);
            if status != 0 {
                zeroguess = SUNFALSE;
                return if status < 0 {
                    SUNLS_PSOLVE_FAIL_UNREC
                } else {
                    SUNLS_PSOLVE_FAIL_REC
                };
            }
        } else {
            N_VScale(ONE, &r_star, &r);
        }

        if scale_b {
            N_VProd(sb.as_ref().expect("sb"), &r, &r_star);
        } else {
            N_VScale(ONE, &r, &r_star);
        }

        /* Initialize beta_denom to the dot product of r0 with r0 */
        let mut beta_denom = N_VDotProd(&r_star, &r_star);

        /* Set r_norm to L2 norm of r_star = sb P1_inv r_0, return if small */
        let r_norm = SUNRsqrt(beta_denom);
        let mut rho = r_norm;
        res_norm = r_norm;

        if r_norm <= delta {
            zeroguess = SUNFALSE;
            return SUN_SUCCESS;
        }

        /* Copy r_star to r and p */
        N_VScale(ONE, &r_star, &r);
        N_VScale(ONE, &r_star, &p);

        /* Set x = sx x if non-zero guess */
        if scale_x && !zeroguess {
            N_VProd(sx.as_ref().expect("sx"), x, x);
        }

        /* Begin main iteration loop */
        for l in 0..l_max {
            nli += 1;

            /* Generate Ap = A-tilde p */

            /* Apply x-scaling: vtemp = sx_inv p */
            if scale_x {
                N_VDiv(&p, sx.as_ref().expect("sx"), &vtemp);
            } else {
                N_VScale(ONE, &p, &vtemp);
            }

            /* Apply right preconditioner: vtemp = P2_inv sx_inv p */
            if preOnRight {
                N_VScale(ONE, &vtemp, &Ap);
                status = (psolve.expect("psolve"))(&mut P_data, &Ap, &vtemp, delta, SUN_PREC_RIGHT);
                if status != 0 {
                    zeroguess = SUNFALSE;
                    return if status < 0 {
                        SUNLS_PSOLVE_FAIL_UNREC
                    } else {
                        SUNLS_PSOLVE_FAIL_REC
                    };
                }
            }

            /* Apply A: Ap = A P2_inv sx_inv p */
            status = atimes(&mut A_data, &vtemp, &Ap);
            if status != 0 {
                zeroguess = SUNFALSE;
                return if status < 0 {
                    SUNLS_ATIMES_FAIL_UNREC
                } else {
                    SUNLS_ATIMES_FAIL_REC
                };
            }

            /* Apply left preconditioner: vtemp = P1_inv A P2_inv sx_inv p */
            if preOnLeft {
                status = (psolve.expect("psolve"))(&mut P_data, &Ap, &vtemp, delta, SUN_PREC_LEFT);
                if status != 0 {
                    zeroguess = SUNFALSE;
                    return if status < 0 {
                        SUNLS_PSOLVE_FAIL_UNREC
                    } else {
                        SUNLS_PSOLVE_FAIL_REC
                    };
                }
            } else {
                N_VScale(ONE, &Ap, &vtemp);
            }

            /* Apply b-scaling: Ap = sb P1_inv A P2_inv sx_inv p */
            if scale_b {
                N_VProd(sb.as_ref().expect("sb"), &vtemp, &Ap);
            } else {
                N_VScale(ONE, &vtemp, &Ap);
            }

            /* Calculate alpha = <r,r_star>/<Ap,r_star> */
            let mut alpha = N_VDotProd(&Ap, &r_star);
            alpha = beta_denom / alpha;

            /* Update q = r - alpha*Ap */
            N_VLinearSum(ONE, &r, -alpha, &Ap, &q);

            /* Generate u = A-tilde q */

            /* Apply x-scaling: vtemp = sx_inv q */
            if scale_x {
                N_VDiv(&q, sx.as_ref().expect("sx"), &vtemp);
            } else {
                N_VScale(ONE, &q, &vtemp);
            }

            /* Apply right preconditioner: vtemp = P2_inv sx_inv q */
            if preOnRight {
                N_VScale(ONE, &vtemp, &u);
                status = (psolve.expect("psolve"))(&mut P_data, &u, &vtemp, delta, SUN_PREC_RIGHT);
                if status != 0 {
                    zeroguess = SUNFALSE;
                    return if status < 0 {
                        SUNLS_PSOLVE_FAIL_UNREC
                    } else {
                        SUNLS_PSOLVE_FAIL_REC
                    };
                }
            }

            /* Apply A: u = A P2_inv sx_inv u */
            status = atimes(&mut A_data, &vtemp, &u);
            if status != 0 {
                zeroguess = SUNFALSE;
                return if status < 0 {
                    SUNLS_ATIMES_FAIL_UNREC
                } else {
                    SUNLS_ATIMES_FAIL_REC
                };
            }

            /* Apply left preconditioner: vtemp = P1_inv A P2_inv sx_inv p */
            if preOnLeft {
                status = (psolve.expect("psolve"))(&mut P_data, &u, &vtemp, delta, SUN_PREC_LEFT);
                if status != 0 {
                    zeroguess = SUNFALSE;
                    return if status < 0 {
                        SUNLS_PSOLVE_FAIL_UNREC
                    } else {
                        SUNLS_PSOLVE_FAIL_REC
                    };
                }
            } else {
                N_VScale(ONE, &u, &vtemp);
            }

            /* Apply b-scaling: u = sb P1_inv A P2_inv sx_inv u */
            if scale_b {
                N_VProd(sb.as_ref().expect("sb"), &vtemp, &u);
            } else {
                N_VScale(ONE, &vtemp, &u);
            }

            /* Calculate omega = <u,q>/<u,u> */
            let mut omega_denom = N_VDotProd(&u, &u);
            if omega_denom == ZERO {
                omega_denom = ONE;
            }
            let mut omega = N_VDotProd(&u, &q);
            omega /= omega_denom;

            /* Update x = x + alpha*p + omega*q */
            if l == 0 && zeroguess {
                N_VLinearSum(alpha, &p, omega, &q, x);
            } else {
                let cv = [ONE, alpha, omega];
                let Xv = [x.clone(), p.clone(), q.clone()];
                let ier = N_VLinearCombination(3, &cv, &Xv, x);
                if ier != SUN_SUCCESS {
                    return ier;
                }
            }

            /* Update the residual r = q - omega*u */
            N_VLinearSum(ONE, &q, -omega, &u, &r);

            /* Set rho = norm(r) and check convergence */
            rho = SUNRsqrt(N_VDotProd(&r, &r));
            res_norm = rho;

            if rho <= delta {
                converged = SUNTRUE;
                break;
            }

            /* Not yet converged, continue iteration */
            /* Update beta = <rnew,r_star> / <rold,r_star> * alpha / omega */
            let beta_num = N_VDotProd(&r, &r_star);
            let beta = (beta_num / beta_denom) * (alpha / omega);

            /* Update p = r + beta*(p - omega*Ap) */
            let cv = [beta, -alpha * (beta_num / beta_denom), ONE];
            let Xv = [p.clone(), Ap.clone(), r.clone()];
            let ier = N_VLinearCombination(3, &cv, &Xv, &p);
            if ier != SUN_SUCCESS {
                return ier;
            }

            /* update beta_denom for next iteration */
            beta_denom = beta_num;
        }

        /* Main loop finished */
        if converged == SUNTRUE || rho < r_norm {
            /* Apply the x-scaling and right preconditioner: x = P2_inv sx_inv x */
            if scale_x {
                N_VDiv(x, sx.as_ref().expect("sx"), x);
            }
            if preOnRight {
                status = (psolve.expect("psolve"))(&mut P_data, x, &vtemp, delta, SUN_PREC_RIGHT);
                if status != 0 {
                    zeroguess = SUNFALSE;
                    return if status < 0 {
                        SUNLS_PSOLVE_FAIL_UNREC
                    } else {
                        SUNLS_PSOLVE_FAIL_REC
                    };
                }
                N_VScale(ONE, &vtemp, x);
            }

            zeroguess = SUNFALSE;
            if converged == SUNTRUE {
                SUN_SUCCESS
            } else {
                SUNLS_RES_REDUCED
            }
        } else {
            zeroguess = SUNFALSE;
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

pub fn SUNLinSolNumIters_SPBCGS(S: &SUNLinearSolver) -> i32 {
    content_mut(S).numiters
}

pub fn SUNLinSolResNorm_SPBCGS(S: &SUNLinearSolver) -> sunrealtype {
    content_mut(S).resnorm
}

pub fn SUNLinSolResid_SPBCGS(S: &SUNLinearSolver) -> Option<N_Vector> {
    content_mut(S).r.clone()
}

pub fn SUNLinSolLastFlag_SPBCGS(S: &SUNLinearSolver) -> sunindextype {
    content_mut(S).last_flag
}

pub fn SUNLinSolSpace_SPBCGS(
    S: &SUNLinearSolver,
    lenrwLS: &mut i64,
    leniwLS: &mut i64,
) -> SUNErrCode {
    let vtemp = content_mut(S).vtemp.as_ref().expect("vtemp").clone();
    let (mut lrw1, mut liw1) = (0i64, 0i64);
    if vtemp.ops.borrow().nvspace.is_some() {
        N_VSpace(&vtemp, &mut lrw1, &mut liw1);
    }
    *lenrwLS = lrw1 * 9;
    *leniwLS = liw1 * 9;
    SUN_SUCCESS
}

pub fn SUNLinSolFree_SPBCGS(_S: &SUNLinearSolver) -> SUNErrCode {
    SUN_SUCCESS
}
