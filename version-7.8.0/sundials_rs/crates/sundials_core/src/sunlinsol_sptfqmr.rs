//! Port of `src/sunlinsol/sptfqmr/sunlinsol_sptfqmr.c` +
//! `include/sunlinsol/sunlinsol_sptfqmr.h` (scaled preconditioned
//! transpose-free QMR). Same take/restore solve discipline as SPGMR.

use std::any::Any;
use std::cell::RefMut;

use crate::sundials_context::SUNContext;
use crate::sundials_errors::{SUN_ERR_ARG_CORRUPT, SUN_ERR_ARG_INCOMPATIBLE, SUN_SUCCESS};
use crate::sundials_iterative::*;
use crate::sundials_linearsolver::*;
use crate::sundials_math::{SUNRsqrt, SUNSQR};
use crate::sundials_matrix::SUNMatrix;
use crate::sundials_nvector::*;
use crate::sundials_types::*;

const ZERO: sunrealtype = 0.0;
const ONE: sunrealtype = 1.0;

pub const SUNSPTFQMR_MAXL_DEFAULT: i32 = 5;

pub struct SUNLinearSolverContent_SPTFQMR_ {
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
    pub r_star: Option<N_Vector>,
    pub q: Option<N_Vector>,
    pub d: Option<N_Vector>,
    pub v: Option<N_Vector>,
    pub p: Option<N_Vector>,
    pub r: Option<Vec<N_Vector>>,
    pub u: Option<N_Vector>,
    pub vtemp1: Option<N_Vector>,
    pub vtemp2: Option<N_Vector>,
    pub vtemp3: Option<N_Vector>,
}

pub type SUNLinearSolverContent_SPTFQMR = SUNLinearSolverContent_SPTFQMR_;

fn content_mut(S: &SUNLinearSolver) -> RefMut<'_, SUNLinearSolverContent_SPTFQMR_> {
    RefMut::map(S.content.borrow_mut(), |c| {
        c.downcast_mut::<SUNLinearSolverContent_SPTFQMR_>()
            .expect("SPTFQMR SUNLinearSolver content")
    })
}

pub fn SUNLinSol_SPTFQMR(
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
    let maxl = if maxl <= 0 { SUNSPTFQMR_MAXL_DEFAULT } else { maxl };

    /* check that the supplied N_Vector supports all requisite operations */
    {
        let ops = y.ops.borrow();
        if ops.nvclone.is_none()
            || ops.nvlinearsum.is_none()
            || ops.nvconst.is_none()
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
        ops.gettype = Some(SUNLinSolGetType_SPTFQMR);
        ops.getid = Some(SUNLinSolGetID_SPTFQMR);
        ops.setatimes = Some(SUNLinSolSetATimes_SPTFQMR);
        ops.setoptions = Some(SUNLinSolSetOptions_SPTFQMR);
        ops.setpreconditioner = Some(SUNLinSolSetPreconditioner_SPTFQMR);
        ops.setscalingvectors = Some(SUNLinSolSetScalingVectors_SPTFQMR);
        ops.setzeroguess = Some(SUNLinSolSetZeroGuess_SPTFQMR);
        ops.initialize = Some(SUNLinSolInitialize_SPTFQMR);
        ops.setup = Some(SUNLinSolSetup_SPTFQMR);
        ops.solve = Some(SUNLinSolSolve_SPTFQMR);
        ops.numiters = Some(SUNLinSolNumIters_SPTFQMR);
        ops.resnorm = Some(SUNLinSolResNorm_SPTFQMR);
        ops.resid = Some(SUNLinSolResid_SPTFQMR);
        ops.lastflag = Some(SUNLinSolLastFlag_SPTFQMR);
        ops.space = Some(SUNLinSolSpace_SPTFQMR);
        ops.free = Some(SUNLinSolFree_SPTFQMR);
    }

    let r_star = N_VClone(y)?;
    let q = N_VClone(y)?;
    let d = N_VClone(y)?;
    let v = N_VClone(y)?;
    let p = N_VClone(y)?;
    let r = N_VCloneVectorArray(2, y)?;
    let u = N_VClone(y)?;
    let vtemp1 = N_VClone(y)?;
    let vtemp2 = N_VClone(y)?;
    let vtemp3 = N_VClone(y)?;
    *S.content.borrow_mut() = Box::new(SUNLinearSolverContent_SPTFQMR_ {
        last_flag: 0,
        maxl,
        pretype,
        zeroguess: SUNFALSE,
        numiters: 0,
        resnorm: ZERO,
        r_star: Some(r_star),
        q: Some(q),
        d: Some(d),
        v: Some(v),
        p: Some(p),
        r: Some(r),
        u: Some(u),
        vtemp1: Some(vtemp1),
        vtemp2: Some(vtemp2),
        vtemp3: Some(vtemp3),
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

pub fn SUNLinSolSetOptions_SPTFQMR(
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
                    let retval = SUNLinSol_SPTFQMRSetPrecType(S, iarg);
                    if retval != SUN_SUCCESS {
                        return retval;
                    }
                }
                "maxl" => {
                    idx += 1;
                    let iarg: i32 = crate::sundials_utils::atoi(&argv[idx]);
                    let retval = SUNLinSol_SPTFQMRSetMaxl(S, iarg);
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

pub fn SUNLinSol_SPTFQMRSetPrecType(S: &SUNLinearSolver, pretype: i32) -> SUNErrCode {
    content_mut(S).pretype = pretype;
    SUN_SUCCESS
}

pub fn SUNLinSol_SPTFQMRSetMaxl(S: &SUNLinearSolver, maxl: i32) -> SUNErrCode {
    let maxl = if maxl <= 0 { SUNSPTFQMR_MAXL_DEFAULT } else { maxl };
    content_mut(S).maxl = maxl;
    SUN_SUCCESS
}

pub fn SUNLinSolGetType_SPTFQMR(_S: &SUNLinearSolver) -> SUNLinearSolver_Type {
    SUNLINEARSOLVER_ITERATIVE
}

pub fn SUNLinSolGetID_SPTFQMR(_S: &SUNLinearSolver) -> SUNLinearSolver_ID {
    SUNLINEARSOLVER_SPTFQMR
}

pub fn SUNLinSolInitialize_SPTFQMR(S: &SUNLinearSolver) -> SUNErrCode {
    let mut content = content_mut(S);

    /* ensure valid options */
    if content.maxl <= 0 {
        content.maxl = SUNSPTFQMR_MAXL_DEFAULT;
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

pub fn SUNLinSolSetATimes_SPTFQMR(
    S: &SUNLinearSolver,
    ATData: Option<Box<dyn Any>>,
    ATimes: Option<SUNATimesFn>,
) -> SUNErrCode {
    let mut content = content_mut(S);
    content.ATimes = ATimes;
    content.ATData = ATData;
    SUN_SUCCESS
}

pub fn SUNLinSolSetPreconditioner_SPTFQMR(
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

pub fn SUNLinSolSetScalingVectors_SPTFQMR(
    S: &SUNLinearSolver,
    s1: Option<&N_Vector>,
    s2: Option<&N_Vector>,
) -> SUNErrCode {
    let mut content = content_mut(S);
    content.s1 = s1.cloned();
    content.s2 = s2.cloned();
    SUN_SUCCESS
}

pub fn SUNLinSolSetZeroGuess_SPTFQMR(S: &SUNLinearSolver, onoff: sunbooleantype) -> SUNErrCode {
    content_mut(S).zeroguess = onoff;
    SUN_SUCCESS
}

pub fn SUNLinSolSetup_SPTFQMR(S: &SUNLinearSolver, _A: Option<&SUNMatrix>) -> i32 {
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

pub fn SUNLinSolSolve_SPTFQMR(
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
        q,
        d,
        v,
        p,
        r,
        u,
        vtemp1,
        vtemp2,
        vtemp3,
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
        q = content.q.as_ref().expect("q").clone();
        d = content.d.as_ref().expect("d").clone();
        v = content.v.as_ref().expect("v").clone();
        p = content.p.as_ref().expect("p").clone();
        r = content.r.as_ref().expect("r").clone();
        u = content.u.as_ref().expect("u").clone();
        vtemp1 = content.vtemp1.as_ref().expect("vtemp1").clone();
        vtemp2 = content.vtemp2.as_ref().expect("vtemp2").clone();
        vtemp3 = content.vtemp3.as_ref().expect("vtemp3").clone();
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
        let mut b_ok = SUNFALSE;
        let mut temp_val: sunrealtype = -ONE;
        let mut r_curr_norm: sunrealtype = -ONE;
        let mut status;

        /* Check for unsupported use case */
        if preOnRight && !zeroguess {
            zeroguess = SUNFALSE;
            return SUN_ERR_ARG_INCOMPATIBLE;
        }

        /* Set r_star to initial (unscaled) residual r_star = r_0 = b - A*x_0 */
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

        /* Apply left preconditioner and b-scaling to r_star */
        if preOnLeft {
            status = (psolve.expect("psolve"))(&mut P_data, &r_star, &vtemp1, delta, SUN_PREC_LEFT);
            if status != 0 {
                zeroguess = SUNFALSE;
                return if status < 0 {
                    SUNLS_PSOLVE_FAIL_UNREC
                } else {
                    SUNLS_PSOLVE_FAIL_REC
                };
            }
        } else {
            N_VScale(ONE, &r_star, &vtemp1);
        }

        if scale_b {
            N_VProd(sb.as_ref().expect("sb"), &vtemp1, &r_star);
        } else {
            N_VScale(ONE, &vtemp1, &r_star);
        }

        /* Initialize rho[0] */
        let mut rho0 = N_VDotProd(&r_star, &r_star);

        /* Compute norm of initial residual (r_0) */
        let r_init_norm = SUNRsqrt(rho0);
        res_norm = r_init_norm;

        if r_init_norm <= delta {
            zeroguess = SUNFALSE;
            return SUN_SUCCESS;
        }

        /* Set v = A*r_0 (preconditioned and scaled) */
        if scale_x {
            N_VDiv(&r_star, sx.as_ref().expect("sx"), &vtemp1);
        } else {
            N_VScale(ONE, &r_star, &vtemp1);
        }

        if preOnRight {
            N_VScale(ONE, &vtemp1, &v);
            status = (psolve.expect("psolve"))(&mut P_data, &v, &vtemp1, delta, SUN_PREC_RIGHT);
            if status != 0 {
                zeroguess = SUNFALSE;
                return if status < 0 {
                    SUNLS_PSOLVE_FAIL_UNREC
                } else {
                    SUNLS_PSOLVE_FAIL_REC
                };
            }
        }

        status = atimes(&mut A_data, &vtemp1, &v);
        if status != 0 {
            zeroguess = SUNFALSE;
            return if status < 0 {
                SUNLS_ATIMES_FAIL_UNREC
            } else {
                SUNLS_ATIMES_FAIL_REC
            };
        }

        if preOnLeft {
            status = (psolve.expect("psolve"))(&mut P_data, &v, &vtemp1, delta, SUN_PREC_LEFT);
            if status != 0 {
                zeroguess = SUNFALSE;
                return if status < 0 {
                    SUNLS_PSOLVE_FAIL_UNREC
                } else {
                    SUNLS_PSOLVE_FAIL_REC
                };
            }
        } else {
            N_VScale(ONE, &v, &vtemp1);
        }

        if scale_b {
            N_VProd(sb.as_ref().expect("sb"), &vtemp1, &v);
        } else {
            N_VScale(ONE, &vtemp1, &v);
        }

        /* Initialize remaining variables */
        N_VScale(ONE, &r_star, &r[0]);
        N_VScale(ONE, &r_star, &u);
        N_VScale(ONE, &r_star, &p);
        N_VConst(ZERO, &d);

        /* Set x = sx x if non-zero guess */
        if scale_x && !zeroguess {
            N_VProd(sx.as_ref().expect("sx"), x, x);
        }

        let mut tau = r_init_norm;
        let mut v_bar: sunrealtype = ZERO;
        let mut eta: sunrealtype = ZERO;

        /* START outer loop */
        for n in 0..l_max {
            /* Increment linear iteration counter */
            nli += 1;

            /* sigma = r_star^T*v */
            let sigma = N_VDotProd(&r_star, &v);

            /* alpha = rho[0]/sigma */
            let alpha = rho0 / sigma;

            /* q = u-alpha*v */
            N_VLinearSum(ONE, &u, -alpha, &v, &q);

            /* r[1] = r[0]-alpha*A*(u+q) */
            N_VLinearSum(ONE, &u, ONE, &q, &r[1]);
            if scale_x {
                N_VDiv(&r[1], sx.as_ref().expect("sx"), &r[1]);
            }

            if preOnRight {
                N_VScale(ONE, &r[1], &vtemp1);
                status = (psolve.expect("psolve"))(&mut P_data, &vtemp1, &r[1], delta, SUN_PREC_RIGHT);
                if status != 0 {
                    zeroguess = SUNFALSE;
                    return if status < 0 {
                        SUNLS_PSOLVE_FAIL_UNREC
                    } else {
                        SUNLS_PSOLVE_FAIL_REC
                    };
                }
            }

            status = atimes(&mut A_data, &r[1], &vtemp1);
            if status != 0 {
                zeroguess = SUNFALSE;
                return if status < 0 {
                    SUNLS_ATIMES_FAIL_UNREC
                } else {
                    SUNLS_ATIMES_FAIL_REC
                };
            }

            if preOnLeft {
                status = (psolve.expect("psolve"))(&mut P_data, &vtemp1, &r[1], delta, SUN_PREC_LEFT);
                if status != 0 {
                    zeroguess = SUNFALSE;
                    return if status < 0 {
                        SUNLS_PSOLVE_FAIL_UNREC
                    } else {
                        SUNLS_PSOLVE_FAIL_REC
                    };
                }
            } else {
                N_VScale(ONE, &vtemp1, &r[1]);
            }

            if scale_b {
                N_VProd(sb.as_ref().expect("sb"), &r[1], &vtemp1);
            } else {
                N_VScale(ONE, &r[1], &vtemp1);
            }
            N_VLinearSum(ONE, &r[0], -alpha, &vtemp1, &r[1]);

            /* START inner loop */
            for m in 0..2 {
                /* d = [*]+(v_bar^2*eta/alpha)*d */
                let omega;
                if m == 0 {
                    temp_val = N_VDotProd(&r[1], &r[1]);
                    temp_val = SUNRsqrt(temp_val);
                    let mut om = N_VDotProd(&r[0], &r[0]);
                    om = SUNRsqrt(SUNRsqrt(om) * temp_val);
                    omega = om;
                    N_VLinearSum(ONE, &u, SUNSQR(v_bar) * eta / alpha, &d, &d);
                } else {
                    omega = temp_val;
                    N_VLinearSum(ONE, &q, SUNSQR(v_bar) * eta / alpha, &d, &d);
                }

                /* v_bar = omega/tau */
                v_bar = omega / tau;

                /* c = (1+v_bar^2)^(-1/2) */
                let c = ONE / SUNRsqrt(ONE + SUNSQR(v_bar));

                /* tau = tau*v_bar*c */
                tau = tau * v_bar * c;

                /* eta = c^2*alpha */
                eta = SUNSQR(c) * alpha;

                /* x = x+eta*d */
                if n == 0 && m == 0 && zeroguess {
                    N_VScale(eta, &d, x);
                } else {
                    N_VLinearSum(ONE, x, eta, &d, x);
                }

                /* Check for convergence (approximation to norm of residual) */
                r_curr_norm = tau * SUNRsqrt((m + 1) as sunrealtype);
                res_norm = r_curr_norm;

                if r_curr_norm <= delta {
                    converged = SUNTRUE;
                    break;
                }

                /* Decide if actual norm of residual vector should be computed */
                if (r_curr_norm > delta)
                    || (r_curr_norm >= r_init_norm && m == 1 && n == l_max)
                {
                    /* Compute norm of residual ||b-A*x||_2 (prec. and scaled) */
                    if scale_x {
                        N_VDiv(x, sx.as_ref().expect("sx"), &vtemp1);
                    } else {
                        N_VScale(ONE, x, &vtemp1);
                    }

                    if preOnRight {
                        status = (psolve.expect("psolve"))(
                            &mut P_data,
                            &vtemp1,
                            &vtemp2,
                            delta,
                            SUN_PREC_RIGHT,
                        );
                        if status != 0 {
                            zeroguess = SUNFALSE;
                            return if status < 0 {
                                SUNLS_PSOLVE_FAIL_UNREC
                            } else {
                                SUNLS_PSOLVE_FAIL_REC
                            };
                        }
                        N_VScale(ONE, &vtemp2, &vtemp1);
                    }

                    status = atimes(&mut A_data, &vtemp1, &vtemp2);
                    if status != 0 {
                        zeroguess = SUNFALSE;
                        return if status < 0 {
                            SUNLS_ATIMES_FAIL_UNREC
                        } else {
                            SUNLS_ATIMES_FAIL_REC
                        };
                    }

                    if preOnLeft {
                        status = (psolve.expect("psolve"))(
                            &mut P_data,
                            &vtemp2,
                            &vtemp1,
                            delta,
                            SUN_PREC_LEFT,
                        );
                        if status != 0 {
                            zeroguess = SUNFALSE;
                            return if status < 0 {
                                SUNLS_PSOLVE_FAIL_UNREC
                            } else {
                                SUNLS_PSOLVE_FAIL_REC
                            };
                        }
                    } else {
                        N_VScale(ONE, &vtemp2, &vtemp1);
                    }

                    if scale_b {
                        N_VProd(sb.as_ref().expect("sb"), &vtemp1, &vtemp2);
                    } else {
                        N_VScale(ONE, &vtemp1, &vtemp2);
                    }

                    /* Only precondition and scale b once (result reused) */
                    if !b_ok {
                        b_ok = SUNTRUE;
                        if preOnLeft {
                            status = (psolve.expect("psolve"))(
                                &mut P_data,
                                b,
                                &vtemp3,
                                delta,
                                SUN_PREC_LEFT,
                            );
                            if status != 0 {
                                zeroguess = SUNFALSE;
                                return if status < 0 {
                                    SUNLS_PSOLVE_FAIL_UNREC
                                } else {
                                    SUNLS_PSOLVE_FAIL_REC
                                };
                            }
                        } else {
                            N_VScale(ONE, b, &vtemp3);
                        }

                        if scale_b {
                            N_VProd(sb.as_ref().expect("sb"), &vtemp3, &vtemp3);
                        }
                    }
                    N_VLinearSum(ONE, &vtemp3, -ONE, &vtemp2, &vtemp1);
                    r_curr_norm = N_VDotProd(&vtemp1, &vtemp1);
                    r_curr_norm = SUNRsqrt(r_curr_norm);
                    res_norm = r_curr_norm;

                    /* Exit inner loop if we have converged */
                    if r_curr_norm <= delta {
                        converged = SUNTRUE;
                        break;
                    }
                }
            } /* END inner loop */

            /* If converged, then exit outer loop as well */
            if converged == SUNTRUE {
                break;
            }

            /* rho[1] = r_star^T*r_[1] */
            let rho1 = N_VDotProd(&r_star, &r[1]);

            /* beta = rho[1]/rho[0] */
            let beta = rho1 / rho0;

            /* u = r[1]+beta*q */
            N_VLinearSum(ONE, &r[1], beta, &q, &u);

            /* p = u+beta*(q+beta*p) */
            let cv = [SUNSQR(beta), beta, ONE];
            let Xv = [p.clone(), q.clone(), u.clone()];
            let ier = N_VLinearCombination(3, &cv, &Xv, &p);
            if ier != SUN_SUCCESS {
                return ier;
            }

            /* v = A*p */
            if scale_x {
                N_VDiv(&p, sx.as_ref().expect("sx"), &vtemp1);
            } else {
                N_VScale(ONE, &p, &vtemp1);
            }

            if preOnRight {
                N_VScale(ONE, &vtemp1, &v);
                status = (psolve.expect("psolve"))(&mut P_data, &v, &vtemp1, delta, SUN_PREC_RIGHT);
                if status != 0 {
                    zeroguess = SUNFALSE;
                    return if status < 0 {
                        SUNLS_PSOLVE_FAIL_UNREC
                    } else {
                        SUNLS_PSOLVE_FAIL_REC
                    };
                }
            }

            status = atimes(&mut A_data, &vtemp1, &v);
            if status != 0 {
                zeroguess = SUNFALSE;
                return if status < 0 {
                    SUNLS_ATIMES_FAIL_UNREC
                } else {
                    SUNLS_ATIMES_FAIL_REC
                };
            }

            if preOnLeft {
                status = (psolve.expect("psolve"))(&mut P_data, &v, &vtemp1, delta, SUN_PREC_LEFT);
                if status != 0 {
                    zeroguess = SUNFALSE;
                    return if status < 0 {
                        SUNLS_PSOLVE_FAIL_UNREC
                    } else {
                        SUNLS_PSOLVE_FAIL_REC
                    };
                }
            } else {
                N_VScale(ONE, &v, &vtemp1);
            }

            if scale_b {
                N_VProd(sb.as_ref().expect("sb"), &vtemp1, &v);
            } else {
                N_VScale(ONE, &vtemp1, &v);
            }

            /* Shift variable values */
            N_VScale(ONE, &r[1], &r[0]);
            rho0 = rho1;
        } /* END outer loop */

        /* Determine return value */
        if converged == SUNTRUE || r_curr_norm < r_init_norm {
            if scale_x {
                N_VDiv(x, sx.as_ref().expect("sx"), x);
            }

            if preOnRight {
                status = (psolve.expect("psolve"))(&mut P_data, x, &vtemp1, delta, SUN_PREC_RIGHT);
                if status != 0 {
                    zeroguess = SUNFALSE;
                    return if status < 0 {
                        SUNLS_PSOLVE_FAIL_UNREC
                    } else {
                        SUNLS_PSOLVE_FAIL_REC
                    };
                }
                N_VScale(ONE, &vtemp1, x);
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

pub fn SUNLinSolNumIters_SPTFQMR(S: &SUNLinearSolver) -> i32 {
    content_mut(S).numiters
}

pub fn SUNLinSolResNorm_SPTFQMR(S: &SUNLinearSolver) -> sunrealtype {
    content_mut(S).resnorm
}

pub fn SUNLinSolResid_SPTFQMR(S: &SUNLinearSolver) -> Option<N_Vector> {
    content_mut(S).vtemp1.clone()
}

pub fn SUNLinSolLastFlag_SPTFQMR(S: &SUNLinearSolver) -> sunindextype {
    content_mut(S).last_flag
}

pub fn SUNLinSolSpace_SPTFQMR(
    S: &SUNLinearSolver,
    lenrwLS: &mut i64,
    leniwLS: &mut i64,
) -> SUNErrCode {
    let vtemp1 = content_mut(S).vtemp1.as_ref().expect("vtemp1").clone();
    let (mut lrw1, mut liw1) = (0i64, 0i64);
    if vtemp1.ops.borrow().nvspace.is_some() {
        N_VSpace(&vtemp1, &mut lrw1, &mut liw1);
    }
    *lenrwLS = lrw1 * 11;
    *leniwLS = liw1 * 11;
    SUN_SUCCESS
}

pub fn SUNLinSolFree_SPTFQMR(_S: &SUNLinearSolver) -> SUNErrCode {
    SUN_SUCCESS
}
