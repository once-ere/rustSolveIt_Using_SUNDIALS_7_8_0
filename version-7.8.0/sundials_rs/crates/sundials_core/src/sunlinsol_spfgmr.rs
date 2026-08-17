//! Port of `src/sunlinsol/spfgmr/sunlinsol_spfgmr.c` +
//! `include/sunlinsol/sunlinsol_spfgmr.h` (scaled preconditioned flexible
//! GMRES; right preconditioning only). Same take/restore solve discipline
//! as the SPGMR port.

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

pub const SUNSPFGMR_MAXL_DEFAULT: i32 = 5;
pub const SUNSPFGMR_MAXRS_DEFAULT: i32 = 0;
pub const SUNSPFGMR_GSTYPE_DEFAULT: i32 = SUN_MODIFIED_GS;

pub struct SUNLinearSolverContent_SPFGMR_ {
    pub maxl: i32,
    pub pretype: i32,
    pub gstype: i32,
    pub max_restarts: i32,
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
    pub V: Option<Vec<N_Vector>>,
    pub Z: Option<Vec<N_Vector>>,
    pub Hes: Vec<Vec<sunrealtype>>,
    pub givens: Vec<sunrealtype>,
    pub xcor: Option<N_Vector>,
    pub yg: Vec<sunrealtype>,
    pub vtemp: Option<N_Vector>,

    pub cv: Vec<sunrealtype>,
    pub Xv: Vec<N_Vector>,
}

pub type SUNLinearSolverContent_SPFGMR = SUNLinearSolverContent_SPFGMR_;

fn content_mut(S: &SUNLinearSolver) -> RefMut<'_, SUNLinearSolverContent_SPFGMR_> {
    RefMut::map(S.content.borrow_mut(), |c| {
        c.downcast_mut::<SUNLinearSolverContent_SPFGMR_>()
            .expect("SPFGMR SUNLinearSolver content")
    })
}

pub fn SUNLinSol_SPFGMR(
    y: &N_Vector,
    pretype: i32,
    maxl: i32,
    sunctx: &SUNContext,
) -> Option<SUNLinearSolver> {
    /* enabling any preconditioner implies right preconditioning */
    let pretype = if pretype == SUN_PREC_LEFT || pretype == SUN_PREC_RIGHT || pretype == SUN_PREC_BOTH
    {
        SUN_PREC_RIGHT
    } else {
        SUN_PREC_NONE
    };

    /* if maxl input is illegal, set to default */
    let maxl = if maxl <= 0 { SUNSPFGMR_MAXL_DEFAULT } else { maxl };

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
        ops.gettype = Some(SUNLinSolGetType_SPFGMR);
        ops.getid = Some(SUNLinSolGetID_SPFGMR);
        ops.setatimes = Some(SUNLinSolSetATimes_SPFGMR);
        ops.setoptions = Some(SUNLinSolSetOptions_SPFGMR);
        ops.setpreconditioner = Some(SUNLinSolSetPreconditioner_SPFGMR);
        ops.setscalingvectors = Some(SUNLinSolSetScalingVectors_SPFGMR);
        ops.setzeroguess = Some(SUNLinSolSetZeroGuess_SPFGMR);
        ops.initialize = Some(SUNLinSolInitialize_SPFGMR);
        ops.setup = Some(SUNLinSolSetup_SPFGMR);
        ops.solve = Some(SUNLinSolSolve_SPFGMR);
        ops.numiters = Some(SUNLinSolNumIters_SPFGMR);
        ops.resnorm = Some(SUNLinSolResNorm_SPFGMR);
        ops.resid = Some(SUNLinSolResid_SPFGMR);
        ops.lastflag = Some(SUNLinSolLastFlag_SPFGMR);
        ops.space = Some(SUNLinSolSpace_SPFGMR);
        ops.free = Some(SUNLinSolFree_SPFGMR);
    }

    let xcor = N_VClone(y)?;
    let vtemp = N_VClone(y)?;
    *S.content.borrow_mut() = Box::new(SUNLinearSolverContent_SPFGMR_ {
        last_flag: 0,
        maxl,
        pretype,
        gstype: SUNSPFGMR_GSTYPE_DEFAULT,
        max_restarts: SUNSPFGMR_MAXRS_DEFAULT,
        zeroguess: SUNFALSE,
        numiters: 0,
        resnorm: ZERO,
        xcor: Some(xcor),
        vtemp: Some(vtemp),
        s1: None,
        s2: None,
        ATimes: None,
        ATData: None,
        Psetup: None,
        Psolve: None,
        PData: None,
        V: None,
        Z: None,
        Hes: Vec::new(),
        givens: Vec::new(),
        yg: Vec::new(),
        cv: Vec::new(),
        Xv: Vec::new(),
    });

    Some(S)
}

pub fn SUNLinSolSetOptions_SPFGMR(
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
                    let retval = SUNLinSol_SPFGMRSetPrecType(S, iarg);
                    if retval != SUN_SUCCESS {
                        return retval;
                    }
                }
                "gs_type" => {
                    idx += 1;
                    let iarg: i32 = crate::sundials_utils::atoi(&argv[idx]);
                    let retval = SUNLinSol_SPFGMRSetGSType(S, iarg);
                    if retval != SUN_SUCCESS {
                        return retval;
                    }
                }
                "max_restarts" => {
                    idx += 1;
                    let iarg: i32 = crate::sundials_utils::atoi(&argv[idx]);
                    let retval = SUNLinSol_SPFGMRSetMaxRestarts(S, iarg);
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

pub fn SUNLinSol_SPFGMRSetPrecType(S: &SUNLinearSolver, pretype: i32) -> SUNErrCode {
    let pretype = if pretype == SUN_PREC_LEFT || pretype == SUN_PREC_RIGHT || pretype == SUN_PREC_BOTH
    {
        SUN_PREC_RIGHT
    } else {
        SUN_PREC_NONE
    };
    content_mut(S).pretype = pretype;
    SUN_SUCCESS
}

pub fn SUNLinSol_SPFGMRSetGSType(S: &SUNLinearSolver, gstype: i32) -> SUNErrCode {
    content_mut(S).gstype = gstype;
    SUN_SUCCESS
}

pub fn SUNLinSol_SPFGMRSetMaxRestarts(S: &SUNLinearSolver, maxrs: i32) -> SUNErrCode {
    let maxrs = if maxrs < 0 { SUNSPFGMR_MAXRS_DEFAULT } else { maxrs };
    content_mut(S).max_restarts = maxrs;
    SUN_SUCCESS
}

pub fn SUNLinSolGetType_SPFGMR(_S: &SUNLinearSolver) -> SUNLinearSolver_Type {
    SUNLINEARSOLVER_ITERATIVE
}

pub fn SUNLinSolGetID_SPFGMR(_S: &SUNLinearSolver) -> SUNLinearSolver_ID {
    SUNLINEARSOLVER_SPFGMR
}

pub fn SUNLinSolInitialize_SPFGMR(S: &SUNLinearSolver) -> SUNErrCode {
    let mut content = content_mut(S);

    if content.max_restarts < 0 {
        content.max_restarts = SUNSPFGMR_MAXRS_DEFAULT;
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

    let maxl = content.maxl as usize;

    if content.V.is_none() {
        let vtemp = content.vtemp.as_ref().expect("vtemp allocated").clone();
        content.V = N_VCloneVectorArray(content.maxl + 1, &vtemp);
        if content.V.is_none() {
            return crate::sundials_errors::SUN_ERR_MEM_FAIL;
        }
    }

    if content.Z.is_none() {
        let vtemp = content.vtemp.as_ref().expect("vtemp allocated").clone();
        content.Z = N_VCloneVectorArray(content.maxl + 1, &vtemp);
        if content.Z.is_none() {
            return crate::sundials_errors::SUN_ERR_MEM_FAIL;
        }
    }

    if content.Hes.is_empty() {
        content.Hes = vec![vec![0.0; maxl]; maxl + 1];
    }
    if content.givens.is_empty() {
        content.givens = vec![0.0; 2 * maxl];
    }
    if content.yg.is_empty() {
        content.yg = vec![0.0; maxl + 1];
    }
    if content.cv.is_empty() {
        content.cv = vec![0.0; maxl + 1];
    }

    SUN_SUCCESS
}

pub fn SUNLinSolSetATimes_SPFGMR(
    S: &SUNLinearSolver,
    ATData: Option<Box<dyn Any>>,
    ATimes: Option<SUNATimesFn>,
) -> SUNErrCode {
    let mut content = content_mut(S);
    content.ATimes = ATimes;
    content.ATData = ATData;
    SUN_SUCCESS
}

pub fn SUNLinSolSetPreconditioner_SPFGMR(
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

pub fn SUNLinSolSetScalingVectors_SPFGMR(
    S: &SUNLinearSolver,
    s1: Option<&N_Vector>,
    s2: Option<&N_Vector>,
) -> SUNErrCode {
    let mut content = content_mut(S);
    content.s1 = s1.cloned();
    content.s2 = s2.cloned();
    SUN_SUCCESS
}

pub fn SUNLinSolSetZeroGuess_SPFGMR(S: &SUNLinearSolver, onoff: sunbooleantype) -> SUNErrCode {
    content_mut(S).zeroguess = onoff;
    SUN_SUCCESS
}

pub fn SUNLinSolSetup_SPFGMR(S: &SUNLinearSolver, _A: Option<&SUNMatrix>) -> i32 {
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

pub fn SUNLinSolSolve_SPFGMR(
    S: &SUNLinearSolver,
    _A: Option<&SUNMatrix>,
    x: &N_Vector,
    b: &N_Vector,
    delta: sunrealtype,
) -> i32 {
    /* Move solver state into locals (restored at the end) */
    let (
        l_max,
        max_restarts,
        gstype,
        pretype,
        V,
        Z,
        mut Hes,
        mut givens,
        xcor,
        mut yg,
        vtemp,
        s1,
        s2,
        mut A_data,
        mut P_data,
        atimes,
        psolve,
        mut zeroguess,
        mut cv,
    );
    {
        let mut content = content_mut(S);
        l_max = content.maxl;
        max_restarts = content.max_restarts;
        gstype = content.gstype;
        pretype = content.pretype;
        V = content.V.as_ref().expect("V allocated by initialize").clone();
        Z = content.Z.as_ref().expect("Z allocated by initialize").clone();
        Hes = std::mem::take(&mut content.Hes);
        givens = std::mem::take(&mut content.givens);
        xcor = content.xcor.as_ref().expect("xcor").clone();
        yg = std::mem::take(&mut content.yg);
        vtemp = content.vtemp.as_ref().expect("vtemp").clone();
        s1 = content.s1.clone();
        s2 = content.s2.clone();
        A_data = content.ATData.take();
        P_data = content.PData.take();
        atimes = content.ATimes.expect("ATimes set");
        psolve = content.Psolve;
        zeroguess = content.zeroguess;
        cv = std::mem::take(&mut content.cv);
        content.numiters = 0;
    }

    let mut nli: i32 = 0;
    let mut res_norm: sunrealtype = ZERO;

    /* preconditioning flag (right only for SPFGMR) */
    let preOnRight =
        pretype == SUN_PREC_LEFT || pretype == SUN_PREC_RIGHT || pretype == SUN_PREC_BOTH;
    let scale1 = s1.is_some();
    let scale2 = s2.is_some();

    let mut Xv: Vec<N_Vector> = Vec::with_capacity(l_max as usize + 1);

    let flag = (|| -> i32 {
        let mut converged = SUNFALSE;
        let mut krydim: i32 = 0;
        let mut status;

        /* Set vtemp and V[0] to initial (unscaled) residual r_0 = b - A*x_0 */
        if zeroguess {
            N_VScale(ONE, b, &vtemp);
        } else {
            status = atimes(&mut A_data, x, &vtemp);
            if status != 0 {
                zeroguess = SUNFALSE;
                return if status < 0 {
                    SUNLS_ATIMES_FAIL_UNREC
                } else {
                    SUNLS_ATIMES_FAIL_REC
                };
            }
            N_VLinearSum(ONE, b, -ONE, &vtemp, &vtemp);
        }

        /* Apply left scaling to vtemp = r_0 to fill V[0]. */
        if scale1 {
            N_VProd(s1.as_ref().expect("s1"), &vtemp, &V[0]);
        } else {
            N_VScale(ONE, &vtemp, &V[0]);
        }

        /* Set r_norm = beta to L2 norm of V[0] = s1 r_0, and return if small */
        let mut r_norm = N_VDotProd(&V[0], &V[0]);
        r_norm = SUNRsqrt(r_norm);
        let beta = r_norm;
        res_norm = r_norm;

        if r_norm <= delta {
            zeroguess = SUNFALSE;
            return SUN_SUCCESS;
        }

        let mut rho = beta;

        /* Set xcor = 0. */
        N_VConst(ZERO, &xcor);

        /* Begin outer iterations: up to (max_restarts + 1) attempts. */
        let mut ntries = 0;
        while ntries <= max_restarts {
            /* Initialize the Hessenberg matrix Hes and Givens rotation product.
            Normalize the initial vector V[0]. */
            for i in 0..=(l_max as usize) {
                for j in 0..(l_max as usize) {
                    Hes[i][j] = ZERO;
                }
            }
            let mut rotation_product = ONE;
            N_VScale(ONE / r_norm, &V[0], &V[0]);

            /* Inner loop: generate Krylov sequence and Arnoldi basis. */
            let mut l: usize = 0;
            while (l as i32) < l_max {
                nli += 1;
                krydim = (l + 1) as i32;

                /* Apply right scaling: vtemp = s2_inv V[l]. */
                if scale2 {
                    N_VDiv(&V[l], s2.as_ref().expect("s2"), &vtemp);
                } else {
                    N_VScale(ONE, &V[l], &vtemp);
                }

                /* Apply right preconditioner: vtemp = Z[l] = P_inv s2_inv V[l]. */
                if preOnRight {
                    N_VScale(ONE, &vtemp, &V[l + 1]);
                    status = (psolve.expect("psolve"))(
                        &mut P_data,
                        &V[l + 1],
                        &vtemp,
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
                }
                N_VScale(ONE, &vtemp, &Z[l]);

                /* Apply A: V[l+1] = A P_inv s2_inv V[l]. */
                status = atimes(&mut A_data, &vtemp, &V[l + 1]);
                if status != 0 {
                    zeroguess = SUNFALSE;
                    return if status < 0 {
                        SUNLS_ATIMES_FAIL_UNREC
                    } else {
                        SUNLS_ATIMES_FAIL_REC
                    };
                }

                /* Apply left scaling: V[l+1] = s1 A P_inv s2_inv V[l]. */
                if scale1 {
                    N_VProd(s1.as_ref().expect("s1"), &V[l + 1], &V[l + 1]);
                }

                /* Orthogonalize V[l+1] against previous V[i] */
                let mut new_norm = ZERO;
                if gstype == SUN_CLASSICAL_GS {
                    if Xv.len() < l + 2 {
                        Xv.resize(l + 2, V[0].clone());
                    }
                    let ier = SUNClassicalGS(
                        &V,
                        &mut Hes,
                        (l + 1) as i32,
                        l_max,
                        &mut new_norm,
                        &mut cv,
                        &mut Xv,
                    );
                    if ier != SUN_SUCCESS {
                        return ier;
                    }
                } else {
                    let ier = SUNModifiedGS(&V, &mut Hes, (l + 1) as i32, l_max, &mut new_norm);
                    if ier != SUN_SUCCESS {
                        return ier;
                    }
                }
                Hes[l + 1][l] = new_norm;

                /* Update the QR factorization of Hes. */
                if SUNQRfact(krydim, &mut Hes, &mut givens, l as i32) != 0 {
                    zeroguess = SUNFALSE;
                    return SUNLS_QRFACT_FAIL;
                }

                /* Update residual norm estimate; break on convergence. */
                rotation_product *= givens[2 * l + 1];
                rho = (rotation_product * r_norm).abs();
                res_norm = rho;

                if rho <= delta {
                    converged = SUNTRUE;
                    break;
                }

                /* Normalize V[l+1] with norm value from Gram-Schmidt. */
                N_VScale(ONE / Hes[l + 1][l], &V[l + 1], &V[l + 1]);

                l += 1;
            }

            /* Inner loop is done. Compute the new correction vector xcor. */

            /* Construct g, then solve for y. */
            yg[0] = r_norm;
            for i in 1..=(krydim as usize) {
                yg[i] = ZERO;
            }
            if SUNQRsol(krydim, &mut Hes, &givens, &mut yg) != 0 {
                zeroguess = SUNFALSE;
                return SUNLS_QRSOL_FAIL;
            }

            /* Add correction vector Z_l y to xcor. */
            cv[0] = ONE;
            Xv.clear();
            Xv.push(xcor.clone());
            for k in 0..(krydim as usize) {
                cv[k + 1] = yg[k];
                Xv.push(Z[k].clone());
            }
            let ier = N_VLinearCombination(krydim + 1, &cv, &Xv, &xcor);
            if ier != SUN_SUCCESS {
                return ier;
            }

            /* If converged, construct the final solution vector x and return. */
            if converged {
                if zeroguess {
                    N_VScale(ONE, &xcor, x);
                } else {
                    N_VLinearSum(ONE, x, ONE, &xcor, x);
                }
                zeroguess = SUNFALSE;
                return SUN_SUCCESS;
            }

            /* Not yet converged; if allowed, prepare for restart. */
            if ntries == max_restarts {
                break;
            }

            /* Construct last column of Q in yg. */
            let mut s_product = ONE;
            let mut i = krydim as usize;
            while i > 0 {
                yg[i] = s_product * givens[2 * i - 2];
                s_product *= givens[2 * i - 1];
                i -= 1;
            }
            yg[0] = s_product;

            /* Scale r_norm and yg. */
            r_norm *= s_product;
            for i in 0..=(krydim as usize) {
                yg[i] *= r_norm;
            }
            r_norm = r_norm.abs();

            /* Multiply yg by V_(krydim+1) to get last residual vector; restart. */
            Xv.clear();
            for k in 0..=(krydim as usize) {
                cv[k] = yg[k];
                Xv.push(V[k].clone());
            }
            let ier = N_VLinearCombination(krydim + 1, &cv, &Xv, &V[0]);
            if ier != SUN_SUCCESS {
                return ier;
            }

            ntries += 1;
        }

        /* Failed to converge, even after allowed restarts. */
        if rho < beta {
            if zeroguess {
                N_VScale(ONE, &xcor, x);
            } else {
                N_VLinearSum(ONE, x, ONE, &xcor, x);
            }
            zeroguess = SUNFALSE;
            return SUNLS_RES_REDUCED;
        }

        zeroguess = SUNFALSE;
        SUNLS_CONV_FAIL
    })();

    /* restore solver state and write results back */
    {
        let mut content = content_mut(S);
        content.Hes = Hes;
        content.givens = givens;
        content.yg = yg;
        content.cv = cv;
        content.ATData = A_data;
        content.PData = P_data;
        content.zeroguess = zeroguess;
        content.numiters = nli;
        content.resnorm = res_norm;
        content.last_flag = flag as sunindextype;
    }

    flag
}

pub fn SUNLinSolNumIters_SPFGMR(S: &SUNLinearSolver) -> i32 {
    content_mut(S).numiters
}

pub fn SUNLinSolResNorm_SPFGMR(S: &SUNLinearSolver) -> sunrealtype {
    content_mut(S).resnorm
}

pub fn SUNLinSolResid_SPFGMR(S: &SUNLinearSolver) -> Option<N_Vector> {
    content_mut(S).vtemp.clone()
}

pub fn SUNLinSolLastFlag_SPFGMR(S: &SUNLinearSolver) -> sunindextype {
    content_mut(S).last_flag
}

pub fn SUNLinSolSpace_SPFGMR(
    S: &SUNLinearSolver,
    lenrwLS: &mut i64,
    leniwLS: &mut i64,
) -> SUNErrCode {
    let (maxl, vtemp) = {
        let content = content_mut(S);
        (
            content.maxl as i64,
            content.vtemp.as_ref().expect("vtemp").clone(),
        )
    };
    let (mut lrw1, mut liw1) = (0i64, 0i64);
    if vtemp.ops.borrow().nvspace.is_some() {
        N_VSpace(&vtemp, &mut lrw1, &mut liw1);
    }
    *lenrwLS = lrw1 * (2 * maxl + 4) + maxl * (maxl + 5) + 2;
    *leniwLS = liw1 * (2 * maxl + 4);
    SUN_SUCCESS
}

pub fn SUNLinSolFree_SPFGMR(_S: &SUNLinearSolver) -> SUNErrCode {
    SUN_SUCCESS
}
