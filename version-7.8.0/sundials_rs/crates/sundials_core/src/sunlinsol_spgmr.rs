//! Port of `src/sunlinsol/spgmr/sunlinsol_spgmr.c` +
//! `include/sunlinsol/sunlinsol_spgmr.h` (scaled preconditioned GMRES).
//!
//! Solve strategy: all content state is moved into locals at entry
//! (`Option::take` for the callback data boxes, `mem::take` for the
//! workspace arrays, handle clones for vectors), the C algorithm runs
//! without holding the content borrow (so ATimes/Psolve callbacks may
//! freely re-enter integrator state), and everything is restored at the
//! single exit point. `SUNLogInfo` lines compile away at logging level 2.

use std::any::Any;
use std::cell::RefMut;

use crate::sundials_context::SUNContext;
use crate::sundials_errors::{SUN_ERR_ARG_CORRUPT, SUN_SUCCESS};
use crate::sundials_iterative::*;
use crate::sundials_linearsolver::*;
use crate::sundials_matrix::SUNMatrix;
use crate::sundials_nvector::*;
use crate::sundials_types::*;

const ZERO: sunrealtype = 0.0;
const ONE: sunrealtype = 1.0;

pub const SUNSPGMR_MAXL_DEFAULT: i32 = 5;
pub const SUNSPGMR_MAXRS_DEFAULT: i32 = 0;
pub const SUNSPGMR_GSTYPE_DEFAULT: i32 = SUN_MODIFIED_GS;

pub struct SUNLinearSolverContent_SPGMR_ {
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
    pub Hes: Vec<Vec<sunrealtype>>,
    pub givens: Vec<sunrealtype>,
    pub xcor: Option<N_Vector>,
    pub yg: Vec<sunrealtype>,
    pub vtemp: Option<N_Vector>,

    pub cv: Vec<sunrealtype>,
    pub Xv: Vec<N_Vector>,
}

pub type SUNLinearSolverContent_SPGMR = SUNLinearSolverContent_SPGMR_;

fn content_mut(S: &SUNLinearSolver) -> RefMut<'_, SUNLinearSolverContent_SPGMR_> {
    RefMut::map(S.content.borrow_mut(), |c| {
        c.downcast_mut::<SUNLinearSolverContent_SPGMR_>()
            .expect("SPGMR SUNLinearSolver content")
    })
}

pub fn SUNLinSol_SPGMR(
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
    let maxl = if maxl <= 0 { SUNSPGMR_MAXL_DEFAULT } else { maxl };

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

    /* Create linear solver */
    let S = SUNLinSolNewEmpty(sunctx)?;

    /* Attach operations */
    {
        let mut ops = S.ops.borrow_mut();
        ops.gettype = Some(SUNLinSolGetType_SPGMR);
        ops.getid = Some(SUNLinSolGetID_SPGMR);
        ops.setatimes = Some(SUNLinSolSetATimes_SPGMR);
        ops.setoptions = Some(SUNLinSolSetOptions_SPGMR);
        ops.setpreconditioner = Some(SUNLinSolSetPreconditioner_SPGMR);
        ops.setscalingvectors = Some(SUNLinSolSetScalingVectors_SPGMR);
        ops.setzeroguess = Some(SUNLinSolSetZeroGuess_SPGMR);
        ops.initialize = Some(SUNLinSolInitialize_SPGMR);
        ops.setup = Some(SUNLinSolSetup_SPGMR);
        ops.solve = Some(SUNLinSolSolve_SPGMR);
        ops.numiters = Some(SUNLinSolNumIters_SPGMR);
        ops.resnorm = Some(SUNLinSolResNorm_SPGMR);
        ops.resid = Some(SUNLinSolResid_SPGMR);
        ops.lastflag = Some(SUNLinSolLastFlag_SPGMR);
        ops.space = Some(SUNLinSolSpace_SPGMR);
        ops.free = Some(SUNLinSolFree_SPGMR);
    }

    /* Create, attach, fill content */
    let xcor = N_VClone(y)?;
    let vtemp = N_VClone(y)?;
    *S.content.borrow_mut() = Box::new(SUNLinearSolverContent_SPGMR_ {
        last_flag: 0,
        maxl,
        pretype,
        gstype: SUNSPGMR_GSTYPE_DEFAULT,
        max_restarts: SUNSPGMR_MAXRS_DEFAULT,
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
        Hes: Vec::new(),
        givens: Vec::new(),
        yg: Vec::new(),
        cv: Vec::new(),
        Xv: Vec::new(),
    });

    Some(S)
}

pub fn SUNLinSolSetOptions_SPGMR(
    S: &SUNLinearSolver,
    LSid: Option<&str>,
    file_name: Option<&str>,
    argv: &[String],
) -> SUNErrCode {
    /* File-based option control is currently unimplemented */
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
                    let retval = SUNLinSol_SPGMRSetPrecType(S, iarg);
                    if retval != SUN_SUCCESS {
                        return retval;
                    }
                }
                "gs_type" => {
                    idx += 1;
                    let iarg: i32 = crate::sundials_utils::atoi(&argv[idx]);
                    let retval = SUNLinSol_SPGMRSetGSType(S, iarg);
                    if retval != SUN_SUCCESS {
                        return retval;
                    }
                }
                "max_restarts" => {
                    idx += 1;
                    let iarg: i32 = crate::sundials_utils::atoi(&argv[idx]);
                    let retval = SUNLinSol_SPGMRSetMaxRestarts(S, iarg);
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

pub fn SUNLinSol_SPGMRSetPrecType(S: &SUNLinearSolver, pretype: i32) -> SUNErrCode {
    content_mut(S).pretype = pretype;
    SUN_SUCCESS
}

pub fn SUNLinSol_SPGMRSetGSType(S: &SUNLinearSolver, gstype: i32) -> SUNErrCode {
    content_mut(S).gstype = gstype;
    SUN_SUCCESS
}

pub fn SUNLinSol_SPGMRSetMaxRestarts(S: &SUNLinearSolver, maxrs: i32) -> SUNErrCode {
    /* Illegal maxrs implies use of default value */
    let maxrs = if maxrs < 0 { SUNSPGMR_MAXRS_DEFAULT } else { maxrs };
    content_mut(S).max_restarts = maxrs;
    SUN_SUCCESS
}

pub fn SUNLinSolGetType_SPGMR(_S: &SUNLinearSolver) -> SUNLinearSolver_Type {
    SUNLINEARSOLVER_ITERATIVE
}

pub fn SUNLinSolGetID_SPGMR(_S: &SUNLinearSolver) -> SUNLinearSolver_ID {
    SUNLINEARSOLVER_SPGMR
}

pub fn SUNLinSolInitialize_SPGMR(S: &SUNLinearSolver) -> SUNErrCode {
    let mut content = content_mut(S);

    /* ensure valid options */
    if content.max_restarts < 0 {
        content.max_restarts = SUNSPGMR_MAXRS_DEFAULT;
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

    /* allocate solver-specific memory */
    if content.V.is_none() {
        let vtemp = content.vtemp.as_ref().expect("vtemp allocated").clone();
        content.V = N_VCloneVectorArray(content.maxl + 1, &vtemp);
        if content.V.is_none() {
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

    /* Xv is (re)built during solves; nothing to pre-allocate beyond capacity */

    SUN_SUCCESS
}

pub fn SUNLinSolSetATimes_SPGMR(
    S: &SUNLinearSolver,
    ATData: Option<Box<dyn Any>>,
    ATimes: Option<SUNATimesFn>,
) -> SUNErrCode {
    let mut content = content_mut(S);
    content.ATimes = ATimes;
    content.ATData = ATData;
    SUN_SUCCESS
}

pub fn SUNLinSolSetPreconditioner_SPGMR(
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

pub fn SUNLinSolSetScalingVectors_SPGMR(
    S: &SUNLinearSolver,
    s1: Option<&N_Vector>,
    s2: Option<&N_Vector>,
) -> SUNErrCode {
    let mut content = content_mut(S);
    content.s1 = s1.cloned();
    content.s2 = s2.cloned();
    SUN_SUCCESS
}

pub fn SUNLinSolSetZeroGuess_SPGMR(S: &SUNLinearSolver, onff: sunbooleantype) -> SUNErrCode {
    content_mut(S).zeroguess = onff;
    SUN_SUCCESS
}

pub fn SUNLinSolSetup_SPGMR(S: &SUNLinearSolver, _A: Option<&SUNMatrix>) -> i32 {
    /* if user-supplied Psetup routine exists, call that here */
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

pub fn SUNLinSolSolve_SPGMR(
    S: &SUNLinearSolver,
    _A: Option<&SUNMatrix>,
    x: &N_Vector,
    b: &N_Vector,
    delta: sunrealtype,
) -> i32 {
    /* Move solver state into locals (restored in `finish`) */
    let (
        l_max,
        max_restarts,
        gstype,
        pretype,
        V,
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

    /* Set sunbooleantype flags for internal solver options */
    let preOnLeft = pretype == SUN_PREC_LEFT || pretype == SUN_PREC_BOTH;
    let preOnRight = pretype == SUN_PREC_RIGHT || pretype == SUN_PREC_BOTH;
    let scale1 = s1.is_some();
    let scale2 = s2.is_some();

    let mut Xv: Vec<N_Vector> = Vec::with_capacity(l_max as usize + 1);

    /* run the algorithm; captures locals by &mut via closure-free inner fn */
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
        N_VScale(ONE, &vtemp, &V[0]);

        /* Apply left preconditioner and left scaling to V[0] = r_0 */
        if preOnLeft {
            status = (psolve.expect("psolve"))(&mut P_data, &V[0], &vtemp, delta, SUN_PREC_LEFT);
            if status != 0 {
                zeroguess = SUNFALSE;
                return if status < 0 {
                    SUNLS_PSOLVE_FAIL_UNREC
                } else {
                    SUNLS_PSOLVE_FAIL_REC
                };
            }
        } else {
            N_VScale(ONE, &V[0], &vtemp);
        }

        if scale1 {
            N_VProd(s1.as_ref().expect("s1"), &vtemp, &V[0]);
        } else {
            N_VScale(ONE, &vtemp, &V[0]);
        }

        /* Set r_norm = beta to L2 norm of V[0], and return if small */
        let mut r_norm = N_VDotProd(&V[0], &V[0]);
        r_norm = SUNRsqrt_local(r_norm);
        let beta = r_norm;
        res_norm = r_norm;

        if r_norm <= delta {
            zeroguess = SUNFALSE;
            return SUN_SUCCESS;
        }

        /* Initialize rho to avoid compiler warning message */
        let mut rho = beta;

        /* Set xcor = 0 */
        N_VConst(ZERO, &xcor);

        /* Begin outer iterations: up to (max_restarts + 1) attempts */
        let mut ntries = 0;
        while ntries <= max_restarts {
            /* Initialize the Hessenberg matrix Hes and Givens rotation
            product. Normalize the initial vector V[0] */
            for i in 0..=(l_max as usize) {
                for j in 0..(l_max as usize) {
                    Hes[i][j] = ZERO;
                }
            }

            let mut rotation_product = ONE;
            N_VScale(ONE / r_norm, &V[0], &V[0]);

            /* Inner loop: generate Krylov sequence and Arnoldi basis */
            let mut l: usize = 0;
            let mut l_plus_1: usize;
            while (l as i32) < l_max {
                nli += 1;
                l_plus_1 = l + 1;
                krydim = l_plus_1 as i32;

                /* Generate A-tilde V[l] */

                /* Apply right scaling: vtemp = s2_inv V[l] */
                if scale2 {
                    N_VDiv(&V[l], s2.as_ref().expect("s2"), &vtemp);
                } else {
                    N_VScale(ONE, &V[l], &vtemp);
                }

                /* Apply right preconditioner: vtemp = P2_inv s2_inv V[l] */
                if preOnRight {
                    N_VScale(ONE, &vtemp, &V[l_plus_1]);
                    status = (psolve.expect("psolve"))(
                        &mut P_data,
                        &V[l_plus_1],
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

                /* Apply A: V[l+1] = A P2_inv s2_inv V[l] */
                status = atimes(&mut A_data, &vtemp, &V[l_plus_1]);
                if status != 0 {
                    zeroguess = SUNFALSE;
                    return if status < 0 {
                        SUNLS_ATIMES_FAIL_UNREC
                    } else {
                        SUNLS_ATIMES_FAIL_REC
                    };
                }

                /* Apply left preconditioning: vtemp = P1_inv A P2_inv s2_inv V[l] */
                if preOnLeft {
                    status = (psolve.expect("psolve"))(
                        &mut P_data,
                        &V[l_plus_1],
                        &vtemp,
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
                    N_VScale(ONE, &V[l_plus_1], &vtemp);
                }

                /* Apply left scaling: V[l+1] = s1 P1_inv A P2_inv s2_inv V[l] */
                if scale1 {
                    N_VProd(s1.as_ref().expect("s1"), &vtemp, &V[l_plus_1]);
                } else {
                    N_VScale(ONE, &vtemp, &V[l_plus_1]);
                }

                /* Orthogonalize V[l+1] against previous V[i] */
                let mut new_norm = ZERO;
                if gstype == SUN_CLASSICAL_GS {
                    /* vtemp workspace array for classical GS */
                    if Xv.len() < l_plus_1 + 1 {
                        Xv.resize(l_plus_1 + 1, V[0].clone());
                    }
                    let ier = SUNClassicalGS(
                        &V,
                        &mut Hes,
                        l_plus_1 as i32,
                        l_max,
                        &mut new_norm,
                        &mut cv,
                        &mut Xv,
                    );
                    if ier != SUN_SUCCESS {
                        return ier;
                    }
                } else {
                    let ier =
                        SUNModifiedGS(&V, &mut Hes, l_plus_1 as i32, l_max, &mut new_norm);
                    if ier != SUN_SUCCESS {
                        return ier;
                    }
                }
                Hes[l_plus_1][l] = new_norm;

                /* Update the QR factorization of Hes */
                if SUNQRfact(krydim, &mut Hes, &mut givens, l as i32) != 0 {
                    zeroguess = SUNFALSE;
                    return SUNLS_QRFACT_FAIL;
                }

                /* Update residual norm estimate; break if convergence test passes */
                rotation_product *= givens[2 * l + 1];
                rho = (rotation_product * r_norm).abs();
                res_norm = rho;

                if rho <= delta {
                    converged = SUNTRUE;
                    break;
                }

                /* Normalize V[l+1] with norm value from the Gram-Schmidt routine */
                N_VScale(ONE / Hes[l_plus_1][l], &V[l_plus_1], &V[l_plus_1]);

                l += 1;
            }

            /* Inner loop is done. Compute the new correction vector xcor */

            /* Construct g, then solve for y */
            yg[0] = r_norm;
            for i in 1..=(krydim as usize) {
                yg[i] = ZERO;
            }
            if SUNQRsol(krydim, &mut Hes, &givens, &mut yg) != 0 {
                zeroguess = SUNFALSE;
                return SUNLS_QRSOL_FAIL;
            }

            /* Add correction vector V_l y to xcor */
            cv[0] = ONE;
            Xv.clear();
            Xv.push(xcor.clone());
            for k in 0..(krydim as usize) {
                cv[k + 1] = yg[k];
                Xv.push(V[k].clone());
            }
            let ier = N_VLinearCombination(krydim + 1, &cv, &Xv, &xcor);
            if ier != SUN_SUCCESS {
                return ier;
            }

            /* If converged, construct the final solution vector x and return */
            if converged {
                /* Apply right scaling and right precond.: vtemp = P2_inv s2_inv xcor */
                if scale2 {
                    N_VDiv(&xcor, s2.as_ref().expect("s2"), &xcor);
                }

                if preOnRight {
                    status =
                        (psolve.expect("psolve"))(&mut P_data, &xcor, &vtemp, delta, SUN_PREC_RIGHT);
                    if status != 0 {
                        zeroguess = SUNFALSE;
                        return if status < 0 {
                            SUNLS_PSOLVE_FAIL_UNREC
                        } else {
                            SUNLS_PSOLVE_FAIL_REC
                        };
                    }
                } else {
                    N_VScale(ONE, &xcor, &vtemp);
                }

                /* Add vtemp to initial x to get final solution x, and return */
                if zeroguess {
                    N_VScale(ONE, &vtemp, x);
                } else {
                    N_VLinearSum(ONE, x, ONE, &vtemp, x);
                }

                zeroguess = SUNFALSE;
                return SUN_SUCCESS;
            }

            /* Not yet converged; if allowed, prepare for restart */
            if ntries == max_restarts {
                break;
            }

            /* Construct last column of Q in yg */
            let mut s_product = ONE;
            let mut i = krydim as usize;
            while i > 0 {
                yg[i] = s_product * givens[2 * i - 2];
                s_product *= givens[2 * i - 1];
                i -= 1;
            }
            yg[0] = s_product;

            /* Scale r_norm and yg */
            r_norm *= s_product;
            for i in 0..=(krydim as usize) {
                yg[i] *= r_norm;
            }
            r_norm = r_norm.abs();

            /* Multiply yg by V_(krydim+1) to get last residual vector; restart */
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

        /* Failed to converge, even after allowed restarts.
        If the residual norm was reduced below its initial value, compute
        and return x anyway. Otherwise return failure flag. */
        if rho < beta {
            /* Apply right scaling and right precond.: vtemp = P2_inv s2_inv xcor */
            if scale2 {
                N_VDiv(&xcor, s2.as_ref().expect("s2"), &xcor);
            }

            if preOnRight {
                status =
                    (psolve.expect("psolve"))(&mut P_data, &xcor, &vtemp, delta, SUN_PREC_RIGHT);
                if status != 0 {
                    zeroguess = SUNFALSE;
                    return if status < 0 {
                        SUNLS_PSOLVE_FAIL_UNREC
                    } else {
                        SUNLS_PSOLVE_FAIL_REC
                    };
                }
            } else {
                N_VScale(ONE, &xcor, &vtemp);
            }

            /* Add vtemp to initial x to get final solution x, and return */
            if zeroguess {
                N_VScale(ONE, &vtemp, x);
            } else {
                N_VLinearSum(ONE, x, ONE, &vtemp, x);
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

fn SUNRsqrt_local(x: sunrealtype) -> sunrealtype {
    crate::sundials_math::SUNRsqrt(x)
}

pub fn SUNLinSolNumIters_SPGMR(S: &SUNLinearSolver) -> i32 {
    content_mut(S).numiters
}

pub fn SUNLinSolResNorm_SPGMR(S: &SUNLinearSolver) -> sunrealtype {
    content_mut(S).resnorm
}

pub fn SUNLinSolResid_SPGMR(S: &SUNLinearSolver) -> Option<N_Vector> {
    content_mut(S).vtemp.clone()
}

pub fn SUNLinSolLastFlag_SPGMR(S: &SUNLinearSolver) -> sunindextype {
    content_mut(S).last_flag
}

pub fn SUNLinSolSpace_SPGMR(S: &SUNLinearSolver, lenrwLS: &mut i64, leniwLS: &mut i64) -> SUNErrCode {
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
    *lenrwLS = lrw1 * (maxl + 5) + maxl * (maxl + 5) + 2;
    *leniwLS = liw1 * (maxl + 5);
    SUN_SUCCESS
}

pub fn SUNLinSolFree_SPGMR(_S: &SUNLinearSolver) -> SUNErrCode {
    SUN_SUCCESS
}
