//! Port of `src/sunnonlinsol/fixedpoint/sunnonlinsol_fixedpoint.c` +
//! `include/sunnonlinsol/sunnonlinsol_fixedpoint.h` (fixed-point iteration
//! with optional Anderson acceleration and damping).
//!
//! In `AndersonAccelerate` the C code uses the result vector `x` as the
//! temporary `vtemp` (deliberate aliasing) — preserved via a handle clone.
//! The `GetUpdateNorm_FixedPoint` call in the solve loop exists only at
//! logging level >= INFO and is omitted (level 2 reference builds).

use std::any::Any;
use std::cell::RefMut;

use crate::sundials_context::SUNContext;
use crate::sundials_errors::{SUN_ERR_ARG_CORRUPT, SUN_ERR_ARG_INCOMPATIBLE, SUN_SUCCESS};
use crate::sundials_math::SUNRsqrt;
use crate::sundials_nonlinearsolver::*;
use crate::sundials_nvector::*;
use crate::sundials_nvector_senswrapper::N_VNew_SensWrapper;
use crate::sundials_types::*;

const ONE: sunrealtype = 1.0;
const ZERO: sunrealtype = 0.0;

pub struct SUNNonlinearSolverContent_FixedPoint_ {
    pub Sys: Option<SUNNonlinSolSysFn>,
    pub CTest: Option<SUNNonlinSolConvTestFn>,
    pub norm_fn: Option<SUNNonlinSolNormFn>,
    pub norm_fn_data: Option<Box<dyn Any>>,
    pub getupdatenorm_fn: Option<SUNNonlinSolGetUpdateNormFn>,
    pub getupdatenorm_data: Option<Box<dyn Any>>,

    pub m: i32,
    pub imap: Vec<i32>,
    pub damping: sunbooleantype,
    pub beta: sunrealtype,
    pub R: Vec<sunrealtype>,
    pub gamma: Vec<sunrealtype>,
    pub cvals: Vec<sunrealtype>,
    pub df: Option<Vec<N_Vector>>,
    pub dg: Option<Vec<N_Vector>>,
    pub q: Option<Vec<N_Vector>>,
    pub Xvecs: Vec<N_Vector>,
    pub yprev: Option<N_Vector>,
    pub gy: Option<N_Vector>,
    pub fold: Option<N_Vector>,
    pub gold: Option<N_Vector>,
    pub delta: Option<N_Vector>,
    pub curiter: i32,
    pub maxiters: i32,
    pub niters: i64,
    pub nconvfails: i64,
    pub ctest_data: Option<Box<dyn Any>>,
    pub delnrm: sunrealtype,
}

pub type SUNNonlinearSolverContent_FixedPoint = SUNNonlinearSolverContent_FixedPoint_;

fn content_mut(NLS: &SUNNonlinearSolver) -> RefMut<'_, SUNNonlinearSolverContent_FixedPoint_> {
    RefMut::map(NLS.content.borrow_mut(), |c| {
        c.downcast_mut::<SUNNonlinearSolverContent_FixedPoint_>()
            .expect("FixedPoint SUNNonlinearSolver content")
    })
}

pub fn SUNNonlinSol_FixedPoint(
    y: &N_Vector,
    m: i32,
    sunctx: &SUNContext,
) -> Option<SUNNonlinearSolver> {
    /* Check that the supplied N_Vector supports all required operations */
    {
        let ops = y.ops.borrow();
        if ops.nvclone.is_none()
            || ops.nvscale.is_none()
            || ops.nvlinearsum.is_none()
            || ops.nvdotprod.is_none()
        {
            return None;
        }
    }

    /* Create nonlinear linear solver */
    let NLS = SUNNonlinSolNewEmpty(sunctx)?;

    /* Attach operations */
    {
        let mut ops = NLS.ops.borrow_mut();
        ops.gettype = Some(SUNNonlinSolGetType_FixedPoint);
        ops.initialize = Some(SUNNonlinSolInitialize_FixedPoint);
        ops.solve = Some(SUNNonlinSolSolve_FixedPoint);
        ops.free = Some(SUNNonlinSolFree_FixedPoint);
        ops.setsysfn = Some(SUNNonlinSolSetSysFn_FixedPoint);
        ops.setctestfn = Some(SUNNonlinSolSetConvTestFn_FixedPoint);
        ops.setnormfn = Some(SUNNonlinSolSetNormFn_FixedPoint);
        ops.setgetupdatenormfn = Some(SUNNonlinSolSetGetUpdateNormFn_FixedPoint);
        ops.setoptions = Some(SUNNonlinSolSetOptions_FixedPoint);
        ops.setmaxiters = Some(SUNNonlinSolSetMaxIters_FixedPoint);
        ops.getnumiters = Some(SUNNonlinSolGetNumIters_FixedPoint);
        ops.getcuriter = Some(SUNNonlinSolGetCurIter_FixedPoint);
        ops.getnumconvfails = Some(SUNNonlinSolGetNumConvFails_FixedPoint);
    }

    /* Create content and fill allocatable content (C AllocateContent) */
    let yprev = N_VClone(y)?;
    let gy = N_VClone(y)?;
    let delta = N_VClone(y)?;
    let (fold, gold, imap, R, gamma, cvals, df, dg, q);
    if m > 0 {
        fold = Some(N_VClone(y)?);
        gold = Some(N_VClone(y)?);
        imap = vec![0; m as usize];
        R = vec![0.0; (m * m) as usize];
        gamma = vec![0.0; m as usize];
        cvals = vec![0.0; 2 * (m as usize + 1)];
        df = Some(N_VCloneVectorArray(m, y)?);
        dg = Some(N_VCloneVectorArray(m, y)?);
        q = Some(N_VCloneVectorArray(m, y)?);
    } else {
        fold = None;
        gold = None;
        imap = Vec::new();
        R = Vec::new();
        gamma = Vec::new();
        cvals = Vec::new();
        df = None;
        dg = None;
        q = None;
    }

    *NLS.content.borrow_mut() = Box::new(SUNNonlinearSolverContent_FixedPoint_ {
        Sys: None,
        CTest: None,
        norm_fn: None,
        norm_fn_data: None,
        getupdatenorm_fn: None,
        getupdatenorm_data: None,
        m,
        damping: SUNFALSE,
        beta: ONE,
        curiter: 0,
        maxiters: 3,
        niters: 0,
        nconvfails: 0,
        ctest_data: None,
        delnrm: 0.0,
        imap,
        R,
        gamma,
        cvals,
        df,
        dg,
        q,
        Xvecs: Vec::new(),
        yprev: Some(yprev),
        gy: Some(gy),
        fold,
        gold,
        delta: Some(delta),
    });

    Some(NLS)
}

pub fn SUNNonlinSol_FixedPointSens(
    count: i32,
    y: &N_Vector,
    m: i32,
    sunctx: &SUNContext,
) -> Option<SUNNonlinearSolver> {
    /* create sensitivity vector wrapper */
    let w = N_VNew_SensWrapper(count, y)?;

    /* create nonlinear solver using sensitivity vector wrapper */
    let NLS = SUNNonlinSol_FixedPoint(&w, m, sunctx)?;

    /* free sensitivity vector wrapper */
    N_VDestroy(w);

    Some(NLS)
}

pub fn SUNNonlinSolGetType_FixedPoint(_NLS: &SUNNonlinearSolver) -> SUNNonlinearSolver_Type {
    SUNNONLINEARSOLVER_FIXEDPOINT
}

pub fn SUNNonlinSolInitialize_FixedPoint(NLS: &SUNNonlinearSolver) -> SUNErrCode {
    let mut content = content_mut(NLS);

    /* check that all required function pointers have been set */
    if content.Sys.is_none() || content.CTest.is_none() {
        return SUN_ERR_ARG_CORRUPT;
    }

    /* reset the total number of iterations and convergence failures */
    content.niters = 0;
    content.nconvfails = 0;

    SUN_SUCCESS
}

pub fn SUNNonlinSolSolve_FixedPoint(
    NLS: &SUNNonlinearSolver,
    _y0: &N_Vector,
    ycor: &N_Vector,
    w: &N_Vector,
    tol: sunrealtype,
    _callSetup: sunbooleantype,
    mem: &mut Option<Box<dyn Any>>,
) -> i32 {
    /* set local shortcut variables */
    let (yprev, gy, delta, Sys, CTest, m, maxiters);
    {
        let content = content_mut(NLS);
        yprev = content.yprev.as_ref().expect("yprev").clone();
        gy = content.gy.as_ref().expect("gy").clone();
        delta = content.delta.as_ref().expect("delta").clone();
        Sys = content.Sys.expect("Sys set");
        CTest = content.CTest.expect("CTest set");
        m = content.m;
        maxiters = content.maxiters;
    }

    /* initialize iteration and convergence fail counters for this solve */
    {
        let mut content = content_mut(NLS);
        content.niters = 0;
        content.nconvfails = 0;
    }

    let mut retval: i32;

    /* Looping point for attempts at solution of the nonlinear system */
    let mut curiter = 0;
    while curiter < maxiters {
        content_mut(NLS).curiter = curiter;

        /* update previous solution guess */
        N_VScale(ONE, ycor, &yprev);

        /* Compute fixed-point iteration function, store in gy */
        retval = Sys(ycor, &gy, mem);
        if retval != 0 {
            return retval;
        }

        /* perform fixed point update, based on choice of acceleration or not */
        if m == 0 {
            /* basic fixed-point solver */
            N_VScale(ONE, &gy, ycor);
        } else {
            /* Anderson-accelerated solver */
            let ier = AndersonAccelerate(NLS, &gy, ycor, &yprev, curiter);
            if ier != SUN_SUCCESS {
                return ier;
            }
        }

        /* increment nonlinear solver iteration counter */
        content_mut(NLS).niters += 1;

        /* compute change in solution, and call the convergence test function */
        N_VLinearSum(ONE, ycor, -ONE, &yprev, &delta);

        /* test for convergence */
        let mut ctest_data = content_mut(NLS).ctest_data.take();
        retval = CTest(NLS, ycor, &delta, tol, w, &mut ctest_data);
        content_mut(NLS).ctest_data = ctest_data;

        /* return if successful */
        if retval == 0 {
            return SUN_SUCCESS;
        } else if retval == SUN_NLS_SWITCH {
            return SUN_NLS_SWITCH;
        }

        /* check if the iterations should continue; otherwise increment the
        convergence failure count and return error flag */
        if retval != SUN_NLS_CONTINUE {
            content_mut(NLS).nconvfails += 1;
            return retval;
        }

        curiter += 1;
        content_mut(NLS).curiter = curiter;
    }

    /* if we've reached this point, then we exhausted the iteration limit */
    content_mut(NLS).nconvfails += 1;
    SUN_NLS_CONV_RECVR
}

pub fn SUNNonlinSolFree_FixedPoint(_NLS: &SUNNonlinearSolver) -> SUNErrCode {
    SUN_SUCCESS
}

pub fn SUNNonlinSolSetSysFn_FixedPoint(
    NLS: &SUNNonlinearSolver,
    SysFn: Option<SUNNonlinSolSysFn>,
) -> SUNErrCode {
    if SysFn.is_none() {
        return SUN_ERR_ARG_CORRUPT;
    }
    content_mut(NLS).Sys = SysFn;
    SUN_SUCCESS
}

pub fn SUNNonlinSolSetConvTestFn_FixedPoint(
    NLS: &SUNNonlinearSolver,
    CTestFn: Option<SUNNonlinSolConvTestFn>,
    ctest_data: Option<Box<dyn Any>>,
) -> SUNErrCode {
    if CTestFn.is_none() {
        return SUN_ERR_ARG_CORRUPT;
    }
    let mut content = content_mut(NLS);
    content.CTest = CTestFn;
    content.ctest_data = ctest_data;
    SUN_SUCCESS
}

pub fn SUNNonlinSolSetNormFn_FixedPoint(
    NLS: &SUNNonlinearSolver,
    NormFn: Option<SUNNonlinSolNormFn>,
    norm_fn_data: Option<Box<dyn Any>>,
) -> SUNErrCode {
    let mut content = content_mut(NLS);
    content.norm_fn = NormFn;
    content.norm_fn_data = norm_fn_data;
    SUN_SUCCESS
}

pub fn SUNNonlinSolSetGetUpdateNormFn_FixedPoint(
    NLS: &SUNNonlinearSolver,
    GetUpdateNormFn: Option<SUNNonlinSolGetUpdateNormFn>,
    getupdatenorm_data: Option<Box<dyn Any>>,
) -> SUNErrCode {
    let mut content = content_mut(NLS);
    content.getupdatenorm_fn = GetUpdateNormFn;
    content.getupdatenorm_data = getupdatenorm_data;
    SUN_SUCCESS
}

pub fn SUNNonlinSolSetMaxIters_FixedPoint(NLS: &SUNNonlinearSolver, maxiters: i32) -> SUNErrCode {
    content_mut(NLS).maxiters = maxiters;
    SUN_SUCCESS
}

pub fn SUNNonlinSolSetDamping_FixedPoint(NLS: &SUNNonlinearSolver, beta: sunrealtype) -> SUNErrCode {
    let mut content = content_mut(NLS);
    if beta < ONE {
        /* enable damping */
        content.beta = beta;
        content.damping = SUNTRUE;
    } else {
        /* disable damping */
        content.beta = ONE;
        content.damping = SUNFALSE;
    }
    SUN_SUCCESS
}

pub fn SUNNonlinSolGetNumIters_FixedPoint(NLS: &SUNNonlinearSolver, niters: &mut i64) -> SUNErrCode {
    *niters = content_mut(NLS).niters;
    SUN_SUCCESS
}

pub fn SUNNonlinSolGetCurIter_FixedPoint(NLS: &SUNNonlinearSolver, iter: &mut i32) -> SUNErrCode {
    *iter = content_mut(NLS).curiter;
    SUN_SUCCESS
}

pub fn SUNNonlinSolGetNumConvFails_FixedPoint(
    NLS: &SUNNonlinearSolver,
    nconvfails: &mut i64,
) -> SUNErrCode {
    *nconvfails = content_mut(NLS).nconvfails;
    SUN_SUCCESS
}

pub fn SUNNonlinSolGetSysFn_FixedPoint(
    NLS: &SUNNonlinearSolver,
    SysFn: &mut Option<SUNNonlinSolSysFn>,
) -> SUNErrCode {
    *SysFn = content_mut(NLS).Sys;
    SUN_SUCCESS
}

pub fn SUNNonlinSolSetOptions_FixedPoint(
    NLS: &SUNNonlinearSolver,
    NLSid: Option<&str>,
    file_name: Option<&str>,
    argv: &[String],
) -> SUNErrCode {
    if !(file_name.is_none() || file_name == Some("")) {
        return SUN_ERR_ARG_INCOMPATIBLE;
    }

    if !argv.is_empty() {
        let default_id = "sunnonlinearsolver";
        let id = match NLSid {
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
            if key == "damping" {
                idx += 1;
                let farg: sunrealtype =
                    crate::sundials_math::SUNStrToReal(argv[idx].trim());
                let retval = SUNNonlinSolSetDamping_FixedPoint(NLS, farg);
                if retval != SUN_SUCCESS {
                    return retval;
                }
                idx += 1;
                continue;
            }
            idx += 1;
        }
    }
    SUN_SUCCESS
}

/// C `AndersonAccelerate`: computes the Anderson-accelerated fixed point
/// iterate. `x` doubles as the temporary vector, exactly as upstream.
fn AndersonAccelerate(
    NLS: &SUNNonlinearSolver,
    gval: &N_Vector,
    x: &N_Vector,
    xold: &N_Vector,
    iter: i32,
) -> SUNErrCode {
    /* local shortcut variables (vtemp aliases x by design) */
    let vtemp = x.clone();
    let (maa, gold, fold, df, dg, Q, fv, damping, beta);
    let mut R;
    let mut gamma;
    let mut cvals;
    let mut ipt_map;
    {
        let mut content = content_mut(NLS);
        ipt_map = std::mem::take(&mut content.imap);
        maa = content.m;
        gold = content.gold.as_ref().expect("gold").clone();
        fold = content.fold.as_ref().expect("fold").clone();
        df = content.df.as_ref().expect("df").clone();
        dg = content.dg.as_ref().expect("dg").clone();
        Q = content.q.as_ref().expect("q").clone();
        cvals = std::mem::take(&mut content.cvals);
        R = std::mem::take(&mut content.R);
        gamma = std::mem::take(&mut content.gamma);
        fv = content.delta.as_ref().expect("delta").clone();
        damping = content.damping;
        beta = content.beta;
    }
    let maau = maa as usize;
    let mut Xvecs: Vec<N_Vector> = Vec::with_capacity(2 * (maau + 1));

    /* reset ipt_map, i_pt */
    for i in 0..maau {
        ipt_map[i] = 0;
    }
    let i_pt = (iter - 1 - ((iter - 1) / maa) * maa) as usize;

    /* update dg[i_pt], df[i_pt], fv, gold and fold */
    N_VLinearSum(ONE, gval, -ONE, xold, &fv);
    if iter > 0 {
        N_VLinearSum(ONE, gval, -ONE, &gold, &dg[i_pt]); /* dg_new = gval - gold */
        N_VLinearSum(ONE, &fv, -ONE, &fold, &df[i_pt]); /* df_new = fv - fold */
    }
    N_VScale(ONE, gval, &gold);
    N_VScale(ONE, &fv, &fold);

    /* on first iteration, just do basic fixed-point update */
    if iter == 0 {
        N_VScale(ONE, gval, x);
        /* restore taken arrays */
        let mut content = content_mut(NLS);
        content.imap = ipt_map;
        content.cvals = cvals;
        content.R = R;
        content.gamma = gamma;
        return SUN_SUCCESS;
    }

    /* update data structures based on current iteration index */
    if iter == 1 {
        /* second iteration */
        R[0] = N_VDotProd(&df[i_pt], &df[i_pt]);
        R[0] = SUNRsqrt(R[0]);
        N_VScale(ONE / R[0], &df[i_pt], &Q[i_pt]);
        ipt_map[0] = 0;
    } else if iter <= maa {
        /* another iteration before we've reached maa */
        N_VScale(ONE, &df[i_pt], &vtemp);
        let iu = iter as usize;
        for j in 0..(iu - 1) {
            ipt_map[j] = j as i32;
            R[(iu - 1) * maau + j] = N_VDotProd(&Q[j], &vtemp);
            N_VLinearSum(ONE, &vtemp, -R[(iu - 1) * maau + j], &Q[j], &vtemp);
        }
        R[(iu - 1) * maau + iu - 1] = N_VDotProd(&vtemp, &vtemp);
        R[(iu - 1) * maau + iu - 1] = SUNRsqrt(R[(iu - 1) * maau + iu - 1]);
        if R[(iu - 1) * maau + iu - 1] == ZERO {
            N_VScale(ZERO, &vtemp, &Q[i_pt]);
        } else {
            N_VScale(ONE / R[(iu - 1) * maau + iu - 1], &vtemp, &Q[i_pt]);
        }
        ipt_map[iu - 1] = (iu - 1) as i32;
    } else {
        /* we've filled the acceleration subspace, so start recycling */

        /* delete left-most column vector from QR factorization */
        for i in 0..(maau - 1) {
            let a = R[(i + 1) * maau + i];
            let b = R[(i + 1) * maau + i + 1];
            let rtemp = SUNRsqrt(a * a + b * b);
            let c = a / rtemp;
            let s = b / rtemp;
            R[(i + 1) * maau + i] = rtemp;
            R[(i + 1) * maau + i + 1] = ZERO;
            if i < maau - 1 {
                for j in (i + 2)..maau {
                    let a = R[j * maau + i];
                    let b = R[j * maau + i + 1];
                    let rtemp = c * a + s * b;
                    R[j * maau + i + 1] = -s * a + c * b;
                    R[j * maau + i] = rtemp;
                }
            }
            N_VLinearSum(c, &Q[i], s, &Q[i + 1], &vtemp);
            N_VLinearSum(-s, &Q[i], c, &Q[i + 1], &Q[i + 1]);
            N_VScale(ONE, &vtemp, &Q[i]);
        }

        /* shift R to the left by one */
        for i in 1..maau {
            for j in 0..(maau - 1) {
                R[(i - 1) * maau + j] = R[i * maau + j];
            }
        }

        /* add the new df vector */
        N_VScale(ONE, &df[i_pt], &vtemp);
        for j in 0..(maau - 1) {
            R[(maau - 1) * maau + j] = N_VDotProd(&Q[j], &vtemp);
            N_VLinearSum(ONE, &vtemp, -R[(maau - 1) * maau + j], &Q[j], &vtemp);
        }
        R[(maau - 1) * maau + maau - 1] = N_VDotProd(&vtemp, &vtemp);
        R[(maau - 1) * maau + maau - 1] = SUNRsqrt(R[(maau - 1) * maau + maau - 1]);
        N_VScale(ONE / R[(maau - 1) * maau + maau - 1], &vtemp, &Q[maau - 1]);

        /* update the iteration map */
        let mut j = 0usize;
        for i in (i_pt + 1)..maau {
            ipt_map[j] = i as i32;
            j += 1;
        }
        for i in 0..(i_pt + 1) {
            ipt_map[j] = i as i32;
            j += 1;
        }
    }

    /* solve least squares problem and update solution */
    let lAA = if maa < iter { maa } else { iter };
    let ier = N_VDotProdMulti(lAA, &fv, &Q, &mut gamma);
    if ier != SUN_SUCCESS {
        let mut content = content_mut(NLS);
        content.imap = ipt_map;
        content.cvals = cvals;
        content.R = R;
        content.gamma = gamma;
        return ier;
    }

    /* set arrays for fused vector operation */
    cvals[0] = ONE;
    Xvecs.push(gval.clone());
    let mut nvec = 1usize;
    let lAAu = lAA as usize;
    let mut i = lAAu as i64 - 1;
    while i > -1 {
        let iu = i as usize;
        for j in (iu + 1)..lAAu {
            gamma[iu] -= R[j * maau + iu] * gamma[j];
        }
        if gamma[iu] == ZERO {
            gamma[iu] = ZERO;
        } else {
            gamma[iu] /= R[iu * maau + iu];
        }
        cvals[nvec] = -gamma[iu];
        Xvecs.push(dg[ipt_map[iu] as usize].clone());
        nvec += 1;
        i -= 1;
    }

    /* if enabled, apply damping */
    if damping {
        let onembeta = ONE - beta;
        cvals[nvec] = -onembeta;
        Xvecs.push(fv.clone());
        nvec += 1;
        let mut i = lAAu as i64 - 1;
        while i > -1 {
            let iu = i as usize;
            cvals[nvec] = onembeta * gamma[iu];
            Xvecs.push(df[ipt_map[iu] as usize].clone());
            nvec += 1;
            i -= 1;
        }
    }

    /* update solution */
    let ier = N_VLinearCombination(nvec as i32, &cvals, &Xvecs, x);

    /* restore taken arrays */
    let mut content = content_mut(NLS);
    content.imap = ipt_map;
    content.cvals = cvals;
    content.R = R;
    content.gamma = gamma;

    if ier != SUN_SUCCESS {
        return ier;
    }
    SUN_SUCCESS
}
