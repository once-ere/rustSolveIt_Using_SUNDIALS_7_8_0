//! Port of `src/sundomeigest/power/sundomeigest_power.c` +
//! `include/sundomeigest/sundomeigest_power.h` (Power Iteration dominant
//! eigenvalue estimator).
//!
//! The C `PI_CONTENT(DEE)->field` accessor macro becomes a field read/write
//! through the private `content_mut` guard. Following the granular-borrow
//! rule, no content borrow is ever held across an `ATimes` or `rhsfn`
//! callback: `dee_call_atimes`/`dee_call_rhsfn` copy the fn pointer out,
//! `Option::take` the `void*` payload box, invoke, then restore.
//!
//! `SUNDomEigEstimator_SetRhs_Power` installs `dee_DQJtimes_Power` with the
//! DEE itself as `A_data` (C: `(void*)DEE`); that is an owning `Rc` clone
//! stored inside the DEE's own content, i.e. a reference cycle exactly
//! mirroring C's self-pointer. `SUNDomEigEstimator_Destroy_Power` clears the
//! content (C: `free(DEE->content)`), which breaks the cycle; skipping
//! Destroy leaks, precisely as it does in C.
//!
//! Release-mode `SUNAssert`/`SUNCheck*` sites are no-ops per the build
//! config and are omitted; `SUNLog*` calls compile away at logging level 2.

use std::any::Any;
use std::cell::RefMut;

use crate::sundials_context::SUNContext;
use crate::sundials_domeigestimator::*;
use crate::sundials_errors::{SUN_ERR_ARG_CORRUPT, SUN_ERR_USER_FCN_FAIL, SUN_SUCCESS};
use crate::sundials_linearsolver::SUNATimesFn;
use crate::sundials_math::{SUNRabs, SUNRsqrt, SUNMAX};
use crate::sundials_nvector::*;
use crate::sundials_types::*;
use crate::sundials_utils::{sun_format_g, SUNFile};

const MAX_DQITERS: i32 = 3;

const ZERO: sunrealtype = 0.0;
const ONE: sunrealtype = 1.0;

/* Default estimator parameters */
const DEE_NUM_OF_WARMUPS_PI_DEFAULT: i32 = 100;

/* Default Power Iteration parameters */
const DEE_TOL_DEFAULT: sunrealtype = 0.005;
const DEE_MAX_ITER_DEFAULT: i64 = 100;

/* -----------------------------------------------------
 * Power Iteration Implementation of SUNDomEigEstimator
 * ----------------------------------------------------- */

pub struct SUNDomEigEstimatorContent_Power_ {
    pub ATimes: Option<SUNATimesFn>,  /* User provided ATimes function */
    pub ATdata: Option<Box<dyn Any>>, /* ATimes function data*/

    /* workspace vectors */
    pub V: Option<N_Vector>,
    pub q: Option<N_Vector>,
    pub q_prev: Option<N_Vector>,
    pub rhs_linY: Option<N_Vector>,
    pub Fy: Option<N_Vector>,
    pub work: Option<N_Vector>,

    pub num_warmups: i32,      /* Number of preprocessing iterations */
    pub max_iters: i64,        /* Maximum number of power iterations */
    pub num_iters: i64,        /* Number of iterations in last Estimate call */
    pub rhs_linT: sunrealtype, /* Time value for linearization point */

    pub num_ATimes: i64, /* Number of ATimes calls */

    pub rel_tol: sunrealtype, /* Convergence criteria for the power iteration */
    pub res: sunrealtype,     /* Residual from the last Estimate call */

    pub rhsfn: Option<SUNRhsFn>,        /* User provided RHS function */
    pub rhs_data: Option<Box<dyn Any>>, /* RHS function data */
    pub nfevals: i64,                   /* Number of RHS evaluations */

    pub is_complex: sunbooleantype, /* Flag for complex eigenvalue request */
}

pub type SUNDomEigEstimatorContent_Power = SUNDomEigEstimatorContent_Power_;

/// C `PI_CONTENT(DEE)`.
fn content_mut(DEE: &SUNDomEigEstimator) -> RefMut<'_, SUNDomEigEstimatorContent_Power_> {
    RefMut::map(DEE.content.borrow_mut(), |c| {
        c.downcast_mut::<SUNDomEigEstimatorContent_Power_>()
            .expect("Power SUNDomEigEstimator content")
    })
}

/*
* --------------------------------------------------------------------------
* private helpers (borrow-safe callback invocation)
* --------------------------------------------------------------------------
*/

/// One C `PI_CONTENT(DEE)->ATimes(PI_CONTENT(DEE)->ATdata, v, z)` call.
///
/// The fn pointer and the `void*` payload are lifted out of the content
/// before the call and the payload is restored afterwards, so the callback
/// (which may be `dee_DQJtimes_Power` re-entering this very DEE) never meets
/// an outstanding borrow. A missing `ATimes` is C's NULL fn-pointer deref.
fn dee_call_atimes(DEE: &SUNDomEigEstimator, v: &N_Vector, z: &N_Vector) -> i32 {
    let (ATimes, mut ATdata) = {
        let mut content = content_mut(DEE);
        (content.ATimes, content.ATdata.take())
    };
    let ATimes = ATimes.expect("Power SUNDomEigEstimator ATimes");
    let retval = ATimes(&mut ATdata, v, z);
    content_mut(DEE).ATdata = ATdata;
    retval
}

/// One C `PI_CONTENT(DEE)->rhsfn(PI_CONTENT(DEE)->rhs_linT, y, ydot,
/// PI_CONTENT(DEE)->rhs_data)` call plus the `nfevals++` that always follows
/// it in the C source. `rhs_linT` is re-read from the content at every call,
/// exactly as the C macro expansion does.
fn dee_call_rhsfn(DEE: &SUNDomEigEstimator, y: &N_Vector, ydot: &N_Vector) -> i32 {
    let (rhs_linT, rhsfn, mut rhs_data) = {
        let mut content = content_mut(DEE);
        (content.rhs_linT, content.rhsfn, content.rhs_data.take())
    };
    let rhsfn = rhsfn.expect("Power SUNDomEigEstimator rhsfn");
    let retval = rhsfn(rhs_linT, y, ydot, &mut rhs_data);
    let mut content = content_mut(DEE);
    content.rhs_data = rhs_data;
    content.nfevals += 1;
    retval
}

/* ----------------------------------------------------------------------------
 * Function to create a new PI estimator
 */

pub fn SUNDomEigEstimator_Power(
    q: &N_Vector,
    max_iters: i64,
    rel_tol: sunrealtype,
    sunctx: &SUNContext,
) -> Option<SUNDomEigEstimator> {
    /* Check for required vector operations (C `SUNAssertNull`; kept as a
    live check per accepted deviation class 1) */
    {
        let ops = q.ops.borrow();
        if ops.nvclone.is_none()
            || ops.nvdestroy.is_none()
            || ops.nvdotprod.is_none()
            || ops.nvscale.is_none()
        {
            return None;
        }
    }

    /* check for max_iters values; if illegal use defaults */
    let max_iters = if max_iters <= 0 {
        DEE_MAX_ITER_DEFAULT
    } else {
        max_iters
    };

    /* Check if rel_tol > 0 and < 1 */
    let rel_tol = if rel_tol < SUN_SMALL_REAL || rel_tol > ONE - SUN_UNIT_ROUNDOFF {
        DEE_TOL_DEFAULT
    } else {
        rel_tol
    };

    /* Create dominant eigenvalue estimator */
    let DEE = SUNDomEigEstimator_NewEmpty(sunctx)?;

    /* Attach operations */
    {
        let mut ops = DEE.ops.borrow_mut();
        ops.setatimes = Some(SUNDomEigEstimator_SetATimes_Power);
        ops.setrhs = Some(SUNDomEigEstimator_SetRhs_Power);
        ops.setrhslinearizationpoint = Some(SUNDomEigEstimator_SetRhsLinearizationPoint_Power);
        ops.setmaxiters = Some(SUNDomEigEstimator_SetMaxIters_Power);
        ops.setnumpreprocessiters = Some(SUNDomEigEstimator_SetNumPreprocessIters_Power);
        ops.setreltol = Some(SUNDomEigEstimator_SetRelTol_Power);
        ops.setinitialguess = Some(SUNDomEigEstimator_SetInitialGuess_Power);
        ops.initialize = Some(SUNDomEigEstimator_Initialize_Power);
        ops.estimate = Some(SUNDomEigEstimator_Estimate_Power);
        ops.getres = Some(SUNDomEigEstimator_GetRes_Power);
        ops.getnumiters = Some(SUNDomEigEstimator_GetNumIters_Power);
        ops.getnumrhsevals = Some(SUNDomEigEstimator_GetNumRhsEvals_Power);
        ops.getnumatimescalls = Some(SUNDomEigEstimator_GetNumATimesCalls_Power);
        ops.write = Some(SUNDomEigEstimator_Write_Power);
        ops.destroy = Some(SUNDomEigEstimator_Destroy_Power);
    }

    /* Create content and attach it, filling every field */
    *DEE.content.borrow_mut() = Box::new(SUNDomEigEstimatorContent_Power_ {
        ATimes: None,
        ATdata: None,
        V: None,
        q: None,
        q_prev: None,
        rhs_linY: None,
        rhs_linT: ZERO,
        Fy: None,
        work: None,
        is_complex: SUNTRUE,
        max_iters,
        num_warmups: DEE_NUM_OF_WARMUPS_PI_DEFAULT,
        rel_tol,
        res: ZERO,
        rhsfn: None,
        rhs_data: None,
        nfevals: 0,
        num_iters: 0,
        num_ATimes: 0,
    });

    /* Allocate content */
    let content_q = N_VClone(q)?;
    content_mut(&DEE).q = Some(content_q);

    let content_V = N_VClone(q)?;
    content_mut(&DEE).V = Some(content_V);

    /* Initialize the vector V */
    let mut normq = N_VDotProd(q, q);

    normq = SUNRsqrt(normq);

    let V = content_mut(&DEE).V.clone().expect("V");
    N_VScale(ONE / normq, q, &V);

    Some(DEE)
}

/*
 * -----------------------------------------------------------------
 * implementation of dominant eigenvalue estimator operations
 * -----------------------------------------------------------------
 */

pub fn SUNDomEigEstimator_SetATimes_Power(
    DEE: &SUNDomEigEstimator,
    A_data: Option<Box<dyn Any>>,
    ATimes: Option<SUNATimesFn>,
) -> SUNErrCode {
    /* set function pointers to integrator-supplied ATimes routine
    and data, and return with success */
    let mut content = content_mut(DEE);
    content.ATimes = ATimes;
    content.ATdata = A_data;
    SUN_SUCCESS
}

pub fn SUNDomEigEstimator_SetRhs_Power(
    DEE: &SUNDomEigEstimator,
    rhs_data: Option<Box<dyn Any>>,
    RHSfn: Option<SUNRhsFn>,
) -> SUNErrCode {
    /* set function pointers to integrator-supplied RHS routine
    and data, and return with success */
    {
        let mut content = content_mut(DEE);
        content.rhsfn = RHSfn;
        content.rhs_data = rhs_data;
    }

    /* C: DEE->ops->setatimes(DEE, (void*)DEE, dee_DQJtimes_Power) */
    let setatimes = DEE.ops.borrow().setatimes;
    let setatimes = setatimes.expect("Power SUNDomEigEstimator setatimes");
    let _ = setatimes(
        DEE,
        Some(Box::new(DEE.clone())),
        Some(dee_DQJtimes_Power as SUNATimesFn),
    );

    SUN_SUCCESS
}

pub fn SUNDomEigEstimator_SetRhsLinearizationPoint_Power(
    DEE: &SUNDomEigEstimator,
    t: sunrealtype,
    y: &N_Vector,
) -> SUNErrCode {
    let need_clone = content_mut(DEE).rhs_linY.is_none();
    if need_clone {
        let cloned = N_VClone(y);
        content_mut(DEE).rhs_linY = cloned;
    }

    let rhs_linY = {
        let mut content = content_mut(DEE);
        content.rhs_linT = t;
        content.rhs_linY.clone().expect("rhs_linY")
    };

    N_VScale(ONE, y, &rhs_linY);

    SUN_SUCCESS
}

pub fn SUNDomEigEstimator_SetIsReal_Power(
    DEE: &SUNDomEigEstimator,
    real: sunbooleantype,
) -> SUNErrCode {
    /* q_prev is allocated in SUNDomEigEstimator_Initialize_Power, which is expected to be
    called after this routine. If the user calls this routine after initialization, we need
    to free q_prev here. */
    let stale_q_prev = {
        let mut content = content_mut(DEE);

        /* set the complex flag to the opposite of the real flag */
        content.is_complex = !real;

        if !content.is_complex {
            content.q_prev.take()
        } else {
            None
        }
    };
    if let Some(q_prev) = stale_q_prev {
        N_VDestroy(q_prev);
    }

    SUN_SUCCESS
}

pub fn SUNDomEigEstimator_Initialize_Power(DEE: &SUNDomEigEstimator) -> SUNErrCode {
    let mut content = content_mut(DEE);

    if content.rel_tol < SUN_SMALL_REAL || content.rel_tol > ONE - SUN_UNIT_ROUNDOFF {
        content.rel_tol = DEE_TOL_DEFAULT;
    }
    if content.num_warmups < 0 {
        content.num_warmups = DEE_NUM_OF_WARMUPS_PI_DEFAULT;
    }
    if content.max_iters <= 0 {
        content.max_iters = DEE_MAX_ITER_DEFAULT;
    }

    SUN_SUCCESS
}

pub fn SUNDomEigEstimator_SetNumPreprocessIters_Power(
    DEE: &SUNDomEigEstimator,
    num_iters: i32,
) -> SUNErrCode {
    /* Check if num_iters >= 0 */
    let num_iters = if num_iters < 0 {
        DEE_NUM_OF_WARMUPS_PI_DEFAULT
    } else {
        num_iters
    };

    /* set the number of warmups */
    content_mut(DEE).num_warmups = num_iters;
    SUN_SUCCESS
}

pub fn SUNDomEigEstimator_SetRelTol_Power(
    DEE: &SUNDomEigEstimator,
    rel_tol: sunrealtype,
) -> SUNErrCode {
    /* Check if rel_tol > 0 and < 1 */
    let rel_tol = if rel_tol < SUN_SMALL_REAL || rel_tol > ONE - SUN_UNIT_ROUNDOFF {
        DEE_TOL_DEFAULT
    } else {
        rel_tol
    };

    /* set the tolerance */
    content_mut(DEE).rel_tol = rel_tol;
    SUN_SUCCESS
}

pub fn SUNDomEigEstimator_SetMaxIters_Power(
    DEE: &SUNDomEigEstimator,
    max_iters: i64,
) -> SUNErrCode {
    /* Check for legal number of iters */
    let max_iters = if max_iters <= 0 {
        DEE_MAX_ITER_DEFAULT
    } else {
        max_iters
    };

    /* Set max iters */
    content_mut(DEE).max_iters = max_iters;
    SUN_SUCCESS
}

pub fn SUNDomEigEstimator_SetInitialGuess_Power(
    DEE: &SUNDomEigEstimator,
    q: &N_Vector,
) -> SUNErrCode {
    let mut normq = N_VDotProd(q, q);

    normq = SUNRsqrt(normq);

    /* set the initial guess */
    let V = content_mut(DEE).V.clone().expect("V");
    N_VScale(ONE / normq, q, &V);

    SUN_SUCCESS
}

pub fn SUNDomEigEstimator_Estimate_Power(
    DEE: &SUNDomEigEstimator,
    lambdaR: &mut sunrealtype,
    lambdaI: &mut sunrealtype,
) -> SUNErrCode {
    let (is_complex, need_q_prev) = {
        let content = content_mut(DEE);
        (content.is_complex, content.q_prev.is_none())
    };
    if is_complex && need_q_prev {
        /* allocate q_prev vector */
        let q = content_mut(DEE).q.clone().expect("q");
        let cloned = N_VClone(&q);
        content_mut(DEE).q_prev = cloned;
    }

    let mut newlambdaR: sunrealtype = ZERO;
    let mut oldlambdaR: sunrealtype = ZERO;

    /* C leaves `normq` uninitialized; every path that reads it below
    (the is_complex branch) assigns it first, because max_iters >= 1 is
    enforced by the constructor, SetMaxIters and Initialize. */
    let mut normq: sunrealtype = ZERO;

    let (num_warmups, max_iters, rel_tol, V, q, q_prev) = {
        let mut content = content_mut(DEE);
        content.num_ATimes = 0;
        content.num_iters = 0;
        (
            content.num_warmups,
            content.max_iters,
            content.rel_tol,
            content.V.clone().expect("V"),
            content.q.clone().expect("q"),
            content.q_prev.clone(),
        )
    };

    /* Set the initial q = A^{num_warmups}q/||A^{num_warmups}q|| */
    for _i in 0..num_warmups {
        let retval = dee_call_atimes(DEE, &V, &q);
        {
            let mut content = content_mut(DEE);
            content.num_ATimes += 1;
            content.num_iters += 1;
        }
        if retval != 0 {
            return SUN_ERR_USER_FCN_FAIL;
        }

        normq = N_VDotProd(&q, &q);

        normq = SUNRsqrt(normq);
        N_VScale(ONE / normq, &q, &V);
    }

    for _k in 0..max_iters {
        if is_complex {
            N_VScale(ONE, &V, q_prev.as_ref().expect("q_prev"));
        }

        let retval = dee_call_atimes(DEE, &V, &q);
        {
            let mut content = content_mut(DEE);
            content.num_ATimes += 1;
            content.num_iters += 1;
        }
        if retval != 0 {
            return SUN_ERR_USER_FCN_FAIL;
        }

        newlambdaR = N_VDotProd(&V, &q); //Rayleigh quotient

        let res = SUNRabs(newlambdaR - oldlambdaR);
        content_mut(DEE).res = res;
        let converged: sunbooleantype = res <= rel_tol * SUNRabs(newlambdaR);

        if converged && !is_complex {
            break;
        }

        normq = N_VDotProd(&q, &q);

        normq = SUNRsqrt(normq);
        N_VScale(ONE / normq, &q, &V);

        if converged {
            break;
        }

        oldlambdaR = newlambdaR;
    }

    if is_complex {
        let retval = sundomeigestimator_complex_dom_eigs_from_PI(
            DEE,
            newlambdaR,
            normq,
            q_prev.as_ref().expect("q_prev"),
            &V,
            lambdaR,
            lambdaI,
        );
        if retval != 0 {
            return SUN_ERR_USER_FCN_FAIL;
        }
    } else {
        *lambdaR = newlambdaR;
        *lambdaI = ZERO;
    }

    SUN_SUCCESS
}

pub fn SUNDomEigEstimator_GetRes_Power(
    DEE: &SUNDomEigEstimator,
    res: &mut sunrealtype,
) -> SUNErrCode {
    *res = content_mut(DEE).res;

    SUN_SUCCESS
}

pub fn SUNDomEigEstimator_GetNumIters_Power(
    DEE: &SUNDomEigEstimator,
    num_iters: &mut i64,
) -> SUNErrCode {
    *num_iters = content_mut(DEE).num_iters;

    SUN_SUCCESS
}

pub fn SUNDomEigEstimator_GetNumRhsEvals_Power(
    DEE: &SUNDomEigEstimator,
    num_rhs_evals: &mut i64,
) -> SUNErrCode {
    *num_rhs_evals = content_mut(DEE).nfevals;

    SUN_SUCCESS
}

pub fn SUNDomEigEstimator_GetNumATimesCalls_Power(
    DEE: &SUNDomEigEstimator,
    num_ATimes: &mut i64,
) -> SUNErrCode {
    *num_ATimes = content_mut(DEE).num_ATimes;

    SUN_SUCCESS
}

pub fn SUNDomEigEstimator_Write_Power(DEE: &SUNDomEigEstimator, outfile: &SUNFile) -> SUNErrCode {
    /* C: `if (DEE == NULL || outfile == NULL) return SUN_ERR_ARG_CORRUPT;`
    the DEE half cannot fire — the handle is non-NULL by construction. */
    if outfile.is_null() {
        return SUN_ERR_ARG_CORRUPT;
    }

    let content = content_mut(DEE);

    outfile.write_str("\nPower Iteration SUNDomEigEstimator:\n");
    outfile.write_str(&format!("Max. iters               = {}\n", content.max_iters));
    outfile.write_str(&format!("Num. preprocessing iters = {}\n", content.num_warmups));
    outfile.write_str(&format!("Relative tolerance       = {}\n", sun_format_g(content.rel_tol)));
    outfile.write_str(&format!("Residual                 = {}\n", sun_format_g(content.res)));
    outfile.write_str(&format!("Num. iters               = {}\n", content.num_iters));
    outfile.write_str(&format!("Num. ATimes calls        = {}\n\n", content.num_ATimes));

    SUN_SUCCESS
}

pub fn sundomeigestimator_complex_dom_eigs_from_PI(
    DEE: &SUNDomEigEstimator,
    lambdaR: sunrealtype,
    h21: sunrealtype,
    v_prev: &N_Vector,
    v: &N_Vector,
    lambdaR_out: &mut sunrealtype,
    lambdaI_out: &mut sunrealtype,
) -> SUNErrCode {
    /* The threshold for identifying real or complex DEE is experimentally
    determined based on the relative tolerance PI_CONTENT(DEE)->rel_tol */
    let rel_tol = content_mut(DEE).rel_tol;
    let gram_det_tol: sunrealtype = SUNMAX(10.0 * SUN_UNIT_ROUNDOFF, 10.0 * rel_tol);
    let mut cos_qs: sunrealtype = N_VDotProd(v_prev, v);

    /* Safety against roundoff in dot product */
    if cos_qs > ONE {
        cos_qs = ONE;
    }
    if cos_qs < -ONE {
        cos_qs = -ONE;
    }

    /* Use Gram determinant as the near-dependence measure:
       G = [ [1, cos_qs], [cos_qs, 1] ], det(G) = 1 - cos_qs^2
       This assumes v_prev and v are normalized. */
    let gram_det: sunrealtype = ONE - cos_qs * cos_qs;

    if gram_det <= gram_det_tol {
        /* Dominant eigenvalue is real */
        *lambdaR_out = lambdaR;
        *lambdaI_out = ZERO;
        return SUN_SUCCESS;
    } else {
        let det_G_inv: sunrealtype = ONE / gram_det;

        /* Solve for G = [v_prev v]' * [v_prev v] and compute
           projected matrix P = G^{-1} * [v_prev v]' * A * [v_prev v] */

        let h11: sunrealtype = lambdaR;

        let q = content_mut(DEE).q.clone().expect("q");
        let retval = dee_call_atimes(DEE, v, &q);
        content_mut(DEE).num_ATimes += 1;
        if retval != 0 {
            return SUN_ERR_USER_FCN_FAIL;
        }

        let h12: sunrealtype = N_VDotProd(v_prev, &q);
        let h22: sunrealtype = N_VDotProd(v, &q);

        let p11: sunrealtype = det_G_inv * (h11 - cos_qs * h21);
        let p12: sunrealtype = det_G_inv * (h12 - cos_qs * h22);
        let p21: sunrealtype = det_G_inv * (h21 - cos_qs * h11);
        let p22: sunrealtype = det_G_inv * (h22 - cos_qs * h12);

        /* Compute eigenvalues of P */
        let traceP: sunrealtype = p11 + p22;
        let detP: sunrealtype = p11 * p22 - p12 * p21;
        let discrim: sunrealtype = traceP * traceP - 4.0 * detP;
        if discrim >= ZERO {
            /* Dominant eigenvalue is real */
            let sqrt_discrim: sunrealtype = SUNRsqrt(discrim);
            let lam_plus: sunrealtype = (traceP + sqrt_discrim) / 2.0;
            let lam_minus: sunrealtype = (traceP - sqrt_discrim) / 2.0;
            if SUNRabs(lam_plus) >= SUNRabs(lam_minus) {
                *lambdaR_out = lam_plus;
            } else {
                *lambdaR_out = lam_minus;
            }
            /* C assigns ZERO here twice; the repeat is a no-op */
            *lambdaI_out = ZERO;
        } else {
            /* Dominant eigenvalue is complex */
            *lambdaR_out = traceP / 2.0;
            *lambdaI_out = SUNRsqrt(-discrim) / 2.0;
        }
    }

    SUN_SUCCESS
}

pub fn SUNDomEigEstimator_Destroy_Power(DEEptr: &mut Option<SUNDomEigEstimator>) -> SUNErrCode {
    let DEE = match DEEptr.take() {
        None => return SUN_SUCCESS,
        Some(DEE) => DEE,
    };

    let has_content = DEE
        .content
        .borrow()
        .is::<SUNDomEigEstimatorContent_Power_>();

    if has_content {
        /* delete items from within the content structure */
        let (q, q_prev, V, rhs_linY, Fy, work) = {
            let mut content = content_mut(&DEE);
            (
                content.q.take(),
                content.q_prev.take(),
                content.V.take(),
                content.rhs_linY.take(),
                content.Fy.take(),
                content.work.take(),
            )
        };
        if let Some(q) = q {
            N_VDestroy(q);
        }
        if let Some(q_prev) = q_prev {
            N_VDestroy(q_prev);
        }
        if let Some(V) = V {
            N_VDestroy(V);
        }
        if let Some(rhs_linY) = rhs_linY {
            N_VDestroy(rhs_linY);
        }
        if let Some(Fy) = Fy {
            N_VDestroy(Fy);
        }
        if let Some(work) = work {
            N_VDestroy(work);
        }
        /* C: free(DEE->content); DEE->content = NULL;  Dropping the content
        box also drops ATdata, which for the SetRhs path owns an Rc clone of
        this very DEE — that is what breaks the self-reference cycle. */
        *DEE.content.borrow_mut() = Box::new(());
    }

    /* C: free(DEE->ops); free(DEE); *DEEptr = NULL;  ops lives inside the
    handle, so dropping the handle releases both. */
    drop(DEE);
    SUN_SUCCESS
}

/*---------------------------------------------------------------
  dee_DQJtimes_Power:

  This routine generates a difference quotient approximation to
  the Jacobian-vector product f_y(t,y) * v. The approximation is
  Jv = [f(y + v*sig) - f(y)]/sig, where
      sig = sign(y^T v) * sqrt(unit roundoff)
            * max(|y^T v|, ||v||_1) / (v^T v).
  ---------------------------------------------------------------*/
pub fn dee_DQJtimes_Power(
    voidstarDEE: &mut Option<Box<dyn Any>>,
    v: &N_Vector,
    Jv: &N_Vector,
) -> SUNErrCode {
    let DEE: SUNDomEigEstimator = voidstarDEE
        .as_ref()
        .and_then(|data| data.downcast_ref::<SUNDomEigEstimator>())
        .expect("dee_DQJtimes_Power A_data")
        .clone();
    let DEE = &DEE;

    let vdotv = N_VDotProd(v, v);
    if vdotv <= SUN_SMALL_REAL {
        N_VScale(ZERO, v, Jv);
        return SUN_SUCCESS;
    }

    let need_work = content_mut(DEE).work.is_none();
    if need_work {
        let cloned = N_VClone(v);
        content_mut(DEE).work = cloned;
    }
    let need_Fy = content_mut(DEE).Fy.is_none();
    if need_Fy {
        let cloned = N_VClone(v);
        content_mut(DEE).Fy = cloned;
    }

    let (y, work, Fy) = {
        let content = content_mut(DEE);
        (
            content.rhs_linY.clone().expect("rhs_linY"),
            content.work.clone().expect("work"),
            content.Fy.clone().expect("Fy"),
        )
    };

    let mut retval = dee_call_rhsfn(DEE, &y, &Fy);
    if retval != 0 {
        return SUN_ERR_USER_FCN_FAIL;
    }

    /* Initialize perturbation */
    let ydotv = N_VDotProd(&y, v);
    let sq1norm = N_VL1Norm(v);
    let sign: sunrealtype = if ydotv >= ZERO { ONE } else { -ONE };
    let sqrteps = SUNRsqrt(SUN_UNIT_ROUNDOFF);
    let mut sig: sunrealtype = sign * sqrteps * SUNMAX(SUNRabs(ydotv), sq1norm) / vdotv;

    for _iter in 0..MAX_DQITERS {
        /* Set work = y + sig*v */
        N_VLinearSum(sig, v, ONE, &y, &work);

        /* Set Jv = f(tn, y+sig*v) */
        retval = dee_call_rhsfn(DEE, &work, Jv);
        if retval == 0 {
            break;
        }
        if retval < 0 {
            return SUN_ERR_USER_FCN_FAIL;
        }

        /* If f failed recoverably, shrink sig and retry */
        sig *= 0.25;
    }

    /* If retval still isn't 0, return with a recoverable failure */
    if retval > 0 {
        return 1;
    }

    /* Replace Jv by (Jv - fn)/sig */
    let siginv: sunrealtype = ONE / sig;
    N_VLinearSum(siginv, Jv, -siginv, &Fy, Jv);

    SUN_SUCCESS
}
