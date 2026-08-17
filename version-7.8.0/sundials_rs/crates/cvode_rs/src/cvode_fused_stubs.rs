//! Port of `src/cvode/cvode_fused_stubs.c` (fused stub kernels for
//! CVODE).
//!
//! Upstream this file is compiled only when
//! `SUNDIALS_ENABLE_PACKAGE_FUSED_KERNELS` is ON (it becomes the
//! `sundials_cvode_fused_stubs` library). The reference build has fused
//! kernels OFF, so nothing in the workspace calls these functions; they
//! are ported faithfully as available-but-unused code so the module is
//! complete. The C `SUNDIALS_MAYBE_UNUSED` parameter maps to a leading
//! underscore (`_fract`).

use sundials_core::sundials_nvector::{
    N_VAbs, N_VAddConst, N_VCompare, N_VDiv, N_VInv, N_VLinearSum, N_VMin, N_VProd, N_VScale,
    N_Vector,
};
use sundials_core::sundials_types::{sunbooleantype, sunrealtype};

const ZERO: sunrealtype = 0.0;
const PT1: sunrealtype = 0.1;
const FRACT: sunrealtype = 0.1;
const ONEPT5: sunrealtype = 1.50;
const ONE: sunrealtype = 1.0;

/*
 * -----------------------------------------------------------------
 * Compute the ewt vector when the tol type is CV_SS.
 * -----------------------------------------------------------------
 */

pub fn cvEwtSetSS_fused(
    atolmin0: sunbooleantype,
    reltol: sunrealtype,
    Sabstol: sunrealtype,
    ycur: &N_Vector,
    tempv: &N_Vector,
    weight: &N_Vector,
) -> i32 {
    N_VAbs(ycur, tempv);
    N_VScale(reltol, tempv, tempv);
    N_VAddConst(tempv, Sabstol, tempv);
    if atolmin0 {
        if N_VMin(tempv) <= ZERO {
            return -1;
        }
    }
    N_VInv(tempv, weight);
    0
}

/*
 * -----------------------------------------------------------------
 * Compute the ewt vector when the tol type is CV_SV.
 * -----------------------------------------------------------------
 */

pub fn cvEwtSetSV_fused(
    atolmin0: sunbooleantype,
    reltol: sunrealtype,
    Vabstol: &N_Vector,
    ycur: &N_Vector,
    tempv: &N_Vector,
    weight: &N_Vector,
) -> i32 {
    N_VAbs(ycur, tempv);
    N_VLinearSum(reltol, tempv, ONE, Vabstol, tempv);
    if atolmin0 {
        if N_VMin(tempv) <= ZERO {
            return -1;
        }
    }
    N_VInv(tempv, weight);
    0
}

/*
 * -----------------------------------------------------------------
 * Determine if the constraints of the problem are satisfied by
 * the proposed step.
 * -----------------------------------------------------------------
 */

pub fn cvCheckConstraints_fused(
    c: &N_Vector,
    ewt: &N_Vector,
    y: &N_Vector,
    mm: &N_Vector,
    tmp: &N_Vector,
    save: &N_Vector,
) -> i32 {
    N_VCompare(ONEPT5, c, tmp); /* a[i]=1 when |c[i]|=2  */
    N_VProd(tmp, c, tmp); /* a * c                 */
    N_VDiv(tmp, ewt, tmp); /* a * c * wt            */
    N_VScale(-PT1, tmp, save);
    N_VLinearSum(ONE, y, -PT1, tmp, tmp); /* y - 0.1 * a * c * wt  */
    N_VProd(tmp, mm, tmp); /* v = mm*(y-0.1*a*c*wt) */
    0
}

/*
 * -----------------------------------------------------------------
 * Compute the nonlinear residual.
 * -----------------------------------------------------------------
 */

pub fn cvNlsResid_fused(
    rl1: sunrealtype,
    ngamma: sunrealtype,
    zn1: &N_Vector,
    ycor: &N_Vector,
    ftemp: &N_Vector,
    res: &N_Vector,
) -> i32 {
    N_VLinearSum(rl1, zn1, ONE, ycor, res);
    N_VLinearSum(ngamma, ftemp, ONE, res, res);
    0
}

/*
 * -----------------------------------------------------------------
 * Form y with perturbation = FRACT*(func. iter. correction)
 * -----------------------------------------------------------------
 */

pub fn cvDiagSetup_formY(
    h: sunrealtype,
    r: sunrealtype,
    fpred: &N_Vector,
    zn1: &N_Vector,
    ypred: &N_Vector,
    ftemp: &N_Vector,
    y: &N_Vector,
) -> i32 {
    N_VLinearSum(h, fpred, -ONE, zn1, ftemp);
    N_VLinearSum(r, ftemp, ONE, ypred, y);
    0
}

/*
 * -----------------------------------------------------------------
 * Construct M = I - gamma*J with J = diag(deltaf_i/deltay_i)
 * protecting against deltay_i being at roundoff level.
 * -----------------------------------------------------------------
 */

pub fn cvDiagSetup_buildM(
    _fract: sunrealtype, /* SUNDIALS_MAYBE_UNUSED in C */
    uround: sunrealtype,
    h: sunrealtype,
    ftemp: &N_Vector,
    fpred: &N_Vector,
    ewt: &N_Vector,
    bit: &N_Vector,
    bitcomp: &N_Vector,
    y: &N_Vector,
    M: &N_Vector,
) -> i32 {
    N_VLinearSum(ONE, M, -ONE, fpred, M);
    N_VLinearSum(FRACT, ftemp, -h, M, M);
    N_VProd(ftemp, ewt, y);
    /* Protect against deltay_i being at roundoff level */
    N_VCompare(uround, y, bit);
    N_VAddConst(bit, -ONE, bitcomp);
    N_VProd(ftemp, bit, y);
    N_VLinearSum(FRACT, y, -ONE, bitcomp, y);
    N_VDiv(M, y, M);
    N_VProd(M, bit, M);
    N_VLinearSum(ONE, M, -ONE, bitcomp, M);
    0
}

/*
 * -----------------------------------------------------------------
 *  Update M with changed gamma so that M = I - gamma*J.
 * -----------------------------------------------------------------
 */

pub fn cvDiagSolve_updateM(r: sunrealtype, M: &N_Vector) -> i32 {
    N_VInv(M, M);
    N_VAddConst(M, -ONE, M);
    N_VScale(r, M, M);
    N_VAddConst(M, ONE, M);
    0
}
