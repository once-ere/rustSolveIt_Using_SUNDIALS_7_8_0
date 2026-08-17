//! Port of `src/sundials/sundials_iterative.c` +
//! `include/sundials/sundials_iterative.h` + `sundials_iterative_impl.h`.
//!
//! The row-wise Hessenberg `sunrealtype** h` maps to `&mut [Vec<f64>]`
//! (h[i][j]); `N_Vector*` maps to slices of handles. `SUNQRData`'s C
//! `void*` shape becomes a concrete struct (only used internally by
//! kinsol Anderson acceleration and ARKODE LSRK).

use crate::sundials_errors::SUN_SUCCESS;
use crate::sundials_math::{SUNMAX, SUNRabs, SUNRsqrt, SUNSQR};
use crate::sundials_nvector::*;
use crate::sundials_types::*;

const FACTOR: sunrealtype = 1000.0;
const ZERO: sunrealtype = 0.0;
const ONE: sunrealtype = 1.0;

/* types of preconditioning */
pub const SUN_PREC_NONE: i32 = 0;
pub const SUN_PREC_LEFT: i32 = 1;
pub const SUN_PREC_RIGHT: i32 = 2;
pub const SUN_PREC_BOTH: i32 = 3;

/* types of Gram-Schmidt routines */
pub const SUN_MODIFIED_GS: i32 = 1;
pub const SUN_CLASSICAL_GS: i32 = 2;

/// C `struct _SUNQRData` (workspace for the `SUNQRAdd*` routines).
pub struct SUNQRData_ {
    pub vtemp: N_Vector,
    pub vtemp2: Option<N_Vector>,
    pub temp_array: Vec<sunrealtype>,
}

pub type SUNQRData = SUNQRData_;

pub type SUNQRAddFn = fn(
    Q: &[N_Vector],
    R: &mut [sunrealtype],
    df: &N_Vector,
    m: i32,
    mMax: i32,
    QRdata: &mut SUNQRData,
) -> SUNErrCode;

pub fn SUNModifiedGS(
    v: &[N_Vector],
    h: &mut [Vec<sunrealtype>],
    k: i32,
    p: i32,
    new_vk_norm: &mut sunrealtype,
) -> SUNErrCode {
    let k = k as usize;

    let mut vk_norm = N_VDotProd(&v[k], &v[k]);
    vk_norm = SUNRsqrt(vk_norm);
    let k_minus_1 = k - 1;
    let i0 = SUNMAX(k as i32 - p, 0) as usize;

    /* Perform modified Gram-Schmidt */
    for i in i0..k {
        h[i][k_minus_1] = N_VDotProd(&v[i], &v[k]);
        N_VLinearSum(ONE, &v[k], -h[i][k_minus_1], &v[i], &v[k]);
    }

    /* Compute the norm of the new vector at v[k] */
    *new_vk_norm = N_VDotProd(&v[k], &v[k]);
    *new_vk_norm = SUNRsqrt(*new_vk_norm);

    /* If the norm of the new vector at v[k] is less than
    FACTOR (== 1000) times unit roundoff times the norm of the input
    vector v[k], then the vector will be reorthogonalized. */
    let temp = FACTOR * vk_norm;
    if (temp + *new_vk_norm) != temp {
        return SUN_SUCCESS;
    }

    let mut new_norm_2 = ZERO;

    for i in i0..k {
        let new_product = N_VDotProd(&v[i], &v[k]);
        let temp = FACTOR * h[i][k_minus_1];
        if (temp + new_product) == temp {
            continue;
        }
        h[i][k_minus_1] += new_product;
        N_VLinearSum(ONE, &v[k], -new_product, &v[i], &v[k]);
        new_norm_2 += SUNSQR(new_product);
    }

    if new_norm_2 != ZERO {
        let new_product = SUNSQR(*new_vk_norm) - new_norm_2;
        *new_vk_norm = if new_product > ZERO {
            SUNRsqrt(new_product)
        } else {
            ZERO
        };
    }

    SUN_SUCCESS
}

pub fn SUNClassicalGS(
    v: &[N_Vector],
    h: &mut [Vec<sunrealtype>],
    k: i32,
    p: i32,
    new_vk_norm: &mut sunrealtype,
    stemp: &mut [sunrealtype],
    vtemp: &mut [N_Vector],
) -> SUNErrCode {
    let ku = k as usize;
    let k_minus_1 = ku - 1;
    let i0 = SUNMAX(k - p, 0) as usize;

    /* Perform Classical Gram-Schmidt */
    let ier = N_VDotProdMulti((ku - i0 + 1) as i32, &v[ku], &v[i0..], stemp);
    if ier != SUN_SUCCESS {
        return ier;
    }

    let vk_norm = SUNRsqrt(stemp[ku - i0]);
    let mut i = (ku - i0) as i64 - 1;
    while i >= 0 {
        let iu = i as usize;
        h[iu][k_minus_1] = stemp[iu];
        stemp[iu + 1] = -stemp[iu];
        vtemp[iu + 1] = v[iu].clone();
        i -= 1;
    }
    stemp[0] = ONE;
    vtemp[0] = v[ku].clone();

    let ier = N_VLinearCombination((ku - i0 + 1) as i32, stemp, vtemp, &v[ku]);
    if ier != SUN_SUCCESS {
        return ier;
    }

    /* Compute the norm of the new vector at v[k] */
    *new_vk_norm = SUNRsqrt(N_VDotProd(&v[ku], &v[ku]));

    /* Reorthogonalize if necessary */
    if (FACTOR * *new_vk_norm) < vk_norm {
        let ier = N_VDotProdMulti((ku - i0) as i32, &v[ku], &v[i0..], &mut stemp[1..]);
        if ier != SUN_SUCCESS {
            return ier;
        }

        stemp[0] = ONE;
        vtemp[0] = v[ku].clone();
        for i in i0..ku {
            h[i][k_minus_1] += stemp[i - i0 + 1];
            stemp[i - i0 + 1] = -stemp[i - i0 + 1];
            vtemp[i - i0 + 1] = v[i - i0].clone();
        }

        let ier = N_VLinearCombination(k + 1, stemp, vtemp, &v[ku]);
        if ier != SUN_SUCCESS {
            return ier;
        }

        *new_vk_norm = SUNRsqrt(N_VDotProd(&v[ku], &v[ku]));
    }

    SUN_SUCCESS
}

pub fn SUNQRfact(n: i32, h: &mut [Vec<sunrealtype>], q: &mut [sunrealtype], job: i32) -> i32 {
    let n = n as usize;
    let mut code = 0;

    match job {
        0 => {
            /* Compute a new factorization of H */
            for k in 0..n {
                /* Multiply column k by the previous k-1 Givens rotations */
                let mut j = 0i64;
                while j < k as i64 - 1 {
                    let ju = j as usize;
                    let i = 2 * ju;
                    let temp1 = h[ju][k];
                    let temp2 = h[ju + 1][k];
                    let c = q[i];
                    let s = q[i + 1];
                    h[ju][k] = c * temp1 - s * temp2;
                    h[ju + 1][k] = s * temp1 + c * temp2;
                    j += 1;
                }

                /* Compute the Givens rotation components c and s */
                let q_ptr = 2 * k;
                let temp1 = h[k][k];
                let temp2 = h[k + 1][k];
                let (c, s);
                if temp2 == ZERO {
                    c = ONE;
                    s = ZERO;
                } else if SUNRabs(temp2) >= SUNRabs(temp1) {
                    let temp3 = temp1 / temp2;
                    s = -ONE / SUNRsqrt(ONE + SUNSQR(temp3));
                    c = -s * temp3;
                } else {
                    let temp3 = temp2 / temp1;
                    c = ONE / SUNRsqrt(ONE + SUNSQR(temp3));
                    s = -c * temp3;
                }
                q[q_ptr] = c;
                q[q_ptr + 1] = s;
                h[k][k] = c * temp1 - s * temp2;
                if h[k][k] == ZERO {
                    code = (k + 1) as i32;
                }
            }
        }
        _ => {
            /* Update the factored H to which a new column has been added */
            let n_minus_1 = n - 1;
            code = 0;

            /* Multiply the new column by the previous n-1 Givens rotations */
            for k in 0..n_minus_1 {
                let i = 2 * k;
                let temp1 = h[k][n_minus_1];
                let temp2 = h[k + 1][n_minus_1];
                let c = q[i];
                let s = q[i + 1];
                h[k][n_minus_1] = c * temp1 - s * temp2;
                h[k + 1][n_minus_1] = s * temp1 + c * temp2;
            }

            /* Compute new Givens rotation for the last two entries */
            let temp1 = h[n_minus_1][n_minus_1];
            let temp2 = h[n][n_minus_1];
            let (c, s);
            if temp2 == ZERO {
                c = ONE;
                s = ZERO;
            } else if SUNRabs(temp2) >= SUNRabs(temp1) {
                let temp3 = temp1 / temp2;
                s = -ONE / SUNRsqrt(ONE + SUNSQR(temp3));
                c = -s * temp3;
            } else {
                let temp3 = temp2 / temp1;
                c = ONE / SUNRsqrt(ONE + SUNSQR(temp3));
                s = -c * temp3;
            }
            let q_ptr = 2 * n_minus_1;
            q[q_ptr] = c;
            q[q_ptr + 1] = s;
            h[n_minus_1][n_minus_1] = c * temp1 - s * temp2;
            if h[n_minus_1][n_minus_1] == ZERO {
                code = n as i32;
            }
        }
    }

    code
}

pub fn SUNQRsol(n: i32, h: &mut [Vec<sunrealtype>], q: &[sunrealtype], b: &mut [sunrealtype]) -> i32 {
    let n = n as usize;
    let mut code = 0;

    /* Compute Q*b */
    for k in 0..n {
        let q_ptr = 2 * k;
        let c = q[q_ptr];
        let s = q[q_ptr + 1];
        let temp1 = b[k];
        let temp2 = b[k + 1];
        b[k] = c * temp1 - s * temp2;
        b[k + 1] = s * temp1 + c * temp2;
    }

    /* Solve R*x = Q*b */
    let mut k = n as i64 - 1;
    while k >= 0 {
        let ku = k as usize;
        if h[ku][ku] == ZERO {
            code = (ku + 1) as i32;
            break;
        }
        b[ku] /= h[ku][ku];
        for i in 0..ku {
            b[i] -= b[ku] * h[i][ku];
        }
        k -= 1;
    }

    code
}

pub fn SUNQRAdd_MGS(
    Q: &[N_Vector],
    R: &mut [sunrealtype],
    df: &N_Vector,
    m: i32,
    mMax: i32,
    qrdata: &mut SUNQRData,
) -> SUNErrCode {
    let m = m as usize;
    let mMax = mMax as usize;

    N_VScale(ONE, df, &qrdata.vtemp);
    for j in 0..m {
        R[m * mMax + j] = N_VDotProd(&Q[j], &qrdata.vtemp);
        N_VLinearSum(ONE, &qrdata.vtemp, -R[m * mMax + j], &Q[j], &qrdata.vtemp);
    }
    R[m * mMax + m] = N_VDotProd(&qrdata.vtemp, &qrdata.vtemp);
    R[m * mMax + m] = SUNRsqrt(R[m * mMax + m]);
    N_VScale(1.0 / R[m * mMax + m], &qrdata.vtemp, &Q[m]);

    SUN_SUCCESS
}

pub fn SUNQRAdd_ICWY(
    Q: &[N_Vector],
    R: &mut [sunrealtype],
    df: &N_Vector,
    m: i32,
    mMax: i32,
    qrdata: &mut SUNQRData,
) -> SUNErrCode {
    let mu = m as usize;
    let mMaxu = mMax as usize;

    N_VScale(ONE, df, &qrdata.vtemp); /* stores d_fi in temp */

    if m > 0 {
        /* T(1:k-1,k-1)^T = Q(:,1:k-1)^T * Q(:,k-1) */
        let ier = N_VDotProdMulti(
            m,
            &Q[mu - 1],
            Q,
            &mut qrdata.temp_array[(mu - 1) * mMaxu..],
        );
        if ier != SUN_SUCCESS {
            return ier;
        }

        /* T(k-1,k-1) = 1.0 */
        qrdata.temp_array[(mu - 1) * mMaxu + (mu - 1)] = ONE;

        /* R(1:k-1,k) = Q_k-1^T * df */
        let ier = N_VDotProdMulti(m, &qrdata.vtemp, Q, &mut R[mu * mMaxu..]);
        if ier != SUN_SUCCESS {
            return ier;
        }

        /* Solve T^T * R(1:k-1,k) = R(1:k-1,k) */
        for k in 0..mu {
            /* Skip setting the diagonal element because it doesn't change */
            for j in (k + 1)..mu {
                R[mu * mMaxu + j] -= R[mu * mMaxu + k] * qrdata.temp_array[j * mMaxu + k];
            }
        }

        /* Q(:,k-1) = df - Q_k-1 R(1:k-1,k) */
        let vtemp2 = qrdata.vtemp2.as_ref().expect("vtemp2 workspace");
        let ier = N_VLinearCombination(m, &R[mu * mMaxu..], Q, vtemp2);
        if ier != SUN_SUCCESS {
            return ier;
        }
        N_VLinearSum(ONE, &qrdata.vtemp, -ONE, vtemp2, &qrdata.vtemp);
    }

    /* R(k,k) = \| df \| */
    R[mu * mMaxu + mu] = N_VDotProd(&qrdata.vtemp, &qrdata.vtemp);
    R[mu * mMaxu + mu] = SUNRsqrt(R[mu * mMaxu + mu]);
    /* Q(:,k) = df / \| df \| */
    N_VScale(1.0 / R[mu * mMaxu + mu], &qrdata.vtemp, &Q[mu]);

    SUN_SUCCESS
}

pub fn SUNQRAdd_CGS2(
    Q: &[N_Vector],
    R: &mut [sunrealtype],
    df: &N_Vector,
    m: i32,
    mMax: i32,
    qrdata: &mut SUNQRData,
) -> SUNErrCode {
    let mu = m as usize;
    let mMaxu = mMax as usize;

    N_VScale(ONE, df, &qrdata.vtemp); /* temp = df */

    if m > 0 {
        /* s_k = Q_k-1^T df_aa */
        let ier = N_VDotProdMulti(m, &qrdata.vtemp, Q, &mut R[mu * mMaxu..]);
        if ier != SUN_SUCCESS {
            return ier;
        }

        /* y = df - Q_k-1 s_k */
        let vtemp2 = qrdata.vtemp2.as_ref().expect("vtemp2 workspace").clone();
        let ier = N_VLinearCombination(m, &R[mu * mMaxu..], Q, &vtemp2);
        if ier != SUN_SUCCESS {
            return ier;
        }
        N_VLinearSum(ONE, &qrdata.vtemp, -ONE, &vtemp2, &vtemp2);

        /* z_k = Q_k-1^T y */
        let ier = N_VDotProdMulti(m, &vtemp2, Q, &mut qrdata.temp_array);
        if ier != SUN_SUCCESS {
            return ier;
        }

        /* df = y - Q_k-1 z_k */
        let ier = N_VLinearCombination(m, &qrdata.temp_array, Q, &Q[mu]);
        if ier != SUN_SUCCESS {
            return ier;
        }
        N_VLinearSum(ONE, &vtemp2, -ONE, &Q[mu], &qrdata.vtemp);

        /* R(1:k-1,k) = s_k + z_k */
        for j in 0..mu {
            R[mu * mMaxu + j] = R[mu * mMaxu + j] + qrdata.temp_array[j];
        }
    }

    /* R(k,k) = \| df \| */
    R[mu * mMaxu + mu] = N_VDotProd(&qrdata.vtemp, &qrdata.vtemp);
    R[mu * mMaxu + mu] = SUNRsqrt(R[mu * mMaxu + mu]);
    /* Q(:,k) = df / R(k,k) */
    N_VScale(1.0 / R[mu * mMaxu + mu], &qrdata.vtemp, &Q[mu]);

    SUN_SUCCESS
}

pub fn SUNQRAdd_DCGS2(
    Q: &[N_Vector],
    R: &mut [sunrealtype],
    df: &N_Vector,
    m: i32,
    mMax: i32,
    qrdata: &mut SUNQRData,
) -> SUNErrCode {
    let mu = m as usize;
    let mMaxu = mMax as usize;

    N_VScale(ONE, df, &qrdata.vtemp); /* temp = df */

    if m > 0 {
        /* R(1:k-1,k) = Q_k-1^T df_aa */
        let ier = N_VDotProdMulti(m, &qrdata.vtemp, Q, &mut R[mu * mMaxu..]);
        if ier != SUN_SUCCESS {
            return ier;
        }
        /* Delayed reorthogonalization */
        if m > 1 {
            /* s = Q_k-2^T Q(:,k-1) */
            let ier = N_VDotProdMulti(m - 1, &Q[mu - 1], Q, &mut qrdata.temp_array);
            if ier != SUN_SUCCESS {
                return ier;
            }

            /* Q(:,k-1) = Q(:,k-1) - Q_k-2 s */
            let vtemp2 = qrdata.vtemp2.as_ref().expect("vtemp2 workspace");
            let ier = N_VLinearCombination(m - 1, &qrdata.temp_array, Q, vtemp2);
            if ier != SUN_SUCCESS {
                return ier;
            }
            N_VLinearSum(ONE, &Q[mu - 1], -ONE, vtemp2, &Q[mu - 1]);

            /* R(1:k-2,k-1) = R(1:k-2,k-1) + s */
            for j in 0..(mu - 1) {
                R[(mu - 1) * mMaxu + j] = R[(mu - 1) * mMaxu + j] + qrdata.temp_array[j];
            }
        }

        /* df = df - Q(:,k-1) R(1:k-1,k) */
        let vtemp2 = qrdata.vtemp2.as_ref().expect("vtemp2 workspace");
        let ier = N_VLinearCombination(m, &R[mu * mMaxu..], Q, vtemp2);
        if ier != SUN_SUCCESS {
            return ier;
        }
        N_VLinearSum(ONE, &qrdata.vtemp, -ONE, vtemp2, &qrdata.vtemp);
    }

    /* R(k,k) = \| df \| */
    R[mu * mMaxu + mu] = N_VDotProd(&qrdata.vtemp, &qrdata.vtemp);
    R[mu * mMaxu + mu] = SUNRsqrt(R[mu * mMaxu + mu]);
    /* Q(:,k) = df / R(k,k) */
    N_VScale(1.0 / R[mu * mMaxu + mu], &qrdata.vtemp, &Q[mu]);

    SUN_SUCCESS
}

/// C `SUNQRAdd_ICWY_SB` (single-buffer variant; requires a vector with
/// `nvdotprodmultiallreduce`, so unreachable for serial vectors — as in C,
/// where release builds would call through a NULL op).
pub fn SUNQRAdd_ICWY_SB(
    Q: &[N_Vector],
    R: &mut [sunrealtype],
    df: &N_Vector,
    m: i32,
    mMax: i32,
    qrdata: &mut SUNQRData,
) -> SUNErrCode {
    let mu = m as usize;
    let mMaxu = mMax as usize;

    N_VScale(ONE, df, &qrdata.vtemp); /* stores d_fi in temp */

    if m > 0 {
        /* T(1:k-1,k-1)^T = Q(:,1:k-1)^T * Q(:,k-1) */
        let ier = N_VDotProdMultiLocal(
            m,
            &Q[mu - 1],
            Q,
            &mut qrdata.temp_array[(mu - 1) * mMaxu..],
        );
        if ier != SUN_SUCCESS {
            return ier;
        }

        /* R(1:k-1,k) = Q_k-1^T * df (put R values at end of temp_array) */
        let ier = N_VDotProdMultiLocal(
            m,
            &qrdata.vtemp,
            Q,
            &mut qrdata.temp_array[(mu - 1) * mMaxu + mu..],
        );
        if ier != SUN_SUCCESS {
            return ier;
        }

        let ier = N_VDotProdMultiAllReduce(
            m + m,
            &qrdata.vtemp,
            &mut qrdata.temp_array[(mu - 1) * mMaxu..],
        );
        if ier != SUN_SUCCESS {
            return ier;
        }

        /* Move the last values from temp array into R */
        for k in 0..mu {
            R[mu * mMaxu + k] = qrdata.temp_array[(mu - 1) * mMaxu + mu + k];
        }

        /* T(k-1,k-1) = 1.0 */
        qrdata.temp_array[(mu - 1) * mMaxu + (mu - 1)] = ONE;

        /* Solve T^T * R(1:k-1,k) = R(1:k-1,k) */
        for k in 0..mu {
            for j in (k + 1)..mu {
                R[mu * mMaxu + j] -= R[mu * mMaxu + k] * qrdata.temp_array[j * mMaxu + k];
            }
        }

        /* Q(:,k-1) = df - Q_k-1 R(1:k-1,k) */
        let vtemp2 = qrdata.vtemp2.as_ref().expect("vtemp2 workspace");
        let ier = N_VLinearCombination(m, &R[mu * mMaxu..], Q, vtemp2);
        if ier != SUN_SUCCESS {
            return ier;
        }
        N_VLinearSum(ONE, &qrdata.vtemp, -ONE, vtemp2, &qrdata.vtemp);
    }

    /* R(k,k) = \| df \| */
    R[mu * mMaxu + mu] = N_VDotProd(&qrdata.vtemp, &qrdata.vtemp);
    R[mu * mMaxu + mu] = SUNRsqrt(R[mu * mMaxu + mu]);
    /* Q(:,k) = df / \| df \| */
    N_VScale(1.0 / R[mu * mMaxu + mu], &qrdata.vtemp, &Q[mu]);

    SUN_SUCCESS
}

/// C `SUNQRAdd_DCGS2_SB` (single-buffer variant).
pub fn SUNQRAdd_DCGS2_SB(
    Q: &[N_Vector],
    R: &mut [sunrealtype],
    df: &N_Vector,
    m: i32,
    mMax: i32,
    qrdata: &mut SUNQRData,
) -> SUNErrCode {
    let mu = m as usize;
    let mMaxu = mMax as usize;

    N_VScale(ONE, df, &qrdata.vtemp); /* temp = df */

    if m > 0 {
        if m == 1 {
            /* R(1:k-1,k) = Q_k-1^T df_aa */
            let ier = N_VDotProdMulti(m, &qrdata.vtemp, Q, &mut R[mu * mMaxu..]);
            if ier != SUN_SUCCESS {
                return ier;
            }
        } else {
            /* R(1:k-1,k) = Q_k-1^T df_aa (put R values at start of temp) */
            let ier = N_VDotProdMultiLocal(m, &qrdata.vtemp, Q, &mut qrdata.temp_array);
            if ier != SUN_SUCCESS {
                return ier;
            }

            /* s = Q_k-2^T Q(:,k-1) */
            let ier =
                N_VDotProdMultiLocal(m - 1, &Q[mu - 1], Q, &mut qrdata.temp_array[mu..]);
            if ier != SUN_SUCCESS {
                return ier;
            }
            let ier =
                N_VDotProdMultiAllReduce(m + m - 1, &qrdata.vtemp, &mut qrdata.temp_array);
            if ier != SUN_SUCCESS {
                return ier;
            }

            /* Move R values to R */
            for j in 0..mu {
                R[mu * mMaxu + j] = qrdata.temp_array[j];
            }

            /* Q(:,k-1) = Q(:,k-1) - Q_k-2 s */
            let vtemp2 = qrdata.vtemp2.as_ref().expect("vtemp2 workspace");
            let ier = N_VLinearCombination(m - 1, &qrdata.temp_array[mu..], Q, vtemp2);
            if ier != SUN_SUCCESS {
                return ier;
            }
            N_VLinearSum(ONE, &Q[mu - 1], -ONE, vtemp2, &Q[mu - 1]);

            /* R(1:k-2,k-1) = R(1:k-2,k-1) + s */
            for j in 0..(mu - 1) {
                R[(mu - 1) * mMaxu + j] = R[(mu - 1) * mMaxu + j] + qrdata.temp_array[mu + j];
            }
        }

        /* df = df - Q(:,k-1) R(1:k-1,k) */
        let vtemp2 = qrdata.vtemp2.as_ref().expect("vtemp2 workspace");
        let ier = N_VLinearCombination(m, &R[mu * mMaxu..], Q, vtemp2);
        if ier != SUN_SUCCESS {
            return ier;
        }
        N_VLinearSum(ONE, &qrdata.vtemp, -ONE, vtemp2, &qrdata.vtemp);
    }

    /* R(k,k) = \| df \| */
    R[mu * mMaxu + mu] = N_VDotProd(&qrdata.vtemp, &qrdata.vtemp);
    R[mu * mMaxu + mu] = SUNRsqrt(R[mu * mMaxu + mu]);
    /* Q(:,k) = df / R(k,k) */
    N_VScale(1.0 / R[mu * mMaxu + mu], &qrdata.vtemp, &Q[mu]);

    SUN_SUCCESS
}
