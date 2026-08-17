//! Port of `src/sundials/sundials_dense.c` +
//! `include/sundials/sundials_dense.h` (dense direct kernels over column
//! views). Loop structure and arithmetic order match C exactly.

use crate::sundials_direct::{dls_cols, SUNDlsMat};
use crate::sundials_math::{SUNRabs, SUNRsqrt};
use crate::sundials_types::*;

const ZERO: sunrealtype = 0.0;
const ONE: sunrealtype = 1.0;
const TWO: sunrealtype = 2.0;

pub fn SUNDlsMat_DenseGETRF(A: &SUNDlsMat, p: &mut [sunindextype]) -> sunindextype {
    let mut a = A.borrow_mut();
    let (m, n, ldim) = (a.M, a.N, a.ldim);
    let mut cols = dls_cols(&mut a.data, ldim);
    SUNDlsMat_denseGETRF(&mut cols, m, n, p)
}

pub fn SUNDlsMat_DenseGETRS(A: &SUNDlsMat, p: &[sunindextype], b: &mut [sunrealtype]) {
    let mut a = A.borrow_mut();
    let (n, ldim) = (a.N, a.ldim);
    let mut cols = dls_cols(&mut a.data, ldim);
    SUNDlsMat_denseGETRS(&mut cols, n, p, b);
}

pub fn SUNDlsMat_DensePOTRF(A: &SUNDlsMat) -> sunindextype {
    let mut a = A.borrow_mut();
    let (m, ldim) = (a.M, a.ldim);
    let mut cols = dls_cols(&mut a.data, ldim);
    SUNDlsMat_densePOTRF(&mut cols, m)
}

pub fn SUNDlsMat_DensePOTRS(A: &SUNDlsMat, b: &mut [sunrealtype]) {
    let mut a = A.borrow_mut();
    let (m, ldim) = (a.M, a.ldim);
    let mut cols = dls_cols(&mut a.data, ldim);
    SUNDlsMat_densePOTRS(&mut cols, m, b);
}

pub fn SUNDlsMat_DenseGEQRF(A: &SUNDlsMat, beta: &mut [sunrealtype], wrk: &mut [sunrealtype]) -> i32 {
    let mut a = A.borrow_mut();
    let (m, n, ldim) = (a.M, a.N, a.ldim);
    let mut cols = dls_cols(&mut a.data, ldim);
    SUNDlsMat_denseGEQRF(&mut cols, m, n, beta, wrk)
}

pub fn SUNDlsMat_DenseORMQR(
    A: &SUNDlsMat,
    beta: &mut [sunrealtype],
    vn: &mut [sunrealtype],
    vm: &mut [sunrealtype],
    wrk: &mut [sunrealtype],
) -> i32 {
    let mut a = A.borrow_mut();
    let (m, n, ldim) = (a.M, a.N, a.ldim);
    let mut cols = dls_cols(&mut a.data, ldim);
    SUNDlsMat_denseORMQR(&mut cols, m, n, beta, vn, vm, wrk)
}

pub fn SUNDlsMat_DenseCopy(A: &SUNDlsMat, B: &SUNDlsMat) {
    let a = A.borrow();
    let mut b = B.borrow_mut();
    let (m, n, aldim, bldim) = (a.M, a.N, a.ldim, b.ldim);
    for j in 0..n {
        for i in 0..m {
            b.data[(j * bldim + i) as usize] = a.data[(j * aldim + i) as usize];
        }
    }
}

pub fn SUNDlsMat_DenseScale(c: sunrealtype, A: &SUNDlsMat) {
    let mut a = A.borrow_mut();
    let (m, n, ldim) = (a.M, a.N, a.ldim);
    let mut cols = dls_cols(&mut a.data, ldim);
    SUNDlsMat_denseScale(c, &mut cols, m, n);
}

pub fn SUNDlsMat_DenseMatvec(A: &SUNDlsMat, x: &[sunrealtype], y: &mut [sunrealtype]) {
    let mut a = A.borrow_mut();
    let (m, n, ldim) = (a.M, a.N, a.ldim);
    let mut cols = dls_cols(&mut a.data, ldim);
    SUNDlsMat_denseMatvec(&mut cols, x, y, m, n);
}

pub fn SUNDlsMat_denseGETRF(
    a: &mut [&mut [sunrealtype]],
    m: sunindextype,
    n: sunindextype,
    p: &mut [sunindextype],
) -> sunindextype {
    /* k-th elimination step number */
    for k in 0..n as usize {
        /* find l = pivot row number */
        let mut l = k;
        for i in (k + 1)..m as usize {
            if SUNRabs(a[k][i]) > SUNRabs(a[k][l]) {
                l = i;
            }
        }
        p[k] = l as sunindextype;

        /* check for zero pivot element */
        if a[k][l] == ZERO {
            return (k + 1) as sunindextype;
        }

        /* swap a(k,1:n) and a(l,1:n) if necessary */
        if l != k {
            for i in 0..n as usize {
                let temp = a[i][l];
                a[i][l] = a[i][k];
                a[i][k] = temp;
            }
        }

        /* Scale the elements below the diagonal in column k by 1.0/a(k,k). */
        let mult = ONE / a[k][k];
        for i in (k + 1)..m as usize {
            a[k][i] *= mult;
        }

        /* row_i = row_i - [a(i,k)/a(k,k)] row_k, one column at a time */
        for j in (k + 1)..n as usize {
            let a_kj = a[j][k];
            if a_kj != ZERO {
                /* split borrows: col_k (read) and col_j (write) are distinct */
                let (left, right) = a.split_at_mut(j);
                let col_k = &left[k];
                let col_j = &mut right[0];
                for i in (k + 1)..m as usize {
                    col_j[i] -= a_kj * col_k[i];
                }
            }
        }
    }

    0
}

pub fn SUNDlsMat_denseGETRS(
    a: &mut [&mut [sunrealtype]],
    n: sunindextype,
    p: &[sunindextype],
    b: &mut [sunrealtype],
) {
    let n = n as usize;

    /* Permute b, based on pivot information in p */
    for k in 0..n {
        let pk = p[k] as usize;
        if pk != k {
            b.swap(k, pk);
        }
    }

    /* Solve Ly = b, store solution y in b */
    for k in 0..n.saturating_sub(1) {
        let col_k = &a[k];
        for i in (k + 1)..n {
            b[i] -= col_k[i] * b[k];
        }
    }

    /* Solve Ux = y, store solution x in b */
    for k in (1..n).rev() {
        let col_k = &a[k];
        b[k] /= col_k[k];
        for i in 0..k {
            b[i] -= col_k[i] * b[k];
        }
    }
    b[0] /= a[0][0];
}

pub fn SUNDlsMat_densePOTRF(a: &mut [&mut [sunrealtype]], m: sunindextype) -> sunindextype {
    let m = m as usize;
    for j in 0..m {
        if j > 0 {
            for i in j..m {
                for k in 0..j {
                    /* a_col_j[i] -= a_col_k[i] * a_col_k[j] */
                    let (left, right) = a.split_at_mut(j);
                    let a_col_k = &left[k];
                    let a_col_j = &mut right[0];
                    a_col_j[i] -= a_col_k[i] * a_col_k[j];
                }
            }
        }

        let mut a_diag = a[j][j];
        if a_diag <= ZERO {
            return (j + 1) as sunindextype;
        }
        a_diag = SUNRsqrt(a_diag);

        for i in j..m {
            a[j][i] /= a_diag;
        }
    }

    0
}

pub fn SUNDlsMat_densePOTRS(a: &mut [&mut [sunrealtype]], m: sunindextype, b: &mut [sunrealtype]) {
    let m = m as usize;

    /* Solve C y = b, forward substitution - column version. */
    for j in 0..m.saturating_sub(1) {
        let col_j = &a[j];
        b[j] /= col_j[j];
        for i in (j + 1)..m {
            b[i] -= b[j] * col_j[i];
        }
    }
    b[m - 1] /= a[m - 1][m - 1];

    /* Solve C^T x = y, backward substitution - row version. */
    b[m - 1] /= a[m - 1][m - 1];
    for i in (0..m.saturating_sub(1)).rev() {
        let col_i = &a[i];
        for j in (i + 1)..m {
            b[i] -= col_i[j] * b[j];
        }
        b[i] /= col_i[i];
    }
}

pub fn SUNDlsMat_denseGEQRF(
    a: &mut [&mut [sunrealtype]],
    m: sunindextype,
    n: sunindextype,
    beta: &mut [sunrealtype],
    v: &mut [sunrealtype],
) -> i32 {
    let (m, n) = (m as usize, n as usize);

    /* For each column...*/
    for j in 0..n {
        let ajj = a[j][j];

        /* Compute the j-th Householder vector (of length m-j) */
        v[0] = ONE;
        let mut s = ZERO;
        for i in 1..(m - j) {
            v[i] = a[j][i + j];
            s += v[i] * v[i];
        }

        if s != ZERO {
            let mu = SUNRsqrt(ajj * ajj + s);
            let v1 = if ajj <= ZERO { ajj - mu } else { -s / (ajj + mu) };
            let v1_2 = v1 * v1;
            beta[j] = TWO * v1_2 / (s + v1_2);
            for i in 1..(m - j) {
                v[i] /= v1;
            }
        } else {
            beta[j] = ZERO;
        }

        /* Update upper triangle of A (load R) */
        for k in j..n {
            let col_k = &mut a[k];
            let mut s = ZERO;
            for i in 0..(m - j) {
                s += col_k[i + j] * v[i];
            }
            s *= beta[j];
            for i in 0..(m - j) {
                col_k[i + j] -= s * v[i];
            }
        }

        /* Update A (load Householder vector) */
        if j < m - 1 {
            for i in 1..(m - j) {
                a[j][i + j] = v[i];
            }
        }
    }

    0
}

pub fn SUNDlsMat_denseORMQR(
    a: &mut [&mut [sunrealtype]],
    m: sunindextype,
    n: sunindextype,
    beta: &[sunrealtype],
    vn: &[sunrealtype],
    vm: &mut [sunrealtype],
    v: &mut [sunrealtype],
) -> i32 {
    let (m, n) = (m as usize, n as usize);

    /* Initialize vm */
    for i in 0..n {
        vm[i] = vn[i];
    }
    for i in n..m {
        vm[i] = ZERO;
    }

    /* Accumulate (backwards) corrections into vm */
    for j in (0..n).rev() {
        let col_j = &a[j];
        v[0] = ONE;
        let mut s = vm[j];
        for i in 1..(m - j) {
            v[i] = col_j[i + j];
            s += v[i] * vm[i + j];
        }
        s *= beta[j];

        for i in 0..(m - j) {
            vm[i + j] -= s * v[i];
        }
    }

    0
}

pub fn SUNDlsMat_denseCopy(
    a: &[&mut [sunrealtype]],
    b: &mut [&mut [sunrealtype]],
    m: sunindextype,
    n: sunindextype,
) {
    for j in 0..n as usize {
        for i in 0..m as usize {
            b[j][i] = a[j][i];
        }
    }
}

pub fn SUNDlsMat_denseScale(
    c: sunrealtype,
    a: &mut [&mut [sunrealtype]],
    m: sunindextype,
    n: sunindextype,
) {
    for j in 0..n as usize {
        for i in 0..m as usize {
            a[j][i] *= c;
        }
    }
}

pub fn SUNDlsMat_denseAddIdentity(a: &mut [&mut [sunrealtype]], n: sunindextype) {
    for i in 0..n as usize {
        a[i][i] += ONE;
    }
}

pub fn SUNDlsMat_denseMatvec(
    a: &[&mut [sunrealtype]],
    x: &[sunrealtype],
    y: &mut [sunrealtype],
    m: sunindextype,
    n: sunindextype,
) {
    for i in 0..m as usize {
        y[i] = ZERO;
    }
    for j in 0..n as usize {
        let col_j = &a[j];
        for i in 0..m as usize {
            y[i] += col_j[i] * x[j];
        }
    }
}
