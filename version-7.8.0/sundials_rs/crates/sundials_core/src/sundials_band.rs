//! Port of `src/sundials/sundials_band.c` +
//! `include/sundials/sundials_band.h` (band direct kernels).
//! `ROW(i,j,smu) = i - j + smu` indexes within a column view.

use crate::sundials_direct::{dls_cols, SUNDlsMat};
use crate::sundials_math::{SUNMAX, SUNMIN, SUNRabs};
use crate::sundials_types::*;

const ZERO: sunrealtype = 0.0;
const ONE: sunrealtype = 1.0;

fn ROW(i: sunindextype, j: sunindextype, smu: sunindextype) -> sunindextype {
    i - j + smu
}

pub fn SUNDlsMat_BandGBTRF(A: &SUNDlsMat, p: &mut [sunindextype]) -> sunindextype {
    let mut a = A.borrow_mut();
    let (m, mu, ml, s_mu, ldim) = (a.M, a.mu, a.ml, a.s_mu, a.ldim);
    let mut cols = dls_cols(&mut a.data, ldim);
    SUNDlsMat_bandGBTRF(&mut cols, m, mu, ml, s_mu, p)
}

pub fn SUNDlsMat_BandGBTRS(A: &SUNDlsMat, p: &[sunindextype], b: &mut [sunrealtype]) {
    let mut a = A.borrow_mut();
    let (m, s_mu, ml, ldim) = (a.M, a.s_mu, a.ml, a.ldim);
    let mut cols = dls_cols(&mut a.data, ldim);
    SUNDlsMat_bandGBTRS(&mut cols, m, s_mu, ml, p, b);
}

pub fn SUNDlsMat_BandCopy(
    A: &SUNDlsMat,
    B: &SUNDlsMat,
    copymu: sunindextype,
    copyml: sunindextype,
) {
    let a = A.borrow();
    let mut bm = B.borrow_mut();
    let (n, a_smu, a_ldim) = (a.M, a.s_mu, a.ldim);
    let (b_smu, b_ldim) = (bm.s_mu, bm.ldim);
    let copySize = copymu + copyml + 1;
    for j in 0..n {
        let a_base = j * a_ldim + a_smu - copymu;
        let b_base = j * b_ldim + b_smu - copymu;
        for i in 0..copySize {
            bm.data[(b_base + i) as usize] = a.data[(a_base + i) as usize];
        }
    }
}

pub fn SUNDlsMat_BandScale(c: sunrealtype, A: &SUNDlsMat) {
    let mut a = A.borrow_mut();
    let (n, mu, ml, smu, ldim) = (a.M, a.mu, a.ml, a.s_mu, a.ldim);
    let mut cols = dls_cols(&mut a.data, ldim);
    SUNDlsMat_bandScale(c, &mut cols, n, mu, ml, smu);
}

pub fn SUNDlsMat_BandMatvec(A: &SUNDlsMat, x: &[sunrealtype], y: &mut [sunrealtype]) {
    let mut a = A.borrow_mut();
    let (n, mu, ml, smu, ldim) = (a.M, a.mu, a.ml, a.s_mu, a.ldim);
    let mut cols = dls_cols(&mut a.data, ldim);
    SUNDlsMat_bandMatvec(&mut cols, x, y, n, mu, ml, smu);
}

pub fn SUNDlsMat_bandGBTRF(
    a: &mut [&mut [sunrealtype]],
    n: sunindextype,
    mu: sunindextype,
    ml: sunindextype,
    smu: sunindextype,
    p: &mut [sunindextype],
) -> sunindextype {
    /* zero out the first smu - mu rows of the rectangular array a */
    let num_rows = smu - mu;
    if num_rows > 0 {
        for c in 0..n as usize {
            for r in 0..num_rows as usize {
                a[c][r] = ZERO;
            }
        }
    }

    /* k = elimination step number */
    for k in 0..(n - 1) {
        let ku = k as usize;
        let last_row_k = SUNMIN(n - 1, k + ml);

        /* find l = pivot row number */
        let mut l = k;
        let mut max = SUNRabs(a[ku][smu as usize]);
        for i in (k + 1)..=last_row_k {
            let val = SUNRabs(a[ku][ROW(i, k, smu) as usize]);
            if val > max {
                l = i;
                max = val;
            }
        }
        let storage_l = ROW(l, k, smu) as usize;
        p[ku] = l;

        /* check for zero pivot element */
        if a[ku][storage_l] == ZERO {
            return k + 1;
        }

        /* swap a(l,k) and a(k,k) if necessary */
        let swap = l != k;
        if swap {
            a[ku].swap(storage_l, smu as usize);
        }

        /* Scale the elements below the diagonal in column k by -1.0/a(k,k). */
        let mult = -ONE / a[ku][smu as usize];
        for i in (k + 1)..=last_row_k {
            a[ku][ROW(i, k, smu) as usize] *= mult;
        }

        /* row_i = row_i - [a(i,k)/a(k,k)] row_k, column at a time */
        let last_col_k = SUNMIN(k + smu, n - 1);
        for j in (k + 1)..=last_col_k {
            let ju = j as usize;
            let storage_l = ROW(l, j, smu) as usize;
            let storage_k = ROW(k, j, smu) as usize;

            let a_kj = a[ju][storage_l];

            /* Swap the elements a(k,j) and a(k,l) if l!=k. */
            if swap {
                a[ju][storage_l] = a[ju][storage_k];
                a[ju][storage_k] = a_kj;
            }

            /* a(i,j) = a(i,j) - [a(i,k)/a(k,k)]*a(k,j) */
            if a_kj != ZERO {
                let (left, right) = a.split_at_mut(ju);
                let col_k = &left[ku];
                let col_j = &mut right[0];
                for i in (k + 1)..=last_row_k {
                    col_j[ROW(i, j, smu) as usize] += a_kj * col_k[ROW(i, k, smu) as usize];
                }
            }
        }
    }

    /* set the last pivot row to be n-1 and check for a zero pivot */
    p[(n - 1) as usize] = n - 1;
    if a[(n - 1) as usize][smu as usize] == ZERO {
        return n;
    }

    0
}

pub fn SUNDlsMat_bandGBTRS(
    a: &mut [&mut [sunrealtype]],
    n: sunindextype,
    smu: sunindextype,
    ml: sunindextype,
    p: &[sunindextype],
    b: &mut [sunrealtype],
) {
    /* Solve Ly = Pb, store solution y in b */
    for k in 0..(n - 1) {
        let ku = k as usize;
        let l = p[ku] as usize;
        let mult = b[l];
        if l != ku {
            b[l] = b[ku];
            b[ku] = mult;
        }
        let last_row_k = SUNMIN(n - 1, k + ml);
        for i in (k + 1)..=last_row_k {
            b[i as usize] += mult * a[ku][(smu + i - k) as usize];
        }
    }

    /* Solve Ux = y, store solution x in b */
    for k in (0..n).rev() {
        let ku = k as usize;
        let first_row_k = SUNMAX(0, k - smu);
        b[ku] /= a[ku][smu as usize];
        let mult = -b[ku];
        let mut i = first_row_k;
        while i <= k - 1 {
            b[i as usize] += mult * a[ku][(smu + i - k) as usize];
            i += 1;
        }
    }
}

pub fn SUNDlsMat_bandCopy(
    a: &[&mut [sunrealtype]],
    b: &mut [&mut [sunrealtype]],
    n: sunindextype,
    a_smu: sunindextype,
    b_smu: sunindextype,
    copymu: sunindextype,
    copyml: sunindextype,
) {
    let copySize = (copymu + copyml + 1) as usize;
    for j in 0..n as usize {
        let a_off = (a_smu - copymu) as usize;
        let b_off = (b_smu - copymu) as usize;
        for i in 0..copySize {
            b[j][b_off + i] = a[j][a_off + i];
        }
    }
}

pub fn SUNDlsMat_bandScale(
    c: sunrealtype,
    a: &mut [&mut [sunrealtype]],
    n: sunindextype,
    mu: sunindextype,
    ml: sunindextype,
    smu: sunindextype,
) {
    let colSize = (mu + ml + 1) as usize;
    let off = (smu - mu) as usize;
    for j in 0..n as usize {
        for i in 0..colSize {
            a[j][off + i] *= c;
        }
    }
}

pub fn SUNDlsMat_bandAddIdentity(a: &mut [&mut [sunrealtype]], n: sunindextype, smu: sunindextype) {
    for j in 0..n as usize {
        a[j][smu as usize] += ONE;
    }
}

pub fn SUNDlsMat_bandMatvec(
    a: &[&mut [sunrealtype]],
    x: &[sunrealtype],
    y: &mut [sunrealtype],
    n: sunindextype,
    mu: sunindextype,
    ml: sunindextype,
    smu: sunindextype,
) {
    for i in 0..n as usize {
        y[i] = ZERO;
    }
    for j in 0..n {
        let ju = j as usize;
        let off = (smu - mu) as usize;
        let is = if 0 > j - mu { 0 } else { j - mu };
        let ie = if n - 1 < j + ml { n - 1 } else { j + ml };
        for i in is..=ie {
            y[i as usize] += a[ju][off + (i - j + mu) as usize] * x[j as usize];
        }
    }
}
