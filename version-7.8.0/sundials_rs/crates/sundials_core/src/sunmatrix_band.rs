//! Port of `src/sunmatrix/band/sunmatrix_band.c` +
//! `include/sunmatrix/sunmatrix_band.h` (banded SUNMatrix, LAPACK band
//! storage).
//!
//! `SM_ELEMENT_B(A,i,j)` = `data[j*ldim + (i - j + s_mu)]`. The C
//! `SM_COLUMN_B` pointer supports negative indexing from the diagonal;
//! Rust call sites use `SM_COLUMN_B_ELEM(A, j, i_off)` with
//! `i_off ∈ [-mu, ml]` or direct flat indexing.

use std::cell::RefMut;

use crate::sundials_context::SUNContext;
use crate::sundials_errors::{SUN_ERR_ARG_DIMSMISMATCH, SUN_SUCCESS};
use crate::sundials_math::{SUNMAX, SUNMIN};
use crate::sundials_matrix::*;
use crate::sundials_nvector::{N_VGetArrayPointer, N_VGetLength, N_Vector};
use crate::sundials_types::*;
use crate::sundials_utils::{sun_format_e, SUNFile};

const ZERO: sunrealtype = 0.0;
const ONE: sunrealtype = 1.0;

pub struct SUNMatrixContent_Band_ {
    pub M: sunindextype,
    pub N: sunindextype,
    pub ldim: sunindextype,
    pub mu: sunindextype,
    pub ml: sunindextype,
    pub s_mu: sunindextype,
    pub ldata: sunindextype,
    pub data: Vec<sunrealtype>,
}

pub type SUNMatrixContent_Band = SUNMatrixContent_Band_;

fn content_mut(A: &SUNMatrix) -> RefMut<'_, SUNMatrixContent_Band_> {
    RefMut::map(A.content.borrow_mut(), |c| {
        c.downcast_mut::<SUNMatrixContent_Band_>()
            .expect("band SUNMatrix content")
    })
}

pub fn SM_ROWS_B(A: &SUNMatrix) -> sunindextype {
    content_mut(A).M
}

pub fn SM_COLUMNS_B(A: &SUNMatrix) -> sunindextype {
    content_mut(A).N
}

pub fn SM_LBAND_B(A: &SUNMatrix) -> sunindextype {
    content_mut(A).ml
}

pub fn SM_UBAND_B(A: &SUNMatrix) -> sunindextype {
    content_mut(A).mu
}

pub fn SM_SUBAND_B(A: &SUNMatrix) -> sunindextype {
    content_mut(A).s_mu
}

pub fn SM_LDIM_B(A: &SUNMatrix) -> sunindextype {
    content_mut(A).ldim
}

pub fn SM_LDATA_B(A: &SUNMatrix) -> sunindextype {
    content_mut(A).ldata
}

pub fn SM_DATA_B(A: &SUNMatrix) -> RefMut<'_, Vec<sunrealtype>> {
    RefMut::map(A.content.borrow_mut(), |c| {
        &mut c
            .downcast_mut::<SUNMatrixContent_Band_>()
            .expect("band SUNMatrix content")
            .data
    })
}

/// C `SM_COLUMN_B(A, j)` — the j-th column as a slice with element
/// `[k]` at storage row `k` (the C pointer offset by `s_mu` is applied by
/// callers through `SM_COLUMN_ELEMENT_B`).
pub fn SM_COLUMN_B(A: &SUNMatrix, j: sunindextype) -> RefMut<'_, [sunrealtype]> {
    RefMut::map(A.content.borrow_mut(), |c| {
        let content = c
            .downcast_mut::<SUNMatrixContent_Band_>()
            .expect("band SUNMatrix content");
        let ldim = content.ldim as usize;
        &mut content.data[(j as usize) * ldim..(j as usize) * ldim + ldim]
    })
}

/// C `SM_COLUMN_ELEMENT_B(col_j, i, j)`: index `i - j + s_mu` within the
/// raw column slice returned by [`SM_COLUMN_B`].
pub fn SM_COLUMN_ELEMENT_IDX(
    i: sunindextype,
    j: sunindextype,
    s_mu: sunindextype,
) -> usize {
    (i - j + s_mu) as usize
}

/// C `SM_ELEMENT_B(A, i, j)` read.
pub fn SM_ELEMENT_B(A: &SUNMatrix, i: sunindextype, j: sunindextype) -> sunrealtype {
    let content = content_mut(A);
    content.data[(j * content.ldim + i - j + content.s_mu) as usize]
}

/// C `SM_ELEMENT_B(A, i, j) = v` write.
pub fn SM_ELEMENT_B_set(A: &SUNMatrix, i: sunindextype, j: sunindextype, v: sunrealtype) {
    let mut content = content_mut(A);
    let idx = (j * content.ldim + i - j + content.s_mu) as usize;
    content.data[idx] = v;
}

pub fn SUNBandMatrix(
    N: sunindextype,
    mu: sunindextype,
    ml: sunindextype,
    sunctx: &SUNContext,
) -> Option<SUNMatrix> {
    SUNBandMatrixStorage(N, mu, ml, mu + ml, sunctx)
}

pub fn SUNBandMatrixStorage(
    N: sunindextype,
    mu: sunindextype,
    ml: sunindextype,
    smu: sunindextype,
    sunctx: &SUNContext,
) -> Option<SUNMatrix> {
    if N <= 0 || smu < 0 || ml < 0 {
        return None;
    }

    let A = SUNMatNewEmpty(sunctx)?;

    /* Attach operations */
    {
        let mut ops = A.ops.borrow_mut();
        ops.getid = Some(SUNMatGetID_Band);
        ops.clone = Some(SUNMatClone_Band);
        ops.destroy = Some(SUNMatDestroy_Band);
        ops.zero = Some(SUNMatZero_Band);
        ops.copy = Some(SUNMatCopy_Band);
        ops.scaleadd = Some(SUNMatScaleAdd_Band);
        ops.scaleaddi = Some(SUNMatScaleAddI_Band);
        ops.matvec = Some(SUNMatMatvec_Band);
        ops.mathermitiantransposevec = Some(SUNMatHermitianTransposeVec_Band);
        ops.space = Some(SUNMatSpace_Band);
    }

    /* Create and fill content (data calloc'd: zero-initialized) */
    let colSize = smu + ml + 1;
    *A.content.borrow_mut() = Box::new(SUNMatrixContent_Band_ {
        M: N,
        N,
        mu,
        ml,
        s_mu: smu,
        ldim: colSize,
        ldata: N * colSize,
        data: vec![0.0; (N * colSize) as usize],
    });

    Some(A)
}

pub fn SUNBandMatrix_Print(A: &SUNMatrix, outfile: &SUNFile) {
    outfile.write_str("\n");
    for i in 0..SM_ROWS_B(A) {
        let start = SUNMAX(0, i - SM_LBAND_B(A));
        let finish = SUNMIN(SM_COLUMNS_B(A) - 1, i + SM_UBAND_B(A));
        for _ in 0..start {
            outfile.write_str(&format!("{:12}  ", ""));
        }
        for j in start..=finish {
            outfile.write_str(&format!("{}  ", sun_format_e(SM_ELEMENT_B(A, i, j))));
        }
        outfile.write_str("\n");
    }
}

pub fn SUNBandMatrix_Rows(A: &SUNMatrix) -> sunindextype {
    SM_ROWS_B(A)
}

pub fn SUNBandMatrix_Columns(A: &SUNMatrix) -> sunindextype {
    SM_COLUMNS_B(A)
}

pub fn SUNBandMatrix_LowerBandwidth(A: &SUNMatrix) -> sunindextype {
    SM_LBAND_B(A)
}

pub fn SUNBandMatrix_UpperBandwidth(A: &SUNMatrix) -> sunindextype {
    SM_UBAND_B(A)
}

pub fn SUNBandMatrix_StoredUpperBandwidth(A: &SUNMatrix) -> sunindextype {
    SM_SUBAND_B(A)
}

pub fn SUNBandMatrix_LDim(A: &SUNMatrix) -> sunindextype {
    SM_LDIM_B(A)
}

pub fn SUNBandMatrix_LData(A: &SUNMatrix) -> sunindextype {
    SM_LDATA_B(A)
}

pub fn SUNBandMatrix_Data(A: &SUNMatrix) -> RefMut<'_, Vec<sunrealtype>> {
    SM_DATA_B(A)
}

pub fn SUNBandMatrix_Column(A: &SUNMatrix, j: sunindextype) -> RefMut<'_, [sunrealtype]> {
    SM_COLUMN_B(A, j)
}

pub fn SUNMatGetID_Band(_A: &SUNMatrix) -> SUNMatrix_ID {
    SUNMATRIX_BAND
}

pub fn SUNMatClone_Band(A: &SUNMatrix) -> Option<SUNMatrix> {
    SUNBandMatrixStorage(
        SM_COLUMNS_B(A),
        SM_UBAND_B(A),
        SM_LBAND_B(A),
        SM_SUBAND_B(A),
        &A.sunctx.borrow(),
    )
}

pub fn SUNMatDestroy_Band(A: SUNMatrix) {
    drop(A);
}

pub fn SUNMatZero_Band(A: &SUNMatrix) -> SUNErrCode {
    let mut content = content_mut(A);
    let ldata = content.ldata as usize;
    for i in 0..ldata {
        content.data[i] = ZERO;
    }
    SUN_SUCCESS
}

pub fn SUNMatCopy_Band(A: &SUNMatrix, B: &SUNMatrix) -> SUNErrCode {
    if !compatibleMatrices(A, B) {
        return SUN_ERR_ARG_DIMSMISMATCH;
    }

    /* Grow B if A's bandwidth is larger */
    if (SM_UBAND_B(A) > SM_UBAND_B(B)) || (SM_LBAND_B(A) > SM_LBAND_B(B)) {
        let ml = SUNMAX(SM_LBAND_B(B), SM_LBAND_B(A));
        let mu = SUNMAX(SM_UBAND_B(B), SM_UBAND_B(A));
        let smu = SUNMAX(SM_SUBAND_B(B), SM_SUBAND_B(A));
        let colSize = smu + ml + 1;
        let mut bc = content_mut(B);
        bc.mu = mu;
        bc.ml = ml;
        bc.s_mu = smu;
        bc.ldim = colSize;
        bc.ldata = bc.N * colSize;
        let new_len = (bc.N * colSize) as usize;
        bc.data.resize(new_len, 0.0);
    }

    /* Perform operation */
    let ier = SUNMatZero_Band(B);
    if ier != SUN_SUCCESS {
        return ier;
    }
    let a = content_mut(A);
    let mut b = content_mut(B);
    let (b_ldim, b_smu) = (b.ldim, b.s_mu);
    for j in 0..b.N {
        let a_base = j * a.ldim + a.s_mu;
        let b_base = j * b_ldim + b_smu;
        let mut i = -a.mu;
        while i <= a.ml {
            b.data[(b_base + i) as usize] = a.data[(a_base + i) as usize];
            i += 1;
        }
    }
    SUN_SUCCESS
}

pub fn SUNMatScaleAddI_Band(c: sunrealtype, A: &SUNMatrix) -> SUNErrCode {
    let mut a = content_mut(A);
    let (n, mu, ml, s_mu, ldim) = (a.N, a.mu, a.ml, a.s_mu, a.ldim);
    for j in 0..n {
        let base = j * ldim + s_mu;
        let mut i = -mu;
        while i <= ml {
            a.data[(base + i) as usize] *= c;
            i += 1;
        }
        /* SM_ELEMENT_B(A, j, j) += ONE */
        a.data[base as usize] += ONE;
    }
    SUN_SUCCESS
}

pub fn SUNMatScaleAdd_Band(c: sunrealtype, A: &SUNMatrix, B: &SUNMatrix) -> SUNErrCode {
    if !compatibleMatrices(A, B) {
        return SUN_ERR_ARG_DIMSMISMATCH;
    }

    /* Call separate routine if B has larger bandwidth(s) than A */
    if (SM_UBAND_B(B) > SM_UBAND_B(A)) || (SM_LBAND_B(B) > SM_LBAND_B(A)) {
        return SMScaleAddNew_Band(c, A, B);
    }

    /* Otherwise, perform operation in-place */
    let mut a = content_mut(A);
    let b = content_mut(B);
    let (n, a_ldim, a_smu) = (a.N, a.ldim, a.s_mu);
    let (b_ldim, b_smu, b_mu, b_ml) = (b.ldim, b.s_mu, b.mu, b.ml);
    for j in 0..n {
        let a_base = j * a_ldim + a_smu;
        let b_base = j * b_ldim + b_smu;
        let mut i = -b_mu;
        while i <= b_ml {
            a.data[(a_base + i) as usize] =
                c * a.data[(a_base + i) as usize] + b.data[(b_base + i) as usize];
            i += 1;
        }
    }
    SUN_SUCCESS
}

pub fn SUNMatMatvec_Band(A: &SUNMatrix, x: &N_Vector, y: &N_Vector) -> SUNErrCode {
    if !compatibleMatrixAndVectors(A, x, y) {
        return SUN_ERR_ARG_DIMSMISMATCH;
    }

    let a = content_mut(A);
    let xd = N_VGetArrayPointer(x).expect("band matvec x data");
    let mut yd = N_VGetArrayPointer(y).expect("band matvec y data");

    /* Perform operation */
    let (m, n, mu, ml, s_mu, ldim) = (a.M, a.N, a.mu, a.ml, a.s_mu, a.ldim);
    for i in 0..m as usize {
        yd[i] = ZERO;
    }
    for j in 0..n {
        let base = j * ldim + s_mu - j;
        let is = SUNMAX(0, j - mu);
        let ie = SUNMIN(m - 1, j + ml);
        for i in is..=ie {
            yd[i as usize] += a.data[(base + i) as usize] * xd[j as usize];
        }
    }
    SUN_SUCCESS
}

pub fn SUNMatHermitianTransposeVec_Band(
    A: &SUNMatrix,
    x: &N_Vector,
    y: &N_Vector,
) -> SUNErrCode {
    if !compatibleMatrixAndVectors(A, y, x) {
        return SUN_ERR_ARG_DIMSMISMATCH;
    }

    let a = content_mut(A);
    let xd = N_VGetArrayPointer(x).expect("band hermitian x data");
    let mut yd = N_VGetArrayPointer(y).expect("band hermitian y data");

    /* Perform operation */
    let (m, n, mu, ml, s_mu, ldim) = (a.M, a.N, a.mu, a.ml, a.s_mu, a.ldim);
    for i in 0..m as usize {
        yd[i] = ZERO;
    }
    for j in 0..n {
        let base = j * ldim + s_mu - j;
        let is = SUNMAX(0, j - mu);
        let ie = SUNMIN(m - 1, j + ml);
        for i in is..=ie {
            yd[j as usize] += a.data[(base + i) as usize] * xd[i as usize];
        }
    }
    SUN_SUCCESS
}

pub fn SUNMatSpace_Band(A: &SUNMatrix, lenrw: &mut i64, leniw: &mut i64) -> SUNErrCode {
    *lenrw = SM_COLUMNS_B(A) * (SM_SUBAND_B(A) + SM_LBAND_B(A) + 1);
    *leniw = 7 + SM_COLUMNS_B(A);
    SUN_SUCCESS
}

fn compatibleMatrices(A: &SUNMatrix, B: &SUNMatrix) -> sunbooleantype {
    /* both matrices must have the same shape
    (note that we do not check for identical bandwidth) */
    if SM_ROWS_B(A) != SM_ROWS_B(B) {
        return SUNFALSE;
    }
    if SM_COLUMNS_B(A) != SM_COLUMNS_B(B) {
        return SUNFALSE;
    }
    SUNTRUE
}

fn compatibleMatrixAndVectors(A: &SUNMatrix, x: &N_Vector, y: &N_Vector) -> sunbooleantype {
    if x.ops.borrow().nvgetarraypointer.is_none() || y.ops.borrow().nvgetarraypointer.is_none() {
        return SUNFALSE;
    }
    if (N_VGetLength(x) != SM_COLUMNS_B(A)) || (N_VGetLength(y) != SM_ROWS_B(A)) {
        return SUNFALSE;
    }
    SUNTRUE
}

fn SMScaleAddNew_Band(c: sunrealtype, A: &SUNMatrix, B: &SUNMatrix) -> SUNErrCode {
    /* create new matrix large enough to hold both A and B */
    let ml = SUNMAX(SM_LBAND_B(A), SM_LBAND_B(B));
    let mu = SUNMAX(SM_UBAND_B(A), SM_UBAND_B(B));
    let smu = SUNMIN(SM_COLUMNS_B(A) - 1, mu + ml);
    let C = match SUNBandMatrixStorage(SM_COLUMNS_B(A), mu, ml, smu, &A.sunctx.borrow()) {
        Some(C) => C,
        None => return crate::sundials_errors::SUN_ERR_MEM_FAIL,
    };

    {
        let a = content_mut(A);
        let mut cc = content_mut(&C);
        let (c_ldim, c_smu) = (cc.ldim, cc.s_mu);

        /* scale/add c*A into new matrix */
        for j in 0..a.N {
            let a_base = j * a.ldim + a.s_mu;
            let c_base = j * c_ldim + c_smu;
            let mut i = -a.mu;
            while i <= a.ml {
                cc.data[(c_base + i) as usize] = c * a.data[(a_base + i) as usize];
                i += 1;
            }
        }

        /* add B into new matrix */
        let b = content_mut(B);
        for j in 0..b.N {
            let b_base = j * b.ldim + b.s_mu;
            let c_base = j * c_ldim + c_smu;
            let mut i = -b.mu;
            while i <= b.ml {
                cc.data[(c_base + i) as usize] += b.data[(b_base + i) as usize];
                i += 1;
            }
        }
    }

    /* replace A contents with C contents */
    let c_content = std::mem::replace(&mut *C.content.borrow_mut(), Box::new(()));
    *A.content.borrow_mut() = c_content;

    SUN_SUCCESS
}
