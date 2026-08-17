//! Port of `src/sunmatrix/dense/sunmatrix_dense.c` +
//! `include/sunmatrix/sunmatrix_dense.h` (column-major dense SUNMatrix).
//!
//! `SM_ELEMENT_D(A,i,j)` = `data[j*M + i]`; the C `cols` pointer array is
//! implicit in the flat layout.

use std::cell::RefMut;

use crate::sundials_context::SUNContext;
use crate::sundials_errors::{SUN_ERR_ARG_DIMSMISMATCH, SUN_SUCCESS};
use crate::sundials_matrix::*;
use crate::sundials_nvector::{N_VGetArrayPointer, N_VGetLength, N_Vector};
use crate::sundials_types::*;
use crate::sundials_utils::{sun_format_e, SUNFile};

const ZERO: sunrealtype = 0.0;
const ONE: sunrealtype = 1.0;

pub struct SUNMatrixContent_Dense_ {
    pub M: sunindextype,
    pub N: sunindextype,
    pub ldata: sunindextype,
    pub data: Vec<sunrealtype>,
}

pub type SUNMatrixContent_Dense = SUNMatrixContent_Dense_;

fn content_mut(A: &SUNMatrix) -> RefMut<'_, SUNMatrixContent_Dense_> {
    RefMut::map(A.content.borrow_mut(), |c| {
        c.downcast_mut::<SUNMatrixContent_Dense_>()
            .expect("dense SUNMatrix content")
    })
}

pub fn SM_ROWS_D(A: &SUNMatrix) -> sunindextype {
    content_mut(A).M
}

pub fn SM_COLUMNS_D(A: &SUNMatrix) -> sunindextype {
    content_mut(A).N
}

pub fn SM_LDATA_D(A: &SUNMatrix) -> sunindextype {
    content_mut(A).ldata
}

/// C `SM_DATA_D(A)` (flat column-major data).
pub fn SM_DATA_D(A: &SUNMatrix) -> RefMut<'_, Vec<sunrealtype>> {
    RefMut::map(A.content.borrow_mut(), |c| {
        &mut c
            .downcast_mut::<SUNMatrixContent_Dense_>()
            .expect("dense SUNMatrix content")
            .data
    })
}

/// C `SM_COLUMN_D(A, j)` (the j-th column as a mutable slice).
pub fn SM_COLUMN_D(A: &SUNMatrix, j: sunindextype) -> RefMut<'_, [sunrealtype]> {
    RefMut::map(A.content.borrow_mut(), |c| {
        let content = c
            .downcast_mut::<SUNMatrixContent_Dense_>()
            .expect("dense SUNMatrix content");
        let m = content.M as usize;
        &mut content.data[(j as usize) * m..(j as usize) * m + m]
    })
}

/// C `SM_ELEMENT_D(A, i, j)` read.
pub fn SM_ELEMENT_D(A: &SUNMatrix, i: sunindextype, j: sunindextype) -> sunrealtype {
    let content = content_mut(A);
    content.data[(j * content.M + i) as usize]
}

/// C `SM_ELEMENT_D(A, i, j) = v` write.
pub fn SM_ELEMENT_D_set(A: &SUNMatrix, i: sunindextype, j: sunindextype, v: sunrealtype) {
    let mut content = content_mut(A);
    let m = content.M;
    content.data[(j * m + i) as usize] = v;
}

pub fn SUNDenseMatrix(M: sunindextype, N: sunindextype, sunctx: &SUNContext) -> Option<SUNMatrix> {
    /* return with NULL matrix on illegal dimension input */
    if N <= 0 || M <= 0 {
        return None;
    }

    let A = SUNMatNewEmpty(sunctx)?;

    /* Attach operations */
    {
        let mut ops = A.ops.borrow_mut();
        ops.getid = Some(SUNMatGetID_Dense);
        ops.clone = Some(SUNMatClone_Dense);
        ops.destroy = Some(SUNMatDestroy_Dense);
        ops.zero = Some(SUNMatZero_Dense);
        ops.copy = Some(SUNMatCopy_Dense);
        ops.scaleadd = Some(SUNMatScaleAdd_Dense);
        ops.scaleaddi = Some(SUNMatScaleAddI_Dense);
        ops.matvec = Some(SUNMatMatvec_Dense);
        ops.mathermitiantransposevec = Some(SUNMatHermitianTransposeVec_Dense);
        ops.space = Some(SUNMatSpace_Dense);
    }

    /* Create and fill content (data is calloc'd in C: zero-initialized) */
    *A.content.borrow_mut() = Box::new(SUNMatrixContent_Dense_ {
        M,
        N,
        ldata: M * N,
        data: vec![0.0; (M * N) as usize],
    });

    Some(A)
}

pub fn SUNDenseMatrix_Print(A: &SUNMatrix, outfile: &SUNFile) {
    outfile.write_str("\n");
    for i in 0..SM_ROWS_D(A) {
        for j in 0..SM_COLUMNS_D(A) {
            outfile.write_str(&format!("{}  ", sun_format_e(SM_ELEMENT_D(A, i, j))));
        }
        outfile.write_str("\n");
    }
}

pub fn SUNDenseMatrix_Rows(A: &SUNMatrix) -> sunindextype {
    SM_ROWS_D(A)
}

pub fn SUNDenseMatrix_Columns(A: &SUNMatrix) -> sunindextype {
    SM_COLUMNS_D(A)
}

pub fn SUNDenseMatrix_LData(A: &SUNMatrix) -> sunindextype {
    SM_LDATA_D(A)
}

pub fn SUNDenseMatrix_Data(A: &SUNMatrix) -> RefMut<'_, Vec<sunrealtype>> {
    SM_DATA_D(A)
}

pub fn SUNDenseMatrix_Column(A: &SUNMatrix, j: sunindextype) -> RefMut<'_, [sunrealtype]> {
    SM_COLUMN_D(A, j)
}

pub fn SUNMatGetID_Dense(_A: &SUNMatrix) -> SUNMatrix_ID {
    SUNMATRIX_DENSE
}

pub fn SUNMatClone_Dense(A: &SUNMatrix) -> Option<SUNMatrix> {
    SUNDenseMatrix(SM_ROWS_D(A), SM_COLUMNS_D(A), &A.sunctx.borrow())
}

pub fn SUNMatDestroy_Dense(A: SUNMatrix) {
    drop(A);
}

pub fn SUNMatZero_Dense(A: &SUNMatrix) -> SUNErrCode {
    let mut content = content_mut(A);
    let ldata = content.ldata as usize;
    for i in 0..ldata {
        content.data[i] = ZERO;
    }
    SUN_SUCCESS
}

pub fn SUNMatCopy_Dense(A: &SUNMatrix, B: &SUNMatrix) -> SUNErrCode {
    if !compatibleMatrices(A, B) {
        return SUN_ERR_ARG_DIMSMISMATCH;
    }

    /* Perform operation B_ij = A_ij */
    let a = content_mut(A);
    let mut b = content_mut(B);
    let (m, n) = (a.M, a.N);
    for j in 0..n {
        for i in 0..m {
            b.data[(j * m + i) as usize] = a.data[(j * m + i) as usize];
        }
    }
    SUN_SUCCESS
}

pub fn SUNMatScaleAddI_Dense(c: sunrealtype, A: &SUNMatrix) -> SUNErrCode {
    let mut a = content_mut(A);
    let (m, n) = (a.M, a.N);
    /* Perform operation A = c*A + I */
    for j in 0..n {
        for i in 0..m {
            a.data[(j * m + i) as usize] *= c;
            if i == j {
                a.data[(j * m + i) as usize] += ONE;
            }
        }
    }
    SUN_SUCCESS
}

pub fn SUNMatScaleAdd_Dense(c: sunrealtype, A: &SUNMatrix, B: &SUNMatrix) -> SUNErrCode {
    if !compatibleMatrices(A, B) {
        return SUN_ERR_ARG_DIMSMISMATCH;
    }

    /* Perform operation A = c*A + B */
    let mut a = content_mut(A);
    let b = content_mut(B);
    let (m, n) = (a.M, a.N);
    for j in 0..n {
        for i in 0..m {
            let idx = (j * m + i) as usize;
            a.data[idx] = c * a.data[idx] + b.data[idx];
        }
    }
    SUN_SUCCESS
}

pub fn SUNMatMatvec_Dense(A: &SUNMatrix, x: &N_Vector, y: &N_Vector) -> SUNErrCode {
    if !compatibleMatrixAndVectors(A, x, y) {
        return SUN_ERR_ARG_DIMSMISMATCH;
    }

    let a = content_mut(A);
    let xd = N_VGetArrayPointer(x).expect("dense matvec x data");
    let mut yd = N_VGetArrayPointer(y).expect("dense matvec y data");

    /* Perform operation y = Ax */
    let (m, n) = (a.M, a.N);
    for i in 0..m as usize {
        yd[i] = ZERO;
    }
    for j in 0..n {
        let col_j = &a.data[(j * m) as usize..(j * m + m) as usize];
        for i in 0..m as usize {
            yd[i] += col_j[i] * xd[j as usize];
        }
    }
    SUN_SUCCESS
}

pub fn SUNMatHermitianTransposeVec_Dense(
    A: &SUNMatrix,
    x: &N_Vector,
    y: &N_Vector,
) -> SUNErrCode {
    if !compatibleMatrixAndVectors(A, y, x) {
        return SUN_ERR_ARG_DIMSMISMATCH;
    }

    let a = content_mut(A);
    let xd = N_VGetArrayPointer(x).expect("dense hermitian x data");
    let mut yd = N_VGetArrayPointer(y).expect("dense hermitian y data");

    /* Perform operation y = A^T x */
    let (m, n) = (a.M, a.N);
    for i in 0..n as usize {
        yd[i] = ZERO;
    }
    for i in 0..n {
        let row_i = &a.data[(i * m) as usize..(i * m + m) as usize];
        for j in 0..m as usize {
            yd[i as usize] += row_i[j] * xd[j];
        }
    }
    SUN_SUCCESS
}

pub fn SUNMatSpace_Dense(A: &SUNMatrix, lenrw: &mut i64, leniw: &mut i64) -> SUNErrCode {
    *lenrw = SM_LDATA_D(A);
    *leniw = 3 + SM_COLUMNS_D(A);
    SUN_SUCCESS
}

fn compatibleMatrices(A: &SUNMatrix, B: &SUNMatrix) -> sunbooleantype {
    /* both matrices must have the same shape */
    if (SM_ROWS_D(A) != SM_ROWS_D(B)) || (SM_COLUMNS_D(A) != SM_COLUMNS_D(B)) {
        return SUNFALSE;
    }
    SUNTRUE
}

fn compatibleMatrixAndVectors(A: &SUNMatrix, x: &N_Vector, y: &N_Vector) -> sunbooleantype {
    /* Vectors must provide nvgetarraypointer */
    if x.ops.borrow().nvgetarraypointer.is_none() || y.ops.borrow().nvgetarraypointer.is_none() {
        return SUNFALSE;
    }
    /* Check that the dimensions agree */
    if (N_VGetLength(x) != SM_COLUMNS_D(A)) || (N_VGetLength(y) != SM_ROWS_D(A)) {
        return SUNFALSE;
    }
    SUNTRUE
}
