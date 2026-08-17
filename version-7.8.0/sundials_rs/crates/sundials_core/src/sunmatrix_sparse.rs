//! Port of `src/sunmatrix/sparse/sunmatrix_sparse.c` +
//! `include/sunmatrix/sunmatrix_sparse.h` (CSC/CSR sparse SUNMatrix).
//!
//! The C `rowvals`/`colptrs` (CSC) and `colvals`/`rowptrs` (CSR) fields
//! are aliases of `indexvals`/`indexptrs`; only the canonical arrays are
//! stored. All loop variables stay `sunindextype` (i64) to preserve C's
//! backward loops and sentinel arithmetic exactly.

use std::cell::RefMut;

use crate::sundials_context::SUNContext;
use crate::sundials_errors::{SUN_ERR_ARG_DIMSMISMATCH, SUN_ERR_ARG_WRONGTYPE, SUN_SUCCESS};
use crate::sundials_math::{SUNMAX, SUNMIN, SUNRabs};
use crate::sundials_matrix::*;
use crate::sundials_nvector::{N_VGetArrayPointer, N_VGetLength, N_Vector};
use crate::sundials_types::*;
use crate::sunmatrix_band::{SM_COLUMNS_B, SM_ELEMENT_B, SM_LBAND_B, SM_ROWS_B, SM_UBAND_B};
use crate::sunmatrix_dense::{SM_COLUMNS_D, SM_ELEMENT_D, SM_ROWS_D};
use crate::sundials_utils::{sun_format_e, SUNFile};

const ZERO: sunrealtype = 0.0;
const ONE: sunrealtype = 1.0;

pub const SUN_CSC_MAT: i32 = 0;
pub const SUN_CSR_MAT: i32 = 1;

pub struct SUNMatrixContent_Sparse_ {
    pub M: sunindextype,
    pub N: sunindextype,
    pub NNZ: sunindextype,
    pub NP: sunindextype,
    pub data: Vec<sunrealtype>,
    pub sparsetype: i32,
    pub indexvals: Vec<sunindextype>,
    pub indexptrs: Vec<sunindextype>,
}

pub type SUNMatrixContent_Sparse = SUNMatrixContent_Sparse_;

fn content_mut(A: &SUNMatrix) -> RefMut<'_, SUNMatrixContent_Sparse_> {
    RefMut::map(A.content.borrow_mut(), |c| {
        c.downcast_mut::<SUNMatrixContent_Sparse_>()
            .expect("sparse SUNMatrix content")
    })
}

pub fn SM_ROWS_S(A: &SUNMatrix) -> sunindextype {
    content_mut(A).M
}

pub fn SM_COLUMNS_S(A: &SUNMatrix) -> sunindextype {
    content_mut(A).N
}

pub fn SM_NNZ_S(A: &SUNMatrix) -> sunindextype {
    content_mut(A).NNZ
}

pub fn SM_NP_S(A: &SUNMatrix) -> sunindextype {
    content_mut(A).NP
}

pub fn SM_SPARSETYPE_S(A: &SUNMatrix) -> i32 {
    content_mut(A).sparsetype
}

/// All three compressed-column arrays behind a single borrow.
///
/// `SM_DATA_S`, `SM_INDEXVALS_S` and `SM_INDEXPTRS_S` each take their own
/// `RefCell` borrow, so holding two of them at once panics. The C code they
/// translate holds all three pointers simultaneously — every sparse Jacobian
/// routine does — so callers that need more than one must come through here.
pub fn SM_CONTENT_S(A: &SUNMatrix) -> RefMut<'_, SUNMatrixContent_Sparse_> {
    content_mut(A)
}

/// Alias of [`SM_CONTENT_S`] in the `SUNSparseMatrix_*` naming style.
pub fn SUNSparseMatrix_Content(A: &SUNMatrix) -> RefMut<'_, SUNMatrixContent_Sparse_> {
    content_mut(A)
}

pub fn SM_DATA_S(A: &SUNMatrix) -> RefMut<'_, Vec<sunrealtype>> {
    RefMut::map(A.content.borrow_mut(), |c| {
        &mut c
            .downcast_mut::<SUNMatrixContent_Sparse_>()
            .expect("sparse SUNMatrix content")
            .data
    })
}

pub fn SM_INDEXVALS_S(A: &SUNMatrix) -> RefMut<'_, Vec<sunindextype>> {
    RefMut::map(A.content.borrow_mut(), |c| {
        &mut c
            .downcast_mut::<SUNMatrixContent_Sparse_>()
            .expect("sparse SUNMatrix content")
            .indexvals
    })
}

pub fn SM_INDEXPTRS_S(A: &SUNMatrix) -> RefMut<'_, Vec<sunindextype>> {
    RefMut::map(A.content.borrow_mut(), |c| {
        &mut c
            .downcast_mut::<SUNMatrixContent_Sparse_>()
            .expect("sparse SUNMatrix content")
            .indexptrs
    })
}

pub fn SUNSparseMatrix(
    M: sunindextype,
    N: sunindextype,
    NNZ: sunindextype,
    sparsetype: i32,
    sunctx: &SUNContext,
) -> Option<SUNMatrix> {
    /* return with NULL matrix on illegal input */
    if M <= 0 || N <= 0 || NNZ < 0 || (sparsetype != SUN_CSC_MAT && sparsetype != SUN_CSR_MAT) {
        return None;
    }

    let A = SUNMatNewEmpty(sunctx)?;

    /* Attach operations */
    {
        let mut ops = A.ops.borrow_mut();
        ops.getid = Some(SUNMatGetID_Sparse);
        ops.clone = Some(SUNMatClone_Sparse);
        ops.destroy = Some(SUNMatDestroy_Sparse);
        ops.zero = Some(SUNMatZero_Sparse);
        ops.copy = Some(SUNMatCopy_Sparse);
        ops.scaleadd = Some(SUNMatScaleAdd_Sparse);
        ops.scaleaddi = Some(SUNMatScaleAddI_Sparse);
        ops.matvec = Some(SUNMatMatvec_Sparse);
        ops.mathermitiantransposevec = Some(SUNMatHermitianTransposeVec_Sparse);
        ops.space = Some(SUNMatSpace_Sparse);
    }

    let NP = if sparsetype == SUN_CSC_MAT { N } else { M };

    /* Create and fill content (arrays calloc'd: zero-initialized) */
    *A.content.borrow_mut() = Box::new(SUNMatrixContent_Sparse_ {
        sparsetype,
        M,
        N,
        NNZ,
        NP,
        data: vec![0.0; NNZ as usize],
        indexvals: vec![0; NNZ as usize],
        indexptrs: vec![0; (NP + 1) as usize],
    });

    Some(A)
}

pub fn SUNSparseFromDenseMatrix(
    Ad: &SUNMatrix,
    droptol: sunrealtype,
    sparsetype: i32,
) -> Option<SUNMatrix> {
    if SUNMatGetID(Ad) != SUNMATRIX_DENSE {
        return None;
    }
    if (sparsetype != SUN_CSC_MAT && sparsetype != SUN_CSR_MAT) || droptol < ZERO {
        return None;
    }

    /* set size of new matrix */
    let M = SM_ROWS_D(Ad);
    let N = SM_COLUMNS_D(Ad);

    /* determine total number of nonzeros */
    let mut nnz: sunindextype = 0;
    for j in 0..N {
        for i in 0..M {
            nnz += (SUNRabs(SM_ELEMENT_D(Ad, i, j)) > droptol) as sunindextype;
        }
    }

    /* allocate sparse matrix */
    let As = SUNSparseMatrix(M, N, nnz, sparsetype, &Ad.sunctx.borrow())?;

    /* copy nonzeros from Ad into As, based on CSR/CSC type */
    let mut nnz: sunindextype = 0;
    {
        let mut asc = content_mut(&As);
        if sparsetype == SUN_CSC_MAT {
            for j in 0..N {
                asc.indexptrs[j as usize] = nnz;
                for i in 0..M {
                    if SUNRabs(SM_ELEMENT_D(Ad, i, j)) > droptol {
                        asc.indexvals[nnz as usize] = i;
                        asc.data[nnz as usize] = SM_ELEMENT_D(Ad, i, j);
                        nnz += 1;
                    }
                }
            }
            asc.indexptrs[N as usize] = nnz;
        } else {
            /* SUN_CSR_MAT */
            for i in 0..M {
                asc.indexptrs[i as usize] = nnz;
                for j in 0..N {
                    if SUNRabs(SM_ELEMENT_D(Ad, i, j)) > droptol {
                        asc.indexvals[nnz as usize] = j;
                        asc.data[nnz as usize] = SM_ELEMENT_D(Ad, i, j);
                        nnz += 1;
                    }
                }
            }
            asc.indexptrs[M as usize] = nnz;
        }
    }

    Some(As)
}

pub fn SUNSparseFromBandMatrix(
    Ab: &SUNMatrix,
    droptol: sunrealtype,
    sparsetype: i32,
) -> Option<SUNMatrix> {
    if SUNMatGetID(Ab) != SUNMATRIX_BAND {
        return None;
    }
    if (sparsetype != SUN_CSC_MAT && sparsetype != SUN_CSR_MAT) || droptol < ZERO {
        return None;
    }

    /* set size of new matrix */
    let M = SM_ROWS_B(Ab);
    let N = SM_COLUMNS_B(Ab);

    /* determine total number of nonzeros */
    let mut nnz: sunindextype = 0;
    for j in 0..N {
        let mut i = SUNMAX(0, j - SM_UBAND_B(Ab));
        while i <= SUNMIN(M - 1, j + SM_LBAND_B(Ab)) {
            nnz += (SUNRabs(SM_ELEMENT_B(Ab, i, j)) > droptol) as sunindextype;
            i += 1;
        }
    }

    /* allocate sparse matrix */
    let As = SUNSparseMatrix(M, N, nnz, sparsetype, &Ab.sunctx.borrow())?;

    /* copy nonzeros from Ab into As, based on CSR/CSC type */
    let mut nnz: sunindextype = 0;
    {
        let mut asc = content_mut(&As);
        if sparsetype == SUN_CSC_MAT {
            for j in 0..N {
                asc.indexptrs[j as usize] = nnz;
                let mut i = SUNMAX(0, j - SM_UBAND_B(Ab));
                while i <= SUNMIN(M - 1, j + SM_LBAND_B(Ab)) {
                    if SUNRabs(SM_ELEMENT_B(Ab, i, j)) > droptol {
                        asc.indexvals[nnz as usize] = i;
                        asc.data[nnz as usize] = SM_ELEMENT_B(Ab, i, j);
                        nnz += 1;
                    }
                    i += 1;
                }
            }
            asc.indexptrs[N as usize] = nnz;
        } else {
            /* SUN_CSR_MAT */
            for i in 0..M {
                asc.indexptrs[i as usize] = nnz;
                let mut j = SUNMAX(0, i - SM_LBAND_B(Ab));
                while j <= SUNMIN(N - 1, i + SM_UBAND_B(Ab)) {
                    if SUNRabs(SM_ELEMENT_B(Ab, i, j)) > droptol {
                        asc.indexvals[nnz as usize] = j;
                        asc.data[nnz as usize] = SM_ELEMENT_B(Ab, i, j);
                        nnz += 1;
                    }
                    j += 1;
                }
            }
            asc.indexptrs[M as usize] = nnz;
        }
    }

    Some(As)
}

pub fn SUNSparseMatrix_ToCSR(A: &SUNMatrix, Bout: &mut Option<SUNMatrix>) -> SUNErrCode {
    if SUNMatGetID(A) != SUNMATRIX_SPARSE || SM_SPARSETYPE_S(A) != SUN_CSC_MAT {
        return SUN_ERR_ARG_WRONGTYPE;
    }
    *Bout = SUNSparseMatrix(
        SM_ROWS_S(A),
        SM_COLUMNS_S(A),
        SM_NNZ_S(A),
        SUN_CSR_MAT,
        &A.sunctx.borrow(),
    );
    match Bout {
        Some(B) => format_convert(A, B),
        None => crate::sundials_errors::SUN_ERR_MEM_FAIL,
    }
}

pub fn SUNSparseMatrix_ToCSC(A: &SUNMatrix, Bout: &mut Option<SUNMatrix>) -> SUNErrCode {
    if SUNMatGetID(A) != SUNMATRIX_SPARSE || SM_SPARSETYPE_S(A) != SUN_CSR_MAT {
        return SUN_ERR_ARG_WRONGTYPE;
    }
    *Bout = SUNSparseMatrix(
        SM_ROWS_S(A),
        SM_COLUMNS_S(A),
        SM_NNZ_S(A),
        SUN_CSC_MAT,
        &A.sunctx.borrow(),
    );
    match Bout {
        Some(B) => format_convert(A, B),
        None => crate::sundials_errors::SUN_ERR_MEM_FAIL,
    }
}

pub fn SUNSparseMatrix_Realloc(A: &SUNMatrix) -> SUNErrCode {
    let mut c = content_mut(A);
    /* get total number of nonzeros */
    let nzmax = c.indexptrs[c.NP as usize];
    if nzmax < 0 {
        return crate::sundials_errors::SUN_ERR_ARG_CORRUPT;
    }
    /* perform reallocation */
    c.indexvals.resize(nzmax as usize, 0);
    c.data.resize(nzmax as usize, 0.0);
    c.NNZ = nzmax;
    SUN_SUCCESS
}

pub fn SUNSparseMatrix_Reallocate(A: &SUNMatrix, NNZ: sunindextype) -> SUNErrCode {
    if NNZ < 0 {
        return crate::sundials_errors::SUN_ERR_ARG_OUTOFRANGE;
    }
    let mut c = content_mut(A);
    c.indexvals.resize(NNZ as usize, 0);
    c.data.resize(NNZ as usize, 0.0);
    c.NNZ = NNZ;
    SUN_SUCCESS
}

pub fn SUNSparseMatrix_Print(A: &SUNMatrix, outfile: &SUNFile) {
    let (indexname, matrixtype) = if SM_SPARSETYPE_S(A) == SUN_CSC_MAT {
        ("col", "CSC")
    } else {
        ("row", "CSR")
    };
    outfile.write_str("\n");
    outfile.write_str(&format!(
        "{} by {} {} matrix, NNZ: {} \n",
        SM_ROWS_S(A),
        SM_COLUMNS_S(A),
        matrixtype,
        SM_NNZ_S(A)
    ));
    let c = content_mut(A);
    for j in 0..c.NP {
        outfile.write_str(&format!(
            "{} {} : locations {} to {}\n",
            indexname,
            j,
            c.indexptrs[j as usize],
            c.indexptrs[(j + 1) as usize] - 1
        ));
        outfile.write_str("  ");
        let mut i = c.indexptrs[j as usize];
        while i < c.indexptrs[(j + 1) as usize] {
            outfile.write_str(&format!(
                "{}: {}  ",
                c.indexvals[i as usize],
                sun_format_e(c.data[i as usize])
            ));
            i += 1;
        }
        outfile.write_str("\n");
    }
}

pub fn SUNSparseMatrix_Rows(A: &SUNMatrix) -> sunindextype {
    SM_ROWS_S(A)
}

pub fn SUNSparseMatrix_Columns(A: &SUNMatrix) -> sunindextype {
    SM_COLUMNS_S(A)
}

pub fn SUNSparseMatrix_NNZ(A: &SUNMatrix) -> sunindextype {
    SM_NNZ_S(A)
}

pub fn SUNSparseMatrix_NP(A: &SUNMatrix) -> sunindextype {
    SM_NP_S(A)
}

pub fn SUNSparseMatrix_SparseType(A: &SUNMatrix) -> i32 {
    SM_SPARSETYPE_S(A)
}

pub fn SUNSparseMatrix_Data(A: &SUNMatrix) -> RefMut<'_, Vec<sunrealtype>> {
    SM_DATA_S(A)
}

pub fn SUNSparseMatrix_IndexValues(A: &SUNMatrix) -> RefMut<'_, Vec<sunindextype>> {
    SM_INDEXVALS_S(A)
}

pub fn SUNSparseMatrix_IndexPointers(A: &SUNMatrix) -> RefMut<'_, Vec<sunindextype>> {
    SM_INDEXPTRS_S(A)
}

pub fn SUNMatGetID_Sparse(_A: &SUNMatrix) -> SUNMatrix_ID {
    SUNMATRIX_SPARSE
}

pub fn SUNMatClone_Sparse(A: &SUNMatrix) -> Option<SUNMatrix> {
    SUNSparseMatrix(
        SM_ROWS_S(A),
        SM_COLUMNS_S(A),
        SM_NNZ_S(A),
        SM_SPARSETYPE_S(A),
        &A.sunctx.borrow(),
    )
}

pub fn SUNMatDestroy_Sparse(A: SUNMatrix) {
    drop(A);
}

pub fn SUNMatZero_Sparse(A: &SUNMatrix) -> SUNErrCode {
    let mut c = content_mut(A);
    let nnz = c.NNZ;
    for i in 0..nnz as usize {
        c.data[i] = ZERO;
        c.indexvals[i] = 0;
    }
    let np = c.NP;
    for i in 0..np as usize {
        c.indexptrs[i] = 0;
    }
    c.indexptrs[np as usize] = 0;
    SUN_SUCCESS
}

pub fn SUNMatCopy_Sparse(A: &SUNMatrix, B: &SUNMatrix) -> SUNErrCode {
    if !compatibleMatrices(A, B) {
        return SUN_ERR_ARG_DIMSMISMATCH;
    }

    /* Perform operation */
    let A_nz = {
        let a = content_mut(A);
        a.indexptrs[a.NP as usize]
    };

    /* ensure that B is allocated with at least as much memory as we have
    nonzeros in A */
    {
        let mut b = content_mut(B);
        if b.NNZ < A_nz {
            b.indexvals.resize(A_nz as usize, 0);
            b.data.resize(A_nz as usize, 0.0);
            b.NNZ = A_nz;
        }
    }

    /* zero out B so that copy works correctly */
    let ier = SUNMatZero_Sparse(B);
    if ier != SUN_SUCCESS {
        return ier;
    }

    let a = content_mut(A);
    let mut b = content_mut(B);

    /* copy the data and row indices over */
    for i in 0..A_nz as usize {
        b.data[i] = a.data[i];
        b.indexvals[i] = a.indexvals[i];
    }

    /* copy the column pointers over */
    for i in 0..a.NP as usize {
        b.indexptrs[i] = a.indexptrs[i];
    }
    b.indexptrs[a.NP as usize] = A_nz;

    SUN_SUCCESS
}

pub fn SUNMatScaleAddI_Sparse(c: sunrealtype, A: &SUNMatrix) -> SUNErrCode {
    let (N, M) = if SM_SPARSETYPE_S(A) == SUN_CSC_MAT {
        (SM_COLUMNS_S(A), SM_ROWS_S(A))
    } else {
        (SM_ROWS_S(A), SM_COLUMNS_S(A))
    };

    let mut newvals: sunindextype = 0;
    {
        let mut ac = content_mut(A);
        for j in 0..N {
            /* scan column (row if CSR) of A, searching for diagonal value */
            let mut found = SUNFALSE;
            let (p0, p1) = (ac.indexptrs[j as usize], ac.indexptrs[(j + 1) as usize]);
            let mut i = p0;
            while i < p1 {
                if ac.indexvals[i as usize] == j {
                    found = SUNTRUE;
                    ac.data[i as usize] = ONE + c * ac.data[i as usize];
                } else {
                    ac.data[i as usize] *= c;
                }
                i += 1;
            }
            if !found && j < M {
                newvals += 1;
            }
        }
    }

    /* allocate additional storage if needed */
    let new_nnz = {
        let ac = content_mut(A);
        ac.indexptrs[N as usize] + newvals
    };
    if new_nnz > SM_NNZ_S(A) {
        let ier = SUNSparseMatrix_Reallocate(A, new_nnz);
        if ier != SUN_SUCCESS {
            return ier;
        }
    }

    let mut ac = content_mut(A);
    let mut newvals = newvals;
    let mut j = N - 1;
    while newvals > 0 {
        let mut found = SUNFALSE;
        let p0 = ac.indexptrs[j as usize];
        let mut i = ac.indexptrs[(j + 1) as usize] - 1;
        while i >= p0 {
            if ac.indexvals[i as usize] == j {
                found = SUNTRUE;
            }
            /* Shift elements to make room for diagonal elements */
            ac.indexvals[(i + newvals) as usize] = ac.indexvals[i as usize];
            ac.data[(i + newvals) as usize] = ac.data[i as usize];
            i -= 1;
        }

        ac.indexptrs[(j + 1) as usize] += newvals;
        if !found && j < M {
            /* This column (row) needs a diagonal element added */
            newvals -= 1;
            let idx = (ac.indexptrs[j as usize] + newvals) as usize;
            ac.indexvals[idx] = j;
            ac.data[idx] = ONE;
        }
        j -= 1;
    }

    SUN_SUCCESS
}

pub fn SUNMatScaleAdd_Sparse(c: sunrealtype, A: &SUNMatrix, B: &SUNMatrix) -> SUNErrCode {
    if !compatibleMatrices(A, B) {
        return SUN_ERR_ARG_DIMSMISMATCH;
    }

    /* store shortcuts to matrix dimensions (M is inner, N is outer) */
    let (M, N) = if SM_SPARSETYPE_S(A) == SUN_CSC_MAT {
        (SM_ROWS_S(A), SM_COLUMNS_S(A))
    } else {
        (SM_COLUMNS_S(A), SM_ROWS_S(A))
    };

    /* create work arrays for row indices and nonzero column values */
    let mut w: Vec<sunindextype> = vec![0; M as usize];
    let mut x: Vec<sunrealtype> = vec![0.0; M as usize];

    /* determine if A already contains the sparsity pattern of B */
    let mut newvals: sunindextype = 0;
    {
        let a = content_mut(A);
        let b = content_mut(B);
        for j in 0..N as usize {
            for i in 0..M as usize {
                w[i] = 0;
            }
            let mut i = a.indexptrs[j];
            while i < a.indexptrs[j + 1] {
                w[a.indexvals[i as usize] as usize] += 1;
                i += 1;
            }
            let mut i = b.indexptrs[j];
            while i < b.indexptrs[j + 1] {
                w[b.indexvals[i as usize] as usize] -= 1;
                i += 1;
            }
            for i in 0..M as usize {
                if w[i] < 0 {
                    newvals += 1;
                }
            }
        }
    }

    /* If extra nonzeros required, check whether A has sufficient storage */
    let newmat = {
        let a = content_mut(A);
        newvals > (a.NNZ - a.indexptrs[N as usize])
    };

    if newvals == 0 {
        /* case 1: A already contains sparsity pattern of B */
        let mut a = content_mut(A);
        let b = content_mut(B);
        for j in 0..N as usize {
            for i in 0..M as usize {
                x[i] = ZERO;
            }
            let mut i = b.indexptrs[j];
            while i < b.indexptrs[j + 1] {
                x[b.indexvals[i as usize] as usize] = b.data[i as usize];
                i += 1;
            }
            let mut i = a.indexptrs[j];
            while i < a.indexptrs[j + 1] {
                a.data[i as usize] =
                    c * a.data[i as usize] + x[a.indexvals[i as usize] as usize];
                i += 1;
            }
        }
    } else if !newmat {
        /* case 2: A has sufficient storage but not B's sparsity */
        let mut a = content_mut(A);
        let b = content_mut(B);

        /* determine storage location where last column (row) should end */
        let mut nz = a.indexptrs[N as usize] + newvals;

        /* store pointer past last column (row) from original A */
        let mut cend = a.indexptrs[N as usize];
        a.indexptrs[N as usize] = nz;

        /* iterate through columns (rows) backwards */
        let mut j = N - 1;
        loop {
            for i in 0..M as usize {
                w[i] = 0;
                x[i] = 0.0;
            }

            /* iterate down column (row) of A, collecting nonzeros */
            let mut p = a.indexptrs[j as usize];
            while p < cend {
                w[a.indexvals[p as usize] as usize] += 1;
                x[a.indexvals[p as usize] as usize] = c * a.data[p as usize];
                p += 1;
            }

            /* iterate down column of B, collecting nonzeros */
            let mut p = b.indexptrs[j as usize];
            while p < b.indexptrs[(j + 1) as usize] {
                w[b.indexvals[p as usize] as usize] += 1;
                x[b.indexvals[p as usize] as usize] += b.data[p as usize];
                p += 1;
            }

            /* fill entries of A with this column's (row's) data */
            let mut i = M - 1;
            while i >= 0 {
                if w[i as usize] > 0 {
                    nz -= 1;
                    a.indexvals[nz as usize] = i;
                    a.data[nz as usize] = x[i as usize];
                }
                i -= 1;
            }

            /* store ptr past this col (row) from orig A, update value */
            cend = a.indexptrs[j as usize];
            a.indexptrs[j as usize] = nz;

            if j == 0 {
                break;
            }
            j -= 1;
        }
    } else {
        /* case 3: A must be reallocated with sufficient storage */
        let (new_data, new_indexvals, new_indexptrs, new_nnz);
        {
            let a = content_mut(A);
            let b = content_mut(B);

            let nnz_c = a.indexptrs[N as usize] + newvals;
            let mut Cp: Vec<sunindextype> = vec![0; (a.NP + 1) as usize];
            let mut Ci: Vec<sunindextype> = vec![0; nnz_c as usize];
            let mut Cx: Vec<sunrealtype> = vec![0.0; nnz_c as usize];

            /* initialize total nonzero count */
            let mut nz: sunindextype = 0;

            /* iterate through columns (rows) */
            for j in 0..N as usize {
                Cp[j] = nz;

                for i in 0..M as usize {
                    w[i] = 0;
                    x[i] = 0.0;
                }

                let mut p = a.indexptrs[j];
                while p < a.indexptrs[j + 1] {
                    w[a.indexvals[p as usize] as usize] += 1;
                    x[a.indexvals[p as usize] as usize] = c * a.data[p as usize];
                    p += 1;
                }

                let mut p = b.indexptrs[j];
                while p < b.indexptrs[j + 1] {
                    w[b.indexvals[p as usize] as usize] += 1;
                    x[b.indexvals[p as usize] as usize] += b.data[p as usize];
                    p += 1;
                }

                for i in 0..M as usize {
                    if w[i] > 0 {
                        Ci[nz as usize] = i as sunindextype;
                        Cx[nz as usize] = x[i];
                        nz += 1;
                    }
                }
            }

            /* indicate end of data */
            Cp[N as usize] = nz;

            new_data = Cx;
            new_indexvals = Ci;
            new_indexptrs = Cp;
            new_nnz = nnz_c;
        }

        /* update A's structure with C's values */
        let mut a = content_mut(A);
        a.NNZ = new_nnz;
        a.data = new_data;
        a.indexvals = new_indexvals;
        a.indexptrs = new_indexptrs;
    }

    SUN_SUCCESS
}

pub fn SUNMatMatvec_Sparse(A: &SUNMatrix, x: &N_Vector, y: &N_Vector) -> SUNErrCode {
    if !compatibleMatrixAndVectors(A, x, y) {
        return SUN_ERR_ARG_DIMSMISMATCH;
    }

    /* Perform operation */
    if SM_SPARSETYPE_S(A) == SUN_CSC_MAT {
        Matvec_SparseCSC(A, x, y)
    } else {
        Matvec_SparseCSR(A, x, y)
    }
}

pub fn SUNMatHermitianTransposeVec_Sparse(
    A: &SUNMatrix,
    x: &N_Vector,
    y: &N_Vector,
) -> SUNErrCode {
    if !compatibleMatrixAndVectors(A, y, x) {
        return SUN_ERR_ARG_DIMSMISMATCH;
    }

    /* Perform operation */
    if SM_SPARSETYPE_S(A) == SUN_CSC_MAT {
        MatTransposeVec_SparseCSC(A, x, y)
    } else {
        MatTransposeVec_SparseCSR(A, x, y)
    }
}

pub fn SUNMatSpace_Sparse(A: &SUNMatrix, lenrw: &mut i64, leniw: &mut i64) -> SUNErrCode {
    *lenrw = SM_NNZ_S(A);
    *leniw = 10 + SM_NP_S(A) + SM_NNZ_S(A);
    SUN_SUCCESS
}

fn compatibleMatrices(A: &SUNMatrix, B: &SUNMatrix) -> sunbooleantype {
    if SUNSparseMatrix_Rows(A) != SUNSparseMatrix_Rows(B) {
        return SUNFALSE;
    }
    if SUNSparseMatrix_Columns(A) != SUNSparseMatrix_Columns(B) {
        return SUNFALSE;
    }
    if SM_SPARSETYPE_S(A) != SM_SPARSETYPE_S(B) {
        return SUNFALSE;
    }
    SUNTRUE
}

fn compatibleMatrixAndVectors(A: &SUNMatrix, x: &N_Vector, y: &N_Vector) -> sunbooleantype {
    if x.ops.borrow().nvgetarraypointer.is_none() || y.ops.borrow().nvgetarraypointer.is_none() {
        return SUNFALSE;
    }
    if (SUNSparseMatrix_Columns(A) != N_VGetLength(x))
        || (SUNSparseMatrix_Rows(A) != N_VGetLength(y))
    {
        return SUNFALSE;
    }
    SUNTRUE
}

fn Matvec_SparseCSC(A: &SUNMatrix, x: &N_Vector, y: &N_Vector) -> SUNErrCode {
    let a = content_mut(A);
    let xd = N_VGetArrayPointer(x).expect("sparse matvec x data");
    let mut yd = N_VGetArrayPointer(y).expect("sparse matvec y data");

    /* initialize result */
    for i in 0..a.M as usize {
        yd[i] = ZERO;
    }

    /* iterate through matrix columns */
    for j in 0..a.N as usize {
        let mut i = a.indexptrs[j];
        while i < a.indexptrs[j + 1] {
            yd[a.indexvals[i as usize] as usize] += a.data[i as usize] * xd[j];
            i += 1;
        }
    }

    SUN_SUCCESS
}

fn MatTransposeVec_SparseCSC(A: &SUNMatrix, x: &N_Vector, y: &N_Vector) -> SUNErrCode {
    let a = content_mut(A);
    let xd = N_VGetArrayPointer(x).expect("sparse transpose x data");
    let mut yd = N_VGetArrayPointer(y).expect("sparse transpose y data");

    /* initialize result vector */
    for i in 0..a.N as usize {
        yd[i] = ZERO;
    }

    /* iterate through matrix columns (rows of the transposed matrix) */
    for j in 0..a.N as usize {
        let mut i = a.indexptrs[j];
        while i < a.indexptrs[j + 1] {
            yd[j] += a.data[i as usize] * xd[a.indexvals[i as usize] as usize];
            i += 1;
        }
    }

    SUN_SUCCESS
}

fn Matvec_SparseCSR(A: &SUNMatrix, x: &N_Vector, y: &N_Vector) -> SUNErrCode {
    let a = content_mut(A);
    let xd = N_VGetArrayPointer(x).expect("sparse matvec x data");
    let mut yd = N_VGetArrayPointer(y).expect("sparse matvec y data");

    /* initialize result */
    for i in 0..a.M as usize {
        yd[i] = ZERO;
    }

    /* iterate through matrix rows */
    for i in 0..a.M as usize {
        let mut j = a.indexptrs[i];
        while j < a.indexptrs[i + 1] {
            yd[i] += a.data[j as usize] * xd[a.indexvals[j as usize] as usize];
            j += 1;
        }
    }

    SUN_SUCCESS
}

fn MatTransposeVec_SparseCSR(A: &SUNMatrix, x: &N_Vector, y: &N_Vector) -> SUNErrCode {
    let a = content_mut(A);
    let xd = N_VGetArrayPointer(x).expect("sparse transpose x data");
    let mut yd = N_VGetArrayPointer(y).expect("sparse transpose y data");

    /* initialize result vector */
    for i in 0..a.N as usize {
        yd[i] = ZERO;
    }

    /* iterate over rows (columns of the transposed matrix) */
    for i in 0..a.M as usize {
        let mut j = a.indexptrs[i];
        while j < a.indexptrs[i + 1] {
            yd[a.indexvals[j as usize] as usize] += a.data[j as usize] * xd[i];
            j += 1;
        }
    }

    SUN_SUCCESS
}

/// Copies A into a matrix B in the opposite format of A.
fn format_convert(A: &SUNMatrix, B: &SUNMatrix) -> SUNErrCode {
    if SM_SPARSETYPE_S(A) == SM_SPARSETYPE_S(B) {
        return SUNMatCopy_Sparse(A, B);
    }

    let (n_row, n_col) = if SM_SPARSETYPE_S(A) == SUN_CSR_MAT {
        (SM_ROWS_S(A), SM_COLUMNS_S(A))
    } else {
        (SM_COLUMNS_S(A), SM_ROWS_S(A))
    };

    let ier = SUNMatZero_Sparse(B);
    if ier != SUN_SUCCESS {
        return ier;
    }

    let a = content_mut(A);
    let mut b = content_mut(B);

    let nnz = a.indexptrs[n_row as usize];

    /* compute number of non-zero entries per column (if CSR) or per row (if
    CSC) of A */
    for n in 0..nnz as usize {
        b.indexptrs[a.indexvals[n] as usize] += 1;
    }

    /* cumulative sum the nnz per column to get Bp[] */
    let mut csum: sunindextype = 0;
    for col in 0..n_col as usize {
        let temp = b.indexptrs[col];
        b.indexptrs[col] = csum;
        csum += temp;
    }
    b.indexptrs[n_col as usize] = nnz;

    for row in 0..n_row {
        let mut jj = a.indexptrs[row as usize];
        while jj < a.indexptrs[(row + 1) as usize] {
            let col = a.indexvals[jj as usize];
            let dest = b.indexptrs[col as usize];

            b.indexvals[dest as usize] = row;
            b.data[dest as usize] = a.data[jj as usize];

            b.indexptrs[col as usize] += 1;
            jj += 1;
        }
    }

    let mut last: sunindextype = 0;
    for col in 0..=n_col as usize {
        let temp = b.indexptrs[col];
        b.indexptrs[col] = last;
        last = temp;
    }

    SUN_SUCCESS
}
