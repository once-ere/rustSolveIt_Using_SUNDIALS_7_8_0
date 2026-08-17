//! Port of `src/sunlinsol/dense/sunlinsol_dense.c` +
//! `include/sunlinsol/sunlinsol_dense.h`.

use std::cell::RefMut;

use crate::sundials_context::SUNContext;
use crate::sundials_dense::{SUNDlsMat_denseGETRF, SUNDlsMat_denseGETRS};
use crate::sundials_errors::SUN_SUCCESS;
use crate::sundials_linearsolver::*;
use crate::sundials_matrix::{SUNMatGetID, SUNMatrix, SUNMATRIX_DENSE};
use crate::sundials_nvector::{N_VGetArrayPointer, N_VGetLength, N_VScale, N_Vector};
use crate::sundials_types::*;
use crate::sunmatrix_dense::{SM_COLUMNS_D, SM_DATA_D, SM_ROWS_D};

const ONE: sunrealtype = 1.0;

pub struct SUNLinearSolverContent_Dense_ {
    pub N: sunindextype,
    pub pivots: Vec<sunindextype>,
    pub last_flag: sunindextype,
}

pub type SUNLinearSolverContent_Dense = SUNLinearSolverContent_Dense_;

fn content_mut(S: &SUNLinearSolver) -> RefMut<'_, SUNLinearSolverContent_Dense_> {
    RefMut::map(S.content.borrow_mut(), |c| {
        c.downcast_mut::<SUNLinearSolverContent_Dense_>()
            .expect("dense SUNLinearSolver content")
    })
}

pub fn SUNLinSol_Dense(y: &N_Vector, A: &SUNMatrix, sunctx: &SUNContext) -> Option<SUNLinearSolver> {
    if SUNMatGetID(A) != SUNMATRIX_DENSE {
        return None;
    }
    if SM_ROWS_D(A) != SM_COLUMNS_D(A) {
        return None;
    }
    y.ops.borrow().nvgetarraypointer?;

    let MatrixRows = SM_ROWS_D(A);
    if MatrixRows != N_VGetLength(y) {
        return None;
    }

    /* Create an empty linear solver */
    let S = SUNLinSolNewEmpty(sunctx)?;

    /* Attach operations */
    {
        let mut ops = S.ops.borrow_mut();
        ops.gettype = Some(SUNLinSolGetType_Dense);
        ops.getid = Some(SUNLinSolGetID_Dense);
        ops.initialize = Some(SUNLinSolInitialize_Dense);
        ops.setup = Some(SUNLinSolSetup_Dense);
        ops.solve = Some(SUNLinSolSolve_Dense);
        ops.lastflag = Some(SUNLinSolLastFlag_Dense);
        ops.space = Some(SUNLinSolSpace_Dense);
        ops.free = Some(SUNLinSolFree_Dense);
    }

    /* Create, attach, and fill content */
    *S.content.borrow_mut() = Box::new(SUNLinearSolverContent_Dense_ {
        N: MatrixRows,
        last_flag: 0,
        pivots: vec![0; MatrixRows as usize],
    });

    Some(S)
}

pub fn SUNLinSolGetType_Dense(_S: &SUNLinearSolver) -> SUNLinearSolver_Type {
    SUNLINEARSOLVER_DIRECT
}

pub fn SUNLinSolGetID_Dense(_S: &SUNLinearSolver) -> SUNLinearSolver_ID {
    SUNLINEARSOLVER_DENSE
}

pub fn SUNLinSolInitialize_Dense(S: &SUNLinearSolver) -> SUNErrCode {
    /* all solver-specific memory has already been allocated */
    content_mut(S).last_flag = SUN_SUCCESS as sunindextype;
    SUN_SUCCESS
}

pub fn SUNLinSolSetup_Dense(S: &SUNLinearSolver, A: Option<&SUNMatrix>) -> i32 {
    let A = A.expect("dense setup requires a matrix");

    let (m, n) = (SM_ROWS_D(A), SM_COLUMNS_D(A));
    let last_flag;
    {
        let mut content = content_mut(S);
        let mut data = SM_DATA_D(A);
        let mut A_cols: Vec<&mut [sunrealtype]> = data.chunks_mut(m as usize).collect();

        /* perform LU factorization of input matrix */
        last_flag = SUNDlsMat_denseGETRF(&mut A_cols, m, n, &mut content.pivots);
        content.last_flag = last_flag;
    }

    /* (if nonzero, this row encountered zero-valued pivot) */
    if last_flag > 0 {
        return SUNLS_LUFACT_FAIL;
    }
    SUN_SUCCESS
}

pub fn SUNLinSolSolve_Dense(
    S: &SUNLinearSolver,
    A: Option<&SUNMatrix>,
    x: &N_Vector,
    b: &N_Vector,
    _tol: sunrealtype,
) -> i32 {
    let A = A.expect("dense solve requires a matrix");

    /* copy b into x */
    N_VScale(ONE, b, x);

    let m = SM_ROWS_D(A);
    {
        let mut content = content_mut(S);
        let mut data = SM_DATA_D(A);
        let mut A_cols: Vec<&mut [sunrealtype]> = data.chunks_mut(m as usize).collect();
        let mut xdata = N_VGetArrayPointer(x).expect("dense solve x data");

        /* solve using LU factors */
        SUNDlsMat_denseGETRS(&mut A_cols, m, &content.pivots, &mut xdata);
        content.last_flag = SUN_SUCCESS as sunindextype;
    }
    SUN_SUCCESS
}

pub fn SUNLinSolLastFlag_Dense(S: &SUNLinearSolver) -> sunindextype {
    content_mut(S).last_flag
}

pub fn SUNLinSolSpace_Dense(S: &SUNLinearSolver, lenrwLS: &mut i64, leniwLS: &mut i64) -> SUNErrCode {
    *leniwLS = 2 + content_mut(S).N;
    *lenrwLS = 0;
    SUN_SUCCESS
}

pub fn SUNLinSolFree_Dense(_S: &SUNLinearSolver) -> SUNErrCode {
    SUN_SUCCESS
}
