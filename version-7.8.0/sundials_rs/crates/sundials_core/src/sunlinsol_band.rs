//! Port of `src/sunlinsol/band/sunlinsol_band.c` +
//! `include/sunlinsol/sunlinsol_band.h`.

use std::cell::RefMut;

use crate::sundials_band::{SUNDlsMat_bandGBTRF, SUNDlsMat_bandGBTRS};
use crate::sundials_context::SUNContext;
use crate::sundials_errors::SUN_SUCCESS;
use crate::sundials_linearsolver::*;
use crate::sundials_math::SUNMIN;
use crate::sundials_matrix::{SUNMatGetID, SUNMatrix, SUNMATRIX_BAND};
use crate::sundials_nvector::{N_VGetArrayPointer, N_VGetLength, N_VScale, N_Vector};
use crate::sundials_types::*;
use crate::sunmatrix_band::{
    SM_COLUMNS_B, SM_DATA_B, SM_LBAND_B, SM_LDIM_B, SM_ROWS_B, SM_SUBAND_B, SM_UBAND_B,
};

const ONE: sunrealtype = 1.0;

pub struct SUNLinearSolverContent_Band_ {
    pub N: sunindextype,
    pub pivots: Vec<sunindextype>,
    pub last_flag: sunindextype,
}

pub type SUNLinearSolverContent_Band = SUNLinearSolverContent_Band_;

fn content_mut(S: &SUNLinearSolver) -> RefMut<'_, SUNLinearSolverContent_Band_> {
    RefMut::map(S.content.borrow_mut(), |c| {
        c.downcast_mut::<SUNLinearSolverContent_Band_>()
            .expect("band SUNLinearSolver content")
    })
}

pub fn SUNLinSol_Band(y: &N_Vector, A: &SUNMatrix, sunctx: &SUNContext) -> Option<SUNLinearSolver> {
    if SUNMatGetID(A) != SUNMATRIX_BAND {
        return None;
    }
    if SM_ROWS_B(A) != SM_COLUMNS_B(A) {
        return None;
    }
    y.ops.borrow().nvgetarraypointer?;

    /* Check that A has appropriate storage upper bandwidth for factorization */
    let MatrixRows = SM_ROWS_B(A);
    if SM_SUBAND_B(A) < SUNMIN(MatrixRows - 1, SM_LBAND_B(A) + SM_UBAND_B(A)) {
        return None;
    }
    if MatrixRows != N_VGetLength(y) {
        return None;
    }

    /* Create an empty linear solver */
    let S = SUNLinSolNewEmpty(sunctx)?;

    /* Attach operations */
    {
        let mut ops = S.ops.borrow_mut();
        ops.gettype = Some(SUNLinSolGetType_Band);
        ops.getid = Some(SUNLinSolGetID_Band);
        ops.initialize = Some(SUNLinSolInitialize_Band);
        ops.setup = Some(SUNLinSolSetup_Band);
        ops.solve = Some(SUNLinSolSolve_Band);
        ops.lastflag = Some(SUNLinSolLastFlag_Band);
        ops.space = Some(SUNLinSolSpace_Band);
        ops.free = Some(SUNLinSolFree_Band);
    }

    /* Create, attach, and fill content */
    *S.content.borrow_mut() = Box::new(SUNLinearSolverContent_Band_ {
        N: MatrixRows,
        last_flag: 0,
        pivots: vec![0; MatrixRows as usize],
    });

    Some(S)
}

pub fn SUNLinSolGetType_Band(_S: &SUNLinearSolver) -> SUNLinearSolver_Type {
    SUNLINEARSOLVER_DIRECT
}

pub fn SUNLinSolGetID_Band(_S: &SUNLinearSolver) -> SUNLinearSolver_ID {
    SUNLINEARSOLVER_BAND
}

pub fn SUNLinSolInitialize_Band(S: &SUNLinearSolver) -> SUNErrCode {
    /* all solver-specific memory has already been allocated */
    content_mut(S).last_flag = SUN_SUCCESS as sunindextype;
    SUN_SUCCESS
}

pub fn SUNLinSolSetup_Band(S: &SUNLinearSolver, A: Option<&SUNMatrix>) -> i32 {
    let A = A.expect("band setup requires a matrix");

    /* ensure that storage upper bandwidth is sufficient for fill-in */
    if SM_SUBAND_B(A) < SUNMIN(SM_COLUMNS_B(A) - 1, SM_UBAND_B(A) + SM_LBAND_B(A)) {
        return crate::sundials_errors::SUN_ERR_ARG_INCOMPATIBLE;
    }

    let (n, mu, ml, smu, ldim) = (
        SM_COLUMNS_B(A),
        SM_UBAND_B(A),
        SM_LBAND_B(A),
        SM_SUBAND_B(A),
        SM_LDIM_B(A),
    );
    let last_flag;
    {
        let mut content = content_mut(S);
        let mut data = SM_DATA_B(A);
        let mut A_cols: Vec<&mut [sunrealtype]> = data.chunks_mut(ldim as usize).collect();

        /* perform LU factorization of input matrix */
        last_flag = SUNDlsMat_bandGBTRF(&mut A_cols, n, mu, ml, smu, &mut content.pivots);
        content.last_flag = last_flag;
    }

    /* (if nonzero, that row encountered zero-valued pivot) */
    if last_flag > 0 {
        return SUNLS_LUFACT_FAIL;
    }
    SUN_SUCCESS
}

pub fn SUNLinSolSolve_Band(
    S: &SUNLinearSolver,
    A: Option<&SUNMatrix>,
    x: &N_Vector,
    b: &N_Vector,
    _tol: sunrealtype,
) -> i32 {
    let A = A.expect("band solve requires a matrix");

    /* copy b into x */
    N_VScale(ONE, b, x);

    let (n, smu, ml, ldim) = (SM_COLUMNS_B(A), SM_SUBAND_B(A), SM_LBAND_B(A), SM_LDIM_B(A));
    {
        let mut content = content_mut(S);
        let mut data = SM_DATA_B(A);
        let mut A_cols: Vec<&mut [sunrealtype]> = data.chunks_mut(ldim as usize).collect();
        let mut xdata = N_VGetArrayPointer(x).expect("band solve x data");

        /* solve using LU factors */
        SUNDlsMat_bandGBTRS(&mut A_cols, n, smu, ml, &content.pivots, &mut xdata);
        content.last_flag = SUN_SUCCESS as sunindextype;
    }
    SUN_SUCCESS
}

pub fn SUNLinSolLastFlag_Band(S: &SUNLinearSolver) -> sunindextype {
    content_mut(S).last_flag
}

pub fn SUNLinSolSpace_Band(S: &SUNLinearSolver, lenrwLS: &mut i64, leniwLS: &mut i64) -> SUNErrCode {
    *leniwLS = 2 + content_mut(S).N;
    *lenrwLS = 0;
    SUN_SUCCESS
}

pub fn SUNLinSolFree_Band(_S: &SUNLinearSolver) -> SUNErrCode {
    SUN_SUCCESS
}
