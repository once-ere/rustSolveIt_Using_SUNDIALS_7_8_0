//! Port of `src/sunlinsol/klu/sunlinsol_klu.c` +
//! `include/sunlinsol/sunlinsol_klu.h`.
//!
//! The SUNDIALS wrapper is translated faithfully — same control flow, same
//! return flags, same reciprocal-condition-number logic. What is *not* a
//! translation is the factorization underneath it: SUNDIALS calls KLU from
//! SuiteSparse, which is LGPL-2.1-or-later and so cannot be translated into
//! this BSD-3-Clause tree, and which this port could not call anyway
//! because FFI is forbidden. Every `klu_*` entry point is therefore served
//! by [`crate::sundials_sparse_lu`], an independent sparse LU written here.
//!
//! | KLU entry point | served by |
//! |---|---|
//! | `klu_defaults` | [`KluCommon::default`] |
//! | `klu_analyze` | recorded pattern only — there is no separate symbolic phase |
//! | `klu_factor`, `klu_refactor` | [`SparseLU::factor`] |
//! | `klu_solve` | [`SparseLU::solve`] |
//! | `klu_tsolve` | [`SparseLU::solve_transpose`] |
//! | `klu_rcond` | [`SparseLU::rcond`] |
//! | `klu_condest` | [`SparseLU::condest`] |
//! | `klu_free_symbolic`, `klu_free_numeric` | dropping the owned value |
//!
//! **This does not reproduce KLU's arithmetic bit for bit.** A different
//! elimination order rounds differently, and inside a Newton iteration that
//! is output-observable, so the eleven `*_klu` examples do not print the
//! same digits as their C originals. `differences/ATTRIBUTION.md` measures
//! by how much, and `sundials_sparse_lu`'s module documentation lists every
//! deliberate algorithmic difference.

use std::cell::RefMut;

use crate::sundials_context::SUNContext;
use crate::sundials_errors::{
    SUN_ERR_ARG_CORRUPT, SUN_ERR_ARG_INCOMPATIBLE, SUN_ERR_EXT_FAIL, SUN_ERR_MEM_FAIL,
    SUN_SUCCESS,
};
use crate::sundials_linearsolver::*;
use crate::sundials_math::SUNRpowerR;
use crate::sundials_matrix::{SUNMatGetID, SUNMatrix, SUNMATRIX_SPARSE};
use crate::sundials_nvector::{N_VGetArrayPointer, N_VScale, N_Vector};
use crate::sundials_sparse_lu::{SparseLU, SparseLuError};
use crate::sundials_types::*;
use crate::sunmatrix_sparse::{
    SUNSparseMatrix_Content, SUNSparseMatrix_NP, SUNSparseMatrix_SparseType, SUN_CSC_MAT,
};

const ONE: sunrealtype = 1.0;
const TWOTHIRDS: sunrealtype = 2.0 / 3.0;

/// C `SUNKLU_REINIT_FULL`: the matrix has a new size or a new number of
/// nonzeros, so both phases must be redone.
pub const SUNKLU_REINIT_FULL: i32 = 1;
/// C `SUNKLU_REINIT_PARTIAL`: only the numerical values changed.
pub const SUNKLU_REINIT_PARTIAL: i32 = 2;

/// C `SUNKLU_ORDERING_DEFAULT`. KLU numbers its orderings 0 = AMD,
/// 1 = COLAMD, 2 = natural. This port has one ordering — natural — so the
/// value is recorded for API compatibility and does not select anything.
pub const SUNKLU_ORDERING_DEFAULT: i32 = 1;

/// The subset of KLU's `klu_common` this wrapper reads or writes.
#[derive(Clone, Copy, Debug)]
pub struct KluCommon {
    /// Cheap reciprocal condition number estimate from the last `rcond`.
    pub rcond: sunrealtype,
    /// One-norm condition estimate from the last `condest`.
    pub condest: sunrealtype,
    /// Recorded for API compatibility; see [`SUNKLU_ORDERING_DEFAULT`].
    pub ordering: i32,
}

impl Default for KluCommon {
    /// C `klu_defaults`.
    fn default() -> Self {
        KluCommon { rcond: ONE, condest: ONE, ordering: SUNKLU_ORDERING_DEFAULT }
    }
}

pub struct SUNLinearSolverContent_KLU_ {
    pub last_flag: sunindextype,
    pub first_factorize: i32,
    /// Present once the pattern has been analyzed. There is no separate
    /// symbolic object to keep, so this records only that the phase ran.
    pub symbolic: bool,
    /// The current factorization, if there is one.
    pub numeric: Option<SparseLU>,
    pub common: KluCommon,
    /// `true` when the matrix is compressed-sparse-column, so a plain
    /// solve applies; `false` for compressed-sparse-row, whose arrays
    /// describe the transpose. C selects `klu_solve` / `klu_tsolve` here.
    pub csc: bool,
}

pub type SUNLinearSolverContent_KLU = SUNLinearSolverContent_KLU_;

fn content_mut(S: &SUNLinearSolver) -> RefMut<'_, SUNLinearSolverContent_KLU_> {
    RefMut::map(S.content.borrow_mut(), |c| {
        c.downcast_mut::<SUNLinearSolverContent_KLU_>()
            .expect("KLU SUNLinearSolver content")
    })
}

/// C `SUNLinSol_KLU`.
pub fn SUNLinSol_KLU(y: &N_Vector, A: &SUNMatrix, sunctx: &SUNContext) -> Option<SUNLinearSolver> {
    /* Check compatibility with the supplied SUNMatrix and N_Vector */
    if SUNMatGetID(A) != SUNMATRIX_SPARSE {
        return None;
    }
    y.ops.borrow().nvgetarraypointer?;

    let csc = SUNSparseMatrix_SparseType(A) == SUN_CSC_MAT;

    let S = SUNLinSolNewEmpty(sunctx)?;
    {
        let mut ops = S.ops.borrow_mut();
        ops.gettype = Some(SUNLinSolGetType_KLU);
        ops.getid = Some(SUNLinSolGetID_KLU);
        ops.initialize = Some(SUNLinSolInitialize_KLU);
        ops.setup = Some(SUNLinSolSetup_KLU);
        ops.solve = Some(SUNLinSolSolve_KLU);
        ops.lastflag = Some(SUNLinSolLastFlag_KLU);
        ops.space = Some(SUNLinSolSpace_KLU);
        ops.free = Some(SUNLinSolFree_KLU);
    }
    *S.content.borrow_mut() = Box::new(SUNLinearSolverContent_KLU_ {
        last_flag: 0,
        first_factorize: 1,
        symbolic: false,
        numeric: None,
        common: KluCommon::default(),
        csc,
    });
    Some(S)
}

pub fn SUNLinSolGetType_KLU(_S: &SUNLinearSolver) -> SUNLinearSolver_Type {
    SUNLinearSolver_Type::SUNLINEARSOLVER_DIRECT
}

pub fn SUNLinSolGetID_KLU(_S: &SUNLinearSolver) -> SUNLinearSolver_ID {
    SUNLinearSolver_ID::SUNLINEARSOLVER_KLU
}

/// C `SUNLinSolInitialize_KLU`: force a full factorization on the next
/// setup.
pub fn SUNLinSolInitialize_KLU(S: &SUNLinearSolver) -> SUNErrCode {
    let mut c = content_mut(S);
    c.first_factorize = 1;
    c.last_flag = SUN_SUCCESS as sunindextype;
    SUN_SUCCESS
}

/// C `SUNLinSol_KLUReInit`.
///
/// `nnz` and `reinit_type` describe how much of the previous work is still
/// valid. This port keeps no symbolic object, so both settings reduce to
/// discarding the factorization; the matrix itself is resized by the
/// caller, exactly as in C.
pub fn SUNLinSol_KLUReInit(
    S: &SUNLinearSolver,
    A: &SUNMatrix,
    _nnz: sunindextype,
    reinit_type: i32,
) -> i32 {
    if SUNMatGetID(A) != SUNMATRIX_SPARSE {
        let mut c = content_mut(S);
        c.last_flag = SUN_ERR_ARG_INCOMPATIBLE as sunindextype;
        return SUN_ERR_ARG_INCOMPATIBLE;
    }
    if reinit_type != SUNKLU_REINIT_FULL && reinit_type != SUNKLU_REINIT_PARTIAL {
        let mut c = content_mut(S);
        c.last_flag = SUN_ERR_ARG_INCOMPATIBLE as sunindextype;
        return SUN_ERR_ARG_INCOMPATIBLE;
    }
    let mut c = content_mut(S);
    c.numeric = None;
    c.symbolic = false;
    c.first_factorize = 1;
    c.last_flag = SUN_SUCCESS as sunindextype;
    SUN_SUCCESS
}

/// C `SUNLinSol_KLUSetOrdering`. Accepted for API compatibility; this port
/// implements one ordering (see [`SUNKLU_ORDERING_DEFAULT`]).
pub fn SUNLinSol_KLUSetOrdering(S: &SUNLinearSolver, ordering_choice: i32) -> i32 {
    if !(0..=2).contains(&ordering_choice) {
        let mut c = content_mut(S);
        c.last_flag = SUN_ERR_ARG_INCOMPATIBLE as sunindextype;
        return SUN_ERR_ARG_INCOMPATIBLE;
    }
    let mut c = content_mut(S);
    c.common.ordering = ordering_choice;
    c.last_flag = SUN_SUCCESS as sunindextype;
    SUN_SUCCESS
}

/// One-norm of the matrix held in compressed columns: the largest column
/// sum of absolute values. `klu_condest` computes this internally from the
/// same arrays.
fn one_norm(np: usize, ap: &[sunindextype], ax: &[sunrealtype]) -> sunrealtype {
    let mut norm = 0.0f64;
    for j in 0..np {
        let mut s = 0.0f64;
        for t in ap[j] as usize..ap[j + 1] as usize {
            s += ax[t].abs();
        }
        if s > norm {
            norm = s;
        }
    }
    norm
}

/// C `SUNLinSolSetup_KLU`.
pub fn SUNLinSolSetup_KLU(S: &SUNLinearSolver, A: Option<&SUNMatrix>) -> i32 {
    let Some(A) = A else {
        let mut c = content_mut(S);
        c.last_flag = SUN_ERR_ARG_CORRUPT as sunindextype;
        return SUN_ERR_ARG_CORRUPT;
    };

    let uround_twothirds = SUNRpowerR(SUN_UNIT_ROUNDOFF, TWOTHIRDS);

    /* Ensure that A is a sparse matrix */
    if SUNMatGetID(A) != SUNMATRIX_SPARSE {
        let mut c = content_mut(S);
        c.last_flag = SUN_ERR_ARG_INCOMPATIBLE as sunindextype;
        return SUN_ERR_ARG_INCOMPATIBLE;
    }

    let np = SUNSparseMatrix_NP(A) as usize;
    /* one borrow for all three arrays: taking them separately would be a
    second RefCell borrow of the same content */
    let m = SUNSparseMatrix_Content(A);
    let (ap, ai, ax) = (&m.indexptrs, &m.indexvals, &m.data);

    let mut c = content_mut(S);

    if c.first_factorize != 0 {
        /* Perform symbolic analysis of sparsity structure, then factor.
        There is no separate symbolic object here; the pattern is read
        again by the factorization itself. */
        c.symbolic = true;
        match SparseLU::factor(np, ap, ai, ax) {
            Ok(lu) => c.numeric = Some(lu),
            Err(_) => {
                c.numeric = None;
                c.last_flag = SUN_ERR_EXT_FAIL as sunindextype;
                return SUN_ERR_EXT_FAIL;
            }
        }
        c.first_factorize = 0;
    } else {
        /* not the first decomposition, so just refactor */
        match SparseLU::factor(np, ap, ai, ax) {
            Ok(lu) => c.numeric = Some(lu),
            Err(SparseLuError::Singular(_)) | Err(SparseLuError::Malformed) => {
                c.last_flag = SUNLS_PACKAGE_FAIL_REC as sunindextype;
                return SUNLS_PACKAGE_FAIL_REC;
            }
        }

        /*-----------------------------------------------------------
        Check if a cheap estimate of the reciprocal of the condition
        number is getting too small.  If so, delete the prior numeric
        factorization and recompute it.
        -----------------------------------------------------------*/
        let rcond = match c.numeric.as_ref() {
            Some(lu) => lu.rcond(),
            None => {
                c.last_flag = SUNLS_PACKAGE_FAIL_REC as sunindextype;
                return SUNLS_PACKAGE_FAIL_REC;
            }
        };
        c.common.rcond = rcond;

        if rcond < uround_twothirds {
            /* Condition number may be getting large.
            Compute more accurate estimate */
            let anorm = one_norm(np, ap, ax);
            let condest = match c.numeric.as_ref() {
                Some(lu) => lu.condest(anorm),
                None => {
                    c.last_flag = SUNLS_PACKAGE_FAIL_REC as sunindextype;
                    return SUNLS_PACKAGE_FAIL_REC;
                }
            };
            c.common.condest = condest;

            if condest > (ONE / uround_twothirds) {
                /* More accurate estimate also says the condition number is
                large, so recompute the numeric factorization. This port
                re-pivots on every factorization, so the recomputation is
                the same call; it is kept so the control flow, and the
                failure flag it can raise, match the C. */
                match SparseLU::factor(np, ap, ai, ax) {
                    Ok(lu) => c.numeric = Some(lu),
                    Err(_) => {
                        c.numeric = None;
                        c.last_flag = SUN_ERR_EXT_FAIL as sunindextype;
                        return SUN_ERR_EXT_FAIL;
                    }
                }
            }
        }
    }

    c.last_flag = SUN_SUCCESS as sunindextype;
    SUN_SUCCESS
}

/// C `SUNLinSolSolve_KLU`.
pub fn SUNLinSolSolve_KLU(
    S: &SUNLinearSolver,
    _A: Option<&SUNMatrix>,
    x: &N_Vector,
    b: &N_Vector,
    _tol: sunrealtype,
) -> i32 {
    /* copy b into x */
    N_VScale(ONE, b, x);

    /* access x data array */
    let Some(mut xdata) = N_VGetArrayPointer(x) else {
        let mut c = content_mut(S);
        c.last_flag = SUN_ERR_MEM_FAIL as sunindextype;
        return SUN_ERR_MEM_FAIL;
    };

    let mut c = content_mut(S);
    let csc = c.csc;
    let Some(lu) = c.numeric.as_ref() else {
        c.last_flag = SUNLS_PACKAGE_FAIL_REC as sunindextype;
        return SUNLS_PACKAGE_FAIL_REC;
    };

    /* Call the solver on the linear system. For a compressed-sparse-row
    matrix the stored arrays describe the transpose, which is why C
    selects klu_tsolve there. */
    if csc {
        lu.solve(&mut xdata);
    } else {
        lu.solve_transpose(&mut xdata);
    }

    c.last_flag = SUN_SUCCESS as sunindextype;
    SUN_SUCCESS
}

pub fn SUNLinSolLastFlag_KLU(S: &SUNLinearSolver) -> sunindextype {
    content_mut(S).last_flag
}

/// C `SUNLinSolSpace_KLU`: the factorization is an owned object rather than
/// a caller-supplied workspace, so — as in C, where the KLU structures are
/// opaque — only the wrapper's own two integers are reported.
pub fn SUNLinSolSpace_KLU(
    _S: &SUNLinearSolver,
    lenrwLS: &mut i64,
    leniwLS: &mut i64,
) -> SUNErrCode {
    *leniwLS = 2;
    *lenrwLS = 0;
    SUN_SUCCESS
}

/// C `SUNLinSolFree_KLU`. The factorization is owned, so dropping the
/// content releases it; there is nothing to free explicitly.
pub fn SUNLinSolFree_KLU(_S: &SUNLinearSolver) -> SUNErrCode {
    SUN_SUCCESS
}
