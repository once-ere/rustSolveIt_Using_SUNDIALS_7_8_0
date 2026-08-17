//! Port of `src/sundomeigest/arnoldi/sundomeigest_arnoldi.c` +
//! `include/sundomeigest/sundomeigest_arnoldi.h` (Arnoldi-iteration
//! dominant-eigenvalue estimator: warmup power iterations, a `kry_dim`-step
//! Arnoldi factorization building the Hessenberg matrix `Hes`, then the
//! eigenvalues of the leading `kry_dim x kry_dim` block sorted by magnitude).
//!
//! Deviations from the C file, all deliberate:
//!
//! * **`xgeev_f77` (LAPACK `dgeev`) becomes a native routine.** LAPACK is
//!   excluded from this port (the upstream CMake only builds this module when
//!   `SUNDIALS_ENABLE_LAPACK` is on), so the `jobvl = jobvr = 'N'`
//!   eigenvalue-only path of `dgeev` is provided by [`xgeev_f77`] below: an
//!   EISPACK-`hqr`-style Francis double-shift QR iteration run directly on
//!   the (already upper Hessenberg) input. `dgeev`'s balancing (`dgebal`) and
//!   Hessenberg reduction (`dgehrd`) stages are omitted — the second is a
//!   mathematical no-op for upper Hessenberg input (`dlarfg` returns
//!   `tau = 0`), and the first is a diagonal similarity that changes only
//!   rounding. The computed eigenvalues therefore agree with LAPACK's to
//!   within last-digit rounding, which is the same `LAPACK -> native`
//!   verification exception already carried by `cvRoberts_dnsL`.
//! * **`Hes` and the LAPACK arrays are zero-initialized.** C `malloc`s them;
//!   `SUNModifiedGS` only ever writes rows `0..=j+1` of column `j`, so the
//!   strictly-below-subdiagonal entries copied into `LAPACK_A` by the packing
//!   loop are indeterminate in C (reading them is UB). Zero is the value the
//!   algorithm intends (the matrix is upper Hessenberg by construction), so
//!   the port seeds them with zero — accepted deviation class 5.
//! * **`ATdata` is the estimator itself.** C's `SetRhs` registers
//!   `dee_DQJtimes_Arnoldi` with `A_data = (void*)DEE`; the port stores an
//!   `Rc` clone of the handle in the `Option<Box<dyn Any>>` data token. That
//!   makes the content reference-cycle-own the estimator, so
//!   `SUNDomEigEstimator_Destroy_Arnoldi` clears the content box (C's
//!   `free(DEE->content)`) to break the cycle before releasing the handle.
//! * **`SUNModifiedGS`'s aliased out-param.** C passes `&Hes[i+1][i]` as
//!   `new_vk_norm`, i.e. a pointer into the same `h` array. The port writes a
//!   local and stores it into `Hes[i+1][i]` right after the call: identical,
//!   because `SUNModifiedGS` writes `*new_vk_norm` only after its last read
//!   of `h[.][k-1]` and never touches row `k` itself.
//! * `SUNAssert`/`SUNCheckCall`/`SUNCheckLastErr` are release no-ops per the
//!   build config; NULL-deref UB maps to a panic at the same site, and the
//!   vector-operation presence checks in the constructor are kept as plain
//!   `if`s returning `None` (accepted deviation class 1).

use std::any::Any;
use std::cell::RefMut;

use crate::sundials_context::SUNContext;
use crate::sundials_domeigestimator::{SUNDomEigEstimator, SUNDomEigEstimator_NewEmpty, SUNRhsFn};
use crate::sundials_errors::{SUN_ERR_EXT_FAIL, SUN_ERR_USER_FCN_FAIL, SUN_SUCCESS};
use crate::sundials_iterative::SUNModifiedGS;
use crate::sundials_linearsolver::SUNATimesFn;
use crate::sundials_math::{SUNRabs, SUNRsqrt, SUNMAX, SUNMIN};
use crate::sundials_nvector::{
    N_VClone, N_VCloneVectorArray, N_VDestroy, N_VDestroyVectorArray, N_VDotProd, N_VL1Norm,
    N_VLinearSum, N_VScale, N_Vector,
};
use crate::sundials_types::*;
use crate::sundials_utils::SUNFile;

const MAX_DQITERS: i32 = 3;

const ZERO: sunrealtype = 0.0;
const ONE: sunrealtype = 1.0;

/* Default estimator parameters */
const DEE_NUM_OF_WARMUPS_ARNOLDI_DEFAULT: i32 = 100;
const DEE_TOL_OF_WARMUPS_ARNOLDI_DEFAULT: sunrealtype = 0.005;

/* Default Arnoldi Iteration parameters */
const DEE_KRYLOV_DIM_DEFAULT: i32 = 3;

/* -----------------------------------------------------
 * Arnoldi Iteration Implementation of SUNDomEigEstimator
 * ----------------------------------------------------- */

/// C `struct SUNDomEigEstimatorContent_Arnoldi_`.
///
/// C NULL pointers map to `None` for the object handles and callbacks and to
/// an empty `Vec` for the `malloc`'d scratch arrays. `Hes` and `LAPACK_arr`
/// are the row-wise `sunrealtype**` arrays.
pub struct SUNDomEigEstimatorContent_Arnoldi_ {
    pub ATimes: Option<SUNATimesFn>,  /* User provided ATimes function */
    pub ATdata: Option<Box<dyn Any>>, /* ATimes function data*/

    /* Krylov subspace vectors */
    pub V: Option<Vec<N_Vector>>,
    pub q: Option<N_Vector>,
    pub rhs_linY: Option<N_Vector>,
    pub Fy: Option<N_Vector>,
    pub work: Option<N_Vector>,

    pub kry_dim: i32,                  /* Krylov subspace dimension */
    pub num_warmups: i32,              /* Number of preprocessing iterations */
    pub num_iters: i64,                /* Number of iterations in last Estimate call */
    pub warmup_to_tol: sunbooleantype, /* Whether to use warmup iterations */
    pub tol_warmup: sunrealtype,       /* Tolerance for warmup iterations */
    pub rhs_linT: sunrealtype,         /* Time value for linearization point */

    pub num_ATimes: i64, /* Number of ATimes calls */

    pub rhsfn: Option<SUNRhsFn>,        /* User provided RHS function */
    pub rhs_data: Option<Box<dyn Any>>, /* RHS function data */
    pub nfevals: i64,                   /* Number of RHS evaluations */

    /* The vector which holds rows of the Hessenberg matrix in the given order */
    pub LAPACK_A: Vec<sunrealtype>,
    pub LAPACK_wr: Vec<sunrealtype>,   /* Real parts of eigenvalues */
    pub LAPACK_wi: Vec<sunrealtype>,   /* Imaginary parts of eigenvalues */
    pub LAPACK_work: Vec<sunrealtype>, /* Workspace array */
    pub LAPACK_lwork: sunindextype,    /* Dimension of the workspace array */
    pub LAPACK_arr: Vec<Vec<sunrealtype>>, /* an array to sort eigenvalues*/

    pub Hes: Vec<Vec<sunrealtype>>, /* Hessenberg matrix Hes */
}

pub type SUNDomEigEstimatorContent_Arnoldi = SUNDomEigEstimatorContent_Arnoldi_;

/// C `Arnoldi_CONTENT(DEE)`.
fn content_mut(DEE: &SUNDomEigEstimator) -> RefMut<'_, SUNDomEigEstimatorContent_Arnoldi_> {
    RefMut::map(DEE.content.borrow_mut(), |c| {
        c.downcast_mut::<SUNDomEigEstimatorContent_Arnoldi_>()
            .expect("Arnoldi SUNDomEigEstimator content")
    })
}

/*
 * -----------------------------------------------------------------
 * exported functions
 * -----------------------------------------------------------------
 */

/* ----------------------------------------------------------------------------
 * Function to create a new Arnoldi estimator
 */

pub fn SUNDomEigEstimator_Arnoldi(
    q: &N_Vector,
    kry_dim: i32,
    sunctx: &SUNContext,
) -> Option<SUNDomEigEstimator> {
    /* Check if kry_dim >= 2 */
    let kry_dim = if kry_dim < 3 {
        DEE_KRYLOV_DIM_DEFAULT
    } else {
        kry_dim
    };

    /* Check for required vector operations */
    {
        let ops = q.ops.borrow();
        if ops.nvclone.is_none()
            || ops.nvdestroy.is_none()
            || ops.nvdotprod.is_none()
            || ops.nvscale.is_none()
        {
            return None;
        }
    }

    /* Create dominant eigenvalue estimator */
    let DEE = SUNDomEigEstimator_NewEmpty(sunctx)?;

    /* Attach operations */
    {
        let mut ops = DEE.ops.borrow_mut();
        ops.setatimes = Some(SUNDomEigEstimator_SetATimes_Arnoldi);
        ops.setrhs = Some(SUNDomEigEstimator_SetRhs_Arnoldi);
        ops.setrhslinearizationpoint = Some(SUNDomEigEstimator_SetRhsLinearizationPoint_Arnoldi);
        ops.setnumpreprocessiters = Some(SUNDomEigEstimator_SetNumPreprocessIters_Arnoldi);
        ops.setreltol = Some(SUNDomEigEstimator_SetRelTol_Arnoldi);
        ops.setinitialguess = Some(SUNDomEigEstimator_SetInitialGuess_Arnoldi);
        ops.initialize = Some(SUNDomEigEstimator_Initialize_Arnoldi);
        ops.estimate = Some(SUNDomEigEstimator_Estimate_Arnoldi);
        ops.getnumiters = Some(SUNDomEigEstimator_GetNumIters_Arnoldi);
        ops.getnumrhsevals = Some(SUNDomEigEstimator_GetNumRhsEvals_Arnoldi);
        ops.getnumatimescalls = Some(SUNDomEigEstimator_GetNumATimesCalls_Arnoldi);
        ops.write = Some(SUNDomEigEstimator_Write_Arnoldi);
        ops.destroy = Some(SUNDomEigEstimator_Destroy_Arnoldi);
    }

    /* Create content, attach content, fill content */
    *DEE.content.borrow_mut() = Box::new(SUNDomEigEstimatorContent_Arnoldi_ {
        ATimes: None,
        ATdata: None,
        V: None,
        q: None,
        rhs_linY: None,
        rhs_linT: ZERO,
        Fy: None,
        work: None,
        kry_dim,
        num_warmups: DEE_NUM_OF_WARMUPS_ARNOLDI_DEFAULT,
        num_iters: 0,
        num_ATimes: 0,
        warmup_to_tol: SUNFALSE,
        tol_warmup: DEE_TOL_OF_WARMUPS_ARNOLDI_DEFAULT,
        rhsfn: None,
        rhs_data: None,
        nfevals: 0,
        LAPACK_A: Vec::new(),
        LAPACK_wr: Vec::new(),
        LAPACK_wi: Vec::new(),
        LAPACK_work: Vec::new(),
        LAPACK_lwork: 0,
        LAPACK_arr: Vec::new(),
        Hes: Vec::new(),
    });

    /* Allocate content */
    let q_clone = N_VClone(q)?;
    content_mut(&DEE).q = Some(q_clone);

    let V = N_VCloneVectorArray(kry_dim + 1, q)?;
    content_mut(&DEE).V = Some(V);

    /* Initialize the vector V[0] */
    let mut normq = N_VDotProd(q, q);

    normq = SUNRsqrt(normq);

    let v0 = content_mut(&DEE).V.as_ref().expect("V")[0].clone();
    N_VScale(ONE / normq, q, &v0);

    Some(DEE)
}

/*
 * -----------------------------------------------------------------
 * implementation of dominant eigenvalue estimator operations
 * -----------------------------------------------------------------
 */

pub fn SUNDomEigEstimator_SetATimes_Arnoldi(
    DEE: &SUNDomEigEstimator,
    A_data: Option<Box<dyn Any>>,
    ATimes: Option<SUNATimesFn>,
) -> SUNErrCode {
    /* set function pointers to integrator-supplied ATimes routine
    and data, and return with success */
    let mut content = content_mut(DEE);
    content.ATimes = ATimes;
    content.ATdata = A_data;
    SUN_SUCCESS
}

pub fn SUNDomEigEstimator_SetRhs_Arnoldi(
    DEE: &SUNDomEigEstimator,
    rhs_data: Option<Box<dyn Any>>,
    RHSfn: Option<SUNRhsFn>,
) -> SUNErrCode {
    /* set function pointers to integrator-supplied RHS routine
    and data, and return with success */
    {
        let mut content = content_mut(DEE);
        content.rhsfn = RHSfn;
        content.rhs_data = rhs_data;
    }

    /* C: DEE->ops->setatimes(DEE, (void*)DEE, dee_DQJtimes_Arnoldi) -- the
    `void*` estimator self-pointer becomes an `Rc` clone of the handle. */
    let setatimes = DEE.ops.borrow().setatimes.expect("setatimes");
    let _ = setatimes(DEE, Some(Box::new(DEE.clone())), Some(dee_DQJtimes_Arnoldi));

    SUN_SUCCESS
}

pub fn SUNDomEigEstimator_SetRhsLinearizationPoint_Arnoldi(
    DEE: &SUNDomEigEstimator,
    t: sunrealtype,
    y: &N_Vector,
) -> SUNErrCode {
    if content_mut(DEE).rhs_linY.is_none() {
        let rhs_linY = N_VClone(y);
        content_mut(DEE).rhs_linY = rhs_linY;
    }

    content_mut(DEE).rhs_linT = t;

    let rhs_linY = content_mut(DEE)
        .rhs_linY
        .as_ref()
        .expect("rhs_linY")
        .clone();
    N_VScale(ONE, y, &rhs_linY);

    SUN_SUCCESS
}

pub fn SUNDomEigEstimator_Initialize_Arnoldi(DEE: &SUNDomEigEstimator) -> SUNErrCode {
    {
        let mut content = content_mut(DEE);
        if content.kry_dim < 2 {
            content.kry_dim = DEE_KRYLOV_DIM_DEFAULT;
        }
        if content.num_warmups < 0 {
            content.num_warmups = DEE_NUM_OF_WARMUPS_ARNOLDI_DEFAULT;
        }
    }

    /* C asserts ATimes/V/q here; the assertions are release no-ops */

    {
        let mut guard = content_mut(DEE);
        let content = &mut *guard;
        let kry_dim = content.kry_dim as usize;
        if content.LAPACK_A.is_empty() {
            content.LAPACK_A = vec![ZERO; kry_dim * kry_dim];
        }
        if content.LAPACK_wr.is_empty() {
            content.LAPACK_wr = vec![ZERO; kry_dim];
        }
        if content.LAPACK_wi.is_empty() {
            content.LAPACK_wi = vec![ZERO; kry_dim];
        }
    }

    /* query the workspace size (call with lwork = -1) */
    let jobvl: u8 = b'N';
    let jobvr: u8 = b'N';
    let N: sunindextype = content_mut(DEE).kry_dim as sunindextype;
    let lda: sunindextype = content_mut(DEE).kry_dim as sunindextype;
    let mut info: sunindextype = 0;
    let lwork: sunindextype = -1;
    let mut work: sunrealtype = ZERO;

    let (mut LAPACK_A, mut LAPACK_wr, mut LAPACK_wi) = {
        let mut guard = content_mut(DEE);
        let content = &mut *guard;
        (
            std::mem::take(&mut content.LAPACK_A),
            std::mem::take(&mut content.LAPACK_wr),
            std::mem::take(&mut content.LAPACK_wi),
        )
    };
    xgeev_f77(
        jobvl,
        jobvr,
        N,
        &mut LAPACK_A,
        lda,
        &mut LAPACK_wr,
        &mut LAPACK_wi,
        std::slice::from_mut(&mut work),
        lwork,
        &mut info,
    );

    let mut guard = content_mut(DEE);
    let content = &mut *guard;
    content.LAPACK_A = LAPACK_A;
    content.LAPACK_wr = LAPACK_wr;
    content.LAPACK_wi = LAPACK_wi;

    /* The workspace size is returned as the first entry of the work array */
    content.LAPACK_lwork = work as sunindextype;

    let lwork_size = content.LAPACK_lwork as usize;
    content.LAPACK_work = vec![ZERO; lwork_size];

    /* LAPACK array */
    let kry_dim = content.kry_dim as usize;
    content.LAPACK_arr = Vec::with_capacity(kry_dim);
    for _k in 0..kry_dim {
        content.LAPACK_arr.push(vec![ZERO; 2]);
    }

    /* Hessenberg matrix Hes */
    content.Hes = Vec::with_capacity(kry_dim + 1);
    for _k in 0..=kry_dim {
        content.Hes.push(vec![ZERO; kry_dim]);
    }

    SUN_SUCCESS
}

pub fn SUNDomEigEstimator_SetNumPreprocessIters_Arnoldi(
    DEE: &SUNDomEigEstimator,
    num_iters: i32,
) -> SUNErrCode {
    /* Check if num_iters >= 0 */
    let num_iters = if num_iters < 0 {
        DEE_NUM_OF_WARMUPS_ARNOLDI_DEFAULT
    } else {
        num_iters
    };

    /* set the number of warmups */
    content_mut(DEE).num_warmups = num_iters;

    SUN_SUCCESS
}

pub fn SUNDomEigEstimator_SetRelTol_Arnoldi(
    DEE: &SUNDomEigEstimator,
    tol: sunrealtype,
) -> SUNErrCode {
    /* set the tolerance for preprocessing iterations */
    if tol < ZERO {
        content_mut(DEE).warmup_to_tol = SUNFALSE;
        return SUN_SUCCESS;
    }
    let tol = if tol == ZERO || tol > ONE - SUN_UNIT_ROUNDOFF {
        DEE_TOL_OF_WARMUPS_ARNOLDI_DEFAULT
    } else {
        tol
    };
    content_mut(DEE).tol_warmup = tol;

    /* set the type of warmup iterations */
    content_mut(DEE).warmup_to_tol = SUNTRUE;

    SUN_SUCCESS
}

pub fn SUNDomEigEstimator_SetInitialGuess_Arnoldi(
    DEE: &SUNDomEigEstimator,
    q: &N_Vector,
) -> SUNErrCode {
    let mut normq = N_VDotProd(q, q);

    normq = SUNRsqrt(normq);

    /* set the initial guess */
    let v0 = content_mut(DEE).V.as_ref().expect("V")[0].clone();
    N_VScale(ONE / normq, q, &v0);

    SUN_SUCCESS
}

pub fn SUNDomEigEstimator_Estimate_Arnoldi(
    DEE: &SUNDomEigEstimator,
    lambdaR: &mut sunrealtype,
    lambdaI: &mut sunrealtype,
) -> SUNErrCode {
    /* Move estimator state into locals so that no content borrow is held
    across an ATimes call (which re-enters this estimator whenever the
    internal difference-quotient ATimes is in use); everything is written
    back at the single exit point below. */
    let n: sunindextype;
    let V: Vec<N_Vector>;
    let q: N_Vector;
    let LAPACK_lwork: sunindextype;
    let mut Hes: Vec<Vec<sunrealtype>>;
    let mut LAPACK_A: Vec<sunrealtype>;
    let mut LAPACK_wr: Vec<sunrealtype>;
    let mut LAPACK_wi: Vec<sunrealtype>;
    let mut LAPACK_work: Vec<sunrealtype>;
    let mut LAPACK_arr: Vec<Vec<sunrealtype>>;
    {
        let mut guard = content_mut(DEE);
        let content = &mut *guard;
        n = content.kry_dim as sunindextype;
        V = content.V.as_ref().expect("V").clone();
        q = content.q.as_ref().expect("q").clone();
        LAPACK_lwork = content.LAPACK_lwork;
        Hes = std::mem::take(&mut content.Hes);
        LAPACK_A = std::mem::take(&mut content.LAPACK_A);
        LAPACK_wr = std::mem::take(&mut content.LAPACK_wr);
        LAPACK_wi = std::mem::take(&mut content.LAPACK_wi);
        LAPACK_work = std::mem::take(&mut content.LAPACK_work);
        LAPACK_arr = std::mem::take(&mut content.LAPACK_arr);
        content.num_ATimes = 0;
        content.num_iters = 0;
    }

    let flag = (|| -> SUNErrCode {
        let mut normq: sunrealtype;
        let mut new_lambda: sunrealtype = ZERO;
        let mut old_lambda: sunrealtype = ZERO;

        /* Set the initial q = A^{num_warmups}q/||A^{num_warmups}q|| */
        let mut i: i32 = 0;
        loop {
            if i >= content_mut(DEE).num_warmups {
                break;
            }

            let (ATimes, mut ATdata) = {
                let mut guard = content_mut(DEE);
                let content = &mut *guard;
                (content.ATimes.expect("ATimes"), content.ATdata.take())
            };
            let retval = ATimes(&mut ATdata, &V[0], &q);
            {
                let mut guard = content_mut(DEE);
                let content = &mut *guard;
                content.ATdata = ATdata;
                content.num_ATimes += 1;
                content.num_iters += 1;
            }
            if retval != 0 {
                return SUN_ERR_USER_FCN_FAIL;
            }

            let warmup_to_tol = content_mut(DEE).warmup_to_tol;
            if warmup_to_tol {
                new_lambda = N_VDotProd(&V[0], &q); //Rayleigh quotient
            }

            normq = N_VDotProd(&q, &q);

            normq = SUNRsqrt(normq);
            N_VScale(ONE / normq, &q, &V[0]);

            if warmup_to_tol {
                let res = SUNRabs(new_lambda - old_lambda);
                old_lambda = new_lambda;
                if res <= content_mut(DEE).tol_warmup * SUNRabs(new_lambda) {
                    break;
                }
            }

            i += 1;
        }

        let mut i: i32 = 0;
        while (i as sunindextype) < n {
            /* Compute the next Krylov vector */
            let (ATimes, mut ATdata) = {
                let mut guard = content_mut(DEE);
                let content = &mut *guard;
                (content.ATimes.expect("ATimes"), content.ATdata.take())
            };
            let retval = ATimes(&mut ATdata, &V[i as usize], &V[i as usize + 1]);
            {
                let mut guard = content_mut(DEE);
                let content = &mut *guard;
                content.ATdata = ATdata;
                content.num_ATimes += 1;
                content.num_iters += 1;
            }
            if retval != 0 {
                return SUN_ERR_USER_FCN_FAIL;
            }

            /* C passes &Hes[i+1][i] as the out-param; see the module note */
            let mut new_vk_norm = ZERO;
            let _ = SUNModifiedGS(&V, &mut Hes, i + 1, n as i32, &mut new_vk_norm);
            Hes[i as usize + 1][i as usize] = new_vk_norm;

            /* Unitize the computed orthogonal vector */
            N_VScale(
                ONE / Hes[i as usize + 1][i as usize],
                &V[i as usize + 1],
                &V[i as usize + 1],
            );

            i += 1;
        }

        /* Pack the Hessenberg matrix in column-major order for LAPACK dgeev_ call */
        let mut k: usize = 0;
        for j in 0..(n as usize) {
            for i in 0..(n as usize) {
                LAPACK_A[k] = Hes[i][j];
                k += 1;
            }
        }

        let jobvl: u8 = b'N'; // Do not compute left eigenvectors
        let jobvr: u8 = b'N'; // Do not compute right eigenvectors

        /* Call the eigenvalue routine
            return info values refer to
          = 0:  successful exit
          < 0:  if info = -i, the i-th argument had an illegal value.
          > 0:  if info = i, the QR algorithm failed to compute all the
                eigenvalues, and no eigenvectors have been computed;
                elements i+1:N of LAPACK_wr and LAPACK_wi contain
                eigenvalues which have converged.
        */
        let lda: sunindextype = n;
        let mut info: sunindextype = 0;
        let lwork: sunindextype = LAPACK_lwork;
        /* C also passes ldvl = ldvr = n with NULL left/right eigenvector
        arrays; unused for jobvl = jobvr = 'N'. */
        xgeev_f77(
            jobvl,
            jobvr,
            n,
            &mut LAPACK_A,
            lda,
            &mut LAPACK_wr,
            &mut LAPACK_wi,
            &mut LAPACK_work,
            lwork,
            &mut info,
        );

        if info != 0 {
            return SUN_ERR_EXT_FAIL;
        }

        /* order the eigenvalues by their magnitude */
        for i in 0..(n as usize) {
            LAPACK_arr[i][0] = LAPACK_wr[i];
            LAPACK_arr[i][1] = LAPACK_wi[i];
        }

        /* Sort the array using qsort */
        LAPACK_arr[0..(n as usize)].sort_by(|a, b| sundomeigest_Compare(a, b).cmp(&0));

        /* Substitute the ordered eigenvalues back in LAPACK_w* */
        for i in 0..(n as usize) {
            LAPACK_wr[i] = LAPACK_arr[i][0];
            LAPACK_wi[i] = LAPACK_arr[i][1];
        }

        /* Copy the dominant eigenvalue */
        *lambdaR = LAPACK_wr[0];
        *lambdaI = LAPACK_wi[0];

        SUN_SUCCESS
    })();

    /* restore estimator state */
    {
        let mut guard = content_mut(DEE);
        let content = &mut *guard;
        content.Hes = Hes;
        content.LAPACK_A = LAPACK_A;
        content.LAPACK_wr = LAPACK_wr;
        content.LAPACK_wi = LAPACK_wi;
        content.LAPACK_work = LAPACK_work;
        content.LAPACK_arr = LAPACK_arr;
    }

    flag
}

pub fn SUNDomEigEstimator_GetNumIters_Arnoldi(
    DEE: &SUNDomEigEstimator,
    num_iters: &mut i64,
) -> SUNErrCode {
    *num_iters = content_mut(DEE).num_iters;

    SUN_SUCCESS
}

pub fn SUNDomEigEstimator_GetNumRhsEvals_Arnoldi(
    DEE: &SUNDomEigEstimator,
    num_rhs_evals: &mut i64,
) -> SUNErrCode {
    *num_rhs_evals = content_mut(DEE).nfevals;

    SUN_SUCCESS
}

pub fn SUNDomEigEstimator_GetNumATimesCalls_Arnoldi(
    DEE: &SUNDomEigEstimator,
    num_ATimes: &mut i64,
) -> SUNErrCode {
    *num_ATimes = content_mut(DEE).num_ATimes;

    SUN_SUCCESS
}

pub fn SUNDomEigEstimator_Write_Arnoldi(DEE: &SUNDomEigEstimator, outfile: &SUNFile) -> SUNErrCode {
    let content = content_mut(DEE);

    outfile.write_str("\nArnoldi Iteration SUNDomEigEstimator:\n");
    outfile.write_str(&format!("Krylov dimension         = {}\n", content.kry_dim));
    outfile.write_str(&format!(
        "Num. preprocessing iters = {}\n",
        content.num_warmups
    ));
    outfile.write_str(&format!(
        "Num. iters               = {}\n",
        content.num_iters
    ));
    outfile.write_str(&format!(
        "Num. ATimes calls        = {}\n\n",
        content.num_ATimes
    ));

    SUN_SUCCESS
}

pub fn SUNDomEigEstimator_Destroy_Arnoldi(DEEptr: &mut Option<SUNDomEigEstimator>) -> SUNErrCode {
    let DEE = match DEEptr.as_ref() {
        None => return SUN_SUCCESS,
        Some(DEE) => DEE.clone(),
    };

    if DEE
        .content
        .borrow()
        .is::<SUNDomEigEstimatorContent_Arnoldi_>()
    {
        /* delete items from within the content structure */
        let (q, rhs_linY, Fy, work, V, kry_dim) = {
            let mut guard = content_mut(&DEE);
            let content = &mut *guard;
            (
                content.q.take(),
                content.rhs_linY.take(),
                content.Fy.take(),
                content.work.take(),
                content.V.take(),
                content.kry_dim,
            )
        };
        if let Some(q) = q {
            N_VDestroy(q);
        }
        if let Some(rhs_linY) = rhs_linY {
            N_VDestroy(rhs_linY);
        }
        if let Some(Fy) = Fy {
            N_VDestroy(Fy);
        }
        if let Some(work) = work {
            N_VDestroy(work);
        }
        if let Some(V) = V {
            N_VDestroyVectorArray(V, kry_dim + 1);
        }
        {
            let mut guard = content_mut(&DEE);
            let content = &mut *guard;
            content.LAPACK_A = Vec::new();
            content.LAPACK_wr = Vec::new();
            content.LAPACK_wi = Vec::new();
            content.LAPACK_work = Vec::new();
            /* free LAPACK_arr */
            content.LAPACK_arr = Vec::new();
            /* free Hes */
            content.Hes = Vec::new();
        }

        /* C: free(DEE->content); DEE->content = NULL; -- dropping the content
        box also releases the ATimes data token, which holds the estimator's
        own handle whenever SetRhs installed the internal DQ ATimes. */
        let old_content = std::mem::replace(&mut *DEE.content.borrow_mut(), Box::new(()));
        drop(old_content);
    }

    /* C: free(DEE->ops); free(DEE); *DEEptr = NULL; */
    drop(DEE);
    *DEEptr = None;
    SUN_SUCCESS
}

// Comparison function for qsort
pub fn sundomeigest_Compare(a: &[sunrealtype], b: &[sunrealtype]) -> i32 {
    let cplx_a = a;
    let cplx_b = b;

    let mag_a = SUNRsqrt(cplx_a[0] * cplx_a[0] + cplx_a[1] * cplx_a[1]);
    let mag_b = SUNRsqrt(cplx_b[0] * cplx_b[0] + cplx_b[1] * cplx_b[1]);
    (mag_b > mag_a) as i32 - (mag_b < mag_a) as i32 // Descending order
}

/*---------------------------------------------------------------
dee_DQJtimes_Arnoldi:

This routine generates a difference quotient approximation to
the Jacobian-vector product f_y(t,y) * v. The approximation is
Jv = [f(y + v*sig) - f(y)]/sig, where
    sig = sign(y^T v) * sqrt(unit roundoff)
          * max(|y^T v|, ||v||_1) / (v^T v).
---------------------------------------------------------------*/
pub fn dee_DQJtimes_Arnoldi(
    voidstarDEE: &mut Option<Box<dyn Any>>,
    v: &N_Vector,
    Jv: &N_Vector,
) -> SUNErrCode {
    let DEE: SUNDomEigEstimator = voidstarDEE
        .as_ref()
        .and_then(|data| data.downcast_ref::<SUNDomEigEstimator>())
        .cloned()
        .expect("dee_DQJtimes_Arnoldi: ATimes data is not a SUNDomEigEstimator");

    let vdotv = N_VDotProd(v, v);
    if vdotv <= SUN_SMALL_REAL {
        N_VScale(ZERO, v, Jv);
        return SUN_SUCCESS;
    }

    if content_mut(&DEE).work.is_none() {
        let work = N_VClone(v);
        content_mut(&DEE).work = work;
    }
    if content_mut(&DEE).Fy.is_none() {
        let Fy = N_VClone(v);
        content_mut(&DEE).Fy = Fy;
    }

    let (y, work, Fy, rhsfn, rhs_linT) = {
        let mut guard = content_mut(&DEE);
        let content = &mut *guard;
        (
            content.rhs_linY.as_ref().expect("rhs_linY").clone(),
            content.work.as_ref().expect("work").clone(),
            content.Fy.as_ref().expect("Fy").clone(),
            content.rhsfn.expect("rhsfn"),
            content.rhs_linT,
        )
    };
    /* the RHS data token is handed to the callback and restored on every
    return path below; C's function-scope `sig`/`siginv`/`iter` are only live
    inside this section */
    let mut rhs_data = content_mut(&DEE).rhs_data.take();

    let flag = (|| -> SUNErrCode {
        let mut retval = rhsfn(rhs_linT, &y, &Fy, &mut rhs_data);
        content_mut(&DEE).nfevals += 1;
        if retval != 0 {
            return SUN_ERR_USER_FCN_FAIL;
        }

        /* Initialize perturbation */
        let ydotv = N_VDotProd(&y, v);
        let sq1norm = N_VL1Norm(v);
        let sign = if ydotv >= ZERO { ONE } else { -ONE };
        let sqrteps = SUNRsqrt(SUN_UNIT_ROUNDOFF);
        let mut sig = sign * sqrteps * SUNMAX(SUNRabs(ydotv), sq1norm) / vdotv;

        for _iter in 0..MAX_DQITERS {
            /* Set work = y + sig*v */
            N_VLinearSum(sig, v, ONE, &y, &work);

            /* Set Jv = f(tn, y+sig*v) */
            retval = rhsfn(rhs_linT, &work, Jv, &mut rhs_data);
            content_mut(&DEE).nfevals += 1;
            if retval == 0 {
                break;
            }
            if retval < 0 {
                return SUN_ERR_USER_FCN_FAIL;
            }

            /* If f failed recoverably, shrink sig and retry */
            sig *= 0.25;
        }

        /* If retval still isn't 0, return with a recoverable failure */
        if retval > 0 {
            return 1;
        }

        /* Replace Jv by (Jv - fn)/sig */
        let siginv = ONE / sig;
        N_VLinearSum(siginv, Jv, -siginv, &Fy, Jv);

        SUN_SUCCESS
    })();

    content_mut(&DEE).rhs_data = rhs_data;

    flag
}

/*---------------------------------------------------------------
xgeev_f77:

Native stand-in for the LAPACK `dgeev` call this module makes with
`jobvl = jobvr = 'N'` (eigenvalues only, no eigenvectors). LAPACK is
excluded from this port, so the eigenvalues of the `n x n` column-major
matrix `a` (leading dimension `lda`) are computed here by a Francis
double-shift QR iteration (the EISPACK `hqr` algorithm of Martin, Peters
and Wilkinson) applied directly to the matrix, which this module always
supplies in upper Hessenberg form.

`a` is overwritten. On return `wr`/`wi` hold the real/imaginary parts of
the eigenvalues, and `info` is
  = 0: successful exit,
  > 0: the QR iteration failed; elements info+1:n of wr/wi have converged.
With `lwork = -1` the routine performs LAPACK's workspace query: it stores
the required workspace length (the `3*n` minimum documented for
`jobvl = jobvr = 'N'`) in `work[0]` and returns.
---------------------------------------------------------------*/
fn xgeev_f77(
    _jobvl: u8,
    _jobvr: u8,
    n: sunindextype,
    a: &mut [sunrealtype],
    lda: sunindextype,
    wr: &mut [sunrealtype],
    wi: &mut [sunrealtype],
    work: &mut [sunrealtype],
    lwork: sunindextype,
    info: &mut sunindextype,
) {
    /* workspace query */
    if lwork == -1 {
        work[0] = (3 * n) as sunrealtype;
        *info = 0;
        return;
    }

    *info = 0;

    let nn = n as usize;
    let lda = lda as usize;
    let idx = |i: isize, j: isize| -> usize { (i as usize) + (j as usize) * lda };

    /* compute the matrix norm (used when a diagonal 2x2 block vanishes) */
    let mut norm = ZERO;
    let mut kcol: usize = 0;
    for i in 0..nn {
        for j in kcol..nn {
            norm += SUNRabs(a[i + j * lda]);
        }
        kcol = i;
    }

    let mut en: isize = nn as isize - 1;
    let mut t = ZERO;
    let mut itn: i32 = 30 * (nn as i32);

    /* search for the next eigenvalues */
    while en >= 0 {
        let mut its: i32 = 0;
        let na: isize = en - 1;
        let enm2: isize = na - 1;

        let mut l: isize;
        let mut x: sunrealtype;
        /* `y`/`w` are only read on the two-roots path, which always assigns
        them first; they are seeded here because the borrow checker does not
        correlate `one_root` with that path. */
        let mut y: sunrealtype = ZERO;
        let mut w: sunrealtype = ZERO;
        let one_root: bool;

        loop {
            /* look for a single small sub-diagonal element */
            l = en;
            while l > 0 {
                let mut s = SUNRabs(a[idx(l - 1, l - 1)]) + SUNRabs(a[idx(l, l)]);
                if s == ZERO {
                    s = norm;
                }
                let tst1 = s;
                let tst2 = tst1 + SUNRabs(a[idx(l, l - 1)]);
                if tst2 == tst1 {
                    break;
                }
                l -= 1;
            }

            /* form shift */
            x = a[idx(en, en)];
            if l == en {
                one_root = true;
                break;
            }
            y = a[idx(na, na)];
            w = a[idx(en, na)] * a[idx(na, en)];
            if l == na {
                one_root = false;
                break;
            }
            if itn == 0 {
                /* the eigenvalues 1:en (1-based) have not been determined */
                *info = en as sunindextype + 1;
                return;
            }
            if its == 10 || its == 20 {
                /* form exceptional shift */
                t += x;
                for i in 0..=en {
                    a[idx(i, i)] -= x;
                }
                let s = SUNRabs(a[idx(en, na)]) + SUNRabs(a[idx(na, enm2)]);
                x = 0.75 * s;
                y = x;
                w = -0.4375 * s * s;
            }
            its += 1;
            itn -= 1;

            /* look for two consecutive small sub-diagonal elements */
            let mut m: isize = enm2;
            let mut p: sunrealtype;
            let mut q: sunrealtype;
            let mut r: sunrealtype;
            loop {
                let zz = a[idx(m, m)];
                let rr = x - zz;
                let ss = y - zz;
                p = (rr * ss - w) / a[idx(m + 1, m)] + a[idx(m, m + 1)];
                q = a[idx(m + 1, m + 1)] - zz - rr - ss;
                r = a[idx(m + 2, m + 1)];
                let s = SUNRabs(p) + SUNRabs(q) + SUNRabs(r);
                p /= s;
                q /= s;
                r /= s;
                if m == l {
                    break;
                }
                let tst1 = SUNRabs(p)
                    * (SUNRabs(a[idx(m - 1, m - 1)]) + SUNRabs(zz) + SUNRabs(a[idx(m + 1, m + 1)]));
                let tst2 = tst1 + SUNRabs(a[idx(m, m - 1)]) * (SUNRabs(q) + SUNRabs(r));
                if tst2 == tst1 {
                    break;
                }
                m -= 1;
            }

            let mp2 = m + 2;
            let mut i = mp2;
            while i <= en {
                a[idx(i, i - 2)] = ZERO;
                if i != mp2 {
                    a[idx(i, i - 3)] = ZERO;
                }
                i += 1;
            }

            /* double qr step involving rows l to en and columns m to en */
            let mut k = m;
            while k <= na {
                let notlas = k != na;
                if k != m {
                    p = a[idx(k, k - 1)];
                    q = a[idx(k + 1, k - 1)];
                    r = if notlas { a[idx(k + 2, k - 1)] } else { ZERO };
                    x = SUNRabs(p) + SUNRabs(q) + SUNRabs(r);
                    if x == ZERO {
                        k += 1;
                        continue;
                    }
                    p /= x;
                    q /= x;
                    r /= x;
                }
                let s = dee_sign(SUNRsqrt(p * p + q * q + r * r), p);
                if k != m {
                    a[idx(k, k - 1)] = -s * x;
                } else if l != m {
                    a[idx(k, k - 1)] = -a[idx(k, k - 1)];
                }
                p += s;
                x = p / s;
                y = q / s;
                let zz = r / s;
                q /= p;
                r /= p;
                if notlas {
                    /* row modification */
                    let mut j = k;
                    while j <= en {
                        let pp = a[idx(k, j)] + q * a[idx(k + 1, j)] + r * a[idx(k + 2, j)];
                        a[idx(k, j)] -= pp * x;
                        a[idx(k + 1, j)] -= pp * y;
                        a[idx(k + 2, j)] -= pp * zz;
                        j += 1;
                    }
                    let jmax = SUNMIN(en, k + 3);
                    /* column modification */
                    let mut i = l;
                    while i <= jmax {
                        let pp = x * a[idx(i, k)] + y * a[idx(i, k + 1)] + zz * a[idx(i, k + 2)];
                        a[idx(i, k)] -= pp;
                        a[idx(i, k + 1)] -= pp * q;
                        a[idx(i, k + 2)] -= pp * r;
                        i += 1;
                    }
                } else {
                    /* row modification */
                    let mut j = k;
                    while j <= en {
                        let pp = a[idx(k, j)] + q * a[idx(k + 1, j)];
                        a[idx(k, j)] -= pp * x;
                        a[idx(k + 1, j)] -= pp * y;
                        j += 1;
                    }
                    let jmax = SUNMIN(en, k + 3);
                    /* column modification */
                    let mut i = l;
                    while i <= jmax {
                        let pp = x * a[idx(i, k)] + y * a[idx(i, k + 1)];
                        a[idx(i, k)] -= pp;
                        a[idx(i, k + 1)] -= pp * q;
                        i += 1;
                    }
                }
                k += 1;
            }
        }

        if one_root {
            /* one root found */
            wr[en as usize] = x + t;
            wi[en as usize] = ZERO;
            en = na;
        } else {
            /* two roots found */
            let p = (y - x) / 2.0;
            let q = p * p + w;
            let mut zz = SUNRsqrt(SUNRabs(q));
            let x = x + t;
            if q >= ZERO {
                /* real pair */
                zz = p + dee_sign(zz, p);
                wr[na as usize] = x + zz;
                wr[en as usize] = wr[na as usize];
                if zz != ZERO {
                    wr[en as usize] = x - w / zz;
                }
                wi[na as usize] = ZERO;
                wi[en as usize] = ZERO;
            } else {
                /* complex pair */
                wr[na as usize] = x + p;
                wr[en as usize] = x + p;
                wi[na as usize] = zz;
                wi[en as usize] = -zz;
            }
            en = enm2;
        }
    }
}

/// Fortran `SIGN(a, b)`: `|a|` carrying the sign of `b`, with `b = -0.0`
/// treated as positive (F77 semantics, unlike `copysign`).
fn dee_sign(a: sunrealtype, b: sunrealtype) -> sunrealtype {
    if b >= ZERO {
        SUNRabs(a)
    } else {
        -SUNRabs(a)
    }
}
