//! Port of `src/cvodes/cvodes_bandpre.c` (+ `src/cvodes/cvodes_bandpre_impl.h`
//! and `include/cvodes/cvodes_bandpre.h` folded).
//!
//! Banded difference-quotient Jacobian-based preconditioner and solver
//! routines for use with the CVSLS linear solver interface, plus the
//! PART II backward-problem wrapper (`CVBandPrecInitB`).
//!
//! The preconditioner data lives in `cvls_mem.P_data`
//! (`Option<Box<dyn Any>>` holding a [`CVBandPrecDataRec`]); the CVSLS
//! interface (`cvodes_ls`) `Option::take`s that box around each
//! psetup/psolve invocation, so the callbacks here receive it as
//! `&mut Option<Box<dyn Any>>` and downcast.

use std::any::Any;

use crate::cvodes_impl::*;
use crate::cvodes_ls::{
    cvls_mem_mut, CVLsMemRec, CVodeSetPreconditioner, CVLS_ILL_INPUT, CVLS_LMEM_NULL,
    CVLS_MEM_FAIL, CVLS_NO_ADJ, CVLS_PMEM_NULL, CVLS_SUCCESS, CVLS_SUNLS_FAIL,
};
use sundials_core::sundials_errors::SUN_SUCCESS;
use sundials_core::sundials_linearsolver::{
    SUNLinSolInitialize, SUNLinSolSolve, SUNLinSolSpace, SUNLinearSolver,
};
use sundials_core::sundials_math::{SUNMAX, SUNMIN, SUNRabs, SUNRsqrt};
use sundials_core::sundials_matrix::{
    SUNMatCopy, SUNMatScaleAddI, SUNMatSpace, SUNMatZero, SUNMatrix,
};
use sundials_core::sundials_nvector::{
    N_VClone, N_VGetArrayPointer, N_VScale, N_VSpace, N_VWrmsNorm, N_Vector,
};
use sundials_core::sundials_types::*;
use sundials_core::sunlinsol_band::{SUNLinSolSetup_Band, SUNLinSol_Band};
use sundials_core::sunmatrix_band::{
    SM_COLUMN_ELEMENT_IDX, SM_SUBAND_B, SUNBandMatrixStorage, SUNBandMatrix_Column,
};

/* File-scope constants (shadow the same-named `cvodes_impl` constants,
which carry identical values, exactly as the C `#define`s shadow
nothing -- each C file redefines them locally). */
const MIN_INC_MULT: sunrealtype = 1000.0;
const ZERO: sunrealtype = 0.0;
const ONE: sunrealtype = 1.0;
const TWO: sunrealtype = 2.0;

/*-----------------------------------------------------------------
  Type: CVBandPrecData (cvodes_bandpre_impl.h)
  -----------------------------------------------------------------*/

pub struct CVBandPrecDataRec {
    /* Data set by user in CVBandPrecInit */
    pub N: sunindextype,
    pub ml: sunindextype,
    pub mu: sunindextype,

    /* Data set by CVBandPrecSetup */
    pub savedJ: SUNMatrix,
    pub savedP: SUNMatrix,
    pub LS: SUNLinearSolver,
    pub tmp1: N_Vector,
    pub tmp2: N_Vector,

    /* Rhs calls */
    pub nfeBP: i64,

    /* Pointer to cvode_mem */
    pub cvode_mem: CVodeMem,
}

pub type CVBandPrecData = Box<CVBandPrecDataRec>;

/*-----------------------------------------------------------------
  CVBANDPRE error messages (cvodes_bandpre_impl.h)
  -----------------------------------------------------------------*/

pub const MSGBP_MEM_NULL: &str = "Integrator memory is NULL.";
pub const MSGBP_LMEM_NULL: &str =
    "Linear solver memory is NULL. One of the SPILS linear solvers must be attached.";
pub const MSGBP_MEM_FAIL: &str = "A memory request failed.";
pub const MSGBP_BAD_NVECTOR: &str = "A required vector operation is not implemented.";
pub const MSGBP_SUNMAT_FAIL: &str = "An error arose from a SUNBandMatrix routine.";
pub const MSGBP_SUNLS_FAIL: &str = "An error arose from a SUNBandLinearSolver routine.";
pub const MSGBP_PMEM_NULL: &str =
    "Band preconditioner memory is NULL. CVBandPrecInit must be called.";
pub const MSGBP_RHSFUNC_FAILED: &str =
    "The right-hand side routine failed in an unrecoverable manner.";

pub const MSGBP_NO_ADJ: &str = "Illegal attempt to call before calling CVodeAdjInit.";
pub const MSGBP_BAD_WHICH: &str = "Illegal value for parameter which.";

/*================================================================
  PART I - Forward Problems
  ================================================================*/

/*-----------------------------------------------------------------
  Initialization, Free, and Get Functions
  NOTE: The band linear solver assumes a serial/OpenMP/Pthreads
        implementation of the NVECTOR package. Therefore,
        CVBandPrecInit will first test for a compatible N_Vector
        internal representation by checking that the function
        N_VGetArrayPointer exists.
  -----------------------------------------------------------------*/
pub fn CVBandPrecInit(
    cvode_mem: &CVodeMem,
    N: sunindextype,
    mu: sunindextype,
    ml: sunindextype,
) -> i32 {
    /* NULL-mem check: handled by type system */
    let cv_mem = cvode_mem;

    /* Test if the CVSLS linear solver interface has been attached */
    let attached = {
        let mem = cv_mem.borrow();
        mem.cv_lmem.as_ref().is_some_and(|b| b.is::<CVLsMemRec>())
    };
    if !attached {
        cvProcessError(
            Some(cv_mem),
            CVLS_LMEM_NULL,
            line!() as i32,
            "CVBandPrecInit",
            file!(),
            MSGBP_LMEM_NULL,
        );
        return CVLS_LMEM_NULL;
    }

    /* Test compatibility of NVECTOR package with the BAND preconditioner */
    let (tempv, sunctx) = {
        let mem = cv_mem.borrow();
        (
            mem.cv_tempv.clone().expect("cv_tempv"),
            mem.cv_sunctx.clone(),
        )
    };
    if tempv.ops.borrow().nvgetarraypointer.is_none() {
        cvProcessError(
            Some(cv_mem),
            CVLS_ILL_INPUT,
            line!() as i32,
            "CVBandPrecInit",
            file!(),
            MSGBP_BAD_NVECTOR,
        );
        return CVLS_ILL_INPUT;
    }

    /* Allocate data memory (Rust: the record is assembled at the end of
    this function; the C malloc-NULL branch has no analogue) */

    /* Load pointers and bandwidths into pdata block. */
    let mup = SUNMIN(N - 1, SUNMAX(0, mu));
    let mlp = SUNMIN(N - 1, SUNMAX(0, ml));

    /* Initialize nfeBP counter */
    let nfeBP: i64 = 0;

    /* Allocate memory for saved banded Jacobian approximation. */
    let savedJ = match SUNBandMatrixStorage(N, mup, mlp, mup, &sunctx) {
        None => {
            cvProcessError(
                Some(cv_mem),
                CVLS_MEM_FAIL,
                line!() as i32,
                "CVBandPrecInit",
                file!(),
                MSGBP_MEM_FAIL,
            );
            return CVLS_MEM_FAIL;
        }
        Some(m) => m,
    };

    /* Allocate memory for banded preconditioner. */
    let storagemu = SUNMIN(N - 1, mup + mlp);
    let savedP = match SUNBandMatrixStorage(N, mup, mlp, storagemu, &sunctx) {
        None => {
            cvProcessError(
                Some(cv_mem),
                CVLS_MEM_FAIL,
                line!() as i32,
                "CVBandPrecInit",
                file!(),
                MSGBP_MEM_FAIL,
            );
            return CVLS_MEM_FAIL;
        }
        Some(m) => m,
    };

    /* Allocate memory for banded linear solver */
    let LS = match SUNLinSol_Band(&tempv, &savedP, &sunctx) {
        None => {
            cvProcessError(
                Some(cv_mem),
                CVLS_MEM_FAIL,
                line!() as i32,
                "CVBandPrecInit",
                file!(),
                MSGBP_MEM_FAIL,
            );
            return CVLS_MEM_FAIL;
        }
        Some(s) => s,
    };

    /* allocate memory for temporary N_Vectors */
    let tmp1 = match N_VClone(&tempv) {
        None => {
            cvProcessError(
                Some(cv_mem),
                CVLS_MEM_FAIL,
                line!() as i32,
                "CVBandPrecInit",
                file!(),
                MSGBP_MEM_FAIL,
            );
            return CVLS_MEM_FAIL;
        }
        Some(v) => v,
    };
    let tmp2 = match N_VClone(&tempv) {
        None => {
            cvProcessError(
                Some(cv_mem),
                CVLS_MEM_FAIL,
                line!() as i32,
                "CVBandPrecInit",
                file!(),
                MSGBP_MEM_FAIL,
            );
            return CVLS_MEM_FAIL;
        }
        Some(v) => v,
    };

    /* initialize band linear solver object */
    let flag = SUNLinSolInitialize(&LS);
    if flag != SUN_SUCCESS {
        cvProcessError(
            Some(cv_mem),
            CVLS_SUNLS_FAIL,
            line!() as i32,
            "CVBandPrecInit",
            file!(),
            MSGBP_SUNLS_FAIL,
        );
        return CVLS_SUNLS_FAIL;
    }

    /* make sure P_data is free from any previous allocations */
    let pfree = cvls_mem_mut(cv_mem).pfree;
    if let Some(pfree) = pfree {
        pfree(cv_mem);
    }

    {
        let mut ls_mem = cvls_mem_mut(cv_mem);

        /* Point to the new P_data field in the LS memory */
        ls_mem.P_data = Some(Box::new(CVBandPrecDataRec {
            N,
            ml: mlp,
            mu: mup,
            savedJ,
            savedP,
            LS,
            tmp1,
            tmp2,
            nfeBP,
            cvode_mem: cv_mem.clone(),
        }));

        /* Attach the pfree function */
        ls_mem.pfree = Some(cvBandPrecFree);
    }

    /* Attach preconditioner solve and setup functions */
    CVodeSetPreconditioner(cvode_mem, Some(cvBandPrecSetup), Some(cvBandPrecSolve))
}

pub fn CVBandPrecGetWorkSpace(cvode_mem: &CVodeMem, lenrwBP: &mut i64, leniwBP: &mut i64) -> i32 {
    /* NULL-mem check: handled by type system */
    let cv_mem = cvode_mem;

    let attached = {
        let mem = cv_mem.borrow();
        mem.cv_lmem.as_ref().is_some_and(|b| b.is::<CVLsMemRec>())
    };
    if !attached {
        cvProcessError(
            Some(cv_mem),
            CVLS_LMEM_NULL,
            line!() as i32,
            "CVBandPrecGetWorkSpace",
            file!(),
            MSGBP_LMEM_NULL,
        );
        return CVLS_LMEM_NULL;
    }

    let handles = {
        let ls_mem = cvls_mem_mut(cv_mem);
        ls_mem
            .P_data
            .as_ref()
            .and_then(|b| b.downcast_ref::<CVBandPrecDataRec>())
            .map(|p| (p.savedJ.clone(), p.savedP.clone(), p.LS.clone()))
    };
    let (savedJ, savedP, LS) = match handles {
        None => {
            cvProcessError(
                Some(cv_mem),
                CVLS_PMEM_NULL,
                line!() as i32,
                "CVBandPrecGetWorkSpace",
                file!(),
                MSGBP_PMEM_NULL,
            );
            return CVLS_PMEM_NULL;
        }
        Some(h) => h,
    };

    /* sum space requirements for all objects in pdata */
    *leniwBP = 4;
    *lenrwBP = 0;
    let tempv = cv_mem.borrow().cv_tempv.clone().expect("cv_tempv");
    if tempv.ops.borrow().nvspace.is_some() {
        let mut lrw1: sunindextype = 0;
        let mut liw1: sunindextype = 0;
        N_VSpace(&tempv, &mut lrw1, &mut liw1);
        *leniwBP += 2 * liw1;
        *lenrwBP += 2 * lrw1;
    }
    if savedJ.ops.borrow().space.is_some() {
        let mut lrw: i64 = 0;
        let mut liw: i64 = 0;
        let flag = SUNMatSpace(&savedJ, &mut lrw, &mut liw);
        if flag != 0 {
            return -1;
        }
        *leniwBP += liw;
        *lenrwBP += lrw;
    }
    if savedP.ops.borrow().space.is_some() {
        let mut lrw: i64 = 0;
        let mut liw: i64 = 0;
        let flag = SUNMatSpace(&savedP, &mut lrw, &mut liw);
        if flag != 0 {
            return -1;
        }
        *leniwBP += liw;
        *lenrwBP += lrw;
    }
    if LS.ops.borrow().space.is_some() {
        let mut lrw: i64 = 0;
        let mut liw: i64 = 0;
        let flag = SUNLinSolSpace(&LS, &mut lrw, &mut liw);
        if flag != 0 {
            return -1;
        }
        *leniwBP += liw;
        *lenrwBP += lrw;
    }

    CVLS_SUCCESS
}

pub fn CVBandPrecGetNumRhsEvals(cvode_mem: &CVodeMem, nfevalsBP: &mut i64) -> i32 {
    /* NULL-mem check: handled by type system */
    let cv_mem = cvode_mem;

    let attached = {
        let mem = cv_mem.borrow();
        mem.cv_lmem.as_ref().is_some_and(|b| b.is::<CVLsMemRec>())
    };
    if !attached {
        cvProcessError(
            Some(cv_mem),
            CVLS_LMEM_NULL,
            line!() as i32,
            "CVBandPrecGetNumRhsEvals",
            file!(),
            MSGBP_LMEM_NULL,
        );
        return CVLS_LMEM_NULL;
    }

    let nfeBP = {
        let ls_mem = cvls_mem_mut(cv_mem);
        ls_mem
            .P_data
            .as_ref()
            .and_then(|b| b.downcast_ref::<CVBandPrecDataRec>())
            .map(|p| p.nfeBP)
    };
    match nfeBP {
        None => {
            cvProcessError(
                Some(cv_mem),
                CVLS_PMEM_NULL,
                line!() as i32,
                "CVBandPrecGetNumRhsEvals",
                file!(),
                MSGBP_PMEM_NULL,
            );
            CVLS_PMEM_NULL
        }
        Some(nfeBP) => {
            *nfevalsBP = nfeBP;
            CVLS_SUCCESS
        }
    }
}

/*-----------------------------------------------------------------
  cvBandPrecSetup
  -----------------------------------------------------------------
  Together cvBandPrecSetup and cvBandPrecSolve use a banded
  difference quotient Jacobian to create a preconditioner.
  cvBandPrecSetup calculates a new J, if necessary, then
  calculates P = I - gamma*J, and does an LU factorization of P.

  The value to be returned by the cvBandPrecSetup function is
    0  if successful, or
    1  if the band factorization failed.
  -----------------------------------------------------------------*/
fn cvBandPrecSetup(
    t: sunrealtype,
    y: &N_Vector,
    fy: &N_Vector,
    jok: sunbooleantype,
    jcurPtr: &mut sunbooleantype,
    gamma: sunrealtype,
    bp_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    /* Assume matrix and lpivots have already been allocated. */
    let pdata: &mut CVBandPrecDataRec = bp_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<CVBandPrecDataRec>())
        .expect("bp_data is CVBandPrecData");
    let cv_mem = pdata.cvode_mem.clone();

    if jok {
        /* If jok = SUNTRUE, use saved copy of J. */
        *jcurPtr = SUNFALSE;
        let retval = SUNMatCopy(&pdata.savedJ, &pdata.savedP);
        if retval < 0 {
            cvProcessError(
                Some(&cv_mem),
                -1,
                line!() as i32,
                "cvBandPrecSetup",
                file!(),
                MSGBP_SUNMAT_FAIL,
            );
            return -1;
        }
        if retval > 0 {
            return 1;
        }
    } else {
        /* If jok = SUNFALSE, call CVBandPDQJac for new J value. */
        *jcurPtr = SUNTRUE;
        let retval = SUNMatZero(&pdata.savedJ);
        if retval < 0 {
            cvProcessError(
                Some(&cv_mem),
                -1,
                line!() as i32,
                "cvBandPrecSetup",
                file!(),
                MSGBP_SUNMAT_FAIL,
            );
            return -1;
        }
        if retval > 0 {
            return 1;
        }

        let tmp1 = pdata.tmp1.clone();
        let tmp2 = pdata.tmp2.clone();
        let retval = cvBandPrecDQJac(pdata, t, y, fy, &tmp1, &tmp2);
        if retval < 0 {
            cvProcessError(
                Some(&cv_mem),
                -1,
                line!() as i32,
                "cvBandPrecSetup",
                file!(),
                MSGBP_RHSFUNC_FAILED,
            );
            return -1;
        }
        if retval > 0 {
            return 1;
        }

        let retval = SUNMatCopy(&pdata.savedJ, &pdata.savedP);
        if retval < 0 {
            cvProcessError(
                Some(&cv_mem),
                -1,
                line!() as i32,
                "cvBandPrecSetup",
                file!(),
                MSGBP_SUNMAT_FAIL,
            );
            return -1;
        }
        if retval > 0 {
            return 1;
        }
    }

    /* Scale and add identity to get savedP = I - gamma*J. */
    let retval = SUNMatScaleAddI(-gamma, &pdata.savedP);
    if retval != 0 {
        cvProcessError(
            Some(&cv_mem),
            -1,
            line!() as i32,
            "cvBandPrecSetup",
            file!(),
            MSGBP_SUNMAT_FAIL,
        );
        return -1;
    }

    /* Do LU factorization of matrix and return error flag */
    SUNLinSolSetup_Band(&pdata.LS, Some(&pdata.savedP))
}

/*-----------------------------------------------------------------
  cvBandPrecSolve
  -----------------------------------------------------------------
  cvBandPrecSolve solves a linear system P z = r, where P is the
  matrix computed by cvBandPrecond.

  The value returned by the cvBandPrecSolve function is always 0,
  indicating success.
  -----------------------------------------------------------------*/
fn cvBandPrecSolve(
    _t: sunrealtype,
    _y: &N_Vector,
    _fy: &N_Vector,
    r: &N_Vector,
    z: &N_Vector,
    _gamma: sunrealtype,
    _delta: sunrealtype,
    _lr: i32,
    bp_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    /* Assume matrix and lpivots have already been allocated. */
    let pdata: &CVBandPrecDataRec = bp_data
        .as_ref()
        .and_then(|b| b.downcast_ref::<CVBandPrecDataRec>())
        .expect("bp_data is CVBandPrecData");

    /* Call banded solver object to do the work */
    SUNLinSolSolve(&pdata.LS, Some(&pdata.savedP), z, r, ZERO)
}

fn cvBandPrecFree(cv_mem: &CVodeMem) -> i32 {
    let attached = {
        let mem = cv_mem.borrow();
        mem.cv_lmem.as_ref().is_some_and(|b| b.is::<CVLsMemRec>())
    };
    if !attached {
        return 0;
    }

    let pdata = cvls_mem_mut(cv_mem).P_data.take();
    if pdata.is_none() {
        return 0;
    }

    /* SUNLinSolFree(LS) / SUNMatDestroy(savedP, savedJ) /
    N_VDestroy(tmp1, tmp2): dropping the record releases everything
    (C frees each explicitly and leaves cvls_mem->P_data dangling;
    the Rust take() leaves it None) */
    drop(pdata);

    0
}

/*-----------------------------------------------------------------
  cvBandPrecDQJac
  -----------------------------------------------------------------
  This routine generates a banded difference quotient approximation
  to the Jacobian of f(t,y). It assumes that a band SUNMatrix is
  stored column-wise, and that elements within each column are
  contiguous. This makes it possible to get the address of a column
  of J via the accessor function SUNBandMatrix_Column() and to
  write a simple for loop to set each of the elements of a column
  in succession.
  -----------------------------------------------------------------*/
fn cvBandPrecDQJac(
    pdata: &mut CVBandPrecDataRec,
    t: sunrealtype,
    y: &N_Vector,
    fy: &N_Vector,
    ftemp: &N_Vector,
    ytemp: &N_Vector,
) -> i32 {
    let cv_mem = pdata.cvode_mem.clone();

    /* Copy the fields C reads through cv_mem-> out of the mem
    (granular borrow: nothing is held across callbacks/vector ops) */
    let (uround, h, f, ewt, constraints) = {
        let mem = cv_mem.borrow();
        (
            mem.cv_uround,
            mem.cv_h,
            mem.cv_f.expect("cv_f"),
            mem.cv_ewt.clone().expect("cv_ewt"),
            mem.cv_constraints.clone(),
        )
    };

    /* Load ytemp with y = predicted y vector. */
    N_VScale(ONE, y, ytemp);

    /* Set minimum increment based on uround and norm of f. */
    let srur = SUNRsqrt(uround);
    let fnorm = N_VWrmsNorm(fy, &ewt);
    let minInc = if fnorm != ZERO {
        MIN_INC_MULT * SUNRabs(h) * uround * (pdata.N as sunrealtype) * fnorm
    } else {
        ONE
    };

    /* Set bandwidth and number of column groups for band differencing. */
    let width = pdata.ml + pdata.mu + 1;
    let ngroups = SUNMIN(width, pdata.N);

    let s_mu = SM_SUBAND_B(&pdata.savedJ);

    for group in 1..=ngroups {
        /* Increment all y_j in group. */
        {
            let y_data = N_VGetArrayPointer(y).expect("y data");
            let ewt_data = N_VGetArrayPointer(&ewt).expect("ewt data");
            let mut ytemp_data = N_VGetArrayPointer(ytemp).expect("ytemp data");
            let cns_data = constraints
                .as_ref()
                .map(|c| N_VGetArrayPointer(c).expect("constraints data"));

            let mut j = group - 1;
            while j < pdata.N {
                let ju = j as usize;
                let mut inc = SUNMAX(srur * SUNRabs(y_data[ju]), minInc / ewt_data[ju]);
                let yj = y_data[ju];

                /* Adjust sign(inc) again if yj has an inequality constraint. */
                if let Some(cns_data) = &cns_data {
                    let conj = cns_data[ju];
                    if SUNRabs(conj) == ONE {
                        if (yj + inc) * conj < ZERO {
                            inc = -inc;
                        }
                    } else if SUNRabs(conj) == TWO {
                        if (yj + inc) * conj <= ZERO {
                            inc = -inc;
                        }
                    }
                }

                ytemp_data[ju] += inc;
                j += width;
            }
        }

        /* Evaluate f with incremented y. */
        let retval = {
            let mut user_data = cv_mem.borrow_mut().cv_user_data.take();
            let retval = f(t, ytemp, ftemp, &mut user_data);
            cv_mem.borrow_mut().cv_user_data = user_data;
            retval
        };
        pdata.nfeBP += 1;
        if retval != 0 {
            return retval;
        }

        /* Restore ytemp, then form and load difference quotients. */
        {
            let y_data = N_VGetArrayPointer(y).expect("y data");
            let ewt_data = N_VGetArrayPointer(&ewt).expect("ewt data");
            let fy_data = N_VGetArrayPointer(fy).expect("fy data");
            let ftemp_data = N_VGetArrayPointer(ftemp).expect("ftemp data");
            let mut ytemp_data = N_VGetArrayPointer(ytemp).expect("ytemp data");
            let cns_data = constraints
                .as_ref()
                .map(|c| N_VGetArrayPointer(c).expect("constraints data"));

            let mut j = group - 1;
            while j < pdata.N {
                let ju = j as usize;
                let yj = y_data[ju];
                ytemp_data[ju] = y_data[ju];
                let mut col_j = SUNBandMatrix_Column(&pdata.savedJ, j);
                let mut inc = SUNMAX(srur * SUNRabs(y_data[ju]), minInc / ewt_data[ju]);

                /* Adjust sign(inc) as before. */
                if let Some(cns_data) = &cns_data {
                    let conj = cns_data[ju];
                    if SUNRabs(conj) == ONE {
                        if (yj + inc) * conj < ZERO {
                            inc = -inc;
                        }
                    } else if SUNRabs(conj) == TWO {
                        if (yj + inc) * conj <= ZERO {
                            inc = -inc;
                        }
                    }
                }

                let inc_inv = ONE / inc;
                let i1 = SUNMAX(0, j - pdata.mu);
                let i2 = SUNMIN(j + pdata.ml, pdata.N - 1);
                let mut i = i1;
                while i <= i2 {
                    col_j[SM_COLUMN_ELEMENT_IDX(i, j, s_mu)] =
                        inc_inv * (ftemp_data[i as usize] - fy_data[i as usize]);
                    i += 1;
                }
                drop(col_j);
                j += width;
            }
        }
    }

    0
}

/*================================================================
  PART II - Backward Problems
  ================================================================*/

/*---------------------------------------------------------------
  User-Callable initialization function: wrapper for the backward
  phase around the corresponding CVODES functions
  ---------------------------------------------------------------*/
pub fn CVBandPrecInitB(
    cvode_mem: &CVodeMem,
    which: i32,
    nB: sunindextype,
    muB: sunindextype,
    mlB: sunindextype,
) -> i32 {
    /* Check if cvode_mem exists: handled by the type system */
    let cv_mem = cvode_mem;

    /* Was ASA initialized? */
    let adjMallocDone = cv_mem.borrow().cv_adjMallocDone;
    if adjMallocDone == SUNFALSE {
        cvProcessError(
            Some(cv_mem),
            CVLS_NO_ADJ,
            line!() as i32,
            "CVBandPrecInitB",
            file!(),
            MSGBP_NO_ADJ,
        );
        return CVLS_NO_ADJ;
    }
    let ca_mem = cv_mem.borrow().cv_adj_mem.clone().expect("cv_adj_mem");

    /* Check which */
    let nbckpbs = ca_mem.borrow().ca_nbckpbs;
    if which >= nbckpbs {
        cvProcessError(
            Some(cv_mem),
            CVLS_ILL_INPUT,
            line!() as i32,
            "CVBandPrecInitB",
            file!(),
            MSGBP_BAD_WHICH,
        );
        return CVLS_ILL_INPUT;
    }

    /* Find the CVodeBMem entry in the linked list corresponding to which
    (C list head = index 0; `for (p = head; p; p = p->next)` ≡ iter()) */
    let cvB_mem = {
        let ca = ca_mem.borrow();
        ca.cvB_mem
            .iter()
            .find(|cvB_mem| which == cvB_mem.borrow().cv_index)
            .cloned()
    };
    /* C walks off the end of the list and dereferences NULL when `which`
    is not present (UB → deterministic panic here) */
    let cvB_mem = cvB_mem.expect("cvB_mem for which");

    /* cv_mem corresponding to 'which' problem. */
    let cvodeB_mem = cvB_mem.borrow().cv_mem.clone().expect("cvB_mem->cv_mem");

    /* Set pfree */
    cvB_mem.borrow_mut().cv_pfree = None;

    /* Initialize the band preconditioner for this backward problem. */
    CVBandPrecInit(&cvodeB_mem, nB, muB, mlB)
}
