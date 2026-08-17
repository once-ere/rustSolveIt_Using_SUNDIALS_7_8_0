//! Port of `src/arkode/arkode_bandpre.c` (+ `src/arkode/arkode_bandpre_impl.h`
//! and `include/arkode/arkode_bandpre.h` folded).
//!
//! Banded difference-quotient Jacobian-based preconditioner and solver
//! routines for use with the ARKLS linear solver interface.
//!
//! The preconditioner data lives in `arkls_mem.P_data`
//! (`Option<Box<dyn Any>>` holding an [`ARKBandPrecDataRec`]); the ARKLS
//! interface (`arkode_ls`) `Option::take`s that box around each
//! psetup/psolve invocation, so the callbacks here receive it as
//! `&mut Option<Box<dyn Any>>` and downcast (never a clone — the very
//! same box is handed back on every path).
//!
//! `arkLs_AccessARKODELMem(arkode_mem, __func__, &ark_mem, &arkls_mem)`
//! translates, per the frozen seam, to a presence probe on
//! `ark_mem.ark_lmem` (`step_getlinmem` is a *presence* probe in the Rust
//! seam, and the ARKLS record itself lives in `ark_mem.ark_lmem`)
//! followed by `arkls_mem_mut(ark_mem)` at each use. Its NULL-`arkode_mem`
//! branch is handled by the type system. The probe is deliberately
//! non-panicking: it never calls `step_getlinmem` (which C dereferences
//! unconditionally).
//!
//! `arkAllocVec` / `arkFreeVec` are used exactly where C uses them, so
//! `ark_mem.lrw` / `ark_mem.liw` are incremented and decremented on the
//! same paths (including every allocation-failure unwind).
//!
//! Reference build: SUNDIALS_LOGGING_LEVEL = 2 (this file has no
//! SUNLogInfo/SUNLogDebug call sites), profiling OFF, error checks OFF.

use std::any::Any;

use crate::arkode::{arkAllocVec, arkFreeVec};
use crate::arkode_impl::*;
use crate::arkode_ls::{
    arkls_mem_mut, ARKLsMemRec, ARKodeSetPreconditioner, ARKLS_ILL_INPUT, ARKLS_LMEM_NULL,
    ARKLS_MEM_FAIL, ARKLS_PMEM_NULL, ARKLS_SUCCESS, ARKLS_SUNLS_FAIL, MSG_LS_LMEM_NULL,
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
    N_VGetArrayPointer, N_VScale, N_VSpace, N_VWrmsNorm, N_Vector,
};
use sundials_core::sundials_types::*;
use sundials_core::sunlinsol_band::{SUNLinSolSetup_Band, SUNLinSol_Band};
use sundials_core::sunmatrix_band::{
    SM_COLUMN_ELEMENT_IDX, SM_SUBAND_B, SUNBandMatrixStorage, SUNBandMatrix_Column,
};

/* `ZERO` and `ONE` are `#define`d in `arkode_bandpre.c` with values
identical to the ones `arkode_impl.h` already provides, so the contract
copies are used here; `TWO` comes from `arkode_impl.h` in the C as well. */
const MIN_INC_MULT: sunrealtype = 1000.0;

/*---------------------------------------------------------------
 Type: ARKBandPrecData (arkode_bandpre_impl.h)
---------------------------------------------------------------*/

pub struct ARKBandPrecDataRec {
    /* Data set by user in ARKBandPrecInit */
    pub N: sunindextype,
    pub ml: sunindextype,
    pub mu: sunindextype,

    /* Data set by ARKBandPrecSetup */
    pub savedJ: SUNMatrix,
    pub savedP: SUNMatrix,
    pub LS: SUNLinearSolver,
    pub tmp1: Option<N_Vector>,
    pub tmp2: Option<N_Vector>,

    /* Rhs calls */
    pub nfeBP: i64,

    /* Pointer to arkode_mem */
    pub arkode_mem: ARKodeMem,
}

pub type ARKBandPrecData = Box<ARKBandPrecDataRec>;

/*---------------------------------------------------------------
 ARKBANDPRE error messages (arkode_bandpre_impl.h)
---------------------------------------------------------------*/

pub const MSG_BP_MEM_NULL: &str = "Integrator memory is NULL.";
pub const MSG_BP_LMEM_NULL: &str =
    "Linear solver memory is NULL. The SPILS interface must be attached.";
pub const MSG_BP_MEM_FAIL: &str = "A memory request failed.";
pub const MSG_BP_BAD_NVECTOR: &str = "A required vector operation is not implemented.";
pub const MSG_BP_SUNMAT_FAIL: &str = "An error arose from a SUNBandMatrix routine.";
pub const MSG_BP_SUNLS_FAIL: &str = "An error arose from a SUNBandLinearSolver routine.";
pub const MSG_BP_PMEM_NULL: &str =
    "Band preconditioner memory is NULL. ARKBandPrecInit must be called.";
pub const MSG_BP_RHSFUNC_FAILED: &str =
    "The right-hand side routine failed in an unrecoverable manner.";

/*---------------------------------------------------------------
 Initialization, Free, and Get Functions
 NOTE: The band linear solver assumes a serial implementation
       of the NVECTOR package. Therefore, ARKBandPrecInit will
       first test for a compatible N_Vector internal
       representation by checking that the function
       N_VGetArrayPointer exists.
---------------------------------------------------------------*/
pub fn ARKBandPrecInit(
    arkode_mem: &ARKodeMem,
    N: sunindextype,
    mu: sunindextype,
    ml: sunindextype,
) -> i32 {
    /* access ARKodeMem and ARKLsMem structures */
    let ark_mem = arkode_mem;
    let attached = {
        let mem = ark_mem.borrow();
        mem.ark_lmem.as_ref().is_some_and(|b| b.is::<ARKLsMemRec>())
    };
    if !attached {
        arkProcessError(
            Some(ark_mem),
            ARKLS_LMEM_NULL,
            line!() as i32,
            "ARKBandPrecInit",
            file!(),
            MSG_LS_LMEM_NULL,
        );
        return ARKLS_LMEM_NULL;
    }

    /* Test compatibility of NVECTOR package with the BAND preconditioner */
    let (tempv1, sunctx) = {
        let mem = ark_mem.borrow();
        (mem.tempv1.clone().expect("tempv1"), mem.sunctx.clone())
    };
    if tempv1.ops.borrow().nvgetarraypointer.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARKLS_ILL_INPUT,
            line!() as i32,
            "ARKBandPrecInit",
            file!(),
            MSG_BP_BAD_NVECTOR,
        );
        return ARKLS_ILL_INPUT;
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
            arkProcessError(
                Some(ark_mem),
                ARKLS_MEM_FAIL,
                line!() as i32,
                "ARKBandPrecInit",
                file!(),
                MSG_BP_MEM_FAIL,
            );
            return ARKLS_MEM_FAIL;
        }
        Some(m) => m,
    };

    /* Allocate memory for banded preconditioner. */
    let storagemu = SUNMIN(N - 1, mup + mlp);
    let savedP = match SUNBandMatrixStorage(N, mup, mlp, storagemu, &sunctx) {
        None => {
            arkProcessError(
                Some(ark_mem),
                ARKLS_MEM_FAIL,
                line!() as i32,
                "ARKBandPrecInit",
                file!(),
                MSG_BP_MEM_FAIL,
            );
            return ARKLS_MEM_FAIL;
        }
        Some(m) => m,
    };

    /* Allocate memory for banded linear solver */
    let LS = match SUNLinSol_Band(&tempv1, &savedP, &sunctx) {
        None => {
            arkProcessError(
                Some(ark_mem),
                ARKLS_MEM_FAIL,
                line!() as i32,
                "ARKBandPrecInit",
                file!(),
                MSG_BP_MEM_FAIL,
            );
            return ARKLS_MEM_FAIL;
        }
        Some(s) => s,
    };

    /* allocate memory for temporary N_Vectors */
    let mut tmp1: Option<N_Vector> = None;
    if !arkAllocVec(ark_mem, &tempv1, &mut tmp1) {
        arkProcessError(
            Some(ark_mem),
            ARKLS_MEM_FAIL,
            line!() as i32,
            "ARKBandPrecInit",
            file!(),
            MSG_BP_MEM_FAIL,
        );
        return ARKLS_MEM_FAIL;
    }

    let mut tmp2: Option<N_Vector> = None;
    if !arkAllocVec(ark_mem, &tempv1, &mut tmp2) {
        arkFreeVec(ark_mem, &mut tmp1);
        arkProcessError(
            Some(ark_mem),
            ARKLS_MEM_FAIL,
            line!() as i32,
            "ARKBandPrecInit",
            file!(),
            MSG_BP_MEM_FAIL,
        );
        return ARKLS_MEM_FAIL;
    }

    /* initialize band linear solver object */
    let retval = SUNLinSolInitialize(&LS);
    if retval != SUN_SUCCESS {
        arkFreeVec(ark_mem, &mut tmp1);
        arkFreeVec(ark_mem, &mut tmp2);
        arkProcessError(
            Some(ark_mem),
            ARKLS_SUNLS_FAIL,
            line!() as i32,
            "ARKBandPrecInit",
            file!(),
            MSG_BP_SUNLS_FAIL,
        );
        return ARKLS_SUNLS_FAIL;
    }

    /* make sure s_P_data is free from any previous allocations */
    let pfree = arkls_mem_mut(ark_mem).pfree;
    if let Some(pfree) = pfree {
        pfree(ark_mem);
    }

    {
        let mut arkls_mem = arkls_mem_mut(ark_mem);

        /* Point to the new P_data field in the LS memory */
        arkls_mem.P_data = Some(Box::new(ARKBandPrecDataRec {
            N,
            ml: mlp,
            mu: mup,
            savedJ,
            savedP,
            LS,
            tmp1,
            tmp2,
            nfeBP,
            arkode_mem: ark_mem.clone(),
        }));

        /* Attach the pfree function */
        arkls_mem.pfree = Some(ARKBandPrecFree);
    }

    /* Attach preconditioner solve and setup functions */
    ARKodeSetPreconditioner(arkode_mem, Some(ARKBandPrecSetup), Some(ARKBandPrecSolve))
}

pub fn ARKBandPrecGetWorkSpace(
    arkode_mem: &ARKodeMem,
    lenrwBP: &mut i64,
    leniwBP: &mut i64,
) -> i32 {
    /* access ARKodeMem and ARKLsMem structures */
    let ark_mem = arkode_mem;
    let attached = {
        let mem = ark_mem.borrow();
        mem.ark_lmem.as_ref().is_some_and(|b| b.is::<ARKLsMemRec>())
    };
    if !attached {
        arkProcessError(
            Some(ark_mem),
            ARKLS_LMEM_NULL,
            line!() as i32,
            "ARKBandPrecGetWorkSpace",
            file!(),
            MSG_LS_LMEM_NULL,
        );
        return ARKLS_LMEM_NULL;
    }

    /* Return immediately if ARKBandPrecData is NULL */
    let handles = {
        let arkls_mem = arkls_mem_mut(ark_mem);
        arkls_mem
            .P_data
            .as_ref()
            .and_then(|b| b.downcast_ref::<ARKBandPrecDataRec>())
            .map(|p| (p.savedJ.clone(), p.savedP.clone(), p.LS.clone()))
    };
    let (savedJ, savedP, LS) = match handles {
        None => {
            arkProcessError(
                Some(ark_mem),
                ARKLS_PMEM_NULL,
                line!() as i32,
                "ARKBandPrecGetWorkSpace",
                file!(),
                MSG_BP_PMEM_NULL,
            );
            return ARKLS_PMEM_NULL;
        }
        Some(h) => h,
    };

    /* sum space requirements for all objects in pdata */
    *leniwBP = 4;
    *lenrwBP = 0;
    let tempv1 = ark_mem.borrow().tempv1.clone().expect("tempv1");
    if tempv1.ops.borrow().nvspace.is_some() {
        let mut lrw1: sunindextype = 0;
        let mut liw1: sunindextype = 0;
        N_VSpace(&tempv1, &mut lrw1, &mut liw1);
        *leniwBP += 2 * liw1;
        *lenrwBP += 2 * lrw1;
    }
    if savedJ.ops.borrow().space.is_some() {
        let mut lrw: i64 = 0;
        let mut liw: i64 = 0;
        let retval = SUNMatSpace(&savedJ, &mut lrw, &mut liw);
        if retval == 0 {
            *leniwBP += liw;
            *lenrwBP += lrw;
        }
    }
    if savedP.ops.borrow().space.is_some() {
        let mut lrw: i64 = 0;
        let mut liw: i64 = 0;
        let retval = SUNMatSpace(&savedP, &mut lrw, &mut liw);
        if retval == 0 {
            *leniwBP += liw;
            *lenrwBP += lrw;
        }
    }
    if LS.ops.borrow().space.is_some() {
        let mut lrw: i64 = 0;
        let mut liw: i64 = 0;
        let retval = SUNLinSolSpace(&LS, &mut lrw, &mut liw);
        if retval == SUN_SUCCESS {
            *leniwBP += liw;
            *lenrwBP += lrw;
        }
    }

    ARKLS_SUCCESS
}

pub fn ARKBandPrecGetNumRhsEvals(arkode_mem: &ARKodeMem, nfevalsBP: &mut i64) -> i32 {
    /* access ARKodeMem and ARKLsMem structures */
    let ark_mem = arkode_mem;
    let attached = {
        let mem = ark_mem.borrow();
        mem.ark_lmem.as_ref().is_some_and(|b| b.is::<ARKLsMemRec>())
    };
    if !attached {
        arkProcessError(
            Some(ark_mem),
            ARKLS_LMEM_NULL,
            line!() as i32,
            "ARKBandPrecGetNumRhsEvals",
            file!(),
            MSG_LS_LMEM_NULL,
        );
        return ARKLS_LMEM_NULL;
    }

    /* Return immediately if ARKBandPrecData is NULL */
    let nfeBP = {
        let arkls_mem = arkls_mem_mut(ark_mem);
        arkls_mem
            .P_data
            .as_ref()
            .and_then(|b| b.downcast_ref::<ARKBandPrecDataRec>())
            .map(|p| p.nfeBP)
    };
    match nfeBP {
        None => {
            arkProcessError(
                Some(ark_mem),
                ARKLS_PMEM_NULL,
                line!() as i32,
                "ARKBandPrecGetNumRhsEvals",
                file!(),
                MSG_BP_PMEM_NULL,
            );
            ARKLS_PMEM_NULL
        }
        Some(nfeBP) => {
            /* set output */
            *nfevalsBP = nfeBP;
            ARKLS_SUCCESS
        }
    }
}

/*---------------------------------------------------------------
 ARKBandPrecSetup:

 Together ARKBandPrecSetup and ARKBandPrecSolve use a banded
 difference quotient Jacobian to create a preconditioner.
 ARKBandPrecSetup calculates a new J, if necessary, then
 calculates P = I - gamma*J, and does an LU factorization of P.

 The parameters of ARKBandPrecSetup are as follows:

 t       is the current value of the independent variable.

 y       is the current value of the dependent variable vector,
         namely the predicted value of y(t).

 fy      is the vector f(t,y).

 jok     is an input flag indicating whether Jacobian-related
         data needs to be recomputed, as follows:
           jok == SUNFALSE means recompute Jacobian-related data
                  from scratch.
           jok == SUNTRUE means that Jacobian data from the
                  previous PrecSetup call will be reused
                  (with the current value of gamma).
         A ARKBandPrecSetup call with jok == SUNTRUE should only
         occur after a call with jok == SUNFALSE.

 *jcurPtr is a pointer to an output integer flag which is
          set by ARKBandPrecond as follows:
            *jcurPtr = SUNTRUE if Jacobian data was recomputed.
            *jcurPtr = SUNFALSE if Jacobian data was not recomputed,
                       but saved data was reused.

 gamma   is the scalar appearing in the Newton matrix.

 bp_data is a pointer to preconditioner data (set by ARKBandPrecInit)

 The value to be returned by the ARKBandPrecSetup function is
   0  if successful, or
   1  if the band factorization failed.
---------------------------------------------------------------*/
fn ARKBandPrecSetup(
    t: sunrealtype,
    y: &N_Vector,
    fy: &N_Vector,
    jok: sunbooleantype,
    jcurPtr: &mut sunbooleantype,
    gamma: sunrealtype,
    bp_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    /* Assume matrix and lpivots have already been allocated. */
    let pdata: &mut ARKBandPrecDataRec = bp_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<ARKBandPrecDataRec>())
        .expect("bp_data is ARKBandPrecData");

    let ark_mem = pdata.arkode_mem.clone();

    if jok {
        /* If jok = SUNTRUE, use saved copy of J. */
        *jcurPtr = SUNFALSE;
        let retval = SUNMatCopy(&pdata.savedJ, &pdata.savedP);
        if retval < 0 {
            arkProcessError(
                Some(&ark_mem),
                -1,
                line!() as i32,
                "ARKBandPrecSetup",
                file!(),
                MSG_BP_SUNMAT_FAIL,
            );
            return -1;
        }
        if retval > 0 {
            return 1;
        }
    } else {
        /* If jok = SUNFALSE, call ARKBandPDQJac for new J value. */
        *jcurPtr = SUNTRUE;
        let retval = SUNMatZero(&pdata.savedJ);
        if retval < 0 {
            arkProcessError(
                Some(&ark_mem),
                -1,
                line!() as i32,
                "ARKBandPrecSetup",
                file!(),
                MSG_BP_SUNMAT_FAIL,
            );
            return -1;
        }
        if retval > 0 {
            return 1;
        }

        let tmp1 = pdata.tmp1.clone().expect("tmp1");
        let tmp2 = pdata.tmp2.clone().expect("tmp2");
        let retval = ARKBandPDQJac(pdata, t, y, fy, &tmp1, &tmp2);
        if retval < 0 {
            arkProcessError(
                Some(&ark_mem),
                -1,
                line!() as i32,
                "ARKBandPrecSetup",
                file!(),
                MSG_BP_RHSFUNC_FAILED,
            );
            return -1;
        }
        if retval > 0 {
            return 1;
        }

        let retval = SUNMatCopy(&pdata.savedJ, &pdata.savedP);
        if retval < 0 {
            arkProcessError(
                Some(&ark_mem),
                -1,
                line!() as i32,
                "ARKBandPrecSetup",
                file!(),
                MSG_BP_SUNMAT_FAIL,
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
        arkProcessError(
            Some(&ark_mem),
            -1,
            line!() as i32,
            "ARKBandPrecSetup",
            file!(),
            MSG_BP_SUNMAT_FAIL,
        );
        return -1;
    }

    /* Do LU factorization of matrix and return error flag */
    SUNLinSolSetup_Band(&pdata.LS, Some(&pdata.savedP))
}

/*---------------------------------------------------------------
 ARKBandPrecSolve:

 ARKBandPrecSolve solves a linear system P z = r, where P is the
 matrix computed by ARKBandPrecond.

 The parameters of ARKBandPrecSolve used here are as follows:

 r is the right-hand side vector of the linear system.

 bp_data is a pointer to preconditioner data (set by ARKBandPrecInit)

 z is the output vector computed by ARKBandPrecSolve.

 The value returned by the ARKBandPrecSolve function is always 0,
 indicating success.
---------------------------------------------------------------*/
fn ARKBandPrecSolve(
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
    /* Assume matrix and linear solver have already been allocated. */
    let pdata: &ARKBandPrecDataRec = bp_data
        .as_ref()
        .and_then(|b| b.downcast_ref::<ARKBandPrecDataRec>())
        .expect("bp_data is ARKBandPrecData");

    /* Call banded solver object to do the work */
    SUNLinSolSolve(&pdata.LS, Some(&pdata.savedP), z, r, ZERO)
}

/*---------------------------------------------------------------
 ARKBandPrecFree:

 Frees data associated with the ARKBand preconditioner.
---------------------------------------------------------------*/
fn ARKBandPrecFree(ark_mem: &ARKodeMem) -> i32 {
    /* Return immediately if ARKodeMem, ARKLsMem or ARKBandPrecData are
    NULL (C probes `ark_mem->step_getlinmem((void*)ark_mem)`; the Rust
    seam makes that a presence probe and the record itself lives in
    `ark_mem.ark_lmem`, so probe that directly -- never panicking) */
    let attached = {
        let mem = ark_mem.borrow();
        mem.ark_lmem.as_ref().is_some_and(|b| b.is::<ARKLsMemRec>())
    };
    if !attached {
        return 0;
    }

    let pdata = arkls_mem_mut(ark_mem).P_data.take();
    let pdata = match pdata {
        None => return 0,
        Some(p) => p,
    };
    let mut pdata = match pdata.downcast::<ARKBandPrecDataRec>() {
        Ok(p) => p,
        Err(other) => {
            /* `P_data` is no longer this module's record (`ARKodeSetUserData`
            overwrites it with the user's data pointer); hand it back
            untouched instead of freeing it as C would. */
            let mut arkls_mem = arkls_mem_mut(ark_mem);
            arkls_mem.P_data = Some(other);
            return 0;
        }
    };

    /* SUNLinSolFree(LS) / SUNMatDestroy(savedP, savedJ): dropping the
    record releases both matrices and the linear solver. tmp1 and tmp2 go
    through arkFreeVec exactly as in C so that ark_mem->lrw / ark_mem->liw
    are decremented (C frees each explicitly and leaves arkls_mem->P_data
    dangling; the Rust take() leaves it None) */
    arkFreeVec(ark_mem, &mut pdata.tmp1);
    arkFreeVec(ark_mem, &mut pdata.tmp2);

    drop(pdata);

    0
}

/*---------------------------------------------------------------
 ARKBandPDQJac:

 This routine generates a banded difference quotient approximation to
 the Jacobian of f(t,y). It assumes that a band matrix of type
 SUNDlsMat is stored column-wise, and that elements within each column
 are contiguous. This makes it possible to get the address of a column
 of J via the macro SUNDLS_BAND_COL and to write a simple for loop to set
 each of the elements of a column in succession.
---------------------------------------------------------------*/
fn ARKBandPDQJac(
    pdata: &mut ARKBandPrecDataRec,
    t: sunrealtype,
    y: &N_Vector,
    fy: &N_Vector,
    ftemp: &N_Vector,
    ytemp: &N_Vector,
) -> i32 {
    let ark_mem = pdata.arkode_mem.clone();

    /* Access implicit RHS function (C dereferences step_getimplicitrhs
    unconditionally; a missing probe is treated exactly like the NULL
    `fi` it would report -- accepted deviation class 1) */
    let fi = {
        let step_getimplicitrhs = ark_mem.borrow().step_getimplicitrhs;
        match step_getimplicitrhs {
            None => None,
            Some(step_getimplicitrhs) => step_getimplicitrhs(&ark_mem),
        }
    };
    let fi = match fi {
        None => return -1,
        Some(fi) => fi,
    };

    /* Copy the fields C reads through ark_mem-> out of the mem
    (granular borrow: nothing is held across callbacks/vector ops) */
    let (uround, h, ewt, rwt, constraints) = {
        let mem = ark_mem.borrow();
        (
            mem.uround,
            mem.h,
            mem.ewt.clone().expect("ewt"),
            mem.rwt.clone().expect("rwt"),
            mem.constraints.clone(),
        )
    };

    /* Load ytemp with y = predicted y vector. */
    N_VScale(ONE, y, ytemp);

    /* Set minimum increment based on uround and norm of f. */
    let srur = SUNRsqrt(uround);
    let fnorm = N_VWrmsNorm(fy, &rwt);
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

        /* call the user-supplied pre-RHS function (if supplied), then call RHS */
        let PreRhsFn = ark_mem.borrow().PreRhsFn;
        if let Some(PreRhsFn) = PreRhsFn {
            let retval = {
                let mut user_data = ark_mem.borrow_mut().user_data.take();
                let retval = PreRhsFn(t, ytemp, &mut user_data);
                ark_mem.borrow_mut().user_data = user_data;
                retval
            };
            if retval != 0 {
                return ARK_PRERHSFN_FAIL;
            }
        }
        let retval = {
            let mut user_data = ark_mem.borrow_mut().user_data.take();
            let retval = fi(t, ytemp, ftemp, &mut user_data);
            ark_mem.borrow_mut().user_data = user_data;
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
