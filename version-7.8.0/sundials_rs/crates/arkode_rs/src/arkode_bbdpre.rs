//! Port of `src/arkode/arkode_bbdpre.c` (+ `src/arkode/arkode_bbdpre_impl.h`
//! and `include/arkode/arkode_bbdpre.h` folded).
//!
//! Band-block-diagonal preconditioner (a block-diagonal matrix with
//! banded blocks) for use with ARKODE and the ARKLS linear solver
//! interface. The upstream file is written against the MPI-parallel
//! NVECTOR; this port is the serial build (the file itself has no MPI
//! `#ifdef`s — it wraps the local data in `N_VNewEmpty_Serial` vectors
//! exactly as the C does).
//!
//! The preconditioner data lives in `arkls_mem.P_data`
//! (`Option<Box<dyn Any>>` holding an [`ARKBBDPrecDataRec`]); the ARKLS
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
use std::rc::Rc;

use crate::arkode::{arkAllocVec, arkFreeVec};
use crate::arkode_impl::*;
use crate::arkode_ls::{
    arkls_mem_mut, ARKLsMemRec, ARKodeSetPreconditioner, ARKLS_ILL_INPUT, ARKLS_LMEM_NULL,
    ARKLS_MEM_FAIL, ARKLS_PMEM_NULL, ARKLS_SUCCESS, MSG_LS_LMEM_NULL,
};
use sundials_core::nvector_serial::N_VNewEmpty_Serial;
use sundials_core::sundials_linearsolver::{
    SUNLinSolInitialize, SUNLinSolSolve, SUNLinSolSpace, SUNLinearSolver,
};
use sundials_core::sundials_math::{SUNMAX, SUNMIN, SUNRabs, SUNRsqrt};
use sundials_core::sundials_matrix::{
    SUNMatCopy, SUNMatScaleAddI, SUNMatSpace, SUNMatZero, SUNMatrix,
};
use sundials_core::sundials_nvector::{
    N_VGetArrayPointer, N_VScale, N_VSetArrayPointer, N_VSpace, N_VWrmsNorm, N_Vector,
};
use sundials_core::sundials_types::*;
use sundials_core::sunlinsol_band::{SUNLinSolSetup_Band, SUNLinSol_Band};
use sundials_core::sunmatrix_band::{
    SM_COLUMN_ELEMENT_IDX, SM_SUBAND_B, SUNBandMatrixStorage, SUNBandMatrix_Column,
};

/* `ZERO` and `ONE` are `#define`d in `arkode_bbdpre.c` with values
identical to the ones `arkode_impl.h` already provides, so the contract
copies are used here; `TWO` comes from `arkode_impl.h` in the C as well. */
const MIN_INC_MULT: sunrealtype = 1000.0;

/*---------------------------------------------------------------
 User-supplied function types (include/arkode/arkode_bbdpre.h)
---------------------------------------------------------------*/

pub type ARKLocalFn = fn(
    Nlocal: sunindextype,
    t: sunrealtype,
    y: &N_Vector,
    g: &N_Vector,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32;

pub type ARKCommFn = fn(
    Nlocal: sunindextype,
    t: sunrealtype,
    y: &N_Vector,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32;

/*---------------------------------------------------------------
 Type: ARKBBDPrecData (arkode_bbdpre_impl.h)
---------------------------------------------------------------*/

pub struct ARKBBDPrecDataRec {
    /* passed by user to ARKBBDPrecAlloc and used by PrecSetup/PrecSolve */
    pub mudq: sunindextype,
    pub mldq: sunindextype,
    pub mukeep: sunindextype,
    pub mlkeep: sunindextype,
    pub dqrely: sunrealtype,
    pub gloc: ARKLocalFn,
    pub cfn: Option<ARKCommFn>,

    /* set by ARKBBDPrecSetup and used by ARKBBDPrecSolve */
    pub savedJ: SUNMatrix,
    pub savedP: SUNMatrix,
    pub LS: SUNLinearSolver,
    pub tmp1: Option<N_Vector>,
    pub tmp2: Option<N_Vector>,
    pub tmp3: Option<N_Vector>,
    pub zlocal: N_Vector,
    pub rlocal: N_Vector,

    /* set by ARKBBDPrecAlloc and used by ARKBBDPrecSetup */
    pub n_local: sunindextype,

    /* available for optional output */
    pub rpwsize: i64,
    pub ipwsize: i64,
    pub nge: i64,

    /* pointer to arkode_mem */
    pub arkode_mem: ARKodeMem,
}

pub type ARKBBDPrecData = Box<ARKBBDPrecDataRec>;

/*---------------------------------------------------------------
 ARKBBDPRE error messages (arkode_bbdpre_impl.h)
---------------------------------------------------------------*/

pub const MSG_BBD_MEM_NULL: &str = "Integrator memory is NULL.";
pub const MSG_BBD_LMEM_NULL: &str =
    "Linear solver memory is NULL. One of the SPILS linear solvers must be attached.";
pub const MSG_BBD_MEM_FAIL: &str = "A memory request failed.";
pub const MSG_BBD_BAD_NVECTOR: &str = "A required vector operation is not implemented.";
pub const MSG_BBD_SUNMAT_FAIL: &str = "An error arose from a SUNBandMatrix routine.";
pub const MSG_BBD_SUNLS_FAIL: &str = "An error arose from a SUNBandLinearSolver routine.";
pub const MSG_BBD_PMEM_NULL: &str =
    "BBD peconditioner memory is NULL. ARKBBDPrecInit must be called.";
pub const MSG_BBD_FUNC_FAILED: &str =
    "The gloc or cfn routine failed in an unrecoverable manner.";

/*---------------------------------------------------------------
 User-Callable Functions: initialization, reinit and free
---------------------------------------------------------------*/
pub fn ARKBBDPrecInit(
    arkode_mem: &ARKodeMem,
    Nlocal: sunindextype,
    mudq: sunindextype,
    mldq: sunindextype,
    mukeep: sunindextype,
    mlkeep: sunindextype,
    dqrely: sunrealtype,
    gloc: ARKLocalFn,
    cfn: Option<ARKCommFn>,
) -> i32 {
    /* access ARKodeMem and ARKLsMem structure */
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
            "ARKBBDPrecInit",
            file!(),
            MSG_LS_LMEM_NULL,
        );
        return ARKLS_LMEM_NULL;
    }

    /* Test compatibility of NVECTOR package with the BBD preconditioner */
    let (tempv1, sunctx) = {
        let mem = ark_mem.borrow();
        (mem.tempv1.clone().expect("tempv1"), mem.sunctx.clone())
    };
    let bad_nvector = {
        let ops = tempv1.ops.borrow();
        ops.nvgetarraypointer.is_none() || ops.nvsetarraypointer.is_none()
    };
    if bad_nvector {
        arkProcessError(
            Some(ark_mem),
            ARKLS_ILL_INPUT,
            line!() as i32,
            "ARKBBDPrecInit",
            file!(),
            MSG_BBD_BAD_NVECTOR,
        );
        return ARKLS_ILL_INPUT;
    }

    /* Allocate data memory (Rust: the record is assembled at the end of
    this function; the C malloc-NULL branch has no analogue) */

    /* Set pointers to gloc and cfn; load half-bandwidths */
    let mudq = SUNMIN(Nlocal - 1, SUNMAX(0, mudq));
    let mldq = SUNMIN(Nlocal - 1, SUNMAX(0, mldq));
    let muk = SUNMIN(Nlocal - 1, SUNMAX(0, mukeep));
    let mlk = SUNMIN(Nlocal - 1, SUNMAX(0, mlkeep));

    /* Allocate memory for saved Jacobian */
    let savedJ = match SUNBandMatrixStorage(Nlocal, muk, mlk, muk, &sunctx) {
        None => {
            arkProcessError(
                Some(ark_mem),
                ARKLS_MEM_FAIL,
                line!() as i32,
                "ARKBBDPrecInit",
                file!(),
                MSG_BBD_MEM_FAIL,
            );
            return ARKLS_MEM_FAIL;
        }
        Some(m) => m,
    };

    /* Allocate memory for preconditioner matrix */
    let storage_mu = SUNMIN(Nlocal - 1, muk + mlk);
    let savedP = match SUNBandMatrixStorage(Nlocal, muk, mlk, storage_mu, &sunctx) {
        None => {
            arkProcessError(
                Some(ark_mem),
                ARKLS_MEM_FAIL,
                line!() as i32,
                "ARKBBDPrecInit",
                file!(),
                MSG_BBD_MEM_FAIL,
            );
            return ARKLS_MEM_FAIL;
        }
        Some(m) => m,
    };

    /* Allocate memory for temporary N_Vectors */

    let zlocal = match N_VNewEmpty_Serial(Nlocal, &sunctx) {
        None => {
            arkProcessError(
                Some(ark_mem),
                ARKLS_MEM_FAIL,
                line!() as i32,
                "ARKBBDPrecInit",
                file!(),
                MSG_BBD_MEM_FAIL,
            );
            return ARKLS_MEM_FAIL;
        }
        Some(v) => v,
    };

    let rlocal = match N_VNewEmpty_Serial(Nlocal, &sunctx) {
        None => {
            arkProcessError(
                Some(ark_mem),
                ARKLS_MEM_FAIL,
                line!() as i32,
                "ARKBBDPrecInit",
                file!(),
                MSG_BBD_MEM_FAIL,
            );
            return ARKLS_MEM_FAIL;
        }
        Some(v) => v,
    };

    let mut tmp1: Option<N_Vector> = None;
    if !arkAllocVec(ark_mem, &tempv1, &mut tmp1) {
        arkProcessError(
            Some(ark_mem),
            ARKLS_MEM_FAIL,
            line!() as i32,
            "ARKBBDPrecInit",
            file!(),
            MSG_BBD_MEM_FAIL,
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
            "ARKBBDPrecInit",
            file!(),
            MSG_BBD_MEM_FAIL,
        );
        return ARKLS_MEM_FAIL;
    }

    let mut tmp3: Option<N_Vector> = None;
    if !arkAllocVec(ark_mem, &tempv1, &mut tmp3) {
        arkFreeVec(ark_mem, &mut tmp1);
        arkFreeVec(ark_mem, &mut tmp2);
        arkProcessError(
            Some(ark_mem),
            ARKLS_MEM_FAIL,
            line!() as i32,
            "ARKBBDPrecInit",
            file!(),
            MSG_BBD_MEM_FAIL,
        );
        return ARKLS_MEM_FAIL;
    }

    /* Allocate memory for banded linear solver */
    let LS = match SUNLinSol_Band(&rlocal, &savedP, &sunctx) {
        None => {
            arkFreeVec(ark_mem, &mut tmp1);
            arkFreeVec(ark_mem, &mut tmp2);
            arkFreeVec(ark_mem, &mut tmp3);
            arkProcessError(
                Some(ark_mem),
                ARKLS_MEM_FAIL,
                line!() as i32,
                "ARKBBDPrecInit",
                file!(),
                MSG_BBD_MEM_FAIL,
            );
            return ARKLS_MEM_FAIL;
        }
        Some(s) => s,
    };

    /* initialize band linear solver object
    (upstream guards the failure branch with `if (pdata->LS == NULL)`
    rather than `if (retval != SUN_SUCCESS)`; `LS` was just proven
    non-NULL immediately above, so that branch -- which would free
    everything and return ARKLS_SUNLS_FAIL with MSG_BBD_SUNLS_FAIL -- is
    dead code in C and is therefore not reproduced. The call itself is
    made, in the same place, for its side effects.) */
    let _ = SUNLinSolInitialize(&LS);

    /* Set dqrely based on input dqrely (0 implies default). */
    let dqrely = if dqrely > ZERO {
        dqrely
    } else {
        SUNRsqrt(ark_mem.borrow().uround)
    };

    /* Store Nlocal to be used in ARKBBDPrecSetup */
    let n_local = Nlocal;

    /* Set work space sizes and initialize nge */
    let mut rpwsize: i64 = 0;
    let mut ipwsize: i64 = 0;
    if tempv1.ops.borrow().nvspace.is_some() {
        let mut lrw1: sunindextype = 0;
        let mut liw1: sunindextype = 0;
        N_VSpace(&tempv1, &mut lrw1, &mut liw1);
        rpwsize += 3 * lrw1;
        ipwsize += 3 * liw1;
    }
    if rlocal.ops.borrow().nvspace.is_some() {
        let mut lrw1: sunindextype = 0;
        let mut liw1: sunindextype = 0;
        N_VSpace(&rlocal, &mut lrw1, &mut liw1);
        rpwsize += 2 * lrw1;
        ipwsize += 2 * liw1;
    }
    if savedJ.ops.borrow().space.is_some() {
        let mut lrw: i64 = 0;
        let mut liw: i64 = 0;
        let _ = SUNMatSpace(&savedJ, &mut lrw, &mut liw);
        rpwsize += lrw;
        ipwsize += liw;
    }
    if savedP.ops.borrow().space.is_some() {
        let mut lrw: i64 = 0;
        let mut liw: i64 = 0;
        let _ = SUNMatSpace(&savedP, &mut lrw, &mut liw);
        rpwsize += lrw;
        ipwsize += liw;
    }
    if LS.ops.borrow().space.is_some() {
        let mut lrw: i64 = 0;
        let mut liw: i64 = 0;
        let _ = SUNLinSolSpace(&LS, &mut lrw, &mut liw);
        rpwsize += lrw;
        ipwsize += liw;
    }
    let nge: i64 = 0;

    /* make sure P_data is free from any previous allocations */
    let pfree = arkls_mem_mut(ark_mem).pfree;
    if let Some(pfree) = pfree {
        pfree(ark_mem);
    }

    {
        let mut arkls_mem = arkls_mem_mut(ark_mem);

        /* Point to the new P_data field in the LS memory */
        arkls_mem.P_data = Some(Box::new(ARKBBDPrecDataRec {
            mudq,
            mldq,
            mukeep: muk,
            mlkeep: mlk,
            dqrely,
            gloc,
            cfn,
            savedJ,
            savedP,
            LS,
            tmp1,
            tmp2,
            tmp3,
            zlocal,
            rlocal,
            n_local,
            rpwsize,
            ipwsize,
            nge,
            arkode_mem: ark_mem.clone(),
        }));

        /* Attach the pfree function */
        arkls_mem.pfree = Some(ARKBBDPrecFree);
    }

    /* Attach preconditioner solve and setup functions */
    ARKodeSetPreconditioner(arkode_mem, Some(ARKBBDPrecSetup), Some(ARKBBDPrecSolve))
}

/*-------------------------------------------------------------*/
pub fn ARKBBDPrecReInit(
    arkode_mem: &ARKodeMem,
    mudq: sunindextype,
    mldq: sunindextype,
    dqrely: sunrealtype,
) -> i32 {
    /* access ARKodeMem and ARKLsMem structure */
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
            "ARKBBDPrecReInit",
            file!(),
            MSG_LS_LMEM_NULL,
        );
        return ARKLS_LMEM_NULL;
    }

    /* Return immediately ARKBBDPrecData is NULL */
    let has_pdata = {
        let arkls_mem = arkls_mem_mut(ark_mem);
        arkls_mem
            .P_data
            .as_ref()
            .and_then(|b| b.downcast_ref::<ARKBBDPrecDataRec>())
            .is_some()
    };
    if !has_pdata {
        arkProcessError(
            Some(ark_mem),
            ARKLS_PMEM_NULL,
            line!() as i32,
            "ARKBBDPrecReInit",
            file!(),
            MSG_BBD_PMEM_NULL,
        );
        return ARKLS_PMEM_NULL;
    }

    /* Set dqrely based on input dqrely (0 implies default).
    (Rust: `uround` is read before taking the ARKLS guard -- that guard
    is an ark_mem borrow_mut, so the read cannot happen inside it; the
    read has no side effects, so C-observable order is unchanged.) */
    let dqrely = if dqrely > ZERO {
        dqrely
    } else {
        SUNRsqrt(ark_mem.borrow().uround)
    };

    {
        let mut arkls_mem = arkls_mem_mut(ark_mem);
        let pdata = arkls_mem
            .P_data
            .as_mut()
            .and_then(|b| b.downcast_mut::<ARKBBDPrecDataRec>())
            .expect("P_data is ARKBBDPrecData");

        /* Load half-bandwidths */
        let Nlocal = pdata.n_local;
        pdata.mudq = SUNMIN(Nlocal - 1, SUNMAX(0, mudq));
        pdata.mldq = SUNMIN(Nlocal - 1, SUNMAX(0, mldq));

        pdata.dqrely = dqrely;

        /* Re-initialize nge */
        pdata.nge = 0;
    }

    ARKLS_SUCCESS
}

/*-------------------------------------------------------------*/
pub fn ARKBBDPrecGetWorkSpace(
    arkode_mem: &ARKodeMem,
    lenrwBBDP: &mut i64,
    leniwBBDP: &mut i64,
) -> i32 {
    /* access ARKodeMem and ARKLsMem structure */
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
            "ARKBBDPrecGetWorkSpace",
            file!(),
            MSG_LS_LMEM_NULL,
        );
        return ARKLS_LMEM_NULL;
    }

    /* Return immediately ARKBBDPrecData is NULL */
    let sizes = {
        let arkls_mem = arkls_mem_mut(ark_mem);
        arkls_mem
            .P_data
            .as_ref()
            .and_then(|b| b.downcast_ref::<ARKBBDPrecDataRec>())
            .map(|p| (p.rpwsize, p.ipwsize))
    };
    match sizes {
        None => {
            arkProcessError(
                Some(ark_mem),
                ARKLS_PMEM_NULL,
                line!() as i32,
                "ARKBBDPrecGetWorkSpace",
                file!(),
                MSG_BBD_PMEM_NULL,
            );
            ARKLS_PMEM_NULL
        }
        Some((rpwsize, ipwsize)) => {
            /* set outputs */
            *lenrwBBDP = rpwsize;
            *leniwBBDP = ipwsize;
            ARKLS_SUCCESS
        }
    }
}

/*-------------------------------------------------------------*/
pub fn ARKBBDPrecGetNumGfnEvals(arkode_mem: &ARKodeMem, ngevalsBBDP: &mut i64) -> i32 {
    /* access ARKodeMem and ARKLsMem structure */
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
            "ARKBBDPrecGetNumGfnEvals",
            file!(),
            MSG_LS_LMEM_NULL,
        );
        return ARKLS_LMEM_NULL;
    }

    /* Return immediately if ARKBBDPrecData is NULL */
    let nge = {
        let arkls_mem = arkls_mem_mut(ark_mem);
        arkls_mem
            .P_data
            .as_ref()
            .and_then(|b| b.downcast_ref::<ARKBBDPrecDataRec>())
            .map(|p| p.nge)
    };
    match nge {
        None => {
            arkProcessError(
                Some(ark_mem),
                ARKLS_PMEM_NULL,
                line!() as i32,
                "ARKBBDPrecGetNumGfnEvals",
                file!(),
                MSG_BBD_PMEM_NULL,
            );
            ARKLS_PMEM_NULL
        }
        Some(nge) => {
            /* set output */
            *ngevalsBBDP = nge;
            ARKLS_SUCCESS
        }
    }
}

/*---------------------------------------------------------------
 ARKBBDPrecSetup:

 ARKBBDPrecSetup generates and factors a banded block of the
 preconditioner matrix on each processor, via calls to the
 user-supplied gloc and cfn functions. It uses difference
 quotient approximations to the Jacobian elements.

 ARKBBDPrecSetup calculates a new J, if necessary, then
 calculates P = M - gamma*J, and does an LU factorization of P.

 The parameters of ARKBBDPrecSetup used here are as follows:

 t       is the current value of the independent variable.

 y       is the current value of the dependent variable vector,
         namely the predicted value of y(t).

 fy      is the vector f(t,y).

 jok     is an input flag indicating whether Jacobian-related
         data needs to be recomputed, as follows:
           jok == SUNFALSE means recompute Jacobian-related data
                  from scratch.
           jok == SUNTRUE  means that Jacobian data from the
                  previous ARKBBDPrecon call can be reused
                  (with the current value of gamma).
         A ARKBBDPrecon call with jok == SUNTRUE should only occur
         after a call with jok == SUNFALSE.

 jcurPtr is a pointer to an output integer flag which is
         set by ARKBBDPrecon as follows:
           *jcurPtr = SUNTRUE if Jacobian data was recomputed.
           *jcurPtr = SUNFALSE if Jacobian data was not recomputed,
                      but saved data was reused.

 gamma   is the scalar appearing in the Newton matrix.

 bbd_data is a pointer to the preconditioner data set by
          ARKBBDPrecInit

 Return value:
 The value returned by this ARKBBDPrecSetup function is the int
   0  if successful,
   1  for a recoverable error (step will be retried).
---------------------------------------------------------------*/
fn ARKBBDPrecSetup(
    t: sunrealtype,
    y: &N_Vector,
    _fy: &N_Vector,
    jok: sunbooleantype,
    jcurPtr: &mut sunbooleantype,
    gamma: sunrealtype,
    bbd_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    let pdata: &mut ARKBBDPrecDataRec = bbd_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<ARKBBDPrecDataRec>())
        .expect("bbd_data is ARKBBDPrecData");

    let ark_mem = pdata.arkode_mem.clone();

    /* If jok = SUNTRUE, use saved copy of J */
    if jok {
        *jcurPtr = SUNFALSE;
        let retval = SUNMatCopy(&pdata.savedJ, &pdata.savedP);
        if retval < 0 {
            arkProcessError(
                Some(&ark_mem),
                -1,
                line!() as i32,
                "ARKBBDPrecSetup",
                file!(),
                MSG_BBD_SUNMAT_FAIL,
            );
            return -1;
        }
        if retval > 0 {
            return 1;
        }

        /* Otherwise call ARKBBDDQJac for new J value */
    } else {
        *jcurPtr = SUNTRUE;
        let retval = SUNMatZero(&pdata.savedJ);
        if retval < 0 {
            arkProcessError(
                Some(&ark_mem),
                -1,
                line!() as i32,
                "ARKBBDPrecSetup",
                file!(),
                MSG_BBD_SUNMAT_FAIL,
            );
            return -1;
        }
        if retval > 0 {
            return 1;
        }

        let tmp1 = pdata.tmp1.clone().expect("tmp1");
        let tmp2 = pdata.tmp2.clone().expect("tmp2");
        let tmp3 = pdata.tmp3.clone().expect("tmp3");
        let retval = ARKBBDDQJac(pdata, t, y, &tmp1, &tmp2, &tmp3);
        if retval < 0 {
            arkProcessError(
                Some(&ark_mem),
                -1,
                line!() as i32,
                "ARKBBDPrecSetup",
                file!(),
                MSG_BBD_FUNC_FAILED,
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
                "ARKBBDPrecSetup",
                file!(),
                MSG_BBD_SUNMAT_FAIL,
            );
            return -1;
        }
        if retval > 0 {
            return 1;
        }
    }

    /* Scale and add I to get P = I - gamma*J */
    let retval = SUNMatScaleAddI(-gamma, &pdata.savedP);
    if retval != 0 {
        arkProcessError(
            Some(&ark_mem),
            -1,
            line!() as i32,
            "ARKBBDPrecSetup",
            file!(),
            MSG_BBD_SUNMAT_FAIL,
        );
        return -1;
    }

    /* Do LU factorization of matrix and return error flag */
    SUNLinSolSetup_Band(&pdata.LS, Some(&pdata.savedP))
}

/*---------------------------------------------------------------
 ARKBBDPrecSolve:

 ARKBBDPrecSolve solves a linear system P z = r, with the
 band-block-diagonal preconditioner matrix P generated and
 factored by ARKBBDPrecSetup.

 The parameters of ARKBBDPrecSolve used here are as follows:

 r is the right-hand side vector of the linear system.

 bbd_data is a pointer to the preconditioner data set by
   ARKBBDPrecInit.

 z is the output vector computed by ARKBBDPrecSolve.

 The value returned by the ARKBBDPrecSolve function is the same
 as the value returned from the linear solver object.
---------------------------------------------------------------*/
fn ARKBBDPrecSolve(
    _t: sunrealtype,
    _y: &N_Vector,
    _fy: &N_Vector,
    r: &N_Vector,
    z: &N_Vector,
    _gamma: sunrealtype,
    _delta: sunrealtype,
    _lr: i32,
    bbd_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    let pdata: &ARKBBDPrecDataRec = bbd_data
        .as_ref()
        .and_then(|b| b.downcast_ref::<ARKBBDPrecDataRec>())
        .expect("bbd_data is ARKBBDPrecData");

    /* Attach local data arrays for r and z to rlocal and zlocal
    (Rust: move the owned buffers into the local wrappers; C aliases
    the raw pointers). If r and z alias, zlocal gets a scratch buffer
    instead -- the band solve copies b into x before factor-solving in
    place, so the result written back to z is identical to C. */
    let r_data = {
        let mut g = N_VGetArrayPointer(r).expect("r data");
        std::mem::take(&mut *g)
    };
    let z_aliases_r = Rc::ptr_eq(r, z);
    let z_data = if z_aliases_r {
        vec![0.0; r_data.len()]
    } else {
        let mut g = N_VGetArrayPointer(z).expect("z data");
        std::mem::take(&mut *g)
    };
    N_VSetArrayPointer(r_data, &pdata.rlocal);
    N_VSetArrayPointer(z_data, &pdata.zlocal);

    /* Call banded solver object to do the work */
    let retval = SUNLinSolSolve(
        &pdata.LS,
        Some(&pdata.savedP),
        &pdata.zlocal,
        &pdata.rlocal,
        ZERO,
    );

    /* Detach local data arrays from rlocal and zlocal (move the
    buffers back; C sets the local wrappers' pointers to NULL) */
    let r_data = {
        let mut g = N_VGetArrayPointer(&pdata.rlocal).expect("rlocal data");
        std::mem::take(&mut *g)
    };
    let z_data = {
        let mut g = N_VGetArrayPointer(&pdata.zlocal).expect("zlocal data");
        std::mem::take(&mut *g)
    };
    if !z_aliases_r {
        N_VSetArrayPointer(r_data, r);
    }
    N_VSetArrayPointer(z_data, z);

    retval
}

/*-------------------------------------------------------------*/
fn ARKBBDPrecFree(ark_mem: &ARKodeMem) -> i32 {
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
    let mut pdata = match pdata.downcast::<ARKBBDPrecDataRec>() {
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

    /* SUNLinSolFree(LS) / N_VDestroy(zlocal, rlocal) /
    SUNMatDestroy(savedP, savedJ): dropping the record releases all of
    them. tmp1..tmp3 go through arkFreeVec exactly as in C so that
    ark_mem->lrw / ark_mem->liw are decremented (C frees each explicitly
    and leaves arkls_mem->P_data dangling; the Rust take() leaves it
    None) */
    arkFreeVec(ark_mem, &mut pdata.tmp1);
    arkFreeVec(ark_mem, &mut pdata.tmp2);
    arkFreeVec(ark_mem, &mut pdata.tmp3);

    drop(pdata);

    0
}

/*---------------------------------------------------------------
 ARKBBDDQJac:

 This routine generates a banded difference quotient approximation
 to the local block of the Jacobian of g(t,y). It assumes that a
 band matrix of type SUNMatrix is stored columnwise, and that
 elements within each column are contiguous. All matrix elements
 are generated as difference quotients, by way of calls to the
 user routine gloc.  By virtue of the band structure, the number
 of these calls is bandwidth + 1, where bandwidth = mldq + mudq + 1.
 But the band matrix kept has bandwidth = mlkeep + mukeep + 1.
 This routine also assumes that the local elements of a vector are
 stored contiguously.
---------------------------------------------------------------*/
fn ARKBBDDQJac(
    pdata: &mut ARKBBDPrecDataRec,
    t: sunrealtype,
    y: &N_Vector,
    gy: &N_Vector,
    ytemp: &N_Vector,
    gtemp: &N_Vector,
) -> i32 {
    let ark_mem = pdata.arkode_mem.clone();

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

    /* Load ytemp with y = predicted solution vector */
    N_VScale(ONE, y, ytemp);

    /* Call cfn and gloc to get base value of g(t,y) */
    if let Some(cfn) = pdata.cfn {
        let retval = {
            let mut user_data = ark_mem.borrow_mut().user_data.take();
            let retval = cfn(pdata.n_local, t, y, &mut user_data);
            ark_mem.borrow_mut().user_data = user_data;
            retval
        };
        if retval != 0 {
            return retval;
        }
    }

    let gloc = pdata.gloc;
    let retval = {
        let mut user_data = ark_mem.borrow_mut().user_data.take();
        let retval = gloc(pdata.n_local, t, ytemp, gy, &mut user_data);
        ark_mem.borrow_mut().user_data = user_data;
        retval
    };
    pdata.nge += 1;
    if retval != 0 {
        return retval;
    }

    /* Set minimum increment based on uround and norm of g */
    let gnorm = N_VWrmsNorm(gy, &rwt);
    let minInc = if gnorm != ZERO {
        MIN_INC_MULT * SUNRabs(h) * uround * (pdata.n_local as sunrealtype) * gnorm
    } else {
        ONE
    };

    /* Set bandwidth and number of column groups for band differencing */
    let width = pdata.mldq + pdata.mudq + 1;
    let ngroups = SUNMIN(width, pdata.n_local);

    let s_mu = SM_SUBAND_B(&pdata.savedJ);

    /* Loop over groups */
    for group in 1..=ngroups {
        /* Increment all y_j in group */
        {
            let y_data = N_VGetArrayPointer(y).expect("y data");
            let ewt_data = N_VGetArrayPointer(&ewt).expect("ewt data");
            let mut ytemp_data = N_VGetArrayPointer(ytemp).expect("ytemp data");
            let cns_data = constraints
                .as_ref()
                .map(|c| N_VGetArrayPointer(c).expect("constraints data"));

            let mut j = group - 1;
            while j < pdata.n_local {
                let ju = j as usize;
                let mut inc = SUNMAX(pdata.dqrely * SUNRabs(y_data[ju]), minInc / ewt_data[ju]);
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

        /* Evaluate g with incremented y */
        let retval = {
            let mut user_data = ark_mem.borrow_mut().user_data.take();
            let retval = gloc(pdata.n_local, t, ytemp, gtemp, &mut user_data);
            ark_mem.borrow_mut().user_data = user_data;
            retval
        };
        pdata.nge += 1;
        if retval != 0 {
            return retval;
        }

        /* Restore ytemp, then form and load difference quotients */
        {
            let y_data = N_VGetArrayPointer(y).expect("y data");
            let ewt_data = N_VGetArrayPointer(&ewt).expect("ewt data");
            let gy_data = N_VGetArrayPointer(gy).expect("gy data");
            let gtemp_data = N_VGetArrayPointer(gtemp).expect("gtemp data");
            let mut ytemp_data = N_VGetArrayPointer(ytemp).expect("ytemp data");
            let cns_data = constraints
                .as_ref()
                .map(|c| N_VGetArrayPointer(c).expect("constraints data"));

            let mut j = group - 1;
            while j < pdata.n_local {
                let ju = j as usize;
                let yj = y_data[ju];
                ytemp_data[ju] = y_data[ju];
                let mut col_j = SUNBandMatrix_Column(&pdata.savedJ, j);
                let mut inc = SUNMAX(pdata.dqrely * SUNRabs(y_data[ju]), minInc / ewt_data[ju]);

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
                let i1 = SUNMAX(0, j - pdata.mukeep);
                let i2 = SUNMIN(j + pdata.mlkeep, pdata.n_local - 1);
                let mut i = i1;
                while i <= i2 {
                    col_j[SM_COLUMN_ELEMENT_IDX(i, j, s_mu)] =
                        inc_inv * (gtemp_data[i as usize] - gy_data[i as usize]);
                    i += 1;
                }
                drop(col_j);
                j += width;
            }
        }
    }

    0
}
