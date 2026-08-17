//! Port of `src/cvodes/cvodes_bbdpre.c` (+ `src/cvodes/cvodes_bbdpre_impl.h`
//! and `include/cvodes/cvodes_bbdpre.h` folded).
//!
//! Band-block-diagonal preconditioner (banded blocks) for use with
//! CVODES and the CVSLS linear solver interface, plus the PART II
//! backward-problem wrappers (`CVBBDPrecInitB`, `CVBBDPrecReInitB`,
//! `cvGlocWrapper`, `cvCfnWrapper`). The upstream file is written
//! against the MPI-parallel NVECTOR; this port is the serial build (the
//! file itself has no MPI `#ifdef`s — it wraps the local data in
//! `N_VNewEmpty_Serial` vectors exactly as the C does).
//!
//! The preconditioner data lives in `cvls_mem.P_data`
//! (`Option<Box<dyn Any>>` holding a [`CVBBDPrecDataRec`]); the CVSLS
//! interface (`cvodes_ls`) `Option::take`s that box around each
//! psetup/psolve invocation, so the callbacks here receive it as
//! `&mut Option<Box<dyn Any>>` and downcast. The backward-problem
//! user functions live in `cvB_mem.cv_pmem`
//! (`Option<Box<dyn Any>>` holding a [`CVBBDPrecDataRecB`]).

use std::any::Any;
use std::rc::Rc;

use crate::cvodes_impl::*;
use crate::cvodes_ls::{
    cvls_mem_mut, CVLsMemRec, CVodeSetPreconditioner, CVLS_ILL_INPUT, CVLS_LMEM_NULL,
    CVLS_MEM_FAIL, CVLS_NO_ADJ, CVLS_PMEM_NULL, CVLS_SUCCESS, CVLS_SUNLS_FAIL,
};
use sundials_core::nvector_serial::N_VNewEmpty_Serial;
use sundials_core::sundials_errors::SUN_SUCCESS;
use sundials_core::sundials_linearsolver::{
    SUNLinSolInitialize, SUNLinSolSolve, SUNLinSolSpace, SUNLinearSolver,
};
use sundials_core::sundials_math::{SUNMAX, SUNMIN, SUNRabs, SUNRsqrt};
use sundials_core::sundials_matrix::{
    SUNMatCopy, SUNMatScaleAddI, SUNMatSpace, SUNMatZero, SUNMatrix,
};
use sundials_core::sundials_nvector::{
    N_VClone, N_VGetArrayPointer, N_VScale, N_VSetArrayPointer, N_VSpace, N_VWrmsNorm, N_Vector,
};
use sundials_core::sundials_types::*;
use sundials_core::sunlinsol_band::{SUNLinSolSetup_Band, SUNLinSol_Band};
use sundials_core::sunmatrix_band::{
    SM_COLUMN_ELEMENT_IDX, SM_SUBAND_B, SUNBandMatrixStorage, SUNBandMatrix_Column,
};

/* File-scope constants (shadow the same-named `cvodes_impl` constants,
which carry identical values — each C file redefines them locally). */
const MIN_INC_MULT: sunrealtype = 1000.0;
const ZERO: sunrealtype = 0.0;
const ONE: sunrealtype = 1.0;
const TWO: sunrealtype = 2.0;

/* -----------------------------------------------------------------
 * FORWARD PROBLEMS: user-supplied function types
 * (include/cvodes/cvodes_bbdpre.h)
 * ----------------------------------------------------------------- */

pub type CVLocalFn = fn(
    Nlocal: sunindextype,
    t: sunrealtype,
    y: &N_Vector,
    g: &N_Vector,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32;

pub type CVCommFn = fn(
    Nlocal: sunindextype,
    t: sunrealtype,
    y: &N_Vector,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32;

/* -----------------------------------------------------------------
 * BACKWARD PROBLEMS: user-supplied function types
 * (include/cvodes/cvodes_bbdpre.h)
 * ----------------------------------------------------------------- */

pub type CVLocalFnB = fn(
    NlocalB: sunindextype,
    t: sunrealtype,
    y: &N_Vector,
    yB: &N_Vector,
    gB: &N_Vector,
    user_dataB: &mut Option<Box<dyn Any>>,
) -> i32;

pub type CVCommFnB = fn(
    NlocalB: sunindextype,
    t: sunrealtype,
    y: &N_Vector,
    yB: &N_Vector,
    user_dataB: &mut Option<Box<dyn Any>>,
) -> i32;

/*-----------------------------------------------------------------
  Type: CVBBDPrecData (cvodes_bbdpre_impl.h)
  -----------------------------------------------------------------*/

pub struct CVBBDPrecDataRec {
    /* passed by user to CVBBDPrecInit and used by PrecSetup/PrecSolve */
    pub mudq: sunindextype,
    pub mldq: sunindextype,
    pub mukeep: sunindextype,
    pub mlkeep: sunindextype,
    pub dqrely: sunrealtype,
    pub gloc: CVLocalFn,
    pub cfn: Option<CVCommFn>,

    /* set by CVBBDPrecSetup and used by CVBBDPrecSolve */
    pub savedJ: SUNMatrix,
    pub savedP: SUNMatrix,
    pub LS: SUNLinearSolver,
    pub tmp1: N_Vector,
    pub tmp2: N_Vector,
    pub tmp3: N_Vector,
    pub zlocal: N_Vector,
    pub rlocal: N_Vector,

    /* set by CVBBDPrecInit and used by CVBBDPrecSetup */
    pub n_local: sunindextype,

    /* available for optional output */
    pub rpwsize: i64,
    pub ipwsize: i64,
    pub nge: i64,

    /* pointer to cvode_mem */
    pub cvode_mem: CVodeMem,
}

pub type CVBBDPrecData = Box<CVBBDPrecDataRec>;

/*-----------------------------------------------------------------
  Type: CVBBDPrecDataB (cvodes_bbdpre_impl.h)
  -----------------------------------------------------------------*/

pub struct CVBBDPrecDataRecB {
    /* BBD user functions (glocB and cfnB) for backward run */
    pub glocB: CVLocalFnB,
    pub cfnB: Option<CVCommFnB>,
}

pub type CVBBDPrecDataB = Box<CVBBDPrecDataRecB>;

/*-----------------------------------------------------------------
  CVBBDPRE error messages (cvodes_bbdpre_impl.h)
  -----------------------------------------------------------------*/

pub const MSGBBD_MEM_NULL: &str = "Integrator memory is NULL.";
pub const MSGBBD_LMEM_NULL: &str =
    "Linear solver memory is NULL. One of the SPILS linear solvers must be attached.";
pub const MSGBBD_MEM_FAIL: &str = "A memory request failed.";
pub const MSGBBD_BAD_NVECTOR: &str = "A required vector operation is not implemented.";
pub const MSGBBD_SUNMAT_FAIL: &str = "An error arose from a SUNBandMatrix routine.";
pub const MSGBBD_SUNLS_FAIL: &str = "An error arose from a SUNBandLinearSolver routine.";
pub const MSGBBD_PMEM_NULL: &str =
    "BBD peconditioner memory is NULL. CVBBDPrecInit must be called.";
pub const MSGBBD_FUNC_FAILED: &str =
    "The gloc or cfn routine failed in an unrecoverable manner.";

pub const MSGBBD_NO_ADJ: &str = "Illegal attempt to call before calling CVodeAdjInit.";
pub const MSGBBD_BAD_WHICH: &str = "Illegal value for the which parameter.";
pub const MSGBBD_PDATAB_NULL: &str =
    "BBD preconditioner memory is NULL for the backward integration.";
pub const MSGBBD_BAD_TINTERP: &str = "Bad t for interpolation.";

/*================================================================
  PART I - forward problems
  ================================================================*/

/*-----------------------------------------------------------------
  User-Callable Functions: initialization, reinit and free
  -----------------------------------------------------------------*/
pub fn CVBBDPrecInit(
    cvode_mem: &CVodeMem,
    Nlocal: sunindextype,
    mudq: sunindextype,
    mldq: sunindextype,
    mukeep: sunindextype,
    mlkeep: sunindextype,
    dqrely: sunrealtype,
    gloc: CVLocalFn,
    cfn: Option<CVCommFn>,
) -> i32 {
    /* NULL-mem check: handled by type system */
    let cv_mem = cvode_mem;

    /* Test if the CVSLS linear solver interface has been created */
    let attached = {
        let mem = cv_mem.borrow();
        mem.cv_lmem.as_ref().is_some_and(|b| b.is::<CVLsMemRec>())
    };
    if !attached {
        cvProcessError(
            Some(cv_mem),
            CVLS_LMEM_NULL,
            line!() as i32,
            "CVBBDPrecInit",
            file!(),
            MSGBBD_LMEM_NULL,
        );
        return CVLS_LMEM_NULL;
    }

    /* Test compatibility of NVECTOR package with the BBD preconditioner */
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
            "CVBBDPrecInit",
            file!(),
            MSGBBD_BAD_NVECTOR,
        );
        return CVLS_ILL_INPUT;
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
            cvProcessError(
                Some(cv_mem),
                CVLS_MEM_FAIL,
                line!() as i32,
                "CVBBDPrecInit",
                file!(),
                MSGBBD_MEM_FAIL,
            );
            return CVLS_MEM_FAIL;
        }
        Some(m) => m,
    };

    /* Allocate memory for preconditioner matrix */
    let storage_mu = SUNMIN(Nlocal - 1, muk + mlk);
    let savedP = match SUNBandMatrixStorage(Nlocal, muk, mlk, storage_mu, &sunctx) {
        None => {
            cvProcessError(
                Some(cv_mem),
                CVLS_MEM_FAIL,
                line!() as i32,
                "CVBBDPrecInit",
                file!(),
                MSGBBD_MEM_FAIL,
            );
            return CVLS_MEM_FAIL;
        }
        Some(m) => m,
    };

    /* Allocate memory for temporary N_Vectors */
    let zlocal = match N_VNewEmpty_Serial(Nlocal, &sunctx) {
        None => {
            cvProcessError(
                Some(cv_mem),
                CVLS_MEM_FAIL,
                line!() as i32,
                "CVBBDPrecInit",
                file!(),
                MSGBBD_MEM_FAIL,
            );
            return CVLS_MEM_FAIL;
        }
        Some(v) => v,
    };
    let rlocal = match N_VNewEmpty_Serial(Nlocal, &sunctx) {
        None => {
            cvProcessError(
                Some(cv_mem),
                CVLS_MEM_FAIL,
                line!() as i32,
                "CVBBDPrecInit",
                file!(),
                MSGBBD_MEM_FAIL,
            );
            return CVLS_MEM_FAIL;
        }
        Some(v) => v,
    };
    let tmp1 = match N_VClone(&tempv) {
        None => {
            cvProcessError(
                Some(cv_mem),
                CVLS_MEM_FAIL,
                line!() as i32,
                "CVBBDPrecInit",
                file!(),
                MSGBBD_MEM_FAIL,
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
                "CVBBDPrecInit",
                file!(),
                MSGBBD_MEM_FAIL,
            );
            return CVLS_MEM_FAIL;
        }
        Some(v) => v,
    };
    let tmp3 = match N_VClone(&tempv) {
        None => {
            cvProcessError(
                Some(cv_mem),
                CVLS_MEM_FAIL,
                line!() as i32,
                "CVBBDPrecInit",
                file!(),
                MSGBBD_MEM_FAIL,
            );
            return CVLS_MEM_FAIL;
        }
        Some(v) => v,
    };

    /* Allocate memory for banded linear solver */
    let LS = match SUNLinSol_Band(&rlocal, &savedP, &sunctx) {
        None => {
            cvProcessError(
                Some(cv_mem),
                CVLS_MEM_FAIL,
                line!() as i32,
                "CVBBDPrecInit",
                file!(),
                MSGBBD_MEM_FAIL,
            );
            return CVLS_MEM_FAIL;
        }
        Some(s) => s,
    };

    /* initialize band linear solver object */
    let flag = SUNLinSolInitialize(&LS);
    if flag != SUN_SUCCESS {
        cvProcessError(
            Some(cv_mem),
            CVLS_SUNLS_FAIL,
            line!() as i32,
            "CVBBDPrecInit",
            file!(),
            MSGBBD_SUNLS_FAIL,
        );
        return CVLS_SUNLS_FAIL;
    }

    /* Set pdata->dqrely based on input dqrely (0 implies default). */
    let dqrely = if dqrely > ZERO {
        dqrely
    } else {
        SUNRsqrt(cv_mem.borrow().cv_uround)
    };

    /* Store Nlocal to be used in CVBBDPrecSetup */
    let n_local = Nlocal;

    /* Set work space sizes and initialize nge */
    let mut rpwsize: i64 = 0;
    let mut ipwsize: i64 = 0;
    if tempv.ops.borrow().nvspace.is_some() {
        let mut lrw1: sunindextype = 0;
        let mut liw1: sunindextype = 0;
        N_VSpace(&tempv, &mut lrw1, &mut liw1);
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
    let pfree = cvls_mem_mut(cv_mem).pfree;
    if let Some(pfree) = pfree {
        pfree(cv_mem);
    }

    {
        let mut ls_mem = cvls_mem_mut(cv_mem);

        /* Point to the new P_data field in the LS memory */
        ls_mem.P_data = Some(Box::new(CVBBDPrecDataRec {
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
            cvode_mem: cv_mem.clone(),
        }));

        /* Attach the pfree function */
        ls_mem.pfree = Some(cvBBDPrecFree);
    }

    /* Attach preconditioner solve and setup functions */
    CVodeSetPreconditioner(cvode_mem, Some(cvBBDPrecSetup), Some(cvBBDPrecSolve))
}

pub fn CVBBDPrecReInit(
    cvode_mem: &CVodeMem,
    mudq: sunindextype,
    mldq: sunindextype,
    dqrely: sunrealtype,
) -> i32 {
    /* NULL-mem check: handled by type system */
    let cv_mem = cvode_mem;

    /* Test if the LS linear solver interface has been created */
    let attached = {
        let mem = cv_mem.borrow();
        mem.cv_lmem.as_ref().is_some_and(|b| b.is::<CVLsMemRec>())
    };
    if !attached {
        cvProcessError(
            Some(cv_mem),
            CVLS_LMEM_NULL,
            line!() as i32,
            "CVBBDPrecReInit",
            file!(),
            MSGBBD_LMEM_NULL,
        );
        return CVLS_LMEM_NULL;
    }

    /* Test if the preconditioner data is non-NULL */
    let has_pdata = {
        let ls_mem = cvls_mem_mut(cv_mem);
        ls_mem
            .P_data
            .as_ref()
            .and_then(|b| b.downcast_ref::<CVBBDPrecDataRec>())
            .is_some()
    };
    if !has_pdata {
        cvProcessError(
            Some(cv_mem),
            CVLS_PMEM_NULL,
            line!() as i32,
            "CVBBDPrecReInit",
            file!(),
            MSGBBD_PMEM_NULL,
        );
        return CVLS_PMEM_NULL;
    }

    /* Set pdata->dqrely based on input dqrely (0 implies default).
    (Rust: cv_uround is read before taking the CVLS guard — the guard
    is a cv_mem borrow_mut, so the read cannot happen inside it; the
    read has no side effects, C-observable order is unchanged.) */
    let dqrely = if dqrely > ZERO {
        dqrely
    } else {
        SUNRsqrt(cv_mem.borrow().cv_uround)
    };

    {
        let mut ls_mem = cvls_mem_mut(cv_mem);
        let pdata = ls_mem
            .P_data
            .as_mut()
            .and_then(|b| b.downcast_mut::<CVBBDPrecDataRec>())
            .expect("P_data is CVBBDPrecData");

        /* Load half-bandwidths */
        let Nlocal = pdata.n_local;
        pdata.mudq = SUNMIN(Nlocal - 1, SUNMAX(0, mudq));
        pdata.mldq = SUNMIN(Nlocal - 1, SUNMAX(0, mldq));

        pdata.dqrely = dqrely;

        /* Re-initialize nge */
        pdata.nge = 0;
    }

    CVLS_SUCCESS
}

pub fn CVBBDPrecGetWorkSpace(
    cvode_mem: &CVodeMem,
    lenrwBBDP: &mut i64,
    leniwBBDP: &mut i64,
) -> i32 {
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
            "CVBBDPrecGetWorkSpace",
            file!(),
            MSGBBD_LMEM_NULL,
        );
        return CVLS_LMEM_NULL;
    }

    let sizes = {
        let ls_mem = cvls_mem_mut(cv_mem);
        ls_mem
            .P_data
            .as_ref()
            .and_then(|b| b.downcast_ref::<CVBBDPrecDataRec>())
            .map(|p| (p.rpwsize, p.ipwsize))
    };
    match sizes {
        None => {
            cvProcessError(
                Some(cv_mem),
                CVLS_PMEM_NULL,
                line!() as i32,
                "CVBBDPrecGetWorkSpace",
                file!(),
                MSGBBD_PMEM_NULL,
            );
            CVLS_PMEM_NULL
        }
        Some((rpwsize, ipwsize)) => {
            *lenrwBBDP = rpwsize;
            *leniwBBDP = ipwsize;
            CVLS_SUCCESS
        }
    }
}

pub fn CVBBDPrecGetNumGfnEvals(cvode_mem: &CVodeMem, ngevalsBBDP: &mut i64) -> i32 {
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
            "CVBBDPrecGetNumGfnEvals",
            file!(),
            MSGBBD_LMEM_NULL,
        );
        return CVLS_LMEM_NULL;
    }

    let nge = {
        let ls_mem = cvls_mem_mut(cv_mem);
        ls_mem
            .P_data
            .as_ref()
            .and_then(|b| b.downcast_ref::<CVBBDPrecDataRec>())
            .map(|p| p.nge)
    };
    match nge {
        None => {
            cvProcessError(
                Some(cv_mem),
                CVLS_PMEM_NULL,
                line!() as i32,
                "CVBBDPrecGetNumGfnEvals",
                file!(),
                MSGBBD_PMEM_NULL,
            );
            CVLS_PMEM_NULL
        }
        Some(nge) => {
            *ngevalsBBDP = nge;
            CVLS_SUCCESS
        }
    }
}

/*-----------------------------------------------------------------
  Function : cvBBDPrecSetup
  -----------------------------------------------------------------
  cvBBDPrecSetup generates and factors a banded block of the
  preconditioner matrix on each processor, via calls to the
  user-supplied gloc and cfn functions. It uses difference
  quotient approximations to the Jacobian elements.

  cvBBDPrecSetup calculates a new J, if necessary, then calculates
  P = I - gamma*J, and does an LU factorization of P.

  Return value:
    0  if successful,
    1  for a recoverable error (step will be retried).
  -----------------------------------------------------------------*/
fn cvBBDPrecSetup(
    t: sunrealtype,
    y: &N_Vector,
    _fy: &N_Vector,
    jok: sunbooleantype,
    jcurPtr: &mut sunbooleantype,
    gamma: sunrealtype,
    bbd_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    let pdata: &mut CVBBDPrecDataRec = bbd_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<CVBBDPrecDataRec>())
        .expect("bbd_data is CVBBDPrecData");
    let cv_mem = pdata.cvode_mem.clone();

    /* If jok = SUNTRUE, use saved copy of J */
    if jok {
        *jcurPtr = SUNFALSE;
        let retval = SUNMatCopy(&pdata.savedJ, &pdata.savedP);
        if retval < 0 {
            cvProcessError(
                Some(&cv_mem),
                -1,
                line!() as i32,
                "cvBBDPrecSetup",
                file!(),
                MSGBBD_SUNMAT_FAIL,
            );
            return -1;
        }
        if retval > 0 {
            return 1;
        }

        /* Otherwise call cvBBDDQJac for new J value */
    } else {
        *jcurPtr = SUNTRUE;
        let retval = SUNMatZero(&pdata.savedJ);
        if retval < 0 {
            cvProcessError(
                Some(&cv_mem),
                -1,
                line!() as i32,
                "cvBBDPrecSetup",
                file!(),
                MSGBBD_SUNMAT_FAIL,
            );
            return -1;
        }
        if retval > 0 {
            return 1;
        }

        let tmp1 = pdata.tmp1.clone();
        let tmp2 = pdata.tmp2.clone();
        let tmp3 = pdata.tmp3.clone();
        let retval = cvBBDDQJac(pdata, t, y, &tmp1, &tmp2, &tmp3);
        if retval < 0 {
            cvProcessError(
                Some(&cv_mem),
                -1,
                line!() as i32,
                "cvBBDPrecSetup",
                file!(),
                MSGBBD_FUNC_FAILED,
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
                "cvBBDPrecSetup",
                file!(),
                MSGBBD_SUNMAT_FAIL,
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
        cvProcessError(
            Some(&cv_mem),
            -1,
            line!() as i32,
            "cvBBDPrecSetup",
            file!(),
            MSGBBD_SUNMAT_FAIL,
        );
        return -1;
    }

    /* Do LU factorization of matrix and return error flag */
    SUNLinSolSetup_Band(&pdata.LS, Some(&pdata.savedP))
}

/*-----------------------------------------------------------------
  Function : cvBBDPrecSolve
  -----------------------------------------------------------------
  cvBBDPrecSolve solves a linear system P z = r, with the
  band-block-diagonal preconditioner matrix P generated and
  factored by cvBBDPrecSetup.

  The value returned by the cvBBDPrecSolve function is always 0,
  indicating success.
  -----------------------------------------------------------------*/
fn cvBBDPrecSolve(
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
    let pdata: &CVBBDPrecDataRec = bbd_data
        .as_ref()
        .and_then(|b| b.downcast_ref::<CVBBDPrecDataRec>())
        .expect("bbd_data is CVBBDPrecData");

    /* Attach local data arrays for r and z to rlocal and zlocal
    (Rust: move the owned buffers into the local wrappers; C aliases
    the raw pointers). If r and z alias, zlocal gets a scratch buffer
    instead — the band solve copies b into x before factor-solving in
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
    let retval = SUNLinSolSolve(&pdata.LS, Some(&pdata.savedP), &pdata.zlocal, &pdata.rlocal, ZERO);

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

fn cvBBDPrecFree(cv_mem: &CVodeMem) -> i32 {
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

    /* SUNLinSolFree(LS) / N_VDestroy(tmp1..3, zlocal, rlocal) /
    SUNMatDestroy(savedP, savedJ): dropping the record releases
    everything (C frees each explicitly and leaves cvls_mem->P_data
    dangling; the Rust take() leaves it None) */
    drop(pdata);

    0
}

/*-----------------------------------------------------------------
  Function : cvBBDDQJac
  -----------------------------------------------------------------
  This routine generates a banded difference quotient approximation
  to the local block of the Jacobian of g(t,y). It assumes that a
  band SUNMatrix is stored columnwise, and that elements within each
  column are contiguous. All matrix elements are generated as
  difference quotients, by way of calls to the user routine gloc.
  By virtue of the band structure, the number of these calls is
  bandwidth + 1, where bandwidth = mldq + mudq + 1.
  But the band matrix kept has bandwidth = mlkeep + mukeep + 1.
  This routine also assumes that the local elements of a vector are
  stored contiguously.
  -----------------------------------------------------------------*/
fn cvBBDDQJac(
    pdata: &mut CVBBDPrecDataRec,
    t: sunrealtype,
    y: &N_Vector,
    gy: &N_Vector,
    ytemp: &N_Vector,
    gtemp: &N_Vector,
) -> i32 {
    let cv_mem = pdata.cvode_mem.clone();

    /* Copy the fields C reads through cv_mem-> out of the mem
    (granular borrow: nothing is held across callbacks/vector ops) */
    let (uround, h, ewt, constraints) = {
        let mem = cv_mem.borrow();
        (
            mem.cv_uround,
            mem.cv_h,
            mem.cv_ewt.clone().expect("cv_ewt"),
            mem.cv_constraints.clone(),
        )
    };

    /* Load ytemp with y = predicted solution vector */
    N_VScale(ONE, y, ytemp);

    /* Call cfn and gloc to get base value of g(t,y) */
    if let Some(cfn) = pdata.cfn {
        let retval = {
            let mut user_data = cv_mem.borrow_mut().cv_user_data.take();
            let retval = cfn(pdata.n_local, t, y, &mut user_data);
            cv_mem.borrow_mut().cv_user_data = user_data;
            retval
        };
        if retval != 0 {
            return retval;
        }
    }

    let gloc = pdata.gloc;
    let retval = {
        let mut user_data = cv_mem.borrow_mut().cv_user_data.take();
        let retval = gloc(pdata.n_local, t, ytemp, gy, &mut user_data);
        cv_mem.borrow_mut().cv_user_data = user_data;
        retval
    };
    pdata.nge += 1;
    if retval != 0 {
        return retval;
    }

    /* Set minimum increment based on uround and norm of g */
    let gnorm = N_VWrmsNorm(gy, &ewt);
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
            let mut user_data = cv_mem.borrow_mut().cv_user_data.take();
            let retval = gloc(pdata.n_local, t, ytemp, gtemp, &mut user_data);
            cv_mem.borrow_mut().cv_user_data = user_data;
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

/*================================================================
  PART II - Backward Problems
  ================================================================*/

/*---------------------------------------------------------------
  User-Callable Functions: initialization, reinit and free
  ---------------------------------------------------------------*/
pub fn CVBBDPrecInitB(
    cvode_mem: &CVodeMem,
    which: i32,
    NlocalB: sunindextype,
    mudqB: sunindextype,
    mldqB: sunindextype,
    mukeepB: sunindextype,
    mlkeepB: sunindextype,
    dqrelyB: sunrealtype,
    glocB: CVLocalFnB,
    cfnB: Option<CVCommFnB>,
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
            "CVBBDPrecInitB",
            file!(),
            MSGBBD_NO_ADJ,
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
            "CVBBDPrecInitB",
            file!(),
            MSGBBD_BAD_WHICH,
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

    /* Initialize the BBD preconditioner for this backward problem. */
    let flag = CVBBDPrecInit(
        &cvodeB_mem,
        NlocalB,
        mudqB,
        mldqB,
        mukeepB,
        mlkeepB,
        dqrelyB,
        cvGlocWrapper,
        Some(cvCfnWrapper),
    );
    if flag != CV_SUCCESS {
        return flag;
    }

    /* Allocate memory for CVBBDPrecDataB to store the user-provided
    functions which will be called from the wrappers (Rust: the record
    is built inline below; the C malloc-NULL branch has no analogue) */

    {
        let mut b = cvB_mem.borrow_mut();

        /* set pointers to user-provided functions */
        /* Attach pmem and pfree */
        b.cv_pmem = Some(Box::new(CVBBDPrecDataRecB { glocB, cfnB }));
        b.cv_pfree = Some(CVBBDPrecFreeB);
    }

    CVLS_SUCCESS
}

pub fn CVBBDPrecReInitB(
    cvode_mem: &CVodeMem,
    which: i32,
    mudqB: sunindextype,
    mldqB: sunindextype,
    dqrelyB: sunrealtype,
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
            "CVBBDPrecReInitB",
            file!(),
            MSGBBD_NO_ADJ,
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
            "CVBBDPrecReInitB",
            file!(),
            MSGBBD_BAD_WHICH,
        );
        return CVLS_ILL_INPUT;
    }

    /* Find the CVodeBMem entry in the linked list corresponding to which */
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

    /* cv_mem corresponding to 'which' backward problem. */
    let cvodeB_mem = cvB_mem.borrow().cv_mem.clone().expect("cvB_mem->cv_mem");

    /* ReInitialize the BBD preconditioner for this backward problem. */
    CVBBDPrecReInit(&cvodeB_mem, mudqB, mldqB, dqrelyB)
}

fn CVBBDPrecFreeB(cvB_mem: &CVodeBMem) -> i32 {
    cvB_mem.borrow_mut().cv_pmem = None;
    0
}

/*----------------------------------------------------------------
  Wrapper functions
  ----------------------------------------------------------------*/

/* cvGlocWrapper interfaces to the CVLocalFnB routine provided by the user */
fn cvGlocWrapper(
    NlocalB: sunindextype,
    t: sunrealtype,
    yB: &N_Vector,
    gB: &N_Vector,
    cvode_mem: &mut Option<Box<dyn Any>>,
) -> i32 {
    /* The backward integrator's user_data is the FORWARD cvode_mem
    (set by CVodeCreateB); C casts the void* straight back. */
    let cv_mem: CVodeMem = cvode_mem
        .as_ref()
        .and_then(|b| b.downcast_ref::<CVodeMem>())
        .expect("backward user_data is the forward CVodeMem")
        .clone();
    let ca_mem = cv_mem.borrow().cv_adj_mem.clone().expect("cv_adj_mem");
    let cvB_mem = ca_mem
        .borrow()
        .ca_bckpbCrt
        .clone()
        .expect("ca_bckpbCrt");
    let glocB = {
        let b = cvB_mem.borrow();
        b.cv_pmem
            .as_ref()
            .and_then(|p| p.downcast_ref::<CVBBDPrecDataRecB>())
            .expect("cv_pmem is CVBBDPrecDataB")
            .glocB
    };

    /* Get forward solution from interpolation */
    let (IMget, ytmp) = {
        let ca = ca_mem.borrow();
        (
            ca.ca_IMget.expect("ca_IMget"),
            ca.ca_ytmp.clone().expect("ca_ytmp"),
        )
    };
    let flag = IMget(&cv_mem, t, &ytmp, &[]);
    if flag != CV_SUCCESS {
        cvProcessError(
            Some(&cv_mem),
            -1,
            line!() as i32,
            "cvGlocWrapper",
            file!(),
            MSGBBD_BAD_TINTERP,
        );
        return -1;
    }

    /* Call user's adjoint glocB routine */
    let mut user_dataB = cvB_mem.borrow_mut().cv_user_data.take();
    let retval = glocB(NlocalB, t, &ytmp, yB, gB, &mut user_dataB);
    cvB_mem.borrow_mut().cv_user_data = user_dataB;
    retval
}

/* cvCfnWrapper interfaces to the CVCommFnB routine provided by the user */
fn cvCfnWrapper(
    NlocalB: sunindextype,
    t: sunrealtype,
    yB: &N_Vector,
    cvode_mem: &mut Option<Box<dyn Any>>,
) -> i32 {
    /* The backward integrator's user_data is the FORWARD cvode_mem
    (set by CVodeCreateB); C casts the void* straight back. */
    let cv_mem: CVodeMem = cvode_mem
        .as_ref()
        .and_then(|b| b.downcast_ref::<CVodeMem>())
        .expect("backward user_data is the forward CVodeMem")
        .clone();
    let ca_mem = cv_mem.borrow().cv_adj_mem.clone().expect("cv_adj_mem");
    let cvB_mem = ca_mem
        .borrow()
        .ca_bckpbCrt
        .clone()
        .expect("ca_bckpbCrt");
    let cfnB = {
        let b = cvB_mem.borrow();
        b.cv_pmem
            .as_ref()
            .and_then(|p| p.downcast_ref::<CVBBDPrecDataRecB>())
            .expect("cv_pmem is CVBBDPrecDataB")
            .cfnB
    };
    let cfnB = match cfnB {
        None => return 0,
        Some(f) => f,
    };

    /* Get forward solution from interpolation */
    let (IMget, ytmp) = {
        let ca = ca_mem.borrow();
        (
            ca.ca_IMget.expect("ca_IMget"),
            ca.ca_ytmp.clone().expect("ca_ytmp"),
        )
    };
    let flag = IMget(&cv_mem, t, &ytmp, &[]);
    if flag != CV_SUCCESS {
        cvProcessError(
            Some(&cv_mem),
            -1,
            line!() as i32,
            "cvCfnWrapper",
            file!(),
            MSGBBD_BAD_TINTERP,
        );
        return -1;
    }

    /* Call user's adjoint cfnB routine */
    let mut user_dataB = cvB_mem.borrow_mut().cv_user_data.take();
    let retval = cfnB(NlocalB, t, &ytmp, yB, &mut user_dataB);
    cvB_mem.borrow_mut().cv_user_data = user_dataB;
    retval
}
