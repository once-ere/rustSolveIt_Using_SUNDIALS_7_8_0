//! Port of `src/ida/ida_bbdpre.c` (+ `src/ida/ida_bbdpre_impl.h` and
//! `include/ida/ida_bbdpre.h` folded).
//!
//! Band-block-diagonal preconditioner, i.e. a block-diagonal matrix with
//! banded blocks, for use with IDA and the IDALS linear solver
//! interface. With only one processor in use a banded matrix results
//! rather than a block-diagonal matrix with banded blocks; the upstream
//! file has no MPI `#ifdef`s — it wraps the local data in
//! `N_VNewEmpty_Serial` vectors exactly as the C does.
//!
//! The preconditioner data lives in `idals_mem.pdata`
//! (`Option<Box<dyn Any>>` holding an [`IBBDPrecDataRec`]); the IDALS
//! interface (`ida_ls`) `Option::take`s that box around each
//! psetup/psolve invocation, so the callbacks here receive it as
//! `&mut Option<Box<dyn Any>>` and downcast.

use std::any::Any;
use std::rc::Rc;

use crate::ida_impl::*;
use crate::ida_ls::{
    idals_mem_mut, IDALsMemRec, IDASetPreconditioner, IDALS_ILL_INPUT, IDALS_LMEM_NULL,
    IDALS_MEM_FAIL, IDALS_PMEM_NULL, IDALS_SUCCESS, IDALS_SUNLS_FAIL,
};
use sundials_core::nvector_serial::N_VNewEmpty_Serial;
use sundials_core::sundials_errors::SUN_SUCCESS;
use sundials_core::sundials_linearsolver::{
    SUNLinSolInitialize, SUNLinSolSolve, SUNLinSolSpace, SUNLinearSolver,
};
use sundials_core::sundials_math::{SUNMAX, SUNMIN, SUNRabs, SUNRsqrt};
use sundials_core::sundials_matrix::{SUNMatSpace, SUNMatZero, SUNMatrix};
use sundials_core::sundials_nvector::{
    N_VClone, N_VGetArrayPointer, N_VScale, N_VSetArrayPointer, N_VSpace, N_Vector,
};
use sundials_core::sundials_types::*;
use sundials_core::sunlinsol_band::{SUNLinSolSetup_Band, SUNLinSol_Band};
use sundials_core::sunmatrix_band::{
    SM_COLUMN_ELEMENT_IDX, SM_SUBAND_B, SUNBandMatrixStorage, SUNBandMatrix_Column,
};

const ZERO: sunrealtype = 0.0;
const ONE: sunrealtype = 1.0;
const TWO: sunrealtype = 2.0;

/* -----------------------------------------------------------------
 * User-supplied function types (include/ida/ida_bbdpre.h)
 * ----------------------------------------------------------------- */

pub type IDABBDLocalFn = fn(
    Nlocal: sunindextype,
    tt: sunrealtype,
    yy: &N_Vector,
    yp: &N_Vector,
    gval: &N_Vector,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32;

pub type IDABBDCommFn = fn(
    Nlocal: sunindextype,
    tt: sunrealtype,
    yy: &N_Vector,
    yp: &N_Vector,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32;

/*
 * -----------------------------------------------------------------
 * Definition of IBBDPrecData (ida_bbdpre_impl.h)
 * -----------------------------------------------------------------
 */

pub struct IBBDPrecDataRec {
    /* passed by user to IDABBDPrecAlloc and used by
    IDABBDPrecSetup/IDABBDPrecSolve functions */
    pub mudq: sunindextype,
    pub mldq: sunindextype,
    pub mukeep: sunindextype,
    pub mlkeep: sunindextype,
    pub rel_yy: sunrealtype,
    pub glocal: IDABBDLocalFn,
    pub gcomm: Option<IDABBDCommFn>,

    /* set by IDABBDPrecSetup and used by IDABBDPrecSetup and
    IDABBDPrecSolve functions */
    pub n_local: sunindextype,
    pub PP: SUNMatrix,
    pub LS: SUNLinearSolver,
    pub zlocal: N_Vector,
    pub rlocal: N_Vector,
    pub tempv1: N_Vector,
    pub tempv2: N_Vector,
    pub tempv3: N_Vector,
    pub tempv4: N_Vector,

    /* available for optional output */
    pub rpwsize: i64,
    pub ipwsize: i64,
    pub nge: i64,

    /* pointer to ida_mem */
    pub ida_mem: IDAMem,
}

pub type IBBDPrecData = Box<IBBDPrecDataRec>;

/*
 * -----------------------------------------------------------------
 * IDABBDPRE error messages (ida_bbdpre_impl.h)
 * -----------------------------------------------------------------
 */

pub const MSGBBD_MEM_NULL: &str = "Integrator memory is NULL.";
pub const MSGBBD_LMEM_NULL: &str =
    "Linear solver memory is NULL. One of the SPILS linear solvers must be attached.";
pub const MSGBBD_MEM_FAIL: &str = "A memory request failed.";
pub const MSGBBD_BAD_NVECTOR: &str = "A required vector operation is not implemented.";
pub const MSGBBD_SUNMAT_FAIL: &str = "An error arose from a SUNBandMatrix routine.";
pub const MSGBBD_SUNLS_FAIL: &str = "An error arose from a SUNBandLinearSolver routine.";
pub const MSGBBD_PMEM_NULL: &str =
    "BBD peconditioner memory is NULL. IDABBDPrecInit must be called.";
pub const MSGBBD_FUNC_FAILED: &str =
    "The Glocal or Gcomm routine failed in an unrecoverable manner.";

/*---------------------------------------------------------------
  User-Callable Functions: initialization, reinit and free
  ---------------------------------------------------------------*/
#[allow(clippy::too_many_arguments)]
pub fn IDABBDPrecInit(
    ida_mem: &IDAMem,
    Nlocal: sunindextype,
    mudq: sunindextype,
    mldq: sunindextype,
    mukeep: sunindextype,
    mlkeep: sunindextype,
    dq_rel_yy: sunrealtype,
    Gres: IDABBDLocalFn,
    Gcomm: Option<IDABBDCommFn>,
) -> i32 {
    /* NULL-mem check: handled by type system */
    let IDA_mem = ida_mem;

    /* Test if the LS linear solver interface has been created */
    let attached = {
        let mem = IDA_mem.borrow();
        mem.ida_lmem.as_ref().is_some_and(|b| b.is::<IDALsMemRec>())
    };
    if !attached {
        IDAProcessError(
            Some(IDA_mem),
            IDALS_LMEM_NULL,
            line!() as i32,
            "IDABBDPrecInit",
            file!(),
            MSGBBD_LMEM_NULL,
        );
        return IDALS_LMEM_NULL;
    }

    /* Test compatibility of NVECTOR package with the BBD preconditioner */
    let (tempv1, sunctx) = {
        let mem = IDA_mem.borrow();
        (
            mem.ida_tempv1.clone().expect("ida_tempv1"),
            mem.ida_sunctx.clone(),
        )
    };
    if tempv1.ops.borrow().nvgetarraypointer.is_none() {
        IDAProcessError(
            Some(IDA_mem),
            IDALS_ILL_INPUT,
            line!() as i32,
            "IDABBDPrecInit",
            file!(),
            MSGBBD_BAD_NVECTOR,
        );
        return IDALS_ILL_INPUT;
    }

    /* Allocate data memory (Rust: the record is assembled at the end of
    this function; the C malloc-NULL branch has no analogue) */

    /* Set pointers to glocal and gcomm; load half-bandwidths. */
    let mudq = SUNMIN(Nlocal - 1, SUNMAX(0, mudq));
    let mldq = SUNMIN(Nlocal - 1, SUNMAX(0, mldq));
    let muk = SUNMIN(Nlocal - 1, SUNMAX(0, mukeep));
    let mlk = SUNMIN(Nlocal - 1, SUNMAX(0, mlkeep));

    /* Set extended upper half-bandwidth for PP (required for pivoting). */
    let storage_mu = SUNMIN(Nlocal - 1, muk + mlk);

    /* Allocate memory for preconditioner matrix. */
    let PP = match SUNBandMatrixStorage(Nlocal, muk, mlk, storage_mu, &sunctx) {
        None => {
            IDAProcessError(
                Some(IDA_mem),
                IDALS_MEM_FAIL,
                line!() as i32,
                "IDABBDPrecInit",
                file!(),
                MSGBBD_MEM_FAIL,
            );
            return IDALS_MEM_FAIL;
        }
        Some(m) => m,
    };

    /* Allocate memory for temporary N_Vectors */
    let zlocal = match N_VNewEmpty_Serial(Nlocal, &sunctx) {
        None => {
            IDAProcessError(
                Some(IDA_mem),
                IDALS_MEM_FAIL,
                line!() as i32,
                "IDABBDPrecInit",
                file!(),
                MSGBBD_MEM_FAIL,
            );
            return IDALS_MEM_FAIL;
        }
        Some(v) => v,
    };
    let rlocal = match N_VNewEmpty_Serial(Nlocal, &sunctx) {
        None => {
            IDAProcessError(
                Some(IDA_mem),
                IDALS_MEM_FAIL,
                line!() as i32,
                "IDABBDPrecInit",
                file!(),
                MSGBBD_MEM_FAIL,
            );
            return IDALS_MEM_FAIL;
        }
        Some(v) => v,
    };
    let tempv1_p = match N_VClone(&tempv1) {
        None => {
            IDAProcessError(
                Some(IDA_mem),
                IDALS_MEM_FAIL,
                line!() as i32,
                "IDABBDPrecInit",
                file!(),
                MSGBBD_MEM_FAIL,
            );
            return IDALS_MEM_FAIL;
        }
        Some(v) => v,
    };
    let tempv2_p = match N_VClone(&tempv1) {
        None => {
            IDAProcessError(
                Some(IDA_mem),
                IDALS_MEM_FAIL,
                line!() as i32,
                "IDABBDPrecInit",
                file!(),
                MSGBBD_MEM_FAIL,
            );
            return IDALS_MEM_FAIL;
        }
        Some(v) => v,
    };
    let tempv3_p = match N_VClone(&tempv1) {
        None => {
            IDAProcessError(
                Some(IDA_mem),
                IDALS_MEM_FAIL,
                line!() as i32,
                "IDABBDPrecInit",
                file!(),
                MSGBBD_MEM_FAIL,
            );
            return IDALS_MEM_FAIL;
        }
        Some(v) => v,
    };
    let tempv4_p = match N_VClone(&tempv1) {
        None => {
            IDAProcessError(
                Some(IDA_mem),
                IDALS_MEM_FAIL,
                line!() as i32,
                "IDABBDPrecInit",
                file!(),
                MSGBBD_MEM_FAIL,
            );
            return IDALS_MEM_FAIL;
        }
        Some(v) => v,
    };

    /* Allocate memory for banded linear solver */
    let LS = match SUNLinSol_Band(&rlocal, &PP, &sunctx) {
        None => {
            IDAProcessError(
                Some(IDA_mem),
                IDALS_MEM_FAIL,
                line!() as i32,
                "IDABBDPrecInit",
                file!(),
                MSGBBD_MEM_FAIL,
            );
            return IDALS_MEM_FAIL;
        }
        Some(s) => s,
    };

    /* initialize band linear solver object */
    let flag = SUNLinSolInitialize(&LS);
    if flag != SUN_SUCCESS {
        IDAProcessError(
            Some(IDA_mem),
            IDALS_SUNLS_FAIL,
            line!() as i32,
            "IDABBDPrecInit",
            file!(),
            MSGBBD_SUNLS_FAIL,
        );
        return IDALS_SUNLS_FAIL;
    }

    /* Set rel_yy based on input value dq_rel_yy (0 implies default). */
    let rel_yy = if dq_rel_yy > ZERO {
        dq_rel_yy
    } else {
        SUNRsqrt(IDA_mem.borrow().ida_uround)
    };

    /* Store Nlocal to be used in IDABBDPrecSetup */
    let n_local = Nlocal;

    /* Set work space sizes and initialize nge. */
    let mut rpwsize: i64 = 0;
    let mut ipwsize: i64 = 0;
    if tempv1.ops.borrow().nvspace.is_some() {
        let mut lrw1: sunindextype = 0;
        let mut liw1: sunindextype = 0;
        N_VSpace(&tempv1, &mut lrw1, &mut liw1);
        rpwsize += 4 * lrw1;
        ipwsize += 4 * liw1;
    }
    if rlocal.ops.borrow().nvspace.is_some() {
        let mut lrw1: sunindextype = 0;
        let mut liw1: sunindextype = 0;
        N_VSpace(&rlocal, &mut lrw1, &mut liw1);
        rpwsize += 2 * lrw1;
        ipwsize += 2 * liw1;
    }
    if PP.ops.borrow().space.is_some() {
        let mut lrw: i64 = 0;
        let mut liw: i64 = 0;
        let _ = SUNMatSpace(&PP, &mut lrw, &mut liw);
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

    /* make sure pdata is free from any previous allocations */
    let pfree = idals_mem_mut(IDA_mem).pfree;
    if let Some(pfree) = pfree {
        pfree(IDA_mem);
    }

    {
        let mut ls_mem = idals_mem_mut(IDA_mem);

        /* Point to the new pdata field in the LS memory */
        ls_mem.pdata = Some(Box::new(IBBDPrecDataRec {
            mudq,
            mldq,
            mukeep: muk,
            mlkeep: mlk,
            rel_yy,
            glocal: Gres,
            gcomm: Gcomm,
            n_local,
            PP,
            LS,
            zlocal,
            rlocal,
            tempv1: tempv1_p,
            tempv2: tempv2_p,
            tempv3: tempv3_p,
            tempv4: tempv4_p,
            rpwsize,
            ipwsize,
            nge,
            ida_mem: IDA_mem.clone(),
        }));

        /* Attach the pfree function */
        ls_mem.pfree = Some(IDABBDPrecFree);
    }

    /* Attach preconditioner solve and setup functions */
    IDASetPreconditioner(ida_mem, Some(IDABBDPrecSetup), Some(IDABBDPrecSolve))
}

/*-------------------------------------------------------------*/
pub fn IDABBDPrecReInit(
    ida_mem: &IDAMem,
    mudq: sunindextype,
    mldq: sunindextype,
    dq_rel_yy: sunrealtype,
) -> i32 {
    /* NULL-mem check: handled by type system */
    let IDA_mem = ida_mem;

    /* Test if the LS linear solver interface has been created */
    let attached = {
        let mem = IDA_mem.borrow();
        mem.ida_lmem.as_ref().is_some_and(|b| b.is::<IDALsMemRec>())
    };
    if !attached {
        IDAProcessError(
            Some(IDA_mem),
            IDALS_LMEM_NULL,
            line!() as i32,
            "IDABBDPrecReInit",
            file!(),
            MSGBBD_LMEM_NULL,
        );
        return IDALS_LMEM_NULL;
    }

    /* Test if the preconditioner data is non-NULL */
    let has_pdata = {
        let ls_mem = idals_mem_mut(IDA_mem);
        ls_mem
            .pdata
            .as_ref()
            .and_then(|b| b.downcast_ref::<IBBDPrecDataRec>())
            .is_some()
    };
    if !has_pdata {
        IDAProcessError(
            Some(IDA_mem),
            IDALS_PMEM_NULL,
            line!() as i32,
            "IDABBDPrecReInit",
            file!(),
            MSGBBD_PMEM_NULL,
        );
        return IDALS_PMEM_NULL;
    }

    /* Set rel_yy based on input value dq_rel_yy (0 implies default).
    (Rust: ida_uround is read before taking the IDALS guard — the guard
    is an IDA_mem borrow_mut, so the read cannot happen inside it; the
    read has no side effects, C-observable order is unchanged.) */
    let rel_yy = if dq_rel_yy > ZERO {
        dq_rel_yy
    } else {
        SUNRsqrt(IDA_mem.borrow().ida_uround)
    };

    {
        let mut ls_mem = idals_mem_mut(IDA_mem);
        let pdata = ls_mem
            .pdata
            .as_mut()
            .and_then(|b| b.downcast_mut::<IBBDPrecDataRec>())
            .expect("pdata is IBBDPrecData");

        /* Load half-bandwidths. */
        let Nlocal = pdata.n_local;
        pdata.mudq = SUNMIN(Nlocal - 1, SUNMAX(0, mudq));
        pdata.mldq = SUNMIN(Nlocal - 1, SUNMAX(0, mldq));

        pdata.rel_yy = rel_yy;

        /* Re-initialize nge */
        pdata.nge = 0;
    }

    IDALS_SUCCESS
}

/*-------------------------------------------------------------*/
pub fn IDABBDPrecGetWorkSpace(
    ida_mem: &IDAMem,
    lenrwBBDP: &mut i64,
    leniwBBDP: &mut i64,
) -> i32 {
    /* NULL-mem check: handled by type system */
    let IDA_mem = ida_mem;

    let attached = {
        let mem = IDA_mem.borrow();
        mem.ida_lmem.as_ref().is_some_and(|b| b.is::<IDALsMemRec>())
    };
    if !attached {
        IDAProcessError(
            Some(IDA_mem),
            IDALS_LMEM_NULL,
            line!() as i32,
            "IDABBDPrecGetWorkSpace",
            file!(),
            MSGBBD_LMEM_NULL,
        );
        return IDALS_LMEM_NULL;
    }

    let sizes = {
        let ls_mem = idals_mem_mut(IDA_mem);
        ls_mem
            .pdata
            .as_ref()
            .and_then(|b| b.downcast_ref::<IBBDPrecDataRec>())
            .map(|p| (p.rpwsize, p.ipwsize))
    };
    match sizes {
        None => {
            IDAProcessError(
                Some(IDA_mem),
                IDALS_PMEM_NULL,
                line!() as i32,
                "IDABBDPrecGetWorkSpace",
                file!(),
                MSGBBD_PMEM_NULL,
            );
            IDALS_PMEM_NULL
        }
        Some((rpwsize, ipwsize)) => {
            *lenrwBBDP = rpwsize;
            *leniwBBDP = ipwsize;
            IDALS_SUCCESS
        }
    }
}

/*-------------------------------------------------------------*/
pub fn IDABBDPrecGetNumGfnEvals(ida_mem: &IDAMem, ngevalsBBDP: &mut i64) -> i32 {
    /* NULL-mem check: handled by type system */
    let IDA_mem = ida_mem;

    let attached = {
        let mem = IDA_mem.borrow();
        mem.ida_lmem.as_ref().is_some_and(|b| b.is::<IDALsMemRec>())
    };
    if !attached {
        IDAProcessError(
            Some(IDA_mem),
            IDALS_LMEM_NULL,
            line!() as i32,
            "IDABBDPrecGetNumGfnEvals",
            file!(),
            MSGBBD_LMEM_NULL,
        );
        return IDALS_LMEM_NULL;
    }

    let nge = {
        let ls_mem = idals_mem_mut(IDA_mem);
        ls_mem
            .pdata
            .as_ref()
            .and_then(|b| b.downcast_ref::<IBBDPrecDataRec>())
            .map(|p| p.nge)
    };
    match nge {
        None => {
            IDAProcessError(
                Some(IDA_mem),
                IDALS_PMEM_NULL,
                line!() as i32,
                "IDABBDPrecGetNumGfnEvals",
                file!(),
                MSGBBD_PMEM_NULL,
            );
            IDALS_PMEM_NULL
        }
        Some(nge) => {
            *ngevalsBBDP = nge;
            IDALS_SUCCESS
        }
    }
}

/*---------------------------------------------------------------
  IDABBDPrecSetup:

  IDABBDPrecSetup generates a band-block-diagonal preconditioner
  matrix, where the local block (on this processor) is a band
  matrix. Each local block is computed by a difference quotient
  scheme via calls to the user-supplied routines glocal, gcomm.
  After generating the block in the band matrix PP, this routine
  does an LU factorization in place in PP.

  The IDABBDPrecSetup parameters used here are as follows:

  tt is the current value of the independent variable t.

  yy is the current value of the dependent variable vector,
     namely the predicted value of y(t).

  yp is the current value of the derivative vector y',
     namely the predicted value of y'(t).

  c_j is the scalar in the system Jacobian, proportional to 1/hh.

  bbd_data is the pointer to BBD memory set by IDABBDInit

  The argument rr is not used.

  Return value:
  The value returned by this IDABBDPrecSetup function is a int
  flag indicating whether it was successful. This value is
     0    if successful,
   > 0    for a recoverable error (step will be retried), or
   < 0    for a nonrecoverable error (step fails).
 ----------------------------------------------------------------*/
fn IDABBDPrecSetup(
    tt: sunrealtype,
    yy: &N_Vector,
    yp: &N_Vector,
    _rr: &N_Vector,
    c_j: sunrealtype,
    bbd_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    let pdata: &mut IBBDPrecDataRec = bbd_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<IBBDPrecDataRec>())
        .expect("bbd_data is IBBDPrecData");

    let IDA_mem = pdata.ida_mem.clone();

    /* Call IBBDDQJac for a new Jacobian calculation and store in PP.
    (C assigns the SUNMatZero flag to `retval` and then immediately
    overwrites it with the IBBDDQJac flag — the value is dead.) */
    let _retval = SUNMatZero(&pdata.PP);
    let tempv1 = pdata.tempv1.clone();
    let tempv2 = pdata.tempv2.clone();
    let tempv3 = pdata.tempv3.clone();
    let tempv4 = pdata.tempv4.clone();
    let retval = IBBDDQJac(
        pdata, tt, c_j, yy, yp, &tempv1, &tempv2, &tempv3, &tempv4,
    );
    if retval < 0 {
        IDAProcessError(
            Some(&IDA_mem),
            -1,
            line!() as i32,
            "IDABBDPrecSetup",
            file!(),
            MSGBBD_FUNC_FAILED,
        );
        return -1;
    }
    if retval > 0 {
        return 1;
    }

    /* Do LU factorization of matrix and return error flag */
    SUNLinSolSetup_Band(&pdata.LS, Some(&pdata.PP))
}

/*---------------------------------------------------------------
  IDABBDPrecSolve

  The function IDABBDPrecSolve computes a solution to the linear
  system P z = r, where P is the left preconditioner defined by
  the routine IDABBDPrecSetup.

  The IDABBDPrecSolve parameters used here are as follows:

  rvec is the input right-hand side vector r.

  zvec is the computed solution vector z.

  bbd_data is the pointer to BBD data set by IDABBDInit.

  The arguments tt, yy, yp, rr, c_j and delta are NOT used.

  IDABBDPrecSolve returns the value returned from the linear
  solver object.
  ---------------------------------------------------------------*/
#[allow(clippy::too_many_arguments)]
fn IDABBDPrecSolve(
    _tt: sunrealtype,
    _yy: &N_Vector,
    _yp: &N_Vector,
    _rr: &N_Vector,
    rvec: &N_Vector,
    zvec: &N_Vector,
    _c_j: sunrealtype,
    _delta: sunrealtype,
    bbd_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    let pdata: &IBBDPrecDataRec = bbd_data
        .as_ref()
        .and_then(|b| b.downcast_ref::<IBBDPrecDataRec>())
        .expect("bbd_data is IBBDPrecData");

    /* Attach local data arrays for rvec and zvec to rlocal and zlocal
    (Rust: move the owned buffers into the local wrappers; C aliases
    the raw pointers). If rvec and zvec alias, zlocal gets a scratch
    buffer instead — the band solve copies b into x before factor-
    solving in place, so the result written back to zvec is identical
    to C. */
    let r_data = {
        let mut g = N_VGetArrayPointer(rvec).expect("rvec data");
        std::mem::take(&mut *g)
    };
    let z_aliases_r = Rc::ptr_eq(rvec, zvec);
    let z_data = if z_aliases_r {
        vec![0.0; r_data.len()]
    } else {
        let mut g = N_VGetArrayPointer(zvec).expect("zvec data");
        std::mem::take(&mut *g)
    };
    N_VSetArrayPointer(r_data, &pdata.rlocal);
    N_VSetArrayPointer(z_data, &pdata.zlocal);

    /* Call banded solver object to do the work */
    let retval = SUNLinSolSolve(
        &pdata.LS,
        Some(&pdata.PP),
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
        N_VSetArrayPointer(r_data, rvec);
    }
    N_VSetArrayPointer(z_data, zvec);

    retval
}

/*-------------------------------------------------------------*/
fn IDABBDPrecFree(IDA_mem: &IDAMem) -> i32 {
    let attached = {
        let mem = IDA_mem.borrow();
        mem.ida_lmem.as_ref().is_some_and(|b| b.is::<IDALsMemRec>())
    };
    if !attached {
        return 0;
    }

    let pdata = idals_mem_mut(IDA_mem).pdata.take();
    if pdata.is_none() {
        return 0;
    }

    /* SUNLinSolFree(LS) / N_VDestroy(rlocal, zlocal, tempv1..4) /
    SUNMatDestroy(PP): dropping the record releases everything (C frees
    each explicitly and leaves idals_mem->pdata dangling; the Rust
    take() leaves it None) */
    drop(pdata);

    0
}

/*---------------------------------------------------------------
  IBBDDQJac

  This routine generates a banded difference quotient approximation
  to the local block of the Jacobian of G(t,y,y'). It assumes that
  a band matrix of type SUNMatrix is stored column-wise, and that
  elements within each column are contiguous.

  All matrix elements are generated as difference quotients, by way
  of calls to the user routine glocal. By virtue of the band
  structure, the number of these calls is bandwidth + 1, where
  bandwidth = mldq + mudq + 1. But the band matrix kept has
  bandwidth = mlkeep + mukeep + 1. This routine also assumes that
  the local elements of a vector are stored contiguously.

  Return values are: 0 (success), > 0 (recoverable error),
  or < 0 (nonrecoverable error).
  ----------------------------------------------------------------*/
#[allow(clippy::too_many_arguments)]
fn IBBDDQJac(
    pdata: &mut IBBDPrecDataRec,
    tt: sunrealtype,
    cj: sunrealtype,
    yy: &N_Vector,
    yp: &N_Vector,
    gref: &N_Vector,
    ytemp: &N_Vector,
    yptemp: &N_Vector,
    gtemp: &N_Vector,
) -> i32 {
    let IDA_mem = pdata.ida_mem.clone();

    /* Copy the fields C reads through IDA_mem-> out of the mem
    (granular borrow: nothing is held across callbacks/vector ops) */
    let (hh, ewt, constraints) = {
        let mem = IDA_mem.borrow();
        (
            mem.ida_hh,
            mem.ida_ewt.clone().expect("ida_ewt"),
            mem.ida_constraints.clone(),
        )
    };

    /* Initialize ytemp and yptemp. */
    N_VScale(ONE, yy, ytemp);
    N_VScale(ONE, yp, yptemp);

    /* Obtain pointers as required to the data array of vectors.
    (Rust: the data guards are taken in scoped blocks around each
    component loop — C holds the raw pointers across the glocal
    callbacks, which a RefCell guard may not do.) */

    /* Call gcomm and glocal to get base value of G(t,y,y'). */
    if let Some(gcomm) = pdata.gcomm {
        let retval = {
            let mut user_data = IDA_mem.borrow_mut().ida_user_data.take();
            let retval = gcomm(pdata.n_local, tt, yy, yp, &mut user_data);
            IDA_mem.borrow_mut().ida_user_data = user_data;
            retval
        };
        if retval != 0 {
            return retval;
        }
    }

    let glocal = pdata.glocal;
    let retval = {
        let mut user_data = IDA_mem.borrow_mut().ida_user_data.take();
        let retval = glocal(pdata.n_local, tt, yy, yp, gref, &mut user_data);
        IDA_mem.borrow_mut().ida_user_data = user_data;
        retval
    };
    pdata.nge += 1;
    if retval != 0 {
        return retval;
    }

    /* Set bandwidth and number of column groups for band differencing. */
    let width = pdata.mldq + pdata.mudq + 1;
    let ngroups = SUNMIN(width, pdata.n_local);

    let s_mu = SM_SUBAND_B(&pdata.PP);

    /* Loop over groups. */
    for group in 1..=ngroups {
        /* Loop over the components in this group. */
        {
            let ydata = N_VGetArrayPointer(yy).expect("yy data");
            let ypdata = N_VGetArrayPointer(yp).expect("yp data");
            let ewtdata = N_VGetArrayPointer(&ewt).expect("ewt data");
            let mut ytempdata = N_VGetArrayPointer(ytemp).expect("ytemp data");
            let mut yptempdata = N_VGetArrayPointer(yptemp).expect("yptemp data");
            let cnsdata = constraints
                .as_ref()
                .map(|c| N_VGetArrayPointer(c).expect("constraints data"));

            let mut j = group - 1;
            while j < pdata.n_local {
                let ju = j as usize;
                let yj = ydata[ju];
                let ypj = ypdata[ju];
                let ewtj = ewtdata[ju];

                /* Set increment inc to yj based on rel_yy*abs(yj), with
                adjustments using ypj and ewtj if this is small, and a further
                adjustment to give it the same sign as hh*ypj. */
                let mut inc = pdata.rel_yy
                    * SUNMAX(SUNRabs(yj), SUNMAX(SUNRabs(hh * ypj), ONE / ewtj));
                if hh * ypj < ZERO {
                    inc = -inc;
                }
                inc = (yj + inc) - yj;

                /* Adjust sign(inc) again if yj has an inequality constraint. */
                if let Some(cnsdata) = &cnsdata {
                    let conj = cnsdata[ju];
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

                /* Increment yj and ypj. */
                ytempdata[ju] += inc;
                yptempdata[ju] += cj * inc;
                j += width;
            }
        }

        /* Evaluate G with incremented y and yp arguments. */
        let retval = {
            let mut user_data = IDA_mem.borrow_mut().ida_user_data.take();
            let retval = glocal(pdata.n_local, tt, ytemp, yptemp, gtemp, &mut user_data);
            IDA_mem.borrow_mut().ida_user_data = user_data;
            retval
        };
        pdata.nge += 1;
        if retval != 0 {
            return retval;
        }

        /* Loop over components of the group again; restore ytemp and yptemp. */
        {
            let ydata = N_VGetArrayPointer(yy).expect("yy data");
            let ypdata = N_VGetArrayPointer(yp).expect("yp data");
            let ewtdata = N_VGetArrayPointer(&ewt).expect("ewt data");
            let gtempdata = N_VGetArrayPointer(gtemp).expect("gtemp data");
            let grefdata = N_VGetArrayPointer(gref).expect("gref data");
            let mut ytempdata = N_VGetArrayPointer(ytemp).expect("ytemp data");
            let mut yptempdata = N_VGetArrayPointer(yptemp).expect("yptemp data");
            let cnsdata = constraints
                .as_ref()
                .map(|c| N_VGetArrayPointer(c).expect("constraints data"));

            let mut j = group - 1;
            while j < pdata.n_local {
                let ju = j as usize;
                ytempdata[ju] = ydata[ju];
                let yj = ytempdata[ju];
                yptempdata[ju] = ypdata[ju];
                let ypj = yptempdata[ju];
                let ewtj = ewtdata[ju];

                /* Set increment inc as before .*/
                let mut inc = pdata.rel_yy
                    * SUNMAX(SUNRabs(yj), SUNMAX(SUNRabs(hh * ypj), ONE / ewtj));
                if hh * ypj < ZERO {
                    inc = -inc;
                }
                inc = (yj + inc) - yj;
                if let Some(cnsdata) = &cnsdata {
                    let conj = cnsdata[ju];
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

                /* Form difference quotients and load into PP. */
                let inc_inv = ONE / inc;
                let mut col_j = SUNBandMatrix_Column(&pdata.PP, j);
                let i1 = SUNMAX(0, j - pdata.mukeep);
                let i2 = SUNMIN(j + pdata.mlkeep, pdata.n_local - 1);
                let mut i = i1;
                while i <= i2 {
                    col_j[SM_COLUMN_ELEMENT_IDX(i, j, s_mu)] =
                        inc_inv * (gtempdata[i as usize] - grefdata[i as usize]);
                    i += 1;
                }
                drop(col_j);
                j += width;
            }
        }
    }

    0
}
