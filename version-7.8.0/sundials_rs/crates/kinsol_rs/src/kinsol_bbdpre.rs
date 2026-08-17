//! Port of `src/kinsol/kinsol_bbdpre.c` (+ `src/kinsol/kinsol_bbdpre_impl.h`
//! and `include/kinsol/kinsol_bbdpre.h` folded).
//!
//! Band-block-diagonal preconditioner (a block-diagonal matrix with
//! banded blocks) for use with KINSol and the KINLS linear solver
//! interface. With only one process a plain banded matrix results —
//! diagonal blocking occurs at the process level — so the serial build
//! ported here is the single-block case. The upstream file is written
//! against the MPI-parallel NVECTOR but contains no MPI `#ifdef`s: it
//! wraps the process-local data in serial vectors exactly as ported
//! here (`zlocal` = `N_VNew_Serial`, i.e. it owns its own buffer;
//! `rlocal` = `N_VNewEmpty_Serial`, i.e. the caller's buffer is
//! attached to it for the duration of a solve).
//!
//! The preconditioner data lives in `kinls_mem.pdata`
//! (`Option<Box<dyn Any>>` holding a [`KBBDPrecDataRec`]); the KINLS
//! interface (`kinsol_ls`) `Option::take`s that box around each
//! pset/psolve invocation, so the callbacks here receive it as
//! `&mut Option<Box<dyn Any>>` and downcast. `pdata == None` is the
//! "pass `kin_user_data`" (user-supplied preconditioner) case, which is
//! why the getters below report `KINLS_PMEM_NULL` when the downcast
//! fails, exactly as C reports it for a NULL `pdata`.
//!
//! Attachment of the KINLS interface itself is probed non-panickingly
//! (`b.is::<KINLsMemRec>()`) so the `KINLS_LMEM_NULL` paths return the
//! C flag instead of unwinding.

use std::any::Any;

use crate::kinsol_impl::*;
use crate::kinsol_ls::{
    kinls_mem_mut, KINLsMemRec, KINSetPreconditioner, KINLS_ILL_INPUT, KINLS_LMEM_NULL,
    KINLS_MEM_FAIL, KINLS_PMEM_NULL, KINLS_SUCCESS, KINLS_SUNLS_FAIL,
};
use sundials_core::nvector_serial::{N_VNewEmpty_Serial, N_VNew_Serial};
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

/* -----------------------------------------------------------------
 * KINBBDPRE return values (include/kinsol/kinsol_bbdpre.h)
 * ----------------------------------------------------------------- */

pub const KINBBDPRE_SUCCESS: i32 = 0;
pub const KINBBDPRE_PDATA_NULL: i32 = -11;
pub const KINBBDPRE_FUNC_UNRECVR: i32 = -12;

/* -----------------------------------------------------------------
 * User-supplied function types (include/kinsol/kinsol_bbdpre.h)
 * ----------------------------------------------------------------- */

pub type KINBBDCommFn =
    fn(Nlocal: sunindextype, u: &N_Vector, user_data: &mut Option<Box<dyn Any>>) -> i32;

pub type KINBBDLocalFn = fn(
    Nlocal: sunindextype,
    uu: &N_Vector,
    gval: &N_Vector,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32;

/*------------------------------------------------------------------
  Definition of KBBDData (kinsol_bbdpre_impl.h)
  ------------------------------------------------------------------*/

pub struct KBBDPrecDataRec {
    /* passed by user to KINBBDPrecAlloc, used by pset/psolve functions */
    pub mudq: sunindextype,
    pub mldq: sunindextype,
    pub mukeep: sunindextype,
    pub mlkeep: sunindextype,
    pub rel_uu: sunrealtype, /* relative error for the Jacobian DQ routine */
    pub gloc: KINBBDLocalFn,
    pub gcomm: Option<KINBBDCommFn>,

    /* set by KINBBDPrecSetup and used by KINBBDPrecSetup and
    KINBBDPrecSolve functions */
    pub n_local: sunindextype,
    pub PP: SUNMatrix,
    pub LS: SUNLinearSolver,
    pub rlocal: N_Vector,
    pub zlocal: N_Vector,
    pub tempv1: N_Vector,
    pub tempv2: N_Vector,
    pub tempv3: N_Vector,

    /* available for optional output */
    pub rpwsize: i64,
    pub ipwsize: i64,
    pub nge: i64,

    /* pointer to KINSol memory */
    pub kin_mem: KINMem,
}

pub type KBBDPrecData = Box<KBBDPrecDataRec>;

/*
 *-----------------------------------------------------------------
 * KINBBDPRE error messages (kinsol_bbdpre_impl.h)
 *-----------------------------------------------------------------
 */

pub const MSGBBD_MEM_NULL: &str = "KINSOL Memory is NULL.";
pub const MSGBBD_LMEM_NULL: &str =
    "Linear solver memory is NULL. One of the SPILS linear solvers must be attached.";
pub const MSGBBD_MEM_FAIL: &str = "A memory request failed.";
pub const MSGBBD_BAD_NVECTOR: &str = "A required vector operation is not implemented.";
pub const MSGBBD_SUNMAT_FAIL: &str = "An error arose from a SUNBandMatrix routine.";
pub const MSGBBD_SUNLS_FAIL: &str = "An error arose from a SUNBandLinearSolver routine.";
pub const MSGBBD_PMEM_NULL: &str =
    "BBD peconditioner memory is NULL. IDABBDPrecInit must be called.";
pub const MSGBBD_FUNC_FAILED: &str = "The gloc or gcomm routine failed in an unrecoverable manner.";

/*------------------------------------------------------------------
  user-callable functions
  ------------------------------------------------------------------*/

/*------------------------------------------------------------------
  KINBBDPrecInit
  ------------------------------------------------------------------*/
#[allow(clippy::too_many_arguments)]
pub fn KINBBDPrecInit(
    kinmem: &KINMem,
    Nlocal: sunindextype,
    mudq: sunindextype,
    mldq: sunindextype,
    mukeep: sunindextype,
    mlkeep: sunindextype,
    dq_rel_uu: sunrealtype,
    gloc: KINBBDLocalFn,
    gcomm: Option<KINBBDCommFn>,
) -> i32 {
    /* NULL-mem check: handled by the type system */
    let kin_mem = kinmem;

    /* Test if the LS linear solver interface has been created */
    let attached = {
        let mem = kin_mem.borrow();
        mem.kin_lmem.as_ref().is_some_and(|b| b.is::<KINLsMemRec>())
    };
    if !attached {
        KINProcessError(
            Some(kin_mem),
            KINLS_LMEM_NULL,
            line!() as i32,
            "KINBBDPrecInit",
            file!(),
            MSGBBD_LMEM_NULL,
        );
        return KINLS_LMEM_NULL;
    }

    /* Test compatibility of NVECTOR package with the BBD preconditioner */
    /* Note: Do NOT need to check for N_VScale since has already been checked for in KINSOL */
    let (vtemp1, sunctx) = {
        let mem = kin_mem.borrow();
        (
            mem.kin_vtemp1.clone().expect("kin_vtemp1"),
            mem.kin_sunctx.clone(),
        )
    };
    if vtemp1.ops.borrow().nvgetarraypointer.is_none() {
        KINProcessError(
            Some(kin_mem),
            KINLS_ILL_INPUT,
            line!() as i32,
            "KINBBDPrecInit",
            file!(),
            MSGBBD_BAD_NVECTOR,
        );
        return KINLS_ILL_INPUT;
    }

    /* Allocate data memory (Rust: the record is assembled at the end of
    this function; the C malloc-NULL branch has no analogue) */

    /* Set pointers to gloc and gcomm; load half-bandwidths */
    let mudq = SUNMIN(Nlocal - 1, SUNMAX(0, mudq));
    let mldq = SUNMIN(Nlocal - 1, SUNMAX(0, mldq));
    let muk = SUNMIN(Nlocal - 1, SUNMAX(0, mukeep));
    let mlk = SUNMIN(Nlocal - 1, SUNMAX(0, mlkeep));

    /* Set extended upper half-bandwidth for PP (required for pivoting) */
    let storage_mu = SUNMIN(Nlocal - 1, muk + mlk);

    /* Allocate memory for preconditioner matrix */
    let PP = match SUNBandMatrixStorage(Nlocal, muk, mlk, storage_mu, &sunctx) {
        None => {
            KINProcessError(
                Some(kin_mem),
                KINLS_MEM_FAIL,
                line!() as i32,
                "KINBBDPrecInit",
                file!(),
                MSGBBD_MEM_FAIL,
            );
            return KINLS_MEM_FAIL;
        }
        Some(m) => m,
    };

    /* Allocate memory for temporary N_Vectors */
    let zlocal = match N_VNew_Serial(Nlocal, &sunctx) {
        None => {
            KINProcessError(
                Some(kin_mem),
                KINLS_MEM_FAIL,
                line!() as i32,
                "KINBBDPrecInit",
                file!(),
                MSGBBD_MEM_FAIL,
            );
            return KINLS_MEM_FAIL;
        }
        Some(v) => v,
    };

    let rlocal = match N_VNewEmpty_Serial(Nlocal, &sunctx) {
        /* empty vector */
        None => {
            KINProcessError(
                Some(kin_mem),
                KINLS_MEM_FAIL,
                line!() as i32,
                "KINBBDPrecInit",
                file!(),
                MSGBBD_MEM_FAIL,
            );
            return KINLS_MEM_FAIL;
        }
        Some(v) => v,
    };

    let tempv1 = match N_VClone(&vtemp1) {
        None => {
            KINProcessError(
                Some(kin_mem),
                KINLS_MEM_FAIL,
                line!() as i32,
                "KINBBDPrecInit",
                file!(),
                MSGBBD_MEM_FAIL,
            );
            return KINLS_MEM_FAIL;
        }
        Some(v) => v,
    };

    let tempv2 = match N_VClone(&vtemp1) {
        None => {
            KINProcessError(
                Some(kin_mem),
                KINLS_MEM_FAIL,
                line!() as i32,
                "KINBBDPrecInit",
                file!(),
                MSGBBD_MEM_FAIL,
            );
            return KINLS_MEM_FAIL;
        }
        Some(v) => v,
    };

    let tempv3 = match N_VClone(&vtemp1) {
        None => {
            KINProcessError(
                Some(kin_mem),
                KINLS_MEM_FAIL,
                line!() as i32,
                "KINBBDPrecInit",
                file!(),
                MSGBBD_MEM_FAIL,
            );
            return KINLS_MEM_FAIL;
        }
        Some(v) => v,
    };

    /* Allocate memory for banded linear solver */
    let LS = match SUNLinSol_Band(&zlocal, &PP, &sunctx) {
        None => {
            KINProcessError(
                Some(kin_mem),
                KINLS_MEM_FAIL,
                line!() as i32,
                "KINBBDPrecInit",
                file!(),
                MSGBBD_MEM_FAIL,
            );
            return KINLS_MEM_FAIL;
        }
        Some(s) => s,
    };

    /* initialize band linear solver object */
    let flag = SUNLinSolInitialize(&LS);
    if flag != SUN_SUCCESS {
        KINProcessError(
            Some(kin_mem),
            KINLS_SUNLS_FAIL,
            line!() as i32,
            "KINBBDPrecInit",
            file!(),
            MSGBBD_SUNLS_FAIL,
        );
        return KINLS_SUNLS_FAIL;
    }

    /* Set rel_uu based on input value dq_rel_uu (0 implies default) */
    let rel_uu = if dq_rel_uu > ZERO {
        dq_rel_uu
    } else {
        SUNRsqrt(kin_mem.borrow().kin_uround)
    };

    /* Store Nlocal to be used in KINBBDPrecSetup */
    let n_local = Nlocal;

    /* Set work space sizes and initialize nge */
    let mut rpwsize: i64 = 0;
    let mut ipwsize: i64 = 0;
    if vtemp1.ops.borrow().nvspace.is_some() {
        let mut lrw1: sunindextype = 0;
        let mut liw1: sunindextype = 0;
        N_VSpace(&vtemp1, &mut lrw1, &mut liw1);
        rpwsize += 3 * lrw1;
        ipwsize += 3 * liw1;
    }
    if zlocal.ops.borrow().nvspace.is_some() {
        let mut lrw1: sunindextype = 0;
        let mut liw1: sunindextype = 0;
        N_VSpace(&zlocal, &mut lrw1, &mut liw1);
        rpwsize += lrw1;
        ipwsize += liw1;
    }
    if rlocal.ops.borrow().nvspace.is_some() {
        let mut lrw1: sunindextype = 0;
        let mut liw1: sunindextype = 0;
        N_VSpace(&rlocal, &mut lrw1, &mut liw1);
        rpwsize += lrw1;
        ipwsize += liw1;
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
    let pfree = kinls_mem_mut(kin_mem).pfree;
    if let Some(pfree) = pfree {
        pfree(kin_mem);
    }

    {
        let mut ls_mem = kinls_mem_mut(kin_mem);

        /* Point to the new pdata field in the LS memory */
        ls_mem.pdata = Some(Box::new(KBBDPrecDataRec {
            mudq,
            mldq,
            mukeep: muk,
            mlkeep: mlk,
            rel_uu,
            gloc,
            gcomm,
            n_local,
            PP,
            LS,
            rlocal,
            zlocal,
            tempv1,
            tempv2,
            tempv3,
            rpwsize,
            ipwsize,
            nge,
            kin_mem: kin_mem.clone(),
        }));

        /* Attach the pfree function */
        ls_mem.pfree = Some(KINBBDPrecFree);
    }

    /* Attach preconditioner solve and setup functions */
    KINSetPreconditioner(kinmem, Some(KINBBDPrecSetup), Some(KINBBDPrecSolve))
}

/*------------------------------------------------------------------
  KINBBDPrecGetWorkSpace
  ------------------------------------------------------------------*/
pub fn KINBBDPrecGetWorkSpace(kinmem: &KINMem, lenrwBBDP: &mut i64, leniwBBDP: &mut i64) -> i32 {
    /* NULL-mem check: handled by the type system */
    let kin_mem = kinmem;

    let attached = {
        let mem = kin_mem.borrow();
        mem.kin_lmem.as_ref().is_some_and(|b| b.is::<KINLsMemRec>())
    };
    if !attached {
        KINProcessError(
            Some(kin_mem),
            KINLS_LMEM_NULL,
            line!() as i32,
            "KINBBDPrecGetWorkSpace",
            file!(),
            MSGBBD_LMEM_NULL,
        );
        return KINLS_LMEM_NULL;
    }

    let sizes = {
        let ls_mem = kinls_mem_mut(kin_mem);
        ls_mem
            .pdata
            .as_ref()
            .and_then(|b| b.downcast_ref::<KBBDPrecDataRec>())
            .map(|p| (p.rpwsize, p.ipwsize))
    };
    match sizes {
        None => {
            KINProcessError(
                Some(kin_mem),
                KINLS_PMEM_NULL,
                line!() as i32,
                "KINBBDPrecGetWorkSpace",
                file!(),
                MSGBBD_PMEM_NULL,
            );
            KINLS_PMEM_NULL
        }
        Some((rpwsize, ipwsize)) => {
            *lenrwBBDP = rpwsize;
            *leniwBBDP = ipwsize;
            KINLS_SUCCESS
        }
    }
}

/*------------------------------------------------------------------
 KINBBDPrecGetNumGfnEvals
 -------------------------------------------------------------------*/
pub fn KINBBDPrecGetNumGfnEvals(kinmem: &KINMem, ngevalsBBDP: &mut i64) -> i32 {
    /* NULL-mem check: handled by the type system */
    let kin_mem = kinmem;

    let attached = {
        let mem = kin_mem.borrow();
        mem.kin_lmem.as_ref().is_some_and(|b| b.is::<KINLsMemRec>())
    };
    if !attached {
        KINProcessError(
            Some(kin_mem),
            KINLS_LMEM_NULL,
            line!() as i32,
            "KINBBDPrecGetNumGfnEvals",
            file!(),
            MSGBBD_LMEM_NULL,
        );
        return KINLS_LMEM_NULL;
    }

    let nge = {
        let ls_mem = kinls_mem_mut(kin_mem);
        ls_mem
            .pdata
            .as_ref()
            .and_then(|b| b.downcast_ref::<KBBDPrecDataRec>())
            .map(|p| p.nge)
    };
    match nge {
        None => {
            KINProcessError(
                Some(kin_mem),
                KINLS_PMEM_NULL,
                line!() as i32,
                "KINBBDPrecGetNumGfnEvals",
                file!(),
                MSGBBD_PMEM_NULL,
            );
            KINLS_PMEM_NULL
        }
        Some(nge) => {
            *ngevalsBBDP = nge;
            KINLS_SUCCESS
        }
    }
}

/*------------------------------------------------------------------
  KINBBDPrecSetup

  KINBBDPrecSetup generates and factors a banded block of the
  preconditioner matrix on each processor, via calls to the
  user-supplied gloc and gcomm functions. It uses difference
  quotient approximations to the Jacobian elements.

  KINBBDPrecSetup calculates a new Jacobian, stored in banded
  matrix PP and does an LU factorization of P in place in PP.

  The parameters of KINBBDPrecSetup are as follows:

  uu      is the current value of the dependent variable vector,
          namely the solutin to func(uu)=0

  uscale  is the dependent variable scaling vector (i.e. uu)

  fval    is the vector f(u)

  fscale  is the function scaling vector

  bbd_data is the pointer to BBD data set by KINBBDInit.

  Note: The value to be returned by the KINBBDPrecSetup function
  is a flag indicating whether it was successful. This value is:
    0 if successful,
    > 0 for a recoverable error - step will be retried.
  ------------------------------------------------------------------*/
fn KINBBDPrecSetup(
    uu: &N_Vector,
    uscale: &N_Vector,
    _fval: &N_Vector,
    _fscale: &N_Vector,
    bbd_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    let pdata: &mut KBBDPrecDataRec = bbd_data
        .as_mut()
        .and_then(|b| b.downcast_mut::<KBBDPrecDataRec>())
        .expect("bbd_data is KBBDPrecData");

    let kin_mem = pdata.kin_mem.clone();

    /* Call KBBDDQJac for a new Jacobian calculation and store in PP */
    let retval = SUNMatZero(&pdata.PP);
    if retval != 0 {
        KINProcessError(
            Some(&kin_mem),
            -1,
            line!() as i32,
            "KINBBDPrecSetup",
            file!(),
            MSGBBD_SUNMAT_FAIL,
        );
        return -1;
    }

    let tempv1 = pdata.tempv1.clone();
    let tempv2 = pdata.tempv2.clone();
    let tempv3 = pdata.tempv3.clone();
    let retval = KBBDDQJac(pdata, uu, uscale, &tempv1, &tempv2, &tempv3);
    if retval != 0 {
        KINProcessError(
            Some(&kin_mem),
            -1,
            line!() as i32,
            "KINBBDPrecSetup",
            file!(),
            MSGBBD_FUNC_FAILED,
        );
        return -1;
    }

    /* Do LU factorization of P and return error flag */
    SUNLinSolSetup_Band(&pdata.LS, Some(&pdata.PP))
}

/*------------------------------------------------------------------
  INBBDPrecSolve

  KINBBDPrecSolve solves a linear system P z = r, with the
  banded blocked preconditioner matrix P generated and factored
  by KINBBDPrecSetup. Here, r comes in as vv and z is
  returned in vv as well.

  The parameters for KINBBDPrecSolve are as follows:

  uu     an N_Vector giving the current iterate for the system

  uscale an N_Vector giving the diagonal entries of the
         uu scaling matrix

  fval   an N_Vector giving the current function value

  fscale an N_Vector giving the diagonal entries of the
         function scaling matrix

   vv  vector initially set to the right-hand side vector r, but
       which upon return contains a solution of the linear system
       P*z = r

  bbd_data is the pointer to BBD data set by KINBBDInit.

  Note: The value returned by the KINBBDPrecSolve function is a
  flag returned from the lienar solver object.
  ------------------------------------------------------------------*/

fn KINBBDPrecSolve(
    _uu: &N_Vector,
    _uscale: &N_Vector,
    _fval: &N_Vector,
    _fscale: &N_Vector,
    vv: &N_Vector,
    bbd_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    let pdata: &KBBDPrecDataRec = bbd_data
        .as_ref()
        .and_then(|b| b.downcast_ref::<KBBDPrecDataRec>())
        .expect("bbd_data is KBBDPrecData");

    /* Attach local data array for vv to rlocal (Rust: move the owned
    buffer into the empty local wrapper; C aliases the raw pointer, so
    the `vd[i] = zd[i]` copy below writes straight through into vv) */
    let vd = {
        let mut g = N_VGetArrayPointer(vv).expect("vv data");
        std::mem::take(&mut *g)
    };
    N_VSetArrayPointer(vd, &pdata.rlocal);

    /* Call banded solver object to do the work */
    let retval = SUNLinSolSolve(
        &pdata.LS,
        Some(&pdata.PP),
        &pdata.zlocal,
        &pdata.rlocal,
        ZERO,
    );

    /* Copy result into vv (the buffer currently held by rlocal IS vv's
    own buffer: take it back, copy, and return it to vv; C leaves
    rlocal aliasing vv's data, the Rust wrapper is left empty) */
    let mut vd = {
        let mut g = N_VGetArrayPointer(&pdata.rlocal).expect("rlocal data");
        std::mem::take(&mut *g)
    };
    {
        let zd = N_VGetArrayPointer(&pdata.zlocal).expect("zlocal data");
        let mut i: sunindextype = 0;
        while i < pdata.n_local {
            vd[i as usize] = zd[i as usize];
            i += 1;
        }
    }
    N_VSetArrayPointer(vd, vv);

    retval
}

/*------------------------------------------------------------------
  KINBBDPrecFree
  ------------------------------------------------------------------*/
fn KINBBDPrecFree(kin_mem: &KINMem) -> i32 {
    let attached = {
        let mem = kin_mem.borrow();
        mem.kin_lmem.as_ref().is_some_and(|b| b.is::<KINLsMemRec>())
    };
    if !attached {
        return 0;
    }

    let pdata = kinls_mem_mut(kin_mem).pdata.take();
    if pdata.is_none() {
        return 0;
    }

    /* SUNLinSolFree(LS) / N_VDestroy(zlocal, rlocal, tempv1..3) /
    SUNMatDestroy(PP): dropping the record releases everything (C frees
    each explicitly and leaves kinls_mem->pdata dangling; the Rust
    take() leaves it None) */
    drop(pdata);

    0
}

/*------------------------------------------------------------------
  KBBDDQJac

  This routine generates a banded difference quotient
  approximation to the Jacobian of f(u). It assumes that a band
  matrix of type SUNMatrix is stored column-wise, and that elements
  within each column are contiguous. All matrix elements are
  generated as difference quotients, by way of calls to the user
  routine gloc. By virtue of the band structure, the number of
  these calls is bandwidth + 1, where bandwidth = ml + mu + 1.
  This routine also assumes that the local elements of a vector
  are stored contiguously.
  ------------------------------------------------------------------*/
fn KBBDDQJac(
    pdata: &mut KBBDPrecDataRec,
    uu: &N_Vector,
    uscale: &N_Vector,
    gu: &N_Vector,
    gtemp: &N_Vector,
    utemp: &N_Vector,
) -> i32 {
    let kin_mem = pdata.kin_mem.clone();

    /* load utemp with uu = predicted solution vector */
    N_VScale(ONE, uu, utemp);

    /* set pointers to the data for all vectors (Rust: the guards are
    re-taken inside each block below so none is held across a user
    callback; no vector's buffer moves in between) */

    /* Call gcomm and gloc to get base value of g(uu) */
    if let Some(gcomm) = pdata.gcomm {
        let retval = {
            let mut user_data = kin_mem.borrow_mut().kin_user_data.take();
            let retval = gcomm(pdata.n_local, uu, &mut user_data);
            kin_mem.borrow_mut().kin_user_data = user_data;
            retval
        };
        if retval != 0 {
            return retval;
        }
    }

    let gloc = pdata.gloc;
    let retval = {
        let mut user_data = kin_mem.borrow_mut().kin_user_data.take();
        let retval = gloc(pdata.n_local, uu, gu, &mut user_data);
        kin_mem.borrow_mut().kin_user_data = user_data;
        retval
    };
    pdata.nge += 1;
    if retval != 0 {
        return retval;
    }

    /* Set bandwidth and number of column groups for band differencing */
    let width = pdata.mldq + pdata.mudq + 1;
    let ngroups = SUNMIN(width, pdata.n_local);

    let s_mu = SM_SUBAND_B(&pdata.PP);

    /* Loop over groups */
    for group in 1..=ngroups {
        /* increment all u_j in group */
        {
            let udata = N_VGetArrayPointer(uu).expect("uu data");
            let uscdata = N_VGetArrayPointer(uscale).expect("uscale data");
            let mut utempdata = N_VGetArrayPointer(utemp).expect("utemp data");

            let mut j = group - 1;
            while j < pdata.n_local {
                let ju = j as usize;
                let inc = pdata.rel_uu * SUNMAX(SUNRabs(udata[ju]), ONE / uscdata[ju]);
                utempdata[ju] += inc;
                j += width;
            }
        }

        /* Evaluate g with incremented u */
        let retval = {
            let mut user_data = kin_mem.borrow_mut().kin_user_data.take();
            let retval = gloc(pdata.n_local, utemp, gtemp, &mut user_data);
            kin_mem.borrow_mut().kin_user_data = user_data;
            retval
        };
        pdata.nge += 1;
        if retval != 0 {
            return retval;
        }

        /* restore utemp, then form and load difference quotients */
        {
            let udata = N_VGetArrayPointer(uu).expect("uu data");
            let uscdata = N_VGetArrayPointer(uscale).expect("uscale data");
            let gudata = N_VGetArrayPointer(gu).expect("gu data");
            let gtempdata = N_VGetArrayPointer(gtemp).expect("gtemp data");
            let mut utempdata = N_VGetArrayPointer(utemp).expect("utemp data");

            let mut j = group - 1;
            while j < pdata.n_local {
                let ju = j as usize;
                utempdata[ju] = udata[ju];
                let mut col_j = SUNBandMatrix_Column(&pdata.PP, j);
                let inc = pdata.rel_uu * SUNMAX(SUNRabs(udata[ju]), ONE / uscdata[ju]);
                let inc_inv = ONE / inc;
                let i1 = SUNMAX(0, j - pdata.mukeep);
                let i2 = SUNMIN(j + pdata.mlkeep, pdata.n_local - 1);
                let mut i = i1;
                while i <= i2 {
                    col_j[SM_COLUMN_ELEMENT_IDX(i, j, s_mu)] =
                        inc_inv * (gtempdata[i as usize] - gudata[i as usize]);
                    i += 1;
                }
                drop(col_j);
                j += width;
            }
        }
    }

    0
}
