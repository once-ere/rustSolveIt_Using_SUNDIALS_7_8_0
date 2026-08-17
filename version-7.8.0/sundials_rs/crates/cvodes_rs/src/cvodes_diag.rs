//! Port of `src/cvodes/cvodes_diag.c` (+ `src/cvodes/cvodes_diag_impl.h` and
//! `include/cvodes/cvodes_diag.h` folded).
//!
//! CVDIAG is the CVODES-specific diagonal linear solver. Its memory record
//! (`CVDiagMemRec`) lives in `cv_mem.cv_lmem` as `Box<dyn Any>`; the
//! `CVDiag` attach routine wires `CVDiagInit`/`CVDiagSetup`/`CVDiagSolve`/
//! `CVDiagFree` into the integrator's `cv_linit`/`cv_lreinit`/`cv_lsetup`/
//! `cv_lsolve`/`cv_lfree` fn-pointer slots.
//!
//! Relative to the CVODE version this file adds PART II (backward
//! problems): `CVDiagB` looks the backward problem up by index and attaches
//! CVDIAG to its integrator memory. The fused-kernel branches of
//! `cvode_diag.c` do not exist upstream here (CVODES ships no fused
//! kernels), so the plain N_Vector sequences are the only path.

use std::cell::RefMut;

use crate::cvodes_impl::*;
use sundials_core::sundials_nvector::*;
use sundials_core::sundials_types::*;

/* ---------------------
 * CVDIAG return values (include/cvodes/cvodes_diag.h)
 * --------------------- */

pub const CVDIAG_SUCCESS: i32 = 0;
pub const CVDIAG_MEM_NULL: i32 = -1;
pub const CVDIAG_LMEM_NULL: i32 = -2;
pub const CVDIAG_ILL_INPUT: i32 = -3;
pub const CVDIAG_MEM_FAIL: i32 = -4;

/* Additional last_flag values */

pub const CVDIAG_INV_FAIL: i32 = -5;
pub const CVDIAG_RHSFUNC_UNRECVR: i32 = -6;
pub const CVDIAG_RHSFUNC_RECVR: i32 = -7;

/* Return values for adjoint module */

pub const CVDIAG_NO_ADJ: i32 = -101;

/* Other Constants */

const FRACT: sunrealtype = 0.1;
const ONE: sunrealtype = 1.0;

/*
 * -----------------------------------------------------------------
 * Types: CVDiagMemRec, CVDiagMem (cvodes_diag_impl.h)
 * -----------------------------------------------------------------
 * This structure contains CVDiag solver-specific data.
 * -----------------------------------------------------------------
 */

pub struct CVDiagMemRec {
    pub di_gammasv: sunrealtype, /* gammasv = gamma at the last call to setup
                                 or solve                                  */

    pub di_M: N_Vector, /* M = (I - gamma J)^{-1} , gamma = h / l1   */

    pub di_bit: N_Vector, /* temporary storage vector                  */

    pub di_bitcomp: N_Vector, /* temporary storage vector                  */

    pub di_nfeDI: i64, /* no. of calls to f due to difference
                       quotient diagonal Jacobian approximation  */

    pub di_last_flag: i64, /* last error return flag                  */
}

pub type CVDiagMem = Box<CVDiagMemRec>;

/* Error Messages (cvodes_diag_impl.h) */

pub const MSGDG_CVMEM_NULL: &str = "Integrator memory is NULL.";
pub const MSGDG_MEM_FAIL: &str = "A memory request failed.";
pub const MSGDG_BAD_NVECTOR: &str = "A required vector operation is not implemented.";
pub const MSGDG_LMEM_NULL: &str = "CVDIAG memory is NULL.";
pub const MSGDG_RHSFUNC_FAILED: &str =
    "The right-hand side routine failed in an unrecoverable manner.";

pub const MSGDG_NO_ADJ: &str = "Illegal attempt to call before calling CVodeAdjMalloc.";
pub const MSGDG_BAD_WHICH: &str = "Illegal value for which.";

/* C `(CVDiagMem)cv_mem->cv_lmem` downcast */
fn cvdiag_mem_mut(cv_mem: &CVodeMem) -> RefMut<'_, CVDiagMemRec> {
    RefMut::map(cv_mem.borrow_mut(), |m| {
        m.cv_lmem
            .as_mut()
            .expect("cvdiag_mem")
            .downcast_mut::<CVDiagMemRec>()
            .expect("CVDIAG lmem")
    })
}

/*
 * ================================================================
 *
 *                   PART I - forward problems
 *
 * ================================================================
 */

/*
 * -----------------------------------------------------------------
 * CVDiag
 * -----------------------------------------------------------------
 * This routine initializes the memory record and sets various function
 * fields specific to the diagonal linear solver module.  CVDense first
 * calls the existing lfree routine if this is not NULL.  Then it sets
 * the cv_linit, cv_lsetup, cv_lsolve, cv_lfree fields in (*cvode_mem)
 * to be CVDiagInit, CVDiagSetup, CVDiagSolve, and CVDiagFree,
 * respectively.  It allocates memory for a structure of type
 * CVDiagMemRec and sets the cv_lmem field in (*cvode_mem) to the
 * address of this structure.  It sets setupNonNull in (*cvode_mem) to
 * SUNTRUE.  Finally, it allocates memory for M, bit, and bitcomp.
 * The CVDiag return value is SUCCESS = 0, LMEM_FAIL = -1, or
 * LIN_ILL_INPUT=-2.
 * -----------------------------------------------------------------
 */

pub fn CVDiag(cvode_mem: &CVodeMem) -> i32 {
    /* NULL-mem check: handled by type system */
    let cv_mem = cvode_mem;

    /* Check if N_VCompare and N_VInvTest are present */
    let tempv = cv_mem.borrow().cv_tempv.as_ref().expect("cv_tempv").clone();
    {
        let ops = tempv.ops.borrow();
        if ops.nvcompare.is_none() || ops.nvinvtest.is_none() {
            drop(ops);
            cvProcessError(
                Some(cv_mem),
                CVDIAG_ILL_INPUT,
                line!() as i32,
                "CVDiag",
                file!(),
                MSGDG_BAD_NVECTOR,
            );
            return CVDIAG_ILL_INPUT;
        }
    }

    let lfree = cv_mem.borrow().cv_lfree;
    if let Some(lfree) = lfree {
        lfree(cv_mem);
    }

    /* Set four main function fields in cv_mem */
    {
        let mut m = cv_mem.borrow_mut();
        m.cv_linit = Some(CVDiagInit);
        m.cv_lreinit = Some(CVDiagInit);
        m.cv_lsetup = Some(CVDiagSetup);
        m.cv_lsolve = Some(CVDiagSolve);
        m.cv_lfree = Some(CVDiagFree);
    }

    /* Get memory for CVDiagMemRec
    (C malloc-failure path unreachable: Rust allocation aborts, not NULL) */

    /* Allocate memory for M, bit, and bitcomp */

    let di_M = match N_VClone(&tempv) {
        Some(v) => v,
        None => {
            cvProcessError(
                Some(cv_mem),
                CVDIAG_MEM_FAIL,
                line!() as i32,
                "CVDiag",
                file!(),
                MSGDG_MEM_FAIL,
            );
            return CVDIAG_MEM_FAIL;
        }
    };

    let di_bit = match N_VClone(&tempv) {
        Some(v) => v,
        None => {
            cvProcessError(
                Some(cv_mem),
                CVDIAG_MEM_FAIL,
                line!() as i32,
                "CVDiag",
                file!(),
                MSGDG_MEM_FAIL,
            );
            /* di_M dropped here (C: N_VDestroy(di_M); free(cvdiag_mem)) */
            return CVDIAG_MEM_FAIL;
        }
    };

    let di_bitcomp = match N_VClone(&tempv) {
        Some(v) => v,
        None => {
            cvProcessError(
                Some(cv_mem),
                CVDIAG_MEM_FAIL,
                line!() as i32,
                "CVDiag",
                file!(),
                MSGDG_MEM_FAIL,
            );
            /* di_M, di_bit dropped here (C: N_VDestroy both; free(cvdiag_mem)) */
            return CVDIAG_MEM_FAIL;
        }
    };

    let cvdiag_mem = CVDiagMemRec {
        di_gammasv: 0.0, /* C leaves this uninitialized; always written by
                         CVDiagSetup before its first read in CVDiagSolve */
        di_M,
        di_bit,
        di_bitcomp,
        di_nfeDI: 0,
        di_last_flag: CVDIAG_SUCCESS as i64,
    };

    /* Attach linear solver memory to integrator memory */
    cv_mem.borrow_mut().cv_lmem = Some(Box::new(cvdiag_mem));

    CVDIAG_SUCCESS
}

/*
 * -----------------------------------------------------------------
 * CVDiagGetWorkSpace
 * -----------------------------------------------------------------
 */

pub fn CVDiagGetWorkSpace(cvode_mem: &CVodeMem, lenrwLS: &mut i64, leniwLS: &mut i64) -> i32 {
    /* NULL-mem check: handled by type system */
    let cv_mem = cvode_mem;

    let m = cv_mem.borrow();
    *lenrwLS = 3 * m.cv_lrw1;
    *leniwLS = 3 * m.cv_liw1;

    CVDIAG_SUCCESS
}

/*
 * -----------------------------------------------------------------
 * CVDiagGetNumRhsEvals
 * -----------------------------------------------------------------
 */

pub fn CVDiagGetNumRhsEvals(cvode_mem: &CVodeMem, nfevalsLS: &mut i64) -> i32 {
    /* NULL-mem check: handled by type system */
    let cv_mem = cvode_mem;

    if cv_mem.borrow().cv_lmem.is_none() {
        cvProcessError(
            Some(cv_mem),
            CVDIAG_LMEM_NULL,
            line!() as i32,
            "CVDiagGetNumRhsEvals",
            file!(),
            MSGDG_LMEM_NULL,
        );
        return CVDIAG_LMEM_NULL;
    }

    *nfevalsLS = cvdiag_mem_mut(cv_mem).di_nfeDI;

    CVDIAG_SUCCESS
}

/*
 * -----------------------------------------------------------------
 * CVDiagGetLastFlag
 * -----------------------------------------------------------------
 */

pub fn CVDiagGetLastFlag(cvode_mem: &CVodeMem, flag: &mut i64) -> i32 {
    /* NULL-mem check: handled by type system */
    let cv_mem = cvode_mem;

    if cv_mem.borrow().cv_lmem.is_none() {
        cvProcessError(
            Some(cv_mem),
            CVDIAG_LMEM_NULL,
            line!() as i32,
            "CVDiagGetLastFlag",
            file!(),
            MSGDG_LMEM_NULL,
        );
        return CVDIAG_LMEM_NULL;
    }

    *flag = cvdiag_mem_mut(cv_mem).di_last_flag;

    CVDIAG_SUCCESS
}

/*
 * -----------------------------------------------------------------
 * CVDiagGetReturnFlagName
 * -----------------------------------------------------------------
 */

pub fn CVDiagGetReturnFlagName(flag: i64) -> String {
    let name = if flag == CVDIAG_SUCCESS as i64 {
        "CVDIAG_SUCCESS"
    } else if flag == CVDIAG_MEM_NULL as i64 {
        "CVDIAG_MEM_NULL"
    } else if flag == CVDIAG_LMEM_NULL as i64 {
        "CVDIAG_LMEM_NULL"
    } else if flag == CVDIAG_ILL_INPUT as i64 {
        "CVDIAG_ILL_INPUT"
    } else if flag == CVDIAG_MEM_FAIL as i64 {
        "CVDIAG_MEM_FAIL"
    } else if flag == CVDIAG_INV_FAIL as i64 {
        "CVDIAG_INV_FAIL"
    } else if flag == CVDIAG_RHSFUNC_UNRECVR as i64 {
        "CVDIAG_RHSFUNC_UNRECVR"
    } else if flag == CVDIAG_RHSFUNC_RECVR as i64 {
        "CVDIAG_RHSFUNC_RECVR"
    } else if flag == CVDIAG_NO_ADJ as i64 {
        "CVDIAG_NO_ADJ"
    } else {
        "NONE"
    };

    name.to_string()
}

/*
 * -----------------------------------------------------------------
 * CVDiagInit
 * -----------------------------------------------------------------
 * This routine does remaining initializations specific to the diagonal
 * linear solver.
 * -----------------------------------------------------------------
 */

fn CVDiagInit(cv_mem: &CVodeMem) -> i32 {
    let mut cvdiag_mem = cvdiag_mem_mut(cv_mem);

    cvdiag_mem.di_nfeDI = 0;

    cvdiag_mem.di_last_flag = CVDIAG_SUCCESS as i64;
    0
}

/*
 * -----------------------------------------------------------------
 * CVDiagSetup
 * -----------------------------------------------------------------
 * This routine does the setup operations for the diagonal linear
 * solver.  It constructs a diagonal approximation to the Newton matrix
 * M = I - gamma*J, updates counters, and inverts M.
 * -----------------------------------------------------------------
 */

fn CVDiagSetup(
    cv_mem: &CVodeMem,
    _convfail: i32,
    ypred: &N_Vector,
    fpred: &N_Vector,
    jcurPtr: &mut sunbooleantype,
    vtemp1: &N_Vector,
    vtemp2: &N_Vector,
    _vtemp3: &N_Vector,
) -> i32 {
    /* Rename work vectors for use as temporary values of y and f */
    let ftemp = vtemp1;
    let y = vtemp2;

    /* Copy scalars / clone handles out of the mem (granular borrows) */
    let (r, h, tn, uround, zn1, ewt, f);
    {
        let m = cv_mem.borrow();
        /* Form y with perturbation = FRACT*(func. iter. correction) */
        r = FRACT * m.cv_rl1;
        h = m.cv_h;
        tn = m.cv_tn;
        uround = m.cv_uround;
        zn1 = m.cv_zn[1].as_ref().expect("cv_zn[1]").clone();
        ewt = m.cv_ewt.as_ref().expect("cv_ewt").clone();
        f = m.cv_f.expect("cv_f");
    }
    let (di_M, di_bit, di_bitcomp) = {
        let cvdiag_mem = cvdiag_mem_mut(cv_mem);
        (
            cvdiag_mem.di_M.clone(),
            cvdiag_mem.di_bit.clone(),
            cvdiag_mem.di_bitcomp.clone(),
        )
    };

    N_VLinearSum(h, fpred, -ONE, &zn1, ftemp);
    N_VLinearSum(r, ftemp, ONE, ypred, y);

    /* Evaluate f at perturbed y */
    let mut user_data = cv_mem.borrow_mut().cv_user_data.take();
    let retval = f(tn, y, &di_M, &mut user_data);
    cv_mem.borrow_mut().cv_user_data = user_data;
    cvdiag_mem_mut(cv_mem).di_nfeDI += 1;
    if retval < 0 {
        cvProcessError(
            Some(cv_mem),
            CVDIAG_RHSFUNC_UNRECVR,
            line!() as i32,
            "CVDiagSetup",
            file!(),
            MSGDG_RHSFUNC_FAILED,
        );
        cvdiag_mem_mut(cv_mem).di_last_flag = CVDIAG_RHSFUNC_UNRECVR as i64;
        return -1;
    }
    if retval > 0 {
        cvdiag_mem_mut(cv_mem).di_last_flag = CVDIAG_RHSFUNC_RECVR as i64;
        return 1;
    }

    /* Construct M = I - gamma*J with J = diag(deltaf_i/deltay_i) */
    N_VLinearSum(ONE, &di_M, -ONE, fpred, &di_M);
    N_VLinearSum(FRACT, ftemp, -h, &di_M, &di_M);
    N_VProd(ftemp, &ewt, y);
    /* Protect against deltay_i being at roundoff level */
    N_VCompare(uround, y, &di_bit);
    N_VAddConst(&di_bit, -ONE, &di_bitcomp);
    N_VProd(ftemp, &di_bit, y);
    N_VLinearSum(FRACT, y, -ONE, &di_bitcomp, y);
    N_VDiv(&di_M, y, &di_M);
    N_VProd(&di_M, &di_bit, &di_M);
    N_VLinearSum(ONE, &di_M, -ONE, &di_bitcomp, &di_M);

    /* Invert M with test for zero components */
    let invOK = N_VInvTest(&di_M, &di_M);
    if !invOK {
        cvdiag_mem_mut(cv_mem).di_last_flag = CVDIAG_INV_FAIL as i64;
        return 1;
    }

    /* Set jcur = SUNTRUE, save gamma in gammasv, and return */
    /* (C's jcurPtr aliases cv_mem->cv_jcur; the caller in cvodes_nls.rs
    writes the out-param back into cv_jcur on every return path, and
    nothing between this store and that write-back reads cv_jcur) */
    *jcurPtr = SUNTRUE;
    let gamma = cv_mem.borrow().cv_gamma;
    {
        let mut cvdiag_mem = cvdiag_mem_mut(cv_mem);
        cvdiag_mem.di_gammasv = gamma;
        cvdiag_mem.di_last_flag = CVDIAG_SUCCESS as i64;
    }
    0
}

/*
 * -----------------------------------------------------------------
 * CVDiagSolve
 * -----------------------------------------------------------------
 * This routine performs the solve operation for the diagonal linear
 * solver.  If necessary it first updates gamma in M = I - gamma*J.
 * -----------------------------------------------------------------
 */

fn CVDiagSolve(
    cv_mem: &CVodeMem,
    b: &N_Vector,
    _weight: &N_Vector,
    _ycur: &N_Vector,
    _fcur: &N_Vector,
) -> i32 {
    /* Copy scalars / clone handles out of the mem (granular borrows) */
    let gamma = cv_mem.borrow().cv_gamma;
    let (gammasv, di_M) = {
        let cvdiag_mem = cvdiag_mem_mut(cv_mem);
        (cvdiag_mem.di_gammasv, cvdiag_mem.di_M.clone())
    };

    /* If gamma has changed, update factor in M, and save gamma value */

    if gammasv != gamma {
        let r = gamma / gammasv;
        N_VInv(&di_M, &di_M);
        N_VAddConst(&di_M, -ONE, &di_M);
        N_VScale(r, &di_M, &di_M);
        N_VAddConst(&di_M, ONE, &di_M);
        let invOK = N_VInvTest(&di_M, &di_M);
        if !invOK {
            cvdiag_mem_mut(cv_mem).di_last_flag = CVDIAG_INV_FAIL as i64;
            return 1;
        }
        cvdiag_mem_mut(cv_mem).di_gammasv = gamma;
    }

    /* Apply M-inverse to b */
    N_VProd(b, &di_M, b);

    cvdiag_mem_mut(cv_mem).di_last_flag = CVDIAG_SUCCESS as i64;
    0
}

/*
 * -----------------------------------------------------------------
 * CVDiagFree
 * -----------------------------------------------------------------
 * This routine frees memory specific to the diagonal linear solver.
 * -----------------------------------------------------------------
 */

fn CVDiagFree(cv_mem: &CVodeMem) -> i32 {
    /* Dropping the record destroys di_M, di_bit, di_bitcomp
    (C: N_VDestroy x3 + free(cvdiag_mem)) and sets cv_lmem = NULL */
    cv_mem.borrow_mut().cv_lmem = None;

    0
}

/*
 * ================================================================
 *
 *                   PART II - backward problems
 *
 * ================================================================
 */

/*
 * CVDiagB
 *
 * Wrappers for the backward phase around the corresponding
 * CVODES functions
 */

pub fn CVDiagB(cvode_mem: &CVodeMem, which: i32) -> i32 {
    /* Check if cvode_mem exists: handled by type system */
    let cv_mem = cvode_mem;

    /* Was ASA initialized? */
    if cv_mem.borrow().cv_adjMallocDone == SUNFALSE {
        cvProcessError(
            Some(cv_mem),
            CVDIAG_NO_ADJ,
            line!() as i32,
            "CVDiagB",
            file!(),
            MSGDG_NO_ADJ,
        );
        return CVDIAG_NO_ADJ;
    }
    let ca_mem = cv_mem
        .borrow()
        .cv_adj_mem
        .as_ref()
        .expect("cv_adj_mem")
        .clone();

    /* Check which */
    if which >= ca_mem.borrow().ca_nbckpbs {
        cvProcessError(
            Some(cv_mem),
            CVDIAG_ILL_INPUT,
            line!() as i32,
            "CVDiagB",
            file!(),
            MSGDG_BAD_WHICH,
        );
        return CVDIAG_ILL_INPUT;
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
    matches no backward problem; that UB maps to a panic here */
    let cvB_mem = cvB_mem.expect("cvB_mem");

    let cvodeB_mem = cvB_mem.borrow().cv_mem.as_ref().expect("cv_mem").clone();

    let flag = CVDiag(&cvodeB_mem);

    flag
}
