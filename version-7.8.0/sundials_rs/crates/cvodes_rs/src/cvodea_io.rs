//! Port of `src/cvodes/cvodea_io.c` — the optional input and output
//! functions for the adjoint (ASA) module in the CVODES solver.
//!
//! Binding notes (all locked workspace conventions):
//!
//! - The C `cvode_mem == NULL` guard at the head of every function is
//!   handled by the type system (`&CVodeMem`), so it is elided; the
//!   `MSGCV_NO_MEM` branch is unreachable in safe Rust.
//! - `ca_mem = cv_mem->cv_adj_mem;` and the `cvB_mem` list walk both
//!   dereference a possibly-NULL pointer in C. Those sites map to a
//!   deterministic panic (accepted deviation class 5). In particular
//!   `CVodeSetConstraintsB` reproduces the upstream missing-`return`
//!   after its `CV_NO_ADJ` diagnostic: the error is reported and then the
//!   NULL `cv_adj_mem` is dereferenced, exactly as in C.
//! - The C intrusive lists become `Vec`s in `CVadjMemRec` with index 0 =
//!   list head, so `for (p = head; p; p = p->next)` is a forward `Vec`
//!   iteration and `p->next` is the following index.
//! - `void*` addresses that C merely copies (checkpoint addresses) map to
//!   `Rc` handle clones (`Option<CVckpntMem>`); the owned `void* user_data`
//!   token maps to `Option<Box<dyn Any>>` and `CVodeGetUserDataB` SWAPS it
//!   with the caller's out-param (deviation class 6), like
//!   `CVodeGetUserData`.
//! - `CVodeGetAdjCVodeBmem` returns `Option<CVodeMem>` (C NULL = `None`).

use std::any::Any;

use sundials_core::sundials_nonlinearsolver::SUNNonlinearSolver;
use sundials_core::sundials_nvector::{N_VScale, N_Vector};
use sundials_core::sundials_types::{sunbooleantype, sunrealtype, SUNFALSE};

use crate::cvodes_impl::*;
use crate::cvodes_io::{
    CVodeSetConstraints, CVodeSetInitStep, CVodeSetMaxNumSteps, CVodeSetMaxOrd, CVodeSetMaxStep,
    CVodeSetMinStep, CVodeSetQuadErrCon, CVodeSetStabLimDet,
};
use crate::cvodes_nls::CVodeSetNonlinearSolver;

/*
 * =================================================================
 * CVODEA PRIVATE CONSTANTS
 * =================================================================
 */

const ONE: sunrealtype = 1.0;

/*
 * =================================================================
 * Private helpers (the repeated C boilerplate of this file)
 * =================================================================
 */

/// C: `ca_mem = cv_mem->cv_adj_mem;`
///
/// Reached only after the `cv_adjMallocDone` test in C, except in
/// `CVodeSetConstraintsB` where the upstream code omits the `return` and
/// dereferences NULL. Deviation class 5: deterministic panic at the
/// same site.
fn cvodea_adj_mem(cv_mem: &CVodeMem) -> CVadjMem {
    cv_mem
        .borrow()
        .cv_adj_mem
        .clone()
        .expect("cv_mem->cv_adj_mem = NULL (C dereferences a NULL pointer here)")
}

/// C: `cvB_mem = ca_mem->cvB_mem;` then
/// `while (cvB_mem != NULL) { if (which == cvB_mem->cv_index) break;
/// cvB_mem = cvB_mem->cv_next; }`.
///
/// Falling off the end leaves `cvB_mem == NULL`, which every caller then
/// dereferences (deviation class 5: deterministic panic).
fn cvodea_bck_problem(ca_mem: &CVadjMem, which: i32) -> CVodeBMem {
    let ca_mem = ca_mem.borrow();
    for cvB_mem in ca_mem.cvB_mem.iter() {
        if which == cvB_mem.borrow().cv_index {
            return cvB_mem.clone();
        }
    }
    panic!("cvB_mem = NULL (C dereferences the NULL list tail here)");
}

/// C: `cvodeB_mem = (void*)(cvB_mem->cv_mem);`
///
/// `cv_mem` is set by `CVodeCreateB` and is never NULL for a backward
/// problem that appears in the list (deviation class 5).
fn cvodea_bck_cvode_mem(cvB_mem: &CVodeBMem) -> CVodeMem {
    cvB_mem
        .borrow()
        .cv_mem
        .clone()
        .expect("cvB_mem->cv_mem = NULL (C would pass NULL to a CVode* setter)")
}

/*
 * =================================================================
 * EXPORTED FUNCTIONS IMPLEMENTATION
 * =================================================================
 */

/*
 * -----------------------------------------------------------------
 * Optional input functions for ASA
 * -----------------------------------------------------------------
 */

pub fn CVodeSetAdjNoSensi(cvode_mem: &CVodeMem) -> i32 {
    /* Check if cvode_mem exists: NULL-mem check handled by type system */
    let cv_mem = cvode_mem;

    /* Was ASA initialized? */
    let adjMallocDone = cv_mem.borrow().cv_adjMallocDone;
    if adjMallocDone == SUNFALSE {
        cvProcessError(
            Some(cv_mem),
            CV_NO_ADJ,
            line!() as i32,
            "CVodeSetAdjNoSensi",
            file!(),
            MSGCV_NO_ADJ,
        );
        return CV_NO_ADJ;
    }
    let ca_mem = cvodea_adj_mem(cv_mem);

    ca_mem.borrow_mut().ca_IMstoreSensi = SUNFALSE;

    CV_SUCCESS
}

/*
 * -----------------------------------------------------------------
 * Optional input functions for backward integration
 * -----------------------------------------------------------------
 */

pub fn CVodeSetNonlinearSolverB(cvode_mem: &CVodeMem, which: i32, NLS: &SUNNonlinearSolver) -> i32 {
    /* Check if cvode_mem exists: NULL-mem check handled by type system */
    let cv_mem = cvode_mem;

    /* Was ASA initialized? */
    let adjMallocDone = cv_mem.borrow().cv_adjMallocDone;
    if adjMallocDone == SUNFALSE {
        cvProcessError(
            Some(cv_mem),
            CV_NO_ADJ,
            line!() as i32,
            "CVodeSetNonlinearSolverB",
            file!(),
            MSGCV_NO_ADJ,
        );
        return CV_NO_ADJ;
    }
    let ca_mem = cvodea_adj_mem(cv_mem);

    /* Check which */
    let nbckpbs = ca_mem.borrow().ca_nbckpbs;
    if which >= nbckpbs {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeSetNonlinearSolverB",
            file!(),
            MSGCV_BAD_WHICH,
        );
        return CV_ILL_INPUT;
    }

    /* Find the CVodeBMem entry in the linked list corresponding to which */
    let cvB_mem = cvodea_bck_problem(&ca_mem, which);

    let cvodeB_mem = cvodea_bck_cvode_mem(&cvB_mem);

    CVodeSetNonlinearSolver(&cvodeB_mem, NLS)
}

pub fn CVodeSetUserDataB(
    cvode_mem: &CVodeMem,
    which: i32,
    user_dataB: Option<Box<dyn Any>>,
) -> i32 {
    /* Check if cvode_mem exists: NULL-mem check handled by type system */
    let cv_mem = cvode_mem;

    /* Was ASA initialized? */
    let adjMallocDone = cv_mem.borrow().cv_adjMallocDone;
    if adjMallocDone == SUNFALSE {
        cvProcessError(
            Some(cv_mem),
            CV_NO_ADJ,
            line!() as i32,
            "CVodeSetUserDataB",
            file!(),
            MSGCV_NO_ADJ,
        );
        return CV_NO_ADJ;
    }
    let ca_mem = cvodea_adj_mem(cv_mem);

    /* Check which */
    let nbckpbs = ca_mem.borrow().ca_nbckpbs;
    if which >= nbckpbs {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeSetUserDataB",
            file!(),
            MSGCV_BAD_WHICH,
        );
        return CV_ILL_INPUT;
    }

    /* Find the CVodeBMem entry in the linked list corresponding to which */
    let cvB_mem = cvodea_bck_problem(&ca_mem, which);

    cvB_mem.borrow_mut().cv_user_data = user_dataB;

    CV_SUCCESS
}

/// C `CVodeGetUserDataB` returns the stored `void*` without ownership
/// transfer. The safe-Rust token cannot be aliased, so the stored box is
/// SWAPPED with `user_dataB`; the caller must hand it back (via
/// `CVodeSetUserDataB` or a second swap) before the integrator next
/// invokes a backward user callback.
pub fn CVodeGetUserDataB(
    cvode_mem: &CVodeMem,
    which: i32,
    user_dataB: &mut Option<Box<dyn Any>>,
) -> i32 {
    /* Check if cvode_mem exists: NULL-mem check handled by type system */
    let cv_mem = cvode_mem;

    /* Was ASA initialized? */
    let adjMallocDone = cv_mem.borrow().cv_adjMallocDone;
    if adjMallocDone == SUNFALSE {
        cvProcessError(
            Some(cv_mem),
            CV_NO_ADJ,
            line!() as i32,
            "CVodeGetUserDataB",
            file!(),
            MSGCV_NO_ADJ,
        );
        return CV_NO_ADJ;
    }
    let ca_mem = cvodea_adj_mem(cv_mem);

    /* Check which */
    let nbckpbs = ca_mem.borrow().ca_nbckpbs;
    if which >= nbckpbs {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeGetUserDataB",
            file!(),
            MSGCV_BAD_WHICH,
        );
        return CV_ILL_INPUT;
    }

    /* Find the CVodeBMem entry in the linked list corresponding to which */
    let cvB_mem = cvodea_bck_problem(&ca_mem, which);

    std::mem::swap(&mut cvB_mem.borrow_mut().cv_user_data, user_dataB);

    CV_SUCCESS
}

pub fn CVodeSetMaxOrdB(cvode_mem: &CVodeMem, which: i32, maxordB: i32) -> i32 {
    /* Check if cvode_mem exists: NULL-mem check handled by type system */
    let cv_mem = cvode_mem;
    let flag: i32;

    /* Was ASA initialized? */
    let adjMallocDone = cv_mem.borrow().cv_adjMallocDone;
    if adjMallocDone == SUNFALSE {
        cvProcessError(
            Some(cv_mem),
            CV_NO_ADJ,
            line!() as i32,
            "CVodeSetMaxOrdB",
            file!(),
            MSGCV_NO_ADJ,
        );
        return CV_NO_ADJ;
    }
    let ca_mem = cvodea_adj_mem(cv_mem);

    /* Check which */
    let nbckpbs = ca_mem.borrow().ca_nbckpbs;
    if which >= nbckpbs {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeSetMaxOrdB",
            file!(),
            MSGCV_BAD_WHICH,
        );
        return CV_ILL_INPUT;
    }

    /* Find the CVodeBMem entry in the linked list corresponding to which */
    let cvB_mem = cvodea_bck_problem(&ca_mem, which);

    let cvodeB_mem = cvodea_bck_cvode_mem(&cvB_mem);

    flag = CVodeSetMaxOrd(&cvodeB_mem, maxordB);

    flag
}

pub fn CVodeSetMaxNumStepsB(cvode_mem: &CVodeMem, which: i32, mxstepsB: i64) -> i32 {
    /* Check if cvode_mem exists: NULL-mem check handled by type system */
    let cv_mem = cvode_mem;
    let flag: i32;

    /* Was ASA initialized? */
    let adjMallocDone = cv_mem.borrow().cv_adjMallocDone;
    if adjMallocDone == SUNFALSE {
        cvProcessError(
            Some(cv_mem),
            CV_NO_ADJ,
            line!() as i32,
            "CVodeSetMaxNumStepsB",
            file!(),
            MSGCV_NO_ADJ,
        );
        return CV_NO_ADJ;
    }
    let ca_mem = cvodea_adj_mem(cv_mem);

    /* Check which */
    let nbckpbs = ca_mem.borrow().ca_nbckpbs;
    if which >= nbckpbs {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeSetMaxNumStepsB",
            file!(),
            MSGCV_BAD_WHICH,
        );
        return CV_ILL_INPUT;
    }

    /* Find the CVodeBMem entry in the linked list corresponding to which */
    let cvB_mem = cvodea_bck_problem(&ca_mem, which);

    let cvodeB_mem = cvodea_bck_cvode_mem(&cvB_mem);

    flag = CVodeSetMaxNumSteps(&cvodeB_mem, mxstepsB);

    flag
}

pub fn CVodeSetStabLimDetB(cvode_mem: &CVodeMem, which: i32, stldetB: sunbooleantype) -> i32 {
    /* Check if cvode_mem exists: NULL-mem check handled by type system */
    let cv_mem = cvode_mem;
    let flag: i32;

    /* Was ASA initialized? */
    let adjMallocDone = cv_mem.borrow().cv_adjMallocDone;
    if adjMallocDone == SUNFALSE {
        cvProcessError(
            Some(cv_mem),
            CV_NO_ADJ,
            line!() as i32,
            "CVodeSetStabLimDetB",
            file!(),
            MSGCV_NO_ADJ,
        );
        return CV_NO_ADJ;
    }
    let ca_mem = cvodea_adj_mem(cv_mem);

    /* Check which */
    let nbckpbs = ca_mem.borrow().ca_nbckpbs;
    if which >= nbckpbs {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeSetStabLimDetB",
            file!(),
            MSGCV_BAD_WHICH,
        );
        return CV_ILL_INPUT;
    }

    /* Find the CVodeBMem entry in the linked list corresponding to which */
    let cvB_mem = cvodea_bck_problem(&ca_mem, which);

    let cvodeB_mem = cvodea_bck_cvode_mem(&cvB_mem);

    flag = CVodeSetStabLimDet(&cvodeB_mem, stldetB);

    flag
}

pub fn CVodeSetInitStepB(cvode_mem: &CVodeMem, which: i32, hinB: sunrealtype) -> i32 {
    /* Check if cvode_mem exists: NULL-mem check handled by type system */
    let cv_mem = cvode_mem;
    let flag: i32;

    /* Was ASA initialized? */
    let adjMallocDone = cv_mem.borrow().cv_adjMallocDone;
    if adjMallocDone == SUNFALSE {
        cvProcessError(
            Some(cv_mem),
            CV_NO_ADJ,
            line!() as i32,
            "CVodeSetInitStepB",
            file!(),
            MSGCV_NO_ADJ,
        );
        return CV_NO_ADJ;
    }
    let ca_mem = cvodea_adj_mem(cv_mem);

    /* Check which */
    let nbckpbs = ca_mem.borrow().ca_nbckpbs;
    if which >= nbckpbs {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeSetInitStepB",
            file!(),
            MSGCV_BAD_WHICH,
        );
        return CV_ILL_INPUT;
    }

    /* Find the CVodeBMem entry in the linked list corresponding to which */
    let cvB_mem = cvodea_bck_problem(&ca_mem, which);

    let cvodeB_mem = cvodea_bck_cvode_mem(&cvB_mem);

    flag = CVodeSetInitStep(&cvodeB_mem, hinB);

    flag
}

pub fn CVodeSetMinStepB(cvode_mem: &CVodeMem, which: i32, hminB: sunrealtype) -> i32 {
    /* Check if cvode_mem exists: NULL-mem check handled by type system */
    let cv_mem = cvode_mem;
    let flag: i32;

    /* Was ASA initialized? */
    let adjMallocDone = cv_mem.borrow().cv_adjMallocDone;
    if adjMallocDone == SUNFALSE {
        cvProcessError(
            Some(cv_mem),
            CV_NO_ADJ,
            line!() as i32,
            "CVodeSetMinStepB",
            file!(),
            MSGCV_NO_ADJ,
        );
        return CV_NO_ADJ;
    }
    let ca_mem = cvodea_adj_mem(cv_mem);

    /* Check which */
    let nbckpbs = ca_mem.borrow().ca_nbckpbs;
    if which >= nbckpbs {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeSetMinStepB",
            file!(),
            MSGCV_BAD_WHICH,
        );
        return CV_ILL_INPUT;
    }

    /* Find the CVodeBMem entry in the linked list corresponding to which */
    let cvB_mem = cvodea_bck_problem(&ca_mem, which);

    let cvodeB_mem = cvodea_bck_cvode_mem(&cvB_mem);

    flag = CVodeSetMinStep(&cvodeB_mem, hminB);

    flag
}

pub fn CVodeSetMaxStepB(cvode_mem: &CVodeMem, which: i32, hmaxB: sunrealtype) -> i32 {
    /* Check if cvode_mem exists: NULL-mem check handled by type system */
    let cv_mem = cvode_mem;
    let flag: i32;

    /* Was ASA initialized? */
    let adjMallocDone = cv_mem.borrow().cv_adjMallocDone;
    if adjMallocDone == SUNFALSE {
        cvProcessError(
            Some(cv_mem),
            CV_NO_ADJ,
            line!() as i32,
            "CVodeSetMaxStepB",
            file!(),
            MSGCV_NO_ADJ,
        );
        return CV_NO_ADJ;
    }
    let ca_mem = cvodea_adj_mem(cv_mem);

    /* Check which */
    let nbckpbs = ca_mem.borrow().ca_nbckpbs;
    if which >= nbckpbs {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeSetMaxStepB",
            file!(),
            MSGCV_BAD_WHICH,
        );
        return CV_ILL_INPUT;
    }

    /* Find the CVodeBMem entry in the linked list corresponding to which */
    let cvB_mem = cvodea_bck_problem(&ca_mem, which);

    let cvodeB_mem = cvodea_bck_cvode_mem(&cvB_mem);

    flag = CVodeSetMaxStep(&cvodeB_mem, hmaxB);

    flag
}

pub fn CVodeSetConstraintsB(
    cvode_mem: &CVodeMem,
    which: i32,
    constraintsB: Option<&N_Vector>,
) -> i32 {
    /* Is cvode_mem valid? NULL-mem check handled by type system */
    let cv_mem = cvode_mem;
    let flag: i32;

    /* Is ASA initialized? */
    let adjMallocDone = cv_mem.borrow().cv_adjMallocDone;
    if adjMallocDone == SUNFALSE {
        cvProcessError(
            Some(cv_mem),
            CV_NO_ADJ,
            line!() as i32,
            "CVodeSetConstraintsB",
            file!(),
            MSGCV_NO_ADJ,
        );
        /* NOTE: upstream omits the `return (CV_NO_ADJ);` present in every
        sibling function, so C falls through and dereferences the NULL
        `cv_adj_mem` below. Reproduced verbatim: the helper panics
        (accepted deviation class 5). */
    }
    let ca_mem = cvodea_adj_mem(cv_mem);

    /* Check the value of which */
    let nbckpbs = ca_mem.borrow().ca_nbckpbs;
    if which >= nbckpbs {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeSetConstraintsB",
            file!(),
            MSGCV_BAD_WHICH,
        );
        return CV_ILL_INPUT;
    }

    /* Find the CVodeBMem entry in the linked list corresponding to 'which'. */
    let cvB_mem = cvodea_bck_problem(&ca_mem, which);
    let cvodeB_mem = cvodea_bck_cvode_mem(&cvB_mem);

    flag = CVodeSetConstraints(&cvodeB_mem, constraintsB);
    flag
}

/*
 * CVodeSetQuad*B
 *
 * Wrappers for the backward phase around the corresponding
 * CVODES quadrature optional input functions
 */

pub fn CVodeSetQuadErrConB(cvode_mem: &CVodeMem, which: i32, errconQB: sunbooleantype) -> i32 {
    /* Check if cvode_mem exists: NULL-mem check handled by type system */
    let cv_mem = cvode_mem;
    let flag: i32;

    /* Was ASA initialized? */
    let adjMallocDone = cv_mem.borrow().cv_adjMallocDone;
    if adjMallocDone == SUNFALSE {
        cvProcessError(
            Some(cv_mem),
            CV_NO_ADJ,
            line!() as i32,
            "CVodeSetQuadErrConB",
            file!(),
            MSGCV_NO_ADJ,
        );
        return CV_NO_ADJ;
    }
    let ca_mem = cvodea_adj_mem(cv_mem);

    /* Check which */
    let nbckpbs = ca_mem.borrow().ca_nbckpbs;
    if which >= nbckpbs {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeSetQuadErrConB",
            file!(),
            MSGCV_BAD_WHICH,
        );
        return CV_ILL_INPUT;
    }

    /* Find the CVodeBMem entry in the linked list corresponding to which */
    let cvB_mem = cvodea_bck_problem(&ca_mem, which);

    let cvodeB_mem = cvodea_bck_cvode_mem(&cvB_mem);

    flag = CVodeSetQuadErrCon(&cvodeB_mem, errconQB);

    flag
}

/*
 * -----------------------------------------------------------------
 * Optional output functions for backward integration
 * -----------------------------------------------------------------
 */

/*
 * CVodeGetAdjCVodeBmem
 *
 * This function returns a handle to the CVODES memory allocated for the
 * backward problem. This handle can then be used to call any of the
 * CVodeGet* CVODES routines to extract optional output for the backward
 * integration phase. (C returns `void*`; NULL maps to `None`.)
 */

pub fn CVodeGetAdjCVodeBmem(cvode_mem: &CVodeMem, which: i32) -> Option<CVodeMem> {
    /* Check if cvode_mem exists: NULL-mem check handled by type system */
    let cv_mem = cvode_mem;

    /* Was ASA initialized? */
    let adjMallocDone = cv_mem.borrow().cv_adjMallocDone;
    if adjMallocDone == SUNFALSE {
        cvProcessError(
            Some(cv_mem),
            0,
            line!() as i32,
            "CVodeGetAdjCVodeBmem",
            file!(),
            MSGCV_NO_ADJ,
        );
        return None;
    }
    let ca_mem = cvodea_adj_mem(cv_mem);

    /* Check which */
    let nbckpbs = ca_mem.borrow().ca_nbckpbs;
    if which >= nbckpbs {
        cvProcessError(
            Some(cv_mem),
            0,
            line!() as i32,
            "CVodeGetAdjCVodeBmem",
            file!(),
            MSGCV_BAD_WHICH,
        );
        return None;
    }

    /* Find the CVodeBMem entry in the linked list corresponding to which */
    let cvB_mem = cvodea_bck_problem(&ca_mem, which);

    /* C copies the (possibly NULL) pointer without dereferencing it */
    let cvodeB_mem = cvB_mem.borrow().cv_mem.clone();

    cvodeB_mem
}

/*
 * CVodeGetAdjCheckPointsInfo
 *
 * This routine loads an array of nckpnts structures of type CVadjCheckPointRec.
 * The user must allocate space for ckpnt. (C `CVadjCheckPointRec*` becomes a
 * `&mut [CVadjCheckPointRec]`; a short slice panics where C would write past
 * the end of the caller's buffer — deviation class 5.)
 */

pub fn CVodeGetAdjCheckPointsInfo(cvode_mem: &CVodeMem, ckpnt: &mut [CVadjCheckPointRec]) -> i32 {
    /* Check if cvode_mem exists: NULL-mem check handled by type system */
    let cv_mem = cvode_mem;

    /* Was ASA initialized? */
    let adjMallocDone = cv_mem.borrow().cv_adjMallocDone;
    if adjMallocDone == SUNFALSE {
        cvProcessError(
            Some(cv_mem),
            CV_NO_ADJ,
            line!() as i32,
            "CVodeGetAdjCheckPointsInfo",
            file!(),
            MSGCV_NO_ADJ,
        );
        return CV_NO_ADJ;
    }
    let ca_mem = cvodea_adj_mem(cv_mem);

    let ca = ca_mem.borrow();

    /* ck_mem = ca_mem->ck_mem;  (Vec index 0 = C list head) */
    let mut i: usize = 0;

    while i < ca.ck_mem.len() {
        let ck_mem = &ca.ck_mem[i];
        let ck = ck_mem.borrow();

        ckpnt[i].my_addr = Some(ck_mem.clone());
        ckpnt[i].next_addr = ca.ck_mem.get(i + 1).cloned();
        ckpnt[i].t0 = ck.ck_t0;
        ckpnt[i].t1 = ck.ck_t1;
        ckpnt[i].nstep = ck.ck_nst;
        ckpnt[i].order = ck.ck_q;
        ckpnt[i].step = ck.ck_h;

        /* ck_mem = ck_mem->ck_next; */
        i += 1;
    }

    CV_SUCCESS
}

/*
 * -----------------------------------------------------------------
 * Undocumented Development User-Callable Functions
 * -----------------------------------------------------------------
 */

/*
 * CVodeGetAdjDataPointHermite
 *
 * This routine returns the solution stored in the data structure
 * at the 'which' data point. Cubic Hermite interpolation.
 */

pub fn CVodeGetAdjDataPointHermite(
    cvode_mem: &CVodeMem,
    which: i32,
    t: &mut sunrealtype,
    y: Option<&N_Vector>,
    yd: Option<&N_Vector>,
) -> i32 {
    /* Check if cvode_mem exists: NULL-mem check handled by type system */
    let cv_mem = cvode_mem;

    /* Was ASA initialized? */
    let adjMallocDone = cv_mem.borrow().cv_adjMallocDone;
    if adjMallocDone == SUNFALSE {
        cvProcessError(
            Some(cv_mem),
            CV_NO_ADJ,
            line!() as i32,
            "CVodeGetAdjDataPointHermite",
            file!(),
            MSGCV_NO_ADJ,
        );
        return CV_NO_ADJ;
    }
    let ca_mem = cvodea_adj_mem(cv_mem);

    /* dt_mem = ca_mem->dt_mem; (indexed below) */

    let IMtype = ca_mem.borrow().ca_IMtype;
    if IMtype != CV_HERMITE {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeGetAdjDataPointHermite",
            file!(),
            MSGCV_WRONG_INTERP,
        );
        return CV_ILL_INPUT;
    }

    let dt_pnt = ca_mem.borrow().dt_mem[which as usize].clone();

    *t = dt_pnt.borrow().t;

    /* content = (CVhermiteDataMem)(dt_mem[which]->content); */
    let (content_y, content_yd) = {
        let d = dt_pnt.borrow();
        let content = d
            .content
            .as_ref()
            .expect("dt_mem[which]->content = NULL (C dereferences a NULL pointer here)")
            .downcast_ref::<CVhermiteDataMemRec>()
            .expect("dt_mem[which]->content is not a CVhermiteDataMem");
        (content.y.clone(), content.yd.clone())
    };

    if let Some(y) = y {
        N_VScale(
            ONE,
            content_y
                .as_ref()
                .expect("content->y = NULL (C dereferences a NULL pointer here)"),
            y,
        );
    }

    if let Some(yd) = yd {
        N_VScale(
            ONE,
            content_yd
                .as_ref()
                .expect("content->yd = NULL (C dereferences a NULL pointer here)"),
            yd,
        );
    }

    CV_SUCCESS
}

/*
 * CVodeGetAdjDataPointPolynomial
 *
 * This routine returns the solution stored in the data structure
 * at the 'which' data point. Polynomial interpolation.
 */

pub fn CVodeGetAdjDataPointPolynomial(
    cvode_mem: &CVodeMem,
    which: i32,
    t: &mut sunrealtype,
    order: &mut i32,
    y: Option<&N_Vector>,
) -> i32 {
    /* Check if cvode_mem exists: NULL-mem check handled by type system */
    let cv_mem = cvode_mem;

    /* Was ASA initialized? */
    let adjMallocDone = cv_mem.borrow().cv_adjMallocDone;
    if adjMallocDone == SUNFALSE {
        cvProcessError(
            Some(cv_mem),
            CV_NO_ADJ,
            line!() as i32,
            "CVodeGetAdjDataPointPolynomial",
            file!(),
            MSGCV_NO_ADJ,
        );
        return CV_NO_ADJ;
    }
    let ca_mem = cvodea_adj_mem(cv_mem);

    /* dt_mem = ca_mem->dt_mem; (indexed below) */

    let IMtype = ca_mem.borrow().ca_IMtype;
    if IMtype != CV_POLYNOMIAL {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeGetAdjDataPointPolynomial",
            file!(),
            MSGCV_WRONG_INTERP,
        );
        return CV_ILL_INPUT;
    }

    let dt_pnt = ca_mem.borrow().dt_mem[which as usize].clone();

    *t = dt_pnt.borrow().t;

    /* content = (CVpolynomialDataMem)(dt_mem[which]->content); */
    let (content_y, content_order) = {
        let d = dt_pnt.borrow();
        let content = d
            .content
            .as_ref()
            .expect("dt_mem[which]->content = NULL (C dereferences a NULL pointer here)")
            .downcast_ref::<CVpolynomialDataMemRec>()
            .expect("dt_mem[which]->content is not a CVpolynomialDataMem");
        (content.y.clone(), content.order)
    };

    if let Some(y) = y {
        N_VScale(
            ONE,
            content_y
                .as_ref()
                .expect("content->y = NULL (C dereferences a NULL pointer here)"),
            y,
        );
    }

    *order = content_order;

    CV_SUCCESS
}

/*
 * CVodeGetAdjCurrentCheckPoint
 *
 * Returns the address of the 'active' check point. (C `void** addr`
 * becomes `&mut Option<CVckpntMem>`; the write is a handle clone, i.e. a
 * pointer copy with no ownership transfer.)
 */

pub fn CVodeGetAdjCurrentCheckPoint(cvode_mem: &CVodeMem, addr: &mut Option<CVckpntMem>) -> i32 {
    /* Check if cvode_mem exists: NULL-mem check handled by type system */
    let cv_mem = cvode_mem;

    /* Was ASA initialized? */
    let adjMallocDone = cv_mem.borrow().cv_adjMallocDone;
    if adjMallocDone == SUNFALSE {
        cvProcessError(
            Some(cv_mem),
            CV_NO_ADJ,
            line!() as i32,
            "CVodeGetAdjCurrentCheckPoint",
            file!(),
            MSGCV_NO_ADJ,
        );
        return CV_NO_ADJ;
    }
    let ca_mem = cvodea_adj_mem(cv_mem);

    *addr = ca_mem.borrow().ca_ckpntData.clone();

    CV_SUCCESS
}
