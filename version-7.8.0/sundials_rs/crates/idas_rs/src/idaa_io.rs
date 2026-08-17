//! Port of `src/idas/idaa_io.c` — the optional input and output functions
//! for the adjoint (ASA) module in the IDAS solver.
//!
//! Binding notes (all locked workspace conventions; the sibling
//! `cvodes_rs::cvodea_io` renders the same upstream file family the same
//! way, but the IDAS C is the ground truth for every line below):
//!
//! - The C `ida_mem == NULL` guard at the head of every function is
//!   handled by the type system (`&IDAMem`), so it is elided; the
//!   `MSGAM_NULL_IDAMEM` branch is unreachable in safe Rust. Noted at each
//!   site.
//! - `IDAADJ_mem = IDA_mem->ida_adj_mem;` and the `IDAB_mem` list walk both
//!   dereference a possibly-NULL pointer in C (the walk falls off the end
//!   as NULL when `which` matches no entry, and every caller then reads
//!   `IDAB_mem->IDA_mem`). Those sites map to a deterministic panic
//!   (ARCHITECTURE accepted deviation class 5). Unlike CVODES's
//!   `CVodeSetConstraintsB`, **every** IDAS function here has its
//!   `return (IDA_NO_ADJ);`, so the NULL `ida_adj_mem` path is genuinely
//!   unreachable after the `ida_adjMallocDone` test.
//! - The C intrusive lists become `Vec`s in `IDAadjMemRec` with index 0 =
//!   list head, so `for (p = head; p; p = p->next)` is a forward `Vec`
//!   iteration and `p->next` is the following index (`ck_next` in
//!   `IDAGetAdjCheckPointsInfo` = `ck_mem.get(i + 1)`).
//! - `void*` addresses that C merely copies (checkpoint addresses) map to
//!   `Rc` handle clones (`Option<IDAckpntMem>`); the owned `void*
//!   user_data` token maps to `Option<Box<dyn Any>>` and `IDAGetUserDataB`
//!   SWAPS it with the caller's out-param (deviation class 6), exactly as
//!   the contract's user-data box protocol requires.
//! - `IDAGetAdjIDABmem` returns `Option<IDAMem>` (C NULL = `None`); C
//!   copies `IDAB_mem->IDA_mem` without dereferencing it, so the `None`
//!   case is propagated rather than panicking.
//! - `IDASetQuadErrConB`'s C parameter is declared `int errconQB` in
//!   `idas.h` but is forwarded verbatim to `IDASetQuadErrCon`'s
//!   `sunbooleantype errconQ` (C `sunbooleantype` IS `int`), and every
//!   reference example passes `SUNTRUE`; the port therefore types it
//!   `sunbooleantype`, keeping meaning and call sites 1:1.
//! - Borrow discipline: a `RefCell` borrow is never held across
//!   `IDAProcessError`, an `N_Vector` op, or a forwarded `IDASet*`/`IDAGet*`
//!   call — each site copies the needed field into a local and drops the
//!   guard first.
//!
//! Fragment protocol (see `idas_impl.rs`): idaa_io.c's module-scope
//! `#define ONE SUN_RCONST(1.0)` is NOT redefined here; `ONE` comes from
//! `crate::idas_impl::*`.

use std::any::Any;

use sundials_core::sundials_nonlinearsolver::SUNNonlinearSolver;
use sundials_core::sundials_nvector::{N_VScale, N_Vector};
use sundials_core::sundials_types::{sunbooleantype, sunrealtype, SUNFALSE};

use crate::idas_impl::*;
use crate::idas_io::{
    IDAGetConsistentIC, IDASetConstraints, IDASetId, IDASetInitStep, IDASetMaxNumSteps,
    IDASetMaxOrd, IDASetMaxStep, IDASetQuadErrCon, IDASetSuppressAlg,
};
use crate::idas_nls::IDASetNonlinearSolver;

/*
 * =================================================================
 * IDAA PRIVATE CONSTANTS
 * =================================================================
 *
 * `#define ONE SUN_RCONST(1.0)` — shared via `idas_impl` (fragment rule).
 */

/*
 * =================================================================
 * Private helpers (the repeated C boilerplate of this file)
 * =================================================================
 */

/// C: `IDAADJ_mem = IDA_mem->ida_adj_mem;`
///
/// Reached only after the `ida_adjMallocDone` test returned `IDA_NO_ADJ`,
/// so a NULL here is C undefined behaviour; deviation class 5 maps it to a
/// deterministic panic at the same site.
fn idaa_adj_mem(IDA_mem: &IDAMem) -> IDAadjMem {
    IDA_mem
        .borrow()
        .ida_adj_mem
        .clone()
        .expect("IDA_mem->ida_adj_mem = NULL (C dereferences a NULL pointer here)")
}

/// C: `IDAB_mem = IDAADJ_mem->IDAB_mem;` then
/// `while (IDAB_mem != NULL) { if (which == IDAB_mem->ida_index) break;
/// IDAB_mem = IDAB_mem->ida_next; }`.
///
/// Falling off the end leaves `IDAB_mem == NULL`, which every caller then
/// dereferences (deviation class 5: deterministic panic).
fn idaa_bck_problem(IDAADJ_mem: &IDAadjMem, which: i32) -> IDABMem {
    let IDAADJ_mem = IDAADJ_mem.borrow();
    for IDAB_mem in IDAADJ_mem.IDAB_mem.iter() {
        if which == IDAB_mem.borrow().ida_index {
            return IDAB_mem.clone();
        }
    }
    panic!("IDAB_mem = NULL (C dereferences the NULL list tail here)");
}

/// C: `ida_memB = (void*)(IDAB_mem->IDA_mem);` at a site that immediately
/// passes the result to an `IDASet*`/`IDAGet*` entry point.
///
/// `IDA_mem` is set by `IDACreateB` and is never NULL for a backward
/// problem that appears in the list (deviation class 5).
fn idaa_bck_ida_mem(IDAB_mem: &IDABMem) -> IDAMem {
    IDAB_mem
        .borrow()
        .IDA_mem
        .clone()
        .expect("IDAB_mem->IDA_mem = NULL (C would pass NULL to an IDA* setter)")
}

/*
 * -----------------------------------------------------------------
 * Optional input functions for ASA
 * -----------------------------------------------------------------
 */

/*
 * -----------------------------------------------------------------
 * IDAAdjSetNoSensi
 * -----------------------------------------------------------------
 * Disables the forward sensitivity analysis in IDASolveF.
 * -----------------------------------------------------------------
 */

pub fn IDAAdjSetNoSensi(ida_mem: &IDAMem) -> i32 {
    /* Is ida_mem valid? NULL-mem check handled by type system */
    let IDA_mem = ida_mem;

    /* Is ASA initialized? */
    let adjMallocDone = IDA_mem.borrow().ida_adjMallocDone;
    if adjMallocDone == SUNFALSE {
        IDAProcessError(
            Some(IDA_mem),
            IDA_NO_ADJ,
            line!() as i32,
            "IDAAdjSetNoSensi",
            file!(),
            MSGAM_NO_ADJ,
        );
        return IDA_NO_ADJ;
    }
    let IDAADJ_mem = idaa_adj_mem(IDA_mem);

    IDAADJ_mem.borrow_mut().ia_storeSensi = SUNFALSE;

    IDA_SUCCESS
}

/*
 * -----------------------------------------------------------------
 * Optional input functions for backward integration
 * -----------------------------------------------------------------
 */

pub fn IDASetNonlinearSolverB(ida_mem: &IDAMem, which: i32, NLS: &SUNNonlinearSolver) -> i32 {
    /* Check if ida_mem exists: NULL-mem check handled by type system */
    let IDA_mem = ida_mem;

    /* Was ASA initialized? */
    let adjMallocDone = IDA_mem.borrow().ida_adjMallocDone;
    if adjMallocDone == SUNFALSE {
        IDAProcessError(
            Some(IDA_mem),
            IDA_NO_ADJ,
            line!() as i32,
            "IDASetNonlinearSolverB",
            file!(),
            MSGAM_NO_ADJ,
        );
        return IDA_NO_ADJ;
    }
    let IDAADJ_mem = idaa_adj_mem(IDA_mem);

    /* Check the value of which */
    let nbckpbs = IDAADJ_mem.borrow().ia_nbckpbs;
    if which >= nbckpbs {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDASetNonlinearSolverB",
            file!(),
            MSGAM_BAD_WHICH,
        );
        return IDA_ILL_INPUT;
    }

    /* Find the IDABMem entry in the linked list corresponding to 'which' */
    let IDAB_mem = idaa_bck_problem(&IDAADJ_mem, which);

    let ida_memB = idaa_bck_ida_mem(&IDAB_mem);

    IDASetNonlinearSolver(&ida_memB, NLS)
}

/// C stores the raw `void* user_dataB` pointer; the safe port takes
/// ownership of the token box (locked user-data box protocol).
pub fn IDASetUserDataB(ida_mem: &IDAMem, which: i32, user_dataB: Option<Box<dyn Any>>) -> i32 {
    /* Is ida_mem valid? NULL-mem check handled by type system */
    let IDA_mem = ida_mem;

    /* Is ASA initialized? */
    let adjMallocDone = IDA_mem.borrow().ida_adjMallocDone;
    if adjMallocDone == SUNFALSE {
        IDAProcessError(
            Some(IDA_mem),
            IDA_NO_ADJ,
            line!() as i32,
            "IDASetUserDataB",
            file!(),
            MSGAM_NO_ADJ,
        );
        return IDA_NO_ADJ;
    }
    let IDAADJ_mem = idaa_adj_mem(IDA_mem);

    /* Check the value of which */
    let nbckpbs = IDAADJ_mem.borrow().ia_nbckpbs;
    if which >= nbckpbs {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDASetUserDataB",
            file!(),
            MSGAM_BAD_WHICH,
        );
        return IDA_ILL_INPUT;
    }

    /* Find the IDABMem entry in the linked list corresponding to 'which'. */
    let IDAB_mem = idaa_bck_problem(&IDAADJ_mem, which);

    /* Set user data for this backward problem. */
    IDAB_mem.borrow_mut().ida_user_data = user_dataB;

    IDA_SUCCESS
}

pub fn IDASetMaxOrdB(ida_mem: &IDAMem, which: i32, maxordB: i32) -> i32 {
    /* Is ida_mem valid? NULL-mem check handled by type system */
    let IDA_mem = ida_mem;

    /* Is ASA initialized? */
    let adjMallocDone = IDA_mem.borrow().ida_adjMallocDone;
    if adjMallocDone == SUNFALSE {
        IDAProcessError(
            Some(IDA_mem),
            IDA_NO_ADJ,
            line!() as i32,
            "IDASetMaxOrdB",
            file!(),
            MSGAM_NO_ADJ,
        );
        return IDA_NO_ADJ;
    }
    let IDAADJ_mem = idaa_adj_mem(IDA_mem);

    /* Check the value of which */
    let nbckpbs = IDAADJ_mem.borrow().ia_nbckpbs;
    if which >= nbckpbs {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDASetMaxOrdB",
            file!(),
            MSGAM_BAD_WHICH,
        );
        return IDA_ILL_INPUT;
    }

    /* Find the IDABMem entry in the linked list corresponding to 'which'. */
    let IDAB_mem = idaa_bck_problem(&IDAADJ_mem, which);
    let ida_memB = idaa_bck_ida_mem(&IDAB_mem);

    IDASetMaxOrd(&ida_memB, maxordB)
}

pub fn IDASetMaxNumStepsB(ida_mem: &IDAMem, which: i32, mxstepsB: i64) -> i32 {
    /* Is ida_mem valid? NULL-mem check handled by type system */
    let IDA_mem = ida_mem;

    /* Is ASA initialized? */
    let adjMallocDone = IDA_mem.borrow().ida_adjMallocDone;
    if adjMallocDone == SUNFALSE {
        IDAProcessError(
            Some(IDA_mem),
            IDA_NO_ADJ,
            line!() as i32,
            "IDASetMaxNumStepsB",
            file!(),
            MSGAM_NO_ADJ,
        );
        return IDA_NO_ADJ;
    }
    let IDAADJ_mem = idaa_adj_mem(IDA_mem);

    /* Check the value of which */
    let nbckpbs = IDAADJ_mem.borrow().ia_nbckpbs;
    if which >= nbckpbs {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDASetMaxNumStepsB",
            file!(),
            MSGAM_BAD_WHICH,
        );
        return IDA_ILL_INPUT;
    }

    /* Find the IDABMem entry in the linked list corresponding to 'which'. */
    let IDAB_mem = idaa_bck_problem(&IDAADJ_mem, which);
    let ida_memB = idaa_bck_ida_mem(&IDAB_mem);

    IDASetMaxNumSteps(&ida_memB, mxstepsB)
}

pub fn IDASetInitStepB(ida_mem: &IDAMem, which: i32, hinB: sunrealtype) -> i32 {
    /* Is ida_mem valid? NULL-mem check handled by type system */
    let IDA_mem = ida_mem;

    /* Is ASA initialized? */
    let adjMallocDone = IDA_mem.borrow().ida_adjMallocDone;
    if adjMallocDone == SUNFALSE {
        IDAProcessError(
            Some(IDA_mem),
            IDA_NO_ADJ,
            line!() as i32,
            "IDASetInitStepB",
            file!(),
            MSGAM_NO_ADJ,
        );
        return IDA_NO_ADJ;
    }
    let IDAADJ_mem = idaa_adj_mem(IDA_mem);

    /* Check the value of which */
    let nbckpbs = IDAADJ_mem.borrow().ia_nbckpbs;
    if which >= nbckpbs {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDASetInitStepB",
            file!(),
            MSGAM_BAD_WHICH,
        );
        return IDA_ILL_INPUT;
    }

    /* Find the IDABMem entry in the linked list corresponding to 'which'. */
    let IDAB_mem = idaa_bck_problem(&IDAADJ_mem, which);
    let ida_memB = idaa_bck_ida_mem(&IDAB_mem);

    IDASetInitStep(&ida_memB, hinB)
}

pub fn IDASetMaxStepB(ida_mem: &IDAMem, which: i32, hmaxB: sunrealtype) -> i32 {
    /* Is ida_mem valid? NULL-mem check handled by type system */
    let IDA_mem = ida_mem;

    /* Is ASA initialized? */
    let adjMallocDone = IDA_mem.borrow().ida_adjMallocDone;
    if adjMallocDone == SUNFALSE {
        IDAProcessError(
            Some(IDA_mem),
            IDA_NO_ADJ,
            line!() as i32,
            "IDASetMaxStepB",
            file!(),
            MSGAM_NO_ADJ,
        );
        return IDA_NO_ADJ;
    }
    let IDAADJ_mem = idaa_adj_mem(IDA_mem);

    /* Check the value of which */
    let nbckpbs = IDAADJ_mem.borrow().ia_nbckpbs;
    if which >= nbckpbs {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDASetMaxStepB",
            file!(),
            MSGAM_BAD_WHICH,
        );
        return IDA_ILL_INPUT;
    }

    /* Find the IDABMem entry in the linked list corresponding to 'which'. */
    let IDAB_mem = idaa_bck_problem(&IDAADJ_mem, which);
    let ida_memB = idaa_bck_ida_mem(&IDAB_mem);

    IDASetMaxStep(&ida_memB, hmaxB)
}

pub fn IDASetSuppressAlgB(ida_mem: &IDAMem, which: i32, suppressalgB: sunbooleantype) -> i32 {
    /* Is ida_mem valid? NULL-mem check handled by type system */
    let IDA_mem = ida_mem;

    /* Is ASA initialized? */
    let adjMallocDone = IDA_mem.borrow().ida_adjMallocDone;
    if adjMallocDone == SUNFALSE {
        IDAProcessError(
            Some(IDA_mem),
            IDA_NO_ADJ,
            line!() as i32,
            "IDASetSuppressAlgB",
            file!(),
            MSGAM_NO_ADJ,
        );
        return IDA_NO_ADJ;
    }
    let IDAADJ_mem = idaa_adj_mem(IDA_mem);

    /* Check the value of which */
    let nbckpbs = IDAADJ_mem.borrow().ia_nbckpbs;
    if which >= nbckpbs {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDASetSuppressAlgB",
            file!(),
            MSGAM_BAD_WHICH,
        );
        return IDA_ILL_INPUT;
    }

    /* Find the IDABMem entry in the linked list corresponding to 'which'. */
    let IDAB_mem = idaa_bck_problem(&IDAADJ_mem, which);
    let ida_memB = idaa_bck_ida_mem(&IDAB_mem);

    IDASetSuppressAlg(&ida_memB, suppressalgB)
}

pub fn IDASetIdB(ida_mem: &IDAMem, which: i32, idB: Option<&N_Vector>) -> i32 {
    /* Is ida_mem valid? NULL-mem check handled by type system */
    let IDA_mem = ida_mem;

    /* Is ASA initialized? */
    let adjMallocDone = IDA_mem.borrow().ida_adjMallocDone;
    if adjMallocDone == SUNFALSE {
        IDAProcessError(
            Some(IDA_mem),
            IDA_NO_ADJ,
            line!() as i32,
            "IDASetIdB",
            file!(),
            MSGAM_NO_ADJ,
        );
        return IDA_NO_ADJ;
    }
    let IDAADJ_mem = idaa_adj_mem(IDA_mem);

    /* Check the value of which */
    let nbckpbs = IDAADJ_mem.borrow().ia_nbckpbs;
    if which >= nbckpbs {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDASetIdB",
            file!(),
            MSGAM_BAD_WHICH,
        );
        return IDA_ILL_INPUT;
    }

    /* Find the IDABMem entry in the linked list corresponding to 'which'. */
    let IDAB_mem = idaa_bck_problem(&IDAADJ_mem, which);
    let ida_memB = idaa_bck_ida_mem(&IDAB_mem);

    IDASetId(&ida_memB, idB)
}

pub fn IDASetConstraintsB(ida_mem: &IDAMem, which: i32, constraintsB: Option<&N_Vector>) -> i32 {
    /* Is ida_mem valid? NULL-mem check handled by type system */
    let IDA_mem = ida_mem;

    /* Is ASA initialized? */
    let adjMallocDone = IDA_mem.borrow().ida_adjMallocDone;
    if adjMallocDone == SUNFALSE {
        IDAProcessError(
            Some(IDA_mem),
            IDA_NO_ADJ,
            line!() as i32,
            "IDASetConstraintsB",
            file!(),
            MSGAM_NO_ADJ,
        );
        return IDA_NO_ADJ;
    }
    let IDAADJ_mem = idaa_adj_mem(IDA_mem);

    /* Check the value of which */
    let nbckpbs = IDAADJ_mem.borrow().ia_nbckpbs;
    if which >= nbckpbs {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDASetConstraintsB",
            file!(),
            MSGAM_BAD_WHICH,
        );
        return IDA_ILL_INPUT;
    }

    /* Find the IDABMem entry in the linked list corresponding to 'which'. */
    let IDAB_mem = idaa_bck_problem(&IDAADJ_mem, which);
    let ida_memB = idaa_bck_ida_mem(&IDAB_mem);

    IDASetConstraints(&ida_memB, constraintsB)
}

/*
 * ----------------------------------------------------------------
 * Input quadrature functions for ASA
 * ----------------------------------------------------------------
 */

pub fn IDASetQuadErrConB(ida_mem: &IDAMem, which: i32, errconQB: sunbooleantype) -> i32 {
    /* Is ida_mem valid? NULL-mem check handled by type system */
    let IDA_mem = ida_mem;

    /* Is ASA initialized? */
    let adjMallocDone = IDA_mem.borrow().ida_adjMallocDone;
    if adjMallocDone == SUNFALSE {
        IDAProcessError(
            Some(IDA_mem),
            IDA_NO_ADJ,
            line!() as i32,
            "IDASetQuadErrConB",
            file!(),
            MSGAM_NO_ADJ,
        );
        return IDA_NO_ADJ;
    }
    let IDAADJ_mem = idaa_adj_mem(IDA_mem);

    /* Check the value of which */
    let nbckpbs = IDAADJ_mem.borrow().ia_nbckpbs;
    if which >= nbckpbs {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDASetQuadErrConB",
            file!(),
            MSGAM_BAD_WHICH,
        );
        return IDA_ILL_INPUT;
    }

    /* Find the IDABMem entry in the linked list corresponding to 'which'. */
    let IDAB_mem = idaa_bck_problem(&IDAADJ_mem, which);
    let ida_memB = idaa_bck_ida_mem(&IDAB_mem);

    IDASetQuadErrCon(&ida_memB, errconQB)
}

/*
 * -----------------------------------------------------------------
 * Optional output functions for backward integration
 * -----------------------------------------------------------------
 */

/*
 * IDAGetAdjIDABmem
 *
 * This function returns a handle to the IDAS memory allocated for the
 * backward problem. This handle can then be used to call any of the
 * IDAGet* IDAS routines to extract optional output for the backward
 * integration phase. (C returns `void*`; NULL maps to `None`.)
 */

pub fn IDAGetAdjIDABmem(ida_mem: &IDAMem, which: i32) -> Option<IDAMem> {
    /* Is ida_mem valid? NULL-mem check handled by type system */
    let IDA_mem = ida_mem;

    /* Is ASA initialized? */
    let adjMallocDone = IDA_mem.borrow().ida_adjMallocDone;
    if adjMallocDone == SUNFALSE {
        IDAProcessError(
            Some(IDA_mem),
            0,
            line!() as i32,
            "IDAGetAdjIDABmem",
            file!(),
            MSGAM_NO_ADJ,
        );
        return None;
    }
    let IDAADJ_mem = idaa_adj_mem(IDA_mem);

    /* Check the value of which */
    let nbckpbs = IDAADJ_mem.borrow().ia_nbckpbs;
    if which >= nbckpbs {
        IDAProcessError(
            Some(IDA_mem),
            0,
            line!() as i32,
            "IDAGetAdjIDABmem",
            file!(),
            MSGAM_BAD_WHICH,
        );
        return None;
    }

    /* Find the IDABMem entry in the linked list corresponding to 'which'. */
    let IDAB_mem = idaa_bck_problem(&IDAADJ_mem, which);

    /* C copies the (possibly NULL) pointer without dereferencing it */
    let ida_memB = IDAB_mem.borrow().IDA_mem.clone();

    ida_memB
}

/*
 * IDAGetAdjCheckPointsInfo
 *
 * Loads an array of nckpnts structures of type IDAadjCheckPointRec
 * defined below.
 *
 * The user must allocate space for ckpnt (ncheck+1). (C
 * `IDAadjCheckPointRec*` becomes `&mut [IDAadjCheckPointRec]`; a short
 * slice panics where C would write past the end of the caller's buffer —
 * deviation class 5.)
 */

pub fn IDAGetAdjCheckPointsInfo(ida_mem: &IDAMem, ckpnt: &mut [IDAadjCheckPointRec]) -> i32 {
    /* Is ida_mem valid? NULL-mem check handled by type system */
    let IDA_mem = ida_mem;

    /* Is ASA initialized? */
    let adjMallocDone = IDA_mem.borrow().ida_adjMallocDone;
    if adjMallocDone == SUNFALSE {
        IDAProcessError(
            Some(IDA_mem),
            IDA_NO_ADJ,
            line!() as i32,
            "IDAGetAdjCheckPointsInfo",
            file!(),
            MSGAM_NO_ADJ,
        );
        return IDA_NO_ADJ;
    }
    let IDAADJ_mem = idaa_adj_mem(IDA_mem);

    let IDAADJ = IDAADJ_mem.borrow();

    /* i = 0; ck_mem = IDAADJ_mem->ck_mem;  (Vec index 0 = C list head) */
    let mut i: usize = 0;

    while i < IDAADJ.ck_mem.len() {
        let ck_mem = &IDAADJ.ck_mem[i];
        let ck = ck_mem.borrow();

        ckpnt[i].my_addr = Some(ck_mem.clone());
        ckpnt[i].next_addr = IDAADJ.ck_mem.get(i + 1).cloned();
        ckpnt[i].t0 = ck.ck_t0;
        ckpnt[i].t1 = ck.ck_t1;
        ckpnt[i].nstep = ck.ck_nst;
        ckpnt[i].order = ck.ck_kk;
        ckpnt[i].step = ck.ck_hh;

        /* ck_mem = ck_mem->ck_next; i++; */
        i += 1;
    }

    IDA_SUCCESS
}

/* IDAGetConsistentICB
 *
 * Returns the consistent initial conditions computed by IDACalcICB or
 * IDACalcICBS
 *
 * It must be preceded by a successful call to IDACalcICB or IDACalcICBS
 * for 'which' backward problem.
 */

pub fn IDAGetConsistentICB(
    ida_mem: &IDAMem,
    which: i32,
    yyB0_mod: Option<&N_Vector>,
    ypB0_mod: Option<&N_Vector>,
) -> i32 {
    /* Is ida_mem valid? NULL-mem check handled by type system */
    let IDA_mem = ida_mem;
    let flag: i32;

    /* Is ASA initialized? */
    let adjMallocDone = IDA_mem.borrow().ida_adjMallocDone;
    if adjMallocDone == SUNFALSE {
        IDAProcessError(
            Some(IDA_mem),
            IDA_NO_ADJ,
            line!() as i32,
            "IDAGetConsistentICB",
            file!(),
            MSGAM_NO_ADJ,
        );
        return IDA_NO_ADJ;
    }
    let IDAADJ_mem = idaa_adj_mem(IDA_mem);

    /* Check the value of which */
    let nbckpbs = IDAADJ_mem.borrow().ia_nbckpbs;
    if which >= nbckpbs {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDAGetConsistentICB",
            file!(),
            MSGAM_BAD_WHICH,
        );
        return IDA_ILL_INPUT;
    }

    /* Find the IDABMem entry in the linked list corresponding to 'which'. */
    let IDAB_mem = idaa_bck_problem(&IDAADJ_mem, which);
    let ida_memB = idaa_bck_ida_mem(&IDAB_mem);

    flag = IDAGetConsistentIC(&ida_memB, yyB0_mod, ypB0_mod);

    flag
}

/*-----------------------------------------------------------------*/

/// C `IDAGetUserDataB` returns the stored `void*` without ownership
/// transfer. The safe-Rust token cannot be aliased, so the stored box is
/// SWAPPED with `user_dataB`; the caller must hand it back (via
/// `IDASetUserDataB` or a second swap) before the integrator next invokes
/// a backward user callback.
pub fn IDAGetUserDataB(ida_mem: &IDAMem, which: i32, user_dataB: &mut Option<Box<dyn Any>>) -> i32 {
    /* Check if IDA_mem exists: NULL-mem check handled by type system */
    let IDA_mem = ida_mem;

    /* Was ASA initialized? */
    let adjMallocDone = IDA_mem.borrow().ida_adjMallocDone;
    if adjMallocDone == SUNFALSE {
        IDAProcessError(
            Some(IDA_mem),
            IDA_NO_ADJ,
            line!() as i32,
            "IDAGetUserDataB",
            file!(),
            MSGAM_NO_ADJ,
        );
        return IDA_NO_ADJ;
    }
    let IDAa_mem = idaa_adj_mem(IDA_mem);

    /* Check which */
    let nbckpbs = IDAa_mem.borrow().ia_nbckpbs;
    if which >= nbckpbs {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDAGetUserDataB",
            file!(),
            MSGAM_BAD_WHICH,
        );
        return IDA_ILL_INPUT;
    }

    /* Find the IDABMem entry in the linked list corresponding to which */
    let IDAB_mem = idaa_bck_problem(&IDAa_mem, which);

    std::mem::swap(&mut IDAB_mem.borrow_mut().ida_user_data, user_dataB);

    IDA_SUCCESS
}

/*
 * -----------------------------------------------------------------
 * Undocumented development user-callable functions
 * -----------------------------------------------------------------
 */

/*
 * -----------------------------------------------------------------
 * IDAGetAdjDataPointHermite
 * -----------------------------------------------------------------
 * Returns the 2 vectors stored for cubic Hermite interpolation at
 * the data point 'which'. The user must allocate space for yy and
 * yd.
 *
 * Returns IDA_MEM_NULL if ida_mem is NULL, IDA_ILL_INPUT if the
 * interpolation type previously specified is not IDA_HERMITE or
 * IDA_SUCCESS otherwise.
 *
 */
pub fn IDAGetAdjDataPointHermite(
    ida_mem: &IDAMem,
    which: i32,
    t: &mut sunrealtype,
    yy: Option<&N_Vector>,
    yd: Option<&N_Vector>,
) -> i32 {
    /* Is ida_mem valid? NULL-mem check handled by type system */
    let IDA_mem = ida_mem;

    /* Is ASA initialized? */
    let adjMallocDone = IDA_mem.borrow().ida_adjMallocDone;
    if adjMallocDone == SUNFALSE {
        IDAProcessError(
            Some(IDA_mem),
            IDA_NO_ADJ,
            line!() as i32,
            "IDAGetAdjDataPointHermite",
            file!(),
            MSGAM_NO_ADJ,
        );
        return IDA_NO_ADJ;
    }
    let IDAADJ_mem = idaa_adj_mem(IDA_mem);

    /* dt_mem = IDAADJ_mem->dt_mem; (indexed below) */

    let interpType = IDAADJ_mem.borrow().ia_interpType;
    if interpType != IDA_HERMITE {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDAGetAdjDataPointHermite",
            file!(),
            MSGAM_WRONG_INTERP,
        );
        return IDA_ILL_INPUT;
    }

    /* C indexes dt_mem[which] with no bound check (deviation class 5). */
    let dt_pnt = IDAADJ_mem.borrow().dt_mem[which as usize].clone();

    *t = dt_pnt.borrow().t;

    /* content = (IDAhermiteDataMem)dt_mem[which]->content; */
    let (content_y, content_yd) = {
        let d = dt_pnt.borrow();
        let content = d
            .content
            .as_ref()
            .expect("dt_mem[which]->content = NULL (C dereferences a NULL pointer here)")
            .downcast_ref::<IDAhermiteDataMemRec>()
            .expect("dt_mem[which]->content is not an IDAhermiteDataMem");
        (content.y.clone(), content.yd.clone())
    };

    if let Some(yy) = yy {
        N_VScale(
            ONE,
            content_y
                .as_ref()
                .expect("content->y = NULL (C dereferences a NULL pointer here)"),
            yy,
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

    IDA_SUCCESS
}

/*
 * IDAGetAdjDataPointPolynomial
 *
 * Returns the vector stored for polynomial interpolation at the
 * data point 'which'. The user must allocate space for y.
 *
 * Returns IDA_MEM_NULL if ida_mem is NULL, IDA_ILL_INPUT if the
 * interpolation type previously specified is not IDA_POLYNOMIAL or
 * IDA_SUCCESS otherwise.
 */

pub fn IDAGetAdjDataPointPolynomial(
    ida_mem: &IDAMem,
    which: i32,
    t: &mut sunrealtype,
    order: &mut i32,
    y: Option<&N_Vector>,
) -> i32 {
    /* Is ida_mem valid? NULL-mem check handled by type system */
    let IDA_mem = ida_mem;

    /* Is ASA initialized? */
    let adjMallocDone = IDA_mem.borrow().ida_adjMallocDone;
    if adjMallocDone == SUNFALSE {
        IDAProcessError(
            Some(IDA_mem),
            IDA_NO_ADJ,
            line!() as i32,
            "IDAGetAdjDataPointPolynomial",
            file!(),
            MSGAM_NO_ADJ,
        );
        return IDA_NO_ADJ;
    }
    let IDAADJ_mem = idaa_adj_mem(IDA_mem);

    /* dt_mem = IDAADJ_mem->dt_mem; (indexed below) */

    let interpType = IDAADJ_mem.borrow().ia_interpType;
    if interpType != IDA_POLYNOMIAL {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDAGetAdjDataPointPolynomial",
            file!(),
            MSGAM_WRONG_INTERP,
        );
        return IDA_ILL_INPUT;
    }

    /* C indexes dt_mem[which] with no bound check (deviation class 5). */
    let dt_pnt = IDAADJ_mem.borrow().dt_mem[which as usize].clone();

    *t = dt_pnt.borrow().t;

    /* content = (IDApolynomialDataMem)dt_mem[which]->content; */
    let (content_y, content_order) = {
        let d = dt_pnt.borrow();
        let content = d
            .content
            .as_ref()
            .expect("dt_mem[which]->content = NULL (C dereferences a NULL pointer here)")
            .downcast_ref::<IDApolynomialDataMemRec>()
            .expect("dt_mem[which]->content is not an IDApolynomialDataMem");
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

    IDA_SUCCESS
}

/*
 * IDAGetAdjCurrentCheckPoint
 *
 * Returns the address of the 'active' check point. (C `void** addr`
 * becomes `&mut Option<IDAckpntMem>`; the write is a handle clone, i.e. a
 * pointer copy with no ownership transfer.)
 */

pub fn IDAGetAdjCurrentCheckPoint(ida_mem: &IDAMem, addr: &mut Option<IDAckpntMem>) -> i32 {
    /* NULL-mem check handled by type system */
    let IDA_mem = ida_mem;

    /* Is ASA initialized? */
    let adjMallocDone = IDA_mem.borrow().ida_adjMallocDone;
    if adjMallocDone == SUNFALSE {
        IDAProcessError(
            Some(IDA_mem),
            IDA_NO_ADJ,
            line!() as i32,
            "IDAGetAdjCurrentCheckPoint",
            file!(),
            MSGAM_NO_ADJ,
        );
        return IDA_NO_ADJ;
    }
    let IDAADJ_mem = idaa_adj_mem(IDA_mem);

    *addr = IDAADJ_mem.borrow().ia_ckpntData.clone();

    IDA_SUCCESS
}
