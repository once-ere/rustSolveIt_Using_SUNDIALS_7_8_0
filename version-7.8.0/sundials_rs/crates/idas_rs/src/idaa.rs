//! Port of `src/idas/idaa.c` — the IDAA adjoint sensitivity module
//! (checkpointing of the forward run, the backward-problem list, the Hermite
//! and Newton-polynomial interpolation modules, and the adjoint wrappers).
//!
//! Structural mapping of the C intrusive lists (fixed by the contract in
//! [`crate::idas_impl`]):
//!
//! * `IDAADJ_mem->ck_mem` (checkpoint list, head = most recent checkpoint)
//!   becomes `IDAadjMemRec::ck_mem: Vec<IDAckpntMem>` with index 0 = list
//!   head. `ck_mem->ck_next` ≡ the next index; `ck_mem->ck_next == NULL` ≡
//!   "is the LAST element" (the `t_initial` checkpoint built by
//!   `IDAAckpntInit`). `IDAAckpntDelete(&head)` ≡ `ck_mem.remove(0)` —
//!   dropping the record releases exactly the `N_Vector`s that routine
//!   destroys (`ck_phi`/`ck_phiQ`/`ck_phiS`/`ck_phiQS`, `ck_phi_alloc` of
//!   each), so it has no separate Rust counterpart.
//! * `IDAADJ_mem->IDAB_mem` (backward-problem list, head = most recently
//!   created) becomes `IDAadjMemRec::IDAB_mem: Vec<IDABMem>` with index 0 =
//!   list head; `for (p = IDAB_mem; p; p = p->ida_next)` ≡
//!   `for p in IDAB_mem.iter()`, and the C "push at head" of `IDACreateB` is
//!   `IDAB_mem.insert(0, x)`.
//! * `IDAADJ_mem->dt_mem` (array of `steps+1` data points) becomes
//!   `Vec<IDAdtpntMem>`; `dt_mem[i]->content` is the `Option<Box<dyn Any>>`
//!   holding an `IDAhermiteDataMemRec` / `IDApolynomialDataMemRec` by value.
//!
//! Scratch-buffer note (same decision as `cvodes_rs::cvodea`): the C code
//! borrows `IDA_mem->ida_cvals` / `ida_Xvecs` / `ida_Zvecs` / `ida_dvals` as
//! scratch for the fused vector(-array) calls. Those are pure
//! write-then-read-immediately scratch areas, so this port builds the
//! identical arrays as function-local `Vec`s / arrays instead. The values,
//! the `nvec`/`nsum` arguments and therefore the arithmetic are bit-identical,
//! while a mutable borrow of the mem is never held across a vector operation.
//!
//! `IDAAdjInit` deviates from the C in exactly one unreachable place: when
//! `IDAAdataMalloc` fails, C `free()`s the adjoint record but leaves
//! `IDA_mem->ida_adj_mem` dangling; the port clears the field instead.

use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

use sundials_core::sundials_math::SUNRabs;
use sundials_core::sundials_nvector::{
    N_VClone, N_VCloneVectorArray, N_VDestroy, N_VLinearCombination,
    N_VLinearCombinationVectorArray, N_VLinearSum, N_VScale, N_VScaleVectorArray, N_Vector,
};
use sundials_core::sundials_types::{sunbooleantype, sunrealtype, SUNFALSE, SUNTRUE};

use crate::idas_impl::*;

/* -----------------------------------------------------------------
 * Symbols provided by the sibling idas modules (`idas.c` fragments,
 * `idas_ic.c` and `idas_io.c`). They are pulled through the crate
 * prelude so this module does not have to know how `idas.c` was split
 * into fragment modules; the prelude re-exports every idas module at
 * crate level.
 * -----------------------------------------------------------------*/
use crate::prelude::{
    IDACalcIC, IDACreate, IDAFree, IDAGetNumSteps, IDAGetQuad, IDAGetSolution, IDAInit,
    IDAQuadInit, IDAQuadReInit, IDAQuadSStolerances, IDAQuadSVtolerances, IDAQuadSensReInit,
    IDAReInit, IDASStolerances, IDASVtolerances, IDASensReInit, IDASetInitStep, IDASetStopTime,
    IDASetUserData, IDASolve,
};

/*=================================================================*/
/*                 IDAA Private Constants                          */
/*=================================================================*/

/* ZERO, ONE, TWO, HUNDRED and FUZZ_FACTOR (= 1.0e6, IDAA's own value)
live in `idas_impl` per the fragment-file protocol. */

/*=================================================================*/
/* Shortcuts for the handle model                                  */
/*=================================================================*/

/// C: `IDAADJ_mem = IDA_mem->ida_adj_mem;`
///
/// The C code dereferences this without a NULL test in several places
/// (`IDAGetAdjY`, the interpolation routines); a missing adjoint memory is
/// C undefined behavior and maps to a panic at the same site.
fn IDAADJ_mem_of(IDA_mem: &IDAMem) -> IDAadjMem {
    IDA_mem
        .borrow()
        .ida_adj_mem
        .clone()
        .expect("ida_adj_mem (IDAAdjInit not called)")
}

/// C: `dt_mem[i]`
fn dt_pnt(IDAADJ_mem: &IDAadjMem, i: i64) -> IDAdtpntMem {
    IDAADJ_mem.borrow().dt_mem[i as usize].clone()
}

/// C: `dt_mem[i]->t`
fn dt_t(IDAADJ_mem: &IDAadjMem, i: i64) -> sunrealtype {
    IDAADJ_mem.borrow().dt_mem[i as usize].borrow().t
}

/// C: `content = (IDAhermiteDataMem)(dt_mem[i]->content)` followed by reads
/// of `content->y`, `->yd`, `->yS`, `->ySd` (handle copies, as in C).
fn herm_content(
    IDAADJ_mem: &IDAadjMem,
    i: i64,
) -> (N_Vector, N_Vector, Vec<N_Vector>, Vec<N_Vector>) {
    let d = dt_pnt(IDAADJ_mem, i);
    let db = d.borrow();
    let content = db
        .content
        .as_ref()
        .expect("dt_mem content")
        .downcast_ref::<IDAhermiteDataMemRec>()
        .expect("Hermite content");
    (
        content.y.clone().expect("content->y"),
        content.yd.clone().expect("content->yd"),
        content.yS.clone(),
        content.ySd.clone(),
    )
}

/// C: `content = (IDApolynomialDataMem)(dt_mem[i]->content)`. `yd`/`ySd` are
/// non-NULL only for the first data point, hence the `Option`/possibly-empty
/// `Vec`.
fn poly_content(
    IDAADJ_mem: &IDAadjMem,
    i: i64,
) -> (
    N_Vector,
    Option<N_Vector>,
    Vec<N_Vector>,
    Vec<N_Vector>,
    i32,
) {
    let d = dt_pnt(IDAADJ_mem, i);
    let db = d.borrow();
    let content = db
        .content
        .as_ref()
        .expect("dt_mem content")
        .downcast_ref::<IDApolynomialDataMemRec>()
        .expect("polynomial content");
    (
        content.y.clone().expect("content->y"),
        content.yd.clone(),
        content.yS.clone(),
        content.ySd.clone(),
        content.order,
    )
}

/// C: `IDAB_mem = IDAADJ_mem->IDAB_mem; while (IDAB_mem != NULL) { if (which
/// == IDAB_mem->ida_index) break; IDAB_mem = IDAB_mem->ida_next; }`
///
/// This search is inlined eleven times in `idaa.c`; it is factored here with
/// identical semantics. If no entry matches, C leaves `IDAB_mem == NULL` and
/// dereferences it immediately (undefined behavior) — that maps to a panic.
fn IDAAfindBckpb(IDAADJ_mem: &IDAadjMem, which: i32) -> IDABMem {
    let list = IDAADJ_mem.borrow().IDAB_mem.clone();
    for IDAB_mem in list.iter() {
        if which == IDAB_mem.borrow().ida_index {
            return IDAB_mem.clone();
        }
    }
    panic!("no backward problem with index {}", which);
}

/*=================================================================*/
/*                  Exported Functions                             */
/*=================================================================*/

/*
 * IDAAdjInit
 *
 * This routine allocates space for the global IDAA memory
 * structure.
 */

pub fn IDAAdjInit(ida_mem: &IDAMem, steps: i64, interp: i32) -> i32 {
    /* Check arguments */

    /* NULL-mem check: handled by the type system */
    let IDA_mem = ida_mem;

    if steps <= 0 {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDAAdjInit",
            file!(),
            MSGAM_BAD_STEPS,
        );
        return IDA_ILL_INPUT;
    }

    if (interp != IDA_HERMITE) && (interp != IDA_POLYNOMIAL) {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDAAdjInit",
            file!(),
            MSGAM_BAD_INTERP,
        );
        return IDA_ILL_INPUT;
    }

    /* Allocate memory block for IDAadjMem. */
    let IDAADJ_mem: IDAadjMem = Rc::new(RefCell::new(IDAadjMemRec::zeroed()));

    /* Attach IDAS memory for forward runs */
    IDA_mem.borrow_mut().ida_adj_mem = Some(IDAADJ_mem.clone());

    {
        let mut ia = IDAADJ_mem.borrow_mut();

        /* Initialization of check points. */
        ia.ck_mem = Vec::new();
        ia.ia_nckpnts = 0;
        ia.ia_ckpntData = None;

        /* Initialization of interpolation data. */
        ia.ia_interpType = interp;
        ia.ia_nsteps = steps;

        /* Last index used in IDAAfindIndex, initialize to invalid value */
        ia.ia_ilast = -1;
    }

    /* Allocate space for the array of Data Point structures. */
    if !IDAAdataMalloc(IDA_mem) {
        IDA_mem.borrow_mut().ida_adj_mem = None;
        IDAProcessError(
            Some(IDA_mem),
            IDA_MEM_FAIL,
            line!() as i32,
            "IDAAdjInit",
            file!(),
            MSGAM_MEM_FAIL,
        );
        return IDA_MEM_FAIL;
    }

    {
        let mut ia = IDAADJ_mem.borrow_mut();

        /* Attach functions for the appropriate interpolation module */
        match interp {
            IDA_HERMITE => {
                ia.ia_malloc = Some(IDAAhermiteMalloc);
                ia.ia_free = Some(IDAAhermiteFree);
                ia.ia_getY = Some(IDAAhermiteGetY);
                ia.ia_storePnt = Some(IDAAhermiteStorePnt);
            }

            IDA_POLYNOMIAL => {
                ia.ia_malloc = Some(IDAApolynomialMalloc);
                ia.ia_free = Some(IDAApolynomialFree);
                ia.ia_getY = Some(IDAApolynomialGetY);
                ia.ia_storePnt = Some(IDAApolynomialStorePnt);
            }

            _ => {}
        }

        /* The interpolation module has not been initialized yet */
        ia.ia_mallocDone = SUNFALSE;

        /* By default we will store but not interpolate sensitivities
         *  - storeSensi will be set in IDASolveF to SUNFALSE if FSA is not enabled
         *    or if the user forced this through IDAAdjSetNoSensi
         *  - interpSensi will be set in IDASolveB to SUNTRUE if storeSensi is SUNTRUE
         *    and if at least one backward problem requires sensitivities
         *  - noInterp will be set in IDACalcICB to SUNTRUE before the call to
         *    IDACalcIC and SUNFALSE after.*/

        ia.ia_storeSensi = SUNTRUE;
        ia.ia_interpSensi = SUNFALSE;
        ia.ia_noInterp = SUNFALSE;

        /* Initialize backward problems. */
        ia.IDAB_mem = Vec::new();
        ia.ia_bckpbCrt = None;
        ia.ia_nbckpbs = 0;

        /* IDASolveF and IDASolveB not called yet. */
        ia.ia_firstIDAFcall = SUNTRUE;
        ia.ia_tstopIDAFcall = SUNFALSE;

        ia.ia_firstIDABcall = SUNTRUE;

        ia.ia_rootret = SUNFALSE;
    }

    /* Adjoint module initialized and allocated. */
    {
        let mut m = IDA_mem.borrow_mut();
        m.ida_adj = SUNTRUE;
        m.ida_adjMallocDone = SUNTRUE;
    }

    IDA_SUCCESS
}

/*
 * IDAAdjReInit
 *
 * IDAAdjReInit reinitializes the IDAS memory structure for ASA
 */

pub fn IDAAdjReInit(ida_mem: &IDAMem) -> i32 {
    /* Check arguments */

    /* NULL-mem check: handled by the type system */
    let IDA_mem = ida_mem;

    /* Was ASA previously initialized? */
    if !IDA_mem.borrow().ida_adjMallocDone {
        IDAProcessError(
            Some(IDA_mem),
            IDA_NO_ADJ,
            line!() as i32,
            "IDAAdjReInit",
            file!(),
            MSGAM_NO_ADJ,
        );
        return IDA_NO_ADJ;
    }

    let IDAADJ_mem = IDAADJ_mem_of(IDA_mem);

    let mut ia = IDAADJ_mem.borrow_mut();

    /* Free all stored  checkpoints. */
    ia.ck_mem.clear();

    ia.ia_nckpnts = 0;
    ia.ia_ckpntData = None;

    /* Flags for tracking the first calls to IDASolveF and IDASolveF. */
    ia.ia_firstIDAFcall = SUNTRUE;
    ia.ia_tstopIDAFcall = SUNFALSE;
    ia.ia_firstIDABcall = SUNTRUE;

    IDA_SUCCESS
}

/*
 * IDAAdjFree
 *
 * IDAAdjFree routine frees the memory allocated by IDAAdjInit.
 */

pub fn IDAAdjFree(ida_mem: &IDAMem) {
    /* NULL-mem check: handled by the type system */
    let IDA_mem = ida_mem;

    if IDA_mem.borrow().ida_adjMallocDone {
        /* Data for adjoint. */
        let IDAADJ_mem = IDAADJ_mem_of(IDA_mem);

        /* Delete check points one by one */
        IDAADJ_mem.borrow_mut().ck_mem.clear();

        IDAAdataFree(IDA_mem);

        /* Free all backward problems. */
        loop {
            let IDAB_mem = {
                let mut ia = IDAADJ_mem.borrow_mut();
                if ia.IDAB_mem.is_empty() {
                    None
                } else {
                    Some(ia.IDAB_mem.remove(0))
                }
            };
            match IDAB_mem {
                None => break,
                Some(IDAB_mem) => IDAAbckpbDelete(&IDAB_mem),
            }
        }

        /* Free IDAA memory. */
        let mut m = IDA_mem.borrow_mut();
        m.ida_adj_mem = None;
        m.ida_adjMallocDone = SUNFALSE;
    }
}

/*
 * =================================================================
 * PRIVATE FUNCTIONS FOR BACKWARD PROBLEMS
 * =================================================================
 */

/// C: `IDAAbckpbDelete(IDABMem* IDAB_memPtr)`. The list-head move is done by
/// the caller (`IDAADJ_mem->IDAB_mem.remove(0)`); this performs the teardown
/// of the removed entry.
fn IDAAbckpbDelete(IDAB_mem: &IDABMem) {
    /* Free IDAS memory for this backward problem. */
    let mut ida_mem = IDAB_mem.borrow_mut().IDA_mem.take();
    IDAFree(&mut ida_mem);

    /* Free linear solver memory. */
    let lfree = IDAB_mem.borrow().ida_lfree;
    if let Some(lfree) = lfree {
        let _ = lfree(IDAB_mem);
    }

    /* Free preconditioner memory. */
    let pfree = IDAB_mem.borrow().ida_pfree;
    if let Some(pfree) = pfree {
        let _ = pfree(IDAB_mem);
    }

    /* Free any workspace vectors. */
    let yy = IDAB_mem.borrow_mut().ida_yy.take();
    if let Some(yy) = yy {
        N_VDestroy(yy);
    }
    let yp = IDAB_mem.borrow_mut().ida_yp.take();
    if let Some(yp) = yp {
        N_VDestroy(yp);
    }
}

/*=================================================================*/
/*                    Wrappers for IDAA                            */
/*=================================================================*/

/*
 *                      IDASolveF
 *
 * This routine integrates to tout and returns solution into yout.
 * In the same time, it stores check point data every 'steps' steps.
 *
 * IDASolveF can be called repeatedly by the user. The last tout
 *  will be used as the starting time for the backward integration.
 *
 *  ncheckPtr points to the number of check points stored so far.
 */

pub fn IDASolveF(
    ida_mem: &IDAMem,
    tout: sunrealtype,
    tret: &mut sunrealtype,
    yret: &N_Vector,
    ypret: &N_Vector,
    itask: i32,
    ncheckPtr: &mut i32,
) -> i32 {
    /* Is the mem OK? NULL-mem check: handled by the type system */
    let IDA_mem = ida_mem;

    /* Is ASA initialized ? */
    if !IDA_mem.borrow().ida_adjMallocDone {
        IDAProcessError(
            Some(IDA_mem),
            IDA_NO_ADJ,
            line!() as i32,
            "IDASolveF",
            file!(),
            MSGAM_NO_ADJ,
        );
        return IDA_NO_ADJ;
    }
    let IDAADJ_mem = IDAADJ_mem_of(IDA_mem);

    /* Check for yret != NULL: handled by the type system */
    /* Check for ypret != NULL: handled by the type system */
    /* Check for tret != NULL: handled by the type system */

    /* Check for valid itask */
    if (itask != IDA_NORMAL) && (itask != IDA_ONE_STEP) {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDASolveF",
            file!(),
            MSG_BAD_ITASK,
        );
        return IDA_ILL_INPUT;
    }

    /* All memory checks done, proceed ... */

    let mut flag: i32 = 0;

    /* If tstop is enabled, store some info */
    let (tstopset, tstop) = {
        let m = IDA_mem.borrow();
        (m.ida_tstopset, m.ida_tstop)
    };
    if tstopset {
        let mut ia = IDAADJ_mem.borrow_mut();
        ia.ia_tstopIDAFcall = SUNTRUE;
        ia.ia_tstopIDAF = tstop;
    }

    /* On the first step:
     *   - set tinitial
     *   - initialize list of check points
     *   - if needed, initialize the interpolation module
     *   - load dt_mem[0]
     * On subsequent steps, test if taking a new step is necessary.
     */
    if IDAADJ_mem.borrow().ia_firstIDAFcall {
        let tn = IDA_mem.borrow().ida_tn;
        IDAADJ_mem.borrow_mut().ia_tinitial = tn;

        match IDAAckpntInit(IDA_mem) {
            None => {
                IDAProcessError(
                    Some(IDA_mem),
                    IDA_MEM_FAIL,
                    line!() as i32,
                    "IDASolveF",
                    file!(),
                    MSG_MEM_FAIL,
                );
                return IDA_MEM_FAIL;
            }
            Some(ck_mem) => {
                IDAADJ_mem.borrow_mut().ck_mem = vec![ck_mem];
            }
        }

        if !IDAADJ_mem.borrow().ia_mallocDone {
            /* Do we need to store sensitivities? */
            if !IDA_mem.borrow().ida_sensi {
                IDAADJ_mem.borrow_mut().ia_storeSensi = SUNFALSE;
            }

            /* Allocate space for interpolation data */
            let ia_malloc = IDAADJ_mem.borrow().ia_malloc.expect("ia_malloc");
            let allocOK = ia_malloc(IDA_mem);
            if !allocOK {
                IDAProcessError(
                    Some(IDA_mem),
                    IDA_MEM_FAIL,
                    line!() as i32,
                    "IDASolveF",
                    file!(),
                    MSG_MEM_FAIL,
                );
                return IDA_MEM_FAIL;
            }

            /* Rename phi and, if needed, phiS for use in interpolation
            (handle copies: ia_Y/ia_YS alias the integrator's divided
            difference arrays exactly as the C pointer copies do) */
            for i in 0..MXORDP1 {
                let phi_i = IDA_mem.borrow().ida_phi[i].clone();
                IDAADJ_mem.borrow_mut().ia_Y[i] = phi_i;
            }
            if IDAADJ_mem.borrow().ia_storeSensi {
                for i in 0..MXORDP1 {
                    let phiS_i = IDA_mem.borrow().ida_phiS[i].clone();
                    IDAADJ_mem.borrow_mut().ia_YS[i] = phiS_i;
                }
            }

            IDAADJ_mem.borrow_mut().ia_mallocDone = SUNTRUE;
        }

        let (dt0, ck_t0, storePnt) = {
            let ia = IDAADJ_mem.borrow();
            /* bind separately: the inner Ref must drop before `ia` does */
            let dt0 = ia.dt_mem[0].clone();
            let ck0 = ia.ck_mem[0].clone();
            let storePnt = ia.ia_storePnt.expect("ia_storePnt");
            let ck_t0 = ck0.borrow().ck_t0;
            (dt0, ck_t0, storePnt)
        };
        dt0.borrow_mut().t = ck_t0;
        let _ = storePnt(IDA_mem, &dt0);

        IDAADJ_mem.borrow_mut().ia_firstIDAFcall = SUNFALSE;
    } else if itask == IDA_NORMAL {
        /* When in normal mode, check if tout was passed or if a previous root was
        not reported and return an interpolated solution. No changes to ck_mem
        or dt_mem are needed. */

        /* flag to signal if an early return is needed */
        let mut earlyret = SUNFALSE;

        /* if a root needs to be reported compare tout to troot otherwise compare
        to the current time tn */
        let (rootret, troot) = {
            let ia = IDAADJ_mem.borrow();
            (ia.ia_rootret, ia.ia_troot)
        };
        let (tn, hh) = {
            let m = IDA_mem.borrow();
            (m.ida_tn, m.ida_hh)
        };
        let ttest = if rootret { troot } else { tn };

        if (ttest - tout) * hh >= ZERO {
            /* ttest is after tout, interpolate to tout */
            *tret = tout;
            flag = IDAGetSolution(IDA_mem, tout, yret, ypret);
            earlyret = SUNTRUE;
        } else if rootret {
            /* tout is after troot, interpolate to troot */
            *tret = troot;
            /* C assigns flag here and overwrites it on the next line
            (idaa.c:532-533); the (void)-call convention keeps the call */
            let _ = IDAGetSolution(IDA_mem, troot, yret, ypret);
            flag = IDA_ROOT_RETURN;
            IDAADJ_mem.borrow_mut().ia_rootret = SUNFALSE;
            earlyret = SUNTRUE;
        }

        /* return if necessary */
        if earlyret {
            let nst = IDA_mem.borrow().ida_nst;
            let mut ia = IDAADJ_mem.borrow_mut();
            *ncheckPtr = ia.ia_nckpnts;
            ia.ia_newData = SUNTRUE;
            let head = ia.ck_mem[0].clone();
            ia.ia_ckpntData = Some(head);
            let nsteps = ia.ia_nsteps;
            ia.ia_np = nst % nsteps + 1;
            return flag;
        }
    }

    /* Integrate to tout (in IDA_ONE_STEP mode) while loading check points */
    let mut nstloc: i64 = 0;
    loop {
        /* Check for too many steps */

        let (mxstep, tn_now) = {
            let m = IDA_mem.borrow();
            (m.ida_mxstep, m.ida_tn)
        };
        if (mxstep > 0) && (nstloc >= mxstep) {
            IDAProcessError(
                Some(IDA_mem),
                IDA_TOO_MUCH_WORK,
                line!() as i32,
                "IDASolveF",
                file!(),
                &MSG_MAX_STEPS(tn_now),
            );
            flag = IDA_TOO_MUCH_WORK;
            break;
        }

        /* Perform one step of the integration */

        flag = IDASolve(IDA_mem, tout, tret, yret, ypret, IDA_ONE_STEP);
        if flag < 0 {
            break;
        }

        nstloc += 1;

        /* Test if a new check point is needed */

        let (nst, tn) = {
            let m = IDA_mem.borrow();
            (m.ida_nst, m.ida_tn)
        };
        let nsteps = IDAADJ_mem.borrow().ia_nsteps;

        if nst % nsteps == 0 {
            {
                let ia = IDAADJ_mem.borrow();
                ia.ck_mem[0].borrow_mut().ck_t1 = tn;
            }

            /* Create a new check point, load it, and append it to the list */
            match IDAAckpntNew(IDA_mem) {
                None => {
                    flag = IDA_MEM_FAIL;
                    break;
                }
                Some(tmp) => {
                    let mut ia = IDAADJ_mem.borrow_mut();
                    ia.ck_mem.insert(0, tmp);
                    ia.ia_nckpnts += 1;
                }
            }

            IDA_mem.borrow_mut().ida_forceSetup = SUNTRUE;

            /* Reset i=0 and load dt_mem[0] */
            let (dt0, ck_t0, storePnt) = {
                let ia = IDAADJ_mem.borrow();
                /* bind separately: the inner Ref must drop before `ia` does */
                let dt0 = ia.dt_mem[0].clone();
                let ck0 = ia.ck_mem[0].clone();
                let storePnt = ia.ia_storePnt.expect("ia_storePnt");
                let ck_t0 = ck0.borrow().ck_t0;
                (dt0, ck_t0, storePnt)
            };
            dt0.borrow_mut().t = ck_t0;
            let _ = storePnt(IDA_mem, &dt0);
        } else {
            /* Load next point in dt_mem */
            let (dti, storePnt) = {
                let ia = IDAADJ_mem.borrow();
                (
                    ia.dt_mem[(nst % nsteps) as usize].clone(),
                    ia.ia_storePnt.expect("ia_storePnt"),
                )
            };
            dti.borrow_mut().t = tn;
            let _ = storePnt(IDA_mem, &dti);
        }

        /* Set t1 field of the current check point structure
        for the case in which there will be no future
        check points */
        {
            let ia = IDAADJ_mem.borrow();
            ia.ck_mem[0].borrow_mut().ck_t1 = tn;
        }

        /* tfinal is now set to tn */
        IDAADJ_mem.borrow_mut().ia_tfinal = tn;

        /* Return if in IDA_ONE_STEP mode */
        if itask == IDA_ONE_STEP {
            break;
        }

        /* IDA_NORMAL_STEP returns */

        /* Return if tout reached */
        let hh = IDA_mem.borrow().ida_hh;
        if (*tret - tout) * hh >= ZERO {
            /* If this was a root return, save the root time to return later */
            if flag == IDA_ROOT_RETURN {
                let mut ia = IDAADJ_mem.borrow_mut();
                ia.ia_rootret = SUNTRUE;
                ia.ia_troot = *tret;
            }

            /* Get solution value at tout to return now */
            *tret = tout;
            flag = IDAGetSolution(IDA_mem, tout, yret, ypret);

            /* Reset tretlast in IDA_mem so that IDAGetQuad and IDAGetSens
             * evaluate quadratures and/or sensitivities at the proper time */
            IDA_mem.borrow_mut().ida_tretlast = tout;

            break;
        }

        /* Return if tstop or a root was found */
        if (flag == IDA_TSTOP_RETURN) || (flag == IDA_ROOT_RETURN) {
            break;
        }
    } /* end of for(;;) */

    /* Get ncheck from IDAADJ_mem */
    let nst = IDA_mem.borrow().ida_nst;
    {
        let mut ia = IDAADJ_mem.borrow_mut();
        *ncheckPtr = ia.ia_nckpnts;

        /* Data is available for the last interval */
        ia.ia_newData = SUNTRUE;
        let head = ia.ck_mem[0].clone();
        ia.ia_ckpntData = Some(head);
        let nsteps = ia.ia_nsteps;
        ia.ia_np = nst % nsteps + 1;
    }

    flag
}

/*
 * =================================================================
 * FUNCTIONS FOR BACKWARD PROBLEMS
 * =================================================================
 */

pub fn IDACreateB(ida_mem: &IDAMem, which: &mut i32) -> i32 {
    /* Is the mem OK? NULL-mem check: handled by the type system */
    let IDA_mem = ida_mem;

    /* Is ASA initialized ? */
    if !IDA_mem.borrow().ida_adjMallocDone {
        IDAProcessError(
            Some(IDA_mem),
            IDA_NO_ADJ,
            line!() as i32,
            "IDACreateB",
            file!(),
            MSGAM_NO_ADJ,
        );
        return IDA_NO_ADJ;
    }
    let IDAADJ_mem = IDAADJ_mem_of(IDA_mem);

    /* Allocate a new IDABMem struct. */
    let mut new_IDAB_mem = IDABMemRec::zeroed();

    /* Allocate the IDAMem struct needed by this backward problem. */
    let sunctx = IDA_mem.borrow().ida_sunctx.clone();
    let ida_memB = match IDACreate(&sunctx) {
        None => {
            IDAProcessError(
                Some(IDA_mem),
                IDA_MEM_FAIL,
                line!() as i32,
                "IDACreateB",
                file!(),
                MSG_MEM_FAIL,
            );
            return IDA_MEM_FAIL;
        }
        Some(m) => m,
    };

    /* We need to ensure Ns is set in the new IDAS object so that Ns is accessible
    in the Python callbacks which only have access to ida_memB, not the original cvode_mem */
    let Ns = IDA_mem.borrow().ida_Ns;
    ida_memB.borrow_mut().ida_Ns = Ns;

    /* Save ida_mem in ida_memB as user data. */
    let _ = IDASetUserData(&ida_memB, Some(Box::new(IDA_mem.clone())));

    /* Initialize fields in the IDABMem struct. */
    new_IDAB_mem.ida_index = IDAADJ_mem.borrow().ia_nbckpbs;
    new_IDAB_mem.IDA_mem = Some(ida_memB);

    new_IDAB_mem.ida_res = None;
    new_IDAB_mem.ida_resS = None;
    new_IDAB_mem.ida_rhsQ = None;
    new_IDAB_mem.ida_rhsQS = None;

    new_IDAB_mem.ida_user_data = None;

    new_IDAB_mem.ida_lmem = None;
    new_IDAB_mem.ida_lfree = None;
    new_IDAB_mem.ida_pmem = None;
    new_IDAB_mem.ida_pfree = None;

    new_IDAB_mem.ida_yy = None;
    new_IDAB_mem.ida_yp = None;

    new_IDAB_mem.ida_res_withSensi = SUNFALSE;
    new_IDAB_mem.ida_rhsQ_withSensi = SUNFALSE;

    /* Attach the new object to the beginning of the linked list IDAADJ_mem->IDAB_mem. */
    let mut ia = IDAADJ_mem.borrow_mut();
    ia.IDAB_mem.insert(0, Rc::new(RefCell::new(new_IDAB_mem)));

    /* Return the assigned index. This id is used as identificator and has to be passed
    to IDAInitB and other ***B functions that set the optional inputs for  this
    backward problem. */
    *which = ia.ia_nbckpbs;

    /*Increase the counter of the backward problems stored. */
    ia.ia_nbckpbs += 1;

    IDA_SUCCESS
}

pub fn IDAInitB(
    ida_mem: &IDAMem,
    which: i32,
    resB: IDAResFnB,
    tB0: sunrealtype,
    yyB0: &N_Vector,
    ypB0: &N_Vector,
) -> i32 {
    /* NULL-mem check: handled by the type system */
    let IDA_mem = ida_mem;

    /* Is ASA initialized ? */
    if !IDA_mem.borrow().ida_adjMallocDone {
        IDAProcessError(
            Some(IDA_mem),
            IDA_NO_ADJ,
            line!() as i32,
            "IDAInitB",
            file!(),
            MSGAM_NO_ADJ,
        );
        return IDA_NO_ADJ;
    }
    let IDAADJ_mem = IDAADJ_mem_of(IDA_mem);

    /* Check the initial time for this backward problem against the adjoint data. */
    let (ia_tinitial, ia_tfinal) = {
        let ia = IDAADJ_mem.borrow();
        (ia.ia_tinitial, ia.ia_tfinal)
    };
    if (tB0 < ia_tinitial) || (tB0 > ia_tfinal) {
        IDAProcessError(
            Some(IDA_mem),
            IDA_BAD_TB0,
            line!() as i32,
            "IDAInitB",
            file!(),
            MSGAM_BAD_TB0,
        );
        return IDA_BAD_TB0;
    }

    /* Check the value of which */
    if which >= IDAADJ_mem.borrow().ia_nbckpbs {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDAInitB",
            file!(),
            MSGAM_BAD_WHICH,
        );
        return IDA_ILL_INPUT;
    }

    /* Find the IDABMem entry in the linked list corresponding to 'which'. */
    let IDAB_mem = IDAAfindBckpb(&IDAADJ_mem, which);

    /* Get the IDAMem corresponding to this backward problem. */
    let ida_memB = IDAB_mem
        .borrow()
        .IDA_mem
        .clone()
        .expect("IDAB_mem->IDA_mem");

    /* Call the IDAInit for this backward problem. */
    let flag = IDAInit(&ida_memB, IDAAres, tB0, yyB0, ypB0);
    if IDA_SUCCESS != flag {
        return flag;
    }

    {
        let mut b = IDAB_mem.borrow_mut();

        /* Copy residual function in IDAB_mem. */
        b.ida_res = Some(resB);
        b.ida_res_withSensi = SUNFALSE;

        /* Initialized the initial time field. */
        b.ida_t0 = tB0;
    }

    /* Allocate and initialize space workspace vectors. */
    /* NOTE (upstream): ida_yp is cloned from yyB0, not ypB0 — preserved. */
    let yy = N_VClone(yyB0).expect("N_VClone(yyB0)");
    let yp = N_VClone(yyB0).expect("N_VClone(yyB0)");
    N_VScale(ONE, yyB0, &yy);
    N_VScale(ONE, ypB0, &yp);
    {
        let mut b = IDAB_mem.borrow_mut();
        b.ida_yy = Some(yy);
        b.ida_yp = Some(yp);
    }

    flag
}

pub fn IDAInitBS(
    ida_mem: &IDAMem,
    which: i32,
    resS: IDAResFnBS,
    tB0: sunrealtype,
    yyB0: &N_Vector,
    ypB0: &N_Vector,
) -> i32 {
    /* NULL-mem check: handled by the type system */
    let IDA_mem = ida_mem;

    /* Is ASA initialized ? */
    if !IDA_mem.borrow().ida_adjMallocDone {
        IDAProcessError(
            Some(IDA_mem),
            IDA_NO_ADJ,
            line!() as i32,
            "IDAInitBS",
            file!(),
            MSGAM_NO_ADJ,
        );
        return IDA_NO_ADJ;
    }
    let IDAADJ_mem = IDAADJ_mem_of(IDA_mem);

    /* Check the initial time for this backward problem against the adjoint data. */
    let (ia_tinitial, ia_tfinal) = {
        let ia = IDAADJ_mem.borrow();
        (ia.ia_tinitial, ia.ia_tfinal)
    };
    if (tB0 < ia_tinitial) || (tB0 > ia_tfinal) {
        IDAProcessError(
            Some(IDA_mem),
            IDA_BAD_TB0,
            line!() as i32,
            "IDAInitBS",
            file!(),
            MSGAM_BAD_TB0,
        );
        return IDA_BAD_TB0;
    }

    /* Were sensitivities active during the forward integration? */
    if !IDAADJ_mem.borrow().ia_storeSensi {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDAInitBS",
            file!(),
            MSGAM_BAD_SENSI,
        );
        return IDA_ILL_INPUT;
    }

    /* Check the value of which */
    if which >= IDAADJ_mem.borrow().ia_nbckpbs {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDAInitBS",
            file!(),
            MSGAM_BAD_WHICH,
        );
        return IDA_ILL_INPUT;
    }

    /* Find the IDABMem entry in the linked list corresponding to 'which'. */
    let IDAB_mem = IDAAfindBckpb(&IDAADJ_mem, which);

    /* Get the IDAMem corresponding to this backward problem. */
    let ida_memB = IDAB_mem
        .borrow()
        .IDA_mem
        .clone()
        .expect("IDAB_mem->IDA_mem");

    /* Allocate and set the IDAS object */
    let flag = IDAInit(&ida_memB, IDAAres, tB0, yyB0, ypB0);

    if flag != IDA_SUCCESS {
        return flag;
    }

    {
        let mut b = IDAB_mem.borrow_mut();

        /* Copy residual function pointer in IDAB_mem. */
        b.ida_res_withSensi = SUNTRUE;
        b.ida_resS = Some(resS);

        /* Allocate space and initialize the yy and yp vectors. */
        b.ida_t0 = tB0;
    }

    let yy = N_VClone(yyB0).expect("N_VClone(yyB0)");
    let yp = N_VClone(ypB0).expect("N_VClone(ypB0)");
    N_VScale(ONE, yyB0, &yy);
    N_VScale(ONE, ypB0, &yp);
    {
        let mut b = IDAB_mem.borrow_mut();
        b.ida_yy = Some(yy);
        b.ida_yp = Some(yp);
    }

    IDA_SUCCESS
}

pub fn IDAReInitB(
    ida_mem: &IDAMem,
    which: i32,
    tB0: sunrealtype,
    yyB0: &N_Vector,
    ypB0: &N_Vector,
) -> i32 {
    /* NULL-mem check: handled by the type system */
    let IDA_mem = ida_mem;

    /* Is ASA initialized ? */
    if !IDA_mem.borrow().ida_adjMallocDone {
        IDAProcessError(
            Some(IDA_mem),
            IDA_NO_ADJ,
            line!() as i32,
            "IDAReInitB",
            file!(),
            MSGAM_NO_ADJ,
        );
        return IDA_NO_ADJ;
    }
    let IDAADJ_mem = IDAADJ_mem_of(IDA_mem);

    /* Check the initial time for this backward problem against the adjoint data. */
    let (ia_tinitial, ia_tfinal) = {
        let ia = IDAADJ_mem.borrow();
        (ia.ia_tinitial, ia.ia_tfinal)
    };
    if (tB0 < ia_tinitial) || (tB0 > ia_tfinal) {
        IDAProcessError(
            Some(IDA_mem),
            IDA_BAD_TB0,
            line!() as i32,
            "IDAReInitB",
            file!(),
            MSGAM_BAD_TB0,
        );
        return IDA_BAD_TB0;
    }

    /* Check the value of which */
    if which >= IDAADJ_mem.borrow().ia_nbckpbs {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDAReInitB",
            file!(),
            MSGAM_BAD_WHICH,
        );
        return IDA_ILL_INPUT;
    }

    /* Find the IDABMem entry in the linked list corresponding to 'which'. */
    let IDAB_mem = IDAAfindBckpb(&IDAADJ_mem, which);

    /* Get the IDAMem corresponding to this backward problem. */
    let ida_memB = IDAB_mem
        .borrow()
        .IDA_mem
        .clone()
        .expect("IDAB_mem->IDA_mem");

    /* Call the IDAReInit for this backward problem. */
    IDAReInit(&ida_memB, tB0, yyB0, ypB0)
}

pub fn IDASStolerancesB(
    ida_mem: &IDAMem,
    which: i32,
    relTolB: sunrealtype,
    absTolB: sunrealtype,
) -> i32 {
    /* Is ida_mem valid? NULL-mem check: handled by the type system */
    let IDA_mem = ida_mem;

    /* Is ASA initialized? */
    if !IDA_mem.borrow().ida_adjMallocDone {
        IDAProcessError(
            Some(IDA_mem),
            IDA_NO_ADJ,
            line!() as i32,
            "IDASStolerancesB",
            file!(),
            MSGAM_NO_ADJ,
        );
        return IDA_NO_ADJ;
    }
    let IDAADJ_mem = IDAADJ_mem_of(IDA_mem);

    /* Check the value of which */
    if which >= IDAADJ_mem.borrow().ia_nbckpbs {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDASStolerancesB",
            file!(),
            MSGAM_BAD_WHICH,
        );
        return IDA_ILL_INPUT;
    }

    /* Find the IDABMem entry in the linked list corresponding to 'which'. */
    let IDAB_mem = IDAAfindBckpb(&IDAADJ_mem, which);

    /* Get the IDAMem corresponding to this backward problem. */
    let ida_memB = IDAB_mem
        .borrow()
        .IDA_mem
        .clone()
        .expect("IDAB_mem->IDA_mem");

    /* Set tolerances and return. */
    IDASStolerances(&ida_memB, relTolB, absTolB)
}

pub fn IDASVtolerancesB(
    ida_mem: &IDAMem,
    which: i32,
    relTolB: sunrealtype,
    absTolB: &N_Vector,
) -> i32 {
    /* Is ida_mem valid? NULL-mem check: handled by the type system */
    let IDA_mem = ida_mem;

    /* Is ASA initialized? */
    if !IDA_mem.borrow().ida_adjMallocDone {
        IDAProcessError(
            Some(IDA_mem),
            IDA_NO_ADJ,
            line!() as i32,
            "IDASVtolerancesB",
            file!(),
            MSGAM_NO_ADJ,
        );
        return IDA_NO_ADJ;
    }
    let IDAADJ_mem = IDAADJ_mem_of(IDA_mem);

    /* Check the value of which */
    if which >= IDAADJ_mem.borrow().ia_nbckpbs {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDASVtolerancesB",
            file!(),
            MSGAM_BAD_WHICH,
        );
        return IDA_ILL_INPUT;
    }

    /* Find the IDABMem entry in the linked list corresponding to 'which'. */
    let IDAB_mem = IDAAfindBckpb(&IDAADJ_mem, which);

    /* Get the IDAMem corresponding to this backward problem. */
    let ida_memB = IDAB_mem
        .borrow()
        .IDA_mem
        .clone()
        .expect("IDAB_mem->IDA_mem");

    /* Set tolerances and return. */
    IDASVtolerances(&ida_memB, relTolB, absTolB)
}

pub fn IDAQuadSStolerancesB(
    ida_mem: &IDAMem,
    which: i32,
    reltolQB: sunrealtype,
    abstolQB: sunrealtype,
) -> i32 {
    /* Is ida_mem valid? NULL-mem check: handled by the type system */
    let IDA_mem = ida_mem;

    /* Is ASA initialized? */
    if !IDA_mem.borrow().ida_adjMallocDone {
        IDAProcessError(
            Some(IDA_mem),
            IDA_NO_ADJ,
            line!() as i32,
            "IDAQuadSStolerancesB",
            file!(),
            MSGAM_NO_ADJ,
        );
        return IDA_NO_ADJ;
    }
    let IDAADJ_mem = IDAADJ_mem_of(IDA_mem);

    /* Check the value of which */
    if which >= IDAADJ_mem.borrow().ia_nbckpbs {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDAQuadSStolerancesB",
            file!(),
            MSGAM_BAD_WHICH,
        );
        return IDA_ILL_INPUT;
    }

    /* Find the IDABMem entry in the linked list corresponding to 'which'. */
    let IDAB_mem = IDAAfindBckpb(&IDAADJ_mem, which);
    let ida_memB = IDAB_mem
        .borrow()
        .IDA_mem
        .clone()
        .expect("IDAB_mem->IDA_mem");

    IDAQuadSStolerances(&ida_memB, reltolQB, abstolQB)
}

pub fn IDAQuadSVtolerancesB(
    ida_mem: &IDAMem,
    which: i32,
    reltolQB: sunrealtype,
    abstolQB: &N_Vector,
) -> i32 {
    /* Is ida_mem valid? NULL-mem check: handled by the type system */
    let IDA_mem = ida_mem;

    /* Is ASA initialized? */
    if !IDA_mem.borrow().ida_adjMallocDone {
        IDAProcessError(
            Some(IDA_mem),
            IDA_NO_ADJ,
            line!() as i32,
            "IDAQuadSVtolerancesB",
            file!(),
            MSGAM_NO_ADJ,
        );
        return IDA_NO_ADJ;
    }
    let IDAADJ_mem = IDAADJ_mem_of(IDA_mem);

    /* Check the value of which */
    if which >= IDAADJ_mem.borrow().ia_nbckpbs {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDAQuadSVtolerancesB",
            file!(),
            MSGAM_BAD_WHICH,
        );
        return IDA_ILL_INPUT;
    }

    /* Find the IDABMem entry in the linked list corresponding to 'which'. */
    let IDAB_mem = IDAAfindBckpb(&IDAADJ_mem, which);
    let ida_memB = IDAB_mem
        .borrow()
        .IDA_mem
        .clone()
        .expect("IDAB_mem->IDA_mem");

    IDAQuadSVtolerances(&ida_memB, reltolQB, abstolQB)
}

pub fn IDAQuadInitB(ida_mem: &IDAMem, which: i32, rhsQB: IDAQuadRhsFnB, yQB0: &N_Vector) -> i32 {
    /* Is ida_mem valid? NULL-mem check: handled by the type system */
    let IDA_mem = ida_mem;

    /* Is ASA initialized? */
    if !IDA_mem.borrow().ida_adjMallocDone {
        IDAProcessError(
            Some(IDA_mem),
            IDA_NO_ADJ,
            line!() as i32,
            "IDAQuadInitB",
            file!(),
            MSGAM_NO_ADJ,
        );
        return IDA_NO_ADJ;
    }
    let IDAADJ_mem = IDAADJ_mem_of(IDA_mem);

    /* Check the value of which */
    if which >= IDAADJ_mem.borrow().ia_nbckpbs {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDAQuadInitB",
            file!(),
            MSGAM_BAD_WHICH,
        );
        return IDA_ILL_INPUT;
    }

    /* Find the IDABMem entry in the linked list corresponding to 'which'. */
    let IDAB_mem = IDAAfindBckpb(&IDAADJ_mem, which);
    let ida_memB = IDAB_mem
        .borrow()
        .IDA_mem
        .clone()
        .expect("IDAB_mem->IDA_mem");

    let flag = IDAQuadInit(&ida_memB, IDAArhsQ, yQB0);
    if IDA_SUCCESS != flag {
        return flag;
    }

    {
        let mut b = IDAB_mem.borrow_mut();
        b.ida_rhsQ_withSensi = SUNFALSE;
        b.ida_rhsQ = Some(rhsQB);
    }

    flag
}

pub fn IDAQuadInitBS(ida_mem: &IDAMem, which: i32, rhsQS: IDAQuadRhsFnBS, yQB0: &N_Vector) -> i32 {
    /* NULL-mem check: handled by the type system */
    let IDA_mem = ida_mem;

    /* Is ASA initialized ? */
    if !IDA_mem.borrow().ida_adjMallocDone {
        IDAProcessError(
            Some(IDA_mem),
            IDA_NO_ADJ,
            line!() as i32,
            "IDAQuadInitBS",
            file!(),
            MSGAM_NO_ADJ,
        );
        return IDA_NO_ADJ;
    }
    let IDAADJ_mem = IDAADJ_mem_of(IDA_mem);

    /* Check the value of which */
    if which >= IDAADJ_mem.borrow().ia_nbckpbs {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDAQuadInitBS",
            file!(),
            MSGAM_BAD_WHICH,
        );
        return IDA_ILL_INPUT;
    }

    /* Find the IDABMem entry in the linked list corresponding to 'which'. */
    let IDAB_mem = IDAAfindBckpb(&IDAADJ_mem, which);

    /* Get the IDAMem corresponding to this backward problem. */
    let ida_memB = IDAB_mem
        .borrow()
        .IDA_mem
        .clone()
        .expect("IDAB_mem->IDA_mem");

    /* Allocate and set the IDAS object */
    let flag = IDAQuadInit(&ida_memB, IDAArhsQ, yQB0);

    if flag != IDA_SUCCESS {
        return flag;
    }

    {
        /* Copy RHS function pointer in IDAB_mem and enable quad sensitivities. */
        let mut b = IDAB_mem.borrow_mut();
        b.ida_rhsQ_withSensi = SUNTRUE;
        b.ida_rhsQS = Some(rhsQS);
    }

    IDA_SUCCESS
}

pub fn IDAQuadReInitB(ida_mem: &IDAMem, which: i32, yQB0: &N_Vector) -> i32 {
    /* Is ida_mem valid? NULL-mem check: handled by the type system */
    let IDA_mem = ida_mem;

    /* Is ASA initialized? */
    if !IDA_mem.borrow().ida_adjMallocDone {
        IDAProcessError(
            Some(IDA_mem),
            IDA_NO_ADJ,
            line!() as i32,
            "IDAQuadReInitB",
            file!(),
            MSGAM_NO_ADJ,
        );
        return IDA_NO_ADJ;
    }
    let IDAADJ_mem = IDAADJ_mem_of(IDA_mem);

    /* Check the value of which */
    if which >= IDAADJ_mem.borrow().ia_nbckpbs {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDAQuadReInitB",
            file!(),
            MSGAM_BAD_WHICH,
        );
        return IDA_ILL_INPUT;
    }

    /* Find the IDABMem entry in the linked list corresponding to 'which'. */
    let IDAB_mem = IDAAfindBckpb(&IDAADJ_mem, which);
    let ida_memB = IDAB_mem
        .borrow()
        .IDA_mem
        .clone()
        .expect("IDAB_mem->IDA_mem");

    IDAQuadReInit(&ida_memB, yQB0)
}

/*
 * ----------------------------------------------------------------
 * Function : IDACalcICB
 * ----------------------------------------------------------------
 * IDACalcIC calculates corrected initial conditions for a DAE
 * backward system (index-one in semi-implicit form).
 * It uses Newton iteration combined with a Linesearch algorithm.
 * Calling IDACalcICB is optional. It is only necessary when the
 * initial conditions do not solve the given system.  I.e., if
 * yB0 and ypB0 are known to satisfy the backward problem, then
 * a call to IDACalcIC is NOT necessary (for index-one problems).
 */

pub fn IDACalcICB(
    ida_mem: &IDAMem,
    which: i32,
    tout1: sunrealtype,
    yy0: &N_Vector,
    yp0: &N_Vector,
) -> i32 {
    /* Is ida_mem valid? NULL-mem check: handled by the type system */
    let IDA_mem = ida_mem;

    /* Is ASA initialized? */
    if !IDA_mem.borrow().ida_adjMallocDone {
        IDAProcessError(
            Some(IDA_mem),
            IDA_NO_ADJ,
            line!() as i32,
            "IDACalcICB",
            file!(),
            MSGAM_NO_ADJ,
        );
        return IDA_NO_ADJ;
    }
    let IDAADJ_mem = IDAADJ_mem_of(IDA_mem);

    /* Check the value of which */
    if which >= IDAADJ_mem.borrow().ia_nbckpbs {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDACalcICB",
            file!(),
            MSGAM_BAD_WHICH,
        );
        return IDA_ILL_INPUT;
    }

    /* Find the IDABMem entry in the linked list corresponding to 'which'. */
    let IDAB_mem = IDAAfindBckpb(&IDAADJ_mem, which);
    let ida_memB = IDAB_mem
        .borrow()
        .IDA_mem
        .clone()
        .expect("IDAB_mem->IDA_mem");

    /* The wrapper for user supplied res function requires ia_bckpbCrt from
    IDAAdjMem to be set to current problem. */
    IDAADJ_mem.borrow_mut().ia_bckpbCrt = Some(IDAB_mem.clone());

    /* Save (y, y') in yyTmp and ypTmp for use in the res wrapper.*/
    /* yyTmp and ypTmp workspaces are safe to use if IDAADataStore is not called.*/
    let (yyTmp, ypTmp) = {
        let ia = IDAADJ_mem.borrow();
        (
            ia.ia_yyTmp.clone().expect("ia_yyTmp"),
            ia.ia_ypTmp.clone().expect("ia_ypTmp"),
        )
    };
    N_VScale(ONE, yy0, &yyTmp);
    N_VScale(ONE, yp0, &ypTmp);

    /* Set noInterp flag to SUNTRUE, so IDAARes will use user provided values for
    y and y' and will not call the interpolation routine(s). */
    IDAADJ_mem.borrow_mut().ia_noInterp = SUNTRUE;

    let flag = IDACalcIC(&ida_memB, IDA_YA_YDP_INIT, tout1);

    /* Set interpolation on in IDAARes. */
    IDAADJ_mem.borrow_mut().ia_noInterp = SUNFALSE;

    flag
}

/*
 * ----------------------------------------------------------------
 * Function : IDACalcICBS
 * ----------------------------------------------------------------
 * IDACalcIC calculates corrected initial conditions for a DAE
 * backward system (index-one in semi-implicit form) that also
 * dependes on the sensivities.
 *
 * It calls IDACalcIC for the 'which' backward problem.
 */

pub fn IDACalcICBS(
    ida_mem: &IDAMem,
    which: i32,
    tout1: sunrealtype,
    yy0: &N_Vector,
    yp0: &N_Vector,
    yyS0: &[N_Vector],
    ypS0: &[N_Vector],
) -> i32 {
    /* Is ida_mem valid? NULL-mem check: handled by the type system */
    let IDA_mem = ida_mem;

    /* Is ASA initialized? */
    if !IDA_mem.borrow().ida_adjMallocDone {
        IDAProcessError(
            Some(IDA_mem),
            IDA_NO_ADJ,
            line!() as i32,
            "IDACalcICBS",
            file!(),
            MSGAM_NO_ADJ,
        );
        return IDA_NO_ADJ;
    }
    let IDAADJ_mem = IDAADJ_mem_of(IDA_mem);

    /* Were sensitivities active during the forward integration? */
    if !IDAADJ_mem.borrow().ia_storeSensi {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDACalcICBS",
            file!(),
            MSGAM_BAD_SENSI,
        );
        return IDA_ILL_INPUT;
    }

    /* Check the value of which */
    if which >= IDAADJ_mem.borrow().ia_nbckpbs {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDACalcICBS",
            file!(),
            MSGAM_BAD_WHICH,
        );
        return IDA_ILL_INPUT;
    }

    /* Find the IDABMem entry in the linked list corresponding to 'which'. */
    let IDAB_mem = IDAAfindBckpb(&IDAADJ_mem, which);
    let ida_memB = IDAB_mem
        .borrow()
        .IDA_mem
        .clone()
        .expect("IDAB_mem->IDA_mem");

    /* Was InitBS called for this problem? */
    if !IDAB_mem.borrow().ida_res_withSensi {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDACalcICBS",
            file!(),
            MSGAM_NO_INITBS,
        );
        return IDA_ILL_INPUT;
    }

    /* The wrapper for user supplied res function requires ia_bckpbCrt from
    IDAAdjMem to be set to current problem. */
    IDAADJ_mem.borrow_mut().ia_bckpbCrt = Some(IDAB_mem.clone());

    /* Save (y, y') and (y_p, y'_p) in yyTmp, ypTmp and yySTmp, ypSTmp.The wrapper
    for residual will use these values instead of calling interpolation routine.*/

    /* The four workspaces variables are safe to use if IDAADataStore is not called.*/
    let (yyTmp, ypTmp, yySTmp, ypSTmp) = {
        let ia = IDAADJ_mem.borrow();
        (
            ia.ia_yyTmp.clone().expect("ia_yyTmp"),
            ia.ia_ypTmp.clone().expect("ia_ypTmp"),
            ia.ia_yySTmp.clone(),
            ia.ia_ypSTmp.clone(),
        )
    };
    N_VScale(ONE, yy0, &yyTmp);
    N_VScale(ONE, yp0, &ypTmp);

    let Ns = IDA_mem.borrow().ida_Ns;
    let cvals = vec![ONE; Ns as usize];

    let retval = N_VScaleVectorArray(Ns, &cvals, yyS0, &yySTmp);
    if retval != IDA_SUCCESS {
        return IDA_VECTOROP_ERR;
    }

    let retval = N_VScaleVectorArray(Ns, &cvals, ypS0, &ypSTmp);
    if retval != IDA_SUCCESS {
        return IDA_VECTOROP_ERR;
    }

    /* Set noInterp flag to SUNTRUE, so IDAARes will use user provided values for
    y and y' and will not call the interpolation routine(s). */
    IDAADJ_mem.borrow_mut().ia_noInterp = SUNTRUE;

    let flag = IDACalcIC(&ida_memB, IDA_YA_YDP_INIT, tout1);

    /* Set interpolation on in IDAARes. */
    IDAADJ_mem.borrow_mut().ia_noInterp = SUNFALSE;

    flag
}

/*
 * IDASolveB
 *
 * This routine performs the backward integration from tB0
 * to tinitial through a sequence of forward-backward runs in
 * between consecutive check points. It returns the values of
 * the adjoint variables and any existing quadrature variables
 * at tinitial.
 *
 * On a successful return, IDASolveB returns IDA_SUCCESS.
 *
 * NOTE that IDASolveB DOES NOT return the solution for the
 * backward problem(s). Use IDAGetB to extract the solution
 * for any given backward problem.
 *
 * If there are multiple backward problems and multiple check points,
 * IDASolveB may not succeed in getting all problems to take one step
 * when called in ONE_STEP mode.
 */

pub fn IDASolveB(ida_mem: &IDAMem, tBout: sunrealtype, itaskB: i32) -> i32 {
    /* Is the mem OK? NULL-mem check: handled by the type system */
    let IDA_mem = ida_mem;

    /* C modifies its `tBout` parameter in the fuzz-tolerance branch below */
    let mut tBout = tBout;
    let mut flag: i32 = 0;

    /* Is ASA initialized ? */
    if !IDA_mem.borrow().ida_adjMallocDone {
        IDAProcessError(
            Some(IDA_mem),
            IDA_NO_ADJ,
            line!() as i32,
            "IDASolveB",
            file!(),
            MSGAM_NO_ADJ,
        );
        return IDA_NO_ADJ;
    }
    let IDAADJ_mem = IDAADJ_mem_of(IDA_mem);

    if IDAADJ_mem.borrow().ia_nbckpbs == 0 {
        IDAProcessError(
            Some(IDA_mem),
            IDA_NO_BCK,
            line!() as i32,
            "IDASolveB",
            file!(),
            MSGAM_NO_BCK,
        );
        return IDA_NO_BCK;
    }
    let IDAB_mem: Vec<IDABMem> = IDAADJ_mem.borrow().IDAB_mem.clone();

    /* Check whether IDASolveF has been called */
    if IDAADJ_mem.borrow().ia_firstIDAFcall {
        IDAProcessError(
            Some(IDA_mem),
            IDA_NO_FWD,
            line!() as i32,
            "IDASolveB",
            file!(),
            MSGAM_NO_FWD,
        );
        return IDA_NO_FWD;
    }

    let (ia_tinitial, ia_tfinal) = {
        let ia = IDAADJ_mem.borrow();
        (ia.ia_tinitial, ia.ia_tfinal)
    };
    let sign: sunrealtype = if ia_tfinal - ia_tinitial > ZERO {
        1.0
    } else {
        -1.0
    };

    /* If this is the first call, loop over all backward problems and
     *   - check that tB0 is valid
     *   - check that tBout is ahead of tB0 in the backward direction
     *   - check whether we need to interpolate forward sensitivities
     */
    if IDAADJ_mem.borrow().ia_firstIDABcall {
        /* First IDABMem struct. */
        for tmp_IDAB_mem in IDAB_mem.iter() {
            let (bmem, res_withSensi, rhsQ_withSensi) = {
                let b = tmp_IDAB_mem.borrow();
                (
                    b.IDA_mem.clone().expect("IDAB_mem->IDA_mem"),
                    b.ida_res_withSensi,
                    b.ida_rhsQ_withSensi,
                )
            };
            let tBn = bmem.borrow().ida_tn;

            if (sign * (tBn - ia_tinitial) < ZERO) || (sign * (ia_tfinal - tBn) < ZERO) {
                IDAProcessError(
                    Some(IDA_mem),
                    IDA_BAD_TB0,
                    line!() as i32,
                    "IDASolveB",
                    file!(),
                    MSGAM_BAD_TB0,
                );
                return IDA_BAD_TB0;
            }

            if sign * (tBn - tBout) <= ZERO {
                IDAProcessError(
                    Some(IDA_mem),
                    IDA_ILL_INPUT,
                    line!() as i32,
                    "IDASolveB",
                    file!(),
                    MSGAM_BAD_TBOUT,
                );
                return IDA_ILL_INPUT;
            }

            if res_withSensi || rhsQ_withSensi {
                IDAADJ_mem.borrow_mut().ia_interpSensi = SUNTRUE;
            }
        }

        let (interpSensi, storeSensi) = {
            let ia = IDAADJ_mem.borrow();
            (ia.ia_interpSensi, ia.ia_storeSensi)
        };
        if interpSensi && !storeSensi {
            IDAProcessError(
                Some(IDA_mem),
                IDA_ILL_INPUT,
                line!() as i32,
                "IDASolveB",
                file!(),
                MSGAM_BAD_SENSI,
            );
            return IDA_ILL_INPUT;
        }

        IDAADJ_mem.borrow_mut().ia_firstIDABcall = SUNFALSE;
    }

    /* Check for valid itask */
    if (itaskB != IDA_NORMAL) && (itaskB != IDA_ONE_STEP) {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDASolveB",
            file!(),
            MSG_BAD_ITASK,
        );
        return IDA_ILL_INPUT;
    }

    /* Check if tBout is legal */
    if (sign * (tBout - ia_tinitial) < ZERO) || (sign * (ia_tfinal - tBout) < ZERO) {
        let uround = IDA_mem.borrow().ida_uround;
        let tfuzz = HUNDRED * uround * (SUNRabs(ia_tinitial) + SUNRabs(ia_tfinal));
        if (sign * (tBout - ia_tinitial) < ZERO) && (SUNRabs(tBout - ia_tinitial) < tfuzz) {
            tBout = ia_tinitial;
        } else {
            IDAProcessError(
                Some(IDA_mem),
                IDA_ILL_INPUT,
                line!() as i32,
                "IDASolveB",
                file!(),
                MSGAM_BAD_TBOUT,
            );
            return IDA_ILL_INPUT;
        }
    }

    /* Loop through the check points and stop as soon as a backward
     * problem has its tn value behind the current check point's t0_
     * value (in the backward direction) */

    let mut ck_idx: usize = 0;

    let mut gotCkpnt = SUNFALSE;

    loop {
        let ck_t0 = {
            let ck = IDAADJ_mem.borrow().ck_mem[ck_idx].clone();
            let t0 = ck.borrow().ck_t0;
            t0
        };

        for tmp_IDAB_mem in IDAB_mem.iter() {
            let bmem = tmp_IDAB_mem
                .borrow()
                .IDA_mem
                .clone()
                .expect("IDAB_mem->IDA_mem");
            let tBn = bmem.borrow().ida_tn;

            if sign * (tBn - ck_t0) > ZERO {
                gotCkpnt = SUNTRUE;
                break;
            }

            if (itaskB == IDA_NORMAL) && (tBn == ck_t0) && (sign * (tBout - ck_t0) >= ZERO) {
                gotCkpnt = SUNTRUE;
                break;
            }
        }

        if gotCkpnt {
            break;
        }

        /* C: if (ck_mem->ck_next == NULL) break; */
        if ck_idx + 1 >= IDAADJ_mem.borrow().ck_mem.len() {
            break;
        }

        ck_idx += 1;
    }

    /* Loop while propagating backward problems */
    loop {
        let ck_mem = {
            let ia = IDAADJ_mem.borrow();
            ia.ck_mem[ck_idx].clone()
        };

        /* Store interpolation data if not available.
        This is the 2nd forward integration pass */
        let is_ckpntData = {
            let ia = IDAADJ_mem.borrow();
            match &ia.ia_ckpntData {
                None => false,
                Some(c) => Rc::ptr_eq(c, &ck_mem),
            }
        };
        if !is_ckpntData {
            flag = IDAAdataStore(IDA_mem, &ck_mem);
            if flag != IDA_SUCCESS {
                break;
            }
        }

        /* Starting with the current check point from above, loop over check points
        while propagating backward problems */

        let ck_t0 = ck_mem.borrow().ck_t0;
        let mut errIndex: i32 = 0;

        for tmp_IDAB_mem in IDAB_mem.iter() {
            /* Decide if current backward problem is "active" in this check point */
            let mut isActive = SUNTRUE;

            let bmem = tmp_IDAB_mem
                .borrow()
                .IDA_mem
                .clone()
                .expect("IDAB_mem->IDA_mem");
            let tBn = bmem.borrow().ida_tn;

            if (tBn == ck_t0) && (sign * (tBout - ck_t0) < ZERO) {
                isActive = SUNFALSE;
            }
            if (tBn == ck_t0) && (itaskB == IDA_ONE_STEP) {
                isActive = SUNFALSE;
            }
            if sign * (tBn - ck_t0) < ZERO {
                isActive = SUNFALSE;
            }

            if isActive {
                /* Store the address of current backward problem memory
                 * in IDAADJ_mem to be used in the wrapper functions */
                IDAADJ_mem.borrow_mut().ia_bckpbCrt = Some(tmp_IDAB_mem.clone());

                /* Integrate current backward problem */
                let _ = IDASetStopTime(&bmem, ck_t0);
                let (yy, yp) = {
                    let b = tmp_IDAB_mem.borrow();
                    (
                        b.ida_yy.clone().expect("IDAB_mem->ida_yy"),
                        b.ida_yp.clone().expect("IDAB_mem->ida_yp"),
                    )
                };
                let mut tBret: sunrealtype = ZERO;
                flag = IDASolve(&bmem, tBout, &mut tBret, &yy, &yp, itaskB);

                /* Set the time at which we will report solution and/or quadratures */
                tmp_IDAB_mem.borrow_mut().ida_tout = tBret;

                /* If an error occurred, exit while loop */
                if flag < 0 {
                    errIndex = tmp_IDAB_mem.borrow().ida_index;
                    break;
                }
            } else {
                flag = IDA_SUCCESS;
                tmp_IDAB_mem.borrow_mut().ida_tout = tBn;
            }

            /* Move to next backward problem */
        } /* End of while: iteration through backward problems. */

        /* If an error occurred, return now */
        if flag < 0 {
            IDAProcessError(
                Some(IDA_mem),
                flag,
                line!() as i32,
                "IDASolveB",
                file!(),
                &MSGAM_BACK_ERROR(errIndex),
            );
            return flag;
        }

        /* If in IDA_ONE_STEP mode, return now (flag = IDA_SUCCESS) */
        if itaskB == IDA_ONE_STEP {
            break;
        }

        /* If all backward problems have successfully reached tBout, return now */
        let mut reachedTBout = SUNTRUE;

        for tmp_IDAB_mem in IDAB_mem.iter() {
            let ida_tout = tmp_IDAB_mem.borrow().ida_tout;
            if sign * (ida_tout - tBout) > ZERO {
                reachedTBout = SUNFALSE;
                break;
            }
        }

        if reachedTBout {
            break;
        }

        /* Move check point in linked list to next one */
        ck_idx += 1;
    } /* End of loop. */

    flag
}

/*
 * IDAGetB
 *
 * IDAGetB returns the state variables at the same time (also returned
 * in tret) as that at which IDASolveBreturned the solution.
 */

pub fn IDAGetB(
    ida_mem: &IDAMem,
    which: i32,
    tret: &mut sunrealtype,
    yy: &N_Vector,
    yp: &N_Vector,
) -> i32 {
    /* Is ida_mem valid? NULL-mem check: handled by the type system */
    let IDA_mem = ida_mem;

    /* Is ASA initialized? */
    if !IDA_mem.borrow().ida_adjMallocDone {
        IDAProcessError(
            Some(IDA_mem),
            IDA_NO_ADJ,
            line!() as i32,
            "IDAGetB",
            file!(),
            MSGAM_NO_ADJ,
        );
        return IDA_NO_ADJ;
    }
    let IDAADJ_mem = IDAADJ_mem_of(IDA_mem);

    /* Check the value of which */
    if which >= IDAADJ_mem.borrow().ia_nbckpbs {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDAGetB",
            file!(),
            MSGAM_BAD_WHICH,
        );
        return IDA_ILL_INPUT;
    }

    /* Find the IDABMem entry in the linked list corresponding to 'which'. */
    let IDAB_mem = IDAAfindBckpb(&IDAADJ_mem, which);

    let (b_yy, b_yp, ida_tout) = {
        let b = IDAB_mem.borrow();
        (
            b.ida_yy.clone().expect("IDAB_mem->ida_yy"),
            b.ida_yp.clone().expect("IDAB_mem->ida_yp"),
            b.ida_tout,
        )
    };
    N_VScale(ONE, &b_yy, yy);
    N_VScale(ONE, &b_yp, yp);
    *tret = ida_tout;

    IDA_SUCCESS
}

/*
 * IDAGetQuadB
 *
 * IDAGetQuadB returns the quadrature variables at the same
 * time (also returned in tret) as that at which IDASolveB
 * returned the solution.
 */

pub fn IDAGetQuadB(ida_mem: &IDAMem, which: i32, tret: &mut sunrealtype, qB: &N_Vector) -> i32 {
    /* Is ida_mem valid? NULL-mem check: handled by the type system */
    let IDA_mem = ida_mem;

    /* Is ASA initialized? */
    if !IDA_mem.borrow().ida_adjMallocDone {
        IDAProcessError(
            Some(IDA_mem),
            IDA_NO_ADJ,
            line!() as i32,
            "IDAGetQuadB",
            file!(),
            MSGAM_NO_ADJ,
        );
        return IDA_NO_ADJ;
    }
    let IDAADJ_mem = IDAADJ_mem_of(IDA_mem);

    /* Check the value of which */
    if which >= IDAADJ_mem.borrow().ia_nbckpbs {
        IDAProcessError(
            Some(IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "IDAGetQuadB",
            file!(),
            MSGAM_BAD_WHICH,
        );
        return IDA_ILL_INPUT;
    }

    /* Find the IDABMem entry in the linked list corresponding to 'which'. */
    let IDAB_mem = IDAAfindBckpb(&IDAADJ_mem, which);
    let ida_memB = IDAB_mem
        .borrow()
        .IDA_mem
        .clone()
        .expect("IDAB_mem->IDA_mem");

    /* If the integration for this backward problem has not started yet,
     * simply return the current value of qB (i.e. the final conditions) */

    let mut nstB: i64 = 0;
    let mut flag = IDAGetNumSteps(&ida_memB, &mut nstB);
    if IDA_SUCCESS != flag {
        return flag;
    }

    if nstB == 0 {
        let phiQ0 = ida_memB.borrow().ida_phiQ[0]
            .clone()
            .expect("IDAB_mem->IDA_mem->ida_phiQ[0]");
        N_VScale(ONE, &phiQ0, qB);
        *tret = IDAB_mem.borrow().ida_tout;
    } else {
        flag = IDAGetQuad(&ida_memB, tret, qB);
    }
    flag
}

/*=================================================================*/
/*                Private Functions Implementation                 */
/*=================================================================*/

/*
 * IDAAckpntInit
 *
 * This routine initializes the check point linked list with
 * information from the initial time.
 */

fn IDAAckpntInit(IDA_mem: &IDAMem) -> Option<IDAckpntMem> {
    /* Allocate space for ckdata */
    let mut ck_mem = IDAckpntMemRec::zeroed();

    let (tn, quadr, errconQ, sensi, Ns, quadr_sensi, errconQS) = {
        let m = IDA_mem.borrow();
        (
            m.ida_tn,
            m.ida_quadr,
            m.ida_errconQ,
            m.ida_sensi,
            m.ida_Ns,
            m.ida_quadr_sensi,
            m.ida_errconQS,
        )
    };

    ck_mem.ck_t0 = tn;
    ck_mem.ck_nst = 0;
    ck_mem.ck_kk = 1;
    ck_mem.ck_hh = ZERO;

    /* Test if we need to carry quadratures */
    ck_mem.ck_quadr = quadr && errconQ;

    /* Test if we need to carry sensitivities */
    ck_mem.ck_sensi = sensi;
    if ck_mem.ck_sensi {
        ck_mem.ck_Ns = Ns;
    }

    /* Test if we need to carry quadrature sensitivities */
    ck_mem.ck_quadr_sensi = quadr_sensi && errconQS;

    /* Alloc 3: current order, i.e. 1,  +   2. */
    ck_mem.ck_phi_alloc = 3;

    if !IDAAckpntAllocVectors(IDA_mem, &mut ck_mem) {
        /* Dropping the partially built record destroys exactly the vectors
        the C cleanup loops free. */
        return None;
    }
    /* Save phi* vectors from IDA_mem to ck_mem. */
    IDAAckpntCopyVectors(IDA_mem, &mut ck_mem);

    /* Next in list: the caller places this record at the head of ck_mem */

    Some(Rc::new(RefCell::new(ck_mem)))
}

/*
 * IDAAckpntNew
 *
 * This routine allocates space for a new check point and sets
 * its data from current values in IDA_mem.
 */

fn IDAAckpntNew(IDA_mem: &IDAMem) -> Option<IDAckpntMem> {
    /* Allocate space for ckdata */
    let mut ck_mem = IDAckpntMemRec::zeroed();

    {
        let m = IDA_mem.borrow();

        ck_mem.ck_nst = m.ida_nst;
        ck_mem.ck_tretlast = m.ida_tretlast;
        ck_mem.ck_kk = m.ida_kk;
        ck_mem.ck_kused = m.ida_kused;
        ck_mem.ck_knew = m.ida_knew;
        ck_mem.ck_phase = m.ida_phase;
        ck_mem.ck_ns = m.ida_ns;
        ck_mem.ck_hh = m.ida_hh;
        ck_mem.ck_hused = m.ida_hused;
        ck_mem.ck_eta = m.ida_eta;
        ck_mem.ck_cj = m.ida_cj;
        ck_mem.ck_cjlast = m.ida_cjlast;
        ck_mem.ck_cjold = m.ida_cjold;
        ck_mem.ck_cjratio = m.ida_cjratio;
        ck_mem.ck_ss = m.ida_ss;
        ck_mem.ck_ssS = m.ida_ssS;
        ck_mem.ck_t0 = m.ida_tn;

        for j in 0..MXORDP1 {
            ck_mem.ck_psi[j] = m.ida_psi[j];
            ck_mem.ck_alpha[j] = m.ida_alpha[j];
            ck_mem.ck_beta[j] = m.ida_beta[j];
            ck_mem.ck_sigma[j] = m.ida_sigma[j];
            ck_mem.ck_gamma[j] = m.ida_gamma[j];
        }

        /* Test if we need to carry quadratures */
        ck_mem.ck_quadr = m.ida_quadr && m.ida_errconQ;

        /* Test if we need to carry sensitivities */
        ck_mem.ck_sensi = m.ida_sensi;
        if ck_mem.ck_sensi {
            ck_mem.ck_Ns = m.ida_Ns;
        }

        /* Test if we need to carry quadrature sensitivities */
        ck_mem.ck_quadr_sensi = m.ida_quadr_sensi && m.ida_errconQS;

        ck_mem.ck_phi_alloc = if (m.ida_kk + 2) < MXORDP1 as i32 {
            m.ida_kk + 2
        } else {
            MXORDP1 as i32
        };
    }

    if !IDAAckpntAllocVectors(IDA_mem, &mut ck_mem) {
        return None;
    }

    /* Save phi* vectors from IDA_mem to ck_mem. */
    IDAAckpntCopyVectors(IDA_mem, &mut ck_mem);

    Some(Rc::new(RefCell::new(ck_mem)))
}

/* IDAAckpntDelete
 *
 * C deletes the first check point in the list and returns the new list head,
 * destroying exactly the N_Vectors that check point owns (`ck_phi_alloc`
 * entries of `ck_phi`, plus `ck_phiQ`/`ck_phiS`/`ck_phiQS` when the matching
 * flag is set). Under the handle model, removing the record from
 * `IDAADJ_mem->ck_mem` and dropping it releases the same vectors, so this
 * routine has no Rust counterpart — `ck_mem.remove(0)` / `ck_mem.clear()` at
 * the call sites is the port.
 */

/*
 * IDAAckpntAllocVectors
 *
 * Allocate checkpoint's phi, phiQ, phiS, phiQS vectors needed to save
 * current state of IDAMem.
 *
 * (Takes the not-yet-shared record by `&mut`; on failure the caller drops it,
 * which performs exactly the C cleanup loops.)
 */
fn IDAAckpntAllocVectors(IDA_mem: &IDAMem, ck_mem: &mut IDAckpntMemRec) -> sunbooleantype {
    let (tempv1, eeQ, Ns) = {
        let m = IDA_mem.borrow();
        (m.ida_tempv1.clone(), m.ida_eeQ.clone(), m.ida_Ns)
    };
    let tempv1 = tempv1.expect("ida_tempv1");

    for j in 0..ck_mem.ck_phi_alloc as usize {
        match N_VClone(&tempv1) {
            None => return SUNFALSE,
            Some(v) => ck_mem.ck_phi[j] = Some(v),
        }
    }

    /* Do we need to carry quadratures? */
    if ck_mem.ck_quadr {
        let eeQ = eeQ.clone().expect("ida_eeQ");
        for j in 0..ck_mem.ck_phi_alloc as usize {
            match N_VClone(&eeQ) {
                None => return SUNFALSE,
                Some(v) => ck_mem.ck_phiQ[j] = Some(v),
            }
        }
    }

    /* Do we need to carry sensitivities? */
    if ck_mem.ck_sensi {
        for j in 0..ck_mem.ck_phi_alloc as usize {
            match N_VCloneVectorArray(Ns, &tempv1) {
                None => return SUNFALSE,
                Some(vs) => ck_mem.ck_phiS[j] = vs,
            }
        }
    }

    /* Do we need to carry quadrature sensitivities? */
    if ck_mem.ck_quadr_sensi {
        let eeQ = eeQ.expect("ida_eeQ");
        for j in 0..ck_mem.ck_phi_alloc as usize {
            match N_VCloneVectorArray(Ns, &eeQ) {
                None => return SUNFALSE,
                Some(vs) => ck_mem.ck_phiQS[j] = vs,
            }
        }
    }

    SUNTRUE
}

/*
 * IDAAckpntCopyVectors
 *
 * Copy phi* vectors from IDAMem in the corresponding vectors from checkpoint
 *
 */
fn IDAAckpntCopyVectors(IDA_mem: &IDAMem, ck_mem: &mut IDAckpntMemRec) {
    let n_alloc = ck_mem.ck_phi_alloc as usize;
    let Ns = IDA_mem.borrow().ida_Ns;

    /* Save phi* arrays from IDA_mem */

    let cvals = vec![ONE; n_alloc];

    {
        let Xvecs: Vec<N_Vector> = {
            let m = IDA_mem.borrow();
            (0..n_alloc)
                .map(|j| m.ida_phi[j].clone().expect("ida_phi[j]"))
                .collect()
        };
        let Zvecs: Vec<N_Vector> = (0..n_alloc)
            .map(|j| ck_mem.ck_phi[j].clone().expect("ck_phi[j]"))
            .collect();

        let _ = N_VScaleVectorArray(ck_mem.ck_phi_alloc, &cvals, &Xvecs, &Zvecs);
    }

    if ck_mem.ck_quadr {
        let Xvecs: Vec<N_Vector> = {
            let m = IDA_mem.borrow();
            (0..n_alloc)
                .map(|j| m.ida_phiQ[j].clone().expect("ida_phiQ[j]"))
                .collect()
        };
        let Zvecs: Vec<N_Vector> = (0..n_alloc)
            .map(|j| ck_mem.ck_phiQ[j].clone().expect("ck_phiQ[j]"))
            .collect();

        let _ = N_VScaleVectorArray(ck_mem.ck_phi_alloc, &cvals, &Xvecs, &Zvecs);
    }

    /* C fills cvals[j*Ns+is] = ONE whenever (ck_sensi || ck_quadr_sensi);
    the local scratch below is the same all-ONE array. */

    if ck_mem.ck_sensi {
        let n = n_alloc * Ns as usize;
        let cvals = vec![ONE; n];
        let mut Xvecs: Vec<N_Vector> = Vec::with_capacity(n);
        let mut Zvecs: Vec<N_Vector> = Vec::with_capacity(n);

        {
            let m = IDA_mem.borrow();
            for j in 0..n_alloc {
                for is in 0..Ns as usize {
                    Xvecs.push(m.ida_phiS[j][is].clone());
                    Zvecs.push(ck_mem.ck_phiS[j][is].clone());
                }
            }
        }

        let _ = N_VScaleVectorArray(ck_mem.ck_phi_alloc * Ns, &cvals, &Xvecs, &Zvecs);
    }

    if ck_mem.ck_quadr_sensi {
        let n = n_alloc * Ns as usize;
        let cvals = vec![ONE; n];
        let mut Xvecs: Vec<N_Vector> = Vec::with_capacity(n);
        let mut Zvecs: Vec<N_Vector> = Vec::with_capacity(n);

        {
            let m = IDA_mem.borrow();
            for j in 0..n_alloc {
                for is in 0..Ns as usize {
                    Xvecs.push(m.ida_phiQS[j][is].clone());
                    Zvecs.push(ck_mem.ck_phiQS[j][is].clone());
                }
            }
        }

        let _ = N_VScaleVectorArray(ck_mem.ck_phi_alloc * Ns, &cvals, &Xvecs, &Zvecs);
    }
}

/*
 * IDAAdataMalloc
 *
 * This routine allocates memory for storing information at all
 * intermediate points between two consecutive check points.
 * This data is then used to interpolate the forward solution
 * at any other time.
 */

fn IDAAdataMalloc(IDA_mem: &IDAMem) -> sunbooleantype {
    let IDAADJ_mem = IDAADJ_mem_of(IDA_mem);

    IDAADJ_mem.borrow_mut().dt_mem = Vec::new();

    let nsteps = IDAADJ_mem.borrow().ia_nsteps;

    let mut dt_mem: Vec<IDAdtpntMem> = Vec::with_capacity((nsteps + 1) as usize);
    for _i in 0..=nsteps {
        dt_mem.push(Rc::new(RefCell::new(IDAdtpntMemRec {
            t: ZERO,
            content: None,
        })));
    }

    /* Attach the allocated dt_mem to IDAADJ_mem. */
    IDAADJ_mem.borrow_mut().dt_mem = dt_mem;
    SUNTRUE
}

/*
 * IDAAdataFree
 *
 * This routine frees the memory allocated for data storage.
 */

fn IDAAdataFree(IDA_mem: &IDAMem) {
    let IDAADJ_mem = IDAADJ_mem_of(IDA_mem);

    /* Destroy data points by calling the interpolation's 'free' routine. */
    let ia_free = IDAADJ_mem.borrow().ia_free.expect("ia_free");
    ia_free(IDA_mem);

    IDAADJ_mem.borrow_mut().dt_mem = Vec::new();
}

/*
 * IDAAdataStore
 *
 * This routine integrates the forward model starting at the check
 * point ck_mem and stores y and yprime at all intermediate
 * steps.
 *
 * Return values:
 *   - the flag that IDASolve may return on error
 *   - IDA_REIFWD_FAIL if no check point is available for this hot start
 *   - IDA_SUCCESS
 */

fn IDAAdataStore(IDA_mem: &IDAMem, ck_mem: &IDAckpntMem) -> i32 {
    let IDAADJ_mem = IDAADJ_mem_of(IDA_mem);

    /* Initialize IDA_mem with data from ck_mem. */
    let flag = IDAAckpntGet(IDA_mem, ck_mem);
    if flag != IDA_SUCCESS {
        return IDA_REIFWD_FAIL;
    }

    /* Set first structure in dt_mem[0] */
    let (dt0, storePnt) = {
        let ia = IDAADJ_mem.borrow();
        (ia.dt_mem[0].clone(), ia.ia_storePnt.expect("ia_storePnt"))
    };
    let ck_t0 = ck_mem.borrow().ck_t0;
    dt0.borrow_mut().t = ck_t0;
    let _ = storePnt(IDA_mem, &dt0);

    /* Decide whether TSTOP must be activated */
    let (tstopIDAFcall, tstopIDAF) = {
        let ia = IDAADJ_mem.borrow();
        (ia.ia_tstopIDAFcall, ia.ia_tstopIDAF)
    };
    if tstopIDAFcall {
        let _ = IDASetStopTime(IDA_mem, tstopIDAF);
    }

    let (ia_tinitial, ia_tfinal) = {
        let ia = IDAADJ_mem.borrow();
        (ia.ia_tinitial, ia.ia_tfinal)
    };
    let sign: sunrealtype = if ia_tfinal - ia_tinitial > ZERO {
        1.0
    } else {
        -1.0
    };

    let ck_t1 = ck_mem.borrow().ck_t1;
    let (yyTmp, ypTmp) = {
        let ia = IDAADJ_mem.borrow();
        (
            ia.ia_yyTmp.clone().expect("ia_yyTmp"),
            ia.ia_ypTmp.clone().expect("ia_ypTmp"),
        )
    };

    /* Run IDASolve in IDA_ONE_STEP mode to set following structures in dt_mem[i]. */
    let mut i: i64 = 1;
    let mut t: sunrealtype = ZERO;
    loop {
        let flag = IDASolve(IDA_mem, ck_t1, &mut t, &yyTmp, &ypTmp, IDA_ONE_STEP);
        if flag < 0 {
            return IDA_FWD_FAIL;
        }

        let (dti, storePnt) = {
            let ia = IDAADJ_mem.borrow();
            (
                ia.dt_mem[i as usize].clone(),
                ia.ia_storePnt.expect("ia_storePnt"),
            )
        };
        dti.borrow_mut().t = t;
        let _ = storePnt(IDA_mem, &dti);

        i += 1;

        if !(sign * (ck_t1 - t) > ZERO) {
            break;
        }
    }

    /* New data is now available. */
    {
        let mut ia = IDAADJ_mem.borrow_mut();
        ia.ia_ckpntData = Some(ck_mem.clone());
        ia.ia_newData = SUNTRUE;
        ia.ia_np = i;
    }

    IDA_SUCCESS
}

/*
 * CVAckpntGet
 *
 * This routine prepares IDAS for a hot restart from
 * the check point ck_mem
 */

fn IDAAckpntGet(IDA_mem: &IDAMem, ck_mem: &IDAckpntMem) -> i32 {
    let IDAADJ_mem = IDAADJ_mem_of(IDA_mem);

    /* C: ck_mem->ck_next == NULL, i.e. this is the t_initial check point
    (the tail of the list, which in the Vec model is the last element). */
    let next_is_null = {
        let ia = IDAADJ_mem.borrow();
        ia.ck_mem
            .last()
            .map_or(false, |tail| Rc::ptr_eq(tail, ck_mem))
    };

    if next_is_null {
        /* In this case, we just call the reinitialization routine,
         * but make sure we use the same initial stepsize as on
         * the first run. */

        let h0u = IDA_mem.borrow().ida_h0u;
        let _ = IDASetInitStep(IDA_mem, h0u);

        let (ck_t0, phi0, phi1) = {
            let ck = ck_mem.borrow();
            (
                ck.ck_t0,
                ck.ck_phi[0].clone().expect("ck_phi[0]"),
                ck.ck_phi[1].clone().expect("ck_phi[1]"),
            )
        };
        let flag = IDAReInit(IDA_mem, ck_t0, &phi0, &phi1);
        if flag != IDA_SUCCESS {
            return flag;
        }

        if ck_mem.borrow().ck_quadr {
            let phiQ0 = ck_mem.borrow().ck_phiQ[0].clone().expect("ck_phiQ[0]");
            let flag = IDAQuadReInit(IDA_mem, &phiQ0);
            if flag != IDA_SUCCESS {
                return flag;
            }
        }

        if ck_mem.borrow().ck_sensi {
            let ism = IDA_mem.borrow().ida_ism;
            let (phiS0, phiS1) = {
                let ck = ck_mem.borrow();
                (ck.ck_phiS[0].clone(), ck.ck_phiS[1].clone())
            };
            let flag = IDASensReInit(IDA_mem, ism, &phiS0, &phiS1);
            if flag != IDA_SUCCESS {
                return flag;
            }
        }

        if ck_mem.borrow().ck_quadr_sensi {
            let phiQS0 = ck_mem.borrow().ck_phiQS[0].clone();
            let flag = IDAQuadSensReInit(IDA_mem, &phiQS0);
            if flag != IDA_SUCCESS {
                return flag;
            }
        }
    } else {
        /* Copy parameters from check point data structure */
        {
            let ck = ck_mem.borrow();
            let mut m = IDA_mem.borrow_mut();
            m.ida_nst = ck.ck_nst;
            m.ida_tretlast = ck.ck_tretlast;
            m.ida_kk = ck.ck_kk;
            m.ida_kused = ck.ck_kused;
            m.ida_knew = ck.ck_knew;
            m.ida_phase = ck.ck_phase;
            m.ida_ns = ck.ck_ns;
            m.ida_hh = ck.ck_hh;
            m.ida_hused = ck.ck_hused;
            m.ida_eta = ck.ck_eta;
            m.ida_cj = ck.ck_cj;
            m.ida_cjlast = ck.ck_cjlast;
            m.ida_cjold = ck.ck_cjold;
            m.ida_cjratio = ck.ck_cjratio;
            m.ida_tn = ck.ck_t0;
            m.ida_ss = ck.ck_ss;
            m.ida_ssS = ck.ck_ssS;
        }

        let n_alloc = ck_mem.borrow().ck_phi_alloc as usize;
        let Ns = IDA_mem.borrow().ida_Ns;

        /* Copy the arrays from check point data structure */
        for j in 0..n_alloc {
            let ck_phi_j = ck_mem.borrow().ck_phi[j].clone().expect("ck_phi[j]");
            let phi_j = IDA_mem.borrow().ida_phi[j].clone().expect("ida_phi[j]");
            N_VScale(ONE, &ck_phi_j, &phi_j);
        }

        if ck_mem.borrow().ck_quadr {
            for j in 0..n_alloc {
                let ck_phiQ_j = ck_mem.borrow().ck_phiQ[j].clone().expect("ck_phiQ[j]");
                let phiQ_j = IDA_mem.borrow().ida_phiQ[j].clone().expect("ida_phiQ[j]");
                N_VScale(ONE, &ck_phiQ_j, &phiQ_j);
            }
        }

        if ck_mem.borrow().ck_sensi {
            for is in 0..Ns as usize {
                for j in 0..n_alloc {
                    let ck_phiS_ji = ck_mem.borrow().ck_phiS[j][is].clone();
                    let phiS_ji = IDA_mem.borrow().ida_phiS[j][is].clone();
                    N_VScale(ONE, &ck_phiS_ji, &phiS_ji);
                }
            }
        }

        if ck_mem.borrow().ck_quadr_sensi {
            for is in 0..Ns as usize {
                for j in 0..n_alloc {
                    let ck_phiQS_ji = ck_mem.borrow().ck_phiQS[j][is].clone();
                    let phiQS_ji = IDA_mem.borrow().ida_phiQS[j][is].clone();
                    N_VScale(ONE, &ck_phiQS_ji, &phiQS_ji);
                }
            }
        }

        {
            let ck = ck_mem.borrow();
            let mut m = IDA_mem.borrow_mut();
            for j in 0..MXORDP1 {
                m.ida_psi[j] = ck.ck_psi[j];
                m.ida_alpha[j] = ck.ck_alpha[j];
                m.ida_beta[j] = ck.ck_beta[j];
                m.ida_sigma[j] = ck.ck_sigma[j];
                m.ida_gamma[j] = ck.ck_gamma[j];
            }

            /* Force a call to setup */
            m.ida_forceSetup = SUNTRUE;
        }
    }

    IDA_SUCCESS
}

/*
 * -----------------------------------------------------------------
 * Functions specific to cubic Hermite interpolation
 * -----------------------------------------------------------------
 */

/*
 * IDAAhermiteMalloc
 *
 * This routine allocates memory for storing information at all
 * intermediate points between two consecutive check points.
 * This data is then used to interpolate the forward solution
 * at any other time.
 */

fn IDAAhermiteMalloc(IDA_mem: &IDAMem) -> sunbooleantype {
    let mut allocOK = SUNTRUE;

    let IDAADJ_mem = IDAADJ_mem_of(IDA_mem);

    let (tempv1, Ns) = {
        let m = IDA_mem.borrow();
        (m.ida_tempv1.clone().expect("ida_tempv1"), m.ida_Ns)
    };

    /* Allocate space for the vectors yyTmp and ypTmp. */
    match N_VClone(&tempv1) {
        None => return SUNFALSE,
        Some(v) => IDAADJ_mem.borrow_mut().ia_yyTmp = Some(v),
    }
    match N_VClone(&tempv1) {
        None => return SUNFALSE,
        Some(v) => IDAADJ_mem.borrow_mut().ia_ypTmp = Some(v),
    }

    let storeSensi = IDAADJ_mem.borrow().ia_storeSensi;

    /* Allocate space for sensitivities temporary vectors. */
    if storeSensi {
        match N_VCloneVectorArray(Ns, &tempv1) {
            None => {
                let mut ia = IDAADJ_mem.borrow_mut();
                ia.ia_yyTmp = None;
                ia.ia_ypTmp = None;
                return SUNFALSE;
            }
            Some(vs) => IDAADJ_mem.borrow_mut().ia_yySTmp = vs,
        }

        match N_VCloneVectorArray(Ns, &tempv1) {
            None => {
                let mut ia = IDAADJ_mem.borrow_mut();
                ia.ia_yyTmp = None;
                ia.ia_ypTmp = None;
                ia.ia_yySTmp = Vec::new();
                return SUNFALSE;
            }
            Some(vs) => IDAADJ_mem.borrow_mut().ia_ypSTmp = vs,
        }
    }

    /* Allocate space for the content field of the dt structures */

    let nsteps = IDAADJ_mem.borrow().ia_nsteps;
    let mut ii: i64 = 0;

    for i in 0..=nsteps {
        let y = match N_VClone(&tempv1) {
            None => {
                ii = i;
                allocOK = SUNFALSE;
                break;
            }
            Some(v) => v,
        };

        let yd = match N_VClone(&tempv1) {
            None => {
                ii = i;
                allocOK = SUNFALSE;
                break;
            }
            Some(v) => v,
        };

        let mut yS: Vec<N_Vector> = Vec::new();
        let mut ySd: Vec<N_Vector> = Vec::new();

        if storeSensi {
            yS = match N_VCloneVectorArray(Ns, &tempv1) {
                None => {
                    ii = i;
                    allocOK = SUNFALSE;
                    break;
                }
                Some(vs) => vs,
            };

            ySd = match N_VCloneVectorArray(Ns, &tempv1) {
                None => {
                    ii = i;
                    allocOK = SUNFALSE;
                    break;
                }
                Some(vs) => vs,
            };
        }

        let content = IDAhermiteDataMemRec {
            y: Some(y),
            yd: Some(yd),
            yS,
            ySd,
        };

        let d = dt_pnt(&IDAADJ_mem, i);
        d.borrow_mut().content = Some(Box::new(content));
    }

    /* If an error occurred, deallocate and return */

    if !allocOK {
        {
            let mut ia = IDAADJ_mem.borrow_mut();
            ia.ia_yyTmp = None;
            ia.ia_ypTmp = None;

            if storeSensi {
                ia.ia_yySTmp = Vec::new();
                ia.ia_ypSTmp = Vec::new();
            }
        }

        for i in 0..ii {
            let d = dt_pnt(&IDAADJ_mem, i);
            d.borrow_mut().content = None;
        }
    }

    allocOK
}

/*
 * IDAAhermiteFree
 *
 * This routine frees the memory allocated for data storage.
 */

fn IDAAhermiteFree(IDA_mem: &IDAMem) {
    let IDAADJ_mem = IDAADJ_mem_of(IDA_mem);

    let storeSensi = {
        let mut ia = IDAADJ_mem.borrow_mut();
        ia.ia_yyTmp = None;
        ia.ia_ypTmp = None;
        ia.ia_storeSensi
    };

    if storeSensi {
        let mut ia = IDAADJ_mem.borrow_mut();
        ia.ia_yySTmp = Vec::new();
        ia.ia_ypSTmp = Vec::new();
    }

    let nsteps = IDAADJ_mem.borrow().ia_nsteps;

    for i in 0..=nsteps {
        /* content might be None, if IDAAdjInit was called but IDASolveF was
        not — dropping it is the no-op the C `if (content)` guard produces. */
        let d = dt_pnt(&IDAADJ_mem, i);
        d.borrow_mut().content = None;
    }
}

/*
 * IDAAhermiteStorePnt
 *
 * This routine stores a new point (y,yd) in the structure d for use
 * in the cubic Hermite interpolation.
 * Note that the time is already stored.
 */

fn IDAAhermiteStorePnt(IDA_mem: &IDAMem, d: &IDAdtpntMem) -> i32 {
    let IDAADJ_mem = IDAADJ_mem_of(IDA_mem);

    let storeSensi = IDAADJ_mem.borrow().ia_storeSensi;

    let (y, yd, yS, ySd) = {
        let db = d.borrow();
        let content = db
            .content
            .as_ref()
            .expect("dt_mem content")
            .downcast_ref::<IDAhermiteDataMemRec>()
            .expect("Hermite content");
        (
            content.y.clone().expect("content->y"),
            content.yd.clone().expect("content->yd"),
            content.yS.clone(),
            content.ySd.clone(),
        )
    };

    /* Load solution(s) */
    let phi0 = IDA_mem.borrow().ida_phi[0].clone().expect("ida_phi[0]");
    N_VScale(ONE, &phi0, &y);

    if storeSensi {
        let Ns = IDA_mem.borrow().ida_Ns;
        let cvals = vec![ONE; Ns as usize];

        let phiS0 = IDA_mem.borrow().ida_phiS[0].clone();
        let retval = N_VScaleVectorArray(Ns, &cvals, &phiS0, &yS);
        if retval != IDA_SUCCESS {
            return IDA_VECTOROP_ERR;
        }
    }

    /* Load derivative(s). */
    let _ = IDAAGettnSolutionYp(IDA_mem, &yd);

    if storeSensi {
        let _ = IDAAGettnSolutionYpS(IDA_mem, &ySd);
    }

    0
}

/*
 * IDAAhermiteGetY
 *
 * This routine uses cubic piece-wise Hermite interpolation for
 * the forward solution vector.
 * It is typically called by the wrapper routines before calling
 * user provided routines (fB, djacB, bjacB, jtimesB, psolB) but
 * can be directly called by the user through IDAGetAdjY
 */

fn IDAAhermiteGetY(
    IDA_mem: &IDAMem,
    t: sunrealtype,
    yy: &N_Vector,
    yp: &N_Vector,
    yyS: &[N_Vector],
    ypS: &[N_Vector],
) -> i32 {
    let IDAADJ_mem = IDAADJ_mem_of(IDA_mem);

    /* Local value of Ns */
    let NS: i32 = if IDAADJ_mem.borrow().ia_interpSensi && !yyS.is_empty() {
        IDA_mem.borrow().ida_Ns
    } else {
        0
    };

    /* Get the index in dt_mem */
    let mut index: i64 = 0;
    let mut newpoint: sunbooleantype = SUNFALSE;
    let flag = IDAAfindIndex(IDA_mem, t, &mut index, &mut newpoint);
    if flag != IDA_SUCCESS {
        return flag;
    }

    /* If we are beyond the left limit but close enough,
    then return y at the left limit. */

    if index == 0 {
        let (c0y, c0yd, c0yS, c0ySd) = herm_content(&IDAADJ_mem, 0);
        N_VScale(ONE, &c0y, yy);
        N_VScale(ONE, &c0yd, yp);

        if NS > 0 {
            let cvals = vec![ONE; NS as usize];

            let retval = N_VScaleVectorArray(NS, &cvals, &c0yS, yyS);
            if retval != IDA_SUCCESS {
                return IDA_VECTOROP_ERR;
            }

            let retval = N_VScaleVectorArray(NS, &cvals, &c0ySd, ypS);
            if retval != IDA_SUCCESS {
                return IDA_VECTOROP_ERR;
            }
        }

        return IDA_SUCCESS;
    }

    /* Extract stuff from the appropriate data points */
    let t0 = dt_t(&IDAADJ_mem, index - 1);
    let t1 = dt_t(&IDAADJ_mem, index);
    let delta = t1 - t0;

    let (y0, yd0, yS0, ySd0) = herm_content(&IDAADJ_mem, index - 1);

    if newpoint {
        /* Recompute Y0 and Y1 */
        let (y1, yd1, yS1, ySd1) = herm_content(&IDAADJ_mem, index);

        /* Y1 = delta (yd1 + yd0) - 2 (y1 - y0) */
        let cvals: [sunrealtype; 4] = [-TWO, TWO, delta, delta];
        let Xvecs: [N_Vector; 4] = [y1.clone(), y0.clone(), yd1.clone(), yd0.clone()];

        let iaY1 = IDAADJ_mem.borrow().ia_Y[1].clone().expect("ia_Y[1]");
        let retval = N_VLinearCombination(4, &cvals, &Xvecs, &iaY1);
        if retval != IDA_SUCCESS {
            return IDA_VECTOROP_ERR;
        }

        /* Y0 = y1 - y0 - delta * yd0 */
        let cvals: [sunrealtype; 3] = [ONE, -ONE, -delta];
        let Xvecs: [N_Vector; 3] = [y1.clone(), y0.clone(), yd0.clone()];

        let iaY0 = IDAADJ_mem.borrow().ia_Y[0].clone().expect("ia_Y[0]");
        let retval = N_VLinearCombination(3, &cvals, &Xvecs, &iaY0);
        if retval != IDA_SUCCESS {
            return IDA_VECTOROP_ERR;
        }

        /* Recompute YS0 and YS1, if needed */

        if NS > 0 {
            /* YS1 = delta (ySd1 + ySd0) - 2 (yS1 - yS0) */
            let cvals: [sunrealtype; 4] = [-TWO, TWO, delta, delta];
            let XXvecs: [Vec<N_Vector>; 4] = [yS1.clone(), yS0.clone(), ySd1.clone(), ySd0.clone()];

            let iaYS1 = IDAADJ_mem.borrow().ia_YS[1].clone();
            let retval = N_VLinearCombinationVectorArray(NS, 4, &cvals, &XXvecs, &iaYS1);
            if retval != IDA_SUCCESS {
                return IDA_VECTOROP_ERR;
            }

            /* YS0 = yS1 - yS0 - delta * ySd0 */
            let cvals: [sunrealtype; 3] = [ONE, -ONE, -delta];
            let XXvecs: [Vec<N_Vector>; 3] = [yS1.clone(), yS0.clone(), ySd0.clone()];

            let iaYS0 = IDAADJ_mem.borrow().ia_YS[0].clone();
            let retval = N_VLinearCombinationVectorArray(NS, 3, &cvals, &XXvecs, &iaYS0);
            if retval != IDA_SUCCESS {
                return IDA_VECTOROP_ERR;
            }
        }
    }

    /* Perform the actual interpolation. */

    /* For y. */
    let mut factor1 = t - t0;

    let mut factor2 = factor1 / delta;
    factor2 = factor2 * factor2;

    let factor3 = factor2 * (t - t1) / delta;

    let cvals: [sunrealtype; 4] = [ONE, factor1, factor2, factor3];

    /* y = y0 + factor1 yd0 + factor2 * Y[0] + factor3 Y[1] */
    let (iaY0, iaY1) = {
        let ia = IDAADJ_mem.borrow();
        (
            ia.ia_Y[0].clone().expect("ia_Y[0]"),
            ia.ia_Y[1].clone().expect("ia_Y[1]"),
        )
    };
    let Xvecs: [N_Vector; 4] = [y0.clone(), yd0.clone(), iaY0.clone(), iaY1.clone()];

    let retval = N_VLinearCombination(4, &cvals, &Xvecs, yy);
    if retval != IDA_SUCCESS {
        return IDA_VECTOROP_ERR;
    }

    /* Sensi Interpolation. */

    /* yS = yS0 + factor1 ySd0 + factor2 * YS[0] + factor3 YS[1], if needed */
    if NS > 0 {
        let (iaYS0, iaYS1) = {
            let ia = IDAADJ_mem.borrow();
            (ia.ia_YS[0].clone(), ia.ia_YS[1].clone())
        };
        let XXvecs: [Vec<N_Vector>; 4] = [yS0.clone(), ySd0.clone(), iaYS0, iaYS1];

        let retval = N_VLinearCombinationVectorArray(NS, 4, &cvals, &XXvecs, yyS);
        if retval != IDA_SUCCESS {
            return IDA_VECTOROP_ERR;
        }
    }

    /* For y'. */
    factor1 = factor1 / delta / delta; /* factor1 = 2(t-t0)/(t1-t0)^2             */
    factor2 = factor1 * ((3.0 * t - 2.0 * t1 - t0) / delta); /* factor2 = (t-t0)(3*t-2*t1-t0)/(t1-t0)^3 */
    factor1 *= 2.0;

    let cvals: [sunrealtype; 3] = [ONE, factor1, factor2];

    /* yp = yd0 + factor1 Y[0] + factor 2 Y[1] */
    let Xvecs: [N_Vector; 3] = [yd0.clone(), iaY0.clone(), iaY1.clone()];

    let retval = N_VLinearCombination(3, &cvals, &Xvecs, yp);
    if retval != IDA_SUCCESS {
        return IDA_VECTOROP_ERR;
    }

    /* Sensi interpolation for 1st derivative. */

    /* ypS = ySd0 + factor1 YS[0] + factor 2 YS[1], if needed */
    if NS > 0 {
        let (iaYS0, iaYS1) = {
            let ia = IDAADJ_mem.borrow();
            (ia.ia_YS[0].clone(), ia.ia_YS[1].clone())
        };
        let XXvecs: [Vec<N_Vector>; 3] = [ySd0.clone(), iaYS0, iaYS1];

        let retval = N_VLinearCombinationVectorArray(NS, 3, &cvals, &XXvecs, ypS);
        if retval != IDA_SUCCESS {
            return IDA_VECTOROP_ERR;
        }
    }

    IDA_SUCCESS
}

/*
 * -----------------------------------------------------------------
 * Functions specific to Polynomial interpolation
 * -----------------------------------------------------------------
 */

/*
 * IDAApolynomialMalloc
 *
 * This routine allocates memory for storing information at all
 * intermediate points between two consecutive check points.
 * This data is then used to interpolate the forward solution
 * at any other time.
 *
 * Information about the first derivative is stored only for the first
 * data point.
 */

fn IDAApolynomialMalloc(IDA_mem: &IDAMem) -> sunbooleantype {
    let mut allocOK = SUNTRUE;

    let IDAADJ_mem = IDAADJ_mem_of(IDA_mem);

    let (tempv1, Ns) = {
        let m = IDA_mem.borrow();
        (m.ida_tempv1.clone().expect("ida_tempv1"), m.ida_Ns)
    };

    /* Allocate space for the vectors yyTmp and ypTmp */
    match N_VClone(&tempv1) {
        None => return SUNFALSE,
        Some(v) => IDAADJ_mem.borrow_mut().ia_yyTmp = Some(v),
    }
    match N_VClone(&tempv1) {
        None => return SUNFALSE,
        Some(v) => IDAADJ_mem.borrow_mut().ia_ypTmp = Some(v),
    }

    let storeSensi = IDAADJ_mem.borrow().ia_storeSensi;

    if storeSensi {
        match N_VCloneVectorArray(Ns, &tempv1) {
            None => {
                let mut ia = IDAADJ_mem.borrow_mut();
                ia.ia_yyTmp = None;
                ia.ia_ypTmp = None;
                return SUNFALSE;
            }
            Some(vs) => IDAADJ_mem.borrow_mut().ia_yySTmp = vs,
        }

        match N_VCloneVectorArray(Ns, &tempv1) {
            None => {
                let mut ia = IDAADJ_mem.borrow_mut();
                ia.ia_yyTmp = None;
                ia.ia_ypTmp = None;
                ia.ia_yySTmp = Vec::new();
                return SUNFALSE;
            }
            Some(vs) => IDAADJ_mem.borrow_mut().ia_ypSTmp = vs,
        }
    }

    /* Allocate space for the content field of the dt structures */

    let nsteps = IDAADJ_mem.borrow().ia_nsteps;
    let mut ii: i64 = 0;

    for i in 0..=nsteps {
        let y = match N_VClone(&tempv1) {
            None => {
                ii = i;
                allocOK = SUNFALSE;
                break;
            }
            Some(v) => v,
        };

        /* Allocate space for yp also. Needed for the most left point interpolation. */
        let yd: Option<N_Vector> = if i == 0 {
            match N_VClone(&tempv1) {
                None => {
                    ii = i;
                    allocOK = SUNFALSE;
                    break;
                }
                Some(v) => Some(v),
            }
        } else {
            /* Not the first data point. */
            None
        };

        let mut yS: Vec<N_Vector> = Vec::new();
        let mut ySd: Vec<N_Vector> = Vec::new();

        if storeSensi {
            yS = match N_VCloneVectorArray(Ns, &tempv1) {
                None => {
                    ii = i;
                    allocOK = SUNFALSE;
                    break;
                }
                Some(vs) => vs,
            };

            if i == 0 {
                ySd = match N_VCloneVectorArray(Ns, &tempv1) {
                    None => {
                        ii = i;
                        allocOK = SUNFALSE;
                        break;
                    }
                    Some(vs) => vs,
                };
            }
        }

        let content = IDApolynomialDataMemRec {
            y: Some(y),
            yS,
            yd,
            ySd,
            order: 0,
        };

        let d = dt_pnt(&IDAADJ_mem, i);
        d.borrow_mut().content = Some(Box::new(content));
    }

    /* If an error occurred, deallocate and return */
    if !allocOK {
        {
            let mut ia = IDAADJ_mem.borrow_mut();
            ia.ia_yyTmp = None;
            ia.ia_ypTmp = None;
            if storeSensi {
                ia.ia_yySTmp = Vec::new();
                ia.ia_ypSTmp = Vec::new();
            }
        }

        for i in 0..ii {
            let d = dt_pnt(&IDAADJ_mem, i);
            d.borrow_mut().content = None;
        }
    }

    allocOK
}

/*
 * IDAApolynomialFree
 *
 * This routine frees the memory allocated for data storage.
 */

fn IDAApolynomialFree(IDA_mem: &IDAMem) {
    let IDAADJ_mem = IDAADJ_mem_of(IDA_mem);

    let storeSensi = {
        let mut ia = IDAADJ_mem.borrow_mut();
        ia.ia_yyTmp = None;
        ia.ia_ypTmp = None;
        ia.ia_storeSensi
    };

    if storeSensi {
        let mut ia = IDAADJ_mem.borrow_mut();
        ia.ia_yySTmp = Vec::new();
        ia.ia_ypSTmp = Vec::new();
    }

    let nsteps = IDAADJ_mem.borrow().ia_nsteps;

    for i in 0..=nsteps {
        /* content might be None, if IDAAdjInit was called but IDASolveF was not. */
        let d = dt_pnt(&IDAADJ_mem, i);
        d.borrow_mut().content = None;
    }
}

/*
 * IDAApolynomialStorePnt
 *
 * This routine stores a new point y in the structure d for use
 * in the Polynomial interpolation.
 *
 * Note that the time is already stored. Information about the
 * first derivative is available only for the first data point,
 * in which case content->yp is non-null.
 */

fn IDAApolynomialStorePnt(IDA_mem: &IDAMem, d: &IDAdtpntMem) -> i32 {
    let IDAADJ_mem = IDAADJ_mem_of(IDA_mem);

    let storeSensi = IDAADJ_mem.borrow().ia_storeSensi;

    let (y, yd, yS, ySd) = {
        let db = d.borrow();
        let content = db
            .content
            .as_ref()
            .expect("dt_mem content")
            .downcast_ref::<IDApolynomialDataMemRec>()
            .expect("polynomial content");
        (
            content.y.clone().expect("content->y"),
            content.yd.clone(),
            content.yS.clone(),
            content.ySd.clone(),
        )
    };

    let phi0 = IDA_mem.borrow().ida_phi[0].clone().expect("ida_phi[0]");
    N_VScale(ONE, &phi0, &y);

    /* copy also the derivative for the first data point (in this case
    content->yp is non-null). */
    if let Some(yd) = yd.as_ref() {
        let _ = IDAAGettnSolutionYp(IDA_mem, yd);
    }

    if storeSensi {
        let Ns = IDA_mem.borrow().ida_Ns;
        let cvals = vec![ONE; Ns as usize];

        let phiS0 = IDA_mem.borrow().ida_phiS[0].clone();
        let retval = N_VScaleVectorArray(Ns, &cvals, &phiS0, &yS);
        if retval != IDA_SUCCESS {
            return IDA_VECTOROP_ERR;
        }

        /* store the derivative if it is the first data point. */
        if !ySd.is_empty() {
            let _ = IDAAGettnSolutionYpS(IDA_mem, &ySd);
        }
    }

    let kused = IDA_mem.borrow().ida_kused;
    {
        let mut db = d.borrow_mut();
        let content = db
            .content
            .as_mut()
            .expect("dt_mem content")
            .downcast_mut::<IDApolynomialDataMemRec>()
            .expect("polynomial content");
        content.order = kused;
    }

    0
}

/*
 * IDAApolynomialGetY
 *
 * This routine uses polynomial interpolation for the forward solution vector.
 * It is typically called by the wrapper routines before calling
 * user provided routines (fB, djacB, bjacB, jtimesB, psolB)) but
 * can be directly called by the user through CVodeGetAdjY.
 */

fn IDAApolynomialGetY(
    IDA_mem: &IDAMem,
    t: sunrealtype,
    yy: &N_Vector,
    yp: &N_Vector,
    yyS: &[N_Vector],
    ypS: &[N_Vector],
) -> i32 {
    let IDAADJ_mem = IDAADJ_mem_of(IDA_mem);

    /* Local value of Ns */
    let NS: i32 = if IDAADJ_mem.borrow().ia_interpSensi && !yyS.is_empty() {
        IDA_mem.borrow().ida_Ns
    } else {
        0
    };

    /* Get the index in dt_mem */
    let mut index: i64 = 0;
    let mut newpoint: sunbooleantype = SUNFALSE;
    let flag = IDAAfindIndex(IDA_mem, t, &mut index, &mut newpoint);
    if flag != IDA_SUCCESS {
        return flag;
    }

    /* If we are beyond the left limit but close enough,
    then return y at the left limit. */

    if index == 0 {
        let (cy, cyd, cyS, cySd, _order) = poly_content(&IDAADJ_mem, 0);
        N_VScale(ONE, &cy, yy);
        N_VScale(ONE, &cyd.expect("content->yd"), yp);

        if NS > 0 {
            let cvals = vec![ONE; NS as usize];

            let retval = N_VScaleVectorArray(NS, &cvals, &cyS, yyS);
            if retval != IDA_SUCCESS {
                return IDA_VECTOROP_ERR;
            }

            let retval = N_VScaleVectorArray(NS, &cvals, &cySd, ypS);
            if retval != IDA_SUCCESS {
                return IDA_VECTOROP_ERR;
            }
        }

        return IDA_SUCCESS;
    }

    /* Scaling factor */
    let delt = SUNRabs(dt_t(&IDAADJ_mem, index) - dt_t(&IDAADJ_mem, index - 1));

    /* Find the direction of the forward integration */
    let (ia_tinitial, ia_tfinal) = {
        let ia = IDAADJ_mem.borrow();
        (ia.ia_tinitial, ia.ia_tfinal)
    };
    let dir: i32 = if ia_tfinal - ia_tinitial > ZERO {
        1
    } else {
        -1
    };

    /* Establish the base point depending on the integration direction.
    Modify the base if there are not enough points for the current order */

    let base: i64;
    let order: i32;

    if dir == 1 {
        let mut b = index;
        let (_cy, _cyd, _cyS, _cySd, o) = poly_content(&IDAADJ_mem, b);
        order = o;
        if index < order as i64 {
            b += order as i64 - index;
        }
        base = b;
    } else {
        let mut b = index - 1;
        let (_cy, _cyd, _cyS, _cySd, o) = poly_content(&IDAADJ_mem, b);
        order = o;
        let ia_np = IDAADJ_mem.borrow().ia_np;
        if ia_np - index > order as i64 {
            b -= index + order as i64 - ia_np;
        }
        base = b;
    }

    /* Recompute Y (divided differences for Newton polynomial) if needed */

    if newpoint {
        /* Store 0-th order DD */
        if dir == 1 {
            for j in 0..=order as i64 {
                let tj = dt_t(&IDAADJ_mem, base - j);
                IDAADJ_mem.borrow_mut().ia_T[j as usize] = tj;

                let (cy, _cyd, cyS, _cySd, _o) = poly_content(&IDAADJ_mem, base - j);

                let iaYj = IDAADJ_mem.borrow().ia_Y[j as usize]
                    .clone()
                    .expect("ia_Y[j]");
                N_VScale(ONE, &cy, &iaYj);

                if NS > 0 {
                    let cvals = vec![ONE; NS as usize];
                    let iaYSj = IDAADJ_mem.borrow().ia_YS[j as usize].clone();
                    let retval = N_VScaleVectorArray(NS, &cvals, &cyS, &iaYSj);
                    if retval != IDA_SUCCESS {
                        return IDA_VECTOROP_ERR;
                    }
                }
            }
        } else {
            for j in 0..=order as i64 {
                let tj = dt_t(&IDAADJ_mem, base - 1 + j);
                IDAADJ_mem.borrow_mut().ia_T[j as usize] = tj;

                let (cy, _cyd, cyS, _cySd, _o) = poly_content(&IDAADJ_mem, base - 1 + j);

                let iaYj = IDAADJ_mem.borrow().ia_Y[j as usize]
                    .clone()
                    .expect("ia_Y[j]");
                N_VScale(ONE, &cy, &iaYj);

                if NS > 0 {
                    let cvals = vec![ONE; NS as usize];
                    let iaYSj = IDAADJ_mem.borrow().ia_YS[j as usize].clone();
                    let retval = N_VScaleVectorArray(NS, &cvals, &cyS, &iaYSj);
                    if retval != IDA_SUCCESS {
                        return IDA_VECTOROP_ERR;
                    }
                }
            }
        }

        /* Compute higher-order DD */
        for i in 1..=order {
            let mut j = order;
            while j >= i {
                let (Tj, Tji) = {
                    let ia = IDAADJ_mem.borrow();
                    (ia.ia_T[j as usize], ia.ia_T[(j - i) as usize])
                };
                let factor = delt / (Tj - Tji);

                let (iaYj, iaYjm1) = {
                    let ia = IDAADJ_mem.borrow();
                    (
                        ia.ia_Y[j as usize].clone().expect("ia_Y[j]"),
                        ia.ia_Y[(j - 1) as usize].clone().expect("ia_Y[j-1]"),
                    )
                };
                N_VLinearSum(factor, &iaYj, -factor, &iaYjm1, &iaYj);

                for is in 0..NS as usize {
                    let (iaYSj_is, iaYSjm1_is) = {
                        let ia = IDAADJ_mem.borrow();
                        (
                            ia.ia_YS[j as usize][is].clone(),
                            ia.ia_YS[(j - 1) as usize][is].clone(),
                        )
                    };
                    N_VLinearSum(factor, &iaYSj_is, -factor, &iaYSjm1_is, &iaYSj_is);
                }

                j -= 1;
            }
        }
    }

    /* Perform the actual interpolation for yy using nested multiplications */

    let mut cvals = vec![ZERO; (order + 1) as usize];
    cvals[0] = ONE;
    for i in 0..order as usize {
        let Ti = IDAADJ_mem.borrow().ia_T[i];
        cvals[i + 1] = cvals[i] * (t - Ti) / delt;
    }

    {
        let iaY: Vec<N_Vector> = {
            let ia = IDAADJ_mem.borrow();
            (0..=order as usize)
                .map(|j| ia.ia_Y[j].clone().expect("ia_Y[j]"))
                .collect()
        };
        let retval = N_VLinearCombination(order + 1, &cvals, &iaY, yy);
        if retval != IDA_SUCCESS {
            return IDA_VECTOROP_ERR;
        }
    }

    if NS > 0 {
        let iaYS: Vec<Vec<N_Vector>> = {
            let ia = IDAADJ_mem.borrow();
            (0..=order as usize).map(|j| ia.ia_YS[j].clone()).collect()
        };
        let retval = N_VLinearCombinationVectorArray(NS, order + 1, &cvals, &iaYS, yyS);
        if retval != IDA_SUCCESS {
            return IDA_VECTOROP_ERR;
        }
    }

    /* Perform the actual interpolation for yp.

       Writing p(t) = y0 + (t-t0)*f[t0,t1] + ... + (t-t0)(t-t1)...(t-tn)*f[t0,t1,...tn],
       denote psi_k(t) = (t-t0)(t-t1)...(t-tk).

       The formula used for p'(t) is:
         - p'(t) = f[t0,t1] + psi_1'(t)*f[t0,t1,t2] + ... + psi_n'(t)*f[t0,t1,...,tn]

       We recursively compute psi_k'(t) from:
         - psi_k'(t) = (t-tk)*psi_{k-1}'(t) + psi_{k-1}

       psi_k is rescaled with 1/delt each time is computed, because the Newton DDs from Y were
       scaled with delt.
    */

    let mut Psi = ONE;
    let mut Psiprime = ZERO;

    for i in 1..=order as usize {
        let Tim1 = IDAADJ_mem.borrow().ia_T[i - 1];
        let factor = (t - Tim1) / delt;

        Psiprime = Psi / delt + factor * Psiprime;
        Psi = Psi * factor;

        cvals[i - 1] = Psiprime;
    }

    {
        /* C: `IDAADJ_mem->ia_Y + 1` — the sub-array starting at index 1 */
        let iaY1: Vec<N_Vector> = {
            let ia = IDAADJ_mem.borrow();
            (1..=order as usize)
                .map(|j| ia.ia_Y[j].clone().expect("ia_Y[j]"))
                .collect()
        };
        let retval = N_VLinearCombination(order, &cvals, &iaY1, yp);
        if retval != IDA_SUCCESS {
            return IDA_VECTOROP_ERR;
        }
    }

    if NS > 0 {
        /* C: `IDAADJ_mem->ia_YS + 1` */
        let iaYS1: Vec<Vec<N_Vector>> = {
            let ia = IDAADJ_mem.borrow();
            (1..=order as usize).map(|j| ia.ia_YS[j].clone()).collect()
        };
        let retval = N_VLinearCombinationVectorArray(NS, order, &cvals, &iaYS1, ypS);
        if retval != IDA_SUCCESS {
            return IDA_VECTOROP_ERR;
        }
    }

    IDA_SUCCESS
}

/*
 * IDAAGettnSolutionYp
 *
 * Evaluates the first derivative of the solution at the last time returned by
 * IDASolve (tretlast).
 *
 * The function implements the same algorithm as in IDAGetSolution but in the
 * particular case when  t=tn (i.e. delta=0).
 *
 * This function was implemented to avoid calls to IDAGetSolution which computes
 * y by doing a loop that is not necessary for this particular situation.
 */

fn IDAAGettnSolutionYp(IDA_mem: &IDAMem, yp: &N_Vector) -> i32 {
    if IDA_mem.borrow().ida_nst == 0 {
        /* If no integration was done, return the yp supplied by user.*/
        let phi1 = IDA_mem.borrow().ida_phi[1].clone().expect("ida_phi[1]");
        N_VScale(ONE, &phi1, yp);

        return 0;
    }

    /* Compute yp as in IDAGetSolution for this particular case when t=tn. */

    let (kused, psi) = {
        let m = IDA_mem.borrow();
        (m.ida_kused, m.ida_psi)
    };

    let mut kord = kused;
    if kused == 0 {
        kord = 1;
    }

    let mut C = ONE;
    let mut D = ZERO;
    let mut gam = ZERO;
    let mut dvals = [ZERO; MAXORD_DEFAULT];
    for j in 1..=kord as usize {
        D = D * gam + C / psi[j - 1];
        C = C * gam;
        gam = psi[j - 1] / psi[j];

        dvals[j - 1] = D;
    }

    /* C: `IDA_mem->ida_phi + 1` */
    let phi1: Vec<N_Vector> = {
        let m = IDA_mem.borrow();
        (1..=kord as usize)
            .map(|j| m.ida_phi[j].clone().expect("ida_phi[j]"))
            .collect()
    };

    let retval = N_VLinearCombination(kord, &dvals, &phi1, yp);
    if retval != IDA_SUCCESS {
        return IDA_VECTOROP_ERR;
    }

    0
}

/*
 * IDAAGettnSolutionYpS
 *
 * Same as IDAAGettnSolutionYp, but for first derivative of the sensitivities.
 *
 */

fn IDAAGettnSolutionYpS(IDA_mem: &IDAMem, ypS: &[N_Vector]) -> i32 {
    let Ns = IDA_mem.borrow().ida_Ns;

    if IDA_mem.borrow().ida_nst == 0 {
        /* If no integration was done, return the ypS supplied by user.*/
        let cvals = vec![ONE; Ns as usize];

        let phiS1 = IDA_mem.borrow().ida_phiS[1].clone();
        let retval = N_VScaleVectorArray(Ns, &cvals, &phiS1, ypS);
        if retval != IDA_SUCCESS {
            return IDA_VECTOROP_ERR;
        }

        return 0;
    }

    let (kused, psi) = {
        let m = IDA_mem.borrow();
        (m.ida_kused, m.ida_psi)
    };

    let mut kord = kused;
    if kused == 0 {
        kord = 1;
    }

    let mut C = ONE;
    let mut D = ZERO;
    let mut gam = ZERO;
    let mut dvals = [ZERO; MAXORD_DEFAULT];
    for j in 1..=kord as usize {
        D = D * gam + C / psi[j - 1];
        C = C * gam;
        gam = psi[j - 1] / psi[j];

        dvals[j - 1] = D;
    }

    /* C: `IDA_mem->ida_phiS + 1` */
    let phiS1: Vec<Vec<N_Vector>> = {
        let m = IDA_mem.borrow();
        (1..=kord as usize).map(|j| m.ida_phiS[j].clone()).collect()
    };

    let retval = N_VLinearCombinationVectorArray(Ns, kord, &dvals, &phiS1, ypS);
    if retval != IDA_SUCCESS {
        return IDA_VECTOROP_ERR;
    }

    0
}

/*
 * IDAAfindIndex
 *
 * Finds the index in the array of data point structures such that
 *     dt_mem[index-1].t <= t < dt_mem[index].t
 * If index is changed from the previous invocation, then newpoint = SUNTRUE
 *
 * If t is beyond the leftmost limit, but close enough, index=0.
 *
 * Returns IDA_SUCCESS if successful and IDA_GETY_BADT if unable to
 * find index (t is too far beyond limits).
 */

fn IDAAfindIndex(
    ida_mem: &IDAMem,
    t: sunrealtype,
    index: &mut i64,
    newpoint: &mut sunbooleantype,
) -> i32 {
    let IDA_mem = ida_mem;
    let IDAADJ_mem = IDAADJ_mem_of(IDA_mem);

    *newpoint = SUNFALSE;

    /* Find the direction of integration */
    let (ia_tinitial, ia_tfinal) = {
        let ia = IDAADJ_mem.borrow();
        (ia.ia_tinitial, ia.ia_tfinal)
    };
    let sign: sunrealtype = if ia_tfinal - ia_tinitial > ZERO {
        1.0
    } else {
        -1.0
    };

    /* If this is the first time we use new data */
    if IDAADJ_mem.borrow().ia_newData {
        let np = IDAADJ_mem.borrow().ia_np;
        let mut ia = IDAADJ_mem.borrow_mut();
        ia.ia_ilast = np - 1;
        *newpoint = SUNTRUE;
        ia.ia_newData = SUNFALSE;
    }

    /* Search for index starting from ilast */
    let ilast = IDAADJ_mem.borrow().ia_ilast;
    let to_left = sign * (t - dt_t(&IDAADJ_mem, ilast - 1)) < ZERO;
    let to_right = sign * (t - dt_t(&IDAADJ_mem, ilast)) > ZERO;

    if to_left {
        /* look for a new index to the left */

        *newpoint = SUNTRUE;

        *index = ilast;
        loop {
            if *index == 0 {
                break;
            }
            if sign * (t - dt_t(&IDAADJ_mem, *index - 1)) <= ZERO {
                *index -= 1;
            } else {
                break;
            }
        }

        if *index == 0 {
            IDAADJ_mem.borrow_mut().ia_ilast = 1;
        } else {
            IDAADJ_mem.borrow_mut().ia_ilast = *index;
        }

        if *index == 0 {
            /* t is beyond leftmost limit. Is it too far? */
            let uround = IDA_mem.borrow().ida_uround;
            if SUNRabs(t - dt_t(&IDAADJ_mem, 0)) > FUZZ_FACTOR * uround {
                return IDA_GETY_BADT;
            }
        }
    } else if to_right {
        /* look for a new index to the right */

        *newpoint = SUNTRUE;

        *index = ilast;
        loop {
            if sign * (t - dt_t(&IDAADJ_mem, *index)) > ZERO {
                *index += 1;
            } else {
                break;
            }
        }

        IDAADJ_mem.borrow_mut().ia_ilast = *index;
    } else {
        /* ilast is still OK */

        *index = ilast;
    }

    IDA_SUCCESS
}

/*
 * IDAGetAdjY
 *
 * This routine returns the interpolated forward solution at time t.
 * The user must allocate space for y.
 */

pub fn IDAGetAdjY(ida_mem: &IDAMem, t: sunrealtype, yy: &N_Vector, yp: &N_Vector) -> i32 {
    /* NULL-mem check: handled by the type system */
    let IDA_mem = ida_mem;
    let IDAADJ_mem = IDAADJ_mem_of(IDA_mem);

    let getY = IDAADJ_mem.borrow().ia_getY.expect("ia_getY");

    /* C passes NULL for yyS/ypS; the empty slice is the contract's NULL mapping */
    getY(IDA_mem, t, yy, yp, &[], &[])
}

/*=================================================================*/
/*             Wrappers for adjoint system                         */
/*=================================================================*/

/*
 * IDAAres
 *
 * This routine interfaces to the RhsFnB routine provided by
 * the user.
 */

fn IDAAres(
    tt: sunrealtype,
    yyB: &N_Vector,
    ypB: &N_Vector,
    rrB: &N_Vector,
    ida_mem: &mut Option<Box<dyn Any>>,
) -> i32 {
    /* C: IDA_mem = (IDAMem)ida_mem — the backward integrator's user_data IS
    the forward IDAS memory (set by IDACreateB). */
    let IDA_mem: IDAMem = ida_mem
        .as_ref()
        .expect("IDAAres user_data")
        .downcast_ref::<IDAMem>()
        .expect("IDAAres user_data is the forward IDAMem")
        .clone();

    let IDAADJ_mem = IDAADJ_mem_of(&IDA_mem);

    /* Get the current backward problem. */
    let IDAB_mem = IDAADJ_mem
        .borrow()
        .ia_bckpbCrt
        .clone()
        .expect("ia_bckpbCrt");

    /* Get forward solution from interpolation. */
    let (noInterp, interpSensi, yyTmp, ypTmp, yySTmp, ypSTmp, getY) = {
        let ia = IDAADJ_mem.borrow();
        (
            ia.ia_noInterp,
            ia.ia_interpSensi,
            ia.ia_yyTmp.clone().expect("ia_yyTmp"),
            ia.ia_ypTmp.clone().expect("ia_ypTmp"),
            ia.ia_yySTmp.clone(),
            ia.ia_ypSTmp.clone(),
            ia.ia_getY.expect("ia_getY"),
        )
    };

    if noInterp == SUNFALSE {
        let flag = if interpSensi {
            getY(&IDA_mem, tt, &yyTmp, &ypTmp, &yySTmp, &ypSTmp)
        } else {
            getY(&IDA_mem, tt, &yyTmp, &ypTmp, &[], &[])
        };

        if flag != IDA_SUCCESS {
            IDAProcessError(
                Some(&IDA_mem),
                -1,
                line!() as i32,
                "IDAAres",
                file!(),
                &MSGAM_BAD_TINTERP(tt),
            );
            return -1;
        }
    }

    /* Call the user supplied residual. */
    let res_withSensi = IDAB_mem.borrow().ida_res_withSensi;

    if res_withSensi {
        let resS = IDAB_mem.borrow().ida_resS.expect("IDAB_mem->ida_resS");
        let mut user_dataB = IDAB_mem.borrow_mut().ida_user_data.take();
        let retval = resS(
            tt,
            &yyTmp,
            &ypTmp,
            &yySTmp,
            &ypSTmp,
            yyB,
            ypB,
            rrB,
            &mut user_dataB,
        );
        IDAB_mem.borrow_mut().ida_user_data = user_dataB;
        retval
    } else {
        let res = IDAB_mem.borrow().ida_res.expect("IDAB_mem->ida_res");
        let mut user_dataB = IDAB_mem.borrow_mut().ida_user_data.take();
        let retval = res(tt, &yyTmp, &ypTmp, yyB, ypB, rrB, &mut user_dataB);
        IDAB_mem.borrow_mut().ida_user_data = user_dataB;
        retval
    }
}

/*
 *IDAArhsQ
 *
 * This routine interfaces to the IDAQuadRhsFnB routine provided by
 * the user.
 *
 * It is passed to IDAQuadInit calls for backward problem, so it must
 * be of IDAQuadRhsFn type.
 */

fn IDAArhsQ(
    tt: sunrealtype,
    yyB: &N_Vector,
    ypB: &N_Vector,
    resvalQB: &N_Vector,
    ida_mem: &mut Option<Box<dyn Any>>,
) -> i32 {
    let IDA_mem: IDAMem = ida_mem
        .as_ref()
        .expect("IDAArhsQ user_data")
        .downcast_ref::<IDAMem>()
        .expect("IDAArhsQ user_data is the forward IDAMem")
        .clone();

    let IDAADJ_mem = IDAADJ_mem_of(&IDA_mem);

    /* Get current backward problem. */
    let IDAB_mem = IDAADJ_mem
        .borrow()
        .ia_bckpbCrt
        .clone()
        .expect("ia_bckpbCrt");

    /* retval = IDA_SUCCESS; (overwritten on every path below) */

    /* Get forward solution from interpolation. */
    let (noInterp, interpSensi, yyTmp, ypTmp, yySTmp, ypSTmp, getY) = {
        let ia = IDAADJ_mem.borrow();
        (
            ia.ia_noInterp,
            ia.ia_interpSensi,
            ia.ia_yyTmp.clone().expect("ia_yyTmp"),
            ia.ia_ypTmp.clone().expect("ia_ypTmp"),
            ia.ia_yySTmp.clone(),
            ia.ia_ypSTmp.clone(),
            ia.ia_getY.expect("ia_getY"),
        )
    };

    if noInterp == SUNFALSE {
        let flag = if interpSensi {
            getY(&IDA_mem, tt, &yyTmp, &ypTmp, &yySTmp, &ypSTmp)
        } else {
            getY(&IDA_mem, tt, &yyTmp, &ypTmp, &[], &[])
        };

        if flag != IDA_SUCCESS {
            IDAProcessError(
                Some(&IDA_mem),
                -1,
                line!() as i32,
                "IDAArhsQ",
                file!(),
                &MSGAM_BAD_TINTERP(tt),
            );
            return -1;
        }
    }

    /* Call user's adjoint quadrature RHS routine */
    let rhsQ_withSensi = IDAB_mem.borrow().ida_rhsQ_withSensi;

    if rhsQ_withSensi {
        let rhsQS = IDAB_mem.borrow().ida_rhsQS.expect("IDAB_mem->ida_rhsQS");
        let mut user_dataB = IDAB_mem.borrow_mut().ida_user_data.take();
        let retval = rhsQS(
            tt,
            &yyTmp,
            &ypTmp,
            &yySTmp,
            &ypSTmp,
            yyB,
            ypB,
            resvalQB,
            &mut user_dataB,
        );
        IDAB_mem.borrow_mut().ida_user_data = user_dataB;
        retval
    } else {
        let rhsQ = IDAB_mem.borrow().ida_rhsQ.expect("IDAB_mem->ida_rhsQ");
        let mut user_dataB = IDAB_mem.borrow_mut().ida_user_data.take();
        let retval = rhsQ(tt, &yyTmp, &ypTmp, yyB, ypB, resvalQB, &mut user_dataB);
        IDAB_mem.borrow_mut().ida_user_data = user_dataB;
        retval
    }
}
