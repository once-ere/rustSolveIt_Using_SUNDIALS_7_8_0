//! Port of `src/cvodes/cvodea.c` — the CVODEA adjoint sensitivity module.
//!
//! Structural mapping of the C intrusive lists (fixed by the contract in
//! [`crate::cvodes_impl`]):
//!
//! * `ca_mem->ck_mem` (checkpoint list, head = most recent checkpoint) becomes
//!   `CVadjMemRec::ck_mem: Vec<CVckpntMem>` with index 0 = list head.
//!   `ck->ck_next` ≡ the next index; `ck->ck_next == NULL` ≡ "is the last
//!   element" (the `t_initial` checkpoint created by `CVAckpntInit`).
//!   `CVAckpntDelete(&head)` ≡ `ck_mem.remove(0)` (dropping the record
//!   releases exactly the vectors the C routine destroys).
//! * `ca_mem->cvB_mem` (backward-problem list, head = most recently created)
//!   becomes `CVadjMemRec::cvB_mem: Vec<CVodeBMem>` with index 0 = list head.
//!   `for (p = cvB_mem; p; p = p->cv_next)` ≡ `for p in cvB_mem.iter()`.
//! * `ca_mem->dt_mem` (array of `steps+1` data points) becomes
//!   `Vec<CVdtpntMem>`; `dt_mem[i]->content` is the `Option<Box<dyn Any>>`
//!   holding a `CVhermiteDataMemRec` / `CVpolynomialDataMemRec` by value.
//!
//! Scratch-buffer note: the C code borrows `cv_mem->cv_cvals/cv_Xvecs/
//! cv_Zvecs` as scratch for the fused vector-array calls. Those are pure
//! write-then-read-immediately scratch areas; this port builds the identical
//! arrays as function-local `Vec`s instead, which keeps the values, the
//! `nvec` arguments and therefore the arithmetic bit-identical while avoiding
//! a mutable borrow of the mem across vector operations.

use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

use sundials_core::sundials_math::SUNRabs;
use sundials_core::sundials_nvector::{
    N_VClone, N_VCloneVectorArray, N_VDestroy, N_VLinearCombination,
    N_VLinearCombinationVectorArray, N_VLinearSum, N_VLinearSumVectorArray, N_VScale,
    N_VScaleVectorArray, N_Vector,
};
use sundials_core::sundials_types::{sunbooleantype, sunrealtype, SUNFALSE, SUNTRUE};

use crate::cvodes_impl::*;

/* -----------------------------------------------------------------
 * Symbols provided by the sibling cvodes modules (`cvodes.c` fragments
 * and `cvodes_io.c`). They are pulled through the crate prelude so this
 * module does not have to know how `cvodes.c` was split into fragment
 * modules; the prelude re-exports every cvodes module at crate level.
 * -----------------------------------------------------------------*/
use crate::prelude::{
    cvSensRhsWrapper, CVode, CVodeCreate, CVodeFree, CVodeGetDky, CVodeGetNumSteps, CVodeGetQuad,
    CVodeInit, CVodeQuadInit, CVodeQuadReInit, CVodeQuadSStolerances, CVodeQuadSVtolerances,
    CVodeQuadSensReInit, CVodeReInit, CVodeSStolerances, CVodeSVtolerances, CVodeSensReInit,
    CVodeSetInitStep, CVodeSetMaxHnilWarns, CVodeSetStopTime, CVodeSetUserData,
};

/*
 * =================================================================
 * CVODEA PRIVATE CONSTANTS
 * =================================================================
 */

/* NOTE: `cvodea.c` re-`#define`s these; in particular its FUZZ_FACTOR
(1000000.0) differs from the one `cvodes.c` uses (100.0), so these
module-local definitions deliberately shadow the ones re-exported from
`cvodes_impl`. */
const ZERO: sunrealtype = 0.0; /* real 0.0   */
const ONE: sunrealtype = 1.0; /* real 1.0   */
const TWO: sunrealtype = 2.0; /* real 2.0   */
const HUNDRED: sunrealtype = 100.0; /* real 100.0 */
const FUZZ_FACTOR: sunrealtype = 1000000.0; /* fuzz factor for IMget */

/*
 * =================================================================
 * SHORTCUTS FOR THE HANDLE MODEL
 * =================================================================
 */

/// C: `ca_mem = cv_mem->cv_adj_mem;`
///
/// The C code dereferences this without a NULL test in several places
/// (`CVodeGetAdjY`, the interpolation routines); a missing adjoint memory
/// is C undefined behavior and maps to a panic at the same site.
fn ca_mem_of(cv_mem: &CVodeMem) -> CVadjMem {
    cv_mem
        .borrow()
        .cv_adj_mem
        .clone()
        .expect("cv_adj_mem (CVodeAdjInit not called)")
}

/// C: `dt_mem[i]`
fn dt_pnt(ca_mem: &CVadjMem, i: i64) -> CVdtpntMem {
    ca_mem.borrow().dt_mem[i as usize].clone()
}

/// C: `dt_mem[i]->t`
fn dt_t(ca_mem: &CVadjMem, i: i64) -> sunrealtype {
    ca_mem.borrow().dt_mem[i as usize].borrow().t
}

/// C: `content = (CVhermiteDataMem)(dt_mem[i]->content)` followed by reads
/// of `content->y`, `->yd`, `->yS`, `->ySd` (handle copies, as in C).
fn herm_content(ca_mem: &CVadjMem, i: i64) -> (N_Vector, N_Vector, Vec<N_Vector>, Vec<N_Vector>) {
    let d = dt_pnt(ca_mem, i);
    let db = d.borrow();
    let content = db
        .content
        .as_ref()
        .expect("dt_mem content")
        .downcast_ref::<CVhermiteDataMemRec>()
        .expect("Hermite content");
    (
        content.y.clone().expect("content->y"),
        content.yd.clone().expect("content->yd"),
        content.yS.clone(),
        content.ySd.clone(),
    )
}

/// C: `content = (CVpolynomialDataMem)(dt_mem[i]->content)`.
fn poly_content(ca_mem: &CVadjMem, i: i64) -> (N_Vector, Vec<N_Vector>, i32) {
    let d = dt_pnt(ca_mem, i);
    let db = d.borrow();
    let content = db
        .content
        .as_ref()
        .expect("dt_mem content")
        .downcast_ref::<CVpolynomialDataMemRec>()
        .expect("polynomial content");
    (
        content.y.clone().expect("content->y"),
        content.yS.clone(),
        content.order,
    )
}

/// C: `cvB_mem = ca_mem->cvB_mem; while (cvB_mem != NULL) { if (which ==
/// cvB_mem->cv_index) break; cvB_mem = cvB_mem->cv_next; }`
///
/// This search is inlined ten times in `cvodea.c`; it is factored here with
/// identical semantics. If no entry matches, C leaves `cvB_mem == NULL` and
/// dereferences it immediately (undefined behavior) — that maps to a panic.
fn CVAfindBckpb(ca_mem: &CVadjMem, which: i32) -> CVodeBMem {
    let list = ca_mem.borrow().cvB_mem.clone();
    for cvB_mem in list.iter() {
        if which == cvB_mem.borrow().cv_index {
            return cvB_mem.clone();
        }
    }
    panic!("no backward problem with index {}", which);
}

/*
 * =================================================================
 * EXPORTED FUNCTIONS IMPLEMENTATION
 * =================================================================
 */

/*
 * CVodeAdjInit
 *
 * This routine initializes ASA and allocates space for the adjoint
 * memory structure.
 */

pub fn CVodeAdjInit(cvode_mem: &CVodeMem, steps: i64, interp: i32) -> i32 {
    /* ---------------
     * Check arguments
     * --------------- */

    /* NULL-mem check: handled by type system */
    let cv_mem = cvode_mem;

    if steps <= 0 {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeAdjInit",
            file!(),
            MSGCV_BAD_STEPS,
        );
        return CV_ILL_INPUT;
    }

    if (interp != CV_HERMITE) && (interp != CV_POLYNOMIAL) {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeAdjInit",
            file!(),
            MSGCV_BAD_INTERP,
        );
        return CV_ILL_INPUT;
    }

    /* ----------------------------
     * Allocate CVODEA memory block
     * ---------------------------- */

    let mut ca_mem = CVadjMemRec::zeroed();

    /* ------------------------------
     * Initialization of check points
     * ------------------------------ */

    /* Set Check Points linked list to NULL */
    ca_mem.ck_mem = Vec::new();

    /* Initialize nckpnts to ZERO */
    ca_mem.ca_nckpnts = 0;

    /* No interpolation data is available */
    ca_mem.ca_ckpntData = None;

    /* ------------------------------------
     * Initialization of interpolation data
     * ------------------------------------ */

    /* Interpolation type */

    ca_mem.ca_IMtype = interp;

    /* Number of steps between check points */

    ca_mem.ca_nsteps = steps;

    /* Last index used in CVAfindIndex, initialize to invalid value */
    ca_mem.ca_ilast = -1;

    /* Allocate space for the array of Data Point structures */

    ca_mem.dt_mem = Vec::with_capacity((steps + 1) as usize);
    for _i in 0..=steps {
        ca_mem.dt_mem.push(Rc::new(RefCell::new(CVdtpntMemRec {
            t: ZERO,
            content: None,
        })));
    }

    /* Attach functions for the appropriate interpolation module */

    match interp {
        CV_HERMITE => {
            ca_mem.ca_IMmalloc = Some(CVAhermiteMalloc);
            ca_mem.ca_IMfree = Some(CVAhermiteFree);
            ca_mem.ca_IMget = Some(CVAhermiteGetY);
            ca_mem.ca_IMstore = Some(CVAhermiteStorePnt);
        }

        CV_POLYNOMIAL => {
            ca_mem.ca_IMmalloc = Some(CVApolynomialMalloc);
            ca_mem.ca_IMfree = Some(CVApolynomialFree);
            ca_mem.ca_IMget = Some(CVApolynomialGetY);
            ca_mem.ca_IMstore = Some(CVApolynomialStorePnt);
        }

        _ => {}
    }

    /* The interpolation module has not been initialized yet */

    ca_mem.ca_IMmallocDone = SUNFALSE;

    /* By default we will store but not interpolate sensitivities
     *  - IMstoreSensi will be set in CVodeF to SUNFALSE if FSA is not enabled
     *    or if the user can force this through CVodeSetAdjNoSensi
     *  - IMinterpSensi will be set in CVodeB to SUNTRUE if IMstoreSensi is
     *    SUNTRUE and if at least one backward problem requires sensitivities */

    ca_mem.ca_IMstoreSensi = SUNTRUE;
    ca_mem.ca_IMinterpSensi = SUNFALSE;

    /* ------------------------------------
     * Initialize list of backward problems
     * ------------------------------------ */

    ca_mem.cvB_mem = Vec::new();
    ca_mem.ca_bckpbCrt = None;
    ca_mem.ca_nbckpbs = 0;

    /* --------------------------------
     * CVodeF and CVodeB not called yet
     * -------------------------------- */

    ca_mem.ca_firstCVodeFcall = SUNTRUE;
    ca_mem.ca_tstopCVodeFcall = SUNFALSE;

    ca_mem.ca_firstCVodeBcall = SUNTRUE;

    ca_mem.ca_rootret = SUNFALSE;

    /* ---------------------------------------------
     * Attach ca_mem to CVodeMem structure;
     * ASA initialized and allocated
     * --------------------------------------------- */

    let mut m = cv_mem.borrow_mut();
    m.cv_adj_mem = Some(Rc::new(RefCell::new(ca_mem)));
    m.cv_adj = SUNTRUE;
    m.cv_adjMallocDone = SUNTRUE;

    CV_SUCCESS
}

/* CVodeAdjReInit
 *
 * This routine reinitializes the CVODEA memory structure assuming that the
 * the number of steps between check points and the type of interpolation
 * remain unchanged.
 * The list of check points (and associated memory) is deleted.
 * The list of backward problems is kept (however, new backward problems can
 * be added to this list by calling CVodeCreateB).
 * The CVODES memory for the forward and backward problems can be reinitialized
 * separately by calling CVodeReInit and CVodeReInitB, respectively.
 * NOTE: if a completely new list of backward problems is also needed, then
 *       simply free the adjoint memory (by calling CVodeAdjFree) and reinitialize
 *       ASA with CVodeAdjInit.
 */

pub fn CVodeAdjReInit(cvode_mem: &CVodeMem) -> i32 {
    /* Check cvode_mem: NULL-mem check handled by type system */
    let cv_mem = cvode_mem;

    /* Was ASA initialized? */
    if !cv_mem.borrow().cv_adjMallocDone {
        cvProcessError(
            Some(cv_mem),
            CV_NO_ADJ,
            line!() as i32,
            "CVodeAdjReInit",
            file!(),
            MSGCV_NO_ADJ,
        );
        return CV_NO_ADJ;
    }

    let ca_mem = ca_mem_of(cv_mem);

    /* Free current list of Check Points */

    /* Initialization of check points */

    let mut ca = ca_mem.borrow_mut();
    ca.ck_mem.clear();
    ca.ca_nckpnts = 0;
    ca.ca_ckpntData = None;

    /* CVodeF and CVodeB not called yet */

    ca.ca_firstCVodeFcall = SUNTRUE;
    ca.ca_tstopCVodeFcall = SUNFALSE;
    ca.ca_firstCVodeBcall = SUNTRUE;

    CV_SUCCESS
}

/*
 * CVodeAdjFree
 *
 * This routine frees the memory allocated by CVodeAdjInit.
 */

pub fn CVodeAdjFree(cvode_mem: &CVodeMem) {
    /* NULL-mem check: handled by type system */
    let cv_mem = cvode_mem;

    if cv_mem.borrow().cv_adjMallocDone {
        let ca_mem = ca_mem_of(cv_mem);

        /* Delete check points one by one */
        ca_mem.borrow_mut().ck_mem.clear();

        /* Free vectors at all data points */
        let imfree = {
            let ca = ca_mem.borrow();
            if ca.ca_IMmallocDone {
                ca.ca_IMfree
            } else {
                None
            }
        };
        if let Some(imfree) = imfree {
            imfree(cv_mem);
        }
        ca_mem.borrow_mut().dt_mem.clear();

        /* Delete backward problems one by one */
        loop {
            let cvB_mem = {
                let mut ca = ca_mem.borrow_mut();
                if ca.cvB_mem.is_empty() {
                    None
                } else {
                    Some(ca.cvB_mem.remove(0))
                }
            };
            match cvB_mem {
                None => break,
                Some(cvB_mem) => CVAbckpbDelete(&cvB_mem),
            }
        }

        /* Free CVODEA memory */
        let mut m = cv_mem.borrow_mut();
        m.cv_adj_mem = None;
        m.cv_adjMallocDone = SUNFALSE;
    }
}

/*
 * CVodeF
 *
 * This routine integrates to tout and returns solution into yout.
 * In the same time, it stores check point data every 'steps' steps.
 *
 * CVodeF can be called repeatedly by the user.
 *
 * ncheckPtr points to the number of check points stored so far.
 */

pub fn CVodeF(
    cvode_mem: &CVodeMem,
    tout: sunrealtype,
    yout: &N_Vector,
    tret: &mut sunrealtype,
    itask: i32,
    ncheckPtr: &mut i32,
) -> i32 {
    /* Check if cvode_mem exists: NULL-mem check handled by type system */
    let cv_mem = cvode_mem;

    /* Was ASA initialized? */
    if !cv_mem.borrow().cv_adjMallocDone {
        cvProcessError(
            Some(cv_mem),
            CV_NO_ADJ,
            line!() as i32,
            "CVodeF",
            file!(),
            MSGCV_NO_ADJ,
        );
        return CV_NO_ADJ;
    }

    let ca_mem = ca_mem_of(cv_mem);

    /* Check for yout != NULL: handled by type system */
    /* Check for tret != NULL: handled by type system */

    /* Check for valid itask */
    if (itask != CV_NORMAL) && (itask != CV_ONE_STEP) {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeF",
            file!(),
            MSGCV_BAD_ITASK,
        );
        return CV_ILL_INPUT;
    }

    /* All error checking done */

    let mut flag: i32 = 0;

    /* If tstop is enabled, store some info */
    let (tstopset, tstop) = {
        let m = cv_mem.borrow();
        (m.cv_tstopset, m.cv_tstop)
    };
    if tstopset {
        let mut ca = ca_mem.borrow_mut();
        ca.ca_tstopCVodeFcall = SUNTRUE;
        ca.ca_tstopCVodeF = tstop;
    }

    /* On the first step:
     *   - set tinitial
     *   - initialize list of check points
     *   - if needed, initialize the interpolation module
     *   - load dt_mem[0]
     * On subsequent steps, test if taking a new step is necessary.
     */
    if ca_mem.borrow().ca_firstCVodeFcall {
        let tn = cv_mem.borrow().cv_tn;
        ca_mem.borrow_mut().ca_tinitial = tn;

        match CVAckpntInit(cv_mem) {
            None => {
                cvProcessError(
                    Some(cv_mem),
                    CV_MEM_FAIL,
                    line!() as i32,
                    "CVodeF",
                    file!(),
                    MSGCV_MEM_FAIL,
                );
                return CV_MEM_FAIL;
            }
            Some(ck_mem) => {
                ca_mem.borrow_mut().ck_mem = vec![ck_mem];
            }
        }

        if !ca_mem.borrow().ca_IMmallocDone {
            /* Do we need to store sensitivities? */
            if !cv_mem.borrow().cv_sensi {
                ca_mem.borrow_mut().ca_IMstoreSensi = SUNFALSE;
            }

            /* Allocate space for interpolation data */
            let immalloc = ca_mem.borrow().ca_IMmalloc.expect("ca_IMmalloc");
            let allocOK = immalloc(cv_mem);
            if !allocOK {
                cvProcessError(
                    Some(cv_mem),
                    CV_MEM_FAIL,
                    line!() as i32,
                    "CVodeF",
                    file!(),
                    MSGCV_MEM_FAIL,
                );
                return CV_MEM_FAIL;
            }

            /* Rename zn and, if needed, znS for use in interpolation
            (handle copies: ca_Y/ca_YS alias the integrator's Nordsieck
            arrays exactly as the C pointer copies do) */
            for i in 0..L_MAX {
                let zn_i = cv_mem.borrow().cv_zn[i].clone();
                ca_mem.borrow_mut().ca_Y[i] = zn_i;
            }
            if ca_mem.borrow().ca_IMstoreSensi {
                for i in 0..L_MAX {
                    let znS_i = cv_mem.borrow().cv_znS[i].clone();
                    ca_mem.borrow_mut().ca_YS[i] = znS_i;
                }
            }

            ca_mem.borrow_mut().ca_IMmallocDone = SUNTRUE;
        }

        let (dt0, ck_t0, imstore) = {
            let ca = ca_mem.borrow();
            let dt0 = ca.dt_mem[0].clone();
            let ck_t0 = ca.ck_mem[0].borrow().ck_t0;
            let imstore = ca.ca_IMstore.expect("ca_IMstore");
            (dt0, ck_t0, imstore)
        };
        dt0.borrow_mut().t = ck_t0;
        let _ = imstore(cv_mem, &dt0);

        ca_mem.borrow_mut().ca_firstCVodeFcall = SUNFALSE;
    } else if itask == CV_NORMAL {
        /* When in normal mode, check if tout was passed or if a previous root was
        not reported and return an interpolated solution. No changes to ck_mem
        or dt_mem are needed. */

        /* flag to signal if an early return is needed */
        let mut earlyret = SUNFALSE;

        /* if a root needs to be reported compare tout to troot otherwise compare
        to the current time tn */
        let (rootret, troot) = {
            let ca = ca_mem.borrow();
            (ca.ca_rootret, ca.ca_troot)
        };
        let (tn, h) = {
            let m = cv_mem.borrow();
            (m.cv_tn, m.cv_h)
        };
        let ttest = if rootret { troot } else { tn };

        if (ttest - tout) * h >= ZERO {
            /* ttest is after tout, interpolate to tout */
            *tret = tout;
            flag = CVodeGetDky(cv_mem, tout, 0, yout);
            earlyret = SUNTRUE;
        } else if rootret {
            /* tout is after troot, interpolate to troot */
            *tret = troot;
            /* C assigns the CVodeGetDky return to `flag` and immediately
             * overwrites it with CV_ROOT_RETURN (cvodea.c:545-546). */
            let _ = CVodeGetDky(cv_mem, troot, 0, yout);
            flag = CV_ROOT_RETURN;
            ca_mem.borrow_mut().ca_rootret = SUNFALSE;
            earlyret = SUNTRUE;
        }

        /* return if necessary */
        if earlyret {
            let nst = cv_mem.borrow().cv_nst;
            let mut ca = ca_mem.borrow_mut();
            *ncheckPtr = ca.ca_nckpnts;
            ca.ca_IMnewData = SUNTRUE;
            let head = ca.ck_mem[0].clone();
            ca.ca_ckpntData = Some(head);
            let nsteps = ca.ca_nsteps;
            ca.ca_np = nst % nsteps + 1;
            return flag;
        }
    }

    /* Integrate to tout (in CV_ONE_STEP mode) while loading check points */
    let mut nstloc: i64 = 0;
    loop {
        /* Check for too many steps */

        let (mxstep, tn_now) = {
            let m = cv_mem.borrow();
            (m.cv_mxstep, m.cv_tn)
        };
        if (mxstep > 0) && (nstloc >= mxstep) {
            cvProcessError(
                Some(cv_mem),
                CV_TOO_MUCH_WORK,
                line!() as i32,
                "CVodeF",
                file!(),
                &MSGCV_MAX_STEPS(tn_now),
            );
            flag = CV_TOO_MUCH_WORK;
            break;
        }

        /* Perform one step of the integration */

        flag = CVode(cv_mem, tout, yout, tret, CV_ONE_STEP);
        if flag < 0 {
            break;
        }

        nstloc += 1;

        /* Test if a new check point is needed */

        let (nst, tn) = {
            let m = cv_mem.borrow();
            (m.cv_nst, m.cv_tn)
        };
        let nsteps = ca_mem.borrow().ca_nsteps;

        if nst % nsteps == 0 {
            {
                let ca = ca_mem.borrow();
                ca.ck_mem[0].borrow_mut().ck_t1 = tn;
            }

            /* Create a new check point, load it, and append it to the list */
            match CVAckpntNew(cv_mem) {
                None => {
                    cvProcessError(
                        Some(cv_mem),
                        CV_MEM_FAIL,
                        line!() as i32,
                        "CVodeF",
                        file!(),
                        MSGCV_MEM_FAIL,
                    );
                    flag = CV_MEM_FAIL;
                    break;
                }
                Some(tmp) => {
                    let mut ca = ca_mem.borrow_mut();
                    ca.ck_mem.insert(0, tmp);
                    ca.ca_nckpnts += 1;
                }
            }
            cv_mem.borrow_mut().cv_forceSetup = SUNTRUE;

            /* Reset i=0 and load dt_mem[0] */
            let (dt0, ck_t0, imstore) = {
                let ca = ca_mem.borrow();
                let dt0 = ca.dt_mem[0].clone();
                let ck_t0 = ca.ck_mem[0].borrow().ck_t0;
                let imstore = ca.ca_IMstore.expect("ca_IMstore");
                (dt0, ck_t0, imstore)
            };
            dt0.borrow_mut().t = ck_t0;
            let _ = imstore(cv_mem, &dt0);
        } else {
            /* Load next point in dt_mem */
            let (dti, imstore) = {
                let ca = ca_mem.borrow();
                (
                    ca.dt_mem[(nst % nsteps) as usize].clone(),
                    ca.ca_IMstore.expect("ca_IMstore"),
                )
            };
            dti.borrow_mut().t = tn;
            let _ = imstore(cv_mem, &dti);
        }

        /* Set t1 field of the current check point structure
        for the case in which there will be no future
        check points */
        {
            let ca = ca_mem.borrow();
            ca.ck_mem[0].borrow_mut().ck_t1 = tn;
        }

        /* tfinal is now set to tn */
        ca_mem.borrow_mut().ca_tfinal = tn;

        /* Return if in CV_ONE_STEP mode */
        if itask == CV_ONE_STEP {
            break;
        }

        /* CV_NORMAL_STEP returns */

        /* Return if tout reached */
        let h = cv_mem.borrow().cv_h;
        if (*tret - tout) * h >= ZERO {
            /* If this was a root return, save the root time to return later */
            if flag == CV_ROOT_RETURN {
                let mut ca = ca_mem.borrow_mut();
                ca.ca_rootret = SUNTRUE;
                ca.ca_troot = *tret;
            }

            /* Get solution value at tout to return now */
            *tret = tout;
            flag = CVodeGetDky(cv_mem, tout, 0, yout);

            /* Reset tretlast in cv_mem so that CVodeGetQuad and CVodeGetSens
             * evaluate quadratures and/or sensitivities at the proper time */
            cv_mem.borrow_mut().cv_tretlast = tout;

            break;
        }

        /* Return if tstop or a root was found */
        if (flag == CV_TSTOP_RETURN) || (flag == CV_ROOT_RETURN) {
            break;
        }
    } /* end of for(;;) */

    /* Get ncheck from ca_mem */
    let nst = cv_mem.borrow().cv_nst;
    {
        let mut ca = ca_mem.borrow_mut();
        *ncheckPtr = ca.ca_nckpnts;

        /* Data is available for the last interval */
        ca.ca_IMnewData = SUNTRUE;
        let head = ca.ck_mem[0].clone();
        ca.ca_ckpntData = Some(head);
        let nsteps = ca.ca_nsteps;
        ca.ca_np = nst % nsteps + 1;
    }

    flag
}

/*
 * =================================================================
 * FUNCTIONS FOR BACKWARD PROBLEMS
 * =================================================================
 */

pub fn CVodeCreateB(cvode_mem: &CVodeMem, lmmB: i32, which: &mut i32) -> i32 {
    /* Check if cvode_mem exists: NULL-mem check handled by type system */
    let cv_mem = cvode_mem;

    /* Was ASA initialized? */
    if !cv_mem.borrow().cv_adjMallocDone {
        cvProcessError(
            Some(cv_mem),
            CV_NO_ADJ,
            line!() as i32,
            "CVodeCreateB",
            file!(),
            MSGCV_NO_ADJ,
        );
        return CV_NO_ADJ;
    }
    let ca_mem = ca_mem_of(cv_mem);

    /* Allocate space for new CVodeBMem object */

    let mut new_cvB_mem = CVodeBMemRec::zeroed();

    /* Create and set a new CVODES object for the backward problem */

    let sunctx = cv_mem.borrow().cv_sunctx.clone();
    let cvodeB_mem = match CVodeCreate(lmmB, &sunctx) {
        None => {
            cvProcessError(
                Some(cv_mem),
                CV_MEM_FAIL,
                line!() as i32,
                "CVodeCreateB",
                file!(),
                MSGCV_MEM_FAIL,
            );
            return CV_MEM_FAIL;
        }
        Some(m) => m,
    };

    /* We need to ensure Ns is set in the new CVODES object so that Ns is accessible
    in the Python callbacks which only have access to cvodeB_mem, not the original cvode_mem */
    let Ns = cv_mem.borrow().cv_Ns;
    cvodeB_mem.borrow_mut().cv_Ns = Ns;

    /* C: CVodeSetUserData(cvodeB_mem, cvode_mem) — the backward integrator's
    user_data IS the forward integrator memory; CVArhs/CVArhsQ recover it. */
    let _ = CVodeSetUserData(&cvodeB_mem, Some(Box::new(cv_mem.clone())));

    let _ = CVodeSetMaxHnilWarns(&cvodeB_mem, -1);

    /* Set/initialize fields in the new CVodeBMem object, new_cvB_mem */

    new_cvB_mem.cv_index = ca_mem.borrow().ca_nbckpbs;

    new_cvB_mem.cv_mem = Some(cvodeB_mem);

    new_cvB_mem.cv_f = None;
    new_cvB_mem.cv_fs = None;

    new_cvB_mem.cv_fQ = None;
    new_cvB_mem.cv_fQs = None;

    new_cvB_mem.cv_user_data = None;

    new_cvB_mem.cv_lmem = None;
    new_cvB_mem.cv_lfree = None;
    new_cvB_mem.cv_pmem = None;
    new_cvB_mem.cv_pfree = None;

    new_cvB_mem.cv_y = None;

    new_cvB_mem.cv_f_withSensi = SUNFALSE;
    new_cvB_mem.cv_fQ_withSensi = SUNFALSE;

    /* Attach the new object to the linked list cvB_mem */

    let mut ca = ca_mem.borrow_mut();
    ca.cvB_mem.insert(0, Rc::new(RefCell::new(new_cvB_mem)));

    /* Return the index of the newly created CVodeBMem object.
     * This must be passed to CVodeInitB and to other ***B
     * functions to set optional inputs for this backward problem */

    *which = ca.ca_nbckpbs;

    ca.ca_nbckpbs += 1;

    CV_SUCCESS
}

pub fn CVodeInitB(
    cvode_mem: &CVodeMem,
    which: i32,
    fB: CVRhsFnB,
    tB0: sunrealtype,
    yB0: &N_Vector,
) -> i32 {
    /* Check if cvode_mem exists: NULL-mem check handled by type system */
    let cv_mem = cvode_mem;

    /* Was ASA initialized? */

    if !cv_mem.borrow().cv_adjMallocDone {
        cvProcessError(
            Some(cv_mem),
            CV_NO_ADJ,
            line!() as i32,
            "CVodeInitB",
            file!(),
            MSGCV_NO_ADJ,
        );
        return CV_NO_ADJ;
    }
    let ca_mem = ca_mem_of(cv_mem);

    /* Check the value of which */

    if which >= ca_mem.borrow().ca_nbckpbs {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeInitB",
            file!(),
            MSGCV_BAD_WHICH,
        );
        return CV_ILL_INPUT;
    }

    /* Find the CVodeBMem entry in the linked list corresponding to which */

    let cvB_mem = CVAfindBckpb(&ca_mem, which);

    let cvodeB_mem = cvB_mem.borrow().cv_mem.clone().expect("cvB_mem->cv_mem");

    /* Allocate and set the CVODES object */

    let flag = CVodeInit(&cvodeB_mem, CVArhs, tB0, yB0);

    if flag != CV_SUCCESS {
        return flag;
    }

    /* Copy fB function in cvB_mem */

    {
        let mut b = cvB_mem.borrow_mut();
        b.cv_f_withSensi = SUNFALSE;
        b.cv_f = Some(fB);

        /* Allocate space and initialize the y Nvector in cvB_mem */

        b.cv_t0 = tB0;
    }
    let y = N_VClone(yB0).expect("N_VClone(yB0)");
    N_VScale(ONE, yB0, &y);
    cvB_mem.borrow_mut().cv_y = Some(y);

    CV_SUCCESS
}

pub fn CVodeInitBS(
    cvode_mem: &CVodeMem,
    which: i32,
    fBs: CVRhsFnBS,
    tB0: sunrealtype,
    yB0: &N_Vector,
) -> i32 {
    /* Check if cvode_mem exists: NULL-mem check handled by type system */
    let cv_mem = cvode_mem;

    /* Was ASA initialized? */

    if !cv_mem.borrow().cv_adjMallocDone {
        cvProcessError(
            Some(cv_mem),
            CV_NO_ADJ,
            line!() as i32,
            "CVodeInitBS",
            file!(),
            MSGCV_NO_ADJ,
        );
        return CV_NO_ADJ;
    }
    let ca_mem = ca_mem_of(cv_mem);

    /* Check the value of which */

    if which >= ca_mem.borrow().ca_nbckpbs {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeInitBS",
            file!(),
            MSGCV_BAD_WHICH,
        );
        return CV_ILL_INPUT;
    }

    /* Find the CVodeBMem entry in the linked list corresponding to which */

    let cvB_mem = CVAfindBckpb(&ca_mem, which);

    let cvodeB_mem = cvB_mem.borrow().cv_mem.clone().expect("cvB_mem->cv_mem");

    /* Allocate and set the CVODES object */

    let flag = CVodeInit(&cvodeB_mem, CVArhs, tB0, yB0);

    if flag != CV_SUCCESS {
        return flag;
    }

    /* Copy fBs function in cvB_mem */

    {
        let mut b = cvB_mem.borrow_mut();
        b.cv_f_withSensi = SUNTRUE;
        b.cv_fs = Some(fBs);

        /* Allocate space and initialize the y Nvector in cvB_mem */

        b.cv_t0 = tB0;
    }
    let y = N_VClone(yB0).expect("N_VClone(yB0)");
    N_VScale(ONE, yB0, &y);
    cvB_mem.borrow_mut().cv_y = Some(y);

    CV_SUCCESS
}

pub fn CVodeReInitB(cvode_mem: &CVodeMem, which: i32, tB0: sunrealtype, yB0: &N_Vector) -> i32 {
    /* Check if cvode_mem exists: NULL-mem check handled by type system */
    let cv_mem = cvode_mem;

    /* Was ASA initialized? */
    if !cv_mem.borrow().cv_adjMallocDone {
        cvProcessError(
            Some(cv_mem),
            CV_NO_ADJ,
            line!() as i32,
            "CVodeReInitB",
            file!(),
            MSGCV_NO_ADJ,
        );
        return CV_NO_ADJ;
    }
    let ca_mem = ca_mem_of(cv_mem);

    /* Check the value of which */
    if which >= ca_mem.borrow().ca_nbckpbs {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeReInitB",
            file!(),
            MSGCV_BAD_WHICH,
        );
        return CV_ILL_INPUT;
    }

    /* Find the CVodeBMem entry in the linked list corresponding to which */
    let cvB_mem = CVAfindBckpb(&ca_mem, which);

    let cvodeB_mem = cvB_mem.borrow().cv_mem.clone().expect("cvB_mem->cv_mem");

    /* Reinitialize CVODES object */

    CVodeReInit(&cvodeB_mem, tB0, yB0)
}

pub fn CVodeSStolerancesB(
    cvode_mem: &CVodeMem,
    which: i32,
    reltolB: sunrealtype,
    abstolB: sunrealtype,
) -> i32 {
    /* Check if cvode_mem exists: NULL-mem check handled by type system */
    let cv_mem = cvode_mem;

    /* Was ASA initialized? */

    if !cv_mem.borrow().cv_adjMallocDone {
        cvProcessError(
            Some(cv_mem),
            CV_NO_ADJ,
            line!() as i32,
            "CVodeSStolerancesB",
            file!(),
            MSGCV_NO_ADJ,
        );
        return CV_NO_ADJ;
    }
    let ca_mem = ca_mem_of(cv_mem);

    /* Check the value of which */

    if which >= ca_mem.borrow().ca_nbckpbs {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeSStolerancesB",
            file!(),
            MSGCV_BAD_WHICH,
        );
        return CV_ILL_INPUT;
    }

    /* Find the CVodeBMem entry in the linked list corresponding to which */

    let cvB_mem = CVAfindBckpb(&ca_mem, which);

    let cvodeB_mem = cvB_mem.borrow().cv_mem.clone().expect("cvB_mem->cv_mem");

    /* Set tolerances */

    CVodeSStolerances(&cvodeB_mem, reltolB, abstolB)
}

pub fn CVodeSVtolerancesB(
    cvode_mem: &CVodeMem,
    which: i32,
    reltolB: sunrealtype,
    abstolB: &N_Vector,
) -> i32 {
    /* Check if cvode_mem exists: NULL-mem check handled by type system */
    let cv_mem = cvode_mem;

    /* Was ASA initialized? */

    if !cv_mem.borrow().cv_adjMallocDone {
        cvProcessError(
            Some(cv_mem),
            CV_NO_ADJ,
            line!() as i32,
            "CVodeSVtolerancesB",
            file!(),
            MSGCV_NO_ADJ,
        );
        return CV_NO_ADJ;
    }
    let ca_mem = ca_mem_of(cv_mem);

    /* Check the value of which */

    if which >= ca_mem.borrow().ca_nbckpbs {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeSVtolerancesB",
            file!(),
            MSGCV_BAD_WHICH,
        );
        return CV_ILL_INPUT;
    }

    /* Find the CVodeBMem entry in the linked list corresponding to which */

    let cvB_mem = CVAfindBckpb(&ca_mem, which);

    let cvodeB_mem = cvB_mem.borrow().cv_mem.clone().expect("cvB_mem->cv_mem");

    /* Set tolerances */

    CVodeSVtolerances(&cvodeB_mem, reltolB, abstolB)
}

pub fn CVodeQuadInitB(cvode_mem: &CVodeMem, which: i32, fQB: CVQuadRhsFnB, yQB0: &N_Vector) -> i32 {
    /* Check if cvode_mem exists: NULL-mem check handled by type system */
    let cv_mem = cvode_mem;

    /* Was ASA initialized? */
    if !cv_mem.borrow().cv_adjMallocDone {
        cvProcessError(
            Some(cv_mem),
            CV_NO_ADJ,
            line!() as i32,
            "CVodeQuadInitB",
            file!(),
            MSGCV_NO_ADJ,
        );
        return CV_NO_ADJ;
    }
    let ca_mem = ca_mem_of(cv_mem);

    /* Check which */
    if which >= ca_mem.borrow().ca_nbckpbs {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeQuadInitB",
            file!(),
            MSGCV_BAD_WHICH,
        );
        return CV_ILL_INPUT;
    }

    /* Find the CVodeBMem entry in the linked list corresponding to which */
    let cvB_mem = CVAfindBckpb(&ca_mem, which);

    let cvodeB_mem = cvB_mem.borrow().cv_mem.clone().expect("cvB_mem->cv_mem");

    let flag = CVodeQuadInit(&cvodeB_mem, CVArhsQ, yQB0);
    if flag != CV_SUCCESS {
        return flag;
    }

    let mut b = cvB_mem.borrow_mut();
    b.cv_fQ_withSensi = SUNFALSE;
    b.cv_fQ = Some(fQB);

    CV_SUCCESS
}

pub fn CVodeQuadInitBS(
    cvode_mem: &CVodeMem,
    which: i32,
    fQBs: CVQuadRhsFnBS,
    yQB0: &N_Vector,
) -> i32 {
    /* Check if cvode_mem exists: NULL-mem check handled by type system */
    let cv_mem = cvode_mem;

    /* Was ASA initialized? */
    if !cv_mem.borrow().cv_adjMallocDone {
        cvProcessError(
            Some(cv_mem),
            CV_NO_ADJ,
            line!() as i32,
            "CVodeQuadInitBS",
            file!(),
            MSGCV_NO_ADJ,
        );
        return CV_NO_ADJ;
    }
    let ca_mem = ca_mem_of(cv_mem);

    /* Check which */
    if which >= ca_mem.borrow().ca_nbckpbs {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeQuadInitBS",
            file!(),
            MSGCV_BAD_WHICH,
        );
        return CV_ILL_INPUT;
    }

    /* Find the CVodeBMem entry in the linked list corresponding to which */
    let cvB_mem = CVAfindBckpb(&ca_mem, which);

    let cvodeB_mem = cvB_mem.borrow().cv_mem.clone().expect("cvB_mem->cv_mem");

    let flag = CVodeQuadInit(&cvodeB_mem, CVArhsQ, yQB0);
    if flag != CV_SUCCESS {
        return flag;
    }

    let mut b = cvB_mem.borrow_mut();
    b.cv_fQ_withSensi = SUNTRUE;
    b.cv_fQs = Some(fQBs);

    CV_SUCCESS
}

pub fn CVodeQuadReInitB(cvode_mem: &CVodeMem, which: i32, yQB0: &N_Vector) -> i32 {
    /* Check if cvode_mem exists: NULL-mem check handled by type system */
    let cv_mem = cvode_mem;

    /* Was ASA initialized? */
    if !cv_mem.borrow().cv_adjMallocDone {
        cvProcessError(
            Some(cv_mem),
            CV_NO_ADJ,
            line!() as i32,
            "CVodeQuadReInitB",
            file!(),
            MSGCV_NO_ADJ,
        );
        return CV_NO_ADJ;
    }
    let ca_mem = ca_mem_of(cv_mem);

    /* Check the value of which */
    if which >= ca_mem.borrow().ca_nbckpbs {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeQuadReInitB",
            file!(),
            MSGCV_BAD_WHICH,
        );
        return CV_ILL_INPUT;
    }

    /* Find the CVodeBMem entry in the linked list corresponding to which */
    let cvB_mem = CVAfindBckpb(&ca_mem, which);

    let cvodeB_mem = cvB_mem.borrow().cv_mem.clone().expect("cvB_mem->cv_mem");

    let flag = CVodeQuadReInit(&cvodeB_mem, yQB0);
    if flag != CV_SUCCESS {
        return flag;
    }

    CV_SUCCESS
}

pub fn CVodeQuadSStolerancesB(
    cvode_mem: &CVodeMem,
    which: i32,
    reltolQB: sunrealtype,
    abstolQB: sunrealtype,
) -> i32 {
    /* Check if cvode_mem exists: NULL-mem check handled by type system */
    let cv_mem = cvode_mem;

    /* Was ASA initialized? */
    if !cv_mem.borrow().cv_adjMallocDone {
        cvProcessError(
            Some(cv_mem),
            CV_NO_ADJ,
            line!() as i32,
            "CVodeQuadSStolerancesB",
            file!(),
            MSGCV_NO_ADJ,
        );
        return CV_NO_ADJ;
    }
    let ca_mem = ca_mem_of(cv_mem);

    /* Check which */
    if which >= ca_mem.borrow().ca_nbckpbs {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeQuadSStolerancesB",
            file!(),
            MSGCV_BAD_WHICH,
        );
        return CV_ILL_INPUT;
    }

    /* Find the CVodeBMem entry in the linked list corresponding to which */
    let cvB_mem = CVAfindBckpb(&ca_mem, which);

    let cvodeB_mem = cvB_mem.borrow().cv_mem.clone().expect("cvB_mem->cv_mem");

    CVodeQuadSStolerances(&cvodeB_mem, reltolQB, abstolQB)
}

pub fn CVodeQuadSVtolerancesB(
    cvode_mem: &CVodeMem,
    which: i32,
    reltolQB: sunrealtype,
    abstolQB: &N_Vector,
) -> i32 {
    /* Check if cvode_mem exists: NULL-mem check handled by type system */
    let cv_mem = cvode_mem;

    /* Was ASA initialized? */
    if !cv_mem.borrow().cv_adjMallocDone {
        cvProcessError(
            Some(cv_mem),
            CV_NO_ADJ,
            line!() as i32,
            "CVodeQuadSVtolerancesB",
            file!(),
            MSGCV_NO_ADJ,
        );
        return CV_NO_ADJ;
    }
    let ca_mem = ca_mem_of(cv_mem);

    /* Check which */
    if which >= ca_mem.borrow().ca_nbckpbs {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeQuadSVtolerancesB",
            file!(),
            MSGCV_BAD_WHICH,
        );
        return CV_ILL_INPUT;
    }

    /* Find the CVodeBMem entry in the linked list corresponding to which */
    let cvB_mem = CVAfindBckpb(&ca_mem, which);

    let cvodeB_mem = cvB_mem.borrow().cv_mem.clone().expect("cvB_mem->cv_mem");

    CVodeQuadSVtolerances(&cvodeB_mem, reltolQB, abstolQB)
}

/*
 * CVodeB
 *
 * This routine performs the backward integration towards tBout
 * of all backward problems that were defined.
 * When necessary, it performs a forward integration between two
 * consecutive check points to update interpolation data.
 *
 * On a successful return, CVodeB returns CV_SUCCESS.
 *
 * NOTE that CVodeB DOES NOT return the solution for the backward
 * problem(s). Use CVodeGetB to extract the solution at tBret
 * for any given backward problem.
 *
 * If there are multiple backward problems and multiple check points,
 * CVodeB may not succeed in getting all problems to take one step
 * when called in ONE_STEP mode.
 */

pub fn CVodeB(cvode_mem: &CVodeMem, tBout: sunrealtype, itaskB: i32) -> i32 {
    /* Check if cvode_mem exists: NULL-mem check handled by type system */
    let cv_mem = cvode_mem;

    /* C modifies its `tBout` parameter in the fuzz-tolerance branch below */
    let mut tBout = tBout;
    let mut flag: i32 = 0;

    /* Was ASA initialized? */

    if !cv_mem.borrow().cv_adjMallocDone {
        cvProcessError(
            Some(cv_mem),
            CV_NO_ADJ,
            line!() as i32,
            "CVodeB",
            file!(),
            MSGCV_NO_ADJ,
        );
        return CV_NO_ADJ;
    }
    let ca_mem = ca_mem_of(cv_mem);

    /* Check if any backward problem has been defined */

    if ca_mem.borrow().ca_nbckpbs == 0 {
        cvProcessError(
            Some(cv_mem),
            CV_NO_BCK,
            line!() as i32,
            "CVodeB",
            file!(),
            MSGCV_NO_BCK,
        );
        return CV_NO_BCK;
    }
    let cvB_mem: Vec<CVodeBMem> = ca_mem.borrow().cvB_mem.clone();

    /* Check whether CVodeF has been called */

    if ca_mem.borrow().ca_firstCVodeFcall {
        cvProcessError(
            Some(cv_mem),
            CV_NO_FWD,
            line!() as i32,
            "CVodeB",
            file!(),
            MSGCV_NO_FWD,
        );
        return CV_NO_FWD;
    }

    let (ca_tinitial, ca_tfinal) = {
        let ca = ca_mem.borrow();
        (ca.ca_tinitial, ca.ca_tfinal)
    };
    let sign: sunrealtype = if ca_tfinal - ca_tinitial > ZERO {
        1.0
    } else {
        -1.0
    };

    /* If this is the first call, loop over all backward problems and
     *   - check that tB0 is valid
     *   - check that tBout is ahead of tB0 in the backward direction
     *   - check whether we need to interpolate forward sensitivities
     */

    if ca_mem.borrow().ca_firstCVodeBcall {
        for tmp_cvB_mem in cvB_mem.iter() {
            let (bmem, cv_index, f_withSensi, fQ_withSensi) = {
                let b = tmp_cvB_mem.borrow();
                (
                    b.cv_mem.clone().expect("cvB_mem->cv_mem"),
                    b.cv_index,
                    b.cv_f_withSensi,
                    b.cv_fQ_withSensi,
                )
            };
            let tBn = bmem.borrow().cv_tn;

            if (sign * (tBn - ca_tinitial) < ZERO) || (sign * (ca_tfinal - tBn) < ZERO) {
                cvProcessError(
                    Some(cv_mem),
                    CV_BAD_TB0,
                    line!() as i32,
                    "CVodeB",
                    file!(),
                    &MSGCV_BAD_TB0(cv_index),
                );
                return CV_BAD_TB0;
            }

            if sign * (tBn - tBout) <= ZERO {
                cvProcessError(
                    Some(cv_mem),
                    CV_ILL_INPUT,
                    line!() as i32,
                    "CVodeB",
                    file!(),
                    MSGCV_BAD_TBOUT,
                );
                return CV_ILL_INPUT;
            }

            if f_withSensi || fQ_withSensi {
                ca_mem.borrow_mut().ca_IMinterpSensi = SUNTRUE;
            }
        }

        let (IMinterpSensi, IMstoreSensi) = {
            let ca = ca_mem.borrow();
            (ca.ca_IMinterpSensi, ca.ca_IMstoreSensi)
        };
        if IMinterpSensi && !IMstoreSensi {
            cvProcessError(
                Some(cv_mem),
                CV_ILL_INPUT,
                line!() as i32,
                "CVodeB",
                file!(),
                MSGCV_BAD_SENSI,
            );
            return CV_ILL_INPUT;
        }

        ca_mem.borrow_mut().ca_firstCVodeBcall = SUNFALSE;
    }

    /* Check if itaskB is legal */

    if (itaskB != CV_NORMAL) && (itaskB != CV_ONE_STEP) {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeB",
            file!(),
            MSGCV_BAD_ITASKB,
        );
        return CV_ILL_INPUT;
    }

    /* Check if tBout is legal */

    if (sign * (tBout - ca_tinitial) < ZERO) || (sign * (ca_tfinal - tBout) < ZERO) {
        let uround = cv_mem.borrow().cv_uround;
        let tfuzz = HUNDRED * uround * (SUNRabs(ca_tinitial) + SUNRabs(ca_tfinal));
        if (sign * (tBout - ca_tinitial) < ZERO) && (SUNRabs(tBout - ca_tinitial) < tfuzz) {
            tBout = ca_tinitial;
        } else {
            cvProcessError(
                Some(cv_mem),
                CV_ILL_INPUT,
                line!() as i32,
                "CVodeB",
                file!(),
                MSGCV_BAD_TBOUT,
            );
            return CV_ILL_INPUT;
        }
    }

    /* Loop through the check points and stop as soon as a backward
     * problem has its tn value behind the current check point's t0_
     * value (in the backward direction) */

    let mut ck_idx: usize = 0;

    let mut gotCheckpoint = SUNFALSE;

    loop {
        let ck_t0 = {
            let ck = ca_mem.borrow().ck_mem[ck_idx].clone();
            let ck_t0 = ck.borrow().ck_t0;
            ck_t0
        };

        for tmp_cvB_mem in cvB_mem.iter() {
            let bmem = tmp_cvB_mem
                .borrow()
                .cv_mem
                .clone()
                .expect("cvB_mem->cv_mem");
            let tBn = bmem.borrow().cv_tn;

            if sign * (tBn - ck_t0) > ZERO {
                gotCheckpoint = SUNTRUE;
                break;
            }

            if (itaskB == CV_NORMAL) && (tBn == ck_t0) && (sign * (tBout - ck_t0) >= ZERO) {
                gotCheckpoint = SUNTRUE;
                break;
            }
        }

        if gotCheckpoint {
            break;
        }

        /* C: if (ck_mem->ck_next == NULL) break; */
        if ck_idx + 1 >= ca_mem.borrow().ck_mem.len() {
            break;
        }

        ck_idx += 1;
    }

    /* Starting with the current check point from above, loop over check points
    while propagating backward problems */

    loop {
        let ck_mem = {
            let ca = ca_mem.borrow();
            ca.ck_mem[ck_idx].clone()
        };

        /* Store interpolation data if not available.
        This is the 2nd forward integration pass */

        let is_ckpntData = {
            let ca = ca_mem.borrow();
            match &ca.ca_ckpntData {
                None => false,
                Some(c) => Rc::ptr_eq(c, &ck_mem),
            }
        };
        if !is_ckpntData {
            flag = CVAdataStore(cv_mem, &ck_mem);
            if flag != CV_SUCCESS {
                break;
            }
        }

        let ck_t0 = ck_mem.borrow().ck_t0;

        /* Loop through all backward problems and, if needed,
         * propagate their solution towards tBout */

        let mut errIndex: i32 = 0;
        for tmp_cvB_mem in cvB_mem.iter() {
            /* Decide if current backward problem is "active" in this check point */

            let mut isActive = SUNTRUE;

            let bmem = tmp_cvB_mem
                .borrow()
                .cv_mem
                .clone()
                .expect("cvB_mem->cv_mem");
            let tBn = bmem.borrow().cv_tn;

            if (tBn == ck_t0) && (sign * (tBout - ck_t0) < ZERO) {
                isActive = SUNFALSE;
            }
            if (tBn == ck_t0) && (itaskB == CV_ONE_STEP) {
                isActive = SUNFALSE;
            }

            if sign * (tBn - ck_t0) < ZERO {
                isActive = SUNFALSE;
            }

            if isActive {
                /* Store the address of current backward problem memory
                 * in ca_mem to be used in the wrapper functions */
                ca_mem.borrow_mut().ca_bckpbCrt = Some(tmp_cvB_mem.clone());

                /* Integrate current backward problem */
                let _ = CVodeSetStopTime(&bmem, ck_t0);
                let yB = tmp_cvB_mem.borrow().cv_y.clone().expect("cvB_mem->cv_y");
                let mut tBret: sunrealtype = ZERO;
                flag = CVode(&bmem, tBout, &yB, &mut tBret, itaskB);

                /* Set the time at which we will report solution and/or quadratures */
                tmp_cvB_mem.borrow_mut().cv_tout = tBret;

                /* If an error occurred, exit while loop */
                if flag < 0 {
                    errIndex = tmp_cvB_mem.borrow().cv_index;
                    break;
                }
            } else {
                flag = CV_SUCCESS;
                tmp_cvB_mem.borrow_mut().cv_tout = tBn;
            }

            /* Move to next backward problem */
        }

        /* If an error occurred, return now */

        if flag < 0 {
            cvProcessError(
                Some(cv_mem),
                flag,
                line!() as i32,
                "CVodeB",
                file!(),
                &MSGCV_BACK_ERROR(errIndex),
            );
            return flag;
        }

        /* If in CV_ONE_STEP mode, return now (flag = CV_SUCCESS) */

        if itaskB == CV_ONE_STEP {
            break;
        }

        /* If all backward problems have successfully reached tBout, return now */

        let mut reachedTBout = SUNTRUE;

        for tmp_cvB_mem in cvB_mem.iter() {
            let cv_tout = tmp_cvB_mem.borrow().cv_tout;
            if sign * (cv_tout - tBout) > ZERO {
                reachedTBout = SUNFALSE;
                break;
            }
        }

        if reachedTBout {
            break;
        }

        /* Move check point in linked list to next one */

        ck_idx += 1;
    }

    flag
}

pub fn CVodeGetB(cvode_mem: &CVodeMem, which: i32, tret: &mut sunrealtype, yB: &N_Vector) -> i32 {
    /* Check if cvode_mem exists: NULL-mem check handled by type system */
    let cv_mem = cvode_mem;

    /* Was ASA initialized? */
    if !cv_mem.borrow().cv_adjMallocDone {
        cvProcessError(
            Some(cv_mem),
            CV_NO_ADJ,
            line!() as i32,
            "CVodeGetB",
            file!(),
            MSGCV_NO_ADJ,
        );
        return CV_NO_ADJ;
    }

    let ca_mem = ca_mem_of(cv_mem);

    /* Check the value of which */
    if which >= ca_mem.borrow().ca_nbckpbs {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeGetB",
            file!(),
            MSGCV_BAD_WHICH,
        );
        return CV_ILL_INPUT;
    }

    /* Find the CVodeBMem entry in the linked list corresponding to which */
    let cvB_mem = CVAfindBckpb(&ca_mem, which);

    let (y, cv_tout) = {
        let b = cvB_mem.borrow();
        (b.cv_y.clone().expect("cvB_mem->cv_y"), b.cv_tout)
    };
    N_VScale(ONE, &y, yB);
    *tret = cv_tout;

    CV_SUCCESS
}

/*
 * CVodeGetQuadB
 */

pub fn CVodeGetQuadB(
    cvode_mem: &CVodeMem,
    which: i32,
    tret: &mut sunrealtype,
    qB: &N_Vector,
) -> i32 {
    /* Check if cvode_mem exists: NULL-mem check handled by type system */
    let cv_mem = cvode_mem;

    /* Was ASA initialized? */
    if !cv_mem.borrow().cv_adjMallocDone {
        cvProcessError(
            Some(cv_mem),
            CV_NO_ADJ,
            line!() as i32,
            "CVodeGetQuadB",
            file!(),
            MSGCV_NO_ADJ,
        );
        return CV_NO_ADJ;
    }

    let ca_mem = ca_mem_of(cv_mem);

    /* Check the value of which */
    if which >= ca_mem.borrow().ca_nbckpbs {
        cvProcessError(
            Some(cv_mem),
            CV_ILL_INPUT,
            line!() as i32,
            "CVodeGetQuadB",
            file!(),
            MSGCV_BAD_WHICH,
        );
        return CV_ILL_INPUT;
    }

    /* Find the CVodeBMem entry in the linked list corresponding to which */
    let cvB_mem = CVAfindBckpb(&ca_mem, which);

    let cvodeB_mem = cvB_mem.borrow().cv_mem.clone().expect("cvB_mem->cv_mem");

    /* If the integration for this backward problem has not started yet,
     * simply return the current value of qB (i.e. the final conditions) */

    let mut nstB: i64 = 0;
    let mut flag = CVodeGetNumSteps(&cvodeB_mem, &mut nstB);

    if nstB == 0 {
        let znQ0 = cvodeB_mem.borrow().cv_znQ[0]
            .clone()
            .expect("cvB_mem->cv_mem->cv_znQ[0]");
        N_VScale(ONE, &znQ0, qB);
        *tret = cvB_mem.borrow().cv_tout;
    } else {
        flag = CVodeGetQuad(&cvodeB_mem, tret, qB);
    }

    flag
}

/*
 * =================================================================
 * PRIVATE FUNCTIONS FOR CHECK POINTS
 * =================================================================
 */

/*
 * CVAckpntInit
 *
 * This routine initializes the check point linked list with
 * information from the initial time.
 */

fn CVAckpntInit(cv_mem: &CVodeMem) -> Option<CVckpntMem> {
    /* Allocate space for ckdata */
    let mut ck_mem = CVckpntMemRec::zeroed();

    let tempv = cv_mem.borrow().cv_tempv.clone().expect("cv_tempv");

    ck_mem.ck_zn[0] = Some(N_VClone(&tempv)?);

    ck_mem.ck_zn[1] = Some(N_VClone(&tempv)?);

    /* ck_mem->ck_zn[qmax] was not allocated */
    ck_mem.ck_zqm = 0;

    /* Load ckdata from cv_mem */
    let zn0 = cv_mem.borrow().cv_zn[0].clone().expect("cv_zn[0]");
    N_VScale(ONE, &zn0, ck_mem.ck_zn[0].as_ref().expect("ck_zn[0]"));
    ck_mem.ck_t0 = cv_mem.borrow().cv_tn;
    ck_mem.ck_nst = 0;
    ck_mem.ck_q = 1;
    ck_mem.ck_h = ZERO;

    /* Do we need to carry quadratures */
    let (quadr, errconQ, sensi, Ns, quadr_sensi, errconQS) = {
        let m = cv_mem.borrow();
        (
            m.cv_quadr,
            m.cv_errconQ,
            m.cv_sensi,
            m.cv_Ns,
            m.cv_quadr_sensi,
            m.cv_errconQS,
        )
    };
    ck_mem.ck_quadr = quadr && errconQ;

    if ck_mem.ck_quadr {
        let tempvQ = cv_mem.borrow().cv_tempvQ.clone().expect("cv_tempvQ");
        ck_mem.ck_znQ[0] = Some(N_VClone(&tempvQ)?);

        let znQ0 = cv_mem.borrow().cv_znQ[0].clone().expect("cv_znQ[0]");
        N_VScale(ONE, &znQ0, ck_mem.ck_znQ[0].as_ref().expect("ck_znQ[0]"));
    }

    /* Do we need to carry sensitivities? */
    ck_mem.ck_sensi = sensi;

    if ck_mem.ck_sensi {
        ck_mem.ck_Ns = Ns;

        ck_mem.ck_znS[0] = N_VCloneVectorArray(Ns, &tempv)?;

        let cvals = vec![ONE; Ns as usize];

        let znS0 = cv_mem.borrow().cv_znS[0].clone();
        let _ = N_VScaleVectorArray(Ns, &cvals, &znS0, &ck_mem.ck_znS[0]);
    }

    /* Do we need to carry quadrature sensitivities? */
    ck_mem.ck_quadr_sensi = quadr_sensi && errconQS;

    if ck_mem.ck_quadr_sensi {
        let tempvQ = cv_mem.borrow().cv_tempvQ.clone().expect("cv_tempvQ");
        ck_mem.ck_znQS[0] = N_VCloneVectorArray(Ns, &tempvQ)?;

        let cvals = vec![ONE; Ns as usize];

        let znQS0 = cv_mem.borrow().cv_znQS[0].clone();
        let _ = N_VScaleVectorArray(Ns, &cvals, &znQS0, &ck_mem.ck_znQS[0]);
    }

    /* Next in list: the caller places this record at the head of ck_mem */

    Some(Rc::new(RefCell::new(ck_mem)))
}

/*
 * CVAckpntNew
 *
 * This routine allocates space for a new check point and sets
 * its data from current values in cv_mem.
 */

fn CVAckpntNew(cv_mem: &CVodeMem) -> Option<CVckpntMem> {
    /* Allocate space for ckdata */
    let mut ck_mem = CVckpntMemRec::zeroed();

    let (q, qmax, Ns, quadr, errconQ, sensi, quadr_sensi, errconQS) = {
        let m = cv_mem.borrow();
        (
            m.cv_q,
            m.cv_qmax,
            m.cv_Ns,
            m.cv_quadr,
            m.cv_errconQ,
            m.cv_sensi,
            m.cv_quadr_sensi,
            m.cv_errconQS,
        )
    };

    let tempv = cv_mem.borrow().cv_tempv.clone().expect("cv_tempv");

    /* Test if we need to allocate space for the last zn.
     * NOTE: zn(qmax) may be needed for a hot restart, if an order
     * increase is deemed necessary at the first step after a check point */
    ck_mem.ck_zqm = if q < qmax { qmax } else { 0 };

    for j in 0..=q as usize {
        ck_mem.ck_zn[j] = Some(N_VClone(&tempv)?);
    }

    if q < qmax {
        ck_mem.ck_zn[qmax as usize] = Some(N_VClone(&tempv)?);
    }

    /* Test if we need to carry quadratures */
    ck_mem.ck_quadr = quadr && errconQ;

    if ck_mem.ck_quadr {
        let tempvQ = cv_mem.borrow().cv_tempvQ.clone().expect("cv_tempvQ");
        for j in 0..=q as usize {
            ck_mem.ck_znQ[j] = Some(N_VClone(&tempvQ)?);
        }

        if q < qmax {
            ck_mem.ck_znQ[qmax as usize] = Some(N_VClone(&tempvQ)?);
        }
    }

    /* Test if we need to carry sensitivities */
    ck_mem.ck_sensi = sensi;

    if ck_mem.ck_sensi {
        ck_mem.ck_Ns = Ns;

        for j in 0..=q as usize {
            ck_mem.ck_znS[j] = N_VCloneVectorArray(Ns, &tempv)?;
        }

        if q < qmax {
            ck_mem.ck_znS[qmax as usize] = N_VCloneVectorArray(Ns, &tempv)?;
        }
    }

    /* Test if we need to carry quadrature sensitivities */
    ck_mem.ck_quadr_sensi = quadr_sensi && errconQS;

    if ck_mem.ck_quadr_sensi {
        let tempvQ = cv_mem.borrow().cv_tempvQ.clone().expect("cv_tempvQ");
        for j in 0..=q as usize {
            ck_mem.ck_znQS[j] = N_VCloneVectorArray(Ns, &tempvQ)?;
        }

        if q < qmax {
            ck_mem.ck_znQS[qmax as usize] = N_VCloneVectorArray(Ns, &tempvQ)?;
        }
    }

    /* Load check point data from cv_mem */

    {
        let cvals = vec![ONE; (q + 1) as usize];

        let Xvecs: Vec<N_Vector> = {
            let m = cv_mem.borrow();
            (0..=q as usize)
                .map(|j| m.cv_zn[j].clone().expect("cv_zn[j]"))
                .collect()
        };
        let Zvecs: Vec<N_Vector> = (0..=q as usize)
            .map(|j| ck_mem.ck_zn[j].clone().expect("ck_zn[j]"))
            .collect();

        let _ = N_VScaleVectorArray(q + 1, &cvals, &Xvecs, &Zvecs);
    }

    if q < qmax {
        let znqmax = cv_mem.borrow().cv_zn[qmax as usize]
            .clone()
            .expect("cv_zn[qmax]");
        N_VScale(
            ONE,
            &znqmax,
            ck_mem.ck_zn[qmax as usize].as_ref().expect("ck_zn[qmax]"),
        );
    }

    if ck_mem.ck_quadr {
        let cvals = vec![ONE; (q + 1) as usize];

        let Xvecs: Vec<N_Vector> = {
            let m = cv_mem.borrow();
            (0..=q as usize)
                .map(|j| m.cv_znQ[j].clone().expect("cv_znQ[j]"))
                .collect()
        };
        let Zvecs: Vec<N_Vector> = (0..=q as usize)
            .map(|j| ck_mem.ck_znQ[j].clone().expect("ck_znQ[j]"))
            .collect();

        let _ = N_VScaleVectorArray(q + 1, &cvals, &Xvecs, &Zvecs);

        if q < qmax {
            let znQqmax = cv_mem.borrow().cv_znQ[qmax as usize]
                .clone()
                .expect("cv_znQ[qmax]");
            N_VScale(
                ONE,
                &znQqmax,
                ck_mem.ck_znQ[qmax as usize].as_ref().expect("ck_znQ[qmax]"),
            );
        }
    }

    if ck_mem.ck_sensi {
        let n = (Ns * (q + 1)) as usize;
        let mut cvals = vec![ZERO; n];
        let mut Xvecs: Vec<N_Vector> = Vec::with_capacity(n);
        let mut Zvecs: Vec<N_Vector> = Vec::with_capacity(n);

        {
            let m = cv_mem.borrow();
            for j in 0..=q as usize {
                for is in 0..Ns as usize {
                    cvals[j * Ns as usize + is] = ONE;
                    Xvecs.push(m.cv_znS[j][is].clone());
                    Zvecs.push(ck_mem.ck_znS[j][is].clone());
                }
            }
        }

        let _ = N_VScaleVectorArray(Ns * (q + 1), &cvals, &Xvecs, &Zvecs);

        if q < qmax {
            let cvals = vec![ONE; Ns as usize];

            let znSqmax = cv_mem.borrow().cv_znS[qmax as usize].clone();
            let _ = N_VScaleVectorArray(Ns, &cvals, &znSqmax, &ck_mem.ck_znS[qmax as usize]);
        }
    }

    if ck_mem.ck_quadr_sensi {
        let n = (Ns * (q + 1)) as usize;
        let mut cvals = vec![ZERO; n];
        let mut Xvecs: Vec<N_Vector> = Vec::with_capacity(n);
        let mut Zvecs: Vec<N_Vector> = Vec::with_capacity(n);

        {
            let m = cv_mem.borrow();
            for j in 0..=q as usize {
                for is in 0..Ns as usize {
                    cvals[j * Ns as usize + is] = ONE;
                    Xvecs.push(m.cv_znQS[j][is].clone());
                    Zvecs.push(ck_mem.ck_znQS[j][is].clone());
                }
            }
        }

        /* NOTE: upstream passes `cv_mem->cv_Ns` here (not Ns*(q+1)); only the
        first Ns pairs are actually scaled. Preserved verbatim. */
        let _ = N_VScaleVectorArray(Ns, &cvals, &Xvecs, &Zvecs);

        if q < qmax {
            let cvals = vec![ONE; Ns as usize];

            let znQSqmax = cv_mem.borrow().cv_znQS[qmax as usize].clone();
            let _ = N_VScaleVectorArray(Ns, &cvals, &znQSqmax, &ck_mem.ck_znQS[qmax as usize]);
        }
    }

    {
        let m = cv_mem.borrow();
        for j in 0..=L_MAX {
            ck_mem.ck_tau[j] = m.cv_tau[j];
        }
        for j in 0..=NUM_TESTS {
            ck_mem.ck_tq[j] = m.cv_tq[j];
        }
        for j in 0..=q as usize {
            ck_mem.ck_l[j] = m.cv_l[j];
        }
        ck_mem.ck_nst = m.cv_nst;
        ck_mem.ck_tretlast = m.cv_tretlast;
        ck_mem.ck_q = m.cv_q;
        ck_mem.ck_qprime = m.cv_qprime;
        ck_mem.ck_qwait = m.cv_qwait;
        ck_mem.ck_L = m.cv_L;
        ck_mem.ck_gammap = m.cv_gammap;
        ck_mem.ck_h = m.cv_h;
        ck_mem.ck_hprime = m.cv_hprime;
        ck_mem.ck_hscale = m.cv_hscale;
        ck_mem.ck_eta = m.cv_eta;
        ck_mem.ck_etamax = m.cv_etamax;
        ck_mem.ck_t0 = m.cv_tn;
        ck_mem.ck_saved_tq5 = m.cv_saved_tq5;
    }

    Some(Rc::new(RefCell::new(ck_mem)))
}

/*
 * CVAckpntDelete
 *
 * C deletes the first check point in the list and returns the new list
 * head, destroying exactly the N_Vectors that check point owns. Under the
 * handle model, removing the record from `ca_mem->ck_mem` and dropping it
 * releases the same vectors, so this routine has no Rust counterpart —
 * `ck_mem.remove(0)` / `ck_mem.clear()` at the call sites is the port.
 */

/*
 * =================================================================
 * PRIVATE FUNCTIONS FOR BACKWARD PROBLEMS
 * =================================================================
 */

/// C: `CVAbckpbDelete(CVodeBMem* cvB_memPtr)`. The list-head move is done by
/// the caller (`ca_mem->cvB_mem.remove(0)`); this performs the teardown of
/// the removed entry.
fn CVAbckpbDelete(cvB_mem: &CVodeBMem) {
    /* Free CVODES memory in tmp */
    let mut cvode_mem = cvB_mem.borrow_mut().cv_mem.take();
    CVodeFree(&mut cvode_mem);

    /* Free linear solver memory */
    let lfree = cvB_mem.borrow().cv_lfree;
    if let Some(lfree) = lfree {
        let _ = lfree(cvB_mem);
    }

    /* Free preconditioner memory */
    let pfree = cvB_mem.borrow().cv_pfree;
    if let Some(pfree) = pfree {
        let _ = pfree(cvB_mem);
    }

    /* Free workspace Nvector */
    let y = cvB_mem.borrow_mut().cv_y.take();
    if let Some(y) = y {
        N_VDestroy(y);
    }
}

/*
 * =================================================================
 * PRIVATE FUNCTIONS FOR INTERPOLATION
 * =================================================================
 */

/*
 * CVAdataStore
 *
 * This routine integrates the forward model starting at the check
 * point ck_mem and stores y and yprime at all intermediate steps.
 *
 * Return values:
 * CV_SUCCESS
 * CV_REIFWD_FAIL
 * CV_FWD_FAIL
 */

fn CVAdataStore(cv_mem: &CVodeMem, ck_mem: &CVckpntMem) -> i32 {
    let ca_mem = ca_mem_of(cv_mem);

    /* Initialize cv_mem with data from ck_mem */
    let flag = CVAckpntGet(cv_mem, ck_mem);
    if flag != CV_SUCCESS {
        return CV_REIFWD_FAIL;
    }

    /* Set first structure in dt_mem[0] */
    let (dt0, imstore) = {
        let ca = ca_mem.borrow();
        (ca.dt_mem[0].clone(), ca.ca_IMstore.expect("ca_IMstore"))
    };
    let ck_t0 = ck_mem.borrow().ck_t0;
    dt0.borrow_mut().t = ck_t0;
    let _ = imstore(cv_mem, &dt0);

    /* Decide whether TSTOP must be activated */
    let (tstopCVodeFcall, tstopCVodeF) = {
        let ca = ca_mem.borrow();
        (ca.ca_tstopCVodeFcall, ca.ca_tstopCVodeF)
    };
    if tstopCVodeFcall {
        let _ = CVodeSetStopTime(cv_mem, tstopCVodeF);
    }

    let (ca_tinitial, ca_tfinal) = {
        let ca = ca_mem.borrow();
        (ca.ca_tinitial, ca.ca_tfinal)
    };
    let sign: sunrealtype = if ca_tfinal - ca_tinitial > ZERO {
        1.0
    } else {
        -1.0
    };

    let ck_t1 = ck_mem.borrow().ck_t1;
    let ytmp = ca_mem.borrow().ca_ytmp.clone().expect("ca_ytmp");

    /* Run CVode to set following structures in dt_mem[i] */
    let mut i: i64 = 1;
    let mut t: sunrealtype = ZERO;
    loop {
        let flag = CVode(cv_mem, ck_t1, &ytmp, &mut t, CV_ONE_STEP);
        if flag < 0 {
            return CV_FWD_FAIL;
        }

        let (dti, imstore) = {
            let ca = ca_mem.borrow();
            (
                ca.dt_mem[i as usize].clone(),
                ca.ca_IMstore.expect("ca_IMstore"),
            )
        };
        dti.borrow_mut().t = t;
        let _ = imstore(cv_mem, &dti);
        i += 1;

        if !(sign * (ck_t1 - t) > ZERO) {
            break;
        }
    }

    let mut ca = ca_mem.borrow_mut();
    ca.ca_IMnewData = SUNTRUE; /* New data is now available    */
    ca.ca_ckpntData = Some(ck_mem.clone()); /* starting at this check point */
    ca.ca_np = i; /* and we have this many points */

    CV_SUCCESS
}

/*
 * CVAckpntGet
 *
 * This routine prepares CVODES for a hot restart from
 * the check point ck_mem
 */

fn CVAckpntGet(cv_mem: &CVodeMem, ck_mem: &CVckpntMem) -> i32 {
    let ca_mem = ca_mem_of(cv_mem);

    /* C: ck_mem->ck_next == NULL, i.e. this is the t_initial check point
    (the tail of the list, which in the Vec model is the last element). */
    let next_is_null = {
        let ca = ca_mem.borrow();
        ca.ck_mem
            .last()
            .map_or(false, |tail| Rc::ptr_eq(tail, ck_mem))
    };

    if next_is_null {
        /* In this case, we just call the reinitialization routine,
         * but make sure we use the same initial stepsize as on
         * the first run. */

        let h0u = cv_mem.borrow().cv_h0u;
        let _ = CVodeSetInitStep(cv_mem, h0u);

        let (ck_t0, zn0) = {
            let ck = ck_mem.borrow();
            (ck.ck_t0, ck.ck_zn[0].clone().expect("ck_zn[0]"))
        };
        let flag = CVodeReInit(cv_mem, ck_t0, &zn0);
        if flag != CV_SUCCESS {
            return flag;
        }

        if ck_mem.borrow().ck_quadr {
            let znQ0 = ck_mem.borrow().ck_znQ[0].clone().expect("ck_znQ[0]");
            let flag = CVodeQuadReInit(cv_mem, &znQ0);
            if flag != CV_SUCCESS {
                return flag;
            }
        }

        if ck_mem.borrow().ck_sensi {
            let ism = cv_mem.borrow().cv_ism;
            let znS0 = ck_mem.borrow().ck_znS[0].clone();
            let flag = CVodeSensReInit(cv_mem, ism, &znS0);
            if flag != CV_SUCCESS {
                return flag;
            }
        }

        if ck_mem.borrow().ck_quadr_sensi {
            let znQS0 = ck_mem.borrow().ck_znQS[0].clone();
            let flag = CVodeQuadSensReInit(cv_mem, &znQS0);
            if flag != CV_SUCCESS {
                return flag;
            }
        }
    } else {
        let qmax = cv_mem.borrow().cv_qmax;

        /* Copy parameters from check point data structure */

        {
            let ck = ck_mem.borrow();
            let mut m = cv_mem.borrow_mut();
            m.cv_nst = ck.ck_nst;
            m.cv_tretlast = ck.ck_tretlast;
            m.cv_q = ck.ck_q;
            m.cv_qprime = ck.ck_qprime;
            m.cv_qwait = ck.ck_qwait;
            m.cv_L = ck.ck_L;
            m.cv_gammap = ck.ck_gammap;
            m.cv_h = ck.ck_h;
            m.cv_hprime = ck.ck_hprime;
            m.cv_hscale = ck.ck_hscale;
            m.cv_eta = ck.ck_eta;
            m.cv_etamax = ck.ck_etamax;
            m.cv_tn = ck.ck_t0;
            m.cv_saved_tq5 = ck.ck_saved_tq5;
        }

        let (q, Ns) = {
            let m = cv_mem.borrow();
            (m.cv_q, m.cv_Ns)
        };

        /* Copy the arrays from check point data structure */

        {
            let cvals = vec![ONE; (q + 1) as usize];

            let Xvecs: Vec<N_Vector> = {
                let ck = ck_mem.borrow();
                (0..=q as usize)
                    .map(|j| ck.ck_zn[j].clone().expect("ck_zn[j]"))
                    .collect()
            };
            let Zvecs: Vec<N_Vector> = {
                let m = cv_mem.borrow();
                (0..=q as usize)
                    .map(|j| m.cv_zn[j].clone().expect("cv_zn[j]"))
                    .collect()
            };

            let retval = N_VScaleVectorArray(q + 1, &cvals, &Xvecs, &Zvecs);
            if retval != CV_SUCCESS {
                return CV_VECTOROP_ERR;
            }
        }

        if q < qmax {
            let ckznqmax = ck_mem.borrow().ck_zn[qmax as usize]
                .clone()
                .expect("ck_zn[qmax]");
            let znqmax = cv_mem.borrow().cv_zn[qmax as usize]
                .clone()
                .expect("cv_zn[qmax]");
            N_VScale(ONE, &ckznqmax, &znqmax);
        }

        if ck_mem.borrow().ck_quadr {
            let cvals = vec![ONE; (q + 1) as usize];

            let Xvecs: Vec<N_Vector> = {
                let ck = ck_mem.borrow();
                (0..=q as usize)
                    .map(|j| ck.ck_znQ[j].clone().expect("ck_znQ[j]"))
                    .collect()
            };
            let Zvecs: Vec<N_Vector> = {
                let m = cv_mem.borrow();
                (0..=q as usize)
                    .map(|j| m.cv_znQ[j].clone().expect("cv_znQ[j]"))
                    .collect()
            };

            let retval = N_VScaleVectorArray(q + 1, &cvals, &Xvecs, &Zvecs);
            if retval != CV_SUCCESS {
                return CV_VECTOROP_ERR;
            }

            if q < qmax {
                let ckznQqmax = ck_mem.borrow().ck_znQ[qmax as usize]
                    .clone()
                    .expect("ck_znQ[qmax]");
                let znQqmax = cv_mem.borrow().cv_znQ[qmax as usize]
                    .clone()
                    .expect("cv_znQ[qmax]");
                N_VScale(ONE, &ckznQqmax, &znQqmax);
            }
        }

        if ck_mem.borrow().ck_sensi {
            let n = (Ns * (q + 1)) as usize;
            let mut cvals = vec![ZERO; n];
            let mut Xvecs: Vec<N_Vector> = Vec::with_capacity(n);
            let mut Zvecs: Vec<N_Vector> = Vec::with_capacity(n);

            {
                let ck = ck_mem.borrow();
                let m = cv_mem.borrow();
                for j in 0..=q as usize {
                    for is in 0..Ns as usize {
                        cvals[j * Ns as usize + is] = ONE;
                        Xvecs.push(ck.ck_znS[j][is].clone());
                        Zvecs.push(m.cv_znS[j][is].clone());
                    }
                }
            }

            let retval = N_VScaleVectorArray(Ns * (q + 1), &cvals, &Xvecs, &Zvecs);
            if retval != CV_SUCCESS {
                return CV_VECTOROP_ERR;
            }

            if q < qmax {
                let cvals = vec![ONE; Ns as usize];

                let ckznSqmax = ck_mem.borrow().ck_znS[qmax as usize].clone();
                let znSqmax = cv_mem.borrow().cv_znS[qmax as usize].clone();

                let retval = N_VScaleVectorArray(Ns, &cvals, &ckznSqmax, &znSqmax);
                if retval != CV_SUCCESS {
                    return CV_VECTOROP_ERR;
                }
            }
        }

        if ck_mem.borrow().ck_quadr_sensi {
            let n = (Ns * (q + 1)) as usize;
            let mut cvals = vec![ZERO; n];
            let mut Xvecs: Vec<N_Vector> = Vec::with_capacity(n);
            let mut Zvecs: Vec<N_Vector> = Vec::with_capacity(n);

            {
                let ck = ck_mem.borrow();
                let m = cv_mem.borrow();
                for j in 0..=q as usize {
                    for is in 0..Ns as usize {
                        cvals[j * Ns as usize + is] = ONE;
                        Xvecs.push(ck.ck_znQS[j][is].clone());
                        Zvecs.push(m.cv_znQS[j][is].clone());
                    }
                }
            }

            let retval = N_VScaleVectorArray(Ns * (q + 1), &cvals, &Xvecs, &Zvecs);
            if retval != CV_SUCCESS {
                return CV_VECTOROP_ERR;
            }

            if q < qmax {
                let cvals = vec![ONE; Ns as usize];

                let ckznQSqmax = ck_mem.borrow().ck_znQS[qmax as usize].clone();
                let znQSqmax = cv_mem.borrow().cv_znQS[qmax as usize].clone();

                let retval = N_VScaleVectorArray(Ns, &cvals, &ckznQSqmax, &znQSqmax);
                if retval != CV_SUCCESS {
                    return CV_VECTOROP_ERR;
                }
            }
        }

        {
            let ck = ck_mem.borrow();
            let mut m = cv_mem.borrow_mut();
            for j in 0..=L_MAX {
                m.cv_tau[j] = ck.ck_tau[j];
            }
            for j in 0..=NUM_TESTS {
                m.cv_tq[j] = ck.ck_tq[j];
            }
            for j in 0..=q as usize {
                m.cv_l[j] = ck.ck_l[j];
            }

            /* Force a call to setup */

            m.cv_forceSetup = SUNTRUE;
        }
    }

    CV_SUCCESS
}

/*
 * -----------------------------------------------------------------
 * Functions for interpolation
 * -----------------------------------------------------------------
 */

/*
 * CVAfindIndex
 *
 * Finds the index in the array of data point structures such that
 *     dt_mem[index-1].t <= t < dt_mem[index].t
 * If index is changed from the previous invocation, then newpoint = SUNTRUE
 *
 * If t is beyond the leftmost limit, but close enough, index=0.
 *
 * Returns CV_SUCCESS if successful and CV_GETY_BADT if unable to
 * find index (t is too far beyond limits).
 */

fn CVAfindIndex(
    cv_mem: &CVodeMem,
    t: sunrealtype,
    index: &mut i64,
    newpoint: &mut sunbooleantype,
) -> i32 {
    let ca_mem = ca_mem_of(cv_mem);

    *newpoint = SUNFALSE;

    /* Find the direction of integration */
    let (ca_tinitial, ca_tfinal) = {
        let ca = ca_mem.borrow();
        (ca.ca_tinitial, ca.ca_tfinal)
    };
    let sign: sunrealtype = if ca_tfinal - ca_tinitial > ZERO {
        1.0
    } else {
        -1.0
    };

    /* If this is the first time we use new data */
    if ca_mem.borrow().ca_IMnewData {
        let np = ca_mem.borrow().ca_np;
        let mut ca = ca_mem.borrow_mut();
        ca.ca_ilast = np - 1;
        *newpoint = SUNTRUE;
        ca.ca_IMnewData = SUNFALSE;
    }

    /* Search for index starting from ilast */
    let ilast = ca_mem.borrow().ca_ilast;
    let to_left = sign * (t - dt_t(&ca_mem, ilast - 1)) < ZERO;
    let to_right = sign * (t - dt_t(&ca_mem, ilast)) > ZERO;

    if to_left {
        /* look for a new index to the left */

        *newpoint = SUNTRUE;

        *index = ilast;
        loop {
            if *index == 0 {
                break;
            }
            if sign * (t - dt_t(&ca_mem, *index - 1)) <= ZERO {
                *index -= 1;
            } else {
                break;
            }
        }

        if *index == 0 {
            ca_mem.borrow_mut().ca_ilast = 1;
        } else {
            ca_mem.borrow_mut().ca_ilast = *index;
        }

        if *index == 0 {
            /* t is beyond leftmost limit. Is it too far? */
            let uround = cv_mem.borrow().cv_uround;
            if SUNRabs(t - dt_t(&ca_mem, 0)) > FUZZ_FACTOR * uround {
                return CV_GETY_BADT;
            }
        }
    } else if to_right {
        /* look for a new index to the right */

        *newpoint = SUNTRUE;

        *index = ilast;
        loop {
            if sign * (t - dt_t(&ca_mem, *index)) > ZERO {
                *index += 1;
            } else {
                break;
            }
        }

        ca_mem.borrow_mut().ca_ilast = *index;
    } else {
        /* ilast is still OK */

        *index = ilast;
    }

    CV_SUCCESS
}

/*
 * CVodeGetAdjY
 *
 * This routine returns the interpolated forward solution at time t.
 * The user must allocate space for y.
 */

pub fn CVodeGetAdjY(cvode_mem: &CVodeMem, t: sunrealtype, y: &N_Vector) -> i32 {
    /* NULL-mem check: handled by type system */
    let cv_mem = cvode_mem;

    let ca_mem = ca_mem_of(cv_mem);

    let imget = ca_mem.borrow().ca_IMget.expect("ca_IMget");

    /* C passes NULL for yS; the empty slice is the contract's NULL mapping */
    imget(cv_mem, t, y, &[])
}

/*
 * -----------------------------------------------------------------
 * Functions specific to cubic Hermite interpolation
 * -----------------------------------------------------------------
 */

/*
 * CVAhermiteMalloc
 *
 * This routine allocates memory for storing information at all
 * intermediate points between two consecutive check points.
 * This data is then used to interpolate the forward solution
 * at any other time.
 */

fn CVAhermiteMalloc(cv_mem: &CVodeMem) -> sunbooleantype {
    let mut allocOK = SUNTRUE;

    let ca_mem = ca_mem_of(cv_mem);

    let (tempv, Ns) = {
        let m = cv_mem.borrow();
        (m.cv_tempv.clone().expect("cv_tempv"), m.cv_Ns)
    };

    /* Allocate space for the vectors ytmp and yStmp */

    match N_VClone(&tempv) {
        None => return SUNFALSE,
        Some(v) => ca_mem.borrow_mut().ca_ytmp = Some(v),
    }

    let IMstoreSensi = ca_mem.borrow().ca_IMstoreSensi;

    if IMstoreSensi {
        match N_VCloneVectorArray(Ns, &tempv) {
            None => {
                ca_mem.borrow_mut().ca_ytmp = None;
                return SUNFALSE;
            }
            Some(vs) => ca_mem.borrow_mut().ca_yStmp = vs,
        }
    }

    /* Allocate space for the content field of the dt structures */

    let nsteps = ca_mem.borrow().ca_nsteps;
    let mut ii: i64 = 0;

    for i in 0..=nsteps {
        let y = match N_VClone(&tempv) {
            None => {
                ii = i;
                allocOK = SUNFALSE;
                break;
            }
            Some(v) => v,
        };

        let yd = match N_VClone(&tempv) {
            None => {
                ii = i;
                allocOK = SUNFALSE;
                break;
            }
            Some(v) => v,
        };

        let mut yS: Vec<N_Vector> = Vec::new();
        let mut ySd: Vec<N_Vector> = Vec::new();

        if IMstoreSensi {
            yS = match N_VCloneVectorArray(Ns, &tempv) {
                None => {
                    ii = i;
                    allocOK = SUNFALSE;
                    break;
                }
                Some(vs) => vs,
            };

            ySd = match N_VCloneVectorArray(Ns, &tempv) {
                None => {
                    ii = i;
                    allocOK = SUNFALSE;
                    break;
                }
                Some(vs) => vs,
            };
        }

        let content = CVhermiteDataMemRec {
            y: Some(y),
            yd: Some(yd),
            yS,
            ySd,
        };

        let d = dt_pnt(&ca_mem, i);
        d.borrow_mut().content = Some(Box::new(content));
    }

    /* If an error occurred, deallocate and return */

    if !allocOK {
        ca_mem.borrow_mut().ca_ytmp = None;

        if IMstoreSensi {
            ca_mem.borrow_mut().ca_yStmp = Vec::new();
        }

        for i in 0..ii {
            let d = dt_pnt(&ca_mem, i);
            d.borrow_mut().content = None;
        }
    }

    allocOK
}

/*
 * CVAhermiteFree
 *
 * This routine frees the memory allocated for data storage.
 */

fn CVAhermiteFree(cv_mem: &CVodeMem) {
    let ca_mem = ca_mem_of(cv_mem);

    ca_mem.borrow_mut().ca_ytmp = None;

    let IMstoreSensi = ca_mem.borrow().ca_IMstoreSensi;
    if IMstoreSensi {
        ca_mem.borrow_mut().ca_yStmp = Vec::new();
    }

    let nsteps = ca_mem.borrow().ca_nsteps;

    for i in 0..=nsteps {
        let d = dt_pnt(&ca_mem, i);
        d.borrow_mut().content = None;
    }
}

/*
 * CVAhermiteStorePnt ( -> IMstore )
 *
 * This routine stores a new point (y,yd) in the structure d for use
 * in the cubic Hermite interpolation.
 * Note that the time is already stored.
 */

fn CVAhermiteStorePnt(cv_mem: &CVodeMem, d: &CVdtpntMem) -> i32 {
    let ca_mem = ca_mem_of(cv_mem);

    let IMstoreSensi = ca_mem.borrow().ca_IMstoreSensi;

    let (y, yd, yS, ySd) = {
        let db = d.borrow();
        let content = db
            .content
            .as_ref()
            .expect("dt_mem content")
            .downcast_ref::<CVhermiteDataMemRec>()
            .expect("Hermite content");
        (
            content.y.clone().expect("content->y"),
            content.yd.clone().expect("content->yd"),
            content.yS.clone(),
            content.ySd.clone(),
        )
    };

    /* Load solution */

    let zn0 = cv_mem.borrow().cv_zn[0].clone().expect("cv_zn[0]");
    N_VScale(ONE, &zn0, &y);

    if IMstoreSensi {
        let Ns = cv_mem.borrow().cv_Ns;
        let cvals = vec![ONE; Ns as usize];

        let znS0 = cv_mem.borrow().cv_znS[0].clone();
        let retval = N_VScaleVectorArray(Ns, &cvals, &znS0, &yS);
        if retval != CV_SUCCESS {
            return CV_VECTOROP_ERR;
        }
    }

    /* Load derivative */

    let nst = cv_mem.borrow().cv_nst;

    if nst == 0 {
        let f = cv_mem.borrow().cv_f.expect("cv_f");
        let tn = cv_mem.borrow().cv_tn;

        /* retval = */
        let mut user_data = cv_mem.borrow_mut().cv_user_data.take();
        let _ = f(tn, &y, &yd, &mut user_data);
        cv_mem.borrow_mut().cv_user_data = user_data;

        if IMstoreSensi {
            let (tempv, ftemp) = {
                let m = cv_mem.borrow();
                (
                    m.cv_tempv.clone().expect("cv_tempv"),
                    m.cv_ftemp.clone().expect("cv_ftemp"),
                )
            };

            /* retval = */
            let _ = cvSensRhsWrapper(cv_mem, tn, &y, &yd, &yS, &ySd, &tempv, &ftemp);
        }
    } else {
        let h = cv_mem.borrow().cv_h;
        let zn1 = cv_mem.borrow().cv_zn[1].clone().expect("cv_zn[1]");
        N_VScale(ONE / h, &zn1, &yd);

        if IMstoreSensi {
            let Ns = cv_mem.borrow().cv_Ns;
            let cvals = vec![ONE / h; Ns as usize];

            let znS1 = cv_mem.borrow().cv_znS[1].clone();
            let retval = N_VScaleVectorArray(Ns, &cvals, &znS1, &ySd);
            if retval != CV_SUCCESS {
                return CV_VECTOROP_ERR;
            }
        }
    }

    0
}

/*
 * CVAhermiteGetY ( -> IMget )
 *
 * This routine uses cubic piece-wise Hermite interpolation for
 * the forward solution vector.
 * It is typically called by the wrapper routines before calling
 * user provided routines (fB, djacB, bjacB, jtimesB, psolB) but
 * can be directly called by the user through CVodeGetAdjY
 */

fn CVAhermiteGetY(cv_mem: &CVodeMem, t: sunrealtype, y: &N_Vector, yS: &[N_Vector]) -> i32 {
    let ca_mem = ca_mem_of(cv_mem);

    /* Local value of Ns */

    let NS: i32 = if ca_mem.borrow().ca_IMinterpSensi && !yS.is_empty() {
        cv_mem.borrow().cv_Ns
    } else {
        0
    };

    /* Get the index in dt_mem */

    let mut index: i64 = 0;
    let mut newpoint: sunbooleantype = SUNFALSE;
    let flag = CVAfindIndex(cv_mem, t, &mut index, &mut newpoint);
    if flag != CV_SUCCESS {
        return flag;
    }

    /* If we are beyond the left limit but close enough,
    then return y at the left limit. */

    if index == 0 {
        let (c0y, _c0yd, c0yS, _c0ySd) = herm_content(&ca_mem, 0);
        N_VScale(ONE, &c0y, y);

        if NS > 0 {
            let cvals = vec![ONE; NS as usize];

            let retval = N_VScaleVectorArray(NS, &cvals, &c0yS, yS);
            if retval != CV_SUCCESS {
                return CV_VECTOROP_ERR;
            }
        }

        return CV_SUCCESS;
    }

    /* Extract stuff from the appropriate data points */

    let t0 = dt_t(&ca_mem, index - 1);
    let t1 = dt_t(&ca_mem, index);
    let delta = t1 - t0;

    let (y0, yd0, yS0, ySd0) = herm_content(&ca_mem, index - 1);

    if newpoint {
        /* Recompute Y0 and Y1 */

        let (y1, yd1, yS1, ySd1) = herm_content(&ca_mem, index);

        /* Y1 = delta (yd1 + yd0) - 2 (y1 - y0) */
        let cvals: [sunrealtype; 4] = [-TWO, TWO, delta, delta];
        let Xvecs: [N_Vector; 4] = [y1.clone(), y0.clone(), yd1.clone(), yd0.clone()];

        let caY1 = ca_mem.borrow().ca_Y[1].clone().expect("ca_Y[1]");
        let retval = N_VLinearCombination(4, &cvals, &Xvecs, &caY1);
        if retval != CV_SUCCESS {
            return CV_VECTOROP_ERR;
        }

        /* Y0 = y1 - y0 - delta * yd0 */
        let cvals: [sunrealtype; 3] = [ONE, -ONE, -delta];
        let Xvecs: [N_Vector; 3] = [y1.clone(), y0.clone(), yd0.clone()];

        let caY0 = ca_mem.borrow().ca_Y[0].clone().expect("ca_Y[0]");
        let retval = N_VLinearCombination(3, &cvals, &Xvecs, &caY0);
        if retval != CV_SUCCESS {
            return CV_VECTOROP_ERR;
        }

        /* Recompute YS0 and YS1, if needed */

        if NS > 0 {
            /* YS1 = delta (ySd1 + ySd0) - 2 (yS1 - yS0) */
            let cvals: [sunrealtype; 4] = [-TWO, TWO, delta, delta];
            let XXvecs: [Vec<N_Vector>; 4] = [yS1.clone(), yS0.clone(), ySd1.clone(), ySd0.clone()];

            let caYS1 = ca_mem.borrow().ca_YS[1].clone();
            let retval = N_VLinearCombinationVectorArray(NS, 4, &cvals, &XXvecs, &caYS1);
            if retval != CV_SUCCESS {
                return CV_VECTOROP_ERR;
            }

            /* YS0 = yS1 - yS0 - delta * ySd0 */
            let cvals: [sunrealtype; 3] = [ONE, -ONE, -delta];
            let XXvecs: [Vec<N_Vector>; 3] = [yS1.clone(), yS0.clone(), ySd0.clone()];

            let caYS0 = ca_mem.borrow().ca_YS[0].clone();
            let retval = N_VLinearCombinationVectorArray(NS, 3, &cvals, &XXvecs, &caYS0);
            if retval != CV_SUCCESS {
                return CV_VECTOROP_ERR;
            }
        }
    }

    /* Perform the actual interpolation. */

    let factor1 = t - t0;

    let mut factor2 = factor1 / delta;
    factor2 = factor2 * factor2;

    let factor3 = factor2 * (t - t1) / delta;

    let cvals: [sunrealtype; 4] = [ONE, factor1, factor2, factor3];

    /* y = y0 + factor1 yd0 + factor2 * Y[0] + factor3 Y[1] */
    let (caY0, caY1) = {
        let ca = ca_mem.borrow();
        (
            ca.ca_Y[0].clone().expect("ca_Y[0]"),
            ca.ca_Y[1].clone().expect("ca_Y[1]"),
        )
    };
    let Xvecs: [N_Vector; 4] = [y0.clone(), yd0.clone(), caY0, caY1];

    let retval = N_VLinearCombination(4, &cvals, &Xvecs, y);
    if retval != CV_SUCCESS {
        return CV_VECTOROP_ERR;
    }

    /* yS = yS0 + factor1 ySd0 + factor2 * YS[0] + factor3 YS[1], if needed */
    if NS > 0 {
        let (caYS0, caYS1) = {
            let ca = ca_mem.borrow();
            (ca.ca_YS[0].clone(), ca.ca_YS[1].clone())
        };
        let XXvecs: [Vec<N_Vector>; 4] = [yS0.clone(), ySd0.clone(), caYS0, caYS1];

        let retval = N_VLinearCombinationVectorArray(NS, 4, &cvals, &XXvecs, yS);
        if retval != CV_SUCCESS {
            return CV_VECTOROP_ERR;
        }
    }

    CV_SUCCESS
}

/*
 * -----------------------------------------------------------------
 * Functions specific to Polynomial interpolation
 * -----------------------------------------------------------------
 */

/*
 * CVApolynomialMalloc
 *
 * This routine allocates memory for storing information at all
 * intermediate points between two consecutive check points.
 * This data is then used to interpolate the forward solution
 * at any other time.
 */

fn CVApolynomialMalloc(cv_mem: &CVodeMem) -> sunbooleantype {
    let mut allocOK = SUNTRUE;

    let ca_mem = ca_mem_of(cv_mem);

    let (tempv, Ns) = {
        let m = cv_mem.borrow();
        (m.cv_tempv.clone().expect("cv_tempv"), m.cv_Ns)
    };

    /* Allocate space for the vectors ytmp and yStmp */

    match N_VClone(&tempv) {
        None => return SUNFALSE,
        Some(v) => ca_mem.borrow_mut().ca_ytmp = Some(v),
    }

    let IMstoreSensi = ca_mem.borrow().ca_IMstoreSensi;

    if IMstoreSensi {
        match N_VCloneVectorArray(Ns, &tempv) {
            None => {
                ca_mem.borrow_mut().ca_ytmp = None;
                return SUNFALSE;
            }
            Some(vs) => ca_mem.borrow_mut().ca_yStmp = vs,
        }
    }

    /* Allocate space for the content field of the dt structures */

    let nsteps = ca_mem.borrow().ca_nsteps;
    let mut ii: i64 = 0;

    for i in 0..=nsteps {
        let y = match N_VClone(&tempv) {
            None => {
                ii = i;
                allocOK = SUNFALSE;
                break;
            }
            Some(v) => v,
        };

        let mut yS: Vec<N_Vector> = Vec::new();

        if IMstoreSensi {
            yS = match N_VCloneVectorArray(Ns, &tempv) {
                None => {
                    ii = i;
                    allocOK = SUNFALSE;
                    break;
                }
                Some(vs) => vs,
            };
        }

        let content = CVpolynomialDataMemRec {
            y: Some(y),
            yS,
            order: 0,
        };

        let d = dt_pnt(&ca_mem, i);
        d.borrow_mut().content = Some(Box::new(content));
    }

    /* If an error occurred, deallocate and return */

    if !allocOK {
        ca_mem.borrow_mut().ca_ytmp = None;

        if IMstoreSensi {
            ca_mem.borrow_mut().ca_yStmp = Vec::new();
        }

        for i in 0..ii {
            let d = dt_pnt(&ca_mem, i);
            d.borrow_mut().content = None;
        }
    }

    allocOK
}

/*
 * CVApolynomialFree
 *
 * This routine frees the memory allocated for data storage.
 */

fn CVApolynomialFree(cv_mem: &CVodeMem) {
    let ca_mem = ca_mem_of(cv_mem);

    ca_mem.borrow_mut().ca_ytmp = None;

    let IMstoreSensi = ca_mem.borrow().ca_IMstoreSensi;
    if IMstoreSensi {
        ca_mem.borrow_mut().ca_yStmp = Vec::new();
    }

    let nsteps = ca_mem.borrow().ca_nsteps;

    for i in 0..=nsteps {
        let d = dt_pnt(&ca_mem, i);
        d.borrow_mut().content = None;
    }
}

/*
 * CVApolynomialStorePnt ( -> IMstore )
 *
 * This routine stores a new point y in the structure d for use
 * in the Polynomial interpolation.
 * Note that the time is already stored.
 */

fn CVApolynomialStorePnt(cv_mem: &CVodeMem, d: &CVdtpntMem) -> i32 {
    let ca_mem = ca_mem_of(cv_mem);

    let IMstoreSensi = ca_mem.borrow().ca_IMstoreSensi;

    let (y, yS) = {
        let db = d.borrow();
        let content = db
            .content
            .as_ref()
            .expect("dt_mem content")
            .downcast_ref::<CVpolynomialDataMemRec>()
            .expect("polynomial content");
        (content.y.clone().expect("content->y"), content.yS.clone())
    };

    let zn0 = cv_mem.borrow().cv_zn[0].clone().expect("cv_zn[0]");
    N_VScale(ONE, &zn0, &y);

    if IMstoreSensi {
        let Ns = cv_mem.borrow().cv_Ns;
        let cvals = vec![ONE; Ns as usize];

        let znS0 = cv_mem.borrow().cv_znS[0].clone();
        let retval = N_VScaleVectorArray(Ns, &cvals, &znS0, &yS);
        if retval != CV_SUCCESS {
            return CV_VECTOROP_ERR;
        }
    }

    let qu = cv_mem.borrow().cv_qu;
    {
        let mut db = d.borrow_mut();
        let content = db
            .content
            .as_mut()
            .expect("dt_mem content")
            .downcast_mut::<CVpolynomialDataMemRec>()
            .expect("polynomial content");
        content.order = qu;
    }

    0
}

/*
 * CVApolynomialGetY ( -> IMget )
 *
 * This routine uses polynomial interpolation for the forward solution vector.
 * It is typically called by the wrapper routines before calling
 * user provided routines (fB, djacB, bjacB, jtimesB, psolB)) but
 * can be directly called by the user through CVodeGetAdjY.
 */

fn CVApolynomialGetY(cv_mem: &CVodeMem, t: sunrealtype, y: &N_Vector, yS: &[N_Vector]) -> i32 {
    let ca_mem = ca_mem_of(cv_mem);

    /* Local value of Ns */

    let NS: i32 = if ca_mem.borrow().ca_IMinterpSensi && !yS.is_empty() {
        cv_mem.borrow().cv_Ns
    } else {
        0
    };

    /* Get the index in dt_mem */

    let mut index: i64 = 0;
    let mut newpoint: sunbooleantype = SUNFALSE;
    let flag = CVAfindIndex(cv_mem, t, &mut index, &mut newpoint);
    if flag != CV_SUCCESS {
        return flag;
    }

    /* If we are beyond the left limit but close enough,
    then return y at the left limit. */

    if index == 0 {
        let (cy, cyS, _order) = poly_content(&ca_mem, 0);
        N_VScale(ONE, &cy, y);

        if NS > 0 {
            let cvals = vec![ONE; NS as usize];
            let retval = N_VScaleVectorArray(NS, &cvals, &cyS, yS);
            if retval != CV_SUCCESS {
                return CV_VECTOROP_ERR;
            }
        }

        return CV_SUCCESS;
    }

    /* Scaling factor */

    let dt = SUNRabs(dt_t(&ca_mem, index) - dt_t(&ca_mem, index - 1));

    /* Find the direction of the forward integration */

    let (ca_tinitial, ca_tfinal) = {
        let ca = ca_mem.borrow();
        (ca.ca_tinitial, ca.ca_tfinal)
    };
    let dir: i32 = if ca_tfinal - ca_tinitial > ZERO {
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
        let (_cy, _cyS, o) = poly_content(&ca_mem, b);
        order = o;
        if index < order as i64 {
            b += order as i64 - index;
        }
        base = b;
    } else {
        let mut b = index - 1;
        let (_cy, _cyS, o) = poly_content(&ca_mem, b);
        order = o;
        let ca_np = ca_mem.borrow().ca_np;
        if ca_np - index > order as i64 {
            b -= index + order as i64 - ca_np;
        }
        base = b;
    }

    /* Recompute Y (divided differences for Newton polynomial) if needed */

    if newpoint {
        /* Store 0-th order DD */
        if dir == 1 {
            for j in 0..=order as i64 {
                let tj = dt_t(&ca_mem, base - j);
                ca_mem.borrow_mut().ca_T[j as usize] = tj;

                let (cy, cyS, _o) = poly_content(&ca_mem, base - j);

                let caYj = ca_mem.borrow().ca_Y[j as usize].clone().expect("ca_Y[j]");
                N_VScale(ONE, &cy, &caYj);

                if NS > 0 {
                    let cvals = vec![ONE; NS as usize];
                    let caYSj = ca_mem.borrow().ca_YS[j as usize].clone();
                    let retval = N_VScaleVectorArray(NS, &cvals, &cyS, &caYSj);
                    if retval != CV_SUCCESS {
                        return CV_VECTOROP_ERR;
                    }
                }
            }
        } else {
            for j in 0..=order as i64 {
                let tj = dt_t(&ca_mem, base - 1 + j);
                ca_mem.borrow_mut().ca_T[j as usize] = tj;

                let (cy, cyS, _o) = poly_content(&ca_mem, base - 1 + j);

                let caYj = ca_mem.borrow().ca_Y[j as usize].clone().expect("ca_Y[j]");
                N_VScale(ONE, &cy, &caYj);

                if NS > 0 {
                    let cvals = vec![ONE; NS as usize];
                    let caYSj = ca_mem.borrow().ca_YS[j as usize].clone();
                    let retval = N_VScaleVectorArray(NS, &cvals, &cyS, &caYSj);
                    if retval != CV_SUCCESS {
                        return CV_VECTOROP_ERR;
                    }
                }
            }
        }

        /* Compute higher-order DD */
        for i in 1..=order {
            let mut j = order;
            while j >= i {
                let (Tj, Tji) = {
                    let ca = ca_mem.borrow();
                    (ca.ca_T[j as usize], ca.ca_T[(j - i) as usize])
                };
                let factor = dt / (Tj - Tji);

                let (caYj, caYjm1) = {
                    let ca = ca_mem.borrow();
                    (
                        ca.ca_Y[j as usize].clone().expect("ca_Y[j]"),
                        ca.ca_Y[(j - 1) as usize].clone().expect("ca_Y[j-1]"),
                    )
                };
                N_VLinearSum(factor, &caYj, -factor, &caYjm1, &caYj);

                if NS > 0 {
                    /* C passes ca_YS[j] as both X and Z (the same N_Vector*),
                    so the aliased path is taken here as well. */
                    let (caYSj, caYSjm1) = {
                        let ca = ca_mem.borrow();
                        (
                            ca.ca_YS[j as usize].clone(),
                            ca.ca_YS[(j - 1) as usize].clone(),
                        )
                    };
                    let retval =
                        N_VLinearSumVectorArray(NS, factor, &caYSj, -factor, &caYSjm1, &caYSj);
                    if retval != CV_SUCCESS {
                        return CV_VECTOROP_ERR;
                    }
                }

                j -= 1;
            }
        }
    }

    /* Perform the actual interpolation using nested multiplications */

    let mut cvals = vec![ZERO; (order + 1) as usize];
    cvals[0] = ONE;
    for i in 0..order as usize {
        let Ti = ca_mem.borrow().ca_T[i];
        cvals[i + 1] = cvals[i] * (t - Ti) / dt;
    }

    let caY: Vec<N_Vector> = {
        let ca = ca_mem.borrow();
        (0..=order as usize)
            .map(|j| ca.ca_Y[j].clone().expect("ca_Y[j]"))
            .collect()
    };
    let retval = N_VLinearCombination(order + 1, &cvals, &caY, y);
    if retval != CV_SUCCESS {
        return CV_VECTOROP_ERR;
    }

    if NS > 0 {
        let caYS: Vec<Vec<N_Vector>> = {
            let ca = ca_mem.borrow();
            (0..=order as usize).map(|j| ca.ca_YS[j].clone()).collect()
        };
        let retval = N_VLinearCombinationVectorArray(NS, order + 1, &cvals, &caYS, yS);
        if retval != CV_SUCCESS {
            return CV_VECTOROP_ERR;
        }
    }

    CV_SUCCESS
}

/*
 * =================================================================
 * WRAPPERS FOR ADJOINT SYSTEM
 * =================================================================
 */
/*
 * CVArhs
 *
 * This routine interfaces to the CVRhsFnB (or CVRhsFnBS) routine
 * provided by the user.
 */

fn CVArhs(
    t: sunrealtype,
    yB: &N_Vector,
    yBdot: &N_Vector,
    cvode_mem: &mut Option<Box<dyn Any>>,
) -> i32 {
    /* C: cv_mem = (CVodeMem)cvode_mem — the backward integrator's user_data
    IS the forward CVODES memory (set by CVodeCreateB). */
    let cv_mem: CVodeMem = cvode_mem
        .as_ref()
        .expect("CVArhs user_data")
        .downcast_ref::<CVodeMem>()
        .expect("CVArhs user_data is the forward CVodeMem")
        .clone();

    let ca_mem = ca_mem_of(&cv_mem);

    let cvB_mem = ca_mem.borrow().ca_bckpbCrt.clone().expect("ca_bckpbCrt");

    /* Get forward solution from interpolation */

    let (IMinterpSensi, ytmp, yStmp, imget) = {
        let ca = ca_mem.borrow();
        (
            ca.ca_IMinterpSensi,
            ca.ca_ytmp.clone().expect("ca_ytmp"),
            ca.ca_yStmp.clone(),
            ca.ca_IMget.expect("ca_IMget"),
        )
    };

    let flag = if IMinterpSensi {
        imget(&cv_mem, t, &ytmp, &yStmp)
    } else {
        imget(&cv_mem, t, &ytmp, &[])
    };

    if flag != CV_SUCCESS {
        cvProcessError(
            Some(&cv_mem),
            -1,
            line!() as i32,
            "CVArhs",
            file!(),
            &MSGCV_BAD_TINTERP(t),
        );
        return -1;
    }

    /* Call the user's RHS function */

    let f_withSensi = cvB_mem.borrow().cv_f_withSensi;

    if f_withSensi {
        let fs = cvB_mem.borrow().cv_fs.expect("cvB_mem->cv_fs");
        let mut user_dataB = cvB_mem.borrow_mut().cv_user_data.take();
        let retval = fs(t, &ytmp, &yStmp, yB, yBdot, &mut user_dataB);
        cvB_mem.borrow_mut().cv_user_data = user_dataB;
        retval
    } else {
        let f = cvB_mem.borrow().cv_f.expect("cvB_mem->cv_f");
        let mut user_dataB = cvB_mem.borrow_mut().cv_user_data.take();
        let retval = f(t, &ytmp, yB, yBdot, &mut user_dataB);
        cvB_mem.borrow_mut().cv_user_data = user_dataB;
        retval
    }
}

/*
 * CVArhsQ
 *
 * This routine interfaces to the CVQuadRhsFnB (or CVQuadRhsFnBS) routine
 * provided by the user.
 */

fn CVArhsQ(
    t: sunrealtype,
    yB: &N_Vector,
    qBdot: &N_Vector,
    cvode_mem: &mut Option<Box<dyn Any>>,
) -> i32 {
    let cv_mem: CVodeMem = cvode_mem
        .as_ref()
        .expect("CVArhsQ user_data")
        .downcast_ref::<CVodeMem>()
        .expect("CVArhsQ user_data is the forward CVodeMem")
        .clone();

    let ca_mem = ca_mem_of(&cv_mem);

    let cvB_mem = ca_mem.borrow().ca_bckpbCrt.clone().expect("ca_bckpbCrt");

    /* Get forward solution from interpolation */

    let (IMinterpSensi, ytmp, yStmp, imget) = {
        let ca = ca_mem.borrow();
        (
            ca.ca_IMinterpSensi,
            ca.ca_ytmp.clone().expect("ca_ytmp"),
            ca.ca_yStmp.clone(),
            ca.ca_IMget.expect("ca_IMget"),
        )
    };

    if IMinterpSensi {
        /* flag = */
        let _ = imget(&cv_mem, t, &ytmp, &yStmp);
    } else {
        /* flag = */
        let _ = imget(&cv_mem, t, &ytmp, &[]);
    }

    /* Call the user's RHS function */

    let fQ_withSensi = cvB_mem.borrow().cv_fQ_withSensi;

    if fQ_withSensi {
        let fQs = cvB_mem.borrow().cv_fQs.expect("cvB_mem->cv_fQs");
        let mut user_dataB = cvB_mem.borrow_mut().cv_user_data.take();
        let retval = fQs(t, &ytmp, &yStmp, yB, qBdot, &mut user_dataB);
        cvB_mem.borrow_mut().cv_user_data = user_dataB;
        retval
    } else {
        let fQ = cvB_mem.borrow().cv_fQ.expect("cvB_mem->cv_fQ");
        let mut user_dataB = cvB_mem.borrow_mut().cv_user_data.take();
        let retval = fQ(t, &ytmp, yB, qBdot, &mut user_dataB);
        cvB_mem.borrow_mut().cv_user_data = user_dataB;
        retval
    }
}
