//! Port of `src/kinsol/kinsol.c` — the main KINSOL solver (independent of
//! the linear solver in use).
//!
//! EXPORTED FUNCTIONS: [`KINCreate`], [`KINInit`], [`KINSol`], [`KINFree`].
//!
//! PRIVATE FUNCTIONS: `KINCheckNvector`, `KINAllocVectors`, `KINFreeVectors`,
//! `KINSolInit`, `KINLinSolDrv`, `KINFullNewton`, `KINLineSearch`,
//! `KINConstraint`, `KINFP`, `KINPicardAA`, `KINPicardFcnEval`, `KINStop`,
//! `KINForcingTerm`, `KINScFNorm`, `KINScSNorm`, `AndersonAcc`,
//! `AndersonAccQRDelete`.
//!
//! `KINPrintInfo` and `KINProcessError` (defined at the bottom of `kinsol.c`
//! upstream) live in [`crate::kinsol_impl`] so every kinsol module shares one
//! definition; they are NOT redefined here.
//!
//! ## Build configuration assumed by this translation
//!
//! `SUNDIALS_LOGGING_LEVEL = 2` (= `SUNDIALS_LOGGING_WARNING`), profiling
//! OFF, error checks OFF, serial only.
//!
//! **Every** `KINPrintInfo` call site in `kinsol.c` sits inside
//! `#if SUNDIALS_LOGGING_LEVEL >= SUNDIALS_LOGGING_INFO` (= 3), so at the
//! reference logging level the calls do not exist and are omitted here — the
//! omitted call is marked with a `/* KINPrintInfo(...) — omitted */` comment
//! at each site so the mapping stays auditable. (SUNDIALS 7.x removed
//! `KINSetPrintLevel`; the informational output is a pure logger feature and
//! no serial reference `.out` contains any of it. The `nni = ...` lines in
//! e.g. `kinFerTron_dns.out` are the examples' own `printf` statistics.)
//! `SUNLogExtraDebugVec` (three sites in `KINFP`), `SUNDIALS_MARK_FUNCTION_*`
//! and `SUNDIALS_MAYBE_UNUSED` compile away likewise.
//!
//! ## Borrow discipline
//!
//! `KINMem = Rc<RefCell<KINMemRec>>`. No `borrow`/`borrow_mut` guard is ever
//! held across a user callback, an `N_Vector`/matrix op, a linear-solver hook
//! or another borrow of the same mem. Vector handles are `Rc` clones (exactly
//! C's pointer copies), so `kin_mem->kin_uu = u` aliases the user's vector
//! natively and needs no copy-back.

use std::cell::RefCell;
use std::rc::Rc;

use crate::kinsol_aa::{KINFreeAA, KINInitAA};
use crate::kinsol_impl::*;
use crate::kinsol_orth::{kinQRAdd, KINFreeOrth, KINInitOrth};
use sundials_core::sundials_context::SUNContext;
use sundials_core::sundials_math::*;
use sundials_core::sundials_nvector::*;
use sundials_core::sundials_types::*;

/*
 * =================================================================
 * KINSOL PRIVATE CONSTANTS
 * =================================================================
 */

const HALF: sunrealtype = 0.5;
const ZERO: sunrealtype = 0.0;
const ONE: sunrealtype = 1.0;
const ONEPT5: sunrealtype = 1.5;
const TWO: sunrealtype = 2.0;
const THREE: sunrealtype = 3.0;
const FIVE: sunrealtype = 5.0;
const TWELVE: sunrealtype = 12.0;
const POINT1: sunrealtype = 0.1;
const POINT01: sunrealtype = 0.01;
const POINT99: sunrealtype = 0.99;
const THOUSAND: sunrealtype = 1000.0;
const ONETHIRD: sunrealtype = 0.3333333333333333;
const TWOTHIRDS: sunrealtype = 0.6666666666666667;
const POINT9: sunrealtype = 0.9;
const POINT0001: sunrealtype = 0.0001;

/*
 * =================================================================
 * KINSOL ROUTINE-SPECIFIC CONSTANTS
 * =================================================================
 */

/*
 * Control constants for lower-level functions used by KINSol
 * ----------------------------------------------------------
 *
 * KINStop return value requesting more iterations
 *    RETRY_ITERATION
 *    CONTINUE_ITERATIONS
 *
 * KINFullNewton, KINLineSearch, KINFP, and KINPicardAA return values:
 *    KIN_SUCCESS
 *    KIN_SYSFUNC_FAIL
 *    STEP_TOO_SMALL
 *
 * KINConstraint return values:
 *    KIN_SUCCESS
 *    CONSTR_VIOLATED
 */

const RETRY_ITERATION: i32 = -998;
const CONTINUE_ITERATIONS: i32 = -999;
const STEP_TOO_SMALL: i32 = -997;
const CONSTR_VIOLATED: i32 = -996;

/*
 * Algorithmic constants
 * ---------------------
 *
 * MAX_RECVR   max. no. of attempts to correct a recoverable func error
 */

const MAX_RECVR: i32 = 5;

/* Keys for KINPrintInfo (PRNT_RETVAL .. PRNT_OTHER) live in
 * `kinsol_impl.rs` together with `KINPrintInfo` itself. */

/*
 * =================================================================
 * Vector-field accessors and callback invocation helpers
 *
 * Each accessor borrows the mem, clones the `Rc` handle (= C's pointer
 * copy) and drops the borrow immediately, so no guard is ever live
 * across a vector op or callback. A `None` field is C's NULL pointer
 * dereference (UB) and maps to a deterministic panic.
 * =================================================================
 */

#[inline]
fn get_uu(kin_mem: &KINMem) -> N_Vector {
    kin_mem.borrow().kin_uu.clone().expect("kin_uu")
}

#[inline]
fn get_unew(kin_mem: &KINMem) -> N_Vector {
    kin_mem.borrow().kin_unew.clone().expect("kin_unew")
}

#[inline]
fn get_fval(kin_mem: &KINMem) -> N_Vector {
    kin_mem.borrow().kin_fval.clone().expect("kin_fval")
}

#[inline]
fn get_gval(kin_mem: &KINMem) -> N_Vector {
    kin_mem.borrow().kin_gval.clone().expect("kin_gval")
}

#[inline]
fn get_uscale(kin_mem: &KINMem) -> N_Vector {
    kin_mem.borrow().kin_uscale.clone().expect("kin_uscale")
}

#[inline]
fn get_fscale(kin_mem: &KINMem) -> N_Vector {
    kin_mem.borrow().kin_fscale.clone().expect("kin_fscale")
}

#[inline]
fn get_pp(kin_mem: &KINMem) -> N_Vector {
    kin_mem.borrow().kin_pp.clone().expect("kin_pp")
}

#[inline]
fn get_constraints(kin_mem: &KINMem) -> N_Vector {
    kin_mem
        .borrow()
        .kin_constraints
        .clone()
        .expect("kin_constraints")
}

#[inline]
fn get_vtemp1(kin_mem: &KINMem) -> N_Vector {
    kin_mem.borrow().kin_vtemp1.clone().expect("kin_vtemp1")
}

#[inline]
fn get_vtemp2(kin_mem: &KINMem) -> N_Vector {
    kin_mem.borrow().kin_vtemp2.clone().expect("kin_vtemp2")
}

#[inline]
fn get_fold_aa(kin_mem: &KINMem) -> N_Vector {
    kin_mem.borrow().kin_fold_aa.clone().expect("kin_fold_aa")
}

#[inline]
fn get_gold_aa(kin_mem: &KINMem) -> N_Vector {
    kin_mem.borrow().kin_gold_aa.clone().expect("kin_gold_aa")
}

#[inline]
fn get_df_aa(kin_mem: &KINMem, i: usize) -> N_Vector {
    kin_mem.borrow().kin_df_aa[i].clone()
}

#[inline]
fn get_dg_aa(kin_mem: &KINMem, i: usize) -> N_Vector {
    kin_mem.borrow().kin_dg_aa[i].clone()
}

/// Invoke the user system function
/// (C: `kin_mem->kin_func(uu, fval, kin_mem->kin_user_data)`).
///
/// The `user_data` box is taken out of the mem around the call and
/// restored on every path, so the callback may re-enter KINSOL freely.
fn kin_call_func(kin_mem: &KINMem, uu: &N_Vector, fval: &N_Vector) -> i32 {
    let func = kin_mem.borrow().kin_func.expect("kin_func set");
    let mut user_data = kin_mem.borrow_mut().kin_user_data.take();
    let retval = func(uu, fval, &mut user_data);
    kin_mem.borrow_mut().kin_user_data = user_data;
    retval
}

/// Invoke `kin_mem->kin_damping_fn(...)`.
///
/// C passes `&(kin_mem->kin_beta)` / `&(kin_mem->kin_beta_aa)` so the
/// callee writes the damping factor straight into the mem; the channel is
/// preserved here by seeding a local from the field, passing `&mut local`
/// and writing it back after the call. `use_beta_aa` selects which field
/// the pointer aliased.
fn kin_call_damping_fn(
    kin_mem: &KINMem,
    iter: i64,
    u_val: &N_Vector,
    g_val: &N_Vector,
    qt_fn_1d: Option<&mut [sunrealtype]>,
    depth: i64,
    use_beta_aa: bool,
) -> i32 {
    let damping_fn = kin_mem.borrow().kin_damping_fn.expect("kin_damping_fn set");
    let mut beta = if use_beta_aa {
        kin_mem.borrow().kin_beta_aa
    } else {
        kin_mem.borrow().kin_beta
    };
    let mut user_data = kin_mem.borrow_mut().kin_user_data.take();
    let retval = damping_fn(
        iter,
        u_val,
        g_val,
        qt_fn_1d,
        depth,
        &mut user_data,
        &mut beta,
    );
    kin_mem.borrow_mut().kin_user_data = user_data;
    {
        let mut m = kin_mem.borrow_mut();
        if use_beta_aa {
            m.kin_beta_aa = beta;
        } else {
            m.kin_beta = beta;
        }
    }
    retval
}

/*
 * =================================================================
 * EXPORTED FUNCTIONS IMPLEMENTATION
 * =================================================================
 */

/*
 * -----------------------------------------------------------------
 * Creation and allocation functions
 * -----------------------------------------------------------------
 */

/*
 * Function : KINCreate
 *
 * KINCreate creates an internal memory block for a problem to
 * be solved by KINSOL. If successful, KINCreate returns a pointer
 * to the problem memory. This pointer should be passed to
 * KINInit. If an initialization error occurs, KINCreate prints
 * an error message to standard error and returns NULL.
 */

pub fn KINCreate(sunctx: &SUNContext) -> Option<KINMem> {
    /* Test inputs */
    /* NULL sunctx check: handled by the type system */

    /* malloc failure branch: allocation cannot fail observably in Rust */

    /* Zero out kin_mem */
    let mut kin_mem = KINMemRec::zeroed(sunctx.clone());

    kin_mem.kin_sunctx = sunctx.clone();
    kin_mem.python = None;

    /* set uround (unit roundoff) */

    let uround: sunrealtype = SUN_UNIT_ROUNDOFF;
    kin_mem.kin_uround = uround;

    /* set default values for solver optional inputs */

    kin_mem.kin_func = None;
    kin_mem.kin_user_data = None;
    kin_mem.kin_uu = None;
    kin_mem.kin_unew = None;
    kin_mem.kin_fval = None;
    kin_mem.kin_gval = None;
    kin_mem.kin_uscale = None;
    kin_mem.kin_fscale = None;
    kin_mem.kin_pp = None;
    kin_mem.kin_constraints = None;
    kin_mem.kin_vtemp1 = None;
    kin_mem.kin_vtemp2 = None;
    kin_mem.kin_vtemp3 = None;
    kin_mem.kin_fold_aa = None;
    kin_mem.kin_gold_aa = None;
    kin_mem.kin_df_aa = Vec::new();
    kin_mem.kin_dg_aa = Vec::new();
    kin_mem.kin_q_aa = Vec::new();
    kin_mem.kin_T_aa = Vec::new();
    kin_mem.kin_gamma_aa = Vec::new();
    kin_mem.kin_R_aa = Vec::new();
    kin_mem.kin_cv = Vec::new();
    kin_mem.kin_Xv = Vec::new();
    kin_mem.kin_lmem = None;
    kin_mem.kin_beta = ONE;
    kin_mem.kin_damping = SUNFALSE;
    kin_mem.kin_m_aa = 0;
    kin_mem.kin_m_aa_alloc = 0;
    kin_mem.kin_delay_aa = 0;
    kin_mem.kin_current_depth = 0;
    kin_mem.kin_damping_fn = None;
    kin_mem.kin_depth_fn = None;
    kin_mem.kin_orth_aa = KIN_ORTH_MGS;
    kin_mem.kin_orth_aa_alloc = 0;
    kin_mem.kin_qr_func = None;
    kin_mem.kin_qr_data = None;
    kin_mem.kin_beta_aa = ONE;
    kin_mem.kin_damping_aa = SUNFALSE;
    kin_mem.kin_dot_prod_sb = SUNFALSE;
    kin_mem.kin_constraintsSet = SUNFALSE;
    kin_mem.kin_ret_newest = SUNFALSE;
    kin_mem.kin_mxiter = MXITER_DEFAULT;
    kin_mem.kin_noInitSetup = SUNFALSE;
    kin_mem.kin_msbset = MSBSET_DEFAULT;
    kin_mem.kin_noResMon = SUNFALSE;
    kin_mem.kin_msbset_sub = MSBSET_SUB_DEFAULT;
    kin_mem.kin_update_fnorm_sub = SUNFALSE;
    kin_mem.kin_mxnbcf = MXNBCF_DEFAULT;
    kin_mem.kin_sthrsh = TWO;
    kin_mem.kin_noMinEps = SUNFALSE;
    kin_mem.kin_mxnstepin = ZERO;
    kin_mem.kin_sqrt_relfunc = SUNRsqrt(uround);
    kin_mem.kin_scsteptol = SUNRpowerR(uround, TWOTHIRDS);
    kin_mem.kin_fnormtol = SUNRpowerR(uround, ONETHIRD);
    kin_mem.kin_etaflag = KIN_ETACHOICE1;
    kin_mem.kin_eta = POINT1; /* default for KIN_ETACONSTANT */
    kin_mem.kin_eta_alpha = TWO; /* default for KIN_ETACHOICE2  */
    kin_mem.kin_eta_gamma = POINT9; /* default for KIN_ETACHOICE2  */
    kin_mem.kin_MallocDone = SUNFALSE;
    kin_mem.kin_eval_omega = SUNTRUE;
    kin_mem.kin_omega = ZERO; /* default to using min/max    */
    kin_mem.kin_omega_min = OMEGA_MIN;
    kin_mem.kin_omega_max = OMEGA_MAX;

    /* initialize lrw and liw */

    kin_mem.kin_lrw = 17;
    kin_mem.kin_liw = 22;

    /* NOTE: needed since KINInit could be called after KINSetConstraints */

    kin_mem.kin_lrw1 = 0;
    kin_mem.kin_liw1 = 0;

    Some(Rc::new(RefCell::new(kin_mem)))
}

/*
 * Function : KINInit
 *
 * KINInit allocates memory for a problem or execution of KINSol.
 * If memory is successfully allocated, KIN_SUCCESS is returned.
 * Otherwise, an error message is printed and an error flag
 * returned.
 */

pub fn KINInit(kinmem: &KINMem, func: KINSysFn, tmpl: &N_Vector) -> i32 {
    /* check kinmem: NULL check handled by the type system */
    let kin_mem = kinmem;

    /* NULL func check: handled by the type system */

    /* check if all required vector operations are implemented */

    let nvectorOK = KINCheckNvector(tmpl);
    if !nvectorOK {
        KINProcessError(
            Some(kin_mem),
            KIN_ILL_INPUT,
            line!() as i32,
            "KINInit",
            file!(),
            MSG_BAD_NVECTOR,
        );
        return KIN_ILL_INPUT;
    }

    /* set space requirements for one N_Vector */

    let mut lrw1: sunindextype = 0;
    let mut liw1: sunindextype = 0;
    if tmpl.ops.borrow().nvspace.is_some() {
        N_VSpace(tmpl, &mut lrw1, &mut liw1);
        let mut m = kin_mem.borrow_mut();
        m.kin_lrw1 = lrw1;
        m.kin_liw1 = liw1;
    } else {
        let mut m = kin_mem.borrow_mut();
        m.kin_lrw1 = 0;
        m.kin_liw1 = 0;
    }

    /* allocate necessary vectors */

    let allocOK = KINAllocVectors(kin_mem, tmpl);
    if !allocOK {
        KINProcessError(
            Some(kin_mem),
            KIN_MEM_FAIL,
            line!() as i32,
            "KINInit",
            file!(),
            MSG_MEM_FAIL,
        );
        /* C additionally does `free(kin_mem)` here (leaving the caller's
        pointer dangling); the Rust handle is owned by the caller, so
        there is nothing to free — the record simply keeps
        kin_MallocDone == SUNFALSE. */
        return KIN_MEM_FAIL;
    }

    {
        let mut m = kin_mem.borrow_mut();

        /* copy the input parameter into KINSol state */

        m.kin_func = Some(func);

        /* set the linear solver addresses to NULL */

        m.kin_linit = None;
        m.kin_lsetup = None;
        m.kin_lsolve = None;
        m.kin_lfree = None;
        m.kin_lmem = None;

        /* problem memory has been successfully allocated */

        m.kin_MallocDone = SUNTRUE;
    }

    KIN_SUCCESS
}

/*
 * -----------------------------------------------------------------
 * Main solver function
 * -----------------------------------------------------------------
 */

/*
 * Function : KINSol
 *
 * KINSol (main KINSOL driver routine) manages the computational
 * process of computing an approximate solution of the nonlinear
 * system F(uu) = 0. The KINSol routine calls the following
 * subroutines:
 *
 *  KINSolInit    checks if initial guess satisfies user-supplied
 *                constraints and initializes linear solver
 *
 *  KINLinSolDrv  interfaces with linear solver to find a
 *                solution of the system J(uu)*x = b (calculate
 *                Newton step)
 *
 *  KINFullNewton/KINLineSearch  implement the global strategy
 *
 *  KINForcingTerm  computes the forcing term (eta)
 *
 *  KINStop  determines if an approximate solution has been found
 */

pub fn KINSol(
    kinmem: &KINMem,
    u: &N_Vector,
    strategy_in: i32,
    u_scale: &N_Vector,
    f_scale: &N_Vector,
) -> i32 {
    let mut ret: i32;
    let mut sflag: i32;

    /* initialize to avoid compiler warning messages */

    let mut maxStepTaken: sunbooleantype = SUNFALSE;
    let mut fnormp: sunrealtype = -ONE;
    let mut f1normp: sunrealtype = -ONE;

    /* initialize epsmin to avoid compiler warning message */

    let mut epsmin: sunrealtype = ZERO;

    /* check for kinmem non-NULL: handled by the type system */
    let kin_mem = kinmem;

    if kin_mem.borrow().kin_MallocDone == SUNFALSE {
        KINProcessError(
            Some(kin_mem),
            KIN_NO_MALLOC,
            line!() as i32,
            "KINSol",
            file!(),
            MSG_NO_MALLOC,
        );
        return KIN_NO_MALLOC;
    }

    /* load input arguments (N_Vector handles are Rc clones: the same
    aliasing C gets from the pointer assignment) */

    {
        let mut m = kin_mem.borrow_mut();
        m.kin_uu = Some(u.clone());
        m.kin_uscale = Some(u_scale.clone());
        m.kin_fscale = Some(f_scale.clone());
        m.kin_globalstrategy = strategy_in;
    }

    /* Setup Anderson acceleration for FP or Picard */

    let (globalstrategy, m_aa) = {
        let m = kin_mem.borrow();
        (m.kin_globalstrategy, m.kin_m_aa)
    };
    if (globalstrategy == KIN_FP || globalstrategy == KIN_PICARD) && m_aa != 0 {
        /* Initialize Anderson acceleration workspace */
        ret = KINInitAA(kin_mem);
        if ret != 0 {
            KINProcessError(
                Some(kin_mem),
                ret,
                line!() as i32,
                "KINSol",
                file!(),
                "Initializing Anderson acceleration failed",
            );
            return ret;
        }

        /* Initialize orthogonalization workspace */
        ret = KINInitOrth(kin_mem);
        if ret != 0 {
            KINProcessError(
                Some(kin_mem),
                ret,
                line!() as i32,
                "KINSol",
                file!(),
                "Initializing the orthogonalization method failed",
            );
            return ret;
        }
    }

    /* CSW:
       Call fixed point solver if requested.  Note that this should probably
       be forked off to a FPSOL solver instead of kinsol in the future. */
    if kin_mem.borrow().kin_globalstrategy == KIN_FP {
        if kin_mem.borrow().kin_uu.is_none() {
            KINProcessError(
                Some(kin_mem),
                KIN_ILL_INPUT,
                line!() as i32,
                "KINSol",
                file!(),
                MSG_UU_NULL,
            );
            return KIN_ILL_INPUT;
        }

        if kin_mem.borrow().kin_constraintsSet != SUNFALSE {
            KINProcessError(
                Some(kin_mem),
                KIN_ILL_INPUT,
                line!() as i32,
                "KINSol",
                file!(),
                MSG_CONSTRAINTS_NOTOK,
            );
            return KIN_ILL_INPUT;
        }

        /* KINPrintInfo(kin_mem, PRNT_TOL, "KINSOL", __func__, INFO_TOL,
        kin_scsteptol, kin_fnormtol) — omitted (logging level < INFO) */

        {
            let mut m = kin_mem.borrow_mut();
            m.kin_nfe = 0;
            m.kin_nnilset = 0;
            m.kin_nnilset_sub = 0;
            m.kin_nni = 0;
            m.kin_nbcf = 0;
            m.kin_nbktrk = 0;
        }
        ret = KINFP(kin_mem);

        match ret {
            KIN_SYSFUNC_FAIL => {
                KINProcessError(
                    Some(kin_mem),
                    KIN_SYSFUNC_FAIL,
                    line!() as i32,
                    "KINSol",
                    file!(),
                    MSG_SYSFUNC_FAILED,
                );
            }
            KIN_MAXITER_REACHED => {
                KINProcessError(
                    Some(kin_mem),
                    KIN_MAXITER_REACHED,
                    line!() as i32,
                    "KINSol",
                    file!(),
                    MSG_MAXITER_REACHED,
                );
            }
            _ => {}
        }

        return ret;
    }

    /* initialize solver */
    ret = KINSolInit(kin_mem);
    if ret != KIN_SUCCESS {
        return ret;
    }

    kin_mem.borrow_mut().kin_ncscmx = 0;

    /* Note: The following logic allows the choice of whether or not
       to force a call to the linear solver setup upon a given call to
       KINSol */

    {
        let mut m = kin_mem.borrow_mut();
        if m.kin_noInitSetup {
            m.kin_sthrsh = ONE;
        } else {
            m.kin_sthrsh = TWO;
        }
    }

    /* if eps is to be bounded from below, set the bound */

    {
        let m = kin_mem.borrow();
        if m.kin_inexact_ls && !(m.kin_noMinEps) {
            epsmin = POINT01 * m.kin_fnormtol;
        }
    }

    /* if omega is zero at this point, make sure it will be evaluated
       at each iteration based on the provided min/max bounds and the
       current function norm. */
    {
        let mut m = kin_mem.borrow_mut();
        if m.kin_omega == ZERO {
            m.kin_eval_omega = SUNTRUE;
        } else {
            m.kin_eval_omega = SUNFALSE;
        }
    }

    /* CSW:
       Call fixed point solver for Picard method if requested.
       Note that this should probably be forked off to a part of an
       FPSOL solver instead of kinsol in the future. */
    if kin_mem.borrow().kin_globalstrategy == KIN_PICARD {
        if kin_mem.borrow().kin_gval.is_none() {
            let unew = get_unew(kin_mem);
            let gval = N_VClone(&unew);
            if gval.is_none() {
                KINProcessError(
                    Some(kin_mem),
                    KIN_MEM_FAIL,
                    line!() as i32,
                    "KINSol",
                    file!(),
                    MSG_MEM_FAIL,
                );
                return KIN_MEM_FAIL;
            }
            let mut m = kin_mem.borrow_mut();
            m.kin_gval = gval;
            m.kin_liw += m.kin_liw1;
            m.kin_lrw += m.kin_lrw1;
        }
        ret = KINPicardAA(kin_mem);

        return ret;
    }

    'main_loop: loop {
        {
            let mut m = kin_mem.borrow_mut();
            m.kin_retry_nni = SUNFALSE;
            m.kin_nni += 1;
        }

        /* calculate the epsilon (stopping criteria for iterative linear solver)
           for this iteration based on eta from the routine KINForcingTerm */

        {
            let mut m = kin_mem.borrow_mut();
            if m.kin_inexact_ls {
                m.kin_eps = (m.kin_eta + m.kin_uround) * m.kin_fnorm;
                if !(m.kin_noMinEps) {
                    m.kin_eps = SUNMAX(epsmin, m.kin_eps);
                }
            }
        }

        /* `repeat_nni:` — `goto repeat_nni` is `continue 'repeat_nni`, and
        falling off the C label block is `break 'repeat_nni` */
        'repeat_nni: loop {
            /* call the appropriate routine to calculate an acceptable step pp */

            sflag = 0;

            let strategy = kin_mem.borrow().kin_globalstrategy;
            if strategy == KIN_NONE {
                /* Full Newton Step*/

                /* call KINLinSolDrv to calculate the (approximate) Newton step, pp */
                ret = KINLinSolDrv(kin_mem);
                if ret != KIN_SUCCESS {
                    break 'main_loop;
                }

                sflag = KINFullNewton(kin_mem, &mut fnormp, &mut f1normp, &mut maxStepTaken);

                /* if sysfunc failed unrecoverably, stop */
                if (sflag == KIN_SYSFUNC_FAIL) || (sflag == KIN_REPTD_SYSFUNC_ERR) {
                    ret = sflag;
                    break 'main_loop;
                }
            } else if strategy == KIN_LINESEARCH {
                /* Line Search */

                /* call KINLinSolDrv to calculate the (approximate) Newton step, pp */
                ret = KINLinSolDrv(kin_mem);
                if ret != KIN_SUCCESS {
                    break 'main_loop;
                }

                sflag = KINLineSearch(kin_mem, &mut fnormp, &mut f1normp, &mut maxStepTaken);

                /* if sysfunc failed unrecoverably, stop */
                if (sflag == KIN_SYSFUNC_FAIL) || (sflag == KIN_REPTD_SYSFUNC_ERR) {
                    ret = sflag;
                    break 'main_loop;
                }

                /* if too many beta condition failures, then stop iteration */
                let too_many_bcf = {
                    let m = kin_mem.borrow();
                    m.kin_nbcf > m.kin_mxnbcf
                };
                if too_many_bcf {
                    ret = KIN_LINESEARCH_BCFAIL;
                    break 'main_loop;
                }
            }

            let strategy = kin_mem.borrow().kin_globalstrategy;
            if (strategy != KIN_PICARD) && (strategy != KIN_FP) {
                /* evaluate eta by calling the forcing term routine */
                if kin_mem.borrow().kin_callForcingTerm {
                    KINForcingTerm(kin_mem, fnormp);
                }

                kin_mem.borrow_mut().kin_fnorm = fnormp;

                /* call KINStop to check if tolerances where met by this iteration */
                ret = KINStop(kin_mem, maxStepTaken, sflag);

                if ret == RETRY_ITERATION {
                    kin_mem.borrow_mut().kin_retry_nni = SUNTRUE;
                    continue 'repeat_nni;
                }
            }

            break 'repeat_nni;
        }

        /* update uu after the iteration */
        {
            let unew = get_unew(kin_mem);
            let uu = get_uu(kin_mem);
            N_VScale(ONE, &unew, &uu);
        }

        kin_mem.borrow_mut().kin_f1norm = f1normp;

        /* print the current nni, fnorm, and nfe values */

        /* KINPrintInfo(kin_mem, PRNT_NNI, "KINSOL", __func__, INFO_NNI,
        kin_nni, kin_nfe, kin_fnorm) — omitted (logging level < INFO) */

        if ret != CONTINUE_ITERATIONS {
            break 'main_loop;
        }
    } /* end of loop; return */

    /* KINPrintInfo(kin_mem, PRNT_RETVAL, "KINSOL", __func__, INFO_RETVAL, ret)
    — omitted (logging level < INFO) */

    match ret {
        KIN_SYSFUNC_FAIL => {
            KINProcessError(
                Some(kin_mem),
                KIN_SYSFUNC_FAIL,
                line!() as i32,
                "KINSol",
                file!(),
                MSG_SYSFUNC_FAILED,
            );
        }
        KIN_REPTD_SYSFUNC_ERR => {
            KINProcessError(
                Some(kin_mem),
                KIN_REPTD_SYSFUNC_ERR,
                line!() as i32,
                "KINSol",
                file!(),
                MSG_SYSFUNC_REPTD,
            );
        }
        KIN_LSETUP_FAIL => {
            KINProcessError(
                Some(kin_mem),
                KIN_LSETUP_FAIL,
                line!() as i32,
                "KINSol",
                file!(),
                MSG_LSETUP_FAILED,
            );
        }
        KIN_LSOLVE_FAIL => {
            KINProcessError(
                Some(kin_mem),
                KIN_LSOLVE_FAIL,
                line!() as i32,
                "KINSol",
                file!(),
                MSG_LSOLVE_FAILED,
            );
        }
        KIN_LINSOLV_NO_RECOVERY => {
            KINProcessError(
                Some(kin_mem),
                KIN_LINSOLV_NO_RECOVERY,
                line!() as i32,
                "KINSol",
                file!(),
                MSG_LINSOLV_NO_RECOVERY,
            );
        }
        KIN_LINESEARCH_NONCONV => {
            KINProcessError(
                Some(kin_mem),
                KIN_LINESEARCH_NONCONV,
                line!() as i32,
                "KINSol",
                file!(),
                MSG_LINESEARCH_NONCONV,
            );
        }
        KIN_LINESEARCH_BCFAIL => {
            KINProcessError(
                Some(kin_mem),
                KIN_LINESEARCH_BCFAIL,
                line!() as i32,
                "KINSol",
                file!(),
                MSG_LINESEARCH_BCFAIL,
            );
        }
        KIN_MAXITER_REACHED => {
            KINProcessError(
                Some(kin_mem),
                KIN_MAXITER_REACHED,
                line!() as i32,
                "KINSol",
                file!(),
                MSG_MAXITER_REACHED,
            );
        }
        KIN_MXNEWT_5X_EXCEEDED => {
            KINProcessError(
                Some(kin_mem),
                KIN_MXNEWT_5X_EXCEEDED,
                line!() as i32,
                "KINSol",
                file!(),
                MSG_MXNEWT_5X_EXCEEDED,
            );
        }
        _ => {}
    }

    ret
}

/*
 * -----------------------------------------------------------------
 * Deallocation function
 * -----------------------------------------------------------------
 */

/*
 * Function : KINFree
 *
 * This routine frees the problem memory allocated by KINInit.
 * Such memory includes all the vectors allocated by
 * KINAllocVectors, and the memory lmem for the linear solver
 * (deallocated by a call to lfree).
 */

pub fn KINFree(kinmem: &mut Option<KINMem>) {
    if kinmem.is_none() {
        return;
    }

    let kin_mem = kinmem.as_ref().unwrap().clone();
    KINFreeVectors(&kin_mem);

    /* call lfree if non-NULL */

    let lfree = kin_mem.borrow().kin_lfree;
    if let Some(lfree) = lfree {
        let _ = lfree(&kin_mem);
    }

    /* free Anderson acceleration workspace */
    KINFreeAA(&kin_mem);

    /* free orthogonalization workspace */
    KINFreeOrth(&kin_mem);

    kin_mem.borrow_mut().python = None;

    *kinmem = None;
}

/*
 * =================================================================
 * PRIVATE FUNCTIONS
 * =================================================================
 */

/*
 * Function : KINCheckNvector
 *
 * This routine checks if all required vector operations are
 * implemented (excluding those required by KINConstraint). If all
 * necessary operations are present, then KINCheckNvector returns
 * SUNTRUE. Otherwise, SUNFALSE is returned.
 */

fn KINCheckNvector(tmpl: &N_Vector) -> sunbooleantype {
    let ops = tmpl.ops.borrow();
    if ops.nvclone.is_none()
        || ops.nvdestroy.is_none()
        || ops.nvlinearsum.is_none()
        || ops.nvprod.is_none()
        || ops.nvdiv.is_none()
        || ops.nvscale.is_none()
        || ops.nvabs.is_none()
        || ops.nvinv.is_none()
        || ops.nvmaxnorm.is_none()
        || ops.nvmin.is_none()
        || ops.nvwl2norm.is_none()
    {
        SUNFALSE
    } else {
        SUNTRUE
    }
}

/*
 * -----------------------------------------------------------------
 * Memory allocation/deallocation
 * -----------------------------------------------------------------
 */

/*
 * Function : KINAllocVectors
 *
 * This routine allocates the KINSol vectors. If all memory
 * allocations are successful, KINAllocVectors returns SUNTRUE.
 * Otherwise all allocated memory is freed and KINAllocVectors
 * returns SUNFALSE.
 */

fn KINAllocVectors(kin_mem: &KINMem, tmpl: &N_Vector) -> sunbooleantype {
    /* allocate unew, fval, pp, vtemp1 and vtemp2. */
    /* allocate df, dg, q, for Anderson Acceleration, Broyden and EN */
    /* allocate L, for Low Sync Anderson Acceleration */

    if kin_mem.borrow().kin_unew.is_none() {
        let unew = N_VClone(tmpl);
        if unew.is_none() {
            return SUNFALSE;
        }
        let mut m = kin_mem.borrow_mut();
        m.kin_unew = unew;
        m.kin_liw += m.kin_liw1;
        m.kin_lrw += m.kin_lrw1;
    }

    if kin_mem.borrow().kin_fval.is_none() {
        let fval = N_VClone(tmpl);
        if fval.is_none() {
            let unew = kin_mem.borrow_mut().kin_unew.take().expect("kin_unew");
            N_VDestroy(unew);
            let mut m = kin_mem.borrow_mut();
            m.kin_liw -= m.kin_liw1;
            m.kin_lrw -= m.kin_lrw1;
            return SUNFALSE;
        }
        let mut m = kin_mem.borrow_mut();
        m.kin_fval = fval;
        m.kin_liw += m.kin_liw1;
        m.kin_lrw += m.kin_lrw1;
    }

    if kin_mem.borrow().kin_pp.is_none() {
        let pp = N_VClone(tmpl);
        if pp.is_none() {
            let (unew, fval) = {
                let mut m = kin_mem.borrow_mut();
                (
                    m.kin_unew.take().expect("kin_unew"),
                    m.kin_fval.take().expect("kin_fval"),
                )
            };
            N_VDestroy(unew);
            N_VDestroy(fval);
            let mut m = kin_mem.borrow_mut();
            m.kin_liw -= 2 * m.kin_liw1;
            m.kin_lrw -= 2 * m.kin_lrw1;
            return SUNFALSE;
        }
        let mut m = kin_mem.borrow_mut();
        m.kin_pp = pp;
        m.kin_liw += m.kin_liw1;
        m.kin_lrw += m.kin_lrw1;
    }

    if kin_mem.borrow().kin_vtemp1.is_none() {
        let vtemp1 = N_VClone(tmpl);
        if vtemp1.is_none() {
            let (unew, fval, pp) = {
                let mut m = kin_mem.borrow_mut();
                (
                    m.kin_unew.take().expect("kin_unew"),
                    m.kin_fval.take().expect("kin_fval"),
                    m.kin_pp.take().expect("kin_pp"),
                )
            };
            N_VDestroy(unew);
            N_VDestroy(fval);
            N_VDestroy(pp);
            let mut m = kin_mem.borrow_mut();
            m.kin_liw -= 3 * m.kin_liw1;
            m.kin_lrw -= 3 * m.kin_lrw1;
            return SUNFALSE;
        }
        let mut m = kin_mem.borrow_mut();
        m.kin_vtemp1 = vtemp1;
        m.kin_liw += m.kin_liw1;
        m.kin_lrw += m.kin_lrw1;
    }

    if kin_mem.borrow().kin_vtemp2.is_none() {
        let vtemp2 = N_VClone(tmpl);
        if vtemp2.is_none() {
            let (unew, fval, pp, vtemp1) = {
                let mut m = kin_mem.borrow_mut();
                (
                    m.kin_unew.take().expect("kin_unew"),
                    m.kin_fval.take().expect("kin_fval"),
                    m.kin_pp.take().expect("kin_pp"),
                    m.kin_vtemp1.take().expect("kin_vtemp1"),
                )
            };
            N_VDestroy(unew);
            N_VDestroy(fval);
            N_VDestroy(pp);
            N_VDestroy(vtemp1);
            let mut m = kin_mem.borrow_mut();
            m.kin_liw -= 4 * m.kin_liw1;
            m.kin_lrw -= 4 * m.kin_lrw1;
            return SUNFALSE;
        }
        let mut m = kin_mem.borrow_mut();
        m.kin_vtemp2 = vtemp2;
        m.kin_liw += m.kin_liw1;
        m.kin_lrw += m.kin_lrw1;
    }

    SUNTRUE
}

/*
 * KINFreeVectors
 *
 * This routine frees the KINSol vectors allocated by
 * KINAllocVectors.
 */

fn KINFreeVectors(kin_mem: &KINMem) {
    if kin_mem.borrow().kin_unew.is_some() {
        let v = kin_mem.borrow_mut().kin_unew.take().expect("kin_unew");
        N_VDestroy(v);
        let mut m = kin_mem.borrow_mut();
        m.kin_lrw -= m.kin_lrw1;
        m.kin_liw -= m.kin_liw1;
    }

    if kin_mem.borrow().kin_fval.is_some() {
        let v = kin_mem.borrow_mut().kin_fval.take().expect("kin_fval");
        N_VDestroy(v);
        let mut m = kin_mem.borrow_mut();
        m.kin_lrw -= m.kin_lrw1;
        m.kin_liw -= m.kin_liw1;
    }

    if kin_mem.borrow().kin_pp.is_some() {
        let v = kin_mem.borrow_mut().kin_pp.take().expect("kin_pp");
        N_VDestroy(v);
        let mut m = kin_mem.borrow_mut();
        m.kin_lrw -= m.kin_lrw1;
        m.kin_liw -= m.kin_liw1;
    }

    if kin_mem.borrow().kin_vtemp1.is_some() {
        let v = kin_mem.borrow_mut().kin_vtemp1.take().expect("kin_vtemp1");
        N_VDestroy(v);
        let mut m = kin_mem.borrow_mut();
        m.kin_lrw -= m.kin_lrw1;
        m.kin_liw -= m.kin_liw1;
    }

    if kin_mem.borrow().kin_vtemp2.is_some() {
        let v = kin_mem.borrow_mut().kin_vtemp2.take().expect("kin_vtemp2");
        N_VDestroy(v);
        let mut m = kin_mem.borrow_mut();
        m.kin_lrw -= m.kin_lrw1;
        m.kin_liw -= m.kin_liw1;
    }

    if kin_mem.borrow().kin_gval.is_some() {
        let v = kin_mem.borrow_mut().kin_gval.take().expect("kin_gval");
        N_VDestroy(v);
        let mut m = kin_mem.borrow_mut();
        m.kin_lrw -= m.kin_lrw1;
        m.kin_liw -= m.kin_liw1;
    }

    if kin_mem.borrow().kin_constraints.is_some() {
        let v = kin_mem
            .borrow_mut()
            .kin_constraints
            .take()
            .expect("kin_constraints");
        N_VDestroy(v);
        let mut m = kin_mem.borrow_mut();
        m.kin_lrw -= m.kin_lrw1;
        m.kin_liw -= m.kin_liw1;
    }
}

/*
 * -----------------------------------------------------------------
 * Initial setup
 * -----------------------------------------------------------------
 */

/*
 * KINSolInit
 *
 * KINSolInit initializes the problem for the specific input
 * received in this call to KINSol (which calls KINSolInit). All
 * problem specification inputs are checked for errors.
 *
 * The possible return values for KINSolInit are:
 *   KIN_SUCCESS : indicates a normal initialization
 *
 *   KIN_ILL_INPUT : indicates that an input error has been found
 *
 *   KIN_INITIAL_GUESS_OK : indicates that the guess uu
 *                          satisfied the system func(uu) = 0
 *                          within the tolerances specified
 */

fn KINSolInit(kin_mem: &KINMem) -> i32 {
    let mut retval: i32;
    let fmax: sunrealtype;

    /* check for illegal input parameters */

    if kin_mem.borrow().kin_uu.is_none() {
        KINProcessError(
            Some(kin_mem),
            KIN_ILL_INPUT,
            line!() as i32,
            "KINSolInit",
            file!(),
            MSG_UU_NULL,
        );
        return KIN_ILL_INPUT;
    }

    /* check for valid strategy */

    let globalstrategy = kin_mem.borrow().kin_globalstrategy;
    if (globalstrategy != KIN_NONE)
        && (globalstrategy != KIN_LINESEARCH)
        && (globalstrategy != KIN_PICARD)
        && (globalstrategy != KIN_FP)
    {
        KINProcessError(
            Some(kin_mem),
            KIN_ILL_INPUT,
            line!() as i32,
            "KINSolInit",
            file!(),
            MSG_BAD_GLSTRAT,
        );
        return KIN_ILL_INPUT;
    }

    if kin_mem.borrow().kin_uscale.is_none() {
        KINProcessError(
            Some(kin_mem),
            KIN_ILL_INPUT,
            line!() as i32,
            "KINSolInit",
            file!(),
            MSG_BAD_USCALE,
        );
        return KIN_ILL_INPUT;
    }

    if N_VMin(&get_uscale(kin_mem)) <= ZERO {
        KINProcessError(
            Some(kin_mem),
            KIN_ILL_INPUT,
            line!() as i32,
            "KINSolInit",
            file!(),
            MSG_USCALE_NONPOSITIVE,
        );
        return KIN_ILL_INPUT;
    }

    if kin_mem.borrow().kin_fscale.is_none() {
        KINProcessError(
            Some(kin_mem),
            KIN_ILL_INPUT,
            line!() as i32,
            "KINSolInit",
            file!(),
            MSG_BAD_FSCALE,
        );
        return KIN_ILL_INPUT;
    }

    if N_VMin(&get_fscale(kin_mem)) <= ZERO {
        KINProcessError(
            Some(kin_mem),
            KIN_ILL_INPUT,
            line!() as i32,
            "KINSolInit",
            file!(),
            MSG_FSCALE_NONPOSITIVE,
        );
        return KIN_ILL_INPUT;
    }

    if kin_mem.borrow().kin_constraints.is_some()
        && ((globalstrategy == KIN_PICARD) || (globalstrategy == KIN_FP))
    {
        KINProcessError(
            Some(kin_mem),
            KIN_ILL_INPUT,
            line!() as i32,
            "KINSolInit",
            file!(),
            MSG_CONSTRAINTS_NOTOK,
        );
        return KIN_ILL_INPUT;
    }

    /* set the constraints flag */

    if kin_mem.borrow().kin_constraints.is_none() {
        kin_mem.borrow_mut().kin_constraintsSet = SUNFALSE;
    } else {
        kin_mem.borrow_mut().kin_constraintsSet = SUNTRUE;
        let constraints = get_constraints(kin_mem);
        let bad_ops = {
            let ops = constraints.ops.borrow();
            ops.nvconstrmask.is_none() || ops.nvminquotient.is_none()
        };
        if bad_ops {
            KINProcessError(
                Some(kin_mem),
                KIN_ILL_INPUT,
                line!() as i32,
                "KINSolInit",
                file!(),
                MSG_BAD_NVECTOR,
            );
            return KIN_ILL_INPUT;
        }
    }

    /* check the initial guess uu against the constraints */

    if kin_mem.borrow().kin_constraintsSet {
        let constraints = get_constraints(kin_mem);
        let uu = get_uu(kin_mem);
        let vtemp1 = get_vtemp1(kin_mem);
        if !N_VConstrMask(&constraints, &uu, &vtemp1) {
            KINProcessError(
                Some(kin_mem),
                KIN_ILL_INPUT,
                line!() as i32,
                "KINSolInit",
                file!(),
                MSG_INITIAL_CNSTRNT,
            );
            return KIN_ILL_INPUT;
        }
    }

    /* all error checking is complete at this point */
    /* KINPrintInfo(kin_mem, PRNT_TOL, "KINSOL", __func__, INFO_TOL,
    kin_scsteptol, kin_fnormtol) — omitted (logging level < INFO) */

    /* calculate the default value for mxnewtstep (maximum Newton step) */

    let mxnstepin = kin_mem.borrow().kin_mxnstepin;
    if mxnstepin == ZERO {
        let uu = get_uu(kin_mem);
        let uscale = get_uscale(kin_mem);
        let mxnewtstep = THOUSAND * N_VWL2Norm(&uu, &uscale);
        kin_mem.borrow_mut().kin_mxnewtstep = mxnewtstep;
    } else {
        kin_mem.borrow_mut().kin_mxnewtstep = mxnstepin;
    }

    {
        let mut m = kin_mem.borrow_mut();
        if m.kin_mxnewtstep < ONE {
            m.kin_mxnewtstep = ONE;
        }

        /* additional set-up for inexact linear solvers */

        if m.kin_inexact_ls {
            /* set up the coefficients for the eta calculation */

            m.kin_callForcingTerm = m.kin_etaflag != KIN_ETACONSTANT;

            /* this value is always used for choice #1 */

            if m.kin_etaflag == KIN_ETACHOICE1 {
                m.kin_eta_alpha = (ONE + SUNRsqrt(FIVE)) * HALF;
            }

            /* initial value for eta set to 0.5 for other than the
               KIN_ETACONSTANT option */

            if m.kin_etaflag != KIN_ETACONSTANT {
                m.kin_eta = HALF;
            }

            /* disable residual monitoring if using an inexact linear solver */

            m.kin_noResMon = SUNTRUE;
        } else {
            m.kin_callForcingTerm = SUNFALSE;
        }

        /* initialize counters */

        m.kin_nfe = 0;
        m.kin_nnilset = 0;
        m.kin_nnilset_sub = 0;
        m.kin_nni = 0;
        m.kin_nbcf = 0;
        m.kin_nbktrk = 0;
    }

    /* see if the initial guess uu satisfies the nonlinear system */
    let uu = get_uu(kin_mem);
    let fval = get_fval(kin_mem);
    retval = kin_call_func(kin_mem, &uu, &fval);
    kin_mem.borrow_mut().kin_nfe += 1;

    if retval < 0 {
        KINProcessError(
            Some(kin_mem),
            KIN_SYSFUNC_FAIL,
            line!() as i32,
            "KINSolInit",
            file!(),
            MSG_SYSFUNC_FAILED,
        );
        return KIN_SYSFUNC_FAIL;
    } else if retval > 0 {
        KINProcessError(
            Some(kin_mem),
            KIN_FIRST_SYSFUNC_ERR,
            line!() as i32,
            "KINSolInit",
            file!(),
            MSG_SYSFUNC_FIRST,
        );
        return KIN_FIRST_SYSFUNC_ERR;
    }

    let fscale = get_fscale(kin_mem);
    fmax = KINScFNorm(kin_mem, &fval, &fscale);
    if fmax <= (POINT01 * kin_mem.borrow().kin_fnormtol) {
        let fnorm = N_VWL2Norm(&fval, &fscale);
        kin_mem.borrow_mut().kin_fnorm = fnorm;
        return KIN_INITIAL_GUESS_OK;
    }

    /* KINPrintInfo(kin_mem, PRNT_FMAX, "KINSOL", __func__, INFO_FMAX, fmax)
    — omitted (logging level < INFO) */

    /* initialize the linear solver if linit != NULL */

    let linit = kin_mem.borrow().kin_linit;
    if let Some(linit) = linit {
        retval = linit(kin_mem);
        if retval != 0 {
            KINProcessError(
                Some(kin_mem),
                KIN_LINIT_FAIL,
                line!() as i32,
                "KINSolInit",
                file!(),
                MSG_LINIT_FAIL,
            );
            return KIN_LINIT_FAIL;
        }
    }

    /* initialize the L2 (Euclidean) norms of f for the linear iteration steps */

    let fnorm = N_VWL2Norm(&fval, &fscale);
    {
        let mut m = kin_mem.borrow_mut();
        m.kin_fnorm = fnorm;
        m.kin_f1norm = HALF * fnorm * fnorm;
        m.kin_fnorm_sub = fnorm;
    }

    /* KINPrintInfo(kin_mem, PRNT_NNI, "KINSOL", __func__, INFO_NNI,
    kin_nni, kin_nfe, kin_fnorm) — omitted (logging level < INFO) */

    /* problem has now been successfully initialized */

    KIN_SUCCESS
}

/*
 * -----------------------------------------------------------------
 * Step functions
 * -----------------------------------------------------------------
 */

/*
 * KINLinSolDrv
 *
 * This routine handles the process of solving for the approximate
 * solution of the Newton equations in the Newton iteration.
 * Subsequent routines handle the nonlinear aspects of its
 * application.
 */

fn KINLinSolDrv(kin_mem: &KINMem) -> i32 {
    {
        let mut m = kin_mem.borrow_mut();
        if (m.kin_nni - m.kin_nnilset) >= m.kin_msbset {
            m.kin_sthrsh = TWO;
            m.kin_update_fnorm_sub = SUNTRUE;
        }
    }

    loop {
        kin_mem.borrow_mut().kin_jacCurrent = SUNFALSE;

        let (sthrsh, lsetup) = {
            let m = kin_mem.borrow();
            (m.kin_sthrsh, m.kin_lsetup)
        };
        if (sthrsh > ONEPT5) && lsetup.is_some() {
            let lsetup = lsetup.expect("kin_lsetup");
            let retval = lsetup(kin_mem);
            {
                let mut m = kin_mem.borrow_mut();
                m.kin_jacCurrent = SUNTRUE;
                m.kin_nnilset = m.kin_nni;
                m.kin_nnilset_sub = m.kin_nni;
            }
            if retval != 0 {
                return KIN_LSETUP_FAIL;
            }
        }

        /* rename vectors for readability */

        let b = get_unew(kin_mem);
        let x = get_pp(kin_mem);

        /* load b with the current value of -fval */

        let fval = get_fval(kin_mem);
        N_VScale(-ONE, &fval, &b);

        /* call the generic 'lsolve' routine to solve the system Jx = b */

        /* C passes &(kin_mem->kin_sJpnorm) / &(kin_mem->kin_sFdotJp): the
        write-back channel is preserved by seeding locals from the fields
        and storing them back immediately after the call. */
        let lsolve = kin_mem.borrow().kin_lsolve.expect("kin_lsolve");
        let (mut sJpnorm, mut sFdotJp) = {
            let m = kin_mem.borrow();
            (m.kin_sJpnorm, m.kin_sFdotJp)
        };
        let retval = lsolve(kin_mem, &x, &b, &mut sJpnorm, &mut sFdotJp);
        {
            let mut m = kin_mem.borrow_mut();
            m.kin_sJpnorm = sJpnorm;
            m.kin_sFdotJp = sFdotJp;
        }

        if retval == 0 {
            return KIN_SUCCESS;
        } else if retval < 0 {
            return KIN_LSOLVE_FAIL;
        } else {
            let m = kin_mem.borrow();
            if m.kin_lsetup.is_none() || m.kin_jacCurrent {
                return KIN_LINSOLV_NO_RECOVERY;
            }
        }

        /* loop back only if the linear solver setup is in use
           and Jacobian information is not current */

        kin_mem.borrow_mut().kin_sthrsh = TWO;
    }
}

/*
 * KINFullNewton
 *
 * This routine is the main driver for the Full Newton
 * algorithm. Its purpose is to compute unew = uu + pp in the
 * direction pp from uu, taking the full Newton step. The
 * step may be constrained if the constraint conditions are
 * violated, or if the norm of pp is greater than mxnewtstep.
 */

fn KINFullNewton(
    kin_mem: &KINMem,
    fnormp: &mut sunrealtype,
    f1normp: &mut sunrealtype,
    maxStepTaken: &mut sunbooleantype,
) -> i32 {
    let mut pnorm: sunrealtype;
    let mut ratio: sunrealtype;
    let mut fOK: sunbooleantype;
    let mut retval: i32;

    *maxStepTaken = SUNFALSE;
    pnorm = {
        let pp = get_pp(kin_mem);
        let uscale = get_uscale(kin_mem);
        N_VWL2Norm(&pp, &uscale)
    };
    ratio = ONE;
    let mxnewtstep = kin_mem.borrow().kin_mxnewtstep;
    if pnorm > mxnewtstep {
        ratio = mxnewtstep / pnorm;
        let pp = get_pp(kin_mem);
        N_VScale(ratio, &pp, &pp);
        pnorm = mxnewtstep;
    }

    /* KINPrintInfo(kin_mem, PRNT_PNORM, "KINSOL", __func__, INFO_PNORM, pnorm)
    — omitted (logging level < INFO) */

    /* If constraints are active, then constrain the step accordingly */

    {
        let mut m = kin_mem.borrow_mut();
        m.kin_stepl = pnorm;
        m.kin_stepmul = ONE;
    }
    if kin_mem.borrow().kin_constraintsSet {
        retval = KINConstraint(kin_mem);
        if retval == CONSTR_VIOLATED {
            /* Apply stepmul set in KINConstraint */
            let stepmul = kin_mem.borrow().kin_stepmul;
            ratio *= stepmul;
            let pp = get_pp(kin_mem);
            N_VScale(stepmul, &pp, &pp);
            pnorm *= stepmul;
            kin_mem.borrow_mut().kin_stepl = pnorm;

            /* KINPrintInfo(kin_mem, PRNT_PNORM, "KINSOL", __func__, INFO_PNORM,
            pnorm) — omitted (logging level < INFO) */

            if pnorm <= kin_mem.borrow().kin_scsteptol {
                let uu = get_uu(kin_mem);
                let unew = get_unew(kin_mem);
                N_VLinearSum(ONE, &uu, ONE, &pp, &unew);
                return STEP_TOO_SMALL;
            }
        }
    }

    /* Attempt (at most MAX_RECVR times) to evaluate function at the new iterate */

    fOK = SUNFALSE;

    for _ircvr in 1..=MAX_RECVR {
        /* compute the iterate unew = uu + pp */
        {
            let uu = get_uu(kin_mem);
            let pp = get_pp(kin_mem);
            let unew = get_unew(kin_mem);
            N_VLinearSum(ONE, &uu, ONE, &pp, &unew);
        }

        /* evaluate func(unew) and its norm, and return */
        let unew = get_unew(kin_mem);
        let fval = get_fval(kin_mem);
        retval = kin_call_func(kin_mem, &unew, &fval);
        kin_mem.borrow_mut().kin_nfe += 1;

        /* if func was successful, accept pp */
        if retval == 0 {
            fOK = SUNTRUE;
            break;
        }
        /* if func failed unrecoverably, give up */
        else if retval < 0 {
            return KIN_SYSFUNC_FAIL;
        }

        /* func failed recoverably; cut step in half and try again */
        ratio *= HALF;
        let pp = get_pp(kin_mem);
        N_VScale(HALF, &pp, &pp);
        pnorm *= HALF;
        kin_mem.borrow_mut().kin_stepl = pnorm;
    }

    /* If func() failed recoverably MAX_RECVR times, give up */

    if !fOK {
        return KIN_REPTD_SYSFUNC_ERR;
    }

    /* Evaluate function norms */

    {
        let fval = get_fval(kin_mem);
        let fscale = get_fscale(kin_mem);
        *fnormp = N_VWL2Norm(&fval, &fscale);
    }
    *f1normp = HALF * (*fnormp) * (*fnormp);

    /* scale sFdotJp and sJpnorm by ratio for later use in KINForcingTerm */

    {
        let mut m = kin_mem.borrow_mut();
        m.kin_sFdotJp *= ratio;
        m.kin_sJpnorm *= ratio;
    }

    /* KINPrintInfo(kin_mem, PRNT_FNORM, "KINSOL", __func__, INFO_FNORM, *fnormp)
    — omitted (logging level < INFO) */

    if pnorm > (POINT99 * kin_mem.borrow().kin_mxnewtstep) {
        *maxStepTaken = SUNTRUE;
    }

    KIN_SUCCESS
}

/*
 * KINLineSearch
 *
 * The routine KINLineSearch implements the LineSearch algorithm.
 * Its purpose is to find unew = uu + rl * pp in the direction pp
 * from uu so that:
 *                                    t
 *  func(unew) <= func(uu) + alpha * g  (unew - uu) (alpha = 1.e-4)
 *
 *    and
 *                                   t
 *  func(unew) >= func(uu) + beta * g  (unew - uu) (beta = 0.9)
 *
 * where 0 < rlmin <= rl <= rlmax.
 *
 * Note:
 *             mxnewtstep
 *  rlmax = ----------------   if uu+pp is feasible
 *          ||uscale*pp||_L2
 *
 *  rlmax = 1   otherwise
 *
 *    and
 *
 *                 scsteptol
 *  rlmin = --------------------------
 *          ||           pp         ||
 *          || -------------------- ||_L-infinity
 *          || (1/uscale + SUNRabs(uu)) ||
 *
 *
 * If the system function fails unrecoverably at any time, KINLineSearch
 * returns KIN_SYSFUNC_FAIL which will halt the solver.
 *
 * We attempt to correct recoverable system function failures only before
 * the alpha-condition loop; i.e. when the solution is updated with the
 * full Newton step (possibly reduced due to constraint violations).
 * Once we find a feasible pp, we assume that any update up to pp is
 * feasible.
 *
 * If the step size is limited due to constraint violations and/or
 * recoverable system function failures, we set rlmax=1 to ensure
 * that the update remains feasible during the attempts to enforce
 * the beta-condition (this is not an issue while enforcing the alpha
 * condition, as rl can only decrease from 1 at that stage)
 */

fn KINLineSearch(
    kin_mem: &KINMem,
    fnormp: &mut sunrealtype,
    f1normp: &mut sunrealtype,
    maxStepTaken: &mut sunbooleantype,
) -> i32 {
    /* C declares pnorm, ratio, slpi, rlmin, rlength, rl, rlmax, rldiff,
    rltmp, rlprev, pt1trl, f1nprv, rllo, rlinc, alpha_cond, beta_cond,
    rl_a, tmp1, rl_b, tmp2, disc, ircvr, nbktrk_l, retval, firstBacktrack
    and fOK at the top of the function; the write-once ones are declared
    at their assignment site here so no binding needs a spurious `mut`. */
    let mut pnorm: sunrealtype;
    let mut ratio: sunrealtype;
    let mut rl: sunrealtype;
    let mut rlmax: sunrealtype;
    let mut rltmp: sunrealtype;
    let mut rlprev: sunrealtype;
    let mut f1nprv: sunrealtype;
    let mut alpha_cond: sunrealtype;
    let mut beta_cond: sunrealtype;
    let mut nbktrk_l: i32;
    let mut retval: i32;
    let mut firstBacktrack: sunbooleantype;
    let mut fOK: sunbooleantype;

    /* Initializations */

    nbktrk_l = 0; /* local backtracking counter */
    ratio = ONE; /* step change ratio          */
    let alpha: sunrealtype = POINT0001;
    let beta: sunrealtype = POINT9;

    firstBacktrack = SUNTRUE;
    *maxStepTaken = SUNFALSE;

    rlprev = ZERO;
    f1nprv = ZERO;

    /* Compute length of Newton step */

    pnorm = {
        let pp = get_pp(kin_mem);
        let uscale = get_uscale(kin_mem);
        N_VWL2Norm(&pp, &uscale)
    };
    let mxnewtstep = kin_mem.borrow().kin_mxnewtstep;
    rlmax = mxnewtstep / pnorm;
    kin_mem.borrow_mut().kin_stepl = pnorm;

    /* If the full Newton step is too large, set it to the maximum allowable value */

    if pnorm > mxnewtstep {
        ratio = mxnewtstep / pnorm;
        let pp = get_pp(kin_mem);
        N_VScale(ratio, &pp, &pp);
        pnorm = mxnewtstep;
        rlmax = ONE;
        kin_mem.borrow_mut().kin_stepl = pnorm;
    }

    /* If constraint checking is activated, check and correct violations */

    kin_mem.borrow_mut().kin_stepmul = ONE;

    if kin_mem.borrow().kin_constraintsSet {
        retval = KINConstraint(kin_mem);
        if retval == CONSTR_VIOLATED {
            /* Apply stepmul set in KINConstraint */
            let stepmul = kin_mem.borrow().kin_stepmul;
            let pp = get_pp(kin_mem);
            N_VScale(stepmul, &pp, &pp);
            ratio *= stepmul;
            pnorm *= stepmul;
            rlmax = ONE;
            kin_mem.borrow_mut().kin_stepl = pnorm;

            /* KINPrintInfo(kin_mem, PRNT_PNORM1, "KINSOL", __func__,
            INFO_PNORM1, pnorm) — omitted (logging level < INFO) */

            if pnorm <= kin_mem.borrow().kin_scsteptol {
                let uu = get_uu(kin_mem);
                let unew = get_unew(kin_mem);
                N_VLinearSum(ONE, &uu, ONE, &pp, &unew);
                return STEP_TOO_SMALL;
            }
        }
    }

    /* Attempt (at most MAX_RECVR times) to evaluate function at the new iterate */

    fOK = SUNFALSE;

    for _ircvr in 1..=MAX_RECVR {
        /* compute the iterate unew = uu + pp */
        {
            let uu = get_uu(kin_mem);
            let pp = get_pp(kin_mem);
            let unew = get_unew(kin_mem);
            N_VLinearSum(ONE, &uu, ONE, &pp, &unew);
        }

        /* evaluate func(unew) and its norm, and return */
        let unew = get_unew(kin_mem);
        let fval = get_fval(kin_mem);
        retval = kin_call_func(kin_mem, &unew, &fval);
        kin_mem.borrow_mut().kin_nfe += 1;

        /* if func was successful, accept pp */
        if retval == 0 {
            fOK = SUNTRUE;
            break;
        }
        /* if func failed unrecoverably, give up */
        else if retval < 0 {
            return KIN_SYSFUNC_FAIL;
        }

        /* func failed recoverably; cut step in half and try again */
        let pp = get_pp(kin_mem);
        N_VScale(HALF, &pp, &pp);
        ratio *= HALF;
        pnorm *= HALF;
        rlmax = ONE;
        kin_mem.borrow_mut().kin_stepl = pnorm;
    }

    /* If func() failed recoverably MAX_RECVR times, give up */

    if !fOK {
        return KIN_REPTD_SYSFUNC_ERR;
    }

    /* Evaluate function norms */

    {
        let fval = get_fval(kin_mem);
        let fscale = get_fscale(kin_mem);
        *fnormp = N_VWL2Norm(&fval, &fscale);
    }
    *f1normp = HALF * (*fnormp) * (*fnormp);

    /* Estimate the line search value rl (lambda) to satisfy both ALPHA and BETA conditions */

    let slpi: sunrealtype = kin_mem.borrow().kin_sFdotJp * ratio;
    let rlength: sunrealtype = {
        let pp = get_pp(kin_mem);
        let uu = get_uu(kin_mem);
        KINScSNorm(kin_mem, &pp, &uu)
    };
    let rlmin: sunrealtype = kin_mem.borrow().kin_scsteptol / rlength;
    rl = ONE;

    /* KINPrintInfo(kin_mem, PRNT_LAM, "KINSOL", __func__, INFO_LAM, rlmin,
    kin_f1norm, pnorm) — omitted (logging level < INFO) */

    /* Loop until the ALPHA condition is satisfied. Terminate if rl becomes too small */

    loop {
        /* Evaluate test quantity */

        alpha_cond = kin_mem.borrow().kin_f1norm + (alpha * slpi * rl);

        /* KINPrintInfo(kin_mem, PRNT_ALPHA, "KINSOL", __func__, INFO_ALPHA,
        *fnormp, *f1normp, alpha_cond, rl) — omitted (logging level < INFO) */

        /* If ALPHA condition is satisfied, break out from loop */

        if (*f1normp) <= alpha_cond {
            break;
        }

        /* Backtracking. Use quadratic fit the first time and cubic fit afterwards. */

        if firstBacktrack {
            let f1norm = kin_mem.borrow().kin_f1norm;
            rltmp = -slpi / (TWO * ((*f1normp) - f1norm - slpi));
            firstBacktrack = SUNFALSE;
        } else {
            let f1norm = kin_mem.borrow().kin_f1norm;
            let mut tmp1 = (*f1normp) - f1norm - (rl * slpi);
            let tmp2 = f1nprv - f1norm - (rlprev * slpi);
            let mut rl_a = ((ONE / (rl * rl)) * tmp1) - ((ONE / (rlprev * rlprev)) * tmp2);
            let mut rl_b = ((-rlprev / (rl * rl)) * tmp1) + ((rl / (rlprev * rlprev)) * tmp2);
            tmp1 = ONE / (rl - rlprev);
            rl_a *= tmp1;
            rl_b *= tmp1;
            let disc = (rl_b * rl_b) - (THREE * rl_a * slpi);

            if SUNRabs(rl_a) < kin_mem.borrow().kin_uround {
                /* cubic is actually just a quadratic (rl_a ~ 0) */
                rltmp = -slpi / (TWO * rl_b);
            } else {
                /* real cubic */
                rltmp = (-rl_b + SUNRsqrt(disc)) / (THREE * rl_a);
            }
        }
        if rltmp > (HALF * rl) {
            rltmp = HALF * rl;
        }

        /* Set new rl (do not allow a reduction by a factor larger than 10) */

        rlprev = rl;
        f1nprv = *f1normp;
        let pt1trl = POINT1 * rl;
        rl = SUNMAX(pt1trl, rltmp);
        nbktrk_l += 1;

        /* Update unew and re-evaluate function */

        {
            let uu = get_uu(kin_mem);
            let pp = get_pp(kin_mem);
            let unew = get_unew(kin_mem);
            N_VLinearSum(ONE, &uu, rl, &pp, &unew);
        }

        let unew = get_unew(kin_mem);
        let fval = get_fval(kin_mem);
        retval = kin_call_func(kin_mem, &unew, &fval);
        kin_mem.borrow_mut().kin_nfe += 1;
        if retval != 0 {
            return KIN_SYSFUNC_FAIL;
        }

        {
            let fscale = get_fscale(kin_mem);
            *fnormp = N_VWL2Norm(&fval, &fscale);
        }
        *f1normp = HALF * (*fnormp) * (*fnormp);

        /* Check if rl (lambda) is too small */

        if rl < rlmin {
            /* unew sufficiently distinct from uu cannot be found.
               copy uu into unew (step remains unchanged) and
               return STEP_TOO_SMALL */
            let uu = get_uu(kin_mem);
            let unew = get_unew(kin_mem);
            N_VScale(ONE, &uu, &unew);
            return STEP_TOO_SMALL;
        }
    } /* end ALPHA condition loop */

    /* ALPHA condition is satisfied. Now check the BETA condition */

    beta_cond = kin_mem.borrow().kin_f1norm + (beta * slpi * rl);

    if (*f1normp) < beta_cond {
        /* BETA condition not satisfied */

        if (rl == ONE) && (pnorm < kin_mem.borrow().kin_mxnewtstep) {
            loop {
                rlprev = rl;
                /* dead store in the C source too (kinsol.c:1657): f1nprv is
                 * only read inside the ALPHA loop above.  Kept for fidelity. */
                #[allow(unused_assignments)]
                {
                    f1nprv = *f1normp;
                }
                rl = SUNMIN(TWO * rl, rlmax);
                nbktrk_l += 1;

                {
                    let uu = get_uu(kin_mem);
                    let pp = get_pp(kin_mem);
                    let unew = get_unew(kin_mem);
                    N_VLinearSum(ONE, &uu, rl, &pp, &unew);
                }
                let unew = get_unew(kin_mem);
                let fval = get_fval(kin_mem);
                retval = kin_call_func(kin_mem, &unew, &fval);
                kin_mem.borrow_mut().kin_nfe += 1;
                if retval != 0 {
                    return KIN_SYSFUNC_FAIL;
                }
                {
                    let fscale = get_fscale(kin_mem);
                    *fnormp = N_VWL2Norm(&fval, &fscale);
                }
                *f1normp = HALF * (*fnormp) * (*fnormp);

                {
                    let f1norm = kin_mem.borrow().kin_f1norm;
                    alpha_cond = f1norm + (alpha * slpi * rl);
                    beta_cond = f1norm + (beta * slpi * rl);
                }

                /* KINPrintInfo(kin_mem, PRNT_BETA, "KINSOL", __func__, INFO_BETA,
                *f1normp, beta_cond, rl) — omitted (logging level < INFO) */

                if !(((*f1normp) <= alpha_cond) && ((*f1normp) < beta_cond) && (rl < rlmax)) {
                    break;
                }
            }
        } /* end if (rl == ONE) block */

        if (rl < ONE) || ((rl > ONE) && (*f1normp > alpha_cond)) {
            let mut rllo = SUNMIN(rl, rlprev);
            let mut rldiff = SUNRabs(rlprev - rl);

            loop {
                let rlinc = HALF * rldiff;
                rl = rllo + rlinc;
                nbktrk_l += 1;

                {
                    let uu = get_uu(kin_mem);
                    let pp = get_pp(kin_mem);
                    let unew = get_unew(kin_mem);
                    N_VLinearSum(ONE, &uu, rl, &pp, &unew);
                }
                let unew = get_unew(kin_mem);
                let fval = get_fval(kin_mem);
                retval = kin_call_func(kin_mem, &unew, &fval);
                kin_mem.borrow_mut().kin_nfe += 1;
                if retval != 0 {
                    return KIN_SYSFUNC_FAIL;
                }
                {
                    let fscale = get_fscale(kin_mem);
                    *fnormp = N_VWL2Norm(&fval, &fscale);
                }
                *f1normp = HALF * (*fnormp) * (*fnormp);

                {
                    let f1norm = kin_mem.borrow().kin_f1norm;
                    alpha_cond = f1norm + (alpha * slpi * rl);
                    beta_cond = f1norm + (beta * slpi * rl);
                }

                /* KINPrintInfo(kin_mem, PRNT_ALPHABETA, "KINSOL", __func__,
                INFO_ALPHABETA, *f1normp, alpha_cond, beta_cond, rl)
                — omitted (logging level < INFO) */

                if (*f1normp) > alpha_cond {
                    rldiff = rlinc;
                } else if *f1normp < beta_cond {
                    rllo = rl;
                    rldiff -= rlinc;
                }

                if !((*f1normp > alpha_cond) || ((*f1normp < beta_cond) && (rldiff >= rlmin))) {
                    break;
                }
            }

            if (*f1normp < beta_cond) || ((rldiff < rlmin) && (*f1normp > alpha_cond)) {
                /* beta condition could not be satisfied or rldiff too small
                   and alpha_cond not satisfied, so set unew to last u value
                   that satisfied the alpha condition and continue */

                {
                    let uu = get_uu(kin_mem);
                    let pp = get_pp(kin_mem);
                    let unew = get_unew(kin_mem);
                    N_VLinearSum(ONE, &uu, rllo, &pp, &unew);
                }
                let unew = get_unew(kin_mem);
                let fval = get_fval(kin_mem);
                retval = kin_call_func(kin_mem, &unew, &fval);
                kin_mem.borrow_mut().kin_nfe += 1;
                if retval != 0 {
                    return KIN_SYSFUNC_FAIL;
                }
                {
                    let fscale = get_fscale(kin_mem);
                    *fnormp = N_VWL2Norm(&fval, &fscale);
                }
                *f1normp = HALF * (*fnormp) * (*fnormp);

                /* increment beta-condition failures counter */

                kin_mem.borrow_mut().kin_nbcf += 1;
            }
        } /* end of if (rl < ONE) block */
    } /* end of if (f1normp < beta_cond) block */

    /* Update number of backtracking operations */

    kin_mem.borrow_mut().kin_nbktrk += nbktrk_l as i64;

    /* KINPrintInfo(kin_mem, PRNT_ADJ, "KINSOL", __func__, INFO_ADJ, nbktrk_l)
    — omitted (logging level < INFO) */

    /* scale sFdotJp and sJpnorm by rl * ratio for later use in KINForcingTerm */

    {
        let mut m = kin_mem.borrow_mut();
        m.kin_sFdotJp = m.kin_sFdotJp * rl * ratio;
        m.kin_sJpnorm = m.kin_sJpnorm * rl * ratio;
    }

    if (rl * pnorm) > (POINT99 * kin_mem.borrow().kin_mxnewtstep) {
        *maxStepTaken = SUNTRUE;
    }

    KIN_SUCCESS
}

/*
 * Function : KINConstraint
 *
 * This routine checks if the proposed solution vector uu + pp
 * violates any constraints. If a constraint is violated, then the
 * scalar stepmul is determined such that uu + stepmul * pp does
 * not violate any constraints.
 *
 * Note: This routine is called by the functions
 *       KINLineSearch and KINFullNewton.
 */

fn KINConstraint(kin_mem: &KINMem) -> i32 {
    let uu = get_uu(kin_mem);
    let pp = get_pp(kin_mem);
    let constraints = get_constraints(kin_mem);
    let vtemp1 = get_vtemp1(kin_mem);
    let vtemp2 = get_vtemp2(kin_mem);

    N_VLinearSum(ONE, &uu, ONE, &pp, &vtemp1);

    /* if vtemp1[i] violates constraint[i] then vtemp2[i] = 1
       else vtemp2[i] = 0 (vtemp2 is the mask vector) */

    if N_VConstrMask(&constraints, &vtemp1, &vtemp2) {
        return KIN_SUCCESS;
    }

    /* vtemp1[i] = SUNRabs(pp[i]) */

    N_VAbs(&pp, &vtemp1);

    /* consider vtemp1[i] only if vtemp2[i] = 1 (constraint violated) */

    N_VProd(&vtemp2, &vtemp1, &vtemp1);

    N_VAbs(&uu, &vtemp2);
    let stepmul = POINT9 * N_VMinQuotient(&vtemp2, &vtemp1);
    kin_mem.borrow_mut().kin_stepmul = stepmul;

    CONSTR_VIOLATED
}

/*
 * -----------------------------------------------------------------
 * Stopping tests
 * -----------------------------------------------------------------
 */

/*
 * KINStop
 *
 * This routine checks the current iterate unew to see if the
 * system func(unew) = 0 is satisfied by a variety of tests.
 *
 * strategy is one of KIN_NONE or KIN_LINESEARCH
 * sflag    is one of KIN_SUCCESS, STEP_TOO_SMALL
 */

fn KINStop(kin_mem: &KINMem, maxStepTaken: sunbooleantype, sflag: i32) -> i32 {
    /* C declares fmax, rlength, omexp and delta at the top of the function */

    /* Check for too small a step */

    if sflag == STEP_TOO_SMALL {
        let (has_lsetup, jacCurrent) = {
            let m = kin_mem.borrow();
            (m.kin_lsetup.is_some(), m.kin_jacCurrent)
        };
        if has_lsetup && !jacCurrent {
            /* If the Jacobian is out of date, update it and retry */
            kin_mem.borrow_mut().kin_sthrsh = TWO;
            return RETRY_ITERATION;
        } else {
            /* Give up */
            if kin_mem.borrow().kin_globalstrategy == KIN_NONE {
                return KIN_STEP_LT_STPTOL;
            } else {
                return KIN_LINESEARCH_NONCONV;
            }
        }
    }

    /* Check tolerance on scaled function norm at the current iterate */

    let fmax: sunrealtype = {
        let fval = get_fval(kin_mem);
        let fscale = get_fscale(kin_mem);
        KINScFNorm(kin_mem, &fval, &fscale)
    };

    /* KINPrintInfo(kin_mem, PRNT_FMAX, "KINSOL", __func__, INFO_FMAX, fmax)
    — omitted (logging level < INFO) */

    if fmax <= kin_mem.borrow().kin_fnormtol {
        return KIN_SUCCESS;
    }

    /* Check if the scaled distance between the last two steps is too small */
    /* NOTE: pp used as work space to store this distance */

    let delta = get_pp(kin_mem);
    {
        let unew = get_unew(kin_mem);
        let uu = get_uu(kin_mem);
        N_VLinearSum(ONE, &unew, -ONE, &uu, &delta);
    }
    let rlength: sunrealtype = {
        let unew = get_unew(kin_mem);
        KINScSNorm(kin_mem, &delta, &unew)
    };

    if rlength <= kin_mem.borrow().kin_scsteptol {
        let (has_lsetup, jacCurrent) = {
            let m = kin_mem.borrow();
            (m.kin_lsetup.is_some(), m.kin_jacCurrent)
        };
        if has_lsetup && !jacCurrent {
            /* If the Jacobian is out of date, update it and retry */
            kin_mem.borrow_mut().kin_sthrsh = TWO;
            return CONTINUE_ITERATIONS;
        } else {
            /* give up */
            return KIN_STEP_LT_STPTOL;
        }
    }

    /* Check if the maximum number of iterations is reached */

    {
        let m = kin_mem.borrow();
        if m.kin_nni >= m.kin_mxiter {
            return KIN_MAXITER_REACHED;
        }
    }

    /* Check for consecutive number of steps taken of size mxnewtstep
       and if not maxStepTaken, then set ncscmx to 0 */

    {
        let mut m = kin_mem.borrow_mut();
        if maxStepTaken {
            m.kin_ncscmx += 1;
        } else {
            m.kin_ncscmx = 0;
        }

        if m.kin_ncscmx == 5 {
            return KIN_MXNEWT_5X_EXCEEDED;
        }
    }

    /* Proceed according to the type of linear solver used */

    let mut m = kin_mem.borrow_mut();
    if m.kin_inexact_ls {
        /* We're doing inexact Newton.
           Load threshold for reevaluating the Jacobian. */

        m.kin_sthrsh = rlength;
    } else if !(m.kin_noResMon) {
        /* We're doing modified Newton and the user did not disable residual monitoring.
           Check if it is time to monitor residual. */

        if (m.kin_nni - m.kin_nnilset_sub) >= m.kin_msbset_sub {
            /* Residual monitoring needed */

            m.kin_nnilset_sub = m.kin_nni;

            /* If indicated, estimate new OMEGA value */
            if m.kin_eval_omega {
                let omexp = SUNMAX(ZERO, (m.kin_fnorm / m.kin_fnormtol) - ONE);
                let omega_min = m.kin_omega_min;
                let omega_max = m.kin_omega_max;
                m.kin_omega = if omexp > TWELVE {
                    omega_max
                } else {
                    SUNMIN(omega_min * SUNRexp(omexp), omega_max)
                };
            }
            /* Check if making satisfactory progress */

            if m.kin_fnorm > m.kin_omega * m.kin_fnorm_sub {
                /* Insufficient progress */
                if m.kin_lsetup.is_some() && !(m.kin_jacCurrent) {
                    /* If the Jacobian is out of date, update it and retry */
                    m.kin_sthrsh = TWO;
                    return CONTINUE_ITERATIONS;
                } else {
                    /* Otherwise, we cannot do anything, so just return. */
                }
            } else {
                /* Sufficient progress */
                m.kin_fnorm_sub = m.kin_fnorm;
                m.kin_sthrsh = ONE;
            }
        } else {
            /* Residual monitoring not needed */

            /* Reset sthrsh */
            if m.kin_retry_nni || m.kin_update_fnorm_sub {
                m.kin_fnorm_sub = m.kin_fnorm;
            }
            if m.kin_update_fnorm_sub {
                m.kin_update_fnorm_sub = SUNFALSE;
            }
            m.kin_sthrsh = ONE;
        }
    }

    /* if made it to here, then the iteration process is not finished
       so return CONTINUE_ITERATIONS flag */

    CONTINUE_ITERATIONS
}

/*
 * KINForcingTerm
 *
 * This routine computes eta, the scaling factor in the linear
 * convergence stopping tolerance eps when choice #1 or choice #2
 * forcing terms are used. Eta is computed here for all but the
 * first iterative step, which is set to the default in routine
 * KINSolInit.
 *
 * This routine was written by Homer Walker of Utah State
 * University with subsequent modifications by Allan Taylor @ LLNL.
 *
 * It is based on the concepts of the paper 'Choosing the forcing
 * terms in an inexact Newton method', SIAM J Sci Comput, 17
 * (1996), pp 16 - 32, or Utah State University Research Report
 * 6/94/75 of the same title.
 */

fn KINForcingTerm(kin_mem: &KINMem, fnormp: sunrealtype) {
    /* No callback / vector op below, so one borrow for the whole body is
    safe (nothing here can re-enter the mem). */
    let mut m = kin_mem.borrow_mut();

    let eta_max: sunrealtype = POINT9;
    let eta_min: sunrealtype = POINT0001;
    let mut eta_safe: sunrealtype = HALF;

    /* choice #1 forcing term */

    if m.kin_etaflag == KIN_ETACHOICE1 {
        /* compute the norm of f + Jp , scaled L2 norm */

        let linmodel_norm = SUNRsqrt(
            (m.kin_fnorm * m.kin_fnorm) + (TWO * m.kin_sFdotJp) + (m.kin_sJpnorm * m.kin_sJpnorm),
        );

        /* form the safeguarded for choice #1 */

        eta_safe = SUNRpowerR(m.kin_eta, m.kin_eta_alpha);
        m.kin_eta = SUNRabs(fnormp - linmodel_norm) / m.kin_fnorm;
    }

    /* choice #2 forcing term */

    if m.kin_etaflag == KIN_ETACHOICE2 {
        eta_safe = m.kin_eta_gamma * SUNRpowerR(m.kin_eta, m.kin_eta_alpha);

        m.kin_eta = m.kin_eta_gamma * SUNRpowerR(fnormp / m.kin_fnorm, m.kin_eta_alpha);
    }

    /* apply safeguards */

    if eta_safe < POINT1 {
        eta_safe = ZERO;
    }
    m.kin_eta = SUNMAX(m.kin_eta, eta_safe);
    m.kin_eta = SUNMAX(m.kin_eta, eta_min);
    m.kin_eta = SUNMIN(m.kin_eta, eta_max);
}

/*
 * -----------------------------------------------------------------
 * Norm functions
 * -----------------------------------------------------------------
 */

/*
 * Function : KINScFNorm
 *
 * This routine computes the max norm for scaled vectors. The
 * scaling vector is scale, and the vector of which the norm is to
 * be determined is vv. The returned value, fnormval, is the
 * resulting scaled vector norm. This routine uses N_Vector
 * functions from the vector module.
 */

fn KINScFNorm(kin_mem: &KINMem, v: &N_Vector, scale: &N_Vector) -> sunrealtype {
    let vtemp1 = get_vtemp1(kin_mem);
    N_VProd(scale, v, &vtemp1);
    N_VMaxNorm(&vtemp1)
}

/*
 * Function : KINScSNorm
 *
 * This routine computes the max norm of the scaled steplength, ss.
 * Here ucur is the current step and usc is the u scale factor.
 */

fn KINScSNorm(kin_mem: &KINMem, v: &N_Vector, u: &N_Vector) -> sunrealtype {
    let uscale = get_uscale(kin_mem);
    let vtemp1 = get_vtemp1(kin_mem);
    let vtemp2 = get_vtemp2(kin_mem);

    N_VInv(&uscale, &vtemp1);
    N_VAbs(u, &vtemp2);
    N_VLinearSum(ONE, &vtemp1, ONE, &vtemp2, &vtemp1);
    N_VDiv(v, &vtemp1, &vtemp1);

    let length = N_VMaxNorm(&vtemp1);

    length
}

/*
 * =======================================================================
 * Picard and fixed point solvers
 * =======================================================================
 */

/*
 * KINPicardAA
 *
 * This routine is the main driver for the Picard iteration with
 * accelerated fixed point.
 */

fn KINPicardAA(kin_mem: &KINMem) -> i32 {
    let mut retval: i32; /* return value from user func */
    let mut ret: i32; /* iteration status            */
    let mut epsmin: sunrealtype;
    /* C also declares `long int iter_aa` and `sunrealtype fnormp` here;
    both are single-assignment and are declared at their use site below
    (C's `fnormp = -ONE` initializer is a provably-dead store that only
    silences a compiler warning). */

    let delta = get_vtemp1(kin_mem); /* temporary workspace vector  */
    ret = CONTINUE_ITERATIONS;
    epsmin = ZERO;

    /* initialize iteration count */
    kin_mem.borrow_mut().kin_nni = 0;

    /* if eps is to be bounded from below, set the bound */
    {
        let m = kin_mem.borrow();
        if m.kin_inexact_ls && !(m.kin_noMinEps) {
            epsmin = POINT01 * m.kin_fnormtol;
        }
    }

    while ret == CONTINUE_ITERATIONS {
        /* update iteration count */
        kin_mem.borrow_mut().kin_nni += 1;

        /* Update the forcing term for the inexact linear solves */
        {
            let mut m = kin_mem.borrow_mut();
            if m.kin_inexact_ls {
                m.kin_eps = (m.kin_eta + m.kin_uround) * m.kin_fnorm;
                if !(m.kin_noMinEps) {
                    m.kin_eps = SUNMAX(epsmin, m.kin_eps);
                }
            }
        }

        /* evaluate g = uu - L^{-1}func(uu) and return if failed.
           For Picard, assume that the fval vector has been filled
           with an eval of the nonlinear residual prior to this call. */
        {
            let gval = get_gval(kin_mem);
            let uu = get_uu(kin_mem);
            let fval = get_fval(kin_mem);
            retval = KINPicardFcnEval(kin_mem, &gval, &uu, &fval);
        }

        if retval < 0 {
            ret = KIN_SYSFUNC_FAIL;
            break;
        }

        /* compute new solution */
        let (m_aa, nni, delay_aa) = {
            let m = kin_mem.borrow();
            (m.kin_m_aa, m.kin_nni, m.kin_delay_aa)
        };
        if m_aa == 0 || nni - 1 < delay_aa {
            let (damping, has_damping_fn) = {
                let m = kin_mem.borrow();
                (m.kin_damping, m.kin_damping_fn.is_some())
            };
            if damping || has_damping_fn {
                if has_damping_fn {
                    let uu = get_uu(kin_mem);
                    let fval = get_fval(kin_mem);
                    retval = kin_call_damping_fn(kin_mem, nni, &uu, &fval, None, 0, false);
                    if retval != 0 {
                        KINProcessError(
                            Some(kin_mem),
                            KIN_DAMPING_FN_ERR,
                            line!() as i32,
                            "KINPicardAA",
                            file!(),
                            "The damping function failed.",
                        );
                        ret = KIN_DAMPING_FN_ERR;
                        break;
                    }
                    let bad_beta = {
                        let m = kin_mem.borrow();
                        m.kin_beta <= ZERO || m.kin_beta > ONE
                    };
                    if bad_beta {
                        KINProcessError(
                            Some(kin_mem),
                            KIN_DAMPING_FN_ERR,
                            line!() as i32,
                            "KINPicardAA",
                            file!(),
                            "The damping parameter is outside of the range (0, 1].",
                        );
                        ret = KIN_DAMPING_FN_ERR;
                        break;
                    }
                }

                /* damped fixed point */
                let beta = kin_mem.borrow().kin_beta;
                let uu = get_uu(kin_mem);
                let gval = get_gval(kin_mem);
                let unew = get_unew(kin_mem);
                N_VLinearSum(ONE - beta, &uu, beta, &gval, &unew);
            } else {
                /* standard fixed point */
                let gval = get_gval(kin_mem);
                let unew = get_unew(kin_mem);
                N_VScale(ONE, &gval, &unew);
            }
        } else {
            /* compute iteration count for Anderson acceleration */
            let iter_aa: i64;
            if delay_aa > 0 {
                iter_aa = nni - 1 - delay_aa;
            } else {
                iter_aa = nni - 1;
            }

            let gval = get_gval(kin_mem);
            let unew = get_unew(kin_mem);
            let uu = get_uu(kin_mem);
            /* C passes kin_mem->kin_R_aa / kin_mem->kin_gamma_aa directly;
            move them out of the mem around the call so AndersonAcc can
            take its own granular borrows, then hand them back. */
            let mut R = std::mem::take(&mut kin_mem.borrow_mut().kin_R_aa);
            let mut gamma = std::mem::take(&mut kin_mem.borrow_mut().kin_gamma_aa);
            retval = AndersonAcc(
                kin_mem,   /* kinsol memory            */
                &gval,     /* G(u_cur)       in        */
                &delta,    /* F(u_cur)       in (temp) */
                &unew,     /* u_new output   out       */
                &uu,       /* u_cur input    in        */
                iter_aa,   /* AA iteration   in        */
                &mut R,    /* R matrix       in/out    */
                &mut gamma, /* gamma vector   in (temp) */
            );
            {
                let mut m = kin_mem.borrow_mut();
                m.kin_R_aa = R;
                m.kin_gamma_aa = gamma;
            }
            if retval != 0 {
                ret = retval;
                break;
            }
        }

        /* Fill the Newton residual based on the new solution iterate */
        let unew = get_unew(kin_mem);
        let fval = get_fval(kin_mem);
        retval = kin_call_func(kin_mem, &unew, &fval);
        kin_mem.borrow_mut().kin_nfe += 1;

        if retval < 0 {
            ret = KIN_SYSFUNC_FAIL;
            break;
        }

        /* Measure || F(x) ||_max */
        {
            let fscale = get_fscale(kin_mem);
            let fnorm = KINScFNorm(kin_mem, &fval, &fscale);
            kin_mem.borrow_mut().kin_fnorm = fnorm;
        }

        /* KINPrintInfo(kin_mem, PRNT_FMAX, "KINSOL", __func__, INFO_FMAX,
        kin_fnorm) — omitted (logging level < INFO) */

        /* print the current iter, fnorm, and nfe values:
        KINPrintInfo(kin_mem, PRNT_NNI, ...) — omitted (logging level < INFO) */

        /* Check if the maximum number of iterations is reached */
        {
            let m = kin_mem.borrow();
            if m.kin_nni >= m.kin_mxiter {
                ret = KIN_MAXITER_REACHED;
            }
        }
        {
            let m = kin_mem.borrow();
            if m.kin_fnorm <= m.kin_fnormtol {
                ret = KIN_SUCCESS;
            }
        }

        /* Update the solution. Always return the newest iteration. Note this is
           also consistent with last function evaluation. */
        {
            let unew = get_unew(kin_mem);
            let uu = get_uu(kin_mem);
            N_VScale(ONE, &unew, &uu);
        }

        if ret == CONTINUE_ITERATIONS && kin_mem.borrow().kin_callForcingTerm {
            /* evaluate eta by calling the forcing term routine */
            let fval = get_fval(kin_mem);
            let fscale = get_fscale(kin_mem);
            let fnormp = N_VWL2Norm(&fval, &fscale);
            KINForcingTerm(kin_mem, fnormp);
        }
    } /* end of loop; return */

    /* KINPrintInfo(kin_mem, PRNT_RETVAL, "KINSOL", __func__, INFO_RETVAL, ret)
    — omitted (logging level < INFO) */

    ret
}

/*
 * KINPicardFcnEval
 *
 * This routine evaluates the Picard fixed point function
 * using the linear solver, gval = u - L^{-1}F(u).
 * The function assumes the user has defined L either through
 * a user-supplied matvec if using a SPILS solver or through
 * a supplied matrix if using a dense solver.  This assumption is
 * tested by a check on the strategy and the requisite functionality
 * within the linear solve routines.
 *
 * This routine fills gval = uu - L^{-1}F(uu) given uu and fval = F(uu).
 */

fn KINPicardFcnEval(kin_mem: &KINMem, gval: &N_Vector, uval: &N_Vector, fval1: &N_Vector) -> i32 {
    {
        let mut m = kin_mem.borrow_mut();
        if (m.kin_nni - m.kin_nnilset) >= m.kin_msbset {
            m.kin_sthrsh = TWO;
            m.kin_update_fnorm_sub = SUNTRUE;
        }
    }

    loop {
        kin_mem.borrow_mut().kin_jacCurrent = SUNFALSE;

        let (sthrsh, lsetup) = {
            let m = kin_mem.borrow();
            (m.kin_sthrsh, m.kin_lsetup)
        };
        if (sthrsh > ONEPT5) && lsetup.is_some() {
            let lsetup = lsetup.expect("kin_lsetup");
            let retval = lsetup(kin_mem);
            {
                let mut m = kin_mem.borrow_mut();
                m.kin_jacCurrent = SUNTRUE;
                m.kin_nnilset = m.kin_nni;
                m.kin_nnilset_sub = m.kin_nni;
            }
            if retval != 0 {
                return KIN_LSETUP_FAIL;
            }
        }

        /* call the generic 'lsolve' routine to solve the system Lx = -fval
           Note that we are using gval to hold x. */
        N_VScale(-ONE, fval1, fval1);

        /* C passes &(kin_mem->kin_sJpnorm) / &(kin_mem->kin_sFdotJp) */
        let lsolve = kin_mem.borrow().kin_lsolve.expect("kin_lsolve");
        let (mut sJpnorm, mut sFdotJp) = {
            let m = kin_mem.borrow();
            (m.kin_sJpnorm, m.kin_sFdotJp)
        };
        let retval = lsolve(kin_mem, gval, fval1, &mut sJpnorm, &mut sFdotJp);
        {
            let mut m = kin_mem.borrow_mut();
            m.kin_sJpnorm = sJpnorm;
            m.kin_sFdotJp = sFdotJp;
        }

        if retval == 0 {
            /* Update gval = uval + gval since gval = -L^{-1}F(uu)  */
            N_VLinearSum(ONE, uval, ONE, gval, gval);
            return KIN_SUCCESS;
        } else if retval < 0 {
            return KIN_LSOLVE_FAIL;
        } else {
            let m = kin_mem.borrow();
            if m.kin_lsetup.is_none() || m.kin_jacCurrent {
                return KIN_LINSOLV_NO_RECOVERY;
            }
        }

        /* loop back only if the linear solver setup is in use
           and matrix information is not current */

        kin_mem.borrow_mut().kin_sthrsh = TWO;
    }
}

/*
 * KINFP
 *
 * This routine is the main driver for the fixed point iteration with
 * Anderson Acceleration.
 */

fn KINFP(kin_mem: &KINMem) -> i32 {
    let mut retval: i32; /* return value from user func */
    let mut ret: i32; /* iteration status            */
    let mut tolfac: sunrealtype; /* tolerance adjustment factor */
    /* C also declares `long int iter_aa` here; it is single-assignment and
    is declared at its use site below. */

    let delta = get_vtemp1(kin_mem); /* temporary workspace vector  */
    ret = CONTINUE_ITERATIONS;
    /* C: `tolfac = ONE;` here — a provably-dead store (every path through
    the loop body assigns tolfac before the tolerance test reads it), so
    it is dropped to keep the build warning-free. */

    /* SUNLogExtraDebugVec(KIN_LOGGER, "begin", kin_uu, "u_0(:) =")
    — omitted (logging level < EXTRA_DEBUG) */

    /* initialize iteration count */
    kin_mem.borrow_mut().kin_nni = 0;

    while ret == CONTINUE_ITERATIONS {
        /* update iteration count */
        kin_mem.borrow_mut().kin_nni += 1;

        /* evaluate func(uu) and return if failed */
        let uu = get_uu(kin_mem);
        let fval = get_fval(kin_mem);
        retval = kin_call_func(kin_mem, &uu, &fval);
        kin_mem.borrow_mut().kin_nfe += 1;

        /* SUNLogExtraDebugVec(... "G_%ld(:) =", kin_nni - 1) — omitted */

        if retval < 0 {
            ret = KIN_SYSFUNC_FAIL;
            break;
        }

        /* compute new solution */
        let (m_aa, nni, delay_aa) = {
            let m = kin_mem.borrow();
            (m.kin_m_aa, m.kin_nni, m.kin_delay_aa)
        };
        if m_aa == 0 || nni - 1 < delay_aa {
            let (damping, has_damping_fn) = {
                let m = kin_mem.borrow();
                (m.kin_damping, m.kin_damping_fn.is_some())
            };
            if damping || has_damping_fn {
                if has_damping_fn {
                    let uu = get_uu(kin_mem);
                    let fval = get_fval(kin_mem);
                    retval = kin_call_damping_fn(kin_mem, nni, &uu, &fval, None, 0, false);
                    if retval != 0 {
                        KINProcessError(
                            Some(kin_mem),
                            KIN_DAMPING_FN_ERR,
                            line!() as i32,
                            "KINFP",
                            file!(),
                            "The damping function failed.",
                        );
                        ret = KIN_DAMPING_FN_ERR;
                        break;
                    }
                    let bad_beta = {
                        let m = kin_mem.borrow();
                        m.kin_beta <= ZERO || m.kin_beta > ONE
                    };
                    if bad_beta {
                        KINProcessError(
                            Some(kin_mem),
                            KIN_DAMPING_FN_ERR,
                            line!() as i32,
                            "KINFP",
                            file!(),
                            "The damping parameter is outside of the range (0, 1].",
                        );
                        ret = KIN_DAMPING_FN_ERR;
                        break;
                    }
                }

                /* damped fixed point */
                let beta = kin_mem.borrow().kin_beta;
                let uu = get_uu(kin_mem);
                let fval = get_fval(kin_mem);
                let unew = get_unew(kin_mem);
                N_VLinearSum(ONE - beta, &uu, beta, &fval, &unew);

                /* tolerance adjustment */
                tolfac = beta;
            } else {
                /* standard fixed point */
                let fval = get_fval(kin_mem);
                let unew = get_unew(kin_mem);
                N_VScale(ONE, &fval, &unew);

                /* tolerance adjustment */
                tolfac = ONE;
            }
        } else {
            /* compute iteration count for Anderson acceleration */
            let iter_aa: i64;
            if delay_aa > 0 {
                iter_aa = nni - 1 - delay_aa;
            } else {
                iter_aa = nni - 1;
            }

            /* apply Anderson acceleration */
            let fval = get_fval(kin_mem);
            let unew = get_unew(kin_mem);
            let uu = get_uu(kin_mem);
            let mut R = std::mem::take(&mut kin_mem.borrow_mut().kin_R_aa);
            let mut gamma = std::mem::take(&mut kin_mem.borrow_mut().kin_gamma_aa);
            retval = AndersonAcc(
                kin_mem, &fval, &delta, &unew, &uu, iter_aa, &mut R, &mut gamma,
            );
            {
                let mut m = kin_mem.borrow_mut();
                m.kin_R_aa = R;
                m.kin_gamma_aa = gamma;
            }
            if retval != 0 {
                ret = retval;
                break;
            }

            /* tolerance adjustment (first iteration is standard fixed point) */
            let (damping_aa, has_damping_fn) = {
                let m = kin_mem.borrow();
                (m.kin_damping_aa, m.kin_damping_fn.is_some())
            };
            if iter_aa == 0 && (damping_aa || has_damping_fn) {
                tolfac = kin_mem.borrow().kin_beta;
            } else {
                tolfac = ONE;
            }
        }

        /* SUNLogExtraDebugVec(... "u_%ld(:) =", kin_nni) — omitted */

        /* compute change between iterations */
        {
            let unew = get_unew(kin_mem);
            let uu = get_uu(kin_mem);
            N_VLinearSum(ONE, &unew, -ONE, &uu, &delta);
        }

        /* measure || g(x) - x || */
        {
            let fscale = get_fscale(kin_mem);
            let fnorm = KINScFNorm(kin_mem, &delta, &fscale);
            kin_mem.borrow_mut().kin_fnorm = fnorm;
        }

        /* KINPrintInfo(kin_mem, PRNT_FMAX, ...) — omitted (level < INFO) */

        /* print the current iter, fnorm, and nfe values:
        KINPrintInfo(kin_mem, PRNT_NNI, ...) — omitted (level < INFO) */

        /* Check if the maximum number of iterations is reached */
        {
            let m = kin_mem.borrow();
            if m.kin_nni >= m.kin_mxiter {
                ret = KIN_MAXITER_REACHED;
            }
        }
        {
            let m = kin_mem.borrow();
            if m.kin_fnorm <= (tolfac * m.kin_fnormtol) {
                ret = KIN_SUCCESS;
            }
        }

        /* Update the solution if taking another iteration or returning the newest
           iterate. Otherwise return the solution consistent with the last function
           evaluation. */
        if ret == CONTINUE_ITERATIONS || kin_mem.borrow().kin_ret_newest {
            let unew = get_unew(kin_mem);
            let uu = get_uu(kin_mem);
            N_VScale(ONE, &unew, &uu);
        }
    } /* end of loop; return */

    /* KINPrintInfo(kin_mem, PRNT_RETVAL, "KINSOL", __func__, INFO_RETVAL, ret)
    — omitted (logging level < INFO) */

    ret
}

/*
 * ========================================================================
 * Anderson Acceleration
 * ========================================================================
 */

fn AndersonAccQRDelete(kin_mem: &KINMem, Q: &[N_Vector], R: &mut [sunrealtype], depth: i32) -> i32 {
    /* Delete left-most column vector from QR factorization.
    C declares a, b, temp, c, s at the top of the function; they are
    per-iteration values here so they are declared inside the loop. */

    let vtemp2 = get_vtemp2(kin_mem);
    let d = depth as usize;

    for i in 0..(depth - 1) {
        let iu = i as usize;
        let mut a = R[(iu + 1) * d + iu];
        let mut b = R[(iu + 1) * d + iu + 1];
        let mut temp = SUNRsqrt(a * a + b * b);
        let c = a / temp;
        let s = b / temp;
        R[(iu + 1) * d + iu] = temp;
        R[(iu + 1) * d + iu + 1] = ZERO;
        /* OK to reuse temp */
        if i < depth - 1 {
            for j in (i + 2)..depth {
                let ju = j as usize;
                a = R[ju * d + iu];
                b = R[ju * d + iu + 1];
                temp = c * a + s * b;
                R[ju * d + iu + 1] = -s * a + c * b;
                R[ju * d + iu] = temp;
            }
        }
        N_VLinearSum(c, &Q[iu], s, &Q[iu + 1], &vtemp2);
        N_VLinearSum(-s, &Q[iu], c, &Q[iu + 1], &Q[iu + 1]);
        N_VScale(ONE, &vtemp2, &Q[iu]);
    }

    /* Shift R to the left by one. */
    for i in 1..depth {
        for j in 0..(depth - 1) {
            R[(i as usize - 1) * d + j as usize] = R[i as usize * d + j as usize];
        }
    }

    /* If ICWY orthogonalization, then update T */
    if kin_mem.borrow().kin_orth_aa == KIN_ORTH_ICWY {
        /* kin_T_aa is the storage C shares with `qr_data->temp_array`;
        see the note at the kin_qr_func call site in AndersonAcc. */
        let mut T_aa = std::mem::take(&mut kin_mem.borrow_mut().kin_T_aa);

        if kin_mem.borrow().kin_dot_prod_sb {
            if depth > 1 {
                for i in 2..depth {
                    let iu = i as usize;
                    let _ = N_VDotProdMultiLocal(i, &Q[iu - 1], Q, &mut T_aa[(iu - 1) * d..]);
                }
                let _ = N_VDotProdMultiAllReduce(depth * depth, &Q[d - 1], &mut T_aa);
            }
            for i in 1..depth {
                let iu = i as usize;
                T_aa[(iu - 1) * d + (iu - 1)] = ONE;
            }
        } else {
            T_aa[0] = ONE;
            for i in 2..depth {
                let iu = i as usize;
                let _ = N_VDotProdMulti(i - 1, &Q[iu - 1], Q, &mut T_aa[(iu - 1) * d..]);
                T_aa[(iu - 1) * d + (iu - 1)] = ONE;
            }
        }

        kin_mem.borrow_mut().kin_T_aa = T_aa;
    }

    KIN_SUCCESS
}

/// C `AndersonAcc`. The `cv`/`Xv` fused-operation scratch arrays are
/// shortcuts onto `kin_mem->kin_cv` / `kin_mem->kin_Xv` in C; they are
/// moved out of the mem here and handed back on every return path (this
/// wrapper is the single exit point) so the body can take granular
/// borrows freely.
fn AndersonAcc(
    kin_mem: &KINMem,
    gval: &N_Vector,
    fv: &N_Vector,
    x: &N_Vector,
    xold: &N_Vector,
    iter: i64,
    R: &mut [sunrealtype],
    gamma: &mut [sunrealtype],
) -> i32 {
    /* local shortcuts for fused vector operation */
    let mut cv = std::mem::take(&mut kin_mem.borrow_mut().kin_cv);
    let mut Xv = std::mem::take(&mut kin_mem.borrow_mut().kin_Xv);

    let ret = andersonAccBody(
        kin_mem, gval, fv, x, xold, iter, R, gamma, &mut cv, &mut Xv,
    );

    {
        let mut m = kin_mem.borrow_mut();
        m.kin_cv = cv;
        m.kin_Xv = Xv;
    }

    ret
}

fn andersonAccBody(
    kin_mem: &KINMem,
    gval: &N_Vector,
    fv: &N_Vector,
    x: &N_Vector,
    xold: &N_Vector,
    iter: i64,
    R: &mut [sunrealtype],
    gamma: &mut [sunrealtype],
    cv: &mut Vec<sunrealtype>,
    Xv: &mut Vec<N_Vector>,
) -> i32 {
    let mut retval: i32;
    /* C also declares `long int lAA`, `sunrealtype alfa` and
    `sunrealtype onembeta` at the top; each is single-assignment and is
    declared at its use site below. */

    /* Compute residual F(x) = G(x_old) - x_old */
    N_VLinearSum(ONE, gval, -ONE, xold, fv);

    if iter > 0 {
        /* If we've filled the acceleration subspace, start recycling */
        let (current_depth, m_aa) = {
            let m = kin_mem.borrow();
            (m.kin_current_depth, m.kin_m_aa)
        };
        if current_depth == m_aa {
            /* Move the left-most column vector (oldest value) to the end so it gets
               overwritten with the newest value below. */
            {
                let mut m = kin_mem.borrow_mut();
                let tmp_dg = m.kin_dg_aa[0].clone();
                let tmp_df = m.kin_df_aa[0].clone();
                for i in 1..m_aa as usize {
                    m.kin_dg_aa[i - 1] = m.kin_dg_aa[i].clone();
                    m.kin_df_aa[i - 1] = m.kin_df_aa[i].clone();
                }
                let last = m_aa as usize - 1;
                m.kin_dg_aa[last] = tmp_dg;
                m.kin_df_aa[last] = tmp_df;
            }

            /* Delete left-most column vector from QR factorization */
            let q_aa: Vec<N_Vector> = kin_mem.borrow().kin_q_aa.clone();
            retval = AndersonAccQRDelete(kin_mem, &q_aa, R, m_aa as i32);
            if retval != 0 {
                return retval;
            }

            kin_mem.borrow_mut().kin_current_depth -= 1;
        }

        let current_depth = kin_mem.borrow().kin_current_depth;

        /* compute dg_new = gval - gval_old */
        {
            let gold_aa = get_gold_aa(kin_mem);
            let dg = get_dg_aa(kin_mem, current_depth as usize);
            N_VLinearSum(ONE, gval, -ONE, &gold_aa, &dg);
        }

        /* compute df_new = fval - fval_old */
        {
            let fold_aa = get_fold_aa(kin_mem);
            let df = get_df_aa(kin_mem, current_depth as usize);
            N_VLinearSum(ONE, fv, -ONE, &fold_aa, &df);
        }

        kin_mem.borrow_mut().kin_current_depth += 1;
    }

    /* KINPrintInfo(kin_mem, PRNT_OTHER, "KINSOL", __func__, "current_depth = %i",
    kin_current_depth) — omitted (logging level < INFO) */

    {
        let gold_aa = get_gold_aa(kin_mem);
        N_VScale(ONE, gval, &gold_aa);
    }
    {
        let fold_aa = get_fold_aa(kin_mem);
        N_VScale(ONE, fv, &fold_aa);
    }

    /* on first iteration, do fixed point update */
    if kin_mem.borrow().kin_current_depth == 0 {
        let (damping_aa, has_damping_fn) = {
            let m = kin_mem.borrow();
            (m.kin_damping_aa, m.kin_damping_fn.is_some())
        };
        if damping_aa || has_damping_fn {
            if has_damping_fn {
                let nni = kin_mem.borrow().kin_nni;
                retval = kin_call_damping_fn(kin_mem, nni, xold, gval, None, 0, true);
                if retval != 0 {
                    KINProcessError(
                        Some(kin_mem),
                        KIN_DAMPING_FN_ERR,
                        line!() as i32,
                        "AndersonAcc",
                        file!(),
                        "The damping function failed.",
                    );
                    return KIN_DAMPING_FN_ERR;
                }
                let bad_beta = {
                    let m = kin_mem.borrow();
                    m.kin_beta_aa <= ZERO || m.kin_beta_aa > ONE
                };
                if bad_beta {
                    KINProcessError(
                        Some(kin_mem),
                        KIN_DAMPING_FN_ERR,
                        line!() as i32,
                        "AndersonAcc",
                        file!(),
                        "The damping parameter is outside of the range (0, 1].",
                    );
                    return KIN_DAMPING_FN_ERR;
                }
            }

            /* damped fixed point */
            let beta_aa = kin_mem.borrow().kin_beta_aa;
            N_VLinearSum(ONE - beta_aa, xold, beta_aa, gval, x);
        } else {
            /* standard fixed point */
            N_VScale(ONE, gval, x);
        }

        return KIN_SUCCESS;
    }

    /* Add a column to the QR factorization */

    if kin_mem.borrow().kin_current_depth == 1 {
        let df0 = get_df_aa(kin_mem, 0);
        let q0 = kin_mem.borrow().kin_q_aa[0].clone();
        R[0] = SUNRsqrt(N_VDotProd(&df0, &df0));
        let alfa = ONE / R[0];
        N_VScale(alfa, &df0, &q0);
    } else {
        let (current_depth, m_aa) = {
            let m = kin_mem.borrow();
            (m.kin_current_depth, m.kin_m_aa)
        };
        let q_aa: Vec<N_Vector> = kin_mem.borrow().kin_q_aa.clone();
        let df = get_df_aa(kin_mem, current_depth as usize - 1);

        /* C calls through `kin_mem->kin_qr_func` here and discards the
        return value. `kinQRAdd` is that call: it owns the `temp_array`
        aliasing contract (C aliases `qr_data->temp_array` with
        `kin_mem->kin_T_aa` for ICWY — a T matrix that persists across
        calls and that AndersonAccQRDelete also updates — and with
        `kin_mem->kin_cv` for CGS2/DCGS2, pure per-call scratch; see the
        `kinsol_orth` module notes), so that contract lives in exactly
        one place. */
        let _ = kinQRAdd(
            kin_mem,
            &q_aa,
            R,
            &df,
            current_depth as i32 - 1,
            m_aa as i32,
        );
    }

    /* Adjust the depth */
    if kin_mem.borrow().kin_depth_fn.is_some() {
        let depth_fn = kin_mem.borrow().kin_depth_fn.expect("kin_depth_fn");
        let mut new_depth: i64 = kin_mem.borrow().kin_current_depth;

        let current_depth = kin_mem.borrow().kin_current_depth;
        let nni = kin_mem.borrow().kin_nni;
        let df_aa: Vec<N_Vector> = kin_mem.borrow().kin_df_aa.clone();
        let mut user_data = kin_mem.borrow_mut().kin_user_data.take();
        retval = depth_fn(
            nni,
            xold,
            gval,
            fv,
            &df_aa,
            R,
            current_depth,
            &mut user_data,
            &mut new_depth,
            None,
        );
        kin_mem.borrow_mut().kin_user_data = user_data;
        if retval != 0 {
            KINProcessError(
                Some(kin_mem),
                KIN_DEPTH_FN_ERR,
                line!() as i32,
                "AndersonAcc",
                file!(),
                "The depth function failed.",
            );
            return KIN_DEPTH_FN_ERR;
        }

        new_depth = SUNMIN(new_depth, kin_mem.borrow().kin_current_depth);
        new_depth = SUNMAX(new_depth, 0);

        /* KINPrintInfo(kin_mem, PRNT_OTHER, "KINSOL", __func__,
        "new_depth = %i", new_depth) — omitted (logging level < INFO) */

        if new_depth == 0 {
            kin_mem.borrow_mut().kin_current_depth = new_depth;

            /* do fixed point update */
            let (damping_aa, has_damping_fn) = {
                let m = kin_mem.borrow();
                (m.kin_damping_aa, m.kin_damping_fn.is_some())
            };
            if damping_aa || has_damping_fn {
                if has_damping_fn {
                    let nni = kin_mem.borrow().kin_nni;
                    retval = kin_call_damping_fn(kin_mem, nni, xold, gval, None, 0, true);
                    if retval != 0 {
                        KINProcessError(
                            Some(kin_mem),
                            KIN_DAMPING_FN_ERR,
                            line!() as i32,
                            "AndersonAcc",
                            file!(),
                            "The damping function failed.",
                        );
                        return KIN_DAMPING_FN_ERR;
                    }
                    let bad_beta = {
                        let m = kin_mem.borrow();
                        m.kin_beta_aa <= ZERO || m.kin_beta_aa > ONE
                    };
                    if bad_beta {
                        KINProcessError(
                            Some(kin_mem),
                            KIN_DAMPING_FN_ERR,
                            line!() as i32,
                            "AndersonAcc",
                            file!(),
                            "The damping parameter is outside of the range (0, 1].",
                        );
                        return KIN_DAMPING_FN_ERR;
                    }
                }

                /* damped fixed point */
                let beta_aa = kin_mem.borrow().kin_beta_aa;
                N_VLinearSum(ONE - beta_aa, xold, beta_aa, gval, x);
            } else {
                /* standard fixed point */
                N_VScale(ONE, gval, x);
            }

            return KIN_SUCCESS;
        }

        /* TODO(DJG): In the future, update QRDelete to support removing arbitrary
           columns from the factorization */
        if new_depth < kin_mem.borrow().kin_current_depth {
            /* Remove columns from the left one at a time.
            NOTE: C's loop bound `j < kin_mem->kin_current_depth - new_depth`
            is re-evaluated every iteration while the body decrements
            kin_current_depth, so the loop runs ceil((depth - new_depth)/2)
            times, not (depth - new_depth) times. Transcribed literally. */
            let mut j: i64 = 0;
            while j < kin_mem.borrow().kin_current_depth - new_depth {
                let current_depth = kin_mem.borrow().kin_current_depth;
                {
                    let mut m = kin_mem.borrow_mut();
                    let tmp_dg = m.kin_dg_aa[0].clone();
                    let tmp_df = m.kin_df_aa[0].clone();
                    for i in 1..current_depth as usize {
                        m.kin_dg_aa[i - 1] = m.kin_dg_aa[i].clone();
                        m.kin_df_aa[i - 1] = m.kin_df_aa[i].clone();
                    }
                    let last = current_depth as usize - 1;
                    m.kin_dg_aa[last] = tmp_dg;
                    m.kin_df_aa[last] = tmp_df;
                }

                let q_aa: Vec<N_Vector> = kin_mem.borrow().kin_q_aa.clone();
                retval = AndersonAccQRDelete(kin_mem, &q_aa, R, current_depth as i32);
                if retval != 0 {
                    return retval;
                }

                kin_mem.borrow_mut().kin_current_depth -= 1;
                j += 1;
            }
        }
    }

    /* Solve least squares problem and update solution */
    let lAA: i64 = kin_mem.borrow().kin_current_depth;

    /* Compute Q^T fv */
    {
        let q_aa: Vec<N_Vector> = kin_mem.borrow().kin_q_aa.clone();
        retval = N_VDotProdMulti(lAA as i32, fv, &q_aa, gamma);
    }
    if retval != KIN_SUCCESS {
        return KIN_VECTOROP_ERR;
    }

    /* Compute the damping factor before overwriting gamma below so we can pass
       gamma = Q^T fv (just computed above) to the damping function as it can be
       used to compute the acceleration gain = sqrt(1 - ||Q^T fv||^2/||fv||^2). */
    if kin_mem.borrow().kin_damping_fn.is_some() {
        let nni = kin_mem.borrow().kin_nni;
        retval = kin_call_damping_fn(kin_mem, nni, xold, gval, Some(&mut *gamma), lAA, true);
        if retval != 0 {
            KINProcessError(
                Some(kin_mem),
                KIN_DAMPING_FN_ERR,
                line!() as i32,
                "AndersonAcc",
                file!(),
                "The damping function failed.",
            );
            return KIN_DAMPING_FN_ERR;
        }
        let bad_beta = {
            let m = kin_mem.borrow();
            m.kin_beta_aa <= ZERO || m.kin_beta_aa > ONE
        };
        if bad_beta {
            KINProcessError(
                Some(kin_mem),
                KIN_DAMPING_FN_ERR,
                line!() as i32,
                "AndersonAcc",
                file!(),
                "The damping parameter is outside of the range (0, 1].",
            );
            return KIN_DAMPING_FN_ERR;
        }
    }

    /* set arrays for fused vector operation.
    C writes cv[nvec] / Xv[nvec] into the malloc'd scratch arrays of size
    2*(m_aa+1); the Rust scratch Vecs are rebuilt by pushing in exactly the
    same order (they are write-only scratch, never read elsewhere), which
    keeps the indices in range whatever length the AA allocator chose. */
    cv.clear();
    Xv.clear();
    cv.push(ONE);
    Xv.push(gval.clone());
    let mut nvec: i32 = 1; /* C: declared `int nvec = 0` at function top */

    /* Solve the upper triangular system R gamma = Q^T fv */
    let m_aa = kin_mem.borrow().kin_m_aa;
    for i in (0..lAA).rev() {
        for j in (i + 1)..lAA {
            gamma[i as usize] =
                gamma[i as usize] - R[(j * m_aa + i) as usize] * gamma[j as usize];
        }
        gamma[i as usize] /= R[(i * m_aa + i) as usize];

        cv.push(-gamma[i as usize]);
        Xv.push(get_dg_aa(kin_mem, i as usize));
        nvec += 1;
    }

    /* if enabled, apply damping */
    let (damping_aa, has_damping_fn) = {
        let m = kin_mem.borrow();
        (m.kin_damping_aa, m.kin_damping_fn.is_some())
    };
    if damping_aa || has_damping_fn {
        let onembeta = ONE - kin_mem.borrow().kin_beta_aa;
        cv.push(-onembeta);
        Xv.push(fv.clone());
        nvec += 1;
        for i in (0..lAA).rev() {
            cv.push(onembeta * gamma[i as usize]);
            Xv.push(get_df_aa(kin_mem, i as usize));
            nvec += 1;
        }
    }

    /* update solution */
    retval = N_VLinearCombination(nvec, &cv[..], &Xv[..], x);
    if retval != KIN_SUCCESS {
        return KIN_VECTOROP_ERR;
    }

    KIN_SUCCESS
}
