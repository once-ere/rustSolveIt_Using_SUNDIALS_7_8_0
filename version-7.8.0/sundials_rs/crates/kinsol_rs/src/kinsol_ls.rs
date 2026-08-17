//! Port of `src/kinsol/kinsol_ls.c` (+ `src/kinsol/kinsol_ls_impl.h` and
//! `include/kinsol/kinsol_ls.h` folded).
//!
//! KINSOL's linear solver interface (KINLS): attaches a generic
//! `SUNLinearSolver` (and optional `SUNMatrix`) to KINSOL, provides the
//! `kin_linit`/`kin_lsetup`/`kin_lsolve`/`kin_lfree` solver hooks, the
//! difference-quotient dense/band Jacobians and J*v product, and the
//! ATimes/PSetup/PSolve trampolines registered with the LS.
//!
//! Storage model. `kin_mem.kin_lmem` is `Option<Box<dyn Any>>` holding a
//! [`KINLsMemRec`] BY VALUE; [`kinls_mem_mut`] is the one accessor
//! (a `RefMut::map` over `kin_mem.borrow_mut()`). The returned guard IS a
//! borrow of the mem: never hold it across `KINProcessError`, a user
//! callback, a vector/matrix/LS operation, or a second borrow of the same
//! mem. Modules that attach an internal preconditioner (`kinsol_bbdpre`)
//! reach `pdata` through the same accessor and must probe attachment
//! non-panickingly first, e.g.
//! `kin_mem.borrow().kin_lmem.as_ref().is_some_and(|b| b.is::<KINLsMemRec>())`,
//! before returning `KINLS_LMEM_NULL`.
//!
//! Data-token model (C `void*` fields `J_data`/`pdata`/`jt_data`): in C
//! each field holds either `kin_mem` (internal routine) or
//! `kin_mem->kin_user_data` (user routine). Here the field is
//! `Option<Box<dyn Any>>`: `Some(box)` is a module-owned token (a `KINMem`
//! clone for the internal KINLS routines, or whatever an internal
//! preconditioner module stored), while `None` means "pass the solver's
//! `kin_user_data`" — the invoker `Option::take`s the corresponding box
//! around the callback and restores it on EVERY path (success, early
//! return, error). This reproduces the C pointer aliasing without double
//! ownership; the only divergence is that a C snapshot of a *stale*
//! `kin_user_data` cannot occur — the current `kin_user_data` is always
//! passed. For `J_data`/`jt_data` that matches C exactly
//! (`kinLsInitialize`'s "reset just in case" assignments refresh them);
//! for `pdata` C keeps the `KINSetLinearSolver`-time snapshot forever, so
//! a `KINSetUserData` call AFTER `KINSetLinearSolver` diverges (accepted
//! deviation class 6, see ARCHITECTURE.md).
//!
//! Upstream quirk preserved verbatim: `KINSetLinearSolver` computes the
//! LOCAL variables `iterative`/`matrixbased` for its compatibility checks
//! but NEVER assigns the same-named `KINLsMemRec` fields, which therefore
//! keep their `memset(0)` value `SUNFALSE` for the entire solve (grep
//! confirms no other kinsol source writes them). Two sites read them:
//! `kinLsInitialize`'s `tol_fac` branch (so `tol_fac` is always `ONE`;
//! unobservable anyway because every SUNDIALS iterative `SUNLinearSolver`
//! implements `setscalingvectors`, which makes the second conjunct false
//! too) and a `KINPrintInfo` block that is compiled out at
//! `SUNDIALS_LOGGING_LEVEL` 2. The fields stay `SUNFALSE` here.
//!
//! Logging: both `KINPrintInfo` call sites in `kinLsSolve` sit inside
//! `#if SUNDIALS_LOGGING_LEVEL >= SUNDIALS_LOGGING_INFO` and so do not
//! exist in the reference build (level 2 = WARNING); they are omitted at
//! translation time. The module-local keys/formats they would use
//! (`PRNT_NLI`, `PRNT_EPS`, [`INFO_NLI`], [`INFO_EPS`]) are still defined
//! below because `kinsol_ls_impl.h` declares them.

use std::any::Any;
use std::cell::RefMut;

use crate::kinsol_impl::*;
use sundials_core::sundials_errors::{
    SUN_ERR_ARG_CORRUPT, SUN_ERR_ARG_INCOMPATIBLE, SUN_ERR_EXT_FAIL, SUN_ERR_MEM_FAIL, SUN_SUCCESS,
};
use sundials_core::sundials_linearsolver::*;
use sundials_core::sundials_math::{SUNMAX, SUNMIN, SUNRabs, SUNRsqrt};
use sundials_core::sundials_matrix::*;
use sundials_core::sundials_nvector::*;
use sundials_core::sundials_types::*;
use sundials_core::sundials_utils::sun_format_g;
use sundials_core::sunmatrix_band::{
    SM_COLUMN_ELEMENT_IDX, SUNBandMatrix_Column, SUNBandMatrix_Columns,
    SUNBandMatrix_LowerBandwidth, SUNBandMatrix_StoredUpperBandwidth, SUNBandMatrix_UpperBandwidth,
};
use sundials_core::sunmatrix_dense::{SUNDenseMatrix_Column, SUNDenseMatrix_Columns};

/* constants */
const ZERO: sunrealtype = 0.0;
const ONE: sunrealtype = 1.0;

/*==================================================================
  KINLS Constants (include/kinsol/kinsol_ls.h)
  ==================================================================*/

pub const KINLS_SUCCESS: i32 = 0;

pub const KINLS_MEM_NULL: i32 = -1;
pub const KINLS_LMEM_NULL: i32 = -2;
pub const KINLS_ILL_INPUT: i32 = -3;
pub const KINLS_MEM_FAIL: i32 = -4;
pub const KINLS_PMEM_NULL: i32 = -5;
pub const KINLS_JACFUNC_ERR: i32 = -6;
pub const KINLS_SUNMAT_FAIL: i32 = -7;
pub const KINLS_SUNLS_FAIL: i32 = -8;

/*------------------------------------------------------------------
  keys for KINPrintInfo (kinsol_ls_impl.h; do not use 1 -> conflict
  with PRNT_RETVAL)
  ------------------------------------------------------------------*/

pub const PRNT_NLI: i32 = 101;
pub const PRNT_EPS: i32 = 102;

/*==================================================================
  KINLS user-supplied function prototypes
  (include/kinsol/kinsol_ls.h)
  ==================================================================*/

pub type KINLsJacFn = fn(
    u: &N_Vector,
    fu: &N_Vector,
    J: &SUNMatrix,
    user_data: &mut Option<Box<dyn Any>>,
    tmp1: &N_Vector,
    tmp2: &N_Vector,
) -> i32;

pub type KINLsPrecSetupFn = fn(
    uu: &N_Vector,
    uscale: &N_Vector,
    fval: &N_Vector,
    fscale: &N_Vector,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32;

pub type KINLsPrecSolveFn = fn(
    uu: &N_Vector,
    uscale: &N_Vector,
    fval: &N_Vector,
    fscale: &N_Vector,
    vv: &N_Vector,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32;

pub type KINLsJacTimesVecFn = fn(
    v: &N_Vector,
    Jv: &N_Vector,
    uu: &N_Vector,
    new_uu: &mut sunbooleantype,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32;

/*------------------------------------------------------------------
  Types : struct KINLsMemRec, KINLsMem (kinsol_ls_impl.h)
  ------------------------------------------------------------------*/

pub struct KINLsMemRec {
    /* Linear solver type information (NEVER assigned upstream — see the
    module docs; both keep the memset(0) value SUNFALSE) */
    pub iterative: sunbooleantype,   /* is the solver iterative?    */
    pub matrixbased: sunbooleantype, /* is a matrix structure used? */

    /* Jacobian construction & storage */
    pub jacDQ: sunbooleantype,   /* SUNTRUE if using internal DQ Jacobian approx. */
    pub jac: Option<KINLsJacFn>, /* Jacobian routine to be called                 */
    /* C `J_data`: `Some` = module-owned token (a KINMem clone for the
    internal DQ routine); `None` = pass `kin_user_data` at call time. */
    pub J_data: Option<Box<dyn Any>>,

    /* Linear solver, matrix and vector objects/pointers */
    pub LS: SUNLinearSolver,  /* generic iterative linear solver object        */
    pub J: Option<SUNMatrix>, /* problem Jacobian                              */

    /* Solver tolerance adjustment factor (if needed, see kinLsSolve)     */
    pub tol_fac: sunrealtype,

    /* Statistics and associated parameters */
    pub nje: i64,     /* no. of calls to jac                           */
    pub nfeDQ: i64,   /* no. of calls to F due to DQ Jacobian or J*v
                      approximations                                 */
    pub npe: i64,     /* npe = total number of precond calls           */
    pub nli: i64,     /* nli = total number of linear iterations       */
    pub nps: i64,     /* nps = total number of psolve calls            */
    pub ncfl: i64,    /* ncfl = total number of convergence failures   */
    pub njtimes: i64, /* njtimes = total number of calls to jtimes     */

    pub new_uu: sunbooleantype, /* flag indicating if the iterate has been
                                updated - the Jacobian must be updated or
                                reevaluated (meant to be used by a
                                user-supplied jtimes function                 */

    pub last_flag: i32, /* last error return flag                        */

    /* Preconditioner computation
       (a) user-provided:
           - pdata == user_data (here: `None` = pass kin_user_data)
           - pfree == NULL (the user dealocates memory)
       (b) internal preconditioner module
           - pdata == module token (`Some`)
           - pfree == set by the prec. module and called in kinLsFree */
    pub pset: Option<KINLsPrecSetupFn>,
    pub psolve: Option<KINLsPrecSolveFn>,
    pub pfree: Option<fn(kin_mem: &KINMem) -> i32>,
    pub pdata: Option<Box<dyn Any>>,

    /* Jacobian times vector computation
       (a) jtimes function provided by the user:
           - jt_data == user_data (here: `None`)
           - jtimesDQ == SUNFALSE
       (b) internal jtimes
           - jt_data == kin_mem token (`Some`)
           - jtimesDQ == SUNTRUE */
    pub jtimesDQ: sunbooleantype,
    pub jtimes: Option<KINLsJacTimesVecFn>,
    pub jt_func: Option<KINSysFn>,
    pub jt_data: Option<Box<dyn Any>>,
}

pub type KINLsMem = Box<KINLsMemRec>;

/// Downcast helper: view `kin_mem.kin_lmem` as the KINLS memory record.
/// Panics if no linear solver memory is attached or it is not a KINLS
/// record (the C code would blindly cast the `void*` — UB → panic).
/// NEVER hold the returned guard across `KINProcessError`, a callback, an
/// N_Vector op on a user-visible vector, a SUNLinearSolver/SUNMatrix
/// call, or another borrow of the same mem.
pub fn kinls_mem_mut(kin_mem: &KINMem) -> RefMut<'_, KINLsMemRec> {
    RefMut::map(kin_mem.borrow_mut(), |m| {
        m.kin_lmem
            .as_mut()
            .expect("kin_lmem set")
            .downcast_mut::<KINLsMemRec>()
            .expect("KINLS linear solver memory")
    })
}

/*------------------------------------------------------------------
  Error messages (kinsol_ls_impl.h)
  ------------------------------------------------------------------*/

pub const MSG_LS_KINMEM_NULL: &str = "KINSOL memory is NULL.";
pub const MSG_LS_MEM_FAIL: &str = "A memory request failed.";
pub const MSG_LS_BAD_NVECTOR: &str = "A required vector operation is not implemented.";
pub const MSG_LS_LMEM_NULL: &str = "Linear solver memory is NULL.";
pub const MSG_LS_NEG_MAXRS: &str = "maxrs < 0 illegal.";
pub const MSG_LS_BAD_SIZES: &str =
    "Illegal bandwidth parameter(s). Must have 0 <=  ml, mu <= N-1.";

pub const MSG_LS_JACFUNC_FAILED: &str = "The Jacobian routine failed in an unrecoverable manner.";
pub const MSG_LS_PSET_FAILED: &str =
    "The preconditioner setup routine failed in an unrecoverable manner.";
pub const MSG_LS_PSOLVE_FAILED: &str =
    "The preconditioner solve routine failed in an unrecoverable manner.";
pub const MSG_LS_JTIMES_FAILED: &str =
    "The Jacobian x vector routine failed in an unrecoverable manner.";
pub const MSG_LS_MATZERO_FAILED: &str = "The SUNMatZero routine failed in an unrecoverable manner.";

/*------------------------------------------------------------------
  Info messages (kinsol_ls_impl.h). Both builders exist only for the
  `KINPrintInfo` call sites that the reference logging level compiles
  out (see the module docs).
  ------------------------------------------------------------------*/

/* INFO_NLI: "nli_inc = %d" */
pub fn INFO_NLI(nli_inc: i32) -> String {
    format!("nli_inc = {}", nli_inc)
}

/* INFO_EPS: "residual norm = " SUN_FORMAT_G "  eps = " SUN_FORMAT_G */
pub fn INFO_EPS(res_norm: sunrealtype, eps: sunrealtype) -> String {
    format!(
        "residual norm = {}  eps = {}",
        sun_format_g(res_norm),
        sun_format_g(eps)
    )
}

/*==================================================================
  KINLS Exported functions -- Required
  ==================================================================*/

/*---------------------------------------------------------------
  KINSetLinearSolver specifies the linear solver
  ---------------------------------------------------------------*/
pub fn KINSetLinearSolver(kinmem: &KINMem, LS: &SUNLinearSolver, A: Option<&SUNMatrix>) -> i32 {
    /* NULL-kinmem check: handled by type system */
    /* NULL-LS check: handled by type system */
    let kin_mem = kinmem;

    /* Test if solver is compatible with LS interface */
    {
        let ops = LS.ops.borrow();
        if ops.gettype.is_none() || ops.solve.is_none() {
            KINProcessError(
                Some(kin_mem),
                KINLS_ILL_INPUT,
                line!() as i32,
                "KINSetLinearSolver",
                file!(),
                "LS object is missing a required operation",
            );
            return KINLS_ILL_INPUT;
        }
    }

    /* Retrieve the LS type */
    let LSType = SUNLinSolGetType(LS);

    /* Return with error if LS has 'matrix-embedded' type */
    if LSType == SUNLINEARSOLVER_MATRIX_EMBEDDED {
        KINProcessError(
            Some(kin_mem),
            KINLS_ILL_INPUT,
            line!() as i32,
            "KINSetLinearSolver",
            file!(),
            "KINSOL is incompatible with MATRIX_EMBEDDED LS objects",
        );
        return KINLS_ILL_INPUT;
    }

    /* Set flags based on LS type. NOTE (upstream): these locals are used
    for the checks below and for kin_inexact_ls, but the identically
    named KINLsMemRec fields are never assigned — see the module docs. */
    let iterative = LSType != SUNLINEARSOLVER_DIRECT;
    let matrixbased = LSType != SUNLINEARSOLVER_ITERATIVE;

    /* check for required vector operations for KINLS interface */
    let vtemp1 = kin_mem
        .borrow()
        .kin_vtemp1
        .as_ref()
        .expect("kin_vtemp1") /* C dereferences unconditionally (UB if unset) */
        .clone();
    {
        let ops = vtemp1.ops.borrow();
        if ops.nvconst.is_none() || ops.nvdotprod.is_none() {
            KINProcessError(
                Some(kin_mem),
                KINLS_ILL_INPUT,
                line!() as i32,
                "KINSetLinearSolver",
                file!(),
                MSG_LS_BAD_NVECTOR,
            );
            return KINLS_ILL_INPUT;
        }
    }

    /* Check for compatible LS type, matrix and "atimes" support */
    if iterative {
        if LS.ops.borrow().setscalingvectors.is_none() && vtemp1.ops.borrow().nvgetlength.is_none()
        {
            KINProcessError(
                Some(kin_mem),
                KINLS_ILL_INPUT,
                line!() as i32,
                "KINSetLinearSolver",
                file!(),
                MSG_LS_BAD_NVECTOR,
            );
            return KINLS_ILL_INPUT;
        }

        if !matrixbased && LS.ops.borrow().setatimes.is_none() {
            KINProcessError(
                Some(kin_mem),
                KINLS_ILL_INPUT,
                line!() as i32,
                "KINSetLinearSolver",
                file!(),
                "Incompatible inputs: iterative LS must support ATimes routine",
            );
            return KINLS_ILL_INPUT;
        }

        if matrixbased && A.is_none() {
            KINProcessError(
                Some(kin_mem),
                KINLS_ILL_INPUT,
                line!() as i32,
                "KINSetLinearSolver",
                file!(),
                "Incompatible inputs: matrix-iterative LS requires non-NULL matrix",
            );
            return KINLS_ILL_INPUT;
        }
    } else if A.is_none() {
        KINProcessError(
            Some(kin_mem),
            KINLS_ILL_INPUT,
            line!() as i32,
            "KINSetLinearSolver",
            file!(),
            "Incompatible inputs: direct LS requires non-NULL matrix",
        );
        return KINLS_ILL_INPUT;
    }

    /* free any existing system solver attached to KIN */
    let lfree = kin_mem.borrow().kin_lfree;
    if let Some(lfree) = lfree {
        lfree(kin_mem);
    }

    /* Determine if this is an iterative linear solver */
    kin_mem.borrow_mut().kin_inexact_ls = iterative;

    /* Set four main system linear solver function fields in kin_mem */
    {
        let mut m = kin_mem.borrow_mut();
        m.kin_linit = Some(kinLsInitialize);
        m.kin_lsetup = Some(kinLsSetup);
        m.kin_lsolve = Some(kinLsSolve);
        m.kin_lfree = Some(kinLsFree);
    }

    /* Get memory for KINLsMemRec (C: malloc + memset(0) then the default
    assignment block below; malloc failure is unreachable here). The
    struct literal carries exactly the state the C code holds after its
    default-assignment block (through `last_flag = KINLS_SUCCESS`). */
    let kin_func = kin_mem.borrow().kin_func;
    let mut kinls_mem: KINLsMem = Box::new(KINLsMemRec {
        /* memset(0) baseline — never assigned by any kinsol source */
        iterative: SUNFALSE,
        matrixbased: SUNFALSE,

        /* set SUNLinearSolver pointer */
        LS: LS.clone(),

        /* Set defaults for Jacobian-related fields */
        jacDQ: A.is_some(),
        jac: if A.is_some() {
            Some(kinLsDQJac as KINLsJacFn)
        } else {
            None
        },
        J_data: if A.is_some() {
            Some(Box::new(kin_mem.clone())) /* C: J_data = kin_mem */
        } else {
            None /* C: J_data = NULL (jac is NULL too, never invoked) */
        },
        jtimesDQ: SUNTRUE,
        jtimes: Some(kinLsDQJtimes),
        jt_func: kin_func,
        jt_data: Some(Box::new(kin_mem.clone())), /* C: jt_data = kin_mem */

        /* Set defaults for preconditioner-related fields */
        pset: None,
        psolve: None,
        pfree: None,
        pdata: None, /* C: pdata = kin_mem->kin_user_data (pass-through) */

        /* Initialize counters (kinLsInitializeCounters below re-zeros) */
        nje: 0,
        nfeDQ: 0,
        npe: 0,
        nli: 0,
        nps: 0,
        ncfl: 0,
        njtimes: 0,

        /* Set default values for the rest of the LS parameters */
        last_flag: KINLS_SUCCESS,

        /* memset(0) baseline for fields assigned further below */
        new_uu: SUNFALSE,
        tol_fac: 0.0,
        J: None,
    });

    /* Initialize counters */
    let _ = kinLsInitializeCounters(&mut kinls_mem);

    /* If LS supports ATimes, attach KINLs routine */
    if LS.ops.borrow().setatimes.is_some() {
        let retval = SUNLinSolSetATimes(LS, Some(Box::new(kin_mem.clone())), Some(kinLsATimes));
        if retval != SUN_SUCCESS {
            KINProcessError(
                Some(kin_mem),
                KINLS_SUNLS_FAIL,
                line!() as i32,
                "KINSetLinearSolver",
                file!(),
                "Error in calling SUNLinSolSetATimes",
            );
            drop(kinls_mem);
            return KINLS_SUNLS_FAIL;
        }
    }

    /* If LS supports preconditioning, initialize pset/psol to NULL */
    if LS.ops.borrow().setpreconditioner.is_some() {
        let retval = SUNLinSolSetPreconditioner(LS, Some(Box::new(kin_mem.clone())), None, None);
        if retval != SUN_SUCCESS {
            KINProcessError(
                Some(kin_mem),
                KINLS_SUNLS_FAIL,
                line!() as i32,
                "KINSetLinearSolver",
                file!(),
                "Error in calling SUNLinSolSetPreconditioner",
            );
            drop(kinls_mem);
            return KINLS_SUNLS_FAIL;
        }
    }

    /* initialize tolerance scaling factor */
    kinls_mem.tol_fac = -ONE;

    /* set SUNMatrix pointer (can be NULL) */
    kinls_mem.J = A.cloned();

    /* Attach linear solver memory to integrator memory */
    kin_mem.borrow_mut().kin_lmem = Some(kinls_mem);

    KINLS_SUCCESS
}

/*==================================================================
  Optional Set routines
  ==================================================================*/

/*------------------------------------------------------------------
  KINSetJacFn specifies the Jacobian function
  ------------------------------------------------------------------*/
pub fn KINSetJacFn(kinmem: &KINMem, jac: Option<KINLsJacFn>) -> i32 {
    /* access KINLsMem structure */
    let retval = kinLs_AccessLMem(kinmem, "KINSetJacFn");
    if retval != KIN_SUCCESS {
        return retval;
    }
    let kin_mem = kinmem;

    /* return with failure if jac cannot be used */
    if jac.is_some() && kinls_mem_mut(kin_mem).J.is_none() {
        KINProcessError(
            Some(kin_mem),
            KINLS_ILL_INPUT,
            line!() as i32,
            "KINSetJacFn",
            file!(),
            "Jacobian routine cannot be supplied for NULL SUNMatrix",
        );
        return KINLS_ILL_INPUT;
    }

    if jac.is_some() {
        let mut ls = kinls_mem_mut(kin_mem);
        ls.jacDQ = SUNFALSE;
        ls.jac = jac;
        ls.J_data = None; /* C: J_data = kin_mem->kin_user_data */
    } else {
        let mut ls = kinls_mem_mut(kin_mem);
        ls.jacDQ = SUNTRUE;
        ls.jac = Some(kinLsDQJac);
        ls.J_data = Some(Box::new(kin_mem.clone())); /* C: J_data = kin_mem */
    }

    KINLS_SUCCESS
}

/*------------------------------------------------------------------
  KINSetPreconditioner sets the preconditioner setup and solve
  functions
  ------------------------------------------------------------------*/
pub fn KINSetPreconditioner(
    kinmem: &KINMem,
    psetup: Option<KINLsPrecSetupFn>,
    psolve: Option<KINLsPrecSolveFn>,
) -> i32 {
    /* access KINLsMem structure */
    let retval = kinLs_AccessLMem(kinmem, "KINSetPreconditioner");
    if retval != KIN_SUCCESS {
        return retval;
    }
    let kin_mem = kinmem;

    /* store function pointers for user-supplied routines in KINLS interface */
    {
        let mut ls = kinls_mem_mut(kin_mem);
        ls.pset = psetup;
        ls.psolve = psolve;
    }

    /* issue error if LS object does not support user-supplied preconditioning */
    let LS = kinls_mem_mut(kin_mem).LS.clone();
    if LS.ops.borrow().setpreconditioner.is_none() {
        KINProcessError(
            Some(kin_mem),
            KINLS_ILL_INPUT,
            line!() as i32,
            "KINSetPreconditioner",
            file!(),
            "SUNLinearSolver object does not support user-supplied preconditioning",
        );
        return KINLS_ILL_INPUT;
    }

    /* notify iterative linear solver to call KINLs interface routines */
    let kinls_psetup: Option<SUNPSetupFn> = if psetup.is_none() {
        None
    } else {
        Some(kinLsPSetup)
    };
    let kinls_psolve: Option<SUNPSolveFn> = if psolve.is_none() {
        None
    } else {
        Some(kinLsPSolve)
    };
    let retval = SUNLinSolSetPreconditioner(
        &LS,
        Some(Box::new(kin_mem.clone())),
        kinls_psetup,
        kinls_psolve,
    );
    if retval != SUN_SUCCESS {
        KINProcessError(
            Some(kin_mem),
            KINLS_SUNLS_FAIL,
            line!() as i32,
            "KINSetPreconditioner",
            file!(),
            "Error in calling SUNLinSolSetPreconditioner",
        );
        return KINLS_SUNLS_FAIL;
    }

    KINLS_SUCCESS
}

/*------------------------------------------------------------------
  KINSetJacTimesVecFn sets the matrix-vector product function
  ------------------------------------------------------------------*/
pub fn KINSetJacTimesVecFn(kinmem: &KINMem, jtv: Option<KINLsJacTimesVecFn>) -> i32 {
    /* access KINLsMem structure */
    let retval = kinLs_AccessLMem(kinmem, "KINSetJacTimesVecFn");
    if retval != KIN_SUCCESS {
        return retval;
    }
    let kin_mem = kinmem;

    /* issue error if LS object does not support user-supplied ATimes */
    let LS = kinls_mem_mut(kin_mem).LS.clone();
    if LS.ops.borrow().setatimes.is_none() {
        KINProcessError(
            Some(kin_mem),
            KINLS_ILL_INPUT,
            line!() as i32,
            "KINSetJacTimesVecFn",
            file!(),
            "SUNLinearSolver object does not support user-supplied ATimes routine",
        );
        return KINLS_ILL_INPUT;
    }

    /* store function pointers for user-supplied routine in KINLs
    interface (NULL jtimes implies use of DQ default) */
    if jtv.is_some() {
        let mut ls = kinls_mem_mut(kin_mem);
        ls.jtimesDQ = SUNFALSE;
        ls.jtimes = jtv;
        ls.jt_data = None; /* C: jt_data = kin_mem->kin_user_data */
    } else {
        let kin_func = kin_mem.borrow().kin_func;
        let mut ls = kinls_mem_mut(kin_mem);
        ls.jtimesDQ = SUNTRUE;
        ls.jtimes = Some(kinLsDQJtimes);
        ls.jt_func = kin_func;
        ls.jt_data = Some(Box::new(kin_mem.clone())); /* C: jt_data = kin_mem */
    }

    KINLS_SUCCESS
}

/* KINSetJacTimesVecSysFn specifies an alternative user-supplied system function
   to use in the internal finite difference Jacobian-vector product */
pub fn KINSetJacTimesVecSysFn(kinmem: &KINMem, jtimesSysFn: Option<KINSysFn>) -> i32 {
    /* access KINLsMem structure */
    let retval = kinLs_AccessLMem(kinmem, "KINSetJacTimesVecSysFn");
    if retval != KIN_SUCCESS {
        return retval;
    }
    let kin_mem = kinmem;

    /* check if using internal finite difference approximation */
    if !kinls_mem_mut(kin_mem).jtimesDQ {
        KINProcessError(
            Some(kin_mem),
            KINLS_ILL_INPUT,
            line!() as i32,
            "KINSetJacTimesVecSysFn",
            file!(),
            "Internal finite-difference Jacobian-vector product is disabled.",
        );
        return KINLS_ILL_INPUT;
    }

    /* store function pointers for system function (NULL implies use kin_func) */
    if jtimesSysFn.is_some() {
        kinls_mem_mut(kin_mem).jt_func = jtimesSysFn;
    } else {
        let kin_func = kin_mem.borrow().kin_func;
        kinls_mem_mut(kin_mem).jt_func = kin_func;
    }

    KINLS_SUCCESS
}

/*==================================================================
  Optional Get routines
  ==================================================================*/

pub fn KINGetJac(kinmem: &KINMem, J: &mut Option<SUNMatrix>) -> i32 {
    /* access KINLsMem structure; set output and return */
    let retval = kinLs_AccessLMem(kinmem, "KINGetJac");
    if retval != KINLS_SUCCESS {
        return retval;
    }
    *J = kinls_mem_mut(kinmem).J.clone();
    KINLS_SUCCESS
}

pub fn KINGetJacNumIters(kinmem: &KINMem, nni_J: &mut i64) -> i32 {
    /* access KINLsMem structure; set output and return */
    let retval = kinLs_AccessLMem(kinmem, "KINGetJacNumIters");
    if retval != KINLS_SUCCESS {
        return retval;
    }
    *nni_J = kinmem.borrow().kin_nnilset;
    KINLS_SUCCESS
}

/*------------------------------------------------------------------
  KINGetLinWorkSpace returns the integer and real workspace size
  ------------------------------------------------------------------*/
pub fn KINGetLinWorkSpace(kinmem: &KINMem, lenrwLS: &mut i64, leniwLS: &mut i64) -> i32 {
    /* access KINLsMem structure */
    let retval = kinLs_AccessLMem(kinmem, "KINGetLinWorkSpace");
    if retval != KIN_SUCCESS {
        return retval;
    }
    let kin_mem = kinmem;

    /* start with fixed sizes plus vector/matrix pointers */
    *lenrwLS = 1;
    *leniwLS = 21;

    /* add N_Vector sizes */
    let vtemp1 = kin_mem
        .borrow()
        .kin_vtemp1
        .as_ref()
        .expect("kin_vtemp1")
        .clone();
    if vtemp1.ops.borrow().nvspace.is_some() {
        let mut lrw1: sunindextype = 0;
        let mut liw1: sunindextype = 0;
        N_VSpace(&vtemp1, &mut lrw1, &mut liw1);
        *lenrwLS += lrw1;
        *leniwLS += liw1;
    }

    /* add LS sizes */
    let LS = kinls_mem_mut(kin_mem).LS.clone();
    if LS.ops.borrow().space.is_some() {
        let mut lrw: i64 = 0;
        let mut liw: i64 = 0;
        let retval = SUNLinSolSpace(&LS, &mut lrw, &mut liw);
        if retval == 0 {
            *lenrwLS += lrw;
            *leniwLS += liw;
        }
    }

    KINLS_SUCCESS
}

/*------------------------------------------------------------------
  KINGetNumJacEvals returns the number of Jacobian evaluations
  ------------------------------------------------------------------*/
pub fn KINGetNumJacEvals(kinmem: &KINMem, njevals: &mut i64) -> i32 {
    /* access KINLsMem structure; set output value and return */
    let retval = kinLs_AccessLMem(kinmem, "KINGetNumJacEvals");
    if retval != KIN_SUCCESS {
        return retval;
    }
    *njevals = kinls_mem_mut(kinmem).nje;
    KINLS_SUCCESS
}

/*------------------------------------------------------------------
  KINGetNumPrecEvals returns the total number of preconditioner
  evaluations
  ------------------------------------------------------------------*/
pub fn KINGetNumPrecEvals(kinmem: &KINMem, npevals: &mut i64) -> i32 {
    /* access KINLsMem structure; set output value and return */
    let retval = kinLs_AccessLMem(kinmem, "KINGetNumPrecEvals");
    if retval != KIN_SUCCESS {
        return retval;
    }
    *npevals = kinls_mem_mut(kinmem).npe;
    KINLS_SUCCESS
}

/*------------------------------------------------------------------
  KINGetNumPrecSolves returns the total number of times the
  preconditioner was applied
  ------------------------------------------------------------------*/
pub fn KINGetNumPrecSolves(kinmem: &KINMem, npsolves: &mut i64) -> i32 {
    /* access KINLsMem structure; set output value and return */
    let retval = kinLs_AccessLMem(kinmem, "KINGetNumPrecSolves");
    if retval != KIN_SUCCESS {
        return retval;
    }
    *npsolves = kinls_mem_mut(kinmem).nps;
    KINLS_SUCCESS
}

/*------------------------------------------------------------------
  KINGetNumLinIters returns the total number of linear
  iterations
  ------------------------------------------------------------------*/
pub fn KINGetNumLinIters(kinmem: &KINMem, nliters: &mut i64) -> i32 {
    /* access KINLsMem structure; set output value and return */
    let retval = kinLs_AccessLMem(kinmem, "KINGetNumLinIters");
    if retval != KIN_SUCCESS {
        return retval;
    }
    *nliters = kinls_mem_mut(kinmem).nli;
    KINLS_SUCCESS
}

/*------------------------------------------------------------------
  KINGetNumLinConvFails returns the total number of convergence
  failures
  ------------------------------------------------------------------*/
pub fn KINGetNumLinConvFails(kinmem: &KINMem, nlcfails: &mut i64) -> i32 {
    /* access KINLsMem structure; set output value and return */
    let retval = kinLs_AccessLMem(kinmem, "KINGetNumLinConvFails");
    if retval != KIN_SUCCESS {
        return retval;
    }
    *nlcfails = kinls_mem_mut(kinmem).ncfl;
    KINLS_SUCCESS
}

/*------------------------------------------------------------------
  KINGetNumJtimesEvals returns the number of times the matrix
  vector product was computed
  ------------------------------------------------------------------*/
pub fn KINGetNumJtimesEvals(kinmem: &KINMem, njvevals: &mut i64) -> i32 {
    /* access KINLsMem structure; set output value and return */
    let retval = kinLs_AccessLMem(kinmem, "KINGetNumJtimesEvals");
    if retval != KIN_SUCCESS {
        return retval;
    }
    *njvevals = kinls_mem_mut(kinmem).njtimes;
    KINLS_SUCCESS
}

/*------------------------------------------------------------------
  KINGetNumLinFuncEvals returns the number of calls to the user's
  F routine by the linear solver module
  ------------------------------------------------------------------*/
pub fn KINGetNumLinFuncEvals(kinmem: &KINMem, nfevals: &mut i64) -> i32 {
    /* access KINLsMem structure; set output value and return */
    let retval = kinLs_AccessLMem(kinmem, "KINGetNumLinFuncEvals");
    if retval != KIN_SUCCESS {
        return retval;
    }
    *nfevals = kinls_mem_mut(kinmem).nfeDQ;
    KINLS_SUCCESS
}

/*------------------------------------------------------------------
  KINGetLastLinFlag returns the last flag set in the KINLS
  function
  ------------------------------------------------------------------*/
pub fn KINGetLastLinFlag(kinmem: &KINMem, flag: &mut i64) -> i32 {
    /* access KINLsMem structure; set output value and return */
    let retval = kinLs_AccessLMem(kinmem, "KINGetLastLinFlag");
    if retval != KIN_SUCCESS {
        return retval;
    }
    *flag = kinls_mem_mut(kinmem).last_flag as i64;
    KINLS_SUCCESS
}

/*------------------------------------------------------------------
  KINGetLinReturnFlagName
  ------------------------------------------------------------------*/
pub fn KINGetLinReturnFlagName(flag: i64) -> String {
    /* C mallocs a 30-char buffer and sprintf's into it; the owned
    String is the Rust equivalent of the caller-freed char*. */
    let name = if flag == KINLS_SUCCESS as i64 {
        "KINLS_SUCCESS"
    } else if flag == KINLS_MEM_NULL as i64 {
        "KINLS_MEM_NULL"
    } else if flag == KINLS_LMEM_NULL as i64 {
        "KINLS_LMEM_NULL"
    } else if flag == KINLS_ILL_INPUT as i64 {
        "KINLS_ILL_INPUT"
    } else if flag == KINLS_MEM_FAIL as i64 {
        "KINLS_MEM_FAIL"
    } else if flag == KINLS_PMEM_NULL as i64 {
        "KINLS_PMEM_NULL"
    } else if flag == KINLS_JACFUNC_ERR as i64 {
        "KINLS_JACFUNC_ERR"
    } else if flag == KINLS_SUNMAT_FAIL as i64 {
        "KINLS_SUNMAT_FAIL"
    } else if flag == KINLS_SUNLS_FAIL as i64 {
        "KINLS_SUNLS_FAIL"
    } else {
        "NONE"
    };
    name.to_string()
}

/*==================================================================
  KINLS Private functions
  ==================================================================*/

/*------------------------------------------------------------------
  kinLsATimes

  This routine coordinates the generation of the matrix-vector
  product z = J*v by calling either kinLsDQJtimes, which uses
  a difference quotient approximation for J*v, or by calling the
  user-supplied routine KINLsJacTimesVecFn if it is non-null.
  ------------------------------------------------------------------*/
pub fn kinLsATimes(kinmem: &mut Option<Box<dyn Any>>, v: &N_Vector, z: &N_Vector) -> i32 {
    /* access KINLsMem structure */
    let kin_mem = match kinLs_AccessLMemToken(kinmem, "kinLsATimes") {
        Ok(m) => m,
        Err(retval) => return retval,
    };

    /* call Jacobian-times-vector product routine
    (either user-supplied or internal DQ) */
    let jtimes = kinls_mem_mut(&kin_mem).jtimes.expect("jtimes");
    let uu = kin_mem
        .borrow()
        .kin_uu
        .as_ref()
        .expect("kin_uu")
        .clone();
    /* C passes &(kinls_mem->new_uu): the callee may write the flag back
    through that alias, so mirror it into a local and store it after */
    let mut new_uu = kinls_mem_mut(&kin_mem).new_uu;
    let use_field = kinls_mem_mut(&kin_mem).jt_data.is_some();
    let mut jt_data = if use_field {
        kinls_mem_mut(&kin_mem).jt_data.take()
    } else {
        kin_mem.borrow_mut().kin_user_data.take()
    };
    let retval = jtimes(v, z, &uu, &mut new_uu, &mut jt_data);
    if use_field {
        kinls_mem_mut(&kin_mem).jt_data = jt_data;
    } else {
        kin_mem.borrow_mut().kin_user_data = jt_data;
    }
    {
        let mut ls = kinls_mem_mut(&kin_mem);
        ls.new_uu = new_uu; /* write-back of the aliased flag */
        ls.njtimes += 1;
    }
    retval
}

/*---------------------------------------------------------------
  kinLsPSetup:

  This routine interfaces between the generic iterative linear
  solvers and the user's psetup routine. It passes to psetup all
  required state information from kin_mem. Its return value
  is the same as that returned by psetup. Note that the generic
  iterative linear solvers guarantee that kinLsPSetup will only
  be called in the case that the user's psetup routine is non-NULL.
  ---------------------------------------------------------------*/
pub fn kinLsPSetup(kinmem: &mut Option<Box<dyn Any>>) -> i32 {
    /* access KINLsMem structure */
    let kin_mem = match kinLs_AccessLMemToken(kinmem, "kinLsPSetup") {
        Ok(m) => m,
        Err(retval) => return retval,
    };

    /* Call user pset routine to update preconditioner */
    let pset = kinls_mem_mut(&kin_mem).pset.expect("pset");
    let (uu, uscale, fval, fscale) = {
        let m = kin_mem.borrow();
        (
            m.kin_uu.as_ref().expect("kin_uu").clone(),
            m.kin_uscale.as_ref().expect("kin_uscale").clone(),
            m.kin_fval.as_ref().expect("kin_fval").clone(),
            m.kin_fscale.as_ref().expect("kin_fscale").clone(),
        )
    };
    let use_field = kinls_mem_mut(&kin_mem).pdata.is_some();
    let mut pdata = if use_field {
        kinls_mem_mut(&kin_mem).pdata.take()
    } else {
        kin_mem.borrow_mut().kin_user_data.take()
    };
    let retval = pset(&uu, &uscale, &fval, &fscale, &mut pdata);
    if use_field {
        kinls_mem_mut(&kin_mem).pdata = pdata;
    } else {
        kin_mem.borrow_mut().kin_user_data = pdata;
    }
    kinls_mem_mut(&kin_mem).npe += 1;
    retval
}

/*------------------------------------------------------------------
  kinLsPSolve

  This routine interfaces between the generic iterative linear
  solvers and the user's psolve routine. It passes to psolve all
  required state information from kinsol_mem. Its return value is
  the same as that returned by psolve. Note that the generic
  SUNLinSol solver guarantees that kinLsPSolve will not be called
  in the case in which preconditioning is not done. This is the only
  case in which the user's psolve routine is allowed to be NULL.
  ------------------------------------------------------------------*/
pub fn kinLsPSolve(
    kinmem: &mut Option<Box<dyn Any>>,
    r: &N_Vector,
    z: &N_Vector,
    tol: sunrealtype,
    lr: i32,
) -> i32 {
    let _ = (tol, lr); /* SUNDIALS_MAYBE_UNUSED */

    /* access KINLsMem structure */
    let kin_mem = match kinLs_AccessLMemToken(kinmem, "kinLsPSolve") {
        Ok(m) => m,
        Err(retval) => return retval,
    };

    /* copy the rhs into z before the psolve call */
    /* Note: z returns with the solution */
    N_VScale(ONE, r, z);

    /* note: user-supplied preconditioning with KINSOL does not
    support either the 'tol' or 'lr' inputs */
    let psolve = kinls_mem_mut(&kin_mem).psolve.expect("psolve");
    let (uu, uscale, fval, fscale) = {
        let m = kin_mem.borrow();
        (
            m.kin_uu.as_ref().expect("kin_uu").clone(),
            m.kin_uscale.as_ref().expect("kin_uscale").clone(),
            m.kin_fval.as_ref().expect("kin_fval").clone(),
            m.kin_fscale.as_ref().expect("kin_fscale").clone(),
        )
    };
    let use_field = kinls_mem_mut(&kin_mem).pdata.is_some();
    let mut pdata = if use_field {
        kinls_mem_mut(&kin_mem).pdata.take()
    } else {
        kin_mem.borrow_mut().kin_user_data.take()
    };
    let retval = psolve(&uu, &uscale, &fval, &fscale, z, &mut pdata);
    if use_field {
        kinls_mem_mut(&kin_mem).pdata = pdata;
    } else {
        kin_mem.borrow_mut().kin_user_data = pdata;
    }
    kinls_mem_mut(&kin_mem).nps += 1;
    retval
}

/*------------------------------------------------------------------
  kinLsDQJac

  This routine is a wrapper for the Dense and Band implementations
  of the difference quotient Jacobian approximation routines.
  ------------------------------------------------------------------*/
pub fn kinLsDQJac(
    u: &N_Vector,
    fu: &N_Vector,
    Jac: &SUNMatrix,
    kinmem: &mut Option<Box<dyn Any>>,
    tmp1: &N_Vector,
    tmp2: &N_Vector,
) -> i32 {
    /* access KINMem structure */
    let kin_mem = match kinmem
        .as_ref()
        .and_then(|b| b.downcast_ref::<KINMem>())
        .cloned()
    {
        Some(m) => m,
        None => {
            KINProcessError(
                None,
                KINLS_MEM_NULL,
                line!() as i32,
                "kinLsDQJac",
                file!(),
                MSG_LS_KINMEM_NULL,
            );
            return KINLS_MEM_NULL;
        }
    };

    /* Jac non-NULL check (C: KINLS_LMEM_NULL / MSG_LS_LMEM_NULL):
    handled by the type system */

    /* Call the matrix-structure-specific DQ approximation routine */
    let retval;
    if SUNMatGetID(Jac) == SUNMATRIX_DENSE {
        retval = kinLsDenseDQJac(u, fu, Jac, &kin_mem, tmp1, tmp2);
    } else if SUNMatGetID(Jac) == SUNMATRIX_BAND {
        retval = kinLsBandDQJac(u, fu, Jac, &kin_mem, tmp1, tmp2);
    } else {
        KINProcessError(
            Some(&kin_mem),
            KIN_ILL_INPUT,
            line!() as i32,
            "kinLsDQJac",
            file!(),
            "unrecognized matrix type for kinLsDQJac",
        );
        retval = KIN_ILL_INPUT;
    }
    retval
}

/*------------------------------------------------------------------
  kinLsDenseDQJac

  This routine generates a dense difference quotient approximation
  to the Jacobian of F(u). It assumes a dense SUNMatrix input
  stored column-wise, and that elements within each column are
  contiguous. The address of the jth column of J is obtained via
  the function SUNDenseMatrix_Column() and this pointer is
  associated with an N_Vector using the N_VGetArrayPointer and
  N_VSetArrayPointer functions. Finally, the actual computation of
  the jth column of the Jacobian is done with a call to N_VLinearSum.

  The increment used in the finite-difference approximation
    J_ij = ( F_i(u+sigma_j * e_j) - F_i(u)  ) / sigma_j
  is
   sigma_j = max{|u_j|, |1/uscale_j|} * sqrt(uround)

  Note: uscale_j = 1/typ(u_j)

  NOTE: Any type of failure of the system function here leads to an
        unrecoverable failure of the Jacobian function and thus of
        the linear solver setup function, stopping KINSOL.
  ------------------------------------------------------------------*/
pub fn kinLsDenseDQJac(
    u: &N_Vector,
    fu: &N_Vector,
    Jac: &SUNMatrix,
    kin_mem: &KINMem,
    tmp1: &N_Vector,
    tmp2: &N_Vector,
) -> i32 {
    let mut retval: i32 = 0;

    /* access LsMem interface structure: through kinls_mem_mut below */

    /* access matrix dimension */
    let N = SUNDenseMatrix_Columns(Jac);

    /* Save pointer to the array in tmp2 (C saves the raw pointer; here
    the owned buffer is cloned out and handed back at the end) */
    let tmp2_data = N_VGetArrayPointer(tmp2).expect("tmp2 data").to_vec();

    /* Rename work vectors for readability */
    let ftemp = tmp1;
    let jthCol = tmp2;

    /* Obtain pointers to the data for u and uscale (C caches the raw
    data pointers; here the borrows are re-taken per use and never held
    across the system-function callback or a vector op) */
    let uscale = kin_mem
        .borrow()
        .kin_uscale
        .as_ref()
        .expect("kin_uscale")
        .clone();

    /* This is the only for loop for 0..N-1 in KINSOL */

    let mut j: sunindextype = 0;
    while j < N {
        /* Generate the jth col of J(u) */

        /* Set data address of jthCol, and save u_j values and scaling */
        /* C: N_VSetArrayPointer(SUNDenseMatrix_Column(Jac, j), jthCol) —
        copy the column in; the N_VLinearSum result is written back
        below (write-through of the C alias). */
        let col_data = SUNDenseMatrix_Column(Jac, j).to_vec();
        N_VSetArrayPointer(col_data, jthCol);

        let ujsaved = {
            let u_data = N_VGetArrayPointer(u).expect("u data");
            u_data[j as usize]
        };
        let ujscale = {
            let uscale_data = N_VGetArrayPointer(&uscale).expect("uscale data");
            ONE / uscale_data[j as usize]
        };

        /* Compute increment */
        let sign = if ujsaved >= ZERO { ONE } else { -ONE };
        let sqrt_relfunc = kin_mem.borrow().kin_sqrt_relfunc;
        let inc = sqrt_relfunc * SUNMAX(SUNRabs(ujsaved), ujscale) * sign;

        /* Increment u_j, call F(u), and return if error occurs */
        {
            let mut u_data = N_VGetArrayPointer(u).expect("u data");
            u_data[j as usize] += inc;
        }

        let func = kin_mem.borrow().kin_func.expect("kin_func");
        let mut user_data = kin_mem.borrow_mut().kin_user_data.take();
        retval = func(u, ftemp, &mut user_data);
        kin_mem.borrow_mut().kin_user_data = user_data;
        kinls_mem_mut(kin_mem).nfeDQ += 1;
        if retval != 0 {
            break;
        }

        /* reset u_j */
        {
            let mut u_data = N_VGetArrayPointer(u).expect("u data");
            u_data[j as usize] = ujsaved;
        }

        /* Construct difference quotient in jthCol */
        let inc_inv = ONE / inc;
        N_VLinearSum(inc_inv, ftemp, -inc_inv, fu, jthCol);

        /* write the computed column back into the matrix (C wrote it
        through the aliased column pointer) */
        {
            let jth_data = N_VGetArrayPointer(jthCol).expect("jthCol data");
            let mut col_j = SUNDenseMatrix_Column(Jac, j);
            col_j.copy_from_slice(&jth_data);
        }

        j += 1;
    }

    /* Restore original array pointer in tmp2 */
    N_VSetArrayPointer(tmp2_data, tmp2);

    retval
}

/*------------------------------------------------------------------
  kinLsBandDQJac

  This routine generates a banded difference quotient approximation
  to the Jacobian of F(u).  It assumes a SUNBandMatrix input stored
  column-wise, and that elements within each column are contiguous.
  This makes it possible to get the address of a column of J via the
  function SUNBandMatrix_Column() and to write a simple for loop to
  set each of the elements of a column in succession.

  NOTE: Any type of failure of the system function her leads to an
        unrecoverable failure of the Jacobian function and thus of
        the linear solver setup function, stopping KINSOL.
  ------------------------------------------------------------------*/
pub fn kinLsBandDQJac(
    u: &N_Vector,
    fu: &N_Vector,
    Jac: &SUNMatrix,
    kin_mem: &KINMem,
    tmp1: &N_Vector,
    tmp2: &N_Vector,
) -> i32 {
    /* access LsMem interface structure: through kinls_mem_mut below */

    /* access matrix dimensions */
    let N = SUNBandMatrix_Columns(Jac);
    let mupper = SUNBandMatrix_UpperBandwidth(Jac);
    let mlower = SUNBandMatrix_LowerBandwidth(Jac);
    let s_mu = SUNBandMatrix_StoredUpperBandwidth(Jac); /* SM_COLUMN_ELEMENT_B offset */

    /* Rename work vectors for use as temporary values of u and fu */
    let futemp = tmp1;
    let utemp = tmp2;

    /* Obtain pointers to the data for ewt, fy, futemp, y, ytemp (C
    caches the raw data pointers; here the borrows are re-taken per
    phase and never held across the system-function callback) */
    let uscale = kin_mem
        .borrow()
        .kin_uscale
        .as_ref()
        .expect("kin_uscale")
        .clone();

    /* Load utemp with u */
    N_VScale(ONE, u, utemp);

    /* Set bandwidth and number of column groups for band differencing */
    let width = mlower + mupper + 1;
    let ngroups = SUNMIN(width, N);

    let mut group: sunindextype = 1;
    while group <= ngroups {
        /* Increment all utemp components in group */
        {
            let sqrt_relfunc = kin_mem.borrow().kin_sqrt_relfunc;
            let u_data = N_VGetArrayPointer(u).expect("u data");
            let uscale_data = N_VGetArrayPointer(&uscale).expect("uscale data");
            let mut utemp_data = N_VGetArrayPointer(utemp).expect("utemp data");
            let mut j = group - 1;
            while j < N {
                let inc = sqrt_relfunc
                    * SUNMAX(
                        SUNRabs(u_data[j as usize]),
                        ONE / SUNRabs(uscale_data[j as usize]),
                    );
                utemp_data[j as usize] += inc;
                j += width;
            }
        }

        /* Evaluate f with incremented u */
        let func = kin_mem.borrow().kin_func.expect("kin_func");
        let mut user_data = kin_mem.borrow_mut().kin_user_data.take();
        let retval = func(utemp, futemp, &mut user_data);
        kin_mem.borrow_mut().kin_user_data = user_data;
        if retval != 0 {
            return retval;
        }

        /* Restore utemp components, then form and load difference quotients */
        {
            let sqrt_relfunc = kin_mem.borrow().kin_sqrt_relfunc;
            let u_data = N_VGetArrayPointer(u).expect("u data");
            let uscale_data = N_VGetArrayPointer(&uscale).expect("uscale data");
            let mut utemp_data = N_VGetArrayPointer(utemp).expect("utemp data");
            let futemp_data = N_VGetArrayPointer(futemp).expect("futemp data");
            let fu_data = N_VGetArrayPointer(fu).expect("fu data");
            let mut j = group - 1;
            while j < N {
                utemp_data[j as usize] = u_data[j as usize];
                let mut col_j = SUNBandMatrix_Column(Jac, j);
                let inc = sqrt_relfunc
                    * SUNMAX(
                        SUNRabs(u_data[j as usize]),
                        ONE / SUNRabs(uscale_data[j as usize]),
                    );
                let inc_inv = ONE / inc;
                let i1 = SUNMAX(0, j - mupper);
                let i2 = SUNMIN(j + mlower, N - 1);
                let mut i = i1;
                while i <= i2 {
                    /* C: SM_COLUMN_ELEMENT_B(col_j, i, j) = ... */
                    col_j[SM_COLUMN_ELEMENT_IDX(i, j, s_mu)] =
                        inc_inv * (futemp_data[i as usize] - fu_data[i as usize]);
                    i += 1;
                }
                j += width;
            }
        }

        group += 1;
    }

    /* Increment counter nfeDQ */
    kinls_mem_mut(kin_mem).nfeDQ += ngroups;

    0
}

/*------------------------------------------------------------------
  kinLsDQJtimes

  This routine generates the matrix-vector product z = J*v using a
  difference quotient approximation. The approximation is
  J*v = [func(uu + sigma*v) - func(uu)]/sigma. Here sigma is based
  on the dot products (uscale*uu, uscale*v) and
  (uscale*v, uscale*v), the L1Norm(uscale*v), and on sqrt_relfunc
  (the square root of the relative error in the function). Note
  that v in the argument list has already been both preconditioned
  and unscaled.

  NOTE: Unlike the DQ Jacobian functions for direct linear solvers
        (which are called from within the lsetup function), this
        function is called from within the lsolve function and thus
        a recovery may still be possible even if the system function
        fails (recoverably).
  ------------------------------------------------------------------*/
pub fn kinLsDQJtimes(
    v: &N_Vector,
    Jv: &N_Vector,
    u: &N_Vector,
    new_u: &mut sunbooleantype,
    kinmem: &mut Option<Box<dyn Any>>,
) -> i32 {
    let _ = new_u; /* SUNDIALS_MAYBE_UNUSED */

    /* access KINLsMem structure */
    let kin_mem = match kinLs_AccessLMemToken(kinmem, "kinLsDQJtimes") {
        Ok(m) => m,
        Err(retval) => return retval,
    };

    /* ensure that NVector supplies requisite routines */
    {
        let ops = v.ops.borrow();
        if ops.nvprod.is_none()
            || ops.nvdotprod.is_none()
            || ops.nvl1norm.is_none()
            || ops.nvlinearsum.is_none()
        {
            KINProcessError(
                Some(&kin_mem),
                KINLS_ILL_INPUT,
                line!() as i32,
                "kinLsDQJtimes",
                file!(),
                MSG_LS_BAD_NVECTOR,
            );
            return KINLS_ILL_INPUT;
        }
    }

    let (uscale, vtemp1, vtemp2) = {
        let m = kin_mem.borrow();
        (
            m.kin_uscale.as_ref().expect("kin_uscale").clone(),
            m.kin_vtemp1.as_ref().expect("kin_vtemp1").clone(),
            m.kin_vtemp2.as_ref().expect("kin_vtemp2").clone(),
        )
    };

    /* scale the vector v and put Du*v into vtemp1 */
    N_VProd(v, &uscale, &vtemp1);

    /* scale u and put into Jv (used as a temporary storage) */
    N_VProd(u, &uscale, Jv);

    /* compute dot product (Du*u).(Du*v) */
    let sutsv = N_VDotProd(Jv, &vtemp1);

    /* compute dot product (Du*v).(Du*v) */
    let vtv = N_VDotProd(&vtemp1, &vtemp1);

    /* compute differencing factor -- this is from p. 469, Brown and Saad paper */
    let sq1norm = N_VL1Norm(&vtemp1);
    let sign = if sutsv >= ZERO { ONE } else { -ONE };
    let sqrt_relfunc = kin_mem.borrow().kin_sqrt_relfunc;
    let sigma = sign * sqrt_relfunc * SUNMAX(SUNRabs(sutsv), sq1norm) / vtv;
    let sigma_inv = ONE / sigma;

    /* compute the u-prime at which to evaluate the function func */
    N_VLinearSum(ONE, u, sigma, v, &vtemp1);

    /* call the system function to calculate func(u+sigma*v) */
    let jt_func = kinls_mem_mut(&kin_mem).jt_func.expect("jt_func");
    let mut user_data = kin_mem.borrow_mut().kin_user_data.take();
    let retval = jt_func(&vtemp1, &vtemp2, &mut user_data);
    kin_mem.borrow_mut().kin_user_data = user_data;
    kinls_mem_mut(&kin_mem).nfeDQ += 1;
    if retval != 0 {
        return retval;
    }

    /* finish the computation of the difference quotient */
    let fval = kin_mem
        .borrow()
        .kin_fval
        .as_ref()
        .expect("kin_fval")
        .clone();
    N_VLinearSum(sigma_inv, &vtemp2, -sigma_inv, &fval, Jv);

    0
}

/*------------------------------------------------------------------
  kinLsInitialize performs remaining initializations specific
  to the iterative linear solver interface (and solver itself)
  ------------------------------------------------------------------*/
pub fn kinLsInitialize(kin_mem: &KINMem) -> i32 {
    /* Access KINLsMem structure */
    if kin_mem.borrow().kin_lmem.is_none() {
        KINProcessError(
            Some(kin_mem),
            KINLS_LMEM_NULL,
            line!() as i32,
            "kinLsInitialize",
            file!(),
            MSG_LS_LMEM_NULL,
        );
        return KINLS_LMEM_NULL;
    }

    /* Test for valid combinations of matrix & Jacobian routines: */
    let J = kinls_mem_mut(kin_mem).J.clone();
    if J.is_none() {
        /* If SUNMatrix A is NULL: ensure 'jac' function pointer is NULL */
        let mut ls = kinls_mem_mut(kin_mem);
        ls.jacDQ = SUNFALSE;
        ls.jac = None;
        ls.J_data = None;
    } else if kinls_mem_mut(kin_mem).jacDQ {
        /* If J is non-NULL, and 'jac' is not user-supplied:
        - if A is dense or band, ensure that our DQ approx. is used
        - otherwise => error */
        let J = J.as_ref().expect("J");
        let mut retval = 0;
        if J.ops.borrow().getid.is_some() {
            let id = SUNMatGetID(J);
            if id == SUNMATRIX_DENSE || id == SUNMATRIX_BAND {
                let mut ls = kinls_mem_mut(kin_mem);
                ls.jac = Some(kinLsDQJac);
                ls.J_data = Some(Box::new(kin_mem.clone())); /* C: J_data = kin_mem */
            } else {
                retval += 1;
            }
        } else {
            retval += 1;
        }
        if retval != 0 {
            KINProcessError(
                Some(kin_mem),
                KINLS_ILL_INPUT,
                line!() as i32,
                "kinLsInitialize",
                file!(),
                "No Jacobian constructor available for SUNMatrix type",
            );
            kinls_mem_mut(kin_mem).last_flag = KINLS_ILL_INPUT;
            return KINLS_ILL_INPUT;
        }

        /* check for required vector operations for kinLsDQJac routine */
        let vtemp1 = kin_mem
            .borrow()
            .kin_vtemp1
            .as_ref()
            .expect("kin_vtemp1")
            .clone();
        {
            let ops = vtemp1.ops.borrow();
            if ops.nvlinearsum.is_none()
                || ops.nvscale.is_none()
                || ops.nvgetarraypointer.is_none()
                || ops.nvsetarraypointer.is_none()
            {
                KINProcessError(
                    Some(kin_mem),
                    KINLS_ILL_INPUT,
                    line!() as i32,
                    "kinLsInitialize",
                    file!(),
                    MSG_LS_BAD_NVECTOR,
                );
                return KINLS_ILL_INPUT;
            }
        }
    } else {
        /* If J is non-NULL, and 'jac' is user-supplied,
        reset J_data pointer (just in case) */
        kinls_mem_mut(kin_mem).J_data = None; /* C: J_data = kin_mem->kin_user_data */
    }

    /* Prohibit Picard iteration with DQ Jacobian approximation or difference-quotient J*v */
    let globalstrategy = kin_mem.borrow().kin_globalstrategy;
    let (jacDQ, jtimesDQ) = {
        let ls = kinls_mem_mut(kin_mem);
        (ls.jacDQ, ls.jtimesDQ)
    };
    if (globalstrategy == KIN_PICARD) && jacDQ && jtimesDQ {
        KINProcessError(
            Some(kin_mem),
            KINLS_ILL_INPUT,
            line!() as i32,
            "kinLsInitialize",
            file!(),
            MSG_NOL_FAIL,
        );
        return KINLS_ILL_INPUT;
    }

    /* error-checking is complete, begin initializations */

    /* Initialize counters */
    let _ = kinLsInitializeCounters(&mut kinls_mem_mut(kin_mem));

    /* Set Jacobian-related fields, based on jtimesDQ */
    if kinls_mem_mut(kin_mem).jtimesDQ {
        let mut ls = kinls_mem_mut(kin_mem);
        ls.jtimes = Some(kinLsDQJtimes);
        ls.jt_data = Some(Box::new(kin_mem.clone())); /* C: jt_data = kin_mem */
    } else {
        kinls_mem_mut(kin_mem).jt_data = None; /* C: jt_data = kin_mem->kin_user_data */
    }

    /* if J is NULL and: NOT preconditioning or do NOT need to setup the
    preconditioner, then set the lsetup function to NULL */
    let (J_is_none, psolve_is_none, pset_is_none) = {
        let ls = kinls_mem_mut(kin_mem);
        (ls.J.is_none(), ls.psolve.is_none(), ls.pset.is_none())
    };
    if J_is_none && (psolve_is_none || pset_is_none) {
        kin_mem.borrow_mut().kin_lsetup = None;
    }

    /* Set scaling vectors assuming RIGHT preconditioning */
    /* NOTE: retval is non-zero only if LS == NULL        */
    let LS = kinls_mem_mut(kin_mem).LS.clone();
    if LS.ops.borrow().setscalingvectors.is_some() {
        let fscale = kin_mem
            .borrow()
            .kin_fscale
            .as_ref()
            .expect("kin_fscale")
            .clone();
        let retval = SUNLinSolSetScalingVectors(&LS, Some(&fscale), Some(&fscale));
        if retval != SUN_SUCCESS {
            KINProcessError(
                Some(kin_mem),
                KINLS_SUNLS_FAIL,
                line!() as i32,
                "kinLsInitialize",
                file!(),
                "Error in calling SUNLinSolSetScalingVectors",
            );
            return KINLS_SUNLS_FAIL;
        }
    }

    /* If the linear solver is iterative or matrix-iterative, and if left/right
    scaling are not supported, we must update linear solver tolerances in an
    attempt to account for the fscale vector.  We make the following assumptions:
      1. fscale_i = fs_mean, for i=0,...,n-1 (i.e. the weights are homogeneous)
      2. the linear solver uses a basic 2-norm to measure convergence
    Hence (using the notation from sunlinsol_spgmr.h, with S = diag(fscale)),
          || bbar - Abar xbar ||_2 < tol
      <=> || S b - S A x ||_2 < tol
      <=> || S (b - A x) ||_2 < tol
      <=> \sum_{i=0}^{n-1} (fscale_i (b - A x)_i)^2 < tol^2
      <=> fs_mean^2 \sum_{i=0}^{n-1} (b - A x_i)^2 < tol^2
      <=> \sum_{i=0}^{n-1} (b - A x_i)^2 < tol^2 / fs_mean^2
      <=> || b - A x ||_2 < tol / fs_mean
      <=> || b - A x ||_2 < tol * tol_fac
    So we compute tol_fac = sqrt(N) / ||fscale||_L2 for scaling desired tolerances */
    /* NOTE (upstream): `iterative` is never assigned and is therefore
    always SUNFALSE, so the else branch always runs — see module docs. */
    let iterative = kinls_mem_mut(kin_mem).iterative;
    if iterative && LS.ops.borrow().setscalingvectors.is_none() {
        let (vtemp1, fscale) = {
            let m = kin_mem.borrow();
            (
                m.kin_vtemp1.as_ref().expect("kin_vtemp1").clone(),
                m.kin_fscale.as_ref().expect("kin_fscale").clone(),
            )
        };
        N_VConst(ONE, &vtemp1);
        let tol_fac =
            SUNRsqrt(N_VGetLength(&vtemp1) as sunrealtype) / N_VWL2Norm(&fscale, &vtemp1);
        kinls_mem_mut(kin_mem).tol_fac = tol_fac;
    } else {
        kinls_mem_mut(kin_mem).tol_fac = ONE;
    }

    /* Call LS initialize routine, and return result */
    let last_flag = SUNLinSolInitialize(&LS);
    kinls_mem_mut(kin_mem).last_flag = last_flag;
    last_flag
}

/*------------------------------------------------------------------
  kinLsSetup call the LS setup routine
  ------------------------------------------------------------------*/
pub fn kinLsSetup(kin_mem: &KINMem) -> i32 {
    /* Access KINLsMem structure */
    if kin_mem.borrow().kin_lmem.is_none() {
        KINProcessError(
            Some(kin_mem),
            KINLS_LMEM_NULL,
            line!() as i32,
            "kinLsSetup",
            file!(),
            MSG_LS_LMEM_NULL,
        );
        return KINLS_LMEM_NULL;
    }

    /* recompute if J if it is non-NULL */
    let J = kinls_mem_mut(kin_mem).J.clone();
    if let Some(J) = &J {
        /* Increment nje counter. */
        kinls_mem_mut(kin_mem).nje += 1;

        /* Clear the linear system matrix if necessary */
        let LS = kinls_mem_mut(kin_mem).LS.clone();
        if SUNLinSolGetType(&LS) == SUNLINEARSOLVER_DIRECT {
            let retval = SUNMatZero(J);
            if retval != 0 {
                KINProcessError(
                    Some(kin_mem),
                    KINLS_SUNMAT_FAIL,
                    line!() as i32,
                    "kinLsSetup",
                    file!(),
                    MSG_LS_MATZERO_FAILED,
                );
                kinls_mem_mut(kin_mem).last_flag = KINLS_SUNMAT_FAIL;
                return KINLS_SUNMAT_FAIL;
            }
        }

        /* Call Jacobian routine */
        let jac = kinls_mem_mut(kin_mem).jac.expect("jac");
        let (uu, fval, vtemp1, vtemp2) = {
            let m = kin_mem.borrow();
            (
                m.kin_uu.as_ref().expect("kin_uu").clone(),
                m.kin_fval.as_ref().expect("kin_fval").clone(),
                m.kin_vtemp1.as_ref().expect("kin_vtemp1").clone(),
                m.kin_vtemp2.as_ref().expect("kin_vtemp2").clone(),
            )
        };
        let use_field = kinls_mem_mut(kin_mem).J_data.is_some();
        let mut J_data = if use_field {
            kinls_mem_mut(kin_mem).J_data.take()
        } else {
            kin_mem.borrow_mut().kin_user_data.take()
        };
        let retval = jac(&uu, &fval, J, &mut J_data, &vtemp1, &vtemp2);
        if use_field {
            kinls_mem_mut(kin_mem).J_data = J_data;
        } else {
            kin_mem.borrow_mut().kin_user_data = J_data;
        }
        if retval != 0 {
            KINProcessError(
                Some(kin_mem),
                KINLS_JACFUNC_ERR,
                line!() as i32,
                "kinLsSetup",
                file!(),
                MSG_LS_JACFUNC_FAILED,
            );
            kinls_mem_mut(kin_mem).last_flag = KINLS_JACFUNC_ERR;
            return KINLS_JACFUNC_ERR;
        }
    }

    /* Call LS setup routine -- the LS will call kinLsPSetup (if applicable) */
    let LS = kinls_mem_mut(kin_mem).LS.clone();
    let last_flag = SUNLinSolSetup(&LS, J.as_ref());
    kinls_mem_mut(kin_mem).last_flag = last_flag;

    /* save nni value from most recent lsetup call */
    let kin_nni = kin_mem.borrow().kin_nni;
    kin_mem.borrow_mut().kin_nnilset = kin_nni;

    last_flag
}

/*------------------------------------------------------------------
  kinLsSolve interfaces between KINSOL and the generic
  SUNLinearSolver object
  ------------------------------------------------------------------*/
pub fn kinLsSolve(
    kin_mem: &KINMem,
    xx: &N_Vector,
    bb: &N_Vector,
    sJpnorm: &mut sunrealtype,
    sFdotJp: &mut sunrealtype,
) -> i32 {
    /* Access KINLsMem structure */
    if kin_mem.borrow().kin_lmem.is_none() {
        KINProcessError(
            Some(kin_mem),
            KINLS_LMEM_NULL,
            line!() as i32,
            "kinLsSolve",
            file!(),
            MSG_LS_LMEM_NULL,
        );
        return KINLS_LMEM_NULL;
    }

    /* Set linear solver tolerance as input value times scaling factor
    (to account for possible lack of support for left/right scaling
    vectors in SUNLinSol object) */
    let kin_eps = kin_mem.borrow().kin_eps;
    let tol_fac = kinls_mem_mut(kin_mem).tol_fac;
    let tol = kin_eps * tol_fac;

    /* Set initial guess x = 0 to LS */
    N_VConst(ZERO, xx);

    /* Set zero initial guess flag */
    let LS = kinls_mem_mut(kin_mem).LS.clone();
    let mut retval = SUNLinSolSetZeroGuess(&LS, SUNTRUE);
    if retval != SUN_SUCCESS {
        return -1;
    }

    /* set flag required for user-supplied J*v routine */
    kinls_mem_mut(kin_mem).new_uu = SUNTRUE;

    /* Call solver */
    let J = kinls_mem_mut(kin_mem).J.clone();
    retval = SUNLinSolSolve(&LS, J.as_ref(), xx, bb, tol);

    /* Retrieve solver statistics */
    let mut nli_inc: i32 = 0;
    if LS.ops.borrow().numiters.is_some() {
        nli_inc = SUNLinSolNumIters(&LS);
    }

    /* SUNDIALS_LOGGING_LEVEL >= INFO block (PRNT_NLI / PRNT_EPS
    KINPrintInfo calls) is compiled out at the reference level */

    /* Increment counters nli and ncfl */
    {
        let mut ls = kinls_mem_mut(kin_mem);
        ls.nli += nli_inc as i64;
        if retval != SUN_SUCCESS {
            ls.ncfl += 1;
        }
    }

    /* Interpret solver return value */
    kinls_mem_mut(kin_mem).last_flag = retval;

    if (retval != SUN_SUCCESS) && (retval != SUNLS_RES_REDUCED) {
        match retval {
            SUNLS_ATIMES_FAIL_REC | SUNLS_PSOLVE_FAIL_REC => return 1,
            SUN_ERR_ARG_CORRUPT
            | SUN_ERR_ARG_INCOMPATIBLE
            | SUN_ERR_MEM_FAIL
            | SUNLS_GS_FAIL
            | SUNLS_CONV_FAIL
            | SUNLS_QRFACT_FAIL
            | SUNLS_LUFACT_FAIL
            | SUNLS_QRSOL_FAIL => {}
            SUNLS_PACKAGE_FAIL_REC => {
                KINProcessError(
                    Some(kin_mem),
                    SUNLS_PACKAGE_FAIL_REC,
                    line!() as i32,
                    "kinLsSolve",
                    file!(),
                    "Failure in SUNLinSol external package",
                );
            }
            SUN_ERR_EXT_FAIL => {
                KINProcessError(
                    Some(kin_mem),
                    SUN_ERR_EXT_FAIL,
                    line!() as i32,
                    "kinLsSolve",
                    file!(),
                    "Failure in SUNLinSol external package",
                );
            }
            SUNLS_ATIMES_FAIL_UNREC => {
                KINProcessError(
                    Some(kin_mem),
                    SUNLS_ATIMES_FAIL_UNREC,
                    line!() as i32,
                    "kinLsSolve",
                    file!(),
                    MSG_LS_JTIMES_FAILED,
                );
            }
            SUNLS_PSOLVE_FAIL_UNREC => {
                KINProcessError(
                    Some(kin_mem),
                    SUNLS_PSOLVE_FAIL_UNREC,
                    line!() as i32,
                    "kinLsSolve",
                    file!(),
                    MSG_LS_PSOLVE_FAILED,
                );
            }
            _ => {
                KINProcessError(
                    Some(kin_mem),
                    retval,
                    line!() as i32,
                    "kinLsSolve",
                    file!(),
                    "Unrecognized error return value from SUNLinSolSolve",
                );
            }
        }
        return retval;
    }

    /* SUNLinSolSolve returned SUN_SUCCESS or SUNLS_RES_REDUCED */

    /* Compute auxiliary values for use in the linesearch and in KINForcingTerm.
    These will be subsequently corrected if the step is reduced by constraints
    or the linesearch. */
    let (globalstrategy, inexact_ls, etaflag) = {
        let m = kin_mem.borrow();
        (m.kin_globalstrategy, m.kin_inexact_ls, m.kin_etaflag)
    };
    if globalstrategy != KIN_FP {
        /* sJpnorm is the norm of the scaled product (scaled by fscale) of the
        current Jacobian matrix J and the step vector p (= solution vector xx) */
        if inexact_ls && etaflag == KIN_ETACHOICE1 {
            /* C passes `kin_mem` as the void* token */
            let mut atimes_token: Option<Box<dyn Any>> = Some(Box::new(kin_mem.clone()));
            let retval = kinLsATimes(&mut atimes_token, xx, bb);
            if retval > 0 {
                kinls_mem_mut(kin_mem).last_flag = SUNLS_ATIMES_FAIL_REC;
                return 1;
            } else if retval < 0 {
                kinls_mem_mut(kin_mem).last_flag = SUNLS_ATIMES_FAIL_UNREC;
                return -1;
            }
            let fscale = kin_mem
                .borrow()
                .kin_fscale
                .as_ref()
                .expect("kin_fscale")
                .clone();
            *sJpnorm = N_VWL2Norm(bb, &fscale);
        }

        /* sFdotJp is the dot product of the scaled f vector and the scaled
        vector J*p, where the scaling uses fscale */
        if (inexact_ls && etaflag == KIN_ETACHOICE1) || globalstrategy == KIN_LINESEARCH {
            let (fscale, fval) = {
                let m = kin_mem.borrow();
                (
                    m.kin_fscale.as_ref().expect("kin_fscale").clone(),
                    m.kin_fval.as_ref().expect("kin_fval").clone(),
                )
            };
            N_VProd(bb, &fscale, bb);
            N_VProd(bb, &fscale, bb);
            *sFdotJp = N_VDotProd(&fval, bb);
        }
    }

    0
}

/*------------------------------------------------------------------
  kinLsFree frees memory associated with the KINLs system
  solver interface
  ------------------------------------------------------------------*/
pub fn kinLsFree(kin_mem: &KINMem) -> i32 {
    /* NULL KINMem check: handled by type system */

    /* Return immediately if kin_mem->kin_lmem is NULL */
    if kin_mem.borrow().kin_lmem.is_none() {
        return KINLS_SUCCESS;
    }

    /* Nullify SUNMatrix pointer */
    kinls_mem_mut(kin_mem).J = None;

    /* Free preconditioner memory (if applicable) */
    let pfree = kinls_mem_mut(kin_mem).pfree;
    if let Some(pfree) = pfree {
        pfree(kin_mem);
    }

    /* free KINLs interface structure (C `free()` leaves kin_lmem
    dangling; the Rust drop clears it) */
    kin_mem.borrow_mut().kin_lmem = None;

    KINLS_SUCCESS
}

/*------------------------------------------------------------------
  kinLsInitializeCounters resets counters for the LS interface
  ------------------------------------------------------------------*/
pub fn kinLsInitializeCounters(kinls_mem: &mut KINLsMemRec) -> i32 {
    kinls_mem.nje = 0;
    kinls_mem.nfeDQ = 0;
    kinls_mem.npe = 0;
    kinls_mem.nli = 0;
    kinls_mem.nps = 0;
    kinls_mem.ncfl = 0;
    kinls_mem.njtimes = 0;
    0
}

/*---------------------------------------------------------------
  kinLs_AccessLMem

  Public-API flavor of the C helper: with `&KINMem` the NULL-mem
  check vanishes (handled by the type system); this verifies that
  linear solver memory is attached. Callers then use
  `kinls_mem_mut` for field access.
  ---------------------------------------------------------------*/
pub fn kinLs_AccessLMem(kinmem: &KINMem, fname: &str) -> i32 {
    /* NULL-mem check: handled by type system */
    if kinmem.borrow().kin_lmem.is_none() {
        KINProcessError(
            Some(kinmem),
            KINLS_LMEM_NULL,
            line!() as i32,
            fname,
            file!(),
            MSG_LS_LMEM_NULL,
        );
        return KINLS_LMEM_NULL;
    }
    KINLS_SUCCESS
}

/*---------------------------------------------------------------
  kinLs_AccessLMemToken

  Callback flavor of the C `kinLs_AccessLMem`: the C `void* kinmem`
  argument arrives as a data token holding a `KINMem` clone. A
  missing/foreign token maps to the C NULL check.
  ---------------------------------------------------------------*/
pub fn kinLs_AccessLMemToken(
    kinmem: &Option<Box<dyn Any>>,
    fname: &str,
) -> Result<KINMem, i32> {
    let kin_mem = match kinmem
        .as_ref()
        .and_then(|b| b.downcast_ref::<KINMem>())
        .cloned()
    {
        Some(m) => m,
        None => {
            KINProcessError(
                None,
                KINLS_MEM_NULL,
                line!() as i32,
                fname,
                file!(),
                MSG_LS_KINMEM_NULL,
            );
            return Err(KINLS_MEM_NULL);
        }
    };
    if kin_mem.borrow().kin_lmem.is_none() {
        KINProcessError(
            Some(&kin_mem),
            KINLS_LMEM_NULL,
            line!() as i32,
            fname,
            file!(),
            MSG_LS_LMEM_NULL,
        );
        return Err(KINLS_LMEM_NULL);
    }
    Ok(kin_mem)
}

/*---------------------------------------------------------------
  EOF
  ---------------------------------------------------------------*/
