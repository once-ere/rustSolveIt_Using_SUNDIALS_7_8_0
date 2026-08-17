//! Port of `src/ida/ida_ls.c` (+ `src/ida/ida_ls_impl.h` and
//! `include/ida/ida_ls.h` folded, plus `IDASetJacTimesResFn` whose
//! prototype lives in `include/ida/ida.h` but whose body is in
//! `ida_ls.c`).
//!
//! IDA's linear solver interface (IDALS): attaches a generic
//! `SUNLinearSolver` (and optional `SUNMatrix`) to IDA, provides the
//! `ida_linit`/`ida_lsetup`/`ida_lsolve`/`ida_lperf`/`ida_lfree`
//! integrator hooks, the difference-quotient dense/band Jacobians of
//! `F_y + c_j*F_y'` and the DQ J*v product, and the
//! ATimes/PSetup/PSolve trampolines registered with the LS.
//!
//! Data-token model (C `void*` fields `J_data`/`pdata`/`jt_data`): in C
//! each field holds either `IDA_mem` (internal IDALS routine) or
//! `IDA_mem->ida_user_data` (user routine). Here the field is
//! `Option<Box<dyn Any>>`: `Some(box)` is a module-owned token (an
//! `IDAMem` clone for the internal IDALS routines, or whatever an
//! internal preconditioner module — e.g. IDABBDPRE — stored), while
//! `None` means "pass the integrator's `ida_user_data`" — the invoker
//! `Option::take`s the corresponding box around the callback and
//! restores it on EVERY path, including early returns and error paths.
//! This reproduces the C pointer aliasing without double ownership; the
//! only divergence is that a C snapshot of a *stale* `ida_user_data`
//! (user data replaced after the Set* call) cannot occur — the current
//! `ida_user_data` is always passed. For `J_data`/`jt_data` that matches
//! C exactly (`idaLsInitialize`'s "reset just in case" assignments
//! refresh them); for `pdata` C keeps the `IDASetLinearSolver`-time
//! snapshot forever, so an `IDASetUserData` call AFTER
//! `IDASetLinearSolver` diverges: C's pset/psolve keep seeing the old
//! pointer, this port passes the new box (accepted deviation class 6,
//! see ARCHITECTURE.md).
//!
//! Note that `idaLsDQJtimes` calls `jt_res` with `IDA_mem->ida_user_data`
//! (NOT with the `jt_data` token it was handed) — that is the C
//! behavior and it is safe here because `idaLsATimes` only takes
//! `ida_user_data` when `jt_data` is `None`, which never happens while
//! the DQ jtimes is installed.
//!
//! Granular borrow discipline: no `IDA_mem` borrow (and in particular no
//! `idals_mem_mut` guard) is held across a user callback, an N_Vector
//! op on a user-visible vector, an `IDAProcessError` call, or a
//! SUNLinearSolver/SUNMatrix call.
//!
//! Build config: `SUNDIALS_LOGGING_LEVEL=2` (all `SUNLogInfo`/
//! `SUNLogInfoIf` call sites compile away and are omitted at translation
//! time; the `IDA_WARNING` messages raised by `idaLsPerf` are kept),
//! profiling off, error checks off, serial branches only.

use std::any::Any;
use std::cell::RefMut;

use crate::ida_impl::*;
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
    SUNBandMatrix_LowerBandwidth, SUNBandMatrix_StoredUpperBandwidth,
    SUNBandMatrix_UpperBandwidth,
};
use sundials_core::sunmatrix_dense::{SUNDenseMatrix_Column, SUNDenseMatrix_Columns};

/* constants (ida_ls.c). ZERO/ONE/TWO/PT9 are the identically-valued
constants already published by `ida_impl` (fragment protocol) and are
used from there; only the two that `ida_impl` does not carry are
defined here. */
const MAX_ITERS: i32 = 3; /* max. number of attempts to recover in DQ J*v */
const PT25: sunrealtype = 0.25;
const PT05: sunrealtype = 0.05;

/*=================================================================
  IDALS Constants (include/ida/ida_ls.h)
  =================================================================*/

pub const IDALS_SUCCESS: i32 = 0;
pub const IDALS_MEM_NULL: i32 = -1;
pub const IDALS_LMEM_NULL: i32 = -2;
pub const IDALS_ILL_INPUT: i32 = -3;
pub const IDALS_MEM_FAIL: i32 = -4;
pub const IDALS_PMEM_NULL: i32 = -5;
pub const IDALS_JACFUNC_UNRECVR: i32 = -6;
pub const IDALS_JACFUNC_RECVR: i32 = -7;
pub const IDALS_SUNMAT_FAIL: i32 = -8;
pub const IDALS_SUNLS_FAIL: i32 = -9;

/*=================================================================
  IDALS user-supplied function prototypes (include/ida/ida_ls.h)
  =================================================================*/

pub type IDALsJacFn = fn(
    t: sunrealtype,
    c_j: sunrealtype,
    y: &N_Vector,
    yp: &N_Vector,
    r: &N_Vector,
    Jac: &SUNMatrix,
    user_data: &mut Option<Box<dyn Any>>,
    tmp1: &N_Vector,
    tmp2: &N_Vector,
    tmp3: &N_Vector,
) -> i32;

pub type IDALsPrecSetupFn = fn(
    tt: sunrealtype,
    yy: &N_Vector,
    yp: &N_Vector,
    rr: &N_Vector,
    c_j: sunrealtype,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32;

pub type IDALsPrecSolveFn = fn(
    tt: sunrealtype,
    yy: &N_Vector,
    yp: &N_Vector,
    rr: &N_Vector,
    rvec: &N_Vector,
    zvec: &N_Vector,
    c_j: sunrealtype,
    delta: sunrealtype,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32;

pub type IDALsJacTimesSetupFn = fn(
    tt: sunrealtype,
    yy: &N_Vector,
    yp: &N_Vector,
    rr: &N_Vector,
    c_j: sunrealtype,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32;

pub type IDALsJacTimesVecFn = fn(
    tt: sunrealtype,
    yy: &N_Vector,
    yp: &N_Vector,
    rr: &N_Vector,
    v: &N_Vector,
    Jv: &N_Vector,
    c_j: sunrealtype,
    user_data: &mut Option<Box<dyn Any>>,
    tmp1: &N_Vector,
    tmp2: &N_Vector,
) -> i32;

/*-----------------------------------------------------------------
  Types : IDALsMemRec, IDALsMem (ida_ls_impl.h)
  -----------------------------------------------------------------*/

pub struct IDALsMemRec {
    /* Linear solver type information */
    pub iterative: sunbooleantype,   /* is the solver iterative?    */
    pub matrixbased: sunbooleantype, /* is a matrix structure used? */

    /* Jacobian construction & storage */
    pub jacDQ: sunbooleantype,   /* SUNTRUE if using internal DQ Jacobian approx. */
    pub jac: Option<IDALsJacFn>, /* Jacobian routine to be called                 */
    /* C `J_data`: `Some` = module-owned token (an IDAMem clone for the
    internal DQ routine); `None` = pass `ida_user_data` at call time. */
    pub J_data: Option<Box<dyn Any>>,

    /* Linear solver, matrix and vector objects/pointers */
    pub LS: SUNLinearSolver,        /* generic linear solver object          */
    pub J: Option<SUNMatrix>,       /* J = dF/dy + cj*dF/dy'                 */
    pub ytemp: Option<N_Vector>,    /* temp vector used by IDAAtimesDQ       */
    pub yptemp: Option<N_Vector>,   /* temp vector used by IDAAtimesDQ       */
    pub x: Option<N_Vector>,        /* temp vector used by the solve function */
    pub ycur: Option<N_Vector>,     /* current y vector in Newton iteration  */
    pub ypcur: Option<N_Vector>,    /* current yp vector in Newton iteration */
    pub rcur: Option<N_Vector>,     /* rcur = F(tn, ycur, ypcur)             */

    /* Matrix-based solver, scale solution to account for change in cj */
    pub scalesol: sunbooleantype,

    /* Iterative solver tolerance */
    pub eplifac: sunrealtype, /* nonlinear -> linear tol scaling factor       */
    pub nrmfac: sunrealtype,  /* integrator -> LS norm conversion factor      */

    /* Statistics and associated parameters */
    pub dqincfac: sunrealtype, /* dqincfac = optional increment factor in Jv   */
    pub nje: i64,              /* nje = no. of calls to jac                    */
    pub npe: i64,              /* npe = total number of precond calls          */
    pub nli: i64,              /* nli = total number of linear iterations      */
    pub nps: i64,              /* nps = total number of psolve calls           */
    pub ncfl: i64,             /* ncfl = total number of convergence failures  */
    pub nreDQ: i64,            /* nreDQ = total number of calls to res         */
    pub njtsetup: i64,         /* njtsetup = total number of calls to jtsetup  */
    pub njtimes: i64,          /* njtimes = total number of calls to jtimes    */
    pub nst0: i64,             /* nst0 = saved nst (for performance monitor)   */
    pub nni0: i64,             /* nni0 = saved nni (for performance monitor)   */
    pub ncfn0: i64,            /* ncfn0 = saved ncfn (for performance monitor) */
    pub ncfl0: i64,            /* ncfl0 = saved ncfl (for performance monitor) */
    pub nwarn: i64,            /* nwarn = no. of warnings (for perf. monitor)  */
    pub nstlj: i64,            /* nstlj = nst at last jac/pset call            */
    pub tnlj: sunrealtype,     /* tnlj = t_n at last jac/pset call             */

    pub last_flag: i32, /* last error return flag                       */

    /* Preconditioner computation
     * (a) user-provided:
     *     - pdata == user_data (here: `None` = pass ida_user_data)
     *     - pfree == NULL (the user deallocates memory)
     * (b) internal preconditioner module
     *     - pdata == module token (`Some`)
     *     - pfree == set by the prec. module and called in idaLsFree */
    pub pset: Option<IDALsPrecSetupFn>,
    pub psolve: Option<IDALsPrecSolveFn>,
    pub pfree: Option<fn(IDA_mem: &IDAMem) -> i32>,
    pub pdata: Option<Box<dyn Any>>,

    /* Jacobian times vector computation
     * (a) jtimes function provided by the user:
     *     - jt_data == user_data (here: `None`)
     *     - jtimesDQ == SUNFALSE
     * (b) internal jtimes
     *     - jt_data == ida_mem token (`Some`)
     *     - jtimesDQ == SUNTRUE */
    pub jtimesDQ: sunbooleantype,
    pub jtsetup: Option<IDALsJacTimesSetupFn>,
    pub jtimes: Option<IDALsJacTimesVecFn>,
    pub jt_res: Option<IDAResFn>,
    pub jt_data: Option<Box<dyn Any>>,
}

pub type IDALsMem = Box<IDALsMemRec>;

/*---------------------------------------------------------------
  Error and Warning Messages (ida_ls_impl.h)

  `MSG_LS_TIME` = "at t = " SUN_FORMAT_G ", " and `MSG_LS_FRMT` =
  SUN_FORMAT_G "." are folded into the two warning builders below
  (SUN_FORMAT_G = "%.15g" = `sun_format_g`).
  ---------------------------------------------------------------*/

/* Error Messages */
pub const MSG_LS_IDAMEM_NULL: &str = "Integrator memory is NULL.";
pub const MSG_LS_MEM_FAIL: &str = "A memory request failed.";
pub const MSG_LS_BAD_NVECTOR: &str = "A required vector operation is not implemented.";
pub const MSG_LS_BAD_SIZES: &str =
    "Illegal bandwidth parameter(s). Must have 0 <=  ml, mu <= N-1.";
pub const MSG_LS_BAD_LSTYPE: &str = "Incompatible linear solver type.";
pub const MSG_LS_LMEM_NULL: &str = "Linear solver memory is NULL.";
pub const MSG_LS_BAD_GSTYPE: &str = "gstype has an illegal value.";
pub const MSG_LS_NEG_MAXRS: &str = "maxrs < 0 illegal.";
pub const MSG_LS_NEG_EPLIFAC: &str = "eplifac < 0.0 illegal.";
pub const MSG_LS_NEG_DQINCFAC: &str = "dqincfac < 0.0 illegal.";
pub const MSG_LS_PSET_FAILED: &str =
    "The preconditioner setup routine failed in an unrecoverable manner.";
pub const MSG_LS_PSOLVE_FAILED: &str =
    "The preconditioner solve routine failed in an unrecoverable manner.";
pub const MSG_LS_JTSETUP_FAILED: &str =
    "The Jacobian x vector setup routine failed in an unrecoverable manner.";
pub const MSG_LS_JTIMES_FAILED: &str =
    "The Jacobian x vector routine failed in an unrecoverable manner.";
pub const MSG_LS_JACFUNC_FAILED: &str = "The Jacobian routine failed in an unrecoverable manner.";
pub const MSG_LS_MATZERO_FAILED: &str = "The SUNMatZero routine failed in an unrecoverable manner.";

/* Warning Messages */
pub fn MSG_LS_CFN_WARN(t: sunrealtype, rate: sunrealtype) -> String {
    format!(
        "Warning: at t = {}, poor iterative algorithm performance. Nonlinear convergence failure \
         rate is {}.",
        sun_format_g(t),
        sun_format_g(rate)
    )
}

pub fn MSG_LS_CFL_WARN(t: sunrealtype, rate: sunrealtype) -> String {
    format!(
        "Warning: at t = {}, poor iterative algorithm performance. Linear convergence failure \
         rate is {}.",
        sun_format_g(t),
        sun_format_g(rate)
    )
}

/// Downcast helper: view `ida_mem.ida_lmem` as the IDALS memory record.
/// Panics if no linear solver memory is attached or it is not an IDALS
/// record (the C code blindly casts the `void*` — UB → panic, accepted
/// deviation class 5). Callers that must *probe* attachment without
/// panicking (e.g. IDABBDPRE returning `IDALS_LMEM_NULL`) test
/// `ida_mem.borrow().ida_lmem.as_ref().is_some_and(|b| b.is::<IDALsMemRec>())`
/// first.
///
/// The returned guard IS a borrow of the integrator memory: NEVER hold
/// it across `IDAProcessError`, a user callback, an N_Vector/SUNMatrix/
/// SUNLinearSolver/SUNNonlinearSolver op, or another borrow of the same
/// mem — copy the fields out in a scoped block, drop the guard, then
/// call.
pub fn idals_mem_mut(ida_mem: &IDAMem) -> RefMut<'_, IDALsMemRec> {
    RefMut::map(ida_mem.borrow_mut(), |m| {
        m.ida_lmem
            .as_mut()
            .expect("ida_lmem set")
            .downcast_mut::<IDALsMemRec>()
            .expect("IDALS linear solver memory")
    })
}

/*===============================================================
  IDALS Exported functions -- Required
  ===============================================================*/

/*---------------------------------------------------------------
  IDASetLinearSolver specifies the linear solver
  ---------------------------------------------------------------*/
pub fn IDASetLinearSolver(ida_mem: &IDAMem, LS: &SUNLinearSolver, A: Option<&SUNMatrix>) -> i32 {
    /* NULL ida_mem check: handled by type system */
    /* NULL LS check: handled by type system */

    /* Test if solver is compatible with LS interface */
    {
        let ops = LS.ops.borrow();
        if ops.gettype.is_none() || ops.solve.is_none() {
            IDAProcessError(
                Some(ida_mem),
                IDALS_ILL_INPUT,
                line!() as i32,
                "IDASetLinearSolver",
                file!(),
                "LS object is missing a required operation",
            );
            return IDALS_ILL_INPUT;
        }
    }

    /* Retrieve the LS type */
    let LSType = SUNLinSolGetType(LS);

    /* Set flags based on LS type */
    let iterative = LSType != SUNLINEARSOLVER_DIRECT;
    let matrixbased =
        (LSType != SUNLINEARSOLVER_ITERATIVE) && (LSType != SUNLINEARSOLVER_MATRIX_EMBEDDED);

    /* Test if vector is compatible with LS interface */
    let ida_tempv1 = ida_mem
        .borrow()
        .ida_tempv1
        .as_ref()
        .expect("ida_tempv1") /* C dereferences unconditionally (UB if unset) */
        .clone();
    {
        let ops = ida_tempv1.ops.borrow();
        if ops.nvconst.is_none() || ops.nvwrmsnorm.is_none() {
            IDAProcessError(
                Some(ida_mem),
                IDALS_ILL_INPUT,
                line!() as i32,
                "IDASetLinearSolver",
                file!(),
                MSG_LS_BAD_NVECTOR,
            );
            return IDALS_ILL_INPUT;
        }
    }

    /* Ensure that A is NULL when LS is matrix-embedded */
    if (LSType == SUNLINEARSOLVER_MATRIX_EMBEDDED) && A.is_some() {
        IDAProcessError(
            Some(ida_mem),
            IDALS_ILL_INPUT,
            line!() as i32,
            "IDASetLinearSolver",
            file!(),
            "Incompatible inputs: matrix-embedded LS requires NULL matrix",
        );
        return IDALS_ILL_INPUT;
    }

    /* Check for compatible LS type, matrix and "atimes" support */
    if iterative {
        if ida_tempv1.ops.borrow().nvgetlength.is_none() {
            IDAProcessError(
                Some(ida_mem),
                IDALS_ILL_INPUT,
                line!() as i32,
                "IDASetLinearSolver",
                file!(),
                MSG_LS_BAD_NVECTOR,
            );
            return IDALS_ILL_INPUT;
        }

        if LSType != SUNLINEARSOLVER_MATRIX_EMBEDDED {
            let missing = {
                let ops = LS.ops.borrow();
                ops.resid.is_none() || ops.numiters.is_none()
            };
            if missing {
                IDAProcessError(
                    Some(ida_mem),
                    IDALS_ILL_INPUT,
                    line!() as i32,
                    "IDASetLinearSolver",
                    file!(),
                    "Iterative LS object requires 'resid' and 'numiters' routines",
                );
                return IDALS_ILL_INPUT;
            }
        }

        if !matrixbased
            && (LSType != SUNLINEARSOLVER_MATRIX_EMBEDDED)
            && LS.ops.borrow().setatimes.is_none()
        {
            IDAProcessError(
                Some(ida_mem),
                IDALS_ILL_INPUT,
                line!() as i32,
                "IDASetLinearSolver",
                file!(),
                "Incompatible inputs: iterative LS must support ATimes routine",
            );
            return IDALS_ILL_INPUT;
        }

        if matrixbased && A.is_none() {
            IDAProcessError(
                Some(ida_mem),
                IDALS_ILL_INPUT,
                line!() as i32,
                "IDASetLinearSolver",
                file!(),
                "Incompatible inputs: matrix-iterative LS requires non-NULL matrix",
            );
            return IDALS_ILL_INPUT;
        }
    } else if A.is_none() {
        IDAProcessError(
            Some(ida_mem),
            IDALS_ILL_INPUT,
            line!() as i32,
            "IDASetLinearSolver",
            file!(),
            "Incompatible inputs: direct LS requires non-NULL matrix",
        );
        return IDALS_ILL_INPUT;
    }

    /* free any existing system solver attached to IDA */
    let lfree = ida_mem.borrow().ida_lfree;
    if let Some(lfree) = lfree {
        lfree(ida_mem);
    }

    /* Set four main system linear solver function fields in IDA_mem */
    {
        let mut m = ida_mem.borrow_mut();
        m.ida_linit = Some(idaLsInitialize);
        m.ida_lsetup = Some(idaLsSetup);
        m.ida_lsolve = Some(idaLsSolve);
        m.ida_lfree = Some(idaLsFree);

        /* Set ida_lperf if using an iterative SUNLinearSolver object */
        m.ida_lperf = if iterative { Some(idaLsPerf) } else { None };
    }

    /* Allocate memory for IDALsMemRec (C: malloc + memset(0), then the
    default assignments below; malloc failure is unreachable here). The
    struct literal carries exactly the state the C code holds after its
    default-assignment block (through `last_flag = IDALS_SUCCESS`). */
    let ida_res = ida_mem.borrow().ida_res;
    let mut idals_mem: IDALsMem = Box::new(IDALsMemRec {
        /* set SUNLinearSolver pointer */
        LS: LS.clone(),
        /* Linear solver type information */
        iterative,
        matrixbased,
        /* Set defaults for Jacobian-related fields */
        J: A.cloned(),
        jacDQ: A.is_some(),
        jac: if A.is_some() {
            Some(idaLsDQJac as IDALsJacFn)
        } else {
            None
        },
        J_data: if A.is_some() {
            Some(Box::new(ida_mem.clone())) /* C: J_data = IDA_mem */
        } else {
            None
        },
        jtimesDQ: SUNTRUE,
        jtsetup: None,
        jtimes: Some(idaLsDQJtimes),
        jt_res: ida_res,
        jt_data: Some(Box::new(ida_mem.clone())), /* C: jt_data = IDA_mem */
        /* Set defaults for preconditioner-related fields */
        pset: None,
        psolve: None,
        pfree: None,
        pdata: None, /* C: pdata = IDA_mem->ida_user_data (pass-through) */
        /* Initialize counters (idaLsInitializeCounters below re-zeros) */
        nje: 0,
        npe: 0,
        nli: 0,
        nps: 0,
        ncfl: 0,
        nreDQ: 0,
        njtsetup: 0,
        njtimes: 0,
        /* Set default values for the rest of the Ls parameters */
        eplifac: PT05,
        dqincfac: ONE,
        last_flag: IDALS_SUCCESS,
        /* memset(0) baseline for fields assigned further below or only
        by idaLsPerf/idaLsSetup */
        scalesol: SUNFALSE,
        nrmfac: 0.0,
        ytemp: None,
        yptemp: None,
        x: None,
        ycur: None,
        ypcur: None,
        rcur: None,
        nst0: 0,
        nni0: 0,
        ncfn0: 0,
        ncfl0: 0,
        nwarn: 0,
        nstlj: 0,
        tnlj: 0.0,
    });

    /* Initialize counters */
    let _ = idaLsInitializeCounters(&mut idals_mem);

    /* If LS supports ATimes, attach IDALs routine */
    if LS.ops.borrow().setatimes.is_some() {
        let retval = SUNLinSolSetATimes(LS, Some(Box::new(ida_mem.clone())), Some(idaLsATimes));
        if retval != SUN_SUCCESS {
            IDAProcessError(
                Some(ida_mem),
                IDALS_SUNLS_FAIL,
                line!() as i32,
                "IDASetLinearSolver",
                file!(),
                "Error in calling SUNLinSolSetATimes",
            );
            drop(idals_mem);
            return IDALS_SUNLS_FAIL;
        }
    }

    /* If LS supports preconditioning, initialize pset/psol to NULL */
    if LS.ops.borrow().setpreconditioner.is_some() {
        let retval = SUNLinSolSetPreconditioner(LS, Some(Box::new(ida_mem.clone())), None, None);
        if retval != SUN_SUCCESS {
            IDAProcessError(
                Some(ida_mem),
                IDALS_SUNLS_FAIL,
                line!() as i32,
                "IDASetLinearSolver",
                file!(),
                "Error in calling SUNLinSolSetPreconditioner",
            );
            drop(idals_mem);
            return IDALS_SUNLS_FAIL;
        }
    }

    /* Allocate memory for ytemp, yptemp and x */
    match N_VClone(&ida_tempv1) {
        Some(ytemp) => idals_mem.ytemp = Some(ytemp),
        None => {
            IDAProcessError(
                Some(ida_mem),
                IDALS_MEM_FAIL,
                line!() as i32,
                "IDASetLinearSolver",
                file!(),
                MSG_LS_MEM_FAIL,
            );
            drop(idals_mem);
            return IDALS_MEM_FAIL;
        }
    }

    match N_VClone(&ida_tempv1) {
        Some(yptemp) => idals_mem.yptemp = Some(yptemp),
        None => {
            IDAProcessError(
                Some(ida_mem),
                IDALS_MEM_FAIL,
                line!() as i32,
                "IDASetLinearSolver",
                file!(),
                MSG_LS_MEM_FAIL,
            );
            if let Some(ytemp) = idals_mem.ytemp.take() {
                N_VDestroy(ytemp);
            }
            drop(idals_mem);
            return IDALS_MEM_FAIL;
        }
    }

    match N_VClone(&ida_tempv1) {
        Some(x) => idals_mem.x = Some(x),
        None => {
            IDAProcessError(
                Some(ida_mem),
                IDALS_MEM_FAIL,
                line!() as i32,
                "IDASetLinearSolver",
                file!(),
                MSG_LS_MEM_FAIL,
            );
            if let Some(ytemp) = idals_mem.ytemp.take() {
                N_VDestroy(ytemp);
            }
            if let Some(yptemp) = idals_mem.yptemp.take() {
                N_VDestroy(yptemp);
            }
            drop(idals_mem);
            return IDALS_MEM_FAIL;
        }
    }

    /* For iterative LS, compute sqrtN */
    if iterative {
        idals_mem.nrmfac =
            SUNRsqrt(N_VGetLength(idals_mem.ytemp.as_ref().expect("ytemp")) as sunrealtype);
    }

    /* For matrix-based LS, enable solution scaling */
    if matrixbased {
        idals_mem.scalesol = SUNTRUE;
    } else {
        idals_mem.scalesol = SUNFALSE;
    }

    /* Attach linear solver memory to integrator memory */
    ida_mem.borrow_mut().ida_lmem = Some(idals_mem);

    IDALS_SUCCESS
}

/*===============================================================
  Optional Set routines
  ===============================================================*/

/* IDASetJacFn specifies the Jacobian function */
pub fn IDASetJacFn(ida_mem: &IDAMem, jac: Option<IDALsJacFn>) -> i32 {
    /* access IDALsMem structure */
    let retval = idaLs_AccessLMem(ida_mem, "IDASetJacFn");
    if retval != IDALS_SUCCESS {
        return retval;
    }

    /* return with failure if jac cannot be used */
    if jac.is_some() && idals_mem_mut(ida_mem).J.is_none() {
        IDAProcessError(
            Some(ida_mem),
            IDALS_ILL_INPUT,
            line!() as i32,
            "IDASetJacFn",
            file!(),
            "Jacobian routine cannot be supplied for NULL SUNMatrix",
        );
        return IDALS_ILL_INPUT;
    }

    /* set Jacobian routine pointer, and update relevant flags */
    if jac.is_some() {
        let mut ls = idals_mem_mut(ida_mem);
        ls.jacDQ = SUNFALSE;
        ls.jac = jac;
        ls.J_data = None; /* C: J_data = IDA_mem->ida_user_data */
    } else {
        let mut ls = idals_mem_mut(ida_mem);
        ls.jacDQ = SUNTRUE;
        ls.jac = Some(idaLsDQJac);
        ls.J_data = Some(Box::new(ida_mem.clone())); /* C: J_data = IDA_mem */
    }

    IDALS_SUCCESS
}

/* IDASetEpsLin specifies the nonlinear -> linear tolerance scale factor */
pub fn IDASetEpsLin(ida_mem: &IDAMem, eplifac: sunrealtype) -> i32 {
    /* access IDALsMem structure */
    let retval = idaLs_AccessLMem(ida_mem, "IDASetEpsLin");
    if retval != IDALS_SUCCESS {
        return retval;
    }

    /* Check for legal eplifac */
    if eplifac < ZERO {
        IDAProcessError(
            Some(ida_mem),
            IDALS_ILL_INPUT,
            line!() as i32,
            "IDASetEpsLin",
            file!(),
            MSG_LS_NEG_EPLIFAC,
        );
        return IDALS_ILL_INPUT;
    }

    idals_mem_mut(ida_mem).eplifac = if eplifac == ZERO { PT05 } else { eplifac };

    IDALS_SUCCESS
}

/* IDASetWRMSNormFactor sets or computes the factor to use when converting from
   the integrator tolerance to the linear solver tolerance (WRMS to L2 norm). */
pub fn IDASetLSNormFactor(ida_mem: &IDAMem, nrmfac: sunrealtype) -> i32 {
    /* access IDALsMem structure */
    let retval = idaLs_AccessLMem(ida_mem, "IDASetLSNormFactor");
    if retval != IDALS_SUCCESS {
        return retval;
    }

    if nrmfac > ZERO {
        /* user-provided factor */
        idals_mem_mut(ida_mem).nrmfac = nrmfac;
    } else if nrmfac < ZERO {
        /* compute factor for WRMS norm with dot product */
        let ytemp = idals_mem_mut(ida_mem)
            .ytemp
            .as_ref()
            .expect("ytemp")
            .clone();
        N_VConst(ONE, &ytemp);
        idals_mem_mut(ida_mem).nrmfac = SUNRsqrt(N_VDotProd(&ytemp, &ytemp));
    } else {
        /* compute default factor for WRMS norm from vector length */
        let ytemp = idals_mem_mut(ida_mem)
            .ytemp
            .as_ref()
            .expect("ytemp")
            .clone();
        idals_mem_mut(ida_mem).nrmfac = SUNRsqrt(N_VGetLength(&ytemp) as sunrealtype);
    }

    IDALS_SUCCESS
}

/* IDASetLinearSolutionScaling enables or disables scaling the linear solver
   solution to account for changes in cj. */
pub fn IDASetLinearSolutionScaling(ida_mem: &IDAMem, onoff: sunbooleantype) -> i32 {
    /* access IDALsMem structure */
    let retval = idaLs_AccessLMem(ida_mem, "IDASetLinearSolutionScaling");
    if retval != IDALS_SUCCESS {
        return retval;
    }

    /* check for valid solver type */
    if !idals_mem_mut(ida_mem).matrixbased {
        return IDALS_ILL_INPUT;
    }

    /* set solution scaling flag */
    idals_mem_mut(ida_mem).scalesol = onoff;

    IDALS_SUCCESS
}

/* IDASetIncrementFactor specifies increment factor for DQ approximations to Jv */
pub fn IDASetIncrementFactor(ida_mem: &IDAMem, dqincfac: sunrealtype) -> i32 {
    /* access IDALsMem structure */
    let retval = idaLs_AccessLMem(ida_mem, "IDASetIncrementFactor");
    if retval != IDALS_SUCCESS {
        return retval;
    }

    /* Check for legal dqincfac */
    if dqincfac <= ZERO {
        IDAProcessError(
            Some(ida_mem),
            IDALS_ILL_INPUT,
            line!() as i32,
            "IDASetIncrementFactor",
            file!(),
            MSG_LS_NEG_DQINCFAC,
        );
        return IDALS_ILL_INPUT;
    }

    idals_mem_mut(ida_mem).dqincfac = dqincfac;

    IDALS_SUCCESS
}

/* IDASetPreconditioner specifies the user-supplied psetup and psolve routines */
pub fn IDASetPreconditioner(
    ida_mem: &IDAMem,
    psetup: Option<IDALsPrecSetupFn>,
    psolve: Option<IDALsPrecSolveFn>,
) -> i32 {
    /* access IDALsMem structure */
    let retval = idaLs_AccessLMem(ida_mem, "IDASetPreconditioner");
    if retval != IDALS_SUCCESS {
        return retval;
    }

    /* store function pointers for user-supplied routines in IDALs interface */
    {
        let mut ls = idals_mem_mut(ida_mem);
        ls.pset = psetup;
        ls.psolve = psolve;
    }

    /* issue error if LS object does not allow user-supplied preconditioning */
    let LS = idals_mem_mut(ida_mem).LS.clone();
    if LS.ops.borrow().setpreconditioner.is_none() {
        IDAProcessError(
            Some(ida_mem),
            IDALS_ILL_INPUT,
            line!() as i32,
            "IDASetPreconditioner",
            file!(),
            "SUNLinearSolver object does not support user-supplied preconditioning",
        );
        return IDALS_ILL_INPUT;
    }

    /* notify iterative linear solver to call IDALs interface routines */
    let idals_psetup: Option<SUNPSetupFn> = if psetup.is_none() {
        None
    } else {
        Some(idaLsPSetup)
    };
    let idals_psolve: Option<SUNPSolveFn> = if psolve.is_none() {
        None
    } else {
        Some(idaLsPSolve)
    };
    let retval = SUNLinSolSetPreconditioner(
        &LS,
        Some(Box::new(ida_mem.clone())),
        idals_psetup,
        idals_psolve,
    );
    if retval != SUN_SUCCESS {
        IDAProcessError(
            Some(ida_mem),
            IDALS_SUNLS_FAIL,
            line!() as i32,
            "IDASetPreconditioner",
            file!(),
            "Error in calling SUNLinSolSetPreconditioner",
        );
        return IDALS_SUNLS_FAIL;
    }

    IDALS_SUCCESS
}

/* IDASetJacTimes specifies the user-supplied Jacobian-vector product
   setup and multiply routines */
pub fn IDASetJacTimes(
    ida_mem: &IDAMem,
    jtsetup: Option<IDALsJacTimesSetupFn>,
    jtimes: Option<IDALsJacTimesVecFn>,
) -> i32 {
    /* access IDALsMem structure */
    let retval = idaLs_AccessLMem(ida_mem, "IDASetJacTimes");
    if retval != IDALS_SUCCESS {
        return retval;
    }

    /* issue error if LS object does not allow user-supplied ATimes */
    let LS = idals_mem_mut(ida_mem).LS.clone();
    if LS.ops.borrow().setatimes.is_none() {
        IDAProcessError(
            Some(ida_mem),
            IDALS_ILL_INPUT,
            line!() as i32,
            "IDASetJacTimes",
            file!(),
            "SUNLinearSolver object does not support user-supplied ATimes routine",
        );
        return IDALS_ILL_INPUT;
    }

    /* store function pointers for user-supplied routines in IDALs
    interface (NULL jtimes implies use of DQ default) */
    if jtimes.is_some() {
        let mut ls = idals_mem_mut(ida_mem);
        ls.jtimesDQ = SUNFALSE;
        ls.jtsetup = jtsetup;
        ls.jtimes = jtimes;
        ls.jt_data = None; /* C: jt_data = IDA_mem->ida_user_data */
    } else {
        let ida_res = ida_mem.borrow().ida_res;
        let mut ls = idals_mem_mut(ida_mem);
        ls.jtimesDQ = SUNTRUE;
        ls.jtsetup = None;
        ls.jtimes = Some(idaLsDQJtimes);
        ls.jt_res = ida_res;
        ls.jt_data = Some(Box::new(ida_mem.clone())); /* C: jt_data = IDA_mem */
    }

    IDALS_SUCCESS
}

/* IDASetJacTimesResFn specifies an alternative user-supplied DAE residual
   function to use in the internal finite difference Jacobian-vector
   product */
pub fn IDASetJacTimesResFn(ida_mem: &IDAMem, jtimesResFn: Option<IDAResFn>) -> i32 {
    /* access IDALsMem structure */
    let retval = idaLs_AccessLMem(ida_mem, "IDASetJacTimesResFn");
    if retval != IDALS_SUCCESS {
        return retval;
    }

    /* check if using internal finite difference approximation */
    if !idals_mem_mut(ida_mem).jtimesDQ {
        IDAProcessError(
            Some(ida_mem),
            IDALS_ILL_INPUT,
            line!() as i32,
            "IDASetJacTimesResFn",
            file!(),
            "Internal finite-difference Jacobian-vector product is disabled.",
        );
        return IDALS_ILL_INPUT;
    }

    /* store function pointers for Res function (NULL implies use DAE Res) */
    if jtimesResFn.is_some() {
        idals_mem_mut(ida_mem).jt_res = jtimesResFn;
    } else {
        let ida_res = ida_mem.borrow().ida_res;
        idals_mem_mut(ida_mem).jt_res = ida_res;
    }

    IDALS_SUCCESS
}

/*===============================================================
  Optional Get routines
  ===============================================================*/

pub fn IDAGetJac(ida_mem: &IDAMem, J: &mut Option<SUNMatrix>) -> i32 {
    /* access IDALsMem structure; set output and return */
    let retval = idaLs_AccessLMem(ida_mem, "IDAGetJac");
    if retval != IDALS_SUCCESS {
        return retval;
    }
    *J = idals_mem_mut(ida_mem).J.clone();
    IDALS_SUCCESS
}

pub fn IDAGetJacCj(ida_mem: &IDAMem, cj_J: &mut sunrealtype) -> i32 {
    /* access IDALsMem structure; set output and return */
    let retval = idaLs_AccessLMem(ida_mem, "IDAGetJacCj");
    if retval != IDALS_SUCCESS {
        return retval;
    }
    *cj_J = ida_mem.borrow().ida_cjold;
    IDALS_SUCCESS
}

pub fn IDAGetJacTime(ida_mem: &IDAMem, t_J: &mut sunrealtype) -> i32 {
    /* access IDALsMem structure; set output and return */
    let retval = idaLs_AccessLMem(ida_mem, "IDAGetJacTime");
    if retval != IDALS_SUCCESS {
        return retval;
    }
    *t_J = idals_mem_mut(ida_mem).tnlj;
    IDALS_SUCCESS
}

pub fn IDAGetJacNumSteps(ida_mem: &IDAMem, nst_J: &mut i64) -> i32 {
    /* access IDALsMem structure; set output and return */
    let retval = idaLs_AccessLMem(ida_mem, "IDAGetJacNumSteps");
    if retval != IDALS_SUCCESS {
        return retval;
    }
    *nst_J = idals_mem_mut(ida_mem).nstlj;
    IDALS_SUCCESS
}

/* IDAGetLinWorkSpace returns the length of workspace allocated
   for the IDALS linear solver interface */
pub fn IDAGetLinWorkSpace(ida_mem: &IDAMem, lenrwLS: &mut i64, leniwLS: &mut i64) -> i32 {
    /* access IDALsMem structure */
    let retval = idaLs_AccessLMem(ida_mem, "IDAGetLinWorkSpace");
    if retval != IDALS_SUCCESS {
        return retval;
    }

    /* start with fixed sizes plus vector/matrix pointers */
    *lenrwLS = 3;
    *leniwLS = 33;

    /* add N_Vector sizes */
    let ida_tempv1 = ida_mem
        .borrow()
        .ida_tempv1
        .as_ref()
        .expect("ida_tempv1")
        .clone();
    if ida_tempv1.ops.borrow().nvspace.is_some() {
        let mut lrw1: sunindextype = 0;
        let mut liw1: sunindextype = 0;
        N_VSpace(&ida_tempv1, &mut lrw1, &mut liw1);
        *lenrwLS += 3 * lrw1;
        *leniwLS += 3 * liw1;
    }

    /* add LS sizes */
    let LS = idals_mem_mut(ida_mem).LS.clone();
    if LS.ops.borrow().space.is_some() {
        let mut lrw: i64 = 0;
        let mut liw: i64 = 0;
        let retval = SUNLinSolSpace(&LS, &mut lrw, &mut liw);
        if retval == 0 {
            *lenrwLS += lrw;
            *leniwLS += liw;
        }
    }

    IDALS_SUCCESS
}

/* IDAGetNumJacEvals returns the number of Jacobian evaluations */
pub fn IDAGetNumJacEvals(ida_mem: &IDAMem, njevals: &mut i64) -> i32 {
    /* access IDALsMem structure; store output and return */
    let retval = idaLs_AccessLMem(ida_mem, "IDAGetNumJacEvals");
    if retval != IDALS_SUCCESS {
        return retval;
    }
    *njevals = idals_mem_mut(ida_mem).nje;
    IDALS_SUCCESS
}

/* IDAGetNumPrecEvals returns the number of preconditioner evaluations */
pub fn IDAGetNumPrecEvals(ida_mem: &IDAMem, npevals: &mut i64) -> i32 {
    /* access IDALsMem structure; store output and return */
    let retval = idaLs_AccessLMem(ida_mem, "IDAGetNumPrecEvals");
    if retval != IDALS_SUCCESS {
        return retval;
    }
    *npevals = idals_mem_mut(ida_mem).npe;
    IDALS_SUCCESS
}

/* IDAGetNumPrecSolves returns the number of preconditioner solves */
pub fn IDAGetNumPrecSolves(ida_mem: &IDAMem, npsolves: &mut i64) -> i32 {
    /* access IDALsMem structure; store output and return */
    let retval = idaLs_AccessLMem(ida_mem, "IDAGetNumPrecSolves");
    if retval != IDALS_SUCCESS {
        return retval;
    }
    *npsolves = idals_mem_mut(ida_mem).nps;
    IDALS_SUCCESS
}

/* IDAGetNumLinIters returns the number of linear iterations */
pub fn IDAGetNumLinIters(ida_mem: &IDAMem, nliters: &mut i64) -> i32 {
    /* access IDALsMem structure; store output and return */
    let retval = idaLs_AccessLMem(ida_mem, "IDAGetNumLinIters");
    if retval != IDALS_SUCCESS {
        return retval;
    }
    *nliters = idals_mem_mut(ida_mem).nli;
    IDALS_SUCCESS
}

/* IDAGetNumLinConvFails returns the number of linear convergence failures */
pub fn IDAGetNumLinConvFails(ida_mem: &IDAMem, nlcfails: &mut i64) -> i32 {
    /* access IDALsMem structure; store output and return */
    let retval = idaLs_AccessLMem(ida_mem, "IDAGetNumLinConvFails");
    if retval != IDALS_SUCCESS {
        return retval;
    }
    *nlcfails = idals_mem_mut(ida_mem).ncfl;
    IDALS_SUCCESS
}

/* IDAGetNumJTSetupEvals returns the number of calls to the
   user-supplied Jacobian-vector product setup routine */
pub fn IDAGetNumJTSetupEvals(ida_mem: &IDAMem, njtsetups: &mut i64) -> i32 {
    /* access IDALsMem structure; store output and return */
    let retval = idaLs_AccessLMem(ida_mem, "IDAGetNumJTSetupEvals");
    if retval != IDALS_SUCCESS {
        return retval;
    }
    *njtsetups = idals_mem_mut(ida_mem).njtsetup;
    IDALS_SUCCESS
}

/* IDAGetNumJtimesEvals returns the number of calls to the
   Jacobian-vector product multiply routine */
pub fn IDAGetNumJtimesEvals(ida_mem: &IDAMem, njvevals: &mut i64) -> i32 {
    /* access IDALsMem structure; store output and return */
    let retval = idaLs_AccessLMem(ida_mem, "IDAGetNumJtimesEvals");
    if retval != IDALS_SUCCESS {
        return retval;
    }
    *njvevals = idals_mem_mut(ida_mem).njtimes;
    IDALS_SUCCESS
}

/* IDAGetNumLinResEvals returns the number of calls to the DAE
   residual needed for the DQ Jacobian approximation or J*v
   product approximation */
pub fn IDAGetNumLinResEvals(ida_mem: &IDAMem, nrevalsLS: &mut i64) -> i32 {
    /* access IDALsMem structure; store output and return */
    let retval = idaLs_AccessLMem(ida_mem, "IDAGetNumLinResEvals");
    if retval != IDALS_SUCCESS {
        return retval;
    }
    *nrevalsLS = idals_mem_mut(ida_mem).nreDQ;
    IDALS_SUCCESS
}

/* IDAGetLastLinFlag returns the last flag set in a IDALS function */
pub fn IDAGetLastLinFlag(ida_mem: &IDAMem, flag: &mut i64) -> i32 {
    /* access IDALsMem structure; store output and return */
    let retval = idaLs_AccessLMem(ida_mem, "IDAGetLastLinFlag");
    if retval != IDALS_SUCCESS {
        return retval;
    }
    *flag = idals_mem_mut(ida_mem).last_flag as i64;
    IDALS_SUCCESS
}

/* IDAGetLinReturnFlagName translates from the integer error code
   returned by an IDALs routine to the corresponding string
   equivalent for that flag */
pub fn IDAGetLinReturnFlagName(flag: i64) -> String {
    let name = if flag == IDALS_SUCCESS as i64 {
        "IDALS_SUCCESS"
    } else if flag == IDALS_MEM_NULL as i64 {
        "IDALS_MEM_NULL"
    } else if flag == IDALS_LMEM_NULL as i64 {
        "IDALS_LMEM_NULL"
    } else if flag == IDALS_ILL_INPUT as i64 {
        "IDALS_ILL_INPUT"
    } else if flag == IDALS_MEM_FAIL as i64 {
        "IDALS_MEM_FAIL"
    } else if flag == IDALS_PMEM_NULL as i64 {
        "IDALS_PMEM_NULL"
    } else if flag == IDALS_JACFUNC_UNRECVR as i64 {
        "IDALS_JACFUNC_UNRECVR"
    } else if flag == IDALS_JACFUNC_RECVR as i64 {
        "IDALS_JACFUNC_RECVR"
    } else if flag == IDALS_SUNMAT_FAIL as i64 {
        "IDALS_SUNMAT_FAIL"
    } else if flag == IDALS_SUNLS_FAIL as i64 {
        "IDALS_SUNLS_FAIL"
    } else {
        "NONE"
    };
    name.to_string()
}

/*===============================================================
  IDALS Private functions
  ===============================================================*/

/*---------------------------------------------------------------
  idaLsATimes:

  This routine generates the matrix-vector product z = Jv, where
  J is the system Jacobian, by calling either the user provided
  routine or the internal DQ routine.  The return value is
  the same as the value returned by jtimes --
  0 if successful, nonzero otherwise.
  ---------------------------------------------------------------*/
pub fn idaLsATimes(ida_mem: &mut Option<Box<dyn Any>>, v: &N_Vector, z: &N_Vector) -> i32 {
    /* access IDALsMem structure */
    let IDA_mem = match idaLs_AccessLMemToken(ida_mem, "idaLsATimes") {
        Ok(m) => m,
        Err(retval) => return retval,
    };

    /* call Jacobian-times-vector product routine
    (either user-supplied or internal DQ) */
    let (tn, cj) = {
        let m = IDA_mem.borrow();
        (m.ida_tn, m.ida_cj)
    };
    let (jtimes, ycur, ypcur, rcur, ytemp, yptemp) = {
        let ls = idals_mem_mut(&IDA_mem);
        (
            ls.jtimes.expect("jtimes"),
            ls.ycur.as_ref().expect("ycur").clone(),
            ls.ypcur.as_ref().expect("ypcur").clone(),
            ls.rcur.as_ref().expect("rcur").clone(),
            ls.ytemp.as_ref().expect("ytemp").clone(),
            ls.yptemp.as_ref().expect("yptemp").clone(),
        )
    };
    let use_field = idals_mem_mut(&IDA_mem).jt_data.is_some();
    let mut jt_data = if use_field {
        idals_mem_mut(&IDA_mem).jt_data.take()
    } else {
        IDA_mem.borrow_mut().ida_user_data.take()
    };
    let retval = jtimes(
        tn,
        &ycur,
        &ypcur,
        &rcur,
        v,
        z,
        cj,
        &mut jt_data,
        &ytemp,
        &yptemp,
    );
    if use_field {
        idals_mem_mut(&IDA_mem).jt_data = jt_data;
    } else {
        IDA_mem.borrow_mut().ida_user_data = jt_data;
    }
    idals_mem_mut(&IDA_mem).njtimes += 1;
    retval
}

/*---------------------------------------------------------------
  idaLsPSetup:

  This routine interfaces between the generic iterative linear
  solvers and the user's psetup routine.  It passes to psetup all
  required state information from ida_mem.  Its return value
  is the same as that returned by psetup. Note that the generic
  iterative linear solvers guarantee that idaLsPSetup will only
  be called in the case that the user's psetup routine is non-NULL.
  ---------------------------------------------------------------*/
pub fn idaLsPSetup(ida_mem: &mut Option<Box<dyn Any>>) -> i32 {
    /* access IDALsMem structure */
    let IDA_mem = match idaLs_AccessLMemToken(ida_mem, "idaLsPSetup") {
        Ok(m) => m,
        Err(retval) => return retval,
    };

    /* Call user pset routine to update preconditioner and possibly
    reset jcur (pass !jbad as update suggestion) */
    let (tn, cj) = {
        let m = IDA_mem.borrow();
        (m.ida_tn, m.ida_cj)
    };
    let (pset, ycur, ypcur, rcur) = {
        let ls = idals_mem_mut(&IDA_mem);
        (
            ls.pset.expect("pset"),
            ls.ycur.as_ref().expect("ycur").clone(),
            ls.ypcur.as_ref().expect("ypcur").clone(),
            ls.rcur.as_ref().expect("rcur").clone(),
        )
    };
    let use_field = idals_mem_mut(&IDA_mem).pdata.is_some();
    let mut pdata = if use_field {
        idals_mem_mut(&IDA_mem).pdata.take()
    } else {
        IDA_mem.borrow_mut().ida_user_data.take()
    };
    let retval = pset(tn, &ycur, &ypcur, &rcur, cj, &mut pdata);
    if use_field {
        idals_mem_mut(&IDA_mem).pdata = pdata;
    } else {
        IDA_mem.borrow_mut().ida_user_data = pdata;
    }
    idals_mem_mut(&IDA_mem).npe += 1;
    retval
}

/*---------------------------------------------------------------
  idaLsPSolve:

  This routine interfaces between the generic SUNLinSolSolve
  routine and the user's psolve routine.  It passes to psolve all
  required state information from ida_mem.  Its return value is
  the same as that returned by psolve.  Note that the generic
  SUNLinSol solver guarantees that IDASilsPSolve will not be
  called in the case in which preconditioning is not done. This
  is the only case in which the user's psolve routine is allowed
  to be NULL.
  ---------------------------------------------------------------*/
pub fn idaLsPSolve(
    ida_mem: &mut Option<Box<dyn Any>>,
    r: &N_Vector,
    z: &N_Vector,
    tol: sunrealtype,
    lr: i32,
) -> i32 {
    let _ = lr; /* SUNDIALS_MAYBE_UNUSED */

    /* access IDALsMem structure */
    let IDA_mem = match idaLs_AccessLMemToken(ida_mem, "idaLsPSolve") {
        Ok(m) => m,
        Err(retval) => return retval,
    };

    /* call the user-supplied psolve routine, and accumulate count */
    let (tn, cj) = {
        let m = IDA_mem.borrow();
        (m.ida_tn, m.ida_cj)
    };
    let (psolve, ycur, ypcur, rcur) = {
        let ls = idals_mem_mut(&IDA_mem);
        (
            ls.psolve.expect("psolve"),
            ls.ycur.as_ref().expect("ycur").clone(),
            ls.ypcur.as_ref().expect("ypcur").clone(),
            ls.rcur.as_ref().expect("rcur").clone(),
        )
    };
    let use_field = idals_mem_mut(&IDA_mem).pdata.is_some();
    let mut pdata = if use_field {
        idals_mem_mut(&IDA_mem).pdata.take()
    } else {
        IDA_mem.borrow_mut().ida_user_data.take()
    };
    let retval = psolve(tn, &ycur, &ypcur, &rcur, r, z, cj, tol, &mut pdata);
    if use_field {
        idals_mem_mut(&IDA_mem).pdata = pdata;
    } else {
        IDA_mem.borrow_mut().ida_user_data = pdata;
    }
    idals_mem_mut(&IDA_mem).nps += 1;
    retval
}

/*---------------------------------------------------------------
  idaLsDQJac:

  This routine is a wrapper for the Dense and Band
  implementations of the difference quotient Jacobian
  approximation routines.
---------------------------------------------------------------*/
#[allow(clippy::too_many_arguments)]
pub fn idaLsDQJac(
    t: sunrealtype,
    c_j: sunrealtype,
    y: &N_Vector,
    yp: &N_Vector,
    r: &N_Vector,
    Jac: &SUNMatrix,
    ida_mem: &mut Option<Box<dyn Any>>,
    tmp1: &N_Vector,
    tmp2: &N_Vector,
    tmp3: &N_Vector,
) -> i32 {
    /* access IDAMem structure */
    let IDA_mem = match ida_mem
        .as_ref()
        .and_then(|b| b.downcast_ref::<IDAMem>())
        .cloned()
    {
        Some(m) => m,
        None => {
            IDAProcessError(
                None,
                IDALS_MEM_NULL,
                line!() as i32,
                "idaLsDQJac",
                file!(),
                MSG_LS_IDAMEM_NULL,
            );
            return IDALS_MEM_NULL;
        }
    };

    /* Jac non-NULL check: handled by type system */

    /* Verify that N_Vector supports required operations */
    let ida_tempv1 = IDA_mem
        .borrow()
        .ida_tempv1
        .as_ref()
        .expect("ida_tempv1")
        .clone();
    {
        let ops = ida_tempv1.ops.borrow();
        if ops.nvcloneempty.is_none()
            || ops.nvlinearsum.is_none()
            || ops.nvdestroy.is_none()
            || ops.nvscale.is_none()
            || ops.nvgetarraypointer.is_none()
            || ops.nvsetarraypointer.is_none()
        {
            IDAProcessError(
                Some(&IDA_mem),
                IDALS_ILL_INPUT,
                line!() as i32,
                "idaLsDQJac",
                file!(),
                MSG_LS_BAD_NVECTOR,
            );
            return IDALS_ILL_INPUT;
        }
    }

    /* Call the matrix-structure-specific DQ approximation routine */
    let retval;
    if SUNMatGetID(Jac) == SUNMATRIX_DENSE {
        retval = idaLsDenseDQJac(t, c_j, y, yp, r, Jac, &IDA_mem, tmp1);
    } else if SUNMatGetID(Jac) == SUNMATRIX_BAND {
        retval = idaLsBandDQJac(t, c_j, y, yp, r, Jac, &IDA_mem, tmp1, tmp2, tmp3);
    } else {
        IDAProcessError(
            Some(&IDA_mem),
            IDA_ILL_INPUT,
            line!() as i32,
            "idaLsDQJac",
            file!(),
            "unrecognized matrix type for idaLsDQJac",
        );
        retval = IDA_ILL_INPUT;
    }
    retval
}

/*---------------------------------------------------------------
  idaLsDenseDQJac

  This routine generates a dense difference quotient approximation
  to the Jacobian F_y + c_j*F_y'. It assumes a dense SUNmatrix
  input (stored column-wise, and that elements within each column
  are contiguous). The jth column is computed into the `jthCol`
  vector via N_VLinearSum and written back into the matrix column
  (the C code aliases the column memory with N_VSetArrayPointer;
  the copy-in/copy-out here is bit-identical).
---------------------------------------------------------------*/
#[allow(clippy::too_many_arguments)]
pub fn idaLsDenseDQJac(
    tt: sunrealtype,
    c_j: sunrealtype,
    yy: &N_Vector,
    yp: &N_Vector,
    rr: &N_Vector,
    Jac: &SUNMatrix,
    IDA_mem: &IDAMem,
    tmp1: &N_Vector,
) -> i32 {
    let mut retval: i32 = 0;

    /* access matrix dimension */
    let N = SUNDenseMatrix_Columns(Jac);

    /* Rename work vectors for readability */
    let rtemp = tmp1;

    /* Create an empty vector for matrix column calculations */
    let jthCol = N_VCloneEmpty(tmp1).expect("N_VCloneEmpty");

    /* Obtain integrator state (C caches raw data pointers for ewt, yy,
    yp and the constraints vector before the loop; here the data borrows
    are re-taken per use and never held across the residual callback or
    a vector op. The `hh`/`res` reads are hoisted — accepted deviation
    class 7). */
    let (uround, hh, ewt, constraints, res) = {
        let m = IDA_mem.borrow();
        (
            m.ida_uround,
            m.ida_hh,
            m.ida_ewt.as_ref().expect("ida_ewt").clone(),
            m.ida_constraints.clone(),
            m.ida_res.expect("ida_res"),
        )
    };

    let srur = SUNRsqrt(uround);

    let mut j: sunindextype = 0;
    while j < N {
        /* Generate the jth col of J(tt,yy,yp) as delta(F)/delta(y_j). */

        /* Set data address of jthCol, and save y_j and yp_j values. */
        let col_data = SUNDenseMatrix_Column(Jac, j).to_vec();
        N_VSetArrayPointer(col_data, &jthCol);

        let yj;
        let ypj;
        let mut inc;
        {
            let y_data = N_VGetArrayPointer(yy).expect("yy data");
            let yp_data = N_VGetArrayPointer(yp).expect("yp data");
            let ewt_data = N_VGetArrayPointer(&ewt).expect("ewt data");
            yj = y_data[j as usize];
            ypj = yp_data[j as usize];

            /* Set increment inc to y_j based on sqrt(uround)*abs(y_j), with
            adjustments using yp_j and ewt_j if this is small, and a further
            adjustment to give it the same sign as hh*yp_j. */

            inc = SUNMAX(
                srur * SUNMAX(SUNRabs(yj), SUNRabs(hh * ypj)),
                ONE / ewt_data[j as usize],
            );
        }

        if hh * ypj < ZERO {
            inc = -inc;
        }
        inc = (yj + inc) - yj;

        /* Adjust sign(inc) again if y_j has an inequality constraint. */
        if let Some(constraints) = &constraints {
            let cns_data = N_VGetArrayPointer(constraints).expect("constraints data");
            let conj = cns_data[j as usize];
            if SUNRabs(conj) == ONE {
                if (yj + inc) * conj < ZERO {
                    inc = -inc;
                }
            } else if SUNRabs(conj) == TWO && (yj + inc) * conj <= ZERO {
                inc = -inc;
            }
        }

        /* Increment y_j and yp_j, call res, and break on error return. */
        {
            let mut y_data = N_VGetArrayPointer(yy).expect("yy data");
            y_data[j as usize] += inc;
        }
        {
            let mut yp_data = N_VGetArrayPointer(yp).expect("yp data");
            yp_data[j as usize] += c_j * inc;
        }

        let mut user_data = IDA_mem.borrow_mut().ida_user_data.take();
        retval = res(tt, yy, yp, rtemp, &mut user_data);
        IDA_mem.borrow_mut().ida_user_data = user_data;
        idals_mem_mut(IDA_mem).nreDQ += 1;
        if retval != 0 {
            break;
        }

        /* Construct difference quotient in jthCol */
        let inc_inv = ONE / inc;
        N_VLinearSum(inc_inv, rtemp, -inc_inv, rr, &jthCol);

        /* write the computed column back into the matrix (C wrote it
        through the aliased column pointer) */
        {
            let jth_data = N_VGetArrayPointer(&jthCol).expect("jthCol data");
            let mut col_j = SUNDenseMatrix_Column(Jac, j);
            col_j.copy_from_slice(&jth_data);
        }

        /*  reset y_j, yp_j */
        {
            let mut y_data = N_VGetArrayPointer(yy).expect("yy data");
            y_data[j as usize] = yj;
        }
        {
            let mut yp_data = N_VGetArrayPointer(yp).expect("yp data");
            yp_data[j as usize] = ypj;
        }

        j += 1;
    }

    /* Destroy jthCol vector */
    N_VSetArrayPointer(Vec::new(), &jthCol); /* SHOULDN'T BE NEEDED */
    N_VDestroy(jthCol);

    retval
}

/*---------------------------------------------------------------
  idaLsBandDQJac

  This routine generates a banded difference quotient approximation
  JJ to the DAE system Jacobian J.  It assumes a band SUNMatrix
  input (stored column-wise, and that elements within each column
  are contiguous).  This makes it possible to get the address
  of a column of JJ via the function SUNBandMatrix_Column(). The
  columns of the Jacobian are constructed using mupper + mlower + 1
  calls to the res routine, and appropriate differencing.
  The return value is either IDABAND_SUCCESS = 0, or the nonzero
  value returned by the res routine, if any.
  ---------------------------------------------------------------*/
#[allow(clippy::too_many_arguments)]
pub fn idaLsBandDQJac(
    tt: sunrealtype,
    c_j: sunrealtype,
    yy: &N_Vector,
    yp: &N_Vector,
    rr: &N_Vector,
    Jac: &SUNMatrix,
    IDA_mem: &IDAMem,
    tmp1: &N_Vector,
    tmp2: &N_Vector,
    tmp3: &N_Vector,
) -> i32 {
    let mut retval: i32 = 0;

    /* access matrix dimensions */
    let N = SUNBandMatrix_Columns(Jac);
    let mupper = SUNBandMatrix_UpperBandwidth(Jac);
    let mlower = SUNBandMatrix_LowerBandwidth(Jac);
    let s_mu = SUNBandMatrix_StoredUpperBandwidth(Jac); /* SM_COLUMN_ELEMENT_B offset */

    /* Rename work vectors for use as temporary values of r, y and yp */
    let rtemp = tmp1;
    let ytemp = tmp2;
    let yptemp = tmp3;

    /* Obtain integrator state (C caches raw data pointers for all eight
    vectors before the loop; data borrows here are re-taken per phase and
    never held across the residual callback or a vector op. The
    `hh`/`res` reads are hoisted — accepted deviation class 7). */
    let (uround, hh, ewt, constraints, res) = {
        let m = IDA_mem.borrow();
        (
            m.ida_uround,
            m.ida_hh,
            m.ida_ewt.as_ref().expect("ida_ewt").clone(),
            m.ida_constraints.clone(),
            m.ida_res.expect("ida_res"),
        )
    };

    /* Initialize ytemp and yptemp. */
    N_VScale(ONE, yy, ytemp);
    N_VScale(ONE, yp, yptemp);

    /* Compute miscellaneous values for the Jacobian computation. */
    let srur = SUNRsqrt(uround);
    let width = mlower + mupper + 1;
    let ngroups = SUNMIN(width, N);

    /* Loop over column groups. */
    let mut group: sunindextype = 1;
    while group <= ngroups {
        /* Increment all yy[j] and yp[j] for j in this group. */
        {
            let y_data = N_VGetArrayPointer(yy).expect("yy data");
            let yp_data = N_VGetArrayPointer(yp).expect("yp data");
            let ewt_data = N_VGetArrayPointer(&ewt).expect("ewt data");
            let mut ytemp_data = N_VGetArrayPointer(ytemp).expect("ytemp data");
            let mut yptemp_data = N_VGetArrayPointer(yptemp).expect("yptemp data");
            let cns_guard = constraints
                .as_ref()
                .map(|c| N_VGetArrayPointer(c).expect("constraints data"));
            let mut j = group - 1;
            while j < N {
                let yj = y_data[j as usize];
                let ypj = yp_data[j as usize];
                let ewtj = ewt_data[j as usize];

                /* Set increment inc to yj based on sqrt(uround)*abs(yj), with
                adjustments using ypj and ewtj if this is small, and a further
                adjustment to give it the same sign as hh*ypj. */
                let mut inc = SUNMAX(srur * SUNMAX(SUNRabs(yj), SUNRabs(hh * ypj)), ONE / ewtj);
                if hh * ypj < ZERO {
                    inc = -inc;
                }
                inc = (yj + inc) - yj;

                /* Adjust sign(inc) again if yj has an inequality constraint. */
                if let Some(cns_data) = &cns_guard {
                    let conj = cns_data[j as usize];
                    if SUNRabs(conj) == ONE {
                        if (yj + inc) * conj < ZERO {
                            inc = -inc;
                        }
                    } else if SUNRabs(conj) == TWO && (yj + inc) * conj <= ZERO {
                        inc = -inc;
                    }
                }

                /* Increment yj and ypj. */
                ytemp_data[j as usize] += inc;
                yptemp_data[j as usize] += c_j * inc;

                j += width;
            }
        }

        /* Call res routine with incremented arguments. */
        let mut user_data = IDA_mem.borrow_mut().ida_user_data.take();
        retval = res(tt, ytemp, yptemp, rtemp, &mut user_data);
        IDA_mem.borrow_mut().ida_user_data = user_data;
        idals_mem_mut(IDA_mem).nreDQ += 1;
        if retval != 0 {
            break;
        }

        /* Loop over the indices j in this group again. */
        {
            let y_data = N_VGetArrayPointer(yy).expect("yy data");
            let yp_data = N_VGetArrayPointer(yp).expect("yp data");
            let ewt_data = N_VGetArrayPointer(&ewt).expect("ewt data");
            let mut ytemp_data = N_VGetArrayPointer(ytemp).expect("ytemp data");
            let mut yptemp_data = N_VGetArrayPointer(yptemp).expect("yptemp data");
            let rtemp_data = N_VGetArrayPointer(rtemp).expect("rtemp data");
            let r_data = N_VGetArrayPointer(rr).expect("rr data");
            let cns_guard = constraints
                .as_ref()
                .map(|c| N_VGetArrayPointer(c).expect("constraints data"));
            let mut j = group - 1;
            while j < N {
                /* Reset ytemp and yptemp components that were perturbed. */
                ytemp_data[j as usize] = y_data[j as usize];
                let yj = ytemp_data[j as usize];
                yptemp_data[j as usize] = yp_data[j as usize];
                let ypj = yptemp_data[j as usize];
                let mut col_j = SUNBandMatrix_Column(Jac, j);
                let ewtj = ewt_data[j as usize];

                /* Set increment inc exactly as above. */
                let mut inc = SUNMAX(srur * SUNMAX(SUNRabs(yj), SUNRabs(hh * ypj)), ONE / ewtj);
                if hh * ypj < ZERO {
                    inc = -inc;
                }
                inc = (yj + inc) - yj;
                if let Some(cns_data) = &cns_guard {
                    let conj = cns_data[j as usize];
                    if SUNRabs(conj) == ONE {
                        if (yj + inc) * conj < ZERO {
                            inc = -inc;
                        }
                    } else if SUNRabs(conj) == TWO && (yj + inc) * conj <= ZERO {
                        inc = -inc;
                    }
                }

                /* Load the difference quotient Jacobian elements for column j */
                let inc_inv = ONE / inc;
                let i1 = SUNMAX(0, j - mupper);
                let i2 = SUNMIN(j + mlower, N - 1);
                let mut i = i1;
                while i <= i2 {
                    /* C: SM_COLUMN_ELEMENT_B(col_j, i, j) = ... */
                    col_j[SM_COLUMN_ELEMENT_IDX(i, j, s_mu)] =
                        inc_inv * (rtemp_data[i as usize] - r_data[i as usize]);
                    i += 1;
                }

                j += width;
            }
        }

        group += 1;
    }

    retval
}

/*---------------------------------------------------------------
  idaLsDQJtimes

  This routine generates a difference quotient approximation to
  the matrix-vector product z = Jv, where J is the system
  Jacobian. The approximation is
       Jv = [F(t,y1,yp1) - F(t,y,yp)]/sigma,
  where
       y1 = y + sigma*v,  yp1 = yp + cj*sigma*v,
       sigma = sqrt(Neq)*dqincfac.
  The return value from the call to res is saved in order to set
  the return flag from idaLsSolve.
  ---------------------------------------------------------------*/
#[allow(clippy::too_many_arguments)]
pub fn idaLsDQJtimes(
    tt: sunrealtype,
    yy: &N_Vector,
    yp: &N_Vector,
    rr: &N_Vector,
    v: &N_Vector,
    Jv: &N_Vector,
    c_j: sunrealtype,
    ida_mem: &mut Option<Box<dyn Any>>,
    work1: &N_Vector,
    work2: &N_Vector,
) -> i32 {
    /* access IDALsMem structure */
    let IDA_mem = match idaLs_AccessLMemToken(ida_mem, "idaLsDQJtimes") {
        Ok(m) => m,
        Err(retval) => return retval,
    };

    let LS = idals_mem_mut(&IDA_mem).LS.clone();
    let LSID = SUNLinSolGetID(&LS);
    let mut sig;
    if LSID == SUNLINEARSOLVER_SPGMR || LSID == SUNLINEARSOLVER_SPFGMR {
        let ls = idals_mem_mut(&IDA_mem);
        sig = ls.nrmfac * ls.dqincfac;
    } else {
        let dqincfac = idals_mem_mut(&IDA_mem).dqincfac;
        let ewt = IDA_mem.borrow().ida_ewt.as_ref().expect("ida_ewt").clone();
        sig = dqincfac / N_VWrmsNorm(v, &ewt);
    }

    /* Rename work1 and work2 for readability */
    let y_tmp = work1;
    let yp_tmp = work2;

    /* C re-reads `jt_res` each iteration; hoisted here (accepted
    deviation class 7) */
    let jt_res = idals_mem_mut(&IDA_mem).jt_res.expect("jt_res");

    let mut retval: i32 = 0;
    let mut iter: i32 = 0;
    while iter < MAX_ITERS {
        /* Set y_tmp = yy + sig*v, yp_tmp = yp + cj*sig*v. */
        N_VLinearSum(sig, v, ONE, yy, y_tmp);
        N_VLinearSum(c_j * sig, v, ONE, yp, yp_tmp);

        /* Call res for Jv = F(t, y_tmp, yp_tmp), and return if it failed. */
        let mut user_data = IDA_mem.borrow_mut().ida_user_data.take();
        retval = jt_res(tt, y_tmp, yp_tmp, Jv, &mut user_data);
        IDA_mem.borrow_mut().ida_user_data = user_data;
        idals_mem_mut(&IDA_mem).nreDQ += 1;
        if retval == 0 {
            break;
        }
        if retval < 0 {
            return -1;
        }

        sig *= PT25;
        iter += 1;
    }

    if retval > 0 {
        return 1;
    }

    /* Set Jv to [Jv - rr]/sig and return. */
    let siginv = ONE / sig;
    N_VLinearSum(siginv, Jv, -siginv, rr, Jv);

    0
}

/*---------------------------------------------------------------
 idaLsInitialize

 This routine performs remaining initializations specific
 to the iterative linear solver interface (and solver itself)
---------------------------------------------------------------*/
pub fn idaLsInitialize(IDA_mem: &IDAMem) -> i32 {
    /* access IDALsMem structure */
    if IDA_mem.borrow().ida_lmem.is_none() {
        IDAProcessError(
            Some(IDA_mem),
            IDALS_LMEM_NULL,
            line!() as i32,
            "idaLsInitialize",
            file!(),
            MSG_LS_LMEM_NULL,
        );
        return IDALS_LMEM_NULL;
    }

    /* Test for valid combinations of matrix & Jacobian routines: */
    let J = idals_mem_mut(IDA_mem).J.clone();
    if J.is_none() {
        /* If SUNMatrix A is NULL: ensure 'jac' function pointer is NULL */
        let mut ls = idals_mem_mut(IDA_mem);
        ls.jacDQ = SUNFALSE;
        ls.jac = None;
        ls.J_data = None;
    } else if idals_mem_mut(IDA_mem).jacDQ {
        /* If J is non-NULL, and 'jac' is not user-supplied:
        - if J is dense or band, ensure that our DQ approx. is used
        - otherwise => error */
        let J = J.as_ref().expect("J");
        let mut retval = 0;
        if J.ops.borrow().getid.is_some() {
            let id = SUNMatGetID(J);
            if (id == SUNMATRIX_DENSE) || (id == SUNMATRIX_BAND) {
                let mut ls = idals_mem_mut(IDA_mem);
                ls.jac = Some(idaLsDQJac);
                ls.J_data = Some(Box::new(IDA_mem.clone())); /* C: J_data = IDA_mem */
            } else {
                retval += 1;
            }
        } else {
            retval += 1;
        }
        if retval != 0 {
            IDAProcessError(
                Some(IDA_mem),
                IDALS_ILL_INPUT,
                line!() as i32,
                "idaLsInitialize",
                file!(),
                "No Jacobian constructor available for SUNMatrix type",
            );
            idals_mem_mut(IDA_mem).last_flag = IDALS_ILL_INPUT;
            return IDALS_ILL_INPUT;
        }
    } else {
        /* If J is non-NULL, and 'jac' is user-supplied,
        reset J_data pointer (just in case) */
        idals_mem_mut(IDA_mem).J_data = None; /* C: J_data = IDA_mem->ida_user_data */
    }

    /* reset counters */
    let _ = idaLsInitializeCounters(&mut idals_mem_mut(IDA_mem));

    /* Set Jacobian-related fields, based on jtimesDQ */
    if idals_mem_mut(IDA_mem).jtimesDQ {
        let mut ls = idals_mem_mut(IDA_mem);
        ls.jtsetup = None;
        ls.jtimes = Some(idaLsDQJtimes);
        ls.jt_data = Some(Box::new(IDA_mem.clone())); /* C: jt_data = IDA_mem */
    } else {
        idals_mem_mut(IDA_mem).jt_data = None; /* C: jt_data = IDA_mem->ida_user_data */
    }

    /* if J is NULL and psetup is not present, then idaLsSetup does
    not need to be called, so set the lsetup function to NULL */
    let (J_is_none, pset_is_none) = {
        let ls = idals_mem_mut(IDA_mem);
        (ls.J.is_none(), ls.pset.is_none())
    };
    if J_is_none && pset_is_none {
        IDA_mem.borrow_mut().ida_lsetup = None;
    }

    /* When using a matrix-embedded linear solver disable lsetup call */
    let LS = idals_mem_mut(IDA_mem).LS.clone();
    if SUNLinSolGetType(&LS) == SUNLINEARSOLVER_MATRIX_EMBEDDED {
        IDA_mem.borrow_mut().ida_lsetup = None;
        idals_mem_mut(IDA_mem).scalesol = SUNFALSE;
    }

    /* Call LS initialize routine */
    let last_flag = SUNLinSolInitialize(&LS);
    idals_mem_mut(IDA_mem).last_flag = last_flag;
    last_flag
}

/*---------------------------------------------------------------
 idaLsSetup

 This calls the Jacobian evaluation routine (if using a SUNMatrix
 object), updates counters, and calls the LS 'setup' routine to
 prepare for subsequent calls to the LS 'solve' routine.
---------------------------------------------------------------*/
pub fn idaLsSetup(
    IDA_mem: &IDAMem,
    y: &N_Vector,
    yp: &N_Vector,
    r: &N_Vector,
    vt1: &N_Vector,
    vt2: &N_Vector,
    vt3: &N_Vector,
) -> i32 {
    /* access IDALsMem structure */
    if IDA_mem.borrow().ida_lmem.is_none() {
        IDAProcessError(
            Some(IDA_mem),
            IDALS_LMEM_NULL,
            line!() as i32,
            "idaLsSetup",
            file!(),
            MSG_LS_LMEM_NULL,
        );
        return IDALS_LMEM_NULL;
    }

    /* Immediately return when using matrix-embedded linear solver */
    let LS = idals_mem_mut(IDA_mem).LS.clone();
    if SUNLinSolGetType(&LS) == SUNLINEARSOLVER_MATRIX_EMBEDDED {
        idals_mem_mut(IDA_mem).last_flag = IDALS_SUCCESS;
        return IDALS_SUCCESS;
    }

    /* Set IDALs N_Vector pointers to inputs */
    {
        let mut ls = idals_mem_mut(IDA_mem);
        ls.ycur = Some(y.clone());
        ls.ypcur = Some(yp.clone());
        ls.rcur = Some(r.clone());
    }

    /* Update values for last jac/pset call */
    let (nst, tn, cj) = {
        let m = IDA_mem.borrow();
        (m.ida_nst, m.ida_tn, m.ida_cj)
    };
    {
        let mut ls = idals_mem_mut(IDA_mem);
        ls.nstlj = nst;
        ls.tnlj = tn;
    }

    /* recompute if J if it is non-NULL */
    let J = idals_mem_mut(IDA_mem).J.clone();
    if let Some(J) = &J {
        /* Increment nje counter. */
        idals_mem_mut(IDA_mem).nje += 1;

        /* Clear the linear system matrix if necessary */
        if SUNLinSolGetType(&LS) == SUNLINEARSOLVER_DIRECT {
            let retval = SUNMatZero(J);
            if retval != 0 {
                IDAProcessError(
                    Some(IDA_mem),
                    IDALS_SUNMAT_FAIL,
                    line!() as i32,
                    "idaLsSetup",
                    file!(),
                    MSG_LS_MATZERO_FAILED,
                );
                idals_mem_mut(IDA_mem).last_flag = IDALS_SUNMAT_FAIL;
                return IDALS_SUNMAT_FAIL;
            }
        }

        /* Call Jacobian routine */
        let jac = idals_mem_mut(IDA_mem).jac.expect("jac");
        let use_field = idals_mem_mut(IDA_mem).J_data.is_some();
        let mut J_data = if use_field {
            idals_mem_mut(IDA_mem).J_data.take()
        } else {
            IDA_mem.borrow_mut().ida_user_data.take()
        };
        let retval = jac(tn, cj, y, yp, r, J, &mut J_data, vt1, vt2, vt3);
        if use_field {
            idals_mem_mut(IDA_mem).J_data = J_data;
        } else {
            IDA_mem.borrow_mut().ida_user_data = J_data;
        }
        if retval < 0 {
            IDAProcessError(
                Some(IDA_mem),
                IDALS_JACFUNC_UNRECVR,
                line!() as i32,
                "idaLsSetup",
                file!(),
                MSG_LS_JACFUNC_FAILED,
            );
            idals_mem_mut(IDA_mem).last_flag = IDALS_JACFUNC_UNRECVR;
            return -1;
        }
        if retval > 0 {
            idals_mem_mut(IDA_mem).last_flag = IDALS_JACFUNC_RECVR;
            return 1;
        }
    }

    /* Call LS setup routine -- the LS will call idaLsPSetup if applicable */
    let last_flag = SUNLinSolSetup(&LS, J.as_ref());
    idals_mem_mut(IDA_mem).last_flag = last_flag;
    last_flag
}

/*---------------------------------------------------------------
 idaLsSolve

 This routine interfaces between IDA and the generic
 SUNLinearSolver object LS, by setting the appropriate tolerance
 and scaling vectors, calling the solver, accumulating
 statistics from the solve for use/reporting by IDA, and scaling
 the result if using a non-NULL SUNMatrix and cjratio does not
 equal one.
---------------------------------------------------------------*/
pub fn idaLsSolve(
    IDA_mem: &IDAMem,
    b: &N_Vector,
    weight: &N_Vector,
    ycur: &N_Vector,
    ypcur: &N_Vector,
    rescur: &N_Vector,
) -> i32 {
    /* access IDALsMem structure */
    if IDA_mem.borrow().ida_lmem.is_none() {
        IDAProcessError(
            Some(IDA_mem),
            IDALS_LMEM_NULL,
            line!() as i32,
            "idaLsSolve",
            file!(),
            MSG_LS_LMEM_NULL,
        );
        return IDALS_LMEM_NULL;
    }

    /* If the linear solver is iterative: set convergence test constant tol,
    in terms of the Newton convergence test constant epsNewt and safety
    factors. The factor nrmfac assures that the convergence test is
    applied to the WRMS norm of the residual vector, rather than the
    weighted L2 norm.
    (The C SUNLogInfo "begin-linear-solve" calls compile away at
    SUNDIALS_LOGGING_LEVEL 2.) */
    let (iterative, nrmfac, eplifac) = {
        let ls = idals_mem_mut(IDA_mem);
        (ls.iterative, ls.nrmfac, ls.eplifac)
    };
    let epsNewt = IDA_mem.borrow().ida_epsNewt;
    let mut tol: sunrealtype = if iterative {
        nrmfac * eplifac * epsNewt
    } else {
        ZERO
    };

    /* Set vectors ycur, ypcur and rcur for use by the Atimes and
    Psolve interface routines */
    {
        let mut ls = idals_mem_mut(IDA_mem);
        ls.ycur = Some(ycur.clone());
        ls.ypcur = Some(ypcur.clone());
        ls.rcur = Some(rescur.clone());
    }

    let LS = idals_mem_mut(IDA_mem).LS.clone();
    let x = idals_mem_mut(IDA_mem).x.as_ref().expect("x").clone();

    /* Set scaling vectors for LS to use (if applicable) */
    if LS.ops.borrow().setscalingvectors.is_some() {
        let retval = SUNLinSolSetScalingVectors(&LS, Some(weight), Some(weight));
        if retval != SUN_SUCCESS {
            IDAProcessError(
                Some(IDA_mem),
                IDALS_SUNLS_FAIL,
                line!() as i32,
                "idaLsSolve",
                file!(),
                "Error in calling SUNLinSolSetScalingVectors",
            );
            idals_mem_mut(IDA_mem).last_flag = IDALS_SUNLS_FAIL;
            return IDALS_SUNLS_FAIL;
        }

        /* If solver is iterative and does not support scaling vectors, update the
        tolerance in an attempt to account for weight vector.  We make the
        following assumptions:
          1. w_i = w_mean, for i=0,...,n-1 (i.e. the weights are homogeneous)
          2. the linear solver uses a basic 2-norm to measure convergence
        Hence (using the notation from sunlinsol_spgmr.h, with S = diag(w)),
              || bbar - Abar xbar ||_2 < tol
          <=> || S b - S A x ||_2 < tol
          <=> || S (b - A x) ||_2 < tol
          <=> \sum_{i=0}^{n-1} (w_i (b - A x)_i)^2 < tol^2
          <=> w_mean^2 \sum_{i=0}^{n-1} (b - A x_i)^2 < tol^2
          <=> \sum_{i=0}^{n-1} (b - A x_i)^2 < tol^2 / w_mean^2
          <=> || b - A x ||_2 < tol / w_mean
        So we compute w_mean = ||w||_RMS and scale the desired tolerance accordingly. */
    } else if iterative {
        N_VConst(ONE, &x);
        let w_mean = N_VWrmsNorm(weight, &x);
        tol /= w_mean;
    }

    /* Set initial guess x = 0 to LS */
    N_VConst(ZERO, &x);

    /* Set zero initial guess flag */
    let retval = SUNLinSolSetZeroGuess(&LS, SUNTRUE);
    if retval != SUN_SUCCESS {
        return -1;
    }

    /* C stores the previous nps value in nps_inc here (logging only —
    omitted at SUNDIALS_LOGGING_LEVEL 2) */

    /* If a user-provided jtsetup routine is supplied, call that here */
    let jtsetup = idals_mem_mut(IDA_mem).jtsetup;
    if let Some(jtsetup) = jtsetup {
        let (tn, cj) = {
            let m = IDA_mem.borrow();
            (m.ida_tn, m.ida_cj)
        };
        let use_field = idals_mem_mut(IDA_mem).jt_data.is_some();
        let mut jt_data = if use_field {
            idals_mem_mut(IDA_mem).jt_data.take()
        } else {
            IDA_mem.borrow_mut().ida_user_data.take()
        };
        let last_flag = jtsetup(tn, ycur, ypcur, rescur, cj, &mut jt_data);
        if use_field {
            idals_mem_mut(IDA_mem).jt_data = jt_data;
        } else {
            IDA_mem.borrow_mut().ida_user_data = jt_data;
        }
        {
            let mut ls = idals_mem_mut(IDA_mem);
            ls.last_flag = last_flag;
            ls.njtsetup += 1;
        }
        if last_flag != 0 {
            /* C passes `retval` (the SetZeroGuess result, SUN_SUCCESS
            here) as the error code — preserved verbatim */
            IDAProcessError(
                Some(IDA_mem),
                retval,
                line!() as i32,
                "idaLsSolve",
                file!(),
                MSG_LS_JTSETUP_FAILED,
            );
            return last_flag;
        }
    }

    /* Call solver */
    let J = idals_mem_mut(IDA_mem).J.clone();
    let retval = SUNLinSolSolve(&LS, J.as_ref(), &x, b, tol);

    /* Copy appropriate result to b (depending on solver type) */
    if iterative {
        /* Retrieve solver statistics */
        let nli_inc = SUNLinSolNumIters(&LS);
        let _ = SUNLinSolResNorm(&LS); /* resnorm: logging only at level 2 */

        /* Copy x (or preconditioned residual vector if no iterations required) to b */
        if (nli_inc == 0) && (SUNLinSolGetType(&LS) != SUNLINEARSOLVER_MATRIX_EMBEDDED) {
            let resid = SUNLinSolResid(&LS).expect("resid");
            N_VScale(ONE, &resid, b);
        } else {
            N_VScale(ONE, &x, b);
        }

        /* Increment nli counter */
        idals_mem_mut(IDA_mem).nli += nli_inc as i64;
    } else {
        /* Copy x to b */
        N_VScale(ONE, &x, b);
    }

    /* If using a direct or matrix-iterative solver, scale the correction to
    account for change in cj */
    let scalesol = idals_mem_mut(IDA_mem).scalesol;
    let cjratio = IDA_mem.borrow().ida_cjratio;
    if scalesol && (cjratio != ONE) {
        N_VScale(TWO / (ONE + cjratio), b, b);
    }

    /* Increment ncfl counter */
    if retval != SUN_SUCCESS {
        idals_mem_mut(IDA_mem).ncfl += 1;
    }

    /* Interpret solver return value  */
    idals_mem_mut(IDA_mem).last_flag = retval;

    match retval {
        SUN_SUCCESS => 0,
        SUNLS_RES_REDUCED | SUNLS_CONV_FAIL | SUNLS_PSOLVE_FAIL_REC | SUNLS_PACKAGE_FAIL_REC
        | SUNLS_QRFACT_FAIL | SUNLS_LUFACT_FAIL => 1,
        SUN_ERR_ARG_CORRUPT | SUN_ERR_ARG_INCOMPATIBLE | SUN_ERR_MEM_FAIL | SUNLS_GS_FAIL
        | SUNLS_QRSOL_FAIL => -1,
        SUN_ERR_EXT_FAIL => {
            IDAProcessError(
                Some(IDA_mem),
                SUN_ERR_EXT_FAIL,
                line!() as i32,
                "idaLsSolve",
                file!(),
                "Failure in SUNLinSol external package",
            );
            -1
        }
        SUNLS_PSOLVE_FAIL_UNREC => {
            IDAProcessError(
                Some(IDA_mem),
                SUNLS_PSOLVE_FAIL_UNREC,
                line!() as i32,
                "idaLsSolve",
                file!(),
                MSG_LS_PSOLVE_FAILED,
            );
            -1
        }
        _ => {
            IDAProcessError(
                Some(IDA_mem),
                retval,
                line!() as i32,
                "idaLsSolve",
                file!(),
                "Unrecognized error return value from SUNLinSolSolve",
            );
            -1
        }
    }
}

/*---------------------------------------------------------------
 idaLsPerf: accumulates performance statistics information
 for IDA
---------------------------------------------------------------*/
pub fn idaLsPerf(IDA_mem: &IDAMem, perftask: i32) -> i32 {
    /* access IDALsMem structure */
    if IDA_mem.borrow().ida_lmem.is_none() {
        IDAProcessError(
            Some(IDA_mem),
            IDALS_LMEM_NULL,
            line!() as i32,
            "idaLsPerf",
            file!(),
            MSG_LS_LMEM_NULL,
        );
        return IDALS_LMEM_NULL;
    }

    /* when perftask == 0, store current performance statistics */
    if perftask == 0 {
        let (nst, nni, ncfn) = {
            let m = IDA_mem.borrow();
            (m.ida_nst, m.ida_nni, m.ida_ncfn)
        };
        let mut ls = idals_mem_mut(IDA_mem);
        ls.nst0 = nst;
        ls.nni0 = nni;
        ls.ncfn0 = ncfn;
        ls.ncfl0 = ls.ncfl;
        ls.nwarn = 0;
        return 0;
    }

    /* Compute statistics since last call

    Note: the performance monitor that checked whether the average
      number of linear iterations was too close to maxl has been
      removed, since the 'maxl' value is no longer owned by the
      IDALs interface.
    */
    let (nst, nni, ncfn, tn) = {
        let m = IDA_mem.borrow();
        (m.ida_nst, m.ida_nni, m.ida_ncfn, m.ida_tn)
    };
    let (nst0, nni0, ncfn0, ncfl, ncfl0) = {
        let ls = idals_mem_mut(IDA_mem);
        (ls.nst0, ls.nni0, ls.ncfn0, ls.ncfl, ls.ncfl0)
    };

    let nstd = nst - nst0;
    let nnid = nni - nni0;
    if nstd == 0 || nnid == 0 {
        return 0;
    }

    let rcfn = ((ncfn - ncfn0) as sunrealtype) / (nstd as sunrealtype);
    let rcfl = ((ncfl - ncfl0) as sunrealtype) / (nnid as sunrealtype);
    let lcfn: sunbooleantype = rcfn > PT9;
    let lcfl: sunbooleantype = rcfl > PT9;
    if !(lcfn || lcfl) {
        return 0;
    }
    let nwarn = {
        let mut ls = idals_mem_mut(IDA_mem);
        ls.nwarn += 1;
        ls.nwarn
    };
    if nwarn > 10 {
        return 1;
    }
    if lcfn {
        IDAProcessError(
            Some(IDA_mem),
            IDA_WARNING,
            line!() as i32,
            "idaLsPerf",
            file!(),
            &MSG_LS_CFN_WARN(tn, rcfn),
        );
    }
    if lcfl {
        IDAProcessError(
            Some(IDA_mem),
            IDA_WARNING,
            line!() as i32,
            "idaLsPerf",
            file!(),
            &MSG_LS_CFL_WARN(tn, rcfl),
        );
    }
    0
}

/*---------------------------------------------------------------
 idaLsFree frees memory associates with the IDALs system
 solver interface.
---------------------------------------------------------------*/
pub fn idaLsFree(IDA_mem: &IDAMem) -> i32 {
    /* NULL IDAMem check: handled by type system */

    /* Return immediately if IDA_mem->ida_lmem is NULL */
    if IDA_mem.borrow().ida_lmem.is_none() {
        return IDALS_SUCCESS;
    }

    {
        let mut ls = idals_mem_mut(IDA_mem);

        /* Free N_Vector memory */
        if let Some(ytemp) = ls.ytemp.take() {
            N_VDestroy(ytemp);
        }
        if let Some(yptemp) = ls.yptemp.take() {
            N_VDestroy(yptemp);
        }
        if let Some(x) = ls.x.take() {
            N_VDestroy(x);
        }

        /* Nullify other N_Vector pointers */
        ls.ycur = None;
        ls.ypcur = None;
        ls.rcur = None;

        /* Nullify SUNMatrix pointer */
        ls.J = None;
    }

    /* Free preconditioner memory (if applicable) */
    let pfree = idals_mem_mut(IDA_mem).pfree;
    if let Some(pfree) = pfree {
        pfree(IDA_mem);
    }

    /* free IDALs interface structure (C `free()` leaves ida_lmem
    dangling; clearing it is the safe equivalent) */
    IDA_mem.borrow_mut().ida_lmem = None;

    IDALS_SUCCESS
}

/*---------------------------------------------------------------
 idaLsInitializeCounters resets all counters from an
 IDALsMem structure.
---------------------------------------------------------------*/
pub fn idaLsInitializeCounters(idals_mem: &mut IDALsMemRec) -> i32 {
    idals_mem.nje = 0;
    idals_mem.nreDQ = 0;
    idals_mem.npe = 0;
    idals_mem.nli = 0;
    idals_mem.nps = 0;
    idals_mem.ncfl = 0;
    idals_mem.njtsetup = 0;
    idals_mem.njtimes = 0;
    0
}

/*---------------------------------------------------------------
  idaLs_AccessLMem

  Public-API flavor of the C helper: with `&IDAMem` the NULL-mem
  check vanishes (handled by the type system); this verifies that
  linear solver memory is attached. Callers then use
  `idals_mem_mut` for field access.
  ---------------------------------------------------------------*/
pub fn idaLs_AccessLMem(ida_mem: &IDAMem, fname: &str) -> i32 {
    /* NULL-mem check: handled by type system */
    if ida_mem.borrow().ida_lmem.is_none() {
        IDAProcessError(
            Some(ida_mem),
            IDALS_LMEM_NULL,
            line!() as i32,
            fname,
            file!(),
            MSG_LS_LMEM_NULL,
        );
        return IDALS_LMEM_NULL;
    }
    IDALS_SUCCESS
}

/*---------------------------------------------------------------
  idaLs_AccessLMemToken

  Callback flavor of the C `idaLs_AccessLMem`: the C `void* ida_mem`
  argument arrives as a data token holding an `IDAMem` clone. A
  missing/foreign token maps to the C NULL check.
  ---------------------------------------------------------------*/
pub fn idaLs_AccessLMemToken(
    ida_mem: &Option<Box<dyn Any>>,
    fname: &str,
) -> Result<IDAMem, i32> {
    let IDA_mem = match ida_mem
        .as_ref()
        .and_then(|b| b.downcast_ref::<IDAMem>())
        .cloned()
    {
        Some(m) => m,
        None => {
            IDAProcessError(
                None,
                IDALS_MEM_NULL,
                line!() as i32,
                fname,
                file!(),
                MSG_LS_IDAMEM_NULL,
            );
            return Err(IDALS_MEM_NULL);
        }
    };
    if IDA_mem.borrow().ida_lmem.is_none() {
        IDAProcessError(
            Some(&IDA_mem),
            IDALS_LMEM_NULL,
            line!() as i32,
            fname,
            file!(),
            MSG_LS_LMEM_NULL,
        );
        return Err(IDALS_LMEM_NULL);
    }
    Ok(IDA_mem)
}

/*---------------------------------------------------------------
  EOF
  ---------------------------------------------------------------*/
