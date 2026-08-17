//! Port of `src/cvode/cvode_ls.c` (+ `src/cvode/cvode_ls_impl.h` and
//! `include/cvode/cvode_ls.h` folded).
//!
//! CVODE's linear solver interface (CVLS): attaches a generic
//! `SUNLinearSolver` (and optional `SUNMatrix`) to CVODE, provides the
//! `cv_linit`/`cv_lreinit`/`cv_lsetup`/`cv_lsolve`/`cv_lfree` integrator
//! hooks, the difference-quotient dense/band Jacobians and J*v product,
//! and the ATimes/PSetup/PSolve trampolines registered with the LS.
//!
//! Data-token model (C `void*` fields `J_data`/`P_data`/`jt_data`/
//! `A_data`): in C each field holds either `cv_mem` (internal routine)
//! or `cv_mem->cv_user_data` (user routine). Here the field is
//! `Option<Box<dyn Any>>`: `Some(box)` is a module-owned token (a
//! `CVodeMem` clone for the internal CVLS routines, or whatever an
//! internal preconditioner module stored), while `None` means "pass the
//! integrator's `cv_user_data`" — the invoker `Option::take`s the
//! corresponding box around the callback and restores it on every path.
//! This reproduces the C pointer aliasing without double ownership; the
//! only divergence is that a C snapshot of a *stale* `cv_user_data`
//! (user data replaced after the Set* call) cannot occur — the current
//! `cv_user_data` is always passed. For `J_data`/`jt_data`/`A_data`
//! that matches C exactly (`cvLsInitialize`'s "reset just in case"
//! assignments refresh them); for `P_data` C keeps the attach-time
//! snapshot forever, so a `CVodeSetUserData` call AFTER
//! `CVodeSetLinearSolver` diverges: C's pset/psolve keep seeing the
//! old pointer, this port passes the new box (accepted deviation
//! class 6, see ARCHITECTURE.md).
//!
//! Granular borrow discipline: no `cv_mem` borrow is held across a user
//! callback, an N_Vector op on user-visible vectors, or a
//! SUNLinearSolver/SUNMatrix/SUNNonlinearSolver call.

use std::any::Any;
use std::cell::RefMut;

use crate::cvode_impl::*;
use sundials_core::sundials_errors::{
    SUN_ERR_ARG_CORRUPT, SUN_ERR_ARG_INCOMPATIBLE, SUN_ERR_EXT_FAIL, SUN_ERR_MEM_FAIL, SUN_SUCCESS,
};
use sundials_core::sundials_linearsolver::*;
use sundials_core::sundials_math::{SUNMAX, SUNMIN, SUNRabs, SUNRsqrt};
use sundials_core::sundials_matrix::*;
use sundials_core::sundials_nonlinearsolver::SUNNonlinSolGetCurIter;
use sundials_core::sundials_nvector::*;
use sundials_core::sundials_types::*;
use sundials_core::sunmatrix_band::{
    SM_COLUMN_ELEMENT_IDX, SUNBandMatrix_Column, SUNBandMatrix_Columns,
    SUNBandMatrix_LowerBandwidth, SUNBandMatrix_StoredUpperBandwidth,
    SUNBandMatrix_UpperBandwidth,
};
use sundials_core::sunmatrix_dense::{SUNDenseMatrix_Column, SUNDenseMatrix_Columns};

/* Private constants */
const MIN_INC_MULT: sunrealtype = 1000.0;
const MAX_DQITERS: i32 = 3; /* max. number of attempts to recover in DQ J*v */
const ZERO: sunrealtype = 0.0;
const PT25: sunrealtype = 0.25;
const ONE: sunrealtype = 1.0;
const TWO: sunrealtype = 2.0;

/*=================================================================
  CVLS Constants (include/cvode/cvode_ls.h)
  =================================================================*/

pub const CVLS_SUCCESS: i32 = 0;
pub const CVLS_MEM_NULL: i32 = -1;
pub const CVLS_LMEM_NULL: i32 = -2;
pub const CVLS_ILL_INPUT: i32 = -3;
pub const CVLS_MEM_FAIL: i32 = -4;
pub const CVLS_PMEM_NULL: i32 = -5;
pub const CVLS_JACFUNC_UNRECVR: i32 = -6;
pub const CVLS_JACFUNC_RECVR: i32 = -7;
pub const CVLS_SUNMAT_FAIL: i32 = -8;
pub const CVLS_SUNLS_FAIL: i32 = -9;

/*-----------------------------------------------------------------
  CVLS solver constants (cvode_ls_impl.h)
  -----------------------------------------------------------------*/

pub const CVLS_MSBJ: i64 = 51;
pub const CVLS_DGMAX: sunrealtype = 0.2;
pub const CVLS_EPLIN: sunrealtype = 0.05;

/*-----------------------------------------------------------------
  Error Messages (cvode_ls_impl.h)
  -----------------------------------------------------------------*/

pub const MSG_LS_CVMEM_NULL: &str = "Integrator memory is NULL.";
pub const MSG_LS_MEM_FAIL: &str = "A memory request failed.";
pub const MSG_LS_BAD_NVECTOR: &str = "A required vector operation is not implemented.";
pub const MSG_LS_BAD_LSTYPE: &str = "Incompatible linear solver type.";
pub const MSG_LS_LMEM_NULL: &str = "Linear solver memory is NULL.";
pub const MSG_LS_BAD_SIZES: &str =
    "Illegal bandwidth parameter(s). Must have 0 <=  ml, mu <= N-1.";
pub const MSG_LS_BAD_EPLIN: &str = "eplifac < 0 illegal.";

pub const MSG_LS_PSET_FAILED: &str =
    "The preconditioner setup routine failed in an unrecoverable manner.";
pub const MSG_LS_PSOLVE_FAILED: &str =
    "The preconditioner solve routine failed in an unrecoverable manner.";
pub const MSG_LS_JTSETUP_FAILED: &str =
    "The Jacobian x vector setup routine failed in an unrecoverable manner.";
pub const MSG_LS_JTIMES_FAILED: &str =
    "The Jacobian x vector routine failed in an unrecoverable manner.";
pub const MSG_LS_JACFUNC_FAILED: &str = "The Jacobian routine failed in an unrecoverable manner.";
pub const MSG_LS_SUNMAT_FAILED: &str = "A SUNMatrix routine failed in an unrecoverable manner.";

/*=================================================================
  CVLS user-supplied function prototypes (include/cvode/cvode_ls.h)
  =================================================================*/

pub type CVLsJacFn = fn(
    t: sunrealtype,
    y: &N_Vector,
    fy: &N_Vector,
    Jac: &SUNMatrix,
    user_data: &mut Option<Box<dyn Any>>,
    tmp1: &N_Vector,
    tmp2: &N_Vector,
    tmp3: &N_Vector,
) -> i32;

pub type CVLsPrecSetupFn = fn(
    t: sunrealtype,
    y: &N_Vector,
    fy: &N_Vector,
    jok: sunbooleantype,
    jcurPtr: &mut sunbooleantype,
    gamma: sunrealtype,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32;

pub type CVLsPrecSolveFn = fn(
    t: sunrealtype,
    y: &N_Vector,
    fy: &N_Vector,
    r: &N_Vector,
    z: &N_Vector,
    gamma: sunrealtype,
    delta: sunrealtype,
    lr: i32,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32;

pub type CVLsJacTimesSetupFn = fn(
    t: sunrealtype,
    y: &N_Vector,
    fy: &N_Vector,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32;

pub type CVLsJacTimesVecFn = fn(
    v: &N_Vector,
    Jv: &N_Vector,
    t: sunrealtype,
    y: &N_Vector,
    fy: &N_Vector,
    user_data: &mut Option<Box<dyn Any>>,
    tmp: &N_Vector,
) -> i32;

pub type CVLsLinSysFn = fn(
    t: sunrealtype,
    y: &N_Vector,
    fy: &N_Vector,
    A: &SUNMatrix,
    jok: sunbooleantype,
    jcur: &mut sunbooleantype,
    gamma: sunrealtype,
    user_data: &mut Option<Box<dyn Any>>,
    tmp1: &N_Vector,
    tmp2: &N_Vector,
    tmp3: &N_Vector,
) -> i32;

/*-----------------------------------------------------------------
  Types : CVLsMemRec, CVLsMem (cvode_ls_impl.h)
  -----------------------------------------------------------------*/

pub struct CVLsMemRec {
    /* Linear solver type information */
    pub iterative: sunbooleantype,   /* is the solver iterative?    */
    pub matrixbased: sunbooleantype, /* is a matrix structure used? */

    /* Jacobian construction & storage */
    pub jacDQ: sunbooleantype, /* SUNTRUE if using internal DQ Jac approx.     */
    pub jac: Option<CVLsJacFn>, /* Jacobian routine to be called               */
    /* C `J_data`: `Some` = module-owned token (CVodeMem clone for the
    internal DQ routine); `None` = pass `cv_user_data` at call time. */
    pub J_data: Option<Box<dyn Any>>,
    pub jbad: sunbooleantype,    /* heuristic suggestion for pset                */
    pub dgmax_jbad: sunrealtype, /* if convfail = FAIL_BAD_J and the gamma ratio *
                                  * |gamma/gammap-1| < dgmax_jbad then J is bad  */

    /* Matrix-based solver, scale solution to account for change in gamma */
    pub scalesol: sunbooleantype,

    /* Iterative solver tolerance */
    pub eplifac: sunrealtype, /* nonlinear -> linear tol scaling factor       */
    pub nrmfac: sunrealtype,  /* integrator -> LS norm conversion factor      */

    /* Linear solver, matrix and vector objects/pointers */
    pub LS: SUNLinearSolver,      /* generic linear solver object                 */
    pub A: Option<SUNMatrix>,     /* A = I - gamma * df/dy                        */
    pub savedJ: Option<SUNMatrix>, /* savedJ = old Jacobian                       */
    pub ytemp: Option<N_Vector>,  /* temp vector passed to jtimes and psolve      */
    pub x: Option<N_Vector>,      /* temp vector used by CVLsSolve                */
    pub ycur: Option<N_Vector>,   /* CVODE current y vector in Newton Iteration   */
    pub fcur: Option<N_Vector>,   /* fcur = f(tn, ycur)                           */

    /* Statistics and associated parameters */
    pub msbj: i64,     /* max num steps between jac/pset calls         */
    pub nje: i64,      /* nje = no. of calls to jac                    */
    pub nfeDQ: i64,    /* no. of calls to f due to DQ Jacobian or J*v
                       approximations                               */
    pub nstlj: i64,    /* nstlj = nst at last jac/pset call            */
    pub npe: i64,      /* npe = total number of pset calls             */
    pub nli: i64,      /* nli = total number of linear iterations      */
    pub nps: i64,      /* nps = total number of psolve calls           */
    pub ncfl: i64,     /* ncfl = total number of convergence failures  */
    pub njtsetup: i64, /* njtsetup = total number of calls to jtsetup  */
    pub njtimes: i64,  /* njtimes = total number of calls to jtimes    */
    pub tnlj: sunrealtype, /* tnlj = t_n at last jac/pset call         */

    /* Preconditioner computation
     * (a) user-provided:
     *     - P_data == user_data (here: `None` = pass cv_user_data)
     *     - pfree == NULL (the user deallocates memory for user_data)
     * (b) internal preconditioner module
     *     - P_data == module token (`Some`)
     *     - pfree == set by the prec. module and called in CVodeFree */
    pub pset: Option<CVLsPrecSetupFn>,
    pub psolve: Option<CVLsPrecSolveFn>,
    pub pfree: Option<fn(cv_mem: &CVodeMem) -> i32>,
    pub P_data: Option<Box<dyn Any>>,

    /* Jacobian times vector computation
     * (a) jtimes function provided by the user:
     *     - jt_data == user_data (here: `None`)
     *     - jtimesDQ == SUNFALSE
     * (b) internal jtimes
     *     - jt_data == cvode_mem token (`Some`)
     *     - jtimesDQ == SUNTRUE */
    pub jtimesDQ: sunbooleantype,
    pub jtsetup: Option<CVLsJacTimesSetupFn>,
    pub jtimes: Option<CVLsJacTimesVecFn>,
    pub jt_f: Option<CVRhsFn>,
    pub jt_data: Option<Box<dyn Any>>,

    /* Linear system setup function
     * (a) user-provided linsys function:
     *     - user_linsys = SUNTRUE
     *     - A_data      = user_data (here: `None`)
     * (b) internal linsys function:
     *     - user_linsys = SUNFALSE
     *     - A_data      = cvode_mem token (`Some`) */
    pub user_linsys: sunbooleantype,
    pub linsys: Option<CVLsLinSysFn>,
    pub A_data: Option<Box<dyn Any>>,

    pub last_flag: i32, /* last error flag returned by any function */
}

pub type CVLsMem = Box<CVLsMemRec>;

/// Downcast helper: view `cv_mem.cv_lmem` as the CVLS memory record.
/// Panics if no linear solver memory is attached or it is not a CVLS
/// record (the C code would blindly cast the `void*` — UB → panic).
/// NEVER hold the returned guard across a callback, an N_Vector op on a
/// user-visible vector, or a SUNLinearSolver/SUNMatrix call.
pub fn cvls_mem_mut(cv_mem: &CVodeMem) -> RefMut<'_, CVLsMemRec> {
    RefMut::map(cv_mem.borrow_mut(), |m| {
        m.cv_lmem
            .as_mut()
            .expect("cv_lmem set")
            .downcast_mut::<CVLsMemRec>()
            .expect("CVLS linear solver memory")
    })
}

/*===============================================================
  CVLS Exported functions -- Required
  ===============================================================*/

/*---------------------------------------------------------------
  CVodeSetLinearSolver specifies the linear solver
  ---------------------------------------------------------------*/
pub fn CVodeSetLinearSolver(
    cvode_mem: &CVodeMem,
    LS: &SUNLinearSolver,
    A: Option<&SUNMatrix>,
) -> i32 {
    /* NULL-mem check: handled by type system */
    /* NULL-LS check: handled by type system */

    /* Test if solver is compatible with LS interface */
    {
        let ops = LS.ops.borrow();
        if ops.gettype.is_none() || ops.solve.is_none() {
            cvProcessError(
                Some(cvode_mem),
                CVLS_ILL_INPUT,
                line!() as i32,
                "CVodeSetLinearSolver",
                file!(),
                "LS object is missing a required operation",
            );
            return CVLS_ILL_INPUT;
        }
    }

    /* Retrieve the LS type */
    let LSType = SUNLinSolGetType(LS);

    /* Set flags based on LS type */
    let iterative = LSType != SUNLINEARSOLVER_DIRECT;
    let matrixbased =
        (LSType != SUNLINEARSOLVER_ITERATIVE) && (LSType != SUNLINEARSOLVER_MATRIX_EMBEDDED);

    /* Test if vector is compatible with LS interface */
    let cv_tempv = cvode_mem
        .borrow()
        .cv_tempv
        .as_ref()
        .expect("cv_tempv") /* C dereferences unconditionally (UB if unset) */
        .clone();
    {
        let ops = cv_tempv.ops.borrow();
        if ops.nvconst.is_none() || ops.nvwrmsnorm.is_none() {
            cvProcessError(
                Some(cvode_mem),
                CVLS_ILL_INPUT,
                line!() as i32,
                "CVodeSetLinearSolver",
                file!(),
                MSG_LS_BAD_NVECTOR,
            );
            return CVLS_ILL_INPUT;
        }
    }

    /* Ensure that A is NULL when LS is matrix-embedded */
    if (LSType == SUNLINEARSOLVER_MATRIX_EMBEDDED) && A.is_some() {
        cvProcessError(
            Some(cvode_mem),
            CVLS_ILL_INPUT,
            line!() as i32,
            "CVodeSetLinearSolver",
            file!(),
            "Incompatible inputs: matrix-embedded LS requires NULL matrix",
        );
        return CVLS_ILL_INPUT;
    }

    /* Check for compatible LS type, matrix and "atimes" support */
    if iterative {
        if cv_tempv.ops.borrow().nvgetlength.is_none() {
            cvProcessError(
                Some(cvode_mem),
                CVLS_ILL_INPUT,
                line!() as i32,
                "CVodeSetLinearSolver",
                file!(),
                MSG_LS_BAD_NVECTOR,
            );
            return CVLS_ILL_INPUT;
        }

        if !matrixbased
            && (LSType != SUNLINEARSOLVER_MATRIX_EMBEDDED)
            && LS.ops.borrow().setatimes.is_none()
        {
            cvProcessError(
                Some(cvode_mem),
                CVLS_ILL_INPUT,
                line!() as i32,
                "CVodeSetLinearSolver",
                file!(),
                "Incompatible inputs: iterative LS must support ATimes routine",
            );
            return CVLS_ILL_INPUT;
        }

        if matrixbased && A.is_none() {
            cvProcessError(
                Some(cvode_mem),
                CVLS_ILL_INPUT,
                line!() as i32,
                "CVodeSetLinearSolver",
                file!(),
                "Incompatible inputs: matrix-iterative LS requires non-NULL matrix",
            );
            return CVLS_ILL_INPUT;
        }
    } else if A.is_none() {
        cvProcessError(
            Some(cvode_mem),
            CVLS_ILL_INPUT,
            line!() as i32,
            "CVodeSetLinearSolver",
            file!(),
            "Incompatible inputs: direct LS requires non-NULL matrix",
        );
        return CVLS_ILL_INPUT;
    }

    /* free any existing system solver attached to CVode */
    let lfree = cvode_mem.borrow().cv_lfree;
    if let Some(lfree) = lfree {
        lfree(cvode_mem);
    }

    /* Set four main system linear solver function fields in cv_mem */
    {
        let mut m = cvode_mem.borrow_mut();
        m.cv_linit = Some(cvLsInitialize);
        m.cv_lreinit = Some(cvLsReInitialize);
        m.cv_lsetup = Some(cvLsSetup);
        m.cv_lsolve = Some(cvLsSolve);
        m.cv_lfree = Some(cvLsFree);
    }

    /* Allocate memory for CVLsMemRec (C: malloc + memset(0), then the
    default assignments below; malloc failure is unreachable here).
    The struct literal carries exactly the state the C code holds after
    its default-assignment block (through `last_flag = CVLS_SUCCESS`). */
    let cv_f = cvode_mem.borrow().cv_f;
    let mut cvls_mem: CVLsMem = Box::new(CVLsMemRec {
        /* set SUNLinearSolver pointer */
        LS: LS.clone(),
        /* Linear solver type information */
        iterative,
        matrixbased,
        /* Set defaults for Jacobian-related fields */
        jacDQ: A.is_some(),
        jac: if A.is_some() { Some(cvLsDQJac as CVLsJacFn) } else { None },
        J_data: if A.is_some() {
            Some(Box::new(cvode_mem.clone())) /* C: J_data = cv_mem */
        } else {
            None
        },
        jtimesDQ: SUNTRUE,
        jtsetup: None,
        jtimes: Some(cvLsDQJtimes),
        jt_f: cv_f,
        jt_data: Some(Box::new(cvode_mem.clone())), /* C: jt_data = cv_mem */
        user_linsys: SUNFALSE,
        linsys: Some(cvLsLinSys),
        A_data: Some(Box::new(cvode_mem.clone())), /* C: A_data = cv_mem */
        /* Set defaults for preconditioner-related fields */
        pset: None,
        psolve: None,
        pfree: None,
        P_data: None, /* C: P_data = cv_mem->cv_user_data (pass-through) */
        /* Initialize counters (cvLsInitializeCounters below re-zeros) */
        nje: 0,
        nfeDQ: 0,
        nstlj: 0,
        npe: 0,
        nli: 0,
        nps: 0,
        ncfl: 0,
        njtsetup: 0,
        njtimes: 0,
        tnlj: 0.0, /* memset baseline */
        /* Set default values for the rest of the LS parameters */
        msbj: CVLS_MSBJ,
        jbad: SUNTRUE,
        dgmax_jbad: CVLS_DGMAX,
        eplifac: CVLS_EPLIN,
        last_flag: CVLS_SUCCESS,
        /* memset(0) baseline for fields assigned further below */
        scalesol: SUNFALSE,
        nrmfac: 0.0,
        A: None,
        savedJ: None,
        ytemp: None,
        x: None,
        ycur: None,
        fcur: None,
    });

    /* Initialize counters */
    let _ = cvLsInitializeCounters(&mut cvls_mem);

    /* If LS supports ATimes, attach CVLs routine */
    if LS.ops.borrow().setatimes.is_some() {
        let retval = SUNLinSolSetATimes(LS, Some(Box::new(cvode_mem.clone())), Some(cvLsATimes));
        if retval != SUN_SUCCESS {
            cvProcessError(
                Some(cvode_mem),
                CVLS_SUNLS_FAIL,
                line!() as i32,
                "CVodeSetLinearSolver",
                file!(),
                "Error in calling SUNLinSolSetATimes",
            );
            drop(cvls_mem);
            return CVLS_SUNLS_FAIL;
        }
    }

    /* If LS supports preconditioning, initialize pset/psol to NULL */
    if LS.ops.borrow().setpreconditioner.is_some() {
        let retval =
            SUNLinSolSetPreconditioner(LS, Some(Box::new(cvode_mem.clone())), None, None);
        if retval != SUN_SUCCESS {
            cvProcessError(
                Some(cvode_mem),
                CVLS_SUNLS_FAIL,
                line!() as i32,
                "CVodeSetLinearSolver",
                file!(),
                "Error in calling SUNLinSolSetPreconditioner",
            );
            drop(cvls_mem);
            return CVLS_SUNLS_FAIL;
        }
    }

    /* When using a SUNMatrix object, store pointer to A and initialize savedJ */
    if let Some(A) = A {
        cvls_mem.A = Some(A.clone());
        cvls_mem.savedJ = None; /* allocated in cvLsInitialize */
    }

    /* Allocate memory for ytemp and x */
    match N_VClone(&cv_tempv) {
        Some(ytemp) => cvls_mem.ytemp = Some(ytemp),
        None => {
            cvProcessError(
                Some(cvode_mem),
                CVLS_MEM_FAIL,
                line!() as i32,
                "CVodeSetLinearSolver",
                file!(),
                MSG_LS_MEM_FAIL,
            );
            drop(cvls_mem);
            return CVLS_MEM_FAIL;
        }
    }

    match N_VClone(&cv_tempv) {
        Some(x) => cvls_mem.x = Some(x),
        None => {
            cvProcessError(
                Some(cvode_mem),
                CVLS_MEM_FAIL,
                line!() as i32,
                "CVodeSetLinearSolver",
                file!(),
                MSG_LS_MEM_FAIL,
            );
            if let Some(ytemp) = cvls_mem.ytemp.take() {
                N_VDestroy(ytemp);
            }
            drop(cvls_mem);
            return CVLS_MEM_FAIL;
        }
    }

    /* For iterative LS, compute default norm conversion factor */
    if iterative {
        cvls_mem.nrmfac =
            SUNRsqrt(N_VGetLength(cvls_mem.ytemp.as_ref().expect("ytemp")) as sunrealtype);
    }

    /* Check if solution scaling should be enabled */
    if matrixbased && cvode_mem.borrow().cv_lmm == CV_BDF {
        cvls_mem.scalesol = SUNTRUE;
    } else {
        cvls_mem.scalesol = SUNFALSE;
    }

    /* Attach linear solver memory to integrator memory */
    cvode_mem.borrow_mut().cv_lmem = Some(cvls_mem);

    CVLS_SUCCESS
}

/*===============================================================
  Optional Set routines
  ===============================================================*/

/* CVodeSetJacFn specifies the Jacobian function. */
pub fn CVodeSetJacFn(cvode_mem: &CVodeMem, jac: Option<CVLsJacFn>) -> i32 {
    /* access CVLsMem structure */
    let retval = cvLs_AccessLMem(cvode_mem, "CVodeSetJacFn");
    if retval != CVLS_SUCCESS {
        return retval;
    }

    /* return with failure if jac cannot be used */
    if jac.is_some() && cvls_mem_mut(cvode_mem).A.is_none() {
        cvProcessError(
            Some(cvode_mem),
            CVLS_ILL_INPUT,
            line!() as i32,
            "CVodeSetJacFn",
            file!(),
            "Jacobian routine cannot be supplied for NULL SUNMatrix",
        );
        return CVLS_ILL_INPUT;
    }

    /* set the Jacobian routine pointer, and update relevant flags */
    if jac.is_some() {
        let mut ls = cvls_mem_mut(cvode_mem);
        ls.jacDQ = SUNFALSE;
        ls.jac = jac;
        ls.J_data = None; /* C: J_data = cv_mem->cv_user_data */
    } else {
        let mut ls = cvls_mem_mut(cvode_mem);
        ls.jacDQ = SUNTRUE;
        ls.jac = Some(cvLsDQJac);
        ls.J_data = Some(Box::new(cvode_mem.clone())); /* C: J_data = cv_mem */
    }

    /* ensure the internal linear system function is used */
    {
        let mut ls = cvls_mem_mut(cvode_mem);
        ls.user_linsys = SUNFALSE;
        ls.linsys = Some(cvLsLinSys);
        ls.A_data = Some(Box::new(cvode_mem.clone())); /* C: A_data = cv_mem */
    }

    CVLS_SUCCESS
}

/* CVodeSetDeltaGammaMaxBadJac specifies the maximum gamma ratio change
 * after a NLS convergence failure with a potentially bad Jacobian. If
 * |gamma/gammap-1| < dgmax_jbad then the Jacobian is marked as bad */
pub fn CVodeSetDeltaGammaMaxBadJac(cvode_mem: &CVodeMem, dgmax_jbad: sunrealtype) -> i32 {
    /* Access CVLsMem structure */
    let retval = cvLs_AccessLMem(cvode_mem, "CVodeSetDeltaGammaMaxBadJac");
    if retval != CVLS_SUCCESS {
        return retval;
    }

    /* Set value or use default */
    if dgmax_jbad <= ZERO {
        cvls_mem_mut(cvode_mem).dgmax_jbad = CVLS_DGMAX;
    } else {
        cvls_mem_mut(cvode_mem).dgmax_jbad = dgmax_jbad;
    }

    CVLS_SUCCESS
}

/* CVodeSetEpsLin specifies the nonlinear -> linear tolerance scale factor */
pub fn CVodeSetEpsLin(cvode_mem: &CVodeMem, eplifac: sunrealtype) -> i32 {
    /* access CVLsMem structure */
    let retval = cvLs_AccessLMem(cvode_mem, "CVodeSetEpsLin");
    if retval != CVLS_SUCCESS {
        return retval;
    }

    /* Check for legal eplifac */
    if eplifac < ZERO {
        cvProcessError(
            Some(cvode_mem),
            CVLS_ILL_INPUT,
            line!() as i32,
            "CVodeSetEpsLin",
            file!(),
            MSG_LS_BAD_EPLIN,
        );
        return CVLS_ILL_INPUT;
    }

    cvls_mem_mut(cvode_mem).eplifac = if eplifac == ZERO { CVLS_EPLIN } else { eplifac };

    CVLS_SUCCESS
}

/* CVodeSetLSNormFactor sets or computes the factor to use when converting from
   the integrator tolerance to the linear solver tolerance (WRMS to L2 norm). */
pub fn CVodeSetLSNormFactor(cvode_mem: &CVodeMem, nrmfac: sunrealtype) -> i32 {
    /* access CVLsMem structure */
    let retval = cvLs_AccessLMem(cvode_mem, "CVodeSetLSNormFactor");
    if retval != CVLS_SUCCESS {
        return retval;
    }

    if nrmfac > ZERO {
        /* user-provided factor */
        cvls_mem_mut(cvode_mem).nrmfac = nrmfac;
    } else if nrmfac < ZERO {
        /* compute factor for WRMS norm with dot product */
        let ytemp = cvls_mem_mut(cvode_mem).ytemp.as_ref().expect("ytemp").clone();
        N_VConst(ONE, &ytemp);
        cvls_mem_mut(cvode_mem).nrmfac = SUNRsqrt(N_VDotProd(&ytemp, &ytemp));
    } else {
        /* compute default factor for WRMS norm from vector length */
        let ytemp = cvls_mem_mut(cvode_mem).ytemp.as_ref().expect("ytemp").clone();
        cvls_mem_mut(cvode_mem).nrmfac = SUNRsqrt(N_VGetLength(&ytemp) as sunrealtype);
    }

    CVLS_SUCCESS
}

/* CVodeSetJacEvalFrequency specifies the frequency for recomputing the Jacobian
   matrix and/or preconditioner */
pub fn CVodeSetJacEvalFrequency(cvode_mem: &CVodeMem, msbj: i64) -> i32 {
    /* access CVLsMem structure; store input and return */
    let retval = cvLs_AccessLMem(cvode_mem, "CVodeSetJacEvalFrequency");
    if retval != CVLS_SUCCESS {
        return retval;
    }

    /* Check for legal msbj */
    if msbj < 0 {
        cvProcessError(
            Some(cvode_mem),
            CVLS_ILL_INPUT,
            line!() as i32,
            "CVodeSetJacEvalFrequency",
            file!(),
            "A negative evaluation frequency was provided.",
        );
        return CVLS_ILL_INPUT;
    }

    cvls_mem_mut(cvode_mem).msbj = if msbj == 0 { CVLS_MSBJ } else { msbj };

    CVLS_SUCCESS
}

/* CVodeSetLinearSolutionScaling enables or disables scaling the
   linear solver solution to account for changes in gamma. */
pub fn CVodeSetLinearSolutionScaling(cvode_mem: &CVodeMem, onoff: sunbooleantype) -> i32 {
    /* access CVLsMem structure; store input and return */
    let retval = cvLs_AccessLMem(cvode_mem, "CVodeSetLinearSolutionScaling");
    if retval != CVLS_SUCCESS {
        return retval;
    }

    /* check for valid solver and method type */
    let matrixbased = cvls_mem_mut(cvode_mem).matrixbased;
    if !matrixbased || cvode_mem.borrow().cv_lmm != CV_BDF {
        return CVLS_ILL_INPUT;
    }

    /* set solution scaling flag */
    cvls_mem_mut(cvode_mem).scalesol = onoff;

    CVLS_SUCCESS
}

/* CVodeSetPreconditioner specifies the user-supplied preconditioner
   setup and solve routines */
pub fn CVodeSetPreconditioner(
    cvode_mem: &CVodeMem,
    psetup: Option<CVLsPrecSetupFn>,
    psolve: Option<CVLsPrecSolveFn>,
) -> i32 {
    /* access CVLsMem structure */
    let retval = cvLs_AccessLMem(cvode_mem, "CVodeSetPreconditioner");
    if retval != CVLS_SUCCESS {
        return retval;
    }

    /* store function pointers for user-supplied routines in CVLs interface */
    {
        let mut ls = cvls_mem_mut(cvode_mem);
        ls.pset = psetup;
        ls.psolve = psolve;
    }

    /* issue error if LS object does not allow user-supplied preconditioning */
    let LS = cvls_mem_mut(cvode_mem).LS.clone();
    if LS.ops.borrow().setpreconditioner.is_none() {
        cvProcessError(
            Some(cvode_mem),
            CVLS_ILL_INPUT,
            line!() as i32,
            "CVodeSetPreconditioner",
            file!(),
            "SUNLinearSolver object does not support user-supplied preconditioning",
        );
        return CVLS_ILL_INPUT;
    }

    /* notify iterative linear solver to call CVLs interface routines */
    let cvls_psetup: Option<SUNPSetupFn> = if psetup.is_none() { None } else { Some(cvLsPSetup) };
    let cvls_psolve: Option<SUNPSolveFn> = if psolve.is_none() { None } else { Some(cvLsPSolve) };
    let retval = SUNLinSolSetPreconditioner(
        &LS,
        Some(Box::new(cvode_mem.clone())),
        cvls_psetup,
        cvls_psolve,
    );
    if retval != SUN_SUCCESS {
        cvProcessError(
            Some(cvode_mem),
            CVLS_SUNLS_FAIL,
            line!() as i32,
            "CVodeSetPreconditioner",
            file!(),
            "Error in calling SUNLinSolSetPreconditioner",
        );
        return CVLS_SUNLS_FAIL;
    }

    CVLS_SUCCESS
}

/* CVodeSetJacTimes specifies the user-supplied Jacobian-vector product
   setup and multiply routines */
pub fn CVodeSetJacTimes(
    cvode_mem: &CVodeMem,
    jtsetup: Option<CVLsJacTimesSetupFn>,
    jtimes: Option<CVLsJacTimesVecFn>,
) -> i32 {
    /* access CVLsMem structure */
    let retval = cvLs_AccessLMem(cvode_mem, "CVodeSetJacTimes");
    if retval != CVLS_SUCCESS {
        return retval;
    }

    /* issue error if LS object does not allow user-supplied ATimes */
    let LS = cvls_mem_mut(cvode_mem).LS.clone();
    if LS.ops.borrow().setatimes.is_none() {
        cvProcessError(
            Some(cvode_mem),
            CVLS_ILL_INPUT,
            line!() as i32,
            "CVodeSetJacTimes",
            file!(),
            "SUNLinearSolver object does not support user-supplied ATimes routine",
        );
        return CVLS_ILL_INPUT;
    }

    /* store function pointers for user-supplied routines in CVLs
    interface (NULL jtimes implies use of DQ default) */
    if jtimes.is_some() {
        let mut ls = cvls_mem_mut(cvode_mem);
        ls.jtimesDQ = SUNFALSE;
        ls.jtsetup = jtsetup;
        ls.jtimes = jtimes;
        ls.jt_data = None; /* C: jt_data = cv_mem->cv_user_data */
    } else {
        let cv_f = cvode_mem.borrow().cv_f;
        let mut ls = cvls_mem_mut(cvode_mem);
        ls.jtimesDQ = SUNTRUE;
        ls.jtsetup = None;
        ls.jtimes = Some(cvLsDQJtimes);
        ls.jt_f = cv_f;
        ls.jt_data = Some(Box::new(cvode_mem.clone())); /* C: jt_data = cv_mem */
    }

    CVLS_SUCCESS
}

/* CVodeSetJacTimesRhsFn specifies an alternative user-supplied ODE right-hand
   side function to use in the internal finite difference Jacobian-vector
   product */
pub fn CVodeSetJacTimesRhsFn(cvode_mem: &CVodeMem, jtimesRhsFn: Option<CVRhsFn>) -> i32 {
    /* access CVLsMem structure */
    let retval = cvLs_AccessLMem(cvode_mem, "CVodeSetJacTimesRhsFn");
    if retval != CVLS_SUCCESS {
        return retval;
    }

    /* check if using internal finite difference approximation */
    if !cvls_mem_mut(cvode_mem).jtimesDQ {
        cvProcessError(
            Some(cvode_mem),
            CVLS_ILL_INPUT,
            line!() as i32,
            "CVodeSetJacTimesRhsFn",
            file!(),
            "Internal finite-difference Jacobian-vector product is disabled.",
        );
        return CVLS_ILL_INPUT;
    }

    /* store function pointers for RHS function (NULL implies use ODE RHS) */
    if jtimesRhsFn.is_some() {
        cvls_mem_mut(cvode_mem).jt_f = jtimesRhsFn;
    } else {
        let cv_f = cvode_mem.borrow().cv_f;
        cvls_mem_mut(cvode_mem).jt_f = cv_f;
    }

    CVLS_SUCCESS
}

/* CVodeSetLinSysFn specifies the linear system setup function. */
pub fn CVodeSetLinSysFn(cvode_mem: &CVodeMem, linsys: Option<CVLsLinSysFn>) -> i32 {
    /* access CVLsMem structure */
    let retval = cvLs_AccessLMem(cvode_mem, "CVodeSetLinSysFn");
    if retval != CVLS_SUCCESS {
        return retval;
    }

    /* return with failure if linsys cannot be used */
    if linsys.is_some() && cvls_mem_mut(cvode_mem).A.is_none() {
        cvProcessError(
            Some(cvode_mem),
            CVLS_ILL_INPUT,
            line!() as i32,
            "CVodeSetLinSysFn",
            file!(),
            "Linear system setup routine cannot be supplied for NULL SUNMatrix",
        );
        return CVLS_ILL_INPUT;
    }

    /* set the linear system routine pointer, and update relevant flags */
    if linsys.is_some() {
        let mut ls = cvls_mem_mut(cvode_mem);
        ls.user_linsys = SUNTRUE;
        ls.linsys = linsys;
        ls.A_data = None; /* C: A_data = cv_mem->cv_user_data */
    } else {
        let mut ls = cvls_mem_mut(cvode_mem);
        ls.user_linsys = SUNFALSE;
        ls.linsys = Some(cvLsLinSys);
        ls.A_data = Some(Box::new(cvode_mem.clone())); /* C: A_data = cv_mem */
    }

    CVLS_SUCCESS
}

/*===============================================================
  Optional Get routines
  ===============================================================*/

pub fn CVodeGetJac(cvode_mem: &CVodeMem, J: &mut Option<SUNMatrix>) -> i32 {
    /* access CVLsMem structure; set output and return */
    let retval = cvLs_AccessLMem(cvode_mem, "CVodeGetJac");
    if retval != CVLS_SUCCESS {
        return retval;
    }
    *J = cvls_mem_mut(cvode_mem).savedJ.clone();
    CVLS_SUCCESS
}

pub fn CVodeGetJacTime(cvode_mem: &CVodeMem, t_J: &mut sunrealtype) -> i32 {
    /* access CVLsMem structure; set output and return */
    let retval = cvLs_AccessLMem(cvode_mem, "CVodeGetJacTime");
    if retval != CVLS_SUCCESS {
        return retval;
    }
    *t_J = cvls_mem_mut(cvode_mem).tnlj;
    CVLS_SUCCESS
}

pub fn CVodeGetJacNumSteps(cvode_mem: &CVodeMem, nst_J: &mut i64) -> i32 {
    /* access CVLsMem structure; set output and return */
    let retval = cvLs_AccessLMem(cvode_mem, "CVodeGetJacNumSteps");
    if retval != CVLS_SUCCESS {
        return retval;
    }
    *nst_J = cvls_mem_mut(cvode_mem).nstlj;
    CVLS_SUCCESS
}

/* CVodeGetLinWorkSpace returns the length of workspace allocated
   for the CVLS linear solver interface */
pub fn CVodeGetLinWorkSpace(cvode_mem: &CVodeMem, lenrwLS: &mut i64, leniwLS: &mut i64) -> i32 {
    /* access CVLsMem structure */
    let retval = cvLs_AccessLMem(cvode_mem, "CVodeGetLinWorkSpace");
    if retval != CVLS_SUCCESS {
        return retval;
    }

    /* start with fixed sizes plus vector/matrix pointers */
    *lenrwLS = 2;
    *leniwLS = 30;

    /* add NVector sizes */
    let cv_tempv = cvode_mem
        .borrow()
        .cv_tempv
        .as_ref()
        .expect("cv_tempv")
        .clone();
    if cv_tempv.ops.borrow().nvspace.is_some() {
        let mut lrw1: sunindextype = 0;
        let mut liw1: sunindextype = 0;
        N_VSpace(&cv_tempv, &mut lrw1, &mut liw1);
        *lenrwLS += 2 * lrw1;
        *leniwLS += 2 * liw1;
    }

    /* add SUNMatrix size (only account for the one owned by Ls interface) */
    let savedJ = cvls_mem_mut(cvode_mem).savedJ.clone();
    if let Some(savedJ) = &savedJ {
        if savedJ.ops.borrow().space.is_some() {
            let mut lrw: i64 = 0;
            let mut liw: i64 = 0;
            let retval = SUNMatSpace(savedJ, &mut lrw, &mut liw);
            if retval == 0 {
                *lenrwLS += lrw;
                *leniwLS += liw;
            }
        }
    }

    /* add LS sizes */
    let LS = cvls_mem_mut(cvode_mem).LS.clone();
    if LS.ops.borrow().space.is_some() {
        let mut lrw: i64 = 0;
        let mut liw: i64 = 0;
        let retval = SUNLinSolSpace(&LS, &mut lrw, &mut liw);
        if retval == 0 {
            *lenrwLS += lrw;
            *leniwLS += liw;
        }
    }

    CVLS_SUCCESS
}

/* CVodeGetNumJacEvals returns the number of Jacobian evaluations */
pub fn CVodeGetNumJacEvals(cvode_mem: &CVodeMem, njevals: &mut i64) -> i32 {
    /* access CVLsMem structure; set output value and return */
    let retval = cvLs_AccessLMem(cvode_mem, "CVodeGetNumJacEvals");
    if retval != CVLS_SUCCESS {
        return retval;
    }
    *njevals = cvls_mem_mut(cvode_mem).nje;
    CVLS_SUCCESS
}

/* CVodeGetNumLinRhsEvals returns the number of calls to the ODE
   function needed for the DQ Jacobian approximation or J*v product
   approximation */
pub fn CVodeGetNumLinRhsEvals(cvode_mem: &CVodeMem, nfevalsLS: &mut i64) -> i32 {
    /* access CVLsMem structure; set output value and return */
    let retval = cvLs_AccessLMem(cvode_mem, "CVodeGetNumLinRhsEvals");
    if retval != CVLS_SUCCESS {
        return retval;
    }
    *nfevalsLS = cvls_mem_mut(cvode_mem).nfeDQ;
    CVLS_SUCCESS
}

/* CVodeGetNumPrecEvals returns the number of calls to the
   user- or CVode-supplied preconditioner setup routine */
pub fn CVodeGetNumPrecEvals(cvode_mem: &CVodeMem, npevals: &mut i64) -> i32 {
    /* access CVLsMem structure; set output value and return */
    let retval = cvLs_AccessLMem(cvode_mem, "CVodeGetNumPrecEvals");
    if retval != CVLS_SUCCESS {
        return retval;
    }
    *npevals = cvls_mem_mut(cvode_mem).npe;
    CVLS_SUCCESS
}

/* CVodeGetNumPrecSolves returns the number of calls to the
   user- or CVode-supplied preconditioner solve routine */
pub fn CVodeGetNumPrecSolves(cvode_mem: &CVodeMem, npsolves: &mut i64) -> i32 {
    /* access CVLsMem structure; set output value and return */
    let retval = cvLs_AccessLMem(cvode_mem, "CVodeGetNumPrecSolves");
    if retval != CVLS_SUCCESS {
        return retval;
    }
    *npsolves = cvls_mem_mut(cvode_mem).nps;
    CVLS_SUCCESS
}

/* CVodeGetNumLinIters returns the number of linear iterations
   (if accessible from the LS object) */
pub fn CVodeGetNumLinIters(cvode_mem: &CVodeMem, nliters: &mut i64) -> i32 {
    /* access CVLsMem structure; set output value and return */
    let retval = cvLs_AccessLMem(cvode_mem, "CVodeGetNumLinIters");
    if retval != CVLS_SUCCESS {
        return retval;
    }
    *nliters = cvls_mem_mut(cvode_mem).nli;
    CVLS_SUCCESS
}

/* CVodeGetNumLinConvFails returns the number of linear solver
   convergence failures (as reported by the LS object) */
pub fn CVodeGetNumLinConvFails(cvode_mem: &CVodeMem, nlcfails: &mut i64) -> i32 {
    /* access CVLsMem structure; set output value and return */
    let retval = cvLs_AccessLMem(cvode_mem, "CVodeGetNumLinConvFails");
    if retval != CVLS_SUCCESS {
        return retval;
    }
    *nlcfails = cvls_mem_mut(cvode_mem).ncfl;
    CVLS_SUCCESS
}

/* CVodeGetNumJTSetupEvals returns the number of calls to the
   user-supplied Jacobian-vector product setup routine */
pub fn CVodeGetNumJTSetupEvals(cvode_mem: &CVodeMem, njtsetups: &mut i64) -> i32 {
    /* access CVLsMem structure; set output value and return */
    let retval = cvLs_AccessLMem(cvode_mem, "CVodeGetNumJTSetupEvals");
    if retval != CVLS_SUCCESS {
        return retval;
    }
    *njtsetups = cvls_mem_mut(cvode_mem).njtsetup;
    CVLS_SUCCESS
}

/* CVodeGetNumJtimesEvals returns the number of calls to the
   Jacobian-vector product multiply routine */
pub fn CVodeGetNumJtimesEvals(cvode_mem: &CVodeMem, njvevals: &mut i64) -> i32 {
    /* access CVLsMem structure; set output value and return */
    let retval = cvLs_AccessLMem(cvode_mem, "CVodeGetNumJtimesEvals");
    if retval != CVLS_SUCCESS {
        return retval;
    }
    *njvevals = cvls_mem_mut(cvode_mem).njtimes;
    CVLS_SUCCESS
}

/* CVodeGetLinSolveStats returns statistics related to the linear solve. */
#[allow(clippy::too_many_arguments)]
pub fn CVodeGetLinSolveStats(
    cvode_mem: &CVodeMem,
    njevals: &mut i64,
    nfevalsLS: &mut i64,
    nliters: &mut i64,
    nlcfails: &mut i64,
    npevals: &mut i64,
    npsolves: &mut i64,
    njtsetups: &mut i64,
    njtimes: &mut i64,
) -> i32 {
    /* access CVLsMem structure; set output value and return */
    let retval = cvLs_AccessLMem(cvode_mem, "CVodeGetLinSolveStats");
    if retval != CVLS_SUCCESS {
        return retval;
    }

    let ls = cvls_mem_mut(cvode_mem);
    *njevals = ls.nje;
    *nfevalsLS = ls.nfeDQ;
    *nliters = ls.nli;
    *nlcfails = ls.ncfl;
    *npevals = ls.npe;
    *npsolves = ls.nps;
    *njtsetups = ls.njtsetup;
    *njtimes = ls.njtimes;

    CVLS_SUCCESS
}

/* CVodeGetLastLinFlag returns the last flag set in a CVLS function */
pub fn CVodeGetLastLinFlag(cvode_mem: &CVodeMem, flag: &mut i64) -> i32 {
    /* access CVLsMem structure; set output value and return */
    let retval = cvLs_AccessLMem(cvode_mem, "CVodeGetLastLinFlag");
    if retval != CVLS_SUCCESS {
        return retval;
    }
    *flag = cvls_mem_mut(cvode_mem).last_flag as i64;
    CVLS_SUCCESS
}

/* CVodeGetLinReturnFlagName translates from the integer error code
   returned by an CVLs routine to the corresponding string
   equivalent for that flag */
pub fn CVodeGetLinReturnFlagName(flag: i64) -> String {
    let name = if flag == CVLS_SUCCESS as i64 {
        "CVLS_SUCCESS"
    } else if flag == CVLS_MEM_NULL as i64 {
        "CVLS_MEM_NULL"
    } else if flag == CVLS_LMEM_NULL as i64 {
        "CVLS_LMEM_NULL"
    } else if flag == CVLS_ILL_INPUT as i64 {
        "CVLS_ILL_INPUT"
    } else if flag == CVLS_MEM_FAIL as i64 {
        "CVLS_MEM_FAIL"
    } else if flag == CVLS_PMEM_NULL as i64 {
        "CVLS_PMEM_NULL"
    } else if flag == CVLS_JACFUNC_UNRECVR as i64 {
        "CVLS_JACFUNC_UNRECVR"
    } else if flag == CVLS_JACFUNC_RECVR as i64 {
        "CVLS_JACFUNC_RECVR"
    } else if flag == CVLS_SUNMAT_FAIL as i64 {
        "CVLS_SUNMAT_FAIL"
    } else if flag == CVLS_SUNLS_FAIL as i64 {
        "CVLS_SUNLS_FAIL"
    } else {
        "NONE"
    };
    name.to_string()
}

/*=================================================================
  CVLS private functions
  =================================================================*/

/*-----------------------------------------------------------------
  cvLsATimes

  This routine generates the matrix-vector product z = Mv, where
  M = I - gamma*J. The product J*v is obtained by calling the jtimes
  routine. It is then scaled by -gamma and added to v to obtain M*v.
  The return value is the same as the value returned by jtimes --
  0 if successful, nonzero otherwise.
  -----------------------------------------------------------------*/
pub fn cvLsATimes(cvode_mem: &mut Option<Box<dyn Any>>, v: &N_Vector, z: &N_Vector) -> i32 {
    /* access CVLsMem structure */
    let cv_mem = match cvLs_AccessLMemToken(cvode_mem, "cvLsATimes") {
        Ok(m) => m,
        Err(retval) => return retval,
    };

    /* call Jacobian-times-vector product routine
    (either user-supplied or internal DQ) */
    let tn = cv_mem.borrow().cv_tn;
    let (jtimes, ycur, fcur, ytemp) = {
        let ls = cvls_mem_mut(&cv_mem);
        (
            ls.jtimes.expect("jtimes"),
            ls.ycur.as_ref().expect("ycur").clone(),
            ls.fcur.as_ref().expect("fcur").clone(),
            ls.ytemp.as_ref().expect("ytemp").clone(),
        )
    };
    let use_field = cvls_mem_mut(&cv_mem).jt_data.is_some();
    let mut jt_data = if use_field {
        cvls_mem_mut(&cv_mem).jt_data.take()
    } else {
        cv_mem.borrow_mut().cv_user_data.take()
    };
    let retval = jtimes(v, z, tn, &ycur, &fcur, &mut jt_data, &ytemp);
    if use_field {
        cvls_mem_mut(&cv_mem).jt_data = jt_data;
    } else {
        cv_mem.borrow_mut().cv_user_data = jt_data;
    }
    cvls_mem_mut(&cv_mem).njtimes += 1;
    if retval != 0 {
        return retval;
    }

    /* add contribution from identity matrix */
    let gamma = cv_mem.borrow().cv_gamma;
    N_VLinearSum(ONE, v, -gamma, z, z);

    0
}

/*---------------------------------------------------------------
  cvLsPSetup:

  This routine interfaces between the generic iterative linear
  solvers and the user's psetup routine.  It passes to psetup all
  required state information from cvode_mem.  Its return value
  is the same as that returned by psetup. Note that the generic
  iterative linear solvers guarantee that cvLsPSetup will only
  be called in the case that the user's psetup routine is non-NULL.
  ---------------------------------------------------------------*/
pub fn cvLsPSetup(cvode_mem: &mut Option<Box<dyn Any>>) -> i32 {
    /* access CVLsMem structure */
    let cv_mem = match cvLs_AccessLMemToken(cvode_mem, "cvLsPSetup") {
        Ok(m) => m,
        Err(retval) => return retval,
    };

    /* Call user pset routine to update preconditioner and possibly
    reset jcur (pass !jbad as update suggestion) */
    let (tn, gamma) = {
        let m = cv_mem.borrow();
        (m.cv_tn, m.cv_gamma)
    };
    let (pset, ycur, fcur, jbad) = {
        let ls = cvls_mem_mut(&cv_mem);
        (
            ls.pset.expect("pset"),
            ls.ycur.as_ref().expect("ycur").clone(),
            ls.fcur.as_ref().expect("fcur").clone(),
            ls.jbad,
        )
    };
    let mut jcur = cv_mem.borrow().cv_jcur;
    let use_field = cvls_mem_mut(&cv_mem).P_data.is_some();
    let mut p_data = if use_field {
        cvls_mem_mut(&cv_mem).P_data.take()
    } else {
        cv_mem.borrow_mut().cv_user_data.take()
    };
    let retval = pset(tn, &ycur, &fcur, !jbad, &mut jcur, gamma, &mut p_data);
    if use_field {
        cvls_mem_mut(&cv_mem).P_data = p_data;
    } else {
        cv_mem.borrow_mut().cv_user_data = p_data;
    }
    cv_mem.borrow_mut().cv_jcur = jcur;
    retval
}

/*-----------------------------------------------------------------
  cvLsPSolve

  This routine interfaces between the generic SUNLinSolSolve
  routine and the user's psolve routine.  It passes to psolve all
  required state information from cvode_mem.  Its return value is
  the same as that returned by psolve. Note that the generic
  SUNLinSol solver guarantees that cvLsPSolve will not be called
  in the case in which preconditioning is not done. This is the
  only case in which the user's psolve routine is allowed to be
  NULL.
  -----------------------------------------------------------------*/
pub fn cvLsPSolve(
    cvode_mem: &mut Option<Box<dyn Any>>,
    r: &N_Vector,
    z: &N_Vector,
    tol: sunrealtype,
    lr: i32,
) -> i32 {
    /* access CVLsMem structure */
    let cv_mem = match cvLs_AccessLMemToken(cvode_mem, "cvLsPSolve") {
        Ok(m) => m,
        Err(retval) => return retval,
    };

    /* call the user-supplied psolve routine, and accumulate count */
    let (tn, gamma) = {
        let m = cv_mem.borrow();
        (m.cv_tn, m.cv_gamma)
    };
    let (psolve, ycur, fcur) = {
        let ls = cvls_mem_mut(&cv_mem);
        (
            ls.psolve.expect("psolve"),
            ls.ycur.as_ref().expect("ycur").clone(),
            ls.fcur.as_ref().expect("fcur").clone(),
        )
    };
    let use_field = cvls_mem_mut(&cv_mem).P_data.is_some();
    let mut p_data = if use_field {
        cvls_mem_mut(&cv_mem).P_data.take()
    } else {
        cv_mem.borrow_mut().cv_user_data.take()
    };
    let retval = psolve(tn, &ycur, &fcur, r, z, gamma, tol, lr, &mut p_data);
    if use_field {
        cvls_mem_mut(&cv_mem).P_data = p_data;
    } else {
        cv_mem.borrow_mut().cv_user_data = p_data;
    }
    cvls_mem_mut(&cv_mem).nps += 1;
    retval
}

/*-----------------------------------------------------------------
  cvLsDQJac

  This routine is a wrapper for the Dense and Band
  implementations of the difference quotient Jacobian
  approximation routines.
  ---------------------------------------------------------------*/
#[allow(clippy::too_many_arguments)]
pub fn cvLsDQJac(
    t: sunrealtype,
    y: &N_Vector,
    fy: &N_Vector,
    Jac: &SUNMatrix,
    cvode_mem: &mut Option<Box<dyn Any>>,
    tmp1: &N_Vector,
    tmp2: &N_Vector,
    tmp3: &N_Vector,
) -> i32 {
    let _ = tmp3; /* SUNDIALS_MAYBE_UNUSED */

    /* access CVodeMem structure */
    let cv_mem = match cvode_mem
        .as_ref()
        .and_then(|b| b.downcast_ref::<CVodeMem>())
        .cloned()
    {
        Some(m) => m,
        None => {
            cvProcessError(
                None,
                CVLS_MEM_NULL,
                line!() as i32,
                "cvLsDQJac",
                file!(),
                MSG_LS_CVMEM_NULL,
            );
            return CVLS_MEM_NULL;
        }
    };

    /* Jac non-NULL check: handled by type system */

    /* Verify that N_Vector supports required operations */
    let cv_tempv = cv_mem
        .borrow()
        .cv_tempv
        .as_ref()
        .expect("cv_tempv")
        .clone();
    {
        let ops = cv_tempv.ops.borrow();
        if ops.nvcloneempty.is_none()
            || ops.nvwrmsnorm.is_none()
            || ops.nvlinearsum.is_none()
            || ops.nvdestroy.is_none()
            || ops.nvscale.is_none()
            || ops.nvgetarraypointer.is_none()
            || ops.nvsetarraypointer.is_none()
        {
            cvProcessError(
                Some(&cv_mem),
                CVLS_ILL_INPUT,
                line!() as i32,
                "cvLsDQJac",
                file!(),
                MSG_LS_BAD_NVECTOR,
            );
            return CVLS_ILL_INPUT;
        }
    }

    /* Call the matrix-structure-specific DQ approximation routine */
    let retval;
    if SUNMatGetID(Jac) == SUNMATRIX_DENSE {
        retval = cvLsDenseDQJac(t, y, fy, Jac, &cv_mem, tmp1);
    } else if SUNMatGetID(Jac) == SUNMATRIX_BAND {
        retval = cvLsBandDQJac(t, y, fy, Jac, &cv_mem, tmp1, tmp2);
    } else {
        cvProcessError(
            Some(&cv_mem),
            CVLS_ILL_INPUT,
            line!() as i32,
            "cvLsDQJac",
            file!(),
            "unrecognized matrix type for cvLsDQJac",
        );
        retval = CVLS_ILL_INPUT;
    }
    retval
}

/*-----------------------------------------------------------------
  cvLsDenseDQJac

  This routine generates a dense difference quotient approximation
  to the Jacobian of f(t,y). It assumes that a dense SUNMatrix is
  stored column-wise, and that elements within each column are
  contiguous. The jth column of J is computed into the `jthCol`
  vector via N_VLinearSum and written back into the matrix column
  (the C code aliases the column memory with N_VSetArrayPointer;
  the copy-in/copy-out here is bit-identical).
  -----------------------------------------------------------------*/
pub fn cvLsDenseDQJac(
    t: sunrealtype,
    y: &N_Vector,
    fy: &N_Vector,
    Jac: &SUNMatrix,
    cv_mem: &CVodeMem,
    tmp1: &N_Vector,
) -> i32 {
    let mut retval: i32 = 0;

    /* access matrix dimension */
    let N = SUNDenseMatrix_Columns(Jac);

    /* Rename work vector for readability */
    let ftemp = tmp1;

    /* Create an empty vector for matrix column calculations */
    let jthCol = N_VCloneEmpty(tmp1).expect("N_VCloneEmpty");

    /* Obtain integrator state (C caches raw data pointers; here the
    data borrows are re-taken per use and never held across the RHS
    callback or a vector op) */
    let (uround, h, ewt, constraints, f) = {
        let m = cv_mem.borrow();
        (
            m.cv_uround,
            m.cv_h,
            m.cv_ewt.as_ref().expect("cv_ewt").clone(),
            m.cv_constraints.clone(),
            m.cv_f.expect("cv_f"),
        )
    };

    /* Set minimum increment based on uround and norm of f */
    let srur = SUNRsqrt(uround);
    let fnorm = N_VWrmsNorm(fy, &ewt);
    let minInc = if fnorm != ZERO {
        MIN_INC_MULT * SUNRabs(h) * uround * N as sunrealtype * fnorm
    } else {
        ONE
    };

    let mut j: sunindextype = 0;
    while j < N {
        /* Generate the jth col of J(tn,y) */
        /* C: N_VSetArrayPointer(SUNDenseMatrix_Column(Jac, j), jthCol) —
        copy the column in; the result is written back after the
        N_VLinearSum below (write-through of the C alias). */
        let col_data = SUNDenseMatrix_Column(Jac, j).to_vec();
        N_VSetArrayPointer(col_data, &jthCol);

        let yjsaved;
        let mut inc;
        {
            let y_data = N_VGetArrayPointer(y).expect("y data");
            let ewt_data = N_VGetArrayPointer(&ewt).expect("ewt data");
            yjsaved = y_data[j as usize];
            inc = SUNMAX(srur * SUNRabs(yjsaved), minInc / ewt_data[j as usize]);
        }

        /* Adjust sign(inc) if y_j has an inequality constraint. */
        if let Some(constraints) = &constraints {
            let cns_data = N_VGetArrayPointer(constraints).expect("constraints data");
            let conj = cns_data[j as usize];
            if SUNRabs(conj) == ONE {
                if (yjsaved + inc) * conj < ZERO {
                    inc = -inc;
                }
            } else if SUNRabs(conj) == TWO && (yjsaved + inc) * conj <= ZERO {
                inc = -inc;
            }
        }

        {
            let mut y_data = N_VGetArrayPointer(y).expect("y data");
            y_data[j as usize] += inc;
        }

        let mut user_data = cv_mem.borrow_mut().cv_user_data.take();
        retval = f(t, y, ftemp, &mut user_data);
        cv_mem.borrow_mut().cv_user_data = user_data;
        cvls_mem_mut(cv_mem).nfeDQ += 1;
        if retval != 0 {
            break;
        }

        {
            let mut y_data = N_VGetArrayPointer(y).expect("y data");
            y_data[j as usize] = yjsaved;
        }

        let inc_inv = ONE / inc;
        N_VLinearSum(inc_inv, ftemp, -inc_inv, fy, &jthCol);

        /* write the computed column back into the matrix (C wrote it
        through the aliased column pointer) */
        {
            let jth_data = N_VGetArrayPointer(&jthCol).expect("jthCol data");
            let mut col_j = SUNDenseMatrix_Column(Jac, j);
            col_j.copy_from_slice(&jth_data);
        }

        j += 1;
    }

    /* Destroy jthCol vector */
    N_VSetArrayPointer(Vec::new(), &jthCol); /* SHOULDN'T BE NEEDED */
    N_VDestroy(jthCol);

    retval
}

/*-----------------------------------------------------------------
  cvLsBandDQJac

  This routine generates a banded difference quotient approximation
  to the Jacobian of f(t,y).  It assumes that a band SUNMatrix is
  stored column-wise, and that elements within each column are
  contiguous.
  -----------------------------------------------------------------*/
pub fn cvLsBandDQJac(
    t: sunrealtype,
    y: &N_Vector,
    fy: &N_Vector,
    Jac: &SUNMatrix,
    cv_mem: &CVodeMem,
    tmp1: &N_Vector,
    tmp2: &N_Vector,
) -> i32 {
    let mut retval: i32 = 0;

    /* access matrix dimensions */
    let N = SUNBandMatrix_Columns(Jac);
    let mupper = SUNBandMatrix_UpperBandwidth(Jac);
    let mlower = SUNBandMatrix_LowerBandwidth(Jac);
    let s_mu = SUNBandMatrix_StoredUpperBandwidth(Jac); /* SM_COLUMN_ELEMENT_B offset */

    /* Rename work vectors for use as temporary values of y and f */
    let ftemp = tmp1;
    let ytemp = tmp2;

    /* Obtain integrator state (data borrows re-taken per phase; never
    held across the RHS callback or a vector op) */
    let (uround, h, ewt, constraints, f) = {
        let m = cv_mem.borrow();
        (
            m.cv_uround,
            m.cv_h,
            m.cv_ewt.as_ref().expect("cv_ewt").clone(),
            m.cv_constraints.clone(),
            m.cv_f.expect("cv_f"),
        )
    };

    /* Load ytemp with y = predicted y vector */
    N_VScale(ONE, y, ytemp);

    /* Set minimum increment based on uround and norm of f */
    let srur = SUNRsqrt(uround);
    let fnorm = N_VWrmsNorm(fy, &ewt);
    let minInc = if fnorm != ZERO {
        MIN_INC_MULT * SUNRabs(h) * uround * N as sunrealtype * fnorm
    } else {
        ONE
    };

    /* Set bandwidth and number of column groups for band differencing */
    let width = mlower + mupper + 1;
    let ngroups = SUNMIN(width, N);

    /* Loop over column groups. */
    let mut group: sunindextype = 1;
    while group <= ngroups {
        /* Increment all y_j in group */
        {
            let y_data = N_VGetArrayPointer(y).expect("y data");
            let ewt_data = N_VGetArrayPointer(&ewt).expect("ewt data");
            let mut ytemp_data = N_VGetArrayPointer(ytemp).expect("ytemp data");
            let cns_guard = constraints
                .as_ref()
                .map(|c| N_VGetArrayPointer(c).expect("constraints data"));
            let mut j = group - 1;
            while j < N {
                let mut inc = SUNMAX(
                    srur * SUNRabs(y_data[j as usize]),
                    minInc / ewt_data[j as usize],
                );

                /* Adjust sign(inc) if yj has an inequality constraint. */
                if let Some(cns_data) = &cns_guard {
                    let conj = cns_data[j as usize];
                    if SUNRabs(conj) == ONE {
                        if (ytemp_data[j as usize] + inc) * conj < ZERO {
                            inc = -inc;
                        }
                    } else if SUNRabs(conj) == TWO
                        && (ytemp_data[j as usize] + inc) * conj <= ZERO
                    {
                        inc = -inc;
                    }
                }

                ytemp_data[j as usize] += inc;
                j += width;
            }
        }

        /* Evaluate f with incremented y */
        let mut user_data = cv_mem.borrow_mut().cv_user_data.take();
        retval = f(t, ytemp, ftemp, &mut user_data);
        cv_mem.borrow_mut().cv_user_data = user_data;
        cvls_mem_mut(cv_mem).nfeDQ += 1;
        if retval != 0 {
            break;
        }

        /* Restore ytemp, then form and load difference quotients */
        {
            let y_data = N_VGetArrayPointer(y).expect("y data");
            let ewt_data = N_VGetArrayPointer(&ewt).expect("ewt data");
            let mut ytemp_data = N_VGetArrayPointer(ytemp).expect("ytemp data");
            let ftemp_data = N_VGetArrayPointer(ftemp).expect("ftemp data");
            let fy_data = N_VGetArrayPointer(fy).expect("fy data");
            let cns_guard = constraints
                .as_ref()
                .map(|c| N_VGetArrayPointer(c).expect("constraints data"));
            let mut j = group - 1;
            while j < N {
                ytemp_data[j as usize] = y_data[j as usize];
                let mut col_j = SUNBandMatrix_Column(Jac, j);
                let mut inc = SUNMAX(
                    srur * SUNRabs(y_data[j as usize]),
                    minInc / ewt_data[j as usize],
                );

                /* Adjust sign(inc) as before. */
                if let Some(cns_data) = &cns_guard {
                    let conj = cns_data[j as usize];
                    if SUNRabs(conj) == ONE {
                        if (ytemp_data[j as usize] + inc) * conj < ZERO {
                            inc = -inc;
                        }
                    } else if SUNRabs(conj) == TWO
                        && (ytemp_data[j as usize] + inc) * conj <= ZERO
                    {
                        inc = -inc;
                    }
                }

                let inc_inv = ONE / inc;
                let i1 = SUNMAX(0, j - mupper);
                let i2 = SUNMIN(j + mlower, N - 1);
                let mut i = i1;
                while i <= i2 {
                    /* C: SM_COLUMN_ELEMENT_B(col_j, i, j) = ... */
                    col_j[SM_COLUMN_ELEMENT_IDX(i, j, s_mu)] =
                        inc_inv * (ftemp_data[i as usize] - fy_data[i as usize]);
                    i += 1;
                }
                j += width;
            }
        }

        group += 1;
    }

    retval
}

/*-----------------------------------------------------------------
  cvLsDQJtimes

  This routine generates a difference quotient approximation to
  the Jacobian times vector f_y(t,y) * v. The approximation is
  Jv = [f(y + v*sig) - f(y)]/sig, where sig = 1 / ||v||_WRMS,
  i.e. the WRMS norm of v*sig is 1.
  -----------------------------------------------------------------*/
pub fn cvLsDQJtimes(
    v: &N_Vector,
    Jv: &N_Vector,
    t: sunrealtype,
    y: &N_Vector,
    fy: &N_Vector,
    cvode_mem: &mut Option<Box<dyn Any>>,
    work: &N_Vector,
) -> i32 {
    /* access CVLsMem structure */
    let cv_mem = match cvLs_AccessLMemToken(cvode_mem, "cvLsDQJtimes") {
        Ok(m) => m,
        Err(retval) => return retval,
    };

    /* Initialize perturbation to 1/||v|| */
    let ewt = cv_mem.borrow().cv_ewt.as_ref().expect("cv_ewt").clone();
    let mut sig = ONE / N_VWrmsNorm(v, &ewt);

    let jt_f = cvls_mem_mut(&cv_mem).jt_f.expect("jt_f");

    let mut retval: i32 = 0;
    let mut iter: i32 = 0;
    while iter < MAX_DQITERS {
        /* Set work = y + sig*v */
        N_VLinearSum(sig, v, ONE, y, work);

        /* Set Jv = f(tn, y+sig*v) */
        let mut user_data = cv_mem.borrow_mut().cv_user_data.take();
        retval = jt_f(t, work, Jv, &mut user_data);
        cv_mem.borrow_mut().cv_user_data = user_data;
        cvls_mem_mut(&cv_mem).nfeDQ += 1;
        if retval == 0 {
            break;
        }
        if retval < 0 {
            return -1;
        }

        /* If f failed recoverably, shrink sig and retry */
        sig *= PT25;
        iter += 1;
    }

    /* If retval still isn't 0, return with a recoverable failure */
    if retval > 0 {
        return 1;
    }

    /* Replace Jv by (Jv - fy)/sig */
    let siginv = ONE / sig;
    N_VLinearSum(siginv, Jv, -siginv, fy, Jv);

    0
}

/*-----------------------------------------------------------------
  cvLsLinSys

  Setup the linear system A = I - gamma J
  -----------------------------------------------------------------*/
#[allow(clippy::too_many_arguments)]
fn cvLsLinSys(
    t: sunrealtype,
    y: &N_Vector,
    fy: &N_Vector,
    A: &SUNMatrix,
    jok: sunbooleantype,
    jcur: &mut sunbooleantype,
    gamma: sunrealtype,
    cvode_mem: &mut Option<Box<dyn Any>>,
    vtemp1: &N_Vector,
    vtemp2: &N_Vector,
    vtemp3: &N_Vector,
) -> i32 {
    /* access CVLsMem structure */
    let cv_mem = match cvLs_AccessLMemToken(cvode_mem, "cvLsLinSys") {
        Ok(m) => m,
        Err(retval) => return retval,
    };

    /* Check if Jacobian needs to be updated */
    if jok {
        /* Use saved copy of J */
        *jcur = SUNFALSE;

        /* Overwrite linear system matrix with saved J */
        let savedJ = cvls_mem_mut(&cv_mem).savedJ.as_ref().expect("savedJ").clone();
        let retval = SUNMatCopy(&savedJ, A);
        if retval != 0 {
            cvProcessError(
                Some(&cv_mem),
                CVLS_SUNMAT_FAIL,
                line!() as i32,
                "cvLsLinSys",
                file!(),
                MSG_LS_SUNMAT_FAILED,
            );
            cvls_mem_mut(&cv_mem).last_flag = CVLS_SUNMAT_FAIL;
            return CVLS_SUNMAT_FAIL;
        }
    } else {
        /* Call jac() routine to update J */
        *jcur = SUNTRUE;

        /* Clear the linear system matrix if necessary */
        let LS = cvls_mem_mut(&cv_mem).LS.clone();
        if SUNLinSolGetType(&LS) == SUNLINEARSOLVER_DIRECT {
            let retval = SUNMatZero(A);
            if retval != 0 {
                cvProcessError(
                    Some(&cv_mem),
                    CVLS_SUNMAT_FAIL,
                    line!() as i32,
                    "cvLsLinSys",
                    file!(),
                    MSG_LS_SUNMAT_FAILED,
                );
                cvls_mem_mut(&cv_mem).last_flag = CVLS_SUNMAT_FAIL;
                return CVLS_SUNMAT_FAIL;
            }
        }

        /* Compute new Jacobian matrix */
        let jac = cvls_mem_mut(&cv_mem).jac.expect("jac");
        let use_field = cvls_mem_mut(&cv_mem).J_data.is_some();
        let mut j_data = if use_field {
            cvls_mem_mut(&cv_mem).J_data.take()
        } else {
            cv_mem.borrow_mut().cv_user_data.take()
        };
        let retval = jac(t, y, fy, A, &mut j_data, vtemp1, vtemp2, vtemp3);
        if use_field {
            cvls_mem_mut(&cv_mem).J_data = j_data;
        } else {
            cv_mem.borrow_mut().cv_user_data = j_data;
        }
        if retval < 0 {
            cvProcessError(
                Some(&cv_mem),
                CVLS_JACFUNC_UNRECVR,
                line!() as i32,
                "cvLsLinSys",
                file!(),
                MSG_LS_JACFUNC_FAILED,
            );
            cvls_mem_mut(&cv_mem).last_flag = CVLS_JACFUNC_UNRECVR;
            return -1;
        }
        if retval > 0 {
            cvls_mem_mut(&cv_mem).last_flag = CVLS_JACFUNC_RECVR;
            return 1;
        }

        /* Update saved copy of the Jacobian matrix */
        let savedJ = cvls_mem_mut(&cv_mem).savedJ.as_ref().expect("savedJ").clone();
        let retval = SUNMatCopy(A, &savedJ);
        if retval != 0 {
            cvProcessError(
                Some(&cv_mem),
                CVLS_SUNMAT_FAIL,
                line!() as i32,
                "cvLsLinSys",
                file!(),
                MSG_LS_SUNMAT_FAILED,
            );
            cvls_mem_mut(&cv_mem).last_flag = CVLS_SUNMAT_FAIL;
            return CVLS_SUNMAT_FAIL;
        }
    }

    /* Perform linear combination A = I - gamma*J */
    let retval = SUNMatScaleAddI(-gamma, A);
    if retval != 0 {
        cvProcessError(
            Some(&cv_mem),
            CVLS_SUNMAT_FAIL,
            line!() as i32,
            "cvLsLinSys",
            file!(),
            MSG_LS_SUNMAT_FAILED,
        );
        cvls_mem_mut(&cv_mem).last_flag = CVLS_SUNMAT_FAIL;
        return CVLS_SUNMAT_FAIL;
    }

    CVLS_SUCCESS
}

/*-----------------------------------------------------------------
  cvLsInitialize

  This routine performs remaining initializations specific
  to the iterative linear solver interface (and solver itself)
  -----------------------------------------------------------------*/
pub fn cvLsInitialize(cv_mem: &CVodeMem) -> i32 {
    /* access CVLsMem structure */
    if cv_mem.borrow().cv_lmem.is_none() {
        cvProcessError(
            Some(cv_mem),
            CVLS_LMEM_NULL,
            line!() as i32,
            "cvLsInitialize",
            file!(),
            MSG_LS_LMEM_NULL,
        );
        return CVLS_LMEM_NULL;
    }

    /* Test for valid combinations of matrix & Jacobian routines: */
    let A = cvls_mem_mut(cv_mem).A.clone();
    if let Some(A) = &A {
        /* Matrix-based case */

        if cvls_mem_mut(cv_mem).user_linsys {
            /* User-supplied linear system function, reset A_data (just in case) */
            cvls_mem_mut(cv_mem).A_data = None; /* C: A_data = cv_mem->cv_user_data */
        } else {
            /* Internal linear system function, reset pointers (just in case) */
            {
                let mut ls = cvls_mem_mut(cv_mem);
                ls.linsys = Some(cvLsLinSys);
                ls.A_data = Some(Box::new(cv_mem.clone())); /* C: A_data = cv_mem */
            }

            /* Check if an internal or user-supplied Jacobian function is used */
            if cvls_mem_mut(cv_mem).jacDQ {
                /* Internal difference quotient Jacobian. Check that A is dense or band,
                otherwise return an error */
                let mut retval = 0;
                if A.ops.borrow().getid.is_some() {
                    let id = SUNMatGetID(A);
                    if id == SUNMATRIX_DENSE || id == SUNMATRIX_BAND {
                        let mut ls = cvls_mem_mut(cv_mem);
                        ls.jac = Some(cvLsDQJac);
                        ls.J_data = Some(Box::new(cv_mem.clone())); /* C: J_data = cv_mem */
                    } else {
                        retval += 1;
                    }
                } else {
                    retval += 1;
                }
                if retval != 0 {
                    cvProcessError(
                        Some(cv_mem),
                        CVLS_ILL_INPUT,
                        line!() as i32,
                        "cvLsInitialize",
                        file!(),
                        "No Jacobian constructor available for SUNMatrix type",
                    );
                    cvls_mem_mut(cv_mem).last_flag = CVLS_ILL_INPUT;
                    return CVLS_ILL_INPUT;
                }
            } else {
                /* User-supplied Jacobian, reset J_data pointer (just in case) */
                cvls_mem_mut(cv_mem).J_data = None; /* C: J_data = cv_mem->cv_user_data */
            }

            /* Allocate internally saved Jacobian if not already done */
            if cvls_mem_mut(cv_mem).savedJ.is_none() {
                match SUNMatClone(A) {
                    Some(savedJ) => cvls_mem_mut(cv_mem).savedJ = Some(savedJ),
                    None => {
                        cvProcessError(
                            Some(cv_mem),
                            CVLS_MEM_FAIL,
                            line!() as i32,
                            "cvLsInitialize",
                            file!(),
                            MSG_LS_MEM_FAIL,
                        );
                        cvls_mem_mut(cv_mem).last_flag = CVLS_MEM_FAIL;
                        return CVLS_MEM_FAIL;
                    }
                }
            }
        } /* end matrix-based case */
    } else {
        /* Matrix-free case: ensure 'jac' and `linsys` function pointers are NULL */
        let mut ls = cvls_mem_mut(cv_mem);
        ls.jacDQ = SUNFALSE;
        ls.jac = None;
        ls.J_data = None;

        ls.user_linsys = SUNFALSE;
        ls.linsys = None;
        ls.A_data = None;
    }

    /* reset counters */
    let _ = cvLsInitializeCounters(&mut cvls_mem_mut(cv_mem));

    /* Set Jacobian-vector product related fields, based on jtimesDQ */
    if cvls_mem_mut(cv_mem).jtimesDQ {
        let mut ls = cvls_mem_mut(cv_mem);
        ls.jtsetup = None;
        ls.jtimes = Some(cvLsDQJtimes);
        ls.jt_data = Some(Box::new(cv_mem.clone())); /* C: jt_data = cv_mem */
    } else {
        cvls_mem_mut(cv_mem).jt_data = None; /* C: jt_data = cv_mem->cv_user_data */
    }

    /* if A is NULL and psetup is not present, then cvLsSetup does
    not need to be called, so set the lsetup function to NULL */
    let (A_is_none, pset_is_none) = {
        let ls = cvls_mem_mut(cv_mem);
        (ls.A.is_none(), ls.pset.is_none())
    };
    if A_is_none && pset_is_none {
        cv_mem.borrow_mut().cv_lsetup = None;
    }

    /* When using a matrix-embedded linear solver, disable lsetup call and solution scaling */
    let LS = cvls_mem_mut(cv_mem).LS.clone();
    if SUNLinSolGetType(&LS) == SUNLINEARSOLVER_MATRIX_EMBEDDED {
        cv_mem.borrow_mut().cv_lsetup = None;
        cvls_mem_mut(cv_mem).scalesol = SUNFALSE;
    }

    /* Call LS initialize routine, and return result */
    let flag = SUNLinSolInitialize(&LS);
    cvls_mem_mut(cv_mem).last_flag = flag;
    flag
}

pub fn cvLsReInitialize(cv_mem: &CVodeMem) -> i32 {
    /* access CVLsMem structure */
    if cv_mem.borrow().cv_lmem.is_none() {
        cvProcessError(
            Some(cv_mem),
            CVLS_LMEM_NULL,
            line!() as i32,
            "cvLsReInitialize",
            file!(),
            MSG_LS_LMEM_NULL,
        );
        return CVLS_LMEM_NULL;
    }

    /* Initialize counters */
    let _ = cvLsInitializeCounters(&mut cvls_mem_mut(cv_mem));

    CVLS_SUCCESS
}

/*-----------------------------------------------------------------
  cvLsSetup

  This conditionally calls the LS 'setup' routine.

  When using a SUNMatrix object, this determines whether
  to update a Jacobian matrix (or use a stored version), based
  on heuristics regarding previous convergence issues, the number
  of time steps since it was last updated, etc.; it then creates
  the system matrix from this, the 'gamma' factor and the
  identity matrix, A = I-gamma*J.

  This routine then calls the LS 'setup' routine with A.
  -----------------------------------------------------------------*/
#[allow(clippy::too_many_arguments)]
pub fn cvLsSetup(
    cv_mem: &CVodeMem,
    convfail: i32,
    ypred: &N_Vector,
    fpred: &N_Vector,
    jcurPtr: &mut sunbooleantype,
    vtemp1: &N_Vector,
    vtemp2: &N_Vector,
    vtemp3: &N_Vector,
) -> i32 {
    /* access CVLsMem structure */
    if cv_mem.borrow().cv_lmem.is_none() {
        cvProcessError(
            Some(cv_mem),
            CVLS_LMEM_NULL,
            line!() as i32,
            "cvLsSetup",
            file!(),
            MSG_LS_LMEM_NULL,
        );
        return CVLS_LMEM_NULL;
    }

    /* Immediately return when using matrix-embedded linear solver */
    let LS = cvls_mem_mut(cv_mem).LS.clone();
    if SUNLinSolGetType(&LS) == SUNLINEARSOLVER_MATRIX_EMBEDDED {
        cvls_mem_mut(cv_mem).last_flag = CVLS_SUCCESS;
        return CVLS_SUCCESS;
    }

    /* Set CVLs N_Vector pointers to current solution and rhs */
    {
        let mut ls = cvls_mem_mut(cv_mem);
        ls.ycur = Some(ypred.clone());
        ls.fcur = Some(fpred.clone());
    }

    /* Use nst, gamma/gammap, and convfail to set J/P eval. flag jok */
    let (gamma, gammap, nst, first_step_after_resize, tn) = {
        let m = cv_mem.borrow();
        (
            m.cv_gamma,
            m.cv_gammap,
            m.cv_nst,
            m.first_step_after_resize,
            m.cv_tn,
        )
    };
    let dgamma = SUNRabs((gamma / gammap) - ONE);
    {
        let mut ls = cvls_mem_mut(cv_mem);
        ls.jbad = (nst == 0)
            || first_step_after_resize
            || (nst >= ls.nstlj + ls.msbj)
            || ((convfail == CV_FAIL_BAD_J) && (dgamma < ls.dgmax_jbad))
            || (convfail == CV_FAIL_OTHER);
    }

    /* Setup the linear system if necessary */
    let A = cvls_mem_mut(cv_mem).A.clone();
    if let Some(A) = &A {
        /* Update J if appropriate and evaluate A = I - gamma J */
        let (linsys, jbad) = {
            let ls = cvls_mem_mut(cv_mem);
            (ls.linsys.expect("linsys"), ls.jbad)
        };
        let use_field = cvls_mem_mut(cv_mem).A_data.is_some();
        let mut a_data = if use_field {
            cvls_mem_mut(cv_mem).A_data.take()
        } else {
            cv_mem.borrow_mut().cv_user_data.take()
        };
        let retval = linsys(
            tn, ypred, fpred, A, !jbad, jcurPtr, gamma, &mut a_data, vtemp1, vtemp2, vtemp3,
        );
        if use_field {
            cvls_mem_mut(cv_mem).A_data = a_data;
        } else {
            cv_mem.borrow_mut().cv_user_data = a_data;
        }

        /* jcurPtr aliases cv_jcur in C (cvode_nls.c:307): write the linsys
        result through so any pset reached via SUNLinSolSetup observes it */
        cv_mem.borrow_mut().cv_jcur = *jcurPtr;

        /* Update J eval count and step when J was last updated */
        if *jcurPtr {
            let (nst_now, tn_now) = {
                let m = cv_mem.borrow();
                (m.cv_nst, m.cv_tn)
            };
            let mut ls = cvls_mem_mut(cv_mem);
            ls.nje += 1;
            ls.nstlj = nst_now;
            ls.tnlj = tn_now;
        }

        /* Check linsys() return value and return if necessary */
        if retval != CVLS_SUCCESS {
            if cvls_mem_mut(cv_mem).user_linsys {
                if retval < 0 {
                    cvProcessError(
                        Some(cv_mem),
                        CVLS_JACFUNC_UNRECVR,
                        line!() as i32,
                        "cvLsSetup",
                        file!(),
                        MSG_LS_JACFUNC_FAILED,
                    );
                    cvls_mem_mut(cv_mem).last_flag = CVLS_JACFUNC_UNRECVR;
                    return -1;
                } else {
                    cvls_mem_mut(cv_mem).last_flag = CVLS_JACFUNC_RECVR;
                    return 1;
                }
            } else {
                return retval;
            }
        }
    } else {
        /* Matrix-free case, set jcur to jbad */
        *jcurPtr = cvls_mem_mut(cv_mem).jbad;
        /* write through the C alias (jcurPtr == &cv_jcur) so cvLsPSetup
        passes the fresh suggestion to the user psetup */
        cv_mem.borrow_mut().cv_jcur = *jcurPtr;
    }

    /* Call LS setup routine -- the LS may call cvLsPSetup, who will
    pass the heuristic suggestions above to the user code(s) */
    let flag = SUNLinSolSetup(&LS, A.as_ref());
    cvls_mem_mut(cv_mem).last_flag = flag;

    /* re-read through the C alias: a user psetup reached via
    SUNLinSolSetup -> cvLsPSetup may have overridden jcur */
    *jcurPtr = cv_mem.borrow().cv_jcur;

    /* If Matrix-free, update heuristics flags */
    if A.is_none() {
        /* If user set jcur to SUNTRUE, increment npe and save nst value */
        if *jcurPtr {
            let (nst_now, tn_now) = {
                let m = cv_mem.borrow();
                (m.cv_nst, m.cv_tn)
            };
            let mut ls = cvls_mem_mut(cv_mem);
            ls.npe += 1;
            ls.nstlj = nst_now;
            ls.tnlj = tn_now;
        }

        /* Update jcur flag if we suggested an update */
        if cvls_mem_mut(cv_mem).jbad {
            *jcurPtr = SUNTRUE;
            /* write through the C alias */
            cv_mem.borrow_mut().cv_jcur = SUNTRUE;
        }
    }

    flag
}

/*-----------------------------------------------------------------
  cvLsSolve

  This routine interfaces between CVode and the generic
  SUNLinearSolver object LS, by setting the appropriate tolerance
  and scaling vectors, calling the solver, and accumulating
  statistics from the solve for use/reporting by CVode.
  -----------------------------------------------------------------*/
pub fn cvLsSolve(
    cv_mem: &CVodeMem,
    b: &N_Vector,
    weight: &N_Vector,
    ynow: &N_Vector,
    fnow: &N_Vector,
) -> i32 {
    /* access CVLsMem structure */
    if cv_mem.borrow().cv_lmem.is_none() {
        cvProcessError(
            Some(cv_mem),
            CVLS_LMEM_NULL,
            line!() as i32,
            "cvLsSolve",
            file!(),
            MSG_LS_LMEM_NULL,
        );
        return CVLS_LMEM_NULL;
    }

    /* get current nonlinear solver iteration */
    let NLS = cv_mem.borrow().NLS.as_ref().expect("NLS").clone();
    let mut curiter: i32 = 0;
    let mut retval: i32 = SUNNonlinSolGetCurIter(&NLS, &mut curiter);
    let _ = retval; /* C stores this value in `retval` but never reads it */

    let (iterative, eplifac, nrmfac) = {
        let ls = cvls_mem_mut(cv_mem);
        (ls.iterative, ls.eplifac, ls.nrmfac)
    };

    /* If the linear solver is iterative:
    test norm(b), if small, return x = 0 or x = b;
    set linear solver tolerance (in left/right scaled 2-norm) */
    let mut delta: sunrealtype;
    if iterative {
        let deltar = eplifac * cv_mem.borrow().cv_tq[4];
        let bnorm = N_VWrmsNorm(b, weight);

        if bnorm <= deltar {
            if curiter > 0 {
                N_VConst(ZERO, b);
            }
            cvls_mem_mut(cv_mem).last_flag = CVLS_SUCCESS;
            return CVLS_SUCCESS;
        }
        /* Adjust tolerance for 2-norm */
        delta = deltar * nrmfac;
    } else {
        delta = ZERO;
    }

    /* Set vectors ycur and fcur for use by the Atimes and Psolve
    interface routines */
    {
        let mut ls = cvls_mem_mut(cv_mem);
        ls.ycur = Some(ynow.clone());
        ls.fcur = Some(fnow.clone());
    }

    let LS = cvls_mem_mut(cv_mem).LS.clone();
    let x = cvls_mem_mut(cv_mem).x.as_ref().expect("x").clone();

    /* Set scaling vectors for LS to use (if applicable) */
    if LS.ops.borrow().setscalingvectors.is_some() {
        retval = SUNLinSolSetScalingVectors(&LS, Some(weight), Some(weight));
        if retval != SUN_SUCCESS {
            cvProcessError(
                Some(cv_mem),
                CVLS_SUNLS_FAIL,
                line!() as i32,
                "cvLsSolve",
                file!(),
                "Error in calling SUNLinSolSetScalingVectors",
            );
            cvls_mem_mut(cv_mem).last_flag = CVLS_SUNLS_FAIL;
            return CVLS_SUNLS_FAIL;
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
        So we compute w_mean = ||w||_RMS = ||w||_2 and scale the desired tolerance accordingly. */
    } else if iterative {
        N_VConst(ONE, &x);
        let w_mean = N_VWrmsNorm(weight, &x);
        delta /= w_mean;
    }

    /* Set initial guess x = 0 to LS */
    N_VConst(ZERO, &x);

    /* Set zero initial guess flag */
    retval = SUNLinSolSetZeroGuess(&LS, SUNTRUE);
    if retval != SUN_SUCCESS {
        return -1;
    }

    /* C stores the previous nps value in nps_inc here (logging only —
    omitted at SUNDIALS_LOGGING_LEVEL 2) */

    /* If a user-provided jtsetup routine is supplied, call that here */
    let jtsetup = cvls_mem_mut(cv_mem).jtsetup;
    if let Some(jtsetup) = jtsetup {
        let tn = cv_mem.borrow().cv_tn;
        let use_field = cvls_mem_mut(cv_mem).jt_data.is_some();
        let mut jt_data = if use_field {
            cvls_mem_mut(cv_mem).jt_data.take()
        } else {
            cv_mem.borrow_mut().cv_user_data.take()
        };
        let last_flag = jtsetup(tn, ynow, fnow, &mut jt_data);
        if use_field {
            cvls_mem_mut(cv_mem).jt_data = jt_data;
        } else {
            cv_mem.borrow_mut().cv_user_data = jt_data;
        }
        {
            let mut ls = cvls_mem_mut(cv_mem);
            ls.last_flag = last_flag;
            ls.njtsetup += 1;
        }
        if last_flag != 0 {
            /* C passes `retval` (the SetZeroGuess result, SUN_SUCCESS
            here) as the error code — preserved verbatim */
            cvProcessError(
                Some(cv_mem),
                retval,
                line!() as i32,
                "cvLsSolve",
                file!(),
                MSG_LS_JTSETUP_FAILED,
            );
            return last_flag;
        }
    }

    /* Call solver, and copy x to b */
    let A = cvls_mem_mut(cv_mem).A.clone();
    retval = SUNLinSolSolve(&LS, A.as_ref(), &x, b, delta);
    N_VScale(ONE, &x, b);

    /* If using a direct or matrix-iterative solver, BDF method, and gamma has changed,
    scale the correction to account for change in gamma */
    let scalesol = cvls_mem_mut(cv_mem).scalesol;
    let gamrat = cv_mem.borrow().cv_gamrat;
    if scalesol && gamrat != ONE {
        N_VScale(TWO / (ONE + gamrat), b, b);
    }

    /* Retrieve statistics from iterative linear solvers */
    let mut nli_inc: i32 = 0;
    if iterative {
        if LS.ops.borrow().resnorm.is_some() {
            /* resnorm: logging only at level 2 — call kept, value unused */
            let _ = SUNLinSolResNorm(&LS);
        }
        if LS.ops.borrow().numiters.is_some() {
            nli_inc = SUNLinSolNumIters(&LS);
        }
    }

    /* Increment counters nli and ncfl */
    {
        let mut ls = cvls_mem_mut(cv_mem);
        ls.nli += nli_inc as i64;
        if retval != SUN_SUCCESS {
            ls.ncfl += 1;
        }
    }

    /* Interpret solver return value  */
    cvls_mem_mut(cv_mem).last_flag = retval;

    match retval {
        SUN_SUCCESS => 0,
        SUNLS_RES_REDUCED => {
            /* allow reduction but not solution on first Newton iteration,
            otherwise return with a recoverable failure */
            if curiter == 0 {
                0
            } else {
                1
            }
        }
        SUNLS_CONV_FAIL | SUNLS_ATIMES_FAIL_REC | SUNLS_PSOLVE_FAIL_REC
        | SUNLS_PACKAGE_FAIL_REC | SUNLS_QRFACT_FAIL | SUNLS_LUFACT_FAIL => 1,
        SUN_ERR_ARG_CORRUPT | SUN_ERR_ARG_INCOMPATIBLE | SUN_ERR_MEM_FAIL | SUNLS_GS_FAIL
        | SUNLS_QRSOL_FAIL => -1,
        SUN_ERR_EXT_FAIL => {
            cvProcessError(
                Some(cv_mem),
                SUN_ERR_EXT_FAIL,
                line!() as i32,
                "cvLsSolve",
                file!(),
                "Failure in SUNLinSol external package",
            );
            -1
        }
        SUNLS_ATIMES_FAIL_UNREC => {
            cvProcessError(
                Some(cv_mem),
                SUNLS_ATIMES_FAIL_UNREC,
                line!() as i32,
                "cvLsSolve",
                file!(),
                MSG_LS_JTIMES_FAILED,
            );
            -1
        }
        SUNLS_PSOLVE_FAIL_UNREC => {
            cvProcessError(
                Some(cv_mem),
                SUNLS_PSOLVE_FAIL_UNREC,
                line!() as i32,
                "cvLsSolve",
                file!(),
                MSG_LS_PSOLVE_FAILED,
            );
            -1
        }
        _ => {
            cvProcessError(
                Some(cv_mem),
                retval,
                line!() as i32,
                "cvLsSolve",
                file!(),
                "Unrecognized error return value from SUNLinSolSolve",
            );
            -1
        }
    }
}

/*-----------------------------------------------------------------
  cvLsFree

  This routine frees memory associates with the CVLs system
  solver interface.
  -----------------------------------------------------------------*/
pub fn cvLsFree(cv_mem: &CVodeMem) -> i32 {
    /* NULL CVodeMem check: handled by type system */

    /* Return immediately if CVLsMem is NULL */
    if cv_mem.borrow().cv_lmem.is_none() {
        return CVLS_SUCCESS;
    }

    {
        let mut ls = cvls_mem_mut(cv_mem);

        /* Free N_Vector memory */
        if let Some(ytemp) = ls.ytemp.take() {
            N_VDestroy(ytemp);
        }
        if let Some(x) = ls.x.take() {
            N_VDestroy(x);
        }

        /* Free savedJ memory */
        if let Some(savedJ) = ls.savedJ.take() {
            SUNMatDestroy(savedJ);
        }

        /* Nullify other N_Vector pointers */
        ls.ycur = None;
        ls.fcur = None;

        /* Nullify other SUNMatrix pointer */
        ls.A = None;
    }

    /* Free preconditioner memory (if applicable) */
    let pfree = cvls_mem_mut(cv_mem).pfree;
    if let Some(pfree) = pfree {
        pfree(cv_mem);
    }

    /* free CVLs interface structure */
    cv_mem.borrow_mut().cv_lmem = None;

    CVLS_SUCCESS
}

/*-----------------------------------------------------------------
  cvLsInitializeCounters

  This routine resets all counters from an CVLsMem structure.
  -----------------------------------------------------------------*/
pub fn cvLsInitializeCounters(cvls_mem: &mut CVLsMemRec) -> i32 {
    cvls_mem.nje = 0;
    cvls_mem.nfeDQ = 0;
    cvls_mem.nstlj = 0;
    cvls_mem.npe = 0;
    cvls_mem.nli = 0;
    cvls_mem.nps = 0;
    cvls_mem.ncfl = 0;
    cvls_mem.njtsetup = 0;
    cvls_mem.njtimes = 0;
    0
}

/*---------------------------------------------------------------
  cvLs_AccessLMem

  Public-API flavor of the C helper: with `&CVodeMem` the NULL-mem
  check vanishes (handled by the type system); this verifies that
  linear solver memory is attached. Callers then use
  `cvls_mem_mut` for field access.
  ---------------------------------------------------------------*/
pub fn cvLs_AccessLMem(cvode_mem: &CVodeMem, fname: &str) -> i32 {
    /* NULL-mem check: handled by type system */
    if cvode_mem.borrow().cv_lmem.is_none() {
        cvProcessError(
            Some(cvode_mem),
            CVLS_LMEM_NULL,
            line!() as i32,
            fname,
            file!(),
            MSG_LS_LMEM_NULL,
        );
        return CVLS_LMEM_NULL;
    }
    CVLS_SUCCESS
}

/*---------------------------------------------------------------
  cvLs_AccessLMemToken

  Callback flavor of the C `cvLs_AccessLMem`: the C `void*
  cvode_mem` argument arrives as a data token holding a `CVodeMem`
  clone. A missing/foreign token maps to the C NULL check.
  ---------------------------------------------------------------*/
pub fn cvLs_AccessLMemToken(
    cvode_mem: &Option<Box<dyn Any>>,
    fname: &str,
) -> Result<CVodeMem, i32> {
    let cv_mem = match cvode_mem
        .as_ref()
        .and_then(|b| b.downcast_ref::<CVodeMem>())
        .cloned()
    {
        Some(m) => m,
        None => {
            cvProcessError(
                None,
                CVLS_MEM_NULL,
                line!() as i32,
                fname,
                file!(),
                MSG_LS_CVMEM_NULL,
            );
            return Err(CVLS_MEM_NULL);
        }
    };
    if cv_mem.borrow().cv_lmem.is_none() {
        cvProcessError(
            Some(&cv_mem),
            CVLS_LMEM_NULL,
            line!() as i32,
            fname,
            file!(),
            MSG_LS_LMEM_NULL,
        );
        return Err(CVLS_LMEM_NULL);
    }
    Ok(cv_mem)
}

/*---------------------------------------------------------------
  EOF
  ---------------------------------------------------------------*/
