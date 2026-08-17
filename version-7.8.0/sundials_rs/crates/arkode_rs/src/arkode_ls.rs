//! Port of `src/arkode/arkode_ls.c` (+ `src/arkode/arkode_ls_impl.h` and
//! `include/arkode/arkode_ls.h` folded).
//!
//! ARKODE's linear solver interface (ARKLS). Two independent records live
//! here: `ARKLsMemRec` for the system linear solver `A = M - gamma*J`, and
//! `ARKLsMassMemRec` for the non-identity mass-matrix solver (no CVODE
//! analogue). Both are stored BY VALUE in `ark_mem.ark_lmem` /
//! `ark_mem.ark_mass_mem` (see `arkode_impl.rs`) and reached through
//! [`arkls_mem_mut`] / [`arkls_mass_mem_mut`]; the C
//! `step_getlinmem` / `step_getmassmem` hooks degrade to *presence probes*
//! returning `sunbooleantype`, so every C `if (ark_mem->step_getmassmem)`
//! guard translates unchanged.
//!
//! Data-token model (C `void*` fields `J_data` / `P_data` / `Jt_data` /
//! `A_data` / `M_data` / `mt_data`): in C each field holds either `ark_mem`
//! (internal routine) or `ark_mem->user_data` (user routine). Here the
//! field is `Option<Box<dyn Any>>`: `Some(box)` is a module-owned token (an
//! `ARKodeMem` clone for the internal ARKLS routines, or whatever an
//! internal preconditioner module stored), while `None` means "pass the
//! integrator's `user_data`" — the invoker `Option::take`s the
//! corresponding box around the callback and restores it on EVERY path.
//! This reproduces the C pointer aliasing without double ownership; the
//! only divergence is that a C snapshot of a *stale* `user_data` (user data
//! replaced after the Set* call) cannot occur — the current `user_data` is
//! always passed (accepted deviation class 6, ARCHITECTURE.md).
//!
//! `mt_data` is the ONE exception: C never assigns `ark_mem->user_data` to
//! it (`ARKodeSetMassLinearSolver` sets it to NULL, `ARKodeSetMassTimes`
//! stores the caller's `mtimes_data` verbatim, and `arkLSSetMassUserData`
//! deliberately leaves it alone — `arkode_ls.c:445/:1824/:2300`). So for
//! this field `None` is C's NULL and is passed through to
//! `mtimes`/`mtsetup` as-is; it never falls back to `user_data`.
//!
//! The `jcur` seam: `step_getgammas` hands out a clone of the stepper's
//! shared [`ARKJcurPtr`] cell, and `arkLsSetup` receives the SAME cell as
//! `jcurPtr`. A user/BANDPRE/BBDPRE `psetup` reached re-entrantly through
//! `SUNLinSolSetup` -> [`arkLsPSetup`] writes through it and `arkLsSetup`
//! reads the result afterwards (`if (*jcurPtr)` -> `npe++`/`nstlj`/`tnlj`).
//! The user-facing `ARKLsPrecSetupFn` / `ARKLsLinSysFn` keep
//! `jcurPtr: &mut sunbooleantype` (identical in shape to CVODE, so example
//! code is unchanged); each call site copies the cell out, calls, and
//! writes the result back into the cell before any bookkeeping that
//! depends on it.
//!
//! Granular borrow discipline: no `ark_mem` borrow (including an
//! `arkls_mem_mut` / `arkls_mass_mem_mut` guard, which IS a borrow of the
//! mem) is ever held across `arkProcessError`, a user callback, an
//! N_Vector op on a user-visible vector, a `step_*` call, or a
//! SUNLinearSolver / SUNMatrix call.

use std::any::Any;
use std::cell::{Cell, RefMut};

use crate::arkode::{arkAllocVec, arkFreeVec};
use crate::arkode_impl::*;
use sundials_core::sundials_errors::{
    SUN_ERR_ARG_CORRUPT, SUN_ERR_ARG_INCOMPATIBLE, SUN_ERR_EXT_FAIL, SUN_ERR_MEM_FAIL, SUN_SUCCESS,
};
use sundials_core::sundials_linearsolver::*;
use sundials_core::sundials_math::{SUNMAX, SUNMIN, SUNRabs, SUNRsqrt};
use sundials_core::sundials_matrix::*;
use sundials_core::sundials_nvector::*;
use sundials_core::sundials_types::*;
use sundials_core::sunmatrix_band::{
    SM_COLUMN_ELEMENT_IDX, SUNBandMatrix_Column, SUNBandMatrix_Columns,
    SUNBandMatrix_LowerBandwidth, SUNBandMatrix_StoredUpperBandwidth, SUNBandMatrix_UpperBandwidth,
};
use sundials_core::sunmatrix_dense::{SUNDenseMatrix_Column, SUNDenseMatrix_Columns};

/* constants */
const MIN_INC_MULT: sunrealtype = 1000.0;
const PT25: sunrealtype = 0.25;
/* ZERO, ONE and TWO come from arkode_impl (identical values to the
`arkode_ls.c` `#define`s; `TWO` is only defined in `arkode_impl.h`). */

/*=================================================================
  ARKLS Constants (include/arkode/arkode_ls.h)
  =================================================================*/

pub const ARKLS_SUCCESS: i32 = 0;
pub const ARKLS_MEM_NULL: i32 = -1;
pub const ARKLS_LMEM_NULL: i32 = -2;
pub const ARKLS_ILL_INPUT: i32 = -3;
pub const ARKLS_MEM_FAIL: i32 = -4;
pub const ARKLS_PMEM_NULL: i32 = -5;
pub const ARKLS_MASSMEM_NULL: i32 = -6;
pub const ARKLS_JACFUNC_UNRECVR: i32 = -7;
pub const ARKLS_JACFUNC_RECVR: i32 = -8;
pub const ARKLS_MASSFUNC_UNRECVR: i32 = -9;
pub const ARKLS_MASSFUNC_RECVR: i32 = -10;
pub const ARKLS_SUNMAT_FAIL: i32 = -11;
pub const ARKLS_SUNLS_FAIL: i32 = -12;

/*---------------------------------------------------------------
  ARKLS solver constants (arkode_ls_impl.h):

  ARKLS_MSBJ   default maximum number of steps between Jacobian /
               preconditioner evaluations

  ARKLS_EPLIN  default value for factor by which the tolerance
               on the nonlinear iteration is multiplied to get
               a tolerance on the linear iteration
  ---------------------------------------------------------------*/
pub const ARKLS_MSBJ: i64 = 51;
pub const ARKLS_EPLIN: sunrealtype = 0.05;

/*---------------------------------------------------------------
  Error Messages (arkode_ls_impl.h)
  ---------------------------------------------------------------*/

pub const MSG_LS_ARKMEM_NULL: &str = "Integrator memory is NULL.";
pub const MSG_LS_MEM_FAIL: &str = "A memory request failed.";
pub const MSG_LS_BAD_NVECTOR: &str = "A required vector operation is not implemented.";
pub const MSG_LS_BAD_LSTYPE: &str = "Incompatible linear solver type.";
pub const MSG_LS_LMEM_NULL: &str = "Linear solver memory is NULL.";
pub const MSG_LS_MASSMEM_NULL: &str = "Mass matrix solver memory is NULL.";
pub const MSG_LS_BAD_SIZES: &str =
    "Illegal bandwidth parameter(s). Must have 0 <=  ml, mu <= N-1.";

pub const MSG_LS_PSET_FAILED: &str =
    "The preconditioner setup routine failed in an unrecoverable manner.";
pub const MSG_LS_PSOLVE_FAILED: &str =
    "The preconditioner solve routine failed in an unrecoverable manner.";
pub const MSG_LS_JTSETUP_FAILED: &str =
    "The Jacobian x vector setup routine failed in an unrecoverable manner.";
pub const MSG_LS_JTIMES_FAILED: &str =
    "The Jacobian x vector routine failed in an unrecoverable manner.";
pub const MSG_LS_MTSETUP_FAILED: &str =
    "The mass matrix x vector setup routine failed in an unrecoverable manner.";
pub const MSG_LS_MTIMES_FAILED: &str =
    "The mass matrix x vector routine failed in an unrecoverable manner.";

pub const MSG_LS_JACFUNC_FAILED: &str = "The Jacobian routine failed in an unrecoverable manner.";
pub const MSG_LS_MASSFUNC_FAILED: &str =
    "The mass matrix routine failed in an unrecoverable manner.";
pub const MSG_LS_SUNMAT_FAILED: &str = "A SUNMatrix routine failed in an unrecoverable manner.";

/*=================================================================
  ARKLS user-supplied function prototypes (include/arkode/arkode_ls.h)
  =================================================================*/

pub type ARKLsJacFn = fn(
    t: sunrealtype,
    y: &N_Vector,
    fy: &N_Vector,
    Jac: &SUNMatrix,
    user_data: &mut Option<Box<dyn Any>>,
    tmp1: &N_Vector,
    tmp2: &N_Vector,
    tmp3: &N_Vector,
) -> i32;

pub type ARKLsMassFn = fn(
    t: sunrealtype,
    M: &SUNMatrix,
    user_data: &mut Option<Box<dyn Any>>,
    tmp1: &N_Vector,
    tmp2: &N_Vector,
    tmp3: &N_Vector,
) -> i32;

pub type ARKLsPrecSetupFn = fn(
    t: sunrealtype,
    y: &N_Vector,
    fy: &N_Vector,
    jok: sunbooleantype,
    jcurPtr: &mut sunbooleantype,
    gamma: sunrealtype,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32;

pub type ARKLsPrecSolveFn = fn(
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

pub type ARKLsJacTimesSetupFn = fn(
    t: sunrealtype,
    y: &N_Vector,
    fy: &N_Vector,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32;

pub type ARKLsJacTimesVecFn = fn(
    v: &N_Vector,
    Jv: &N_Vector,
    t: sunrealtype,
    y: &N_Vector,
    fy: &N_Vector,
    user_data: &mut Option<Box<dyn Any>>,
    tmp: &N_Vector,
) -> i32;

/// C `SUNMatrix M` is NULL when there is no (or a matrix-free) mass
/// matrix, hence `Option<&SUNMatrix>`; `A` is only ever non-NULL at the
/// single call site (`arkLsSetup` guards on `arkls_mem->A != NULL`).
pub type ARKLsLinSysFn = fn(
    t: sunrealtype,
    y: &N_Vector,
    fy: &N_Vector,
    A: &SUNMatrix,
    M: Option<&SUNMatrix>,
    jok: sunbooleantype,
    jcur: &mut sunbooleantype,
    gamma: sunrealtype,
    user_data: &mut Option<Box<dyn Any>>,
    tmp1: &N_Vector,
    tmp2: &N_Vector,
    tmp3: &N_Vector,
) -> i32;

pub type ARKLsMassTimesSetupFn =
    fn(t: sunrealtype, mtimes_data: &mut Option<Box<dyn Any>>) -> i32;

pub type ARKLsMassTimesVecFn = fn(
    v: &N_Vector,
    Mv: &N_Vector,
    t: sunrealtype,
    mtimes_data: &mut Option<Box<dyn Any>>,
) -> i32;

pub type ARKLsMassPrecSetupFn =
    fn(t: sunrealtype, user_data: &mut Option<Box<dyn Any>>) -> i32;

pub type ARKLsMassPrecSolveFn = fn(
    t: sunrealtype,
    r: &N_Vector,
    z: &N_Vector,
    delta: sunrealtype,
    lr: i32,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32;

/*---------------------------------------------------------------
  Types: ARKLsMemRec, ARKLsMem (arkode_ls_impl.h)
  ---------------------------------------------------------------*/
pub struct ARKLsMemRec {
    /* Linear solver type information */
    pub iterative: sunbooleantype,   /* is the solver iterative?    */
    pub matrixbased: sunbooleantype, /* is a matrix structure used? */

    /* Jacobian construction & storage */
    pub jacDQ: sunbooleantype,   /* SUNTRUE if using internal DQ Jacobian approx. */
    pub jac: Option<ARKLsJacFn>, /* Jacobian routine to be called                 */
    /* C `J_data`: `Some` = module-owned token (an `ARKodeMem` clone for the
    internal DQ routine); `None` = pass `ark_mem.user_data` at call time. */
    pub J_data: Option<Box<dyn Any>>,
    pub jbad: sunbooleantype, /* heuristic suggestion for pset                 */

    /* Matrix-based solver, scale solution to account for change in gamma */
    pub scalesol: sunbooleantype,

    /* Iterative solver tolerance */
    pub eplifac: sunrealtype, /* nonlinear -> linear tol scaling factor        */
    pub nrmfac: sunrealtype,  /* integrator -> LS norm conversion factor       */

    /* Linear solver, matrix and vector objects/pointers */
    pub LS: SUNLinearSolver,       /* generic linear solver object                  */
    pub A: Option<SUNMatrix>,      /* A = M - gamma * df/dy                         */
    pub savedJ: Option<SUNMatrix>, /* savedJ = old Jacobian                         */
    pub ytemp: Option<N_Vector>,   /* temp vector passed to jtimes and psolve       */
    pub x: Option<N_Vector>,       /* solution vector used by SUNLinearSolver       */
    pub ycur: Option<N_Vector>,    /* ptr to current y vector in ARKLs solve        */
    pub fcur: Option<N_Vector>,    /* ptr to current fcur = fI(tcur, ycur)          */

    /* Statistics and associated parameters */
    pub msbj: i64,         /* max num steps between jac/pset calls         */
    pub tcur: sunrealtype, /* 'time' for current ARKLs solve               */
    pub nje: i64,          /* no. of calls to jac                          */
    pub nfeDQ: i64,        /* no. of calls to f due to DQ Jacobian or J*v
                           approximations                               */
    pub nstlj: i64,        /* value of nst at the last jac/pset call       */
    pub npe: i64,          /* npe = total number of pset calls             */
    pub nli: i64,          /* nli = total number of linear iterations      */
    pub nps: i64,          /* nps = total number of psolve calls           */
    pub ncfl: i64,         /* ncfl = total number of convergence failures  */
    pub njtsetup: i64,     /* njtsetup = total number of calls to jtsetup  */
    pub njtimes: i64,      /* njtimes = total number of calls to jtimes    */
    pub tnlj: sunrealtype, /* tnlj = t_n at last jac/pset call             */

    /* Preconditioner computation
      (a) user-provided:
          - P_data == user_data (here: `None` = pass `ark_mem.user_data`)
          - pfree == NULL (the user dealocates memory for user_data)
      (b) internal preconditioner module
          - P_data == module token (`Some`)
          - pfree == set by the prec. module and called in ARKodeFree  */
    pub pset: Option<ARKLsPrecSetupFn>,
    pub psolve: Option<ARKLsPrecSolveFn>,
    pub pfree: Option<fn(ark_mem: &ARKodeMem) -> i32>,
    pub P_data: Option<Box<dyn Any>>,

    /* Jacobian times vector computation
      (a) jtimes function provided by the user:
          - Jt_data == user_data (here: `None`)
          - jtimesDQ == SUNFALSE
      (b) internal jtimes
          - Jt_data == arkode_mem token (`Some`)
          - jtimesDQ == SUNTRUE   */
    pub jtimesDQ: sunbooleantype,
    pub jtsetup: Option<ARKLsJacTimesSetupFn>,
    pub jtimes: Option<ARKLsJacTimesVecFn>,
    pub Jt_f: Option<ARKRhsFn>,
    pub Jt_data: Option<Box<dyn Any>>,

    /* Linear system setup function
     * (a) user-provided linsys function:
     *     - user_linsys = SUNTRUE
     *     - A_data      = user_data (here: `None`)
     * (b) internal linsys function:
     *     - user_linsys = SUNFALSE
     *     - A_data      = arkode_mem token (`Some`) */
    pub user_linsys: sunbooleantype,
    pub linsys: Option<ARKLsLinSysFn>,
    pub A_data: Option<Box<dyn Any>>,

    pub last_flag: i32, /* last error flag returned by any function */
}

pub type ARKLsMem = Box<ARKLsMemRec>;

/*---------------------------------------------------------------
  Types: ARKLsMassMemRec, ARKLsMassMem (arkode_ls_impl.h)
  ---------------------------------------------------------------*/
pub struct ARKLsMassMemRec {
    /* Linear solver type information */
    pub iterative: sunbooleantype,   /* is the solver iterative?    */
    pub matrixbased: sunbooleantype, /* is a matrix structure used? */

    /* Mass matrix construction & storage */
    pub mass: Option<ARKLsMassFn>, /* user-provided mass matrix routine to call   */
    pub M: Option<SUNMatrix>,      /* mass matrix structure                       */
    pub M_lu: Option<SUNMatrix>,   /* mass matrix structure for LU decomposition  */
    /* C `M_data`: always `ark_mem->user_data` wherever it is read
    (`ARKodeSetMassFn` sets it, and `mass` is unusable otherwise), so
    `None` = pass `ark_mem.user_data` at call time. */
    pub M_data: Option<Box<dyn Any>>,

    /* Iterative solver tolerance */
    pub eplifac: sunrealtype, /* nonlinear -> linear tol scaling factor      */
    pub nrmfac: sunrealtype,  /* integrator -> LS norm conversion factor     */

    /* Statistics and associated parameters */
    pub time_dependent: sunbooleantype, /* flag whether M depends on t        */
    pub msetuptime: sunrealtype,        /* "t" value at last msetup call      */
    pub nmsetups: i64,                  /* total # mass matrix-solver setups  */
    pub nmsolves: i64,                  /* total # mass matrix-solver solves  */
    pub nmtsetup: i64,                  /* total # calls to mtsetup           */
    pub nmtimes: i64,                   /* total # calls to mtimes            */
    pub nmvsetup: i64,                  /* total # calls to matvec setup      */
    pub npe: i64,                       /* total # pset calls                 */
    pub nli: i64,                       /* total # linear iterations          */
    pub nps: i64,                       /* total # psolve calls               */
    pub ncfl: i64,                      /* total # convergence failures       */

    /* Linear solver, matrix and vector objects/pointers */
    pub LS: SUNLinearSolver,    /* generic linear solver object                */
    pub x: Option<N_Vector>,    /* solution vector used by SUNLinearSolver     */
    pub ycur: Option<N_Vector>, /* ptr to ARKODE current y vector              */

    /* Preconditioner computation
      (a) user-provided:
          - P_data == user_data (here: `None` = pass `ark_mem.user_data`)
          - pfree == NULL (the user dealocates memory for user_data)
      (b) internal preconditioner module
          - P_data == module token (`Some`)
          - pfree == set by the prec. module and called in ARKodeFree  */
    pub pset: Option<ARKLsMassPrecSetupFn>,
    pub psolve: Option<ARKLsMassPrecSolveFn>,
    pub pfree: Option<fn(ark_mem: &ARKodeMem) -> i32>,
    pub P_data: Option<Box<dyn Any>>,

    /* Mass matrix times vector setup and product routines, data.
    C never re-points `mt_data` at `user_data` (`arkLSSetMassUserData`
    deliberately leaves it alone), so `Some(box)` is the caller's own
    token and `None` is C's NULL — unlike the other data-token fields this
    one is passed to the callbacks unchanged and never substitutes
    `ark_mem.user_data`. */
    pub mtsetup: Option<ARKLsMassTimesSetupFn>,
    pub mtimes: Option<ARKLsMassTimesVecFn>,
    pub mt_data: Option<Box<dyn Any>>,

    pub last_flag: i32, /* last error flag returned by any function    */
}

pub type ARKLsMassMem = Box<ARKLsMassMemRec>;

/// Downcast helper: view `ark_mem.ark_lmem` as the ARKLS system memory
/// record. Panics if no linear solver memory is attached or it is not an
/// ARKLS record (the C code would blindly cast the `void*` — UB maps to a
/// deterministic panic). The guard IS a borrow of the mem: NEVER hold it
/// across `arkProcessError`, a callback, an N_Vector / SUNMatrix /
/// SUNLinearSolver op, a `step_*` call, or another borrow of the same mem.
pub fn arkls_mem_mut(ark_mem: &ARKodeMem) -> RefMut<'_, ARKLsMemRec> {
    RefMut::map(ark_mem.borrow_mut(), |m| {
        m.ark_lmem
            .as_mut()
            .expect("ark_lmem set")
            .downcast_mut::<ARKLsMemRec>()
            .expect("ARKLS linear solver memory")
    })
}

/// Downcast helper: view `ark_mem.ark_mass_mem` as the ARKLS mass-matrix
/// memory record. Same panic and borrow rules as [`arkls_mem_mut`].
pub fn arkls_mass_mem_mut(ark_mem: &ARKodeMem) -> RefMut<'_, ARKLsMassMemRec> {
    RefMut::map(ark_mem.borrow_mut(), |m| {
        m.ark_mass_mem
            .as_mut()
            .expect("ark_mass_mem set")
            .downcast_mut::<ARKLsMassMemRec>()
            .expect("ARKLS mass matrix solver memory")
    })
}

/*===============================================================
  Exported routines
  ===============================================================*/

/*---------------------------------------------------------------
  ARKodeSetLinearSolver specifies the linear solver.
  ---------------------------------------------------------------*/
pub fn ARKodeSetLinearSolver(
    arkode_mem: &ARKodeMem,
    LS: &SUNLinearSolver,
    A: Option<&SUNMatrix>,
) -> i32 {
    /* NULL arkode_mem check: handled by the type system */
    let ark_mem = arkode_mem;

    /* Guard against use for time steppers that do not need an algebraic solver */
    if !ark_mem.borrow().step_supports_implicit {
        arkProcessError(
            Some(ark_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetLinearSolver",
            file!(),
            "time-stepping module does not require an algebraic solver",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    /* NULL LS check: handled by the type system */

    /* Test if solver is compatible with LS interface */
    {
        let ops = LS.ops.borrow();
        if ops.gettype.is_none() || ops.solve.is_none() {
            drop(ops);
            arkProcessError(
                Some(ark_mem),
                ARKLS_ILL_INPUT,
                line!() as i32,
                "ARKodeSetLinearSolver",
                file!(),
                "LS object is missing a required operation",
            );
            return ARKLS_ILL_INPUT;
        }
    }

    /* Retrieve the LS type */
    let LSType = SUNLinSolGetType(LS);

    /* Set flags based on LS type */
    let iterative = LSType != SUNLINEARSOLVER_DIRECT;
    let matrixbased =
        (LSType != SUNLINEARSOLVER_ITERATIVE) && (LSType != SUNLINEARSOLVER_MATRIX_EMBEDDED);

    /* Test if vector is compatible with LS interface */
    let tempv1 = ark_mem
        .borrow()
        .tempv1
        .as_ref()
        .expect("tempv1") /* C dereferences unconditionally (UB if unset) */
        .clone();
    {
        let ops = tempv1.ops.borrow();
        if ops.nvconst.is_none() || ops.nvwrmsnorm.is_none() {
            drop(ops);
            arkProcessError(
                Some(ark_mem),
                ARKLS_ILL_INPUT,
                line!() as i32,
                "ARKodeSetLinearSolver",
                file!(),
                MSG_LS_BAD_NVECTOR,
            );
            return ARKLS_ILL_INPUT;
        }
    }

    /* Ensure that A is NULL when LS is matrix-embedded */
    if (LSType == SUNLINEARSOLVER_MATRIX_EMBEDDED) && A.is_some() {
        arkProcessError(
            Some(ark_mem),
            ARKLS_ILL_INPUT,
            line!() as i32,
            "ARKodeSetLinearSolver",
            file!(),
            "Incompatible inputs: matrix-embedded LS requires NULL matrix",
        );
        return ARKLS_ILL_INPUT;
    }

    /* Check for compatible LS type, matrix and "atimes" support */
    if iterative {
        if tempv1.ops.borrow().nvgetlength.is_none() {
            arkProcessError(
                Some(ark_mem),
                ARKLS_ILL_INPUT,
                line!() as i32,
                "ARKodeSetLinearSolver",
                file!(),
                MSG_LS_BAD_NVECTOR,
            );
            return ARKLS_ILL_INPUT;
        }

        if !matrixbased
            && (LSType != SUNLINEARSOLVER_MATRIX_EMBEDDED)
            && LS.ops.borrow().setatimes.is_none()
        {
            arkProcessError(
                Some(ark_mem),
                ARKLS_ILL_INPUT,
                line!() as i32,
                "ARKodeSetLinearSolver",
                file!(),
                "Incompatible inputs: iterative LS must support ATimes routine",
            );
            return ARKLS_ILL_INPUT;
        }

        if matrixbased && A.is_none() {
            arkProcessError(
                Some(ark_mem),
                ARKLS_ILL_INPUT,
                line!() as i32,
                "ARKodeSetLinearSolver",
                file!(),
                "Incompatible inputs: matrix-iterative LS requires non-NULL matrix",
            );
            return ARKLS_ILL_INPUT;
        }
    } else if A.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARKLS_ILL_INPUT,
            line!() as i32,
            "ARKodeSetLinearSolver",
            file!(),
            "Incompatible inputs: direct LS requires non-NULL matrix",
        );
        return ARKLS_ILL_INPUT;
    }

    /* Test whether time stepper module is supplied, with required routines */
    let missing_stepper = {
        let m = ark_mem.borrow();
        m.step_attachlinsol.is_none()
            || m.step_getlinmem.is_none()
            || m.step_getimplicitrhs.is_none()
            || m.step_getgammas.is_none()
    };
    if missing_stepper {
        arkProcessError(
            Some(ark_mem),
            ARKLS_ILL_INPUT,
            line!() as i32,
            "ARKodeSetLinearSolver",
            file!(),
            "Missing time step module or associated routines",
        );
        return ARKLS_ILL_INPUT;
    }

    /* C: malloc + memset(0) for ARKLsMemRec (allocation failure is
    unreachable here). The struct literal below carries exactly the state
    the C code holds after its default-assignment block. `Jt_f` is fetched
    first so that the "missing implicit RHS fcn" check below matches C. */
    let step_getimplicitrhs = ark_mem
        .borrow()
        .step_getimplicitrhs
        .expect("step_getimplicitrhs");
    let Jt_f = step_getimplicitrhs(ark_mem);

    if Jt_f.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARKLS_ILL_INPUT,
            line!() as i32,
            "ARKodeSetLinearSolver",
            file!(),
            "Time step module is missing implicit RHS fcn",
        );
        return ARKLS_ILL_INPUT;
    }

    let mut arkls_mem: ARKLsMem = Box::new(ARKLsMemRec {
        /* set SUNLinearSolver pointer */
        LS: LS.clone(),
        /* Linear solver type information */
        iterative,
        matrixbased,
        /* Set defaults for Jacobian-related fields */
        jacDQ: A.is_some(),
        jac: if A.is_some() {
            Some(arkLsDQJac as ARKLsJacFn)
        } else {
            None
        },
        J_data: if A.is_some() {
            Some(Box::new(ark_mem.clone())) /* C: J_data = ark_mem */
        } else {
            None
        },
        jtimesDQ: SUNTRUE,
        jtsetup: None,
        jtimes: Some(arkLsDQJtimes),
        Jt_data: Some(Box::new(ark_mem.clone())), /* C: Jt_data = ark_mem */
        Jt_f,
        user_linsys: SUNFALSE,
        linsys: Some(arkLsLinSys),
        A_data: Some(Box::new(ark_mem.clone())), /* C: A_data = ark_mem */
        /* Set defaults for preconditioner-related fields */
        pset: None,
        psolve: None,
        pfree: None,
        P_data: None, /* C: P_data = ark_mem->user_data (pass-through) */
        /* Counters (arkLsInitializeCounters below re-zeros them) */
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
        msbj: ARKLS_MSBJ,
        jbad: SUNTRUE,
        eplifac: ARKLS_EPLIN,
        last_flag: ARKLS_SUCCESS,
        /* memset(0) baseline for fields assigned further below */
        scalesol: SUNFALSE,
        nrmfac: 0.0,
        tcur: 0.0,
        A: None,
        savedJ: None,
        ytemp: None,
        x: None,
        ycur: None,
        fcur: None,
    });

    /* Initialize counters */
    let _ = arkLsInitializeCounters(&mut arkls_mem);

    /* If LS supports ATimes, attach ARKLs routine */
    if LS.ops.borrow().setatimes.is_some() {
        let retval = SUNLinSolSetATimes(LS, Some(Box::new(ark_mem.clone())), Some(arkLsATimes));
        if retval != SUN_SUCCESS {
            arkProcessError(
                Some(ark_mem),
                ARKLS_SUNLS_FAIL,
                line!() as i32,
                "ARKodeSetLinearSolver",
                file!(),
                "Error in calling SUNLinSolSetATimes",
            );
            drop(arkls_mem);
            return ARKLS_SUNLS_FAIL;
        }
    }

    /* If LS supports preconditioning, initialize pset/psol to NULL */
    if LS.ops.borrow().setpreconditioner.is_some() {
        let retval = SUNLinSolSetPreconditioner(LS, Some(Box::new(ark_mem.clone())), None, None);
        if retval != SUN_SUCCESS {
            arkProcessError(
                Some(ark_mem),
                ARKLS_SUNLS_FAIL,
                line!() as i32,
                "ARKodeSetLinearSolver",
                file!(),
                "Error in calling SUNLinSolSetPreconditioner",
            );
            drop(arkls_mem);
            return ARKLS_SUNLS_FAIL;
        }
    }

    /* When using a SUNMatrix object, store pointer to A and initialize savedJ */
    if let Some(A) = A {
        arkls_mem.A = Some(A.clone());
        arkls_mem.savedJ = None; /* allocated in arkLsInitialize */
    }

    /* Allocate memory for ytemp and x */
    if !arkAllocVec(ark_mem, &tempv1, &mut arkls_mem.ytemp) {
        arkProcessError(
            Some(ark_mem),
            ARKLS_MEM_FAIL,
            line!() as i32,
            "ARKodeSetLinearSolver",
            file!(),
            MSG_LS_MEM_FAIL,
        );
        drop(arkls_mem);
        return ARKLS_MEM_FAIL;
    }

    if !arkAllocVec(ark_mem, &tempv1, &mut arkls_mem.x) {
        arkProcessError(
            Some(ark_mem),
            ARKLS_MEM_FAIL,
            line!() as i32,
            "ARKodeSetLinearSolver",
            file!(),
            MSG_LS_MEM_FAIL,
        );
        arkFreeVec(ark_mem, &mut arkls_mem.ytemp);
        drop(arkls_mem);
        return ARKLS_MEM_FAIL;
    }

    /* For iterative LS, compute default norm conversion factor */
    if iterative {
        arkls_mem.nrmfac =
            SUNRsqrt(N_VGetLength(arkls_mem.ytemp.as_ref().expect("ytemp")) as sunrealtype);
    }

    /* For matrix-based LS, enable solution scaling */
    if matrixbased {
        arkls_mem.scalesol = SUNTRUE;
    } else {
        arkls_mem.scalesol = SUNFALSE;
    }

    /* Attach ARKLs interface to time stepper module */
    let step_attachlinsol = ark_mem
        .borrow()
        .step_attachlinsol
        .expect("step_attachlinsol");
    let lmem: Option<Box<dyn Any>> = Some(arkls_mem);
    let retval = step_attachlinsol(
        ark_mem,
        Some(arkLsInitialize),
        Some(arkLsSetup),
        Some(arkLsSolve),
        Some(arkLsFree),
        LSType,
        lmem,
    );
    if retval != ARK_SUCCESS {
        arkProcessError(
            Some(ark_mem),
            retval,
            line!() as i32,
            "ARKodeSetLinearSolver",
            file!(),
            "Failed to attach to time stepper module",
        );
        /* C: N_VDestroy(x); N_VDestroy(ytemp); free(arkls_mem). The box was
        moved into the callee, which drops the record (and with it the two
        vector handles) on any path that does not store it. */
        return retval;
    }

    ARKLS_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeSetMassLinearSolver specifies the iterative mass-matrix
  linear solver and user-supplied routine to perform the
  mass-matrix-vector product.
  ---------------------------------------------------------------*/
pub fn ARKodeSetMassLinearSolver(
    arkode_mem: &ARKodeMem,
    LS: &SUNLinearSolver,
    M: Option<&SUNMatrix>,
    time_dep: sunbooleantype,
) -> i32 {
    /* NULL arkode_mem check: handled by the type system */
    let ark_mem = arkode_mem;

    /* Guard against use for time steppers that do not support mass matrices */
    if !ark_mem.borrow().step_supports_massmatrix {
        arkProcessError(
            Some(ark_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetMassLinearSolver",
            file!(),
            "time-stepping module does not support non-identity mass matrices",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    /* NULL LS check: handled by the type system */

    /* Test if solver is compatible with LS interface */
    {
        let ops = LS.ops.borrow();
        if ops.gettype.is_none() || ops.solve.is_none() {
            drop(ops);
            arkProcessError(
                Some(ark_mem),
                ARKLS_ILL_INPUT,
                line!() as i32,
                "ARKodeSetMassLinearSolver",
                file!(),
                "LS object is missing a required operation",
            );
            return ARKLS_ILL_INPUT;
        }
    }

    /* Retrieve the LS type */
    let LSType = SUNLinSolGetType(LS);

    /* Set flags based on LS type */
    let iterative = LSType != SUNLINEARSOLVER_DIRECT;
    let matrixbased =
        (LSType != SUNLINEARSOLVER_ITERATIVE) && (LSType != SUNLINEARSOLVER_MATRIX_EMBEDDED);

    /* Test if vector is compatible with LS interface */
    let tempv1 = ark_mem
        .borrow()
        .tempv1
        .as_ref()
        .expect("tempv1")
        .clone();
    {
        let ops = tempv1.ops.borrow();
        if ops.nvconst.is_none() || ops.nvwrmsnorm.is_none() {
            drop(ops);
            arkProcessError(
                Some(ark_mem),
                ARKLS_ILL_INPUT,
                line!() as i32,
                "ARKodeSetMassLinearSolver",
                file!(),
                MSG_LS_BAD_NVECTOR,
            );
            return ARKLS_ILL_INPUT;
        }
    }

    /* Ensure that M is NULL when LS is matrix-embedded */
    if (LSType == SUNLINEARSOLVER_MATRIX_EMBEDDED) && M.is_some() {
        arkProcessError(
            Some(ark_mem),
            ARKLS_ILL_INPUT,
            line!() as i32,
            "ARKodeSetMassLinearSolver",
            file!(),
            "Incompatible inputs: matrix-embedded LS requires NULL matrix",
        );
        return ARKLS_ILL_INPUT;
    }

    /* Check for compatible LS type, matrix and "atimes" support */
    if iterative {
        if tempv1.ops.borrow().nvgetlength.is_none() {
            arkProcessError(
                Some(ark_mem),
                ARKLS_ILL_INPUT,
                line!() as i32,
                "ARKodeSetMassLinearSolver",
                file!(),
                MSG_LS_BAD_NVECTOR,
            );
            return ARKLS_ILL_INPUT;
        }

        if !matrixbased
            && (LSType != SUNLINEARSOLVER_MATRIX_EMBEDDED)
            && LS.ops.borrow().setatimes.is_none()
        {
            arkProcessError(
                Some(ark_mem),
                ARKLS_ILL_INPUT,
                line!() as i32,
                "ARKodeSetMassLinearSolver",
                file!(),
                "Incompatible inputs: iterative LS must support ATimes routine",
            );
            return ARKLS_ILL_INPUT;
        }

        if matrixbased && M.is_none() {
            arkProcessError(
                Some(ark_mem),
                ARKLS_ILL_INPUT,
                line!() as i32,
                "ARKodeSetMassLinearSolver",
                file!(),
                "Incompatible inputs: matrix-iterative LS requires non-NULL matrix",
            );
            return ARKLS_ILL_INPUT;
        }
    } else if M.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARKLS_ILL_INPUT,
            line!() as i32,
            "ARKodeSetMassLinearSolver",
            file!(),
            "Incompatible inputs: direct LS requires non-NULL matrix",
        );
        return ARKLS_ILL_INPUT;
    }

    /* Test whether time stepper module is supplied, with required routines */
    let missing_stepper = {
        let m = ark_mem.borrow();
        m.step_attachmasssol.is_none() || m.step_getmassmem.is_none()
    };
    if missing_stepper {
        arkProcessError(
            Some(ark_mem),
            ARKLS_ILL_INPUT,
            line!() as i32,
            "ARKodeSetMassLinearSolver",
            file!(),
            "Missing time step module or associated routines",
        );
        return ARKLS_ILL_INPUT;
    }

    /* C: malloc + memset(0) for ARKLsMassMemRec (allocation failure is
    unreachable here), followed by the default-assignment block. */
    let mut arkls_mem: ARKLsMassMem = Box::new(ARKLsMassMemRec {
        /* set SUNLinearSolver pointer */
        LS: LS.clone(),
        /* Linear solver type information */
        iterative,
        matrixbased,
        /* Set flag indicating time-dependence */
        time_dependent: time_dep,
        /* Set mass-matrix routines to NULL */
        mass: None,
        M_data: None,
        mtsetup: None,
        mtimes: None,
        mt_data: None,
        /* Set defaults for preconditioner-related fields */
        pset: None,
        psolve: None,
        pfree: None,
        P_data: None, /* C: P_data = ark_mem->user_data (pass-through) */
        /* Counters (arkLsInitializeMassCounters below re-zeros them) */
        nmsetups: 0,
        nmsolves: 0,
        nmtsetup: 0,
        nmtimes: 0,
        nmvsetup: 0,
        npe: 0,
        nli: 0,
        nps: 0,
        ncfl: 0,
        msetuptime: 0.0, /* memset baseline */
        /* Set default values for the rest of the LS parameters */
        eplifac: ARKLS_EPLIN,
        last_flag: ARKLS_SUCCESS,
        /* memset(0) baseline for fields assigned further below */
        nrmfac: 0.0,
        M: None,
        M_lu: None,
        x: None,
        ycur: None,
    });

    /* Initialize counters */
    let _ = arkLsInitializeMassCounters(&mut arkls_mem);

    /* If LS supports ATimes, attach ARKLs routine */
    if LS.ops.borrow().setatimes.is_some() {
        let retval = SUNLinSolSetATimes(LS, Some(Box::new(ark_mem.clone())), None);
        if retval != SUN_SUCCESS {
            arkProcessError(
                Some(ark_mem),
                ARKLS_SUNLS_FAIL,
                line!() as i32,
                "ARKodeSetMassLinearSolver",
                file!(),
                "Error in calling SUNLinSolSetATimes",
            );
            drop(arkls_mem);
            return ARKLS_SUNLS_FAIL;
        }
    }

    /* If LS supports preconditioning, initialize pset/psol to NULL */
    if LS.ops.borrow().setpreconditioner.is_some() {
        let retval = SUNLinSolSetPreconditioner(LS, Some(Box::new(ark_mem.clone())), None, None);
        if retval != SUN_SUCCESS {
            arkProcessError(
                Some(ark_mem),
                ARKLS_SUNLS_FAIL,
                line!() as i32,
                "ARKodeSetMassLinearSolver",
                file!(),
                "Error in calling SUNLinSolSetPreconditioner",
            );
            drop(arkls_mem);
            return ARKLS_SUNLS_FAIL;
        }
    }

    /* When using a non-NULL SUNMatrix object, store pointer to M and, for direct
       linear solvers, create M_lu to store the factorization of M */
    if let Some(M) = M {
        arkls_mem.M = Some(M.clone());
        if !iterative {
            match SUNMatClone(M) {
                Some(M_lu) => arkls_mem.M_lu = Some(M_lu),
                None => {
                    arkProcessError(
                        Some(ark_mem),
                        ARKLS_MEM_FAIL,
                        line!() as i32,
                        "ARKodeSetMassLinearSolver",
                        file!(),
                        MSG_LS_MEM_FAIL,
                    );
                    drop(arkls_mem);
                    return ARKLS_MEM_FAIL;
                }
            }
        } else {
            arkls_mem.M_lu = Some(M.clone());
        }
    }

    /* Allocate memory for x */
    if !arkAllocVec(ark_mem, &tempv1, &mut arkls_mem.x) {
        arkProcessError(
            Some(ark_mem),
            ARKLS_MEM_FAIL,
            line!() as i32,
            "ARKodeSetMassLinearSolver",
            file!(),
            MSG_LS_MEM_FAIL,
        );
        if !iterative {
            if let Some(M_lu) = arkls_mem.M_lu.take() {
                SUNMatDestroy(M_lu);
            }
        }
        drop(arkls_mem);
        return ARKLS_MEM_FAIL;
    }

    /* For iterative LS, compute default norm conversion factor */
    if iterative {
        arkls_mem.nrmfac =
            SUNRsqrt(N_VGetLength(arkls_mem.x.as_ref().expect("x")) as sunrealtype);
    }

    /* Attach ARKLs interface to time stepper module */
    let step_attachmasssol = ark_mem
        .borrow()
        .step_attachmasssol
        .expect("step_attachmasssol");
    let mass_mem: Option<Box<dyn Any>> = Some(arkls_mem);
    let retval = step_attachmasssol(
        ark_mem,
        Some(arkLsMassInitialize),
        Some(arkLsMassSetup),
        Some(arkLsMTimes),
        Some(arkLsMassSolve),
        Some(arkLsMassFree),
        time_dep,
        LSType,
        mass_mem,
    );
    if retval != ARK_SUCCESS {
        arkProcessError(
            Some(ark_mem),
            retval,
            line!() as i32,
            "ARKodeSetMassLinearSolver",
            file!(),
            "Failed to attach to time stepper module",
        );
        /* C: N_VDestroy(x); SUNMatDestroy(M_lu); free(arkls_mem). The box was
        moved into the callee, which drops the record (and with it those
        handles) on any path that does not store it. */
        return retval;
    }

    ARKLS_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeSetJacFn specifies the Jacobian function.
  ---------------------------------------------------------------*/
pub fn ARKodeSetJacFn(arkode_mem: &ARKodeMem, jac: Option<ARKLsJacFn>) -> i32 {
    /* NULL arkode_mem check: handled by the type system */
    let ark_mem = arkode_mem;

    /* Guard against use for time steppers that do not need an algebraic solver */
    if !ark_mem.borrow().step_supports_implicit {
        arkProcessError(
            Some(ark_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetJacFn",
            file!(),
            "time-stepping module does not require an algebraic solver",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    /* access ARKLsMem structure */
    let retval = arkLs_AccessLMem(ark_mem, "ARKodeSetJacFn");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* return with failure if jac cannot be used */
    if jac.is_some() && arkls_mem_mut(ark_mem).A.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARKLS_ILL_INPUT,
            line!() as i32,
            "ARKodeSetJacFn",
            file!(),
            "Jacobian routine cannot be supplied for NULL SUNMatrix",
        );
        return ARKLS_ILL_INPUT;
    }

    /* set the Jacobian routine pointer, and update relevant flags */
    if jac.is_some() {
        let mut ls = arkls_mem_mut(ark_mem);
        ls.jacDQ = SUNFALSE;
        ls.jac = jac;
        ls.J_data = None; /* C: J_data = ark_mem->user_data */
    } else {
        let J_data: Box<dyn Any> = Box::new(ark_mem.clone());
        let mut ls = arkls_mem_mut(ark_mem);
        ls.jacDQ = SUNTRUE;
        ls.jac = Some(arkLsDQJac);
        ls.J_data = Some(J_data); /* C: J_data = ark_mem */
    }

    /* ensure the internal linear system function is used */
    {
        let A_data: Box<dyn Any> = Box::new(ark_mem.clone());
        let mut ls = arkls_mem_mut(ark_mem);
        ls.user_linsys = SUNFALSE;
        ls.linsys = Some(arkLsLinSys);
        ls.A_data = Some(A_data); /* C: A_data = ark_mem */
    }

    ARKLS_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeSetMassFn specifies the mass matrix function.
  ---------------------------------------------------------------*/
pub fn ARKodeSetMassFn(arkode_mem: &ARKodeMem, mass: Option<ARKLsMassFn>) -> i32 {
    /* NULL arkode_mem check: handled by the type system */
    let ark_mem = arkode_mem;

    /* Guard against use for time steppers that do not support mass matrices */
    if !ark_mem.borrow().step_supports_massmatrix {
        arkProcessError(
            Some(ark_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetMassFn",
            file!(),
            "time-stepping module does not support non-identity mass matrices",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    /* access ARKLsMassMem structure */
    let retval = arkLs_AccessMassMem(ark_mem, "ARKodeSetMassFn");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* return with failure if mass cannot be used */
    if mass.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARKLS_ILL_INPUT,
            line!() as i32,
            "ARKodeSetMassFn",
            file!(),
            "Mass-matrix routine must be non-NULL",
        );
        return ARKLS_ILL_INPUT;
    }
    if arkls_mass_mem_mut(ark_mem).M.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARKLS_ILL_INPUT,
            line!() as i32,
            "ARKodeSetMassFn",
            file!(),
            "Mass-matrix routine cannot be supplied for NULL SUNMatrix",
        );
        return ARKLS_ILL_INPUT;
    }

    /* set mass matrix routine pointer and return */
    {
        let mut ls = arkls_mass_mem_mut(ark_mem);
        ls.mass = mass;
        ls.M_data = None; /* C: M_data = ark_mem->user_data */
    }

    ARKLS_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeSetEpsLin specifies the nonlinear -> linear tolerance
  scale factor.
  ---------------------------------------------------------------*/
pub fn ARKodeSetEpsLin(arkode_mem: &ARKodeMem, eplifac: sunrealtype) -> i32 {
    /* NULL arkode_mem check: handled by the type system */
    let ark_mem = arkode_mem;

    /* Guard against use for time steppers that do not need an algebraic solver */
    if !ark_mem.borrow().step_supports_implicit {
        arkProcessError(
            Some(ark_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetEpsLin",
            file!(),
            "time-stepping module does not require an algebraic solver",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    /* access ARKLsMem structure */
    let retval = arkLs_AccessLMem(ark_mem, "ARKodeSetEpsLin");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* store input and return */
    arkls_mem_mut(ark_mem).eplifac = if eplifac <= ZERO { ARKLS_EPLIN } else { eplifac };

    ARKLS_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeSetLSNormFactor sets or computes the factor to use when
  converting from the integrator tolerance (WRMS norm) to the
  linear solver tolerance (L2 norm).
  ---------------------------------------------------------------*/
pub fn ARKodeSetLSNormFactor(arkode_mem: &ARKodeMem, nrmfac: sunrealtype) -> i32 {
    /* NULL arkode_mem check: handled by the type system */
    let ark_mem = arkode_mem;

    /* Guard against use for time steppers that do not need an algebraic solver */
    if !ark_mem.borrow().step_supports_implicit {
        arkProcessError(
            Some(ark_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetLSNormFactor",
            file!(),
            "time-stepping module does not require an algebraic solver",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    /* access ARKLsMem structure */
    let retval = arkLs_AccessLMem(ark_mem, "ARKodeSetLSNormFactor");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* store input and return */
    if nrmfac > ZERO {
        /* set user-provided factor */
        arkls_mem_mut(ark_mem).nrmfac = nrmfac;
    } else if nrmfac < ZERO {
        /* Ensure that vector support N_VDotProd */
        let tempv1 = ark_mem
            .borrow()
            .tempv1
            .as_ref()
            .expect("tempv1")
            .clone();
        if tempv1.ops.borrow().nvdotprod.is_none() {
            arkProcessError(
                Some(ark_mem),
                ARKLS_ILL_INPUT,
                line!() as i32,
                "ARKodeSetLSNormFactor",
                file!(),
                "N_VDotProd unimplemented (required for ARKodeSetLSNormFactor)",
            );
            return ARKLS_ILL_INPUT;
        }

        /* compute factor for WRMS norm with dot product */
        N_VConst(ONE, &tempv1);
        arkls_mem_mut(ark_mem).nrmfac = SUNRsqrt(N_VDotProd(&tempv1, &tempv1));
    } else {
        /* compute default factor for WRMS norm from vector length */
        let tempv1 = ark_mem
            .borrow()
            .tempv1
            .as_ref()
            .expect("tempv1")
            .clone();
        arkls_mem_mut(ark_mem).nrmfac = SUNRsqrt(N_VGetLength(&tempv1) as sunrealtype);
    }

    ARKLS_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeSetJacEvalFrequency specifies the frequency for
  recomputing the Jacobian matrix and/or preconditioner.
  ---------------------------------------------------------------*/
pub fn ARKodeSetJacEvalFrequency(arkode_mem: &ARKodeMem, msbj: i64) -> i32 {
    /* NULL arkode_mem check: handled by the type system */
    let ark_mem = arkode_mem;

    /* Guard against use for time steppers that do not need an algebraic solver */
    if !ark_mem.borrow().step_supports_implicit {
        arkProcessError(
            Some(ark_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetJacEvalFrequency",
            file!(),
            "time-stepping module does not require an algebraic solver",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    /* access ARKLsMem structure */
    let retval = arkLs_AccessLMem(ark_mem, "ARKodeSetJacEvalFrequency");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* store input and return */
    arkls_mem_mut(ark_mem).msbj = if msbj <= 0 { ARKLS_MSBJ } else { msbj };

    ARKLS_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeSetLinearSolutionScaling enables or disables scaling the
  linear solver solution to account for changes in gamma.
  ---------------------------------------------------------------*/
pub fn ARKodeSetLinearSolutionScaling(arkode_mem: &ARKodeMem, onoff: sunbooleantype) -> i32 {
    /* NULL arkode_mem check: handled by the type system */
    let ark_mem = arkode_mem;

    /* Guard against use for time steppers that do not need an algebraic solver */
    if !ark_mem.borrow().step_supports_implicit {
        arkProcessError(
            Some(ark_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetLinearSolutionScaling",
            file!(),
            "time-stepping module does not require an algebraic solver",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    /* access ARKLsMem structure */
    let retval = arkLs_AccessLMem(ark_mem, "ARKodeSetLinearSolutionScaling");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* check for valid solver type */
    if !arkls_mem_mut(ark_mem).matrixbased {
        return ARKLS_ILL_INPUT;
    }

    /* set solution scaling flag */
    arkls_mem_mut(ark_mem).scalesol = onoff;

    ARKLS_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeSetPreconditioner specifies the user-supplied
  preconditioner setup and solve routines.
  ---------------------------------------------------------------*/
pub fn ARKodeSetPreconditioner(
    arkode_mem: &ARKodeMem,
    psetup: Option<ARKLsPrecSetupFn>,
    psolve: Option<ARKLsPrecSolveFn>,
) -> i32 {
    /* NULL arkode_mem check: handled by the type system */
    let ark_mem = arkode_mem;

    /* Guard against use for time steppers that do not need an algebraic solver */
    if !ark_mem.borrow().step_supports_implicit {
        arkProcessError(
            Some(ark_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetPreconditioner",
            file!(),
            "time-stepping module does not require an algebraic solver",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    /* access ARKLsMem structure */
    let retval = arkLs_AccessLMem(ark_mem, "ARKodeSetPreconditioner");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* issue error if LS object does not allow user-supplied preconditioning */
    let LS = arkls_mem_mut(ark_mem).LS.clone();
    if LS.ops.borrow().setpreconditioner.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARKLS_ILL_INPUT,
            line!() as i32,
            "ARKodeSetPreconditioner",
            file!(),
            "SUNLinearSolver object does not support user-supplied preconditioning",
        );
        return ARKLS_ILL_INPUT;
    }

    /* store function pointers for user-supplied routines */
    {
        let mut ls = arkls_mem_mut(ark_mem);
        ls.pset = psetup;
        ls.psolve = psolve;
    }

    /* notify linear solver to call ARKLs interface routines */
    let arkls_psetup: Option<SUNPSetupFn> = if psetup.is_none() {
        None
    } else {
        Some(arkLsPSetup)
    };
    let arkls_psolve: Option<SUNPSolveFn> = if psolve.is_none() {
        None
    } else {
        Some(arkLsPSolve)
    };
    let retval = SUNLinSolSetPreconditioner(
        &LS,
        Some(Box::new(ark_mem.clone())),
        arkls_psetup,
        arkls_psolve,
    );
    if retval != SUN_SUCCESS {
        arkProcessError(
            Some(ark_mem),
            ARKLS_SUNLS_FAIL,
            line!() as i32,
            "ARKodeSetPreconditioner",
            file!(),
            "Error in calling SUNLinSolSetPreconditioner",
        );
        return ARKLS_SUNLS_FAIL;
    }

    ARKLS_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeSetJacTimes specifies the user-supplied Jacobian-vector
  product setup and multiply routines.
  ---------------------------------------------------------------*/
pub fn ARKodeSetJacTimes(
    arkode_mem: &ARKodeMem,
    jtsetup: Option<ARKLsJacTimesSetupFn>,
    jtimes: Option<ARKLsJacTimesVecFn>,
) -> i32 {
    /* NULL arkode_mem check: handled by the type system */
    let ark_mem = arkode_mem;

    /* Guard against use for time steppers that do not need an algebraic solver */
    if !ark_mem.borrow().step_supports_implicit {
        arkProcessError(
            Some(ark_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetJacTimes",
            file!(),
            "time-stepping module does not require an algebraic solver",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    /* access ARKLsMem structure */
    let retval = arkLs_AccessLMem(ark_mem, "ARKodeSetJacTimes");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* issue error if LS object does not allow user-supplied ATimes */
    let LS = arkls_mem_mut(ark_mem).LS.clone();
    if LS.ops.borrow().setatimes.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARKLS_ILL_INPUT,
            line!() as i32,
            "ARKodeSetJacTimes",
            file!(),
            "SUNLinearSolver object does not support user-supplied ATimes routine",
        );
        return ARKLS_ILL_INPUT;
    }

    /* store function pointers for user-supplied routines in ARKLs
       interface (NULL jtimes implies use of DQ default) */
    if jtimes.is_some() {
        let mut ls = arkls_mem_mut(ark_mem);
        ls.jtimesDQ = SUNFALSE;
        ls.jtsetup = jtsetup;
        ls.jtimes = jtimes;
        ls.Jt_data = None; /* C: Jt_data = ark_mem->user_data */
    } else {
        let step_getimplicitrhs = ark_mem
            .borrow()
            .step_getimplicitrhs
            .expect("step_getimplicitrhs");
        let Jt_f = step_getimplicitrhs(ark_mem);
        let Jt_data: Box<dyn Any> = Box::new(ark_mem.clone());
        {
            let mut ls = arkls_mem_mut(ark_mem);
            ls.jtimesDQ = SUNTRUE;
            ls.jtsetup = None;
            ls.jtimes = Some(arkLsDQJtimes);
            ls.Jt_data = Some(Jt_data); /* C: Jt_data = ark_mem */
            ls.Jt_f = Jt_f;
        }

        if Jt_f.is_none() {
            arkProcessError(
                Some(ark_mem),
                ARKLS_ILL_INPUT,
                line!() as i32,
                "ARKodeSetJacTimes",
                file!(),
                "Time step module is missing implicit RHS fcn",
            );
            return ARKLS_ILL_INPUT;
        }
    }

    ARKLS_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeSetJacTimesRhsFn specifies an alternative user-supplied
  ODE right-hand side function to use in the internal finite
  difference Jacobian-vector product.
  ---------------------------------------------------------------*/
pub fn ARKodeSetJacTimesRhsFn(arkode_mem: &ARKodeMem, jtimesRhsFn: Option<ARKRhsFn>) -> i32 {
    /* NULL arkode_mem check: handled by the type system */
    let ark_mem = arkode_mem;

    /* Guard against use for time steppers that do not need an algebraic solver */
    if !ark_mem.borrow().step_supports_implicit {
        arkProcessError(
            Some(ark_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetJacTimesRhsFn",
            file!(),
            "time-stepping module does not require an algebraic solver",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    /* access ARKLsMem structure */
    let retval = arkLs_AccessLMem(ark_mem, "ARKodeSetJacTimesRhsFn");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* check if using internal finite difference approximation */
    if !arkls_mem_mut(ark_mem).jtimesDQ {
        arkProcessError(
            Some(ark_mem),
            ARKLS_ILL_INPUT,
            line!() as i32,
            "ARKodeSetJacTimesRhsFn",
            file!(),
            "Internal finite-difference Jacobian-vector product is disabled.",
        );
        return ARKLS_ILL_INPUT;
    }

    /* store function pointers for RHS function (NULL implies use ODE RHS) */
    if jtimesRhsFn.is_some() {
        arkls_mem_mut(ark_mem).Jt_f = jtimesRhsFn;
    } else {
        let step_getimplicitrhs = ark_mem
            .borrow()
            .step_getimplicitrhs
            .expect("step_getimplicitrhs");
        let Jt_f = step_getimplicitrhs(ark_mem);
        arkls_mem_mut(ark_mem).Jt_f = Jt_f;

        if Jt_f.is_none() {
            arkProcessError(
                Some(ark_mem),
                ARKLS_ILL_INPUT,
                line!() as i32,
                "ARKodeSetJacTimesRhsFn",
                file!(),
                "Time step module is missing implicit RHS fcn",
            );
            return ARKLS_ILL_INPUT;
        }
    }

    ARKLS_SUCCESS
}

/* ARKodeSetLinSysFn specifies the linear system setup function. */
pub fn ARKodeSetLinSysFn(arkode_mem: &ARKodeMem, linsys: Option<ARKLsLinSysFn>) -> i32 {
    /* NULL arkode_mem check: handled by the type system */
    let ark_mem = arkode_mem;

    /* Guard against use for time steppers that do not need an algebraic solver */
    if !ark_mem.borrow().step_supports_implicit {
        arkProcessError(
            Some(ark_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetLinSysFn",
            file!(),
            "time-stepping module does not require an algebraic solver",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    /* access ARKLsMem structure */
    let retval = arkLs_AccessLMem(ark_mem, "ARKodeSetLinSysFn");
    if retval != ARKLS_SUCCESS {
        return retval;
    }

    /* return with failure if linsys cannot be used */
    if linsys.is_some() && arkls_mem_mut(ark_mem).A.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARKLS_ILL_INPUT,
            line!() as i32,
            "ARKodeSetLinSysFn",
            file!(),
            "Linear system setup routine cannot be supplied for NULL SUNMatrix",
        );
        return ARKLS_ILL_INPUT;
    }

    /* set the linear system routine pointer, and update relevant flags */
    if linsys.is_some() {
        let mut ls = arkls_mem_mut(ark_mem);
        ls.user_linsys = SUNTRUE;
        ls.linsys = linsys;
        ls.A_data = None; /* C: A_data = ark_mem->user_data */
    } else {
        let A_data: Box<dyn Any> = Box::new(ark_mem.clone());
        let mut ls = arkls_mem_mut(ark_mem);
        ls.user_linsys = SUNFALSE;
        ls.linsys = Some(arkLsLinSys);
        ls.A_data = Some(A_data); /* C: A_data = ark_mem */
    }

    ARKLS_SUCCESS
}

pub fn ARKodeGetJac(arkode_mem: &ARKodeMem, J: &mut Option<SUNMatrix>) -> i32 {
    /* NULL arkode_mem check: handled by the type system */
    let ark_mem = arkode_mem;

    /* Return NULL for incompatible steppers */
    if !ark_mem.borrow().step_supports_implicit {
        *J = None;
        return ARK_SUCCESS;
    }

    /* access ARKLsMem structure */
    let retval = arkLs_AccessLMem(ark_mem, "ARKodeGetJac");
    if retval != ARKLS_SUCCESS {
        return retval;
    }

    /* set output and return */
    *J = arkls_mem_mut(ark_mem).savedJ.clone();
    ARKLS_SUCCESS
}

pub fn ARKodeGetJacTime(arkode_mem: &ARKodeMem, t_J: &mut sunrealtype) -> i32 {
    /* NULL arkode_mem check: handled by the type system */
    let ark_mem = arkode_mem;

    /* Return an error for incompatible steppers */
    if !ark_mem.borrow().step_supports_implicit {
        arkProcessError(
            Some(ark_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeGetJacTime",
            file!(),
            "time-stepping module does not require an algebraic solver",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    /* access ARKLsMem structure */
    let retval = arkLs_AccessLMem(ark_mem, "ARKodeGetJacTime");
    if retval != ARKLS_SUCCESS {
        return retval;
    }

    /* set output and return */
    *t_J = arkls_mem_mut(ark_mem).tnlj;
    ARKLS_SUCCESS
}

pub fn ARKodeGetJacNumSteps(arkode_mem: &ARKodeMem, nst_J: &mut i64) -> i32 {
    /* NULL arkode_mem check: handled by the type system */
    let ark_mem = arkode_mem;

    /* Return 0 for incompatible steppers */
    if !ark_mem.borrow().step_supports_implicit {
        *nst_J = 0;
        return ARK_SUCCESS;
    }

    /* access ARKLsMem structure */
    let retval = arkLs_AccessLMem(ark_mem, "ARKodeGetJacNumSteps");
    if retval != ARKLS_SUCCESS {
        return retval;
    }

    /* set output and return */
    *nst_J = arkls_mem_mut(ark_mem).nstlj;
    ARKLS_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeGetLinWorkSpace returns the length of workspace allocated for
  the ARKLS linear solver interface.
  ---------------------------------------------------------------*/
pub fn ARKodeGetLinWorkSpace(arkode_mem: &ARKodeMem, lenrw: &mut i64, leniw: &mut i64) -> i32 {
    /* NULL arkode_mem check: handled by the type system */
    let ark_mem = arkode_mem;

    /* Return 0 for incompatible steppers */
    if !ark_mem.borrow().step_supports_implicit {
        *lenrw = 0;
        *leniw = 0;
        return ARK_SUCCESS;
    }

    /* access ARKLsMem structure */
    let retval = arkLs_AccessLMem(ark_mem, "ARKodeGetLinWorkSpace");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* start with fixed sizes plus vector/matrix pointers */
    *lenrw = 3;
    *leniw = 30;

    /* add NVector sizes */
    let x = arkls_mem_mut(ark_mem).x.as_ref().expect("x").clone();
    if x.ops.borrow().nvspace.is_some() {
        let mut lrw1: sunindextype = 0;
        let mut liw1: sunindextype = 0;
        N_VSpace(&x, &mut lrw1, &mut liw1);
        *lenrw += 2 * lrw1;
        *leniw += 2 * liw1;
    }

    /* add SUNMatrix size (only account for the one owned by Ls interface) */
    let savedJ = arkls_mem_mut(ark_mem).savedJ.clone();
    if let Some(savedJ) = &savedJ {
        if savedJ.ops.borrow().space.is_some() {
            let mut lrw: i64 = 0;
            let mut liw: i64 = 0;
            let retval = SUNMatSpace(savedJ, &mut lrw, &mut liw);
            if retval == 0 {
                *lenrw += lrw;
                *leniw += liw;
            }
        }
    }

    /* add LS sizes */
    let LS = arkls_mem_mut(ark_mem).LS.clone();
    if LS.ops.borrow().space.is_some() {
        let mut lrw: i64 = 0;
        let mut liw: i64 = 0;
        let retval = SUNLinSolSpace(&LS, &mut lrw, &mut liw);
        if retval == SUN_SUCCESS {
            *lenrw += lrw;
            *leniw += liw;
        }
    }

    ARKLS_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeGetNumJacEvals returns the number of Jacobian evaluations
  ---------------------------------------------------------------*/
pub fn ARKodeGetNumJacEvals(arkode_mem: &ARKodeMem, njevals: &mut i64) -> i32 {
    let ark_mem = arkode_mem;

    /* Return 0 for incompatible steppers */
    if !ark_mem.borrow().step_supports_implicit {
        *njevals = 0;
        return ARK_SUCCESS;
    }

    /* access ARKLsMem structure */
    let retval = arkLs_AccessLMem(ark_mem, "ARKodeGetNumJacEvals");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* set output and return */
    *njevals = arkls_mem_mut(ark_mem).nje;
    ARKLS_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeGetNumLinRhsEvals returns the number of calls to the ODE
  function needed for the DQ Jacobian approximation or J*v product
  approximation.
  ---------------------------------------------------------------*/
pub fn ARKodeGetNumLinRhsEvals(arkode_mem: &ARKodeMem, nfevalsLS: &mut i64) -> i32 {
    let ark_mem = arkode_mem;

    /* Return 0 for incompatible steppers */
    if !ark_mem.borrow().step_supports_implicit {
        *nfevalsLS = 0;
        return ARK_SUCCESS;
    }

    /* access ARKLsMem structure */
    let retval = arkLs_AccessLMem(ark_mem, "ARKodeGetNumLinRhsEvals");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* set output and return */
    *nfevalsLS = arkls_mem_mut(ark_mem).nfeDQ;
    ARKLS_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeGetNumPrecEvals returns the number of calls to the
  user- or ARKODE-supplied preconditioner setup routine.
  ---------------------------------------------------------------*/
pub fn ARKodeGetNumPrecEvals(arkode_mem: &ARKodeMem, npevals: &mut i64) -> i32 {
    let ark_mem = arkode_mem;

    /* Return 0 for incompatible steppers */
    if !ark_mem.borrow().step_supports_implicit {
        *npevals = 0;
        return ARK_SUCCESS;
    }

    /* access ARKLsMem structure */
    let retval = arkLs_AccessLMem(ark_mem, "ARKodeGetNumPrecEvals");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* set output and return */
    *npevals = arkls_mem_mut(ark_mem).npe;
    ARKLS_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeGetNumPrecSolves returns the number of calls to the
  user- or ARKODE-supplied preconditioner solve routine.
  ---------------------------------------------------------------*/
pub fn ARKodeGetNumPrecSolves(arkode_mem: &ARKodeMem, npsolves: &mut i64) -> i32 {
    let ark_mem = arkode_mem;

    /* Return 0 for incompatible steppers */
    if !ark_mem.borrow().step_supports_implicit {
        *npsolves = 0;
        return ARK_SUCCESS;
    }

    /* access ARKLsMem structure */
    let retval = arkLs_AccessLMem(ark_mem, "ARKodeGetNumPrecSolves");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* set output and return */
    *npsolves = arkls_mem_mut(ark_mem).nps;
    ARKLS_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeGetNumLinIters returns the number of linear iterations
  (if accessible from the LS object).
  ---------------------------------------------------------------*/
pub fn ARKodeGetNumLinIters(arkode_mem: &ARKodeMem, nliters: &mut i64) -> i32 {
    let ark_mem = arkode_mem;

    /* Return 0 for incompatible steppers */
    if !ark_mem.borrow().step_supports_implicit {
        *nliters = 0;
        return ARK_SUCCESS;
    }

    /* access ARKLsMem structure */
    let retval = arkLs_AccessLMem(ark_mem, "ARKodeGetNumLinIters");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* set output and return */
    *nliters = arkls_mem_mut(ark_mem).nli;
    ARKLS_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeGetNumLinConvFails returns the number of linear solver
  convergence failures (as reported by the LS object).
  ---------------------------------------------------------------*/
pub fn ARKodeGetNumLinConvFails(arkode_mem: &ARKodeMem, nlcfails: &mut i64) -> i32 {
    let ark_mem = arkode_mem;

    /* Return 0 for incompatible steppers */
    if !ark_mem.borrow().step_supports_implicit {
        *nlcfails = 0;
        return ARK_SUCCESS;
    }

    /* access ARKLsMem structure */
    let retval = arkLs_AccessLMem(ark_mem, "ARKodeGetNumLinConvFails");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* set output and return */
    *nlcfails = arkls_mem_mut(ark_mem).ncfl;
    ARKLS_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeGetNumJTSetupEvals returns the number of calls to the
  user-supplied Jacobian-vector product setup routine.
  ---------------------------------------------------------------*/
pub fn ARKodeGetNumJTSetupEvals(arkode_mem: &ARKodeMem, njtsetups: &mut i64) -> i32 {
    let ark_mem = arkode_mem;

    /* Return 0 for incompatible steppers */
    if !ark_mem.borrow().step_supports_implicit {
        *njtsetups = 0;
        return ARK_SUCCESS;
    }

    /* access ARKLsMem structure */
    let retval = arkLs_AccessLMem(ark_mem, "ARKodeGetNumJTSetupEvals");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* set output and return */
    *njtsetups = arkls_mem_mut(ark_mem).njtsetup;
    ARKLS_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeGetNumJtimesEvals returns the number of calls to the
  Jacobian-vector product multiply routine.
  ---------------------------------------------------------------*/
pub fn ARKodeGetNumJtimesEvals(arkode_mem: &ARKodeMem, njvevals: &mut i64) -> i32 {
    let ark_mem = arkode_mem;

    /* Return 0 for incompatible steppers */
    if !ark_mem.borrow().step_supports_implicit {
        *njvevals = 0;
        return ARK_SUCCESS;
    }

    /* access ARKLsMem structures */
    let retval = arkLs_AccessLMem(ark_mem, "ARKodeGetNumJtimesEvals");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* set output and return */
    *njvevals = arkls_mem_mut(ark_mem).njtimes;
    ARKLS_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeGetNumMassMultSetups returns the number of calls to the
  mass matrix-vector setup routine.
  ---------------------------------------------------------------*/
pub fn ARKodeGetNumMassMultSetups(arkode_mem: &ARKodeMem, nmvsetups: &mut i64) -> i32 {
    let ark_mem = arkode_mem;

    /* Return 0 for incompatible steppers */
    if !ark_mem.borrow().step_supports_massmatrix {
        *nmvsetups = 0;
        return ARK_SUCCESS;
    }

    /* access ARKMassMem structure */
    let retval = arkLs_AccessMassMem(ark_mem, "ARKodeGetNumMassMultSetups");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* set output and return */
    *nmvsetups = arkls_mass_mem_mut(ark_mem).nmvsetup;
    ARKLS_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeGetLastLinFlag returns the last flag set in a ARKLS
  function.
  ---------------------------------------------------------------*/
pub fn ARKodeGetLastLinFlag(arkode_mem: &ARKodeMem, flag: &mut i64) -> i32 {
    let ark_mem = arkode_mem;

    /* Return success for incompatible steppers */
    if !ark_mem.borrow().step_supports_implicit {
        *flag = ARKLS_SUCCESS as i64;
        return ARK_SUCCESS;
    }

    /* access ARKLsMem structure */
    let retval = arkLs_AccessLMem(ark_mem, "ARKodeGetLastLinFlag");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* set output and return */
    *flag = arkls_mem_mut(ark_mem).last_flag as i64;
    ARKLS_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeGetLinReturnFlagName translates from the integer error code
  returned by an ARKLs routine to the corresponding string
  equivalent for that flag
  ---------------------------------------------------------------*/
pub fn ARKodeGetLinReturnFlagName(flag: i64) -> String {
    let name = if flag == ARKLS_SUCCESS as i64 {
        "ARKLS_SUCCESS"
    } else if flag == ARKLS_MEM_NULL as i64 {
        "ARKLS_MEM_NULL"
    } else if flag == ARKLS_LMEM_NULL as i64 {
        "ARKLS_LMEM_NULL"
    } else if flag == ARKLS_ILL_INPUT as i64 {
        "ARKLS_ILL_INPUT"
    } else if flag == ARKLS_MEM_FAIL as i64 {
        "ARKLS_MEM_FAIL"
    } else if flag == ARKLS_MASSMEM_NULL as i64 {
        "ARKLS_MASSMEM_NULL"
    } else if flag == ARKLS_JACFUNC_UNRECVR as i64 {
        "ARKLS_JACFUNC_UNRECVR"
    } else if flag == ARKLS_JACFUNC_RECVR as i64 {
        "ARKLS_JACFUNC_RECVR"
    } else if flag == ARKLS_MASSFUNC_UNRECVR as i64 {
        "ARKLS_MASSFUNC_UNRECVR"
    } else if flag == ARKLS_MASSFUNC_RECVR as i64 {
        "ARKLS_MASSFUNC_RECVR"
    } else if flag == ARKLS_SUNMAT_FAIL as i64 {
        "ARKLS_SUNMAT_FAIL"
    } else if flag == ARKLS_SUNLS_FAIL as i64 {
        "ARKLS_SUNLS_FAIL"
    } else {
        "NONE"
    };
    name.to_string()
}

/*---------------------------------------------------------------
  ARKodeSetMassEpsLin specifies the nonlinear -> linear tolerance
  scale factor for mass matrix linear systems.
  ---------------------------------------------------------------*/
pub fn ARKodeSetMassEpsLin(arkode_mem: &ARKodeMem, eplifac: sunrealtype) -> i32 {
    let ark_mem = arkode_mem;

    /* Guard against use for time steppers that do not support mass matrices */
    if !ark_mem.borrow().step_supports_massmatrix {
        arkProcessError(
            Some(ark_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetMassEpsLin",
            file!(),
            "time-stepping module does not support non-identity mass matrices",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    /* access ARKLsMassMem structure */
    let retval = arkLs_AccessMassMem(ark_mem, "ARKodeSetMassEpsLin");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* store input and return */
    arkls_mass_mem_mut(ark_mem).eplifac = if eplifac <= ZERO { ARKLS_EPLIN } else { eplifac };

    ARKLS_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeSetMassLSNormFactor sets or computes the factor to use when
  converting from the integrator tolerance (WRMS norm) to the
  linear solver tolerance (L2 norm).
  ---------------------------------------------------------------*/
pub fn ARKodeSetMassLSNormFactor(arkode_mem: &ARKodeMem, nrmfac: sunrealtype) -> i32 {
    let ark_mem = arkode_mem;

    /* Guard against use for time steppers that do not support mass matrices */
    if !ark_mem.borrow().step_supports_massmatrix {
        arkProcessError(
            Some(ark_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetMassLSNormFactor",
            file!(),
            "time-stepping module does not support non-identity mass matrices",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    /* access ARKLsMem structures */
    let retval = arkLs_AccessMassMem(ark_mem, "ARKodeSetMassLSNormFactor");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* store input and return */
    if nrmfac > ZERO {
        /* set user-provided factor */
        arkls_mass_mem_mut(ark_mem).nrmfac = nrmfac;
    } else if nrmfac < ZERO {
        /* Ensure that vector support N_VDotProd */
        let tempv1 = ark_mem
            .borrow()
            .tempv1
            .as_ref()
            .expect("tempv1")
            .clone();
        if tempv1.ops.borrow().nvdotprod.is_none() {
            arkProcessError(
                Some(ark_mem),
                ARKLS_ILL_INPUT,
                line!() as i32,
                "ARKodeSetMassLSNormFactor",
                file!(),
                "N_VDotProd unimplemented (required for ARKodeSetMassLSNormFactor)",
            );
            return ARKLS_ILL_INPUT;
        }

        /* compute factor for WRMS norm with dot product */
        N_VConst(ONE, &tempv1);
        arkls_mass_mem_mut(ark_mem).nrmfac = SUNRsqrt(N_VDotProd(&tempv1, &tempv1));
    } else {
        /* compute default factor for WRMS norm from vector length */
        let tempv1 = ark_mem
            .borrow()
            .tempv1
            .as_ref()
            .expect("tempv1")
            .clone();
        arkls_mass_mem_mut(ark_mem).nrmfac = SUNRsqrt(N_VGetLength(&tempv1) as sunrealtype);
    }

    ARKLS_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeSetMassPreconditioner specifies the user-supplied
  preconditioner setup and solve routines.
  ---------------------------------------------------------------*/
pub fn ARKodeSetMassPreconditioner(
    arkode_mem: &ARKodeMem,
    psetup: Option<ARKLsMassPrecSetupFn>,
    psolve: Option<ARKLsMassPrecSolveFn>,
) -> i32 {
    let ark_mem = arkode_mem;

    /* Guard against use for time steppers that do not support mass matrices */
    if !ark_mem.borrow().step_supports_massmatrix {
        arkProcessError(
            Some(ark_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetMassPreconditioner",
            file!(),
            "time-stepping module does not support non-identity mass matrices",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    /* access ARKLsMassMem structure */
    let retval = arkLs_AccessMassMem(ark_mem, "ARKodeSetMassPreconditioner");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* issue error if LS object does not allow user-supplied preconditioning */
    let LS = arkls_mass_mem_mut(ark_mem).LS.clone();
    if LS.ops.borrow().setpreconditioner.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARKLS_ILL_INPUT,
            line!() as i32,
            "ARKodeSetMassPreconditioner",
            file!(),
            "SUNLinearSolver object does not support user-supplied preconditioning",
        );
        return ARKLS_ILL_INPUT;
    }

    /* store function pointers for user-supplied routines in ARKLs interface */
    {
        let mut ls = arkls_mass_mem_mut(ark_mem);
        ls.pset = psetup;
        ls.psolve = psolve;
    }

    /* notify linear solver to call ARKLs interface routines */
    let arkls_mpsetup: Option<SUNPSetupFn> = if psetup.is_none() {
        None
    } else {
        Some(arkLsMPSetup)
    };
    let arkls_mpsolve: Option<SUNPSolveFn> = if psolve.is_none() {
        None
    } else {
        Some(arkLsMPSolve)
    };
    let retval = SUNLinSolSetPreconditioner(
        &LS,
        Some(Box::new(ark_mem.clone())),
        arkls_mpsetup,
        arkls_mpsolve,
    );
    if retval != SUN_SUCCESS {
        arkProcessError(
            Some(ark_mem),
            ARKLS_SUNLS_FAIL,
            line!() as i32,
            "ARKodeSetMassPreconditioner",
            file!(),
            "Error in calling SUNLinSolSetPreconditioner",
        );
        return ARKLS_SUNLS_FAIL;
    }

    ARKLS_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeSetMassTimes specifies the user-supplied mass
  matrix-vector product setup and multiply routines.

  `mtimes_data` is stored verbatim (arkode_ls.c:1824) and handed to
  `mtimes`/`mtsetup` unchanged: `None` here means those callbacks receive
  `None`, exactly as C hands them NULL. It is NOT re-pointed at the
  integrator's `user_data` by a later `ARKodeSetUserData` — to share state
  with the RHS callback, clone an `Rc` handle into both boxes.
  ---------------------------------------------------------------*/
pub fn ARKodeSetMassTimes(
    arkode_mem: &ARKodeMem,
    mtsetup: Option<ARKLsMassTimesSetupFn>,
    mtimes: Option<ARKLsMassTimesVecFn>,
    mtimes_data: Option<Box<dyn Any>>,
) -> i32 {
    let ark_mem = arkode_mem;

    /* Guard against use for time steppers that do not support mass matrices */
    if !ark_mem.borrow().step_supports_massmatrix {
        arkProcessError(
            Some(ark_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetMassTimes",
            file!(),
            "time-stepping module does not support non-identity mass matrices",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    /* access ARKLsMassMem structure */
    let retval = arkLs_AccessMassMem(ark_mem, "ARKodeSetMassTimes");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* issue error if mtimes function is unusable */
    if mtimes.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARKLS_ILL_INPUT,
            line!() as i32,
            "ARKodeSetMassTimes",
            file!(),
            "non-NULL mtimes function must be supplied",
        );
        return ARKLS_ILL_INPUT;
    }

    /* issue error if LS object does not allow user-supplied ATimes */
    let LS = arkls_mass_mem_mut(ark_mem).LS.clone();
    if LS.ops.borrow().setatimes.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARKLS_ILL_INPUT,
            line!() as i32,
            "ARKodeSetMassTimes",
            file!(),
            "SUNLinearSolver object does not support user-supplied ATimes routine",
        );
        return ARKLS_ILL_INPUT;
    }

    /* store pointers for user-supplied routines and data structure
       in ARKLs interface */
    {
        let mut ls = arkls_mass_mem_mut(ark_mem);
        ls.mtsetup = mtsetup;
        ls.mtimes = mtimes;
        ls.mt_data = mtimes_data;
    }

    /* notify linear solver to call ARKLs interface routine */
    let retval = SUNLinSolSetATimes(
        &LS,
        Some(Box::new(ark_mem.clone())),
        Some(arkLsMTimesATimes),
    );
    if retval != SUN_SUCCESS {
        arkProcessError(
            Some(ark_mem),
            ARKLS_SUNLS_FAIL,
            line!() as i32,
            "ARKodeSetMassTimes",
            file!(),
            "Error in calling SUNLinSolSetATimes",
        );
        return ARKLS_SUNLS_FAIL;
    }

    ARKLS_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeGetMassWorkSpace
  ---------------------------------------------------------------*/
pub fn ARKodeGetMassWorkSpace(arkode_mem: &ARKodeMem, lenrw: &mut i64, leniw: &mut i64) -> i32 {
    let ark_mem = arkode_mem;

    /* Return 0 for incompatible steppers */
    if !ark_mem.borrow().step_supports_massmatrix {
        *lenrw = 0;
        *leniw = 0;
        return ARK_SUCCESS;
    }

    /* access ARKLsMassMem structure */
    let retval = arkLs_AccessMassMem(ark_mem, "ARKodeGetMassWorkSpace");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* start with fixed sizes plus vector/matrix pointers */
    *lenrw = 2;
    *leniw = 23;

    /* add NVector sizes */
    let tempv1 = ark_mem
        .borrow()
        .tempv1
        .as_ref()
        .expect("tempv1")
        .clone();
    if tempv1.ops.borrow().nvspace.is_some() {
        let mut lrw1: sunindextype = 0;
        let mut liw1: sunindextype = 0;
        N_VSpace(&tempv1, &mut lrw1, &mut liw1);
        *lenrw += lrw1;
        *leniw += liw1;
    }

    /* add SUNMatrix size (only account for the one owned by Ls interface) */
    let (iterative, M_lu) = {
        let ls = arkls_mass_mem_mut(ark_mem);
        (ls.iterative, ls.M_lu.clone())
    };
    if !iterative {
        if let Some(M_lu) = &M_lu {
            if M_lu.ops.borrow().space.is_some() {
                let mut lrw: i64 = 0;
                let mut liw: i64 = 0;
                let retval = SUNMatSpace(M_lu, &mut lrw, &mut liw);
                if retval == 0 {
                    *lenrw += lrw;
                    *leniw += liw;
                }
            }
        }
    }

    /* add LS sizes */
    let LS = arkls_mass_mem_mut(ark_mem).LS.clone();
    if LS.ops.borrow().space.is_some() {
        let mut lrw: i64 = 0;
        let mut liw: i64 = 0;
        let retval = SUNLinSolSpace(&LS, &mut lrw, &mut liw);
        if retval == SUN_SUCCESS {
            *lenrw += lrw;
            *leniw += liw;
        }
    }

    ARKLS_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeGetNumMassSetups returns the number of mass matrix
  solver 'setup' calls
  ---------------------------------------------------------------*/
pub fn ARKodeGetNumMassSetups(arkode_mem: &ARKodeMem, nmsetups: &mut i64) -> i32 {
    let ark_mem = arkode_mem;

    /* Return 0 for incompatible steppers */
    if !ark_mem.borrow().step_supports_massmatrix {
        *nmsetups = 0;
        return ARK_SUCCESS;
    }

    /* access ARKLsMassMem structure */
    let retval = arkLs_AccessMassMem(ark_mem, "ARKodeGetNumMassSetups");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* set output and return */
    *nmsetups = arkls_mass_mem_mut(ark_mem).nmsetups;
    ARKLS_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeGetNumMassMult returns the number of calls to the user-
  supplied or internal mass matrix-vector product multiply routine.
  ---------------------------------------------------------------*/
pub fn ARKodeGetNumMassMult(arkode_mem: &ARKodeMem, nmvevals: &mut i64) -> i32 {
    let ark_mem = arkode_mem;

    /* Return 0 for incompatible steppers */
    if !ark_mem.borrow().step_supports_massmatrix {
        *nmvevals = 0;
        return ARK_SUCCESS;
    }

    /* access ARKLsMassMem structure */
    let retval = arkLs_AccessMassMem(ark_mem, "ARKodeGetNumMassMult");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* set output and return */
    *nmvevals = arkls_mass_mem_mut(ark_mem).nmtimes;
    ARKLS_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeGetNumMassSolves returns the number of mass matrix
  solver 'solve' calls
  ---------------------------------------------------------------*/
pub fn ARKodeGetNumMassSolves(arkode_mem: &ARKodeMem, nmsolves: &mut i64) -> i32 {
    let ark_mem = arkode_mem;

    /* Return 0 for incompatible steppers */
    if !ark_mem.borrow().step_supports_massmatrix {
        *nmsolves = 0;
        return ARK_SUCCESS;
    }

    /* access ARKLsMassMem structure */
    let retval = arkLs_AccessMassMem(ark_mem, "ARKodeGetNumMassSolves");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* set output and return */
    *nmsolves = arkls_mass_mem_mut(ark_mem).nmsolves;
    ARKLS_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeGetNumMassPrecEvals returns the number of calls to the
  user- or ARKODE-supplied preconditioner setup routine.
  ---------------------------------------------------------------*/
pub fn ARKodeGetNumMassPrecEvals(arkode_mem: &ARKodeMem, npevals: &mut i64) -> i32 {
    let ark_mem = arkode_mem;

    /* Return 0 for incompatible steppers */
    if !ark_mem.borrow().step_supports_massmatrix {
        *npevals = 0;
        return ARK_SUCCESS;
    }

    /* access ARKLsMassMem structure */
    let retval = arkLs_AccessMassMem(ark_mem, "ARKodeGetNumMassPrecEvals");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* set output and return */
    *npevals = arkls_mass_mem_mut(ark_mem).npe;
    ARKLS_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeGetNumMassPrecSolves returns the number of calls to the
  user- or ARKODE-supplied preconditioner solve routine.
  ---------------------------------------------------------------*/
pub fn ARKodeGetNumMassPrecSolves(arkode_mem: &ARKodeMem, npsolves: &mut i64) -> i32 {
    let ark_mem = arkode_mem;

    /* Return 0 for incompatible steppers */
    if !ark_mem.borrow().step_supports_massmatrix {
        *npsolves = 0;
        return ARK_SUCCESS;
    }

    /* access ARKLsMassMem structure */
    let retval = arkLs_AccessMassMem(ark_mem, "ARKodeGetNumMassPrecSolves");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* set output and return */
    *npsolves = arkls_mass_mem_mut(ark_mem).nps;
    ARKLS_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeGetNumMassIters returns the number of mass matrix solver
  linear iterations (if accessible from the LS object).
  ---------------------------------------------------------------*/
pub fn ARKodeGetNumMassIters(arkode_mem: &ARKodeMem, nmiters: &mut i64) -> i32 {
    let ark_mem = arkode_mem;

    /* Return 0 for incompatible steppers */
    if !ark_mem.borrow().step_supports_massmatrix {
        *nmiters = 0;
        return ARK_SUCCESS;
    }

    /* access ARKLsMassMem structure */
    let retval = arkLs_AccessMassMem(ark_mem, "ARKodeGetNumMassIters");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* set output and return */
    *nmiters = arkls_mass_mem_mut(ark_mem).nli;
    ARKLS_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeGetNumMassConvFails returns the number of linear solver
  convergence failures (as reported by the LS object).
  ---------------------------------------------------------------*/
pub fn ARKodeGetNumMassConvFails(arkode_mem: &ARKodeMem, nmcfails: &mut i64) -> i32 {
    let ark_mem = arkode_mem;

    /* Return 0 for incompatible steppers */
    if !ark_mem.borrow().step_supports_massmatrix {
        *nmcfails = 0;
        return ARK_SUCCESS;
    }

    /* access ARKLsMassMem structure */
    let retval = arkLs_AccessMassMem(ark_mem, "ARKodeGetNumMassConvFails");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* set output and return */
    *nmcfails = arkls_mass_mem_mut(ark_mem).ncfl;
    ARKLS_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeGetCurrentMassMatrix returns the current mass matrix.
  ---------------------------------------------------------------*/
pub fn ARKodeGetCurrentMassMatrix(arkode_mem: &ARKodeMem, M: &mut Option<SUNMatrix>) -> i32 {
    let ark_mem = arkode_mem;

    /* Return NULL for incompatible steppers */
    if !ark_mem.borrow().step_supports_massmatrix {
        *M = None;
        return ARK_SUCCESS;
    }

    /* access ARKLsMassMem structure */
    let retval = arkLs_AccessMassMem(ark_mem, "ARKodeGetCurrentMassMatrix");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* set output and return */
    *M = arkls_mass_mem_mut(ark_mem).M.clone();
    ARKLS_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeGetNumMTSetups returns the number of calls to the
  user-supplied mass matrix-vector product setup routine.
  ---------------------------------------------------------------*/
pub fn ARKodeGetNumMTSetups(arkode_mem: &ARKodeMem, nmtsetups: &mut i64) -> i32 {
    let ark_mem = arkode_mem;

    /* Return 0 for incompatible steppers */
    if !ark_mem.borrow().step_supports_massmatrix {
        *nmtsetups = 0;
        return ARK_SUCCESS;
    }

    /* access ARKLsMassMem structure */
    let retval = arkLs_AccessMassMem(ark_mem, "ARKodeGetNumMTSetups");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* set output value and return */
    *nmtsetups = arkls_mass_mem_mut(ark_mem).nmtsetup;
    ARKLS_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeGetLastMassFlag returns the last flag set in a ARKLS
  function.
  ---------------------------------------------------------------*/
pub fn ARKodeGetLastMassFlag(arkode_mem: &ARKodeMem, flag: &mut i64) -> i32 {
    let ark_mem = arkode_mem;

    /* Return ARKLS_SUCCESS for incompatible steppers */
    if !ark_mem.borrow().step_supports_massmatrix {
        *flag = ARKLS_SUCCESS as i64;
        return ARK_SUCCESS;
    }

    /* access ARKLsMassMem structure */
    let retval = arkLs_AccessMassMem(ark_mem, "ARKodeGetLastMassFlag");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* set output and return */
    *flag = arkls_mass_mem_mut(ark_mem).last_flag as i64;
    ARKLS_SUCCESS
}

/*===============================================================
  ARKLS Private functions
  ===============================================================*/

/// arkLSSetUserData sets user_data pointers in arkLS.
///
/// C assigns the raw `user_data` pointer into each data field; under the
/// token model that is exactly `None` ("pass the integrator's current
/// `user_data` at call time"), so the box itself is not needed here — the
/// argument is kept to preserve the C argument list (and because
/// `ARKodeSetUserData` has taken the box out of `ark_mem` for the
/// duration of the `step_setuserdata` hook).
pub fn arkLSSetUserData(ark_mem: &ARKodeMem, user_data: &mut Option<Box<dyn Any>>) -> i32 {
    let _ = user_data;

    /* access ARKLsMem structure */
    let retval = arkLs_AccessLMem(ark_mem, "arkLSSetUserData");
    if retval != ARKLS_SUCCESS {
        return retval;
    }

    let mut ls = arkls_mem_mut(ark_mem);

    /* Set data for Jacobian */
    if !ls.jacDQ {
        ls.J_data = None;
    }

    /* Set data for Jtimes */
    if !ls.jtimesDQ {
        ls.Jt_data = None;
    }

    /* Set data for LinSys */
    if ls.user_linsys {
        ls.A_data = None;
    }

    /* Set data for Preconditioner */
    ls.P_data = None;

    ARKLS_SUCCESS
}

/// arkLSMassSetUserData sets user_data pointers in arkLSMass (same token
/// treatment as [`arkLSSetUserData`]).
pub fn arkLSSetMassUserData(ark_mem: &ARKodeMem, user_data: &mut Option<Box<dyn Any>>) -> i32 {
    let _ = user_data;

    /* access ARKLsMem structure */
    let retval = arkLs_AccessMassMem(ark_mem, "arkLSSetMassUserData");
    if retval != ARKLS_SUCCESS {
        return retval;
    }

    let mut ls = arkls_mass_mem_mut(ark_mem);

    /* Set data for mass matrix */
    if ls.mass.is_some() {
        ls.M_data = None;
    }

    /* Data for Mtimes is set in arkLSSetMassTimes */

    /* Set data for Preconditioner */
    ls.P_data = None;

    ARKLS_SUCCESS
}

/*---------------------------------------------------------------
  arkLsATimes:

  This routine generates the matrix-vector product z = Av, where
  A = M - gamma*J. The product M*v is obtained either by calling
  the mtimes routine or by just using v (if M=I).  The product
  J*v is obtained by calling the jtimes routine. It is then scaled
  by -gamma and added to M*v to obtain A*v. The return value is
  the same as the values returned by jtimes and mtimes --
  0 if successful, nonzero otherwise.
  ---------------------------------------------------------------*/
pub fn arkLsATimes(arkode_mem: &mut Option<Box<dyn Any>>, v: &N_Vector, z: &N_Vector) -> i32 {
    /* access ARKodeMem and ARKLsMem structures */
    let ark_mem = match arkLs_AccessARKODELMemToken(arkode_mem, "arkLsATimes") {
        Ok(m) => m,
        Err(retval) => return retval,
    };

    /* Access mass matrix solver (if it exists) */
    let step_getmassmem = ark_mem.borrow().step_getmassmem;
    let ark_step_massmem = match step_getmassmem {
        Some(step_getmassmem) => step_getmassmem(&ark_mem),
        None => SUNFALSE,
    };

    /* get gamma values from time step module */
    let step_getgammas = ark_mem.borrow().step_getgammas.expect("step_getgammas");
    let mut gamma: sunrealtype = ZERO;
    let mut gamrat: sunrealtype = ZERO;
    let mut jcur: Option<ARKJcurPtr> = None;
    let mut dgamma_fail: sunbooleantype = SUNFALSE;
    let retval = step_getgammas(
        &ark_mem,
        &mut gamma,
        &mut gamrat,
        &mut jcur,
        &mut dgamma_fail,
    );
    if retval != ARK_SUCCESS {
        arkProcessError(
            Some(&ark_mem),
            retval,
            line!() as i32,
            "arkLsATimes",
            file!(),
            "An error occurred in ark_step_getgammas",
        );
        return retval;
    }

    /* call Jacobian-times-vector product routine
       (either user-supplied or internal DQ) */
    let (jtimes, tcur, ycur, fcur, ytemp) = {
        let ls = arkls_mem_mut(&ark_mem);
        (
            ls.jtimes.expect("jtimes"),
            ls.tcur,
            ls.ycur.as_ref().expect("ycur").clone(),
            ls.fcur.as_ref().expect("fcur").clone(),
            ls.ytemp.as_ref().expect("ytemp").clone(),
        )
    };
    let use_field = arkls_mem_mut(&ark_mem).Jt_data.is_some();
    let mut jt_data = if use_field {
        arkls_mem_mut(&ark_mem).Jt_data.take()
    } else {
        ark_mem.borrow_mut().user_data.take()
    };
    let retval = jtimes(v, z, tcur, &ycur, &fcur, &mut jt_data, &ytemp);
    if use_field {
        arkls_mem_mut(&ark_mem).Jt_data = jt_data;
    } else {
        ark_mem.borrow_mut().user_data = jt_data;
    }
    arkls_mem_mut(&ark_mem).njtimes += 1;
    if retval != 0 {
        return retval;
    }

    /* Compute mass matrix vector product and add to result */
    if ark_step_massmem {
        let retval = arkLsMTimes(&ark_mem, v, &ytemp);
        if retval != 0 {
            return retval;
        }
        N_VLinearSum(ONE, &ytemp, -gamma, z, z);
    } else {
        N_VLinearSum(ONE, v, -gamma, z, z);
    }

    0
}

/*---------------------------------------------------------------
  arkLsPSetup:

  This routine interfaces between the generic iterative linear
  solvers and the user's psetup routine.  It passes to psetup all
  required state information from arkode_mem.  Its return value
  is the same as that returned by psetup. Note that the generic
  iterative linear solvers guarantee that arkLsPSetup will only
  be called in the case that the user's psetup routine is non-NULL.
  ---------------------------------------------------------------*/
pub fn arkLsPSetup(arkode_mem: &mut Option<Box<dyn Any>>) -> i32 {
    /* access ARKodeMem and ARKLsMem structures */
    let ark_mem = match arkLs_AccessARKODELMemToken(arkode_mem, "arkLsPSetup") {
        Ok(m) => m,
        Err(retval) => return retval,
    };

    /* get gamma values from time step module */
    let step_getgammas = ark_mem.borrow().step_getgammas.expect("step_getgammas");
    let mut gamma: sunrealtype = ZERO;
    let mut gamrat: sunrealtype = ZERO;
    let mut jcur: Option<ARKJcurPtr> = None;
    let mut dgamma_fail: sunbooleantype = SUNFALSE;
    let retval = step_getgammas(
        &ark_mem,
        &mut gamma,
        &mut gamrat,
        &mut jcur,
        &mut dgamma_fail,
    );
    if retval != ARK_SUCCESS {
        arkProcessError(
            Some(&ark_mem),
            retval,
            line!() as i32,
            "arkLsPSetup",
            file!(),
            "An error occurred in ark_step_getgammas",
        );
        return retval;
    }

    /* Call user pset routine to update preconditioner and possibly
       reset jcur (pass !jbad as update suggestion). `jcur` is the shared
       cell that `arkLsSetup` was handed as `jcurPtr`: copy out, call,
       write back, so `arkLsSetup` observes the user's update. */
    let (pset, tcur, ycur, fcur, jbad) = {
        let ls = arkls_mem_mut(&ark_mem);
        (
            ls.pset.expect("pset"),
            ls.tcur,
            ls.ycur.as_ref().expect("ycur").clone(),
            ls.fcur.as_ref().expect("fcur").clone(),
            ls.jbad,
        )
    };
    let jcur_cell = jcur.expect("jcur");
    let mut jcur_flag = jcur_cell.get();
    let use_field = arkls_mem_mut(&ark_mem).P_data.is_some();
    let mut p_data = if use_field {
        arkls_mem_mut(&ark_mem).P_data.take()
    } else {
        ark_mem.borrow_mut().user_data.take()
    };
    let retval = pset(
        tcur,
        &ycur,
        &fcur,
        !jbad,
        &mut jcur_flag,
        gamma,
        &mut p_data,
    );
    if use_field {
        arkls_mem_mut(&ark_mem).P_data = p_data;
    } else {
        ark_mem.borrow_mut().user_data = p_data;
    }
    jcur_cell.set(jcur_flag);
    retval
}

/*---------------------------------------------------------------
  arkLsPSolve:

  This routine interfaces between the generic SUNLinSolSolve
  routine and the user's psolve routine.  It passes to psolve all
  required state information from arkode_mem.  Its return value
  is the same as that returned by psolve. Note that the generic
  SUNLinSol solver guarantees that arkLsPSolve will not be
  called in the case in which preconditioning is not done. This
  is the only case in which the user's psolve routine is allowed
  to be NULL.
  ---------------------------------------------------------------*/
pub fn arkLsPSolve(
    arkode_mem: &mut Option<Box<dyn Any>>,
    r: &N_Vector,
    z: &N_Vector,
    tol: sunrealtype,
    lr: i32,
) -> i32 {
    /* access ARKodeMem and ARKLsMem structures */
    let ark_mem = match arkLs_AccessARKODELMemToken(arkode_mem, "arkLsPSolve") {
        Ok(m) => m,
        Err(retval) => return retval,
    };

    /* get gamma values from time step module */
    let step_getgammas = ark_mem.borrow().step_getgammas.expect("step_getgammas");
    let mut gamma: sunrealtype = ZERO;
    let mut gamrat: sunrealtype = ZERO;
    let mut jcur: Option<ARKJcurPtr> = None;
    let mut dgamma_fail: sunbooleantype = SUNFALSE;
    let retval = step_getgammas(
        &ark_mem,
        &mut gamma,
        &mut gamrat,
        &mut jcur,
        &mut dgamma_fail,
    );
    if retval != ARK_SUCCESS {
        arkProcessError(
            Some(&ark_mem),
            retval,
            line!() as i32,
            "arkLsPSolve",
            file!(),
            "An error occurred in ark_step_getgammas",
        );
        return retval;
    }

    /* call the user-supplied psolve routine, and accumulate count */
    let (psolve, tcur, ycur, fcur) = {
        let ls = arkls_mem_mut(&ark_mem);
        (
            ls.psolve.expect("psolve"),
            ls.tcur,
            ls.ycur.as_ref().expect("ycur").clone(),
            ls.fcur.as_ref().expect("fcur").clone(),
        )
    };
    let use_field = arkls_mem_mut(&ark_mem).P_data.is_some();
    let mut p_data = if use_field {
        arkls_mem_mut(&ark_mem).P_data.take()
    } else {
        ark_mem.borrow_mut().user_data.take()
    };
    let retval = psolve(tcur, &ycur, &fcur, r, z, gamma, tol, lr, &mut p_data);
    if use_field {
        arkls_mem_mut(&ark_mem).P_data = p_data;
    } else {
        ark_mem.borrow_mut().user_data = p_data;
    }
    arkls_mem_mut(&ark_mem).nps += 1;
    retval
}

/*---------------------------------------------------------------
  arkLsMTimes:

  This routine generates the matrix-vector product z = Mv, where
  M is the system mass matrix, by calling the user-supplied mtimes
  routine. The return value is the same as the value returned
  by mtimes -- 0 if successful, nonzero otherwise.

  C's single `arkLsMTimes(void*, ...)` serves both as the stepper's
  `ARKMassMultFn` and as the mass LS `SUNATimesFn`; those are distinct
  Rust signatures, so the `SUNATimesFn` role lives in
  [`arkLsMTimesATimes`], which unwraps the data token and delegates here.
  ---------------------------------------------------------------*/
pub fn arkLsMTimes(arkode_mem: &ARKodeMem, v: &N_Vector, z: &N_Vector) -> i32 {
    let ark_mem = arkode_mem;

    /* access ARKodeMem and ARKLsMassMem structures */
    let retval = arkLs_AccessARKODEMassMem(ark_mem, "arkLsMTimes");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* perform multiply by either calling the user-supplied routine
       (default), or asking the SUNMatrix to do the multiply */
    let mtimes = arkls_mass_mem_mut(ark_mem).mtimes;
    if let Some(mtimes) = mtimes {
        /* call user-supplied mtimes routine, increment counter and return */
        let tcur = ark_mem.borrow().tcur;
        /* `mt_data` is passed exactly as stored (NULL included) */
        let mut mt_data = arkls_mass_mem_mut(ark_mem).mt_data.take();
        let mretval = mtimes(v, z, tcur, &mut mt_data);
        arkls_mass_mem_mut(ark_mem).mt_data = mt_data;
        if mretval == 0 {
            arkls_mass_mem_mut(ark_mem).nmtimes += 1;
        } else {
            arkProcessError(
                Some(ark_mem),
                mretval,
                line!() as i32,
                "arkLsMTimes",
                file!(),
                "Error in user mass matrix-vector product routine",
            );
        }
        return mretval;
    } else {
        let M = arkls_mass_mem_mut(ark_mem).M.clone();
        if let Some(M) = M {
            /* try to ask SUNMatrix to do the multiply; increment counter and return */
            if M.ops.borrow().matvec.is_some() {
                let mretval = SUNMatMatvec(&M, v, z);
                if mretval == 0 {
                    arkls_mass_mem_mut(ark_mem).nmtimes += 1;
                } else {
                    arkProcessError(
                        Some(ark_mem),
                        mretval,
                        line!() as i32,
                        "arkLsMTimes",
                        file!(),
                        "Error in SUNMatrix mass matrix-vector product routine",
                    );
                }
                return mretval;
            }
        }
    }

    /* if we made it here, then no matrix-vector product is available.
       C reports `retval`, which still holds the ARKLS_SUCCESS returned by
       the access helper above. */
    arkProcessError(
        Some(ark_mem),
        retval,
        line!() as i32,
        "arkLsMTimes",
        file!(),
        "Missing mass matrix-vector product routine",
    );
    -1
}

/// C `arkLsMTimes` in its `SUNATimesFn` role (the mass `SUNLinearSolver`
/// calls it with the `A_data` token registered by
/// `ARKodeSetMassTimes` / `ARKodeSetMassLinearSolver`).
pub fn arkLsMTimesATimes(
    arkode_mem: &mut Option<Box<dyn Any>>,
    v: &N_Vector,
    z: &N_Vector,
) -> i32 {
    let ark_mem = match arkode_mem
        .as_ref()
        .and_then(|b| b.downcast_ref::<ARKodeMem>())
        .cloned()
    {
        Some(m) => m,
        None => {
            arkProcessError(
                None,
                ARKLS_MEM_NULL,
                line!() as i32,
                "arkLsMTimes",
                file!(),
                MSG_LS_ARKMEM_NULL,
            );
            return ARKLS_MEM_NULL;
        }
    };
    arkLsMTimes(&ark_mem, v, z)
}

/*---------------------------------------------------------------
  arkLsMPSetup:

  This routine interfaces between the generic linear solver and
  the user's mass matrix psetup routine.  It passes to psetup all
  required state information from arkode_mem.  Its return value
  is the same as that returned by psetup.  Note that the generic
  linear solvers guarantee that arkLsMPSetup will only be
  called if the user's psetup routine is non-NULL.
  ---------------------------------------------------------------*/
pub fn arkLsMPSetup(arkode_mem: &mut Option<Box<dyn Any>>) -> i32 {
    /* access ARKodeMem and ARKLsMassMem structures */
    let ark_mem = match arkLs_AccessARKODEMassMemToken(arkode_mem, "arkLsMPSetup") {
        Ok(m) => m,
        Err(retval) => return retval,
    };

    /* only proceed if the mass matrix is time-independent or if
       pset has not been called previously */
    let (time_dependent, npe) = {
        let ls = arkls_mass_mem_mut(&ark_mem);
        (ls.time_dependent, ls.npe)
    };
    if !time_dependent && npe != 0 {
        return 0;
    }

    /* call user-supplied pset routine and increment counter */
    let pset = arkls_mass_mem_mut(&ark_mem).pset.expect("pset");
    let tcur = ark_mem.borrow().tcur;
    let use_field = arkls_mass_mem_mut(&ark_mem).P_data.is_some();
    let mut p_data = if use_field {
        arkls_mass_mem_mut(&ark_mem).P_data.take()
    } else {
        ark_mem.borrow_mut().user_data.take()
    };
    let retval = pset(tcur, &mut p_data);
    if use_field {
        arkls_mass_mem_mut(&ark_mem).P_data = p_data;
    } else {
        ark_mem.borrow_mut().user_data = p_data;
    }
    arkls_mass_mem_mut(&ark_mem).npe += 1;
    retval
}

/*---------------------------------------------------------------
  arkLsMPSolve:

  This routine interfaces between the generic LS routine and the
  user's mass matrix psolve routine.  It passes to psolve all
  required state information from arkode_mem.  Its return value is
  the same as that returned by psolve. Note that the generic
  solver guarantees that arkLsMPSolve will not be called in the
  case in which preconditioning is not done. This is the only case
  in which the user's psolve routine is allowed to be NULL.
  ---------------------------------------------------------------*/
pub fn arkLsMPSolve(
    arkode_mem: &mut Option<Box<dyn Any>>,
    r: &N_Vector,
    z: &N_Vector,
    tol: sunrealtype,
    lr: i32,
) -> i32 {
    /* access ARKodeMem and ARKLsMassMem structures */
    let ark_mem = match arkLs_AccessARKODEMassMemToken(arkode_mem, "arkLsMPSolve") {
        Ok(m) => m,
        Err(retval) => return retval,
    };

    /* call the user-supplied psolve routine, and accumulate count */
    let psolve = arkls_mass_mem_mut(&ark_mem).psolve.expect("psolve");
    let tcur = ark_mem.borrow().tcur;
    let use_field = arkls_mass_mem_mut(&ark_mem).P_data.is_some();
    let mut p_data = if use_field {
        arkls_mass_mem_mut(&ark_mem).P_data.take()
    } else {
        ark_mem.borrow_mut().user_data.take()
    };
    let retval = psolve(tcur, r, z, tol, lr, &mut p_data);
    if use_field {
        arkls_mass_mem_mut(&ark_mem).P_data = p_data;
    } else {
        ark_mem.borrow_mut().user_data = p_data;
    }
    arkls_mass_mem_mut(&ark_mem).nps += 1;
    retval
}

/*---------------------------------------------------------------
  arkLsDQJac:

  This routine is a wrapper for the Dense and Band
  implementations of the difference quotient Jacobian
  approximation routines.
  ---------------------------------------------------------------*/
#[allow(clippy::too_many_arguments)]
pub fn arkLsDQJac(
    t: sunrealtype,
    y: &N_Vector,
    fy: &N_Vector,
    Jac: &SUNMatrix,
    arkode_mem: &mut Option<Box<dyn Any>>,
    tmp1: &N_Vector,
    tmp2: &N_Vector,
    tmp3: &N_Vector,
) -> i32 {
    let _ = tmp3; /* SUNDIALS_MAYBE_UNUSED */

    /* access ARKodeMem and ARKLsMem structures */
    let ark_mem = match arkLs_AccessARKODELMemToken(arkode_mem, "arkLsDQJac") {
        Ok(m) => m,
        Err(retval) => return retval,
    };

    /* Jac non-NULL check: handled by the type system */

    /* Access implicit RHS function */
    let step_getimplicitrhs = ark_mem
        .borrow()
        .step_getimplicitrhs
        .expect("step_getimplicitrhs");
    let fi = step_getimplicitrhs(&ark_mem);
    let fi = match fi {
        Some(fi) => fi,
        None => {
            arkProcessError(
                Some(&ark_mem),
                ARKLS_ILL_INPUT,
                line!() as i32,
                "arkLsDQJac",
                file!(),
                "Time step module is missing implicit RHS fcn",
            );
            return ARKLS_ILL_INPUT;
        }
    };

    /* Verify that N_Vector supports required routines */
    let tempv1 = ark_mem
        .borrow()
        .tempv1
        .as_ref()
        .expect("tempv1")
        .clone();
    {
        let ops = tempv1.ops.borrow();
        if ops.nvcloneempty.is_none()
            || ops.nvwrmsnorm.is_none()
            || ops.nvlinearsum.is_none()
            || ops.nvdestroy.is_none()
            || ops.nvscale.is_none()
            || ops.nvgetarraypointer.is_none()
            || ops.nvsetarraypointer.is_none()
        {
            drop(ops);
            arkProcessError(
                Some(&ark_mem),
                ARKLS_ILL_INPUT,
                line!() as i32,
                "arkLsDQJac",
                file!(),
                MSG_LS_BAD_NVECTOR,
            );
            return ARKLS_ILL_INPUT;
        }
    }

    /* Call the matrix-structure-specific DQ approximation routine */
    let retval;
    if SUNMatGetID(Jac) == SUNMATRIX_DENSE {
        retval = arkLsDenseDQJac(t, y, fy, Jac, &ark_mem, fi, tmp1);
    } else if SUNMatGetID(Jac) == SUNMATRIX_BAND {
        retval = arkLsBandDQJac(t, y, fy, Jac, &ark_mem, fi, tmp1, tmp2);
    } else {
        arkProcessError(
            Some(&ark_mem),
            ARKLS_ILL_INPUT,
            line!() as i32,
            "arkLsDQJac",
            file!(),
            "arkLsDQJac not implemented for this SUNMatrix type!",
        );
        retval = ARKLS_ILL_INPUT;
    }
    retval
}

/*---------------------------------------------------------------
  arkLsDenseDQJac:

  This routine generates a dense difference quotient approximation
  to the Jacobian of f(t,y). It assumes a dense SUNMatrix input
  (stored column-wise, and that elements within each column are
  contiguous). The jth column of J is computed into the `jthCol`
  vector via N_VLinearSum and written back into the matrix column
  (the C code aliases the column memory with N_VSetArrayPointer;
  the copy-in/copy-out here is bit-identical).

  C's `ARKLsMem arkls_mem` parameter is dropped: it is exactly
  `arkls_mem_mut(ark_mem)`, and holding that guard across the RHS
  callback would violate the borrow discipline, so `nfeDQ` is bumped
  through the accessor instead.
  ---------------------------------------------------------------*/
pub fn arkLsDenseDQJac(
    t: sunrealtype,
    y: &N_Vector,
    fy: &N_Vector,
    Jac: &SUNMatrix,
    ark_mem: &ARKodeMem,
    fi: ARKRhsFn,
    tmp1: &N_Vector,
) -> i32 {
    let mut retval: i32 = 0;

    /* access matrix dimension */
    let N = SUNDenseMatrix_Columns(Jac);

    /* Rename work vector for readability */
    let ftemp = tmp1;

    /* Create an empty vector for matrix column calculations */
    let jthCol = N_VCloneEmpty(tmp1).expect("N_VCloneEmpty");

    /* Obtain integrator state (C caches raw data pointers; here the data
    borrows are re-taken per use and never held across the RHS callback
    or a vector op) */
    let (uround, h, ewt, rwt, constraints) = {
        let m = ark_mem.borrow();
        (
            m.uround,
            m.h,
            m.ewt.as_ref().expect("ewt").clone(),
            m.rwt.as_ref().expect("rwt").clone(),
            m.constraints.clone(),
        )
    };

    /* Set minimum increment based on uround and norm of f */
    let srur = SUNRsqrt(uround);
    let fnorm = N_VWrmsNorm(fy, &rwt);
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

        /* call the user-supplied pre-RHS function (if supplied), then call RHS */
        let PreRhsFn = ark_mem.borrow().PreRhsFn;
        if let Some(PreRhsFn) = PreRhsFn {
            let mut user_data = ark_mem.borrow_mut().user_data.take();
            let prretval = PreRhsFn(t, y, &mut user_data);
            ark_mem.borrow_mut().user_data = user_data;
            if prretval != 0 {
                return ARK_PRERHSFN_FAIL;
            }
        }
        let mut user_data = ark_mem.borrow_mut().user_data.take();
        retval = fi(t, y, ftemp, &mut user_data);
        ark_mem.borrow_mut().user_data = user_data;
        arkls_mem_mut(ark_mem).nfeDQ += 1;
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

/*---------------------------------------------------------------
  arkLsBandDQJac:

  This routine generates a banded difference quotient approximation
  to the Jacobian of f(t,y).  It assumes a band SUNMatrix input
  (stored column-wise, and that elements within each column are
  contiguous).

  As with [`arkLsDenseDQJac`], C's `ARKLsMem arkls_mem` parameter is
  dropped in favour of `arkls_mem_mut(ark_mem)` at the one use site.
  ---------------------------------------------------------------*/
#[allow(clippy::too_many_arguments)]
pub fn arkLsBandDQJac(
    t: sunrealtype,
    y: &N_Vector,
    fy: &N_Vector,
    Jac: &SUNMatrix,
    ark_mem: &ARKodeMem,
    fi: ARKRhsFn,
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
    let (uround, h, ewt, rwt, constraints) = {
        let m = ark_mem.borrow();
        (
            m.uround,
            m.h,
            m.ewt.as_ref().expect("ewt").clone(),
            m.rwt.as_ref().expect("rwt").clone(),
            m.constraints.clone(),
        )
    };

    /* Load ytemp with y = predicted y vector */
    N_VScale(ONE, y, ytemp);

    /* Set minimum increment based on uround and norm of f */
    let srur = SUNRsqrt(uround);
    let fnorm = N_VWrmsNorm(fy, &rwt);
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

        /* call the user-supplied pre-RHS function (if supplied), then call RHS */
        let PreRhsFn = ark_mem.borrow().PreRhsFn;
        if let Some(PreRhsFn) = PreRhsFn {
            let mut user_data = ark_mem.borrow_mut().user_data.take();
            let prretval = PreRhsFn(t, ytemp, &mut user_data);
            ark_mem.borrow_mut().user_data = user_data;
            if prretval != 0 {
                return ARK_PRERHSFN_FAIL;
            }
        }
        let mut user_data = ark_mem.borrow_mut().user_data.take();
        retval = fi(t, ytemp, ftemp, &mut user_data);
        ark_mem.borrow_mut().user_data = user_data;
        arkls_mem_mut(ark_mem).nfeDQ += 1;
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

/*---------------------------------------------------------------
  arkLsDQJtimes:

  This routine generates a difference quotient approximation to
  the Jacobian-vector product fi_y(t,y) * v. The approximation is
  Jv = [fi(y + v*sig) - fi(y)]/sig, where sig = 1 / ||v||_WRMS,
  i.e. the WRMS norm of v*sig is 1.
  ---------------------------------------------------------------*/
pub fn arkLsDQJtimes(
    v: &N_Vector,
    Jv: &N_Vector,
    t: sunrealtype,
    y: &N_Vector,
    fy: &N_Vector,
    arkode_mem: &mut Option<Box<dyn Any>>,
    work: &N_Vector,
) -> i32 {
    /* access ARKodeMem and ARKLsMem structures */
    let ark_mem = match arkLs_AccessARKODELMemToken(arkode_mem, "arkLsDQJtimes") {
        Ok(m) => m,
        Err(retval) => return retval,
    };

    /* Initialize perturbation to 1/||v|| */
    let ewt = ark_mem.borrow().ewt.as_ref().expect("ewt").clone();
    let mut sig = ONE / N_VWrmsNorm(v, &ewt);

    /* deviation class 7: C re-reads `arkls_mem->Jt_f` on every retry */
    let Jt_f = arkls_mem_mut(&ark_mem).Jt_f.expect("Jt_f");

    let mut retval: i32 = 0;
    let mut iter: i32 = 0;
    while iter < MAX_DQITERS {
        /* Set work = y + sig*v */
        N_VLinearSum(sig, v, ONE, y, work);

        /* Set Jv = f(tn, y+sig*v), after calling pre-RHS function (if supplied) */
        let PreRhsFn = ark_mem.borrow().PreRhsFn;
        if let Some(PreRhsFn) = PreRhsFn {
            let mut user_data = ark_mem.borrow_mut().user_data.take();
            let prretval = PreRhsFn(t, work, &mut user_data);
            ark_mem.borrow_mut().user_data = user_data;
            if prretval != 0 {
                return ARK_PRERHSFN_FAIL;
            }
        }
        let mut user_data = ark_mem.borrow_mut().user_data.take();
        retval = Jt_f(t, work, Jv, &mut user_data);
        ark_mem.borrow_mut().user_data = user_data;
        arkls_mem_mut(&ark_mem).nfeDQ += 1;
        if retval == 0 {
            break;
        }
        if retval < 0 {
            return -1;
        }

        /* If fi failed recoverably, shrink sig and retry */
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
  arkLsLinSys

  Setup the linear system A = I - gamma J or A = M - gamma J
  -----------------------------------------------------------------*/
#[allow(clippy::too_many_arguments)]
fn arkLsLinSys(
    t: sunrealtype,
    y: &N_Vector,
    fy: &N_Vector,
    A: &SUNMatrix,
    M: Option<&SUNMatrix>,
    jok: sunbooleantype,
    jcur: &mut sunbooleantype,
    gamma: sunrealtype,
    arkode_mem: &mut Option<Box<dyn Any>>,
    vtemp1: &N_Vector,
    vtemp2: &N_Vector,
    vtemp3: &N_Vector,
) -> i32 {
    /* access ARKodeMem and ARKLsMem structures */
    let ark_mem = match arkLs_AccessARKODELMemToken(arkode_mem, "arkLsLinSys") {
        Ok(m) => m,
        Err(retval) => return retval,
    };

    /* Check if Jacobian needs to be updated */
    if jok {
        /* Use saved copy of J */
        *jcur = SUNFALSE;

        /* Overwrite linear system matrix with saved J */
        let savedJ = arkls_mem_mut(&ark_mem)
            .savedJ
            .as_ref()
            .expect("savedJ")
            .clone();
        let retval = SUNMatCopy(&savedJ, A);
        if retval != 0 {
            arkProcessError(
                Some(&ark_mem),
                ARKLS_SUNMAT_FAIL,
                line!() as i32,
                "arkLsLinSys",
                file!(),
                MSG_LS_SUNMAT_FAILED,
            );
            arkls_mem_mut(&ark_mem).last_flag = ARKLS_SUNMAT_FAIL;
            return ARKLS_SUNMAT_FAIL;
        }
    } else {
        /* Call jac() routine to update J */
        *jcur = SUNTRUE;

        /* Clear the linear system matrix if necessary (direct linear solvers) */
        if !arkls_mem_mut(&ark_mem).iterative {
            let retval = SUNMatZero(A);
            if retval != 0 {
                arkProcessError(
                    Some(&ark_mem),
                    ARKLS_SUNMAT_FAIL,
                    line!() as i32,
                    "arkLsLinSys",
                    file!(),
                    MSG_LS_SUNMAT_FAILED,
                );
                arkls_mem_mut(&ark_mem).last_flag = ARKLS_SUNMAT_FAIL;
                return ARKLS_SUNMAT_FAIL;
            }
        }

        /* Compute new Jacobian matrix */
        let jac = arkls_mem_mut(&ark_mem).jac.expect("jac");
        let use_field = arkls_mem_mut(&ark_mem).J_data.is_some();
        let mut j_data = if use_field {
            arkls_mem_mut(&ark_mem).J_data.take()
        } else {
            ark_mem.borrow_mut().user_data.take()
        };
        let retval = jac(t, y, fy, A, &mut j_data, vtemp1, vtemp2, vtemp3);
        if use_field {
            arkls_mem_mut(&ark_mem).J_data = j_data;
        } else {
            ark_mem.borrow_mut().user_data = j_data;
        }
        if retval < 0 {
            arkProcessError(
                Some(&ark_mem),
                ARKLS_JACFUNC_UNRECVR,
                line!() as i32,
                "arkLsLinSys",
                file!(),
                MSG_LS_JACFUNC_FAILED,
            );
            arkls_mem_mut(&ark_mem).last_flag = ARKLS_JACFUNC_UNRECVR;
            return -1;
        }
        if retval > 0 {
            arkls_mem_mut(&ark_mem).last_flag = ARKLS_JACFUNC_RECVR;
            return 1;
        }

        /* Update saved copy of the Jacobian matrix */
        let savedJ = arkls_mem_mut(&ark_mem)
            .savedJ
            .as_ref()
            .expect("savedJ")
            .clone();
        let retval = SUNMatCopy(A, &savedJ);
        if retval != 0 {
            arkProcessError(
                Some(&ark_mem),
                ARKLS_SUNMAT_FAIL,
                line!() as i32,
                "arkLsLinSys",
                file!(),
                MSG_LS_SUNMAT_FAILED,
            );
            arkls_mem_mut(&ark_mem).last_flag = ARKLS_SUNMAT_FAIL;
            return ARKLS_SUNMAT_FAIL;
        }
    }

    /* Perform linear combination A = I - gamma*J or A = M - gamma*J */
    let retval = match M {
        None => SUNMatScaleAddI(-gamma, A),
        Some(M) => SUNMatScaleAdd(-gamma, A, M),
    };

    /* Check matrix operation return value */
    if retval != 0 {
        arkProcessError(
            Some(&ark_mem),
            ARKLS_SUNMAT_FAIL,
            line!() as i32,
            "arkLsLinSys",
            file!(),
            MSG_LS_SUNMAT_FAILED,
        );
        arkls_mem_mut(&ark_mem).last_flag = ARKLS_SUNMAT_FAIL;
        return ARKLS_SUNMAT_FAIL;
    }

    ARKLS_SUCCESS
}

/*---------------------------------------------------------------
  arkLsInitialize performs remaining initializations specific
  to the linear solver interface (and solver itself)
  ---------------------------------------------------------------*/
pub fn arkLsInitialize(ark_mem: &ARKodeMem) -> i32 {
    /* access ARKLsMem structure */
    let retval = arkLs_AccessLMem(ark_mem, "arkLsInitialize");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* access ARKLsMassMem (if applicable) */
    let mut have_massmem = SUNFALSE;
    let step_getmassmem = ark_mem.borrow().step_getmassmem;
    if let Some(step_getmassmem) = step_getmassmem {
        if step_getmassmem(ark_mem) {
            let retval = arkLs_AccessMassMem(ark_mem, "arkLsInitialize");
            if retval != ARK_SUCCESS {
                return retval;
            }
            have_massmem = SUNTRUE;
        }
    }

    /* Test for valid combinations of matrix & Jacobian routines: */
    let A = arkls_mem_mut(ark_mem).A.clone();
    if let Some(A) = &A {
        /* Matrix-based case */

        if !arkls_mem_mut(ark_mem).user_linsys {
            /* Internal linear system function, reset pointers (just in case) */
            {
                let A_data: Box<dyn Any> = Box::new(ark_mem.clone());
                let mut ls = arkls_mem_mut(ark_mem);
                ls.linsys = Some(arkLsLinSys);
                ls.A_data = Some(A_data); /* C: A_data = ark_mem */
            }

            /* Check if an internal or user-supplied Jacobian function is used */
            if arkls_mem_mut(ark_mem).jacDQ {
                /* Internal difference quotient Jacobian. Check that A is dense or band,
                   otherwise return an error */
                let mut retval = 0;
                if A.ops.borrow().getid.is_some() {
                    let id = SUNMatGetID(A);
                    if (id == SUNMATRIX_DENSE) || (id == SUNMATRIX_BAND) {
                        let J_data: Box<dyn Any> = Box::new(ark_mem.clone());
                        let mut ls = arkls_mem_mut(ark_mem);
                        ls.jac = Some(arkLsDQJac);
                        ls.J_data = Some(J_data); /* C: J_data = ark_mem */
                    } else {
                        retval += 1;
                    }
                } else {
                    retval += 1;
                }
                if retval != 0 {
                    arkProcessError(
                        Some(ark_mem),
                        ARKLS_ILL_INPUT,
                        line!() as i32,
                        "arkLsInitialize",
                        file!(),
                        "No Jacobian constructor available for SUNMatrix type",
                    );
                    arkls_mem_mut(ark_mem).last_flag = ARKLS_ILL_INPUT;
                    return ARKLS_ILL_INPUT;
                }
            }

            /* Allocate internally saved Jacobian if not already done */
            if arkls_mem_mut(ark_mem).savedJ.is_none() {
                match SUNMatClone(A) {
                    Some(savedJ) => arkls_mem_mut(ark_mem).savedJ = Some(savedJ),
                    None => {
                        arkProcessError(
                            Some(ark_mem),
                            ARKLS_MEM_FAIL,
                            line!() as i32,
                            "arkLsInitialize",
                            file!(),
                            MSG_LS_MEM_FAIL,
                        );
                        arkls_mem_mut(ark_mem).last_flag = ARKLS_MEM_FAIL;
                        return ARKLS_MEM_FAIL;
                    }
                }
            }
        } /* end matrix-based case */
    } else {
        /* Matrix-free case: ensure 'jac' and 'linsys' function pointers are NULL */
        let mut ls = arkls_mem_mut(ark_mem);
        ls.jacDQ = SUNFALSE;
        ls.jac = None;
        ls.J_data = None;

        ls.user_linsys = SUNFALSE;
        ls.linsys = None;
        ls.A_data = None;
    }

    /* Test for valid combination of system matrix and mass matrix (if applicable) */
    if have_massmem {
        let M = arkls_mass_mem_mut(ark_mem).M.clone();

        /* A and M must both be NULL or non-NULL */
        if A.is_none() != M.is_none() {
            arkProcessError(
                Some(ark_mem),
                ARKLS_ILL_INPUT,
                line!() as i32,
                "arkLsInitialize",
                file!(),
                "Cannot combine NULL and non-NULL System and mass matrices",
            );
            arkls_mem_mut(ark_mem).last_flag = ARKLS_ILL_INPUT;
            return ARKLS_ILL_INPUT;
        }

        /* If A is non-NULL, A and M must have matching types (if accessible) */
        if let Some(A) = &A {
            let M = M.as_ref().expect("mass matrix M");
            let mut retval = 0;
            if A.ops.borrow().getid.is_none() != M.ops.borrow().getid.is_none() {
                retval += 1;
            }
            if A.ops.borrow().getid.is_some() && SUNMatGetID(A) != SUNMatGetID(M) {
                retval += 1;
            }
            if retval != 0 {
                arkProcessError(
                    Some(ark_mem),
                    ARKLS_ILL_INPUT,
                    line!() as i32,
                    "arkLsInitialize",
                    file!(),
                    "System and mass matrices have incompatible types",
                );
                arkls_mem_mut(ark_mem).last_flag = ARKLS_ILL_INPUT;
                return ARKLS_ILL_INPUT;
            }
        }

        /* If either system or mass matrix solver is matrix-embedded, then both must be */
        let sys_LS = arkls_mem_mut(ark_mem).LS.clone();
        let mass_LS = arkls_mass_mem_mut(ark_mem).LS.clone();
        if (SUNLinSolGetType(&sys_LS) == SUNLINEARSOLVER_MATRIX_EMBEDDED)
            && (SUNLinSolGetType(&mass_LS) != SUNLINEARSOLVER_MATRIX_EMBEDDED)
        {
            arkProcessError(
                Some(ark_mem),
                ARKLS_ILL_INPUT,
                line!() as i32,
                "arkLsInitialize",
                file!(),
                "mismatched matrix-embedded LS types (system and mass must match)",
            );
            arkls_mem_mut(ark_mem).last_flag = ARKLS_ILL_INPUT;
            return ARKLS_ILL_INPUT;
        }
        if (SUNLinSolGetType(&sys_LS) != SUNLINEARSOLVER_MATRIX_EMBEDDED)
            && (SUNLinSolGetType(&mass_LS) == SUNLINEARSOLVER_MATRIX_EMBEDDED)
        {
            arkProcessError(
                Some(ark_mem),
                ARKLS_ILL_INPUT,
                line!() as i32,
                "arkLsInitialize",
                file!(),
                "mismatched matrix-embedded LS types (system and mass must match)",
            );
            arkls_mem_mut(ark_mem).last_flag = ARKLS_ILL_INPUT;
            return ARKLS_ILL_INPUT;
        }
    }

    /* reset counters */
    let _ = arkLsInitializeCounters(&mut arkls_mem_mut(ark_mem));

    /* Set Jacobian-vector product related fields, based on jtimesDQ */
    if arkls_mem_mut(ark_mem).jtimesDQ {
        let Jt_data: Box<dyn Any> = Box::new(ark_mem.clone());
        let mut ls = arkls_mem_mut(ark_mem);
        ls.jtsetup = None;
        ls.jtimes = Some(arkLsDQJtimes);
        ls.Jt_data = Some(Jt_data); /* C: Jt_data = ark_mem */
    }

    /* If A is NULL and psetup is not present, then arkLsSetup does
       not need to be called, so set the lsetup function to NULL (if possible) */
    let (A_is_none, pset_is_none) = {
        let ls = arkls_mem_mut(ark_mem);
        (ls.A.is_none(), ls.pset.is_none())
    };
    let step_disablelsetup = ark_mem.borrow().step_disablelsetup;
    if A_is_none && pset_is_none && step_disablelsetup.is_some() {
        step_disablelsetup.expect("step_disablelsetup")(ark_mem);
    }

    /* When using a matrix-embedded linear solver, disable lsetup call and solution scaling */
    let LS = arkls_mem_mut(ark_mem).LS.clone();
    if SUNLinSolGetType(&LS) == SUNLINEARSOLVER_MATRIX_EMBEDDED {
        let step_disablelsetup = ark_mem
            .borrow()
            .step_disablelsetup
            .expect("step_disablelsetup");
        step_disablelsetup(ark_mem);
        arkls_mem_mut(ark_mem).scalesol = SUNFALSE;
    }

    /* Call LS initialize routine, and return result */
    let last_flag = SUNLinSolInitialize(&LS);
    arkls_mem_mut(ark_mem).last_flag = last_flag;
    last_flag
}

/*---------------------------------------------------------------
  arkLsSetup conditionally calls the LS 'setup' routine.

  When using a SUNMatrix object, this determines whether
  to update a Jacobian matrix (or use a stored version), based
  on heuristics regarding previous convergence issues, the number
  of time steps since it was last updated, etc.; it then creates
  the system matrix from this, the 'gamma' factor and the
  mass/identity matrix,
  A = M-gamma*J.

  This routine then calls the LS 'setup' routine with A.
  ---------------------------------------------------------------*/
#[allow(clippy::too_many_arguments)]
pub fn arkLsSetup(
    ark_mem: &ARKodeMem,
    convfail: i32,
    tpred: sunrealtype,
    ypred: &N_Vector,
    fpred: &N_Vector,
    jcurPtr: &Cell<sunbooleantype>,
    vtemp1: &N_Vector,
    vtemp2: &N_Vector,
    vtemp3: &N_Vector,
) -> i32 {
    /* access ARKLsMem structure */
    let retval = arkLs_AccessLMem(ark_mem, "arkLsSetup");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* Immediately return when using matrix-embedded linear solver */
    let LS = arkls_mem_mut(ark_mem).LS.clone();
    if SUNLinSolGetType(&LS) == SUNLINEARSOLVER_MATRIX_EMBEDDED {
        arkls_mem_mut(ark_mem).last_flag = ARKLS_SUCCESS;
        return ARKLS_SUCCESS;
    }

    /* Set ARKLs time and N_Vector pointers to current time,
       solution and rhs */
    {
        let mut ls = arkls_mem_mut(ark_mem);
        ls.tcur = tpred;
        ls.ycur = Some(ypred.clone());
        ls.fcur = Some(fpred.clone());
    }

    /* get gamma values from time step module */
    let step_getgammas = ark_mem.borrow().step_getgammas.expect("step_getgammas");
    let mut gamma: sunrealtype = ZERO;
    let mut gamrat: sunrealtype = ZERO;
    let mut jcur: Option<ARKJcurPtr> = None;
    let mut dgamma_fail: sunbooleantype = SUNFALSE;
    let last_flag = step_getgammas(
        ark_mem,
        &mut gamma,
        &mut gamrat,
        &mut jcur,
        &mut dgamma_fail,
    );
    arkls_mem_mut(ark_mem).last_flag = last_flag;
    if last_flag != 0 {
        arkProcessError(
            Some(ark_mem),
            last_flag,
            line!() as i32,
            "arkLsSetup",
            file!(),
            "An error occurred in ark_step_getgammas",
        );
        return last_flag;
    }
    let _ = jcur; /* C: `jcur` is unused after this point (jcurPtr aliases it) */

    /* Use initsetup, gamma/gammap, and convfail to set J/P eval. flag jbad;
       Note: the "ARK_FAIL_BAD_J" test is asking whether the nonlinear
       solver converged due to a bad system Jacobian AND our gamma was
       fine, indicating that the J and/or P were invalid */
    let (initsetup, nst) = {
        let m = ark_mem.borrow();
        (m.initsetup, m.nst)
    };
    {
        let mut ls = arkls_mem_mut(ark_mem);
        ls.jbad = initsetup
            || (nst >= ls.nstlj + ls.msbj)
            || ((convfail == ARK_FAIL_BAD_J) && (!dgamma_fail))
            || (convfail == ARK_FAIL_OTHER);
    }

    /* Check for mass matrix module and setup mass matrix */
    let step_getmassmem = ark_mem.borrow().step_getmassmem;
    let ark_step_massmem = match step_getmassmem {
        Some(step_getmassmem) => step_getmassmem(ark_mem),
        None => SUNFALSE,
    };

    let mut M: Option<SUNMatrix> = None;
    if ark_step_massmem {
        /* Set shortcut to the mass matrix (NULL if matrix-free) */
        M = arkls_mass_mem_mut(ark_mem).M.clone();

        /* Setup mass matrix linear solver (including recomputation of mass matrix) */
        let last_flag = arkLsMassSetup(ark_mem, tpred, vtemp1, vtemp2, vtemp3);
        arkls_mem_mut(ark_mem).last_flag = last_flag;
        if last_flag != 0 {
            arkProcessError(
                Some(ark_mem),
                ARKLS_SUNMAT_FAIL,
                line!() as i32,
                "arkLsSetup",
                file!(),
                "Error setting up mass-matrix linear solver",
            );
            return last_flag;
        }
    }

    /* Setup the linear system if necessary */
    let A = arkls_mem_mut(ark_mem).A.clone();
    if let Some(A) = &A {
        /* Update J if appropriate and evaluate A = I-gamma*J or A = M-gamma*J */
        let (linsys, jbad) = {
            let ls = arkls_mem_mut(ark_mem);
            (ls.linsys.expect("linsys"), ls.jbad)
        };
        let use_field = arkls_mem_mut(ark_mem).A_data.is_some();
        let mut a_data = if use_field {
            arkls_mem_mut(ark_mem).A_data.take()
        } else {
            ark_mem.borrow_mut().user_data.take()
        };
        /* jcurPtr is the stepper's shared jcur cell: copy out, call, write
        back before the bookkeeping below reads it (invariant B). */
        let mut jcur_flag = jcurPtr.get();
        let retval = linsys(
            tpred,
            ypred,
            fpred,
            A,
            M.as_ref(),
            !jbad,
            &mut jcur_flag,
            gamma,
            &mut a_data,
            vtemp1,
            vtemp2,
            vtemp3,
        );
        if use_field {
            arkls_mem_mut(ark_mem).A_data = a_data;
        } else {
            ark_mem.borrow_mut().user_data = a_data;
        }
        jcurPtr.set(jcur_flag);

        /* Update J eval count and step when J was last updated */
        if jcurPtr.get() {
            let nst_now = ark_mem.borrow().nst;
            let mut ls = arkls_mem_mut(ark_mem);
            ls.nje += 1;
            ls.nstlj = nst_now;
            ls.tnlj = tpred;
        }

        /* Check linsys() return value and return if necessary */
        if retval != ARKLS_SUCCESS {
            if arkls_mem_mut(ark_mem).user_linsys {
                if retval < 0 {
                    arkProcessError(
                        Some(ark_mem),
                        ARKLS_JACFUNC_UNRECVR,
                        line!() as i32,
                        "arkLsSetup",
                        file!(),
                        MSG_LS_JACFUNC_FAILED,
                    );
                    arkls_mem_mut(ark_mem).last_flag = ARKLS_JACFUNC_UNRECVR;
                    return -1;
                } else {
                    arkls_mem_mut(ark_mem).last_flag = ARKLS_JACFUNC_RECVR;
                    return 1;
                }
            } else {
                return retval;
            }
        }
    } else {
        /* Matrix-free case, set jcur to jbad */
        jcurPtr.set(arkls_mem_mut(ark_mem).jbad);
    }

    /* Call LS setup routine -- the LS may call arkLsPSetup, who will
       pass the heuristic suggestions above to the user code(s) */
    let last_flag = SUNLinSolSetup(&LS, A.as_ref());
    arkls_mem_mut(ark_mem).last_flag = last_flag;

    /* If the SUNMatrix was NULL, update heuristics flags. Note that a user
    psetup reached re-entrantly through SUNLinSolSetup -> arkLsPSetup may
    have written the shared jcur cell, so `jcurPtr.get()` re-reads it. */
    if A.is_none() {
        /* If user set jcur to SUNTRUE, increment npe and save nst value */
        if jcurPtr.get() {
            let nst_now = ark_mem.borrow().nst;
            let mut ls = arkls_mem_mut(ark_mem);
            ls.npe += 1;
            ls.nstlj = nst_now;
            ls.tnlj = tpred;
        }

        /* Update jcurPtr flag if we suggested an update */
        if arkls_mem_mut(ark_mem).jbad {
            jcurPtr.set(SUNTRUE);
        }
    }

    last_flag
}

/*---------------------------------------------------------------
  arkLsSolve: interfaces between ARKODE and the generic
  SUNLinearSolver object LS, by setting the appropriate tolerance
  and scaling vectors, calling the solver, and accumulating
  statistics from the solve for use/reporting by ARKODE.

  When using a non-NULL SUNMatrix, this will additionally scale
  the solution appropriately when gamrat != 1.
  ---------------------------------------------------------------*/
pub fn arkLsSolve(
    ark_mem: &ARKodeMem,
    b: &N_Vector,
    tnow: sunrealtype,
    ynow: &N_Vector,
    fnow: &N_Vector,
    eRNrm: sunrealtype,
    mnewt: i32,
) -> i32 {
    /* access ARKLsMem structure */
    let retval = arkLs_AccessLMem(ark_mem, "arkLsSolve");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* Set scalar tcur and vectors ycur and fcur for use by the
       Atimes and Psolve interface routines */
    {
        let mut ls = arkls_mem_mut(ark_mem);
        ls.tcur = tnow;
        ls.ycur = Some(ynow.clone());
        ls.fcur = Some(fnow.clone());
    }

    let (iterative, eplifac, nrmfac) = {
        let ls = arkls_mem_mut(ark_mem);
        (ls.iterative, ls.eplifac, ls.nrmfac)
    };
    let (rwt, ewt) = {
        let m = ark_mem.borrow();
        (
            m.rwt.as_ref().expect("rwt").clone(),
            m.ewt.as_ref().expect("ewt").clone(),
        )
    };

    /* If the linear solver is iterative:
       test norm(b), if small, return x = 0 or x = b;
       set linear solver tolerance (in left/right scaled 2-norm) */
    let mut delta: sunrealtype;
    if iterative {
        let deltar = eplifac * eRNrm;
        let bnorm = N_VWrmsNorm(b, &rwt);

        if bnorm <= deltar {
            if mnewt > 0 {
                N_VConst(ZERO, b);
            }
            arkls_mem_mut(ark_mem).last_flag = ARKLS_SUCCESS;
            return ARKLS_SUCCESS;
        }
        /* Adjust tolerance for 2-norm */
        delta = deltar * nrmfac;
    } else {
        delta = ZERO;
    }

    let LS = arkls_mem_mut(ark_mem).LS.clone();
    let x = arkls_mem_mut(ark_mem).x.as_ref().expect("x").clone();

    /* Set scaling vectors for LS to use (if applicable) */
    if LS.ops.borrow().setscalingvectors.is_some() {
        let retval = SUNLinSolSetScalingVectors(&LS, Some(&rwt), Some(&ewt));
        if retval != SUN_SUCCESS {
            arkProcessError(
                Some(ark_mem),
                ARKLS_SUNLS_FAIL,
                line!() as i32,
                "arkLsSolve",
                file!(),
                "Error in call to SUNLinSolSetScalingVectors",
            );
            arkls_mem_mut(ark_mem).last_flag = ARKLS_SUNLS_FAIL;
            return ARKLS_SUNLS_FAIL;
        }

        /* If solver is iterative and does not support scaling vectors, update the
         tolerance in an attempt to account for ewt/rwt vectors.  We make the
         following assumptions:
           1. rwt_i = rwt_mean, for i=0,...,n-1 (i.e. the residual units are identical)
           2. the linear solver uses a basic 2-norm to measure convergence
         Hence (using the notation from sunlinsol_spgmr.h, with S = diag(rwt)),
               || bbar - Abar xbar ||_2 < tol
           <=> || S b - S A x ||_2 < tol
           <=> || S (b - A x) ||_2 < tol
           <=> \sum_{i=0}^{n-1} (rwt_i (b - A x)_i)^2 < tol^2
           <=> rwt_mean^2 \sum_{i=0}^{n-1} (b - A x_i)^2 < tol^2
           <=> \sum_{i=0}^{n-1} (b - A x_i)^2 < tol^2 / rwt_mean^2
           <=> || b - A x ||_2 < tol / rwt_mean
         So we compute rwt_mean = ||rwt||_RMS and scale the desired tolerance accordingly. */
    } else if iterative {
        N_VConst(ONE, &x);
        let rwt_mean = N_VWrmsNorm(&rwt, &x);
        delta /= rwt_mean;
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
    let jtsetup = arkls_mem_mut(ark_mem).jtsetup;
    if let Some(jtsetup) = jtsetup {
        let use_field = arkls_mem_mut(ark_mem).Jt_data.is_some();
        let mut jt_data = if use_field {
            arkls_mem_mut(ark_mem).Jt_data.take()
        } else {
            ark_mem.borrow_mut().user_data.take()
        };
        let last_flag = jtsetup(tnow, ynow, fnow, &mut jt_data);
        if use_field {
            arkls_mem_mut(ark_mem).Jt_data = jt_data;
        } else {
            ark_mem.borrow_mut().user_data = jt_data;
        }
        {
            let mut ls = arkls_mem_mut(ark_mem);
            ls.last_flag = last_flag;
            ls.njtsetup += 1;
        }
        if last_flag != 0 {
            arkProcessError(
                Some(ark_mem),
                last_flag,
                line!() as i32,
                "arkLsSolve",
                file!(),
                MSG_LS_JTSETUP_FAILED,
            );
            return last_flag;
        }
    }

    /* Call solver, and copy x to b */
    let A = arkls_mem_mut(ark_mem).A.clone();
    let retval = SUNLinSolSolve(&LS, A.as_ref(), &x, b, delta);
    N_VScale(ONE, &x, b);

    /* If using a direct or matrix-iterative solver, scale the correction to
       account for change in gamma (this is only beneficial if M==I) */
    if arkls_mem_mut(ark_mem).scalesol {
        let step_getgammas = ark_mem.borrow().step_getgammas.expect("step_getgammas");
        let mut gamma: sunrealtype = ZERO;
        let mut gamrat: sunrealtype = ZERO;
        let mut jcur: Option<ARKJcurPtr> = None;
        let mut dgamma_fail: sunbooleantype = SUNFALSE;
        let last_flag = step_getgammas(
            ark_mem,
            &mut gamma,
            &mut gamrat,
            &mut jcur,
            &mut dgamma_fail,
        );
        arkls_mem_mut(ark_mem).last_flag = last_flag;
        if last_flag != ARK_SUCCESS {
            arkProcessError(
                Some(ark_mem),
                last_flag,
                line!() as i32,
                "arkLsSolve",
                file!(),
                "An error occurred in ark_step_getgammas",
            );
            return last_flag;
        }
        if gamrat != ONE {
            N_VScale(TWO / (ONE + gamrat), b, b);
        }
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
        let mut ls = arkls_mem_mut(ark_mem);
        ls.nli += nli_inc as i64;
        if retval != SUN_SUCCESS {
            ls.ncfl += 1;
        }
    }

    /* Interpret solver return value  */
    arkls_mem_mut(ark_mem).last_flag = retval;

    match retval {
        SUN_SUCCESS => 0,
        SUNLS_RES_REDUCED => {
            /* allow reduction but not solution on first nonlinear iteration,
               otherwise return with a recoverable failure */
            if mnewt == 0 {
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
            arkProcessError(
                Some(ark_mem),
                SUN_ERR_EXT_FAIL,
                line!() as i32,
                "arkLsSolve",
                file!(),
                "Failure in SUNLinSol external package",
            );
            -1
        }
        SUNLS_ATIMES_FAIL_UNREC => {
            arkProcessError(
                Some(ark_mem),
                SUNLS_ATIMES_FAIL_UNREC,
                line!() as i32,
                "arkLsSolve",
                file!(),
                MSG_LS_JTIMES_FAILED,
            );
            -1
        }
        SUNLS_PSOLVE_FAIL_UNREC => {
            arkProcessError(
                Some(ark_mem),
                SUNLS_PSOLVE_FAIL_UNREC,
                line!() as i32,
                "arkLsSolve",
                file!(),
                MSG_LS_PSOLVE_FAILED,
            );
            -1
        }
        /* C's switch has no default; control falls through to `return (0)` */
        _ => 0,
    }
}

/*---------------------------------------------------------------
  arkLsFree frees memory associates with the ARKLs system
  solver interface.
  ---------------------------------------------------------------*/
pub fn arkLsFree(ark_mem: &ARKodeMem) -> i32 {
    /* NULL ARKodeMem check: handled by the type system */

    /* Return immediately if ARKLsMem is NULL */
    let step_getlinmem = ark_mem.borrow().step_getlinmem.expect("step_getlinmem");
    if !step_getlinmem(ark_mem) {
        return ARKLS_SUCCESS;
    }

    {
        let mut ls = arkls_mem_mut(ark_mem);

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
    let pfree = arkls_mem_mut(ark_mem).pfree;
    if let Some(pfree) = pfree {
        pfree(ark_mem);
    }

    /* free ARKLs interface structure */
    ark_mem.borrow_mut().ark_lmem = None;

    ARKLS_SUCCESS
}

/*---------------------------------------------------------------
  arkLsMassInitialize performs remaining initializations specific
  to the mass matrix solver interface (and solver itself)
  ---------------------------------------------------------------*/
pub fn arkLsMassInitialize(ark_mem: &ARKodeMem) -> i32 {
    /* access ARKLsMassMem structure */
    let retval = arkLs_AccessMassMem(ark_mem, "arkLsMassInitialize");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* reset counters */
    let _ = arkLsInitializeMassCounters(&mut arkls_mass_mem_mut(ark_mem));

    /* perform checks for matrix-based mass system */
    let (M, mass_is_none, mtimes_is_none) = {
        let ls = arkls_mass_mem_mut(ark_mem);
        (ls.M.clone(), ls.mass.is_none(), ls.mtimes.is_none())
    };
    if let Some(M) = &M {
        /* check for user-provided mass matrix constructor */
        if mass_is_none {
            arkProcessError(
                Some(ark_mem),
                ARKLS_ILL_INPUT,
                line!() as i32,
                "arkLsMassInitialize",
                file!(),
                "Missing user-provided mass-matrix routine",
            );
            arkls_mass_mem_mut(ark_mem).last_flag = ARKLS_ILL_INPUT;
            return ARKLS_ILL_INPUT;
        }
        /* check that someone can perform matrix-vector product */
        if mtimes_is_none && M.ops.borrow().matvec.is_none() {
            arkProcessError(
                Some(ark_mem),
                ARKLS_ILL_INPUT,
                line!() as i32,
                "arkLsMassInitialize",
                file!(),
                "No available mass matrix-vector product routine",
            );
            arkls_mass_mem_mut(ark_mem).last_flag = ARKLS_ILL_INPUT;
            return ARKLS_ILL_INPUT;
        }
    }

    /* perform checks for matrix-free mass system */
    let LS = arkls_mass_mem_mut(ark_mem).LS.clone();
    if M.is_none()
        && mtimes_is_none
        && (SUNLinSolGetType(&LS) != SUNLINEARSOLVER_MATRIX_EMBEDDED)
    {
        arkProcessError(
            Some(ark_mem),
            ARKLS_ILL_INPUT,
            line!() as i32,
            "arkLsMassInitialize",
            file!(),
            "Missing user-provided mass matrix-vector product routine",
        );
        arkls_mass_mem_mut(ark_mem).last_flag = ARKLS_ILL_INPUT;
        return ARKLS_ILL_INPUT;
    }

    /* ensure that a mass matrix solver exists (C: `arkls_mem->LS == NULL`;
    the Rust field is non-optional, so this check cannot fire) */

    /* if M is NULL and neither pset or mtsetup are present, then
       arkLsMassSetup does not need to be called, so set the
       msetup function to NULL */
    let (pset_is_none, mtsetup_is_none) = {
        let ls = arkls_mass_mem_mut(ark_mem);
        (ls.pset.is_none(), ls.mtsetup.is_none())
    };
    let step_disablemsetup = ark_mem.borrow().step_disablemsetup;
    if M.is_none() && pset_is_none && mtsetup_is_none && step_disablemsetup.is_some() {
        step_disablemsetup.expect("step_disablemsetup")(ark_mem);
    }

    /* When using a matrix-embedded linear solver, disable lsetup call */
    if SUNLinSolGetType(&LS) == SUNLINEARSOLVER_MATRIX_EMBEDDED {
        let step_disablemsetup = ark_mem
            .borrow()
            .step_disablemsetup
            .expect("step_disablemsetup");
        step_disablemsetup(ark_mem);
    }

    /* Call LS initialize routine */
    let last_flag = SUNLinSolInitialize(&LS);
    arkls_mass_mem_mut(ark_mem).last_flag = last_flag;
    last_flag
}

/*---------------------------------------------------------------
  arkLsMassSetup calls the LS 'setup' routine.
  ---------------------------------------------------------------*/
pub fn arkLsMassSetup(
    ark_mem: &ARKodeMem,
    t: sunrealtype,
    vtemp1: &N_Vector,
    vtemp2: &N_Vector,
    vtemp3: &N_Vector,
) -> i32 {
    /* access ARKLsMassMem structure */
    let retval = arkLs_AccessMassMem(ark_mem, "arkLsMassSetup");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* Immediately return when using matrix-embedded linear solver */
    let LS = arkls_mass_mem_mut(ark_mem).LS.clone();
    if SUNLinSolGetType(&LS) == SUNLINEARSOLVER_MATRIX_EMBEDDED {
        arkls_mass_mem_mut(ark_mem).last_flag = ARKLS_SUCCESS;
        return ARKLS_SUCCESS;
    }

    /* if the most recent setup essentially matches the current time,
       just return with success */
    let uround = ark_mem.borrow().uround;
    if SUNRabs(arkls_mass_mem_mut(ark_mem).msetuptime - t) < FUZZ_FACTOR * uround {
        arkls_mass_mem_mut(ark_mem).last_flag = ARKLS_SUCCESS;
        return ARKLS_SUCCESS;
    }

    /* Determine whether to call user-provided mtsetup routine */
    let mut call_mtsetup = SUNFALSE;
    let (mtsetup, time_dependent, nmtsetup) = {
        let ls = arkls_mass_mem_mut(ark_mem);
        (ls.mtsetup, ls.time_dependent, ls.nmtsetup)
    };
    if mtsetup.is_some() && (time_dependent || (nmtsetup == 0)) {
        call_mtsetup = SUNTRUE;
    }

    /* call user-provided mtsetup routine if applicable */
    if call_mtsetup {
        let mtsetup = mtsetup.expect("mtsetup");
        /* `mt_data` is passed exactly as stored (NULL included) */
        let mut mt_data = arkls_mass_mem_mut(ark_mem).mt_data.take();
        let last_flag = mtsetup(t, &mut mt_data);
        arkls_mass_mem_mut(ark_mem).mt_data = mt_data;
        {
            let mut ls = arkls_mass_mem_mut(ark_mem);
            ls.last_flag = last_flag;
            ls.nmtsetup += 1;
            ls.msetuptime = t;
        }
        if last_flag != 0 {
            arkProcessError(
                Some(ark_mem),
                last_flag,
                line!() as i32,
                "arkLsMassSetup",
                file!(),
                MSG_LS_MTSETUP_FAILED,
            );
            return last_flag;
        }
    }

    /* Perform user-facing setup based on whether this is matrix-free */
    let call_lssetup;
    let call_mvsetup;
    let M = arkls_mass_mem_mut(ark_mem).M.clone();
    match &M {
        None => {
            /*** matrix-free -- only call LS setup if preconditioner setup exists ***/
            call_lssetup = arkls_mass_mem_mut(ark_mem).pset.is_some();
            /*** matrix-free -- dont call matvec setup ***/
            call_mvsetup = SUNFALSE;
        }
        Some(M) => {
            /*** matrix-based ***/

            /* If mass matrix is not time dependent, and if it has been set up
               previously, then just reuse existing matrix and factorization */
            let (time_dependent, nmsetups, iterative) = {
                let ls = arkls_mass_mem_mut(ark_mem);
                (ls.time_dependent, ls.nmsetups, ls.iterative)
            };
            if !time_dependent && (nmsetups > 0) {
                arkls_mass_mem_mut(ark_mem).last_flag = ARKLS_SUCCESS;
                return ARKLS_SUCCESS;
            }

            /* Clear the mass matrix if necessary (direct linear solvers) */
            if !iterative {
                let retval = SUNMatZero(M);
                if retval != 0 {
                    arkProcessError(
                        Some(ark_mem),
                        ARKLS_SUNMAT_FAIL,
                        line!() as i32,
                        "arkLsMassSetup",
                        file!(),
                        MSG_LS_SUNMAT_FAILED,
                    );
                    arkls_mass_mem_mut(ark_mem).last_flag = ARKLS_SUNMAT_FAIL;
                    return ARKLS_SUNMAT_FAIL;
                }
            }

            /* Call user-supplied routine to fill the mass matrix */
            let mass = arkls_mass_mem_mut(ark_mem).mass.expect("mass");
            let use_field = arkls_mass_mem_mut(ark_mem).M_data.is_some();
            let mut m_data = if use_field {
                arkls_mass_mem_mut(ark_mem).M_data.take()
            } else {
                ark_mem.borrow_mut().user_data.take()
            };
            let retval = mass(t, M, &mut m_data, vtemp1, vtemp2, vtemp3);
            if use_field {
                arkls_mass_mem_mut(ark_mem).M_data = m_data;
            } else {
                ark_mem.borrow_mut().user_data = m_data;
            }
            arkls_mass_mem_mut(ark_mem).msetuptime = t;
            if retval < 0 {
                arkProcessError(
                    Some(ark_mem),
                    ARKLS_MASSFUNC_UNRECVR,
                    line!() as i32,
                    "arkLsMassSetup",
                    file!(),
                    MSG_LS_MASSFUNC_FAILED,
                );
                arkls_mass_mem_mut(ark_mem).last_flag = ARKLS_MASSFUNC_UNRECVR;
                return -1;
            }
            if retval > 0 {
                arkls_mass_mem_mut(ark_mem).last_flag = ARKLS_MASSFUNC_RECVR;
                return 1;
            }

            /* Copy M into M_lu for factorization (direct linear solvers) */
            if !iterative {
                let M_lu = arkls_mass_mem_mut(ark_mem)
                    .M_lu
                    .as_ref()
                    .expect("M_lu")
                    .clone();
                let retval = SUNMatCopy(M, &M_lu);
                if retval != 0 {
                    arkProcessError(
                        Some(ark_mem),
                        ARKLS_SUNMAT_FAIL,
                        line!() as i32,
                        "arkLsMassSetup",
                        file!(),
                        MSG_LS_SUNMAT_FAILED,
                    );
                    arkls_mass_mem_mut(ark_mem).last_flag = ARKLS_SUNMAT_FAIL;
                    return ARKLS_SUNMAT_FAIL;
                }
            }

            /* signal call to matvec setup routine only if the user didn't provide
               mtimes and the SUNMatrix implements the matvecsetup routine */
            if arkls_mass_mem_mut(ark_mem).mtimes.is_none()
                && M.ops.borrow().matvecsetup.is_some()
            {
                call_mvsetup = SUNTRUE;
            } else {
                call_mvsetup = SUNFALSE;
            }

            /* signal call to LS setup routine */
            call_lssetup = SUNTRUE;
        }
    }

    /* Call matvec setup routine if applicable */
    if call_mvsetup {
        let M = M.as_ref().expect("mass matrix M");
        let retval = SUNMatMatvecSetup(M);
        arkls_mass_mem_mut(ark_mem).nmvsetup += 1;
        if retval != 0 {
            arkProcessError(
                Some(ark_mem),
                ARKLS_SUNMAT_FAIL,
                line!() as i32,
                "arkLsMassSetup",
                file!(),
                MSG_LS_SUNMAT_FAILED,
            );
            arkls_mass_mem_mut(ark_mem).last_flag = ARKLS_SUNMAT_FAIL;
            return ARKLS_SUNMAT_FAIL;
        }
    }

    /* Call LS setup routine if applicable, and return */
    if call_lssetup {
        let M_lu = arkls_mass_mem_mut(ark_mem).M_lu.clone();
        let last_flag = SUNLinSolSetup(&LS, M_lu.as_ref());
        let mut ls = arkls_mass_mem_mut(ark_mem);
        ls.last_flag = last_flag;
        ls.nmsetups += 1;
    }

    arkls_mass_mem_mut(ark_mem).last_flag
}

/*---------------------------------------------------------------
  arkLsMassSolve: interfaces between ARKODE and the generic
  SUNLinearSolver object LS, by setting the appropriate tolerance
  and scaling vectors, calling the solver, and accumulating
  statistics from the solve for use/reporting by ARKODE.
  ---------------------------------------------------------------*/
pub fn arkLsMassSolve(ark_mem: &ARKodeMem, b: &N_Vector, nlscoef: sunrealtype) -> i32 {
    /* access ARKLsMassMem structure */
    let retval = arkLs_AccessMassMem(ark_mem, "arkLsMassSolve");
    if retval != ARK_SUCCESS {
        return retval;
    }

    let (iterative, eplifac, nrmfac) = {
        let ls = arkls_mass_mem_mut(ark_mem);
        (ls.iterative, ls.eplifac, ls.nrmfac)
    };
    let (rwt, ewt) = {
        let m = ark_mem.borrow();
        (
            m.rwt.as_ref().expect("rwt").clone(),
            m.ewt.as_ref().expect("ewt").clone(),
        )
    };

    /* Set input tolerance for iterative solvers (in 2-norm) */
    let mut delta: sunrealtype;
    if iterative {
        delta = eplifac * nlscoef * nrmfac;
    } else {
        delta = ZERO;
    }

    let LS = arkls_mass_mem_mut(ark_mem).LS.clone();
    let x = arkls_mass_mem_mut(ark_mem).x.as_ref().expect("x").clone();

    /* Set initial guess x = 0 for LS */
    N_VConst(ZERO, &x);

    /* Set scaling vectors for LS to use (if applicable) */
    if LS.ops.borrow().setscalingvectors.is_some() {
        let retval = SUNLinSolSetScalingVectors(&LS, Some(&rwt), Some(&ewt));
        if retval != SUN_SUCCESS {
            arkProcessError(
                Some(ark_mem),
                ARKLS_SUNLS_FAIL,
                line!() as i32,
                "arkLsMassSolve",
                file!(),
                "Error in call to SUNLinSolSetScalingVectors",
            );
            arkls_mass_mem_mut(ark_mem).last_flag = ARKLS_SUNLS_FAIL;
            return ARKLS_SUNLS_FAIL;
        }

        /* If solver is iterative and does not support scaling vectors, update the
         tolerance in an attempt to account for rwt vector (see arkLsSolve for
         the derivation of rwt_mean). */
    } else if iterative {
        N_VConst(ONE, &x);
        let rwt_mean = N_VWrmsNorm(&rwt, &x);
        delta /= rwt_mean;
    }

    /* Set initial guess x = 0 for LS */
    N_VConst(ZERO, &x);

    /* Set zero initial guess flag */
    let retval = SUNLinSolSetZeroGuess(&LS, SUNTRUE);
    if retval != SUN_SUCCESS {
        return -1;
    }

    /* C stores the previous nps value in nps_inc here (logging only —
    omitted at SUNDIALS_LOGGING_LEVEL 2) */

    /* Call solver, copy x to b, and increment mass solver counter */
    let M_lu = arkls_mass_mem_mut(ark_mem).M_lu.clone();
    let retval = SUNLinSolSolve(&LS, M_lu.as_ref(), &x, b, delta);
    N_VScale(ONE, &x, b);
    arkls_mass_mem_mut(ark_mem).nmsolves += 1;

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
        let mut ls = arkls_mass_mem_mut(ark_mem);
        ls.nli += nli_inc as i64;
        if retval != SUN_SUCCESS {
            ls.ncfl += 1;
        }
    }

    /* Interpret solver return value  */
    arkls_mass_mem_mut(ark_mem).last_flag = retval;

    match retval {
        SUN_SUCCESS => 0,
        SUNLS_RES_REDUCED | SUNLS_CONV_FAIL | SUNLS_ATIMES_FAIL_REC | SUNLS_PSOLVE_FAIL_REC
        | SUNLS_PACKAGE_FAIL_REC | SUNLS_QRFACT_FAIL | SUNLS_LUFACT_FAIL => 1,
        SUN_ERR_ARG_CORRUPT | SUN_ERR_ARG_INCOMPATIBLE | SUN_ERR_MEM_FAIL | SUNLS_GS_FAIL
        | SUNLS_QRSOL_FAIL => -1,
        SUN_ERR_EXT_FAIL => {
            arkProcessError(
                Some(ark_mem),
                SUN_ERR_EXT_FAIL,
                line!() as i32,
                "arkLsMassSolve",
                file!(),
                "Failure in SUNLinSol external package",
            );
            -1
        }
        SUNLS_ATIMES_FAIL_UNREC => {
            arkProcessError(
                Some(ark_mem),
                SUNLS_ATIMES_FAIL_UNREC,
                line!() as i32,
                "arkLsMassSolve",
                file!(),
                MSG_LS_MTIMES_FAILED,
            );
            -1
        }
        SUNLS_PSOLVE_FAIL_UNREC => {
            arkProcessError(
                Some(ark_mem),
                SUNLS_PSOLVE_FAIL_UNREC,
                line!() as i32,
                "arkLsMassSolve",
                file!(),
                MSG_LS_PSOLVE_FAILED,
            );
            -1
        }
        _ => {
            arkProcessError(
                Some(ark_mem),
                retval,
                line!() as i32,
                "arkLsMassSolve",
                file!(),
                "Unrecognized error return value from SUNLinSolSolve",
            );
            -1
        }
    }
}

/*---------------------------------------------------------------
  arkLsMassFree frees memory associates with the ARKLs mass
  matrix solver interface.
  ---------------------------------------------------------------*/
pub fn arkLsMassFree(ark_mem: &ARKodeMem) -> i32 {
    /* NULL ARKodeMem check: handled by the type system */

    /* Return immediately if ARKLsMassMem is NULL */
    let step_getmassmem = ark_mem.borrow().step_getmassmem.expect("step_getmassmem");
    if !step_getmassmem(ark_mem) {
        return ARKLS_SUCCESS;
    }

    /* detach ARKLs interface routines from LS object (ignore return values) */
    let LS = arkls_mass_mem_mut(ark_mem).LS.clone();
    {
        let setatimes = LS.ops.borrow().setatimes.is_some();
        if setatimes {
            let _ = SUNLinSolSetATimes(&LS, None, None);
        }

        let setpreconditioner = LS.ops.borrow().setpreconditioner.is_some();
        if setpreconditioner {
            let _ = SUNLinSolSetPreconditioner(&LS, None, None, None);
        }
    }

    let iterative = arkls_mass_mem_mut(ark_mem).iterative;
    {
        let mut ls = arkls_mass_mem_mut(ark_mem);

        /* Free N_Vector memory */
        if let Some(x) = ls.x.take() {
            N_VDestroy(x);
        }

        /* Free M_lu memory (direct linear solvers) */
        if !iterative {
            if let Some(M_lu) = ls.M_lu.take() {
                SUNMatDestroy(M_lu);
            }
        }
        ls.M_lu = None;

        /* Nullify other N_Vector pointers */
        ls.ycur = None;

        /* Nullify other SUNMatrix pointer */
        ls.M = None;
    }

    /* Free preconditioner memory (if applicable) */
    let pfree = arkls_mass_mem_mut(ark_mem).pfree;
    if let Some(pfree) = pfree {
        pfree(ark_mem);
    }

    /* free ARKLs interface structure */
    ark_mem.borrow_mut().ark_mass_mem = None;

    ARKLS_SUCCESS
}

/*---------------------------------------------------------------
  arkLsInitializeCounters and arkLsInitializeMassCounters:

  These routines reset all counters from an ARKLsMem or
  ARKLsMassMem structure.
  ---------------------------------------------------------------*/
pub fn arkLsInitializeCounters(arkls_mem: &mut ARKLsMemRec) -> i32 {
    arkls_mem.nje = 0;
    arkls_mem.nfeDQ = 0;
    arkls_mem.nstlj = 0;
    arkls_mem.npe = 0;
    arkls_mem.nli = 0;
    arkls_mem.nps = 0;
    arkls_mem.ncfl = 0;
    arkls_mem.njtsetup = 0;
    arkls_mem.njtimes = 0;
    0
}

pub fn arkLsInitializeMassCounters(arkls_mem: &mut ARKLsMassMemRec) -> i32 {
    arkls_mem.nmsetups = 0;
    arkls_mem.nmsolves = 0;
    arkls_mem.nmtsetup = 0;
    arkls_mem.nmtimes = 0;
    arkls_mem.nmvsetup = 0;
    arkls_mem.npe = 0;
    arkls_mem.nli = 0;
    arkls_mem.nps = 0;
    arkls_mem.ncfl = 0;
    arkls_mem.msetuptime = -SUN_BIG_REAL;
    0
}

/*---------------------------------------------------------------
  arkLs_AccessARKODELMem, arkLs_AccessLMem,
  arkLs_AccessARKODEMassMem and arkLs_AccessMassMem:

  Shortcut routines to verify that the ls_mem / mass_mem records
  are attached. If either is missing they return ARKLS_LMEM_NULL or
  ARKLS_MASSMEM_NULL (the C `arkode_mem == NULL` -> ARKLS_MEM_NULL
  check is handled by the type system for the `&ARKodeMem` flavors,
  and survives in the `*Token` flavors below, which receive the C
  `void*` data token registered with the SUNLinearSolver).

  Callers then reach the record itself through `arkls_mem_mut` /
  `arkls_mass_mem_mut` at each use site.
  ---------------------------------------------------------------*/
pub fn arkLs_AccessARKODELMem(arkode_mem: &ARKodeMem, fname: &str) -> i32 {
    /* NULL arkode_mem check: handled by the type system */
    if arkode_mem.borrow().ark_lmem.is_none() {
        arkProcessError(
            Some(arkode_mem),
            ARKLS_LMEM_NULL,
            line!() as i32,
            fname,
            file!(),
            MSG_LS_LMEM_NULL,
        );
        return ARKLS_LMEM_NULL;
    }
    ARKLS_SUCCESS
}

pub fn arkLs_AccessLMem(ark_mem: &ARKodeMem, fname: &str) -> i32 {
    if ark_mem.borrow().ark_lmem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARKLS_LMEM_NULL,
            line!() as i32,
            fname,
            file!(),
            MSG_LS_LMEM_NULL,
        );
        return ARKLS_LMEM_NULL;
    }
    ARKLS_SUCCESS
}

pub fn arkLs_AccessARKODEMassMem(arkode_mem: &ARKodeMem, fname: &str) -> i32 {
    /* NULL arkode_mem check: handled by the type system */
    if arkode_mem.borrow().ark_mass_mem.is_none() {
        arkProcessError(
            Some(arkode_mem),
            ARKLS_MASSMEM_NULL,
            line!() as i32,
            fname,
            file!(),
            MSG_LS_MASSMEM_NULL,
        );
        return ARKLS_MASSMEM_NULL;
    }
    ARKLS_SUCCESS
}

pub fn arkLs_AccessMassMem(ark_mem: &ARKodeMem, fname: &str) -> i32 {
    if ark_mem.borrow().ark_mass_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARKLS_MASSMEM_NULL,
            line!() as i32,
            fname,
            file!(),
            MSG_LS_MASSMEM_NULL,
        );
        return ARKLS_MASSMEM_NULL;
    }
    ARKLS_SUCCESS
}

/// Callback flavor of C's `arkLs_AccessARKODELMem`: the C `void* arkode_mem`
/// argument arrives as the data token registered with the SUNLinearSolver
/// (a boxed `ARKodeMem` clone). A missing/foreign token maps to the C NULL
/// check.
pub fn arkLs_AccessARKODELMemToken(
    arkode_mem: &Option<Box<dyn Any>>,
    fname: &str,
) -> Result<ARKodeMem, i32> {
    let ark_mem = match arkode_mem
        .as_ref()
        .and_then(|b| b.downcast_ref::<ARKodeMem>())
        .cloned()
    {
        Some(m) => m,
        None => {
            arkProcessError(
                None,
                ARKLS_MEM_NULL,
                line!() as i32,
                fname,
                file!(),
                MSG_LS_ARKMEM_NULL,
            );
            return Err(ARKLS_MEM_NULL);
        }
    };
    if ark_mem.borrow().ark_lmem.is_none() {
        arkProcessError(
            Some(&ark_mem),
            ARKLS_LMEM_NULL,
            line!() as i32,
            fname,
            file!(),
            MSG_LS_LMEM_NULL,
        );
        return Err(ARKLS_LMEM_NULL);
    }
    Ok(ark_mem)
}

/// Callback flavor of C's `arkLs_AccessARKODEMassMem` (see
/// [`arkLs_AccessARKODELMemToken`]).
pub fn arkLs_AccessARKODEMassMemToken(
    arkode_mem: &Option<Box<dyn Any>>,
    fname: &str,
) -> Result<ARKodeMem, i32> {
    let ark_mem = match arkode_mem
        .as_ref()
        .and_then(|b| b.downcast_ref::<ARKodeMem>())
        .cloned()
    {
        Some(m) => m,
        None => {
            arkProcessError(
                None,
                ARKLS_MEM_NULL,
                line!() as i32,
                fname,
                file!(),
                MSG_LS_ARKMEM_NULL,
            );
            return Err(ARKLS_MEM_NULL);
        }
    };
    if ark_mem.borrow().ark_mass_mem.is_none() {
        arkProcessError(
            Some(&ark_mem),
            ARKLS_MASSMEM_NULL,
            line!() as i32,
            fname,
            file!(),
            MSG_LS_MASSMEM_NULL,
        );
        return Err(ARKLS_MASSMEM_NULL);
    }
    Ok(ark_mem)
}

/*===============================================================
  EOF
  ===============================================================*/
