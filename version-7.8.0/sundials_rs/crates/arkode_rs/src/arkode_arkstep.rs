//! Port of `src/arkode/arkode_arkstep.c` + `src/arkode/arkode_arkstep_impl.h`
//! + the constants of `include/arkode/arkode_arkstep.h` — ARKODE's additive
//! Runge–Kutta time stepper (explicit table, implicit table, or both, with
//! optional identity / fixed / time-dependent mass matrix).
//!
//! The public `ARKStep*` optional-input/output routines live in
//! `arkode_arkstep_io.rs` and the nonlinear-solver interface in
//! `arkode_arkstep_nls.rs`, exactly as in the C tree; this module owns the
//! stepper's content record, the `step_*` function table, the stage loop and
//! the Butcher-table setup/checks.
//!
//! Binding notes (all mandated by `arkode_impl.rs`, the frozen contract):
//!
//! * `ark_mem.step_mem` holds `ARKodeARKStepMemRec` BY VALUE inside
//!   `Option<Box<dyn Any>>`; [`arkStep_mem_mut`] is the one downcast helper.
//!   The guard it returns IS `ark_mem.borrow_mut()`, so it is never held
//!   across `arkProcessError`, a user callback, an `N_Vector` operation, a
//!   linear/nonlinear-solver call, or another borrow of the same mem —
//!   every such site copies the fields it needs into locals, drops the
//!   guard, calls, and writes results back.
//! * `step_mem->jcur` is the shared [`ARKJcurPtr`] cell (`Rc<Cell<..>>`) so
//!   that `arkStep_GetGammas` can hand `arkode_ls.rs` the very address the
//!   C code passes as `jcurPtr` and a preconditioner-setup routine reached
//!   re-entrantly through it writes through the same flag.
//! * `step_mem->lmem` / `step_mem->mass_mem` are presence flags: the ARKLS
//!   records themselves live in `ark_mem.ark_lmem` / `ark_mem.ark_mass_mem`
//!   (contract §4), so `arkStep_GetLmem` / `arkStep_GetMassMem` are
//!   `sunbooleantype` probes.
//! * `step_mem->cvals` / `step_mem->Xvecs`: C keeps one reusable
//!   `sunrealtype*` / `N_Vector*` scratch pair for the fused vector
//!   operations. An `N_Vector` handle array cannot hold C's NULL slots, so
//!   the fields here only model C's *allocation state* — which is
//!   observable, because `ark_mem.lrw` / `ark_mem.liw` are reported by
//!   `ARKodeGetWorkSpace` and printed by `ark_KrylovDemo_prec` — while each
//!   fused operation builds its `nvec`-long operand list in a local `Vec`,
//!   pushing exactly the same values in exactly the same order.
//! * Every C `pow` site would map to `SUNRpowerR`/`SUNRpowerI`; ARKStep has
//!   none (its adaptivity lives in `arkode_adapt.rs`).

use std::any::Any;
use std::cell::{Cell, RefMut};

use sundials_core::sundials_adjointcheckpointscheme::{
    SUNAdjointCheckpointScheme_InsertVector, SUNAdjointCheckpointScheme_LoadVector,
    SUNAdjointCheckpointScheme_NeedsSaving,
};
use sundials_core::sundials_adjointstepper::{
    SUNAdjRhsFn, SUNAdjointStepper, SUNAdjointStepper_Create, SUNAdjointStepper_RecomputeFwd,
    SUNAdjointStepper_SetUserData,
};
use sundials_core::sundials_context::SUNContext;
use sundials_core::sundials_errors::{SUN_ERR_CHECKPOINT_NOT_FOUND, SUN_ERR_OP_FAIL, SUN_SUCCESS};
use sundials_core::sundials_linearsolver::{
    SUNLinearSolver_Type, SUNLINEARSOLVER_ITERATIVE, SUNLINEARSOLVER_MATRIX_ITERATIVE,
};
use sundials_core::sundials_math::SUNRabs;
use sundials_core::sundials_nonlinearsolver::{
    SUNNonlinSolFree, SUNNonlinSolSetup, SUNNonlinearSolver,
};
use sundials_core::sundials_nvector::{
    N_VConst, N_VDotProd, N_VDotProdLocal, N_VDotProdMultiAllReduce, N_VGetVectorID,
    N_VLinearCombination, N_VLinearSum, N_VScale, N_VSpace, N_VWrmsNorm, N_Vector,
    SUNDIALS_NVEC_MANYVECTOR,
};
use sundials_core::sundials_stepper::{
    SUNStepper, SUNStepper_GetContentAs, SUNStepper_SetDestroyFn, SUNStepper_SetReInitFn,
};
use sundials_core::sundials_types::*;
use sundials_core::sundials_utils::{sun_format_g, SUNFile};
use sundials_core::sunnonlinsol_newton::SUNNonlinSol_Newton;

use sundials_core::nvector_manyvector::N_VGetSubvector_ManyVector;

use crate::arkode::{
    arkAllocVec, arkAllocVecArray, arkCreate, arkEwtSetSmallReal, arkFreeVec, arkInit,
    arkPredict_Bootstrap, arkPredict_CutoffOrder, arkPredict_MaximumOrder,
    arkPredict_VariableOrder, arkResizeVec, ARKodeFree,
};
use crate::arkode_arkstep_io::{
    arkStep_GetCurrentGamma, arkStep_GetEstLocalErrors, arkStep_GetNonlinSolvStats,
    arkStep_GetNumLinSolvSetups, arkStep_GetNumNonlinSolvConvFails, arkStep_GetNumNonlinSolvIters,
    arkStep_GetNumRhsEvals, arkStep_GetStageIndex, arkStep_PrintAllStats, arkStep_SetAutonomous,
    arkStep_SetDeduceImplicitRhs, arkStep_SetDefaults, arkStep_SetDeltaGammaMax,
    arkStep_SetLSetupFrequency, arkStep_SetLinear, arkStep_SetMaxNonlinIters,
    arkStep_SetNonlinCRDown, arkStep_SetNonlinConvCoef, arkStep_SetNonlinRDiv,
    arkStep_SetNonlinear, arkStep_SetOptions, arkStep_SetOrder, arkStep_SetPredictorMethod,
    arkStep_SetRelaxFn, arkStep_SetStagePredictFn, arkStep_SetUserData, arkStep_WriteParameters,
    ARKStepSetTables,
};
use crate::arkode_arkstep_nls::{
    arkStep_GetNonlinearSystemData, arkStep_Nls, arkStep_NlsInit, arkStep_SetNlsRhsFn,
    arkStep_SetNonlinearSolver,
};
use crate::arkode_butcher::{
    ARKodeButcherTable, ARKodeButcherTable_IsStifflyAccurate, ARKodeButcherTable_Space,
    ARKodeButcherTable_Write,
};
use crate::arkode_butcher_dirk::{
    ARKODE_ARK548L2SAb_DIRK_8_4_5, ARKodeButcherTable_LoadDIRK, ARKODE_ARK2_DIRK_3_1_2,
    ARKODE_ARK324L2SA_DIRK_4_2_3, ARKODE_ARK437L2SA_DIRK_7_3_4, ARKODE_BACKWARD_EULER_1_1,
    ARKODE_ESDIRK325L2SA_5_2_3, ARKODE_ESDIRK436L2SA_6_3_4, ARKODE_ESDIRK547L2SA2_7_4_5,
};
use crate::arkode_butcher_erk::{
    ARKODE_ARK548L2SAb_ERK_8_4_5, ARKodeButcherTable_LoadERK, ARKODE_ARK2_ERK_3_1_2,
    ARKODE_ARK324L2SA_ERK_4_2_3, ARKODE_ARK437L2SA_ERK_7_3_4, ARKODE_BOGACKI_SHAMPINE_4_2_3,
    ARKODE_FORWARD_EULER_1_1, ARKODE_RALSTON_3_1_2, ARKODE_SOFRONIOU_SPALETTA_5_3_4,
    ARKODE_TSITOURAS_7_4_5, ARKODE_VERNER_10_6_7, ARKODE_VERNER_13_7_8, ARKODE_VERNER_16_8_9,
    ARKODE_VERNER_9_5_6,
};
use crate::arkode_impl::*;
use crate::arkode_io::{
    ARKodeGetNumSteps, ARKodeSetAdjointCheckpointScheme, ARKodeSetFixedStep, ARKodeSetMaxNumSteps,
    ARKodeSetNonlinearSolver, ARKodeSetUserData,
};
use crate::arkode_ls::{arkLsInitializeCounters, arkls_mem_mut};
use crate::arkode_sunstepper::{arkSUNStepperSelfDestruct, ARKodeCreateSUNStepper};

/*===============================================================
ARK time step module constants (arkode_arkstep_impl.h)

MAXCOR / CRDOWN / DGMAX / RDIV / MSBP / NLSCOEF are byte-identical
duplicates of the MRIStep values and are hoisted into
`arkode_impl.rs` (contract §7); only the mass matrix types stay here.
===============================================================*/

/* Mass matrix types */
pub const MASS_IDENTITY: i32 = 0;
pub const MASS_FIXED: i32 = 1;
pub const MASS_TIMEDEP: i32 = 2;

/*===============================================================
Reusable ARKStep Error Messages (arkode_arkstep_impl.h)
===============================================================*/

/* Initialization and I/O error messages */
pub const MSG_ARKSTEP_NO_MEM: &str = "Time step module memory is NULL.";
pub const MSG_NLS_INIT_FAIL: &str = "The nonlinear solver's init routine failed.";

/* Other error messages */
pub const MSG_ARK_MISSING_FE: &str =
    "Cannot specify that method is explicit without providing a function pointer to fe(t,y).";
pub const MSG_ARK_MISSING_FI: &str =
    "Cannot specify that method is implicit without providing a function pointer to fi(t,y).";
pub const MSG_ARK_MISSING_F: &str = "Cannot specify that method is ImEx without providing \
                                     function pointers to fi(t,y) and fe(t,y).";

/*===============================================================
ARKStep Constants (include/arkode/arkode_arkstep.h)

C declares these `static const int`, and `arkStep_SetButcherTables`
keeps them in plain `int` locals before handing them to
`ARKodeButcherTable_Load{ERK,DIRK}` — reproduced verbatim.
===============================================================*/

/* Default Butcher tables for each method/order */

/*    explicit */
pub const ARKSTEP_DEFAULT_ERK_1: i32 = ARKODE_FORWARD_EULER_1_1;
pub const ARKSTEP_DEFAULT_ERK_2: i32 = ARKODE_RALSTON_3_1_2;
pub const ARKSTEP_DEFAULT_ERK_3: i32 = ARKODE_BOGACKI_SHAMPINE_4_2_3;
pub const ARKSTEP_DEFAULT_ERK_4: i32 = ARKODE_SOFRONIOU_SPALETTA_5_3_4;
pub const ARKSTEP_DEFAULT_ERK_5: i32 = ARKODE_TSITOURAS_7_4_5;
pub const ARKSTEP_DEFAULT_ERK_6: i32 = ARKODE_VERNER_9_5_6;
pub const ARKSTEP_DEFAULT_ERK_7: i32 = ARKODE_VERNER_10_6_7;
pub const ARKSTEP_DEFAULT_ERK_8: i32 = ARKODE_VERNER_13_7_8;
pub const ARKSTEP_DEFAULT_ERK_9: i32 = ARKODE_VERNER_16_8_9;

/*    implicit */
pub const ARKSTEP_DEFAULT_DIRK_1: i32 = ARKODE_BACKWARD_EULER_1_1;
pub const ARKSTEP_DEFAULT_DIRK_2: i32 = ARKODE_ARK2_DIRK_3_1_2;
pub const ARKSTEP_DEFAULT_DIRK_3: i32 = ARKODE_ESDIRK325L2SA_5_2_3;
pub const ARKSTEP_DEFAULT_DIRK_4: i32 = ARKODE_ESDIRK436L2SA_6_3_4;
pub const ARKSTEP_DEFAULT_DIRK_5: i32 = ARKODE_ESDIRK547L2SA2_7_4_5;

/*    ImEx */
pub const ARKSTEP_DEFAULT_ARK_ETABLE_2: i32 = ARKODE_ARK2_ERK_3_1_2;
pub const ARKSTEP_DEFAULT_ARK_ETABLE_3: i32 = ARKODE_ARK324L2SA_ERK_4_2_3;
pub const ARKSTEP_DEFAULT_ARK_ETABLE_4: i32 = ARKODE_ARK437L2SA_ERK_7_3_4;
pub const ARKSTEP_DEFAULT_ARK_ETABLE_5: i32 = ARKODE_ARK548L2SAb_ERK_8_4_5;
pub const ARKSTEP_DEFAULT_ARK_ITABLE_2: i32 = ARKODE_ARK2_DIRK_3_1_2;
pub const ARKSTEP_DEFAULT_ARK_ITABLE_3: i32 = ARKODE_ARK324L2SA_DIRK_4_2_3;
pub const ARKSTEP_DEFAULT_ARK_ITABLE_4: i32 = ARKODE_ARK437L2SA_DIRK_7_3_4;
pub const ARKSTEP_DEFAULT_ARK_ITABLE_5: i32 = ARKODE_ARK548L2SAb_DIRK_8_4_5;

/*===============================================================
ARK time step module data structure
===============================================================*/

/// C `struct ARKodeARKStepMemRec` (`arkode_arkstep_impl.h`).
pub struct ARKodeARKStepMemRec {
    /* ARK problem specification */
    pub fe: Option<ARKRhsFn>, /* My' = fe(t,y) + fi(t,y) */
    pub fi: Option<ARKRhsFn>,
    pub autonomous: sunbooleantype, /* SUNTRUE if fi depends on t     */
    pub linear: sunbooleantype,     /* SUNTRUE if fi is linear        */
    pub linear_timedep: sunbooleantype, /* SUNTRUE if dfi/dy depends on t */
    pub explicit: sunbooleantype,   /* SUNTRUE if fe is enabled       */
    pub implicit: sunbooleantype,   /* SUNTRUE if fi is enabled       */
    pub deduce_rhs: sunbooleantype, /* SUNTRUE if fi is deduced after
                                    a nonlinear solve                */

    /* Adjoint problem specification */
    pub adj_fe: Option<SUNAdjRhsFn>,

    /* ARK method storage and parameters */
    pub Fe: Vec<N_Vector>, /* explicit RHS at each stage; empty == C NULL */
    pub Fi: Vec<N_Vector>, /* implicit RHS at each stage; empty == C NULL */
    pub z: Vec<N_Vector>,  /* stages (for relaxation);    empty == C NULL */
    pub sdata: Option<N_Vector>, /* old stage data in residual */
    pub zpred: Option<N_Vector>, /* predicted stage solution   */
    pub zcor: Option<N_Vector>, /* stage correction           */
    pub q: i32,            /* method order               */
    pub p: i32,            /* embedding order            */
    pub istage: i32,       /* current stage              */
    pub stages: i32,       /* number of stages           */
    pub Be: Option<ARKodeButcherTable>, /* ERK Butcher table */
    pub Bi: Option<ARKodeButcherTable>, /* IRK Butcher table */

    /* User-supplied stage predictor routine */
    pub stage_predict: Option<ARKStagePredictFn>,

    /* (Non)Linear solver parameters & data */
    pub NLS: Option<SUNNonlinearSolver>, /* generic SUNNonlinearSolver object     */
    pub ownNLS: sunbooleantype,          /* flag indicating ownership of NLS      */
    pub nls_fi: Option<ARKRhsFn>,        /* fi(t,y) used in the nonlinear solver  */
    pub gamma: sunrealtype,              /* gamma = h * A(i,i)                       */
    pub gammap: sunrealtype,             /* gamma at the last setup call             */
    pub gamrat: sunrealtype,             /* gamma / gammap                           */
    pub dgmax: sunrealtype,              /* call lsetup if |gamma/gammap-1| >= dgmax */

    pub predictor: i32,      /* implicit prediction method to use        */
    pub crdown: sunrealtype, /* nonlinear conv rate estimation constant  */
    pub rdiv: sunrealtype,   /* nonlin divergence if delnrm/delnrm_p > rdiv */
    /// C `step_mem->crate` — renamed because `crate` is a Rust keyword.
    pub crate_: sunrealtype, /* estimated nonlin convergence rate        */
    pub delnrm_p: sunrealtype, /* norm of previous nonlinear solver update */
    pub delnrm: sunrealtype, /* norm of current nonlinear solver update  */
    pub eRNrm: sunrealtype,  /* estimated residual norm, used in nonlin
                             and linear solver convergence tests       */
    pub nlscoef: sunrealtype, /* coefficient in nonlin. convergence test  */

    pub msbp: i32,  /* positive => max # steps between lsetup
                    negative => call at each Newton iter      */
    pub nstlp: i64, /* step number of last setup call           */

    pub maxcor: i32, /* max num iterations for solving the
                     nonlinear equation                        */

    pub convfail: i32, /* NLS fail flag (for interface routines)   */
    /// C `sunbooleantype jcur` — is Jacobian info for lin solver current?
    /// Shared cell so `step_getgammas` can hand out its address exactly as
    /// C hands out `&step_mem->jcur` (contract, "THE jcur SEAM").
    pub jcur: ARKJcurPtr,
    pub fn_implicit: Option<N_Vector>, /* alias to saved implicit function evaluation */

    /* Linear Solver Data */
    pub linit: Option<ARKLinsolInitFn>,
    pub lsetup: Option<ARKLinsolSetupFn>,
    pub lsolve: Option<ARKLinsolSolveFn>,
    pub lfree: Option<ARKLinsolFreeFn>,
    /// C `void* lmem`. The `ARKLsMemRec` itself is owned by
    /// `ark_mem.ark_lmem` (contract §4); this mirrors `lmem != NULL`.
    pub lmem: sunbooleantype,
    /// C `SUNLinearSolver_Type lsolve_type`, initialized to `-1` — a value
    /// outside the enum, so `None` models it.
    pub lsolve_type: Option<SUNLinearSolver_Type>,

    /* Mass matrix solver data */
    pub minit: Option<ARKMassInitFn>,
    pub msetup: Option<ARKMassSetupFn>,
    pub mmult: Option<ARKMassMultFn>,
    pub msolve: Option<ARKMassSolveFn>,
    pub mfree: Option<ARKMassFreeFn>,
    /// C `void* mass_mem`; the record is owned by `ark_mem.ark_mass_mem`.
    pub mass_mem: sunbooleantype,
    pub mass_type: i32, /* 0=identity, 1=fixed, 2=time-dep */
    pub msolve_type: Option<SUNLinearSolver_Type>,

    /* Counters */
    pub nfe: i64,       /* num fe calls               */
    pub nfi: i64,       /* num fi calls               */
    pub nsetups: i64,   /* num setup calls            */
    pub nls_iters: i64, /* num nonlinear solver iters */
    pub nls_fails: i64, /* num nonlinear solver fails */

    /* Reusable arrays for fused vector operations (see the module docs:
    these model C's allocation state; the operand lists themselves are
    built locally at each fused-operation site). */
    pub cvals: Vec<sunrealtype>, /* scalar array for fused ops       */
    pub Xvecs: Vec<Option<N_Vector>>, /* array of vectors for fused ops */
    pub nfusedopvecs: i32,       /* length of cvals and Xvecs arrays */

    /* Data for using ARKStep with external polynomial forcing */
    pub expforcing: sunbooleantype, /* add forcing to explicit RHS */
    pub impforcing: sunbooleantype, /* add forcing to implicit RHS */
    pub tshift: sunrealtype,        /* time normalization shift    */
    pub tscale: sunrealtype,        /* time normalization scaling  */
    pub forcing: Vec<N_Vector>,     /* array of forcing vectors    */
    pub nforcing: i32,              /* number of forcing vectors   */
    pub stage_times: Vec<sunrealtype>, /* workspace for applying forcing */
    pub stage_coefs: Vec<sunrealtype>, /* workspace for applying forcing */
}

impl ARKodeARKStepMemRec {
    /// C `malloc(sizeof(struct ARKodeARKStepMemRec))` +
    /// `memset(step_mem, 0, sizeof(struct ARKodeARKStepMemRec))`.
    pub fn zeroed() -> ARKodeARKStepMemRec {
        ARKodeARKStepMemRec {
            fe: None,
            fi: None,
            autonomous: SUNFALSE,
            linear: SUNFALSE,
            linear_timedep: SUNFALSE,
            explicit: SUNFALSE,
            implicit: SUNFALSE,
            deduce_rhs: SUNFALSE,
            adj_fe: None,
            Fe: Vec::new(),
            Fi: Vec::new(),
            z: Vec::new(),
            sdata: None,
            zpred: None,
            zcor: None,
            q: 0,
            p: 0,
            istage: 0,
            stages: 0,
            Be: None,
            Bi: None,
            stage_predict: None,
            NLS: None,
            ownNLS: SUNFALSE,
            nls_fi: None,
            gamma: 0.0,
            gammap: 0.0,
            gamrat: 0.0,
            dgmax: 0.0,
            predictor: 0,
            crdown: 0.0,
            rdiv: 0.0,
            crate_: 0.0,
            delnrm_p: 0.0,
            delnrm: 0.0,
            eRNrm: 0.0,
            nlscoef: 0.0,
            msbp: 0,
            nstlp: 0,
            maxcor: 0,
            convfail: 0,
            jcur: ARKJcurPtr::new(Cell::new(SUNFALSE)),
            fn_implicit: None,
            linit: None,
            lsetup: None,
            lsolve: None,
            lfree: None,
            lmem: SUNFALSE,
            lsolve_type: None,
            minit: None,
            msetup: None,
            mmult: None,
            msolve: None,
            mfree: None,
            mass_mem: SUNFALSE,
            mass_type: 0,
            msolve_type: None,
            nfe: 0,
            nfi: 0,
            nsetups: 0,
            nls_iters: 0,
            nls_fails: 0,
            cvals: Vec::new(),
            Xvecs: Vec::new(),
            nfusedopvecs: 0,
            expforcing: SUNFALSE,
            impforcing: SUNFALSE,
            tshift: 0.0,
            tscale: 0.0,
            forcing: Vec::new(),
            nforcing: 0,
            stage_times: Vec::new(),
            stage_coefs: Vec::new(),
        }
    }
}

/// C `(ARKodeARKStepMem) ark_mem->step_mem`.
///
/// Panics if no step memory is attached or it is not ARKStep's record (C
/// would blindly cast the `void*` — UB maps to a panic, deviation class 5).
/// The returned guard IS `ark_mem.borrow_mut()`: NEVER hold it across
/// `arkProcessError`, a user callback, an `N_Vector`/matrix/LS/NLS
/// operation, a `step_*` dispatch, or another borrow of the same mem.
pub fn arkStep_mem_mut(ark_mem: &ARKodeMem) -> RefMut<'_, ARKodeARKStepMemRec> {
    RefMut::map(ark_mem.borrow_mut(), |m| {
        m.step_mem
            .as_mut()
            .expect("step_mem set")
            .downcast_mut::<ARKodeARKStepMemRec>()
            .expect("ARKStep step memory")
    })
}

/// Value C prints for `SUNLinearSolver_Type` (`%i` of the enum) and compares
/// against `-1` for "unset".
fn lsolve_type_as_int(t: Option<SUNLinearSolver_Type>) -> i32 {
    match t {
        Some(t) => t as i32,
        None => -1,
    }
}

/*===============================================================
Exported functions
===============================================================*/

/// C `void* ARKStepCreate(ARKRhsFn fe, ARKRhsFn fi, sunrealtype t0,
/// N_Vector y0, SUNContext sunctx)`.
pub fn ARKStepCreate(
    fe: Option<ARKRhsFn>,
    fi: Option<ARKRhsFn>,
    t0: sunrealtype,
    y0: &N_Vector,
    sunctx: &SUNContext,
) -> Option<ARKodeMem> {
    let retval: i32;

    /* Check that at least one of fe, fi is supplied and is to be used */
    if fe.is_none() && fi.is_none() {
        arkProcessError(
            None,
            ARK_ILL_INPUT,
            line!() as i32,
            "ARKStepCreate",
            file!(),
            MSG_ARK_NULL_F,
        );
        return None;
    }

    /* Check for legal input parameters: NULL y0 handled by the type system */

    /* NULL sunctx check: handled by the type system */

    /* Create ark_mem structure and set default values */
    let ark_mem = match arkCreate(sunctx) {
        Some(ark_mem) => ark_mem,
        None => {
            arkProcessError(
                None,
                ARK_MEM_NULL,
                line!() as i32,
                "ARKStepCreate",
                file!(),
                MSG_ARK_NO_MEM,
            );
            return None;
        }
    };

    /* Allocate ARKodeARKStepMem structure, and initialize to zero
    (allocation cannot fail observably in Rust) */
    let step_mem = ARKodeARKStepMemRec::zeroed();

    /* Attach step_mem structure and function pointers to ark_mem */
    {
        let mut m = ark_mem.borrow_mut();
        m.step_attachlinsol = Some(arkStep_AttachLinsol);
        m.step_attachmasssol = Some(arkStep_AttachMasssol);
        m.step_disablelsetup = Some(arkStep_DisableLSetup);
        m.step_disablemsetup = Some(arkStep_DisableMSetup);
        m.step_getlinmem = Some(arkStep_GetLmem);
        m.step_getmassmem = Some(arkStep_GetMassMem);
        m.step_getimplicitrhs = Some(arkStep_GetImplicitRHS);
        m.step_mmult = None;
        m.step_getgammas = Some(arkStep_GetGammas);
        m.step_init = Some(arkStep_Init);
        m.step_fullrhs = Some(arkStep_FullRHS);
        m.step = Some(arkStep_TakeStep_Z);
        m.step_setuserdata = Some(arkStep_SetUserData);
        m.step_printallstats = Some(arkStep_PrintAllStats);
        m.step_writeparameters = Some(arkStep_WriteParameters);
        m.step_setusecompensatedsums = None;
        m.step_resize = Some(arkStep_Resize);
        m.step_free = Some(arkStep_Free);
        m.step_printmem = Some(arkStep_PrintMem);
        m.step_setdefaults = Some(arkStep_SetDefaults);
        m.step_computestate = Some(arkStep_ComputeState);
        m.step_setoptions = Some(arkStep_SetOptions);
        m.step_setrelaxfn = Some(arkStep_SetRelaxFn);
        m.step_setorder = Some(arkStep_SetOrder);
        m.step_setnonlinearsolver = Some(arkStep_SetNonlinearSolver);
        m.step_setlinear = Some(arkStep_SetLinear);
        m.step_setnonlinear = Some(arkStep_SetNonlinear);
        m.step_setautonomous = Some(arkStep_SetAutonomous);
        m.step_setnlsrhsfn = Some(arkStep_SetNlsRhsFn);
        m.step_setdeduceimplicitrhs = Some(arkStep_SetDeduceImplicitRhs);
        m.step_setnonlincrdown = Some(arkStep_SetNonlinCRDown);
        m.step_setnonlinrdiv = Some(arkStep_SetNonlinRDiv);
        m.step_setdeltagammamax = Some(arkStep_SetDeltaGammaMax);
        m.step_setlsetupfrequency = Some(arkStep_SetLSetupFrequency);
        m.step_setpredictormethod = Some(arkStep_SetPredictorMethod);
        m.step_setmaxnonliniters = Some(arkStep_SetMaxNonlinIters);
        m.step_setnonlinconvcoef = Some(arkStep_SetNonlinConvCoef);
        m.step_setstagepredictfn = Some(arkStep_SetStagePredictFn);
        m.step_getnumrhsevals = Some(arkStep_GetNumRhsEvals);
        m.step_getnumlinsolvsetups = Some(arkStep_GetNumLinSolvSetups);
        m.step_getcurrentgamma = Some(arkStep_GetCurrentGamma);
        m.step_getestlocalerrors = Some(arkStep_GetEstLocalErrors);
        m.step_getnonlinearsystemdata = Some(arkStep_GetNonlinearSystemData);
        m.step_getnumnonlinsolviters = Some(arkStep_GetNumNonlinSolvIters);
        m.step_getnumnonlinsolvconvfails = Some(arkStep_GetNumNonlinSolvConvFails);
        m.step_getnonlinsolvstats = Some(arkStep_GetNonlinSolvStats);
        m.step_setforcing = Some(arkStep_SetInnerForcing);
        m.step_getstageindex = Some(arkStep_GetStageIndex);
        m.step_supports_adaptive = SUNTRUE;
        m.step_supports_implicit = SUNTRUE;
        m.step_supports_massmatrix = SUNTRUE;
        m.step_supports_relaxation = SUNTRUE;
        m.step_mem = Some(Box::new(step_mem));
    }

    /* Set default values for optional inputs */
    retval = arkStep_SetDefaults(&ark_mem);
    if retval != ARK_SUCCESS {
        arkProcessError(
            Some(&ark_mem),
            retval,
            line!() as i32,
            "ARKStepCreate",
            file!(),
            "Error setting default solver options",
        );
        let mut ark_mem = Some(ark_mem);
        ARKodeFree(&mut ark_mem);
        return None;
    }

    /* Set implicit/explicit problem based on function pointers */
    {
        let mut step_mem = arkStep_mem_mut(&ark_mem);
        step_mem.explicit = if fe.is_none() { SUNFALSE } else { SUNTRUE };
        step_mem.implicit = if fi.is_none() { SUNFALSE } else { SUNTRUE };
    }

    /* Allocate the general ARK stepper vectors using y0 as a template */
    /* NOTE: Fe, Fi, cvals and Xvecs will be allocated later on
    (based on the number of ARK stages) */

    /* Clone the input vector to create sdata, zpred and zcor */
    let mut sdata: Option<N_Vector> = None;
    if !arkAllocVec(&ark_mem, y0, &mut sdata) {
        let mut ark_mem = Some(ark_mem);
        ARKodeFree(&mut ark_mem);
        return None;
    }
    arkStep_mem_mut(&ark_mem).sdata = sdata;

    let mut zpred: Option<N_Vector> = None;
    if !arkAllocVec(&ark_mem, y0, &mut zpred) {
        let mut ark_mem = Some(ark_mem);
        ARKodeFree(&mut ark_mem);
        return None;
    }
    arkStep_mem_mut(&ark_mem).zpred = zpred;

    let mut zcor: Option<N_Vector> = None;
    if !arkAllocVec(&ark_mem, y0, &mut zcor) {
        let mut ark_mem = Some(ark_mem);
        ARKodeFree(&mut ark_mem);
        return None;
    }
    arkStep_mem_mut(&ark_mem).zcor = zcor;

    /* Copy the input parameters into ARKODE state */
    {
        let mut step_mem = arkStep_mem_mut(&ark_mem);
        step_mem.fe = fe;
        step_mem.fi = fi;
    }

    /* Update the ARKODE workspace requirements */
    {
        let mut m = ark_mem.borrow_mut();
        m.liw += 41; /* fcn/data ptr, int, long int, sunindextype, sunbooleantype */
        m.lrw += 10;
    }

    /* If an implicit component is to be solved, create default Newton NLS object */
    arkStep_mem_mut(&ark_mem).ownNLS = SUNFALSE;
    let implicit = arkStep_mem_mut(&ark_mem).implicit;
    if implicit {
        let sunctx_mem = ark_mem.borrow().sunctx.clone();
        let NLS = match SUNNonlinSol_Newton(y0, &sunctx_mem) {
            Some(NLS) => NLS,
            None => {
                arkProcessError(
                    Some(&ark_mem),
                    ARK_MEM_FAIL,
                    line!() as i32,
                    "ARKStepCreate",
                    file!(),
                    "Error creating default Newton solver",
                );
                let mut ark_mem = Some(ark_mem);
                ARKodeFree(&mut ark_mem);
                return None;
            }
        };
        let retval = ARKodeSetNonlinearSolver(&ark_mem, &NLS);
        if retval != ARK_SUCCESS {
            arkProcessError(
                Some(&ark_mem),
                ARK_MEM_FAIL,
                line!() as i32,
                "ARKStepCreate",
                file!(),
                "Error attaching default Newton solver",
            );
            let mut ark_mem = Some(ark_mem);
            ARKodeFree(&mut ark_mem);
            return None;
        }
        arkStep_mem_mut(&ark_mem).ownNLS = SUNTRUE;
    }

    {
        let mut step_mem = arkStep_mem_mut(&ark_mem);

        /* Set the linear solver addresses to NULL (we check != NULL later) */
        step_mem.linit = None;
        step_mem.lsetup = None;
        step_mem.lsolve = None;
        step_mem.lfree = None;
        step_mem.lmem = SUNFALSE;
        step_mem.lsolve_type = None; /* C: -1 */

        /* Set the mass matrix solver addresses to NULL */
        step_mem.minit = None;
        step_mem.msetup = None;
        step_mem.mmult = None;
        step_mem.msolve = None;
        step_mem.mfree = None;
        step_mem.mass_mem = SUNFALSE;
        step_mem.mass_type = MASS_IDENTITY;
        step_mem.msolve_type = None; /* C: -1 */

        /* Initialize initial error norm  */
        step_mem.eRNrm = ONE;

        /* Initialize all the counters */
        step_mem.nfe = 0;
        step_mem.nfi = 0;
        step_mem.nsetups = 0;
        step_mem.nstlp = 0;
        step_mem.nls_iters = 0;
        step_mem.nls_fails = 0;

        /* Initialize fused op work space */
        step_mem.cvals = Vec::new();
        step_mem.Xvecs = Vec::new();
        step_mem.nfusedopvecs = 0;

        /* Initialize external polynomial forcing data */
        step_mem.expforcing = SUNFALSE;
        step_mem.impforcing = SUNFALSE;
        step_mem.forcing = Vec::new();
        step_mem.nforcing = 0;

        /* Initialize saved fi alias */
        step_mem.fn_implicit = None;
    }

    /* Initialize main ARKODE infrastructure */
    let retval = arkInit(&ark_mem, t0, y0, FIRST_INIT);
    if retval != ARK_SUCCESS {
        arkProcessError(
            Some(&ark_mem),
            retval,
            line!() as i32,
            "ARKStepCreate",
            file!(),
            "Unable to initialize main ARKODE infrastructure",
        );
        let mut ark_mem = Some(ark_mem);
        ARKodeFree(&mut ark_mem);
        return None;
    }

    Some(ark_mem)
}

/*---------------------------------------------------------------
ARKStepReInit:

This routine re-initializes the ARKStep module to solve a new
problem of the same size as was previously solved. This routine
should also be called when the problem dynamics or desired solvers
have changed dramatically, so that the problem integration should
resume as if started from scratch.

Note all internal counters are set to 0 on re-initialization.
---------------------------------------------------------------*/
pub fn ARKStepReInit(
    arkode_mem: &ARKodeMem,
    fe: Option<ARKRhsFn>,
    fi: Option<ARKRhsFn>,
    t0: sunrealtype,
    y0: &N_Vector,
) -> i32 {
    let mut retval: i32;

    /* access ARKodeMem and ARKodeARKStepMem structures */
    retval = arkStep_AccessARKODEStepMem(arkode_mem, "ARKStepReInit");
    if retval != ARK_SUCCESS {
        return retval;
    }
    let ark_mem = arkode_mem;

    /* Check if ark_mem was allocated */
    if !ark_mem.borrow().MallocDone {
        arkProcessError(
            Some(ark_mem),
            ARK_NO_MALLOC,
            line!() as i32,
            "ARKStepReInit",
            file!(),
            MSG_ARK_NO_MALLOC,
        );
        return ARK_NO_MALLOC;
    }

    /* Check that at least one of fe, fi is supplied and is to be used */
    if fe.is_none() && fi.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "ARKStepReInit",
            file!(),
            MSG_ARK_NULL_F,
        );
        return ARK_ILL_INPUT;
    }

    /* Check that y0 is supplied: handled by the type system */

    {
        let mut step_mem = arkStep_mem_mut(ark_mem);

        /* Set implicit/explicit problem based on function pointers */
        step_mem.explicit = if fe.is_none() { SUNFALSE } else { SUNTRUE };
        step_mem.implicit = if fi.is_none() { SUNFALSE } else { SUNTRUE };

        /* Copy the input parameters into ARKODE state */
        step_mem.fe = fe;
        step_mem.fi = fi;

        /* Initialize initial error norm  */
        step_mem.eRNrm = ONE;
    }

    /* Initialize main ARKODE infrastructure */
    retval = arkInit(ark_mem, t0, y0, FIRST_INIT);
    if retval != ARK_SUCCESS {
        arkProcessError(
            Some(ark_mem),
            retval,
            line!() as i32,
            "ARKStepReInit",
            file!(),
            "Unable to reinitialize main ARKODE infrastructure",
        );
        return retval;
    }

    /* Initialize all the counters */
    let lmem = {
        let mut step_mem = arkStep_mem_mut(ark_mem);
        step_mem.nfe = 0;
        step_mem.nfi = 0;
        step_mem.nsetups = 0;
        step_mem.nstlp = 0;
        step_mem.lmem
    };

    if lmem {
        arkLsInitializeCounters(&mut arkls_mem_mut(ark_mem));
    }

    ARK_SUCCESS
}

/*===============================================================
Interface routines supplied to ARKODE
===============================================================*/

/*---------------------------------------------------------------
arkStep_Resize:

This routine resizes the memory within the ARKStep module.
---------------------------------------------------------------*/
pub fn arkStep_Resize(
    ark_mem: &ARKodeMem,
    y0: &N_Vector,
    hscale: sunrealtype,
    t0: sunrealtype,
    resize: Option<ARKVecResizeFn>,
    resize_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    /* C: SUNDIALS_MAYBE_UNUSED hscale, t0 */
    let _ = hscale;
    let _ = t0;

    let mut retval: i32;

    /* access ARKodeARKStepMem structure */
    retval = arkStep_AccessStepMem(ark_mem, "arkStep_Resize");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* Determine change in vector sizes */
    let mut lrw1: sunindextype = 0;
    let mut liw1: sunindextype = 0;
    if y0.ops.borrow().nvspace.is_some() {
        N_VSpace(y0, &mut lrw1, &mut liw1);
    }
    let (lrw_diff, liw_diff) = {
        let mut m = ark_mem.borrow_mut();
        let lrw_diff = lrw1 - m.lrw1;
        let liw_diff = liw1 - m.liw1;
        m.lrw1 = lrw1;
        m.liw1 = liw1;
        (lrw_diff, liw_diff)
    };

    /* Resize the sdata, zpred and zcor vectors */
    let mut sdata = arkStep_mem_mut(ark_mem).sdata.take();
    let ok = arkResizeVec(
        ark_mem,
        resize,
        resize_data,
        lrw_diff,
        liw_diff,
        y0,
        &mut sdata,
    );
    arkStep_mem_mut(ark_mem).sdata = sdata;
    if !ok {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_FAIL,
            line!() as i32,
            "arkStep_Resize",
            file!(),
            "Unable to resize vector",
        );
        return ARK_MEM_FAIL;
    }

    let mut zpred = arkStep_mem_mut(ark_mem).zpred.take();
    let ok = arkResizeVec(
        ark_mem,
        resize,
        resize_data,
        lrw_diff,
        liw_diff,
        y0,
        &mut zpred,
    );
    arkStep_mem_mut(ark_mem).zpred = zpred;
    if !ok {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_FAIL,
            line!() as i32,
            "arkStep_Resize",
            file!(),
            "Unable to resize vector",
        );
        return ARK_MEM_FAIL;
    }

    let mut zcor = arkStep_mem_mut(ark_mem).zcor.take();
    let ok = arkResizeVec(
        ark_mem,
        resize,
        resize_data,
        lrw_diff,
        liw_diff,
        y0,
        &mut zcor,
    );
    arkStep_mem_mut(ark_mem).zcor = zcor;
    if !ok {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_FAIL,
            line!() as i32,
            "arkStep_Resize",
            file!(),
            "Unable to resize vector",
        );
        return ARK_MEM_FAIL;
    }

    /* Resize the ARKStep vectors */
    /*     Fe */
    let (has_Fe, has_Fi, stages) = {
        let step_mem = arkStep_mem_mut(ark_mem);
        (
            !step_mem.Fe.is_empty(),
            !step_mem.Fi.is_empty(),
            step_mem.stages,
        )
    };
    if has_Fe {
        for i in 0..stages as usize {
            let mut v = Some(arkStep_mem_mut(ark_mem).Fe[i].clone());
            let ok = arkResizeVec(ark_mem, resize, resize_data, lrw_diff, liw_diff, y0, &mut v);
            if !ok {
                arkProcessError(
                    Some(ark_mem),
                    ARK_MEM_FAIL,
                    line!() as i32,
                    "arkStep_Resize",
                    file!(),
                    "Unable to resize vector",
                );
                return ARK_MEM_FAIL;
            }
            arkStep_mem_mut(ark_mem).Fe[i] = v.expect("resized Fe[i]");
        }
    }
    /*     Fi */
    if has_Fi {
        for i in 0..stages as usize {
            let mut v = Some(arkStep_mem_mut(ark_mem).Fi[i].clone());
            let ok = arkResizeVec(ark_mem, resize, resize_data, lrw_diff, liw_diff, y0, &mut v);
            if !ok {
                arkProcessError(
                    Some(ark_mem),
                    ARK_MEM_FAIL,
                    line!() as i32,
                    "arkStep_Resize",
                    file!(),
                    "Unable to resize vector",
                );
                return ARK_MEM_FAIL;
            }
            arkStep_mem_mut(ark_mem).Fi[i] = v.expect("resized Fi[i]");
        }
    }

    /* If a NLS object was previously used, destroy and recreate default Newton
    NLS object (can be replaced by user-defined object if desired) */
    let (nls, ownNLS) = {
        let step_mem = arkStep_mem_mut(ark_mem);
        (step_mem.NLS.clone(), step_mem.ownNLS)
    };
    if nls.is_some() && ownNLS {
        /* destroy existing NLS object (C clears the two fields only AFTER a
           successful free, so an error return leaves `NLS`/`ownNLS` intact) */
        retval = SUNNonlinSolFree(nls);
        if retval != ARK_SUCCESS {
            return retval;
        }
        {
            let mut step_mem = arkStep_mem_mut(ark_mem);
            step_mem.NLS = None;
            step_mem.ownNLS = SUNFALSE;
        }

        /* create new Newton NLS object */
        let sunctx = ark_mem.borrow().sunctx.clone();
        let NLS = match SUNNonlinSol_Newton(y0, &sunctx) {
            Some(NLS) => NLS,
            None => {
                arkProcessError(
                    Some(ark_mem),
                    ARK_MEM_FAIL,
                    line!() as i32,
                    "arkStep_Resize",
                    file!(),
                    "Error creating default Newton solver",
                );
                return ARK_MEM_FAIL;
            }
        };

        /* attach new Newton NLS object */
        retval = ARKodeSetNonlinearSolver(ark_mem, &NLS);
        if retval != ARK_SUCCESS {
            arkProcessError(
                Some(ark_mem),
                ARK_MEM_FAIL,
                line!() as i32,
                "arkStep_Resize",
                file!(),
                "Error attaching default Newton solver",
            );
            return ARK_MEM_FAIL;
        }
        arkStep_mem_mut(ark_mem).ownNLS = SUNTRUE;
    }

    /* reset nonlinear solver counters */
    {
        let mut step_mem = arkStep_mem_mut(ark_mem);
        if step_mem.NLS.is_some() {
            step_mem.nsetups = 0;
        }
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
arkStep_ComputeState:

Computes y based on the current prediction and given correction.
---------------------------------------------------------------*/
pub fn arkStep_ComputeState(ark_mem: &ARKodeMem, zcor: &N_Vector, z: &N_Vector) -> i32 {
    /* access ARKodeARKStepMem structure */
    let retval = arkStep_AccessStepMem(ark_mem, "arkStep_ComputeState");
    if retval != ARK_SUCCESS {
        return retval;
    }

    let zpred = arkStep_mem_mut(ark_mem).zpred.clone().expect("zpred");
    N_VLinearSum(ONE, &zpred, ONE, zcor, z);

    ARK_SUCCESS
}

/*---------------------------------------------------------------
arkStep_Free frees all ARKStep memory.
---------------------------------------------------------------*/
pub fn arkStep_Free(ark_mem: &ARKodeMem) {
    /* nothing to do if ark_mem is already NULL: handled by the type system */

    /* conditional frees on non-NULL ARKStep module */
    if ark_mem.borrow().step_mem.is_none() {
        return;
    }

    /* free the Butcher tables */
    let Be = arkStep_mem_mut(ark_mem).Be.take();
    if let Some(Be) = Be {
        let mut Bliw: sunindextype = 0;
        let mut Blrw: sunindextype = 0;
        ARKodeButcherTable_Space(Some(&Be), &mut Bliw, &mut Blrw);
        drop(Be); /* C: ARKodeButcherTable_Free(step_mem->Be) */
        let mut m = ark_mem.borrow_mut();
        m.liw -= Bliw;
        m.lrw -= Blrw;
    }
    let Bi = arkStep_mem_mut(ark_mem).Bi.take();
    if let Some(Bi) = Bi {
        let mut Bliw: sunindextype = 0;
        let mut Blrw: sunindextype = 0;
        ARKodeButcherTable_Space(Some(&Bi), &mut Bliw, &mut Blrw);
        drop(Bi); /* C: ARKodeButcherTable_Free(step_mem->Bi) */
        let mut m = ark_mem.borrow_mut();
        m.liw -= Bliw;
        m.lrw -= Blrw;
    }

    /* free the nonlinear solver memory (if applicable) */
    let (nls, ownNLS) = {
        let step_mem = arkStep_mem_mut(ark_mem);
        (step_mem.NLS.clone(), step_mem.ownNLS)
    };
    if nls.is_some() && ownNLS {
        arkStep_mem_mut(ark_mem).NLS = None;
        let _ = SUNNonlinSolFree(nls);
        arkStep_mem_mut(ark_mem).ownNLS = SUNFALSE;
    }
    arkStep_mem_mut(ark_mem).NLS = None;

    /* free the linear solver memory */
    let lfree = arkStep_mem_mut(ark_mem).lfree;
    if let Some(lfree) = lfree {
        let _ = lfree(ark_mem);
        arkStep_mem_mut(ark_mem).lmem = SUNFALSE;
    }

    /* free the mass matrix solver memory */
    let mfree = arkStep_mem_mut(ark_mem).mfree;
    if let Some(mfree) = mfree {
        let _ = mfree(ark_mem);
        arkStep_mem_mut(ark_mem).mass_mem = SUNFALSE;
    }

    /* free the sdata, zpred and zcor vectors */
    let mut sdata = arkStep_mem_mut(ark_mem).sdata.take();
    if sdata.is_some() {
        arkFreeVec(ark_mem, &mut sdata);
    }
    arkStep_mem_mut(ark_mem).sdata = None;

    let mut zpred = arkStep_mem_mut(ark_mem).zpred.take();
    if zpred.is_some() {
        arkFreeVec(ark_mem, &mut zpred);
    }
    arkStep_mem_mut(ark_mem).zpred = None;

    let mut zcor = arkStep_mem_mut(ark_mem).zcor.take();
    if zcor.is_some() {
        arkFreeVec(ark_mem, &mut zcor);
    }
    arkStep_mem_mut(ark_mem).zcor = None;

    let stages = arkStep_mem_mut(ark_mem).stages;

    /* free the RHS vectors */
    let mut Fe = std::mem::take(&mut arkStep_mem_mut(ark_mem).Fe);
    if !Fe.is_empty() {
        for j in 0..stages as usize {
            let mut v = Some(Fe[j].clone());
            arkFreeVec(ark_mem, &mut v);
        }
        Fe.clear(); /* C: free(step_mem->Fe); step_mem->Fe = NULL; */
        ark_mem.borrow_mut().liw -= stages as i64;
    }
    let mut Fi = std::mem::take(&mut arkStep_mem_mut(ark_mem).Fi);
    if !Fi.is_empty() {
        for j in 0..stages as usize {
            let mut v = Some(Fi[j].clone());
            arkFreeVec(ark_mem, &mut v);
        }
        Fi.clear();
        ark_mem.borrow_mut().liw -= stages as i64;
    }

    /* free stage vectors */
    let mut z = std::mem::take(&mut arkStep_mem_mut(ark_mem).z);
    if !z.is_empty() {
        for j in 0..stages as usize {
            let mut v = Some(z[j].clone());
            arkFreeVec(ark_mem, &mut v);
        }
        z.clear();
        ark_mem.borrow_mut().liw -= stages as i64;
    }

    /* free the reusable arrays for fused vector interface */
    {
        let nfusedopvecs = arkStep_mem_mut(ark_mem).nfusedopvecs;
        let had_cvals = !arkStep_mem_mut(ark_mem).cvals.is_empty();
        if had_cvals {
            arkStep_mem_mut(ark_mem).cvals = Vec::new();
            ark_mem.borrow_mut().lrw -= nfusedopvecs as i64;
        }
        let had_Xvecs = !arkStep_mem_mut(ark_mem).Xvecs.is_empty();
        if had_Xvecs {
            arkStep_mem_mut(ark_mem).Xvecs = Vec::new();
            ark_mem.borrow_mut().liw -= nfusedopvecs as i64;
        }
        arkStep_mem_mut(ark_mem).nfusedopvecs = 0;
    }

    /* free work arrays for MRI forcing */
    if !arkStep_mem_mut(ark_mem).stage_times.is_empty() {
        arkStep_mem_mut(ark_mem).stage_times = Vec::new();
        ark_mem.borrow_mut().lrw -= stages as i64;
    }

    if !arkStep_mem_mut(ark_mem).stage_coefs.is_empty() {
        arkStep_mem_mut(ark_mem).stage_coefs = Vec::new();
        ark_mem.borrow_mut().lrw -= stages as i64;
    }

    /* free the time stepper module itself */
    ark_mem.borrow_mut().step_mem = None;
}

/*---------------------------------------------------------------
arkStep_PrintMem:

This routine outputs the memory from the ARKStep structure to
a specified file pointer (useful when debugging).
---------------------------------------------------------------*/
pub fn arkStep_PrintMem(ark_mem: &ARKodeMem, outfile: &SUNFile) {
    /* access ARKodeARKStepMem structure */
    let retval = arkStep_AccessStepMem(ark_mem, "arkStep_PrintMem");
    if retval != ARK_SUCCESS {
        return;
    }

    let (
        q,
        p,
        istage,
        stages,
        maxcor,
        msbp,
        predictor,
        lsolve_type,
        msolve_type,
        convfail,
        nfe,
        nfi,
        nsetups,
        nstlp,
        linear,
        linear_timedep,
        explicit,
        implicit,
        jcur,
        Be,
        Bi,
        gamma,
        gammap,
        gamrat,
        crate_,
        eRNrm,
        nlscoef,
        crdown,
        rdiv,
        dgmax,
    ) = {
        let s = arkStep_mem_mut(ark_mem);
        (
            s.q,
            s.p,
            s.istage,
            s.stages,
            s.maxcor,
            s.msbp,
            s.predictor,
            lsolve_type_as_int(s.lsolve_type),
            lsolve_type_as_int(s.msolve_type),
            s.convfail,
            s.nfe,
            s.nfi,
            s.nsetups,
            s.nstlp,
            s.linear,
            s.linear_timedep,
            s.explicit,
            s.implicit,
            s.jcur.get(),
            s.Be.clone(),
            s.Bi.clone(),
            s.gamma,
            s.gammap,
            s.gamrat,
            s.crate_,
            s.eRNrm,
            s.nlscoef,
            s.crdown,
            s.rdiv,
            s.dgmax,
        )
    };

    /* output integer quantities */
    outfile.write_str(&format!("ARKStep: q = {q}\n"));
    outfile.write_str(&format!("ARKStep: p = {p}\n"));
    outfile.write_str(&format!("ARKStep: istage = {istage}\n"));
    outfile.write_str(&format!("ARKStep: stages = {stages}\n"));
    outfile.write_str(&format!("ARKStep: maxcor = {maxcor}\n"));
    outfile.write_str(&format!("ARKStep: msbp = {msbp}\n"));
    outfile.write_str(&format!("ARKStep: predictor = {predictor}\n"));
    outfile.write_str(&format!("ARKStep: lsolve_type = {lsolve_type}\n"));
    outfile.write_str(&format!("ARKStep: msolve_type = {msolve_type}\n"));
    outfile.write_str(&format!("ARKStep: convfail = {convfail}\n"));

    /* output long integer quantities */
    outfile.write_str(&format!("ARKStep: nfe = {nfe}\n"));
    outfile.write_str(&format!("ARKStep: nfi = {nfi}\n"));
    outfile.write_str(&format!("ARKStep: nsetups = {nsetups}\n"));
    outfile.write_str(&format!("ARKStep: nstlp = {nstlp}\n"));

    /* output boolean quantities */
    outfile.write_str(&format!("ARKStep: user_linear = {}\n", linear as i32));
    outfile.write_str(&format!(
        "ARKStep: user_linear_timedep = {}\n",
        linear_timedep as i32
    ));
    outfile.write_str(&format!("ARKStep: user_explicit = {}\n", explicit as i32));
    outfile.write_str(&format!("ARKStep: user_implicit = {}\n", implicit as i32));
    outfile.write_str(&format!("ARKStep: jcur = {}\n", jcur as i32));

    /* output sunrealtype quantities */
    if let Some(Be) = &Be {
        outfile.write_str("ARKStep: explicit Butcher table:\n");
        ARKodeButcherTable_Write(Some(Be), outfile);
    }
    if let Some(Bi) = &Bi {
        outfile.write_str("ARKStep: implicit Butcher table:\n");
        ARKodeButcherTable_Write(Some(Bi), outfile);
    }
    outfile.write_str(&format!("ARKStep: gamma = {}\n", sun_format_g(gamma)));
    outfile.write_str(&format!("ARKStep: gammap = {}\n", sun_format_g(gammap)));
    outfile.write_str(&format!("ARKStep: gamrat = {}\n", sun_format_g(gamrat)));
    outfile.write_str(&format!("ARKStep: crate = {}\n", sun_format_g(crate_)));
    outfile.write_str(&format!("ARKStep: eRNrm = {}\n", sun_format_g(eRNrm)));
    outfile.write_str(&format!("ARKStep: nlscoef = {}\n", sun_format_g(nlscoef)));
    outfile.write_str(&format!("ARKStep: crdown = {}\n", sun_format_g(crdown)));
    outfile.write_str(&format!("ARKStep: rdiv = {}\n", sun_format_g(rdiv)));
    outfile.write_str(&format!("ARKStep: dgmax = {}\n", sun_format_g(dgmax)));

    /* SUNDIALS_DEBUG_PRINTVEC vector output: not defined in this build */
}

/*---------------------------------------------------------------
arkStep_AttachLinsol:

This routine attaches the various set of system linear solver
interface routines, data structure, and solver type to the
ARKStep module.
---------------------------------------------------------------*/
pub fn arkStep_AttachLinsol(
    ark_mem: &ARKodeMem,
    linit: Option<ARKLinsolInitFn>,
    lsetup: Option<ARKLinsolSetupFn>,
    lsolve: Option<ARKLinsolSolveFn>,
    lfree: Option<ARKLinsolFreeFn>,
    lsolve_type: SUNLinearSolver_Type,
    lmem: Option<Box<dyn Any>>,
) -> i32 {
    /* access ARKodeARKStepMem structure */
    let retval = arkStep_AccessStepMem(ark_mem, "arkStep_AttachLinsol");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* free any existing system solver */
    let old_lfree = arkStep_mem_mut(ark_mem).lfree;
    if let Some(old_lfree) = old_lfree {
        let _ = old_lfree(ark_mem);
    }

    /* Attach the provided routines, data structure and solve type. The
    `lmem` record itself is owned by `ark_mem.ark_lmem` (contract §4). */
    let have_lmem = lmem.is_some();
    ark_mem.borrow_mut().ark_lmem = lmem;

    let mut step_mem = arkStep_mem_mut(ark_mem);
    step_mem.linit = linit;
    step_mem.lsetup = lsetup;
    step_mem.lsolve = lsolve;
    step_mem.lfree = lfree;
    step_mem.lmem = have_lmem;
    step_mem.lsolve_type = Some(lsolve_type);

    /* Reset all linear solver counters */
    step_mem.nsetups = 0;
    step_mem.nstlp = 0;

    ARK_SUCCESS
}

/*---------------------------------------------------------------
arkStep_AttachMasssol:

This routine attaches the set of mass matrix linear solver
interface routines, data structure, and solver type to the
ARKStep module.
---------------------------------------------------------------*/
#[allow(clippy::too_many_arguments)]
pub fn arkStep_AttachMasssol(
    ark_mem: &ARKodeMem,
    minit: Option<ARKMassInitFn>,
    msetup: Option<ARKMassSetupFn>,
    mmult: Option<ARKMassMultFn>,
    msolve: Option<ARKMassSolveFn>,
    mfree: Option<ARKMassFreeFn>,
    time_dep: sunbooleantype,
    msolve_type: SUNLinearSolver_Type,
    mass_mem: Option<Box<dyn Any>>,
) -> i32 {
    /* access ARKodeARKStepMem structure */
    let retval = arkStep_AccessStepMem(ark_mem, "arkStep_AttachMasssol");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* free any existing mass matrix solver */
    let old_mfree = arkStep_mem_mut(ark_mem).mfree;
    if let Some(old_mfree) = old_mfree {
        let _ = old_mfree(ark_mem);
    }

    /* Attach the provided routines, data structure and solve type. The
    `mass_mem` record itself is owned by `ark_mem.ark_mass_mem`. */
    let have_mass_mem = mass_mem.is_some();
    ark_mem.borrow_mut().ark_mass_mem = mass_mem;

    {
        let mut step_mem = arkStep_mem_mut(ark_mem);
        step_mem.minit = minit;
        step_mem.msetup = msetup;
        step_mem.mmult = mmult;
        step_mem.msolve = msolve;
        step_mem.mfree = mfree;
        step_mem.mass_mem = have_mass_mem;
        step_mem.mass_type = if time_dep { MASS_TIMEDEP } else { MASS_FIXED };
        step_mem.msolve_type = Some(msolve_type);
    }

    /* Attach mmult function pointer to ark_mem as well */
    ark_mem.borrow_mut().step_mmult = mmult;

    ARK_SUCCESS
}

/*---------------------------------------------------------------
arkStep_DisableLSetup:

This routine NULLifies the lsetup function pointer in the
ARKStep module.
---------------------------------------------------------------*/
pub fn arkStep_DisableLSetup(ark_mem: &ARKodeMem) {
    /* access ARKodeARKStepMem structure */
    if ark_mem.borrow().step_mem.is_none() {
        return;
    }

    /* nullify the lsetup function pointer */
    arkStep_mem_mut(ark_mem).lsetup = None;
}

/*---------------------------------------------------------------
arkStep_DisableMSetup:

This routine NULLifies the msetup function pointer in the
ARKStep module.
---------------------------------------------------------------*/
pub fn arkStep_DisableMSetup(ark_mem: &ARKodeMem) {
    /* access ARKodeARKStepMem structure */
    if ark_mem.borrow().step_mem.is_none() {
        return;
    }

    /* nullify the msetup function pointer */
    arkStep_mem_mut(ark_mem).msetup = None;
}

/*---------------------------------------------------------------
arkStep_GetLmem:

This routine reports whether a system linear solver interface
memory structure, lmem, is attached (contract §4: the record
itself lives in `ark_mem.ark_lmem` and is reached with
`arkls_mem_mut`).
---------------------------------------------------------------*/
pub fn arkStep_GetLmem(ark_mem: &ARKodeMem) -> sunbooleantype {
    /* access ARKodeARKStepMem structure, and return lmem */
    let retval = arkStep_AccessStepMem(ark_mem, "arkStep_GetLmem");
    if retval != ARK_SUCCESS {
        return SUNFALSE;
    }
    arkStep_mem_mut(ark_mem).lmem
}

/*---------------------------------------------------------------
arkStep_GetMassMem:

This routine reports whether a mass matrix solver interface
memory structure, mass_mem, is attached.
---------------------------------------------------------------*/
pub fn arkStep_GetMassMem(ark_mem: &ARKodeMem) -> sunbooleantype {
    /* access ARKodeARKStepMem structure, and return mass_mem */
    let retval = arkStep_AccessStepMem(ark_mem, "arkStep_GetMassMem");
    if retval != ARK_SUCCESS {
        return SUNFALSE;
    }
    arkStep_mem_mut(ark_mem).mass_mem
}

/*---------------------------------------------------------------
arkStep_GetImplicitRHS:

This routine returns the implicit RHS function pointer, fi.
---------------------------------------------------------------*/
pub fn arkStep_GetImplicitRHS(ark_mem: &ARKodeMem) -> Option<ARKRhsFn> {
    /* access ARKodeARKStepMem structure, and return fi */
    let retval = arkStep_AccessStepMem(ark_mem, "arkStep_GetImplicitRHS");
    if retval != ARK_SUCCESS {
        return None;
    }
    arkStep_mem_mut(ark_mem).fi
}

/*---------------------------------------------------------------
arkStep_GetGammas:

This routine fills the current value of gamma, and states
whether the gamma ratio fails the dgmax criteria.
---------------------------------------------------------------*/
pub fn arkStep_GetGammas(
    ark_mem: &ARKodeMem,
    gamma: &mut sunrealtype,
    gamrat: &mut sunrealtype,
    jcur: &mut Option<ARKJcurPtr>,
    dgamma_fail: &mut sunbooleantype,
) -> i32 {
    /* access ARKodeARKStepMem structure */
    let retval = arkStep_AccessStepMem(ark_mem, "arkStep_GetGammas");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* set outputs */
    let step_mem = arkStep_mem_mut(ark_mem);
    *gamma = step_mem.gamma;
    *gamrat = step_mem.gamrat;
    *jcur = Some(step_mem.jcur.clone());
    *dgamma_fail = SUNRabs(*gamrat - ONE) >= step_mem.dgmax;

    ARK_SUCCESS
}

/*---------------------------------------------------------------
arkStep_Init:

This routine is called just prior to performing internal time
steps (after all user "set" routines have been called) from
within arkInitialSetup.

For all initialization types, this routine sets the relevant
TakeStep routine based on the current problem configuration.

With initialization type FIRST_INIT this routine:
- sets/checks the ARK Butcher tables to be used
- allocates any memory that depends on the number of ARK stages,
  method order, or solver options
- checks for consistency between the system and mass matrix
  linear solvers (if applicable)
- initializes and sets up the system and mass matrix linear
  solvers (if applicable)
- initializes and sets up the nonlinear solver (if applicable)
- allocates the interpolation data structure (if needed based
  on ARKStep solver options)
- updates the call_fullrhs flag if necessary

With initialization type FIRST_INIT or RESIZE_INIT, this routine:
- sets the relevant TakeStep routine based on the current
  problem configuration
- checks for consistency between the system and mass matrix
  linear solvers (if applicable)
- initializes and sets up the system and mass matrix linear
  solvers (if applicable)
- initializes and sets up the nonlinear solver (if applicable)

With initialization type RESET_INIT, this routine does nothing.
---------------------------------------------------------------*/
pub fn arkStep_Init(ark_mem: &ARKodeMem, init_type: i32) -> i32 {
    let mut retval: i32;

    /* access ARKodeARKStepMem structure */
    retval = arkStep_AccessStepMem(ark_mem, "arkStep_Init");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* immediately return if reset */
    if init_type == RESET_INIT {
        return ARK_SUCCESS;
    }

    /* initializations/checks for (re-)initialization call */
    if init_type == FIRST_INIT {
        /* enforce use of arkEwtSmallReal if using a fixed step size for
        an explicit method, an internal error weight function, not
        using an iterative mass matrix solver with rwt=ewt, and not
        performing accumulated temporal error estimation */
        let mut reset_efun: sunbooleantype = SUNTRUE;
        let (implicit, msolve_type) = {
            let s = arkStep_mem_mut(ark_mem);
            (s.implicit, s.msolve_type)
        };
        if implicit {
            reset_efun = SUNFALSE;
        }
        {
            let m = ark_mem.borrow();
            if !m.fixedstep {
                reset_efun = SUNFALSE;
            }
            if m.user_efun {
                reset_efun = SUNFALSE;
            }
            if m.AccumErrorType != ARK_ACCUMERROR_NONE {
                reset_efun = SUNFALSE;
            }
            if m.rwt_is_ewt && (msolve_type == Some(SUNLINEARSOLVER_ITERATIVE)) {
                reset_efun = SUNFALSE;
            }
            if m.rwt_is_ewt && (msolve_type == Some(SUNLINEARSOLVER_MATRIX_ITERATIVE)) {
                reset_efun = SUNFALSE;
            }
        }
        if reset_efun {
            let mut m = ark_mem.borrow_mut();
            m.user_efun = SUNFALSE;
            m.efun = Some(arkEwtSetSmallReal);
            /* C `ark_mem->e_data = ark_mem` is a self-alias no `Box` can
            hold (deviation class 6); `arkEwtSetSmallReal` ignores its data
            argument entirely, so the slot is cleared instead. */
            m.e_data = None;
        }

        /* Create Butcher tables (if not already set) */
        retval = arkStep_SetButcherTables(ark_mem);
        if retval != ARK_SUCCESS {
            arkProcessError(
                Some(ark_mem),
                ARK_ILL_INPUT,
                line!() as i32,
                "arkStep_Init",
                file!(),
                "Could not create Butcher table(s)",
            );
            return ARK_ILL_INPUT;
        }

        /* Check that Butcher tables are OK */
        retval = arkStep_CheckButcherTables(ark_mem);
        if retval != ARK_SUCCESS {
            arkProcessError(
                Some(ark_mem),
                ARK_ILL_INPUT,
                line!() as i32,
                "arkStep_Init",
                file!(),
                "Error in Butcher table(s)",
            );
            return ARK_ILL_INPUT;
        }

        /* Retrieve/store method and embedding orders now that tables are finalized */
        let (q, p) = {
            let s = arkStep_mem_mut(ark_mem);
            match &s.Bi {
                Some(Bi) => {
                    let Bi = Bi.borrow();
                    (Bi.q, Bi.p)
                }
                None => {
                    let Be = s.Be.as_ref().expect("Be").borrow();
                    (Be.q, Be.p)
                }
            }
        };
        {
            let mut s = arkStep_mem_mut(ark_mem);
            s.q = q;
            s.p = p;
        }
        {
            let mut m = ark_mem.borrow_mut();
            let hadapt_mem = m.hadapt_mem.as_mut().expect("hadapt_mem");
            hadapt_mem.q = q;
            hadapt_mem.p = p;
        }

        /* Ensure that if adaptivity or error accumulation is enabled, then
        method includes embedding coefficients */
        let (fixedstep, accum_error_type) = {
            let m = ark_mem.borrow();
            (m.fixedstep, m.AccumErrorType)
        };
        if (!fixedstep || (accum_error_type != ARK_ACCUMERROR_NONE)) && (p <= 0) {
            arkProcessError(
                Some(ark_mem),
                ARK_ILL_INPUT,
                line!() as i32,
                "arkStep_Init",
                file!(),
                "Temporal error estimation cannot be performed without embedding coefficients",
            );
            return ARK_ILL_INPUT;
        }

        /* Relaxation is incompatible with implicit RHS deduction */
        let (implicit, deduce_rhs, explicit, mass_type, stages, nforcing) = {
            let s = arkStep_mem_mut(ark_mem);
            (
                s.implicit,
                s.deduce_rhs,
                s.explicit,
                s.mass_type,
                s.stages,
                s.nforcing,
            )
        };
        let relax_enabled = ark_mem.borrow().relax_enabled;
        if relax_enabled && implicit && deduce_rhs {
            arkProcessError(
                Some(ark_mem),
                ARK_ILL_INPUT,
                line!() as i32,
                "arkStep_Init",
                file!(),
                "Relaxation cannot be performed when deducing implicit RHS values",
            );
            return ARK_ILL_INPUT;
        }

        /* Allocate ARK RHS vector memory, update storage requirements */
        let (ewt, lrw1, liw1) = {
            let m = ark_mem.borrow();
            (m.ewt.clone().expect("ewt"), m.lrw1, m.liw1)
        };

        /*   Allocate Fe[0] ... Fe[stages-1] if needed */
        if explicit {
            let (mut lrw, mut liw) = {
                let m = ark_mem.borrow();
                (m.lrw, m.liw)
            };
            let mut Fe = std::mem::take(&mut arkStep_mem_mut(ark_mem).Fe);
            let ok = arkAllocVecArray(stages, &ewt, &mut Fe, lrw1, &mut lrw, liw1, &mut liw);
            arkStep_mem_mut(ark_mem).Fe = Fe;
            {
                let mut m = ark_mem.borrow_mut();
                m.lrw = lrw;
                m.liw = liw;
            }
            if !ok {
                return ARK_MEM_FAIL;
            }
        }

        /*   Allocate Fi[0] ... Fi[stages-1] if needed */
        if implicit {
            let (mut lrw, mut liw) = {
                let m = ark_mem.borrow();
                (m.lrw, m.liw)
            };
            let mut Fi = std::mem::take(&mut arkStep_mem_mut(ark_mem).Fi);
            let ok = arkAllocVecArray(stages, &ewt, &mut Fi, lrw1, &mut lrw, liw1, &mut liw);
            arkStep_mem_mut(ark_mem).Fi = Fi;
            {
                let mut m = ark_mem.borrow_mut();
                m.lrw = lrw;
                m.liw = liw;
            }
            if !ok {
                return ARK_MEM_FAIL;
            }
        }

        /* Allocate stage storage for relaxation with implicit/IMEX methods or if a
        fixed mass matrix is present (since we store f(t,y) not M^{-1} f(t,y)) */
        if relax_enabled && (implicit || mass_type == MASS_FIXED) {
            let (mut lrw, mut liw) = {
                let m = ark_mem.borrow();
                (m.lrw, m.liw)
            };
            let mut z = std::mem::take(&mut arkStep_mem_mut(ark_mem).z);
            let ok = arkAllocVecArray(stages, &ewt, &mut z, lrw1, &mut lrw, liw1, &mut liw);
            arkStep_mem_mut(ark_mem).z = z;
            {
                let mut m = ark_mem.borrow_mut();
                m.lrw = lrw;
                m.liw = liw;
            }
            if !ok {
                return ARK_MEM_FAIL;
            }
        }

        /* Allocate reusable arrays for fused vector operations */
        let nfusedopvecs = 2 * stages + 2 + nforcing;
        arkStep_mem_mut(ark_mem).nfusedopvecs = nfusedopvecs;
        if arkStep_mem_mut(ark_mem).cvals.is_empty() {
            arkStep_mem_mut(ark_mem).cvals = vec![ZERO; nfusedopvecs as usize];
            ark_mem.borrow_mut().lrw += nfusedopvecs as i64;
        }
        if arkStep_mem_mut(ark_mem).Xvecs.is_empty() {
            arkStep_mem_mut(ark_mem).Xvecs = vec![None; nfusedopvecs as usize];
            ark_mem.borrow_mut().liw += nfusedopvecs as i64; /* pointers */
        }

        /* Allocate workspace for MRI forcing -- need to allocate here as the
        number of stages may not be set before this point */
        if arkStep_mem_mut(ark_mem).stage_times.is_empty() {
            arkStep_mem_mut(ark_mem).stage_times = vec![ZERO; stages as usize];
            ark_mem.borrow_mut().lrw += stages as i64;
        }

        if arkStep_mem_mut(ark_mem).stage_coefs.is_empty() {
            arkStep_mem_mut(ark_mem).stage_coefs = vec![ZERO; stages as usize];
            ark_mem.borrow_mut().lrw += stages as i64;
        }

        /* Override the interpolant degree (if needed), used in arkInitialSetup */
        {
            let mut m = ark_mem.borrow_mut();
            if q > 1 && m.interp_degree > (q - 1) {
                /* Limit max degree to at most one less than the method global order */
                m.interp_degree = q - 1;
            } else if q == 1 && m.interp_degree > 1 {
                /* Allow for linear interpolant with first order methods to ensure
                solution values are returned at the time interval end points */
                m.interp_degree = 1;
            }
        }

        /* Higher-order predictors require interpolation */
        let interp_type = ark_mem.borrow().interp_type;
        let predictor = arkStep_mem_mut(ark_mem).predictor;
        if interp_type == ARK_INTERP_NONE && predictor != 0 {
            arkProcessError(
                Some(ark_mem),
                ARK_ILL_INPUT,
                line!() as i32,
                "arkStep_Init",
                file!(),
                "Non-trival predictors require an interpolation module",
            );
            return ARK_ILL_INPUT;
        }
    }

    /* set appropriate TakeStep routine based on problem configuration */
    {
        let mut m = ark_mem.borrow_mut();
        if m.do_adjoint {
            m.step = Some(arkStep_TakeStep_ERK_Adjoint);
        } else {
            m.step = Some(arkStep_TakeStep_Z);
        }
    }

    /* Check for consistency between mass system and system linear system modules
    (e.g., if lsolve is direct, msolve needs to match) */
    let (mass_type, lmem, lsolve_type, msolve_type, minit, msetup, linit, has_nls) = {
        let s = arkStep_mem_mut(ark_mem);
        (
            s.mass_type,
            s.lmem,
            s.lsolve_type,
            s.msolve_type,
            s.minit,
            s.msetup,
            s.linit,
            s.NLS.is_some(),
        )
    };
    if (mass_type != MASS_IDENTITY) && lmem {
        if lsolve_type != msolve_type {
            arkProcessError(
                Some(ark_mem),
                ARK_ILL_INPUT,
                line!() as i32,
                "arkStep_Init",
                file!(),
                "Incompatible linear and mass matrix solvers",
            );
            return ARK_ILL_INPUT;
        }
    }

    /* Perform mass matrix solver initialization and setup (if applicable) */
    if mass_type != MASS_IDENTITY {
        /* Call minit (if it exists) */
        if let Some(minit) = minit {
            retval = minit(ark_mem);
            if retval != 0 {
                arkProcessError(
                    Some(ark_mem),
                    ARK_MASSINIT_FAIL,
                    line!() as i32,
                    "arkStep_Init",
                    file!(),
                    MSG_ARK_MASSINIT_FAIL,
                );
                return ARK_MASSINIT_FAIL;
            }
        }

        /* Call msetup (if it exists) */
        if let Some(msetup) = msetup {
            let (tcur, tempv1, tempv2, tempv3) = {
                let m = ark_mem.borrow();
                (
                    m.tcur,
                    m.tempv1.clone().expect("tempv1"),
                    m.tempv2.clone().expect("tempv2"),
                    m.tempv3.clone().expect("tempv3"),
                )
            };
            retval = msetup(ark_mem, tcur, &tempv1, &tempv2, &tempv3);
            if retval != 0 {
                arkProcessError(
                    Some(ark_mem),
                    ARK_MASSSETUP_FAIL,
                    line!() as i32,
                    "arkStep_Init",
                    file!(),
                    MSG_ARK_MASSSETUP_FAIL,
                );
                return ARK_MASSSETUP_FAIL;
            }
        }
    }

    /* Call linit (if it exists) */
    if let Some(linit) = linit {
        retval = linit(ark_mem);
        if retval != 0 {
            arkProcessError(
                Some(ark_mem),
                ARK_LINIT_FAIL,
                line!() as i32,
                "arkStep_Init",
                file!(),
                MSG_ARK_LINIT_FAIL,
            );
            return ARK_LINIT_FAIL;
        }
    }

    /* Initialize the nonlinear solver object (if it exists) */
    if has_nls {
        retval = arkStep_NlsInit(ark_mem);
        if retval != ARK_SUCCESS {
            arkProcessError(
                Some(ark_mem),
                ARK_NLS_INIT_FAIL,
                line!() as i32,
                "arkStep_Init",
                file!(),
                "Unable to initialize SUNNonlinearSolver object",
            );
            return ARK_NLS_INIT_FAIL;
        }
    }

    /* Signal to shared arkode module that full RHS evaluations are required */
    ark_mem.borrow_mut().call_fullrhs = SUNTRUE;

    ARK_SUCCESS
}

/*------------------------------------------------------------------------------
arkStep_FullRHS:

Rewriting the problem
  My' = fe(t,y) + fi(t,y)
in the form
  y' = M^{-1}*[ fe(t,y) + fi(t,y) ],
this routine computes the full right-hand side vector,
  f = M^{-1}*[ fe(t,y) + fi(t,y) ]

See the C source for the full description of the three 'modes'
(ARK_FULLRHS_START / ARK_FULLRHS_END / ARK_FULLRHS_OTHER) and of
which stored values may be reused in each.
----------------------------------------------------------------------------*/
pub fn arkStep_FullRHS(
    ark_mem: &ARKodeMem,
    t: sunrealtype,
    y: &N_Vector,
    f: &N_Vector,
    mode: i32,
) -> i32 {
    let mut retval: i32;
    let stage_coefs: sunrealtype = ONE;

    /* access ARKodeARKStepMem structure */
    retval = arkStep_AccessStepMem(ark_mem, "arkStep_FullRHS");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* local shortcuts for use with fused vector operations: the operand
    lists are built locally (see the module docs) */

    let (mass_type, msetup, msolve, implicit, explicit, fi, fe, nlscoef, stages) = {
        let s = arkStep_mem_mut(ark_mem);
        (
            s.mass_type,
            s.msetup,
            s.msolve,
            s.implicit,
            s.explicit,
            s.fi,
            s.fe,
            s.nlscoef,
            s.stages,
        )
    };

    /* setup mass-matrix if required (use output f as a temporary) */
    if (mass_type == MASS_TIMEDEP) && msetup.is_some() {
        let (tempv2, tempv3) = {
            let m = ark_mem.borrow();
            (
                m.tempv2.clone().expect("tempv2"),
                m.tempv3.clone().expect("tempv3"),
            )
        };
        retval = (msetup.expect("msetup"))(ark_mem, t, f, &tempv2, &tempv3);
        if retval != ARK_SUCCESS {
            return ARK_MASSSETUP_FAIL;
        }
    }

    /* perform RHS functions contingent on 'mode' argument */
    match mode {
        ARK_FULLRHS_START => {
            /* compute the full RHS */
            if !ark_mem.borrow().fn_is_current {
                /* call the user-supplied pre-RHS function (if supplied) */
                let PreRhsFn = ark_mem.borrow().PreRhsFn;
                if let Some(PreRhsFn) = PreRhsFn {
                    let mut user_data = ark_mem.borrow_mut().user_data.take();
                    retval = PreRhsFn(t, y, &mut user_data);
                    ark_mem.borrow_mut().user_data = user_data;
                    if retval != 0 {
                        arkProcessError(
                            Some(ark_mem),
                            ARK_PRERHSFN_FAIL,
                            line!() as i32,
                            "arkStep_FullRHS",
                            file!(),
                            &MSG_ARK_PRERHSFN_FAIL(t),
                        );
                        return ARK_PRERHSFN_FAIL;
                    }
                }

                /* compute the implicit component */
                if implicit {
                    let Fi0 = arkStep_mem_mut(ark_mem).Fi[0].clone();
                    let mut user_data = ark_mem.borrow_mut().user_data.take();
                    retval = (fi.expect("fi"))(t, y, &Fi0, &mut user_data);
                    ark_mem.borrow_mut().user_data = user_data;
                    arkStep_mem_mut(ark_mem).nfi += 1;
                    if retval != 0 {
                        arkProcessError(
                            Some(ark_mem),
                            ARK_RHSFUNC_FAIL,
                            line!() as i32,
                            "arkStep_FullRHS",
                            file!(),
                            &MSG_ARK_RHSFUNC_FAILED(t),
                        );
                        return ARK_RHSFUNC_FAIL;
                    }

                    /* compute and store M(t)^{-1} fi */
                    if mass_type == MASS_TIMEDEP {
                        let h = ark_mem.borrow().h;
                        retval = (msolve.expect("msolve"))(ark_mem, &Fi0, nlscoef / h);
                        if retval != 0 {
                            arkProcessError(
                                Some(ark_mem),
                                ARK_MASSSOLVE_FAIL,
                                line!() as i32,
                                "arkStep_FullRHS",
                                file!(),
                                "Mass matrix solver failure",
                            );
                            return ARK_MASSSOLVE_FAIL;
                        }
                    }
                }

                /* compute the explicit component */
                if explicit {
                    let Fe0 = arkStep_mem_mut(ark_mem).Fe[0].clone();
                    let mut user_data = ark_mem.borrow_mut().user_data.take();
                    retval = (fe.expect("fe"))(t, y, &Fe0, &mut user_data);
                    ark_mem.borrow_mut().user_data = user_data;
                    arkStep_mem_mut(ark_mem).nfe += 1;
                    if retval != 0 {
                        arkProcessError(
                            Some(ark_mem),
                            ARK_RHSFUNC_FAIL,
                            line!() as i32,
                            "arkStep_FullRHS",
                            file!(),
                            &MSG_ARK_RHSFUNC_FAILED(t),
                        );
                        return ARK_RHSFUNC_FAIL;
                    }

                    /* compute and store M(t)^{-1} fe */
                    if mass_type == MASS_TIMEDEP {
                        let h = ark_mem.borrow().h;
                        retval = (msolve.expect("msolve"))(ark_mem, &Fe0, nlscoef / h);
                        if retval != 0 {
                            arkProcessError(
                                Some(ark_mem),
                                ARK_MASSSOLVE_FAIL,
                                line!() as i32,
                                "arkStep_FullRHS",
                                file!(),
                                "Mass matrix solver failure",
                            );
                            return ARK_MASSSOLVE_FAIL;
                        }
                    }
                }
            }

            /* combine RHS vector(s) into output */
            if explicit && implicit {
                /* ImEx */
                let (Fi0, Fe0) = {
                    let s = arkStep_mem_mut(ark_mem);
                    (s.Fi[0].clone(), s.Fe[0].clone())
                };
                N_VLinearSum(ONE, &Fi0, ONE, &Fe0, f);
            } else if implicit {
                /* implicit */
                let Fi0 = arkStep_mem_mut(ark_mem).Fi[0].clone();
                N_VScale(ONE, &Fi0, f);
            } else {
                /* explicit */
                let Fe0 = arkStep_mem_mut(ark_mem).Fe[0].clone();
                N_VScale(ONE, &Fe0, f);
            }

            /* compute M^{-1} f for output but do not store */
            if mass_type == MASS_FIXED {
                let h = ark_mem.borrow().h;
                retval = (msolve.expect("msolve"))(ark_mem, f, nlscoef / h);
                if retval != 0 {
                    arkProcessError(
                        Some(ark_mem),
                        ARK_MASSSOLVE_FAIL,
                        line!() as i32,
                        "arkStep_FullRHS",
                        file!(),
                        "Mass matrix solver failure",
                    );
                    return ARK_MASSSOLVE_FAIL;
                }
            }

            /* apply external polynomial (MRI) forcing (M = I required) */
            arkStep_FullRHS_ApplyForcing(ark_mem, t, stage_coefs, f);
        }

        ARK_FULLRHS_END => {
            /* compute the full RHS */
            if !ark_mem.borrow().fn_is_current {
                /* determine if RHS functions need to be recomputed */
                let mut recomputeRHS: sunbooleantype = SUNFALSE;

                if explicit {
                    let Be = arkStep_mem_mut(ark_mem).Be.clone().expect("Be");
                    if !ARKodeButcherTable_IsStifflyAccurate(Some(&Be)) {
                        recomputeRHS = SUNTRUE;
                    }
                }

                if implicit {
                    let Bi = arkStep_mem_mut(ark_mem).Bi.clone().expect("Bi");
                    if !ARKodeButcherTable_IsStifflyAccurate(Some(&Bi)) {
                        recomputeRHS = SUNTRUE;
                    }
                }

                /* Stiffly Accurate methods are not SA when relaxation is enabled */
                if ark_mem.borrow().relax_enabled {
                    recomputeRHS = SUNTRUE;
                }

                /* recompute RHS functions */
                if recomputeRHS {
                    /* call the user-supplied pre-RHS function (if supplied) */
                    let PreRhsFn = ark_mem.borrow().PreRhsFn;
                    if let Some(PreRhsFn) = PreRhsFn {
                        let mut user_data = ark_mem.borrow_mut().user_data.take();
                        retval = PreRhsFn(t, y, &mut user_data);
                        ark_mem.borrow_mut().user_data = user_data;
                        if retval != 0 {
                            arkProcessError(
                                Some(ark_mem),
                                ARK_PRERHSFN_FAIL,
                                line!() as i32,
                                "arkStep_FullRHS",
                                file!(),
                                &MSG_ARK_PRERHSFN_FAIL(t),
                            );
                            return ARK_PRERHSFN_FAIL;
                        }
                    }

                    /* compute the implicit component */
                    if implicit {
                        let Fi0 = arkStep_mem_mut(ark_mem).Fi[0].clone();
                        let mut user_data = ark_mem.borrow_mut().user_data.take();
                        retval = (fi.expect("fi"))(t, y, &Fi0, &mut user_data);
                        ark_mem.borrow_mut().user_data = user_data;
                        arkStep_mem_mut(ark_mem).nfi += 1;
                        if retval != 0 {
                            arkProcessError(
                                Some(ark_mem),
                                ARK_RHSFUNC_FAIL,
                                line!() as i32,
                                "arkStep_FullRHS",
                                file!(),
                                &MSG_ARK_RHSFUNC_FAILED(t),
                            );
                            return ARK_RHSFUNC_FAIL;
                        }

                        /* compute and store M(t)^{-1} fi */
                        if mass_type == MASS_TIMEDEP {
                            let h = ark_mem.borrow().h;
                            retval = (msolve.expect("msolve"))(ark_mem, &Fi0, nlscoef / h);
                            if retval != 0 {
                                arkProcessError(
                                    Some(ark_mem),
                                    ARK_MASSSOLVE_FAIL,
                                    line!() as i32,
                                    "arkStep_FullRHS",
                                    file!(),
                                    "Mass matrix solver failure",
                                );
                                return ARK_MASSSOLVE_FAIL;
                            }
                        }
                    }

                    /* compute the explicit component */
                    if explicit {
                        let Fe0 = arkStep_mem_mut(ark_mem).Fe[0].clone();
                        let mut user_data = ark_mem.borrow_mut().user_data.take();
                        retval = (fe.expect("fe"))(t, y, &Fe0, &mut user_data);
                        ark_mem.borrow_mut().user_data = user_data;
                        arkStep_mem_mut(ark_mem).nfe += 1;
                        if retval != 0 {
                            arkProcessError(
                                Some(ark_mem),
                                ARK_RHSFUNC_FAIL,
                                line!() as i32,
                                "arkStep_FullRHS",
                                file!(),
                                &MSG_ARK_RHSFUNC_FAILED(t),
                            );
                            return ARK_RHSFUNC_FAIL;
                        }

                        /* compute and store M(t)^{-1} fi */
                        if mass_type == MASS_TIMEDEP {
                            let h = ark_mem.borrow().h;
                            retval = (msolve.expect("msolve"))(ark_mem, &Fe0, nlscoef / h);
                            if retval != 0 {
                                arkProcessError(
                                    Some(ark_mem),
                                    ARK_MASSSOLVE_FAIL,
                                    line!() as i32,
                                    "arkStep_FullRHS",
                                    file!(),
                                    "Mass matrix solver failure",
                                );
                                return ARK_MASSSOLVE_FAIL;
                            }
                        }
                    }
                } else {
                    if explicit {
                        let (Fe_last, Fe0) = {
                            let s = arkStep_mem_mut(ark_mem);
                            (s.Fe[(stages - 1) as usize].clone(), s.Fe[0].clone())
                        };
                        N_VScale(ONE, &Fe_last, &Fe0);
                    }
                    if implicit {
                        let (Fi_last, Fi0) = {
                            let s = arkStep_mem_mut(ark_mem);
                            (s.Fi[(stages - 1) as usize].clone(), s.Fi[0].clone())
                        };
                        N_VScale(ONE, &Fi_last, &Fi0);
                    }
                }
            }

            /* combine RHS vector(s) into output */
            if explicit && implicit {
                /* ImEx */
                let (Fi0, Fe0) = {
                    let s = arkStep_mem_mut(ark_mem);
                    (s.Fi[0].clone(), s.Fe[0].clone())
                };
                N_VLinearSum(ONE, &Fi0, ONE, &Fe0, f);
            } else if implicit {
                /* implicit */
                let Fi0 = arkStep_mem_mut(ark_mem).Fi[0].clone();
                N_VScale(ONE, &Fi0, f);
            } else {
                /* explicit */
                let Fe0 = arkStep_mem_mut(ark_mem).Fe[0].clone();
                N_VScale(ONE, &Fe0, f);
            }

            /* compute M^{-1} f for output but do not store */
            if mass_type == MASS_FIXED {
                let h = ark_mem.borrow().h;
                retval = (msolve.expect("msolve"))(ark_mem, f, nlscoef / h);
                if retval != 0 {
                    arkProcessError(
                        Some(ark_mem),
                        ARK_MASSSOLVE_FAIL,
                        line!() as i32,
                        "arkStep_FullRHS",
                        file!(),
                        "Mass matrix solver failure",
                    );
                    return ARK_MASSSOLVE_FAIL;
                }
            }

            /* apply external polynomial (MRI) forcing (M = I required) */
            arkStep_FullRHS_ApplyForcing(ark_mem, t, stage_coefs, f);
        }

        ARK_FULLRHS_OTHER => {
            /* call the user-supplied pre-RHS function (if supplied) */
            let PreRhsFn = ark_mem.borrow().PreRhsFn;
            if let Some(PreRhsFn) = PreRhsFn {
                let mut user_data = ark_mem.borrow_mut().user_data.take();
                retval = PreRhsFn(t, y, &mut user_data);
                ark_mem.borrow_mut().user_data = user_data;
                if retval != 0 {
                    arkProcessError(
                        Some(ark_mem),
                        ARK_PRERHSFN_FAIL,
                        line!() as i32,
                        "arkStep_FullRHS",
                        file!(),
                        &MSG_ARK_PRERHSFN_FAIL(t),
                    );
                    return ARK_PRERHSFN_FAIL;
                }
            }

            /* compute the implicit component and store in sdata */
            if implicit {
                let sdata = arkStep_mem_mut(ark_mem).sdata.clone().expect("sdata");
                let mut user_data = ark_mem.borrow_mut().user_data.take();
                retval = (fi.expect("fi"))(t, y, &sdata, &mut user_data);
                ark_mem.borrow_mut().user_data = user_data;
                arkStep_mem_mut(ark_mem).nfi += 1;
                if retval != 0 {
                    arkProcessError(
                        Some(ark_mem),
                        ARK_RHSFUNC_FAIL,
                        line!() as i32,
                        "arkStep_FullRHS",
                        file!(),
                        &MSG_ARK_RHSFUNC_FAILED(t),
                    );
                    return ARK_RHSFUNC_FAIL;
                }
            }

            /* compute the explicit component and store in ark_tempv2 */
            if explicit {
                let tempv2 = ark_mem.borrow().tempv2.clone().expect("tempv2");
                let mut user_data = ark_mem.borrow_mut().user_data.take();
                retval = (fe.expect("fe"))(t, y, &tempv2, &mut user_data);
                ark_mem.borrow_mut().user_data = user_data;
                arkStep_mem_mut(ark_mem).nfe += 1;
                if retval != 0 {
                    arkProcessError(
                        Some(ark_mem),
                        ARK_RHSFUNC_FAIL,
                        line!() as i32,
                        "arkStep_FullRHS",
                        file!(),
                        &MSG_ARK_RHSFUNC_FAILED(t),
                    );
                    return ARK_RHSFUNC_FAIL;
                }
            }

            /* combine RHS vector(s) into output */
            if explicit && implicit {
                /* ImEx */
                let sdata = arkStep_mem_mut(ark_mem).sdata.clone().expect("sdata");
                let tempv2 = ark_mem.borrow().tempv2.clone().expect("tempv2");
                N_VLinearSum(ONE, &sdata, ONE, &tempv2, f);
            } else if implicit {
                /* implicit */
                let sdata = arkStep_mem_mut(ark_mem).sdata.clone().expect("sdata");
                N_VScale(ONE, &sdata, f);
            } else {
                /* explicit */
                let tempv2 = ark_mem.borrow().tempv2.clone().expect("tempv2");
                N_VScale(ONE, &tempv2, f);
            }

            /* compute M^{-1} f for output but do not store */
            if mass_type != MASS_IDENTITY {
                let h = ark_mem.borrow().h;
                retval = (msolve.expect("msolve"))(ark_mem, f, nlscoef / h);
                if retval != 0 {
                    arkProcessError(
                        Some(ark_mem),
                        ARK_MASSSOLVE_FAIL,
                        line!() as i32,
                        "arkStep_FullRHS",
                        file!(),
                        "Mass matrix solver failure",
                    );
                    return ARK_MASSSOLVE_FAIL;
                }
            }

            /* apply external polynomial (MRI) forcing (M = I required) */
            arkStep_FullRHS_ApplyForcing(ark_mem, t, stage_coefs, f);
        }

        _ => {
            /* return with RHS failure if unknown mode is passed */
            arkProcessError(
                Some(ark_mem),
                ARK_RHSFUNC_FAIL,
                line!() as i32,
                "arkStep_FullRHS",
                file!(),
                "Unknown full RHS mode",
            );
            return ARK_RHSFUNC_FAIL;
        }
    }

    ARK_SUCCESS
}

/// The `apply external polynomial (MRI) forcing (M = I required)` block that
/// closes each of `arkStep_FullRHS`'s three modes verbatim:
///
/// ```text
/// cvals[0] = ONE; Xvecs[0] = f; nvec = 1;
/// arkStep_ApplyForcing(step_mem, &t, &stage_coefs, 1, &nvec);
/// N_VLinearCombination(nvec, cvals, Xvecs, f);
/// ```
fn arkStep_FullRHS_ApplyForcing(
    ark_mem: &ARKodeMem,
    t: sunrealtype,
    stage_coefs: sunrealtype,
    f: &N_Vector,
) {
    let (expforcing, impforcing) = {
        let s = arkStep_mem_mut(ark_mem);
        (s.expforcing, s.impforcing)
    };
    if expforcing || impforcing {
        let mut cvals: Vec<sunrealtype> = Vec::new();
        let mut Xvecs: Vec<N_Vector> = Vec::new();
        cvals.push(ONE);
        Xvecs.push(f.clone());
        let mut nvec: i32 = 1;
        {
            let step_mem = arkStep_mem_mut(ark_mem);
            arkStep_ApplyForcing(
                &step_mem,
                &mut cvals,
                &mut Xvecs,
                &[t],
                &[stage_coefs],
                1,
                &mut nvec,
            );
        }
        let _ = N_VLinearCombination(nvec, &cvals, &Xvecs, f);
    }
}

/*---------------------------------------------------------------
arkStep_TakeStep_Z:

This routine serves the primary purpose of the ARKStep module:
it performs a single ARK step (with embedding, if possible).
This version solves for each ARK stage vector, z_i.

The output variable dsmPtr should contain estimate of the
weighted local error if an embedding is present; otherwise it
should be 0.

The input/output variable nflagPtr is used to gauge convergence
of any algebraic solvers within the step.  At the start of a new
time step, this will initially have the value FIRST_CALL.  On
return from this function, nflagPtr should have a value:
          0 => algebraic solve completed successfully
         >0 => solve did not converge at this step size
               (but may with a smaller stepsize)
         <0 => solve encountered an unrecoverable failure

The return value from this routine is:
          0 => step completed successfully
         >0 => step encountered recoverable failure;
               reduce step and retry (if possible)
         <0 => step encountered unrecoverable failure
---------------------------------------------------------------*/
pub fn arkStep_TakeStep_Z(
    ark_mem: &ARKodeMem,
    dsmPtr: &mut sunrealtype,
    nflagPtr: &mut i32,
) -> i32 {
    let mut retval: i32;
    let mut implicit_stage: sunbooleantype;
    let save_stages: sunbooleantype;
    let mut stiffly_accurate: sunbooleantype;
    let save_fn_for_interp: sunbooleantype;
    let imex_method: sunbooleantype;
    let save_fn_for_residual: sunbooleantype;
    let eval_rhs: sunbooleantype;
    let is_start: i32;
    let mode: i32;

    /* access ARKodeARKStepMem structure */
    retval = arkStep_AccessStepMem(ark_mem, "arkStep_TakeStep_Z");
    if retval != ARK_SUCCESS {
        return retval;
    }

    let (implicit, explicit, mass_type, stages, deduce_rhs, nlscoef, predictor, autonomous) = {
        let s = arkStep_mem_mut(ark_mem);
        (
            s.implicit,
            s.explicit,
            s.mass_type,
            s.stages,
            s.deduce_rhs,
            s.nlscoef,
            s.predictor,
            s.autonomous,
        )
    };

    /* if problem will involve no algebraic solvers, initialize nflagPtr to success */
    if (!implicit) && (mass_type == MASS_IDENTITY) {
        *nflagPtr = ARK_SUCCESS;
    }

    /* initialize the current stage index */
    arkStep_mem_mut(ark_mem).istage = 0;

    /* call nonlinear solver setup if it exists */
    let NLS = arkStep_mem_mut(ark_mem).NLS.clone();
    if let Some(NLS) = &NLS {
        let has_setup = NLS.ops.borrow().setup.is_some();
        if has_setup {
            let zcor0 = ark_mem.borrow().tempv3.clone().expect("tempv3");
            /* set guess to all 0 (since using predictor-corrector form) */
            N_VConst(ZERO, &zcor0);
            let mut nls_mem: Option<Box<dyn Any>> = Some(Box::new(ark_mem.clone()));
            retval = SUNNonlinSolSetup(NLS, &zcor0, &mut nls_mem);

            if retval < 0 {
                return ARK_NLS_SETUP_FAIL;
            }
            if retval > 0 {
                return ARK_NLS_SETUP_RECVR;
            }
        }
    }

    /* check if we need to store stage values */
    let relax_enabled = ark_mem.borrow().relax_enabled;
    save_stages = relax_enabled && (implicit || mass_type == MASS_FIXED);

    /* check for an ImEx method */
    imex_method = implicit && explicit;

    /* check for implicit method with an explicit first stage */
    {
        let mut is_start_local = 1;
        implicit_stage = SUNFALSE;
        if implicit {
            let A00 = {
                let s = arkStep_mem_mut(ark_mem);
                let Bi = s.Bi.as_ref().expect("Bi").borrow();
                Bi.A[0][0]
            };
            if SUNRabs(A00) > TINY {
                implicit_stage = SUNTRUE;
                is_start_local = 0;
            }
        }
        is_start = is_start_local;
    }

    /* explicit first stage -- store stage if necessary for relaxation or checkpointing */
    if is_start == 1 {
        if save_stages {
            let (yn, z0) = {
                let yn = ark_mem.borrow().yn.clone().expect("yn");
                let z0 = arkStep_mem_mut(ark_mem).z[0].clone();
                (yn, z0)
            };
            N_VScale(ONE, &yn, &z0);
        }

        let checkpoint_scheme = ark_mem.borrow().checkpoint_scheme.clone();
        if let Some(checkpoint_scheme) = &checkpoint_scheme {
            let mut do_save: sunbooleantype = SUNFALSE;
            let (checkpoint_step_idx, tn, yn) = {
                let m = ark_mem.borrow();
                (m.checkpoint_step_idx, m.tn, m.yn.clone().expect("yn"))
            };
            let errcode = SUNAdjointCheckpointScheme_NeedsSaving(
                checkpoint_scheme,
                checkpoint_step_idx,
                0,
                tn,
                &mut do_save,
            );
            if errcode != SUN_SUCCESS {
                arkProcessError(
                    Some(ark_mem),
                    ARK_ADJ_CHECKPOINT_FAIL,
                    line!() as i32,
                    "arkStep_TakeStep_Z",
                    file!(),
                    &format!("SUNAdjointCheckpointScheme_NeedsSaving returned {errcode}"),
                );
                return ARK_ADJ_CHECKPOINT_FAIL;
            }

            if do_save {
                let errcode = SUNAdjointCheckpointScheme_InsertVector(
                    checkpoint_scheme,
                    checkpoint_step_idx,
                    0,
                    tn,
                    &yn,
                );

                if errcode != SUN_SUCCESS {
                    arkProcessError(
                        Some(ark_mem),
                        ARK_ADJ_CHECKPOINT_FAIL,
                        line!() as i32,
                        "arkStep_TakeStep_Z",
                        file!(),
                        &format!("SUNAdjointCheckpointScheme_InsertVector returned {errcode}"),
                    );
                    return ARK_ADJ_CHECKPOINT_FAIL;
                }
            }
        }
    }

    /* check if the method is Stiffly Accurate (SA) */
    stiffly_accurate = SUNTRUE;
    if explicit {
        let Be = arkStep_mem_mut(ark_mem).Be.clone().expect("Be");
        if !ARKodeButcherTable_IsStifflyAccurate(Some(&Be)) {
            stiffly_accurate = SUNFALSE;
        }
    }

    if implicit {
        let Bi = arkStep_mem_mut(ark_mem).Bi.clone().expect("Bi");
        if !ARKodeButcherTable_IsStifflyAccurate(Some(&Bi)) {
            stiffly_accurate = SUNFALSE;
        }
    }

    /* Save f(tn, yn) for Hermite interpolation */
    save_fn_for_interp = ark_mem.borrow().interp_type == ARK_INTERP_HERMITE;

    /* For an implicit or ImEx method using the trivial predictor with an
    autonomous problem with an identity or fixed mass matrix, save fi(tn, yn)
    for reuse in the first residual evaluation of each stage solve */
    save_fn_for_residual = implicit && predictor == 0 && autonomous && mass_type != MASS_TIMEDEP;

    /* Call the RHS if needed. */
    eval_rhs = !implicit_stage || save_fn_for_interp || save_fn_for_residual;

    if !ark_mem.borrow().fn_is_current && eval_rhs {
        /* If saving the RHS evaluation for reuse in the residual, call the full RHS
        for all implicit methods or for ImEx methods with an explicit first
        stage. ImEx methods with an implicit first stage may not need to evaluate
        fe depending on the interpolation type (covered by save_fn_for_interp) */
        let res_full_rhs: sunbooleantype = save_fn_for_residual && implicit_stage && !imex_method;

        if !implicit_stage || save_fn_for_interp || res_full_rhs {
            /* Need full RHS evaluation. If this is the first step, then we evaluate
            or copy the RHS values from an earlier evaluation (e.g., to compute
            h0). For subsequent steps treat this call as an evaluation at the end
            of the just completed step (tn, yn) and potentially reuse the
            evaluation (FSAL method) or save the value for later use. */
            let (initsetup, tn, yn, fn_, step_fullrhs) = {
                let m = ark_mem.borrow();
                (
                    m.initsetup,
                    m.tn,
                    m.yn.clone().expect("yn"),
                    m.fn_.clone().expect("fn"),
                    m.step_fullrhs.expect("step_fullrhs"),
                )
            };
            mode = if initsetup {
                ARK_FULLRHS_START
            } else {
                ARK_FULLRHS_END
            };
            retval = step_fullrhs(ark_mem, tn, &yn, &fn_, mode);
            if retval != 0 {
                return ARK_RHSFUNC_FAIL;
            }
            ark_mem.borrow_mut().fn_is_current = SUNTRUE;
        } else {
            /* For an ImEx method with implicit first stage and an interpolation
            method that does not need fn (e.g., Lagrange), only evaluate fi (if
            necessary) for reuse in the residual */
            if stiffly_accurate {
                let (Fi_last, Fi0) = {
                    let s = arkStep_mem_mut(ark_mem);
                    (s.Fi[(stages - 1) as usize].clone(), s.Fi[0].clone())
                };
                N_VScale(ONE, &Fi_last, &Fi0);
            } else {
                /* call the user-supplied pre-RHS function (if supplied) */
                let (PreRhsFn, tn, yn) = {
                    let m = ark_mem.borrow();
                    (m.PreRhsFn, m.tn, m.yn.clone().expect("yn"))
                };
                if let Some(PreRhsFn) = PreRhsFn {
                    let mut user_data = ark_mem.borrow_mut().user_data.take();
                    retval = PreRhsFn(tn, &yn, &mut user_data);
                    ark_mem.borrow_mut().user_data = user_data;
                    if retval != 0 {
                        return ARK_PRERHSFN_FAIL;
                    }
                }
                let (fi, Fi0) = {
                    let s = arkStep_mem_mut(ark_mem);
                    (s.fi.expect("fi"), s.Fi[0].clone())
                };
                let mut user_data = ark_mem.borrow_mut().user_data.take();
                retval = fi(tn, &yn, &Fi0, &mut user_data);
                ark_mem.borrow_mut().user_data = user_data;
                arkStep_mem_mut(ark_mem).nfi += 1;

                if retval < 0 {
                    return ARK_RHSFUNC_FAIL;
                }
                if retval > 0 {
                    return ARK_UNREC_RHSFUNC_ERR;
                }
            }
        }
    }

    /* Set alias to implicit RHS evaluation for reuse in residual */
    arkStep_mem_mut(ark_mem).fn_implicit = None;
    if save_fn_for_residual {
        if !implicit_stage {
            /* Explicit first stage -- Fi[0] will be retained */
            let Fi0 = arkStep_mem_mut(ark_mem).Fi[0].clone();
            arkStep_mem_mut(ark_mem).fn_implicit = Some(Fi0);
        } else {
            /* Implicit first stage -- Fi[0] will be overwritten */
            if imex_method || mass_type == MASS_FIXED {
                /* Copy from Fi[0] as fn includes fe or M^{-1} */
                let Fi0 = arkStep_mem_mut(ark_mem).Fi[0].clone();
                let tempv5 = ark_mem.borrow().tempv5.clone().expect("tempv5");
                N_VScale(ONE, &Fi0, &tempv5);
                arkStep_mem_mut(ark_mem).fn_implicit = Some(tempv5);
            } else {
                /* fn is the same as Fi[0] but will not be overwritten */
                let fn_ = ark_mem.borrow().fn_.clone().expect("fn");
                arkStep_mem_mut(ark_mem).fn_implicit = Some(fn_);
            }
        }
    }

    /* loop over internal stages to the step */
    for is in is_start..stages {
        /* store current stage index */
        arkStep_mem_mut(ark_mem).istage = is;

        /* determine whether implicit solve is required */
        implicit_stage = SUNFALSE;
        if implicit {
            let Aii = {
                let s = arkStep_mem_mut(ark_mem);
                let Bi = s.Bi.as_ref().expect("Bi").borrow();
                Bi.A[is as usize][is as usize]
            };
            if SUNRabs(Aii) > TINY {
                implicit_stage = SUNTRUE;
            }
        }

        /* determine if the stage RHS will be deduced from the implicit solve */
        let deduce_stage: sunbooleantype = deduce_rhs && implicit_stage;

        /* set current stage time(s) */
        {
            let ci = {
                let s = arkStep_mem_mut(ark_mem);
                if implicit {
                    s.Bi.as_ref().expect("Bi").borrow().c[is as usize]
                } else {
                    s.Be.as_ref().expect("Be").borrow().c[is as usize]
                }
            };
            let mut m = ark_mem.borrow_mut();
            let tn = m.tn;
            let h = m.h;
            m.tcur = tn + ci * h;
        }

        /* setup time-dependent mass matrix */
        let msetup = arkStep_mem_mut(ark_mem).msetup;
        if (mass_type == MASS_TIMEDEP) && msetup.is_some() {
            let (tcur, tempv1, tempv2, tempv3) = {
                let m = ark_mem.borrow();
                (
                    m.tcur,
                    m.tempv1.clone().expect("tempv1"),
                    m.tempv2.clone().expect("tempv2"),
                    m.tempv3.clone().expect("tempv3"),
                )
            };
            retval = (msetup.expect("msetup"))(ark_mem, tcur, &tempv1, &tempv2, &tempv3);
            if retval != ARK_SUCCESS {
                return ARK_MASSSETUP_FAIL;
            }
        }

        /* if implicit, call built-in and user-supplied predictors
        (results placed in zpred) */
        if implicit_stage {
            let zpred = arkStep_mem_mut(ark_mem).zpred.clone().expect("zpred");
            retval = arkStep_Predict(ark_mem, is, &zpred);
            if retval != ARK_SUCCESS {
                return retval;
            }

            /* if a user-supplied predictor routine is provided, call that here.
            Note that arkStep_Predict is *still* called, so this user-supplied
            routine can just 'clean up' the built-in prediction, if desired. */
            let stage_predict = arkStep_mem_mut(ark_mem).stage_predict;
            if let Some(stage_predict) = stage_predict {
                let tcur = ark_mem.borrow().tcur;
                let mut user_data = ark_mem.borrow_mut().user_data.take();
                retval = stage_predict(tcur, &zpred, &mut user_data);
                ark_mem.borrow_mut().user_data = user_data;

                if retval < 0 {
                    return ARK_USER_PREDICT_FAIL;
                }
                if retval > 0 {
                    return TRY_AGAIN;
                }
            }
        }

        /* set up explicit data for evaluation of ARK stage (store in sdata) */
        retval = arkStep_StageSetup(ark_mem, implicit_stage);
        if retval != ARK_SUCCESS {
            return retval;
        }

        /* perform implicit solve if required */
        if implicit_stage {
            /* implicit solve result is stored in ark_mem->ycur;
            return with positive value on anything but success */
            *nflagPtr = arkStep_Nls(ark_mem, *nflagPtr);
            if *nflagPtr != ARK_SUCCESS {
                return TRY_AGAIN;
            }

        /* otherwise no implicit solve is needed */
        } else {
            /* if M is fixed, solve with it to compute update (place back in sdata) */
            if mass_type == MASS_FIXED {
                /* perform solve; return with positive value on anything but success */
                let (msolve, sdata) = {
                    let s = arkStep_mem_mut(ark_mem);
                    (s.msolve.expect("msolve"), s.sdata.clone().expect("sdata"))
                };
                *nflagPtr = msolve(ark_mem, &sdata, nlscoef);
                if *nflagPtr != ARK_SUCCESS {
                    return TRY_AGAIN;
                }
            }

            /* set y to be yn + sdata (either computed in arkStep_StageSetup,
            or updated in prev. block) */
            let (yn, ycur) = {
                let m = ark_mem.borrow();
                (m.yn.clone().expect("yn"), m.ycur.clone().expect("ycur"))
            };
            let sdata = arkStep_mem_mut(ark_mem).sdata.clone().expect("sdata");
            N_VLinearSum(ONE, &yn, ONE, &sdata, &ycur);
        }

        /* apply user-supplied stage postprocessing function (if supplied) unless
        this is the last stage of a FSAL method, then apply the user-supplied
        step postprocessing function instead (if supplied) */
        /* NOTE: with internally inconsistent IMEX methods (c_i^E != c_i^I) the value
        of tcur corresponds to the stage time from the implicit table (c_i^I). */
        let (PostProcessStepFn, PostProcessStageFn, tcur, ycur) = {
            let m = ark_mem.borrow();
            (
                m.PostProcessStepFn,
                m.PostProcessStageFn,
                m.tcur,
                m.ycur.clone().expect("ycur"),
            )
        };
        if is == stages - 1 && stiffly_accurate && PostProcessStepFn.is_some() {
            let mut user_data = ark_mem.borrow_mut().user_data.take();
            retval = (PostProcessStepFn.expect("PostProcessStepFn"))(tcur, &ycur, &mut user_data);
            ark_mem.borrow_mut().user_data = user_data;
            if retval != 0 {
                return ARK_POSTPROCESS_STEP_FAIL;
            }
        } else if let Some(PostProcessStageFn) = PostProcessStageFn {
            let mut user_data = ark_mem.borrow_mut().user_data.take();
            retval = PostProcessStageFn(tcur, &ycur, &mut user_data);
            ark_mem.borrow_mut().user_data = user_data;
            if retval != 0 {
                return ARK_POSTPROCESS_STAGE_FAIL;
            }
        }

        /* successful stage solve */

        /*    store stage (if necessary for relaxation) */
        if save_stages {
            let z_is = arkStep_mem_mut(ark_mem).z[is as usize].clone();
            N_VScale(ONE, &ycur, &z_is);
        }

        /*    checkpoint stage for adjoint (if necessary) */
        let checkpoint_scheme = ark_mem.borrow().checkpoint_scheme.clone();
        if let Some(checkpoint_scheme) = &checkpoint_scheme {
            let mut do_save: sunbooleantype = SUNFALSE;
            let checkpoint_step_idx = ark_mem.borrow().checkpoint_step_idx;
            let errcode = SUNAdjointCheckpointScheme_NeedsSaving(
                checkpoint_scheme,
                checkpoint_step_idx,
                is as suncountertype,
                tcur,
                &mut do_save,
            );
            if errcode != SUN_SUCCESS {
                arkProcessError(
                    Some(ark_mem),
                    ARK_ADJ_CHECKPOINT_FAIL,
                    line!() as i32,
                    "arkStep_TakeStep_Z",
                    file!(),
                    &format!("SUNAdjointCheckpointScheme_NeedsSaving returned {errcode}"),
                );
                return ARK_ADJ_CHECKPOINT_FAIL;
            }

            if do_save {
                let errcode = SUNAdjointCheckpointScheme_InsertVector(
                    checkpoint_scheme,
                    checkpoint_step_idx,
                    is as suncountertype,
                    tcur,
                    &ycur,
                );

                if errcode != SUN_SUCCESS {
                    arkProcessError(
                        Some(ark_mem),
                        ARK_ADJ_CHECKPOINT_FAIL,
                        line!() as i32,
                        "arkStep_TakeStep_Z",
                        file!(),
                        &format!("SUNAdjointCheckpointScheme_InsertVector returned {errcode}"),
                    );
                    return ARK_ADJ_CHECKPOINT_FAIL;
                }
            }
        }

        /* call the user-supplied pre-RHS function (if supplied) */
        /* NOTE: with internally inconsistent IMEX methods (c_i^E != c_i^I) the value
        of tcur corresponds to the stage time from the implicit table (c_i^I). */
        let PreRhsFn = ark_mem.borrow().PreRhsFn;
        if let Some(PreRhsFn) = PreRhsFn {
            if (implicit && !deduce_stage) || explicit {
                let mut user_data = ark_mem.borrow_mut().user_data.take();
                retval = PreRhsFn(tcur, &ycur, &mut user_data);
                ark_mem.borrow_mut().user_data = user_data;
                if retval != 0 {
                    return ARK_PRERHSFN_FAIL;
                }
            }
        }

        /*    store implicit RHS (value in Fi[is] is from preceding nonlinear iteration) */
        if implicit {
            if !deduce_stage {
                let (fi, Fi_is) = {
                    let s = arkStep_mem_mut(ark_mem);
                    (s.fi.expect("fi"), s.Fi[is as usize].clone())
                };
                let mut user_data = ark_mem.borrow_mut().user_data.take();
                retval = fi(tcur, &ycur, &Fi_is, &mut user_data);
                ark_mem.borrow_mut().user_data = user_data;
                arkStep_mem_mut(ark_mem).nfi += 1;

                if retval < 0 {
                    return ARK_RHSFUNC_FAIL;
                }
                if retval > 0 {
                    return ARK_UNREC_RHSFUNC_ERR;
                }
            } else {
                let (gamma, zcor, sdata, Fi_is) = {
                    let s = arkStep_mem_mut(ark_mem);
                    (
                        s.gamma,
                        s.zcor.clone().expect("zcor"),
                        s.sdata.clone().expect("sdata"),
                        s.Fi[is as usize].clone(),
                    )
                };
                if mass_type == MASS_FIXED {
                    let (mmult, tempv1) = {
                        let mmult = arkStep_mem_mut(ark_mem).mmult.expect("mmult");
                        let tempv1 = ark_mem.borrow().tempv1.clone().expect("tempv1");
                        (mmult, tempv1)
                    };
                    retval = mmult(ark_mem, &zcor, &tempv1);
                    if retval != ARK_SUCCESS {
                        return ARK_MASSMULT_FAIL;
                    }

                    N_VLinearSum(ONE / gamma, &tempv1, -ONE / gamma, &sdata, &Fi_is);
                } else {
                    N_VLinearSum(ONE / gamma, &zcor, -ONE / gamma, &sdata, &Fi_is);
                }
            }
        }

        /*    store explicit RHS */
        if explicit {
            let (fe, Fe_is, ci) = {
                let s = arkStep_mem_mut(ark_mem);
                let ci = s.Be.as_ref().expect("Be").borrow().c[is as usize];
                (s.fe.expect("fe"), s.Fe[is as usize].clone(), ci)
            };
            let (tn, h) = {
                let m = ark_mem.borrow();
                (m.tn, m.h)
            };
            let mut user_data = ark_mem.borrow_mut().user_data.take();
            retval = fe(tn + ci * h, &ycur, &Fe_is, &mut user_data);
            ark_mem.borrow_mut().user_data = user_data;
            arkStep_mem_mut(ark_mem).nfe += 1;

            if retval < 0 {
                return ARK_RHSFUNC_FAIL;
            }
            if retval > 0 {
                return ARK_UNREC_RHSFUNC_ERR;
            }
        }

        /* if using a time-dependent mass matrix, update Fe[is] and/or Fi[is] with M(t)^{-1} */
        if mass_type == MASS_TIMEDEP {
            /* If the implicit stage was deduced, it already includes M(t)^{-1} */
            if implicit && !deduce_stage {
                let (msolve, Fi_is) = {
                    let s = arkStep_mem_mut(ark_mem);
                    (s.msolve.expect("msolve"), s.Fi[is as usize].clone())
                };
                *nflagPtr = msolve(ark_mem, &Fi_is, nlscoef);

                if *nflagPtr != ARK_SUCCESS {
                    return TRY_AGAIN;
                }
            }
            if explicit {
                let (msolve, Fe_is) = {
                    let s = arkStep_mem_mut(ark_mem);
                    (s.msolve.expect("msolve"), s.Fe[is as usize].clone())
                };
                *nflagPtr = msolve(ark_mem, &Fe_is, nlscoef);
                if *nflagPtr != ARK_SUCCESS {
                    return TRY_AGAIN;
                }
            }
        }
    } /* loop over stages */

    /* compute time-evolved solution (in ark_ycur), error estimate (in dsm).
    This can fail recoverably due to nonconvergence of the mass matrix solve,
    so handle that appropriately. */
    {
        let mut m = ark_mem.borrow_mut();
        let tn = m.tn;
        let h = m.h;
        m.tcur = tn + h;
    }

    if mass_type == MASS_FIXED {
        *nflagPtr = arkStep_ComputeSolutions_MassFixed(ark_mem, dsmPtr);
    } else {
        *nflagPtr = arkStep_ComputeSolutions(ark_mem, dsmPtr);
    }

    if *nflagPtr < 0 {
        return *nflagPtr;
    }
    if *nflagPtr > 0 {
        return TRY_AGAIN;
    }

    let checkpoint_scheme = ark_mem.borrow().checkpoint_scheme.clone();
    if let Some(checkpoint_scheme) = &checkpoint_scheme {
        let mut do_save: sunbooleantype = SUNFALSE;
        let (checkpoint_step_idx, tn, h, ycur) = {
            let m = ark_mem.borrow();
            (
                m.checkpoint_step_idx,
                m.tn,
                m.h,
                m.ycur.clone().expect("ycur"),
            )
        };
        let Be_stages = {
            let s = arkStep_mem_mut(ark_mem);
            let stages = s.Be.as_ref().expect("Be").borrow().stages;
            stages
        };
        let errcode = SUNAdjointCheckpointScheme_NeedsSaving(
            checkpoint_scheme,
            checkpoint_step_idx,
            Be_stages as suncountertype,
            tn + h,
            &mut do_save,
        );
        if errcode != SUN_SUCCESS {
            arkProcessError(
                Some(ark_mem),
                ARK_ADJ_CHECKPOINT_FAIL,
                line!() as i32,
                "arkStep_TakeStep_Z",
                file!(),
                &format!("SUNAdjointCheckpointScheme_NeedsSaving returned {errcode}"),
            );
            return ARK_ADJ_CHECKPOINT_FAIL;
        }
        if do_save {
            let errcode = SUNAdjointCheckpointScheme_InsertVector(
                checkpoint_scheme,
                checkpoint_step_idx,
                Be_stages as suncountertype,
                tn + h,
                &ycur,
            );
            if errcode != SUN_SUCCESS {
                arkProcessError(
                    Some(ark_mem),
                    ARK_ADJ_CHECKPOINT_FAIL,
                    line!() as i32,
                    "arkStep_TakeStep_Z",
                    file!(),
                    &format!("SUNAdjointCheckpointScheme_NeedsSaving returned {errcode}"),
                );
                return ARK_ADJ_CHECKPOINT_FAIL;
            }
        }
    }

    ARK_SUCCESS
}

/*===============================================================
Internal utility routines
===============================================================*/

/*---------------------------------------------------------------
arkStep_AccessARKODEStepMem:

Shortcut routine to check that the ark_mem and step_mem
structures are present.  If either is missing it returns
ARK_MEM_NULL.  (C also unpacks the two pointers; here the caller
reaches the step memory with `arkStep_mem_mut`.)
---------------------------------------------------------------*/
pub fn arkStep_AccessARKODEStepMem(arkode_mem: &ARKodeMem, fname: &str) -> i32 {
    /* access ARKodeMem structure: NULL arkode_mem handled by the type system */
    let ark_mem = arkode_mem;

    /* access ARKodeARKStepMem structure */
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            fname,
            file!(),
            MSG_ARKSTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }
    ARK_SUCCESS
}

/*---------------------------------------------------------------
arkStep_AccessStepMem:

Shortcut routine to check that the step_mem structure is
present.  If it is missing it returns ARK_MEM_NULL.
---------------------------------------------------------------*/
pub fn arkStep_AccessStepMem(ark_mem: &ARKodeMem, fname: &str) -> i32 {
    /* access ARKodeARKStepMem structure */
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            fname,
            file!(),
            MSG_ARKSTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }
    ARK_SUCCESS
}

/*---------------------------------------------------------------
arkStep_SetButcherTables

This routine determines the ERK/DIRK/ARK method to use, based
on the desired accuracy and information on whether the problem
is explicit, implicit or imex.
---------------------------------------------------------------*/
pub fn arkStep_SetButcherTables(ark_mem: &ARKodeMem) -> i32 {
    let mut etable: i32;
    let mut itable: i32;

    /* access ARKodeARKStepMem structure */
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            None,
            ARK_MEM_NULL,
            line!() as i32,
            "arkStep_SetButcherTables",
            file!(),
            MSG_ARKSTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    /* if tables have already been specified, just return */
    let (has_Be, has_Bi, explicit, implicit, q) = {
        let s = arkStep_mem_mut(ark_mem);
        (s.Be.is_some(), s.Bi.is_some(), s.explicit, s.implicit, s.q)
    };
    if has_Be || has_Bi {
        return ARK_SUCCESS;
    }

    /* initialize table numbers to illegal values */
    etable = -1;
    itable = -1;

    /**** ImEx methods ****/
    if explicit && implicit {
        match q {
            2 => {
                etable = ARKSTEP_DEFAULT_ARK_ETABLE_2;
                itable = ARKSTEP_DEFAULT_ARK_ITABLE_2;
            }
            3 => {
                etable = ARKSTEP_DEFAULT_ARK_ETABLE_3;
                itable = ARKSTEP_DEFAULT_ARK_ITABLE_3;
            }
            4 => {
                etable = ARKSTEP_DEFAULT_ARK_ETABLE_4;
                itable = ARKSTEP_DEFAULT_ARK_ITABLE_4;
            }
            5 => {
                etable = ARKSTEP_DEFAULT_ARK_ETABLE_5;
                itable = ARKSTEP_DEFAULT_ARK_ITABLE_5;
            }
            _ => {
                /* no available method, set default */
                arkProcessError(
                    Some(ark_mem),
                    ARK_WARNING,
                    line!() as i32,
                    "arkStep_SetButcherTables",
                    file!(),
                    "No ImEx method at requested order, using q=5.",
                );
                etable = ARKSTEP_DEFAULT_ARK_ETABLE_5;
                itable = ARKSTEP_DEFAULT_ARK_ITABLE_5;
            }
        }

    /**** implicit methods ****/
    } else if implicit {
        match q {
            1 => itable = ARKSTEP_DEFAULT_DIRK_1,
            2 => itable = ARKSTEP_DEFAULT_DIRK_2,
            3 => itable = ARKSTEP_DEFAULT_DIRK_3,
            4 => itable = ARKSTEP_DEFAULT_DIRK_4,
            5 => itable = ARKSTEP_DEFAULT_DIRK_5,
            _ => {
                /* no available method, set default */
                arkProcessError(
                    Some(ark_mem),
                    ARK_WARNING,
                    line!() as i32,
                    "arkStep_SetButcherTables",
                    file!(),
                    "No implicit method at requested order, using q=5.",
                );
                itable = ARKSTEP_DEFAULT_DIRK_5;
            }
        }

    /**** explicit methods ****/
    } else {
        match q {
            1 => etable = ARKSTEP_DEFAULT_ERK_1,
            2 => etable = ARKSTEP_DEFAULT_ERK_2,
            3 => etable = ARKSTEP_DEFAULT_ERK_3,
            4 => etable = ARKSTEP_DEFAULT_ERK_4,
            5 => etable = ARKSTEP_DEFAULT_ERK_5,
            6 => etable = ARKSTEP_DEFAULT_ERK_6,
            7 => etable = ARKSTEP_DEFAULT_ERK_7,
            8 => etable = ARKSTEP_DEFAULT_ERK_8,
            9 => etable = ARKSTEP_DEFAULT_ERK_9,
            _ => {
                /* no available method, set default */
                arkProcessError(
                    Some(ark_mem),
                    ARK_WARNING,
                    line!() as i32,
                    "arkStep_SetButcherTables",
                    file!(),
                    "No explicit method at requested order, using q=9.",
                );
                etable = ARKSTEP_DEFAULT_ERK_9;
            }
        }
    }

    if etable > -1 {
        let Be = ARKodeButcherTable_LoadERK(etable);
        arkStep_mem_mut(ark_mem).Be = Be;
    }
    if itable > -1 {
        let Bi = ARKodeButcherTable_LoadDIRK(itable);
        arkStep_mem_mut(ark_mem).Bi = Bi;
    }

    /* note Butcher table space requirements */
    let Be = arkStep_mem_mut(ark_mem).Be.clone();
    let mut Bliw: sunindextype = 0;
    let mut Blrw: sunindextype = 0;
    if let Some(Be) = &Be {
        ARKodeButcherTable_Space(Some(Be), &mut Bliw, &mut Blrw);
    }
    {
        let mut m = ark_mem.borrow_mut();
        m.liw += Bliw;
        m.lrw += Blrw;
    }

    let Bi = arkStep_mem_mut(ark_mem).Bi.clone();
    let mut Bliw: sunindextype = 0;
    let mut Blrw: sunindextype = 0;
    if let Some(Bi) = &Bi {
        ARKodeButcherTable_Space(Some(Bi), &mut Bliw, &mut Blrw);
    }
    {
        let mut m = ark_mem.borrow_mut();
        m.liw += Bliw;
        m.lrw += Blrw;
    }

    /* set [redundant] ARK stored values for stage numbers and method orders */
    if let Some(Be) = &Be {
        let (stages, q, p) = {
            let B = Be.borrow();
            (B.stages, B.q, B.p)
        };
        let mut s = arkStep_mem_mut(ark_mem);
        s.stages = stages;
        s.q = q;
        s.p = p;
    }
    if let Some(Bi) = &Bi {
        let (stages, q, p) = {
            let B = Bi.borrow();
            (B.stages, B.q, B.p)
        };
        let mut s = arkStep_mem_mut(ark_mem);
        s.stages = stages;
        s.q = q;
        s.p = p;
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
arkStep_CheckButcherTables

This routine runs through the explicit and/or implicit Butcher
tables to ensure that they meet all necessary requirements,
including:
  strictly lower-triangular (ERK)
  lower-triangular with some nonzeros on diagonal (IRK)
  method order q > 0 (all)
  embedding order q > 0 (all -- if adaptive time-stepping enabled)
  stages > 0 (all)

Returns ARK_SUCCESS if tables pass, ARK_INVALID_TABLE otherwise.
---------------------------------------------------------------*/
pub fn arkStep_CheckButcherTables(ark_mem: &ARKodeMem) -> i32 {
    let mut okay: sunbooleantype;
    let tol: sunrealtype = 100.0 * SUN_UNIT_ROUNDOFF;

    /* access ARKodeARKStepMem structure */
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            None,
            ARK_MEM_NULL,
            line!() as i32,
            "arkStep_CheckButcherTables",
            file!(),
            MSG_ARKSTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    let (explicit, implicit, Be, Bi, stages, q, p) = {
        let s = arkStep_mem_mut(ark_mem);
        (
            s.explicit,
            s.implicit,
            s.Be.clone(),
            s.Bi.clone(),
            s.stages,
            s.q,
            s.p,
        )
    };
    let (fixedstep, relax_enabled) = {
        let m = ark_mem.borrow();
        (m.fixedstep, m.relax_enabled)
    };

    /* check that the expected tables are set */
    if explicit && Be.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_INVALID_TABLE,
            line!() as i32,
            "arkStep_CheckButcherTables",
            file!(),
            "explicit table is NULL!",
        );
        return ARK_INVALID_TABLE;
    }

    if implicit && Bi.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_INVALID_TABLE,
            line!() as i32,
            "arkStep_CheckButcherTables",
            file!(),
            "implicit table is NULL!",
        );
        return ARK_INVALID_TABLE;
    }

    /* check that stages > 0 */
    if stages < 1 {
        arkProcessError(
            Some(ark_mem),
            ARK_INVALID_TABLE,
            line!() as i32,
            "arkStep_CheckButcherTables",
            file!(),
            "stages < 1!",
        );
        return ARK_INVALID_TABLE;
    }

    /* check that method order q > 0 */
    if q < 1 {
        arkProcessError(
            Some(ark_mem),
            ARK_INVALID_TABLE,
            line!() as i32,
            "arkStep_CheckButcherTables",
            file!(),
            "method order < 1!",
        );
        return ARK_INVALID_TABLE;
    }

    /* check that embedding order p > 0 */
    if (p < 1) && (!fixedstep) {
        arkProcessError(
            Some(ark_mem),
            ARK_INVALID_TABLE,
            line!() as i32,
            "arkStep_CheckButcherTables",
            file!(),
            "embedding order < 1, but ARKodeSetFixedStep was not called!",
        );
        return ARK_INVALID_TABLE;
    }

    /* check that embedding exists */
    if (p > 0) && (!fixedstep) {
        if implicit {
            if Bi.as_ref().expect("Bi").borrow().d.is_empty() {
                arkProcessError(
                    Some(ark_mem),
                    ARK_INVALID_TABLE,
                    line!() as i32,
                    "arkStep_CheckButcherTables",
                    file!(),
                    "no implicit embedding, but ARKodeSetFixedStep was not called!",
                );
                return ARK_INVALID_TABLE;
            }
        }
        if explicit {
            if Be.as_ref().expect("Be").borrow().d.is_empty() {
                arkProcessError(
                    Some(ark_mem),
                    ARK_INVALID_TABLE,
                    line!() as i32,
                    "arkStep_CheckButcherTables",
                    file!(),
                    "no explicit embedding, but ARKodeSetFixedStep was not called!",
                );
                return ARK_INVALID_TABLE;
            }
        }
    }

    /* check that ERK table is strictly lower triangular */
    if explicit {
        okay = SUNTRUE;
        {
            let B = Be.as_ref().expect("Be").borrow();
            for i in 0..stages as usize {
                for j in i..stages as usize {
                    if SUNRabs(B.A[i][j]) > tol {
                        okay = SUNFALSE;
                    }
                }
            }
        }
        if !okay {
            arkProcessError(
                Some(ark_mem),
                ARK_INVALID_TABLE,
                line!() as i32,
                "arkStep_CheckButcherTables",
                file!(),
                "Ae Butcher table is implicit!",
            );
            return ARK_INVALID_TABLE;
        }
    }

    /* check that IRK table is implicit and lower triangular */
    if implicit {
        okay = SUNFALSE;
        {
            let B = Bi.as_ref().expect("Bi").borrow();
            for i in 0..stages as usize {
                if SUNRabs(B.A[i][i]) > tol {
                    okay = SUNTRUE;
                }
            }
        }
        if !okay {
            arkProcessError(
                Some(ark_mem),
                ARK_INVALID_TABLE,
                line!() as i32,
                "arkStep_CheckButcherTables",
                file!(),
                "Ai Butcher table is explicit!",
            );
            return ARK_INVALID_TABLE;
        }

        okay = SUNTRUE;
        {
            let B = Bi.as_ref().expect("Bi").borrow();
            for i in 0..stages as usize {
                for j in (i + 1)..stages as usize {
                    if SUNRabs(B.A[i][j]) > tol {
                        okay = SUNFALSE;
                    }
                }
            }
        }
        if !okay {
            arkProcessError(
                Some(ark_mem),
                ARK_INVALID_TABLE,
                line!() as i32,
                "arkStep_CheckButcherTables",
                file!(),
                "Ai Butcher table has entries above diagonal!",
            );
            return ARK_INVALID_TABLE;
        }
    }

    /* Check if the method is compatible with relaxation */
    if relax_enabled {
        if q < 2 {
            arkProcessError(
                Some(ark_mem),
                ARK_INVALID_TABLE,
                line!() as i32,
                "arkStep_CheckButcherTables",
                file!(),
                "The Butcher table(s) must be at least second order when using relaxation!",
            );
            return ARK_INVALID_TABLE;
        }

        if explicit {
            /* Check if all b values are positive */
            for i in 0..stages as usize {
                let bi = Be.as_ref().expect("Be").borrow().b[i];
                if bi < ZERO {
                    arkProcessError(
                        Some(ark_mem),
                        ARK_INVALID_TABLE,
                        line!() as i32,
                        "arkStep_CheckButcherTables",
                        file!(),
                        "The explicit Butcher table has a negative b value but relaxation enabled!",
                    );
                    return ARK_INVALID_TABLE;
                }
            }
        }

        if implicit {
            /* Check if all b values are positive */
            for i in 0..stages as usize {
                let bi = Bi.as_ref().expect("Bi").borrow().b[i];
                if bi < ZERO {
                    arkProcessError(
                        Some(ark_mem),
                        ARK_INVALID_TABLE,
                        line!() as i32,
                        "arkStep_CheckButcherTables",
                        file!(),
                        "The implicit Butcher table has a negative b value but relaxation enabled!",
                    );
                    return ARK_INVALID_TABLE;
                }
            }
        }
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
arkStep_Predict

This routine computes the prediction for a specific internal
stage solution, storing the result in yguess.  The
prediction is done using the interpolation structure in
extrapolation mode, hence stages "far" from the previous time
interval are predicted using lower order polynomials than the
"nearby" stages.
---------------------------------------------------------------*/
pub fn arkStep_Predict(ark_mem: &ARKodeMem, istage: i32, yguess: &N_Vector) -> i32 {
    let retval: i32;
    let mut tau: sunrealtype;

    /* access ARKodeARKStepMem structure */
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            None,
            ARK_MEM_NULL,
            line!() as i32,
            "arkStep_Predict",
            file!(),
            MSG_ARKSTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    let (predictor, explicit, implicit) = {
        let s = arkStep_mem_mut(ark_mem);
        (s.predictor, s.explicit, s.implicit)
    };

    /* verify that interpolation structure is provided */
    if ark_mem.borrow().interp.is_none() && (predictor > 0) && (predictor < 4) {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "arkStep_Predict",
            file!(),
            "Interpolation structure is NULL",
        );
        return ARK_MEM_NULL;
    }

    /* local shortcuts for use with fused vector operations: the operand
    lists are built locally (see the module docs) */
    let mut cvals: Vec<sunrealtype> = Vec::new();
    let mut Xvecs: Vec<N_Vector> = Vec::new();

    /* if the first step, use initial condition as guess */
    let (initsetup, ark_h, hold, yn) = {
        let m = ark_mem.borrow();
        (m.initsetup, m.h, m.hold, m.yn.clone().expect("yn"))
    };
    if initsetup {
        N_VScale(ONE, &yn, yguess);
        return ARK_SUCCESS;
    }

    /* set evaluation time tau as relative shift from previous successful time */
    tau = {
        let s = arkStep_mem_mut(ark_mem);
        let Bi = s.Bi.as_ref().expect("Bi").borrow();
        Bi.c[istage as usize]
    } * ark_h
        / hold;

    /* use requested predictor formula */
    #[allow(clippy::never_loop)]
    loop {
        match predictor {
            1 => {
                /***** Interpolatory Predictor 1 -- all to max order *****/
                retval = arkPredict_MaximumOrder(ark_mem, tau, yguess);
                if retval != ARK_ILL_INPUT {
                    return retval;
                }
                break;
            }

            2 => {
                /***** Interpolatory Predictor 2 -- decrease order w/ increasing
                level of extrapolation *****/
                retval = arkPredict_VariableOrder(ark_mem, tau, yguess);
                if retval != ARK_ILL_INPUT {
                    return retval;
                }
                break;
            }

            3 => {
                /***** Cutoff predictor: max order interpolatory output for stages "close"
                to previous step, first-order predictor for subsequent stages *****/
                retval = arkPredict_CutoffOrder(ark_mem, tau, yguess);
                if retval != ARK_ILL_INPUT {
                    return retval;
                }
                break;
            }

            4 => {
                /***** Bootstrap predictor: if any previous stage in step has nonzero c_i,
                construct a quadratic Hermite interpolant for prediction; otherwise
                use the trivial predictor.  The actual calculations are performed in
                arkPredict_Bootstrap, but here we need to determine the appropriate
                stage, c_j, to use. *****/

                /* determine if any previous stages in step meet criteria */
                let mut jstage: i32 = -1;
                for i in 0..istage {
                    let ci = {
                        let s = arkStep_mem_mut(ark_mem);
                        let Bi = s.Bi.as_ref().expect("Bi").borrow();
                        Bi.c[i as usize]
                    };
                    jstage = if ci != ZERO { i } else { jstage };
                }

                /* if using the trivial predictor, break */
                if jstage == -1 {
                    break;
                }

                /* find the "optimal" previous stage to use */
                for i in 0..istage {
                    let (ci, cj) = {
                        let s = arkStep_mem_mut(ark_mem);
                        let Bi = s.Bi.as_ref().expect("Bi").borrow();
                        (Bi.c[i as usize], Bi.c[jstage as usize])
                    };
                    if (ci > cj) && (ci != ZERO) {
                        jstage = i;
                    }
                }

                /* set stage time, stage RHS and interpolation values */
                let (cj, ci) = {
                    let s = arkStep_mem_mut(ark_mem);
                    let Bi = s.Bi.as_ref().expect("Bi").borrow();
                    (Bi.c[jstage as usize], Bi.c[istage as usize])
                };
                /* C's `sunrealtype h;` local (distinct from `ark_mem->h`) */
                let h = ark_h * cj;
                tau = ark_h * ci;
                let mut nvec: i32 = 0;
                if implicit {
                    /* Implicit piece */
                    cvals.push(ONE);
                    Xvecs.push(arkStep_mem_mut(ark_mem).Fi[jstage as usize].clone());
                    nvec += 1;
                }
                if explicit {
                    /* Explicit piece */
                    cvals.push(ONE);
                    Xvecs.push(arkStep_mem_mut(ark_mem).Fe[jstage as usize].clone());
                    nvec += 1;
                }

                /* call predictor routine */
                retval = arkPredict_Bootstrap(ark_mem, h, tau, nvec, &mut cvals, &mut Xvecs, yguess);
                if retval != ARK_ILL_INPUT {
                    return retval;
                }
                break;
            }

            5 => {
                /***** Minimal correction predictor: use all previous stage
                information in this step *****/

                /* set arrays for fused vector operation */
                let mut nvec: i32 = 0;
                if explicit {
                    /* Explicit pieces */
                    for jstage in 0..istage {
                        let (aij, Fe_j) = {
                            let s = arkStep_mem_mut(ark_mem);
                            let a = s.Be.as_ref().expect("Be").borrow().A[istage as usize]
                                [jstage as usize];
                            (a, s.Fe[jstage as usize].clone())
                        };
                        cvals.push(ark_h * aij);
                        Xvecs.push(Fe_j);
                        nvec += 1;
                    }
                }
                if implicit {
                    /* Implicit pieces */
                    for jstage in 0..istage {
                        let (aij, Fi_j) = {
                            let s = arkStep_mem_mut(ark_mem);
                            let a = s.Bi.as_ref().expect("Bi").borrow().A[istage as usize]
                                [jstage as usize];
                            (a, s.Fi[jstage as usize].clone())
                        };
                        cvals.push(ark_h * aij);
                        Xvecs.push(Fi_j);
                        nvec += 1;
                    }
                }
                cvals.push(ONE);
                Xvecs.push(yn.clone());
                nvec += 1;

                /* compute predictor */
                retval = N_VLinearCombination(nvec, &mut cvals, &mut Xvecs, yguess);
                if retval != 0 {
                    return ARK_VECTOROP_ERR;
                }
                return ARK_SUCCESS;
            }

            _ => {
                break;
            }
        }
    }

    /* if we made it here, use the trivial predictor (previous step solution) */
    N_VScale(ONE, &yn, yguess);
    ARK_SUCCESS
}

/*---------------------------------------------------------------
arkStep_StageSetup

This routine sets up the stage data for computing the RK
residual, along with the step- and method-related factors
gamma, gammap and gamrat.  See the C source for the full
derivation of the three internal modes:

Explicit (any):
  sdata = h*sum_{j=0}^{i-1} (Ae(i,j)*Fe(j) + Ai(i,j)*Fi(j))
Implicit, M:
  sdata = M*(yn - zp) + h*sum_{j=0}^{i-1} (Ae(i,j)*Fe(j) + Ai(i,j)*Fi(j))
Implicit, I or M(t):
  sdata = yn - zp + h*sum_{j=0}^{i-1} (Ae(i,j)*Fe(j) + Ai(i,j)*Fi(j))
---------------------------------------------------------------*/
pub fn arkStep_StageSetup(ark_mem: &ARKodeMem, implicit: sunbooleantype) -> i32 {
    let mut retval: i32;

    /* access ARKodeARKStepMem structure */
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            None,
            ARK_MEM_NULL,
            line!() as i32,
            "arkStep_StageSetup",
            file!(),
            MSG_ARKSTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    /* Set shortcut to current stage index */
    let (i, step_explicit, step_implicit, mass_type, sdata, zpred) = {
        let s = arkStep_mem_mut(ark_mem);
        (
            s.istage,
            s.explicit,
            s.implicit,
            s.mass_type,
            s.sdata.clone().expect("sdata"),
            s.zpred.clone().expect("zpred"),
        )
    };

    /* local shortcuts for fused vector operations: built locally */
    let mut cvals: Vec<sunrealtype> = Vec::new();
    let mut Xvecs: Vec<N_Vector> = Vec::new();

    let (h, tn, firststage, yn) = {
        let m = ark_mem.borrow();
        (m.h, m.tn, m.firststage, m.yn.clone().expect("yn"))
    };

    /* Update gamma if stage is implicit */
    if implicit {
        let aii = {
            let s = arkStep_mem_mut(ark_mem);
            let a = s.Bi.as_ref().expect("Bi").borrow().A[i as usize][i as usize];
            a
        };
        let mut s = arkStep_mem_mut(ark_mem);
        s.gamma = h * aii;
        if firststage {
            s.gammap = s.gamma;
        }
        /* protect x/x != 1.0 */
        s.gamrat = if firststage { ONE } else { s.gamma / s.gammap };
    }

    /* If implicit, initialize sdata to yn - zpred (here: zpred = zp), and set
    first entries for eventual N_VLinearCombination call */
    let mut nvec: i32 = 0;
    if implicit {
        N_VLinearSum(ONE, &yn, -ONE, &zpred, &sdata);
        cvals.push(ONE);
        Xvecs.push(sdata.clone());
        nvec = 1;
    }

    /* If implicit with fixed M!=I, update sdata with M*sdata */
    if implicit && (mass_type == MASS_FIXED) {
        let tempv1 = ark_mem.borrow().tempv1.clone().expect("tempv1");
        N_VScale(ONE, &sdata, &tempv1);
        let mmult = arkStep_mem_mut(ark_mem).mmult.expect("mmult");
        retval = mmult(ark_mem, &tempv1, &sdata);
        if retval != ARK_SUCCESS {
            return ARK_MASSMULT_FAIL;
        }
    }

    /* Update sdata with prior stage information */
    if step_explicit {
        /* Explicit pieces */
        for j in 0..i {
            let (aij, Fe_j) = {
                let s = arkStep_mem_mut(ark_mem);
                let a = s.Be.as_ref().expect("Be").borrow().A[i as usize][j as usize];
                (a, s.Fe[j as usize].clone())
            };
            cvals.push(h * aij);
            Xvecs.push(Fe_j);
            nvec += 1;
        }
    }
    if step_implicit {
        /* Implicit pieces */
        for j in 0..i {
            let (aij, Fi_j) = {
                let s = arkStep_mem_mut(ark_mem);
                let a = s.Bi.as_ref().expect("Bi").borrow().A[i as usize][j as usize];
                (a, s.Fi[j as usize].clone())
            };
            cvals.push(h * aij);
            Xvecs.push(Fi_j);
            nvec += 1;
        }
    }

    /* apply external polynomial (MRI) forcing (M = I required) */
    let (expforcing, impforcing) = {
        let s = arkStep_mem_mut(ark_mem);
        (s.expforcing, s.impforcing)
    };
    if expforcing || impforcing {
        let jmax: i32 = if expforcing { i } else { i + 1 };
        {
            let mut s = arkStep_mem_mut(ark_mem);
            for j in 0..jmax as usize {
                let (cjj, aij) = if expforcing {
                    let B = s.Be.as_ref().expect("Be").borrow();
                    (B.c[j], B.A[i as usize][j])
                } else {
                    let B = s.Bi.as_ref().expect("Bi").borrow();
                    (B.c[j], B.A[i as usize][j])
                };
                s.stage_times[j] = tn + cjj * h;
                s.stage_coefs[j] = h * aij;
            }
        }

        let s = arkStep_mem_mut(ark_mem);
        let stage_times = s.stage_times.clone();
        let stage_coefs = s.stage_coefs.clone();
        arkStep_ApplyForcing(
            &s,
            &mut cvals,
            &mut Xvecs,
            &stage_times,
            &stage_coefs,
            jmax,
            &mut nvec,
        );
    }

    /* call fused vector operation to do the work */
    retval = N_VLinearCombination(nvec, &cvals, &Xvecs, &sdata);
    if retval != 0 {
        return ARK_VECTOROP_ERR;
    }

    /* return with success */
    ARK_SUCCESS
}

/*---------------------------------------------------------------
arkStep_ComputeSolutions

This routine calculates the final RK solution using the existing
data.  This solution is placed directly in ark_ycur.  This routine
also computes the error estimate ||y-ytilde||_WRMS, where ytilde
is the embedded solution, and the norm weights come from
ark_ewt.  This norm value is returned.  The vector form of this
estimated error (y-ytilde) is stored in ark_mem->tempv1, in case
the calling routine wishes to examine the error locations.

This version assumes either an identity or time-dependent mass
matrix (identical steps).
---------------------------------------------------------------*/
pub fn arkStep_ComputeSolutions(ark_mem: &ARKodeMem, dsmPtr: &mut sunrealtype) -> i32 {
    let mut retval: i32;

    /* access ARKodeARKStepMem structure */
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            None,
            ARK_MEM_NULL,
            line!() as i32,
            "arkStep_ComputeSolutions",
            file!(),
            MSG_ARKSTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    /* set N_Vector shortcuts, and shortcut to time at end of step */
    let (y, yerr, h, tn, yn, ewt, fixedstep, accum_error_type) = {
        let m = ark_mem.borrow();
        (
            m.ycur.clone().expect("ycur"),
            m.tempv1.clone().expect("tempv1"),
            m.h,
            m.tn,
            m.yn.clone().expect("yn"),
            m.ewt.clone().expect("ewt"),
            m.fixedstep,
            m.AccumErrorType,
        )
    };

    /* local shortcuts for fused vector operations: built locally */
    let mut cvals: Vec<sunrealtype> = Vec::new();
    let mut Xvecs: Vec<N_Vector> = Vec::new();

    let (explicit, implicit, stages, expforcing, impforcing) = {
        let s = arkStep_mem_mut(ark_mem);
        (s.explicit, s.implicit, s.stages, s.expforcing, s.impforcing)
    };

    /* initialize output */
    *dsmPtr = ZERO;

    /* check if the method is stiffly accurate */
    let mut stiffly_accurate: sunbooleantype = SUNTRUE;

    if explicit {
        let Be = arkStep_mem_mut(ark_mem).Be.clone().expect("Be");
        if !ARKodeButcherTable_IsStifflyAccurate(Some(&Be)) {
            stiffly_accurate = SUNFALSE;
        }
    }

    if implicit {
        let Bi = arkStep_mem_mut(ark_mem).Bi.clone().expect("Bi");
        if !ARKodeButcherTable_IsStifflyAccurate(Some(&Bi)) {
            stiffly_accurate = SUNFALSE;
        }
    }

    /* If the method is stiffly accurate, ycur is already the new solution */

    if !stiffly_accurate {
        /* Compute time step solution (if necessary) */
        /*   set arrays for fused vector operation */
        cvals.push(ONE);
        Xvecs.push(yn.clone());
        let mut nvec: i32 = 1;
        for j in 0..stages as usize {
            if explicit {
                /* Explicit pieces */
                let (bj, Fe_j) = {
                    let s = arkStep_mem_mut(ark_mem);
                    let b = s.Be.as_ref().expect("Be").borrow().b[j];
                    (b, s.Fe[j].clone())
                };
                cvals.push(h * bj);
                Xvecs.push(Fe_j);
                nvec += 1;
            }
            if implicit {
                /* Implicit pieces */
                let (bj, Fi_j) = {
                    let s = arkStep_mem_mut(ark_mem);
                    let b = s.Bi.as_ref().expect("Bi").borrow().b[j];
                    (b, s.Fi[j].clone())
                };
                cvals.push(h * bj);
                Xvecs.push(Fi_j);
                nvec += 1;
            }
        }

        /* apply external polynomial (MRI) forcing (M = I required) */
        if expforcing || impforcing {
            {
                let mut s = arkStep_mem_mut(ark_mem);
                for j in 0..stages as usize {
                    let (cj, bj) = if expforcing {
                        let B = s.Be.as_ref().expect("Be").borrow();
                        (B.c[j], B.b[j])
                    } else {
                        let B = s.Bi.as_ref().expect("Bi").borrow();
                        (B.c[j], B.b[j])
                    };
                    s.stage_times[j] = tn + cj * h;
                    s.stage_coefs[j] = h * bj;
                }
            }

            let s = arkStep_mem_mut(ark_mem);
            let stage_times = s.stage_times.clone();
            let stage_coefs = s.stage_coefs.clone();
            arkStep_ApplyForcing(
                &s,
                &mut cvals,
                &mut Xvecs,
                &stage_times,
                &stage_coefs,
                stages,
                &mut nvec,
            );
        }

        /*   call fused vector operation to do the work */
        retval = N_VLinearCombination(nvec, &cvals, &Xvecs, &y);
        if retval != 0 {
            return ARK_VECTOROP_ERR;
        }

        let PostProcessStepFn = ark_mem.borrow().PostProcessStepFn;
        if let Some(PostProcessStepFn) = PostProcessStepFn {
            let (tcur, ycur) = {
                let m = ark_mem.borrow();
                (m.tcur, m.ycur.clone().expect("ycur"))
            };
            let mut user_data = ark_mem.borrow_mut().user_data.take();
            retval = PostProcessStepFn(tcur, &ycur, &mut user_data);
            ark_mem.borrow_mut().user_data = user_data;
            if retval != 0 {
                return ARK_POSTPROCESS_STEP_FAIL;
            }
        }
    }

    /* Compute yerr (if temporal error estimation is enabled). */
    if !fixedstep || (accum_error_type != ARK_ACCUMERROR_NONE) {
        /* set arrays for fused vector operation */
        cvals.clear();
        Xvecs.clear();
        let mut nvec: i32 = 0;
        for j in 0..stages as usize {
            if explicit {
                /* Explicit pieces */
                let (bj, dj, Fe_j) = {
                    let s = arkStep_mem_mut(ark_mem);
                    let B = s.Be.as_ref().expect("Be").borrow();
                    let (b, d) = (B.b[j], B.d[j]);
                    drop(B);
                    (b, d, s.Fe[j].clone())
                };
                cvals.push(h * (bj - dj));
                Xvecs.push(Fe_j);
                nvec += 1;
            }
            if implicit {
                /* Implicit pieces */
                let (bj, dj, Fi_j) = {
                    let s = arkStep_mem_mut(ark_mem);
                    let B = s.Bi.as_ref().expect("Bi").borrow();
                    let (b, d) = (B.b[j], B.d[j]);
                    drop(B);
                    (b, d, s.Fi[j].clone())
                };
                cvals.push(h * (bj - dj));
                Xvecs.push(Fi_j);
                nvec += 1;
            }
        }

        /* apply external polynomial (MRI) forcing (M = I required) */
        if expforcing || impforcing {
            {
                let mut s = arkStep_mem_mut(ark_mem);
                for j in 0..stages as usize {
                    let (cj, bj, dj) = if expforcing {
                        let B = s.Be.as_ref().expect("Be").borrow();
                        (B.c[j], B.b[j], B.d[j])
                    } else {
                        let B = s.Bi.as_ref().expect("Bi").borrow();
                        (B.c[j], B.b[j], B.d[j])
                    };
                    s.stage_times[j] = tn + cj * h;
                    s.stage_coefs[j] = h * (bj - dj);
                }
            }

            let s = arkStep_mem_mut(ark_mem);
            let stage_times = s.stage_times.clone();
            let stage_coefs = s.stage_coefs.clone();
            arkStep_ApplyForcing(
                &s,
                &mut cvals,
                &mut Xvecs,
                &stage_times,
                &stage_coefs,
                stages,
                &mut nvec,
            );
        }

        /* call fused vector operation to do the work */
        retval = N_VLinearCombination(nvec, &cvals, &Xvecs, &yerr);
        if retval != 0 {
            return ARK_VECTOROP_ERR;
        }

        /* fill error norm */
        *dsmPtr = N_VWrmsNorm(&yerr, &ewt);
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
arkStep_ComputeSolutions_MassFixed

This routine calculates the final RK solution using the existing
data.  This solution is placed directly in ark_ycur.  This routine
also computes the error estimate ||y-ytilde||_WRMS, where ytilde
is the embedded solution, and the norm weights come from
ark_ewt.  This norm value is returned.  The vector form of this
estimated error (y-ytilde) is stored in ark_mem->tempv1, in case
the calling routine wishes to examine the error locations.

This version assumes a fixed mass matrix.
---------------------------------------------------------------*/
pub fn arkStep_ComputeSolutions_MassFixed(ark_mem: &ARKodeMem, dsmPtr: &mut sunrealtype) -> i32 {
    let mut retval: i32;

    /* access ARKodeARKStepMem structure */
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            None,
            ARK_MEM_NULL,
            line!() as i32,
            "arkStep_ComputeSolutions_MassFixed",
            file!(),
            MSG_ARKSTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    /* set N_Vector shortcuts, and shortcut to time at end of step */
    let (y, yerr, h, yn, ewt, fixedstep) = {
        let m = ark_mem.borrow();
        (
            m.ycur.clone().expect("ycur"),
            m.tempv1.clone().expect("tempv1"),
            m.h,
            m.yn.clone().expect("yn"),
            m.ewt.clone().expect("ewt"),
            m.fixedstep,
        )
    };

    /* local shortcuts for fused vector operations: built locally */
    let mut cvals: Vec<sunrealtype> = Vec::new();
    let mut Xvecs: Vec<N_Vector> = Vec::new();

    let (explicit, implicit, stages, msolve, nlscoef) = {
        let s = arkStep_mem_mut(ark_mem);
        (
            s.explicit,
            s.implicit,
            s.stages,
            s.msolve.expect("msolve"),
            s.nlscoef,
        )
    };

    /* initialize output */
    *dsmPtr = ZERO;

    /* check if the method is stiffly accurate */
    let mut stiffly_accurate: sunbooleantype = SUNTRUE;

    if explicit {
        let Be = arkStep_mem_mut(ark_mem).Be.clone().expect("Be");
        if !ARKodeButcherTable_IsStifflyAccurate(Some(&Be)) {
            stiffly_accurate = SUNFALSE;
        }
    }

    if implicit {
        let Bi = arkStep_mem_mut(ark_mem).Bi.clone().expect("Bi");
        if !ARKodeButcherTable_IsStifflyAccurate(Some(&Bi)) {
            stiffly_accurate = SUNFALSE;
        }
    }

    /* If the method is stiffly accurate, ycur is already the new solution */

    if !stiffly_accurate {
        /* compute y RHS (store in y) */
        /*   set arrays for fused vector operation */
        let mut nvec: i32 = 0;
        for j in 0..stages as usize {
            if explicit {
                /* Explicit pieces */
                let (bj, Fe_j) = {
                    let s = arkStep_mem_mut(ark_mem);
                    let b = s.Be.as_ref().expect("Be").borrow().b[j];
                    (b, s.Fe[j].clone())
                };
                cvals.push(h * bj);
                Xvecs.push(Fe_j);
                nvec += 1;
            }
            if implicit {
                /* Implicit pieces */
                let (bj, Fi_j) = {
                    let s = arkStep_mem_mut(ark_mem);
                    let b = s.Bi.as_ref().expect("Bi").borrow().b[j];
                    (b, s.Fi[j].clone())
                };
                cvals.push(h * bj);
                Xvecs.push(Fi_j);
                nvec += 1;
            }
        }

        /*   call fused vector operation to compute RHS */
        retval = N_VLinearCombination(nvec, &cvals, &Xvecs, &y);
        if retval != 0 {
            return ARK_VECTOROP_ERR;
        }

        /* solve for y update (stored in y) */
        retval = msolve(ark_mem, &y, nlscoef);
        if retval < 0 {
            /* indicate too much error, step with smaller step */
            *dsmPtr = 2.0;
            /* place old solution into y */
            N_VScale(ONE, &yn, &y);
            return CONV_FAIL;
        }

        /* compute y = yn + update */
        N_VLinearSum(ONE, &yn, ONE, &y, &y);

        let PostProcessStepFn = ark_mem.borrow().PostProcessStepFn;
        if let Some(PostProcessStepFn) = PostProcessStepFn {
            let (tcur, ycur) = {
                let m = ark_mem.borrow();
                (m.tcur, m.ycur.clone().expect("ycur"))
            };
            let mut user_data = ark_mem.borrow_mut().user_data.take();
            retval = PostProcessStepFn(tcur, &ycur, &mut user_data);
            ark_mem.borrow_mut().user_data = user_data;
            if retval != 0 {
                return ARK_POSTPROCESS_STEP_FAIL;
            }
        }
    }

    /* compute yerr (if step adaptivity enabled) */
    if !fixedstep {
        /* compute yerr RHS vector */
        /*   set arrays for fused vector operation */
        cvals.clear();
        Xvecs.clear();
        let mut nvec: i32 = 0;
        for j in 0..stages as usize {
            if explicit {
                /* Explicit pieces */
                let (bj, dj, Fe_j) = {
                    let s = arkStep_mem_mut(ark_mem);
                    let B = s.Be.as_ref().expect("Be").borrow();
                    let (b, d) = (B.b[j], B.d[j]);
                    drop(B);
                    (b, d, s.Fe[j].clone())
                };
                cvals.push(h * (bj - dj));
                Xvecs.push(Fe_j);
                nvec += 1;
            }
            if implicit {
                /* Implicit pieces */
                let (bj, dj, Fi_j) = {
                    let s = arkStep_mem_mut(ark_mem);
                    let B = s.Bi.as_ref().expect("Bi").borrow();
                    let (b, d) = (B.b[j], B.d[j]);
                    drop(B);
                    (b, d, s.Fi[j].clone())
                };
                cvals.push(h * (bj - dj));
                Xvecs.push(Fi_j);
                nvec += 1;
            }
        }

        /*   call fused vector operation to compute yerr RHS */
        retval = N_VLinearCombination(nvec, &cvals, &Xvecs, &yerr);
        if retval != 0 {
            return ARK_VECTOROP_ERR;
        }

        /* solve for yerr */
        retval = msolve(ark_mem, &yerr, nlscoef);
        if retval < 0 {
            /* next attempt will reduce step by 'etacf';
            insert dsmPtr placeholder here */
            *dsmPtr = 2.0;
            return CONV_FAIL;
        }
        /* fill error norm */
        *dsmPtr = N_VWrmsNorm(&yerr, &ewt);
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
arkStep_TakeStep_ERK_Adjoint:

This routine performs a single backwards step of the discrete
adjoint of the ERK method.

Since we are not doing error control during the adjoint integration,
the output variable dsmPtr should should be 0.

The input/output variable nflagPtr is used to gauge convergence
of any algebraic solvers within the step. In this case, it should
always be 0 since we do not do any algebraic solves.

The return value from this routine is:
          0 => step completed successfully
         >0 => step encountered recoverable failure;
               reduce step and retry (if possible)
         <0 => step encountered unrecoverable failure
---------------------------------------------------------------*/
pub fn arkStep_TakeStep_ERK_Adjoint(
    ark_mem: &ARKodeMem,
    dsmPtr: &mut sunrealtype,
    nflagPtr: &mut i32,
) -> i32 {
    let mut retval: i32;

    /* access ARKodeARKStepMem structure */
    retval = arkStep_AccessStepMem(ark_mem, "arkStep_TakeStep_ERK_Adjoint");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* local shortcuts for readability */
    let adj_stepper: SUNAdjointStepper = ark_mem
        .borrow()
        .user_data
        .as_ref()
        .and_then(|b| b.downcast_ref::<SUNAdjointStepper>())
        .cloned()
        .expect("SUNAdjointStepper user_data");
    /* cvals / Xvecs operand lists are built locally (see the module docs) */
    let (sens_np1, sens_n, nst) = {
        let m = ark_mem.borrow();
        (
            m.yn.clone().expect("yn"),
            m.ycur.clone().expect("ycur"),
            m.nst,
        )
    };
    let (sens_tmp, stage_values, stages, Be) = {
        let s = arkStep_mem_mut(ark_mem);
        (
            s.sdata.clone().expect("sdata"),
            s.Fe.clone(),
            s.stages,
            s.Be.clone().expect("Be"),
        )
    };
    let sens_tmp_Lambda = N_VGetSubvector_ManyVector(&sens_tmp, 0);
    let sens_np1_lambda = N_VGetSubvector_ManyVector(&sens_np1, 0);

    /* which adjoint step is being processed */
    ark_mem.borrow_mut().adj_step_idx = adj_stepper.final_step_idx.get() - nst;

    /* determine if method has fsal property */
    let fsal: sunbooleantype =
        (SUNRabs(Be.borrow().A[0][0]) <= TINY) && ARKodeButcherTable_IsStifflyAccurate(Some(&Be));

    /* For FSAL ERK methods, Ae[s-1][s-1] == b[s-1] = 0 so F[s-1] is always zero */
    if fsal {
        N_VConst(0.0, &stage_values[(stages - 1) as usize]);
    }

    /* Loop over stages */
    let mut is: i32 = stages - if fsal { 2 } else { 1 };
    while is >= 0 {
        /* Consider solving a forward IVP from t0 to tf, tf > t0.
        The adjoint ODE is solved backwards in time with step size h' = -h
        where h is the forward time step used. So at this point in the
        code ark_mem->h is h', however, the adjoint formulae need h. */
        let h = ark_mem.borrow().h;
        let adj_h: sunrealtype = -h;

        /* which stage is being processed -- needed for loading checkpoints */
        ark_mem.borrow_mut().adj_stage_idx = is as suncountertype;

        /* Set current stage time(s) and index */
        {
            let ci = Be.borrow().c[is as usize];
            let mut m = ark_mem.borrow_mut();
            let tn = m.tn;
            let h = m.h;
            m.tcur = tn + h * (1.0 - ci);
        }

        /*
         * Compute partial current stage value \Lambda
         */
        let mut cvals: Vec<sunrealtype> = Vec::new();
        let mut Xvecs: Vec<N_Vector> = Vec::new();
        let mut nvec: i32 = 0;
        for js in (is + 1)..stages {
            /* h sum_{j=i}^{s} Ae_{ji} \Lambda_{j} */
            cvals.push(adj_h * Be.borrow().A[js as usize][is as usize]);
            Xvecs.push(N_VGetSubvector_ManyVector(&stage_values[js as usize], 0));
            nvec += 1;
        }
        cvals.push(adj_h * Be.borrow().b[is as usize]);
        Xvecs.push(sens_np1_lambda.clone());
        nvec += 1;

        /* h be_i \lambda_{n+1} + h sum_{j=i}^{s} Ae_{ji} \Lambda_{j} */
        retval = N_VLinearCombination(nvec, &cvals, &Xvecs, &sens_tmp_Lambda);
        if retval != 0 {
            return ARK_VECTOROP_ERR;
        }

        /* Compute the stages \Lambda_i and \nu_i by evaluating f_{y}^*(t_i, z_i, p) and
        f_{p}^*(t_i, z_i, p) and applying them to sens_tmp_Lambda (in sens_tmp). This is
        done in fe which retrieves z_i from the checkpoint data */
        let (fe, tcur) = {
            let fe = arkStep_mem_mut(ark_mem).fe.expect("fe");
            let tcur = ark_mem.borrow().tcur;
            (fe, tcur)
        };
        let mut user_data = ark_mem.borrow_mut().user_data.take();
        retval = fe(tcur, &sens_tmp, &stage_values[is as usize], &mut user_data);
        ark_mem.borrow_mut().user_data = user_data;
        arkStep_mem_mut(ark_mem).nfe += 1;

        /* The checkpoint was not found, so we need to recompute at least
        this step forward in time. We first seek the last checkpointed step
        solution, then recompute from there. */
        if ark_mem.borrow().load_checkpoint_fail {
            let tempv2 = ark_mem.borrow().tempv2.clone().expect("tempv2");
            let mut checkpoint = N_VGetSubvector_ManyVector(&tempv2, 0);
            let curr_step: suncountertype = ark_mem.borrow().adj_step_idx;
            let mut start_step: suncountertype = curr_step;

            let checkpoint_scheme = ark_mem
                .borrow()
                .checkpoint_scheme
                .clone()
                .expect("checkpoint_scheme");
            let mut errcode: SUNErrCode = SUN_ERR_CHECKPOINT_NOT_FOUND;
            let mut i: suncountertype = 0;
            while i <= curr_step {
                let mut checkpoint_t: sunrealtype = 0.0;
                errcode = SUNAdjointCheckpointScheme_LoadVector(
                    &checkpoint_scheme,
                    start_step,
                    stages as suncountertype,
                    /*peek=*/ SUNTRUE,
                    &mut checkpoint,
                    &mut checkpoint_t,
                );
                if errcode == SUN_SUCCESS {
                    /* OK, now we have the last checkpoint that stored as (start_step, stages).
                    This represents the last step solution that was checkpointed. As such, we
                    want to recompute start_step+1 to stop_step. */
                    start_step += 1;
                    let t0 = checkpoint_t;
                    let tf = ark_mem.borrow().tn;
                    errcode = SUNAdjointStepper_RecomputeFwd(
                        &adj_stepper,
                        start_step,
                        t0,
                        &checkpoint,
                        tf,
                    );
                    if errcode != SUN_SUCCESS {
                        arkProcessError(
                            Some(ark_mem),
                            ARK_ADJ_RECOMPUTE_FAIL,
                            line!() as i32,
                            "arkStep_TakeStep_ERK_Adjoint",
                            file!(),
                            &format!("SUNAdjointStepper_RecomputeFwd returned {errcode}"),
                        );
                        return ARK_ADJ_RECOMPUTE_FAIL;
                    }
                    return arkStep_TakeStep_ERK_Adjoint(ark_mem, dsmPtr, nflagPtr);
                }
                i += 1;
                start_step -= 1;
            }
            if errcode != SUN_SUCCESS {
                arkProcessError(
                    Some(ark_mem),
                    ARK_ADJ_RECOMPUTE_FAIL,
                    line!() as i32,
                    "arkStep_TakeStep_ERK_Adjoint",
                    file!(),
                    "Could not load or recompute missing step",
                );
                return ARK_ADJ_RECOMPUTE_FAIL;
            }
        } else if retval > 0 {
            return ARK_UNREC_RHSFUNC_ERR;
        } else if retval < 0 {
            arkProcessError(
                Some(ark_mem),
                ARK_RHSFUNC_FAIL,
                line!() as i32,
                "arkStep_TakeStep_ERK_Adjoint",
                file!(),
                &format!("The right hand side function failed returned {retval}"),
            );
            return ARK_RHSFUNC_FAIL;
        }

        is -= 1;
    }

    /* Throw away the step solution */
    let mut checkpoint_t: sunrealtype = ZERO;
    let tempv2 = ark_mem.borrow().tempv2.clone().expect("tempv2");
    let mut checkpoint = N_VGetSubvector_ManyVector(&tempv2, 0);

    let (checkpoint_scheme, adj_step_idx) = {
        let m = ark_mem.borrow();
        (
            m.checkpoint_scheme.clone().expect("checkpoint_scheme"),
            m.adj_step_idx,
        )
    };
    let errcode = SUNAdjointCheckpointScheme_LoadVector(
        &checkpoint_scheme,
        adj_step_idx,
        0,
        /*peek=*/ SUNFALSE,
        &mut checkpoint,
        &mut checkpoint_t,
    );
    if errcode != SUN_SUCCESS {
        arkProcessError(
            Some(ark_mem),
            ARK_ADJ_CHECKPOINT_FAIL,
            line!() as i32,
            "arkStep_TakeStep_ERK_Adjoint",
            file!(),
            &format!("SUNAdjointCheckpointScheme_LoadVector returned {errcode}"),
        );
        return ARK_ADJ_CHECKPOINT_FAIL;
    }

    /* Now compute the time step solution. We cannot use arkStep_ComputeSolutions because the
    adjoint calculation for the time step solution is different than the forward case. */

    let mut cvals: Vec<sunrealtype> = Vec::new();
    let mut Xvecs: Vec<N_Vector> = Vec::new();
    let mut nvec: i32 = 0;
    for j in 0..stages as usize {
        cvals.push(ONE);
        /* this needs to be the stage values [Lambda_i, nu_i] */
        Xvecs.push(stage_values[j].clone());
        nvec += 1;
    }
    cvals.push(ONE);
    Xvecs.push(sens_np1.clone());
    nvec += 1;

    /* \lambda_n = \lambda_{n+1} + \sum_{j=1}^{s} \Lambda_j
    \mu_n     = \mu_{n+1} + \sum_{j=1}^{s} \nu_j */
    retval = N_VLinearCombination(nvec, &cvals, &Xvecs, &sens_n);
    if retval != 0 {
        return ARK_VECTOROP_ERR;
    }

    *dsmPtr = ZERO;
    *nflagPtr = 0;

    ARK_SUCCESS
}

/*---------------------------------------------------------------
Utility routines for interfacing with SUNAdjointStepper
---------------------------------------------------------------*/

/// C `arkStep_fe_Adj(sunrealtype t, N_Vector sens_partial_stage,
/// N_Vector sens_complete_stage, void* content)` — installed as ARKStep's
/// `fe` for the adjoint memory, hence the [`ARKRhsFn`] shape.
///
/// `void* content` is the `SUNAdjointStepper` token stored in the adjoint
/// ARKODE memory's `user_data`.
pub fn arkStep_fe_Adj(
    t: sunrealtype,
    sens_partial_stage: &N_Vector,
    sens_complete_stage: &N_Vector,
    content: &mut Option<Box<dyn Any>>,
) -> i32 {
    let errcode: SUNErrCode;

    let adj_stepper: SUNAdjointStepper = content
        .as_ref()
        .and_then(|b| b.downcast_ref::<SUNAdjointStepper>())
        .cloned()
        .expect("SUNAdjointStepper content");
    let check_scheme = adj_stepper.checkpoint_scheme.borrow().clone();
    let adj_sunstepper = adj_stepper.adj_sunstepper.borrow().clone();
    let mut ark_mem_out: Option<ARKodeMem> = None;
    let _ = SUNStepper_GetContentAs::<ARKodeMem>(&adj_sunstepper, &mut ark_mem_out);
    let ark_mem = ark_mem_out.expect("ARKodeMem stepper content");

    /* access ARKodeARKStepMem structure */
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            None,
            ARK_MEM_NULL,
            line!() as i32,
            "arkStep_fe_Adj",
            file!(),
            MSG_ARKSTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    let adj_fe = arkStep_mem_mut(&ark_mem).adj_fe.expect("adj_fe");

    let tempv2 = ark_mem.borrow().tempv2.clone().expect("tempv2");
    let mut checkpoint = N_VGetSubvector_ManyVector(&tempv2, 0);
    let mut checkpoint_t: sunrealtype = 0.0;

    ark_mem.borrow_mut().load_checkpoint_fail = SUNFALSE;

    let (adj_step_idx, adj_stage_idx) = {
        let m = ark_mem.borrow();
        (m.adj_step_idx, m.adj_stage_idx)
    };
    errcode = SUNAdjointCheckpointScheme_LoadVector(
        &check_scheme,
        adj_step_idx,
        adj_stage_idx,
        SUNFALSE,
        &mut checkpoint,
        &mut checkpoint_t,
    );

    /* Checkpoint was not found, recompute the missing step */
    if errcode == SUN_ERR_CHECKPOINT_NOT_FOUND {
        ark_mem.borrow_mut().load_checkpoint_fail = SUNTRUE;
        return 1;
    }

    /* C: `void* user_data = adj_stepper->user_data;` aliases the FORWARD
    integrator's `user_data`, which the forward RHS also dereferences during
    `SUNAdjointStepper_RecomputeFwd`. A `Box` cannot alias and moving it
    would strand the forward RHS, so (accepted deviation class 6, the shape
    used for `ARKodeRootMemRec::root_data`) the token is taken from the
    adjoint stepper for the duration of the call and restored on every
    path. */
    let mut user_data = adj_stepper.user_data.borrow_mut().take();

    /* Evaluate f_{y}^*(t_i, z_i, p) \Lambda_i and f_{p}^*(t_i, z_i, p) \nu_i */
    let retval = adj_fe(
        t,
        &checkpoint,
        sens_partial_stage,
        sens_complete_stage,
        &mut user_data,
    );
    *adj_stepper.user_data.borrow_mut() = user_data;
    retval
}

pub fn arkStepCompatibleWithAdjointSolver(
    ark_mem: &ARKodeMem,
    lineno: i32,
    fname: &str,
    filename: &str,
) -> i32 {
    let (fi, fe, mass_type) = {
        let s = arkStep_mem_mut(ark_mem);
        (s.fi, s.fe, s.mass_type)
    };
    let (fixedstep, relax_enabled, has_constraints) = {
        let m = ark_mem.borrow();
        (m.fixedstep, m.relax_enabled, m.constraints.is_some())
    };

    if !fixedstep {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            lineno,
            fname,
            filename,
            "ARKStep must be using a fixed step to work with SUNAdjointStepper",
        );
        return ARK_ILL_INPUT;
    }

    if fi.is_some() {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            lineno,
            fname,
            filename,
            "SUNAdjointStepper requires fi = NULL (it only supports explicit RK methods)",
        );
        return ARK_ILL_INPUT;
    }

    if fe.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            lineno,
            fname,
            filename,
            "SUNAdjointStepper requires fe != NULL (it only supports explicit RK methods)",
        );
        return ARK_ILL_INPUT;
    }

    if relax_enabled {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            lineno,
            fname,
            filename,
            "SUNAdjointStepper is not compatible with relaxation",
        );
        return ARK_ILL_INPUT;
    }

    if mass_type != MASS_IDENTITY {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            lineno,
            fname,
            filename,
            "SUNAdjointStepper is not compatible with non-identity mass matrices",
        );
        return ARK_ILL_INPUT;
    }

    if has_constraints {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            lineno,
            fname,
            filename,
            "SUNAdjointStepper is not compatible with constraints",
        );
        return ARK_ILL_INPUT;
    }

    ARK_SUCCESS
}

fn arkStep_SUNStepperReInit(stepper: &SUNStepper, t0: sunrealtype, y0: &N_Vector) -> SUNErrCode {
    let mut arkode_mem: Option<ARKodeMem> = None;
    let _ = SUNStepper_GetContentAs::<ARKodeMem>(stepper, &mut arkode_mem);
    let arkode_mem = match arkode_mem {
        Some(arkode_mem) => arkode_mem,
        None => {
            arkProcessError(
                None,
                ARK_ILL_INPUT,
                line!() as i32,
                "arkStep_SUNStepperReInit",
                file!(),
                "The ARKStep memory pointer is NULL",
            );
            return ARK_ILL_INPUT;
        }
    };

    let retval = arkStep_AccessARKODEStepMem(&arkode_mem, "arkStep_SUNStepperReInit");
    if retval != 0 {
        arkProcessError(
            None,
            ARK_ILL_INPUT,
            line!() as i32,
            "arkStep_SUNStepperReInit",
            file!(),
            "The ARKStep memory pointer is NULL",
        );
        return ARK_ILL_INPUT;
    }
    let ark_mem = &arkode_mem;
    let (fe, fi) = {
        let s = arkStep_mem_mut(ark_mem);
        (s.fe, s.fi)
    };

    let last_flag = ARKStepReInit(&arkode_mem, fe, fi, t0, y0);
    *stepper.last_flag.borrow_mut() = last_flag;
    if last_flag != ARK_SUCCESS {
        arkProcessError(
            Some(ark_mem),
            last_flag,
            line!() as i32,
            "arkStep_SUNStepperReInit",
            file!(),
            "ARKStepReInit return an error\n",
        );
        return SUN_ERR_OP_FAIL;
    }

    SUN_SUCCESS
}

/// C `ARKStepCreateAdjointStepper(void* arkode_mem, SUNAdjRhsFn adj_fe,
/// SUNAdjRhsFn adj_fi, sunrealtype tf, N_Vector sf, SUNContext sunctx,
/// SUNAdjointStepper* adj_stepper_ptr)`.
#[allow(clippy::too_many_arguments)]
pub fn ARKStepCreateAdjointStepper(
    arkode_mem: &ARKodeMem,
    adj_fe: Option<SUNAdjRhsFn>,
    adj_fi: Option<SUNAdjRhsFn>,
    tf: sunrealtype,
    sf: &N_Vector,
    sunctx: &SUNContext,
    adj_stepper_ptr: &mut Option<SUNAdjointStepper>,
) -> i32 {
    let mut retval = arkStep_AccessARKODEStepMem(arkode_mem, "ARKStepCreateAdjointStepper");
    if retval != 0 {
        arkProcessError(
            None,
            ARK_ILL_INPUT,
            line!() as i32,
            "ARKStepCreateAdjointStepper",
            file!(),
            "The ARKStep memory pointer is NULL",
        );
        return ARK_ILL_INPUT;
    }
    let ark_mem = arkode_mem;

    if arkStepCompatibleWithAdjointSolver(
        ark_mem,
        line!() as i32,
        "ARKStepCreateAdjointStepper",
        file!(),
    ) != 0
    {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "ARKStepCreateAdjointStepper",
            file!(),
            "ark_mem provided is not compatible with adjoint calculation",
        );
        return ARK_ILL_INPUT;
    }

    if adj_fi.is_some() {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "ARKStepCreateAdjointStepper",
            file!(),
            "Implicit methods are not yet supported by the adjoint stepper.",
        );
        return ARK_ILL_INPUT;
    }

    if adj_fe.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "ARKStepCreateAdjointStepper",
            file!(),
            "adj_fe cannot be NULL.",
        );
        return ARK_ILL_INPUT;
    }

    if N_VGetVectorID(sf) != SUNDIALS_NVEC_MANYVECTOR {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "ARKStepCreateAdjointStepper",
            file!(),
            "Incompatible vector type provided for adjoint calculation",
        );
        return ARK_ILL_INPUT;
    }

    /*
      Create and configure the ARKStep stepper for the adjoint system
    */
    let mut nst: i64 = 0;
    retval = ARKodeGetNumSteps(arkode_mem, &mut nst);
    if retval != 0 {
        arkProcessError(
            Some(ark_mem),
            retval,
            line!() as i32,
            "ARKStepCreateAdjointStepper",
            file!(),
            "ARKodeGetNumSteps failed",
        );
        return retval;
    }

    let sunctx_fwd = ark_mem.borrow().sunctx.clone();
    let arkode_mem_adj = match ARKStepCreate(Some(arkStep_fe_Adj), None, tf, sf, &sunctx_fwd) {
        Some(arkode_mem_adj) => arkode_mem_adj,
        None => {
            arkProcessError(
                Some(ark_mem),
                ARK_MEM_NULL,
                line!() as i32,
                "ARKStepCreateAdjointStepper",
                file!(),
                "ARKStepCreate returned NULL",
            );
            return ARK_MEM_NULL;
        }
    };
    let ark_mem_adj = &arkode_mem_adj;

    arkStep_mem_mut(ark_mem_adj).adj_fe = adj_fe;
    ark_mem_adj.borrow_mut().do_adjoint = SUNTRUE;

    let h = ark_mem.borrow().h;
    retval = ARKodeSetFixedStep(&arkode_mem_adj, -h);
    if retval != 0 {
        arkProcessError(
            Some(ark_mem),
            retval,
            line!() as i32,
            "ARKStepCreateAdjointStepper",
            file!(),
            "ARKodeSetFixedStep failed",
        );
        return retval;
    }

    /* TODO(CJB): when we add support for implicit methods, we should call ARKodeSetLinear here. */

    let (Be, Bi) = {
        let s = arkStep_mem_mut(ark_mem);
        (s.Be.clone(), s.Bi.clone())
    };
    let (Be_q, Be_p) = {
        let B = Be.as_ref().expect("Be").borrow();
        (B.q, B.p)
    };
    retval = ARKStepSetTables(&arkode_mem_adj, Be_q, Be_p, Bi.as_ref(), Be.as_ref());
    if retval != 0 {
        arkProcessError(
            Some(ark_mem),
            retval,
            line!() as i32,
            "ARKStepCreateAdjointStepper",
            file!(),
            "ARKStepSetTables failed",
        );
        return retval;
    }

    retval = ARKodeSetMaxNumSteps(&arkode_mem_adj, nst);
    if retval != 0 {
        arkProcessError(
            Some(ark_mem),
            retval,
            line!() as i32,
            "ARKStepCreateAdjointStepper",
            file!(),
            "ARKodeSetMaxNumSteps failed",
        );
        return retval;
    }

    let checkpoint_scheme = ark_mem.borrow().checkpoint_scheme.clone();
    retval = ARKodeSetAdjointCheckpointScheme(&arkode_mem_adj, checkpoint_scheme.as_ref());
    if retval != 0 {
        arkProcessError(
            Some(ark_mem),
            retval,
            line!() as i32,
            "ARKStepCreateAdjointStepper",
            file!(),
            "ARKodeSetAdjointCheckpointScheme failed",
        );
        return retval;
    }

    let mut errcode: SUNErrCode;

    let mut fwd_stepper: Option<SUNStepper> = None;
    retval = ARKodeCreateSUNStepper(arkode_mem, &mut fwd_stepper);
    if retval != 0 {
        arkProcessError(
            Some(ark_mem),
            retval,
            line!() as i32,
            "ARKStepCreateAdjointStepper",
            file!(),
            "ARKodeCreateSUNStepper failed",
        );
        return retval;
    }
    let fwd_stepper = fwd_stepper.expect("fwd_stepper");

    errcode = SUNStepper_SetReInitFn(&fwd_stepper, Some(arkStep_SUNStepperReInit));
    if errcode != SUN_SUCCESS {
        retval = ARK_SUNSTEPPER_ERR;
        arkProcessError(
            Some(ark_mem),
            retval,
            line!() as i32,
            "ARKStepCreateAdjointStepper",
            file!(),
            "SUNStepper_SetReInitFn failed",
        );
        return retval;
    }

    let mut adj_stepper: Option<SUNStepper> = None;
    retval = ARKodeCreateSUNStepper(&arkode_mem_adj, &mut adj_stepper);
    if retval != 0 {
        arkProcessError(
            Some(ark_mem),
            retval,
            line!() as i32,
            "ARKStepCreateAdjointStepper",
            file!(),
            "ARKodeCreateSUNStepper failed",
        );
        return retval;
    }
    let adj_stepper = adj_stepper.expect("adj_stepper");

    errcode = SUNStepper_SetReInitFn(&adj_stepper, Some(arkStep_SUNStepperReInit));
    if errcode != SUN_SUCCESS {
        retval = ARK_SUNSTEPPER_ERR;
        arkProcessError(
            Some(ark_mem),
            retval,
            line!() as i32,
            "ARKStepCreateAdjointStepper",
            file!(),
            "SUNStepper_SetReInitFn failed",
        );
        return retval;
    }

    /* Setting this ensures that the ARKodeMem underneath the adj_stepper
    is destroyed with the SUNStepper_Destroy call. */
    errcode = SUNStepper_SetDestroyFn(&adj_stepper, Some(arkSUNStepperSelfDestruct));
    if errcode != SUN_SUCCESS {
        retval = ARK_SUNSTEPPER_ERR;
        arkProcessError(
            Some(ark_mem),
            retval,
            line!() as i32,
            "ARKStepCreateAdjointStepper",
            file!(),
            "SUNStepper_SetDestroyFn failed",
        );
        return retval;
    }

    /* SUNAdjointStepper will own the SUNSteppers and destroy them */
    errcode = SUNAdjointStepper_Create(
        fwd_stepper,
        SUNTRUE,
        adj_stepper,
        SUNTRUE,
        nst - 1,
        tf,
        sf,
        checkpoint_scheme.expect("checkpoint_scheme"),
        sunctx,
        adj_stepper_ptr,
    );
    if errcode != SUN_SUCCESS {
        retval = ARK_SUNADJSTEPPER_ERR;
        arkProcessError(
            Some(ark_mem),
            retval,
            line!() as i32,
            "ARKStepCreateAdjointStepper",
            file!(),
            "SUNAdjointStepper_Create failed",
        );
        return retval;
    }

    /* C: SUNAdjointStepper_SetUserData(*adj_stepper_ptr, ark_mem->user_data)
    ALIASES the forward integrator's `user_data` into the adjoint stepper --
    both `adj_fe` (through `arkStep_fe_Adj`) and the forward RHS (during
    SUNAdjointStepper_RecomputeFwd) dereference it. A `Box` cannot alias and
    moving it would strand the forward RHS, so (deviation class 6) the token
    is left with the forward memory; the example/integration layer must hand
    the adjoint stepper its own copy with SUNAdjointStepper_SetUserData. */
    errcode =
        SUNAdjointStepper_SetUserData(adj_stepper_ptr.as_ref().expect("adj_stepper_ptr"), None);
    if errcode != SUN_SUCCESS {
        retval = ARK_SUNADJSTEPPER_ERR;
        arkProcessError(
            Some(ark_mem),
            retval,
            line!() as i32,
            "ARKStepCreateAdjointStepper",
            file!(),
            "SUNAdjointStepper_SetUserData failed",
        );
        return retval;
    }

    /* We need access to the adjoint solver to access the parameter Jacobian inside of ARKStep's
    backwards integration of the the adjoint problem. */
    retval = ARKodeSetUserData(
        &arkode_mem_adj,
        Some(Box::new(
            adj_stepper_ptr.as_ref().expect("adj_stepper_ptr").clone(),
        )),
    );
    if retval != 0 {
        arkProcessError(
            Some(ark_mem),
            retval,
            line!() as i32,
            "ARKStepCreateAdjointStepper",
            file!(),
            "ARKodeSetUserData failed",
        );
        return retval;
    }

    ARK_SUCCESS
}

/*===============================================================
Internal utility routines for interacting with MRIStep
===============================================================*/

/*------------------------------------------------------------------------------
arkStep_ApplyForcing

Determines the scaling values and vectors necessary for the MRI polynomial
forcing terms. This occurs through appending scaling values and N_Vector
pointers to the cvals and Xvecs operand lists.

stage_times -- The times at which to evaluate the forcing.

stage_coefs -- Scaling factors (method A, b, or b - d coefficients) applied to
forcing vectors.

jmax -- the number of values in stage_times and stage_coefs (the stage index
for explicit methods or the index + 1 for implicit methods).

nvec -- On input, nvec is the next available entry in the cvals/Xvecs arrays.
This value is incremented for each value/vector appended to the cvals/Xvecs
arrays so on return it is the total number of values/vectors in the linear
combination.

In C the two arrays are `step_mem->cvals`/`step_mem->Xvecs` written at
`offset = *nvec`; here they are the caller's operand lists, whose length is
exactly `*nvec` on entry, so appending writes the very same slots.
----------------------------------------------------------------------------*/
pub fn arkStep_ApplyForcing(
    step_mem: &ARKodeARKStepMemRec,
    cvals: &mut Vec<sunrealtype>,
    Xvecs: &mut Vec<N_Vector>,
    stage_times: &[sunrealtype],
    stage_coefs: &[sunrealtype],
    jmax: i32,
    nvec: &mut i32,
) {
    /* Shortcuts to step_mem data */
    let tshift = step_mem.tshift;
    let tscale = step_mem.tscale;
    let nforcing = step_mem.nforcing;
    let forcing = &step_mem.forcing;

    /* Offset into vals and vecs arrays */
    let offset = *nvec as usize;

    /* Initialize scaling values, set vectors */
    for k in 0..nforcing as usize {
        cvals.push(ZERO);
        Xvecs.push(forcing[k].clone());
    }

    for j in 0..jmax as usize {
        let tau = (stage_times[j] - tshift) / tscale;
        let mut taui = ONE;

        for k in 0..nforcing as usize {
            cvals[offset + k] += stage_coefs[j] * taui;
            taui *= tau;
        }
    }

    /* Update vector count for linear combination */
    *nvec += nforcing;
}

/*------------------------------------------------------------------------------
arkStep_SetInnerForcing

Sets an array of coefficient vectors for a time-dependent external polynomial
forcing term in the ODE RHS i.e., y' = fe(t,y) + fi(t,y) + p(t). This
function is primarily intended for use with multirate integration methods
(e.g., MRIStep) where ARKStep is used to solve a modified ODE at a fast time
scale. The polynomial is of the form

p(t) = sum_{i = 0}^{nvecs - 1} forcing[i] * ((t - tshift) / (tscale))^i

where tshift and tscale are used to normalize the time t (e.g., with MRIGARK
methods).
----------------------------------------------------------------------------*/
pub fn arkStep_SetInnerForcing(
    ark_mem: &ARKodeMem,
    tshift: sunrealtype,
    tscale: sunrealtype,
    forcing: &[N_Vector],
    nvecs: i32,
) -> i32 {
    /* access ARKodeARKStepMem structure */
    let retval = arkStep_AccessStepMem(ark_mem, "arkStep_SetInnerForcing");
    if retval != ARK_SUCCESS {
        return retval;
    }

    if nvecs > 0 {
        /* enable forcing */
        {
            let mut step_mem = arkStep_mem_mut(ark_mem);
            if step_mem.explicit {
                step_mem.expforcing = SUNTRUE;
                step_mem.impforcing = SUNFALSE;
            } else {
                step_mem.expforcing = SUNFALSE;
                step_mem.impforcing = SUNTRUE;
            }
            step_mem.tshift = tshift;
            step_mem.tscale = tscale;
            step_mem.forcing = forcing.to_vec();
            step_mem.nforcing = nvecs;
        }

        /* If cvals and Xvecs are not allocated then arkStep_Init has not been
        called and the number of stages has not been set yet. These arrays will
        be allocated in arkStep_Init and take into account the value of nforcing.
        On subsequent calls will check if enough space has allocated in case
        nforcing has increased since the original allocation. */
        let (have_cvals, have_Xvecs, nfusedopvecs, stages) = {
            let s = arkStep_mem_mut(ark_mem);
            (
                !s.cvals.is_empty(),
                !s.Xvecs.is_empty(),
                s.nfusedopvecs,
                s.stages,
            )
        };
        if have_cvals && have_Xvecs {
            /* check if there are enough reusable arrays for fused operations */
            if (nfusedopvecs - nvecs) < (2 * stages + 2) {
                /* free current work space */
                if have_cvals {
                    arkStep_mem_mut(ark_mem).cvals = Vec::new();
                    ark_mem.borrow_mut().lrw -= nfusedopvecs as i64;
                }
                if have_Xvecs {
                    arkStep_mem_mut(ark_mem).Xvecs = Vec::new();
                    ark_mem.borrow_mut().liw -= nfusedopvecs as i64;
                }

                /* allocate reusable arrays for fused vector operations */
                let nfusedopvecs = 2 * stages + 2 + nvecs;
                arkStep_mem_mut(ark_mem).nfusedopvecs = nfusedopvecs;

                arkStep_mem_mut(ark_mem).cvals = vec![ZERO; nfusedopvecs as usize];
                ark_mem.borrow_mut().lrw += nfusedopvecs as i64;

                arkStep_mem_mut(ark_mem).Xvecs = vec![None; nfusedopvecs as usize];
                ark_mem.borrow_mut().liw += nfusedopvecs as i64;
            }
        }
    } else {
        /* disable forcing */
        let mut step_mem = arkStep_mem_mut(ark_mem);
        step_mem.expforcing = SUNFALSE;
        step_mem.impforcing = SUNFALSE;
        step_mem.tshift = ZERO;
        step_mem.tscale = ONE;
        step_mem.forcing = Vec::new();
        step_mem.nforcing = 0;
    }

    0
}

/*===============================================================
Internal utility routines for relaxation
===============================================================*/

/* -----------------------------------------------------------------------------
 * arkStep_RelaxDeltaE
 *
 * Computes the change in the relaxation functions for use in relaxation methods
 * delta_e = h * sum_i b_i * <relax_jac(z_i), f_i>
 *
 * With implicit and IMEX methods it is necessary to store the method stages
 * (or compute the delta_e estimate along the way) to avoid inconsistencies
 * between z_i, F(z_i), and J_relax(z_i) that arise from reconstructing stages
 * from stored RHS values like with ERK methods. As such the take step function
 * stores the stages along the way but only when there is an implicit RHS. When
 * a fixed mass matrix is present the stages are also stored to avoid additional
 * mass matrix solves in reconstructing the stages for an ERK method.
 * ---------------------------------------------------------------------------*/
pub fn arkStep_RelaxDeltaE(
    ark_mem: &ARKodeMem,
    relax_jac_fn: Option<ARKRelaxJacFn>,
    num_relax_jac_evals: &mut i64,
    delta_e_out: &mut sunrealtype,
) -> i32 {
    let mut retval: i32;
    let (mut z_stage, J_relax, h, yn) = {
        let m = ark_mem.borrow();
        (
            m.tempv2.clone().expect("tempv2"),
            m.tempv3.clone().expect("tempv3"),
            m.h,
            m.yn.clone().expect("yn"),
        )
    };
    let mut rhs_tmp: N_Vector;
    let mut bi: sunrealtype;

    /* Access the stepper memory structure */
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "arkStep_RelaxDeltaE",
            file!(),
            MSG_ARKSTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    /* Initialize output */
    *delta_e_out = ZERO;

    /* Set arrays for fused vector operation: built locally */

    let (stages, explicit, implicit, mass_type, msolve, nlscoef) = {
        let s = arkStep_mem_mut(ark_mem);
        (
            s.stages,
            s.explicit,
            s.implicit,
            s.mass_type,
            s.msolve,
            s.nlscoef,
        )
    };

    let use_local_dotprod = {
        let ops = J_relax.ops.borrow();
        ops.nvdotprodlocal.is_some() && ops.nvdotprodmultiallreduce.is_some()
    };

    for i in 0..stages as usize {
        if implicit || mass_type == MASS_FIXED {
            /* Use stored stages */
            z_stage = arkStep_mem_mut(ark_mem).z[i].clone();
        } else {
            /* Reconstruct explicit stages */
            let mut cvals: Vec<sunrealtype> = Vec::new();
            let mut Xvecs: Vec<N_Vector> = Vec::new();
            let mut nvec: i32 = 0;

            cvals.push(ONE);
            Xvecs.push(yn.clone());
            nvec += 1;

            for j in 0..i {
                let (aij, Fe_j) = {
                    let s = arkStep_mem_mut(ark_mem);
                    let a = s.Be.as_ref().expect("Be").borrow().A[i][j];
                    (a, s.Fe[j].clone())
                };
                cvals.push(h * aij);
                Xvecs.push(Fe_j);
                nvec += 1;
            }

            retval = N_VLinearCombination(nvec, &cvals, &Xvecs, &z_stage);
            if retval != 0 {
                return ARK_VECTOROP_ERR;
            }
        }

        /* Evaluate the Jacobian at z_i */
        let mut user_data = ark_mem.borrow_mut().user_data.take();
        retval = (relax_jac_fn.expect("relax_jac_fn"))(&z_stage, &J_relax, &mut user_data);
        ark_mem.borrow_mut().user_data = user_data;
        *num_relax_jac_evals += 1;
        if retval < 0 {
            return ARK_RELAX_JAC_FAIL;
        }
        if retval > 0 {
            return ARK_RELAX_JAC_RECV;
        }

        /* Reset temporary RHS alias */
        rhs_tmp = z_stage.clone();

        /* Compute delta_e = h * sum_i b_i * <relax_jac(z_i), f_i> */
        if explicit && implicit {
            let (be_i, bi_i, Fe_i, Fi_i) = {
                let s = arkStep_mem_mut(ark_mem);
                let be = s.Be.as_ref().expect("Be").borrow().b[i];
                let bi_ = s.Bi.as_ref().expect("Bi").borrow().b[i];
                (be, bi_, s.Fe[i].clone(), s.Fi[i].clone())
            };
            N_VLinearSum(be_i, &Fe_i, bi_i, &Fi_i, &rhs_tmp);
            bi = ONE;
        } else if explicit {
            let (be_i, Fe_i) = {
                let s = arkStep_mem_mut(ark_mem);
                let be = s.Be.as_ref().expect("Be").borrow().b[i];
                (be, s.Fe[i].clone())
            };
            if mass_type == MASS_FIXED {
                N_VScale(ONE, &Fe_i, &rhs_tmp);
            } else {
                rhs_tmp = Fe_i;
            }
            bi = be_i;
        } else {
            let (bi_i, Fi_i) = {
                let s = arkStep_mem_mut(ark_mem);
                let bi_ = s.Bi.as_ref().expect("Bi").borrow().b[i];
                (bi_, s.Fi[i].clone())
            };
            if mass_type == MASS_FIXED {
                N_VScale(ONE, &Fi_i, &rhs_tmp);
            } else {
                rhs_tmp = Fi_i;
            }
            bi = bi_i;
        }

        if mass_type == MASS_FIXED {
            retval = (msolve.expect("msolve"))(ark_mem, &rhs_tmp, nlscoef);
            if retval != 0 {
                return ARK_MASSSOLVE_FAIL;
            }
        }

        /* Update estimate of relaxation function change */
        if use_local_dotprod {
            *delta_e_out += bi * N_VDotProdLocal(&J_relax, &rhs_tmp);
        } else {
            *delta_e_out += bi * N_VDotProd(&J_relax, &rhs_tmp);
        }
    }

    if use_local_dotprod {
        retval = N_VDotProdMultiAllReduce(1, &J_relax, std::slice::from_mut(delta_e_out));
        if retval != 0 {
            return ARK_VECTOROP_ERR;
        }
    }

    *delta_e_out *= h;

    ARK_SUCCESS
}

/* -----------------------------------------------------------------------------
 * arkStep_GetOrder
 *
 * Returns the method order
 * ---------------------------------------------------------------------------*/
pub fn arkStep_GetOrder(ark_mem: &ARKodeMem) -> i32 {
    arkStep_mem_mut(ark_mem).q
}

/*===============================================================
EOF
===============================================================*/
