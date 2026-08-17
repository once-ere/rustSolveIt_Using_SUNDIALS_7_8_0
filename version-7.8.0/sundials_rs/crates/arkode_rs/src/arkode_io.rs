//! Port of `src/arkode/arkode_io.c` — the optional input and output
//! functions for the ARKODE infrastructure (shared by every stepper).
//!
//! Mapping notes (workspace-wide conventions, see `ARCHITECTURE.md`):
//! * C `void* arkode_mem` becomes `&ARKodeMem`, so every
//!   `if (arkode_mem == NULL)` guard is handled by the type system.
//! * `T* out` becomes `&mut T` in the same position with the same name;
//!   `N_Vector*` out-params become `&mut Option<N_Vector>`.
//! * `FILE*` becomes `&SUNFile`; `SUN_FORMAT_G` is `sun_format_g`.
//! * C `arkAccessHAdaptMem(arkode_mem, fname, &ark_mem, &hadapt_mem)`
//!   (defined in `arkode.c`) cannot hand out a pointer into the mem here
//!   because `hadapt_mem` is held BY VALUE, so it degenerates to the
//!   module-private presence check [`arkAccessHAdaptMem`] below; each use
//!   site then reads/writes `ark_mem.hadapt_mem` directly, exactly where C
//!   dereferenced its `hadapt_mem` local.
//! * Where C dereferences `ark_mem->hadapt_mem` with no NULL check (UB if
//!   absent), the port uses `.expect(...)` — accepted deviation class 5.
//! * `e_data`/`r_data`: C stores the `ark_mem` self-pointer for the
//!   built-in `arkEwtSetSS`/`arkRwtSet`, and the raw `user_data` pointer
//!   once the user supplies their own. A `Box` cannot alias, so
//!   `ARKodeSetDefaults` stores an `ARKodeMem` handle clone (the built-in
//!   functions downcast it back), while `ARKodeSetUserData` leaves the
//!   slot `None` — accepted deviation class 6: the `efun`/`rfun` invokers
//!   in `arkode.rs` pass the CURRENT `ark_mem.user_data` whenever
//!   `user_efun`/`user_rfun` is set.

use std::any::Any;

use sundials_core::sunadaptcontroller_imexgus::{
    SUNAdaptController_ImExGus, SUNAdaptController_SetParams_ImExGus,
};
use sundials_core::sunadaptcontroller_soderlind::{
    SUNAdaptController_ExpGus, SUNAdaptController_H0211, SUNAdaptController_H0321,
    SUNAdaptController_H211, SUNAdaptController_H312, SUNAdaptController_I,
    SUNAdaptController_ImpGus, SUNAdaptController_PI, SUNAdaptController_PID,
    SUNAdaptController_SetParams_ExpGus, SUNAdaptController_SetParams_I,
    SUNAdaptController_SetParams_ImpGus, SUNAdaptController_SetParams_PI,
    SUNAdaptController_SetParams_PID, SUNAdaptController_Soderlind,
};
use sundials_core::sundials_adaptcontroller::{
    SUNAdaptController, SUNAdaptController_Destroy, SUNAdaptController_Reset,
    SUNAdaptController_SetErrorBias, SUNAdaptController_Space, SUNAdaptController_Write,
};
use sundials_core::sundials_adjointcheckpointscheme::SUNAdjointCheckpointScheme;
use sundials_core::sundials_errors::SUN_SUCCESS;
use sundials_core::sundials_math::SUNRcopysign;
use sundials_core::sundials_nonlinearsolver::SUNNonlinearSolver;
use sundials_core::sundials_nvector::{N_VMaxNorm, N_VScale, N_Vector};
use sundials_core::sundials_types::*;
use sundials_core::sundials_utils::{sun_format_g, sunfprintf_long, sunfprintf_real, SUNFile};

use crate::arkode::{arkAllocVec, arkEwtSetSS, arkFreeVec, arkRwtSet};
use crate::arkode::{ARKodeSStolerances, ARKodeSVtolerances};
use crate::arkode_impl::*;
use crate::arkode_interp::{
    arkInterpCreate_Hermite, arkInterpCreate_Lagrange, arkInterpFree, arkInterpSetDegree,
};
use crate::arkode_relaxation::arkRelaxPrintAllStats;
use crate::arkode_user_controller::ARKUserControl;

/* Local numeric constant used only by ARKodeSetConstraints (C literal
   SUN_RCONST(2.5); ZERO/HALF/ONE come from arkode_impl). */
const TWOPT5: sunrealtype = 2.5;

/*---------------------------------------------------------------
  arkAccessHAdaptMem (arkode.c):

  Shortcut routine to unpack ark_mem and hadapt_mem structures from
  a void* pointer.  The `arkode_mem == NULL` branch is handled by the
  type system, and `hadapt_mem` lives BY VALUE inside `ark_mem`, so all
  that remains is the presence check; callers then touch
  `ark_mem.hadapt_mem` directly.  Module-private so it cannot collide
  with the `arkode.rs` definition.
  ---------------------------------------------------------------*/
fn arkAccessHAdaptMem(ark_mem: &ARKodeMem, _fname: &str) -> i32 {
    if ark_mem.borrow().hadapt_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "arkAccessHAdaptMem",
            file!(),
            MSG_ARKADAPT_NO_MEM,
        );
        return ARK_MEM_NULL;
    }
    ARK_SUCCESS
}

/*===============================================================
  ARKODE optional input functions
  ===============================================================*/

/*---------------------------------------------------------------
  ARKodeSetDefaults:

  Resets all optional inputs to ARKODE default values.  Does not
  change problem-defining function pointers fe and fi or
  user_data pointer.  Also leaves alone any data
  structures/options related to root-finding (those can be reset
  using ARKodeRootInit) or post-processing a step (ProcessStep).
  ---------------------------------------------------------------*/
pub fn ARKodeSetDefaults(arkode_mem: &ARKodeMem) -> i32 {
    /* NULL-mem check: handled by type system */

    /* C `e_data = ark_mem` / `r_data = ark_mem`: the built-in ewt/rwt
    functions receive the mem through their data token, so the token holds
    an ARKodeMem handle clone (the Rc cycle this creates is broken by
    ARKodeFree, exactly as CVODE's cv_e_data token is). */
    let e_token: Box<dyn Any> = Box::new(arkode_mem.clone());
    let r_token: Box<dyn Any> = Box::new(arkode_mem.clone());

    /* Set default values for integrator optional inputs */
    {
        let mut ark_mem = arkode_mem.borrow_mut();
        ark_mem.use_compensated_sums = SUNFALSE;
        ark_mem.fixedstep = SUNFALSE; /* default to use adaptive steps */
        ark_mem.reltol = 1.0e-4; /* relative tolerance */
        ark_mem.itol = ARK_SS; /* scalar-scalar solution tolerances */
        ark_mem.ritol = ARK_SS; /* scalar-scalar residual tolerances */
        ark_mem.Sabstol = 1.0e-9; /* solution absolute tolerance */
        ark_mem.atolmin0 = SUNFALSE; /* min(abstol) > 0 */
        ark_mem.SRabstol = 1.0e-9; /* residual absolute tolerance */
        ark_mem.Ratolmin0 = SUNFALSE; /* min(Rabstol) > 0 */
        ark_mem.user_efun = SUNFALSE; /* no user-supplied ewt function */
        ark_mem.efun = Some(arkEwtSetSS as ARKEwtFn); /* built-in scalar-scalar ewt function */
        ark_mem.e_data = Some(e_token); /* ewt function data */
        ark_mem.user_rfun = SUNFALSE; /* no user-supplied rwt function */
        ark_mem.rfun = Some(arkRwtSet as ARKRwtFn); /* built-in rwt function */
        ark_mem.r_data = Some(r_token); /* rwt function data */
        ark_mem.mxstep = MXSTEP_DEFAULT; /* max number of steps */
        ark_mem.mxhnil = MXHNIL; /* max warns of t+h==t */
        ark_mem.maxnef = MAXNEF; /* max error test fails */
        ark_mem.maxncf = MAXNCF; /* max convergence fails */
        ark_mem.maxconstrfails = MAXCONSTRFAILS; /* max number of constraint fails */
        ark_mem.preallocated = SUNFALSE; /* data was not preallocated */
        ark_mem.hin = ZERO; /* determine initial step on-the-fly */
        ark_mem.hmin = ZERO; /* no minimum step size */
        ark_mem.hmax_inv = ZERO; /* no maximum step size */
        ark_mem.tstopset = SUNFALSE; /* no stop time set */
        ark_mem.tstopinterp = SUNFALSE; /* copy at stop time */
        ark_mem.tstop = ZERO; /* no fixed stop time */
        let hadapt_mem = ark_mem.hadapt_mem.as_mut().expect("hadapt_mem allocated");
        hadapt_mem.etamx1 = ETAMX1; /* max change on first step */
        hadapt_mem.etamxf = ETAMXF; /* max change on error-failed step */
        hadapt_mem.etamin = ETAMIN; /* min bound on time step reduction */
        hadapt_mem.small_nef = SMALL_NEF; /* num error fails before ETAMXF enforced */
        hadapt_mem.etacf = ETACF; /* max change on convergence failure */
        hadapt_mem.cfl = CFLFAC; /* explicit stability factor */
        hadapt_mem.safety = SAFETY; /* step adaptivity safety factor  */
        hadapt_mem.growth = GROWTH; /* step adaptivity growth factor */
        hadapt_mem.lbound = HFIXED_LB; /* step adaptivity no-change lower bound */
        hadapt_mem.ubound = HFIXED_UB; /* step adaptivity no-change upper bound */
        hadapt_mem.expstab = None; /* no explicit stability fn */
        hadapt_mem.estab_data = None; /* no explicit stability fn data */
        hadapt_mem.pq = PQ; /* embedding order */
        hadapt_mem.p = 0; /* no default embedding order */
        hadapt_mem.q = 0; /* no default method order */
        hadapt_mem.adjust = ADJUST; /* controller order adjustment */
    }

    /* Set stepper defaults (if provided) */
    let step_setdefaults = arkode_mem.borrow().step_setdefaults;
    if let Some(step_setdefaults) = step_setdefaults {
        let retval = step_setdefaults(arkode_mem);
        if retval != ARK_SUCCESS {
            return retval;
        }
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeSetOrder:

  Specifies the method order
  ---------------------------------------------------------------*/
pub fn ARKodeSetOrder(arkode_mem: &ARKodeMem, ord: i32) -> i32 {
    /* NULL-mem check: handled by type system */

    /* Call stepper routine (if provided) */
    let step_setorder = arkode_mem.borrow().step_setorder;
    if let Some(step_setorder) = step_setorder {
        step_setorder(arkode_mem, ord)
    } else {
        arkProcessError(
            Some(arkode_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetOrder",
            file!(),
            "time-stepping module does not support this function",
        );
        ARK_STEPPER_UNSUPPORTED
    }
}

/*---------------------------------------------------------------
  ARKodeSetInterpolantType:

  Specifies use of the Lagrange or Hermite interpolation modules.
    itype == ARK_INTERP_HERMITE specifies the Hermite (nonstiff)
      interpolation module.
    itype == ARK_INTERP_LAGRANGE specifies the Lagrange (stiff)
      interpolation module.
    itype == ARK_INTERP_NONE disables interpolation.

  Return values:
     ARK_SUCCESS on success.
     ARK_MEM_NULL on NULL-valued arkode_mem input.
     ARK_MEM_FAIL if the interpolation module cannot be allocated.
     ARK_ILL_INPUT if the itype argument is not recognized.
  ---------------------------------------------------------------*/
pub fn ARKodeSetInterpolantType(arkode_mem: &ARKodeMem, itype: i32) -> i32 {
    /* NULL-mem check: handled by type system */

    /* check for legal itype input */
    if (itype != ARK_INTERP_HERMITE) && (itype != ARK_INTERP_LAGRANGE) && (itype != ARK_INTERP_NONE)
    {
        arkProcessError(
            Some(arkode_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "ARKodeSetInterpolantType",
            file!(),
            "Illegal interpolation type input.",
        );
        return ARK_ILL_INPUT;
    }

    /* do not change type once the module has been initialized */
    if arkode_mem.borrow().initialized {
        arkProcessError(
            Some(arkode_mem),
            ARK_INTERP_FAIL,
            line!() as i32,
            "ARKodeSetInterpolantType",
            file!(),
            "Type cannot be specified after module initialization.",
        );
        return ARK_ILL_INPUT;
    }

    /* delete any existing interpolation module */
    let old_interp = arkode_mem.borrow_mut().interp.take();
    if let Some(old_interp) = old_interp {
        arkInterpFree(arkode_mem, Some(&old_interp));
        /* `ark_mem->interp = NULL` already done by the take above */
    }

    /* create requested interpolation module, initially specifying
    the maximum possible interpolant degree. */
    if itype == ARK_INTERP_HERMITE {
        let interp_degree = arkode_mem.borrow().interp_degree;
        let interp = arkInterpCreate_Hermite(arkode_mem, interp_degree);
        let is_null = interp.is_none();
        arkode_mem.borrow_mut().interp = interp;
        if is_null {
            arkProcessError(
                Some(arkode_mem),
                ARK_MEM_FAIL,
                line!() as i32,
                "ARKodeSetInterpolantType",
                file!(),
                "Unable to allocate interpolation structure",
            );
            return ARK_MEM_FAIL;
        }
        arkode_mem.borrow_mut().interp_type = ARK_INTERP_HERMITE;
    } else if itype == ARK_INTERP_LAGRANGE {
        let interp_degree = arkode_mem.borrow().interp_degree;
        let interp = arkInterpCreate_Lagrange(arkode_mem, interp_degree);
        let is_null = interp.is_none();
        arkode_mem.borrow_mut().interp = interp;
        if is_null {
            /* upstream omits the `return ARK_MEM_FAIL` here (unlike the
            Hermite branch); preserved verbatim */
            arkProcessError(
                Some(arkode_mem),
                ARK_MEM_FAIL,
                line!() as i32,
                "ARKodeSetInterpolantType",
                file!(),
                "Unable to allocate interpolation structure",
            );
        }
        arkode_mem.borrow_mut().interp_type = ARK_INTERP_LAGRANGE;
    } else {
        let mut ark_mem = arkode_mem.borrow_mut();
        ark_mem.interp = None;
        ark_mem.interp_type = ARK_INTERP_NONE;
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeSetInterpolantDegree:

  Specifies the polynomial degree for the dense output
  interpolation module.

  Return values:
     ARK_SUCCESS on success.
     ARK_MEM_NULL on NULL-valued arkode_mem input or nonexistent
       interpolation module.
     ARK_INTERP_FAIL if the interpolation module is already
       initialized.
     ARK_ILL_INPUT if the degree is illegal.
  ---------------------------------------------------------------*/
pub fn ARKodeSetInterpolantDegree(arkode_mem: &ARKodeMem, degree: i32) -> i32 {
    /* NULL-mem check: handled by type system */

    /* do not change degree once the module has been initialized */
    if arkode_mem.borrow().initialized {
        arkProcessError(
            Some(arkode_mem),
            ARK_INTERP_FAIL,
            line!() as i32,
            "ARKodeSetInterpolantDegree",
            file!(),
            "Degree cannot be specified after module initialization.",
        );
        return ARK_ILL_INPUT;
    }

    if degree > ARK_INTERP_MAX_DEGREE {
        arkProcessError(
            Some(arkode_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "ARKodeSetInterpolantDegree",
            file!(),
            "Illegal degree specified.",
        );
        return ARK_ILL_INPUT;
    } else if degree < 0 {
        arkode_mem.borrow_mut().interp_degree = ARK_INTERP_MAX_DEGREE;
    } else {
        arkode_mem.borrow_mut().interp_degree = degree;
    }

    /* Set the degree now if possible otherwise it will be used when creating the
    interpolation module */
    let interp = arkode_mem.borrow().interp.clone();
    if let Some(interp) = interp {
        let interp_degree = arkode_mem.borrow().interp_degree;
        return arkInterpSetDegree(arkode_mem, Some(&interp), interp_degree);
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeSetNonlinearSolver:

  This routine attaches a SUNNonlinearSolver object to the
  time-stepping module.
  ---------------------------------------------------------------*/
pub fn ARKodeSetNonlinearSolver(arkode_mem: &ARKodeMem, NLS: &SUNNonlinearSolver) -> i32 {
    /* NULL-mem check: handled by type system */

    /* Guard against use for time steppers that do not need an algebraic solver */
    if !arkode_mem.borrow().step_supports_implicit {
        arkProcessError(
            Some(arkode_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetNonlinearSolver",
            file!(),
            "time-stepping module does not require an algebraic solver",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    /* Call stepper routine (if provided) */
    let step_setnonlinearsolver = arkode_mem.borrow().step_setnonlinearsolver;
    if let Some(step_setnonlinearsolver) = step_setnonlinearsolver {
        step_setnonlinearsolver(arkode_mem, NLS)
    } else {
        arkProcessError(
            Some(arkode_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetNonlinearSolver",
            file!(),
            "time-stepping module does not support this function",
        );
        ARK_STEPPER_UNSUPPORTED
    }
}

/*---------------------------------------------------------------
  ARKodeSetLinear:

  Specifies that the implicit portion of the problem is linear,
  and to tighten the linear solver tolerances while taking only
  one Newton iteration.  DO NOT USE IN COMBINATION WITH THE
  FIXED-POINT SOLVER.  Automatically tightens DeltaGammaMax
  to ensure that step size changes cause Jacobian recomputation.

  The argument should be 1 or 0, where 1 indicates that the
  Jacobian of fi with respect to y depends on time, and
  0 indicates that it is not time dependent.  Alternately, when
  using an iterative linear solver this flag denotes time
  dependence of the preconditioner.
  ---------------------------------------------------------------*/
pub fn ARKodeSetLinear(arkode_mem: &ARKodeMem, timedepend: i32) -> i32 {
    /* NULL-mem check: handled by type system */

    /* Guard against use for time steppers that do not need an algebraic solver */
    if !arkode_mem.borrow().step_supports_implicit {
        arkProcessError(
            Some(arkode_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetLinear",
            file!(),
            "time-stepping module does not require an algebraic solver",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    /* Call stepper routine (if provided) */
    let step_setlinear = arkode_mem.borrow().step_setlinear;
    if let Some(step_setlinear) = step_setlinear {
        step_setlinear(arkode_mem, timedepend)
    } else {
        arkProcessError(
            Some(arkode_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetLinear",
            file!(),
            "time-stepping module does not support this function",
        );
        ARK_STEPPER_UNSUPPORTED
    }
}

/*---------------------------------------------------------------
  ARKodeSetNonlinear:

  Specifies that the implicit portion of the problem is nonlinear.
  Used to undo a previous call to ARKodeSetLinear.
  ---------------------------------------------------------------*/
pub fn ARKodeSetNonlinear(arkode_mem: &ARKodeMem) -> i32 {
    /* NULL-mem check: handled by type system */

    /* Guard against use for time steppers that do not need an algebraic solver */
    if !arkode_mem.borrow().step_supports_implicit {
        arkProcessError(
            Some(arkode_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetNonlinear",
            file!(),
            "time-stepping module does not require an algebraic solver",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    /* Call stepper routine (if provided) */
    let step_setnonlinear = arkode_mem.borrow().step_setnonlinear;
    if let Some(step_setnonlinear) = step_setnonlinear {
        step_setnonlinear(arkode_mem)
    } else {
        arkProcessError(
            Some(arkode_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetNonlinear",
            file!(),
            "time-stepping module does not support this function",
        );
        ARK_STEPPER_UNSUPPORTED
    }
}

pub fn ARKodeSetAutonomous(arkode_mem: &ARKodeMem, autonomous: sunbooleantype) -> i32 {
    /* NULL-mem check: handled by type system */

    /* Guard against use for time steppers that do not need an algebraic solver */
    if !arkode_mem.borrow().step_supports_implicit {
        arkProcessError(
            Some(arkode_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetAutonomous",
            file!(),
            "time-stepping module does not require an algebraic solver",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    /* Call stepper routine (if provided) */
    let step_setautonomous = arkode_mem.borrow().step_setautonomous;
    if let Some(step_setautonomous) = step_setautonomous {
        step_setautonomous(arkode_mem, autonomous)
    } else {
        arkProcessError(
            Some(arkode_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetAutonomous",
            file!(),
            "time-stepping module does not support this function",
        );
        ARK_STEPPER_UNSUPPORTED
    }
}

/*---------------------------------------------------------------
  ARKodeSetNlsRhsFn:

  This routine sets an alternative user-supplied implicit ODE
  right-hand side function to use in the evaluation of nonlinear
  system functions.
  ---------------------------------------------------------------*/
pub fn ARKodeSetNlsRhsFn(arkode_mem: &ARKodeMem, nls_fi: Option<ARKRhsFn>) -> i32 {
    /* NULL-mem check: handled by type system */

    /* Guard against use for time steppers that do not need an algebraic solver */
    if !arkode_mem.borrow().step_supports_implicit {
        arkProcessError(
            Some(arkode_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetNlsRhsFn",
            file!(),
            "time-stepping module does not require an algebraic solver",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    /* Call stepper routine (if provided) */
    let step_setnlsrhsfn = arkode_mem.borrow().step_setnlsrhsfn;
    if let Some(step_setnlsrhsfn) = step_setnlsrhsfn {
        step_setnlsrhsfn(arkode_mem, nls_fi)
    } else {
        arkProcessError(
            Some(arkode_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetNlsRhsFn",
            file!(),
            "time-stepping module does not support this function",
        );
        ARK_STEPPER_UNSUPPORTED
    }
}

/*---------------------------------------------------------------
  ARKodeSetDeduceImplicitRhs:

  Specifies if an optimization is used to avoid an evaluation of
  fi after a nonlinear solve for an implicit stage.  If stage
  postprocessecing in enabled, this option is ignored, and the
  RHS is never deduced.

  An argument of SUNTRUE indicates that the RHS should be deduced,
  and SUNFALSE indicates that the RHS should be computed with
  an additional evaluation.
  ---------------------------------------------------------------*/
pub fn ARKodeSetDeduceImplicitRhs(arkode_mem: &ARKodeMem, deduce: sunbooleantype) -> i32 {
    /* NULL-mem check: handled by type system */

    /* Guard against use for time steppers that do not need an algebraic solver */
    if !arkode_mem.borrow().step_supports_implicit {
        arkProcessError(
            Some(arkode_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetDeduceImplicitRhs",
            file!(),
            "time-stepping module does not require an algebraic solver",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    /* Call stepper routine (if provided) */
    let step_setdeduceimplicitrhs = arkode_mem.borrow().step_setdeduceimplicitrhs;
    if let Some(step_setdeduceimplicitrhs) = step_setdeduceimplicitrhs {
        step_setdeduceimplicitrhs(arkode_mem, deduce)
    } else {
        arkProcessError(
            Some(arkode_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetDeduceImplicitRhs",
            file!(),
            "time-stepping module does not support this function",
        );
        ARK_STEPPER_UNSUPPORTED
    }
}

/*---------------------------------------------------------------
  ARKodeSetNonlinCRDown:

  Specifies the user-provided nonlinear convergence constant
  crdown.  Legal values are strictly positive; illegal values
  imply a reset to the default.
  ---------------------------------------------------------------*/
pub fn ARKodeSetNonlinCRDown(arkode_mem: &ARKodeMem, crdown: sunrealtype) -> i32 {
    /* NULL-mem check: handled by type system */

    /* Guard against use for time steppers that do not need an algebraic solver */
    if !arkode_mem.borrow().step_supports_implicit {
        arkProcessError(
            Some(arkode_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetNonlinCRDown",
            file!(),
            "time-stepping module does not require an algebraic solver",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    /* Call stepper routine (if provided) */
    let step_setnonlincrdown = arkode_mem.borrow().step_setnonlincrdown;
    if let Some(step_setnonlincrdown) = step_setnonlincrdown {
        step_setnonlincrdown(arkode_mem, crdown)
    } else {
        arkProcessError(
            Some(arkode_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetNonlinCRDown",
            file!(),
            "time-stepping module does not support this function",
        );
        ARK_STEPPER_UNSUPPORTED
    }
}

/*---------------------------------------------------------------
  ARKodeSetNonlinRDiv:

  Specifies the user-provided nonlinear convergence constant
  rdiv.  Legal values are strictly positive; illegal values
  imply a reset to the default.
  ---------------------------------------------------------------*/
pub fn ARKodeSetNonlinRDiv(arkode_mem: &ARKodeMem, rdiv: sunrealtype) -> i32 {
    /* NULL-mem check: handled by type system */

    /* Guard against use for time steppers that do not need an algebraic solver */
    if !arkode_mem.borrow().step_supports_implicit {
        arkProcessError(
            Some(arkode_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetNonlinRDiv",
            file!(),
            "time-stepping module does not require an algebraic solver",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    /* Call stepper routine (if provided) */
    let step_setnonlinrdiv = arkode_mem.borrow().step_setnonlinrdiv;
    if let Some(step_setnonlinrdiv) = step_setnonlinrdiv {
        step_setnonlinrdiv(arkode_mem, rdiv)
    } else {
        arkProcessError(
            Some(arkode_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetNonlinRDiv",
            file!(),
            "time-stepping module does not support this function",
        );
        ARK_STEPPER_UNSUPPORTED
    }
}

/*---------------------------------------------------------------
  ARKodeSetDeltaGammaMax:

  Specifies the user-provided linear setup decision constant
  dgmax.  Legal values are strictly positive; illegal values imply
  a reset to the default.
  ---------------------------------------------------------------*/
pub fn ARKodeSetDeltaGammaMax(arkode_mem: &ARKodeMem, dgmax: sunrealtype) -> i32 {
    /* NULL-mem check: handled by type system */

    /* Guard against use for time steppers that do not need an algebraic solver */
    if !arkode_mem.borrow().step_supports_implicit {
        arkProcessError(
            Some(arkode_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetDeltaGammaMax",
            file!(),
            "time-stepping module does not require an algebraic solver",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    /* Call stepper routine (if provided) */
    let step_setdeltagammamax = arkode_mem.borrow().step_setdeltagammamax;
    if let Some(step_setdeltagammamax) = step_setdeltagammamax {
        step_setdeltagammamax(arkode_mem, dgmax)
    } else {
        arkProcessError(
            Some(arkode_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetDeltaGammaMax",
            file!(),
            "time-stepping module does not support this function",
        );
        ARK_STEPPER_UNSUPPORTED
    }
}

/*---------------------------------------------------------------
  ARKodeSetLSetupFrequency:

  Specifies the user-provided linear setup decision constant
  msbp.  Positive values give the frequency for calling lsetup;
  negative values imply recomputation of lsetup at each nonlinear
  solve; a zero value implies a reset to the default.
  ---------------------------------------------------------------*/
pub fn ARKodeSetLSetupFrequency(arkode_mem: &ARKodeMem, msbp: i32) -> i32 {
    /* NULL-mem check: handled by type system */

    /* Guard against use for time steppers that do not need an algebraic solver */
    if !arkode_mem.borrow().step_supports_implicit {
        arkProcessError(
            Some(arkode_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetLSetupFrequency",
            file!(),
            "time-stepping module does not require an algebraic solver",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    /* Call stepper routine (if provided) */
    let step_setlsetupfrequency = arkode_mem.borrow().step_setlsetupfrequency;
    if let Some(step_setlsetupfrequency) = step_setlsetupfrequency {
        step_setlsetupfrequency(arkode_mem, msbp)
    } else {
        arkProcessError(
            Some(arkode_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetLSetupFrequency",
            file!(),
            "time-stepping module does not support this function",
        );
        ARK_STEPPER_UNSUPPORTED
    }
}

/*---------------------------------------------------------------
  ARKodeSetPredictorMethod:

  Specifies the method to use for predicting implicit solutions.
  ---------------------------------------------------------------*/
pub fn ARKodeSetPredictorMethod(arkode_mem: &ARKodeMem, pred_method: i32) -> i32 {
    /* NULL-mem check: handled by type system */

    /* Guard against use for time steppers that do not need an algebraic solver */
    if !arkode_mem.borrow().step_supports_implicit {
        arkProcessError(
            Some(arkode_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetPredictorMethod",
            file!(),
            "time-stepping module does not require an algebraic solver",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    /* Higher-order predictors require interpolation */
    if arkode_mem.borrow().interp_type == ARK_INTERP_NONE && pred_method != 0 {
        arkProcessError(
            Some(arkode_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "ARKodeSetPredictorMethod",
            file!(),
            "Non-trival predictors require an interpolation module",
        );
        return ARK_ILL_INPUT;
    }

    /* Call stepper routine (if provided) */
    let step_setpredictormethod = arkode_mem.borrow().step_setpredictormethod;
    if let Some(step_setpredictormethod) = step_setpredictormethod {
        step_setpredictormethod(arkode_mem, pred_method)
    } else {
        arkProcessError(
            Some(arkode_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetPredictorMethod",
            file!(),
            "time-stepping module does not support this function",
        );
        ARK_STEPPER_UNSUPPORTED
    }
}

/*---------------------------------------------------------------
  ARKodeSetMaxNonlinIters:

  Specifies the maximum number of nonlinear iterations during
  one solve.  A non-positive input implies a reset to the
  default value.
  ---------------------------------------------------------------*/
pub fn ARKodeSetMaxNonlinIters(arkode_mem: &ARKodeMem, maxcor: i32) -> i32 {
    /* NULL-mem check: handled by type system */

    /* Guard against use for time steppers that do not need an algebraic solver */
    if !arkode_mem.borrow().step_supports_implicit {
        arkProcessError(
            Some(arkode_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetMaxNonlinIters",
            file!(),
            "time-stepping module does not require an algebraic solver",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    /* Call stepper routine (if provided) */
    let step_setmaxnonliniters = arkode_mem.borrow().step_setmaxnonliniters;
    if let Some(step_setmaxnonliniters) = step_setmaxnonliniters {
        step_setmaxnonliniters(arkode_mem, maxcor)
    } else {
        arkProcessError(
            Some(arkode_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetMaxNonlinIters",
            file!(),
            "time-stepping module does not support this function",
        );
        ARK_STEPPER_UNSUPPORTED
    }
}

/*---------------------------------------------------------------
  ARKodeSetNonlinConvCoef:

  Specifies the coefficient in the nonlinear solver convergence
  test.  A non-positive input implies a reset to the default value.
  ---------------------------------------------------------------*/
pub fn ARKodeSetNonlinConvCoef(arkode_mem: &ARKodeMem, nlscoef: sunrealtype) -> i32 {
    /* NULL-mem check: handled by type system */

    /* Guard against use for time steppers that do not need an algebraic solver */
    if !arkode_mem.borrow().step_supports_implicit {
        arkProcessError(
            Some(arkode_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetNonlinConvCoef",
            file!(),
            "time-stepping module does not require an algebraic solver",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    /* Call stepper routine (if provided) */
    let step_setnonlinconvcoef = arkode_mem.borrow().step_setnonlinconvcoef;
    if let Some(step_setnonlinconvcoef) = step_setnonlinconvcoef {
        step_setnonlinconvcoef(arkode_mem, nlscoef)
    } else {
        arkProcessError(
            Some(arkode_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetNonlinConvCoef",
            file!(),
            "time-stepping module does not support this function",
        );
        ARK_STEPPER_UNSUPPORTED
    }
}

/*---------------------------------------------------------------
  ARKodeSetStagePredictFn:  Specifies a user-provided step
  predictor function having type ARKStagePredictFn.  A
  NULL input function disables calls to this routine.
  ---------------------------------------------------------------*/
pub fn ARKodeSetStagePredictFn(
    arkode_mem: &ARKodeMem,
    PredictStage: Option<ARKStagePredictFn>,
) -> i32 {
    /* NULL-mem check: handled by type system */

    /* Guard against use for time steppers that do not need an algebraic solver */
    if !arkode_mem.borrow().step_supports_implicit {
        arkProcessError(
            Some(arkode_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetStagePredictFn",
            file!(),
            "time-stepping module does not require an algebraic solver",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    /* Call stepper routine (if provided) */
    let step_setstagepredictfn = arkode_mem.borrow().step_setstagepredictfn;
    if let Some(step_setstagepredictfn) = step_setstagepredictfn {
        step_setstagepredictfn(arkode_mem, PredictStage)
    } else {
        arkProcessError(
            Some(arkode_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetStagePredictFn",
            file!(),
            "time-stepping module does not support this function",
        );
        ARK_STEPPER_UNSUPPORTED
    }
}

/*---------------------------------------------------------------
  ARKodeSetUserData:

  Specifies the user data pointer for f
  ---------------------------------------------------------------*/
pub fn ARKodeSetUserData(arkode_mem: &ARKodeMem, user_data: Option<Box<dyn Any>>) -> i32 {
    /* NULL-mem check: handled by type system */
    {
        let mut ark_mem = arkode_mem.borrow_mut();
        ark_mem.user_data = user_data;

        /* Set data for efun.
        C: `e_data = user_data` (pointer alias). A Box cannot alias, so the
        slot is cleared and the efun invoker passes `user_data` whenever
        `user_efun` is set -- accepted deviation class 6. */
        if ark_mem.user_efun {
            ark_mem.e_data = None;
        }

        /* Set data for rfun (same treatment as e_data above) */
        if ark_mem.user_rfun {
            ark_mem.r_data = None;
        }

        /* Set data for root finding.
        C: `root_mem->root_data = user_data`; per the frozen contract
        `root_data` STAYS None and the arkRootCheck / arkRootfind routines
        pass the CURRENT `ark_mem.user_data` to gfun. */
        if let Some(root_mem) = ark_mem.root_mem.as_mut() {
            root_mem.root_data = None;
        }
    }

    /* Set user data into stepper (if provided) */
    let step_setuserdata = arkode_mem.borrow().step_setuserdata;
    if let Some(step_setuserdata) = step_setuserdata {
        /* hand the hook the very same box (never a clone), restoring it on
        every path */
        let mut user_data = arkode_mem.borrow_mut().user_data.take();
        let retval = step_setuserdata(arkode_mem, &mut user_data);
        arkode_mem.borrow_mut().user_data = user_data;
        return retval;
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeSetAdaptController:

  Specifies a non-default SUNAdaptController time step controller
  object. If a NULL-valued SUNAdaptController is input, the
  default will be re-enabled.
  ---------------------------------------------------------------*/
pub fn ARKodeSetAdaptController(arkode_mem: &ARKodeMem, C: Option<&SUNAdaptController>) -> i32 {
    /* NULL-mem check: handled by type system */

    /* Guard against use for non-adaptive time stepper modules */
    if !arkode_mem.borrow().step_supports_adaptive {
        arkProcessError(
            Some(arkode_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetAdaptController",
            file!(),
            "time-stepping module does not support temporal adaptivity",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    /* If the stepper has provided a custom function, then call it and return */
    let step_setadaptcontroller = arkode_mem.borrow().step_setadaptcontroller;
    if let Some(step_setadaptcontroller) = step_setadaptcontroller {
        return step_setadaptcontroller(arkode_mem, C);
    }

    /* Otherwise, call a utility routine to replace the current controller object */
    arkReplaceAdaptController(arkode_mem, C, SUNFALSE)
}

/*---------------------------------------------------------------
  ARKodeSetAdaptControllerByName:

  Specifies a SUNAdaptController time step controller object by
  its name.
  ---------------------------------------------------------------*/
pub fn ARKodeSetAdaptControllerByName(arkode_mem: &ARKodeMem, cname: &str) -> i32 {
    /* NULL-mem check: handled by type system */
    let sunctx = arkode_mem.borrow().sunctx.clone();

    /* Create new controller based on the name */
    let C: Option<SUNAdaptController> = if cname == "Soderlind" {
        SUNAdaptController_Soderlind(&sunctx)
    } else if cname == "PID" {
        SUNAdaptController_PID(&sunctx)
    } else if cname == "PI" {
        SUNAdaptController_PI(&sunctx)
    } else if cname == "I" {
        SUNAdaptController_I(&sunctx)
    } else if cname == "ExpGus" {
        SUNAdaptController_ExpGus(&sunctx)
    } else if cname == "ImpGus" {
        SUNAdaptController_ImpGus(&sunctx)
    } else if cname == "ImExGus" {
        SUNAdaptController_ImExGus(&sunctx)
    } else if cname == "H0211" {
        SUNAdaptController_H0211(&sunctx)
    } else if cname == "H0321" {
        SUNAdaptController_H0321(&sunctx)
    } else if cname == "H211" {
        SUNAdaptController_H211(&sunctx)
    } else if cname == "H312" {
        SUNAdaptController_H312(&sunctx)
    } else {
        arkProcessError(
            None,
            ARK_ILL_INPUT,
            line!() as i32,
            "ARKodeSetAdaptControllerByName",
            file!(),
            "Unknown controller",
        );
        return ARK_ILL_INPUT;
    };

    let C = match C {
        Some(C) => C,
        None => {
            arkProcessError(
                Some(arkode_mem),
                ARK_MEM_FAIL,
                line!() as i32,
                "ARKodeSetAdaptControllerByName",
                file!(),
                "SUNAdaptController allocation failure",
            );
            return ARK_MEM_FAIL;
        }
    };

    /* Send controller to be used by ARKODE */
    let retval = ARKodeSetAdaptController(arkode_mem, Some(&C));
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* Update controller ownership flag */
    arkode_mem
        .borrow_mut()
        .hadapt_mem
        .as_mut()
        .expect("hadapt_mem allocated")
        .owncontroller = SUNTRUE;

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeSetMaxNumSteps:

  Specifies the maximum number of integration steps
  ---------------------------------------------------------------*/
pub fn ARKodeSetMaxNumSteps(arkode_mem: &ARKodeMem, mxsteps: i64) -> i32 {
    /* NULL-mem check: handled by type system */

    /* Passing mxsteps=0 sets the default. Passing mxsteps<0 disables the test. */
    if mxsteps == 0 {
        arkode_mem.borrow_mut().mxstep = MXSTEP_DEFAULT;
    } else {
        arkode_mem.borrow_mut().mxstep = mxsteps;
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeSetMaxHnilWarns:

  Specifies the maximum number of warnings for small h
  ---------------------------------------------------------------*/
pub fn ARKodeSetMaxHnilWarns(arkode_mem: &ARKodeMem, mxhnil: i32) -> i32 {
    /* NULL-mem check: handled by type system */

    /* Guard against use for non-adaptive time stepper modules */
    if !arkode_mem.borrow().step_supports_adaptive {
        arkProcessError(
            Some(arkode_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetMaxHnilWarns",
            file!(),
            "time-stepping module does not support temporal adaptivity",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    /* Passing mxhnil=0 sets the default, otherwise use input. */
    if mxhnil == 0 {
        arkode_mem.borrow_mut().mxhnil = 10;
    } else {
        arkode_mem.borrow_mut().mxhnil = mxhnil;
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeSetInitStep:

  Specifies the initial step size
  ---------------------------------------------------------------*/
pub fn ARKodeSetInitStep(arkode_mem: &ARKodeMem, hin: sunrealtype) -> i32 {
    /* NULL-mem check: handled by type system */

    /* Guard against hin==0 for non-adaptive time stepper modules */
    if (!arkode_mem.borrow().step_supports_adaptive) && (hin == ZERO) {
        arkProcessError(
            Some(arkode_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetInitStep",
            file!(),
            "time-stepping module does not support temporal adaptivity",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    {
        let mut ark_mem = arkode_mem.borrow_mut();

        /* Passing hin=0 sets the default, otherwise use input. */
        if hin == ZERO {
            ark_mem.hin = ZERO;
        } else {
            ark_mem.hin = hin;
        }

        /* Clear previous initial step */
        ark_mem.h0u = ZERO;
    }

    /* Reset error controller (e.g., error and step size history) */
    let hcontroller = arkode_mem
        .borrow()
        .hadapt_mem
        .as_ref()
        .expect("hadapt_mem allocated")
        .hcontroller
        .clone();
    if let Some(hcontroller) = hcontroller {
        let retval = SUNAdaptController_Reset(&hcontroller);
        if retval != SUN_SUCCESS {
            return ARK_CONTROLLER_ERR;
        }
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeSetMinStep:

  Specifies the minimum step size
  ---------------------------------------------------------------*/
pub fn ARKodeSetMinStep(arkode_mem: &ARKodeMem, hmin: sunrealtype) -> i32 {
    /* NULL-mem check: handled by type system */

    /* Guard against use for non-adaptive time stepper modules */
    if !arkode_mem.borrow().step_supports_adaptive {
        arkProcessError(
            Some(arkode_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetMinStep",
            file!(),
            "time-stepping module does not support temporal adaptivity",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    /* Passing a value <= 0 sets hmin = 0 */
    if hmin <= ZERO {
        arkode_mem.borrow_mut().hmin = ZERO;
        return ARK_SUCCESS;
    }

    /* check that hmin and hmax are agreeable */
    if hmin * arkode_mem.borrow().hmax_inv > ONE {
        arkProcessError(
            Some(arkode_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "ARKodeSetMinStep",
            file!(),
            MSG_ARK_BAD_HMIN_HMAX,
        );
        return ARK_ILL_INPUT;
    }

    /* set the value */
    arkode_mem.borrow_mut().hmin = hmin;

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeSetMaxStep:

  Specifies the maximum step size
  ---------------------------------------------------------------*/
pub fn ARKodeSetMaxStep(arkode_mem: &ARKodeMem, hmax: sunrealtype) -> i32 {
    /* NULL-mem check: handled by type system */

    /* Guard against use for non-adaptive time stepper modules */
    if !arkode_mem.borrow().step_supports_adaptive {
        arkProcessError(
            Some(arkode_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetMaxStep",
            file!(),
            "time-stepping module does not support temporal adaptivity",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    /* Passing a value <= 0 sets hmax = infinity */
    if hmax <= ZERO {
        arkode_mem.borrow_mut().hmax_inv = ZERO;
        return ARK_SUCCESS;
    }

    /* check that hmax and hmin are agreeable */
    let hmax_inv = ONE / hmax;
    if hmax_inv * arkode_mem.borrow().hmin > ONE {
        arkProcessError(
            Some(arkode_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "ARKodeSetMaxStep",
            file!(),
            MSG_ARK_BAD_HMIN_HMAX,
        );
        return ARK_ILL_INPUT;
    }

    /* set the value */
    arkode_mem.borrow_mut().hmax_inv = hmax_inv;

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeSetStopTime:

  Specifies the time beyond which the integration is not to proceed.
  ---------------------------------------------------------------*/
pub fn ARKodeSetStopTime(arkode_mem: &ARKodeMem, tstop: sunrealtype) -> i32 {
    /* NULL-mem check: handled by type system */

    /* If ARKODE was called at least once, test if tstop is legal
    (i.e. if it was not already passed).
    If ARKodeSetStopTime is called before the first call to ARKODE,
    tstop will be checked in ARKODE. */
    let (nst, tcur, h) = {
        let ark_mem = arkode_mem.borrow();
        (ark_mem.nst, ark_mem.tcur, ark_mem.h)
    };
    if nst > 0 && (tstop - tcur) * h < ZERO {
        arkProcessError(
            Some(arkode_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "ARKodeSetStopTime",
            file!(),
            &MSG_ARK_BAD_TSTOP(tstop, tcur),
        );
        return ARK_ILL_INPUT;
    }

    {
        let mut ark_mem = arkode_mem.borrow_mut();
        ark_mem.tstop = tstop;
        ark_mem.tstopset = SUNTRUE;
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeSetInterpolateStopTime:

  Specifies to use interpolation to fill the solution output at
  the stop time (instead of a copy).
  ---------------------------------------------------------------*/
pub fn ARKodeSetInterpolateStopTime(arkode_mem: &ARKodeMem, interp: sunbooleantype) -> i32 {
    /* NULL-mem check: handled by type system */
    arkode_mem.borrow_mut().tstopinterp = interp;
    ARK_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeClearStopTime:

  Disable the stop time.
  ---------------------------------------------------------------*/
pub fn ARKodeClearStopTime(arkode_mem: &ARKodeMem) -> i32 {
    /* NULL-mem check: handled by type system */

    arkode_mem.borrow_mut().tstopset = SUNFALSE;

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeSetFixedStep:

  Specifies to use a fixed time step size instead of performing
  any form of temporal adaptivity.  ARKODE will use this step size
  for all steps (unless tstop is set, in which case it may need to
  modify that last step approaching tstop.  If any solver failure
  occurs in the timestepping module, ARKODE will typically
  immediately return with an error message indicating that the
  selected step size cannot be used.

  Any nonzero argument will result in the use of that fixed step
  size; an argument of 0 will re-enable temporal adaptivity.
  ---------------------------------------------------------------*/
pub fn ARKodeSetFixedStep(arkode_mem: &ARKodeMem, hfixed: sunrealtype) -> i32 {
    /* NULL-mem check: handled by type system */

    /* ensure that when hfixed=0, the time step module supports adaptivity */
    if (hfixed == ZERO) && (!arkode_mem.borrow().step_supports_adaptive) {
        arkProcessError(
            Some(arkode_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetFixedStep",
            file!(),
            "temporal adaptivity is not supported by this time step module",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    /* re-attach internal error weight functions if necessary */
    if (hfixed == ZERO) && (!arkode_mem.borrow().user_efun) {
        let (itol, Vabstol, reltol, Sabstol) = {
            let ark_mem = arkode_mem.borrow();
            (
                ark_mem.itol,
                ark_mem.Vabstol.clone(),
                ark_mem.reltol,
                ark_mem.Sabstol,
            )
        };
        let retval = if itol == ARK_SV && Vabstol.is_some() {
            ARKodeSVtolerances(arkode_mem, reltol, Vabstol.as_ref().expect("Vabstol"))
        } else {
            ARKodeSStolerances(arkode_mem, reltol, Sabstol)
        };
        if retval != ARK_SUCCESS {
            return retval;
        }
    }

    /* set ark_mem "fixedstep" entry */
    {
        let mut ark_mem = arkode_mem.borrow_mut();
        if hfixed != ZERO {
            ark_mem.fixedstep = SUNTRUE;
            ark_mem.hin = hfixed;
        } else {
            ark_mem.fixedstep = SUNFALSE;
        }
    }

    /* Notify ARKODE to use hfixed as the initial step size, and return */
    ARKodeSetInitStep(arkode_mem, hfixed)
}

/*---------------------------------------------------------------
  ARKodeSetStepDirection:

  Specifies the direction of integration (forward or backward)
  based on the sign of stepdir. If 0, the direction will remain
  unchanged. Note that if a fixed step size was previously set,
  this function can change the sign of that.

  This should only be called after ARKodeReset, or between
  creating a stepper and ARKodeEvolve.
  ---------------------------------------------------------------*/
pub fn ARKodeSetStepDirection(arkode_mem: &ARKodeMem, stepdir: sunrealtype) -> i32 {
    /* stepdir is a sunrealtype because the direction typically comes from a time
     * step h or tend-tstart which are sunrealtypes. If stepdir was in int,
     * conversions would be required which can cause undefined behavior when
     * greater than MAX_INT */
    /* NULL-mem check: handled by type system */
    let mut h: sunrealtype = ZERO;

    /* do not change direction once the module has been initialized i.e., after calling
    ARKodeEvolve unless ReInit or Reset are called. */
    if !arkode_mem.borrow().initsetup {
        arkProcessError(
            Some(arkode_mem),
            ARK_STEP_DIRECTION_ERR,
            line!() as i32,
            "ARKodeSetStepDirection",
            file!(),
            "Step direction cannot be specified after module initialization.",
        );
        return ARK_STEP_DIRECTION_ERR;
    }

    if stepdir != ZERO {
        let retval = ARKodeGetStepDirection(arkode_mem, &mut h);
        if retval != ARK_SUCCESS {
            arkProcessError(
                Some(arkode_mem),
                retval,
                line!() as i32,
                "ARKodeSetStepDirection",
                file!(),
                "Unable to access step direction",
            );
            return retval;
        }

        if h != SUNRcopysign(h, stepdir) {
            {
                let mut ark_mem = arkode_mem.borrow_mut();
                /* Reverse the sign of h. If adaptive, h will be overwritten anyway by the
                 * initial step estimation since ARKodeReset must be called before this.
                 * However, the sign of h will be used to check if the integration
                 * direction and stop time are consistent, e.g., in ARKodeSetStopTime, so
                 * we should not set h = 0. */
                ark_mem.h = -h;
                /* Clear previous initial step and force an initial step recomputation.
                 * Normally, this would not occur after a reset, but it is necessary here
                 * because the timestep used in one direction may not be suitable for the
                 * other */
                ark_mem.h0u = ZERO;
                /* Reverse the step if in fixed mode. If adaptive, reset to 0 to clear any
                 * old value from a call to ARKodeSetInit */
                ark_mem.hin = if ark_mem.fixedstep { -h } else { ZERO };
            }

            /* Reset error controller (e.g., error and step size history) */
            let hcontroller = arkode_mem
                .borrow()
                .hadapt_mem
                .as_ref()
                .and_then(|hadapt_mem| hadapt_mem.hcontroller.clone());
            if let Some(hcontroller) = hcontroller {
                let err = SUNAdaptController_Reset(&hcontroller);
                if err != SUN_SUCCESS {
                    arkProcessError(
                        Some(arkode_mem),
                        ARK_CONTROLLER_ERR,
                        line!() as i32,
                        "ARKodeSetStepDirection",
                        file!(),
                        "Unable to reset error controller object",
                    );
                    return ARK_CONTROLLER_ERR;
                }
            }
        }
    }

    let step_setstepdirection = arkode_mem.borrow().step_setstepdirection;
    if let Some(step_setstepdirection) = step_setstepdirection {
        return step_setstepdirection(arkode_mem, stepdir);
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeSetRootDirection:

  Specifies the direction of zero-crossings to be monitored.
  The default is to monitor both crossings.
  ---------------------------------------------------------------*/
pub fn ARKodeSetRootDirection(arkode_mem: &ARKodeMem, rootdir: &[i32]) -> i32 {
    /* NULL-mem check: handled by type system */
    if arkode_mem.borrow().root_mem.is_none() {
        arkProcessError(
            Some(arkode_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "ARKodeSetRootDirection",
            file!(),
            MSG_ARK_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    let nrtfn = arkode_mem
        .borrow()
        .root_mem
        .as_ref()
        .expect("root_mem")
        .nrtfn;
    if nrtfn == 0 {
        arkProcessError(
            Some(arkode_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "ARKodeSetRootDirection",
            file!(),
            MSG_ARK_NO_ROOT,
        );
        return ARK_ILL_INPUT;
    }
    {
        let mut ark_mem = arkode_mem.borrow_mut();
        let ark_root_mem = ark_mem.root_mem.as_mut().expect("root_mem");
        for i in 0..nrtfn as usize {
            ark_root_mem.rootdir[i] = rootdir[i];
        }
    }
    ARK_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeSetNoInactiveRootWarn:

  Disables issuing a warning if some root function appears
  to be identically zero at the beginning of the integration
  ---------------------------------------------------------------*/
pub fn ARKodeSetNoInactiveRootWarn(arkode_mem: &ARKodeMem) -> i32 {
    /* NULL-mem check: handled by type system */
    if arkode_mem.borrow().root_mem.is_none() {
        arkProcessError(
            Some(arkode_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "ARKodeSetNoInactiveRootWarn",
            file!(),
            MSG_ARK_NO_ROOT,
        );
        return ARK_MEM_NULL;
    }
    arkode_mem
        .borrow_mut()
        .root_mem
        .as_mut()
        .expect("root_mem")
        .mxgnull = 0;
    ARK_SUCCESS
}

/*------------------------------------------------------------------------------
  ARKodeSetPreStepFn:
  ARKodeSetPostStepFn:

  Specifies user-provided step pre- and post-step functions.

  The pre-step function is called just prior to taking a step and the post-step
  function is called immediately after completing a successful step.

  IF THE SUPPLIED FUNCTION MODIFIES ANY OF THE ACTIVE STATE DATA, THEN ALL
  THEORETICAL GUARANTEES OF SOLUTION ACCURACY AND STABILITY ARE LOST.
  ----------------------------------------------------------------------------*/
pub fn ARKodeSetPreStepFn(arkode_mem: &ARKodeMem, prestep_fn: Option<ARKPreStepFn>) -> i32 {
    /* NULL-mem check: handled by type system */

    /* NULL argument disables the pre-step function */
    arkode_mem.borrow_mut().PreStepFn = prestep_fn;

    ARK_SUCCESS
}

pub fn ARKodeSetPostStepFn(arkode_mem: &ARKodeMem, poststep_fn: Option<ARKPostStepFn>) -> i32 {
    /* NULL-mem check: handled by type system */

    /* NULL argument disables the post-step function */
    arkode_mem.borrow_mut().PostStepFn = poststep_fn;

    ARK_SUCCESS
}

/*------------------------------------------------------------------------------
  ARKodeSetPreRhsFn:

  Specifies user-provided pre-RHS function.

  The pre-RHS function is called on a state vector just prior to computing the
  RHS. For problems with partitioned RHS functions that are called with
  identical inputs, this is only called before the first RHS evaluation.

  IF THE SUPPLIED FUNCTION MODIFIES ANY OF THE ACTIVE STATE DATA, THEN ALL
  THEORETICAL GUARANTEES OF SOLUTION ACCURACY AND STABILITY ARE LOST.
  ---------------------------------------------------------------*/
pub fn ARKodeSetPreRhsFn(arkode_mem: &ARKodeMem, prerhs_fn: Option<ARKPreRhsFn>) -> i32 {
    /* NULL-mem check: handled by type system */

    /* NULL argument disables the pre-RHS function */
    arkode_mem.borrow_mut().PreRhsFn = prerhs_fn;

    ARK_SUCCESS
}

/*------------------------------------------------------------------------------
  ARKodeSetPostprocessStepFn:
  ARKodeSetPostprocessStageFn:

  Specifies user-provided step and stage post-processing functions.

  These functions are called immediately after computing a stage or the new step
  solution (before error checks or other post-step actions e.g., constraint
  handling or relaxation).

  IF THE SUPPLIED FUNCTION MODIFIES ANY OF THE ACTIVE STATE DATA, THEN ALL
  THEORETICAL GUARANTEES OF SOLUTION ACCURACY AND STABILITY ARE LOST.

  While it is possible to perform stage postprocessing when
  ARKodeSetDeduceImplicitRhs is enabled, this can cause implicit RHS evaluations
  to be inconsistent with the postprocessed values (this similarly applies when
  using step post processing with FSAL methods). It is strongly recommended to
  disable ARKodeSetDeduceImplicitRhs in order to guarantee postprocessing
  constraints are enforced.
  ----------------------------------------------------------------------------*/
pub fn ARKodeSetPostprocessStepFn(
    arkode_mem: &ARKodeMem,
    ProcessStep: Option<ARKPostProcessFn>,
) -> i32 {
    /* NULL-mem check: handled by type system */

    /* NULL argument disables the postprocessing function */
    arkode_mem.borrow_mut().PostProcessStepFn = ProcessStep;

    ARK_SUCCESS
}

pub fn ARKodeSetPostprocessStageFn(
    arkode_mem: &ARKodeMem,
    ProcessStage: Option<ARKPostProcessFn>,
) -> i32 {
    /* NULL-mem check: handled by type system */

    /* NULL argument disables the postprocessing function */
    arkode_mem.borrow_mut().PostProcessStageFn = ProcessStage;

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeSetConstraints:

  Activates or Deactivates inequality constraint checking.
  ---------------------------------------------------------------*/
pub fn ARKodeSetConstraints(arkode_mem: &ARKodeMem, constraints: Option<&N_Vector>) -> i32 {
    /* NULL-mem check: handled by type system */

    /* Guard against use for non-adaptive time stepper modules */
    if !arkode_mem.borrow().step_supports_adaptive && constraints.is_some() {
        arkProcessError(
            Some(arkode_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetConstraints",
            file!(),
            "time-stepping module does not support temporal adaptivity",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    /* If there are no constraints, destroy data structures */
    let constraints = match constraints {
        None => {
            /* C: arkFreeVec(ark_mem, &ark_mem->constraints) */
            let mut v = arkode_mem.borrow_mut().constraints.take();
            arkFreeVec(arkode_mem, &mut v);
            arkode_mem.borrow_mut().constraints = v;
            return ARK_SUCCESS;
        }
        Some(constraints) => constraints,
    };

    /* Test if required vector ops. are defined */
    {
        let ops = constraints.ops.borrow();
        if ops.nvdiv.is_none()
            || ops.nvmaxnorm.is_none()
            || ops.nvcompare.is_none()
            || ops.nvprod.is_none()
            || ops.nvconstrmask.is_none()
            || ops.nvminquotient.is_none()
        {
            drop(ops);
            arkProcessError(
                Some(arkode_mem),
                ARK_ILL_INPUT,
                line!() as i32,
                "ARKodeSetConstraints",
                file!(),
                MSG_ARK_BAD_NVECTOR,
            );
            return ARK_ILL_INPUT;
        }
    }

    /* Check the constraints vector */
    let temptest = N_VMaxNorm(constraints);
    if (temptest > TWOPT5) || (temptest < HALF) {
        arkProcessError(
            Some(arkode_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "ARKodeSetConstraints",
            file!(),
            MSG_ARK_BAD_CONSTR,
        );
        return ARK_ILL_INPUT;
    }

    /* Allocate the internal constrains vector (if necessary) */
    let mut v = arkode_mem.borrow_mut().constraints.take();
    let allocOK = arkAllocVec(arkode_mem, constraints, &mut v);
    arkode_mem.borrow_mut().constraints = v;
    if !allocOK {
        return ARK_MEM_FAIL;
    }

    /* Load the constraints vector */
    let internal = arkode_mem
        .borrow()
        .constraints
        .as_ref()
        .expect("constraints")
        .clone();
    N_VScale(ONE, constraints, &internal);

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeSetMaxNumConstrFails:

  Set max number of allowed constraint failures in a step before
  returning an error
  ---------------------------------------------------------------*/
pub fn ARKodeSetMaxNumConstrFails(arkode_mem: &ARKodeMem, maxfails: i32) -> i32 {
    /* NULL-mem check: handled by type system */

    /* Guard against use for non-adaptive time stepper modules */
    if !arkode_mem.borrow().step_supports_adaptive {
        arkProcessError(
            Some(arkode_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetMaxNumConstrFails",
            file!(),
            "time-stepping module does not support temporal adaptivity",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    /* Passing maxfails = 0 sets the default, otherwise set to input */
    if maxfails <= 0 {
        arkode_mem.borrow_mut().maxconstrfails = MAXCONSTRFAILS;
    } else {
        arkode_mem.borrow_mut().maxconstrfails = maxfails;
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeSetCFLFraction:

  Specifies the safety factor to use on the maximum explicitly-
  stable step size.  Allowable values must be within the open
  interval (0,1).  A non-positive input implies a reset to
  the default value.
  ---------------------------------------------------------------*/
pub fn ARKodeSetCFLFraction(arkode_mem: &ARKodeMem, cfl_frac: sunrealtype) -> i32 {
    let retval = arkAccessHAdaptMem(arkode_mem, "ARKodeSetCFLFraction");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* Guard against use for non-adaptive time stepper modules */
    if !arkode_mem.borrow().step_supports_adaptive {
        arkProcessError(
            Some(arkode_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetCFLFraction",
            file!(),
            "time-stepping module does not support temporal adaptivity",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    /* set positive-valued parameters, otherwise set default */
    {
        let mut ark_mem = arkode_mem.borrow_mut();
        let hadapt_mem = ark_mem.hadapt_mem.as_mut().expect("hadapt_mem allocated");
        if cfl_frac <= ZERO {
            hadapt_mem.cfl = CFLFAC;
        } else {
            hadapt_mem.cfl = cfl_frac;
        }
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeSetAdaptivityAdjustment:

  Adjusts the method order supplied to the temporal adaptivity
  controller.  For example, if the user expects order reduction
  due to problem stiffness, they may request that the controller
  assume a reduced order of accuracy for the method by specifying
  a value adjust < 0.
  ---------------------------------------------------------------*/
pub fn ARKodeSetAdaptivityAdjustment(arkode_mem: &ARKodeMem, adjust: i32) -> i32 {
    let retval = arkAccessHAdaptMem(arkode_mem, "ARKodeSetAdaptivityAdjustment");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* Guard against use for non-adaptive time stepper modules */
    if !arkode_mem.borrow().step_supports_adaptive {
        arkProcessError(
            Some(arkode_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetAdaptivityAdjustment",
            file!(),
            "time-stepping module does not support temporal adaptivity",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    /* store requested adjustment */
    arkode_mem
        .borrow_mut()
        .hadapt_mem
        .as_mut()
        .expect("hadapt_mem allocated")
        .adjust = adjust;
    ARK_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeSetSafetyFactor:

  Specifies the safety factor to use on the error-based predicted
  time step size.  Allowable values must be within the open
  interval (0,1).  A non-positive input implies a reset to the
  default value.
  ---------------------------------------------------------------*/
pub fn ARKodeSetSafetyFactor(arkode_mem: &ARKodeMem, safety: sunrealtype) -> i32 {
    let retval = arkAccessHAdaptMem(arkode_mem, "ARKodeSetSafetyFactor");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* Guard against use for non-adaptive time stepper modules */
    if !arkode_mem.borrow().step_supports_adaptive {
        arkProcessError(
            Some(arkode_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetSafetyFactor",
            file!(),
            "time-stepping module does not support temporal adaptivity",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    /* check for allowable parameters */
    if safety > ONE {
        arkProcessError(
            Some(arkode_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "ARKodeSetSafetyFactor",
            file!(),
            "Illegal safety factor",
        );
        return ARK_ILL_INPUT;
    }

    /* set positive-valued parameters, otherwise set default */
    {
        let mut ark_mem = arkode_mem.borrow_mut();
        let hadapt_mem = ark_mem.hadapt_mem.as_mut().expect("hadapt_mem allocated");
        if safety <= ZERO {
            hadapt_mem.safety = SAFETY;
        } else {
            hadapt_mem.safety = safety;
        }
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeSetErrorBias:

  Specifies the error bias to use when performing adaptive-step
  error control.  Allowable values must be >= 1.0.  Any illegal
  value implies a reset to the default value.
  ---------------------------------------------------------------*/
pub fn ARKodeSetErrorBias(arkode_mem: &ARKodeMem, bias: sunrealtype) -> i32 {
    let retval = arkAccessHAdaptMem(arkode_mem, "ARKodeSetErrorBias");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* Guard against use for non-adaptive time stepper modules */
    if !arkode_mem.borrow().step_supports_adaptive {
        arkProcessError(
            Some(arkode_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetErrorBias",
            file!(),
            "time-stepping module does not support temporal adaptivity",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    /* Return an error if there is not a current SUNAdaptController */
    let hcontroller = arkode_mem
        .borrow()
        .hadapt_mem
        .as_ref()
        .expect("hadapt_mem allocated")
        .hcontroller
        .clone();
    let hcontroller = match hcontroller {
        Some(hcontroller) => hcontroller,
        None => {
            arkProcessError(
                Some(arkode_mem),
                ARK_MEM_NULL,
                line!() as i32,
                "ARKodeSetErrorBias",
                file!(),
                "SUNAdaptController NULL -- must be set before setting the error bias",
            );
            return ARK_MEM_NULL;
        }
    };

    /* set allowed value, otherwise set default */
    let retval = if bias < ONE {
        SUNAdaptController_SetErrorBias(&hcontroller, -ONE)
    } else {
        SUNAdaptController_SetErrorBias(&hcontroller, bias)
    };
    if retval != SUN_SUCCESS {
        arkProcessError(
            Some(arkode_mem),
            ARK_CONTROLLER_ERR,
            line!() as i32,
            "ARKodeSetErrorBias",
            file!(),
            "SUNAdaptController_SetErrorBias failure",
        );
        return ARK_CONTROLLER_ERR;
    }
    ARK_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeSetMaxGrowth:

  Specifies the maximum step size growth factor to be allowed
  between successive integration steps.  Note: the first step uses
  a separate maximum growth factor.  Allowable values must be
  > 1.0.  Any illegal value implies a reset to the default.
  ---------------------------------------------------------------*/
pub fn ARKodeSetMaxGrowth(arkode_mem: &ARKodeMem, mx_growth: sunrealtype) -> i32 {
    let retval = arkAccessHAdaptMem(arkode_mem, "ARKodeSetMaxGrowth");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* Guard against use for non-adaptive time stepper modules */
    if !arkode_mem.borrow().step_supports_adaptive {
        arkProcessError(
            Some(arkode_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetMaxGrowth",
            file!(),
            "time-stepping module does not support temporal adaptivity",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    /* set allowed value, otherwise set default */
    {
        let mut ark_mem = arkode_mem.borrow_mut();
        let hadapt_mem = ark_mem.hadapt_mem.as_mut().expect("hadapt_mem allocated");
        if mx_growth <= ONE {
            hadapt_mem.growth = GROWTH;
        } else {
            hadapt_mem.growth = mx_growth;
        }
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeSetMinReduction:

  Specifies the minimum possible step size reduction factor to be
  allowed between successive integration steps. Allowable values
  must be > 0.0 and < 1.0. Any illegal value implies a reset to
  the default.
  ---------------------------------------------------------------*/
pub fn ARKodeSetMinReduction(arkode_mem: &ARKodeMem, eta_min: sunrealtype) -> i32 {
    let retval = arkAccessHAdaptMem(arkode_mem, "ARKodeSetMinReduction");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* Guard against use for non-adaptive time stepper modules */
    if !arkode_mem.borrow().step_supports_adaptive {
        arkProcessError(
            Some(arkode_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetMinReduction",
            file!(),
            "time-stepping module does not support temporal adaptivity",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    /* set allowed value, otherwise set default */
    {
        let mut ark_mem = arkode_mem.borrow_mut();
        let hadapt_mem = ark_mem.hadapt_mem.as_mut().expect("hadapt_mem allocated");
        if eta_min >= ONE || eta_min <= ZERO {
            hadapt_mem.etamin = ETAMIN;
        } else {
            hadapt_mem.etamin = eta_min;
        }
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeSetFixedStepBounds:

  Specifies the step size growth interval within which the step
  size will remain unchanged.  Allowable values must enclose the
  value 1.0.  Any illegal interval implies a reset to the default.
  ---------------------------------------------------------------*/
pub fn ARKodeSetFixedStepBounds(
    arkode_mem: &ARKodeMem,
    lb: sunrealtype,
    ub: sunrealtype,
) -> i32 {
    let retval = arkAccessHAdaptMem(arkode_mem, "ARKodeSetFixedStepBounds");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* Guard against use for non-adaptive time stepper modules */
    if !arkode_mem.borrow().step_supports_adaptive {
        arkProcessError(
            Some(arkode_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetFixedStepBounds",
            file!(),
            "time-stepping module does not support temporal adaptivity",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    /* set allowable interval, otherwise set defaults */
    {
        let mut ark_mem = arkode_mem.borrow_mut();
        let hadapt_mem = ark_mem.hadapt_mem.as_mut().expect("hadapt_mem allocated");
        if (lb <= ONE) && (ub >= ONE) {
            hadapt_mem.lbound = lb;
            hadapt_mem.ubound = ub;
        } else {
            hadapt_mem.lbound = HFIXED_LB;
            hadapt_mem.ubound = HFIXED_UB;
        }
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeSetMaxFirstGrowth:

  Specifies the user-provided time step adaptivity constant
  etamx1.  Legal values are greater than 1.0.  Illegal values
  imply a reset to the default value.
  ---------------------------------------------------------------*/
pub fn ARKodeSetMaxFirstGrowth(arkode_mem: &ARKodeMem, etamx1: sunrealtype) -> i32 {
    let retval = arkAccessHAdaptMem(arkode_mem, "ARKodeSetMaxFirstGrowth");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* Guard against use for non-adaptive time stepper modules */
    if !arkode_mem.borrow().step_supports_adaptive {
        arkProcessError(
            Some(arkode_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetMaxFirstGrowth",
            file!(),
            "time-stepping module does not support temporal adaptivity",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    /* if argument legal set it, otherwise set default */
    {
        let mut ark_mem = arkode_mem.borrow_mut();
        let hadapt_mem = ark_mem.hadapt_mem.as_mut().expect("hadapt_mem allocated");
        if etamx1 <= ONE {
            hadapt_mem.etamx1 = ETAMX1;
        } else {
            hadapt_mem.etamx1 = etamx1;
        }
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeSetMaxEFailGrowth:

  Specifies the user-provided time step adaptivity constant
  etamxf. Legal values are in the interval (0,1].  Illegal values
  imply a reset to the default value.
  ---------------------------------------------------------------*/
pub fn ARKodeSetMaxEFailGrowth(arkode_mem: &ARKodeMem, etamxf: sunrealtype) -> i32 {
    let retval = arkAccessHAdaptMem(arkode_mem, "ARKodeSetMaxEFailGrowth");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* Guard against use for non-adaptive time stepper modules */
    if !arkode_mem.borrow().step_supports_adaptive {
        arkProcessError(
            Some(arkode_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetMaxEFailGrowth",
            file!(),
            "time-stepping module does not support temporal adaptivity",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    /* if argument legal set it, otherwise set default */
    {
        let mut ark_mem = arkode_mem.borrow_mut();
        let hadapt_mem = ark_mem.hadapt_mem.as_mut().expect("hadapt_mem allocated");
        if (etamxf <= ZERO) || (etamxf > ONE) {
            hadapt_mem.etamxf = ETAMXF;
        } else {
            hadapt_mem.etamxf = etamxf;
        }
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeSetSmallNumEFails:

  Specifies the user-provided time step adaptivity constant
  small_nef.  Legal values are > 0.  Illegal values
  imply a reset to the default value.
  ---------------------------------------------------------------*/
pub fn ARKodeSetSmallNumEFails(arkode_mem: &ARKodeMem, small_nef: i32) -> i32 {
    let retval = arkAccessHAdaptMem(arkode_mem, "ARKodeSetSmallNumEFails");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* Guard against use for non-adaptive time stepper modules */
    if !arkode_mem.borrow().step_supports_adaptive {
        arkProcessError(
            Some(arkode_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetSmallNumEFails",
            file!(),
            "time-stepping module does not support temporal adaptivity",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    /* if argument legal set it, otherwise set default */
    {
        let mut ark_mem = arkode_mem.borrow_mut();
        let hadapt_mem = ark_mem.hadapt_mem.as_mut().expect("hadapt_mem allocated");
        if small_nef <= 0 {
            hadapt_mem.small_nef = SMALL_NEF;
        } else {
            hadapt_mem.small_nef = small_nef;
        }
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeSetMaxCFailGrowth:

  Specifies the user-provided time step adaptivity constant
  etacf. Legal values are in the interval (0,1].  Illegal values
  imply a reset to the default value.
  ---------------------------------------------------------------*/
pub fn ARKodeSetMaxCFailGrowth(arkode_mem: &ARKodeMem, etacf: sunrealtype) -> i32 {
    let retval = arkAccessHAdaptMem(arkode_mem, "ARKodeSetMaxCFailGrowth");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* Guard against use for non-adaptive time stepper modules */
    if !arkode_mem.borrow().step_supports_adaptive {
        arkProcessError(
            Some(arkode_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetMaxCFailGrowth",
            file!(),
            "time-stepping module does not support temporal adaptivity",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    /* if argument legal set it, otherwise set default */
    {
        let mut ark_mem = arkode_mem.borrow_mut();
        let hadapt_mem = ark_mem.hadapt_mem.as_mut().expect("hadapt_mem allocated");
        if (etacf <= ZERO) || (etacf > ONE) {
            hadapt_mem.etacf = ETACF;
        } else {
            hadapt_mem.etacf = etacf;
        }
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeSetStabilityFn:

  Specifies the user-provided explicit time step stability
  function to use.  A NULL input function implies a reset to
  the default function (empty).
  ---------------------------------------------------------------*/
pub fn ARKodeSetStabilityFn(
    arkode_mem: &ARKodeMem,
    EStab: Option<ARKExpStabFn>,
    estab_data: Option<Box<dyn Any>>,
) -> i32 {
    let retval = arkAccessHAdaptMem(arkode_mem, "ARKodeSetStabilityFn");
    if retval != ARK_SUCCESS {
        return retval;
    }

    /* Guard against use for non-adaptive time stepper modules */
    if !arkode_mem.borrow().step_supports_adaptive {
        arkProcessError(
            Some(arkode_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetStabilityFn",
            file!(),
            "time-stepping module does not support temporal adaptivity",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    /* NULL argument sets default, otherwise set inputs */
    {
        let mut ark_mem = arkode_mem.borrow_mut();
        let hadapt_mem = ark_mem.hadapt_mem.as_mut().expect("hadapt_mem allocated");
        hadapt_mem.expstab = EStab;
        hadapt_mem.estab_data = estab_data;
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeSetMaxErrTestFails:

  Specifies the maximum number of error test failures during one
  step try.  A non-positive input implies a reset to
  the default value.
  ---------------------------------------------------------------*/
pub fn ARKodeSetMaxErrTestFails(arkode_mem: &ARKodeMem, maxnef: i32) -> i32 {
    /* NULL-mem check: handled by type system */

    /* Guard against use for non-adaptive time stepper modules */
    if !arkode_mem.borrow().step_supports_adaptive {
        arkProcessError(
            Some(arkode_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetMaxErrTestFails",
            file!(),
            "time-stepping module does not support temporal adaptivity",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    /* argument <= 0 sets default, otherwise set input */
    if maxnef <= 0 {
        arkode_mem.borrow_mut().maxnef = MAXNEF;
    } else {
        arkode_mem.borrow_mut().maxnef = maxnef;
    }
    ARK_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeSetMaxConvFails:

  Specifies the maximum number of nonlinear convergence failures
  during one step try.  A non-positive input implies a reset to
  the default value.
  ---------------------------------------------------------------*/
pub fn ARKodeSetMaxConvFails(arkode_mem: &ARKodeMem, maxncf: i32) -> i32 {
    /* NULL-mem check: handled by type system */

    /* Guard against use for time steppers that do not need an algebraic solver */
    if !arkode_mem.borrow().step_supports_implicit {
        arkProcessError(
            Some(arkode_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeSetMaxConvFails",
            file!(),
            "time-stepping module does not require an algebraic solver",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    /* argument <= 0 sets default, otherwise set input */
    if maxncf <= 0 {
        arkode_mem.borrow_mut().maxncf = MAXNCF;
    } else {
        arkode_mem.borrow_mut().maxncf = maxncf;
    }
    ARK_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeSetAccumulatedErrorType:

  This routine sets the accumulated temporal error estimation
  strategy.
  ---------------------------------------------------------------*/
pub fn ARKodeSetAccumulatedErrorType(arkode_mem: &ARKodeMem, accum_type: ARKAccumError) -> i32 {
    let retval = ARKodeResetAccumulatedError(arkode_mem);
    if retval != ARK_SUCCESS {
        return retval;
    }
    arkode_mem.borrow_mut().AccumErrorType = accum_type;
    ARK_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeResetAccumulatedError:

  This routine resets the accumulated temporal error estimate.
  ---------------------------------------------------------------*/
pub fn ARKodeResetAccumulatedError(arkode_mem: &ARKodeMem) -> i32 {
    /* NULL-mem check: handled by type system */

    /* Guard against use for non-adaptive time stepper modules */
    if !arkode_mem.borrow().step_supports_adaptive {
        arkProcessError(
            Some(arkode_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeResetAccumulatedError",
            file!(),
            "time-stepping module does not support temporal adaptivity",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    /* Reset value and counter, and return */
    {
        let mut ark_mem = arkode_mem.borrow_mut();
        ark_mem.AccumErrorStart = ark_mem.tn;
        ark_mem.AccumError = ZERO;
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeSetAdjointCheckpointScheme:
  ARKodeSetAdjointCheckpointIndex:

  Specifies the checkpointing scheme and index to be used for adjoint
  sensitivity analysis.
  ---------------------------------------------------------------*/
pub fn ARKodeSetAdjointCheckpointScheme(
    arkode_mem: &ARKodeMem,
    checkpoint_scheme: Option<&SUNAdjointCheckpointScheme>,
) -> i32 {
    /* NULL-mem check: handled by type system */

    /* handle clone in = C pointer copy */
    arkode_mem.borrow_mut().checkpoint_scheme = checkpoint_scheme.cloned();

    ARK_SUCCESS
}

pub fn ARKodeSetAdjointCheckpointIndex(arkode_mem: &ARKodeMem, step_index: suncountertype) -> i32 {
    /* NULL-mem check: handled by type system */

    if step_index < 0 {
        arkProcessError(
            Some(arkode_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "ARKodeSetAdjointCheckpointIndex",
            file!(),
            "step_index must be >= 0",
        );
        return ARK_ILL_INPUT;
    }

    arkode_mem.borrow_mut().checkpoint_step_idx = step_index;

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeSetUseCompensatedSums:

  Specifies that ARKode should use compensated summation to reduce
  the effects of floating-point roundoff.
  ---------------------------------------------------------------*/
pub fn ARKodeSetUseCompensatedSums(arkode_mem: &ARKodeMem, onoff: sunbooleantype) -> i32 {
    /* NULL-mem check: handled by type system */

    arkode_mem.borrow_mut().use_compensated_sums = onoff;

    /* Call stepper routine (if provided) */
    let step_setusecompensatedsums = arkode_mem.borrow().step_setusecompensatedsums;
    if let Some(step_setusecompensatedsums) = step_setusecompensatedsums {
        return step_setusecompensatedsums(arkode_mem, onoff);
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeInit:

  Allocates internal data structures for an ARKODE stepper module
  before the first call to ARKodeEvolve.

  **THIS MUST BE CALLED AFTER ALL "SET" ROUTINES.**
  ---------------------------------------------------------------*/
pub fn ARKodeInit(arkode_mem: &ARKodeMem) -> i32 {
    /* NULL-mem check: handled by type system */

    /* For now, prohibit the user from calling this after data has
    already been initialized */
    if arkode_mem.borrow().initialized {
        arkProcessError(
            Some(arkode_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "ARKodeInit",
            file!(),
            "Time stepper data has already been allocated",
        );
        return ARK_ILL_INPUT;
    }

    /* Call step_init routine with "FIRST_INIT" flag, requesting
    that the time stepper module allocate any remaining internal
    data */
    let step_init = arkode_mem.borrow().step_init;
    let step_init = match step_init {
        Some(step_init) => step_init,
        None => {
            arkProcessError(
                Some(arkode_mem),
                ARK_ILL_INPUT,
                line!() as i32,
                "ARKodeInit",
                file!(),
                "Time stepper module is missing",
            );
            return ARK_ILL_INPUT;
        }
    };
    let retval = step_init(arkode_mem, FIRST_INIT);
    if retval != ARK_SUCCESS {
        arkProcessError(
            Some(arkode_mem),
            retval,
            line!() as i32,
            "ARKodeInit",
            file!(),
            "Error in initialization of time stepper module",
        );
    }
    arkode_mem.borrow_mut().preallocated = SUNTRUE;
    retval
}

/*===============================================================
  ARKODE optional output utility functions
  ===============================================================*/

/*---------------------------------------------------------------
  ARKodeGetNumRhsEvals:

  Returns the current number of RHS evaluations
  ---------------------------------------------------------------*/
pub fn ARKodeGetNumRhsEvals(
    arkode_mem: &ARKodeMem,
    partition_index: i32,
    num_rhs_evals: &mut i64,
) -> i32 {
    /* NULL-mem check: handled by type system */

    /* Call stepper routine (if provided) */
    let step_getnumrhsevals = arkode_mem.borrow().step_getnumrhsevals;
    if let Some(step_getnumrhsevals) = step_getnumrhsevals {
        step_getnumrhsevals(arkode_mem, partition_index, num_rhs_evals)
    } else {
        arkProcessError(
            Some(arkode_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeGetNumRhsEvals",
            file!(),
            "time-stepping module does not support this function",
        );
        ARK_STEPPER_UNSUPPORTED
    }
}

/*---------------------------------------------------------------
  ARKodeGetNumStepAttempts:

   Returns the current number of steps attempted by the solver
  ---------------------------------------------------------------*/
pub fn ARKodeGetNumStepAttempts(arkode_mem: &ARKodeMem, nstep_attempts: &mut i64) -> i32 {
    /* NULL-mem check: handled by type system */

    *nstep_attempts = arkode_mem.borrow().nst_attempts;
    ARK_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeGetNumSteps:

  Returns the current number of integration steps
  ---------------------------------------------------------------*/
pub fn ARKodeGetNumSteps(arkode_mem: &ARKodeMem, nsteps: &mut i64) -> i32 {
    /* NULL-mem check: handled by type system */

    *nsteps = arkode_mem.borrow().nst;
    ARK_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeGetActualInitStep:

  Returns the step size used on the first step
  ---------------------------------------------------------------*/
pub fn ARKodeGetActualInitStep(arkode_mem: &ARKodeMem, hinused: &mut sunrealtype) -> i32 {
    /* NULL-mem check: handled by type system */

    *hinused = arkode_mem.borrow().h0u;
    ARK_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeGetLastStep:

  Returns the step size used on the last successful step
  ---------------------------------------------------------------*/
pub fn ARKodeGetLastStep(arkode_mem: &ARKodeMem, hlast: &mut sunrealtype) -> i32 {
    /* NULL-mem check: handled by type system */

    *hlast = arkode_mem.borrow().hold;
    ARK_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeGetCurrentStep:

  Returns the step size to be attempted on the next step
  ---------------------------------------------------------------*/
pub fn ARKodeGetCurrentStep(arkode_mem: &ARKodeMem, hcur: &mut sunrealtype) -> i32 {
    /* NULL-mem check: handled by type system */

    *hcur = arkode_mem.borrow().next_h;
    ARK_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeGetStepDirection:

  Gets the direction of integration (forward or backward) based
  on the sign of stepdir. A value of 0 indicates integration can
  proceed in either direction.
  ---------------------------------------------------------------*/
pub fn ARKodeGetStepDirection(arkode_mem: &ARKodeMem, stepdir: &mut sunrealtype) -> i32 {
    /* NULL-mem check: handled by type system */
    /* C additionally reports (but does not act on) `stepdir == NULL`;
    unreachable through `&mut sunrealtype` */

    let ark_mem = arkode_mem.borrow();
    *stepdir = if ark_mem.fixedstep || ark_mem.h == ZERO {
        ark_mem.hin
    } else {
        ark_mem.h
    };
    ARK_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeGetCurrentState:

  Returns the current solution (before or after as step) or
  stage value (during step solve).
  ---------------------------------------------------------------*/
pub fn ARKodeGetCurrentState(arkode_mem: &ARKodeMem, state: &mut Option<N_Vector>) -> i32 {
    /* NULL-mem check: handled by type system */

    /* handle clone out = C pointer copy */
    *state = arkode_mem.borrow().ycur.clone();
    ARK_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeGetEstLocalErrors:

  Returns an estimate of the local error
  ---------------------------------------------------------------*/
pub fn ARKodeGetEstLocalErrors(arkode_mem: &ARKodeMem, ele: &N_Vector) -> i32 {
    /* NULL-mem check: handled by type system */

    /* Call stepper-specific routine (if provided); otherwise return an error */
    let step_getestlocalerrors = arkode_mem.borrow().step_getestlocalerrors;
    if let Some(step_getestlocalerrors) = step_getestlocalerrors {
        step_getestlocalerrors(arkode_mem, ele)
    } else {
        arkProcessError(
            Some(arkode_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeGetEstLocalErrors",
            file!(),
            "time-stepping module does provide a temporal error estimate",
        );
        ARK_STEPPER_UNSUPPORTED
    }
}

/*---------------------------------------------------------------
  ARKodeGetLastTime:

  Returns the last saved value of the independent variable
  ---------------------------------------------------------------*/
pub fn ARKodeGetLastTime(arkode_mem: &ARKodeMem, tn: &mut sunrealtype) -> i32 {
    /* NULL-mem check: handled by type system */

    *tn = arkode_mem.borrow().tn;
    ARK_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeGetLastState:

  Returns the last saved time step solution.
  ---------------------------------------------------------------*/
pub fn ARKodeGetLastState(arkode_mem: &ARKodeMem, yn: &mut Option<N_Vector>) -> i32 {
    /* NULL-mem check: handled by type system */

    /* handle clone out = C pointer copy */
    *yn = arkode_mem.borrow().yn.clone();
    ARK_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeGetCurrentTime:

  Returns the current value of the independent variable
  ---------------------------------------------------------------*/
pub fn ARKodeGetCurrentTime(arkode_mem: &ARKodeMem, tcur: &mut sunrealtype) -> i32 {
    /* NULL-mem check: handled by type system */

    *tcur = arkode_mem.borrow().tcur;
    ARK_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeGetCurrentGamma: Returns the current value of gamma
  ---------------------------------------------------------------*/
pub fn ARKodeGetCurrentGamma(arkode_mem: &ARKodeMem, gamma: &mut sunrealtype) -> i32 {
    /* NULL-mem check: handled by type system */

    /* Guard against use for time steppers that do not need an algebraic solver */
    if !arkode_mem.borrow().step_supports_implicit {
        arkProcessError(
            Some(arkode_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeGetCurrentGamma",
            file!(),
            "time-stepping module does not require an algebraic solver",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    /* Call stepper routine to compute the state (if provided) */
    let step_getcurrentgamma = arkode_mem.borrow().step_getcurrentgamma;
    if let Some(step_getcurrentgamma) = step_getcurrentgamma {
        step_getcurrentgamma(arkode_mem, gamma)
    } else {
        arkProcessError(
            Some(arkode_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeGetCurrentGamma",
            file!(),
            "time-stepping module does not support this function",
        );
        ARK_STEPPER_UNSUPPORTED
    }
}

/*---------------------------------------------------------------
  ARKodeGetTolScaleFactor:

  Returns a suggested factor for scaling tolerances
  ---------------------------------------------------------------*/
pub fn ARKodeGetTolScaleFactor(arkode_mem: &ARKodeMem, tolsfact: &mut sunrealtype) -> i32 {
    /* NULL-mem check: handled by type system */

    /* Guard against use for time steppers that do not use tolerances
    (i.e., neither supports adaptivity nor needs an algebraic solver) */
    {
        let ark_mem = arkode_mem.borrow();
        if (!ark_mem.step_supports_implicit) && (!ark_mem.step_supports_adaptive) {
            drop(ark_mem);
            arkProcessError(
                Some(arkode_mem),
                ARK_STEPPER_UNSUPPORTED,
                line!() as i32,
                "ARKodeGetTolScaleFactor",
                file!(),
                "time-stepping module does not use tolerances",
            );
            return ARK_STEPPER_UNSUPPORTED;
        }
    }

    *tolsfact = arkode_mem.borrow().tolsf;
    ARK_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeGetErrWeights:

  This routine returns the current error weight vector.
  ---------------------------------------------------------------*/
pub fn ARKodeGetErrWeights(arkode_mem: &ARKodeMem, eweight: &N_Vector) -> i32 {
    /* NULL-mem check: handled by type system */

    /* Guard against use for time steppers that do not use tolerances
    (i.e., neither supports adaptivity nor needs an algebraic solver) */
    {
        let ark_mem = arkode_mem.borrow();
        if (!ark_mem.step_supports_implicit) && (!ark_mem.step_supports_adaptive) {
            drop(ark_mem);
            arkProcessError(
                Some(arkode_mem),
                ARK_STEPPER_UNSUPPORTED,
                line!() as i32,
                "ARKodeGetErrWeights",
                file!(),
                "time-stepping module does not use tolerances",
            );
            return ARK_STEPPER_UNSUPPORTED;
        }
    }

    let ewt = arkode_mem.borrow().ewt.as_ref().expect("ewt").clone();
    N_VScale(ONE, &ewt, eweight);
    ARK_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeGetResWeights:

  This routine returns the current residual weight vector.
  ---------------------------------------------------------------*/
pub fn ARKodeGetResWeights(arkode_mem: &ARKodeMem, rweight: &N_Vector) -> i32 {
    /* NULL-mem check: handled by type system */

    /* Guard against use for time steppers that do not support mass matrices */
    if !arkode_mem.borrow().step_supports_massmatrix {
        arkProcessError(
            Some(arkode_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeGetResWeights",
            file!(),
            "time-stepping module does not support non-identity mass matrices",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    let rwt = arkode_mem.borrow().rwt.as_ref().expect("rwt").clone();
    N_VScale(ONE, &rwt, rweight);
    ARK_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeGetWorkSpace:

  Returns integrator work space requirements
  ---------------------------------------------------------------*/
pub fn ARKodeGetWorkSpace(arkode_mem: &ARKodeMem, lenrw: &mut i64, leniw: &mut i64) -> i32 {
    /* NULL-mem check: handled by type system */

    let ark_mem = arkode_mem.borrow();
    *leniw = ark_mem.liw;
    *lenrw = ark_mem.lrw;
    ARK_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeGetNumGEvals:

  Returns the current number of calls to g (for rootfinding)
  ---------------------------------------------------------------*/
pub fn ARKodeGetNumGEvals(arkode_mem: &ARKodeMem, ngevals: &mut i64) -> i32 {
    /* NULL-mem check: handled by type system */
    if arkode_mem.borrow().root_mem.is_none() {
        arkProcessError(
            Some(arkode_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "ARKodeGetNumGEvals",
            file!(),
            MSG_ARK_NO_MEM,
        );
        return ARK_MEM_NULL;
    }
    *ngevals = arkode_mem.borrow().root_mem.as_ref().expect("root_mem").nge;
    ARK_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeGetRootInfo:

  Returns pointer to array rootsfound showing roots found
  ---------------------------------------------------------------*/
pub fn ARKodeGetRootInfo(arkode_mem: &ARKodeMem, rootsfound: &mut [i32]) -> i32 {
    /* NULL-mem check: handled by type system */
    if arkode_mem.borrow().root_mem.is_none() {
        arkProcessError(
            Some(arkode_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "ARKodeGetRootInfo",
            file!(),
            MSG_ARK_NO_MEM,
        );
        return ARK_MEM_NULL;
    }
    let ark_mem = arkode_mem.borrow();
    let ark_root_mem = ark_mem.root_mem.as_ref().expect("root_mem");
    for i in 0..ark_root_mem.nrtfn as usize {
        rootsfound[i] = ark_root_mem.iroots[i];
    }
    ARK_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeGetStepStats:

  Returns step statistics
  ---------------------------------------------------------------*/
pub fn ARKodeGetStepStats(
    arkode_mem: &ARKodeMem,
    nsteps: &mut i64,
    hinused: &mut sunrealtype,
    hlast: &mut sunrealtype,
    hcur: &mut sunrealtype,
    tcur: &mut sunrealtype,
) -> i32 {
    /* NULL-mem check: handled by type system */

    let ark_mem = arkode_mem.borrow();
    *nsteps = ark_mem.nst;
    *hinused = ark_mem.h0u;
    *hlast = ark_mem.hold;
    *hcur = ark_mem.next_h;
    *tcur = ark_mem.tcur;
    ARK_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeGetAccumulatedError:

  This routine returns the accumulated temporal error estimate.
  ---------------------------------------------------------------*/
pub fn ARKodeGetAccumulatedError(arkode_mem: &ARKodeMem, accum_error: &mut sunrealtype) -> i32 {
    /* NULL-mem check: handled by type system */

    /* Return an error if the stepper cannot accumulate temporal error */
    if !arkode_mem.borrow().step_supports_adaptive {
        arkProcessError(
            Some(arkode_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeGetAccumulatedError",
            file!(),
            "time-stepping module does not support accumulated error estimation",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    let (tcur, AccumErrorStart, AccumErrorType, AccumError, reltol) = {
        let ark_mem = arkode_mem.borrow();
        (
            ark_mem.tcur,
            ark_mem.AccumErrorStart,
            ark_mem.AccumErrorType,
            ark_mem.AccumError,
            ark_mem.reltol,
        )
    };

    /* Get time since last accumulated error reset */
    let time_interval = tcur - AccumErrorStart;

    /* Fill output based on error accumulation type */
    if AccumErrorType == ARK_ACCUMERROR_MAX {
        *accum_error = AccumError * reltol;
    } else if AccumErrorType == ARK_ACCUMERROR_SUM {
        *accum_error = AccumError * reltol;
    } else if AccumErrorType == ARK_ACCUMERROR_AVG {
        *accum_error = AccumError * reltol / time_interval;
    } else {
        arkProcessError(
            Some(arkode_mem),
            ARK_WARNING,
            line!() as i32,
            "ARKodeGetAccumulatedError",
            file!(),
            "temporal error accumulation is currently disabled",
        );
        return ARK_WARNING;
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeGetNumConstrFails:

  Returns the current number of constraint fails
  ---------------------------------------------------------------*/
pub fn ARKodeGetNumConstrFails(arkode_mem: &ARKodeMem, nconstrfails: &mut i64) -> i32 {
    /* NULL-mem check: handled by type system */

    *nconstrfails = arkode_mem.borrow().nconstrfails;
    ARK_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeGetNumExpSteps:

  Returns the current number of stability-limited steps
  ---------------------------------------------------------------*/
pub fn ARKodeGetNumExpSteps(arkode_mem: &ARKodeMem, nsteps: &mut i64) -> i32 {
    /* NULL-mem check: handled by type system */

    *nsteps = arkode_mem
        .borrow()
        .hadapt_mem
        .as_ref()
        .expect("hadapt_mem allocated")
        .nst_exp;
    ARK_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeGetNumAccSteps:

  Returns the current number of accuracy-limited steps
  ---------------------------------------------------------------*/
pub fn ARKodeGetNumAccSteps(arkode_mem: &ARKodeMem, nsteps: &mut i64) -> i32 {
    /* NULL-mem check: handled by type system */

    *nsteps = arkode_mem
        .borrow()
        .hadapt_mem
        .as_ref()
        .expect("hadapt_mem allocated")
        .nst_acc;
    ARK_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeGetNumErrTestFails:

  Returns the current number of error test failures
  ---------------------------------------------------------------*/
pub fn ARKodeGetNumErrTestFails(arkode_mem: &ARKodeMem, netfails: &mut i64) -> i32 {
    /* NULL-mem check: handled by type system */

    *netfails = arkode_mem.borrow().netf;
    ARK_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeComputeState:

  Computes y based on the current prediction and a given
  correction.
  ---------------------------------------------------------------*/
pub fn ARKodeComputeState(arkode_mem: &ARKodeMem, zcor: &N_Vector, z: &N_Vector) -> i32 {
    /* NULL-mem check: handled by type system */

    /* Guard against use for incompatible time stepper modules */
    if !arkode_mem.borrow().step_supports_implicit {
        arkProcessError(
            Some(arkode_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeComputeState",
            file!(),
            "time-stepping module does not support algebraic solvers",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    /* Call stepper routine to compute the state (if provided) */
    let step_computestate = arkode_mem.borrow().step_computestate;
    if let Some(step_computestate) = step_computestate {
        step_computestate(arkode_mem, zcor, z)
    } else {
        arkProcessError(
            Some(arkode_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeComputeState",
            file!(),
            "time-stepping module does not support this function",
        );
        ARK_STEPPER_UNSUPPORTED
    }
}

/*---------------------------------------------------------------
  ARKodeGetNonlinearSystemData:

  This routine provides access to the relevant data needed to
  compute the nonlinear system function.
  ---------------------------------------------------------------*/
pub fn ARKodeGetNonlinearSystemData(
    arkode_mem: &ARKodeMem,
    tcur: &mut sunrealtype,
    zpred: &mut Option<N_Vector>,
    z: &mut Option<N_Vector>,
    Fi: &mut Option<N_Vector>,
    gamma: &mut sunrealtype,
    sdata: &mut Option<N_Vector>,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    /* NULL-mem check: handled by type system */

    /* Guard against use for incompatible time stepper modules */
    if !arkode_mem.borrow().step_supports_implicit {
        arkProcessError(
            Some(arkode_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeGetNonlinearSystemData",
            file!(),
            "time-stepping module does not support algebraic solvers",
        );
        return ARK_STEPPER_UNSUPPORTED;
    }

    /* Call stepper routine to compute the state (if provided) */
    let step_getnonlinearsystemdata = arkode_mem.borrow().step_getnonlinearsystemdata;
    if let Some(step_getnonlinearsystemdata) = step_getnonlinearsystemdata {
        step_getnonlinearsystemdata(arkode_mem, tcur, zpred, z, Fi, gamma, sdata, user_data)
    } else {
        arkProcessError(
            Some(arkode_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeGetNonlinearSystemData",
            file!(),
            "time-stepping module does not support this function",
        );
        ARK_STEPPER_UNSUPPORTED
    }
}

/*---------------------------------------------------------------
  ARKodeGetNumNonlinSolvIters:

  Returns the current number of nonlinear solver iterations
  ---------------------------------------------------------------*/
pub fn ARKodeGetNumNonlinSolvIters(arkode_mem: &ARKodeMem, nniters: &mut i64) -> i32 {
    /* NULL-mem check: handled by type system */

    /* Call stepper routine to compute the state (if provided) */
    let step_getnumnonlinsolviters = arkode_mem.borrow().step_getnumnonlinsolviters;
    if let Some(step_getnumnonlinsolviters) = step_getnumnonlinsolviters {
        step_getnumnonlinsolviters(arkode_mem, nniters)
    } else {
        *nniters = 0;
        ARK_SUCCESS
    }
}

/*---------------------------------------------------------------
  ARKodeGetNumNonlinSolvConvFails:

  Returns the current number of nonlinear solver convergence fails
  ---------------------------------------------------------------*/
pub fn ARKodeGetNumNonlinSolvConvFails(arkode_mem: &ARKodeMem, nnfails: &mut i64) -> i32 {
    /* NULL-mem check: handled by type system */

    /* Call stepper routine to compute the state (if provided) */
    let step_getnumnonlinsolvconvfails = arkode_mem.borrow().step_getnumnonlinsolvconvfails;
    if let Some(step_getnumnonlinsolvconvfails) = step_getnumnonlinsolvconvfails {
        step_getnumnonlinsolvconvfails(arkode_mem, nnfails)
    } else {
        *nnfails = 0;
        ARK_SUCCESS
    }
}

/*---------------------------------------------------------------
  ARKodeGetNonlinSolvStats:

  Returns nonlinear solver statistics
  ---------------------------------------------------------------*/
pub fn ARKodeGetNonlinSolvStats(
    arkode_mem: &ARKodeMem,
    nniters: &mut i64,
    nnfails: &mut i64,
) -> i32 {
    /* NULL-mem check: handled by type system */

    /* Call stepper routine to compute the state (if provided) */
    let step_getnonlinsolvstats = arkode_mem.borrow().step_getnonlinsolvstats;
    if let Some(step_getnonlinsolvstats) = step_getnonlinsolvstats {
        step_getnonlinsolvstats(arkode_mem, nniters, nnfails)
    } else {
        *nnfails = 0;
        *nniters = *nnfails;
        ARK_SUCCESS
    }
}

/*---------------------------------------------------------------
  ARKodeGetNumStepSolveFails:

  Returns the current number of failed steps due to an algebraic
  solver convergence failure.
  ---------------------------------------------------------------*/
pub fn ARKodeGetNumStepSolveFails(arkode_mem: &ARKodeMem, nncfails: &mut i64) -> i32 {
    /* NULL-mem check: handled by type system */

    *nncfails = arkode_mem.borrow().ncfn;
    ARK_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeGetNumLinSolvSetups:

  Returns the current number of calls to the lsetup routine
  ---------------------------------------------------------------*/
pub fn ARKodeGetNumLinSolvSetups(arkode_mem: &ARKodeMem, nlinsetups: &mut i64) -> i32 {
    /* NULL-mem check: handled by type system */

    /* Call stepper routine to compute the state (if provided) */
    let step_getnumlinsolvsetups = arkode_mem.borrow().step_getnumlinsolvsetups;
    if let Some(step_getnumlinsolvsetups) = step_getnumlinsolvsetups {
        step_getnumlinsolvsetups(arkode_mem, nlinsetups)
    } else {
        *nlinsetups = 0;
        ARK_SUCCESS
    }
}

/*---------------------------------------------------------------
  ARKodeGetUserData:

  Returns the user data pointer
  ---------------------------------------------------------------*/
pub fn ARKodeGetUserData(arkode_mem: &ARKodeMem, user_data: &mut Option<Box<dyn Any>>) -> i32 {
    /* NULL-mem check: handled by type system */
    /* C hands back the stored pointer without transferring ownership; the
    safe port SWAPS the token with the caller's out-param (accepted
    deviation class 6) -- hand it back before the next callback. */
    std::mem::swap(&mut arkode_mem.borrow_mut().user_data, user_data);

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  ARKodeGetStageIndex:

  Returns the index of the current stage and the total number of
  stages. If this is not supplied by the time-stepping module
  then an error is returned and the values are set to (-1, -1).
  ---------------------------------------------------------------*/
pub fn ARKodeGetStageIndex(arkode_mem: &ARKodeMem, stage: &mut i32, max_stages: &mut i32) -> i32 {
    /* NULL-mem check: handled by type system */

    /* Call stepper routine to compute the state (if provided) */
    let step_getstageindex = arkode_mem.borrow().step_getstageindex;
    if let Some(step_getstageindex) = step_getstageindex {
        step_getstageindex(arkode_mem, stage, max_stages)
    } else {
        *stage = -1;
        *max_stages = -1;
        arkProcessError(
            Some(arkode_mem),
            ARK_STEPPER_UNSUPPORTED,
            line!() as i32,
            "ARKodeGetStageIndex",
            file!(),
            "time-stepping module does not support this function",
        );
        ARK_STEPPER_UNSUPPORTED
    }
}

/*-----------------------------------------------------------------
  ARKodePrintAllStats

  Prints the current value of all statistics
  ---------------------------------------------------------------*/

pub fn ARKodePrintAllStats(
    arkode_mem: &ARKodeMem,
    outfile: &SUNFile,
    fmt: SUNOutputFormat,
) -> i32 {
    /* NULL-mem check: handled by type system */

    if fmt != SUNOutputFormat::SUN_OUTPUTFORMAT_TABLE
        && fmt != SUNOutputFormat::SUN_OUTPUTFORMAT_CSV
    {
        arkProcessError(
            Some(arkode_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "ARKodePrintAllStats",
            file!(),
            "Invalid formatting option.",
        );
        return ARK_ILL_INPUT;
    }

    /* Copy every statistic out under one borrow (printing and the
    relaxation/stepper hooks must not hold a borrow of the mem), then
    print in the exact C order. */
    let (
        tcur,
        nst,
        nst_attempts,
        nst_exp,
        nst_acc,
        netf,
        ncfn,
        nconstrfails,
        h0u,
        hold,
        next_h,
        nge,
        relax_enabled,
    );
    {
        let ark_mem = arkode_mem.borrow();
        tcur = ark_mem.tcur;
        nst = ark_mem.nst;
        nst_attempts = ark_mem.nst_attempts;
        let hadapt_mem = ark_mem.hadapt_mem.as_ref().expect("hadapt_mem allocated");
        nst_exp = hadapt_mem.nst_exp;
        nst_acc = hadapt_mem.nst_acc;
        netf = ark_mem.netf;
        ncfn = ark_mem.ncfn;
        nconstrfails = ark_mem.nconstrfails;
        h0u = ark_mem.h0u;
        hold = ark_mem.hold;
        next_h = ark_mem.next_h;
        nge = ark_mem.root_mem.as_ref().map(|root_mem| root_mem.nge);
        relax_enabled = ark_mem.relax_enabled;
    }

    sunfprintf_real(outfile, fmt, SUNTRUE, "Current time", tcur);
    sunfprintf_long(outfile, fmt, SUNFALSE, "Steps", nst);
    sunfprintf_long(outfile, fmt, SUNFALSE, "Step attempts", nst_attempts);
    sunfprintf_long(outfile, fmt, SUNFALSE, "Stability limited steps", nst_exp);
    sunfprintf_long(outfile, fmt, SUNFALSE, "Accuracy limited steps", nst_acc);
    sunfprintf_long(outfile, fmt, SUNFALSE, "Error test fails", netf);
    sunfprintf_long(outfile, fmt, SUNFALSE, "NLS step fails", ncfn);
    sunfprintf_long(
        outfile,
        fmt,
        SUNFALSE,
        "Inequality constraint fails",
        nconstrfails,
    );
    sunfprintf_real(outfile, fmt, SUNFALSE, "Initial step size", h0u);
    sunfprintf_real(outfile, fmt, SUNFALSE, "Last step size", hold);
    sunfprintf_real(outfile, fmt, SUNFALSE, "Current step size", next_h);
    if let Some(nge) = nge {
        sunfprintf_long(outfile, fmt, SUNFALSE, "Root fn evals", nge);
    }

    /* Print relaxation stats */
    if relax_enabled {
        let retval = arkRelaxPrintAllStats(arkode_mem, outfile, fmt);
        if retval != ARK_SUCCESS {
            return retval;
        }
    }

    /* Print stepper stats (if provided) */
    let step_printallstats = arkode_mem.borrow().step_printallstats;
    if let Some(step_printallstats) = step_printallstats {
        return step_printallstats(arkode_mem, outfile, fmt);
    }

    ARK_SUCCESS
}

/*-----------------------------------------------------------------*/

/// C returns a `malloc`'d `char*`; the Rust port returns `String`.
pub fn ARKodeGetReturnFlagName(flag: i64) -> String {
    let name = match flag {
        f if f == ARK_SUCCESS as i64 => "ARK_SUCCESS",
        f if f == ARK_TSTOP_RETURN as i64 => "ARK_TSTOP_RETURN",
        f if f == ARK_ROOT_RETURN as i64 => "ARK_ROOT_RETURN",
        f if f == ARK_WARNING as i64 => "ARK_WARNING",
        f if f == ARK_TOO_MUCH_WORK as i64 => "ARK_TOO_MUCH_WORK",
        f if f == ARK_TOO_MUCH_ACC as i64 => "ARK_TOO_MUCH_ACC",
        f if f == ARK_ERR_FAILURE as i64 => "ARK_ERR_FAILURE",
        f if f == ARK_CONV_FAILURE as i64 => "ARK_CONV_FAILURE",
        f if f == ARK_LINIT_FAIL as i64 => "ARK_LINIT_FAIL",
        f if f == ARK_LSETUP_FAIL as i64 => "ARK_LSETUP_FAIL",
        f if f == ARK_LSOLVE_FAIL as i64 => "ARK_LSOLVE_FAIL",
        f if f == ARK_RHSFUNC_FAIL as i64 => "ARK_RHSFUNC_FAIL",
        f if f == ARK_FIRST_RHSFUNC_ERR as i64 => "ARK_FIRST_RHSFUNC_ERR",
        f if f == ARK_REPTD_RHSFUNC_ERR as i64 => "ARK_REPTD_RHSFUNC_ERR",
        f if f == ARK_UNREC_RHSFUNC_ERR as i64 => "ARK_UNREC_RHSFUNC_ERR",
        f if f == ARK_RTFUNC_FAIL as i64 => "ARK_RTFUNC_FAIL",
        f if f == ARK_LFREE_FAIL as i64 => "ARK_LFREE_FAIL",
        f if f == ARK_MASSINIT_FAIL as i64 => "ARK_MASSINIT_FAIL",
        f if f == ARK_MASSSETUP_FAIL as i64 => "ARK_MASSSETUP_FAIL",
        f if f == ARK_MASSSOLVE_FAIL as i64 => "ARK_MASSSOLVE_FAIL",
        f if f == ARK_MASSFREE_FAIL as i64 => "ARK_MASSFREE_FAIL",
        f if f == ARK_MASSMULT_FAIL as i64 => "ARK_MASSMULT_FAIL",
        f if f == ARK_CONSTR_FAIL as i64 => "ARK_CONSTR_FAIL",
        f if f == ARK_MEM_FAIL as i64 => "ARK_MEM_FAIL",
        f if f == ARK_MEM_NULL as i64 => "ARK_MEM_NULL",
        f if f == ARK_ILL_INPUT as i64 => "ARK_ILL_INPUT",
        f if f == ARK_NO_MALLOC as i64 => "ARK_NO_MALLOC",
        f if f == ARK_BAD_K as i64 => "ARK_BAD_K",
        f if f == ARK_BAD_T as i64 => "ARK_BAD_T",
        f if f == ARK_BAD_DKY as i64 => "ARK_BAD_DKY",
        f if f == ARK_TOO_CLOSE as i64 => "ARK_TOO_CLOSE",
        f if f == ARK_VECTOROP_ERR as i64 => "ARK_VECTOROP_ERR",
        f if f == ARK_NLS_INIT_FAIL as i64 => "ARK_NLS_INIT_FAIL",
        f if f == ARK_NLS_SETUP_FAIL as i64 => "ARK_NLS_SETUP_FAIL",
        f if f == ARK_NLS_SETUP_RECVR as i64 => "ARK_NLS_SETUP_RECVR",
        f if f == ARK_NLS_OP_ERR as i64 => "ARK_NLS_OP_ERR",
        f if f == ARK_INNERSTEP_ATTACH_ERR as i64 => "ARK_INNERSTEP_ATTACH_ERR",
        f if f == ARK_INNERSTEP_FAIL as i64 => "ARK_INNERSTEP_FAIL",
        f if f == ARK_OUTERTOINNER_FAIL as i64 => "ARK_OUTERTOINNER_FAIL",
        f if f == ARK_INNERTOOUTER_FAIL as i64 => "ARK_INNERTOOUTER_FAIL",
        /* ARK_POSTPROCESS_FAIL is the same value as ARK_POSTPROCESS_STEP_FAIL */
        f if f == ARK_POSTPROCESS_STEP_FAIL as i64 => "ARK_POSTPROCESS_STEP_FAIL",
        f if f == ARK_POSTPROCESS_STAGE_FAIL as i64 => "ARK_POSTPROCESS_STAGE_FAIL",
        f if f == ARK_PRESTEPFN_FAIL as i64 => "ARK_PRESTEPFN_FAIL",
        f if f == ARK_POSTSTEPFN_FAIL as i64 => "ARK_POSTSTEPFN_FAIL",
        f if f == ARK_PRERHSFN_FAIL as i64 => "ARK_PRERHSFN_FAIL",
        f if f == ARK_USER_PREDICT_FAIL as i64 => "ARK_USER_PREDICT_FAIL",
        f if f == ARK_INTERP_FAIL as i64 => "ARK_INTERP_FAIL",
        f if f == ARK_INVALID_TABLE as i64 => "ARK_INVALID_TABLE",
        f if f == ARK_CONTEXT_ERR as i64 => "ARK_CONTEXT_ERR",
        f if f == ARK_RELAX_FAIL as i64 => "ARK_RELAX_FAIL",
        f if f == ARK_RELAX_MEM_NULL as i64 => "ARK_RELAX_MEM_NULL",
        f if f == ARK_RELAX_FUNC_FAIL as i64 => "ARK_RELAX_FUNC_FAIL",
        f if f == ARK_RELAX_JAC_FAIL as i64 => "ARK_RELAX_JAC_FAIL",
        f if f == ARK_CONTROLLER_ERR as i64 => "ARK_CONTROLLER_ERR",
        f if f == ARK_STEPPER_UNSUPPORTED as i64 => "ARK_STEPPER_UNSUPPORTED",
        f if f == ARK_ADJ_RECOMPUTE_FAIL as i64 => "ARK_ADJ_RECOMPUTE_FAIL",
        f if f == ARK_ADJ_CHECKPOINT_FAIL as i64 => "ARK_ADJ_CHECKPOINT_FAIL",
        f if f == ARK_SUNADJSTEPPER_ERR as i64 => "ARK_SUNADJSTEPPER_ERR",
        f if f == ARK_DOMEIG_FAIL as i64 => "ARK_DOMEIG_FAIL",
        f if f == ARK_MAX_STAGE_LIMIT_FAIL as i64 => "ARK_MAX_STAGE_LIMIT_FAIL",
        f if f == ARK_SUNSTEPPER_ERR as i64 => "ARK_SUNSTEPPER_ERR",
        f if f == ARK_STEP_DIRECTION_ERR as i64 => "ARK_STEP_DIRECTION_ERR",
        f if f == ARK_UNRECOGNIZED_ERROR as i64 => "ARK_UNRECOGNIZED_ERROR",
        f if f == ARK_STEP_H0_FAIL as i64 => "ARK_STEP_H0_FAIL",
        _ => "NONE",
    };

    name.to_string()
}

/*===============================================================
  ARKODE parameter output utility routine
  ===============================================================*/

/*---------------------------------------------------------------
  ARKodeWriteParameters:

  Outputs all solver parameters to the provided file pointer.
  ---------------------------------------------------------------*/
pub fn ARKodeWriteParameters(arkode_mem: &ARKodeMem, fp: &SUNFile) -> i32 {
    /* NULL-mem check: handled by type system */

    /* Copy the parameters out under one borrow (writing must not hold a
    borrow of the mem), then print in the exact C order. */
    let (
        hmin,
        hmax_inv,
        fixedstep,
        itol,
        reltol,
        Sabstol,
        rwt_is_ewt,
        ritol,
        SRabstol,
        hin,
        etamx1,
        etamxf,
        small_nef,
        etacf,
        cfl,
        safety,
        growth,
        lbound,
        ubound,
        expstab_is_none,
        hcontroller,
        maxnef,
        maxncf,
    );
    {
        let ark_mem = arkode_mem.borrow();
        hmin = ark_mem.hmin;
        hmax_inv = ark_mem.hmax_inv;
        fixedstep = ark_mem.fixedstep;
        itol = ark_mem.itol;
        reltol = ark_mem.reltol;
        Sabstol = ark_mem.Sabstol;
        rwt_is_ewt = ark_mem.rwt_is_ewt;
        ritol = ark_mem.ritol;
        SRabstol = ark_mem.SRabstol;
        hin = ark_mem.hin;
        let hadapt_mem = ark_mem.hadapt_mem.as_ref().expect("hadapt_mem allocated");
        etamx1 = hadapt_mem.etamx1;
        etamxf = hadapt_mem.etamxf;
        small_nef = hadapt_mem.small_nef;
        etacf = hadapt_mem.etacf;
        cfl = hadapt_mem.cfl;
        safety = hadapt_mem.safety;
        growth = hadapt_mem.growth;
        lbound = hadapt_mem.lbound;
        ubound = hadapt_mem.ubound;
        expstab_is_none = hadapt_mem.expstab.is_none();
        hcontroller = hadapt_mem.hcontroller.clone();
        maxnef = ark_mem.maxnef;
        maxncf = ark_mem.maxncf;
    }

    /* print integrator parameters to file */
    fp.write_str("ARKODE solver parameters:\n");
    if hmin != ZERO {
        fp.write_str(&format!("  Minimum step size = {}\n", sun_format_g(hmin)));
    }
    if hmax_inv != ZERO {
        fp.write_str(&format!(
            "  Maximum step size = {}\n",
            sun_format_g(ONE / hmax_inv)
        ));
    }
    if fixedstep {
        fp.write_str("  Fixed time-stepping enabled\n");
    }
    if itol == ARK_WF {
        fp.write_str("  User provided error weight function\n");
    } else {
        fp.write_str(&format!(
            "  Solver relative tolerance = {}\n",
            sun_format_g(reltol)
        ));
        if itol == ARK_SS {
            fp.write_str(&format!(
                "  Solver absolute tolerance = {}\n",
                sun_format_g(Sabstol)
            ));
        } else {
            fp.write_str("  Vector-valued solver absolute tolerance\n");
        }
    }
    if !rwt_is_ewt {
        if ritol == ARK_WF {
            fp.write_str("  User provided residual weight function\n");
        } else if ritol == ARK_SS {
            fp.write_str(&format!(
                "  Absolute residual tolerance = {}\n",
                sun_format_g(SRabstol)
            ));
        } else {
            fp.write_str("  Vector-valued residual absolute tolerance\n");
        }
    }
    if hin != ZERO {
        fp.write_str(&format!("  Initial step size = {}\n", sun_format_g(hin)));
    }
    fp.write_str("\n");
    fp.write_str(&format!(
        "  Maximum step increase (first step) = {}\n",
        sun_format_g(etamx1)
    ));
    fp.write_str(&format!(
        "  Step reduction factor on multiple error fails = {}\n",
        sun_format_g(etamxf)
    ));
    fp.write_str(&format!(
        "  Minimum error fails before above factor is used = {}\n",
        small_nef
    ));
    fp.write_str(&format!(
        "  Step reduction factor on nonlinear convergence failure = {}\n",
        sun_format_g(etacf)
    ));
    fp.write_str(&format!(
        "  Explicit safety factor = {}\n",
        sun_format_g(cfl)
    ));
    fp.write_str(&format!("  Safety factor = {}\n", sun_format_g(safety)));
    fp.write_str(&format!("  Growth factor = {}\n", sun_format_g(growth)));
    fp.write_str(&format!(
        "  Step growth lower bound = {}\n",
        sun_format_g(lbound)
    ));
    fp.write_str(&format!(
        "  Step growth upper bound = {}\n",
        sun_format_g(ubound)
    ));
    if expstab_is_none {
        fp.write_str("  No explicit stability function supplied\n");
    } else {
        fp.write_str("  User provided explicit stability function\n");
    }
    if let Some(hcontroller) = hcontroller {
        let _ = SUNAdaptController_Write(&hcontroller, fp);
    }

    fp.write_str(&format!(
        "  Maximum number of error test failures = {}\n",
        maxnef
    ));
    fp.write_str(&format!(
        "  Maximum number of convergence test failures = {}\n",
        maxncf
    ));

    /* Call stepper routine (if provided) */
    let step_writeparameters = arkode_mem.borrow().step_writeparameters;
    if let Some(step_writeparameters) = step_writeparameters {
        return step_writeparameters(arkode_mem, fp);
    }

    ARK_SUCCESS
}

/*===============================================================
  ARKODE-IO internal utility functions
  ===============================================================*/

/*---------------------------------------------------------------
  arkReplaceAdaptController

  Replaces the current SUNAdaptController time step controller
  object. If a NULL-valued SUNAdaptController is input, the
  default will be re-enabled.
  ---------------------------------------------------------------*/
pub fn arkReplaceAdaptController(
    ark_mem: &ARKodeMem,
    C: Option<&SUNAdaptController>,
    take_ownership: sunbooleantype,
) -> i32 {
    let mut lenrw: i64 = 0;
    let mut leniw: i64 = 0;

    /* Remove current SUNAdaptController object
    (delete if owned, and then nullify pointer) */
    let (owncontroller, has_controller) = {
        let m = ark_mem.borrow();
        let hadapt_mem = m.hadapt_mem.as_ref().expect("hadapt_mem allocated");
        (hadapt_mem.owncontroller, hadapt_mem.hcontroller.is_some())
    };
    if owncontroller && has_controller {
        {
            let hcontroller = ark_mem
                .borrow()
                .hadapt_mem
                .as_ref()
                .expect("hadapt_mem allocated")
                .hcontroller
                .clone()
                .expect("hcontroller");
            let retval = SUNAdaptController_Space(&hcontroller, &mut lenrw, &mut leniw);
            if retval == SUN_SUCCESS {
                let mut m = ark_mem.borrow_mut();
                m.liw -= leniw;
                m.lrw -= lenrw;
            }
        }

        let owned = ark_mem
            .borrow_mut()
            .hadapt_mem
            .as_mut()
            .expect("hadapt_mem allocated")
            .hcontroller
            .take();
        let retval = SUNAdaptController_Destroy(owned);
        ark_mem
            .borrow_mut()
            .hadapt_mem
            .as_mut()
            .expect("hadapt_mem allocated")
            .owncontroller = SUNFALSE;
        if retval != SUN_SUCCESS {
            arkProcessError(
                Some(ark_mem),
                ARK_MEM_FAIL,
                line!() as i32,
                "arkReplaceAdaptController",
                file!(),
                "SUNAdaptController_Destroy failure",
            );
            return ARK_MEM_FAIL;
        }
    }
    ark_mem
        .borrow_mut()
        .hadapt_mem
        .as_mut()
        .expect("hadapt_mem allocated")
        .hcontroller = None;

    /* On NULL-valued input, create default SUNAdaptController object */
    let C: SUNAdaptController = match C {
        None => {
            let sunctx = ark_mem.borrow().sunctx.clone();
            let C = SUNAdaptController_I(&sunctx);
            let C = match C {
                Some(C) => C,
                None => {
                    arkProcessError(
                        Some(ark_mem),
                        ARK_MEM_FAIL,
                        line!() as i32,
                        "arkReplaceAdaptController",
                        file!(),
                        "SUNAdaptController_I allocation failure",
                    );
                    return ARK_MEM_FAIL;
                }
            };
            ark_mem
                .borrow_mut()
                .hadapt_mem
                .as_mut()
                .expect("hadapt_mem allocated")
                .owncontroller = SUNTRUE;
            C
        }
        Some(C) => {
            ark_mem
                .borrow_mut()
                .hadapt_mem
                .as_mut()
                .expect("hadapt_mem allocated")
                .owncontroller = take_ownership;
            C.clone()
        }
    };

    /* Attach new SUNAdaptController object */
    let retval = SUNAdaptController_Space(&C, &mut lenrw, &mut leniw);
    if retval == SUN_SUCCESS {
        let mut m = ark_mem.borrow_mut();
        m.liw += leniw;
        m.lrw += lenrw;
    }
    ark_mem
        .borrow_mut()
        .hadapt_mem
        .as_mut()
        .expect("hadapt_mem allocated")
        .hcontroller = Some(C);

    ARK_SUCCESS
}

/*===============================================================
  ARKODE + XBraid interface utility functions
  ===============================================================*/

/*---------------------------------------------------------------
  arkSetForcePass:

  Ignore the value of kflag after the temporal error test and
  force the step to pass.
  ---------------------------------------------------------------*/
pub fn arkSetForcePass(arkode_mem: &ARKodeMem, force_pass: sunbooleantype) -> i32 {
    /* NULL-mem check: handled by type system */

    arkode_mem.borrow_mut().force_pass = force_pass;

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  arkGetLastKFlag:

  The last kflag value returned by the temporal error test.
  ---------------------------------------------------------------*/
pub fn arkGetLastKFlag(arkode_mem: &ARKodeMem, last_kflag: &mut i32) -> i32 {
    /* NULL-mem check: handled by type system */

    *last_kflag = arkode_mem.borrow().last_kflag;

    ARK_SUCCESS
}

/*===============================================================
  Deprecated functions
  ===============================================================*/

/*---------------------------------------------------------------
  arkSetAdaptivityMethod:

  Specifies the built-in time step adaptivity algorithm (and
  optionally, its associated parameters) to use.  All parameters
  will be checked for validity when used by the solver.

  Users should transition to constructing non-default SUNAdaptController
  objects directly, and providing those directly to the integrator
  via the time-stepping module *SetController routines.
  ---------------------------------------------------------------*/
pub fn arkSetAdaptivityMethod(
    arkode_mem: &ARKodeMem,
    imethod: i32,
    idefault: i32,
    pq: i32,
    adapt_params: Option<&[sunrealtype; 3]>,
) -> i32 {
    let mut lenrw: i64 = 0;
    let mut leniw: i64 = 0;
    /* NULL-mem check: handled by type system */

    /* Check for illegal inputs */
    if (idefault != 1) && adapt_params.is_none() {
        arkProcessError(
            Some(arkode_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "arkSetAdaptivityMethod",
            file!(),
            "NULL-valued adapt_params provided",
        );
        return ARK_ILL_INPUT;
    }

    /* Remove current SUNAdaptController object
    (delete if owned, and then nullify pointer) */
    let (owncontroller, has_controller) = {
        let m = arkode_mem.borrow();
        let hadapt_mem = m.hadapt_mem.as_ref().expect("hadapt_mem allocated");
        (hadapt_mem.owncontroller, hadapt_mem.hcontroller.is_some())
    };
    if owncontroller && has_controller {
        {
            let hcontroller = arkode_mem
                .borrow()
                .hadapt_mem
                .as_ref()
                .expect("hadapt_mem allocated")
                .hcontroller
                .clone()
                .expect("hcontroller");
            let retval = SUNAdaptController_Space(&hcontroller, &mut lenrw, &mut leniw);
            if retval == SUN_SUCCESS {
                let mut m = arkode_mem.borrow_mut();
                m.liw -= leniw;
                m.lrw -= lenrw;
            }
        }

        let owned = arkode_mem
            .borrow_mut()
            .hadapt_mem
            .as_mut()
            .expect("hadapt_mem allocated")
            .hcontroller
            .take();
        let retval = SUNAdaptController_Destroy(owned);
        arkode_mem
            .borrow_mut()
            .hadapt_mem
            .as_mut()
            .expect("hadapt_mem allocated")
            .owncontroller = SUNFALSE;
        if retval != SUN_SUCCESS {
            arkProcessError(
                Some(arkode_mem),
                ARK_MEM_FAIL,
                line!() as i32,
                "arkSetAdaptivityMethod",
                file!(),
                "SUNAdaptController_Destroy failure",
            );
            return ARK_MEM_FAIL;
        }
    }
    arkode_mem
        .borrow_mut()
        .hadapt_mem
        .as_mut()
        .expect("hadapt_mem allocated")
        .hcontroller = None;

    /* set adaptivity parameters from inputs */
    let mut k1 = ZERO;
    let mut k2 = ZERO;
    let mut k3 = ZERO;
    if idefault != 1 {
        let adapt_params = adapt_params.expect("adapt_params");
        k1 = adapt_params[0];
        k2 = adapt_params[1];
        k3 = adapt_params[2];
    }
    arkode_mem
        .borrow_mut()
        .hadapt_mem
        .as_mut()
        .expect("hadapt_mem allocated")
        .pq = pq;

    /* Create new SUNAdaptController object based on "imethod" input, optionally setting
    the specified controller parameters */
    let sunctx = arkode_mem.borrow().sunctx.clone();
    let C: SUNAdaptController;
    match imethod {
        ARK_ADAPT_PID => {
            let Cnew = SUNAdaptController_PID(&sunctx);
            let Cnew = match Cnew {
                Some(Cnew) => Cnew,
                None => {
                    arkProcessError(
                        Some(arkode_mem),
                        ARK_MEM_FAIL,
                        line!() as i32,
                        "arkSetAdaptivityMethod",
                        file!(),
                        "SUNAdaptController_PID allocation failure",
                    );
                    return ARK_MEM_FAIL;
                }
            };
            if idefault != 1 {
                let retval = SUNAdaptController_SetParams_PID(&Cnew, k1, -k2, k3);
                if retval != SUN_SUCCESS {
                    let _ = SUNAdaptController_Destroy(Some(Cnew));
                    arkProcessError(
                        Some(arkode_mem),
                        ARK_CONTROLLER_ERR,
                        line!() as i32,
                        "arkSetAdaptivityMethod",
                        file!(),
                        "SUNAdaptController_SetParams_PID failure",
                    );
                    return ARK_CONTROLLER_ERR;
                }
            }
            C = Cnew;
        }
        ARK_ADAPT_PI => {
            let Cnew = SUNAdaptController_PI(&sunctx);
            let Cnew = match Cnew {
                Some(Cnew) => Cnew,
                None => {
                    arkProcessError(
                        Some(arkode_mem),
                        ARK_MEM_FAIL,
                        line!() as i32,
                        "arkSetAdaptivityMethod",
                        file!(),
                        "SUNAdaptController_PI allocation failure",
                    );
                    return ARK_MEM_FAIL;
                }
            };
            if idefault != 1 {
                let retval = SUNAdaptController_SetParams_PI(&Cnew, k1, -k2);
                if retval != SUN_SUCCESS {
                    let _ = SUNAdaptController_Destroy(Some(Cnew));
                    arkProcessError(
                        Some(arkode_mem),
                        ARK_CONTROLLER_ERR,
                        line!() as i32,
                        "arkSetAdaptivityMethod",
                        file!(),
                        "SUNAdaptController_SetParams_PI failure",
                    );
                    return ARK_CONTROLLER_ERR;
                }
            }
            C = Cnew;
        }
        ARK_ADAPT_I => {
            let Cnew = SUNAdaptController_I(&sunctx);
            let Cnew = match Cnew {
                Some(Cnew) => Cnew,
                None => {
                    arkProcessError(
                        Some(arkode_mem),
                        ARK_MEM_FAIL,
                        line!() as i32,
                        "arkSetAdaptivityMethod",
                        file!(),
                        "SUNAdaptController_I allocation failure",
                    );
                    return ARK_MEM_FAIL;
                }
            };
            if idefault != 1 {
                let retval = SUNAdaptController_SetParams_I(&Cnew, k1);
                if retval != SUN_SUCCESS {
                    let _ = SUNAdaptController_Destroy(Some(Cnew));
                    arkProcessError(
                        Some(arkode_mem),
                        ARK_CONTROLLER_ERR,
                        line!() as i32,
                        "arkSetAdaptivityMethod",
                        file!(),
                        "SUNAdaptController_SetParams_I failure",
                    );
                    return ARK_CONTROLLER_ERR;
                }
            }
            C = Cnew;
        }
        ARK_ADAPT_EXP_GUS => {
            let Cnew = SUNAdaptController_ExpGus(&sunctx);
            let Cnew = match Cnew {
                Some(Cnew) => Cnew,
                None => {
                    arkProcessError(
                        Some(arkode_mem),
                        ARK_MEM_FAIL,
                        line!() as i32,
                        "arkSetAdaptivityMethod",
                        file!(),
                        "SUNAdaptController_ExpGus allocation failure",
                    );
                    return ARK_MEM_FAIL;
                }
            };
            if idefault != 1 {
                let retval = SUNAdaptController_SetParams_ExpGus(&Cnew, k1, k2);
                if retval != SUN_SUCCESS {
                    let _ = SUNAdaptController_Destroy(Some(Cnew));
                    arkProcessError(
                        Some(arkode_mem),
                        ARK_CONTROLLER_ERR,
                        line!() as i32,
                        "arkSetAdaptivityMethod",
                        file!(),
                        "SUNAdaptController_SetParams_ExpGus failure",
                    );
                    return ARK_CONTROLLER_ERR;
                }
            }
            C = Cnew;
        }
        ARK_ADAPT_IMP_GUS => {
            let Cnew = SUNAdaptController_ImpGus(&sunctx);
            let Cnew = match Cnew {
                Some(Cnew) => Cnew,
                None => {
                    arkProcessError(
                        Some(arkode_mem),
                        ARK_MEM_FAIL,
                        line!() as i32,
                        "arkSetAdaptivityMethod",
                        file!(),
                        "SUNAdaptController_ImpGus allocation failure",
                    );
                    return ARK_MEM_FAIL;
                }
            };
            if idefault != 1 {
                let retval = SUNAdaptController_SetParams_ImpGus(&Cnew, k1, k2);
                if retval != SUN_SUCCESS {
                    let _ = SUNAdaptController_Destroy(Some(Cnew));
                    arkProcessError(
                        Some(arkode_mem),
                        ARK_CONTROLLER_ERR,
                        line!() as i32,
                        "arkSetAdaptivityMethod",
                        file!(),
                        "SUNAdaptController_SetParams_ImpGus failure",
                    );
                    return ARK_CONTROLLER_ERR;
                }
            }
            C = Cnew;
        }
        ARK_ADAPT_IMEX_GUS => {
            let Cnew = SUNAdaptController_ImExGus(&sunctx);
            let Cnew = match Cnew {
                Some(Cnew) => Cnew,
                None => {
                    arkProcessError(
                        Some(arkode_mem),
                        ARK_MEM_FAIL,
                        line!() as i32,
                        "arkSetAdaptivityMethod",
                        file!(),
                        "SUNAdaptController_ImExGus allocation failure",
                    );
                    return ARK_MEM_FAIL;
                }
            };
            if idefault != 1 {
                let retval = SUNAdaptController_SetParams_ImExGus(&Cnew, k1, k2, k3, k3);
                if retval != SUN_SUCCESS {
                    let _ = SUNAdaptController_Destroy(Some(Cnew));
                    arkProcessError(
                        Some(arkode_mem),
                        ARK_CONTROLLER_ERR,
                        line!() as i32,
                        "arkSetAdaptivityMethod",
                        file!(),
                        "SUNAdaptController_SetParams_ImExGus failure",
                    );
                    return ARK_CONTROLLER_ERR;
                }
            }
            C = Cnew;
        }
        _ => {
            arkProcessError(
                Some(arkode_mem),
                ARK_ILL_INPUT,
                line!() as i32,
                "arkSetAdaptivityMethod",
                file!(),
                "Illegal imethod",
            );
            return ARK_ILL_INPUT;
        }
    }

    /* Attach new SUNAdaptController object */
    let retval = SUNAdaptController_Space(&C, &mut lenrw, &mut leniw);
    if retval == SUN_SUCCESS {
        let mut m = arkode_mem.borrow_mut();
        m.liw += leniw;
        m.lrw += lenrw;
    }
    {
        let mut m = arkode_mem.borrow_mut();
        let hadapt_mem = m.hadapt_mem.as_mut().expect("hadapt_mem allocated");
        hadapt_mem.hcontroller = Some(C);
        hadapt_mem.owncontroller = SUNTRUE;
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  arkSetAdaptivityFn:

  Specifies the user-provided time step adaptivity function to use.
  If 'hfun' is NULL-valued, then the default I controller will
  be used instead.

  Users should transition to constructing a custom SUNAdaptController
  object, and providing this directly to the integrator
  via the time-stepping module *SetController routines.
  ---------------------------------------------------------------*/
pub fn arkSetAdaptivityFn(
    arkode_mem: &ARKodeMem,
    hfun: Option<ARKAdaptFn>,
    h_data: Option<Box<dyn Any>>,
) -> i32 {
    let mut lenrw: i64 = 0;
    let mut leniw: i64 = 0;
    /* NULL-mem check: handled by type system */

    /* Remove current SUNAdaptController object
    (delete if owned, and then nullify pointer) */
    let (owncontroller, has_controller) = {
        let m = arkode_mem.borrow();
        let hadapt_mem = m.hadapt_mem.as_ref().expect("hadapt_mem allocated");
        (hadapt_mem.owncontroller, hadapt_mem.hcontroller.is_some())
    };
    if owncontroller && has_controller {
        {
            let hcontroller = arkode_mem
                .borrow()
                .hadapt_mem
                .as_ref()
                .expect("hadapt_mem allocated")
                .hcontroller
                .clone()
                .expect("hcontroller");
            let retval = SUNAdaptController_Space(&hcontroller, &mut lenrw, &mut leniw);
            if retval == SUN_SUCCESS {
                let mut m = arkode_mem.borrow_mut();
                m.liw -= leniw;
                m.lrw -= lenrw;
            }
        }

        let owned = arkode_mem
            .borrow_mut()
            .hadapt_mem
            .as_mut()
            .expect("hadapt_mem allocated")
            .hcontroller
            .take();
        let retval = SUNAdaptController_Destroy(owned);
        arkode_mem
            .borrow_mut()
            .hadapt_mem
            .as_mut()
            .expect("hadapt_mem allocated")
            .owncontroller = SUNFALSE;
        if retval != SUN_SUCCESS {
            arkProcessError(
                Some(arkode_mem),
                ARK_MEM_FAIL,
                line!() as i32,
                "arkSetAdaptivityFn",
                file!(),
                "SUNAdaptController_Destroy failure",
            );
            return ARK_MEM_FAIL;
        }
    }
    arkode_mem
        .borrow_mut()
        .hadapt_mem
        .as_mut()
        .expect("hadapt_mem allocated")
        .hcontroller = None;

    /* Create new SUNAdaptController object depending on NULL-ity of 'hfun' */
    let sunctx = arkode_mem.borrow().sunctx.clone();
    let C: SUNAdaptController = if hfun.is_none() {
        match SUNAdaptController_I(&sunctx) {
            Some(C) => C,
            None => {
                arkProcessError(
                    Some(arkode_mem),
                    ARK_MEM_FAIL,
                    line!() as i32,
                    "arkSetAdaptivityFn",
                    file!(),
                    "SUNAdaptController_I allocation failure",
                );
                return ARK_MEM_FAIL;
            }
        }
    } else {
        match ARKUserControl(&sunctx, arkode_mem, hfun, h_data) {
            Some(C) => C,
            None => {
                arkProcessError(
                    Some(arkode_mem),
                    ARK_MEM_FAIL,
                    line!() as i32,
                    "arkSetAdaptivityFn",
                    file!(),
                    "ARKUserControl allocation failure",
                );
                return ARK_MEM_FAIL;
            }
        }
    };

    /* Attach new SUNAdaptController object */
    let retval = SUNAdaptController_Space(&C, &mut lenrw, &mut leniw);
    if retval == SUN_SUCCESS {
        let mut m = arkode_mem.borrow_mut();
        m.liw += leniw;
        m.lrw += lenrw;
    }
    {
        let mut m = arkode_mem.borrow_mut();
        let hadapt_mem = m.hadapt_mem.as_mut().expect("hadapt_mem allocated");
        hadapt_mem.hcontroller = Some(C);
        hadapt_mem.owncontroller = SUNTRUE;
    }

    ARK_SUCCESS
}

/*===============================================================
  EOF
  ===============================================================*/
