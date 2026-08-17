//! Port of `src/arkode/arkode_mristep_io.c`: the optional input and output
//! functions for the ARKODE MRIStep time stepper module.
//!
//! Conventions used throughout (all fixed by `arkode_impl.rs` / ARCHITECTURE):
//!  * `mriStep_AccessARKODEStepMem` / `mriStep_AccessStepMem` become inline
//!    presence checks (`ark_mem.borrow().step_mem.is_none()` + the same
//!    `MSG_MRISTEP_NO_MEM` error, reported under the *calling* function's
//!    name exactly as C's `fname` argument does) followed by
//!    `mriStep_mem_mut(...)` at each use site.
//!  * `FILE*` -> `&SUNFile`, `fprintf` -> `SUNFile::write_str` +
//!    `format!`, `SUN_FORMAT_G` -> `sun_format_g`.
//!  * C `void* step_mem->lmem` is `ark_mem.ark_lmem` in this port (see the
//!    `ARKodeMemRec::ark_lmem` docs); `step_getlinmem` is a presence probe
//!    and `arkls_mem_mut` reaches the record itself.
//!  * `arkProcessError` varargs map to a pre-formatted message.
//!  * NULL checks on parameters that became `&T` / `&mut T` are unreachable
//!    and dropped (noted at each site).

use std::any::Any;

use crate::arkode::*;
use crate::arkode_impl::*;
use crate::arkode_io::*;
use crate::arkode_ls::*;
use crate::arkode_mri_tables::*;
use crate::arkode_mristep::*;
use crate::arkode_mristep_controller::SUNAdaptController_MRIStep;
use crate::arkode_root::ARKodeRootInit;

use sundials_core::sundials_adaptcontroller::{
    SUNAdaptController, SUNAdaptController_GetType, SUN_ADAPTCONTROLLER_MRI_H_TOL,
    SUN_ADAPTCONTROLLER_NONE,
};
use sundials_core::sundials_errors::SUN_SUCCESS;
use sundials_core::sundials_linearsolver::SUNLinearSolver;
use sundials_core::sundials_matrix::SUNMatrix;
use sundials_core::sundials_nonlinearsolver::{
    SUNNonlinSolFree, SUNNonlinSolSetMaxIters, SUNNonlinearSolver,
};
use sundials_core::sundials_nvector::{N_VScale, N_Vector};
use sundials_core::sundials_types::*;
use sundials_core::sundials_utils::{sun_format_g, sunfprintf_long, sunfprintf_real, SUNFile};

/*===============================================================
  Exported optional input functions.
  ===============================================================*/

/*---------------------------------------------------------------
  MRIStepSetCoupling:

  Specifies to use a customized coupling structure for the slow
  portion of the system.
  ---------------------------------------------------------------*/
pub fn MRIStepSetCoupling(arkode_mem: &ARKodeMem, MRIC: &MRIStepCoupling) -> i32 {
    let ark_mem = arkode_mem;
    let mut Tlrw: sunindextype = 0;
    let mut Tliw: sunindextype = 0;

    /* access ARKodeMem and ARKodeMRIStepMem structures */
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "MRIStepSetCoupling",
            file!(),
            MSG_MRISTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    /* check for illegal inputs: `MRIC == NULL` is handled by the type system */

    /* clear any existing parameters and coupling structure */
    {
        let mut step_mem = mriStep_mem_mut(ark_mem);
        step_mem.stages = 0;
        step_mem.q = 0;
        step_mem.p = 0;
    }
    let old_MRIC = mriStep_mem_mut(ark_mem).MRIC.take();
    MRIStepCoupling_Space(old_MRIC.as_ref(), &mut Tliw, &mut Tlrw);
    MRIStepCoupling_Free(old_MRIC);
    {
        let mut m = ark_mem.borrow_mut();
        m.liw -= Tliw;
        m.lrw -= Tlrw;
    }

    /* set the relevant parameters */
    {
        let (stages, q, p) = {
            let c = MRIC.borrow();
            (c.stages, c.q, c.p)
        };
        let mut step_mem = mriStep_mem_mut(ark_mem);
        step_mem.stages = stages;
        step_mem.q = q;
        step_mem.p = p;
    }

    /* copy the coupling structure in step memory */
    let copy = MRIStepCoupling_Copy(Some(MRIC));
    let copy_failed = copy.is_none();
    mriStep_mem_mut(ark_mem).MRIC = copy;
    if copy_failed {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "MRIStepSetCoupling",
            file!(),
            MSG_MRISTEP_NO_COUPLING,
        );
        return ARK_MEM_NULL;
    }
    let new_MRIC = mriStep_mem_mut(ark_mem).MRIC.clone();
    MRIStepCoupling_Space(new_MRIC.as_ref(), &mut Tliw, &mut Tlrw);
    {
        let mut m = ark_mem.borrow_mut();
        m.liw += Tliw;
        m.lrw += Tlrw;
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  MRIStepSetPreInnerFn:

  Sets the user-supplied function called BEFORE the inner evolve
  ---------------------------------------------------------------*/
pub fn MRIStepSetPreInnerFn(arkode_mem: &ARKodeMem, prefn: Option<MRIStepPreInnerFn>) -> i32 {
    let ark_mem = arkode_mem;

    /* access ARKodeMem and ARKodeMRIStepMem structures */
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "MRIStepSetPreInnerFn",
            file!(),
            MSG_MRISTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    /* Set pre inner evolve function */
    mriStep_mem_mut(ark_mem).pre_inner_evolve = prefn;

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  MRIStepSetPostInnerFn:

  Sets the user-supplied function called AFTER the inner evolve
  ---------------------------------------------------------------*/
pub fn MRIStepSetPostInnerFn(arkode_mem: &ARKodeMem, postfn: Option<MRIStepPostInnerFn>) -> i32 {
    let ark_mem = arkode_mem;

    /* access ARKodeMem and ARKodeMRIStepMem structures */
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "MRIStepSetPostInnerFn",
            file!(),
            MSG_MRISTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    /* Set pre inner evolve function */
    mriStep_mem_mut(ark_mem).post_inner_evolve = postfn;

    ARK_SUCCESS
}

/*===============================================================
  Exported optional output functions.
  ===============================================================*/

/*---------------------------------------------------------------
  mriStep_GetNumRhsEvals:

  Returns the current number of RHS calls
  ---------------------------------------------------------------*/
pub fn mriStep_GetNumRhsEvals(
    ark_mem: &ARKodeMem,
    partition_index: i32,
    rhs_evals: &mut i64,
) -> i32 {
    /* access ARKodeMRIStepMem structure */
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "mriStep_GetNumRhsEvals",
            file!(),
            MSG_MRISTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    /* `rhs_evals == NULL` is handled by the type system */

    if partition_index > 1 {
        arkProcessError(
            Some(ark_mem),
            ARK_ILL_INPUT,
            line!() as i32,
            "mriStep_GetNumRhsEvals",
            file!(),
            "Invalid partition index",
        );
        return ARK_ILL_INPUT;
    }

    let (nfse, nfsi) = {
        let step_mem = mriStep_mem_mut(ark_mem);
        (step_mem.nfse, step_mem.nfsi)
    };
    match partition_index {
        0 => *rhs_evals = nfse,
        1 => *rhs_evals = nfsi,
        _ => *rhs_evals = nfse + nfsi,
    }

    ARK_SUCCESS
}

pub fn MRIStepGetNumRhsEvals(
    arkode_mem: &ARKodeMem,
    nfse_evals: &mut i64,
    nfsi_evals: &mut i64,
) -> i32 {
    let mut retval: i32;

    retval = ARKodeGetNumRhsEvals(arkode_mem, 0, nfse_evals);
    if retval != ARK_SUCCESS {
        return retval;
    }

    retval = ARKodeGetNumRhsEvals(arkode_mem, 1, nfsi_evals);
    if retval != ARK_SUCCESS {
        return retval;
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  MRIStepGetCurrentCoupling:

  Sets pointer to the slow coupling structure currently in use.
  ---------------------------------------------------------------*/
pub fn MRIStepGetCurrentCoupling(
    arkode_mem: &ARKodeMem,
    MRIC: &mut Option<MRIStepCoupling>,
) -> i32 {
    let ark_mem = arkode_mem;

    /* access ARKodeMem and ARKodeMRIStepMem structures */
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "MRIStepGetCurrentCoupling",
            file!(),
            MSG_MRISTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    /* get coupling structure from step_mem */
    *MRIC = mriStep_mem_mut(ark_mem).MRIC.clone();

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  MRIStepGetLastInnerStepFlag:

  Returns the last return value from the inner stepper.
  ---------------------------------------------------------------*/
pub fn MRIStepGetLastInnerStepFlag(arkode_mem: &ARKodeMem, flag: &mut i32) -> i32 {
    let ark_mem = arkode_mem;

    /* access ARKodeMem and ARKodeMRIStepMem structures */
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "MRIStepGetLastInnerStepFlag",
            file!(),
            MSG_MRISTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    /* get the last return value from the inner stepper */
    let stepper = mriStep_mem_mut(ark_mem).stepper.clone();
    *flag = *stepper.expect("stepper").last_flag.borrow();

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  MRIStepGetNumInnerStepperFails:

  Returns the number of recoverable failures encountered by the
  inner stepper.
  ---------------------------------------------------------------*/
pub fn MRIStepGetNumInnerStepperFails(arkode_mem: &ARKodeMem, inner_fails: &mut i64) -> i32 {
    let ark_mem = arkode_mem;

    /* access ARKodeMem and ARKodeMRIStepMem structures */
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "MRIStepGetNumInnerStepperFails",
            file!(),
            MSG_MRISTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    /* set output from step_mem */
    *inner_fails = mriStep_mem_mut(ark_mem).inner_fails;

    ARK_SUCCESS
}

/*===============================================================
  Private functions attached to ARKODE
  ===============================================================*/

/*---------------------------------------------------------------
  mriStep_SetOption:

  Provides command-line control over MRIStep-specific "set" routines.
  ---------------------------------------------------------------*/
pub fn mriStep_SetOptions(
    ark_mem: &ARKodeMem,
    argidx: &mut i32,
    argv: &[String],
    offset: usize,
    arg_used: &mut sunbooleantype,
) -> i32 {
    /* The only MRIStep-specific "Set" routine takes a custom MRIStepCoupling
       table; however, these may be specified by name, so here we'll support
       a key to specify the MRIStepCoupling table name,
       create the table with that name, attach it to MRIStep (who copies its
       values), and then free the table. */
    if &argv[*argidx as usize][offset..] == "coupling_table_name" {
        *argidx += 1;
        let Coupling = MRIStepCoupling_LoadTableByName(&argv[*argidx as usize]);
        let Coupling = match Coupling {
            None => {
                arkProcessError(
                    Some(ark_mem),
                    ARK_ILL_INPUT,
                    line!() as i32,
                    "mriStep_SetOptions",
                    file!(),
                    &format!(
                        "error setting key {} {} (invalid table name)",
                        argv[(*argidx as usize) - 1],
                        argv[*argidx as usize]
                    ),
                );
                return ARK_ILL_INPUT;
            }
            Some(Coupling) => Coupling,
        };
        let retval = MRIStepSetCoupling(ark_mem, &Coupling);
        MRIStepCoupling_Free(Some(Coupling));
        if retval != ARK_SUCCESS {
            arkProcessError(
                Some(ark_mem),
                retval,
                line!() as i32,
                "mriStep_SetOptions",
                file!(),
                &format!(
                    "error setting key {} {} (SetCoupling failed)",
                    argv[(*argidx as usize) - 1],
                    argv[*argidx as usize]
                ),
            );
            return retval;
        }
        *arg_used = SUNTRUE;
        return ARK_SUCCESS;
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  mriStep_SetAdaptController:

  Specifies a temporal adaptivity controller for MRIStep to use.
  If a non-MRI controller is provided, this just passes that
  through to arkReplaceAdaptController.  However, if an MRI
  controller is provided, then this wraps that inside a
  "SUNAdaptController_MRIStep" wrapper, which will properly
  interact with the fast integration module.
  ---------------------------------------------------------------*/
pub fn mriStep_SetAdaptController(ark_mem: &ARKodeMem, C: Option<&SUNAdaptController>) -> i32 {
    /* Retrieve the controller type (C `SUNAdaptController_GetType(NULL)`
       returns SUN_ADAPTCONTROLLER_NONE) */
    let ctype = match C {
        Some(C) => SUNAdaptController_GetType(C),
        None => SUN_ADAPTCONTROLLER_NONE,
    };

    /* If this does not have MRI type, then just pass to ARKODE */
    if ctype != SUN_ADAPTCONTROLLER_MRI_H_TOL {
        return arkReplaceAdaptController(ark_mem, C, SUNFALSE);
    }

    /* Create the mriStepControl wrapper, pass that to ARKODE, and give ownership
       of the wrapper to ARKODE */
    let Cwrapper = SUNAdaptController_MRIStep(ark_mem, C);
    arkReplaceAdaptController(ark_mem, Cwrapper.as_ref(), SUNTRUE)
}

/*---------------------------------------------------------------
  mriStep_SetUserData:

  Passes user-data pointer to attached linear solver module.
  ---------------------------------------------------------------*/
pub fn mriStep_SetUserData(ark_mem: &ARKodeMem, user_data: &mut Option<Box<dyn Any>>) -> i32 {
    /* access ARKodeMRIStepMem structure */
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "mriStep_SetUserData",
            file!(),
            MSG_MRISTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    /* set user data in ARKODELS mem (C tests `step_mem->lmem != NULL`; the
       ARKLS record lives in `ark_mem.ark_lmem` in this port) */
    if ark_mem.borrow().ark_lmem.is_some() {
        let retval = arkLSSetUserData(ark_mem, user_data);
        if retval != ARKLS_SUCCESS {
            return retval;
        }
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  mriStep_SetDefaults:

  Resets all MRIStep optional inputs to their default values.
  Does not change problem-defining function pointers or
  user_data pointer.
  ---------------------------------------------------------------*/
pub fn mriStep_SetDefaults(ark_mem: &ARKodeMem) -> i32 {
    let mut Clenrw: sunindextype = 0;
    let mut Cleniw: sunindextype = 0;

    /* access ARKodeMRIStepMem structure */
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "mriStep_SetDefaults",
            file!(),
            MSG_MRISTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    /* Set default values for integrator optional inputs */
    {
        let mut step_mem = mriStep_mem_mut(ark_mem);
        step_mem.q = 3; /* method order */
        step_mem.p = 0; /* embedding order */
        step_mem.predictor = 0; /* trivial predictor */
        step_mem.linear = SUNFALSE; /* nonlinear problem */
        step_mem.linear_timedep = SUNTRUE; /* dfs/dy depends on t */
        step_mem.deduce_rhs = SUNFALSE; /* deduce fi on result of NLS */
        step_mem.maxcor = MAXCOR; /* max nonlinear iters/stage */
        step_mem.nlscoef = NLSCOEF; /* nonlinear tolerance coefficient */
        step_mem.crdown = CRDOWN; /* nonlinear convergence estimate coeff. */
        step_mem.rdiv = RDIV; /* nonlinear divergence tolerance */
        step_mem.dgmax = DGMAX; /* max gamma change to recompute J or P */
        step_mem.msbp = MSBP; /* max steps between updating J or P */
        step_mem.stages = 0; /* no stages */
        step_mem.istage = 0; /* implicit solver stage index */
        step_mem.cur_stage = 0; /* current stage index */
        step_mem.jcur.set(SUNFALSE);
        step_mem.convfail = ARK_NO_FAILURES;
        step_mem.stage_predict = None; /* no user-supplied stage predictor */
    }

    /* Remove pre-existing nonlinear solver object */
    let (old_nls, own_nls) = {
        let mut step_mem = mriStep_mem_mut(ark_mem);
        (step_mem.NLS.take(), step_mem.ownNLS)
    };
    if old_nls.is_some() && own_nls {
        let _ = SUNNonlinSolFree(old_nls);
    }
    /* step_mem->NLS = NULL: done by the `take()` above */

    /* Remove pre-existing coupling table */
    let old_MRIC = mriStep_mem_mut(ark_mem).MRIC.take();
    if old_MRIC.is_some() {
        MRIStepCoupling_Space(old_MRIC.as_ref(), &mut Cleniw, &mut Clenrw);
        {
            let mut m = ark_mem.borrow_mut();
            m.lrw -= Clenrw;
            m.liw -= Cleniw;
        }
        MRIStepCoupling_Free(old_MRIC);
    }
    /* step_mem->MRIC = NULL: done by the `take()` above */

    /* Load the default SUNAdaptController */
    let retval = arkReplaceAdaptController(ark_mem, None, SUNTRUE);
    if retval != 0 {
        return retval;
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  mriStep_SetLinear:

  Specifies that the implicit slow function, fsi(t,y), is linear
  in y, and to tighten the linear solver tolerances while taking
  only one Newton iteration.  DO NOT USE IN COMBINATION WITH THE
  FIXED-POINT SOLVER.  Automatically tightens DeltaGammaMax
  to ensure that step size changes cause Jacobian recomputation.

  The argument should be 1 or 0, where 1 indicates that the
  Jacobian of fs with respect to y depends on time, and
  0 indicates that it is not time dependent.  Alternately, when
  using an iterative linear solver this flag denotes time
  dependence of the preconditioner.
  ---------------------------------------------------------------*/
pub fn mriStep_SetLinear(ark_mem: &ARKodeMem, timedepend: i32) -> i32 {
    /* access ARKodeMRIStepMem structure */
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "mriStep_SetLinear",
            file!(),
            MSG_MRISTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    /* set parameters */
    {
        let mut step_mem = mriStep_mem_mut(ark_mem);
        step_mem.linear = SUNTRUE;
        step_mem.linear_timedep = timedepend == 1;
        step_mem.dgmax = 100.0 * SUN_UNIT_ROUNDOFF;
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  mriStep_SetNonlinear:

  Specifies that the implicit slow function, fsi(t,y), is
  nonlinear in y.  Used to undo a previous call to
  mriStep_SetLinear.  Automatically loosens DeltaGammaMax back to
  default value.
  ---------------------------------------------------------------*/
pub fn mriStep_SetNonlinear(ark_mem: &ARKodeMem) -> i32 {
    /* access ARKodeMRIStepMem structure */
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "mriStep_SetNonlinear",
            file!(),
            MSG_MRISTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    /* set parameters */
    {
        let mut step_mem = mriStep_mem_mut(ark_mem);
        step_mem.linear = SUNFALSE;
        step_mem.linear_timedep = SUNTRUE;
        step_mem.dgmax = DGMAX;
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  mriStep_SetOrder:

  Specifies the method order
  ---------------------------------------------------------------*/
pub fn mriStep_SetOrder(ark_mem: &ARKodeMem, ord: i32) -> i32 {
    let mut Tlrw: sunindextype = 0;
    let mut Tliw: sunindextype = 0;

    /* access ARKodeMRIStepMem structure */
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "mriStep_SetOrder",
            file!(),
            MSG_MRISTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    /* check for illegal inputs */
    {
        let mut step_mem = mriStep_mem_mut(ark_mem);
        if ord <= 0 {
            step_mem.q = 3;
        } else {
            step_mem.q = ord;
        }

        /* Clear tables, the user is requesting a change in method or a reset to
        defaults. Tables will be set in InitialSetup. */
        step_mem.stages = 0;
        step_mem.p = 0;
    }
    let old_MRIC = mriStep_mem_mut(ark_mem).MRIC.take();
    MRIStepCoupling_Space(old_MRIC.as_ref(), &mut Tliw, &mut Tlrw);
    MRIStepCoupling_Free(old_MRIC);
    {
        let mut m = ark_mem.borrow_mut();
        m.liw -= Tliw;
        m.lrw -= Tlrw;
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  mriStep_SetNonlinCRDown:

  Specifies the user-provided nonlinear convergence constant
  crdown.  Legal values are strictly positive; illegal values
  imply a reset to the default.
  ---------------------------------------------------------------*/
pub fn mriStep_SetNonlinCRDown(ark_mem: &ARKodeMem, crdown: sunrealtype) -> i32 {
    /* access ARKodeMRIStepMem structure */
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "mriStep_SetNonlinCRDown",
            file!(),
            MSG_MRISTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    /* if argument legal set it, otherwise set default */
    {
        let mut step_mem = mriStep_mem_mut(ark_mem);
        if crdown <= ZERO {
            step_mem.crdown = CRDOWN;
        } else {
            step_mem.crdown = crdown;
        }
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  mriStep_SetNonlinRDiv:

  Specifies the user-provided nonlinear convergence constant
  rdiv.  Legal values are strictly positive; illegal values
  imply a reset to the default.
  ---------------------------------------------------------------*/
pub fn mriStep_SetNonlinRDiv(ark_mem: &ARKodeMem, rdiv: sunrealtype) -> i32 {
    /* access ARKodeMRIStepMem structure */
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "mriStep_SetNonlinRDiv",
            file!(),
            MSG_MRISTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    /* if argument legal set it, otherwise set default */
    {
        let mut step_mem = mriStep_mem_mut(ark_mem);
        if rdiv <= ZERO {
            step_mem.rdiv = RDIV;
        } else {
            step_mem.rdiv = rdiv;
        }
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  mriStep_SetDeltaGammaMax:

  Specifies the user-provided linear setup decision constant
  dgmax.  Legal values are strictly positive; illegal values imply
  a reset to the default.
  ---------------------------------------------------------------*/
pub fn mriStep_SetDeltaGammaMax(ark_mem: &ARKodeMem, dgmax: sunrealtype) -> i32 {
    /* access ARKodeMRIStepMem structure */
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "mriStep_SetDeltaGammaMax",
            file!(),
            MSG_MRISTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    /* if argument legal set it, otherwise set default */
    {
        let mut step_mem = mriStep_mem_mut(ark_mem);
        if dgmax <= ZERO {
            step_mem.dgmax = DGMAX;
        } else {
            step_mem.dgmax = dgmax;
        }
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  mriStep_SetLSetupFrequency:

  Specifies the user-provided linear setup decision constant
  msbp.  Positive values give the frequency for calling lsetup;
  negative values imply recomputation of lsetup at each nonlinear
  solve; a zero value implies a reset to the default.
  ---------------------------------------------------------------*/
pub fn mriStep_SetLSetupFrequency(ark_mem: &ARKodeMem, msbp: i32) -> i32 {
    /* access ARKodeMRIStepMem structure */
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "mriStep_SetLSetupFrequency",
            file!(),
            MSG_MRISTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    /* if argument legal set it, otherwise set default */
    {
        let mut step_mem = mriStep_mem_mut(ark_mem);
        if msbp == 0 {
            step_mem.msbp = MSBP;
        } else {
            step_mem.msbp = msbp;
        }
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  mriStep_SetPredictorMethod:

  Specifies the method to use for predicting implicit solutions.
  Non-default choices are {1,2,3,4}, all others will use default
  (trivial) predictor.
  ---------------------------------------------------------------*/
pub fn mriStep_SetPredictorMethod(ark_mem: &ARKodeMem, pred_method: i32) -> i32 {
    /* access ARKodeMRIStepMem structure */
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "mriStep_SetPredictorMethod",
            file!(),
            MSG_MRISTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    /* set parameter */
    mriStep_mem_mut(ark_mem).predictor = pred_method;

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  mriStep_SetMaxNonlinIters:

  Specifies the maximum number of nonlinear iterations during
  one solve.  A non-positive input implies a reset to the
  default value.
  ---------------------------------------------------------------*/
pub fn mriStep_SetMaxNonlinIters(ark_mem: &ARKodeMem, maxcor: i32) -> i32 {
    /* access ARKodeMRIStepMem structure */
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "mriStep_SetMaxNonlinIters",
            file!(),
            MSG_MRISTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    /* Return error message if no NLS module is present */
    let no_nls = mriStep_mem_mut(ark_mem).NLS.is_none();
    if no_nls {
        arkProcessError(
            Some(ark_mem),
            ARK_NLS_OP_ERR,
            line!() as i32,
            "mriStep_SetMaxNonlinIters",
            file!(),
            "No SUNNonlinearSolver object is present",
        );
        return ARK_ILL_INPUT;
    }

    /* argument <= 0 sets default, otherwise set input */
    {
        let mut step_mem = mriStep_mem_mut(ark_mem);
        if maxcor <= 0 {
            step_mem.maxcor = MAXCOR;
        } else {
            step_mem.maxcor = maxcor;
        }
    }

    /* send argument to NLS structure */
    let (nls, maxcor_set) = {
        let step_mem = mriStep_mem_mut(ark_mem);
        (step_mem.NLS.clone().expect("NLS"), step_mem.maxcor)
    };
    let retval = SUNNonlinSolSetMaxIters(&nls, maxcor_set);
    if retval != SUN_SUCCESS {
        arkProcessError(
            Some(ark_mem),
            ARK_NLS_OP_ERR,
            line!() as i32,
            "mriStep_SetMaxNonlinIters",
            file!(),
            "Error setting maxcor in SUNNonlinearSolver object",
        );
        return ARK_NLS_OP_ERR;
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  mriStep_SetNonlinConvCoef:

  Specifies the coefficient in the nonlinear solver convergence
  test.  A non-positive input implies a reset to the default value.
  ---------------------------------------------------------------*/
pub fn mriStep_SetNonlinConvCoef(ark_mem: &ARKodeMem, nlscoef: sunrealtype) -> i32 {
    /* access ARKodeMRIStepMem structure */
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "mriStep_SetNonlinConvCoef",
            file!(),
            MSG_MRISTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    /* argument <= 0 sets default, otherwise set input */
    {
        let mut step_mem = mriStep_mem_mut(ark_mem);
        if nlscoef <= ZERO {
            step_mem.nlscoef = NLSCOEF;
        } else {
            step_mem.nlscoef = nlscoef;
        }
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  mriStep_SetStagePredictFn:  Specifies a user-provided step
  predictor function having type ARKStagePredictFn.  A
  NULL input function disables calls to this routine.
  ---------------------------------------------------------------*/
pub fn mriStep_SetStagePredictFn(
    ark_mem: &ARKodeMem,
    PredictStage: Option<ARKStagePredictFn>,
) -> i32 {
    /* access ARKodeMRIStepMem structure and set function pointer */
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "mriStep_SetStagePredictFn",
            file!(),
            MSG_MRISTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    mriStep_mem_mut(ark_mem).stage_predict = PredictStage;
    ARK_SUCCESS
}

/*---------------------------------------------------------------
  mriStep_SetDeduceImplicitRhs:

  Specifies if an optimization is used to avoid an evaluation of
  fi after a nonlinear solve for an implicit stage.  If stage
  postprocessecing in enabled, this option is ignored, and fi is
  never deduced.

  An argument of SUNTRUE indicates that fi is deduced to compute
  fi(z_i), and SUNFALSE indicates that fi(z_i) is computed with
  an additional evaluation of fi.
  ---------------------------------------------------------------*/
pub fn mriStep_SetDeduceImplicitRhs(ark_mem: &ARKodeMem, deduce: sunbooleantype) -> i32 {
    /* access ARKodeMRIStepMem structure and set function pointer */
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "mriStep_SetDeduceImplicitRhs",
            file!(),
            MSG_MRISTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    mriStep_mem_mut(ark_mem).deduce_rhs = deduce;
    ARK_SUCCESS
}

/*---------------------------------------------------------------
  mriStep_GetCurrentGamma: Returns the current value of gamma
  ---------------------------------------------------------------*/
pub fn mriStep_GetCurrentGamma(ark_mem: &ARKodeMem, gamma: &mut sunrealtype) -> i32 {
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "mriStep_GetCurrentGamma",
            file!(),
            MSG_MRISTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }
    *gamma = mriStep_mem_mut(ark_mem).gamma;
    /* C returns the (successful) access retval */
    ARK_SUCCESS
}

/*---------------------------------------------------------------
  mriStep_GetEstLocalErrors: Returns the current local truncation
  error estimate vector
  ---------------------------------------------------------------*/
pub fn mriStep_GetEstLocalErrors(ark_mem: &ARKodeMem, ele: &N_Vector) -> i32 {
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "mriStep_GetEstLocalErrors",
            file!(),
            MSG_MRISTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    /* return an error if local truncation error is not computed */
    let (fixedstep, AccumErrorType, tempv1) = {
        let m = ark_mem.borrow();
        (m.fixedstep, m.AccumErrorType, m.tempv1.clone())
    };
    let p = mriStep_mem_mut(ark_mem).p;
    if (fixedstep && (AccumErrorType == ARK_ACCUMERROR_NONE)) || (p <= 0) {
        return ARK_STEPPER_UNSUPPORTED;
    }

    /* otherwise, copy local truncation error vector to output */
    N_VScale(ONE, &tempv1.expect("tempv1"), ele);
    ARK_SUCCESS
}

/*---------------------------------------------------------------
  mriStep_GetNumLinSolvSetups:

  Returns the current number of calls to the lsetup routine
  ---------------------------------------------------------------*/
pub fn mriStep_GetNumLinSolvSetups(ark_mem: &ARKodeMem, nlinsetups: &mut i64) -> i32 {
    /* access ARKodeMRIStepMem structure */
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "mriStep_GetNumLinSolvSetups",
            file!(),
            MSG_MRISTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    /* get value from step_mem */
    *nlinsetups = mriStep_mem_mut(ark_mem).nsetups;

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  mriStep_GetNumNonlinSolvIters:

  Returns the current number of nonlinear solver iterations
  ---------------------------------------------------------------*/
pub fn mriStep_GetNumNonlinSolvIters(ark_mem: &ARKodeMem, nniters: &mut i64) -> i32 {
    /* access ARKodeMRIStepMem structure */
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "mriStep_GetNumNonlinSolvIters",
            file!(),
            MSG_MRISTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    *nniters = mriStep_mem_mut(ark_mem).nls_iters;

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  mriStep_GetNumNonlinSolvConvFails:

  Returns the current number of nonlinear solver convergence fails
  ---------------------------------------------------------------*/
pub fn mriStep_GetNumNonlinSolvConvFails(ark_mem: &ARKodeMem, nnfails: &mut i64) -> i32 {
    /* access ARKodeMRIStepMem structure */
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "mriStep_GetNumNonlinSolvConvFails",
            file!(),
            MSG_MRISTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    /* set output from step_mem */
    *nnfails = mriStep_mem_mut(ark_mem).nls_fails;

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  mriStep_GetNonlinSolvStats:

  Returns nonlinear solver statistics
  ---------------------------------------------------------------*/
pub fn mriStep_GetNonlinSolvStats(
    ark_mem: &ARKodeMem,
    nniters: &mut i64,
    nnfails: &mut i64,
) -> i32 {
    /* access ARKodeMRIStepMem structure */
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "mriStep_GetNonlinSolvStats",
            file!(),
            MSG_MRISTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    {
        let step_mem = mriStep_mem_mut(ark_mem);
        *nniters = step_mem.nls_iters;
        *nnfails = step_mem.nls_fails;
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  mriStep_GetStageIndex:

  Returns the current stage index and number of stages
  ---------------------------------------------------------------*/
pub fn mriStep_GetStageIndex(ark_mem: &ARKodeMem, stage: &mut i32, max_stages: &mut i32) -> i32 {
    /* access ARKodeMRIStepMem structure */
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "mriStep_GetStageIndex",
            file!(),
            MSG_MRISTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    {
        let step_mem = mriStep_mem_mut(ark_mem);
        *stage = step_mem.cur_stage;
        *max_stages = step_mem.stages;
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  mriStep_PrintAllStats:

  Prints integrator statistics
  ---------------------------------------------------------------*/
pub fn mriStep_PrintAllStats(
    ark_mem: &ARKodeMem,
    outfile: &SUNFile,
    fmt: SUNOutputFormat,
) -> i32 {
    /* access ARKode MRIStepMem structure */
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "mriStep_PrintAllStats",
            file!(),
            MSG_MRISTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    let (nfse, nfsi, inner_fails, nls_iters, nls_fails, nsetups) = {
        let step_mem = mriStep_mem_mut(ark_mem);
        (
            step_mem.nfse,
            step_mem.nfsi,
            step_mem.inner_fails,
            step_mem.nls_iters,
            step_mem.nls_fails,
            step_mem.nsetups,
        )
    };
    let nst = ark_mem.borrow().nst;

    /* function evaluations */
    sunfprintf_long(outfile, fmt, SUNTRUE, "Explicit slow RHS fn evals", nfse);
    sunfprintf_long(outfile, fmt, SUNFALSE, "Implicit slow RHS fn evals", nfsi);

    /* inner stepper and nonlinear solver stats */
    sunfprintf_long(outfile, fmt, SUNFALSE, "Inner stepper failures", inner_fails);
    sunfprintf_long(outfile, fmt, SUNFALSE, "NLS iters", nls_iters);
    sunfprintf_long(outfile, fmt, SUNFALSE, "NLS fails", nls_fails);
    if nst > 0 {
        sunfprintf_real(
            outfile,
            fmt,
            SUNFALSE,
            "NLS iters per step",
            (nls_iters as sunrealtype) / (nst as sunrealtype),
        );
    }

    /* linear solver stats */
    sunfprintf_long(outfile, fmt, SUNFALSE, "LS setups", nsetups);
    let step_getlinmem = ark_mem.borrow().step_getlinmem.expect("step_getlinmem");
    if step_getlinmem(ark_mem) {
        let (nje, nfeDQ, npe, nps, nli, ncfl, njtsetup, njtimes) = {
            let arkls_mem = arkls_mem_mut(ark_mem);
            (
                arkls_mem.nje,
                arkls_mem.nfeDQ,
                arkls_mem.npe,
                arkls_mem.nps,
                arkls_mem.nli,
                arkls_mem.ncfl,
                arkls_mem.njtsetup,
                arkls_mem.njtimes,
            )
        };
        sunfprintf_long(outfile, fmt, SUNFALSE, "Jac fn evals", nje);
        sunfprintf_long(outfile, fmt, SUNFALSE, "LS RHS fn evals", nfeDQ);
        sunfprintf_long(outfile, fmt, SUNFALSE, "Prec setup evals", npe);
        sunfprintf_long(outfile, fmt, SUNFALSE, "Prec solves", nps);
        sunfprintf_long(outfile, fmt, SUNFALSE, "LS iters", nli);
        sunfprintf_long(outfile, fmt, SUNFALSE, "LS fails", ncfl);
        sunfprintf_long(outfile, fmt, SUNFALSE, "Jac-times setups", njtsetup);
        sunfprintf_long(outfile, fmt, SUNFALSE, "Jac-times evals", njtimes);
        if nls_iters > 0 {
            sunfprintf_real(
                outfile,
                fmt,
                SUNFALSE,
                "LS iters per NLS iter",
                (nli as sunrealtype) / (nls_iters as sunrealtype),
            );
            sunfprintf_real(
                outfile,
                fmt,
                SUNFALSE,
                "Jac evals per NLS iter",
                (nje as sunrealtype) / (nls_iters as sunrealtype),
            );
            sunfprintf_real(
                outfile,
                fmt,
                SUNFALSE,
                "Prec evals per NLS iter",
                (npe as sunrealtype) / (nls_iters as sunrealtype),
            );
        }
    }

    ARK_SUCCESS
}

/*---------------------------------------------------------------
  mriStep_WriteParameters:

  Outputs all solver parameters to the provided file pointer.
  ---------------------------------------------------------------*/
pub fn mriStep_WriteParameters(ark_mem: &ARKodeMem, fp: &SUNFile) -> i32 {
    /* access ARKodeMRIStepMem structure */
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "mriStep_WriteParameters",
            file!(),
            MSG_MRISTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    let (
        q,
        linear,
        linear_timedep,
        explicit_rhs,
        implicit_rhs,
        predictor,
        nlscoef,
        maxcor,
        crdown,
        rdiv,
        dgmax,
        msbp,
    ) = {
        let step_mem = mriStep_mem_mut(ark_mem);
        (
            step_mem.q,
            step_mem.linear,
            step_mem.linear_timedep,
            step_mem.explicit_rhs,
            step_mem.implicit_rhs,
            step_mem.predictor,
            step_mem.nlscoef,
            step_mem.maxcor,
            step_mem.crdown,
            step_mem.rdiv,
            step_mem.dgmax,
            step_mem.msbp,
        )
    };

    /* print integrator parameters to file */
    fp.write_str("MRIStep time step module parameters:\n");
    fp.write_str(&format!("  Method order {q}\n"));
    if linear {
        fp.write_str("  Linear implicit problem");
        if linear_timedep {
            fp.write_str(" (time-dependent Jacobian)\n");
        } else {
            fp.write_str(" (time-independent Jacobian)\n");
        }
    }
    if explicit_rhs && implicit_rhs {
        fp.write_str("  ImEx slow time scale\n");
    } else if implicit_rhs {
        fp.write_str("  Implicit slow time scale\n");
    } else {
        fp.write_str("  Explicit slow time scale\n");
    }

    if implicit_rhs {
        fp.write_str(&format!("  Implicit predictor method = {predictor}\n"));
        fp.write_str(&format!(
            "  Implicit solver tolerance coefficient = {}\n",
            sun_format_g(nlscoef)
        ));
        fp.write_str(&format!(
            "  Maximum number of nonlinear corrections = {maxcor}\n"
        ));
        fp.write_str(&format!(
            "  Nonlinear convergence rate constant = {}\n",
            sun_format_g(crdown)
        ));
        fp.write_str(&format!(
            "  Nonlinear divergence tolerance = {}\n",
            sun_format_g(rdiv)
        ));
        fp.write_str(&format!(
            "  Gamma factor LSetup tolerance = {}\n",
            sun_format_g(dgmax)
        ));
        fp.write_str(&format!(
            "  Number of steps between LSetup calls = {msbp}\n"
        ));
    }
    fp.write_str("\n");

    ARK_SUCCESS
}

/*===============================================================
  Exported-but-deprecated user-callable functions.
  ===============================================================*/

pub fn MRIStepResize(
    arkode_mem: &ARKodeMem,
    y0: &N_Vector,
    t0: sunrealtype,
    resize: Option<ARKVecResizeFn>,
    resize_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    ARKodeResize(arkode_mem, y0, ONE, t0, resize, resize_data)
}

pub fn MRIStepReset(arkode_mem: &ARKodeMem, tR: sunrealtype, yR: &N_Vector) -> i32 {
    ARKodeReset(arkode_mem, tR, yR)
}

pub fn MRIStepSStolerances(
    arkode_mem: &ARKodeMem,
    reltol: sunrealtype,
    abstol: sunrealtype,
) -> i32 {
    ARKodeSStolerances(arkode_mem, reltol, abstol)
}

pub fn MRIStepSVtolerances(
    arkode_mem: &ARKodeMem,
    reltol: sunrealtype,
    abstol: &N_Vector,
) -> i32 {
    ARKodeSVtolerances(arkode_mem, reltol, abstol)
}

pub fn MRIStepWFtolerances(arkode_mem: &ARKodeMem, efun: ARKEwtFn) -> i32 {
    ARKodeWFtolerances(arkode_mem, efun)
}

pub fn MRIStepSetLinearSolver(
    arkode_mem: &ARKodeMem,
    LS: &SUNLinearSolver,
    A: Option<&SUNMatrix>,
) -> i32 {
    ARKodeSetLinearSolver(arkode_mem, LS, A)
}

pub fn MRIStepRootInit(arkode_mem: &ARKodeMem, nrtfn: i32, g: Option<ARKRootFn>) -> i32 {
    ARKodeRootInit(arkode_mem, nrtfn, g)
}

pub fn MRIStepSetDefaults(arkode_mem: &ARKodeMem) -> i32 {
    ARKodeSetDefaults(arkode_mem)
}

pub fn MRIStepSetOrder(arkode_mem: &ARKodeMem, ord: i32) -> i32 {
    ARKodeSetOrder(arkode_mem, ord)
}

pub fn MRIStepSetInterpolantType(arkode_mem: &ARKodeMem, itype: i32) -> i32 {
    ARKodeSetInterpolantType(arkode_mem, itype)
}

pub fn MRIStepSetInterpolantDegree(arkode_mem: &ARKodeMem, degree: i32) -> i32 {
    ARKodeSetInterpolantDegree(arkode_mem, degree)
}

pub fn MRIStepSetDenseOrder(arkode_mem: &ARKodeMem, dord: i32) -> i32 {
    ARKodeSetInterpolantDegree(arkode_mem, dord)
}

pub fn MRIStepSetNonlinearSolver(arkode_mem: &ARKodeMem, NLS: &SUNNonlinearSolver) -> i32 {
    ARKodeSetNonlinearSolver(arkode_mem, NLS)
}

pub fn MRIStepSetNlsRhsFn(arkode_mem: &ARKodeMem, nls_fi: Option<ARKRhsFn>) -> i32 {
    ARKodeSetNlsRhsFn(arkode_mem, nls_fi)
}

pub fn MRIStepSetLinear(arkode_mem: &ARKodeMem, timedepend: i32) -> i32 {
    ARKodeSetLinear(arkode_mem, timedepend)
}

pub fn MRIStepSetNonlinear(arkode_mem: &ARKodeMem) -> i32 {
    ARKodeSetNonlinear(arkode_mem)
}

pub fn MRIStepSetMaxNumSteps(arkode_mem: &ARKodeMem, mxsteps: i64) -> i32 {
    ARKodeSetMaxNumSteps(arkode_mem, mxsteps)
}

pub fn MRIStepSetNonlinCRDown(arkode_mem: &ARKodeMem, crdown: sunrealtype) -> i32 {
    ARKodeSetNonlinCRDown(arkode_mem, crdown)
}

pub fn MRIStepSetNonlinRDiv(arkode_mem: &ARKodeMem, rdiv: sunrealtype) -> i32 {
    ARKodeSetNonlinRDiv(arkode_mem, rdiv)
}

pub fn MRIStepSetDeltaGammaMax(arkode_mem: &ARKodeMem, dgmax: sunrealtype) -> i32 {
    ARKodeSetDeltaGammaMax(arkode_mem, dgmax)
}

pub fn MRIStepSetLSetupFrequency(arkode_mem: &ARKodeMem, msbp: i32) -> i32 {
    ARKodeSetLSetupFrequency(arkode_mem, msbp)
}

pub fn MRIStepSetPredictorMethod(arkode_mem: &ARKodeMem, pred_method: i32) -> i32 {
    ARKodeSetPredictorMethod(arkode_mem, pred_method)
}

pub fn MRIStepSetMaxNonlinIters(arkode_mem: &ARKodeMem, maxcor: i32) -> i32 {
    ARKodeSetMaxNonlinIters(arkode_mem, maxcor)
}

pub fn MRIStepSetNonlinConvCoef(arkode_mem: &ARKodeMem, nlscoef: sunrealtype) -> i32 {
    ARKodeSetNonlinConvCoef(arkode_mem, nlscoef)
}

pub fn MRIStepSetMaxHnilWarns(arkode_mem: &ARKodeMem, mxhnil: i32) -> i32 {
    ARKodeSetMaxHnilWarns(arkode_mem, mxhnil)
}

pub fn MRIStepSetInterpolateStopTime(arkode_mem: &ARKodeMem, interp: sunbooleantype) -> i32 {
    ARKodeSetInterpolateStopTime(arkode_mem, interp)
}

pub fn MRIStepSetStopTime(arkode_mem: &ARKodeMem, tstop: sunrealtype) -> i32 {
    ARKodeSetStopTime(arkode_mem, tstop)
}

pub fn MRIStepClearStopTime(arkode_mem: &ARKodeMem) -> i32 {
    ARKodeClearStopTime(arkode_mem)
}

pub fn MRIStepSetFixedStep(arkode_mem: &ARKodeMem, hfixed: sunrealtype) -> i32 {
    ARKodeSetFixedStep(arkode_mem, hfixed)
}

pub fn MRIStepSetRootDirection(arkode_mem: &ARKodeMem, rootdir: &[i32]) -> i32 {
    ARKodeSetRootDirection(arkode_mem, rootdir)
}

pub fn MRIStepSetNoInactiveRootWarn(arkode_mem: &ARKodeMem) -> i32 {
    ARKodeSetNoInactiveRootWarn(arkode_mem)
}

pub fn MRIStepSetUserData(arkode_mem: &ARKodeMem, user_data: Option<Box<dyn Any>>) -> i32 {
    ARKodeSetUserData(arkode_mem, user_data)
}

pub fn MRIStepSetPostprocessStepFn(
    arkode_mem: &ARKodeMem,
    ProcessStep: Option<ARKPostProcessFn>,
) -> i32 {
    ARKodeSetPostprocessStepFn(arkode_mem, ProcessStep)
}

pub fn MRIStepSetPostprocessStageFn(
    arkode_mem: &ARKodeMem,
    ProcessStage: Option<ARKPostProcessFn>,
) -> i32 {
    ARKodeSetPostprocessStageFn(arkode_mem, ProcessStage)
}

pub fn MRIStepSetStagePredictFn(
    arkode_mem: &ARKodeMem,
    PredictStage: Option<ARKStagePredictFn>,
) -> i32 {
    ARKodeSetStagePredictFn(arkode_mem, PredictStage)
}

pub fn MRIStepSetDeduceImplicitRhs(arkode_mem: &ARKodeMem, deduce: sunbooleantype) -> i32 {
    ARKodeSetDeduceImplicitRhs(arkode_mem, deduce)
}

pub fn MRIStepSetJacFn(arkode_mem: &ARKodeMem, jac: Option<ARKLsJacFn>) -> i32 {
    ARKodeSetJacFn(arkode_mem, jac)
}

pub fn MRIStepSetJacEvalFrequency(arkode_mem: &ARKodeMem, msbj: i64) -> i32 {
    ARKodeSetJacEvalFrequency(arkode_mem, msbj)
}

pub fn MRIStepSetLinearSolutionScaling(arkode_mem: &ARKodeMem, onoff: sunbooleantype) -> i32 {
    ARKodeSetLinearSolutionScaling(arkode_mem, onoff)
}

pub fn MRIStepSetEpsLin(arkode_mem: &ARKodeMem, eplifac: sunrealtype) -> i32 {
    ARKodeSetEpsLin(arkode_mem, eplifac)
}

pub fn MRIStepSetLSNormFactor(arkode_mem: &ARKodeMem, nrmfac: sunrealtype) -> i32 {
    ARKodeSetLSNormFactor(arkode_mem, nrmfac)
}

pub fn MRIStepSetPreconditioner(
    arkode_mem: &ARKodeMem,
    psetup: Option<ARKLsPrecSetupFn>,
    psolve: Option<ARKLsPrecSolveFn>,
) -> i32 {
    ARKodeSetPreconditioner(arkode_mem, psetup, psolve)
}

pub fn MRIStepSetJacTimes(
    arkode_mem: &ARKodeMem,
    jtsetup: Option<ARKLsJacTimesSetupFn>,
    jtimes: Option<ARKLsJacTimesVecFn>,
) -> i32 {
    ARKodeSetJacTimes(arkode_mem, jtsetup, jtimes)
}

pub fn MRIStepSetJacTimesRhsFn(arkode_mem: &ARKodeMem, jtimesRhsFn: Option<ARKRhsFn>) -> i32 {
    ARKodeSetJacTimesRhsFn(arkode_mem, jtimesRhsFn)
}

pub fn MRIStepSetLinSysFn(arkode_mem: &ARKodeMem, linsys: Option<ARKLsLinSysFn>) -> i32 {
    ARKodeSetLinSysFn(arkode_mem, linsys)
}

pub fn MRIStepEvolve(
    arkode_mem: &ARKodeMem,
    tout: sunrealtype,
    yout: &N_Vector,
    tret: &mut sunrealtype,
    itask: i32,
) -> i32 {
    ARKodeEvolve(arkode_mem, tout, yout, tret, itask)
}

pub fn MRIStepGetDky(arkode_mem: &ARKodeMem, t: sunrealtype, k: i32, dky: &N_Vector) -> i32 {
    ARKodeGetDky(arkode_mem, t, k, dky)
}

pub fn MRIStepComputeState(arkode_mem: &ARKodeMem, zcor: &N_Vector, z: &N_Vector) -> i32 {
    ARKodeComputeState(arkode_mem, zcor, z)
}

pub fn MRIStepGetNumLinSolvSetups(arkode_mem: &ARKodeMem, nlinsetups: &mut i64) -> i32 {
    ARKodeGetNumLinSolvSetups(arkode_mem, nlinsetups)
}

pub fn MRIStepGetWorkSpace(arkode_mem: &ARKodeMem, lenrw: &mut i64, leniw: &mut i64) -> i32 {
    ARKodeGetWorkSpace(arkode_mem, lenrw, leniw)
}

pub fn MRIStepGetNumSteps(arkode_mem: &ARKodeMem, nssteps: &mut i64) -> i32 {
    ARKodeGetNumSteps(arkode_mem, nssteps)
}

pub fn MRIStepGetLastStep(arkode_mem: &ARKodeMem, hlast: &mut sunrealtype) -> i32 {
    ARKodeGetLastStep(arkode_mem, hlast)
}

pub fn MRIStepGetCurrentTime(arkode_mem: &ARKodeMem, tcur: &mut sunrealtype) -> i32 {
    ARKodeGetCurrentTime(arkode_mem, tcur)
}

pub fn MRIStepGetCurrentState(arkode_mem: &ARKodeMem, state: &mut Option<N_Vector>) -> i32 {
    ARKodeGetCurrentState(arkode_mem, state)
}

pub fn MRIStepGetCurrentGamma(arkode_mem: &ARKodeMem, gamma: &mut sunrealtype) -> i32 {
    ARKodeGetCurrentGamma(arkode_mem, gamma)
}

pub fn MRIStepGetTolScaleFactor(arkode_mem: &ARKodeMem, tolsfact: &mut sunrealtype) -> i32 {
    ARKodeGetTolScaleFactor(arkode_mem, tolsfact)
}

pub fn MRIStepGetErrWeights(arkode_mem: &ARKodeMem, eweight: &N_Vector) -> i32 {
    ARKodeGetErrWeights(arkode_mem, eweight)
}

pub fn MRIStepGetNumGEvals(arkode_mem: &ARKodeMem, ngevals: &mut i64) -> i32 {
    ARKodeGetNumGEvals(arkode_mem, ngevals)
}

pub fn MRIStepGetRootInfo(arkode_mem: &ARKodeMem, rootsfound: &mut [i32]) -> i32 {
    ARKodeGetRootInfo(arkode_mem, rootsfound)
}

pub fn MRIStepGetUserData(arkode_mem: &ARKodeMem, user_data: &mut Option<Box<dyn Any>>) -> i32 {
    ARKodeGetUserData(arkode_mem, user_data)
}

pub fn MRIStepPrintAllStats(
    arkode_mem: &ARKodeMem,
    outfile: &SUNFile,
    fmt: SUNOutputFormat,
) -> i32 {
    ARKodePrintAllStats(arkode_mem, outfile, fmt)
}

/// C returns a `malloc`'d `char*`; the Rust port returns `String`.
pub fn MRIStepGetReturnFlagName(flag: i64) -> String {
    ARKodeGetReturnFlagName(flag)
}

pub fn MRIStepWriteParameters(arkode_mem: &ARKodeMem, fp: &SUNFile) -> i32 {
    ARKodeWriteParameters(arkode_mem, fp)
}

pub fn MRIStepWriteCoupling(arkode_mem: &ARKodeMem, fp: &SUNFile) -> i32 {
    let ark_mem = arkode_mem;

    /* access ARKodeMem and ARKodeMRIStepMem structures */
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "MRIStepWriteCoupling",
            file!(),
            MSG_MRISTEP_NO_MEM,
        );
        return ARK_MEM_NULL;
    }

    /* check that coupling structure is non-NULL (otherwise report error) */
    let MRIC = mriStep_mem_mut(ark_mem).MRIC.clone();
    let MRIC = match MRIC {
        None => {
            arkProcessError(
                Some(ark_mem),
                ARK_MEM_NULL,
                line!() as i32,
                "MRIStepWriteCoupling",
                file!(),
                "Coupling structure is NULL",
            );
            return ARK_MEM_NULL;
        }
        Some(MRIC) => MRIC,
    };

    /* write coupling structure to specified file */
    fp.write_str("\nMRIStep coupling structure:\n");
    MRIStepCoupling_Write(Some(&MRIC), fp);

    ARK_SUCCESS
}

#[allow(clippy::too_many_arguments)]
pub fn MRIStepGetNonlinearSystemData(
    arkode_mem: &ARKodeMem,
    tcur: &mut sunrealtype,
    zpred: &mut Option<N_Vector>,
    z: &mut Option<N_Vector>,
    Fi: &mut Option<N_Vector>,
    gamma: &mut sunrealtype,
    sdata: &mut Option<N_Vector>,
    user_data: &mut Option<Box<dyn Any>>,
) -> i32 {
    ARKodeGetNonlinearSystemData(arkode_mem, tcur, zpred, z, Fi, gamma, sdata, user_data)
}

pub fn MRIStepGetNumNonlinSolvIters(arkode_mem: &ARKodeMem, nniters: &mut i64) -> i32 {
    ARKodeGetNumNonlinSolvIters(arkode_mem, nniters)
}

pub fn MRIStepGetNumNonlinSolvConvFails(arkode_mem: &ARKodeMem, nnfails: &mut i64) -> i32 {
    ARKodeGetNumNonlinSolvConvFails(arkode_mem, nnfails)
}

pub fn MRIStepGetNonlinSolvStats(
    arkode_mem: &ARKodeMem,
    nniters: &mut i64,
    nnfails: &mut i64,
) -> i32 {
    ARKodeGetNonlinSolvStats(arkode_mem, nniters, nnfails)
}

pub fn MRIStepGetNumStepSolveFails(arkode_mem: &ARKodeMem, nncfails: &mut i64) -> i32 {
    ARKodeGetNumStepSolveFails(arkode_mem, nncfails)
}

pub fn MRIStepGetJac(arkode_mem: &ARKodeMem, J: &mut Option<SUNMatrix>) -> i32 {
    ARKodeGetJac(arkode_mem, J)
}

pub fn MRIStepGetJacTime(arkode_mem: &ARKodeMem, t_J: &mut sunrealtype) -> i32 {
    ARKodeGetJacTime(arkode_mem, t_J)
}

pub fn MRIStepGetJacNumSteps(arkode_mem: &ARKodeMem, nst_J: &mut i64) -> i32 {
    ARKodeGetJacNumSteps(arkode_mem, nst_J)
}

pub fn MRIStepGetLinWorkSpace(
    arkode_mem: &ARKodeMem,
    lenrwLS: &mut i64,
    leniwLS: &mut i64,
) -> i32 {
    ARKodeGetLinWorkSpace(arkode_mem, lenrwLS, leniwLS)
}

pub fn MRIStepGetNumJacEvals(arkode_mem: &ARKodeMem, njevals: &mut i64) -> i32 {
    ARKodeGetNumJacEvals(arkode_mem, njevals)
}

pub fn MRIStepGetNumPrecEvals(arkode_mem: &ARKodeMem, npevals: &mut i64) -> i32 {
    ARKodeGetNumPrecEvals(arkode_mem, npevals)
}

pub fn MRIStepGetNumPrecSolves(arkode_mem: &ARKodeMem, npsolves: &mut i64) -> i32 {
    ARKodeGetNumPrecSolves(arkode_mem, npsolves)
}

pub fn MRIStepGetNumLinIters(arkode_mem: &ARKodeMem, nliters: &mut i64) -> i32 {
    ARKodeGetNumLinIters(arkode_mem, nliters)
}

pub fn MRIStepGetNumLinConvFails(arkode_mem: &ARKodeMem, nlcfails: &mut i64) -> i32 {
    ARKodeGetNumLinConvFails(arkode_mem, nlcfails)
}

pub fn MRIStepGetNumJTSetupEvals(arkode_mem: &ARKodeMem, njtsetups: &mut i64) -> i32 {
    ARKodeGetNumJTSetupEvals(arkode_mem, njtsetups)
}

pub fn MRIStepGetNumJtimesEvals(arkode_mem: &ARKodeMem, njvevals: &mut i64) -> i32 {
    ARKodeGetNumJtimesEvals(arkode_mem, njvevals)
}

pub fn MRIStepGetNumLinRhsEvals(arkode_mem: &ARKodeMem, nfevalsLS: &mut i64) -> i32 {
    ARKodeGetNumLinRhsEvals(arkode_mem, nfevalsLS)
}

pub fn MRIStepGetLastLinFlag(arkode_mem: &ARKodeMem, flag: &mut i64) -> i32 {
    ARKodeGetLastLinFlag(arkode_mem, flag)
}

/// C returns a `malloc`'d `char*`; the Rust port returns `String`.
pub fn MRIStepGetLinReturnFlagName(flag: i64) -> String {
    ARKodeGetLinReturnFlagName(flag)
}

pub fn MRIStepFree(arkode_mem: &mut Option<ARKodeMem>) {
    ARKodeFree(arkode_mem)
}

pub fn MRIStepPrintMem(arkode_mem: &ARKodeMem, outfile: &SUNFile) {
    ARKodePrintMem(arkode_mem, outfile)
}

/*===============================================================
  EOF
  ===============================================================*/
