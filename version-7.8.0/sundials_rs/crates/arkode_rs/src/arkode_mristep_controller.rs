//! Port of `src/arkode/arkode_mristep_controller.c`: MRIStep's multirate
//! adaptivity controller layer.
//!
//! This module wraps a user-supplied `SUN_ADAPTCONTROLLER_MRI_H_TOL`
//! controller inside a plain `SUN_ADAPTCONTROLLER_H`-looking object so that
//! ARKODE's single-rate controller calls (`EstimateStep`, `UpdateH`) are
//! translated into the multirate calls (`EstimateStepTol`, `UpdateMRIHTol`)
//! using MRIStep's knowledge of the slow/fast time-scale relationship.
//!
//! Deviations from the C, all forced and all unobservable:
//!  * C `mriStepControlContent` caches three raw pointers: `ark_mem`,
//!    `step_mem` and `C`. `step_mem` is `ark_mem->step_mem` — the very same
//!    object — and in this port the MRIStep record lives BY VALUE inside
//!    `ark_mem.step_mem`, so it cannot be aliased by a second field. The
//!    content therefore stores only `ark_mem` (an `Rc` clone = the C pointer
//!    copy) and reaches the stepper record through `mriStep_mem_mut`, which
//!    is exactly what the C pointer denotes.
//!  * The `content == NULL` malloc-failure branch of
//!    `SUNAdaptController_MRIStep` is unreachable (`Box::new` cannot fail)
//!    and is omitted.
//!  * `ark_mem == NULL` in `EstimateStep`/`UpdateH` is unreachable (the
//!    handle is stored by value); the companion `step_mem == NULL` half of
//!    each check is kept verbatim.

use std::cell::RefMut;

use crate::arkode_impl::*;
use crate::arkode_mristep::*;

use sundials_core::sundials_adaptcontroller::*;
use sundials_core::sundials_errors::{SUN_ERR_MEM_FAIL, SUN_SUCCESS};
use sundials_core::sundials_types::*;
use sundials_core::sundials_utils::SUNFile;

/*===============================================================
  MRIStep SUNAdaptController wrapper content
  (declared in arkode_mristep_impl.h)
  ===============================================================*/

/// C `struct _mriStepControlContent`. See the module docs for why the C
/// `step_mem` member is not mirrored here.
pub struct mriStepControlContent {
    /// ARKODE memory pointer
    pub ark_mem: ARKodeMem,
    /// attached controller pointer
    pub C: SUNAdaptController,
}

/// Downcast helper for this module's controller content. Panics if the
/// object is not a `SUNAdaptController_MRIStep` wrapper (C would blindly
/// cast the `void*` — UB maps to a deterministic panic). Never hold the
/// guard across a sub-controller call.
fn mriStepControl_content_mut(C: &SUNAdaptController) -> RefMut<'_, mriStepControlContent> {
    RefMut::map(C.content.borrow_mut(), |c| {
        c.downcast_mut::<mriStepControlContent>()
            .expect("MRIStep SUNAdaptController content")
    })
}

/// C macro `MRICONTROL_C(C)` — the wrapped multirate controller.
fn MRICONTROL_C(C: &SUNAdaptController) -> SUNAdaptController {
    let content = mriStepControl_content_mut(C);
    content.C.clone()
}

/// C macro `MRICONTROL_A(C)` — the ARKODE memory structure.
fn MRICONTROL_A(C: &SUNAdaptController) -> ARKodeMem {
    let content = mriStepControl_content_mut(C);
    content.ark_mem.clone()
}

/*--------------------------------------------
  MRIStep SUNAdaptController wrapper functions
  --------------------------------------------*/

pub fn SUNAdaptController_MRIStep(
    ark_mem: &ARKodeMem,
    CMRI: Option<&SUNAdaptController>,
) -> Option<SUNAdaptController> {
    /* Return with failure if input controller is NULL or has
       unsupported type */
    let CMRI = match CMRI {
        None => {
            return None;
        }
        Some(CMRI) => CMRI,
    };
    if SUNAdaptController_GetType(CMRI) != SUN_ADAPTCONTROLLER_MRI_H_TOL {
        return None;
    }

    /* Return with failure if stepper is inaccessible */
    if ark_mem.borrow().step_mem.is_none() {
        arkProcessError(
            Some(ark_mem),
            ARK_MEM_NULL,
            line!() as i32,
            "SUNAdaptController_MRIStep",
            file!(),
            MSG_MRISTEP_NO_MEM,
        );
        return None;
    }

    /* Create an empty controller object */
    let sunctx = CMRI.sunctx.borrow().clone();
    let C = SUNAdaptController_NewEmpty(&sunctx)?;

    /* Attach operations */
    {
        let mut ops = C.ops.borrow_mut();
        ops.gettype = Some(SUNAdaptController_GetType_MRIStep);
        ops.estimatestep = Some(SUNAdaptController_EstimateStep_MRIStep);
        ops.reset = Some(SUNAdaptController_Reset_MRIStep);
        ops.write = Some(SUNAdaptController_Write_MRIStep);
        ops.updateh = Some(SUNAdaptController_UpdateH_MRIStep);
        ops.space = Some(SUNAdaptController_Space_MRIStep);
    }

    /* Create content (infallible here) and attach ARKODE memory, MRI stepper
       memory and MRI controller objects */
    *C.content.borrow_mut() = Box::new(mriStepControlContent {
        ark_mem: ark_mem.clone(),
        C: CMRI.clone(),
    });

    /* Attach content and return */
    Some(C)
}

pub fn SUNAdaptController_EstimateStep_MRIStep(
    C: &SUNAdaptController,
    H: sunrealtype,
    P: i32,
    DSM: sunrealtype,
    Hnew: &mut sunrealtype,
) -> SUNErrCode {
    /* Shortcuts to ARKODE and MRIStep memory (`ark_mem == NULL` cannot
       happen; the `step_mem == NULL` half of the C check is kept) */
    let ark_mem = MRICONTROL_A(C);
    if ark_mem.borrow().step_mem.is_none() {
        return SUN_ERR_MEM_FAIL;
    }

    /* C passes `&(step_mem->inner_rtol_factor_new)` straight into the
       controller: copy the field out, call, then write it back. */
    let (inner_rtol_factor, inner_dsm, mut inner_rtol_factor_new) = {
        let step_mem = mriStep_mem_mut(&ark_mem);
        (
            step_mem.inner_rtol_factor,
            step_mem.inner_dsm,
            step_mem.inner_rtol_factor_new,
        )
    };

    /* Estimate slow stepsize from MRI controller */
    let CMRI = MRICONTROL_C(C);
    let retval = SUNAdaptController_EstimateStepTol(
        &CMRI,
        H,
        inner_rtol_factor,
        P,
        DSM,
        inner_dsm,
        Hnew,
        &mut inner_rtol_factor_new,
    );
    mriStep_mem_mut(&ark_mem).inner_rtol_factor_new = inner_rtol_factor_new;

    retval
}

pub fn SUNAdaptController_UpdateH_MRIStep(
    C: &SUNAdaptController,
    H: sunrealtype,
    DSM: sunrealtype,
) -> SUNErrCode {
    /* Shortcuts to ARKODE and MRIStep memory */
    let ark_mem = MRICONTROL_A(C);
    if ark_mem.borrow().step_mem.is_none() {
        return SUN_ERR_MEM_FAIL;
    }

    let (inner_rtol_factor, inner_dsm) = {
        let step_mem = mriStep_mem_mut(&ark_mem);
        (step_mem.inner_rtol_factor, step_mem.inner_dsm)
    };

    /* Update MRI controller */
    let CMRI = MRICONTROL_C(C);
    let retval = SUNAdaptController_UpdateMRIHTol(&CMRI, H, inner_rtol_factor, DSM, inner_dsm);
    if retval != SUN_SUCCESS {
        return retval;
    }

    /* Update inner controller parameter to most-recent prediction */
    {
        let mut step_mem = mriStep_mem_mut(&ark_mem);
        step_mem.inner_rtol_factor = step_mem.inner_rtol_factor_new;
    }

    /* return with success*/
    SUN_SUCCESS
}

pub fn SUNAdaptController_GetType_MRIStep(C: &SUNAdaptController) -> SUNAdaptController_Type {
    let CMRI = MRICONTROL_C(C);
    SUNAdaptController_GetType(&CMRI)
}

pub fn SUNAdaptController_Reset_MRIStep(C: &SUNAdaptController) -> SUNErrCode {
    let CMRI = MRICONTROL_C(C);
    SUNAdaptController_Reset(&CMRI)
}

pub fn SUNAdaptController_Write_MRIStep(C: &SUNAdaptController, fptr: &SUNFile) -> SUNErrCode {
    let CMRI = MRICONTROL_C(C);
    SUNAdaptController_Write(&CMRI, fptr)
}

pub fn SUNAdaptController_Space_MRIStep(
    C: &SUNAdaptController,
    lenrw: &mut i64,
    leniw: &mut i64,
) -> SUNErrCode {
    let CMRI = MRICONTROL_C(C);
    SUNAdaptController_Space(&CMRI, lenrw, leniw)
}

/*===============================================================
  EOF
  ===============================================================*/
