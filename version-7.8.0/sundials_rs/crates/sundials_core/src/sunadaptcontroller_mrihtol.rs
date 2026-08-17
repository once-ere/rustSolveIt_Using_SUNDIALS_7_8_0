//! Port of `src/sunadaptcontroller/mrihtol/sunadaptcontroller_mrihtol.c` +
//! `include/sunadaptcontroller/sunadaptcontroller_mrihtol.h`.
//!
//! The slow (`HControl`) and fast (`TolControl`) sub-controllers are stored
//! as `SUNAdaptController` handle clones (C pointer copies) and dispatched
//! through the generic layer; sub-handle clones are taken out of `content`
//! before each generic call so no `RefCell` borrow is held across dispatch.
//! Release-mode `SUNAssert*`/`SUNCheckCall` type checks are omitted; the
//! `SUNCheckCall`-wrapped sub-controller calls are evaluated with `let _ =`.

use std::cell::RefMut;

use crate::sundials_adaptcontroller::*;
use crate::sundials_context::SUNContext;
use crate::sundials_errors::{SUN_ERR_ARG_INCOMPATIBLE, SUN_SUCCESS};
use crate::sundials_math::{SUNStrToReal, SUNMAX, SUNMIN};
use crate::sundials_types::*;
use crate::sundials_utils::{sun_format_g, SUNFile};

/* ------------------
 * Default parameters
 * ------------------ */

/*   maximum relative change for inner tolerance factor */
const INNER_MAX_RELCH: sunrealtype = 20.0;
/*   minimum tolerance factor for inner solver */
const INNER_MIN_TOLFAC: sunrealtype = 1.0e-5;
/*   maximum tolerance factor for inner solver */
const INNER_MAX_TOLFAC: sunrealtype = 1.0;

/* ----------------------------------------------------
 * MRI H+tolerance implementation of SUNAdaptController
 * ---------------------------------------------------- */

pub struct SUNAdaptControllerContent_MRIHTol_ {
    pub HControl: SUNAdaptController,
    pub TolControl: SUNAdaptController,
    pub inner_max_relch: sunrealtype,
    pub inner_min_tolfac: sunrealtype,
    pub inner_max_tolfac: sunrealtype,
}

pub type SUNAdaptControllerContent_MRIHTol = SUNAdaptControllerContent_MRIHTol_;

fn content_mut(C: &SUNAdaptController) -> RefMut<'_, SUNAdaptControllerContent_MRIHTol_> {
    RefMut::map(C.content.borrow_mut(), |c| {
        c.downcast_mut::<SUNAdaptControllerContent_MRIHTol_>()
            .expect("MRIHTol SUNAdaptController content")
    })
}

/// Clone the two sub-controller handles out of `content` (C pointer copies)
/// so generic-layer dispatch does not hold this controller's content borrow.
fn sub_controllers(C: &SUNAdaptController) -> (SUNAdaptController, SUNAdaptController) {
    let content = content_mut(C);
    (content.HControl.clone(), content.TolControl.clone())
}

/* -----------------------------------------------------------------
 * exported functions
 * ----------------------------------------------------------------- */

/* -----------------------------------------------------------------
 * Function to create a new MRIHTol controller
 */

pub fn SUNAdaptController_MRIHTol(
    HControl: &SUNAdaptController,
    TolControl: &SUNAdaptController,
    sunctx: &SUNContext,
) -> Option<SUNAdaptController> {
    /* Create an empty controller object */
    let C = SUNAdaptController_NewEmpty(sunctx)?;

    /* Attach operations */
    {
        let mut ops = C.ops.borrow_mut();
        ops.gettype = Some(SUNAdaptController_GetType_MRIHTol);
        ops.estimatesteptol = Some(SUNAdaptController_EstimateStepTol_MRIHTol);
        ops.reset = Some(SUNAdaptController_Reset_MRIHTol);
        ops.setoptions = Some(SUNAdaptController_SetOptions_MRIHTol);
        ops.setdefaults = Some(SUNAdaptController_SetDefaults_MRIHTol);
        ops.write = Some(SUNAdaptController_Write_MRIHTol);
        ops.seterrorbias = Some(SUNAdaptController_SetErrorBias_MRIHTol);
        ops.updatemrihtol = Some(SUNAdaptController_UpdateMRIHTol_MRIHTol);
        ops.space = Some(SUNAdaptController_Space_MRIHTol);
    }

    /* Create content, attach input controllers, set parameters to
    default values */
    *C.content.borrow_mut() = Box::new(SUNAdaptControllerContent_MRIHTol_ {
        HControl: HControl.clone(),
        TolControl: TolControl.clone(),
        inner_max_relch: INNER_MAX_RELCH,
        inner_min_tolfac: INNER_MIN_TOLFAC,
        inner_max_tolfac: INNER_MAX_TOLFAC,
    });

    Some(C)
}

/* ----------------------------------------------------------------------------
 * Function to control set routines via the command line or file
 */

pub fn SUNAdaptController_SetOptions_MRIHTol(
    C: &SUNAdaptController,
    Cid: Option<&str>,
    file_name: Option<&str>,
    argv: &[String],
) -> SUNErrCode {
    /* File-based option control is currently unimplemented */
    if !(file_name.is_none() || file_name == Some("")) {
        return SUN_ERR_ARG_INCOMPATIBLE;
    }

    if !argv.is_empty() {
        let ier = setFromCommandLine_MRIHTol(C, Cid, argv);
        if ier != SUN_SUCCESS {
            return ier;
        }
    }

    SUN_SUCCESS
}

/* -----------------------------------------------------------------
 * Function to control MRIHTol parameters from the command line
 */

fn setFromCommandLine_MRIHTol(
    C: &SUNAdaptController,
    Cid: Option<&str>,
    argv: &[String],
) -> SUNErrCode {
    /* Prefix for options to set */
    let default_id = "sunadaptcontroller";
    let id = match Cid {
        Some(s) if !s.is_empty() => s,
        _ => default_id,
    };
    let prefix = format!("{id}.");

    let mut write_parameters: sunbooleantype = SUNFALSE;
    let mut idx = 1;
    while idx < argv.len() {
        /* skip command-line arguments that do not begin with correct prefix */
        if !argv[idx].starts_with(&prefix) {
            idx += 1;
            continue;
        }
        let key = &argv[idx][prefix.len()..];

        /* control over SetParams function */
        if key == "params_mrihtol" {
            idx += 1;
            let rarg1 = SUNStrToReal(argv[idx].trim());
            idx += 1;
            let rarg2 = SUNStrToReal(argv[idx].trim());
            idx += 1;
            let rarg3 = SUNStrToReal(argv[idx].trim());
            let retval = SUNAdaptController_SetParams_MRIHTol(C, rarg1, rarg2, rarg3);
            if retval != SUN_SUCCESS {
                return retval;
            }
            idx += 1;
            continue;
        }

        /* check whether it was requested that all parameters be printed to screen */
        if key == "write_parameters" {
            write_parameters = SUNTRUE;
            idx += 1;
            continue;
        }

        idx += 1;
    }

    /* Call SUNAdaptController_Write (if requested) now that all
    command-line options have been set -- WARNING: this knows
    nothing about MPI, so it could be redundantly written by all
    processes if requested. */
    if write_parameters {
        let retval = SUNAdaptController_Write(C, &SUNFile::Stdout);
        if retval != SUN_SUCCESS {
            return retval;
        }
    }

    SUN_SUCCESS
}

/* -----------------------------------------------------------------
 * Function to set MRIHTol parameters
 */

pub fn SUNAdaptController_SetParams_MRIHTol(
    C: &SUNAdaptController,
    inner_max_relch: sunrealtype,
    inner_min_tolfac: sunrealtype,
    inner_max_tolfac: sunrealtype,
) -> SUNErrCode {
    let mut content = content_mut(C);
    if inner_max_relch < 1.0 {
        content.inner_max_relch = INNER_MAX_RELCH;
    } else {
        content.inner_max_relch = inner_max_relch;
    }
    if inner_min_tolfac <= 0.0 {
        content.inner_min_tolfac = INNER_MIN_TOLFAC;
    } else {
        content.inner_min_tolfac = inner_min_tolfac;
    }
    if inner_max_tolfac <= 0.0 {
        content.inner_max_tolfac = INNER_MAX_TOLFAC;
    } else {
        content.inner_max_tolfac = inner_max_tolfac;
    }
    SUN_SUCCESS
}

/* -----------------------------------------------------------------
 * Function to get slow and fast sub-controllers
 */

pub fn SUNAdaptController_GetSlowController_MRIHTol(
    C: &SUNAdaptController,
    Cslow: &mut Option<SUNAdaptController>,
) -> SUNErrCode {
    *Cslow = Some(content_mut(C).HControl.clone());
    SUN_SUCCESS
}

pub fn SUNAdaptController_GetFastController_MRIHTol(
    C: &SUNAdaptController,
    Cfast: &mut Option<SUNAdaptController>,
) -> SUNErrCode {
    *Cfast = Some(content_mut(C).TolControl.clone());
    SUN_SUCCESS
}

/* -----------------------------------------------------------------
 * implementation of controller operations
 * ----------------------------------------------------------------- */

pub fn SUNAdaptController_GetType_MRIHTol(_C: &SUNAdaptController) -> SUNAdaptController_Type {
    SUN_ADAPTCONTROLLER_MRI_H_TOL
}

pub fn SUNAdaptController_EstimateStepTol_MRIHTol(
    C: &SUNAdaptController,
    H: sunrealtype,
    tolfac: sunrealtype,
    P: i32,
    DSM: sunrealtype,
    dsm: sunrealtype,
    Hnew: &mut sunrealtype,
    tolfacnew: &mut sunrealtype,
) -> SUNErrCode {
    let (HControl, TolControl, inner_max_relch, inner_min_tolfac, inner_max_tolfac);
    {
        let content = content_mut(C);
        HControl = content.HControl.clone();
        TolControl = content.TolControl.clone();
        inner_max_relch = content.inner_max_relch;
        inner_min_tolfac = content.inner_min_tolfac;
        inner_max_tolfac = content.inner_max_tolfac;
    }
    let mut tolfacest: sunrealtype = 0.0;

    /* Call slow time scale sub-controller to fill Hnew -- note that all heuristics
    bounds on Hnew will be enforced by the time integrator itself */
    let _ = SUNAdaptController_EstimateStep(&HControl, H, P, DSM, Hnew);

    /* Call fast time scale sub-controller with order=1: no matter the integrator
    order, we expect its error to be proportional to the tolerance factor */
    let _ = SUNAdaptController_EstimateStep(&TolControl, tolfac, 0, dsm, &mut tolfacest);

    /* Enforce bounds on estimated tolerance factor */
    /*     keep relative change within bounds */
    tolfacest = SUNMAX(tolfacest, tolfac / inner_max_relch);
    tolfacest = SUNMIN(tolfacest, tolfac * inner_max_relch);
    /*     enforce absolute min/max bounds */
    tolfacest = SUNMAX(tolfacest, inner_min_tolfac);
    tolfacest = SUNMIN(tolfacest, inner_max_tolfac);

    /* Set result and return */
    *tolfacnew = tolfacest;
    SUN_SUCCESS
}

pub fn SUNAdaptController_Reset_MRIHTol(C: &SUNAdaptController) -> SUNErrCode {
    let (HControl, TolControl) = sub_controllers(C);
    let _ = SUNAdaptController_Reset(&HControl);
    let _ = SUNAdaptController_Reset(&TolControl);
    SUN_SUCCESS
}

pub fn SUNAdaptController_SetDefaults_MRIHTol(C: &SUNAdaptController) -> SUNErrCode {
    let (HControl, TolControl) = sub_controllers(C);
    let _ = SUNAdaptController_SetDefaults(&HControl);
    let _ = SUNAdaptController_SetDefaults(&TolControl);
    let mut content = content_mut(C);
    content.inner_max_relch = INNER_MAX_RELCH;
    content.inner_min_tolfac = INNER_MIN_TOLFAC;
    content.inner_max_tolfac = INNER_MAX_TOLFAC;
    SUN_SUCCESS
}

pub fn SUNAdaptController_Write_MRIHTol(C: &SUNAdaptController, fptr: &SUNFile) -> SUNErrCode {
    let (HControl, TolControl, inner_max_relch, inner_min_tolfac, inner_max_tolfac);
    {
        let content = content_mut(C);
        HControl = content.HControl.clone();
        TolControl = content.TolControl.clone();
        inner_max_relch = content.inner_max_relch;
        inner_min_tolfac = content.inner_min_tolfac;
        inner_max_tolfac = content.inner_max_tolfac;
    }
    fptr.write_str("Multirate H-Tol SUNAdaptController module:\n");
    fptr.write_str(&format!(
        "  inner_max_relch  = {}\n",
        sun_format_g(inner_max_relch)
    ));
    fptr.write_str(&format!(
        "  inner_min_tolfac = {}\n",
        sun_format_g(inner_min_tolfac)
    ));
    fptr.write_str(&format!(
        "  inner_max_tolfac = {}\n",
        sun_format_g(inner_max_tolfac)
    ));
    fptr.write_str("\nSlow step controller:\n");
    let _ = SUNAdaptController_Write(&HControl, fptr);
    fptr.write_str("\nFast tolerance controller:\n");
    let _ = SUNAdaptController_Write(&TolControl, fptr);
    SUN_SUCCESS
}

pub fn SUNAdaptController_SetErrorBias_MRIHTol(
    C: &SUNAdaptController,
    bias: sunrealtype,
) -> SUNErrCode {
    let (HControl, TolControl) = sub_controllers(C);
    let _ = SUNAdaptController_SetErrorBias(&HControl, bias);
    let _ = SUNAdaptController_SetErrorBias(&TolControl, bias);
    SUN_SUCCESS
}

pub fn SUNAdaptController_UpdateMRIHTol_MRIHTol(
    C: &SUNAdaptController,
    H: sunrealtype,
    tolfac: sunrealtype,
    DSM: sunrealtype,
    dsm: sunrealtype,
) -> SUNErrCode {
    let (HControl, TolControl) = sub_controllers(C);
    let _ = SUNAdaptController_UpdateH(&HControl, H, DSM);
    let _ = SUNAdaptController_UpdateH(&TolControl, tolfac, dsm);
    SUN_SUCCESS
}

pub fn SUNAdaptController_Space_MRIHTol(
    C: &SUNAdaptController,
    lenrw: &mut i64,
    leniw: &mut i64,
) -> SUNErrCode {
    let (HControl, TolControl) = sub_controllers(C);
    let (mut lrw, mut liw): (i64, i64) = (0, 0);
    let _ = SUNAdaptController_Space(&HControl, lenrw, leniw);
    let _ = SUNAdaptController_Space(&TolControl, &mut lrw, &mut liw);
    *lenrw += lrw;
    *leniw += liw;
    SUN_SUCCESS
}
