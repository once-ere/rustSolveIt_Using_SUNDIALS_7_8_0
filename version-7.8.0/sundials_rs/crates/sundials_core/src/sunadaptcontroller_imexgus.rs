//! Port of `src/sunadaptcontroller/imexgus/sunadaptcontroller_imexgus.c` +
//! `include/sunadaptcontroller/sunadaptcontroller_imexgus.h`.
//!
//! `hp` is zero-initialized at construction (C leaves it as malloc garbage;
//! it is never read before the first `UpdateH` sets it on any C-defined path).
//! `SUN_FORMAT_G` output maps to `sundials_utils::sun_format_g`.

use std::cell::RefMut;

use crate::sundials_adaptcontroller::*;
use crate::sundials_context::SUNContext;
use crate::sundials_errors::{SUN_ERR_ARG_INCOMPATIBLE, SUN_SUCCESS};
use crate::sundials_math::{SUNRpowerR, SUNStrToReal, SUNMIN};
use crate::sundials_types::*;
use crate::sundials_utils::{sun_format_g, SUNFile};

/* ------------------
 * Default parameters
 * ------------------ */

const DEFAULT_K1E: sunrealtype = 0.367;
const DEFAULT_K2E: sunrealtype = 0.268;
const DEFAULT_K1I: sunrealtype = 0.95;
const DEFAULT_K2I: sunrealtype = 0.95;
const DEFAULT_BIAS: sunrealtype = 1.0;

/* ----------------------------------------------------
 * ImEx Gustafsson implementation of SUNAdaptController
 * ---------------------------------------------------- */

pub struct SUNAdaptControllerContent_ImExGus_ {
    pub k1i: sunrealtype, /* internal controller parameters */
    pub k2i: sunrealtype,
    pub k1e: sunrealtype,
    pub k2e: sunrealtype,
    pub bias: sunrealtype,          /* error bias factor */
    pub ep: sunrealtype,            /* error from previous step */
    pub hp: sunrealtype,            /* previous step size */
    pub firststep: sunbooleantype,  /* flag indicating first step */
}

pub type SUNAdaptControllerContent_ImExGus = SUNAdaptControllerContent_ImExGus_;

fn content_mut(C: &SUNAdaptController) -> RefMut<'_, SUNAdaptControllerContent_ImExGus_> {
    RefMut::map(C.content.borrow_mut(), |c| {
        c.downcast_mut::<SUNAdaptControllerContent_ImExGus_>()
            .expect("ImExGus SUNAdaptController content")
    })
}

/* -----------------------------------------------------------------
 * exported functions
 * ----------------------------------------------------------------- */

/* -----------------------------------------------------------------
 * Function to create a new ImExGus controller
 */

pub fn SUNAdaptController_ImExGus(sunctx: &SUNContext) -> Option<SUNAdaptController> {
    /* Create an empty controller object */
    let C = SUNAdaptController_NewEmpty(sunctx)?;

    /* Attach operations */
    {
        let mut ops = C.ops.borrow_mut();
        ops.gettype = Some(SUNAdaptController_GetType_ImExGus);
        ops.estimatestep = Some(SUNAdaptController_EstimateStep_ImExGus);
        ops.reset = Some(SUNAdaptController_Reset_ImExGus);
        ops.setoptions = Some(SUNAdaptController_SetOptions_ImExGus);
        ops.setdefaults = Some(SUNAdaptController_SetDefaults_ImExGus);
        ops.write = Some(SUNAdaptController_Write_ImExGus);
        ops.seterrorbias = Some(SUNAdaptController_SetErrorBias_ImExGus);
        ops.updateh = Some(SUNAdaptController_UpdateH_ImExGus);
        ops.space = Some(SUNAdaptController_Space_ImExGus);
    }

    /* Create and attach content (fields overwritten by SetDefaults + Reset) */
    *C.content.borrow_mut() = Box::new(SUNAdaptControllerContent_ImExGus_ {
        k1i: 0.0,
        k2i: 0.0,
        k1e: 0.0,
        k2e: 0.0,
        bias: 0.0,
        ep: 0.0,
        hp: 0.0,
        firststep: SUNFALSE,
    });

    /* Fill content with default/reset values */
    let _ = SUNAdaptController_SetDefaults_ImExGus(&C);
    let _ = SUNAdaptController_Reset_ImExGus(&C);

    Some(C)
}

/* ----------------------------------------------------------------------------
 * Function to control set routines via the command line or file
 */

pub fn SUNAdaptController_SetOptions_ImExGus(
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
        let ier = setFromCommandLine_ImExGus(C, Cid, argv);
        if ier != SUN_SUCCESS {
            return ier;
        }
    }

    SUN_SUCCESS
}

/* -----------------------------------------------------------------
 * Function to control ImExGus parameters from the command line
 */

fn setFromCommandLine_ImExGus(
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
        if key == "params_imexgus" {
            idx += 1;
            let rarg1 = SUNStrToReal(argv[idx].trim());
            idx += 1;
            let rarg2 = SUNStrToReal(argv[idx].trim());
            idx += 1;
            let rarg3 = SUNStrToReal(argv[idx].trim());
            idx += 1;
            let rarg4 = SUNStrToReal(argv[idx].trim());
            let retval = SUNAdaptController_SetParams_ImExGus(C, rarg1, rarg2, rarg3, rarg4);
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
 * Function to set ImExGus parameters
 */

pub fn SUNAdaptController_SetParams_ImExGus(
    C: &SUNAdaptController,
    k1e: sunrealtype,
    k2e: sunrealtype,
    k1i: sunrealtype,
    k2i: sunrealtype,
) -> SUNErrCode {
    let mut content = content_mut(C);
    content.k1e = k1e;
    content.k2e = k2e;
    content.k1i = k1i;
    content.k2i = k2i;
    SUN_SUCCESS
}

/* -----------------------------------------------------------------
 * implementation of controller operations
 * ----------------------------------------------------------------- */

pub fn SUNAdaptController_GetType_ImExGus(_C: &SUNAdaptController) -> SUNAdaptController_Type {
    SUN_ADAPTCONTROLLER_H
}

pub fn SUNAdaptController_EstimateStep_ImExGus(
    C: &SUNAdaptController,
    h: sunrealtype,
    p: i32,
    dsm: sunrealtype,
    hnew: &mut sunrealtype,
) -> SUNErrCode {
    let content = content_mut(C);

    /* order parameter to use */
    let ord = p + 1;

    /* compute estimated time step size, modifying the first step formula */
    if content.firststep {
        /* set usable time-step adaptivity parameters -- first step */
        let k = -1.0 / ord as sunrealtype;
        let e = content.bias * dsm;
        *hnew = h * SUNRpowerR(e, k);
    } else {
        /* set usable time-step adaptivity parameters -- subsequent steps */
        let k1e = -content.k1e / ord as sunrealtype;
        let k2e = -content.k2e / ord as sunrealtype;
        let k1i = -content.k1i / ord as sunrealtype;
        let k2i = -content.k2i / ord as sunrealtype;
        let e1 = content.bias * dsm;
        let e2 = e1 / content.ep;
        let hrat = h / content.hp;
        *hnew = h * SUNMIN(
            hrat * SUNRpowerR(e1, k1i) * SUNRpowerR(e2, k2i),
            SUNRpowerR(e1, k1e) * SUNRpowerR(e2, k2e),
        );
    }

    /* return with success */
    SUN_SUCCESS
}

pub fn SUNAdaptController_Reset_ImExGus(C: &SUNAdaptController) -> SUNErrCode {
    let mut content = content_mut(C);
    content.ep = 1.0;
    content.firststep = SUNTRUE;
    SUN_SUCCESS
}

pub fn SUNAdaptController_SetDefaults_ImExGus(C: &SUNAdaptController) -> SUNErrCode {
    content_mut(C).bias = DEFAULT_BIAS;
    SUNAdaptController_SetParams_ImExGus(C, DEFAULT_K1E, DEFAULT_K2E, DEFAULT_K1I, DEFAULT_K2I)
}

pub fn SUNAdaptController_Write_ImExGus(C: &SUNAdaptController, fptr: &SUNFile) -> SUNErrCode {
    let content = content_mut(C);
    fptr.write_str("ImEx Gustafsson SUNAdaptController module:\n");
    fptr.write_str(&format!("  k1e = {}\n", sun_format_g(content.k1e)));
    fptr.write_str(&format!("  k2e = {}\n", sun_format_g(content.k2e)));
    fptr.write_str(&format!("  k1i = {}\n", sun_format_g(content.k1i)));
    fptr.write_str(&format!("  k2i = {}\n", sun_format_g(content.k2i)));
    fptr.write_str(&format!("  bias factor = {}\n", sun_format_g(content.bias)));
    fptr.write_str(&format!("  previous error = {}\n", sun_format_g(content.ep)));
    fptr.write_str(&format!("  previous step = {}\n", sun_format_g(content.hp)));
    SUN_SUCCESS
}

pub fn SUNAdaptController_SetErrorBias_ImExGus(
    C: &SUNAdaptController,
    bias: sunrealtype,
) -> SUNErrCode {
    let mut content = content_mut(C);
    /* set allowed value, otherwise set default */
    if bias <= 0.0 {
        content.bias = DEFAULT_BIAS;
    } else {
        content.bias = bias;
    }

    SUN_SUCCESS
}

pub fn SUNAdaptController_UpdateH_ImExGus(
    C: &SUNAdaptController,
    h: sunrealtype,
    dsm: sunrealtype,
) -> SUNErrCode {
    let mut content = content_mut(C);
    content.ep = content.bias * dsm;
    content.hp = h;
    content.firststep = SUNFALSE;
    SUN_SUCCESS
}

pub fn SUNAdaptController_Space_ImExGus(
    _C: &SUNAdaptController,
    lenrw: &mut i64,
    leniw: &mut i64,
) -> SUNErrCode {
    *lenrw = 7;
    *leniw = 1;
    SUN_SUCCESS
}
