//! Port of `src/sunadaptcontroller/soderlind/sunadaptcontroller_soderlind.c` +
//! `include/sunadaptcontroller/sunadaptcontroller_soderlind.h`.
//!
//! `SUN_FORMAT_G` output maps to `sundials_utils::sun_format_g`. The C
//! `SODERLIND_*` macro accessors become field reads through `content_mut`.
//! Release-mode `SUNCheck*`/`SUNAssert` sites are omitted per build config.

use std::cell::RefMut;

use crate::sundials_adaptcontroller::*;
use crate::sundials_context::SUNContext;
use crate::sundials_errors::{SUN_ERR_ARG_INCOMPATIBLE, SUN_SUCCESS};
use crate::sundials_math::{SUNRpowerR, SUNStrToReal};
use crate::sundials_types::*;
use crate::sundials_utils::{sun_format_g, SUNFile};

/* ------------------
 * Default parameters
 * ------------------ */

const DEFAULT_K1: sunrealtype = 1.25; /* H_{0}321 parameters */
const DEFAULT_K2: sunrealtype = 0.5;
const DEFAULT_K3: sunrealtype = -0.75;
const DEFAULT_K4: sunrealtype = 0.25;
const DEFAULT_K5: sunrealtype = 0.75;
const DEFAULT_PID_K1: sunrealtype = 0.58; /* PID parameters */
const DEFAULT_PID_K2: sunrealtype = -0.21;
const DEFAULT_PID_K3: sunrealtype = 0.1;
const DEFAULT_PI_K1: sunrealtype = 0.8; /* PI parameters */
const DEFAULT_PI_K2: sunrealtype = -0.31;
const DEFAULT_I_K1: sunrealtype = 1.0; /* I parameters */
const DEFAULT_EXPGUS_K1: sunrealtype = 0.367; /* Explicit Gustafsson parameters */
const DEFAULT_EXPGUS_K2: sunrealtype = 0.268;
const DEFAULT_IMPGUS_K1: sunrealtype = 0.98; /* Implicit Gustafsson parameters */
const DEFAULT_IMPGUS_K2: sunrealtype = 0.95;
const DEFAULT_BIAS: sunrealtype = 1.0;

/* ----------------------------------------------------
 * Soderlind implementation of SUNAdaptController
 * ---------------------------------------------------- */

pub struct SUNAdaptControllerContent_Soderlind_ {
    pub k1: sunrealtype, /* internal controller parameters */
    pub k2: sunrealtype,
    pub k3: sunrealtype,
    pub k4: sunrealtype,
    pub k5: sunrealtype,
    pub bias: sunrealtype, /* error bias factor */
    pub ep: sunrealtype,   /* error from previous step */
    pub epp: sunrealtype,  /* error from 2 steps ago */
    pub hp: sunrealtype,   /* previous step size */
    pub hpp: sunrealtype,  /* step size from 2 steps ago */
    pub firststeps: i32,   /* flag to handle first few steps */
    pub historysize: i32,  /* number of past step sizes or errors needed */
}

pub type SUNAdaptControllerContent_Soderlind = SUNAdaptControllerContent_Soderlind_;

fn content_mut(C: &SUNAdaptController) -> RefMut<'_, SUNAdaptControllerContent_Soderlind_> {
    RefMut::map(C.content.borrow_mut(), |c| {
        c.downcast_mut::<SUNAdaptControllerContent_Soderlind_>()
            .expect("Soderlind SUNAdaptController content")
    })
}

/* -----------------------------------------------------------------
 * exported functions
 * ----------------------------------------------------------------- */

/* -----------------------------------------------------------------
 * Function to create a new Soderlind controller (a.k.a., H_{0}321)
 */

pub fn SUNAdaptController_Soderlind(sunctx: &SUNContext) -> Option<SUNAdaptController> {
    /* Create an empty controller object */
    let C = SUNAdaptController_NewEmpty(sunctx)?;

    /* Attach operations */
    {
        let mut ops = C.ops.borrow_mut();
        ops.gettype = Some(SUNAdaptController_GetType_Soderlind);
        ops.estimatestep = Some(SUNAdaptController_EstimateStep_Soderlind);
        ops.reset = Some(SUNAdaptController_Reset_Soderlind);
        ops.setoptions = Some(SUNAdaptController_SetOptions_Soderlind);
        ops.setdefaults = Some(SUNAdaptController_SetDefaults_Soderlind);
        ops.write = Some(SUNAdaptController_Write_Soderlind);
        ops.seterrorbias = Some(SUNAdaptController_SetErrorBias_Soderlind);
        ops.updateh = Some(SUNAdaptController_UpdateH_Soderlind);
        ops.space = Some(SUNAdaptController_Space_Soderlind);
    }

    /* Create content (all fields are overwritten by SetDefaults + Reset below) */
    *C.content.borrow_mut() = Box::new(SUNAdaptControllerContent_Soderlind_ {
        k1: 0.0,
        k2: 0.0,
        k3: 0.0,
        k4: 0.0,
        k5: 0.0,
        bias: 0.0,
        ep: 0.0,
        epp: 0.0,
        hp: 0.0,
        hpp: 0.0,
        firststeps: 0,
        historysize: 0,
    });

    /* Fill content with default/reset values */
    let _ = SUNAdaptController_SetDefaults_Soderlind(&C);
    let _ = SUNAdaptController_Reset_Soderlind(&C);

    Some(C)
}

/* ----------------------------------------------------------------------------
 * Function to control set routines via the command line or file
 */

pub fn SUNAdaptController_SetOptions_Soderlind(
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
        let ier = setFromCommandLine_Soderlind(C, Cid, argv);
        if ier != SUN_SUCCESS {
            return ier;
        }
    }

    SUN_SUCCESS
}

/* -----------------------------------------------------------------
 * Function to control Soderlind parameters from the command line
 */

fn setFromCommandLine_Soderlind(
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

        /* control over SetParams_Soderlind function */
        if key == "params_soderlind" {
            idx += 1;
            let rarg1 = SUNStrToReal(argv[idx].trim());
            idx += 1;
            let rarg2 = SUNStrToReal(argv[idx].trim());
            idx += 1;
            let rarg3 = SUNStrToReal(argv[idx].trim());
            idx += 1;
            let rarg4 = SUNStrToReal(argv[idx].trim());
            idx += 1;
            let rarg5 = SUNStrToReal(argv[idx].trim());
            let retval =
                SUNAdaptController_SetParams_Soderlind(C, rarg1, rarg2, rarg3, rarg4, rarg5);
            if retval != SUN_SUCCESS {
                return retval;
            }
            idx += 1;
            continue;
        }

        /* control over SetParams_PID function */
        if key == "params_pid" {
            idx += 1;
            let rarg1 = SUNStrToReal(argv[idx].trim());
            idx += 1;
            let rarg2 = SUNStrToReal(argv[idx].trim());
            idx += 1;
            let rarg3 = SUNStrToReal(argv[idx].trim());
            let retval = SUNAdaptController_SetParams_PID(C, rarg1, rarg2, rarg3);
            if retval != SUN_SUCCESS {
                return retval;
            }
            idx += 1;
            continue;
        }

        /* control over SetParams_PI function */
        if key == "params_pi" {
            idx += 1;
            let rarg1 = SUNStrToReal(argv[idx].trim());
            idx += 1;
            let rarg2 = SUNStrToReal(argv[idx].trim());
            let retval = SUNAdaptController_SetParams_PI(C, rarg1, rarg2);
            if retval != SUN_SUCCESS {
                return retval;
            }
            idx += 1;
            continue;
        }

        /* control over SetParams_I function */
        if key == "params_i" {
            idx += 1;
            let rarg1 = SUNStrToReal(argv[idx].trim());
            let retval = SUNAdaptController_SetParams_I(C, rarg1);
            if retval != SUN_SUCCESS {
                return retval;
            }
            idx += 1;
            continue;
        }

        /* control over SetParams_ExpGus function */
        if key == "params_expgus" {
            idx += 1;
            let rarg1 = SUNStrToReal(argv[idx].trim());
            idx += 1;
            let rarg2 = SUNStrToReal(argv[idx].trim());
            let retval = SUNAdaptController_SetParams_ExpGus(C, rarg1, rarg2);
            if retval != SUN_SUCCESS {
                return retval;
            }
            idx += 1;
            continue;
        }

        /* control over SetParams_ImpGus function */
        if key == "params_impgus" {
            idx += 1;
            let rarg1 = SUNStrToReal(argv[idx].trim());
            idx += 1;
            let rarg2 = SUNStrToReal(argv[idx].trim());
            let retval = SUNAdaptController_SetParams_ImpGus(C, rarg1, rarg2);
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
 * Function to set Soderlind parameters
 */

pub fn SUNAdaptController_SetParams_Soderlind(
    C: &SUNAdaptController,
    k1: sunrealtype,
    k2: sunrealtype,
    k3: sunrealtype,
    k4: sunrealtype,
    k5: sunrealtype,
) -> SUNErrCode {
    let mut content = content_mut(C);
    content.k1 = k1;
    content.k2 = k2;
    content.k3 = k3;
    content.k4 = k4;
    content.k5 = k5;

    if k5 != 0.0 || k3 != 0.0 {
        content.historysize = 2;
    } else if k4 != 0.0 || k2 != 0.0 {
        content.historysize = 1;
    } else {
        content.historysize = 0;
    }
    SUN_SUCCESS
}

/* -----------------------------------------------------------------
 * Function to create a PID controller (subset of Soderlind)
 */

pub fn SUNAdaptController_PID(sunctx: &SUNContext) -> Option<SUNAdaptController> {
    let C = SUNAdaptController_Soderlind(sunctx)?;

    let _ = SUNAdaptController_SetParams_PID(&C, DEFAULT_PID_K1, DEFAULT_PID_K2, DEFAULT_PID_K3);

    Some(C)
}

/* -----------------------------------------------------------------
 * Function to set PID parameters
 */

pub fn SUNAdaptController_SetParams_PID(
    C: &SUNAdaptController,
    k1: sunrealtype,
    k2: sunrealtype,
    k3: sunrealtype,
) -> SUNErrCode {
    SUNAdaptController_SetParams_Soderlind(C, k1, k2, k3, 0.0, 0.0)
}

/* -----------------------------------------------------------------
 * Function to create a PI controller (subset of Soderlind)
 */

pub fn SUNAdaptController_PI(sunctx: &SUNContext) -> Option<SUNAdaptController> {
    let C = SUNAdaptController_Soderlind(sunctx)?;

    let _ = SUNAdaptController_SetParams_PI(&C, DEFAULT_PI_K1, DEFAULT_PI_K2);

    Some(C)
}

/* -----------------------------------------------------------------
 * Function to set PI parameters
 */

pub fn SUNAdaptController_SetParams_PI(
    C: &SUNAdaptController,
    k1: sunrealtype,
    k2: sunrealtype,
) -> SUNErrCode {
    SUNAdaptController_SetParams_Soderlind(C, k1, k2, 0.0, 0.0, 0.0)
}

/* -----------------------------------------------------------------
 * Function to create an I controller (subset of Soderlind)
 */

pub fn SUNAdaptController_I(sunctx: &SUNContext) -> Option<SUNAdaptController> {
    let C = SUNAdaptController_Soderlind(sunctx)?;

    let _ = SUNAdaptController_SetParams_I(&C, DEFAULT_I_K1);

    Some(C)
}

/* -----------------------------------------------------------------
 * Function to set PI parameters
 */

pub fn SUNAdaptController_SetParams_I(C: &SUNAdaptController, k1: sunrealtype) -> SUNErrCode {
    SUNAdaptController_SetParams_Soderlind(C, k1, 0.0, 0.0, 0.0, 0.0)
}

/* -----------------------------------------------------------------
 * Function to create an explicit Gustafsson controller
 */

pub fn SUNAdaptController_ExpGus(sunctx: &SUNContext) -> Option<SUNAdaptController> {
    let C = SUNAdaptController_Soderlind(sunctx)?;

    let _ = SUNAdaptController_SetParams_ExpGus(&C, DEFAULT_EXPGUS_K1, DEFAULT_EXPGUS_K2);

    Some(C)
}

/* -----------------------------------------------------------------
 * Function to set explicit Gustafsson parameters
 */

pub fn SUNAdaptController_SetParams_ExpGus(
    C: &SUNAdaptController,
    k1: sunrealtype,
    k2: sunrealtype,
) -> SUNErrCode {
    SUNAdaptController_SetParams_Soderlind(C, k1 + k2, -k2, 0.0, 0.0, 0.0)
}

/* -----------------------------------------------------------------
 * Function to create an implicit Gustafsson controller
 */

pub fn SUNAdaptController_ImpGus(sunctx: &SUNContext) -> Option<SUNAdaptController> {
    let C = SUNAdaptController_Soderlind(sunctx)?;

    let _ = SUNAdaptController_SetParams_ImpGus(&C, DEFAULT_IMPGUS_K1, DEFAULT_IMPGUS_K2);

    Some(C)
}

/* -----------------------------------------------------------------
 * Function to set explicit Gustafsson parameters
 */

pub fn SUNAdaptController_SetParams_ImpGus(
    C: &SUNAdaptController,
    k1: sunrealtype,
    k2: sunrealtype,
) -> SUNErrCode {
    SUNAdaptController_SetParams_Soderlind(C, k1 + k2, -k2, 0.0, 1.0, 0.0)
}

/* -----------------------------------------------------------------
 * Function to create an H_{0}211 controller (subset of Soderlind)
 */

pub fn SUNAdaptController_H0211(sunctx: &SUNContext) -> Option<SUNAdaptController> {
    let C = SUNAdaptController_Soderlind(sunctx)?;

    let _ = SUNAdaptController_SetParams_Soderlind(&C, 0.5, 0.5, 0.0, -0.5, 0.0);

    Some(C)
}

/* -----------------------------------------------------------------
 * Function to create an H_{0}321 controller (subset of Soderlind)
 */

pub fn SUNAdaptController_H0321(sunctx: &SUNContext) -> Option<SUNAdaptController> {
    let C = SUNAdaptController_Soderlind(sunctx)?;

    let _ = SUNAdaptController_SetParams_Soderlind(&C, 1.25, 0.5, -0.75, 0.25, 0.75);

    Some(C)
}

/* -----------------------------------------------------------------
 * Function to create an H211 controller (subset of Soderlind)
 */

pub fn SUNAdaptController_H211(sunctx: &SUNContext) -> Option<SUNAdaptController> {
    let C = SUNAdaptController_Soderlind(sunctx)?;

    let _ = SUNAdaptController_SetParams_Soderlind(&C, 0.25, 0.25, 0.0, -0.25, 0.0);

    Some(C)
}

/* -----------------------------------------------------------------
 * Function to create an H312 controller (subset of Soderlind)
 */

pub fn SUNAdaptController_H312(sunctx: &SUNContext) -> Option<SUNAdaptController> {
    let C = SUNAdaptController_Soderlind(sunctx)?;

    let _ = SUNAdaptController_SetParams_Soderlind(
        &C,
        1.0 / 8.0,
        0.25,
        1.0 / 8.0,
        -3.0 / 8.0,
        -1.0 / 8.0,
    );

    Some(C)
}

/* -----------------------------------------------------------------
 * implementation of controller operations
 * ----------------------------------------------------------------- */

pub fn SUNAdaptController_GetType_Soderlind(_C: &SUNAdaptController) -> SUNAdaptController_Type {
    SUN_ADAPTCONTROLLER_H
}

pub fn SUNAdaptController_EstimateStep_Soderlind(
    C: &SUNAdaptController,
    h: sunrealtype,
    p: i32,
    dsm: sunrealtype,
    hnew: &mut sunrealtype,
) -> SUNErrCode {
    let content = content_mut(C);

    /* order parameter to use */
    let ord = p + 1;
    let e1 = content.bias * dsm;

    /* Handle the case of insufficient history */
    if content.firststeps < content.historysize {
        /* Fall back onto an I controller */
        *hnew = h * SUNRpowerR(e1, -1.0 / ord as sunrealtype);
        return SUN_SUCCESS;
    }

    let k1 = -content.k1 / ord as sunrealtype;
    *hnew = h * SUNRpowerR(e1, k1);

    /* This branching is not ideal, but it's more efficient than computing extra
     * math operations with degenerate k values. */
    if content.historysize > 0 {
        let k2 = -content.k2 / ord as sunrealtype;
        let hrat1 = h / content.hp;
        *hnew *= SUNRpowerR(content.ep, k2) * SUNRpowerR(hrat1, content.k4);

        if content.historysize > 1 {
            let k3 = -content.k3 / ord as sunrealtype;
            let hrat2 = content.hp / content.hpp;
            *hnew *= SUNRpowerR(content.epp, k3) * SUNRpowerR(hrat2, content.k5);
        }
    }

    /* return with success */
    SUN_SUCCESS
}

pub fn SUNAdaptController_Reset_Soderlind(C: &SUNAdaptController) -> SUNErrCode {
    let mut content = content_mut(C);
    content.ep = 1.0;
    content.epp = 1.0;
    content.hp = 1.0;
    content.hpp = 1.0;
    content.firststeps = 0;
    SUN_SUCCESS
}

pub fn SUNAdaptController_SetDefaults_Soderlind(C: &SUNAdaptController) -> SUNErrCode {
    content_mut(C).bias = DEFAULT_BIAS;
    SUNAdaptController_SetParams_Soderlind(
        C, DEFAULT_K1, DEFAULT_K2, DEFAULT_K3, DEFAULT_K4, DEFAULT_K5,
    )
}

pub fn SUNAdaptController_Write_Soderlind(C: &SUNAdaptController, fptr: &SUNFile) -> SUNErrCode {
    let content = content_mut(C);
    fptr.write_str("Soderlind SUNAdaptController module:\n");
    fptr.write_str(&format!("  k1 = {}\n", sun_format_g(content.k1)));
    fptr.write_str(&format!("  k2 = {}\n", sun_format_g(content.k2)));
    fptr.write_str(&format!("  k3 = {}\n", sun_format_g(content.k3)));
    fptr.write_str(&format!("  k4 = {}\n", sun_format_g(content.k4)));
    fptr.write_str(&format!("  k5 = {}\n", sun_format_g(content.k5)));
    fptr.write_str(&format!("  bias factor = {}\n", sun_format_g(content.bias)));
    fptr.write_str(&format!("  previous error = {}\n", sun_format_g(content.ep)));
    fptr.write_str(&format!(
        "  previous-previous error = {}\n",
        sun_format_g(content.epp)
    ));
    fptr.write_str(&format!("  previous step = {}\n", sun_format_g(content.hp)));
    fptr.write_str(&format!(
        "  previous-previous step = {}\n",
        sun_format_g(content.hpp)
    ));
    fptr.write_str(&format!("  firststeps = {}\n", content.firststeps));
    fptr.write_str(&format!("  historysize = {}\n", content.historysize));
    SUN_SUCCESS
}

pub fn SUNAdaptController_SetErrorBias_Soderlind(
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

pub fn SUNAdaptController_UpdateH_Soderlind(
    C: &SUNAdaptController,
    h: sunrealtype,
    dsm: sunrealtype,
) -> SUNErrCode {
    let mut content = content_mut(C);
    content.epp = content.ep;
    content.ep = content.bias * dsm;
    content.hpp = content.hp;
    content.hp = h;
    if content.firststeps < content.historysize {
        content.firststeps += 1;
    }
    SUN_SUCCESS
}

pub fn SUNAdaptController_Space_Soderlind(
    _C: &SUNAdaptController,
    lenrw: &mut i64,
    leniw: &mut i64,
) -> SUNErrCode {
    *lenrw = 10;
    *leniw = 2;
    SUN_SUCCESS
}
