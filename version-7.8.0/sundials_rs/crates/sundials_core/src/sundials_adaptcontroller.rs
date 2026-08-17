//! Port of `src/sundials/sundials_adaptcontroller.c` +
//! `include/sundials/sundials_adaptcontroller.h` (generic SUNAdaptController).
//!
//! C `FILE*` maps to `crate::sundials_utils::SUNFile`. Handle arguments are
//! `&SUNAdaptController` (non-null by construction), so C `C == NULL` guard
//! branches vanish; `Destroy`/`DestroyEmpty` accept `Option` (NULL allowed).

use std::any::Any;
use std::cell::RefCell;
use std::rc::Rc;

use crate::sundials_context::SUNContext;
use crate::sundials_errors::{SUN_ERR_ARG_INCOMPATIBLE, SUN_SUCCESS};
use crate::sundials_math::SUNStrToReal;
use crate::sundials_types::*;
use crate::sundials_utils::SUNFile;

/* -----------------------------------------------------------------
 * SUNAdaptController types:
 *    NONE - empty controller (does nothing)
 *    H    - controls a single-rate step size
 *    MRI_H_TOL - controls slow step and fast relative tolerances
 * ----------------------------------------------------------------- */

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SUNAdaptController_Type {
    SUN_ADAPTCONTROLLER_NONE,
    SUN_ADAPTCONTROLLER_H,
    SUN_ADAPTCONTROLLER_MRI_H_TOL,
}
pub use SUNAdaptController_Type::*;

/* Structure containing function pointers to controller operations */
#[derive(Default, Clone)]
pub struct _generic_SUNAdaptController_Ops {
    /* REQUIRED of all controller implementations. */
    pub gettype: Option<fn(&SUNAdaptController) -> SUNAdaptController_Type>,

    /* REQUIRED for controllers of SUN_ADAPTCONTROLLER_H type. */
    pub estimatestep:
        Option<fn(&SUNAdaptController, sunrealtype, i32, sunrealtype, &mut sunrealtype) -> SUNErrCode>,

    /* REQUIRED for controllers of SUN_ADAPTCONTROLLER_MRI_H_TOL type. */
    pub estimatesteptol: Option<
        fn(
            &SUNAdaptController,
            sunrealtype,
            sunrealtype,
            i32,
            sunrealtype,
            sunrealtype,
            &mut sunrealtype,
            &mut sunrealtype,
        ) -> SUNErrCode,
    >,

    /* OPTIONAL for all SUNAdaptController implementations. */
    pub destroy: Option<fn(&SUNAdaptController) -> SUNErrCode>,
    pub reset: Option<fn(&SUNAdaptController) -> SUNErrCode>,
    pub setoptions:
        Option<fn(&SUNAdaptController, Option<&str>, Option<&str>, &[String]) -> SUNErrCode>,
    pub setdefaults: Option<fn(&SUNAdaptController) -> SUNErrCode>,
    pub write: Option<fn(&SUNAdaptController, &SUNFile) -> SUNErrCode>,
    pub seterrorbias: Option<fn(&SUNAdaptController, sunrealtype) -> SUNErrCode>,
    pub updateh: Option<fn(&SUNAdaptController, sunrealtype, sunrealtype) -> SUNErrCode>,
    pub updatemrihtol:
        Option<fn(&SUNAdaptController, sunrealtype, sunrealtype, sunrealtype, sunrealtype) -> SUNErrCode>,
    pub space: Option<fn(&SUNAdaptController, &mut i64, &mut i64) -> SUNErrCode>,
}

pub type SUNAdaptController_Ops = _generic_SUNAdaptController_Ops;

/* A SUNAdaptController is a structure with an implementation-dependent
   'content' field, and a pointer to a structure of
   operations corresponding to that implementation. */
pub struct _generic_SUNAdaptController {
    pub content: RefCell<Box<dyn Any>>,
    pub ops: RefCell<_generic_SUNAdaptController_Ops>,
    pub sunctx: RefCell<SUNContext>,
}

pub type SUNAdaptController = Rc<_generic_SUNAdaptController>;

/* -----------------------------------------------------------------
 * Create a new empty SUNAdaptController object
 * ----------------------------------------------------------------- */

pub fn SUNAdaptController_NewEmpty(sunctx: &SUNContext) -> Option<SUNAdaptController> {
    Some(Rc::new(_generic_SUNAdaptController {
        content: RefCell::new(Box::new(())),
        ops: RefCell::new(_generic_SUNAdaptController_Ops::default()),
        sunctx: RefCell::new(sunctx.clone()),
    }))
}

/* -----------------------------------------------------------------
 * Free a generic SUNAdaptController (assumes content is already empty)
 * ----------------------------------------------------------------- */

pub fn SUNAdaptController_DestroyEmpty(C: Option<SUNAdaptController>) {
    drop(C);
}

/* -----------------------------------------------------------------
 * Required functions in the 'ops' structure for non-NULL controller
 * ----------------------------------------------------------------- */

pub fn SUNAdaptController_GetType(C: &SUNAdaptController) -> SUNAdaptController_Type {
    let f = C.ops.borrow().gettype;
    match f {
        Some(f) => f(C),
        None => SUN_ADAPTCONTROLLER_NONE,
    }
}

/* -----------------------------------------------------------------
 * internal utility routines
 * ----------------------------------------------------------------- */

/// C `sunadctrlSetFromCommandLine`: processes `<Cid>.defaults` and
/// `<Cid>.error_bias <real>` tokens.
fn sunadctrlSetFromCommandLine(
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

    let mut idx = 1;
    while idx < argv.len() {
        /* skip command-line arguments that do not begin with correct prefix */
        if !argv[idx].starts_with(&prefix) {
            idx += 1;
            continue;
        }
        let key = &argv[idx][prefix.len()..];

        /* control over SetDefaults function */
        if key == "defaults" {
            let retval = SUNAdaptController_SetDefaults(C);
            if retval != SUN_SUCCESS {
                return retval;
            }
            idx += 1;
            continue;
        }

        /* control over SetErrorBias function */
        if key == "error_bias" {
            idx += 1;
            let rarg = SUNStrToReal(argv[idx].trim());
            let retval = SUNAdaptController_SetErrorBias(C, rarg);
            if retval != SUN_SUCCESS {
                return retval;
            }
            idx += 1;
            continue;
        }

        /* Note that although SUNAdaptController_Write is part of the base class,
        it should NOT be called until all options have been set, so we process
        that in the implementations instead of here. */
        idx += 1;
    }

    SUN_SUCCESS
}

/* -----------------------------------------------------------------
 * Optional functions in the 'ops' structure
 * ----------------------------------------------------------------- */

pub fn SUNAdaptController_Destroy(C: Option<SUNAdaptController>) -> SUNErrCode {
    match C {
        None => SUN_SUCCESS,
        Some(C) => {
            /* if the destroy operation exists use it */
            let f = C.ops.borrow().destroy;
            if let Some(f) = f {
                return f(&C);
            }
            /* if we reach this point destroy == None; dropping the handle
            releases content and ops */
            drop(C);
            SUN_SUCCESS
        }
    }
}

pub fn SUNAdaptController_EstimateStep(
    C: &SUNAdaptController,
    h: sunrealtype,
    p: i32,
    dsm: sunrealtype,
    hnew: &mut sunrealtype,
) -> SUNErrCode {
    let mut ier = SUN_SUCCESS;
    *hnew = h; /* initialize output with identity */
    let f = C.ops.borrow().estimatestep;
    if let Some(f) = f {
        ier = f(C, h, p, dsm, hnew);
    }
    ier
}

pub fn SUNAdaptController_EstimateStepTol(
    C: &SUNAdaptController,
    H: sunrealtype,
    tolfac: sunrealtype,
    P: i32,
    DSM: sunrealtype,
    dsm: sunrealtype,
    Hnew: &mut sunrealtype,
    tolfacnew: &mut sunrealtype,
) -> SUNErrCode {
    let mut ier = SUN_SUCCESS;
    *Hnew = H; /* initialize outputs with identity */
    *tolfacnew = tolfac;
    let f = C.ops.borrow().estimatesteptol;
    if let Some(f) = f {
        ier = f(C, H, tolfac, P, DSM, dsm, Hnew, tolfacnew);
    }
    ier
}

pub fn SUNAdaptController_Reset(C: &SUNAdaptController) -> SUNErrCode {
    let mut ier = SUN_SUCCESS;
    let f = C.ops.borrow().reset;
    if let Some(f) = f {
        ier = f(C);
    }
    ier
}

pub fn SUNAdaptController_SetOptions(
    C: &SUNAdaptController,
    Cid: Option<&str>,
    file_name: Option<&str>,
    argv: &[String],
) -> SUNErrCode {
    /* File-based option control is currently unimplemented */
    if !(file_name.is_none() || file_name == Some("")) {
        return SUN_ERR_ARG_INCOMPATIBLE;
    }

    /* First, process all base-class options */
    if !argv.is_empty() {
        let ier = sunadctrlSetFromCommandLine(C, Cid, argv);
        if ier != SUN_SUCCESS {
            return ier;
        }
    }

    /* Second, ask the implementation to process any remaining options */
    let f = C.ops.borrow().setoptions;
    match f {
        Some(f) => f(C, Cid, file_name, argv),
        None => SUN_SUCCESS,
    }
}

pub fn SUNAdaptController_SetDefaults(C: &SUNAdaptController) -> SUNErrCode {
    let mut ier = SUN_SUCCESS;
    let f = C.ops.borrow().setdefaults;
    if let Some(f) = f {
        ier = f(C);
    }
    ier
}

pub fn SUNAdaptController_Write(C: &SUNAdaptController, fptr: &SUNFile) -> SUNErrCode {
    let mut ier = SUN_SUCCESS;
    let f = C.ops.borrow().write;
    if let Some(f) = f {
        ier = f(C, fptr);
    }
    ier
}

pub fn SUNAdaptController_SetErrorBias(C: &SUNAdaptController, bias: sunrealtype) -> SUNErrCode {
    let mut ier = SUN_SUCCESS;
    let f = C.ops.borrow().seterrorbias;
    if let Some(f) = f {
        ier = f(C, bias);
    }
    ier
}

pub fn SUNAdaptController_UpdateH(
    C: &SUNAdaptController,
    h: sunrealtype,
    dsm: sunrealtype,
) -> SUNErrCode {
    let mut ier = SUN_SUCCESS;
    let f = C.ops.borrow().updateh;
    if let Some(f) = f {
        ier = f(C, h, dsm);
    }
    ier
}

pub fn SUNAdaptController_UpdateMRIHTol(
    C: &SUNAdaptController,
    H: sunrealtype,
    tolfac: sunrealtype,
    DSM: sunrealtype,
    dsm: sunrealtype,
) -> SUNErrCode {
    let mut ier = SUN_SUCCESS;
    let f = C.ops.borrow().updatemrihtol;
    if let Some(f) = f {
        ier = f(C, H, tolfac, DSM, dsm);
    }
    ier
}

pub fn SUNAdaptController_Space(
    C: &SUNAdaptController,
    lenrw: &mut i64,
    leniw: &mut i64,
) -> SUNErrCode {
    let mut ier = SUN_SUCCESS;
    *lenrw = 0; /* initialize outputs with identity */
    *leniw = 0;
    let f = C.ops.borrow().space;
    if let Some(f) = f {
        ier = f(C, lenrw, leniw);
    }
    ier
}
